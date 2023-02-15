# 搜索路径的方法集合
import json
import torch
import numpy as np
import copy
import bisect
from geopy import distance
from queue import PriorityQueue


default_speed = 8.334  # 30 km/h 转化为 m/s


class SearchNode(object):
    """
    部分搜索算法，会依赖于优先队列，这时需要一个结构体来保存每一个搜索分支节点的当前状态。
    """

    def __init__(self, trace_loc, trace_tim, rid, date_time, log_prob):
        """
        维护一些变量
        应该不会炸内存
        Args:
            trace_loc: 该节点搜索分支的位置序列
            trace_tim: 该节点搜索分支的时间序列
            rid: 该节点的位置 rid
            date_time: 该节点的时间对象，为距离零点的秒数。用于计算下一个节点的时间
            log_prob: 该分支的条件概率取对数值 log(P(l_s -> l_i | l_s, l_d, t_s))
        """
        self.trace_loc = copy.deepcopy(trace_loc)
        self.trace_tim = copy.deepcopy(trace_tim)
        self.rid = rid
        self.date_time = date_time
        self.log_prob = log_prob

    def __ge__(self, other):
        if not isinstance(other, SearchNode):
            raise TypeError('>= require SearchNode')
        else:
            return self.log_prob >= other.log_prob

    def __le__(self, other):
        if not isinstance(other, SearchNode):
            raise TypeError('<= require SearchNone')
        else:
            return self.log_prob <= other.log_prob

    def __lt__(self, other):
        if not isinstance(other, SearchNode):
            raise TypeError('< require SearchNone')
        else:
            return self.log_prob < other.log_prob


class Searcher(object):
    """
    路径搜索器，这里会实现一批搜索算法
    """

    def __init__(self, device, adjacent_list, road_center_gps=None, road_length=None, road_time_distribution=None):
        """
        这里初始化一些搜索算法共有的数据

        Args:
            device: torch 设备
            adjacent_list: 路网邻接表
            road_center_gps: 路段的 GPS 信息
            road_length: 路段的长度信息
            road_time_distribution: 路段的平均通行时间
        """
        self.device = device
        self.adjacent_list = adjacent_list
        self.road_center_gps = road_center_gps
        self.road_length = road_length
        self.road_time_distribution = road_time_distribution.astype(int)

    def random_search(self, gen_model, trace_loc, trace_tim, des, top_k=3):
        """
        直接根据模型预测的候选集概率随机选择。同时会根据 top_k 筛选掉低概率的下一跳。
        Args:
            gen_model: 生成器
            trace_loc: 已知轨迹位置序列
            trace_tim: 已知轨迹时间序列
            des: 目的地
            top_k: 筛除低概率的下一跳

        Returns:
            trace_loc: 生成轨迹的 rid 序列
            trace_tim: 生成轨迹的 tim 序列
        """
        trace_loc = copy.deepcopy(trace_loc)
        trace_tim = copy.deepcopy(trace_tim)
        des_tensor = torch.LongTensor([des]).to(self.device)
        # 使用 now_minute 每步加上 1 分钟来预估时间
        off_time = 0
        now_rid = trace_loc[-1]
        start_time = trace_tim[-1]
        weekday_off = 0
        if start_time >= 1440:
            weekday_off = 1440
        step = len(trace_loc) - 1
        max_step = 50  # 因为选的都是 5KM 内的轨迹，所以不可能有这么长的 rid 序列
        while now_rid != des and step < max_step:
            # 这里是贪心的做法，非常害怕效果不好
            # 但我们是短距离，其实应该还好
            if str(now_rid) in self.adjacent_list:
                candidate_set = self.adjacent_list[str(now_rid)]
                # 构建模型输入
                trace_loc_tensor = torch.LongTensor(trace_loc).to(self.device).unsqueeze(0)
                trace_tim_tensor = torch.LongTensor(trace_tim).to(self.device).unsqueeze(0)
                candidate_set_tensor = torch.LongTensor(candidate_set).to(self.device).unsqueeze(0)
                candidate_prob = gen_model.predict(trace_loc=trace_loc_tensor, trace_time=trace_tim_tensor,
                                                   des=des_tensor, candidate_set=candidate_set_tensor)
                # 删除概率过小的干扰候选点
                # 删除原则：TopK? 固定阈值？先选用 TopK 吧
                select_topk = min(top_k, len(candidate_set))
                val, index = torch.topk(candidate_prob, select_topk, dim=1)
                # 对筛选后的概率变成和为 1
                val = val.squeeze(0).detach().numpy()  # (select_topk)
                val = val / np.sum(val)
                select_candidate_index = np.random.choice(index.squeeze(0), p=val)
                # 把生成的点放到轨迹序列中
                now_rid = candidate_set[select_candidate_index]
                trace_loc.append(now_rid)
                off_time += 1
                next_time = (start_time - weekday_off) + int(off_time / 5)
                # 规范到 0 ~ 1440 之间
                next_time = next_time % 1440
                next_time += weekday_off
                trace_tim.append(next_time)
                step += 1
            else:
                # 走到死路了
                break
        return trace_loc, trace_tim

    def beam_search(self, gen_model, trace_loc, trace_tim, des, width=15, max_step=50, prob_threshold=1e-4):
        """
        使用文本生成中常用的搜索算法 beam_search
        References: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        这个写的有点问题，可以参考知乎
        References: https://zhuanlan.zhihu.com/p/114669778
        Args:
            gen_model: 生成器
            trace_loc: 已知轨迹位置序列
            trace_tim: 已知轨迹时间序列
            des: 目的地
            width: beam search 的宽度，即每层记录 width 个较优搜索结果
            max_step: 最大搜索层数
            prob_threshold: 条件概率差阈值。如果最终找到的多条轨迹条件概率差值小于该阈值，则可以进行随机选择。

        Returns:
            trace_loc: 生成轨迹的 rid 序列
            trace_tim: 生成轨迹的 tim 序列
        """
        trace_loc = copy.deepcopy(trace_loc)
        trace_tim = copy.deepcopy(trace_tim)
        des_center_gps = self.road_center_gps[str(des)]
        des_tensor = torch.LongTensor([des]).to(self.device)
        # 第一层的 beam 只有包含根节点
        # 构建第一个搜索节点（根节点）
        start_time = trace_tim[-1]
        weekday_off = 0
        if start_time >= 1440:
            weekday_off = 1440
        start_date_time = start_time - weekday_off
        start_node = SearchNode(trace_loc=trace_loc, trace_tim=trace_tim, rid=trace_loc[-1],
                                date_time=start_date_time, log_prob=0)
        cur_beam = [start_node]  # 存放当前搜索层的分支节点数组
        end_beam = []  # 存放找到终点的搜索分支节点数组
        # 有没有可能生成回路？
        step = 0
        while step < max_step:
            next_beam = []
            for node in cur_beam:
                # 遍历当前层的分支节点的每一个子节点
                cur_rid = str(node.rid)
                if cur_rid in self.adjacent_list:
                    candidate_set = self.adjacent_list[cur_rid]
                    candidate_dis = []
                    for c in candidate_set:
                        candidate_gps = self.road_center_gps[str(c)]
                        d = distance.distance((des_center_gps[1], des_center_gps[0]),
                                              (candidate_gps[1], candidate_gps[0])).kilometers * 10
                        candidate_dis.append(d)
                    # 构建模型输入
                    trace_loc_tensor = torch.LongTensor(node.trace_loc).to(self.device).unsqueeze(0)
                    trace_tim_tensor = torch.LongTensor(node.trace_tim).to(self.device).unsqueeze(0)
                    candidate_set_tensor = torch.LongTensor(candidate_set).to(self.device).unsqueeze(0)
                    candidate_dis_tensor = torch.FloatTensor(candidate_dis).to(self.device).unsqueeze(0)
                    candidate_prob = gen_model.predict(trace_loc=trace_loc_tensor, trace_time=trace_tim_tensor,
                                                       des=des_tensor, candidate_set=candidate_set_tensor,
                                                       candidate_dis=candidate_dis_tensor)
                    candidate_log_prob = torch.log(candidate_prob).squeeze(0)  # (candidate_size)
                    pre_log_prob = node.log_prob
                    candidate_log_prob += pre_log_prob
                    # 更新时间
                    # 这里先默认每一步耗费 1 分钟
                    next_time = (node.date_time + 1) % 1440
                    next_time_code = next_time + weekday_off
                    new_trace_tim = node.trace_tim + [next_time_code]
                    for index, c in enumerate(candidate_set):
                        new_trace_loc = node.trace_loc + [c]
                        # 构建新的分支节点
                        new_node = SearchNode(trace_loc=new_trace_loc, trace_tim=new_trace_tim, rid=c,
                                              date_time=next_time, log_prob=candidate_log_prob[index].item())
                        if c == des:
                            # 找到终点了
                            end_beam.append(new_node)
                        else:
                            # 没找到终点继续找
                            next_beam.append(new_node)

            # 检查是否找到了 width 个分支
            if len(end_beam) == width:
                break

            # 没有的话，就选取 next_beam 中 width 个最优结果作为下一层的搜索节点
            next_beam = sorted(next_beam, key=lambda x: x.log_prob, reverse=True)
            cur_beam = next_beam[:width]
            step += 1
        # 如果没有找到一个到终点的轨迹，则返回当前搜索分支节点
        if len(end_beam) == 0:
            end_beam = cur_beam
        # 根据找到的 end_beam 返回生成的轨迹
        # 这里可以稍微有点多样性，就是如果我次一点的轨迹跟最优的条件概率差别不大的话，就可以按概率随机选。
        end_beam = sorted(end_beam, key=lambda x: x.log_prob, reverse=True)
        # 检验各结果与最优结果的条件概率差
        index = 1
        best_prob = np.exp(end_beam[0].log_prob)
        while index < len(end_beam):
            prob = np.exp(end_beam[index].log_prob)
            if best_prob - prob > prob_threshold:
                break
            index += 1
        # 在 [0, index) 之间随机返回一个下标作为结果
        return_index = np.random.randint(index)
        return end_beam[return_index].trace_loc, end_beam[return_index].trace_tim

    def astar_search(self, gen_model, trace_loc, trace_tim, des, default_len, max_step=500):
        """
        A* 搜索
        Args:
            gen_model: 生成模型
            trace_loc: 当前轨迹位置序列
            trace_tim: 当前轨迹时间序列
            des: 目的地 rid
            default_len: 真实轨迹的长度
            max_step: 最大搜索次数

        Returns:
            trace_loc: 生成轨迹的 rid 序列
            trace_tim: 生成轨迹的 tim 序列
        """
        close_set = set()
        open_set = PriorityQueue()  # unvisited node set
        rid2node = {}  # the dict key is rid, and value is corresponding search node
        trace_loc = copy.deepcopy(trace_loc)
        trace_tim = copy.deepcopy(trace_tim)
        des_center_gps = self.road_center_gps[str(des)]
        des_tensor = torch.LongTensor([des]).to(self.device)
        # put l_s into open_set
        # 构建第一个搜索节点（根节点）
        start_time = trace_tim[-1]
        weekday_off = 0
        if start_time >= 1440:
            weekday_off = 1440
        start_date_time = (start_time - weekday_off) * 60  # 转化为秒数
        start_node = SearchNode(trace_loc=trace_loc, trace_tim=trace_tim, rid=trace_loc[-1],
                                date_time=start_date_time, log_prob=0)
        open_set.put((start_node.log_prob, start_node))
        rid2node[trace_loc[-1]] = start_node
        max_search_step = max_step
        step = 0
        # find flag
        best_default_len = 15
        best_trace = None
        best_score = 0
        while not open_set.empty() and step < max_search_step:
            # get min f cost node to search
            cost, now_node = open_set.get()
            # put now node in close_set
            now_rid = now_node.rid
            if now_rid in close_set:
                # now_node is an old node
                # now_rid has been visited
                continue
            close_set.add(now_rid)
            # del now_id from rid2node, because now node has been visited
            rid2node.pop(now_rid, None)
            if now_rid == des:
                # find destination
                trace_loc = now_node.trace_loc
                trace_tim = now_node.trace_tim
                best_trace = (trace_loc, trace_tim)
                # finish search for query_i
                break
            else:
                # search now's adjacent rid
                if str(now_rid) in self.adjacent_list:
                    candidate_set = self.adjacent_list[str(now_rid)]
                    if len(candidate_set) == 1:
                        # 只有这一个选择，那就不用计算了
                        # 此时选择不增加 Log prob
                        candidate_log_prob = now_node.log_prob
                        # 更新时间
                        # 使用 road_time_distribution 来更新时间
                        now_time_hour = now_node.date_time // 3600  # 当前小时段
                        if self.road_time_distribution[now_time_hour][now_rid] != 1:
                            cost_time = self.road_time_distribution[now_time_hour][now_rid]
                        else:
                            # 使用默认速度计算
                            cost_time = int(round(self.road_length[str(now_rid)] / default_speed))  # 单位秒
                        next_time = (now_node.date_time + cost_time) % 86400
                        next_time_code = next_time // 60 + weekday_off
                        new_trace_tim = now_node.trace_tim + [next_time_code]
                        for index, c in enumerate(candidate_set):
                            c_score = np.exp(candidate_log_prob)
                            if c not in rid2node and c not in close_set:
                                # c 没有被搜索到过
                                new_trace_loc = now_node.trace_loc + [c]
                                # 构建新的分支节点
                                new_node = SearchNode(trace_loc=new_trace_loc, trace_tim=new_trace_tim, rid=c,
                                                      date_time=next_time, log_prob=candidate_log_prob)
                                # put c in open_set
                                open_set.put((-candidate_log_prob, new_node))
                                rid2node[c] = new_node
                                if (len(new_node.trace_loc) == default_len or
                                    (best_default_len < default_len == len(new_node.trace_loc))) \
                                        and c_score > best_score:
                                    # give the default recommended trace
                                    best_trace = (new_node.trace_loc, new_node.trace_tim)
                                    best_score = c_score
                            elif c in rid2node and rid2node[c].log_prob < candidate_log_prob:
                                # update search node
                                rid2node[c].log_prob = candidate_log_prob
                                new_trace_loc = now_node.trace_loc + [c]
                                rid2node[c].trace_loc = new_trace_loc
                                rid2node[c].trace_tim = copy.deepcopy(new_trace_tim)
                                rid2node[c].date_time = next_time
                                # there seems no way to update c in open_set, so just put c in open_set.
                                # the higher priority c will be searched first.
                                # so this is still work.
                                open_set.put((-candidate_log_prob, rid2node[c]))
                                if (len(rid2node[c].trace_loc) == default_len or
                                    (best_default_len < default_len == len(rid2node[c].trace_loc))) \
                                        and c_score > best_score:
                                    # give the default recommended trace
                                    best_trace = (rid2node[c].trace_loc, rid2node[c].trace_tim)
                                    best_score = c_score
                    else:
                        candidate_dis = []
                        for c in candidate_set:
                            candidate_gps = self.road_center_gps[str(c)]
                            d = distance.distance((des_center_gps[1], des_center_gps[0]),
                                                  (candidate_gps[1], candidate_gps[0])).kilometers * 10
                            candidate_dis.append(d)
                        # 构建模型输入
                        trace_loc_tensor = torch.LongTensor(now_node.trace_loc).to(self.device).unsqueeze(0)
                        trace_tim_tensor = torch.LongTensor(now_node.trace_tim).to(self.device).unsqueeze(0)
                        candidate_set_tensor = torch.LongTensor(candidate_set).to(self.device).unsqueeze(0)
                        candidate_dis_tensor = torch.FloatTensor(candidate_dis).to(self.device).unsqueeze(0)
                        # 根据模型计算 f 函数值
                        with torch.no_grad():
                            candidate_f_cost = gen_model.predict(trace_loc=trace_loc_tensor, trace_time=trace_tim_tensor,
                                                                 des=des_tensor, candidate_set=candidate_set_tensor,
                                                                 candidate_dis=candidate_dis_tensor)
                        # update each candidate's node
                        # calculate next datetime
                        candidate_log_prob = torch.log(candidate_f_cost).squeeze(0)  # (candidate_size)
                        pre_log_prob = now_node.log_prob
                        # 累计条件概率和
                        candidate_log_prob += pre_log_prob
                        # 更新时间
                        # 使用 road_time_distribution 来更新时间
                        now_time_hour = now_node.date_time // 3600  # 当前小时段
                        if self.road_time_distribution[now_time_hour][now_rid] != 1:
                            cost_time = self.road_time_distribution[now_time_hour][now_rid]
                        else:
                            # 使用默认速度计算
                            cost_time = int(round(self.road_length[str(now_rid)] / default_speed))  # 单位秒
                        next_time = (now_node.date_time + cost_time) % 86400
                        next_time_code = next_time // 60 + weekday_off
                        new_trace_tim = now_node.trace_tim + [next_time_code]
                        for index, c in enumerate(candidate_set):
                            c_log_prob = candidate_log_prob[index].item()
                            c_score = np.exp(c_log_prob)
                            if c not in rid2node and c not in close_set:
                                # c 没有被搜索到过
                                new_trace_loc = now_node.trace_loc + [c]
                                # 构建新的分支节点
                                new_node = SearchNode(trace_loc=new_trace_loc, trace_tim=new_trace_tim, rid=c,
                                                      date_time=next_time, log_prob=c_log_prob)
                                # put c in open_set
                                open_set.put((-c_log_prob, new_node))
                                rid2node[c] = new_node
                                if (len(new_node.trace_loc) == default_len or
                                    (best_default_len < default_len == len(new_node.trace_loc)))\
                                        and c_score > best_score:
                                    # give the default recommended trace
                                    best_trace = (new_node.trace_loc, new_node.trace_tim)
                                    best_score = c_score
                            elif c in rid2node and rid2node[c].log_prob < c_log_prob:
                                # update search node
                                rid2node[c].log_prob = c_log_prob
                                new_trace_loc = now_node.trace_loc + [c]
                                rid2node[c].trace_loc = new_trace_loc
                                rid2node[c].trace_tim = copy.deepcopy(new_trace_tim)
                                rid2node[c].date_time = next_time
                                # there seems no way to update c in open_set, so just put c in open_set.
                                # the higher priority c will be searched first.
                                # so this is still work.
                                open_set.put((-c_log_prob, rid2node[c]))
                                if (len(rid2node[c].trace_loc) == default_len or
                                    (best_default_len < default_len == len(rid2node[c].trace_loc))) \
                                        and c_score > best_score:
                                    # give the default recommended trace
                                    best_trace = (rid2node[c].trace_loc, rid2node[c].trace_tim)
                                    best_score = c_score
                step += 1
        if best_trace is not None:
            return best_trace[0], best_trace[1]
        else:
            return trace_loc, trace_tim

    def astar_search_in_region(self, gen_model, trace_loc, trace_tim, des, default_len, region_rid_set, max_step=500):
        """
        仅在当前区域内进行搜索 A* 搜索
        Args:
            gen_model: 生成模型
            trace_loc: 当前轨迹位置序列
            trace_tim: 当前轨迹时间序列
            des: 目的地 rid
            default_len: 真实轨迹的长度
            region_rid_set: 区域的路段集合
            max_step: 最大搜索次数

        Returns:
            trace_loc: 生成轨迹的 rid 序列
            trace_tim: 生成轨迹的 tim 序列
        """
        close_set = set()
        open_set = PriorityQueue()  # unvisited node set
        rid2node = {}  # the dict key is rid, and value is corresponding search node
        trace_loc = copy.deepcopy(trace_loc)
        trace_tim = copy.deepcopy(trace_tim)
        des_center_gps = self.road_center_gps[str(des)]
        des_tensor = torch.LongTensor([des]).to(self.device)
        # put l_s into open_set
        # 构建第一个搜索节点（根节点）
        start_time = trace_tim[-1]
        weekday_off = 0
        if start_time >= 1440:
            weekday_off = 1440
        start_date_time = (start_time - weekday_off) * 60  # 转化为秒数
        start_node = SearchNode(trace_loc=trace_loc, trace_tim=trace_tim, rid=trace_loc[-1],
                                date_time=start_date_time, log_prob=0)
        open_set.put((start_node.log_prob, start_node))
        rid2node[trace_loc[-1]] = start_node
        max_search_step = max_step
        step = 0
        # find flag
        best_default_len = 15
        best_trace = None
        best_score = 0
        while not open_set.empty() and step < max_search_step:
            # get min f cost node to search
            cost, now_node = open_set.get()
            # put now node in close_set
            now_rid = now_node.rid
            if now_rid in close_set:
                # now_node is an old node
                # now_rid has been visited
                continue
            close_set.add(now_rid)
            # del now_id from rid2node, because now node has been visited
            rid2node.pop(now_rid, None)
            if now_rid == des:
                # find destination
                trace_loc = now_node.trace_loc
                trace_tim = now_node.trace_tim
                best_trace = (trace_loc, trace_tim)
                # finish search for query_i
                break
            else:
                # search now's adjacent rid
                if now_rid in region_rid_set and str(now_rid) in self.adjacent_list:
                    # 只在当前区域内进行搜索，区域外的点要没是终点，要么就是干扰项
                    candidate_set = self.adjacent_list[str(now_rid)]
                    if len(candidate_set) == 1:
                        # 只有这一个选择，那就不用计算了
                        # 此时选择不增加 Log prob
                        candidate_log_prob = now_node.log_prob
                        # 更新时间
                        # 使用 road_time_distribution 来更新时间
                        now_time_hour = now_node.date_time // 3600  # 当前小时段
                        if self.road_time_distribution[now_time_hour][now_rid] != 1:
                            cost_time = self.road_time_distribution[now_time_hour][now_rid]
                        else:
                            # 使用默认速度计算
                            cost_time = int(round(self.road_length[str(now_rid)] / default_speed))  # 单位秒
                        next_time = (now_node.date_time + cost_time) % 86400
                        next_time_code = next_time // 60 + weekday_off
                        new_trace_tim = now_node.trace_tim + [next_time_code]
                        for index, c in enumerate(candidate_set):
                            c_score = np.exp(candidate_log_prob)
                            if c not in rid2node and c not in close_set:
                                # c 没有被搜索到过
                                new_trace_loc = now_node.trace_loc + [c]
                                # 构建新的分支节点
                                new_node = SearchNode(trace_loc=new_trace_loc, trace_tim=new_trace_tim, rid=c,
                                                      date_time=next_time, log_prob=candidate_log_prob)
                                # put c in open_set
                                open_set.put((-candidate_log_prob, new_node))
                                rid2node[c] = new_node
                                if (len(new_node.trace_loc) == default_len or
                                    (best_default_len < default_len == len(new_node.trace_loc))) \
                                        and c_score > best_score:
                                    # give the default recommended trace
                                    best_trace = (new_node.trace_loc, new_node.trace_tim)
                                    best_score = c_score
                            elif c in rid2node and rid2node[c].log_prob < candidate_log_prob:
                                # update search node
                                rid2node[c].log_prob = candidate_log_prob
                                new_trace_loc = now_node.trace_loc + [c]
                                rid2node[c].trace_loc = new_trace_loc
                                rid2node[c].trace_tim = copy.deepcopy(new_trace_tim)
                                rid2node[c].date_time = next_time
                                # there seems no way to update c in open_set, so just put c in open_set.
                                # the higher priority c will be searched first.
                                # so this is still work.
                                open_set.put((-candidate_log_prob, rid2node[c]))
                                if (len(rid2node[c].trace_loc) == default_len or
                                    (best_default_len < default_len == len(rid2node[c].trace_loc))) \
                                        and c_score > best_score:
                                    # give the default recommended trace
                                    best_trace = (rid2node[c].trace_loc, rid2node[c].trace_tim)
                                    best_score = c_score
                    else:
                        candidate_dis = []
                        for c in candidate_set:
                            candidate_gps = self.road_center_gps[str(c)]
                            d = distance.distance((des_center_gps[1], des_center_gps[0]),
                                                  (candidate_gps[1], candidate_gps[0])).kilometers * 10
                            candidate_dis.append(d)
                        # 构建模型输入
                        trace_loc_tensor = torch.LongTensor(now_node.trace_loc).to(self.device).unsqueeze(0)
                        trace_tim_tensor = torch.LongTensor(now_node.trace_tim).to(self.device).unsqueeze(0)
                        candidate_set_tensor = torch.LongTensor(candidate_set).to(self.device).unsqueeze(0)
                        candidate_dis_tensor = torch.FloatTensor(candidate_dis).to(self.device).unsqueeze(0)
                        # 根据模型计算 f 函数值
                        with torch.no_grad():
                            candidate_f_cost = gen_model.predict(trace_loc=trace_loc_tensor, trace_time=trace_tim_tensor,
                                                                 des=des_tensor, candidate_set=candidate_set_tensor,
                                                                 candidate_dis=candidate_dis_tensor)
                        # update each candidate's node
                        # calculate next datetime
                        candidate_log_prob = torch.log(candidate_f_cost).squeeze(0)  # (candidate_size)
                        pre_log_prob = now_node.log_prob
                        # 累计条件概率和
                        candidate_log_prob += pre_log_prob
                        # 更新时间
                        # 使用 road_time_distribution 来更新时间
                        now_time_hour = now_node.date_time // 3600  # 当前小时段
                        if self.road_time_distribution[now_time_hour][now_rid] != 1:
                            cost_time = self.road_time_distribution[now_time_hour][now_rid]
                        else:
                            # 使用默认速度计算
                            cost_time = int(round(self.road_length[str(now_rid)] / default_speed))  # 单位秒
                        next_time = (now_node.date_time + cost_time) % 86400
                        next_time_code = next_time // 60 + weekday_off
                        new_trace_tim = now_node.trace_tim + [next_time_code]
                        for index, c in enumerate(candidate_set):
                            c_log_prob = candidate_log_prob[index].item()
                            c_score = np.exp(c_log_prob)
                            if c not in rid2node and c not in close_set:
                                # c 没有被搜索到过
                                new_trace_loc = now_node.trace_loc + [c]
                                # 构建新的分支节点
                                new_node = SearchNode(trace_loc=new_trace_loc, trace_tim=new_trace_tim, rid=c,
                                                      date_time=next_time, log_prob=c_log_prob)
                                # put c in open_set
                                open_set.put((-c_log_prob, new_node))
                                rid2node[c] = new_node
                                if (len(new_node.trace_loc) == default_len or
                                    (best_default_len < default_len == len(new_node.trace_loc)))\
                                        and c_score > best_score:
                                    # give the default recommended trace
                                    best_trace = (new_node.trace_loc, new_node.trace_tim)
                                    best_score = c_score
                            elif c in rid2node and rid2node[c].log_prob < c_log_prob:
                                # update search node
                                rid2node[c].log_prob = c_log_prob
                                new_trace_loc = now_node.trace_loc + [c]
                                rid2node[c].trace_loc = new_trace_loc
                                rid2node[c].trace_tim = copy.deepcopy(new_trace_tim)
                                rid2node[c].date_time = next_time
                                # there seems no way to update c in open_set, so just put c in open_set.
                                # the higher priority c will be searched first.
                                # so this is still work.
                                open_set.put((-c_log_prob, rid2node[c]))
                                if (len(rid2node[c].trace_loc) == default_len or
                                    (best_default_len < default_len == len(rid2node[c].trace_loc))) \
                                        and c_score > best_score:
                                    # give the default recommended trace
                                    best_trace = (rid2node[c].trace_loc, rid2node[c].trace_tim)
                                    best_score = c_score
                step += 1
        if best_trace is not None:
            return best_trace[0], best_trace[1]
        else:
            return trace_loc, trace_tim

    def road_random_sample(self, gen_model, trace_loc, trace_tim, des, default_len):
        """
        在对抗学习中，使用 astar 感觉效果不好，不如尝试像 SeqGAN 那种的根据概率的生成方式
        Args:
            gen_model: 区域生成模型
            trace_loc: 开始路段
            trace_tim: 开始时间
            des: 终点路段
            default_len: 真实轨迹长度

        Returns:
            trace_loc, trace_tim
        """
        trace_loc = copy.deepcopy(trace_loc)
        trace_tim = copy.deepcopy(trace_tim)
        des_tensor = torch.LongTensor([des]).to(self.device)
        # 使用 now_minute 每步加上 1 分钟来预估时间
        weekday_off = 0
        start_time = trace_tim[-1]
        if start_time >= 1440:
            weekday_off = 1440
        step = len(trace_loc) - 1
        max_step = max(20, default_len)
        now_road = trace_loc[-1]
        now_date_time = (start_time - weekday_off) * 60  # 转化为秒数
        des_center_gps = self.road_center_gps[str(des)]
        while now_road != des and step < max_step:
            # 这里是贪心的做法，非常害怕效果不好
            # 但我们是短距离，其实应该还好
            if str(now_road) in self.adjacent_list:
                # search now's adjacent rid
                candidate_set = self.adjacent_list[str(now_road)]
                if len(candidate_set) == 0:
                    # 没有下一跳了。死路了，就 Break 就好了
                    break
                assert len(candidate_set) != 0
                if len(candidate_set) == 1:
                    # 只有这一个选择，那就不用计算了
                    # 更新时间
                    # 按照区域平均通勤距离/默认速度计算
                    next_road = candidate_set[0]
                    # 使用平均通勤时间
                    now_hour = now_date_time // 3600
                    if self.road_time_distribution[now_hour][now_road] != 1:
                        cost_time = self.road_time_distribution[now_hour][now_road]
                    else:
                        cost_time = int(round(self.road_length[str(now_road)] / default_speed))  # 单位秒
                    next_time = (now_date_time + cost_time) % 86400
                    next_time_code = next_time // 60 + weekday_off
                    trace_tim.append(next_time_code)
                    trace_loc.append(next_road)
                    now_road = next_road
                else:
                    candidate_dis = []
                    for c in candidate_set:
                        candidate_gps = self.road_center_gps[str(c)]
                        d = distance.distance((des_center_gps[1], des_center_gps[0]),
                                              (candidate_gps[1], candidate_gps[0])).kilometers * 10
                        candidate_dis.append(d)
                    # 构建模型输入
                    trace_loc_tensor = torch.LongTensor(trace_loc).to(self.device).unsqueeze(0)
                    trace_tim_tensor = torch.LongTensor(trace_tim).to(self.device).unsqueeze(0)
                    candidate_set_tensor = torch.LongTensor(candidate_set).to(self.device).unsqueeze(0)
                    candidate_dis_tensor = torch.FloatTensor(candidate_dis).to(self.device).unsqueeze(0)
                    # 根据模型计算 f 函数值
                    with torch.no_grad():
                        candidate_f_prob = gen_model.predict(trace_loc=trace_loc_tensor,
                                                             trace_time=trace_tim_tensor,
                                                             des=des_tensor, candidate_set=candidate_set_tensor,
                                                             candidate_dis=candidate_dis_tensor)
                    # 根据概率随机选择一个点来作为下一跳
                    try:
                        next_road_index = candidate_f_prob.squeeze(0).multinomial(1).item()
                    except RuntimeError:
                        print(candidate_set_tensor)
                        print(trace_loc_tensor)
                        print(trace_tim_tensor)
                        print(candidate_dis_tensor)
                        print(candidate_f_prob)
                        print(des_tensor)
                        exit()
                    next_road = candidate_set[next_road_index]
                    # 更新时间
                    # 使用平均通勤时间
                    now_hour = now_date_time // 3600
                    if self.road_time_distribution[now_hour][now_road] != 1:
                        cost_time = self.road_time_distribution[now_hour][now_road]
                    else:
                        cost_time = int(round(self.road_length[str(now_road)] / default_speed))  # 单位秒
                    next_time = (now_date_time + cost_time) % 86400
                    next_time_code = next_time // 60 + weekday_off
                    trace_tim.append(next_time_code)
                    trace_loc.append(next_road)
                    now_road = next_road
            else:
                # 走到死路了
                break
            step += 1
        if trace_loc[-1] == des:
            return trace_loc, trace_tim
        else:
            # 没找到，返回 default_len 个点
            return trace_loc[:default_len], trace_tim[:default_len]

    def naive_astar_search(self, start_rid, start_time, des, max_step=5000):
        close_set = set()
        open_set = PriorityQueue()  # unvisited node set
        rid2node = {}  # the dict key is rid, and value is corresponding search node
        trace_loc = [start_rid]
        trace_tim = [start_time]
        des_center_gps = self.road_center_gps[str(des)]
        # put l_s into open_set
        # 构建第一个搜索节点（根节点）
        start_time = trace_tim[-1]
        weekday_off = 0
        if start_time >= 1440:
            weekday_off = 1440
        start_date_time = (start_time - weekday_off) * 60  # 转化为秒数
        start_node = SearchNode(trace_loc=trace_loc, trace_tim=trace_tim, rid=trace_loc[-1],
                                date_time=start_date_time, log_prob=0)
        open_set.put((start_node.log_prob, start_node))
        rid2node[trace_loc[-1]] = start_node
        max_search_step = max_step
        step = 0
        # find flag
        best_default_len = 15
        best_trace = None
        best_cost = 0
        while not open_set.empty() and step < max_search_step:
            # get min f cost node to search
            cost, now_node = open_set.get()
            # put now node in close_set
            now_rid = now_node.rid
            if now_rid in close_set:
                # now_node is an old node
                # now_rid has been visited
                continue
            close_set.add(now_rid)
            # del now_id from rid2node, because now node has been visited
            rid2node.pop(now_rid, None)
            if now_rid == des:
                # find destination
                trace_loc = now_node.trace_loc
                trace_tim = now_node.trace_tim
                best_trace = (trace_loc, trace_tim)
                # finish search for query_i
                break
            else:
                # search now's adjacent rid
                if str(now_rid) in self.adjacent_list:
                    candidate_set = self.adjacent_list[str(now_rid)]
                    candidate_h_cost = []
                    for c in candidate_set:
                        candidate_gps = self.road_center_gps[str(c)]
                        d = distance.distance((des_center_gps[1], des_center_gps[0]),
                                              (candidate_gps[1], candidate_gps[0])).kilometers * 10
                        candidate_h_cost.append(d)
                    pre_log_prob = now_node.log_prob
                    # 使用 road_time_distribution 来更新时间
                    now_time_hour = now_node.date_time // 3600  # 当前小时段
                    if self.road_time_distribution[now_time_hour][now_rid] != 1:
                        cost_time = self.road_time_distribution[now_time_hour][now_rid]
                    else:
                        # 使用默认速度计算
                        cost_time = int(round(self.road_length[str(now_rid)] / default_speed))  # 单位秒
                    next_time = (now_node.date_time + cost_time) % 86400
                    next_time_code = next_time // 60 + weekday_off
                    new_trace_tim = now_node.trace_tim + [next_time_code]
                    for index, c in enumerate(candidate_set):
                        c_cost = pre_log_prob + candidate_h_cost[index] + self.road_length[str(c)]
                        if c not in rid2node and c not in close_set:
                            # c 没有被搜索到过
                            new_trace_loc = now_node.trace_loc + [c]
                            # 构建新的分支节点
                            new_node = SearchNode(trace_loc=new_trace_loc, trace_tim=new_trace_tim, rid=c,
                                                  date_time=next_time, log_prob=c_cost)
                            # put c in open_set
                            open_set.put((c_cost, new_node))
                            rid2node[c] = new_node
                            if (len(new_node.trace_loc) >= best_default_len) \
                                    and c_cost < best_cost:
                                # give the default recommended trace
                                best_trace = (new_node.trace_loc, new_node.trace_tim)
                                best_cost = c_cost
                            elif c in rid2node and rid2node[c].log_prob > c_cost:
                                # update search node
                                rid2node[c].log_prob = c_cost
                                new_trace_loc = now_node.trace_loc + [c]
                                rid2node[c].trace_loc = new_trace_loc
                                rid2node[c].trace_tim = copy.deepcopy(new_trace_tim)
                                rid2node[c].date_time = next_time
                                # there seems no way to update c in open_set, so just put c in open_set.
                                # the higher priority c will be searched first.
                                # so this is still work.
                                open_set.put((c_cost, rid2node[c]))
                                if (len(rid2node[c].trace_loc) >= best_default_len) \
                                        and c_cost < best_cost:
                                    # give the default recommended trace
                                    best_trace = (rid2node[c].trace_loc, rid2node[c].trace_tim)
                                    best_cost = c_cost
                step += 1
        if best_trace is not None:
            return best_trace[0], best_trace[1]
        else:
            return trace_loc, trace_tim


class DoubleLayerSearcher(object):
    """
    双层搜索器，先生成区域路径，再根据区域路径生成最终的道路路径
    """

    def __init__(self, device, adjacent_list, road_center_gps, road_length,
                 region_adjacent_list, region_dist, region_transfer_freq,
                 rid2region, region2rid, road_time_distribution, region_time_distribution):
        """
        这里初始化一些搜索算法共有的数据

        Args:
            device: torch 设备
            adjacent_list: 路网邻接表
            road_center_gps: 路段的 GPS 信息
            road_length: 路段的长度信息，用于预测时间
            region_adjacent_list: 区域邻接信息
            region_dist: 区域之间的距离信息，用于预测时间。单位米。(用一个 np.array 做就可以了)
            region_transfer_freq: 区域间转移边界路段频率
            rid2region: 路段与区域之间的映射字典
            region2rid: 区域与路段之间的映射字典
        """
        self.device = device
        self.adjacent_list = adjacent_list
        self.road_center_gps = road_center_gps
        self.road_length = road_length
        self.region_adjacent_list = region_adjacent_list
        self.region_dist = region_dist
        self.region_transfer_freq = region_transfer_freq
        self.rid2region = rid2region
        self.region2rid = region2rid
        self.region_time_distribution = region_time_distribution.astype(int)
        # 套用 road 层搜索器，降低代码难度
        self.road_search = Searcher(device, adjacent_list, road_center_gps, road_length, road_time_distribution)
        # log
        self.fail_find_log = set()
        self.region_fail_find_log = set()

    def astar_search(self, region_model, road_model, start_rid, start_tim, des, default_len, max_step=500):
        """
        先生成区域层的路径，再使用 road_model 进行细化。区域层 Base 方法最短路径法。
        Args:
            region_model: 区域生成器
            road_model: 路段生成器
            start_rid: 出发路段
            start_tim: 出发时刻
            des: 目的路段
            default_len: 默认生成路径长度
            max_step: 最大搜索步长

        Returns:
            trace_loc: 生成轨迹的 rid 序列
            trace_tim: 生成轨迹的 tim 序列
        """
        # 先将起始点映射为区域，进行生成
        start_region = self.rid2region[str(start_rid)]
        des_region = self.rid2region[str(des)]
        region_trace, is_astar = self.astar_search_in_region(region_model, start_region, start_tim, des_region,
                                                             max_step)
        if region_trace is None:
            # 区域上就没生成
            # 这不应该发生
            print('start_region {}, start_tim {}, des_region {}'.format(start_region, start_tim, des_region))
            raise AssertionError('region is None')
            # return [start_rid], [start_tim]
        if is_astar != 1:
            self.region_fail_find_log.add((start_region, des_region))
        # 使用 road_model 生成细化的路段路径
        now_region = region_trace[0]
        trace_loc = [start_rid]
        trace_tim = [start_tim]
        for target_region in region_trace[1:]:
            # 生成从 now_region 到 target_region 的路径
            # 根据邻接性选择中间目的路段
            # 因为我们 region transfer prob 里面没有考虑目的 region
            # 所以可能不是所有边界路段都可达
            border_road_dict = self.region_transfer_freq[str(now_region)][str(target_region)]
            border_road_set = border_road_dict['transfer_rid']
            border_road_freq = np.array(border_road_dict['transfer_freq'])
            intermediate_road_set = np.random.choice(border_road_set, len(border_road_set),
                                                 p=border_road_freq / np.sum(border_road_freq),
                                                 replace=False)
            now_region_rid_set = set(self.region2rid[str(now_region)])
            success = False
            for intermediate_road in intermediate_road_set:
                # 尝试生成到达边界路段的轨迹
                trace_loc, trace_tim = self.road_search.astar_search_in_region(road_model, trace_loc, trace_tim,
                                                                               intermediate_road,
                                                                               default_len, now_region_rid_set, max_step)

                # 检查 trace_loc 是否找到了
                if trace_loc[-1] == intermediate_road:
                    success = True
                    break
                else:
                    # 如果没有成功，做一个记录
                    self.fail_find_log.add((trace_loc[-1], intermediate_road))
            if not success:
                return trace_loc, trace_tim, is_astar
            # 接着往下找
            now_region = target_region
        # 找最后一段
        assert now_region == region_trace[-1]
        now_region_rid_set = set(self.region2rid[str(now_region)])
        if des in now_region_rid_set:
            trace_loc, trace_tim = self.road_search.astar_search_in_region(road_model, trace_loc, trace_tim, des,
                                                                           default_len, now_region_rid_set, max_step)
        else:
            # 区域没有生成对
            trace_loc, trace_tim = self.road_search.astar_search(road_model, trace_loc, trace_tim, des,
                                                                 default_len, max_step)
        if trace_loc[-1] != des:
            self.fail_find_log.add((trace_loc[-1], des))
        return trace_loc, trace_tim, is_astar

    def astar_search_in_region(self, region_model, start_region, start_time, des_region, max_step=500):
        """
        以下均是在区域层进行搜索
        Args:
            region_model:
            start_region:
            start_time:
            des_region:
            max_step:

        Returns:
            仅返回区域路段序列
        """
        close_set = set()
        open_set = PriorityQueue()  # unvisited node set
        region2node = {}  # the dict key is rid, and value is corresponding search node
        trace_loc = [start_region]
        trace_tim = [start_time]
        des_tensor = torch.LongTensor([des_region]).to(self.device)
        # put l_s into open_set
        # 构建第一个搜索节点（根节点）
        weekday_off = 0
        if start_time >= 1440:
            weekday_off = 1440
        start_date_time = (start_time - weekday_off) * 60  # 转化为秒数
        start_node = SearchNode(trace_loc=trace_loc, trace_tim=trace_tim, rid=start_region,
                                date_time=start_date_time, log_prob=0)
        open_set.put((start_node.log_prob, start_node))
        region2node[trace_loc[-1]] = start_node
        max_search_step = max_step
        step = 0
        best_default_len = 10
        best_trace = None
        best_score = 0
        while not open_set.empty() and step < max_search_step:
            # get min f cost node to search
            cost, now_node = open_set.get()
            # put now node in close_set
            now_region = now_node.rid
            if now_region in close_set:
                # now_node is an old node
                # now_region has been visited
                continue
            close_set.add(now_region)
            # del now_id from rid2node, because now node has been visited
            region2node.pop(now_region, None)
            if now_region == des_region:
                # find destination
                trace_loc = now_node.trace_loc
                trace_tim = now_node.trace_tim
                best_trace = (trace_loc, trace_tim)
                break
            else:
                # search now's adjacent rid
                if str(now_region) in self.region_adjacent_list:
                    candidate_region_dict = self.region_adjacent_list[str(now_region)]
                    candidate_set = [eval(k) for k in candidate_region_dict.keys()]
                    if len(candidate_set) == 0:
                        # 没有下一跳邻居，跳过
                        continue
                    if len(candidate_set) == 1:
                        # 只有这一个选择，那就不用计算了
                        # 此时选择不增加 Log prob
                        candidate_log_prob = now_node.log_prob
                        c_score = np.exp(candidate_log_prob)
                        # 更新时间
                        # 按照区域平均通勤距离/默认速度计算
                        candidate_region = candidate_set[0]
                        # 使用平均通勤时间
                        now_hour = now_node.date_time // 3600
                        if self.region_time_distribution[now_hour][now_region] != 1:
                            cost_time = self.region_time_distribution[now_hour][now_region]
                        else:
                            cost_time = int(
                                round(self.region_dist[now_region][candidate_region] / default_speed))  # 单位秒
                        next_time = (now_node.date_time + cost_time) % 86400
                        next_time_code = next_time // 60 + weekday_off
                        new_trace_tim = now_node.trace_tim + [next_time_code]
                        if candidate_region not in region2node and candidate_region not in close_set:
                            # c 没有被搜索到过
                            new_trace_loc = now_node.trace_loc + [candidate_region]
                            # 构建新的分支节点

                            new_node = SearchNode(trace_loc=new_trace_loc, trace_tim=new_trace_tim, rid=candidate_region,
                                                  date_time=next_time, log_prob=candidate_log_prob)
                            # put c in open_set
                            open_set.put((-candidate_log_prob, new_node))
                            region2node[candidate_region] = new_node
                            if (len(new_node.trace_loc) == best_default_len) \
                                    and c_score > best_score:
                                # give the default recommended trace
                                best_trace = (new_node.trace_loc, new_node.trace_tim)
                                best_score = c_score
                        elif candidate_region in region2node and region2node[candidate_region].log_prob \
                                < candidate_log_prob:
                            # update search node
                            region2node[candidate_region].log_prob = candidate_log_prob
                            new_trace_loc = now_node.trace_loc + [candidate_region]
                            region2node[candidate_region].trace_loc = new_trace_loc
                            region2node[candidate_region].trace_tim = new_trace_tim
                            region2node[candidate_region].date_time = next_time
                            # there seems no way to update c in open_set, so just put c in open_set.
                            # the higher priority c will be searched first.
                            # so this is still work.
                            open_set.put((-candidate_log_prob, region2node[candidate_region]))
                            if (len(region2node[candidate_region].trace_loc) == best_default_len) \
                                    and c_score > best_score:
                                # give the default recommended trace
                                best_trace = (region2node[candidate_region].trace_loc, region2node[candidate_region].trace_tim)
                                best_score = c_score
                    else:
                        candidate_dis = []
                        for c in candidate_set:
                            candidate_dis.append(self.region_dist[c][des_region] / 100)  # 单位百米
                        # 构建模型输入
                        trace_loc_tensor = torch.LongTensor(now_node.trace_loc).to(self.device).unsqueeze(0)
                        trace_tim_tensor = torch.LongTensor(now_node.trace_tim).to(self.device).unsqueeze(0)
                        candidate_set_tensor = torch.LongTensor(candidate_set).to(self.device).unsqueeze(0)
                        candidate_dis_tensor = torch.FloatTensor(candidate_dis).to(self.device).unsqueeze(0)
                        # 根据模型计算 f 函数值
                        with torch.no_grad():
                            candidate_f_cost = region_model.predict(trace_loc=trace_loc_tensor,
                                                                    trace_time=trace_tim_tensor,
                                                                    des=des_tensor, candidate_set=candidate_set_tensor,
                                                                    candidate_dis=candidate_dis_tensor)
                        # update each candidate's node
                        # calculate next datetime
                        candidate_log_prob = torch.log(candidate_f_cost).squeeze(0)  # (candidate_size)
                        pre_log_prob = now_node.log_prob
                        # 累计条件概率和
                        candidate_log_prob += pre_log_prob
                        for index, c in enumerate(candidate_set):
                            c_log_prob = candidate_log_prob[index].item()
                            c_score = np.exp(c_log_prob)
                            # 更新时间
                            # 使用平均通勤时间
                            now_hour = now_node.date_time // 3600
                            if self.region_time_distribution[now_hour][now_region] != 1:
                                cost_time = self.region_time_distribution[now_hour][now_region]
                            else:
                                cost_time = int(
                                    round(self.region_dist[now_region][c] / default_speed))  # 单位秒
                            next_time = (now_node.date_time + cost_time) % 86400
                            next_time_code = next_time // 60 + weekday_off
                            new_trace_tim = now_node.trace_tim + [next_time_code]
                            if c not in region2node and c not in close_set:
                                # c 没有被搜索到过
                                new_trace_loc = now_node.trace_loc + [c]
                                # 构建新的分支节点
                                new_node = SearchNode(trace_loc=new_trace_loc, trace_tim=new_trace_tim, rid=c,
                                                      date_time=next_time, log_prob=c_log_prob)
                                # put c in open_set
                                open_set.put((-c_log_prob, new_node))
                                region2node[c] = new_node
                                if (len(new_node.trace_loc) == best_default_len) \
                                        and c_score > best_score:
                                    # give the default recommended trace
                                    best_trace = (new_node.trace_loc, new_node.trace_tim)
                                    best_score = c_score
                            elif c in region2node and region2node[c].log_prob < c_log_prob:
                                # update search node
                                region2node[c].log_prob = c_log_prob
                                new_trace_loc = now_node.trace_loc + [c]
                                region2node[c].trace_loc = new_trace_loc
                                region2node[c].trace_tim = new_trace_tim
                                region2node[c].date_time = next_time
                                # there seems no way to update c in open_set, so just put c in open_set.
                                # the higher priority c will be searched first.
                                # so this is still work.
                                open_set.put((-c_log_prob, region2node[c]))
                                if (len(region2node[c].trace_loc) == best_default_len) \
                                        and c_score > best_score:
                                    # give the default recommended trace
                                    best_trace = (region2node[c].trace_loc, region2node[c].trace_tim)
                                    best_score = c_score
                step += 1
        # 检查是否找到
        assert best_trace is not None
        if best_trace[0][-1] == des_region:
            return best_trace[0], 1
        else:
            return best_trace[0], 0

    def dijkstra_region_search(self, start_region, des_region):
        """
        使用 Dijkstra 算法来寻找最短路径，因为区域层节点数也就 800 多个应该还是比较好找的。
        Args:
            start_region:
            des_region:

        Returns:
            trace_loc
        """
        dist = {}  # 存放已知节点到起点的最短距离
        queue = PriorityQueue()  # 以距离起点最近的队列
        start_node = SearchNode(trace_loc=[start_region], trace_tim=None, rid=start_region,
                                date_time=None, log_prob=0)
        queue.put((0, start_node))
        while not queue.empty():
            cost, now_node = queue.get()
            now_region = now_node.rid
            # 需要检查 now_region 是否已经被访问过了
            if now_region in dist:
                continue
            now_dist = now_node.log_prob
            dist[now_region] = now_dist
            if now_region == des_region:
                # 找到了
                return now_node.trace_loc
            else:
                # 没找到，继续遍历
                if str(now_region) in self.region_adjacent_list:
                    candidate_region_dict = self.region_adjacent_list[str(now_region)]
                    candidate_set = [eval(k) for k in candidate_region_dict.keys()]
                    if len(candidate_set) == 0:
                        # 没有下一跳邻居，跳过
                        continue
                    for candidate_region in candidate_set:
                        candidate_trace_loc = now_node.trace_loc + [candidate_region]
                        candidate_dist = now_dist + self.region_dist[now_region][candidate_region]
                        candidate_node = SearchNode(trace_loc=candidate_trace_loc, trace_tim=None, rid=candidate_region,
                                                    date_time=None, log_prob=candidate_dist)
                        # 可能会重复放入，但是总是最优的会被先取出来
                        queue.put((candidate_dist, candidate_node))
        # 代表没有找到，这应该不可能，因此返回 start_region
        return None

    def region_random_sample(self, region_model, trace_loc, trace_tim, des, default_len):
        """
        在对抗学习中，使用 astar 感觉效果不好，不如尝试像 SeqGAN 那种的根据概率的生成方式
        Args:
            region_model: 区域生成模型
            trace_loc: 开始区域
            trace_tim: 开始时间
            des: 终点区域
            default_len: 真实轨迹长度
            max_len: 生成的最大步长

        Returns:
            region_trace_loc, region_trace_tim
        """
        trace_loc = copy.deepcopy(trace_loc)
        trace_tim = copy.deepcopy(trace_tim)
        des_tensor = torch.LongTensor([des]).to(self.device)
        # 使用 now_minute 每步加上 1 分钟来预估时间
        weekday_off = 0
        start_time = trace_tim[-1]
        if start_time >= 1440:
            weekday_off = 1440
        step = len(trace_loc) - 1
        max_step = max(20, default_len)
        now_region = trace_loc[-1]
        now_date_time = (start_time - weekday_off) * 60  # 转化为秒数
        while now_region != des and step < max_step:
            # 这里是贪心的做法，非常害怕效果不好
            # 但我们是短距离，其实应该还好
            if str(now_region) in self.region_adjacent_list:
                # search now's adjacent rid
                candidate_region_dict = self.region_adjacent_list[str(now_region)]
                candidate_set = [eval(k) for k in candidate_region_dict.keys()]
                if len(candidate_set) == 0:
                    # 没有下一跳了。死路了，就 Break 就好了
                    break
                assert len(candidate_set) != 0
                if len(candidate_set) == 1:
                    # 只有这一个选择，那就不用计算了
                    # 更新时间
                    # 按照区域平均通勤距离/默认速度计算
                    next_region = candidate_set[0]
                    # 使用平均通勤时间
                    now_hour = now_date_time // 3600
                    if self.region_time_distribution[now_hour][next_region] != 1:
                        cost_time = self.region_time_distribution[now_hour][next_region]
                    else:
                        cost_time = int(
                            round(self.region_dist[now_region][next_region] / default_speed))  # 单位秒
                    next_time = (now_date_time + cost_time) % 86400
                    next_time_code = next_time // 60 + weekday_off
                    trace_tim.append(next_time_code)
                    trace_loc.append(next_region)
                    now_region = next_region
                else:
                    candidate_dis = []
                    for c in candidate_set:
                        candidate_dis.append(self.region_dist[c][des] / 100)  # 单位百米
                    # 构建模型输入
                    trace_loc_tensor = torch.LongTensor(trace_loc).to(self.device).unsqueeze(0)
                    trace_tim_tensor = torch.LongTensor(trace_tim).to(self.device).unsqueeze(0)
                    candidate_set_tensor = torch.LongTensor(candidate_set).to(self.device).unsqueeze(0)
                    candidate_dis_tensor = torch.FloatTensor(candidate_dis).to(self.device).unsqueeze(0)
                    # 根据模型计算 f 函数值
                    with torch.no_grad():
                        candidate_f_prob = region_model.predict(trace_loc=trace_loc_tensor,
                                                                trace_time=trace_tim_tensor,
                                                                des=des_tensor, candidate_set=candidate_set_tensor,
                                                                candidate_dis=candidate_dis_tensor)
                    # 根据概率随机选择一个点来作为下一跳
                    next_region_index = candidate_f_prob.squeeze(0).multinomial(1).item()
                    next_region = candidate_set[next_region_index]
                    # 更新时间
                    # 使用平均通勤时间
                    now_hour = now_date_time // 3600
                    if self.region_time_distribution[now_hour][now_region] != 1:
                        cost_time = self.region_time_distribution[now_hour][now_region]
                    else:
                        cost_time = int(
                            round(self.region_dist[now_region][next_region] / default_speed))  # 单位秒
                    next_time = (now_date_time + cost_time) % 86400
                    next_time_code = next_time // 60 + weekday_off
                    trace_tim.append(next_time_code)
                    trace_loc.append(next_region)
                    now_region = next_region
            else:
                # 走到死路了
                break
            step += 1
        if trace_loc[-1] == des:
            return trace_loc, trace_tim
        else:
            # 没找到，返回 default_len 个点
            return trace_loc[:default_len], trace_tim[:default_len]

    def save_fail_log(self):
        # 要转换成 json 可保存的格式
        error_list = []
        for error in self.fail_find_log:
            error_list.append([int(error[0]), int(error[1])])
        with open('./log/find_error_log.json', 'w') as f:
            json.dump(error_list, f)
        error_list = []
        for error in self.region_fail_find_log:
            error_list.append([int(error[0]), int(error[1])])
        with open('./log/region_find_error_log.json', 'w') as f:
            json.dump(error_list, f)

    def astar_search_only_road(self, region_model, road_model, start_rid, start_tim, des, true_region_trace,
                               default_len, max_step=500):
        # 使用 road_model 生成细化的路段路径
        now_region = true_region_trace[0]
        trace_loc = [start_rid]
        trace_tim = [start_tim]
        for target_region in true_region_trace[1:]:
            # 生成从 now_region 到 target_region 的路径
            # 根据邻接性选择中间目的路段
            # 因为我们 region transfer prob 里面没有考虑目的 region
            # 所以可能不是所有边界路段都可达
            border_road_dict = self.region_transfer_freq[str(now_region)][str(target_region)]
            border_road_set = border_road_dict['transfer_rid']
            border_road_freq = np.array(border_road_dict['transfer_freq'])
            intermediate_road_set = np.random.choice(border_road_set, len(border_road_set),
                                                     p=border_road_freq / np.sum(border_road_freq),
                                                     replace=False)
            now_region_rid_set = set(self.region2rid[str(now_region)])
            success = False
            for intermediate_road in intermediate_road_set:
                # 尝试生成到达边界路段的轨迹
                trace_loc, trace_tim = self.road_search.astar_search_in_region(road_model, trace_loc, trace_tim,
                                                                               intermediate_road,
                                                                               default_len, now_region_rid_set,
                                                                               max_step)

                # 检查 trace_loc 是否找到了
                if trace_loc[-1] == intermediate_road:
                    success = True
                    break
                else:
                    # 如果没有成功，做一个记录
                    self.fail_find_log.add((trace_loc[-1], intermediate_road))
            if not success:
                return trace_loc, trace_tim
            # 接着往下找
            now_region = target_region
        # 找最后一段
        assert now_region == true_region_trace[-1]
        now_region_rid_set = set(self.region2rid[str(now_region)])
        assert des in now_region_rid_set
        trace_loc, trace_tim = self.road_search.astar_search_in_region(road_model, trace_loc, trace_tim, des,
                                                                       default_len, now_region_rid_set, max_step)
        if trace_loc[-1] != des:
            self.fail_find_log.add((trace_loc[-1], des))
        return trace_loc, trace_tim