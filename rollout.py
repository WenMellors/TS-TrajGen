import copy
import torch
from utils.evaluate_funcs import dtw_metric
import numpy as np


class Rollout(object):
    """
    使用 Monte Carlo 算法为生成轨迹每一步的选择计算一个基于判别器的反馈
    """

    def __init__(self, searcher, generator, device, od_distinct_route, road_gps):
        """

        Args:
            searcher (Seacher): 根据生成器搜索生成路径
            generator (Generator): 生成器
        """
        self.searcher = searcher
        self.generator = copy.deepcopy(generator)  # 因为 Rollout 使用的是上一代的生成器，并不是最新的生成器
        self.device = device
        self.od_distinct_route = od_distinct_route
        self.road_gps = road_gps

    def get_region_reward(self, generate_trace, des, rollout_times, discriminator):
        """
        对一条生成轨迹的每一步，使用 Monte Carlo 算法预估一个判别器的反馈值
        备注：当生成器不具备多样性时，即多次蒙特卡洛采样结果将一致，此时直接将最终的生成结果的
        Args:
            generate_trace (tuple of list): 生成的轨迹 (trace_loc, trace_tim)
            des (int): 目的地 rid
            rollout_times (int): Monte Carlo 搜索的次数
            discriminator (Object): 判别器

        Returns:
            reward (list): 所生成轨迹每一步的反馈值 shape (seq_len - 1)
        """
        # 第一个点是起点，是人为给定的，所以不计算他的反馈值
        seq_len = len(generate_trace[0])
        rewards = []
        yaw_loss = [0]
        for j in range(2, seq_len):
            input_trace_loc = generate_trace[0][:j]
            input_trace_tim = generate_trace[1][:j]
            for i in range(rollout_times):
                search_trace_loc, search_trace_tim = self.searcher.region_random_sample(
                    region_model=self.generator, trace_loc=input_trace_loc, trace_tim=input_trace_tim, des=des,
                    default_len=seq_len)
                # discriminator_input = region2img(search_trace_loc).unsqueeze(0).to(self.device)
                # shape: (1)
                search_trace_loc = torch.tensor(search_trace_loc).unsqueeze(0).to(self.device)  # (1, seq_len)
                search_trace_tim = torch.tensor(search_trace_tim).unsqueeze(0).to(self.device)  # (1, seq_len)
                # shape: (2)
                score = discriminator.predict(trace_loc=search_trace_loc, trace_time=search_trace_tim).squeeze(0)
                # 取轨迹为真的概率值
                score = score[1].item()
                # 计算偏航损失的部分
                yaw_distance = self.yaw_loss(search_trace_loc, des)  # 这是一个减分项
                if i == 0:
                    rewards.append(score)
                    yaw_loss.append(yaw_distance)
                else:
                    rewards[-1] += score
                    yaw_loss[-1] += yaw_distance
            # 反馈值取平均
            rewards[-1] /= rollout_times
            yaw_loss[-1] /= rollout_times
        # 计算最后一个点的反馈值
        # discriminator_input = region2img(generate_trace[0]).unsqueeze(0).to(self.device)
        # shape: (1)
        search_trace_loc = torch.tensor(generate_trace[0]).unsqueeze(0).to(self.device)  # (1, seq_len)
        search_trace_tim = torch.tensor(generate_trace[1]).unsqueeze(0).to(self.device)  # (1, seq_len)
        score = discriminator.predict(trace_loc=search_trace_loc, trace_time=search_trace_tim).squeeze(0)
        # 取轨迹为真的概率值
        score = score[1].item()
        rewards.append(score)
        yaw_distance = self.yaw_loss(search_trace_loc, des)
        yaw_loss.append(yaw_distance)
        return rewards, yaw_loss

    def get_road_reward(self, generate_trace, des, rollout_times, discriminator):
        # 第一个点是起点，是人为给定的，所以不计算他的反馈值
        seq_len = len(generate_trace[0])
        rewards = []
        yaw_loss = [0]
        for j in range(2, seq_len):
            input_trace_loc = generate_trace[0][:j]
            input_trace_tim = generate_trace[1][:j]
            for i in range(rollout_times):
                search_trace_loc, search_trace_tim = self.searcher.road_random_sample(
                    gen_model=self.generator, trace_loc=input_trace_loc, trace_tim=input_trace_tim, des=des,
                    default_len=seq_len)
                # discriminator_input = region2img(search_trace_loc).unsqueeze(0).to(self.device)
                # shape: (1)
                search_trace_loc = torch.tensor(search_trace_loc).unsqueeze(0).to(self.device)  # (1, seq_len)
                search_trace_tim = torch.tensor(search_trace_tim).unsqueeze(0).to(self.device)  # (1, seq_len)
                # shape: (2)
                score = discriminator.predict(trace_loc=search_trace_loc, trace_time=search_trace_tim).squeeze(0)
                # 取轨迹为真的概率值
                score = score[1].item()
                # 计算偏航损失的部分
                yaw_distance = self.yaw_loss(search_trace_loc, des)  # 这是一个减分项
                if i == 0:
                    rewards.append(score)
                    yaw_loss.append(yaw_distance)
                else:
                    rewards[-1] += score
                    yaw_loss[-1] += yaw_distance
            # 反馈值取平均
            rewards[-1] /= rollout_times
            yaw_loss[-1] /= rollout_times
        # 计算最后一个点的反馈值
        # discriminator_input = region2img(generate_trace[0]).unsqueeze(0).to(self.device)
        # shape: (1)
        search_trace_loc = torch.tensor(generate_trace[0]).unsqueeze(0).to(self.device)  # (1, seq_len)
        search_trace_tim = torch.tensor(generate_trace[1]).unsqueeze(0).to(self.device)  # (1, seq_len)
        score = discriminator.predict(trace_loc=search_trace_loc, trace_time=search_trace_tim).squeeze(0)
        # 取轨迹为真的概率值
        score = score[1].item()
        rewards.append(score)
        yaw_distance = self.yaw_loss(search_trace_loc, des)
        yaw_loss.append(yaw_distance)
        return rewards, yaw_loss

    def get_reward_direct(self, generate_trace, discriminator):
        """
        直接判断每个点的真实性
        Args:
            generate_trace (tuple of list): 生成的轨迹 (trace_loc, trace_tim)
            discriminator (Object): 判别器

        Returns:
            reward (list): 所生成轨迹每一步的反馈值 shape (seq_len - 1)
        """
        # 第一个点是起点，是人为给定的，所以不计算他的反馈值
        seq_len = len(generate_trace[0])
        rewards = []
        for j in range(2, seq_len + 1):
            input_trace_loc = torch.tensor(generate_trace[0][:j]).unsqueeze(0).to(self.device)
            input_trace_tim = torch.tensor(generate_trace[1][:j]).unsqueeze(0).to(self.device)
            # shape: (2)
            score = discriminator.predict(trace_loc=input_trace_loc, trace_time=input_trace_tim).squeeze(0)
            # 取轨迹为真的概率值
            score = score[1].item()
            rewards.append(score)
        return rewards

    def update_params(self, new_generator):
        """
        在每轮生成器训练完后，应该 update Rollout 中的生成器
        Args:
            new_generator: 新一代的生成器
        """
        state_dict = copy.deepcopy(new_generator.state_dict())
        self.generator.load_state_dict(state_dict)

    def yaw_loss(self, generate_trace_loc, des):
        """
        与相同 OD 的历史轨迹比较，从而得出偏航损失
        Args:
            generate_trace_loc: 生成的轨迹
            des: 目的路段

        Returns:
            yaw_loss: 偏航损失
        """
        generate_trace_loc = generate_trace_loc[0].tolist()
        origin = generate_trace_loc[0]
        od_key = '{}-{}'.format(origin, des)
        if od_key not in self.od_distinct_route:
            return 0.0
        history_trace = self.od_distinct_route[od_key]
        # 选用 dwt 作为距离指标
        generate_gps_list = []
        for rid in generate_trace_loc:
            now_gps = self.road_gps[str(rid)]
            generate_gps_list.append([now_gps[1], now_gps[0]])
        history_trace_distance = []
        for trace in history_trace:
            dtw = dtw_metric(trace, generate_gps_list)
            history_trace_distance.append(dtw)
        min_dtw = np.min(history_trace_distance)
        return min_dtw
