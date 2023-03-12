# 生成区域生成器 G 函数部分预训练数据
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from geopy import distance
import numpy as np
import os

max_step = 4
random_encode = True  # 随机步数 encode，主要是减少数据量，避免过拟合，因为区域轨迹都比较短，所以就不跳步了


dataset_name = 'Xian'

if dataset_name == 'BJ_Taxi':
    # 读取路段与区域之间的映射关系
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/kaffpa_tarjan_rid2region.json', 'r') as f:
        rid2region = json.load(f)
    # 读取区域邻接表
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/kaffpa_tarjan_region_adjacent_list.json', 'r') as f:
        region_adjacent_list = json.load(f)
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/kaffpa_tarjan_region_gps.json', 'r') as f:
        region_gps = json.load(f)
    train_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_train.csv')
    eval_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_eval.csv')
    test_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_test.csv')
elif dataset_name == 'Porto_Taxi':
    # 读取路段与区域之间的映射关系
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_rid2region.json', 'r') as f:
        rid2region = json.load(f)
    # 读取区域邻接表
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_region_adjacent_list.json', 'r') as f:
        region_adjacent_list = json.load(f)
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_region_gps.json', 'r') as f:
        region_gps = json.load(f)
    train_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_train.csv')
    eval_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_eval.csv')
    test_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_test.csv')
else:
    # 读取路段与区域之间的映射关系
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/rid2region.json', 'r') as f:
        rid2region = json.load(f)
    # 读取区域邻接表
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/rid_gps.json', 'r') as f:
        rid_gps = json.load(f)
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/region2rid.json', 'r') as f:
        region2rid = json.load(f)
    region_gps = {}
    for region in region2rid:
        rid_set = region2rid[region]
        lat_list = []
        lon_list = []
        for rid in rid_set:
            rid_center_gps = rid_gps[str(rid)]
            lon_list.append(rid_center_gps[0])
            lat_list.append(rid_center_gps[1])
        # TODO: 这里是几何中心，不一定科学
        region_center = (np.average(lon_list), np.average(lat_list))
        region_gps[region] = region_center
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/region_gps.json', 'w') as f:
        json.dump(region_gps, f)
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/region_adjacent_list.json', 'r') as f:
        region_adjacent_list = json.load(f)
    train_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/xianshi_mm_region_train.csv')
    eval_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/xianshi_mm_region_eval.csv')
    test_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/xianshi_mm_region_test.csv')


def encode_time(timestamp):
    """
    编码时间
    """
    # 按
    time = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
    if time.weekday() == 5 or time.weekday() == 6:
        return time.hour * 60 + time.minute + 1440
    else:
        return time.hour * 60 + time.minute


def encode_trace(trace, fp):
    """
    编码轨迹

    Args:
        trace: 一条轨迹记录
        fp: 写入编码结果的文件
    """
    region_list = [int(i) for i in trace['region_list'].split(',')]
    time_list = [encode_time(i) for i in trace['time_list'].split(',')]
    des = region_list[-1]
    des_gps = region_gps[str(des)]
    # 训练数据还是感觉有点多
    # 这里为了避免过拟合，还是随机步数 encode 吧
    # 可以做个对比实验看哪个效果好一点
    if not random_encode:
        for i in range(1, len(region_list)):
            cur_loc = region_list[:i]
            cur_time = time_list[:i]
            cur_region = cur_loc[-1]
            if str(cur_region) not in region_adjacent_list or str(region_list[i]) not in region_adjacent_list[str(cur_region)]:
                # 这不应该发生，如果发生了则舍弃掉后面的路径
                # 说明发生了断路
                return
            candidate_set = list(region_adjacent_list[str(cur_region)].keys())
            if len(candidate_set) > 1:
                # 对于有多个候选点的才有学习的价值
                target = str(region_list[i])
                target_index = 0
                candidate_dis = []
                for index, c in enumerate(candidate_set):
                    if c == target:
                        target_index = index
                    c_gps = region_gps[c]
                    dis = distance.distance((des_gps[1], des_gps[0]), (c_gps[1], c_gps[0])).kilometers  # 单位为千米
                    candidate_dis.append(dis)
                # 开始写入编码结果
                cur_loc_str = ",".join([str(i) for i in cur_loc])
                cur_time_str = ",".join([str(i) for i in cur_time])
                candidate_set_str = ",".join([str(i) for i in candidate_set])
                candidate_dis_str = ",".join([str(i) for i in candidate_dis])
                fp.write("\"{}\",\"{}\",{},\"{}\",\"{}\",{}\n".format(cur_loc_str, cur_time_str, des, candidate_set_str,
                                                                      candidate_dis_str, target_index))
    else:
        i = 1
        while i < len(region_list):
            cur_loc = region_list[:i]
            cur_time = time_list[:i]
            cur_region = cur_loc[-1]
            if str(cur_region) not in region_adjacent_list or str(region_list[i]) not in region_adjacent_list[
                str(cur_region)]:
                # 这不应该发生，如果发生了则舍弃掉后面的路径
                # 说明发生了断路
                return
            candidate_set = list(region_adjacent_list[str(cur_region)].keys())
            if len(candidate_set) > 1:
                # 对于有多个候选点的才有学习的价值
                target = str(region_list[i])
                target_index = 0
                candidate_dis = []
                for index, c in enumerate(candidate_set):
                    if c == target:
                        target_index = index
                    c_gps = region_gps[c]
                    dis = distance.distance((des_gps[1], des_gps[0]), (c_gps[1], c_gps[0])).kilometers * 10  # 单位为百米
                    candidate_dis.append(dis)
                # 开始写入编码结果
                cur_loc_str = ",".join([str(i) for i in cur_loc])
                cur_time_str = ",".join([str(i) for i in cur_time])
                candidate_set_str = ",".join([str(i) for i in candidate_set])
                candidate_dis_str = ",".join([str(i) for i in candidate_dis])
                fp.write("\"{}\",\"{}\",{},\"{}\",\"{}\",{}\n".format(cur_loc_str, cur_time_str, des, candidate_set_str,
                                                                      candidate_dis_str, target_index))
            # i 不再是 ++ 而是随机加一定步数
            step = np.random.randint(1, max_step)
            i += step


if __name__ == '__main__':
    if dataset_name == 'BJ_Taxi':
        train_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_region_pretrain_input_train'), 'w')
        eval_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_region_pretrain_input_eval'), 'w')
        test_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_region_pretrain_input_test'), 'w')
    elif dataset_name == 'Porto_Taxi':
        train_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_region_pretrain_input_train'), 'w')
        eval_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_region_pretrain_input_eval'), 'w')
        test_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_region_pretrain_input_test'), 'w')
    else:
        assert dataset_name == 'Xian'
        train_output = open(
            '/mnt/data/jwj/TS_TrajGen_data_archive/Xian/{}.csv'.format('xianshi_region_pretrain_input_train'), 'w')
        eval_output = open(
            '/mnt/data/jwj/TS_TrajGen_data_archive/Xian/{}.csv'.format('xianshi_region_pretrain_input_eval'), 'w')
        test_output = open(
            '/mnt/data/jwj/TS_TrajGen_data_archive/Xian/{}.csv'.format('xianshi_region_pretrain_input_test'), 'w')
    train_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
    eval_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
    test_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
    for index, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc='encode train traj'):
        encode_trace(row, train_output)
    for index, row in tqdm(eval_data.iterrows(), total=eval_data.shape[0], desc='encode eval traj'):
        encode_trace(row, eval_output)
    for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc='encode test traj'):
        encode_trace(row, test_output)
    train_output.close()
    eval_output.close()
    test_output.close()
