import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from geopy import distance
import numpy as np
import os

dataset_name = 'BJ_Taxi'
max_step = 4
random_encode = True  # 随机步数 encode，主要是减少数据量，避免过拟合
if dataset_name == 'BJ_Taxi':
    train_data = pd.read_csv('/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_train.csv')
    test_data = pd.read_csv('/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_test.csv')
else:
    train_data = pd.read_csv('/mnt/data/jwj/Porto_Taxi/porto_mm_train.csv')
    test_data = pd.read_csv('/mnt/data/jwj/Porto_Taxi/porto_mm_test.csv')


# 读取路网邻接表
if dataset_name == 'BJ_Taxi':
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/adjacent_list.json', 'r') as f:
        adjacent_list = json.load(f)
else:
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_adjacent_list.json', 'r') as f:
        adjacent_list = json.load(f)
# 读取路网信息表
if dataset_name == 'BJ_Taxi':
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/rid_gps.json', 'r') as f:
        rid_gps = json.load(f)
else:
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_rid_gps.json', 'r') as f:
        rid_gps = json.load(f)


def encode_time(timestamp):
    """
    编码时间
    """
    # 按一分钟编码，周末与工作日区分开来
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
    rid_list = [int(i) for i in trace['rid_list'].split(',')]
    time_list = [encode_time(i) for i in trace['time_list'].split(',')]
    des = rid_list[-1]
    des_gps = rid_gps[str(des)]
    # 训练数据还是感觉有点多
    # 这里为了避免过拟合，还是随机步数 encode 吧
    # 可以做个对比实验看哪个效果好一点
    if not random_encode:
        for i in range(1, len(rid_list)):
            cur_loc = rid_list[:i]
            cur_time = time_list[:i]
            cur_rid = cur_loc[-1]
            if str(cur_rid) not in adjacent_list or rid_list[i] not in adjacent_list[str(cur_rid)]:
                # 这不应该发生，如果发生了则舍弃掉后面的路径
                # 说明发生了断路
                return
            candidate_set = adjacent_list[str(cur_rid)]
            if len(candidate_set) > 1:
                # 对于有多个候选点的才有学习的价值
                target = rid_list[i]
                target_index = 0
                candidate_dis = []
                for index, c in enumerate(candidate_set):
                    if c == target:
                        target_index = index
                    c_gps = rid_gps[str(c)]
                    dis = distance.distance((des_gps[1], des_gps[0]), (c_gps[1], c_gps[0])).kilometers * 10 # 单位为百米
                    candidate_dis.append(dis)
                # 开始写入编码结果
                cur_loc_str = ",".join([str(i) for i in cur_loc])
                cur_time_str = ",".join([str(i) for i in cur_time])
                candidate_set_str = ",".join([str(i) for i in candidate_set])
                candidate_dis_str = ",".join([str(i) for i in candidate_dis])
                fp.write("\"{}\",\"{}\",{},\"{}\",\"{}\",{}\n".format(cur_loc_str, cur_time_str, des, candidate_set_str, candidate_dis_str, target_index))
    else:
        i = 1
        while i < len(rid_list):
            cur_loc = rid_list[:i]
            cur_time = time_list[:i]
            cur_rid = cur_loc[-1]
            if str(cur_rid) not in adjacent_list or rid_list[i] not in adjacent_list[str(cur_rid)]:
                # 这不应该发生，如果发生了则舍弃掉后面的路径
                # 说明发生了断路
                return
            candidate_set = adjacent_list[str(cur_rid)]
            if len(candidate_set) > 1:
                # 对于有多个候选点的才有学习的价值
                target = rid_list[i]
                target_index = 0
                candidate_dis = []
                for index, c in enumerate(candidate_set):
                    if c == target:
                        target_index = index
                    c_gps = rid_gps[str(c)]
                    dis = distance.distance((des_gps[1], des_gps[0]), (c_gps[1], c_gps[0])).kilometers * 10 # 单位为百米
                    candidate_dis.append(dis)
                # 开始写入编码结果
                cur_loc_str = ",".join([str(i) for i in cur_loc])
                cur_time_str = ",".join([str(i) for i in cur_time])
                candidate_set_str = ",".join([str(i) for i in candidate_set])
                candidate_dis_str = ",".join([str(i) for i in candidate_dis])
                fp.write("\"{}\",\"{}\",{},\"{}\",\"{}\",{}\n".format(cur_loc_str, cur_time_str, des, candidate_set_str, candidate_dis_str, target_index))
            # i 不再是 ++ 而是随机加一定步数
            step = np.random.randint(1, max_step)
            i += step


if __name__ == '__main__':
    train_rate = 0.9
    total_data_num = train_data.shape[0]
    train_num = int(total_data_num * train_rate)
    if dataset_name == 'BJ_Taxi':
        train_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_pretrain_input_train'), 'w')
        eval_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_pretrain_input_eval'), 'w')
        test_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_pretrain_input_test'), 'w')
    else:
        train_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_pretrain_input_train'), 'w')
        eval_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_pretrain_input_eval'), 'w')
        test_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_pretrain_input_test'), 'w')
    train_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
    eval_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
    test_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
    for index, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc='encode train traj'):
        if index <= train_num:
            encode_trace(row, train_output)
        else:
            encode_trace(row, eval_output)
    for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc='encode test traj'):
        encode_trace(row, test_output)
    train_output.close()
    eval_output.close()
    test_output.close()
