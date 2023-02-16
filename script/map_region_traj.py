# 将路段轨迹映射为区域轨迹
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

dataset_name = 'BJ_Taxi'

if dataset_name == 'BJ_Taxi':
    # 读取路段与区域之间的映射关系
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/kaffpa_tarjan_rid2region.json', 'r') as f:
        rid2region = json.load(f)
    # 读取区域邻接表
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/kaffpa_tarjan_region_adjacent_list.json', 'r') as f:
        region_adjacent_list = json.load(f)
    train_mm_traj = pd.read_csv('/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_train.csv')
    test_mm_traj = pd.read_csv('/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_test.csv')
    # 开始 Map
    headers = 'mm_id,entity_id,traj_id,region_list,time_list\n'
    train_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_train.csv', 'w')
    eval_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_eval.csv', 'w')
    test_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_test.csv', 'w')
else:
    # 读取路段与区域之间的映射关系
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_rid2region.json', 'r') as f:
        rid2region = json.load(f)
    # 读取区域邻接表
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_region_adjacent_list.json', 'r') as f:
        region_adjacent_list = json.load(f)
    train_mm_traj = pd.read_csv('/mnt/data/jwj/Porto_Taxi/porto_mm_train.csv')
    test_mm_traj = pd.read_csv('/mnt/data/jwj/Porto_Taxi/porto_mm_test.csv')
    # 开始 Map
    headers = 'mm_id,entity_id,traj_id,region_list,time_list\n'
    train_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_train.csv', 'w')
    eval_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_eval.csv', 'w')
    test_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_test.csv', 'w')
train_file.write(headers)
eval_file.write(headers)
test_file.write(headers)


def write_row(write_file, write_row, region_list, time_list):
    """
    写入结果
    Args:
        write_file:
        write_row:
        region_list:
        time_list:

    Returns:

    """
    mm_id = write_row['mm_id']
    entity_id = write_row['entity_id']
    traj_id = write_row['traj_id']
    map_region_str = ','.join([str(x) for x in region_list])
    map_time_str = ','.join(time_list)
    write_file.write('{},{},{},\"{}\",\"{}\"\n'.format(mm_id, entity_id, traj_id,
                                                     map_region_str, map_time_str))


train_rate = 0.9
total_data_num = train_mm_traj.shape[0]
train_num = int(total_data_num * train_rate)

for index, row in tqdm(train_mm_traj.iterrows(), total=train_mm_traj.shape[0], desc='map traj'):
    mm_id = row['mm_id']
    # map
    rid_list = row['rid_list'].split(',')
    time_list = row['time_list'].split(',')
    start_region = rid2region[rid_list[0]]
    start_time = time_list[0]
    map_region_list = [start_region]
    map_time_list = [start_time]
    for j, rid in enumerate(rid_list[1:]):
        map_region = rid2region[rid]
        if map_region != map_region_list[-1]:
            map_region_list.append(map_region)
            map_time_list.append(time_list[j+1])
    if index <= train_num:
        write_row(train_file, row, map_region_list, map_time_list)
    else:
        write_row(eval_file, row, map_region_list, map_time_list)

for index, row in tqdm(test_mm_traj.iterrows(), total=test_mm_traj.shape[0], desc='map traj'):
    mm_id = row['mm_id']
    # map
    rid_list = row['rid_list'].split(',')
    time_list = row['time_list'].split(',')
    start_region = rid2region[rid_list[0]]
    start_time = time_list[0]
    map_region_list = [start_region]
    map_time_list = [start_time]
    for j, rid in enumerate(rid_list[1:]):
        map_region = rid2region[rid]
        if map_region != map_region_list[-1]:
            map_region_list.append(map_region)
            map_time_list.append(time_list[j+1])
    write_row(test_file, row, map_region_list, map_time_list)


train_file.close()
eval_file.close()
test_file.close()
