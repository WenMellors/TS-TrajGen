# 将路段轨迹映射为区域轨迹
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# 读取路段与区域之间的映射关系
with open('../data/kaffpa_tarjan_rid2region.json', 'r') as f:
    rid2region = json.load(f)

# 读取区域邻接表
with open('../data/kaffpa_tarjan_region_adjacent_list.json', 'r') as f:
    region_adjacent_list = json.load(f)

mm_traj = pd.read_csv('../data/201511_week1_mm_filter_by_user.csv')

# 加载预划分好的标记
train_id_set = np.load('../data/201511_train_id_set.npy')
eval_id_set = np.load('../data/201511_eval_id_set.npy')
test_id_set = np.load('../data/201511_test_id_set.npy')
# 转换成 set 加速查询
train_id_set = set(train_id_set)
eval_id_set = set(eval_id_set)
test_id_set = set(test_id_set)
# 开始 Map
headers = 'mm_id,entity_id,traj_id,region_list,time_list\n'
train_file = open('../data/201511_week1_mm_region_train.csv', 'w')
eval_file = open('../data/201511_week1_mm_region_eval.csv', 'w')
test_file = open('../data/201511_week1_mm_region_test.csv', 'w')

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


for index, row in tqdm(mm_traj.iterrows(), total=mm_traj.shape[0], desc='map traj'):
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
    if mm_id in train_id_set:
        write_row(train_file, row, map_region_list, map_time_list)
    elif mm_id in eval_id_set:
        write_row(eval_file, row, map_region_list, map_time_list)
    else:
        write_row(test_file, row, map_region_list, map_time_list)

train_file.close()
eval_file.close()
test_file.close()
