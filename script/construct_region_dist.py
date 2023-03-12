# 因为区域数目不多，所以这里可以都计算一个
import json
import os
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from geopy import distance
# 内存可能会炸吗？

# 主要对于 OD 间有多条轨迹的需要取一个平均
distance_dict = {}  # 记录 f_region, t_region, distance
dataset_name = 'Xian'
# 读取路网长度字典
road_len_file = '../data/Xian/road_length.json'
if not os.path.exists(road_len_file):
    road_info = pd.read_csv('../data/Xian/xian.geo')
    road_length = {}
    for index, row in tqdm(road_info.iterrows(), desc='cal road length'):
        rid = row['geo_id']
        length = row['length']
        road_length[str(rid)] = length
    # 保存
    with open(road_len_file, 'w') as f:
        json.dump(road_length, f)
else:
    with open(road_len_file, 'r') as f:
        road_length = json.load(f)

# 读取路段区域映射表
with open('../data/Xian/rid2region.json', 'r') as f:
    rid2region = json.load(f)


# 开始遍历轨迹数据
traj = pd.read_csv('../data/Xian/xianshi_partA_traj_mm_processed.csv')
for index, row in tqdm(traj.iterrows(), desc='count traj', total=traj.shape[0]):
    rid_list = [int(i) for i in row['rid_list'].split(',')]
    if len(rid_list) < 2:
        continue
    count_length = 0
    step_length = []
    for i in range(len(rid_list)):
        # 因为 road_length 的单位是米，所以这里做个缩放感觉会好一点
        # 目前搞成了千米
        count_length += road_length[str(rid_list[i])]
        step_length.append(count_length)
    for i in range(len(rid_list)):
        f_rid = rid_list[i]
        f_region = rid2region[str(f_rid)]
        for j in range(i + 1, len(rid_list)):
            t_rid = rid_list[j]
            t_region = rid2region[str(t_rid)]
            travel_length = step_length[j] - step_length[i]
            if t_region != f_region:
                if f_region not in distance_dict:
                    distance_dict[f_region] = {}
                    distance_dict[f_region][t_region] = (travel_length, 1)
                elif t_region not in distance_dict[f_region]:
                    distance_dict[f_region][t_region] = (travel_length, 1)
                else:
                    pair = distance_dict[f_region][t_region]
                    distance_dict[f_region][t_region] = (pair[0] + travel_length, pair[1] + 1)

# 还需要计算 f_region 与 t_region 之间的直线距离
with open('../data/Xian/region_gps.json', 'r') as f:
    region_gps = json.load(f)

# 根据统计得到的 distance_dict 以及经纬度来生成距离矩阵
region_num = len(region_gps)
region_dist = np.zeros((region_num, region_num), dtype=float)
for f_region in tqdm(range(region_num), desc='generate region dist'):
    f_gps = region_gps[str(f_region)]
    for t_region in range(region_num):
        if f_region != t_region:
            if f_region in distance_dict and t_region in distance_dict[f_region]:
                pair = distance_dict[f_region][t_region]
                avg_length = pair[0] / pair[1]
                region_dist[f_region][t_region] = avg_length
            else:
                region_dist[f_region][t_region] = -1

np.save('../data/Xian/region_count_dist', region_dist)
