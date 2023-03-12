# 将路网输出为 metis 格式的图描述
import numpy as np
import pandas as pd
from tqdm import tqdm

dataset_name = 'Xian'

with open('../data/{}/xian.graph'.format(dataset_name), 'w') as f:

    if dataset_name == 'Xian':
        road_info = pd.read_csv('../data/Xian/xian.geo')
        road_rel = pd.read_csv('../data/Xian/xian.rel')

    total_road_num = road_info.shape[0]

    # 需要删除孤立的路段，因此需要做一个重编码（双映射）
    # 找到孤立点
    outlier_set = set(road_info['geo_id'])
    for index, row in tqdm(road_rel.iterrows(), total=road_rel.shape[0], desc='find outlier'):
        f_id = row['origin_id']
        t_id = row['destination_id']
        if f_id in outlier_set:
            outlier_set.remove(f_id)
        if t_id in outlier_set:
            outlier_set.remove(t_id)
    print(outlier_set)
    rid2new = {}
    new2rid = {}
    new_id = 1
    for rid in range(total_road_num):
        if rid not in outlier_set:
            # 这个点不需要被删除，重新编码
            rid2new[rid] = new_id
            new2rid[new_id] = rid
            new_id += 1

    # 因为图分割算法只能处理无向图，所以这里边需要做额外的处理
    total_road_num = len(new2rid)
    assert total_road_num + 1 == new_id
    road_undirected_adj_mx = np.zeros((total_road_num, total_road_num)).astype(int)
    road_undirected_rel = {}
    total_edge_num = 0
    # 注意 road id 从 1 开始
    for index, row in tqdm(road_rel.iterrows(), total=road_rel.shape[0]):
        from_road = rid2new[row['origin_id']]
        to_road = rid2new[row['destination_id']]
        if from_road == to_road:
            # 自环就跳过了
            continue
        min_road = min(from_road, to_road)
        max_road = max(from_road, to_road)
        if min_road not in road_undirected_rel:
            road_undirected_rel[min_road] = {max_road}
            road_undirected_adj_mx[min_road - 1][max_road - 1] = 1
            road_undirected_adj_mx[max_road - 1][min_road - 1] = 1
            total_edge_num += 1
        elif max_road not in road_undirected_rel[min_road]:
            road_undirected_rel[min_road].add(max_road)
            road_undirected_adj_mx[min_road - 1][max_road - 1] = 1
            road_undirected_adj_mx[max_road - 1][min_road - 1] = 1
            total_edge_num += 1

    f.write('{} {}\n'.format(total_road_num, total_edge_num))
    print(total_road_num, total_edge_num)
    output_cnt = 0
    for road_id in range(1, total_road_num + 1):
        road_adjacent = (np.where(road_undirected_adj_mx[road_id - 1] == 1)[0] + 1).tolist()
        output_cnt += len(road_adjacent)
        adjacent_str = ' '.join([str(x) for x in road_adjacent])
        f.write(adjacent_str + '\n')
    print(output_cnt)
