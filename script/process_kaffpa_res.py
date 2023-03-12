# 生成 region2rid 以及 rid2region 文件
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import tarjan

road_info = pd.read_csv('../data/Xian/xian.geo')
road_rel = pd.read_csv('../data/Xian/xian.rel')
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

print(outlier_set)  # {8802, 10545}
total_road_num = road_info.shape[0]
rid2new = {}
new2rid = {}
new_id = 1
for rid in range(total_road_num):
    if rid not in outlier_set:
        # 这个点不需要被删除，重新编码
        rid2new[rid] = new_id
        new2rid[new_id] = rid
        new_id += 1

region2rid = {}
rid2region = {}
with open('../data/Xian/tmppartition50', 'r') as f:
    road_id = 1
    for line in f:
        region_id = int(line)
        origin_road_id = new2rid[road_id]
        rid2region[origin_road_id] = region_id
        if region_id not in region2rid:
            region2rid[region_id] = [origin_road_id]
        else:
            region2rid[region_id].append(origin_road_id)
        road_id += 1

# kaffpa 无法保证划分出来的子图是无向连通的，因此我们这里需要做个检查，然后把不连通的切开
region2rid_true = {}
rid2region_true = {}
new_region_id = 0

with open('../data/Xian/adjacent_list.json', 'r') as f:
    adjacent_list = json.load(f)

assert len(rid2region) == road_info.shape[0] - len(outlier_set)

for region_id in region2rid:
    rid_set = region2rid[region_id]
    # 创建 networkx 图
    # nodes = []
    # edges = set()
    # for rid in rid_set:
    #     nodes.append(rid)
    #     if str(rid) in adjacent_list:
    #         for adjacent_rid in adjacent_list[str(rid)]:
    #             if adjacent_rid in rid_set:
    #                 min_road = min(rid, adjacent_rid)
    #                 max_road = max(rid, adjacent_rid)
    #                 edges.add((min_road, max_road))
    # G = nx.Graph()
    # G.add_nodes_from(nodes)
    # G.add_edges_from(list(edges))
    # for sub_graph in nx.connected_components(G):
    #     assert new_region_id not in region2rid_true
    #     region2rid_true[new_region_id] = list(sub_graph)
    #     for rid in sub_graph:
    #         assert rid not in rid2region_true
    #         rid2region_true[rid] = new_region_id
    #     new_region_id += 1
    # 使用有向图的强连通分量
    G = {}
    for rid in rid_set:
        rid_adjacent_road_list = []
        if str(rid) in adjacent_list:
            for adjacent_road in adjacent_list[str(rid)]:
                if adjacent_road in rid_set:
                    rid_adjacent_road_list.append(adjacent_road)
        G[rid] = rid_adjacent_road_list
    strongly_connected_g = tarjan.tarjan(G)
    for sub_graph in strongly_connected_g:
        assert new_region_id not in region2rid_true
        region2rid_true[new_region_id] = sub_graph
        for rid in sub_graph:
            assert rid not in rid2region_true
            rid2region_true[rid] = new_region_id
        new_region_id += 1

assert len(rid2region_true) == len(rid2region)
print('total strong connected region {}'.format(len(region2rid_true)))
# 对聚类结果进行个简单统计
region_rid_cnt = []
for key in region2rid_true:
    values = region2rid_true[key]
    region_rid_cnt.append(len(values))

# 画一个直方图
# df = pd.DataFrame()
# df['region_road_cnt'] = region_rid_cnt
# sns.histplot(df, x='region_road_cnt', bins=100)
# plt.show()

# 尝试对小的区域进行一个合并，如果两个邻近区域合并后，能够成为一个较大的连通区域，则进行合并
# 目的是减小区域的数目
# 将过小的区域和很大的区域丢在一起，再用强连通分量划分一遍
# merge_rid_set = []
# merge_region_cnt = 0
# for region in region2rid_true:
#     region_rid_set = region2rid_true[region]
#     if len(region_rid_set) <= 30:
#         for rid in region_rid_set:
#             merge_rid_set.append(rid)
#         merge_region_cnt += 1
#
# # 计算强连通分量
# G = {}
# for rid in merge_rid_set:
#     rid_adjacent_road_list = []
#     if str(rid) in adjacent_list:
#         for adjacent_road in adjacent_list[str(rid)]:
#             if adjacent_road in merge_rid_set:
#                 rid_adjacent_road_list.append(adjacent_road)
#     G[rid] = rid_adjacent_road_list
#
# strongly_connected_g = tarjan.tarjan(G)
# print(merge_region_cnt)
# print(len(strongly_connected_g))
# print(len(region2rid_true) - merge_region_cnt + len(strongly_connected_g))
# # 根据划分后的区域，重新生成 region2rid 以及 rid2region
# region2rid_tarjan = {}
# rid2region_tarjan = {}
# new_region_id = 0
# for region in region2rid_true:
#     region_rid_set = region2rid_true[region]
#     if 30 < len(region_rid_set):
#         # 没有被切割
#         assert new_region_id not in region2rid_tarjan
#         region2rid_tarjan[new_region_id] = region_rid_set
#         for rid in region_rid_set:
#             assert rid not in rid2region_tarjan
#             rid2region_tarjan[rid] = new_region_id
#         new_region_id += 1
#
# for sub_graph in strongly_connected_g:
#     assert new_region_id not in region2rid_tarjan
#     region2rid_tarjan[new_region_id] = sub_graph
#     for rid in sub_graph:
#         assert rid not in rid2region_tarjan
#         rid2region_tarjan[rid] = new_region_id
#     new_region_id += 1
#
# # 对聚类结果进行个简单统计
# region_rid_cnt = []
# for key in region2rid_tarjan:
#     values = region2rid_tarjan[key]
#     region_rid_cnt.append(len(values))
#
# df = pd.DataFrame()
# df['region_road_cnt'] = region_rid_cnt
# sns.histplot(df, x='region_road_cnt', bins=100)
# plt.show()


with open('../data/Xian/region2rid.json', 'w') as f:
    json.dump(region2rid_true, f)
with open('../data/Xian/rid2region.json', 'w') as f:
    json.dump(rid2region_true, f)

region_rid_cnt = np.array(region_rid_cnt)
print('avg rid per region: {}, max region: {}, min region: {}, total region: {}'.format(
    np.average(region_rid_cnt), np.max(region_rid_cnt), np.min(region_rid_cnt), region_rid_cnt.shape[0]))


