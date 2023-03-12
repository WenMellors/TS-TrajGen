import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
# 根据路网连通性，构建交通区域的连通性（邻接矩阵）

# 读取路段邻接表
rid_rel = pd.read_csv('../data/Xian/xian.rel')
rid_adjacent_list = {}
for index, row in tqdm(rid_rel.iterrows(), total=rid_rel.shape[0], desc='cal road adjacent list'):
    f_rid = str(row['origin_id'])
    t_rid = row['destination_id']
    if f_rid not in rid_adjacent_list:
        rid_adjacent_list[f_rid] = [t_rid]
    else:
        rid_adjacent_list[f_rid].append(t_rid)
with open('../data/Xian/adjacent_list.json', 'w') as f:
    json.dump(rid_adjacent_list, f)
# 读取路段与区域之间的映射关系
with open('../data/Xian/rid2region.json', 'r') as f:
    rid2region = json.load(f)

with open('../data/Xian/region2rid.json', 'r') as f:
    region2rid = json.load(f)

region_adjacent_list = {}
"""
区域的邻接表如下构建:
    当前区域: {
        下游区域: 边界路段集合（这个边界路段是下游区域中的路段）
    }
"""
# 使用稀疏矩阵构建邻接矩阵
region_adj_row = []
region_adj_col = []
region_adj_data = []

for region in tqdm(region2rid, desc="cal region adjacent"):
    # region 是 str
    # 遍历该区域所包含的路段，这些路段的可达路段所属的区域即为可达区域
    next_region_dict = {}
    rid_set = region2rid[region]
    for rid in rid_set:
        # rid 是 int
        if str(rid) in rid_adjacent_list:
            for next_rid in rid_adjacent_list[str(rid)]:
                # next_rid 是 int
                # 查找下游路段所属的
                next_region = rid2region[str(next_rid)]
                if int(region) != next_region:
                    # next_region 是当前区域的下游区域
                    if next_region not in next_region_dict:
                        next_region_dict[next_region] = set()
                        next_region_dict[next_region].add(next_rid)
                        # 将边加入稀疏邻接矩阵中
                        region_adj_row.append(int(region))
                        region_adj_col.append(next_region)
                        region_adj_data.append(1.0)
                    else:
                        next_region_dict[next_region].add(next_rid)
                        # 无需加入稀疏矩阵中
    # 将 set 转换为 list
    for next_region in next_region_dict:
        rid_set = next_region_dict[next_region]
        next_region_dict[next_region] = list(rid_set)
    region_adjacent_list[region] = next_region_dict

total_region = len(region2rid)
region_adj_mx = sp.coo_matrix((region_adj_data, (region_adj_row, region_adj_col)),
                              shape=(total_region, total_region))
# 保存生成结果
sp.save_npz("../data/Xian/region_adj_mx", region_adj_mx)
with open('../data/Xian/region_adjacent_list.json', 'w') as f:
    json.dump(region_adjacent_list, f)

# 进行一些简单的统计
adjacent_cnt = []
border_rid_cnt = []
for region in region_adjacent_list:
    adjacent_cnt.append(len(region_adjacent_list[region]))
    for next_region in region_adjacent_list[region]:
        border_rid_cnt.append(len(region_adjacent_list[region][next_region]))
adjacent_cnt = np.array(adjacent_cnt)
border_rid_cnt = np.array(border_rid_cnt)
print('region adjacent avg: {}, max {}, min {}'.format(np.average(adjacent_cnt), np.max(adjacent_cnt),
                                                       np.min(adjacent_cnt)))
print('border road avg: {}, max {}, min {}'.format(np.average(border_rid_cnt), np.max(border_rid_cnt),
                                                   np.min(border_rid_cnt)))
