# 根据轨迹统计各邻接区域之间边界路段作为转移道路的频率作为选择概率
import json
import pandas as pd
from tqdm import tqdm

region_transfer_cnt = {}

with open('../data/Xian/rid2region.json', 'r') as f:
    rid2region = json.load(f)

mm_traj = pd.read_csv('../data/Xian/xianshi_partA_traj_mm_processed.csv')

for index, row in tqdm(mm_traj.iterrows(), total=mm_traj.shape[0]):
    rid_list = row['rid_list'].split(',')
    prev_region = rid2region[rid_list[0]]
    for j, rid in enumerate(rid_list[1:]):
        now_region = rid2region[rid]
        if prev_region != now_region:
            # 发生了区域之间的迁移，迁移路段为 rid
            if prev_region not in region_transfer_cnt:
                region_transfer_cnt[prev_region] = {now_region: {rid: 1}}
            elif now_region not in region_transfer_cnt[prev_region]:
                region_transfer_cnt[prev_region][now_region] = {rid: 1}
            elif rid not in region_transfer_cnt[prev_region][now_region]:
                region_transfer_cnt[prev_region][now_region][rid] = 1
            else:
                region_transfer_cnt[prev_region][now_region][rid] += 1
            # update 一下区域
            prev_region = now_region

# 预处理一下结果
final_result = {}
for region_f in region_transfer_cnt:
    final_result[region_f] = {}
    for region_t in region_transfer_cnt[region_f]:
        border_rid_set = []
        border_rid_cnt = []
        rid_cnt = region_transfer_cnt[region_f][region_t]
        for rid in rid_cnt:
            border_rid_set.append(int(rid))
            border_rid_cnt.append(rid_cnt[rid])
        final_result[region_f][region_t] = {'transfer_rid': border_rid_set, 'transfer_freq': border_rid_cnt}

with open('../data/Xian/region_adjacent_list.json', 'r') as f:
    region_adjacent_list = json.load(f)

f_not_exist = 0
cnt = 0
# 需要检查是否存在邻接的区域之间没有转移的
for region_f in region_adjacent_list:
    region_f = eval(region_f)
    if region_f not in final_result:
        # 应该很少
        final_result[region_f] = {}
        f_not_exist += 1
    for region_t in region_adjacent_list[str(region_f)]:
        region_t = eval(region_t)
        if region_t not in final_result[region_f]:
            # 给均等的频次
            border_rid_set = region_adjacent_list[str(region_f)][str(region_t)]
            border_rid_cnt = [1 for rid in border_rid_set]
            final_result[region_f][region_t] = {'transfer_rid': border_rid_set, 'transfer_freq': border_rid_cnt}
            cnt += 1

print('f_region not exist cnt is ', f_not_exist)
print('transfer not exist cnt is ', cnt)

# 保存统计结果
with open('../data/Xian/region_transfer_prob.json', 'w') as f:
    json.dump(final_result, f)
