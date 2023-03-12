# region_feature 由预训练好的路段 node_feature 聚类得到

import os
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import torch
from generator.distance_gat_fc import DistanceGatFC
import json
from loss import mask_mape_loss
import numpy as np
from utils.map_manager import MapManager

# 训练参数
dataset_name = 'Xian'
device = 'cuda:0'
batch_size = 128
config = {
    'embed_dim': 128,
    'gps_emb_dim': 5,
    'num_of_heads': 4,
    'concat': False,
    'device': device,
    'distance_mode': 'l2'
}
train_rate = 0.6
eval_rate = 0.2
max_epoch = 50
learning_rate = 0.0005
weight_decay = 0.001
lr_patience = 2
lr_decay_ratio = 0.1
early_stop_lr = 1e-6

save_folder = './save/Xian/'
save_file_name = 'gat_fc.pt'
temp_folder = './temp/gat/'
train = True
debug = False

# 加载 rel
road_num = 17378
adjacent_np_file = './data/Xian/adjacent_mx.npz'
adj_mx = sp.load_npz(adjacent_np_file)
# 加载 node_feature
node_feature_file = './data/Xian/node_feature.pt'
node_features = torch.load(node_feature_file).to(device)
map_manager = MapManager(dataset_name=dataset_name)
data_feature = {
    'adj_mx': adj_mx,
    'node_features': node_features,
    'img_width': map_manager.img_width,
    'img_height': map_manager.img_height
}
# 加载模型
road_gat = DistanceGatFC(config=config, data_feature=data_feature).to(device)
road_gat.load_state_dict(torch.load(os.path.join(save_folder, save_file_name), map_location=device))

road_gat._setup_node_emb()
# (road_num, feature_dim)
node_emb_feature = road_gat.node_emb_feature
# 构建 road 与 region 的映射矩阵
with open('./data/Xian/rid2region.json', 'r') as f:
    rid2region = json.load(f)
with open('./data/Xian/region2rid.json', 'r') as f:
    region2rid = json.load(f)
region_num = len(region2rid)
# (region_num, road_num)
region2rid_mat = np.zeros((region_num, road_num))
for rid in tqdm(rid2region):
    region = rid2region[rid]
    region2rid_mat[region][int(rid)] = 1.0

region2rid_mat = torch.FloatTensor(region2rid_mat).to(device)
region_feature = torch.matmul(region2rid_mat, node_emb_feature)
# 保存 region 的 feature
torch.save(region_feature, './data/Xian/region_feature.pt')

