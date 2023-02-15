import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.data_util import encode_time
from utils.parser import str2bool
from search import DoubleLayerSearcher
import json
from generator.generator_v4 import GeneratorV4
import torch
import scipy.sparse as sp
import argparse
import os
import math

parser = argparse.ArgumentParser()
parser.add_argument('--local', type=str2bool, default=False)
parser.add_argument('--dataset_name', type=str, default='BJ_Taxi')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name
device = args.device

archive_data_folder = 'TS_TrajGen_data_archive'

if local:
    data_root = './data/'
else:
    data_root = '/mnt/data/jwj/'

if dataset_name == 'BJ_Taxi':
    # 这里我们进行修改，配合后续新工作的实验
    true_traj = pd.read_csv(os.path.join(data_root, dataset_name, 'chaoyang_traj_mm_test.csv'))
    pretrain_gen_file = './save/BJ_Taxi/function_g_fc.pt'
    pretrain_gat_file = './save/BJ_Taxi/gat_fc.pt'
    ganerate_trace_file = os.path.join(data_root, dataset_name, 'TS_TrajGen_chaoyang_generate.csv')
    pretrain_region_gen_file = './save/BJ_Taxi/region_function_g_fc.pt'
    pretrain_region_gat_file = './save/BJ_Taxi/region_gat_fc.pt'
else:
    # Porto_Taxi
    true_traj = pd.read_csv(os.path.join(data_root, dataset_name, 'porto_mm_test.csv'))
    pretrain_gen_file = './save/Porto_Taxi/function_g_fc.pt'
    pretrain_gat_file = './save/Porto_Taxi/gat_fc.pt'
    ganerate_trace_file = os.path.join(data_root, dataset_name, 'TS_TrajGen_generate.csv')
    pretrain_region_gen_file = './save/Porto_Taxi/region_function_g_fc.pt'
    pretrain_region_gat_file = './save/Porto_Taxi/region_gat_fc.pt'

gen_config = {
    "function_g": {
        "road_emb_size": 256,
        "time_emb_size": 50,
        "hidden_size": 256,
        "dropout_p": 0.6,
        "lstm_layer_num": 2,
        "pretrain_road_rep": None,
        "dis_weight": 0.5,
        "device": device
    },
    "function_h": {  # 0.7937
        'embed_dim': 256,
        'gps_emb_dim': 10,
        'num_of_heads': 5,
        'concat': False,
        'device': device,
        'distance_mode': 'l2'
    },
    'dis_weight': 0.45
}

region_gen_config = {
    "function_g": {
        "road_emb_size": 128,
        "time_emb_size": 32,
        "hidden_size": 128,
        "dropout_p": 0.6,
        "lstm_layer_num": 2,
        "pretrain_road_rep": None,
        "dis_weight": 0.5,
        "device": device
    },
    "function_h": {
        'embed_dim': 128,
        'gps_emb_dim': 5,
        'num_of_heads': 5,
        'concat': False,
        'device': device,
        'distance_mode': 'l2',
        'no_gps_emb': True
    },
    'dis_weight': 0.45
}
if dataset_name == 'BJ_Taxi':
    # 加载道路级别 node_feature
    node_feature_file = os.path.join(data_root, archive_data_folder, 'node_feature.pt') #  './data/node_feature.pt'
    node_features = torch.load(node_feature_file).to(device)
    adjacent_np_file = os.path.join(data_root, archive_data_folder, 'adjacent_mx.npz')  # './data/adjacent_mx.npz'
    adj_mx = sp.load_npz(adjacent_np_file)
    # 加载区域级别 region_feature
    region_adjacent_np_file = os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region_adj_mx.npz')
    region_adj_mx = sp.load_npz(region_adjacent_np_file)
    region_feature_file = os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region_feature.pt')
    region_features = torch.load(region_feature_file, map_location=device)
    # 数据集的大小
    road_num = 40306
    time_size = 2880
    loc_pad = road_num
    time_pad = time_size
    lon_range = 0.2507  # 地图经度的跨度
    lat_range = 0.21  # 地图纬度的跨度
    img_unit = 0.005  # 这样画出来的大小，大概是 0.42 km * 0.55 km 的格子
    lon_0 = 116.25
    lat_0 = 39.79  # 地图最左下角的坐标（即原点坐标）
    img_width = math.ceil(lon_range / img_unit) + 1  # 图像的宽度
    img_height = math.ceil(lat_range / img_unit) + 1  # 映射出的图像的高度
    data_feature = {
        'road_num': road_num + 1,
        'time_size': time_size + 1,
        'road_pad': loc_pad,
        'time_pad': time_pad,
        'adj_mx': adj_mx,
        'node_features': node_features,
        'img_width': img_width,
        'img_height': img_height
    }
    with open(os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region2rid.json'), 'r') as f:
        region2rid = json.load(f)
    region_num = len(region2rid)
    region_img_unit = 0.001
    # region_img_width = math.ceil(lon_range / region_img_unit) + 1  # 图像的宽度
    # region_img_height = math.ceil(lat_range / region_img_unit) + 1  # 映射出的图像的高度
    region_data_feature = {
        'road_num': region_num + 1,
        'time_size': time_size + 1,
        'road_pad': region_num,
        'time_pad': time_pad,
        'adj_mx': region_adj_mx,
        'node_features': region_features,
        'img_width': 252,
        'img_height': 211
    }
    # 读取路网邻接表
    with open(os.path.join(data_root, archive_data_folder, 'adjacent_list.json'), 'r') as f:
        adjacent_list = json.load(f)
    # 读取路网 GPS
    with open(os.path.join(data_root, archive_data_folder, 'rid_gps.json'), 'r') as f:
        rid_gps = json.load(f)
    # 读取路段长度信息
    with open(os.path.join(data_root, archive_data_folder, 'road_length.json'), 'r') as f:
        road_length = json.load(f)
    # 区域相关信息
    with open(os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region_adjacent_list.json'), 'r') as f:
        region_adjacent_list = json.load(f)
    region_dist = np.load(os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region_dist.npy'))
    with open(os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_region_transfer_prob.json'), 'r') as f:
        region_transfer_freq = json.load(f)
    with open(os.path.join(data_root, archive_data_folder, 'kaffpa_tarjan_rid2region.json'), 'r') as f:
        rid2region = json.load(f)
    road_time_distribution = np.load(os.path.join(data_root, archive_data_folder, 'road_time_distribution.npy'))
    region_time_distribution = np.load(os.path.join(data_root, archive_data_folder,
                                                    'kaffpa_tarjan_region_time_distribution.npy'))
else:
    # Porto_Taxi
    # 加载道路级别 node_feature
    node_feature_file = os.path.join(data_root, archive_data_folder, 'porto_node_feature.pt')
    node_features = torch.load(node_feature_file).to(device)
    adjacent_np_file = os.path.join(data_root, archive_data_folder, 'porto_adjacent_mx.npz')
    adj_mx = sp.load_npz(adjacent_np_file)
    # 加载区域级别 region_feature
    region_adjacent_np_file = os.path.join(data_root, archive_data_folder, 'porto_region_adj_mx.npz')
    region_adj_mx = sp.load_npz(region_adjacent_np_file)
    region_feature_file = os.path.join(data_root, archive_data_folder, 'porto_region_feature.pt')
    region_features = torch.load(region_feature_file, map_location=device)
    # 数据集的大小
    road_num = 11095
    time_size = 2880
    loc_pad = road_num
    time_pad = time_size
    lon_range = 0.133
    lat_range = 0.046
    img_unit = 0.005
    lon_0 = -8.6887
    lat_0 = 41.1405
    img_width = math.ceil(lon_range / img_unit) + 1
    img_height = math.ceil(lat_range / img_unit) + 1
    data_feature = {
        'road_num': road_num + 1,
        'time_size': time_size + 1,
        'road_pad': loc_pad,
        'time_pad': time_pad,
        'adj_mx': adj_mx,
        'node_features': node_features,
        'img_height': img_height,
        'img_width': img_width
    }
    with open(os.path.join(data_root, archive_data_folder, 'porto_region2rid.json'), 'r') as f:
        region2rid = json.load(f)
    region_num = len(region2rid)
    region_img_unit = 0.001
    region_img_width = math.ceil(lon_range / region_img_unit) + 1  # 图像的宽度
    region_img_height = math.ceil(lat_range / region_img_unit) + 1  # 映射出的图像的高度
    region_data_feature = {
        'road_num': region_num + 1,
        'time_size': time_size + 1,
        'road_pad': region_num,
        'time_pad': time_pad,
        'adj_mx': region_adj_mx,
        'node_features': region_features,
        'img_width': region_img_width,
        'img_height': region_img_height
    }

    # 读取路网邻接表
    with open(os.path.join(data_root, archive_data_folder, 'porto_adjacent_list.json'), 'r') as f:
        adjacent_list = json.load(f)
    # 读取路网 GPS
    with open(os.path.join(data_root, archive_data_folder, 'porto_rid_gps.json'), 'r') as f:
        rid_gps = json.load(f)
    # 读取路段长度信息
    with open(os.path.join(data_root, archive_data_folder, 'porto_road_length.json'), 'r') as f:
        road_length = json.load(f)
    # 区域相关信息
    with open(os.path.join(data_root, archive_data_folder, 'porto_region_adjacent_list.json'), 'r') as f:
        region_adjacent_list = json.load(f)
    region_dist = np.load(os.path.join(data_root, archive_data_folder, 'porto_region_dist.npy'))
    with open(os.path.join(data_root, archive_data_folder, 'porto_region_transfer_prob.json'), 'r') as f:
        region_transfer_freq = json.load(f)
    with open(os.path.join(data_root, archive_data_folder, 'porto_rid2region.json'), 'r') as f:
        rid2region = json.load(f)
    road_time_distribution = np.load(os.path.join(data_root, archive_data_folder, 'porto_road_time_distribution.npy'))
    region_time_distribution = np.load(os.path.join(data_root, archive_data_folder,
                                                    'porto_region_time_distribution.npy'))

# 初始化生成器
road_generator = GeneratorV4(config=gen_config, data_feature=data_feature).to(device)
road_generatorv1_state = torch.load(pretrain_gen_file, map_location=device)
road_generator.function_g.load_state_dict(road_generatorv1_state)
road_gat_state = torch.load(pretrain_gat_file, map_location=device)
road_generator.function_h.load_state_dict(road_gat_state)
road_generator.train(False)

region_generator = GeneratorV4(config=region_gen_config, data_feature=region_data_feature).to(device)
region_generatorv1_state = torch.load(pretrain_region_gen_file, map_location=device)
region_generator.function_g.load_state_dict(region_generatorv1_state)
region_gat_state = torch.load(pretrain_region_gat_file, map_location=device)
region_generator.function_h.load_state_dict(region_gat_state)
region_generator.train(False)

searcher = DoubleLayerSearcher(device=device, adjacent_list=adjacent_list, road_center_gps=rid_gps, road_length=road_length,
                               region_adjacent_list=region_adjacent_list, region_dist=region_dist, region_transfer_freq=region_transfer_freq,
                               rid2region=rid2region, road_time_distribution=road_time_distribution,
                               region_time_distribution=region_time_distribution, region2rid=region2rid)
# 对每条轨迹都进行一个生成，并将生成结果保存至本地
f = open(ganerate_trace_file, 'w')
f.write("traj_id,rid_list,time_list\n")
fail_cnt = 0
region_astar_fail_cnt = 0
for index, row in tqdm(true_traj.iterrows(), total=true_traj.shape[0]):
    rid_list = [int(i) for i in row['rid_list'].split(',')]
    mm_id = row['traj_id']
    time_list = list(map(encode_time, row['time_list'].split(',')))
    with torch.no_grad():
        gen_trace_loc, gen_trace_tim, is_astar = searcher.astar_search(region_model=region_generator,
                                                                       road_model=road_generator,
                                                                       start_rid=rid_list[0], start_tim=time_list[0],
                                                                       des=rid_list[-1],
                                                                       default_len=len(rid_list), max_step=5000)
    f.write('{},\"{}\",\"{}\"\n'.format(str(mm_id), ','.join([str(rid) for rid in gen_trace_loc]),
                                        ','.join([str(time) for time in gen_trace_tim])))
    if gen_trace_loc[-1] != rid_list[-1]:
        fail_cnt += 1
    if is_astar == 0:
        region_astar_fail_cnt += 1

print('fail cnt ', fail_cnt)
print('region astar fail cnt ', region_astar_fail_cnt)
f.close()
searcher.save_fail_log()

