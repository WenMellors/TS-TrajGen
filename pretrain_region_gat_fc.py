import os
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import torch
from generator.distance_gat_fc import DistanceGatFC
from torch.utils.data import DataLoader
from utils.ListDataset import ListDataset
from utils.utils import get_logger
import json
import numpy as np
from utils.parser import str2bool
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--local', type=str2bool, default=False)
parser.add_argument('--dataset_name', type=str, default='BJ_Taxi')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--debug', type=str2bool, default=False)
args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name
device = args.device
debug = args.debug

archive_data_folder = 'TS_TrajGen_data_archive'

if local:
    data_root = './data/'
else:
    data_root = '/mnt/data/jwj/TS_TrajGen_data_archive/'

# 训练参数
batch_size = 32
if dataset_name == 'BJ_Taxi' or dataset_name == 'Porto_Taxi':
    config = {
        'embed_dim': 128,
        'gps_emb_dim': 5,
        'num_of_heads': 5,
        'concat': False,
        'device': device,
        'distance_mode': 'l2',
        'no_gps_emb': True
    }
else:
    # Xian
    config = {
        'embed_dim': 68,
        'gps_emb_dim': 5,
        'num_of_heads': 4,
        'concat': False,
        'device': device,
        'distance_mode': 'l2',
        'no_gps_emb': True
    }
max_epoch = 50
learning_rate = 0.0005
weight_decay = 0.0001
lr_patience = 2
lr_decay_ratio = 0.01
early_stop_lr = 1e-6

save_folder = './save/{}'.format(dataset_name)
save_file_name = 'region_gat_fc.pt'
temp_folder = './temp/{}/gat/'.format(dataset_name)
train = True

logger = get_logger(name='RegionGatDis')
logger.info('read data')
with open(os.path.join(data_root, dataset_name, 'region2rid.json'), 'r') as f:
    region2rid = json.load(f)
# 数据集的大小
road_num = len(region2rid)
road_num_with_pad = road_num + 1
adjacent_np_file = os.path.join(data_root, dataset_name, 'region_adj_mx.npz')

adj_mx = sp.load_npz(adjacent_np_file)

# 加载区域 region_feature
node_feature_file = os.path.join(data_root, dataset_name, 'region_feature.pt')
node_features = torch.load(node_feature_file, map_location='cpu').to(device)

data_feature = {
    'adj_mx': adj_mx,
    'node_features': node_features
}

# 加载模型
gat = DistanceGatFC(config=config, data_feature=data_feature).to(device)
logger.info('init gat')
logger.info(gat)
optimizer = torch.optim.Adam(gat.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=lr_patience, factor=lr_decay_ratio)
# 加载训练数据
# 读取训练输入数据
if dataset_name == 'BJ_Taxi':
    train_data = pd.read_csv('./data/201511_region_pretrain_input_train.csv')
    eval_data = pd.read_csv('./data/201511_region_pretrain_input_eval.csv')
    test_data = pd.read_csv('./data/201511_region_pretrain_input_test.csv')
else:
    # Xian
    train_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_region_pretrain_input_train.csv'))
    eval_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_region_pretrain_input_eval.csv'))
    test_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_region_pretrain_input_test.csv'))

train_data = train_data.values.tolist()
eval_data = eval_data.values.tolist()
test_data = test_data.values.tolist()

train_num = len(train_data)
eval_num = len(eval_data)
test_num = len(test_data)
total_data = train_num + eval_num + test_num
logger.info('total input record is {}. train set: {}, val set {}, test set {}'.format(total_data, train_num,
                                                                                      eval_num, test_num))

train_dataset = ListDataset(train_data)
eval_dataset = ListDataset(eval_data)
test_dataset = ListDataset(test_data)

region_dist = np.load(os.path.join(data_root, dataset_name, 'region_count_dist.npy'))


# 自定义收集函数
def collate_fn(indices):
    batch_des = []
    batch_candidate_set = []
    batch_candidate_dis = []
    batch_target = []
    candidate_set_len = []
    for item in indices:
        batch_des.append(item[2])
        candidate_set = [int(i) for i in item[3].split(',')]
        # 获取每个候选区域与目标区域的距离
        candidate_dis = []
        for c in candidate_set:
            dis = region_dist[c][item[2]]
            if dis == -1:
                # 不可能被选中的
                dis = 100000
            candidate_dis.append(dis/100)  # 转化为百米
        batch_candidate_set.append(candidate_set)
        batch_candidate_dis.append(candidate_dis)
        batch_target.append(item[5])
        candidate_set_len.append(len(candidate_set))
    # 补齐
    max_candidate_size = max(candidate_set_len)
    for i in range(len(batch_des)):
        # 对于候选集，选择非下一跳的点进行补齐
        while len(batch_candidate_set[i]) < max_candidate_size:
            # 因为我们已经干掉了 candidate_set len 为 1 的点了
            assert len(batch_candidate_set[i]) != 1, 'candidate set is 1!'
            pad_index = np.random.randint(len(batch_candidate_set[i]))
            if pad_index != batch_target[i]:
                batch_candidate_set[i].append(batch_candidate_set[i][pad_index])
                batch_candidate_dis[i].append(batch_candidate_dis[i][pad_index])
    return [torch.LongTensor(batch_des).to(device), torch.LongTensor(batch_candidate_set).to(device),
            torch.FloatTensor(batch_candidate_dis).to(device), torch.LongTensor(batch_target).to(device)]


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)


if train:
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    metrics = []
    for epoch in range(max_epoch):
        # train
        logger.info('start train epoch {}'.format(epoch))
        gat.train(True)
        train_loss = 0
        for des, candidate_set, candidate_distance, target in tqdm(train_loader, desc='train model'):
            optimizer.zero_grad()
            loss = gat.calculate_loss(candidate_set=candidate_set, candidate_distance=candidate_distance, des=des,
                                      target=target)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        # val
        gat.train(False)
        val_hit = 0
        for des, candidate_set, candidate_distance, target in tqdm(val_loader, desc='val model'):
            with torch.no_grad():
                candidate_score = gat.predict(candidate_set=candidate_set, des=des, candidate_distance=candidate_distance)
            target = target.tolist()
            val, index = torch.topk(candidate_score, 1, dim=1)
            for i, p in enumerate(index):
                if target[i] in p:
                    val_hit += 1
        val_ac = val_hit / eval_num
        metrics.append(val_ac)
        lr_scheduler.step(val_ac)
        # store temp model
        torch.save(gat.state_dict(), os.path.join(temp_folder, 'region_gat_{}.pt'.format(epoch)))
        lr = optimizer.param_groups[0]['lr']
        logger.info('==> Train Epoch {}: Train Loss {:.6f}, val ac {}, lr {}'.format(epoch, train_loss, val_ac, lr))
        if lr < early_stop_lr:
            logger.info('early stop')
            break
    # load best epoch
    best_epoch = np.argmin(metrics)
    load_temp_file = 'region_gat_{}.pt'.format(best_epoch)
    logger.info('load best from {}'.format(best_epoch))
    gat.load_state_dict(torch.load(os.path.join(temp_folder, load_temp_file)))
else:
    gat.load_state_dict(torch.load(os.path.join(save_folder, save_file_name), map_location=device))
# 开始评估
gat.train(False)
test_hit = 0
for des, candidate_set, candidate_distance, target in tqdm(test_loader, desc='test model'):
    with torch.no_grad():
        candidate_score = gat.predict(candidate_set=candidate_set, des=des, candidate_distance=candidate_distance)
    target = target.tolist()
    val, index = torch.topk(candidate_score, 1, dim=1)
    for i, p in enumerate(index):
        if target[i] in p:
            test_hit += 1
test_ac = test_hit / test_num
logger.info('==> Test Result: test ac {}'.format(test_ac))
# 保存模型
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
torch.save(gat.state_dict(), os.path.join(save_folder, save_file_name))
# 删除 temp 文件
for rt, dirs, files in os.walk(temp_folder):
    for name in files:
        remove_path = os.path.join(rt, name)
        os.remove(remove_path)
