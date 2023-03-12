# 预训练生成器v1
from generator.function_g_fc import FunctionGFC
import pandas as pd
from utils.ListDataset import ListDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from utils.util import get_logger
from tqdm import tqdm
import json
import argparse
from utils.parser import str2bool


parser = argparse.ArgumentParser()
parser.add_argument('--local', type=str2bool, default=False)
parser.add_argument('--dataset_name', type=str, default='BJ_Taxi')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name
device = args.device

if local:
    data_root = './data/'
else:
    data_root = '/mnt/data/jwj/TS_TrajGen_data_archive/'

# 训练相关参数
max_epoch = 25
batch_size = 32
learning_rate = 0.001
weight_decay = 0.00001
lr_patience = 2
lr_decay_ratio = 0.001
save_folder = './save/{}'.format(dataset_name)
save_file_name = 'region_function_g_fc.pt'
temp_folder = './temp/{}/gan/'.format(dataset_name)
early_stop_lr = 1e-6
train = True
with open(os.path.join(data_root, dataset_name, 'region2rid.json'), 'r') as f:
    region2rid = json.load(f)
# 数据集的大小
road_num = len(region2rid)
time_size = 2880
loc_pad = road_num
time_pad = time_size
data_feature = {
    'road_num': road_num + 1,
    'time_size': time_size + 1,
    'road_pad': loc_pad,
    'time_pad': time_pad
}
if dataset_name == 'BJ_Taxi' or dataset_name == 'Porto_Taxi':
    # 生成器 config
    gen_config = {
        "road_emb_size": 128,  # 这里下调一下网络参数，因为区域数目比较少
        "time_emb_size": 32,
        "hidden_size": 128,
        "dropout_p": 0.6,
        "lstm_layer_num": 2,
        "pretrain_road_rep": None,
        "dis_weight": 0.5,
        "device": device
    }
else:
    gen_config = {
        "road_emb_size": 64,  # 就 1 千多个
        "time_emb_size": 16,
        "hidden_size": 64,
        "dropout_p": 0.6,
        "lstm_layer_num": 2,
        "pretrain_road_rep": None,
        "dis_weight": 0.5,
        "device": device
    }

logger = get_logger(name='RegionGeneratorv1')
logger.info('read data')
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


# 自定义收集函数
def collate_fn(indices):
    batch_trace_loc = []
    batch_trace_time = []
    batch_des = []
    batch_candidate_set = []
    batch_candidate_dis = []
    batch_target = []
    trace_loc_len = []
    candidate_set_len = []
    for item in indices:
        trace_loc = [int(i) for i in item[0].split(',')]
        trace_time = [int(i) for i in item[1].split(',')]
        batch_des.append(item[2])
        candidate_set = [int(i) for i in item[3].split(',')]
        candidate_dis = [float(i) for i in item[4].split(',')]
        batch_trace_loc.append(trace_loc)
        batch_trace_time.append(trace_time)
        batch_candidate_set.append(candidate_set)
        batch_candidate_dis.append(candidate_dis)
        batch_target.append(item[5])
        trace_loc_len.append(len(trace_loc))
        candidate_set_len.append(len(candidate_set))
    # 补齐
    max_trace_len = max(trace_loc_len)
    max_candidate_size = max(candidate_set_len)
    for i in range(len(batch_trace_loc)):
        pad_len = max_trace_len - len(batch_trace_loc[i])
        batch_trace_loc[i] += [loc_pad] * pad_len
        batch_trace_time[i] += [time_pad] * pad_len
        # 对于候选集，选择非下一跳的点进行补齐
        while len(batch_candidate_set[i]) < max_candidate_size:
            # 因为我们已经干掉了 candidate_set len 为 1 的点了
            assert len(batch_candidate_set[i]) != 1, 'candidate set is 1!'
            pad_index = np.random.randint(len(batch_candidate_set[i]))
            if pad_index != batch_target[i]:
                batch_candidate_set[i].append(batch_candidate_set[i][pad_index])
                batch_candidate_dis[i].append(batch_candidate_dis[i][pad_index])
    return [torch.LongTensor(batch_trace_loc).to(device), torch.LongTensor(batch_trace_time).to(device), torch.LongTensor(batch_des).to(device),
            torch.LongTensor(batch_candidate_set).to(device), torch.FloatTensor(batch_candidate_dis).to(device), torch.LongTensor(batch_target).to(device)]


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# 加载模型
gen_model = FunctionGFC(gen_config, data_feature).to(device)
logger.info('init genv1')
logger.info(gen_model)
optimizer = torch.optim.Adam(gen_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=lr_patience, factor=lr_decay_ratio)

# 开始训练
if train:
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    metrics = []
    for epoch in range(max_epoch):
        # train
        logger.info('start train epoch {}'.format(epoch))
        gen_model.train(True)
        train_loss = 0
        for trace_loc, trace_time, des, candidate_set, candidate_dis, target in tqdm(train_loader, desc='train model'):
            optimizer.zero_grad()
            trace_mask = ~(trace_loc == loc_pad)
            loss = gen_model.calculate_loss(trace_loc=trace_loc, trace_time=trace_time, des=des, candidate_set=candidate_set, candidate_dis=candidate_dis,
                                            target=target, trace_mask=trace_mask)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        # val
        val_hit = 0
        gen_model.train(False)
        for trace_loc, trace_time, des, candidate_set, candidate_dis, target in tqdm(val_loader, desc='val model'):
            trace_mask = ~(trace_loc == loc_pad)
            score = gen_model.predict_g(trace_loc=trace_loc, trace_time=trace_time, des=des, candidate_set=candidate_set, candidate_dis=candidate_dis, trace_mask=trace_mask)
            target = target.tolist()
            val, index = torch.topk(score, 1, dim=1)
            for i, p in enumerate(index):
                if target[i] in p:
                    val_hit += 1
        val_ac = val_hit / eval_num
        metrics.append(val_ac)
        lr_scheduler.step(val_ac)
        # store temp model
        torch.save(gen_model.state_dict(), os.path.join(temp_folder, 'region_function_g_fc_{}.pt'.format(epoch)))
        lr = optimizer.param_groups[0]['lr']
        logger.info('==> Train Epoch {}: Train Loss {:.6f}, val AC {:.6f}, lr {}'.format(epoch, train_loss, val_ac, lr))
        if lr < early_stop_lr:
            logger.info('early stop')
            break
    # load best epoch
    best_epoch = np.argmax(metrics)
    load_temp_file = 'region_function_g_fc_{}.pt'.format(best_epoch)
    logger.info('load best from {}'.format(best_epoch))
    gen_model.load_state_dict(torch.load(os.path.join(temp_folder, load_temp_file)))
else:
    gen_model.load_state_dict(torch.load(os.path.join(save_folder, save_file_name), map_location=device))
# 开始评估
test_hit = 0
gen_model.train(False)
for trace_loc, trace_time, des, candidate_set, candidate_dis, target in tqdm(test_loader, desc='test model'):
    trace_mask = ~(trace_loc == loc_pad)
    score = gen_model.predict_g(trace_loc=trace_loc, trace_time=trace_time, des=des, candidate_set=candidate_set, candidate_dis=candidate_dis, trace_mask=trace_mask)
    target = target.tolist()
    val, index = torch.topk(score, 1, dim=1)
    for i, p in enumerate(index):
        if target[i] in p:
            test_hit += 1
test_ac = test_hit / test_num
logger.info('==> Test Result: ac {:.6f}'.format(test_ac))
# 保存模型
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
torch.save(gen_model.state_dict(), os.path.join(save_folder, save_file_name))
# 删除 temp 文件
for rt, dirs, files in os.walk(temp_folder):
    for name in files:
        remove_path = os.path.join(rt, name)
        os.remove(remove_path)
