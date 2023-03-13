## 参考 SeqGAN 来强化学习生成器
import json
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split
from geopy import distance
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from utils.ListDataset import ListDataset
from torch.utils.data import DataLoader
from generator.generator_v4 import GeneratorV4
from discriminator.discriminator_v1 import DiscriminatorV1
from search import Searcher
from rollout import Rollout
from loss import gan_loss
from utils.util import get_logger
from utils.data_util import encode_time
from utils.evaluate_funcs import edit_distance, hausdorff_metric, dtw_metric


save_folder = './save/our_gan'
exp_id = 1  # 用于区分不同次调参的保存结果文件
pretrain_gan_file = './save/function_g_fc.pt'
pretrain_gat_file = './save/gat_4.pt'
trajectory_file = './data/201511_week1_short_traj.csv'
device = 'cuda:0'
learning_rate = 0.0005
weight_decay = 0.0001
lr_patience = 2
lr_decay_ratio = 0.1
dis_train_rate = 0.8
batch_size = 64
pretrain_discriminator = False
debug = False
clip = 5.0
if debug:
    total_epoch = 1
    pretrain_dis_epoch = 1
    dis_sample_num = 10
    gen_sample_num = 1
    rollout_times = 1
else:
    total_epoch = 20
    pretrain_dis_epoch = 5
    dis_sample_num = 5000
    gen_sample_num = 2000  # 生成器训练的时间复杂度很高
    rollout_times = 8
# 生成器 config
gen_config = {
    "function_g": {
        "road_emb_size": 256,  # 需要和路网表征预训练部分维度一致
        "time_emb_size": 50,
        "hidden_size": 256,
        "dropout_p": 0.6,
        "lstm_layer_num": 2,
        "pretrain_road_rep": None,
        "dis_weight": 0.5,
        "device": device
    },
    "function_h": {
        'embed_dim': 256,
        'gps_emb_dim': 10,
        'num_of_heads': 5,
        'concat': False,
        'device': device,
        'distance_mode': 'l2'
    },
    'dis_weight': 0.45
}
# 加载 node_feature
node_feature_file = './data/node_feature.pt'
node_features = torch.load(node_feature_file).to(device)
adjacent_np_file = './data/adjacent_mx.npz'
adj_mx = sp.load_npz(adjacent_np_file)
# 判别器 config
dis_config = {
    "road_emb_size": 256,  # 需要和路网表征预训练部分维度一致
    "hidden_size": 256,
    "dropout_p": 0.6,
    "lstm_layer_num": 2,
    "pretrain_road_rep": None,
    "device": device
}

# init logger
logger = get_logger()
# build road adjacent list

# 读取 od 历史轨迹
with open('./data/od_distinct_route.json', 'r') as f:
    od_distinct_route = json.load(f)
# 读取路网邻接表
with open('./data/adjacent_list.json', 'r') as f:
    adjacent_list = json.load(f)
# 读取路网 GPS
with open('./data/rid_gps.json', 'r') as f:
    rid_gps = json.load(f)
# 读取路网长度
with open('./data/road_length.json', 'r') as f:
    road_length = json.load(f)
road_time_distribution = np.load('./data/road_time_distribution.npy')
# 数据集的大小
road_num = 40306
time_size = 2880
loc_pad = road_num
time_pad = time_size
data_feature = {
    'road_num': road_num + 1,
    'time_size': time_size + 1,
    'road_pad': loc_pad,
    'time_pad': time_pad,
    'adj_mx': adj_mx,
    'node_features': node_features
}


def collate_fn(indices):
    """
    自定义 DataLoader 收集函数
    Args:
        indices: 一个 batch 的数据

    Returns:

    """
    trace_loc = []
    trace_tim = []
    label = []
    for i in indices:
        trace_loc.append(torch.tensor(i[0]))
        trace_tim.append(torch.tensor(i[1]))
        label.append(i[2])
    trace_loc = pad_sequence(trace_loc, batch_first=True, padding_value=loc_pad)
    trace_tim = pad_sequence(trace_tim, batch_first=True, padding_value=time_pad)
    label = torch.tensor(label)
    trace_mask = ~(trace_loc == loc_pad)
    return [trace_loc.to(device), trace_tim.to(device), label.to(device), trace_mask.to(device)]


# 组织返回判别器的训练数据
def generate_discriminator_data(pos, gen_model):
    """

    Args:
        pos (pandas.Dataframe): 正样本轨迹的 df 形式数据，包含三列 trace_loc trace_tim, trace_label
        gen_model (Generator): 生成器，用于生成样本

    Returns:
        train_dataloader (DataLoader): 返回组织好的训练数据
        eval_dataloader (DataLoader)
    """
    data = []
    for index, row in tqdm(pos.iterrows(), total=pos.shape[0], desc='generate discriminator data'):
        trace_loc = list(map(int, row['rid_list'].split(',')))
        trace_tim = list(map(encode_time, row['time_list'].split(',')))
        data.append([trace_loc, trace_tim, 1])
        neg_trace_loc, neg_trace_tim = searcher.road_random_sample(gen_model=generator, trace_loc=[trace_loc[0]],
                                                                   trace_tim=[trace_tim[0]], des=trace_loc[-1],
                                                                   default_len=len(trace_loc))
        data.append([neg_trace_loc, neg_trace_tim, 0])
    dataset = ListDataset(data)
    # 还是划分一个训练集和验证集吧
    train_num = int(len(dataset) * dis_train_rate)
    eval_num = len(dataset) - train_num
    train_dataset, eval_dataset = random_split(dataset, [train_num, eval_num])
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn), \
        DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def train_discriminator(max_epoch):
    """
    根据抽样的真实轨迹与生成的轨迹预训练判别器
    """
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator.train(False)
    discriminator.train(True)
    for epoch in range(max_epoch):
        # 每一轮随机抽取 dis_sample_num 条真实轨迹作为正样本，对应生成的 dis_sample_num 条轨迹作为负样本
        pos_sample_index = np.random.randint(0, total_trace, size=dis_sample_num)
        pos_sample = trace.iloc[pos_sample_index]
        train_loader, eval_loader = generate_discriminator_data(gen_model=generator, pos=pos_sample)
        train_total_loss = 0
        # 训练
        discriminator.train(True)
        for batch in tqdm(train_loader, desc='train discriminator'):
            dis_optimizer.zero_grad()
            score = discriminator.forward(trace_loc=batch[0], trace_time=batch[1],
                                          trace_mask=batch[3])
            loss = discriminator.loss_func(score, batch[2])
            loss.backward()
            train_total_loss += loss.item()
            dis_optimizer.step()
        # 验证
        discriminator.train(False)
        eval_hit = 0
        eval_total_cnt = len(eval_loader.dataset)
        for batch in tqdm(eval_loader, desc='eval discriminator'):
            score = discriminator.forward(trace_loc=batch[0], trace_time=batch[1],
                                          trace_mask=batch[3])
            truth = batch[2]
            val, index = torch.topk(score, 1, dim=1)
            for i, p in enumerate(index):
                if truth[i] in p:
                    eval_hit += 1
        avg_ac = eval_hit / eval_total_cnt
        logger.info('train discriminator epoch {}: loss {:.6f}, top1 ac {:.6f}'.format(epoch, train_total_loss, avg_ac))


def train_generator(stage):
    """
    对抗学习更新生成器

    Args:
        stage (int): 设置一个难度等级，先学短轨迹、再学中轨迹、再学长轨迹
    """
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 设置判别器为非训练模式
    discriminator.train(False)
    # 每一轮随机抽取 gen_sample_num 条真实轨迹作为正样本，对应生成的 gen_sample_num 条轨迹作为负样本
    pos_sample_index = np.random.randint(0, total_trace, size=gen_sample_num)
    pos_sample = trace.iloc[pos_sample_index]
    total_edit_distance = 0
    total_hausdorff = 0
    total_dtw = 0
    total_cnt = 0
    for index, row in tqdm(pos_sample.iterrows(), total=pos_sample.shape[0], desc='train generator with true trajectory'):
        trace_loc = list(map(int, row['rid_list'].split(',')))
        trace_tim = list(map(encode_time, row['time_list'].split(',')))
        # 根据这个轨迹的 OD 生成轨迹
        generator.train(False)
        neg_trace_loc, neg_trace_tim = searcher.road_random_sample(gen_model=generator, trace_loc=[trace_loc[0]],
                                                                   trace_tim=[trace_tim[0]], des=trace_loc[-1],
                                                                   default_len=len(trace_loc))
        reward, yaw_distance = rollout.get_road_reward(generate_trace=(neg_trace_loc, neg_trace_tim), des=trace_loc[-1],
                                                       rollout_times=rollout_times, discriminator=discriminator)
        # 计算模型对每一步的候选集概率预测值与所选择下一跳在候选集中的下标
        generator.train(True)
        seq_len = len(neg_trace_loc)
        if seq_len <= 1:
            # 这个生成特别失败，但不应该有这种情况呀
            continue
        des_center_gps = rid_gps[str(trace_loc[-1])]
        candidate_prob_list = []
        gen_candidate = []
        for i in range(1, seq_len):
            des_tensor = torch.tensor([trace_loc[-1]]).to(device)
            input_trace_loc = neg_trace_loc[:i]
            input_trace_tim = neg_trace_tim[:i]
            now_rid = input_trace_loc[-1]
            candidate_set = adjacent_list[str(now_rid)]
            candidate_dis = []
            for c in candidate_set:
                candidate_gps = rid_gps[str(c)]
                d = distance.distance((des_center_gps[1], des_center_gps[0]),
                                      (candidate_gps[1], candidate_gps[0])).kilometers * 10
                candidate_dis.append(d)
            # 构建模型输入
            trace_loc_tensor = torch.LongTensor(input_trace_loc).to(device).unsqueeze(0)
            trace_tim_tensor = torch.LongTensor(input_trace_tim).to(device).unsqueeze(0)
            candidate_set_tensor = torch.LongTensor(candidate_set).to(device).unsqueeze(0)
            candidate_dis_tensor = torch.FloatTensor(candidate_dis).to(device).unsqueeze(0)
            candidate_prob = generator.predict(trace_loc=trace_loc_tensor, trace_time=trace_tim_tensor,
                                               des=des_tensor, candidate_set=candidate_set_tensor,
                                               candidate_dis=candidate_dis_tensor)
            candidate_prob_list.append(candidate_prob.squeeze(0))
            # 获取所选择的下一跳在候选集中的下标
            choose_index = candidate_set.index(neg_trace_loc[i])
            gen_candidate.append(choose_index)
        # 计算 loss
        reward = torch.tensor(reward).to(device)
        yaw_distance = torch.tensor(yaw_distance).to(device)
        gen_candidate = torch.tensor(gen_candidate).to(device)
        loss = gan_loss(candidate_prob=candidate_prob_list, gen_candidate=gen_candidate, reward=reward,
                        yaw_loss=yaw_distance)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), clip)
        gen_optimizer.zero_grad()
        loss.backward()
        gen_optimizer.step()
        # 更新模型的 gat cache
        generator.function_h.update_node_emb()
        # 计算评估指标
        total_edit_distance += edit_distance(neg_trace_loc, trace_loc)
        generate_gps_list = []
        for road_id in neg_trace_loc:
            now_gps = rid_gps[str(road_id)]
            generate_gps_list.append([now_gps[1], now_gps[0]])
        true_gps_list = []
        for road_id in trace_loc:
            now_gps = rid_gps[str(road_id)]
            true_gps_list.append([now_gps[1], now_gps[0]])
        true_gps_list = np.array(true_gps_list)
        generate_gps_list = np.array(generate_gps_list)
        total_hausdorff += hausdorff_metric(true_gps_list, generate_gps_list)
        total_dtw += dtw_metric(true_gps_list, generate_gps_list)
        total_cnt += 1
    logger.info('evaluate generator:')
    logger.info('avg EDT {}, avg hausdorff {}, avg dtw {}'.format(total_edit_distance / total_cnt,
                                                                  total_hausdorff / total_cnt,
                                                                  total_dtw / total_cnt))


if __name__ == '__main__':
    logger.info('load true trajectory.')
    # 加载真实轨迹数据
    trace = pd.read_csv(trajectory_file)
    total_trace = trace.shape[0]
    searcher = Searcher(device=device, adjacent_list=adjacent_list, road_center_gps=rid_gps, road_length=road_length,
                        road_time_distribution=road_time_distribution)
    # 加载预训练生成器
    logger.info('load pretrain generator from ' + pretrain_gan_file + ' and ' + pretrain_gat_file)
    generator = GeneratorV4(config=gen_config, data_feature=data_feature).to(device)
    generatorv1_state = torch.load(pretrain_gan_file, map_location=device)
    generator.function_g.load_state_dict(generatorv1_state)
    gat_state = torch.load(pretrain_gat_file, map_location=device)
    generator.function_h.load_state_dict(gat_state)
    # 开始对抗学习
    rollout = Rollout(searcher=searcher, generator=generator, device=device, od_distinct_route=od_distinct_route,
                      road_gps=rid_gps)
    # 预训练判别器
    logger.info('start pretrain discriminator.')
    discriminator = DiscriminatorV1(config=dis_config, data_feature=data_feature).to(device)
    if pretrain_discriminator:
        logger.info('start pretrain discriminator.')
        train_discriminator(max_epoch=pretrain_dis_epoch)
    else:
        logger.info('load discriminator from save pt file.')
        discriminator_state = torch.load(os.path.join(save_folder, 'adversarial_discriminator.pt'),
                                         map_location=device)
        discriminator.load_state_dict(discriminator_state)
    for epoch in range(total_epoch):
        logger.info('start train generator at epoch {}'.format(epoch))
        train_generator(stage=1)
        logger.info('start train discriminator at epoch {}'.format(epoch))
        train_discriminator(max_epoch=1)
        rollout.update_params(generator)
    # 保存本次训练的生成器与判别器
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(generator.state_dict(), os.path.join(save_folder, 'adversarial_3_generator_{}.pt'.format(exp_id)))
    torch.save(discriminator.state_dict(), os.path.join(save_folder, 'adversarial_discriminator.pt'))
