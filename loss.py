import torch
from torch.nn.utils.rnn import pad_sequence


def gan_loss(candidate_prob, gen_candidate, reward, yaw_loss):
    """
    References: https://github.com/ZiJianZhao/SeqGAN-PyTorch
    不太清楚这个 loss 是不是 Policy Gradient 应该是吧
    简单来说就是基于判别器每一步的 Reward 以及 Generator 在每一步的预测概率来 update
    Generator 在一步中做出的选择，判别器返回一个 Reward，将 Reward 和 prob 相乘取负数得到 loss
    这样最小化 loss 也就是最大化期望 Reward
    Args:
        candidate_prob (list of tensor): (candidate_size) 生成器对每一步预测的候选集概率。
            因为不同步的候选集大小不一致，所以这里没法传张量，只能是 List。
            应该只有这个参数里面的 tensor 是带梯度的
        gen_candidate (tensor): (seq_len) 生成器最终选择的候选点在候选集中的索引。
        reward (tensor): (seq_len) 判别器对生成器每一步选择的 Reward。

    Returns:
        loss (tensor): Reward-Refined NLLLoss.
    """
    candidate_prob = pad_sequence(candidate_prob, batch_first=True)  # (seq_len, candidate_size)
    # 选出 gen_candidate 对应的 prob
    select_prob = torch.gather(candidate_prob, dim=1, index=gen_candidate.unsqueeze(1)).squeeze(1)  # (seq_len)
    select_prob_neg_log = - torch.log(select_prob)
    # reward 相当于一个权重，在一般的 NLL loss 中可以视作 reward=1 的情况（就是特别真）
    # 这个 loss 的目的就是要模型强化那些特别真的选择
    # reward_refined_prob_log = select_prob_neg_log * reward
    # reward_refined_nll_loss = torch.sum(reward_refined_prob_log, dim=0)
    # 还需要将每个点的偏航损失带入梯度中
    # PLAN A: 计算每个点的置信度概率，然后乘以相应的偏航损失，得到加权后的损失
    # PLAN A: 问题就是序列一长起来，cum_prob 就很小了，这样导致后面的几乎没怎么学到可能
    # PLAN B: 看距离的增量，每一步决策影响不了前面，只能影响后面，所以看这一步做出之后的 yaw_loss 的变化值
    # 在这个基础上，引入偏航情况，就是要通过轨迹距离来判断哪些决策是做的对的
    # 标准呢？就是距离降幅越大的步骤做的越对
    # 因此应该使用 PLAN B
    with torch.no_grad():
        yaw_distance_decrease_val = yaw_loss[:-1] - yaw_loss[1:]  # yaw_loss[i] - yaw_loss[i+1]
        # 值为正，就表明距离变小的，是好的。值越大，降幅越大
        # 因为是权重，所以映射到 [0, 1]
        yaw_distance_decrease_rate_min = torch.min(yaw_distance_decrease_val)
        yaw_distance_decrease_rate_max = torch.max(yaw_distance_decrease_val)
        if yaw_distance_decrease_rate_max != yaw_distance_decrease_rate_min:
            yaw_distance_decrease_rate = (yaw_distance_decrease_val - yaw_distance_decrease_rate_min) /\
                                         (yaw_distance_decrease_rate_max - yaw_distance_decrease_rate_min)
        else:
            # 如果最大、最小一样的话，就会出现除 0，也就是说没有好坏，权重赋予 0
            yaw_distance_decrease_rate = yaw_distance_decrease_val - yaw_distance_decrease_rate_min
    # select_prob_neg_log_weight = reward + yaw_distance_decrease_rate
    select_prob_neg_log_weight = reward
    loss = torch.sum(select_prob_neg_log * select_prob_neg_log_weight, dim=0)
    # reward_refined_prob = select_prob * reward
    # reward_refined_log_prob = torch.log(reward_refined_prob)
    # loss = - torch.sum(reward_refined_log_prob, dim=0)
    return loss


def mask_mape_loss(pred, target):
    mask = target != 0
    return torch.mean((torch.abs(pred[mask] - target[mask])/target[mask]))
