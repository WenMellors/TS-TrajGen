import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from generator.function_g_fc import FunctionGFC
from generator.distance_gat_fc import DistanceGatFC


class GeneratorV4(nn.Module):
    """第二版生成器
    将第一版生成器作为轨迹连贯性预测模块，并引入目的导向模块

    Args:
        nn.Module (torch.nn.Module): 继承 pytorch Module 接口类
    """

    def __init__(self, config, data_feature):
        """模型初始化

        Args:
            config (dict): 配置字典
            data_feature (dict): 数据集相关参数
        """
        super(GeneratorV4, self).__init__()
        # 模型参数
        self.dis_weight = config['dis_weight']
        # 距离和 Matcher 评分之间的权重
        self.w1 = nn.Parameter(torch.tensor([self.dis_weight]))
        self.function_g = FunctionGFC(config['function_g'], data_feature)
        self.function_h = DistanceGatFC(config['function_h'], data_feature)

    def forward(self, trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask=None, cache=True):
        """前馈过程

        Args:
            trace_loc (tensor): 当前轨迹位置序列. (batch_size, seq_len)
            trace_time (tensor): 当前轨迹时间序列. (batch_size, seq_len)
            des (tensor): 当前轨迹的目的地. (batch_size)
            candidate_set (tensor): 候选下一跳集合. (batch_size, candidate_size)
            candidate_dis (tensor): 候选下一跳距离终点的距离
            trace_mask (tensor, optional): 轨迹 padding mask, 1 表示非补齐值, 0 表示为补齐值. Defaults to None. (batch_size, seq_len)
            cache (bool): 是否缓存加速

        Return:
            candidate_prob (tensor): 对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        g_score = self.function_g.predict_g(trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask)
        if cache:
            h_score = self.function_h.predict_next(candidate_set, des, candidate_dis)
        else:
            raise NotImplementedError()
            # 这里是废案
            # h_score = self.function_h.predict(candidate_set, des, candidate_dis)
        # 这里先直接两个直接对半加权
        return torch.add((1 - self.w1) * g_score, self.w1 * h_score)

    def predict(self, trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask=None, cache=True):
        """预测

        Args:
            trace_loc (tensor): 当前轨迹位置序列. (batch_size, seq_len)
            trace_time (tensor): 当前轨迹时间序列. (batch_size, seq_len)
            des (tensor): 当前轨迹的目的地. (batch_size)
            candidate_set (tensor): 候选下一跳集合. (batch_size, candidate_size)
            candidate_dis (tensor): 候选下一跳距离终点的距离
            trace_mask (tensor, optional): 轨迹 padding mask, 1 表示非补齐值, 0 表示为补齐值. Defaults to None. (batch_size, seq_len)
            cache (bool): 是否缓存加速

        Return:
            candidate_prob (tensor): softmax 后对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        score = self.forward(trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask, cache)
        return torch.softmax(score, dim=1)

    def calculate_loss(self, trace_loc, trace_time, des, candidate_set, candidate_dis, target, trace_mask=None):
        """前馈过程

        Args:
            trace_loc (tensor): 当前轨迹位置序列. (batch_size, seq_len)
            trace_time (tensor): 当前轨迹时间序列. (batch_size, seq_len)
            des (tensor): 当前轨迹的目的地. (batch_size)
            candidate_set (tensor): 候选下一跳集合. (batch_size, candidate_size)
            candidate_dis (tensor): 候选下一跳距离终点的距离
            target (tensor): 真实的下一跳. (batch_size)
            trace_mask (tensor, optional): 轨迹 padding mask, 1 表示非补齐值, 0 表示为补齐值. Defaults to None. (batch_size, seq_len)

        Return:
            loss (tensor): 交叉损失熵 (1)
        """
        score = self.forward(trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask, False)
        loss = self.loss_func(score, target)
        return loss
