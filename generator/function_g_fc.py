import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class IntraAttention(nn.Module):
    """对轨迹经过 LSTM 后的隐藏层向量序列做 Attention 强化
    key: 当前轨迹经过 LSTM 后的隐藏层向量序列
    query: 轨迹向量序列的最后一个状态
    """

    def __init__(self, hidden_size):
        super(IntraAttention, self).__init__()
        # 模型参数
        self.hidden_size = hidden_size
        # 模型结构
        self.w1 = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)
        self.w2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.w3 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)

    def forward(self, query, key, mask=None):
        """前馈

        Args:
            query (tensor): shape (batch_size, hidden_size)
            key (tensor): shape (batch_size, seq_len, hidden_size)
            mask (tensor): padding mask, 1 表示非补齐值, 0 表示补齐值 shape (batch_size, seq_len)
        Return:
            attn_hidden (tensor): shape (batch_size, hidden_size)
        """
        attn_weight = torch.bmm(key, query.unsqueeze(2)).squeeze(2) # shape (batch_size, seq_len)
        if mask is not None:
            mask = attn_weight.masked_fill(mask==0, -1e9) # mask 设置为一个充分大的负数，这样 softmax 之后能够接近于 0
        attn_weight = torch.softmax(attn_weight, dim=1).unsqueeze(2)
        attn_hidden = torch.sum(attn_weight * key, dim=1)
        return attn_hidden


class Matcher(nn.Module):
    """Matcher 匹配打分
    根据当前轨迹隐藏层表征与候选集表征之间计算一个匹配程度，也就是下一跳的评分

    目前因为候选集表征与隐藏层表征维度不一样，所以对候选集表征过一个线性映射来计算。
    """

    def __init__(self, hidden_size, road_emb_size):
        super(Matcher, self).__init__()
        # 模型参数
        self.hidden_size = hidden_size
        self.road_emb_size = road_emb_size
        # 模型结构
        self.linear = nn.Linear(in_features=self.road_emb_size, out_features=self.hidden_size)

    def foward(self, trace_hidden, candidate_emb):
        """前馈

        Args:
            trace_hidden (tensor): 轨迹隐藏层表征 (batch_size, hidden_size)
            candidate_emb (tensor): 候选集表征 (batch_size, candidate_size, road_emb_size)
        """
        candidate_hidden = self.linear(candidate_emb).permute(0, 2, 1) # (batch_size, hidden_size, candidate_size)
        score = torch.bmm(trace_hidden.unsqueeze(1), candidate_hidden).squeeze(1) # (batch_size, candidate_size)
        return score


class MatcherV2(nn.Module):
    """候选集与当前轨迹状态之间的注意力模块
    目前实现类似于 TraceAttention
    """

    def __init__(self, candidate_input_size, hidden_size, dropout_p):
        super(MatcherV2, self).__init__()
        self.out_linear = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        self.w1_linear = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.w2_linear = nn.Linear(in_features=candidate_input_size, out_features=hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key):
        """
        计算 query 与 key 之间的相似度
        计算方法为可学习前馈神经网络：attn_weight = w_out * tanh(w_1 * query + w_2 * key)
        Args:
            query: 当前轨迹的状态表征向量。shape: (batch_size, hidden_size)
            key: 候选集的嵌入向量。shape: (batch_size, candidate_size, candidate_input_size)
            hidden_size = road_emb_size

        Returns:
            candidate_weight: 当前状态与候选集之间的相关性向量。shape: (batch_size, candidate_size)
        """
        query_hidden = torch.relu(self.w1_linear(query).unsqueeze(1))  # shape: (batch_size, 1, hidden_size)
        key_hidden = torch.relu(self.w2_linear(key))  # shape: (batch_size, candidate_size, hidden_size)
        candidate_weight = torch.tanh(query_hidden + key_hidden)  # shape: (batch_size, candidate_size, hidden_size)
        candidate_weight = self.dropout(candidate_weight)
        out = self.out_linear(candidate_weight).squeeze(2)  # shape: (batch_size, candidate_size)
        return out


class FunctionGFC(nn.Module):
    """取消 Matcher 方法，直接使用 FC

    Args:
        nn.Module (torch.nn.Module): 继承 pytorch Module 接口类
    """

    def __init__(self, config, data_feature):
        """模型初始化

        Args:
            config (dict): 配置字典
            data_feature (dict): 数据集相关参数
        """
        super(FunctionGFC, self).__init__()

        # 模型参数
        self.road_emb_size = config['road_emb_size']
        self.time_emb_size = config['time_emb_size']
        self.hidden_size = config['hidden_size']
        self.lstm_layer_num = config['lstm_layer_num']
        self.dropout_p = config['dropout_p']
        self.dis_weight = config['dis_weight']
        # 计算输入层的大小
        self.input_size = self.road_emb_size + self.time_emb_size
        # Embedding 层
        self.road_emb = nn.Embedding(num_embeddings=data_feature['road_num'], embedding_dim=self.road_emb_size, padding_idx=data_feature['road_pad'])
        # 路段嵌入层应该可以加载预训练好的路网表征
        self.time_emb = nn.Embedding(num_embeddings=data_feature['time_size'], embedding_dim=self.time_emb_size, padding_idx=data_feature['time_pad'])
        # LSTM 层 捕捉序列性
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.lstm_layer_num, batch_first=True)
        # 轨迹隐藏状态加强层
        self.intra_attn = IntraAttention(hidden_size=self.hidden_size)
        # Dropout 层
        self.dropout = nn.Dropout(p=self.dropout_p)
        # Output Linear
        self.out_fc = nn.Linear(in_features=self.hidden_size, out_features=data_feature['road_num'])
        # 损失函数
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask=None, train=True):
        """前馈过程

        Args:
            trace_loc (tensor): 当前轨迹位置序列. (batch_size, seq_len)
            trace_time (tensor): 当前轨迹时间序列. (batch_size, seq_len)
            des (tensor): 当前轨迹的目的地. (batch_size)
            candidate_set (tensor): 候选下一跳集合. (batch_size, candidate_size)
            candidate_dis (tensor): 候选下一跳距离终点的距离
            trace_mask (tensor, optional): 轨迹 padding mask, 1 表示非补齐值, 0 表示为补齐值. Defaults to None. (batch_size, seq_len)
            train (bool): 是否为训练时，因为距离的倒数是不需要训练的

        Return:
            candidate_prob (tensor): 对候选集下一跳的概率预测 (batch_size, candidate_size)
        """

        # Multi-Model Embedding
        trace_loc_emb = self.road_emb(trace_loc) # (batch_size, seq_len, road_emb_size)
        trace_time_emb = self.time_emb(trace_time) # (batch_size, seq_len, time_emb_size)
        # 将输入嵌入向量拼接起来
        input_emb = torch.cat([trace_loc_emb, trace_time_emb], dim=2) # (batch_size, seq_len, input_size)
        # 输入也 Dropout 一下
        input_emb = self.dropout(input_emb)
        self.lstm.flatten_parameters()
        if trace_mask is not None:
            # LSTM with Mask
            trace_origin_len = torch.sum(trace_mask, dim=1).tolist() # (batch_size)
            pack_input = pack_padded_sequence(input_emb, lengths=trace_origin_len, batch_first=True, enforce_sorted=False)
            pack_lstm_hidden, (hn, cn) = self.lstm(pack_input)
            lstm_hidden, _ = pad_packed_sequence(pack_lstm_hidden, batch_first=True) # (batch_size, seq_len, hidden_size)
        else:
            lstm_hidden, (hn, cn) = self.lstm(input_emb)
        # Self Attn with Mask
        if trace_mask is not None:
            # 获取各轨迹最后一个非补齐值对应的 hidden
            lstm_last_index = torch.sum(trace_mask, dim=1) - 1 # (batch_size)
            lstm_last_index = lstm_last_index.reshape(lstm_last_index.shape[0], 1, -1) # (batch_size, 1, 1)
            lstm_last_index = lstm_last_index.repeat(1, 1, self.hidden_size) # (batch_size, 1, hidden_size)
            lstm_last_hidden = torch.gather(lstm_hidden, dim=1, index=lstm_last_index).squeeze \
                (1) # (batch_size, hidden_size)
        else:
            lstm_last_hidden = lstm_hidden[:, -1]
        attn_hidden = self.intra_attn(query=lstm_last_hidden, key=lstm_hidden, mask=trace_mask) # (batch_size, hidden_size)
        # dropout
        attn_hidden = self.dropout(attn_hidden)
        # 使用线性层直接预测
        score = self.out_fc(attn_hidden)  # (batch_size, road_num)
        # 根据 candidate_set 选出对应 candidate 的 score
        candidate_score = torch.gather(score, dim=1, index=candidate_set)  # (batch_size, road_num)
        return candidate_score

    def predict(self, trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask=None):
        """预测

        Args:
            trace_loc (tensor): 当前轨迹位置序列. (batch_size, seq_len)
            trace_time (tensor): 当前轨迹时间序列. (batch_size, seq_len)
            des (tensor): 当前轨迹的目的地. (batch_size)
            candidate_set (tensor): 候选下一跳集合. (batch_size, candidate_size)
            candidate_dis (tensor): 候选下一跳距离终点的距离
            trace_mask (tensor, optional): 轨迹 padding mask, 1 表示非补齐值, 0 表示为补齐值. Defaults to None. (batch_size, seq_len)

        Return:
            candidate_prob (tensor): softmax 后对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        score = self.forward(trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask, False)
        return torch.softmax(score, dim=1)

    def predict_g(self, trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask=None):
        """预测，只是用 G 部分

        Args:
            trace_loc (tensor): 当前轨迹位置序列. (batch_size, seq_len)
            trace_time (tensor): 当前轨迹时间序列. (batch_size, seq_len)
            des (tensor): 当前轨迹的目的地. (batch_size)
            candidate_set (tensor): 候选下一跳集合. (batch_size, candidate_size)
            candidate_dis (tensor): 候选下一跳距离终点的距离
            trace_mask (tensor, optional): 轨迹 padding mask, 1 表示非补齐值, 0 表示为补齐值. Defaults to None. (batch_size, seq_len)

        Return:
            candidate_prob (tensor): softmax 后对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        score = self.forward(trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask, True)
        return torch.softmax(score, dim=1)

    def get_g_hidden(self, trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask=None):
        # Multi-Model Embedding
        trace_loc_emb = self.road_emb(trace_loc)  # (batch_size, seq_len, road_emb_size)
        trace_time_emb = self.time_emb(trace_time)  # (batch_size, seq_len, time_emb_size)
        # 将输入嵌入向量拼接起来
        input_emb = torch.cat([trace_loc_emb, trace_time_emb], dim=2)  # (batch_size, seq_len, input_size)
        # 输入也 Dropout 一下
        input_emb = self.dropout(input_emb)
        self.lstm.flatten_parameters()
        if trace_mask is not None:
            # LSTM with Mask
            trace_origin_len = torch.sum(trace_mask, dim=1).tolist()  # (batch_size)
            pack_input = pack_padded_sequence(input_emb, lengths=trace_origin_len, batch_first=True,
                                              enforce_sorted=False)
            pack_lstm_hidden, (hn, cn) = self.lstm(pack_input)
            lstm_hidden, _ = pad_packed_sequence(pack_lstm_hidden,
                                                 batch_first=True)  # (batch_size, seq_len, hidden_size)
        else:
            lstm_hidden, (hn, cn) = self.lstm(input_emb)
        # Self Attn with Mask
        if trace_mask is not None:
            # 获取各轨迹最后一个非补齐值对应的 hidden
            lstm_last_index = torch.sum(trace_mask, dim=1) - 1  # (batch_size)
            lstm_last_index = lstm_last_index.reshape(lstm_last_index.shape[0], 1, -1)  # (batch_size, 1, 1)
            lstm_last_index = lstm_last_index.repeat(1, 1, self.hidden_size)  # (batch_size, 1, hidden_size)
            lstm_last_hidden = torch.gather(lstm_hidden, dim=1, index=lstm_last_index).squeeze \
                (1)  # (batch_size, hidden_size)
        else:
            lstm_last_hidden = lstm_hidden[:, -1]
        attn_hidden = self.intra_attn(query=lstm_last_hidden, key=lstm_hidden,
                                      mask=trace_mask)  # (batch_size, hidden_size)
        # dropout
        attn_hidden = self.dropout(attn_hidden)  # (batch_size, hidden_size)
        return attn_hidden

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
        score = self.forward(trace_loc, trace_time, des, candidate_set, candidate_dis, trace_mask, True)
        loss = self.loss_func(score, target)
        return loss
