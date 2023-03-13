import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DiscriminatorV1(nn.Module):
    """第一版判别器
    Embedding + LSTM + Linear 来做一个轨迹的二分类。
    
    """

    def __init__(self, config, data_feature):
        """初始化

        Args:
            config (dict): 配置
            data_feature (dict): 数据相关特征
        """
        super(DiscriminatorV1, self).__init__()

        #### 模型参数
        self.road_emb_size = config['road_emb_size']
        # self.time_emb_size = config['time_emb_size']
        self.hidden_size = config['hidden_size']
        self.lstm_layer_num = config['lstm_layer_num']
        self.dropout_p = config['dropout_p']
        #### 模型结构
        # 计算输入层的大小
        self.input_size = self.road_emb_size  # + self.time_emb_size
        # Embedding 层
        self.road_emb = nn.Embedding(num_embeddings=data_feature['road_num'], embedding_dim=self.road_emb_size, padding_idx=data_feature['road_pad'])
        # 路段嵌入层应该可以加载预训练好的路网表征
        # self.time_emb = nn.Embedding(num_embeddings=data_feature['time_size'], embedding_dim=self.time_emb_size,
        # padding_idx=data_feature['time_pad'])
        # LSTM 层 捕捉序列性
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.lstm_layer_num, batch_first=True)
        # Dropout 层
        self.dropout = nn.Dropout(p=self.dropout_p)
        # 输出层
        self.out_linear = nn.Linear(in_features=self.hidden_size, out_features=2)  # 二分类问题
        # 损失函数
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, trace_loc, trace_time, trace_mask=None):
        """前馈过程

        Args:
            trace_loc (tensor): 当前轨迹位置序列. (batch_size, seq_len)
            trace_time (tensor): 当前轨迹时间序列. (batch_size, seq_len)
            trace_mask (tensor, optional): 轨迹 padding mask, 1 表示非补齐值, 0 表示为补齐值. Defaults to None. (batch_size, seq_len)
        
        Return:
            trace_score (tensor): 轨迹是否为真的分数. (batch_size, 2) 0 维度表示为假的概率，1 表示为真的概率。
        """

        # Multi-Model Embedding
        trace_loc_emb = self.road_emb(trace_loc)  # (batch_size, seq_len, road_emb_size)
        # trace_time_emb = self.time_emb(trace_time) # (batch_size, seq_len, time_emb_size)

        # 将输入嵌入向量拼接起来
        # input_emb = torch.cat([trace_loc_emb, trace_time_emb], dim=2) # (batch_size, seq_len, input_size)
        input_emb = trace_loc_emb
        if trace_mask is not None:
            # LSTM with Mask
            trace_origin_len = torch.sum(trace_mask, dim=1).tolist() # (batch_size)
            pack_input = pack_padded_sequence(input_emb, lengths=trace_origin_len, batch_first=True, enforce_sorted=False)
            pack_lstm_hidden, (hn, cn) = self.lstm(pack_input)
            # (batch_size, seq_len, hidden_size)
            lstm_hidden, _ = pad_packed_sequence(pack_lstm_hidden, batch_first=True)
        else:
            lstm_hidden, (hn, cn) = self.lstm(input_emb)
        # 抽取最后一层的 lstm_hidden
        if trace_mask is not None:
            # 获取各轨迹最后一个非补齐值对应的 hidden
            lstm_last_index = torch.sum(trace_mask, dim=1) - 1  # (batch_size)
            lstm_last_index = lstm_last_index.reshape(lstm_last_index.shape[0], 1, -1)  # (batch_size, 1, 1)
            lstm_last_index = lstm_last_index.repeat(1, 1, self.hidden_size)  # (batch_size, 1, hidden_size)
            # (batch_size, hidden_size)
            lstm_last_hidden = torch.gather(lstm_hidden, dim=1, index=lstm_last_index).squeeze(1)
        else:
            lstm_last_hidden = lstm_hidden[:, -1]
        # dropout
        lstm_last_hidden = self.dropout(lstm_last_hidden)
        # 输出
        trace_score = self.out_linear(lstm_last_hidden)
        return trace_score

    def predict(self, trace_loc, trace_time, trace_mask=None):
        """预测

        Args:
            trace_loc (tensor): 当前轨迹位置序列. (batch_size, seq_len)
            trace_time (tensor): 当前轨迹时间序列. (batch_size, seq_len)
            trace_mask (tensor, optional): 轨迹 padding mask, 1 表示非补齐值, 0 表示为补齐值. Defaults to None. (batch_size, seq_len)
        
        Return:
            candidate_prob (tensor): softmax 后对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        score = self.forward(trace_loc, trace_time, trace_mask)
        return torch.softmax(score, dim=1)
    
    def calculate_loss(self, trace_loc, trace_time, target, trace_mask=None):
        """前馈过程

        Args:
            trace_loc (tensor): 当前轨迹位置序列. (batch_size, seq_len)
            trace_time (tensor): 当前轨迹时间序列. (batch_size, seq_len)
            target (tensor): 真实的下一跳. (batch_size)
            trace_mask (tensor, optional): 轨迹 padding mask, 1 表示非补齐值, 0 表示为补齐值. Defaults to None. (batch_size, seq_len)
        
        Return:
            loss (tensor): 交叉损失熵 (1)
        """
        score = self.forward(trace_loc, trace_time, trace_mask)
        loss = self.loss_func(score, target)
        return loss