# 使用 GAT 学习路段表征，希望该表征能够拟合各路段间的路径距离
from generator.gat import GATLayerImp3
import torch
import torch.nn as nn


class DistanceGatFC(nn.Module):

    def __init__(self, config, data_feature):
        super(DistanceGatFC, self).__init__()
        self.embed_dim = config['embed_dim']
        self.num_of_heads = config['num_of_heads']
        self.concat = config['concat']
        self.device = config['device']
        # self.dis_emb_dim = config['dis_emb_dim']
        self.gps_emb_dim = config['gps_emb_dim']
        self.distance_mode = config['distance_mode']
        self.no_gps_emb = config.get('no_gps_emb', False)
        # self.fc1_dim = config['fc1_dim']
        # 加载图信息
        self.adj_mx = data_feature.get('adj_mx')
        self.Apt = torch.LongTensor([self.adj_mx.row.tolist(), self.adj_mx.col.tolist()]).to(self.device)
        self.node_features = data_feature['node_features']
        if not self.no_gps_emb:
            self.feature_dim = self.node_features.shape[1] - 2 + self.gps_emb_dim * 2
        else:
            self.feature_dim = self.node_features.shape[1]
        self.gat_encoder = GATLayerImp3(num_in_features=self.feature_dim, num_out_features=self.embed_dim,
                                        num_of_heads=self.num_of_heads, concat=self.concat, device=self.device)
        # self.gat_encoder2 = GATLayerImp3(num_in_features=self.embed_dim, num_out_features=self.embed_dim,
        #                                  num_of_heads=self.num_of_heads, concat=self.concat, device=self.device)
        # 对 lat 与 lon 做一个 Embedding
        if not self.no_gps_emb:
            self.lat_embed = nn.Embedding(num_embeddings=data_feature['img_height'], embedding_dim=self.gps_emb_dim)
            self.lon_embed = nn.Embedding(num_embeddings=data_feature['img_width'], embedding_dim=self.gps_emb_dim)
        # 距离直接做个映射吧
        # 综合表征，距离，做一个 MLP 预测
        self.emb_fc = nn.Linear(in_features=self.embed_dim * 2, out_features=10)
        self.dis_fc = nn.Linear(in_features=1, out_features=10)
        self.out_fc = nn.Linear(in_features=20, out_features=1)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.node_emb_feature = None

    def forward(self, from_node, to_node, distance):
        """
        计算 from_node 与 to_node 之间的距离
        Args:
            from_node (batch_size): 源节点的编号
            to_node (batch_size): 目标节点的编号
            distance (batch_size): 两点之间的之间距离

        Returns:
            route_distance (batch_size): 返回模型预估的路径距离
        """
        if not self.no_gps_emb:
            # 需要对 node_features 的后两列，也就是 lon_grid 与 lat_grid 做一个 embedding
            lon_feature = self.node_features[:, -2].long()
            lat_feature = self.node_features[:, -1].long()
            lon_feature_emb = self.lon_embed(lon_feature)
            lat_feature_emb = self.lat_embed(lat_feature)
            in_feature = torch.cat([lon_feature_emb, lat_feature_emb, self.node_features[:, :-2]], dim=1)
        else:
            in_feature = self.node_features
        # GAT
        encode_feature = self.gat_encoder([in_feature, self.Apt])[0]
        # encode_feature2 = self.gat_encoder2([encode_feature, self.Apt])[0]
        # 计算表征距离
        from_node_emb = encode_feature[from_node]
        to_node_emb = encode_feature[to_node]
        emb_fc_out = torch.relu(self.emb_fc(torch.cat([from_node_emb, to_node_emb], dim=1)))
        dis_fc_out = torch.relu(self.dis_fc(distance.unsqueeze(1)))
        final_out = self.out_fc(torch.cat([emb_fc_out, dis_fc_out], dim=1)).squeeze(1)
        return final_out

    def calculate_loss(self, candidate_set, des, candidate_distance, target):
        """
        训练函数
        Args:
            candidate_set (batch_size, candidate_set): 候选集
            des (batch_size): 目的地
            candidate_distance (batch_size): 候选集距离目的地的距离
            target (batch_size): 真实的下一跳下标

        Returns:
            loss (tensor)
        """
        candidate_size = candidate_set.shape[1]
        from_node = candidate_set.flatten()
        # (batch_size, candidate_size)
        des_extend = des.unsqueeze(1).repeat(1, candidate_size)
        to_node = des_extend.flatten()
        distance = candidate_distance.flatten()
        score = self.forward(from_node, to_node, distance)
        candidate_score = score.reshape(candidate_set.shape[0], candidate_set.shape[1])
        return self.loss_func(candidate_score, target)

    def predict(self, candidate_set, des, candidate_distance):
        """
        预测评估
        Args:
            candidate_set (batch_size, candidate_set): 候选集
            des (batch_size): 目的地
            candidate_distance (batch_size): 候选集距离目的地的距离

        Returns:
            candidate_score
        """
        candidate_size = candidate_set.shape[1]
        from_node = candidate_set.flatten()
        # (batch_size, candidate_size)
        des_extend = des.unsqueeze(1).repeat(1, candidate_size)
        to_node = des_extend.flatten()
        distance = candidate_distance.flatten()
        score = self.forward(from_node, to_node, distance)
        candidate_score = score.reshape(candidate_set.shape[0], candidate_set.shape[1])
        return torch.softmax(candidate_score, dim=1)

    def predict_cache(self, from_node, to_node, distance):
        """
        计算 from_node 与 to_node 之间的距离，预测时使用缓存的 gat 特征，来加速预测。
        Args:
            from_node (batch_size): 源节点的编号
            to_node (batch_size): 目标节点的编号
            distance (batch_size): 两点之间的之间 gps 距
        Returns:
            route_distance (batch_size): 返回模型预估的路径距离
        """
        # 计算表征距离
        from_node_emb = self.node_emb_feature[from_node]
        to_node_emb = self.node_emb_feature[to_node]
        emb_fc_out = torch.relu(self.emb_fc(torch.cat([from_node_emb, to_node_emb], dim=1)))
        dis_fc_out = torch.relu(self.dis_fc(distance.unsqueeze(1)))
        final_out = self.out_fc(torch.cat([emb_fc_out, dis_fc_out], dim=1)).squeeze(1)
        return final_out

    def predict_next(self, candidate_set, des, candidate_distance):
        """
        预测下一跳
        Args:
            candidate_set: 候选集。shape (batch_size, candidate_size)
            des: 目的地。shape (batch_size)
            candidate_distance: 候选集到目的地的距离。(batch_size, candidate_size)

        Returns:
            candidate_prob: 候选集的下一跳概率 (batch_size, candidate_size)
        """
        # 预测的时候，不需要动态计算 GAT，直接缓存一次就可以了
        self._setup_node_emb()
        candidate_size = candidate_set.shape[1]
        from_node = candidate_set.flatten()
        # (batch_size, candidate_size)
        des_extend = des.unsqueeze(1).repeat(1, candidate_size)
        to_node = des_extend.flatten()
        distance = candidate_distance.flatten()
        score = self.predict_cache(from_node, to_node, distance)
        candidate_score = score.reshape(candidate_set.shape[0], candidate_set.shape[1])
        return torch.softmax(candidate_score, dim=1)

    def get_h_hidden(self, candidate_set, des, candidate_distance, cache=True):
        candidate_size = candidate_set.shape[1]
        from_node = candidate_set.flatten()
        # (batch_size, candidate_size)
        des_extend = des.unsqueeze(1).repeat(1, candidate_size)
        to_node = des_extend.flatten()
        distance = candidate_distance.flatten()
        if cache:
            self._setup_node_emb()
            from_node_emb = self.node_emb_feature[from_node]
            to_node_emb = self.node_emb_feature[to_node]
        else:
            if not self.no_gps_emb:
                # 需要对 node_features 的后两列，也就是 lon_grid 与 lat_grid 做一个 embedding
                lon_feature = self.node_features[:, -2].long()
                lat_feature = self.node_features[:, -1].long()
                lon_feature_emb = self.lon_embed(lon_feature)
                lat_feature_emb = self.lat_embed(lat_feature)
                in_feature = torch.cat([lon_feature_emb, lat_feature_emb, self.node_features[:, :-2]], dim=1)
            else:
                in_feature = self.node_features
            # GAT
            encode_feature = self.gat_encoder([in_feature, self.Apt])[0]
            from_node_emb = encode_feature[from_node]
            to_node_emb = encode_feature[to_node]
        emb_fc_out = torch.relu(self.emb_fc(torch.cat([from_node_emb, to_node_emb], dim=1)))
        dis_fc_out = torch.relu(self.dis_fc(distance.unsqueeze(1)))
        h_hidden = torch.cat([emb_fc_out, dis_fc_out], dim=1)  # (batch_size * candidate_set, 20)
        return h_hidden.reshape(candidate_set.shape[0], candidate_set.shape[1], -1)

    def _setup_node_emb(self):
        if self.node_emb_feature is None:
            if not self.no_gps_emb:
                # 需要对 node_features 的后两列，也就是 lon_grid 与 lat_grid 做一个 embedding
                lon_feature = self.node_features[:, -2].long()
                lat_feature = self.node_features[:, -1].long()
                lon_feature_emb = self.lon_embed(lon_feature)
                lat_feature_emb = self.lat_embed(lat_feature)
                in_feature = torch.cat([lon_feature_emb, lat_feature_emb, self.node_features[:, :-2]], dim=1)
            else:
                in_feature = self.node_features
            # GAT
            encode_feature = self.gat_encoder([in_feature, self.Apt])[0]
            # encode_feature2 = self.gat_encoder2([encode_feature, self.Apt])[0]
            self.node_emb_feature = encode_feature

    def update_node_emb(self):
        """
        更新 cache 中的 node_emb
        Returns:

        """
        if not self.no_gps_emb:
            # 需要对 node_features 的后两列，也就是 lon_grid 与 lat_grid 做一个 embedding
            lon_feature = self.node_features[:, -2].long()
            lat_feature = self.node_features[:, -1].long()
            lon_feature_emb = self.lon_embed(lon_feature)
            lat_feature_emb = self.lat_embed(lat_feature)
            in_feature = torch.cat([lon_feature_emb, lat_feature_emb, self.node_features[:, :-2]], dim=1)
        else:
            in_feature = self.node_features
        # GAT
        encode_feature = self.gat_encoder([in_feature, self.Apt])[0]
        # encode_feature2 = self.gat_encoder2([encode_feature, self.Apt])[0]
        self.node_emb_feature = encode_feature
