# 通过网格像素化，将轨迹转化为图像，用于对抗学习
import json
import numpy as np
import math
from geopy import distance
import torchvision.transforms as transforms

# lon_range = 0.2507  # 地图经度的跨度
# lat_range = 0.21  # 地图纬度的跨度
# img_unit = 0.005  # 这样画出来的大小，大概是 0.42 km * 0.55 km 的格子
# lon_0 = 116.25
# lat_0 = 39.79  # 地图最左下角的坐标（即原点坐标）
# img_width = math.ceil(lon_range / img_unit) + 1  # 图像的宽度
# img_height = math.ceil(lat_range / img_unit) + 1  # 映射出的图像的高度
#
# region_img_unit = 0.001
# region_img_width = math.ceil(lon_range / region_img_unit) + 1  # 图像的宽度
# region_img_height = math.ceil(lat_range / region_img_unit) + 1  # 映射出的图像的高度
#
#
# with open('./data/kaffpa_region2rid.json', 'r') as f:
#     region2rid = json.load(f)
#
# with open('./data/rid_gps.json', 'r') as f:
#     rid_gps_dict = json.load(f)
#
# with open('./data/kaffpa_region_adjacent_list.json', 'r') as f:
#     region_adjacent_list = json.load(f)
#
# img_size = 64
# # 将轨迹转化为图片
# transform = transforms.Compose([
#         transforms.ToPILImage(mode='L'),
#         transforms.Resize(img_size),
#         transforms.CenterCrop(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5), (0.5))]
# )


class MapManager(object):

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if self.dataset_name == 'Xian':
            self.lon_0 = 108.8555109
            self.lon_1 = 109.0313147
            self.lat_0 = 34.22585602
            self.lat_1 = 34.29639323
        else:
            raise NotImplementedError()
        self.img_unit = 0.005  # 这样画出来的大小，大概是 0.42 km * 0.55 km 的格子
        self.img_width = math.ceil((self.lon_1 - self.lon_0) / self.img_unit) + 1  # 图像的宽度
        self.img_height = math.ceil((self.lat_1 - self.lat_0) / self.img_unit) + 1  # 映射出的图像的高度

    def gps2grid(self, lon, lat):
        """
        GPS 经纬度点映射为图像网格
        Args:
            lon: 经度
            lat: 纬度

        Returns:
            x, y: 映射的网格的 x 与 y 坐标
        """
        x = math.floor((lon - self.lon_0) / self.img_unit)
        y = math.floor((lat - self.lat_0) / self.img_unit)
        return x, y


    # def gps_trace2img(trace):
    #     """
    #
    #     Args:
    #         trace: GPS 数组，即一条轨迹的 GPS
    #
    #     Returns:
    #         img (numpy.array)： H * W * 1 的二值图像
    #     """
    #     # H * W * C C 表示信道数，因为是二值图像，所以 C = 1
    #     img = np.ones((img_height, img_width, 1), dtype=np.uint8)  # 255 为白色，0 为黑色
    #     img = img * 255  # 初始画布为全白
    #     for point in trace:
    #         x, y = gps2grid(point[0], point[1])
    #         img[y][x] = 0
    #         # 周围的方块也标记成黑色
    #         if x > 0:
    #             # 左边
    #             img[y][x-1] = 0
    #             if y > 0:
    #                 # 左下
    #                 img[y-1][x-1] = 0
    #             if y + 1 < img_height:
    #                 # 左上
    #                 img[y+1][x-1] = 0
    #         if y > 0:
    #             # 下边
    #             img[y-1][x] = 0
    #             if x < img_width - 1:
    #                 # 右下
    #                 img[y-1][x+1] = 0
    #         if y < img_height - 1:
    #             # 上
    #             img[y+1][x] = 0
    #         if x < img_width - 1:
    #             # 右
    #             img[y][x+1] = 0
    #             if y < img_height - 1:
    #                 # 右上
    #                 img[y+1][x+1] = 0
    #     return img

    #
    # def region2img(region_trace):
    #     """
    #     将区域轨迹转化为 np 图片矩阵
    #     Args:
    #         region_trace:
    #
    #     Returns:
    #
    #     """
    #     # 先将 region trace 转化为边界路段轨迹
    #     border_trace = [rid_gps_dict[str(np.random.choice(region2rid[str(region_trace[0])]))]]
    #     for i in range(1, len(region_trace)):
    #         # 寻找 i-1 区域与 i 区域的边界路段
    #         border_road_list = region_adjacent_list[str(region_trace[i - 1])][str(region_trace[i])]
    #         border_trace.append(rid_gps_dict[str(np.random.choice(border_road_list))])
    #     img_np = gps_trace2img(border_trace)
    #     img_trans = transform(img_np)
    #     return img_trans
