import json
import os
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
import hausdorff
from scipy.spatial.distance import euclidean, cosine, cityblock, chebyshev
from fastdtw import fastdtw
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Geod
from functools import partial
from scipy.stats import entropy
from datetime import datetime
from collections import Counter
from geopy import distance
import scipy
import math
import numpy as np
# import geopandas as gpd
from matplotlib import pyplot as plt

debug = False

# 读取路网长度字典与路段 GPS 字典
road_len_file = './data/porto_road_length.json'
if not os.path.exists(road_len_file) or not os.path.exists('./data/porto_rid_gps.json'):
    road_info = pd.read_csv('./data/porto.geo')
    road_length = {}
    road_gps = {}
    for index, row in tqdm(road_info.iterrows(), desc='cal road length'):
        rid = row['geo_id']
        length = row['length']
        coordinate = row['coordinates'].replace('[', '')
        coordinate = coordinate.replace(']', '').split(',')
        lon1 = float(coordinate[0])
        lat1 = float(coordinate[1])
        lon2 = float(coordinate[2])
        lat2 = float(coordinate[3])
        center_gps = ((lon1 + lon2) / 2, (lat1 + lat2) / 2)
        road_gps[str(rid)] = center_gps
        road_length[str(rid)] = length
    # 保存
    with open(road_len_file, 'w') as f:
        json.dump(road_length, f)
    with open('./data/porto_rid_gps.json', 'w') as f:
        json.dump(road_gps, f)
else:
    with open(road_len_file, 'r') as f:
        road_length = json.load(f)
    with open('./data/porto_rid_gps.json', 'r') as f:
        road_gps = json.load(f)

# 划分网格
# lon_range = 0.2507  # 地图经度的跨度
# lat_range = 0.21  # 地图纬度的跨度
lon_range = 0.133  # porto
lat_range = 0.046  # porto
# lon_0 = 116.25
# lat_0 = 39.79  # 地图最左下角的坐标（即原点坐标）
lon_0 = -8.6887
lat_0 = 41.1405
img_unit = 0.005  # 这样画出来的大小，大概是 0.42 km * 0.55 km 的格子
img_width = math.ceil(lon_range / img_unit) + 1  # 图像的宽度
img_height = math.ceil(lat_range / img_unit) + 1  # 映射出的图像的高度
road_pad = 11095
max_distance = 100  # 这里设置一个出行上限阈值
real_max_distance = 31  # 这个值是真实数据中的上限
max_radius = 31.6764 * 31.6764
real_max_radius = 7.2  # 这个值是真实数据中的上限
# 手动构建 distance_bins, radius_bins
travel_distance_bins = np.arange(0, real_max_distance, float(real_max_distance) / 1000).tolist()
# 将从 real max distance 到 max_distance 设置为一个 bin
travel_distance_bins.append(real_max_distance + 1)
travel_distance_bins.append(max_distance)
travel_distance_bins = np.array(travel_distance_bins)
travel_radius_bins = np.arange(0, real_max_radius, float(real_max_radius) / 100).tolist()
travel_radius_bins.append(real_max_radius + 1)
travel_radius_bins.append(max_radius)
travel_radius_bins = np.array(travel_radius_bins)

# 预先计算 road 与 grid 的映射关系
if not os.path.exists('./data/porto_road2grid.json'):
    road2grid = {}
    for road in road_gps:
        gps = road_gps[road]
        x = math.ceil((gps[0] - lon_0) / img_unit)
        y = math.ceil((gps[1] - lat_0) / img_unit)
        road2grid[road] = (x, y)
    # 缓存
    with open('./data/porto_road2grid.json', 'w') as f:
        json.dump(road2grid, f)
else:
    with open('./data/porto_road2grid.json', 'r') as f:
        road2grid = json.load(f)

road_num = len(road2grid)


def cal_polygon_area(polygon, mode=1):
    """
    计算经纬度多边形的覆盖面积（平方米不会算，先用平方度来做）

    Args:
        polygon (list): 多边形顶点经纬度数组
        mode (int): 1: 平方度， 2：平方千米

    Returns:
        area (float)
    """
    if mode == 1:
        if len(polygon) < 3:
            return 0
        area = Polygon(polygon)
        return area.area
    else:
        if len(polygon) < 3:
            return 0
        geod = Geod(ellps="WGS84")
        area, _ = geod.geometry_area_perimeter(orient(Polygon(polygon)))  # 单位平方米
        return area / 1000000


def arr_to_distribution(arr, min, max, bins=10000):
    """
    convert an array to a probability distribution
    :param arr: np.array, input array
    :param min: float, minimum of converted value
    :param max: float, maximum of converted value
    :param bins: int, number of bins between min and max
    :return: np.array, output distribution array
    """
    distribution, base = np.histogram(
        arr, np.arange(
            min, max, float(
                max - min) / bins))
    return distribution


def get_geogradius(rid_lat, rid_lon):
    """
    get the std of the distances of all points away from center as `gyration radius`
    :param trajs:
    :return:
    """
    if len(rid_lat) == 0:
        return 0
    lng1, lat1 = np.mean(rid_lon), np.mean(rid_lat)
    rad = []
    for i in range(len(rid_lat)):
        lng2 = rid_lon[i]
        lat2 = rid_lat[i]
        dis = distance.distance((lat1, lng1), (lat2, lng2)).kilometers
        rad.append(dis)
    rad = np.mean(rad)
    return rad


def count_statistics(trace_set, mode):
    """
    统计轨迹的特性
    出行距离分布、出行覆盖面积分布
    轨迹集合的网格访问频次与路段访问频次
    （分时间片与不分时间片都要统计）
    Args:
        trace_set (pandas.DataFrame): 轨迹表集
        mode (bool): 表示真实轨迹还是生成轨迹，因为生成轨迹生成的是时间片

    Returns:
        count_result
    """
    travel_distance_hour = {}
    travel_radius_hour = {}
    travel_distance_total = []
    travel_radius_total = []
    # top_location_cnt = Counter()
    total_grid = img_width * img_height
    grid_od_cnt = np.zeros((total_grid, total_grid), dtype=np.int)
    grid_time_od_cnt = np.zeros((24, total_grid, total_grid), dtype=np.int)
    rid_cnt = np.zeros(road_num, dtype=np.int)
    rid_time_cnt = np.zeros((24, road_num), dtype=np.int)
    for index, row in tqdm(trace_set.iterrows(), total=trace_set.shape[0], desc='count trajectory'):
        rid_list = [int(x) for x in row['rid_list'].split(',')]
        # 删除补齐值
        rid_list = np.array(rid_list)
        rid_list = rid_list[rid_list != road_pad].tolist()
        if len(rid_list) == 0:
            continue
        time_list = row['time_list'].split(',')
        # 以轨迹的出发时间来归类轨迹为哪一个时间片的轨迹
        if mode:
            start_timestamp = time_list[0]
            start_time = datetime.strptime(start_timestamp, '%Y-%m-%dT%H:%M:%SZ')
            start_hour = start_time.hour
        else:
            start_hour = (int(time_list[0]) % 1440) // 60
        travel_distance = 0
        pre_gps = None
        rid_lat = []
        rid_lon = []
        # 计算轨迹的 OD 流
        if str(rid_list[0]) not in road2grid:
            # 这是条废物轨迹，不做统计
            continue
        start_rid_grid = road2grid[str(rid_list[0])]
        des_rid_grid = None
        start_rid_grid_index = start_rid_grid[0] * img_height + start_rid_grid[1]
        for rid_index, rid in enumerate(rid_list):
            if rid == road_pad:
                # 补齐值后面都是补齐的轨迹了
                des_rid_grid = road2grid[str(rid_list[rid_index - 1])]
                break
            # 考虑到某些方法生成的轨迹路段并不邻接，所以这里使用轨迹的 GPS 来计算距离
            gps = road_gps[str(rid)]
            if pre_gps is None:
                pre_gps = gps
            else:
                travel_distance += distance.distance((gps[1], gps[0]), (pre_gps[1], pre_gps[0])).kilometers
                pre_gps = gps
            rid_lat.append(gps[1])
            rid_lon.append(gps[0])
            if mode:
                rid_time = datetime.strptime(time_list[rid_index], '%Y-%m-%dT%H:%M:%SZ')
                rid_hour = rid_time.hour
            else:
                rid_hour = (int(time_list[rid_index]) % 1440) // 60
            rid_cnt[rid] += 1
            rid_time_cnt[rid_hour, rid] += 1
        travel_distance_total.append(travel_distance)
        # 计算 radius
        travel_radius = get_geogradius(rid_lat=rid_lat, rid_lon=rid_lon)
        # top_location_cnt.update(rid_list)
        travel_radius_total.append(travel_radius)
        # 计算 grid od
        if des_rid_grid is None:
            des_rid_grid = road2grid[str(rid_list[-1])]
        des_rid_grid_index = des_rid_grid[0] * img_height + des_rid_grid[1]
        grid_od_cnt[start_rid_grid_index][des_rid_grid_index] += 1
        grid_time_od_cnt[start_hour][start_rid_grid_index][des_rid_grid_index] += 1
        if start_hour not in travel_distance_hour:
            travel_distance_hour[start_hour] = [travel_distance]
            travel_radius_hour[start_hour] = [travel_radius]
        else:
            travel_distance_hour[start_hour].append(travel_distance)
            travel_radius_hour[start_hour].append(travel_radius)
        if index == 1000 and debug:
            break
    # 计算 travel_distance 与 travel_radius 的分布
    travel_distance_total_distribution, _ = np.histogram(travel_distance_total, travel_distance_bins)
    travel_radius_total_distribution, _ = np.histogram(travel_radius_total, travel_radius_bins)
    # 将 grid_od_cnt 平展然后计算成频率
    # 不要除以总数，这样会把值整得很小，感觉会有误差
    grid_od_cnt = grid_od_cnt.flatten()
    grid_time_od_cnt = grid_time_od_cnt.reshape(24, -1)
    grid_time_od_cnt = grid_time_od_cnt
    result = {'travel_distance_total_distribution': travel_distance_total_distribution,
              'travel_radius_total_distribution': travel_radius_total_distribution,
              'travel_distance_hour_distribution': np.zeros((24, travel_distance_total_distribution.shape[0])),
              'travel_radius_hour_distribution': np.zeros((24, travel_radius_total_distribution.shape[0])),
              'grid_od_freq': grid_od_cnt, 'grid_time_od_freq': grid_time_od_cnt,
              'rid_freq': rid_cnt, 'rid_time_freq': np.zeros((24, road_num))}
    # 统计 top 50 位置集合以及对应的频率
    # top_50_location = [x[0] for x in top_location_cnt.most_common(50)]
    # top_50_location_freq = result['rid_freq'][top_50_location]
    # result['top_50_location'] = top_50_location
    # result['top_50_location_freq'] = top_50_location_freq
    # if mode:
    #     top_100_location = [x[0] for x in top_location_cnt.most_common(100)]
    #     # 计算 top_100_location 的访问频率
    #     location_frequency = rid_cnt / np.sum(rid_cnt)
    #     top_100_location_frequency = location_frequency[top_100_location]
    #     # 标准化
    #     result['top_location_frequency_distribution'] = arr_to_distribution(top_100_location_frequency, 0, 1, 100)
    # else:
    #     top_100_location = [x[0] for x in top_location_cnt.most_common(100)]
    #     location_frequency = rid_cnt / np.sum(rid_cnt)
    #     top_100_location_frequency = location_frequency[top_100_location]
    #     result['top_location_frequency_distribution'] = arr_to_distribution(top_100_location_frequency, 0, 1, 100)
    for hour in range(24):
        if hour in travel_distance_hour:
            result['rid_time_freq'][hour] = rid_time_cnt[hour]
            result['travel_distance_hour_distribution'][hour], _ = np.histogram(travel_distance_hour[hour], travel_distance_bins)
            result['travel_radius_hour_distribution'][hour], _ = np.histogram(travel_radius_hour[hour], travel_radius_bins)
    return result


def js_divergence(p, q):
    """JS散度

    Args:
        p(np.array):
        q(np.array):

    Returns:

    """
    m = (p + q) / 2
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


# def grid_distribution_map(grid_cnt, name1='grid_distribution'):
#     """
#
#     Args:
#         grid_cnt (np.array): shape (img_width, img_height) 网格访问频次
#         name1 (string): 分布图的名字
#
#     Returns:
#
#     """
#     geojson = dict()
#     geojson['type'] = 'FeatureCollection'
#     obj_list = []
#     # 计算各网格的坐标
#     grid_coordinates = []
#     grid_xy = []
#     for x in range(grid_cnt.shape[0]):
#         for y in range(grid_cnt.shape[1]):
#             x_0 = lon_0 + img_unit * x
#             y_0 = lat_0 + img_unit * y
#             x_1 = x_0 + img_unit
#             y_1 = y_0 + img_unit
#             coordinates = [[x_0, y_0], [x_1, y_0], [x_1, y_1], [x_0, y_1], [x_0, y_0]]
#             grid_coordinates.append(coordinates)
#             grid_xy.append((x, y))
#     for (i, grid) in enumerate(grid_coordinates):
#         obj = dict()
#         obj['type'] = 'Feature'
#         obj['properties'] = dict()
#         obj['properties']['cnt'] = int(grid_cnt[grid_xy[i][0], grid_xy[i][1]])
#         obj['geometry'] = dict()
#         obj['geometry']['type'] = 'Polygon'
#         obj['geometry']['coordinates'] = [grid]
#         obj_list.append(obj)
#     geojson['features'] = obj_list
#     json.dump(geojson, open('data/temp_geojson.json', 'w'))
#
#     gpd_geojson = gpd.read_file('data/temp_geojson.json')
#
#     gpd_geojson.plot('cnt', legend=True, cmap=plt.cm.Reds)
#     plt.title(name1)
#     plt.savefig('./save/test_result/{}_geojson.png'.format(name1))


def edit_distance(trace1, trace2):
    """
    the edit distance between two trajectory
    Args:
        trace1:
        trace2:
    Returns:
        edit_distance
    """
    matrix = [[i + j for j in range(len(trace2) + 1)] for i in range(len(trace1) + 1)]
    for i in range(1, len(trace1) + 1):
        for j in range(1, len(trace2) + 1):
            if trace1[i - 1] == trace2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(trace1)][len(trace2)]


def hausdorff_metric(truth, pred, distance='haversine'):
    """豪斯多夫距离
    ref: https://github.com/mavillan/py-hausdorff

    Args:
        truth: 经纬度点，(trace_len, 2)
        pred: 经纬度点，(trace_len, 2)
        distance: dist计算方法，包括haversine，manhattan，euclidean，chebyshev，cosine

    Returns:

    """
    return hausdorff.hausdorff_distance(truth, pred, distance=distance)


def haversine(array_x, array_y):
    R = 6378.0
    radians = np.pi / 180.0
    lat_x = radians * array_x[0]
    lon_x = radians * array_x[1]
    lat_y = radians * array_y[0]
    lon_y = radians * array_y[1]
    dlon = lon_y - lon_x
    dlat = lat_y - lat_x
    a = (pow(math.sin(dlat/2.0), 2.0) + math.cos(lat_x) * math.cos(lat_y) * pow(math.sin(dlon/2.0), 2.0))
    return R * 2 * math.asin(math.sqrt(a))


def dtw_metric(truth, pred, distance='haversine'):
    """动态时间规整算法
    ref: https://github.com/slaypni/fastdtw

    Args:
        truth: 经纬度点，(trace_len, 2)
        pred: 经纬度点，(trace_len, 2)
        distance: dist计算方法，包括haversine，manhattan，euclidean，chebyshev，cosine

    Returns:

    """
    if distance == 'haversine':
        distance, path = fastdtw(truth, pred, dist=haversine)
    elif distance == 'manhattan':
        distance, path = fastdtw(truth, pred, dist=cityblock)
    elif distance == 'euclidean':
        distance, path = fastdtw(truth, pred, dist=euclidean)
    elif distance == 'chebyshev':
        distance, path = fastdtw(truth, pred, dist=chebyshev)
    elif distance == 'cosine':
        distance, path = fastdtw(truth, pred, dist=cosine)
    else:
        distance, path = fastdtw(truth, pred, dist=euclidean)
    return distance


rad = math.pi / 180.0
R = 6378137.0


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Usage
    -----
    Compute the great circle distance, in meter, between (lon1,lat1) and (lon2,lat2)

    Parameters
    ----------
    param lat1: float, latitude of the first point
    param lon1: float, longitude of the first point
    param lat2: float, latitude of the se*cond point
    param lon2: float, longitude of the second point

    Returns
    -------x
    d: float
       Great circle distance between (lon1,lat1) and (lon2,lat2)
    """

    dlat = rad * (lat2 - lat1)
    dlon = rad * (lon2 - lon1)
    a = (math.sin(dlat / 2.0) * math.sin(dlat / 2.0) +
         math.cos(rad * lat1) * math.cos(rad * lat2) *
         math.sin(dlon / 2.0) * math.sin(dlon / 2.0))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def s_edr(t0, t1, eps):
    """
    Usage
    -----
    The Edit Distance on Real sequence between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    edr : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    # C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    C = np.full((n0 + 1, n1 + 1), np.inf)
    C[:, 0] = np.arange(n0 + 1)
    C[0, :] = np.arange(n1 + 1)
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if great_circle_distance(t0[i - 1][0], t0[i - 1][1], t1[j - 1][0], t1[j - 1][1]) < eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
    edr = float(C[n0][n1]) / max([n0, n1])
    return edr


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


def rid_cnt2heat_level(rid_cnt):
    min = 0
    max = np.max(rid_cnt)
    level_num = 100
    bin_size = max // level_num
    rid_heat_level = rid_cnt // bin_size
    return rid_heat_level

