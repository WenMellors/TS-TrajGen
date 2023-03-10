import json
import os
import argparse


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--local', type=str2bool,
                    default=True, help='whether save the trained model')
parser.add_argument('--dataset_name', type=str,
                    default='Xian')
args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name

if local:
    data_root = '../data/'
else:
    data_root = '/mnt/data/jwj/'


with open(os.path.join(data_root, dataset_name, 'rid_gps.json'), 'r') as f:
    road_gps = json.load(f)

lon_0 = 360
lat_0 = 90
lon_1 = -360
lat_1 = 0

for road in road_gps:
    lon, lat = road_gps[str(road)]
    lon_0 = min(lon, lon_0)
    lat_0 = min(lat, lat_0)
    lon_1 = max(lon, lon_1)
    lat_1 = max(lat, lat_1)

print('lon_0: {}, lon_1: {}, lat_0: {}, lat_1: {}'.format(lon_0, lon_1, lat_0, lat_1))