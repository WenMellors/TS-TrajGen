# Continuous Trajectory Generation Based on Two-Stage GAN (TS-TrajGen)


## Abstract

Simulating the human mobility and generating large-scale trajectories are of great use in many real-world applications, such as urban planning, epidemic spreading analysis, and geographic privacy protect. Although many previous works have studied the problem of trajectory generation, the continuity of the generated trajectories has been neglected, which makes these methods useless for practical urban simulation scenarios. To solve this problem, we propose a novel two-stage generative adversarial framework to generate the continuous trajectory on the road network, namely TS-TrajGen, which efficiently integrates prior domain knowledge of human mobility with model-free learning paradigm. Specifically, we build the generator under the human mobility hypothesis of the A* algorithm to learn the human mobility behavior. For the discriminator, we combine the sequential reward with the mobility yaw reward to enhance the effectiveness of the generator. Finally, we propose a novel two-stage generation process to overcome the weak point of the existing stochastic generation process. Extensive experiments on two real-world datasets and two case studies demonstrate that our framework yields significant improvements over the state-of-the-art methods.

## OpenSource Dataset
The two Dataset _BJ-Taxi_ and _Porto-Taxi_ can be download at [Baidu Pan with code 5608](https://pan.baidu.com/s/1Im0g15cFfdlJ57Q6diMmzQ?pwd=5608). Each dataset has following 3 file:
* `xx.geo`: the geo file which stores the road segment infomation of the road network.
* `xx.rel`: the rel file which stores the adjacent information between road segments.
* `xx.csv`: the map matched trajectory file.

Note: the `geo` and `rel` file format is defined by our [LibCity](https://github.com/LibCity/Bigscity-LibCity) project, you can refer more information about this two file in [docs](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html). In fact, these two files are still in CSV format (you can load it by `pd.read_csv`), but we have defined different suffixes to distinguish them.

## Cite

 W. Jiang, W. X. Zhao, J. Wang, and J. Jiang, "Continuous trajectory generation based on two-stage GAN," in Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI’23), 2023.

## Tutorial: How to apply TS-TrajGen to your own trajectory dataset

* prepare your map matched trajectory composed of road segments, and corresponding road network information. The road network information store as atomic files format (.geo and .rel) defined by [LibCity](https://github.com/LibCity/Bigscity-LibCity), which is our previous work. The detailed data format can be referred at [Docs](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html) 
  * For example, you may have:
    * `xianshi_mm_train.csv`: the map matched trajectory for training.
    * `xianshi_mm_test.csv`: the map matched trajectory for testing.
    * `xian.geo`: the geo file of xian road network.
    * `xian.rel`: the rel file of xian road network.
* run `/script/preprocess_pretrain_input.py`
* Now you can pretrain road-level generator with Imitate Learning paradigm.
  * run `pretrain_gat_fc.py` for function H.
  * run `pretrain_function_g_fc.py` for function G.
* For the region-level, you should build the region structures first. We build regions using [KaHIP](https://github.com/KaHIP/KaHIP), which is a graph partition library.
  * run `/script/process_kahip_graph_format.py` to generate KaHIP's input.
  * use `./build/kaffpa ./data/xian.graph --k 100 --preconfiguration=strong` to conduct graph partition
  * run `/script/process_kaffpa_res.py` to process KaHIP's output and generate regions.
  * run `/script/construct_traffic_zone_relation.py` to calculate regions' adjacent relationships.
  * run `/script/map_region_traj.py` to map the road-level traj to region level.
  * run `/script/encode_region_traj.py` to encode the region-level trajectories to pretrain input of models.
  * run `/script/prepare_region_feature.py` to calculate region GAT node feature based on road-level node feature.
  * run `/script/construct_region_dist.py` to calculate gps distance between regions.
  * run `pretrain_region_function_g_fc.py` to pretrain region-level function G.
  * run `pretrain_region_gat_fc.py` to pretrain region-level function H.
* run `train_gan.py` and `train_region_gan.py` to adversarial learning.
  * Note that the GAN may not work very well in the new dataset, but our pretrain mechanism with Imitate Learning is already powerful.
* run `our_model_generate.py` to generate trajectories based on the OD-input from the test dataset, .e.g, `xianshi_mm_test.csv`.
