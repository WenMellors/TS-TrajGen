# Continuous Trajectory Generation Based on Two-Stage GAN (TS-TrajGen)

We will open the source code soon.

## Abstract

Simulating the human mobility and generating large-scale trajectories are of great use in many real-world applications, such as urban planning, epidemic spreading analysis, and geographic privacy protect. Although many previous works have studied the problem of trajectory generation, the continuity of the generated trajectories has been neglected, which makes these methods useless for practical urban simulation scenarios. To solve this problem, we propose a novel two-stage generative adversarial framework to generate the continuous trajectory on the road network, namely TS-TrajGen, which efficiently integrates prior domain knowledge of human mobility with model-free learning paradigm. Specifically, we build the generator under the human mobility hypothesis of the A* algorithm to learn the human mobility behavior. For the discriminator, we combine the sequential reward with the mobility yaw reward to enhance the effectiveness of the generator. Finally, we propose a novel two-stage generation process to overcome the weak point of the existing stochastic generation process. Extensive experiments on two real-world datasets and two case studies demonstrate that our framework yields significant improvements over the state-of-the-art methods.

## Tutorial: How to apply TS-TrajGen to your own trajectory dataset


