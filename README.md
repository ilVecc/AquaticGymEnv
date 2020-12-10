# AquaGym environment
The goal of this AI project is to build a gym environment that simulates the motion of a water
 drone with differential drive moving in a cluttered environment and subject to a superimposed
 non-deterministic water flow. The idea is that the water flow changes the drone trajectory and so a
 correction is needed in order to avoid the collisions. We offer two environments, differing in
  the domain of the action space:
- continuous, both thrusters with domain `[-0.3,+0.3]`
- discrete, offering 8 actions `[reverse, rotate left, rotate right, full throttle, reverse left, 
    full throttle left, reverse right, full throttle right]`

### Roadmap
1. Create an `N x M` world with obstacles (rectangles or circles) and constant water flow from
   direction `d`;
2. Model the environment with a state space spanned by a vector `[posX, posY, angle]`, where the
   angle is w.r.t. the "up" direction and an action space spanned by a vector `[thrustL, thrustR]` 
   (it's a differential drive, so we have two independent motors) in the continuous case;
3. Provide a random policy and look at the behaviour of the overall;
4. Provide a reinforcement learning policy obtained via Deep-Q-Network;
5. Utilize the Safe Policy Improvement algorithm on the DQN policy in order to appreciate the
   powerfulness of the improvement approach. 

### Design choices
We define the domains for 
- the thrusters power in `[-0.3,+0.3]`;
- the angle in `[-pi,+pi]`;
- the world dimensions are `100x100`, so the position is in `[0,100]x[0,100]`; 
- the wave speed in `[-0.05,+0.05]` with a variance of `0.001`;
- the reward is `+100` when the goal is reached, `-100` when hitting an obstacle or the border 
   and `-0.1` for each non-ending move; both collisions and reaching the goal are ending events;

### Install
The AquaGym environment can also be deployed as a pip package; simply type `pip install -e gym
 -aqua` in order to create the package.
