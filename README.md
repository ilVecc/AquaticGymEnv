# AquaGym environment
The goal of this AI project is to build a gym environment that simulates the motion of a water
 drone with differential drive moving in a cluttered environment and subject to a superimposed
 non-deterministic water flow. The idea is that the water flow changes the drone trajectory and so
 correction is needed in order to avoid the collisions. One of the challenges is the continuous
  domain both of the actions and the state, which we must deal with carefulness. 

### Roadmap
1. Create an `N x M` world with obstacles (rectangles or circles) and constant water flow from
   direction `d`;
2. Model the environment with a state space spanned by a vector `[posX, posY, angle]`, where the
   angle is w.r.t. the "up" direction and an action space spanned by a vector `[thrustL, thrustR]` 
   (it's a differential drive, so we have two independent motors);
3. Provide a random policy and look at the behaviour of the overall;
4. Provide a reinforcement learning policy obtained via Deep-Q-Network;
5. Utilize the Safe Policy Improvement algorithm on the DQN policy in order to appreciate the
   powerfulness of the improvement approach. 

### Design choices
We define the domains for 
- the thrusters power in `[-10,+10]`;
- the angle in `[-π,+π]`;
- the world dimensions are `1000x1000`, so the position is in `[0,1000]x[0,1000]`; 
- the wave speed in `[-1,+1]` with a variance of `0.01`;
- the reward is `+100` when the goal is reached, `-100` when the drone hits an obstacle and `-0.1` 
  for each non-ending move; both collisions and reaching the goal are ending events;
     
For the discretization, each thruster has 5 distinct values (`-10, -5, 0, +5, +10`), for a total
 of `5x5=25` actions, and the position is given by rounding up the actual position, for a total
  of `1000x1000=1000000` states. 

### Install
The AquaGym environment can also be deployed as a pip package; simply type `pip install -e gym
 -aqua` in order to create the package.
