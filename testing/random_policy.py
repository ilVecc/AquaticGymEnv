import gym
import gym_aqua


def to_infinity():
    time = 0
    while True:
        time += 1
        yield time


if __name__ == '__main__':
    
    # we register the environments in gym_aqua/__init__.py
    # so now we're ready
    
    env = gym.make('AquaSmall-v1')
    env.reset()
    
    for i in to_infinity():
        env.render()
        
        state, reward, done, _ = env.step(env.action_space.sample())
        if done:
            if reward < 0:
                print("schiantati o troppe mosse")
            else:
                print("goal!")
            break

    env.close()
