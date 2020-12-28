from policies.dqn_policy import DQNAquaPolicy
from plot_dqn_progress import PerformanceUtils
from utils import assure_exists

if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go
    
    policy_savepath = 'train_dqn_3'
    
    assure_exists(policy_savepath)
    
    # train network
    policy = DQNAquaPolicy(policy_savepath, load_network=False,
                           env_size='small', with_obstacles=False, with_waves=False)
    
    improve = True
    if improve:
        episodes_score = policy.train(debug=True, render=False)
        PerformanceUtils.show_recent(episodes_score, policy_savepath)
        PerformanceUtils.show_overall(episodes_score, policy_savepath)
    
    print("Training completed")
