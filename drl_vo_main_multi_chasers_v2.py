import argparse
import yaml
import numpy as np
import random
import os
import torch
import time
from algos.SAC_AE.sac_ae import SacAeAgent
from algos.SAC_AE.utils import ReplayBuffer, eval_policy_mode, set_seed_everywhere, RatioCombinedReplayBuffer
from info import *
from env_drl_vo_adv import CrowdSim
from env_drl_vo_chaser_ben import CrowdSimOrcaNavigator
from test_chaser import eval_chaser_policy
from torch.utils.tensorboard import SummaryWriter
from threading import Lock, Thread
from tqdm import tqdm


def load_yaml_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML config file
        
    Returns:
        dict: Configuration dictionary, empty dict if file not found or invalid
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config file: {e}")
        return {}


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, chaser_agent, eval_or_test_env, current_steps, other_chaser_agents=None, 
                eval_episodes=100, save_directory=None, 
                if_save_video=False, final_test=False):
    avg_reward = 0.
    avg_chaser_reward = 0.
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    escape = 0
    collision = 0
    timeout = 0
    collision_cases = []
    timeout_cases = []
    KL_divergence = np.zeros((eval_episodes, len(other_chaser_agents))) if other_chaser_agents is not None else None
    if_save_data = False
    for i in range(eval_episodes):
        kl_divergence = np.zeros((eval_or_test_env.max_episode_step, len(other_chaser_agents))) if other_chaser_agents is not None else None
        if_save_data = (i < 10 or final_test)
        if final_test:
            obs, robot_goal_emotion_state, chaser_obs, chaser_goal_emotion_state = eval_or_test_env.reset(save_data=if_save_data)
        else:
            obs, robot_goal_emotion_state, chaser_obs, chaser_goal_emotion_state = eval_or_test_env.reset(save_data=if_save_data)
        # eval_or_test_env.render()
        # time.sleep(0.2)
        done = False
        ep_step = 0
        
        while ep_step < eval_or_test_env.max_episode_step:
            # t1 = time.time()
            with eval_policy_mode(policy):
                action = policy.select_action(obs, robot_goal_emotion_state)
            with eval_policy_mode(chaser_agent):
                chaser_action = chaser_agent.select_action(chaser_obs, chaser_goal_emotion_state)
            # action = eval_or_test_env.dwa_compute_action()
            obs, robot_goal_emotion_state, reward, done, info, chaser_obs, chaser_goal_emotion_state, chaser_reward, chaser_info = eval_or_test_env.step(action, chaser_action, eval=True, save_data=if_save_data)
            if other_chaser_agents is not None:
                for j, other_chaser_agent in enumerate(other_chaser_agents):
                    mu_p,log_std_p = chaser_agent.output_distribution(obs, robot_goal_emotion_state)
                    mu_q,log_std_q = other_chaser_agent.output_distribution(chaser_obs, chaser_goal_emotion_state)
                    var_p = np.power(log_std_p, 2)
                    var_q = np.power(log_std_q, 2)
                    kl_divergence[i,j] = (log_std_q - log_std_p + (var_p + np.power(mu_p-mu_q,2)) / (2.0 * var_q) - 0.5).sum()

            # print('time: ', time.time() - t1) # 7.5ms
            # eval_or_test_env.render()
            # time.sleep(0.2)
            avg_reward += reward
            avg_chaser_reward += chaser_reward
            ep_step = ep_step + 1
            if done or ep_step == eval_or_test_env.max_episode_step or isinstance(info, ReachGoal):
                if kl_divergence is not None:
                    KL_divergence[i] = kl_divergence.sum(axis=0)
                if ep_step == eval_or_test_env.max_episode_step:
                    timeout += 1
                    timeout_cases.append(i)
                    timeout_times.append(eval_or_test_env.time_limit)
                    print('evaluation episode ' + str(i) + ', time out: ' + str(ep_step))
                else:
                    if isinstance(info, ReachGoal):
                        escape += 1
                        print('evaluation episode ' + str(i) + ', goal reaching at evaluation step: ' + str(ep_step))
                    elif isinstance(chaser_info, ReachGoal):
                        success += 1
                        success_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', chaser catch the robot at evaluation step: ' + str(ep_step))
                    elif isinstance(info, Collision):
                        collision += 1
                        collision_cases.append(i)
                        collision_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', collision occur at evaluation step: ' + str(ep_step))  
                    elif isinstance(info, DigitCrazy):
                        collision += 1
                        collision_cases.append(i)
                        collision_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', crazy digit at evaluation step: ' + str(ep_step))     
                   
                    elif isinstance(chaser_info, Collision):
                        collision += 1
                        collision_cases.append(i)
                        collision_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', chaser collision occur at evaluation step: ' + str(ep_step))       
                break
        if save_directory is not None:
            if isinstance(chaser_info, ReachGoal):
                file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '.npz'
            else:
                file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '_fail' + '.npz'
            if if_save_data:
                np.savez_compressed(file_name, **eval_or_test_env.log_env)
                if if_save_video:
                    eval_or_test_env.save_video(current_steps, i)
                # policy.save_features(save_directory)

    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    escape_rate = escape / eval_episodes
    assert success + collision + timeout + escape == eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_or_test_env.time_limit
    KL_divergence = np.mean(KL_divergence, axis=0) if other_chaser_agents is not None else None

    if other_chaser_agents is None:
        return success_rate, collision_rate, escape_rate, avg_nav_time
    else:
        return success_rate, collision_rate, escape_rate, avg_nav_time, KL_divergence
    
def prefilling_replay_buffer(agent, chaser_agent, buffer, env, size):
    # Import tqdm for progress bar
    
    # Sample the chaser success episodes to fill the buffer
    pbar = tqdm(total=size, desc="Prefilling buffer")
    success_count = 0
    total_count = 0
    added_transitions = 0
    
    while added_transitions < size:
        obs, robot_goal_emotion_state, chaser_obs, chaser_goal_emotion_state = env.reset()
        done = False
        episode_transitions = []
        ep_step = 0
        
        while ep_step < env.max_episode_step:
            with eval_policy_mode(agent):
                action = agent.select_action(obs, robot_goal_emotion_state)
            with eval_policy_mode(chaser_agent):
                chaser_action = chaser_agent.select_action(chaser_obs, chaser_goal_emotion_state)
            
            next_obs, next_robot_goal_emotion_state, reward, done, info, next_chaser_obs, next_chaser_goal_emotion_state, chaser_reward, chaser_info = env.step(action, chaser_action)
            
            # Store transition for potential addition to buffer
            is_terminal = float(done) if ep_step < env.max_episode_step - 1 else 0.0
            episode_transitions.append((chaser_obs, chaser_goal_emotion_state, chaser_action, chaser_reward, next_chaser_obs, next_chaser_goal_emotion_state, is_terminal))
            
            # Update current states
            obs = next_obs
            robot_goal_emotion_state = next_robot_goal_emotion_state
            chaser_obs = next_chaser_obs
            chaser_goal_emotion_state = next_chaser_goal_emotion_state
            
            if done or ep_step == env.max_episode_step - 1 or done == True:
                total_count += 1
                break
            ep_step += 1
        
        # If chaser caught the robot or make a collision, add all transitions from this episode(chaser related)
        if isinstance(chaser_info, ReachGoal) or isinstance(chaser_info, Collision):
            success_count += 1
            for transition in episode_transitions:
                if added_transitions < size:
                    buffer.add(*transition)
                    added_transitions += 1
                    pbar.update(1)
                else:
                    break
        
        # Update progress bar
        pbar.set_postfix({'effective_rate': f"{success_count/(total_count):.2%}", 'filled': f"{added_transitions}/{size}"})
    
    pbar.close()
    print(f"Prefilling complete. Added {added_transitions} transitions from {success_count} successful episodes to buffer.")


def main():
    parser = argparse.ArgumentParser()

    # Add config file option
    parser.add_argument('--config', default='', type=str, help='Path to YAML config file')
    
    # Parse just the config argument first
    config_args, _ = parser.parse_known_args()
    
    # Load config from YAML if provided
    config = {}
    if config_args.config:
        config = load_yaml_config(config_args.config)
    
    # Set defaults from config for all other arguments
    parser.add_argument('--policy', default=config.get('policy', 'drl_vo'), type=str)
    parser.add_argument('--chaser_policy', default=config.get('chaser_policy', 'drl_vo'), type=str)
    # options, differential, lip, digit_mujoco
    parser.add_argument('--robot_model', default=config.get('robot_model', 'differential'), type=str)
    parser.add_argument('--robot_eval_model', default=config.get('robot_eval_model', 'differential'), type=str)
    # options, differential, lip, digit_mujoco
    parser.add_argument('--robot_test_model', default=config.get('robot_test_model', 'differential'), type=str)
    parser.add_argument('--chaser_model', default=config.get('chaser_model', 'differential'), type=str)
    parser.add_argument('--chaser_eval_model', default=config.get('chaser_eval_model', 'differential'), type=str)
    parser.add_argument('--chaser_test_model', default=config.get('chaser_test_model', 'differential'), type=str)
    parser.add_argument('--use_angular', action='store_true', default=config.get('use_angular', False))
    parser.add_argument('--random_radius', action='store_true', default=config.get('random_radius', True))
    parser.add_argument('--training_object', default=config.get('training_object', 'chaser'), type=str)
    #environment settings
    parser.add_argument('--random_chaser_reset', action='store_true', default=config.get('random_chaser_reset', False))
    parser.add_argument('--random_robot_goal', action='store_true', default=config.get('random_robot_goal', False))
    # device
    parser.add_argument('--device', type=str, default=config.get('device', 'cuda'))
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--seed', default=config.get('seed', 1), type=int)
    # Time steps initial random policy is used
    parser.add_argument('--start_timesteps', default=config.get('start_timesteps', 10000), type=int)
    # How often (time steps) we evaluate
    parser.add_argument('--eval_freq', default=config.get('eval_freq', 20000), type=int)
    # How often (time steps) we save the trained model
    parser.add_argument('--save_model_freq', default=config.get('save_model_freq', 200000), type=int)
    # Max time steps to run environment
    parser.add_argument('--max_timesteps', default=config.get('max_timesteps', 6e6), type=int)
    # replay buffer
    parser.add_argument("--replay_buffer_capacity", default=config.get('replay_buffer_capacity', 150000), type=int)
    parser.add_argument("--prefilling_size", type=int, default=config.get('prefilling_size', 10000))
    parser.add_argument("--prefill_ratio", default=config.get('prefill_ratio', [0.8, 0.2]), type=float, nargs='+')
    # training
    parser.add_argument('--batch_size', default=config.get('batch_size', 128), type=int)
    parser.add_argument('--hidden_dim', default=config.get('hidden_dim', 1024), type=int)
    # critic
    parser.add_argument('--critic_lr', default=config.get('critic_lr', 1e-3), type=float)
    parser.add_argument('--critic_beta', default=config.get('critic_beta', 0.9), type=float)
    parser.add_argument('--critic_tau', default=config.get('critic_tau', 0.01), type=float)
    parser.add_argument('--critic_target_update_freq', default=config.get('critic_target_update_freq', 2), type=int)
    # actor
    parser.add_argument('--actor_lr', default=config.get('actor_lr', 1e-3), type=float)
    parser.add_argument('--actor_beta', default=config.get('actor_beta', 0.9), type=float)
    parser.add_argument('--actor_log_std_min', default=config.get('actor_log_std_min', -10), type=float)
    parser.add_argument('--actor_log_std_max', default=config.get('actor_log_std_max', 2), type=float)
    parser.add_argument('--actor_update_freq', default=config.get('actor_update_freq', 2), type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default=config.get('encoder_type', 'pixel'), type=str)
    parser.add_argument('--encoder_feature_dim', default=config.get('encoder_feature_dim', 50), type=int)
    parser.add_argument('--encoder_lr', default=config.get('encoder_lr', 1e-3), type=float)
    parser.add_argument('--encoder_tau', default=config.get('encoder_tau', 0.05), type=float)
    # pixel: with decoder, identity: without decoder
    parser.add_argument('--decoder_type', default=config.get('decoder_type', 'identity'), type=str)
    parser.add_argument('--decoder_lr', default=config.get('decoder_lr', 1e-3), type=float)
    parser.add_argument('--decoder_update_freq', default=config.get('decoder_update_freq', 1), type=int)
    parser.add_argument('--decoder_latent_lambda', default=config.get('decoder_latent_lambda', 1e-6), type=float)
    parser.add_argument('--decoder_weight_lambda', default=config.get('decoder_weight_lambda', 1e-7), type=float)
    parser.add_argument('--num_layers', default=config.get('num_layers', 4), type=int)
    parser.add_argument('--num_filters', default=config.get('num_filters', 32), type=int)
    # sac
    parser.add_argument('--discount', default=config.get('discount', 0.99), type=float)
    parser.add_argument('--init_temperature', default=config.get('init_temperature', 0.1), type=float)
    parser.add_argument('--alpha_lr', default=config.get('alpha_lr', 1e-4), type=float)
    parser.add_argument('--alpha_beta', default=config.get('alpha_beta', 0.5), type=float)

    # Model load file name, "" doesn't load, "default" uses file_name
    # args.load_model, format, step_NO_success_NO
    # parser.add_argument("--load_model", type=str, default="step_2740000_success_90")
    # by sending the previous model path, it is default to prefill the replay buffer with interaction with the previous model
    parser.add_argument("--itr", type=int, default=config.get('itr', 1))
    parser.add_argument("--load_model_path", type=str, default=config.get('load_model_path', ""))
    parser.add_argument("--load_pre_model_path", type=str, default=config.get('load_pre_model_path', ""))
    parser.add_argument("--load_chaser", action='store_true', default=config.get('load_chaser', False))
    parser.add_argument("--load_test_path", type=str, default=config.get('load_test_path', ""))
    # environment settings
    parser.add_argument("--action_dim", type=int, default=config.get('action_dim', 2))
    parser.add_argument("--lidar_dim", type=int, default=config.get('lidar_dim', 1800))
    parser.add_argument("--lidar_feature_dim", type=int, default=config.get('lidar_feature_dim', 50))
    # lidar to image
    parser.add_argument('--image_size', default=config.get('image_size', 100), type=int)
    parser.add_argument('--frame_stack', default=config.get('frame_stack', 10), type=int)
    # 2 robot speed, 2 local goal
    parser.add_argument("--robot_goal_state_dim", type=int, default=config.get('robot_goal_state_dim', 4))
    parser.add_argument("--laser_angle_resolute", type=float, default=config.get('laser_angle_resolute', 0.003490659))
    parser.add_argument("--laser_min_range", type=float, default=config.get('laser_min_range', 0.27))
    parser.add_argument("--laser_max_range", type=float, default=config.get('laser_max_range', 6.0))
    parser.add_argument("--human_num_max", type=int, default=config.get('human_num_max', 4))
    parser.add_argument("--static_obstacle_num_max", type=int, default=config.get('static_obstacle_num_max', 3))
    parser.add_argument("--square_width", type=float, default=config.get('square_width', 10.0))
    parser.add_argument("--chaser_num", type=int, default=config.get('chaser_num', 1))
    parser.add_argument("--timeout_penalty", type=float, default=config.get('timeout_penalty', -0.6))
    args = parser.parse_args()

    #get chaser models list
    # chaser_models_path = [
    #     '/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-03-13/seed_1/itr_1/chaser_models/chaser_0/step_1680000chaser_0_success_62',
    #     '/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-03-13/seed_1/itr_1/chaser_models/chaser_1/step_1680000chaser_1_success_70',
    #     '/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-03-13/seed_1/itr_1/chaser_models/chaser_2/step_1680000chaser_2_success_61'
    # ]
    # Use chaser models from config if available
    # chaser_models_path = config.get('chaser_models', [
    #     '/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-03-13/seed_1/itr_1/chaser_models/chaser_0/step_1680000chaser_0_success_62',
    #     '/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-03-13/seed_1/itr_1/chaser_models/chaser_1/step_1680000chaser_1_success_70',
    #     '/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-03-13/seed_1/itr_1/chaser_models/chaser_2/step_1680000chaser_2_success_61'
    # ])
    chaser_models_path =  ['/mnt/ssd1/stilrmy/Navigation/drl_vo_1_adversaries/2025-07-05/seed_1/itr_1/chaser_models/chaser_0/step_140000_chaser_0_success_61_benchmark_64']

    print("---------------------------------------")
    print(f"Policy: {args.policy}, RobotModel: {args.robot_model}, RobotEvalModel: {args.robot_eval_model}, Seed: {args.seed}")
    print("---------------------------------------")
    date = time.strftime("%Y-%m-%d")
    file_prefix = '/mnt/ssd1/stilrmy/Navigation/' + args.policy + '_' + str(args.chaser_num) + '_adversaries' + '/' + date
    file_prefix = file_prefix + '/seed_' + str(args.seed) + '/itr_' + str(args.itr)  

    
    file_results = file_prefix + '/results'
    file_models = file_prefix + '/models'
    file_chaser_models = file_prefix + '/chaser_models'
    file_evaluation_episodes = file_prefix + '/evaluation_episodes'
    file_final_test_episodes = file_prefix + '/final_test_episodes'
    file_buffer = file_prefix + '/buffer'
    file_chaser_buffer = file_prefix + '/chaser_buffer'


    if not os.path.exists(file_results):
        os.makedirs(file_results)

    if not os.path.exists(file_models):
        os.makedirs(file_models)

    if not os.path.exists(file_chaser_models):
        os.makedirs(file_chaser_models)

    if not os.path.exists(file_evaluation_episodes):
        os.makedirs(file_evaluation_episodes)
        
    if not os.path.exists(file_final_test_episodes):
        os.makedirs(file_final_test_episodes)
        
    if not os.path.exists(file_buffer):
        os.makedirs(file_buffer)

    if not os.path.exists(file_chaser_buffer):
        os.makedirs(file_chaser_buffer)


    writer = SummaryWriter(log_dir=file_results)

    action_range = np.array([[-0.5, -1.5],
                             [1.0,  1.5]])
    
    action_resolution = [0.05, 0.1]
    action_num = [int((action_range[1, 0] - action_range[0, 0]) / action_resolution[0]) + 1,
                  int((action_range[1, 1] - action_range[0, 1]) / action_resolution[1]) + 1]
    action_choice_1 = np.linspace(action_range[0, 0], action_range[1, 0], num=action_num[0])
    action_choice_2 = np.linspace(action_range[0, 1], action_range[1, 1], num=action_num[1])
    action_choice_dim = action_num[0] * action_num[1]
    action_choices = np.zeros((action_num[0], action_num[1], 2), dtype=np.float32)
    for i in range(action_num[0]):
        for j in range(action_num[1]):
            action_choices[i, j, :] = np.array([action_choice_1[i], action_choice_2[j]]) 
    action_choices = np.reshape(action_choices, (action_choice_dim, 2))

    chaser_action_range = np.array([[-0.5, -1.5],
                             [1.0,  1.5]])
    chaser_action_resolution = [0.05, 0.1]
    chaser_action_num = [int((chaser_action_range[1, 0] - chaser_action_range[0, 0]) / chaser_action_resolution[0]) + 1,
                    int((chaser_action_range[1, 1] - chaser_action_range[0, 1]) / chaser_action_resolution[1]) + 1]
    chaser_action_choice_1 = np.linspace(chaser_action_range[0, 0], chaser_action_range[1, 0], num=chaser_action_num[0])
    chaser_action_choice_2 = np.linspace(chaser_action_range[0, 1], chaser_action_range[1, 1], num=chaser_action_num[1])
    chaser_action_choice_dim = chaser_action_num[0] * chaser_action_num[1]
    chaser_action_choices = np.zeros((chaser_action_num[0], chaser_action_num[1], 2), dtype=np.float32)
    for i in range(chaser_action_num[0]):
        for j in range(chaser_action_num[1]):
            chaser_action_choices[i, j, :] = np.array([chaser_action_choice_1[i], chaser_action_choice_2[j]])
    chaser_action_choices = np.reshape(chaser_action_choices, (chaser_action_choice_dim, 2))
    
    if args.robot_model == 'digit_mujoco':
        cfg_digit_env_train = DigitEnvConfig()
        digit_env_train = DigitEnvFlat(cfg_digit_env_train, file_evaluation_episodes)
        env = CrowdSim(args, action_range, action_choices=action_choices, digit_env=digit_env_train)
        if_save_video = True
    elif args.robot_model == 'lip':
        env = CrowdSim(args, action_range, action_choices=action_choices)
        if_save_video = False
    elif args.robot_model == 'differential':
        env = CrowdSim(args, action_range, chaser_action_range, action_choices=action_choices, chaser_action_choices=chaser_action_choices)
        if_save_video = False
    elif args.robot_model == 'ackermann':
        env = CrowdSim(args, action_range, chaser_action_range, action_choices=action_choices, chaser_action_choices=chaser_action_choices)
        if_save_video = False
    else:
        raise NotImplementedError(args.robot_model)
    
    chaser_ben = CrowdSimOrcaNavigator(args, action_range, chaser_action_range)
    
    if args.robot_eval_model == 'digit_mujoco':
        cfg_digit_env_eval = DigitEnvConfig()
        cfg_digit_env_eval.vis_record.visualize = True
        cfg_digit_env_eval.vis_record.record = True
        digit_env_eval = DigitEnvFlat(cfg_digit_env_eval, file_evaluation_episodes)
        
        eval_env = CrowdSim(args, action_range, action_choices=action_choices, digit_env=digit_env_eval)
        if_save_video = True
    elif args.robot_eval_model == 'lip':
        eval_env = CrowdSim(args, action_range, action_choices=action_choices)
        if_save_video = False
    elif args.robot_eval_model == 'differential':
        eval_env = CrowdSim(args, action_range, chaser_action_range, action_choices=action_choices, chaser_action_choices=chaser_action_choices)
        if_save_video = False
    elif args.robot_eval_model == 'ackermann':
        eval_env = CrowdSim(args, action_range, chaser_action_range, action_choices=action_choices, chaser_action_choices=chaser_action_choices)
        if_save_video = False
    else:
        raise NotImplementedError(args.robot_model)

    if args.robot_test_model == 'digit_mujoco':
        cfg_digit_env_test = DigitEnvConfig()
        cfg_digit_env_test.vis_record.visualize = True
        cfg_digit_env_test.vis_record.record = True
        digit_env_test = DigitEnvFlat(cfg_digit_env_test, file_final_test_episodes)
        
        if_save_video_test = True
        test_env = CrowdSim(args, action_range, action_choices=action_choices, digit_env=digit_env_test)
    elif args.robot_test_model == 'lip':
        if_save_video_test = False
        test_env = CrowdSim(args, action_range, action_choices=action_choices)
    elif args.robot_test_model == 'differential':
        if_save_video_test = False
        test_env = CrowdSim(args, action_range, chaser_action_range, action_choices=action_choices, chaser_action_choices=chaser_action_choices)
    elif args.robot_test_model == 'ackermann':
        if_save_video_test = False
        test_env = CrowdSim(args, action_range, chaser_action_range, action_choices=action_choices, chaser_action_choices=chaser_action_choices)
    else:
        raise NotImplementedError(args.robot_test_model)
    # please manually set seeds when test with digit_arsim
    # otherwise, set it as args.seed
    # set_seed_everywhere(args.seed)
    set_seed_everywhere(args.seed)

    device = torch.device(args.device)
    obs_shape = (3, args.image_size, args.image_size)
    robot_goal_state_dim = args.robot_goal_state_dim
    action_shape = (2,)
    chaser_agents = []
    for i in range(args.chaser_num):
        set_seed_everywhere(args.seed + i)
        chaser_agents.append(SacAeAgent(
            obs_shape,
            robot_goal_state_dim,
            action_shape,
            chaser_action_range,
            device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_latent_lambda=args.decoder_latent_lambda,
            decoder_weight_lambda=args.decoder_weight_lambda,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        ))
    

    agent = SacAeAgent(
            obs_shape,
            robot_goal_state_dim,
            action_shape,
            action_range,
            device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_latent_lambda=args.decoder_latent_lambda,
            decoder_weight_lambda=args.decoder_weight_lambda,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    

    
    
    replay_buffer = ReplayBuffer(
        obs_shape,
        robot_goal_state_dim, 
        action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        obs_dtype=np.float32
    )

    chaser_replay_buffers = []
    for i in range(args.chaser_num):
        chaser_replay_buffer = ReplayBuffer(
            obs_shape,
            robot_goal_state_dim, 
            action_shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device,
            obs_dtype=np.float32
        )
        chaser_replay_buffers.append(chaser_replay_buffer)

    prefilled_replay_buffers = []
    prefill_ratio = args.prefill_ratio
    for i in range(args.chaser_num):
        prefilled_replay_buffer = ReplayBuffer(
            obs_shape,
            robot_goal_state_dim, 
            action_shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device,
            obs_dtype=np.float32
        )
        prefilled_replay_buffers.append(prefilled_replay_buffer)

    checkpoint_steps = 0



    if args.load_model_path != "":
        # replay_buffer.load(file_buffer)
        agent.load(args.load_model_path)
        # args.load_model, format, step_NO_success_NO
        # extract the first number
        print("robot model loaded")
    # Check what's loaded on cuda:7 after model loading
    
   
    if args.load_test_path != "":
        print('start to test')
        agent.load(args.load_test_path)
        test_times = 200
        current_step = 0
        success_rate, collision_rate, avg_nav_time = eval_policy(agent, test_env, current_step, 
                                                                 eval_episodes=test_times, save_directory=file_final_test_episodes, 
                                                                 if_save_video=if_save_video_test, final_test=True)
        print('success_rate, collision_rate, avg_nav_time')
        print(success_rate, collision_rate, avg_nav_time)
        return

    if args.load_chaser:
        for i in range(args.chaser_num):
            chaser_agents[i].load(chaser_models_path[i])
        print("chaser model loaded")

    if args.load_pre_model_path != "":
        pre_agent = SacAeAgent(
            obs_shape,
            robot_goal_state_dim,
            action_shape,
            action_range,
            device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_latent_lambda=args.decoder_latent_lambda,
            decoder_weight_lambda=args.decoder_weight_lambda
        )
        pre_agent.load(args.load_pre_model_path)
        print("pre_agent model loaded")
        print("prefilling replay buffer")
        for i in range(args.chaser_num):
            chaser_agent = chaser_agents[i]
            prefill_replay_buffer = prefilled_replay_buffers[i]
            prefilling_replay_buffer(pre_agent, chaser_agent, prefill_replay_buffer, env, args.prefilling_size)
        print("prefilling done")

    #pre-evaluation
    # success_rate, collision_rate, escape_rate, avg_nav_time = eval_policy(agent, chaser_agents[0], eval_env, 0,
    #                                                          eval_episodes=100, save_directory=file_evaluation_episodes,
    #                                                          if_save_video=if_save_video, final_test=False)
    # print('success_rate, collision_rate, escape_rate, avg_nav_time')
    # print(success_rate, collision_rate, escape_rate, avg_nav_time)
    # for i in range(args.chaser_num):
    #     results = eval_chaser_policy(chaser_agents[i], chaser_ben, 100,save_directory=file_evaluation_episodes, mode = 'silent')
    #     print('benchmark chaser{} success_rate, collision_rate, avg_nav_time'.format(i))
    #     print(results['catch_rate'], results['collision_rate'], results['avg_chase_time'])
    # #testing code


    
    evaluations = []
    for i in range(args.chaser_num):
        evaluations.append([])

    # Temporary storage for transitions within the current episode for each chaser
    episode_transitions_storage = {i: [] for i in range(args.chaser_num)}

    obs, robot_goal_emotion_state, chaser_obs, chaser_goal_emotion_state = env.reset()
    done = False
    episode_reward = 0
    episode_chaser_reward = 0
    episode_timesteps = 0
    episode_num = 0
    prev_prefill_ratio = args.prefill_ratio
    transition_speed = 0.2 
    idx = np.random.randint(args.chaser_num)
    # Keep track of the chaser index for the current episode
    current_episode_idx = idx
    # No need to select buffer here, it's done when adding
    # chaser_agent = chaser_agents[idx]
    # chaser_replay_buffer = chaser_replay_buffers[idx]
    writer_lock = Lock()
    for t in range(checkpoint_steps + 1, int(args.max_timesteps) + 1):
        if t == args.start_timesteps:
            print('replay buffer has been initialized')

        # Select the current chaser agent based on the episode index
        chaser_agent = chaser_agents[current_episode_idx]

        # Perform action
        # sample action for data collection
        if t < args.start_timesteps and args.load_chaser == False:
            if args.load_chaser:
                with eval_policy_mode(chaser_agent):
                    chaser_action = chaser_agent.sample_action(chaser_obs, chaser_goal_emotion_state)
            else:
                chaser_action = np.random.uniform(chaser_action_range[0], chaser_action_range[1])
            if args.load_model_path != "":
                with eval_policy_mode(agent):
                    action = agent.sample_action(obs, robot_goal_emotion_state)
            else:
                # If robot model isn't loaded, maybe use random action or handle error
                # action = env.action_space.sample() # Example if env has action_space
                raise ValueError('Please load robot model or define behavior for start_timesteps')
        else:
            with eval_policy_mode(agent):
                action = agent.sample_action(obs, robot_goal_emotion_state)
            # Chaser agent is already selected based on current_episode_idx
            with eval_policy_mode(chaser_agent):
                chaser_action = chaser_agent.sample_action(chaser_obs, chaser_goal_emotion_state)

        next_obs, next_robot_goal_emotion_state, reward, done, info, next_chaser_obs, next_chaser_goal_emotion_state, chaser_reward, chaser_info = env.step(action, chaser_action)

        episode_timesteps += 1

        if episode_timesteps == env.max_episode_step:
            done_bool = 0.0
            chaser_reward = args.timeout_penalty
        else:
            done_bool = float(done)

        # Store data temporarily for the current episode's chaser
        transition = (chaser_obs, chaser_goal_emotion_state, chaser_action, chaser_reward, next_chaser_obs, next_chaser_goal_emotion_state, done_bool)
        episode_transitions_storage[current_episode_idx].append(transition)

        # --- Remove the old buffer adding logic here ---

        obs = next_obs
        robot_goal_emotion_state = next_robot_goal_emotion_state
        episode_reward += reward
        chaser_obs = next_chaser_obs
        chaser_goal_emotion_state = next_chaser_goal_emotion_state
        episode_chaser_reward += chaser_reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            num_updates = args.start_timesteps if t == args.start_timesteps and not args.load_chaser else 1

            # Define a function for the thread to execute
            def update_agent(agent_to_update, buffer, agent_id):
                for _ in range(num_updates):
                    # Check if buffer has enough samples before updating
                    with writer_lock:
                        agent_to_update.update(buffer, writer, t, agent_id)
                    # else: # Optional: print a warning or skip update
                    #    print(f"Skipping update for chaser {agent_id} at step {t}, buffer size {len(buffer)} < batch size {args.batch_size}")


            # Create and start threads for each chaser agent
            threads = []
            for i in range(args.chaser_num):
                if args.load_pre_model_path != "" and args.prefill_ratio[0] > 0.2:
                    buffer = RatioCombinedReplayBuffer([prefilled_replay_buffers[i], chaser_replay_buffers[i]], args.batch_size,prefill_ratio)
                else:
                    buffer = chaser_replay_buffers[i]
                thread = Thread(
                    target=update_agent,
                    args=(chaser_agents[i], buffer, i)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

        if done or episode_timesteps == env.max_episode_step:

            chaser_replay_buffer = chaser_replay_buffers[current_episode_idx]
            transitions_to_add = episode_transitions_storage[current_episode_idx]
            for trans in transitions_to_add:
                chaser_replay_buffer.add(*trans)

            # Print episode end reason
            if episode_timesteps == env.max_episode_step:
                print(f'total step {t}, train episode {episode_num+1}, chaser {current_episode_idx}, time out: {episode_timesteps}')
            else:
                if isinstance(info, ReachGoal):
                    print(f'total step {t}, train episode {episode_num+1}, chaser {current_episode_idx}, goal reaching at train step: {episode_timesteps}')
                elif isinstance(chaser_info, ReachGoal):
                    print(f'total step {t}, train episode {episode_num+1}, chaser {current_episode_idx}, chaser catch navigator at train step: {episode_timesteps}')
                elif isinstance(info, Collision):
                    print(f'total step {t}, train episode {episode_num+1}, chaser {current_episode_idx}, collision occur at train step: {episode_timesteps}')
                elif isinstance(chaser_info, Collision):
                    print(f'total step {t}, train episode {episode_num+1}, chaser {current_episode_idx}, chaser collision occur at train step: {episode_timesteps}')
                else:
                    # This case might indicate an issue if not ReachGoal or Collision
                    print(f'total step {t}, train episode {episode_num+1}, chaser {current_episode_idx}, ended at step: {episode_timesteps} with info: {info}, chaser_info: {chaser_info}')
                    # raise ValueError('Invalid end signal from environment') # Or handle appropriately

            # Reset environment
            obs, robot_goal_emotion_state, chaser_obs, chaser_goal_emotion_state = env.reset()
            done = False
            episode_num += 1
            # Log reward for the chaser that just finished the episode
            writer.add_scalar(f'train/chaser_{current_episode_idx}_episode_reward', episode_reward, episode_num)
            writer.add_scalar(f'train/chaser_{current_episode_idx}_episode_chaser_reward', episode_chaser_reward, episode_num)

            # Clear transitions for the finished episode
            episode_transitions_storage[current_episode_idx] = []

            # Select next chaser for the new episode *after* processing the previous one
            idx = np.random.randint(args.chaser_num)
            current_episode_idx = idx # Update the index for the new episode

            episode_reward = 0
            episode_chaser_reward = 0
            episode_timesteps = 0

        # Evaluate episode
        if t % args.eval_freq == 0:
            # ... (evaluation logic seems okay, but ensure indices match) ...
            total_success_rate = 0
            agent_success_rates = [] # This list seems unused later

            for j in range(args.chaser_num): # Iterate through each chaser for evaluation
                chaser_agent_eval = chaser_agents[j] # Use the specific chaser for its evaluation
                other_chaser_agents_eval = [chaser_agents[k] for k in range(args.chaser_num) if k != j] # Define others relative to j

                # Pass other_chaser_agents only when needed (e.g., for KL divergence)
                if j == 0 and len(other_chaser_agents_eval) > 0: # Example: Calculate KL only for chaser 0 vs others
                     success_rate, collision_rate, escape_rate, avg_nav_time, KL_divergences = eval_policy(
                         agent, chaser_agent_eval, eval_env, t,
                         other_chaser_agents=other_chaser_agents_eval,
                         save_directory=file_evaluation_episodes,
                         if_save_video=if_save_video)
                     # Log KL divergences
                     for kl_idx, kl_div in enumerate(KL_divergences):
                         other_agent_real_idx = [k for k in range(args.chaser_num) if k != j][kl_idx]
                         writer.add_scalar(f'eval/KL_divergence/chaser_{j}_vs_{other_agent_real_idx}', kl_div, t)
                else: # Evaluate without calculating KL divergence for others or if only one chaser
                     success_rate, collision_rate, escape_rate, avg_nav_time = eval_policy(
                         agent, chaser_agent_eval, eval_env, t,
                         other_chaser_agents=None, # Explicitly pass None
                         save_directory=file_evaluation_episodes,
                         if_save_video=if_save_video)


            
                print(f'chaser {j} success_rate: {success_rate:.2f}, collision_rate: {collision_rate:.2f}, escape_rate: {escape_rate:.2f}, avg_nav_time: {avg_nav_time:.2f} at step {t}')
                writer.add_scalar(f'eval/success_rate/chaser_{j}', success_rate, t)
                writer.add_scalar(f'eval/collision_rate/chaser_{j}', collision_rate, t)
                writer.add_scalar(f'eval/escape_rate/chaser_{j}', escape_rate, t)
                writer.add_scalar(f'eval/avg_nav_time/chaser_{j}', avg_nav_time, t)
                print('---------------------------------------')
                print('benchmark testing for chaser', j)
                results = eval_chaser_policy(chaser_agents[j],  chaser_ben, 100, save_directory=file_evaluation_episodes, mode = 'silent')
                print("\nChaser Performance Results:")
                print(f"Catch Rate: {results['catch_rate']:.3f}")
                print(f"Collision Rate: {results['collision_rate']:.3f}")
                writer.add_scalar(f'benchmark/catch_rate/chaser_{j}', results['catch_rate'], t)
                writer.add_scalar(f'benchmark/collision_rate/chaser_{j}', results['collision_rate'], t)
                writer.add_scalar(f'benchmark/avg_chase_time/chaser_{j}', results['avg_chase_time'], t)
            

                # evaluations[j].append(success_rate) # Correct index j
                total_success_rate += success_rate

                # Save model based on individual chaser performance
                if success_rate > 0.2 or t % args.save_model_freq == 0:
                    file_name = f'/step_{t}_chaser_{j}_success_{int(success_rate * 100)}_benchmark_{int(results["catch_rate"]*100)}' 
                    # Create chaser-specific directory if it doesn't exist
                    chaser_model_dir = f'{file_chaser_models}/chaser_{j}'
                    os.makedirs(chaser_model_dir, exist_ok=True)
                    file_path = f'{chaser_model_dir}{file_name}'
                    # Save the specific chaser agent
                    chaser_agent_eval.save(file_path)
            #update the prefill_ratio according to the average success rate, if the sr is larger than 0.2, change the prefill_ratio to [1-success_rate, success_rate]
            # else, keep the original prefill_ratio
            average_success_rate = total_success_rate / args.chaser_num
            target_ratio = [1 - average_success_rate, average_success_rate] if average_success_rate > 0.2 else args.prefill_ratio
            prefill_ratio = [
                prev_prefill_ratio[0] + transition_speed * (target_ratio[0] - prev_prefill_ratio[0]),
                prev_prefill_ratio[1] + transition_speed * (target_ratio[1] - prev_prefill_ratio[1])
            ]
            prev_prefill_ratio = prefill_ratio  # Store for next iteration
            print(f'Total average success rate at step {t}: {average_success_rate:.2f}')
            # Saving a dummy file based on average success rate - seems okay if intended
            file_name = f'/step_{t}_success_{int(average_success_rate * 100)}_benchmark_{int(results["catch_rate"]*100)}'
            np.savetxt(f'{file_results}{file_name}.txt', np.random.rand(1, 1)) # Smaller dummy file


    print('final test')
    # Final test needs clarification: which chaser agent to test against?
    # Option 1: Test against one specific chaser (e.g., chaser 0)
    # Option 2: Average results over tests against each chaser
    # Assuming Option 1 (test against chaser 0) for now:
    final_chaser_agent = chaser_agents[0]
    success_rate, collision_rate, escape_rate, avg_nav_time = eval_policy(agent, final_chaser_agent, test_env, t, # Use test_env
                                                             eval_episodes=500, save_directory=file_final_test_episodes,
                                                             if_save_video=if_save_video_test, final_test=True) # Use if_save_video_test
    print(f'Final Test Results (against chaser 0):')
    print(f'Success Rate: {success_rate:.2f}, Collision Rate: {collision_rate:.2f}, Escape Rate: {escape_rate:.2f}, Avg Nav Time: {avg_nav_time:.2f}')

    # If you want to average over all chasers for final test:
    # final_results = {'success': [], 'collision': [], 'escape': [], 'time': []}
    # for final_idx in range(args.chaser_num):
    #     final_chaser_agent = chaser_agents[final_idx]
    #     sr, cr, er, ant = eval_policy(agent, final_chaser_agent, test_env, t,
    #                                    eval_episodes=100, # Maybe fewer episodes per chaser?
    #                                    save_directory=file_final_test_episodes,
    #                                    if_save_video=if_save_video_test, final_test=True)
    #     final_results['success'].append(sr)
    #     final_results['collision'].append(cr)
    #     final_results['escape'].append(er)
    #     final_results['time'].append(ant)
    #     print(f'Final Test Results (against chaser {final_idx}): SR={sr:.2f}, CR={cr:.2f}, ER={er:.2f}, Time={ant:.2f}')
    # print(f'Final Test Results (Average): SR={np.mean(final_results["success"]):.2f}, CR={np.mean(final_results["collision"]):.2f}, ER={np.mean(final_results["escape"]):.2f}, Time={np.mean(final_results["time"]):.2f}')


if __name__ == "__main__":
    main()
