import argparse
import numpy as np
import random
import os
import torch
import time
from algos.SAC_AE.sac_ae import SacAeAgent
from algos.SAC_AE.utils import ReplayBuffer, eval_policy_mode, set_seed_everywhere, RatioCombinedReplayBuffer
from info import *
from env_drl_vo_adv import CrowdSim
from env_drl_vo_parallel import CrowdSim as CrowdSimEval
from torch.utils.tensorboard import SummaryWriter
from threading import Lock
from tqdm import tqdm
import time



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, chaser_agent, eval_or_test_env, current_steps, 
                eval_episodes=500, save_directory=None, 
                if_save_video=False, final_test=False):
    avg_reward = 0.
    avg_chaser_reward = 0.
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    catch = 0
    collision = 0
    timeout = 0
    collision_cases = []
    timeout_cases = []
    if_save_data = False
    for i in range(eval_episodes):
        if_save_data = (i < 5 or final_test)
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
            # print('time: ', time.time() - t1) # 7.5ms
            # eval_or_test_env.render()
            # time.sleep(0.2)
            avg_reward += reward
            avg_chaser_reward += chaser_reward
            ep_step = ep_step + 1
            if done or ep_step == eval_or_test_env.max_episode_step or isinstance(info, ReachGoal):
                if ep_step == eval_or_test_env.max_episode_step:
                    timeout += 1
                    timeout_cases.append(i)
                    timeout_times.append(eval_or_test_env.time_limit)
                    print('evaluation episode ' + str(i) + ', time out: ' + str(ep_step))
                else:
                    if isinstance(info, ReachGoal):
                        success += 1
                        success_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', goal reaching at evaluation step: ' + str(ep_step))
                    elif isinstance(chaser_info, ReachGoal):
                        catch += 1
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
    catch_rate = catch / eval_episodes
    assert success + collision + timeout + catch == eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_or_test_env.time_limit

    
    return success_rate, collision_rate, catch_rate, avg_nav_time

def eval_policy_benchmark(policy, eval_or_test_env, current_steps, 
                eval_episodes=500, save_directory=None, 
                if_save_video=False, final_test=False):
    avg_reward = 0.
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    collision_cases = []
    timeout_cases = []
    if_save_data = False
    for i in range(eval_episodes):
        if_save_data = (i < 5 or final_test)

        obs, robot_goal_emotion_state  = eval_or_test_env.reset(save_data=if_save_data)

        # eval_or_test_env.render()
        # time.sleep(0.2)
        done = False
        ep_step = 0
        
        while ep_step < eval_or_test_env.max_episode_step:
            # t1 = time.time()
            with eval_policy_mode(policy):
                action = policy.select_action(obs, robot_goal_emotion_state)
            # action = eval_or_test_env.dwa_compute_action()
            obs, robot_goal_emotion_state, reward, done, info = eval_or_test_env.step(action, eval=True, save_data=if_save_data)
            # print('time: ', time.time() - t1) # 7.5ms
            # eval_or_test_env.render()
            # time.sleep(0.2)
            avg_reward += reward
            ep_step = ep_step + 1
            if done or ep_step == eval_or_test_env.max_episode_step or isinstance(info, ReachGoal):
                if ep_step == eval_or_test_env.max_episode_step:
                    timeout += 1
                    timeout_cases.append(i)
                    timeout_times.append(eval_or_test_env.time_limit)
                    print('evaluation episode ' + str(i) + ', time out: ' + str(ep_step))
                else:
                    if isinstance(info, ReachGoal):
                        success += 1
                        success_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', goal reaching at evaluation step: ' + str(ep_step))
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
                    else:
                        raise ValueError('Invalid end signal from environment')
                break
        if save_directory is not None:
            if isinstance(info, ReachGoal):
                file_name = save_directory + '/eval_bm_' + str(current_steps) + '_' + str(i) + '.npz'
            else:
                file_name = save_directory + '/eval_bm_' + str(current_steps) + '_' + str(i) + '_fail' + '.npz'
            if if_save_data:
                np.savez_compressed(file_name, **eval_or_test_env.log_env)
                if if_save_video:
                    eval_or_test_env.save_video(current_steps, i)
                # policy.save_features(save_directory)

    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_or_test_env.time_limit

    
    return success_rate, collision_rate, avg_nav_time

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
        
        # If chaser caught the robot, add all transitions from this episode
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
        pbar.set_postfix({'success_rate': f"{success_count/(total_count):.2%}", 'filled': f"{added_transitions}/{size}"})
    
    pbar.close()
    print(f"Prefilling complete. Added {added_transitions} transitions from {success_count} successful episodes to buffer.")
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--policy', default='drl_vo', type=str)
    parser.add_argument('--chaser_policy', default='drl_vo', type=str)
    # options, differential, lip, digit_mujoco
    parser.add_argument('--robot_model', default='differential', type=str)
    parser.add_argument('--robot_eval_model', default='differential', type=str)
    # options, differential, lip, digit_mujoco
    parser.add_argument('--robot_test_model', default='differential', type=str)
    parser.add_argument('--chaser_model', default='differential', type=str)
    parser.add_argument('--chaser_eval_model', default='differential', type=str)
    parser.add_argument('--chaser_test_model', default='differential', type=str)
    parser.add_argument('--use_angular', action='store_true', default=False)
    parser.add_argument('--random_radius', action='store_true', default=True)
    parser.add_argument('--training_object', default='robot', type=str)
    parser.add_argument('--random_robot_goal', action='store_true', default=False)
    parser.add_argument('--random_chaser_reset', action='store_true', default=False)

    # device
    parser.add_argument('--device', type=str, default='cuda')
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--seed', default=1, type=int)
    # Time steps initial random policy is used
    parser.add_argument('--start_timesteps', default=10000, type=int)
    # How often (time steps) we evaluate
    parser.add_argument('--eval_freq', default=20000, type=int)
    # How often (time steps) we save the trained model
    parser.add_argument('--save_model_freq', default=50000, type=int)
    # Max time steps to run environment
    parser.add_argument('--max_timesteps', default=6e6, type=int)
    # replay buffer
    parser.add_argument("--replay_buffer_capacity", default=150000, type=int)
    parser.add_argument("--prefill", action='store_true', default=False)
    parser.add_argument("--prefilling_size", type=int, default=10000)
    parser.add_argument("--prefill_ratio", default=[0.8, 0.2], type=float, nargs='+')
    parser.add_argument("--main_buffer_ratio", default=0.6, type=float)
    # training
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # pixel: with decoder, identity: without decoder
    parser.add_argument('--decoder_type', default='identity', type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)

    # Model load file name, "" doesn't load, "default" uses file_name
    # args.load_model, format, step_NO_success_NO
    # parser.add_argument("--load_model", type=str, default="step_2740000_success_90")
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--load_model_path", type=str, default="")
    parser.add_argument("--load_test_model_path", type=str, default="")
    parser.add_argument("--load_chaser", action='store_true', default=False)
    
    # environment settings
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--lidar_dim", type=int, default=1800)
    parser.add_argument("--lidar_feature_dim", type=int, default=50)
    # lidar to image
    parser.add_argument('--image_size', default=100, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # 2 robot speed, 2 local goal
    parser.add_argument("--robot_goal_state_dim", type=int, default=4)
    parser.add_argument("--laser_angle_resolute", type=float, default=0.003490659)
    parser.add_argument("--laser_min_range", type=float, default=0.27)
    parser.add_argument("--laser_max_range", type=float, default=6.0)
    parser.add_argument("--human_num_max", type=int, default=4)
    parser.add_argument("--static_obstacle_num_max", type=int, default=3)
    parser.add_argument("--square_width", type=float, default=10.0)
    parser.add_argument("--chaser_num", type=int, default=1)
    args = parser.parse_args()

    #get chaser models list
    chaser_models_path = ['/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-05-07/seed_1/itr_1/chaser_models/chaser_0/step_480000_chaser_0_success_54_benchmark_65',
    '/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-05-07/seed_1/itr_1/chaser_models/chaser_1/step_380000_chaser_1_success_45_benchmark_67',
    '/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-05-07/seed_1/itr_1/chaser_models/chaser_2/step_380000_chaser_2_success_46_benchmark_72']
    pre_chaser_path = ['/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-03-13/seed_1/itr_1/chaser_models/chaser_0/step_1680000chaser_0_success_62',
    '/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-03-13/seed_1/itr_1/chaser_models/chaser_1/step_1680000chaser_1_success_70',
    '/mnt/ssd1/stilrmy/Navigation/drl_vo_3_adversaries/2025-03-13/seed_1/itr_1/chaser_models/chaser_2/step_1680000chaser_2_success_61']


    print("---------------------------------------")
    print(f"Policy: {args.policy}, RobotModel: {args.robot_model}, RobotEvalModel: {args.robot_eval_model}, Seed: {args.seed}")
    print("---------------------------------------")
    date = time.strftime("%Y-%m-%d")
    file_prefix = '/mnt/ssd1/stilrmy/Navigation/' + args.policy + '_' + args.robot_model + '_' + args.robot_eval_model + '/' + date
        
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
    else:
        raise NotImplementedError(args.robot_model)
    
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
    else:
        raise NotImplementedError(args.robot_test_model)
    # please manually set seeds when test with digit_arsim
    # otherwise, set it as args.seed
    # set_seed_everywhere(args.seed)
    args.human_num_max = 10
    ben_env = CrowdSimEval(args, action_range,  action_choices=action_choices)
    set_seed_everywhere(args.seed)

    device = torch.device(args.device)
    obs_shape = (args.frame_stack, args.image_size, args.image_size)
    robot_goal_state_dim = args.robot_goal_state_dim
    action_shape = (2,)
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
    
    chaser_agents = []
    pre_chaser_agents = []
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
        pre_chaser_agents.append(SacAeAgent(
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
    replay_buffers = []
    for i in range(args.chaser_num):
        replay_buffer = ReplayBuffer(
            obs_shape,
            robot_goal_state_dim, 
            action_shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device,
            obs_dtype=np.float32
        )
        replay_buffers.append(replay_buffer)
    pre_replay_buffers = []
    for i in range(args.chaser_num):
        pre_replay_buffer = ReplayBuffer(
            obs_shape,
            robot_goal_state_dim, 
            action_shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device,
            obs_dtype=np.float32
        )
        pre_replay_buffers.append(pre_replay_buffer)

    if args.prefill == True:
        for i in range(args.chaser_num):
            pre_chaser_agents[i].load(pre_chaser_path[i])
            print("pre chaser loaded")
            print("prefilling replay buffer")
            prefilling_replay_buffer(agent, pre_chaser_agents[i], pre_replay_buffers[i], env, args.prefilling_size)
        print("prefilling done")
    checkpoint_steps = 0

   
    if args.load_model_path != "":
        # replay_buffer.load(file_buffer)
        agent.load(args.load_model_path)
        # test_times = 300
        # current_step = 0
        # success_rate, collision_rate, avg_nav_time = eval_policy_benchmark(agent, ben_env, current_step, 
        #                                                          eval_episodes=test_times, save_directory=file_final_test_episodes, 
        #                                                          if_save_video=if_save_video_test, final_test=True)
        # # args.load_model, format, step_NO_success_NO
        # # extract the first number
        # print('success_rate, collision_rate, avg_nav_time')
        # print(success_rate, collision_rate, avg_nav_time)
        

    if args.load_test_model_path != "":
        print('start to test')
        agent.load(args.load_test_model_path)
        test_times = 100
        current_step = 0
        success_rate, collision_rate, avg_nav_time = eval_policy_benchmark(agent, ben_env, current_step, 
                                                                 eval_episodes=test_times, save_directory=file_final_test_episodes, 
                                                                 if_save_video=if_save_video_test, final_test=True)
        print('success_rate, collision_rate, avg_nav_time')
        print(success_rate, collision_rate, avg_nav_time)
        exit()
        return

    if args.load_chaser:
        for i in range(args.chaser_num):
            chaser_agents[i].load(chaser_models_path[i])
        print("chaser model loaded")


    


    evaluations = []

    obs, robot_goal_emotion_state, chaser_obs, chaser_goal_emotion_state = env.reset()
    done = False
    episode_reward = 0
    episode_chaser_reward = 0
    episode_timesteps = 0
    episode_num = 0
    buffer_ratio=[args.main_buffer_ratio, (1 - args.main_buffer_ratio) * 0.5, (1 - args.main_buffer_ratio) * 0.5]
    prev_prefill_ratio = args.prefill_ratio
    prefill_ratio = prev_prefill_ratio
    transition_speed = 0.2 
    idx = np.random.randint(args.chaser_num)
    chaser_agent = chaser_agents[idx]
    replay_buffer = replay_buffers[idx]

    for t in range(checkpoint_steps + 1, int(args.max_timesteps) + 1):
        if t == args.start_timesteps:
            print('replay buffer has been initialized')
        # Perform action
        # sample action for data collection
        with eval_policy_mode(agent):
            action = agent.sample_action(obs, robot_goal_emotion_state)
        with eval_policy_mode(chaser_agent):
            chaser_action = chaser_agent.sample_action(chaser_obs, chaser_goal_emotion_state)
        next_obs, next_robot_goal_emotion_state, reward, done, info, next_chaser_obs, next_chaser_goal_emotion_state, chaser_reward, chaser_info = env.step(action, chaser_action)

        episode_timesteps += 1

        if episode_timesteps == env.max_episode_step:
            done_bool = 0.0
        else:
            done_bool = float(done)

        # Store data in replay buffer
        replay_buffer.add(
            obs, robot_goal_emotion_state, action, reward, next_obs, next_robot_goal_emotion_state, done_bool)
      

        obs = next_obs
        robot_goal_emotion_state = next_robot_goal_emotion_state
        episode_reward += reward
        chaser_obs = next_chaser_obs
        chaser_goal_emotion_state = next_chaser_goal_emotion_state
        episode_chaser_reward += chaser_reward



        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            temp_replay_buffer = []
            temp_buffer_ratio = []
            for i in range(args.chaser_num):
                temp_replay_buffer.append(replay_buffers[i])
                temp_buffer_ratio.append(buffer_ratio[i] * prefill_ratio[1])
            for i in range(args.chaser_num):
                temp_replay_buffer.append(pre_replay_buffers[i])
                temp_buffer_ratio.append(buffer_ratio[i] * prefill_ratio[0])
            mixed_replay_buffer = RatioCombinedReplayBuffer(temp_replay_buffer,args.batch_size,temp_buffer_ratio)
            num_updates = args.start_timesteps if t == args.start_timesteps else 1
            for _ in range(num_updates):
                agent.update(mixed_replay_buffer, writer, t)

        if done or episode_timesteps == env.max_episode_step or isinstance(info, ReachGoal) or isinstance(chaser_info, ReachGoal): 
            if episode_timesteps == env.max_episode_step:
                print('total step ' + str(t) + ', train episode ' + str(episode_num+1) + ', with chaser ' + str(idx) + ', time out: ' + str(episode_timesteps))
            else:
                if isinstance(info, ReachGoal):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num+1) + 
                        ', with chaser ' + str(idx) + ', goal reaching at train step: ' + str(episode_timesteps))
                elif isinstance(chaser_info, ReachGoal):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num+1) + 
                        ', with chaser ' + str(idx) + ', chaser catch the robot at train step: ' + str(episode_timesteps))
                elif isinstance(info, Collision):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num+1) + 
                        ', with chaser ' + str(idx) + ', collision occur at train step: ' + str(episode_timesteps))                  
                elif isinstance(chaser_info, Collision):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num+1) + 
                        ', with chaser ' + str(idx) + ', chaser collision occur at train step: ' + str(episode_timesteps))
                else:
                    raise ValueError('Invalid end signal from environment')

            # Reset environment
            
            obs, robot_goal_emotion_state, chaser_obs, chaser_goal_emotion_state = env.reset()
            done = False
            episode_num += 1
            writer.add_scalar('train/episode_reward', episode_reward, episode_num)
            idx = np.random.randint(args.chaser_num)
            chaser_agent = chaser_agents[idx]
            replay_buffer = replay_buffers[idx]
            episode_reward = 0
            episode_chaser_reward = 0
            episode_timesteps = 0

        # Evaluate episode
        if t % args.eval_freq == 0:
            success_rate, collision_rate, catch_rate, avg_nav_time = eval_policy(agent, chaser_agent, eval_env, t, 
                                                                     save_directory=file_evaluation_episodes, 
                                                                     if_save_video=if_save_video)
            benchmark_success_rate, benchmark_collision_rate, benchmark_avg_nav_time = eval_policy_benchmark(agent, ben_env, t, 
                                                                     save_directory=file_evaluation_episodes, 
                                                                     if_save_video=if_save_video)
            file_name = '/step_' + str(t) + '_success_' + str(int(success_rate * 100)) + '_benchmark_success_' + str(int(benchmark_success_rate * 100))
            print('success_rate, collision_rate, catch_rate, avg_nav_time at step ' + str(t))
            print(success_rate, collision_rate, catch_rate, avg_nav_time)
            print('benchmark_success_rate, benchmark_collision_rate, benchmark_avg_nav_time at step ' + str(t))
            print(benchmark_success_rate, benchmark_collision_rate, benchmark_avg_nav_time)
            writer.add_scalar('eval/success_rate', success_rate, t)
            writer.add_scalar('eval/collision_rate', collision_rate, t)
            writer.add_scalar('eval/catch_rate', catch_rate, t)
            writer.add_scalar('eval/avg_nav_time', avg_nav_time, t)
            writer.add_scalar('eval/benchmark_success_rate', benchmark_success_rate, t)
            writer.add_scalar('eval/benchmark_collision_rate', benchmark_collision_rate, t)
            writer.add_scalar('eval/benchmark_avg_nav_time', benchmark_avg_nav_time, t)
            evaluations.append(success_rate)
            np.savetxt(file_results + file_name + '.txt', evaluations)
            if success_rate > 0.7 or t % args.save_model_freq == 0:
                agent.save(file_models + file_name)
                # replay_buffer.save(file_buffer)
            #update the prefill_ratio according to the average success rate, if the sr is larger than 0.2, change the prefill_ratio to [1-success_rate, success_rate]
            # else, keep the original prefill_ratio
            target_ratio = [1 - success_rate, success_rate] if success_rate > 0.2 else args.prefill_ratio
            prefill_ratio = [
                prev_prefill_ratio[0] + transition_speed * (target_ratio[0] - prev_prefill_ratio[0]),
                prev_prefill_ratio[1] + transition_speed * (target_ratio[1] - prev_prefill_ratio[1])
            ]
            prev_prefill_ratio = prefill_ratio  # Store for next iteration
         
    print('final test')
    success_rate, collision_rate, avg_nav_time, escape_rate = eval_policy(agent, chaser_agent, eval_env, t, 
                                                             eval_episodes=500, save_directory=file_final_test_episodes,
                                                             if_save_video=if_save_video, final_test=True)
    print('success_rate, collision_rate, avg_nav_time')
    print(success_rate, collision_rate, avg_nav_time)


if __name__ == "__main__":
    main()
