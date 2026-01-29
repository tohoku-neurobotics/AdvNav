import argparse
import numpy as np
import random
import os
import torch
import time
from algos.SAC_AE.sac_ae import SacAeAgent
from algos.SAC_AE.utils import ReplayBuffer, eval_policy_mode, set_seed_everywhere
from info import *
# Use an environment suitable for single-agent evaluation/training
from env_drl_vo_eval import CrowdSim as EnvForSelectorTraining 
from torch.utils.tensorboard import SummaryWriter
from threading import Lock
import time

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_selector_policy(selector_agent, nav_agent_aggressive, nav_agent_progressive, eval_env, current_steps, 
                         eval_episodes=100, save_directory=None, 
                         if_save_video=False, final_test=False):
    avg_reward = 0.
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    total_aggressive_steps = 0
    total_progressive_steps = 0
    total_steps_all_episodes = 0
    
    if_save_data = False
    for i in range(eval_episodes):
        if_save_data = (i < 5 or final_test)
        
        obs, robot_goal_emotion_state = eval_env.reset(save_data=if_save_data)
        
        done = False
        ep_step = 0
        ep_aggressive_steps = 0
        ep_progressive_steps = 0
        
        while ep_step < eval_env.max_episode_step:
            with eval_policy_mode(selector_agent):
                # Selector chooses a continuous action
                selector_action_cont = selector_agent.select_action(obs, robot_goal_emotion_state) 
            
            # Discretize selector action to choose navigation agent
            if selector_action_cont[0] >= 0: # Example thresholding
                active_nav_agent = nav_agent_aggressive
                ep_aggressive_steps += 1
            else:
                active_nav_agent = nav_agent_progressive
                ep_progressive_steps += 1

            # Selected navigation agent determines the navigation action
            with eval_policy_mode(active_nav_agent):
                nav_action = active_nav_agent.select_action(obs, robot_goal_emotion_state)
            
            # Environment steps with the chosen navigation action
            # Assuming eval_env.step takes only the robot's action
            obs, robot_goal_emotion_state, reward, done, info = eval_env.step(nav_action, eval=True, save_data=if_save_data)
            
            avg_reward += reward
            ep_step += 1
            
            if done or ep_step == eval_env.max_episode_step:
                total_steps_all_episodes += ep_step
                total_aggressive_steps += ep_aggressive_steps
                total_progressive_steps += ep_progressive_steps
                
                if ep_step == eval_env.max_episode_step:
                    timeout += 1
                    timeout_times.append(eval_env.time_limit)
                    print(f'Evaluation episode {i}, time out: {ep_step}')
                else:
                    if isinstance(info, ReachGoal):
                        success += 1
                        success_times.append(eval_env.global_time)
                        print(f'Evaluation episode {i}, goal reaching at evaluation step: {ep_step}')
                    elif isinstance(info, Collision):
                        collision += 1
                        collision_times.append(eval_env.global_time)
                        print(f'Evaluation episode {i}, collision occur at evaluation step: {ep_step}')
                    # Add other potential 'info' types if necessary (e.g., DigitCrazy)
                    else:
                        # Fallback if info is unexpected, treat as timeout or collision based on preference
                        print(f'Evaluation episode {i}, ended with unexpected info: {type(info)} at step {ep_step}. Treating as timeout.')
                        timeout += 1 
                        timeout_times.append(eval_env.time_limit) # Or treat as collision maybe?

                break
                
        # Save episode data if needed (similar to original eval_policy)
        if save_directory is not None and if_save_data:
            status = "success" if isinstance(info, ReachGoal) else "fail"
            filename = os.path.join(save_directory, f'eval_selector_{current_steps}_{i}_{status}.npz')
            np.savez_compressed(filename, **eval_env.log_env)
            # Add video saving if needed and configured

    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    timeout_rate = timeout / eval_episodes
    
    # Ensure rates sum to 1 (or close due to potential unexpected info handling)
    # assert success + collision + timeout == eval_episodes, f"Counts don't match: S={success}, C={collision}, T={timeout}, Total={eval_episodes}"
    if abs(success + collision + timeout - eval_episodes) > 1e-5 : # Allow for small float issues or unexpected info handling
       print(f"Warning: Evaluation episode counts mismatch. S={success}, C={collision}, T={timeout}, Total={eval_episodes}")


    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_env.time_limit
    avg_aggressive_prop = total_aggressive_steps / total_steps_all_episodes if total_steps_all_episodes > 0 else 0
    avg_progressive_prop = total_progressive_steps / total_steps_all_episodes if total_steps_all_episodes > 0 else 0

    return success_rate, collision_rate, timeout_rate, avg_nav_time, avg_aggressive_prop, avg_progressive_prop

def main():
    parser = argparse.ArgumentParser()

    # --- Arguments for Selector Training ---
    parser.add_argument('--policy', default='drl_vo_selector', type=str, help="Policy name identifier for saving.")
    parser.add_argument('--model_path_aggressive', type=str, required=True, help="Path to the pre-trained aggressive navigation agent model.")
    parser.add_argument('--model_path_progressive', type=str, required=True, help="Path to the pre-trained progressive navigation agent model.")
    
    # Environment parameters (should match navigation agents' training)
    parser.add_argument('--robot_model', default='differential', type=str, help="Robot model used during training/evaluation (e.g., differential, lip).")
    parser.add_argument('--robot_eval_model', default='differential', type=str)
    parser.add_argument('--robot_test_model', default='differential', type=str) # For final testing if needed
    parser.add_argument('--use_angular', action='store_true', default=False)
    parser.add_argument('--random_radius', action='store_true', default=True)
    parser.add_argument('--random_robot_goal', action='store_true', default=False) # Keep if env needs it

    # --- General Training Parameters ---
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--start_timesteps', default=10000, type=int, help="Steps for initial random selector actions.")
    parser.add_argument('--eval_freq', default=20000, type=int)
    parser.add_argument('--save_model_freq', default=50000, type=int)
    parser.add_argument('--max_timesteps', default=3e6, type=int) # Adjust as needed for selector training
    parser.add_argument("--replay_buffer_capacity", default=150000, type=int)
    
    # --- Selector Agent SAC_AE Parameters ---
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # Critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # Maybe use 0.005 like original?
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # Actor (Selector)
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # Encoder/Decoder (Shared)
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float) # Maybe use 0.005 like original?
    parser.add_argument('--decoder_type', default='identity', type=str) # Keep identity if not reconstructing obs
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float) # Maybe 0.0?
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float) # Maybe 0.0?
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # SAC specific
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float) # Maybe 0.9 like original?

    # --- Model Loading (Selector) ---
    parser.add_argument("--load_selector_model_path", type=str, default="", help="Path to load a pre-trained selector model.")
    parser.add_argument("--itr", type=int, default=1, help="Iteration number for saving.")

    # --- Environment Settings (Should match nav agents) ---
    # parser.add_argument("--action_dim", type=int, default=2) # Selector action dim is 1
    parser.add_argument("--lidar_dim", type=int, default=1800)
    # parser.add_argument("--lidar_feature_dim", type=int, default=50) # Defined by encoder_feature_dim
    parser.add_argument('--image_size', default=100, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument("--robot_goal_state_dim", type=int, default=4)
    parser.add_argument("--laser_angle_resolute", type=float, default=0.003490659)
    parser.add_argument("--laser_min_range", type=float, default=0.27)
    parser.add_argument("--laser_max_range", type=float, default=6.0)
    parser.add_argument("--human_num_max", type=int, default=4) # Max humans in training env
    parser.add_argument("--static_obstacle_num_max", type=int, default=3) # Max static obstacles
    parser.add_argument("--square_width", type=float, default=10.0) # Env dimension factor
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Training Selector Policy: {args.policy}, Base Robot Model: {args.robot_model}, Seed: {args.seed}")
    print(f"Aggressive Nav Model: {args.model_path_aggressive}")
    print(f"Progressive Nav Model: {args.model_path_progressive}")
    print("---------------------------------------")
    date = time.strftime("%Y-%m-%d")
    # Define file paths for selector training
    file_prefix_base = '/mnt/ssd1/stilrmy/Navigation/' + args.policy + '_' + args.robot_model
    if args.use_angular:
        file_prefix_base += '_angular'
    if args.random_radius:
        file_prefix_base += '_random_radius'
    
    file_prefix = f"{file_prefix_base}/{date}/seed_{args.seed}/itr_{args.itr}"

    file_results = file_prefix + '/selector_results'
    file_models = file_prefix + '/selector_models' # Save selector models here
    file_evaluation_episodes = file_prefix + '/selector_evaluation_episodes'
    file_final_test_episodes = file_prefix + '/selector_final_test_episodes'
    file_buffer = file_prefix + '/selector_buffer'

    # Create directories
    os.makedirs(file_results, exist_ok=True)
    os.makedirs(file_models, exist_ok=True)
    os.makedirs(file_evaluation_episodes, exist_ok=True)
    os.makedirs(file_final_test_episodes, exist_ok=True)
    os.makedirs(file_buffer, exist_ok=True)

    writer = SummaryWriter(log_dir=file_results)

    # --- Define Action Spaces ---
    # Navigation action space (needed for loading nav agents)
    nav_action_range = np.array([[-0.5, -1.5], [1.0, 1.5]])
    
    # Selector action space (1D continuous)
    selector_action_dim = 1
    selector_action_shape = (selector_action_dim,)
    selector_action_range = np.array([[-1.0], [1.0]]) # Example range [-1, 1]

    # --- Initialize Environment ---
    # Use the environment class assumed suitable for single-agent interaction
    # Ensure args passed are compatible with this environment class
    # Remove chaser-specific args if EnvForSelectorTraining doesn't expect them
    env = EnvForSelectorTraining(args, nav_action_range) 
    eval_env = EnvForSelectorTraining(args, nav_action_range)
    test_env = EnvForSelectorTraining(args, nav_action_range) # For final testing

    # Set seeds
    set_seed_everywhere(args.seed)

    device = torch.device(args.device)

    # --- Initialize Navigation Agents (Load pre-trained) ---
    # Common parameters for loading nav agents
    obs_shape = (args.frame_stack, args.image_size, args.image_size)
    robot_goal_state_dim = args.robot_goal_state_dim
    nav_action_shape = (2,) # Navigation action is (v, w)

    nav_agent_params = {
        'obs_shape': obs_shape,
        'robot_goal_state_dim': robot_goal_state_dim,
        'action_shape': nav_action_shape,
        'action_range': nav_action_range,
        'device': device,
        'hidden_dim': args.hidden_dim,
        'encoder_type': args.encoder_type,
        'encoder_feature_dim': args.encoder_feature_dim,
        'decoder_type': args.decoder_type, # Match decoder type used for training nav agents
        'num_layers': args.num_layers,
        'num_filters': args.num_filters,
        # Add other necessary params if SacAeAgent constructor changed significantly
    }

    nav_agent_aggressive = SacAeAgent(**nav_agent_params)
    nav_agent_progressive = SacAeAgent(**nav_agent_params)

    print(f"Loading aggressive navigation agent from: {args.model_path_aggressive}")
    nav_agent_aggressive.load(args.model_path_aggressive)
    nav_agent_aggressive.train(False) # Set to evaluation mode

    print(f"Loading progressive navigation agent from: {args.model_path_progressive}")
    nav_agent_progressive.load(args.model_path_progressive)
    nav_agent_progressive.train(False) # Set to evaluation mode

    # --- Initialize Selector Agent ---
    selector_agent = SacAeAgent(
        obs_shape=obs_shape,
        robot_goal_state_dim=robot_goal_state_dim,
        action_shape=selector_action_shape, # Use 1D action shape
        action_range=selector_action_range, # Use selector action range
        device=device,
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

    # --- Initialize Replay Buffer (for Selector) ---
    selector_replay_buffer = ReplayBuffer(
        obs_shape=obs_shape,
        robot_goal_state_dim=robot_goal_state_dim, 
        action_shape=selector_action_shape, # Selector's action shape
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        obs_dtype=np.float32 # Or match env obs dtype
    )

    # --- Load Selector Model (if specified) ---
    checkpoint_steps = 0
    if args.load_selector_model_path != "":
        print(f"Loading selector agent from: {args.load_selector_model_path}")
        selector_agent.load(args.load_selector_model_path)
        # Optionally load replay buffer if saved previously
        # selector_replay_buffer.load(file_buffer) 
        try:
           checkpoint_steps = int(args.load_selector_model_path.split('_step_')[1].split('_')[0])
           print(f"Resuming training from step {checkpoint_steps}")
        except:
           print("Could not parse step number from selector model path, starting from step 0.")
           checkpoint_steps = 0


    # --- Training Loop ---
    evaluations = [] # Track success rate or other metric

    obs, robot_goal_emotion_state = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(checkpoint_steps + 1, int(args.max_timesteps) + 1):
        
        # Select action randomly or based on policy
        if t < args.start_timesteps:
            # Sample random action from the selector's action space
            selector_action_cont = np.random.uniform(selector_action_range[0,0], selector_action_range[1,0], size=selector_action_shape)
        else:
            with eval_policy_mode(selector_agent):
                # Selector samples a continuous action
                selector_action_cont = selector_agent.sample_action(obs, robot_goal_emotion_state)

        # Discretize action to choose navigation agent
        if selector_action_cont[0] >= 0: # Threshold at 0
            active_nav_agent = nav_agent_aggressive
        else:
            active_nav_agent = nav_agent_progressive

        # Get navigation action from the selected agent
        with eval_policy_mode(active_nav_agent):
            nav_action = active_nav_agent.select_action(obs, robot_goal_emotion_state)

        # Perform action in the environment using the navigation action
        next_obs, next_robot_goal_emotion_state, reward, done, info = env.step(nav_action)

        episode_timesteps += 1

        # Determine done_bool for replay buffer
        # Use timeout condition from environment directly
        done_bool = float(done) if episode_timesteps < env.max_episode_step else 0.0 

        # Store transition for the *selector* agent in its replay buffer
        # Store the *continuous* selector action that led to the choice
        selector_replay_buffer.add(
            obs, robot_goal_emotion_state, selector_action_cont, reward, 
            next_obs, next_robot_goal_emotion_state, done_bool
        )

        # Update state
        obs = next_obs
        robot_goal_emotion_state = next_robot_goal_emotion_state
        episode_reward += reward

        # Train selector agent
        if t >= args.start_timesteps:
            num_updates = args.start_timesteps if t == args.start_timesteps else 1
            for _ in range(num_updates):
                # Pass the selector's replay buffer to the update function
                selector_agent.update(selector_replay_buffer, writer, t) 

        # Handle episode end
        if done or episode_timesteps == env.max_episode_step:
            result = "timeout" if episode_timesteps == env.max_episode_step else type(info).__name__
            print(f'Total Step: {t}, Train Episode: {episode_num+1}, Steps: {episode_timesteps}, Result: {result}, Reward: {episode_reward:.2f}')

            # Reset environment
            obs, robot_goal_emotion_state = env.reset()
            done = False
            writer.add_scalar('train/episode_reward', episode_reward, episode_num + 1)
            writer.add_scalar('train/episode_steps', episode_timesteps, episode_num + 1)
            episode_num += 1
            episode_reward = 0
            episode_timesteps = 0

        # Evaluate selector policy
        if t % args.eval_freq == 0:
            print(f"\n--- Evaluating Selector at Step {t} ---")
            eval_success_rate, eval_collision_rate, eval_timeout_rate, eval_avg_nav_time, \
            eval_agg_prop, eval_prog_prop = eval_selector_policy(
                selector_agent, nav_agent_aggressive, nav_agent_progressive, eval_env, t,
                eval_episodes=50, # Use fewer episodes for faster eval during training
                save_directory=file_evaluation_episodes 
                # if_save_video=if_save_video # Add if video saving is desired/possible
            )
            
            print(f"Evaluation Results @ Step {t}:")
            print(f"  Success Rate: {eval_success_rate:.3f}, Collision Rate: {eval_collision_rate:.3f}, Timeout Rate: {eval_timeout_rate:.3f}")
            print(f"  Avg Nav Time: {eval_avg_nav_time:.3f}")
            print(f"  Agent Proportions (Aggressive/Progressive): {eval_agg_prop:.3f} / {eval_prog_prop:.3f}")
            print("-------------------------------------------\n")

            writer.add_scalar('eval/success_rate', eval_success_rate, t)
            writer.add_scalar('eval/collision_rate', eval_collision_rate, t)
            writer.add_scalar('eval/timeout_rate', eval_timeout_rate, t)
            writer.add_scalar('eval/avg_nav_time', eval_avg_nav_time, t)
            writer.add_scalar('eval/aggressive_prop', eval_agg_prop, t)
            writer.add_scalar('eval/progressive_prop', eval_prog_prop, t)
            
            evaluations.append(eval_success_rate) # Track success rate
            
            # Save selector model periodically or based on performance
            if t % args.save_model_freq == 0 or eval_success_rate >= max(evaluations[-5:] if len(evaluations)>5 else [0]): # Save if good performance
                save_path = os.path.join(file_models, f'selector_step_{t}_success_{int(eval_success_rate*100)}')
                print(f"Saving selector model to: {save_path}")
                selector_agent.save(save_path)
                # Optionally save replay buffer
                # selector_replay_buffer.save(file_buffer) 
                # Save evaluation results log
                np.savetxt(os.path.join(file_results, f'eval_history_step_{t}.txt'), evaluations)


    # --- Final Test ---
    print('\n--- Final Selector Test ---')
    test_success_rate, test_collision_rate, test_timeout_rate, test_avg_nav_time, \
    test_agg_prop, test_prog_prop = eval_selector_policy(
        selector_agent, nav_agent_aggressive, nav_agent_progressive, test_env, t, 
        eval_episodes=100, # Use more episodes for final test
        save_directory=file_final_test_episodes, 
        final_test=True
        # if_save_video=if_save_video_test # Add if needed
    )
    print("\nFinal Test Results:")
    print(f"  Success Rate: {test_success_rate:.3f}, Collision Rate: {test_collision_rate:.3f}, Timeout Rate: {test_timeout_rate:.3f}")
    print(f"  Avg Nav Time: {test_avg_nav_time:.3f}")
    print(f"  Agent Proportions (Aggressive/Progressive): {test_agg_prop:.3f} / {test_prog_prop:.3f}")
    
    # Save final results summary
    final_results_summary = {
        'success_rate': test_success_rate,
        'collision_rate': test_collision_rate,
        'timeout_rate': test_timeout_rate,
        'avg_nav_time': test_avg_nav_time,
        'aggressive_prop': test_agg_prop,
        'progressive_prop': test_prog_prop,
        'total_steps': t
    }
    np.save(os.path.join(file_results, 'final_test_summary.npy'), final_results_summary)
    print("-------------------------------------------\n")


if __name__ == "__main__":
    main() 