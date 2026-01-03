import torch 
import torch.nn.functional as F
import random
import gymnasium as gym
import numpy as np
from model import DeepQNetwork
import os
from dotenv import load_dotenv 
import argparse

load_dotenv()


def load_checkpoint(net, target_net, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    net.load_state_dict(checkpoint['model_state_dict'])
    if target_net is not None:
        target_net.load_state_dict(checkpoint['target_model_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch


def sample_episode(env, net: DeepQNetwork, device, p_random: float = 0.0, verbose=False, render=False):
    """Sample a single episode"""
    net.eval() 
    episode = []
    state, info = env.reset()

    done = False
    total_reward = 0
    
    while not done:
        # Choose action
        prob = random.uniform(0, 1)
        if prob <= p_random:
            action = env.action_space.sample()
        else: 
            with torch.no_grad():
                tensorized_state = torch.tensor(state, dtype=torch.float32).to(device=device)
                action_scores = net(tensorized_state).cpu().numpy()
                action = np.argmax(action_scores).astype(np.int64)

            if verbose:
                print(f"State: {state}, Action scores: {action_scores}, Action: {action}")
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        if verbose:
            print(f"Action: {action}, Reward: {reward}, Total: {total_reward}, Done: {done}")

        episode.append((state, action, reward, done, next_state))
        state = next_state

        if render:
            env.render()

    return episode, total_reward


def evaluate(env, net: DeepQNetwork, device, n_trials: int = 100, p_random: float = 0.0, verbose=False, render=False):
    """
    Evaluate the network over multiple episodes.
    
    Args:
        env: Gymnasium environment
        net: The Q-network to evaluate
        device: torch device
        n_trials: Number of episodes to run for evaluation
        p_random: Probability of random action (0.0 for greedy)
        verbose: Print detailed info
        render: Render the environment
    
    Returns:
        dict with evaluation metrics
    """
    net.eval()
    total_rewards = []
    episode_lengths = []
    
    with torch.no_grad():
        for trial in range(n_trials):
            if verbose:
                print(f"\n--- Trial {trial + 1}/{n_trials} ---")
            
            episode, episode_reward = sample_episode(
                env,
                net,
                device,
                p_random=p_random,
                verbose=verbose,
                render=render
            )
            
            total_rewards.append(episode_reward)
            episode_lengths.append(len(episode))
            
            if verbose or (trial + 1) % 10 == 0:
                print(f"Trial {trial + 1}: Reward = {episode_reward}, Length = {len(episode)}")
    
    results = {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "min_reward": np.min(total_rewards),
        "max_reward": np.max(total_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "all_rewards": total_rewards,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DQN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Epsilon for epsilon-greedy (default: 0.0 for greedy)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Gymnasium environment name (default: CartPole-v1)')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension size (default: 512)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    render_mode = "human" if args.render else None
    env = gym.make(args.env, render_mode=render_mode)
    
    # Get environment dimensions
    observation_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    
    print(f"\nEnvironment: {args.env}")
    print(f"Observation space: {observation_size}")
    print(f"Action space: {action_space_size}")
    
    # Initialize network
    net = DeepQNetwork(observation_size, action_space_size, args.hidden_dim)
    net = net.to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    epoch = load_checkpoint(net, None, args.checkpoint, device)
    
    # Run evaluation
    print(f"\nRunning evaluation with {args.n_trials} episodes...")
    print(f"Epsilon: {args.epsilon}")
    
    results = evaluate(
        env,
        net,
        device,
        n_trials=args.n_trials,
        p_random=args.epsilon,
        verbose=args.verbose,
        render=args.render
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Checkpoint epoch: {epoch}")
    print(f"Number of trials: {args.n_trials}")
    print(f"\nReward Statistics:")
    print(f"  Mean:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Min:    {results['min_reward']:.2f}")
    print(f"  Max:    {results['max_reward']:.2f}")
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean:   {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print("="*50)
    
    # Optional: Print reward distribution
    if args.verbose:
        print("\nReward Distribution:")
        rewards = results['all_rewards']
        bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        for i in range(len(bins)-1):
            count = sum(1 for r in rewards if bins[i] <= r < bins[i+1])
            if count > 0:
                print(f"  {bins[i]:3d}-{bins[i+1]:3d}: {'█' * count} ({count})")
    
    env.close()


if __name__ == "__main__":
    main()