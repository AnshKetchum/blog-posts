import torch 
import torch.nn.functional as F
from torch.optim import Adam
import random
import gymnasium as gym
import numpy as np
from model import DeepQNetwork
from tqdm import tqdm
import wandb
import os
from dotenv import load_dotenv 
from replay_buffer import ReplayBuffer

load_dotenv()

def save_checkpoint(net, target_net, optimizer, epoch, replay_buffer, checkpoint_dir="checkpoints"):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'target_model_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'buffer_size': len(replay_buffer),
    }, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(net, target_net, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['target_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch

# Initialize model
def initialize_model(device, state_dimension_size, action_space_size, hidden_dimension=512, 
                     checkpoint_path=None, learning_rate=1e-3):
    """
    Initialize model, optionally loading from checkpoint for resuming training.
    
    Args:
        device: torch device
        state_dimension_size: input dimension
        action_space_size: output dimension
        hidden_dimension: hidden layer size
        checkpoint_path: path to checkpoint to resume from (optional)
        learning_rate: learning rate for optimizer
    
    Returns:
        net, target_net, optimizer, start_epoch
    """
    net = DeepQNetwork(state_dimension_size, action_space_size, hidden_dimension)
    net = net.to(device)
    
    # Initialize target network
    target_net = DeepQNetwork(state_dimension_size, action_space_size, hidden_dimension)
    target_net = target_net.to(device)
    
    # Initialize optimizer
    optimizer = Adam(net.parameters(), learning_rate)
    
    start_epoch = 0
    
    # Load from checkpoint if provided
    if checkpoint_path is not None:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch = load_checkpoint(net, target_net, optimizer, checkpoint_path, device)
        start_epoch += 1  # Start from next epoch
        print(f"Resuming training from epoch {start_epoch}")
    else:
        # Initialize target network with same weights as main network
        target_net.load_state_dict(net.state_dict())
    
    target_net.eval()  # Target network always in eval mode
    
    return net, target_net, optimizer, start_epoch

def sample_episode(env, net: DeepQNetwork, device, p_random: float = 1.0, verbose = False):
    # 2. Start a new episode
    net.eval() 
    episode = []
    state, info = env.reset()

    done = False
    while not done:
        # 3. Choose an action (here: random, or by the model)
        prob = random.uniform(0, 1)
        if prob <= p_random:
            action = env.action_space.sample()
        else: 
            with torch.no_grad():
                tensorized_state = torch.tensor(state, dtype=torch.float32).to(device=device)
                action_scores = net(tensorized_state).cpu().numpy()
                action = np.argmax(action_scores).astype(np.int64)

            if verbose:
                print("Got action scores", action_scores, "argmax", action)
        
        # 4. Take a step with the action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 5. Observe what happens
        if verbose:
            print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")

        episode.append((state, action, reward, done, next_state))
        state = next_state

        # (optional) Render
        env.render()

    return episode 

def sample_k_episodes(env, net: DeepQNetwork, device, p_random: float, k: int, verbose = False):
    episodes = []
    for i in range(k):
        episode = sample_episode(
            env,
            net,
            device,
            p_random,
            verbose,
        )
        episodes.append(episode)
    
    return episodes

def evaluate(env, net: DeepQNetwork, device, n_trials: int = 10):
    """
    Evaluate the network over multiple episodes.
    
    Args:
        env: Gymnasium environment
        net: The Q-network to evaluate
        device: torch device
        n_trials: Number of episodes to run for evaluation
    
    Returns:
        dict with evaluation metrics
    """
    net.eval()
    total_rewards = []
    
    with torch.no_grad():
        for _ in range(n_trials):
            episode = sample_episode(
                env,
                net,
                device,
                p_random=0.0,  # Greedy policy during evaluation
                verbose=False
            )
            episode_reward = sum([transition[2] for transition in episode])
            total_rewards.append(episode_reward)
    
    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "min_reward": np.min(total_rewards),
        "max_reward": np.max(total_rewards),
    }

def train_step(replay_buffer: ReplayBuffer, net: DeepQNetwork, target_net: DeepQNetwork, 
               optimizer: torch.optim.Optimizer, device, batch_size: int = 8, 
               discount_rate: float = 0.99):
    # Sample from replay buffer
    if len(replay_buffer) < batch_size:
        return None, None, None
    
    sampled_transitions = replay_buffer.sample(batch_size)
    
    net.train()
    target_net.eval()  # Target network always in eval mode
    optimizer.zero_grad()
    
    # Prepare batched tensors
    initial_states = []
    actions = []
    rewards = []
    target_states = []
    dones = []
    
    for state, action, reward, done, next_state in sampled_transitions:
        initial_states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        target_states.append(next_state)
    
    # Convert to tensors
    initial_states = torch.tensor(np.array(initial_states), device=device, dtype=torch.float32)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32).unsqueeze(1)  # (B, 1)
    dones = torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(1)  # (B, 1)
    target_states = torch.tensor(np.array(target_states), device=device, dtype=torch.float32)
    
    # Forward pass with main network
    initial_scores = net(initial_states)  # (B, num_actions)
    
    # Forward pass with target network (frozen)
    with torch.no_grad():
        target_scores = target_net(target_states)  # (B, num_actions)
    
    # Select Q-values for taken actions
    actions_tensor = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(1)  # (B, 1)
    initial_q_values = initial_scores.gather(1, actions_tensor)  # (B, 1)
    
    # Compute target Q-values using target network
    max_target_q_values = target_scores.max(dim=1, keepdim=True)[0]  # (B, 1)
    target = rewards + discount_rate * max_target_q_values * (1 - dones)  # (B, 1)
    
    # Compute loss
    loss = F.mse_loss(initial_q_values, target)
    
    loss.backward()
    optimizer.step()
    
    return initial_scores, target_scores, loss


def train(env, net: DeepQNetwork, target_net: DeepQNetwork, optimizer: torch.optim.Optimizer,
          train_steps: int, device, probability_schedule: np.ndarray, batch_size: int, 
          discount_rate: float = 0.99, evaluation_frequency = 5, 
          replay_buffer_capacity = 10000, episode_collection_frequency = 5, 
          episodes_per_collection = 5, target_update_frequency = 100, 
          checkpoint_frequency = 250, checkpoint_dir = "checkpoints", 
          start_epoch = 0, prefill_buffer = True):
    """
    Train DQN agent with improved efficiency.
    
    Args:
        episode_collection_frequency: Collect new episodes every N training steps (default: 5)
        episodes_per_collection: Number of episodes to collect each time (default: 5)
        prefill_buffer: Whether to prefill replay buffer before training (default: True)
    """
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    
    # Prefill replay buffer with random experience
    if prefill_buffer:
        print(f"Prefilling replay buffer with {replay_buffer_capacity // 10} transitions...")
        prefill_episodes = replay_buffer_capacity // 10 // 200  # Assume ~200 steps per episode
        prefill_episodes = max(prefill_episodes, 10)  # At least 10 episodes
        
        episodes = sample_k_episodes(env, net, device, p_random=1.0, k=prefill_episodes)
        for episode in episodes:
            for transition in episode:
                replay_buffer.push(transition)
        
        print(f"Prefilled buffer with {len(replay_buffer)} transitions from {prefill_episodes} episodes")
    
    # Track best model
    best_mean_reward = -float('inf')
    best_checkpoint_path = None
    
    for epoch in tqdm(range(start_epoch, train_steps)):
        # Collect experience periodically instead of every step
        if epoch % episode_collection_frequency == 0:
            epsilon = probability_schedule[min(epoch, len(probability_schedule)-1)]
            episodes = sample_k_episodes(env, net, device, epsilon, episodes_per_collection)
            
            for episode in episodes:
                for transition in episode:
                    replay_buffer.push(transition)
        
        # Train from replay buffer (can do multiple gradient steps per collection)
        initial_scores, target_scores, loss = train_step(
            replay_buffer,
            net,
            target_net,
            optimizer,
            device,
            batch_size,
            discount_rate
        )
        
        # Update target network periodically
        if epoch % target_update_frequency == 0 and epoch > 0:
            target_net.load_state_dict(net.state_dict())
            print(f"Updated target network at epoch {epoch}")
        
        # Log training loss if training occurred
        if loss is not None:
            mean_loss = loss.item()
            wandb.log({
                "train/loss": mean_loss,
                "train/buffer_size": len(replay_buffer),
                "train/epsilon": probability_schedule[min(epoch, len(probability_schedule)-1)],
            }, step=epoch)
        
        # Evaluation
        if epoch % evaluation_frequency == 0 or epoch == train_steps - 1:
            eval_metrics = evaluate(env, net, device, n_trials=10)
            
            # Log evaluation metrics
            wandb.log({
                "eval/mean_reward": eval_metrics["mean_reward"],
                "eval/std_reward": eval_metrics["std_reward"],
                "eval/min_reward": eval_metrics["min_reward"],
                "eval/max_reward": eval_metrics["max_reward"],
            }, step=epoch)
            
            print(f"Epoch {epoch}: Mean Reward {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}, "
                  f"Buffer size: {len(replay_buffer)}")
            
            # Save best model
            if eval_metrics["mean_reward"] > best_mean_reward:
                best_mean_reward = eval_metrics["mean_reward"]
                best_checkpoint_path = save_checkpoint(
                    net, target_net, optimizer, epoch, replay_buffer, 
                    checkpoint_dir=os.path.join(checkpoint_dir, "best")
                )
                print(f"New best model! Mean reward: {best_mean_reward:.2f}")
        
        # Periodic checkpoint
        if epoch % checkpoint_frequency == 0 and epoch > 0:
            save_checkpoint(
                net, target_net, optimizer, epoch, replay_buffer, 
                checkpoint_dir=checkpoint_dir
            )
    
    # Save final checkpoint
    final_checkpoint_path = save_checkpoint(
        net, target_net, optimizer, train_steps - 1, replay_buffer,
        checkpoint_dir=os.path.join(checkpoint_dir, "final")
    )
    
    print(f"\nTraining complete!")
    print(f"Best model (reward {best_mean_reward:.2f}): {best_checkpoint_path}")
    print(f"Final model: {final_checkpoint_path}")

if __name__ == "__main__":
    OBSERVATION_SIZE = 4
    ACTION_SPACE_SIZE = 2
    learning_rate = 1e-3
    batch_size = 64
    discount_rate = 0.99
    
    train_steps = 2048  # Increased since we're training more efficiently
    target_update_frequency = 100
    replay_buffer_capacity = 10000
    evaluation_frequency = 50
    checkpoint_frequency = 250
    checkpoint_dir = "checkpoints"
    
    # New efficiency parameters
    episode_collection_frequency = 5  # Collect episodes every 5 training steps
    episodes_per_collection = 5       # Collect 5 episodes at a time
    prefill_buffer = True             # Prefill buffer before training
    
    # Optional: Resume from checkpoint
    # resume_checkpoint = None  # Set to path to resume training
    resume_checkpoint = "checkpoints/checkpoint_epoch_500.pt"
    
    # Epsilon decay
    probability_schedule = np.linspace(1.0, 0.01, int(train_steps * 0.8)).tolist()
    probability_schedule += [0.01] * (train_steps - len(probability_schedule))
    
    # Create environment
    env = gym.make("CartPole-v1", render_mode="human")
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize network (with optional checkpoint loading)
    net, target_net, optimizer, start_epoch = initialize_model(
        device, 
        OBSERVATION_SIZE, 
        ACTION_SPACE_SIZE,
        checkpoint_path=resume_checkpoint,
        learning_rate=learning_rate
    )
    
    # WandB init
    project_name = os.getenv("WANDB_PROJECT_NAME", None)
    assert project_name
    
    wandb_config = {
        "batch_size": batch_size,
        "lr": learning_rate,
        "num_epochs": train_steps,
        "target_update_freq": target_update_frequency,
        "checkpoint_freq": checkpoint_frequency,
        "episode_collection_freq": episode_collection_frequency,
        "episodes_per_collection": episodes_per_collection,
        "discount_rate": discount_rate,
        "buffer_capacity": replay_buffer_capacity,
        "prefill_buffer": prefill_buffer,
        "resumed_from": resume_checkpoint if resume_checkpoint else "scratch",
        "start_epoch": start_epoch,
    }
    
    wandb.init(
        project=project_name,
        config=wandb_config,
        resume="allow" if resume_checkpoint else None,
    )
    wandb.watch(net, log="all", log_freq=100)
    
    train(
        env,
        net,
        target_net,
        optimizer,
        train_steps,
        device,
        probability_schedule,
        batch_size,
        discount_rate,
        evaluation_frequency,
        replay_buffer_capacity,
        episode_collection_frequency,
        episodes_per_collection,
        target_update_frequency,
        checkpoint_frequency,
        checkpoint_dir,
        start_epoch,
        prefill_buffer
    )
    
    env.close()