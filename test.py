import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from collections import deque

# --- Hyperparameters ---
STACK_SIZE = 4
EVAL_EPISODES = 500
GPU_ID = 4

# --- Image Preprocessing ---
mspacman_color = 210 + 164 + 74  # 448

def preprocess_observation(obs):
    """
    Preprocesses a 210x160x3 frame to an 88x80x1 frame as per assignment.
    """
    img = obs[1:176:2, ::2]  # crop and downsize -> (88, 80, 3)
    img = img.sum(axis=2)  # to greyscale -> (88, 80)
    img[img == mspacman_color] = 0  # Improve contrast
    
    # Normalize from -128 to 127 as specified in assignment
    img = (img // 3 - 128).astype(np.int8)
    
    return img.reshape(88, 80, 1)  # (H, W, C)

# --- Q-Network (CNN) Definition ---
class QNetwork(nn.Module):
    """Convolutional Neural Network for Atari games"""
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        # Input shape: (N, 4, 88, 80) -> (Batch, Channels, Height, Width)
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4)  # -> (N, 32, 21, 19)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # -> (N, 64, 9, 8)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # -> (N, 64, 7, 6)
        
        # Flattened size: 64 * 7 * 6 = 2688
        self.fc1 = nn.Linear(2688, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        # x is (N, C, H, W) - expects normalized input in [-1, 1] range
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)  # (N, 2688)
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def select_action(state, policy_net, device):
    """
    Selects action greedily (no exploration).
    """
    with torch.no_grad():
        # Convert state to float and normalize to [-1, 1]
        state_float = state.float().to(device) / 128.0
        return policy_net(state_float).max(1)[1].view(1, 1)

def main():
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(GPU_ID)}")
    else:
        print("CUDA not available. Using CPU.")
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = gym.make('MsPacman-v0')
    action_size = env.action_space.n
    print(f"Action space size: {action_size}")
    
    # Load the best model
    policy_net = QNetwork(action_size).to(device)
    best_model_path = 'models/mspacman_dqn_best.pth'
    
    try:
        policy_net.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from: {best_model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {best_model_path}")
        print("Please ensure the model file exists.")
        return
    
    policy_net.eval()  # Set to evaluation mode
    
    # Run evaluation
    print(f"\nRunning evaluation for {EVAL_EPISODES} episodes...")
    eval_rewards = []
    frame_stack = deque(maxlen=STACK_SIZE)
    
    for eval_ep in range(EVAL_EPISODES):
        obs, _ = env.reset()
        proc_obs = preprocess_observation(obs)
        
        # Initialize frame stack
        for _ in range(STACK_SIZE):
            frame_stack.append(proc_obs)
        
        state_np = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
        state = torch.tensor(state_np, dtype=torch.int8).unsqueeze(0)
        
        done = False
        episode_reward = 0
        
        while not done:
            # Select action greedily
            action = select_action(state, policy_net, device)
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = bool(terminated or truncated)
            
            if not done:
                proc_next_obs = preprocess_observation(next_obs)
                frame_stack.append(proc_next_obs)
                
                next_state_np = np.concatenate(list(frame_stack), axis=2).transpose(2, 0, 1)
                state = torch.tensor(next_state_np, dtype=torch.int8).unsqueeze(0)
            
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
        
        # Print progress
        if (eval_ep + 1) % 100 == 0:
            current_avg = np.mean(eval_rewards)
            print(f"  Episode {eval_ep + 1}/{EVAL_EPISODES} completed | "
                  f"Current Average: {current_avg:.2f}")
    
    env.close()
    
    # Report results
    mean_reward = float(np.mean(eval_rewards))
    std_reward = float(np.std(eval_rewards))
    min_reward = float(np.min(eval_rewards))
    max_reward = float(np.max(eval_rewards))
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Best Model)")
    print("="*60)
    print(f"Episodes:         {EVAL_EPISODES}")
    print(f"Mean Reward:      {mean_reward:.2f}")
    print(f"Std Deviation:    {std_reward:.2f}")
    print(f"Min Reward:       {min_reward:.2f}")
    print(f"Max Reward:       {max_reward:.2f}")
    print("="*60)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Histogram of Evaluation Rewards
    plt.figure(figsize=(12, 6))
    plt.hist(eval_rewards, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Evaluation Reward Distribution - Best Model ({EVAL_EPISODES} Episodes)')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.axvline(mean_reward, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {mean_reward:.2f}')
    plt.axvline(mean_reward + std_reward, color='orange', linestyle='dotted', linewidth=2,
                label=f'+1 Std: {mean_reward + std_reward:.2f}')
    plt.axvline(mean_reward - std_reward, color='orange', linestyle='dotted', linewidth=2,
                label=f'-1 Std: {mean_reward - std_reward:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot1_path = os.path.join('plots', 'eval_reward_histogram.png')
    plt.savefig(plot1_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'\nHistogram saved to: {plot1_path}')
    
    # Plot 2: Reward progression over evaluation episodes
    plt.figure(figsize=(12, 6))
    plt.plot(eval_rewards, alpha=0.6, linewidth=1)
    plt.axhline(mean_reward, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {mean_reward:.2f}')
    plt.fill_between(range(len(eval_rewards)), 
                     mean_reward - std_reward, 
                     mean_reward + std_reward, 
                     alpha=0.2, color='orange', 
                     label=f'±1 Std Dev')
    plt.title(f'Evaluation Rewards Across Episodes - Best Model')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot2_path = os.path.join('plots', 'eval_reward_progression.png')
    plt.savefig(plot2_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Progression plot saved to: {plot2_path}')

if __name__ == '__main__':
    main()
