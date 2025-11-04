import gymnasium as gym  # type: ignore
import ale_py  # Register ALE environments
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random
import gc  # For garbage collection
from collections import deque, namedtuple
from torch.optim.lr_scheduler import StepLR  # <-- IMPORTED SCHEDULER

# Limit CPU threads to prevent spawning too many processes
torch.set_num_threads(4)  # Limit PyTorch to 4 threads
torch.set_num_interop_threads(4)  # Limit inter-op parallelism
os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # Limit Intel MKL threads

# --- Hyperparameters ---
LEARNING_RATE = 0.0001  # Standard for Atari DQN
DISCOUNT_FACTOR = 0.99  # Required by assignment
NUM_EPISODES = 6000  # As suggested by assignment
EVAL_EPISODES = 500  # Required by assignment
STACK_SIZE = 4

# DQN-Specific Hyperparameters
REPLAY_BUFFER_SIZE = 50000  # Reduced to save RAM (~3 GB vs 56 GB)
BATCH_SIZE = 32  # Standard for Atari
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 1000000  # Reduced: decay over 1M steps (was 2M)
TARGET_UPDATE_FREQ = 5000  # Reduced: update every 5k steps (was 15k)
LEARNING_STARTS = 5000  # Reduced from 50k since buffer is smaller

# --- NEW: Scheduler Hyperparameters ---
SCHEDULER_STEP_SIZE = 1  # StepLR step_size (decay every call to scheduler.step())
SCHEDULER_GAMMA = 0.9  # Decay LR by 10%
SCHEDULER_CALL_FREQ = 1000000  # Call scheduler.step() every 1M global steps

GPU_ID = 4  # Change this to the appropriate GPU ID if needed

# Define the Transition tuple for the Replay Buffer
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# --- Image Preprocessing ---
# As specified in the assignment PDF
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

# --- Replay Buffer Definition ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- Epsilon-Greedy Action Selection ---
def select_action(state, policy_net, epsilon, n_actions, device):
    """
    Selects an action using epsilon-greedy policy.
    state is a (1, 4, 88, 80) torch tensor with int8 values.
    """
    if random.random() > epsilon:
        # Exploit: Choose the best action from the Q-network
        with torch.no_grad():
            # Convert state to float and normalize to [-1, 1]
            state_float = state.float().to(device) / 128.0
            return policy_net(state_float).max(1)[1].view(1, 1)
    else:
        # Explore: Choose a random action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

# --- Optimization Function ---
def optimize_model(policy_net, target_net, optimizer, memory, device):
    if len(memory) < BATCH_SIZE:
        return None  # Not enough samples in memory to train

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    
    # Filter out None states and concatenate
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    # Concatenate all tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    done_batch = torch.cat(batch.done).to(device)
    
    # Convert states to float and normalize to [-1, 1] for the network
    state_batch_float = state_batch.float().to(device) / 128.0
    non_final_next_states_float = non_final_next_states.float().to(device) / 128.0

    # 1. Get Q(s_t, a) from the policy network
    current_q_values = policy_net(state_batch_float).gather(1, action_batch)

    # 2. Get max Q(s_{t+1}, a') from the target network
    next_q_values = torch.zeros(BATCH_SIZE, device=device)
    
    with torch.no_grad():
        if non_final_next_states.shape[0] > 0:
            next_q_values[non_final_mask] = target_net(non_final_next_states_float).max(1)[0]

    # 3. Compute the target Q-value: R + gamma * max Q(s', a') * (1 - done)
    # When done=1 (terminal state), the future value is 0
    target_q_values = reward_batch + (DISCOUNT_FACTOR * next_q_values * (1 - done_batch))

    # 4. Compute Loss (Huber loss for stability)
    loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))

    # Optimization
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    
    return current_q_values.max().item()

# --- Main Training and Evaluation Function ---
def main():
    # 1. Setup Environment and Networks
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

    policy_net = QNetwork(action_size).to(device)
    target_net = QNetwork(action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    
    # --- NEW: Initialize scheduler ---
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    
    memory = ReplayBuffer(REPLAY_BUFFER_SIZE)

    all_episode_rewards = []
    all_max_q_values = []
    
    # --- NEW: Track best model ---
    best_avg_reward = -float('inf')
    os.makedirs('models', exist_ok=True)
    best_model_path = os.path.join('models', 'mspacman_dqn_best.pth')
    
    epsilon = EPSILON_START
    global_step = 0
    
    frame_stack = deque(maxlen=STACK_SIZE)

    # 2. Training Loop
    print("Starting training...")
    print(f"Learning will start after {LEARNING_STARTS} steps...")
    
    try:
        for episode in range(NUM_EPISODES):
            obs, _ = env.reset()
            proc_obs = preprocess_observation(obs)
            
            # Initialize frame stack with the first frame repeated
            for _ in range(STACK_SIZE):
                frame_stack.append(proc_obs)
            
            state_np = np.concatenate(list(frame_stack), axis=2)
            state_np_chw = state_np.transpose(2, 0, 1)  # (C, H, W)
            # State is a (1, 4, 88, 80) int8 tensor
            state = torch.tensor(state_np_chw, dtype=torch.int8).unsqueeze(0)
            
            episode_reward = 0
            episode_max_q = -float('inf')
            episode_steps = 0
            done = False

            while not done:
                # Select action
                action = select_action(state, policy_net, epsilon, action_size, device)
                
                # Take action
                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = bool(terminated or truncated)
                
                # Clip rewards to [-1, 1] for stability (common practice in Atari)
                clipped_reward = np.clip(reward, -1, 1)
                episode_reward += reward  # Track actual reward
                
                # Prepare tensors for replay buffer
                reward_tensor = torch.tensor([clipped_reward], dtype=torch.float32)

                if not done:
                    # Preprocess new frame and add to stack
                    proc_next_obs = preprocess_observation(next_obs)
                    frame_stack.append(proc_next_obs)
                    
                    # Create next_state tensor
                    next_state_np = np.concatenate(list(frame_stack), axis=2)
                    next_state_np_chw = next_state_np.transpose(2, 0, 1)
                    next_state = torch.tensor(next_state_np_chw, dtype=torch.int8).unsqueeze(0)
                else:
                    next_state = None  # Terminal state

                # Store transition (all tensors stored on CPU to save GPU memory)
                # CRITICAL: Detach tensors to break computation graph and prevent memory leak
                done_tensor = torch.tensor([float(done)], dtype=torch.float32)
                memory.push(state.detach().cpu(), action.detach().cpu(), 
                           next_state.detach().cpu() if next_state is not None else None, 
                           reward_tensor, done_tensor)

                if not done:
                    state = next_state

                # Increment global step counter
                global_step += 1
                episode_steps += 1
                
                # Start training after LEARNING_STARTS steps
                if global_step >= LEARNING_STARTS:
                    # Optimize model
                    batch_max_q = optimize_model(policy_net, target_net, optimizer, memory, device)
                    
                    if batch_max_q is not None:
                        episode_max_q = max(episode_max_q, batch_max_q)
                    
                    # Step the scheduler every SCHEDULER_CALL_FREQ global steps
                    if global_step % SCHEDULER_CALL_FREQ == 0:
                        scheduler.step()
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"   [Step {global_step}] Learning rate updated to: {current_lr:.6f}")
                    
                    # Linearly decay epsilon
                    if global_step > LEARNING_STARTS:
                        decay_step = global_step - LEARNING_STARTS
                        if decay_step < EPSILON_DECAY_STEPS:
                            epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (decay_step / EPSILON_DECAY_STEPS)
                        else:
                            epsilon = EPSILON_END
                    else:
                        epsilon = EPSILON_START # Still 1.0 before learning starts
                    
                    # Update target network based on steps
                    if global_step % TARGET_UPDATE_FREQ == 0:
                        target_net.load_state_dict(policy_net.state_dict())
                        # print(f"   [Step {global_step}] Target network updated")

                if done:
                    all_episode_rewards.append(episode_reward)
                    all_max_q_values.append(episode_max_q if episode_max_q != -float('inf') else 0)
                    break
            
            # --- MODIFIED: Print progress and save best model ---
            if (episode + 1) % 100 == 0:
                avg_reward = 0
                if len(all_episode_rewards) >= 100:
                    avg_reward = np.mean(all_episode_rewards[-100:])
                    
                    print(f'Episode {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.1f} | '
                          f'Avg(100): {avg_reward:.2f} | Steps: {episode_steps} | '
                          f'Epsilon: {epsilon:.4f} | Global Step: {global_step}')
                    
                    # --- NEW: Save best model logic ---
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        torch.save(policy_net.state_dict(), best_model_path)
                        print(f"  *** New best model saved with avg reward: {avg_reward:.2f} ***")
                else:
                    # Print simpler log if not enough episodes for 100-avg
                    print(f'Episode {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.1f} | '
                          f'Epsilon: {epsilon:.4f} | Global Step: {global_step}')
                
                # Force garbage collection every 100 episodes to free memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save checkpoint (optional, good for resuming)
            if (episode + 1) % 1000 == 0:
                checkpoint_path = os.path.join('models', f'mspacman_dqn_episode_{episode+1}.pth')
                torch.save({
                    'episode': episode + 1,
                    'policy_net_state_dict': policy_net.state_dict(),
                    'target_net_state_dict': target_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epsilon': epsilon,
                    'global_step': global_step,
                }, checkpoint_path)
                print(f"  Checkpoint saved to: {checkpoint_path}")

    except KeyboardInterrupt:
        print("\n\n*** Training interrupted by user ***")
        print(f"Saving emergency checkpoint at episode {episode+1}...")
        emergency_path = os.path.join('models', f'mspacman_dqn_emergency_ep{episode+1}.pth')
        torch.save({
            'episode': episode + 1,
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epsilon': epsilon,
            'global_step': global_step,
        }, emergency_path)
        print(f"Emergency checkpoint saved to: {emergency_path}")
        raise
    except Exception as e:
        print(f"\n\n*** FATAL ERROR at episode {episode+1}, step {global_step} ***")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nSaving emergency checkpoint...")
        emergency_path = os.path.join('models', f'mspacman_dqn_emergency_ep{episode+1}.pth')
        torch.save({
            'episode': episode + 1,
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epsilon': epsilon,
            'global_step': global_step,
        }, emergency_path)
        print(f"Emergency checkpoint saved to: {emergency_path}")
        raise

    print("Training finished.")
    
    # Save final model
    final_model_path = os.path.join('models', 'mspacman_dqn_final.pth')
    torch.save(policy_net.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # 3. Post-Training Evaluation
    
    # --- NEW: Load the *best* model for evaluation ---
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for evaluation...")
        policy_net.load_state_dict(torch.load(best_model_path))
    else:
        print("\nNo best model found, evaluating with final model.")
    
    policy_net.eval()  # Set policy net to evaluation mode
    
    print(f"Running final evaluation for {EVAL_EPISODES} episodes...")
    eval_rewards = []
    eval_frame_stack = deque(maxlen=STACK_SIZE)
    
    for eval_ep in range(EVAL_EPISODES):
        obs, _ = env.reset()
        proc_obs = preprocess_observation(obs)
        for _ in range(STACK_SIZE):
            eval_frame_stack.append(proc_obs)
        
        state_np = np.concatenate(list(eval_frame_stack), axis=2).transpose(2, 0, 1)
        state = torch.tensor(state_np, dtype=torch.int8).unsqueeze(0)
        
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                # Use greedy policy (epsilon = 0) for evaluation
                action = select_action(state, policy_net, 0.0, action_size, device)
                
                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = bool(terminated or truncated)
                
                if not done:
                    proc_next_obs = preprocess_observation(next_obs)
                    eval_frame_stack.append(proc_next_obs)
                    
                    next_state_np = np.concatenate(list(eval_frame_stack), axis=2).transpose(2, 0, 1)
                    state = torch.tensor(next_state_np, dtype=torch.int8).unsqueeze(0)
                
                episode_reward += reward

        eval_rewards.append(episode_reward)
        
        if (eval_ep + 1) % 100 == 0:
            print(f"  Evaluation episode {eval_ep + 1}/{EVAL_EPISODES} completed")
    
    env.close()

    # 4. Report and Plot Results
    mean_reward = float(np.mean(eval_rewards))
    std_reward = float(np.std(eval_rewards))
    print("\n--- Evaluation Results (using BEST model) ---")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Standard Deviation: {std_reward:.2f}")

    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Max Q-Values vs. Training Episodes
    plt.figure(figsize=(12, 6))
    plt.plot(all_max_q_values)
    plt.title('Max Q-Value vs. Training Episodes (MsPacman-v0)')
    plt.xlabel('Episode')
    plt.ylabel('Max Q-Value')
    plt.grid(True)
    plot1_path = os.path.join('plots', 'mspacman_v0_q_values.png')
    plt.savefig(plot1_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Plot 1 saved to: {plot1_path}')

    # Plot 2: Episode Rewards vs. Training Episodes with Moving Average
    plt.figure(figsize=(12, 6))
    plt.plot(all_episode_rewards, label='Episode Reward', alpha=0.6)
    if len(all_episode_rewards) >= 100:
        moving_avg = np.convolve(all_episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(all_episode_rewards)), moving_avg, 
                 label='100-Episode Moving Average', color='red', linewidth=2)
    plt.title('Episode Reward vs. Training Episodes for MsPacman-v0')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plot2_path = os.path.join('plots', 'mspacman_v0_rewards.png')
    plt.savefig(plot2_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Plot 2 saved to: {plot2_path}')

    # Plot 3: Histogram of Evaluation Rewards
    plt.figure(figsize=(12, 6))
    plt.hist(eval_rewards, bins=30, edgecolor='black')
    plt.title(f'Histogram of Rewards over {EVAL_EPISODES} Evaluation Episodes (Best Model)')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.axvline(mean_reward, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {mean_reward:.2f}')
    plt.legend()
    plt.grid(True)
    plot3_path = os.path.join('plots', 'mspacman_v0_eval_histogram.png')
    plt.savefig(plot3_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Plot 3 saved to: {plot3_path}')


if __name__ == '__main__':
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    main()