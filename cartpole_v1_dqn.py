import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import random
from collections import deque, namedtuple


# --- Stable & Proven Hyperparameters ---
LEARNING_RATE = 0.0001      # MUCH LOWER: This is the most critical fix.
DISCOUNT_FACTOR = 0.95
NUM_EPISODES = 5000         # Your 5000 is fine.
EVAL_EPISODES = 500

# DQN-Specific Hyperparameters
REPLAY_BUFFER_SIZE = 20000
BATCH_SIZE = 128            # Back to 128 for stability
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 4000  # 80% of total episodes is a good heuristic
TARGET_UPDATE_FREQ = 10     # Back to a standard update frequency
LEARNING_STARTS = 2000      # Start learning a bit earlier

GPU_ID = 4


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Increased network capacity
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state, policy_net, epsilon, n_actions, device):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(policy_net, target_net, optimizer, memory, device, batch_size):
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                   device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    current_q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = torch.zeros(batch_size, device=device)

    with torch.no_grad():
        if non_final_next_states.shape[0] > 0:
            next_q_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    target_q_values = reward_batch + (DISCOUNT_FACTOR * next_q_values)
    loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return current_q_values.max().item()


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(GPU_ID)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    print(f"Using device: {device}")

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = QNetwork(state_size, action_size).to(device)
    target_net = QNetwork(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    memory = ReplayBuffer(REPLAY_BUFFER_SIZE)

    all_episode_rewards = []
    all_max_q_values = []
    
    epsilon = EPSILON_START
    global_step = 0
    best_avg_reward = 0

    print(f"\nStarting training with DQN...")
    print(f"Extended exploration over {EPSILON_DECAY_STEPS} episodes")
    print(f"Learning starts after {LEARNING_STARTS} steps\n")

    for episode in range(NUM_EPISODES):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        episode_reward = 0
        episode_max_q = -float('inf')
        done = False

        while not done:
            action = select_action(state, policy_net, epsilon, action_size, device)
            step_result = env.step(action.item())

            if len(step_result) == 5:
                next_state_np, reward, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                next_state_np, reward, done, _ = step_result

            episode_reward += reward
            reward = torch.tensor([reward], device=device, dtype=torch.float32)

            if not done:
                next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None

            memory.push(state, action, next_state, reward, 
                       torch.tensor([done], device=device, dtype=torch.float32))

            if not done:
                state = next_state

            global_step += 1

            if global_step >= LEARNING_STARTS:
                batch_max_q = optimize_model(policy_net, target_net, optimizer, memory, device, BATCH_SIZE)
                if batch_max_q is not None:
                    episode_max_q = max(episode_max_q, batch_max_q)

            if done:
                all_episode_rewards.append(episode_reward)
                all_max_q_values.append(episode_max_q if episode_max_q != -float('inf') else 0)
                break

        # Linear epsilon decay over EPSILON_DECAY_STEPS episodes
        if episode < EPSILON_DECAY_STEPS:
            epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY_STEPS
        else:
            epsilon = EPSILON_END

        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(all_episode_rewards[-100:]) if len(all_episode_rewards) >= 100 else np.mean(all_episode_rewards)
            print(f'Episode {episode+1}/{NUM_EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.4f}')

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                os.makedirs('models', exist_ok=True)
                torch.save(policy_net.state_dict(), 'models/cartpole_best.pth')
                print(f"  → Best model saved! Avg: {avg_reward:.2f}")

            if avg_reward >= 475:
                print(f"Solved at episode {episode+1}!")
                break

    print("Training finished.")

    # Load best model
    if os.path.exists('models/cartpole_best.pth'):
        policy_net.load_state_dict(torch.load('models/cartpole_best.pth'))
        print(f"\nLoaded best model (training avg: {best_avg_reward:.2f})")

    # Evaluation
    print(f"\nRunning evaluation for {EVAL_EPISODES} episodes...")
    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
                step_result = env.step(action.item())

                if len(step_result) == 5:
                    state_np, reward, terminated, truncated, _ = step_result
                    done = bool(terminated or truncated)
                else:
                    state_np, reward, done, _ = step_result

                if not done:
                    state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)

                episode_reward += reward

        eval_rewards.append(episode_reward)

    env.close()

    mean_reward = float(np.mean(eval_rewards))
    std_reward = float(np.std(eval_rewards))
    print("\n--- Evaluation Results ---")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Standard Deviation: {std_reward:.2f}")

    # Plots
    os.makedirs('plots', exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(all_max_q_values)
    plt.title('Max Q-Value vs. Training Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Max Q-Value')
    plt.grid(True)
    plt.savefig('plots/cartpole_v1_q_values.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(all_episode_rewards, label='Episode Reward', alpha=0.6)
    if len(all_episode_rewards) >= 100:
        moving_avg = np.convolve(all_episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(all_episode_rewards)), moving_avg, 
                label='100-Episode Moving Average', color='red', linewidth=2)
    plt.title('Episode Reward vs. Training Episodes for CartPole-v1')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/cartpole_v1_rewards.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(eval_rewards, bins=30, edgecolor='black')
    plt.title(f'Histogram of Rewards over {EVAL_EPISODES} Evaluation Episodes')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.axvline(mean_reward, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_reward:.2f}')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/cartpole_v1_eval_histogram.png', bbox_inches='tight', dpi=300)
    plt.close()

    print('Plots saved.')


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()
