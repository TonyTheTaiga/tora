import os


import random


import time


from collections import deque


from typing import Dict, Tuple


import gym


import matplotlib.pyplot as plt


import numpy as np


import torch


import torch.nn as nn


import torch.optim as optim


from tora import Tora


def safe_value(value):
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0

        return float(value)

    elif isinstance(value, bool):
        return int(value)

    elif isinstance(value, str):
        return None

    else:
        try:
            return float(value)

        except (ValueError, TypeError):
            return None


def log_metric(client, name, value, step):
    value = safe_value(value)

    if value is not None:
        client.log(name=name, value=value, step=step)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size)
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        target_update_freq: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ):
        self.action_dim = action_dim

        self.batch_size = batch_size

        self.gamma = gamma

        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start

        self.epsilon_end = epsilon_end

        self.epsilon_decay = epsilon_decay

        self.device = device

        self.update_counter = 0

        self.q_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)

        self.target_network = DQN(state_dim, action_dim, hidden_dim).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size)

        self.criterion = nn.MSELoss()

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.q_network(state_tensor)

            return q_values.argmax().item()

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_tensor = torch.FloatTensor(states).to(self.device)

        actions_tensor = torch.LongTensor(actions).to(self.device)

        rewards_tensor = torch.FloatTensor(rewards).to(self.device)

        next_states_tensor = torch.FloatTensor(next_states).to(self.device)

        dones_tensor = torch.FloatTensor(dones).to(self.device)

        current_q_values = (
            self.q_network(states_tensor)
            .gather(1, actions_tensor.unsqueeze(1))
            .squeeze(1)
        )

        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]

            target_q_values = (
                rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
            )

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.update_counter += 1

        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()


def train_dqn(
    env_name: str, hyperparams: Dict, tora: Tora, seed: int = 42
) -> np.ndarray:
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    env = gym.make(env_name)

    env.seed(seed)

    episodes = hyperparams["episodes"]

    max_steps = hyperparams["max_steps"]

    hidden_dim = hyperparams["hidden_dim"]

    buffer_size = hyperparams["buffer_size"]

    batch_size = hyperparams["batch_size"]

    gamma = hyperparams["gamma"]

    lr = hyperparams["lr"]

    target_update_freq = hyperparams["target_update_freq"]

    epsilon_start = hyperparams["epsilon_start"]

    epsilon_end = hyperparams["epsilon_end"]

    epsilon_decay = hyperparams["epsilon_decay"]

    eval_frequency = hyperparams["eval_frequency"]

    eval_episodes = hyperparams["eval_episodes"]

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        lr=lr,
        target_update_freq=target_update_freq,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        device=device,
    )

    train_rewards = []

    eval_rewards = []

    episode_lengths = []

    losses = []

    epsilons = []

    best_eval_reward = float("-inf")

    start_time = time.time()

    total_steps = 0

    for episode in range(1, episodes + 1):
        state = env.reset()

        episode_reward = 0

        episode_loss = 0

        episode_step = 0

        done = False

        while not done and episode_step < max_steps:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)

            agent.buffer.push(state, action, reward, next_state, done)

            loss = agent.update()

            episode_loss += loss

            state = next_state

            episode_reward += reward

            episode_step += 1

            total_steps += 1

        train_rewards.append(episode_reward)

        episode_lengths.append(episode_step)

        epsilons.append(agent.epsilon)

        losses.append(episode_loss / max(1, episode_step))

        log_metric(tora, "train_reward", episode_reward, episode)

        log_metric(tora, "episode_length", episode_step, episode)

        log_metric(tora, "epsilon", agent.epsilon, episode)

        log_metric(tora, "loss", episode_loss / max(1, episode_step), episode)

        if episode % eval_frequency == 0:
            eval_reward = evaluate_agent(env, agent, eval_episodes, max_steps)

            eval_rewards.append(eval_reward)

            log_metric(tora, "eval_reward", eval_reward, episode)

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward

                torch.save(agent.q_network.state_dict(), "best_dqn_model.pt")

                log_metric(tora, "best_eval_reward", best_eval_reward, episode)

                print(f"New best model saved with reward: {best_eval_reward:.2f}")

        if episode % 10 == 0:
            elapsed_time = time.time() - start_time

            print(
                f"Episode {episode}/{episodes} | "
                f"Train reward: {episode_reward:.2f} | "
                f"Avg train reward (10 ep): {np.mean(train_rewards[-10:]):.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {episode_loss / max(1, episode_step):.5f} | "
                f"Steps: {episode_step} | "
                f"Elapsed time: {elapsed_time:.2f}s"
            )

    agent.q_network.load_state_dict(torch.load("best_dqn_model.pt"))

    final_eval_reward = evaluate_agent(env, agent, eval_episodes * 2, max_steps)

    log_metric(tora, "final_eval_reward", final_eval_reward, episodes)

    print(f"Final evaluation: {final_eval_reward:.2f}")

    log_metric(tora, "time_to_train", time.time() - start_time, episodes)

    log_metric(tora, "total_steps", total_steps, episodes)

    env.close()

    return np.array(train_rewards)


def evaluate_agent(env, agent, episodes: int, max_steps: int) -> float:
    total_reward = 0

    for _ in range(episodes):
        state = env.reset()

        episode_reward = 0

        done = False

        step = 0

        while not done and step < max_steps:
            action = agent.select_action(state, training=False)

            next_state, reward, done, _ = env.step(action)

            state = next_state

            episode_reward += reward

            step += 1

        total_reward += episode_reward

    return total_reward / episodes


def visualize_and_save_results(
    rewards: np.ndarray, eval_rewards: np.ndarray, smoothing_window: int = 10
):
    os.makedirs("results", exist_ok=True)

    def smooth(data, window):
        return np.convolve(data, np.ones(window) / window, mode="valid")

    smoothed_rewards = smooth(rewards, smoothing_window)

    episodes = range(1, len(smoothed_rewards) + 1)

    eval_episodes = range(10, len(eval_rewards) * 10 + 1, 10)

    plt.figure(figsize=(12, 6))

    plt.plot(
        episodes,
        smoothed_rewards,
        label=f"Training Reward (Smoothed, window={smoothing_window})",
    )

    plt.plot(
        eval_episodes, eval_rewards, label="Evaluation Reward", color="red", marker="o"
    )

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title("Training and Evaluation Rewards")

    plt.legend()

    plt.grid(True)

    plt.savefig("results/dqn_rewards.png")


if __name__ == "__main__":
    env_name = "CartPole-v1"

    hyperparams = {
        "episodes": 300,
        "max_steps": 500,
        "hidden_dim": 128,
        "buffer_size": 10000,
        "batch_size": 64,
        "gamma": 0.99,
        "lr": 1e-3,
        "target_update_freq": 10,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "eval_frequency": 10,
        "eval_episodes": 5,
    }

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.n

    env.close()

    hyperparams.update(
        {
            "env_name": env_name,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }
    )

    tora = Tora.create_experiment(
        name=f"DQN_{env_name}",
        description=f"Deep Q-Network agent for {env_name} environment",
        hyperparams=hyperparams,
        tags=["reinforcement-learning", "dqn", "cartpole"],
    )

    rewards = train_dqn(
        env_name=env_name,
        hyperparams=hyperparams,
        tora=tora,
    )

    eval_rewards = []

    for episode in range(
        10, hyperparams["episodes"] + 1, hyperparams["eval_frequency"]
    ):
        try:
            reward = tora.get_metric(name="eval_reward", step=episode).get("value", 0)

            eval_rewards.append(reward)

        except:
            pass

    visualize_and_save_results(rewards, np.array(eval_rewards))

    tora.shutdown()
