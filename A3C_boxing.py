import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym

# Defining the Model's architecture
class Model(nn.Module):
    def __init__(self, action_size):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 128)
        self.fc2a = nn.Linear(128, action_size)
        self.fc2s = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        action_values = self.fc2a(x)
        state_value = self.fc2s(x)
        return action_values, state_value

# Preprocessing the Atari Frames
class PreprocessAtari(gym.ObservationWrapper):
    def __init__(self, env):
        super(PreprocessAtari, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(42, 42, 4), dtype=np.uint8)

    def observation(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (42, 42), interpolation=cv2.INTER_AREA)
        return img

# Creating the Environment
def make_env():
    env = gym.make("BoxingDeterministic-v4")
    env = PreprocessAtari(env)
    return env

# Agent class
class Agent():
    def __init__(self, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Model(action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        action_values, _ = self.network(state)
        action_probs = F.softmax(action_values, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def learn(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32, device=self.device) / 255.0
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device) / 255.0
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        self.optimizer.zero_grad()
        action_values, state_values = self.network(states)
        next_action_values, _ = self.network(next_states)

        selected_action_values = torch.gather(action_values, 1, actions)
        next_selected_action_values = torch.max(next_action_values, dim=1, keepdim=True)[0]

        target_values = rewards + 0.99 * next_selected_action_values * (1 - dones)
        advantage = target_values - state_values

        value_loss = F.smooth_l1_loss(state_values, target_values.detach())
        policy_loss = -(advantage.detach() * torch.log_softmax(action_values, dim=-1)).mean()

        loss = policy_loss + value_loss

        loss.backward()
        self.optimizer.step()

# Training the Agent
def train(agent, env, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        
# Creating the environment and agent
env = make_env()
action_size = env.action_space.n
agent = Agent(action_size)

# Training the agent
train(agent, env)
