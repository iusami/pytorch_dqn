import numpy as np
import time

import gym
from gym import wrappers

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#https://elix-tech.github.io/ja/2016/06/29/dqn-ja.html
def huberloss(y_true, y_pred):
    err = torch.abs((y_true - y_pred).float())
    quad_part = torch.clamp(err,0.0, 1.0)
    linear_part = err - quad_part
    loss = torch.mean(0.5 * torch.mul(quad_part, quad_part) + linear_part)
    return loss

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.dense1 = nn.Linear(4,16)
        self.dense2 = nn.Linear(16,16)
        self.dense3 = nn.Linear(16,2)
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

def replay(model, criterion, optimizer, memory, batch_size, gamma, targetQN, use_gpu):
    inputs = np.zeros((batch_size, 4))
    targets = np.zeros((batch_size, 2))
    mini_batch = memory.sample(batch_size)

    for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
        equall_flag = 0
        inputs[i:i + 1] = state_b
        target = reward_b

        model.eval()

        if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
            equall_flag = 1
            next_state_b = torch.from_numpy(next_state_b).float()
            if use_gpu :
                next_state_b = next_state_b.to("cuda")
            retmainQs = model(next_state_b)[0]
            #print("retmainQs:",retmainQs)
            next_act = np.argmax(retmainQs.cpu().detach().numpy())
            #print("next_act:",next_act)
            target = reward_b + gamma * targetQN(next_state_b)[0][next_act]
            #print("targetQN(next_state_b)[0]:",targetQN(next_state_b)[0])

        state_b = torch.from_numpy(state_b).float()
        inputs_tensor = torch.from_numpy(inputs).float()
        if use_gpu:
            state_b = state_b.to("cuda")
            inputs_tensor = inputs_tensor.to("cuda")
        targets[i] = model(state_b).cpu().detach().numpy()
        
        if equall_flag == 1:
            targets[i][action_b] = target.cpu().detach().numpy()
        else:
            targets[i][action_b] = target

        model.train()
        outputs = model(inputs_tensor)
        loss = criterion(outputs, torch.from_numpy(targets).float().to("cuda"))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]
    
    def len(self):
        return len(self.buffer)

class Actor:
    def get_action(self, state, episode, targetQN, use_gpu):
        epsilon = 0.001 + 0.9 / (1.0+episode)
        
        if epsilon <= np.random.uniform(0, 1):
            targetQN.eval()
            state = torch.from_numpy(state).float()
            if use_gpu==True:
                state = state.to("cuda")
            retTargetQs = targetQN(state)[0]
            action = np.argmax(retTargetQs.cpu().detach().numpy())
        
        else:
            action = np.random.choice([0, 1])
        
        return action

DQN_MODE = 1
LENDER_MODE = 1

env = gym.make("CartPole-v0")
num_episodes = 299  # 総試行回数
max_number_of_steps = 200  # 1試行のstep数
goal_average_reward = 195  # この報酬を超えると学習終了
num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納
gamma = 0.99    # 割引係数
islearned = 0  # 学習が終わったフラグ
isrender = 0  # 描画フラグ
# ---
hidden_size = 16               # Q-networkの隠れ層のニューロンの数
learning_rate = 0.00001         # Q-networkの学習係数
memory_size = 10000            # バッファーメモリの大きさ
batch_size = 32                # Q-networkを更新するバッチの大記載
use_gpu = True

mainQN = QNet()
targetQN = QNet()
if use_gpu:
    mainQN.cuda()
    targetQN.cuda()
memory = Memory(max_size=memory_size)
actor = Actor()

criterion = huberloss
optimizer = optim.Adam(mainQN.parameters(), lr = learning_rate)

for episode in range(num_episodes):
    env.reset()
    state, reward, done, _ = env.step(env.action_space.sample())
    state = np.reshape(state, [1,4])
    episode_reward = 0

    targetQN = mainQN

    for t in range(max_number_of_steps + 1):
        env.render()
        if (islearned == 1) and LENDER_MODE:
            env.render()
            time.sleep(0.1)
            print(state[0,0])

        action = actor.get_action(state, episode, mainQN, use_gpu)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1,4])

        if done:
            next_state = np.zeros(state.shape)
            if t < 195:
                reward = -1
            else:
                reward = 1
        else:
            reward = 0
        
        episode_reward += 1

        memory.add((state,action,reward,next_state))
        state = next_state

        if (memory.len() > batch_size) and not islearned:
            mainQN = replay(mainQN, criterion, optimizer, memory, batch_size, gamma, targetQN, use_gpu)
        
        if DQN_MODE:
            targetQN = mainQN
        
        if done:
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))
            print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, total_reward_vec.mean()))
            #print(type(total_reward_vec))
            #print(total_reward_vec)
            break
    
    if total_reward_vec.mean() >= goal_average_reward:
        print('Episode %d train agent successfuly!' % episode)
        islearned = 1
        if isrender == 0:   # 学習済みフラグを更新
            isrender = 1
            env = wrappers.Monitor(env, './movie/cartpoleDQN',force=True)  # 動画保存する場合
env.close()