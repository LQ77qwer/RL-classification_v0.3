import logging
from dataloader import Dataloader
from agent import DQNAgent
from environment import Environment
import numpy as np
import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler("training_log.txt"),
                        logging.StreamHandler()
                    ])

dataloader = Dataloader('mini_dataset.csv')
env = Environment(dataloader.features, dataloader.labels)
state_size = dataloader.features.shape[1]
action_size = 2  # 假设二分类任务

agent = DQNAgent(state_size, action_size)
episodes = 1000

for e in range(episodes):
    state = env.reset()
    state = np.array(state).reshape(1, state_size)
    total_reward = 0

    for time_step in range(env.num_samples):
        action = agent.act(state)
        next_state, reward, done, accuracy = env.step(action)
        next_state = np.array(next_state).reshape(1, state_size)

        # 打印当前状态和动作信息
        # logging.info(f"Episode: {e + 1}, Time Step: {time_step + 1}, Current State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

        # 存储每一步的记忆
        agent.remember(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

        if done:
            # accuracy = env.get_accuracy()
            logging.info(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}, Accuracy: {accuracy:.2f}")
            break

    # 经验回放
    agent.replay(agent.batch_size)

    # 更新目标网络
    if e % 10 == 0:
        agent.update_target_model()
