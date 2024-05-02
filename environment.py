import numpy as np
import logging



class Environment:
    def __init__(self,features,labels,accuracy_threshold=0.75):
        self.features = features
        self.labels = labels
        self.num_samples = len(labels)
        self.accuracy_threshold = accuracy_threshold
        self.reset()

    def reset(self):
        # 重置环境状态
        self.current_index = 0
        self.correct_predictions = 0
        self.cumulative_reward = 0  # 累计奖励初始化
        return self.features.iloc[0]

    def step(self, action):
        accuracy = 0.0
        # 获取当前动作的正确标签
        correct_action = self.labels.iloc[self.current_index]

        # 即时奖励机制
        if action == correct_action:
            reward = 0.1  # 较小的正奖励
            self.correct_predictions += 1  # 增加正确预测的计数
        else:
            reward = -0.1  # 较小的负奖励
        # 日志输出当前动作、正确动作和奖励
        # logging.info(f"Action: {action}, Correct: {correct_action}, Reward: {reward}, Correct Predictions: {self.correct_predictions}")

        # 更新累计奖励
        self.cumulative_reward += reward
        # 检查是否达到提供额外奖励的条件
        if self.cumulative_reward >= 1:
            reward += 0.5  # 提供额外的正奖励
            self.cumulative_reward -= 1  # 重置累计奖励

        # 更新到下一个状态
        self.current_index += 1

        # 判断是否完成所有数据的处理，即是否到达episode的末尾
        done = self.current_index >= self.num_samples

        # 如果到达episode的末尾，根据整个episode的准确率给出总体奖励
        if done:
            accuracy = self.correct_predictions / self.num_samples
            if accuracy >= self.accuracy_threshold:
                reward = 5  # 较大的正奖励，表示整个episode的准确率达到了目标阈值
            else:
                reward = -5  # 较大的负奖励，表示整个episode的准确率未达到目标阈值
            self.reset()  # 重置环境为下一个episode做准备
            logging.info(f"Episode End: Accuracy={accuracy}, Final Reward={reward}")

        # 获取下一个状态
        next_state = self.features.iloc[self.current_index % self.num_samples]

        return next_state, reward, done, accuracy

    def get_accuracy(self):
        if self.num_samples == 0:  # 防止 num_samples 未正确设置时除以零
            return 0
        return self.correct_predictions / self.num_samples