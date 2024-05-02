import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Updated discount factor
        self.epsilon = 1.0
        self.epsilon_decay = 0.995  # Slower decay
        self.epsilon_min = 0.01
        self.learning_rate = 0.001  # Lower learning rate
        self.batch_size = batch_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))  # Increased complexity
        model.add(Dense(64, activation='relu'))  # Increased complexity
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # 将评估网络的权重复制到目标网络
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # 存储经验
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 基于当前状态做出行动决策
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

            # 随机抽取一个批量的经验
        minibatch = random.sample(self.memory, batch_size)

        # 初始化两个数组，用于批量更新网络
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward  # 这里的奖励是基于整体准确率计算的，且对每个步骤相同
            if not done:
                # 使用目标网络来估计最大未来回报
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

            # 获取模型对当前状态的预测目标值，并更新采取的行动的目标值
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # 保存状态和更新后的目标值，稍后一起进行批量更新
            states[i] = state
            targets[i] = target_f

        # 执行一次批量更新，而不是每次循环更新一次
        self.model.fit(states, targets, epochs=1, verbose=0)

        # 递减 epsilon，减少随机行动
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay