import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class Dataloader:
    def __init__(self,file):
        self.data = pd.read_csv(file)
        self.features = self.data.drop('bug',axis=1)
        self.labels = self.data['bug']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocess_data()

    def preprocess_data(self):
        # input := np.array
        # thats where we should modify the signals, e.g. standardization
        # LQ's change
        # 移除高VIF的特征
        # self.features = self.features.drop(['cbo', 'ca', 'ce'], axis=1)
        # 移除低相关性特征
        # self.features = self.features.drop(['cbm', 'noc', 'mfa', 'ic'], axis=1)

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 应用SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

        # 更新类属性
        self.X_train, self.X_test, self.y_train, self.y_test = X_resampled, X_test_scaled, y_resampled, y_test



