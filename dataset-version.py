import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

df = pd.read_csv('new_merged_dataset.csv')
pd.set_option('display.max_columns', None)  # None意味着不限制显示的列数
pd.set_option('display.max_rows', None)  # 如果你也想显示所有行，可以设置这个选项
# print(df['cbo'].describe())


'''低相关性查找'''
# Calculate the correlation matrix
correlation_matrix = df.corr()

# Extract the correlation values with the target variable 'bug'
feature_correlation_with_bug = correlation_matrix['bug'].sort_values()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Generate a bar plot
sns.barplot(x=feature_correlation_with_bug.values, y=feature_correlation_with_bug.index, palette="viridis")

# Add labels and title
plt.title('Correlation of Features with the Target Variable "bug"')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')

# Show the plot
# plt.tight_layout()
# plt.show()


'''随机森林特征重要性评估'''
X = df.drop('bug', axis=1)
y = df['bug']
# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 获取特征重要性
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

# print(feature_importances)


'''VIF'''
# 标准化数据，因为VIF对尺度敏感
scaler = StandardScaler()
df_reduced = df.drop(['cbo', 'ca'], axis=1)  # 移除高 VIF 特征
# 重新进行标准化
features_scaled_reduced = scaler.fit_transform(df_reduced.drop(['bug'], axis=1))
df_features_reduced = pd.DataFrame(features_scaled_reduced, columns=df_reduced.columns[:-1])

# 重新计算 VIF
vif_data_reduced = pd.DataFrame()
vif_data_reduced['Feature'] = df_features_reduced.columns
vif_data_reduced['VIF'] = [variance_inflation_factor(df_features_reduced.values, i) for i in range(df_features_reduced.shape[1])]

# 打印新的 VIF 值
# print(vif_data_reduced.sort_values(by='VIF', ascending=False))


'''PCA'''
# 标准化数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df.drop(['bug'], axis=1))

# 应用 PCA
pca = PCA(n_components=0.95)  # 保留 95% 的方差
features_reduced = pca.fit_transform(features_scaled)

# 查看主成分数量和解释方差
print("Number of components:", pca.n_components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)

'''pca的准确率'''
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(features_reduced, df['bug'], test_size=0.3, random_state=42)

# 训练模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 预测与评估
predictions = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, predictions))
