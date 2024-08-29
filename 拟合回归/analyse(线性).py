import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 读取上传的xlsx文件
file_path = '副本特征数据.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 数据清理：去掉不必要的列和缺失值
data_cleaned = data.drop(columns=['Unnamed: 1', 'Unnamed: 3', 'Unnamed: 5']).dropna()

# 选择特征和目标变量
X = data_cleaned[['length/mm', 'width/mm']]
y = data_cleaned['weight/kg']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# 输出回归系数
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# 可视化展示
plt.figure(figsize=(14, 7))

# 散点图：体长 vs 体重
plt.subplot(1, 2, 1)
plt.scatter(data_cleaned['length/mm'], data_cleaned['weight/kg'], color='blue')
plt.plot(X_test['length/mm'], y_pred, color='red', linewidth=2)
plt.xlabel('length/mm')
plt.ylabel('weight/kg')
plt.title('Length vs Weight')

# 散点图：体宽 vs 体重
plt.subplot(1, 2, 2)
plt.scatter(data_cleaned['width/mm'], data_cleaned['weight/kg'], color='green')
plt.plot(X_test['width/mm'], y_pred, color='red', linewidth=2)
plt.xlabel('width/mm')
plt.ylabel('weight/kg')
plt.title('Width vs Weight')

plt.tight_layout()
plt.show()
