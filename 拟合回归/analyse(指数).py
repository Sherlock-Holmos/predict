import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import make_pipeline

# 读取数据
file_path = '副本特征数据.xlsx'
df = pd.read_excel(file_path)

# 数据准备
X = df[['length/mm', 'width/mm']]
y = df['weight/kg']

# 线性回归
linear_model = LinearRegression()
linear_model.fit(X, y)
predictions_linear = linear_model.predict(X)

plt.scatter(df['length/mm'], y, color='blue', label='Actual')
plt.plot(df['length/mm'], predictions_linear, color='red', label='Linear Fit')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.title('Linear Regression')
plt.legend()
plt.show()

# 多项式回归
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
predictions_poly = poly_model.predict(X_poly)

plt.scatter(df['length/mm'], y, color='blue', label='Actual')
plt.plot(df['length/mm'], predictions_poly, color='green', label='Polynomial Fit')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.title('Polynomial Regression')
plt.legend()
plt.show()


# 指数回归
exp_transformer = FunctionTransformer(np.log1p, validate=True)
X_exp = exp_transformer.fit_transform(X)
exp_model = LinearRegression()
exp_model.fit(X_exp, y)

# 预测
predictions_exp = exp_model.predict(X_exp)

# 将预测值重塑为二维数组
predictions_exp = predictions_exp.reshape(-1, 1)

# 反变换
inverse_exp = exp_transformer.inverse_transform(np.zeros_like(predictions_exp))  # 创建一个适合的二维数组作为反变换的基准
predictions_exp = np.exp(predictions_exp) - 1  # 计算指数反变换

plt.scatter(df['length/mm'], y, color='blue', label='Actual')
plt.plot(df['length/mm'], predictions_exp, color='orange', label='Exponential Fit')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.title('Exponential Regression')
plt.legend()
plt.show()

# 对数回归
log_transformer = FunctionTransformer(np.log1p, validate=True)
X_log = log_transformer.fit_transform(X)
log_model = LinearRegression()
log_model.fit(X_log, y)
predictions_log = log_transformer.inverse_transform(log_model.predict(X_log))

plt.scatter(df['length/mm'], y, color='blue', label='Actual')
plt.plot(df['length/mm'], predictions_log, color='purple', label='Logarithmic Fit')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.title('Logarithmic Regression')
plt.legend()
plt.show()
