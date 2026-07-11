import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
'''
多項式回歸（Polynomial Regression）
是線性回歸的擴展，透過增加變數的多項式項來捕捉非線性關係。
'''
# 生成樣本數據
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # 特徵數據
 # 標籤數據，帶有噪聲
# y = 2 * X**2 + 3 * X + 5 + noise(a * x^2 + b * x + c + noise)
# 二次項，x 的平方乘以係數 a 決定了曲線的彎曲程度。
# 一次項，x 乘以係數 b 決定了曲線的斜率。常數項 c 決定了曲線的垂直位置。
# a > 0，拋物線開口向上；如果 a < 0，拋物線開口向下。
# 生成基於二次多項式的標籤數據，並添加隨機噪聲
y = 2 * X**2 + 3 * X + 5 + np.random.randn(100, 1) * 10

# 可視化原始數據
plt.scatter(X, y, color='blue', label='Original Data')

# 構建多項式特徵
poly = PolynomialFeatures(degree=2)  # 二次多項式
X_poly = poly.fit_transform(X)

# 訓練多項式回歸模型
model = LinearRegression()
model.fit(X_poly, y)

# 預測
X_fit = np.linspace(0, 10, 100).reshape(-1, 1)  # 測試數據
X_fit_poly = poly.transform(X_fit)
y_fit = model.predict(X_fit_poly)

# 可視化回歸結果
plt.plot(X_fit, y_fit, color='red', label='Polynomial Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Regression Example')
plt.show()