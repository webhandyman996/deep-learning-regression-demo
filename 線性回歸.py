import numpy as np #隨機數據生成
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 模擬數據：房屋面積 (平方米) 與價格 (萬元)
np.random.seed(42)
n_samples = 100
TwSquare = 0.3025
# 房屋面積 (平方米) 與價格 (萬元) 的關係
area = np.random.uniform(50, 100 * TwSquare, n_samples).reshape(-1, 1)  # 房屋面積 (50 到 200 平方米)
#  # 房屋面積 (平方米) 與價格 (萬元) 的關係
# 房價 = 3 * 面積 + 隨機噪聲
price = 3 * area + np.random.randn(n_samples, 1) * 20  # 房價，帶噪聲

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(area, price, test_size=0.2, random_state=42)

# 訓練線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測房價
y_pred = model.predict(X_test)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 可視化結果
plt.scatter(area, price, color='blue', label='Observed Data')  # 原始數據
plt.plot(X_test, y_pred, color='red', label='Linear Regression Fit')  # 回歸線
plt.xlabel('Area (sqm)')
plt.ylabel('Price (10k)')
plt.legend()
plt.title('Linear Regression: Area vs Price')
plt.show()