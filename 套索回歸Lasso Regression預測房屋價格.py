import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 模擬數據：房屋特徵 (面積、房間數、樓層數) 與房價 (單位：萬元)
np.random.seed(42)
n_samples = 100
# 房屋特徵：面積 (平方米)、房間數、樓層數
features = np.random.uniform(50, 200, (n_samples, 3))  # 每列代表不同特徵
# 房價：基於特徵生成，並添加噪聲
prices = 0.8 * features[:, 0] + 0.5 * features[:, 1] - 0.2 * features[:, 2] + np.random.randn(n_samples) * 10

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)

# 訓練套索回歸模型
lasso_model = Lasso(alpha=0.1)  # alpha 是正則化強度參數
lasso_model.fit(X_train, y_train)

# 預測房價
y_pred = lasso_model.predict(X_test)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 可視化結果（僅展示測試集的真實值與預測值）
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Prices (10k)')
plt.ylabel('Predicted Prices (10k)')
plt.legend()
plt.title('Lasso Regression: Predicted vs Actual Prices')
plt.show()