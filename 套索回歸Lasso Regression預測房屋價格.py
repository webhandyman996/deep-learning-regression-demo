"""
套索回歸 (Lasso Regression) 預測房屋價格
==========================================================
本程式使用 Lasso 回歸（L1 正則化線性回歸）以房屋面積、
房間數、樓層數三項特徵預測房屋價格（單位：萬元）。

Lasso 回歸原理：
  一般線性回歸最小化殘差平方和 (RSS)：
      RSS = Σ(yᵢ - ŷᵢ)²

  Lasso 在 RSS 之上加入 L1 懲罰項，最小化目標函數：
      J(w) = RSS + α · Σ|wⱼ|

  其中：
    α (alpha) ：正則化強度。α 越大，懲罰越重，係數被壓縮
                越多，甚至直接歸零（自動特徵選擇）。
    Σ|wⱼ|     ：所有特徵係數絕對值之和（L1 範數）。

Lasso 與 Ridge 的差異：
  ┌──────────┬─────────────────┬──────────────────┐
  │          │ Lasso (L1)      │ Ridge (L2)       │
  ├──────────┼─────────────────┼──────────────────┤
  │ 懲罰項   │ α · Σ|wⱼ|      │ α · Σwⱼ²        │
  │ 係數行為 │ 可壓縮為 0      │ 趨近 0 但不為 0  │
  │ 特徵選擇 │ 是（稀疏解）    │ 否               │
  └──────────┴─────────────────┴──────────────────┘
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ──────────────────────────────────────────────
# 1. 產生模擬資料
# ──────────────────────────────────────────────
# 固定隨機種子，確保每次執行結果一致
np.random.seed(42)
n_samples = 100  # 樣本數量

# 特徵矩陣：每列為一筆樣本，三個欄位分別代表
#   features[:, 0] — 房屋面積（平方米），範圍 50～200
#   features[:, 1] — 房間數，範圍 50～200（此處為示意比例數值）
#   features[:, 2] — 樓層數，範圍 50～200（此處為示意比例數值）
features = np.random.uniform(50, 200, (n_samples, 3))

# 目標變數：房屋價格（萬元）
# 公式：價格 = 0.8×面積 + 0.5×房間數 - 0.2×樓層數 + 隨機噪聲
# 注意：樓層數的係數為負值，代表樓層越高，房價反而略低（示意設定）
# 隨機噪聲模擬真實市場波動，標準差為 10 萬元
prices = (
    0.8  * features[:, 0]           # 面積對房價的正向貢獻
    + 0.5  * features[:, 1]         # 房間數對房價的正向貢獻
    - 0.2  * features[:, 2]         # 樓層數對房價的負向貢獻
    + np.random.randn(n_samples) * 10  # 高斯噪聲
)

# ──────────────────────────────────────────────
# 2. 分割訓練集與測試集
# ──────────────────────────────────────────────
# test_size=0.2：80% 用於訓練，20% 用於測試（共 20 筆測試樣本）
# random_state=42：固定分割方式，確保結果可重現
X_train, X_test, y_train, y_test = train_test_split(
    features, prices, test_size=0.2, random_state=42
)

# ──────────────────────────────────────────────
# 3. 訓練 Lasso 回歸模型
# ──────────────────────────────────────────────
# alpha：L1 正則化強度
#   - alpha=0   等同於普通線性回歸（無懲罰）
#   - alpha 越大，越多係數被壓縮至 0（特徵選擇效果越強）
#   - 此處設為 0.1，對係數施加輕度懲罰
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)  # 以訓練集擬合模型，求解各特徵係數

# 印出各特徵的回歸係數（可觀察哪些特徵被 Lasso 壓縮為 0）
print("各特徵回歸係數（面積、房間數、樓層數）：", lasso_model.coef_)
print(f"截距 (intercept)：{lasso_model.intercept_:.2f}")

# ──────────────────────────────────────────────
# 4. 預測房屋價格
# ──────────────────────────────────────────────
# 使用測試集進行預測，模型未見過這些資料
y_pred = lasso_model.predict(X_test)

# ──────────────────────────────────────────────
# 5. 評估模型效能
# ──────────────────────────────────────────────
# 均方誤差 (MSE)：MSE = (1/n) × Σ(y_actual - y_pred)²
# 值越小代表預測越準確，單位為萬元的平方
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# ──────────────────────────────────────────────
# 6. 視覺化預測結果
# ──────────────────────────────────────────────
# 散點圖：x 軸為實際房價，y 軸為模型預測房價
# 理想情況下所有點應落在 y=x 的對角線上
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')

# 參考線：y=x 的理想擬合線（完美預測時所有點落在此線上）
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red', linestyle='--', label='Ideal Fit'
)

plt.xlabel('Actual Prices (10k)')    # X 軸：實際房價（萬元）
plt.ylabel('Predicted Prices (10k)') # Y 軸：預測房價（萬元）
plt.legend()
plt.title('Lasso Regression: Predicted vs Actual Prices')
plt.show()