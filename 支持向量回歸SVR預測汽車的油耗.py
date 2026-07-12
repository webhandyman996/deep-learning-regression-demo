import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
"""
支持向量回歸 (SVR) 預測汽車油耗 - 互動式參數調整版
==========================================================
本程式使用支持向量回歸 (Support Vector Regression, SVR) 搭配 RBF 核函數，
以汽車的車重、引擎大小、氣缸數預測每百公里油耗 (L/100km)。
提供三個滑桿讓使用者即時調整 SVR 的超參數 C、Gamma、Epsilon，
並在散點圖上即時更新預測結果與均方誤差 (MSE)。

SVR 超參數說明：
  C (懲罰參數)  ：控制對訓練誤差的容忍度。C 越大，模型越嚴格擬合訓練資料
                  (低偏差、高變異)；C 越小，允許更多誤差 (高偏差、低變異)。
  Gamma         ：RBF 核函數的寬度參數，決定單一訓練樣本的影響範圍。
                  Gamma 越大，影響範圍越窄 (容易過擬合)；越小，影響範圍越廣。
  Epsilon (ε)   ：不敏感帶寬度。在 ε 範圍內的誤差不計入損失函數，
                  控制支持向量的數量與模型的稀疏程度。
"""
# 模擬數據：汽車特徵 (車重、引擎大小、氣缸數) 與油耗 (單位：L/100km)
np.random.seed(42)
n_samples = 100
# 汽車特徵：車重 (kg)、引擎大小 (L)、氣缸數
features = np.random.uniform(1000, 3000, (n_samples, 3))  # 每列代表不同特徵
# 油耗：基於特徵生成，並添加噪聲
fuel_consumption = 0.005 * features[:, 0] + 0.8 * features[:, 1] + 0.3 * features[:, 2] + np.random.randn(n_samples) * 2

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features, fuel_consumption, test_size=0.2, random_state=42)

# 訓練支持向量回歸模型
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)  # 使用 RBF 核
svr_model.fit(X_train, y_train)

# 預測油耗
y_pred = svr_model.predict(X_test)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 可視化結果（僅展示測試集的真實值與預測值）
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Fuel Consumption (L/100km)')
plt.ylabel('Predicted Fuel Consumption (L/100km)')
plt.legend()
plt.title('Support Vector Regression: Predicted vs Actual Fuel Consumption')
plt.show()