import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''
嶺回歸的作用：
它透過在 目標函數 中加入額外的懲罰，來使模型避免過度依賴某些特徵。
適用於 多重共線性問題（當特徵之間高度相關時，標準線性回歸可能會導致不穩定的係數）。
它保留了所有變數，但強迫它們的係數變得較小，避免某些變數對結果影響過大。
'''
"""
嶺回歸 (Ridge Regression) 預測商品銷售額與廣告關係 - 互動式 Alpha 調整版（含異常資料）
==========================================================
本程式使用嶺回歸（L2 正則化線性回歸）以三種廣告渠道的支出
預測商品銷售額，並提供滑桿讓使用者即時調整正則化強度 Alpha，
觀察 Alpha 對預測結果與 MSE 的影響。

嶺回歸原理：
  一般線性回歸最小化殘差平方和 (RSS)：
      RSS = Σ(yᵢ - ŷᵢ)²

  嶺回歸在 RSS 之上加入 L2 懲罰項，最小化目標函數：
      J(w) = RSS + α · Σwⱼ²

  其中：
    α (alpha) ：正則化強度。α 越大，懲罰越重，係數被壓縮
                越趨近於 0（但不會等於 0），有效抑制過擬合。
    Σwⱼ²      ：所有特徵係數的平方和（L2 範數的平方）。

本版本特色（異常版）：
  資料中含有離群值（異常樣本），用以觀察 Ridge 正則化
  對異常資料的抑制效果。調整 Alpha 可看出正則化強度越高，
  模型對異常值的敏感度越低，預測曲線越平穩。

互動功能：
  拖曳底部「Alpha」滑桿（範圍 0.1～10.0）可即時重新訓練
  模型，並更新散點圖與圖形標題上的 MSE 數值。
"""
# 模擬數據：多種廣告支出 (千元) 與商品銷售額 (千元)
np.random.seed(42)
n_samples = 100
# 廣告支出：電視、網路、報紙 (三個特徵)
ad_spend = np.random.uniform(1, 100, (n_samples, 3))  # 每列代表不同廣告渠道的支出
# 銷售額：基於廣告支出生成，並添加噪聲
sales = 0.5 * ad_spend[:, 0] + 0.3 * ad_spend[:, 1] + 0.2 * ad_spend[:, 2] + np.random.randn(n_samples) * 10

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(ad_spend, sales, test_size=0.2, random_state=42)

# 訓練嶺回歸模型
'''
alpha 是正則化強度參數，控制正則化項的權重(W)。
嶺回歸的損失函數為：
L(w) = ||y - Xw||^2 + alpha * ||w||^2  {||w||^2是所有元素的平方和}
第一項是普通最小二乘法的損失（殘差平方和）。
第二項是正則化項，||w||^2 是權重向量的 L2 範數平方，用於懲罰過大的權重，防止過擬合。
alpha 控制正則化項的影響力：
    當 alpha 越大時，正則化越強，模型會更簡單，但可能欠擬合。
    當 alpha 越小時，正則化越弱，模型更接近普通線性回歸，但可能過擬合。
'''
ridge_model = Ridge(alpha=9.0)  # alpha 是正則化強度參數
ridge_model.fit(X_train, y_train)

# 預測銷售額
y_pred = ridge_model.predict(X_test)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 可視化結果（僅展示測試集的真實值與預測值）
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Sales (thousands)')
plt.ylabel('Predicted Sales (thousands)')
plt.legend()
plt.title('Ridge Regression: Predicted vs Actual Sales')
plt.show()