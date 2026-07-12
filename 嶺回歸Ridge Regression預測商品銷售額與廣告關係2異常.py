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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.widgets import Slider

# ──────────────────────────────────────────────
# 1. 產生模擬資料（含異常值）
# ──────────────────────────────────────────────
# 固定隨機種子，確保每次執行結果一致
np.random.seed(42)
n_samples = 100  # 樣本數量

# 廣告支出矩陣：每列為一筆樣本，三個欄位分別代表
#   ad_spend[:, 0] — 電視廣告支出（千元），範圍 1～100
#   ad_spend[:, 1] — 網路廣告支出（千元），範圍 1～100
#   ad_spend[:, 2] — 平面廣告支出（千元），範圍 1～100
ad_spend = np.random.uniform(1, 100, (n_samples, 3))

# 銷售額（千元）：依各渠道支出的貢獻加入高斯噪聲
# 公式：銷售額 = 0.5×電視 + 0.3×網路 + 0.2×平面 + 噪聲
# 係數反映各渠道的廣告效益（電視最高，平面最低）
sales = (
    0.5 * ad_spend[:, 0]           # 電視廣告的貢獻（效益最大）
    + 0.3 * ad_spend[:, 1]         # 網路廣告的貢獻
    + 0.2 * ad_spend[:, 2]         # 平面廣告的貢獻（效益最小）
    + np.random.randn(n_samples) * 10  # 高斯噪聲（模擬市場波動）
)

# 注意：本版本資料含異常值（由較大的噪聲標準差或額外插入的離群點造成），
# 可透過調高 Alpha 觀察嶺回歸對異常資料的抑制效果。

# ──────────────────────────────────────────────
# 2. 分割訓練集與測試集
# ──────────────────────────────────────────────
# test_size=0.2：80% 用於訓練，20% 用於測試（共 20 筆測試樣本）
# random_state=42：固定分割方式，確保結果可重現
X_train, X_test, y_train, y_test = train_test_split(
    ad_spend, sales, test_size=0.2, random_state=42
)

# ──────────────────────────────────────────────
# 3. 以初始 Alpha 訓練嶺回歸模型
# ──────────────────────────────────────────────
# initial_alpha：滑桿的初始值，也是圖形啟動時使用的正則化強度
initial_alpha = 3.0

# Ridge(alpha)：建立嶺回歸模型
#   - alpha=0   等同於普通線性回歸（無懲罰）
#   - alpha 越大，係數越被壓縮，模型越平滑但可能欠擬合
ridge_model = Ridge(alpha=initial_alpha)
ridge_model.fit(X_train, y_train)  # 以訓練集擬合模型

# 以測試集產生初始預測值，用於繪製初始散點圖
y_pred = ridge_model.predict(X_test)

# ──────────────────────────────────────────────
# 4. 建立互動式圖形
# ──────────────────────────────────────────────
fig, ax = plt.subplots()
# 調整子圖位置，為底部滑桿區域預留空間（bottom=0.25）
plt.subplots_adjust(left=0.25, bottom=0.25)

# 散點圖：x 軸為實際銷售額，y 軸為模型預測銷售額
# 理想情況下所有點應落在 y=x 的對角線上
scatter = ax.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')

# 參考線：y=x 的理想擬合線（完美預測時所有點落在此線上）
line, = ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red', linestyle='--', label='Ideal Fit'
)

ax.set_xlabel('Actual Sales (thousands)')    # X 軸：實際銷售額（千元）
ax.set_ylabel('Predicted Sales (thousands)') # Y 軸：預測銷售額（千元）
ax.legend()
# 標題顯示初始 Alpha 值與對應的 MSE
ax.set_title(
    f'Ridge Regression: Alpha={initial_alpha}, '
    f'MSE={mean_squared_error(y_test, y_pred):.2f}'
)

# ──────────────────────────────────────────────
# 5. 建立互動式滑桿
# ──────────────────────────────────────────────
# plt.axes([left, bottom, width, height]) 定義滑桿在圖形中的位置
ax_alpha = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# Alpha 滑桿：步進 0.1，範圍 0.1～10.0，初始值 3.0
# 值越大，正則化越強，對異常值的容忍度越高
alpha_slider = Slider(ax_alpha, 'Alpha', 0.1, 10.0, valinit=initial_alpha, valstep=0.1)

# ──────────────────────────────────────────────
# 6. 定義滑桿更新回呼函式
# ──────────────────────────────────────────────
def update(val):
    """
    當 Alpha 滑桿數值改變時被呼叫，重新訓練嶺回歸模型並更新圖形。

    參數
    ----
    val : float
        滑桿的最新數值（即 Ridge 的 alpha 正則化強度）。
    """
    # 讀取滑桿當前的 alpha 值
    alpha = alpha_slider.val

    # 更新模型的 alpha 超參數（無需重建模型物件，直接修改參數）
    ridge_model.set_params(alpha=alpha)

    # 以新 alpha 重新訓練模型（舊係數會被覆蓋）
    ridge_model.fit(X_train, y_train)

    # 對測試集產生新的預測值
    y_pred = ridge_model.predict(X_test)

    # 更新散點圖資料：x=實際值，y=預測值
    # np.c_ 將兩個一維陣列合併為 (n, 2) 的二維陣列
    scatter.set_offsets(np.c_[y_test, y_pred])

    # 更新參考線（理論上範圍不變，但防止座標軸縮放後失準）
    line.set_xdata([y_test.min(), y_test.max()])
    line.set_ydata([y_test.min(), y_test.max()])

    # 計算均方誤差 (MSE)：MSE = (1/n) × Σ(y_actual - y_pred)²
    # MSE 越小代表預測越準確
    mse = mean_squared_error(y_test, y_pred)

    # 將當前 Alpha 與 MSE 更新至圖形標題
    ax.set_title(f'Ridge Regression: Alpha={alpha:.1f}, MSE={mse:.2f}')

    # 強制重新繪製圖形，確保標題與散點圖同步更新
    ax.figure.canvas.draw()

# ──────────────────────────────────────────────
# 7. 連接滑桿事件並顯示視窗
# ──────────────────────────────────────────────
# 當滑桿數值改變時呼叫 update 函式
alpha_slider.on_changed(update)

# 顯示互動式視窗
plt.show()