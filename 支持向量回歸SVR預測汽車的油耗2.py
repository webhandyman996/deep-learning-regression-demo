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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.widgets import Slider

# ──────────────────────────────────────────────
# 1. 產生模擬資料
# ──────────────────────────────────────────────
# 固定隨機種子，確保每次執行結果相同
np.random.seed(42)
n_samples = 100  # 樣本數量

# 特徵矩陣：每列為一筆樣本，三個欄位分別代表
#   features[:, 0] — 車重 (kg)，範圍 1000～3000
#   features[:, 1] — 引擎大小 (公升)，範圍 1000～3000（注意：此處為示意數值）
#   features[:, 2] — 氣缸數，範圍 1000～3000（注意：此處為示意數值）
features = np.random.uniform(1000, 3000, (n_samples, 3))

# 目標變數：油耗 (L/100km)
# 公式：油耗 = 0.005×車重 + 0.8×引擎大小 + 0.3×氣缸數 + 隨機噪聲
# 隨機噪聲模擬真實量測誤差，標準差為 2
fuel_consumption = (
    0.005 * features[:, 0]   # 車重對油耗的貢獻
    + 0.8 * features[:, 1]   # 引擎大小對油耗的貢獻
    + 0.3 * features[:, 2]   # 氣缸數對油耗的貢獻
    + np.random.randn(n_samples) * 2  # 高斯噪聲
)

# ──────────────────────────────────────────────
# 2. 分割訓練集與測試集
# ──────────────────────────────────────────────
# test_size=0.2：80% 用於訓練，20% 用於測試
# random_state=42：固定分割方式，確保可重現
X_train, X_test, y_train, y_test = train_test_split(
    features, fuel_consumption, test_size=0.2, random_state=42
)

# ──────────────────────────────────────────────
# 3. 建立互動式圖形
# ──────────────────────────────────────────────
fig, ax = plt.subplots()
# 調整子圖位置，為底部滑桿區域預留空間
# 參數 [left, bottom, right_margin, top_margin]
plt.subplots_adjust(left=0.25, bottom=0.4)

# 初始散點圖：x 軸為實際值，y 軸為預測值
# 理想情況下所有點應落在 y=x 的對角線上
scatter = ax.scatter(y_test, y_test, color='blue', label='Predicted vs Actual')

# 參考線：y=x 的理想擬合線（完美預測時所有點落在此線上）
line, = ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red', linestyle='--', label='Ideal Fit'
)

ax.set_xlabel('Actual Fuel Consumption (L/100km)')   # X 軸：實際油耗
ax.set_ylabel('Predicted Fuel Consumption (L/100km)') # Y 軸：預測油耗
ax.legend()
ax.set_title('Support Vector Regression: Adjust Parameters')

# ──────────────────────────────────────────────
# 4. 建立互動式滑桿
# ──────────────────────────────────────────────
# plt.axes([left, bottom, width, height]) 定義滑桿在圖形中的位置與大小
ax_c       = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_gamma   = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_epsilon = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# Slider(ax, label, valmin, valmax, valinit, valstep)
slider_c       = Slider(ax_c,       'C',       1,    200, valinit=100, valstep=1)    # C：1～200，初始值 100
slider_gamma   = Slider(ax_gamma,   'Gamma',   0.01, 1.0, valinit=0.1, valstep=0.01) # Gamma：0.01～1.0，初始值 0.1
slider_epsilon = Slider(ax_epsilon, 'Epsilon', 0.01, 1.0, valinit=0.1, valstep=0.01) # Epsilon：0.01～1.0，初始值 0.1

# ──────────────────────────────────────────────
# 5. 定義滑桿更新回呼函式
# ──────────────────────────────────────────────
def update(val):
    """
    當任一滑桿數值改變時被呼叫，重新訓練 SVR 模型並更新圖形。

    參數
    ----
    val : float 或 None
        滑桿的最新數值（此函式不直接使用 val，而是從各滑桿物件讀取）。
        傳入 None 可作為手動初始化呼叫。
    """
    # 從滑桿讀取當前超參數值
    C       = slider_c.val
    gamma   = slider_gamma.val
    epsilon = slider_epsilon.val

    # 建立並訓練 SVR 模型
    # kernel='rbf'：使用徑向基函數核，公式：K(x, x') = exp(-γ · ||x - x'||²)
    # C       ：懲罰參數，控制訓練誤差的容忍度
    # gamma   ：RBF 核函數的 γ 值
    # epsilon ：不敏感帶寬度 ε，帶內誤差不計入損失
    svr_model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    svr_model.fit(X_train, y_train)  # 以訓練集擬合模型

    # 對測試集進行預測
    y_pred = svr_model.predict(X_test)

    # 更新散點圖資料：x=實際值，y=預測值
    # np.c_ 將兩個一維陣列合併為 (n, 2) 的二維陣列
    scatter.set_offsets(np.c_[y_test, y_pred])

    # 計算均方誤差 (MSE)，評估模型預測準確度
    # MSE = (1/n) × Σ(y_actual - y_pred)²，值越小代表預測越準確
    mse = mean_squared_error(y_test, y_pred)

    # 將當前超參數與 MSE 顯示在圖形標題
    ax.set_title(f'SVR: C={C:.1f}, Gamma={gamma:.2f}, Epsilon={epsilon:.2f}, MSE={mse:.2f}')

    # 要求 matplotlib 重新繪製圖形（非阻塞）
    fig.canvas.draw_idle()

# ──────────────────────────────────────────────
# 6. 連接滑桿事件並執行初始繪圖
# ──────────────────────────────────────────────
# 當滑桿數值改變時呼叫 update 函式
slider_c.on_changed(update)
slider_gamma.on_changed(update)
slider_epsilon.on_changed(update)

# 以初始超參數值執行一次 update，確保圖形在啟動時即顯示正確結果
update(None)

# 顯示互動式視窗
plt.show()