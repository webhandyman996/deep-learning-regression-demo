"""
多項式回歸預測煞車距離 - 互動式次數調整版
==========================================================
本程式示範多項式回歸 (Polynomial Regression) 如何擬合
汽車速度 (km/h) 與煞車距離 (m) 之間的非線性關係。

物理背景：
  根據運動學，煞車距離與速度的關係近似於二次曲線：
      d = v² / (2·a) + v·t_reaction
  其中 a 為煞車加速度，t_reaction 為反應時間。
  本程式以帶噪聲的模擬資料驗證此規律。

互動功能：
  拖曳底部「Degree」滑桿可即時調整多項式次數 (1～10)，
  觀察次數過低 (欠擬合) 或過高 (過擬合) 對曲線的影響，
  並在圖形左上角顯示當前擬合的多項式方程式。

多項式回歸原理：
  1. 將原始特徵 x 透過 PolynomialFeatures 展開為
     [1, x, x², x³, ..., xⁿ]
  2. 再以普通線性回歸 (LinearRegression / 最小平方法) 求係數，
     得到 y = w₀ + w₁x + w₂x² + ... + wₙxⁿ
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.widgets import Slider

# ──────────────────────────────────────────────
# 1. 產生模擬資料
# ──────────────────────────────────────────────
# 固定隨機種子，確保每次執行結果一致
np.random.seed(42)

# 汽車速度：從 20 到 120 km/h 均勻隨機抽取 50 個樣本
# reshape(-1, 1) 將一維陣列轉為 sklearn 所需的二維欄向量 (50, 1)
speed = np.random.uniform(20, 120, 50).reshape(-1, 1)

# 煞車距離 (m)：依物理公式加入高斯噪聲（標準差 10 m）模擬量測誤差
# 真實關係：d = 0.05·v² + 2·v + 5（二次函數）
braking_distance = (
    0.05 * speed**2   # 速度平方項（動能主導，最主要的影響）
    + 2 * speed        # 速度一次項（反應距離）
    + 5                # 常數項（固定延遲距離）
    + np.random.randn(50, 1) * 10  # 量測噪聲
)

# ──────────────────────────────────────────────
# 2. 建立互動式圖形
# ──────────────────────────────────────────────
fig, ax = plt.subplots()
# 調整子圖位置，為底部滑桿區域預留空間（bottom=0.25）
plt.subplots_adjust(left=0.25, bottom=0.25)

# 散點圖：顯示模擬的原始觀測資料
scatter = ax.scatter(speed, braking_distance, color='blue', label='Observed Data')

# 回歸曲線：初始為空，待 update() 呼叫後填入資料
line, = ax.plot([], [], color='red', label='Polynomial Regression Fit')

ax.set_xlabel('Speed (km/h)')        # X 軸：汽車速度
ax.set_ylabel('Braking Distance (m)') # Y 軸：煞車距離
ax.legend()
ax.set_title('Polynomial Regression: Adjust Degree')

# ──────────────────────────────────────────────
# 3. 建立互動式滑桿
# ──────────────────────────────────────────────
# plt.axes([left, bottom, width, height]) 定義滑桿在圖形中的位置
ax_degree = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# Degree 滑桿：整數步進，範圍 1～10，初始值 2（對應物理二次關係）
degree_slider = Slider(ax_degree, 'Degree', 1, 10, valinit=2, valstep=1)

# 在圖形左上角預留文字區域，用於顯示當前擬合的多項式方程式
# transform=ax.transAxes 表示座標以軸的比例 (0～1) 表示，而非資料座標
equation_text = ax.text(
    0.05, 0.95, '',
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top'
)

# ──────────────────────────────────────────────
# 4. 定義滑桿更新回呼函式
# ──────────────────────────────────────────────
def update(val):
    """
    當 Degree 滑桿數值改變時被呼叫，重新訓練多項式回歸模型並更新圖形。

    參數
    ----
    val : float 或 int
        滑桿的最新數值（即多項式次數）。
        傳入固定值（如 2）可作為手動初始化呼叫。
    """
    # 讀取滑桿當前值並轉為整數（多項式次數必須為正整數）
    degree = int(degree_slider.val)

    # ── 步驟 A：特徵展開 ──────────────────────
    # PolynomialFeatures(degree=d) 將 [x] 展開為 [1, x, x², ..., xᵈ]
    # fit_transform 同時計算展開參數並套用至訓練資料
    poly = PolynomialFeatures(degree=degree)
    speed_poly = poly.fit_transform(speed)  # 形狀：(50, degree+1)

    # ── 步驟 B：訓練線性回歸模型 ──────────────
    # 在展開後的特徵空間上套用最小平方法，求解各項係數
    model = LinearRegression()
    model.fit(speed_poly, braking_distance)

    # ── 步驟 C：產生平滑預測曲線 ──────────────
    # 在 20～120 km/h 之間均勻取 100 個點，作為繪圖用的輸入
    speed_fit = np.linspace(20, 120, 100).reshape(-1, 1)
    # 使用 transform（非 fit_transform）套用相同的展開參數
    speed_fit_poly = poly.transform(speed_fit)
    braking_distance_fit = model.predict(speed_fit_poly)  # 預測煞車距離

    # ── 步驟 D：更新回歸曲線 ──────────────────
    line.set_xdata(speed_fit)
    line.set_ydata(braking_distance_fit)
    ax.relim()           # 重新計算資料範圍（因曲線資料已更新）
    ax.autoscale_view()  # 自動調整坐標軸範圍以容納所有資料

    # ── 步驟 E：組合並顯示多項式方程式 ──────────
    # model.coef_ 形狀為 (1, degree+1)，第 0 項對應常數（與 intercept 重複）
    # model.intercept_ 為截距 w₀
    coeffs = model.coef_.flatten()       # 各項係數 [w₀, w₁, w₂, ...]
    intercept = model.intercept_.flatten()[0]  # 截距（獨立於 coef_ 之外）

    # 從截距開始，依序附加各次項（跳過 coeffs[0]，因其對應常數項已由 intercept 表示）
    equation = f"y = {intercept:.2f}"
    for i, coef in enumerate(coeffs[1:], start=1):
        # 正負號統一以 " + " 分隔（負係數時會顯示 " + -x.xx"，便於閱讀）
        equation += f" + {coef:.2f}x^{i}"

    equation_text.set_text(equation)  # 更新圖形上的方程式文字

    # 要求 matplotlib 重新繪製圖形（非阻塞，避免凍結 GUI）
    fig.canvas.draw_idle()

# ──────────────────────────────────────────────
# 5. 連接滑桿事件並執行初始繪圖
# ──────────────────────────────────────────────
# 當滑桿數值改變時呼叫 update 函式
degree_slider.on_changed(update)

# 以初始次數（2）執行一次 update，確保圖形在啟動時即顯示正確結果
update(2)

# 顯示互動式視窗
plt.show()