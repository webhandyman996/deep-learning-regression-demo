"""
邏輯回歸 (Logistic Regression) 預測顧客是否會購買商品 - 互動式決策門檻版
==========================================================
本程式使用邏輯回歸以顧客的瀏覽時間（分鐘）預測其購買意願（0/1），
並提供滑桿讓使用者即時調整決策門檻 (Threshold)，
觀察門檻值對分類結果的影響。

邏輯回歸原理：
  邏輯回歸是一種二元分類模型，將線性組合輸出透過
  Sigmoid 函數對應至 [0, 1] 的機率值：

      P(y=1 | x) = σ(w·x + b) = 1 / (1 + e^(-(w·x + b)))

  其中：
    w   ：特徵係數（瀏覽時間的權重）
    b   ：截距
    σ() ：Sigmoid 函數，輸出介於 0～1 之間的機率

決策門檻 (Threshold) 的作用：
  ┌──────────────────────────────────────────────────┐
  │  P(y=1|x) ≥ Threshold  →  預測為「購買」(1)     │
  │  P(y=1|x) <  Threshold  →  預測為「不購買」(0)  │
  └──────────────────────────────────────────────────┘
  預設門檻為 0.5，調高門檻可降低誤報（提高精確率 Precision），
  調低門檻可減少漏報（提高召回率 Recall）。

互動功能：
  拖曳底部「Threshold」滑桿（範圍 0.1～0.9）可即時
  更新散點圖上的分類結果與綠色門檻線的位置。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib.widgets import Slider

# ──────────────────────────────────────────────
# 1. 產生模擬資料
# ──────────────────────────────────────────────
# 固定隨機種子，確保每次執行結果一致
np.random.seed(42)

# 顧客瀏覽時間：從 1 到 20 分鐘均勻隨機抽取 100 個樣本
# reshape(-1, 1) 將一維陣列轉為 sklearn 所需的二維欄向量 (100, 1)
browse_time = np.random.uniform(1, 20, 100).reshape(-1, 1)

# 購買標籤：0 = 不購買，1 = 購買
# 規則：瀏覽時間 + 高斯噪聲（標準差 2）> 10 分鐘 → 傾向購買
# 加入噪聲模擬真實情境中的不確定性（有人瀏覽久卻不買，或短暫瀏覽就購買）
# astype(int) 將布林值 True/False 轉為整數 1/0
# ravel() 將二維陣列攤平為一維（sklearn 分類器要求 y 為一維）
purchase = (browse_time + np.random.randn(100, 1) * 2 > 10).astype(int).ravel()

# ──────────────────────────────────────────────
# 2. 分割訓練集與測試集
# ──────────────────────────────────────────────
# test_size=0.2：80% 用於訓練，20% 用於測試
# random_state=42：固定分割方式，確保結果可重現
X_train, X_test, y_train, y_test = train_test_split(
    browse_time, purchase, test_size=0.2, random_state=42
)

# ──────────────────────────────────────────────
# 3. 訓練邏輯回歸模型
# ──────────────────────────────────────────────
# LogisticRegression()：使用預設超參數（L2 正則化，C=1.0，solver='lbfgs'）
# 訓練後模型儲存 Sigmoid 函數的係數 w 與截距 b
model = LogisticRegression()
model.fit(X_train, y_train)  # 以訓練集擬合模型（最大化對數似然函數）

# ──────────────────────────────────────────────
# 4. 產生 Sigmoid 曲線資料
# ──────────────────────────────────────────────
# 在 1～20 分鐘範圍內均勻取 100 個點，作為繪製平滑曲線的輸入
browse_time_fit = np.linspace(1, 20, 100).reshape(-1, 1)

# predict_proba 回傳 (n, 2) 矩陣：[:, 0] 為不購買機率，[:, 1] 為購買機率
# 取 [:, 1] 得到各瀏覽時間對應的「購買機率」P(y=1|x)
purchase_prob = model.predict_proba(browse_time_fit)[:, 1]

# ──────────────────────────────────────────────
# 5. 建立互動式圖形
# ──────────────────────────────────────────────
fig, ax = plt.subplots()
# 調整子圖位置，為底部滑桿區域預留空間（bottom=0.25）
plt.subplots_adjust(left=0.25, bottom=0.25)

# 散點圖：顯示原始觀測資料（x=瀏覽時間，y=實際購買標籤 0/1）
scatter = ax.scatter(browse_time, purchase, color='blue', label='Observed Data')

# Sigmoid 曲線：顯示模型預測的購買機率（紅色實線）
line, = ax.plot(browse_time_fit, purchase_prob, color='red', label='Logistic Regression Fit')

# 決策門檻線：初始為 y=0.5 的水平線（綠色虛線）
# 機率高於此線的樣本被分類為「購買」
threshold_line, = ax.plot(
    browse_time_fit,
    [0.5] * len(browse_time_fit),
    color='green', linestyle='--', label='Threshold'
)

ax.set_xlabel('Browse Time (minutes)')  # X 軸：瀏覽時間（分鐘）
ax.set_ylabel('Purchase Probability')   # Y 軸：購買機率（0～1）
ax.legend()
ax.set_title('Logistic Regression: Adjust Threshold')

# ──────────────────────────────────────────────
# 6. 建立互動式滑桿
# ──────────────────────────────────────────────
# plt.axes([left, bottom, width, height]) 定義滑桿在圖形中的位置
ax_threshold = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# Threshold 滑桿：步進 0.01，範圍 0.1～0.9，初始值 0.5
# 刻意避開 0 與 1 的極端值，確保分類結果具有實際意義
threshold_slider = Slider(ax_threshold, 'Threshold', 0.1, 0.9, valinit=0.5, valstep=0.01)

# ──────────────────────────────────────────────
# 7. 定義滑桿更新回呼函式
# ──────────────────────────────────────────────
def update(val):
    """
    當 Threshold 滑桿數值改變時被呼叫，
    根據新門檻重新對原始資料進行分類，並更新圖形。

    參數
    ----
    val : float
        滑桿的最新數值（即決策門檻，範圍 0.1～0.9）。

    說明
    ----
    此函式不重新訓練模型，僅改變判斷「購買/不購買」的
    機率切割點，展示門檻對分類結果的影響。
    - 提高門檻 → 更嚴格，預測「購買」的顧客變少（減少誤報）
    - 降低門檻 → 更寬鬆，預測「購買」的顧客變多（減少漏報）
    """
    # 讀取滑桿當前的門檻值
    threshold = threshold_slider.val

    # 對原始瀏覽時間資料套用門檻進行分類
    # purchase_prob 為 browse_time_fit（均勻 100 點）的機率，
    # 此處需對應 browse_time（原始觀測點）重新計算機率再分類
    # 注意：重新以 model.predict_proba 對原始資料取機率，確保點位正確對應
    original_prob = model.predict_proba(browse_time)[:, 1]  # 原始樣本的購買機率
    predicted_purchase = (original_prob >= threshold).astype(int)  # 依門檻分類

    # 更新散點圖：y 值從原始標籤改為依門檻重新分類的預測標籤
    # np.c_ 將兩個一維陣列合併為 (n, 2) 的二維陣列供 set_offsets 使用
    scatter.set_offsets(np.c_[browse_time, predicted_purchase])

    # 更新門檻線的 y 值（水平線移動至新門檻高度）
    threshold_line.set_ydata([threshold] * len(browse_time_fit))

    # 要求 matplotlib 重新繪製圖形（非阻塞，避免凍結 GUI）
    fig.canvas.draw_idle()

# ──────────────────────────────────────────────
# 8. 連接滑桿事件並顯示視窗
# ──────────────────────────────────────────────
# 當滑桿數值改變時呼叫 update 函式
threshold_slider.on_changed(update)

# 顯示互動式視窗
plt.show()