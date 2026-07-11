import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib.widgets import Slider

# 產生假資料(模擬數據)：顧客瀏覽時間 (minutes) 與是否購買商品 (0: 不購買, 1: 購買)
np.random.seed(42)
browse_time = np.random.uniform(1, 20, 100).reshape(-1, 1)  # 瀏覽時間 (1 到 20 分鐘)
purchase = (browse_time + np.random.randn(100, 1) * 2 > 10).astype(int).ravel()  # 是否購買 (0 或 1)

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(browse_time, purchase, test_size=0.2, random_state=42)

# 訓練邏輯回歸模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測購買概率
browse_time_fit = np.linspace(1, 20, 100).reshape(-1, 1)  # 測試數據 (瀏覽時間範圍)
purchase_prob = model.predict_proba(browse_time_fit)[:, 1]  # 預測購買的概率

# 初始化圖形
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
scatter = ax.scatter(browse_time, purchase, color='blue', label='Observed Data')
line, = ax.plot(browse_time_fit, purchase_prob, color='red', label='Logistic Regression Fit')
threshold_line, = ax.plot(browse_time_fit, [0.5] * len(browse_time_fit), color='green', linestyle='--', label='Threshold')
ax.set_xlabel('Browse Time (minutes)')
ax.set_ylabel('Purchase Probability')
ax.legend()
ax.set_title('Logistic Regression: Adjust Threshold')

# 添加滑桿
ax_threshold = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
threshold_slider = Slider(ax_threshold, 'Threshold', 0.1, 0.9, valinit=0.5, valstep=0.01)

# 更新函數
def update(val):
    threshold = threshold_slider.val  # 獲取滑桿的當前值
    predicted_purchase = (purchase_prob >= threshold).astype(int)  # 根據閾值進行分類
    scatter.set_offsets(np.c_[browse_time, predicted_purchase])  # 更新散點圖
    threshold_line.set_ydata([threshold] * len(browse_time_fit))  # 更新閾值線
    fig.canvas.draw_idle()

# 連接滑桿更新事件
threshold_slider.on_changed(update)

plt.show()