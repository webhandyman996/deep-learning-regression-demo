import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.widgets import Slider

# 模擬數據：多種廣告支出 (千元) 與商品銷售額 (千元)
np.random.seed(42)
n_samples = 100
ad_spend = np.random.uniform(1, 100, (n_samples, 3))  # 每列代表不同廣告渠道的支出
sales = 0.5 * ad_spend[:, 0] + 0.3 * ad_spend[:, 1] + 0.2 * ad_spend[:, 2] + np.random.randn(n_samples) * 10

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(ad_spend, sales, test_size=0.2, random_state=42)

# 初始化嶺回歸模型
initial_alpha = 3.0
ridge_model = Ridge(alpha=initial_alpha)
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)

# 初始化圖形
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
scatter = ax.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
line, = ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
ax.set_xlabel('Actual Sales (thousands)')
ax.set_ylabel('Predicted Sales (thousands)')
ax.legend()
ax.set_title(f'Ridge Regression: Alpha={initial_alpha}, MSE={mean_squared_error(y_test, y_pred):.2f}')

# 添加滑桿
ax_alpha = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
alpha_slider = Slider(ax_alpha, 'Alpha', 0.1, 10.0, valinit=initial_alpha, valstep=0.1)

# 更新函數
def update(val):
    alpha = alpha_slider.val  # 獲取滑塊的當前值
    ridge_model.set_params(alpha=alpha)  # 更新模型的 alpha 值
    ridge_model.fit(X_train, y_train)  # 重新訓練模型
    y_pred = ridge_model.predict(X_test)  # 生成新的預測值
    
    # 更新散點圖
    scatter.set_offsets(np.c_[y_test, y_pred])  # 更新散點圖數據
    
    # 更新紅色虛線
    line.set_xdata([y_test.min(), y_test.max()])
    line.set_ydata([y_test.min(), y_test.max()])
    
    # 計算新的 MSE 並更新標題
    mse = mean_squared_error(y_test, y_pred)  # 重新計算 MSE
    ax.set_title(f'Ridge Regression: Alpha={alpha:.1f}, MSE={mse:.2f}')  # 更新標題
    
    ax.figure.canvas.draw()  # 確保標題更新

# 連接滑塊更新事件
alpha_slider.on_changed(update)

plt.show()