import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.widgets import Slider

# 模擬數據：汽車特徵 (車重、引擎大小、氣缸數) 與油耗 (單位：L/100km)
np.random.seed(42)
n_samples = 100
# 汽車特徵：車重 (kg)、引擎大小 (L)、氣缸數
features = np.random.uniform(1000, 3000, (n_samples, 3))  # 每列代表不同特徵
# 油耗：基於特徵生成，並添加噪聲
fuel_consumption = 0.005 * features[:, 0] + 0.8 * features[:, 1] + 0.3 * features[:, 2] + np.random.randn(n_samples) * 2

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features, fuel_consumption, test_size=0.2, random_state=42)

# 初始化圖形
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)
scatter = ax.scatter(y_test, y_test, color='blue', label='Predicted vs Actual')
line, = ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
ax.set_xlabel('Actual Fuel Consumption (L/100km)')
ax.set_ylabel('Predicted Fuel Consumption (L/100km)')
ax.legend()
ax.set_title('Support Vector Regression: Adjust Parameters')

# 添加滑桿
ax_c = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_gamma = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_epsilon = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_c = Slider(ax_c, 'C', 1, 200, valinit=100, valstep=1)
slider_gamma = Slider(ax_gamma, 'Gamma', 0.01, 1.0, valinit=0.1, valstep=0.01)
slider_epsilon = Slider(ax_epsilon, 'Epsilon', 0.01, 1.0, valinit=0.1, valstep=0.01)

# 更新函數
def update(val):
    C = slider_c.val
    gamma = slider_gamma.val
    epsilon = slider_epsilon.val
    
    # 訓練支持向量回歸模型rbf核函數公式： K(x, x') = exp(-γ * ||x - x'||^2)
    svr_model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    svr_model.fit(X_train, y_train)
    
    # 預測油耗
    y_pred = svr_model.predict(X_test)
    
    # 更新散點圖
    scatter.set_offsets(np.c_[y_test, y_pred])
    
    # 計算新的 MSE 並更新標題
    mse = mean_squared_error(y_test, y_pred)
    ax.set_title(f'SVR: C={C:.1f}, Gamma={gamma:.2f}, Epsilon={epsilon:.2f}, MSE={mse:.2f}')
    fig.canvas.draw_idle()

# 連接滑桿更新事件
slider_c.on_changed(update)
slider_gamma.on_changed(update)
slider_epsilon.on_changed(update)

# 初始化圖形
update(None)

plt.show()