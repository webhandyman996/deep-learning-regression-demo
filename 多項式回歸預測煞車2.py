import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.widgets import Slider

# 使用拉桿觀察增加項次時的變化
# 模擬數據：汽車速度 (km/h) 與煞車距離 (m)
np.random.seed(42)
speed = np.random.uniform(20, 120, 50).reshape(-1, 1)  # 汽車速度 (20 到 120 km/h)
braking_distance = 0.05 * speed**2 + 2 * speed + 5 + np.random.randn(50, 1) * 10  # 煞車距離，帶噪聲

# 初始化圖形
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
scatter = ax.scatter(speed, braking_distance, color='blue', label='Observed Data')
line, = ax.plot([], [], color='red', label='Polynomial Regression Fit')
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Braking Distance (m)')
ax.legend()
ax.set_title('Polynomial Regression: Adjust Degree')

# 添加滑桿
ax_degree = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
degree_slider = Slider(ax_degree, 'Degree', 1, 10, valinit=2, valstep=1)

# 添加多項式方程的顯示
equation_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')

# 更新函數
def update(val):
    degree = int(degree_slider.val)  # 獲取滑桿的當前值
    poly = PolynomialFeatures(degree=degree)  # 更新多項式次數
    speed_poly = poly.fit_transform(speed)
    
    # 訓練多項式回歸模型
    model = LinearRegression()
    model.fit(speed_poly, braking_distance)
    
    # 預測煞車距離
    speed_fit = np.linspace(20, 120, 100).reshape(-1, 1)
    speed_fit_poly = poly.transform(speed_fit)
    braking_distance_fit = model.predict(speed_fit_poly)
    
    # 更新回歸曲線
    line.set_xdata(speed_fit)
    line.set_ydata(braking_distance_fit)
    ax.relim()
    ax.autoscale_view()
    
    # 獲取多項式係數並生成方程
    coeffs = model.coef_.flatten()
    intercept = model.intercept_.flatten()[0]
    equation = f"y = {intercept:.2f}"
    for i, coef in enumerate(coeffs[1:], start=1):
        equation += f" + {coef:.2f}x^{i}"
    equation_text.set_text(equation)  # 更新方程顯示
    
    fig.canvas.draw_idle()

# 連接滑桿更新事件
degree_slider.on_changed(update)

# 初始化回歸曲線
update(2)

plt.show()