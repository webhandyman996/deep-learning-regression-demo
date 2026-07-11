import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.widgets import Slider

# 模擬數據：汽車速度 (km/h) 與煞車距離 (m)
np.random.seed(42)
speed = np.random.uniform(20, 120, 50).reshape(-1, 1)  # 汽車速度 (20 到 120 km/h)

# 初始化多項式係數
'''
𝑎
 主要決定高速時煞車距離的急劇增加，和動能消散有關。
𝑏
 則是與速度的線性關係，影響低速時的煞車距離。
𝑐
 代表最低限度的移動距離，即使車速為零，仍可能有小幅前進。
'''
a_init, b_init, c_init = 0.05, 2, 5  # 初始值

# 計算煞車距離
def calculate_braking_distance(a, b, c):
    return a * speed**2 + b * speed + c + np.random.randn(50, 1) * 10

braking_distance = calculate_braking_distance(a_init, b_init, c_init)

# 初始化圖形
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)
scatter = ax.scatter(speed, braking_distance, color='blue', label='Observed Data')
line, = ax.plot([], [], color='red', label='Polynomial Regression Fit')
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Braking Distance (m)')
ax.legend()
ax.set_title('Polynomial Regression: Adjust Coefficients')

# 添加滑桿
ax_a = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_b = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_c = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_a = Slider(ax_a, 'a (x^2)', 0.01, 0.1, valinit=a_init, valstep=0.01)
slider_b = Slider(ax_b, 'b (x)', 0, 5, valinit=b_init, valstep=0.1)
slider_c = Slider(ax_c, 'c (constant)', 0, 10, valinit=c_init, valstep=0.1)

# 更新函數
def update(val):
    a = slider_a.val
    b = slider_b.val
    c = slider_c.val
    
    # 更新煞車距離
    braking_distance = calculate_braking_distance(a, b, c)
    scatter.set_offsets(np.c_[speed, braking_distance])
    
    # 構建多項式特徵
    poly = PolynomialFeatures(degree=2)
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
    fig.canvas.draw_idle()

# 連接滑桿更新事件
slider_a.on_changed(update)
slider_b.on_changed(update)
slider_c.on_changed(update)

# 初始化回歸曲線
update(None)

plt.show()