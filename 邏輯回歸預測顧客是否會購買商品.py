import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 產生假資料(模擬數據)：顧客瀏覽時間 (minutes) 與是否購買商品 (0: 不購買, 1: 購買)
np.random.seed(42)
# 瀏覽時間 (1 到 20 分鐘)
browse_time = np.random.uniform(1, 20, 100).reshape(-1, 1)  
# 是否購買 (0 或 1)
purchase = (browse_time + np.random.randn(100, 1) * 2 > 10).astype(int).ravel()  

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(browse_time, purchase, test_size=0.2, random_state=42)

# 訓練邏輯回歸模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 評估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# 預測購買概率
browse_time_fit = np.linspace(1, 20, 100).reshape(-1, 1)  # 測試數據 (瀏覽時間範圍)
purchase_prob = model.predict_proba(browse_time_fit)[:, 1]  # 預測購買的概率

# 可視化結果
plt.scatter(browse_time, purchase, color='blue', label='Observed Data')
plt.plot(browse_time_fit, purchase_prob, color='red', label='Logistic Regression Fit')
plt.xlabel('Browse Time (minutes)')
plt.ylabel('Purchase Probability')
plt.legend()
plt.title('Logistic Regression: Browse Time vs Purchase Probability')
plt.show()