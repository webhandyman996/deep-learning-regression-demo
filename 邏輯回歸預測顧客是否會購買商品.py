import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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