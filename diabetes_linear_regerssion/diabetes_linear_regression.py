from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset (sklearn自帶的)
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# 分割data (train: 80%, test: 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立Linear Regression model
lr = LinearRegression()

# 使用train set訓練模型
lr.fit(X_train, y_train)

# 使用test set進行預測
y_pred_test = lr.predict(X_test)
y_pred_train = lr.predict(X_train)

# 計算MSE
mse_test = mean_squared_error(y_test, y_pred_test)
print('Mean Squared Error (test):', mse_test)

mse_train = mean_squared_error(y_train, y_pred_train)
print('Mean Squared Error (train):', mse_train)

