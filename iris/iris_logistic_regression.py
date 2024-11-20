from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model (會根據數據集來自動選擇是二分類還是多分類)
# max_iter: 最大迭代次數 (default=100)
lr = LogisticRegression(max_iter=90)
# lr = LogisticRegression(multi_class='ovr') # 多分類轉成多個二分類
# lr = LogisticRegression(multi_class='multinomial') # Softmax回歸做多分類

# Fit the model
lr.fit(X_train, y_train)

# Predict the test set
y_pred = lr.predict(X_test)

# Calculate the accuracy
print("Accuracy: ", accuracy_score(y_test, y_pred))
