import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Iris.csv')

X = df.drop(columns=['Species']) 
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 40)
print(f"Scikit-Learn KNN Model Train Complete!")
print(f"Testing Accuracy: {accuracy * 100:.2f}%")
print("=" * 40)
