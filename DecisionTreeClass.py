import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('music.csv')

X = data.drop(columns=['genre'])
y = data['genre']
x_train , x_test, y_train, y_test = train_test_split(X , y, test_size=0.1)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
score = accuracy_score(y_test, predictions)
print(predictions)
print(score)