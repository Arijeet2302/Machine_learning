from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
study_hrs = [[1], [2], [3], [8], [9], [10]]
result = ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass"]
knn.fit(study_hrs, result)

# Predict on a new data point
data = [[6.3]]
res = knn.predict(data)
probability = knn.predict_proba(data)
print(res)
print(probability)