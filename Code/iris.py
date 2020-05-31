from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
iris_dataset=load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
test=knn.predict(x_test)
print(knn.score(x_test,y_test))
print(confusion_matrix(y_test,test))
x_new=np.array([[5,2.9,1,0.2]])
prediction=knn.predict(x_new)
print(prediction)
print(iris_dataset['target_names'][prediction])