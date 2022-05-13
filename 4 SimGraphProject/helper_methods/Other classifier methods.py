from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
import numpy as np
import json
import os


def process(path):
    files = os.listdir(path)
    data = []
    for file in files:
        f = open(path + "/" + file, "r")
        f_json = json.load(f)
        adj_matrix_1 = np.array(f_json["adj_matrix_1"]) 
        data.append(np.reshape(adj_matrix_1,-1))
        adj_matrix_2 = np.array(f_json["adj_matrix_2"]) 
        data.append(np.reshape(adj_matrix_2,-1))

    return np.array(data)


"""
# loading the iris dataset
iris = datasets.load_iris()

# X -> features, y -> label
X = iris.data
y = iris.target
print(X.shape, "\n", y.shape)
"""

folder = "A06"
path = "C:/Users/91876/Downloads/SimGNN-main/dataset/" + folder + "/Training/" + folder + "T_"
X_train = process(path + str(1))
for i in range(2,5):
    data = process(path + str(i))
    X_train = np.append(X_train, data, axis = 0)

path = "C:/Users/91876/Downloads/SimGNN-main/dataset/" + folder + "/Testing/" + folder + "E_"

X_test = process(path + str(1))
for i in range(2,5):
    data = process(path + str(i))
    X_test = np.append(X_test, data, axis = 0)
    
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)
    
test_len = 450
y_test = np.full((test_len), 1)
for i in range(2,5):
    y_test = np.append(y_test, np.full((test_len),i) , axis = 0)

y_train = y_test
print("y_test shape", y_test.shape)

"""
DECISION TREE CLASSIFIER:

# dividing X, y into train and test data
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 8).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test)

accuracy = dtree_model.score(X_test, y_test)
print(accuracy*100, "%")                        # 30.833333333333336 %

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)
print(cm)


KNN CLASSIFIER : 
folder = "A09"
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10, weights = 'distance').fit(X_train, y_train)
 
# accuracy on X_test
accuracy = knn.score(X_test, y_test)
print(accuracy*100, "%")    #31.666666666666664 %
 
# creating a confusion matrix
knn_predictions = knn.predict(X_test)
cm = confusion_matrix(y_test, knn_predictions)
print(cm)

SVM CLASSIFICATION :
folder = "A06"
# training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'poly', degree = 4, C = 0.20, probability = True).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
 
# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test)
print(accuracy*100,"%")             #32.5%
 
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
print(cm)
"""

=======
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
import numpy as np
import json
import os


def process(path):
    files = os.listdir(path)
    data = []
    for file in files:
        f = open(path + "/" + file, "r")
        f_json = json.load(f)
        adj_matrix_1 = np.array(f_json["adj_matrix_1"]) 
        data.append(np.reshape(adj_matrix_1,-1))
        adj_matrix_2 = np.array(f_json["adj_matrix_2"]) 
        data.append(np.reshape(adj_matrix_2,-1))

    return np.array(data)


"""
# loading the iris dataset
iris = datasets.load_iris()

# X -> features, y -> label
X = iris.data
y = iris.target
print(X.shape, "\n", y.shape)
"""

folder = "A06"
path = "C:/Users/91876/Downloads/SimGNN-main/dataset/" + folder + "/Training/" + folder + "T_"
X_train = process(path + str(1))
for i in range(2,5):
    data = process(path + str(i))
    X_train = np.append(X_train, data, axis = 0)

path = "C:/Users/91876/Downloads/SimGNN-main/dataset/" + folder + "/Testing/" + folder + "E_"

X_test = process(path + str(1))
for i in range(2,5):
    data = process(path + str(i))
    X_test = np.append(X_test, data, axis = 0)
    
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)
    
test_len = 450
y_test = np.full((test_len), 1)
for i in range(2,5):
    y_test = np.append(y_test, np.full((test_len),i) , axis = 0)

y_train = y_test
print("y_test shape", y_test.shape)

"""
DECISION TREE CLASSIFIER:

# dividing X, y into train and test data
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 8).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test)

accuracy = dtree_model.score(X_test, y_test)
print(accuracy*100, "%")                        # 30.833333333333336 %

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)
print(cm)


KNN CLASSIFIER : 
folder = "A09"
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10, weights = 'distance').fit(X_train, y_train)
 
# accuracy on X_test
accuracy = knn.score(X_test, y_test)
print(accuracy*100, "%")    #31.666666666666664 %
 
# creating a confusion matrix
knn_predictions = knn.predict(X_test)
cm = confusion_matrix(y_test, knn_predictions)
print(cm)

SVM CLASSIFICATION :
folder = "A06"
# training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'poly', degree = 4, C = 0.20, probability = True).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
 
# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test)
print(accuracy*100,"%")             #32.5%
 
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
print(cm)
"""

>>>>>>> 62eea7bafc86743dc67b77f6cb84a836bbfc4192
