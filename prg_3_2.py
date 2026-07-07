#Practical–2:Write a python code to implement decision tree for below given dataset. Identify Job offered or not.

# Import necessary libraries
import pandas as pd
from sklearn import datasets
import sklearn.model_selection as ms
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 

#read data from xls file
iris=pd.read_csv("D:/4050/DataSets/prg_3_2.xls")

# define data and target variables
y=iris["species"]
x=iris[["sepal_length","sepal_width","petal_length","petal_width"]]

print(y)
print(x)

x_train, x_test, y_train, y_test = ms.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

dtc = tree.DecisionTreeClassifier(random_state=42)

dtc.fit(x_train, y_train)

#predict the class labels for x_test
y_predict=dtc.predict(x_test)

#print accuracy
print('Accuracy of Decision Tree-Test: ', accuracy_score(y_predict, y_test))

#print confusion matrix
print('\n','Confusion Matrix - Test:','\n',confusion_matrix(y_test,y_predict))

#print precision, recall and f1-score
print(classification_report(y_test,y_predict))