import pandas as pd 
import numpy as np 
import matplotlib. pyplot as plt 
data=pd.read_csv("Iris.csv")
data=data.drop( 'Id', axis=1)
X = data.iloc[:, :-1].values
y = data.iloc[:, - 1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn. linear_model import LogisticRegression 
classifier = LogisticRegression()
classifier. fit(X_train, y_train) 
abc=classifier.predict([[1,5,2,1]])
print (abc)
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() 
classifier. fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred)) 
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score 
print('accuracy is', accuracy_score(y_pred,y_test)*100)
from sklearn. tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier() 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred)) 
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score 
print('accuracy is', accuracy_score(y_pred,y_test)*100)
from sklearn import metrics 
from sklearn. neighbors import KNeighborsClassifier
k_range = list(range(1,26))
scores = [ ] 
for k in k_range:

   classifier = KNeighborsClassifier(n_neighbors=k) 
   classifier. fit (X_train, y_train) 
   y_pred = classifier.predict(X_train) 
   scores.append(metrics.accuracy_score(y_train, y_pred))

plt.plot(k_range, scores) 
plt.xlabel( 'Value of k for KNN' ) 
plt.ylabel( 'Accuracy Score') 
plt. title( 'Accuracy Scores for Values of k of k-Nearest-Neighbors' ) 
plt. show()

classifier = KNeighborsClassifier(n_neighbors=1) 
classifier. fit(X_train, y_train) 
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred)) 
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score 
print('accuracy is', accuracy_score(y_pred, y_test)*100)
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier() 
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test) 
print(classification_report(y_test, y_pred) ) 
print(classification_report(y_test, y_pred)) 
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score 
print('accuracy is', accuracy_score(y_pred, y_test) *100)
