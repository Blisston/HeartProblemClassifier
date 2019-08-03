import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('heart.csv')
x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,13].values



from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size =1/3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test) 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train);
y_pred  = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))



from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier();
classifier.fit(x_train,y_train)
y_pred  = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))



from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(x_train,y_train)
y_pred  = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)
y_pred  = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))



from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy")
classifier.fit(x_train,y_train)
y_pred  = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))





from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 80,criterion="entropy")
classifier.fit(x_train,y_train)
y_pred  = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))