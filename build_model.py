import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import datasets




names = ['SL', 'SW', 'PL', 'PW']



dataset = datasets.load_iris()

#print(data)

# data.to_csv('iris.csv', encoding='utf-8', index=False)

# data = pd.read_excel('iris.xls')
# data['SL']= data['SL'].fillna(data['SL']).median()
# data['SW']= data['SW'].fillna(data['SW']).median()
# X = data.drop('Classification',axis =1)
# y= data['Classification']
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(dataset.data,dataset.target,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
decision_tree_classifier=dt_clf.fit(X_train,y_train)
pickle.dump(decision_tree_classifier,open('model.pkl','wb'))