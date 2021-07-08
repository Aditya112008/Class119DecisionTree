import csv 
import pandas as pd 
import plotly.express as px 
from io import StringIO

col_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','label']
df = pd.read_csv("./diabetes.csv",names = col_names).iloc[1:]

print(df.head())

features = ['Pregnancies','Glucose','BMI','Age','Insulin','BloodPressure','DiabetesPedigreeFunction']
X = df[features]
y = df.label

#split the data into training and testing and fit it in the model 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#splitting data in training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Initialising the Decision Tree Model
clf = DecisionTreeClassifier()

#Fitting the data into the model
clf = clf.fit(X_train,y_train)

#Calculating the accuracy of the model
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#model fitting is a measure of how well a machine learning generalizes to similar data to that on which it was trained

#a model that is well fitted produces more accurate outcomes
#a model that is over-fitted matches the data too closely 
#a model that is under-fitted does not match closely enough

#Now that we have have built a decision tree model, that can predict with an accuracy score of 0.68 if a person has diabetes or not based on their data, is there a way we can visualise it?

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image
import pydotplus

dot_data = StringIO() #Where we will store the data from our decision tree classifier as text.

export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['0','1'])

print(dot_data.getvalue())