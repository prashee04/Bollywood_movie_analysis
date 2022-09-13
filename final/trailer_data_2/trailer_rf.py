import pandas as pd, numpy as np


import os
os.getcwd()

os.chdir("c:\\Users\\harleyquinn\\Documents\\GitHub\\Bollywood_movie_analysis\\trailer-data")

df=pd.read_csv('trailers_list.csv',on_bad_lines='skip')  

df.fillna("1", inplace = True)

df = df.dropna(axis=1)

X=df.drop('year-of-release',axis=1)
y=df['year-of-release']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25,random_state=42)

import category_encoders as ce

encoder = ce.OrdinalEncoder(cols=['movie_name'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)

trailers="Accuracy : ", accuracy_score(Y_test,Y_pred)

def model_trailer_rf(a,b,c):
    inputo=np.array([[a,b,c]])

    y_pred=classifier.predict(inputo)
    y_pred=y_pred[0]
    return y_pred

model_trailer_rf(1,1,0)