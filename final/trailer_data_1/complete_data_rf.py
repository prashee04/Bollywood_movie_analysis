import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.getcwd()

os.chdir("c:\\Users\\harleyquinn\\Documents\\GitHub\\Bollywood_movie_analysis\\trailer-data")

df=pd.read_csv('complete_data.csv',on_bad_lines='skip')  

df.fillna("1", inplace = True)

df = df.dropna(axis=1)

df['gender'] = np.where((df['gender'] == 'man'), 1, 0)

X=df.drop('gender',axis=1)
y=df['gender']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor(random_state = 0)

import category_encoders as ce

encoder = ce.OrdinalEncoder(cols=['emotion','movie_name'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor(random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

cutoff = 0.7                              
y_pred_classes = np.zeros_like(Y_pred)    
y_pred_classes[Y_pred > cutoff] = 1

y_test_classes = np.zeros_like(Y_pred)
y_test_classes[Y_test > cutoff] = 1

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

from sklearn.metrics import accuracy_score
accuracy_score(y_test_classes, y_pred_classes)



def model_complete_data_rf(a,b,c):
    inputo=np.array([[a,b,c]])

    y_pred=classifier.predict(inputo)
    y_pred=y_pred[0]
    return y_pred

model_complete_data_rf(1,1,0)