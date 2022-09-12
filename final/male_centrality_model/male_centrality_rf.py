import pandas as pd, numpy as np


import os
os.getcwd()
os.chdir("c:\\Users\\harleyquinn\\Documents\\GitHub\\Bollywood_movie_analysis\\wikipedia-data")
df= pd.read_csv('male_mentions_centrality.csv')  
df=df.drop('MOVIE NAME',axis=1)
df=df.drop('CAST',axis=1)
df.fillna(value=1, inplace = True)

X=df.drop('TOTAL CENTRALITY',axis=1)
y=df['TOTAL CENTRALITY']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor(random_state = 0)

from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor(random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

cutoff = 0.7                              
y_pred_classes = np.zeros_like(Y_pred)    
y_pred_classes[Y_pred > cutoff] = 1    

y_test_classes = np.zeros_like(Y_pred)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

from sklearn.metrics import accuracy_score
accuracy_score(y_test_classes, y_pred_classes)


def model_male_centrality_rf(a,b,c):
    inputo=np.array([[a,b,c]])

    y_pred=classifier.predict(inputo)
    y_pred=y_pred[0]
    return y_pred

model_male_centrality_rf(1,7,3.714286)