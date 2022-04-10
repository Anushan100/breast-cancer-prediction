import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
bc=load_breast_cancer()
df = pd.DataFrame(bc.data, columns = bc.feature_names)
df['diagnosis']=bc.target
x=df.drop(columns=["diagnosis"])
y=df["diagnosis"]
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.5,random_state=3)
logreg = LogisticRegression(C=0.1, penalty='l2', solver='liblinear')
model=logreg.fit(x_train,y_train)
import pickle
pickle.dump(logreg,open('finmod.pkl','wb'))
loadmodel=pickle.load(open('finmod.pkl','rb'))