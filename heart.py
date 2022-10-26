import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model,preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import pickle


#import Data
data = pd.read_csv("heart.csv")

d={'Presence':1,'Absence':0}

dependent=data['Heart Disease'].map(d)

data = data[['Age','Sex','Chest pain type','BP','Cholesterol','FBS over 120','EKG results','Max HR']]

x=np.array(data)
y=np.array(dependent)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.09)


best=0
#testing and training using actual data to know correlation btn them
#test_size is the percentage used for testing the remaining goes to training
for _ in range(1000):
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)

    #use the linearRegression module
    linear=KNeighborsClassifier(n_neighbors=7)

    #minimizing error(gradient decent)
    linear.fit(x_train,y_train)

    #test the accuracy of the error
    acc=linear.score(x_test,y_test)

    #printing out some values for verification
    #test model accuracy level


    if acc>best:
        best=acc
        with open('heartprediction.pickle','wb') as f:
           pickle.dump(linear,f)

savedmodel=open('heartprediction.pickle', 'rb')
newmodel=pickle.load(savedmodel)


myvalues=[[149,14345,1]]

predicted=newmodel.predict(x_test)
names=['absense','presence']


#loop through prediction to see if your data is corresponding well
for x in range(len(predicted)):
    print('predicted:',names[predicted[x]],'actual:',x_test[x],'final',names[y_test[x]])

