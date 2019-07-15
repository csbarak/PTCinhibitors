from main import getDataMatrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import pandas as pd
from sklearn import ensemble


#data= getDataMatrix(r"C:\Users\vilenchi\Dropbox\InProgress\Akabayov 2018\aligned.mol2",r"C:\Users\vilenchi\Dropbox\InProgress\Akabayov 2018\summary_2.0.sort")
train_data = getDataMatrix(r"/Users/sdannyvi/Dropbox/InProgress/Akabayov2018/aligned.mol2",r"/Users/sdannyvi/Dropbox/InProgress/Akabayov2018/summary_2.0.sort")

train_data = train_data.dropna()
print(train_data.shape)

best_oob=1000
best_mae=1000
best_d=0
best_ntrees=0
best_maxf=0

#for d in range(2,8):
#    for n_trees in range(50,55):
#        for maxf in range(5,15):
#            regr = RandomForestRegressor(max_depth=d, random_state=0, n_estimators=n_trees,max_features=maxf,oob_score=True)
#            regr.fit(data.iloc[:,0:-1],data.iloc[:,-1])
#            if(regr.oob_score_ < best_oob):
#                best_oob=regr.oob_score_
#                best_d=d
#                best_ntrees=n_trees
#                best_maxf=maxf
#print('depth=',best_d,'n_tress=',best_ntrees,'max features =',best_maxf,'oob=',best_oob)

X_train=train_data.iloc[:,1:-1]
y_train=train_data.iloc[:,-1]

regr1 = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=100,max_features=6,oob_score=True,criterion='mse')
regr = AdaBoostRegressor(base_estimator=regr1, learning_rate=1, loss='linear',n_estimators=50, random_state=0)
regr.fit(X_train,y_train)

y_pred=regr.predict(X_train)
err=y_train-y_pred
print('adaboost train error=',np.mean(np.abs(err)))


###### TEST 1 #######

#data_test= getDataMatrix(r"C:\Users\vilenchi\Dropbox\InProgress\Akabayov 2018\test10a.mol2",r"C:\Users\vilenchi\Dropbox\InProgress\Akabayov 2018\summary_2.0.sort")
data_test= getDataMatrix(r"/Users/sdannyvi/Dropbox/InProgress/Akabayov2018/test10a.mol2",r"/Users/sdannyvi/Dropbox/InProgress/Akabayov2018/summary_2.0.sort")

#data_test = data.dropna()
X_test=data_test.iloc[:,1:-1]
y_test=data_test.iloc[:,-1]


y_pred=regr.predict(X_test)
err=y_test-y_pred
#print('ada test error=',np.mean(np.abs(err)))

data_test['pred']=y_pred
#print(data_test[['NAME','BOND','pred']])

###### TEST2 #######

data_test= getDataMatrix(r"/Users/sdannyvi/Dropbox/InProgress/Akabayov2018/test4.mol2",r"/Users/sdannyvi/Dropbox/InProgress/Akabayov2018/summary_2.0.sort")

print(data_test)

#data = data.dropna()
X_test=data_test.iloc[:,1:-1]
y_test=data_test.iloc[:,-1]


y_pred=regr.predict(X_test)
err=y_test-y_pred
print('ada test error=',np.mean(np.abs(err)))

data_test['pred']=y_pred
print(data_test[['NAME','BOND','pred']])

