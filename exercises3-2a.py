import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from time import time

data=pd.read_csv('data_sheet_III/data_function_approximation.txt', sep=' ')
data.columns=['x','y']
X=data['x'].values.reshape(-1,1)
y=data['y'].values


mlp=MLPRegressor(solver='lbfgs',activation='logistic',random_state=0)


architecture=[(2,3),(7,5),(7,5,3),(5),(3),(15)]
fig,ax=plt.subplots(3,2)
fig.tight_layout()
ax=ax.ravel()
y_pred=[]
X_test=np.linspace(0,20,num=100).reshape(-1,1)
for arch,axis in zip(architecture,ax):
    print(arch)
    start_time=time()
    mlp.set_params(hidden_layer_sizes=arch)
    mlp.fit(X,y)
    fit_time=np.round_(time()-start_time,decimals=4)
    y_pred.append(mlp.predict(X_test))
    axis.plot(X,y)
    axis.plot(X_test,y_pred[-1])
    axis.set_title(str(arch)+' ('+str(fit_time)+' sec.)')
    axis.label_outer()
    print('loss',mlp.loss_*10**5)
plt.show()
