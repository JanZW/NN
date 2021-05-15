import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from time import time
from itertools import product

print('loading data...')
data=np.loadtxt('p53_old_2010/K8.data',\
                        usecols=(k for k in range(5407)),\
                        delimiter=',',\
                        converters = {k: lambda s:float(s.replace(b'?',b'').strip() or np.nan) for k in range(5407)})
labels=np.loadtxt('p53_old_2010/K8.data',usecols=5408,delimiter=',',dtype=str)

#Drop rows with nan
idx=~np.isnan(data).any(axis=1)
data=data[idx]
labels=labels[idx]
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.3)

ar=[3,5,7,10]
for s in range(1,4):
    for arch in product(ar,repeat=s):
        print(arch)
        clf=MLPClassifier(hidden_layer_sizes=(arch),
                            random_state=0
                            )
        start_time=time()
        clf.fit(X_train,y_train)
        print("time:",time()-start_time,"seconds")
        print("score:",clf.score(X_test,y_test))
#(100,50,20,10) yields 0.9943303380402225
#(100,50,10) yields 0.9954000855798032
#(75,50,10) yields 0.9958279845956355
#(20,10,5) yields 0.9961489088575096
#(5,3) yields 0.9958279845956355
#(5) yields 0.9907593411008437 in 31.48779845237732 seconds
