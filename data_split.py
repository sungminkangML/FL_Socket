# Dividing the entire MNIST dataset for clients in i.i.d, balanced manner

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = './FL_implementation/Distributed_FL/'
df_train = pd.read_csv(PATH+'data/mnist/mnist_train.csv')
    
train_pixels = df_train.iloc[:, 1:]
train_labels = df_train.iloc[:, 0]

#plt.imshow(np.array(df_train.iloc[1, 1:]).reshape(28, 28), cmap='gray')
#plt.show()

df_list = list()
n_per_class = list()

### For Distributed_FL : Dividing IID datset - same number of data per label, per client
for i in range(10):
    n_per_class.append(df_train[df_train['label']==i].shape[0])
    df_list.append(df_train[df_train['label']==i])

n_clnts = 10    # number of students
n_per_clnt = np.divide(np.array(n_per_class), n_clnts).astype(int)  # number of data per class and per clients

def div_iid_dataset(clnt_id, df_list):
    df = pd.DataFrame()
    for t in range(10):
        df = pd.concat([df, df_list[t].sample(n=n_per_clnt[t])])
    df.to_csv(PATH+f'data/mnist/div_5/clnt{clnt_id}_train.csv')
    
#for clnt in range(1, n_clnts+1):
#    div_iid_dataset(clnt, df_list)



### For P2P_FL : Dividing datset - odd & even
def div_oddeven_dataset(df, path):
    df_even = pd.DataFrame()
    df_odd = pd.DataFrame()
    for i in range(10):
        if i % 2 == 0:
            df_even = pd.concat([df_even, df_train[df_train['label']==i]])
        else:
            df_odd = pd.concat([df_odd, df_train[df_train['label']==i]])  
    df_even.to_csv(PATH + 'P2P_FL/dataset/mnist/mnist_train_even.csv')
    df_odd.to_csv(PATH + 'P2P_FL/dataset/mnist/mnist_train_odds.csv')

#div_oddeven_dataset(df_train, PATH)
