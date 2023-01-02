# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:42:23 2022

@author: benjamin
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import numpy as np
import itertools

os.chdir(r"D:\machine learning final project")

all_file_name = os.listdir()

def load_data(file_name):
    df = pd.read_csv(r".\{}\eeg.csv".format(file_name))

    # because:there is duplicate data
    df = df[::2]
    
    # element to plot grpah
    time_step = df["id"]
    sensor_1 = df["ch1"]
    sensor_2 = df["ch2"]
    sensor_3 = df["ch3"]
    sensor_4 = df["ch4"]
    sensor_5 = df["ch5"]

    return time_step,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5

# load data for feature concentration
path = r"D:\machine learning final project\final data"
all_process_name = os.listdir(path)

def load_data_process(path,file_name):
    df = pd.read_csv(path+r"\{}".format(file_name))

    # because:there is duplicate data
    
    # element to plot grpah
    time_step = df["id"]
    sensor_1 = df["ch1"]
    sensor_2 = df["ch2"]
    sensor_3 = df["ch3"]
    sensor_4 = df["ch4"]
    sensor_5 = df["ch5"]

    return time_step,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5

def auto_sublpot(time_step,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,name):
# Creating 2 subplots
    fig, ax = plt.subplots(5)
    
    # add title on top
    fig.suptitle(name)
    
    # Accessing each axes object to plot the data through returned array
    ax[0].plot(time_step,sensor_1,color="orange")
    ax[1].plot(time_step,sensor_2,color="green")
    ax[2].plot(time_step,sensor_3,color="blue")
    ax[3].plot(time_step,sensor_4,color="black")
    ax[4].plot(time_step,sensor_5,color="magenta")
    plt.show()

# add ten timestep to one (concertration)
# does not step 2 ,because data processed
def feature_concentration(data):
    previous = 0
    ten_to_one = []
    for current in range(10,len(data),10):
        add_up = sum(data[previous:current])
        ten_to_one.append(add_up)
        previous = current
    
    return list(ten_to_one)

# write to csv
def create_csv(name,time_step,fc_sensor_1,fc_sensor_2,fc_sensor_3,fc_sensor_4,fc_sensor_5):
    with open(r"D:\machine learning final project\final data\{}.csv".format(name),"w", newline='') as f:  
        cw = csv.writer(f)
        cw.writerow(["id","ch1","ch2","ch3","ch4","ch5"])
        for i in range(len(fc_sensor_1)):
            cw.writerow([i,fc_sensor_1[i],fc_sensor_2[i],fc_sensor_3[i],fc_sensor_4[i],fc_sensor_5[i]])

# chinese song and japan song
# use with all_process_name
# id is not needed
def data_window(path , name ,len_window):
    df = pd.read_csv(path+"/"+name,usecols=[1,2,3,4,5])

    # store process data
    content = []
    previous = 0
    
    # can ignore padding problem
    for current in range(len_window , len(df["ch1"]) , len_window):
    #for current in range(5,10,5):
        tem = df[previous:current]
        transposed = np.array(tem).T.tolist()
        flat_ls = list(itertools.chain(*transposed))
        content.append(flat_ls)
        previous = current
        
    return content

# for lstm
def data_window_2(path , name ):
    df = pd.read_csv(path+"/"+name,usecols=[1,2,3,4,5])
    df = df.T
    df = (df.values.tolist())

    output = []
    previous = 0
    for current in range(5,len(df[0]),5):
        first_column = df[0][previous:current]
        second_column = df[1][previous:current]
        thrid_column = df[2][previous:current]
        four_column = df[3][previous:current]
        five_column = df[4][previous:current]
    
        output.append([first_column,second_column,thrid_column,four_column,five_column])
        previous = current
    
    # change type
    output = np.array(output)
    # fix the reshape
    output = output.reshape(int(current/5),5,5)
    
    return output
# In[]
for name in all_file_name:
    time_step,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5 = load_data(name)
    #auto_sublpot(time_step,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,name)
    
    # feature enginerring
    fc_sensor_1 = feature_concentration(sensor_1)
    fc_sensor_2 = feature_concentration(sensor_2)
    fc_sensor_3 = feature_concentration(sensor_3)
    fc_sensor_4 = feature_concentration(sensor_4)
    fc_sensor_5 = feature_concentration(sensor_5)

    # write to csv
    create_csv(name,time_step,fc_sensor_1,fc_sensor_2,fc_sensor_3,fc_sensor_4,fc_sensor_5)


# In[]

data = []
label = []
for name in all_process_name:
    print(name)
    new_data = data_window(path , name ,10)
    if name.startswith("c") == True:
        new_label = [0 for i in range(len(new_data))]
    if name.startswith("j") == True:
        new_label = [1 for i in range(len(new_data))]
    
    data.extend(new_data)
    label.extend(new_label)
    
# In[]
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM ,Dropout

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# In[]
# data for lstm
pre_data = [] 
label_2 = []
for name in all_process_name:
    new_data = data_window_2(path , name)
    if name.startswith("c") == True:
        new_label = [0 for i in range(len(new_data))]
    if name.startswith("j") == True:
        new_label = [1 for i in range(len(new_data))]
    
    pre_data.extend(new_data)
    label_2.extend(new_label)

x_train,x_test,y_train,y_test = train_test_split( np.array(pre_data) ,np.array(label_2) ,test_size = 0.3)

# In[]
# fail not same dim

model = Sequential()
model.add(LSTM(64, input_shape=(5, 5)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))
model.compile(loss='mse', optimizer="adam")
model.fit(x_train, y_train, epochs=3 ,batch_size=1, validation_data=(x_test,y_test))
#model.summary()

test_preds = model.predict(x_test) 
print(accuracy_score(y_test, test_preds))

# In[]
#for machine learning
x_train,x_test,y_train,y_test = train_test_split( data ,label ,test_size = 0.3)
# In[]

decision_tree = DecisionTreeClassifier(random_state=0)

decision_tree.fit(x_train, y_train)

predictions = decision_tree.predict(x_test)
print(predictions[:5])


print(accuracy_score(y_test, predictions))
#x = cross_val_score(clf, x_train, y_train, cv=10)
# In[]

random_forest = RandomForestClassifier(max_depth=100, random_state=0)
random_forest.fit(x_train, y_train)

predictions = random_forest.predict(x_test)
print(predictions[:5])
print(accuracy_score(y_test, predictions))
# In[]

xgboost = XGBClassifier(eta=0.3 ,max_depth=16)
xgboost.fit(pd.DataFrame(x_train), pd.DataFrame(y_train))

predictions = xgboost.predict(pd.DataFrame(x_test))
print(predictions[:5])
print(accuracy_score(pd.DataFrame(y_test), pd.DataFrame(predictions)))
# In[]
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(pd.DataFrame(x_train), pd.DataFrame(y_train))

predictions = logistic.predict(pd.DataFrame(x_test))
print(predictions[:5])
print(accuracy_score(pd.DataFrame(y_test), pd.DataFrame(predictions)))
print("yes")
# In[]
from sklearn.neighbors import KNeighborsClassifier
knclassification = KNeighborsClassifier(n_neighbors=3)
knclassification.fit(pd.DataFrame(x_train), pd.DataFrame(y_train))

predictions = knclassification.predict(pd.DataFrame(x_test))
print(predictions[:5])
print(accuracy_score(pd.DataFrame(y_test), pd.DataFrame(predictions)))
# In[]
from sklearn import svm
svm_clf = svm.SVC()
svm_clf.fit(pd.DataFrame(x_train), pd.DataFrame(y_train))

predictions = svm_clf.predict(pd.DataFrame(x_test))
print(predictions[:5])
print(accuracy_score(pd.DataFrame(y_test), pd.DataFrame(predictions)))


# In[]
# Get the unique values of 'B' column
print(df.ch1.unique().tolist())

.describe().transpose()