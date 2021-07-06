# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 23:33:27 2020

@author: Abebe
"""

import os
import numpy as np

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib as plt
import matplotlib.pyplot as py
from sklearn.model_selection import KFold
import pickle
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from keras.utils import to_categorical
import random
from keras import optimizers

from natsort import natsorted

from Custom_batch_drug_CNN_v1 import DataGenerator
from load_data_APCA_600 import val_y, auc_avg

test_file = open('C:/Users/CML_Ye/OneDrive - 금오공과대학교/바탕 화면/data/ca_AP_list/' +'valid_drug_file.pkl', "rb")
dataset_1 = pickle.load(test_file)
test_IDs= natsorted(dataset_1['AP_data'])
test_ca_ID = natsorted(dataset_1['ca_val'])

train_file = open('C:/Users/CML_Ye/OneDrive - 금오공과대학교/바탕 화면/data/ca_AP_list/' +'train_drug_file.pkl', "rb")
dataset = pickle.load(train_file)
IDs= np.array(natsorted(dataset['AP_data']))
ca_IDs = np.array(natsorted(dataset['ca_val']))


tr_directory = r'C:/Users/CML_Ye/OneDrive - 금오공과대학교/바탕 화면/data/training/'

test_directory = r'C:/Users/CML_Ye/OneDrive - 금오공과대학교/바탕 화면/data/validation/'


tr_ID = os.listdir(tr_directory)
test_ID = os.listdir(test_directory)

# test_IDs = np.concatenate(test_IDs).tolist()

# IDs = np.concatenate(IDs).tolist()


def Drug_Model():
    
    Model = tf.keras.models.Sequential()
    # model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    Model.add(tf.keras.layers.Conv1D(filters=3, kernel_size=32, strides =4, activation='relu',input_shape=(1000,1)))
    Model.add(tf.keras.layers.BatchNormalization())

    Model.add(tf.keras.layers.MaxPooling1D(pool_size = 4, strides =2))
    # Model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=4, strides =4, activation='relu'))
    
    Model.add(tf.keras.layers.Conv1D(filters=3, kernel_size=16, strides =4, activation='relu'))
    Model.add(tf.keras.layers.BatchNormalization())
    Model.add(tf.keras.layers.MaxPooling1D(pool_size = 4, strides = 2))

    # Model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=4, strides =2, activation='relu'))
    # Model.add(tf.keras.layers.BatchNormalization())
    
    # Model.add(tf.keras.layers.MaxPooling1D(pool_size = 2, strides = 1))
    Model.add(tf.keras.layers.Dropout(0.25))
    Model.add(tf.keras.layers.Flatten())
    
    Model.add(tf.keras.layers.Dense(units =5, kernel_initializer ='uniform', activation = 'relu'))
    Model.add(tf.keras.layers.Dense(3,activation ='softmax'))
    Model.summary()
       
    return Model


def Model_Train(tr_directory,ca_IDs, IDs, params, batch_size, n_split):
           
    cv = KFold(n_split, random_state =100, shuffle = True)
    for i, (train_index, valid_index) in enumerate(cv.split(IDs)):
        Train_list = IDs[train_index]
        Valid_list = IDs[valid_index]
        
        Train_ca_list = ca_IDs[train_index]
        Valid_ca_list = ca_IDs[valid_index]
        Train_list, Valid_list = np.ndarray.tolist(Train_list), np.ndarray.tolist(Valid_list)
        Train_ca_list, Valid_ca_list = np.ndarray.tolist(Train_ca_list), np.ndarray.tolist(Valid_ca_list)
        
        
        Tr_data = DataGenerator(tr_directory, Train_list,Train_ca_list ,encoder, **params)
        Val_data = DataGenerator(tr_directory, Valid_list,Valid_ca_list, encoder, **params)

        Model_path = 'C:/Users/CML_Ye/OneDrive - 금오공과대학교/문서/CNN/test11/'
    
        if not os.path.exists(Model_path):
            os.mkdir(Model_path)
        
        save_path = Model_path + str(i) + '-{epoch:02d}-{loss:.4f}-{accuracy:.4f}-valid-{val_loss:.4f}-{val_accuracy:.4f}-Model.hdf5'
        checkpoint = ModelCheckpoint(filepath = save_path, monitor = 'val_accuracy', verbose = 1, save_best_only = True)
        callbacks_list = [checkpoint]
        
        
        model = Drug_Model()
        model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
        optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        hist = model.fit_generator(generator = Tr_data, steps_per_epoch = len(Train_list)//batch_size, epochs = 300, validation_data = Val_data, 
                                      validation_steps = len(Valid_list)//batch_size, use_multiprocessing = True, workers = 0, verbose=1, callbacks = callbacks_list)
        scores = model.evaluate_generator(Val_data, steps = len(Valid_list)//batch_size)
        
        fig, loss_ax = py.subplots(figsize = (10,7))
        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'r', linestyle ='--', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'b', linestyle ='--', label='val loss')

        acc_ax.plot(hist.history['accuracy'], 'r', label='train acc')
        acc_ax.plot(hist.history['val_accuracy'], 'b', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        py.show()

    return model, scores

batch_size = 20

Label = ['high', 'inter', 'low']
encoder = LabelEncoder()
encoder.fit(Label)

params = {'dim': (1000, ), 'batch_size': batch_size, 'n_classes': 3, 'n_channels':1, 'shuffle': True}


model, Tr_scores = Model_Train(tr_directory,ca_IDs, IDs, params, batch_size, n_split= 10)




MODEL_SAVE_FOLDER_PATH2 = 'C:/Users/CML_Ye/OneDrive - 금오공과대학교/문서/CNN/3layer_lr/'
model_name = '7-233-0.3147-0.8515-valid-0.2418-0.9254-Model.hdf5'

model = tf.keras.models.load_model(MODEL_SAVE_FOLDER_PATH2+model_name)

# model = tf.keras.models.load_model('C:/Users/CML_Ye/OneDrive - 금오공과대학교/문서/CNN/3layer/'+
#                                    '0-08-0.4954-0.7452-valid-0.3997-0.8163-Model.hdf5')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

def val_data(tr_directory, te_ID, test_IDs):

    
    vali =[]
    
    for i in range(len(te_ID)):
        match_list =list()
        for word in test_IDs:
            if te_ID[i] in word:
                match_list.append(word)
                choiceList = random.choice(match_list)
        vali.append(choiceList)
    return vali

def ca(vm_vm, ca_ca):
    dd =[]
    i =0
    while len(dd) != 2000 :
        i =i+1
        ca = str(ca_ca[i]).split('_')
        
        for k in range(len(vm_vm)):
            vm = str(vm_vm[k]).split('_')
            
            if vm[1] == ca[1] and vm[3] == ca[4]:
                print(ca_ca[i])
                print(vm_vm[k])
                dd.append(ca_ca[i])  
            
    return dd

ca_ca = test_ca_ID

vm_vm = ap
caa = ca(ap,test_ca_ID)


def make_AUC(test_directory, te_ID, test_IDs, model, Label, dataset):
    fpr = dict()
    tpr = dict()
    auc_temp = dict()
    roc_auc = dict()
    auc_value = []
    
    f_score =[]
    acc = []
    
    params = {'dim': (1000, ), 'batch_size': 16, 'n_classes': 3, 'n_channels':1, 'shuffle': False}

    for number in range(dataset):
        ap =val_data(test_directory, test_ID, test_IDs)
        # test,test_y = data_load2(test_directory, ap)
        # test = test.astype("float32")
        # y_pred_cate = model.predict(test)
        
        test_y = val_y(ap) 

        Valid_drug_data = DataGenerator(test_directory, ap, encoder, **params)
        pred = model.predict(Valid_drug_data, verbose =1)
        
        y_pred = np.argmax(pred, axis =1)
        
        # Y_te_cate = to_categorical(test_y)
        # y_pred_cate = pred
        
        f_temp = f1_score(test_y, y_pred, average= None)
        f_score.append(f_temp)
        
        accuracy = accuracy_score(test_y, y_pred)
        acc.append(accuracy) # pred = model.predict(Valid_drug_data)
        Y_te_cate = to_categorical(test_y, num_classes=3)
        y_pred_cate = pred
        
        # roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(len(Label)):
            fpr[i], tpr[i], _ = roc_curve(Y_te_cate[:,i], y_pred_cate[:,i])
            auc_temp =auc(fpr[i],tpr[i])
            auc_value.append(auc_temp)
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            
        # color = plt.cm.rainbow(np.linspace(0, 1, len(Label)))
        # lw = 2 # line width
        # py.figure(figsize = (12, 10))
        # py.rc('axes', labelsize = 30)
        # py.rc('xtick', labelsize = 25)
        # py.rc('ytick', labelsize = 25)    
        # for i, c, in zip(range(len(Label)), color):
        #     py.plot(fpr[i], tpr[i], c=c, lw=lw, label = 'area = %0.2f' %roc_auc[i] + ' of %s' %Label[i])
        # py.plot([0, 1], [0, 1], color = 'gray', lw=lw, linestyle ='--')
        # py.xlim([0.0, 1.0])
        # py.ylim([0.0, 1.05])
        # py.xlabel('False Positive Rate')
        # py.ylabel('True Positive Rate')
        # py.legend(loc='Lower right')
        # py.show()
        
    return np.stack(auc_value), np.stack(f_score), np.stack(acc)



model_folder = os.listdir('C:/Users/CML_Ye/OneDrive - 금오공과대학교/문서/CNN/3layer_lr0.0001_batch20/')

model_name = []
maximum = []
for i,f in enumerate(model_folder):
    print(i)

    model = tf.keras.models.load_model('C:/Users/CML_Ye/OneDrive - 금오공과대학교/문서/CNN/3layer_lr0.0001_batch20/'+ model_folder[i])
        # model_name.append(f)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    AUC_a,_,_= make_AUC(test_directory, test_ID, test_IDs, model, Label, dataset=10)
    low,inter,high = auc_avg(AUC_a)
    maximum.append([np.median(high),np.median(inter),np.median(low)])


auc_all, f, acc = make_AUC(test_directory, test_ID, test_IDs, model, Label, dataset=10)
high,inter,low = auc_avg(auc_all)

# from collections import Counter
# import scipy.stats as st

np.savetxt('fscore.csv',f)


def data_histogram(label, name):
    
    # label= np.delete(label,np.where(label < 0.7))
    # counter = Counter(label)
    
    # print(counter.most_common(1))
    # w = label + 1
    
    # rv = st.rv_discrete(values=(label, w/w.sum()))
    
    # print("median:", rv.median())
    # print("95% CI:", rv.interval(0.95))
          
    print('max: {: .3f}'.format(np.max(label)))
    print('median: {: .3f}'.format(np.median(label)))
    print('min: {: .3f}'.format(np.min(label)))
    
 
    py.figure(figsize=(8,8))
    n, bins, patches = py.hist(label, bins=10, linewidth=6, rwidth = 0.9)
    py.grid(axis='y', alpha=0.75)
    py.xticks(fontsize = 17)
    py.yticks(fontsize = 17)
    py.xlabel('AUC_Value', fontsize = 20)
    py.ylabel('Frequency', fontsize = 20)
    py.title(name, fontsize = 20)
    
frequency_high = data_histogram(label = high, name = 'High')
frequency_inter = data_histogram(label = inter, name = 'Inter')
frequency_low = data_histogram(label = low, name = 'Low')





