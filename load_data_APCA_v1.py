# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:53:30 2020

@author: Abebe
"""

# import os
import numpy as np
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

def drug_List(drug_name):
    
    drug_list_file = r'C:/Users/CML_Ye/OneDrive - 금오공과대학교/문서/cmax1-4/drug_list.csv'
    df =pd.read_csv(drug_list_file, delimiter = ',')
    y_label = df.values
    i = 0
    while drug_name != y_label[i,0]:
        i = i+1
        
    return y_label[i, 1]


def set_Label(risk_level):
    
    if risk_level == 'high':
        Label = np.zeros(1)
    elif risk_level == 'inter':
        Label = np.ones(1)
    else:
        Label = np.ones(1)*2
        
    return Label

    # file_directory='C:/Users/CML_Ye/OneDrive - 금오공과대학교/바탕 화면/data/training/bepridil/132.00/'
    # file_ID = Train_list[0]
    
def data_load2(file_directory, file_ID):
    
    a = file_ID.split('_')
    data =pd.read_csv(file_directory+ file_ID , header =None,delim_whitespace=True)
    criteria =pd.read_csv('C:/Users/CML_Ye/OneDrive - 금오공과대학교/바탕 화면/data/training/bepridil/0.00/bepridil_0.00_vmcheck_smp0.plt' ,
                          header =None,delim_whitespace=True)
    sc = MinMaxScaler()  
    sc.fit_transform(criteria)
    
    # print(file_ID + str(len(data)))
    drug_lisk = drug_List(a[0])
    
    
    # sample = []
    
    # if len(data) == 1001:
    #     re=data.drop(data.index[0])
    #     sample.append(re)
        
    # elif len(data) == 2001:
    #     tmp=data.drop(data.index[0])
    #     for j in range(len(tmp)):
    #         if (j+2)%2 == 0:
    #             sample.append(tmp.iloc[j])
            
    # elif len(data) ==2000:
        
    #     for j in range(len(data)):
    #         if (j+2)%2 == 0:
    #             sample.append(data.loc[j])
                
    # elif len(data) == 1000:
    #     sample.append(data)
        
    # data_re=pd.DataFrame(np.vstack(sample))
    data_MM=sc.transform(data)
    
    return data_MM[:,1], drug_lisk


def val_y(IDs):

    drug_lisk = []
    for i in range(len(IDs)):
        a = IDs[i].split('_')
        
        drug_label = set_Label(drug_List(a[0]))
        drug_lisk.append(drug_label)
        
    
    y = np.vstack(drug_lisk)
    
    return y 

def make_AUC(AUC_a):
    
    avg = []
    avg1 =[]
    avg2 =[]

    for i,v in enumerate(range(len(AUC_a))):
        if (i+3) % 3 ==0:
            avg.append(i)
        elif (i+3) % 3 ==1:
            avg1.append(i)
        elif (i+3) % 3 ==2:
            avg2.append(i)
    return np.stack(avg),np.stack(avg1),np.stack(avg2)

def auc_avg(AUC_a):
    high_auc =[]
    inter_auc =[]
    low_auc =[]
    
    # avg,avg1= extract_index(AUC_a)
    avg,avg1,avg2 = make_AUC(AUC_a)

    for i,v in enumerate(avg):
        temp = AUC_a[v]
        high_auc.append(temp)
        
    for i,v in enumerate(avg1):
        temp = AUC_a[v]
        inter_auc.append(temp)  
        
    for i,v in enumerate(avg2):
        temp = AUC_a[v]
        low_auc.append(temp)
        
    return np.stack(high_auc,axis =0),np.stack(inter_auc,axis =0),np.stack(low_auc,axis =0)