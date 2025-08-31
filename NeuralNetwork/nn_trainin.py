#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Reshape, Input, Flatten, concatenate, Dense, BatchNormalization ,Conv1D,Conv2D,Conv3D,MaxPool1D, MaxPool2D,MaxPool3D, GlobalMaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow.math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scipy
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import pydot
import pydotplus
import random
import pickle
from math import log
from sklearn.preprocessing import MinMaxScaler



NNrun = sys.argv[1]
POPnum = sys.argv[2]
option= sys.argv[3]
bins= sys.argv[4] #fragment windows
ancestries=3 # 
samplePOP=100
TOTALPOP=1000
GEN=19
scale=1000
SCALE=scale


POPname_list=["REDS"]
POP1_list=[0.45]
POP2_list=[0.1]
POPname=POPname_list[int(POPnum)-1]
POP1=POP1_list[int(POPnum)-1]
POP2=POP2_list[int(POPnum)-1]

AMcomb_file='./AM_combinations_pulses_false_scale_'+str(scale)+'_popsize_'+str(TOTALPOP)+'.txt'
AMcomb = np.loadtxt(AMcomb_file, dtype='str', delimiter=' ')
n=len(AMcomb)

x= ["NA" for i in range(n)]
count=0
count2=0
runvec = []
AM1vec = []
AM2vec = []
AM3vec = []
SB1vec = []
SB2vec = []
prop_pop1_t1_vec = []
prop_pop2_t1_vec = []
prop_pop3_t1_vec = []
#prop_pop1_t2_vec = []
#prop_pop2_t2_vec = []
#prop_pop3_t2_vec = []
for AMFMSB in AMcomb:
    run = AMFMSB[0]
    AM1 = AMFMSB[1]
    AM2 = AMFMSB[2]
    AM3 = AMFMSB[3]
    SB1 = AMFMSB[4]
    SB2 = AMFMSB[5]
    prop_pop1_t1 = AMFMSB[6]
    prop_pop2_t1 = AMFMSB[7]
    prop_pop3_t1 = AMFMSB[8]
#    prop_pop1_t2 = AMFMSB[9]
#    prop_pop2_t2 = AMFMSB[10]
#    prop_pop3_t2 = AMFMSB[11]
    #print(run)
    input_frag ='./REDS/fragments/Frags_onlyAM_GEN_19_POP1_0.45_POP2_0.1_totalpop_1000_pop_REDS_scale_1000_run_'+str(run)+'.txt'
    frag_lengths = np.loadtxt(input_frag, dtype='str', delimiter='; ')
#    print(frag_lengths)
#    print("frag_lengths")
    frag_lengths_2 = [[int(cell.replace(';', '')) for cell in row] for row in frag_lengths]
    frag_lengths_3 = np.reshape(frag_lengths_2,(samplePOP,2,ancestries,bins))
     ##sort individuals. first by max ancestry, then by largest mean length fragment of this ancestry.
    maxanc_vec = []
    meanlen_maxanc_vec = []
    prop_maxanc_vec =[]
    for ind in range(samplePOP):
        #n_frags=np.sum(frag_lengths_3[ind][0],axis=1)
        total_lengths=np.sum(frag_lengths_3[ind][0]*range(bins),axis=1)
        maxanc=np.argmax(total_lengths)
        prop_maxanc=(total_lengths)[maxanc]
        maxanc_vec.append(maxanc)
        prop_maxanc_vec.append(prop_maxanc)
    maxanc_meanlen_array = np.array([range(samplePOP),maxanc_vec,prop_maxanc_vec])
    frag_lengths_4=frag_lengths_3[np.lexsort((maxanc_meanlen_array[1,:],maxanc_meanlen_array[2,:]))]
    frag_lengths_5 = np.swapaxes(frag_lengths_4,1,3)
    runvec.append(float(run))
    AM1vec.append(float(AM1))
    AM2vec.append(float(AM2))
    AM3vec.append(float(AM3))
    SB1vec.append(float(SB1))
    SB2vec.append(float(SB2))
    prop_pop1_t1_vec.append(float(prop_pop1_t1))
    prop_pop2_t1_vec.append(float(prop_pop2_t1))
    prop_pop3_t1_vec.append(float(prop_pop3_t1))
#    prop_pop1_t2_vec.append(float(prop_pop1_t2))
#    prop_pop2_t2_vec.append(float(prop_pop2_t2))
#    prop_pop3_t2_vec.append(float(prop_pop3_t2))
    x[count]=frag_lengths_5
    count=count+1
    if (count%100==0):
        print(count)

x=np.array(x)
list_indices=np.array(list(range(len(AMcomb[:,1].astype(float)))))
random.shuffle(list_indices)
list_indices=list_indices.astype(int)
runvec=AMcomb[:,0][list_indices].tolist()    
AM1vec_t1=AMcomb[:,1].astype(float)[list_indices]
AM2vec_t1=AMcomb[:,2].astype(float)[list_indices] 
AM3vec_t1=AMcomb[:,3].astype(float)[list_indices]
SB1vec_t1=AMcomb[:,4].astype(float)[list_indices]
SB2vec_t1=AMcomb[:,5].astype(float)[list_indices]
prop_pop1_t1_vec=AMcomb[:,6].astype(float)[list_indices]
prop_pop2_t1_vec=AMcomb[:,7].astype(float)[list_indices]
prop_pop3_t1_vec=AMcomb[:,8].astype(float)[list_indices]
prop_pop1_t2_vec=AMcomb[:,9].astype(float)[list_indices]
prop_pop2_t2_vec=AMcomb[:,10].astype(float)[list_indices]
prop_pop3_t2_vec=AMcomb[:,11].astype(float)[list_indices]

AM1vec_t1=(np.log(AM1vec_t1)/np.log(3))
AM2vec_t1=(np.log(AM2vec_t1)/np.log(3))
AM3vec_t1=(np.log(AM3vec_t1)/np.log(3))
AM1vec_t1=AM1vec_t1.tolist()
AM2vec_t1=AM2vec_t1.tolist()
AM3vec_t1=AM3vec_t1.tolist()

AM_to_norm = np.concatenate([AM1vec_t1,AM2vec_t1,AM3vec_t1]).reshape(-1,1)
scaler_AM = MinMaxScaler(feature_range=(0,1))
normalized_AM = scaler_AM.fit_transform(AM_to_norm).reshape(3,len(AM1vec_t1)).T
#inverse_AM = scaler_AM.inverse_transform(normalized_AM)
AM1vec_t1=normalized_AM[:,0]
AM2vec_t1=normalized_AM[:,1]
AM3vec_t1=normalized_AM[:,2]

SB_to_norm = np.concatenate([SB1vec_t1,SB2vec_t1]).reshape(-1,1)
scaler_SB = MinMaxScaler(feature_range=(0,1))
normalized_SB = scaler_SB.fit_transform(SB_to_norm).reshape(2,len(SB1vec_t1)).T
inverse_SB = scaler_SB.inverse_transform(normalized_SB)
SB1vec_t1=normalized_SB[:,0]
SB2vec_t1=normalized_SB[:,1]

prop_to_norm = np.concatenate([prop_pop1_t1_vec,prop_pop2_t1_vec,prop_pop3_t1_vec,prop_pop1_t2_vec,prop_pop2_t2_vec,prop_pop3_t2_vec]).reshape(-1,1)
scaler_prop = MinMaxScaler(feature_range=(0,1))
normalized_prop = scaler_prop.fit_transform(prop_to_norm).reshape(6,len(prop_pop1_t1_vec)).T
#inverse_prop = scaler_prop.inverse_transform(normalized_prop)
prop_pop1_t1_vec=normalized_prop[:,0]
prop_pop2_t1_vec=normalized_prop[:,1]
prop_pop3_t1_vec=normalized_prop[:,2]
prop_pop1_t2_vec=normalized_prop[:,3]
prop_pop2_t2_vec=normalized_prop[:,4]
prop_pop3_t2_vec=normalized_prop[:,5]

###
x_trainvaltest=x[:,:,range(22-int(bins),22),:,:]
x_trainval, x_test = train_test_split(x_trainvaltest, test_size=0.2, shuffle= False)
runvec_trainval, runvec_test = train_test_split(runvec, test_size=0.2, shuffle= False)
AM1vec_t1_trainval, AM1vec_t1_test = train_test_split(AM1vec_t1, test_size=0.2, shuffle= False)
AM2vec_t1_trainval, AM2vec_t1_test = train_test_split(AM2vec_t1, test_size=0.2, shuffle= False)
AM3vec_t1_trainval, AM3vec_t1_test = train_test_split(AM3vec_t1, test_size=0.2, shuffle= False)
SB1vec_t1_trainval, SB1vec_t1_test = train_test_split(SB1vec_t1, test_size=0.2, shuffle= False)
SB2vec_t1_trainval, SB2vec_t1_test = train_test_split(SB2vec_t1, test_size=0.2, shuffle= False)
y_trainval=[AM1vec_t1_trainval,
            AM2vec_t1_trainval,
            AM3vec_t1_trainval,
            SB1vec_t1_trainval,
            SB2vec_t1_trainval]
y_test=[AM1vec_t1_test,
             AM2vec_t1_test,
             AM3vec_t1_test,
             SB1vec_t1_test,
             SB2vec_t1_test]
x_trainval_pop = np.mean(x_trainval.tolist(),axis=1)
x_test_pop = np.mean(x_test.tolist(),axis=1)
x_trainval_pop_normalized=[]
for i in range(len(x_trainval_pop)):
    x_trainval_pop_normalized_to_append=[x_trainval_pop[i]/np.apply_over_axes(np.sum, x_trainval_pop[i],axes=[0,1]),
    x_trainval_pop[i]/np.sum(x_trainval_pop[i]),
    x_trainval_pop[i]]
    x_trainval_pop_normalized.append(x_trainval_pop_normalized_to_append[int(option)])
    
x_test_pop_normalized=[]
for i in range(len(x_test_pop)):
    x_test_pop_normalized_to_append=[x_test_pop[i]/np.apply_over_axes(np.sum, x_test_pop[i],axes=[0,1]),
    x_test_pop[i]/np.sum(x_test_pop[i]),
    x_test_pop[i]]
    x_test_pop_normalized.append(x_test_pop_normalized_to_append[int(option)])

x_trainval_pop_normalized=np.array(x_trainval_pop_normalized)
x_test_pop_normalized=np.array(x_test_pop_normalized)
####

#input
from keras.regularizers import l2
input_layer= Input(shape=(bins,ancestries,2))
#Flatten input
flatten_input = Flatten()(input_layer)

#Common dense branch
common_dense_layer0 = Dense(units=512, activation='relu')(flatten_input)
common_dense_layer1 = Dense(units=256, activation='relu')(common_dense_layer0)
common_dense_layer2 = Dense(units=128, activation='relu')(common_dense_layer1)
common_dense_layer3 = Dense(units=64, activation='relu')(common_dense_layer2)
common_dropout = Dropout(0.2)(common_dense_layer3)

#Final AM1 hidden layer, dropout and output layer
am1_dense_layer = Dense(units=32, activation='relu')(common_dropout)
am1_dropout1 = Dropout(0.2)(am1_dense_layer)
am1_output = Dense(units=1, name='am1_output', activation='sigmoid')(am1_dropout1)

#Final AM2 hidden layer, dropout and output layer
am2_dense_layer = Dense(units=32, activation='relu')(common_dropout)
am2_dropout1 = Dropout(0.2)(am2_dense_layer)
am2_output = Dense(units=1, name='am2_output', activation='sigmoid')(am2_dropout1)

#Final AM3 hidden layer, dropout and output layer
am3_dense_layer = Dense(units=32, activation='relu')(common_dropout)
am3_dropout1 = Dropout(0.2)(am3_dense_layer)
am3_output = Dense(units=1, name='am3_output', activation='sigmoid')(am3_dropout1)


#Final SB1 hidden layer, dropout and output layer
sb1_dense_layer = Dense(units=32, activation='relu')(common_dropout)
sb1_dropout1 = Dropout(0.2)(sb1_dense_layer)
sb1_output = Dense(units=1, name='sb1_output', activation='sigmoid')(sb1_dropout1)

#Final SB2 hidden layer, dropout and output layer
sb2_dense_layer = Dense(units=32, activation='relu')(common_dropout)
sb2_dropout1 = Dropout(0.2)(sb2_dense_layer)
sb2_output = Dense(units=1, name='sb2_output', activation='sigmoid')(sb2_dropout1)


model = Model(inputs=input_layer, outputs=[am1_output, am2_output, am3_output, sb1_output, sb2_output])
model.compile(optimizer='adam',
              #loss=tf.keras.losses.Huber(delta=5.0))  
              loss='mse')
              #loss={'am1_output_m9': 'mae', 'am2_output_m9': 'mae','am3_output_m9': 'mae', 'sb1_output_m9': 'mae', 'sb2_output_m9': 'mae'})
              #loss_weights={'am1_output': 0.1, 'am2_output': 0.1, 'am3_output': 0.1, 'sb1_output': 10., 'sb2_output': 10.,},
              #metrics={'am1_output_m7': 'accuracy','am2_output_m7': 'accuracy','am3_output_m7': 'accuracy','sb1_output_m7': 'accuracy','sb2_output_m7': 'accuracy'})
              
model.summary()
####
n_epochs=40
history = model.fit(
	x_trainval_pop_normalized,
        y_trainval,
	epochs=n_epochs,
	batch_size=64,
	validation_split=0.2
)
####
history_df=pd.DataFrame(history.history)
history_df['NN']=NNrun
with open("./network/loss_epochs_onlyAM_REDS_bins_" + str(bins) + "_normoption_" + str(option) + ".txt", 'a') as f:
    historyAsString = history_df.to_string(header=True, index=False)
    f.write(historyAsString)
    
"""
#gnomix
with open("/rds/general/user/amassand/home/SLIM_models/MODELS/pops/Bootstrap_"+str(POPname)+"_log_gnomix_maxanc", 'rb') as f:
    bootstrap_gnomix_Aut= pickle.load(f)

with open("/rds/general/user/amassand/home/SLIM_models/MODELS/pops/Bootstrap_"+str(POPname)+"_log_gnomix_maxanc_X_onlyfemales", 'rb') as f:
    bootstrap_gnomix_X= pickle.load(f)

bootstrap_gnomix_X=np.array(bootstrap_gnomix_X[0])
bootstrap_gnomix_Aut=np.array(bootstrap_gnomix_Aut[0])
#rfmix
with open("/rds/general/user/amassand/home/SLIM_models/MODELS/pops/Bootstrap_"+str(POPname)+"_log_rfmix_maxanc", 'rb') as f:
    bootstrap_rfmix_Aut= pickle.load(f)

with open("/rds/general/user/amassand/home/SLIM_models/MODELS/pops/Bootstrap_"+str(POPname)+"_log_rfmix_maxanc_X_onlyfemales", 'rb') as f:
    bootstrap_rfmix_X= pickle.load(f)

bootstrap_rfmix_X=np.array(bootstrap_rfmix_X[0])
bootstrap_rfmix_Aut=np.array(bootstrap_rfmix_Aut[0])

####

target_panel=np.loadtxt("/rds/general/project/human-popgen-datasets/live/HGDP_1000g/RFMix/target_panel.txt",delimiter='\t',dtype="str")
targetAFRAME_panel=np.loadtxt("/rds/general/project/human-popgen-datasets/live/HGDP_1000g/RFMix/targetAFRAME_panel.txt",delimiter='\t',dtype="str")
all_inds=np.concatenate([target_panel,targetAFRAME_panel],axis=0)
inds_1=np.where(all_inds[:,1]==str(POPname))[0]*6
inds_2=np.where(all_inds[:,1]==str(POPname))[0]*6+1
inds_3=np.where(all_inds[:,1]==str(POPname))[0]*6+2
inds_4=np.where(all_inds[:,1]==str(POPname))[0]*6+3
inds_5=np.where(all_inds[:,1]==str(POPname))[0]*6+4
inds_6=np.where(all_inds[:,1]==str(POPname))[0]*6+5
indices=np.sort(np.concatenate([inds_1,inds_2,inds_3,inds_4,inds_5,inds_6]))
females_1=np.where((all_inds[:,1]==str(POPname))&(all_inds[:,2]=="F"))[0]*6
females_2=np.where((all_inds[:,1]==str(POPname))&(all_inds[:,2]=="F"))[0]*6+1
females_3=np.where((all_inds[:,1]==str(POPname))&(all_inds[:,2]=="F"))[0]*6+2
females_4=np.where((all_inds[:,1]==str(POPname))&(all_inds[:,2]=="F"))[0]*6+3
females_5=np.where((all_inds[:,1]==str(POPname))&(all_inds[:,2]=="F"))[0]*6+4
females_6=np.where((all_inds[:,1]==str(POPname))&(all_inds[:,2]=="F"))[0]*6+5
males_1=np.where((all_inds[:,1]==str(POPname))&(all_inds[:,2]=="M"))[0]*6
males_2=np.where((all_inds[:,1]==str(POPname))&(all_inds[:,2]=="M"))[0]*6+1
males_3=np.where((all_inds[:,1]==str(POPname))&(all_inds[:,2]=="M"))[0]*6+2
indices_X=np.sort(np.concatenate([females_1,females_2,females_3,females_4,females_5,females_6]))
n_females=np.shape(np.where((all_inds[:,1]==str(POPname))&(all_inds[:,2]=="F")))[1]
n_males=np.shape(np.where((all_inds[:,1]==str(POPname))&(all_inds[:,2]=="M")))[1]

####
#rfmix
x_real_pop_bootstrap_rfmix=[]
for i in range(1000):
    bootstrap_rfmix_pop_X=np.mean(bootstrap_rfmix_X[i].reshape(22,int(np.shape(indices_X)[0]/3),3),axis=1)
    bootstrap_rfmix_pop_Aut=np.mean(bootstrap_rfmix_Aut[i].reshape(22,int(np.shape(indices)[0]/3),3),axis=1)
    x_real_pop=np.stack((bootstrap_rfmix_pop_Aut,bootstrap_rfmix_pop_X),axis=-1)
    x_real_pop_bootstrap_rfmix.append(x_real_pop)    

x_real_pop_bootstrap_rfmix=np.array(x_real_pop_bootstrap_rfmix)
x_real_pop_bootstrap_rfmix=x_real_pop_bootstrap_rfmix[:,range(22-int(bins),22),:,:]
x_real_pop_bootstrap_rfmix=x_real_pop_bootstrap_rfmix[:,:,np.array((2,0,1)),:]
x_real_pop_bootstrap_rfmix_normalized=[]
for i in range(len(x_real_pop_bootstrap_rfmix)):
    x_real_pop_bootstrap_rfmix_normalized_to_append=[x_real_pop_bootstrap_rfmix[i]/np.apply_over_axes(np.sum, x_real_pop_bootstrap_rfmix[i],axes=[0,1]),
    x_real_pop_bootstrap_rfmix[i]/np.sum(x_real_pop_bootstrap_rfmix[i]),
    x_real_pop_bootstrap_rfmix[i]]
    x_real_pop_bootstrap_rfmix_normalized.append(x_real_pop_bootstrap_rfmix_normalized_to_append[int(option)])
    
x_real_pop_bootstrap_rfmix_normalized=np.array(x_real_pop_bootstrap_rfmix_normalized)
####
# gnomix
x_real_pop_bootstrap_gnomix=[]
for i in range(1000):
    bootstrap_gnomix_pop_X=np.mean(bootstrap_gnomix_X[i].reshape(22,int(np.shape(indices_X)[0]/3),3),axis=1)
    bootstrap_gnomix_pop_Aut=np.mean(bootstrap_gnomix_Aut[i].reshape(22,int(np.shape(indices)[0]/3),3),axis=1)
    x_real_pop=np.stack((bootstrap_gnomix_pop_Aut,bootstrap_gnomix_pop_X),axis=-1)
    x_real_pop_bootstrap_gnomix.append(x_real_pop)    

x_real_pop_bootstrap_gnomix=np.array(x_real_pop_bootstrap_gnomix)
x_real_pop_bootstrap_gnomix=x_real_pop_bootstrap_gnomix[:,range(22-int(bins),22),:,:]
x_real_pop_bootstrap_gnomix_normalized=[]
for i in range(len(x_real_pop_bootstrap_gnomix)):
    x_real_pop_bootstrap_gnomix_normalized_to_append=[x_real_pop_bootstrap_gnomix[i]/np.apply_over_axes(np.sum, x_real_pop_bootstrap_gnomix[i],axes=[0,1]),
    x_real_pop_bootstrap_gnomix[i]/np.sum(x_real_pop_bootstrap_gnomix[i]),
    x_real_pop_bootstrap_gnomix[i]]
    x_real_pop_bootstrap_gnomix_normalized.append(x_real_pop_bootstrap_gnomix_normalized_to_append[int(option)])


x_real_pop_bootstrap_gnomix_normalized=np.array(x_real_pop_bootstrap_gnomix_normalized)
####
"""


y_pred= np.array(model.predict(x_test_pop_normalized))
y_pred=y_pred[:,:,0]
y_test_df=pd.DataFrame(y_test).T
y_test_df.columns = ['AM1_true', 'AM2_true', 'AM3_true', 'SB1_true', 'SB2_true']
y_pred_df=pd.DataFrame(y_pred).T
y_pred_df.columns = ['AM1_pred', 'AM2_pred', 'AM3_pred','SB1_pred', 'SB2_pred']
y_true_pred=pd.concat([y_test_df,y_pred_df],axis=1)
####

"""
y_real_pred_rfmix= np.array(model.predict(x_real_pop_bootstrap_rfmix_normalized))
y_real_pred_rfmix=y_real_pred_rfmix[:,:,0]
y_real_pred_rfmix_df=pd.DataFrame(y_real_pred_rfmix).T
y_real_pred_rfmix_df.columns = ['AM1', 'AM2', 'AM3', 'SB1', 'SB2']
####
y_real_pred_gnomix= np.array(model.predict(x_real_pop_bootstrap_gnomix_normalized))
y_real_pred_gnomix=y_real_pred_gnomix[:,:,0]
y_real_pred_gnomix_df=pd.DataFrame(y_real_pred_gnomix).T
y_real_pred_gnomix_df.columns = ['AM1', 'AM2', 'AM3', 'SB1', 'SB2']
####
"""

y_true_pred[['NN']]=NNrun
y_true_pred.to_csv("./REDS/network/truepred_onlyAM_REDS_bins_" + str(bins) + "_normoption_" + str(option) + ".txt", index=None, sep=' ', mode='a')

file_mse_object = open("./REDS/network/MSE_R2_test_onlyAM_REDS_bins_" + str(bins) + "_normoption_" + str(option) + ".txt", 'a')
file_mse_object.write("AM1 MSE "+str(mean_squared_error(y_true_pred['AM1_true'],y_true_pred['AM1_pred']))+" "+str(NNrun)+"\n")
file_mse_object.write("AM2 MSE "+str(mean_squared_error(y_true_pred['AM2_true'],y_true_pred['AM2_pred']))+" "+str(NNrun)+"\n")
file_mse_object.write("AM3 MSE "+str(mean_squared_error(y_true_pred['AM3_true'],y_true_pred['AM3_pred']))+" "+str(NNrun)+"\n")
file_mse_object.write("SB1 MSE "+str(mean_squared_error(y_true_pred['SB1_true'],y_true_pred['SB1_pred']))+" "+str(NNrun)+"\n")
file_mse_object.write("SB2 MSE "+str(mean_squared_error(y_true_pred['SB2_true'],y_true_pred['SB2_pred']))+" "+str(NNrun)+"\n")
file_mse_object.write("AM1 R2 "+str(r2_score(y_true_pred['AM1_true'],y_true_pred['AM1_pred']))+" "+str(NNrun)+"\n")
file_mse_object.write("AM2 R2 "+str(r2_score(y_true_pred['AM2_true'],y_true_pred['AM2_pred']))+" "+str(NNrun)+"\n")
file_mse_object.write("AM3 R2 "+str(r2_score(y_true_pred['AM3_true'],y_true_pred['AM3_pred']))+" "+str(NNrun)+"\n")
file_mse_object.write("SB2 R2 "+str(r2_score(y_true_pred['SB1_true'],y_true_pred['SB1_pred']))+" "+str(NNrun)+"\n")
file_mse_object.write("SB1 R2 "+str(r2_score(y_true_pred['SB2_true'],y_true_pred['SB2_pred']))+" "+str(NNrun)+"\n")
file_mse_object.close()


"""
file_pred_gnomix_object = open("/rds/general/project/human-popgen-datasets/live/CNN_AM/log_pulses_false_newgenmap/predicted_pulses_FALSE_"+str(POPname)+"_gnomix_maxanc_bins_"+str(bins)+"_normoption_"+str(option)+"_onlyfemales.txt", 'a')
file_pred_gnomix_object.write("AM1 pred "+str(np.mean(y_real_pred_gnomix_df,axis=0)['AM1'])+" "+str(NNrun)+"\n")
file_pred_gnomix_object.write("AM2 pred "+str(np.mean(y_real_pred_gnomix_df,axis=0)['AM2'])+" "+str(NNrun)+"\n")
file_pred_gnomix_object.write("AM3 pred "+str(np.mean(y_real_pred_gnomix_df,axis=0)['AM3'])+" "+str(NNrun)+"\n")
file_pred_gnomix_object.write("SB1 pred "+str(np.mean(y_real_pred_gnomix_df,axis=0)['SB1'])+" "+str(NNrun)+"\n")
file_pred_gnomix_object.write("SB2 pred "+str(np.mean(y_real_pred_gnomix_df,axis=0)['SB2'])+" "+str(NNrun)+"\n")
file_pred_gnomix_object.close()

file_pred_rfmix_object = open("/rds/general/project/human-popgen-datasets/live/CNN_AM/log_pulses_false_newgenmap/predicted_pulses_FALSE_"+str(POPname)+"_rfmix_maxanc_bins_"+str(bins)+"_normoption_"+str(option)+"_onlyfemales.txt", 'a')
file_pred_rfmix_object.write("AM1 pred "+str(np.mean(y_real_pred_rfmix_df,axis=0)['AM1'])+" "+str(NNrun)+"\n")
file_pred_rfmix_object.write("AM2 pred "+str(np.mean(y_real_pred_rfmix_df,axis=0)['AM2'])+" "+str(NNrun)+"\n")
file_pred_rfmix_object.write("AM3 pred "+str(np.mean(y_real_pred_rfmix_df,axis=0)['AM3'])+" "+str(NNrun)+"\n")
file_pred_rfmix_object.write("SB1 pred "+str(np.mean(y_real_pred_rfmix_df,axis=0)['SB1'])+" "+str(NNrun)+"\n")
file_pred_rfmix_object.write("SB2 pred "+str(np.mean(y_real_pred_rfmix_df,axis=0)['SB2'])+" "+str(NNrun)+"\n")
file_pred_rfmix_object.close()
"""
