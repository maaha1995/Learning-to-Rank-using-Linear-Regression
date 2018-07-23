#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 02:09:38 2017

@author: mahalakshmimaddu
"""

import numpy as np
from numpy.linalg import inv
import scipy as sc
from sklearn.cluster import KMeans
import pandas as pd  

centres1 = []

def kMeans(M,data):
        kmeans = KMeans(n_clusters=M,random_state=0)
        kmeans.fit(data)
        centroid = kmeans.cluster_centers_
        
        labels=kmeans.labels_
        clusters = {}
        for label in range(0,M):
                cluster_index = np.where(labels == label)[0]
                cluster_data = []
                for p in cluster_index :
                    cluster_data.append(data[p])
                cluster_data = np.array(cluster_data)
                clusters[label] = cluster_data
        return centroid,clusters
    
    
def give_spreads(cluster,M):
        spread = []
        for i in range(M):
            spread.append(np.linalg.pinv(np.cov((np.array(cluster[i]).T))))
        return spread

    
def evaluate_design_matrix(data_matrix, centers, spreads):
        
        design_matrix = []
        basis_functions = np.exp(np.sum(np.matmul(data_matrix-centers,spreads) * (data_matrix - centers),axis=2)/(-2)).T
        return np.insert(basis_functions,0,1, axis = 1)
        
        
def closed_form_solution(lambda_v,design_matrix,targets): 
        return np.linalg.solve(
        lambda_v * np.identity(design_matrix.shape[1]) + np.matmul(design_matrix.T, design_matrix),
        np.matmul(design_matrix.T,targets) ).flatten()
     
def errorD(design_matrix,wmls,target):
        return np.sum((np.matmul(design_matrix,wmls.T) - target)**2) 
      
def error_functionW(weights):
        return np.dot(weights.T,weights)*(0.5)

def final_error(ED,EW,lamda):
        return ED+(lamda*EW)

def ermns(test_set,errorw):
        return np.sqrt((2*errorw)/len(test_set))
    
    
    
### Importing the Files     


   
dfi = pd.read_table('/home/csgrad/mmaddu/mlproj2/input1.csv', sep=';|,', engine='python', header=None)
syn_input_data = dfi.values

dfo = pd.read_table('/home/csgrad/mmaddu/mlproj2/output1.csv', sep=';|,', engine='python', header=None)
syn_output_data = dfo.values




#####Diving the data into training,test and validation


n = syn_input_data.shape[0]
end_training = int(np.floor(n*0.80))
#end_training = 50000
start_validation =int(np.floor(n*0.80))
end_validation =int(np.floor(n * 0.90))
start_test = int(np.floor(n * 0.90))
end_test = n


training = syn_input_data[:end_training,:]
validation = syn_input_data[start_validation:end_validation,:]
test = syn_input_data[start_test:end_test,:]

training_target = syn_output_data[:end_training,:]
validation_target = syn_output_data[start_validation:end_validation,:]
test_target = syn_output_data[start_test:end_test,:]



###Hyper Parameters for Closed Form Solution

M=10
lambda1 = 1



####Training Set /Closed Form

centroid,means = kMeans(M,training)
spreads = give_spreads(means,M)
centroid = centroid[ : ,np.newaxis, :]
data_matrix = training[np.newaxis, : , :]
design_matrix = evaluate_design_matrix(data_matrix,centroid,spreads)#spreads)
wml_star = closed_form_solution(lambda1,design_matrix,training_target)

eD = errorD(design_matrix,wml_star,training_target)
eW = error_functionW(wml_star)
tErr = final_error(eD,eW,lambda1)
eRMS = ermns(training,eW)
print(eRMS)



###Validation Set / Closed Form

centroidV,meansV = kMeans(M,validation)
spreadsV = give_spreads(meansV,M)
centroidV = centroidV[ : ,np.newaxis, :]
data_matrixV = validation[np.newaxis, : , :]
design_matrixV = evaluate_design_matrix(data_matrixV,centroidV,spreadsV)
wml_star = closed_form_solution(lambda1,design_matrix,training_target)

eDV = errorD(design_matrixV,wml_star,validation_target)
eWV = error_functionW(wml_star)
tErrV = final_error(eDV,eWV,lambda1)
eRMSV = ermns(validation,eWV)
print(eRMSV)




###Test Set  / Closed Form

centroidT,meansT = kMeans(M,test)
spreadsT = give_spreads(meansT,M)
centroidT = centroidT[ : ,np.newaxis, :]
data_matrixT = test[np.newaxis, : , :]
design_matrixT = evaluate_design_matrix(data_matrixT,centroidT,spreadsT)
wml_star = closed_form_solution(lambda1,design_matrix,training_target)

eDT = errorD(design_matrixT,wml_star,test_target)
eWT = error_functionW(wml_star)
tErrT = final_error(eDT,eWT,lambda1)
eRMST = ermns(test,eWT)
print(eRMST)





































