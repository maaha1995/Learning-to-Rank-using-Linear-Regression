#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:06:07 2017

@author: mahalakshmimaddu
"""


import numpy as np
from numpy.linalg import inv
import scipy as sc
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2
import math
import pandas as pd
import matplotlib.pyplot as plt

def computeDesignMatrix(X, centers, spreads):
        basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads)*(X - centers),axis=2)/(-2)).T
        return np.insert(basis_func_outputs, 0, 1, axis=1)

def closed_form_sol(L2_lambda, design_matrix, output_data):
    return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) + np.matmul(design_matrix.T, design_matrix),
        np.matmul(design_matrix.T, output_data)).flatten()

def getKmeans(input, Cluster_No):
    kmeans = kmeans2(input, Cluster_No, minit='points')
    centroids = kmeans[0]
    centroids = centroids[:, np.newaxis, :]
    cluster_labels = np.array(kmeans[1])
    return centroids,cluster_labels

def eRms(input_dataset, designmatrix, wml, lambda1):
    output = np.matmul(designmatrix, wml).reshape([-1,1])
    error = np.sum(np.power((input_dataset - output),2))/2 + lambda1*0.5*np.matmul(wml.T, wml)
    eRms = math.sqrt(2*error/input_dataset.shape[0])
    return eRms

def SGD_sol(learning_rate,
            minibatch_size,
            validationSteps,
            num_epochs,
            L2_lambda,
            training_design_matrix,
            training_target,
            patience,
            validation_target,
            validation_design_matrix ):

         weights = np.random.randn(1, training_design_matrix.shape[1])
         best_weights = np.random.randn(1, training_design_matrix.shape[1])
         N,_ = training_design_matrix.shape
         current_iteration_count=0
         
         best_error = 20

         #minibatch_size=1
         minibatch_size = int(training_design_matrix.shape[0]/ validationSteps)

         random_basis_function = np.random.choice(training_design_matrix.shape[0],minibatch_size,replace=False)

         for epoch in range(num_epochs):
             for i in range(int(N/minibatch_size)):

                 random_basis_function = np.random.choice(training_design_matrix.shape[0],minibatch_size,replace=False)
                 basis_function = training_design_matrix[random_basis_function]
                 target_values = training_target[random_basis_function]

                 ED = np.matmul((np.matmul(basis_function, weights.T) - target_values).T, basis_function)
                 E = (ED + L2_lambda*weights)/minibatch_size

                 weights = weights - learning_rate * E


             current_error = eRms(validation_target, validation_design_matrix, weights.T, L2_lambda)

             if current_error <= best_error:
                 best_error = current_error
                 best_weights = weights
             else:
                 if current_iteration_count > patience: break
                 else: current_iteration_count = current_iteration_count + 1

         return best_weights.flatten()




###Importing the LeToR data
letor_input_data = np.genfromtxt(
'Querylevelnorm_X.csv', delimiter=',')
letor_output_data = np.genfromtxt(
'Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])


###Partitioning the data

n = letor_input_data.shape[0]
end_training = int(np.floor(n*0.80))
start_validation =int(np.floor(n*0.80))
end_validation =int(np.floor(n * 0.90))
start_test = int(np.floor(n * 0.90))
end_test = n

training_input = letor_input_data[:end_training,:]
validation_input = letor_input_data[start_validation:end_validation,:]
test_input = letor_input_data[start_test:end_test,:]
training_target = letor_output_data[:end_training,:]
validation_target = letor_output_data[start_validation:end_validation,:]
test_target = letor_output_data[start_test:end_test,:]


N , M = training_input.shape


##################################################################
####SClosed Form Solution
##################################################################





#Hyperparamters
Cluster_No = 11
lambda1 =0.6
learning_rate = 0.05
print("\n")

print("\t LeToR Data")

print("\n")

print("Number of Clusters for LeToR Data = ", Cluster_No)
print("Value of Lambda for LeToR Data = ", lambda1)
print("Value of Learning rate for LeToR Data = ", learning_rate)


print("\n")


####Design Matrix for Training

centroids_tr,clusterlabels_tr = getKmeans(training_input,Cluster_No)
spreads = np.ndarray(shape=(Cluster_No,M,M))
for i in range(Cluster_No):
   spreads[i] = np.linalg.pinv(np.cov(training_input[np.where(clusterlabels_tr ==i)].T))


designmatrix_tr = computeDesignMatrix(training_input[np.newaxis, :, :], centroids_tr, spreads)

designmatrix_V = computeDesignMatrix(validation_input[np.newaxis, :, :], centroids_tr, spreads)

designmatrix_T = computeDesignMatrix(test_input[np.newaxis, :, :], centroids_tr, spreads)


wml = closed_form_sol(lambda1, designmatrix_tr, training_target)

eRms_tr= eRms(training_target, designmatrix_tr, wml, lambda1)
print("LeToR-Training-CF ERMS = ",eRms_tr)

eRms_V= eRms(validation_target, designmatrix_V, wml, lambda1)
print("LeToR-Validation-CF ERMS = ",eRms_V)

eRms_T= eRms(test_target, designmatrix_T, wml, lambda1)
print("LeToR-Test ERMS-CF = ",eRms_T)

##################################################################
####Stochastic Gradient Solution
##################################################################




####Hyper Parameters
learning_rate = 0.05
minibatch_size = 20 #minibatch = 1 for stochastic gradient descent, must be lower than N
validationSteps = 10
num_epochs = 100 #higher is better
L2_lambda = lambda1
patience = 10





wSGD = SGD_sol(learning_rate, minibatch_size, validationSteps, num_epochs, L2_lambda, designmatrix_tr, training_target, patience, validation_target,designmatrix_V)


#####Training Error

SGDeRMS_tr = eRms(training_target,designmatrix_tr,wSGD,L2_lambda)
print("LeToR-Training-SGD ERMS =",SGDeRMS_tr)
#####Validation Error

SGDeRMS_V = eRms(validation_target,designmatrix_V,wSGD,L2_lambda)
print("LeToR-Validation-SGD ERMS = ",SGDeRMS_V)
#####Test Error
SGDeRMS_T = eRms(test_target,designmatrix_T,wSGD,L2_lambda)
print("LeToR-Test-SGD ERMS = ",SGDeRMS_T)





##################################################################
####Synthetic Data
##################################################################



####Importing the Synthetic data
   
dfi = pd.read_table('input.csv', sep=';|,', engine='python', header=None)
syn_input_data = dfi.values

dfo = pd.read_table('output.csv', sep=';|,', engine='python', header=None)
syn_output_data = dfo.values


###Partitioning the data

n = syn_input_data.shape[0]
end_training = int(np.floor(n*0.80))
start_validation =int(np.floor(n*0.80))
end_validation =int(np.floor(n * 0.90))
start_test = int(np.floor(n * 0.90))
end_test = n

training_input = syn_input_data[:end_training,:]
validation_input = syn_input_data[start_validation:end_validation,:]
test_input = syn_input_data[start_test:end_test,:]
training_target = syn_output_data[:end_training,:]
validation_target = syn_output_data[start_validation:end_validation,:]
test_target = syn_output_data[start_test:end_test,:]


N , M = training_input.shape



##################################################################
####SClosed Form Solution
##################################################################



print("\n")

#Hyperparamters
Cluster_No = 6
lambda1 =0.1
learning_rate = 0.05



print("\t Synthetic Data")
print("\n")
print("Number of Clusters for Synthetic Data = ", Cluster_No)
print("Value of Lambda for Synthetic Data = ", lambda1)
print("Value of Learning Rate for Synthetic Data = ", learning_rate)


print("\n")

####Design Matrix for Training

centroids_tr,clusterlabels_tr = getKmeans(training_input,Cluster_No)
spreads = np.ndarray(shape=(Cluster_No,M,M))
for i in range(Cluster_No):
   spreads[i] = np.linalg.pinv(np.cov(training_input[np.where(clusterlabels_tr ==i)].T))


designmatrix_tr = computeDesignMatrix(training_input[np.newaxis, :, :], centroids_tr, spreads)

designmatrix_V = computeDesignMatrix(validation_input[np.newaxis, :, :], centroids_tr, spreads)

designmatrix_T = computeDesignMatrix(test_input[np.newaxis, :, :], centroids_tr, spreads)


wml = closed_form_sol(lambda1, designmatrix_tr, training_target)

eRms_tr= eRms(training_target, designmatrix_tr, wml, lambda1)
print("Synthetic-Training-CF ERMS = ",eRms_tr)

eRms_V= eRms(validation_target, designmatrix_V, wml, lambda1)
print("Synthetic-Validation-CF ERMS = ",eRms_V)

eRms_T= eRms(test_target, designmatrix_T, wml, lambda1)
print("Synthetic-Test ERMS-CF = ",eRms_T)

##################################################################
####Stochastic Gradient Solution
##################################################################




####Hyper Parameters
learning_rate = 0.05
minibatch_size = 20 #minibatch = 1 for stochastic gradient descent, must be lower than N
validationSteps = 10
num_epochs = 100 #higher is better
L2_lambda = lambda1
patience = 10





wSGD = SGD_sol(learning_rate, minibatch_size, validationSteps, num_epochs, L2_lambda, designmatrix_tr, training_target, patience, validation_target,designmatrix_V)


#####Training Error

SGDeRMS_tr = eRms(training_target,designmatrix_tr,wSGD,L2_lambda)
print("Synthetic-Training-SGD ERMS =",SGDeRMS_tr)
#####Validation Error

SGDeRMS_V = eRms(validation_target,designmatrix_V,wSGD,L2_lambda)
print("Synthetic-Validation-SGD ERMS = ",SGDeRMS_V)
#####Test Error
SGDeRMS_T = eRms(test_target,designmatrix_T,wSGD,L2_lambda)
print("Synthetic-Test-SGD ERMS = ",SGDeRMS_T)






####plots for Cluster_no Vs Erms-Test
#Incrementing cluster no values by 5 from 1 to 50

Clusterno_array = [1,6,11,16,21,26,32,37,42,47]
Ermstest_array = [0.7530136645703516,0.7636426883571519,0.8148819725279984,0.8445718914643472,0.8547460438039938,0.8596164560239363,0.8662676018510036,0.8710310765396275,0.8701046061567302,0.8731968744200991]
plt.plot(Clusterno_array,Ermstest_array)
plt.xlabel('Cluster numbers')
plt.ylabel('test Error')
plt.title('Plot between Clusternumbers and test error for Cluster number incrementing by 5')
plt.show()


####for Cluster number = 1
###Relationship between Lambda and Error test for different clusterno values

Lambda_array1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9]
Ermtest_array1 = [0.7530136645703516, 0.7552135903242055, 0.7573439757069543, 0.7594075491144974, 0.7614069048023252, 0.7633445110048269, 0.7652227174622469, 0.7670437624063253, 0.7688097790505638]
plt.plot(Lambda_array1,Ermtest_array1)
plt.xlabel('Lambda values')
plt.ylabel('test Error')
plt.title('Plot between lambda and test error for Cluster number = 1')
plt.show()


Lambda_array2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9]
Ermtest_array2 = [0.7633347297648283, 0.759461088118034, 0.7750113113311182, 0.7824684123960662, 0.7879459224806535, 0.7947034346855054, 0.8069557924034638, 0.7990988785039944, 0.8124618587787223]
plt.plot(Lambda_array2,Ermtest_array2)
plt.xlabel('Lambda values')
plt.ylabel('test Error')
plt.title('Plot between lambda and test error for Cluster number = 6')
plt.show()

Lambda_array3 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9]
Ermtest_array3 = [0.7669704493446595, 0.7706107609215239, 0.791709919265346, 0.8056058440652495, 0.8095493036666173, 0.8215459664183878, 0.825312001013776, 0.8318675187401078, 0.834967728952202]
plt.plot(Lambda_array3,Ermtest_array3)
plt.xlabel('Lambda values')
plt.ylabel('test Error')
plt.title('Plot between lambda and test error for Cluster number = 11')
plt.show()

Lambda_array4 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9]
Ermtest_array4 = [0.7612835398846745, 0.7992001432397942, 0.8053394601213512, 0.8224761277070649, 0.8306085030573764, 0.8439754625667458, 0.8347978548381665, 0.8532459180630128, 0.8396063520682918]
plt.plot(Lambda_array4,Ermtest_array4)
plt.xlabel('Lambda values')
plt.ylabel('test Error')
plt.title('Plot between lambda and test error for Cluster number = 16')
plt.show()

Lambda_array5 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9]
Ermtest_array5 = [0.7784208542423675, 0.8047478032912716, 0.8231376189263215, 0.8427926458140718, 0.8324734783176334, 0.8580959651949535, 0.8549123790998037, 0.843184552452672, 0.8510767454772117]
plt.plot(Lambda_array5,Ermtest_array5)
plt.xlabel('Lambda values')
plt.ylabel('test Error')
plt.title('Plot between lambda and test error for Cluster number = 21')
plt.show()

Lambda_array6 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9]
Ermtest_array6 = [0.7814394159881075, 0.8110838748779395, 0.8345022875908219, 0.8380412975486473, 0.8550760596068465, 0.854193558620264, 0.85935835848885, 0.8578941519374413, 0.8629339947547497]
plt.plot(Lambda_array6,Ermtest_array6)
plt.xlabel('Lambda values')
plt.ylabel('test Error')
plt.title('Plot between lambda and test error for Cluster number = 26')
plt.show()

Lambda_array7 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9]
Ermtest_array7 = [0.7939941941759933, 0.8218024751157377, 0.833780167040465, 0.8597095750795138, 0.8613378893546697, 0.8600519636619881, 0.8657101124352724, 0.8760741096426207, 0.862269777948161]
plt.plot(Lambda_array7,Ermtest_array7)
plt.xlabel('Lambda values')
plt.ylabel('test Error')
plt.title('Plot between lambda and test error for Cluster number = 31')
plt.show()

Lambda_array8 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9]
Ermtest_array8 = [0.8018926225979527, 0.831083968553598, 0.8586184232474716, 0.8530697839793451, 0.8603278258865317, 0.8674176267591728, 0.8684293270382577, 0.8610446898637292, 0.86291936099881]
plt.plot(Lambda_array8,Ermtest_array8)
plt.xlabel('Lambda values')
plt.ylabel('test Error')
plt.title('Plot between lambda and test error for Cluster number = 36')
plt.show()

Lambda_array9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9]
Ermtest_array9 = [0.8022122390087161, 0.8470851704823577, 0.8588614761323317, 0.8670890031959104, 0.8769873235048978, 0.864887292765536, 0.8610248009168252, 0.866754180230945, 0.8708467322595576]
plt.plot(Lambda_array9,Ermtest_array9)
plt.xlabel('Lambda values')
plt.ylabel('test Error')
plt.title('Plot between lambda and test error for Cluster number = 41')
plt.show()

Lambda_array10 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9]
Ermtest_array10 = [0.8149374951125796, 0.839492485826261, 0.8653260813323276, 0.8700747940486855, 0.8631144661651514, 0.8664879527195054, 0.8662813934386169, 0.8690203870632027, 0.8678828340598502]
plt.plot(Lambda_array10,Ermtest_array10)
plt.xlabel('Lambda values')
plt.ylabel('test Error')
plt.title('Plot between lambda and test error for Cluster number = 46')
plt.show()