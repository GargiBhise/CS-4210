#-------------------------------------------------------------------------
# AUTHOR: Gargi Bhise
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/
#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

#reading the training data by using Pandas library
df = pd.read_csv('optdigits.tra', sep=',', header=None)
#getting the first 64 fields to create the feature training data and convert them to NumPy array
X_training = np.array(df.values)[:, :64]
#getting the last field to create the class training data and convert them to NumPy array
y_training = np.array(df.values)[:, -1]

#reading the testing data by using Pandas library
df = pd.read_csv('optdigits.tes', sep=',', header=None)
#getting the first 64 fields to create the feature testing data and convert them to NumPy array
X_test = np.array(df.values)[:, :64]
#getting the last field to create the class testing data and convert them to NumPy array
y_test = np.array(df.values)[:, -1]

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape

highest_accuracy = 0
best_parameters = ()

for c_value in c:
    for degree_value in degree:
        for kernel_type in kernel:
            for dfs_type in decision_function_shape:
                
                # Defining the SVM classifier 
                svm_classifier = svm.SVC(C=c_value, degree=degree_value, kernel=kernel_type, decision_function_shape=dfs_type)
                
                # Train the SVM on the training data
                svm_classifier.fit(X_training, y_training)
                
                successful_predictions = 0 
                # Making a prediction for each individual test sample
                for (test_feature, test_target) in zip(X_test, y_test):
                    pred = svm_classifier.predict([test_feature])
                    if pred == test_target: 
                        successful_predictions += 1 

                acc = successful_predictions / len(y_test)
                
                # Verifying if the accuracy is surpassing the highest accuracy so far
                if acc > highest_accuracy:
                    highest_accuracy = acc
                    optimal_parameters = (c_value, degree_value, kernel_type, dfs_type)

print(f"Highest SVM accuracy so far: {highest_accuracy}, Parameters: C={best_parameters[0]}, degree={best_parameters[1]}, kernel={best_parameters[2]}, decision_function_shape={best_parameters[3]}")
