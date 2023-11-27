#-------------------------------------------------------------------------
# AUTHOR: Gargi Bhise
# FOR: CS 4210- Assignment #4
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

highest_accuracy = {'Perceptron': 0, 'MLP': 0}
best_params = {'Perceptron': {}, 'MLP': {}}

for learning_rate in n: #iterates over n

    for shuffle in r: #iterates over r

        #iterates over both algorithms

        for algorithm in ['Perceptron', 'MLP']: #iterates over the algorithms

            #Create a Neural Network classifier

            if algorithm == 'Perceptron':
                clf = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, hidden_layer_sizes=(25,), shuffle=shuffle, max_iter=1000)

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            if accuracy > highest_accuracy[algorithm]:
                highest_accuracy[algorithm] = accuracy
                best_params[algorithm] = {'learning_rate': learning_rate, 'shuffle': shuffle}
                print(f'Highest {algorithm} accuracy so far: {accuracy}, Parameters: learning rate={learning_rate}, shuffle={shuffle}')


