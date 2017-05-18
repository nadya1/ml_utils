__author__ = 'nadyaK'
__date__ = '04/09/2017'

import numpy as np

def get_numpy_data(data_sframe, features, output):
    """Convert SFrame to Numpy Array"""

    #Add constant
    data_sframe['constant'] = 1  #constant column (to  create an 'intercept')
    features = ['constant'] + features  # add the column 'constant' to the front of the features list

    # select the columns of data_SFrame given by the features list sframe
    features_sframe = data_sframe[features]

    #Convert to numpy-matrix
    feature_matrix = features_sframe.to_numpy() # convert features_SFrame into a numpy matrix:
    output_sarray = data_sframe[output] ## assign the column of data_sframe associated with the output
    output_array = output_sarray.to_numpy() # convert the SArray into a numpy array

    return (feature_matrix, output_array)

def predict_output(feature_matrix, weights):
    """Predicting output given regression weights"""
    predictions = np.dot(feature_matrix, weights)
    return (predictions)

def feature_derivative(errors, feature):
    """Computing derivative: dot product of these vectors(np.arrays) """
    derivative =  2 * np.dot(errors,feature)
    return(derivative)

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    """  If feature_is_constant is True, derivative is 
         twice the dot product of errors and feature"""
    derivative = 2 * np.dot(errors,feature)
    if not feature_is_constant:
        derivative += 2*l2_penalty*weight
    return derivative

def compute_cost_function(errors, weights, l2_penalty):
    """sum data points of the sqd-doff btw observed-vs-predicted + L2 penalty
    Cost(w) = SUM[(prediction - output) ^ 2]  + l2_penalty * (w[0] ^ 2 + ... + w[k] ^ 2).
    applying derivate: 
    Cost(w) = 2*SUM[ error*[feature_i] ] + 2*l2_penalty*w[i].            
    """
    return np.dot(errors,errors) + l2_penalty * (np.dot(weights,weights) - weights[0] ** 2)

def print_coefficients(model):
    """print format: x^n.....x^2"""
    w = list(model.coefficients['value'])
    w.reverse()
    print "\nLearned polynomial:"
    print np.poly1d(w)
    return w