# coding=utf-8
__author__ = 'nadyaK'
__date__ = '04/09/2017'

import numpy as np

def compute_RSS(predictions,output):
	"""Residual Sum of Squares (RSS)"""
	residuals = predictions - output
	RSS = (residuals ** 2).sum()
	return (RSS)

def get_numpy_data(data_sframe, features, output, constant='constant'):
	"""Convert SFrame to Numpy Array"""

	#Add constant
	data_sframe[constant] = 1  #constant column (to  create an 'intercept')
	features = [constant] + features  # add the column 'constant' to the front of the features list

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

def normalize_features(feature_matrix, axis=0):
	norms = np.linalg.norm(feature_matrix, axis=axis)
	normalized_features = feature_matrix / norms
	return (normalized_features, norms)

def get_normalized_data(dataset,features,target):
	feature_matrix,output = get_numpy_data(dataset,features,target)
	feature_matrix_norm,norms = normalize_features(feature_matrix)
	return feature_matrix_norm, output, norms

def get_nonzero_weights(weights):
	weights_nnz = np.array(weights).nonzero()
	return weights[weights_nnz]

def get_euclidean_distance(source, query):
	"""Euclidean norm is the L2 norm or L2 distance."""
	return np.sqrt(np.sum((source - query) ** 2))

def get_euclidean_distance_matrix(source, matrix_query, axis=1):
	"""Euclidean norm is the L2 norm  (np.matrix-vectors)
	   return a list-distances. (np.argmin to find min-idx-distance )"""
	return np.sqrt(np.sum((source - matrix_query) ** 2, axis=axis))

def find_k_nearest_neighbors(k, source, matrix_query, axis=1):
	"""returns the indices of the k closest training sorce."""
	return np.argsort(get_euclidean_distance_matrix(source, matrix_query, axis))[:k]

def single_prediction_k_nearest_neighbors(k, source, matrix_query, output_values):
	""" extract multiple values using a list of indices 
		e.g: output_train[[6, 10]] """
	k_nearest = find_k_nearest_neighbors(k, source, matrix_query)
	return np.average(output_values[k_nearest])

def probability_predictions(scores):
	""" Sigmoid function
	    P(yi=+1|xi,w)= 1/(1+exp(−wTh(xi))"""
	# return map(lambda xi: 1/float(1+math.exp(-xi)),scores)
	predictions = 1 / (1. + np.exp(-scores)) #exp (takes np.arrays)
	return predictions

def compute_probabilities(features, weights):
	""" h(x) = intercept + slope*x --> w0x0 + w1xi1 + w2xi2 ... 
		e.g: x0 = [1,1,1,1] #intercept, x1 = [2,0,3,4] ...
	    features = np.array([x0,x1,x2])
	    weights = np.array([0,1,-2])
	    return np.array([0.05 0.02 0.05 0.88])
	"""
	probabilities = []
	for idx in range(len(features[0])):
		hx = np.dot(features[:,idx],weights) # #extrace rows per features
		probabilities.append(probability_predictions(hx))  # sigmoid function
	return np.array(probabilities)

def compute_derivative_for_wi(feature_i, output, probabilities):
	"""Contribution to derivative for w1 """
	output = np.array(map(lambda y: 0 if y <=0 else 1, output))
	return (feature_i * (output-probabilities)).sum()

def compute_accuracy(n_correct, n_total):
	""" accuracy = # correctly classified examples / total examples"""
	return n_correct/n_total

