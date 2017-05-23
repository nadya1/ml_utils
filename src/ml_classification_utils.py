# coding=utf-8
__author__ = 'nadyaK'
__date__ = '05/12/2017'

import ml_numpy_utils as np_utils

class LogisticRegression(object):
	"""simple linear regression model """

	def __init__(self):
		self.__name__ = "logistic_regression_model"

	def _check_iteration(self, itr):
		check_itr = False
		if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) or (
						itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
			check_itr = True
		return check_itr

	def _check_log_likelihood(self, itr, feature_matrix, sentiment, coefficients, max_iter):
		if self._check_iteration(itr):
			lp = np_utils.compute_log_likelihood(feature_matrix,sentiment, coefficients)
			print 'iteration %*d: log likelihood of observed labels = %.8f' % (
				int(np_utils.np.ceil(np_utils.np.log10(max_iter))),itr,lp)

	def logistic_regression(self, feature_matrix, sentiment, initial_coefficients,
							step_size, max_iter, check_likelihood=True):
		coefficients = np_utils.np.array(initial_coefficients) # make sure it's a numpy array
		for itr in xrange(max_iter):

			# Predict P(y_i = +1|x_i,w)
			predictions = np_utils.predict_probability(feature_matrix,coefficients)

			# Compute indicator value for (y_i = +1)
			indicator = (sentiment == +1) #otherwise false

			# Compute the errors as indicator - predictions
			errors = indicator - predictions

			for j in xrange(len(coefficients)): # loop over each coefficient
				#feature_matrix[:,j] is column associated with coefficients[j].
				# Compute the derivative for coefficients[j].
				feature = feature_matrix[:,j]
				derivative = np_utils.feature_derivative(errors,feature)

				# add the step size times the derivative to the current coefficient
				coefficients[j] += step_size * derivative

			if check_likelihood:
				self._check_log_likelihood(itr, feature_matrix, sentiment, coefficients, max_iter)

		return coefficients

	def _check_log_likelihood_with_L2(self, itr, feature_matrix, sentiment, coefficients, max_iter, l2_penalty):
		if self._check_iteration(itr):
			lp = np_utils.compute_log_likelihood_with_L2(feature_matrix,sentiment, coefficients, l2_penalty)
			print 'iteration %*d: log likelihood of observed labels = %.8f' % (
				int(np_utils.np.ceil(np_utils.np.log10(max_iter))),itr,lp)

	def logistic_regression_with_L2(self, feature_matrix, sentiment, initial_coefficients,
									step_size, l2_penalty, max_iter, check_likelihood=True):
		coefficients = np_utils.np.array(initial_coefficients) # make sure it's a numpy array

		for itr in xrange(max_iter):
			# Predict P(y_i = +1|x_i,w) using your predict_probability() function
			predictions = np_utils.predict_probability(feature_matrix,coefficients)

			# Compute indicator value for (y_i = +1)
			indicator = (sentiment == +1)

			# Compute the errors as indicator - predictions
			errors = indicator - predictions
			for j in xrange(len(coefficients)): # loop over each coefficient
				#is_intercept = (j == 0)
				is_constant = True if j == 0 else False
				# Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
				# Compute the derivative for coefficients[j]. Save it in a variable called derivative
				feature = feature_matrix[:,j]
				derivative = np_utils.feature_derivative_with_L2(errors,feature,coefficients[j],l2_penalty,is_constant)
				# add the step size times the derivative to the current coefficient
				coefficients[j] += step_size * derivative

			if check_likelihood:
				self._check_log_likelihood_with_L2(itr, feature_matrix, sentiment, coefficients, max_iter, l2_penalty)

		return coefficients
#==================================================================
#                   Helper Functions
#==================================================================
def get_model_classification_accuracy(model,data,true_labels):
	""" e.g model = sentiment_model,
		    data = test_data,
		    labels = test_data['sentiment'] (+1,-1,-1....)
		accuracy = # correctly classified examples / total examples"""

	# First get the predictions
	predictions = model.predict(data)
	# Compute the number of correctly classified examples
	corrected_classified_labels = (predictions == true_labels).sum()
	# Then compute accuracy by dividing num_correct by total number of examples
	total_examples = float(data.num_rows())
	accuracy = corrected_classified_labels / total_examples

	return accuracy

def get_majority_class_accuracy(model, data_set):
	""" majority class -> sentiment == +1 (positive classify)"""
	majority_class = (model.predict(data_set) == +1).sum()
	n_total_data = float(data_set.num_rows())
	accuracy = majority_class/n_total_data
	return accuracy

def get_classification_accuracy(feature_matrix,sentiment,coefficients):
	scores = np_utils.np.dot(feature_matrix,coefficients)
	apply_threshold = np_utils.np.vectorize(lambda x:1. if x > 0  else -1.)
	predictions = apply_threshold(scores)
	num_correct = (predictions == sentiment).sum()
	accuracy = num_correct / float(len(feature_matrix))
	return accuracy