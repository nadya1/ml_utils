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

class DecisionTree(object):
	"""Decision Tree model """

	def __init__(self):
		self.__name__ = "decision_tree_model"

	def intermediate_node_num_mistakes(self, labels_in_node):
		# Corner case: If labels_in_node is empty, return 0
		if len(labels_in_node) == 0:
			return 0

		# Count the number of 1's (safe loans)
		positives = len(labels_in_node[labels_in_node == 1])   #(filter(lambda x: x == 1)).sum()

		# Count the number of -1's (risky loans)
		negatives = len(labels_in_node[labels_in_node == -1])

		# Return the number of mistakes that the majority classifier makes.
		majority = negatives if positives > negatives else positives

		return majority

	def best_splitting_feature(self, data, features, target):
		best_feature = None # Keep track of the best feature
		best_error = 10     # Keep track of the best error so far
		# Note: Since error is always <= 1, we should intialize it with something larger than 1.

		# Convert to float to make sure error gets computed correctly.
		num_data_points = float(len(data))

		# Loop through each feature to consider splitting on that feature
		for feature in features:
			# The left split will have all data points where the feature value is 0
			left_split = data[data[feature] == 0]

			# The right split will have all data points where the feature value is 1
			right_split = data[data[feature] == 1]

			# Calculate the number of misclassified examples in the left split.
			# Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
			left_mistakes = self.intermediate_node_num_mistakes(left_split[target]) #target has all 1|-1  ....

			# Calculate the number of misclassified examples in the right split.
			right_mistakes = self.intermediate_node_num_mistakes(right_split[target])

			# Compute the classification error of this split.
			# Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
			error = (left_mistakes + right_mistakes) / num_data_points

			# If this is the best error we have found so far, store the feature as best_feature and the error as best_error
			if error < best_error:
				best_feature = feature
				best_error = error

		return best_feature # Return the best feature we found

	def create_leaf(self, target_values):
		# Create a leaf node
		leaf = {'splitting_feature':None,'left':None,'right':None,'is_leaf':True}

		# Count the number of data points that are +1 and -1 in this node.
		num_ones = len(target_values[target_values == +1])
		num_minus_ones = len(target_values[target_values == -1])

		# For the leaf node, set the prediction to be the majority class.
		# Store the predicted class (1 or -1) in leaf['prediction']
		leaf['prediction'] = 1 if num_ones > num_minus_ones else -1

		# Return the leaf node
		return leaf

	def decision_tree_create(self, data,features,target,current_depth=0,max_depth=10, verbose=True):
		remaining_features = features[:] # Make a copy of the features.

		target_values = data[target]
		if verbose:
			print "--------------------------------------------------------------------"
			print "Subtree, depth = %s (%s data points)." % (current_depth,len(target_values))

		# Stopping condition 1
		# (Check if there are mistakes at current node.
		if self.intermediate_node_num_mistakes(target_values) == 0:
			if verbose: print "Stopping condition 1 reached."
			# If not mistakes at current node, make current node a leaf node
			return self.create_leaf(target_values)

		# Stopping condition 2
		# (check if there are remaining features to consider splitting on)
		if remaining_features == []:
			if verbose: print "Stopping condition 2 reached."
			# If there are no remaining features to consider, make current node a leaf node
			return self.create_leaf(target_values)

		# Additional stopping condition (limit tree depth)
		if current_depth >= max_depth:
			if verbose: print "Reached maximum depth. Stopping for now."
			# If the max tree depth has been reached, make current node a leaf node
			return self.create_leaf(target_values)

		# Find the best splitting feature (recall the function best_splitting_feature implemented above)
		splitting_feature = self.best_splitting_feature(data,features,target)

		# Split on the best feature that we found.
		left_split = data[data[splitting_feature] == 0]
		right_split = data[data[splitting_feature] == 1]
		remaining_features.remove(splitting_feature)
		if verbose: print "Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split))

		# Create a leaf node if the split is "perfect"
		if len(left_split) == len(data):
			if verbose: print "Creating leaf node."
			return self.create_leaf(left_split[target])

		if len(right_split) == len(data):
			if verbose: print "Creating leaf node."
			return self.create_leaf(right_split[target])

		# Repeat (recurse) on left and right subtrees
		left_tree = self.decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth, verbose)

		right_tree = self.decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth, verbose)

		leaf = {'is_leaf':False,'prediction':None,
				'splitting_feature':splitting_feature,
				'left':left_tree,'right':right_tree,}

		return leaf

	def count_nodes(self, tree):
		if tree['is_leaf']:
			return 1
		return 1 + self.count_nodes(tree['left']) + self.count_nodes(tree['right'])

	def classify(self, tree, x, verbose=False):
		# if the node is a leaf node.
		if tree['is_leaf']:
			if verbose:
				print "At leaf, predicting %s" % tree['prediction']
			return tree['prediction']
		else:
			# split on feature.
			split_feature_value = x[tree['splitting_feature']]
			if verbose:
				print "Split on %s = %s" % (tree['splitting_feature'],split_feature_value)
			if split_feature_value == 0:
				#print "--> left"
				return self.classify(tree['left'],x,verbose)
			else:
				# print "--> right"
				return self.classify(tree['right'],x,verbose)

	def evaluate_classification_error(self, tree, data, target):
		# Apply the classify(tree, x) to each row in your data
		prediction = data.apply(lambda x:self.classify(tree,x))

		# Once you've made the predictions, calculate the classification error and return it
		n_data = float(len(data))
		num_correct = (prediction == data[target]).sum()
		num_mistakes = n_data - num_correct

		classification_error = num_mistakes / n_data

		return classification_error

	def print_stump(self, tree,name='root'):
		split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
		if split_name is None:
			print "(leaf, label: %s)" % tree['prediction']
			return None
		split_feature,split_value = split_name.split('.')
		print '                       %s' % name
		print '         |---------------|----------------|'
		print '         |                                |'
		print '         |                                |'
		print '         |                                |'
		print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
		print '         |                                |'
		print '         |                                |'
		print '         |                                |'
		print '    (%s)                         (%s)' % (
		('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
		('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

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

