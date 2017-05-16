__author__ = 'nadyaK'
__date__ = '04/09/2017'
 
from math import sqrt

class SimpleLinearRegression(object):
	"""simple linear regression model """

	def __init__(self):
		self.__name__ = "simple_linear_regression_model"

	def simple_linear_regression(self, input_feature, output):
		"""compute the slope(w1) & intercept(w0) for a simple linear regression """

		# compute the sum of input_feature and output
		sum_yi = (output).sum()
		sum_xi = (input_feature).sum()

		# compute the product of the output and the input_feature and its sum
		sum_xiyi = (output * input_feature).sum()

		# compute the squared value of the input_feature and its sum
		sum_xi2 = (input_feature ** 2).sum()
		N = float(len(input_feature))
		numerator = sum_xiyi - ((sum_yi * sum_xi) / N)

		# use the formula for the slope (w1)
		slope = (numerator) / float(sum_xi2 - ((sum_xi * sum_xi) / N))

		# use the formula for the intercept (w0)
		intercept = sum_yi / N - slope * (sum_xi / N)

		return (intercept,slope)

	def get_regression_predictions(self, input_feature, intercept, slope):
		""" w0:intercept  w1:slope
			yi_predict = (w0 + w1*xi)"""
		return (intercept + slope * input_feature)

	def inverse_regression_predictions(self, output, intercept, slope):
		""" yi_predict = (w0 + w1*xi)
		   inverse:   xi = (yi_predict - w0)/w1 """
		return ((output - intercept) / float(slope))

	def get_residual_sum_of_squares(self, input_feature, output, intercept, slope):
		"""make predictions and evaluate the model using Residual Sum of Squares (RSS)"""
		predicted_values = self.get_regression_predictions(input_feature,intercept,slope)
		return compute_RSS(predicted_values,output)

class GrandientDescent(object):
	"""Gradient Descent model """

	def __init__(self):
		self.__name__ = "gradient_descent_model"
		self.initial_intercept = 0
		self.initial_slope = 0
		self.simple_linear_reg = SimpleLinearRegression()

	def compute_prediction_errors(self, y_real, y_predicted):
		return (y_predicted - y_real)

	def update_intercept(self, derivate_slope, step_size, intercept):
		"""compute the derivative: sum(errors)"""
		adjustment = step_size * derivate_slope
		new_intercept = intercept - adjustment
		return (new_intercept)

	def update_slope(self, derivate, step_size, slope):
		"""compute the derivative: sum(errors*input)"""
		adjustment = step_size * derivate
		new_slope = slope - adjustment
		return new_slope

	def calculate_magnitude(self, derivate_intercept, derivate_slope):
		return sqrt(derivate_intercept**2 + derivate_slope**2)

	def calculate_gradiend_descent(self, x, y, step_size, tolerance):
		""" 1. Compute the predicted values given the current slope and intercept
			2. Compute the prediction errors (prediction - Y)
			3. Update the intercept:
				compute the derivative: sum(errors)
				compute the adjustment as step_size times the derivative
				decrease the intercept by the adjustment
			4. Update the slope:
				compute the derivative: sum(errors*input)
				compute the adjustment as step_size times the derivative
				decrease the slope by the adjustment
			5. Compute the magnitude of the gradient
			6. Check for convergence
		"""
		converged = False
		step = 1

		while not converged:
			# print "\n\nstep: %s"% step
			y_predict = self.simple_linear_reg.get_regression_predictions(x,self.initial_intercept,self.initial_slope)
			# print "1) predictions: %s" % y_predict
			errors = self.compute_prediction_errors(y, y_predict)
			# print "2) predictions: %s" % errors
			derivate_intercept = sum(errors)
			self.initial_intercept = self.update_intercept(derivate_intercept, step_size, self.initial_intercept)
			# print "3) New Intercept: %s" % self.initial_intercept
			derivate_slope = sum(errors*x)
			self.initial_slope = self.update_slope(derivate_slope, step_size, self.initial_slope)
			# print "4) New Slope: %s" %self.initial_slope
			magnitude = self.calculate_magnitude(derivate_intercept, derivate_slope)
			# print "5) Magnitude: %s" %magnitude
			check_if_converged  = not(magnitude > tolerance)
			# print "6) Converge?: %s" % check_if_converged
			if check_if_converged:
				converged = True

			# if step in [1,2,3,78]:
			# 	predicted_y = self.initial_intercept + self.initial_slope*x
			# 	plt.plot(x,y,'r.', x,predicted_y,'g-', [x.mean()], [y.mean()],'b*')
			# 	plt.ylim(0,23)
			# 	# plt.show()
			# 	plt.savefig(os.path.join(self.save_plots, 'step_%s.png'%step))
			step+=1
		return self.initial_intercept, self.initial_slope

#==================================================================
#                   Helper Functions
#==================================================================
def compute_RSS(predictions,output):
	"""Residual Sum of Squares (RSS)"""
	residuals = predictions - output
	RSS = (residuals ** 2).sum()
	return (RSS)

