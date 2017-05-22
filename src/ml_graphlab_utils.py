__author__ = 'nadyaK'
__date__ = "04/01/2017"

import graphlab
import string

def set_canvas_target(mode='browser'):
	"""mode: brower, ipynb, headless, none"""
	graphlab.canvas.set_target(mode)

def create_linear_regression(dataset, target, features, l2_penalty=0.,l1_penalty=0., validation_set=None, verbose=False):
	"""Create a LinearRegression to predict a scalar target variable 
	as a linear function of one or more features. 
	More args:  solver='auto', feature_rescaling=True,convergence_threshold=0.01, 
				step_size=1.0,lbfgs_memory_level=11, max_iterations=10 """
	model = graphlab.linear_regression.create(dataset, target, features,
											  l2_penalty=l2_penalty,l1_penalty=l1_penalty,
											  validation_set=validation_set,verbose=verbose)
	return model

def create_logistic_classifier_model(dataset, target, features, validation_set=None, verbose=False): 
	return graphlab.logistic_classifier.create(dataset,target,features, validation_set=validation_set, verbose=verbose)

def create_nearest_neighbors_model(dataset, features, label, distance=None, verbose=False):
	return graphlab.nearest_neighbors.create(dataset,features=features,label=label, distance=distance, verbose=verbose)

def create_popularity_recommender_model(dataset, user_id, item_id, verbose=False):
	return graphlab.popularity_recommender.create(dataset, user_id=user_id, item_id=item_id, verbose=verbose)

def create_similarity_recommender_model(dataset, user_id, item_id, verbose=False):
	return graphlab.item_similarity_recommender.create(dataset, user_id=user_id, item_id=item_id, verbose=verbose)

def get_text_analytics_count(txt_dataset):
	"""Convert the content of string/dict/list type SArrays to a dictionary of
	(word, count) pairs. e.g return [{'quick': 1, 'brown': 1, 'the': 1]"""
	return graphlab.text_analytics.count_words(txt_dataset)

def get_text_analytics_tf_idf(txt_dataset):
	""" Compute the TF-IDF scores for each word in each document. The collection
		of documents must be in bag-of-words format."""
	return graphlab.text_analytics.tf_idf(txt_dataset)

def get_cosine_distance(SFrame_from, SFrame_to):
	"""Compute the cosine distance between between two dictionaries or two
	lists of equal length"""
	return graphlab.distances.cosine(SFrame_from,SFrame_to)

#*******************
# Graphlab helpers *
#*******************
def load_data(path_to_data):
	return graphlab.SFrame(path_to_data)

def split_data(SFrame, percentage, seed=0):
	"""Randomly split the rows The first SFrame contains M rows, 
	sampled uniformly (without replacement) from the original SFrame"""
	train_data, test_data = SFrame.random_split(percentage, seed=seed)
	return train_data, test_data

def transform_column_entry(sframe, col_name, col_entry, trasform_by):
	"""SFrames receive functions and apply to all rows"""
	sframe[col_name] = sframe[col_name].apply(lambda x: trasform_by if x == col_entry else x)
	return sframe

def select_column_from_entry(sframe, col_name, select_entry):
	return sframe[sframe[col_name] == select_entry]

def get_model_coefficients_dict(model):
	"""return a dict: e.g {'power_n': coeff_value}"""
	coefficients = model.get("coefficients")
	return convert_sframe_to_simple_dict(coefficients,'name','value')

def get_nonzero_weights_features(model):
	""" e.g coeff_dict: {'bedroom': 3.0, 'waterfromt:0.0}
		return a list nnz: e.g ['bedroom']""" 
	coeff_dict = get_model_coefficients_dict(model)
	return filter(lambda x: coeff_dict[x] > 0, coeff_dict)

def get_topk_from_sframe(sf, col_name, topk, order_reverse=False):
	"""order_reverse: True (least topk) False (most topk)"""
	return sf.topk(col_name,k=topk,reverse=order_reverse)

def sframe_dict_trim(sf, col_name, significant_words, exclude=False):
	"""e.g  sf = graphlab.SArray([{"this":1, "is":1, "dog":2}, {"and":2....}])
			significant_words = ['is', 'love']
		find and select words that are only in significant_words per dict
        return (["is"])"""
	return sf[col_name].dict_trim_by_keys(significant_words,exclude=exclude)

#*******************
#  Python helpers  *
#*******************
def find_key_max(dict_values):
	return max(dict_values,key=dict_values.get)

def find_key_min(dict_values):
	return min(dict_values,key=dict_values.get)

def convert_sframe_to_simple_dict(sframe, sf_key_column, sf_values_column):
	return dict(zip(list(sframe[sf_key_column]), list(sframe[sf_values_column])))

def remove_punctuation(text):
	return text.translate(None, string.punctuation)

