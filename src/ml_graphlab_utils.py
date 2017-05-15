__author__ = 'nadyaK'
__date__ = "04/01/2017"

import graphlab

def set_canvas_target(mode='browser'):
	"""mode: brower, ipynb, headless, none"""
	graphlab.canvas.set_target(mode)

def create_linear_regression(dataset, target, features, validation_set=None, verbose=False):
	"""Create a LinearRegression to predict a scalar target variable 
	as a linear function of one or more features. 
	More args:  solver='auto', feature_rescaling=True,convergence_threshold=0.01, 
				step_size=1.0,lbfgs_memory_level=11, max_iterations=10 """ 
	return graphlab.linear_regression.create(dataset, target,features, validation_set=validation_set, verbose=verbose)

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

def convert_sframe_to_simple_dict(sframe, sf_key_column, sf_values_column):
	return dict(zip(list(sframe[sf_key_column]), list(sframe[sf_values_column])))

#*******************
#  Python helpers  *
#*******************
def find_key_max(dict_values):
	return max(dict_values,key=dict_values.get)

def find_key_min(dict_values):
	return min(dict_values,key=dict_values.get)