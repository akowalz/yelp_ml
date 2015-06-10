from sklearn import svm
#import matplotlib.pyplot as plt
from sklearn import grid_search
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn import tree
from sklearn.externals import joblib
from sklearn import cross_validation
from pprint import pprint
import json
import os.path
import pickle
import transformer
import pdb
import numpy as np
from sys import argv
from random import shuffle

attributes = [
		{
			'name':'categories',
			'type':'list',
			'values': pickle.load(open("pickles/categories", "r")) + ["none"],
			'enabled':True
		},
		{
			'name':'city',
			'type':'string',
			'values': pickle.load(open("pickles/cities", 'r')),
			'enabled': False
		},
		{
			'name':'Alcohol',
			'type':'string',
			'values':['full_bar', 'none', 'beer_and_wine'],
			'default': 'none',
			'enabled': False
		},
		{
			'name':'Has TV',
			'type':'bool',
			'default': False,
			'enabled': False
		},
		{
			'name':'Ambience',
			'type':'dict',
			'values': ['romantic', 'intimate', 'classy', 'hipster', 'divey', 'touristy', 'trendy', 'upscale', 'casual'],
			'default': {'romantic':False, 'intimate':False,'classy':False,
						'hipster':False, 'divey':False, 'touristy':False,
						'trendy':False, 'upscale':False, 'casual':True},
			'enabled': False
		},
		{
			'name':'Noise Level',
			'type':'string',
			'values':['average','loud','quiet','very_loud'],
			'default':'average',
			'enabled': False
		},
		{
			'name':'Good for Kids',
			'type':'bool',
			'default':True,  # ehhhh
			'enabled':False
		},
		{
			'name':'Attire',
			'type':'string',
			'values':['casual','dressy','formal'],
			'default':'casual',
			'enabled': False
		},
		{
			'name':'Delivery',
			'type':'bool',
			'default': False,
			'enabled': False
		},
		{
			'name':'Accepts Credit Cards',
			'type':'bool',
			'default':True,
			'enabled': True
		},
		{
			'name':'Outdoor Seating',
			'type':'bool',
			'default':False,
			'enabled': False
		},
		{
			'name':'Takes Reservations',
			'type':'bool',
			'default':False,  # ????
			'enabled': False
		},
		{
			'name':'Waiter Service',
			'type':'bool',
			'default':True,  # ????
			'enabled': True
		},
		{
			'name':'Wi-Fi',
			'type':'string',
			'values':['no','free','paid'],
			'default':'no',
			'enabled': True
		},
		{
			'name':'Parking',
			'type':'dict',
			'values':['garage', 'street', 'validated', 'lot', 'valet'],
			'default':{'garage':False,'street':True,'validated':False,'lot':False,'valet':False},
			'enabled': False
		},
		{
			'name':'Good For',
			'type':'dict',
			'values':['dessert', 'latenight', 'lunch', 'dinner','breakfast', 'brunch'],
			'default':{'dessert':False,'latenight':False,'lunch':True,'dinner':True,'breakfast':False},
			'enabled': False
		},
		{
			'name':'Good For Groups',
			'type':'bool',
			'default':True,
			'enabled':False
		},
		{
			'name':'Price Range',
			'type':'int',
			'default':2,
			'enabled':True
		}
    ]

def attribute_density_count(attributes):
	"""
	Figures out which instances has the most information, was the basis for the
	function below, as we learned there are ~13K instances with all 16
	attributes
	"""
	enabled_attrs = filter(lambda a: a['enabled'], attributes)
	n_attrs = len(enabled_attrs)

	print "There are {} enabled attributes".format(n_attrs)

	counts = {}
	for i in range(n_attrs+1):
		counts[i] = 0

	with open('restaurant_listings.json', 'r') as f:
		for line in f:
			instance = json.loads(line)
			n_instance_attrs = 0
			for attr in enabled_attrs:
				if attr['name'] in instance['attributes'] or attr['name'] in instance:
					n_instance_attrs += 1

			counts[n_instance_attrs] +=1

	return counts

def get_dense_instances(attributes):
	""" RUN ME!!!!! """
	enabled_attrs = filter(lambda a: a['enabled'], attributes)
	n_attrs = len(enabled_attrs)
	n_dense_instances = 0

	with open('restaurant_listings.json', 'r') as f:
		with open('restaurant_listings_dense.json', 'w') as wf:
			for line in f:
				instance = json.loads(line)
				n_instance_attrs = 0
				for attr in enabled_attrs:
					if attr['name'] in instance['attributes'] or attr['name'] in instance:
						n_instance_attrs += 1
				if n_instance_attrs == n_attrs:
					n_dense_instances += 1
					print "Writing dense instance"
					wf.write(line)

		print "Wrote {} instances".format(n_dense_instances)


def classify():
	t = transformer.Transformer(attributes)

	inputs = []
	stars = []
	ratings = []

	if not os.path.isfile('pickles/inputs') or '--dry-run' in argv:
		print('Existing transformed data not found. Transforming.')
		with open('restaurant_listings.json', 'r') as f:
			for line in f:
				data = json.loads(line)
				instance = t.transform_instance(data)
				inputs += [instance[0]]
				for i in instance[0]:
					try:
						np.isnan(i)
					except TypeError:
						pdb.set_trace()
				stars.append(instance[1])
				ratings.append(instance[2])

		with open('inputs', 'w') as f:
			pickle.dump(inputs, f)

		with open('stars', 'w') as f:
			pickle.dump(stars, f)

		with open('ratings', 'w') as f:
			pickle.dump(ratings, f)
	else:
		print('Loading transformed data from file.')
		with open('pickles/inputs', 'r') as f:
			inputs = pickle.load(f)
		with open('pickles/ratings_inputs', 'r') as f:
			ratings = pickle.load(f)
		with open('pickles/star_inputs', 'r') as f:
			stars = pickle.load(f)

	# print('---Making Linear SVC---')
	# lsvc = svm.LinearSVC()
	# lsvc.fit(inputs, stars)

	# correct_count = 0
	# for i in range(len(inputs)):
	# 	correct_star = '{:.1f}'.format(stars[i])
	# 	predicted_star = '{:.1f}'.format(lsvc.predict(inputs[i])[0])
	# 	print 'Predicted {}, actual {}.'.format(predicted_star, correct_star)
	# 	if correct_star == predicted_star:
	# 		correct_count += 1

	# print '{} correct out of {}, {}%'.format(correct_count, len(inputs), (float(correct_count)/float(len(inputs)))*100)


# joblib.dump(lsvc, 'lsvc.pkl')

	# print('---Making SVC---')
	# svc = svm.SVC()
	# svc.fit(inputs, stars)
	# print(svc)
	# joblib.dump(svc, 'svc.pkl')



class Classifier:
	def __init__(self, data_path, attributes, model=svm.SVC(),
			ltype='Classification', output_type='stars',
			load_data_path='', dry_run=False):
		self.data_path = data_path #json data
		self.ltype = ltype #learning type, classification vs. regression
		self.attributes = attributes #attributes we're using from data

		self.load_data_path = load_data_path
		self.dry_run = dry_run
		self.transformer = transformer.Transformer(self.attributes)

		self.inputs = []
		self.outputs = []

		self.star_threshold = 3.5
		self.output_type = output_type

		self.model = model
		self.transform_data()

	def transform_data(self):

		if self.load_data_path and not self.dry_run:
			print "Loading transformed data from {}".format(self.load_data_path)
			self.inputs, self.outputs = pickle.load(open(self.load_data_path,'r'))
		else:
			print "Transforming instances in {}, output type is {}".format(self.data_path, self.output_type)
			print "Using attributes:"
			for attr in self.attributes:
				if attr['enabled']:
					print attr['name']

			with open(self.data_path) as f:
				for line in f:
					raw_instance = json.loads(line)
					if not self.instance_has_all_enabled_attributes(raw_instance):
						continue
					inst, stars, reviews = self.transformer.transform_instance(raw_instance)

					self.inputs.append(inst)

					if self.output_type == 'stars':
						stars = self.transform_rating(stars)
						self.outputs.append(stars)
					else:
						self.outputs.append(reviews)

			write_path = "pickles/data/instances.pkl"
			print "Writing transformed instances to ", write_path

			with open(write_path, 'w') as f:
				pickle.dump((self.inputs, self.outputs), f)

		print "Finished.  There are {} instances and {} outputs".format(len(self.inputs), len(self.outputs))

	def instance_has_all_enabled_attributes(self, instance):
		for attr in self.attributes:
			if attr['enabled']:
				if not attr['name'] in instance and not attr['name'] in instance['attributes']:
					return False
		return True

	def transform_rating(self, stars):
		if self.ltype == "Regression":
			return stars
		elif self.ltype == "Classification":
			return stars > self.star_threshold
		else:
			raise Exception(self.ltype, "Invalid learning type")

	def test_train_split(self,test_size=0.3):
		print "Splitting data for testing/training"
		self.x_train, self.x_test, self.y_train, self.y_test = cross_validation.train_test_split(
				self.inputs, self.outputs, test_size=test_size, random_state=0)

	def fit_model_on_training_set(self):
		print "Fitting model on {} samples".format(len(self.x_train))
		self.model.fit(self.x_train, self.y_train)
		print "Model fit"

	def fit_model(self):
		print "Fitting model on {} samples".format(len(self.inputs))
		self.model.fit(self.inputs, self.outputs)
		print "Model fit"

	def score(self):
		scores = self.model.score(self.x_test, self.y_test)
		print "Scores based on current test set: {}".format(scores)
		return scores

	def cross_validate(self, n_folds=5):
		print "Beginnning {}-fold cross validation".format(n_folds)
		scores = cross_validation.cross_val_score(
				self.model, self.inputs, self.outputs, cv=n_folds)
		print "Finished cross validation, results:"
		print scores
		print "Average accuracy: {}".format(sum(scores)/float(len(scores)))
		zero_class_scores = self.zero_classifier()
		print "The zero classifier would predict with {} accuracy".format(zero_class_scores)

		return sum(scores)/float(len(scores))

	def save_model(self, path):
		print "Writing model to {}".format(path)
		joblib.dump(self.model, path)

	def load_model(self, path):
		print "Loading model from {}".format(path)
		joblib.load(path)

	def zero_classifier(self):
		pos_class = 0
		neg_class = 0
		for output in self.outputs:
			if output:
				pos_class += 1
			else:
				neg_class += 1

		more_probable_class = pos_class if (pos_class > neg_class) else neg_class

		return more_probable_class/float(len(self.outputs))

	def choose_params(self):
		param_grid = [
				  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
					{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
						 ]
		self.model = grid_search.GridSearchCV(self.model, param_grid)
		joblib.dump(self.model, 'gs_model.pkl')
		self.cross_validate()

	def make_graph(self):
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel("Training examples")
		plt.ylabel("Score")
		train_sizes, train_scores, test_scores = learning_curve(
			estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.grid()

		plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
						 train_scores_mean + train_scores_std, alpha=0.1,
						 color="r")
		plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
						 test_scores_mean + test_scores_std, alpha=0.1, color="g")
		plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
				 label="Training score")
		plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
				 label="Cross-validation score")

		plt.legend(loc="best")
		return plt

c = Classifier(
		'restaurant_listings.json',
		attributes,
		tree.DecisionTreeClassifier(),
		load_data_path='pickles/data/instances.pkl',
		dry_run=False)

c.cross_validate(5)

def find_best_attributes(attributes):
	print "disabling all attributes"
	for attr in attributes:
		attr['enabled'] = False

	best_score = float('-inf')

	shuffle(attributes)
	for attr in attributes:
		print "Trying ", attr['name']
		attr['enabled'] = True

		c = Classifier(
				'restaurant_listings.json',
				attributes,
				tree.DecisionTreeClassifier(),
				load_data_path='pickles/data/instances.pkl',
				dry_run=True)

		zero_score = c.zero_classifier()
		cv_score = c.cross_validate(10)

		score = cv_score - zero_score
		if score > best_score:
			print "{} HELPS >>>>>>".format(attr['name'])
			best_score = score
		else:
			print "{} DOES NOT HELP <<<<<<".format(attr['name'])
			attr['enabled'] = False

	print "Final contenders:"
	for attr in attributes:
		if attr['enabled']:
			print attr['name']

	return attributes

find_best_attributes(attributes)
