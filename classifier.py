from sklearn import svm
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

attributes = [
		{
			'name':'city',
			'type':'string',
			'values': pickle.load(open("pickles/cities", 'r')),
			'enabled': True
		},
		{
			'name':'Alcohol',
			'type':'string',
			'values':['full_bar', 'none', 'beer_and_wine'],
			'default': 'none',
			'enabled': True
		},
		{
			'name':'Has TV',
			'type':'bool',
			'default': False,
			'enabled': True
		},
		{
			'name':'Ambience',
			'type':'dict',
			'values': ['romantic', 'intimate', 'classy', 'hipster', 'divey', 'touristy', 'trendy', 'upscale', 'casual'],
			'default': {'romantic':False, 'intimate':False,'classy':False,
						'hipster':False, 'divey':False, 'touristy':False,
						'trendy':False, 'upscale':False, 'casual':True},
			'enabled': True
		},
		{
			'name':'Noise Level',
			'type':'string',
			'values':['average','loud','quiet','very_loud'],
			'default':'average',
			'enabled': True
		},
		{
			'name':'Good for Kids',
			'type':'bool',
			'default':True,  # ehhhh
			'enabled':True
		},
		{
			'name':'Attire',
			'type':'string',
			'values':['casual','dressy','formal'],
			'default':'casual',
			'enabled': True
		},
		{
			'name':'Delivery',
			'type':'bool',
			'default': False,
			'enabled': True
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
			'enabled': True
		},
		{
			'name':'Takes Reservations',
			'type':'bool',
			'default':False,  # ????
			'enabled': True
		},
		{
			'name':'Waiter Service',
			'type':'bool',
			'default':False,  # ????
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
			'enabled': True
		},
		{
			'name':'Good For',
			'type':'dict',
			'values':['dessert', 'latenight', 'lunch', 'dinner','breakfast', 'brunch'],
			'default':{'dessert':False,'latenight':False,'lunch':True,'dinner':True,'breakfast':False},
			'enabled': True
		},
		{
			'name':'Good For Groups',
			'type':'bool',
			'default':True,
			'enabled':True
		},
		{
			'name':'Price Range',
			'type':'int',
			'default':2,
			'enabled':True
		}

		# TODO:
		# - city
		# - category
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

	print('---Making Linear SVC---')
	lsvc = svm.LinearSVC()
	lsvc.fit(inputs, stars)

	correct_count = 0
	for i in range(len(inputs)):
		correct_star = '{:.1f}'.format(stars[i])
		predicted_star = '{:.1f}'.format(lsvc.predict(inputs[i])[0])
		print 'Predicted {}, actual {}.'.format(predicted_star, correct_star)
		if correct_star == predicted_star:
			correct_count += 1

	print '{} correct out of {}, {}%'.format(correct_count, len(inputs), (float(correct_count)/float(len(inputs)))*100)


# joblib.dump(lsvc, 'lsvc.pkl')

	print('---Making SVC---')
	svc = svm.SVC()
	svc.fit(inputs, stars)
	print(svc)
	joblib.dump(svc, 'svc.pkl')



class Classifier:
	def __init__(self, data_path, attributes, model=svm.SVC(),
			ltype='Classification', output_type='stars',
			input_path=None,output_path=None):
		self.data_path = data_path #json data
		self.ltype = ltype #learning type, classification vs. regression
		self.attributes = attributes #attributes we're using from data

		self.input_path = input_path # if we want to try fitting another algorithm without reloading the data
		self.output_path = output_path
		self.transformer = transformer.Transformer(self.attributes)

		self.inputs = []
		self.outputs = []

		self.star_threshold = 3.5
		self.output_type = output_type

		self.model = model
		self.transform_data()

	def transform_data(self):

		if self.input_path and self.output_path:
			print "Loading transformed data from {} and {}".format(
					self.output_path, self.output_path)
			self.inputs = pickle.load(open(self.input_path,'r'))
			self.outputs = pickle.load(open(self.output_path,'r'))
		else:
			print "Transforming instances in {}, output type is {}".format(self.data_path, self.output_type)
			with open(self.data_path) as f:
				for line in f:
					inst, stars, reviews = self.transformer.transform_instance(json.loads(line))

					self.inputs.append(inst)

					if self.output_type == 'stars':
						stars = self.transform_rating(stars)
						self.outputs.append(stars)
					else:
						self.outputs.append(reviews)

			print "Writing inputs to 'pickles/data/inputs_01.pkl'"

			with open("pickles/data/inputs_01.pkl", 'w') as f:
				pickle.dump(self.inputs, f)
			print "Writing outputs to 'pickles/data/outputs_01.pkl'"
			with open("pickles/data/outputs_01.pkl", 'w') as f:
				pickle.dump(self.outputs, f)

		print "Finished.  There are {} instances and {} outputs".format(len(self.inputs), len(self.outputs))

	def transform_rating(self, stars):
		if self.ltype == "Regression":
			return stars
		elif self.ltype == "Classification":
			return stars > self.star_threshold
		else:
			raise InputError(self.ltype, "Invalid learning type")

	def test_train_split(self,test_size=0.3):
		print "Splitting data for testing/training"
		self.x_train, self.x_test, self.y_train, self.y_test = cross_validation.train_test_split(
				self.inputs, self.outputs, test_size=test_size, random_state=0)

	def fit_model(self):
		print "Fitting model"
		self.model.fit(self.x_train, self.y_train)
		print "Model fit"

	def score(self):
		scores = self.model.score(self.x_test, self.y_test)
		print "Scores based on current test set: {}".format(scores)
		return scores

	def save_model(self, path):
		print "Writing model to {}".format(path)
		joblib.dump(self.model, path)

	def load_model(self, path):
		print "Loading model from {}".format(path)
		joblib.load(path)


c = Classifier('restaurant_listings_dense.json', attributes, svm.SVC(kernel='linear'),
		input_path='pickles/data/inputs_01.pkl', output_path='pickles/data/outputs_01.pkl')
c.test_train_split(test_size=0.3)
c.load_model("pickles/models/svc_02.pkl_11.npy")

c.score()
