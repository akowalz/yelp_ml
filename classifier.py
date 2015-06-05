from sklearn import svm
from sklearn.externals import joblib
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

get_dense_instances(attributes)
