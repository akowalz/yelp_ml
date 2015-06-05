from sklearn import svm
import json
import os.path
import pickle
import transformer
import pdb
from sklearn.externals import joblib
import numpy as np

attributes = [
		{
			'name':'Alcohol',
			'type':'string',
			'values':['full_bar', 'none', 'beer_and_wine'],
			'default': 'none'
		},
		{
			'name':'Has TV',
			'type':'bool',
			'default': False
		},
		{
			'name':'Ambience',
			'type':'dict',
			'values': ['romantic', 'intimate', 'classy', 'hipster', 'divey', 'touristy', 'trendy', 'upscale', 'casual'],
			'default': False
		},
		{
			'name':'Noise Level',
			'type':'string',
			'values':['average','loud','quiet','very_loud'],
			'default':'average'
		},
		{
			'name':'Good for Kids',
			'type':'bool',
			'default':True  # ehhhh
		},
		{
			'name':'Attire',
			'type':'string',
			'values':['casual','dressy','formal'],
			'default':'casual'
		},
		{
			'name':'Delivery',
			'type':'bool',
			'default': False
		},
		{
			'name':'Accepts Credit Cards',
			'type':'bool',
			'default':True
		},
		{
			'name':'Outdoor Seating',
			'type':'bool',
			'default':False
		},
		{
			'name':'Takes Reservations',
			'type':'bool',
			'default':False  # ????
		},
		{
			'name':'Waiter Service',
			'type':'bool',
			'default':False  # ????
		},
		{
			'name':'Wi-Fi',
			'type':'string',
			'values':['no','free','paid'],
			'default':'no'
		},
		{
			'name':'Parking',
			'type':'dict',
			'values':['garage', 'street', 'validated', 'lot', 'valet']
		},
		{
			'name':'Good For',
			'type':'dict',
			'values':['dessert', 'latenight', 'lunch', 'dinner','breakfast', 'brunch']
		},
		{
			'name':'Good For Groups',
			'type':'bool',
			'default':True
		}, 
		{
			'name':'Price Range',
			'type':'int',
			'default':2
		}

		# TODO:
		# - city
		# - category
    ]

t = transformer.Transformer(attributes)


inputs = []
stars = []
ratings = []

if not os.path.isfile('pickles/inputs'):
	print('Existing transformed data not found. Transforming.')
	with open('yelp_academic_dataset_business.json', 'r') as f:
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

