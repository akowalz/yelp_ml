from sklearn import svm
import json
import pickle
import transformer
import pdb

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

with open('yelp_academic_dataset_business.json', 'r') as f:
	standard_length = 0
	for line in f:
		data = json.loads(line)
		instance = t.transform_instance(data)
		inputs += [instance[0]]
		if len(instance[0]) < standard_length:
			pdb.set_trace()
			print instance
		standard_length = len(instance[0])
		print standard_length
		stars.append(instance[1])
		ratings.append(instance[2])

print len(inputs) # sanity check
print len(stars)

classifier = svm.SVC()
classifier.fit(inputs, stars)
print classifier

