import sklearn
import json
import pickle
import transformer

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
         }
    ]

t = transformer.Transformer(attributes)

with open('yelp_academic_dataset_business.json', 'r') as f:
	for line in f:
		data = json.loads(line)
		print t.transform_instance(data)
