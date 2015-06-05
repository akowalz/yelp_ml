import json

categories = ['food', 'restaurant', 'restaurants']

outfile = open('restaurant_listings.json', 'w')

with open('yelp_academic_dataset_business.json', 'r') as infile:
	for line in infile:
		data = json.loads(line)
		for category in categories:
			if category in map(lambda x: x.lower(), data['categories']):
				json.dump(data, outfile)
				outfile.write('\n')
				
