import json

with open("yelp_academic_dataset_business.json", "r") as f:
	prices = {} 
	for line in f:
		data = json.loads(line)
		attributes = data['attributes']
		if 'Price Range' in attributes:
			if attributes['Price Range'] in prices:
				prices[attributes['Price Range']] += 1
			else:
				prices[attributes['Price Range']] = 1
	
	print prices
