import sklearn
import pickle
import json

pickles_dir = "stars"
yelp_json = "yelp_academic_dataset_business.json"

with open(yelp_json, 'r') as f:
	stars = {}
	for line in f:
		data = json.loads(line)
		if "stars" in data.keys():
			stars[data["business_id"]] = data["stars"]

with open(pickles_dir, 'w') as o:
	pickle.dump(stars, o)
