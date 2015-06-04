import json
import pickle

def get_ambiences(outpath):
    with open("yelp_academic_dataset_business.json") as f:
        amb = []
        for line in f:
            data = json.loads(line)
            if "Ambience" in data["attributes"].keys():
                for c in data["attributes"]["Ambience"]:
                    if c not in amb:
                        amb.append(c)

    with open(outpath, "w") as f:
        pickle.dump(amb, f)

def get_toplevel_key(key, outpath):
    with open("yelp_academic_dataset_business.json") as f:
        values = []
        for line in f:
            data = json.loads(line)
            if data[key] not in values:
                values.append(data[key])

    with open(outpath, "w") as f:
        pickle.dump(values, f)

def get_toplevel_array_values(key, outpath):
    """ Like the other one, but works for array valued keys """
    with open("yelp_academic_dataset_business.json") as f:
        values = []
        for line in f:
            data = json.loads(line)
            for item in data[key]:
                if item not in values:
                    values.append(item)

    with open(outpath, "w") as f:
        pickle.dump(values, f)

def get_good_for(outpath):
    with open("yelp_academic_dataset_business.json") as f:
        amb = []
        for line in f:
            data = json.loads(line)
            if "Good For" in data["attributes"].keys():
                for c in data["attributes"]["Good For"]:
                    if c not in amb:
                        amb.append(c)

    with open(outpath, "w") as f:
        pickle.dump(amb, f)


def get_all_attributes_and_values(outpath):
    values = {}
    with open("yelp_academic_dataset_business.json") as f:
        for line in f:
					data = json.loads(line)
					attrs = data["attributes"]
					cats = ["Food", "Restaurants"]
					if in_category(data["categories"], cats):
						for attr, val in attrs.iteritems():
								if attr in values.keys():
										values[attr]["count"] += 1
										if val not in values[attr]["values"] and not isinstance(val, dict):
												values[attr]["values"].append(val)
								else:
										values[attr] = {
														"count" : 1,
														"values": []
														}
										if isinstance(val, dict):
												values[attr]["values"] = val.keys()
										else:
												values[attr]["values"].append(val)


    for k,v in values.iteritems():
        print "{} : {}".format(k,v)

    with open(outpath, "w") as f:
        pickle.dump(values, f)

def in_category(all_cats, subset):
	all_cats = [cat.lower() for cat in all_cats]
	subset = [cat.lower() for cat in subset]
	for cat in subset:
		if cat in all_cats:
			return True


get_all_attributes_and_values("food_attributes")
