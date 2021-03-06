import pickle
import json
from sklearn.preprocessing import LabelEncoder
import pdb

class Transformer:
	def __init__(self, attributes):
		self.attributes = attributes
		self.encoders = self.build_encoders()

	def build_encoders(self):
		"""
		Creates Label encoders based on possible values and keeps them a dict
		to use for later encoding
		"""
		encoders = {}
		for attr in self.attributes:
			if attr['type'] == 'string' or attr['type'] == 'list':
				le = LabelEncoder()
				le.fit(attr['values'])
				encoders[attr['name']] = le
		return encoders

	def encode_boolean(self, value, attribute_name=None):
		""" turns True of False into 0 or 1, or default value if given """
		if value is None or value == {}:
			attr = self.get_attribute_info(attribute_name)
			if 'default' in attr:
				value = attr['default']
			else:
				return -1
		return int(value)

	def encode_string(self, attribute_name, value):
		"""
		Looks up string and value in attributes hash and returns the label
		encoded integer
		"""
		if value is None:
			attr = self.get_attribute_info(attribute_name)
			if 'default' in attr:
				value = attr['default']
			else:
				return -1
		encoder = self.encoders[attribute_name]
		normalized_label_value = self.normalize_label(encoder, value)
		return normalized_label_value

	def encode_list(self, attr, attr_values):
		""" This is more like 'encode category' because that's the only list we
		have..."""
		ignored_cats = ["food", "restaurants", "restaurant"]
		filtered_categories = filter(lambda c: c.lower() not in ignored_cats, attr_values)

		if len(filtered_categories) == 0:
			category = "none"
		else:
			category = filtered_categories[0]
		return self.encode_string(attr, category)


	def normalize_label(self, encoder, value):
		highest_label_value = len(encoder.classes_)
		label_value = encoder.transform([value])[0]

		return float(label_value)/float(highest_label_value)

	def encode_dict(self, attribute_name, value_dict):
		"""
		Labels values in dicts of booleans (data has a lot of these)
		"""

		output = []

		attr = self.get_attribute_info(attribute_name)
		attr_values = self.get_attribute_values(attribute_name)

		if value_dict is None or value_dict == {}:
			if 'defualt' in attr:
				value_dict = attr['default']
			else:
				return [0] * len(attr_values)
		else:
			for value in attr_values:
				if value in value_dict:
					encoding = self.encode_boolean(value_dict[value],
							attribute_name)
				else:
					encoding = 0
				output.append(encoding)

			return output

	def encode_attribute(self, attr, attr_value):
		attr_type = self.get_attribute_type(attr)
		if attr_type == "string":
			return self.encode_string(attr, attr_value)
		elif attr_type == "bool":
			return self.encode_boolean(attr_value, attr)
		elif attr_type == "dict":
			return self.encode_dict(attr, attr_value)
		elif attr_type == "int":
			if attr_value == None:
				return 2  # hack for bad data in the price ranges, right?
			else:
				return attr_value
		elif attr_type == "list":
			return self.encode_list(attr, attr_value)

	def get_attribute_type(self, attr_name):
		return self.get_attribute_info(attr_name)['type']

	def get_attribute_values(self, attr_name):
		return self.get_attribute_info(attr_name)['values']

	def get_attribute_info(self, attr_name):
		for attr in self.attributes:
			if attr_name == attr['name']:
				return attr

	def enabled_attributes(self):
		return filter(lambda attr: attr['enabled'],self.attributes)


	def transform_instance(self, instance):
		"""
		Main Public method
		Takes raw data as dict and returns an tuple of
		(attributes, stars, rating)
		the three are arrays ready to hand to sklearn, the first for inputs and
		the second two for outputs
		"""
		attribute_dict = instance['attributes']
		encoded_instance = []
		enabled_attributes = self.enabled_attributes()
		for attr in enabled_attributes:
			if attr['name'] in attribute_dict:
				value = attribute_dict[attr['name']]
			elif attr['name'] in instance:
				# for top level keys
				value = instance[attr['name']]
			else:
				value = None

			name = attr['name']
			encoding = self.encode_attribute(name, value)

			if isinstance(encoding, list):
				encoded_instance += encoding
			else:
				encoded_instance.append(encoding)

		return (encoded_instance, instance["stars"], instance["review_count"])
