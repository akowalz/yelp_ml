import pickle
import json
from sklearn.preprocessing import LabelEncoder
import pdb

class Transformer:
	def __init__(self, attributes):
		self.attributes = attributes
		self.encoders = self.build_encoders()

	def encode_boolean(self, value):
		return int(value)

	def encode_string(self, attribute_name, value):
		encoder = self.encoders[attribute_name]
		return encoder.transform([value])[0]

	def encode_dict(self, attribute_name, value_dict):
		output = []
		attr_values = self.get_attribute_values(attribute_name)
		for value in attr_values:
			if value in value_dict:
				encoding = self.encode_boolean(value_dict[value])	
			else:
				encoding = 0
			output.append(encoding)	

		return output

	def build_encoders(self):
		encoders = {}
		for attr in self.attributes:
			if attr['type'] == 'string':
				le = LabelEncoder()
				le.fit(attr['values'])
				encoders[attr['name']] = le
	
		return encoders

	def encode_attribute(self, attr, attr_value):
		attr_type = self.get_attribute_type(attr)
		if attr_type == "string":
			return self.encode_string(attr, attr_value)
		elif attr_type == "bool":
			return self.encode_boolean(attr_value)
		elif attr_type == "dict":
			return self.encode_dict(attr, attr_value)

	def transform_instance(self, instance):
		attribute_dict = instance['attributes']
		encoded_instance = []
		for attr in self.attributes:
			if attr['name'] in attribute_dict.keys():
				value = attribute_dict[attr['name']]
				name = attr['name']
				encoding = self.encode_attribute(name, value)
				if isinstance(encoding, list):
					encoded_instance += encoding
				else:
					encoded_instance.append(encoding)
			else:
				encoded_instance.append(-1)

		return (encoded_instance, instance["stars"])

	def get_attribute_type(self, attr_name):
		for attr in self.attributes:
			if attr_name == attr['name']:
				return attr['type']

	def get_attribute_values(self, attr_name):
		for attr in self.attributes:
			if attr_name == attr['name']:
				return attr['values']

