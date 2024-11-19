import json

with open('data/tfrecord-00001-of-01000_307.json', 'r') as file:
    data = json.load(file)

# Now 'data' contains the JSON file contents as a Python dictionary (or list)

with open('kian_test_data.json', 'w') as json_file:
    json.dump({'roads': data['roads'], 'objects':data['objects']}, json_file, indent=4)

print(data.keys())
# print(data['roads'][0].keys())
print(data['objects'][0].keys())
print(json.dumps(data["objects"][0]["valid"], indent=4))
# print(json.dumps(data['objects'][0] , indent=4) )
with open('kian_test_data_single.json', 'r') as json_file:
    data2 = json.load(json_file)

data2 = json.dumps(data2, indent =4 )

# print(data2)