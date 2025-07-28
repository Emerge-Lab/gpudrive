import shutil
import os
import json

src_file = 'town01.json'
base_name = 'town01'
ext = '.json'


# Create 9 more copies with unique names
for i in range(2, 11):
    new_name = f"{base_name}_{i}{ext}"
    # Copy the file
    shutil.copy(src_file, new_name)
    # Update the "name" field in the copied file
    with open(new_name, 'r+') as f:
        data = json.load(f)
        data['name'] = new_name
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()