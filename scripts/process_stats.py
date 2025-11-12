import re
import pandas as pd
import os

data = os.popen('bash scripts/collect_stats.bash r43').read()
print(data)

lines = data.strip().split('\n')
stats = {}
current_item = None

for line in lines:
    if line.startswith('output/'):
        current_item = line.split('/')[1].split('_')[0]
        if current_item not in stats:
            stats[current_item] = {}
    elif re.match(r'^\d+\.\d+$', line):
        stats[current_item]['training_time'] = float(line)
    elif 'ckpt.ply' in line:
        size_mb = int(re.search(r'(\d+)M', line).group(1))
        stats[current_item]['model_size_mb'] = size_mb
    elif 'element tetrahedron' in line:
        primitives = int(line.split()[-1])
        stats[current_item]['primitives'] = primitives

order = [
    'bicycle', 'flowers', 'garden', 'stump', 'treehill', 'room', 'counter',
    'kitchen', 'bonsai', 'truck', 'train', 'drjohnson', 'playroom'
]

data_list = []
for item in order:
    if item in stats:
        data_list.append({
            'item': item,
            'training_time': stats[item].get('training_time'),
            'model_size_mb': stats[item].get('model_size_mb'),
            'primitives': stats[item].get('primitives')
        })

df = pd.DataFrame(data_list)
df.to_csv('stats.csv', index=False)
