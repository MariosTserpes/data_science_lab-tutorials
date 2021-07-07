# EXAMPLE OF K-NN Using Manhattan Norm

#### Step 0

dataset = [[-1, 1.5, 0],
	       [ 3, 0.1, 1],
	       [1.396561688, 4.400293529, 0],
	       [2.38807019, 1.850220317, 0],
	       [0.06407232, 3.005305973, 0],
	       [7.627531214, 10.759262235, 1],
	       [0.33, 2.088, 0],
	       [1.922596716, 7.77106367, 1],
	       [2.675418651, -0.242068655, 1],
	       [4.44, -1.23, 1]]

dataset[2]

# Define distance function
# It is included in Step 1. Specifically, the Manhattan distance between vector   dataset and point   dataset[2] is calculated.

# Step 1 Find distances

# Example of calculating Mnhattan Distance
from math import sqrt
#sum(abs(val1-val2) for val1, val2 in zip(a,b))
 
# calculate the Manhattan distance between two vectors
def manhattan_distance(row1, row2):
	
	for i in range(len(row1)-1):
		manh_distance = sum(abs(row1[i] - row2[i]) for row1[i], row2[i] in zip(row1, row2))
	return manh_distance
 
# Test distance function
dataset = [[-1, 1.5, 0],
	       [ 3, 0.1, 1],
	       [1.396561688, 4.400293529, 0],
	       [2.38807019, 1.850220317, 0],
	       [0.06407232, 3.005305973, 0],
	       [7.627531214, 10.759262235, 1],
	       [0.33, 2.088, 0],
	       [1.922596716, 7.77106367, 1],
	       [2.675418651, -0.242068655, 1],
	       [4.44, -1.23, 1]]
row0 = dataset[2]
for row in dataset:
	distance = manhattan_distance(row0, row)
	print(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = manhattan_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
neighbors = get_neighbors(dataset, dataset[2], 5)
for neighbor in neighbors:
	print(neighbor)

def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
prediction = predict_classification(dataset, dataset[2], 5)
print('Expected %d, Got %d.' % (dataset[0][-1], prediction))