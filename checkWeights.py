import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("model.h5")

# Get the weights of the input layer
weights = model.weights[0]

# Convert the weights to a numpy array
weights_array = weights.numpy()

# Get the number of input neurons and output neurons
num_inputs = weights_array.shape[0]
num_outputs = weights_array.shape[1]

# Print the weight values in a table
print("Input Neuron".ljust(15) + "Output Neuron".ljust(15) + "Weight Value")
for i in range(num_inputs):
  for j in range(num_outputs):
    weight = weights_array[i][j]
    print(str(i).ljust(15) + str(j).ljust(15) + str(weight))


# Importance
importances = []
for i in range(num_inputs):
  # Average the absolute values of the weights for this input neuron
  importance = sum(abs(weights_array[i])) / num_outputs
  importances.append(importance)

# Print the importances
print(importances)