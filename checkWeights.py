import tensorflow as tf

model = tf.keras.models.load_model("model.h5") # Load the model

weights = model.weights[0] # Get the weights of the input layer

weights_array = weights.numpy() # Convert to array

num_inputs = weights_array.shape[0] # Number of input neurons
num_outputs = weights_array.shape[1] # Number of hidden output neurons

# Print the weight values with formatting
print("Input Neuron".ljust(15) + "Output Neuron".ljust(15) + "Weight Value")
for i in range(num_inputs):
  for j in range(num_outputs):
    weight = weights_array[i][j]
    print(str(i).ljust(15) + str(j).ljust(15) + str(weight))


# Importance of each input
spacing = 28
importances = []
for i in range(num_inputs):
  importances.append(round((sum(abs(weights_array[i])) / num_outputs), 8)) # Average the absolute values of the weights for this input neuron

# Print results
print()
print("Labels: ".ljust(spacing), end="")
labels = ["Title Length", "Total Word Count", "Number of Code Snippets", "Total Code Length", "Number of Images", "Number of Tags"]
for label in labels:
    print(label.ljust(spacing), end="")

print()
print("Not normalized: ".ljust(spacing), end="")
# Print the importances
for important in importances:
    print(str(important).ljust(spacing), end="")

print()
print("Normalized: ".ljust(spacing), end="")
# Print the normalized importances
normalized_importances = tf.keras.utils.normalize(importances, axis=-1)[0]
for norm_important in normalized_importances:
    print(str(round(norm_important, 8)).ljust(spacing), end="")