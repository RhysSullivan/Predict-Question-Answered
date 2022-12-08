import sys
from model import TensorFlowModel

if __name__ == "__main__":
  model = TensorFlowModel.from_filename(sys.argv[1])
  weights = model.weights()

  # Print the weight values with formatting
  spacing = 15
  print("Input Neuron".ljust(spacing) + "Output Neuron".ljust(spacing) + "Weight Value")
  for i in range(len(weights)):
    for j in range(len(weights[i])):
      print(str(i).ljust(spacing) + str(j).ljust(spacing) + str(weights[i][j]))

  # Print importances
  spacing = 28
  print("\n".ljust(spacing), end="")
  for label in ["Title Length", "Total Word Count", "Number of Code Snippets", "Total Code Length", "Number of Images", "Number of Tags"]:
    print(label.ljust(spacing), end="")
    
  print("\nNot Normalized:".ljust(spacing), end="")
  for important in model.importances(normalized = False):
    print(str(round(important, 8)).ljust(spacing), end="")

  print("\nNormalized:".ljust(spacing), end="")
  for norm_important in model.importances(normalized = True):
    print(str(round(norm_important, 8)).ljust(spacing), end="")