import sys
from data_cleaner import StackOverflowPost, parsePosts
from model import TensorFlowModel
from random import sample

if __name__ == "__main__":
    posts: list[StackOverflowPost] = parsePosts(["data/sample.xml"])
    model = TensorFlowModel.from_filename(sys.argv[1])

    predictionSample = sample(posts, 5)
    for i in range(len(predictionSample)):
        prediction = model.predict(predictionSample[i].to_tensor_flow_input())
        print(f"Question #{i + 1}")
        predictionSample[i].print_metadata()
        print(f"\nPrediction: {prediction}")
        print(f"Actual: {predictionSample[i].is_answered}\n")