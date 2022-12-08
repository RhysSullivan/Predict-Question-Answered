from data_cleaner import StackOverflowPost, parsePosts
from model import TensorFlowModel

if __name__ == "__main__":
    posts: list[StackOverflowPost] = parsePosts(["data/sample.xml"], 5)
    model = TensorFlowModel.from_filename("model.h5")

    for i in range(len(posts)):
        prediction = model.predict(posts[i].to_tensor_flow_input())
        print(f"Question #{i+1}")
        posts[i].demo()
        print(f"\nPrediction: {prediction}")
        print(f"Actual: {posts[i].is_answered}\n")