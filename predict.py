import sys
from data_cleaner import StackOverflowPost, parsePosts
from model import TensorFlowModel
from random import sample

if __name__ == "__main__":
    posts: list[StackOverflowPost] = parsePosts(["data/sample.xml"])
    model = TensorFlowModel.from_filename(sys.argv[1])

    prediction_sample = sample(posts, 5)
    predicted_correctly = 0
    for i in range(len(prediction_sample)):
        (prediction, confidence) = model.predict_with_confidence_score(prediction_sample[i].to_tensor_flow_input())
        print(f"Question #{i + 1}")
        print(f"Link: {prediction_sample[i].link}")
        prediction_sample[i].print_metadata()
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {round(confidence * 100, 2)}%")
        print(f"Actual: {prediction_sample[i].is_answered}\n")
        if prediction == prediction_sample[i].is_answered:
            predicted_correctly += 1
    
    print(f"Predicted {predicted_correctly} out of {len(prediction_sample)} correctly. ({round(predicted_correctly / len(prediction_sample) * 100, 2)}% accuracy)")