from __future__ import annotations
import tensorflow as tf
import numpy as np
from data_cleaner import StackOverflowPost, parseChunkedPosts

INPUT_PARAMETER_COUNT = 6
HIDDEN_LAYER_STRUCTURE = [11, 8]
TOTAL_SIZE = 515817
TEST_SIZE = TOTAL_SIZE // 10
TRAIN_SIZE = TOTAL_SIZE - TEST_SIZE
MODEL_FILE_NAME = f"model_{TRAIN_SIZE}_{TEST_SIZE}_{HIDDEN_LAYER_STRUCTURE}.h5"

TensorFlowInputType = list[float]
TensorFlowOutputType = bool

# Credit: https://www.tensorflow.org/tutorials/quickstart/beginner
class TensorFlowModel():

    def __init__(self, model = None):
        if model is not None:
            self.model = model
        else:
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.InputLayer(input_shape=(INPUT_PARAMETER_COUNT,))) # input layer
            for neuronCount in HIDDEN_LAYER_STRUCTURE: # hidden layer(s)
                self.model.add(tf.keras.layers.Dense(neuronCount, activation = "relu")) 
            self.model.add(tf.keras.layers.Dropout(0.2)) # helps prevent overfitting
            self.model.add(tf.keras.layers.Dense(1, activation = "sigmoid")) # output layer

    @classmethod
    def from_filename(cls, filename: str) -> TensorFlowModel:
        model = tf.keras.models.load_model(filename)
        return cls(model)

    def train(self, input_train: list[TensorFlowInputType], output_train: list[TensorFlowOutputType]):
        predictions = self.model(np.array(input_train[:1])).numpy()
        tf.nn.softmax(predictions).numpy()
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        loss_fn(np.array(output_train[:1]), predictions).numpy()
        self.model.compile(optimizer = "adam", loss = loss_fn, metrics = ["accuracy"])
        self.model.fit(np.array(input_train), np.array(output_train), epochs = 5)

    def save(self, filename: str):
        self.model.save(filename)

    def evaluate(self, input_test: list[TensorFlowInputType], output_test: list[TensorFlowOutputType]):
        self.model.evaluate(np.array(input_test), np.array(output_test), verbose = 2)

    def predict(self, input: TensorFlowInputType) -> TensorFlowOutputType:
        result: float = self.model.predict(np.array([input]))[0][0] # value between [0, 1]
        return bool(round(result))

    def predict_with_confidence_score(self, input: TensorFlowInputType) -> tuple(TensorFlowOutputType, float):
        result: float = self.model.predict(np.array([input]))[0][0] # value between [0, 1]
        prediction = bool(round(result))
        return (True, result) if prediction else (False, 1 - result)

    def validate(self, input: TensorFlowInputType, output: TensorFlowOutputType) -> bool:
        return self.predict(input) == output

    def weights(self) -> list[list[float]]:
        return self.model.weights[0].numpy() # weights of the input layer as an array

    def importances(self, normalized: bool = False) -> list[float]:
        weights = self.weights()
        importances = [(sum(abs(weights[i])) / len(weights[i])) for i in range(INPUT_PARAMETER_COUNT)] # average the absolute values of the weights for each input neuron
        return importances if not normalized else tf.keras.utils.normalize(importances, axis=-1)[0]

if __name__ == "__main__":
    dataset: list[StackOverflowPost] = parseChunkedPosts()
    input_all: list[TensorFlowInputType] = [post.to_tensor_flow_input() for post in dataset]
    output_all: list[TensorFlowOutputType] = [post.to_tensor_flow_output() for post in dataset]

    input_train = input_all[:TRAIN_SIZE]
    output_train = output_all[:TRAIN_SIZE]
    input_test = input_all[-TEST_SIZE:]
    output_test = output_all[-TEST_SIZE:]

    model = TensorFlowModel()
    model.train(input_train, output_train)
    model.evaluate(input_test, output_test)
    model.save(MODEL_FILE_NAME)
