from __future__ import annotations
import tensorflow as tf
import numpy as np

TensorFlowInputType = list[list[int]] # StackOverflowPost
TensorFlowOutputType = int # str

OPTIONS: list[TensorFlowOutputType] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # ["Not Answered", "Answered"]
MODEL_FILE_NAME = "model.h5"

# Credit: https://www.tensorflow.org/tutorials/quickstart/beginner
class TensorFlowModel():

    def __init__(self, model = None):
        if model is not None:
            self.model = model
        else:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape = (28, 28)), # input layer
                tf.keras.layers.Dense(128, activation = "relu"), # hidden layer
                tf.keras.layers.Dropout(0.2), # helps prevent overfitting
                tf.keras.layers.Dense(len(OPTIONS)) # output layer
            ])

    @classmethod
    def from_filename(cls, filename) -> TensorFlowModel:
        model = tf.keras.models.load_model(filename)
        return cls(model)

    def train(self, input_train: list[TensorFlowInputType], output_train: list[TensorFlowOutputType]):
        predictions = self.model(input_train[:1]).numpy()
        tf.nn.softmax(predictions).numpy()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        loss_fn(output_train[:1], predictions).numpy()
        self.model.compile(optimizer = "adam", loss = loss_fn, metrics = ["accuracy"])
        self.model.fit(input_train, output_train, epochs = 5)

    def save(self, filename: str):
        self.model.save(filename)

    def evaluate(self, input_test: list[TensorFlowInputType], output_test: list[TensorFlowOutputType]):
        self.model.evaluate(input_test, output_test, verbose = 2)
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        probability_model(input_test[:5])

    def predict(self, input: TensorFlowInputType) -> TensorFlowOutputType:
        return OPTIONS[np.argmax(self.model.predict(np.array([input]))[0])]

    def validate(self, input: TensorFlowInputType, output: TensorFlowOutputType) -> bool:
        return self.predict(input) == output


if __name__ == "__main__":
    # This is a dataset that classifies handwritten digits
    # TODO: replace with StackOverflow dataset
    dataset = tf.keras.datasets.mnist
    (input_train, output_train), (input_test, output_test) = dataset.load_data()
    input_train = input_train / 255.0
    input_test = input_test / 255.0

    model = TensorFlowModel()
    model.train(input_train, output_train)
    # model.save(MODEL_FILE_NAME)
    # model = TensorFlowModel.from_filename(MODEL_FILE_NAME)

    model.evaluate(input_test, output_test)
    prediction = model.predict(input_test[0])
    validation = model.validate(input_test[1], output_test[1])
    print(validation)
