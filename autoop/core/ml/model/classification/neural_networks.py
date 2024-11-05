import numpy as np
from autoop.core.ml.model.model import Model


class NeuralNetwork(Model):
    def __init__(
        self,
        name: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.01,
        num_iterations: int = 10,
    ):
        """
        Initializes the Neural Network with given architecture and parameters.
        Args:
            name (str): The name of the model.
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output classes.
            learning_rate (float, optional): Learning rate for weight updates.
                                             Defaults to 0.01.
            num_iterations (int, optional): Number of iterations for training.
                                             Defaults to 10000.
        """
        super()._init_(name=name)
        self._input_size: int = input_size
        self._hidden_size: int = hidden_size
        self._output_size: int = output_size
        self._learning_rate: float = learning_rate
        self._num_iterations: int = num_iterations

        self._W1: np.ndarray = np.random.randn(
            self._input_size, self._hidden_size
        )
        self._b1: np.ndarray = np.zeros((1, self._hidden_size))
        self._W2: np.ndarray = np.random.randn(
            self._hidden_size, self._output_size
        )
        self._b2: np.ndarray = np.zeros((1, self._output_size))

    @property
    def input_size(self) -> int:
        """Get the number of input features."""
        return self._input_size

    @input_size.setter
    def input_size(self, size: int) -> None:
        """
        Set the number of input features, updates weights and resets biases.
        """
        self._input_size = size
        self._W1 = np.random.randn(self._input_size, self._hidden_size)
        self._b1 = np.zeros((1, self._hidden_size))

    @property
    def hidden_size(self) -> int:
        """Get the number of neurons in the hidden layer."""
        return self._hidden_size

    @hidden_size.setter
    def hidden_size(self, size: int) -> None:
        """
        Set the number of neurons in the hidden layer,
        updates weights and resets biases.
        """
        self._hidden_size = size
        self._W1 = np.random.randn(self._input_size, self._hidden_size)
        self._b1 = np.zeros((1, self._hidden_size))
        self._W2 = np.random.randn(self._hidden_size, self._output_size)
        self._b2 = np.zeros((1, self._output_size))

    @property
    def output_size(self) -> int:
        """Get the number of output classes."""
        return self._output_size

    @output_size.setter
    def output_size(self, size: int) -> None:
        """
        Set the number of output classes,
        updates weights and resets biases.
        """
        self._output_size = size
        self._W2 = np.random.randn(self._hidden_size, self._output_size)
        self._b2 = np.zeros((1, self._output_size))

    @property
    def learning_rate(self) -> float:
        """Get the learning rate."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, rate: float) -> None:
        """Set the learning rate."""
        self._learning_rate = rate

    @property
    def num_iterations(self) -> int:
        """Get the number of training iterations."""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, iterations: int) -> None:
        """Set the number of training iterations."""
        self._num_iterations = iterations

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid activation function.
        Args:
            z (np.ndarray): Input value(s) to the activation function.
        Returns:
            np.ndarray: The output after applying the sigmoid function,
                        with values between 0 and 1.
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the sigmoid function.
        Args:
            z (np.ndarray): Output values of the sigmoid function.
        Returns:
            np.ndarray: The derivative of the sigmoid function evaluated at z,
                        which represents the slope of the sigmoid curve at
                        each point.
        """
        return z * (1 - z)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the neural network using the provided training data.
        Args:
            X (np.ndarray): The input feature matrix.
            y (np.ndarray): The target output values.
        """
        for i in range(self._num_iterations):
            z1 = np.dot(X, self._W1) + self._b1
            a1 = self.sigmoid(z1)
            z2 = np.dot(a1, self._W2) + self._b2
            a2 = self.sigmoid(z2)

            loss = -np.mean(
                y * np.log(a2 + 1e-10) + (1 - y) * np.log(1 - a2 + 1e-10)
            )

            dz2 = a2 - y
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dz1 = np.dot(dz2, self._W2.T) * self.sigmoid_derivative(a1)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self._W1 -= self._learning_rate * dW1
            self._b1 -= self._learning_rate * db1
            self._W2 -= self._learning_rate * dW2
            self._b2 -= self._learning_rate * db2

            if i % 1000 == 0:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for the input samples.
        Args:
            X (np.ndarray): The input feature matrix.
        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        z1 = np.dot(X, self._W1) + self._b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self._W2) + self._b2
        a2 = self.sigmoid(z2)
        predictions = [1 if i > 0.5 else 0 for i in a2]
        return np.array(predictions)
