import random
import math


class LogisticRegression:
    def __init__(self, lr: float = 1e-2,
                       max_iter: int = 5000,
                       verbose: bool = True):
        self.lr = lr
        self.max_iter = max_iter
        self.verbose = verbose
        self._data: list[list[float]] = []
        self._y: list[int] = []
        self.weights: list[float] = []
        self.bias: float = 0.0
        self.error_history: list[float] = []
        self.early_stopping_error: float = 1e-4

    def fit(self, x: list[list[float]],
                  y: list[int]) -> None:
        self._data = x
        self._y = y
        if len(self._data) == 0 or len(self._data[0]) == 0:
            raise RuntimeError("Data is empty")
        self.weights = [random.gauss(mu=0, sigma=1) for _ in range(len(self._data[0]))]
        self.bias = random.gauss(mu=0, sigma=1)
        for epoch in range(1, self.max_iter + 1):
            predictions = self.predict(self._data)
            bce_error = self.calculate_error(predictions, self._y)
            self.error_history.append(bce_error)
            ind_errors = list(map(lambda pair: pair[0] - pair[1], zip(self._y, predictions)))
            if self.verbose and epoch % 200 == 0:
                print(f"Epoch: {epoch}, BCE Error: {bce_error}")
            self.bias += self.lr * sum(ind_errors) / len(self._data)
            self.weights = list(map(lambda w: w[1] + self.lr * (sum(map(lambda data: ind_errors[data[0]] * data[1][w[0]],
                                                                        enumerate(self._data))) / len(self._data)),
                                    enumerate(self.weights)))
            if epoch > 2 and abs(self.error_history[-1] - self.error_history[-2]) < self.early_stopping_error:
                break

    def predict(self, data: list[list[float]]) -> list[float]:
        if len(self.weights) == 0:
            raise RuntimeError("Regressor has not been initialized yet")
        y = map(lambda row: sum(map(lambda x: x[0] * x[1], zip(row, self.weights))) + self.bias, data)
        return list(map(lambda value: 1 / (1 + math.exp(value * -1)), y))

    def predict_class(self, data: list[list[float]]) -> list[int]:
        return list(map(lambda value: value > 0.5, self.predict(data)))

    @staticmethod
    def calculate_error(predictions: list[float], trues: list[int]) -> float:
        return -sum(map(lambda pair: math.log2(pair[0]) if pair[1] == 1 else math.log2(1 - pair[0]),
                        zip(predictions, trues))) / len(predictions)



