import random


class LinearRegression:
    def __init__(self, lr: float = 1e-2,
                       max_iter: int = 1000,
                       verbose: bool = True):
        self.lr = lr
        self.max_iter = max_iter
        self.verbose = verbose
        self._data: list[list[float]] = []
        self._y: list[float] = []
        self.weights: list[float] = []
        self.bias: float = 0.0
        self.error_history: list[float] = []
        self.early_stopping_error: float = 1e-4

    def fit(self, x: list[list[float]],
                  y: list[float]) -> None:
        self._data = x
        self._y = y
        if len(self._data) == 0 or len(self._data[0]) == 0:
            raise RuntimeError("Data is empty")
        self.weights = [random.gauss(mu=0, sigma=1) for _ in range(len(self._data[0]))]
        self.bias = random.gauss(mu=0, sigma=1)
        for epoch in range(1, self.max_iter + 1):
            predictions = self.predict(self._data)
            mse_error = self.calculate_mse_error(predictions, self._y)
            self.error_history.append(mse_error)
            ind_errors = list(map(lambda pair: pair[0] - pair[1], zip(self._y, predictions)))
            if self.verbose:
                print(f"Epoch: {epoch}, MSE Error: {mse_error}")
            self.bias += self.lr * sum(ind_errors) / len(self._data)
            self.weights = list(map(lambda w: w[1] + self.lr * (sum(map(lambda data: ind_errors[data[0]] * data[1][w[0]],
                                                                        enumerate(self._data))) / len(self._data)),
                                    enumerate(self.weights)))
            if epoch > 2 and abs(self.error_history[-1] - self.error_history[-2]) < self.early_stopping_error:
                break

    def predict(self, data: list[list[float]]) -> list[float]:
        if len(self.weights) == 0:
            raise RuntimeError("Regressor has not been initialized yet")
        return list(map(lambda row: sum(map(lambda x: x[0]*x[1],
                                            zip(row, self.weights))) + self.bias,
                        data))

    @staticmethod
    def calculate_mse_error(y1: list[float], y2: list[float]) -> float:
        if len(y1) != len(y2):
            raise RuntimeError("Number of elements does not match")
        return sum(map(lambda pair: (pair[0] - pair[1]) ** 2, zip(y1, y2))) / len(y1)
