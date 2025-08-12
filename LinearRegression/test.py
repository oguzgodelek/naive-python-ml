from linear_regression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


def test():
    model = LinearRegression()
    n_of_points = 50
    x = np.linspace(0, 2, n_of_points)
    y = list(x * 2 + 2 + np.random.random(n_of_points))
    x = list(map(lambda z: [z], x))

    model.fit(x, y)
    predictions = model.predict(x)
    plt.plot(x, y, 'ro', label="Original")
    plt.plot(x, predictions, 'b-', label="Prediction")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test()
