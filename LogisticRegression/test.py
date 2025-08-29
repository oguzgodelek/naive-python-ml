from logistic_regression import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


def test():
    model = LogisticRegression()
    n_of_points_per_group = 100
    x1 = np.random.normal(scale=2, size=(n_of_points_per_group, 2))
    y1 = [1 for _ in range(n_of_points_per_group)]
    x2 = np.random.normal(scale=2, size=(n_of_points_per_group, 2)) + 5
    y2 = [0 for _ in range(n_of_points_per_group)]

    x = list(map(lambda point: list(point), np.concatenate((x1, x2), axis=0)))
    y = y1 + y2

    model.fit(x, y)
    assignments = model.predict_class(x)

    plt.scatter(list(map(lambda x: x[0], x)), list(map(lambda x: x[1], x)), c=assignments)
    plt.show()


if __name__ == '__main__':
    test()
