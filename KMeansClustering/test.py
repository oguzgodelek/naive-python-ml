from kmeans import KMeansClustering
import numpy as np
import matplotlib.pyplot as plt


def draw_2d_points(model, points):
    assignments = model.predict(points)
    plt.scatter(list(map(lambda x: x[0], points)), list(map(lambda x: x[1], points)),
                c=assignments)
    plt.show()


def test():
    first_distribution = np.random.multivariate_normal((0, 0), ((2, 0), (0, 2)), 500)
    second_distribution = np.random.multivariate_normal((5, 5), ((2, 0), (0, 2)), 500)
    third_distribution = np.random.multivariate_normal((15, 15), ((5, 0), (0, 2)), 500)
    points = np.concatenate((first_distribution, second_distribution, third_distribution))
    model = KMeansClustering(n_clusters=4)
    model.fit(points)
    draw_2d_points(model, points)
    plt.plot(model.error_history)
    plt.title("K-Means Clustering Average Distance vs epoch")
    plt.show()


if __name__ == '__main__':
    test()
