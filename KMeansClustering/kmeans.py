import random


class KMeansClustering:
    def __init__(self, n_clusters: int = 3,
                       max_iter: int = 300,
                       verbose: bool = False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.e = 1e-7  # Small constant to avoid zero division error
        self.early_stopping_error = 1e-5
        self.centroids: list[list[float]] = []
        self.assignments: list[int] = []
        self.error_history: list[float] = []
        self._data: list[list[float]] = []
        self.dimension: int = -1
        self.boundaries: list[dict[str, float]] = []

    def fit(self, data: list[list[float]]) -> None:
        if self.n_clusters > len(data):
            raise RuntimeError('Number of cluster is bigger than the number of points')
        if len(data) == 0:
            raise RuntimeError('There is no point in the data.')
        self._data = data
        self.dimension = len(data[0])
        self.boundaries = [{'max': max(map(lambda x: x[idx], self._data)),
                            'min': min(map(lambda x: x[idx], self._data))}
                           for idx in range(self.dimension)]
        self._data = data
        # Randomly initialize centroids
        self.centroids = [list(map(lambda idx: random.randint(0, int(self.boundaries[idx]['max'] - self.boundaries[idx]['min'])) + self.boundaries[idx]['min'] ,
                                   range(self.dimension)))
                          for _ in range(self.n_clusters)]
        self.assignments = [-1 for _ in range(len(data))]
        for idx in range(self.max_iter):
            self.error_history.append(self._assign_points())
            if self.verbose:
                print(f'Iteration {idx+1}: {self.error_history[-1]}')
            # Early stopping
            if idx > 2 and abs(self.error_history[-1] - self.error_history[-2]) < self.early_stopping_error:
                break
            self._update_centroids()

    # It is a generic Lx distance function controlled by l parameter
    @staticmethod
    def _distance(vector1: list[float],
                  vector2: list[float],
                  l: float = 2) -> float:
        return sum([(x - y) ** l for x, y in zip(vector1, vector2)]) ** (1/l)

    @staticmethod
    def argmin(distance_matrix: list[list[float]]) -> list[int]:
        return list(map(lambda x: sorted(enumerate(x),
                                         key=lambda y: y[1])[0][0],
                        distance_matrix))

    def _assign_points(self):
        centroid_distances = [[self._distance(self._data[idx], centroid) for centroid in self.centroids]
                              for idx in range(len(self._data))]

        self.assignments = self.argmin(centroid_distances)
        # Return average error which is the distance to the centroids
        return sum([centroid_distances[idx][self.assignments[idx]] for idx in range(len(self._data))]) / len(self._data)

    def _update_centroids(self):
        cluster_points = [list(map(lambda x: x[1], filter(lambda x: self.assignments[x[0]] == idx, enumerate(self._data))))
                          for idx in range(self.n_clusters)]

        self.centroids = list(map(lambda group: [sum(map(lambda y: y[0], group)) / (len(group) + self.e), sum(map(lambda y: y[1], group)) / (len(group) + self.e)],
                                  cluster_points))

    def predict(self, points: list[list[float]]) -> list[int]:
        centroid_distances = [[self._distance(points[idx], centroid) for centroid in self.centroids]
                              for idx in range(len(points))]
        assignments = self.argmin(centroid_distances)
        return assignments
