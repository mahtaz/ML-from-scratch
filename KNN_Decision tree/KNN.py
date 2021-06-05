class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def comparing(self, x):
        distances, k_neighbor_labels = [], []
        for i in self.X_train:
            distances.append(euclidean_distance(x, i))
        ks = np.argsort(distances)[:self.k]  # indexes of first n nearest neighbours
        for i in ks:
            k_neighbor_labels.append(self.y_train[i])
        most_repeated = Counter(k_neighbor_labels).most_common(1)[0][0]
        return most_repeated

    def predict(self, X):
        predicted_y = np.array([self.comparing(x) for x in X])
        return predicted_y