class DecisionTree:

    def __init__(self, max_depth=k, root=None, x_features=None, thresh=0.8):
        self.max_depth = max_depth
        self.x_features = x_features
        self.root = None
        self.thresh = thresh

    def fit(self, X, y):
        self.x_features = len(X[0]) if not self.x_features else min(self.x_features, len(X[0]))
        self.root = self.build_tree(X, y)

    def final_split(self, X, y, features):
        max_gain = -99
        for i in features:
            X_sub = X[:, i]  # each feature and its whole data as input to infogain
            spliting_values = np.unique(X_sub)
            for spliting_value in spliting_values:
                infogain = self.info_gain(y, X_sub, spliting_value)

                if infogain > max_gain:
                    max_gain = infogain
                    where_to_split = i
                    split_boundry = spliting_value
        return where_to_split, split_boundry

    def info_gain(self, y, x, split_value):
        parent_node_entropy = entropy(y)
        left_node, right_node = self.parent_split(x, split_value)
        left_node_entropy = (len(left_node) / len(y)) * entropy(y[left_node])
        right_node_entropy = (len(right_node) / len(y)) * entropy(y[right_node])
        info_gain = parent_node_entropy - right_node_entropy - left_node_entropy
        return info_gain

    def parent_split(self, x, split_boundry):  # split a node to left and right like gini index
        left_node, right_node = [], []
        for i in range(len(x)):
            if x[i] <= split_boundry:
                left_node.append(i)
            else:
                right_node.append(i)
        return left_node, right_node

    def tree_spliting(self, x, node):
        if node.is_leaf_node():
            return node.value
        elif x[node.feature] <= node.spliting_value:
            return self.tree_spliting(x, node.left)
        else:
            return self.tree_spliting(x, node.right)

    def predict(self, X):
        l = []
        for i in X:
            l.append(self.tree_spliting(i, self.root))
        return np.array(l)

