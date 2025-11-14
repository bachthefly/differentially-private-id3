import numpy as np
import pandas as pd


class Node:
    def __init__(self, key=None, leaf=False, label=None,
                 threshold=None, children=None, left=None, right=None):
        self.key = key
        self.leaf = leaf
        self.label = label
        self.threshold = threshold
        self.children = children or {}
        self.left = left
        self.right = right


class DPDecisionTree:
    """
    Differentially Private ID3 Decision Tree
    """
    def __init__(self, epsilon1: float = 1.0, max_depth: int = 5):
        assert epsilon1 > 0, "epsilon1 must be > 0 for DP"
        self.epsilon1 = epsilon1
        self.max_depth = max_depth
        self.root = None
        self.target_col = None

    #----------------------------
    # Laplace noise
    #----------------------------
    def laplace(self, scale: float):
        return np.random.laplace(0.0, scale)

    #----------------------------
    # DP entropy from class counts
    #----------------------------
    def dp_entropy_from_counts(self, counts, epsilon: float) -> float:
        assert epsilon > 0, "epsilon must be > 0"

        counts = np.array(counts, dtype=float)
        counts = counts + np.random.laplace(0.0, 1.0 / epsilon, size=len(counts))

        total = counts.sum()
        p = counts / total
        return float(-np.sum(p * np.log2(p)))

    #----------------------------
    # DP noisy majority vote
    #----------------------------
    def dp_majority_label(self, labels, epsilon: float):
        assert epsilon > 0

        labels = np.array(labels)
        classes, counts = np.unique(labels, return_counts=True)
        counts = counts + np.random.laplace(0.0, 1.0 / epsilon, size=len(counts))
        return classes[int(np.argmax(counts))]

    #----------------------------
    # DP entropy for categorical split
    #----------------------------
    def find_entropy_split_categorical(self, D, a, N, epsilon2):
        tot = 0.0
        for v in D[a].unique():
            Dj = D[D[a] == v]

            count_D_j = len(Dj) + self.laplace(1.0 / epsilon2)
            count_D_j = max(count_D_j, 1e-6)

            subtree_entropy = 0.0
            labels_j = Dj[self.target_col].values
            for c in np.unique(labels_j):
                true_count = np.sum(labels_j == c)
                noisy_count = true_count + self.laplace(1.0 / epsilon2)
                noisy_count = max(noisy_count, 1e-6)

                p_i = noisy_count / count_D_j
                subtree_entropy -= p_i * np.log2(p_i)

            tot += subtree_entropy * (count_D_j / N)

        return float(tot)

    #----------------------------
    # DP entropy for numeric split
    #----------------------------
    def find_entropy_split_numeric(self, D, a, N, epsilon2):
        col = D[a].astype(float).values
        y = D[self.target_col].values

        idx = np.argsort(col)
        col_sorted = col[idx]
        y_sorted = y[idx]

        unique_vals = np.unique(col_sorted)
        if len(unique_vals) <= 1:
            return float("inf"), None

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        best_tot = float("inf")
        best_t = None

        for t in thresholds:
            left_mask = col_sorted < t
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            y_left = y_sorted[left_mask]
            y_right = y_sorted[right_mask]

            count_left = len(y_left) + self.laplace(1.0 / epsilon2)
            count_left = max(count_left, 1e-6)
            _, counts_L = np.unique(y_left, return_counts=True)
            H_left = self.dp_entropy_from_counts(counts_L, epsilon2)

            count_right = len(y_right) + self.laplace(1.0 / epsilon2)
            count_right = max(count_right, 1e-6)
            _, counts_R = np.unique(y_right, return_counts=True)
            H_right = self.dp_entropy_from_counts(counts_R, epsilon2)

            tot = H_left * (count_left / N) + H_right * (count_right / N)

            if tot < best_tot:
                best_tot = tot
                best_t = t

        return float(best_tot), best_t

    #----------------------------
    # Core DP-ID3 recursive build
    #----------------------------
    def _dp_id3_core(self, D, A, epsilon1, depth):
        true_N = len(D)
        N = true_N + self.laplace(1.0 / epsilon1)

        labels = D[self.target_col]
        unique_labels = labels.unique()

        if depth >= self.max_depth or len(A) == 0 or len(unique_labels) == 1:
            return Node(leaf=True, label=self.dp_majority_label(labels, epsilon1))

        epsilon2 = epsilon1 / (2.0 * len(A))
        assert epsilon2 > 0

        G = {}
        best_thresholds = {}

        for a in A:
            col = D[a]
            if pd.api.types.is_numeric_dtype(col):
                cond_entropy, t = self.find_entropy_split_numeric(D, a, N, epsilon2)
                G[a] = cond_entropy
                best_thresholds[a] = t
            else:
                cond_entropy = self.find_entropy_split_categorical(D, a, N, epsilon2)
                G[a] = cond_entropy

        a_hat = min(G, key=G.get)
        root = Node(key=a_hat)

        if pd.api.types.is_numeric_dtype(D[a_hat]):
            threshold = best_thresholds[a_hat]
            left_df = D[D[a_hat] < threshold]
            right_df = D[D[a_hat] >= threshold]
            remaining = [x for x in A if x != a_hat]

            root.threshold = threshold
            root.left = self._dp_id3_core(left_df, remaining, epsilon1, depth + 1)
            root.right = self._dp_id3_core(right_df, remaining, epsilon1, depth + 1)
            root.label = self.dp_majority_label(labels, epsilon1)
            return root

        remaining = [x for x in A if x != a_hat]
        for v, sub_df in D.groupby(a_hat):
            root.children[v] = self._dp_id3_core(sub_df, remaining, epsilon1, depth + 1)

        root.label = self.dp_majority_label(labels, epsilon1)
        return root

    #----------------------------
    # Public training function
    #----------------------------
    def fit(self, df, target_col):
        self.target_col = target_col
        feature_cols = [c for c in df.columns if c != target_col]
        self.root = self._dp_id3_core(df, feature_cols, self.epsilon1, depth=0)

    #----------------------------
    # Prediction functions
    #----------------------------
    def predict_one(self, record):
        node = self.root
        get = record.get if isinstance(record, dict) else record.__getitem__

        while not node.leaf:
            if node.threshold is not None:
                val = float(get(node.key))
                node = node.left if val < node.threshold else node.right
            else:
                child = node.children.get(get(node.key))
                if child is None:
                    return node.label
                node = child

        return node.label

    def predict(self, df):
        return np.array([self.predict_one(row) for _, row in df.iterrows()])


if __name__ == "__main__":
    data = {
        "age": [23, 31, 45, 50, 35, 40],
        "color": ["red", "red", "blue", "blue", "red", "blue"],
        "label": ["yes", "yes", "no", "no", "yes", "no"],
    }
    df = pd.DataFrame(data)

    tree = DPDecisionTree(epsilon1=1.0, max_depth=3)
    tree.fit(df, "label")

    for _, r in df.iterrows():
        print(r.to_dict(), "->", tree.predict_one(r))
