import numpy as np
import pandas as pd


class Node:
    def __init__(self, key=None, leaf=False, label=None,
                 threshold=None, children=None, left=None, right=None):
        """
        Simple tree node for DP-ID3.

        key       : feature name this node splits on (None for leaf)
        leaf      : True if node is a leaf
        label     : predicted label at this node (for leaf, or fallback for internal)
        threshold : numeric split threshold (for continuous features)
        children  : dict for categorical splits: value -> child Node
        left      : left child for numeric split (value < threshold)
        right     : right child for numeric split (value >= threshold)
        """
        self.key = key
        self.leaf = leaf
        self.label = label
        self.threshold = threshold
        self.children = children or {}
        self.left = left
        self.right = right


class DPDecisionTree:
    """
    Differentially Private ID3 Decision Tree.
    Supports both categorical and numeric features in a single implementation.
    """

    def __init__(self, epsilon1: float = 1.0, max_depth: int = 5):
        """
        Initialize DP-ID3 with privacy and depth parameters.

        Parameters:
        epsilon1 : float
            Global privacy budget used at each node. Must be > 0.
        max_depth : int
            Maximum tree depth (root at depth 0).
        """
        assert epsilon1 > 0, "epsilon1 must be > 0 for DP"
        self.epsilon1 = epsilon1
        self.max_depth = max_depth
        self.root: Node | None = None
        self.target_col: str | None = None

    #----------------------------
    # Laplace noise
    #----------------------------
    def laplace(self, scale: float):
        """Draw a Laplace(0, scale) sample."""
        return np.random.laplace(0.0, scale)

    #----------------------------
    # DP entropy from class counts
    #----------------------------
    def dp_entropy_from_counts(self, counts, epsilon: float) -> float:
        """
        Differentially private entropy from class counts.

        Adds Laplace noise to each class count with scale 1/epsilon,
        converts to probabilities, and computes:
            H = -sum_i p_i log2(p_i)
        in a numerically safe way (only over positive, finite p_i).
        """
        assert epsilon > 0

        # Add DP Laplace noise to counts
        counts = np.array(counts, dtype=float)
        counts = counts + np.random.laplace(0.0, 1.0 / epsilon, size=len(counts))

        # Ensure counts are non-negative and non-trivial
        counts[counts < 0] = 0.0
        total = counts.sum()

        # If everything got killed by noise, entropy is 0 by convention
        if total <= 0 or not np.isfinite(total):
            return 0.0

        p = counts / total

        # Only keep strictly positive, finite probabilities
        mask = (p > 0) & np.isfinite(p)
        if not np.any(mask):
            return 0.0

        p_safe = p[mask]

        return float(-np.sum(p_safe * np.log2(p_safe)))

    #----------------------------
    # DP noisy majority vote
    #----------------------------
    def dp_majority_label(self, labels, epsilon: float):
        """
        Differentially private majority label ("noisy mode").

        Computes class counts, adds Laplace(1/epsilon) noise to each count,
        and returns the label with the highest noisy count.
        """
        assert epsilon > 0

        labels = np.array(labels)
        classes, counts = np.unique(labels, return_counts=True)
        counts = counts + np.random.laplace(0.0, 1.0 / epsilon, size=len(counts))
        return classes[int(np.argmax(counts))]

    #----------------------------
    # DP entropy for categorical split
    #----------------------------
    def find_entropy_split_categorical(self, D, a, N, epsilon2):
        """
        Compute DP conditional entropy H(label | a) for a categorical attribute a.

        For each value v of attribute a:
            - add Laplace noise to |D[a == v]|
            - add Laplace noise to each class count in that subset
            - compute noisy subtree entropy
            - weight by noisy branch proportion and sum up
        """
        tot = 0.0
        for v in D[a].unique():
            Dj = D[D[a] == v]

            # noisy branch size
            count_D_j = len(Dj) + self.laplace(1.0 / epsilon2)
            count_D_j = max(count_D_j, 1e-6)

            subtree_entropy = 0.0
            labels_j = Dj[self.target_col].values

            for c in np.unique(labels_j):
                true_count = np.sum(labels_j == c)
                noisy_count = true_count + self.laplace(1.0 / epsilon2)
                noisy_count = max(noisy_count, 1e-6)

                p_i = noisy_count / count_D_j
                if p_i > 0 and np.isfinite(p_i):
                    subtree_entropy -= p_i * np.log2(p_i)

            tot += subtree_entropy * (count_D_j / N)

        return float(tot)

    #----------------------------
    # DP entropy for numeric split
    #----------------------------
    def find_entropy_split_numeric(self, D, a, N, epsilon2):
        """
        FIND_ENTROPY_SPLIT for a numeric attribute.

        - Sort values of a
        - Consider candidate thresholds at midpoints between distinct values
        - For each threshold t, compute two-way split entropy using DP
        - Return smallest conditional entropy and corresponding threshold
        """
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

            # noisy branch sizes
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
        """
        Recursive core of DP-ID3.

        D       : current subset of data
        A       : list of candidate attributes to split on
        epsilon1: privacy budget at this node
        depth   : current depth (root = 0)
        """
        true_N = len(D)
        N = true_N + self.laplace(1.0 / epsilon1)

        labels = D[self.target_col]
        unique_labels = labels.unique()

        # Base case: max depth, no attributes, or pure labels
        if depth >= self.max_depth or len(A) == 0 or len(unique_labels) == 1:
            return Node(leaf=True, label=self.dp_majority_label(labels, epsilon1))

        # Split privacy for attribute selection
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

        # Choose attribute with smallest conditional entropy
        a_hat = min(G, key=G.get)
        root = Node(key=a_hat)

        # Numeric split
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

        # Categorical split
        remaining = [x for x in A if x != a_hat]
        for v, sub_df in D.groupby(a_hat):
            root.children[v] = self._dp_id3_core(sub_df, remaining, epsilon1, depth + 1)

        root.label = self.dp_majority_label(labels, epsilon1)
        return root

    #----------------------------
    # Public training function
    #----------------------------
    def fit(self, df, target_col):
        """
        Train DP-ID3 tree on a DataFrame.

        df         : pandas DataFrame with features + label
        target_col : name of the column to predict
        """
        self.target_col = target_col
        feature_cols = [c for c in df.columns if c != target_col]
        self.root = self._dp_id3_core(df, feature_cols, self.epsilon1, depth=0)

    #----------------------------
    # Prediction functions
    #----------------------------
    def predict_one(self, record):
        """
        Predict a single record (pandas Series or dict).
        """
        node = self.root
        get = record.get if isinstance(record, dict) else record.__getitem__

        while not node.leaf:
            if node.threshold is not None:
                # Numeric split
                val = float(get(node.key))
                node = node.left if val < node.threshold else node.right
            else:
                # Categorical split
                child = node.children.get(get(node.key))
                if child is None:
                    return node.label
                node = child

        return node.label

    def predict(self, df):
        """
        Predict all rows in a DataFrame.
        """
        return np.array([self.predict_one(row) for _, row in df.iterrows()])
