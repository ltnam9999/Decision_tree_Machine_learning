import numpy as np

from core.node import Node


class BaseDecisionTree:

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        """Huấn luyện cây quyết định."""
        if hasattr(X, "columns") and feature_names is None:
            feature_names = list(X.columns)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X phải là ma trận 2 chiều.")
        if len(X) != len(y):
            raise ValueError("X và y phải có cùng số dòng.")

        self.feature_names = feature_names
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        """Xây cây bằng đệ quy."""
        n_samples = X.shape[0]

        if self._should_stop(y, depth, n_samples):
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        best_split = self._best_split(X, y)

        if best_split is None or best_split["gain"] <= 0:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        left_node = self._grow_tree(
            best_split["X_left"],
            best_split["y_left"],
            depth + 1,
        )
        right_node = self._grow_tree(
            best_split["X_right"],
            best_split["y_right"],
            depth + 1,
        )

        return Node(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left_node=left_node,
            right_node=right_node,
        )

    def _should_stop(self, y, depth, n_samples):
        """Điều kiện dừng khi xây cây."""
        reached_max_depth = self.max_depth is not None and depth >= self.max_depth
        too_few_samples = n_samples < self.min_samples_split
        pure_node = self._is_pure(y)

        return reached_max_depth or too_few_samples or pure_node

    def _best_split(self, X, y):
        """Tìm thuộc tính và ngưỡng chia có điểm gain tốt nhất."""
        best_gain = -1
        best_split = None
        n_features = X.shape[1]

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            thresholds = self._candidate_thresholds(feature_values)

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]
                gain = self._calculate_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "gain": gain,
                        "X_left": X[left_mask],
                        "y_left": y_left,
                        "X_right": X[right_mask],
                        "y_right": y_right,
                    }

        return best_split

    def _candidate_thresholds(self, values):
        unique_values = np.unique(values)
        if len(unique_values) <= 1:
            return []

        return (unique_values[:-1] + unique_values[1:]) / 2

    def predict(self, X):
        """Dự đoán cho nhiều dòng dữ liệu."""
        if self.root is None:
            raise ValueError("Cây chưa được huấn luyện. Hãy gọi fit trước.")

        X = np.asarray(X, dtype=float)
        return np.array([self._traverse_tree(row, self.root) for row in X])

    def _traverse_tree(self, row, node):
        """Đi từ gốc xuống lá để lấy kết quả dự đoán."""
        if node.is_leaf_node():
            return node.value

        if row[node.feature_index] <= node.threshold:
            return self._traverse_tree(row, node.left_node)

        return self._traverse_tree(row, node.right_node)

    def print_tree(self, export_path=None):
        if self.root is None:
            raise ValueError("Cây chưa được huấn luyện. Hãy gọi fit trước.")

        lines = []
        self._build_tree_lines(self.root, depth=0, branch_name="Root", lines=lines)
        tree_text = "\n".join(lines)

        print(tree_text)

        if export_path is not None:
            with open(export_path, "w", encoding="utf-8") as file:
                file.write(tree_text)

    def _build_tree_lines(self, node, depth, branch_name, lines):
        """Tạo dạng text thụt lề cho cây."""
        indent = "    " * depth

        if node.is_leaf_node():
            lines.append(f"{indent}{branch_name}: Predict = {node.value}")
            return

        feature_name = self._feature_name(node.feature_index)
        condition = f"{feature_name} <= {node.threshold:.4f}"
        lines.append(f"{indent}{branch_name}: if {condition}")

        self._build_tree_lines(node.left_node, depth + 1, "True", lines)
        self._build_tree_lines(node.right_node, depth + 1, "False", lines)

    def _feature_name(self, feature_index):
        """Lấy tên cột nếu có, nếu không thì dùng dạng feature_index."""
        if self.feature_names is not None:
            return self.feature_names[feature_index]
        return f"feature_{feature_index}"

    def _is_pure(self, y):
        """Mặc định: node thuần khi chỉ có một giá trị y."""
        return len(np.unique(y)) == 1

    def _calculate_gain(self, y_parent, y_left, y_right):
        """Lớp con phải tự cài đặt hàm này."""
        raise NotImplementedError

    def _leaf_value(self, y):
        """Lớp con phải tự cài đặt hàm này."""
        raise NotImplementedError
