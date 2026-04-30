import numpy as np

from core.base_tree import BaseDecisionTree


class DecisionTreeClassifier(BaseDecisionTree):
    def _entropy(self, y):
        """Tính entropy của một vector nhãn."""
        labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        # Chỉ cộng các xác suất > 0 để tránh log2(0).
        return -np.sum(probabilities * np.log2(probabilities))

    def _calculate_gain(self, y_parent, y_left, y_right):
        """Tính Information Gain sau khi chia node cha thành 2 node con."""
        parent_entropy = self._entropy(y_parent)

        n_parent = len(y_parent)
        weight_left = len(y_left) / n_parent
        weight_right = len(y_right) / n_parent

        child_entropy = (
            weight_left * self._entropy(y_left)
            + weight_right * self._entropy(y_right)
        )

        return parent_entropy - child_entropy

    def _leaf_value(self, y):
        """Lấy nhãn xuất hiện nhiều nhất tại nút lá."""
        labels, counts = np.unique(y, return_counts=True)
        most_common_index = np.argmax(counts)
        return labels[most_common_index]
