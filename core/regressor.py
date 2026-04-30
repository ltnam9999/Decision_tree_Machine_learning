import numpy as np

from core.base_tree import BaseDecisionTree


class DecisionTreeRegressor(BaseDecisionTree):
    def _variance(self, y):
        """Tính phương sai của y."""
        if len(y) == 0:
            return 0
        return np.var(y)

    def _calculate_gain(self, y_parent, y_left, y_right):
        """Tính mức giảm phương sai sau khi chia node."""
        parent_variance = self._variance(y_parent)

        n_parent = len(y_parent)
        weight_left = len(y_left) / n_parent
        weight_right = len(y_right) / n_parent

        child_variance = (
            weight_left * self._variance(y_left)
            + weight_right * self._variance(y_right)
        )

        return parent_variance - child_variance

    def _leaf_value(self, y):
        """Giá trị dự đoán tại nút lá là trung bình y."""
        return float(np.mean(y))

    def _is_pure(self, y):
        """Với hồi quy, coi node thuần nếu phương sai gần bằng 0."""
        return np.var(y) < 1e-12
