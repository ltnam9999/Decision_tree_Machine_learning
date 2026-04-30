class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left_node=None,
        right_node=None,
        value=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.value = value

    def is_leaf_node(self):
        """Kiểm tra nút hiện tại có phải nút lá không."""
        return self.value is not None
