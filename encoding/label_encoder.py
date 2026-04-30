import numpy as np
import pandas as pd


class LabelEncoderCustom:

    def __init__(self):
        self.class_to_int = {}
        self.int_to_class = {}

    def fit(self, values):
        """Học bảng ánh xạ từ nhãn chữ sang số nguyên."""
        series = pd.Series(values)

        # Bỏ qua NaN khi học nhãn. NaN sẽ được mã hóa riêng thành -1.
        unique_values = series.dropna().unique()

        self.class_to_int = {
            label: index for index, label in enumerate(unique_values)
        }
        self.int_to_class = {
            index: label for label, index in self.class_to_int.items()
        }
        return self

    def transform(self, values):
        """Chuyển nhãn chữ thành số nguyên."""
        if not self.class_to_int:
            raise ValueError("Encoder chưa được fit.")

        encoded_values = []
        for value in values:
            if pd.isna(value):
                encoded_values.append(-1)
            elif value in self.class_to_int:
                encoded_values.append(self.class_to_int[value])
            else:
                raise ValueError(f"Nhãn mới chưa từng xuất hiện khi fit: {value}")

        return np.array(encoded_values)

    def fit_transform(self, values):
        """Fit encoder rồi transform ngay trên cùng dữ liệu."""
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        """Giải mã số nguyên về nhãn ban đầu."""
        decoded_values = []
        for value in values:
            value = int(value)
            if value == -1:
                decoded_values.append(None)
            else:
                decoded_values.append(self.int_to_class.get(value, None))

        return np.array(decoded_values, dtype=object)

    def classes_(self):
        """Trả về danh sách các nhãn đã học."""
        return list(self.class_to_int.keys())
