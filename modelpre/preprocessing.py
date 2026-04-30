import numpy as np
import pandas as pd

from encoding.label_encoder import LabelEncoderCustom


def load_data(csv_path, target_column, task_type="classification"):
    data = pd.read_csv(csv_path)

    if target_column not in data.columns:
        raise ValueError(f"Không tìm thấy cột target: {target_column}")

    encoders = {}

    # Mã hóa các cột dạng chữ/object/category/bool thành số nguyên.
    text_columns = data.select_dtypes(include=["object", "category", "bool"]).columns
    for column in text_columns:
        encoder = LabelEncoderCustom()
        data[column] = encoder.fit_transform(data[column])
        encoders[column] = encoder

    # Xử lý giá trị thiếu ở cột số bằng trung bình.
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            mean_value = data[column].mean()
            if pd.isna(mean_value):
                mean_value = 0
            data[column] = data[column].fillna(mean_value)

    X_df = data.drop(columns=[target_column])
    y_series = data[target_column]

    if task_type == "regression":
        y = y_series.to_numpy(dtype=float)
    elif task_type == "classification":
        y = y_series.to_numpy()
    else:
        raise ValueError("task_type phải là 'classification' hoặc 'regression'.")

    X = X_df.to_numpy(dtype=float)
    feature_names = list(X_df.columns)

    return X, y, encoders, feature_names


def train_test_split_custom(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Tự viết hàm chia train/test bằng numpy, không dùng sklearn.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số dòng.")

    n_samples = len(X)
    if n_samples < 2:
        raise ValueError("Cần ít nhất 2 dòng dữ liệu để chia train/test.")

    if 0 < test_size < 1:
        n_test = int(np.ceil(n_samples * test_size))
    else:
        n_test = int(test_size)

    if n_test <= 0 or n_test >= n_samples:
        raise ValueError("test_size phải tạo ra số mẫu test từ 1 đến n_samples - 1.")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
