import argparse
from pathlib import Path

import numpy as np

from core.regressor import DecisionTreeRegressor
from src.evaluate import evaluate_regression
from modelpre.preprocessing import load_data, train_test_split_custom


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Decision Tree Regressor tự code từ đầu."
    )
    parser.add_argument(
        "--csv",
        default="data/data.csv",
        help="Đường dẫn CSV. Có thể dùng đường dẫn tương đối từ thư mục project.",
    )
    parser.add_argument(
        "--target",
        default="score",
        help="Tên cột giá trị cần dự đoán.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Tỉ lệ dữ liệu test.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Độ sâu tối đa của cây.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = project_dir / csv_path

    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)

    target_column = args.target

    X, y, _, feature_names = load_data(
        csv_path=csv_path,
        target_column=target_column,
        task_type="regression",
    )

    X_train, X_test, y_train, y_test = train_test_split_custom(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
    )

    model = DecisionTreeRegressor(max_depth=args.max_depth, min_samples_split=2)
    model.fit(X_train, y_train, feature_names=feature_names)

    y_pred = model.predict(X_test)

    print("Predictions:", list(np.round(y_pred, 2)))
    print("Actual:", list(np.round(y_test, 2)))

    evaluate_regression(
        y_test,
        y_pred,
        save_path=results_dir / "metrics_summary.json",
    )

    print("\n===== Tree Structure =====")
    model.print_tree(export_path=results_dir / "regression_tree.txt")


if __name__ == "__main__":
    main()
