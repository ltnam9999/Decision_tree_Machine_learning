import argparse
from pathlib import Path

from core.classifier import DecisionTreeClassifier
from evaluate import evaluate_classification
from modelpre.preprocessing import load_data, train_test_split_custom


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Decision Tree Classifier tự code từ đầu."
    )
    parser.add_argument(
        "--csv",
        default="data/data.csv",
        help="Đường dẫn CSV. Có thể dùng đường dẫn tương đối từ thư mục project.",
    )
    parser.add_argument(
        "--target",
        default="play",
        help="Tên cột nhãn cần phân loại.",
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

    X, y, encoders, feature_names = load_data(
        csv_path=csv_path,
        target_column=target_column,
        task_type="classification",
    )

    X_train, X_test, y_train, y_test = train_test_split_custom(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
    )

    model = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_split=2)
    model.fit(X_train, y_train, feature_names=feature_names)

    y_pred = model.predict(X_test)

    # Giải mã nhãn phân loại để output dễ đọc hơn.
    target_encoder = encoders.get(target_column)
    if target_encoder is not None:
        y_test_print = target_encoder.inverse_transform(y_test)
        y_pred_print = target_encoder.inverse_transform(y_pred)
    else:
        y_test_print = y_test
        y_pred_print = y_pred

    print("Predictions:", list(y_pred_print))
    print("Actual:", list(y_test_print))

    evaluate_classification(
        y_test_print,
        y_pred_print,
        save_path=results_dir / "metrics_summary.json",
    )

    print("\n===== Tree Structure =====")
    model.print_tree(export_path=results_dir / "classification_tree.txt")


if __name__ == "__main__":
    main()
