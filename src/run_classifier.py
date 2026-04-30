import sys

from src.train_clf import main


# Cau hinh dataset can chay
CSV_PATH = "data/drink_classification.csv"
TARGET_COLUMN = "Drink"
MAX_DEPTH = 5
TEST_SIZE = 0.3


if __name__ == "__main__":
    # Gia lap tham so dong lenh de train_clf.py co the chay bang nut Run.
    sys.argv = [
        "train_clf.py",
        "--csv",
        CSV_PATH,
        "--target",
        TARGET_COLUMN,
        "--max-depth",
        str(MAX_DEPTH),
        "--test-size",
        str(TEST_SIZE),
    ]
    main()
