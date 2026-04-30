import sys

from src.train_reg import main


# Cau hinh dataset can chay
CSV_PATH = "data/drink_regression.csv"
TARGET_COLUMN = "Amount_ml"
MAX_DEPTH = 5
TEST_SIZE = 0.3


if __name__ == "__main__":
    # Gia lap tham so dong lenh de train_reg.py co the chay bang nut Run.
    sys.argv = [
        "train_reg.py",
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
