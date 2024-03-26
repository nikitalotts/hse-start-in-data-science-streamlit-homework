import os


def set_constants(
) -> None:
    os.environ['DATA_PATH'] = './data/winequalityN.csv'
    os.environ['DATA_TARGET_FEATURE'] = 'quality'
    os.environ['MODEL_WEIGHTS_PATH'] = "./src/model_weights/catboost_classifier_weights.cbm"


if __name__ == "__main__":
    set_constants()