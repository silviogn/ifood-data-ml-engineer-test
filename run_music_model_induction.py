import warnings

from src.ml.model_induction import build_model
from src.data.preprocessing import transform_data
from src.parameters import *

warnings.simplefilter("ignore")


def run():
    transform_data(PATH_TO_RAW_DATA, PATH_TO_PROCESSED_DATA)
    build_model(PATH_TO_PROCESSED_DATA, PATH_TO_MUSIC_MODEL, PATH_TO_MUSIC_MODEL_FEATURES)


if __name__ == '__main__':
    run()
