import warnings

from src.ml.model_induction import MusicModelInduction
from src.data.preprocessing import MusicDataPreprocessing
from src.parameters import *

warnings.simplefilter("ignore")


def run():
    try:
        MusicDataPreprocessing.convert_data()
        MusicModelInduction.build_model(PATH_TO_PROCESSED_DATA, PATH_TO_MUSIC_MODEL, PATH_TO_MUSIC_MODEL_FEATURES)
    except Exception as ex:
        print(ex.args)


if __name__ == '__main__':
    run()
