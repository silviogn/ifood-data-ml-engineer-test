import warnings

from src.ml.model_induction import MusicModelInduction
from src.data.preprocessing import MusicDataPreprocessing
from src.parameters import *
from src.logging_app import log

warnings.simplefilter("ignore")


def run():
    """Executes the data convertion and model induction."""
    try:
        MusicDataPreprocessing.convert_data()
        MusicModelInduction.build_model(PATH_TO_PROCESSED_DATA, PATH_TO_MUSIC_MODEL, PATH_TO_MUSIC_MODEL_FEATURES)
    except Exception as excep:
        log(excep)


if __name__ == '__main__':
    run()
