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
        MusicModelInduction().build(data_file=PATH_TO_PROCESSED_DATA, model_file=PATH_TO_MUSIC_MODEL,
                                  model_features_file=PATH_TO_MUSIC_MODEL_FEATURES)
        print("Model done!!!")
    except Exception as excep:
        log(excep)


if __name__ == '__main__':
    run()
