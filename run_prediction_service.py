import warnings
import pandas as pd

from flask import Flask, request, Response
from sklearn.externals import joblib
from src.parameters import *
from flasgger import Swagger
from flasgger.utils import swag_from
from flasgger import LazyString, LazyJSONEncoder

from src.service.service_helper import MusicServiceHelper
from src.logging_app import *

warnings.simplefilter("ignore")
app = Flask(__name__)
app.config["SWAGGER"] = {"title": "Italian Music Prediction Api", "uiversion": 3}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/swagger/",
}

template = dict(
    swaggerUiPrefix=LazyString(lambda: request.environ.get("HTTP_X_SCRIPT_NAME", ""))
)

app.json_encoder = LazyJSONEncoder
swagger = Swagger(app, config=swagger_config, template=template)


@app.route("/predict/music", methods=['POST'])
@swag_from("/src/service/swagger_config.yml")
def predict():
    try:
        json_request = request.json
        prediction_result = music_service_helper.predicts_music_region(json_request, data_frame_genre,
                                                                       data_frame_regions, music_model_features,
                                                                       random_forests_classifier)
        response = Response(status=200)
        response.data = prediction_result
        return response
    except Exception as excep:
        log(excep)
        response = Response(status=400)
        response.data = "Api Exception"
        return response


if __name__ == '__main__':
    """ Prediction service initialization. Loads the model, model features and data to the memory."""
    try:
        random_forests_classifier = joblib.load(PATH_TO_MUSIC_MODEL)

        if random_forests_classifier is None:
            raise Exception("Model not found. Train the model.")

        music_model_features = joblib.load(PATH_TO_MUSIC_MODEL_FEATURES)

        if music_model_features is None:
            raise Exception("Model features not found. Train the model.")

        data_frame_regions = pd.read_csv(PATH_TO_REGIONS_DATA)

        data_frame_genre = pd.read_csv(PATH_TO_GENRE_DATA)

        music_service_helper = MusicServiceHelper()

        app.run(debug=True)
    except Exception as excep:
        log(excep)

