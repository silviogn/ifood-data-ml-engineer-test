import traceback
import pandas as pd
import warnings
import json

from flask import Flask, request, jsonify, Response
from sklearn.externals import joblib
from src.parameters import *

from flasgger import Swagger
from flasgger.utils import swag_from
from flasgger import LazyString, LazyJSONEncoder

from service.service_helper import *


warnings.simplefilter("ignore")
app = Flask(__name__)
app.config["SWAGGER"] = {"title": "Italian Music Prediction Api", "uiversion": 2}


swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/swagger/",
}

template = dict(
    swaggerUiPrefix=LazyString(lambda: request.environ.get("HTTP_X_SCRIPT_NAME", ""))
)

app.json_encoder = LazyJSONEncoder
swagger = Swagger(app, config=swagger_config, template=template)


@app.route("/predict/music", methods=['POST'])
@swag_from("/service/swagger_config.yml")
def predict():
    try:
        json_request = request.json

        data_frame_prediction = pd.DataFrame(json_request)
        data_frame_prediction = data_frame_prediction.reindex(columns=music_model_features).fillna(0)

        artist_genre_series = data_frame_prediction["artist_genre"]
        data_frame_prediction["artist_genre"] = data_frame_prediction.apply(
            lambda row: get_genre_id(row["artist_genre"], data_frame_genre), axis=1)

        data_frame_prediction["artist_region"] = random_forests_classifier.predict(data_frame_prediction)

        data_frame_prediction["artist_region"] = data_frame_prediction.apply(
            lambda row: get_region_name(row["artist_region"], data_frame_regions), axis=1)

        data_frame_prediction["artist_genre"] = artist_genre_series

        predictions_return = json.dumps(data_frame_prediction.to_dict(orient='records'))

        response = Response(status=200)
        response.data = predictions_return
        return response
    except:
        response = Response(status=400)
        response.data = "Api Exception"
        return response


if __name__ == '__main__':
    try:
        random_forests_classifier = joblib.load(PATH_TO_MUSIC_MODEL)

        if random_forests_classifier is None:
            raise Exception("Model not found. Train the model.")

        print("Model Loaded: {}".format(PATH_TO_MUSIC_MODEL))

        music_model_features = joblib.load(PATH_TO_MUSIC_MODEL_FEATURES)

        if music_model_features is None:
            raise Exception("Model features not found. Train the model.")

        print("Features Loaded: {}".format(PATH_TO_MUSIC_MODEL_FEATURES))

        data_frame_regions = pd.read_csv(PATH_TO_REGIONS_DATA)
        data_frame_genre = pd.read_csv(PATH_TO_GENRE_DATA)

    except Exception as e:
        print(e.args)

    app.run(debug=True)
