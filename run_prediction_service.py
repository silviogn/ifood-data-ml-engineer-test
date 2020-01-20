import traceback
import pandas as pd
import warnings

from flask import Flask, request, jsonify
from sklearn.externals import joblib
from src.parameters import *

from flasgger import Swagger
from flasgger.utils import swag_from
from flasgger import LazyString, LazyJSONEncoder


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
        print(json_request)
        query_data = pd.get_dummies(pd.DataFrame(json_request))
        query_data = query_data.reindex(columns=music_model_features).fillna(0)
        prediction = list(random_forests_classifier.predict(query_data))
        return jsonify({"predictions:": str(prediction)})
    except Exception as ex:
        return jsonify({'Service Error': traceback.format_exc()})


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

    except Exception as e:
        print(e.args)

    app.run(debug=True)
