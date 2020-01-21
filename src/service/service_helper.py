import pandas as pd
import json

from sklearn.ensemble import RandomForestClassifier


class MusicServiceHelper:

    def predicts_music_region(self, json_request: json, data_frame_genre: pd.DataFrame,
                              data_frame_regions: pd.DataFrame, music_model_features: list,
                              random_forests_classifier: RandomForestClassifier) -> json:
        """Predicts the region of music to a list of unclassified instances."""
        data_frame_prediction = pd.DataFrame(json_request)
        data_frame_prediction = data_frame_prediction.reindex(columns=music_model_features).fillna(0)

        artist_genre_series = data_frame_prediction["artist_genre"]
        data_frame_prediction["artist_genre"] = data_frame_prediction.apply(
            lambda row: self.__get_genre_id(row["artist_genre"], data_frame_genre), axis=1)

        data_frame_prediction["artist_region"] = random_forests_classifier.predict(data_frame_prediction)

        data_frame_prediction["artist_region"] = data_frame_prediction.apply(
            lambda row: self.__get_region_name(row["artist_region"], data_frame_regions), axis=1)

        data_frame_prediction["artist_genre"] = artist_genre_series

        return json.dumps(data_frame_prediction.to_dict(orient='records'))

    @staticmethod
    def __get_genre_id(genre_text: str, data_frame_genre: pd.DataFrame) -> int:
        return (data_frame_genre[data_frame_genre["artist_genre_raw"] == genre_text]["artist_genre"]).values[0]

    @staticmethod
    def __get_region_name(region_id: int, data_frame_regions: pd.DataFrame) -> str:
        return (data_frame_regions[data_frame_regions["artist_region"] == region_id])["artist_region_raw"].values[0]
