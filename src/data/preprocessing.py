import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import LabelEncoder
from flatten_json import flatten



def transform_data(raw_data_path: str, processed_data_path) -> None:

    # Load the data from json format and convert to tabular.
    with open(raw_data_path, "r", encoding="utf-8-sig") as readfile:
        data_music_flattened = (flatten(json.loads(document)) for document in readfile)

        data_frame_music = pd.DataFrame(data_music_flattened)

    # Replaces the None values for Not a Number value (NaN)
    data_frame_music[data_frame_music == 'None'] = np.nan

    # Drop the rows with null/Nan values
    data_frame_music = data_frame_music.dropna()

    # Encodes the values that are not number. Due sci-kit algorithms does not support strings only numeric values
    label_encoder = LabelEncoder()

    data_frame_music['artist_region_raw'] = data_frame_music['artist_region']

    fields_to_encode = ['id_song', 'artist_genre', 'artist_region', 'artist_artist_id', 'artist_name', 'lyrics', 'song']
    for name in fields_to_encode:
        data_frame_music[name] = label_encoder.fit_transform(data_frame_music[name])

    #data_frame_regions = pd.DataFrame(data_frame_music[['artist_region', 'artist_region_raw']].groupby(['artist_region',
     #                                                                                      'artist_region_raw']))
    data_frame_regions = data_frame_music[['artist_region', 'artist_region_raw']].drop_duplicates().to_csv()
    print(data_frame_regions.head())

    # Persists the converted and preprocessed dataset.
    data_frame_music.to_csv(processed_data_path)


