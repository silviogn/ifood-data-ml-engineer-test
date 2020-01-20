import pandas as pd


def get_genre_id(genre_text: str, data_frame_genre: pd.DataFrame) -> int:
    return (data_frame_genre[data_frame_genre["artist_genre_raw"] == genre_text]["artist_genre"]).values[0]


def get_region_name(region_id: int, data_frame_regions: pd.DataFrame ) -> str:
    return (data_frame_regions[data_frame_regions["artist_region"] == region_id])["artist_region_raw"].values[0]
