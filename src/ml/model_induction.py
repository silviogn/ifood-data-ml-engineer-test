import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


# Scale the data to increase the accuracy
# http://scikit-learn.org/stable/modules/preprocessing.html
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
# Often a models will make some assumptions about the distribution
# or scale of your features. Standardization is a way to make your data
# fit these assumptions and improve the algorithm's performance.
# Scaling is a method for standarization of the data.
# X = scale(X)
def scale_data(x: pd.DataFrame) -> pd.DataFrame:
    standard_scaler = StandardScaler()
    standard_scaler.fit(x)
    return standard_scaler.transform(x)


def search_hyperparameters(x, y):
    #Searching the best hyperparameter in this case the number of classifiers
    param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
                                   170, 180, 190, 200]}

    random_forests_classifier = RandomForestClassifier()

    grid_search_cv = GridSearchCV(random_forests_classifier, param_grid, cv=10)

    grid_search_cv.fit(x, y)

    print("Random Forests Accuracy: {}".format(grid_search_cv.best_score_))

    print("Best Hyperparameters: {}".format(grid_search_cv.best_params_))

    return grid_search_cv.best_params_


def build_model(file: str, model_file: str, model_features_file: str):
    music_data_frame = pd.read_csv(file)

    model_class_data = music_data_frame['artist_region']

    music_data_frame = music_data_frame.drop(['artist_region', 'Unnamed: 0', 'id_song', 'artist_artist_id', 'lyrics', 'song'], axis = 1)

    model_features = list(music_data_frame.columns)

    joblib.dump(model_features, model_features_file)

    model_features_data = scale_data(music_data_frame.values)

    best_hyperparametes = search_hyperparameters(model_features_data, model_class_data)

    random_forests_classifier = RandomForestClassifier(n_estimators=best_hyperparametes['n_estimators'])

    random_forests_classifier.fit(model_features_data, model_class_data)

    model_features_data = pd.DataFrame(model_features_data)

    joblib.dump(random_forests_classifier, model_file)

