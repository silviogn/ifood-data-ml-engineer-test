# Machine Learning Model Server build and execution

The idea of the proposed machine learning model is to predict the Italy region of certain music according to its characteristics. The model was built consuming the Italian Musica Dataset that initially is available in the JSON format. The system is composed of two parts. First, the script that preprocesses the data and built the model and second the service that predicts the regions according to the user requests. The instructions on how to install and execute are described below. 

## Initial Setup

Open the terminal. This guide was executed on Windows 10. Maybe some modifications will be necessary for other operational systems.
All the commands described below should be executed in the terminal.

1. Install the virtualenv if necessary.
    ```
    pip3 install virtualenv
    ```
   
2. Enter into the project root directory.
    ```
    cd .\ifood-data-ml-engineer-test-master\
    ```
    
3. Create a virtual environment.
    ```
    virtualenv venv
    ```
4. Activate the virtual environment.
    ```
    venv\Scripts\activate.bat
    ```

5. Install the requirements of the project.
    ```
   pip install -r requeriments.txt
   ```
 
6. Build automatically the machine learning model.
    ```
   py run_music_model_induction.py
   ```
 
 7. Start the server.
    ```
    py run_prediction_service.py
    ```
    
 8. The Swagger documentation is available at the link below. 
     
    http://127.0.0.1:5000/swagger/
    
    
9. The music prediction service can be consumed as described below.
      
    http://127.0.0.1:5000/predict/music
    
     Method: POST
     
    Body:
    ```json    
   [
    {
        "artist_genre":"pop", 
        "musical_features_acousticness":0.571, 
        "musical_features_valence":0.484, 
        "musical_features_danceability":0.481, 
        "musical_features_duration_ms":262413, 
        "musical_features_loudness":-6.436, 
        "musical_features_energy":0.546, 
        "musical_features_liveness":0.795, 
        "musical_features_tempo":126.021, 
        "musical_features_speechiness":0.0398, 
        "musical_features_instrumentalness":0
    }
   , 
    {
        "artist_genre":"rock", 
        "musical_features_acousticness":0.571, 
        "musical_features_valence":0.484, 
        "musical_features_danceability":0.481, 
        "musical_features_duration_ms":262413, 
        "musical_features_loudness":-6.436, 
        "musical_features_energy":0.546, 
        "musical_features_liveness":0.795, 
        "musical_features_tempo":126.021, 
        "musical_features_speechiness":0.0398, 
        "musical_features_instrumentalness":0 
    }
   ]
   ```
















