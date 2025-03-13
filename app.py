from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

# Load the trained model
model_path = r'optimized_random_forest_model.pkl'
label_encoder_artist_path = r'artist_encoder.pkl'
label_encoder_mood_path = r'mood_encoder.pkl'
scaler_path = r'scaler.pkl'
data_path = r'data_moods.csv'

# Load all necessary files
try:
    model = joblib.load(model_path)
    label_encoder_artist = joblib.load(label_encoder_artist_path)
    label_encoder_mood = joblib.load(label_encoder_mood_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(data_path)
except FileNotFoundError as e:
    raise FileNotFoundError(f"File not found: {e.filename}")

# Get unique artist and mood values for dropdown
artists = df['artist'].unique()
moods = df['mood'].unique()

# Sort artists alphabetically for better UX
sorted_artists = sorted(artists)


def get_songs_by_artist(artist_input):
    """Retrieve songs by the selected artist."""
    artist_songs = df[df['artist'] == artist_input].copy()
    artist_songs['YouTube'] = artist_songs['name'].apply(
        lambda song: f"<button onclick=\"window.open('https://www.youtube.com/results?search_query={song}', '_blank')\">Search on YouTube</button>"
    )
    return artist_songs[['name', 'album', 'artist', 'popularity', 'YouTube']]


def recommend_songs_based_on_model(artist_input, mood_input):
    """Recommend songs based on the selected artist and mood."""
    artist_encoded = label_encoder_artist.transform([artist_input])[0]
    mood_encoded = label_encoder_mood.transform([mood_input])[0]

    input_features = [[artist_encoded, mood_encoded]]
    predicted_features = model.predict(input_features)[0]
    predicted_features = scaler.inverse_transform([predicted_features])[0]

    predicted_df = pd.DataFrame([predicted_features], columns=[
        'danceability', 'acousticness', 'energy',
        'instrumentalness', 'liveness', 'valence',
        'loudness', 'speechiness', 'tempo', 'key',
        'time_signature'
    ])

    distances = euclidean_distances(
        df[['danceability', 'acousticness', 'energy', 'instrumentalness',
            'liveness', 'valence', 'loudness', 'speechiness',
            'tempo', 'key', 'time_signature']],
        predicted_df
    )

    df['distance'] = distances
    sorted_df = df.sort_values(by='distance').copy()

    sorted_df['YouTube'] = sorted_df['name'].apply(
        lambda song: f"<button onclick=\"window.open('https://www.youtube.com/results?search_query={song}', '_blank')\">Search on YouTube</button>"
    )
    return sorted_df[['name', 'album', 'artist', 'popularity', 'YouTube']].head(5)


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main page and handle form submissions."""
    artist_choice = None
    mood_choice = None
    artist_songs = None
    recommended_songs = None

    if request.method == 'POST':
        artist_choice = request.form.get('artist')
        mood_choice = request.form.get('mood')

        if artist_choice:
            artist_songs = get_songs_by_artist(artist_choice)

        if artist_choice and mood_choice:
            recommended_songs = recommend_songs_based_on_model(artist_choice, mood_choice)

    # Convert DataFrames to HTML tables
    artist_songs_html = artist_songs.to_html(classes='table table-striped', index=False, escape=False) if artist_songs is not None else None
    recommended_songs_html = recommended_songs.to_html(classes='table table-striped', index=False, escape=False) if recommended_songs is not None else None

    return render_template(
        'index.html',
        artists=sorted_artists,
        moods=moods,
        artist_songs_html=artist_songs_html,
        recommended_songs_html=recommended_songs_html,
        artist_choice=artist_choice,
        mood_choice=mood_choice
    )


if __name__ == '__main__':
    app.run(debug=True)
