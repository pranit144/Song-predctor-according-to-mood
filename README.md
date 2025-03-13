# Song Recommendation System

## Overview
This project is a **Flask-based Song Recommendation System** that suggests songs based on the selected artist and mood. It leverages machine learning models to predict song features and find the most similar songs using **Euclidean distance**.

## Features
- **Search for songs by artist**: Retrieve songs from a specific artist.
- **Recommend songs based on mood**: Select an artist and mood to get recommended songs.
- **YouTube Integration**: Each song recommendation includes a button to search for it on YouTube.
- **Machine Learning Model**: Uses an optimized **Random Forest model** trained on song features.
- **Flask Web Interface**: User-friendly web interface for song selection and recommendation.

## Project Structure
```
├── app.py                   # Flask application
├── templates/
│   ├── index.html           # HTML template for the web interface
├── static/
│   ├── styles.css           # CSS file for styling (if applicable)
├── data_moods.csv           # Dataset containing song data
├── optimized_random_forest_model.pkl  # Trained ML model
├── artist_encoder.pkl       # Label encoder for artist names
├── mood_encoder.pkl         # Label encoder for moods
├── scaler.pkl               # Scaler for feature normalization
├── requirements.txt         # Required dependencies
├── README.md                # Project documentation
```

## Installation
### Prerequisites
- Python 3.7+
- Flask
- Scikit-learn
- Pandas
- Joblib

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/song-recommendation.git
   cd song-recommendation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage
1. Select an **artist** from the dropdown menu.
2. Click **Search** to view songs by the selected artist.
3. Select both **artist and mood** to get song recommendations.
4. Click the **YouTube button** to search for a song on YouTube.

## Machine Learning Model
The system uses a **Random Forest model** trained on various song features:
- **Input Features**: Encoded artist and mood.
- **Predicted Features**: Danceability, energy, valence, tempo, etc.
- **Similarity Calculation**: Euclidean distance is used to find the most similar songs.

## Technologies Used
- **Flask** (Backend framework)
- **Scikit-learn** (Machine Learning)
- **Pandas** (Data handling)
- **Joblib** (Model loading)
- **HTML/CSS** (Frontend)

## Future Improvements
- Implement **collaborative filtering** for better recommendations.
- Add **user authentication** and **playlist saving**.
- Improve UI/UX with a **React-based frontend**.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Contributors
- [Your Name](https://github.com/pranit144)

---
Enjoy your personalized song recommendations! 🎵

