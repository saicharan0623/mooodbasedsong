import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random
from PIL import Image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# -------------------------------------------------------------------
# 1. Configuration & Setup
# -------------------------------------------------------------------

# Load Spotify credentials from Streamlit secrets
try:
    CLIENT_ID = st.secrets['spotify']['client_id']
    CLIENT_SECRET = st.secrets['spotify']['client_secret']
    REDIRECT_URI = st.secrets['spotify']['redirect_uri']
    
    # Display connection info for debugging
    if st.sidebar.checkbox("Show Debug Info", value=False):
        st.sidebar.write(f"**Redirect URI:** {REDIRECT_URI}")
        st.sidebar.write(f"**Client ID:** {CLIENT_ID[:10]}...")
        
except (FileNotFoundError, KeyError) as e:
    st.error("⚠️ Spotify credentials not found in Streamlit secrets.")
    st.info("""
    **Setup Instructions:**
    
    1. Go to your app settings in Streamlit Cloud
    2. Navigate to the Secrets section
    3. Add the following:
    
    ```toml
    [spotify]
    client_id = "your_spotify_client_id"
    client_secret = "your_spotify_client_secret"
    redirect_uri = "https://mooodbasedsong.streamlit.app"
    ```
    
    4. Make sure the redirect_uri matches what's in your Spotify Dashboard
    """)
    st.stop()

# Define the required scopes for Spotify
SCOPES = "user-modify-playback-state user-read-playback-state user-read-currently-playing"

# Map emotions to Spotify audio features
SPOTIFY_MOOD_MAP = {
    'happy': {'min_valence': 0.7, 'min_energy': 0.7, 'seed_genres': ['happy', 'dance', 'pop']},
    'sad': {'max_valence': 0.3, 'max_energy': 0.4, 'seed_genres': ['sad', 'acoustic', 'blues']},
    'angry': {'max_valence': 0.4, 'min_energy': 0.8, 'seed_genres': ['metal', 'rock', 'punk']},
    'fear': {'max_energy': 0.4, 'max_valence': 0.4, 'seed_genres': ['ambient', 'classical', 'chill']},
    'surprise': {'min_energy': 0.7, 'min_valence': 0.6, 'seed_genres': ['pop', 'electronic', 'party']},
    'disgust': {'max_energy': 0.3, 'seed_genres': ['lo-fi', 'jazz', 'soul']},
    'neutral': {'min_valence': 0.4, 'max_valence': 0.6, 'seed_genres': ['indie', 'pop', 'chill']}
}

# Emoji map for emotions
EMOJI_MAP = {
    'happy': '😊', 'sad': '😢', 'angry': '😠',
    'fear': '😨', 'surprise': '😲', 'disgust': '🤢',
    'neutral': '😐'
}

# Create data directory
os.makedirs('data', exist_ok=True)

# -------------------------------------------------------------------
# 2. Load DeepFace with Error Handling
# -------------------------------------------------------------------

@st.cache_resource
def load_deepface():
    """Load DeepFace model safely"""
    try:
        from deepface import DeepFace
        # Pre-warm the model with a dummy image
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, silent=True)
        except:
            pass  # Expected to fail, just warming up
        return DeepFace
    except Exception as e:
        st.error(f"Failed to load DeepFace: {e}")
        return None

# -------------------------------------------------------------------
# 3. Emotion Detector Class
# -------------------------------------------------------------------

class EmotionDetector:
    def __init__(self, deepface_model):
        self.DeepFace = deepface_model
        self.emotion_history = []

    def detect_emotion_from_frame(self, frame):
        """Detect emotion from a single frame"""
        if self.DeepFace is None:
            return None
            
        try:
            # Ensure frame is in correct format
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            result = self.DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]

            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            confidence = emotions[dominant_emotion]

            return {
                'emotion': dominant_emotion,
                'confidence': confidence / 100.0,
                'all_emotions': {k: v/100.0 for k, v in emotions.items()},
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return None

# -------------------------------------------------------------------
# 4. Mood Tracker Class
# -------------------------------------------------------------------

class MoodTracker:
    def __init__(self, log_file='data/mood_history.csv'):
        self.log_file = log_file
        self.load_data()

    def load_data(self):
        if os.path.exists(self.log_file):
            self.mood_data = pd.read_csv(self.log_file).to_dict('records')
        else:
            self.mood_data = []

    def log_emotion(self, emotion_data, song_data=None):
        log_entry = {
            'timestamp': emotion_data['timestamp'],
            'emotion': emotion_data['emotion'],
            'confidence': emotion_data['confidence'],
            'song_title': song_data['name'] if song_data else None,
            'song_artist': song_data['artists'][0]['name'] if song_data else None,
            'song_url': song_data['external_urls']['spotify'] if song_data else None
        }
        self.mood_data.append(log_entry)
        pd.DataFrame(self.mood_data).to_csv(self.log_file, index=False)
    
    def get_mood_df(self):
        return pd.DataFrame(self.mood_data)

# -------------------------------------------------------------------
# 5. Spotify Client Class
# -------------------------------------------------------------------

class SpotifyClient:
    def __init__(self):
        self.sp_oauth = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope=SCOPES,
            cache_path=None
        )

    def get_auth_url(self):
        return self.sp_oauth.get_authorize_url()

    def get_token(self, auth_code):
        try:
            token_info = self.sp_oauth.get_access_token(auth_code, check_cache=False)
            return token_info
        except Exception as e:
            st.error(f"Error getting Spotify token: {e}")
            return None

    def get_spotify_client(self, token_info):
        return spotipy.Spotify(auth=token_info['access_token'])

    def get_recommendations_for_emotion(self, sp, emotion):
        if emotion not in SPOTIFY_MOOD_MAP:
            emotion = 'neutral'
        
        params = SPOTIFY_MOOD_MAP[emotion]
        
        try:
            recs = sp.recommendations(
                seed_genres=params.get('seed_genres'),
                min_valence=params.get('min_valence'),
                max_valence=params.get('max_valence'),
                min_energy=params.get('min_energy'),
                max_energy=params.get('max_energy'),
                limit=10
            )
            return recs['tracks']
        except Exception as e:
            st.warning(f"Could not get recommendations: {e}")
            return None

    def play_track(self, sp, track_uri):
        try:
            devices = sp.devices()
            if not devices['devices']:
                st.warning("No active Spotify device found. Please open Spotify and start playing.")
                return False
            
            sp.start_playback(uris=[track_uri])
            return True
        except spotipy.exceptions.SpotifyException as e:
            if "NO_ACTIVE_DEVICE" in str(e):
                st.warning("No active Spotify device. Please open Spotify and try again.")
            else:
                st.error(f"Spotify error: {e}")
            return False

# -------------------------------------------------------------------
# 6. Session State Initialization
# -------------------------------------------------------------------

def initialize_session_state():
    """Initialize all necessary keys in Streamlit's session state."""
    defaults = {
        'auth_code': None,
        'token_info': None,
        'sp_client': None,
        'spotify_auth': SpotifyClient(),
        'deepface_model': None,
        'emotion_detector': None,
        'mood_tracker': MoodTracker(),
        'current_emotion': "neutral",
        'last_emotion_change': time.time(),
        'current_song': None,
        'detecting': False,
        'last_detection_time': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -------------------------------------------------------------------
# 7. Spotify Authentication Handler
# -------------------------------------------------------------------

def handle_spotify_auth():
    """Manages the Spotify OAuth login and token retrieval."""
    
    query_params = st.query_params
    if "code" in query_params and not st.session_state.token_info:
        st.session_state.auth_code = query_params["code"]
        st.query_params.clear()

        token_info = st.session_state.spotify_auth.get_token(st.session_state.auth_code)
        if token_info:
            st.session_state.token_info = token_info
            st.session_state.sp_client = st.session_state.spotify_auth.get_spotify_client(token_info)
            st.rerun()

    if not st.session_state.sp_client:
        st.header("🎵 Login to Spotify to Begin")
        st.write("This app will detect your emotions and play music that matches your mood!")
        auth_url = st.session_state.spotify_auth.get_auth_url()
        st.link_button("🔐 Login with Spotify", auth_url)
        st.stop()
    else:
        st.success("✅ Logged in to Spotify!")

# -------------------------------------------------------------------
# 8. Music Update Logic
# -------------------------------------------------------------------

def update_music_for_emotion(emotion):
    """Update Spotify playback based on detected emotion"""
    try:
        sp = st.session_state.sp_client
        if not sp:
            return

        with st.spinner(f"Finding {emotion} music..."):
            tracks = st.session_state.spotify_auth.get_recommendations_for_emotion(sp, emotion)
        
        if tracks:
            selected_track = random.choice(tracks)
            track_uri = selected_track['uri']
            
            # Try to play the track
            success = st.session_state.spotify_auth.play_track(sp, track_uri)
            
            # Store the song regardless of playback success (for display)
            st.session_state.current_song = selected_track
            
            if success:
                st.success(f"🎵 Now playing: {selected_track['name']} by {selected_track['artists'][0]['name']}")
            else:
                st.info(f"💿 Song selected: {selected_track['name']} by {selected_track['artists'][0]['name']}")
                st.warning("⚠️ Could not play automatically. Please open Spotify on any device and press play!")
        else:
            st.warning("Could not get song recommendations. Please try again.")
                
    except Exception as e:
        st.error(f"Error updating music: {e}")

# -------------------------------------------------------------------
# 9. Main App Logic
# -------------------------------------------------------------------

def main_app():
    """Runs the main application logic after login."""
    
    st.header(f"Current Mood: {EMOJI_MAP.get(st.session_state.current_emotion, '😐')} {st.session_state.current_emotion.capitalize()}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Emotion Detection")
        
        # Camera input
        camera_photo = st.camera_input("Take a photo to detect your emotion")
        
        if camera_photo is not None:
            # Load the image
            image = Image.open(camera_photo)
            img_array = np.array(image)
            
            # Convert to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            
            # Detect emotion
            current_time = time.time()
            if current_time - st.session_state.last_detection_time > 3:  # Rate limit
                st.session_state.last_detection_time = current_time
                
                with st.spinner("Analyzing your emotion..."):
                    emotion_data = st.session_state.emotion_detector.detect_emotion_from_frame(img_bgr)
                
                if emotion_data and emotion_data['confidence'] > 0.5:
                    detected_emotion = emotion_data['emotion']
                    confidence = emotion_data['confidence']
                    
                    st.write(f"**Detected:** {EMOJI_MAP.get(detected_emotion, '😐')} {detected_emotion.capitalize()} ({confidence:.1%} confidence)")
                    
                    # Show all emotions
                    with st.expander("See all emotion scores"):
                        for emo, score in sorted(emotion_data['all_emotions'].items(), key=lambda x: x[1], reverse=True):
                            col_emo, col_bar = st.columns([1, 3])
                            with col_emo:
                                st.write(f"**{emo.capitalize()}**")
                            with col_bar:
                                st.progress(float(score))  # Convert to Python float
                                st.caption(f"{score:.1%}")
                    
                    # Update music if emotion changed significantly
                    if detected_emotion != st.session_state.current_emotion:
                        st.session_state.current_emotion = detected_emotion
                        st.session_state.last_emotion_change = current_time
                        
                        # Update music first
                        update_music_for_emotion(detected_emotion)
                        
                        # Log the emotion with the new song
                        st.session_state.mood_tracker.log_emotion(emotion_data, st.session_state.current_song)
                        
                        st.rerun()
                else:
                    st.warning("Could not detect emotion clearly. Please try again with better lighting.")

    with col2:
        st.subheader("🎵 Now Playing")
        
        if st.session_state.current_song:
            track = st.session_state.current_song
            artist = track['artists'][0]['name']
            title = track['name']
            
            if track['album']['images']:
                image_url = track['album']['images'][0]['url']
                st.image(image_url, use_container_width=True)
            
            st.write(f"**{title}**")
            st.write(f"by {artist}")
            
            if 'external_urls' in track:
                st.link_button("🎧 Open in Spotify", track['external_urls']['spotify'])
        else:
            st.info("No song playing yet. Take a photo to detect your mood!")
            
            # Add manual recommendation button
            if st.button("🎲 Get Random Recommendations"):
                update_music_for_emotion(st.session_state.current_emotion)
                st.rerun()
            
            st.caption("💡 Make sure Spotify is open on one of your devices!")
    
    # Mood History Section
    st.markdown("---")
    st.subheader("📊 Mood History")
    
    df = st.session_state.mood_tracker.get_mood_df()
    
    if not df.empty:
        col_data, col_chart = st.columns([2, 1])
        
        with col_data:
            st.dataframe(df.tail(10), use_container_width=True)
        
        with col_chart:
            if len(df) > 0:
                emotion_counts = df['emotion'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 6))
                colors = ['#1DB954', '#FF6B6B', '#FFA500', '#9B59B6', '#3498DB', '#E74C3C', '#95A5A6']
                ax.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%', colors=colors)
                ax.set_title('Emotion Distribution')
                st.pyplot(fig)
    else:
        st.info("No mood data logged yet. Start detecting emotions to build your history!")

# -------------------------------------------------------------------
# 10. Main Execution
# -------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(
        page_title="Mood Music Player", 
        layout="wide",
        page_icon="🎵"
    )
    
    st.title("🎵 Emotion-Based Spotify Player")
    st.write("Let AI detect your mood and play the perfect music! 🎧")
    
    # Initialize session state
    initialize_session_state()
    
    # Load DeepFace model
    if st.session_state.deepface_model is None:
        with st.spinner("Loading AI models... This may take a moment on first load."):
            st.session_state.deepface_model = load_deepface()
            if st.session_state.deepface_model:
                st.session_state.emotion_detector = EmotionDetector(st.session_state.deepface_model)
    
    if st.session_state.deepface_model is None:
        st.error("Failed to load emotion detection models. Please refresh the page.")
        st.stop()
    
    # Run auth flow
    handle_spotify_auth()
    
    # Run main app
    main_app()
