import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace
from datetime import datetime
import os
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import random

# -------------------------------------------------------------------
# 1. Configuration & Setup
# -------------------------------------------------------------------

# Load Spotify credentials from Streamlit secrets
try:
    CLIENT_ID = st.secrets['spotify']['client_id']
    CLIENT_SECRET = st.secrets['spotify']['client_secret']
    REDIRECT_URI = st.secrets['spotify']['redirect_uri']
except FileNotFoundError:
    st.error("`secrets.toml` file not found. Please create it.")
    st.stop()
except KeyError:
    st.error("Spotify credentials not found in `secrets.toml`.")
    st.stop()


# Define the required scopes for Spotify
# We need to control playback and read user's playback state
SCOPES = "user-modify-playback-state user-read-playback-state user-read-currently-playing"

# Map emotions to Spotify audio features
# This is more effective than just searching for "happy songs"
SPOTIFY_MOOD_MAP = {
    'happy': {'min_valence': 0.7, 'min_energy': 0.7, 'seed_genres': ['happy', 'dance', 'pop']},
    'sad': {'max_valence': 0.3, 'max_energy': 0.4, 'seed_genres': ['sad', 'acoustic', 'blues']},
    'angry': {'max_valence': 0.4, 'min_energy': 0.8, 'seed_genres': ['metal', 'rock', 'punk']},
    'fear': {'max_energy': 0.4, 'max_valence': 0.4, 'seed_genres': ['ambient', 'classical', 'chill']},
    'surprise': {'min_energy': 0.7, 'min_valence': 0.6, 'seed_genres': ['pop', 'electronic', 'party']},
    'disgust': {'max_energy': 0.3, 'seed_genres': ['lo-fi', 'jazz', 'soul']},
    'neutral': {'min_valence': 0.4, 'max_valence': 0.6, 'seed_genres': ['indie', 'pop', 'background']}
}

# Create data directory
os.makedirs('data', exist_ok=True)

# -------------------------------------------------------------------
# 2. Emotion Detector Class (From your notebook, slightly adapted)
# -------------------------------------------------------------------

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_history = []

    def detect_emotion_from_frame(self, frame):
        try:
            result = DeepFace.analyze(
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
            # st.error(f"Error in DeepFace: {e}")
            return None

    def draw_emotion_on_frame(self, frame, emotion_data):
        frame_copy = frame.copy()
        if emotion_data:
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            emoji_map = {
                'happy': '😊', 'sad': '😢', 'angry': '😠',
                'fear': '😨', 'surprise': '😲', 'disgust': '🤢',
                'neutral': '😐'
            }
            emoji = emoji_map.get(emotion, '🙂')
            text = f"{emoji} {emotion.upper()} ({confidence:.2f})"
            
            # Draw text background
            cv2.rectangle(frame_copy, (10, 10), (400, 60), (0, 0, 0), -1)
            cv2.putText(
                frame_copy, text, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
        return frame_copy

# -------------------------------------------------------------------
# 3. Mood Tracker Class (From your notebook, adapted for Streamlit)
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
# 4. Spotify Client Class (Replaces YouTubeMusicPlayer)
# -------------------------------------------------------------------

class SpotifyClient:
    def __init__(self):
        self.sp_oauth = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope=SCOPES,
            cache_path=None  # We will manage tokens in session state
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
            emotion = 'neutral' # Default
        
        params = SPOTIFY_MOOD_MAP[emotion]
        
        try:
            recs = sp.recommendations(
                seed_genres=params.get('seed_genres'),
                min_valence=params.get('min_valence'),
                max_valence=params.get('max_valence'),
                min_energy=params.get('min_energy'),
                max_energy=params.get('max_energy'),
                limit=10 # Get 10 recommendations
            )
            return recs['tracks']
        except Exception as e:
            st.warning(f"Could not get recommendations: {e}")
            return None

    def play_track(self, sp, track_uri):
        try:
            # Check for active devices
            devices = sp.devices()
            if not devices['devices']:
                st.warning("No active Spotify device found. Please open Spotify on one of your devices and start playing.")
                return False
            
            # Play the track on the active device
            sp.start_playback(uris=[track_uri])
            st.success("Playing song on your active Spotify device!")
            return True
        except spotipy.exceptions.SpotifyException as e:
            if "Player command failed: NO_ACTIVE_DEVICE" in str(e):
                st.warning("No active Spotify device. Please open Spotify (web, desktop, or mobile) and try again.")
            else:
                st.error(f"Spotify error: {e}")
            return False

# -------------------------------------------------------------------
# 5. Streamlit App Logic
# -------------------------------------------------------------------

def initialize_session_state():
    """Initialize all necessary keys in Streamlit's session state."""
    if 'auth_code' not in st.session_state:
        st.session_state.auth_code = None
    if 'token_info' not in st.session_state:
        st.session_state.token_info = None
    if 'sp_client' not in st.session_state:
        st.session_state.sp_client = None
    if 'spotify_auth' not in st.session_state:
        st.session_state.spotify_auth = SpotifyClient()
    if 'emotion_detector' not in st.session_state:
        st.session_state.emotion_detector = EmotionDetector()
    if 'mood_tracker' not in st.session_state:
        st.session_state.mood_tracker = MoodTracker()
    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = "neutral"
    if 'last_emotion_change' not in st.session_state:
        st.session_state.last_emotion_change = time.time()
    if 'current_song' not in st.session_state:
        st.session_state.current_song = None
    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = False

def handle_spotify_auth():
    """Manages the Spotify OAuth login and token retrieval."""
    
    # Check if we got a code back in the URL query parameters
    query_params = st.query_params
    if "code" in query_params and not st.session_state.token_info:
        st.session_state.auth_code = query_params["code"][0]
        st.query_params.clear() # Clear code from URL

        # Exchange code for token
        token_info = st.session_state.spotify_auth.get_token(st.session_state.auth_code)
        if token_info:
            st.session_state.token_info = token_info
            st.session_state.sp_client = st.session_state.spotify_auth.get_spotify_client(token_info)
            st.rerun() # Rerun to show the main app

    # If not logged in, show login button
    if not st.session_state.sp_client:
        st.header("Login to Spotify to Begin")
        auth_url = st.session_state.spotify_auth.get_auth_url()
        st.link_button("Login with Spotify", auth_url)
        st.stop()
    else:
        st.success("Logged in to Spotify! 🎉")

def main_app():
    """Runs the main application logic after login."""
    
    st.header(f"Current Mood: {st.session_state.current_emotion.capitalize()}")
    
    # Placeholder for the webcam feed and song info
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Webcam Feed")
        webcam_placeholder = st.empty()
        
        # Start/Stop button
        if st.button("Start/Stop Webcam"):
            st.session_state.run_webcam = not st.session_state.run_webcam
            
        if st.session_state.run_webcam:
            webrtc_ctx = webrtc_streamer(
                key="webcam",
                video_transformer_factory=EmotionVideoTransformer,
                async_transform=True,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
        else:
            webcam_placeholder.image("https://via.placeholder.com/640x480.png?text=Webcam+Offline", use_column_width=True)

    with col2:
        st.subheader("Now Playing")
        song_placeholder = st.empty()
        if st.session_state.current_song:
            track = st.session_state.current_song
            artist = track['artists'][0]['name']
            title = track['name']
            image_url = track['album']['images'][0]['url']
            
            song_placeholder.image(image_url, caption=f"{title} by {artist}")
        else:
            song_placeholder.info("No song selected yet. Start the webcam to detect your
             mood!")
    
    # Mood Dashboard
    st.subheader("Mood History")
    df = st.session_state.mood_tracker.get_mood_df()
    if not df.empty:
        st.dataframe(df.tail(10))
        
        st.subheader("Mood Distribution")
        fig, ax = plt.subplots()
        df['emotion'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        st.pyplot(fig)
    else:
        st.info("No mood data logged yet.")


# -------------------------------------------------------------------
# 6. Webcam Video Transformer (The core of streamlit-webrtc)
# -------------------------------------------------------------------

class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.emotion_detector = st.session_state.emotion_detector
        self.last_frame_time = time.time()
        self.last_emotion = "neutral"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Process every N seconds to save resources
        current_time = time.time()
        emotion_data = None
        
        if current_time - self.last_frame_time > 2: # Analyze every 2 seconds
            self.last_frame_time = current_time
            
            emotion_data = self.emotion_detector.detect_emotion_from_frame(img)
            
            if emotion_data and emotion_data['confidence'] > 0.5:
                detected_emotion = emotion_data['emotion']
                self.last_emotion = detected_emotion
                
                # Check if emotion changed and enough time has passed
                time_since_last_change = time.time() - st.session_state.last_emotion_change
                
                if (detected_emotion != st.session_state.current_emotion and 
                    time_since_last_change > 10): # Wait 10s before changing
                    
                    st.session_state.current_emotion = detected_emotion
                    st.session_state.last_emotion_change = time.time()
                    
                    # This is a bit of a hack to run Spotify logic from a thread
                    # A more robust solution would use a separate queue
                    self.update_music(detected_emotion, emotion_data)
        
        # Draw emotion on frame
        frame_with_emotion = self.emotion_detector.draw_emotion_on_frame(img, emotion_data)
        
        return av.VideoFrame.from_ndarray(frame_with_emotion, format="bgr24")

    def update_music(self, emotion, emotion_data):
        """Finds and plays music. Called from the transformer thread."""
        try:
            sp = st.session_state.sp_client
            if not sp:
                return

            tracks = st.session_state.spotify_auth.get_recommendations_for_emotion(sp, emotion)
            
            if tracks:
                selected_track = random.choice(tracks)
                track_uri = selected_track['uri']
                
                # Play the track
                st.session_state.spotify_auth.play_track(sp, track_uri)
                
                # Log and update state
                st.session_state.current_song = selected_track
                st.session_state.mood_tracker.log_emotion(emotion_data, selected_track)
                
        except Exception as e:
            # We are in a thread, so we can't use st.error
            print(f"Error in update_music thread: {e}")


# -------------------------------------------------------------------
# 7. Main Execution
# -------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(page_title="Mood Music Player", layout="wide")
    st.title("🎵 Emotion-Based Spotify Player")
    
    # Initialize session state
    initialize_session_state()
    
    # Run auth flow
    handle_spotify_auth()
    
    # Run main app
    main_app()