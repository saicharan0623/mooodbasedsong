import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os, time, random
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from PIL import Image

# -------------------------------------------------------------------
# 1️⃣ CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(page_title="Mood Music Player", page_icon="🎵", layout="wide")

# Load Spotify credentials from Streamlit secrets
try:
    CLIENT_ID = st.secrets["spotify"]["client_id"]
    CLIENT_SECRET = st.secrets["spotify"]["client_secret"]
    REDIRECT_URI = st.secrets["spotify"]["redirect_uri"]
except Exception:
    st.error("⚠️ Missing Spotify credentials. Add them in Streamlit Secrets.")
    st.stop()

SCOPES = "user-read-currently-playing user-read-playback-state user-top-read user-modify-playback-state"

SPOTIFY_MOOD_MAP = {
    "happy": {"energy": 0.8, "valence": 0.8, "genres": ["pop", "dance", "happy"]},
    'sad': {'min_valence': 0.7, 'min_energy': 0.7, 'seed_genres': ['happy', 'dance', 'pop']},
    "angry": {"energy": 0.9, "valence": 0.3, "genres": ["rock", "metal", "hard-rock"]},
    "fear": {"energy": 0.4, "valence": 0.3, "genres": ["ambient", "chill", "sleep"]},
    "surprise": {"energy": 0.7, "valence": 0.7, "genres": ["edm", "party", "pop"]},
    "disgust": {"energy": 0.4, "valence": 0.3, "genres": ["jazz", "soul", "blues"]},
    "neutral": {"energy": 0.5, "valence": 0.5, "genres": ["indie", "pop", "chill"]},
}

EMOJI_MAP = {
    "happy": "😊", "sad": "😢", "angry": "😠",
    "fear": "😨", "surprise": "😲", "disgust": "🤢", "neutral": "😐"
}

os.makedirs("data", exist_ok=True)

# -------------------------------------------------------------------
# 2️⃣ LOAD DEEPFACE
# -------------------------------------------------------------------
@st.cache_resource
def load_deepface():
    from deepface import DeepFace
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False, silent=True)
    except Exception:
        pass
    return DeepFace

DeepFace = load_deepface()

# -------------------------------------------------------------------
# 3️⃣ HELPER CLASSES
# -------------------------------------------------------------------

class EmotionDetector:
    def __init__(self, model):
        self.model = model

    def detect(self, frame):
        try:
            result = self.model.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)
            if isinstance(result, list): result = result[0]
            dominant = result["dominant_emotion"]
            return dominant, result["emotion"][dominant] / 100, result["emotion"]
        except Exception:
            return None, 0, {}

class SpotifyClient:
    def __init__(self):
        self.oauth = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope=SCOPES,
            cache_path=None
        )

    def get_auth_url(self):
        return self.oauth.get_authorize_url()

    def get_token(self, code):
        return self.oauth.get_access_token(code, check_cache=False)

    def get_client(self, token_info):
        return spotipy.Spotify(auth=token_info["access_token"])

    def recommend_by_mood(self, sp, emotion):
        mood = SPOTIFY_MOOD_MAP.get(emotion, SPOTIFY_MOOD_MAP["neutral"])
        try:
            recs = sp.recommendations(
                seed_genres=mood["genres"],
                limit=10,
                target_valence=mood["valence"],
                target_energy=mood["energy"]
            )
            if recs and recs.get("tracks"):
                return recs["tracks"]
        except Exception:
            pass
        # Fallback: search
        query = random.choice(mood["genres"])
        results = sp.search(q=query, type="track", limit=10)
        return results.get("tracks", {}).get("items", [])

    def play_track(self, sp, uri):
        try:
            devices = sp.devices()
            if not devices["devices"]:
                st.warning("⚠️ No active Spotify device found. Open Spotify and play something once.")
                return False
            sp.start_playback(uris=[uri])
            return True
        except Exception as e:
            st.warning(f"Could not play automatically: {e}")
            return False

# -------------------------------------------------------------------
# 4️⃣ SESSION STATE INIT
# -------------------------------------------------------------------
if "spotify_client" not in st.session_state:
    st.session_state.spotify_auth = SpotifyClient()
    st.session_state.spotify_client = None
    st.session_state.token_info = None
    st.session_state.current_emotion = "neutral"
    st.session_state.current_song = None
    st.session_state.detector = EmotionDetector(DeepFace)

# -------------------------------------------------------------------
# 5️⃣ AUTH HANDLER
# -------------------------------------------------------------------
query_params = st.experimental_get_query_params()
if "code" in query_params and not st.session_state.token_info:
    code = query_params["code"][0]
    st.experimental_set_query_params()  # clear
    token_info = st.session_state.spotify_auth.get_token(code)
    if token_info:
        st.session_state.token_info = token_info
        st.session_state.spotify_client = st.session_state.spotify_auth.get_client(token_info)
        st.rerun()

if not st.session_state.spotify_client:
    st.header("🎵 Login to Spotify")
    st.write("Authorize the app to control playback and get music recommendations.")
    auth_url = st.session_state.spotify_auth.get_auth_url()
    st.link_button("🔐 Login with Spotify", auth_url)
    st.stop()
else:
    st.success("✅ Logged in to Spotify!")

# -------------------------------------------------------------------
# 6️⃣ MAIN APP LOGIC
# -------------------------------------------------------------------
st.header(f"Your Current Mood: {EMOJI_MAP[st.session_state.current_emotion]} {st.session_state.current_emotion.capitalize()}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Detect Your Emotion")
    camera_photo = st.camera_input("Take a photo to analyze your mood")

    if camera_photo:
        img = Image.open(camera_photo)
        frame = np.array(img)
        emo, conf, all_scores = st.session_state.detector.detect(frame)

        if emo:
            st.write(f"**Detected Emotion:** {EMOJI_MAP[emo]} {emo.capitalize()} ({conf:.0%} confidence)")
            st.session_state.current_emotion = emo

            with st.expander("See all emotions"):
                for e, s in all_scores.items():
                    st.write(f"{e.capitalize()}: {s:.1f}%")

            sp = st.session_state.spotify_client
            tracks = st.session_state.spotify_auth.recommend_by_mood(sp, emo)
            if tracks:
                song = random.choice(tracks)
                st.session_state.current_song = song
                st.image(song["album"]["images"][0]["url"], use_container_width=True)
                st.write(f"**{song['name']}** by {song['artists'][0]['name']}")
                st.link_button("🎧 Open in Spotify", song["external_urls"]["spotify"])
                st.session_state.spotify_auth.play_track(sp, song["uri"])
            else:
                st.warning("No tracks found for this mood.")
        else:
            st.warning("😕 Could not detect emotion. Try again with better lighting.")

with col2:
    st.subheader("🎶 Now Playing")
    if st.session_state.current_song:
        song = st.session_state.current_song
        st.image(song["album"]["images"][0]["url"], use_container_width=True)
        st.write(f"**{song['name']}**")
        st.write(f"by {song['artists'][0]['name']}")
        st.link_button("🎧 Open in Spotify", song["external_urls"]["spotify"])
    else:
        st.info("No song yet — take a photo to detect your mood!")

