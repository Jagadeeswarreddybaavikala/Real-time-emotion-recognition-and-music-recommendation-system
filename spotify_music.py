import spotipy
from spotipy.oauth2 import SpotifyOAuth



sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="60f927e46b72486ca5f0892290f95b1d",
    client_secret="b65f4dbd234748dba0b5f72301514453",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-modify-playback-state user-read-playback-state"
))

emotion_playlists = {
    'happy': "spotify:playlist:37i9dQZF1DXdPec7aLTmlC",
    'sad': "spotify:playlist:37i9dQZF1DX7qK8ma5wgG1",
    'angry': "spotify:playlist:37i9dQZF1DX1xdVrT4J2rb",
    'neutral': "spotify:playlist:37i9dQZF1DX4Zwz4vEur8j",
    'fear': "spotify:playlist:37i9dQZF1DX4sWSpwq3LiO",
    'disgust': "spotify:playlist:37i9dQZF1DX32NsLKyzScr",
    'surprise': "spotify:playlist:37i9dQZF1DX0SM0LYsmbMT"
}

def play_music_for_emotion(emotion):
    try:
        playlist_uri = emotion_playlists.get(emotion.lower())
        if playlist_uri:
            sp.start_playback(context_uri=playlist_uri)
            print(f"Playing {emotion} playlist on Spotify.")
        else:
            print(f"Playlist not available for emotion: {emotion}")
    except Exception as e:
        print(f"Spotify playback error: {e}")

if __name__ == "__main__":
    emotion = input("Enter your emotion (happy/sad/angry/etc): ")
    play_music_for_emotion(emotion)
