"""
Music Recommendation System
Maps detected emotions to appropriate songs
Supports 10 emotions with 10 songs each
"""

import json
import random
import os
from typing import Dict, List, Tuple
import pygame
import threading
import time

class MusicRecommendationSystem:
    def __init__(self):
      
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        self.music_database = {
            'Angry': [
                {'title': 'Break Stuff', 'artist': 'Limp Bizkit', 'file': 'angry_1.mp3', 'genre': 'Nu Metal'},
                {'title': 'Bodies', 'artist': 'Drowning Pool', 'file': 'angry_2.mp3', 'genre': 'Alternative Metal'},
                {'title': 'Chop Suey!', 'artist': 'System of a Down', 'file': 'angry_3.mp3', 'genre': 'Alternative Metal'},
                {'title': 'Killing in the Name', 'artist': 'Rage Against the Machine', 'file': 'angry_4.mp3', 'genre': 'Rap Metal'},
                {'title': 'Du Hast', 'artist': 'Rammstein', 'file': 'angry_5.mp3', 'genre': 'Industrial Metal'},
                {'title': 'Freak on a Leash', 'artist': 'Korn', 'file': 'angry_6.mp3', 'genre': 'Nu Metal'},
                {'title': 'B.Y.O.B.', 'artist': 'System of a Down', 'file': 'angry_7.mp3', 'genre': 'Alternative Metal'},
                {'title': 'Down with the Sickness', 'artist': 'Disturbed', 'file': 'angry_8.mp3', 'genre': 'Heavy Metal'},
                {'title': 'Wait and Bleed', 'artist': 'Slipknot', 'file': 'angry_9.mp3', 'genre': 'Nu Metal'},
                {'title': 'Last Resort', 'artist': 'Papa Roach', 'file': 'angry_10.mp3', 'genre': 'Nu Metal'}
            ],
            'Happy': [
                {'title': 'Happy', 'artist': 'Pharrell Williams', 'file': 'happy_1.mp3', 'genre': 'Pop'},
                {'title': 'Uptown Funk', 'artist': 'Mark Ronson ft. Bruno Mars', 'file': 'happy_2.mp3', 'genre': 'Funk'},
                {'title': 'Can\'t Stop the Feeling!', 'artist': 'Justin Timberlake', 'file': 'happy_3.mp3', 'genre': 'Pop'},
                {'title': 'Good as Hell', 'artist': 'Lizzo', 'file': 'happy_4.mp3', 'genre': 'Pop'},
                {'title': 'Walking on Sunshine', 'artist': 'Katrina and the Waves', 'file': 'happy_5.mp3', 'genre': 'Pop Rock'},
                {'title': 'I Gotta Feeling', 'artist': 'The Black Eyed Peas', 'file': 'happy_6.mp3', 'genre': 'Pop'},
                {'title': 'September', 'artist': 'Earth, Wind & Fire', 'file': 'happy_7.mp3', 'genre': 'Funk'},
                {'title': 'Good Vibrations', 'artist': 'The Beach Boys', 'file': 'happy_8.mp3', 'genre': 'Pop'},
                {'title': 'Celebration', 'artist': 'Kool & The Gang', 'file': 'happy_9.mp3', 'genre': 'Funk'},
                {'title': 'Don\'t Worry Be Happy', 'artist': 'Bobby McFerrin', 'file': 'happy_10.mp3', 'genre': 'A Cappella'}
            ],
            'Sad': [
                {'title': 'Mad World', 'artist': 'Gary Jules', 'file': 'sad_1.mp3', 'genre': 'Alternative'},
                {'title': 'Hurt', 'artist': 'Johnny Cash', 'file': 'sad_2.mp3', 'genre': 'Country'},
                {'title': 'Black', 'artist': 'Pearl Jam', 'file': 'sad_3.mp3', 'genre': 'Grunge'},
                {'title': 'Tears in Heaven', 'artist': 'Eric Clapton', 'file': 'sad_4.mp3', 'genre': 'Blues Rock'},
                {'title': 'Nothing Else Matters', 'artist': 'Metallica', 'file': 'sad_5.mp3', 'genre': 'Heavy Metal'},
                {'title': 'Snuff', 'artist': 'Slipknot', 'file': 'sad_6.mp3', 'genre': 'Alternative Metal'},
                {'title': 'Fade to Black', 'artist': 'Metallica', 'file': 'sad_7.mp3', 'genre': 'Heavy Metal'},
                {'title': 'Creep', 'artist': 'Radiohead', 'file': 'sad_8.mp3', 'genre': 'Alternative Rock'},
                {'title': 'Everybody Hurts', 'artist': 'R.E.M.', 'file': 'sad_9.mp3', 'genre': 'Alternative Rock'},
                {'title': 'The Sound of Silence', 'artist': 'Simon & Garfunkel', 'file': 'sad_10.mp3', 'genre': 'Folk Rock'}
            ],
            'Fear': [
                {'title': 'Thriller', 'artist': 'Michael Jackson', 'file': 'fear_1.mp3', 'genre': 'Pop'},
                {'title': 'Psycho Killer', 'artist': 'Talking Heads', 'file': 'fear_2.mp3', 'genre': 'New Wave'},
                {'title': 'Somebody\'s Watching Me', 'artist': 'Rockwell', 'file': 'fear_3.mp3', 'genre': 'Pop'},
                {'title': 'Monster Mash', 'artist': 'Bobby Pickett', 'file': 'fear_4.mp3', 'genre': 'Novelty'},
                {'title': 'Disturbia', 'artist': 'Rihanna', 'file': 'fear_5.mp3', 'genre': 'Pop'},
                {'title': 'Superstition', 'artist': 'Stevie Wonder', 'file': 'fear_6.mp3', 'genre': 'Soul'},
                {'title': 'Ghostbusters', 'artist': 'Ray Parker Jr.', 'file': 'fear_7.mp3', 'genre': 'Pop'},
                {'title': 'Time Warp', 'artist': 'Rocky Horror Picture Show', 'file': 'fear_8.mp3', 'genre': 'Musical'},
                {'title': 'This is Halloween', 'artist': 'Nightmare Before Christmas', 'file': 'fear_9.mp3', 'genre': 'Soundtrack'},
                {'title': 'Werewolves of London', 'artist': 'Warren Zevon', 'file': 'fear_10.mp3', 'genre': 'Rock'}
            ],
            'Surprise': [
                {'title': 'Bohemian Rhapsody', 'artist': 'Queen', 'file': 'surprise_1.mp3', 'genre': 'Rock'},
                {'title': 'Thunderstruck', 'artist': 'AC/DC', 'file': 'surprise_2.mp3', 'genre': 'Hard Rock'},
                {'title': 'Mr. Blue Sky', 'artist': 'Electric Light Orchestra', 'file': 'surprise_3.mp3', 'genre': 'Rock'},
                {'title': 'Come On Eileen', 'artist': 'Dexys Midnight Runners', 'file': 'surprise_4.mp3', 'genre': 'Pop'},
                {'title': 'Mambo No. 5', 'artist': 'Lou Bega', 'file': 'surprise_5.mp3', 'genre': 'Latin Pop'},
                {'title': 'Crazy Train', 'artist': 'Ozzy Osbourne', 'file': 'surprise_6.mp3', 'genre': 'Heavy Metal'},
                {'title': 'Jump', 'artist': 'Van Halen', 'file': 'surprise_7.mp3', 'genre': 'Hard Rock'},
                {'title': 'Sweet Child O\' Mine', 'artist': 'Guns N\' Roses', 'file': 'surprise_8.mp3', 'genre': 'Hard Rock'},
                {'title': 'Livin\' on a Prayer', 'artist': 'Bon Jovi', 'file': 'surprise_9.mp3', 'genre': 'Rock'},
                {'title': 'Eye of the Tiger', 'artist': 'Survivor', 'file': 'surprise_10.mp3', 'genre': 'Rock'}
            ],
            'Disgust': [
                {'title': 'Toxicity', 'artist': 'System of a Down', 'file': 'disgust_1.mp3', 'genre': 'Alternative Metal'},
                {'title': 'Smells Like Teen Spirit', 'artist': 'Nirvana', 'file': 'disgust_2.mp3', 'genre': 'Grunge'},
                {'title': 'Prison Song', 'artist': 'System of a Down', 'file': 'disgust_3.mp3', 'genre': 'Alternative Metal'},
                {'title': 'Closer', 'artist': 'Nine Inch Nails', 'file': 'disgust_4.mp3', 'genre': 'Industrial'},
                {'title': 'Head Like a Hole', 'artist': 'Nine Inch Nails', 'file': 'disgust_5.mp3', 'genre': 'Industrial'},
                {'title': 'Zombie', 'artist': 'The Cranberries', 'file': 'disgust_6.mp3', 'genre': 'Alternative Rock'},
                {'title': 'Man in the Box', 'artist': 'Alice in Chains', 'file': 'disgust_7.mp3', 'genre': 'Grunge'},
                {'title': 'Bulls on Parade', 'artist': 'Rage Against the Machine', 'file': 'disgust_8.mp3', 'genre': 'Rap Metal'},
                {'title': 'Judith', 'artist': 'A Perfect Circle', 'file': 'disgust_9.mp3', 'genre': 'Alternative Metal'},
                {'title': 'The Beautiful People', 'artist': 'Marilyn Manson', 'file': 'disgust_10.mp3', 'genre': 'Industrial Metal'}
            ],
            'Neutral': [
                {'title': 'Weightless', 'artist': 'Marconi Union', 'file': 'neutral_1.mp3', 'genre': 'Ambient'},
                {'title': 'Clair de Lune', 'artist': 'Claude Debussy', 'file': 'neutral_2.mp3', 'genre': 'Classical'},
                {'title': 'Gymnopédie No. 1', 'artist': 'Erik Satie', 'file': 'neutral_3.mp3', 'genre': 'Classical'},
                {'title': 'The Girl from Ipanema', 'artist': 'Stan Getz & João Gilberto', 'file': 'neutral_4.mp3', 'genre': 'Bossa Nova'},
                {'title': 'Fly Me to the Moon', 'artist': 'Frank Sinatra', 'file': 'neutral_5.mp3', 'genre': 'Jazz'},
                {'title': 'Autumn Leaves', 'artist': 'Miles Davis', 'file': 'neutral_6.mp3', 'genre': 'Jazz'},
                {'title': 'Blue in Green', 'artist': 'Miles Davis', 'file': 'neutral_7.mp3', 'genre': 'Jazz'},
                {'title': 'Summertime', 'artist': 'Ella Fitzgerald', 'file': 'neutral_8.mp3', 'genre': 'Jazz'},
                {'title': 'Moon River', 'artist': 'Audrey Hepburn', 'file': 'neutral_9.mp3', 'genre': 'Jazz'},
                {'title': 'Take Five', 'artist': 'Dave Brubeck', 'file': 'neutral_10.mp3', 'genre': 'Jazz'}
            ],
            'Contempt': [
                {'title': 'You\'re So Vain', 'artist': 'Carly Simon', 'file': 'contempt_1.mp3', 'genre': 'Pop'},
                {'title': 'Irreplaceable', 'artist': 'Beyoncé', 'file': 'contempt_2.mp3', 'genre': 'R&B'},
                {'title': 'So What', 'artist': 'P!nk', 'file': 'contempt_3.mp3', 'genre': 'Pop Rock'},
                {'title': 'Since U Been Gone', 'artist': 'Kelly Clarkson', 'file': 'contempt_4.mp3', 'genre': 'Pop Rock'},
                {'title': 'Stronger', 'artist': 'Kelly Clarkson', 'file': 'contempt_5.mp3', 'genre': 'Pop'},
                {'title': 'Fighter', 'artist': 'Christina Aguilera', 'file': 'contempt_6.mp3', 'genre': 'Pop'},
                {'title': 'Confident', 'artist': 'Demi Lovato', 'file': 'contempt_7.mp3', 'genre': 'Pop'},
                {'title': 'Roar', 'artist': 'Katy Perry', 'file': 'contempt_8.mp3', 'genre': 'Pop'},
                {'title': 'Stronger (What Doesn\'t Kill You)', 'artist': 'Kelly Clarkson', 'file': 'contempt_9.mp3', 'genre': 'Pop'},
                {'title': 'Bad Blood', 'artist': 'Taylor Swift', 'file': 'contempt_10.mp3', 'genre': 'Pop'}
            ],
            'Excited': [
                {'title': 'Pump It', 'artist': 'The Black Eyed Peas', 'file': 'excited_1.mp3', 'genre': 'Hip Hop'},
                {'title': 'Crazy in Love', 'artist': 'Beyoncé ft. Jay-Z', 'file': 'excited_2.mp3', 'genre': 'R&B'},
                {'title': 'Can\'t Hold Us', 'artist': 'Macklemore & Ryan Lewis', 'file': 'excited_3.mp3', 'genre': 'Hip Hop'},
                {'title': 'Shut Up and Dance', 'artist': 'Walk the Moon', 'file': 'excited_4.mp3', 'genre': 'Pop Rock'},
                {'title': 'Party Rock Anthem', 'artist': 'LMFAO', 'file': 'excited_5.mp3', 'genre': 'Electronic'},
                {'title': 'Titanium', 'artist': 'David Guetta ft. Sia', 'file': 'excited_6.mp3', 'genre': 'Electronic'},
                {'title': 'Levels', 'artist': 'Avicii', 'file': 'excited_7.mp3', 'genre': 'Electronic'},
                {'title': 'Feel It Still', 'artist': 'Portugal. The Man', 'file': 'excited_8.mp3', 'genre': 'Alternative Rock'},
                {'title': 'High Hopes', 'artist': 'Panic! At The Disco', 'file': 'excited_9.mp3', 'genre': 'Pop Rock'},
                {'title': 'Dynamite', 'artist': 'BTS', 'file': 'excited_10.mp3', 'genre': 'Pop'}
            ],
            'Confused': [
                {'title': 'What\'s Going On', 'artist': 'Marvin Gaye', 'file': 'confused_1.mp3', 'genre': 'Soul'},
                {'title': 'Everybody\'s Talkin\'', 'artist': 'Harry Nilsson', 'file': 'confused_2.mp3', 'genre': 'Pop'},
                {'title': 'Mad World', 'artist': 'Tears for Fears', 'file': 'confused_3.mp3', 'genre': 'New Wave'},
                {'title': 'Losing My Religion', 'artist': 'R.E.M.', 'file': 'confused_4.mp3', 'genre': 'Alternative Rock'},
                {'title': 'Once in a Lifetime', 'artist': 'Talking Heads', 'file': 'confused_5.mp3', 'genre': 'New Wave'},
                {'title': 'How Soon Is Now?', 'artist': 'The Smiths', 'file': 'confused_6.mp3', 'genre': 'Alternative Rock'},
                {'title': 'Brain Damage', 'artist': 'Pink Floyd', 'file': 'confused_7.mp3', 'genre': 'Progressive Rock'},
                {'title': 'Basket Case', 'artist': 'Green Day', 'file': 'confused_8.mp3', 'genre': 'Punk Rock'},
                {'title': 'Semi-Charmed Life', 'artist': 'Third Eye Blind', 'file': 'confused_9.mp3', 'genre': 'Alternative Rock'},
                {'title': 'What I Got', 'artist': 'Sublime', 'file': 'confused_10.mp3', 'genre': 'Ska Punk'}
            ]
        }
        
        self.current_song = None
        self.is_playing = False
        self.songs_folder = "data/songs/"
        
    def get_recommendations(self, emotion: str, num_songs: int = 5) -> List[Dict]:
        """Get song recommendations based on detected emotion"""
        if emotion not in self.music_database:
            emotion = 'Neutral'  
            
        songs = self.music_database[emotion]
        
      
        if num_songs >= len(songs):
            return songs
        else:
            return random.sample(songs, num_songs)
    
    def get_all_songs_for_emotion(self, emotion: str) -> List[Dict]:
        """Get all 10 songs for a specific emotion"""
        return self.music_database.get(emotion, self.music_database['Neutral'])
    
    def play_song(self, song_info: Dict, songs_folder: str = None):
        """Play a song using pygame"""
        if songs_folder is None:
            songs_folder = self.songs_folder
            
        song_path = os.path.join(songs_folder, song_info['file'])
        
        try:
            if os.path.exists(song_path):
                pygame.mixer.music.load(song_path)
                pygame.mixer.music.play()
                self.current_song = song_info
                self.is_playing = True
                print(f"♪ Now playing: {song_info['title']} by {song_info['artist']}")
            else:
                print(f"♪ Would play: {song_info['title']} by {song_info['artist']} ♪")
               
                self.play_notification_sound(song_info)
                
        except Exception as e:
            print(f"❌ Error playing song: {e}")
            self.play_notification_sound(song_info)
    
    def play_notification_sound(self, song_info: Dict):
        """Play a notification sound when actual song file is not available"""
        print(f"♪ Would play: {song_info['title']} by {song_info['artist']} ♪")
       
        
    def stop_song(self):
        """Stop the currently playing song"""
        pygame.mixer.music.stop()
        self.is_playing = False
        self.current_song = None
        print("🔇 Music stopped")
    
    def set_volume(self, volume: float):
        """Set the volume (0.0 to 1.0)"""
        pygame.mixer.music.set_volume(volume)
    
    def get_emotion_stats(self) -> Dict:
        """Get statistics about the music database"""
        stats = {}
        for emotion, songs in self.music_database.items():
            stats[emotion] = {
                'song_count': len(songs),
                'genres': list(set([song['genre'] for song in songs]))
            }
        return stats

def demo_music_system():
    """Demonstrate the music recommendation system"""
    music_system = MusicRecommendationSystem()
    
    print("🎵 Music Recommendation System Demo ��")
    print(f"Total emotions supported: {len(music_system.music_database)}")
    
   
    stats = music_system.get_emotion_stats()
    for emotion, stat in stats.items():
        print(f"{emotion}: {stat['song_count']} songs, Genres: {', '.join(stat['genres'][:3])}")
    
    
    print("\n=== Sample Recommendations ===")
    for emotion in ['Happy', 'Sad', 'Angry']:
        recommendations = music_system.get_recommendations(emotion, 3)
        print(f"\n{emotion} recommendations:")
        for song in recommendations:
            print(f"  ♪ {song['title']} by {song['artist']} ({song['genre']})")

if __name__ == "__main__":
    demo_music_system()
