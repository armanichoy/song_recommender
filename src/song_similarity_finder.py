import os
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import argparse
import time

# Step 1: Extract Features from a Song
def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
        # MFCCs capture the timbral aspects of the sound, providing a summary of the audio spectrum.
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Extract Chroma features
        # Chroma features represent the energy content of the 12 pitch classes (C, C#, D, etc.).
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # Extract Tempo
        # Tempo provides the estimated beats per minute (BPM) of the song.
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Extract Spectral Contrast
        # Spectral contrast measures the difference in amplitude between peaks and valleys in the sound spectrum.
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # Return extracted features as a dictionary
        return {
            "mfccs": np.mean(mfccs.T, axis=0),  # Average MFCCs across time
            "chroma": np.mean(chroma.T, axis=0),  # Average chroma features across time
            "tempo": tempo,  # Tempo value
            "spectral_contrast": np.mean(spectral_contrast.T, axis=0)  # Average spectral contrast
        }
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Step 2: Save Song Features to Database
def build_song_database(song_folder, database_path):
    database = {}
    start_time = time.time()
    for file_name in os.listdir(song_folder):
        if file_name.endswith(('.mp3', '.wav')):
            file_path = os.path.join(song_folder, file_name)
            print(f"Processing: {file_name}")
            features = extract_features(file_path)
            if features is not None:
                database[file_name] = features
    with open(database_path, 'wb') as f:
        pickle.dump(database, f)
    elapsed_time = time.time() - start_time
    print(f"Database built successfully in {elapsed_time:.2f} seconds!")

# Step 3: Find Similar Songs
def find_similar_songs(query_path, database_path, top_n=10):
    # Load the existing database
    with open(database_path, 'rb') as f:
        database = pickle.load(f)

    # Extract features for the query song
    query_features = extract_features(query_path)
    if query_features is None:
        print("Error: Could not extract features from the query song.")
        return []

    # Create a combined feature vector for the query song
    query_features_vector = np.concatenate((
        query_features['mfccs'],
        query_features['chroma'],
        [query_features['tempo']],
        query_features['spectral_contrast']
    ))

    similarities = []
    for song, features in database.items():
        # Create a combined feature vector for each song in the database
        song_features_vector = np.concatenate((
            features['mfccs'],
            features['chroma'],
            [features['tempo']],
            features['spectral_contrast']
        ))
        # Compute cosine similarity between the query song and database song
        similarity = cosine_similarity([query_features_vector], [song_features_vector])
        similarities.append((song, similarity[0][0]))

    # Sort songs by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Step 4: Command-line Interface
def main():
    parser = argparse.ArgumentParser(description="Song Similarity Finder")
    parser.add_argument("--mode", choices=["build", "query"], required=True, help="Mode: build database or query similar songs")
    parser.add_argument("--song_folder", type=str, help="Path to folder containing songs (for build mode)")
    parser.add_argument("--query_song", type=str, help="Path to the query song (for query mode)")
    parser.add_argument("--database_path", type=str, default="song_database.pkl", help="Path to the song database")
    parser.add_argument("--top_n", type=int, default=10, help="Number of similar songs to retrieve (for query mode)")

    args = parser.parse_args()

    if args.mode == "build":
        if not args.song_folder:
            print("Error: --song_folder is required in build mode.")
            return
        build_song_database(args.song_folder, args.database_path)
    elif args.mode == "query":
        if not args.query_song:
            print("Error: --query_song is required in query mode.")
            return
        similar_songs = find_similar_songs(args.query_song, args.database_path, args.top_n)
        print("Top similar songs:")
        for song, similarity in similar_songs:
            print(f"{song}: Similarity = {similarity:.2f}")

if __name__ == "__main__":
    main()

