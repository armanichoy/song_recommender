import streamlit as st
import os
import pickle
from song_similarity_finder import extract_features, build_song_database, find_similar_songs

def main():
    st.title("Song Similarity Finder")

    # Navigation
    menu = ["Build Database", "Query Similar Songs"]
    choice = st.sidebar.selectbox("Menu", menu)

    database_path = "song_database.pkl"

    if choice == "Build Database":
        st.header("Build Song Database")
        song_folder = st.text_input("Enter the path to the folder containing songs:")
        if st.button("Build Database"):
            if os.path.exists(song_folder):
                with st.spinner("Building database..."):
                    build_song_database(song_folder, database_path)
                st.success("Database built successfully!")
            else:
                st.error("Invalid folder path. Please check and try again.")

    elif choice == "Query Similar Songs":
        st.header("Find Similar Songs")
        query_song = st.text_input("Enter the path to the query song:")
        top_n = st.slider("Number of similar songs to retrieve:", min_value=1, max_value=20, value=10)

        if st.button("Find Similar Songs"):
            if os.path.exists(query_song):
                if os.path.exists(database_path):
                    with st.spinner("Finding similar songs..."):
                        similar_songs = find_similar_songs(query_song, database_path, top_n)
                    if similar_songs:
                        st.write("Top similar songs:")
                        for song, similarity in similar_songs:
                            st.write(f"{song}: Similarity = {similarity:.2f}")
                    else:
                        st.error("No similar songs found. Ensure the database is properly built.")
                else:
                    st.error("Song database not found. Please build the database first.")
            else:
                st.error("Invalid query song path. Please check and try again.")

if __name__ == "__main__":
    main()

