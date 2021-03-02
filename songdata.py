
import lyricsgenius as lg # https://github.com/johnwmillr/LyricsGenius

GENIUS_API_TOKEN = 'YOUR API TOKEN HERE'

genius = lg.Genius(GENIUS_API_TOKEN, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)

def request_song_info(artist_name, song_cap):
    """
    Fetches most popular lyrics and title for specified artist.
    """

    # format (title, url)
    song_data = []

    try:
        songs = (genius.search_artist(artist_name, max_songs=song_cap, sort='popularity')).songs

        for song in songs:
            song_data.append({
                'title': song.title,
                'artist': song.artist,
                'lyrics': song.lyrics
            })

        print(f"Songs grabbed: {len(songs)}")

    except:  #  Broad catch which will give us the name of artist and song that threw the exception
        print(f"some exception at {artist_name}: {song_cap}")

    return song_data
