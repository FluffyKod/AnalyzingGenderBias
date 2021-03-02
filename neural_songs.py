"""

This script uses a neural network to classify SONGS as feminine positive/negative, masculine positive/negative or neutral.

IMPORTANT: To use this script, you must sign up for Genius API and get your own API token. Enter this API token into songdata.py. If you do not wish to fetch songs from the Genius API, a large set of pre-extracted Taylow Swift songs can be found under data/lyrics-taylor-swift.

"""

##################################
# LIBRARIES
##################################

from songdata import *
from neural_methods import *

from prettytable import PrettyTable
from pprint import pprint
from collections import Counter
import pickle

##################################
# CUSTOM PLOT FUNCTION
##################################

def plot_data(sentiments, artist, filename, gender):

    # generate plots
    fig, ax = plt.subplots()

    labels = [name for name in sentiments]

    x = np.arange(len(labels))
    width = 0.35

    positives = [sentiments[name][gender]['pos'] for name in sentiments]
    negatives = [sentiments[name][gender]['neg'] for name in sentiments]

    rects1 = ax.bar(x - width/2, positives, width,
                    label='Positive', color="#2a9d8f")
    rects2 = ax.bar(x + width/2, negatives, width,
                    label='Negative', color="#e07a5f")


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Occurences')

    if gender == 'f':
        ax.set_title(f'Feminine contexts in songs by {artist}')
    elif gender == 'm':
        ax.set_title(f'Masculine contexts in songs by {artist}')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # save image
    plt.savefig(filename)
    print(f'Plot saved to {filename}.')

    plt.show()

##################################
# MAIN
##################################

def main():
    """
    Performs gender analysis on corpus.
    """

    ##############
    # PARAMETERS #
    ##############

    # The neural network model to be used
    MODEL = 'neural_sentiment_model'

    # the output folder for data
    EXPERIMENT_NAME = 'LADY_GAGA_2'

    # the artist name and number of songs to fetch
    ARTIST = 'Lady Gaga'
    SONG_COUNT = 15
    
    # set the gender of the singer (I, my, mine etc.) and "you-words" (you, your, yours etc.). The words can be found in custom-i.txt and custom-you.txt
    
    # Set 'f' for feminine, 'm' for masculine or None to keep as neutral
    I_GENDER = 'f'
    YOU_GENDER = None

    # Note: You can change/add custom data in the data directory. For example, which female names and male names to be included, or custom neutral words.

    ####################################################################

    # create folder to hold data if not already there
    folder_name = f'_EXPERIMENTS/{EXPERIMENT_NAME}'

    # if folder does not exist, create it
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # get gender word data
    with open('data/gender.pkl', 'rb') as f:
        gender_data = pickle.load(f)
    
    # add custom words to gender data

    if I_GENDER:
        add_gender_dataset(gender_data, 'data/custom-i.txt', I_GENDER)
    
    if YOU_GENDER:
        add_gender_dataset(gender_data, 'data/custom-you.txt', YOU_GENDER)

    # add custom names to gender data
    names = {}

    add_gender_dataset(names, 'data/female_names.txt', 'f')
    add_gender_dataset(names, 'data/male_names.txt', 'm')
    add_gender_dataset(names, 'data/custom-female-names.txt', 'f')
    add_gender_dataset(names, 'data/custom-male-names.txt', 'm')

    # add neutral words
    add_gender_dataset(names, 'data/custom-neutral-words.txt', 'n')

    # keep track of all files in corpus
    sentiments = {}
    analyzed_sentiments = {}

    #  Load saved trained model
    classifier = spacy.load(MODEL)

    # get nlp for word tokenization
    nlp = spacy.load('en_core_web_sm')

    # get song data
    print()
    print('======= FETCHING SONGS =======')
    print()

    song_data = request_song_info(ARTIST, SONG_COUNT)

    print()
    print('======= ANALYZE SONGS =======')
    print()

    # go through every file in songs
    for song in song_data:
        
        title = song['title']
        sentences = song['lyrics'].split('\n')

        print(f'Analyzing {title}...')

        # capture all sentiment data
        feminine_sentiments, masculine_sentiments = feminine_sentence_sentiments(
            nlp, classifier, sentences, gender_data, names)

        # add sentiment scores
        sentiments[title] = {
            'f': feminine_sentiments,
            'm': masculine_sentiments
        }

        # extract data into variables for cleaner print
        f_pos = sentiments[title]['f']["pos"]
        f_neg = sentiments[title]['f']["neg"]
        m_pos = sentiments[title]['m']["pos"]
        m_neg = sentiments[title]['m']["neg"]

        # print results
        header = f'==========|| RESULTS {title.upper()} ||=========='
        delimiter = '=' * len(header)

        print()
        print(header)
        print()

        row_titles = ['Gender', 'Pos', 'Neg', 'Total', 'Score']
        f_row = [
            'F', 
            f_pos, 
            f_neg, 
            f_pos + f_neg, 
            f_pos / f_neg if f_neg != 0 else f_pos
        ]

        m_row = [
            'M', 
            m_pos, 
            m_neg, 
            m_pos + m_neg, 
            m_pos / m_neg if m_neg != 0 else m_pos
        ]

        # create table
        t = PrettyTable(row_titles)
        t.add_row(f_row)
        t.add_row(m_row)

        # print table
        print(t)

        # generate documents from sentiments
        if f_pos + f_neg > 40:
            generate_documents(sentiments[title]['f'], sentiments[title]['m'], title, folder_name)

            analyzed_sentiments[title] = sentiments[title]

    # create barchart
    plot_name = '_EXPERIMENTS/' + EXPERIMENT_NAME + '/f_plot.png'
    plot_data(analyzed_sentiments, ARTIST, plot_name, 'f')

    plot_name = '_EXPERIMENTS/' + EXPERIMENT_NAME + '/m_plot.png'
    plot_data(analyzed_sentiments, ARTIST, plot_name, 'm')

    # save data
    output_name = '_EXPERIMENTS/' + EXPERIMENT_NAME + '/data.pickle'
    with open(output_name, 'wb') as handle:
        pickle.dump(sentiments, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
