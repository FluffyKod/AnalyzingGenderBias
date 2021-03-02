##################################
# LIBRARIES
##################################

from neural_methods import *

from prettytable import PrettyTable
from pprint import pprint
from collections import Counter
import pickle

##################################
# CUSTOM PLOT FUNCTION
##################################

def plot_data(sentiments, filename, gender):

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
        ax.set_title(f'Feminine contexts in different books')
    elif gender == 'm':
        ax.set_title(f'Masculine contexts in different books')

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

    # model to be used
    MODEL = 'neural_sentiment_model'

    # folder with data + experiment name
    DIRECTORY = 'data/books'
    EXPERIMENT_NAME = 'BOOKS'

    ####################################################################

    # keep track of all files in corpus
    sentiments = {}

    # get gender word data
    with open('data/gender.pkl', 'rb') as f:
        gender_data = pickle.load(f)

    # add custom names to gender data
    names = {}

    add_gender_dataset(names, 'data/female_names.txt', 'f')
    add_gender_dataset(names, 'data/male_names.txt', 'm')
    add_gender_dataset(names, 'data/custom-female-names.txt', 'f')
    add_gender_dataset(names, 'data/custom-male-names.txt', 'm')
    add_gender_dataset(names, 'data/custom-neutral-words.txt', 'n')

    #  Load saved trained model
    classifier = spacy.load(MODEL)

    # get nlp for word tokenization
    nlp = spacy.load('en_core_web_sm')

    # get filenames sorted by name
    # filenames = sorted(os.listdir(DIRECTORY),
    #                    key=lambda f: int(re.sub('\D', '', f)))

    # get filenames
    filenames = sorted(os.listdir(DIRECTORY))

    # go through every file in corpus
    for filename in filenames:
        print(f'Analyzing {filename}...')

        # load text
        sentences = get_sentences_from_file(DIRECTORY, filename)

        # capture all sentiment data
        feminine_sentiments, masculine_sentiments = feminine_sentence_sentiments(
            nlp, classifier, sentences, gender_data, names)

        # add sentiment scores
        sentiments[filename] = {
            'f': feminine_sentiments,
            'm': masculine_sentiments
        }

    # go through every file in corpus, print out result and generate images for comparisons.

    # create folder to hold data if not already there
    folder_name = f'_EXPERIMENTS/{EXPERIMENT_NAME}'

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for filename in sentiments:

        # extract data into variables for cleaner print
        f_pos = sentiments[filename]['f']["pos"]
        f_neg = sentiments[filename]['f']["neg"]
        m_pos = sentiments[filename]['m']["pos"]
        m_neg = sentiments[filename]['m']["neg"]
    
        # print results
        text_name = filename[:-4]  # remove .txt ending
        header = f'==========|| RESULTS {text_name.upper()} ||=========='
        delimiter = '=' * len(header)

        print()
        print(header)

        # dictionary scores
        print('\nFEMININE')
        print(f' Pos: {f_pos}')
        print(f' Neg: {f_neg}')
        print(f' Total: {f_pos + f_neg}')
        print(f' Score: {f_pos / f_neg if f_neg != 0 else f_pos}')

        print('\nMASCULINE')
        print(f' Pos: {m_pos}')
        print(f' Neg: {m_neg}')
        print(f' Total: {m_pos + m_neg}')
        print(f' Score: {m_pos / m_neg if m_neg != 0 else m_pos}')

        print()
        print(delimiter)

        # generate image from sentiments
        print('\nGenerating images...\n')

        f_output_name = '_EXPERIMENTS/' + EXPERIMENT_NAME + \
            '/' + text_name + '_feminine.png'
        m_output_name = '_EXPERIMENTS/' + EXPERIMENT_NAME + \
            '/' + text_name + '_masculine.png'

        generate_image(sentiments[filename]['f'], f_output_name)
        generate_image(sentiments[filename]['m'], m_output_name)

        print()
        print(delimiter)
        print()

    # create piecharts + barcharts
    plot_name = '_EXPERIMENTS/' + EXPERIMENT_NAME + '/f_plot.png'
    plot_data(sentiments, plot_name, 'f')

    plot_name = '_EXPERIMENTS/' + EXPERIMENT_NAME + '/m_plot.png'
    plot_data(sentiments, plot_name, 'm')

    # save data
    output_name = '_EXPERIMENTS/' + EXPERIMENT_NAME + '/data.pickle'
    with open(output_name, 'wb') as handle:
        pickle.dump(sentiments, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
