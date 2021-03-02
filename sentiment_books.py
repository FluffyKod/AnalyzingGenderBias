##################################
# LIBRARIES
##################################

from neural_methods import *
from nrc import *

from prettytable import PrettyTable
from pprint import pprint
from collections import Counter
import pickle

##################################
# CUSTOM PREDICTION
##################################

def sentiment_prediction(words, lexicon, default_neg=False):
    """
    Goes through a sentence and classifies as positive or negative or neutral depedning on the number of positive/negative words in sentence.
    """

    sentiments = {
        'n': 0,
        'p': 0
    }

    for word in words:

        if word in lexicon:
            # negative
            if lexicon[word][5] == 1:
                sentiments['n'] += 1

            # positive
            if lexicon[word][6] == 1:
                sentiments['p'] += 1

    # determine positivity
    prediction = 'neutral' if sentiments['p'] == sentiments['n'] else 'pos' if sentiments['p'] > sentiments['n'] else 'neg'

    # default to negative
    if default_neg:
        if prediction == 'neutral':
            prediction = 'neg'

    return prediction
    

##################################
# CUSTOM GENDER SENTIMENTS
##################################

def get_gender_sentiments(
    lexicon, sentences, gender_data, names_data, default_neg
):
    """
    Goes through every sentence in a given text and classifies the sentence as feminine positive, feminine negative, masculine positive, masculine negative or neutral based on words contained in the NRC lexicon.
    """

    # initialize array to hold sentiment data
    feminine_sentiments = {
        'pos': 0,
        'neg': 0,
        'neutral': 0,
        'sents': [],
    }

    masculine_sentiments = {
        'pos': 0,
        'neg': 0,
        'neutral': 0,
        'sents': []
    }

    # loop through every sentence and get prediction
    for sentence in sentences:

        if sentence == '': continue

        # get unique words in sentence
        words = word_tokenize(sentence)
        words = [word.lower() for word in words]

        # get prediction (pos, neg or neutral)
        label = sentiment_prediction(words, lexicon, default_neg)

        # count feminine vs. masculine words in sentence
        gender_count = {
            'f': 0,
            'n': 0,
            'm': 0
        }

        # go through every wordd
        for word in words:
            if word in gender_data:
                gender = gender_data[word]
                gender_count[gender] += 1

            # add name
            if word in names_data:
                gender_count[names_data[word]] += 1

        # update feminine and masculine sentiments
        if gender_count['f'] > 0:
            # feminine word was present
            feminine_sentiments[label] += gender_count['f']
            feminine_sentiments['sents'].append((label, sentence))
        else:
            # no feminine mention, set as neutral
            feminine_sentiments['sents'].append(('neutral', sentence))

        if gender_count['m'] > 0:
            # masculine word was present
            masculine_sentiments[label] += gender_count['m']
            masculine_sentiments['sents'].append((label, sentence))
        else:
            # no masculine mention, set as neutral
            masculine_sentiments['sents'].append(('neutral', sentence))

    # return results
    return feminine_sentiments, masculine_sentiments

##################################
# CUSTOM PLOT FUNCTION
##################################

def plot_data(sentiments, filename, gender):

    # generate plots
    fig, ax = plt.subplots()

    labels = [name for name in sentiments]
    labels = [label[:-4].replace('_', ' ').title() for label in labels]

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
        ax.set_title(f'Feminine contexts in different books using NRC lexicon')
    elif gender == 'm':
        ax.set_title(f'Masculine contexts in different books using NRC lexicon')

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

    # folder with data + experiment name
    DIRECTORY = 'data/books'
    EXPERIMENT_NAME = 'BOOKS_SENTIMENT_L'
    DEFAULT_NEG = False

    ####################################################################

    # keep track of all files in corpus
    sentiments = {}

    # get NRC lexicon
    lexicon = load_lexicon()
    english_words = get_word_list('data/english_10000.txt')

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

    loop_names = names.copy()

    # remove english words from names
    for name in loop_names.keys():
        if name in english_words:
            del names[name]

    # get filenames
    filenames = sorted(os.listdir(DIRECTORY))

    # go through every file in corpus
    for filename in filenames:
        print(f'Analyzing {filename}...')

        # load text
        sentences = get_sentences_from_file(DIRECTORY, filename)

        # capture all sentiment data
        feminine_sentiments, masculine_sentiments = get_gender_sentiments(
            lexicon, sentences, gender_data, names, DEFAULT_NEG)

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