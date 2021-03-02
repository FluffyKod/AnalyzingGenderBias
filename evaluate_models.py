##################################
# LIBRARIES
##################################

from neural_methods import *
from neural_books import *
from sentiment_books import *

##################################
# Load testing data
##################################

def load_testing_data(filename):
    """
    Loads testing data and returns an array with sentence + classification.
    """

    with open(filename, 'r') as f:
        data = f.read().split('\n')

    testing_data = []

    for entry in data:
        info = entry.split('\t')

        try:
            testing_data.append((info[0], info[1]))
        except:
            continue

    return testing_data


def main():
    """
    Goes through all classified sentences and evaluates the two methods.
    """ 

    #  Load saved trained model
    classifier = spacy.load('neural_sentiment_model')

    # get NRC lexicon
    lexicon = load_lexicon()

    data_sets = ['testing_data/imdb_labelled.txt', 'testing_data/amazon_cells_labelled.txt', 'testing_data/yelp_labelled.txt']

    for data_set in data_sets:

        testing_data = load_testing_data(data_set)

        correct = {
            'lexicon': 0,
            'lex_neg': 0,
            'network': 0
        }

        total = len(testing_data)

        for entry in testing_data:
            sentence = entry[0]
            classification = int(entry[1])

            words = word_tokenize(sentence)
            words = [word.lower() for word in words]

            # network (returns 0 for negative, 1 for positive)
            n_prediction, confidence = classify_sentence(sentence, classifier)

            # lexicon (returns 'neg' for negative, 'pos' for positive)
            l_prediction = sentiment_prediction(words, lexicon)

            # check if network was correct
            if n_prediction == -1 and classification == 0:
                correct['network'] += 1

            if n_prediction == 1 and classification == 1:
                correct['network'] += 1

            # check if lexicon was correct
            if l_prediction == 'pos' and classification == 1:
                correct['lexicon'] += 1

            if l_prediction == 'neg' and classification == 0:
                correct['lexicon'] += 1

            x_prediction = sentiment_prediction(words, lexicon, default_neg=True)

            # check if lexicon was correct
            if x_prediction == 'pos' and classification == 1:
                correct['lex_neg'] += 1

            if x_prediction == 'neg' and classification == 0:
                correct['lex_neg'] += 1

        # print out the results
        n_percentage = correct['network'] / total
        l_percentage = correct['lexicon'] / total
        x_percentage = correct['lex_neg'] / total

        print(f'RESULTS FOR {data_set}')
        print(f'Network: {correct["network"]}/{total} = {n_percentage}')
        print(f'Lexicon: {correct["lexicon"]}/{total} = {l_percentage}')
        print(f'Lex neg: {correct["lex_neg"]}/{total} = {x_percentage}')
        print()

if __name__ == '__main__':
    main()