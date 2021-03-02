##################################
# INTRODUCTION
##################################

"""

Feelings in this order:

0. anger
1. anticipation
2. disgust
3. fear
4. joy
5. negative
6. positive
7. sadness
8. surprise
9. trust

Vector: [anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust]

"""

##################################
# LIBRARIES
##################################

import string
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from scipy import spatial
from pprint import pprint
from collections import Counter

##################################
# FUNCTIONS
##################################

def load_lexicon():
    """
    Loads the NRC sentiment lexicon as a dictionary with word keys and numpy arrays containing the boolean representation of the words present feelings.
    """

    print('...Loading NRC lexicon.')

    # open data file
    with open('data/NRC-lexicon.txt', 'r') as f:
        data = f.read()

    # clean data
    data = data.replace('\t', ';')
    data = data.split('\n')

    # initilaize lexicon
    lexicon = {}
    current_word = None

    # transform into lists
    for i in range(len(data)):
        # split word data
        try:
            word, feeling, is_present = data[i].split(';')

            # check if new word
            if i % 10 == 0:
                current_word = word
                lexicon[current_word] = []

            # add sentiment to array
            lexicon[current_word].append(int(is_present))
        except:
            continue
        
    # convert lists into numpy arrays
    for word in lexicon:
        lexicon[word] = np.array(lexicon[word])

    # return lexicon
    return lexicon

def clean_data(data):
    """
    Cleans gutenberg text data.
    """

    data = data.replace('_', '') # note to self: italics words
    data = data.replace('\n', ' ')
    data = data.replace('"', '')
    # data = data.replace('”', '')
    # data = data.replace('“', '')

    return data

def load_sentences(filename):
    """
    Reads and cleans data from a file and returns the result as an array of sentences.
    """

    print(f'...Loading data from {filename}.')

    # open and read the contents of a file
    with open(filename) as f:
        data = f.read()

    # clean the data
    data = clean_data(data)

    # split data into sentences
    sentences = sent_tokenize(data)

    # return the data
    return sentences

def get_most_common_words(filename, n=10, filters=[]):
    """
    Reads and extracts all words from a file, counts the words and returns the most common words.
    """

    # open and read the contents of a file
    with open(filename) as f:
        data = f.read()

    # clean the data
    data = clean_data(data)
    data = data.lower()

    words = word_tokenize(data)
    words = [w for w in words if w not in filters]

    word_collection = Counter(words)

    return word_collection.most_common(n)


def get_sentence_sentiment_vector(lexicon, tokens):
    """
    Creates a sentiment vector for a whole sentence.
    """

    # initialize a vector with 10 zeroes
    sentiment_vector = np.zeros(10)

    # go through every word in sentence
    for token in tokens:
        # if word in lexicon sentiment database
        if token in lexicon:
            # add sentiments
            sentiment_vector += lexicon[token]

    # return the results
    return sentiment_vector

def update_token_vectors(token_vectors, sentiment_vector, tokens):
    """
    Updates every tokens sentiment vector with a new sentiment vector.
    """

    # go through every token
    for token in tokens:
        # create empty sentiment vector if not encountered before
        if token not in token_vectors:
            token_vectors[token] = np.zeros(10)

        # add all values from sentiment_vector to the tokens vector
        token_vectors[token] += sentiment_vector
    

def normalize_token_vectors(token_vectors, verbose=True):
    """
    Normalizes every token vector so that every element in a vector sum to 1.
    """

    if verbose:
        print('...Normalizing token vectors.')

    # go through every token
    for token, vector in token_vectors.items():
        # make sure vector is not empty
        if vector.sum() == 0: continue

        # normalize the vector
        token_vectors[token] = vector / vector.sum()

def get_token_vectors_from_sentences(sentences, lexicon, filters=[]):
    """
    Goes through sentences and constructs sentiment vectors for every word in the text.
    """

    # initialize empty data structure to hold all sentiment vectors
    token_vectors = {}

    # go through every sentence
    for sentence in sentences:
        # split into words
        tokens = word_tokenize(sentence)
        tokens = [word.lower() for word in tokens]

        # filter out stopwords and punctuation etc.
        tokens = [t for t in tokens if t not in filters]

        # get sentence sentiment vector
        sentiment_vector = get_sentence_sentiment_vector(lexicon, tokens)

        # update every word in sentence with the sentiment vector
        update_token_vectors(token_vectors, sentiment_vector, tokens)

    # return the result
    return token_vectors

def get_token_vectors_from_text(filename, lexicon, filters=[]):
    """
    Goes through a text and constructs sentiment vectors for every word in the text.
    """

    # get sentences in text
    sentences = load_sentences(filename)

    return get_token_vectors_from_sentences(sentences, lexicon, filters)

def save_token_vectors(filename, token_vectors):
    """
    Saves a dicitonary with token vectors to a file.
    """

    print(f'...Saving token vectors to {filename}.')

    np.save(filename, token_vectors)

def filter_vectors(token_vectors, filters=[]):
    """
    Removes stopwords from the list of token vectors and returns a filtered version.
    """

    # filter token_vectors for stopwords
    token_vectors = {
        t: v for (t, v) in token_vectors.items() if t not in filters
    }


##################################
# VECTOR CALCULATIONS
##################################

def get_vector_from_query(token_vectors, query):
    """
    Checks if queries are present in the token vector list.
    """

    query = query.lower()

    # check that both queries are valid
    if query not in token_vectors:
        print(f'{query} not in token database,')
        return False

    return token_vectors[query]

def get_similarity(token_vectors, query1, query2):
    """
    Returns the similarity between two word vectors.
    """

    # get vectors
    v1 = get_vector_from_query(token_vectors, query1)
    v2 = get_vector_from_query(token_vectors, query2)

    if v1 is False or v2 is False: return 

    # get the similarity
    similarity = 1 - spatial.distance.cosine(v1, v2)

    return similarity

def get_difference(token_vectors, query1, query2):
    """
    Returns the difference between two word vectors.
    """

    # get vectors
    v1 = get_vector_from_query(token_vectors, query1)
    v2 = get_vector_from_query(token_vectors, query2)

    if v1 is False or v2 is False: return 

    # get the difference
    difference = spatial.distance.cosine(v1, v2)

    return difference

def get_most_similar(token_vectors, query, n):
    """
    Calculates the cosine distances to other words and returns the n words with the most similar sentiment vectors.

    Use negative values for n to get most different vectors.
    """

    # get query vector
    query_vector = get_vector_from_query(token_vectors, query)
    if query_vector is False: return

    similarities = []
    for token, vector in token_vectors.items():
        # do not compare with query
        if token == query: continue

        # make sure vector is not empty
        if vector.sum() == 0: continue

        # get the similarity
        similarity = 1 - spatial.distance.cosine(query_vector, vector)

        # add tuple with information
        similarities.append((token, similarity))

    # sort similarities based on similarity score
    similarities.sort(key = lambda x: x[1], reverse=True)

    # make sure n words are present, otherwise take the maximum
    if n > len(token_vectors): n = len(token_vectors)

    # return the n most similar words
    return similarities[:n]

##################################
# MAIN
##################################

def main():
    """
    Handles and runs the script.
    """

    # constants and setup
    FILENAME = 'data/gutenberg/dorian_gray.txt'
    FILTERS = set(stopwords.words('english')) | set(string.punctuation) | {'--'}

    lexicon = load_lexicon()

    # get token vectors
    token_vectors = get_token_vectors_from_text(FILENAME, lexicon, FILTERS)

    # create emotion vectors
    emotion_vectors = {
        t: np.concatenate([v[:5], v[7:]]) for (t, v) in token_vectors.items()
    }

    # create sentiment vectors
    sentiment_vectors = {
        t: v[5:7] for (t, v) in token_vectors.items()
    }

    # create dataframe
    sentiments = ['negative', 'positive']
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    
    emotion_df = pd.DataFrame(data=emotion_vectors.values(), index=emotion_vectors.keys(), columns=emotions)
    print(emotion_df)

    sentiment_df = pd.DataFrame(data=sentiment_vectors.values(), index=sentiment_vectors.keys(), columns=sentiments)
    print(sentiment_df)

    # normalize every vector
    normalize_token_vectors(emotion_vectors)
    pprint(get_most_similar(emotion_vectors, 'man', 10))

    # save token vectors to
    # save_token_vectors('data/vectors.npy', token_vectors)


if __name__ == '__main__':
    main()