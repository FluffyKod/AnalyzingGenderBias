##################################
# LIBRARIES
##################################

import os
import re
from shutil import copyfile

import spacy
from nltk.tokenize import sent_tokenize, word_tokenize

from PIL import Image
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')

##################################
# HELPER FUNCTIONS
##################################

def get_blank_score():
    """
    Returns a blank word score template
    """

    blank_score = {
        'Positive': 0,
        'Negative': 0,
        'score': 0
    }

    return blank_score


def get_unique_words(text, nlp, lemma=True):
    """
    Returns a list with all unique words from text.
    """

    # USING SPACY
    doc = nlp(text)

    # extract words from doc (ignore punctuation but keep Mr. and Mrs.)
    tokens = [
        token for token in doc
        if token.text.isalpha() or token.text in ['Mr.', 'Mrs.']
    ]

    # remove stop wordsd
    tokens = [token for token in tokens if not token.is_stop]

    # lowercase
    tokens = [token.text.lower() for token in tokens]

    # lemmatize
    if lemma:
        tokens = [
            token.lemma_.lower()
            if token.lemma_ != '-PRON-' else token.text
            for token in tokens
        ]

    # get unique worsd
    unique_words = set(tokens)

    # return the word list
    return unique_words


def get_sentences_from_file(directory, filename):
    """
    Reads a file and returns a cleaned text file.
    """

    # open specified file
    with open(os.path.join(directory, filename)) as f:
        # save text into variable
        sentences = f.read().split('\n')

    # return contents
    return sentences


def get_word_list(filename):
    """
    Reads and returns a set of words in a file sepparated by a new line.
    """ 

    with open(filename, 'r') as f:
        words = f.read().split()

    words = set(words)

    return words


def add_gender_dataset(gender_data, filename, gender):
    """
    Expands gender data dictionary with additional data.
    """

    with open(filename) as f:
        data = f.read().lower().split('\n')

    for entry in data:
        entry = entry.lower()
        if not entry in gender_data:
            gender_data[entry] = gender

##################################
# CLASSIFICATION
##################################


def classify_sentence(sentence, classifier):
    """
    Returns the sentiment prediction of a given sentence.
    """

    # Generate prediction
    result = classifier(sentence)

    # Determine prediction
    if result.cats['pos'] > result.cats['neg']:
        prediction = 1
        confidence_score = result.cats['pos']
    else:
        prediction = -1
        confidence_score = result.cats['neg']

    return prediction, confidence_score


##################################
# ANALYZE FUNCTIONS
##################################


def positive_negative(nlp, classifier, sentences):
    """
    Goes through every sentence in a given text and classifies the sentence as positive or negative.
    """

    # save all classifications
    sentence_classifications = []

    # loop through every sentence and get prediction
    for sentence in sentences:

        # get prediction
        prediction, confidence_score = classify_sentence(sentence, classifier)

        # add prediction to list
        sentence_classifications.append((prediction, sentence))

    return sentence_classifications


def feminine_sentence_sentiments(
    nlp, classifier, sentences, gender_data, names_data
):
    """
    Goes through every sentence in a given text and classifies the sentence as feminine positive, feminine negative, masculine positive, masculine negative or neutral. If there are more "feminine" words in sentence compared to "masculine" words, 
    """

    # initialize array to hold sentiment data
    feminine_sentiments = {
        'pos': 0,
        'neg': 0,
        'sents': [],
    }

    masculine_sentiments = {
        'pos': 0,
        'neg': 0,
        'sents': []
    }

    # loop through every sentence and get prediction
    for sentence in sentences:

        # get prediction
        prediction, confidence_score = classify_sentence(sentence, classifier)

        # assign positive variable based on prediction
        label = 'pos' if prediction == 1 else 'neg'

        # get unique words in sentence
        words = word_tokenize(sentence)
        words = [word.lower() for word in words]

        doc = nlp(sentence)

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

        # go through every name
        for ent in doc.ents:
            # check if a person is found
            if ent.label_ == 'PERSON':
                # extract name
                name = ent.text.lower()

                # check if name in data
                if name in names_data:
                    gender = names_data[name]
                    gender_count[gender] += 1

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
# OUTPUT FUNCTIONS
##################################


def generate_documents(feminine_sentiments, masculine_sentiments, text_name, folder_name):
    """
    Generates an HTML document for a visual representation of the vocab.
    """

    def new_tag(tag, indent=1):
        """
        Adds an indent, a tag and a newline.
        """

        return ('\t' * indent + tag + '\n\n')

    # make text_name into filefriendly format
    text_name = text_name.lower()
    text_name = ''.join([c for c in text_name if c.isalpha() or c == ' '])
    text_name = text_name.replace(' ', '_')

    # get header template
    with open('_TEMPLATES/header.txt') as f:
        header = f.read()

    # get footer template
    with open('_TEMPLATES/footer.txt') as f:
        footer = f.read()

    # start a new formatted document
    formatted_document = ''

    # tags
    tags = {
        'pos': '<p class="positive">',
        'neg': '<p class="negative">',
        'neutral': '<p>',
        'end': '</p>'
    }

    # create heading
    formatted_document += new_tag(f'<h1>{text_name}</h1>', 1)

    # create article wrapper
    formatted_document += new_tag('<section id="articles">', 1)

    # create two articles
    for i, sentiments in enumerate([feminine_sentiments, masculine_sentiments]):
        # create new article
        formatted_article = ''
        
        # feminine
        if i == 0:
            formatted_article += new_tag('<div class="feminine">', 2)
            formatted_article += new_tag('<h2>Feminine</h2>', 3)
        
        # masculine
        else:
            formatted_article += new_tag('<div class="masculine">', 2)
            formatted_article += new_tag('<h2>Masculine</h2>', 3)

        # go through every sentence and add parapgraphs
        for sentence_data in sentiments['sents']:
            
            label, text = sentence_data

            if label == 'pos':
                sentence_tag = tags['pos'] + text + tags['end']

            elif label == 'neg':
                sentence_tag = tags['neg'] + text + tags['end']

            elif label == 'neutral':
                sentence_tag = tags['neutral'] + text + tags['end']

            # add sentence to article
            formatted_article += new_tag(sentence_tag, 3)

        # add end tag
        formatted_article += new_tag('</div>')

        # add article to document
        formatted_document += new_tag(formatted_article)

    # add ending
    formatted_document += new_tag('</section>', 1)

    # compose document
    html_document = header + formatted_document + footer

    # write document to html file
    with open(f'{folder_name}/{text_name}.html', 'w') as f:
        f.write(html_document)

    # copy over styles.css in folder
    copyfile('_TEMPLATES/styles.css', f'{folder_name}/styles.css')

    print(f'HTML document saved to {folder_name}/{text_name}.html')


def pie_charts(sentiments):

    if len(sentiments) == 1:
        fig, ax = plt.subplots()
    else:
        fig, axs = plt.subplots(len(sentiments))

    labels = 'Positive', 'Negative'

    i = 0
    for name in sentiments:
        if len(sentiments) > 1:
            ax = axs[i]

        pos = sentiments[name]['f']['pos']
        neg = sentiments[name]['f']['neg']
        total = pos + neg

        if total == 0:
            continue

        sizes = [pos / total, neg / total]

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title(f'Feminine sentiment in {name}')

        i += 1

    plt.show()

def generate_image(sentiments, filename):
    """
    Outputs an image based on the feminine/masculine sentiment of every sentence.
    """

    # define width
    WIDTH = 75

    # Define RGB color codes
    GREEN = (110, 144, 117)
    RED = (163, 62, 100)
    NEUTRAL = (221, 206, 205)

    # transform every value into an RGB value
    pixel_colors = [
        GREEN if s[0] == 'pos' else
        RED if s[0] == 'neg' else
        NEUTRAL for s in sentiments['sents']
    ]

    # pad list with neutral pixels to create a rectangle list
    additional_pixels = 20 - (len(pixel_colors) % WIDTH)
    pixel_colors += [NEUTRAL for i in range(additional_pixels)]

    # split list into 2D array of rows and columns
    pixel_image = []
    current_row = []
    for color in pixel_colors:
        current_row.append(color)

        # row complete
        if len(current_row) == WIDTH:
            # row complete
            pixel_image.append(current_row)

            # reset row
            current_row = []

    # convert pixels into numpy array
    pixel_array = np.array(pixel_image, dtype=np.uint8)

    # Create image using PIL
    new_image = Image.fromarray(pixel_array)
    new_image.save(filename)

    print(f'Image map saved to {filename}')