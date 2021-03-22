##################################
# LIBRARIES
##################################

from neural_methods import *
from nrc import *

from googletrans import Translator, constants
from pprint import pprint
import random
import os

def translate_lexicon(lexicon, translator, lang, filename, start, end):
    """
    Translates an English word lexicon into another language and keeps the sentiment scores.
    """

    print('Translating lexicon...')

    # total
    total = len(lexicon)
    done = 0

    # go through every word in lexicon and translate
    for word, vector in lexicon.items():

        # if not at start value yet
        if done < start: 
            done += 1
            continue

        # if end record completed
        if done > end: break
        
        # translate the word
        translation = translator.translate(word, src='en', dest=lang)

        # output to file
        with open(filename, 'a') as f:
            word = translation.text
            string_vector = ' '.join(map(str, vector.tolist()))
            line = f'{word};{string_vector}\n'

            f.write(line)

        # print to screen
        print(f"{done}/{total}: {translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")

        done += 1

def main():
    """
    Main function
    """

    LANG = 'da'
    FILENAME = f'data/NRC-lexicon-{LANG}.txt'
    START = 6360
    END = START + 300

    # reset file
    #open(FILENAME, 'w').close()

     # init the Google API translator
    translator = Translator()
    translator = Translator(service_urls=['translate.googleapis.com'])
   
    # get NRC English lexicon
    lexicon = load_lexicon()

    # filter out empty words
    filtered = {word: vector for word, vector in lexicon.items() if not np.array_equal(vector, np.zeros(10))}

    translate_lexicon(filtered, translator, LANG, FILENAME, START, END)


if __name__ == '__main__':
    main()


# get NRC English lexicon
# lexicon = load_lexicon()

# translator = Translator()
# translator = Translator(service_urls=['translate.googleapis.com'])

# print(translator.translate('hello', dest='sv'))

# translate to swedish
#sv_lexicon = translate_lexicon(lexicon, translator, 'sv')


#print(list(sv_lexicon.items())[:5])