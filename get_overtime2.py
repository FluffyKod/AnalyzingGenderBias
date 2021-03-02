##################################
# LIBRARIES
##################################

from nrc import *
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

##################################
# PARAMETERS
##################################

START_YEAR = 1800
END_YEAR = 2000
SAMPLE_SIZE = 50

BOOK_FILENAME = 'data/book_lexicon.txt'
DATA_DIRECTORY = 'results/period_2/'

FILTERS = set(stopwords.words('english')) | set(string.punctuation) | {'--', '“', '”', '’'}

##################################
# FUNCTIONS
##################################

def load_book_database(filename):
    """
    Retrieves a dicitionary with books sorted by the year they were published.
    """

    print(f'...Loading book-year database from {filename}.')

    # initialize empty dictionary
    books = {}

    # read the data
    with open(filename, 'r') as f:
        data = f.read()

    # get records which are on sepparate lines
    records = data.split('\n')
    item_count = 0

    # go through every record
    for r in records:
        # extract information from record
        # format: title;author;year;url

        info = r.split(';')

        # title probably contained semicolon
        if len(info) != 4:
            continue

        year = info[2]
        if not year.isnumeric():
            # something went wrong
            continue

        year = int(year)

        # check if year exists
        if year not in books:
            books[year] = []

        # create new entry
        new_entry = {
            'id': item_count,
            'title': info[0],
            'author': info[1],
            'url': info[3]
        }

        # add entry
        books[year].append(new_entry)
        item_count += 1

    print(f'...Successfully found {item_count} books.')

    # return the books database
    return books

def get_text_link(url):
    """
    Uses BeatifulSoup to scrape for the Plain Text Url for a gutenberg book.
    """

    base_url = 'http://www.gutenberg.org'

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # find the link containing the textfile
    try:
        links = soup.select('table.files a[title="Download"]')

        download_url = base_url + links[-1]['href']
        return download_url
    except:
        return False

def get_book_by_url(url):
    """
    Fetches and cleans a gutenberg book from a url and returns a list of sentences.
    """

    # url has id in the end
    book_id = url.split('etext/')[1]

    # insert book_id to get url for plain text page
    # text_url = f'http://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
    text_url = f'http://www.gutenberg.org/ebooks/{book_id}.txt.utf-8'
    
    # request page and fetch contents
    page = requests.get(text_url)

    # something went wrong, use bs4 instead to fetch the url
    if page.status_code != 200:
        text_url = get_text_link(url)

        if text_url == False: 
            return False
        
        page = requests.get(text_url)

        # still did not work
        if page.status_code != 200:
            print(f'!!! Something went wrong fetching {url}.')
            return False

    # get content
    content = page.text

    # clean data
    content = clean_data(content)

    # split data into sentences
    sentences = sent_tokenize(content)

    # return the data
    return sentences

def merge_vectors(main, new):
    """
    Adds every token value from the new array to the main array.
    """

    # go through every word
    for token in new:

        # add vector to previous vectr
        if token in main:
            main[token] += new[token]

        # otherwise create new token
        else:
            main[token] = new[token]

def average_vectors(vectors, n):
    """
    Computes the average for every vector in vector dictionary.
    """

    for token in vectors:
        vectors[token] /= n
        

def save_dictionary(dictionary, filename):
    """
    Saves a dictionary using pickle to file.
    """

    print(f'...Saving result to {filename}.')

    np.save(filename, dictionary)


##################################
# MAIN
##################################

def main():

    # SETUP 
    sentiments = ['negative', 'positive']
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

    # MAIN VARIABLES
    lexicon = load_lexicon()
    books_lexicon = load_book_database(BOOK_FILENAME)
    result = {}

    # Go through every year
    for current_year in range(START_YEAR, END_YEAR + 1):

        # check if there are any books for the current year, otherwise skip
        if current_year not in books_lexicon: 
            print(f'...No books found for year {current_year}, skipping.')
            continue
        
        # initilize data container for time period
        year_sentiment = {}
        year_emotion = {}

        books = books_lexicon.get(current_year, [])

        # get a sample of 0-SAMPLE_SIZE books from every year
        if len(books) < SAMPLE_SIZE:
            sample = books
        else:
            sample = random.sample(books, SAMPLE_SIZE)

        print(f'...Analyzing {len(sample)} samples from year {current_year}')

        # go through every book and populate the data dictionary
        for s in sample:

            # get the txt file from gutenberg by url
            sentences = get_book_by_url(s['url'])

            # something went wrong, continue
            if sentences == False: continue

            # get token vectors
            token_vectors = get_token_vectors_from_sentences(sentences, lexicon, FILTERS)

            # filter out token_vectors with less than sum of 50
            token_vectors = {
                t: v for (t, v) in token_vectors.items() if v.sum() > 50
            }

            # create emotion vectors
            emotion_vectors = {
                t: np.concatenate([v[:5], v[7:]]) for (t, v) in token_vectors.items()
            }

            # create sentiment vectors
            sentiment_vectors = {
                t: v[5:7] for (t, v) in token_vectors.items()
            }

            # add data to total count
            merge_vectors(year_sentiment, sentiment_vectors)
            merge_vectors(year_emotion, emotion_vectors)

            print(f'\t...Book {s["id"]} done!')

        # all books for current year done, get the average of total vectors
        average_vectors(year_sentiment, len(sample))
        average_vectors(year_emotion, len(sample))

        sentiment_index = list(year_sentiment.keys())
        emotion_index = list(year_emotion.keys())

        # create dataframes
        sentiment_df_raw = pd.DataFrame(data=year_sentiment.values(),columns=sentiments)
        emotion_df_raw = pd.DataFrame(data=year_emotion.values(),columns=emotions)

        # normalize
        normalize_token_vectors(year_sentiment, verbose=False)
        normalize_token_vectors(year_emotion, verbose=False)

        # create data frames for normalized
        sentiment_df = pd.DataFrame(data=year_sentiment.values(), index=sentiment_index, columns=sentiments)
        emotion_df = pd.DataFrame(data=year_emotion.values(), index=emotion_index, columns=emotions)

        # make directory for data
        base_name = DATA_DIRECTORY + f'{current_year}/'

        # make sure path exists
        Path(base_name).mkdir(parents=True, exist_ok=True)

        # save dataframes to files
        sentiment_df_raw.to_csv(base_name + 'sentiment_raw.csv')
        emotion_df_raw.to_csv(base_name + 'emotion_raw.csv')
        sentiment_df.to_csv(base_name + 'sentiment.csv')
        emotion_df.to_csv(base_name + 'emotion.csv')

        print(f'\tData saved to {base_name}')

    # all books have been analyzed, hooray!
    print('...SUCCESS!')


##################################
# GENERAL TESTING
##################################

# books_lexicon = load_book_database(BOOK_FILENAME)

# x_values = books_lexicon.keys()
# y_values = [len(books) for books in books_lexicon.values()]

# plt.bar(x_values, y_values)
# plt.ylim([0,50])
# plt.xlim([1800, 2000])
# plt.show()

# result = np.load('result.npy', allow_pickle=True)
# result = result.item()

# print(result[1905]['sentiment']['woman'])

if __name__ == '__main__':
    main()