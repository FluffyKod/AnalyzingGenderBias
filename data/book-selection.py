"""
Mines publication year from gutenberg books.

Title, author, url, year
"""

##################################
# LIBRARIES
##################################

from pymarc import MARCReader
from bs4 import BeautifulSoup
import requests
import pickle
import time

##################################
# MAIN FUNCTION
##################################

def main():
    """
    Goes through books in Project Gutenbergs catalog and attempts to find the publication year of every book. Stores the books title, author, publication year and url in a textfile.
    """

    ##############
    # PARAMETERS #
    ##############

    # Keep track of previous data mining:
    # RECORD_START = 17314
    # 30000 - 32700

    # the Gutenberg book number to start at + how many books to extract
    RECORD_START = 17314
    BOOK_NUMBER = 2

    FILENAME = 'book_lexicon.txt'

    # optional, output year data as a pickle file
    USE_DICTIONARY = False
    BOOKS_PICKLE = 'books_lexicon.pickle'
    YEAR_BOOKS_PICKLE = 'year_books2_2.pickle'

    ########################################################

    # variables to store the found data in
    books = []
    years_books = {}

    with open('data/catalog.marc', 'rb') as fh:
        reader = MARCReader(fh)

        books_found = 0
        i = 0

        for record in reader:

            i += 1
            
            # lower bound
            if i <= RECORD_START:
                continue

            # upper bound
            if books_found >= BOOK_NUMBER: break

            # new attempt
            print(f'#{i} Attempt')

            lang = record['040']['b']
            if str(lang) != 'eng': 
                continue

            category = record['516']['a']
            if str(category) != 'Electronic text': 
                continue

            title = record.title()
            title = str(title).replace('\r\n', ' ').replace('\n', ' ').strip()
            
            # in format lastname, first name
            author_parts = str(record.author()).split(',')
            try:
                author = author_parts[1].strip() + ' ' + author_parts[0].strip()
            except:
                author = 'None'

            # url
            for f in record.get_fields('856'):
                book_url = str(f).replace('=856  40$u', '')
                # print(url)
                break
            
            if author == None:
                search_string = title
            else:
                search_string = title + ' ' + author
            
            search = search_string.replace(' ', '+')
            
            url = f'https://www.google.com/search?q={search}'

            try:
                page = requests.get(url)
                soup = BeautifulSoup(page.content, 'html.parser')
                bemp = soup.select_one('span.BNeawe.tAd8D.AP7Wnd').text
                # bemp = soup.select_one('span.Eq0J8.LrzXr.kno-fv').text

                if bemp.isnumeric():
                    year = int(bemp)
                elif bemp[-4].isnumeric():
                    year = int(bemp[-4])
                else:
                    continue

                # legit book found
                year = int(bemp)
                entry = f'{title};{author};{year};{book_url}\n'

                short_title = title
                if len(title) > 50:
                    short_title = title[:50] + '...'

                # write to file
                with open(FILENAME, 'a') as book_file:
                    book_file.write(entry)
                
                books_found += 1
                print(f'\t#{books_found}: Book found for {short_title}!')

                if USE_DICTIONARY: 
                    new_entry = {
                        'title': title,
                        'author': author,
                        'year': year,
                        'url': book_url
                    }

                    books.append(new_entry)

                    if year not in years_books:
                        years_books[year] = []

                    years_books[year].append(new_entry)

                    # write to dictionary file
                    with open(BOOKS_PICKLE, 'wb') as handle:
                        pickle.dump(books, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # write to dictionary file
                    with open(YEAR_BOOKS_PICKLE, 'wb') as handle:
                        pickle.dump(years_books, handle, protocol=pickle.HIGHEST_PROTOCOL)

            except:
                continue

            