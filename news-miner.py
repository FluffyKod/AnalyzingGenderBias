import requests
from bs4 import BeautifulSoup
from pprint import pprint

# FUNCTIONS

def get_svt_news(news_url):
    """
    Retrieves a news article from SVT nyheter and returns the title and article
    """
    try:
        news_page = requests.get(news_url)
        news_soup = BeautifulSoup(news_page.text, 'html.parser')

        news = news_soup.find('article')
        article = news.find(class_='nyh_article-body').text
        title = news.find(class_='nyh_article__heading').text

        return (title, article)
    except:
        return ('', '')

def get_dn_news(news_url):
    try:
        news_page = requests.get(news_url)
        news_soup = BeautifulSoup(news_page.text, 'html.parser')

        news = news_soup.find('article')
        article = news.find(class_='article__body').text
        title = news_soup.find(class_='article__title').text

        return (title, article)
    except:
        return ('', '')

def get_dr_news(news_url):
    try:
        news_page = requests.get(news_url)
        news_soup = BeautifulSoup(news_page.text, 'html.parser')

        news = news_soup.find('article')
        article = news.find(class_='dre-article-body').text
        title = news.find(class_='dre-title-text').text

        return (title, article)
    except:
        return ('', '')
    

def fetch_urls(links):

    urls = []
    for a in links:
        try:
            news_url = a['href'][7:].split('&')[0]
            urls.append(news_url)
        except:
            continue

    return set(urls)

# MAIN

def main():
    """
    Mines news from nyheter.
    """

    base_url = 'https://www.google.com/search?q=dr.dk+svensk&tbm=nws&start='
    file_name = 'data/danish_news.txt'
    reset = False

    clean = []

    if reset: open(file_name, 'w').close()

    total = 210

    for start in range(total, total + 100, 10):

        print(start)

        url = f'{base_url}{start}'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')

        links = soup.select('div.kCrYT a', href=True)

        for i, news_url in enumerate(fetch_urls(links)):
            
            # get title and content of article
            title, article = get_dr_news(news_url)

            # clean article
            for c in clean:
                article = article.replace(c, '')

            # write to file
            with open(file_name, 'a') as f:
                f.write(f'{article}\n')

            # output success
            total += 1
            print(f'#{total}: {title}')


if __name__ == '__main__':
    main()
