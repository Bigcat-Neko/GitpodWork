import requests
from bs4 import BeautifulSoup

def get_dailyfx_forex_news():
    url = "https://www.dailyfx.com/forex-news"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/115.0.0.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception("Failed to fetch the page, status code:", response.status_code)
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Example: Assume news headlines are in <h2> tags within article elements.
    articles = soup.find_all("article")
    headlines = []
    for article in articles:
        h2 = article.find("h2")
        if h2:
            headline = h2.get_text(strip=True)
            headlines.append(headline)
    
    return headlines

if __name__ == "__main__":
    try:
        news_headlines = get_dailyfx_forex_news()
        print("Latest Forex News from DailyFX:")
        for headline in news_headlines:
            print("-", headline)
    except Exception as e:
        print("Error:", e)
