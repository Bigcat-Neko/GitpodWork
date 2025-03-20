import requests
from bs4 import BeautifulSoup

SCRAPER_API_KEY = "9f90a8d8d463406010f7cbd132b7590f"  # Replace with your ScraperAPI key

def get_investing_forex_news_with_proxy():
    # The target URL to scrape (Investing.com Forex news page)
    target_url = "https://www.investing.com/news/forex-news"
    
    # Build the ScraperAPI URL; it wraps your target URL with proxy rotation
    proxy_url = f"http://api.scraperapi.com?api_key={SCRAPER_API_KEY}&url={target_url}"
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.investing.com/"
    }
    
    response = requests.get(proxy_url, headers=headers)
    if response.status_code != 200:
        raise Exception("Failed to fetch the page, status code:", response.status_code)
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Adjust these selectors based on the actual HTML structure
    news_items = soup.find_all("article", class_="js-article-item")
    news_headlines = []
    for item in news_items:
        headline_elem = item.find("a", class_="title")
        if headline_elem:
            headline = headline_elem.get_text(strip=True)
            news_headlines.append(headline)
    
    return news_headlines

if __name__ == "__main__":
    try:
        headlines = get_investing_forex_news_with_proxy()
        print("Latest Forex News from Investing.com using ScraperAPI:")
        for h in headlines:
            print("-", h)
    except Exception as e:
        print("Error:", e)
