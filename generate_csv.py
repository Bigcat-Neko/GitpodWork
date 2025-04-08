import asyncio
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from io import StringIO
from playwright.async_api import async_playwright

async def fetch_sentiment_with_playwright():
    async with async_playwright() as p:
        # Launch Chromium; set headless=False for debugging (change to True for production)
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/114.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        # Navigate to DailyFX sentiment page
        await page.goto("https://www.dailyfx.com/sentiment", timeout=60000)
        
        # Try to dismiss a cookie consent prompt if it appears
        try:
            await page.click("text=Accept", timeout=5000)
            print("Cookie consent dismissed.")
        except Exception:
            print("No consent prompt detected.")
        
        # Wait for the table rows to load.
        await page.wait_for_selector("table tbody tr", timeout=60000)
        
        # Retrieve page content and parse it with BeautifulSoup
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            raise Exception("Sentiment table not found.")
        
        print("Table detected with Playwright.")
        
        # Read the HTML table into a pandas DataFrame
        df = pd.read_html(StringIO(str(table)))[0]
        print("DEBUG: Table columns returned:", df.columns.tolist())
        
        # Normalize column names (convert to lower-case and strip spaces)
        df.columns = [c.lower().strip() for c in df.columns]
        if "symbol" in df.columns or "market" in df.columns:
            if "symbol" in df.columns:
                df.rename(columns={"symbol": "pair"}, inplace=True)
            elif "market" in df.columns:
                df.rename(columns={"market": "pair"}, inplace=True)
            df = df[[col for col in df.columns if "pair" in col or "% long" in col or "% short" in col]]
            if "% long" in df.columns and "% short" in df.columns:
                df.rename(columns={"% long": "long_%", "% short": "short_%"}, inplace=True)
            else:
                raise Exception("Sentiment percentages not found in table.")
        else:
            raise Exception("Expected pair/symbol column not found. Columns: " + str(df.columns))
    
        df["asset_type"] = "forex"
        df["timestamp"] = datetime.utcnow()
        
        await browser.close()
        return df

if __name__ == "__main__":
    df = asyncio.run(fetch_sentiment_with_playwright())
    print("Data fetched with Playwright:")
    print(df.head())
