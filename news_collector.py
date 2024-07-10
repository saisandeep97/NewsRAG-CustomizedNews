import requests
import pandas as pd
import os 

# NewsAPI configuration
API_KEY = os.getenv("NEWSAPI_API_KEY")
BASE_URL = "https://newsapi.org/v2/top-headlines"

# Categories and countries
CATEGORIES = ['politics', 'sports', 'business', 'entertainment']
COUNTRIES = ['us', 'in', 'gb', 'au']  # USA, India, UK, Australia (EU is not a country code)

def fetch_news(country, category):
    """Fetch news articles for a specific country and category."""
    params = {
        'apiKey': API_KEY,
        'country': country,
        'category': category,
        'pageSize': 100  # Maximum allowed by NewsAPI
    }
    
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        return response.json()['articles']
    else:
        print(f"Error fetching news for {country} - {category}: {response.status_code}")
        return []

def main():
    all_articles = []
    
    for country in COUNTRIES:
        for category in CATEGORIES:
            articles = fetch_news(country, category)
            for article in articles:
                all_articles.append({
                    'title': article['title'],
                    'description': article['description'],
                    'content': article['content'],
                    'url': article['url'],
                    'publishedAt': article['publishedAt'],
                    'source': article['source']['name'],
                    'country': country,
                    'category': category
                })
    
    # Create a DataFrame
    df = pd.DataFrame(all_articles)
    
    # Save to CSV
    filename = f"news_articles.csv"
    df.to_csv("news_data/"+filename, index=False)
    print(f"Saved {len(df)} articles to {filename}")

if __name__ == "__main__":
    main()