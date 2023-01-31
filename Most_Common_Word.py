# Question: How can we use Python to scrape a website and find the 10 most common words used on the page, while handling potential errors from 
# incorrect user input for the website URL?

import requests
from bs4 import BeautifulSoup
from collections import Counter
# Function to fetch top 10 most common words
def fetch_top_words(site):
    try:
        # Fetch website content
        response = requests.get(site)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text()
            words = text.split()
            word_counts = Counter(words)
            return word_counts.most_common(10)
        else:
            return None
    except:
        return None
# User input for website URL
site = input("Enter a website URL (e.g. https://www.example.com): ")
result = fetch_top_words(site)
if result:
    print("Top 10 most common words:")
    for word, count in result:
        print(f"{word}: {count}")
else:
    print("Error: Invalid website URL or error fetching website data")
    
    
    
