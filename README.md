# Stock Movement Analysis Based on Social Media Sentiment

## Project Overview
The objective of this project is to analyze the relationship between stock prices and social media sentiment using synthetic Reddit data. By generating synthetic sentiment scores for stocks and correlating them with actual stock prices, we aim to uncover patterns and insights that could inform investment strategies.

Code Explanation
Import Libraries:

```
import praw
import pandas as pd
```
praw: This is the Python Reddit API Wrapper, which allows you to interact with the Reddit API easily.
pandas: This library is used for data manipulation and analysis, specifically to create and manage data in a tabular format (DataFrames).

Initialize Reddit API Client:
```
reddit = praw.Reddit(
    client_id='your_client_id',  
    client_secret='your_cklient_secret',  
    user_agent='Stock Market Sentiment Scraper',  
)
```
Here, you initialize the Reddit API client using your credentials. Replace the client_id and client_secret with your own Reddit API credentials to authenticate your requests.

Define Subreddit and Keywords:

```
subreddit_name = 'stocks'  
tech_keywords = [
    'AAPL',    
    'MSFT',    
    'GOOGL',   
    'AMZN',    
    'TSLA',    
    'NVDA',    
    'META',    
    'ADBE',    
    'CRM',     
    'INTC',    
    'AMD',     
    'ORCL',    
    'CSCO',    
    'NFLX',    
    'IBM',     
]
```
This section sets the subreddit you want to scrape (in this case, stocks) and defines a list of tech stock keywords you’re interested in.

Create Data Storage:
```
columns = ['Title', 'Score', 'URL', 'Created', 'Stock']
data = []
```

It defines the structure of the data you’ll be collecting, including the title of the post, score, URL, creation date, and associated stock.

Scrape Data:
```
subreddit = reddit.subreddit(subreddit_name)
cutoff_timestamp = 1609459200  # This is the timestamp for January 1, 2021

for keyword in tech_keywords:
    for submission in subreddit.search(keyword, sort='new', limit=100):  
        if submission.created_utc > cutoff_timestamp:  
            data.append([submission.title, submission.score, submission.url, submission.created_utc, keyword])
```

You first get the subreddit object. Then you loop through each keyword and search for submissions containing that keyword. Only submissions created after January 1, 2021, are saved in the data list.

Convert to DataFrame:
```
df = pd.DataFrame(data, columns=columns)
df['Created'] = pd.to_datetime(df['Created'], unit='s')
```

The collected data is converted into a pandas DataFrame for easier manipulation. The timestamps are also converted into a readable date format.

Display and Save Data:
```
print(df)
df.to_csv('reddit_tech_stock_data_after_2020.csv', index=False)
print("Data has been successfully scraped and saved to 'reddit_tech_stock_data_after_2020.csv'.")
```

Finally, the DataFrame is printed to the console, and the data is saved to a CSV file named reddit_tech_stock_data_after_2020.csv






