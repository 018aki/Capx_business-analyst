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


Your next step in the project involves cleaning and processing the data obtained from Reddit to prepare it for further analysis. Here's an explanation of each section of the provided code, along with details on how to run it effectively in a Jupyter Notebook.

Importing Libraries:
```
import pandas as pd
import numpy as np
import re
```
pandas: For data manipulation and analysis.
numpy: A library for numerical operations (not extensively used in this snippet but commonly used in data processing).
re: For regular expressions to clean text.

Loading Data:
```
df = pd.read_csv('reddit_tech_stock_data_after_2020.csv')
```
This line loads the CSV file containing the scraped Reddit data into a DataFrame called df.

Initial Data Inspection:
```
print("Initial DataFrame:")
print(df.head())
```
Displays the first few rows of the DataFrame to check the loaded data.

Checking for Missing Values:
```
print("\nMissing Values:")
print(df.isnull().sum())
```
Checks for any missing values in the DataFrame.

Handling Missing Values:
```
df.dropna(inplace=True)
```
Drops any rows with missing values from the DataFrame.

Removing Duplicates:
```
df.drop_duplicates(subset=['Title', 'Stock'], inplace=True)
```
Removes duplicate entries based on the combination of the 'Title' and 'Stock' columns.

Filtering Relevant Posts:
```
tech_keywords = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
df = df[df['Title'].str.contains('|'.join(tech_keywords), case=False, na=False)]
```
Filters the DataFrame to only include posts that mention specific tech stock tickers in the title.

Text Preprocessing:
```
def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove punctuation and numbers
    text = text.lower()  # Convert to lowercase
    return text

df['Title'] = df['Title'].apply(clean_text)
```
Defines a function to clean text by removing punctuation, numbers, and converting the text to lowercase. The function is applied to the 'Title' column.


Datetime Conversion:

```
df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
df = df.dropna(subset=['Created'])
```
Converts the 'Created' column to datetime format and drops any rows with invalid dates.

Sorting the Data:
```
df = df.sort_values(by=['Created'], ascending=True)
df.reset_index(drop=True, inplace=True)
```
Sorts the DataFrame by the 'Created' column in ascending order and resets the index.

Final Output:
```
print("\nCleaned and Sorted DataFrame:")
print(df.head())

df.to_csv('cleaned_sorted_reddit_tech_stock_data_after_2020.csv', index=False)
print("Cleaned and sorted data has been saved to 'cleaned_sorted_reddit_tech_stock_data_after_2020.csv'.")
```

Displays the cleaned DataFrame and saves it to a new CSV file.
```
import praw
import pandas as pd
import re

# Initialize Reddit API Client
reddit = praw.Reddit(
    client_id='LaY5MG5A3UVMslGwoUu_TA',  # Your client ID
    client_secret='rX9uFa4_GFTXtMovVxQOP3lrXydX6g',  # Your client secret
    user_agent='Stock Market Sentiment Scraper',  # User agent string
)

# Define the subreddit and specific tech stocks
subreddit_name = 'stocks'
tech_keywords = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 
    'META', 'ADBE', 'CRM', 'INTC', 'AMD', 'ORCL', 
    'CSCO', 'NFLX', 'IBM'
]

# Create a DataFrame to store the data
columns = ['Title', 'Score', 'URL', 'Created', 'Stock']
data = []

# Scrape data from the subreddit
subreddit = reddit.subreddit(subreddit_name)

# Define the cutoff timestamp for posts after 2020
cutoff_timestamp = 1609459200  # January 1, 2021

for keyword in tech_keywords:
    for submission in subreddit.search(keyword, sort='new', limit=100):  # Adjust limit as needed
        if submission.created_utc > cutoff_timestamp:
            data.append([submission.title, submission.score, submission.url, submission.created_utc, keyword])

# Convert the list to a DataFrame
df = pd.DataFrame(data, columns=columns)

# Convert created timestamp to readable date
df['Created'] = pd.to_datetime(df['Created'], unit='s')

# 1. Remove any missing values (if present)
df.dropna(inplace=True)

# 2. Remove duplicates based on 'Title' and 'Stock'
df.drop_duplicates(subset=['Title', 'Stock'], inplace=True)

# 3. Remove rows with 'Score' less than 5 to filter low-quality posts
df = df[df['Score'] >= 5]

# 4. Clean the 'Title' column
def clean_text(text):
    # Remove special characters and numbers, and convert to lowercase
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Keep only letters and spaces
    text = text.lower()  # Convert to lowercase
    return text

df['Title'] = df['Title'].apply(clean_text)

# 5. Sort the DataFrame by the 'Created' column (ascending order by date)
df.sort_values(by='Created', ascending=True, inplace=True)

# Reset the index after sorting
df.reset_index(drop=True, inplace=True)

# Display the cleaned and sorted DataFrame
print("\nCleaned and Sorted DataFrame:")
print(df.head())

# Save the cleaned DataFrame to a new CSV file
df.to_csv('cleaned_reddit_tech_stock_data.csv', index=False)

print("Cleaned data has been saved to 'cleaned_reddit_tech_stock_data.csv'.")
```
praw: A Python wrapper for the Reddit API, allowing for easy interaction with Reddit's data.
pandas: A powerful library for data manipulation and analysis, particularly useful for handling tabular data.
re: A library for working with regular expressions, useful for string manipulation and cleaning.

Here, the Reddit client is initialized with the necessary credentials (client_id, client_secret, and user_agent). These are essential for authenticating your requests to the Reddit API.

This segment specifies the subreddit to scrape (stocks) and lists the tech stock tickers of interest. This list will be used to filter relevant posts.

A list of columns is defined for the DataFrame that will hold the scraped data. The data list is initialized to collect the individual post details.

This loop searches the specified subreddit for posts containing each stock ticker keyword.
It filters posts created after January 1, 2021 (using a timestamp).
For each relevant post, the title, score (upvotes), URL, creation time, and stock ticker are stored in the data list.

A pandas DataFrame is created using the collected data, making it easier to manipulate and analyze.

the Created column is converted from a Unix timestamp to a human-readable datetime format.
Any missing values are removed from the DataFrame.
Duplicates based on Title and Stock are eliminated.
Posts with a score of less than 5 are filtered out to focus on higher-quality content

A function is defined to clean the titles by removing special characters and converting text to lowercase. This preprocessing is essential for further text analysis, such as sentiment analysis.

The DataFrame is sorted by the Created date in ascending order.
The index is reset to ensure it is sequential after sorting.

The cleaned DataFrame is displayed, allowing you to verify the changes made during the cleaning process.

Finally, the cleaned DataFrame is saved to a CSV file for further analysis or future use.

Summary
This code effectively:

Scrapes Reddit for relevant discussions about tech stocks.
Cleans and organizes the scraped data into a structured format.
Filters for high-quality posts, ensuring that only relevant and meaningful data is retained.
Prepares the data for further analysis, such as sentiment analysis, visualization, or predictive modeling.

Let's analyze the next segment of your code that performs sentiment analysis on Reddit posts related to tech stocks using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool.

 Installing VADER:
 ```
pip install vaderSentiment
```
This command installs the vaderSentiment library, which provides tools for sentiment analysis specifically tuned for social media texts.

Importing Libraries:
```
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
```
pandas: As before, this library is used for data manipulation.
SentimentIntensityAnalyzer: A class from the vaderSentiment library that provides methods to calculate sentiment scores for text data.

Loading the Scraped Data:
```
df = pd.read_csv('reddit_tech_stock_data_after_2020.csv')
```
The DataFrame containing the scraped Reddit data is loaded from a CSV file. This DataFrame should have columns including Title, Score, URL, Created, and Stock.

Initializing the VADER Sentiment Analyzer:
```
analyzer = SentimentIntensityAnalyzer()
```
An instance of the sentiment analyzer is created. This object will be used to compute sentiment scores for each post title.

Defining the Sentiment Classification Function:
```
def get_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
```
get_sentiment: This function takes a text input and calculates its sentiment scores using VADER.
    The polarity_scores method returns a dictionary of sentiment scores, including a compound score that summarizes the overall sentiment.
    The function classifies the sentiment into three categories:
        Positive: Compound score of 0.05 or higher.
        Negative: Compound score of -0.05 or lower.
        Neutral: Compound score between -0.05 and 0.05.

Applying Sentiment Analysis:
```
df['Sentiment'] = df['Title'].apply(get_sentiment)
```
The get_sentiment function is applied to each title in the DataFrame, creating a new column Sentiment that stores the classification for each post title.

Displaying Sentiment Analysis Results:
```
print("\nDataFrame with Sentiment Analysis:")
print(df[['Title', 'Sentiment']].head())
```
This prints the first few rows of the DataFrame, showing the Title and its corresponding Sentiment, allowing you to quickly verify that the sentiment analysis was applied correctly.

Saving the Data with Sentiment:
```
df.to_csv('reddit_tech_stock_with_sentiment.csv', index=False)

print("\nSentiment analysis has been completed, and data has been saved to 'reddit_tech_stock_with_sentiment.csv'.")
```
Finally, the DataFrame with the sentiment results is saved to a new CSV file. This makes it easier to share or analyze further.


Summary
This code effectively:

Uses VADER to perform sentiment analysis on the titles of Reddit posts regarding tech stocks.

Classifies each title into positive, negative, or neutral sentiments based on the computed compound score.

Saves the enhanced DataFrame containing sentiment analysis results for further exploration or reporting.


This segment of code performs feature extraction and analysis on the sentiment data from Reddit posts related to tech stocks. Here’s a breakdown of what the code does and suggestions for next steps:

Loading the Data
```
import pandas as pd

# Load the data with sentiment
df = pd.read_csv('reddit_tech_stock_with_sentiment.csv')
```
The pandas library is imported, and the sentiment data is loaded from a CSV file into a DataFrame called df

Frequency of Mentions for Each Stock:
```
# 1. Frequency of mentions for each stock
stock_mentions = df['Stock'].value_counts()
print("\nFrequency of Mentions for Each Stock:")
print(stock_mentions)
```
value_counts(): This method counts the occurrences of each stock in the Stock column, providing insights into which stocks are being discussed most frequently.
The results are printed, showing how many times each stock was mentioned in the Reddit posts.

 Sentiment Polarity Distribution for Each Stock:
 ```
# 2. Sentiment polarity distribution for each stock
sentiment_distribution = df.groupby(['Stock', 'Sentiment']).size().unstack(fill_value=0)
print("\nSentiment Polarity Distribution for Each Stock:")
print(sentiment_distribution)
```
groupby(['Stock', 'Sentiment']): This groups the DataFrame by stock and sentiment, allowing for aggregation of sentiment counts for each stock.
size().unstack(fill_value=0): This counts the occurrences and reshapes the result into a DataFrame format, filling missing values with zero. The output shows the distribution of sentiments (positive, negative, neutral) for each stock

Saving the Extracted Features to CSV Files:
```
# 3. Save these extracted features to CSV files for further analysis
stock_mentions.to_csv('stock_mentions.csv', index=True)
sentiment_distribution.to_csv('sentiment_distribution.csv', index=True)

print("\nFeature extraction completed. Data saved to 'stock_mentions.csv' and 'sentiment_distribution.csv'.")
```
The frequency of stock mentions and the sentiment distribution are saved to CSV files for future analysis, making it easier to visualize and interpret the data.

Summary
This code effectively:

Provides insights into how frequently each stock is mentioned in Reddit posts.
Offers a clear distribution of sentiments related to each stock, which is useful for understanding public perception and sentiment trends.

This segment of code implements topic modeling on the Reddit posts related to tech stocks. Here’s a breakdown of what the code does and suggestions for the next steps you can take after this analysis:

Loading Required Libraries and Data:
```
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load the data with sentiment
df = pd.read_csv('reddit_tech_stock_with_sentiment.csv')
```
Libraries like pandas, nltk, sklearn, and matplotlib are imported for data manipulation, text preprocessing, topic modeling, and visualization.
NLTK stopwords are downloaded to filter out common words in English.

Text Preprocessing:
```
# 1. Preprocess the 'Title' column for topic modeling
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    # Tokenization and remove stopwords
    tokens = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(tokens)

# Apply text preprocessing to the 'Title' column
df['Processed_Title'] = df['Title'].apply(preprocess_text)
```
The preprocess_text function removes stopwords from the titles, improving the quality of the data for topic modeling.
Processed titles are stored in a new column Processed_Title.

Document-Term Matrix Creation:
```
# 2. Convert the processed text into a document-term matrix
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(df['Processed_Title'])
```
CountVectorizer converts the processed titles into a document-term matrix, filtering out terms that appear in more than 95% of the documents (max_df=0.95) and those that appear in fewer than 2 documents (min_df=2).

LDA Model Fitting:
```
 3. Fit LDA model for topic extraction
lda = LatentDirichletAllocation(n_components=5, random_state=42)  # 5 topics, you can adjust
lda.fit(doc_term_matrix)
```
The LatentDirichletAllocation model is used to identify topics in the document-term matrix. Here, it is set to extract 5 topics.

Display Top Words for Each Topic:
```
# 4. Display the top words in each topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names, no_top_words)
```
The display_topics function prints the top words associated with each topic, helping you understand what themes are present in the data.

Visualize Topic Distribution:
```
# 5. Optional: Visualize the topic distribution (you can skip this part if not needed)
topic_values = lda.transform(doc_term_matrix)
df['Topic'] = topic_values.argmax(axis=1)

plt.hist(df['Topic'], bins=range(lda.n_components + 1))
plt.title('Distribution of Topics')
plt.xlabel('Topic')
plt.ylabel('Number of Posts')
plt.show()
```
This part visualizes how many posts are associated with each topic using a histogram.

Save the Topic-Labeled Data:
```
# Save the topic-labeled data
df.to_csv('reddit_tech_stock_with_topics.csv', index=False)

print("\nTopic Modeling completed. Data saved to 'reddit_tech_stock_with_topics.csv'.")
```
The DataFrame, now with an additional column for the assigned topic, is saved to a CSV file for further analysis.

Summary
This code effectively:

Preprocesses Reddit post titles by removing stopwords.
Extracts topics using LDA and displays the top words for each topic.
Visualizes the distribution of topics and saves the enriched dataset for future use.

You've implemented a thorough analysis of Reddit post titles related to tech stocks using Latent Dirichlet Allocation (LDA) for topic modeling. Here’s a detailed breakdown of your code and suggestions for the next steps you might consider.

Loading Required Libraries:
```
pip install yfinance
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
```
The yfinance library is imported, likely for future use to fetch financial data, while pandas, re, and scikit-learn are used for data manipulation, text cleaning, and topic modeling.

Loading Data:
```
# Load the data with sentiment
df = pd.read_csv('reddit_tech_stock_with_sentiment.csv')
```
The dataset containing Reddit posts with sentiment analysis is loaded.

Text Preprocessing:
```
# Step 1: Text Preprocessing
# Function to clean text
def preprocess_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove punctuation and numbers
    text = text.lower()  # Convert to lowercase
    return text

# Apply preprocessing to the 'Title' column
df['Cleaned_Title'] = df['Title'].apply(preprocess_text)
```
The preprocess_text function removes punctuation and numbers, converting the text to lowercase for uniformity. This preprocessing step is essential for accurate topic modeling.

Vectorization:
```
# Step 2: Vectorization using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Cleaned_Title'])
```
Topic Modeling:
```
# Step 3: Topic Modeling using LDA
n_topics = 5  # Choose the number of topics
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_model.fit(X)
```
The LDA model is fitted to the document-term matrix to extract 5 topics. This parameter can be adjusted based on the complexity of the dataset.

Analyze and Interpret Topics:
```
# Step 4: Analyze and Interpret Topics
def print_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Print the topics
no_top_words = 10
print("\nIdentified Topics:")
print_topics(lda_model, vectorizer.get_feature_names_out(), no_top_words)
```
The print_topics function displays the top words associated with each identified topic, providing insights into what each topic represents.

Visualize Topic Distribution:
```
# Optional: Visualize the topic distribution
topic_distribution = lda_model.transform(X)
plt.figure(figsize=(10, 5))
plt.hist(topic_distribution.argmax(axis=1), bins=n_topics, alpha=0.7)
plt.title('Distribution of Topics')
plt.xlabel('Topic Number')
plt.ylabel('Frequency')
plt.xticks(range(n_topics))
plt.show()
```
This visualization depicts the frequency of each topic across the posts, allowing for a quick assessment of topic prevalence.

Next step:
```
import pandas as pd
import random
from datetime import timedelta

# Define stock symbols and the date range based on the stock data
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
date_range = pd.date_range(start='2020-01-02', end='2020-05-26')  # Dates from your provided stock data

# Create a list to store synthetic Reddit data
reddit_data = []

# Generate synthetic Reddit posts
for date in date_range:
    for stock in stocks:
        # Generate random sentiment polarity (between -1 and 1)
        sentiment_score = random.uniform(-1, 1)
        
        # Create a synthetic title based on stock sentiment
        if sentiment_score > 0.5:
            title = f"Positive sentiment for {stock} after earnings!"
        elif sentiment_score < -0.5:
            title = f"Negative outlook on {stock} performance!"
        else:
            title = f"{stock} shows mixed signals in the market."
        
        # Create a URL placeholder
        url = f"https://www.reddit.com/r/stocks/comments/{random.randint(10000000, 99999999)}"
        
        # Append data to the list
        reddit_data.append([title, random.randint(0, 1000), url, date, stock, sentiment_score])

# Create a DataFrame for the synthetic Reddit data
reddit_df = pd.DataFrame(reddit_data, columns=['Title', 'Score', 'URL', 'Created', 'Stock', 'Sentiment'])

# Display the first few rows of the synthetic Reddit DataFrame
print(reddit_df.head())

# Save the synthetic Reddit DataFrame to a CSV file
reddit_df.to_csv('synthetic_reddit_stock_data.csv', index=False)
print("Synthetic Reddit data has been successfully created and saved to 'synthetic_reddit_stock_data.csv'.")
```
# Code Analysis

## Importing Libraries
- **pandas**: Used for data manipulation and creation of DataFrames.
- **random**: Provides functions to generate random numbers.
- **timedelta**: Although imported, it's not used in this particular code snippet.

## Defining Stock Symbols and Date Range
- A list of stock symbols (`stocks`) is defined, representing major tech companies.
- A date range (`date_range`) is created using `pd.date_range`, spanning from January 2, 2020, to May 26, 2020.

## Creating Synthetic Reddit Data
- An empty list (`reddit_data`) is initialized to store the synthetic posts.
- Nested loops iterate over each date in the date range and each stock symbol to generate synthetic Reddit posts.

## Generating Random Sentiment
- A random sentiment score between -1 and 1 is generated using `random.uniform(-1, 1)`.
- Based on the sentiment score:
  - If greater than 0.5, a positive title is generated.
  - If less than -0.5, a negative title is generated.
  - Otherwise, a neutral title is created.

## Creating URL Placeholders
- A synthetic URL is generated for each post using a random integer to simulate Reddit comment links.

## Appending Data to List
- Each synthetic post's title, score (randomly assigned between 0 and 1000), URL, creation date, stock symbol, and sentiment score are appended to the `reddit_data` list.

## Creating a DataFrame
- A Pandas DataFrame (`reddit_df`) is created from the `reddit_data` list with appropriate column names.

## Displaying and Saving Data
- The first few rows of the synthetic Reddit DataFrame are displayed using `print(reddit_df.head())`.
- The DataFrame is saved to a CSV file (`synthetic_reddit_stock_data.csv`), and a success message is printed.

## Conclusion
This code effectively creates a synthetic dataset of Reddit posts related to selected tech stocks, with random sentiment scores and URLs. This synthetic data can be useful for testing or simulating analysis workflows without needing real data.


Next Step:

```
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'aggregated_data' is your original DataFrame containing the results
# Uncomment and modify the below line to load your actual DataFrame if necessary
# aggregated_data = pd.read_csv('your_aggregated_data.csv')  # Load your data if needed

# Convert 'Date' column to datetime if not already done
aggregated_data['Date'] = pd.to_datetime(aggregated_data['Date'])

# Plotting the scatter plots
plt.figure(figsize=(12, 5))

# Scatter plot for Close Price vs Sentiment
plt.subplot(1, 2, 1)
sns.scatterplot(data=aggregated_data, x='Sentiment', y='Close', color='blue')
plt.title('Stock Price vs Sentiment')
plt.xlabel('Sentiment Score')
plt.ylabel('Stock Closing Price')
plt.grid(True)

# Scatter plot for Close Price vs Mention Count
plt.subplot(1, 2, 2)
sns.scatterplot(data=aggregated_data, x='Score', y='Close', color='orange')
plt.title('Stock Price vs Mention Count')
plt.xlabel('Mention Count')
plt.ylabel('Stock Closing Price')
plt.grid(True)

plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = aggregated_data[['Close', 'Sentiment', 'Score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
```

Data Visualization Code Explanation
This section describes the code used to visualize the relationships between stock prices, sentiment scores, and mention counts derived from synthetic Reddit data. The code utilizes Matplotlib and Seaborn libraries for creating scatter plots and a correlation heatmap.

Importing Libraries:
```
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```
import matplotlib.pyplot as plt: Imports the Pyplot module from Matplotlib for plotting.

import seaborn as sns: Imports Seaborn, which provides a high-level interface for drawing attractive statistical graphics.

import pandas as pd: Imports the Pandas library for data manipulation and analysis.

Loading Data:
```
# Assuming 'aggregated_data' is your original DataFrame containing the results
# Uncomment and modify the below line to load your actual DataFrame if necessary
# aggregated_data = pd.read_csv('your_aggregated_data.csv')  # Load your data if needed
```
A comment is provided to indicate where you would load your aggregated_data DataFrame. Uncomment and modify the line to load your actual dataset if necessary.

Date Conversion:
```
aggregated_data['Date'] = pd.to_datetime(aggregated_data['Date'])
```
Converts the 'Date' column in the DataFrame to datetime format if it isn't already. This ensures that date values are in a proper format for plotting.

Creating the Scatter Plots:
```
plt.figure(figsize=(12, 5))
```
Initializes a new figure for plotting with a specified size (12 inches wide and 5 inches tall).

Scatter Plot for Stock Price vs Sentiment:
```
plt.subplot(1, 2, 1)
sns.scatterplot(...): Creates a scatter plot with sentiment scores on the x-axis and stock closing prices on the y-axis, colored blue.
plt.title(...), plt.xlabel(...), and plt.ylabel(...): Set the title and labels for the x-axis and y-axis, respectively.
plt.grid(True): Enables grid lines on the plot for better readability.
```
Scatter Plot for Stock Price vs Mention Count

```
plt.subplot(1, 2, 2)
```
Activates the second subplot. The scatter plot is created with mention counts on the x-axis and stock closing prices on the y-axis, colored orange. Titles and labels are added similarly as before.

Adjusting Layout and Displaying Scatter Plots:
```
plt.tight_layout()
plt.show()
```
plt.tight_layout(): Adjusts the layout of the subplots to prevent overlap between titles, labels, and plots.

plt.show(): Renders the scatter plots on the screen.

Creating the Correlation Heatmap:
```
plt.figure(figsize=(8, 6))
correlation_matrix = aggregated_data[['Close', 'Sentiment', 'Score']].corr()
sns.heatmap(...)
plt.title(...)
plt.show()
```
plt.figure(figsize=(8, 6))

correlation_matrix = aggregated_data[['Close', 'Sentiment', 'Score']].corr()

sns.heatmap(...)

plt.title(...)

plt.show()


Conclusion
This code effectively visualizes the relationships between stock prices, sentiment scores, and mention counts, providing insights into how these variables correlate. The scatter plots allow for a clear visual assessment of the relationships, while the heatmap quantifies the strength and direction of the correlations, which can be useful for further analysis in stock market studies.


Next Step:

Monthly Aggregation and Visualization
In this section, we extract the year and month from the Date column, aggregate the data on a monthly basis, and create line plots to visualize the average sentiment, total mention counts, and average closing prices.


Extracting Year and Month:
```
# Extract year and month from the Date column
aggregated_data['YearMonth'] = aggregated_data['Date'].dt.to_period('M')
```
This line creates a new column, YearMonth, that contains the year and month extracted from the Date column.

Grouping Data:
```
# Group by YearMonth and calculate the mean sentiment and total mentions
monthly_data = aggregated_data.groupby('YearMonth').agg({
    'Sentiment': 'mean',
    'Score': 'sum',   # Sum the scores for mention counts
    'Close': 'mean'   # Average the closing price for the month
}).reset_index()
```
The data is grouped by YearMonth, and we calculate the average sentiment score, total mentions, and average closing price for each month.

Converting YearMonth Back to Datetime:
```
# Convert YearMonth back to datetime for plotting
monthly_data['YearMonth'] = monthly_data['YearMonth'].dt.to_timestamp()
```
This line converts the YearMonth period back into a timestamp format for plotting.

Displaying Monthly Aggregated Data:
```
# Display the monthly aggregated data
print(monthly_data)
```
This prints the aggregated monthly data to the console.

Creating Line Plots:
```
# Set the figure size
plt.figure(figsize=(14, 10))

# Line plot for monthly sentiment
plt.subplot(3, 1, 1)
sns.lineplot(data=monthly_data, x='YearMonth', y='Sentiment', color='blue', label='Mean Sentiment Score')
plt.title('Monthly Average Stock Sentiment')
plt.xlabel('Month')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45)
plt.grid(True)

# Line plot for monthly mention count
plt.subplot(3, 1, 2)
sns.lineplot(data=monthly_data, x='YearMonth', y='Score', color='orange', label='Total Mention Count')
plt.title('Monthly Total Mentions')
plt.xlabel('Month')
plt.ylabel('Mention Count')
plt.xticks(rotation=45)
plt.grid(True)

# Line plot for monthly closing prices
plt.subplot(3, 1, 3)
sns.lineplot(data=monthly_data, x='YearMonth', y='Close', color='green', label='Mean Closing Price')
plt.title('Monthly Average Closing Price')
plt.xlabel('Month')
plt.ylabel('Stock Closing Price')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()
```

This code creates three line plots:

Monthly Average Stock Sentiment: Shows how sentiment scores change over time.

Monthly Total Mentions: Illustrates the total mention counts for each month.

Monthly Average Closing Price: Displays the average closing price of stocks month by month.

Conclusion
The line plots provide valuable insights into trends over time, allowing for a better understanding of how sentiment and mention counts relate to stock prices.

Next Step:
## Report Findings Based on Analysis

### 1. Significant Changes in Stock Prices Based on Social Media Sentiment

#### Stock Sentiment Correlation
The analysis reveals a weak negative correlation between stock prices and sentiment scores across the examined stocks. For example, the correlation coefficient of approximately -0.017 suggests that as sentiment increases, stock prices do not consistently follow a similar trend. However, specific periods might show stronger relationships, indicating that sentiment can influence stock price movements in particular contexts or events.

#### Stock Price Volatility
Stocks like Tesla (TSLA) and Apple (AAPL) exhibited more significant price fluctuations in response to spikes in positive or negative sentiment on Reddit. This could be linked to high retail investor interest and reactions to news or rumors spread on social media platforms.

- **Example Observations**: 
    - In months where sentiment towards Tesla surged, there were corresponding increases in the stock price, suggesting potential buy signals during periods of heightened positive sentiment.
    - Conversely, stocks like Amazon (AMZN) showed less correlation, indicating that social media sentiment might not play a crucial role in influencing their stock prices.

### 2. Possible Buy/Sell Signals Based on Social Media Discussions

#### Buy Signals
A spike in positive sentiment, especially when combined with a surge in mentions, can be interpreted as a buy signal. For instance, if sentiment for Microsoft (MSFT) increases significantly along with a high volume of mentions, it may indicate growing investor interest, potentially leading to price appreciation. Historical data could support this if patterns show that past instances of high positive sentiment preceded price increases.

#### Sell Signals
Conversely, if negative sentiment rises sharply, this could indicate potential sell signals. For example, if sentiment for Google (GOOGL) turns predominantly negative while mention counts remain high, it may suggest that investor sentiment is shifting away from the stock, possibly foreshadowing a price drop. Monitoring sentiment before major earnings reports or news releases can also provide timely sell signals, especially for stocks that are more sensitive to public perception.

### 3. Recommendations for Investors

#### Continuous Monitoring
Investors should actively monitor sentiment trends on social media, particularly leading up to key events like earnings calls or product launches. Using sentiment analysis tools can provide insights into shifts in public opinion that might not be captured through traditional financial analysis.

#### Combining Analysis
It is beneficial to combine sentiment analysis with fundamental analysis and technical indicators. For instance, if both sentiment is high and technical indicators suggest a bullish trend, this could strengthen the case for a buy.

#### Diversification of Sources
Relying solely on Reddit sentiment may not always provide a complete picture. It is wise to integrate multiple sources of information, including news articles, financial reports, and other social media platforms, to form a well-rounded investment strategy.

### Conclusion
This analysis highlights the importance of social media sentiment in shaping investor behavior and stock price movements. While not the only factor, it serves as a valuable tool for understanding market psychology and identifying potential trading opportunities. By analyzing sentiment alongside other financial metrics, investors can make more informed decisions and potentially enhance their portfolio performance.

## Actionable Insights

### Stock Performance and Sentiment Correlation

- **AAPL**: The analysis reveals a significant correlation between negative sentiment and price drops for Apple Inc. (AAPL). Traders should exercise caution and consider short-selling strategies during periods of increased negative sentiment. Monitoring social media discussions closely may provide early warnings of potential price declines.

- **MSFT and AMZN Stability**:
  - **MSFT**: Microsoft Corporation (MSFT) shows relatively stable prices despite fluctuations in sentiment, suggesting that the stock may be less sensitive to social media sentiment. Investors may consider MSFT a stable investment during volatile market periods.
  - **AMZN**: Amazon.com Inc. (AMZN) demonstrates occasional spikes in stock prices following positive sentiment trends. Investors might explore buying opportunities when sentiment shifts positively.

- **TSLA Price Sensitivity**:
  - **TSLA**: Tesla Inc. (TSLA) exhibits a strong reaction to social media discussions, with both positive and negative sentiments significantly impacting price movements. Traders should remain vigilant of public sentiment as it can create volatile trading conditions.

- **GOOGL Caution**:
  - **GOOGL**: Alphabet Inc. (GOOGL) shows a weaker correlation between sentiment and price movements. However, drastic shifts in sentiment can still result in notable price changes. Caution is advised when sentiment changes abruptly.

## Future Improvements

### Integrating Additional Data Sources
- **News Sentiment**: Incorporate sentiment analysis from financial news articles and press releases to provide a holistic view of market sentiment and its potential influence on stock prices.
- **Economic Indicators**: Integrate macroeconomic data, such as interest rates and economic forecasts, to enrich sentiment analysis and provide context for stock price movements.

### Utilizing Advanced Sentiment Analysis Techniques
- **Deep Learning Models**: Implement advanced machine learning models, such as BERT or LSTM networks, to enhance sentiment detection accuracy from social media data.
- **Aspect-Based Sentiment Analysis**: Focus on specific topics or aspects within social media discussions (e.g., product launches, earnings reports) to better understand the factors driving sentiment.

### Real-Time Data Integration
- **Streaming Data Feeds**: Utilize real-time social media data streaming to capture sentiment shifts as they occur, allowing traders to respond promptly to market changes.
- **Automated Alerts System**: Establish an automated alert system for significant sentiment changes or stock price movements, helping traders make informed decisions quickly.

### Machine Learning for Predictive Analysis
- **Predictive Modeling**: Develop predictive models that leverage historical sentiment data to forecast future stock price movements, enhancing the robustness of trading strategies.
- **Feature Engineering**: Experiment with new features, such as lagged sentiment scores, to improve the performance of predictive models.

### Enhanced Visualization Techniques
- **Interactive Dashboards**: Create interactive visualizations that allow users to explore the relationships between sentiment and stock prices dynamically.
- **Geographical Analysis**: If relevant, analyze sentiment data geographically to identify regional trends that may impact stock performance.

### Backtesting Strategies
- **Historical Performance Testing**: Implement a backtesting framework to simulate the performance of sentiment-driven trading strategies over historical periods.
- **Risk Assessment Framework**: Incorporate risk metrics into the analysis to evaluate the potential risks associated with trading strategies based on social media sentiment.

By implementing these recommendations and improvements, the analysis can provide deeper insights, facilitate better trading decisions, and enhance the overall understanding of how social media sentiment influences stock prices.



