import praw
import pandas as pd
from datetime import datetime, timedelta

# Authenticate with the Reddit API
reddit = praw.Reddit(client_id = 'insert your client_id here',
                     client_secret = 'insert your client_secret here',
                     user_agent = 'insert your user_agent here')

# Define the subreddit 
subreddit = reddit.subreddit('worldnews') # subreddit 
required_votes = 500                   

# Retrieve posts
posts = subreddit.new(limit=800)

# Create an empty DataFrame to store the posts
posts_df = pd.DataFrame(columns=['title', 'url', 'votes'])

# Iterate through the posts and store the required data in the DataFrame
for post in posts:
    if post.score >= required_votes:  # Filter based on required number of votes
        posts_df = posts_df.append({'title': post.title, 'url': post.url, 'votes': post.score}, ignore_index=True)

file_path = 'posts.xlsx'

posts_df.to_excel(file_path, index=False)



