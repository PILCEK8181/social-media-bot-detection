import os
import ijson
import json
import random

# Config
DATA_DIR = './data/twibot22/'
TWEET_FILES = [os.path.join(DATA_DIR, f'tweet_{i}.json') for i in range(9)]
OUTPUT_FILE = './temp/tweet_example.json'

# Max tweets to collect
MAX_TWEETS = 20

# Pick a random tweet file
tweet_file = random.choice(TWEET_FILES)
print(f"Randomly selected: {os.path.basename(tweet_file)}")
print(f"Scanning for a random user...")

# Find a random user from the selected tweet file // maebe later TODO
target_user_id = None
collected_tweets = []

try:
    with open(tweet_file, 'rb') as f:
        parser = ijson.items(f, 'item')
        
        count = 0
        for tweet in parser:
            # If we haven't found a user yet, use the first tweet's user
            if target_user_id is None:
                target_user_id = str(tweet.get('uid', ''))
                print(f"Found random user: {target_user_id}")
            
            tweet_uid = str(tweet.get('uid', ''))
            
            if tweet_uid == target_user_id:
                collected_tweets.append(tweet)
                print(f" -> Found tweet {len(collected_tweets)}/{MAX_TWEETS}")
                
                if len(collected_tweets) >= MAX_TWEETS:
                    break
            
            count += 1
            if count % 100000 == 0:
                print(f" -> Scanned {count:,} tweets...")

except FileNotFoundError:
    print(f"File {tweet_file} does not exist.")
    exit(1)

# Save collected tweets
if collected_tweets:
    os.makedirs('./temp', exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        json.dump(collected_tweets, f_out, indent=4, ensure_ascii=False)
    
    print(f"\n=== Success ===")
    print(f"User ID: {target_user_id}")
    print(f"Tweets collected: {len(collected_tweets)}")
    print(f"Saved to: {OUTPUT_FILE}")
else:
    print(f"\nNo tweets found.")
