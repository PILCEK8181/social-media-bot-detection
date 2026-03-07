from playwright.sync_api import sync_playwright
import pandas as pd
import json

# CONFIG
AUTH_TOKEN = "d96a295cea4ec0f0afd07b550d9d1064988f2eb6"
CT0_TOKEN = "6240dd8ac9879a1ae1eb7f146dc3bd241067cb895606d764ac585728d4c0eb95bb2ae156e17bea422fd96d8c96b50a7e02467e6623220d26ac1ea9e0aa1d71370532eae73b2ea198ecbad76025f973a7"

# todo - one is enough?
TARGET_ACCOUNTS = ["nytimes"] # location test

captured_tweets = []
current_profile_data = {} 
CURRENT_TARGET = "" 

# template
def reset_profile_data():
    global current_profile_data
    current_profile_data = {
        "Username": CURRENT_TARGET,
        "Display Name": "Unknown",
        "Bio": "",
        "Followers": 0,
        "Following": 0,
        "Location": "Not provided",
        "Verified": False
    }

# tweets 
def extract_tweets_recursive(obj):
    if isinstance(obj, dict):
        if 'full_text' in obj and 'created_at' in obj:
            captured_tweets.append({
                "Date": obj.get("created_at"),
                "Text": obj.get("full_text"),
                "Likes": obj.get("favorite_count"),
                "Retweets": obj.get("retweet_count"),
                "Replies": obj.get("reply_count")
            })
        for key, value in obj.items():
            extract_tweets_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            extract_tweets_recursive(item)

# profile data
def extract_profile_aggregate(obj):
    global current_profile_data
    if isinstance(obj, dict):
        # 1. Scavenge the Name
        if obj.get('screen_name', '').lower() == CURRENT_TARGET.lower():
            current_profile_data['Display Name'] = obj.get('name', current_profile_data['Display Name'])
            
        # 2. Scavenge the Location 
        if 'location' in obj and obj.get('location'):
            if current_profile_data['Location'] == "Not provided":
                current_profile_data['Location'] = obj.get('location')
                
        # 3. Scavenge the Metrics and bio
        if 'followers_count' in obj and current_profile_data['Followers'] == 0:
            current_profile_data['Followers'] = obj.get('followers_count')
            current_profile_data['Following'] = obj.get('friends_count', 0)
            current_profile_data['Bio'] = obj.get('description') or current_profile_data['Bio']
            
        # 4. Scavenge Verification Status // check both 'is_blue_verified' and 'verified' to cover different API versions
        if 'is_blue_verified' in obj and current_profile_data['Verified'] is False:
            current_profile_data['Verified'] = obj.get('is_blue_verified')
        elif 'verified' in obj and current_profile_data['Verified'] is False:
            current_profile_data['Verified'] = obj.get('verified')

        for key, value in obj.items():
            extract_profile_aggregate(value)
            
    elif isinstance(obj, list):
        for item in obj:
            extract_profile_aggregate(item)

def handle_response(response):
    url = response.url
    if "graphql" in url and response.request.method == "GET":
        try:
            # 1. Profile Data
            if "UserByScreenName" in url or "UserByRestId" in url:
                print("Profile data intercepted, scavenging metadata...")
                data = response.json()
                extract_profile_aggregate(data) 
                
            # 2. Timeline Data
            elif "UserTweets" in url:
                print(f"Intercepted timeline data, getting tweets...")
                data = response.json()
                extract_tweets_recursive(data)

        except Exception as e:
            pass 

def run_scraper():
    global CURRENT_TARGET
    
    with sync_playwright() as p:
        # debug: headless=False to see the browser in action, can switch to True for production
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720}
        )
        
        context.add_cookies([
            {"name": "auth_token", "value": AUTH_TOKEN, "domain": ".x.com", "path": "/"},
            {"name": "ct0", "value": CT0_TOKEN, "domain": ".x.com", "path": "/"},
            {"name": "auth_token", "value": AUTH_TOKEN, "domain": ".twitter.com", "path": "/"},
            {"name": "ct0", "value": CT0_TOKEN, "domain": ".twitter.com", "path": "/"}
        ])

        page = context.new_page()
        page.on("response", handle_response)

        for account in TARGET_ACCOUNTS:
            CURRENT_TARGET = account
            print(f"\nNavigating to @{account}...")
            captured_tweets.clear()
            reset_profile_data() # reset template for new account
            
            page.goto(f"https://x.com/{account}")
            page.wait_for_timeout(4000) 
            
            page.mouse.wheel(0, 3000)
            print("Waiting for network data to parse...")
            page.wait_for_timeout(5000) 
            
            # save tweets
            if captured_tweets:
                df_tweets = pd.DataFrame(captured_tweets)
                df_tweets['Author'] = account 
                tweets_filename = f"../temp/tweets_{account}.csv"
                df_tweets.to_csv(tweets_filename, index=False, encoding='utf-8')
                print(f"SUCCESS! Saved {len(df_tweets)} tweets to {tweets_filename}.")
            else:
                print(f"No tweets found for {account}.")

            # save profile data
            # check
            if current_profile_data['Followers'] > 0:
                df_profile = pd.DataFrame([current_profile_data])
                profile_filename = f"../temp/profile_{account}.csv"
                df_profile.to_csv(profile_filename, index=False, encoding='utf-8')
                print(f"SUCCESS! Saved profile metadata to {profile_filename}.")
            else:
                print(f"No profile metadata found for {account}.")

        print("\nClosing browser.")
        browser.close()

if __name__ == "__main__":
    run_scraper()