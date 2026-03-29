from playwright.sync_api import sync_playwright
import pandas as pd
import json

# CONFIG, set your own tokens here
AUTH_TOKEN = "XXX"
CT0_TOKEN = "XXX"

TARGET_ACCOUNTS = ["Ahoj1", "Charles_leclerc", "elonmusk", "BarackObama", "JoeBiden", "BillGates", "ladygaga", "Cristiano", "rihanna", "KimKardashian"] 

captured_tweets = []
current_profile_data = {} 
CURRENT_TARGET = "" 

# template 
def reset_profile_data():
    global current_profile_data
    current_profile_data = {
        "created_at": "Unknown",
        "description": "",
        "name": "Unknown",
        "public_metrics": {
            "followers_count": 0,
            "following_count": 0,
            "tweet_count": 0,
            "listed_count": 0
        },
        "username": CURRENT_TARGET,
        "verified": False,
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
        # Extract fields scoped to the target user
        if obj.get('screen_name', '').lower() == CURRENT_TARGET.lower():
            current_profile_data['name'] = obj.get('name', current_profile_data['name'])
            current_profile_data['created_at'] = obj.get('created_at', current_profile_data['created_at'])

        # Extract metrics and description
        if 'followers_count' in obj and current_profile_data['public_metrics']['followers_count'] == 0:
            current_profile_data['public_metrics']['followers_count'] = obj.get('followers_count', 0)
            current_profile_data['public_metrics']['following_count'] = obj.get('friends_count', 0)
            current_profile_data['public_metrics']['tweet_count'] = obj.get('statuses_count', 0)
            current_profile_data['public_metrics']['listed_count'] = obj.get('listed_count', 0)
            current_profile_data['description'] = obj.get('description') or current_profile_data['description']
            
        # Extract verification status
        if 'is_blue_verified' in obj and current_profile_data['verified'] is False:
            current_profile_data['verified'] = obj.get('is_blue_verified')
        elif 'verified' in obj and current_profile_data['verified'] is False:
            current_profile_data['verified'] = obj.get('verified')

        # Dig deeper
        for key, value in obj.items():
            extract_profile_aggregate(value)
            
    elif isinstance(obj, list):
        for item in obj:
            extract_profile_aggregate(item)

def handle_response(response):
    url = response.url
    if "graphql" in url and response.request.method == "GET":
        try:
            if "UserByScreenName" in url or "UserByRestId" in url:
                print("Profile data intercepted, scavenging metadata...")
                data = response.json()
                extract_profile_aggregate(data) 
                
            elif "UserTweets" in url:
                print(f"Intercepted timeline data, getting tweets...")
                data = response.json()
                extract_tweets_recursive(data)
        except Exception as e:
            pass 

def run_scraper():
    global CURRENT_TARGET
    
    with sync_playwright() as p:
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
            reset_profile_data() 
            
            page.goto(f"https://x.com/{account}")
            page.wait_for_timeout(4000) 
            
            # Hydration bypass
            print("Bypassing HTML cache to force Page 1 download...")
            try:
                page.locator(f"a[href='/{account}/with_replies']").first.click()
                page.wait_for_timeout(2000)
                page.locator(f"a[href='/{account}']").filter(has_text="Posts").first.click()
                page.wait_for_timeout(3000)
            except Exception as e:
                print("Tab toggle failed, relying on scroll...")
            
            page.mouse.wheel(0, 3000)
            print("Waiting for network data to parse...")
            page.wait_for_timeout(5000) 
            
            # --- SAVE TWEETS ---
            if captured_tweets:
                df_tweets = pd.DataFrame(captured_tweets)
                df_tweets['Author'] = account 
                df_tweets['Date'] = pd.to_datetime(df_tweets['Date'], format='%a %b %d %H:%M:%S %z %Y')
                df_tweets = df_tweets.sort_values(by='Date', ascending=False)
                df_tweets = df_tweets.drop_duplicates(subset=['Text'])
                df_tweets = df_tweets.head(20)
                
                tweets_filename = f"../demo/tweets_{account}.json"
                # date_format='iso' keeps the JSON dates highly readable
                df_tweets.to_json(tweets_filename, orient='records', indent=2, date_format='iso')
                print(f"SUCCESS! Saved {len(df_tweets)} newest tweets to {tweets_filename}.")
            else:
                print(f"No tweets found for {account}.")
            
            # --- FORMAT PROFILE DATES & SAVE ---
            if current_profile_data['public_metrics']['followers_count'] >= 0:
                
                # Convert the created_at string right before saving
                if current_profile_data['created_at'] != 'Unknown':
                    try:
                        dt = pd.to_datetime(current_profile_data['created_at'], format='%a %b %d %H:%M:%S %z %Y')
                        current_profile_data['created_at'] = dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
                    except Exception:
                        pass
                
                profile_filename = f"../demo/profile_{account}.json"
                with open(profile_filename, 'w', encoding='utf-8') as f:
                    json.dump(current_profile_data, f, ensure_ascii=False, indent=2)
                print(f"SUCCESS! Saved profile data to {profile_filename}.")
            else:
                print(f"No profile metadata found for {account} (Network might be slow).")

        print("\nClosing browser.")
        browser.close()

if __name__ == "__main__":
    run_scraper()