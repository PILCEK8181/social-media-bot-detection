# X (Twitter) GraphQL Interceptor Scraper

## Functionality
* Extracts timeline data: Date, Text, Likes, Retweets, and Replies.
* Extracts profile metadata: Username, Display Name, Bio, Followers, Following, Location, and Verified status.
* Utilizes a recursive "aggregator" search pattern to locate data within the JSON, making it highly resilient to X's frequent schema changes.
* Outputs two separate CSV files per user (one for tweets, one for profile data) into a designated temporary directory.

## Requirements
* Python 3.8+
* `pandas`
* `playwright`

### Installation
1. Install the required Python packages:
   `pip install pandas playwright`
2. Install the Playwright Chromium browser binaries:
   `python -m playwright install chromium`

## Implementation & Setup

### 1. Folder Structure
The script is configured to save output files to a relative directory named `../temp/`. 
Ensure this directory exists relative to where you are running the script, or modify the output paths in the script to match your desired destination. Pandas will throw a `FileNotFoundError` if the output directory does not exist.

### 2. Cookie Extraction (Authentication Bypass)
To run the script, you must provide active authentication cookies from a logged-in X account. Do not use your primary personal account for this to avoid potential account locks.
1. Open a standard web browser (Chrome, Firefox, Edge) and log into your designated X account.
2. Open Developer Tools (F12 or Right-Click -> Inspect).
3. Navigate to the Application tab (Chrome/Edge) or Storage tab (Firefox).
4. Under the Cookies section, select `https://x.com`.
5. Locate and copy the values for two specific cookies:
   * `auth_token`
   * `ct0`
6. Paste these values into the `AUTH_TOKEN` and `CT0_TOKEN` variables at the top of the Python script.

### 3. Configuration
Define the target accounts by modifying the `TARGET_ACCOUNTS` list in the script. Use the exact handles without the "@" symbol.

Example:
`TARGET_ACCOUNTS = ["nytimes"]`

## Important Remarks
* Headed Mode: The Playwright instance is set to `headless=False` by design. This makes the browser visible, which helps avoid basic headless-browser detection scripts used by Arkose Labs on X. 
* Cookie Expiration: The `auth_token` and `ct0` cookies will eventually expire or be invalidated by X (typically if you log out manually from the browser where you extracted them). If the script fails to load the timelines, you will need to extract fresh cookies.
* Rate Limiting: Introduce deliberate delays (e.g., `page.wait_for_timeout()`) between profile navigations. Navigating through hundreds of profiles too rapidly will result in X temporarily rate-limiting the account associated with the cookies.


# STATUS

## Last tested 

3/7/2026

## Current metadata output CSV format

Username,Display Name,Bio,Followers,Following,Location,Verified
nytimes,The New York Times,News tips? Share them here: https://t.co/ghL9OoYKMM,53018618,842,{'location': 'New York City'},True

## TODO

Argparsing
Finish up the features
Format the tweets
Pipeline it into project
