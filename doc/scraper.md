# X (Twitter) GraphQL Interceptor Scraper

**Author:** xzacekp00 Patrik Žáček  
**Institution:** BUT FIT (Brno University of Technology, Faculty of Information Technology)  
**Thesis Type:** Bachelor's Thesis 2026  
**Topic:** Detection of Fake Accounts on Social Media Networks  
**Last Updated:** May 1, 2026

---

## Functionality

* Extracts timeline data: Date, Text, Likes, Retweets, and Replies.
* Extracts profile metadata: Username, Display Name, Bio, Followers, Following, Verified status, Creation date.
* Utilizes a recursive "aggregator" search pattern to locate data within the JSON, making it highly resilient to X's frequent schema changes.
* Outputs JSON files per user (profile and tweets) to the project's `demo/` folder.
* Supports both single-user scraping (via `scrape_user()`) and batch scraping (via `run_scraper()`).

## Requirements

* Python 3.11+
* `pandas`
* `playwright`

### Installation

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Install the Playwright Chromium browser binaries:
   ```bash
   playwright install
   ```

## Implementation & Setup

### 1. Folder Structure

The script saves output files to the project's `demo/` folder regardless of where it's executed from. The path is resolved using:
```python
SCRIPT_DIR = Path(__file__).resolve().parent  # src/
PROJECT_ROOT = SCRIPT_DIR.parent              # project root
DEMO_DIR = PROJECT_ROOT / 'demo'              # project/demo/
```


### 2. Cookie Extraction (Authentication Bypass)

To run the script, you must provide active authentication cookies from a logged-in X account. Do not use your primary personal account for this to avoid potential account locks.

1. Open a standard web browser (Chrome, Firefox, Edge) and log into your designated X account.
2. Open Developer Tools (F12 or Right-Click -> Inspect).
3. Navigate to the Application tab (Chrome/Edge) or Storage tab (Firefox).
4. Under the Cookies section, select `https://x.com`.
5. Locate and copy the values for two specific cookies:
   * `auth_token`
   * `ct0`
6. Paste these values into the `AUTH_TOKEN` and `CT0_TOKEN` variables at the top of `src/scrape.py`.

### 3. Configuration

#### For Single-User Scraping (Live Mode)
Used by `bot_detector.py` in live mode:
```python
from scrape import scrape_user

scrape_user("NASA") 
```

#### For Batch Scraping
Define target accounts in `src/scrape.py`:
```python
TARGET_ACCOUNTS = ["Charles_leclerc", "NASA", "elonmusk"]
```

Then run:
```bash
python src/scrape.py
```

## Output Format

### Profile JSON (`profile_<username>.json`)
```json
{
  "created_at": "2011-07-16 00:00:00+00:00",
  "description": "Exploring the cosmos",
  "name": "NASA",
  "public_metrics": {
    "followers_count": 50000000,
    "following_count": 150,
    "tweet_count": 12000,
    "listed_count": 500000
  },
  "username": "NASA",
  "verified": true
}
```

### Tweets JSON (`tweets_<username>.json`)
```json
[
  {
    "Date": "2026-04-15T10:30:00Z",
    "Text": "Exoplanet discovery announcement",
    "Likes": 50000,
    "Retweets": 25000,
    "Replies": 5000,
    "Author": "NASA"
  },
  ...
]
```

## Important Remarks

* **Headed Mode:** The Playwright instance is set to `headless=False` by design. This makes the browser visible, which helps avoid basic headless-browser detection scripts used by Arkose Labs on X.
* **Cookie Expiration:** The `auth_token` and `ct0` cookies will eventually expire or be invalidated by X (typically if you log out manually from the browser where you extracted them). If the script fails to load the timelines, you will need to extract fresh cookies.
* **Rate Limiting:** Introduce deliberate delays between profile navigations (already included in the script via `page.wait_for_timeout()`). Navigating through hundreds of profiles too rapidly will result in X temporarily rate-limiting the account associated with the cookies.
* **Data Privacy:** In live mode, scraped data is not persisted to disk by `bot_detector.py`, temporary files are created and deleted after analysis.

---

## Status

**Last Tested:** May 5th, 2026  
**Status:** Production-ready, integrated with bot detection ensemble  
**Integration Points:**
- Used by `bot_detector.py` with `--mode live` flag
- Called via `scrape_user(username)` function
- Batch scraping available via `run_scraper()` function
