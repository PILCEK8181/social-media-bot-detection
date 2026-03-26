import os
import ijson
import json
import random

# Config
DATA_DIR = './data/twibot22/'
USER_FILE = os.path.join(DATA_DIR, 'user.json')
OUTPUT_FILE = './temp/user_example.json'

# Set to a specific ID (e.g., 'u1001') to search for a specific user. 
# Set to 'r' to pick a random user.
# Leave as None to grab the first user.
TARGET_USER_ID = 'r' 

print(f"Scanning {USER_FILE}...")

# Determine behavior based on TARGET_USER_ID
target_index = -1
if TARGET_USER_ID == 'r':
    # TwiBot-22 has ~1,182,679 users. Pick a random index.
    target_index = random.randint(0, 1000000) 
    print(f"Random mode selected. Searching for user at index: {target_index:,}")

try:
    with open(USER_FILE, 'rb') as f_in:
        parser = ijson.items(f_in, 'item')
        
        count = 0
        for user in parser:
            current_id = str(user.get('id', ''))
            
            # Check match conditions
            is_target_id = (current_id == TARGET_USER_ID)
            is_target_index = (TARGET_USER_ID == 'r' and count == target_index)
            is_first = (TARGET_USER_ID is None)

            if is_target_id or is_target_index or is_first:
                print(f"\n=== User Found (scanned {count + 1:,} records) ===")
                print(f"User ID: {current_id}")
                
                # Save the user dictionary to a JSON file
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
                    json.dump(user, f_out, indent=4, ensure_ascii=False)
                
                print(f"User successfully saved to: {OUTPUT_FILE}")
                break
            
            count += 1
            if count % 100000 == 0:
                print(f" -> Scanned {count:,} users...")

except FileNotFoundError:
    print(f"File {USER_FILE} does not exist.")
except Exception as e:
    print(f"Error reading file: {e}")