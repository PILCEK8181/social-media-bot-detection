# Bot Detector CLI - Comprehensive Usage Guide

Patrik Žáček - xzacekp00
BUT FIT (Brno University of Technology, Faculty of Information Technology) 
Bachelor's Thesis 2026 - Detection of Fake Accounts on Social Media Networks 

The `bot_detector.py` inference engine is a command-line interface designed for both real-time and offline analysis of Twitter/X accounts. The tool processes a target account through a multi-modal machine learning architecture, calculating a composite probability score to determine if the account exhibits structural and behavioral patterns consistent with automated bots.

## Configurable Parameters

The behavior of the inference engine is highly customizable to accommodate different analytical needs. The tool accepts the following command-line arguments:

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--target` | String | The target username to analyze (e.g., `@elonmusk`). The leading `@` symbol is optional. |
| `--mode` | String | Execution mode. Accepts either `live` (initiates real-time scraping) or `demo` (utilizes local offline files). |
| `--threshold` | Float | Probability threshold for the final binary classification. Defaults to `0.5`. Adjusting this allows the user to prioritize precision (higher threshold) or recall (lower threshold). |
| `--verbose` | Flag | If provided, enables detailed logging of the internal feature extraction process, dynamic calculations, and intermediate model artifact loading. |

### Note on Demo Mode Execution
When executing the script utilizing `--mode demo`, the live scraping pipeline is completely bypassed. This mode requires the presence of locally cached data files for the specified target user. These files must be located inside the `demo/` directory of this repository. This mode is specifically designed for testing the inference engine without relying on external network requests or facing potential API rate limits.

---

## Output Specification

To ensure seamless interoperability with external analytical tools, dashboards, or automated moderation pipelines, the tool simultaneously produces two types of output:

### 1. Standard Console Output (`stdout`)
A human-readable summary printed directly to the terminal. It highlights the final classification label (`BOT` or `HUMAN`), the overall confidence score of the meta-classifier, and a breakdown of the raw probabilities extracted from the individual modality branches.

### 2. Structured Output (`results.json`)
A comprehensive, machine-readable JSON record saved directly to the disk in the current working directory. The structure provides total transparency into the ensemble's decision-making process. 

**Example JSON Structure:**
```json
{
  "username": "Charles_leclerc",
  "prediction": "HUMAN",
  "probability": 0.07318698147537055,
  "threshold": 0.7,
  "modality_scores": {
    "metadata": 0.009407240508423429,
    "text": 0.05896920710802078
  },
  "timestamp": "2026-05-01T12:34:29.821522+00:00",
  "display_name": "Charles Leclerc",
  "followers": 3798437,
  "following": 188,
  "tweets": 2406,
  "verified": true
}
```

---

## Usage Examples

### Example 1: Live Analysis with Custom Threshold
This command initiates a real-time query against the platform for a specific user, applying a strict probability threshold of `0.70` before assigning the `BOT` label.

**Command:**
```bash
python bot_detector.py --mode live --target @suspect_user --threshold 0.7
```

**Expected Output:**
```text
[INFO] Fetching timeline for @suspect_user...
[INFO] Analyzed 100 tweets.
[INFO] Extracting features...
--------------------------------------------------
REPORT for @suspect_user
--------------------------------------------------
Verdict: BOT
Confidence: 0.89
Breakdown: Metadata(0.45) | Text(0.92) | Temporal(0.85)
--------------------------------------------------
[INFO] Result saved to results.json
--------------------------------------------------
```

### Example 2: Demo Execution with Verbose Logging
This command tests the pipeline locally using the pre-downloaded files in the `demo/` directory. The `--verbose` flag ensures that the exact feature arrays passed to the Random Forest and the tokenized inputs passed to the RoBERTa model are printed to the console for debugging purposes.

**Command:**
```bash
python bot_detector.py --mode demo --target @sample_user --verbose
```