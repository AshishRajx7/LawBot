# collect_cases.py
import requests
from bs4 import BeautifulSoup
import csv
import time

# Base search URL for Indian Kanoon
BASE_URL = "https://www.indiankanoon.org/search/?formInput="

# 🔹 Topics you can modify (we’ll collect few judgments per topic)
SEARCH_TERMS = [
    "Article 21",
    "Article 14",
    "Right to Education",
    "Fundamental Rights",
    "Reservation Policy",
]

OUTPUT_FILE = "data/cases.csv"

# Create output folder if missing
import os
os.makedirs("data", exist_ok=True)

def fetch_case_links(query):
    """Get top case URLs for a query"""
    url = BASE_URL + requests.utils.quote(query)
    resp = requests.get(url, headers={"User-Agent": "LawBot-Collector/1.0"})
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for link in soup.select(".result_title a")[:5]:  # top 5 per query
        case_title = link.text.strip()
        case_url = "https://www.indiankanoon.org" + link.get("href")
        results.append((case_title, case_url))
    return results

def fetch_case_summary(case_title, case_url):
    """Fetch summary from a case page"""
    resp = requests.get(case_url, headers={"User-Agent": "LawBot-Collector/1.0"})
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract first few paragraphs (skipping repetitive metadata)
    paras = [p.get_text(" ", strip=True) for p in soup.select("p")[:6]]
    text = " ".join(paras)
    return {
        "case_name": case_title,
        "url": case_url,
        "verdict_summary": text[:4000],  # limit length
    }

def main():
    print("⚖️ Starting Indian Kanoon case collector...")
    all_cases = []
    for term in SEARCH_TERMS:
        print(f"🔍 Searching for cases related to: {term}")
        cases = fetch_case_links(term)
        for title, link in cases:
            print(f"   ↳ Fetching: {title}")
            try:
                data = fetch_case_summary(title, link)
                all_cases.append(data)
                time.sleep(1)  # be polite
            except Exception as e:
                print("     ⚠️ Error:", e)
                continue

    # Save to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_name", "url", "verdict_summary"])
        writer.writeheader()
        writer.writerows(all_cases)

    print(f"\n✅ Saved {len(all_cases)} cases to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
