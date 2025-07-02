import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import os

BASE_URL = (
    "https://emm.newsbrief.eu/NewsBrief/dynamic"
    "?language=en"
    "&edition=categoryarticles"
    "&option=FoodSafety"
    "&page={page_num}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}

def debug_page_content(page_num, html_content):
    print("=" * 50)
    print(f"Debug snippet for page {page_num}:")
    print(html_content[:500])
    print("=" * 50)

def scrape_all_pages_until_empty(max_hard_limit=5):
    page_num = 1
    all_links = []

    while page_num <= max_hard_limit:
        url = BASE_URL.format(page_num=page_num)
        print(f"Scraping page {page_num}: {url}")
        resp = requests.get(url, headers=HEADERS)

        if resp.status_code != 200:
            print(f"Stopped at page {page_num} - status code {resp.status_code}")
            break

        html_content = resp.text
        debug_page_content(page_num, html_content)

        soup = BeautifulSoup(html_content, "html.parser")
        article_containers = soup.find_all("div", class_="articlebox_big")
        if not article_containers:
            print(f"No article containers found on page {page_num}, stopping.")
            break

        print(f"Found {len(article_containers)} article containers on page {page_num}.")
        links_found_on_page = 0
        for article in article_containers:
            a_tag = article.find("a", href=True)
            if a_tag:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                all_links.append({"URL": a_tag["href"], "Scrape Date": current_time})
                links_found_on_page += 1
            else:
                print("No <a> tag found in one of the article containers.")

        if links_found_on_page == 0:
            print(f"No links extracted from page {page_num}, stopping.")
            break

        print(f"Extracted {links_found_on_page} links from page {page_num}.")
        page_num += 1

    return all_links

def save_links_to_json(links, json_filename="food_safety_links.json"):
    """
    Save the list of dictionaries (link info) to a JSON file.
    Appends to the file if it already exists.
    """
    # Load existing data if file exists
    if os.path.exists(json_filename):
        with open(json_filename, "r", encoding="utf-8") as f:
            try:
                existing_links = json.load(f)
            except json.JSONDecodeError:
                existing_links = []
    else:
        existing_links = []
    # Combine and remove duplicates (optional)
    all_links = existing_links + links
    # Optionally deduplicate by URL
    seen = set()
    deduped_links = []
    for item in all_links:
        if item["URL"] not in seen:
            deduped_links.append(item)
            seen.add(item["URL"])
    # Save back to file
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(deduped_links, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(links)} new records to {json_filename} (total {len(deduped_links)} records).")

def main():
    while True:
        print("\n--- Starting a new scraping session ---")
        scraped_links = scrape_all_pages_until_empty(max_hard_limit=5)
        print(f"Total links scraped this session: {len(scraped_links)}")
        save_links_to_json(scraped_links)
        print("Scraping session completed. Waiting 24 hours for the next run...\n")
        time.sleep(86400)

if __name__ == "__main__":
    main()
