import os
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import json
from datetime import datetime

def scrape_url(url):
    print(f"[DEBUG] Scraping URL: {url}")
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/58.0.3029.110 Safari/537.3'
        ),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if "text/html" not in content_type:
            print(f"[DEBUG] URL {url} returned non-HTML content: {content_type}")
            return None, f"Non-HTML content: {content_type}"

        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form", "noscript"]):
            tag.decompose()
        main_content = soup.find('main')
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        text = " ".join(text.split())
        print(f"[DEBUG] Successfully scraped URL: {url}")
        return text, None
    except RequestException as e:
        print(f"[DEBUG] Error scraping URL: {url} | Error: {str(e)}")
        return None, str(e)

def process_json_file(json_path, output_directory):
    print(f"[DEBUG] Processing file: {json_path}")
    # Load the JSON input
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            url_records = json.load(f)
    except Exception as e:
        print(f"[DEBUG] Error reading {json_path}: {e}")
        return

    # Remove duplicate URLs
    seen_urls = set()
    unique_records = []
    for rec in url_records:
        url = rec.get('URL')
        if url and url not in seen_urls:
            unique_records.append(rec)
            seen_urls.add(url)
    print(f"[DEBUG] Number of unique URLs after deduplication: {len(unique_records)}")

    # Create output filenames using today's date.
    today_str = datetime.now().strftime("%Y%m%d")
    basename = os.path.splitext(os.path.basename(json_path))[0]
    output_filename = f'{basename}_{today_str}_output.json'
    error_filename = f'{basename}_{today_str}_error_log.json'
    output_path = os.path.join(output_directory, output_filename)
    error_path = os.path.join(output_directory, error_filename)

    print(f"[DEBUG] Output file will be: {output_path}")
    print(f"[DEBUG] Error log file will be: {error_path}")

    results = []
    errors = []

    # Process each record
    for idx, rec in enumerate(unique_records):
        url = rec['URL']
        scrape_date = rec.get('Scrape Date', '')
        print(f"[DEBUG] Processing URL {idx}: {url}")
        content, error = scrape_url(url)
        if error:
            errors.append({
                'URL': url,
                'Scrape Date': scrape_date,
                'Error': error
            })
        else:
            results.append({
                'URL': url,
                'Scrape Date': scrape_date,
                'Content': content
            })

    # Write results and errors to JSON files.
    with open(output_path, mode='w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=4)
    with open(error_path, mode='w', encoding='utf-8') as ferr:
        json.dump(errors, ferr, ensure_ascii=False, indent=4)

    print(f"[DEBUG] Finished processing. Output saved to {output_path}")
    return output_path

def main():
    directory = './'
    json_file = 'food_safety_links.json'
    json_path = os.path.join(directory, json_file)
    if not os.path.exists(json_path):
        print(f"[DEBUG] File not found: {json_path}")
        return
    print(f"[DEBUG] Found file: {json_path}")
    process_json_file(json_path, directory)

if __name__ == "__main__":
    main()
