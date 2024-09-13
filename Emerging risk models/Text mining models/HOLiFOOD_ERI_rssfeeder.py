import feedparser
import csv
from datetime import datetime, timedelta
import time

# Get the current system date
def generate_rss_urls():
    current_date = datetime.utcnow().date()
    return [
        f"https://emm.newsbrief.eu/rss/rss?language=en&type=search&mode=advanced&dateto={current_date}T05%3A59%3A59Z&datefrom={current_date}T00%3A00%3A00Z&category=FoodSafety",
        f"https://emm.newsbrief.eu/rss/rss?language=en&type=search&mode=advanced&dateto={current_date}T11%3A59%3A59Z&datefrom={current_date}T06%3A00%3A00Z&category=FoodSafety",
        f"https://emm.newsbrief.eu/rss/rss?language=en&type=search&mode=advanced&dateto={current_date}T17%3A59%3A59Z&datefrom={current_date}T12%3A00%3A00Z&category=FoodSafety",
        f"https://emm.newsbrief.eu/rss/rss?language=en&type=search&mode=advanced&dateto={current_date}T23%3A59%3A59Z&datefrom={current_date}T18%3A00%3A00Z&category=FoodSafety"
    ]

# Function to parse RSS feeds and extract URLs and publication dates
def parse_rss_feeds(rss_urls):
    news_data = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            news_data.append({
                "url": entry.link,
                "date": entry.published if 'published' in entry else 'No date available'
            })
    return news_data

# Save URLs and dates to a CSV file
def save_data_to_csv(news_data, filename="news_data.csv"):
    seen_urls = set()
    try:
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            for row in reader:
                seen_urls.add(row[0])
    except FileNotFoundError:
        pass

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["URL", "Date"])  # Write header if file is empty

        for data in news_data:
            if data["url"] not in seen_urls:
                writer.writerow([data["url"], data["date"]])
                seen_urls.add(data["url"])

# Run the script continuously
while True:
    rss_urls = generate_rss_urls()
    news_data = parse_rss_feeds(rss_urls)
    save_data_to_csv(news_data)
    print("URLs and dates appended to news_data.csv")
    time.sleep(86400)  # Wait for 1 day before running again
