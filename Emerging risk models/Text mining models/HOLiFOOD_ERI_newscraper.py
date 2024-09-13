import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import csv
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        logging.info(f"Scraping URL: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # Check if the Content-Type header is 'text/html'
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('text/html'):
            logging.warning(f"Invalid Content-Type: {content_type} for URL: {url}")
            return None, f"Invalid Content-Type: {content_type}"

        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text, None
    except RequestException as e:
        logging.error(f"Error scraping URL: {url} - {e}")
        return None, str(e)

def process_file(file_path, output_directory):
    logging.info(f"Processing file: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error reading file: {file_path} - {e}")
        return
    
    if 'url' not in df.columns or 'date' not in df.columns:
        logging.error(f"The CSV file {file_path} must contain both 'url' and 'date' columns.")
        return

    basename = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_directory, f'{basename}_output.csv')
    error_path = os.path.join(output_directory, f'{basename}_error_log.csv')

    with open(output_path, mode='w', newline='', encoding='utf-8') as f, \
         open(error_path, mode='w', newline='', encoding='utf-8') as err_f:
        fieldnames = ['url', 'date', 'content']
        error_fieldnames = ['url', 'date', 'error']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        error_writer = csv.DictWriter(err_f, fieldnames=error_fieldnames, quoting=csv.QUOTE_MINIMAL)
        
        writer.writeheader()
        error_writer.writeheader()

        for index, row in df.iterrows():
            url = row['url']
            date = row['date']
            logging.info(f"Processing URL: {url}")
            content, error = scrape_url(url)
            if error:
                error_writer.writerow({'url': url, 'date': date, 'error': error})
                logging.warning(f"Logged error for URL: {url}")
            else:
                writer.writerow({'url': url, 'date': date, 'content': content})
                logging.info(f"Successfully scraped and logged URL: {url}")

def main():
    # Get the directory where the script is located
    directory = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(directory, 'scraped_output')
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    logging.info(f"Starting processing of directory: {directory}")
    
    # Print the contents of the directory
    logging.info(f"Contents of directory {directory}: {os.listdir(directory)}")
    
    csv_files_found = False
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            csv_files_found = True
            file_path = os.path.join(directory, filename)
            logging.info(f"Found CSV file: {file_path}")
            process_file(file_path, output_directory)
    if not csv_files_found:
        logging.info("No CSV files found in the directory.")
    logging.info("Completed processing all files.")

if __name__ == "__main__":
    main()
