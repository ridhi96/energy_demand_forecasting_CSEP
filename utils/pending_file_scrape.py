from webscrape import download_reports
import re
import os

def pending_download():
    """
    This function returns all the pending yearly report urls that need to be downloaded from a text file
    """
    links = {}
    year = None

    # regex pattern to match folder names
    year_pattern = r'Downloading for folder (\d{4}-\d{2})'

    #regex pattern to match pdf links of files not downloaded 
    link_pattern = r'https:\/\/[^\s,]+'

    with open('/Users/ridhipurohit/Documents/GitHub/energy_project_CSEP/energy_modeling/data/not_downloaded_3.txt', "r") as file:
        for line in file:
            match = re.search(year_pattern, line)
            if match:
                year = match.group(1)
                links[year] = set()

            # find links and add to dictionary under year identified previously
            link_matches = re.finditer(link_pattern, line)
            for link in link_matches:
                if year:
                    links[year].add(link.group())

    return links


def main():
    
    #get pending report page urls
    yearly_urls = pending_download()

    #download pending yearly reports
    for year, urls in yearly_urls.items():
        file_path = os.path.join('/Users/ridhipurohit/Documents/GitHub/energy_project_CSEP/india_energy_data', year)
        print(f"Downloading for folder {year} ")
        download_reports(urls, file_path)
    
if __name__ == "__main__":
    main()