
import os
import requests
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin
import re
import time

def get_yearly_report_urls(snippet):
    """
    This function takes an html snippet corresponding to URLs for annual report pages and returns a
    list of URLs to linked pages for yearly reports.

    Parameters:
        * snippet:  HTML snippet to page of yearly report links

    Returns:
        A list of URLs to other report pages.
    """
    html = snippet
    soup = bs(html, 'html.parser')

    # Find all anchor tags within the current unordered list
    links = soup.find_all('a', href=True)

    return [link['href'] for link in links]


def get_all_report_urls(url):
    """
    This function takes a URL to a page of reports and returns download URLs for all the reports that exist on the page.
    """
    response = requests.get(url)
    soup = bs(response.text, 'html.parser')
    links = soup.find_all('a', href=True)

    report_links = set()
    for link in links:
        href = link['href']
        if href.startswith('https://posoco.in/download'):
            report_links.add(href)
            
    return report_links

def get_pdf_name(url):
    """
    This function takes a URL and returns the second last component of the URL as the PDF name    
    """
    pattern = r"/([^/]+)/[^/]+/?$"
    
    # use re.search to find the match in the URL
    match = re.search(pattern, url)

    # if a match is found, return the captured group (the second last path component)
    if match:
        return match.group(1)
    else:
        return ""
    
def download_reports(urls, save_folder, retry_delay = 5):
    """
    This functions takes in all the yearly report urls and downloads the reports to a specified folder
    """
    for pdf_url in urls:
        # extract the second last path component to use as the file name
        pdf_name = get_pdf_name(pdf_url)

        # create the full local file path to save the PDF
        pdf_path = os.path.join(save_folder, f"{pdf_name}.pdf")

        retry_count = 0
        while retry_count < 3:
            try:
            # download the PDF
                response = requests.get(pdf_url)
            
                if response.status_code == 200:
                    with open(pdf_path, 'wb') as pdf:
                        pdf.write(response.content)
                    break
                else:
                    print(f"Failed to download: {pdf_url}, Status Code: {response.status_code}")
            
            except requests.exceptions.RequestException as e:
                print(f"Connection error: {e}")
                retry_count += 1
                if retry_count < 3:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Max retries reached for: {pdf_url}")

def main():
    
    html_snippet = """</ul>
    </li>
    </ul></div></nav><div class="site-inner"><div class="content-sidebar-wrap"><main class="content"><article class="post-463 page type-page status-publish entry" itemscope itemtype="https://schema.org/CreativeWork"><header class="entry-header"><h1 class="entry-title" itemprop="headline">Daily Reports</h1>
    </header><div class="entry-content" itemprop="text"><ul>
    <li><a href="https://posoco.in/reports/daily-reports/daily-reports-2022-23/">Daily Reports &#8211; 2022-23</a></li>
    <li><a href="https://posoco.in/reports/daily-reports/daily-reports-2021-22/">Daily Reports &#8211; 2021-22</a></li>
    <li><a href="https://posoco.in/reports/daily-reports/daily-reports-2020-21/">Daily Reports &#8211; 2020-21</a></li>
    <li><a href="https://posoco.in/reports/daily-reports/daily-reports-2019-20/">Daily Reports &#8211; 2019-20</a></li>
    <li><a href="https://posoco.in/reports/daily-reports/daily-reports-2018-19/">Daily Reports &#8211; 2018-19</a></li>
    <li><a href="https://posoco.in/reports/daily-reports/daily-reports-2017-18/">Daily Reports &#8211; 2017-18</a></li>
    <li><a href="https://posoco.in/reports/daily-reports/daily-reports-2016-17/">Daily Reports &#8211; 2016-17</a></li>
    <li><a href="https://posoco.in/reports/daily-reports/daily-reports-2015-16/">Daily Reports &#8211; 2015-16</a></li>
    <li><a href="https://posoco.in/reports/daily-reports/daily-reports-2014-15/">Daily Reports &#8211; 2014-15</a></li>
    <li><a href="https://posoco.in/reports/daily-reports/daily-reports-2013-14/">Daily Reports &#8211; 2013-14</a></li>
    </ul>"""
    
    
    save_folder = ['2022-23', '2021-22', '2020-21','2019-20', '2018-19', '2017-18', '2016-17', '2015-16', 
                   '2014-15', '2013-14']  # folder where PDFs will be saved

    #getting yearly report page urls
    yearly_urls = get_yearly_report_urls(html_snippet)

    #download all yearly reports
    for i, url in enumerate(yearly_urls):
        links = get_all_report_urls(url)
        file_path = os.path.join('/Users/ridhipurohit/Documents/GitHub/energy_project_CSEP/india_energy_data', save_folder[i])
        print(f"Downloading for folder {save_folder[i]} ")
        download_reports(links, file_path)
    
if __name__ == "__main__":
    main()