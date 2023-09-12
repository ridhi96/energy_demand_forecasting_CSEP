import PyPDF2
import os
import csv
import re

def read_report(pdf_path):

    try: 
        pdf_file_obj = open(pdf_path, 'rb')    
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                
        # creating a page object and extracting text
        first_page_obj = pdf_reader.pages[0].extract_text()

        # check if current page has relevant text or not    
        if "National Load Despatch Centre" in first_page_obj:
            page_num = 1
        else:
            page_num = 0

        extracted_text = pdf_reader.pages[page_num].extract_text()

        lines = extracted_text.split('\n')

        # process each line and extract data
        date = lines[0].strip()
           
        peak_demand_values = lines[2].split()
        if len(peak_demand_values) == 6:
            peak_demand = {
                        "NR": peak_demand_values[0],
                        "WR": peak_demand_values[1],
                        "SR": peak_demand_values[2],
                        "ER": peak_demand_values[3],
                        "NER": peak_demand_values[4],
                        "Total": peak_demand_values[5],
                    }
                
        peak_shortage_values = lines[3].split()
        if len(peak_shortage_values) == 6:
            peak_shortage = {
                        "NR": peak_shortage_values[0],
                        "WR": peak_shortage_values[1],
                        "SR": peak_shortage_values[2],
                        "ER": peak_shortage_values[3],
                        "NER": peak_shortage_values[4],
                        "Total": peak_shortage_values[5],
                    }
        
        result_dict = {"date": date, "peak_demand": peak_demand, "peak_shortage": peak_shortage}

        pdf_file_obj.close()

        return result_dict
    
    except Exception:
        print(f"Error reading PDF: {pdf_path}")
        return None

def alternate_read_report(pdf_path):
    try: 
        pdf_file_obj = open(pdf_path, 'rb')    
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)

        # creating a page object and extracting text
        first_page_obj = pdf_reader.pages[0].extract_text()

        date_pattern = r'\d{1,2}(?:st|nd|rd|th)\s+[A-Za-z]+(?:\s+\d{4})?'
        date = re.findall(date_pattern, first_page_obj)[0]

        extracted_text = pdf_reader.pages[1].extract_text()
        lines = extracted_text.split('\n')

        # process each line and extract data
        peak_demand_values = lines[1].split()
        if len(peak_demand_values) == 6:
            peak_demand = {
                        "NR": peak_demand_values[0],
                        "WR": peak_demand_values[1],
                        "SR": peak_demand_values[2],
                        "ER": peak_demand_values[3],
                        "NER": peak_demand_values[4],
                        "Total": peak_demand_values[5],
                    }
        
        peak_shortage_values = lines[2].split()
        if len(peak_shortage_values) == 6:
            peak_shortage = {
                        "NR": peak_shortage_values[0],
                        "WR": peak_shortage_values[1],
                        "SR": peak_shortage_values[2],
                        "ER": peak_shortage_values[3],
                        "NER": peak_shortage_values[4],
                        "Total": peak_shortage_values[5],
                    }

        result_dict = {"date": date, "peak_demand": peak_demand, "peak_shortage": peak_shortage}

        pdf_file_obj.close()

        return result_dict
    
    except Exception:
        print(f"Error reading PDF: {pdf_path}")
        return None


def convert_to_csv(data_list):
    headers = ['date', 'NR_peak_demand', 'WR_peak_demand', 'SR_peak_demand', 'ER_peak_demand', 'NER_peak_demand',
               'Total_peak_demand', 'NR_peak_shortage', 'WR_peak_shortage', 'SR_peak_shortage', 'ER_peak_shortage',
               'NER_peak_shortage', 'Total_peak_shortage']
    
    with open('test.csv', 'w', newline='') as csv_file:
        # create csv writer
        writer_obj = csv.DictWriter(csv_file, fieldnames=headers)

        # create header row
        writer_obj.writeheader()

        # extract data and write into csv file
        for data in data_list:
            row = {
                'date': data['date'],
                'NR_peak_demand': data['peak_demand'].get('NR', 0),
                'WR_peak_demand': data['peak_demand'].get('WR', 0),
                'SR_peak_demand': data['peak_demand'].get('SR', 0),
                'ER_peak_demand': data['peak_demand'].get('ER', 0),
                'NER_peak_demand': data['peak_demand'].get('NER', 0),
                'Total_peak_demand': data['peak_demand'].get('Total', 0),
                'NR_peak_shortage': data['peak_shortage'].get('NR', 0),
                'WR_peak_shortage': data['peak_shortage'].get('WR', 0),
                'SR_peak_shortage': data['peak_shortage'].get('SR', 0),
                'ER_peak_shortage': data['peak_shortage'].get('ER', 0),
                'NER_peak_shortage': data['peak_shortage'].get('NER', 0),
                'Total_peak_shortage': data['peak_shortage'].get('Total', 0)
            }
            
            # write the row to the CSV file
            writer_obj.writerow(row)


def main():

    data_list =[]

    folder_path = '/Users/ridhipurohit/Documents/GitHub/energy_project_CSEP/india_energy_data/2022-23'
    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)
        data = alternate_read_report(file_path)
        if data is not None:
            data_list.append(data)
        
    convert_to_csv(data_list)

if __name__ == "__main__":
    main()
