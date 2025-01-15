import sys 
import pandas as pd
import json
def json2csv(json_file,csv_file):
    # Load json into a DataFrame 
    with open(json_file) as f:
        json_data = json.load(f)
    df = pd.DataFrame(json_data.items(), columns=['name', 'value'])
    df.index = range(len(df))
    # Convert DataFrame to CSV and save to a file
    df.to_csv(csv_file, index=False)


