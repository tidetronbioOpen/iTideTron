import sys 
import pandas as pd


# Load CSV into a DataFrame
csv_file = sys.argv[1]  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Convert DataFrame to JSON and save to a file
json_file = sys.argv[2]  # Replace with the desired output file path
df.to_json(json_file, orient='records', lines=True)
