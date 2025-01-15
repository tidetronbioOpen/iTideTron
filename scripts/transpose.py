import sys 
import pandas as pd
# Usage: 
#   python data/scripts/transpose.py "data/probiotics/raw/labels651.csv"

# Load CSV into a DataFrame
csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/probiotics/raw/labels114.csv" # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Convert DataFrame to JSON and save to a file
transposed_file = sys.argv[2] if len(sys.argv) > 2 else "data/probiotics/result/gold_labels.csv" # Replace with the desired output file path
df_transposed = df.T
df_transposed.columns = ["labels"]
df_transposed.to_csv(transposed_file, index=True)

df_t = pd.read_csv(transposed_file) 
