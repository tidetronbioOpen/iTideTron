import os, csv, sys
import pandas as pd

current_path = os.path.dirname(os.path.abspath(__file__))

# Load CSV into a DataFrame
input_file = sys.argv[1] if len(sys.argv) > 1 else "../probiotics/raw/test.csv"
output_file = sys.argv[2] if len(sys.argv) > 2 else "../probiotics/raw/probiotic_predict.csv" 
input_file = os.path.join(current_path, input_file)
output_file = os.path.join(current_path, output_file)
#assert os.path.isfile(input_file), f"{input_file} dose not exists"
#assert os.path.isfile(output_file), f"{input_file} dose not exists"

dat_i = pd.read_csv(input_file, delimiter="\t")
with open(output_file, 'w', newline='') as out_file:
    writer = csv.writer(out_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(["name,feature"])
    for line in dat_i.iloc:
        line["sequence"] = line["sequence"].replace('\n','')
        writer.writerow([line["name"]+','+line["sequence"]])
