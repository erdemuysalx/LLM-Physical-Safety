import pandas as pd
import argparse
import os
from tqdm import tqdm
import json  # Use json library for pretty-printing

def convert_csv_to_json(csv_file, json_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Convert the DataFrame to JSON format with progress bar
    json_data = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Converting to JSON'):
        json_data.append(row.to_dict())
    
    # Convert list to JSON string with pretty-printing
    json_string = json.dumps(json_data, indent=4)  # Use indent for readability

    # Save the JSON data to a file
    with open(json_file, 'w') as f:
        f.write(json_string)

    print(f"CSV data has been converted to JSON and saved as {json_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert CSV to JSON.')
    parser.add_argument('-csv', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('-json', type=str, help='Path to the output JSON file (optional)')

    args = parser.parse_args()

    # Determine the JSON file name
    if args.json:
        json_file = args.json
    else:
        # Create a default JSON file name by replacing the CSV extension with .json
        json_file = os.path.splitext(args.csv)[0] + '.json'

    convert_csv_to_json(args.csv, json_file)

if __name__ == '__main__':
    main()
