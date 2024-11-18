import csv
import argparse
import pandas as pd

def csv_to_arff(csv_file, arff_file, width_px, height_px, width_mm, height_mm, distance_mm):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Rename the columns 'xp' and 'yp' to 'x' and 'y'
    df.rename(columns={'xp': 'x', 'yp': 'y'}, inplace=True)

    # Add a new column called 'confidence' with all ones
    df['confidence'] = 1

    # Set the 'confidence' to zero where there are NaN values in the 'x' or 'y' columns
    df.loc[df['x'].isna() | df['y'].isna(), 'confidence'] = 0

    # Save the modified DataFrame back to a CSV file
    modified_csv_file = csv_file.rsplit('.', 1)[0] + '_modified.csv'
    df.to_csv(modified_csv_file, index=False)

    # Now convert the modified CSV file to an ARFF file
    with open(modified_csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)

        with open(arff_file, 'w') as arfffile:
            arfffile.write('@relation data\n\n')

            # Write metadata
            arfffile.write(f'%@METADATA width_px {width_px}\n')
            arfffile.write(f'%@METADATA height_px {height_px}\n')
            arfffile.write(f'%@METADATA width_mm {width_mm}\n')
            arfffile.write(f'%@METADATA height_mm {height_mm}\n')
            arfffile.write(f'%@METADATA distance_mm {distance_mm}\n\n')

            # Write attributes
            for col in header:
                # Assuming all columns are numeric for simplicity
                arfffile.write(f'@attribute {col} numeric\n')

            arfffile.write('\n@data\n')

            # Write data
            for row in csvreader:
                arfffile.write(','.join(row) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Convert CSV to ARFF format.')
    parser.add_argument('csv_file', help='Path to the input CSV file')

    args = parser.parse_args()

    # Generate the ARFF file name by adding the .arff extension
    arff_file = args.csv_file.rsplit('.', 1)[0] + '.arff'

    # Hardcoded metadata values
    width_px = 1920
    height_px = 1080
    width_mm = 700.00
    height_mm = 396.00
    distance_mm = 570.00

    csv_to_arff(
        args.csv_file,
        arff_file,
        width_px,
        height_px,
        width_mm,
        height_mm,
        distance_mm
    )

if __name__ == '__main__':
    main()

