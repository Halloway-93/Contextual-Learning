import pandas as pd
from scipy.io import arff

def arff_to_csv(arff_path, csv_path):
    # Load ARFF file
    data, meta = arff.loadarff(arff_path)

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # arff_path = "output.arff"  # Replace with your ARFF file path
    # csv_path = "output.csv"   # Replace with your desired CSV file path

    arff_path = "/home/oueld.h/dataRaphaelle/final.arff"  # Replace with your ARFF file path
    csv_path = "/home/oueld.h/dataRaphaelle/final.csv"   # Replace with your desired CSV file path
    arff_to_csv(arff_path, csv_path)
    print(f"Converted {arff_path} to {csv_path}")

