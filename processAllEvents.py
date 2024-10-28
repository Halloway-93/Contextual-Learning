import json
import os
from pathlib import Path
import pandas as pd
import re


def read_file(filepath):
    if filepath.suffix == ".csv":
        print(f"Read data from {filepath}")
        data = pd.read_csv(filepath)
        return data
    elif filepath.suffix == ".tsv":
        print(f"Read data from {filepath}")
        data = pd.read_csv(filepath, sep="\t")
        return data
    elif filepath.suffix == ".json":
        print(f"Read data from {filepath}")
        with open(filepath, "r") as f:
            metadata = json.load(f)
        return metadata
    return None


def process_metadata(df, metadata):
    sub = metadata.get("sub")
    df["sub"] = [sub] * len(df)
    return df


def process_all_events(data_dir, filename="allEvents.csv"):
    all_events = []
    data_dir_path = Path(data_dir)

    for filepath in sorted(data_dir_path.rglob("*")):
        if filepath.is_file():
            if filepath.suffix == ".csv" and filepath.stem.startswith("sub"):
                df = read_file(filepath)
                df["trial"] = [i + 1 for i in range(len(df))]
                all_events.append(df)
            elif filepath.suffix == ".tsv":
                df = read_file(filepath)
                df["sub"] = df["sub_number"]
                df["trial"] = df["trial_number"]
                proba = int(re.search(r"dir(\d+)", str(filepath)).group(1))
                df["proba"] = proba
                df.drop(columns=["sub_number", "trial_number"], inplace=True)
                all_events.append(df)
            elif filepath.suffix == ".json" and filepath.name != "slopes.json":
                metadata = read_file(filepath)
                if all_events:
                    all_events[-1] = process_metadata(all_events[-1], metadata)

    if all_events:
        big_df = pd.concat(all_events, axis=0, ignore_index=True)
        big_df.to_csv(os.path.join(data_dir, filename), index=False)
    else:
        print("No events found to process.")


# Example usage
# data_dir = "path/to/your/data"
# big_df = process_all_raw_data(data_dir)


# Running the code on the server
dirPath1 = "/envau/work/brainets/oueld.h/contextuaLearning/directionCue/results_voluntaryDirection"
process_all_events(dirPath1)
dirPath1 = "/envau/work/brainets/oueld.h/contextuaLearning/directionCue/results_imposeDirection"
process_all_events(dirPath1)
# mesocentre
# dirPath1 = "/scratch/houeld/contextuaLearning/directionCue/results_voluntaryDirection"
dirPath2 = "/envau/work/brainets/oueld.h/contextuaLearning/ColorCue/data"
process_all_events(dirPath2)
dirPath2 = "/envau/work/brainets/oueld.h/contextuaLearning/ColorCue/imposedColorData"
process_all_events(dirPath2)
# Mesocentre
# dirPath2 = "/scratch/houeld/contextuaLearning/ColorCue/data"
