import csv
import os

DIR = "info/"
CSV_FILE = "visualization_correspondences.csv"

def save_policy_info(Q_TABLE_FILE, data):
    file_path = DIR + CSV_FILE

    # Check if file already exists
    file_exists = os.path.isfile(file_path)

    # Track existing filenames to prevent duplicates
    existing_filenames = set()

    if file_exists:
        # Read existing filenames
        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_filenames.add(row["filename"])

    # If this filename already exists, skip writing
    if Q_TABLE_FILE in existing_filenames:
        print(f"Skipping duplicate entry for {Q_TABLE_FILE}.")
        return

    # Append mode
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header once if file didn't exist
        if not file_exists:
            writer.writerow(["filename", "type", "x", "y", "r"])

        # Append row (it's guaranteed to not be a duplicate here)
        writer.writerow([Q_TABLE_FILE, data["type"], data["x"], data["y"], data["r"]])

    print(f"Added new policy info for {Q_TABLE_FILE}.")
        
def get_data_by_filename(filename_to_find):
    with open((DIR + CSV_FILE), "r", newline="") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if row["filename"] == filename_to_find:
                return row
    return None

def main():
    save_policy_info("test_qtable", "y=0,x=0")
    
if __name__ == "__main__":
    main()