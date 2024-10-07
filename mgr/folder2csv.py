import os
import csv

def list_files_in_folder(folder_path, output_csv):
    # Get list of all files in the folder
    file_list = os.listdir(folder_path)
    
    # Filter out directories, only keep files
    file_list = [f for f in file_list if os.path.isfile(os.path.join(folder_path, f))]
    
    # Write the file names (with full path) to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File Path"])  # Writing header
        for file_name in file_list:
            full_path = os.path.join(folder_path, file_name)  # Full path to file
            writer.writerow([full_path])
    
    print(f"File names written to {output_csv}")

# Example usage
folder_path = '../data/Mathiesen-single-pages'  # Replace with your folder path
output_csv = 'file_paths.csv'  # Name of the output CSV file
list_files_in_folder(folder_path, output_csv)
