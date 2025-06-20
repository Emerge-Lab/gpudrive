if __name__ == "__main__":
    
    import os
    import shutil
    from pathlib import Path

    # Define the source folder paths
    validation_folder = "data/processed/wosac/validation"
    validation_extracted_folder = "data/processed/wosac/validation_extracted"

    # Define the destination folder paths
    validation_json_folder = "validation_json"
    validation_tfrecord_folder = "validation_tfrecord"

    # Create destination folders if they don't exist
    os.makedirs(validation_json_folder, exist_ok=True)
    os.makedirs(validation_tfrecord_folder, exist_ok=True)

    # Check if source folders exist
    if not os.path.exists(validation_folder):
        print(f"Error: Folder '{validation_folder}' does not exist.")
        exit(1)

    if not os.path.exists(validation_extracted_folder):
        print(f"Error: Folder '{validation_extracted_folder}' does not exist.")
        exit(1)

    # Get all files and extract base names (without extension)
    validation_files = {}
    validation_extracted_files = {}

    # Process validation folder (json files)
    for filename in os.listdir(validation_folder):
        if filename.endswith('.json'):
            base_name = Path(filename).stem  # Gets filename without extension
            validation_files[base_name] = filename

    # Process validation_extracted folder (tfrecords files)
    for filename in os.listdir(validation_extracted_folder):
        if filename.endswith('.tfrecords'):
            base_name = Path(filename).stem  # Gets filename without extension
            validation_extracted_files[base_name] = filename

    # Find matches
    matches = set(validation_files.keys()) & set(validation_extracted_files.keys())

    print(f"Total matches found: {len(matches)}")

    # Check if we have at least 500 matches
    if len(matches) < 500:
        print(f"Warning: Only {len(matches)} matches found, less than requested 500")
        files_to_copy = sorted(matches)
    else:
        # Take first 500 matches (sorted alphabetically for consistency)
        files_to_copy = sorted(matches)[:500]
        print(f"Copying first 500 matches out of {len(matches)}")

    # Copy matching files
    copied_count = 0
    for base_name in files_to_copy:
        try:
            # Copy json file
            json_src = os.path.join(validation_folder, validation_files[base_name])
            json_dst = os.path.join(validation_json_folder, validation_files[base_name])
            shutil.copy2(json_src, json_dst)
            
            # Copy tfrecords file
            tf_src = os.path.join(validation_extracted_folder, validation_extracted_files[base_name])
            tf_dst = os.path.join(validation_tfrecord_folder, validation_extracted_files[base_name])
            shutil.copy2(tf_src, tf_dst)
            
            copied_count += 1
            if copied_count % 50 == 0:  # Progress update every 50 files
                print(f"Copied {copied_count} files...")
                
        except Exception as e:
            print(f"Error copying files for {base_name}: {e}")

    print(f"\nSuccessfully copied {copied_count} matching files:")
    print(f"JSON files in: {validation_json_folder}")
    print(f"TFRecord files in: {validation_tfrecord_folder}")

    # Write a summary file
    with open("copied_files_summary.txt", 'w') as f:
        f.write(f"Copied Files Summary\n")
        f.write(f"===================\n\n")
        f.write(f"Total matches found: {len(matches)}\n")
        f.write(f"Files copied: {copied_count}\n\n")
        f.write("Copied files:\n")
        for base_name in files_to_copy:
            f.write(f"  {validation_files[base_name]} <-> {validation_extracted_files[base_name]}\n")

    print(f"Summary written to: copied_files_summary.txt")