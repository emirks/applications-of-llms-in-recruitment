import os
import hashlib
from collections import defaultdict
from resume_name_handler import rename_resumes

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

def remove_duplicate_resumes(base_dir):
    hash_dict = defaultdict(list)
    duplicates_removed = 0

    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith('.pdf'):
                filepath = os.path.join(root, filename)
                file_hash = get_file_hash(filepath)
                hash_dict[file_hash].append(filepath)

    for file_hash, file_list in hash_dict.items():
        if len(file_list) > 1:
            # Keep the first file, remove the rest
            for duplicate_file in file_list[1:]:
                os.remove(duplicate_file)
                duplicates_removed += 1
                print(f"Removed duplicate: {duplicate_file}")

    return duplicates_removed

def process_resumes(base_dir):
    print("Removing duplicate resumes...")
    duplicates_removed = remove_duplicate_resumes(base_dir)
    print(f"Total duplicates removed: {duplicates_removed}")

    print("\nRenaming remaining resumes...")
    rename_resumes(base_dir)
    print("Resume renaming completed.")

if __name__ == "__main__":
    base_dir = "resumes"
    process_resumes(base_dir)
    print("Resume processing completed.")
