import os
import re

def rename_resumes(base_dir):
    for root, dirs, files in os.walk(base_dir):
        role_files = {}
        
        # Group files by role
        for filename in files:
            if filename.endswith('.pdf'):
                match = re.match(r'(.+)_resume_\d+\.pdf', filename)
                if match:
                    role = match.group(1)
                    if role not in role_files:
                        role_files[role] = []
                    role_files[role].append(filename)
        
        # Rename files for each role
        for role, filenames in role_files.items():
            filenames.sort(key=lambda x: int(re.search(r'_resume_(\d+)\.pdf', x).group(1)))
            for i, old_name in enumerate(filenames, start=1):
                new_name = f"{role}_resume_{i}.pdf"
                old_path = os.path.join(root, old_name)
                new_path = os.path.join(root, new_name)
                
                # Rename only if the new name is different
                if old_name != new_name:
                    # If the new filename already exists, use a temporary name first
                    if os.path.exists(new_path):
                        temp_name = f"{role}_resume_temp_{i}.pdf"
                        temp_path = os.path.join(root, temp_name)
                        os.rename(old_path, temp_path)
                        old_path = temp_path
                    
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_name} -> {new_name}")

if __name__ == "__main__":
    base_dir = "resumes"
    rename_resumes(base_dir)
    print("Resume renaming completed.")
