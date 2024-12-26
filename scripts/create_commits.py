import os
from resume_statement_extractor.generators.gpt_4o import GPT4oGenerator
from typing import List, Dict, Tuple
import subprocess
import logging
import shutil
import time
from tenacity import retry, wait_exponential, stop_after_attempt

logger = logging.getLogger(__name__)

class CommitGenerator:
    def __init__(self):
        self.gpt4o = GPT4oGenerator()
        # Find git executable
        self.git_exe = self._find_git_executable()
        if not self.git_exe:
            raise RuntimeError("Git executable not found. Please ensure Git is installed and accessible.")
    
    def _find_git_executable(self) -> str:
        """Find the Git executable path"""
        # First try to get from PATH
        git_cmd = shutil.which('git')
        if git_cmd:
            return git_cmd
            
        # Common Git installation locations on Windows
        common_locations = [
            r"C:\Program Files\Git\bin\git.exe",
            r"C:\Program Files (x86)\Git\bin\git.exe",
            os.path.expanduser("~\\AppData\\Local\\Programs\\Git\\bin\\git.exe"),
        ]
        
        for location in common_locations:
            if os.path.exists(location):
                return location
                
        return None
    
    def get_modified_files(self) -> List[str]:
        """Get list of modified files in the repository"""
        try:
            # Get both staged and unstaged modified files
            result = subprocess.run(
                [self.git_exe, 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the status output
            modified_files = []
            for line in result.stdout.splitlines():
                status = line[:2]
                file_path = line[3:].strip()
                
                # Skip directories themselves but include their contents
                if os.path.isdir(file_path):
                    # For untracked directories, walk through them to find all files
                    if status == '??':
                        # Check if this is a new directory with multiple new files
                        # If so, we'll commit the whole directory at once
                        modified_files.append(file_path)
                    continue
                
                # M: modified
                # A: added
                # R: renamed
                # ??: untracked
                if status in ['M ', ' M', 'A ', 'R ', '??']:
                    # For renamed files, get the new name
                    if status == 'R ':
                        file_path = file_path.split(' -> ')[1]
                    # Don't add files that are inside an untracked directory
                    if not any(file_path.startswith(d) for d in modified_files if os.path.isdir(d)):
                        modified_files.append(file_path)
            
            logger.info(f"Found {len(modified_files)} modified files/directories")
            return modified_files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting modified files: {e}")
            return []
        
    def get_file_diff(self, file_path: str) -> str:
        """Get git diff for a specific file"""
        try:
            # If it's a directory, list its contents
            if os.path.isdir(file_path):
                files_list = []
                for root, _, files in os.walk(file_path):
                    for file in files:
                        if not self._is_binary_file(os.path.join(root, file)):
                            files_list.append(os.path.relpath(os.path.join(root, file), file_path))
                return f"New directory: {file_path}\n\nContaining files:\n" + "\n".join(files_list)
            
            # Skip if file is binary
            if self._is_binary_file(file_path):
                logger.info(f"Skipping binary file: {file_path}")
                return ""
            
            # For untracked files, show the entire content
            if not self._is_tracked(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f"New file: {file_path}\n\n" + f.read()
                except UnicodeDecodeError:
                    logger.info(f"Skipping non-text file: {file_path}")
                    return ""
            
            result = subprocess.run(
                [self.git_exe, 'diff', file_path],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except (subprocess.CalledProcessError, UnicodeDecodeError, PermissionError) as e:
            logger.error(f"Error getting diff for {file_path}: {e}")
            return ""
    
    def _is_tracked(self, file_path: str) -> bool:
        """Check if file is tracked by git"""
        try:
            subprocess.run(
                [self.git_exe, 'ls-files', '--error-unmatch', file_path],
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _is_binary_file(self, file_path: str) -> bool:
        """Check if a file is binary"""
        try:
            # Skip certain file extensions known to be binary
            binary_extensions = {'.zip', '.index', '.metadata', '.pyc', '.pyo', '.pyd', 
                               '.so', '.dll', '.exe', '.bin', '.dat', '.db', '.sqlite'}
            if os.path.splitext(file_path)[1].lower() in binary_extensions:
                return True
            
            # Check content for binary data
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk
        except Exception as e:
            logger.error(f"Error checking if file is binary: {e}")
            return True  # Assume binary if we can't read the file

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def generate_commit_message(self, diff: str) -> str:
        """Generate commit message using GPT-4-mini with retry logic"""
        prompt = """
        Given the following git diff, write a concise and descriptive commit message.
        Follow these rules:
        1. Start with a label [feat] or [fix] or [docs] or [refactor] or [style] or [test] or [chore]
        2. Keep it under 72 characters
        3. Focus on the "what" and "why", not the "how"
        4. Be specific but concise
        
        Git diff:
        {diff}
        
        Commit message:
        """.format(diff=diff)
        
        try:
            message = self.gpt4o.generate_with_prompt(prompt, "")
            # Add delay between API calls
            time.sleep(1)  # Wait 1 second between calls
            return message.strip()
        except Exception as e:
            logger.error(f"Error generating commit message: {e}")
            return "Update code with recent changes"

    def propose_commits(self) -> List[Tuple[str, str]]:
        """Generate and propose commit messages for all modified files"""
        modified_files = self.get_modified_files()
        
        if not modified_files:
            logger.info("No modified files found")
            return []
        
        proposals = []
        print("\nProposed commits:")
        print("=" * 50)
        
        for i, file_path in enumerate(modified_files):
            # Get diff for the file
            diff = self.get_file_diff(file_path)
            if not diff:
                logger.warning(f"No changes detected for {file_path}")
                continue
            
            # Generate commit message with progress indicator
            print(f"\nProcessing file {i+1}/{len(modified_files)}: {file_path}")
            commit_message = self.generate_commit_message(diff)
            proposals.append((file_path, commit_message))
            
            # Display proposal
            print(f"Commit message: {commit_message}")
            print("-" * 50)
            
            # Add delay between files
            if i < len(modified_files) - 1:  # Don't wait after the last file
                time.sleep(2)  # Wait 2 seconds between files
        
        return proposals

    def create_atomic_commits(self, proposals: List[Tuple[str, str]]):
        """Create atomic commits for each modified file/directory"""
        for file_path, commit_message in proposals:
            try:
                # If it's a directory, add all its contents
                if os.path.isdir(file_path):
                    for root, _, files in os.walk(file_path):
                        for file in files:
                            full_path = os.path.join(root, file)
                            if not self._is_binary_file(full_path):
                                subprocess.run([self.git_exe, 'add', full_path], check=True)
                else:
                    # Stage the file
                    subprocess.run([self.git_exe, 'add', file_path], check=True)
                
                # Create commit
                subprocess.run(
                    [self.git_exe, 'commit', '-m', commit_message],
                    check=True
                )
                
                logger.info(f"Created commit for {file_path}: {commit_message}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Error creating commit for {file_path}: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create generator and get proposals
    commit_gen = CommitGenerator()
    proposals = commit_gen.propose_commits()
    
    if not proposals:
        print("No changes to commit")
        exit(0)
    
    # Ask for confirmation
    response = input("\nWould you like to proceed with these commits? (y/n): ").lower().strip()
    
    if response == 'y':
        print("\nCreating commits...")
        commit_gen.create_atomic_commits(proposals)
        print("Done!")
    else:
        print("Aborted. No commits were created.") 