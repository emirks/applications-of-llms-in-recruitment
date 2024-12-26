import os
from resume_statement_extractor.generators.gpt_4o import GPT4oGenerator
from typing import List, Dict, Tuple
import subprocess
import logging
import shutil

logger = logging.getLogger(__name__)

class CommitGenerator:
    def __init__(self):
        self.gpt4o = GPT4oGenerator()
        # Find git executable
        self.git_exe = self._find_git_executable()
        if not self.git_exe:
            raise RuntimeError("Git executable not found. Please ensure Git is installed and accessible.")
        
        # Add ignore patterns
        self.ignore_patterns = [
            'data/analysis_results/',  # Ignore analysis results directory
            '.zip',                    # Ignore zip files
            '.pyc',                    # Ignore compiled Python files
            '__pycache__/',           # Ignore Python cache directories
            '.git/',                   # Ignore git directory
            '.index',
            '.metadata',
        ]
    
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
    
    def _should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored"""
        for pattern in self.ignore_patterns:
            if pattern in file_path:
                return True
        return False
    
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
            
            # Parse the status output and filter ignored files
            modified_files = []
            for line in result.stdout.splitlines():
                status = line[:2]
                file_path = line[3:].strip()
                
                # Skip ignored files
                if self._should_ignore(file_path):
                    logger.info(f"Ignoring file: {file_path}")
                    continue
                
                # M: modified
                # A: added
                # R: renamed
                # ??: untracked
                if status in ['M ', ' M', 'A ', 'R ', '??']:
                    # For renamed files, get the new name
                    if status == 'R ':
                        file_path = file_path.split(' -> ')[1]
                    modified_files.append(file_path)
            
            logger.info(f"Found {len(modified_files)} modified files (after filtering)")
            return modified_files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting modified files: {e}")
            return []
        
    def get_file_diff(self, file_path: str) -> str:
        """Get git diff for a specific file"""
        try:
            # For untracked files, show the entire content
            if not self._is_tracked(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f"New file: {file_path}\n\n" + f.read()
            
            result = subprocess.run(
                [self.git_exe, 'diff', file_path],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
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

    def generate_commit_message(self, diff: str) -> str:
        """Generate commit message using GPT-4-mini"""
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
        
        for file_path in modified_files:
            # Get diff for the file
            diff = self.get_file_diff(file_path)
            if not diff:
                logger.warning(f"No changes detected for {file_path}")
                continue
            
            # Generate commit message
            commit_message = self.generate_commit_message(diff)
            proposals.append((file_path, commit_message))
            
            # Display proposal
            print(f"\nFile: {file_path}")
            print(f"Commit message: {commit_message}")
            print("-" * 50)
        
        return proposals

    def create_atomic_commits(self, proposals: List[Tuple[str, str]]):
        """Create atomic commits for each modified file"""
        for file_path, commit_message in proposals:
            try:
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