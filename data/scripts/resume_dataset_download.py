import os
import time
import random
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent

# Rotating User-Agent
ua = UserAgent()

# Set up WebDriver with random user-agent
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run headless (no GUI)
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument(f'user-agent={ua.random}')

driver = webdriver.Chrome(options=chrome_options)

# Fields and roles for CV scraping
fields_and_roles = {
    "information_technology": [
        "software engineer", "data scientist", "network administrator", "it support specialist",
        "cybersecurity analyst", "devops engineer", "cloud solutions architect", "database administrator",
        "ux designer", "it project manager"
    ],
    "healthcare_and_medicine": [
        "registered nurse", "physician assistant", "medical laboratory technician", "pharmacist",
        "physical therapist", "radiologic technologist", "surgeon", "medical coder", "health informatics specialist",
        "occupational therapist"
    ],
    "education_and_training": [
        "elementary school teacher", "high school teacher", "university professor", "instructional designer",
        "corporate trainer", "special education teacher", "teaching assistant", "educational consultant",
        "language instructor", "curriculum developer"
    ],
    "finance_and_accounting": [
        "financial analyst", "accountant", "tax consultant", "investment banker", "auditor",
        "payroll specialist", "credit analyst", "risk manager", "actuary", "treasury analyst"
    ],
    "engineering": [
        "mechanical engineer", "electrical engineer", "civil engineer", "chemical engineer",
        "structural engineer", "environmental engineer", "industrial engineer", "aerospace engineer",
        "petroleum engineer", "biomedical engineer"
    ],
    "marketing_and_sales": [
        "marketing manager", "digital marketing specialist", "sales executive", "content strategist",
        "brand manager", "seo specialist", "social media manager", "account manager",
        "market research analyst", "product manager"
    ],
    "human_resources": [
        "hr manager", "talent acquisition specialist", "compensation and benefits manager", 
        "training and development specialist", "hr generalist", "employee relations specialist",
        "recruitment coordinator", "hr analyst", "diversity and inclusion manager", "hr consultant"
    ],
    "customer_service": [
        "customer service representative", "call center agent", "customer success manager", 
        "help desk specialist", "technical support specialist", "client services manager", 
        "customer experience manager", "account support specialist", "concierge", "online chat support agent"
    ],
    "legal_and_compliance": [
        "corporate lawyer", "paralegal", "compliance officer", "legal assistant", 
        "contract manager", "intellectual property lawyer", "litigation attorney", 
        "legal consultant", "compliance analyst", "immigration lawyer"
    ],
    "logistics_and_supply_chain": [
        "supply chain manager", "logistics coordinator", "procurement specialist", 
        "warehouse manager", "distribution manager", "inventory analyst", "transportation planner", 
        "fleet manager", "supply chain analyst", "freight forwarder"
    ]
}

# Google search query template
query_template = '(intitle:resume OR inurl:resume) -job -jobs -sample -samples -example -examples "{}" filetype:pdf'

# Directory to save resumes
base_dir = "resumes"

# Function to create directory if it doesn't exist
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to download PDFs
def download_pdf(link, folder, role_name, i):
    try:
        # Send a GET request
        response = requests.get(link, timeout=10)
        if response.status_code == 200:
            with open(f'{folder}/{role_name}_resume_{i+1}.pdf', 'wb') as file:
                file.write(response.content)
            print(f'Successfully downloaded: {folder}/{role_name}_resume_{i+1}.pdf from {link}')
            return True
        else:
            print(f'Failed to download {link}: HTTP Status {response.status_code}')
    except Exception as e:
        print(f"Failed to download {link}: {e}")

# Function to get resumes for a role
def get_resumes_for_role(field, role):
    search_query = query_template.format(role)
    print(f"Search query: {search_query}")
    
    # Create directory structure based on field and role
    folder_name = field.replace(" ", "_").lower()
    folder_path = os.path.join(base_dir, folder_name)
    create_dir_if_not_exists(folder_path)
    
    role_name = role.replace(" ", "_").lower()

    # Check if there are already resumes in the directory
    existing_cvs = [f for f in os.listdir(folder_path) if f.startswith(f'{role_name}_resume_')]
    if len(existing_cvs) >= 10:
        print(f"Skipping {role} for {field}: already have {len(existing_cvs)} CVs.")
        return

    pdf_links = []
    
    try:
        # Open Google and search
        driver.get("https://www.google.com")
        search_box = driver.find_element(By.NAME, "q")

        print(f"Attempting search with query: {search_query}")
        search_box.send_keys(search_query)

        time.sleep(random.uniform(2, 5))
        search_box.send_keys(Keys.RETURN)

        # Function to extract links from a page
        def extract_links():
            results = driver.find_elements(By.XPATH, '//a[@href]')
            for result in results:
                href = result.get_attribute('href')
                if href.endswith('.pdf') and href not in pdf_links:
                    pdf_links.append(href)
                    if len(pdf_links) >= 30:  # Gather up to 30 links
                        break
            print(f"Found {len(pdf_links)} PDF links on this page.")

        # Extract links from the first page
        extract_links()

        # Loop through multiple pages until 30 resumes or no next page
        page = 2
        while len(pdf_links) < 30:
            try:
                next_button = driver.find_element(By.ID, "pnnext")
                if next_button.is_displayed():
                    next_button.click()
                    time.sleep(random.uniform(3, 6))
                    extract_links()
                    page += 1
                else:
                    break  # Exit if no next page
            except Exception as e:
                print(f"No more pages or error on page {page}: {e}")
                break

    except Exception as e:
        print(f"Error during Google search or extraction: {e}")
        return

    # Print a warning if fewer than 30 resumes found
    if len(pdf_links) < 30:
        print(f"Warning: Only {len(pdf_links)} resumes found for {role}. Proceeding with what is available.")

    # Download PDFs with retry for failed downloads
    successful_downloads = 0
    failed_links = []
    
    while successful_downloads < len(pdf_links):
        # Attempt to download PDFs
        for i, link in enumerate(pdf_links):
            if download_pdf(link, folder_path, role_name, successful_downloads):
                successful_downloads += 1
            else:
                failed_links.append(link)

    print(f"Completed downloads: {successful_downloads} resumes for {role}.")
    
# Ensure WebDriver is closed properly
try:
    # Iterate through all fields and roles and get resumes
    for field, roles in fields_and_roles.items():
        for role in roles:
            get_resumes_for_role(field, role)
finally:
    # Close WebDriver
    driver.quit()
    print("Script finished and WebDriver closed.")
