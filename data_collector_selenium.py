from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import pytz
import time
import pandas as pd
from bs4 import BeautifulSoup
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import PATHS, ensure_directories

class KinoDataCollector:
    def __init__(self, user_login="Mihai-Edward", debug=True):
        self.debug = debug
        self.user_login = user_login
        self.greek_tz = pytz.timezone('Europe/Athens')
        self.utc_tz = pytz.timezone('UTC')
        self.update_timestamps()
        self.base_url = "https://lotostats.ro/toate-rezultatele-grecia-kino-20-80"
        
        # Updated path handling using config
        ensure_directories()  # Ensure all required directories exist
        self.driver_path = PATHS['DRIVER']
        self.csv_file = PATHS['HISTORICAL_DATA']
        
        # Add data validation tracking
        self.last_collection_time = None
        self.collection_status = {
            'success': False,
            'draws_collected': 0,
            'last_error': None,
            'last_successful_draw': None
        }

    def update_timestamps(self):
        current_utc = datetime.now(self.utc_tz)
        self.current_utc_str = current_utc.strftime('%Y-%m-%d %H:%M:%S')
        self.last_collection_time = current_utc

    def validate_draw_data(self, numbers):
        """Validate drawn numbers before saving"""
        try:
            if not numbers:
                if self.debug:
                    print("Empty numbers list detected")
                return False
                
            if not all(isinstance(n, int) for n in numbers):
                if self.debug:
                    print("Non-integer values detected in numbers")
                return False
                
            if not all(1 <= n <= 80 for n in numbers):
                if self.debug:
                    print("Numbers outside valid range (1-80) detected")
                return False
                
            if len(set(numbers)) != len(numbers):
                if self.debug:
                    print("Duplicate numbers detected")
                return False
                
            return True
            
        except Exception as e:
            if self.debug:
                print(f"Error during validation: {str(e)}")
            return False

    def extract_numbers(self, number_cells):
        numbers = []
        for cell in number_cells:
            style = cell.get_attribute('style')
            text = cell.text.strip()
            if not text:
                text = cell.get_attribute('innerHTML').strip()
            if 'background-color: var(--color_bonus_balls);' not in style and text:
                try:
                    num = int(text)
                    if 1 <= num <= 80:
                        numbers.append(num)
                except ValueError:
                    continue
        return numbers

    def save_draw(self, draw_date, numbers):
        """Save draw to CSV file in the correct format"""
        # Validate data before processing
        if not self.validate_draw_data(numbers):
            if self.debug:
                print(f"Invalid draw data detected for date {draw_date}")
            self.collection_status['last_error'] = "Invalid draw data"
            return False
        
        # Ensure exactly 20 numbers
        if len(numbers) > 20:
            numbers = sorted(numbers)[:20]
        elif len(numbers) < 20:
            print(f"Warning: Draw has less than 20 numbers: {len(numbers)}")
            self.collection_status['last_error'] = "Insufficient numbers"
            return False
        
        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
            
            # Create dictionary with individual number columns
            data = {'date': draw_date}
            for i, num in enumerate(sorted(numbers), 1):
                data[f'number{i}'] = num
            
            # Create DataFrame
            draw_df = pd.DataFrame([data])
            
            # Save to CSV
            if os.path.exists(self.csv_file) and os.path.getsize(self.csv_file) > 0:
                try:
                    existing_df = pd.read_csv(self.csv_file)
                    # Check for duplicates before concatenating
                    if not any(existing_df['date'] == draw_date):
                        combined_df = pd.concat([existing_df, draw_df], ignore_index=True)
                        combined_df.to_csv(self.csv_file, index=False)
                    else:
                        print(f"Draw {draw_date} already exists in history")
                except pd.errors.EmptyDataError:
                    draw_df.to_csv(self.csv_file, index=False)
            else:
                draw_df.to_csv(self.csv_file, index=False)
                print(f"Created new file {self.csv_file} with draw {draw_date}")
            
            # Update collection status
            self.collection_status['success'] = True
            self.collection_status['draws_collected'] += 1
            self.collection_status['last_successful_draw'] = draw_date
            return True
                
        except Exception as e:
            if self.debug:
                print(f"Error saving draw: {str(e)}")
            self.collection_status['last_error'] = str(e)
            return False

    def fetch_latest_draws(self, num_draws=2, delay=1, max_pages=5):
        """
        Fetch latest lottery draws by navigating through paginated results tables.
        
        Args:
            num_draws: Maximum number of draws to collect (default 100)
            delay: Time delay between processing each draw in seconds (default 1)
            max_pages: Maximum number of pages to navigate through (default 3)
        
        Returns:
            List of tuples containing (draw_date, numbers)
        """
        driver = None
        try:
            print(f"\nFetching up to {num_draws} latest draws (up to {max_pages} pages)...")
            self.collection_status['draws_collected'] = 0
            self.update_timestamps()

            # Setup Edge options with additional settings
            edge_options = Options()
            edge_options.add_argument('--headless')
            edge_options.add_argument('--disable-gpu')
            edge_options.add_argument('--no-sandbox')
            edge_options.add_argument('--ignore-certificate-errors')
            edge_options.add_argument('--disable-dev-shm-usage')
            edge_options.add_argument('--disable-blink-features=AutomationControlled')
            edge_options.add_argument('--disable-extensions')
            edge_options.add_experimental_option('excludeSwitches', ['enable-automation'])
            edge_options.add_experimental_option('useAutomationExtension', False)

            # Initialize Edge driver
            service = Service(executable_path=self.driver_path)
            driver = webdriver.Edge(service=service, options=edge_options)
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0'
            })
            print("Browser initialized")

            # Load the initial page
            driver.get(self.base_url)
            time.sleep(5)
            print("Initial page loaded")

            draws = []
            current_page = 1
            total_draws_collected = 0

            # Process pages until we have enough draws or reach max_pages
            while current_page <= max_pages and total_draws_collected < num_draws:
                print(f"Processing page {current_page}...")
                
                # Wait for the table with appropriate timeout 
                wait = WebDriverWait(driver, 30)
                table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#all_results")))
                print(f"Results table found on page {current_page}")

                # Wait for JavaScript to complete rendering
                time.sleep(5)

                # Find all draw rows in the results table
                rows = table.find_elements(By.CSS_SELECTOR, "tbody > tr")
                print(f"Found {len(rows)} rows in the results table on page {current_page}")

                # Process rows on current page
                page_draws_collected = 0
                for row in rows:
                    try:
                        # Get the date cell and parse it with BeautifulSoup
                        date_cell_html = row.find_element(By.CSS_SELECTOR, "td:first-child").get_attribute('innerHTML')
                        soup = BeautifulSoup(date_cell_html, 'html.parser')
                        draw_date = soup.get_text().strip()

                        # Get all number cells in the row
                        number_cells = row.find_elements(By.CSS_SELECTOR, "td:nth-child(2) > div.nrr")
                        numbers = self.extract_numbers(number_cells)

                        if len(numbers) >= 20:
                            draws.append((draw_date, sorted(numbers)))
                            # Save each draw as it's collected
                            self.save_draw(draw_date, numbers)
                            total_draws_collected += 1
                            page_draws_collected += 1
                            
                            # Break if we've collected enough draws
                            if total_draws_collected >= num_draws:
                                break

                        # Introduce a delay between processing each draw
                        time.sleep(delay)

                    except Exception as e:
                        print(f"Error processing row: {str(e)}")
                        self.collection_status['last_error'] = str(e)
                        continue
                
                print(f"Collected {page_draws_collected} draws from page {current_page}")
                
                # If we haven't collected enough draws, try to go to the next page
                if total_draws_collected < num_draws and current_page < max_pages:
                    try:
                        # Calculate the next page number
                        next_page_number = current_page + 1
                        
                        # First make sure we scroll down to make the pagination visible
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(2)
                        
                        # Direct targeting of the next page button
                        next_page_selector = f"a.fg-button[aria-controls='all_results'][data-dt-idx='{next_page_number}']"
                        print(f"Looking for pagination button with selector: {next_page_selector}")
                        
                        # Try to find the element
                        next_page_button = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, next_page_selector))
                        )
                        
                        # Log button status
                        if next_page_button.is_displayed():
                            print("Pagination button is visible")
                        else:
                            print("Pagination button is not visible - scrolling to make it visible")
                            # Try to scroll directly to the button
                            driver.execute_script("arguments[0].scrollIntoView(true);", next_page_button)
                            time.sleep(2)
                        
                        # Click using JavaScript which works even if the element isn't fully visible
                        print(f"Clicking on pagination button {next_page_number} using JavaScript")
                        driver.execute_script("arguments[0].click();", next_page_button)
                        
                        # Wait for page to update
                        time.sleep(5)
                        
                        # Verify that we're on the new page by checking the active pagination button
                        active_page = driver.find_element(By.CSS_SELECTOR, "a.fg-button.ui-state-disabled:not(.previous):not(.next)")
                        if active_page.text.strip() == str(next_page_number):
                            print(f"Successfully navigated to page {next_page_number}")
                            current_page = next_page_number
                        else:
                            print(f"Navigation to page {next_page_number} failed - URL navigation fallback")
                            # Fallback to URL navigation if the click didn't work
                            new_url = f"{self.base_url}?page={next_page_number}"
                            driver.get(new_url)
                            time.sleep(5)
                            current_page = next_page_number
                    
                    except Exception as e:
                        print(f"Error navigating to next page: {str(e)}")
                        # Try URL navigation as a last resort
                        try:
                            new_url = f"{self.base_url}?page={next_page_number}"
                            print(f"Attempting direct navigation to: {new_url}")
                            driver.get(new_url)
                            time.sleep(5)
                            current_page = next_page_number
                            print(f"Successfully navigated to page {next_page_number} via URL")
                        except:
                            print(f"Failed to navigate to page {next_page_number}")
                            break
                else:
                    break

            if draws:
                print(f"\nSuccessfully collected {len(draws)} draws across {current_page} pages")
                self.collection_status['success'] = True
                self.collection_status['draws_collected'] = total_draws_collected
                return draws
            else:
                print("\nNo draws found.")
                self.collection_status['success'] = False
                return []

        except Exception as e:
            print(f"Error: {str(e)}")
            self.collection_status['success'] = False
            self.collection_status['last_error'] = str(e)
            return []

        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

    def sort_historical_draws(self):
        """Sort historical draws from newest to oldest"""
        try:
            if not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0:
                if self.debug:
                    print("No historical draws file found or file is empty")
                return False
                
            # Read the existing CSV file
            df = pd.read_csv(self.csv_file)
            
            # Convert date strings to datetime for proper sorting
            df['date_temp'] = pd.to_datetime(
                df['date'].apply(lambda x: f"{x.split()[1]} {x.split()[0]}"), 
                format='%d-%m-%Y %H:%M'
            )
            
            # Sort by date descending (newest first)
            df = df.sort_values('date_temp', ascending=False)
            
            # Remove temporary column
            df = df.drop('date_temp', axis=1)
            
            # Save back to CSV
            df.to_csv(self.csv_file, index=False)
            
            if self.debug:
                print(f"Successfully sorted {len(df)} draws from newest to oldest")
            return True
            
        except Exception as e:
            if self.debug:
                print(f"Error sorting draws: {str(e)}")
            return False

if __name__ == "__main__":
    collector = KinoDataCollector()
    
    # First try to sort existing data
    print("\nSorting historical draws...")
    collector.sort_historical_draws()
    
    # Then fetch new draws
    draws = collector.fetch_latest_draws()
    if draws:
        print("\nCollected draws:")
        for draw_date, numbers in draws:
            print(f"Date: {draw_date}, Numbers: {', '.join(map(str, numbers))}")
        
        # Sort again after collecting new draws
        print("\nSorting updated historical draws...")
        if collector.sort_historical_draws():
            print("Historical draws successfully sorted from newest to oldest")
        else:
            print("Error occurred while sorting draws")
            
    print("\nCollection Status:", collector.collection_status)