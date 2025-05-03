from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def setup_driver(headless=False):
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--log-level=3")  # suppress unnecessary logs
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', { get: () => false });")
    return driver

# Example function to visit a website and interact with elements
def scrape_website():
    driver = setup_driver(headless=False)  # Set headless=True if you don't need UI
    driver.get("https://gmgn.ai/meme?chain=sol&tab=new_creation")  # Change to your target website

    try:
        # Wait for an element to load
        wait = WebDriverWait(driver, 3)
        example_element = driver.find_elements(By.CSS_SELECTOR, ".css-9enbzl")
        examples = [example.text for example in example_element]
        print("Page Title:", driver.title)
        print("Found Element Text:", examples)

        time.sleep(5) #see if website has anti scrape

    except Exception as e:
        print("Error:", e)

    finally:
        driver.quit()  # Close the browser

# Run the script
if __name__ == "__main__":
    scrape_website()