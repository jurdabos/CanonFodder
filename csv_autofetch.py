from datetime import datetime, timedelta
import logging
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager
# I AM COMMENTING OUT THE FOLLOWING LINE TO TEST IF SSL HANDSHAKES WORK WITHOUT SUPPRESSING VERIFICATION
# os.environ["WDM_SSL_VERIFY"] = "0"
USERNAME = "jurda"
output_dir = "C:\\Users\\jurda\\PycharmProjects\\MyLifeInData\\CSV"
logging.basicConfig(
    filename=os.path.join(output_dir, "fetcher.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
current_date = datetime.now().strftime("%Y%m%d")
csv_filename = f"{USERNAME}_{current_date}.csv"
csv_filepath = os.path.join(output_dir, csv_filename)
lock_file = os.path.join(output_dir, "last_run.lock")
# Checking if the script has already run this week
if os.path.exists(lock_file):
    with open(lock_file, "r") as f:
        last_run = datetime.strptime(f.read().strip(), "%Y-%m-%d")
        current_week = datetime.now().isocalendar()[1]
        last_run_week = last_run.isocalendar()[1]
        if current_week == last_run_week:
            print("Script has already run this week. Exiting.")
            exit()


# Default downloads directory
default_download_dir = "C:\\Users\\jurda\\Downloads"
default_downloaded_file = os.path.join(default_download_dir, f"{USERNAME}.csv")
print("Starting browser...")
edge_options = Options()
edge_options.add_argument("--log-level=3")
edge_options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Edge(
    service=Service(EdgeChromiumDriverManager().install(), options=edge_options)
)
print("Navigating to the website...")
try:
    # Navigate to the website
    driver.get("https://benjaminbenben.com/lastfm-to-csv/")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "lastfm-user"))
    )
    username_input = driver.find_element(By.NAME, "lastfm-user")
    username_input.send_keys(USERNAME)
    username_input.send_keys(Keys.RETURN)
    # Wait for the Save button, monitor file size stability
    print("Waiting for tracks to process...")
    print("Waiting for the Save button with the completed file...")
    time.sleep(1700)
    last_file_size = None
    stable_counter = 0
    save_button = None
    while (
        stable_counter < 3
    ):  # Require 3 consecutive checks where the size does not change
        try:
            save_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//a[contains(@class, 'btn-success') and contains(text(), 'Save')]",
                    )
                )
            )
            save_button_text = save_button.text
            print(f"Found Save button: {save_button_text}")
            # Extract file size in KB from button text
            if "Save" in save_button_text and "KB" in save_button_text:
                file_size = int(save_button_text.split()[1].replace("KB", ""))
                print(f"Current file size: {file_size} KB")

                if file_size == last_file_size:
                    stable_counter += 1  # Increment stability counter
                else:
                    stable_counter = 0  # Reset counter if size changes
                    last_file_size = file_size
            else:
                print(
                    "The Save button does not indicate a valid file size. Retrying..."
                )
            time.sleep(10)
        except Exception as e:
            print(f"Error while waiting for Save button: {e}")
            stable_counter = 0  # Reset counter if the button temporarily disappears

    # Once stable, click the Save button
    if save_button:
        print(f"Final file size detected: {last_file_size} KB. Clicking Save button...")
        save_button.click()
    else:
        print("Save button not found. Exiting.")
        driver.quit()
        exit()

    # Add a delay to ensure the file downloads completely
    time.sleep(10)

    # Move the file from the default downloads directory to the target directory
    if os.path.exists(default_downloaded_file):
        if os.path.exists(csv_filepath):
            print(f"Destination file already exists: {csv_filepath}. Overwriting it.")
            os.remove(csv_filepath)
        try:
            print(f"Moving downloaded file to: {csv_filepath}")
            os.rename(default_downloaded_file, csv_filepath)
            print(f"File successfully moved to: {csv_filepath}")
        except Exception as e:
            print(f"Failed to move the file: {e}")
            print(f"Cleaning up temporary file: {default_downloaded_file}")
            os.remove(default_downloaded_file)  # Clean up the temporary file
    else:
        print(
            f"File not found in the default downloads directory: {default_downloaded_file}"
        )


finally:
    driver.quit()

with open(lock_file, "w") as f:
    f.write(datetime.now().strftime("%Y-%m-%d"))
