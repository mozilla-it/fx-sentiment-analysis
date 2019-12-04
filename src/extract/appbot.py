from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import time
import logging
import os

logging.basicConfig(level=logging.INFO)

def get_appbot_csv(appbot_username="",appbot_password="",output_filename=""):
    options = Options()
    options.headless = True

    tmpdir = "/tmp"

    appbot_login_url = "https://app.appbot.co/users/sign_in"

    profile = webdriver.FirefoxProfile()
    profile.set_preference('browser.download.folderList', 2)
    profile.set_preference('browser.download.manager.showWhenStarting', False)
    profile.set_preference('browser.download.dir',tmpdir)
    profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'text/csv')

    browser = webdriver.Firefox(profile, options=options, executable_path='geckodriver')

    logging.info("Navigating to login page")
    browser.get(appbot_login_url)

    time.sleep(3)

    username = browser.find_element_by_id("user_email")
    password = browser.find_element_by_id("user_password")

    username.send_keys(appbot_username)
    password.send_keys(appbot_password)

    logging.info("Clicking the login button")
    browser.find_element_by_css_selector(".signin_button").click()

    time.sleep(10)

    tries = 99
    i = 0
    while i < tries:
        logging.info("[{}] Clicking the export dropdown".format(i))
        try:
            browser.find_element_by_css_selector(".b-button--ghost").click()
            break
        except:
            i+=1
            time.sleep(3)

    time.sleep(3)

    logging.info("Clicking the CSV button")
    browser.find_element_by_css_selector(".b-calloutmenu__link").click()

    time.sleep(10)

    logging.info("Copying CSV")
    os.system("cp {}/review_export_report.csv {}".format(tmpdir,output_filename))

    browser.quit()
