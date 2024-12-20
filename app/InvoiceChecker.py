# -*- coding: utf8 -*-
import io
import json
import logging
import os
from datetime import datetime, date
from pathlib import Path
 
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from typing import Dict
import base64
import re
import time
 
 
from PIL import Image
import urllib

 
from check_re import CaptchaPredictor
class InvoiceChecker:
    """Optimized system for checking and processing invoices."""
    
    def __init__(self, path: str, data_dir: str, config: dict):
        self.path = path
        self.data_dir = data_dir
        self.config = config
        self.wait_timeout = 20
        self.max_retries = 3
        self.browser = None
        self.predictor = CaptchaPredictor('captcha_good.keras')
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging with proper format and file handling."""
        log_dir = Path(self.path) / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f'log_{datetime.now().strftime("%Y_%m_%d")}.log'
        logging.basicConfig(
            filename=log_file,
            format='%(levelname)s | %(asctime)s | %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO,
            encoding="utf-8"
        )
    
    def _init_chrome_options(self) -> Options:
        """Initialize Chrome options with optimal settings."""
        chrome_options = Options()
        chrome_options.binary_location = str(Path(self.path) / 'bin' / 'chromium' / 'chrome.exe')
        
        # Clear and setup extensions if using proxy
        if self.config.get('use_proxy') == "True":
            ext_path = Path(self.path) / 'bin' / 'chromium' / 'Extensions'
            self._setup_proxy_extension(ext_path)
            chrome_options.add_argument(f"--load-extension={ext_path}")
        
        # Add standard options
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument("--disable-blink-features=AutomationControllered")
        chrome_options.add_argument('--disable-gpu')
        
        if self.config.get('headless') == "True":
            chrome_options.add_argument("--headless=new")
        
        # Add experimental options
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("prefs", self._get_chrome_prefs())
        
        return chrome_options
    
    def _get_chrome_prefs(self) -> dict:
        """Get Chrome preferences configuration."""
        return {
            "profile.default_content_setting_values.notifications": 2,
            "download.default_directory": self.data_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "download_restrictions": 0,
            'safebrowsing.enabled': False,
            'safebrowsing.disable_download_protection': True
        }
    
    def _setup_proxy_extension(self, ext_path: Path) -> None:
        """Setup proxy extension if enabled."""
        # Clear existing extension files
        for f in ext_path.glob("*"):
            if f.is_file():
                f.unlink()
        
        # Create new proxy extension
        proxy_config = {
            'username': self.config.get('proxy_username'),
            'password': self.config.get('proxy_password'),
            'host': self.config.get('proxy_address'),
            'port': self.config.get('proxy_port')
        }
        self._create_proxy_extension(ext_path, proxy_config)
    
    @staticmethod
    def _create_proxy_extension(ext_path: Path, proxy_config: dict) -> None:
        """Create proxy extension files."""
        manifest = {
            "version": "1.0.0",
            "manifest_version": 3,
            "name": "Chrome Proxy",
            "permissions": ["proxy", "webRequest", "webRequestAuthProvider"],
            "host_permissions": ["<all_urls>"],
            "background": {"service_worker": "service-worker.js"},
            "minimum_chrome_version": "108"
        }
        
        worker_js = f"""
        var config = {{
            mode: "fixed_servers",
            rules: {{
                singleProxy: {{
                    scheme: "http",
                    host: "{proxy_config['host']}",
                    port: {proxy_config['port']}
                }},
                bypassList: ["localhost"]
            }}
        }};
        
        chrome.proxy.settings.set({{value: config, scope: "regular"}}, function() {{}});
        
        function callbackFn(details) {{
            return {{
                authCredentials: {{
                    username: "{proxy_config['username']}",
                    password: "{proxy_config['password']}"
                }}
            }};
        }}
        
        chrome.webRequest.onAuthRequired.addListener(
            callbackFn,
            {{urls: ["<all_urls>"]}},
            ['blocking']
        );
        """
        
        ext_path.mkdir(parents=True, exist_ok=True)
        with open(ext_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        with open(ext_path / "service-worker.js", "w") as f:
            f.write(worker_js)
    
    def init_driver(self) -> webdriver.Chrome:
        """Initialize and configure Chrome WebDriver."""
        chrome_options = self._init_chrome_options()
        driver_path = Path(self.path) / 'bin' / 'driver' / 'chromedriver.exe'
        
        driver = webdriver.Chrome(
            service=Service(executable_path=str(driver_path)),
            options=chrome_options
        )
        
        # Configure download behavior
        driver.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
        params = {
            'cmd': 'Page.setDownloadBehavior',
            'params': {'behavior': 'allow', 'downloadPath': self.data_dir}
        }
        driver.execute("send_command", params)
        driver.implicitly_wait(10)
        
        return driver
    
    def process_invoice_row(self, row: pd.Series) -> Dict:
        """Process a single invoice row."""
        try:
            # Prepare invoice data
            mst = row['MST']
            
            
            # Fill form fields
            self._fill_form_safely('mst', mst)
             
            
            # Handle captcha and get result
            self._handle_captcha()            
             
            
        except Exception as e:
            logging.error(f"Error processing invoice {row['MST']}: {str(e)}")
            return {'error': str(e)}
    
    def _fill_form_safely(self, element_id: str, value: str, clear_first: bool = True) -> None:
        """Safely fill a form field with retry logic."""
        for attempt in range(self.max_retries):
            try:
                element = WebDriverWait(self.browser, self.wait_timeout).until(
                    EC.presence_of_element_located((By.NAME, element_id))
                )
                if clear_first:
                    element.clear()
                    element.send_keys(Keys.CONTROL, 'a')
                    element.send_keys(Keys.DELETE)
                element.send_keys(str(value))
                return
            except StaleElementReferenceException:
                if attempt == self.max_retries - 1:
                    raise
    
    def _handle_captcha(self) -> None:
        """Handle captcha solving with retry logic."""
        captcha_xpath = '//*[@id="tcmst"]/form/table/tbody/tr[6]/td[2]/table/tbody/tr/td[2]/div/img'
        capcha_dir = Path(self.path) / "captcha"
        capcha_dir.mkdir(parents=True, exist_ok=True)
        # Initialize the predictor
        

        # Single image prediction
        
        attempt = 0
        while True:
            try:
                img_element = WebDriverWait(self.browser, self.wait_timeout).until(
                    EC.presence_of_element_located((By.XPATH, captcha_xpath))
                )
                capfile = str(capcha_dir / f"captcha_{attempt}.png")
                image_binary= img_element.screenshot_as_png
                
                img = Image.open(io.BytesIO(image_binary))
                img.save(capfile)
                 
                logging.info(capfile)
                solved_captcha = self.predictor.predict(capfile)
                print(f"Predicted text: {solved_captcha}")
                
                
                
                captcha_input = WebDriverWait(self.browser, self.wait_timeout).until(
                    EC.presence_of_element_located((By.ID, 'captcha'))
                )
                captcha_input.clear()
                captcha_input.send_keys(solved_captcha)
               
                result_element = WebDriverWait(self.browser, self.wait_timeout).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "subBtn"))
                )
                result_element.click()
                
                try:
                    xpath_fa = """/html/body/div/div[1]/div[4]/div[2]/div[2]/div/div/div/p"""
                    result_element = WebDriverWait(self.browser, 5).until(
                        EC.presence_of_element_located((By.XPATH, xpath_fa))
                    )
                    logging.info(result_element.text)
                    if result_element.text == "Vui lòng nhập đúng mã xác nhận!":
                        attempt += 1
                        capcha_diff = capcha_dir / "capcha_error" 
                        capcha_diff.mkdir(parents=True, exist_ok=True)    
                        
                        os.replace(capfile,capcha_diff / f'{solved_captcha}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')                              
                        continue
                    else:
                        return False
                except TimeoutException as e:
                    capcha_ok = capcha_dir / "capcha_ok" 
                    capcha_ok.mkdir(parents=True, exist_ok=True)
                    os.replace(capfile,capcha_ok / f"{solved_captcha}.png") 
                    logging.info("solve captcha")
                    break
            
                
            except Exception as e:
                logging.info({str(e)})
                if attempt == 100:
                    raise Exception(f"Failed to solve captcha: {str(e)}")
    
    
    
    
    def _wait_for_result(self) -> pd.DataFrame:
        """Wait for and return the result text."""
        try:
            
            result_element = WebDriverWait(self.browser, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "ta_border"))
            )
            res =  result_element.get_attribute("outerHTML")
            if "<table class" in res:
                df= pd.read_html(io.StringIO(str(res)))
                #remove last rows
                df = df[0].iloc[:-1, :]
                 
                return df
        
        
        
        except TimeoutException:
            raise Exception("Timeout waiting for result")
    
    
    

    
    def process_invoices(self, excel_path: str)  :
        """Process all invoices from Excel file."""
        try:
            # Initialize browser
            self.browser = self.init_driver()
            self.browser.get('https://tracuunnt.gdt.gov.vn/tcnnt/mstdn.jsp')
            time.sleep(5)
            # Handle initial popup if present
            
            
            # Read and process Excel file
            df = pd.read_excel(
                excel_path,
                dtype={'MST': str} 
            )
           
            re = []
            # Process each row
            for idx, row in df.iterrows():
                result = self.process_invoice_row(row)
                
               
            logging.info(result)
             
            
        finally:
            if self.browser:
                self.browser.quit()
    
    def create_report(self, df: pd.DataFrame ) -> Path:
        """Create Excel and Word reports from processed data."""
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        report_dir = Path(self.path) / 'reports'
        report_dir.mkdir(exist_ok=True)
        df.to_excel(report_dir / f'report_{timestamp}.xlsx', index=False)
        return report_dir
    
    def run(self,filename = "template.xlsx") -> None:
        """Main execution method."""
        try:
            template_path = Path(filename)  
            if not template_path.exists():
                raise FileNotFoundError("Template Excel file not found")
            
            # Process invoices and generate reports
            self.process_invoices(str(template_path))
             
            
            logging.info("Invoice processing completed successfully")
            
        except Exception as e:
            logging.error(f"Error during execution: {str(e)}")
            raise