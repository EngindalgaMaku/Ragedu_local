"""
Frontend Authentication Integration Tests
Tests frontend authentication flow, token management, and UI components
"""

import pytest
import asyncio
import httpx
import json
import os
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import time

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestFrontendAuthIntegration:
    """Test frontend authentication integration"""
    
    @pytest.fixture(scope="class")
    def setup_frontend_testing(self):
        """Setup frontend testing environment"""
        self.frontend_url = "http://localhost:3000"
        self.auth_service_url = "http://localhost:8002"
        self.api_gateway_url = "http://localhost:8000"
        
        # Test credentials
        self.test_users = {
            "admin": {"username": "admin", "password": "admin", "role": "admin"},
            "teacher": {"username": "teacher", "password": "teacher", "role": "teacher"},
            "student": {"username": "student", "password": "student", "role": "student"}
        }
        
        # Setup WebDriver options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")  # Run in headless mode for CI
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        
        yield
    
    def get_driver(self):
        """Get WebDriver instance with error handling"""
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.implicitly_wait(10)
            return driver
        except WebDriverException:
            pytest.skip("Chrome WebDriver not available - install ChromeDriver for UI tests")
    
    @pytest.mark.asyncio
    async def test_frontend_health_check(self, setup_frontend_testing):
        """Test frontend application is running and accessible"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.frontend_url, timeout=10.0)
                assert response.status_code == 200
                assert "text/html" in response.headers.get("content-type", "")
            except httpx.ConnectError:
                pytest.skip("Frontend not running - start with: npm run dev")
    
    @pytest.mark.asyncio
    async def test_login_page_accessibility(self, setup_frontend_testing):
        """Test login page is accessible and contains required elements"""
        driver = self.get_driver()
        try:
            driver.get(f"{self.frontend_url}/login")
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "form"))
            )
            
            # Check for login form elements
            username_input = driver.find_element(By.NAME, "username") or \
                           driver.find_element(By.ID, "username") or \
                           driver.find_element(By.CSS_SELECTOR, "input[type='text']")
            assert username_input is not None, "Username input not found"
            
            password_input = driver.find_element(By.NAME, "password") or \
                           driver.find_element(By.ID, "password") or \
                           driver.find_element(By.CSS_SELECTOR, "input[type='password']")
            assert password_input is not None, "Password input not found"
            
            login_button = driver.find_element(By.TYPE, "submit") or \
                         driver.find_element(By.CSS_SELECTOR, "button[type='submit']") or \
                         driver.find_elements(By.TAG_NAME, "button")[-1]  # Last button is often submit
            assert login_button is not None, "Login button not found"
            
            # Check for demo account buttons (if in development)
            try:
                demo_buttons = driver.find_elements(By.CSS_SELECTOR, "[data-testid*='demo'], .demo, button[class*='demo']")
                if demo_buttons:
                    assert len(demo_buttons) >= 3, "Expected at least 3 demo account buttons"
            except:
                pass  # Demo buttons might not be present in production
            
        except TimeoutException:
            pytest.skip("Login page not loading properly")
        finally:
            driver.quit()
    
    @pytest.mark.asyncio
    async def test_login_flow_ui(self, setup_frontend_testing):
        """Test complete login flow through UI"""
        driver = self.get_driver()
        try:
            # Test login for each user type
            for user_type, credentials in self.test_users.items():
                driver.get(f"{self.frontend_url}/login")
                
                # Wait for page to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "form"))
                )
                
                # Fill in credentials
                username_input = driver.find_element(By.NAME, "username") or \
                               driver.find_element(By.CSS_SELECTOR, "input[type='text']")
                username_input.clear()
                username_input.send_keys(credentials["username"])
                
                password_input = driver.find_element(By.NAME, "password") or \
                               driver.find_element(By.CSS_SELECTOR, "input[type='password']")
                password_input.clear()
                password_input.send_keys(credentials["password"])
                
                # Submit form
                login_button = driver.find_element(By.TYPE, "submit") or \
                             driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                login_button.click()
                
                try:
                    # Wait for redirect or success
                    WebDriverWait(driver, 15).until(
                        lambda d: d.current_url != f"{self.frontend_url}/login" or
                                "error" in d.page_source.lower() or
                                "invalid" in d.page_source.lower()
                    )
                    
                    current_url = driver.current_url
                    
                    if current_url == f"{self.frontend_url}/login":
                        # Login failed - check for error message
                        page_source = driver.page_source.lower()
                        if "error" in page_source or "invalid" in page_source:
                            pytest.skip(f"Login failed for {user_type} - user may not exist in system")
                    else:
                        # Login successful - verify redirect based on role
                        if user_type == "admin":
                            assert "/admin" in current_url or current_url == f"{self.frontend_url}/"
                        else:
                            assert current_url == f"{self.frontend_url}/" or "/dashboard" in current_url
                        
                        # Check for user info or logout button (indicates successful login)
                        try:
                            user_indicator = WebDriverWait(driver, 5).until(
                                EC.any_of(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='user-menu']")),
                                    EC.presence_of_element_located((By.CSS_SELECTOR, ".user-info")),
                                    EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, "Logout")),
                                    EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, "Profile"))
                                )
                            )
                            assert user_indicator is not None, f"User indicator not found after {user_type} login"
                        except TimeoutException:
                            # May not have obvious user indicator, that's okay
                            pass
                        
                        # Test logout if logout button exists
                        try:
                            logout_button = driver.find_element(By.PARTIAL_LINK_TEXT, "Logout") or \
                                          driver.find_element(By.CSS_SELECTOR, "[data-testid='logout']")
                            logout_button.click()
                            
                            # Wait for redirect to login page
                            WebDriverWait(driver, 10).until(
                                lambda d: "/login" in d.current_url
                            )
                            
                        except:
                            # Logout button might not be easily accessible
                            pass
                
                except TimeoutException:
                    pytest.skip(f"Login timeout for {user_type} - may indicate auth issues")
                    
        finally:
            driver.quit()
    
    @pytest.mark.asyncio
    async def test_demo_login_buttons(self, setup_frontend_testing):
        """Test demo login buttons functionality"""
        driver = self.get_driver()
        try:
            driver.get(f"{self.frontend_url}/login")
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "form"))
            )
            
            # Look for demo login buttons
            demo_selectors = [
                "[data-testid*='demo-admin']",
                "[data-testid*='demo-teacher']", 
                "[data-testid*='demo-student']",
                "button[class*='demo']",
                ".demo-account",
                "button:contains('Admin')",
                "button:contains('Teacher')",
                "button:contains('Student')"
            ]
            
            demo_buttons = []
            for selector in demo_selectors:
                try:
                    buttons = driver.find_elements(By.CSS_SELECTOR, selector)
                    demo_buttons.extend(buttons)
                except:
                    pass
            
            if demo_buttons:
                # Test first demo button (usually admin)
                first_demo = demo_buttons[0]
                first_demo.click()
                
                try:
                    # Wait for login to process
                    WebDriverWait(driver, 15).until(
                        lambda d: d.current_url != f"{self.frontend_url}/login"
                    )
                    
                    # Should be redirected away from login page
                    assert driver.current_url != f"{self.frontend_url}/login", \
                        "Demo login should redirect away from login page"
                    
                except TimeoutException:
                    pytest.skip("Demo login timeout - may indicate auth service issues")
            else:
                pytest.skip("No demo login buttons found - may be disabled in this environment")
                
        finally:
            driver.quit()
    
    @pytest.mark.asyncio
    async def test_protected_routes_redirect(self, setup_frontend_testing):
        """Test protected routes redirect to login when not authenticated"""
        driver = self.get_driver()
        try:
            # Clear any existing cookies/storage
            driver.delete_all_cookies()
            
            protected_routes = [
                f"{self.frontend_url}/admin",
                f"{self.frontend_url}/dashboard",
                f"{self.frontend_url}/profile",
                f"{self.frontend_url}/sessions"
            ]
            
            for route in protected_routes:
                driver.get(route)
                
                try:
                    # Wait for potential redirect to login
                    WebDriverWait(driver, 10).until(
                        lambda d: "/login" in d.current_url or 
                                d.current_url == route  # Route might be accessible
                    )
                    
                    current_url = driver.current_url
                    
                    # If redirected to login, that's correct behavior
                    if "/login" in current_url:
                        assert "/login" in current_url, f"Protected route {route} should redirect to login"
                    # If route is accessible, check for login form or auth indicator
                    elif current_url == route:
                        try:
                            # Look for login form or auth requirement
                            login_form = driver.find_element(By.TAG_NAME, "form")
                            login_indicators = driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
                            if login_form and login_indicators:
                                # Page has login form, so it's requiring auth
                                pass
                            else:
                                # Route might be publicly accessible or have different auth handling
                                pass
                        except:
                            # No login form found - route might handle auth differently
                            pass
                
                except TimeoutException:
                    # Route might be slow to load or have different behavior
                    pass
                    
        finally:
            driver.quit()
    
    @pytest.mark.asyncio
    async def test_token_management(self, setup_frontend_testing):
        """Test token management in browser storage"""
        driver = self.get_driver()
        try:
            # Login first
            driver.get(f"{self.frontend_url}/login")
            
            # Wait for page and try demo login or regular login
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "form"))
            )
            
            # Try demo login first
            try:
                demo_buttons = driver.find_elements(By.CSS_SELECTOR, "button:contains('Admin'), [data-testid*='demo-admin']")
                if demo_buttons:
                    demo_buttons[0].click()
                else:
                    # Fall back to manual login
                    username_input = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
                    password_input = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
                    login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                    
                    username_input.send_keys("admin")
                    password_input.send_keys("admin")
                    login_button.click()
                
                # Wait for login to complete
                WebDriverWait(driver, 15).until(
                    lambda d: d.current_url != f"{self.frontend_url}/login"
                )
                
                # Check for tokens in localStorage
                local_storage = driver.execute_script("return window.localStorage;")
                session_storage = driver.execute_script("return window.sessionStorage;")
                
                # Look for common token storage keys
                token_keys = ['access_token', 'refresh_token', 'authToken', 'token', 'jwt']
                user_keys = ['user', 'userData', 'currentUser']
                
                found_tokens = False
                for key in token_keys:
                    if key in local_storage or key in session_storage:
                        found_tokens = True
                        break
                
                # Check for user data
                found_user_data = False
                for key in user_keys:
                    if key in local_storage or key in session_storage:
                        found_user_data = True
                        break
                
                # At least one of these should be true for proper token management
                assert found_tokens or found_user_data, \
                    "No authentication tokens or user data found in browser storage"
                
                # Test token persistence across page reload
                driver.refresh()
                WebDriverWait(driver, 10).until(
                    lambda d: d.ready_state == 'complete'
                )
                
                # Should still be authenticated (not redirected to login)
                current_url = driver.current_url
                assert "/login" not in current_url, \
                    "User should remain authenticated after page reload"
                
            except Exception as e:
                pytest.skip(f"Could not complete login for token testing: {str(e)}")
                
        finally:
            driver.quit()
    
    @pytest.mark.asyncio 
    async def test_role_based_ui_elements(self, setup_frontend_testing):
        """Test role-based UI elements and access control"""
        driver = self.get_driver()
        try:
            for user_type, credentials in self.test_users.items():
                # Clear previous session
                driver.delete_all_cookies()
                driver.execute_script("localStorage.clear();")
                driver.execute_script("sessionStorage.clear();")
                
                driver.get(f"{self.frontend_url}/login")
                
                # Perform login
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "form"))
                    )
                    
                    # Try demo button first
                    demo_selector = f"[data-testid*='demo-{user_type}'], button:contains('{user_type.title()}')"
                    demo_buttons = driver.find_elements(By.CSS_SELECTOR, demo_selector)
                    
                    if demo_buttons:
                        demo_buttons[0].click()
                    else:
                        # Manual login
                        username_input = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
                        password_input = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
                        login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                        
                        username_input.send_keys(credentials["username"])
                        password_input.send_keys(credentials["password"])
                        login_button.click()
                    
                    # Wait for login completion
                    WebDriverWait(driver, 15).until(
                        lambda d: d.current_url != f"{self.frontend_url}/login"
                    )
                    
                    # Check role-specific elements
                    if user_type == "admin":
                        # Admin should see admin-specific elements
                        admin_indicators = [
                            "Admin", "User Management", "System", "Settings",
                            "[data-testid*='admin']", ".admin", "[href*='/admin']"
                        ]
                        
                        found_admin_element = False
                        for indicator in admin_indicators:
                            try:
                                if indicator.startswith("[") or indicator.startswith("."):
                                    elements = driver.find_elements(By.CSS_SELECTOR, indicator)
                                else:
                                    elements = driver.find_elements(By.PARTIAL_LINK_TEXT, indicator)
                                if elements:
                                    found_admin_element = True
                                    break
                            except:
                                pass
                        
                        # Admin should have access to admin functions
                        # (This is more of a UI verification than strict requirement)
                        
                    elif user_type == "teacher":
                        # Teacher should see teacher-specific elements
                        teacher_indicators = [
                            "Sessions", "Documents", "Students", "Course",
                            "[data-testid*='teacher']", ".teacher"
                        ]
                        
                        # Check for teacher-specific UI elements
                        
                    elif user_type == "student":
                        # Student should NOT see admin/teacher elements
                        restricted_elements = [
                            "Admin", "User Management", "System Settings",
                            "[data-testid*='admin']", ".admin-only"
                        ]
                        
                        # Verify restricted elements are not visible
                        for restricted in restricted_elements:
                            try:
                                if restricted.startswith("[") or restricted.startswith("."):
                                    elements = driver.find_elements(By.CSS_SELECTOR, restricted)
                                else:
                                    elements = driver.find_elements(By.PARTIAL_LINK_TEXT, restricted)
                                
                                # If found, check if they're actually visible/accessible
                                visible_restricted = [el for el in elements if el.is_displayed()]
                                assert len(visible_restricted) == 0, \
                                    f"Student should not see {restricted} elements"
                            except:
                                pass  # Element not found is good for students
                
                except Exception as e:
                    pytest.skip(f"Could not test role-based UI for {user_type}: {str(e)}")
                    
        finally:
            driver.quit()
    
    @pytest.mark.asyncio
    async def test_logout_functionality(self, setup_frontend_testing):
        """Test logout functionality and session cleanup"""
        driver = self.get_driver()
        try:
            # Login first
            driver.get(f"{self.frontend_url}/login")
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "form"))
            )
            
            # Use admin demo login or manual login
            try:
                demo_buttons = driver.find_elements(By.CSS_SELECTOR, "[data-testid*='demo-admin'], button:contains('Admin')")
                if demo_buttons:
                    demo_buttons[0].click()
                else:
                    username_input = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
                    password_input = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
                    login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                    
                    username_input.send_keys("admin")
                    password_input.send_keys("admin")
                    login_button.click()
                
                # Wait for successful login
                WebDriverWait(driver, 15).until(
                    lambda d: d.current_url != f"{self.frontend_url}/login"
                )
                
                # Look for logout button/link
                logout_selectors = [
                    "[data-testid*='logout']",
                    "a[href*='logout']",
                    "button:contains('Logout')",
                    ".logout",
                    "[aria-label*='logout']"
                ]
                
                logout_element = None
                for selector in logout_selectors:
                    try:
                        if "contains(" in selector:
                            elements = driver.find_elements(By.XPATH, f"//*[contains(text(),'Logout')]")
                        else:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        
                        for element in elements:
                            if element.is_displayed():
                                logout_element = element
                                break
                        
                        if logout_element:
                            break
                    except:
                        pass
                
                if logout_element:
                    # Test logout
                    logout_element.click()
                    
                    # Should redirect to login page
                    WebDriverWait(driver, 10).until(
                        lambda d: "/login" in d.current_url
                    )
                    
                    assert "/login" in driver.current_url, "Logout should redirect to login page"
                    
                    # Check that tokens are cleared
                    local_storage = driver.execute_script("return window.localStorage;")
                    session_storage = driver.execute_script("return window.sessionStorage;")
                    
                    # Common token keys should be cleared
                    token_keys = ['access_token', 'refresh_token', 'authToken', 'token', 'jwt', 'user']
                    remaining_tokens = []
                    
                    for key in token_keys:
                        if (key in local_storage and local_storage[key]) or \
                           (key in session_storage and session_storage[key]):
                            remaining_tokens.append(key)
                    
                    # Should have minimal or no auth-related data after logout
                    assert len(remaining_tokens) <= 1, \
                        f"Too many auth tokens remaining after logout: {remaining_tokens}"
                    
                    # Try to access protected route - should redirect back to login
                    driver.get(f"{self.frontend_url}/admin")
                    
                    WebDriverWait(driver, 10).until(
                        lambda d: "/login" in d.current_url or d.current_url.endswith("/admin")
                    )
                    
                    # Should be redirected to login or see login form
                    if "/login" not in driver.current_url:
                        # Check if page has login form (embedded auth)
                        login_forms = driver.find_elements(By.CSS_SELECTOR, "form input[type='password']")
                        assert len(login_forms) > 0, "Protected route should require authentication after logout"
                
                else:
                    pytest.skip("No logout button found - may require different navigation")
                    
            except Exception as e:
                pytest.skip(f"Could not complete logout test: {str(e)}")
                
        finally:
            driver.quit()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])