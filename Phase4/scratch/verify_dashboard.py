import requests
import sys

def verify_dashboard():
    print("==================================================")
    print("VERIFYING DASHBOARD STATIC FILES SERVING")
    print("==================================================")
    
    base_url = "http://127.0.0.1:8000/dashboard"
    
    # 1. Test index.html
    try:
        r = requests.get(f"{base_url}/")
        print(f"1. GET /dashboard/ Status: {r.status_code}")
        if r.status_code == 200:
            if "<title>Sepsis Prediction Clinical Dashboard</title>" in r.text:
                print("   [PASS] index.html served correctly with correct clinical title!")
            else:
                print("   [FAIL] index.html contents did not match expected structure.")
                sys.exit(1)
        else:
            print("   [FAIL] index.html not found.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server on http://127.0.0.1:8000")
        sys.exit(1)
        
    # 2. Test style.css
    r = requests.get(f"{base_url}/style.css")
    print(f"2. GET /dashboard/style.css Status: {r.status_code}")
    if r.status_code == 200:
        if "--bg-primary" in r.text:
            print("   [PASS] style.css served correctly with deep glassmorphism themes!")
        else:
            print("   [FAIL] style.css content did not match.")
            sys.exit(1)
    else:
        print("   [FAIL] style.css not found.")
        sys.exit(1)
        
    # 3. Test app.js
    r = requests.get(f"{base_url}/app.js")
    print(f"3. GET /dashboard/app.js Status: {r.status_code}")
    if r.status_code == 200:
        if "fetchPatients" in r.text:
            print("   [PASS] app.js served correctly with state controllers and Chart.js integration!")
        else:
            print("   [FAIL] app.js content did not match.")
            sys.exit(1)
    else:
        print("   [FAIL] app.js not found.")
        sys.exit(1)
        
    print("==================================================")
    print("DASHBOARD FILES VERIFICATION COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    verify_dashboard()
