import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def run_api_tests():
    print("==================================================")
    print("STARTING API INTEGRATION TESTS")
    print("==================================================")
    
    # 1. Test root endpoint
    try:
        r = requests.get(f"{BASE_URL}/")
        print(f"1. Root Endpoint Status: {r.status_code}")
        print(f"   Payload: {r.json()}\n")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server. Make sure it is running on http://127.0.0.1:8000")
        sys.exit(1)
        
    # 2. Register a new patient
    import time
    patient_id = f"pat_{int(time.time())}"
    patient_payload = {
        "PatientID": patient_id,
        "Age": 68.5,
        "Gender": 1,
        "Unit1": 1.0,
        "Unit2": 0.0,
        "HospAdmTime": -5.5
    }
    r = requests.post(f"{BASE_URL}/patients", json=patient_payload)
    print(f"2. Register Patient Status: {r.status_code}")
    print(f"   Payload: {r.json()}\n")
    
    # 3. Log vitals hourly
    hours_data = [
        {"hour": 1, "HR": 80.0, "O2Sat": 99.0, "Temp": 36.8, "SBP": 120.0, "MAP": 80.0, "DBP": 70.0, "Resp": 16.0, "ICULOS": 1.0},
        {"hour": 2, "HR": 88.0, "O2Sat": 98.0, "Temp": 37.0, "SBP": 115.0, "MAP": 78.0, "DBP": 68.0, "Resp": 18.0, "ICULOS": 2.0},
        {"hour": 3, "HR": 105.0, "O2Sat": 93.0, "Temp": 38.2, "SBP": 95.0, "MAP": 65.0, "DBP": 55.0, "Resp": 24.0, "ICULOS": 3.0}
    ]
    
    print("3. Logging hourly vitals logs:")
    for data in hours_data:
        r = requests.post(f"{BASE_URL}/patients/{patient_id}/vitals", json=data)
        print(f"   - Hour {data['hour']} Vitals Status: {r.status_code} | Shock Index calculated: {r.json().get('Shock_Index')}")
    print()
    
    # 4. Trigger Multi-Agent Clinical whiteboarding and Sepsis processing
    print("4. Executing Multi-Agent Clinical Processing:")
    r = requests.post(f"{BASE_URL}/patients/{patient_id}/process")
    print(f"   Status: {r.status_code}")
    response_data = r.json()
    print(f"   Risk Probability: {response_data.get('probability')*100:.2f}% | Alarm triggered: {response_data.get('alert_triggered')}")
    print(f"   Clinical Summary:\n   {response_data.get('clinical_summary')}\n")
    if response_data.get('dispatch_message'):
        print(f"   Pager Dispatch Alert: {response_data.get('dispatch_message')}\n")
        
    # 5. Fetch vitals history
    r = requests.get(f"{BASE_URL}/patients/{patient_id}/history")
    print(f"5. Fetch Vitals History Status: {r.status_code} | Total hours logged: {len(r.json())}\n")
    
    # 6. Fetch active alerts
    r = requests.get(f"{BASE_URL}/alerts")
    print(f"6. Fetch Active Alerts Status: {r.status_code}")
    print(f"   Latest Alert Dispatch: {r.json()[0]['dispatch_message'] if len(r.json()) > 0 else 'None'}\n")
    
    # 7. Fetch Explainable AI (XAI) feature contributions
    print(f"7. Fetching Local SHAP/LIME Explanations for Hour 3:")
    r = requests.get(f"{BASE_URL}/patients/{patient_id}/explain/3")
    print(f"   Status: {r.status_code}")
    exp_payload = r.json()
    print("   Top 3 SHAP feature drivers:")
    for contrib in exp_payload.get("shap_contributions", [])[:3]:
        print(f"    - {contrib['feature']} | value: {contrib['value']:.2f} | impact: {contrib['impact']:.4f}")
    print()
    
    print("==================================================")
    print("API INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_api_tests()
