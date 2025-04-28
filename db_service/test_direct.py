import requests

def test_health():
    response = requests.get("http://localhost:8000/health")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    test_health() 