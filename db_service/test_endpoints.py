import requests
import json
import numpy as np

# Update base URLs to match Docker container ports
DB_SERVICE_URL = "http://localhost:8001"
NOMIC_SERVICE_URL = "http://localhost:8000"

def test_health():
    print("Testing health endpoint...")
    response = requests.get(f"{DB_SERVICE_URL}/health")
    print(f"Status code: {response.status_code}")
    print(f"Raw response: {response.text}")
    try:
        print(f"Response JSON: {response.json()}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {str(e)}")
    print()

def test_store():
    print("Testing store endpoint...")
    # Create a sample document with a random embedding
    document = {
        "text": "This is a test document",
        "embedding": np.random.rand(768).tolist(),  # 768-dimensional embedding
        "metadata": {"source": "test", "category": "test"}
    }
    response = requests.post(f"{DB_SERVICE_URL}/store", json=document)
    print(f"Status code: {response.status_code}")
    print(f"Raw response: {response.text}")
    try:
        print(f"Response JSON: {response.json()}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {str(e)}")
    print()
    return response.json()["id"] if response.status_code == 200 else None

def test_search(doc_id):
    print("Testing search endpoint...")
    # Use the same embedding as the stored document
    query = {
        "embedding": np.random.rand(768).tolist(),  # 768-dimensional embedding
        "top_k": 5
    }
    response = requests.post(f"{DB_SERVICE_URL}/search", json=query)
    print(f"Status code: {response.status_code}")
    print(f"Raw response: {response.text}")
    try:
        print(f"Response JSON: {response.json()}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {str(e)}")
    print()

if __name__ == "__main__":
    print("Testing all endpoints...\n")
    test_health()
    doc_id = test_store()
    if doc_id:
        test_search(doc_id) 