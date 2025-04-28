import requests
import json
import numpy as np
from typing import Optional
from requests.exceptions import RequestException

# Update base URLs to match Docker container ports
DB_SERVICE_URL = "http://localhost:8001"
NOMIC_SERVICE_URL = "http://localhost:8000"

def test_health() -> None:
    """Test the health endpoint of the database service"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{DB_SERVICE_URL}/health")
        print(f"Status code: {response.status_code}")
        print(f"Raw response: {response.text}")
        try:
            print(f"Response JSON: {response.json()}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {str(e)}")
    except RequestException as e:
        print(f"Request failed: {str(e)}")
    print()

def test_store() -> Optional[int]:
    """Test the store endpoint of the database service
    
    Returns:
        Optional[int]: The ID of the stored document if successful, None otherwise
    """
    print("Testing store endpoint...")
    # Create a sample document with a random embedding
    document = {
        "text": "This is a test document",
        "embedding": np.random.rand(768).tolist(),  # 768-dimensional embedding
        "metadata": {"source": "test", "category": "test"}
    }
    try:
        response = requests.post(f"{DB_SERVICE_URL}/store", json=document)
        print(f"Status code: {response.status_code}")
        print(f"Raw response: {response.text}")
        try:
            print(f"Response JSON: {response.json()}")
            return response.json()["id"] if response.status_code == 200 else None
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {str(e)}")
    except RequestException as e:
        print(f"Request failed: {str(e)}")
    print()
    return None

def test_search(doc_id: Optional[int] = None) -> None:
    """Test the search endpoint of the database service
    
    Args:
        doc_id (Optional[int]): The ID of a previously stored document
    """
    print("Testing search endpoint...")
    # Use a random embedding for the search query
    query = {
        "embedding": np.random.rand(768).tolist(),  # 768-dimensional embedding
        "top_k": 5
    }
    try:
        response = requests.post(f"{DB_SERVICE_URL}/search", json=query)
        print(f"Status code: {response.status_code}")
        print(f"Raw response: {response.text}")
        try:
            print(f"Response JSON: {response.json()}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {str(e)}")
    except RequestException as e:
        print(f"Request failed: {str(e)}")
    print()

def main() -> None:
    """Run all endpoint tests"""
    print("Testing all endpoints...\n")
    test_health()
    doc_id = test_store()
    if doc_id:
        test_search(doc_id)

if __name__ == "__main__":
    main() 