import os
import requests
from dotenv import load_dotenv
import sys

def test_nomic_api_key():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the API key from environment variables
    api_key = os.getenv('NOMIC_API_KEY')
    
    if not api_key:
        print("Error: NOMIC_API_KEY not found in .env file")
        sys.exit(1)
    
    print("Testing Nomic API key...")
    
    # Test the API key by making a simple request to Nomic's API
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        # Make a simple request to test the API key
        response = requests.get(
            'https://api-atlas.nomic.ai/v1/embedding/text',
            headers=headers
        )
        
        if response.status_code == 200:
            print("✅ Nomic API key is valid!")
        else:
            print(f"❌ Nomic API key test failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error occurred while testing API key: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_nomic_api_key() 