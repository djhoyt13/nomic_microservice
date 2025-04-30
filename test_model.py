from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model():
    try:
        logger.info("Loading model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
        
        text = "The quick brown fox jumps over the lazy dog"
        logger.info(f"Generating embedding for text: {text}")
        
        embedding = model.encode(text, convert_to_numpy=True)
        logger.info(f"Generated embedding of shape: {embedding.shape}")
        logger.info(f"First few values: {embedding[:5]}")
        
        return True
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    success = test_model()
    print(f"\nTest {'passed' if success else 'failed'}") 