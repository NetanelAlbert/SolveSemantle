"""
Hebrew Semantle API Client

Provides interface to communicate with semantle.ishefi.com API
for testing word similarity in the Hebrew Semantle game.
"""

import requests
import logging
from typing import Optional, Dict, Any
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemantheAPIClient:
    """Client for communicating with Hebrew Semantle API"""
    
    def __init__(self):
        self.base_url = "https://semantle.ishefi.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'SolveSemantle/1.0'
        })
    
    def test_word_similarity(self, word: str) -> Optional[float]:
        """
        Test similarity of a Hebrew word with today's target word
        
        Args:
            word: Hebrew word to test
            
        Returns:
            Similarity score (0-100) or None if API call fails
        """
        try:
            # URL encode the Hebrew word
            encoded_word = quote(word)
            url = f"{self.base_url}/distance?word={encoded_word}"
            
            logger.info(f"Testing word: {word}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse JSON response - expecting array with one object
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                similarity = data[0].get('similarity')
                if similarity is not None:
                    logger.info(f"Word '{word}' similarity: {similarity}")
                    return float(similarity)
                else:
                    logger.warning(f"No similarity score in response for word: {word}")
                    return None
            else:
                logger.warning(f"Unexpected response format for word: {word}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for word '{word}': {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse API response for word '{word}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error testing word '{word}': {e}")
            return None
    
    def close(self):
        """Close the HTTP session"""
        self.session.close()


def main():
    """Test the API client with sample Hebrew words"""
    client = SemantheAPIClient()
    
    # Test with common Hebrew words
    test_words = ["שלום", "חיים", "אהבה", "בית", "משפחה"]
    
    print("Testing Hebrew Semantle API Client")
    print("=" * 40)
    
    try:
        for word in test_words:
            similarity = client.test_word_similarity(word)
            if similarity is not None:
                print(f"✓ {word}: {similarity:.2f}")
            else:
                print(f"✗ {word}: API call failed")
            
        print("=" * 40)
        print("API client test completed")
        
    finally:
        client.close()


if __name__ == "__main__":
    main()
