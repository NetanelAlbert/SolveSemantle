"""
Hebrew Word2Vec Language Model for Semantle Solver

This module provides the same Word2Vec model interface used by Hebrew Semantle
for accurate word similarity predictions and intelligent word exploration.
"""

import os
import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np

try:
    from .hebrew_utils import format_hebrew_output
except ImportError:
    from hebrew_utils import format_hebrew_output

try:
    from gensim.models import KeyedVectors
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    KeyedVectors = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HebrewLanguageModel:
    """
    Hebrew Word2Vec language model matching the implementation used by Semantle
    
    This class provides the same similarity calculations and word exploration
    capabilities as the original Hebrew Semantle game.
    """
    
    def __init__(self, model_path: Optional[str] = None, enable_caching: bool = True):
        """
        Initialize Hebrew language model
        
        Args:
            model_path: Path to Word2Vec model file. If None, will look for common paths.
            enable_caching: Enable caching of similarity calculations for performance
        """
        self.model = None  # Optional[KeyedVectors]
        self.model_path = model_path
        self.is_loaded = False
        self.enable_caching = enable_caching
        
        # Performance optimization caches
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.vector_cache: Dict[str, Optional[np.ndarray]] = {}
        self.similar_words_cache: Dict[str, List[Tuple[str, float]]] = {}
        
        if not GENSIM_AVAILABLE:
            logger.error("Gensim is not available. Please install: pip install gensim")
            raise ImportError("gensim package is required for Word2Vec model")
        
        logger.info(f"Initialized HebrewLanguageModel (caching: {enable_caching})")
    
    def _find_model_file(self) -> Optional[str]:
        """
        Search for Word2Vec model file in common locations
        
        Returns:
            Path to model file or None if not found
        """
        # Common model file names and locations
        possible_paths = [
            # Current directory - native gensim format
            "model.mdl",
            # Current directory - binary format  
            "model.bin",
            "hebrew_w2v.bin",
            "word2vec.bin", 
            
            # Models directory
            "models/model.mdl",
            "models/model.bin",
            "models/hebrew_w2v.bin",
            "models/word2vec.bin",
            
            # Project root
            "../model.mdl",
            "../model.bin",
            "../hebrew_w2v.bin",
            "../word2vec.bin"
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                logger.info(f"Found model file: {abs_path}")
                return abs_path
        
        return None
    
    def _is_valid_hebrew_word(self, word: str) -> bool:
        """
        Check if word contains only valid Hebrew characters
        
        Args:
            word: Word to validate
            
        Returns:
            True if word is valid Hebrew, False otherwise
        """
        if not word or len(word) <= 1:
            return False
        
        # Check if all characters are Hebrew (א to ת)
        return all(ord("א") <= ord(c) <= ord("ת") for c in word if c.strip())
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the Word2Vec model from file
        
        Args:
            force_reload: Force reload even if model is already loaded
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if self.is_loaded and not force_reload:
                logger.info("Model already loaded")
                return True
            
            # Find model file
            model_file = self.model_path or self._find_model_file()
            
            if not model_file:
                logger.error("No Word2Vec model file found. Please download from:")
                logger.error("https://drive.google.com/drive/folders/1RDj6Gaa5t4jtd-VtsAqyZWyk6e7o2Xux")
                logger.error("Or specify model path when initializing HebrewLanguageModel")
                return False
            
            logger.info(f"Loading Word2Vec model from: {model_file}")
            
            # Load model with gensim - handle both binary and native formats
            if model_file.endswith('.bin'):
                # Binary Word2Vec format
                self.model = KeyedVectors.load_word2vec_format(model_file, binary=True)
            elif model_file.endswith('.mdl'):
                # Native gensim Word2Vec format
                from gensim.models import Word2Vec
                full_model = Word2Vec.load(model_file)
                self.model = full_model.wv  # Extract KeyedVectors
            else:
                # Try binary format as default
                self.model = KeyedVectors.load_word2vec_format(model_file, binary=True)
            self.is_loaded = True
            
            # Log model statistics
            vocab_size = len(self.model.key_to_index)
            vector_size = self.model.vector_size
            logger.info(f"Model loaded successfully: {vocab_size} words, {vector_size} dimensions")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Word2Vec model: {e}")
            logger.error("Make sure you have downloaded the Hebrew Word2Vec model")
            self.model = None
            self.is_loaded = False
            return False
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get word vector for given Hebrew word (with caching for performance)
        
        Args:
            word: Hebrew word
            
        Returns:
            Word vector as numpy array or None if not found
        """
        try:
            if not self.is_loaded or self.model is None:
                logger.warning("Model not loaded. Call load_model() first.")
                return None
            
            # Check cache first
            if self.enable_caching and word in self.vector_cache:
                return self.vector_cache[word]
            
            # Validate Hebrew word
            if not self._is_valid_hebrew_word(word):
                if self.enable_caching:
                    self.vector_cache[word] = None
                return None
            
            # Get vector if word exists in model
            if word in self.model:
                vector = self.model[word].copy()
                if self.enable_caching:
                    self.vector_cache[word] = vector
                return vector
            else:
                if self.enable_caching:
                    self.vector_cache[word] = None
                return None
                
        except Exception as e:
            logger.error(f"Error getting vector for word '{word}': {e}")
            return None
    
    def calculate_similarity(self, word1: str, word2: str) -> Optional[float]:
        """
        Calculate semantic similarity between two Hebrew words (with caching)
        
        Args:
            word1: First Hebrew word
            word2: Second Hebrew word
            
        Returns:
            Similarity score (0-100) matching Semantle's scale, or None if calculation fails
        """
        try:
            if not self.is_loaded or self.model is None:
                return None
            
            # Check cache first (order-independent key)
            cache_key = (min(word1, word2), max(word1, word2))
            if self.enable_caching and cache_key in self.similarity_cache:
                return self.similarity_cache[cache_key]
            
            # Get vectors for both words
            vec1 = self.get_vector(word1)
            vec2 = self.get_vector(word2)
            
            if vec1 is None or vec2 is None:
                if self.enable_caching:
                    self.similarity_cache[cache_key] = None
                return None
            
            # Calculate cosine similarity using gensim method (matches Semantle implementation)
            similarity = self.model.cosine_similarities(vec1, np.expand_dims(vec2, axis=0))[0]
            
            # Convert to Semantle scale (0-100) and round to 2 decimal places
            result = round(float(similarity) * 100, 2)
            
            # Cache result
            if self.enable_caching:
                self.similarity_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating similarity between '{word1}' and '{word2}': {e}")
            return None
    
    def find_most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find most semantically similar words to the given word (with caching)
        
        Args:
            word: Hebrew word to find similarities for
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples sorted by similarity (highest first)
        """
        try:
            if not self.is_loaded or self.model is None:
                logger.warning("Model not loaded. Call load_model() first.")
                return []
            
            # Check cache first
            cache_key = f"{word}:{topn}"
            if self.enable_caching and cache_key in self.similar_words_cache:
                return self.similar_words_cache[cache_key]
            
            if not self._is_valid_hebrew_word(word) or word not in self.model:
                logger.debug(f"Word '{word}' not found in model")
                if self.enable_caching:
                    self.similar_words_cache[cache_key] = []
                return []
            
            # Get most similar words using gensim
            similar_words = self.model.most_similar(word, topn=topn)
            
            # Convert to Semantle similarity scale (0-100)
            result = []
            for similar_word, cosine_sim in similar_words:
                if self._is_valid_hebrew_word(similar_word):
                    semantle_similarity = round(cosine_sim * 100, 2)
                    result.append((similar_word, semantle_similarity))
            
            # Cache result
            if self.enable_caching:
                self.similar_words_cache[cache_key] = result
            
            logger.debug(f"Found {len(result)} similar words for '{word}'")
            return result
            
        except Exception as e:
            logger.error(f"Error finding similar words for '{word}': {e}")
            return []
    
    def get_word_suggestions(self, current_candidates: List[str], count: int = 10) -> List[str]:
        """
        Get intelligent word suggestions based on current best candidates
        
        This method combines similarity from multiple candidate words to find
        promising new exploration directions.
        
        Args:
            current_candidates: List of current best candidate words
            count: Number of suggestions to return
            
        Returns:
            List of suggested words for exploration
        """
        try:
            if not self.is_loaded or self.model is None:
                logger.warning("Model not loaded. Cannot provide suggestions.")
                return []
            
            if not current_candidates:
                return []
            
            # Collect similar words from all candidates
            all_suggestions = set()
            for candidate in current_candidates:
                similar_words = self.find_most_similar(candidate, topn=20)
                for word, similarity in similar_words:
                    if similarity > 30.0:  # Only consider reasonably similar words
                        all_suggestions.add(word)
            
            # Convert to list and limit count
            suggestions = list(all_suggestions)[:count]
            
            logger.debug(f"Generated {len(suggestions)} word suggestions from {len(current_candidates)} candidates")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating word suggestions: {e}")
            return []
    
    def get_diverse_word_suggestions(self, tested_words: set, current_candidates: List[str], count: int = 25) -> List[str]:
        """
        Get diverse word suggestions using multiple strategies when normal suggestions are exhausted
        
        Args:
            tested_words: Set of already tested words to avoid duplicates
            current_candidates: Current best candidate words
            count: Number of diverse suggestions to return
            
        Returns:
            List of diverse word suggestions for continued exploration
        """
        try:
            if not self.is_loaded or self.model is None:
                logger.warning("Model not loaded. Cannot provide diverse suggestions.")
                return []
            
            all_suggestions = set()
            
            # Strategy 1: Expand search radius for current candidates (lower threshold)
            logger.debug("Strategy 1: Expanding search radius with lower similarity threshold")
            for candidate in current_candidates:
                similar_words = self.find_most_similar(candidate, topn=100)  # Much larger search
                for word, similarity in similar_words:
                    if similarity > 15.0 and word not in tested_words:  # Much lower threshold
                        all_suggestions.add(word)
                        if len(all_suggestions) >= count * 2:  # Get plenty of candidates
                            break
            
            # Strategy 2: Second-degree similarity (words similar to similar words)
            logger.debug("Strategy 2: Second-degree similarity exploration")
            for candidate in current_candidates[:3]:  # Use top 3 candidates
                first_level = self.find_most_similar(candidate, topn=15)
                for word, _ in first_level[:5]:  # Top 5 from first level
                    if word not in tested_words:
                        second_level = self.find_most_similar(word, topn=30)
                        for word2, similarity in second_level:
                            if similarity > 20.0 and word2 not in tested_words:
                                all_suggestions.add(word2)
            
            # Strategy 3: Morphological variations (Hebrew verb forms)
            logger.debug("Strategy 3: Morphological exploration")
            for candidate in current_candidates[:2]:
                if len(candidate) >= 3:  # Focus on longer words
                    # Try to find morphological variations by similarity
                    variations = self.find_most_similar(candidate, topn=50)
                    for word, similarity in variations:
                        # Look for words with similar roots
                        if (similarity > 25.0 and 
                            word not in tested_words and
                            len(word) >= 3 and 
                            abs(len(word) - len(candidate)) <= 3):  # Similar length
                            all_suggestions.add(word)
            
            # Convert to list and return
            diverse_suggestions = list(all_suggestions)[:count]
            
            logger.info(f"Generated {len(diverse_suggestions)} diverse word suggestions using multiple strategies")
            return diverse_suggestions
            
        except Exception as e:
            logger.error(f"Error generating diverse word suggestions: {e}")
            return []
    
    def get_emergency_word_suggestions(self, tested_words: set, count: int = 40) -> List[str]:
        """
        Emergency word suggestions when all other strategies are exhausted
        
        Uses vocabulary sampling and random exploration to continue searching
        
        Args:
            tested_words: Set of already tested words to avoid duplicates
            count: Number of emergency suggestions to return
            
        Returns:
            List of emergency word suggestions
        """
        try:
            if not self.is_loaded or self.model is None:
                logger.warning("Model not loaded. Cannot provide emergency suggestions.")
                return []
            
            emergency_suggestions = []
            vocab_words = list(self.model.key_to_index.keys())
            
            # Strategy 1: High-frequency words that haven't been tested
            logger.debug("Emergency Strategy 1: High-frequency vocabulary sampling")
            for word in vocab_words[:8000]:  # Top 8K frequent words
                if (self._is_valid_hebrew_word(word) and 
                    word not in tested_words and 
                    len(word) >= 3 and len(word) <= 12):  # Reasonable word length
                    emergency_suggestions.append(word)
                    if len(emergency_suggestions) >= count // 2:
                        break
            
            # Strategy 2: Random sampling from mid-frequency vocabulary
            if len(emergency_suggestions) < count:
                logger.debug("Emergency Strategy 2: Random vocabulary sampling")
                import random
                
                # Sample from different frequency ranges
                ranges = [
                    (8000, 20000),   # Mid-frequency
                    (20000, 50000),  # Lower frequency
                    (1000, 8000)     # Revisit high frequency with random sampling
                ]
                
                for start, end in ranges:
                    range_end = min(end, len(vocab_words))
                    if start >= range_end:
                        continue
                        
                    sample_size = min(2000, range_end - start)
                    sampled_words = random.sample(vocab_words[start:range_end], sample_size)
                    
                    for word in sampled_words:
                        if (self._is_valid_hebrew_word(word) and 
                            word not in tested_words and 
                            len(word) >= 3 and len(word) <= 12):
                            emergency_suggestions.append(word)
                            if len(emergency_suggestions) >= count:
                                break
                    
                    if len(emergency_suggestions) >= count:
                        break
            
            logger.info(f"Generated {len(emergency_suggestions)} emergency word suggestions from vocabulary sampling")
            return emergency_suggestions
            
        except Exception as e:
            logger.error(f"Error generating emergency word suggestions: {e}")
            return []
    
    def get_contextual_suggestions(self, tested_words: set, best_word: str, best_similarity: float, count: int = 20) -> List[str]:
        """
        Get contextual word suggestions based on the single best candidate
        
        Focuses on the highest-scoring word to find semantically related words
        
        Args:
            tested_words: Set of already tested words
            best_word: The current best word found
            best_similarity: Similarity score of the best word
            count: Number of suggestions to return
            
        Returns:
            List of contextually related word suggestions
        """
        try:
            if not self.is_loaded or self.model is None or not best_word:
                return []
            
            contextual_suggestions = []
            
            # Focus intensively on the best word
            logger.debug(f"Contextual exploration around best word: {best_word} ({best_similarity:.2f})")
            
            # Get a large number of similar words with very low threshold
            similar_words = self.find_most_similar(best_word, topn=200)
            
            for word, similarity in similar_words:
                if (word not in tested_words and 
                    self._is_valid_hebrew_word(word) and
                    len(word) >= 2):  # Very permissive
                    contextual_suggestions.append(word)
                    if len(contextual_suggestions) >= count:
                        break
            
            logger.info(f"Generated {len(contextual_suggestions)} contextual suggestions around '{best_word}'")
            return contextual_suggestions
            
        except Exception as e:
            logger.error(f"Error generating contextual suggestions: {e}")
            return []

    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get model statistics and information including cache performance
        
        Returns:
            Dictionary with model information and cache statistics
        """
        if not self.is_loaded or self.model is None:
            return {
                'loaded': False,
                'model_path': self.model_path,
                'error': 'Model not loaded'
            }
        
        stats = {
            'loaded': True,
            'model_path': self.model_path or 'Auto-detected',
            'vocabulary_size': len(self.model.key_to_index),
            'vector_dimensions': self.model.vector_size,
            'model_type': 'Word2Vec (KeyedVectors)',
            'gensim_available': GENSIM_AVAILABLE,
            'caching_enabled': self.enable_caching
        }
        
        # Add cache statistics if caching is enabled
        if self.enable_caching:
            stats.update({
                'cache_stats': {
                    'similarity_cache_size': len(self.similarity_cache),
                    'vector_cache_size': len(self.vector_cache),
                    'similar_words_cache_size': len(self.similar_words_cache),
                    'total_cached_items': (len(self.similarity_cache) + 
                                         len(self.vector_cache) + 
                                         len(self.similar_words_cache))
                }
            })
        
        return stats
    
    def clear_cache(self):
        """Clear all performance caches to free memory"""
        if self.enable_caching:
            self.similarity_cache.clear()
            self.vector_cache.clear()
            self.similar_words_cache.clear()
            logger.info("All caches cleared")


def download_model_instructions():
    """Print instructions for downloading the Hebrew Word2Vec model"""
    print("=" * 60)
    print("HEBREW WORD2VEC MODEL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("To use the language model, you need to download the Hebrew Word2Vec model:")
    print()
    print("1. Go to: https://drive.google.com/drive/folders/1RDj6Gaa5t4jtd-VtsAqyZWyk6e7o2Xux")
    print("2. Download the Word2Vec model file (likely named 'model.bin' or similar)")
    print("3. Place it in one of these locations:")
    print("   - Current directory: ./model.bin")
    print("   - Models directory: ./models/model.bin")
    print("   - Project root: ../model.bin")
    print()
    print("Or specify the path when creating HebrewLanguageModel:")
    print("   model = HebrewLanguageModel(model_path='/path/to/model.bin')")
    print()
    print("The model was trained on Hebrew Wikipedia using the same method")
    print("as Hebrew Semantle for maximum compatibility.")
    print("=" * 60)


def main():
    """Test the Hebrew language model"""
    print("Testing Hebrew Language Model")
    print("=" * 50)
    
    try:
        # Initialize model
        model = HebrewLanguageModel()
        
        # Show model stats
        stats = model.get_model_stats()
        print(f"Model loaded: {stats['loaded']}")
        
        if not stats['loaded']:
            print("Model not found. Attempting to load...")
            if not model.load_model():
                download_model_instructions()
                return
        
        # Test with sample words
        test_words = ["שלום", "אהבה", "בית"]
        print(f"\nTesting similarity calculations:")
        
        for i, word1 in enumerate(test_words):
            for word2 in test_words[i+1:]:
                similarity = model.calculate_similarity(word1, word2)
                if similarity is not None:
                    print(f"  {format_hebrew_output(word1)} <-> {format_hebrew_output(word2)}: {similarity:.2f}")
                else:
                    print(f"  {format_hebrew_output(word1)} <-> {format_hebrew_output(word2)}: Not found in model")
        
        # Test word suggestions
        print(f"\nTesting word suggestions for '{format_hebrew_output(test_words[0])}':")
        similar = model.find_most_similar(test_words[0], topn=5)
        for word, sim in similar:
            print(f"  {format_hebrew_output(word)}: {sim:.2f}")
        
        print("\n" + "=" * 50)
        print("Language model test completed successfully!")
        
    except Exception as e:
        logger.error(f"Language model test failed: {e}")
        download_model_instructions()


if __name__ == "__main__":
    main()
