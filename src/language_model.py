"""
Hebrew Word2Vec Language Model for Semantle Solver

This module provides the same Word2Vec model interface used by Hebrew Semantle
for accurate word similarity predictions and intelligent word exploration.
"""

import os
import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple, Set
from pathlib import Path
import numpy as np

try:
    from .hebrew_utils import (
        format_hebrew_output, 
        generate_morphological_variations,
        get_hebrew_word_frequency_features,
        cluster_words_by_similarity,
        calculate_hebrew_similarity,
        COMMON_HEBREW_WORDS
    )
except ImportError:
    from hebrew_utils import (
        format_hebrew_output, 
        generate_morphological_variations,
        get_hebrew_word_frequency_features,
        cluster_words_by_similarity,
        calculate_hebrew_similarity,
        COMMON_HEBREW_WORDS
    )

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
        
        # Enhanced caches
        self.vector_cache: Dict[str, Optional[np.ndarray]] = {}
        self.similar_words_cache: Dict[str, List[Tuple[str, float]]] = {}
        self.morphological_cache: Dict[str, List[str]] = {}  # New cache for morphological variations
        
        # Strategy tracking for adaptive multi-strategy generation
        self.strategy_performance: Dict[str, List[float]] = {
            'semantic': [],
            'morphological': [],
            'frequency': [],
            'clustering': []
        }
        self.current_search_phase = 'exploration'  # 'exploration', 'exploitation', 'convergence'
        self.tested_word_clusters: Set[str] = set()  # Track tested word cluster representatives
        
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

    def get_multi_strategy_word_suggestions(
        self, 
        current_candidates: List[str], 
        tested_words: Set[str], 
        search_phase: str = 'exploration',
        count: int = 15
    ) -> List[str]:
        """
        Get word suggestions using multiple complementary strategies
        
        Combines semantic similarity, morphological patterns, frequency analysis,
        and clustering to generate diverse, high-quality word candidates.
        
        Args:
            current_candidates: Current best candidate words
            tested_words: Set of already tested words to avoid duplicates
            search_phase: Current search phase ('exploration', 'exploitation', 'convergence')
            count: Total number of suggestions to return
            
        Returns:
            List of strategically selected word suggestions
        """
        try:
            if not self.is_loaded or self.model is None:
                logger.warning("Model not loaded. Using fallback suggestions.")
                return self._get_fallback_suggestions(tested_words, count)
            
            if not current_candidates:
                return self._get_exploration_suggestions(tested_words, count)
            
            self.current_search_phase = search_phase
            all_suggestions = {}  # word -> (strategy, score) 
            
            # Strategy 1: Enhanced Semantic Similarity
            semantic_suggestions = self._get_semantic_suggestions(
                current_candidates, tested_words, count // 2
            )
            for word in semantic_suggestions:
                all_suggestions[word] = ('semantic', self._calculate_semantic_score(word, current_candidates))
            
            # Strategy 2: Morphological Pattern Generation
            morphological_suggestions = self._get_morphological_suggestions(
                current_candidates, tested_words, count // 3
            )
            for word in morphological_suggestions:
                all_suggestions[word] = ('morphological', self._calculate_morphological_score(word))
            
            # Strategy 3: Frequency-Based Prioritization
            frequency_suggestions = self._get_frequency_based_suggestions(
                current_candidates, tested_words, count // 4
            )
            for word in frequency_suggestions:
                all_suggestions[word] = ('frequency', self._calculate_frequency_score(word))
            
            # Strategy 4: Semantic Clustering (avoid redundancy)
            if search_phase in ['exploitation', 'convergence']:
                clustered_suggestions = self._get_clustering_suggestions(
                    current_candidates, tested_words, count // 4
                )
                for word in clustered_suggestions:
                    all_suggestions[word] = ('clustering', self._calculate_clustering_score(word, current_candidates))
            
            # Adaptive strategy weighting based on search phase and performance
            final_suggestions = self._apply_adaptive_strategy_weighting(
                all_suggestions, search_phase, count
            )
            
            logger.info(f"Generated {len(final_suggestions)} multi-strategy suggestions "
                       f"(phase: {search_phase})")
            
            return final_suggestions
            
        except Exception as e:
            logger.error(f"Error in multi-strategy word generation: {e}")
            return self.get_word_suggestions(current_candidates, count)
    
    def _get_semantic_suggestions(self, candidates: List[str], tested_words: Set[str], count: int) -> List[str]:
        """Get suggestions based on semantic similarity"""
        suggestions = set()
        
        for candidate in candidates[:3]:  # Use top 3 candidates
            similar_words = self.find_most_similar(candidate, topn=30)
            for word, similarity in similar_words:
                if similarity > 25.0 and word not in tested_words and word not in suggestions:
                    suggestions.add(word)
                    if len(suggestions) >= count:
                        break
            if len(suggestions) >= count:
                break
        
        return list(suggestions)[:count]
    
    def _get_morphological_suggestions(self, candidates: List[str], tested_words: Set[str], count: int) -> List[str]:
        """Get suggestions based on morphological patterns"""
        suggestions = set()
        
        for candidate in candidates[:2]:  # Use top 2 for morphological expansion
            # Check cache first
            if candidate in self.morphological_cache:
                variations = self.morphological_cache[candidate]
            else:
                variations = generate_morphological_variations(candidate)
                self.morphological_cache[candidate] = variations
            
            # Filter valid words using model vocabulary
            for variation in variations:
                if (variation not in tested_words and 
                    variation not in suggestions and
                    variation in self.model and  # Must be in vocabulary
                    self._is_valid_hebrew_word(variation)):
                    suggestions.add(variation)
                    if len(suggestions) >= count:
                        break
            
            if len(suggestions) >= count:
                break
        
        return list(suggestions)[:count]
    
    def _get_frequency_based_suggestions(self, candidates: List[str], tested_words: Set[str], count: int) -> List[str]:
        """Get suggestions prioritized by estimated word frequency"""
        suggestions = []
        
        # Start with common words that haven't been tested
        for word in COMMON_HEBREW_WORDS:
            if word not in tested_words and len(suggestions) < count // 2:
                if word in self.model:  # Must be in vocabulary
                    suggestions.append(word)
        
        # Add high-frequency words from vocabulary
        vocab_words = list(self.model.key_to_index.keys())
        for word in vocab_words[:2000]:  # Top 2K frequent words
            if (word not in tested_words and 
                word not in suggestions and
                len(word) >= 3 and 
                self._is_valid_hebrew_word(word)):
                
                # Use frequency features to prioritize
                features = get_hebrew_word_frequency_features(word)
                if features.get('has_common_prefix', 0) > 0 or features.get('length', 0) <= 5:
                    suggestions.append(word)
                    if len(suggestions) >= count:
                        break
        
        return suggestions[:count]
    
    def _get_clustering_suggestions(self, candidates: List[str], tested_words: Set[str], count: int) -> List[str]:
        """Get suggestions that explore different semantic clusters"""
        suggestions = []
        
        # Get potential words from semantic expansion
        potential_words = set()
        for candidate in candidates:
            similar_words = self.find_most_similar(candidate, topn=50)
            for word, _ in similar_words:
                if word not in tested_words:
                    potential_words.add(word)
        
        # Cluster potential words and select representatives
        if potential_words:
            word_list = list(potential_words)[:100]  # Limit for performance
            clusters = cluster_words_by_similarity(word_list, similarity_threshold=0.6)
            
            for cluster in clusters:
                if suggestions and len(suggestions) >= count:
                    break
                
                # Select the best representative from each cluster
                best_word = None
                best_score = -1
                
                for word in cluster:
                    if word not in tested_words:
                        # Calculate composite score for cluster representative
                        semantic_score = self._calculate_semantic_score(word, candidates)
                        frequency_score = self._calculate_frequency_score(word)
                        composite_score = 0.7 * semantic_score + 0.3 * frequency_score
                        
                        if composite_score > best_score:
                            best_score = composite_score
                            best_word = word
                
                if best_word and best_word not in suggestions:
                    suggestions.append(best_word)
                    # Track that we've explored this cluster
                    self.tested_word_clusters.add(best_word)
        
        return suggestions[:count]
    
    def _calculate_semantic_score(self, word: str, candidates: List[str]) -> float:
        """Calculate semantic relevance score"""
        if not candidates or word not in self.model:
            return 0.0
        
        total_similarity = 0.0
        for candidate in candidates:
            if candidate in self.model:
                try:
                    similarity = self.model.similarity(word, candidate)
                    total_similarity += similarity
                except KeyError:
                    continue
        
        return total_similarity / len(candidates) if candidates else 0.0
    
    def _calculate_morphological_score(self, word: str) -> float:
        """Calculate morphological pattern quality score"""
        features = get_hebrew_word_frequency_features(word)
        
        # Prefer words with common morphological features
        score = 0.0
        score += features.get('has_common_prefix', 0) * 0.3
        score += features.get('has_common_suffix', 0) * 0.2
        score += features.get('common_letter_ratio', 0) * 0.3
        
        # Length preference
        if features.get('is_medium', 0) > 0:  # 4-6 letters
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_frequency_score(self, word: str) -> float:
        """Calculate estimated frequency score"""
        if not word or word not in self.model:
            return 0.0
        
        # Use position in vocabulary as frequency proxy (lower index = higher frequency)
        vocab_position = self.model.key_to_index.get(word, len(self.model.key_to_index))
        vocab_size = len(self.model.key_to_index)
        
        # Normalize to 0-1 scale (higher score = more frequent)
        frequency_score = 1.0 - (vocab_position / vocab_size)
        
        # Boost score for common Hebrew patterns
        features = get_hebrew_word_frequency_features(word)
        pattern_boost = features.get('common_letter_ratio', 0) * 0.2
        
        return min(1.0, frequency_score + pattern_boost)
    
    def _calculate_clustering_score(self, word: str, candidates: List[str]) -> float:
        """Calculate score for semantic cluster diversity"""
        if not candidates:
            return 0.5
        
        # Reward words that are different from existing candidates
        min_similarity = 1.0
        for candidate in candidates:
            similarity = calculate_hebrew_similarity(word, candidate)
            min_similarity = min(min_similarity, similarity)
        
        # Higher score for more diverse words
        diversity_score = 1.0 - min_similarity
        return diversity_score
    
    def _apply_adaptive_strategy_weighting(
        self, 
        suggestions: Dict[str, Tuple[str, float]], 
        search_phase: str, 
        count: int
    ) -> List[str]:
        """Apply adaptive weighting based on search phase and strategy performance"""
        
        # Define phase-based strategy weights
        phase_weights = {
            'exploration': {'semantic': 0.4, 'morphological': 0.3, 'frequency': 0.2, 'clustering': 0.1},
            'exploitation': {'semantic': 0.5, 'morphological': 0.2, 'frequency': 0.1, 'clustering': 0.2}, 
            'convergence': {'semantic': 0.6, 'morphological': 0.1, 'frequency': 0.1, 'clustering': 0.2}
        }
        
        weights = phase_weights.get(search_phase, phase_weights['exploration'])
        
        # Calculate weighted scores
        weighted_suggestions = []
        for word, (strategy, score) in suggestions.items():
            weight = weights.get(strategy, 0.1)
            weighted_score = score * weight
            weighted_suggestions.append((word, weighted_score, strategy))
        
        # Sort by weighted score and return top suggestions
        weighted_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        final_suggestions = []
        strategy_counts = {strategy: 0 for strategy in weights.keys()}
        
        # Ensure diversity across strategies
        for word, score, strategy in weighted_suggestions:
            if len(final_suggestions) >= count:
                break
            
            # Add word if we haven't exceeded strategy quota
            max_per_strategy = max(1, count // len(weights))
            if strategy_counts[strategy] < max_per_strategy:
                final_suggestions.append(word)
                strategy_counts[strategy] += 1
        
        # Fill remaining slots with highest scoring words
        remaining = count - len(final_suggestions)
        if remaining > 0:
            remaining_words = [word for word, _, _ in weighted_suggestions 
                             if word not in final_suggestions]
            final_suggestions.extend(remaining_words[:remaining])
        
        logger.debug(f"Strategy distribution: {strategy_counts}")
        return final_suggestions[:count]
    
    def _get_exploration_suggestions(self, tested_words: Set[str], count: int) -> List[str]:
        """Get initial exploration suggestions when no candidates are available"""
        suggestions = []
        
        # Start with high-frequency common words
        for word in COMMON_HEBREW_WORDS:
            if word not in tested_words and len(suggestions) < count:
                suggestions.append(word)
        
        # Add diverse vocabulary words
        if len(suggestions) < count and self.model:
            vocab_words = list(self.model.key_to_index.keys())
            step_size = len(vocab_words) // (count * 2)  # Spread across vocabulary
            
            for i in range(0, min(len(vocab_words), count * 10), step_size):
                word = vocab_words[i]
                if (word not in tested_words and 
                    word not in suggestions and
                    len(word) >= 3 and
                    self._is_valid_hebrew_word(word)):
                    suggestions.append(word)
                    if len(suggestions) >= count:
                        break
        
        return suggestions[:count]
    
    def _get_fallback_suggestions(self, tested_words: Set[str], count: int) -> List[str]:
        """Fallback suggestions when model is not available"""
        fallback_words = [
            'שלום', 'אהבה', 'חיים', 'בית', 'משפחה', 'עבודה', 'זמן', 'יום',
            'לילה', 'אור', 'חושך', 'מים', 'אש', 'רוח', 'שמים', 'אדמה',
            'אדם', 'איש', 'אישה', 'ילד', 'ילדה', 'אב', 'אם', 'אח', 'אחות'
        ]
        
        suggestions = [word for word in fallback_words 
                      if word not in tested_words][:count]
        
        # Extend with morphological variations if needed
        if len(suggestions) < count:
            for word in suggestions[:3]:  # Use first few as seed
                variations = generate_morphological_variations(word)
                for variation in variations:
                    if variation not in tested_words and variation not in suggestions:
                        suggestions.append(variation)
                        if len(suggestions) >= count:
                            break
                if len(suggestions) >= count:
                    break
        
        return suggestions[:count]

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
