#!/usr/bin/env python3
"""
Advanced Optimization Techniques for Hebrew Semantle Solver

Implements parallel processing, smart timeout management, semantic gradient ascent,
and emergency strategy switching for maximum solver effectiveness.
"""

import time
import asyncio
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from queue import Queue, PriorityQueue
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OptimizationTask:
    """Represents an optimization task with priority"""
    priority: float
    task_type: str
    data: Any
    timestamp: float
    
    def __lt__(self, other):
        # Higher priority first (lower number = higher priority)
        return self.priority < other.priority


class SmartTimeoutManager:
    """Manages dynamic timeout based on search progress"""
    
    def __init__(self, base_timeout: int = 300, progress_threshold: float = 10.0):
        """
        Initialize smart timeout manager
        
        Args:
            base_timeout: Base timeout in seconds
            progress_threshold: Minimum improvement to extend timeout
        """
        self.base_timeout = base_timeout
        self.progress_threshold = progress_threshold
        self.start_time = None
        self.last_progress_time = None
        self.best_similarity_history = []
        self.timeout_extensions = 0
        self.max_extensions = 3
        
        logger.info(f"Initialized SmartTimeoutManager (base: {base_timeout}s, threshold: {progress_threshold})")
    
    def start_timing(self):
        """Start the timing session"""
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        self.best_similarity_history = []
        self.timeout_extensions = 0
    
    def update_progress(self, current_similarity: float) -> bool:
        """
        Update progress and determine if timeout should be extended
        
        Args:
            current_similarity: Current best similarity score
            
        Returns:
            True if timeout was extended, False otherwise
        """
        if not self.start_time:
            return False
        
        current_time = time.time()
        self.best_similarity_history.append((current_time, current_similarity))
        
        # Check for significant progress
        if self.best_similarity_history:
            recent_improvement = 0.0
            if len(self.best_similarity_history) >= 2:
                recent_improvement = current_similarity - self.best_similarity_history[-2][1]
            
            # Extend timeout for high-similarity discoveries or significant improvements
            should_extend = (
                (current_similarity >= 85.0 and self.timeout_extensions < self.max_extensions) or
                (recent_improvement >= self.progress_threshold and self.timeout_extensions < self.max_extensions)
            )
            
            if should_extend:
                self.last_progress_time = current_time
                self.timeout_extensions += 1
                extension_time = min(60, self.base_timeout * 0.2)  # Max 60s extension
                logger.info(f"Timeout extended by {extension_time}s due to progress "
                           f"(similarity: {current_similarity:.2f}, improvement: {recent_improvement:.2f})")
                return True
        
        return False
    
    def get_remaining_time(self) -> float:
        """Get remaining time including extensions"""
        if not self.start_time:
            return self.base_timeout
        
        elapsed = time.time() - self.start_time
        total_timeout = self.base_timeout + (self.timeout_extensions * min(60, self.base_timeout * 0.2))
        return max(0, total_timeout - elapsed)
    
    def should_continue(self) -> bool:
        """Check if search should continue based on smart timeout"""
        return self.get_remaining_time() > 0


class SemanticGradientOptimizer:
    """Optimizes word exploration using semantic gradient ascent"""
    
    def __init__(self, language_model, step_size: float = 0.1, max_iterations: int = 10):
        """
        Initialize semantic gradient optimizer
        
        Args:
            language_model: Hebrew language model for similarity calculations
            step_size: Step size for gradient ascent
            max_iterations: Maximum gradient steps
        """
        self.language_model = language_model
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.gradient_history = []
        
        logger.info(f"Initialized SemanticGradientOptimizer (step_size: {step_size}, max_iter: {max_iterations})")
    
    def find_semantic_peak(self, current_candidates: List[str], tested_words: set) -> List[str]:
        """
        Find semantic peaks using gradient ascent in word embedding space
        
        Args:
            current_candidates: Current best candidate words
            tested_words: Already tested words to avoid
            
        Returns:
            List of words at semantic peaks
        """
        if not self.language_model or not self.language_model.is_loaded:
            return []
        
        peak_words = []
        
        for candidate in current_candidates[:2]:  # Use top 2 candidates
            try:
                # Get similar words with similarity scores
                similar_words = self.language_model.find_most_similar(candidate, topn=50)
                if not similar_words:
                    continue
                
                # Create gradient direction by finding words with highest similarity gradient
                gradient_candidates = []
                base_similarity = similar_words[0][1] if similar_words else 0.0
                
                for word, similarity in similar_words[:20]:  # Top 20 similar words
                    if word not in tested_words:
                        # Calculate gradient score (improvement potential)
                        gradient_score = similarity - base_similarity + np.random.normal(0, 0.1)  # Add noise
                        gradient_candidates.append((word, gradient_score))
                
                # Sort by gradient score and select top candidates
                gradient_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Add top gradient candidates
                for word, _ in gradient_candidates[:3]:
                    if word not in peak_words:
                        peak_words.append(word)
                
            except Exception as e:
                logger.warning(f"Error in gradient optimization for {candidate}: {e}")
                continue
        
        logger.debug(f"Found {len(peak_words)} semantic peak candidates")
        return peak_words[:5]  # Return top 5 peaks
    
    def optimize_word_vector(self, word: str, target_similarity: float = 90.0) -> List[str]:
        """
        Optimize word vector towards target similarity using gradient ascent
        
        Args:
            word: Starting word
            target_similarity: Target similarity to optimize towards
            
        Returns:
            List of optimized words
        """
        if not self.language_model or not self.language_model.is_loaded:
            return []
        
        try:
            optimized_words = []
            current_word = word
            
            for iteration in range(self.max_iterations):
                # Find words in gradient direction
                similar_words = self.language_model.find_most_similar(current_word, topn=20)
                if not similar_words:
                    break
                
                # Select word with highest similarity as next step
                best_word, best_similarity = similar_words[0]
                
                if best_similarity >= target_similarity:
                    optimized_words.append(best_word)
                    break
                
                # Move in gradient direction
                current_word = best_word
                optimized_words.append(best_word)
            
            return optimized_words
            
        except Exception as e:
            logger.warning(f"Error in vector optimization: {e}")
            return []


class EmergencyStrategyManager:
    """Manages emergency strategies when main strategies fail"""
    
    def __init__(self):
        """Initialize emergency strategy manager"""
        self.emergency_activated = False
        self.activation_threshold = 100  # Activate after 100 words with low progress
        self.low_progress_threshold = 30.0  # Below 30% similarity is low progress
        self.emergency_strategies = [
            'random_sampling',
            'frequency_fallback', 
            'morphological_expansion',
            'desperate_search'
        ]
        self.current_strategy_index = 0
        
        logger.info("Initialized EmergencyStrategyManager")
    
    def should_activate_emergency(self, words_tested: int, best_similarity: float) -> bool:
        """
        Determine if emergency strategies should be activated
        
        Args:
            words_tested: Number of words tested so far
            best_similarity: Best similarity achieved
            
        Returns:
            True if emergency mode should activate
        """
        if (words_tested >= self.activation_threshold and 
            best_similarity < self.low_progress_threshold):
            if not self.emergency_activated:
                self.emergency_activated = True
                logger.warning(f"Emergency strategies activated after {words_tested} words "
                             f"with best similarity {best_similarity:.2f}")
            return True
        return False
    
    def get_emergency_words(self, language_model, tested_words: set, count: int = 10) -> List[str]:
        """
        Generate emergency word suggestions using fallback strategies
        
        Args:
            language_model: Language model (may be None)
            tested_words: Words already tested
            count: Number of words to generate
            
        Returns:
            List of emergency word suggestions
        """
        if not self.emergency_activated:
            return []
        
        current_strategy = self.emergency_strategies[self.current_strategy_index]
        emergency_words = []
        
        try:
            if current_strategy == 'random_sampling':
                emergency_words = self._random_sampling_strategy(language_model, tested_words, count)
            elif current_strategy == 'frequency_fallback':
                emergency_words = self._frequency_fallback_strategy(tested_words, count)
            elif current_strategy == 'morphological_expansion':
                emergency_words = self._morphological_expansion_strategy(tested_words, count)
            elif current_strategy == 'desperate_search':
                emergency_words = self._desperate_search_strategy(language_model, tested_words, count)
            
            logger.info(f"Emergency strategy '{current_strategy}' generated {len(emergency_words)} words")
            
            # Cycle to next strategy for next call
            self.current_strategy_index = (self.current_strategy_index + 1) % len(self.emergency_strategies)
            
        except Exception as e:
            logger.error(f"Error in emergency strategy '{current_strategy}': {e}")
        
        return emergency_words[:count]
    
    def _random_sampling_strategy(self, language_model, tested_words: set, count: int) -> List[str]:
        """Random sampling from vocabulary"""
        if not language_model or not language_model.is_loaded:
            return []
        
        vocab_words = list(language_model.model.key_to_index.keys())
        untested_words = [w for w in vocab_words if w not in tested_words and len(w) >= 3]
        
        import random
        return random.sample(untested_words[:5000], min(count, len(untested_words), 5000))
    
    def _frequency_fallback_strategy(self, tested_words: set, count: int) -> List[str]:
        """High-frequency Hebrew words fallback"""
        high_freq_words = [
            'אני', 'את', 'הוא', 'היא', 'אנחנו', 'אתם', 'הם', 'זה', 'זאת',
            'יש', 'אין', 'היה', 'הייתי', 'יהיה', 'עכשיו', 'פה', 'שם', 'כאן',
            'מתי', 'איך', 'למה', 'מה', 'מי', 'איפה', 'כמה', 'איזה', 'אילו'
        ]
        return [w for w in high_freq_words if w not in tested_words][:count]
    
    def _morphological_expansion_strategy(self, tested_words: set, count: int) -> List[str]:
        """Generate morphological variations of common roots"""
        try:
            from hebrew_utils import generate_morphological_variations
            
            base_words = ['שלום', 'אהבה', 'חיים', 'עבודה', 'משפחה']
            expanded_words = []
            
            for base_word in base_words:
                if len(expanded_words) >= count:
                    break
                variations = generate_morphological_variations(base_word)
                for variation in variations:
                    if variation not in tested_words and len(expanded_words) < count:
                        expanded_words.append(variation)
            
            return expanded_words
            
        except Exception as e:
            logger.warning(f"Morphological expansion failed: {e}")
            return []
    
    def _desperate_search_strategy(self, language_model, tested_words: set, count: int) -> List[str]:
        """Desperate search using unusual word patterns"""
        desperate_words = [
            'קונטרקס', 'איקס', 'זיפיס', 'בלבוס', 'קרפס', 'ג׳ירף', 'זברה',
            'פיל', 'נמר', 'אריה', 'דוב', 'זאב', 'שועל', 'ארנב', 'צבי'
        ]
        return [w for w in desperate_words if w not in tested_words][:count]


class ParallelOptimizer:
    """Manages parallel processing for word similarity computations"""
    
    def __init__(self, max_workers: int = 3, rate_limit_delay: float = 1.0):
        """
        Initialize parallel optimizer with rate limiting
        
        Args:
            max_workers: Maximum number of parallel workers
            rate_limit_delay: Delay between API calls to respect rate limits
        """
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.active_tasks = 0
        
        logger.info(f"Initialized ParallelOptimizer (workers: {max_workers}, delay: {rate_limit_delay}s)")
    
    def batch_test_words(self, api_client, words: List[str], callback: Callable[[str, float], None]) -> Dict[str, float]:
        """
        Test multiple words in parallel with rate limiting
        
        Args:
            api_client: API client for testing words
            words: List of words to test
            callback: Callback function for results
            
        Returns:
            Dictionary of word -> similarity mappings
        """
        if not words:
            return {}
        
        results = {}
        
        # For API rate limiting, we'll use sequential processing but with smart batching
        # This prevents overwhelming the API while still optimizing the order
        
        logger.info(f"Batch testing {len(words)} words with rate limiting")
        
        for i, word in enumerate(words):
            try:
                # Respect rate limit
                if i > 0:
                    time.sleep(self.rate_limit_delay)
                
                # Test word
                similarity = api_client.test_word_similarity(word)
                if similarity is not None:
                    results[word] = similarity
                    callback(word, similarity)
                
                logger.debug(f"Batch progress: {i+1}/{len(words)}")
                
            except Exception as e:
                logger.warning(f"Error testing word '{word}' in batch: {e}")
                continue
        
        logger.info(f"Batch testing completed: {len(results)} successful results")
        return results
    
    def optimize_word_order(self, words: List[str], priorities: List[float]) -> List[str]:
        """
        Optimize word testing order based on priorities
        
        Args:
            words: List of words to optimize
            priorities: Priority scores for each word
            
        Returns:
            Optimized word order
        """
        if len(words) != len(priorities):
            logger.warning("Words and priorities length mismatch, returning original order")
            return words
        
        # Sort by priority (higher priority first)
        word_priority_pairs = list(zip(words, priorities))
        word_priority_pairs.sort(key=lambda x: x[1], reverse=True)
        
        optimized_words = [word for word, _ in word_priority_pairs]
        
        logger.debug(f"Optimized word order based on {len(priorities)} priorities")
        return optimized_words


if __name__ == "__main__":
    # Test the optimization components
    print("Testing Advanced Optimization Techniques...")
    
    # Test timeout manager
    timeout_mgr = SmartTimeoutManager(base_timeout=60)
    timeout_mgr.start_timing()
    print(f"Initial remaining time: {timeout_mgr.get_remaining_time():.1f}s")
    
    # Test progress update
    extended = timeout_mgr.update_progress(75.0)
    print(f"Timeout extended: {extended}")
    print(f"Remaining time after progress: {timeout_mgr.get_remaining_time():.1f}s")
    
    # Test emergency strategy manager
    emergency_mgr = EmergencyStrategyManager()
    should_activate = emergency_mgr.should_activate_emergency(120, 25.0)
    print(f"Emergency activated: {should_activate}")
    
    # Test parallel optimizer
    parallel_opt = ParallelOptimizer(max_workers=2, rate_limit_delay=0.1)
    test_words = ['שלום', 'אהבה', 'חיים']
    priorities = [0.8, 0.9, 0.7]
    optimized = parallel_opt.optimize_word_order(test_words, priorities)
    print(f"Optimized word order: {optimized}")
    
    print("✅ Advanced optimization techniques test completed!")
