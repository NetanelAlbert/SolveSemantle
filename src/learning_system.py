#!/usr/bin/env python3
"""
Contextual Learning System for Hebrew Semantle Solver

Implements pattern recognition, adaptive weighting, and strategy optimization
based on historical search patterns with exponential decay memory.
"""

import time
import json
import logging
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, asdict

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SearchPattern:
    """Represents a search pattern for learning"""
    candidate_word: str
    candidate_similarity: float
    suggested_word: str
    suggestion_similarity: float
    success_score: float  # Normalized improvement score
    timestamp: float
    search_phase: str
    strategy_used: str


class ContextualLearningSystem:
    """Learning system that adapts based on successful search patterns"""
    
    def __init__(self, memory_limit: int = 1000, decay_factor: float = 0.95):
        """
        Initialize contextual learning system
        
        Args:
            memory_limit: Maximum number of patterns to remember
            decay_factor: Exponential decay factor for pattern weights (0.9-0.99)
        """
        self.memory_limit = memory_limit
        self.decay_factor = decay_factor
        
        # Pattern storage
        self.search_patterns: deque = deque(maxlen=memory_limit)
        self.pattern_weights: Dict[str, float] = defaultdict(float)
        
        # Strategy performance tracking
        self.strategy_success: Dict[str, List[float]] = defaultdict(list)
        self.strategy_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Word success tracking
        self.word_success_history: Dict[str, List[float]] = defaultdict(list)
        self.word_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Puzzle type classification (simplified)
        self.puzzle_characteristics: Dict[str, Any] = {}
        self.current_puzzle_type = 'unknown'
        
        logger.info(f"Initialized ContextualLearningSystem (memory: {memory_limit}, decay: {decay_factor})")
    
    def record_search_pattern(
        self, 
        candidate_word: str,
        candidate_similarity: float,
        suggested_word: str,
        suggestion_similarity: float,
        search_phase: str,
        strategy_used: str
    ):
        """Record a search pattern for learning"""
        
        # Calculate success score (normalized improvement)
        if candidate_similarity > 0:
            improvement = suggestion_similarity - candidate_similarity
            success_score = max(0, improvement / max(candidate_similarity, 1.0))
        else:
            success_score = suggestion_similarity / 100.0  # Normalize to 0-1
        
        pattern = SearchPattern(
            candidate_word=candidate_word,
            candidate_similarity=candidate_similarity,
            suggested_word=suggested_word,
            suggestion_similarity=suggestion_similarity,
            success_score=success_score,
            timestamp=time.time(),
            search_phase=search_phase,
            strategy_used=strategy_used
        )
        
        # Store pattern
        self.search_patterns.append(pattern)
        
        # Update strategy performance
        self.strategy_success[strategy_used].append(success_score)
        
        # Update word success history
        self.word_success_history[suggested_word].append(success_score)
        
        # Create pattern key for pattern weights
        pattern_key = f"{candidate_word[:3]}→{suggested_word[:3]}:{strategy_used}"
        self.pattern_weights[pattern_key] = success_score
        
        logger.debug(f"Recorded pattern: {candidate_word} → {suggested_word} "
                    f"(score: {success_score:.3f}, strategy: {strategy_used})")
    
    def get_adaptive_word_weight(self, word: str, base_score: float = 1.0) -> float:
        """Get adaptive weight for a word based on historical success"""
        
        if word not in self.word_success_history:
            return base_score
        
        # Calculate weighted average with exponential decay
        success_scores = self.word_success_history[word]
        if not success_scores:
            return base_score
        
        # Apply exponential decay based on recency
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for i, score in enumerate(success_scores[-10:]):  # Last 10 uses
            # More recent uses get higher weights
            time_weight = self.decay_factor ** (len(success_scores) - i - 1)
            weighted_sum += score * time_weight
            weight_sum += time_weight
        
        if weight_sum > 0:
            adaptive_weight = weighted_sum / weight_sum
            # Boost for consistently successful words
            if len(success_scores) >= 3 and adaptive_weight > 0.5:
                adaptive_weight *= 1.2
            return base_score * (1.0 + adaptive_weight)
        
        return base_score
    
    def get_strategy_weight(self, strategy: str) -> float:
        """Get adaptive weight for a strategy based on historical performance"""
        
        if strategy not in self.strategy_success or not self.strategy_success[strategy]:
            return 1.0
        
        # Calculate recent performance
        recent_scores = self.strategy_success[strategy][-20:]  # Last 20 uses
        if not recent_scores:
            return 1.0
        
        # Apply exponential decay
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for i, score in enumerate(recent_scores):
            time_weight = self.decay_factor ** (len(recent_scores) - i - 1)
            weighted_sum += score * time_weight
            weight_sum += time_weight
        
        if weight_sum > 0:
            avg_performance = weighted_sum / weight_sum
            # Scale from 0.5 to 1.5 based on performance
            strategy_weight = 0.5 + avg_performance
            return min(1.5, max(0.5, strategy_weight))
        
        return 1.0
    
    def classify_puzzle_type(self, beam_status: Dict[str, Any]) -> str:
        """Classify current puzzle type based on characteristics"""
        
        best_similarity = beam_status.get('best_similarity', 0)
        words_tested = beam_status.get('tested_count', 0)
        strategy = beam_status.get('strategy', 'balanced')
        
        # Simple classification based on progress patterns
        if best_similarity > 70 and words_tested < 50:
            return 'easy'
        elif best_similarity < 40 and words_tested > 80:
            return 'hard'
        elif strategy == 'exploration' and words_tested > 60:
            return 'exploratory'
        elif strategy == 'exploitation' and best_similarity > 60:
            return 'convergent'
        else:
            return 'normal'
    
    def get_learned_suggestions(
        self, 
        current_candidates: List[str], 
        search_phase: str,
        count: int = 5
    ) -> List[Tuple[str, float]]:
        """Get word suggestions based on learned patterns"""
        
        suggestions = []
        
        # Find similar historical patterns
        for pattern in list(self.search_patterns)[-100:]:  # Recent patterns
            # Match by search phase and success
            if (pattern.search_phase == search_phase and 
                pattern.success_score > 0.2):  # Only successful patterns
                
                # Calculate pattern relevance with decay
                time_decay = self.decay_factor ** ((time.time() - pattern.timestamp) / 3600.0)  # hourly decay
                relevance = pattern.success_score * time_decay
                
                if relevance > 0.1:  # Threshold for relevance
                    suggestions.append((pattern.suggested_word, relevance))
        
        # Sort by relevance and remove duplicates
        suggestions = sorted(set(suggestions), key=lambda x: x[1], reverse=True)
        return suggestions[:count]
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system"""
        
        return {
            'patterns_stored': len(self.search_patterns),
            'strategies_tracked': len(self.strategy_success),
            'words_tracked': len(self.word_success_history),
            'top_strategies': sorted(
                [(s, self.get_strategy_weight(s)) for s in self.strategy_success.keys()],
                key=lambda x: x[1], reverse=True
            )[:5],
            'memory_utilization': len(self.search_patterns) / self.memory_limit,
            'puzzle_type': self.current_puzzle_type,
            'avg_success_by_strategy': {
                strategy: sum(scores) / len(scores) if scores else 0.0
                for strategy, scores in self.strategy_success.items()
            }
        }
    
    def save_learning_data(self, filepath: str):
        """Save learning data to file"""
        try:
            data = {
                'patterns': [asdict(p) for p in self.search_patterns],
                'strategy_success': dict(self.strategy_success),
                'word_success_history': dict(self.word_success_history),
                'pattern_weights': dict(self.pattern_weights),
                'metadata': {
                    'memory_limit': self.memory_limit,
                    'decay_factor': self.decay_factor,
                    'saved_timestamp': time.time()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Learning data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def load_learning_data(self, filepath: str):
        """Load learning data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restore patterns
            for pattern_data in data.get('patterns', []):
                pattern = SearchPattern(**pattern_data)
                self.search_patterns.append(pattern)
            
            # Restore other data
            self.strategy_success.update(data.get('strategy_success', {}))
            self.word_success_history.update(data.get('word_success_history', {}))
            self.pattern_weights.update(data.get('pattern_weights', {}))
            
            logger.info(f"Learning data loaded from {filepath}")
            
        except Exception as e:
            logger.warning(f"Could not load learning data: {e}")
    
    def reset_learning_data(self):
        """Reset all learning data (for testing or fresh start)"""
        self.search_patterns.clear()
        self.pattern_weights.clear()
        self.strategy_success.clear()
        self.word_success_history.clear()
        self.current_puzzle_type = 'unknown'
        logger.info("Learning data reset")


if __name__ == "__main__":
    # Test the learning system
    print("Testing Contextual Learning System...")
    
    learning = ContextualLearningSystem(memory_limit=100, decay_factor=0.9)
    
    # Record some test patterns
    test_patterns = [
        ("שלום", 30.0, "חיים", 45.0, "exploration", "semantic"),
        ("חיים", 45.0, "אהבה", 60.0, "exploitation", "morphological"),
        ("אהבה", 60.0, "משפחה", 75.0, "convergence", "frequency"),
    ]
    
    for pattern in test_patterns:
        learning.record_search_pattern(*pattern)
    
    # Test adaptive weights
    print(f"Adaptive weight for 'חיים': {learning.get_adaptive_word_weight('חיים')}")
    print(f"Strategy weight for 'semantic': {learning.get_strategy_weight('semantic')}")
    
    # Test learned suggestions
    suggestions = learning.get_learned_suggestions(['שלום'], 'exploration', 3)
    print(f"Learned suggestions: {suggestions}")
    
    # Test stats
    stats = learning.get_learning_stats()
    print(f"Learning stats: {stats}")
    
    print("✅ Learning system test completed!")
