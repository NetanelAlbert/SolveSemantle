"""
Hebrew Semantle Beam Search System

Implements beam search algorithm for efficient word exploration
based on semantic similarity scores from the Semantle API.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import heapq

try:
    from .hebrew_utils import format_hebrew_output
except ImportError:
    from hebrew_utils import format_hebrew_output

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WordCandidate:
    """Represents a word candidate with its similarity score"""
    word: str
    similarity: float
    
    def __lt__(self, other):
        """Enable heap operations - use negative similarity for max-heap behavior"""
        return self.similarity < other.similarity


class BeamSearcher:
    """Enhanced beam search system with dynamic width and smart candidate selection"""
    
    def __init__(self, beam_width: int = 5, min_beam_width: int = 2, max_beam_width: int = 8):
        """
        Initialize enhanced beam search system
        
        Args:
            beam_width: Initial beam width (default: 5)
            min_beam_width: Minimum beam width for contraction (default: 2)
            max_beam_width: Maximum beam width for expansion (default: 8)
        """
        self.initial_beam_width = max(1, beam_width)
        self.min_beam_width = max(1, min_beam_width)
        self.max_beam_width = max(beam_width, max_beam_width)
        self.current_beam_width = self.initial_beam_width
        
        self.tested_words: Set[str] = set()
        self.current_beam: List[WordCandidate] = []
        self.best_candidate: Optional[WordCandidate] = None
        
        # Progress tracking for dynamic adjustment
        self.similarity_history: List[float] = []
        self.recent_improvements = 0
        self.stagnation_count = 0
        self.last_best_similarity = 0.0
        
        # Diversity thresholds
        self.min_diversity_threshold = 0.3  # Minimum semantic diversity to maintain
        
        logger.info(f"Initialized Enhanced BeamSearcher - initial width: {self.current_beam_width}, "
                   f"range: [{self.min_beam_width}, {self.max_beam_width}]")
    
    def _calculate_semantic_diversity(self, new_candidate: WordCandidate) -> float:
        """
        Calculate semantic diversity score for a new candidate
        
        Higher diversity means the candidate is semantically different from existing beam.
        This is a simplified heuristic based on word length and character differences.
        
        Args:
            new_candidate: Candidate to evaluate for diversity
            
        Returns:
            Diversity score between 0.0 (low diversity) and 1.0 (high diversity)
        """
        if not self.current_beam:
            return 1.0  # Maximum diversity for first candidate
        
        # Calculate average character-based diversity
        total_diversity = 0.0
        for existing_candidate in self.current_beam:
            # Simple string-based diversity metric (can be enhanced with embeddings)
            word_diff = abs(len(new_candidate.word) - len(existing_candidate.word))
            char_diff = len(set(new_candidate.word) - set(existing_candidate.word))
            
            # Normalize diversity score
            diversity = (word_diff + char_diff) / (len(new_candidate.word) + len(existing_candidate.word) + 1)
            total_diversity += diversity
        
        avg_diversity = total_diversity / len(self.current_beam)
        return min(1.0, avg_diversity)  # Cap at 1.0
    
    def _update_progress_tracking(self, similarity: float):
        """
        Update progress tracking metrics for dynamic beam adjustment
        
        Args:
            similarity: Latest similarity score achieved
        """
        self.similarity_history.append(similarity)
        
        # Check for improvement
        if similarity > self.last_best_similarity:
            self.recent_improvements += 1
            self.stagnation_count = 0
            self.last_best_similarity = similarity
        else:
            self.stagnation_count += 1
            # Reset improvement counter if no progress for a while
            if self.stagnation_count >= 3:
                self.recent_improvements = 0
    
    def _adjust_beam_width(self, similarity: float):
        """
        Dynamically adjust beam width based on search progress
        
        Args:
            similarity: Current similarity score
        """
        old_width = self.current_beam_width
        
        # Expand beam when finding high-similarity words (exploration)
        if similarity >= 60.0 and self.current_beam_width < self.max_beam_width:
            self.current_beam_width = min(self.max_beam_width, self.current_beam_width + 1)
            logger.debug(f"Beam expanded to {self.current_beam_width} (high similarity: {similarity:.2f})")
        
        # Contract beam when stagnating (exploitation)
        elif self.stagnation_count >= 5 and self.current_beam_width > self.min_beam_width:
            self.current_beam_width = max(self.min_beam_width, self.current_beam_width - 1)
            logger.debug(f"Beam contracted to {self.current_beam_width} (stagnation: {self.stagnation_count})")
        
        # Adjust current beam size if width changed
        if self.current_beam_width != old_width:
            self._resize_current_beam()
    
    def _resize_current_beam(self):
        """Resize current beam to match new beam width"""
        if len(self.current_beam) > self.current_beam_width:
            # Keep only the best candidates when contracting
            self.current_beam = sorted(self.current_beam, key=lambda c: c.similarity, reverse=True)[:self.current_beam_width]
            logger.debug(f"Beam resized to {self.current_beam_width} candidates")


    def add_candidate(self, word: str, similarity: float) -> bool:
        """
        Add a new word candidate with enhanced beam management
        
        Uses dynamic beam width and diversity-aware replacement strategy.
        
        Args:
            word: Hebrew word tested
            similarity: Similarity score from API (0-100)
            
        Returns:
            True if candidate was added to beam, False if filtered out
        """
        try:
            # Skip if already tested
            if word in self.tested_words:
                logger.debug(f"Skipping duplicate word: {word}")
                return False
            
            # Add to tested set
            self.tested_words.add(word)
            
            # Create candidate
            candidate = WordCandidate(word, similarity)
            
            # Update progress tracking
            self._update_progress_tracking(similarity)
            
            # Update best candidate
            if self.best_candidate is None or similarity > self.best_candidate.similarity:
                self.best_candidate = candidate
                logger.info(f"New best candidate: {format_hebrew_output(word)} (similarity: {similarity:.2f})")
            
            # Dynamic beam width adjustment
            self._adjust_beam_width(similarity)
            
            # Add to beam with smart replacement logic
            if len(self.current_beam) < self.current_beam_width:
                self.current_beam.append(candidate)
                logger.debug(f"Added candidate to beam: {word} (similarity: {similarity:.2f})")
                return True
            else:
                # Smart replacement considering diversity
                return self._smart_replace_candidate(candidate)
                    
        except Exception as e:
            logger.error(f"Error adding candidate '{word}': {e}")
            return False
    
    def _smart_replace_candidate(self, new_candidate: WordCandidate) -> bool:
        """
        Smart candidate replacement considering both similarity and diversity
        
        Args:
            new_candidate: Candidate to potentially add to beam
            
        Returns:
            True if candidate was added, False if rejected
        """
        # Calculate diversity of new candidate
        diversity_score = self._calculate_semantic_diversity(new_candidate)
        
        # Find worst candidate in beam
        worst_idx = min(range(len(self.current_beam)), 
                       key=lambda i: self.current_beam[i].similarity)
        worst_candidate = self.current_beam[worst_idx]
        
        # Replace if significantly better similarity or if improves diversity
        similarity_improvement = new_candidate.similarity - worst_candidate.similarity
        
        should_replace = False
        replacement_reason = ""
        
        # Always replace if significantly better similarity
        if similarity_improvement > 5.0:
            should_replace = True
            replacement_reason = "similarity improvement"
        
        # Replace if better similarity AND good diversity
        elif similarity_improvement > 0.0 and diversity_score >= self.min_diversity_threshold:
            should_replace = True
            replacement_reason = "similarity + diversity"
        
        # Replace if much better diversity even with slightly lower similarity
        elif diversity_score > 0.7 and similarity_improvement > -2.0:
            should_replace = True
            replacement_reason = "high diversity"
        
        if should_replace:
            self.current_beam[worst_idx] = new_candidate
            logger.debug(f"Smart replacement ({replacement_reason}): {worst_candidate.word} -> {new_candidate.word}")
            return True
        else:
            logger.debug(f"Candidate rejected (insufficient improvement): {new_candidate.word}")
            return False
    
    def get_top_candidates(self, count: Optional[int] = None) -> List[WordCandidate]:
        """
        Get top candidates from current beam, sorted by similarity (highest first)
        
        Args:
            count: Number of top candidates to return (default: all beam candidates)
            
        Returns:
            List of top candidates sorted by similarity score (descending)
        """
        try:
            if not self.current_beam:
                logger.warning("No candidates available in beam")
                return []
            
            # Sort beam by similarity (highest first)
            sorted_beam = sorted(self.current_beam, key=lambda c: c.similarity, reverse=True)
            
            # Return requested count or all if count not specified
            if count is None:
                return sorted_beam
            else:
                return sorted_beam[:count]
                
        except Exception as e:
            logger.error(f"Error retrieving top candidates: {e}")
            return []
    
    def is_word_tested(self, word: str) -> bool:
        """
        Check if word has already been tested
        
        Args:
            word: Hebrew word to check
            
        Returns:
            True if word was already tested, False otherwise
        """
        return word in self.tested_words
    
    def get_beam_status(self) -> Dict[str, any]:
        """
        Get enhanced beam search status and statistics
        
        Returns:
            Dictionary with beam statistics including dynamic adjustments
        """
        return {
            'tested_count': len(self.tested_words),
            'beam_size': len(self.current_beam),
            'beam_width': self.current_beam_width,
            'initial_beam_width': self.initial_beam_width,
            'beam_width_range': f"[{self.min_beam_width}, {self.max_beam_width}]",
            'best_similarity': self.best_candidate.similarity if self.best_candidate else 0.0,
            'best_word': self.best_candidate.word if self.best_candidate else None,
            'recent_improvements': self.recent_improvements,
            'stagnation_count': self.stagnation_count,
            'strategy': 'exploration' if self.current_beam_width > self.initial_beam_width else 
                       'exploitation' if self.current_beam_width < self.initial_beam_width else 'balanced'
        }
    
    def clear_beam(self):
        """Clear all beam search state for a fresh start"""
        self.tested_words.clear()
        self.current_beam.clear()
        self.best_candidate = None
        logger.info("Beam search state cleared")


def main():
    """Test the beam search system with sample data"""
    searcher = BeamSearcher(beam_width=3)
    
    # Test with sample Hebrew words and mock similarity scores
    test_data = [
        ("שלום", 24.15),
        ("חיים", 31.42),
        ("אהבה", 18.76),
        ("בית", 42.33),
        ("משפחה", 28.91),
        ("ישראל", 35.67),
    ]
    
    print("Testing Beam Search System")
    print("=" * 40)
    
    try:
        # Add all test candidates
        for word, similarity in test_data:
            added = searcher.add_candidate(word, similarity)
            print(f"{'✓' if added else '✗'} {format_hebrew_output(word)}: {similarity:.2f} {'(added to beam)' if added else '(filtered out)'}")
        
        print("\n" + "=" * 40)
        print("Beam Search Results:")
        
        # Show beam status
        status = searcher.get_beam_status()
        print(f"Words tested: {status['tested_count']}")
        print(f"Beam size: {status['beam_size']}/{status['beam_width']}")
        print(f"Best candidate: {format_hebrew_output(status['best_word'])} ({status['best_similarity']:.2f})")
        
        print("\nTop candidates in beam:")
        top_candidates = searcher.get_top_candidates()
        for i, candidate in enumerate(top_candidates, 1):
            print(f"  {i}. {format_hebrew_output(candidate.word)}: {candidate.similarity:.2f}")
        
        # Test duplicate detection
        print("\n" + "=" * 40)
        print("Testing duplicate detection:")
        duplicate_added = searcher.add_candidate("שלום", 30.0)  # Already tested
        print(f"Duplicate word '{format_hebrew_output('שלום')}' added: {duplicate_added}")
        
        print("=" * 40)
        print("Beam search test completed")
        
    except Exception as e:
        logger.error(f"Error during beam search test: {e}")


if __name__ == "__main__":
    main()
