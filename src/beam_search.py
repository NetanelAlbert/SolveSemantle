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
    """Beam search system for efficient word exploration in Hebrew Semantle"""
    
    def __init__(self, beam_width: int = 5):
        """
        Initialize beam search system
        
        Args:
            beam_width: Maximum number of candidates to maintain in beam (default: 5)
        """
        self.beam_width = max(1, beam_width)  # Ensure at least 1 candidate
        self.tested_words: Set[str] = set()
        self.current_beam: List[WordCandidate] = []
        self.best_candidate: Optional[WordCandidate] = None
        
        logger.info(f"Initialized BeamSearcher with beam width: {self.beam_width}")
    
    def add_candidate(self, word: str, similarity: float) -> bool:
        """
        Add a new word candidate to the beam search
        
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
            
            # Update best candidate
            if self.best_candidate is None or similarity > self.best_candidate.similarity:
                self.best_candidate = candidate
                logger.info(f"New best candidate: {format_hebrew_output(word)} (similarity: {similarity:.2f})")
            
            # Add to beam if there's space or if better than worst in beam
            if len(self.current_beam) < self.beam_width:
                self.current_beam.append(candidate)
                logger.debug(f"Added candidate to beam: {word} (similarity: {similarity:.2f})")
                return True
            else:
                # Find worst candidate in beam
                worst_idx = min(range(len(self.current_beam)), 
                              key=lambda i: self.current_beam[i].similarity)
                worst_candidate = self.current_beam[worst_idx]
                
                # Replace if current candidate is better
                if similarity > worst_candidate.similarity:
                    self.current_beam[worst_idx] = candidate
                    logger.debug(f"Replaced candidate in beam: {worst_candidate.word} -> {word}")
                    return True
                else:
                    logger.debug(f"Candidate rejected (below beam threshold): {word}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error adding candidate '{word}': {e}")
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
        Get current beam search status and statistics
        
        Returns:
            Dictionary with beam statistics
        """
        return {
            'tested_count': len(self.tested_words),
            'beam_size': len(self.current_beam),
            'beam_width': self.beam_width,
            'best_similarity': self.best_candidate.similarity if self.best_candidate else 0.0,
            'best_word': self.best_candidate.word if self.best_candidate else None
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
