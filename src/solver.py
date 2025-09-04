"""
Hebrew Semantle Solver

Main solving algorithm that uses beam search and API client to solve
Hebrew Semantle puzzles with 5-minute timeout and intelligent exploration.
"""

import time
import logging
from typing import List, Optional, Dict, Any

try:
    # Try relative imports first (when used as module)
    from .api_client import SemantheAPIClient
    from .beam_search import BeamSearcher, WordCandidate
except ImportError:
    # Fall back to absolute imports (when run as script)
    from api_client import SemantheAPIClient
    from beam_search import BeamSearcher, WordCandidate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemantleSolver:
    """Main solver for Hebrew Semantle puzzles"""
    
    def __init__(self, beam_width: int = 5, timeout_seconds: int = 300):
        """
        Initialize the Semantle solver
        
        Args:
            beam_width: Number of top candidates to maintain in beam search
            timeout_seconds: Maximum solving time in seconds (default: 5 minutes)
        """
        self.beam_width = beam_width
        self.timeout_seconds = timeout_seconds
        self.api_client = SemantheAPIClient()
        self.beam_searcher = BeamSearcher(beam_width=beam_width)
        
        # Rate limiting settings
        self.request_delay = 1.0  # Delay between API calls in seconds
        self.last_request_time = 0.0
        
        # Solving statistics
        self.start_time: Optional[float] = None
        self.total_guesses = 0
        self.solution_found = False
        self.solution_word: Optional[str] = None
        
        logger.info(f"Initialized SemantleSolver with beam_width={beam_width}, timeout={timeout_seconds}s")
    
    def _get_initial_word_list(self) -> List[str]:
        """
        Get list of common Hebrew words to start exploration
        
        Returns:
            List of Hebrew words for initial guessing
        """
        # High-frequency Hebrew words for initial exploration
        return [
            "שלום", "חיים", "אהבה", "בית", "משפחה", "ישראל", "עולם", "יום",
            "לילה", "אור", "חושך", "מים", "אש", "רוח", "אדמה", "שמיים",
            "אדם", "אישה", "ילד", "ילדה", "אב", "אם", "אח", "אחות",
            "עיר", "כפר", "דרך", "רחוב", "בית ספר", "עבודה", "כסף", "זמן"
        ]
    
    def _respect_rate_limit(self):
        """Ensure we don't overwhelm the API with requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _test_word(self, word: str) -> Optional[float]:
        """
        Test a word with rate limiting and error handling
        
        Args:
            word: Hebrew word to test
            
        Returns:
            Similarity score or None if API call fails
        """
        try:
            self._respect_rate_limit()
            similarity = self.api_client.test_word_similarity(word)
            
            if similarity is not None:
                self.total_guesses += 1
                logger.info(f"Guess #{self.total_guesses}: {word} → {similarity:.2f}")
                
                # Check if we found the solution (perfect match)
                if similarity >= 99.99:  # Allow for floating point precision
                    self.solution_found = True
                    self.solution_word = word
                    logger.info(f"🎉 SOLUTION FOUND: {word} (similarity: {similarity:.2f})")
                
                return similarity
            else:
                logger.warning(f"API call failed for word: {word}")
                return None
                
        except Exception as e:
            logger.error(f"Error testing word '{word}': {e}")
            return None
    
    def _has_time_remaining(self) -> bool:
        """Check if we still have time remaining for solving"""
        if self.start_time is None:
            return True
            
        elapsed = time.time() - self.start_time
        return elapsed < self.timeout_seconds
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed solving time in seconds"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def _expand_search_from_candidates(self) -> List[str]:
        """
        Generate new words to explore based on current best candidates
        
        This is a simplified expansion - in a full implementation,
        this would use word embeddings or linguistic patterns
        
        Returns:
            List of new words to explore
        """
        top_candidates = self.beam_searcher.get_top_candidates(count=3)
        expansion_words = []
        
        # For now, use a simple heuristic - could be enhanced with word2vec later
        # This is a placeholder that would be improved in Unit 4 with proper word embeddings
        hebrew_word_variations = [
            "דבר", "מילה", "טוב", "רע", "גדול", "קטן", "חדש", "ישן",
            "לבן", "שחור", "אדום", "ירוק", "כחול", "צהוב", "חם", "קר",
            "מהיר", "איטי", "חזק", "חלש", "יפה", "מכוער", "חכם", "טיפש"
        ]
        
        # Filter out already tested words
        for word in hebrew_word_variations:
            if not self.beam_searcher.is_word_tested(word):
                expansion_words.append(word)
                if len(expansion_words) >= 5:  # Limit expansion size
                    break
        
        return expansion_words
    
    def solve(self) -> Dict[str, Any]:
        """
        Solve the current Hebrew Semantle puzzle
        
        Returns:
            Dictionary with solving results and statistics
        """
        try:
            logger.info("Starting Hebrew Semantle solver...")
            self.start_time = time.time()
            
            # Phase 1: Test initial word list
            initial_words = self._get_initial_word_list()
            logger.info(f"Phase 1: Testing {len(initial_words)} initial words")
            
            for word in initial_words:
                if not self._has_time_remaining():
                    logger.info("Timeout reached during initial word testing")
                    break
                
                if self.beam_searcher.is_word_tested(word):
                    continue
                
                similarity = self._test_word(word)
                if similarity is not None:
                    self.beam_searcher.add_candidate(word, similarity)
                    
                    if self.solution_found:
                        break
            
            # Phase 2: Beam search exploration
            if not self.solution_found and self._has_time_remaining():
                logger.info("Phase 2: Beam search exploration")
                
                exploration_rounds = 0
                max_exploration_rounds = 10  # Prevent infinite loops
                
                while (not self.solution_found and 
                       self._has_time_remaining() and 
                       exploration_rounds < max_exploration_rounds):
                    
                    exploration_rounds += 1
                    logger.info(f"Exploration round {exploration_rounds}")
                    
                    # Get words to explore based on current candidates
                    expansion_words = self._expand_search_from_candidates()
                    
                    if not expansion_words:
                        logger.info("No new words to explore - search exhausted")
                        break
                    
                    # Test expansion words
                    for word in expansion_words:
                        if not self._has_time_remaining():
                            logger.info("Timeout reached during exploration")
                            break
                        
                        similarity = self._test_word(word)
                        if similarity is not None:
                            self.beam_searcher.add_candidate(word, similarity)
                            
                            if self.solution_found:
                                break
                    
                    # Show progress
                    status = self.beam_searcher.get_beam_status()
                    logger.info(f"Progress: {status['tested_count']} words tested, "
                              f"best: {status['best_word']} ({status['best_similarity']:.2f})")
            
            # Compile results
            elapsed_time = self._get_elapsed_time()
            beam_status = self.beam_searcher.get_beam_status()
            
            results = {
                'success': self.solution_found,
                'solution_word': self.solution_word,
                'total_guesses': self.total_guesses,
                'elapsed_time': elapsed_time,
                'timeout_reached': elapsed_time >= self.timeout_seconds,
                'best_candidate': {
                    'word': beam_status['best_word'],
                    'similarity': beam_status['best_similarity']
                },
                'words_tested': beam_status['tested_count'],
                'beam_size': beam_status['beam_size']
            }
            
            # Log final results
            if self.solution_found:
                logger.info(f"🎉 PUZZLE SOLVED! Word: {self.solution_word} in {self.total_guesses} guesses")
                logger.info(f"⏱️  Total time: {elapsed_time:.1f}s")
            else:
                logger.info(f"❌ Puzzle not solved within {self.timeout_seconds}s timeout")
                logger.info(f"🥈 Best candidate: {beam_status['best_word']} ({beam_status['best_similarity']:.2f})")
                logger.info(f"📊 Total guesses: {self.total_guesses}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during solving: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_guesses': self.total_guesses,
                'elapsed_time': self._get_elapsed_time()
            }
        
        finally:
            # Clean up
            self.api_client.close()
    
    def get_solving_status(self) -> Dict[str, Any]:
        """
        Get current solving status and progress
        
        Returns:
            Dictionary with current solving statistics
        """
        beam_status = self.beam_searcher.get_beam_status()
        
        return {
            'elapsed_time': self._get_elapsed_time(),
            'total_guesses': self.total_guesses,
            'solution_found': self.solution_found,
            'solution_word': self.solution_word,
            'time_remaining': max(0, self.timeout_seconds - self._get_elapsed_time()),
            'best_candidate': {
                'word': beam_status['best_word'],
                'similarity': beam_status['best_similarity']
            },
            'words_tested': beam_status['tested_count']
        }


def main():
    """Test the complete solver with Hebrew Semantle"""
    print("Hebrew Semantle Solver")
    print("=" * 50)
    print("🎯 Starting puzzle solver...")
    print("⏱️  Timeout: 5 minutes")
    print("🔍 Strategy: Beam search with common Hebrew words")
    print("=" * 50)
    
    try:
        # Create solver with default settings
        solver = SemantleSolver(beam_width=5, timeout_seconds=300)
        
        # Solve the puzzle
        results = solver.solve()
        
        # Display results
        print("\n" + "=" * 50)
        print("FINAL RESULTS")
        print("=" * 50)
        
        if results['success']:
            print(f"🎉 SUCCESS! Solution found: {results['solution_word']}")
            print(f"📊 Total guesses: {results['total_guesses']}")
            print(f"⏱️  Time taken: {results['elapsed_time']:.1f} seconds")
        else:
            print(f"❌ Puzzle not solved")
            if results.get('timeout_reached'):
                print("⏰ Reason: 5-minute timeout reached")
            
            best = results.get('best_candidate', {})
            if best.get('word'):
                print(f"🥈 Best candidate: {best['word']} (similarity: {best['similarity']:.2f})")
            
            print(f"📊 Total guesses: {results['total_guesses']}")
            print(f"⏱️  Time elapsed: {results['elapsed_time']:.1f} seconds")
        
        print(f"🔍 Words tested: {results['words_tested']}")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Solver interrupted by user")
    except Exception as e:
        logger.error(f"Solver failed: {e}")
        print(f"❌ Solver failed: {e}")


if __name__ == "__main__":
    main()
