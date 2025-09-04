"""
Hebrew Semantle Core Solver

Main solving algorithm that combines API client and beam search
to solve Hebrew Semantle puzzles within a 5-minute time limit.
"""

import time
import logging
from typing import List, Optional, Dict, Any
import random

from .api_client import SemantheAPIClient
from .beam_search import BeamSearcher, WordCandidate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemantleSolver:
    """Core solver for Hebrew Semantle puzzles using beam search algorithm"""
    
    # Common Hebrew words to start exploration
    INITIAL_HEBREW_WORDS = [
        # High frequency Hebrew words
        "שלום", "אהבה", "חיים", "בית", "משפחה", "חברים", "עבודה", "זמן",
        "דרך", "יום", "לילה", "אור", "חושך", "שמח", "עצוב", "טוב",
        "רע", "גדול", "קטן", "חדש", "ישן", "יפה", "מים", "אש",
        "רוח", "אדמה", "שמים", "ירח", "שמש", "כוכבים", "ים", "הר",
        "עיר", "כפר", "דלת", "חלון", "שולחן", "כסא", "ספר", "עט",
        "נייר", "מחשב", "טלפון", "רכב", "אוטובוס", "רכבת", "מטוס"
    ]
    
    def __init__(self, beam_width: int = 5, timeout_minutes: int = 5, 
                 rate_limit_seconds: float = 2.0):
        """
        Initialize Semantle solver
        
        Args:
            beam_width: Number of top candidates to maintain in beam search
            timeout_minutes: Maximum solving time in minutes (default: 5)
            rate_limit_seconds: Seconds to wait between API calls (default: 2.0)
        """
        self.api_client = SemantheAPIClient()
        self.beam_searcher = BeamSearcher(beam_width=beam_width)
        self.timeout_seconds = timeout_minutes * 60
        self.rate_limit_seconds = rate_limit_seconds
        
        # Rate limiting and backoff state
        self.consecutive_rate_limit_errors = 0
        self.current_backoff_delay = rate_limit_seconds
        self.max_backoff_delay = 30.0  # Maximum 30 second delay
        
        # Solving state
        self.start_time = None
        self.solve_complete = False
        self.winning_word = None
        self.total_guesses = 0
        
        logger.info(f"Initialized SemantleSolver: beam_width={beam_width}, "
                   f"timeout={timeout_minutes}min, rate_limit={rate_limit_seconds}s")
    
    def is_timeout_reached(self) -> bool:
        """Check if the 5-minute timeout has been reached"""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) >= self.timeout_seconds
    
    def get_remaining_time(self) -> float:
        """Get remaining time in seconds"""
        if self.start_time is None:
            return self.timeout_seconds
        elapsed = time.time() - self.start_time
        return max(0, self.timeout_seconds - elapsed)
    
    def handle_rate_limit_success(self):
        """Reset rate limiting state after successful API call"""
        self.consecutive_rate_limit_errors = 0
        self.current_backoff_delay = self.rate_limit_seconds
    
    def handle_rate_limit_error(self):
        """Handle rate limit error with exponential backoff"""
        self.consecutive_rate_limit_errors += 1
        # Exponential backoff: double the delay each time, up to max
        self.current_backoff_delay = min(
            self.current_backoff_delay * 2, 
            self.max_backoff_delay
        )
        
        logger.warning(f"Rate limit hit! Consecutive errors: {self.consecutive_rate_limit_errors}, "
                      f"next delay: {self.current_backoff_delay:.1f}s")
        
        # Wait with the current backoff delay
        time.sleep(self.current_backoff_delay)
    
    def get_current_delay(self) -> float:
        """Get current delay to use for rate limiting"""
        return self.current_backoff_delay
    
    def test_word(self, word: str) -> Optional[float]:
        """
        Test a word and add it to beam search if successful
        
        Args:
            word: Hebrew word to test
            
        Returns:
            Similarity score if successful, None if failed
        """
        max_retries = 3
        
        for attempt in range(max_retries + 1):
            try:
                similarity = self.api_client.test_word_similarity(word)
                
                if similarity is not None:
                    # Success! Reset rate limiting state
                    self.handle_rate_limit_success()
                    
                    self.total_guesses += 1
                    self.beam_searcher.add_candidate(word, similarity)
                    
                    # Check for winning condition
                    if similarity >= 99.9:  # Allow for small floating point errors
                        logger.info(f"🎉 FOUND WINNING WORD: {word} (similarity: {similarity:.2f})")
                        self.winning_word = word
                        self.solve_complete = True
                        return similarity
                    
                    logger.info(f"Tested {word}: {similarity:.2f}% "
                              f"(#{self.total_guesses}, {self.get_remaining_time():.0f}s left)")
                    return similarity
                else:
                    # API returned None - check if it's a rate limit issue
                    # For now, treat as failure and return None
                    if attempt == max_retries:
                        logger.warning(f"Failed to test word after {max_retries + 1} attempts: {word}")
                    return None
                    
            except Exception as e:
                error_str = str(e)
                
                # Check if this is a rate limit error (429)
                if "429" in error_str or "Too Many Requests" in error_str:
                    if attempt < max_retries:
                        logger.warning(f"Rate limit hit for word '{word}', attempt {attempt + 1}/{max_retries + 1}")
                        self.handle_rate_limit_error()
                        
                        # Check if we still have time to continue
                        if self.is_timeout_reached():
                            logger.warning(f"Timeout reached during rate limit backoff for word: {word}")
                            return None
                        continue
                    else:
                        logger.error(f"Rate limit exceeded for word '{word}' after {max_retries + 1} attempts")
                        return None
                else:
                    # Other error - don't retry
                    logger.error(f"Error testing word '{word}': {e}")
                    return None
        
        return None
    
    def generate_exploration_words(self, count: int = 10) -> List[str]:
        """
        Generate words for exploration based on current beam search state
        
        Args:
            count: Number of words to generate for exploration
            
        Returns:
            List of Hebrew words to test next
        """
        exploration_words = []
        
        try:
            # Get top candidates from beam
            top_candidates = self.beam_searcher.get_top_candidates()
            
            if not top_candidates:
                # No beam candidates yet, use initial words
                available_initial = [w for w in self.INITIAL_HEBREW_WORDS 
                                   if not self.beam_searcher.is_word_tested(w)]
                exploration_words.extend(random.sample(
                    available_initial, 
                    min(count, len(available_initial))
                ))
            else:
                # Generate variations based on top candidates
                # For now, use a simple strategy: use untested initial words
                # This can be enhanced with word similarity models later
                available_initial = [w for w in self.INITIAL_HEBREW_WORDS 
                                   if not self.beam_searcher.is_word_tested(w)]
                exploration_words.extend(random.sample(
                    available_initial, 
                    min(count, len(available_initial))
                ))
            
            logger.debug(f"Generated {len(exploration_words)} exploration words")
            return exploration_words
            
        except Exception as e:
            logger.error(f"Error generating exploration words: {e}")
            return []
    
    def solve(self) -> Dict[str, Any]:
        """
        Main solving algorithm with 5-minute timeout
        
        Returns:
            Dictionary with solving results and statistics
        """
        logger.info("🚀 Starting Hebrew Semantle solver...")
        self.start_time = time.time()
        
        try:
            # Initial exploration with common Hebrew words
            initial_batch_size = min(10, len(self.INITIAL_HEBREW_WORDS))
            initial_words = random.sample(self.INITIAL_HEBREW_WORDS, initial_batch_size)
            
            logger.info(f"Testing initial batch of {len(initial_words)} common Hebrew words...")
            for word in initial_words:
                if self.solve_complete or self.is_timeout_reached():
                    break
                
                self.test_word(word)
                # Use current backoff delay for rate limiting
                time.sleep(self.get_current_delay())
            
            # Main exploration loop
            while not self.solve_complete and not self.is_timeout_reached():
                # Generate next batch of exploration words
                exploration_words = self.generate_exploration_words(count=5)
                
                if not exploration_words:
                    logger.warning("No more words to explore, stopping solver")
                    break
                
                # Test exploration words
                for word in exploration_words:
                    if self.solve_complete or self.is_timeout_reached():
                        break
                    
                    if not self.beam_searcher.is_word_tested(word):
                        self.test_word(word)
                        # Use current backoff delay for rate limiting
                        time.sleep(self.get_current_delay())
                
                # Show progress every few iterations
                status = self.beam_searcher.get_beam_status()
                logger.info(f"Progress: {status['tested_count']} words tested, "
                          f"best: {status['best_word']} ({status['best_similarity']:.2f}%), "
                          f"time remaining: {self.get_remaining_time():.0f}s")
            
        except KeyboardInterrupt:
            logger.info("🛑 Solving interrupted by user")
        except Exception as e:
            logger.error(f"Error during solving: {e}")
        finally:
            # Calculate final statistics
            end_time = time.time()
            total_time = end_time - self.start_time
            
            status = self.beam_searcher.get_beam_status()
            
            result = {
                'success': self.solve_complete,
                'winning_word': self.winning_word,
                'total_time_seconds': total_time,
                'total_guesses': self.total_guesses,
                'timeout_reached': self.is_timeout_reached(),
                'best_word': status['best_word'],
                'best_similarity': status['best_similarity'],
                'words_tested': status['tested_count']
            }
            
            # Log final results
            if self.solve_complete:
                logger.info(f"🎉 SUCCESS! Found winning word: {self.winning_word}")
                logger.info(f"   Time: {total_time:.1f}s, Guesses: {self.total_guesses}")
            elif self.is_timeout_reached():
                logger.info(f"⏰ TIMEOUT after {total_time:.1f}s")
                logger.info(f"   Best word: {status['best_word']} ({status['best_similarity']:.2f}%)")
                logger.info(f"   Total guesses: {self.total_guesses}")
            else:
                logger.info(f"🛑 STOPPED after {total_time:.1f}s")
                logger.info(f"   Best word: {status['best_word']} ({status['best_similarity']:.2f}%)")
                logger.info(f"   Total guesses: {self.total_guesses}")
            
            return result
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.api_client.close()
            logger.info("Solver cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point for the Hebrew Semantle solver"""
    print("🔤 Hebrew Semantle Solver")
    print("=" * 50)
    print("Starting puzzle solver with 5-minute timeout...")
    print("Press Ctrl+C to stop early")
    print("=" * 50)
    
    solver = SemantleSolver()
    
    try:
        result = solver.solve()
        
        print("\n" + "=" * 50)
        print("📊 FINAL RESULTS")
        print("=" * 50)
        
        if result['success']:
            print(f"✅ SUCCESS! Word found: {result['winning_word']}")
            print(f"🕐 Time taken: {result['total_time_seconds']:.1f} seconds")
            print(f"🎯 Total guesses: {result['total_guesses']}")
        else:
            if result['timeout_reached']:
                print(f"⏰ Timeout reached (5 minutes)")
            else:
                print(f"🛑 Stopped early")
            
            print(f"🎯 Total guesses: {result['total_guesses']}")
            print(f"🏆 Best word found: {result['best_word']} ({result['best_similarity']:.2f}%)")
            print(f"🕐 Time elapsed: {result['total_time_seconds']:.1f} seconds")
        
        print(f"📈 Words tested: {result['words_tested']}")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"\n❌ Fatal error: {e}")
    finally:
        solver.cleanup()


if __name__ == "__main__":
    main()
