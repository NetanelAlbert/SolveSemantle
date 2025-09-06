"""
Hebrew Semantle Solver

Main solving algorithm that uses beam search and API client to solve
Hebrew Semantle puzzles with 5-minute timeout and intelligent exploration.
"""

import time
import logging
from typing import List, Optional, Dict, Any, Tuple

try:
    # Try relative imports first (when used as module)
    from .api_client import SemantheAPIClient
    from .beam_search import BeamSearcher, WordCandidate
    from .language_model import HebrewLanguageModel
    from .hebrew_utils import format_hebrew_output
    from .learning_system import ContextualLearningSystem
    from .optimization import SmartTimeoutManager, SemanticGradientOptimizer, EmergencyStrategyManager, ParallelOptimizer
    from .monitoring import ComprehensiveMonitor
except ImportError:
    # Fall back to absolute imports (when run as script)
    from api_client import SemantheAPIClient
    from beam_search import BeamSearcher, WordCandidate
    from language_model import HebrewLanguageModel
    from hebrew_utils import format_hebrew_output
    from learning_system import ContextualLearningSystem
    from optimization import SmartTimeoutManager, SemanticGradientOptimizer, EmergencyStrategyManager, ParallelOptimizer
    from monitoring import ComprehensiveMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemantleSolver:
    """Main solver for Hebrew Semantle puzzles with comprehensive monitoring"""
    
    def __init__(self, beam_width: int = 5, timeout_seconds: int = 300, use_language_model: bool = True, enable_learning: bool = True, enable_optimization: bool = True, enable_monitoring: bool = True):
        """
        Initialize the Semantle solver with all enhancements
        
        Args:
            beam_width: Number of top candidates to maintain in beam search
            timeout_seconds: Maximum solving time in seconds (default: 5 minutes)
            use_language_model: Whether to use Word2Vec model for intelligent exploration
            enable_learning: Whether to enable contextual learning system
            enable_optimization: Whether to enable advanced optimization techniques
            enable_monitoring: Whether to enable comprehensive monitoring and analysis
        """
        self.beam_width = beam_width
        self.timeout_seconds = timeout_seconds
        self.use_language_model = use_language_model
        self.enable_learning = enable_learning
        self.enable_optimization = enable_optimization
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self.api_client = SemantheAPIClient()
        # Enhanced beam searcher with dynamic width adjustment
        self.beam_searcher = BeamSearcher(
            beam_width=beam_width,
            min_beam_width=max(2, beam_width - 3),  # Adaptive minimum
            max_beam_width=min(12, beam_width + 5)   # Adaptive maximum
        )
        
        # Initialize contextual learning system
        if self.enable_learning:
            self.learning_system = ContextualLearningSystem()
        else:
            self.learning_system = None
        
        # Initialize comprehensive monitoring system
        if self.enable_monitoring:
            self.monitor = ComprehensiveMonitor()
        else:
            self.monitor = None
        
        # Initialize advanced optimization components
        if self.enable_optimization:
            self.smart_timeout = SmartTimeoutManager(base_timeout=timeout_seconds, progress_threshold=5.0)
            self.emergency_manager = EmergencyStrategyManager()
            self.parallel_optimizer = ParallelOptimizer(max_workers=1, rate_limit_delay=1.0)  # Respect API limits
            logger.info("Advanced optimization techniques enabled")
        else:
            self.smart_timeout = None
            self.emergency_manager = None
            self.parallel_optimizer = None
        
        # Initialize language model if requested
        self.language_model = None
        if use_language_model:
            try:
                self.language_model = HebrewLanguageModel()
                if self.language_model.load_model():
                    logger.info("Hebrew Word2Vec model loaded successfully")
                    # Initialize gradient optimizer if language model is available
                    if self.enable_optimization:
                        self.gradient_optimizer = SemanticGradientOptimizer(self.language_model)
                    else:
                        self.gradient_optimizer = None
                else:
                    logger.warning("Failed to load Word2Vec model. Using fallback exploration.")
                    self.language_model = None
                    self.gradient_optimizer = None
            except Exception as e:
                logger.warning(f"Language model initialization failed: {e}. Using fallback exploration.")
                self.language_model = None
                self.gradient_optimizer = None
        else:
            self.gradient_optimizer = None
        
        # Rate limiting settings
        self.request_delay = 1.0  # Delay between API calls in seconds
        self.last_request_time = 0.0
        
        # Solving statistics
        self.start_time: Optional[float] = None
        self.total_guesses = 0
        self.solution_found = False
        self.solution_word: Optional[str] = None
        
        strategy_parts = []
        if self.language_model:
            strategy_parts.append("Word2Vec")
        strategy_parts.append("Enhanced Beam Search") 
        if self.enable_learning:
            strategy_parts.append("Learning")
        if self.enable_optimization:
            strategy_parts.append("Advanced Optimization")
        if self.enable_monitoring:
            strategy_parts.append("Comprehensive Monitoring")
        
        strategy = " + ".join(strategy_parts)
        logger.info(f"Initialized SemantleSolver with {strategy}, beam_width={beam_width}, timeout={timeout_seconds}s")
    
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
    
    def _test_word(self, word: str, parent_candidate: Optional[str] = None, parent_similarity: float = 0.0, strategy_used: str = 'unknown') -> Optional[float]:
        """
        Test a single word and record learning patterns and monitoring data
        
        Args:
            word: Hebrew word to test
            parent_candidate: The candidate word that suggested this word (for learning)
            parent_similarity: Similarity score of the parent candidate
            strategy_used: Strategy that generated this word suggestion
            
        Returns:
            Similarity score if successful, None if failed
        """
        try:
            logger.debug(f"Testing word: {word}")
            
            # Apply rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.request_delay:
                time.sleep(self.request_delay - time_since_last)
            
            # Make API request
            similarity = self.api_client.test_word_similarity(word)
            self.last_request_time = time.time()
            self.total_guesses += 1
            
            if similarity is not None:
                logger.info(f"Guess #{self.total_guesses}: {format_hebrew_output(word)} → {similarity:.2f}")
                
                # Log to monitoring system
                if self.monitor and self.enable_monitoring:
                    self.monitor.log_word_test(word, similarity, strategy_used, parent_similarity)
                
                # Record learning pattern if learning is enabled
                if self.learning_system and parent_candidate:
                    search_phase = self._determine_search_phase()
                    self.learning_system.record_search_pattern(
                        candidate_word=parent_candidate,
                        candidate_similarity=parent_similarity,
                        suggested_word=word,
                        suggestion_similarity=similarity,
                        search_phase=search_phase,
                        strategy_used=strategy_used
                    )
                
                # Check if solution found (similarity 100)
                if similarity >= 100.0:
                    self.solution_found = True
                    self.solution_word = word
                    logger.info(f"🎉 SOLUTION FOUND: {format_hebrew_output(word)}")
                
                return similarity
            else:
                logger.warning(f"API call failed for word: {word}")
                return None
                
        except Exception as e:
            logger.error(f"Error testing word '{word}': {e}")
            return None
    
    def _has_time_remaining(self) -> bool:
        """Check if there's time remaining with smart timeout management"""
        if self.smart_timeout and self.enable_optimization:
            return self.smart_timeout.should_continue()
        else:
            # Fallback to basic timeout check
            if self.start_time is None:
                return True
            elapsed = time.time() - self.start_time
            return elapsed < self.timeout_seconds
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed solving time in seconds"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def _create_prioritized_word_queue(self, word_tuples: List[Tuple[str, float]]) -> List[str]:
        """
        Create a prioritized word queue where words are ordered by parent similarity
        
        Words suggested by higher-scoring candidates are tested first.
        Uses parent similarity as primary sort key, alphabetical as secondary.
        
        Args:
            word_tuples: List of (word, parent_similarity) tuples
            
        Returns:
            List of words sorted by priority (highest parent similarity first)
        """
        if not word_tuples:
            return []
        
        # Sort by parent similarity (descending), then by word (ascending) for stability
        prioritized_words = sorted(
            word_tuples,
            key=lambda x: (-x[1], x[0])  # Negative similarity for descending, word for ascending
        )
        
        # Extract just the words for return
        word_list = [word for word, _ in prioritized_words]
        
        logger.debug(f"Prioritized {len(word_list)} words by parent similarity")
        if prioritized_words:
            best_parent_sim = prioritized_words[0][1]
            worst_parent_sim = prioritized_words[-1][1] 
            logger.debug(f"Parent similarity range: {worst_parent_sim:.2f} to {best_parent_sim:.2f}")
        
        return word_list

    def _determine_search_phase(self) -> str:
        """
        Determine current search phase based on progress and beam status
        
        Returns:
            Search phase: 'exploration', 'exploitation', or 'convergence'
        """
        if not self.beam_searcher.best_candidate:
            return 'exploration'
        
        best_similarity = self.beam_searcher.best_candidate.similarity
        beam_status = self.beam_searcher.get_beam_status()
        
        # Convergence phase: High similarity scores (approaching solution)
        if best_similarity >= 80.0:
            return 'convergence'
        
        # Exploitation phase: Good progress, focus on promising areas
        elif best_similarity >= 60.0 and beam_status['recent_improvements'] >= 2:
            return 'exploitation'
        
        # Exploration phase: Early stages or when stuck
        else:
            return 'exploration'

    def _expand_search_from_candidates(self) -> List[str]:
        """
        Generate new words to explore with advanced optimization techniques
        
        Uses multi-strategy word generation with morphological patterns, semantic clustering,
        frequency-based prioritization, adaptive strategy weighting, contextual learning,
        semantic gradient ascent, and emergency strategies.
        
        Returns:
            List of words to explore, optimized for maximum effectiveness
        """
        top_candidates = self.beam_searcher.get_top_candidates(count=3)
        word_tuples = []
        
        # Check for emergency strategy activation
        beam_status = self.beam_searcher.get_beam_status()
        if (self.emergency_manager and self.enable_optimization and 
            self.emergency_manager.should_activate_emergency(beam_status['tested_count'], beam_status['best_similarity'])):
            
            emergency_words = self.emergency_manager.get_emergency_words(
                self.language_model, 
                self.beam_searcher.tested_words, 
                count=8
            )
            if emergency_words:
                logger.info(f"Using emergency strategy with {len(emergency_words)} words")
                self._word_generation_metadata = {}  # No metadata for emergency words
                return emergency_words
        
        # Strategy 1: Use enhanced multi-strategy word generation with learning and optimization
        if self.language_model and top_candidates:
            logger.debug("Using multi-strategy word generation with optimization and learning")
            
            # Determine current search phase
            search_phase = self._determine_search_phase()
            
            # Update puzzle classification in learning system
            if self.learning_system:
                self.learning_system.current_puzzle_type = self.learning_system.classify_puzzle_type(beam_status)
            
            # Get candidate words for the language model
            candidate_words = [candidate.word for candidate in top_candidates]
            tested_words = self.beam_searcher.tested_words.copy()
            
            # Use multi-strategy word generation with adaptive weighting
            if self.language_model.current_search_phase != search_phase:
                self.language_model.current_search_phase = search_phase
            
            suggested_words = self.language_model.get_multi_strategy_word_suggestions(
                current_candidates=candidate_words,
                tested_words=tested_words,
                search_phase=search_phase,
                count=10  # Generate more words for better prioritization
            )
            
            # Add semantic gradient optimization words
            gradient_words = []
            if self.gradient_optimizer and self.enable_optimization:
                gradient_words = self.gradient_optimizer.find_semantic_peak(
                    candidate_words, 
                    tested_words
                )
                logger.debug(f"Gradient optimizer found {len(gradient_words)} peak candidates")
            
            # Get learned suggestions from historical patterns
            learned_suggestions = []
            if self.learning_system:
                learned_suggestions = self.learning_system.get_learned_suggestions(
                    current_candidates=candidate_words,
                    search_phase=search_phase,
                    count=3
                )
            
            # Track parent similarity for priority-based ordering
            for candidate in top_candidates:
                # Get suggestions specific to this candidate for parent tracking
                single_suggestions = self.language_model.get_word_suggestions(
                    [candidate.word], count=3
                )
                
                for word in single_suggestions:
                    if not self.beam_searcher.is_word_tested(word):
                        # Apply learning-based weight adjustment
                        base_similarity = candidate.similarity
                        if self.learning_system:
                            adaptive_weight = self.learning_system.get_adaptive_word_weight(word)
                            adjusted_similarity = base_similarity * adaptive_weight
                        else:
                            adjusted_similarity = base_similarity
                        
                        word_tuples.append((word, adjusted_similarity, candidate.word, 'semantic'))
                        if len(word_tuples) >= 8:
                            break
                
                if len(word_tuples) >= 8:
                    break
            
            # Add multi-strategy words with composite parent similarity
            for word in suggested_words:
                if not self.beam_searcher.is_word_tested(word) and word not in [w for w, _, _, _ in word_tuples]:
                    # Calculate average parent similarity for multi-strategy words
                    avg_parent_similarity = sum(c.similarity for c in top_candidates) / len(top_candidates)
                    
                    # Apply learning-based weight adjustment
                    if self.learning_system:
                        adaptive_weight = self.learning_system.get_adaptive_word_weight(word)
                        adjusted_similarity = avg_parent_similarity * 0.9 * adaptive_weight  # Slight penalty for mixed strategy
                    else:
                        adjusted_similarity = avg_parent_similarity * 0.9
                    
                    # Use average parent as representative
                    avg_parent_word = top_candidates[0].word if top_candidates else 'unknown'
                    word_tuples.append((word, adjusted_similarity, avg_parent_word, 'multi-strategy'))
                    if len(word_tuples) >= 12:
                        break
            
            # Add gradient-optimized words with high priority
            for gradient_word in gradient_words:
                if (not self.beam_searcher.is_word_tested(gradient_word) and 
                    gradient_word not in [w for w, _, _, _ in word_tuples]):
                    # Gradient words get high priority boost
                    best_similarity = top_candidates[0].similarity if top_candidates else 50.0
                    gradient_similarity = best_similarity * 1.1  # 10% boost for gradient words
                    best_parent = top_candidates[0].word if top_candidates else 'gradient'
                    word_tuples.append((gradient_word, gradient_similarity, best_parent, 'gradient'))
            
            # Add learned suggestions with high priority
            for learned_word, relevance in learned_suggestions:
                if (not self.beam_searcher.is_word_tested(learned_word) and 
                    learned_word not in [w for w, _, _, _ in word_tuples]):
                    # Learned suggestions get high priority
                    learned_similarity = relevance * 100.0  # Scale relevance to similarity range
                    best_parent = top_candidates[0].word if top_candidates else 'learned'
                    word_tuples.append((learned_word, learned_similarity, best_parent, 'learned'))
            
            # If we got good suggestions, optimize and return them
            if word_tuples:
                # Convert tuples to (word, similarity) format for prioritization
                prioritization_tuples = [(word, similarity) for word, similarity, _, _ in word_tuples]
                
                # Use advanced optimization for word ordering
                if self.enable_optimization and self.parallel_optimizer:
                    words = [word for word, _ in prioritization_tuples]
                    priorities = [similarity for _, similarity in prioritization_tuples]
                    optimized_words = self.parallel_optimizer.optimize_word_order(words, priorities)
                else:
                    prioritized_words = self._create_prioritized_word_queue(prioritization_tuples)
                    optimized_words = prioritized_words
                
                # Create enhanced word list with metadata for learning
                word_metadata = {word: (parent, strategy) for word, _, parent, strategy in word_tuples}
                
                # Store metadata for learning integration
                self._word_generation_metadata = word_metadata
                
                logger.info(f"Generated {len(optimized_words)} optimized suggestions "
                           f"with priority ordering (phase: {search_phase})")
                return optimized_words
        
        # Strategy 2: Fallback to basic expansion (expanded word list)
        logger.debug("Using fallback word expansion strategy")
        hebrew_word_variations = [
            # Basic words
            "דבר", "מילה", "טוב", "רע", "גדול", "קטן", "חדש", "ישן",
            "לבן", "שחור", "אדום", "ירוק", "כחול", "צהוב", "חם", "קר",
            "מהיר", "איטי", "חזק", "חלש", "יפה", "מכוער", "חכם", "טיפש",
            "אוכל", "מים", "ארץ", "שמש", "ירח", "כוכב", "עץ", "פרח",
            "ספר", "כתיבה", "קריאה", "למידה", "חכמה", "ידע", "מדע", "אמת",
            # Extended vocabulary for better exploration
            "רגש", "תחושה", "מחשבה", "רעיון", "חלום", "מציאות", "זמן", "מקום",
            "דרך", "מסע", "יעד", "תקווה", "פחד", "שמחה", "עצבות", "כעס",
            "אהבה", "שנאה", "חברות", "משפחה", "קהילה", "חברה", "אנושות", "עולם",
            "שמים", "ים", "הר", "עמק", "יער", "מדבר", "עיר", "כפר",
            "בית", "חדר", "מטבח", "חלון", "דלת", "גינה", "רחוב", "שכונה",
            "לב", "ראש", "עיניים", "אוזניים", "פה", "ידיים", "רגליים", "גוף",
            "נפש", "רוח", "נשמה", "לב", "מוח", "זיכרון", "דמיון", "יצירה",
            "אמנות", "מוסיקה", "ציור", "שיר", "סיפור", "משל", "חידה", "תשובה"
        ]
        
        # Filter out already tested words and assign default similarity of 0.0
        word_tuples = []
        for word in hebrew_word_variations:
            if not self.beam_searcher.is_word_tested(word):
                word_tuples.append((word, 0.0))  # Default parent similarity for fallback words
                if len(word_tuples) >= 12:  # Increased expansion size for better coverage
                    break
        
        # Store empty metadata for fallback
        self._word_generation_metadata = {}
        
        # Prioritize even fallback words (though they all have same priority)
        return self._create_prioritized_word_queue(word_tuples)
    
    def solve(self) -> Dict[str, Any]:
        """
        Solve Hebrew Semantle puzzle with comprehensive monitoring and all enhancements
        
        Returns:
            Dictionary with solving results and detailed analytics
        """
        try:
            logger.info("Starting Hebrew Semantle solver with comprehensive monitoring...")
            self.start_time = time.time()
            
            # Start monitoring session
            session_config = {
                "beam_width": self.beam_width,
                "timeout_seconds": self.timeout_seconds,
                "use_language_model": self.use_language_model,
                "enable_learning": self.enable_learning,
                "enable_optimization": self.enable_optimization,
                "enable_monitoring": self.enable_monitoring
            }
            
            if self.monitor and self.enable_monitoring:
                session_id = self.monitor.start_session(session_config)
                self.monitor.log_phase_start("initialization")
            
            # Start smart timeout manager
            if self.smart_timeout and self.enable_optimization:
                self.smart_timeout.start_timing()
            
            # Phase 1: Test initial word list
            if self.monitor and self.enable_monitoring:
                self.monitor.log_phase_start("initial_word_testing")
            
            initial_words = self._get_initial_word_list()
            logger.info(f"Phase 1: Testing {len(initial_words)} initial words")
            
            for word in initial_words:
                if not self._has_time_remaining():
                    logger.info("Timeout reached during initial word testing")
                    break
                
                if self.beam_searcher.is_word_tested(word):
                    continue
                
                similarity = self._test_word(word, strategy_used='initial')
                if similarity is not None:
                    self.beam_searcher.add_candidate(word, similarity)
                    
                    # Update smart timeout with progress
                    if self.smart_timeout and self.enable_optimization:
                        extended = self.smart_timeout.update_progress(similarity)
                        if extended and self.monitor:
                            self.monitor.log_timeout_extension(
                                self.smart_timeout.timeout_extensions, 
                                "initial_progress", 
                                60.0
                            )
                    
                    if self.solution_found:
                        break
            
            # Phase 2: Advanced beam search exploration
            if not self.solution_found and self._has_time_remaining():
                if self.monitor and self.enable_monitoring:
                    self.monitor.log_phase_start("beam_search_exploration")
                
                logger.info("Phase 2: Advanced beam search exploration with optimization")
                
                exploration_rounds = 0
                max_exploration_rounds = 50  # Prevent infinite loops
                consecutive_empty_rounds = 0
                
                while (not self.solution_found and 
                       self._has_time_remaining() and 
                       exploration_rounds < max_exploration_rounds):
                    
                    exploration_rounds += 1
                    logger.info(f"Exploration round {exploration_rounds}")
                    
                    # Get words to explore based on current candidates with optimization
                    expansion_words = self._expand_search_from_candidates()
                    
                    if not expansion_words:
                        consecutive_empty_rounds += 1
                        logger.warning(f"No new words found (attempt {consecutive_empty_rounds}/5)")
                        
                        if consecutive_empty_rounds >= 5:
                            logger.info("Search exhausted after 5 consecutive empty rounds")
                            break
                        else:
                            continue  # Try the next round
                    else:
                        consecutive_empty_rounds = 0  # Reset counter on successful round
                    
                    # Test expansion words with enhanced learning integration and optimization
                    for word in expansion_words:
                        if not self._has_time_remaining():
                            logger.info("Timeout reached during exploration")
                            break
                        
                        # Get word metadata for learning
                        parent_candidate = None
                        parent_similarity = 0.0
                        strategy_used = 'fallback'
                        
                        if hasattr(self, '_word_generation_metadata') and word in self._word_generation_metadata:
                            parent_candidate, strategy_used = self._word_generation_metadata[word]
                            # Find parent similarity
                            for candidate in self.beam_searcher.get_top_candidates(count=3):
                                if candidate.word == parent_candidate:
                                    parent_similarity = candidate.similarity
                                    break
                        
                        similarity = self._test_word(
                            word, 
                            parent_candidate=parent_candidate, 
                            parent_similarity=parent_similarity,
                            strategy_used=strategy_used
                        )
                        if similarity is not None:
                            self.beam_searcher.add_candidate(word, similarity)
                            
                            # Update smart timeout with progress
                            if self.smart_timeout and self.enable_optimization:
                                extended = self.smart_timeout.update_progress(similarity)
                                if extended and self.monitor:
                                    self.monitor.log_timeout_extension(
                                        self.smart_timeout.timeout_extensions, 
                                        f"progress_similarity_{similarity:.1f}", 
                                        60.0
                                    )
                            
                            if self.solution_found:
                                break
                    
                    # Show progress with enhanced beam search metrics
                    status = self.beam_searcher.get_beam_status()
                    best_word_display = format_hebrew_output(status['best_word']) if status['best_word'] else 'None'
                    logger.info(f"Progress: {status['tested_count']} words tested, "
                              f"best: {best_word_display} ({status['best_similarity']:.2f}), "
                              f"beam: {status['beam_size']}/{status['beam_width']}, "
                              f"strategy: {status['strategy']}")
            
            # Compile results with comprehensive analytics
            elapsed_time = self._get_elapsed_time()
            beam_status = self.beam_searcher.get_beam_status()
            
            results = {
                'success': self.solution_found,
                'solution_word': self.solution_word,
                'total_guesses': self.total_guesses,
                'elapsed_time': elapsed_time,
                'timeout_reached': not self._has_time_remaining(),
                'best_candidate': {
                    'word': beam_status['best_word'],
                    'similarity': beam_status['best_similarity']
                },
                'words_tested': beam_status['tested_count'],
                'beam_size': beam_status['beam_size'],
                'beam_width_final': beam_status['beam_width'],
                'beam_width_range': beam_status['beam_width_range'],
                'search_strategy': beam_status['strategy'],
                'strategy_used': f"Word2Vec + Enhanced Beam Search + Learning + Advanced Optimization + Comprehensive Monitoring" if all([self.language_model, self.enable_learning, self.enable_optimization, self.enable_monitoring]) else 
                               f"Word2Vec + Enhanced Beam Search + Learning + Advanced Optimization" if self.language_model and self.enable_learning and self.enable_optimization else 
                               f"Word2Vec + Enhanced Beam Search + Learning" if self.language_model and self.enable_learning else
                               f"Word2Vec + Enhanced Beam Search + Advanced Optimization" if self.language_model and self.enable_optimization else
                               f"Word2Vec + Enhanced Beam Search" if self.language_model else 
                               f"Enhanced Beam Search + Learning + Advanced Optimization + Comprehensive Monitoring" if all([self.enable_learning, self.enable_optimization, self.enable_monitoring]) else
                               f"Enhanced Beam Search + Learning + Advanced Optimization" if self.enable_learning and self.enable_optimization else
                               f"Enhanced Beam Search + Learning" if self.enable_learning else
                               f"Enhanced Beam Search + Advanced Optimization" if self.enable_optimization else "Enhanced Beam Search",
                'language_model_loaded': self.language_model is not None,
                'learning_enabled': self.enable_learning,
                'optimization_enabled': self.enable_optimization,
                'monitoring_enabled': self.enable_monitoring,
                'learning_stats': self.learning_system.get_learning_stats() if self.learning_system else None,
                'optimization_stats': {
                    'smart_timeout_extensions': self.smart_timeout.timeout_extensions if self.smart_timeout else 0,
                    'emergency_activated': self.emergency_manager.emergency_activated if self.emergency_manager else False,
                    'gradient_optimization_used': self.gradient_optimizer is not None
                } if self.enable_optimization else None
            }
            
            # End monitoring session and get comprehensive metrics
            if self.monitor and self.enable_monitoring:
                session_metrics = self.monitor.end_session(results)
                results['session_metrics'] = {
                    'session_id': session_metrics.session_id,
                    'convergence_rate': session_metrics.convergence_rate,
                    'api_calls_per_similarity_bracket': session_metrics.api_calls_per_similarity_bracket,
                    'time_distribution': session_metrics.time_distribution,
                    'milestone_achievements': list(self.monitor.achieved_milestones)
                }
            
            # Log final results with comprehensive info
            if self.solution_found:
                logger.info(f"🎉 PUZZLE SOLVED! Word: {format_hebrew_output(self.solution_word)} in {self.total_guesses} guesses")
                logger.info(f"⏱️  Total time: {elapsed_time:.1f}s")
            else:
                total_time = time.time() - self.start_time
                exploration_rounds = locals().get('exploration_rounds', 0)
                logger.info(f"❌ Puzzle not solved within {total_time:.1f}s and {exploration_rounds} exploration rounds")
                best_word_display = format_hebrew_output(beam_status['best_word']) if beam_status['best_word'] else 'None'
                logger.info(f"🥈 Best candidate: {best_word_display} ({beam_status['best_similarity']:.2f})")
                logger.info(f"📊 Total guesses: {self.total_guesses}")
                
                if self.enable_optimization:
                    logger.info(f"🔧 Optimization: {results['optimization_stats']}")
                
                if self.enable_monitoring and 'session_metrics' in results:
                    logger.info(f"📈 Session: {results['session_metrics']['session_id']}")
            
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
            print(f"🎉 SUCCESS! Solution found: {format_hebrew_output(results['solution_word'])}")
            print(f"📊 Total guesses: {results['total_guesses']}")
            print(f"⏱️  Time taken: {results['elapsed_time']:.1f} seconds")
        else:
            print(f"❌ Puzzle not solved")
            if results.get('timeout_reached'):
                print("⏰ Reason: 5-minute timeout reached")
            
            best = results.get('best_candidate', {})
            if best.get('word'):
                print(f"🥈 Best candidate: {format_hebrew_output(best['word'])} (similarity: {best['similarity']:.2f})")
            
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
