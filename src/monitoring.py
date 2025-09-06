#!/usr/bin/env python3
"""
Comprehensive Monitoring and Analysis System for Hebrew Semantle Solver

Provides detailed metrics collection, strategy effectiveness analysis,
performance comparison tools, and real-time progress visualization.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SolverMetrics:
    """Comprehensive metrics for a solver session"""
    session_id: str
    start_time: float
    end_time: Optional[float]
    success: bool
    solution_word: Optional[str]
    total_guesses: int
    elapsed_time: float
    best_similarity: float
    words_tested: int
    beam_size_final: int
    beam_width_final: int
    search_strategy: str
    strategy_used: str
    learning_enabled: bool
    optimization_enabled: bool
    language_model_loaded: bool
    
    # Detailed breakdowns
    similarity_progression: List[Tuple[float, float]]  # (timestamp, similarity)
    strategy_usage: Dict[str, int]  # strategy -> count
    beam_width_progression: List[Tuple[float, int]]  # (timestamp, width)
    timeout_extensions: int
    emergency_activated: bool
    
    # Performance categories
    api_calls_per_similarity_bracket: Dict[str, int]  # "0-20", "20-40", etc.
    time_distribution: Dict[str, float]  # phase -> time spent
    convergence_rate: float  # guesses per similarity point gained


@dataclass
class StrategyEffectiveness:
    """Analysis of strategy effectiveness"""
    strategy_name: str
    total_uses: int
    success_rate: float
    avg_similarity_improvement: float
    avg_time_per_use: float
    words_generated: int
    high_similarity_hits: int  # 70+ similarity
    
    # Effectiveness scores
    efficiency_score: float  # improvement per time unit
    reliability_score: float  # consistency of performance
    discovery_score: float  # ability to find high similarity words


class ComprehensiveMonitor:
    """Comprehensive monitoring and analysis system"""
    
    def __init__(self, storage_path: str = "solver_metrics"):
        """
        Initialize comprehensive monitoring system
        
        Args:
            storage_path: Directory to store metrics and analysis data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Current session tracking
        self.current_session_id = None
        self.session_start_time = None
        self.current_metrics = None
        self.similarity_progression = []
        self.strategy_usage = defaultdict(int)
        self.beam_width_progression = []
        self.time_distribution = defaultdict(float)
        self.phase_start_time = None
        
        # Word score tracking for top words analysis
        self.all_word_scores = []  # List of (word, similarity, strategy) tuples
        
        # Historical data
        self.session_history = []
        self.strategy_performance = defaultdict(list)
        
        # Real-time monitoring
        self.progress_callbacks = []
        self.milestone_thresholds = [20, 40, 60, 70, 80, 90, 95]
        self.achieved_milestones = set()
        
        logger.info(f"Initialized ComprehensiveMonitor (storage: {storage_path})")
    
    def start_session(self, session_config: Dict[str, Any]) -> str:
        """
        Start monitoring a new solver session
        
        Args:
            session_config: Configuration parameters for the session
            
        Returns:
            Session ID for tracking
        """
        self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_start_time = time.time()
        self.similarity_progression = []
        self.strategy_usage = defaultdict(int)
        self.beam_width_progression = []
        self.time_distribution = defaultdict(float)
        self.achieved_milestones = set()
        self.phase_start_time = self.session_start_time
        self.all_word_scores = []  # Reset word scores for new session
        
        logger.info(f"Started monitoring session: {self.current_session_id}")
        logger.info(f"Session config: {session_config}")
        
        return self.current_session_id
    
    def log_phase_start(self, phase_name: str):
        """Log the start of a solver phase"""
        if self.phase_start_time:
            # Log time spent in previous phase
            phase_time = time.time() - self.phase_start_time
            # Get previous phase name from the last entry or default
            prev_phase = "unknown"
            if self.time_distribution:
                prev_phase = list(self.time_distribution.keys())[-1] if self.time_distribution else "initialization"
            self.time_distribution[prev_phase] += phase_time
        
        self.phase_start_time = time.time()
        logger.debug(f"Phase started: {phase_name}")
    
    def log_word_test(self, word: str, similarity: float, strategy: str, parent_similarity: float = 0.0):
        """
        Log a word test event
        
        Args:
            word: Word that was tested
            similarity: Similarity score received
            strategy: Strategy that generated this word
            parent_similarity: Similarity of parent candidate
        """
        current_time = time.time()
        
        # Track similarity progression
        self.similarity_progression.append((current_time, similarity))
        
        # Track strategy usage
        self.strategy_usage[strategy] += 1
        
        # Track word scores for top words analysis
        self.all_word_scores.append((word, similarity, strategy))
        
        # Check for milestones
        for threshold in self.milestone_thresholds:
            if similarity >= threshold and threshold not in self.achieved_milestones:
                self.achieved_milestones.add(threshold)
                milestone_time = current_time - self.session_start_time
                logger.info(f"🎯 MILESTONE: {threshold}% similarity reached in {milestone_time:.1f}s with word '{word}' (strategy: {strategy})")
                
                # Trigger milestone callbacks
                for callback in self.progress_callbacks:
                    try:
                        callback('milestone', {
                            'threshold': threshold,
                            'word': word,
                            'similarity': similarity,
                            'time': milestone_time,
                            'strategy': strategy
                        })
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")
        
        # Log significant improvements
        if self.similarity_progression and len(self.similarity_progression) > 1:
            prev_similarity = max(s for t, s in self.similarity_progression[:-1])
            improvement = similarity - prev_similarity
            if improvement >= 10.0:
                logger.info(f"📈 SIGNIFICANT IMPROVEMENT: +{improvement:.1f} points with '{word}' (strategy: {strategy})")
    
    def log_beam_width_change(self, new_width: int, reason: str):
        """Log beam width changes"""
        current_time = time.time()
        self.beam_width_progression.append((current_time, new_width))
        logger.debug(f"Beam width changed to {new_width} ({reason})")
    
    def log_strategy_switch(self, from_strategy: str, to_strategy: str, reason: str):
        """Log strategy switches"""
        logger.info(f"Strategy switch: {from_strategy} → {to_strategy} ({reason})")
    
    def log_emergency_activation(self, words_tested: int, best_similarity: float):
        """Log emergency strategy activation"""
        logger.warning(f"🚨 EMERGENCY STRATEGIES ACTIVATED: {words_tested} words tested, best similarity: {best_similarity:.2f}")
    
    def log_timeout_extension(self, extension_number: int, reason: str, additional_time: float):
        """Log smart timeout extensions"""
        logger.info(f"⏰ TIMEOUT EXTENDED #{extension_number}: +{additional_time:.0f}s ({reason})")
    
    def end_session(self, final_results: Dict[str, Any]) -> SolverMetrics:
        """
        End the current monitoring session and compile metrics
        
        Args:
            final_results: Final results from the solver
            
        Returns:
            Compiled metrics for the session
        """
        if not self.current_session_id:
            raise ValueError("No active session to end")
        
        end_time = time.time()
        
        # Finalize time distribution
        if self.phase_start_time:
            final_phase_time = end_time - self.phase_start_time
            self.time_distribution["final_phase"] += final_phase_time
        
        # Calculate additional metrics
        api_calls_per_bracket = self._calculate_similarity_brackets()
        convergence_rate = self._calculate_convergence_rate()
        
        # Create comprehensive metrics
        metrics = SolverMetrics(
            session_id=self.current_session_id,
            start_time=self.session_start_time,
            end_time=end_time,
            success=final_results.get('success', False),
            solution_word=final_results.get('solution_word'),
            total_guesses=final_results.get('total_guesses', 0),
            elapsed_time=final_results.get('elapsed_time', 0),
            best_similarity=final_results.get('best_candidate', {}).get('similarity', 0),
            words_tested=final_results.get('words_tested', 0),
            beam_size_final=final_results.get('beam_size', 0),
            beam_width_final=final_results.get('beam_width_final', 0),
            search_strategy=final_results.get('search_strategy', 'unknown'),
            strategy_used=final_results.get('strategy_used', 'unknown'),
            learning_enabled=final_results.get('learning_enabled', False),
            optimization_enabled=final_results.get('optimization_enabled', False),
            language_model_loaded=final_results.get('language_model_loaded', False),
            similarity_progression=self.similarity_progression.copy(),
            strategy_usage=dict(self.strategy_usage),
            beam_width_progression=self.beam_width_progression.copy(),
            timeout_extensions=final_results.get('optimization_stats', {}).get('smart_timeout_extensions', 0),
            emergency_activated=final_results.get('optimization_stats', {}).get('emergency_activated', False),
            api_calls_per_similarity_bracket=api_calls_per_bracket,
            time_distribution=dict(self.time_distribution),
            convergence_rate=convergence_rate
        )
        
        # Store metrics
        self.session_history.append(metrics)
        self._save_session_metrics(metrics)
        
        logger.info(f"Session {self.current_session_id} completed and metrics saved")
        
        # Print top 10 highest scoring words
        self._print_top_words()
        
        # Reset current session
        self.current_session_id = None
        self.current_metrics = metrics
        
        return metrics
    
    def _print_top_words(self, top_n: int = 10):
        """Print the highest scoring words from the session"""
        if not self.all_word_scores:
            print("No words tested in this session")
            return
        
        # Sort by similarity score (highest first)
        sorted_words = sorted(self.all_word_scores, key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:top_n]
        
        print(f"\n{'='*60}")
        print(f"🏆 TOP {min(top_n, len(top_words))} HIGHEST SCORING WORDS")
        print(f"{'='*60}")
        
        for i, (word, similarity, strategy) in enumerate(top_words, 1):
            # Format Hebrew word for display
            try:
                from hebrew_utils import format_hebrew_output
                formatted_word = format_hebrew_output(word)
            except ImportError:
                formatted_word = word
            
            print(f"{i:2d}. {formatted_word:<15} | {similarity:6.2f}% | {strategy}")
        
        print(f"{'='*60}")
        
        # Additional statistics
        if len(sorted_words) > 0:
            avg_score = sum(score for _, score, _ in sorted_words) / len(sorted_words)
            max_score = sorted_words[0][1]
            min_score = sorted_words[-1][1]
            print(f"📊 Score Statistics: Max: {max_score:.2f}%, Min: {min_score:.2f}%, Avg: {avg_score:.2f}%")
            
            # Strategy breakdown for top words
            strategy_counts = defaultdict(int)
            for _, _, strategy in top_words:
                strategy_counts[strategy] += 1
            
            print(f"🎯 Top Words by Strategy: ", end="")
            strategy_summary = [f"{strategy}: {count}" for strategy, count in strategy_counts.items()]
            print(", ".join(strategy_summary))
            print(f"{'='*60}\n")
    
    def _calculate_similarity_brackets(self) -> Dict[str, int]:
        """Calculate API calls per similarity bracket"""
        brackets = {
            "0-20": 0, "20-40": 0, "40-60": 0, 
            "60-70": 0, "70-80": 0, "80-90": 0, "90-100": 0
        }
        
        for _, similarity in self.similarity_progression:
            if similarity < 20:
                brackets["0-20"] += 1
            elif similarity < 40:
                brackets["20-40"] += 1
            elif similarity < 60:
                brackets["40-60"] += 1
            elif similarity < 70:
                brackets["60-70"] += 1
            elif similarity < 80:
                brackets["70-80"] += 1
            elif similarity < 90:
                brackets["80-90"] += 1
            else:
                brackets["90-100"] += 1
        
        return brackets
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate (guesses per similarity point)"""
        if not self.similarity_progression or len(self.similarity_progression) < 2:
            return 0.0
        
        max_similarity = max(s for _, s in self.similarity_progression)
        min_similarity = min(s for _, s in self.similarity_progression)
        similarity_gained = max_similarity - min_similarity
        
        if similarity_gained <= 0:
            return 0.0
        
        return len(self.similarity_progression) / similarity_gained
    
    def _save_session_metrics(self, metrics: SolverMetrics):
        """Save session metrics to file"""
        try:
            metrics_file = self.storage_path / f"{metrics.session_id}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(metrics), f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save session metrics: {e}")
    
    def analyze_strategy_effectiveness(self, recent_sessions: int = 10) -> List[StrategyEffectiveness]:
        """
        Analyze strategy effectiveness across recent sessions
        
        Args:
            recent_sessions: Number of recent sessions to analyze
            
        Returns:
            List of strategy effectiveness analyses
        """
        if not self.session_history:
            logger.warning("No session history available for analysis")
            return []
        
        # Get recent sessions
        recent_sessions_data = self.session_history[-recent_sessions:]
        strategy_stats = defaultdict(lambda: {
            'uses': 0,
            'successes': 0,
            'similarity_improvements': [],
            'times': [],
            'high_similarity_hits': 0
        })
        
        # Collect strategy data
        for session in recent_sessions_data:
            for strategy, count in session.strategy_usage.items():
                stats = strategy_stats[strategy]
                stats['uses'] += count
                
                if session.success:
                    stats['successes'] += 1
                
                # Calculate improvement contributions
                # This is simplified - in a full implementation, you'd track per-strategy improvements
                if session.best_similarity > 70:
                    stats['high_similarity_hits'] += 1
                
                stats['times'].append(session.elapsed_time / session.total_guesses if session.total_guesses > 0 else 0)
        
        # Calculate effectiveness metrics
        effectiveness_analyses = []
        for strategy, stats in strategy_stats.items():
            if stats['uses'] == 0:
                continue
            
            success_rate = stats['successes'] / len(recent_sessions_data)
            avg_time = statistics.mean(stats['times']) if stats['times'] else 0.0
            efficiency_score = stats['high_similarity_hits'] / (avg_time + 0.1)  # Prevent division by zero
            reliability_score = min(1.0, stats['uses'] / (len(recent_sessions_data) * 5))  # Normalize usage
            discovery_score = stats['high_similarity_hits'] / max(1, stats['uses'])
            
            effectiveness = StrategyEffectiveness(
                strategy_name=strategy,
                total_uses=stats['uses'],
                success_rate=success_rate,
                avg_similarity_improvement=0.0,  # Simplified
                avg_time_per_use=avg_time,
                words_generated=stats['uses'],
                high_similarity_hits=stats['high_similarity_hits'],
                efficiency_score=efficiency_score,
                reliability_score=reliability_score,
                discovery_score=discovery_score
            )
            
            effectiveness_analyses.append(effectiveness)
        
        # Sort by efficiency score
        effectiveness_analyses.sort(key=lambda x: x.efficiency_score, reverse=True)
        
        logger.info(f"Analyzed strategy effectiveness for {len(effectiveness_analyses)} strategies across {len(recent_sessions_data)} sessions")
        return effectiveness_analyses
    
    def generate_performance_report(self, include_recent: int = 5) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            include_recent: Number of recent sessions to include in analysis
            
        Returns:
            Comprehensive performance report
        """
        if not self.session_history:
            return {"error": "No session data available"}
        
        recent_sessions = self.session_history[-include_recent:]
        
        # Calculate aggregate statistics
        total_sessions = len(recent_sessions)
        successful_sessions = sum(1 for s in recent_sessions if s.success)
        success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0.0
        
        avg_guesses = statistics.mean([s.total_guesses for s in recent_sessions]) if recent_sessions else 0.0
        avg_time = statistics.mean([s.elapsed_time for s in recent_sessions]) if recent_sessions else 0.0
        avg_best_similarity = statistics.mean([s.best_similarity for s in recent_sessions]) if recent_sessions else 0.0
        
        # Strategy effectiveness
        strategy_effectiveness = self.analyze_strategy_effectiveness(include_recent)
        
        # Performance trends
        if len(recent_sessions) >= 2:
            recent_success_rate = successful_sessions / total_sessions
            older_sessions = self.session_history[max(0, len(self.session_history)-include_recent*2):-include_recent]
            older_success_rate = sum(1 for s in older_sessions if s.success) / len(older_sessions) if older_sessions else 0.0
            trend = "improving" if recent_success_rate > older_success_rate else "declining" if recent_success_rate < older_success_rate else "stable"
        else:
            trend = "insufficient_data"
        
        # Compile report
        report = {
            "report_generated": datetime.now().isoformat(),
            "analysis_period": f"Recent {include_recent} sessions",
            "summary": {
                "total_sessions": total_sessions,
                "successful_sessions": successful_sessions,
                "success_rate": success_rate,
                "avg_guesses_per_session": avg_guesses,
                "avg_time_per_session": avg_time,
                "avg_best_similarity": avg_best_similarity,
                "performance_trend": trend
            },
            "strategy_effectiveness": [asdict(se) for se in strategy_effectiveness],
            "recent_sessions": [
                {
                    "session_id": s.session_id,
                    "success": s.success,
                    "guesses": s.total_guesses,
                    "time": s.elapsed_time,
                    "best_similarity": s.best_similarity,
                    "strategy_used": s.strategy_used
                }
                for s in recent_sessions
            ],
            "optimization_impact": self._analyze_optimization_impact(recent_sessions)
        }
        
        logger.info(f"Generated performance report for {total_sessions} sessions")
        return report
    
    def _analyze_optimization_impact(self, sessions: List[SolverMetrics]) -> Dict[str, Any]:
        """Analyze the impact of optimization features"""
        learning_sessions = [s for s in sessions if s.learning_enabled]
        optimization_sessions = [s for s in sessions if s.optimization_enabled]
        baseline_sessions = [s for s in sessions if not s.learning_enabled and not s.optimization_enabled]
        
        def calc_avg_similarity(session_list):
            return statistics.mean([s.best_similarity for s in session_list]) if session_list else 0.0
        
        return {
            "learning_impact": {
                "sessions_with_learning": len(learning_sessions),
                "avg_similarity_with_learning": calc_avg_similarity(learning_sessions),
                "avg_similarity_without_learning": calc_avg_similarity([s for s in sessions if not s.learning_enabled])
            },
            "optimization_impact": {
                "sessions_with_optimization": len(optimization_sessions),
                "avg_similarity_with_optimization": calc_avg_similarity(optimization_sessions),
                "avg_similarity_without_optimization": calc_avg_similarity([s for s in sessions if not s.optimization_enabled])
            },
            "baseline_performance": {
                "baseline_sessions": len(baseline_sessions),
                "baseline_avg_similarity": calc_avg_similarity(baseline_sessions)
            },
            "emergency_activation_rate": sum(1 for s in sessions if s.emergency_activated) / len(sessions) if sessions else 0.0,
            "avg_timeout_extensions": statistics.mean([s.timeout_extensions for s in sessions]) if sessions else 0.0
        }
    
    def add_progress_callback(self, callback):
        """Add a callback function for real-time progress updates"""
        self.progress_callbacks.append(callback)
    
    def save_full_analysis(self, filepath: str = None):
        """Save comprehensive analysis to file"""
        if not filepath:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.storage_path / f"comprehensive_analysis_{timestamp}.json"
        
        analysis = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_sessions": len(self.session_history),
            "performance_report": self.generate_performance_report(),
            "strategy_effectiveness": [asdict(se) for se in self.analyze_strategy_effectiveness()],
            "session_summary": [
                {
                    "session_id": s.session_id,
                    "success": s.success,
                    "total_guesses": s.total_guesses,
                    "elapsed_time": s.elapsed_time,
                    "best_similarity": s.best_similarity,
                    "strategy_used": s.strategy_used,
                    "learning_enabled": s.learning_enabled,
                    "optimization_enabled": s.optimization_enabled
                }
                for s in self.session_history
            ]
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"Comprehensive analysis saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")


if __name__ == "__main__":
    # Test the monitoring system
    print("Testing Comprehensive Monitoring System...")
    
    monitor = ComprehensiveMonitor("test_metrics")
    
    # Start a test session
    session_id = monitor.start_session({
        "beam_width": 3,
        "learning_enabled": True,
        "optimization_enabled": True
    })
    
    # Simulate some word tests
    monitor.log_phase_start("initial_phase")
    monitor.log_word_test("שלום", 25.0, "initial", 0.0)
    monitor.log_word_test("חיים", 45.0, "semantic", 25.0)
    monitor.log_word_test("אהבה", 72.0, "morphological", 45.0)  # Should trigger milestone
    
    # Simulate beam width change
    monitor.log_beam_width_change(4, "progress_made")
    
    # End session with mock results
    final_results = {
        "success": False,
        "total_guesses": 3,
        "elapsed_time": 15.5,
        "best_candidate": {"similarity": 72.0},
        "words_tested": 3,
        "beam_size": 3,
        "beam_width_final": 4,
        "search_strategy": "exploitation",
        "strategy_used": "Word2Vec + Learning + Optimization",
        "learning_enabled": True,
        "optimization_enabled": True,
        "language_model_loaded": True,
        "optimization_stats": {
            "smart_timeout_extensions": 1,
            "emergency_activated": False
        }
    }
    
    metrics = monitor.end_session(final_results)
    print(f"Session metrics: {metrics.session_id}, guesses: {metrics.total_guesses}, best: {metrics.best_similarity}")
    
    # Generate performance report
    report = monitor.generate_performance_report(1)
    print(f"Performance report: {report['summary']['success_rate']:.1%} success rate")
    
    print("✅ Comprehensive monitoring system test completed!")
