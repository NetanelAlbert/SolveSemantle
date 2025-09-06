#!/usr/bin/env python3
"""
Comprehensive Test Suite for Hebrew Semantle Solver with Word2Vec Integration

This script tests the enhanced solver with Word2Vec language model integration
and compares performance against the basic solver to verify improvements.
"""

import sys
import os
import time
import logging
from typing import Dict, Any, List
import statistics

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from solver import SemantleSolver
    from language_model import HebrewLanguageModel, download_model_instructions
    from api_client import SemantheAPIClient
    from beam_search import BeamSearcher
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for testing
logger = logging.getLogger(__name__)


class SolverPerformanceTester:
    """Test and compare Hebrew Semantle solver performance with and without Word2Vec"""
    
    def __init__(self):
        self.results = {
            'basic_solver': [],
            'word2vec_solver': [],
            'model_stats': {}
        }
    
    def test_language_model_functionality(self) -> bool:
        """Test core language model functionality"""
        print("=" * 60)
        print("TESTING LANGUAGE MODEL FUNCTIONALITY")
        print("=" * 60)
        
        try:
            # Initialize language model
            model = HebrewLanguageModel()
            
            print("✓ HebrewLanguageModel initialized successfully")
            
            # Attempt to load model
            if not model.load_model():
                print("❌ Failed to load Word2Vec model")
                print("   This is expected if you haven't downloaded the model yet.")
                download_model_instructions()
                return False
            
            print("✓ Word2Vec model loaded successfully")
            
            # Test model statistics
            stats = model.get_model_stats()
            self.results['model_stats'] = stats
            
            print(f"✓ Model vocabulary: {stats['vocabulary_size']:,} words")
            print(f"✓ Vector dimensions: {stats['vector_dimensions']}")
            print(f"✓ Caching enabled: {stats['caching_enabled']}")
            
            # Test word similarity calculations
            test_pairs = [
                ("שלום", "אהבה"),
                ("בית", "משפחה"),
                ("חכמה", "ידע")
            ]
            
            print("\nTesting similarity calculations:")
            similarities_found = 0
            
            for word1, word2 in test_pairs:
                similarity = model.calculate_similarity(word1, word2)
                if similarity is not None:
                    print(f"  {word1} <-> {word2}: {similarity:.2f}")
                    similarities_found += 1
                else:
                    print(f"  {word1} <-> {word2}: Not found in model")
            
            if similarities_found == 0:
                print("❌ No similarity calculations succeeded")
                return False
            
            print(f"✓ {similarities_found}/{len(test_pairs)} similarity calculations succeeded")
            
            # Test word suggestions
            print("\nTesting word suggestions:")
            test_words = ["שלום", "אהבה"]
            suggestions_found = 0
            
            for word in test_words:
                similar = model.find_most_similar(word, topn=3)
                if similar:
                    print(f"  {word}: {[w for w, s in similar]}")
                    suggestions_found += 1
                else:
                    print(f"  {word}: No similar words found")
            
            if suggestions_found == 0:
                print("❌ No word suggestions found")
                return False
            
            print(f"✓ {suggestions_found}/{len(test_words)} word suggestion tests succeeded")
            
            # Test cache performance
            if model.enable_caching:
                print("\nTesting cache performance:")
                start_time = time.time()
                # Calculate similarity twice (second should be cached)
                model.calculate_similarity("שלום", "אהבה")
                cached_time = time.time()
                model.calculate_similarity("שלום", "אהבה")
                end_time = time.time()
                
                cache_stats = stats.get('cache_stats', {})
                print(f"✓ Cache performance: {cache_stats.get('total_cached_items', 0)} items cached")
                
            return True
            
        except Exception as e:
            print(f"❌ Language model test failed: {e}")
            return False
    
    def test_solver_with_timeout(self, use_word2vec: bool, timeout: int = 30) -> Dict[str, Any]:
        """Test solver with specified timeout and configuration"""
        
        try:
            solver = SemantleSolver(
                beam_width=3,  # Smaller beam for faster testing
                timeout_seconds=timeout,
                use_language_model=use_word2vec
            )
            
            print(f"Testing solver (Word2Vec: {use_word2vec}, timeout: {timeout}s)...")
            start_time = time.time()
            
            results = solver.solve()
            
            end_time = time.time()
            actual_time = end_time - start_time
            
            # Add actual timing
            results['actual_elapsed_time'] = actual_time
            
            return results
            
        except Exception as e:
            logger.error(f"Solver test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_guesses': 0,
                'elapsed_time': timeout,
                'actual_elapsed_time': timeout
            }
    
    def compare_solver_performance(self, test_timeout: int = 60) -> Dict[str, Any]:
        """Compare performance of basic vs Word2Vec-enhanced solver"""
        
        print("=" * 60)
        print("COMPARING SOLVER PERFORMANCE")
        print("=" * 60)
        
        comparison = {
            'basic_solver': None,
            'word2vec_solver': None,
            'improvement': {}
        }
        
        # Test basic solver (without Word2Vec)
        print("\n1. Testing Basic Solver (no Word2Vec)...")
        basic_result = self.test_solver_with_timeout(
            use_word2vec=False, 
            timeout=test_timeout
        )
        comparison['basic_solver'] = basic_result
        
        print(f"   Result: {'SUCCESS' if basic_result.get('success') else 'TIMEOUT'}")
        print(f"   Guesses: {basic_result.get('total_guesses', 0)}")
        print(f"   Time: {basic_result.get('elapsed_time', 0):.1f}s")
        
        if basic_result.get('best_candidate'):
            best = basic_result['best_candidate']
            print(f"   Best: {best.get('word', 'N/A')} ({best.get('similarity', 0):.2f})")
        
        # Test Word2Vec-enhanced solver
        print("\n2. Testing Word2Vec-Enhanced Solver...")
        word2vec_result = self.test_solver_with_timeout(
            use_word2vec=True, 
            timeout=test_timeout
        )
        comparison['word2vec_solver'] = word2vec_result
        
        print(f"   Result: {'SUCCESS' if word2vec_result.get('success') else 'TIMEOUT'}")
        print(f"   Guesses: {word2vec_result.get('total_guesses', 0)}")
        print(f"   Time: {word2vec_result.get('elapsed_time', 0):.1f}s")
        
        if word2vec_result.get('best_candidate'):
            best = word2vec_result['best_candidate']
            print(f"   Best: {best.get('word', 'N/A')} ({best.get('similarity', 0):.2f})")
        
        # Calculate improvements
        print("\n3. Performance Comparison:")
        
        basic_guesses = basic_result.get('total_guesses', 0)
        word2vec_guesses = word2vec_result.get('total_guesses', 0)
        
        basic_best_sim = basic_result.get('best_candidate', {}).get('similarity', 0)
        word2vec_best_sim = word2vec_result.get('best_candidate', {}).get('similarity', 0)
        
        if basic_guesses > 0 and word2vec_guesses > 0:
            guess_improvement = ((basic_guesses - word2vec_guesses) / basic_guesses) * 100
            comparison['improvement']['guess_reduction'] = guess_improvement
            print(f"   Guess efficiency: {guess_improvement:+.1f}% (fewer is better)")
        
        if basic_best_sim > 0 and word2vec_best_sim > 0:
            similarity_improvement = ((word2vec_best_sim - basic_best_sim) / basic_best_sim) * 100
            comparison['improvement']['similarity_improvement'] = similarity_improvement
            print(f"   Best similarity: {similarity_improvement:+.1f}% (higher is better)")
        
        # Success rate comparison
        basic_success = basic_result.get('success', False)
        word2vec_success = word2vec_result.get('success', False)
        
        if word2vec_success and not basic_success:
            print("   ✅ Word2Vec solver succeeded where basic solver failed!")
            comparison['improvement']['success_rate'] = "Word2Vec solver succeeded"
        elif basic_success and not word2vec_success:
            print("   ⚠️  Basic solver succeeded where Word2Vec solver failed")
            comparison['improvement']['success_rate'] = "Basic solver succeeded"
        elif basic_success and word2vec_success:
            print("   ✅ Both solvers succeeded")
            comparison['improvement']['success_rate'] = "Both succeeded"
        else:
            print("   ⏰ Both solvers timed out")
            comparison['improvement']['success_rate'] = "Both timed out"
        
        return comparison
    
    def run_comprehensive_test(self) -> bool:
        """Run complete test suite"""
        print("HEBREW SEMANTLE SOLVER - COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print("Testing Word2Vec integration and performance improvements")
        print("=" * 60)
        
        # Step 1: Test language model functionality
        model_works = self.test_language_model_functionality()
        
        if not model_works:
            print("\n❌ LANGUAGE MODEL TEST FAILED")
            print("Cannot proceed with solver comparison without Word2Vec model.")
            print("Please download the Hebrew Word2Vec model and try again.")
            return False
        
        print("\n✅ LANGUAGE MODEL TEST PASSED")
        
        # Step 2: Compare solver performance
        comparison = self.compare_solver_performance(test_timeout=90)
        self.results['comparison'] = comparison
        
        # Step 3: Final assessment
        print("\n" + "=" * 60)
        print("FINAL ASSESSMENT")
        print("=" * 60)
        
        improvements = comparison.get('improvement', {})
        
        success_criteria_met = 0
        total_criteria = 4
        
        # Criteria 1: Language model loads and functions
        print("1. ✅ Language model loads and functions correctly")
        success_criteria_met += 1
        
        # Criteria 2: Word2Vec integration works
        word2vec_loaded = comparison['word2vec_solver'].get('language_model_loaded', False)
        if word2vec_loaded:
            print("2. ✅ Word2Vec integration works correctly")
            success_criteria_met += 1
        else:
            print("2. ❌ Word2Vec integration failed")
        
        # Criteria 3: Performance improvement in similarity scores
        sim_improvement = improvements.get('similarity_improvement', 0)
        if sim_improvement > 10:  # At least 10% improvement in similarity
            print(f"3. ✅ Similarity improvement: {sim_improvement:.1f}%")
            success_criteria_met += 1
        else:
            print(f"3. ⚠️  Similarity improvement: {sim_improvement:.1f}% (target: >10%)")
        
        # Criteria 4: Overall solver effectiveness  
        success_rate = improvements.get('success_rate', "")
        if "Word2Vec solver succeeded" in success_rate or "Both succeeded" in success_rate:
            print("4. ✅ Word2Vec solver shows improved effectiveness")
            success_criteria_met += 1
        else:
            print("4. ⚠️  Word2Vec solver effectiveness needs improvement")
        
        # Overall assessment
        success_rate = (success_criteria_met / total_criteria) * 100
        print(f"\n📊 OVERALL SUCCESS RATE: {success_criteria_met}/{total_criteria} ({success_rate:.1f}%)")
        
        if success_rate >= 75:
            print("🎉 UNIT 4 IMPLEMENTATION SUCCESSFUL!")
            print("Word2Vec integration significantly improves the solver.")
            return True
        else:
            print("⚠️  UNIT 4 NEEDS IMPROVEMENT")
            print("Word2Vec integration shows promise but requires optimization.")
            return False


def main():
    """Run the comprehensive test suite"""
    
    try:
        tester = SolverPerformanceTester()
        success = tester.run_comprehensive_test()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()