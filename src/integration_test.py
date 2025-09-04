"""
Integration test for Hebrew Semantle Solver

Tests the integration between API client, beam search, and main solver
with a short timeout to verify all components work together.
"""

import logging
from .solver import SemantleSolver

# Configure logging for test
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_solver_integration():
    """Test solver integration with a short timeout"""
    print("🧪 Testing Hebrew Semantle Solver Integration")
    print("=" * 50)
    
    # Create solver with short timeout for testing (30 seconds) and conservative rate limiting
    solver = SemantleSolver(beam_width=3, timeout_minutes=0.5, rate_limit_seconds=3.0)
    
    try:
        print("Starting integration test (30-second timeout)...")
        result = solver.solve()
        
        print("\n" + "=" * 50)
        print("📊 INTEGRATION TEST RESULTS")
        print("=" * 50)
        
        # Verify results structure
        expected_keys = {'success', 'winning_word', 'total_time_seconds', 'total_guesses', 
                        'timeout_reached', 'best_word', 'best_similarity', 'words_tested'}
        
        missing_keys = expected_keys - set(result.keys())
        if missing_keys:
            print(f"❌ Missing result keys: {missing_keys}")
            return False
        
        print(f"✅ Result structure valid")
        print(f"🕐 Time: {result['total_time_seconds']:.1f}s")
        print(f"🎯 Guesses: {result['total_guesses']}")
        print(f"📊 Words tested: {result['words_tested']}")
        
        if result['success']:
            print(f"🎉 Found word: {result['winning_word']}")
        else:
            print(f"🏆 Best word: {result['best_word']} ({result['best_similarity']:.2f}%)")
            
            # Check that some progress was made
            if result['words_tested'] > 0:
                print(f"✅ Progress made: {result['words_tested']} words tested")
            else:
                print("❌ No words were tested")
                return False
        
        print("✅ Integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        logger.exception("Integration test error")
        return False
    finally:
        solver.cleanup()


def test_api_and_beam_integration():
    """Test API client and beam search integration separately"""
    print("\n🔧 Testing API + Beam Search Integration")
    print("-" * 50)
    
    try:
        from .api_client import SemantheAPIClient
        from .beam_search import BeamSearcher
        
        # Test API client
        client = SemantheAPIClient()
        searcher = BeamSearcher(beam_width=3)
        
        # Test a few words
        test_words = ["שלום", "אהבה", "בית"]
        
        for word in test_words:
            similarity = client.test_word_similarity(word)
            if similarity is not None:
                added = searcher.add_candidate(word, similarity)
                print(f"✓ {word}: {similarity:.2f}% {'(added to beam)' if added else '(filtered)'}")
            else:
                print(f"✗ {word}: API call failed")
        
        # Check beam status
        status = searcher.get_beam_status()
        print(f"\nBeam status: {status['tested_count']} words, best: {status['best_word']} ({status['best_similarity']:.2f}%)")
        
        client.close()
        print("✅ API + Beam integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ API + Beam integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Hebrew Semantle Solver - Integration Tests")
    print("=" * 60)
    
    # Test components individually first
    api_beam_ok = test_api_and_beam_integration()
    
    if api_beam_ok:
        # Test full solver integration
        solver_ok = test_solver_integration()
        
        if solver_ok:
            print("\n🎉 ALL INTEGRATION TESTS PASSED")
            print("Ready to solve Hebrew Semantle puzzles!")
        else:
            print("\n❌ Solver integration test failed")
    else:
        print("\n❌ Basic integration test failed")