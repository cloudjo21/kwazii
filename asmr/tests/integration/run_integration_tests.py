"""
Integration Test Runner for ASMR System
Runs both Step 1 (Indexing) and Step 2 (Retrieval) tests
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import test classes
from test_step1_indexing import TestFieldIndexing
from test_step2_retrieval import TestQueryRetrieval


class IntegrationTestSuite:
    """Complete integration test suite for ASMR system"""
    
    def __init__(self):
        self.step1_passed = False
        self.step2_passed = False
    
    def run_step1_indexing(self):
        """Run Step 1: Field indexing tests"""
        print("=" * 60)
        print("STEP 1: FIELD INDEXING INTEGRATION TEST")
        print("=" * 60)
        
        # Create test suite for step 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestFieldIndexing)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        self.step1_passed = result.wasSuccessful()
        
        if self.step1_passed:
            print("\\n✓ STEP 1 COMPLETED SUCCESSFULLY")
        else:
            print("\\n✗ STEP 1 FAILED")
            print(f"Failures: {len(result.failures)}")
            print(f"Errors: {len(result.errors)}")
        
        return self.step1_passed
    
    def run_step2_retrieval(self):
        """Run Step 2: Query retrieval tests"""
        if not self.step1_passed:
            print("\\n⚠️  SKIPPING STEP 2: Step 1 must pass first")
            return False
        
        print("\\n" + "=" * 60)
        print("STEP 2: QUERY RETRIEVAL INTEGRATION TEST")
        print("=" * 60)
        
        # Create test suite for step 2
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestQueryRetrieval)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        self.step2_passed = result.wasSuccessful()
        
        if self.step2_passed:
            print("\\n✓ STEP 2 COMPLETED SUCCESSFULLY")
        else:
            print("\\n✗ STEP 2 FAILED")
            print(f"Failures: {len(result.failures)}")
            print(f"Errors: {len(result.errors)}")
        
        return self.step2_passed
    
    def run_complete_suite(self):
        """Run the complete integration test suite"""
        print("🚀 STARTING ASMR INTEGRATION TEST SUITE")
        print("=" * 60)
        
        # Run Step 1
        step1_success = self.run_step1_indexing()
        
        # Run Step 2 if Step 1 passed
        step2_success = self.run_step2_retrieval() if step1_success else False
        
        # Final summary
        print("\\n" + "=" * 60)
        print("INTEGRATION TEST SUITE SUMMARY")
        print("=" * 60)
        
        print(f"Step 1 (Field Indexing): {'✓ PASS' if step1_success else '✗ FAIL'}")
        print(f"Step 2 (Query Retrieval): {'✓ PASS' if step2_success else '✗ FAIL'}")
        
        if step1_success and step2_success:
            print("\\n🎉 ALL INTEGRATION TESTS PASSED!")
            print("\\nSYSTEM COMPONENTS VERIFIED:")
            print("  ✓ Field Configuration System")
            print("  ✓ Sparse Text Indexing (BM25)")
            print("  ✓ Dense Text Indexing (FAISS)")
            print("  ✓ Dense Image Indexing (FAISS)")
            print("  ✓ Batch Document Processing")
            print("  ✓ Query Processing (Text/Image/Multimodal)")
            print("  ✓ Retrieval Pipeline")
            print("  ✓ Complex Retriever Integration")
            print("  ✓ Cross-modal Search (Text-to-Image)")
            print("  ✓ End-to-End Workflow")
            
            return True
        else:
            print("\\n❌ INTEGRATION TESTS FAILED")
            print("\\nPlease check the failed tests above and fix issues before proceeding.")
            return False


def main():
    """Main entry point for integration tests"""
    # Run the complete test suite
    test_suite = IntegrationTestSuite()
    success = test_suite.run_complete_suite()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()