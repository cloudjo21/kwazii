# Integration Tests for ASMR System

This directory contains comprehensive integration tests for the ASMR (Adaptive Small Modular Retrieval) system.

## Test Structure

### Step 1: Field Indexing (`test_step1_indexing.py`)
Tests the complete field indexing pipeline including:

- **Field Configuration**: 7 different field configs covering sparse/dense text and dense image
- **Sample Data**: 10 post-travel review documents with realistic content
- **Batch Indexing**: Efficient batch processing of documents
- **FAISS Integration**: Vector index creation and persistence
- **Index Persistence**: Saving and loading indices

#### Fields Tested:
- `title_sparse` & `title_dense` - Travel destination titles
- `content_sparse` & `content_dense` - Detailed review content  
- `review_text_sparse` & `review_text_dense` - Summary review text
- `review_image` - Travel photos (dense image field)

### Step 2: Query Retrieval (`test_step2_retrieval.py`)
Tests the complete query and retrieval pipeline including:

- **Query Processing**: Text-only, image-only, and multimodal queries
- **Retrieval Types**: Sparse text, dense text, dense image, and cross-modal
- **Integration**: QueryRouter and DocumentRetriever
- **Compatibility**: Query-field matching and target field detection

#### Query Types Tested:
1. **Sparse Text Query**: "amazing food experience restaurant"
2. **Multimodal Query**: "beautiful sunset mountain scenery" + image
3. **Comprehensive Query**: "travel adventure cultural experience"

## Running Tests

### Individual Tests
```bash
# Run Step 1 only
python -m pytest tests/integration/test_step1_indexing.py -v

# Run Step 2 only  
python -m pytest tests/integration/test_step2_retrieval.py -v
```

### Complete Integration Suite
```bash
# Run complete integration test suite
python tests/integration/run_integration_tests.py
```

## Test Features

### Mock Components
Tests use mock components to avoid external dependencies:
- `MockEncoder`: Simulates FDE encoder with deterministic embeddings
- `MockBM25Index`: Simulates BM25 sparse index
- `MockImage`: Simulates PIL Image objects

### Realistic Test Data
- **10 travel review documents** covering diverse destinations
- **Authentic content** with titles, detailed reviews, and image references
- **Consistent theme** allowing meaningful retrieval testing

### Comprehensive Coverage
- ✅ Field configuration and validation
- ✅ Sparse text indexing (BM25)
- ✅ Dense text indexing (FAISS)
- ✅ Dense image indexing (FAISS)
- ✅ Batch document processing
- ✅ Query creation and processing
- ✅ Single-modal and multimodal retrieval
- ✅ Cross-modal search (text-to-image)
- ✅ Complex retriever integration
- ✅ End-to-end pipeline validation

## Expected Output

### Successful Run
```
🚀 STARTING ASMR INTEGRATION TEST SUITE
========================================================
STEP 1: FIELD INDEXING INTEGRATION TEST
✓ All field configurations valid
✓ Sample documents properly structured  
✓ Indexing pipeline completed
✓ Indices saved to disk
✓ DocumentIndex integration successful

STEP 2: QUERY RETRIEVAL INTEGRATION TEST
✓ Sample queries properly structured
✓ Sparse text retrieval completed
✓ Dense multimodal retrieval completed
✓ Comprehensive retrieval completed
✓ QueryRouter integration successful
✓ DocumentRetriever integration successful

🎉 ALL INTEGRATION TESTS PASSED!

SYSTEM COMPONENTS VERIFIED:
  ✓ Field Configuration System
  ✓ Sparse Text Indexing (BM25)
  ✓ Dense Text Indexing (FAISS)
  ✓ Dense Image Indexing (FAISS)
  ✓ Batch Document Processing
  ✓ Query Processing (Text/Image/Multimodal)
  ✓ Retrieval Pipeline
  ✓ Complex Retriever Integration
  ✓ Cross-modal Search (Text-to-Image)
  ✓ End-to-End Workflow
```

## Configuration

Test configurations are centralized in `test_config.py`:
- Sample documents and queries
- Field configurations  
- Mock component implementations
- Test utilities and helpers

## Output Files

Tests generate output files for pipeline continuity:
- `step1_output.py` - Indexing test results
- `step2_output.py` - Retrieval test results

These files can be used for additional testing or analysis.

## Dependencies

### Required for Testing:
- `numpy` - Array operations
- `tempfile` - Temporary test directories
- `unittest` - Test framework

### Mocked (Optional):
- `faiss` - Vector similarity search
- `PIL` - Image processing
- `transformers` - HuggingFace tokenizers
- Custom FDE encoder implementations

## Notes

- Tests are designed to work with or without actual dependencies
- Mock components provide deterministic behavior for consistent testing
- Temporary directories are automatically cleaned up
- Tests validate both functionality and integration points
- Comprehensive error handling and status reporting included