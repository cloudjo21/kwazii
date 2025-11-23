"""
Test Configuration and Utilities for ASMR Integration Tests
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Test data and configurations for integration tests
class TestConfig:
    """Central configuration for integration tests"""
    
    # Theme: Post-travel reviews
    SAMPLE_DOCUMENTS = [
        {
            "doc_id": "review_001",
            "title": "Amazing Weekend in Paris",
            "content": "Paris exceeded all my expectations. The Eiffel Tower at sunset was breathtaking, and the local cafes served incredible croissants. The Seine river cruise was romantic and peaceful.",
            "review_text": "Highly recommend visiting in spring. The weather was perfect and crowds were manageable. 5 stars!",
            "review_image": "paris_sunset.jpg"
        },
        {
            "doc_id": "review_002", 
            "title": "Tokyo Food Adventure",
            "content": "Tokyo's food scene is unparalleled. From street food in Shibuya to high-end sushi in Ginza, every meal was memorable. The ramen shops were particularly outstanding.",
            "review_text": "Must try the conveyor belt sushi and visit Tsukiji fish market early morning. Cultural experience was amazing!",
            "review_image": "tokyo_ramen.jpg"
        },
        {
            "doc_id": "review_003",
            "title": "Safari Experience in Kenya",
            "content": "The wildlife safari in Maasai Mara was life-changing. Witnessed the great migration and saw the Big Five. Local guides were knowledgeable and friendly.",
            "review_text": "Best time to visit is during migration season. Accommodation was comfortable but basic. Unforgettable experience!",
            "review_image": "kenya_safari.jpg"
        },
        {
            "doc_id": "review_004",
            "title": "New York City Marathon",
            "content": "Running the NYC Marathon was a dream come true. The crowd support was incredible, and running through all five boroughs showcased the city's diversity.",
            "review_text": "Training was tough but worth it. The finish line in Central Park was emotional. Great organization by race officials.",
            "review_image": "nyc_marathon.jpg"
        },
        {
            "doc_id": "review_005",
            "title": "Bali Beach Paradise",
            "content": "Bali's beaches are pristine and the sunsets spectacular. Ubud's rice terraces and temples provided cultural depth beyond the beach experience.",
            "review_text": "Perfect honeymoon destination. Balinese hospitality is genuine and the spa treatments were rejuvenating.",
            "review_image": "bali_sunset.jpg"
        },
        {
            "doc_id": "review_006",
            "title": "Iceland Northern Lights",
            "content": "Iceland in winter offered magical northern lights viewing. The Blue Lagoon geothermal spa was relaxing after long days of aurora hunting.",
            "review_text": "Pack warm clothes! Northern lights were visible 3 out of 5 nights. Ice caves tour was phenomenal.",
            "review_image": "iceland_aurora.jpg"
        },
        {
            "doc_id": "review_007",
            "title": "Swiss Alps Hiking",
            "content": "The Swiss Alps provided world-class hiking trails with stunning mountain vistas. Cable cars made accessing high-altitude trails convenient.",
            "review_text": "Expensive but worth every penny. Trail marking was excellent and mountain huts offered great meals.",
            "review_image": "swiss_mountains.jpg"
        },
        {
            "doc_id": "review_008",
            "title": "Brazilian Carnival Experience", 
            "content": "Rio's Carnival was an explosion of color, music, and energy. Participating in street parties and watching parade competitions was thrilling.",
            "review_text": "Book accommodations early as prices skyrocket. Learn some Portuguese phrases for better local interaction.",
            "review_image": "rio_carnival.jpg"
        },
        {
            "doc_id": "review_009",
            "title": "Australian Outback Adventure",
            "content": "The vastness of the Australian Outback is humbling. Uluru at sunrise was spiritual, and stargazing in the desert was spectacular.",
            "review_text": "Guided tours are recommended for safety. Camping under stars was highlight of the trip. Respect Aboriginal culture.",
            "review_image": "uluru_sunrise.jpg"
        },
        {
            "doc_id": "review_010",
            "title": "Mediterranean Cruise Delight",
            "content": "Mediterranean cruise hit perfect balance of relaxation and exploration. Ports in Greece, Italy, and Spain each offered unique cultural experiences.",
            "review_text": "Great value for money. Food on ship was excellent and shore excursions were well-organized. Would cruise again!",
            "review_image": "mediterranean_cruise.jpg"
        }
    ]
    
    # Test queries for different scenarios
    SAMPLE_QUERIES = [
        {
            "name": "sparse_text_query",
            "description": "Query targeting sparse text fields only",
            "text": "amazing food experience restaurant",
            "expected_fields": ["title_sparse", "content_sparse", "review_text_sparse"]
        },
        {
            "name": "multimodal_query", 
            "description": "Query with both text and image for dense fields",
            "text": "beautiful sunset mountain scenery",
            "image": "query_sunset.jpg",
            "expected_fields": ["title_dense", "content_dense", "review_text_dense", "review_image"]
        },
        {
            "name": "comprehensive_query",
            "description": "Query that works across all field types",
            "text": "travel adventure cultural experience",
            "expected_fields": ["all"]
        }
    ]
    
    @staticmethod
    def get_field_configs(test_dir: str) -> Dict[str, Any]:
        """Get field configurations for testing"""
        from asmr.index.config import FieldConfig, TokenizerType, RepresentationType
        
        return {
            # Title field configs (sparse + dense)
            "title_sparse": FieldConfig(
                name="title_sparse",
                tokenizer_type=TokenizerType.MORPH,
                representation_type=RepresentationType.SPARSE
            ),
            "title_dense": FieldConfig(
                name="title_dense",
                tokenizer_type=TokenizerType.HF_AUTO,
                representation_type=RepresentationType.DENSE,
                model_path="bert-base-uncased",
                faiss_index_path=os.path.join(test_dir, "title_dense.faiss")
            ),
            
            # Content field configs (sparse + dense)
            "content_sparse": FieldConfig(
                name="content_sparse",
                tokenizer_type=TokenizerType.MORPH,
                representation_type=RepresentationType.SPARSE
            ),
            "content_dense": FieldConfig(
                name="content_dense",
                tokenizer_type=TokenizerType.HF_AUTO,
                representation_type=RepresentationType.DENSE,
                model_path="bert-base-uncased",
                faiss_index_path=os.path.join(test_dir, "content_dense.faiss")
            ),
            
            # Review text field configs (sparse + dense)
            "review_text_sparse": FieldConfig(
                name="review_text_sparse",
                tokenizer_type=TokenizerType.MORPH,
                representation_type=RepresentationType.SPARSE
            ),
            "review_text_dense": FieldConfig(
                name="review_text_dense",
                tokenizer_type=TokenizerType.HF_AUTO,
                representation_type=RepresentationType.DENSE,
                model_path="bert-base-uncased",
                faiss_index_path=os.path.join(test_dir, "review_text_dense.faiss")
            ),
            
            # Review image field config (dense only)
            "review_image": FieldConfig(
                name="review_image",
                tokenizer_type=TokenizerType.HF_AUTO,  # Not used for images
                representation_type=RepresentationType.DENSE,
                model_path="clip-model",
                faiss_index_path=os.path.join(test_dir, "review_image.faiss")
            )
        }


class MockComponents:
    """Mock components for testing without external dependencies"""
    
    class MockBM25Index:
        """Mock BM25 index implementation"""
        def __init__(self):
            self.documents = []
            self.shape = (0,)
        
        def add_documents(self, documents):
            self.documents.extend(documents)
            self.shape = (len(self.documents),)
        
        def get_score(self, doc_id, terms):
            class MockTermScore:
                def __init__(self, score):
                    self.score = score
            
            class MockDocScore:
                def __init__(self, scores):
                    self.term_scores = [MockTermScore(s) for s in scores]
            
            # Simple mock scoring based on term overlap
            return MockDocScore([0.5] * len(terms))
    
    class MockEncoder:
        """Mock FDE encoder implementation"""
        def __init__(self):
            self.dimension = 768
        
        def encode_text(self, texts, prompt_type=None):
            import numpy as np
            # Return deterministic embeddings for consistent testing
            embeddings = np.random.RandomState(42).rand(len(texts), self.dimension).astype(np.float32)
            return embeddings
        
        def encode_image(self, images):
            import numpy as np
            # Return deterministic embeddings for consistent testing  
            embeddings = np.random.RandomState(123).rand(len(images), self.dimension).astype(np.float32)
            return embeddings
    
    class MockImage:
        """Mock PIL Image implementation"""
        def __init__(self, path=None):
            self.path = path
            self.size = (224, 224)
        
        @staticmethod
        def open(path):
            return MockComponents.MockImage(path)


def setup_test_environment():
    """Set up test environment with temporary directories"""
    test_dir = tempfile.mkdtemp(prefix="asmr_test_")
    return test_dir


def cleanup_test_environment(test_dir: str):
    """Clean up test environment"""
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)


def print_test_summary(test_name: str, results: Dict[str, Any]):
    """Print formatted test summary"""
    print(f"\\n{'='*50}")
    print(f"TEST SUMMARY: {test_name}")
    print(f"{'='*50}")
    
    for key, value in results.items():
        if isinstance(value, (list, dict)):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    
    print(f"{'='*50}")


# Export commonly used items
__all__ = [
    'TestConfig',
    'MockComponents', 
    'setup_test_environment',
    'cleanup_test_environment',
    'print_test_summary'
]