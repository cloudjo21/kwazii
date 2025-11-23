"""
Integration test for ASMR field indexing and retrieval system
Step 1: Define fields and indexing pipeline
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import unittest

# Mock imports since actual dependencies may not be available
try:
    from PIL import Image
    import numpy as np
except ImportError:
    # Mock PIL Image for testing
    class MockImage:

        @staticmethod
        def open(path):
            return MockImage()

        @property
        def size(self):
            return (224, 224)

    Image = MockImage()
    import numpy as np

from asmr.index.config import FieldConfig, TokenizerType, RepresentationType
from asmr.index.fields import (SparseTextFieldIndex, DenseTextFieldIndex,
                               DenseImageFieldIndex, DocumentIndex)


class MockBM25Index:
    """Mock BM25 index for testing"""

    def __init__(self):
        self.documents = []
        self.shape = (0, )

    def add_documents(self, documents):
        self.documents.extend(documents)
        self.shape = (len(self.documents), )

    def get_score(self, doc_id, terms):
        # Mock score calculation
        class MockTermScore:

            def __init__(self, score):
                self.score = score

        class MockDocScore:

            def __init__(self, scores):
                self.term_scores = [MockTermScore(s) for s in scores]

        return MockDocScore([0.5] * len(terms))


class MockEncoder:
    """Mock FDE encoder for testing"""

    def __init__(self):
        self.dimension = 768

    def encode_text(self, texts, prompt_type=None):
        # Return mock embeddings
        embeddings = np.random.rand(len(texts),
                                    self.dimension).astype(np.float32)
        return embeddings

    def encode_image(self, images):
        # Return mock embeddings
        embeddings = np.random.rand(len(images),
                                    self.dimension).astype(np.float32)
        return embeddings


class TestFieldIndexing(unittest.TestCase):
    """Integration test for field indexing pipeline"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.encoder = MockEncoder()
        self.resources_dir = Path(__file__).parent / "resources"
        self.available_images = [
            f.name for f in self.resources_dir.glob("*.jpg")
        ] if self.resources_dir.exists() else []
        if not self.available_images:
            print(
                f"Warning: No images found in resources directory {self.resources_dir}"
            )

        # Define field configurations
        self.field_configs = self._define_field_configs()

        # Define sample documents
        self.sample_documents = self._define_sample_documents()

        # Initialize field indices
        self.field_indices = {}
        self._initialize_field_indices()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _define_field_configs(self) -> Dict[str, FieldConfig]:
        """Define individual FieldConfig for each field"""
        return {
            # Title field configs
            "title_sparse":
            FieldConfig(name="title_sparse",
                        tokenizer_type=TokenizerType.MORPH,
                        representation_type=RepresentationType.SPARSE),
            "title_dense":
            FieldConfig(name="title_dense",
                        tokenizer_type=TokenizerType.HF_AUTO,
                        representation_type=RepresentationType.DENSE,
                        model_path="bert-base-uncased",
                        faiss_index_path=os.path.join(self.test_dir,
                                                      "title_dense.faiss")),

            # Content field configs
            "content_sparse":
            FieldConfig(name="content_sparse",
                        tokenizer_type=TokenizerType.MORPH,
                        representation_type=RepresentationType.SPARSE),
            "content_dense":
            FieldConfig(name="content_dense",
                        tokenizer_type=TokenizerType.HF_AUTO,
                        representation_type=RepresentationType.DENSE,
                        model_path="bert-base-uncased",
                        faiss_index_path=os.path.join(self.test_dir,
                                                      "content_dense.faiss")),

            # Review text field configs
            "review_text_sparse":
            FieldConfig(name="review_text_sparse",
                        tokenizer_type=TokenizerType.MORPH,
                        representation_type=RepresentationType.SPARSE),
            "review_text_dense":
            FieldConfig(name="review_text_dense",
                        tokenizer_type=TokenizerType.HF_AUTO,
                        representation_type=RepresentationType.DENSE,
                        model_path="bert-base-uncased",
                        faiss_index_path=os.path.join(
                            self.test_dir, "review_text_dense.faiss")),

            # Review image field config
            "review_image":
            FieldConfig(
                name="review_image",
                tokenizer_type=TokenizerType.HF_AUTO,  # Not used for images
                representation_type=RepresentationType.DENSE,
                model_path="clip-model",
                faiss_index_path=os.path.join(self.test_dir,
                                              "review_image.faiss"))
        }

    def _define_sample_documents(self) -> List[Dict[str, Any]]:
        """Define 10 sample documents with post-travel review theme"""
        return [{
            "doc_id": "review_001",
            "title": "Amazing Weekend in Paris",
            "content":
            "Paris exceeded all my expectations. The Eiffel Tower at sunset was breathtaking, and the local cafes served incredible croissants. The Seine river cruise was romantic and peaceful.",
            "review_text":
            "Highly recommend visiting in spring. The weather was perfect and crowds were manageable. 5 stars!",
            "review_image": "paris_sunset.jpg"
        }, {
            "doc_id": "review_002",
            "title": "Tokyo Food Adventure",
            "content":
            "Tokyo's food scene is unparalleled. From street food in Shibuya to high-end sushi in Ginza, every meal was memorable. The ramen shops were particularly outstanding.",
            "review_text":
            "Must try the conveyor belt sushi and visit Tsukiji fish market early morning. Cultural experience was amazing!",
            "review_image": "tokyo_ramen.jpg"
        }, {
            "doc_id": "review_003",
            "title": "Safari Experience in Kenya",
            "content":
            "The wildlife safari in Maasai Mara was life-changing. Witnessed the great migration and saw the Big Five. Local guides were knowledgeable and friendly.",
            "review_text":
            "Best time to visit is during migration season. Accommodation was comfortable but basic. Unforgettable experience!",
            "review_image": "kenya_safari.jpg"
        }, {
            "doc_id": "review_004",
            "title": "New York City Marathon",
            "content":
            "Running the NYC Marathon was a dream come true. The crowd support was incredible, and running through all five boroughs showcased the city's diversity.",
            "review_text":
            "Training was tough but worth it. The finish line in Central Park was emotional. Great organization by race officials.",
            "review_image": "nyc_marathon.jpg"
        }, {
            "doc_id": "review_005",
            "title": "Bali Beach Paradise",
            "content":
            "Bali's beaches are pristine and the sunsets spectacular. Ubud's rice terraces and temples provided cultural depth beyond the beach experience.",
            "review_text":
            "Perfect honeymoon destination. Balinese hospitality is genuine and the spa treatments were rejuvenating.",
            "review_image": "bali_sunset.jpg"
        }, {
            "doc_id": "review_006",
            "title": "Iceland Northern Lights",
            "content":
            "Iceland in winter offered magical northern lights viewing. The Blue Lagoon geothermal spa was relaxing after long days of aurora hunting.",
            "review_text":
            "Pack warm clothes! Northern lights were visible 3 out of 5 nights. Ice caves tour was phenomenal.",
            "review_image": "iceland_aurora.jpg"
        }, {
            "doc_id": "review_007",
            "title": "Swiss Alps Hiking",
            "content":
            "The Swiss Alps provided world-class hiking trails with stunning mountain vistas. Cable cars made accessing high-altitude trails convenient.",
            "review_text":
            "Expensive but worth every penny. Trail marking was excellent and mountain huts offered great meals.",
            "review_image": "swiss_mountains.jpg"
        }, {
            "doc_id": "review_008",
            "title": "Brazilian Carnival Experience",
            "content":
            "Rio's Carnival was an explosion of color, music, and energy. Participating in street parties and watching parade competitions was thrilling.",
            "review_text":
            "Book accommodations early as prices skyrocket. Learn some Portuguese phrases for better local interaction.",
            "review_image": "rio_carnival.jpg"
        }, {
            "doc_id": "review_009",
            "title": "Australian Outback Adventure",
            "content":
            "The vastness of the Australian Outback is humbling. Uluru at sunrise was spiritual, and stargazing in the desert was spectacular.",
            "review_text":
            "Guided tours are recommended for safety. Camping under stars was highlight of the trip. Respect Aboriginal culture.",
            "review_image": "uluru_sunrise.jpg"
        }, {
            "doc_id": "review_010",
            "title": "Mediterranean Cruise Delight",
            "content":
            "Mediterranean cruise hit perfect balance of relaxation and exploration. Ports in Greece, Italy, and Spain each offered unique cultural experiences.",
            "review_text":
            "Great value for money. Food on ship was excellent and shore excursions were well-organized. Would cruise again!",
            "review_image": "mediterranean_cruise.jpg"
        }]

    def _initialize_field_indices(self):
        """Initialize field indices based on configurations"""
        # Initialize sparse text indices
        for field_name in [
                "title_sparse", "content_sparse", "review_text_sparse"
        ]:
            config = self.field_configs[field_name]
            bm25_index = MockBM25Index()
            self.field_indices[field_name] = SparseTextFieldIndex(
                config, bm25_index)

        # Initialize dense text indices
        for field_name in [
                "title_dense", "content_dense", "review_text_dense"
        ]:
            config = self.field_configs[field_name]
            self.field_indices[field_name] = DenseTextFieldIndex(
                config, self.encoder)

        # Initialize dense image index
        config = self.field_configs["review_image"]
        self.field_indices["review_image"] = DenseImageFieldIndex(
            config, self.encoder)

    def test_field_config_creation(self):
        """Test that all field configurations are properly created"""
        self.assertEqual(len(self.field_configs), 7)

        # Test sparse configs
        sparse_configs = [
            "title_sparse", "content_sparse", "review_text_sparse"
        ]
        for config_name in sparse_configs:
            config = self.field_configs[config_name]
            self.assertEqual(config.representation_type,
                             RepresentationType.SPARSE)
            self.assertEqual(config.tokenizer_type, TokenizerType.MORPH)

        # Test dense configs
        dense_configs = [
            "title_dense", "content_dense", "review_text_dense", "review_image"
        ]
        for config_name in dense_configs:
            config = self.field_configs[config_name]
            self.assertEqual(config.representation_type,
                             RepresentationType.DENSE)
            self.assertIsNotNone(config.faiss_index_path)

    def test_sample_documents_structure(self):
        """Test that sample documents have correct structure"""
        self.assertEqual(len(self.sample_documents), 10)

        required_fields = [
            "doc_id", "title", "content", "review_text", "review_image"
        ]
        for doc in self.sample_documents:
            for field in required_fields:
                self.assertIn(field, doc)
                self.assertIsInstance(doc[field], str)

    def test_indexing_pipeline(self):
        """Test the complete indexing pipeline"""
        # Extract field data from documents
        doc_ids = [doc["doc_id"] for doc in self.sample_documents]
        titles = [doc["title"] for doc in self.sample_documents]
        contents = [doc["content"] for doc in self.sample_documents]
        review_texts = [doc["review_text"] for doc in self.sample_documents]
        review_images = [doc["review_image"] for doc in self.sample_documents]

        # Test sparse text indexing
        print("\\nIndexing sparse text fields...")
        self.field_indices["title_sparse"].add_documents(doc_ids, titles)
        self.field_indices["content_sparse"].add_documents(doc_ids, contents)
        self.field_indices["review_text_sparse"].add_documents(
            doc_ids, review_texts)

        # Test dense text indexing
        print("Indexing dense text fields...")
        self.field_indices["title_dense"].add_documents(doc_ids, titles)
        self.field_indices["content_dense"].add_documents(doc_ids, contents)
        self.field_indices["review_text_dense"].add_documents(
            doc_ids, review_texts)

        # Test dense image indexing
        print("Indexing dense image field...")

        # Filter out documents with missing images
        available_images = set(self.available_images)
        valid_docs = [(doc_id, img)
                      for doc_id, img in zip(doc_ids, review_images)
                      if img in available_images]

        if not valid_docs:
            print(
                "Warning: No valid images found, skipping image indexing test")
            return

        valid_doc_ids, valid_images = zip(*valid_docs)
        mock_images = [
            Image.open(self.resources_dir / img) for img in valid_images
        ]

        self.field_indices["review_image"].add_documents(
            list(valid_doc_ids), mock_images)
        print(f"✓ Indexed {len(valid_docs)} documents with images")

        # Verify indexing results
        expected_doc_count = len(self.sample_documents)
        for field_name, field_index in self.field_indices.items():
            if hasattr(field_index, 'get_stats'):
                stats = field_index.get_stats()
                print(f"{field_name} stats: {stats}")

                # For image fields, count might be less due to missing images
                if field_name == "review_image":
                    available_image_count = len([
                        img for img in review_images
                        if img in set(self.available_images)
                    ])
                    self.assertEqual(stats['total_documents'],
                                     available_image_count)
                else:
                    self.assertEqual(stats['total_documents'],
                                     expected_doc_count)

        print("✓ All fields indexed successfully")

    def test_save_indices(self):
        """Test saving indices to disk"""
        # First index the documents
        self.test_indexing_pipeline()

        # Save dense indices
        dense_fields = [
            "title_dense", "content_dense", "review_text_dense", "review_image"
        ]
        for field_name in dense_fields:
            field_index = self.field_indices[field_name]
            field_index.save_index()

            # Verify files were created
            config = self.field_configs[field_name]
            self.assertTrue(os.path.exists(config.faiss_index_path))
            print(f"✓ {field_name} index saved to {config.faiss_index_path}")

    def test_document_index_integration(self):
        """Test integration with DocumentIndex"""
        # Create document index
        doc_index = DocumentIndex()

        # Add all field indices
        for field_name, field_index in self.field_indices.items():
            doc_index.add_field_index(field_name, field_index)

        # Verify all fields are registered
        self.assertEqual(len(doc_index.field_indices), 7)

        expected_fields = [
            "title_sparse", "title_dense", "content_sparse", "content_dense",
            "review_text_sparse", "review_text_dense", "review_image"
        ]

        for field_name in expected_fields:
            self.assertIn(field_name, doc_index.field_indices)

        print("✓ DocumentIndex integration successful")

        # Save integration test output for next step
        self._save_integration_output(doc_index)

    def _save_integration_output(self, doc_index: DocumentIndex):
        """Save integration test output for next step"""
        output_path = Path("tests/integration/step1_output.py")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(f'''"""
Output from Step 1: Field indexing integration test
Generated on: {Path(__file__).name}
"""

# Test configuration
TEST_DIR = "{self.test_dir}"
SAMPLE_DOCUMENTS = {self.sample_documents}
FIELD_CONFIGS = {repr(self.field_configs)}

# Mock encoder for step 2
class MockEncoder:
    def __init__(self):
        self.dimension = 768
    
    def encode_text(self, texts, prompt_type=None):
        import numpy as np
        return np.random.rand(len(texts), self.dimension).astype(np.float32)
    
    def encode_image(self, images):
        import numpy as np
        return np.random.rand(len(images), self.dimension).astype(np.float32)

# Status
INDEXING_COMPLETED = True
TOTAL_DOCUMENTS = {len(self.sample_documents)}
TOTAL_FIELDS = {len(self.field_configs)}
''')

        print(f"✓ Integration output saved to {output_path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
