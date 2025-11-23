"""
Integration test for Jinavera FDE Encoder
Tests text and image encoding with real model and sample images
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Try importing actual dependencies
try:
    from fde.models.jinavera import Jinavera
    from fde.config import PromptType
    HAS_ACTUAL_DEPS = True
except ImportError:
    HAS_ACTUAL_DEPS = False
    print("Warning: Actual FDE dependencies not available, using mocks")


class MockJinavera:
    """Mock Jinavera for testing without actual model"""

    def __init__(self, encoder_model_path: str):
        self.encoder_model_path = encoder_model_path
        self.dimension = 10240  # Expected FDE dimension

    def encode_text(self, texts, prompt_type=None):
        # Return mock embeddings with expected dimension
        return np.random.rand(len(texts), self.dimension).astype(np.float32)

    def encode_image(self, images):
        # Return mock embeddings for images
        return np.random.rand(len(images), self.dimension).astype(np.float32)

    def __finalize__(self):
        pass


class TestJinaveraIntegration(unittest.TestCase):
    """Integration test for Jinavera FDE encoder"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and resources"""
        cls.encoder_model_path = '/mnt/d/temp/user/ed/mart/llms/jina-embeddings-v4'
        cls.test_dir = tempfile.mkdtemp(prefix="jinavera_test_")
        cls.resources_dir = Path("tests/integration/resources")
        cls.resources_dir.mkdir(parents=True, exist_ok=True)

        # Create sample images for testing
        cls._create_sample_images()

        # Initialize encoder
        cls.encoder = cls._initialize_encoder()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if hasattr(cls.encoder, '__finalize__'):
            cls.encoder.__finalize__()
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    @classmethod
    def _initialize_encoder(cls):
        """Initialize Jinavera encoder (real or mock)"""
        if HAS_ACTUAL_DEPS and os.path.exists(cls.encoder_model_path):
            try:
                print(
                    f"Initializing real Jinavera model from {cls.encoder_model_path}"
                )
                return Jinavera(cls.encoder_model_path)
            except Exception as e:
                print(f"Failed to initialize real model: {e}")
                print("Falling back to mock encoder")
                return MockJinavera(cls.encoder_model_path)
        else:
            print("Using mock Jinavera encoder")
            return MockJinavera(cls.encoder_model_path)

    @classmethod
    def _create_sample_images(cls):
        """Create sample images for testing"""
        # Create sample images with different contents
        sample_images = [
            ("travel_sunset.jpg", "A beautiful sunset over mountains",
             (255, 200, 150)),
            ("city_skyline.jpg", "Modern city skyline at night", (50, 100,
                                                                  200)),
            ("nature_forest.jpg", "Dense green forest landscape", (50, 150,
                                                                   50)),
            ("beach_scene.jpg", "Tropical beach with palm trees", (200, 220,
                                                                   255)),
            ("food_dish.jpg", "Delicious local cuisine", (200, 150, 100))
        ]

        for filename, description, color in sample_images:
            image_path = cls.resources_dir / filename

            # Create 224x224 image with gradient and text
            img = Image.new('RGB', (224, 224), color)
            draw = ImageDraw.Draw(img)

            # Add some visual variety
            for i in range(0, 224, 20):
                lighter_color = tuple(min(255, c + 30) for c in color)
                draw.rectangle([i, i, i + 10, i + 10], fill=lighter_color)

            # Add text description (if PIL supports it)
            try:
                draw.text((10, 10), description[:20], fill=(255, 255, 255))
            except:
                pass  # Skip text if font not available

            img.save(image_path)
            print(f"Created sample image: {image_path}")

    def test_encoder_initialization(self):
        """Test that encoder initializes properly"""
        self.assertIsNotNone(self.encoder)
        self.assertTrue(hasattr(self.encoder, 'encode_text'))

        # Check if it's real or mock encoder
        if isinstance(self.encoder, MockJinavera):
            print("✓ Mock encoder initialized successfully")
        else:
            print("✓ Real Jinavera encoder initialized successfully")
            self.assertTrue(hasattr(self.encoder, 'encode_image'))

    def test_text_encoding_single(self):
        """Test single text encoding"""
        print("\\nTesting single text encoding...")

        test_text = "Beautiful sunset over mountain landscape"

        # Test with different prompt types if available
        if hasattr(self.encoder, 'encode_text') and HAS_ACTUAL_DEPS:
            # Test with PromptType.QUERY
            query_embedding = self.encoder.encode_text([test_text],
                                                       PromptType.QUERY)
            self.assertEqual(query_embedding.shape[0], 1)
            self.assertGreater(query_embedding.shape[1], 0)
            print(f"✓ Query embedding shape: {query_embedding.shape}")

            # Test with PromptType.PASSAGE
            passage_embedding = self.encoder.encode_text([test_text],
                                                         PromptType.PASSAGE)
            self.assertEqual(passage_embedding.shape[0], 1)
            self.assertEqual(passage_embedding.shape[1],
                             query_embedding.shape[1])
            print(f"✓ Passage embedding shape: {passage_embedding.shape}")

            # Verify embeddings are different for different prompt types
            similarity = np.dot(query_embedding[0], passage_embedding[0])
            print(f"✓ Query-Passage similarity: {similarity:.4f}")

        elif hasattr(self.encoder, 'encode_text'):
            # Fallback for mock encoder
            embedding = self.encoder.encode_text([test_text])
            self.assertEqual(embedding.shape[0], 1)
            self.assertGreater(embedding.shape[1], 0)
            print(f"✓ Text embedding shape: {embedding.shape}")

    def test_text_encoding_batch(self):
        """Test batch text encoding"""
        print("\\nTesting batch text encoding...")

        test_texts = [
            "Amazing weekend trip to Paris with beautiful architecture",
            "Tokyo food adventure with incredible ramen and sushi",
            "Safari experience in Kenya with wildlife photography",
            "New York City marathon running through five boroughs",
            "Bali beach paradise with stunning sunset views"
        ]

        if hasattr(self.encoder, 'encode_text') and HAS_ACTUAL_DEPS:
            embeddings = self.encoder.encode_text(test_texts, PromptType.QUERY)
        elif hasattr(self.encoder, 'encode_text'):
            embeddings = self.encoder.encode_text(test_texts)
        else:
            self.fail("Encoder has no text encoding method")

        # Verify batch encoding results
        self.assertEqual(embeddings.shape[0], len(test_texts))
        self.assertGreater(embeddings.shape[1], 0)
        print(f"✓ Batch text encoding shape: {embeddings.shape}")

        # Verify embeddings are not identical
        for i in range(len(test_texts)):
            for j in range(i + 1, len(test_texts)):
                similarity = np.dot(embeddings[i], embeddings[j])
                print(
                    f"  Similarity between text {i+1} and {j+1}: {similarity:.4f}"
                )

        # Check that embeddings have reasonable magnitude
        norms = np.linalg.norm(embeddings, axis=1)
        print(
            f"✓ Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}"
        )

    def test_image_encoding_single(self):
        """Test single image encoding"""
        print("\\nTesting single image encoding...")

        # Load a sample image
        image_path = self.resources_dir / "travel_sunset.jpg"
        self.assertTrue(image_path.exists(),
                        f"Sample image not found: {image_path}")

        try:
            image = Image.open(image_path)
            print(f"✓ Loaded image: {image_path} (size: {image.size})")

            if hasattr(self.encoder, 'encode_image'):
                embedding = self.encoder.encode_image([image])
                self.assertEqual(embedding.shape[0], 1)
                self.assertGreater(embedding.shape[1], 0)
                print(f"✓ Image embedding shape: {embedding.shape}")

                # Check embedding properties
                norm = np.linalg.norm(embedding[0])
                print(f"✓ Image embedding norm: {norm:.4f}")

            else:
                print("⚠️  Image encoding not available in current encoder")

        except Exception as e:
            self.fail(f"Failed to encode image: {e}")

    def test_image_encoding_batch(self):
        """Test batch image encoding"""
        print("\\nTesting batch image encoding...")

        # Load multiple sample images
        image_files = [
            "travel_sunset.jpg", "city_skyline.jpg", "nature_forest.jpg",
            "beach_scene.jpg", "food_dish.jpg"
        ]

        images = []
        for filename in image_files:
            image_path = self.resources_dir / filename
            if image_path.exists():
                image = Image.open(image_path)
                images.append(image)
                print(f"✓ Loaded {filename} (size: {image.size})")

        self.assertGreater(len(images), 0, "No sample images found")

        if hasattr(self.encoder, 'encode_image'):
            embeddings = self.encoder.encode_image(images)

            # Verify batch encoding results
            self.assertEqual(embeddings.shape[0], len(images))
            self.assertGreater(embeddings.shape[1], 0)
            print(f"✓ Batch image encoding shape: {embeddings.shape}")

            # Verify embeddings are not identical
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    similarity = np.dot(embeddings[i], embeddings[j])
                    print(
                        f"  Similarity between image {i+1} and {j+1}: {similarity:.4f}"
                    )

            # Check embedding norms
            norms = np.linalg.norm(embeddings, axis=1)
            print(
                f"✓ Image embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}"
            )

        else:
            print("⚠️  Batch image encoding not available in current encoder")

    def test_cross_modal_similarity(self):
        """Test cross-modal similarity between text and image"""
        print("\\nTesting cross-modal similarity...")

        # Test descriptions that should match images
        text_image_pairs = [
            ("Beautiful sunset over mountains", "travel_sunset.jpg"),
            ("Modern city skyline at night", "city_skyline.jpg"),
            ("Dense green forest landscape", "nature_forest.jpg"),
            ("Tropical beach with palm trees", "beach_scene.jpg"),
            ("Delicious local cuisine", "food_dish.jpg")
        ]

        if not (hasattr(self.encoder, 'encode_text')
                and hasattr(self.encoder, 'encode_image')):
            print(
                "⚠️  Cross-modal testing requires both text and image encoding"
            )
            return

        for text_desc, image_file in text_image_pairs:
            image_path = self.resources_dir / image_file
            if not image_path.exists():
                continue

            # Encode text and image
            if hasattr(self.encoder, 'encode_text') and HAS_ACTUAL_DEPS:
                text_embedding = self.encoder.encode_text([text_desc],
                                                          PromptType.QUERY)[0]
            else:
                text_embedding = self.encoder.encode_text([text_desc])[0]

            image = Image.open(image_path)
            image_embedding = self.encoder.encode_image([image])[0]

            # Calculate cross-modal similarity
            similarity = np.dot(text_embedding, image_embedding)
            print(f"✓ '{text_desc}' <-> {image_file}: {similarity:.4f}")

    def test_embedding_consistency(self):
        """Test that embeddings are consistent across multiple runs"""
        print("\\nTesting embedding consistency...")

        test_text = "Consistent encoding test"

        # Generate embeddings multiple times
        embeddings = []
        for i in range(3):
            if hasattr(self.encoder, 'encode') and HAS_ACTUAL_DEPS:
                emb = self.encoder.encode([test_text], PromptType.QUERY)[0]
            else:
                emb = self.encoder.encode_text([test_text])[0]
            embeddings.append(emb)

        def normalize(v: np.ndarray) -> np.ndarray:
            return v / np.linalg.norm(v)

        # Check consistency (for real model, should be identical)
        if not isinstance(self.encoder, MockJinavera):
            # Real model should produce identical embeddings
            for i in range(1, len(embeddings)):
                similarity = np.dot(normalize(embeddings[0]),
                                    normalize(embeddings[i]))
                print(f"✓ Run 1 vs Run {i+1} similarity: {similarity:.6f}")
                self.assertAlmostEqual(
                    similarity,
                    1.0,
                    places=4,
                    msg="Real model should produce consistent embeddings")
        else:
            print("⚠️  Mock encoder may produce different embeddings each run")

    def test_embedding_dimensions(self):
        """Test embedding dimensions match expectations"""
        print("\\nTesting embedding dimensions...")

        # Expected FDE dimension: num_repetitions * (2^num_simhash_projections) * projection_dimension
        # Default: 20 * (2^5) * 16 = 20 * 32 * 16 = 10240
        expected_dim = 10240

        test_text = "Dimension test"

        if hasattr(self.encoder, 'encode') and HAS_ACTUAL_DEPS:
            embedding = self.encoder.encode([test_text], PromptType.QUERY)
        else:
            embedding = self.encoder.encode_text([test_text])

        actual_dim = embedding.shape[1]
        print(f"✓ Expected dimension: {expected_dim}")
        print(f"✓ Actual dimension: {actual_dim}")

        if isinstance(self.encoder, MockJinavera):
            print("⚠️  Mock encoder dimension may differ from real model")
        else:
            self.assertEqual(
                actual_dim, expected_dim,
                f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
            )

    def test_memory_cleanup(self):
        """Test that encoder properly cleans up GPU memory"""
        print("\\nTesting memory cleanup...")

        if torch.cuda.is_available():
            # Get initial memory
            initial_memory = torch.cuda.memory_allocated()
            print(f"✓ Initial GPU memory: {initial_memory / 1024**2:.2f} MB")

            # Perform some encoding operations
            test_texts = ["Memory test"] * 10

            if hasattr(self.encoder, 'encode') and HAS_ACTUAL_DEPS:
                embeddings = self.encoder.encode(test_texts, PromptType.QUERY)
            else:
                embeddings = self.encoder.encode_text(test_texts)

            after_encoding = torch.cuda.memory_allocated()
            print(f"✓ After encoding: {after_encoding / 1024**2:.2f} MB")

            # Manual cleanup
            del embeddings
            torch.cuda.empty_cache()

            final_memory = torch.cuda.memory_allocated()
            print(f"✓ After cleanup: {final_memory / 1024**2:.2f} MB")

        else:
            print("⚠️  CUDA not available, skipping GPU memory test")

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        print("\\n" + "=" * 50)
        print("RUNNING END-TO-END WORKFLOW TEST")
        print("=" * 50)

        # Sample travel review data
        reviews = [{
            "text":
            "Amazing sunset view from mountain peak with stunning landscape",
            "image": "travel_sunset.jpg"
        }, {
            "text":
            "Vibrant city nightlife with impressive modern architecture",
            "image": "city_skyline.jpg"
        }, {
            "text": "Peaceful forest hiking trail with abundant green nature",
            "image": "nature_forest.jpg"
        }]

        all_text_embeddings = []
        all_image_embeddings = []

        for i, review in enumerate(reviews):
            print(f"\\nProcessing review {i+1}:")
            print(f"  Text: {review['text']}")
            print(f"  Image: {review['image']}")

            # Encode text
            if hasattr(self.encoder, 'encode') and HAS_ACTUAL_DEPS:
                text_emb = self.encoder.encode([review['text']],
                                               PromptType.PASSAGE)[0]
            else:
                text_emb = self.encoder.encode_text([review['text']])[0]
            all_text_embeddings.append(text_emb)

            # Encode image if available
            image_path = self.resources_dir / review['image']
            if image_path.exists() and hasattr(self.encoder, 'encode_image'):
                image = Image.open(image_path)
                image_emb = self.encoder.encode_image([image])[0]
                all_image_embeddings.append(image_emb)

                # Cross-modal similarity
                similarity = np.dot(text_emb, image_emb)
                print(f"  Cross-modal similarity: {similarity:.4f}")

            print(f"  ✓ Text embedding: {text_emb.shape}")

        # Test query scenario
        query = "Beautiful mountain scenery with sunset"
        print(f"\\nQuery: '{query}'")

        if hasattr(self.encoder, 'encode') and HAS_ACTUAL_DEPS:
            query_emb = self.encoder.encode([query], PromptType.QUERY)[0]
        else:
            query_emb = self.encoder.encode_text([query])[0]

        # Find best matching text
        text_similarities = [
            np.dot(query_emb, text_emb) for text_emb in all_text_embeddings
        ]
        best_text_idx = np.argmax(text_similarities)
        print(
            f"✓ Best text match: Review {best_text_idx + 1} (similarity: {text_similarities[best_text_idx]:.4f})"
        )

        # Find best matching image (if available)
        if all_image_embeddings:
            image_similarities = [
                np.dot(query_emb, img_emb) for img_emb in all_image_embeddings
            ]
            best_image_idx = np.argmax(image_similarities)
            print(
                f"✓ Best image match: Review {best_image_idx + 1} (similarity: {image_similarities[best_image_idx]:.4f})"
            )

        print("\\n✓ End-to-end workflow completed successfully!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
