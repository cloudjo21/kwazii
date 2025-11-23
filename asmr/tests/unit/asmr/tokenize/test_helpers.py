import io
import pytest
from unittest.mock import Mock, patch
from PIL import Image

from asmr.tokenize.helpers import AutoTokenizerWrapper
from asmr.tokenize.helpers import ChunkProcessor
from asmr.tokenize.helpers import MorphTokenizerWrapper
from asmr.tokenize.helpers import SplitTokenizer
from asmr.tokenize.helpers import TokenizerWrapper
from asmr.tokenize.helpers import bytes_to_pil_image
from asmr.tokenize.helpers import split_text


class TestSplitText:
    """Test cases for split_text function"""

    def test_split_text_basic(self):
        """Test basic text splitting functionality"""

        text = "This is a simple test sentence"
        result = split_text(text, 20)

        # Should split into chunks that don't exceed max_length
        for chunk in result:
            assert len(chunk) <= 20

        # Rejoin should give back original text
        rejoined = ' '.join(result)
        assert rejoined == text

    def test_split_text_single_word_longer_than_max(self):
        """Test handling of single word longer than max_length"""
        text = "supercalifragilisticexpialidocious"
        result = split_text(text, 10)

        # Should return the word as is, even if longer than max_length
        assert result == [text]

    def test_split_text_exact_length(self):
        """Test text that exactly matches max_length"""
        text = "twelve chars"  # exactly 12 characters
        result = split_text(text, 12)

        assert result == [text]

    def test_split_text_empty_string(self):
        """Test handling of empty string"""
        result = split_text("", 10)
        assert result == []

    def test_split_text_single_word(self):
        """Test single word input"""
        text = "hello"
        result = split_text(text, 10)
        assert result == [text]

    def test_split_text_multiple_spaces(self):
        """Test handling of multiple spaces"""
        text = "word1    word2     word3"
        result = split_text(text, 20)

        # split() should normalize multiple spaces
        expected_words = ["word1", "word2", "word3"]
        rejoined_words = ' '.join(result).split()
        assert rejoined_words == expected_words


class TestSplitTokenizer:
    """Test cases for SplitTokenizer class"""

    def test_tokenize_basic(self):
        """Test basic tokenization"""

        tokenizer = SplitTokenizer()
        text = "Hello World Test"
        result = tokenizer.tokenize(text)

        assert result == ["hello", "world", "test"]

    def test_tokenize_mixed_case(self):
        """Test mixed case handling"""
        tokenizer = SplitTokenizer()
        text = "HeLLo WoRLd"
        result = tokenizer.tokenize(text)

        assert result == ["hello", "world"]

    def test_tokenize_empty_string(self):
        """Test empty string input"""
        tokenizer = SplitTokenizer()
        result = tokenizer.tokenize("")

        assert result == []

    def test_tokenize_single_word(self):
        """Test single word input"""
        tokenizer = SplitTokenizer()
        result = tokenizer.tokenize("Hello")

        assert result == ["hello"]


class TestAutoTokenizerWrapper:
    """Test cases for AutoTokenizerWrapper class"""

    @patch('asmr.tokenize.helpers.AutoTokenizer')
    def test_init(self, mock_auto_tokenizer):
        """Test initialization"""
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        wrapper = AutoTokenizerWrapper("test-model")

        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "test-model")
        assert wrapper.tokenizer == mock_tokenizer

    @patch('asmr.tokenize.helpers.AutoTokenizer')
    def test_tokenize(self, mock_auto_tokenizer):
        """Test tokenization"""
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = ["hello", "world"]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        wrapper = AutoTokenizerWrapper("test-model")
        result = wrapper.tokenize("hello world")

        mock_tokenizer.tokenize.assert_called_once_with("hello world")
        assert result == ["hello", "world"]


class TestTokenizerWrapper:
    """Test cases for TokenizerWrapper class"""

    def test_init(self):
        """Test initialization"""
        mock_tokenizer = Mock()
        wrapper = TokenizerWrapper(mock_tokenizer)

        assert wrapper.tokenizer == mock_tokenizer

    def test_tokenize(self):
        """Test tokenization"""
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = ["test", "tokens"]
        wrapper = TokenizerWrapper(mock_tokenizer)

        result = wrapper.tokenize("test input")

        mock_tokenizer.tokenize.assert_called_once_with("test input")
        assert result == ["test", "tokens"]

    @patch('asmr.tokenize.helpers.AutoTokenizerWrapper')
    def test_from_auto_tokenizer(self, mock_auto_wrapper_class):
        """Test factory method for AutoTokenizer"""
        mock_auto_wrapper = Mock()
        mock_auto_wrapper_class.return_value = mock_auto_wrapper

        result = TokenizerWrapper.from_auto_tokenizer("test-model")

        mock_auto_wrapper_class.assert_called_once_with("test-model")
        assert isinstance(result, TokenizerWrapper)
        assert result.tokenizer == mock_auto_wrapper

    def test_from_split_tokenizer(self):
        """Test factory method for SplitTokenizer"""
        result = TokenizerWrapper.from_split_tokenizer()

        assert isinstance(result, TokenizerWrapper)
        assert isinstance(result.tokenizer, SplitTokenizer)


class TestBytesToPilImage:
    """Test cases for bytes_to_pil_image function"""

    def test_bytes_to_pil_image_valid_image(self):
        """Test converting valid image bytes to PIL Image"""
        # Create a simple test image
        test_image = Image.new('RGB', (10, 10), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        result = bytes_to_pil_image(img_bytes)

        assert isinstance(result, Image.Image)
        assert result.size == (10, 10)

    def test_bytes_to_pil_image_invalid_bytes(self):
        """Test handling of invalid image bytes"""
        invalid_bytes = b"not an image"

        with pytest.raises(Exception):  # PIL will raise an exception
            bytes_to_pil_image(invalid_bytes)


class TestChunkProcessor:
    """Test cases for ChunkProcessor class"""

    def test_init_default(self):
        """Test default initialization"""
        processor = ChunkProcessor()
        assert processor.max_chunk_size == 512

    def test_init_custom_size(self):
        """Test initialization with custom chunk size"""
        processor = ChunkProcessor(max_chunk_size=256)
        assert processor.max_chunk_size == 256

    def test_chunk_text(self):
        """Test text chunking"""
        processor = ChunkProcessor(max_chunk_size=20)
        mock_tokenizer = Mock()

        text = "This is a test sentence for chunking"
        result = processor.chunk_text(text, mock_tokenizer)

        # Should use split_text internally
        assert isinstance(result, list)
        assert all(len(chunk) <= 20 for chunk in result)
        assert ' '.join(result) == text

    def test_chunk_image_from_pil(self):
        """Test image chunking from PIL Image"""
        processor = ChunkProcessor()
        test_image = Image.new('RGB', (448, 448), color='blue')

        result = processor.chunk_image(test_image, tile_size=(224, 224))

        # Should create 4 chunks (2x2 grid)
        assert len(result) == 4
        assert all(isinstance(chunk, Image.Image) for chunk in result)
        assert all(chunk.size == (224, 224) for chunk in result)

    def test_chunk_image_from_bytes(self):
        """Test image chunking from bytes"""
        processor = ChunkProcessor()

        # Create test image as bytes
        test_image = Image.new('RGB', (300, 200), color='green')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        result = processor.chunk_image(img_bytes, tile_size=(150, 100))

        # Should create 4 chunks (2x2 grid)
        assert len(result) == 4
        assert all(isinstance(chunk, Image.Image) for chunk in result)

    def test_chunk_image_uneven_dimensions(self):
        """Test image chunking with uneven dimensions"""
        processor = ChunkProcessor()
        test_image = Image.new('RGB', (250, 150), color='red')

        result = processor.chunk_image(test_image, tile_size=(224, 224))

        # 250x150 image with 224x224 tiles should create 2 chunks:
        # First: 224x150, Second: 26x150 (remaining width)
        assert len(result) == 2
        assert result[0].size == (224, 150)
        assert result[1].size == (26, 150)

    def test_chunk_image_small_image(self):
        """Test chunking of image smaller than tile size"""
        processor = ChunkProcessor()
        test_image = Image.new('RGB', (100, 100), color='yellow')

        result = processor.chunk_image(test_image, tile_size=(224, 224))

        # Should return single chunk with original size
        assert len(result) == 1
        assert result[0].size == (100, 100)


class TestMorphTokenizerWrapper:
    """Test cases for MorphTokenizerWrapper class"""

    def test_init_with_kiwi_available(self):
        """Test initialization when kiwipiepy is available"""
        # This test will only pass if kiwipiepy is installed
        try:
            wrapper = MorphTokenizerWrapper()
            assert hasattr(wrapper, 'kiwi')
            assert wrapper.kiwi is not None
        except ImportError:
            pytest.skip("kiwipiepy not available")

    @patch('asmr.tokenize.helpers.Kiwi', None)
    def test_init_without_kiwi(self):
        """Test initialization when kiwipiepy is not available"""
        with pytest.raises(ImportError, match="kiwipiepy is not installed"):
            MorphTokenizerWrapper()

    def test_tokenize_basic(self):
        """Test basic tokenization functionality"""
        try:
            wrapper = MorphTokenizerWrapper()
            text = "아버지가방에들어가신다"
            result = wrapper.tokenize(text)

            # Should return list of strings (token forms)
            assert isinstance(result, list)
            assert all(isinstance(token, str) for token in result)
            assert len(result) > 0

            # For this specific example, we expect certain tokens
            expected_forms = ['아버지', '가', '방', '에', '들어가', '시', 'ᆫ다']
            assert result == expected_forms

        except ImportError:
            pytest.skip("kiwipiepy not available")

    def test_tokenize_with_tags_basic(self):
        """Test tokenization with detailed tag information"""
        try:
            wrapper = MorphTokenizerWrapper()
            text = "아버지가방에들어가신다"
            result = wrapper.tokenize_with_tags(text)

            # Should return list of dictionaries
            assert isinstance(result, list)
            assert all(isinstance(token, dict) for token in result)
            assert len(result) > 0

            # Each token should have required keys
            for token in result:
                assert 'form' in token
                assert 'tag' in token
                assert 'start' in token
                assert 'len' in token
                assert isinstance(token['form'], str)
                assert isinstance(token['tag'], str)
                assert isinstance(token['start'], int)
                assert isinstance(token['len'], int)

            # Check specific expected values for the example
            expected_first_token = {
                'form': '아버지',
                'tag': 'NNG',
                'start': 0,
                'len': 3
            }
            assert result[0] == expected_first_token

        except ImportError:
            pytest.skip("kiwipiepy not available")

    def test_tokenize_empty_string(self):
        """Test tokenization of empty string"""
        try:
            wrapper = MorphTokenizerWrapper()
            result = wrapper.tokenize("")

            # Should return empty list
            assert isinstance(result, list)
            assert len(result) == 0

        except ImportError:
            pytest.skip("kiwipiepy not available")

    def test_tokenize_with_tags_empty_string(self):
        """Test tokenization with tags for empty string"""
        try:
            wrapper = MorphTokenizerWrapper()
            result = wrapper.tokenize_with_tags("")

            # Should return empty list
            assert isinstance(result, list)
            assert len(result) == 0

        except ImportError:
            pytest.skip("kiwipiepy not available")

    def test_tokenize_simple_sentence(self):
        """Test tokenization of a simple sentence"""
        try:
            wrapper = MorphTokenizerWrapper()
            text = "안녕하세요"
            result = wrapper.tokenize(text)

            # Should return non-empty list
            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(token, str) for token in result)

        except ImportError:
            pytest.skip("kiwipiepy not available")
