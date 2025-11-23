"""
Example usage of the updated Query class with data types and multimodal support
"""

from asmr.retrieve.query import Query, QueryContent
from asmr.index.config import FieldConfig, TokenizerType, RepresentationType
from PIL import Image

# Example 1: Text-only query with data type
text_query = Query.from_text("machine learning algorithms", data_type="text")
print(f"Has text: {text_query.has_text()}")
print(f"Has image: {text_query.has_image()}")
print(f"Is multimodal: {text_query.is_multimodal()}")
print(f"Text data type: {text_query.get_text_data_type()}")

# Example 2: Image-only query
# image_query = Query.from_image("/path/to/image.jpg", data_type="image")
# print(f"Image data type: {image_query.get_image_data_type()}")

# Example 3: Multimodal query (both text and image)
"""
multimodal_query = Query.from_multimodal(
    text="neural network architecture diagram",
    image="/path/to/diagram.png",
    text_data_type="text",
    image_data_type="image"
)
print(f"Multimodal query - has both: {multimodal_query.is_multimodal()}")
"""

# Example 4: Manual construction with QueryContent
text_content = QueryContent("deep learning", data_type="text")
query_manual = Query(text=text_content)

# Example 5: Field configuration and target field detection
field_configs = {
    "title": FieldConfig(
        name="title",
        tokenizer_type=TokenizerType.SPLIT,
        representation_type=RepresentationType.SPARSE
    ),
    "content": FieldConfig(
        name="content", 
        tokenizer_type=TokenizerType.HF_AUTO,
        representation_type=RepresentationType.DENSE,
        model_path="bert-base-uncased",
        faiss_index_path="/path/to/content.index"
    ),
    "image_field": FieldConfig(
        name="image_field",
        tokenizer_type=TokenizerType.HF_AUTO,
        representation_type=RepresentationType.DENSE,
        model_path="clip-model",
        faiss_index_path="/path/to/image.index"
    )
}

# Get target fields for text query
target_fields = text_query.get_target_fields(field_configs)
print(f"Target fields for text query: {[f.name for f in target_fields]}")

"""
Example usage with retrievers:

# Setup retrievers
retriever = QueryRouter(field_retrievers)

# 1. Traditional retrieval with specific field
results = retriever.retrieve("content", text_query, k=10)

# 2. Smart retrieval using target field detection
smart_results = retriever.retrieve_with_field_configs(text_query, field_configs, k=10)

# 3. Multimodal retrieval
multimodal_results = retriever.smart_retrieve(multimodal_query, field_configs, k=10)

# 4. Multi-field retrieval
multi_results = retriever.multi_field_retrieve(text_query, ["title", "content"], k=10)
"""

print("Updated Query class with data types and multimodal support ready!")