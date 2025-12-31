#!/usr/bin/env python3
"""
Example usage of refactored field indices with FAISS and batch indexing
"""

import sys
from pathlib import Path

# Add the source directories to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root / "asmr" / "src"))
sys.path.insert(0, str(project_root / "fde" / "src"))

try:
    from asmr.index.config import FieldConfig, TokenizerType, RepresentationType

    # from fde.models.jinavera import Jinavera  # Comment out for now
    # from PIL import Image  # Comment out for now
    print("✓ Successfully imported required modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def main():
    print("=== ASMR Index Example Usage ===\n")

    # Example 1: Sparse Text Field Index (this should work)
    print("1. Creating Sparse Text Field Index...")
    try:
        sparse_config = FieldConfig(
            name="titles",
            tokenizer_type=TokenizerType.SPLIT,
            representation_type=RepresentationType.SPARSE,
        )
        print(f"✓ Created sparse config: {sparse_config}")
    except Exception as e:
        print(f"✗ Error creating sparse config: {e}")
        return

    # For now, we'll just demonstrate the configuration works
    print(f"✓ Field name: {sparse_config.name}")
    print(f"✓ Tokenizer type: {sparse_config.tokenizer_type}")
    print(f"✓ Representation type: {sparse_config.representation_type}")

    print("\n2. Testing TokenizerType and RepresentationType enums...")
    print(f"✓ Available tokenizer types: {[t.value for t in TokenizerType]}")
    print(f"✓ Available representation types: {[r.value for r in RepresentationType]}")

    print("\n3. Testing configuration validation...")
    try:
        # This should fail - HF_AUTO requires model_path
        invalid_config = FieldConfig(
            name="test",
            tokenizer_type=TokenizerType.HF_AUTO,
            representation_type=RepresentationType.SPARSE,
        )
    except ValueError as e:
        print(f"✓ Validation works: {e}")

    try:
        # This should fail - DENSE requires faiss_index_path
        invalid_config2 = FieldConfig(
            name="test",
            tokenizer_type=TokenizerType.SPLIT,
            representation_type=RepresentationType.DENSE,
        )
    except ValueError as e:
        print(f"✓ Validation works: {e}")

    print("\n=== Example completed successfully! ===")
    print("\nNext steps to fully implement:")
    print("- Implement DenseTextFieldIndex and DenseImageFieldIndex classes")
    print("- Set up proper FDE encoder integration")
    print("- Add FAISS dependency to pyproject.toml")
    print("- Create actual FAISS index files")


if __name__ == "__main__":
    main()
