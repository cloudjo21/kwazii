"""On-the-fly candidate image indexer.

Builds an ephemeral DenseImageFieldIndex from a list of image URLs and wraps
it in a DocumentRetriever ready for retrieve_with_aggregation_head.

All heavy imports (asmr.index, fde, faiss) are deferred to function bodies so
that the module can be imported without triggering the full dependency chain.
"""

import io
import logging
import urllib.request
from typing import TYPE_CHECKING

from PIL import Image

from asmr.match.config import RankerConfig

if TYPE_CHECKING:
    from asmr.retrieve.helpers import DocumentRetriever
    from fde.base import BaseFdeEncoder

logger = logging.getLogger(__name__)

# Sentinel used as faiss_index_path to signal "ephemeral – do not load from disk".
_EPHEMERAL_PATH = ":ephemeral:"
_DOC_ID_PREFIX = "img_"


def _load_image(url: str, timeout_s: float) -> Image.Image:
    """Load a PIL Image from an HTTP/HTTPS URL or local file path.

    Args:
        url: HTTP(S) URL or local filesystem path.
        timeout_s: Request timeout in seconds (ignored for local paths).

    Returns:
        RGB PIL Image.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    try:
        if url.startswith(("http://", "https://")):
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        return Image.open(url).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to load image from {url!r}: {exc}") from exc


class CandidateImageIndexer:
    """Builds an ephemeral DocumentRetriever from candidate image URLs.

    The index is built fresh on every call to build() — there is no caching or
    disk persistence.  This is intentional: the candidate set is small (≤20
    images) so the overhead of building a FlatIP FAISS index is negligible.

    Attributes:
        encoder: FDE encoder used to embed candidate images.
        config: Ranker configuration (field_name, timeout, etc.).
    """

    def __init__(
        self,
        encoder: BaseFdeEncoder,
        config: RankerConfig,
    ) -> None:
        self._encoder = encoder
        self._config = config

    def load_images(self, urls: list[str]) -> list[Image.Image]:
        """Load PIL Images from URLs or local paths.

        Args:
            urls: List of HTTP(S) URLs or local filesystem paths.

        Returns:
            Loaded RGB PIL Images in the same order as urls.

        Raises:
            ValueError: If any URL fails to load.
        """
        return [_load_image(url, self._config.image_load_timeout_s) for url in urls]

    def build(
        self,
        image_urls: list[str],
    ) -> tuple[DocumentRetriever, list[str]]:
        """Build an ephemeral DocumentRetriever from candidate image URLs.

        Steps:
          1. Assign stable doc_ids ("img_0", "img_1", …).
          2. Load images from URLs.
          3. Encode with the injected encoder.
          4. Build a DenseImageFieldIndex (FAISS IndexFlatIP).
          5. Wrap in DocumentRetriever via QueryRouter.

        Args:
            image_urls: Candidate image URLs (must be non-empty).

        Returns:
            Tuple (retriever, doc_ids) where doc_ids[i] ↔ image_urls[i].

        Raises:
            ValueError: If image_urls is empty or any image fails to load.
        """
        if not image_urls:
            raise ValueError("image_urls must be non-empty")

        from asmr.index.config import FieldConfig, RepresentationType, TokenizerType
        from asmr.index.fields import DenseImageFieldIndex
        from asmr.retrieve.helpers import DocumentRetriever
        from asmr.retrieve.retrievers import DenseImageFieldRetriever, QueryRouter

        doc_ids = [f"{_DOC_ID_PREFIX}{i}" for i in range(len(image_urls))]

        field_cfg = FieldConfig(
            name=self._config.field_name,
            tokenizer_type=TokenizerType.SPLIT,
            representation_type=RepresentationType.DENSE,
            faiss_index_path=_EPHEMERAL_PATH,
        )

        img_index = DenseImageFieldIndex(config=field_cfg, encoder=self._encoder)
        images = self.load_images(image_urls)
        img_index.add_documents(doc_ids, images)

        retriever_field = DenseImageFieldRetriever(field_index=img_index)
        router = QueryRouter({self._config.field_name: retriever_field})
        retriever = DocumentRetriever(fields_retriever=router)

        return retriever, doc_ids
