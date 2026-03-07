import pytest
from unittest.mock import MagicMock, patch, call


class TestPromptType:
    def test_prompt_type_query_value(self):
        from fde.config import PromptType

        assert PromptType.QUERY == "query"

    def test_prompt_type_passage_value(self):
        from fde.config import PromptType

        assert PromptType.PASSAGE == "passage"

    def test_prompt_type_is_str_enum(self):
        from fde.config import PromptType

        assert isinstance(PromptType.QUERY, str)
        assert isinstance(PromptType.PASSAGE, str)


class TestPromptTypeToEncodingTypeMapping:
    def test_mapping_contains_all_prompt_types(self):
        from fde.config import PromptType, PROMPT_TYPE_TO_ENCODING_TYPE

        assert PromptType.QUERY in PROMPT_TYPE_TO_ENCODING_TYPE
        assert PromptType.PASSAGE in PROMPT_TYPE_TO_ENCODING_TYPE

    def test_query_and_passage_have_different_encoding_types(self):
        from fde.config import PromptType, PROMPT_TYPE_TO_ENCODING_TYPE

        query_enc = PROMPT_TYPE_TO_ENCODING_TYPE[PromptType.QUERY]
        passage_enc = PROMPT_TYPE_TO_ENCODING_TYPE[PromptType.PASSAGE]
        assert query_enc != passage_enc


class TestFdeConfig:
    def test_fde_config_creation_with_query_prompt_type(self):
        from fde.config import FdeConfig, PromptType

        cfg = FdeConfig(prompt_type=PromptType.QUERY)

        assert cfg.prompt_type == PromptType.QUERY

    def test_fde_config_creation_with_passage_prompt_type(self):
        from fde.config import FdeConfig, PromptType

        cfg = FdeConfig(prompt_type=PromptType.PASSAGE)

        assert cfg.prompt_type == PromptType.PASSAGE

    def test_fde_config_default_values(self):
        from fde.config import FdeConfig, PromptType

        cfg = FdeConfig(prompt_type=PromptType.QUERY)

        assert cfg.num_repetitions == 20
        assert cfg.num_simhash_projections == 5
        assert cfg.projection_dimension == 16
        assert cfg.final_projection_dimension is None
        assert cfg.seed == 1221

    def test_fde_config_custom_values(self):
        from fde.config import FdeConfig, PromptType

        cfg = FdeConfig(
            prompt_type=PromptType.PASSAGE,
            num_repetitions=10,
            num_simhash_projections=3,
            projection_dimension=8,
            seed=42,
        )

        assert cfg.num_repetitions == 10
        assert cfg.num_simhash_projections == 3
        assert cfg.projection_dimension == 8
        assert cfg.seed == 42

    def test_update_config_sets_encoding_type(self):
        from fde.config import FdeConfig, PromptType

        cfg = FdeConfig(prompt_type=PromptType.QUERY)
        mock_muvfde_config = MagicMock()

        cfg.update_config(mock_muvfde_config)

        mock_muvfde_config.set_encoding_type.assert_called_once()
        mock_muvfde_config.set_num_repetitions.assert_called_once_with(20)
        mock_muvfde_config.set_num_simhash_projections.assert_called_once_with(5)
        mock_muvfde_config.set_projection_dimension.assert_called_once_with(16)
        mock_muvfde_config.set_seed.assert_called_once_with(1221)

    def test_update_config_passage_enables_fill_empty(self):
        from fde.config import FdeConfig, PromptType

        cfg = FdeConfig(prompt_type=PromptType.PASSAGE)
        mock_muvfde_config = MagicMock()

        cfg.update_config(mock_muvfde_config)

        mock_muvfde_config.enable_fill_empty.assert_called_once_with(True)

    def test_update_config_query_does_not_enable_fill_empty(self):
        from fde.config import FdeConfig, PromptType

        cfg = FdeConfig(prompt_type=PromptType.QUERY)
        mock_muvfde_config = MagicMock()

        cfg.update_config(mock_muvfde_config)

        mock_muvfde_config.enable_fill_empty.assert_not_called()

    def test_update_config_with_final_projection_sets_identity_type(self):
        from fde.config import FdeConfig, PromptType

        cfg = FdeConfig(prompt_type=PromptType.QUERY, final_projection_dimension=64)
        mock_muvfde_config = MagicMock()

        cfg.update_config(mock_muvfde_config)

        mock_muvfde_config.set_final_projection_dimension.assert_called_once_with(64)

    def test_update_config_without_final_projection_sets_ams_sketch(self):
        from fde.config import FdeConfig, PromptType

        cfg = FdeConfig(prompt_type=PromptType.QUERY, final_projection_dimension=None)
        mock_muvfde_config = MagicMock()

        cfg.update_config(mock_muvfde_config)

        mock_muvfde_config.set_final_projection_dimension.assert_not_called()
