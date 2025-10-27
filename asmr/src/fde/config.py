import enum

import muvfde
import pydantic

from muvfde.muvfde_ext import fixed_dimensional_encoding_config as MuvFdeConfig


_NUM_REPETITIONS = 20
_NUM_SIMHASH_PROJECTIONS = 5
_PROJECTION_DIMENSION = 16
_FINAL_PROJECTION_DIMENSION = None
_SEED = 1221


class PromptType(enum.StrEnum):
    QUERY = 'query'
    PASSAGE = 'passage'


PROMPT_TYPE_TO_ENCODING_TYPE = {
    PromptType.QUERY: muvfde.encoding_type.DEFAULT_SUM,
    PromptType.PASSAGE: muvfde.encoding_type.AVERAGE,
}


class FdeConfig(pydantic.BaseModel):
    prompt_type: PromptType

    num_repetitions: int = _NUM_REPETITIONS
    num_simhash_projections: int = _NUM_SIMHASH_PROJECTIONS
    projection_dimension: int = _PROJECTION_DIMENSION
    final_projection_dimension: int | None = _FINAL_PROJECTION_DIMENSION
    seed: int = _SEED


    class Config:
        arbitrary_types_allowed = True

    def _set_own_encoding_config(self, muvfde_config: MuvFdeConfig):
        if self.prompt_type not in PROMPT_TYPE_TO_ENCODING_TYPE:
            raise ValueError(f'Unknown prompt type: {self.prompt_type}')
        muvfde_config.set_encoding_type(PROMPT_TYPE_TO_ENCODING_TYPE[self.prompt_type])
        if self.prompt_type == PromptType.PASSAGE:
            muvfde_config.enable_fill_empty(True)

    def update_config(self, muvfde_config: MuvFdeConfig):
        self._set_own_encoding_config(muvfde_config)
        muvfde_config.set_num_repetitions(self.num_repetitions)
        muvfde_config.set_num_simhash_projections(self.num_simhash_projections)
        muvfde_config.set_projection_dimension(self.projection_dimension)
        muvfde_config.set_seed(self.seed)

        if self.final_projection_dimension is not None:
            muvfde_config.set_projection_type(muvfde.projection_type.DEFAULT_IDENTITY)
            muvfde_config.set_final_projection_dimension(self.final_projection_dimension)
        else:
            muvfde_config.set_projection_type(muvfde.projection_type.AMS_SKETCH)

        return muvfde_config

    @classmethod
    def apply_with_prompt_type(cls, prompt_type: PromptType):
        cls_instance = cls(prompt_type=prompt_type)
        muvfde_config = muvfde.fixed_dimensional_encoding_config()
        cls_instance.update_config(muvfde_config)
        return muvfde_config

    @classmethod
    def apply_with_args(cls, prompt_type: PromptType,
                        num_repetitions: int = _NUM_REPETITIONS,
                        num_simhash_projections: int = _NUM_SIMHASH_PROJECTIONS,
                        projection_dimension: int = _PROJECTION_DIMENSION,
                        final_projection_dimension: int | None = _FINAL_PROJECTION_DIMENSION,
                        seed: int = _SEED):
        cls_instance = cls(prompt_type=prompt_type,
                           num_repetitions=num_repetitions,
                           num_simhash_projections=num_simhash_projections,
                           projection_dimension=projection_dimension,
                           final_projection_dimension=final_projection_dimension,
                           seed=seed)
        muvfde_config = muvfde.fixed_dimensional_encoding_config()
        cls_instance.update_config(muvfde_config)
        return muvfde_config
    
