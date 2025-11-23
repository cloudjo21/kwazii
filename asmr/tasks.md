# Tokenization
notes: working directory: asmr/tokenize/
- add auto tokenizer to asmr/tokenize/helpers.py
- and support to create TokenizerWrapper from the one of auto tokenizer and others in TokenizerWrapper
- add converter from image bytes to PIL.Image
- add the representative class to alter TokenizerWrapper, which is for chunking text or image

# Field Index
notes: working directory: asmr/index/
- define sparse field and dense field and they are abstract classes
- add field config: name, tokenizer_type(split or hf_auto), representation_type(sparse or dense)
  - support model_path in field config if hf_auto is set
  - support faiss vector index path if you need to support dense fields
- change TextFieldIndex to SparseTextFeieldIndex
- add DenseTextFieldIndex that inherits dense field
  - and add text encoding indexer to individual python file which is using MultiModalFdeEncoder and its method encode_text
- add dense image field index to asmr/index/fields.py which is inherit dense field
  - and add image encoding indexer to individual python file which is using MultiModalFdeEncoder and its method encode_image

# Query
notes: working directory: asmr/retrieve/
- define Query class to support text and image in query.py
  - parameters like query: str in retrieve, index, or other packages in asmr

# Field Retrieval
notes: working directory: asmr/retrieve/
- add retrievers individually corresponding to the one of the field indexes in asmr/index/fields.py if you need
- update QueryRouter to support Query class in query.py
