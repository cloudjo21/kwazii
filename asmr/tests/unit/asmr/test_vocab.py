import pytest
import marisa_trie

from asmr.vocab import Vocabulary


class TestVocabularyCreation:
    def test_vocabulary_creation_from_token_set(self):
        tokens = {"hello", "world", "test"}
        vocabulary = Vocabulary.from_token_set(tokens)

        assert isinstance(vocabulary, Vocabulary)
        assert isinstance(vocabulary.trie, marisa_trie.Trie)

    def test_vocabulary_empty_token_set(self):
        vocabulary = Vocabulary.from_token_set(set())

        assert len(vocabulary) == 0

    def test_vocabulary_single_token(self):
        vocabulary = Vocabulary.from_token_set({"only"})

        assert len(vocabulary) == 1
        assert "only" in vocabulary


class TestVocabularyTokenIdMapping:
    def test_id_returns_integer(self):
        vocabulary = Vocabulary.from_token_set({"apple", "banana", "cherry"})

        for token in ("apple", "banana", "cherry"):
            assert isinstance(vocabulary.id(token), int)

    def test_token_restores_from_id(self):
        tokens = {"apple", "banana", "cherry"}
        vocabulary = Vocabulary.from_token_set(tokens)

        for token in tokens:
            token_id = vocabulary.id(token)
            assert vocabulary.token(token_id) == token

    def test_ids_are_unique(self):
        tokens = {"x", "y", "z"}
        vocabulary = Vocabulary.from_token_set(tokens)

        ids = [vocabulary.id(t) for t in tokens]
        assert len(set(ids)) == len(ids)

    def test_id_and_token_are_inverse(self):
        vocabulary = Vocabulary.from_token_set({"foo", "bar", "baz"})

        for token in ("foo", "bar", "baz"):
            assert vocabulary.token(vocabulary.id(token)) == token


class TestVocabularyContainsCheck:
    def test_contains_existing_token(self):
        vocabulary = Vocabulary.from_token_set({"present", "here"})

        assert "present" in vocabulary
        assert "here" in vocabulary

    def test_not_contains_missing_token(self):
        vocabulary = Vocabulary.from_token_set({"present"})

        assert "absent" not in vocabulary

    def test_contains_is_case_sensitive(self):
        vocabulary = Vocabulary.from_token_set({"hello"})

        assert "hello" in vocabulary
        assert "HELLO" not in vocabulary
        assert "Hello" not in vocabulary


class TestVocabularyLength:
    def test_length_matches_token_count(self):
        tokens = {"a", "b", "c", "d"}
        vocabulary = Vocabulary.from_token_set(tokens)

        assert len(vocabulary) == 4

    def test_n_vocab_equals_len(self):
        tokens = {"one", "two", "three"}
        vocabulary = Vocabulary.from_token_set(tokens)

        assert vocabulary.n_vocab == len(vocabulary)

    def test_duplicate_tokens_deduplicated(self):
        # set() deduplicates, confirming only unique tokens are stored
        tokens = {"dup", "dup", "unique"}
        vocabulary = Vocabulary.from_token_set(tokens)

        assert len(vocabulary) == 2
