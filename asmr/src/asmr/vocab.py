import marisa_trie


class Vocabulary:
    def __init__(self, trie: marisa_trie.Trie):
        self.trie = trie

    def __len__(self) -> int:
        return len(self.trie)

    def __contains__(self, token: str) -> bool:
        return token in self.trie

    def id(self, token: str) -> int:
        return self.trie.get(token)  # type: ignore[no-any-return]

    def token(self, id: int) -> str:
        return self.trie.restore_key(id)  # type: ignore[no-any-return]

    @property
    def n_vocab(self) -> int:
        return len(self.trie)

    @classmethod
    def from_token_set(cls, token_set: set[str]) -> "Vocabulary":
        trie = marisa_trie.Trie(sorted(token_set))
        return cls(trie)
