import marisa_trie

class Vocabulary:
    def __init__(self, trie: marisa_trie.Trie):
        self.trie = trie

    def __len__(self):
        return len(self.trie)

    def __contains__(self, token):
        return token in self.trie

    def id(self, token):
        return self.trie.get(token)

    def token(self, id):
        return self.trie.get(id)
    
    @property
    def n_vocab(self):
        return len(self.trie)

    @classmethod
    def from_token_set(cls, token_set: set) -> "Vocabulary":
        trie = marisa_trie.Trie(sorted(token_set))
        return cls(trie)
