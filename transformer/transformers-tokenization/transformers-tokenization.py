class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        special_token = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for idx, token in enumerate(special_token):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
            self.vocab_size += 1
    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        unique_word = set()
        for text in texts:
            text = text.lower().split(' ')
            for word in text:
                unique_word.add(word)
        sorted_list = sorted(list(unique_word))

        for idx, token in enumerate(sorted_list):
            self.word_to_id[token] = self.vocab_size
            self.id_to_word[self.vocab_size] = token
            self.vocab_size += 1

        return sorted_list
    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        words = text.lower().split()
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(word, unk_id) for word in words]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        words = [self.id_to_word.get(idx, self.unk_token) for idx in ids]
        return " ".join(words)
         