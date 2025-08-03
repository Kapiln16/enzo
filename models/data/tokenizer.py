class CharTokenizer:
    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 (){}[]:;.,'\"+-*/=%!<>#\\\n\t")
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        # Add EOS token
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = len(self.stoi)
        self.stoi[self.eos_token] = self.eos_token_id
        self.itos[self.eos_token_id] = self.eos_token

    def encode(self, text):
        return [self.stoi.get(ch, self.eos_token_id) for ch in text]

    def decode(self, tokens):
        return ''.join([self.itos.get(token, '') for token in tokens])
