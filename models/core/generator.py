import torch
from models.data.tokenizer import CharTokenizer
from models.core.utils import sample_next_token

class Generator:
    def __init__(self, model, tokenizer: CharTokenizer, device='cpu', max_length=200):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def generate(self, prompt: str, temperature=0.8):
        self.model.eval()
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(self.device)
        
        for _ in range(self.max_length):
            with torch.no_grad():
                logits = self.model(input_ids)
                next_token = sample_next_token(logits[:, -1, :], temperature)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(input_ids[0].tolist())
