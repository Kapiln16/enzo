import torch
import torch.nn.functional as F

def sample_next_token(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze()
