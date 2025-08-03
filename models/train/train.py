import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.core.model import EnzoModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.core.model import EnzoModel


def train(model, dataloader, tokenizer, loss_fn, optimizer, device, epochs=5, save_path="enzo.pt"):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            labels = input_ids.clone()

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def load_model(vocab_size, model_path, device, **kwargs):
    model = EnzoModel(vocab_size=vocab_size, **kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
