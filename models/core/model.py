import torch
import torch.nn as nn

class EnzoModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, hidden_dim=512, num_layers=4, max_len=256):
        super(EnzoModel, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        positions = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            x = self.transformer(x, src_key_padding_mask=~attention_mask)
        else:
            x = self.transformer(x)

        logits = self.fc_out(x)
        return logits
