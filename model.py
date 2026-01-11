import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class ProteinBERT(nn.Module):
    def __init__(self, vocab_size: int = 24, d_model: int = 256, n_layers: int = 6, 
                 n_heads: int = 8, d_ff: int = 1024, max_length: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(max_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len]
        x = self.dropout(x)
        
        if attention_mask is not None:
            attention_mask = (attention_mask == 0)
        
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = self.norm(x)
        
        logits = self.mlm_head(x)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': x
        }
    
    def get_sequence_embedding(self, input_ids, attention_mask=None):
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
            
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(output['hidden_states'])
                sum_embeddings = torch.sum(output['hidden_states'] * mask_expanded, dim=1)
                sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = torch.mean(output['hidden_states'], dim=1)
            
            return pooled

def create_model(vocab_size: int = 24, d_model: int = 256, n_layers: int = 6,
                n_heads: int = 8, d_ff: int = 1024, max_length: int = 512,
                dropout: float = 0.1) -> ProteinBERT:
    model = ProteinBERT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_length=max_length,
        dropout=dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model w/ {total_params:,} total parameters & ({trainable_params:,} trainable)")

    return model
