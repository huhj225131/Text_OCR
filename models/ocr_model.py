from resnet import ResBlock, ResNetBackBone
from positional_encoding import PositionalEncoding1D, PositionalEncoding2D
import torch
import torch.nn as nn
import math
class EncoderDecoderHTR ():
    def __init__(self, vocab_size, d_model = 256,
                 enc_layers=4,
                 dec_layers = 3,
                 nhead=8,
                 ffn_dim=1024,
                 dropout=0.1,
                 max_seq_len=50):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.backbone = ResNetBackBone(d_model=d_model)
        self.pos_enc_2d = PositionalEncoding2D(d_model=d_model)
        self.tgt_pos_enc = PositionalEncoding1D(d_model=d_model,
                                                max_len=max_seq_len)

        self.tgt_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )  
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=enc_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)     
        self.out_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        """Better initialization (TrOCR-style)"""
        # Linear layers: truncated normal
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def encode(self, images):
        """Encode image to features"""
        # CNN: [B, 1, H, W] → [B, C, h, w]
        features = self.backbone(images)
        
        # Flatten: [B, C, h, w] → [B, h*w, C]
        B, C, h, w = features.shape
        features = features.flatten(2).transpose(1, 2)
        
        # Add 2D positional encoding (MATCHED với MJSynth: pass h, w)
        features = self.pos_enc_2d(features, h, w)
        
        # Transformer encoder
        memory = self.encoder(features)
        return memory
    
    def decode(self, tgt_tokens, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """Decode text from memory"""
        # Embed tokens: [B, seq_len] → [B, seq_len, d_model]
        tgt_emb = self.tgt_embed(tgt_tokens) * math.sqrt(self.d_model)
        
        # Add 1D positional encoding
        tgt_emb = self.tgt_pos_enc(tgt_emb)
        
        # Transformer decoder (with cross-attention to encoder memory)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.out_proj(output)
        return logits
    

    def forward(self, images, tgt_tokens, tgt_mask=None, tgt_key_padding_mask=None):
        """Full forward pass (teacher forcing)"""
        memory = self.encode(images)
        logits = self.decode(tgt_tokens, memory, tgt_mask, tgt_key_padding_mask)
        return logits
    
    def generate(self, images, sos_idx, eos_idx, max_len=50):
        """Greedy decoding (inference) - FIXED: per-sample EOS tracking"""
        self.eval()
        with torch.no_grad():
            memory = self.encode(images)
            B = images.size(0)
            
            # Start with <SOS>
            tgt_tokens = torch.full((B, 1), sos_idx, dtype=torch.long, device=images.device)
            finished = torch.zeros(B, dtype=torch.bool, device=images.device)
            
            for _ in range(max_len):
                # Generate causal mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(images.device)
                
                # Decode
                logits = self.decode(tgt_tokens, memory, tgt_mask=tgt_mask)
                
                # Get next token (greedy)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
                # Mark finished sequences (found EOS)
                finished = finished | (next_token.squeeze(1) == eos_idx)
                
                # Append to sequence (replace with PAD if already finished)
                next_token_masked = next_token.clone()
                next_token_masked[finished] = sos_idx  # Use SOS as dummy (will be ignored in decode)
                tgt_tokens = torch.cat([tgt_tokens, next_token_masked], dim=1)
                
                # Stop if all sequences finished
                if finished.all():
                    break
            
            return tgt_tokens
