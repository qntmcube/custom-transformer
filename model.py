import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        
    def forward(self, x):
        return self.embedding_layer(x) * math.sqrt(self.embedding_size)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, embedding_size: int, seq_length: int, dropout: float):
        super().__init__()
        self.embedding_size = embedding_size
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        
        pos_enc = torch.zeros(1, seq_length, embedding_size) # MIGHT BE WRONG
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)

        divisor = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pos_enc[0, :, 0::2] = torch.sin(position * divisor)
        pos_enc[0, :, 1::2] = torch.cos(position * divisor)
        
        self.register_buffer("pos_enc", pos_enc)
        
    def forward(self, x):
        return self.dropout(x + (self.pos_enc[:, :x.shape[1], :]).requires_grad_(False))
    
class Normalization(nn.Module):
    
    def __init__(self, embedding_size, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(embedding_size))
        self.beta = nn.Parameter(torch.zeros(embedding_size))
        
    def forward(self, x):
        std = x.std(dim = -1, keepdim=True)
        mean = x.mean(dim = -1, keepdim=True)
        
        return self.beta + self.gamma * (x -  mean) / (std + self.epsilon)

class FeedForwardBlock(nn.Module):
    
    def __init__(self, embedding_size: int, ff_size: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_size, ff_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_size, embedding_size)
        
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, embedding_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.embedding_size = embedding_size
        assert embedding_size % num_heads == 0, "embedding size is not divisible by number of heads"
        self.num_heads = num_heads
        self.d_head = embedding_size // num_heads
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w_k = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w_v = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w_o = nn.Linear(embedding_size, embedding_size, bias=False)
        
    def forward(self, query, key, value, mask):
        # (batch size = 1, seq_len, embedding_size) -> (batch size = 1, seq_len, num_heads, d_head) 
        #                                           -> (batch size = 1, num_heads, seq_len, d_head)
        q_split = self.w_q(query).view(query.shape[0], query.shape[1], self.num_heads, self.d_head).transpose(1, 2)
        k_split = self.w_k(key).view(key.shape[0], key.shape[1], self.num_heads, self.d_head).transpose(1, 2)
        v_split = self.w_v(value).view(value.shape[0], value.shape[1], self.num_heads, self.d_head).transpose(1, 2)
        
        attention_scores = q_split @ k_split.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = self.dropout(attention_scores.softmax(dim=-1))
        
        attention = attention_scores @ v_split
        return self.w_o(attention.transpose(1, 2).contiguous().view(attention.shape[0], -1, self.d_head * self.num_heads))
    
class ResidualLayer(nn.Module):
    
    def __init__(self, embedding_size, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = Normalization(embedding_size)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    
    def __init__(self, embedding_size, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_layers = nn.ModuleList(ResidualLayer(embedding_size, dropout) for i in range(2))
        
    def forward(self, x, src_mask):
        x = self.residual_layers[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_layers[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    
    def __init__(self, embedding_size, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = Normalization(embedding_size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    
    def __init__(self, embedding_size, masked_attention_block: MultiHeadAttention, 
                 cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.masked_attention_block = masked_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_layers = nn.ModuleList(ResidualLayer(embedding_size, dropout) for _ in range(3))
        
    def forward(self, x, encoder_output, tgt_mask, padding_mask):
        x = self.residual_layers[0](x, lambda x: self.masked_attention_block(x, x, x, tgt_mask))
        x = self.residual_layers[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, padding_mask))
        x = self.residual_layers[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, embedding_size, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = Normalization(embedding_size)
        
    def forward(self, x, encoder_output, tgt_mask, padding_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, padding_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    
    def __init__(self, embedding_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, x):
        return self.linear(x)
    
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, projection: ProjectionLayer, src_embeddings: 
                InputEmbeddings, tgt_embeddings: InputEmbeddings, src_pos_encoding: PositionalEncoding, tgt_pos_encoding: PositionalEncoding):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection
        self.src_embeddings = src_embeddings
        self.tgt_embeddings = tgt_embeddings
        self.src_pos_encoding = src_pos_encoding
        self.tgt_pos_encoding = tgt_pos_encoding
        
    def encode(self, src, src_mask):
        src = self.src_embeddings(src)
        src = self.src_pos_encoding(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, tgt_mask, padding_mask):
        tgt = self.tgt_embeddings(tgt)
        tgt = self.tgt_pos_encoding(tgt)
        return self.decoder(tgt, encoder_output, tgt_mask, padding_mask)
    
    def project(self, x):
        return self.projection(x)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
        return self.project(dec_output)
    
def build_transformer(src_vocab_size, tgt_vocab_size, seq_length, embedding_size = 512, ff_size = 2048, num_heads = 8, dropout = 0.1, num_enc_loops = 6, num_dec_loops = 6):
    src_embeddings = InputEmbeddings(src_vocab_size, embedding_size)
    tgt_embeddings = InputEmbeddings(tgt_vocab_size, embedding_size)
    
    src_pos_enc = PositionalEncoding(embedding_size, seq_length, dropout)
    tgt_pos_enc = PositionalEncoding(embedding_size, seq_length, dropout)
    
    encoder = Encoder(embedding_size,
                      nn.ModuleList(
                        EncoderBlock(
                            embedding_size,
                            MultiHeadAttention(embedding_size, num_heads, dropout), 
                            FeedForwardBlock(embedding_size, ff_size, dropout),
                            dropout) 
                        for _ in range(num_enc_loops)))
    
    decoder = Decoder(embedding_size,
                      nn.ModuleList(
                        DecoderBlock(
                            embedding_size,
                            *(MultiHeadAttention(embedding_size, num_heads, dropout) for _ in range(2)), 
                            FeedForwardBlock(embedding_size, ff_size, dropout),
                            dropout)
                        for _ in range(num_dec_loops)))
    
    projection = ProjectionLayer(embedding_size, tgt_vocab_size)
    
    transfomrer = Transformer(encoder, decoder, projection, src_embeddings, tgt_embeddings, src_pos_enc, tgt_pos_enc)
    
    for parameter in transfomrer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
            
    return transfomrer