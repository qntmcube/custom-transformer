from numpy import dtype
from tokenizers import Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_id = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_id = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_id = tokenizer_tgt.token_to_id("[PAD]")
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        data_at_index = self.ds[index]
        src_text = data_at_index["translation"][self.src_lang]
        tgt_text = data_at_index["translation"][self.tgt_lang]
        
        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_len - len(src_tokens) - 2 # 2 spaces for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(tgt_tokens) - 1 # 1 space for just SOS
        
        if (enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0):
            raise ValueError("Sentence is too long")
        
        encoder_input = torch.cat(
            [
                torch.tensor([self.sos_id], dtype=torch.int64), 
                torch.tensor(src_tokens, dtype=torch.int64), 
                torch.tensor([self.eos_id], dtype=torch.int64), 
                torch.full((enc_num_padding_tokens,), self.pad_id, dtype=torch.int64),
            ], 
            dim=0
        )
        
        decoder_input = torch.cat(
            [
                torch.tensor([self.sos_id], dtype=torch.int64), 
                torch.tensor(tgt_tokens, dtype=torch.int64), 
                torch.full((dec_num_padding_tokens,), self.pad_id, dtype=torch.int64),
            ], 
            dim=0
        )
        
        label = torch.cat(
            [
                torch.tensor(tgt_tokens, dtype=torch.int64), 
                torch.tensor([self.eos_id], dtype=torch.int64), 
                torch.full((dec_num_padding_tokens,), self.pad_id, dtype=torch.int64),
            ], 
            dim=0
        )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_id).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_id).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
        
def causal_mask(size):
        mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
        return mask == 0 # reverse mask
        