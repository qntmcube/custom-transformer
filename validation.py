from tokenizers import Tokenizer
import torch

from dataset import causal_mask
from model import Transformer

def greedy_decode(model: Transformer, src, enc_mask, tokenizer_src: Tokenizer, max_len):
    sos_token = tokenizer_src.token_to_id("[SOS]")
    eos_token = tokenizer_src.token_to_id("[EOS]")
    
    encoding = model.encode(src, enc_mask)
    
    decoder_output = torch.tensor([sos_token]).unsqueeze(0).type_as(src)
    
    while True:
        
        tgt_mask = causal_mask(decoder_output.size(1)).type_as(enc_mask)
        
        decoding = model.decode(decoder_output, encoding, tgt_mask, enc_mask)
        projection = model.project(decoding[:, -1])
        best_token = torch.argmax(projection)
        
        decoder_output = torch.cat([decoder_output, torch.tensor([best_token]).unsqueeze(0).type_as(src)], dim=1)
        
        if decoder_output.size(1) >= max_len or best_token == eos_token:
            break
        
    return decoder_output.squeeze(0)

def run_validation(model: Transformer, val_dataloader, config, print_msg, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, num_val = 2):
    
    model.eval()
    
    console_width = 80
    
    with torch.no_grad():
        
        count = 0
        
        for batch in val_dataloader:
            count += 1
            encoder_input = batch["encoder_input"]
            encoder_mask = batch["encoder_mask"]
            
            # double check that batch size is 1
            assert encoder_input.size(0) == 1, "batch size must be 1 for validation"
            
            decoding = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, config["seq_len"])
            decoded_msg = tokenizer_tgt.decode(decoding.detach().cpu().numpy())
                        
            input_msg = batch["src_text"][0]
            label_msg = batch["tgt_text"][0]
            
            print_msg("-"*console_width)
            
            print_msg(f"ENCODER INPUT: {input_msg}")
            print_msg(f"MODEL OUTPUT: {decoded_msg}")
            print_msg(f"EXPECTED OUTPUT: {label_msg}")
            
            if count == num_val:
                print_msg("-"*console_width)
                break