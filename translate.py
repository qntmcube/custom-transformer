from pathlib import Path
from tokenizers import Tokenizer
from config import get_config, get_file_weights_path
from model import Transformer
from train import get_model
import torch

from validation import greedy_decode

def translate(model: Transformer, src_txt: str, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, config, device):
    src_tokens = tokenizer_src.encode(src_txt).ids
    enc_input = torch.cat([
        torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64), 
        torch.tensor(src_tokens, dtype=torch.int64), 
        torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64), 
        torch.full((config["seq_len"] - len(src_tokens) - 2,), tokenizer_src.token_to_id("[PAD]"), dtype=torch.int64),
    ], dim=0).unsqueeze(0).to(device)
    enc_mask = (enc_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int().to(device)
    
    decoding = greedy_decode(model, enc_input, enc_mask, tokenizer_src, config["seq_len"], device)
    decoded_msg = tokenizer_tgt.decode(decoding.detach().cpu().numpy())
    
    return decoded_msg
    
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device {device}")
    
    config = get_config()

    
    tokenizer_src: Tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt: Tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    model_filename = get_file_weights_path(config, config["preload"])
    print(f"loading model: {model_filename}")
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    
    model.eval()
    with torch.no_grad():
        while True:
            text = input("\033[0mWrite a sentence to be 'translated' lmao\n\033[0;32m")
            
            if text.lower().startswith("e"):
                break
            
            print("\033[1;31m" + translate(model, text, tokenizer_src, tokenizer_tgt, config, device))