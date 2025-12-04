from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import tqdm

from config import get_config, get_file_weights_path
from dataset import BilingualDataset
from model import build_transformer
from validation import run_validation

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])
    
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    
    # check that seq_len is enough
    max_len = 0
    for item in ds_raw:
        src_enc = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_enc = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len = max(max_len, len(src_enc), len(tgt_enc))
      
    assert max_len < config["seq_len"], f"seq len is too short: should be at least {max_len}"
    
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, config["seq_len"], config["d_model"])
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    
    print(f"Using device {device}")
    
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        # This wraps the model to split the input batch across devices
        model = nn.DataParallel(model)
        
    model.to(device)
    
    tb_writer = SummaryWriter(config["experiment_name"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_file_weights_path(config, config["preload"])
        print(f"preloading model: {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        
        batch_iterator = tqdm.tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)
            
            # run data through model
            projection = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            
            # combining seq_len and batch dimension
            loss = loss_fn(projection.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            tb_writer.add_scalar("train loss", loss.item(), global_step)
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
                        
            global_step += 1
            
        tb_writer.flush()
        run_validation(model, val_dataloader, config, device, lambda str: batch_iterator.write(str), tokenizer_src, tokenizer_tgt)
            
        model_filename = get_file_weights_path(config, f"{epoch:02d}")
        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step
            }, model_filename)
        
if __name__ == "__main__":
    config = get_config()
    train_model(config) 
        