from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "learning_rate": 10**-4,
        "seq_len": 800,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "es",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }
    
def get_file_weights_path(config, epoch: str):
    if epoch is None:
        return None
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    if epoch == "latest":
        files = list(Path(model_folder).glob(f"{model_basename}*"))
        if len(files) == 0:
            return None
        files.sort()
        return str(files[-1])
    model_filename = f"{model_basename}{epoch}"
    return str(Path(".") / model_folder / model_filename)