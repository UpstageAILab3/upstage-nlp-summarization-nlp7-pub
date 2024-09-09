import torch
from config.config import load_config
from preprocess.preprocess import Preprocess
from trainer.trainer import load_trainer_for_train, load_tokenizer_and_model_for_train
import wandb
from preprocess.preprocess import prepare_train_dataset


def main():
    # Load configuration
    config = load_config()

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)

    # Load and preprocess data
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'], config['tokenizer']['sep_token'])
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

    # Load trainer
    trainer = load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    main()

