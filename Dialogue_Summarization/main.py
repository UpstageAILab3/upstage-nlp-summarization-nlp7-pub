import torch
from config.config import load_config
from preprocess.preprocess import Preprocess
from trainer.trainer import load_trainer_for_train, load_tokenizer_and_model_for_train
import wandb
from preprocess.preprocess import prepare_train_dataset


def main():
<<<<<<< HEAD
    # config 로드
    config = load_config()

    # 디바이스 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 모델과 토크나이저 모델 로드
    generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)

    # 전처리 로드
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'], config['tokenizer']['sep_token'])
    # preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'], config['tokenizer']['mask_token'])
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

    # trainer 로드
=======
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
>>>>>>> 54570e73c6867250070e456a3d275d63930bc1ff
    trainer = load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    main()

