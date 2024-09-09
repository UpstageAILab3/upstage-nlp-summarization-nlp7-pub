import pandas as pd
import re
import os
from dataset.dataset import  DatasetForTrain
from dataset.dataset import  DatasetForVal
import random

class Preprocess:
    # 마스크 토큰 사용시 __init__에 mask_token: str 추가
    def __init__(self, bos_token: str, eos_token: str, sep_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        # self.mask_token = mask_token

    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        df = pd.read_csv(file_path)
        if is_train:
            return df[['fname', 'dialogue', 'summary']]
        else:
            return df[['fname', 'dialogue']]

    def add_sep_tokens(self, dialogue_with_sep):
        """
        sep토큰 줄 바뀔 때마다 추가
        
        Parameters:
        - dialogue_with_sep:
        
        Returns:
        - str: <sep>토큰 추가한 문장 , 비었으면 공백전달.
        """
        if isinstance(dialogue_with_sep, str):
            return re.sub(r'(\r\n|\n)', f'{self.sep_token}', dialogue_with_sep)
        return ''
    
    # def preprocess_sentence(self, sentence: str) -> str:
    #     """
    #     문장 전처리
        
    #     Parameters:
    #     - sentence (str): 문장
        
    #     Returns:
    #     - str: 전처리 후 문장
    #     """
    #     sentence = sentence.lower()  # 텍스트 소문자화
    #     sentence = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+', '', sentence)  # 자음과 모음 제거
    #     sentence = re.sub(r'\[.*?\]', '', sentence)  # 대괄호로 둘러싸인 텍스트 제거
    #     sentence = re.sub(r"([.,!?])\1+", r"\1", sentence) 
    #     sentence = re.sub(r"[^가-힣a-z0-9#@,-\[\]\(\)]", " ", sentence)  # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    #     sentence = re.sub(r'[" "]+', " ", sentence)  # 여러 개의 공백을 하나의 공백으로 바꿈
    #     sentence = sentence.strip()  # 문장 양쪽 공백 제거
    #     return sentence

    # def apply_permute_or_infill(self, text, mask_token, mask_rate=0.15):
    #     """
    #     permutation 또는 infill masking을 "Person#:"을 제외한 뒤에 문장에 적용하기
        
    #     Parameters:
    #     - text (str): 문장
    #     - mask_token (str): infill에 사용하는 마스크 토큰
    #     - mask_rate (float): 마스 적용하는 비율

    #     Returns:
    #     - str: permutation이나 infill을 적용한 문장
    #     """
    #     def process_content(content, method):
    #         """
    #         permute와 infill중 선택되어 적용된다.
            
    #         Parameters:
    #         - content (str): 문장
    #         - method (str): permutation 또는 infill

    #         Returns:
    #         - str: 위에 과정 거치고 나온 결과
    #         """
    #         if method == "permute":
    #             words = content.split()
    #             return ' '.join(random.sample(words, len(words)))
    #         elif method == "infill":
    #             tokens = content.split()
    #             num_to_mask = int(len(tokens) * mask_rate)
    #             masked_indices = random.sample(range(len(tokens)), num_to_mask)
    #             for idx in masked_indices:
    #                 tokens[idx] = mask_token
    #             return ' '.join(tokens)

    #     # 발화자 태그와 그 뒤의 내용으로 분리
    #     pattern = r'(#Person\d+#:)'
    #     parts = re.split(pattern, text)
    #     processed_parts = []

    #     for i, part in enumerate(parts):
    #         if i % 2 == 0:  # 발화자 태그가 아닌 부분
    #             if part.strip() == "":
    #                 continue
    #             # 어떤 처리 방법을 사용할지 랜덤으로 선택
    #             method = random.choice(["permute", "infill"])
    #             processed_parts.append(process_content(part, method))
    #         else:  # 발화자 태그는 그대로 유지
    #             processed_parts.append(part)
                
        return ''.join(processed_parts)

    def make_input(self, dataset, is_test=False):
        dataset['dialogue'] = dataset['dialogue'].apply(self.add_sep_tokens)
        # dataset['dialogue'] = dataset['dialogue'].apply(self.preprocess_sentence)
        # dataset['dialogue'] = dataset['dialogue'].apply(lambda x: self.apply_permute_or_infill(x, self.mask_token))

        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)

            # processed_encoder_input = dataset['dialogue']
            # processed_decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            # processed_decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)


            # encoder_input = original_encoder_input.tolist() + processed_encoder_input.tolist()
            # decoder_input = original_decoder_input.tolist() + processed_decoder_input.tolist()
            # decoder_output = original_decoder_output.tolist() + processed_decoder_output.tolist()
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()

# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(data_path,'total_train.csv')
    # train_file_path = os.path.join('/root/dialogue/data/total_train.csv')
    
    val_file_path = os.path.join(data_path,'dev.csv')
    # val_file_path = os.path.join('/root/dialogue/data/dev.csv')
    

    # train, validation에 대해 각각 데이터프레임을 구축합니다.
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    encoder_input_train , decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val , decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-'*10, 'Load data complete', '-'*10,)

    tokenized_encoder_inputs = tokenizer(encoder_input_train, return_tensors="pt", padding=True,
                            add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_inputs = tokenizer(decoder_input_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_ouputs = tokenizer(decoder_output_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs,len(encoder_input_train))

    val_tokenized_encoder_inputs = tokenizer(encoder_input_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_inputs = tokenizer(decoder_input_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_ouputs = tokenizer(decoder_output_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_ouputs,len(encoder_input_val))

    print('-'*10, 'Make dataset complete', '-'*10,)
    return train_inputs_dataset, val_inputs_dataset