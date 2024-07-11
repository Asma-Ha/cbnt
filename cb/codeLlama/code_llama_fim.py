import json
import logging
import os
import sys

from os.path import isdir, join, isfile
from typing import List

import torch
from pydantic import BaseModel
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.assertion_utils import assert_not_empty
from utils.delta_time_printer import DeltaTime
from utils.similarity_calcul import SizeFitter
from cb.code_bert_mlm import CodeBertModel, CodeBertMlmFillMask, ListCodeBertPrediction
from accelerate import Accelerator

VOCAB_DIR = 'pre-trained/CodeLlama-7b-hf'
VOCAB_FILE = join(VOCAB_DIR, 'vocab.json')
CODE_BERT_MLM_MODEL = "codellama/CodeLlama-7b-hf"
FILL_MASK_FUNCTION_NAME = 'text-generation'

MASK = '<FILL_ME>'
SPACE_TOKEN = "_"
SPECIAL_TOKENS = ['▁<PRE>', '▁<SUF>', '▁<MID>']
MAX_TOKENS = 100
# default
PREDICTIONS_COUNT = 5
log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
MAX_BATCH_SIZE = 20

PREDS = 0
TOTAL_PRED_TIME = None


class CbSizeFitter(SizeFitter):
    def __init__(self, items_arr, max_size: int = MAX_TOKENS, filling_item=SPACE_TOKEN):
        super(CbSizeFitter, self).__init__(items_arr, size=max_size, filling_item=filling_item)


class CodeLlamaModel(CodeBertModel):
    mask = MASK
    max_tokens = MAX_TOKENS
    special_tokens = SPECIAL_TOKENS
    @staticmethod
    def load_vocab(model_dir):
        """Loads a vocabulary file into a dictionary."""
        import collections
        vocab = collections.OrderedDict()

        tokenizer_file = os.path.join(model_dir, 'tokenizer.json')
        if isfile(tokenizer_file):
            with open(tokenizer_file, 'r') as f:
                tokenizer_data = json.load(f)
                if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
                    vocab_data = tokenizer_data['model']['vocab']
                    for token, index in vocab_data.items():
                        token = token.encode("ascii", "ignore").decode()
                        token = ''.join(token.split())
                        vocab[int(index)] = token
            f.close()
        else:
            log.warning('Vocab file cannot be loaded: {0}'.format(tokenizer_file))
        return vocab

    def init_model_tokenizer(self, pretrained_model_name, vocab_dir):
        print(torch.version)
        print(torch.cuda.is_available())
        print(torch.version.cuda)
        print(torch.cuda.nccl.version())
        if not isdir(vocab_dir) or 0 == len(os.listdir(vocab_dir)):
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = LlamaForCausalLM.from_pretrained(pretrained_model_name, device_map="auto",
                                                          quantization_config=config)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            self.save_pretrained(vocab_dir)
            print('Model size in GB : ', self.model.get_memory_footprint()/1024/1024/1024)

        else:
            self.model = LlamaForCausalLM.from_pretrained(vocab_dir)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(vocab_dir)

    def __init__(self, pretrained_model_name, vocab_dir):
        self.tokenizer = None
        self.model = None
        self.init_model_tokenizer(pretrained_model_name, vocab_dir)
        self.vocab_dict = self.load_vocab(vocab_dir)
        log.info('num threads in torch:' + str(torch.get_num_threads()))

        # @list_to_tuple
        # @lru_cache(maxsize=2, typed=False)

class CodeLlamaFunction(CodeLlamaModel):

    def __init__(self, pretrained_model_name, vocab_dir):
        super().__init__(pretrained_model_name, vocab_dir)
        """from transformers import pipeline
        print('inference with codellama')
        
        self.pipeline_function = pipeline(function_name, model=self.model, tokenizer=self.tokenizer,
                                          torch_dtype=torch.float16, device_map="auto")"""
    def completion_function(self, arg):
        print("context : ", len(self.tokenize(arg['masked_code'])), arg['masked_code'])
        input_ids = self.tokenizer(arg['masked_code'], return_tensors="pt")["input_ids"].cuda()
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=arg['original_token_len'],
                                          num_beams=20, num_return_sequences=PREDICTIONS_COUNT,
                                          pad_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)
        return decoded
    def call_func(self, arg):
        return self.completion_function(arg)



class ListCodeLlamaPrediction(ListCodeBertPrediction):
    def get_original_and_predictions_tokens(self, code_llama_func: CodeLlamaFunction, masked_code, masked_token, suffix,
                                            original_code_tokens, max_size=MAX_TOKENS):
        assert_not_empty(masked_code, masked_token)
        if original_code_tokens is None or len(original_code_tokens) == 0:
            original_code = masked_code.replace(MASK, masked_token)
            original_code_tokens = code_llama_func.tokenize(original_code)
        assert_not_empty(original_code_tokens)
        result = [original_code_tokens]
        for prediction in self.__root__:
            predicted_code = prediction.put_token_inplace(masked_code, suffix)
            predicted_code_tokens = code_llama_func.tokenize(predicted_code)
            result.append(predicted_code_tokens)
        return CbSizeFitter(result, max_size=max_size).fit()



class CodeLlamaFillMask(CodeLlamaFunction, CodeBertMlmFillMask):
    def __init__(self, predictions_number=PREDICTIONS_COUNT):
        super().__init__(CODE_BERT_MLM_MODEL, VOCAB_DIR)
        self.predictions_number = predictions_number

    def call_func(self, arg, batch_size=MAX_BATCH_SIZE):
        global PREDS
        global TOTAL_PRED_TIME
        delta_time = DeltaTime(logging_level=logging.DEBUG)

        if isinstance(arg, list):
            # if there is a list of masked codes
            call_output = [self.call_func(a) for a in arg]
            # at this point we should have a list of a list of objects {'token_str' : str}
            try:
                result = [ListCodeLlamaPrediction.parse_obj(co) for co in call_output]
            except Exception as e:
                log.error('call_output :\n ' + str(call_output))
                raise e

            diff = delta_time.print('{0} masked code'.format(len(result)))
            if diff is not None:
                PREDS = PREDS + (len(arg) * PREDICTIONS_COUNT)
                TOTAL_PRED_TIME = diff if TOTAL_PRED_TIME is None else TOTAL_PRED_TIME + diff
                log.debug('{0} for {1} predictions'.format(TOTAL_PRED_TIME, PREDS))


        else:
            # for recursive call
            call_output = super().call_func(arg)
            result = [{'token_str': token} for token in call_output]

        return result

