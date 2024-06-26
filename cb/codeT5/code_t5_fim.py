import json
import logging
import os
import sys
from functools import lru_cache
from os.path import isdir, join, isfile
from typing import List, Dict

import torch
from pydantic.main import BaseModel
from transformers import RobertaTokenizer, T5ForConditionalGeneration

from utils.assertion_utils import assert_not_empty, is_empty_strip
from utils.caching_utils import list_to_tuple, tuple_to_list
from utils.delta_time_printer import DeltaTime
from utils.similarity_calcul import SizeFitter, cosine_similarity_chunk, torch_cosine
from cb.code_bert_mlm import CodeBertModel, CodeBertMlmFillMask, ListCodeBertPrediction

VOCAB_DIR = 'pre-trained/codet5-base'
VOCAB_FILE = join(VOCAB_DIR, 'vocab.json')
CODE_T5_MODEL = "Salesforce/codet5-base"
SPACE_TOKEN = "Ġ"
MASK = '<extra_id_0>'
MAX_TOKENS = 500

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


class CodeT5Model(CodeBertModel):
    mask = MASK
    @staticmethod
    def load_vocab(vocab_file):
        import collections
        vocab = collections.OrderedDict()
        if vocab_file.endswith('.json'):
            f = open(vocab_file, )
            import json
            reader = json.load(f)
            for token in reader.keys():
                index = reader[token]
                token = token.encode("ascii", "ignore").decode()
                token = ''.join(token.split())
                vocab[index] = token
            f.close()
        else:
            log.warning('Vocab file cannot be loaded: {0}'.format(vocab_file))
        return vocab

    def init_model_tokenizer(self, pretrained_model_name, vocab_dir, vocab_file):
        if not isdir(vocab_dir) or 0 == len(os.listdir(vocab_dir)) or not isfile(vocab_file):
            self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
            tokenizer_config = {'pretrained_model_name_or_path': pretrained_model_name,
                                'max_len': MAX_TOKENS}
            self.tokenizer = RobertaTokenizer.from_pretrained(**tokenizer_config)
            self.save_pretrained(vocab_dir)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(vocab_dir)

            self.tokenizer = RobertaTokenizer.from_pretrained(vocab_dir)

    def __init__(self, pretrained_model_name, vocab_dir, vocab_file):
        self.tokenizer = None
        self.model = None
        self.init_model_tokenizer(pretrained_model_name, vocab_dir, vocab_file)
        self.vocab_dict = self.load_vocab(vocab_file)
        log.info('num threads in torch:' + str(torch.get_num_threads()))


class CodeT5Function(CodeT5Model):

    def __init__(self, pretrained_model_name, vocab_dir, vocab_file):
        super().__init__(pretrained_model_name, vocab_dir, vocab_file)

    def completion_function(self, arg):
        print('prediction with codeT5')
        input_ids = self.tokenizer(arg['masked_code'], return_tensors="pt")["input_ids"]
        outputs = self.model.generate(input_ids, num_beams=20, num_return_sequences=PREDICTIONS_COUNT,
                                      max_new_tokens=arg['original_token_len'])
        print('context : ', arg['masked_code'])
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def call_func(self, arg):
        return self.completion_function(arg)


class ListCodeT5Prediction(ListCodeBertPrediction):

    def get_original_and_predictions_tokens(self, code_t5_func: CodeT5Function, masked_code, masked_token, suffix,
                                            original_code_tokens, max_size=MAX_TOKENS):
        assert_not_empty(masked_code, masked_token)
        if original_code_tokens is None or len(original_code_tokens) == 0:
            original_code = masked_code.replace(MASK, masked_token)
            original_code_tokens = code_t5_func.tokenize(original_code)
        assert_not_empty(original_code_tokens)
        result = [original_code_tokens]
        for prediction in self.__root__:
            predicted_code = prediction.put_token_inplace(masked_code, suffix)
            predicted_code_tokens = code_t5_func.tokenize(predicted_code)
            result.append(predicted_code_tokens)
        return CbSizeFitter(result, max_size=max_size).fit()


class CodeT5FillMask(CodeT5Function, CodeBertMlmFillMask):
    def __init__(self, predictions_number=PREDICTIONS_COUNT):
        super().__init__(CODE_T5_MODEL, VOCAB_DIR, VOCAB_FILE)
        self.predictions_number = predictions_number

    def call_func(self, arg, batch_size=MAX_BATCH_SIZE):
        global PREDS
        global TOTAL_PRED_TIME
        delta_time = DeltaTime(logging_level=logging.DEBUG)

        if isinstance(arg, list):
            # if there is a list of masked codes
            call_output = [self.call_func(a) for a in arg]
            # at this point we should have a list of list of objects {'tken_str' : str}
            try:
                result = [ListCodeBertPrediction.parse_obj(co) for co in call_output]
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

    # this is not used at all - i am just adding smell here...
