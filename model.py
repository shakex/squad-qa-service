# -*- coding: utf-8 -*-
import json
import torch
# import lightseq.inference as lsi
from loguru import logger
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

with open(r"config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if config['use_gpu'] and torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'])
        qa_model = AutoModelForQuestionAnswering.from_pretrained(config['model'])
        qa_model = qa_model.eval()
        self.model = qa_model.to(self.device)
        # self.model = lsi.Bert('/srv/www/gzhd/xiekai/cs-document-ai/docparser/nlp/lightseq_chinese_pretrain_mrc_roberta_wwm_ext_large.hdf5', 128)

    def inference(self, question, context):
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=config['add_special_tokens'],
            return_tensors="pt"
        ).to(self.device)
        input_ids = inputs["input_ids"].tolist()[0]

        with torch.no_grad():
            outputs = self.model(**inputs)
            # outputs = self.model.infer(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
            answer_start = torch.argmax(answer_start_scores)
            answer_stop = torch.argmax(answer_end_scores) + 1
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_stop])).replace(' ', '').replace('[UNK]',
                                                                                                                '')
        return answer


model = Model()


def get_model():
    return model
