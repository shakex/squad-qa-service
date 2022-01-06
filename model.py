# -*- coding: utf-8 -*-
import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

with open(r"config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if cfg.get('qa_server_use_gpu') and torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        qa_model = AutoModelForQuestionAnswering.from_pretrained(config['model_name'])
        qa_model = qa_model.eval()
        self.model = qa_model.to(self.device)

    def inference(self, question, context):
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)
        input_ids = inputs["input_ids"].tolist()[0]

        with torch.no_grad():
            outputs = self.model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits
            answer_start = torch.argmax(answer_start_scores)
            answer_stop = torch.argmax(answer_end_scores) + 1
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_stop])).replace(' ', '').replace('[UNK]',
                                                                                                                '')
        return answer


def get_model():
    return Model()
