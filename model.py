# -*- coding: utf-8 -*-
import json
import torch
import numpy as np
# import lightseq.inference as lsi
from loguru import logger
import onnxruntime as ort
from onnxruntime import InferenceSession, SessionOptions
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

with open(r"config.json") as json_file:
    config = json.load(json_file)


def create_model_for_provider(model_path: str, num_threads: int = 1, use_onnx_quant: bool = False) -> InferenceSession:
    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = num_threads
    if use_onnx_quant:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['CPUExecutionProvider'] if ort.get_device() == 'CPU' else ['CUDAExecutionProvider']

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=providers)
    session.disable_fallback()

    return session


class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if config['use_gpu'] and torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])

        # test torch
        # qa_model = AutoModelForQuestionAnswering.from_pretrained(config['tokenizer'])
        # qa_model = qa_model.eval()

        # test onnx
        qa_model = create_model_for_provider(config['model'], num_threads=config['num_threads'], use_onnx_quant=config['use_onnx_quant'])

        self.model = qa_model
        # self.model = lsi.Bert('/srv/www/gzhd/xiekai/cs-document-ai/docparser/nlp/lightseq_chinese_pretrain_mrc_roberta_wwm_ext_large.hdf5', 128)

    def inference(self, question, context):
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=config['add_special_tokens'],
            return_tensors="pt"
        ).to(self.device)
        input_ids = inputs["input_ids"].tolist()[0]

        # test torch
        # logger.info('start model run [torch]')
        # outputs = self.model(**inputs)
        # logger.info('end model run [torch]')
        # answer_start_scores1 = outputs.start_logits
        # answer_end_scores1 = outputs.end_logits
        # answer_start = torch.argmax(answer_start_scores1)
        # answer_stop = torch.argmax(answer_end_scores1) + 1

        # test onnx
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in inputs.items()}
        logger.info('start model run [onnx]')
        answer_start_scores, answer_end_scores = self.model.run(None, dict(inputs_onnx))
        logger.info('end model run [onnx]')
        answer_start = np.argmax(answer_start_scores)
        answer_stop = np.argmax(answer_end_scores) + 1

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_stop])).replace(' ', '').replace('[UNK]',
                                                                                                                '')
        return answer


model = Model()


model = Model()


def get_model():
    return model
