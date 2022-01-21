# -*- coding: utf-8 -*-
import json
import torch
import numpy as np
from loguru import logger
import onnxruntime as ort
from onnxruntime import InferenceSession, SessionOptions
from transformers import AutoTokenizer

with open(r"config.json") as json_file:
    config = json.load(json_file)


def create_model_for_provider(model_path: str, num_threads: int = 1, use_onnx_quant: bool = False) -> InferenceSession:
    '''
    Loads the model and prepares the CPU backend.
    
    :param model_path: Path to the ONNX model
    :type model_path: str
    :param num_threads: The number of threads to use for the CPU backend, defaults to 1
    :type num_threads: int (optional)
    :param use_onnx_quant: bool = False, defaults to False
    :type use_onnx_quant: bool (optional)
    :return: The InferenceSession object.
    '''
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
        qa_model = create_model_for_provider(config['model'], num_threads=config['num_threads'],
                                             use_onnx_quant=config['use_onnx_quant'])

        self.model = qa_model

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
        logger.info(f'Span: ({answer_start},{answer_stop})')

        answer = ''
        if answer_start == 0 and answer_stop != 1:
            # start未预测成功
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_stop-10:answer_stop])).replace(' ', '').replace(
                '[UNK]', '')
        elif answer_start != 0 and answer_stop == 1:
            # stop未预测成功
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_start+10])).replace(' ', '').replace(
                '[UNK]', '')
        elif answer_start == 0 and answer_stop == 1:
            # start和stop都未预测成功
            answer = ''
        else:
            if answer_start < answer_stop:
                answer = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_stop])).replace(' ', '').replace('[UNK]',
                                                                                                                    '')
            else:
                # start > stop，认为stop预测错误
                answer = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_start + 10])).replace(' ',
                                                                                                             '').replace(
                    '[UNK]', '')
        return answer


model = Model()


def get_model():
    return model
