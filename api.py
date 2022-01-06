# -*- coding: utf-8 -*-
from fastapi import FastAPI, Depends
from model import Model, get_model
from loguru import logger
from pydantic import BaseModel

app = FastAPI()


class QuestionAnsweringRequest(BaseModel):
    question: str
    context: str


class QuestionAnsweringResponse(BaseModel):
    answer: str


@app.post('/questionAnswering', response_model=QuestionAnsweringResponse, name="阅读理解模型推理")
async def question_answering(request: QuestionAnsweringRequest, model: Model = Depends(get_model)):
    logger.info('开始调用阅读理解模型进行推理')
    logger.info(f'Question: {request.question}')
    logger.info(f'Context: {request.context}')
    answer = model.inference(request.question, request.context)
    logger.info(f'Answer: {answer}')
    return QuestionAnsweringResponse(
        answer=answer
    )
