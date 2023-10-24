import re
from typing import List
from pydantic import BaseModel
from enum import Enum
import ray
import requests
from fastapi import FastAPI
from ray import serve

from transformers import pipeline

from models import Response, UserResponse, EconsultType

app = FastAPI()


#  Create user response
def construct_response_message(message_type: str, text: str) -> str:
    if message_type == EconsultType.address_change:
        return f"Zal ik je adres wijziging doorvoeren? Nieuw adres: {text}"
    if message_type == EconsultType.medication:
        return f"Zal ik dit recept voor je aanmaken? Recept voor {text}"
    return ""


# Create task and pass to t5 model to extract useful info.
# The flan model handels questions fairly well.
def construct_task(message_type: str, context):
    if message_type == EconsultType.address_change:
        return f"task: Please answer to the following question. What is the dutch address from this sentence: {context}"
    if message_type == EconsultType.medication:
        return f"task: Please answer to the following question. What is the medication and amount from this sentence: {context}"


# We use a BERT based model to classify type of E-consult.
# We're able to use english labels to classify dutch text, so no extra translation is needed.
def classifier(prompt: str) -> str:
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    )
    response = classifier(
        prompt,
        candidate_labels=EconsultType.to_list(),
    )
    return response["labels"][0]  # type: ignore


def extractor(econsult_type: str, context: str) -> str:
    task_pipe = pipeline(
        "text2text-generation", "google/flan-t5-large", max_new_tokens=50
    )
    task = construct_task(econsult_type, context)
    return task_pipe(task)[0]["generated_text"]  # type: ignore


@serve.deployment
@serve.ingress(app)
class MyFastAPIDeployment:
    @app.post("/")
    def root(self, prompt: str):
        econsult_type = classifier(prompt)  # get econsult-type E.G address
        extracted_information = extractor(
            econsult_type=econsult_type, context=prompt  # type: ignore
        )
        user_message = construct_response_message(
            message_type=econsult_type, text=extracted_information  # type: ignore
        )
        return UserResponse(response=user_message, extracted_info=extracted_information)


app = MyFastAPIDeployment.bind()
