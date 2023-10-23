import re
from typing import List
from pydantic import BaseModel
from enum import Enum
import ray
import requests
from fastapi import FastAPI
from ray import serve

from transformers import pipeline

app = FastAPI()
dutch_address_regex = "^(\d{4}[A-Z]{2}) ([\w\.'\/\- ]+) (\w?[0-9]+[a-zA-Z0-9\- ]*)"


class Response(BaseModel):
    sequence: str
    labels: List[str]
    scores: List[float]
    response: str


class EconsultType(str, Enum):
    # TODO look at optimizing to_list initialize
    # when adding new types also add them to the to_list method
    address_change = "address_change"
    medication = "medication"
    insurance = "insurance"

    @staticmethod
    def to_list() -> List[str]:
        return ["address_change", "medication", "insurance"]


#  Create user response
def construct_response_message(message_type: str, text: str):
    if message_type == EconsultType.address_change:
        return f"Zal ik je adres wijziging doorvoeren? Nieuw adres: {text}"
    if message_type == EconsultType.medication:
        return f"Zal ik dit recept voor je aanmaken? Recept voor {text}"


def construct_task(message_type: str, context):
    if message_type == EconsultType.address_change:
        return f"task: Extract address from given text, text: {context}"
    if message_type == EconsultType.medication:
        return f"task: Extract medication from given text, text: {context}"


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


def extractor(econsult_type: str, context: str):
    task_pipe = pipeline(
        "text2text-generation", model="google/flan-t5-base", max_new_tokens=50
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
        return construct_response_message(
            message_type=econsult_type, text=extracted_information  # type: ignore
        )


app = MyFastAPIDeployment.bind()
