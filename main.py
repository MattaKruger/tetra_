from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from regex import R
from transformers import pipeline
from ray import serve

from models import UserResponse, EconsultType

app = FastAPI()

# We use a BERT based model to classify type of E-consult.
# We're able to use english labels to classify dutch text, so no extra translation is needed.
deberta_v3_base = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# The flan model handels questions fairly well.
google_flan_t5_xl = "google/flan-t5-xl"


def construct_response_message(message_type: str, data: str) -> str:
    if message_type == EconsultType.address_change:
        return f"Zal ik je adres wijziging doorvoeren? Nieuw adres: {data}"
    if message_type == EconsultType.medication:
        return f"Zal ik dit recept voor je aanmaken? Recept voor {data}"
    else:
        return ""


def construct_task(message_type: str, context):
    if message_type == EconsultType.address_change:
        return f"Please answer to the following question. What is the dutch address including postal code from this sentence: {context}"
    if message_type == EconsultType.medication:
        return f"Please answer to the following question. What is the medication and amount from this sentence: {context}"
    else:
        return ""


def classifier(prompt: str) -> str:
    classifier = pipeline(
        "zero-shot-classification", deberta_v3_base, trust_remote_code=False
    )
    response = classifier(
        prompt,
        candidate_labels=EconsultType.to_list(),
    )
    return response["labels"][0]  # type: ignore


def extractor(econsult_type: str, context: str) -> str:
    task_pipe = pipeline(
        "text2text-generation",
        google_flan_t5_xl,
        max_new_tokens=50,
        trust_remote_code=False,
    )
    task = construct_task(econsult_type, context)
    return task_pipe(task)[0]["generated_text"]  # type: ignore


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
@serve.ingress(app)
class MyFastAPIDeployment:
    @app.post("/")
    def root(self, prompt: str):
        econsult_type = classifier(prompt)
        extracted_information = extractor(
            econsult_type=econsult_type, context=prompt  # type: ignore
        )
        user_message = construct_response_message(
            message_type=econsult_type, data=extracted_information  # type: ignore
        )
        return UserResponse(response=user_message, extracted_info=extracted_information)

    @app.websocket("/post")
    async def post(self, ws: WebSocket):
        await ws.accept()
        try:
            while True:
                text = await ws.receive_text()
                econsult_type = classifier(text)  # get econsult-type E.G address
                extracted_information = extractor(
                    econsult_type=econsult_type, context=prompt  # type: ignore
                )
                user_message = construct_response_message(
                    message_type=econsult_type, data=extracted_information  # type: ignore
                )
                await ws.send_text(user_message)
        except WebSocketDisconnect:
            print("Client disconnected")


app = MyFastAPIDeployment.bind()
