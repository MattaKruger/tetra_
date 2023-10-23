from re import L
from typing import List

from ray import serve
from ray.serve.config import HTTPOptions

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect

from transformers import pipeline
from pydantic import BaseModel


class ClassifyPrompt(BaseModel):
    prompt: str
    labels: List[str]


app = FastAPI()


@serve.deployment
@serve.ingress(app)
class App:
    @app.websocket("/ws")
    async def say_hi(self, websocket: WebSocket):
        await websocket.accept()
        response = ""
        while True:
            data = await websocket.receive_text()
            print(data)
            await websocket.send_text(f"Message text was: {data}")
            classifier = classify(response)
            print(classifier)


@serve.deployment()
class ExtractInfo:
    def __init__(self):
        self.model = pipeline(
            task="zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        )

    @app.websocket("/test")
    async def classify(self, ws: WebSocket, text: str, labels: List[str]):
        await ws.accept()
        try:
            while True:
                model_output = self.model(text, labels)
                return model_output["labels"]  # type: ignore
        except WebSocketDisconnect:
            print("Client disconnected.")


serve.run(App.bind(), port=1010, route_prefix="/llm")


# @serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
# @serve.ingress(app)
# class Translator:
#     def __init__(self):
#         # Load model
#         self.model = pipeline("translation_en_to_fr", model="t5-small")

#     @app.websocket("/")
#     def translate(self, text: str) -> str:
#         # Run inference
#         model_output = self.model(text)

#         # Post-process output to return only the translation text
#         return model_output[0]["translation_text"]  # type: ignore


# @serve.deployment
# class Summarizer:
#     def __init__(self, translator):
#         # Load model
#         self.model = pipeline("summarization", model="t5-small")
#         self.translator: DeploymentHandle = translator.options(use_new_handle_api=True)

#     def summarize(self, text: str) -> str:
#         # Run inference
#         model_output = self.model(text, min_length=5, max_length=15)

#         # Post-process output to return only the summary text
#         return model_output[0]["summary_text"]  # type: ignore

#     async def __call__(self, http_request: Request) -> str:
#         english_text: str = await http_request.json()
#         summary = self.summarize(english_text)

#         translation = await self.translator.translate.remote(summary)
#         return translation
