from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ray import serve
from ray.serve.handle import DeploymentHandle

from transformers import pipeline

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@serve.deployment()
@serve.ingress(app)
class FastAPIDeployment:
    # FastAPI will automatically parse the HTTP request for us.
    def __init__(self, deberta_model_handle) -> None:
        self.deberta_handle: DeploymentHandle = deberta_model_handle.options(
            use_new_handle_api=True
        )

    @app.websocket("/")
    async def say_hello(self, ws: WebSocket):
        await ws.accept()
        try:
            while True:
                text = await ws.receive_text()
                classifiers = self.deberta_handle.classify.remote(text)
                print(classifiers)
        except WebSocketDisconnect:
            print("Client disconnected.")


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
)
class ExtractInfo:
    def __init__(self):
        self.classifier = pipeline(
            task="zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        )

    def classify(self, sentence: str):
        return self.classifier(sentence)

    # @app.websocket("/test")
    # async def classify(self, ws: WebSocket, text: str, labels: List[str]):
    #     await ws.accept()
    #     try:
    #         while True:
    #             model_output = self.model(text, labels)
    #             return model_output["labels"]  # type: ignore
    #     except WebSocketDisconnect:
    #         print("Client disconnected.")


f_app = FastAPIDeployment.bind(ExtractInfo.bind())
# serve.run(FastAPIDeployment.bind(), route_prefix="/hello")


# 3: Query the deployment and print the result.
# from websockets.sync.client import connect


# "Hello Theodore!"
