import asyncio
import logging
from queue import Empty

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from ray import serve

logger = logging.getLogger("ray.serve")

fastapi_app = FastAPI()


@serve.deployment
@serve.ingress(fastapi_app)
class Chatbot:
    def __init__(self, model_id: str):
        self.loop = asyncio.get_running_loop()

        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    @fastapi_app.websocket("/")
    async def handle_request(self, ws: WebSocket) -> None:
        await ws.accept()

        conversation = ""
        try:
            while True:
                prompt = await ws.receive_text()
                logger.info(f'Got prompt: "{prompt}"')
                conversation += prompt
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    timeout=0,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                self.loop.run_in_executor(
                    None, self.generate_text, conversation, streamer
                )
                response = ""
                async for text in self.consume_streamer(streamer):
                    await ws.send_text(text)
                    response += text
                await ws.send_text("<<Response Finished>>")
                conversation += response
        except WebSocketDisconnect:
            print("Client disconnected.")

    def generate_text(self, prompt: str, streamer: TextIteratorStreamer):
        input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids
        self.model.generate(input_ids, streamer=streamer, max_length=10000)

    async def consume_streamer(self, streamer: TextIteratorStreamer):
        while True:
            try:
                for token in streamer:
                    logger.info(f'Yielding token: "{token}"')
                    yield token
                break
            except Empty:
                await asyncio.sleep(0.001)


app = Chatbot.bind("microsoft/DialoGPT-small")
