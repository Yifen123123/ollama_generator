import json
from ollama import Client


class OllamaExtractor:
    def __init__(self, host: str, model_name: str):
        self.client = Client(host=host)
        self.model_name = model_name

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        temperature: float = 0.0,
        num_ctx: int = 32768,
        num_predict: int = 1200,
    ) -> dict:
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            format=schema,
            options={
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
            keep_alive=-1,
        )
        content = response["message"]["content"]
        return json.loads(content)
