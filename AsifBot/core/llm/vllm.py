import requests
import json

from asifbot import config

class VLLMClient:
    def __init__(self):
        self.vllm_server = f"http://{config.HOST}:{config.ENDPOINTS.vllm_endpoint}/generate"
        
    def generate(
        self,
        prompt: str, 
        use_beam_search: bool = True,
        n: int = 4,
        temperature: int = 0
    ):
        payload = {
            "prompt": prompt,
            "use_beam_seach": use_beam_search,
            "n": n,
            "temperature": temperature
        }
        json_payload = json.dumps(payload)

        headers = {
            "Content-Type": "application/json"
        }

        # Send the POST request
        response = requests.post(self.vllm_server, data=json_payload, headers=headers)

        return response