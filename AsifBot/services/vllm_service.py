import subprocess

from asifbot import config

vllm = [
    'python -m vllm.entrypoints.api_server',
    f'--model={config.LLM.llm}',
    f'--port={config.ENDPOINTS.vllm_endpoint}'
]

subprocess.run(vllm)