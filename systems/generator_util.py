"""
This file contains the Generator classes and generator factory.
"""
from __future__ import annotations

import os

from openai import OpenAI

# from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
import PyPDF2

OpenAIModelList = ["gpt-4o", "gpt-4o-mini", "gpt-4o-v", "gpt-4o-mini-v"]
TogetherModelList = ["google/gemma-2b-it", "meta-llama/Llama-2-13b-chat-hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"]

def get_api_key(key: str) -> str:
    # get API key from environment or throw an exception if it's not set
    if key not in os.environ:
        print(f"KEY: {key}")
        print(f"{os.environ.keys()}")
        raise ValueError("key not found in environment variables")

    return os.environ[key]


class Generator():
    def __init__(self, model: str, verbose=False):
        self.model = model
        if model in OpenAIModelList:
            self.client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
        elif model in TogetherModelList:
            self.client = Together(api_key=get_api_key("TOGETHER_API_KEY"))

        self.verbose = verbose

    def generate(self, prompt: str) -> str:
        if self.verbose:
            print(f"Generating with {self.model}")
            print(f"Prompt: {prompt}")
        if isinstance(self.client, OpenAI):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4000,
            )
            return response.choices[0].message.content
        elif isinstance(self.client, Together):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
            )
            return response.choices[0].message.content
        else:
            raise Exception(f"Unsupported model: {self.model}")
        

def pdf_to_text(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

