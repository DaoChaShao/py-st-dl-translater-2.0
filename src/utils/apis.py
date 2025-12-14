#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/28 22:44
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   apis.py
# @Desc     :   

from base64 import b64decode
from openai import OpenAI


def api_key_checker(api_key: str, category: str = "OpenAI") -> bool:
    """ Check the API key format
            - The length of an OpenAI API key is 164-digit key

    :param api_key: enter the API key of the deepseek
    :param category: the category of the API key
    :return: True if the API key is valid
    """
    if api_key.startswith("sk-"):
        if category == "OpenAI" and len(api_key) == 164:
            return True
        elif category == "DeepSeek" and len(api_key) == 35:
            return True
    return False


class OpenAIEmbedder(object):

    def __init__(self, api_key: str) -> None:
        """ Initialize the OpenAI Embeddings API

        :param api_key: str: The API key for the OpenAI API
        """
        self._api_key = api_key

    def client(self, prompt: list, model: str, dimensions: int = 1024) -> list:
        """ Initialize the OpenAI Embeddings API
                - dimensions: 256、512、1024、1536

        :param dimensions: int: The number of dimensions for the embedding
        :param model: str: The model to use for the embedding
        :param prompt: list: The input text to be embedded
        :return: None
        """
        client = OpenAI(api_key=self._api_key, base_url="https://api.openai.com/v1")

        response = client.embeddings.create(
            input=prompt,
            model=model,
            dimensions=dimensions,
            encoding_format="float",
            timeout=3,
        )

        return [item.embedding for item in response.data]


class OpenAITextCompleter(object):
    """ OpenAI Completer API Wrapper """

    def __init__(self, api_key: str, temperature: float = 1.0, top_p: float = 1.0) -> None:
        """ Initialise the OpenAI Hyperparameter Tuning API
        :param api_key: str: The API key for the OpenAI API
        :param temperature: float: The temperature for the completion
        :param top_p: float: The top-p for the completion
        """
        self._api_key = api_key
        self._temperature = temperature
        self._top_p = top_p

    def client(self, content: str, prompt: str, model: str = "gpt-4.1-mini") -> str:
        """ Initialise the OpenAI Completion API
        :param content: str: The input text to be completed
        :param prompt: str: The prompt to complete the input text
        :param model: str: The model to use for the completion
        :return: str: The completed text
        """
        client = OpenAI(api_key=self._api_key, base_url="https://api.openai.com/v1")

        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt},
        ]
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=self._temperature,
            top_p=self._top_p,
        )

        return completion.choices[0].message.content


class DeepSeekCompleter(object):
    """ DeepSeek Completer API Wrapper """

    def __init__(self, api_key: str, temperature: float = 0.7) -> None:
        """ Initialise the DeepSeek Completer API
        :param api_key: str: The API key for the DeepSeek API
        :param temperature: float: The temperature for the completion
        """
        self._api_key = api_key
        self._temperature = temperature

    def client(self, content: str, prompt: str, model: str = "deepseek-chat") -> str:
        """ Initialise the DeepSeek Completion API
        :param content: str: The input text to be completed
        :param prompt: str: The prompt to complete the input text
        :param model: str: The model to use for the completion, default is "deepseek-chat-3.5"
        :return: str: The completed text
        """
        client = OpenAI(api_key=self._api_key, base_url="https://api.deepseek.com")
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self._temperature,
            stream=False)

        return response.choices[0].message.content


class OpenAIImageCompleter(object):
    """ OpenAI Image Generation API Wrapper """

    def __init__(self, api_key: str, model: str = "dall-e-3", seed=None) -> None:
        """ Initialise the OpenAI Image API
        :param api_key: str: The API key for the OpenAI API
        :param model: str: The model to use (e.g. "dall-e-3", "gpt-image-1", "dall-e-2")
        :param seed: int: The seed for reproducibility
        """
        self._api_key = api_key
        self._model = model
        self._seed = seed
        self._client = OpenAI(api_key=self._api_key, base_url="https://api.openai.com/v1")

    def client(
            self, prompt: str, quality: str = "standard", resolution="1024x1024", filename: str = "outer"
    ) -> None:
        """ Generate image(s) with the given model
        :param prompt: str: The text prompt
        :param quality: str: "standard" or "hd" (only works for dall-e-3)
        :param resolution: str: e.g. "1024x1024", "1024x1792"
        :param filename: str: The output filename
        """
        result = self._client.images.generate(
            prompt=prompt,
            model=self._model,
            n=1,
            quality=quality,
            size=resolution,
            style="natural",
            response_format="b64_json",
        )

        image_base64 = result.data[0].b64_json
        image_bytes = b64decode(image_base64)

        image_path: str = f"{filename}.png"
        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)
        print(f"The image {filename} has been saved successfully.")


if __name__ == "__main__":
    pass
