from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from os import environ as env
from src.summarize import summarize_text


app = FastAPI()


class TextRequest(BaseModel):
    """
    Represents a request for text summarization.

    Note: the BART model has max generation length = 142 (default) and min generation length = 56.
    Therefore, the maximum summary length `summary_length` > 56.

    Attributes:
        text (str): The input text to be summarized.
        summary_length (int): The desired length (in tokens) of the summary. 
        exclude_words (List[str]): A list of words to be excluded from the generated summary.
    """
    text: str
    summary_length: int
    exclude_words: List[str]


@app.get("/model")
async def get_model() -> Dict[str, str]:
    """
    Get the model name from the environment.

    Returns:
        Dict[str, str]: A dictionary containing the model name.
    """
    return {"model": env["MODEL"]}



@app.get("/tokenizer")
async def get_tokenizer() -> Dict[str, str]:
    """
    Get the tokenizer name from the environment.

    Returns:
        Dict[str, str]: A dictionary containing the tokenizer name.
    """
    return {"tokenizer": env["TOKENIZER"]}


@app.post("/summarize")
async def summarize_text_request(text_request: TextRequest) -> Dict[str, str]:
    """
    Generate a summary for the given text request.

    Args:
        text_request (TextRequest): An instance of the TextRequest class containing the input text, summary length, and excluded words.

    Returns:
        Dict[str, str]: A dictionary containing the generated summary.
    """
    text_request_dict = text_request.model_dump()
    summary = summarize_text(
        text=text_request_dict["text"], 
        summary_length=text_request_dict["summary_length"], 
        exclude_words=text_request_dict["exclude_words"]
    )
    return {
        "summary" : summary
    }
