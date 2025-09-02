import time
import asyncio
from datetime import datetime
import re
import openai
import together
import groq

def print_divider(label:str = "", length:int = 100, char:str = '=', include_timestamp: bool = True):
    """
    Print a divider with a label.

    Args:
        label (str): The label to include in the divider.
        length (int): The total length of the divider.
        char (str): The character to use for the divider.
    """
    print(get_divider(label, length, char, include_timestamp=include_timestamp))

def get_divider(label:str = "", length:int = 100, char:str = '=', include_timestamp: bool = True) -> str:
    """
    Get a divider string with a label.

    Args:
        label (str): The label to include in the divider.
        length (int): The total length of the divider.
        char (str): The character to use for the divider.
    Returns:
        divider (str): The formatted divider string.
    """

    text = f' {label} ' if label else ''
    divider = f'{text.center(length, char)}'
    if include_timestamp:
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp = datetime.now().strftime("%H:%M:%S")
        divider = f'[{timestamp}] {divider}'
    return f'\n{divider}\n'

async def wait(backoff_counter: int):
    """
    Wait for a while before retrying. Exponential backoff strategy.

    Args:
        backoff_counter (int): The number of retries so far. Used to calculate the wait time.
    """
    return await asyncio.sleep(2 ** backoff_counter)

def sleep(backoff_counter: int):
    """
    Sleep for a while before retrying. Exponential backoff strategy.

    Args:
        backoff_counter (int): The number of retries so far. Used to calculate the sleep time.
    """
    time.sleep(2 ** backoff_counter)

def sanitize_filename(text:str) -> str:
    """
    Sanitize a string to be used as a filename.
    Replaces non-alphanumeric characters with underscores and removes multiple underscores.

    Args:
        text (str): The string to sanitize.

    Returns:
        sanitized_text (str): The sanitized string.
    """

    text = re.sub(r'[^a-zA-Z0-9_\.-]', '_', text)
    text = re.sub(r'_{2,}', '_', text)
    return text

def openai_exception_handler(
        e: Exception, 
        try_count: int, 
        class_name: str,
        max_tries: int = 5
    ) -> None:
    """
    Handle exceptions during API calls with exponential backoff.
    """

    if try_count >= max_tries:
        print(f"{class_name} - Max retries reached. Raising exception: {repr(e)}")
        raise e

    message = f"{class_name} - "
    if isinstance(e, (openai.RateLimitError, together.error.RateLimitError, groq.RateLimitError)):
        message += f"Rate limit exceeded: {repr(e)}."
    elif isinstance(e, (openai.APIError, together.error.APIError, groq.APIError)):
        message += f"API error: {repr(e)}."
    # elif isinstance(e, (openai.APIConnectionError, together.error.APIConnectionError, groq.APIConnectionError)):
    #     message += f"Connection error: {repr(e)}."
    elif isinstance(e, (openai.Timeout, together.error.Timeout, groq.Timeout)):
        message += f"Timeout error: {repr(e)}."
    elif isinstance(e, (openai.InternalServerError, together.error.ServiceUnavailableError, groq.InternalServerError)):
        message += f"Internal server error: {repr(e)}."
    elif isinstance(e, (openai.APITimeoutError, together.error.ResponseError, groq.APITimeoutError)):
        message += f"API timeout error: {repr(e)}."
    elif isinstance(e, (openai.UnprocessableEntityError, groq.UnprocessableEntityError)):
        message += f"Unprocessable entity error: {repr(e)}."
    else:
        # message += f"Warning: {repr(e)}."
        raise e

    message += f" Retrying ({try_count}/{max_tries})..."
    print(message)