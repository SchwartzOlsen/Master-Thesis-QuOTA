import os
import openai
import pandas as pd
from tqdm import tqdm
import time
import lib

def build_prompt(template: str, data: dict[str, str]) -> str:
    """
    Builds a prompt from a template and data.
    :param template: The template string with placeholders for data.
    :param data: The data to fill in the placeholders.
    """
    for key in data:
        # check if data[key] is a string
        if isinstance(data[key], str):
            template = template.replace("{" + key + "}", data[key])
    return template

def format_tool_descriptions(tool_names, descriptions_df):
    # Filter descriptions_df to include only the tools in tool_names
    filtered_df = descriptions_df[descriptions_df["tool"].isin(tool_names)]

    # Create combined descriptions in the format "tool: description"
    combined_descriptions = filtered_df.apply(lambda row: f"{row['tool']}: {row['description']}", axis=1)

    return "\n".join(combined_descriptions.tolist())

class Model():
    def __invoke_client__(self, messages: list) -> str:
        """
        Invoke the model with the given arguments.

        Args:
            messages (list): A list of messages to include in the conversation.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def query(self, system_prompt: str, messages: list = [], **kwargs) -> str:
        """
        Queries the LLM model with a given prompt and messages.

        Args:
            system_prompt (str): The system prompt to initialize the conversation. Required.
            messages (list): A list of messages to include in the conversation. Defaults to an empty list.

        Remarks:
            - The system prompt will be used as the first message in the conversation.
        """
        system_prompt:str = system_prompt or kwargs.get("system_prompt", "")
        messages: list = messages or kwargs.get("messages", [])
        
        if not system_prompt:
            raise ValueError("System prompt is required. Please provide a valid prompt.")

        result = self.__invoke_client__(
            [{"role": "system", "content": system_prompt}] + messages,
        )
        
        return result

class GPTModel(Model):
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        """
        Initializes the GPTModel with a specified model and API key.
        :param model: The name of the LLM model to use (default: gpt-3.5-turbo).
        :param api_key: The OpenAI API key. If None, it should be set in the environment.
        """
        super().__init__()
        self.model = model
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI()

    def __invoke_client__(self, messages: list) -> str:
        """
        Invokes the LLM model with the given messages.

        Args:
            messages (list): A list of messages to include in the conversation.

        Remarks:
            - The system prompt will be used as the first message in the conversation.
        """
        if not self.client:
            raise ValueError("Client not set. Please provide a valid OpenAI client.")
        
        max_tries = 6
        try_count = 0

        while try_count < max_tries:
            try_count += 1
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                # Uncomment this to print out token usage each time, e.g.
                # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
                # print(completion.usage)

                return completion.choices[0].message.content.strip()
            except openai.RateLimitError as e:
                print(f"GPTModel - Rate limit exceeded: {repr(e)}. Retrying...")
                time.sleep(2 ** try_count)  # Exponential backoff
                if try_count >= max_tries:
                    raise e
            except openai.APIError as e:
                print(f"GPTModel - API error: {repr(e)}. Retrying...")
                time.sleep(2 ** try_count)  # Exponential backoff
                if try_count >= max_tries:
                    raise e
            except openai.APIConnectionError as e:
                print(f"GPTModel - Connection error: {repr(e)}. Retrying...")
                time.sleep(2 ** try_count)  # Exponential backoff
                if try_count >= max_tries:
                    raise e
            except Exception as e:
                print(f"GPTModel - Error: {repr(e)}. Retrying...")
                if try_count >= max_tries:
                    raise e


class Agent:
    def __init__(self, model: Model):
        """
        Initializes the Agent with a specified model.
        :param model: An instance of a Model subclass.
        """
        self.model = model

    def query(self, system_prompt: str, messages: list = [], **kwargs) -> tuple[str, list]:
        # """
        # Queries the model with a given prompt.
        # :param prompt: The input prompt for the model.
        # :return: The model's response as a string.
        # """
        return (self.model.query(system_prompt, messages, **kwargs), [])

class ReactAgent(Agent):
    def __init__(self, model: Model):
        """
        Initializes the Agent with a specified model.
        :param model: An instance of a Model subclass.
        """
        super().__init__(model)
        with open(os.path.join("prompts", "react_single_step_prompt_single.md"), "r") as file:
            self.template = file.read()

    def query(self, **args: dict[str, str]) -> tuple[str, list]:
        """
        Queries the model with a given prompt and formats the response for React.
        :param prompt: The input prompt for the model.
        :return: The model's response as a JSON object.
        """
        prompt = build_prompt(self.template, args)
        response, retreived_tools = super().query(system_prompt=prompt)
        return (response, retreived_tools)

def run_experiment(agent: Agent, test_df: pd.DataFrame, exclude_true = False) -> pd.DataFrame:
    # add 'action_res' and 'retrieved_tools' columns to test_df
    test_df['action_res'] = None
    test_df['retrieved_tools'] = None

    for i, data in tqdm(test_df.iterrows(), total=len(test_df), desc="Running Experiment"):
        # Query model
        args = data[['query', 'tool_descriptions', 'tool']].to_dict()
        args['exclude_true'] = exclude_true
        response, retrieved_tools = agent.query(**args)
        
        # Add the response to the DataFrame
        test_df.at[i, 'action_res'] = response
        test_df.at[i, 'retrieved_tools'] = retrieved_tools

    # Extract used tools from the response
    test_df['action_res_tools_used'] = test_df.apply(lambda x: lib.data.MetaToolResults.evaluate_tool_selection(
        x['action_res'], x['available_tools'], x['tool'], True), axis=1)
    
    return test_df

if __name__ == "__main__":
    # Instantiate the agent with a GPTModel
    model = GPTModel(model="gpt-3.5-turbo")
    agent = Agent(model=model)
    dataset = [
        {"country": "France", "capital": "Paris"},
        {"country": "My country", "capital": "Copenhagen"},
    ]
    for data in dataset:
        # Build a prompt from the template and
        prompt = build_prompt("What is the capital of {country}?", data)

        # Example query
        response = agent.query(query=prompt)
        correct = response.find(data["capital"]) != -1
        print(response, "✅" if correct else "❌")