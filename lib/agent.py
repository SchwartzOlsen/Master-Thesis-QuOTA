from __future__ import annotations
from typing import Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum, auto

import re
import copy
import json

from groq import Groq, AsyncGroq
from groq.types.chat.chat_completion import ChatCompletion as GroqChatCompletion

from together import Together, AsyncTogether
from together.types.chat_completions import ChatCompletionResponse as TogetherChatCompletionResponse

from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion

import lib

class ClientType(Enum):
    GROQ = auto()
    OPENAI = auto()
    TOGETHER = auto()

class AgentUsageState(BaseModel):
    prompt_tokens: int = 0
    """
    The number of tokens in the prompt.
    """

    completion_tokens: int = 0
    """
    The number of tokens in the completion.
    """

    total_tokens: int = 0
    """
    The total number of tokens (prompt + completion).
    """

    llm_calls: int = 0
    """
    The number of calls made by the agent to a language model.
    """

class AgentRetrieverState(BaseModel):
    queue_history: list[str] = Field(default_factory=list)
    """
    The history of queries made to the retriever.
    """

    retrieved_tools: list[str] = Field(default_factory=list)
    """
    The list of tools retrieved by the retriever.
    """

    true_tools: list[str] = Field(default_factory=list)
    """
    List of true tools to be used in the retriever.
    """

    exclude_true_tools: bool = False
    """
    List of tools to exclude from the retriever.
    This is used to prevent the retriever from retrieving thr true tool(s) in reliability mode.
    """

    subtract_tool_combination_in_co_occurrence: list[str] = Field(default_factory=list)
    """
    List of tools to subtract from the retriever's co-occurrence.
    This is used to exclude the current sample from the retriever's co-occurrence.
    """

    retrieval_banned: bool = False
    """
    If True, the retriever is not allowed to retrieve any tools.
    This is used when the retriever has abused the retrieval action in the past. (Early stopping)
    """

class AgentActionQueueState(BaseModel):
    question: str = ""
    """
    The question being asked to the agent.
    """

    queue: list[tuple[str, str]] = Field(default_factory=list)
    """
    The queue of actions to be executed by the agent.
    
    Each action is a tuple of (action_name, action_input).
    """

    usage: AgentUsageState = Field(default_factory=AgentUsageState)
    """
    The usage statistics for the agent's actions.
    """

class AgentState(BaseModel):
    """
    Represents the state of an agent.
    """
    
    question: str = ""
    """
    The question being asked to the agent.
    """
    
    messages: list[dict[str, str]] = Field(default_factory=list)
    """
    The conversation history between the user and the agent.
    """

    log_text: str = ""
    """
    The log text of the agent's actions and responses.
    """

    usage: AgentUsageState = Field(default_factory=AgentUsageState)
    """
    The usage statistics for the agent's actions.
    """

    tools_chosen: list[str] = Field(default_factory=list)
    """
    The list of tools/actions chosen by the agent during the conversation.
    """
    
    turn: int = 0
    """
    The current turn number in the conversation.
    """
    
    system_prompt: str = ""
    """
    The system prompt that guides the agent's behavior.
    """

    persistant: bool = False
    """
    If True, the agent will persist until a correct action is chosen.
    
    If False, the agent will stop when it has chosen an action.
    """

    true_tools: list[str] = Field(default_factory=list)
    """
    List of correct actions to be used when persist_till_correct_action is True.
    
    Only relevant if `persist_till_correct_action` is True.
    """

    verbose: bool = False
    """
    If True, the agent will print the log to the console.
    """

    retriever_state: AgentRetrieverState = Field(default_factory=AgentRetrieverState)
    """
    The state of the retriever, if any.
    """

    action_queue_state: AgentActionQueueState = Field(default_factory=AgentActionQueueState)
    """
    The state of the action queue, if any.
    """

    last_actions_performed: list[tuple[str, str]] = Field(default_factory=list)
    """
    The last actions performed by the agent.
    This is used to check if the last list of actions performed by the agent is the same as the current list of actions.
    """


    def log(self, message: str):
        """
        Log a message to the agent state.

        Args:
            message (str): The message to log.
        """
        self.log_text += str(message) + "\n"
        if self.verbose:
            print(message)

    @staticmethod
    def create(
        question: str,
        system_prompt: str = "", 
        verbose: bool = False, 
        persistant: bool = False,
        true_tools: list[str] = [],
        retriever_true_tools: list[str] = [],
        retriever_exclude_true_tools: bool = False,
        retriever_subtract_tools: list[str] = [],
        ) -> AgentState:
        
        state = AgentState(
            question=question,
            system_prompt=system_prompt,
            persistant=persistant,
            true_tools=true_tools,
            verbose=verbose,
        )

        state.retriever_state.true_tools = retriever_true_tools
        state.retriever_state.exclude_true_tools = retriever_exclude_true_tools
        state.retriever_state.subtract_tool_combination_in_co_occurrence = retriever_subtract_tools

        state.action_queue_state.question = question

        return state

class Agent():
    MESSAGE_LENGTH_LIMIT = 20

    _client_ = None
    @property
    def client(self):
        """
        Returns the client instance being used by the agent.
        """
        if self._client_ is None:
            if self.client_type.value == ClientType.GROQ.value:
                self._client_ = AsyncGroq()
            elif self.client_type.value == ClientType.OPENAI.value:
                self._client_ = AsyncOpenAI()
            elif self.client_type.value == ClientType.TOGETHER.value:
                self._client_ = AsyncTogether()
            else:
                raise ValueError(f"Unsupported client type \"{self.client_type}\". Supported types are Groq, OpenAI, and Together.")
        
        return self._client_

    def __init__(
            self, 
            system_prompt:str = "", 
            model_name:str = None, 
            client_type: ClientType = None,
            model_tempature:float = 1,
            ):
        
        if client_type is None:
            raise ValueError("Client not set. Please provide a valid OpenAI or Groq client.")
        if model_name is None:
            raise ValueError("Model name not set. Please provide a valid model name.")

        self.client_type = client_type
        if not isinstance(self.client, (Groq, AsyncGroq, OpenAI, AsyncOpenAI, Together, AsyncTogether)):
            raise ValueError("Client must be an instance of Groq, OpenAI, or Together.")

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.model_tempature = model_tempature
        self.reset()

    def reset(self):
        self.messages = [
            # {"role": "system", "content": self.system_prompt}
        ]

    async def invoke_state(self, question:str, state: AgentState):
        """
        Invoke the agent with a question and return the answer, using the provided state.

        Args:
            question (str): The question to be asked.
            state (AgentState): The state of the agent, containing system prompt and conversation history.

        Returns:
            answer (str): The answer to the question.
        """
        
        if state is None or not isinstance(state, AgentState):
            raise ValueError("Invalid state provided. Please provide a valid AgentState.")

        answer, completion_tokens, prompt_tokens, total_tokens = await self._inner_invoke_(question, state.system_prompt, state.messages)

        state.usage.completion_tokens += completion_tokens
        state.usage.prompt_tokens += prompt_tokens
        state.usage.total_tokens += total_tokens
        state.usage.llm_calls += 1

        state.messages.append({"role": "user", "content": question})
        state.messages.append({"role": "assistant", "content": answer})
        
        return answer
    
    async def invoke(self, question:str):
        """
        Invoke the agent with a question and return the answer.
        
        Args:
            question (str): The question to be asked.

        Returns:
            (answer, state) (tuple[str, AgentState]): The answer to the question and the updated state of the agent.
        """
        if not question:
            question = " "

        answer, completion_tokens, prompt_tokens, total_tokens = await self._inner_invoke_(question, self.system_prompt, self.messages)
        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    async def _inner_invoke_(self, question:str, system_prompt:str, conversation_history: list[dict[str, str]] = []):
        # if not question:
        #     question = " "

        if not system_prompt:
            raise ValueError("System prompt not set. Please provide a valid system prompt.")
        
        message_edge_length_limit = Agent.MESSAGE_LENGTH_LIMIT // 2
        messages = (
            [{"role": "system", "content": system_prompt}]
            + (conversation_history[:message_edge_length_limit] + conversation_history[-message_edge_length_limit:] if len(conversation_history) > Agent.MESSAGE_LENGTH_LIMIT else conversation_history) 
            + ([{"role": "user", "content": question}] if question else [])
        )

        completion = await self._call_client_(messages)
        result = str(completion.choices[0].message.content)
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0

        if isinstance(completion, (GroqChatCompletion, OpenAIChatCompletion, TogetherChatCompletionResponse)):
            completion_tokens = completion.usage.completion_tokens
            prompt_tokens = completion.usage.prompt_tokens
            total_tokens = completion.usage.total_tokens

        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        # print(completion.usage)

        return result, completion_tokens, prompt_tokens, total_tokens

    async def _call_client_(self, messages: list):
        MAX_TRIES = 6
        try_count = 0

        MODEL = self.model_name
        TEMPATURE = self.model_tempature
        TIMEOUT = 10

        while try_count < MAX_TRIES:
            try_count += 1
            try:
                if isinstance(self.client, Groq):
                    return self.client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=TEMPATURE,
                        timeout=TIMEOUT,
                    )
                elif isinstance(self.client, AsyncGroq):
                    return await self.client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=TEMPATURE,
                        timeout=TIMEOUT,
                    )
                elif isinstance(self.client, OpenAI):
                    return self.client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=TEMPATURE,
                        timeout=TIMEOUT,
                    )
                elif isinstance(self.client, AsyncOpenAI):
                    return await self.client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=TEMPATURE,
                        timeout=TIMEOUT,
                    )
                elif isinstance(self.client, Together):
                    return self.client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        timeout=TIMEOUT,
                    )
                elif isinstance(self.client, AsyncTogether):
                    return await self.client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        timeout=TIMEOUT,
                    )
                else:
                    raise ValueError(f"Unsupported client type: {type(self.client)}. Supported types are Groq, OpenAI, and Together.")
            except Exception as e:
                lib.utils.openai_exception_handler(
                    e, 
                    try_count,
                    class_name=self.__class__.__name__,
                    max_tries=MAX_TRIES
                )
                await lib.utils.wait(try_count)
                # lib.utils.sleep(try_count)

class ReActAgent(Agent):
    # action_re = re.compile('^Action(?:\s*\d*\s*)*:\s*([\w|\.|\&|\-]+)(?::\s*(.*))?', re.MULTILINE)
    # action_re = re.compile('^Action(?:\s*\d*\s*)*:\s*([\w|\.|\&|\-]+)(?::\s*(\{.*\}))?', re.MULTILINE)
    action_regex_str = '^Action(?:\s*\d*\s*)*:\s*([\w|\.|\&|\-]+)(?::\s*(\{.*\}))?'
    tools_chosen = []

    def __init__(
            self, 
            system_prompt:str = "", 
            model_name:str = None, 
            client_type: ClientType = None,
            actions: dict[str, callable] = {},
            tool_retriever: lib.tools.ToolRetriever = None,
            action_queue: lib.tools.ToolActionQueue = None,
            force_action_queue:bool = False,
            persist_till_true_tool_chosen:bool = False,
            model_tempature:float = 1.0,
            run_queue_automatically:bool = False,
            max_turns:int = 20,
            ):
        """
        Initialize the ReAct Agent.
        Args:
        - system_prompt (str): The system prompt to be used by the agent.
        - model_name (str): The name of the model to be used by the agent.
        - client: The client to be used by the agent.
        - actions (dict): A dictionary of actions to be used by the agent.
        - enable_tool_retriever (bool): If True, the agent can use a tool retriever to find external actions / tools.
        - enable_action_queue (bool): If True, the agent can use an action queue to store actions.
        - force_action_queue (bool): If True, the agent add the chosen action to the action queue instead of executing it.
        - persist_till_true_tool_chosen (bool): If True, the agent will persist until a correct action is chosen.
        - true_tools (list): List of correct actions to be used when persist_till_correct_action is True.
        - verbose (bool): If True, the agent will print the log to the console.
        """

        super().__init__(system_prompt=system_prompt, model_name=model_name, client_type=client_type, model_tempature=model_tempature)
        
        self.known_actions = actions
        
        self.tool_retriever = tool_retriever
        """
        If given, the agent can use a tool retriever to find external actions / tools.
        """

        self.action_queue = action_queue
        """
        If given, the agent can use an action queue to store actions.
        """

        self.force_action_queue = force_action_queue
        """
        If True, the agent add the chosen action to the action queue instead of executing it.

        If False, the agent will execute the action immediately if the action is directly called.

        Only relevant if `action_queue` is given.
        """

        self.persist_till_true_tool_chosen = persist_till_true_tool_chosen
        """
        If True, the agent will persist until a correct action is chosen.

        If False, the agent will stop when it has chosen an action.
        """
        
        self.run_queue_automatically = run_queue_automatically
        """
        If True, the agent will always run any actions in the action queue before returning an answer.
        """
        
        self.max_turns = max_turns

        if self.tool_retriever is not None:
            self.known_actions[lib.tools.ToolRetriever.action_name_retrieve] = None

        if self.action_queue is not None:
            self.known_actions[lib.tools.ToolActionQueue.action_name_add] = None
            self.known_actions[lib.tools.ToolActionQueue.action_name_clear] = None
            self.known_actions[lib.tools.ToolActionQueue.action_name_list] = None
            self.known_actions[lib.tools.ToolActionQueue.action_name_run] = None
            self.known_actions[lib.tools.ToolActionQueue.action_name_remove] = None
        
        self.reset()

    def reset(self):
        super().reset()
        if hasattr(self,'action_queue') and self.action_queue is not None:
            self.action_queue.clear()
        if hasattr(self,'tool_retriever') and self.tool_retriever is not None:
            self.tool_retriever.reset()

    async def invoke(
            self,
            question,
            system_prompt: Optional[str] = None,
            true_tools: list[str] = [],
            reliability_mode: bool = False,
            verbose: bool = False,
            state: Optional[AgentState] = None,
        ):
        """
        Query the agent with a question and return the answer.
        
        Args:
        - question (str): The question to be asked.
        - max_turns (int): The maximum number of turns to be taken by the agent.
        - persist_till_true_tool_chosen (bool): If True, the agent will persist until a correct action is chosen.
        - true_tools (list): List of correct actions to be used when persist_till_correct_action is True.
        - verbose (bool): If True, the agent will print the log to the console.
        
        Returns:
        - answer (str): The answer to the question.
        """

        state = state if state is not None else AgentState.create(
            question=question,
            system_prompt=system_prompt if system_prompt else self.system_prompt,
            persistant=(self.persist_till_true_tool_chosen and not reliability_mode),
            true_tools=true_tools,
            verbose=verbose,
            retriever_true_tools=true_tools,
            retriever_exclude_true_tools=reliability_mode,
            retriever_subtract_tools=[] if reliability_mode else true_tools,
        )

        self.reset()

        state.log(lib.utils.get_divider("Prompt"))
        state.log(state.system_prompt)

        state.log(lib.utils.get_divider("Query"))
        state.log(question)

        next_prompt = question
        should_continue = True
        answer = None

        while (should_continue) and (state.turn < self.max_turns):
            state.turn += 1
            responce = await self.invoke_state(next_prompt, state=state)
            before, sep, _ = responce.partition("PAUSE")
            responce_trimmed = before + sep

            state.log(lib.utils.get_divider("AI Agent"))
            state.log(responce_trimmed)

            action_re = re.compile(self.action_regex_str, re.MULTILINE)
            actions = [
                action_re.match(a).groups() for a in
                responce_trimmed.split('\n') if action_re.match(a)
            ]

            # actions_is_repeated = actions and self._actions_performed_are_same_as_last_(actions, state)

            # if actions_is_repeated and self.tool_retriever is not None and lib.tools.ToolRetriever.action_name_retrieve in [a[0] for a in actions]:
            #     state.retriever_state.retrieval_banned = True

            state.last_actions_performed = actions.copy()
            
            if actions:
                next_prompt = ""
                for j, (action, action_input) in enumerate(actions,1):
                    observation, should_continue_temp = await self._handle_action_(
                        action, 
                        action_input, 
                        state, 
                        j
                    )
                    should_continue = should_continue and should_continue_temp
                    
                    if (len(next_prompt) > 0) and (len(observation) > 0):
                        next_prompt += "\n\n"
                    next_prompt += observation

                state.log(lib.utils.get_divider(f"Observation (continues: {should_continue})"))
                state.log(next_prompt)

                if not should_continue and len(state.tools_chosen) > 0:
                    answer = None # Final answer will be generated below
                    break
                elif not should_continue:
                    answer = next_prompt
                    break
            # If there are queued actions, we can assume the agent wants to continue
            elif self.run_queue_automatically and self.action_queue is not None and len(state.action_queue_state.queue) > 0:
                queued_actions = [a[0] for a in state.action_queue_state.queue].copy()
                for action in queued_actions:
                    state.tools_chosen.append(action)
                break
            # If no actions were found in the newest response, we can assume the agent has finished
            else:
                answer = (
                    "\n".join([f"Action {i}: {action}" for i, action in enumerate(state.tools_chosen, 1)])
                    if len(state.tools_chosen) > 0 else "No action chosen."
                )
                break

        if answer is None:
            answer = (
                "\n".join([f"Action {i}: {action}" for i, action in enumerate(state.tools_chosen, 1)])
                if len(state.tools_chosen) > 0 else "No action chosen. Max turns reached."
            )

        state.log(lib.utils.get_divider("Final Answer"))
        state.log(answer)
        return answer, state
    
    async def _handle_action_(
            self, 
            action: str, 
            action_input: str, 
            state: AgentState,
            index: int = 1,
        ) -> tuple[str, bool]:
        
        """
        Handle the action by running it and returning the observation.

        Args:
            action (str): The action to run.
            action_input (str): The input for the action.
            index (int): The index of the observation.

        Returns:
            tuple[str, bool]: The observation and a boolean indicating if the agent should continue.
        """
        observation = ""
        should_continue = True

        args: dict[str, str] = {}
        try:
            args = json.loads(action_input)
        except Exception as e:
            observation = f"ERROR - Invalid action input: {action_input}. Please provide a valid JSON object.\nexception: {e}\n"
            observation += "Remember actions have the following format:\nAction: <tool_name>: {<tool_input>}\n"
            return observation, should_continue

        if self.tool_retriever is not None and action == lib.tools.ToolRetriever.action_name_retrieve:
            if state.retriever_state.queue_history and state.retriever_state.queue_history.count(action_input) > 1:
                state.retriever_state.retrieval_banned = True

            if state.retriever_state.retrieval_banned:
                observation = (
                    f"Due to unnecessary use of the '{lib.tools.ToolRetriever.action_name_retrieve}' action, you are no longer allowed to use that action.\n"
                    f"Please choose a different action to continue, or try using one of the retrieved tools to complete the task.\n"
                )

                return observation, should_continue

        # Check if the action is an external action
        external_tool = self.tool_retriever.find_tool(action) if self.tool_retriever is not None else None
        if external_tool is not None and (not self.force_action_queue or self.action_queue is None):
            # If persist_till_correct_action is True and the action is not in the correct actions
            # we need to handle the external tool selection
            if state.persistant and (len(state.true_tools) > 0):
                return self._handle_external_tool_selection_([external_tool], state)

            # Else we can return the observation, and the agent should now stop
            return f"Selected action: {external_tool}\n", False
        elif external_tool is not None and self.force_action_queue and self.action_queue is not None:
            observation = await self.action_queue.add(action=action, action_input=action_input, state=state.action_queue_state)
            return observation, should_continue
        
        # If the action is not an external action, check if it's a known action
        if action not in self.known_actions:
            observation = "ERROR - Unknown action: {}: {}".format(action, action_input)
            return observation, should_continue
        try:
            # If it's a known action, run it
            if self.action_queue is not None and action == lib.tools.ToolActionQueue.action_name_run:
                if state.persistant and (len(state.true_tools) > 0):
                    actions = [a[0] for a in state.action_queue_state.queue].copy()
                    self.action_queue.clear(state=state.action_queue_state)
                    return self._handle_external_tool_selection_(actions, state)
                return self.action_queue.run(state=state.action_queue_state), False
            elif self.action_queue is not None and action == lib.tools.ToolActionQueue.action_name_list:
                observation = self.action_queue.list(state=state.action_queue_state)
            elif self.action_queue is not None and action == lib.tools.ToolActionQueue.action_name_clear:
                observation = self.action_queue.clear(state=state.action_queue_state)
            elif self.action_queue is not None and action == lib.tools.ToolActionQueue.action_name_remove:
                observation = self.action_queue.remove(**args, state=state.action_queue_state)
            elif self.action_queue is not None and action == lib.tools.ToolActionQueue.action_name_add:
                observation = await self.action_queue.add(**args, state=state.action_queue_state)
            elif self.tool_retriever is not None and action == lib.tools.ToolRetriever.action_name_retrieve:
                observation = await self.tool_retriever.async_retrieve(**args, state=state.retriever_state)
            else:
                observation = self.known_actions[action](**args)
        except Exception as e:
            observation = f"ERROR - Failed to run action: {action} with input: {action_input}. Exception: {e}"
            return observation, should_continue

        return f'observation {index}: {observation}', should_continue

    def _handle_external_tool_selection_(self, actions:list[str], state: AgentState) -> tuple[str, bool]:
        """
        Handle the external tool selection by checking if the action is in the correct actions.

        Args:
            actions (list[str]): The actions to check.

        Returns:
            tuple[str, bool]: The observation and a boolean indicating if the agent should continue.
        """

        # If persist_till_true_tool_chosen is False, we can return the observation and the agent should stop
        if not (state.persistant and (len(state.true_tools) > 0)):
            return "", True
        
        def get_tool_response(action:str):
            if action in state.true_tools:
                # return f'<{action} is a relevant action, Pretend it gave some relevant information. Note that this is a simulation>'
                return f'<{action} is a relevant action, Pretend it gave some relevant information, but not enough, so continue to fullfill the task. Do not call this action again>'
            else:
                return f'<{action} is not a relevant action, Pretend it did not give any relevant information. Do not call this action again>'
        
        for action in actions:
            state.tools_chosen.append(action)

        # If all correct tools have been chosen, we can stop.
        should_continue = not all([a in state.tools_chosen for a in state.true_tools])

        if should_continue:
            response = "\n".join([
                f"Action {i}: '{action}' gave the following response: {get_tool_response(action)}\n"
                for i, action in enumerate(actions, 1)
            ]) if len(actions) > 0 else "No actions chosen."
        else:
            state.log(f"[INFO] All correct tools have been chosen: {state.true_tools}")
            response = "\n".join([
                f"Action {i}: {action}"
                for i, action in enumerate(actions, 1)
            ]) if len(actions) > 0 else "No actions chosen."

        return response, should_continue
    
    # make a function that checks if the last list of actions performed by the agent is the same as the current list of actions
    def _actions_performed_are_same_as_last_(
            self,
            actions: list[tuple[str, str]],
            state: AgentState,
            ) -> bool:
        """
        Check if the last list of actions performed by the agent is the same as the current list of actions.

        Args:
            actions (list[tuple[str, str]]): The list of actions to check.
            state (AgentState): The state of the agent, containing the last actions performed.

        Returns:
            bool: True if the last list of actions is the same as the current list of actions, False otherwise.
        """
        if len(state.last_actions_performed) != len(actions):
            return False
        for i, action in enumerate(actions):
            if action != state.last_actions_performed[i]:
                return False
        return True

class SingleStepReActAgent(ReActAgent):
    # action_re = re.compile('^Action(?:\s*\d*\s*)*:\s*([\w|\.|\&|\-]+)(?::\s*(\{.*\}))?', re.MULTILINE)
    action_re_str = '^Action(?:\s*\d*\s*)*:\s*([\w|\.|\&|\-]+)(?::\s*(\{.*\}))?'
    """
    A simple agent that can be used to query a model with a single step.
    """

    def __init__(
            self, 
            system_prompt:str = "", 
            retriever_prompt:str = "",
            model_name:str = None, 
            client_type: ClientType = None,
            tool_retriever: lib.tools.ToolRetriever = None,
            retrieval_enabled: bool = False,
            model_tempature:float = 1.0,
            ):
        super().__init__(
            system_prompt=system_prompt, 
            model_name=model_name,
            tool_retriever=tool_retriever,
            client_type=client_type, 
            model_tempature=model_tempature, 
            max_turns=1
        )
        
        self.retriever_prompt = retriever_prompt
        self.retrieval_enabled = retrieval_enabled

    async def invoke(
            self,
            question: str,
            system_prompt: Optional[str] = None,
            retriever_prompt: Optional[str] = None,
            true_tools: list[str] = [],
            reliability_mode: bool = False,
            verbose: bool = False,
            state: Optional[AgentState] = None,
        ):
        """
        Query the agent with a question and return the answer.
        
        Args:
        - question (str): The question to be asked.
        - system_prompt (str): The system prompt to be used by the agent.
        - retriever_prompt (str): The prompt to be used for the retriever.
        - persist_till_true_tool_chosen (bool): If True, the agent will persist until a correct action is chosen.
        - true_tools (list): List of correct actions to be used when persist_till_correct_action is True.
        - verbose (bool): If True, the agent will print the log to the console.
        
        Returns:
        - answer (str): The answer to the question.
        """
        state = state if state is not None else AgentState.create(
            question=question,
            system_prompt=system_prompt if system_prompt else self.system_prompt,
            true_tools=true_tools,
            verbose=verbose,
            retriever_true_tools=true_tools,
            retriever_exclude_true_tools=reliability_mode,
            retriever_subtract_tools=[] if reliability_mode else true_tools,
        )

        self.reset()

        # Insert the tool descriptions into the system prompt
        retrieval_queries: list[str] = []
        if self.retrieval_enabled:
            # Prompt a LLM to retrieve tools based on the question.
            retrieval_response, completion_tokens, prompt_tokens, total_tokens = await self._inner_invoke_(question, retriever_prompt if retriever_prompt else self.retriever_prompt, state.messages)
            
            state.usage.completion_tokens += completion_tokens
            state.usage.prompt_tokens += prompt_tokens
            state.usage.total_tokens += total_tokens

            request_tool_descriptions = [tool.strip() for tool in re.findall(r'Tool(?: \d+)?:\s*(.*)', retrieval_response)]
            # use the retrieval response as a fallback
            if len(request_tool_descriptions) == 0:
                request_tool_descriptions = [retrieval_response]
            retrieval_queries = request_tool_descriptions
        else:
            # If no retrieval is enabled, we use the true tools as the retrieval queries
            retrieval_queries = [self.tool_retriever.df_descriptions.loc[self.tool_retriever.df_descriptions['tool'] == tool, 'description'].values[0] for tool in true_tools]
        
        tool_descriptions = ""
        for retrieval_query in retrieval_queries:
            tool_descriptions += await self.tool_retriever.async_retrieve(retrieval_query, state=state.retriever_state)
    
        state.system_prompt = state.system_prompt.replace("{tool_descriptions}", tool_descriptions)

        state.log(lib.utils.get_divider("Prompt"))
        state.log(state.system_prompt)

        state.log(lib.utils.get_divider("Query"))
        state.log(question)

        response = await self.invoke_state(question, state=state)
        before, sep, _ = response.partition("PAUSE")
        response_trimmed = before + sep

        state.log(lib.utils.get_divider("AI Agent"))
        state.log(response_trimmed)

        action_re = re.compile(self.action_regex_str, re.MULTILINE)
        state.tools_chosen = [
            " ".join(g for g in action_re.match(a).groups() if g) for a in
            response_trimmed.split('\n') if action_re.match(a)
        ]

        answer = (
            "\n".join([f"Action {i}: {action}" for i, action in enumerate(state.tools_chosen, 1)])
            if len(state.tools_chosen) > 0 else "No action chosen."
        )

        state.log(lib.utils.get_divider("Final Answer"))
        state.log(answer)
        return answer, state

#Endregion

class Planner(Agent):

    def __init__(
            self,
            system_prompt: str = "You are a planner agent. Your task is to plan the steps needed to achieve a goal.",
            model_name: str = "gpt-4",
            client_type: ClientType = None,
            verbose: bool = False
            ) -> None:
        super().__init__(
            system_prompt=system_prompt, model_name=model_name, client_type=client_type
        )
        self.verbose = verbose

    def _parse_expression_(self, expression: str) -> dict:
        stack = []
        current = {}
        for token in re.findall(r'Step \d+|AND|OR|\(|\)', expression):
            if token.startswith('Step'):
                if 'steps' not in current:
                    current['steps'] = []
                current['steps'].append(int(token.split()[1]))
            elif token in ('AND', 'OR'):
                current['logic'] = token
            elif token == '(':
                stack.append(current)
                current = {}
            elif token == ')':
                closed = current
                current = stack.pop()
                if 'steps' not in current:
                    current['steps'] = []
                current['steps'].append(closed)
        return current

    def _plan_to_args_(self, plan:str, keyword = 'Step', lkey = 'execution order'):
        args = []
        lines = plan.split('\n')
        for line in lines:
            if line.startswith(keyword): args.append(re.sub(r'{} \d+: '.format(keyword), '', line))
            if lkey in line.lower():
                logic = line.split(': ')[-1]
        args_lookup = {i+1: args[i] for i in range(len(args))}
        try:
            return self._fetch_args_(args_lookup, self._parse_expression_(logic))
        except: 
            return {'steps': args, 'logic': 'AND'}

    def _print_completion_(self, completion:str):
        # out_lines = ['\t-----']
        out_lines = []
        lines = completion.split('\n')
        lines = ['\t' + line for line in lines]
        out_lines.extend(lines)
        # out_lines.append('\t-----')
        out_text = "\n".join(out_lines)
        print(out_text)
        return

    def _fetch_args_(self, args_lookup: dict, logic_exp: dict) -> dict:
        out = copy.deepcopy(logic_exp)
        assert 'steps' in logic_exp.keys()
        for s, step in enumerate(logic_exp['steps']):
            if isinstance(step, int):
                out['steps'][s] = args_lookup[step]
            elif isinstance(step, dict):
                out['steps'][s] = self._fetch_args_(args_lookup, step)
        return out

    async def plan(self, task: str) -> tuple[list[str], Callable]:
        # Generate a plan for the task
        plan = await self.invoke(self.system_prompt + '\n\n' + task)

        if self.verbose:
            lib.utils.print_divider("Planner")
            self._print_completion_(plan)

        plan_steps = self._plan_to_args_(plan)
        if self.verbose:
            lib.utils.print_divider("Plan Steps")
            self._print_completion_(str(plan_steps))
        
        # Generate a function that combines the results of the sub-tasks
        logic_function = await self.invoke(f"Generate a function that combines the results of the sub-tasks for: {task}")
        
        return plan, logic_function

class Adapt:
    """
    ADAPT(·) is a recursive algorithm that generates success heuristic value for a task T.

    Links:
    - [Source](https://allenai.github.io/adaptllm/)
    - [Code](https://github.com/archiki/ADaPT)
    - [Paper](https://arxiv.org/abs/2311.05772)
    """

    def __init__(
            self, 
            executor: Agent, 
            planner: Planner,
            controller: Agent,
            max_depth:int = 5,
        ):
        self.executor = executor
        self.planner = planner
        self.controller = controller
        self.max_depth = max_depth

    async def invoke(self, task:str, current_depth:int = 1):
        # ADAPT(·) Generates success heuristic value completed for the task T. Initialized with k = 1.
        
        # Base case: terminate on reaching maximum depth
        if current_depth > self.max_depth:
            return False
        
        # Execute the task/sub-task to assess if the LLM can directly perform it using LLM-generated success.

        completed = await self.executor.invoke(task)
        if completed is False:
            # Using the LLM, decompose the task into a set of sub-tasks, P, 
            # and a Boolean function, logic(·), 
            # that combines output of the sub-tasks.
            P, LOGIC = await self.planner.plan(task)

            # Get the outputs for individual sub tasks
            O = [
                await self.executor.invoke(sub_task)
                for sub_task in P
            ]

            # Combine the outputs of the sub tasks
            completed = LOGIC(O)
        return completed
