from __future__ import annotations
from typing import Literal, TYPE_CHECKING, Optional

from lib.agent import ClientType, Agent
if TYPE_CHECKING:
    from lib.agent import AgentActionQueueState, AgentRetrieverState
    from lib.experiment import ExperimentRetrieverConfig

import pandas as pd
import re

import lib.data
import lib.embedding
import lib.agent_old

class ToolRetriever:
    action_name_retrieve:str = "find_action"

    def __init__(
            self,
            df_embeddings: pd.DataFrame = None,
            df_descriptions: pd.DataFrame = None,
            df_tool_co_occurrences: pd.DataFrame = None,
            include_commonly_used: bool = True,
            top_k:int = 5,
            similarity_threshold: float = None,
            true_tool_max_sim:float = 0.8,
            true_tool_skip_n:int = 0,
            provide_feedback: bool = False,
        ):
        """
        Initializes the ToolRetriever with embeddings, descriptions, and co-occurrences.

        Args:
            df_embeddings (pd.DataFrame): DataFrame containing tool embeddings.
            df_descriptions (pd.DataFrame): DataFrame containing tool descriptions.
            df_tool_co_occurrences (pd.DataFrame): DataFrame containing tool co-occurrences.
            include_commonly_used (bool): Whether to include commonly used tools in retrieval.
            top_k (int): Number of top tools to retrieve.
            similarity_threshold (float): Threshold for similarity to filter tools. A minimum similarity score to consider a tool relevant in regard to the query.
            true_tool_max_sim (float): Filters out tools that are too similar to the true tool.
            true_tool_skip_n (int): Number of top most similar tools to the true tool(s) to exclude.
            provide_feedback (bool): Whether to provide feedback on tool retrieval. Only meaningful if `similarity_threshold` and `top_k` are set.

        Raises:
            ValueError: If df_embeddings or df_descriptions is None.
            ValueError: If df_tool_co_occurrences is None and include_commonly_used is True.
        """

        if df_embeddings is None or df_descriptions is None:
            raise ValueError("Both df_embeddings and df_descriptions must be provided.")
        if df_tool_co_occurrences is None and include_commonly_used:
            raise ValueError("df_tool_co_occurrences must be provided if include_commonly_used is True.")
        
        self.df_embeddings = df_embeddings
        self.df_descriptions = df_descriptions

        self.df_tool_co_occurrences = df_tool_co_occurrences
        self.include_commonly_used = include_commonly_used and df_tool_co_occurrences is not None
        self.top_k = top_k
        self.true_tool_max_sim = true_tool_max_sim
        self.true_tool_skip_n = true_tool_skip_n
        self.similarity_threshold = similarity_threshold
        self.provide_feedback = provide_feedback
        
        self.embedder = lib.embedding.OpenAIEmbeddings()
        self.true_tools: list[str] = None
        self.exclude_true_tools: bool = False
        self.subtract_tools_from_co_occurrence: list[str] = None
        self.retrieved_tools: list[str] = []
        self.query_history: list[str] = []
        self.cache: dict[str, list[float]] = {}
    
    @staticmethod
    def create_from_config(
        config: ExperimentRetrieverConfig,
        df_embeddings: pd.DataFrame = None,
        df_descriptions: pd.DataFrame = None,
        df_tool_co_occurrences: pd.DataFrame = None,
        ) -> ToolRetriever:
        
        return ToolRetriever(
            df_embeddings=df_embeddings,
            df_descriptions=df_descriptions,
            df_tool_co_occurrences=df_tool_co_occurrences,
            include_commonly_used=config.include_commonly_used,
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
            true_tool_max_sim=config.true_tool_max_sim,
            provide_feedback=config.provide_feedback,
        )

    def reset(self):
        self.retrieved_tools.clear()
        self.query_history.clear()
        self.cache.clear()

    def retrieve(
            self, 
            query: str,
            distance_metric: Literal['cosine', 'euclidean', 'manhattan'] = 'cosine',
            state: Optional[AgentRetrieverState] = None
        ) -> str:

        query = query.strip()

        repeated_query_feedback = self._get_repeated_query_feedback_(query, state=state)
        if self.provide_feedback and repeated_query_feedback:
            self._query_history_append_(query, state=state)
            return repeated_query_feedback
        
        retrieved_tools, n_excluded_tools = self.retrieve_dataframe(
            query, 
            distance_metric=distance_metric,
            state=state
        )

        return self._inner_retrieve_(
            retrieved_tools,
            n_excluded_tools,
            state=state
        )

    async def async_retrieve(
            self,
            query: str,
            distance_metric: Literal['cosine', 'euclidean', 'manhattan'] = 'cosine',
            state: Optional[AgentRetrieverState] = None
        ) -> str:
        
        query = query.strip()

        repeated_query_feedback = self._get_repeated_query_feedback_(query, state=state)
        if self.provide_feedback and repeated_query_feedback:
            self._query_history_append_(query, state=state)
            return repeated_query_feedback
        
        retrieved_tools, n_excluded_tools = await self.async_retrieve_dataframe(
            query,
            distance_metric=distance_metric,
            state=state
        )

        return self._inner_retrieve_(
            retrieved_tools,
            n_excluded_tools,
            state=state
        )
    
    def retrieve_dataframe(
            self, 
            query: str, 
            distance_metric: Literal['cosine', 'euclidean', 'manhattan'] = 'cosine',
            state: Optional[AgentRetrieverState] = None
        ) -> tuple[pd.DataFrame, int]:

        self._query_history_append_(query, state=state)

        tool_embed: list[float] = None
        if query in self.cache:
            tool_embed = self.cache[query]
        else:
            tool_embed = self.embedder.get_embedding(query)
            self.cache[query] = tool_embed

        return self._retrieve_dataframe_from_embedding_(
            tool_embed,
            distance_metric=distance_metric,
            state=state,
        )

    async def async_retrieve_dataframe(
            self, 
            query: str, 
            distance_metric: Literal['cosine', 'euclidean', 'manhattan'] = 'cosine',
            state: Optional[AgentRetrieverState] = None
        ) -> tuple[pd.DataFrame, int]:

        self._query_history_append_(query, state=state)

        tool_embed: list[float] = None
        if query in self.cache:
            tool_embed = self.cache[query]
        else:
            tool_embed = await self.embedder.async_get_embedding(query)
            self.cache[query] = tool_embed

        return self._retrieve_dataframe_from_embedding_(
            tool_embed,
            distance_metric=distance_metric,
            state=state,
        )

    def get_available_actions(self, state: Optional[AgentRetrieverState] = None) -> list[str]:
        true_tools = state.true_tools if state is not None else self.true_tools
        exclude_true_tools = state.exclude_true_tools if state is not None else self.exclude_true_tools
        tools: pd.DataFrame = lib.data.get_available_tools(
           embeddings_df=self.df_embeddings,
           true_tool=true_tools,
           include_true_tool=not exclude_true_tools,
           true_tool_max_sim=self.true_tool_max_sim,
           true_tool_skip_n=self.true_tool_skip_n,
        )
        tool_names = tools["tool"].tolist()
        return tool_names
    
    def find_tool(self, tool_name: str):
        tool = self.df_descriptions[self.df_descriptions['tool'] == tool_name]
        if len(tool) == 0:
            return None
        else:
            tool = tool.iloc[0]
            name = tool['tool']
            return f'{name}'

    def _inner_retrieve_(
            self, 
            retrieved_tools: pd.DataFrame,
            n_excluded_tools: int,
            state: Optional[AgentRetrieverState] = None
        ) -> str:
        observation = "Found the following actions, which I can use:\n\n"

        tool_names = retrieved_tools["tool"].tolist()
        tool_descriptions = retrieved_tools["description"].tolist()
        self._retrieved_tools_append_(tool_names, state=state)

        for i in range(len(tool_names)):
            observation += self._get_tool_observation_(tool_names[i], tool_descriptions[i])

        if self.include_commonly_used and len(tool_names) > 0:
            for tool in tool_names:
                commonly_used_tools = self._get_commonly_used_tools_(tool, k=2, state=state)
                
                if not commonly_used_tools:
                    continue
                
                observation += f"\nCommonly used tools for {tool}:\n"
                if commonly_used_tools:
                    self._retrieved_tools_append_(commonly_used_tools, state=state)

                    for common_tool in commonly_used_tools:
                        common_tool_description = self._get_tool_description_(common_tool)
                        observation += self._get_tool_observation_(common_tool, common_tool_description)
                    observation += "\n"

        observation += self._get_feedback_(n_excluded_tools)
        return observation.strip()

    def _retrieve_dataframe_from_embedding_(
            self, 
            embedding: list[float],
            distance_metric: Literal['cosine', 'euclidean', 'manhattan'] = 'cosine',
            state: Optional[AgentRetrieverState] = None
        ) -> tuple[pd.DataFrame, int]:
        """
        Retrieves a DataFrame of tools based on the query and specified parameters.
        
        Args:
            embedding (list[float]): The embedding of the query.
            exclude_tool (list[str]): List of tools to exclude from the retrieval.
            distance_metric (str): The distance metric to use for similarity calculation. Options are 'cosine', 'euclidean', or 'manhattan'.
        
        Returns:
            (dataframe, n_excluded_tools) (tuple[pd.DataFrame, int]): A DataFrame containing the most similar tools and their descriptions, and the number of tools excluded from the results.
        """

        # exclude_tools = state.true_tools if state is not None else self.true_tools
        true_tools = state.true_tools if state is not None else self.true_tools
        exclude_true_tools = state.exclude_true_tools if state is not None else self.exclude_true_tools

        tools, n_excluded_tools = lib.data.find_most_similar_with_descriptions(
            self.df_embeddings,
            self.df_descriptions,
            embedding,
            top_k=self.top_k,
            include_true_tool=not exclude_true_tools,
            true_tool_max_sim=self.true_tool_max_sim,
            true_tool_skip_n= self.true_tool_skip_n,
            true_tool=true_tools,
            distance_metric=distance_metric,
        )
        return tools, n_excluded_tools

    def _retrieved_tools_append_(self, tools: list[str], state: Optional[AgentRetrieverState] = None):
        if state is not None:
            state.retrieved_tools += tools
            state.retrieved_tools = list(set(state.retrieved_tools))
        else:
            self.retrieved_tools += tools
            self.retrieved_tools = list(set(self.retrieved_tools))

    def _query_history_append_(self, query: str, state: Optional[AgentRetrieverState] = None):
        if state is not None:
            state.queue_history.append(query)
        else:
            self.query_history.append(query)

    def _get_commonly_used_tools_(self, tool:str, k: int = 3, state: Optional[AgentRetrieverState] = None) -> list[str]:
        if self.df_tool_co_occurrences is None or self.df_tool_co_occurrences.empty or self.include_commonly_used is False:
            return []
        
        df = self.df_tool_co_occurrences.copy()
        exclude_tool = state.true_tools if state is not None else self.true_tools
        subtract_tools_in_co_occurrence = state.subtract_tool_combination_in_co_occurrence if state is not None else self.subtract_tools_from_co_occurrence

        # If exclude_tool is set, subtract it from the co-occurrence dataframe
        if (exclude_tool is not None and len(exclude_tool) > 2):
            df = lib.data.ToolCoOccurrence.exclude_tool_combination_from_co_occurrence(
                df, exclude_tool,
            )

        if subtract_tools_in_co_occurrence is not None and len(subtract_tools_in_co_occurrence) > 1:
            df = lib.data.ToolCoOccurrence.subtract_k_from_tool_combination_from_co_occurrence(
                df, subtract_tools_in_co_occurrence,
            )

        # Get the top k tools that co-occur with the given tool
        return lib.data.ToolCoOccurrence.get_top_k_co_occurring_tools(
            df, tool, k=k, threshold=0.05,
        )

    def _get_repeated_query_feedback_(self, query:str, state: Optional[AgentRetrieverState] = None) -> str | None:
        if state is not None:
            query_history = state.queue_history
        else:
            query_history = self.query_history

        if not query_history:
            return None

        if query in query_history:
            n_repeated = query_history.count(query)
            if n_repeated > 1:
                return f"\nNote: The query '{query}' has been repeated {n_repeated} times. Consider reformulating the query for better results."

        return None


    def _get_feedback_(self, n_excluded_tools: int) -> str:
        if (not self.provide_feedback 
            or self.similarity_threshold is None 
            or self.top_k is None 
            or self.top_k <= 0 
            or n_excluded_tools <= 0
            ):
            return ""

        observation = (
            f"\nNote: {n_excluded_tools} tools with high similarity (above {self.similarity_threshold}) were excluded due to the top-{self.top_k} limit."
            f"\nSuggestion: If the current toolset seems insufficient, consider reformulating the query with more detail or clarifying the intended tool functionality to improve results."
            "\n"
        )
        return observation.strip()

    def _get_tool_description_(self, tool:str) -> str:
        if self.df_descriptions is None or self.df_descriptions.empty:
            return ""
        
        tool_description = self.df_descriptions.loc[self.df_descriptions['tool'] == tool, 'description']
        if tool_description.empty:
            return ""
        else:
            return tool_description.values[0]

    def _get_tool_observation_(self, tool:str, description: str) -> str:
        observation = f"{tool}:\n"
        observation += f"""e.g. {tool}: {{"action_input": "input"}}\n"""
        observation += f"{description}\n\n"
        return observation
        
class ToolActionQueue:
    ACTION_REGEX = re.compile('^([\w|\.]+)(?:: (.*))?')

    action_name_add:str = "action_queue_add"
    action_name_remove:str = "action_queue_remove"
    action_name_clear:str = "action_queue_clear"
    action_name_list:str = "action_queue_list"
    action_name_run:str = "action_queue_run_all"

    def __init__(self, tool_retriever: ToolRetriever = None):
        self.queue: list[tuple[str, str]] = []
        self.tool_retriever = tool_retriever

    async def add(self, action:str, action_input:str, state: Optional[AgentActionQueueState] = None) -> str:
        if self.tool_retriever and self.tool_retriever.find_tool(action) is None:
            return f"Invalid tool name: {action}. Please provide a valid tool name."
        
        ID = None # 1-indexed ID for the action
        if state is not None:
            state.queue.append((action, action_input))
            ID = len(state.queue)
        else:
            self.queue.append((action, action_input))
            ID = len(self.queue)

        # return f"Added action '{action}' with input '{action_input}' to queue."
        observation = f"Added action '{action}' with input '{action_input}' to queue, assigned ID: {ID}." + f"""

        Action queue commands:
        {ToolActionQueue.action_name_add}:
        e.g. Action: {ToolActionQueue.action_name_add}: {{ "action": "tol name", "action_input": "Tool input" }}
        Adds an action to the action queue (does not run it). 

        {ToolActionQueue.action_name_remove}:
        e.g. Action: {ToolActionQueue.action_name_remove}: {{"ID_str": "ID"}}
        Removes an action from the action queue by ID (1-indexed).

        {ToolActionQueue.action_name_clear}:
        e.g. Action: {ToolActionQueue.action_name_clear}: {{}}
        Clears the action queue.

        {ToolActionQueue.action_name_list}:
        e.g. Action: {ToolActionQueue.action_name_list}: {{}}
        Lists the actions in the action queue.

        {ToolActionQueue.action_name_run}:
        e.g. Action: {ToolActionQueue.action_name_run}: {{}}
        Runs the actions in the action queue and returns the results.
        """

        return observation.strip()
    
    def list(self, state: Optional[AgentActionQueueState] = None) -> str:
        queue = state.queue if state is not None else self.queue

        if not queue or (queue and not (len(queue) > 0)):
            return "No actions in the queue."
        
        return "\n".join([
            f"Action {i+1}: '{action}'"
            for i, (action, action_input) in enumerate(queue)
        ])
    
    def clear(self, state: Optional[AgentActionQueueState] = None) -> str:
        if state is not None:
            state.queue.clear()
        else:
            self.queue.clear()

        return "All actions cleared from the queue."
    
    def remove(self, ID_str: str, state: Optional[AgentActionQueueState] = None) -> str:
        try:
            ID = int(ID_str.strip())
        except ValueError:
            return f"Invalid ID: {ID_str}. Please provide a valid integer ID."
        
        queue = state.queue if state is not None else self.queue

        if ID < 1 or ID > len(queue):
            return f"Invalid ID: {ID}. Please provide a valid ID between 1 and {len(queue)}."

        action, action_input = None, None
        if state is not None:
            action, action_input = state.queue.pop(ID - 1)
        else:
            action, action_input = self.queue.pop(ID - 1)

        return f"Removed action '{action}' with input '{action_input}' from the queue."

    def run(self, state: Optional[AgentActionQueueState] = None) -> str:
        ret = self.list(state=state)

        if state is not None:
            state.queue.clear()
        else:
            self.clear()

        return ret

class ToolActionQueueReflective(ToolActionQueue):
    def __init__(
            self, 
            tool_retriever: ToolRetriever = None,
            question = None, 
            system_prompt:str = "", 
            model_name:str = None, 
            client_type: ClientType = None,
            model_tempature:float = 1):
        super().__init__(tool_retriever = tool_retriever)
        self.question = question
        self.reflection_model = Agent(
            system_prompt=system_prompt, 
            model_name=model_name, 
            client_type=client_type, 
            model_tempature=model_tempature
        )

    async def add(self, action:str, action_input:str, state: Optional[AgentActionQueueState] = None) -> str:
        observation = await super().add(action, action_input, state) + "\n"
        observation += "Reflection on the action queue:\n"
        observation += await self.reflect(observation, state)
        return observation

    async def reflect(self, observation: str, state: AgentActionQueueState = None) -> str:
        """
        Reflect on how well the tool selection can answer the question.
        """
        queue_descriptions = ""
        queue = state.queue if state is not None else self.queue
        for tool_name, _ in queue:
            if self.tool_retriever.find_tool(tool_name) is None:
                continue
            tool_description = self.tool_retriever.df_descriptions.loc[self.tool_retriever.df_descriptions['tool'] == tool_name, 'description'].values[0]
            queue_descriptions += f"{tool_name}: {tool_description}\n"
        
        # Prepare the reflection prompt
        data_dict = {
            "query": state.question,
            "queue": queue_descriptions,
            "latest_observation": observation
        }
        reflection_prompt = lib.agent_old.build_prompt(template=self.reflection_model.system_prompt, data=data_dict)

        # Invoke the reflection model with the prompt
        answer, completion_tokens, prompt_tokens, total_tokens = await self.reflection_model._inner_invoke_(None, reflection_prompt)
        state.usage.prompt_tokens += prompt_tokens
        state.usage.completion_tokens += completion_tokens
        state.usage.total_tokens += total_tokens
        state.usage.llm_calls += 1

        return answer
