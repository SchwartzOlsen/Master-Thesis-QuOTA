from __future__ import annotations
# from typing import Literal, TYPE_CHECKING
# if TYPE_CHECKING:
#     from lib.agent import AgentActionQueueState, AgentRetrieverState, ReActAgent

import re
import os
import pandas as pd
import numpy as np
import time
import gc
import json
from typing import Optional
from pydantic import BaseModel
import matplotlib.lines as mlines
from enum import Enum, auto
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime
from tqdm import tqdm

import openai
import matplotlib.pyplot as plt
import asyncio
import lib

CONFIGFILENAME = 'config.json'
SIMILARRESULTFILENAME = 'similar.pkl'
RELIABILITYRESULTFILENAME = 'reliability.pkl'
MULTITOOLRESULTFILENAME = 'multi.pkl'
SIMILARLOGFILENAME = 'similar_log.pkl'
RELIABILITYLOGFILENAME = 'reliability_log.pkl'
MULTITOOLLOGFILENAME = 'multi_log.pkl'
DFCONFIGCOLUMN = 'config'
DFDIRPATHCOLUMN = 'dirpath'

class ExperimentSetupType(Enum):
    REACT = auto()
    """
    Default ReAct implementation.
    """
    
    REACT_SINGLE_STEP = auto()
    """
    ReAct with single-step execution (no loop).
    """

    REACT_WITH_CO_OCCURRENCE = auto()
    """
    ReAct with co-occurrence in tool retrieval.
    """

    QUOTA = auto()
    """
    Quota. Same as ReAct, but with Action Queue and co-occurrence in tool retrieval.
    """

    QUOTA_WITH_REFLECTION = auto()
    """
    Quota with reflection when adding new tools to queue.
    """

class ExperimentRetrieverConfig(BaseModel):
    include_commonly_used: Optional[bool] = None
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    true_tool_max_sim: Optional[float] = None
    provide_feedback: Optional[bool] = None

    @staticmethod
    def from_retriever(retriever: lib.tools.ToolRetriever) -> ExperimentRetrieverConfig:
        return ExperimentRetrieverConfig(
            include_commonly_used=retriever.include_commonly_used,
            top_k=retriever.top_k,
            similarity_threshold=retriever.similarity_threshold,
            true_tool_max_sim=retriever.true_tool_max_sim,
            provide_feedback=retriever.provide_feedback,
        )

class ExperimentAgentConfig(BaseModel):
    llm_name: Optional[str] = None
    llm_temperature: Optional[float] = None
    client_type: Optional[int] = None
    max_turns: Optional[int] = None
    action_queue_enabled: Optional[bool] = None
    action_queue_forced: Optional[bool] = None
    persist_till_true_tool_chosen: Optional[bool] = None
    run_queue_automatically: Optional[bool] = None

    @staticmethod
    def from_agent(agent: lib.agent.ReActAgent) -> ExperimentAgentConfig:
        return ExperimentAgentConfig(
            llm_name=agent.model_name,
            llm_temperature=agent.model_tempature,
            client_type=agent.client_type.value if agent.client_type is not None else None,
            max_turns=agent.max_turns,
            action_queue_enabled=agent.action_queue is not None,
            action_queue_forced=agent.force_action_queue,
            persist_till_true_tool_chosen=agent.persist_till_true_tool_chosen,
            run_queue_automatically=agent.run_queue_automatically,
        )

class ExperimentConfig(BaseModel):
    name: str
    description: str
    plot_title: str
    date_time: datetime
    source: str
    agent_config: ExperimentAgentConfig
    tool_retriever_config: Optional[ExperimentRetrieverConfig] = None
    n_samples_single: Optional[int] = None
    n_samples_multi: Optional[int] = None

    def generate_dirname(self) -> str:
        """
        Generate a filename for the experiment config based on the name and date_time.
        """
        dirname = lib.utils.sanitize_filename(self.name)

        agent_config = self.agent_config if self.agent_config is not None else self.agent_config

        if (self.tool_retriever_config is not None 
            and self.tool_retriever_config.include_commonly_used is not None
            and self.tool_retriever_config.include_commonly_used
            ):
            dirname += "_co_occurence"

        if agent_config is not None:
            if agent_config.action_queue_enabled is not None and agent_config.action_queue_enabled:
                dirname += "_queue"
                if agent_config.action_queue_forced is not None and agent_config.action_queue_forced:
                    dirname += "_forced"
            if agent_config.persist_till_true_tool_chosen is not None and agent_config.persist_till_true_tool_chosen:
                dirname += "_persist"
            if agent_config.llm_name is not None:
                dirname += f"_{lib.utils.sanitize_filename(agent_config.llm_name)}"

        if self.source is not None:
            dirname += f"_{lib.utils.sanitize_filename(self.source)}"
        if self.date_time is not None:
            dirname += f"_{self.date_time.strftime('%Y%m%d_%H%M%S')}"

        return dirname

    def generate_filepath(self) -> str:
        """
        Generate a filename for the experiment config based on the name and date_time.
        """
        dirpath = self.generate_dirname()
        filepath = os.path.join(dirpath, CONFIGFILENAME)
        return filepath
    
    def save_to_file(self, dirpath: str = 'results') -> str:
        """
        Save the experiment config to a JSON file.
        """

        full_dirpath = os.path.join(dirpath, self.generate_dirname())
        if not os.path.exists(full_dirpath):
            os.makedirs(full_dirpath)
        
        filepath = os.path.join(dirpath, self.generate_filepath())
        with open(filepath, 'w') as f:
            f.write(self.model_dump_json(indent=2))
        
        return filepath

    @staticmethod 
    def create(
        name: str,
        description: str,
        plot_title: str,
        source: str,
        date_time: datetime,
        agent: lib.agent.ReActAgent,
        n_samples_single: int,
        n_samples_multi: int,
    ) -> ExperimentConfig:
        config = ExperimentConfig(
            name=name,
            description=description,
            plot_title=plot_title,
            date_time=date_time,
            source=source,
            n_samples_single=n_samples_single,
            n_samples_multi=n_samples_multi,
            agent_config=ExperimentAgentConfig.from_agent(agent),
        )
        tool_retriever = agent.tool_retriever if agent.tool_retriever is not None else agent.tool_retriever
        if tool_retriever is not None and isinstance(tool_retriever, lib.tools.ToolRetriever):
            config.tool_retriever_config = ExperimentRetrieverConfig.from_retriever(tool_retriever)
        return config
    
    @staticmethod
    def from_filepath(filepath: str) -> ExperimentConfig:
        """
        Load an experiment config from a JSON file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file '{filepath}' does not exist.")
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        return ExperimentConfig(**config_data)

def _plot_metrics_radar_(
        similar_metrics, 
        reliability_metrics, 
        multi_metrics, 
        combined_metrics, 
        config: ExperimentConfig,
        plot_title: str = 'Results', 
        save_dirpath: str = None,
    ):

    # Plot metrics for each task
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), subplot_kw=dict(polar=True))  # Create a 2x2 grid of polar plots

    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
    metrics = [
        similar_metrics,
        reliability_metrics,
        multi_metrics,
        combined_metrics
    ]
    titles = ['General', 'Reliability', 'Multitool', 'Combined']
    colors = ['blue', 'red', 'green', 'purple']
    # Call make_graph() with each subplot
    for i, ax in enumerate(axes):
        _, _ = lib.data.plot_metrics_radar(metrics[i], titles[i], ax=ax, backgroundcolor=colors[0])  # Pass the current polar subplot

    # Plot title
    fig.suptitle(f'{plot_title}, {config.n_samples_single} single samples, {config.n_samples_multi} multi samples, {config.agent_config.llm_name}', fontsize=16, fontweight='bold')

    plt.tight_layout()  # Adjust layout
    if save_dirpath is not None:
        if not os.path.exists(save_dirpath):
            os.makedirs(save_dirpath)
        plt.savefig(os.path.join(save_dirpath, 'radar_plots.pdf'))
    plt.show()

async def run_experiment(
        agent: lib.agent.ReActAgent, 
        test_df: pd.DataFrame, 
        reliability_mode = False,
        description = "Experiment",
        prompt: Optional[str] = None,
        radar_plot_save_dirpath: Optional[str] = None,
        concurrency_limit: Optional[int] = 32,
        task_batch_size: Optional[int] = 500,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    concurrency_limit = min(concurrency_limit, task_batch_size) 
    sem = asyncio.Semaphore(concurrency_limit)
    
    test_df['available_tools'] = None
    test_df['retrieved_tools'] = None
    test_df['tool_retriever_query_history'] = None
    test_df['turns'] = None


    # test_df['chat_log'] = None
    # test_df['state'] = None
    test_df['action_res_tools_used'] = None
    test_df['action_res'] = None

    test_df['usage_llm_calls'] = None
    test_df['usage_prompt_tokens'] = None
    test_df['usage_completion_tokens'] = None
    test_df['usage_total_tokens'] = None

    test_df_log = test_df[['query']].copy()
    test_df_log['chat_log'] = None
    # test_df_log['state'] = None

    # If the DataFrame is empty, return it immediately
    if test_df.empty:
        print("DataFrame is empty. Returning empty DataFrame.")
        return test_df, test_df_log

    async def process_row(i, data):
        async with sem:
            tool = data['tool']
            tools: list[str] = []
            if isinstance(tool, str):
                tools = [tool]
            elif isinstance(tool, list):
                tools = tool
            else:
                raise ValueError(f"Tool should be a string or a list of strings, but got {type(tool)}")
            
            # Query model
            query = data['query']
            response, state = await agent.invoke(
                query, 
                true_tools=tools,
                reliability_mode=reliability_mode,
                verbose=False,
                system_prompt=prompt
            )
            
            return i, response, state

    n_rows = len(test_df)
    indices_processed = []
    pbar = tqdm(total=n_rows, desc=description)
    # pbar = tqdm(test_df.iterrows(), total=len(test_df), desc=description)
    # for i, data in pbar:

    # Schedule all tasks
    # tasks = [process_row(i, row) for i, row in test_df.iterrows()]

    for start in range(0, n_rows, task_batch_size):
        end = min(start + task_batch_size, n_rows)

        tasks = [process_row(i, test_df.iloc[i]) for i in range(start, end) if i < n_rows]

        for f in asyncio.as_completed(tasks):
            res_tuple: tuple[int, str, lib.agent.AgentState] = None
            
            max_tries = 5
            try_count = 0
            failed = False

            while try_count < max_tries:
                try_count += 1

                try:
                    res_tuple = await f
                    break  # Exit loop if successful
                except Exception as e:
                    if try_count >= max_tries:
                        failed = True
                        break

                    print(f"Error processing row: {e}. Retrying {try_count}/{max_tries}...")
                    
            if failed or res_tuple is None:
                print(f"Failed to process row after {max_tries} retries. Skipping.")
                continue

            i, response, state = res_tuple
            indices_processed.append(i)

            # Add the response to the DataFrame
            test_df.at[i, 'action_res'] = response
            test_df.at[i, 'turns'] = state.turn
            test_df.at[i, 'available_tools'] = agent.tool_retriever.get_available_actions(state=state.retriever_state)
            test_df.at[i, 'retrieved_tools'] = state.retriever_state.retrieved_tools.copy()
            test_df.at[i, 'tool_retriever_query_history'] = state.retriever_state.queue_history.copy()

            test_df.at[i, 'usage_total_tokens'] = state.usage.total_tokens + state.action_queue_state.usage.total_tokens
            test_df.at[i, 'usage_prompt_tokens'] = state.usage.prompt_tokens + state.action_queue_state.usage.prompt_tokens
            test_df.at[i, 'usage_completion_tokens'] = state.usage.completion_tokens + state.action_queue_state.usage.completion_tokens
            test_df.at[i, 'usage_llm_calls'] = state.usage.llm_calls + state.action_queue_state.usage.llm_calls

            # test_df.at[i, 'chat_log'] = state.log_text
            # test_df.at[i, 'state'] = state.model_dump_json(indent=2)
            test_df_log.at[i, 'chat_log'] = state.log_text
            # test_df_log.at[i, 'state'] = state.model_dump_json(indent=2)

            # Extract used tools from the response
            test_df.at[i, 'action_res_tools_used'] = lib.data.MetaToolResults.evaluate_tool_selection(
                test_df.at[i, 'action_res'],
                test_df.at[i, 'available_tools'], 
                test_df.at[i, 'tool'],
                True
            )
            
            metrics = lib.data.compute_action_metrics(
                test_df.iloc[indices_processed], filter_ground_truth=reliability_mode, max_turns=agent.max_turns
            )

            pbar.set_postfix(metrics)
            pbar.update(1)

            if radar_plot_save_dirpath is not None:
                plot_title = f'{description} ({i+1}/{len(test_df)})' if i+1 < len(test_df) else description
                fig, ax = lib.data.plot_metrics_radar(metrics, plot_title)
                # fig.set_size_inches(8, 4)
                fig.savefig(os.path.join(radar_plot_save_dirpath, f'{lib.utils.sanitize_filename(description).lower()}_radar_plot.pdf'), bbox_inches='tight')
                plt.close(fig)
                del fig
            del metrics
        del tasks
        gc.collect()

    pbar.close()

    del pbar
    return test_df, test_df_log

async def test_all_tasks(
        agent: lib.agent.ReActAgent, 
        df_test_single: pd.DataFrame,
        df_test_multi: pd.DataFrame,
        prompt_single: str,
        prompt_multi: str,
        description: str,
        data_source: str,
        experiment_name:str,
        plot_title:str = 'Results',
        show_plots: bool = False,
        concurrency_limit: Optional[int] = 32,
        task_batch_size: Optional[int] = 500,
    ):
    """
    Run all tasks using the provided agent.
    """

    if not isinstance(agent, lib.agent.ReActAgent) or not isinstance(agent, lib.agent.ReActAgent):
        raise ValueError("Both agent and agent_multi must be instances of lib.agent.ReActAgent")
    if not isinstance(description, str):
        raise ValueError("Description must be given")
    if not isinstance(data_source, str):
        raise ValueError("Data source must be given")
    if not isinstance(experiment_name, str):
        raise ValueError("Experiment must be given a name")
    
    config = ExperimentConfig.create(
        name=experiment_name,
        description=description,
        plot_title=plot_title,
        source=data_source,
        date_time=datetime.now(),
        agent=agent,
        n_samples_single=len(df_test_single),
        n_samples_multi=len(df_test_multi),
    )

    results_dirpath = os.path.join('results', config.generate_dirname())
    config.save_to_file(dirpath='results')

    print(f'Experiment: {experiment_name}, description: {description}')
    print(f"Config/results will be saved to '{results_dirpath}'")
    print(f"Running experiment with {len(df_test_single)} single tool samples and {len(df_test_multi)} multi tool samples.")
    print(f"Model: {agent.model_name}, Temperature: {agent.model_tempature}, Max Turns: {agent.max_turns}, Data source: {data_source}")
    print(f"Running concurrently with a limit of {concurrency_limit} tasks.")
    print()

    # Run the agent on each task and save the results
    similar, similar_log = await run_experiment(agent, df_test_single.copy(), prompt=prompt_single, description="Similarity", radar_plot_save_dirpath=results_dirpath, concurrency_limit=concurrency_limit, task_batch_size=task_batch_size)
    similar.to_pickle(os.path.join(results_dirpath, SIMILARRESULTFILENAME))
    similar_log.to_pickle(os.path.join(results_dirpath, SIMILARLOGFILENAME))
    
    del similar
    del similar_log
    gc.collect()

    reliability, reliability_log = await run_experiment(agent, df_test_single.copy(), prompt=prompt_single, reliability_mode=True, description="Reliability", radar_plot_save_dirpath=results_dirpath, concurrency_limit=concurrency_limit, task_batch_size=task_batch_size)
    reliability.to_pickle(os.path.join(results_dirpath, RELIABILITYRESULTFILENAME))
    reliability_log.to_pickle(os.path.join(results_dirpath, RELIABILITYLOGFILENAME))

    del reliability
    del reliability_log
    gc.collect()

    multi, multi_log = await run_experiment(agent, df_test_multi.copy(), prompt=prompt_multi, description="Multi-tool", radar_plot_save_dirpath=results_dirpath, concurrency_limit=concurrency_limit, task_batch_size=task_batch_size)
    multi.to_pickle(os.path.join(results_dirpath, MULTITOOLRESULTFILENAME))
    multi_log.to_pickle(os.path.join(results_dirpath, MULTITOOLLOGFILENAME))

    del multi
    del multi_log
    gc.collect()

    similar, reliability, multi, combined, _ = load_experiment_results_from_path(results_dirpath)

    # Compute metrics for each task
    max_turns = config.agent_config.max_turns if config.agent_config is not None else 20
    similar_metrics = lib.data.compute_action_metrics(similar, filter_ground_truth=False, max_turns=max_turns)
    reliability_metrics = lib.data.compute_action_metrics(reliability, filter_ground_truth=True, max_turns=max_turns)
    multi_metrics = lib.data.compute_action_metrics(multi, filter_ground_truth=False, max_turns=max_turns)
    combined_metrics = lib.data.compute_action_metrics(combined, max_turns=max_turns)
    combined_metrics = lib.data.adjust_combined_metrics(combined_metrics, similar_metrics, reliability_metrics, multi_metrics)

    # Plot metrics for each task
    fig, axes = plt.subplots(1, 3, figsize=(12, 12), subplot_kw=dict(polar=True))  # Create a 1x3 grid of polar plots

    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
    metrics = [
        similar_metrics,
        multi_metrics,
        reliability_metrics,
        combined_metrics
    ]
    titles = ['General', 'Multitool', 'Reliability', 'Combined']
    colors = ['blue', 'red', 'green', 'purple']
    # Call make_graph() with each subplot
    for i, ax in enumerate(axes):
        _, _ = lib.data.plot_metrics_radar(metrics[i], titles[i], ax=ax, backgroundcolor=colors[0])  # Pass the current polar subplot

    # Plot title
    fig.suptitle(f'{plot_title}, {config.n_samples_single} single samples, {config.n_samples_multi} multi samples, {config.agent_config.llm_name}', fontsize=16, fontweight='bold')

    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(results_dirpath, 'radar_plots.pdf'))
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    return similar, reliability, multi, combined_metrics

def get_experiment_overview(
        source: str = None,
        llm_model: str = None,
        from_date: datetime = None,
        dirpath: str = 'results'
    ) -> pd.DataFrame:
    """
    Get an overview of all experiments in the results directory.
    """
    if not os.path.exists(dirpath):
        raise FileNotFoundError(f"Results directory '{dirpath}' does not exist.")
    
    date_time_column = 'date_time'

    experiment_overview: list[dict[str]] = []
    for experiment_dir in os.listdir(dirpath):
        full_dirpath = os.path.join(dirpath, experiment_dir)
        if not os.path.isdir(full_dirpath):
            continue

        config_filepath = os.path.join(full_dirpath, CONFIGFILENAME)
        if not os.path.exists(config_filepath):
            continue

        try:
            config = ExperimentConfig.from_filepath(config_filepath)
            config_dict = config.model_dump(mode='python')

            agent_config_dict = config.agent_config.model_dump(mode='python') if config.agent_config is not None else None
            retriever_config_dict = config.tool_retriever_config.model_dump(mode='python') if config.tool_retriever_config is not None else None
            if agent_config_dict is not None:
                config_dict.update({f'ac_{k}': v for k, v in agent_config_dict.items()})
                config_dict.pop('agent_config', None)  # Remove the original key to avoid duplication
            
            if retriever_config_dict is not None:
                config_dict.update({f'trc_{k}': v for k, v in retriever_config_dict.items()})
                config_dict.pop('tool_retriever_config', None)  # Remove the original key to avoid duplication

            config_dict[DFDIRPATHCOLUMN] = full_dirpath
            config_dict[DFCONFIGCOLUMN] = config.model_dump(mode='python')
            experiment_overview.append(config_dict)
        except Exception as e:
            print(f"Error loading config from {config_filepath}: {e}")

    df = pd.DataFrame(experiment_overview)

    if date_time_column in df.columns:
        # Convert date_time column to datetime objects
        df[date_time_column] = pd.to_datetime(df[date_time_column], errors='coerce')
        # Sort by date_time column
        df = df.sort_values(by=date_time_column, ascending=False)
        df = df.reset_index(drop=True)

    if source is not None:
        df = df[df['source'] == source]

    if llm_model is not None:
        df = df[df['llm_model'] == llm_model]

    if from_date is not None:
        df = df[df[date_time_column] >= from_date]

    df.reset_index(drop=True, inplace=True)
    return df

def inspect_experiments(
    df_experiment_overview: pd.DataFrame,
    row_indices: list[int] | int,
    plot_title: str = None,
    show_legend: bool = True,
    results_dirpath: str = None,
):
    row_indices = [row_indices] if isinstance(row_indices, int) else row_indices
    if not isinstance(row_indices, list):
        raise ValueError("row_indices must be a list of integers or a single integer.")

    row_indices = np.unique(row_indices).tolist()
    for idx in row_indices:
        if not isinstance(idx, int):
            raise ValueError("row_indices must be a list of integers or a single integer.")
        if idx < 0 or idx >= len(df_experiment_overview):
            raise IndexError(f"Index {idx} is out of bounds for the DataFrame with {len(df_experiment_overview)} rows.")

    print(f"Inspecting experiments at indices: {row_indices}")
    
    config_list: list[ExperimentConfig] = []
    similar_metrics_list: list[dict[str]] = []
    reliability_metrics_list: list[dict[str]] = []
    multi_metrics_list: list[dict[str]] = []
    combined_metrics_list: list[dict[str]] = []
    legend_labels = []
    model_descriptions = []

    for idx in row_indices:
        # Inspect the experiment at the given index
        df_similar, df_reliability, df_multi, df_combined, config = load_experiment_results(df_experiment_overview, idx)
        max_turns = config.agent_config.max_turns if config.agent_config is not None else 20
        experiment = df_experiment_overview.iloc[idx]
        
        config_list.append(config)
        legend_label = f"model {idx}"

        if 'description' in experiment:
            model_descriptions.append(experiment.description)
            legend_label += f" - {experiment.description}"
        else:
            model_descriptions.append("")

        legend_labels.append(legend_label)

        similar_metrics_temp: dict[str] = None
        reliability_metrics_temp: dict[str] = None
        multi_metrics_temp: dict[str] = None

        if df_similar is not None and not df_similar.empty:
            similar_metrics_temp = lib.data.compute_action_metrics(df_similar, filter_ground_truth=False, max_turns=max_turns)
            similar_metrics_list.append(similar_metrics_temp)
        else:
            similar_metrics_list.append({})

        if df_reliability is not None and not df_reliability.empty:
            reliability_metrics_temp = lib.data.compute_action_metrics(df_reliability, filter_ground_truth=True, max_turns=max_turns)
            reliability_metrics_list.append(reliability_metrics_temp)
        else:
            reliability_metrics_list.append({})

        if df_multi is not None and not df_multi.empty:
            multi_metrics_temp = lib.data.compute_action_metrics(df_multi, filter_ground_truth=False, max_turns=max_turns)
            multi_metrics_list.append(multi_metrics_temp)
        else:
            multi_metrics_list.append({})

        if df_combined is not None and not df_combined.empty:
            combined_metrics_temp = lib.data.compute_action_metrics(df_combined, max_turns=max_turns)
            combined_metrics_list.append(lib.data.adjust_combined_metrics(combined_metrics_temp, similar_metrics_temp, reliability_metrics_temp, multi_metrics_temp))
        else:
            combined_metrics_list.append({})

    # Plot metrics for each task
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw=dict(polar=True))  # Create a 1x3 grid of polar plots
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
    titles = ['General', 'Multitool', 'Reliability', 'Combined']
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']  # Add more colors if needed
    handles = []

    for i, (config, legend_label, m_similar, m_multi, m_reliability, m_combined) in enumerate(zip(
        config_list, legend_labels, similar_metrics_list, multi_metrics_list, reliability_metrics_list, combined_metrics_list
    )):
        metrics = [m_similar, m_multi, m_reliability, m_combined]
        if not all(metrics):
            print(f"Skipping empty metrics for index {i}.")
            continue
        
        color = colors[i % len(colors)]
        handle = mlines.Line2D([], [], color=color, label=legend_label, marker='.', ls='')
        handles.append(handle)

        for i, ax in enumerate(axes):
            _, _ = lib.data.plot_metrics_radar(metrics[i], titles[i], ax=ax, backgroundcolor=color)  # Pass the current polar subplot
        
    # Plot title
    # plot_title = f'Experiment Overview: {" vs. ".join([config.name for config in config_list])}' if plot_title is None else plot_title
    if plot_title is not None:
        fig.suptitle(f'{plot_title}', fontsize=16, fontweight='bold')

    if show_legend:
        fig.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.06),
            # fontsize='small',
            fontsize='medium',
            # title='Models',
            # title_fontsize='medium',
            title_fontsize='large',
            ncol=len(legend_labels)
        )

    plt.tight_layout()  # Adjust layout
    if results_dirpath is not None:
        if not os.path.exists(results_dirpath):
            os.makedirs(results_dirpath)
        plt.savefig(os.path.join(results_dirpath, 'radar_plots.pdf'))
    plt.show()

    for i, (config, legend_label, description, m_similar, m_reliability, m_multi, m_combined) in enumerate(zip(
        config_list, legend_labels, model_descriptions, similar_metrics_list, reliability_metrics_list, multi_metrics_list, combined_metrics_list
    )):
        print(f'\n{legend_label}')
        # print(f'Similar Metrics:    \t{m_similar}')
        # print(f'Reliability Metrics:\t{m_reliability}')
        # print(f'Multi Metrics:      \t{m_multi}')
        # print(f'Combined Metrics:   \t{m_combined}')

        m_similar['metrics_type'] = 'similar'
        m_reliability['metrics_type'] = 'reliability'
        m_multi['metrics_type'] = 'multi'
        m_combined['metrics_type'] = 'combined'
        df_metrics = pd.DataFrame([m_similar, m_reliability, m_multi, m_combined])
        df_metrics = df_metrics.set_index('metrics_type').T
        print(df_metrics)

    df = df_experiment_overview.iloc[row_indices].copy()
    df = df.drop(columns=[DFCONFIGCOLUMN, DFDIRPATHCOLUMN], errors='ignore')
    df['label'] = legend_labels
    # switch row and column
    df = df.set_index('label').T
    # print(df)
    return df

def inspect_dataset_experiments(
        datasets: list[str],
        from_date: datetime = None,
        save_dirpath: str = None
    ):

    metrics_list = ['General', 'Multitool', 'Reliability']
    dataframes: list[pd.DataFrame] = []
    model_types: set = set()
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']  # Add more colors if needed

    for dataset in datasets:
        df = lib.experiment.get_experiment_overview(source=dataset, from_date=from_date)

        if df is None or df.empty:
            continue

        dataframes.append(df)
        model_types.update(df['description'].unique())

    if len(model_types) > len(colors):
        raise ValueError("Not enough colors for the number of model types. Please add more colors.")

    # Map model types to colors
    model_type_colors = dict(zip(sorted(model_types), colors))

    n_rows = len(dataframes)
    n_columns = len(metrics_list)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(6 * n_columns, 5 * n_rows), subplot_kw=dict(polar=True))
    axes: plt.Axes = axes  # Type hint for better IDE support
    handles: dict[str, mlines.Line2D] = {}


    left_text = None
    for i, (dataset_name, df_experiment) in enumerate(zip(datasets, dataframes)):
        axes_left: plt.Axes = axes[i, 0]
        left_text = axes_left.text(
            -0.15,
            0.6,
            dataset_name,
            rotation=90,
            size=18,
            fontweight='bold',
            verticalalignment='center',
            horizontalalignment='right',
            transform=axes_left.transAxes
        )

        for row_index in range(len(df_experiment)):
            model_type = df_experiment.iloc[row_index]['description']
            dirpath = df_experiment.iloc[row_index][DFDIRPATHCOLUMN]
            df_similar, df_reliability, df_multi, _, config = load_experiment_results_from_path(dirpath)

            max_turns = config.agent_config.max_turns
            metrics_similar = lib.data.compute_action_metrics(df_similar, filter_ground_truth=False, max_turns=max_turns)
            metrics_reliability = lib.data.compute_action_metrics(df_reliability, filter_ground_truth=True, max_turns=max_turns)
            metrics_multi = lib.data.compute_action_metrics(df_multi, filter_ground_truth=False, max_turns=max_turns)

            for j, (metric, metric_label) in enumerate(zip(
                [metrics_similar, metrics_multi, metrics_reliability],
                metrics_list
            )):
                # ax: plt.Axes = axes[i, j]
                color = model_type_colors[model_type]
                if model_type not in handles:
                    handle = mlines.Line2D([], [], color=color, label=model_type, marker='.', ls='')
                    handles[model_type] = handle

                label = metric_label if i == 0 else None  # Only show metric labels on the first row
                _, _ = lib.data.plot_metrics_radar(metric, label, ax=axes[i, j], backgroundcolor=color)

    lgd = fig.legend(
        handles=handles.values(),
        loc='lower center', bbox_to_anchor=(0.5, -0.04),
        # loc='upper center', bbox_to_anchor=(0.5,-0.1),
        fontsize='x-large',
        ncol=len(handles),
        bbox_transform=fig.transFigure
    )

    plt.tight_layout()  # Adjust layout
    if save_dirpath is not None:
        if not os.path.exists(save_dirpath):
            os.makedirs(save_dirpath)
        plt.savefig(os.path.join(save_dirpath, 'dataset_radar_comparison_plot.pdf'), bbox_extra_artists=(lgd, left_text), bbox_inches='tight')
    plt.show()


def load_experiment_results(
        df_experiment_overview: pd.DataFrame,
        row_index: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ExperimentConfig]:
    """
    Load the results for a specific row index from the experiment overview DataFrame.
    """
    df_experiment = df_experiment_overview.iloc[row_index]
    dirpath = df_experiment[DFDIRPATHCOLUMN]
    return load_experiment_results_from_path(dirpath)

def load_experiment_results_from_path(
        dirpath: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ExperimentConfig]:
    """
    Load the experiment results for a specific configuration.
    """

    df_similar: pd.DataFrame = None
    df_reliability: pd.DataFrame = None
    df_multi: pd.DataFrame = None
    df_combined: pd.DataFrame = None
    config: ExperimentConfig = None
    
    config_filepath = os.path.join(dirpath, CONFIGFILENAME)

    similar_filepath = os.path.join(dirpath, SIMILARRESULTFILENAME)
    reliability_filepath = os.path.join(dirpath, RELIABILITYRESULTFILENAME)
    multi_filepath = os.path.join(dirpath, MULTITOOLRESULTFILENAME)

    similar_log_filepath = os.path.join(dirpath, SIMILARLOGFILENAME)
    reliability_log_filepath = os.path.join(dirpath, RELIABILITYLOGFILENAME)
    multi_log_filepath = os.path.join(dirpath, MULTITOOLLOGFILENAME)

    if os.path.exists(config_filepath):
        config = ExperimentConfig.from_filepath(config_filepath)

    combined = []
    if os.path.exists(similar_filepath):
        df_similar = pd.read_pickle(similar_filepath)
        combined.append(df_similar)
        if os.path.exists(similar_log_filepath):
            df_similar_log: pd.DataFrame = pd.read_pickle(similar_log_filepath)
            df_similar_log = df_similar_log.drop(columns=['query'], errors='ignore')
            df_similar = pd.concat([df_similar, df_similar_log], axis=1)

    if os.path.exists(reliability_filepath):
        df_reliability = pd.read_pickle(reliability_filepath)
        # combined.append(df_reliability)
        if os.path.exists(reliability_log_filepath):
            df_reliability_log: pd.DataFrame = pd.read_pickle(reliability_log_filepath)
            df_reliability_log = df_reliability_log.drop(columns=['query'], errors='ignore')
            df_reliability = pd.concat([df_reliability, df_reliability_log], axis=1)

    if os.path.exists(multi_filepath):
        df_multi = pd.read_pickle(multi_filepath)
        combined.append(df_multi)
        if os.path.exists(multi_log_filepath):
            df_multi_log: pd.DataFrame = pd.read_pickle(multi_log_filepath)
            df_multi_log = df_multi_log.drop(columns=['query'], errors='ignore')
            df_multi = pd.concat([df_multi, df_multi_log], axis=1)

    if combined:
        df_combined = pd.concat(combined, ignore_index=True)

    return df_similar, df_reliability, df_multi, df_combined, config

def show_model_agreement_plot(
        df_experiment_overview: pd.DataFrame,
        experiment_index_A: int,
        experiment_index_B: int,
    ):


    df_similar_A, df_reliability_A, df_multi_A, _, config_A = load_experiment_results(
        df_experiment_overview,
        experiment_index_A
    )

    df_similar_B, df_reliability_B, df_multi_B, _, config_B = load_experiment_results(
        df_experiment_overview,
        experiment_index_B
    )

    if config_A.source != config_B.source:
        raise ValueError("Experiments must be from the same source to compare model agreement.")

    COLUMN_PREDS_NAME = 'action_res_tools_used'
    COLUMN_PREDS_Y = 'tool'

    legend_A = f'Model {experiment_index_A}'
    legend_B = f'Model {experiment_index_B}'

    # fig, axes = plt.subplots(1, 3, figsize=(12, 8), subplot_kw=dict(polar=True))  # Create a 1x3 grid of polar plots
    # axes: plt.Axes = axes.flatten()

    for i, (df_A, df_B, reliability_mode, title) in enumerate(zip(
        [df_similar_A, df_reliability_A, df_multi_A],
        [df_similar_B, df_reliability_B, df_multi_B],
        [False, True, False],
        ['General', 'Reliability', 'Multitool']
    )):
        if df_A is None or df_B is None or df_A.empty or df_B.empty:
            print(f"Skipping empty DataFrame for {title}.")
            continue

        if len(df_A) != len(df_B):
            raise ValueError(f"DataFrames for {title} must have the same length to compare model agreement.")

        Pred_A = df_A[COLUMN_PREDS_NAME].to_list()
        Pred_B = df_B[COLUMN_PREDS_NAME].to_list()
        y_true = df_A[COLUMN_PREDS_Y].to_list()
        # Y_B = df_B[COLUMN_PREDS_Y].to_list()

        if reliability_mode == True:
            correct_A = np.array([pred == [] for pred in Pred_A])
            correct_B = np.array([pred == [] for pred in Pred_B])
        else:
            correct_A = np.array([pred == true for pred, true in zip(Pred_A, y_true)])
            correct_B = np.array([pred == true for pred, true in zip(Pred_B, y_true)])

        both_correct = np.sum(correct_A & correct_B)
        a_only = np.sum(correct_A & ~correct_B)
        b_only = np.sum(~correct_A & correct_B)
        both_wrong = np.sum(~correct_A & ~correct_B)

        matrix = np.array([[both_correct, a_only], [b_only, both_wrong]])

        fig, ax = plt.subplots()
        # ax: plt.Axes = axes[i]

        # 2×2 counts
        im = ax.imshow(matrix, cmap='Blues')
        for (i, j), v in np.ndenumerate(matrix):
            ax.text(j, i, str(v), ha='center', va='center')

        ax.set_xticks([0,1]); ax.set_xticklabels([f'{legend_B}: Correct', f'{legend_B}: Wrong'])
        ax.set_yticks([0,1]); ax.set_yticklabels([f'{legend_A}: Correct', f'{legend_A}: Wrong'])
        ax.set_title('Model Agreement (Correct/Wrong)')
        ax.set_title(title)

        plt.colorbar(im, ax=ax)
        # plt.suptitle(f'Model Agreement: {config_A.name} vs {config_B.name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

def load_experiment_setup(
        setupType: ExperimentSetupType,
        client_type: lib.agent.ClientType,
        model_name: str,
        source: str,
        num_samples: int = None,
        random_state: int = 0,
        prompt_dirpath: str = "prompts",
        use_tool_description_augmentation: bool = True,
        verbose: bool = True
    ):

    def print_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if setupType.value not in [e.value for e in ExperimentSetupType]:
        raise ValueError(f"Invalid setupType '{setupType}'. Must be one of {[e.value for e in ExperimentSetupType]}")
    if client_type.value not in [e.value for e in lib.agent.ClientType]:
        raise ValueError(f"Invalid client_type '{client_type}'. Must be one of {[e.value for e in lib.agent.ClientType]}")

    # Data
    dataset_main = lib.data.Datasets.Main()
    df_test = dataset_main.queries
    df_tool_embeddings = dataset_main.tool_embeddings
    df_tool_descriptions = dataset_main.tool_descriptions

    if source is not None:
        df_test = df_test[df_test['source'] == source].reset_index(drop=True)
        df_tool_descriptions = df_tool_descriptions[df_tool_descriptions['source'] == source].reset_index(drop=True)
        df_tool_embeddings = df_tool_embeddings[df_tool_embeddings['tool'].isin(df_tool_descriptions['tool'])].reset_index(drop=True)

    df_tool_co_occurence = lib.data.ToolCoOccurrence.get_co_occurrence_dataframe(
        df=df_test,
        tool_col='tool',
        use_only_multi_tool_use_cases=True,
    )

    df_test_single = df_test[df_test['tool'].apply(lambda x: isinstance(x, list) and len(x) == 1)].reset_index(drop=True)
    if num_samples is not None and len(df_test_single) > num_samples:
        df_test_single = df_test_single.sample(n=num_samples, random_state=random_state).reset_index(drop=True)

    df_test_multi = df_test[df_test['tool'].apply(lambda x: isinstance(x, list) and len(x) > 1)].reset_index(drop=True)
    if num_samples is not None and len(df_test_multi) > num_samples:
        df_test_multi = df_test_multi.sample(n=num_samples, random_state=random_state).reset_index(drop=True)

    df_tool_descriptions_counter_example = None
    if use_tool_description_augmentation:
        df_tool_descriptions_counter_example = generateAugmented(df_tool_embeddings, df_tool_descriptions, source)

    df_descriptions: pd.DataFrame = (df_tool_descriptions_counter_example if df_tool_descriptions_counter_example is not None else df_tool_descriptions)

    print_verbose(f"Data source: {source}, {len(df_test_single)} single tool samples, {len(df_test_multi)} multi tool samples.")

    # Prompts
    with open(os.path.join(prompt_dirpath, "react_prompt_single.md"), "r") as file:
        react_prompt_single = file.read()

    with open(os.path.join(prompt_dirpath, "react_prompt_single_queue.md"), "r") as file:
        react_prompt_single_queue = file.read()

    with open(os.path.join(prompt_dirpath, "react_prompt_multi.md"), "r") as file:
        react_prompt_multi = file.read()

    with open(os.path.join(prompt_dirpath, "react_prompt_multi_queue.md"), "r") as file:
        react_prompt_multi_queue = file.read()

    with open(os.path.join(prompt_dirpath, "reflection_prompt.md"), "r") as file:
        reflection_prompt = file.read()

    with open(os.path.join(prompt_dirpath, "react_single_step_prompt_single.md"), "r") as file:
        react_single_step_prompt_single = file.read()

    with open(os.path.join(prompt_dirpath, "react_single_step_prompt_multi.md"), "r") as file:
        react_single_step_prompt_multi = file.read()

    with open(os.path.join(prompt_dirpath, "retriever_prompt.md"), "r") as file:
        retriever_prompt = file.read()

    # Configuration

    ## Standard Configuration
    MAX_TURNS = 15
    TEMPERATURE = 0
    TOP_K_TOOLS = 10 if setupType.value in [ExperimentSetupType.REACT_SINGLE_STEP.value] else 5
    SIM_THRESHOLD = 0.8
    PROVIDE_FEEDBACK = True
    PERSIST_TILL_TRUE_TOOL_CHOSEN = True

    ## Co-occurrence Configuration
    TOOL_RETRIEVER_INCLUDE_COMMENLY_USED = setupType.value in [ExperimentSetupType.REACT_WITH_CO_OCCURRENCE.value, ExperimentSetupType.QUOTA.value, ExperimentSetupType.QUOTA_WITH_REFLECTION.value]

    ## Action Queue Configuration
    ENABLE_ACTION_QUEUE = setupType.value in [ExperimentSetupType.QUOTA.value, ExperimentSetupType.QUOTA_WITH_REFLECTION.value]
    FORCE_ACTION_QUEUE = True
    RUN_QUEUE_AUTOMATICALLY = True



    # Instances
    tool_retriever = lib.tools.ToolRetriever(
        df_tool_embeddings, 
        df_descriptions,
        df_tool_co_occurrences=df_tool_co_occurence if TOOL_RETRIEVER_INCLUDE_COMMENLY_USED else None,
        include_commonly_used=TOOL_RETRIEVER_INCLUDE_COMMENLY_USED,
        top_k=TOP_K_TOOLS,
        similarity_threshold=SIM_THRESHOLD,
        provide_feedback=PROVIDE_FEEDBACK,
    )

    action_queue = None
    if ENABLE_ACTION_QUEUE and setupType.value in [ExperimentSetupType.QUOTA_WITH_REFLECTION.value]:
        print_verbose("Initializing reflective action queue...")
        action_queue = lib.tools.ToolActionQueueReflective(
            tool_retriever=tool_retriever,
            system_prompt=reflection_prompt,
            model_name=model_name,
            client_type=client_type,
            model_tempature=TEMPERATURE
        )
    elif ENABLE_ACTION_QUEUE:
        print_verbose("Initializing action queue...")
        action_queue = lib.tools.ToolActionQueue(tool_retriever)

    react_agent: lib.agent.ReActAgent | lib.agent.SingleStepReActAgent = None

    if setupType.value in [ExperimentSetupType.REACT_SINGLE_STEP.value]:
        print_verbose("Initializing single-step ReAct agent...")
        react_agent = lib.agent.SingleStepReActAgent(
            model_name=model_name,
            client_type=client_type,
            tool_retriever=tool_retriever,
            model_tempature=TEMPERATURE,
            retrieval_enabled=True,
            retriever_prompt=retriever_prompt
        ) 
    else:
        print_verbose("Initializing ReAct agent...")
        react_agent = lib.agent.ReActAgent(
            model_name=model_name,
            client_type=client_type,
            tool_retriever=tool_retriever,
            action_queue=action_queue,
            force_action_queue=FORCE_ACTION_QUEUE,
            model_tempature=TEMPERATURE,
            run_queue_automatically=RUN_QUEUE_AUTOMATICALLY,
            max_turns=MAX_TURNS,
            persist_till_true_tool_chosen=PERSIST_TILL_TRUE_TOOL_CHOSEN,
        )

    experiment_name = "Quota" if setupType.value in [ExperimentSetupType.QUOTA.value, ExperimentSetupType.QUOTA_WITH_REFLECTION.value] else "ReAct"

    description = experiment_name
    if setupType.value in [ExperimentSetupType.QUOTA_WITH_REFLECTION.value]:
        description += " with Reflection"
    elif setupType.value in [ExperimentSetupType.REACT_WITH_CO_OCCURRENCE.value]:
        description += " with Co-occurrence"
    elif setupType.value in [ExperimentSetupType.REACT_SINGLE_STEP.value]:
        description += " (one step)"
    # elif setupType.value in [ExperimentSetupType.QUOTA.value]:
    #     description += " (base)"
    # elif setupType.value in [ExperimentSetupType.REACT.value]:
    #     description += " (base)"

    plot_title = f"{experiment_name} experiment results"

    prompt_single = ""
    prompt_multi = ""

    if setupType.value in [ExperimentSetupType.REACT_SINGLE_STEP.value]:
        prompt_single = react_single_step_prompt_single
        prompt_multi = react_single_step_prompt_multi
    elif ENABLE_ACTION_QUEUE:
        prompt_single = react_prompt_single_queue
        prompt_multi = react_prompt_multi_queue
    else:
        prompt_single = react_prompt_single
        prompt_multi = react_prompt_multi

    print_verbose(f"Experiment setup: {experiment_name} ({description})\n\tModel: {model_name}\n\tTemperature: {TEMPERATURE}\n\tMax Turns: {MAX_TURNS}\n\tTop-K Tools: {TOP_K_TOOLS}\n\tSimilarity Threshold: {SIM_THRESHOLD}\n\tAction Queue: {ENABLE_ACTION_QUEUE}\n\tCo-occurrence: {TOOL_RETRIEVER_INCLUDE_COMMENLY_USED}")
    return (
        react_agent, 
        df_test_single, 
        df_test_multi,
        df_descriptions,
        df_tool_embeddings,
        prompt_single,
        prompt_multi,
        experiment_name,
        plot_title,
        description
    )

def generateAugmented(
        df_tool_embeddings: pd.DataFrame, 
        df_tool_descriptions: pd.DataFrame, 
        source:str="main"
    ):
    
    if source is None or not isinstance(source, str):
        source = "main"

    dirpath = './data_cache'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    filepath = os.path.join(dirpath, f'tool_descriptions_suggestive_{source}.pkl')

    if os.path.exists(filepath):
        df_tool_descriptions_suggestive:pd.DataFrame = pd.read_pickle(filepath)
        return df_tool_descriptions_suggestive

    # Load tool embeddings
    df_embeddings_similar = df_tool_embeddings.copy()

    tools = df_embeddings_similar['tool'].values
    embeddings = np.vstack(df_embeddings_similar['embedding'].values)

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Set threshold
    # threshold = 0.9
    threshold = 0.8

    # Find similar tools above threshold
    closest_tool = []
    for i, tool in enumerate(tools):
        sims = similarity_matrix[i]
        # Exclude self-match
        similar_indices = np.where((sims >= threshold) & (np.arange(len(sims)) != i))[0]
        # Sort by similarity (descending)
        sorted_similar = sorted(
            [(tools[j], sims[j]) for j in similar_indices],
            key=lambda x: -x[1]
        )
        closest_tool.append(sorted_similar[0][0] if sorted_similar else None)

    # Save to DataFrame
    df_embeddings_similar['closest_tool'] = closest_tool

    # Step 1: Merge to get 'description' for 'tool'
    df_embeddings_similar = df_embeddings_similar.merge(
        df_tool_descriptions[['tool', 'description']],
        on='tool',
        how='left'
    )

    # Step 2: Merge to get 'description' for 'closest_tool', renamed as 'closest_description'
    df_embeddings_similar = df_embeddings_similar.merge(
        df_tool_descriptions[['tool', 'description']].rename(columns={
            'tool': 'closest_tool',
            'description': 'closest_description'
        }),
        on='closest_tool',
        how='left'
    )

    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()
    # model = 'gpt-4'
    model = 'gpt-4o-mini'

    MAX_WORKERS = 16  # Tune this to avoid rate limits

    def invoke_client(messages: list) -> str:
        max_tries = 6
        try_count = 0

        while try_count < max_tries:
            try_count += 1
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return completion.choices[0].message.content.strip()
            except openai.RateLimitError as e:
                print(f"Rate limit exceeded: {repr(e)}. Retrying...")
                time.sleep(2 ** try_count)
            except openai.APIError as e:
                print(f"API error: {repr(e)}. Retrying...")
                time.sleep(2 ** try_count)
            except openai.APIConnectionError as e:
                print(f"Connection error: {repr(e)}. Retrying...")
                time.sleep(2 ** try_count)
            except Exception as e:
                print(f"Error: {repr(e)}. Retrying...")
                time.sleep(2 ** try_count)
        return None

    def generate_exclusion_note(description, closest_description):
        if pd.isna(closest_description) or not str(closest_description).strip():
            return None
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that writes clear and concise contrastive descriptions "
                    "for tools that have mutually exclusive functions."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Main tool description:\n{description}\n\n"
                    f"Most similar tool description:\n{closest_description}\n\n"
                    f"Assume the tools do not overlap in functionality. "
                    f"Write one sentence that starts with: \"This tool cannot be used for...\", "
                    f"and explain what the main tool does NOT do, based on what the similar tool does."
                )
            }
        ]

        result = invoke_client(messages)
        if result:
            return result.strip().strip('"').strip("'")
        return None

    # Parallel execution
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(generate_exclusion_note, row['description'], row['closest_description']): idx
            for idx, row in df_embeddings_similar.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating exclusion notes"):
            idx = futures[future]
            try:
                results.append((idx, future.result()))
            except Exception as e:
                print(f"Row {idx} failed: {e}")
                results.append((idx, None))

    # Assign results back
    for idx, note in results:
        df_embeddings_similar.at[idx, 'exclusion_note'] = note

    # Generate exclusionary descriptions
    def make_exclusion_description(row):
        description = str(row['description']).strip()
        exclusion_note = row['exclusion_note']
        if pd.notnull(exclusion_note):
            if not re.search(r'[.!?]$', description):
                description += "."
            return f"{description} {exclusion_note}"
        return description

    df_embeddings_similar['contextualized_description'] = df_embeddings_similar.apply(make_exclusion_description, axis=1)

    # Generate exclusionary descriptions
    def append_similar_suggestion(row):
        description = str(row['contextualized_description']).strip()
        closest_tool = row['closest_tool']
        if pd.notnull(closest_tool):
            description += f" For that purpose, you can also use the tool '{closest_tool}' instead if it's available, but beware that you can **only use a tool if it is listed as one of the available tools.**"
        return description

    df_embeddings_similar['suggestive'] = df_embeddings_similar.apply(append_similar_suggestion, axis=1)

    df_tool_descriptions_suggestive = df_embeddings_similar[['tool', 'suggestive']].rename(
        columns={'suggestive': 'description'}
    )

    if not os.path.exists(filepath):
        df_tool_descriptions_suggestive.to_pickle(filepath)

    return df_tool_descriptions_suggestive