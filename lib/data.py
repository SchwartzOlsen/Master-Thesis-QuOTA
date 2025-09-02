from typing import Literal

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json 
import re

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import combinations

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

METRIC_N_KEY = "_n"
METRIC_AVG_LLM_CALLS_KEY = "_Avg. llm calls"
METRIC_AVG_TOKENS_KEY = "_Avg. tokens"
METRIC_AVG_AUT_KEY = "_Avg. AUT"
METRIC_AVG_OR_KEY = "_Avg. OR"
METRIC_SPECIFICITY_KEY = "Specificity"

def clean_tool_name(tool_name: str):
    re_invalid_characters = r"[^a-zA-Z0-9]"
    re_CamelCase_to_snake_case = r'(?<!^)(?=[A-Z])'
    name = re.sub(re_invalid_characters, '_', tool_name.strip(' '), 0, re.MULTILINE)
    return re.sub(re_CamelCase_to_snake_case, '_', name).lower()

def print_data_example(data: pd.DataFrame, i: int = 0):
    print(f'Example index: {i}:\n')
    row = data.iloc[i]

    for col in data.columns:
        clean_text = re.sub(r'\\n', '\n', str(row[col]))
        print(f'{col}:\n{clean_text}\n')

def get_available_tools(
    embeddings_df: pd.DataFrame,
    true_tool: str | list[str] | set[str] = None,
    include_true_tool: bool = True,
    true_tool_max_sim: float = None,
    true_tool_skip_n: int = 0,
) -> pd.DataFrame:
    """
    Returns a filtered subset of embeddings_df based on similarity to the true tool(s),
    excluding overly similar tools or true tools based on the configuration.

    Args:
        embeddings_df (pd.DataFrame): DataFrame with columns ['tool', 'embedding'].
        true_tool (str or list of str): The known correct tool(s).
        include_true_tool (bool): Whether to include true tool(s) in the result.
        true_tool_max_sim (float): Max similarity to true tool(s) allowed.
        true_tool_skip_n (int): Number of top most similar tools to the true tool(s) to exclude.

    Returns:
        pd.DataFrame: Filtered embeddings_df with valid tools.
    """

    embeddings_matrix = np.vstack(embeddings_df["embedding"].values)
    mask = np.ones(len(embeddings_df), dtype=bool)

    true_tools = set(true_tool) if isinstance(true_tool, (list, set, tuple)) else {true_tool}
    true_tools.discard(None)
    
    if true_tools:

        true_indices = []

        for tool_name in true_tools:
            tool_row = embeddings_df[embeddings_df["tool"] == tool_name]
            if tool_row.empty:
                continue
            
            true_embedding = np.array(tool_row["embedding"].values[0]).reshape(1, -1)
            sim_to_this_tool = cosine_similarity(embeddings_matrix, true_embedding).flatten()

            # Apply max_sim for this true tool
            if true_tool_max_sim is not None and true_tool_max_sim > 0:
                mask &= sim_to_this_tool <= true_tool_max_sim

            # Apply skip_n for this true tool
            if true_tool_skip_n is not None and true_tool_skip_n > 0:
                skip_indices = np.argsort(sim_to_this_tool)[::-1][:true_tool_skip_n]
                mask[skip_indices] = False

            # Collect true tool index for separate inclusion/exclusion pass
            true_indices.append(tool_row.index[0])

        # Override true tool inclusion/exclusion after all filtering
        for idx in true_indices:
            mask[idx] = include_true_tool

    return embeddings_df[mask].copy()

def find_most_similar_with_descriptions(
    embeddings_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    query_embedding,
    top_k=10,
    similarity_threshold: float = None,
    include_true_tool=True,
    true_tool_max_sim: float = None,
    true_tool=None,
    true_tool_skip_n: int = 0,
    distance_metric: Literal['cosine', 'euclidean', 'manhattan'] = 'cosine',
) -> tuple[pd.DataFrame, int]:
    """
    Retrieves the top-k most similar tools to the query embedding, using the filtered list from get_available_tools.

    Args:
        embeddings_df (pd.DataFrame): DataFrame with columns ['tool', 'embedding'].
        descriptions_df (pd.DataFrame): DataFrame with tool descriptions, with columns ['tool', 'description'].
        query_embedding (np.ndarray): Embedding of the query.
        top_k (int): Number of top similar tools to return.
        similarity_threshold (float, optional): Minimum similarity score to include a tool.
        include_true_tool (bool): Whether to include the true tool in the results.
        true_tool_max_sim (float, optional): Maximum similarity to the true tool allowed.
        true_tool (str or list of str, optional): The known correct tool(s).
        true_tool_skip_n (int): Number of top most similar tools to the true tool(s) to exclude.
        distance_metric (str): Distance metric to use ('cosine', 'euclidean', 'manhattan').

    Returns:
        dataframe (pd.DataFrame): Top-k tools with descriptions and similarity scores.
        n_excluded_tools (int): Number of excluded tools based on the top_k limit.
    """

    available_df = get_available_tools(
        embeddings_df=embeddings_df,
        true_tool=true_tool,
        include_true_tool=include_true_tool,
        true_tool_max_sim=true_tool_max_sim,
        true_tool_skip_n=true_tool_skip_n,
    )

    if available_df.empty:
        return pd.DataFrame(columns=["tool", "embedding", "similarity"]).merge(descriptions_df, on="tool", how="left")

    embeddings_matrix = np.vstack(available_df["embedding"].values)
    
    if distance_metric == 'euclidean':
        distances: np.ndarray = euclidean_distances([query_embedding], embeddings_matrix)[0]
        similarities = 1 / (1 + distances)
    elif distance_metric == 'cosine':
        similarities: np.ndarray = cosine_similarity([query_embedding], embeddings_matrix)[0]
    elif distance_metric == 'manhattan':
        distances: np.ndarray = manhattan_distances([query_embedding], embeddings_matrix)[0]
        similarities = 1 / (1 + distances)
    
    top_k_indices = np.argsort(similarities)[::-1]#[:top_k]

    if similarity_threshold is not None:
        top_k_indices = [i for i in top_k_indices if similarities[i] >= similarity_threshold]

    n_excluded_tools = 0
    if top_k is not None and top_k > 0 and len(top_k_indices) > top_k:
        n_excluded_tools = len(top_k_indices) - top_k
        top_k_indices = top_k_indices[:top_k]

    top_k_indices = np.array(top_k_indices)
    top_k_tools = available_df.iloc[top_k_indices].copy()
    top_k_tools["similarity"] = similarities[top_k_indices]

    return top_k_tools.merge(descriptions_df, on="tool"), n_excluded_tools

def exp_decay(x, k=1.0):
    return np.exp(-k * x)

def adjust_combined_metrics(combined: dict[str], similar: dict[str], reliability: dict[str], multi: dict[str]) -> dict[str]:
    if (combined is None or similar is None or reliability is None or multi is None):
        return combined

    metric_list = [similar, reliability, multi]
    metric_list = [metric for metric in metric_list if metric is not None]

    OR = 0
    AUT = 0
    TOKENS = 0
    LLM_CALLS = 0
    n_total = 0

    # Compute the adjusted metrics
    for metric in metric_list:
        n = metric.get(METRIC_N_KEY, 0)
        if not (n > 0):
            continue
        n_total += n
        if METRIC_AVG_OR_KEY in metric and metric[METRIC_AVG_OR_KEY] is not None:
            OR += metric[METRIC_AVG_OR_KEY] * n
        if METRIC_AVG_AUT_KEY in metric and metric[METRIC_AVG_AUT_KEY] is not None:
            AUT += metric[METRIC_AVG_AUT_KEY] * n
        if METRIC_AVG_TOKENS_KEY in metric and metric[METRIC_AVG_TOKENS_KEY] is not None:
            TOKENS += metric[METRIC_AVG_TOKENS_KEY] * n
        if METRIC_AVG_LLM_CALLS_KEY in metric and metric[METRIC_AVG_LLM_CALLS_KEY] is not None:
            LLM_CALLS += metric[METRIC_AVG_LLM_CALLS_KEY] * n

    if n_total > 0:
        combined[METRIC_AVG_OR_KEY] = OR / n_total
        combined[METRIC_AVG_AUT_KEY] = AUT / n_total
        combined[METRIC_AVG_TOKENS_KEY] = TOKENS / n_total
        combined[METRIC_AVG_LLM_CALLS_KEY] = LLM_CALLS / n_total

    if METRIC_SPECIFICITY_KEY in reliability:
        combined[METRIC_SPECIFICITY_KEY] = reliability[METRIC_SPECIFICITY_KEY]

    return combined

def compute_action_metrics(df: pd.DataFrame, filter_ground_truth: bool = True, max_turns=20):
    """
    Compute action metrics for tool selection evaluation.
    
    Args:
        df (pd.DataFrame): DataFrame containing the evaluation results.
        filter_ground_truth (bool): Whether to filter ground truth tools based on available tools. I.e. if the tool is not available, it is not considered in the evaluation.
    """
    
    TP, FP, FN, TN = 0, 0, 0, 0
    CR_count, OR_count, extra_tools_count, correct_number_of_tools, retrieved_all_count, retrieved_percentage_sum = 0, 0, 0, 0, 0, 0
    all_N = 0 # Replaces FP + TN in specificity since FP can mean "different positive"
    reliability = True

    n = len(df)
    for _, row in df.iterrows():
        if row['action_res'] is None or pd.isna(row['action_res']) or pd.isnull(row['action_res']):
            continue
        # Convert ground truth tool to a set (handles both single & multitool cases)
        available_tools = set(row['available_tools'])
        retrieved_tools = set(row['retrieved_tools'])
        tools_used = set(row['action_res_tools_used'])
        ground_truth = set(row['tool']) if isinstance(row['tool'], list) else {row['tool']}


        # Remove any tools from ground_truth that are not available
        if filter_ground_truth:
            ground_truth = ground_truth.intersection(available_tools)

        reliability = reliability and len(ground_truth) == 0

        ## === Binary Classification Metrics ===
        if tools_used and tools_used == ground_truth:
            TP += 1
        elif not tools_used and not ground_truth:
            TN += 1
        elif not tools_used and ground_truth:
            FN += 1
        else:
            FP += 1

        if not ground_truth:
            all_N += 1

        ## === Custom Metrics ===
        if ground_truth.issubset(tools_used):  # All required tools must be present
            CR_count += 1

        if tools_used - ground_truth:  # At least one unnecessary tool was selected
            OR_count += 1

        extra_tools_count += len(tools_used - ground_truth)  # Count extra tools

        # check if correct number of tools were used
        if len(tools_used) == len(ground_truth):
            correct_number_of_tools += 1

        # check if the correct tools are in the retrieved tools
        if ground_truth.issubset(retrieved_tools):  # All required tools must be present
            retrieved_all_count += 1
        retrieved_percentage = len(ground_truth & retrieved_tools) / len(ground_truth) if len(ground_truth) > 0 else 1
        retrieved_percentage_sum += retrieved_percentage

    # Compute binary classification metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)              if (TP + TN + FP + FN) != 0 else None
    precision = TP / (TP + FP)                              if not reliability and (TP + FP) != 0 else None
    recall = TP / (TP + FN)                                 if (TP + FN) != 0 else None
    f1 = 2 * (precision * recall) / (precision + recall)    if precision != None and recall != None and (precision + recall) != 0 else None
    specificity = TN / all_N                                if (all_N) != 0 else None
    tool_count_accuracy = correct_number_of_tools / n if n != 0 else None

    # Compute custom metrics
    CR = CR_count / n if n > 0 else None
    OR = OR_count / n if n > 0 else None
    AUT = extra_tools_count / n if n > 0 else None

    # Exponential decay of OR and AUT
    OR_INVERTED = exp_decay(OR) if OR is not None else None
    AUT_INVERTED = exp_decay(AUT) if AUT is not None else None

    # Retrieval metrics
    any_retrieved = df['retrieved_tools'].apply(lambda x: len(x) > 0 if x is not None else False).any() if n > 0 and 'retrieved_tools' in df.columns else False
    retrieval_recall = retrieved_all_count / n           if n > 0 and any_retrieved and not reliability else None
    retrieved_percentage = retrieved_percentage_sum / n   if n > 0 and not reliability and any_retrieved else None

    usage_total_tokens: pd.Series = df['usage_total_tokens'] if 'usage_total_tokens' in df.columns else pd.Series(dtype=float)
    usage_llm_calls: pd.Series = df['usage_llm_calls'] if 'usage_llm_calls' in df.columns else pd.Series(dtype=float)
    used_turns:pd.Series = df['turns'] if 'turns' in df.columns else pd.Series(dtype=float)
    if n > 0:
        # used_turns = used_turns.fillna(max_turns)
        used_turns = used_turns.dropna()
        usage_total_tokens = usage_total_tokens.dropna()
        usage_llm_calls = usage_llm_calls.dropna()

    avg_llm_calls = sum(usage_llm_calls) / len(usage_llm_calls) if len(usage_llm_calls) > 0 else None
    avg_total_tokens = sum(usage_total_tokens) / len(usage_total_tokens) if len(usage_total_tokens) > 0 else None

    avg_turns = sum(used_turns) / len(used_turns) if len(used_turns) > 0 and max_turns != 0 and 'turns' in df.columns else None
    # turns_left = max_turns - (sum(used_turns) / n / max_turns) if n > 0 and max_turns != 0 and 'turns' in df.columns else None
    turns_left = (max_turns - avg_turns) / max_turns if n > 0 and avg_turns is not None and max_turns != 0 and 'turns' in df.columns else None
    
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        # Show Accuracy if Specificity is None (Specificity is used for Reliability test, which is essentially the same as accuracy)
        "Accuracy": accuracy if specificity is None else None,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        METRIC_SPECIFICITY_KEY: specificity,
        # Show Tool Count Accuracy if Specificity is None (Specificity is used for Reliability test, which is essentially the same as accuracy)
        "TCA": tool_count_accuracy if specificity is None else None, # Tool Count Accuracy
        # Show Coverage rate if Specificity is None (Specificity is used for Reliability test, which is essentially the same as accuracy)
        "CR": CR if specificity is None else None, # Coverage Rate
        "OR": OR_INVERTED, # Inv. Over-selection Rate
        "AUT": AUT_INVERTED, # Inv. Avg. Unnecessary Tools
        "Retr. Rec.": retrieval_recall, # Retrieval Recall
        "Avg. Retr. %": retrieved_percentage, # Avg. Percentage of Tools Retrieved
        # f"Avg. turns Left / {max_turns}": turns_left,
        f"Avg. turns Left / max turns": turns_left,
        "_Avg. turns used": avg_turns, # Avg. Number of turns Used
        METRIC_AVG_OR_KEY: OR, # Over-selection Rate
        METRIC_AVG_AUT_KEY: AUT, # Avg. Unnecessary Tools
        METRIC_AVG_TOKENS_KEY: avg_total_tokens,
        METRIC_AVG_LLM_CALLS_KEY: avg_llm_calls,
        METRIC_N_KEY: n
    }

def plot_metrics_radar(metrics: dict[str, float], title: str = None, ax = None, backgroundcolor = 'b'):
    # Drop 'TP', 'FP', 'FN' and 'TN' keys
    metrics = {
        k: v for k, v in metrics.items()
        if v != None and k not in ['TP', 'FP', 'FN', 'TN',] and k.startswith('_') == False
    }

    labels = list([key.replace('/', '\n/') for key in metrics.keys()])
    values = list(metrics.values())
    color = 'black'
    N = len(labels)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]  # Close the radar chart
    angles += angles[:1]
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4), subplot_kw=dict(polar=True))
    
    # Remove spines
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")
    # Angle values going from 0 to 2*pi
    HANGLES = np.linspace(0, 2 * np.pi)
    PAD = 0.05
    
    # Used for the equivalent of horizontal lines in cartesian coordinates plots 
    # The last one is also used to add a fill which acts a background color.
    for i in [0, 0.25, 0.5, 0.75, 1]:
        ax.plot(HANGLES, [i] * np.ones(len(HANGLES)), ls=(0, (6, 6)), linewidth=1, color='grey')
        ax.text(-0.4, i + PAD, f"{i*100:.0f}%", size=14)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')

    
    ax.plot(angles, values, linewidth=1, linestyle='solid', color=backgroundcolor, alpha=0.75)
    ax.fill(angles, values, color=backgroundcolor, alpha=0.05)
    if title is not None:
        ax.set_title(title, size=18, color=color, fontweight='bold', pad=10, y=1.08)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"], color=color, size=10)  # Remove radial labels for clarity

    # Remove lines for radial axis (y)
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    XTICKS = ax.xaxis.get_major_ticks()
    for i in range(len(XTICKS)):
        a = angles[i]
        ha = "left" if (0 <= a < np.pi/2) or (3*np.pi/2 < a < 2*np.pi)  else "right"
        # Set XTIck horizontal alignment
        XTICKS[i].label1.set_horizontalalignment(ha)

    ax.grid(True)

    return fig, ax

class Datasets:

    class Main:

        class FileNames:
            # Pickle files is preferred for saving dataframes over csv files, as we can save the data types of the columns
            queries = 'queries.pkl'
            tool_descriptions = 'tool_descriptions.pkl'
            tool_embeddings = 'tool_embeddings.pkl'

        class ColumnTypes:
            queries = {'query': str, 'tool': object, 'source': str}
            tool_descriptions = {'tool': str, 'tool_name': str, 'description': str, 'source': str, 'metadata': object}
            tool_embeddings = {'tool': str, 'embedding': object}

        def __init__(self, load: bool = True):
            self.path: str = os.path.join(data_dir, 'main')

            if not load:
                self.queries, self.tool_descriptions, self.tool_embeddings = Datasets.Main.get_initial_dataframes()
                return

            self.queries = pd.read_pickle(os.path.join(self.path, Datasets.Main.FileNames.queries)).astype(Datasets.Main.ColumnTypes.queries)
            """
            Columns: query, tool, source
            """

            self.tool_descriptions = pd.read_pickle(os.path.join(self.path, Datasets.Main.FileNames.tool_descriptions)).astype(Datasets.Main.ColumnTypes.tool_descriptions)
            """
            Columns: tool, description, source, metadata
            """

            self.tool_embeddings = Datasets.Main.read_embeddings(os.path.join(self.path, Datasets.Main.FileNames.tool_embeddings)).astype(Datasets.Main.ColumnTypes.tool_embeddings)
            """
            Columns: tool, embedding
            """

        def save_data(self):
            if not os.path.exists(self.path):
                os.makedirs(self.path)

            self.queries.to_pickle(os.path.join(self.path, Datasets.Main.FileNames.queries))
            self.tool_descriptions.to_pickle(os.path.join(self.path, Datasets.Main.FileNames.tool_descriptions))
            self.tool_embeddings.to_pickle(os.path.join(self.path, Datasets.Main.FileNames.tool_embeddings))

        def save_data_as_tool_e_format(self, save_dir, filter_tool_embeddings: bool = True):
            tool_dir = os.path.join(save_dir, 'data/')
            if not os.path.exists(tool_dir):
                os.makedirs(tool_dir)

            unique_tools = self.queries.tool.explode().unique().tolist()
            unique_tools_single = self.queries.where(self.queries.tool.apply(len) <= 1).dropna().tool.explode().unique().tolist()
            unique_tools_multi = self.queries.where(self.queries.tool.apply(len) > 1).dropna().tool.explode().unique().tolist()

            df_single_tool: pd.DataFrame = (
                self.queries
                .where(self.queries.tool.apply(len) <= 1)
                .dropna()
                .rename(columns={'tool': 'Tool', 'query': 'Query'})
                .reset_index(drop=True)
            )
            df_single_tool.drop(columns=['source'], inplace=True)
            df_single_tool.Tool = df_single_tool.Tool.apply(lambda x: x[0] if len(x) == 1 else None)

            df_multi_tool: pd.DataFrame = (
                self.queries
                .where(self.queries.tool.apply(len) > 1)
                .dropna()
                .reset_index(drop=True)
            )
            df_multi_tool.drop(columns=['source'], inplace=True)

            # df_descriptions: pd.DataFrame = self.tool_descriptions[['tool', 'description']].copy()
            # df_descriptions: pd.DataFrame = self.tool_descriptions[self.tool_descriptions.tool.isin(unique_tools)][['tool', 'description']].copy()
            df_descriptions: pd.DataFrame = self.tool_descriptions[self.tool_descriptions.tool.isin(unique_tools_single)][['tool', 'description']].copy()
            # df_descriptions: pd.DataFrame = self.tool_descriptions[self.tool_descriptions.tool.isin(unique_tools_multi)][['tool', 'description']]
            series_descriptions = df_descriptions.set_index('tool').description


            df_descriptions_merged: pd.DataFrame = self.tool_descriptions[['tool', 'description']].copy()
            # df_descriptions_merged: pd.DataFrame = self.tool_descriptions[self.tool_descriptions.tool.isin(unique_tools_multi)][['tool', 'description']].copy()
            # df_descriptions_merged: pd.DataFrame = self.tool_descriptions[self.tool_descriptions.tool.isin(unique_tools_single)][['tool', 'description']]
            series_descriptions_merged = df_descriptions_merged.set_index('tool').description
            

            if filter_tool_embeddings:
                df_embeddings: pd.DataFrame = self.tool_embeddings[self.tool_embeddings.tool.isin(unique_tools)][['tool', 'embedding']]
                # df_embeddings: pd.DataFrame = self.tool_embeddings[self.tool_embeddings.tool.isin(unique_tools_single)][['tool', 'embedding']]
            else:
                df_embeddings: pd.DataFrame = self.tool_embeddings[['tool', 'embedding']].copy()
            list_embeddings: list[dict] = df_embeddings.to_dict(orient='records')
            
            df_single_tool.to_csv(
                os.path.join(tool_dir, Datasets.ToolE.FileNames.single_tool),
                index=False
                )    
            df_multi_tool.to_json(
                os.path.join(tool_dir, Datasets.ToolE.FileNames.multi_tool),
                orient='records',
                indent=2,
                )
            series_descriptions.to_json(
                os.path.join(save_dir, Datasets.ToolE.FileNames.descriptions),
                orient='index',
                indent=2,
                )
            series_descriptions_merged.to_json(
                os.path.join(save_dir, Datasets.ToolE.FileNames.merged_descriptions),
                orient='index',
                indent=2,
                )
            pd.to_pickle(
                list_embeddings, 
                os.path.join(save_dir, Datasets.ToolE.FileNames.embeddings)
                )

        @staticmethod
        def read_embeddings(path: str) -> pd.DataFrame:
            embeddings:list[dict] = pd.read_pickle(path)
            return pd.DataFrame(embeddings)

        @staticmethod
        def get_initial_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            df_queries = (
                pd.DataFrame(columns=['query', 'tool', 'source']).astype(Datasets.Main.ColumnTypes.queries)
            )
            
            df_tool_descriptions = (
                pd.DataFrame(columns=['tool', 'tool_name', 'description', 'source', 'metadata']).astype(Datasets.Main.ColumnTypes.tool_descriptions)
            )
            
            df_tool_embeddings = (
                pd.DataFrame(columns=['tool', 'embedding']).astype(Datasets.Main.ColumnTypes.tool_embeddings)
            )
            
            return df_queries, df_tool_descriptions, df_tool_embeddings

        def find_top_k_similar_tools(self, tool, k=10):
            tool_embedding = self.tool_embeddings[self.tool_embeddings.tool == tool].embedding.values[0]
            return find_most_similar_with_descriptions(self.tool_embeddings, self.tool_descriptions, tool_embedding, k)

    class ToolE:
        source_name: str = "ToolE"

        class FileNames:
            single_tool = 'all_clean_data.csv'
            multi_tool = 'multi_tool_query_golden.json'
            descriptions = 'plugin_des.json'
            """
            This file contains descriptions of tools used from the single tool query dataset.
            """

            openai_meta_data = 'plugin_info.json'
            merged_descriptions = 'big_tool_des.json'
            """
            This file contains descriptions of of the total tools found in milvus.
            """

            embeddings = 'tool_embedding.pkl'

        def __init__(self):
            self.path: str = os.path.join(data_dir, 'ToolE')
            
            self.single_tool = pd.read_csv(os.path.join(self.path, 'data', Datasets.ToolE.FileNames.single_tool))
            """
            Columns: Query, Tool
            """

            self.multi_tool = pd.read_json(os.path.join(self.path, 'data', Datasets.ToolE.FileNames.multi_tool))
            """
            Column: query, tool
            """

            self.descriptions = pd.read_json(os.path.join(self.path, Datasets.ToolE.FileNames.descriptions), orient='index')
            """
            Columns: tool, description
            """
            
            self.descriptions = self.descriptions.reset_index()
            self.descriptions.columns = ['tool', 'description']  

            self.openai_meta_data = pd.read_json(os.path.join(self.path, Datasets.ToolE.FileNames.openai_meta_data))
            """
            Columns: name_for_model, name_for_human, description_for_model, description_for_human
            """

            self.merged_descriptions = pd.read_json(os.path.join(self.path, Datasets.ToolE.FileNames.merged_descriptions), orient='index')
            """
            Columns: tool, description
            """
            self.merged_descriptions = self.merged_descriptions.reset_index()
            self.merged_descriptions.columns = ['tool', 'description']

            # self.embeddings = pd.read_pickle(self.path + 'tool_embedding.pkl)
            tool_embeddings:list[dict] = pd.read_pickle(os.path.join(self.path, Datasets.ToolE.FileNames.embeddings))
            self.embeddings = pd.DataFrame(tool_embeddings)
            """
            Columns: tool, embedding
            """

        def parse_to_main(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            source_clean = clean_tool_name(self.source_name)
            df_queries, df_tool_descriptions, df_tool_embeddings = Datasets.Main.get_initial_dataframes()
            
            # df_tool_descriptions_2 = df_tool_descriptions.copy()

            df_single_tool = self.single_tool.copy()
            df_multi_tool = self.multi_tool.copy()
            df_tool_descriptions = self.descriptions.copy()
            df_tool_embeddings = self.embeddings.copy()

            # df_tool_descriptions_2['tool'] = tool_e.openai_meta_data['name_for_model']
            # df_tool_descriptions_2['description'] = tool_e.openai_meta_data['description_for_human']

            # Check if any tools from 'df_tool_descriptions_2' are not in 'df_tool_descriptions'
            # If so, add them to 'df_tool_descriptions'
            # df_missing_tools = df_tool_descriptions_2[~df_tool_descriptions_2['tool'].isin(df_tool_descriptions['tool'])]
            # df_tool_descriptions = pd.concat([df_tool_descriptions, df_missing_tools], ignore_index=True)
            
            def parse_tool(tool: str):
                return f'{source_clean}.{clean_tool_name(tool)}'
            
            def parse_list_of_tools(tools: list[str]):
                return [parse_tool(tool) for tool in tools]
            
            df_tool_descriptions['source'] = self.source_name
            df_tool_descriptions['tool_name'] = df_tool_descriptions['tool']
            df_tool_descriptions['tool'] = df_tool_descriptions['tool'].apply(parse_tool)
            df_tool_descriptions['metadata'] = df_tool_descriptions['source'].apply(lambda x: {'source': x})
            df_tool_descriptions = df_tool_descriptions.drop_duplicates(subset=['tool'], keep='first')
            df_tool_descriptions = df_tool_descriptions.reset_index(drop=True)

            df_tool_embeddings['tool'] = df_tool_embeddings['tool'].apply(parse_tool)
            df_missing_tools = df_tool_descriptions[~df_tool_descriptions['tool'].isin(df_tool_embeddings['tool'])]
            df_tool_embeddings = pd.concat([df_tool_embeddings, df_missing_tools[['tool']]], ignore_index=True)
            df_tool_embeddings = df_tool_embeddings.drop_duplicates(subset=['tool'], keep='first')
            df_tool_embeddings = df_tool_embeddings.reset_index(drop=True)

            df_single_tool.rename(columns={'Tool': 'tool', 'Query': 'query'}, inplace=True)
            df_single_tool['tool'] = df_single_tool['tool'].apply(parse_tool)
            df_single_tool['tool'] = df_single_tool['tool'].apply(lambda x: [x])

            df_multi_tool['tool'] = df_multi_tool['tool'].apply(parse_list_of_tools)

            df_queries = pd.concat([df_single_tool, df_multi_tool], ignore_index=True)
            df_queries['source'] = self.source_name

            return df_queries, df_tool_descriptions, df_tool_embeddings

    class ToolLens:
        source_name: str = "ToolLens"

        def __init__(self):

            self.path: str = os.path.join(data_dir, 'ToolLens/')

            self.train = pd.read_csv(os.path.join(self.path, 'qrels/train.tsv'), sep='\t')
            """
            Columns: query-id, corpus-id, score
            """

            self.test = pd.read_csv(os.path.join(self.path, 'qrels/test.tsv'), sep='\t')
            """
            Columns: query-id, corpus-id, score
            """

            self.corpus = pd.read_json(os.path.join(self.path, 'corpus.jsonl'), lines=True)
            """
            Columns: _id, title, text, metadata
            """

            self.queries = pd.read_json(os.path.join(self.path, 'queries.jsonl'), lines=True)
            """
            Columns: _id, text, metadata
            """

            self.toollens = pd.read_json(os.path.join(self.path, 'toollens.json'))
            """
            Columns: apis, query
            """

        def parse_to_main(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            source_clean = clean_tool_name(self.source_name)
            df_queries, df_tool_descriptions, df_tool_embeddings = Datasets.Main.get_initial_dataframes()
            
            df = self.toollens.copy()
            col_tool_name = 'api_name'
            metadata_columns = [
                'category_name', 
                'tool_name', 
                'api_name', 
                'api_description', 
                'required_parameters',
                'optional_parameters',
                'method',
                'template_response',
            ]

            def parse_tool(tool: dict | pd.Series):
                path_items: list[str] = [source_clean]
                if 'category_name' in tool:
                    path_items.append(clean_tool_name(tool['category_name']))
                if 'tool_name' in tool:
                    path_items.append(clean_tool_name(tool['tool_name']))
                if 'api_name' in tool:
                    path_items.append(clean_tool_name(tool['api_name']))
                return '.'.join(path_items)
            
            def parse_list_of_tools(tools: list[dict]):
                return [parse_tool(tool) for tool in tools]

            def get_description(row: pd.Series):
                items = []
                if 'category_name' in row:
                    items.append(row['category_name'])
                if 'tool_name' in row:
                    items.append(row['tool_name'])
                if 'api_name' in row:
                    items.append(row['api_name'])
                if 'api_description' in row:
                    items.append(row['api_description'])
                return ', '.join(items)
            
            def get_metadata(row: pd.Series):
                dictionary = {key: row[key] for key in metadata_columns if key in row}
                dictionary['source'] = self.source_name
                return dictionary

            df['tool'] = df['apis'].apply(parse_list_of_tools)

            df_queries = df[['query', 'tool']].dropna()
            df_queries['source'] = self.source_name

            df_corpus =  pd.json_normalize(
                df['apis'].explode().drop_duplicates().reset_index(drop=True).dropna(),
                max_level=0
            )
            
            df_tool_descriptions['description'] = df_corpus.apply(get_description, axis=1)
            df_tool_descriptions['metadata'] = df_corpus.apply(get_metadata, axis=1)
            df_tool_descriptions['tool'] = df_corpus.apply(parse_tool, axis=1)
            df_tool_descriptions['tool_name'] = df_corpus[col_tool_name] 
            df_tool_descriptions['source'] = self.source_name

            df_tool_embeddings['tool'] = df_tool_descriptions['tool']

            return df_queries, df_tool_descriptions, df_tool_embeddings
    
    class ReverseChain:
        
        source_name: str = "ReverseChain"

        def __init__(self):
            self.path: str = os.path.join(data_dir, 'reverse-chain')
            
            self.data = pd.read_json(os.path.join(self.path, 'Compositional_multi-tool.json'))
            """
            Columns: APIs, Query, Label
            """
            self.data.set_index('Index', inplace=True)

        def parse_to_main(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            df_queries, df_tool_descriptions, df_tool_embeddings = Datasets.Main.get_initial_dataframes()
            source_clean = clean_tool_name(self.source_name)
            df = self.data.copy()
            metadata_columns = [
                'name',
                'description',
                'input_params',
                'output_params',
                'format'
            ]

            df.rename(columns={'APIs': 'apis', 'Query': 'query', 'Label': 'label'}, inplace=True)

            # Column 'APIs' contains a list of tools, each tool is a dict. 
            # We extract the tool name and description from each tool dict.
            # Apparently, the tool name can be in either 'name' or 'Name' key, 
            # and the description can be in either 'description' or 'Description' key.
            def clean_dict(d: dict):
                return {k.lower(): v for k, v in d.items()}
            
            def clean_list_of_dicts(l: list[dict]):
                return [clean_dict(d) for d in l]

            def parse_tool(tool: dict | pd.Series):
                name = source_clean + '.'
                if not ('name' in tool):
                    raise ValueError(f"Tool does not have a name: {tool}")
                name += clean_tool_name(tool['name'])
                return name

            def parse_list_of_tools(tools: list[dict]):
                return [parse_tool(tool) for tool in tools]

            def get_metadata(row: pd.Series):
                dictionary = {key: row[key] for key in metadata_columns if key in row}
                dictionary['source'] = self.source_name
                return dictionary

            df['apis'] = df['apis'].apply(clean_list_of_dicts)
            df['tool'] = df['apis'].apply(parse_list_of_tools)

            df_queries = df[['query', 'tool']].dropna()
            df_queries['source'] = self.source_name

            # df_single_tool = df[['query', 'tool']].where(df['tool'].apply(len) == 1).dropna()
            # df_single_tool['tool'] = df_single_tool['tool'].apply(lambda x: x[0])
            
            # df_multi_tool = df[['query', 'tool']].where(df['tool'].apply(len) > 1).dropna()

            df_corpus = pd.json_normalize(
                df['apis'].explode().drop_duplicates().reset_index(drop=True).dropna(),
                max_level=0
            )
            
            df_tool_descriptions['description'] = df_corpus['description']
            df_tool_descriptions['tool'] = df_corpus.apply(parse_tool, axis=1)
            df_tool_descriptions['tool_name'] = df_corpus['name']
            df_tool_descriptions['metadata'] = df_corpus.apply(get_metadata, axis=1)
            df_tool_descriptions['source'] = 'ReverseChain'

            # We check for duplicates by column 'tool', and drop duplicates. 
            # Keep the the one with the longest description.
            df_tool_descriptions['description_length'] = df_tool_descriptions['description'].apply(lambda x: len(x))
            df_tool_descriptions.sort_values('description_length', ascending=False, inplace=True)
            df_tool_descriptions = df_tool_descriptions.drop_duplicates(subset='tool', keep='first')
            df_tool_descriptions.drop(columns='description_length', inplace=True)
            
            df_tool_embeddings['tool'] = df_tool_descriptions['tool']
            
            return df_queries, df_tool_descriptions, df_tool_embeddings

    class Berkeley:
        source_name: str = "Berkeley"

        def __init__(self):
            self.path: str = os.path.join(data_dir, 'Berkeley-Function-Calling-Leaderboard/')
            
            self.simple = pd.read_json(os.path.join(self.path, 'BFCL_v3_simple.json'), lines=True)
            self.multiple = pd.read_json(os.path.join(self.path, 'BFCL_v3_multiple.json'), lines=True)

        def _load_file_(file_path: str):
            result = []
            with open(file_path) as f:
                file = f.readlines()
                for line in file:
                    result.append(json.loads(line))
            return result

        def parse_to_main(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            df_queries, df_tool_descriptions, df_tool_embeddings = Datasets.Main.get_initial_dataframes()
            combined = pd.concat([self.simple, self.multiple], axis=0).copy()

            source_clean = clean_tool_name(self.source_name)

            def get_user_query(history_overview: list[list[dict]]) -> str:
                if not len(history_overview) > 0:
                    raise ValueError("History is empty")
                chat_history_user = [item for item in history_overview[0] if item.get('role') == 'user']
                user_query = chat_history_user[0].get('content', '') if len(chat_history_user) > 0 else ''

                user_query = "Use a tool (or multiple if needed) to assist with the following question:\n\n" + user_query

                return user_query

            def get_used_tools(tools: list[dict[str]]) -> list[str]:
                if not len(tools) > 0:
                    raise ValueError("Tools list is empty")
                used_tools = [item.get('name', '') for item in tools]
                return used_tools

            df_queries['query'] = [get_user_query(entry) for entry in combined['question'].tolist()]
            df_queries['tool'] = [get_used_tools(entry) for entry in combined['function'].tolist()]
            df_queries['tool'] = df_queries['tool'].apply(lambda x: [f'{source_clean}.{clean_tool_name(o)}' for o in x])
            df_queries['source'] = self.source_name
            df_queries.drop_duplicates(subset=['query'], keep='first', inplace=True)
            df_queries.reset_index(drop=True, inplace=True)

            tools: list[dict[str]] = np.concatenate(combined['function'].tolist()).ravel().tolist()
            df_tool_descriptions['tool_name'] = [clean_tool_name(tool['name']) for tool in tools]
            df_tool_descriptions['tool'] = df_tool_descriptions['tool_name'].apply(lambda x: f'{source_clean}.{clean_tool_name(x)}')
            df_tool_descriptions['description'] = [tool.get('description', '') for tool in tools]
            df_tool_descriptions['metadata'] = tools
            df_tool_descriptions['source'] = self.source_name
            df_tool_descriptions.drop_duplicates(subset=['tool'], keep='first', inplace=True)
            df_tool_descriptions.reset_index(drop=True, inplace=True)

            df_tool_embeddings['tool'] = df_tool_descriptions['tool'].copy()

            return df_queries, df_tool_descriptions, df_tool_embeddings

class _ToolTestBaseClass:
    action_type = None

    def __init__(self, model_name, dataset_name):
        if self.action_type is None:
            raise Exception('action_type is not defined in the base class')

        self.model_name = model_name
        self.dataset_name = dataset_name

        self.dir_data = os.path.join(MetaToolResults.directory, self.action_type, dataset_name, model_name)
        self.filepath_multi_tool = os.path.join(self.dir_data, MetaToolResults.multi_tool_filename)
        self.filepath_general_test = os.path.join(self.dir_data, MetaToolResults.general_test_filename)
        self.filepath_hallucination_test = os.path.join(self.dir_data, MetaToolResults.hallucination_test_filename)

        self.df_multi_tool: pd.DataFrame = None
        self.df_general_test: pd.DataFrame = None
        self.df_hallucination_test: pd.DataFrame = None

        self.load_data()

    def load_data(self):
        if os.path.exists(self.filepath_multi_tool):
            self.df_multi_tool = pd.read_json(self.filepath_multi_tool)
        else:
            print(f'File not found: {self.filepath_multi_tool}')

        if os.path.exists(self.filepath_general_test):
            self.df_general_test = pd.read_json(self.filepath_general_test)
        else:
            print(f'File not found: {self.filepath_general_test}')

        if os.path.exists(self.filepath_hallucination_test):
            self.df_hallucination_test = pd.read_json(self.filepath_hallucination_test)
        else:
            print(f'File not found: {self.filepath_hallucination_test}')

class MetaToolResults:
    directory = os.path.join(data_dir, 'results', 'MetaTool')
    multi_tool_filename = 'multi_tool_prompt.json'
    general_test_filename = 'general_test.json'
    hallucination_test_filename = 'hallucination_prompt_new.json'

    class TestTought(_ToolTestBaseClass):
        action_type = 'tool_test_thought'

        def __init__(self, model_name, dataset_name):
            super().__init__(model_name, dataset_name)

            self.df_general_test = self._init_cols_(self.df_general_test)
            self.df_hallucination_test = self._init_cols_(self.df_hallucination_test)
            self.df_multi_tool = self._init_cols_(self.df_multi_tool)
        
        def _init_cols_(self, df: pd.DataFrame):
            if df is None:
                return None
            
            COL_RES = 'res'
            COL_ACTION_PROMPT = 'action_prompt'

            COL_TOOL_USAGE = 'res_tool_usage'
            COL_TOOL_AVAILABLE = 'available_tools'

            df[COL_TOOL_AVAILABLE] = df[COL_ACTION_PROMPT].apply(MetaToolResults.get_available_tools_from_action_prompt)
            df[COL_TOOL_USAGE] = df[COL_RES].apply(lambda x: MetaToolResults.evaluate_tool_usage(x))

            return df


    class TestAction(_ToolTestBaseClass):
        action_type = 'tool_test_action'

        def __init__(self, model_name, dataset_name):
            super().__init__(model_name, dataset_name)

            self.df_general_test = self._init_cols_(self.df_general_test)
            self.df_hallucination_test = self._init_cols_(self.df_hallucination_test)
            self.df_multi_tool = self._init_cols_(self.df_multi_tool, False)


        def _init_cols_(self, df: pd.DataFrame, single_tool=True):
            if df is None:
                return None

            COL_RES = 'action_res'
            COL_ACTION_PROMPT = 'action_prompt'
            COL_TOOL = 'tool'

            COL_TOOL_AVAILABLE = 'available_tools'
            COL_TOOL_USAGE = 'action_res_tool_usage'
            COL_TOOLS_USED = 'action_res_tools_used'

            # df[COL_TOOL_USAGE] = df[COL_RES].apply(lambda x: MetaToolResults.evaluate_tool_usage(x))
            df[COL_TOOL_AVAILABLE] = df[COL_ACTION_PROMPT].apply(MetaToolResults.get_available_tools_from_action_prompt)
            df[COL_TOOLS_USED] = df.apply(lambda x: MetaToolResults.evaluate_tool_selection(x[COL_RES], x[COL_TOOL_AVAILABLE], x[COL_TOOL], single_tool), axis=1)

            return df

    class TestReact(_ToolTestBaseClass):
        action_type = 'tool_test_react'

        def __init__(self, model_name, dataset_name):
            super().__init__(model_name, dataset_name)

            self.df_general_test = self._init_cols_(self.df_general_test)
            self.df_hallucination_test = self._init_cols_(self.df_hallucination_test)
            self.df_multi_tool = self._init_cols_(self.df_multi_tool)
        
        def _init_cols_(self, df: pd.DataFrame, single_tool=True):
            if df is None:
                return None

            COL_RES = 'react_res'
            COL_ACTION_PROMPT = 'react_prompt'
            COL_TOOL = 'tool'

            COL_TOOL_AVAILABLE = 'available_tools'
            COL_TOOLS_USED = 'action_res_tools_used'

            # df[COL_TOOL_USAGE] = df[COL_RES].apply(lambda x: MetaToolResults.evaluate_tool_usage(x))
            df[COL_TOOL_AVAILABLE] = df[COL_ACTION_PROMPT].apply(MetaToolResults.get_available_tools_from_action_prompt)
            df[COL_TOOLS_USED] = df.apply(lambda x: MetaToolResults.evaluate_tool_selection(x[COL_RES], x[COL_TOOL_AVAILABLE], x[COL_TOOL], single_tool), axis=1)

            return df

    def get_available_tools_from_action_prompt(text:str):
        return re.findall(r'tool name: ([^,]+)', text)

    def evaluate_tool_usage(text:str):
        """
        Evaluate whether the answer in the text indicates 'yes', 'no', or requires manual analysis based on tool usage awareness rules.
        
        Args:
            text (str): Input text to analyze.
            
        Returns:
            str: 'yes', 'no', or 'manual analysis'.
        """
        # Normalize text to lowercase for case-insensitive matching
        text = text.lower()

        # ------------ region Custom rules ------------

        # Check if first word is "no"
        if text.startswith("no"):
            return "no"
        
        # Check if first word is "yes"
        if text.startswith("yes"):
            return "yes"

        tokens = [t.strip(',.- ') for t in text.split(' ') if t.strip() != '']
        negative_sentences = [
            "not seem necessary", "not think it is necessary", "not need to use",
            "not necessary to use", "do not think i need to use"
        ]

        if any(phrase in text for phrase in negative_sentences):
            return "no"
        
        if 'no' in tokens and 'yes' not in tokens:
            return "no"

        # ------------ endregion ------------

        # ORIGINAL - Check for explicit "no" answers
        # if "no" in text and "yes" not in text:
        #     if any(phrase in text for phrase in [
        #         "not seem necessary", "not think it is necessary", "not need to use",
        #         "not necessary to use", "do not think i need to use"
        #     ]):
        #         return "no"
        #     return "yes"

        # ORIGINAL - Check for explicit "yes" answers
        # if "yes" in text and not any(neg in text for neg in ["no", "not", "don’t"]):
        #     return "yes"

        # ------------ region Custom rules ------------

        if 'yes' in tokens and not any(phrase in tokens for phrase in ["no", "not", "don’t", "don't"]):
            return "yes"
        
        # ------------ endregion ------------

        # Check for phrases indicating tool necessity
        if any(phrase in text for phrase in [
            "i need to use", "i think it is necessary", "i may need to use", "i would need to use",
            "i believe it is necessary to use", "would need access", "might be necessary to use",
            "i might need to use", "tools would be necessary", "may be necessary to use",
            "be beneficial to use", "i will need to use", "might need to use", "would need to rely",
            "may need to access"
        ]):
            return "yes"

        # For all other cases, mark as requiring manual analysis
        return "manual analysis"

    def evaluate_tool_selection(text: str, tool_list: list, ground_truth: list, single_tool=True):
        """
        Evaluate whether the tool selection in the text is correct, based on the rules for tool selection.

        Args:
            text (str): Input text to analyze.
            tool_list (list): List of possible tools to match against the text.
            ground_truth (list): List of correct tools.
            single_tool (bool): Whether the task is single-tool (True) or multi-tool (False).
            
        Returns:
            str: 'correct', 'incorrect', or 'manual analysis'.
        """
        # Normalize text to lowercase for case-insensitive matching
        text = text.lower().split()
        text = [t.strip().strip(""".,- :_\'\"`)(][}{""") for t in text if t.strip() != '']

        # Match tools in the text
        matches = [tool for tool in tool_list if tool.lower() in text]
        
        return matches

        # Handle single-tool tasks
        if single_tool:
            if len(matches) == 0:
                return "None"
            if len(matches) == 1 and "none" not in text:
                return "correct" if matches[0] in ground_truth else "incorrect"
            if len(matches) > 1:
                return "manual analysis"

        # Handle multi-tool tasks
        else:
            if len(matches) < 2:
                return "incorrect"
            if len(matches) == 2:
                return "correct" if all(tool in ground_truth for tool in matches) else "incorrect"
            if len(matches) > 2:
                return "manual analysis"

        return "manual analysis"

class ToolCoOccurrence:

    @staticmethod
    def get_co_occurrence_dataframe(
        df: pd.DataFrame,
        tool_col: str = 'tool',
        tool_count_col: str = 'tool_count',
        use_only_multi_tool_use_cases: bool = False,
        top_k: int = None
    ):
        """
        Computes the co-occurrence matrix of tools used in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing tool usage data.
            tool_col (str): Column name in the DataFrame that contains the list of tools used.
            tool_count_col (str): Column name in the DataFrame to store the count of tools used in each row.
            use_only_multi_tool_use_cases (bool): If True, only include rows where multiple tools are used.
            top_k (int): If specified, returns only the top_k tools based on their co-occurrence.

        Returns:
            tuple: A tuple containing:
                - co_occurrence (pd.DataFrame): Co-occurrence matrix of tools.
                - tool_names (list[str]): List of tool names corresponding to the co-occurrence matrix.

        Raises:
            ValueError: If the specified tool_col does not exist in the DataFrame.
        """

        df_copy = df.copy()

        # Check if the tool_col exists in the DataFrame
        if tool_col not in df_copy.columns:
            raise ValueError(f"Column '{tool_col}' not found in the DataFrame.")

        # Ensure the tool_col is a list of tools
        if not isinstance(df_copy[tool_col].iloc[0], list):
            df_copy[tool_col] = df_copy[tool_col].apply(lambda x: [x] if isinstance(x, str) else x)

        # Check if the tool_count_col exists in the DataFrame, if not specified, it will be calculated
        if tool_count_col not in df_copy.columns:
            # Create a new column with the count of tools in each row
            df_copy[tool_count_col] = df_copy[tool_col].apply(len)

        # If use_only_multi_tool_use_cases is True, filter the DataFrame to only include rows with multiple tools
        if use_only_multi_tool_use_cases:
            df_copy = df_copy[df_copy[tool_count_col] > 1]

        # Create a MultiLabelBinarizer to convert the list of tools into a binary matrix
        mlb = MultiLabelBinarizer()
        tool_matrix = mlb.fit_transform(df_copy[tool_col])
        tool_names = mlb.classes_.tolist()
        tool_ohe = pd.DataFrame(tool_matrix, columns=tool_names)

        # Calculate the co-occurrence matrix
        co_occurrence = tool_ohe.T @ tool_ohe

        # Fill NaN values with 0
        co_occurrence.fillna(0, inplace=True)

        if top_k is not None and top_k > 0:
            # Get the top_k tools based on the sum of co-occurrences
            top_k_indices = co_occurrence.sum().nlargest(top_k).index.tolist()
            co_occurrence = co_occurrence.loc[top_k_indices, top_k_indices]

        return co_occurrence

    @staticmethod
    def exclude_tool_combination_from_co_occurrence(
        co_occurrence: pd.DataFrame,
        tool_combination: list[str],
    ) -> pd.DataFrame:
        """
        Excludes a specific combination of tools from the co-occurrence matrix.

        Args:
            co_occurrence (pd.DataFrame): Co-occurrence matrix of tools.
            tool_combination (list[str]): List of tools to exclude from the co-occurrence matrix.

        Returns:
            pd.DataFrame: Updated co-occurrence matrix with the specified tool combination excluded.
        """
        tool_names = co_occurrence.columns.tolist()

        # Create a mask for the tools to exclude
        mask = [tool in tool_combination for tool in tool_names]
        
        # Exclude the specified tools from the co-occurrence matrix
        return co_occurrence.loc[~np.array(mask), ~np.array(mask)]

    @staticmethod
    def subtract_k_from_tool_combination_from_co_occurrence(
        co_occurrence: pd.DataFrame,
        tool_combination: list[str],
        k: int = 1
    ) -> pd.DataFrame:
        """
        Subtracts a specific combination of tools from the co-occurrence matrix by a factor of k.

        Args:
            co_occurrence (pd.DataFrame): Co-occurrence matrix of tools.
            tool_combination (list[str]): List of tools to subtract from the co-occurrence matrix.
            k (int): Factor by which to subtract the co-occurrence values.

        Returns:
            pd.DataFrame: Updated co-occurrence matrix with the specified tool combination subtracted.
        """
        df = co_occurrence.copy()

        if not tool_combination:
            return df
        elif not (len(list(set(tool_combination))) > 1):
            raise df
        
        # e.g. ['tool1', 'tool2', 'tool3'] -> [('tool1', 'tool2'), ('tool1', 'tool3'), ('tool2', 'tool3')]
        tool_combinations: list[tuple[str, str]] = []
        if len(tool_combination) == 2:
            tool_combinations = [(tool_combination[0], tool_combination[1])]
        else:
            tool_combinations = list(combinations(tool_combination, 2))

        tool_names = df.columns.tolist()
        for tool_1, tool_2 in tool_combinations:
            if tool_1 not in tool_names or tool_2 not in tool_names:
                continue
            
            # Subtract k from the co-occurrence values
            df.loc[tool_1, tool_2] = np.max([df.loc[tool_1, tool_2] - k, 0])
            df.loc[tool_2, tool_1] = np.max([df.loc[tool_2, tool_1] - k, 0])

        return df

    @staticmethod
    def get_top_k_co_occurring_tools(
        co_occurrence: pd.DataFrame,
        tool_name: str,
        k: int = 10,
        threshold: float = 0.8,
    ) -> list[str]:
        """
        Get the top k co-occurring tools for a given tool from the co-occurrence matrix.

        Args:
            co_occurrence (pd.DataFrame): Co-occurrence matrix of tools.
            tool_name (str): Name of the tool for which to find co-occurring tools.
            k (int): Number of top co-occurring tools to return.
            threshold (float): Minimum co-occurrence value to consider a tool as co-occurring.
        
        Returns:
            tools (list[str]): List of top k co-occurring tools for the specified tool.
        """


        df = co_occurrence.copy()

        if tool_name not in df.columns:
            return []

        # normalize the co-occurrence matrix by the diagonal
        # df = df.div(np.sqrt(np.diag(df)), axis=0).div(np.sqrt(np.diag(df)), axis=1)
        df_normalized = df.div(df.values.diagonal(), axis=0)

        # Get the co-occurrence values for the specified tool
        co_occurring_values = df_normalized[tool_name]
        
        # Remove the specified tool from the list
        co_occurring_values = co_occurring_values[co_occurring_values.index != tool_name]
        
        co_occurring_values = co_occurring_values[co_occurring_values >= threshold]
        if co_occurring_values.empty:
            return []
        
        # Sort the values in descending order and get the top k tools
        top_k_tools = co_occurring_values.nlargest(k).index.tolist()

        return top_k_tools
