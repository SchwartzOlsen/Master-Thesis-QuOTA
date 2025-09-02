import sys
import getopt
import os
import matplotlib
matplotlib.use('Agg')
import asyncio

sys.path.append(os.path.abspath(os.path.join('..')))
import lib

RANDOM_STATE = 0

### OpenAI
# https://platform.openai.com/docs/models
# CLIENT_TYPE = lib.agent.ClientType.OPENAI
# MODEL_NAME = 'gpt-4o-mini'    # $0.15 • $0.6  - https://platform.openai.com/docs/models/gpt-4o-mini
# MODEL_NAME = 'gpt-4.1-nano'   # $0.1 • $0.4   - https://platform.openai.com/docs/models/gpt-4.1-nano (NOT USED)
# MODEL_NAME = 'gpt-4.1-mini'   # $0.4 • $1.6   - https://platform.openai.com/docs/models/gpt-4.1-mini (NOT USED)
# MODEL_NAME = 'gpt-4.1'        # $2 • $8       - https://platform.openai.com/docs/models/gpt-4.1 (NOT USED)

### Groq 
# CLIENT_TYPE = lib.agent.ClientType.GROQ
# MODEL_NAME = 'llama-3.1-8b-instant'
# MODEL_NAME = 'llama-3.3-70b-versatile'

### TogetherAI
# CLIENT_TYPE = lib.agent.ClientType.TOGETHER
# MODEL_NAME = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'

if __name__ == '__main__':

    argumentList = sys.argv[1:]
    options = "hmo:"
    long_options = ["experiment_type=", "client_type=", "model_name=", "source=", "max_num_samples="]

    experiment_type: lib.experiment.ExperimentSetupType = None
    client_type: lib.agent.ClientType = None
    model_name: str = None
    source: str = None
    max_num_samples: int = None
    valid_sources = ['ToolE', 'Berkeley', 'ToolLens', 'ReverseChain']

    try:
        arguments, values = getopt.getopt(argumentList, options, long_options)
        for currentArgument, currentValue in arguments:
            if currentArgument in ("--client_type"):
                temp_client_type = [c_type for c_type in lib.agent.ClientType if c_type.name == currentValue]
                if temp_client_type:
                    client_type = temp_client_type[0]
                else:
                    raise ValueError(f"Invalid value for client_type: {currentValue}. Valid values are: {[c_type.name for c_type in lib.agent.ClientType]}")
            elif currentArgument in ("--model_name"):
                model_name = currentValue
            elif currentArgument in ("--source"):
                if currentValue not in valid_sources:
                    raise ValueError(f"Invalid value for source: {currentValue}. Valid values are: {valid_sources}")
                source = currentValue
            elif currentArgument in ("--max_num_samples"):
                if not currentValue.isdigit():
                    raise ValueError(f"Invalid value for max_num_samples: {currentValue}")
                max_num_samples = int(currentValue)
            elif currentArgument in ("--experiment_type"):
                temp_experiment_type = [e_type for e_type in lib.experiment.ExperimentSetupType if e_type.name == currentValue]
                if temp_experiment_type:
                    experiment_type = temp_experiment_type[0]
                else:
                    raise ValueError(f"Invalid value for experiment_type: {currentValue}. Valid values are: {[e_type.name for e_type in lib.experiment.ExperimentSetupType]}")
    except getopt.error as err:
        print (str(err))
        sys.exit(2)

    if client_type is None:
        raise ValueError(f"Client type must be specified.")
    if model_name is None:
        raise ValueError(f"Model name must be specified.")
    if source is None:
        raise ValueError(f"Source must be specified.")
    if experiment_type is None:
        raise ValueError(f"Experiment type must be specified.")

    (
        react_agent, 
        df_test_single, 
        df_test_multi, 
        df_tool_descriptions,
        df_tool_embeddings,
        prompt_single, 
        prompt_multi, 
        experiment_name, 
        plot_title, 
        description
    ) = lib.experiment.load_experiment_setup(
        setupType=experiment_type,
        client_type=client_type,
        model_name=model_name,
        source=source,
        num_samples=max_num_samples,
        random_state=RANDOM_STATE,
    )

    # Run experiment

    concurrency_limit = 64 if client_type.value == lib.agent.ClientType.OPENAI.value else 8

    asyncio.run(
        lib.experiment.test_all_tasks(
            react_agent,
            df_test_single=df_test_single,
            df_test_multi=df_test_multi,
            prompt_single=prompt_single,
            prompt_multi=prompt_multi,
            description=description,
            experiment_name=experiment_name,
            data_source=source if source is not None else "Main",
            plot_title=plot_title,
            concurrency_limit=64
        )
    )
