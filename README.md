1) Generate a sample of parameter combinations for experiments

    - `uv run generate_parameter_sample.py [--n-samples] [--model] [-seed]`
        - `n-samples` : desired number of unique parameter combinations (generated using latin hypercube sampling)
        - `model` : either `cpt` (cumulative prospect theory enabled) or `nocpt`
        - `seed` : random seed

2) Generate initial agent states for experiments
    - `uv run generate_initial_states.py [--n-agents] [--grid-size] [--seed]`
        - `n-agents` : desired number of unique initializations (generated using latin hypercube sampling)
        - `grid-size` : granularity of grid used for value iteration
        - `seed` : random seed

3) Run a set of experiments for a particular model (cpt or nocpt)
    - `uv run experiment.py [--n-steps] [--max-workers] [--model] [--grid-size]`
        - `n-steps` : number of steps to run each agent simulation
        - `max-workers` : specifies desired number of concurrent processes to run experiments in parallel
        - `model` : either `cpt` (cumulative prospect theory enabled) or `nocpt`
        - `grid-size` : granularity of grid used for value iteration (must match the value used to generate initial agent states)