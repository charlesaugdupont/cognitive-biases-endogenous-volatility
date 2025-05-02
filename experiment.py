from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import generate_samples
from tqdm.auto import tqdm
from model import *
import argparse
import pickle
import os

np.random.seed(23)

def process_row(row, n_steps, n_agents, output_dir, idx):
    # unpack model parameters
    alpha, prob_health_decrease, prob_health_increase, gamma, w_delta_scale, omega, eta = row

    # compute optimal policy
    policy, params, _ = value_iteration(
        N=200,
        theta=0.88,
        omega=omega,
        eta=eta,
        beta=0.95,
        alpha=alpha,
        P_H_decrease=prob_health_decrease,
        P_H_increase=prob_health_increase,
        gamma=gamma,
        w_delta_scale=w_delta_scale
    )

    # run agent simulation
    wealth, health = simulate(
        params,
        policy,
        n_steps,
        n_agents
    )

    result = {
        "params": params,
        "wealth": wealth,
        "health": health,
        "policy": policy
    }

    output_file_name = os.path.join(output_dir, f"{idx}.pickle")
    with open(output_file_name, 'wb') as f:
        pickle.dump(result, f)

    return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--n-agents", type=int, default=10000)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="results_shocks_4")
    args = parser.parse_args()

    N_SAMPLES = args.n_samples
    N_AGENTS = args.n_agents
    N_STEPS = args.n_steps
    MAX_WORKERS = args.max_workers
    OUTPUT_DIR = args.output_dir

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    samples = np.array([
        [6.08557311e-01, 5.39491850e-01, 1.31725783e-01, 5.88858294e-01,
            7.51956507e-02, 1.52545705e+00, 6.67335215e-01],
        [3.56310635e-01, 3.10060933e-02, 7.12464054e-01, 4.50248147e-01,
            7.93444014e-01, 3.34334804e+00, 8.61808423e-01],
        [7.78553211e-02, 8.82024781e-02, 9.03681489e-01, 7.06637914e-01,
            7.28233815e-01, 2.84785616e+00, 5.47467877e-01],
        [7.51772516e-01, 5.61035711e-02, 4.67470788e-01, 4.93344766e-01,
            5.09909617e-01, 1.41074885e+00, 5.81688219e-01],
        [2.57820784e-01, 8.17573444e-01, 4.68468951e-01, 4.62690918e-01,
            2.50188626e-01, 2.34886920e+00, 6.81544186e-01],
        [5.54624241e-02, 7.15976137e-01, 8.78986214e-01, 4.32816461e-01,
            7.72181794e-01, 1.15102132e+00, 9.18957187e-01],
        [5.38354343e-01, 1.15837842e-01, 9.49526441e-01, 7.54434383e-01,
            3.06769169e-01, 3.24740161e+00, 6.28190092e-01],
        [5.31362450e-01, 8.29252475e-01, 1.36540532e-01, 4.45594318e-01,
            7.37794547e-01, 2.23148591e+00, 8.02981554e-01],
        [5.40923250e-01, 6.24736777e-01, 2.42987039e-01, 6.56142768e-01,
            9.14735639e-01, 2.15054383e+00, 9.16824344e-01],
        [3.65508749e-01, 9.43248321e-01, 6.28057827e-01, 7.16004514e-01,
            1.90785728e-01, 2.83283130e+00, 9.85049520e-01],
        [6.53341538e-01, 7.13734895e-01, 9.57112220e-01, 4.02838188e-01,
            8.86174763e-01, 2.36785782e+00, 6.43625619e-01],
        [3.65396274e-02, 8.26971097e-01, 3.78387830e-03, 5.04234591e-01,
            1.93273195e-01, 1.47294394e+00, 8.74667117e-01],
        [1.86336158e-02, 2.87691098e-01, 3.04444645e-01, 4.97612675e-01,
            3.92711871e-02, 3.35659900e+00, 5.12331145e-01],
        [5.45411345e-01, 2.23860958e-01, 1.67602812e-01, 6.27147782e-01,
            7.08469262e-01, 3.79988533e+00, 6.16095378e-01],
        [8.39272423e-01, 1.73016491e-03, 7.51309170e-01, 4.74689120e-01,
            9.76027672e-01, 1.11775207e+00, 8.76993943e-01],
        [2.98229178e-01, 5.04852525e-01, 9.97872192e-01, 4.18002836e-01,
            2.83964036e-01, 2.74101579e+00, 5.25016940e-01],
        [3.77476814e-01, 7.71398335e-04, 3.36955169e-01, 5.96810802e-01,
            1.82660147e-01, 3.51270680e+00, 9.22113387e-01],
        [2.17640218e-01, 1.59207999e-02, 7.56552898e-01, 6.01035411e-01,
            1.97684028e-02, 3.40635406e+00, 9.45024070e-01]
    ])

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row, N_STEPS, N_AGENTS, OUTPUT_DIR, idx) for idx,row in enumerate(samples)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            output_file_name = future.result()
