"""This script can be used to evaluate the results of a system run.
The systems can be called using the run_pipeline function in system.api with the following run types:
- baseline: single model name (str)
- cross_validation: list of model names to cross validate, and a final model to merge the results
- reflection: one model for reflection, and another model to critic and decide to accept or reject the reflection
"""

from __future__ import print_function
import json
import os
import argparse
import traceback

import pandas as pd

from benchmark.metrics import Precision, Recall, F1, BleuScore, RougeScore, Success
from systems import ExampleBaselineSystem
from fixtures import SUTFactory, MetricFactory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sut", default="baseline-gpt-4o", help="The system to benchmark")
    parser.add_argument(
        "--workload",
        default="workload/jun-easy.json",
        help="The json file containing the input queries",
    )
    parser.add_argument(
        "--result",
        default="./results",
        help="The root path where the results of the pipelines are stored.",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        help="Whether to print filenames as they are processed",
    )

    args = parser.parse_args()
    sut = args.sut
    workload = args.workload
    RESULT_DIR = args.result

    verbose = bool(args.verbose)
    workload_name = os.path.basename(workload)
    result_file = f"{RESULT_DIR}/{sut}/{workload_name}_measures.csv"

    with open(workload) as f:
        queries = json.load(f)

    result_path = f"{RESULT_DIR}/{sut}/{workload_name}"
    try:
        with open(result_path) as f:
            sut_answers = json.load(f)
    except Exception:
        print(
            f"Could not load the results of the system {sut} for workload {workload_name}. Processing"
        )
        data_path = "data/TODO"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        sut = SUTFactory(sut)
        sut.process_dataset(data_path)
        for idx, query in enumerate(queries):
            if verbose:
                print(f"Processing query {idx}")
            result = sut.serve_query(query)
            sut_answers.append(result)
        with open(result_path, "w") as f:
            json.dump(sut_answers, f)

    workload_measures = []
    for idx, query in enumerate(queries):
        target = query
        predicted = sut_answers[idx]

        applicable_metrics = query.get("metrics", None)
        breakpoint()
        for metric in applicable_metrics:
            metric_fn = MetricFactory(metric)
            try:
                value = metric_fn(predicted, target)
            except Exception:
                print("Exception:", traceback.format_exc())
                if not verbose:
                    print("On query:", idx)
                value = 0

            dict_measures = {
                "workload": workload_name,
                "query_idx": idx,
                "metric": metric.name,
                "value": value,
            }
            workload_measures.append(dict_measures)

    results_df = pd.DataFrame(workload_measures)
    results_df.to_csv(result_file, index=False)
    if verbose:
        print(results_df)

    # Logic to aggregate the results
    # workload_results = []


if __name__ == "__main__":
    main()
