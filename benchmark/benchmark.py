import json
import logging
import math
from nltk.translate.bleu_score import sentence_bleu
import os
import re
from rouge_score import rouge_scorer
from typing import Any, Dict, List, Tuple

from benchmark_api import System

class Executor:
    def __init__(
            self,
            system: System,
            workload_path: str | os.PathLike,
            results_directory: str | os.PathLike,
            verbose=False
        ):
        """
        `workload_path` is a json file containing workloads.
        """
        self.system = system
        if self.system.dataset_directory is None:
            raise Exception("Executor __init__ error: System.process_dataset was not called.")
        with open(workload_path) as f:
            self.workload = json.load(f)
        self.num_queries = len(self.workload)
        self.cur_query_id = 0
        self.results_directory = results_directory
        self.workload_path = workload_path
        self.verbose = verbose
    
    def run_task(self, task: Dict[str, Any]) -> Dict[str, str | Dict | List]:
        """
        Takes a task entry and runs it on the test subject System
        See `workload/` for examples of the input.
        The output is formatted as follows
        {
            "task_id": str
            "model_output": text or json - the content directly outputed by test System
            "subresponses": [ ... {"model_output": ..., "subresponses": ... } ... ]
        }
        """
        if self.verbose:
            print(f"task_id: {task["id"]}")
            print(f"query: {task["query"]}")
        model_output = self.system.serve_query(task["query"])
        response = {}
        response["task_id"] = task["id"]
        response["model_output"] = model_output
        response["subresponses"] = []
        for subtask in task["subtasks"]:
            subresponse = self.run_task(subtask)
            response["subresponses"].append(subresponse)
        return response
    
    def run_next_task(self):
        """
        Iterator model to run all tasks in this test workload.
        """
        if self.cur_query_id >= self.num_queries:
            return None
        result = self.run_task(self.workload[self.cur_query_id])
        self.cur_query_id += 1
        return result
    
    def reset_task_iterator(self):
        self.cur_query_id = 0
    
    def run_workload(self, cache_system_output: bool = True):
        """
        Runs all the tasks in the given workload.
        Results are cached under self.results_directory/output_cache.
        Note that the cache content here are the raw response dicts, not
        evaluation results.
        """
        if self.verbose:
            print(f"------------- Running workload at {self.workload_path} ----------------")
        results = []
        while True:
            result = self.run_next_task()
            if result is None:
                break
            results.append(result)
        
        if cache_system_output:
            #TODO: implement me
            logging.warning("Executor.run_workload not implemented.")
            pass

        return results


class Evaluator:
    def __init__(
            self,
            workload_path: str | os.PathLike,
            task_fixture_directory: str | os.PathLike,
            results_directory: str | os.PathLike
        ):
        """
        `task_fixtures_directory` contains the tasks fixtures for finding valid
        evaluation methods of each type of task and response.
        """
        self.workload_path = workload_path
        self.task_fixture_directory = task_fixture_directory
        self.results_directory = results_directory
        with open(workload_path) as f:
            self.workload = json.load(f)
        with open(task_fixture_directory/'task_fixtures.json') as f:
            self.task_fixtures = json.load(f)
        with open(task_fixture_directory/'input_type_fixtures.json') as f:
            self.input_type_fixtures = json.load(f)
        with open(task_fixture_directory/'output_type_fixtures.json') as f:
            self.output_type_fixtures = json.load(f)
        with open(task_fixture_directory/'answer_type_fixtures.json') as f:
            self.answer_type_fixtures = json.load(f)
        
        self.rouge_score_engine = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def _normalize_string(s: str) -> str:
        """Normalize a string by removing spaces, punctuation, and making it lowercase."""
        return re.sub(r'[^a-z0-9]', '', s.lower())

    def evaluate_response_with_metric(self, system_response: str, target_answer: str | int | float, metric_name: str) -> float:
        # Evaluate a response with a given metric
        # Returns (aggregate score, [list of scores for each response item])
        try:
            system_response_dict = json.loads(system_response)
            system_answer = system_response_dict["answer"]
        except Exception as e:
            # TODO: Add verbose control flag to log the json marshalling failure and clean up trace logging
            print(f"evaluate_response_with_metric: failed to marshal system response into json: {e}.")
            return 0

        if metric_name == "mean_relative_absolute_error":
            try:
                return math.abs((float(system_answer) - float(target_answer)) / float(target_answer))
            except Exception as e:
                logging.error(f"evaluate_answer_with_metric: mean_relative_absolute_error compute error: {e}")
                return -1
        elif metric_name == "mean_absolute_error":
            try:
                return math.abs(float(system_answer) - float(target_answer))
            except Exception as e:
                logging.error(f"evaluate_answer_with_metric: mean_absolute_error compute error: {e}")
                return -1
        elif metric_name == "mean_squared_error":
            try:
                return math.abs((float(system_answer) - float(target_answer)) * float(system_answer) - float(target_answer))
            except Exception as e:
                logging.error(f"evaluate_answer_with_metric: mean_squared_error compute error: {e}")
                return -1
        elif metric_name == "bleu":
            try:
                return sentence_bleu(target_answer.strip().split(), system_answer.strip().split())
            except Exception as e:
                logging.error(f"evaluate_answer_with_metric: bleu compute error: {e}")
                return -1
        elif metric_name == "rouge":
            try:
                return self.rouge_score_engine(target_answer, system_answer)[0]
            except Exception as e:
                logging.error(f"evaluate_answer_with_metric: bleu compute error: {e}")
                return -1
        elif metric_name == "llm":
            logging.error("evaluate_answer_with_metric: llm evaluation not yet implemented")
            return -1
        elif metric_name == "f1":
            logging.error("evaluate_answer_with_metric: f1 evaluation not yet implemented")
            return -1
        elif metric_name == "precision":
            logging.error("evaluate_answer_with_metric: precision evaluation not yet implemented")
            return -1
        elif metric_name == "recall":
            logging.error("evaluate_answer_with_metric: recall evaluation not yet implemented")
            return -1
        logging.error(f"evaluate_answer_with_metric: unrecognized metric {metric_name}")
        return -1

    def _evaluate_result_for_task(self, response: Dict[str, Any], task: Dict[str, Any]):
        """
        Evaluate results on all applicable metrics as specified in the task fixture for a 
        task in the workload.
        The caller should format the responses in the same structure as the workload.
        """
        all_evaluation_results = []
        target_metrics = self.answer_type_fixtures[task['answer_type']]
        system_answer = response["answer"]
        assert(task["id"] == response["task_id"])
        evaluation_result = {"task_id": task["id"]}
        for metric in target_metrics:
            score = self.evaluate_response_with_metric(system_answer, task["model_output"], metric)
            evaluation_result[metric] = score
        all_evaluation_results.append(evaluation_result)
        for i, subtask in enumerate(task["subtasks"]):
            subtask_result = self._evaluate_result_for_task(response["subresponses"][i], subtask)
            all_evaluation_results.append(subtask_result)
        
        return all_evaluation_results

    def evaluate_results(self, responses: List[Dict[str, Any]]):
        """
        Evaluate results on all applicable metrics as specified in the task fixtures.
        The caller should format the responses in the same structure as the workload.
        """
        all_evaluation_results = []
        for task_idx, task in enumerate(self.workload):
            evaluation_results = self._evaluate_result_for_task(responses[task_idx], task)
            all_evaluation_results.append(evaluation_results)
        return all_evaluation_results

class Benchmark:
    def __init__ (
            self,
            system_name: str,
            task_fixture_directory: str | os.PathLike,
            cache_system_output: bool = False,
    ):
        system_module = __import__("system")
        system_class_ = getattr(system_module, system_name)
        self.system = system_class_()
        self.cache_system_output = cache_system_output
        self.task_fixture_directory = task_fixture_directory
    
    def run_benchmark(
            self,
            dataset_directory: str | os.PathLike,
            results_directory: str | os.PathLike,
            workload_path: str | os.PathLike,
            verbose: bool = False
        ):
        self.system.process_dataset(dataset_directory)
        executor = Executor(
            system=self.system,
            workload_path=workload_path,
            results_directory=results_directory,
            verbose=verbose
        )
        results = executor.run_workload(cache_system_output=self.cache_system_output)
        evaluator = Evaluator(
            workload_path=workload_path,
            task_fixture_directory=self.task_fixture_directory,
            results_directory=results_directory
        )
        evaluator.evaluat_results(results)

