# type: ignore
import os
import sys
import re
sys.path.append("./")
from benchmark.benchmark_utils import print_error, print_warning

import json
import pandas as pd
import subprocess
from typeguard import typechecked
from typing import Dict, List

from benchmark.benchmark_api import System
import argparse
import threading
import time

from dotenv import load_dotenv
from text_inspector_tool import TextInspectorTool
from tools import list_filepaths, list_input_filepaths

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    AgentLogger
)
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
load_dotenv()

class SmolagentsDeepResearch(System):
    def __init__(self, model: str, name="example", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.dataset_directory = None  # TODO(SUT engineer): Update me
        self.model = model # claude-3-7-sonnet-latest
        #self.llm = None # TODO(SUT engineer): Update me
        self.verbose = kwargs.get("verbose", False)
       
        self.debug = False
        self.output_dir = kwargs.get("output_dir", os.path.join(os.getcwd(), "testresults"))

        # Ablation studies: add additional parameters
        self.number_sampled_rows = kwargs.get("number_sampled_rows", 100)
        self.max_steps = kwargs.get("max_steps", 20)

        self.question_output_dir = None  # to be set in run()
        self.question_intermediate_dir = None  # to be set in run()
    
    def create_agent(self, model, workload="environment", query_ID="1", result_root_dir="./results"):
    
        code_model_params = {
            "model_id": model, #"claude-3-7-sonnet-latest",
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": 8192,
        }
        code_model = LiteLLMModel(**code_model_params)

        reasoning_model_params = {
            "model_id": model,
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": 8192,
        }
        reason_model = LiteLLMModel(**reasoning_model_params)
        verbosity_level = 2
        # TODO: set up intermediate logging directory
        logger_path = os.path.join(result_root_dir, f"{model}/{workload}/")
        os.makedirs(logger_path, exist_ok=True)
        logger_path = os.path.join(logger_path, f"{query_ID}.txt")
        if os.path.exists(logger_path):
            os.remove(logger_path)
        logger = AgentLogger(level=verbosity_level, log_file=logger_path)

        text_limit = 100000
        
        manager_agent = CodeAgent(
            model=code_model,
            planner_model=reason_model,
            tools=[visualizer, TextInspectorTool(reason_model, text_limit), list_input_filepaths], #process_beach_data
            max_steps=self.max_steps,
            verbosity_level=verbosity_level,
            additional_authorized_imports=["*"],
            planning_interval=4,
            logger=logger,
        )

        return manager_agent

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        # read files based on the dataset_directory, could be pdf, csv, etc.
        # store the data in a dictionary
        # store the dictionary in self.dataset_directory
        self.dataset_directory = dataset_directory
        self.dataset = {}
        # read the files
        for file in os.listdir(dataset_directory):
            if file.endswith(".csv"):
                self.dataset[file] = pd.read_csv(os.path.join(dataset_directory, file))

    def serve_query(self, query: str) -> dict | str:
        results = self.llm.generate(query)
        # print(results)
        return results
