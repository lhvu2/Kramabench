from typing import Any
from system.generator_util import generator_factory, pdf_to_text
from benchmark.benchmark_api import System
import os
import pandas as pd

MAX_NUM_REFLECTIONS = 3

EXECUTOR_PROMPT = """
You are a helpful assistant that executes the given request, and you'll be given:
1. The original request
2. The results and feedbacks of the answer from previous rounds, 
they will be wrapped in <previous_results_and_feedbacks> tags, and each set of result and feedback is wrapped in <result_and_feedback> tags.
If <previous_results_and_feedbacks> is empty, there is no previous results and feedbacks.

And please take the feedbacks into consideration to execute the request again, and put the results in <result> tags.
```
<original_request>
{original_request}
</original_request>

<previous_results_and_feedbacks>
{previous_results_and_feedbacks}
</previous_results_and_feedbacks>
```
"""

REFLECTOR_PROMPT = """
You are a helpful assistant that reviews the results and gives feedbacks, and you'll be given:
1. The original request
2. The results of the answer from another assistant

Please review the results and give feedbacks, and put the feedback in <feedback> tags.
Note that if the result is good enough, you can just return "good" as the feedback.

```
<original_request>
{original_request}
</original_request>

<result>
{result}
</result>
```
"""

def generate_executor_prompt(original_request: str, previous_results_and_feedbacks: list[tuple[str, str]]):
    return EXECUTOR_PROMPT.format(
        original_request=original_request,
        previous_results_and_feedbacks="\n".join(
            [
                f"<result_and_feedback>Results: {result}\nFeedback:     {feedback}\n</result_and_feedback>"
                for result, feedback in previous_results_and_feedbacks
            ]
        ),
    )

def generate_reflector_prompt(original_request: str, result: str):
    return REFLECTOR_PROMPT.format(
        original_request=original_request,
        result=result,
    )

class ExampleReflectionSystem(System):
    def __init__(self, models: dict[str, Any], *args, **kwargs):
        self.name = "reflection"
        self.dataset_directory = None # TODO(user): Update me
        self.models = models
        self.verbose = args.get("verbose", False)
        super().__init__(self.name, *args, **kwargs)
        assert "executor" in models
        assert "reflector" in models
        assert isinstance(models["executor"], str)
        assert isinstance(models["reflector"], str)

        self.executor = generator_factory(models["executor"], verbose=self.verbose)
        self.reflector = generator_factory(models["reflector"], verbose=self.verbose)


    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        # read files based on the dataset_directory, could be pdf, csv, etc.
        # store the data in a dictionary
        # store the dictionary in self.dataset_directory
        self.dataset_directory = dataset_directory
        self.dataset = {}
        # read the files
        for file in os.listdir(dataset_directory):
            if file.endswith(".pdf"):
                self.dataset[file] = pdf_to_text(os.path.join(dataset_directory, file))
            elif file.endswith(".csv"):
                self.dataset[file] = pd.read_csv(os.path.join(dataset_directory, file))

    def serve_query(self, query: str) -> dict | str:
        not_good_enough = True
        num_rounds = 0
        previous_results_and_feedbacks = []
        status = ""
        while not_good_enough and num_rounds < MAX_NUM_REFLECTIONS:
            executor_prompt = generate_executor_prompt(query, previous_results_and_feedbacks)
            result, status = self.executor(executor_prompt)
            reflector_prompt = generate_reflector_prompt(query, result)
            feedback, _ = self.reflector(reflector_prompt)

            if feedback == "<feedback>good</feedback>":
                not_good_enough = False
            num_rounds += 1
            # update the previous results and feedbacks
            previous_results_and_feedbacks.append((result, feedback))
        
        return result, status
