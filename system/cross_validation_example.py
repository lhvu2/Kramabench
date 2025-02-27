from typing import Any
import os
import pandas as pd
from system.generator_util import generator_factory, pdf_to_text
from benchmark.benchmark_api import System


MERGER_PROMPT = """
You are a helpful assistant that merges the results of the models.

You will be given the original request, and the results of the models.

You need to:
1. Take the results of other models as the hints to the original request.
2. Merge the results into a single result that combines the best of all the results.
3. Return the merged result in the same format as the original request. The final results still need to be concise.
4. Please put the final results in <merged_result> tags.

Below is the original request and the results of the other models:

<original_request>
{original_request}
</original_request>

<results_of_other_models>
{results_of_other_models}
</results_of_other_models>
"""


def format_prompt_for_merger(original_request: str, results_of_other_models: list[str]) -> str:
    # Wrap each result in <result> tags
    formatted_results = [f"<result>{result}</result>" for result in results_of_other_models]
    return MERGER_PROMPT.format(
        original_request=original_request,
        results_of_other_models="\n".join(formatted_results)
    )

# TODO: We need to set higher temperature for the suggesters models,
# and lower temperature for the merger model.
class ExampleCrossValidationSystem(System):
    def __init__(self, models: dict[str, Any], *args, **kwargs):
        self.name = "cross_validation"
        self.dataset_directory = None # TODO(user): Update me
        self.models = models
        self.verbose = args.get("verbose", False)
        super().__init__(self.name, *args, **kwargs)
        assert "suggesters" in models
        assert "merger" in models
        assert isinstance(models["suggesters"], list)
        assert isinstance(models["merger"], str)

        # run the models
        self.suggesters = []
        self.merger = None
        for model in models["suggesters"]:
            self.suggesters.append(generator_factory(model, verbose=self.verbose))
        self.merger = generator_factory(models["merger"], verbose=self.verbose)

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
        suggester_results = []
        for model in self.suggesters:
            suggester_results.append(model(query))
        merger_prompt, stats = format_prompt_for_merger(query, suggester_results)
        merger, stats = self.merger(merger_prompt)
        return merger, stats