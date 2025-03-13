from systems.generator_util import Generator, pdf_to_text
from benchmark.benchmark_api import System
import os
import pandas as pd

class ExampleBaselineSystem(System):
    def __init__(self,  model: str, name="baseline",*args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.dataset_directory = None # TODO(user): Update me
        self.model = model
        self.llm = Generator(model)

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
        results = self.llm.generate(query)
        print(results)
        return results


