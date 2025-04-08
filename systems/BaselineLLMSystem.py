from systems.generator_util import Generator, pdf_to_text
from benchmark.benchmark_api import System
import sys
sys.path.append('./')
from utils import * 

class BaselineLLMSystem(System):
    """
    A baseline system that uses a large language model (LLM) to process datasets and serve queries.
    """

    def __init__(self, model: str, name="baseline", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.dataset_directory = None  # TODO(user): Update me
        self.model = model
        self.llm = Generator(model, verbose=self.verbose)
    
    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        """
        Process the dataset located in the specified directory.
        The dataset can contain files in various formats (e.g., PDF, CSV).
        """
        self.dataset_directory = dataset_directory
        self.dataset = {}
        # read the files
        for file in os.listdir(dataset_directory):
            if file.endswith(".pdf"):
                self.dataset[file] = pdf_to_text(os.path.join(dataset_directory, file))
            elif file.endswith(".csv"):
                self.dataset[file] = pd.read_csv(os.path.join(dataset_directory, file))

    def serve_query(self, query: str) -> dict | str:
        """
        Serve a query using the LLM.
        The query should be in natural language, and the response can be in either natural language or JSON format.
        """
        # TODO: Implement the logic to handle different types of queries
        results = self.llm.generate(query)
        return results