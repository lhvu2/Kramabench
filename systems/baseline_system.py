import sys
sys.path.append('./')
from typeguard import typechecked

from systems.generator_util import Generator
from benchmark.benchmark_api import System
from utils.baseline_utils import * 

QUESTION_PROMPT = """
You are a helpful assistant that generates a plan to solve the given request, and you'll be given:Your task is to answer the following question based on the provided data sources.
Question: {query}
Data file names: {file_names}

The following is a snippet of the data files:
{data}

Now think step-by-step carefully. 
First, provide a step-by-step reasoning of how you would arrive at the correct answer.
Do not assume the data files are clean or well-structured (e.g., missing values, inconsistent data type in a column).
Do not assume the data type of the columns is what you see in the data snippet (e.g., 2012 in Year could be a string, instead of an int). So you need to convert it to the correct type if your subsequent code relies on the correct data type (e.g., cast two columns to the same type before joining the two tables).
You have to consider the possible data issues observed in the data snippet and how to handle them.
Output the steps in a JSON format with the following keys:
- id: always "main-task" for the main task. For each subtask, use "subtask-1", "subtask-2", etc.
- query: the question the step is trying to answer. Copy down the question from above for the main task.
- data_sources: the data sources you need to check to answer the question. Include all the file names you need for the main task.
- subtasks: a list of subtasks. Each subtask should have the same structure as the main task.
For example, a JSON object for the task might look like this:
{example_json}
You can have multiple steps, and each step should be a JSON object.
Your output for this task should be a JSON array of JSON objects.
Mark the JSON array with ````json` and ````json` to indicate the start and end of the code block.

Then, provide the corresponding Python code to extract the answer from the data sources. 
The data sources you may need to answer the question are: {file_paths}.

If possible, print the answer (in a JSON format) to each step you provided in the JSON array using the print() function.
Use "id" as the key to print the answer.
For example, if you have an answer to subtask-1, subtask-2, and main-task (i.e., the final answer), you should print it like this:
print(json.dumps(
{{"subtask-1": answer1, 
"subtask-2": answer2, 
"main-task": answer
}}, indent=4))
You can find a suitable indentation for the print statement. Always import json at the beginning of your code.

Mark the code with ````python` and ````python` to indicate the start and end of the code block.
"""

class BaselineLLMSystem(System):
    """
    A baseline system that uses a large language model (LLM) to process datasets and serve queries.
    """

    def __init__(self, model: str, name="baseline", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.dataset_directory = None  # TODO(user): Update me
        self.model = model
        self.llm = Generator(model, verbose=self.verbose)

        # Initialize directories for output and intermediate files
        if variance := kwargs.get('variance'):
            self.variance = variance
        else: self.variance = "one_shot" # Default variance
        # variance has to be one_shot or few_shot
        assert self.variance in ["one_shot", "few_shot"], f"Invalid variance: {self.variance}. Must be one_shot or few_shot."
        
        if verbose := kwargs.get('verbose'):
            self.verbose = verbose
        else: self.verbose = False # Default verbosity
        self.debug = False
        # Set the output directory
        if output_dir := kwargs.get('output_dir'):
            self.output_dir = output_dir
        else: self.output_dir = os.path.join(os.getcwd(), 'testresults') # Default output directory
        self.question_output_dir = None # to be set in run()
        self.question_intermediate_dir = None # to be set in run()
    
    @typechecked
    def run_one_shot(self, query:str, query_id:str) -> Dict[str, str | Dict | List]:
        """
        This function demonstrates a simple one-shot LLM approach to solve the LLMDS benchmark.
        """
        self._init_output_dir(query_id)
            
        # Generate the prompt
        prompt = self.generate_prompt(query)
        if self.debug:
            print(f"{self.name}: Prompt:", prompt)

        # Get the model's response
        messages=[
            {"role": "system", "content": "You are an experienced data scientist."},
            {"role": "user", "content": prompt}
        ]
        response = call_gpt(messages)
        if self.debug:
            print(f"{self.name}: Response:", response)

        # Process the response
        json_fp, code_fp = self.extract_response(response, try_number=0)

        # Execute the code (if necessary)
        output_fp, error_fp = self.execute_code(code_fp, try_number=0)
        # print("Execution Result:", result)

        # Fill in JSON response with the execution result
        answer = self.process_response(json_fp, output_fp, error_fp)

        return answer
    
    @typechecked
    def run_few_shot(self, query: str, query_id: str) -> Dict[str, str | Dict | List]:
        """
        This function demonstrates a simple few-shot LLM approach to solve the LLMDS benchmark.
        """
        self._init_output_dir(query_id)

        # Generate the prompt
        prompt = self.generate_prompt(query)
        if self.debug:
            print(f"{self.name}: Prompt:", prompt)

        messages=[
            {"role": "system", "content": "You are an experienced data scientist."},
        ]

        answer = None

        for try_number in range(5):
            messages.append({"role": "user", "content": prompt})
            # Get the model's response
            response = call_gpt(messages)
            if self.debug:
                print(f"{self.name}: Response:", response)
            messages.append({"role": "assistant", "content": response})

            # Process the response
            if try_number == 0:
                json_fp, code_fp = self.extract_response(response, try_number)
            else: _, code_fp = self.extract_response(response, try_number)

            # Execute the code (if necessary)
            output_fp, error_fp = self.execute_code(code_fp, try_number)
            # print("Execution Result:", result)

            if os.path.getsize(error_fp) > 0:
                prompt = self.generate_error_handling_prompt(code_fp, error_fp)
            else:
                # Fill in JSON response with the execution result
                answer = self.process_response(json_fp, output_fp, error_fp=None)
                break
        
        if answer is None:
            answer = {"id": "main-task", "answer": "Pipeline not successful after 5 tries."}
        return answer
    
    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        """
        Process the dataset located in the specified directory.
        The dataset can contain files in various formats (e.g., PDF, CSV).
        """
        self.dataset_directory = dataset_directory
        self.dataset = {}
        # read the files
        for file in os.listdir(dataset_directory):
            if file.endswith(".csv"):
                try:
                    self.dataset[file] = pd.read_csv(os.path.join(dataset_directory, file), engine='python', on_bad_lines='warn')
                except UnicodeDecodeError:
                    self.dataset[file] = pd.read_csv(os.path.join(dataset_directory, file), engine='python', on_bad_lines='warn', encoding='latin-1')
    
    @typechecked
    def serve_query(self, query: str, query_id:str="default_name-0") -> Dict:
        """
        Serve a query using the LLM.
        The query should be in natural language, and the response can be in either natural language or JSON format.
        """
        # TODO: Implement the logic to handle different types of queries
        if self.variance == "one_shot":
            results = self.run_one_shot(query, query_id)
        elif self.variance == "few_shot":
            results = self.run_few_shot(query, query_id)
        return results

def main():
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    questions = [
        {
            "id": "environment-easy-1",
            "query": "In 2018, how many bacterial exceedances were observed in freshwater beaches?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
        {
            "id": "environment-medium-1",
            "query": "For the freshwater beaches, what is the difference between the percentage exceedance rate in 2023 and the historic average from 2002 to 2022?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        },
        {
            "id": "environment-hard-1",
            "query": "For the marine beaches, what is the Pearson product-moment correlation from 2002 to 2023 between the rainfall amount in inches during the months June, July, August, and September and the percentage exceedance rate?",
            "dataset_directory": os.path.join(current_dir, "../data/environment"),
        }
    ]
    
    # Process each question
    out_dir = os.path.join(current_dir, "../testresults/run2")
    baseline_llm = BaselineLLMSystem(model="gpt-4o", output_dir=out_dir, variance="few_shot", verbose=True)
    # Process the dataset
    baseline_llm.process_dataset(questions[0]["dataset_directory"])
    for question in questions:
        print(f"Processing question: {question['id']}")
        # For debugging purposes, also input question.id
        response = baseline_llm.serve_query(question["query"], question["id"])

if __name__ == "__main__":
    main()