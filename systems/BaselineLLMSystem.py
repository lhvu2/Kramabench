from systems.generator_util import Generator, pdf_to_text
from benchmark.benchmark_api import System
import sys
sys.path.append('./')
from utils.baseline_utils import * 

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
        self.output_dir = None # to be set in run()
        self.question_output_dir = None # to be set in run()
        self.question_intermediate_dir = None # to be set in run()
    
    def _init_output_dir(self, output_dir:str, query_id:str) -> None:
        """
        Initialize the output directory for the question.
        :param question: Question object
        """
        self.output_dir = output_dir
        question_output_dir = os.path.join(self.output_dir, query_id)
        if not os.path.exists(question_output_dir):
            os.makedirs(question_output_dir)
        self.question_output_dir = question_output_dir
        self.question_intermediate_dir = os.path.join(self.question_output_dir, '_intermediate')
        if not os.path.exists(self.question_intermediate_dir):
            os.makedirs(self.question_intermediate_dir)

    def generate_prompt(self, query:str) -> str:
        """
        Generate a prompt for the LLM based on the question.
        :param query: str
        :return: Prompt string
        """
        # Generate the RAG plan
        # TODO: use process_dataset() to get the data
        data = self.get_input_data()
        file_names = list(self.dataset_directory.keys()) # get the file names from the dataset directory
        file_paths = [os.path.join(self.dataset_directory, file) for file in file_names] # get the file paths

        prompt = f"""
        Your task is to answer the following question based on the provided data sources.
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
        {{
            "id": "main-task",
            "query": "{query}",
            "data_sources": ["file1.csv", "file2.csv"]
            "subtasks": [{
                "id": "subtask-1",
                "query": "What is the exceedance rate in 2022?",
                "data_sources": ["water-body-testing-2022.csv"]
            },{
                "id": "subtask-2",
                "query": "What is the column name for the exceedance rate?",
                "data_sources": ["water-body-testing-2022.csv"]
            }]
        }}
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

        # Save the prompt to a txt file
        prompt_fp = os.path.join(self.question_output_dir, f"prompt.txt")
        with open(prompt_fp, 'w') as f:
            f.write(prompt)
        print(f"Prompt saved to {prompt_fp}")
        return prompt
    
    def extract_response(self, response, try_number:int):
        """
        Process the LLM response.
        :param response: LLM response string
        :return: Processed response
        """
        json_fp = ""
        code_fp = ""
        # Save the full response to a txt file
        response_fp = os.path.join(self.question_output_dir, f"initial_response.txt")
        with open(response_fp, 'w') as f:
            f.write(response)
        print(f"Response saved to {response_fp}")

        # Assume the step-by-step plan is fixed after the first try
        if try_number == 0:
            # Extract the JSON array from the response
            json_response = extract_code(response, pattern=r'```json(.*?)```')
            #print("Extracted JSON:", json_response)
            # Save the JSON response to a file
            json_fp = os.path.join(self.question_output_dir, f"answer.json")
            with open(json_fp, 'w') as f:
                f.write(json_response)
            print(f"JSON response saved to {json_fp}")
        
        # Extract the code from the response
        code = extract_code(response, pattern=r'```python(.*?)```')
        #print("Extracted Code:", code)
        # Save the code to a file
        code_fp = os.path.join(self.question_output_dir, '_intermediate', f"pipeline-{try_number}.py") #{question.id}-{try_number}
        with open(code_fp, 'w') as f:
            f.write(code)
        print(f"Code saved to {code_fp}")

        return json_fp, code_fp
    
    def execute_code(self, code_fp, try_number:int):
        """
        Execute the code in the file and save the output.
        :param code_fp: Path to the code file
        :return: Execution result
        """
        # Execute the code and save error messages
        result = subprocess.run(
            ['python', code_fp],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Save the printed output of the code execution
        output_fp = os.path.join(self.question_intermediate_dir, f"pipeline-{try_number}_out.json")
        with open(output_fp, 'w') as f:
            # Clean NaN values to null for strict JSON compliance
            stdout = clean_nan(result.stdout)
            f.write(stdout)

        # Save only compile/runtime errors
        error_fp = os.path.join(self.question_intermediate_dir, f"errors-{try_number}.txt")
        with open(error_fp, 'w') as f:
            f.write(result.stderr)

        return output_fp, error_fp
    
    def process_response(self, json_fp, output_fp):
        """
        Process the response and fill in the JSON response with the execution result.
        :param question: Question object
        :param json_fp: Path to the JSON file
        :param output_fp: Path to the output file
        """
        # Load the JSON response
        try: 
            with open(json_fp, 'r') as f:
                response = json.load(f)
        except json.JSONDecodeError as e:
            print(f"** ERRORS ** decoding response JSON: {e}")
            return

        # Load the output file
        try: 
            with open(output_fp, 'r') as f:
                output = json.load(f)
        except json.JSONDecodeError as e:
            print(f"** ERRORS ** decoding output JSON: {e}")
            return

        # Fill in the JSON response with the execution result
        for step in response:
            id = step['id']
            if id in output:
                step['answer'] = output[id]
            else:
                step['answer'] = "No answer found."

        # Save the updated JSON response
        with open(json_fp, 'w') as f:
            # Clean NaN values to null for strict JSON compliance
            response = clean_nan(response)
            json.dump(response, f, indent=4)
        print(f"Updated JSON response saved to {json_fp}")
    
    def run_one_shot(self, query:str, output_dir:str) -> str:
        """
        This function demonstrates a simple one-shot LLM approach to solve the LLMDS benchmark.
        """
        self._init_output_dir(os.path.join(output_dir, 'one_shot'), question)
            
        # Generate the prompt
        prompt = self.generate_prompt(question)
        print("Prompt:", prompt)

        # Get the model's response
        messages=[
            {"role": "system", "content": "You are an experienced data scientist."},
            {"role": "user", "content": prompt}
        ]
        response = call_gpt(messages)
        print("Response:", response)

        # Process the response
        json_fp, code_fp = self.extract_response(response, try_number=0)

        # Execute the code (if necessary)
        output_fp, error_fp = self.execute_code(code_fp, try_number=0)
        # print("Execution Result:", result)

        # Check if errors were generated
        if os.path.getsize(error_fp) > 0:
            # TODO: Handle the error case for the few-shot LLM
            # For now, just print the error message
            print(f"** ERRORS ** found in {error_fp}. Skipping JSON update.")
        else:
            # Fill in JSON response with the execution result
            self.process_response(json_fp, output_fp)

        return response
    
    def run_few_shot(self, query:str, output_dir:str) -> str:
        """
        This function demonstrates a simple few-shot LLM approach to solve the LLMDS benchmark.
        """
        self._init_output_dir(os.path.join(output_dir, 'few_shot'), question)

        # Generate the prompt
        prompt = self.generate_prompt(question)
        print("Prompt:", prompt)

        messages=[
            {"role": "system", "content": "You are an experienced data scientist."},
        ]

        for try_number in range(5):
            messages.append({"role": "user", "content": prompt})
            # Get the model's response
            response = call_gpt(messages)
            print("Response:", response)
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
                self.process_response(json_fp, output_fp)
                break

        return response
    
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
                self.dataset[file] = pd.read_csv(os.path.join(dataset_directory, file))

    def serve_query(self, query: str) -> dict | str:
        """
        Serve a query using the LLM.
        The query should be in natural language, and the response can be in either natural language or JSON format.
        """
        # TODO: Implement the logic to handle different types of queries
        results = self.llm.generate(query)
        return results