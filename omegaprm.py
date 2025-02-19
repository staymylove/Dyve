import heapq
import math
import random
import re
import json
from typing import List, Tuple, Dict, Any, Optional
import itertools
from transformers import AutoTokenizer
import asyncio  # New import added for async handling
from openai import AsyncOpenAI   # Using AsyncOpenAI as client
import numpy as np


# Helper function to separate reasoning steps
def separate_steps(steps: List[str], mode: str = 'join') -> Any:
    delimiter = "\n\n"
    if mode == 'join':
        if not isinstance(steps, list):
            raise TypeError("For 'join' mode, 'steps' must be a list of strings.")
        return delimiter.join(steps)
    elif mode == 'split':
        if not isinstance(steps, str):
            raise TypeError("For 'split' mode, 'steps' must be a string.")
        return steps.split(delimiter)
    else:
        raise ValueError("Mode should be either 'join' or 'split'.")
    

# def judge_ans(
#         problem_str: str,
#         extracted_groundtruth: str,
#         output_list: List[str],
#         v_list: List[float],
#         aggration_mode: str,
#         extract_answer_fn,
#         judge_correct_fn,
#         normalize=False,
#     ):
#         ans_list = [extract_answer_fn(txt) for txt in output_list]
#         valid_ans_list, valid_v_list = [], []
#         for i, ans in enumerate(ans_list):
#             if ans != INVALID_ANS:
#                 valid_ans_list.append(ans)
#                 valid_v_list.append(v_list[i])
#         if len(valid_ans_list) == 0:
#             return 0

#         if "orm" in aggration_mode and normalize:
#             # score_normalization: this is only necessary for [-1, 1] values
#             valid_v_list = np.array(valid_v_list)
#             valid_v_list -= valid_v_list.min()
#             valid_v_list /= valid_v_list.max() + 1e-3
#             valid_v_list = valid_v_list.tolist()
#         aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)

#         return (
#             1 if judge_correct_fn(problem_str, extracted_groundtruth, aggregated_ans) else 0
#         )





# Helper function to check correctness of a generated response
def check_correctness(generated_response: str, expected_answer: str) -> bool:
    # sentences = re.split(
    #     r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', generated_response.strip()
    # )
    # last_sentence = sentences[-1] if sentences else ''
    # return expected_answer.strip() in last_sentence.strip()
    extract_answer_fn = MATH.extract_answer
    judge_correct_fn = MATH.judge_correct
    answer = extract_answer_fn(generated_response)
    return (
            1 if judge_correct_fn("", expected_answer, answer) else 0
        )


class ProcessRewardModel:
    """
    ProcessRewardModel encapsulates the reward inference process.
    
    It utilizes a chat-based reward model (e.g., 'Llama3.1-8B-PRM-Mistral-Data')
    to evaluate a sequence of reasoning steps. It iteratively sends messages to the model,
    checking that each step receives a positive judgement (i.e., its completion starts with '+').
    """

    def __init__(self, client, model="Llama3.1-8B-PRM-Mistral-Data", temperature=0.0, max_tokens=1):
        """
        Initialize the ProcessRewardModel.

        Parameters:
            client: The chat client instance that provides a `chat.completions.create` method.
            model (str): The model name to be used for generating reward completions.
            temperature (float): Sampling temperature. Default is 0.0 for deterministic outcomes.
            max_tokens (int): Maximum tokens to generate in the reward inference. Default is 1.
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def evaluate(self, problem: str, steps: list, output_type: str = 'bool') -> bool:
        """
        Synchronously evaluate the process reward using asynchronous API calls.
        This method wraps the asynchronous _async_evaluate call.

        Parameters:
            problem (str): The problem or question statement.
            steps (List[str]): A list of reasoning steps.

        Returns:
            bool: True if all steps are positively judged, False otherwise.
        """
        return asyncio.run(self._async_evaluate(problem, steps, output_type))

    async def _async_evaluate(self, problem: str, steps: list, output_type: str = 'bool') -> bool:
        messages = []

        # # Merge every 5 steps into 1 step to reduce evaluation time
        # if 'deepseek' in self.model.lower():
        #     merged_steps = []
        #     current_merge = []
        #     for step in steps:
        #         current_merge.append(step)
        #         if len(current_merge) == 6:
        #             merged_steps.append("\n\n".join(current_merge))
        #             current_merge = []
        #     if current_merge:  # Add any remaining steps
        #         merged_steps.append("\n\n".join(current_merge))

        # steps = merged_steps
        for sdx, step in enumerate(steps):
            if sdx == 0:
                messages.append({
                    'role': 'user',
                    'content': f"{problem}\n\n{step}"
                })
            else:
                messages.append({
                    'role': 'user',
                    'content': step
                })

            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            response = completion.choices[0].message.content.strip().lower()
            if not response.startswith('+'):
                if output_type == 'bool':
                    return False
                else:
                    return [1.0 if i < sdx else 0.0 for i in range(len(steps))]
            messages.append({'role': 'assistant', 'content': '+'})
        
        if output_type == 'bool':
            return True
        else:
            return [1.0] * len(steps)
        
    async def _async_evaluate_system2(self, problem: str, steps: list, output_type: str = 'bool') -> bool:
        messages = []

        # Merge every 5 steps into 1 step to reduce evaluation time

        # if 'deepseek' in self.model.lower():
        #     merged_steps = []
        #     current_merge = []
        #     for step in steps:
        #         current_merge.append(step)
        #         if len(current_merge) == 6:
        #             merged_steps.append("\n\n".join(current_merge))
        #             current_merge = []
        #     if current_merge:  # Add any remaining steps
        #         merged_steps.append("\n\n".join(current_merge))

        # steps = merged_steps
        for sdx, step in enumerate(steps):
            
            if sdx == 0:
                messages.append({
                    'role': 'user', 
                    'content': f"Problem: {problem}\n\nStep: {step}\n\nIs this step correct? You must answer with '+' for correct or '-' for incorrect in the end of your response."
                })
            else:
                messages.append({
                    'role': 'user', 
                    'content': f"Step: {step}\n\nIs this step correct? You must answer with '+' for correct or -' for incorrect in the end of your response."
                })
            
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                n=1,
                temperature=self.temperature,
                max_tokens=8192,
            )
            response = completion.choices[0].message.content

            # print("DyVer Verification:", response)
            
            # New negative checking logic
            content = response.strip().lower()
            last_words = ' '.join(content.split()[-3:])  # Last 3 words
            
            judgment = any(
                '+' in part and '-' not in part
                for part in (
                    content[-5:], 
                    last_words,
                )
            )
            
            if not judgment:
                return [1.0 if i < sdx else 0.0 for i in range(len(steps))]
            messages.append({'role': 'assistant', 'content': '<think>\n\n</think> +'})
        return [1.0] * len(steps)



class LanguageModel:
    def __init__(self, client, model_name="/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct",
                 max_new_tokens=512, temperature=0.7, top_p=0.9):
        """
         Initialize the LanguageModel for async OpenAI calls.
         Removed the LLMService dependency and using async calls via openai.
         
         Parameters:
         - client: An instance of AsyncOpenAI passed externally.
         - model_name (str): API model name to use.
         - max_new_tokens (int): Maximum tokens for generation.
         - temperature (float): Sampling temperature.
         - top_p (float): Nucleus sampling probability.
         """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.default_prompt = (
            "Please complete the answer for the question based on the given steps without generating existing steps again, "
            "and separate your following steps using \n\n.\n\n"
        )
        # Retain tokenizer for chat template operations elsewhere.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(f"/root/{model_name}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            self.tokenizer = None
        # Use the external AsyncOpenAI client.
        self.async_client = client

    async def generate_rollout(self, state_prefix: str, num_copies: int) -> List[str]:
        """
        Asynchronously generate responses using OpenAI's ChatCompletion API.
        
        Parameters:
        - state_prefix (str): The current solution prefix.
        - num_copies (int): The number of response copies to generate.
        
        Returns:
        - List[str]: A list of generated responses.
        """
        response = await self.async_client.completions.create(
            model=self.model_name,
            prompt=state_prefix,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=num_copies,
        )
        return [choice.text for choice in response.choices]

    def update_prompt(self, new_prompt: str):
        """
        Update the default prompt if necessary.
        
        Parameters:
        - new_prompt (str): The new prompt template.
        """
        self.default_prompt = new_prompt

    def evaluate_correctness(self, response: str, expected_answer: str) -> bool:
        """
        Check if the generated solution matches the expected answer.
        
        Parameters:
        - response (str): The complete generated response.
        - expected_answer (str): The expected answer to compare with.
        
        Returns:
        - bool: True if the expected answer is in the final part of the solution.
        """
        return check_correctness(response, expected_answer)


# Define the State class
class State:
    def __init__(self, solution_prefix: str, parent: Optional['State'] = None):
        self.solution_prefix = solution_prefix  # Solution prefix as a single string
        self.parent = parent  # Reference to the parent state
        self.N = 0  # Visit count (number of times selected)
        self.total_rollouts = 0  # Total number of rollouts generated from this state
        self.correct_rollouts = 0  # Number of correct rollouts
        self.MC: Optional[float] = None  # Monte Carlo estimation (c/k)
        self.Q: Dict[str, float] = {}  # Q(s, r): estimated value for each rollout
        self.R: List[str] = []  # Set of all rollouts from this state
        self.incorrect_rollouts: List[str] = []  # List of incorrect rollouts
        self.children: List['State'] = []  # List of child states

    def add_rollout(self, rollout: str):
        self.R.append(rollout)

    def add_incorrect_rollout(self, rollout: str):
        if rollout not in self.incorrect_rollouts:
            self.incorrect_rollouts.append(rollout)

    def get_full_solution(self) -> str:
        # Return the complete solution from the root to this state
        if self.parent:
            return self.parent.get_full_solution() + '\n\n' + self.solution_prefix
        else:
            return self.solution_prefix

    def get_new_text(self) -> str:
        """
        Return the new text added at this node compared to the parent.
        """
        if self.parent:
            parent_text = self.parent.solution_prefix
            new_text = self.solution_prefix[len(parent_text):].strip()
            return new_text
        else:
            # Root node (the question)
            return self.solution_prefix.strip()

    def get_text_with_labels(self) -> Dict[str, Any]:
        """
        Return a nested dictionary where each node contains:
        - 'text': The new text at this node.
        - 'mc_value': The MC value at this node.
        - 'children': A list of child nodes with the same structure.
        """
        data = {
            'text': self.get_new_text(),
            'mc_value': self.MC,
            'children': [child.get_text_with_labels() for child in self.children]
        }
        return data


# Define the Search Tree class
class SearchTree:
    def __init__(self):
        self.root: Optional[State] = None
        self.nodes: List[State] = []  # List of all states

    def add_state(self, state: State):
        self.nodes.append(state)

# Define the Candidate Pool as a priority queue with update capability
class CandidatePool:
    def __init__(self):
        self.heap: List[Tuple[float, int]] = []  # Heap of (-priority, unique_id)
        self.entry_finder: Dict[int, Tuple[float, int]] = {}  # Maps unique_id to (-priority, unique_id)
        self.counter = itertools.count()  # Unique sequence count
        self.id_to_rollout: Dict[int, Tuple[State, str]] = {}  # Maps unique_id to (state, rollout)
        self.latest_id_per_rollout: Dict[Tuple[int, str], int] = {}  # Maps (state_id, rollout) to unique_id

    def add_or_update(self, state: State, rollout: str, priority: float):
        """
        Add a new rollout or update the priority of an existing rollout.

        Parameters:
        - state (State): The state associated with the rollout.
        - rollout (str): The rollout string.
        - priority (float): The new priority score.
        """
        state_id = id(state)  # Unique identifier for the state object
        rollout_key = (state_id, rollout)

        # Check if the rollout already exists in the pool
        if rollout_key in self.latest_id_per_rollout:
            # Previous unique_id exists; it is now outdated
            old_unique_id = self.latest_id_per_rollout[rollout_key]
            # Mark the old entry as invalid by removing it from entry_finder
            if old_unique_id in self.entry_finder:
                del self.entry_finder[old_unique_id]
                del self.id_to_rollout[old_unique_id]

        # Assign a new unique_id for the updated rollout
        unique_id = next(self.counter)
        self.latest_id_per_rollout[rollout_key] = unique_id

        # Add the new entry to the heap and mappings
        heapq.heappush(self.heap, (-priority, unique_id))  # Max-heap using negative priority
        self.entry_finder[unique_id] = (-priority, unique_id)
        self.id_to_rollout[unique_id] = (state, rollout)

    def pop(self) -> Tuple[Optional[State], Optional[str]]:
        """
        Pop the rollout with the highest priority.

        Returns:
        - Tuple[Optional[State], Optional[str]]: The state and rollout string, or (None, None) if empty.
        """
        while self.heap:
            neg_priority, unique_id = heapq.heappop(self.heap)
            # Check if this unique_id is still valid
            if unique_id in self.entry_finder:
                # Valid entry
                state, rollout = self.id_to_rollout.pop(unique_id)
                del self.entry_finder[unique_id]
                # Remove from latest_id_per_rollout
                state_id = id(state)
                rollout_key = (state_id, rollout)
                if self.latest_id_per_rollout.get(rollout_key) == unique_id:
                    del self.latest_id_per_rollout[rollout_key]
                return state, rollout
            # Else, outdated entry; skip
        return None, None

    def is_empty(self) -> bool:
        return not self.entry_finder

# Define the OmegaPRM algorithm
class OmegaPRM:
    def __init__(self, LM: LanguageModel, reward_model, c_puct: float, alpha: float, beta: float, L: int, k: int, N: int,
                 rollout_budget: int, save_data_tree: bool):
        """
        Initialize the OmegaPRM algorithm.

        Parameters:
            LM (LanguageModel): The language model instance.
            reward_model: An instance of ProcessRewardModel to evaluate solution correctness.
            c_puct (float): Exploration constant.
            alpha (float): Weight for MC(s).
            beta (float): Length penalty.
            L (int): Maximum solution length.
            k (int): Number of rollouts for Monte Carlo estimation.
            N (int): Maximum search count.
            rollout_budget (int): Total rollout budget.
            save_data_tree (bool): Whether to save and return the data tree.
        """
        self.LM = LM
        self.reward_model = reward_model
        self.expected_answer = None
        self.c_puct = c_puct
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.k = k
        self.N = N
        self.rollout_budget = rollout_budget
        self.save_data_tree = save_data_tree

        self.T = SearchTree()
        self.C = CandidatePool()

        self.n = 0
        self.total_rollouts = 0

    def reset(self):
        """Reset internal state variables to prepare for a fresh run."""
        self.expected_answer = None
        self.T = SearchTree()  # Reset search tree
        self.C = CandidatePool()  # Reset candidate pool
        self.n = 0
        self.total_rollouts = 0
        self.collected_data = []  # Clear collected data

    async def monte_carlo_estimation(self, state: State):
        """
        Perform Monte Carlo estimation for state by generating k rollouts
        and computing MC(s) = c / k, where c is the number of correct rollouts.
        """
        c = 0  # Correct rollouts count
        incorrect_rollouts = []
        correct_rollouts = []
        batct_rollouts = await self.LM.generate_rollout(state.solution_prefix, self.k)

        # Increment visit count of selected state
        state.N += 1

        for i, rollout in enumerate(batct_rollouts):
            # Increment number of total rollouts
            self.total_rollouts += 1

            # Generate rollout r_i
            state.add_rollout(rollout)

            # Evaluate correctness of final answer in rollout using the reward model.
            full_solution = (state.solution_prefix + '\n\n' + rollout).strip() if state.solution_prefix else rollout
            steps = separate_steps(full_solution, mode='split')
            # If all steps receive a positive judgment, evaluate returns -1.
            is_correct = await self.reward_model._async_evaluate(self.problem, steps)

            if is_correct:
                c += 1
                correct_rollouts.append(rollout)
            else:
                incorrect_rollouts.append(rollout)
                state.add_incorrect_rollout(rollout)  # Track incorrect rollouts

        # Update total rollouts and correct rollouts
        state.total_rollouts += self.k
        state.correct_rollouts += c
        state.MC = state.correct_rollouts / state.total_rollouts if state.total_rollouts > 0 else 0

        if state.MC == 1.0:
            # Add all correct rollouts to the tree as new states
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
        elif state.MC == 0.0:
            # State is incorrect; no further action
            for rollout in incorrect_rollouts:
                self.add_incorrect_rollout_to_tree(state, rollout)
            return
        else:
            # 0 < MC(s) < 1.0
            # Add correct rollouts to the tree
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
            # Add incorrect rollouts to candidate pool with updated priorities
            for rollout in incorrect_rollouts:
                priority = self.compute_selection_score(state, rollout)
                self.C.add_or_update(state, rollout, priority)

    async def run(self, question: str, answer: str) -> List:
        """
        Execute the OmegaPRM algorithm.

        Parameters:
        - question (str): The question to generate solutions for.

        Returns:
        - Collected data: List of dictionaries.
        """
        self.reset()
        self.problem = question  # Store the original question for reward evaluation

        print(f"Running OmegaPRM for question: '{question}'\n")
        # Initialization
        if self.LM.tokenizer is not None:
            question_tamplated = self.LM.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_special_tokens=False,
                add_generation_prompt=True
            )
        else:
            question_tamplated = question
        initial_state = State(solution_prefix=question_tamplated, parent=None)
        self.expected_answer = answer
        self.T.root = initial_state
        self.T.add_state(initial_state)
        self.n = 0

        # Monte Carlo Estimation for initial_state
        await self.monte_carlo_estimation(initial_state)

        # Main loop
        while self.n < self.N and self.total_rollouts < self.rollout_budget and not self.C.is_empty():
            # Selection Phase
            selected_state, selected_rollout = self.selection_phase()
            if selected_state is None or selected_rollout is None:
                break

            await self.expansion_phase_binary_search(selected_state, selected_rollout)

            # Maintenance Phase
            self.maintenance_phase(selected_state)

            # Increment search count
            self.n += 1

        if self.save_data_tree:
            data = self.collect_tree_structure()
        else:
            data = self.collect_solution_prefixes()
        return data

    def compute_Q(self, state: State, rollout: str) -> float:
        """
        Compute Q(s, r) = alpha^{1 - MC(s)} * beta^{len(r)/L}, where len(r) is based on word count.
        """
        # Count words in the rollout
        word_count = len(rollout.split())
        length_penalty = word_count / self.L
        Q_value = (self.alpha ** (1 - state.MC)) * (self.beta ** length_penalty)
        return Q_value

    def compute_U(self, state: State) -> float:
        """
        Compute U(s) = c_puct * sqrt(sum_{s'} N(s')) / (1 + N(s))
        """
        N_total = sum(s.N for s in self.T.nodes)
        if N_total == 0:
            N_total = 1  # Prevent division by zero
        U_s = self.c_puct * (math.sqrt(N_total)) / (1 + state.N)
        return U_s

    def compute_selection_score(self, state: State, rollout: str) -> float:
        """
        Compute selection score: Score(s, r) = Q(s, r) + U(s)
        """
        Q_s_r = self.compute_Q(state, rollout)
        U_s = self.compute_U(state)
        score = Q_s_r + U_s
        return score

    def selection_phase(self) -> Tuple[Optional[State], Optional[str]]:
        """
        Select (state, rollout) with the highest score from candidate pool C.
        """
        selected_state, selected_rollout = self.C.pop()
        return selected_state, selected_rollout

    def add_correct_rollout_to_tree(self, parent_state: State, rollout: str):
        """
        Add the correct rollout to the tree as a child of parent_state.
        """
        new_solution_prefix = (parent_state.solution_prefix + '\n\n' + rollout).strip() if parent_state.solution_prefix else rollout
        new_state = State(solution_prefix=new_solution_prefix, parent=parent_state)
        new_state.MC = 1.0  # Since the rollout is correct
        new_state.total_rollouts = 0
        new_state.correct_rollouts = 0
        self.T.add_state(new_state)
        parent_state.children.append(new_state)  # Add to parent's children

    def add_incorrect_rollout_to_tree(self, parent_state: State, rollout: str):
        """
        Add the incorrect rollout to the tree as a child of parent_state.

        Parameters:
        - parent_state (State): The state from which the rollout was selected.
        - rollout (str): The incorrect rollout string.
        """
        new_solution_prefix = (parent_state.solution_prefix + '\n\n' + rollout).strip() if parent_state.solution_prefix else rollout
        new_state = State(solution_prefix=new_solution_prefix, parent=parent_state)
        new_state.MC = 0.0  # Since the rollout is incorrect
        new_state.total_rollouts = 0
        new_state.correct_rollouts = 0
        self.T.add_state(new_state)
        parent_state.children.append(new_state)  # Add to parent's children

    async def binary_search_incorrect_step(self, s_ast: State, steps: List[str], left: int, right: int):
        """
        Recursively perform binary search to find all incorrect steps in the rollout.
        """
        if left > right:
            return

        mid = (left + right) // 2
        new_steps = steps[left:mid + 1]
        if new_steps:
            prefix_solution = s_ast.solution_prefix + '\n\n' + separate_steps(new_steps, mode='join')
        else:
            prefix_solution = s_ast.solution_prefix
        # Create new state s_new
        s_new = State(solution_prefix=prefix_solution.strip(), parent=s_ast)
        self.T.add_state(s_new)
        s_ast.children.append(s_new)

        # Perform Monte Carlo estimation for s_new
        await self.monte_carlo_estimation(s_new)

        if s_new.MC == 0:
            # Found incorrect step; continue searching in the left half to find earlier incorrect steps
            await self.binary_search_incorrect_step(s_ast, steps, left, mid - 1)
        else:
            # Steps up to mid are correct; continue searching in the right half
            await self.binary_search_incorrect_step(s_new, steps, mid + 1, right)

    async def expansion_phase_binary_search(self, parent_state: State, rollout: str):
        """
        Expansion phase that adds the rollout as a new state and performs Monte Carlo estimation
        using Binary Search to efficiently find the correct rollout.
        """
        # Separate the rollout into individual steps
        steps = separate_steps(rollout, mode='split')

        # Perform binary search to find incorrect steps
        await self.binary_search_incorrect_step(parent_state, steps, 0, len(steps) - 1)

    def maintenance_phase(self, state: State):
        """
        Update statistics and candidate pool for all incorrect rollouts associated with the state.

        Parameters:
        - state (State): The state whose incorrect rollouts need to be updated.
        """

        # Iterate through all incorrect rollouts of the state
        for rollout in state.incorrect_rollouts:
            # Since we've already determined these rollouts are incorrect, no need to re-evaluate correctness

            priority = self.compute_selection_score(state, rollout)
            # Update the candidate pool with the new priority
            self.C.add_or_update(state, rollout, priority)
            # print(f"Updated Incorrect Rollout: '{rollout}' with new priority: {priority:.4f}")

        # print("Maintenance Phase Completed.\n")

    def collect_solution_prefixes(self) -> List[Dict[str, Any]]:
        """
        Collect all solution prefixes and their corresponding MC values from the search tree.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing solution prefixes and their MC values.
        """
        collected_data = []
        for node in self.T.nodes:
            solution_prefix = node.solution_prefix
            mc_value = node.MC
            collected_data.append({
                "solution_prefix": solution_prefix,
                "mc_value": mc_value
            })
        return collected_data

    def collect_tree_structure(self) -> Dict[str, Any]:
        """
        Collect the tree structure starting from the root.

        Returns:
            Dict[str, Any]: A nested dictionary representing the tree structure.
        """
        if self.T.root:
            tree_data = self.T.root.get_text_with_labels()
            return tree_data
        return {}


# Example usage
if __name__ == "__main__":
    # Initialize the Language Model's AsyncOpenAI client for LM.
    from openai import AsyncOpenAI
    lm_client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    
    LM = LanguageModel(
        client=lm_client,
        max_new_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        model_name="DeepSeek-R1-Distill-Qwen-14B"
    )

    # Define the question and expected answer
    question = "Melinda will roll two standard six-sided dice and make a two-digit number with the two numbers she rolls. For example, if she rolls a 6 and a 3, she can either form 36 or 63. What is the probability that she will be able to make an integer between 10 and 20, inclusive? Express your answer as a common fraction."
    expected_answer =  "\\frac{11}{36}"

    client = AsyncOpenAI(
        base_url="http://localhost:8001/v1",
        api_key="token-abc123",
    )  # This is a placeholder; ensure client supports sync chat.completions.create
    reward_model = ProcessRewardModel(client, model="deepseek-14b-prm-filtered-balance-full", temperature=0.0, max_tokens=1)

    # Initialize OmegaPRM with parameters and the reward model instance
    omega_prm = OmegaPRM(
        LM=LM,
        reward_model=reward_model,
        c_puct=0.125,
        alpha=0.5,
        beta=0.9,
        L=500,
        k=8,
        N=10,
        rollout_budget=20,
        save_data_tree=True,
    )

    # Run the OmegaPRM algorithm
    collected_data = asyncio.run(omega_prm.run(question, expected_answer))

    # Save the collected solutions to a JSON file
    with open("collected_solutions2.json", "w") as f:
        json.dump(collected_data, f, indent=4)


