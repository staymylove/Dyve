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


