from omegaprm import OmegaPRM, LanguageModel, ProcessRewardModel
from openai import AsyncOpenAI
import json
from typing import List
from vote_utils import (
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_VOTE,
    PRM_LAST_MAX,
    AGG_FN_MAP,
)
from envs.base_env import INVALID_ANS
import numpy as np
import asyncio
import aiofiles
from tqdm import tqdm
import os
import argparse
from math_verify import parse, verify
from evaluate import load
math = load("competition_math")


def parse_reasoning_paths_and_mc(tree_data):
    """
    Parse all root-to-leaf paths and their corresponding MC values from the tree.
    A reasoning path represents a complete solution path from the root to a leaf node.

    Parameters:
    - tree_data (dict): The nested dictionary representing the tree structure.
                       Each node has 'text', 'mc_value' and 'children' fields.

    Returns:
    - Tuple[List[List[str]], List[List[float]]]: Two lists:
        1. List of reasoning paths, where each path is a list of node texts from root to leaf
        2. List of MC value sequences, where each sequence contains MC values along a path
    """
    reasoning_paths = []  # List to store complete root-to-leaf solution paths
    mc_values_list = []  # List to store MC values along each path

    def traverse(node, current_path, current_mc_values):
        # Add current node's text and MC value to the current path
        current_path.append(node['text'])
        current_mc_values.append(node['mc_value'])

        if len(node['children']) == 0:
            # At a leaf node - we've found a complete reasoning path
            reasoning_paths.append(current_path.copy())
            mc_values_list.append(current_mc_values.copy())
        else:
            # Continue traversing down each child branch
            for child in node['children']:
                traverse(child, current_path, current_mc_values)

        # Backtrack: remove current node as we go back up
        current_path.pop()
        current_mc_values.pop()

    # Start traversal from rootbb
    traverse(tree_data, [], [])
    return reasoning_paths, mc_values_list


def judge_ans(
        problem_str: str,
        extracted_groundtruth: str,
        output_list: List[str],
        v_list: List[float],
        aggration_mode: str,
        extract_answer_fn,
        judge_correct_fn,
        normalize=False,
    ):
        ans_list = [extract_answer_fn(txt)[-1] for txt in output_list]
        valid_ans_list, valid_v_list = [], []
        for i, ans in enumerate(ans_list):
            if ans != INVALID_ANS:
                valid_ans_list.append(ans)
                valid_v_list.append(v_list[i])
        if len(valid_ans_list) == 0:
            return 0

        if "orm" in aggration_mode and normalize:
            # score_normalization: this is only necessary for [-1, 1] values
            valid_v_list = np.array(valid_v_list)
            valid_v_list -= valid_v_list.min()
            valid_v_list /= valid_v_list.max() + 1e-3
            valid_v_list = valid_v_list.tolist()
        aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)
        print("aggregated_ans: ", aggregated_ans)
        print("extracted_groundtruth: ", extracted_groundtruth)
        return (
            1 if math.compute(references=[extracted_groundtruth], predictions=[aggregated_ans])["accuracy"] > 0.99 else 0
        )


def evaluate_with_aggregation(problem_str, extracted_groundtruth, output_list, v_list, extract_answer_fn, judge_correct_fn):
    results = {}
    for aggration_mode, agg_fn in AGG_FN_MAP.items():
        result = judge_ans(
            problem_str=problem_str,
            extracted_groundtruth=extracted_groundtruth,
            output_list=output_list,
            v_list=v_list,
            aggration_mode=aggration_mode,
            extract_answer_fn=extract_answer_fn,
            judge_correct_fn=judge_correct_fn
        )
        results[aggration_mode] = result
    return results


async def save_results_async(output_file: str, data: dict):
    async with aiofiles.open(output_file, 'a') as f:
        await f.write(json.dumps(data) + '\n')

async def evaluate_single_problem(
    prob: dict,
    LM: LanguageModel,
    reward_model: ProcessRewardModel,
    sem: asyncio.Semaphore
) -> dict:
    async with sem:
        try:
            print("Evaluating problem: {}".format(prob["question"]))
            omega_prm = OmegaPRM(
                LM=LM,
                reward_model=reward_model,
                c_puct=0.125,
                alpha=0.5,
                beta=0.9,
                L=500,
                k=8,
                N=10,
                rollout_budget=16,
                save_data_tree=True
            )
            collected_data = await omega_prm.run(prob["question"], prob["expected_answer"])
            
            # Parse reasoning paths and mc_values from the collected data
            reasoning_paths, mc_values_list = parse_reasoning_paths_and_mc(collected_data)
            reasoning_path_strs = []
            for reasoning_path in reasoning_paths:
                reasoning_path_str = " ".join(reasoning_path)
                reasoning_path_strs.append(reasoning_path_str)
            
            print("------------------------------------------------------------")

            evaluation_results = evaluate_with_aggregation(
                problem_str=prob["question"],
                extracted_groundtruth=prob["expected_answer"],
                output_list=reasoning_path_strs,
                v_list=mc_values_list,
                extract_answer_fn=parse,
                judge_correct_fn=verify
            )
            
            result = {
                "question": prob["question"],
                "expected_answer": prob["expected_answer"],
                "reasoning_paths": reasoning_path_strs,
                "mc_values": mc_values_list,
                "evaluation_results": evaluation_results
            }
            print("expected_answer: ", prob["expected_answer"])
            print("Evaluation results: ", evaluation_results)
            return result
        except Exception as e:
            print(f"Error in evaluate_single_problem: {str(e)}")
            return None

async def main(debug: bool = False, resume: bool = False):
    # Initialize the AsyncOpenAI client for the Language Model
    lm_client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )
    
    # Initialize the AsyncOpenAI client for the ProcessRewardModel
    reward_client = AsyncOpenAI(
        base_url="http://localhost:8001/v1",
        api_key="token-abc123"
    )

    # Instantiate LanguageModel and ProcessRewardModel with separate clients
    LM = LanguageModel(
        client=lm_client,
        max_new_tokens=8192,
        temperature=0.6,
        top_p=0.95,
        model_name="DeepSeek-R1-Distill-Qwen-14B"
    )
    
    reward_model = ProcessRewardModel(
        client=reward_client,
        model="deepseek-14b-prm-filtered-balance-full",
        temperature=0.0,
        max_tokens=1
    )
    
    # Load problems from test500.jsonl
    problems = []
    with open('./test500.jsonl', 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems.append({
                'question': problem['problem'],
                'expected_answer': problem['answer']
            })
    
    # If debug flag is active, only evaluate the first 50 problems
    if debug:
        problems = problems[:50]
        print("DEBUG MODE: processing only the first 50 problems.")

    # If resume flag is active, skip already evaluated problems
    if resume:
        if os.path.exists("evaluation_results.jsonl"):
            # Deduplicate the evaluation_results.jsonl file
            dedup = {}
            with open("evaluation_results.jsonl", 'r') as res_file:
                for line in res_file:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            question = rec.get("question")
                            if question is not None:
                                dedup[question] = rec
                        except Exception as e:
                            continue

            # Write deduplicated results back to the file
            with open("evaluation_results.jsonl", 'w') as res_file:
                for rec in dedup.values():
                    res_file.write(json.dumps(rec) + "\n")

            evaluated_questions = set(dedup.keys())
            original_count = len(problems)
            problems = [p for p in problems if p["question"] not in evaluated_questions]
            skipped = original_count - len(problems)
            print(f"Resuming evaluation: Skipping {skipped} already evaluated problems.")
        else:
            print("No previous evaluation results found. Starting from scratch.")

    # Create a semaphore to limit concurrent tasks
    sem = asyncio.Semaphore(30)  # Adjust the number based on your needs
    
    # Create tasks for each problem
    tasks = [
        asyncio.create_task(evaluate_single_problem(prob, LM, reward_model, sem))
        for prob in problems
    ]
    
    results = []
    # Use as_completed to update progress with tqdm
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Processing problems'):
        result = await future
        if result is not None:
            results.append(result)
            # Save result immediately
            await save_results_async("evaluation_results.jsonl", result)

    if results:
        agg_rates = {}
        for result in results:
            for mode, flag in result["evaluation_results"].items():
                agg_rates[mode] = agg_rates.get(mode, 0) + flag
        print("Final Success Rate for each aggregation mode:")
        for mode, total in agg_rates.items():
            print(f"{mode}: {total/len(results) * 100:.2f}%")

    print(f"Evaluation complete. Processed {len(results)} problems successfully.")
    print(f"Results saved to evaluation_results.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (only evaluate the first 50 problems)")
    parser.add_argument("--resume", action="store_true", help="Resume evaluation by skipping already evaluated problems")
    args = parser.parse_args()
    asyncio.run(main(debug=args.debug, resume=args.resume))

