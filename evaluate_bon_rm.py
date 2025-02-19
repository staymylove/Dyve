import json
import asyncio
import aiofiles
from tqdm import tqdm
import os
import argparse
from openai import AsyncOpenAI
from math_verify import parse
from evaluate import load
import numpy as np
from collections import Counter
from evaluate_prm import evaluate_with_aggregation

math = load("competition_math")


def separate_steps(text: str) -> list:
    """Separate the text into reasoning steps."""
    return text.split("\n\n")


async def generate_k_answers(client, question: str, model_name: str, k: int = 5) -> list:
    """Generate k answers for a question using the language model."""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": question}
            ],
            max_tokens=4096,
            temperature=0.6,
            top_p=0.95,
            n=k
        )
        return [choice.message.content.strip() for choice in response.choices]
    except Exception as e:
        print(f"Error in generate_k_answers: {str(e)}")
        return None


async def reward_model_vote(answers: list, prob: dict, reward_model, normalize: bool = True) -> tuple:
    """
    Use reward model to score each answer and evaluate using multiple aggregation methods.
    Returns (evaluation_results, extracted_answers, rm_scores)
    """
    extracted_answers = []
    valid_answers = []
    rm_scores = []

    for ans in answers:
        try:
            extracted = parse(ans)[-1]
            if extracted is not None:
                steps = separate_steps(ans)
                extracted_answers.append(extracted)
                valid_answers.append(ans)
        except:
            continue
    
    if not extracted_answers:
        return None, 0, [], []
    
    # Get reward model scores for each valid answer
    for ans in valid_answers:
        steps = separate_steps(ans)
        # Evaluate using reward model
        # v_list = await reward_model._async_evaluate(prob["question"], steps, output_type='list')
        v_list = await reward_model._async_evaluate_system2(prob["question"], steps, output_type='list')
        print("v_list: ", v_list)
        rm_scores.append(v_list)
    
    # Use evaluate_with_aggregation to get results for all aggregation methods
    evaluation_results = evaluate_with_aggregation(
        problem_str=prob["question"],
        extracted_groundtruth=prob["expected_answer"],
        output_list=valid_answers,
        v_list=rm_scores,
        extract_answer_fn=parse,
        judge_correct_fn=lambda p, g, a: math.compute(references=[g], predictions=[a])["accuracy"] > 0.99
    )
    
    return evaluation_results, extracted_answers, rm_scores


async def evaluate_single_problem(
    prob: dict,
    lm_client: AsyncOpenAI,
    reward_model,
    model_name: str,
    k: int,
    sem: asyncio.Semaphore
) -> dict:
    async with sem:
        try:
            print("Evaluating problem: {}".format(prob["question"]))
            
            # Generate k answers
            answers = await generate_k_answers(lm_client, prob["question"], model_name, k)
            if answers is None or len(answers) == 0:
                return None
            
            # Get reward model voting results
            evaluation_results, extracted_answers, rm_scores = await reward_model_vote(answers, prob, reward_model)
            
            print("------------------------------------------------------------")
            print("Question:", prob["question"])
            print("Expected answer:", prob["expected_answer"])
            print(f"Generated {len(answers)} answers")
            print("Extracted answers:", extracted_answers)
            print("RM scores:", rm_scores)
            print("Evaluation results:", evaluation_results)
            
            result = {
                "question": prob["question"],
                "expected_answer": prob["expected_answer"],
                "generated_answers": answers,
                "extracted_answers": extracted_answers,
                "rm_scores": rm_scores,
                "evaluation_results": evaluation_results
            }
            return result
        except Exception as e:
            print(f"Error in evaluate_single_problem: {str(e)}")
            return None


async def save_results_async(output_file: str, data: dict):
    async with aiofiles.open(output_file, 'a') as f:
        await f.write(json.dumps(data) + '\n')


async def main(k: int = 5, debug: bool = False, resume: bool = False):
    # Initialize the AsyncOpenAI clients
    lm_client = AsyncOpenAI(
        base_url="http://localhost:8010/v1",
        api_key="token-abc123"
    )
    
    reward_client = AsyncOpenAI(
        base_url="http://localhost:8011/v1",
        api_key="token-abc123"
    )
    
    # Initialize the reward model
    from omegaprm import ProcessRewardModel
    reward_model = ProcessRewardModel(
        client=reward_client,
        model="deepseek-r1-14b-cot-math-reasoning-full",
        # model = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
        temperature=0.0,
        max_tokens=1
    )
    
    model_name = "Qwen2.5-MATH-7B-Instruct"
    # model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    # model_name = "DeepSeek-R1-Distill-Qwen-7B"
    
    
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
    output_file = f"bon_qwen_rm_k{k}_results.jsonl"
    results = []
    if resume:
        if os.path.exists(output_file):
            # Deduplicate the results file
            dedup = {}
            with open(output_file, 'r') as res_file:
                for line in res_file:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            question = rec.get("question")
                            if question is not None:
                                dedup[question] = rec
                                results.append(rec)  # Add already processed results
                        except Exception as e:
                            continue

            # Write deduplicated results back to the file
            with open(output_file, 'w') as res_file:
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
        asyncio.create_task(evaluate_single_problem(prob, lm_client, reward_model, model_name, k, sem))
        for prob in problems
    ]
    
    # Use as_completed to update progress with tqdm
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Processing problems'):
        result = await future
        if result is not None:
            results.append(result)
            # Save result immediately
            await save_results_async(output_file, result)

    if results:
        agg_rates = {}
        for result in results:
            for mode, flag in result["evaluation_results"].items():
                agg_rates[mode] = agg_rates.get(mode, 0) + flag
        print("Final Success Rate for each aggregation mode:")
        for mode, total in agg_rates.items():
            print(f"{mode}: {total/len(results) * 100:.2f}%")

    print(f"Evaluation complete. Processed {len(results)} problems successfully.")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8, help="Number of completions to generate per question")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (only evaluate the first 50 problems)")
    parser.add_argument("--resume", action="store_true", help="Resume evaluation by skipping already evaluated problems")
    args = parser.parse_args()
    asyncio.run(main(k=args.k, debug=args.debug, resume=args.resume))
    
    