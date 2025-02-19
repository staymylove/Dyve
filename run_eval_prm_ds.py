import os
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool
from openai import OpenAI
from datasets import load_dataset
import re
# Create a global client variable
client = None


def single_process(d):
    global client
    steps = d['steps']
    messages = []
    generations = []  # Store all intermediate generations
    for sdx, step in enumerate(steps):
        step_info = {
            'step_index': sdx,
            'step_content': step,
            'is_first_step': sdx == 0
        }
        
        if sdx == 0:
            messages.append({
                'role': 'user', 
                'content': f"Problem: {d['problem']}\n\nStep: {step}\n\nIs this step correct? You must answer with '+' for correct or '-' for incorrect in the end of your response."
            })
        else:
            messages.append({
                'role': 'user', 
                'content': f"Step: {step}\n\nIs this step correct? You must answer with '+' for correct or -' for incorrect in the end of your response."
            })
        
        completion = client.chat.completions.create(
            model='DS14B',
            messages=messages,
            n=1,
            temperature=0.,
            max_tokens=1024,
        )
        response = completion.choices[0].message.content
        step_info['response'] = response
        generations.append(step_info)  # Save the generation with step information
        
        pattern = r'(?:\b\w+\b\s*){0,5}\+'
        judgment = re.search(pattern, response.strip().lower())
        if not judgment:
            return {'step': sdx, 'generations': generations}
        messages.append({'role': 'assistant', 'content': '+'})
    return {'step': -1, 'generations': generations}

def main():
    global client
    # Initialize the client
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    os.makedirs('outputs/DS14B', exist_ok=True)

    configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    for config in configs:
        input_data = load_dataset('Qwen/ProcessBench', split=config)
        with Pool(32) as p:
            predictions = list(tqdm(p.imap(single_process, input_data), total=len(input_data),
                                    desc=f'Processing {config}', dynamic_ncols=True))
        
        res_data = []
        for idx, d in enumerate(input_data):
            new_d = d.copy()
            new_d['prediction'] = predictions[idx]['step']
            new_d['generations'] = predictions[idx]['generations']
            new_d['match'] = predictions[idx]['step'] == d['label']
            res_data.append(new_d)
        
        data1 = [e for e in res_data if e['label'] != -1]
        data2 = [e for e in res_data if e['label'] == -1]
        with open(f'outputs/DS14B/{config}_error.jsonl', 'w') as f:
            for e in data1:
                f.write(json.dumps(e) + '\n')
        with open(f'outputs/DS14B/{config}_correct.jsonl', 'w') as f:
            for e in data2:
                f.write(json.dumps(e) + '\n')
        
        acc1 = np.mean([e['match'] for e in data1]) * 100
        acc2 = np.mean([e['match'] for e in data2]) * 100
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
        print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')

if __name__ == '__main__':
    main()
