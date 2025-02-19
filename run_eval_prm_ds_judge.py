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
TEMPLATE = open('./templates/judge_template.txt').read().strip()

def single_process(d):
    global client
    steps = d['steps']
    messages = []
    generations = []  # Store all intermediate generations
    for sdx, step in enumerate(steps):
        try:
            # Wrap each step with <step1>...</step1>, <step2>...</step2>, etc.
            wrapped_steps = [f"<step{i+1}>{s}</step{i+1}>" for i, s in enumerate(steps[:sdx + 1])]
            combined_steps = "\n\n".join(wrapped_steps)
            prompt = TEMPLATE.format(problem=d['problem'], response=combined_steps)
            messages = [{
                'role': 'user',
                'content': prompt
            }]
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model='gpt-4o',
                        messages=messages,
                        n=1,
                        temperature=0.,
                        max_tokens=1024,
                    )
                    
                    # Check if completion or its attributes are None
                    if (completion is None or 
                        completion.choices is None or 
                        len(completion.choices) == 0 or 
                        completion.choices[0] is None or 
                        completion.choices[0].message is None or 
                        completion.choices[0].message.content is None):
                        if attempt < max_retries - 1:
                            continue
                        else:
                            response = "Error: No valid response received"
                            break
                    else:
                        response = completion.choices[0].message.content
                        break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        response = f"Error: {str(e)}"
            
            # Check if the current step is correct
            pattern = r'(?:\b\w+\b\s*){0,5}\+'
            judgment = re.search(pattern, response.strip().lower())
            step_info = {
                'step_index': sdx,
                'step_content': step,
                'response': response,
                'error': None if 'Error:' not in response else response
            }
            generations.append(step_info)
            
            if not judgment:
                return {'step': sdx, 'generations': generations}
            messages.append({'role': 'assistant', 'content': '+'})
            
        except Exception as e:
            print(f"Error processing step {sdx}: {str(e)}")
            step_info = {
                'step_index': sdx,
                'step_content': step,
                'response': f"Error: {str(e)}",
                'error': str(e)
            }
            generations.append(step_info)
            return {'step': sdx, 'generations': generations}
            
    return {'step': -1, 'generations': generations}

def main():
    global client
    # Initialize the client
    client = OpenAI(
        # base_url="http://localhost:8000/v1",
        # api_key="token-abc123",
        base_url="https://a.fe8.cn/v1",
        api_key="sk-QpHUrsblHgB7kAzcpwLmrFz3yKKTiFVlFOW2vgVc7ARfqsXR"
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
