# Standard libraries
import os
import sys
import json
import math
import argparse
import re  # or use 'regex as re' if you really need regex advanced features

# Third-party libraries
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
from adaptive_consistency import AC, BetaStoppingCriteria
from prettytable import PrettyTable

# Local imports
from RankedVotingSC.RankBasedSC import (
    IRV,
    BCV,
    MRRV
)

def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument('--cot_file', type=str, default=None, help='Path to the input file (optional).')
    parser.add_argument('--bon_file', type=str, default=None, help='Path to the input file (optional).')
    parser.add_argument('--sc_file', type=str, default=None, help='Path to the input file (optional).')
    parser.add_argument('--ac_file', type=str, default=None, help='Path to the input file (optional).')
    parser.add_argument('--args.rc_file', type=str, required=True, help='Path to the input file.')
    parser.add_argument('--vote_output_dir', type=str, required=True, help='Directory to save outputs.')

    args = parser.parse_args()
    return args

def generate_answer_from_model(resps, index, reg):
    raw_text = resps[index]
    filtered_answer = re.search(reg, raw_text).group(1) if re.search(reg, raw_text) else 'invalid'
    if filtered_answer == 'invalid':
        filtered_answer = re.search(r'is:?\s*\((\w)\)', raw_text).group(1) if re.search(r'is:?\s*\((\w)\)', raw_text) else 'invalid'
    return filtered_answer

# 1. extract key
def filter_keys(input_file):
    keys_to_keep = ["doc_id", "target", "resps"]
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    filtered_data = []
    for item in data:
        filtered_item = {key: item[key] for key in keys_to_keep if key in item}
        filtered_data.append(filtered_item)
    
    return filtered_data

# 2. match letters
def match_letters(raw_text, pattern):
    """
    Input: "The most likely place where a revolving door would be used as a security measure is a bank. Banks often have revolving doors to control access and limit the number of people entering and exiting at any given time, enhancing security.  While other options might have revolving doors, their primary purpose isn't necessarily security. \n\n**So the answer is (A). The ranking of options by likelihood is: A > D > C > B > E.** \n\n\n\n"
    Output: ['A', 'D', 'C', 'B', 'E']
    """
    match = re.search(pattern, raw_text)
    if not match: 
        match = re.search(r'answer is [*]*\((\w)\)', raw_text)
    if match:
        return list(match.groups())
    else:
        return ['invalid']

if __name__ == "__main__":
    args = parse_args()
    # =============evaluate CoT====================
    # cot_result = None
    # if args.cot_file:
    #     cot_file = args.cot_file
    #     filtered_cot_resps = filter_keys(cot_file)
    #     for item in filtered_cot_resps:
    #         resp = item['resps'][0]
    #         answer = re.search(r'answer is [*]*\((\w)\)', resp)
    #         if answer:
    #             item['resps'] = list(answer.groups())
    #         else:
    #             item['resps'] = ['invalid']
    #     cot_count = 0
    #     data_len = len(filtered_cot_resps)
    #     for item in filtered_cot_resps:
    #         answer = item['resps']
    #         target = item['target']
    #         if answer[0] == target:
    #             cot_count += 1
    #     cot_result = cot_count / data_len
    cot_result = None
    if args.cot_file:
        filtered_cot_resps = filter_keys(args.cot_file)
        cot_count = 0
        for item in filtered_cot_resps:
            resp = item['resps'][0]
            match = re.search(r'answer is [*]*\((\w)\)', resp)
            answer = match.group(1) if match else 'invalid'
            if answer == item['target']:
                cot_count += 1
        cot_result = cot_count / len(filtered_cot_resps)
    # =============evaluate BoN====================
    bon_result = None
    if args.bon_file:
        # model_id = "Qwen/Qwen2.5-7B-Instruct"
        model_id = "/mnt/d/share/models/Qwen2.5/qwen2.5-instruct-7B"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        input_text = "The capital of France is"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]      
        prompt = """
                Given the following problem, provide the answer. **Use the following format**: "(Concise and clear reasoning steps). So the answer is: (X).".
                Question: {question}
                RESPONSE 0: {resp0}
                RESPONSE 1: {resp1}
                RESPONSE 2: {resp2}
                RESPONSE 3: {resp3}
                RESPONSE 4: {resp4}
                RESPONSE 5: {resp5}
                RESPONSE 6: {resp6}
                RESPONSE 7: {resp7}
                Which response is the best? You should also **focus on the fomat I need** and only output the number from 0-7. The best is: {best_resp}.
        """
        all_data = []
        with open(args.bon_file, "r") as data:
                # data = data[:50]
                for item in tqdm.tqdm(data):
                    question = item["doc"]["question"]
                    resps = item["resps"][0]
                    full_prompt = prompt.format(
                        question=question,
                        resp0=resps[0],
                        resp1=resps[1],
                        resp2=resps[2],
                        resp3=resps[3],
                        resp4=resps[4],
                        resp5=resps[5],
                        resp6=resps[6],
                        resp7=resps[7]
                    )
                    messages = [
                        {"role": "system", "content": "Only ouput the number from 0-7. For example, Input: 'Which response is better? only output the number from 0-7. RESPONSE 1: ... \n ... \n RESPONSE 7: ...The best is: \n Ouput: 3'"},
                        {"role": "user", "content": full_prompt},
                    ]
                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(model.device)
                    terminators = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|endoftext|>")
                    ]
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=512,
                        eos_token_id=terminators,
                        # attention_mask=attention_mask,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                    )
                    response = outputs[0][input_ids.shape[-1]:]
                    decoded_response = tokenizer.decode(response, skip_special_tokens=True)
                    resp_index = int(decoded_response) if decoded_response in ["0", "1", "2", "3", "4", "5", "6", "7"] else "invalid"
                    re.search(r"answer is:?\s*\(([A-E])\)", resps[resp_index]).group(1) if resp_index != "invalid" and re.search(r"answer is:?\s*\(([A-E])\)", resps[resp_index]) else "invalid"
                    
                    all_data.append({
                        "doc_id": item["doc_id"],
                        "doc": item["doc"],
                        "target": item["target"],
                        "resps": resps,
                        "best_resp_index": decoded_response,
                        "best_resp": resps[resp_index] if resp_index != "invalid" else "invalid",
                        "response": decoded_response,
                        "filtered_resps": re.search(r"answer is:?\s*\(([A-E])\)", resps[resp_index]).group(1) if resp_index != "invalid" and re.search(r"answer is:?\s*\(([A-E])\)", resps[resp_index]) else "invalid"
                    })
        bon_count = 0
        data_len = len(all_data)
        for item in all_data:
            answer = item['filtered_resps']
            target = item['target']
            if answer == target:
                bon_count += 1
        bon_result = bon_count / data_len
    # =============evaluate SC====================
    sc_result = None
    if args.sc_file:
        sc_count = 0
        with open(args.sc_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                counts = {}
                for resp in item['resps'][0]:  
                    match = re.search(r"So the answer is \((\w)\)", resp)
                    if match:
                        ans = match.group(1)
                    else:
                        ans = 'invalid'
                    counts[ans] = counts.get(ans, 0) + 1
                
                final_answer = max(counts.items(), key=lambda x: x[1])[0]

                if final_answer == item['target']:
                    sc_count += 1
        sc_result = sc_count / len(data)
    # =============evaluate AC====================
    ac_result = None
    if args.ac_file:
        ac_count = 0
        ac = AC(stop_criteria=BetaStoppingCriteria(0.95), max_gens = 8)
        reg = r'[Ss]o the answer is:?\s*\((\w)\)'
        with open(args.ac_file, 'r') as f:
            data = json.load(f)
            ac_samples = []
            for item in data:
                answers = []
                for i in range(8):
                    answers.append(generate_answer_from_model(item['resps'][0], i, reg))
                    if ac.should_stop(answers):
                        break
                most_common = max(set(answers), key = answers.count)
                json_item = {
                    'doc_id': item['doc_id'],
                    'doc': item['doc'],
                    'target': item['target'],
                    'answer': answers,
                    'resps': item['resps'],
                    'ac_result': most_common
                }
            ac_samples.append(json_item)
            for item in ac_samples:
                answer = item['ac_result']
                target = item['target']
                if answer == target:
                    ac_count += 1
        ac_result = ac_count / len(ac_samples)
        ...
    # =============evaluate RankBasedSC====================
    args.rc_file_dir = os.path.dirname(args.rc_file)
    # output_file = os.path.join(args.rc_file_dir, 'votes.json')

    # 1 getkey and filter data in memory
    filtered_data = filter_keys(args.rc_file)

    # 2 extract_answer in memory
    for item in filtered_data:
        new_resps = []
        for resp in item['resps'][0]:
            resp = match_letters(resp, r"([A-E]) > ([A-E]) > ([A-E]) > ([A-E]) > ([A-E])")
            new_resps.append(resp)
        item['resps'] = new_resps
    print("Extract Answer! Done!")

    # 3 voting
    for item in filtered_data:
        resps = item["resps"]

        irv = IRV(resps)
        bcv = BCV(resps)
        mrrv = MRRV(resps)

        irv_answer = irv.run()
        bcv_answer = bcv.run()
        mrr_answer = mrrv.run()

        item["irv_answer"] = irv_answer
        item["bcv_answer"] = bcv_answer
        item["mrr_answer"] = mrr_answer

    # Save final results to votes.json
    # with open(output_file, 'w', encoding='utf-8') as file:
    #     json.dump(filtered_data, file, ensure_ascii=False, indent=4)

    # eval
    irv_count = 0
    bcv_count = 0
    mrrv_count = 0
    data_len = len(filtered_data)

    for item in filtered_data:
        irv_answer = item["irv_answer"]
        bcv_answer = item["bcv_answer"]
        mrr_answer = item["mrr_answer"]
        target = item["target"]

        if irv_answer == target:
            irv_count += 1
        if bcv_answer == target:
            bcv_count += 1
        if mrr_answer == target:
            mrrv_count += 1


    # Output results
    # print("-------- test -------")
    # print("+--------+----------+")
    # print("| Method | Accuracy |")
    # print("+--------+----------+")
    # print(f"| IRV    | {irv_count / data_len:.6f} |")
    # print("+--------+----------+")
    # print(f"| BCV    | {bcv_count / data_len:.6f} |")
    # print("+--------+----------+")
    # print(f"| MRR    | {mrrv_count / data_len:.6f} |")
    # print("+--------+----------+")
    print("-------- test -------")
    methods = []
    accs = []
    if cot_result is not None:
        methods.append("CoT")
        accs.append(f"{cot_result:.6f}")
    if bon_result is not None:
        methods.append("BoN")
        accs.append(f"{bon_result:.6f}")
    if sc_result is not None:
        methods.append("SC")
        accs.append(f"{sc_result:.6f}")
    if ac_result is not None:
        methods.append("AC")
        accs.append(f"{ac_result:.6f}")
    if irv_count is not None:
        methods.append("IRV")
        accs.append(f"{irv_count / data_len:.6f}")
    if bcv_count is not None:
        methods.append("BCV")
        accs.append(f"{bcv_count / data_len:.6f}")
    if mrrv_count is not None:
        methods.append("MRRV")
        accs.append(f"{mrrv_count / data_len:.6f}")
  
    table = PrettyTable()
    table.field_names = methods
    table.add_row(accs)
    print(table)