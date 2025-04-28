import json
import re
import sys
from RankBasedSC import (
    IRV,
    BCV,
    MRRV
)
import math
import numpy as np

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
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <input_file.json>")
        sys.exit(1)

    input_file = sys.argv[1]

    import os
    input_dir = os.path.dirname(input_file)
    output_file = os.path.join(input_dir, 'votes.json')

    # 1 getkey and filter data in memory
    filtered_data = filter_keys(input_file)
    print("Get Key! Done!")

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
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=4)
    print("Vote! Done!")

    # eval
    count1 = 0
    count2 = 0
    count3 = 0
    data_len = len(filtered_data)

    for item in filtered_data:
        irv_answer = item["irv_answer"]
        bcv_answer = item["bcv_answer"]
        mrr_answer = item["mrr_answer"]
        target = item["target"]

        if irv_answer == target:
            count1 += 1
        if bcv_answer == target:
            count2 += 1
        if mrr_answer == target:
            count3 += 1
    print("-------- test -------")
    print("+--------+----------+")
    print("| Method | Accuracy |")
    print("+--------+----------+")
    print(f"| IRV    | {count1 / data_len:.6f} |")
    print("+--------+----------+")
    print(f"| BCV    | {count2 / data_len:.6f} |")
    print("+--------+----------+")
    print(f"| MRR    | {count3 / data_len:.6f} |")
    print("+--------+----------+")