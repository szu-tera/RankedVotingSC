import json
import re
from RankBasedSC import (
    IRV,
    BCV,
    MRRV
)
import math
import numpy as np

# 1. 提取答案
def filter_keys(input_file, output_file):
    keys_to_keep = ["doc_id", "target", "resps", "filtered_resps"]
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    filtered_data = []
    for item in data:
        filtered_item = {key: item[key] for key in keys_to_keep if key in item}
        filtered_data.append(filtered_item)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)

# 2. 匹配
"""
示例:
    输入: "The most likely place where a revolving door would be used as a security measure is a bank. Banks often have revolving doors to control access and limit the number of people entering and exiting at any given time, enhancing security.  While other options might have revolving doors, their primary purpose isn't necessarily security. \n\n**So the answer is (A). The ranking of options by likelihood is: A > D > C > B > E.** \n\n\n\n"
    输出: ['A', 'D', 'C', 'B', 'E']
"""
def match_letters(raw_text):
    # print([raw_text])
    # 匹配1
    match = re.search(r"([A-D]) > ([A-D]) > ([A-D]) > ([A-D])", raw_text)
    # print(match)
    # tips: match: 从头开始匹配，search: 匹配整个字符串
    # 匹配2
    if not match:
        match = re.search(r'\(([A-D])\) > \(([A-D])\) > \(([A-D])\) > \(([A-D])\)', raw_text)
    # 匹配3: 在匹配不到
    if not match: 
        match = re.search(r'answer is [*]*\((\w)\)', raw_text)
    # 匹配4: 再匹配不到，取第一个()包住的
    if not match:
        match = re.search(r'is \(([A-D])\)', raw_text)
    # 匹配不到设置为 invalid
    # match = match if match else ['invalid']
    # match = match if match == ['invalid'] else list(match.groups())
    if match:
        return list(match.groups())
    else:
        return ['invalid']

if __name__ == "__main__":
    # 1 getkey
    input_file = 'samples.json'
    output_file = 'filtered_samples.json'
    filter_keys(input_file, output_file)
    print("Get Key! Done!")

    # 2 extract_answer
    with open("filtered_samples.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    for item in data:
        new_resps = []
        for resp in item['resps'][0]:
            resp = match_letters(resp)
            new_resps.append(resp)
        item['resps'] = new_resps
    # 将结果保存为新的 JSON 文件
    with open("output.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print("Extract Answer! Done!")

    # 3 vote
    # 读取JSON文件
    with open('output.json', 'r') as file:
        data = json.load(file)

    for item in data:
        # 计算vote1_answer
        resps = item["resps"]

        irv = IRV(resps)
        bcv = BCV(resps)
        mrrv = MRRV(resps)
        irv_answer = irv.run()

        # 计算vote2_answer
        bcv_answer = bcv.run()

        # MRR
        mrr_answer = mrrv.run()

        item["irv_answer"] = irv_answer
        item["bcv_answer"] = bcv_answer
        item["mrr_answer"] = mrr_answer

    # 写回JSON文件
    with open('votes.json', 'w') as file:
        json.dump(data, file, indent=4)
    print("Vote! Done!")

    # eval
    with open('votes.json', 'r') as file:
        data = json.load(file)
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    count8 = 0
    count9 = 0
    data_len = len(data)