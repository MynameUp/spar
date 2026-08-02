# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Descriptions] : Test run with 5 OwnBenchmark samples
# ==================================================================

import json
from tqdm import tqdm
import os
import traceback
import glob
from global_config import (
    LLM_MODEL_NAME,
    DO_REFERENCE_SEARCH,
    DO_FUSION_JUDGE,
    FUSION_TEMPLATE,
    SEARCH_ROUTES,
)
from utils import get_md5
import shutil
from pipeline_spar import AcademicSearchTree

file_lst = [
    "./global_config.py",
    "./instruction.py",
    "./run_spr_agent.py",
    "./search_engine.py",
    "./api_web.py",
    "./pipeline_spar.py"
]

sample_num = 5
score_thresh = 0.5
max_depth = 2
relevance_doc_num = 10

src_file = "./benchmark/spar_bench_test5.jsonl"
select_file = f"./benchmark/spar_bench_test5_select_{sample_num}.jsonl"

print(f"select_file: {select_file}")

output_folder = f"./gen_result/Test5_{sample_num}_msearch_{'-'.join(SEARCH_ROUTES)}_depth{max_depth}_do_reference_{DO_REFERENCE_SEARCH}_query_judge_{DO_FUSION_JUDGE}_fusion_{FUSION_TEMPLATE}_no_enddate_no_autocorrect_pasa_score_{score_thresh}"

print(f"output_folder: {output_folder}")

os.makedirs(output_folder, exist_ok=True)

search_agent = AcademicSearchTree(
    max_depth=max_depth, max_docs=relevance_doc_num, similarity_threshold=score_thresh
)

for one in file_lst:
    if os.path.exists(one):
        shutil.copy2(one, output_folder)

already = {}
for one in glob.glob(f"{output_folder}/*.json"):
    with open(one, "r", encoding="utf-8") as fr:
        info = json.load(fr)
    question = info["search_query"]
    already[question] = one

# Create test file from first 5 lines of benchmark
with open("./benchmark/spar_bench.jsonl", "r", encoding="utf-8") as f:
    all_lines = f.readlines()[:5]

with open(src_file, "w", encoding="utf-8") as fw:
    for line in all_lines:
        fw.write(line.strip() + "\n")

lines = all_lines
print(f"lines: {len(lines)}")

with open(select_file, "w", encoding="utf-8") as fw:
    for one in lines:
        fw.write(one.strip() + "\n")

for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Processing lines"):
    try:
        data = json.loads(line)
        question = data["question"]
        end_date = ""

        if question in already:
            print(f"pass: {already[question]}")
            continue

        dest_name = get_md5(question)
        dest_file = os.path.join(output_folder, f"{dest_name}.json")

        sorted_docs = search_agent.search(question, end_date=end_date)

        if "answer" in data:
            search_agent.root.extra["answer"] = data["answer"]

        res = search_agent.root.convert_to_dict()
        with open(dest_file, "w", encoding="utf-8") as fw:
            json.dump(res, fw, indent=2)

        try:
            print("draw search tree")
            search_agent.visualize_tree(f"{output_folder}/{dest_name}")
        except:
            traceback.print_exc()

    except:
        traceback.print_exc()

print(f"output_folder: {output_folder}")