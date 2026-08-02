# -*- coding:utf-8 -*-
"""
Evaluate Recall / Precision of SPAR Test5 results against benchmark gold answers.

gold    : ./benchmark/spar_bench_test5.jsonl  -> question + answer[] (gold title list)
results : ./gen_result/Test5_5_.../*.json     -> search tree
          root.extra.searched_docs = {paper_id: {paper_id, title, ...}}  (all retrieved docs)
          node.docs[] = papers judged relevant (paper_id refs)
"""

import hashlib
import json
import os

BENCH_FILE = "./benchmark/spar_bench_test5.jsonl"
OUTPUT_FOLDER = "./gen_result/Test5_5_msearch_arxiv-openalex_depth2_do_reference_False_query_judge_True_fusion_AUTOMATIC_no_enddate_no_autocorrect_pasa_score_0.5"


def get_md5(string):
    md5_hash = hashlib.md5()
    md5_hash.update(string.encode("utf-8"))
    return md5_hash.hexdigest()


def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    return "".join(letters).lower()


def build_title_index(root):
    """root.extra.searched_docs -> {paper_id: normalized_title}"""
    idx = {}
    searched = (root.get("extra", {}) or {}).get("searched_docs", {}) or {}
    for pid, doc in searched.items():
        if isinstance(doc, dict) and doc.get("title"):
            idx[str(pid)] = keep_letters(doc["title"])
    return idx


def collect_relevant_titles(root, title_index):
    """各节点 docs 中判定相关的论文 -> 归一化标题 set"""
    relevant = set()

    def walk(node):
        for d in node.get("docs", []) or []:
            if isinstance(d, dict):
                pid = d.get("paper_id") or d.get("id")
                if pid and str(pid) in title_index:
                    relevant.add(title_index[str(pid)])
        for child in node.get("children", []) or []:
            walk(child)

    walk(root)
    return relevant


def main():
    with open(BENCH_FILE, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    print(f"benchmark: {BENCH_FILE}  ({len(lines)} items)")
    print("=" * 110)

    rows = []
    for i, ln in enumerate(lines, 1):
        data = json.loads(ln)
        question = data["question"]
        gold = [keep_letters(t) for t in data.get("answer", [])]
        dest = os.path.join(OUTPUT_FOLDER, f"{get_md5(question)}.json")

        if not os.path.exists(dest):
            print(f"[{i}] MISSING result file: {dest}")
            continue

        with open(dest, "r", encoding="utf-8") as f:
            root = json.load(f)

        title_index = build_title_index(root)
        retrieved_all = set(title_index.values())          # 全部检索文档
        relevant_titles = collect_relevant_titles(root, title_index)  # 判定相关
        gold_set = set(gold)

        hits_all = retrieved_all & gold_set
        hits_rel = relevant_titles & gold_set

        recall_all = len(hits_all) / len(gold) if gold else 0.0
        prec_all = len(hits_all) / len(retrieved_all) if retrieved_all else 0.0
        recall_rel = len(hits_rel) / len(gold) if gold else 0.0
        prec_rel = len(hits_rel) / len(relevant_titles) if relevant_titles else 0.0

        rows.append((recall_all, prec_all, recall_rel, prec_rel))

        print(f"[{i}] {question[:78]}")
        print(f"    gold={len(gold)}  retrieved_all={len(retrieved_all)}  relevant={len(relevant_titles)}")
        print(f"    Recall(all)={recall_all:.1%}  Precision(all)={prec_all:.1%}  |  "
              f"Recall(rel)={recall_rel:.1%}  Precision(rel)={prec_rel:.1%}")
        print(f"    HITS: {sorted(hits_all) if hits_all else 'NONE'}")
        missed = gold_set - hits_all
        if missed:
            print(f"    MISSED({len(missed)}): {sorted(missed)}")
        print("-" * 110)

    if rows:
        n = len(rows)
        avg = tuple(100 * sum(r[k] for r in rows) / n for k in range(4))
        print("=" * 110)
        print(f"AVERAGE over {n} items:")
        print(f"    Recall(all)     = {avg[0]:.1f}%")
        print(f"    Precision(all)  = {avg[1]:.1f}%")
        print(f"    Recall(rel)     = {avg[2]:.1f}%")
        print(f"    Precision(rel)  = {avg[3]:.1f}%")


if __name__ == "__main__":
    main()
