# -*- coding:utf-8 -*-
"""测试 google(Serper) + arxiv 完整检索链路：串行 + 并发"""
import threading
import time

from api_web import get_doc_info_from_api, google_search_arxiv_id

TEST_QUERIES = [
    "Survey of synthetic data generation methods for supervised fine-tuning",
    "A Survey of Data Synthesis Approaches",
    "synthetic data for language models survey",
]


def test_serial():
    """串行: google 搜 arxiv id -> arxiv API 取论文详情(完整链路)"""
    print("=== 1. 串行完整链路 (google -> arxiv) ===")
    for q in TEST_QUERIES:
        t0 = time.time()
        doc_info = get_doc_info_from_api(q)
        print(f"[串行] {q[:50]!r} -> {len(doc_info)} papers, 耗时 {time.time()-t0:.1f}s")
        for pid, doc in list(doc_info.items())[:3]:
            print(f"    - {pid}: {doc.get('title', '')[:70]}")


def test_concurrent():
    """并发: 验证 Serper 锁是否生效(复现之前 400 的场景)"""
    print("=== 2. 并发测试 (验证串行化锁) ===")
    results = {}

    def worker(q):
        t0 = time.time()
        try:
            ids = google_search_arxiv_id(q)
            results[q] = (ids, time.time() - t0)
        except Exception as e:
            results[q] = ([], f"ERROR: {e}")

    threads = [threading.Thread(target=worker, args=(q,)) for q in TEST_QUERIES]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = 0
    for q, (ids, info) in results.items():
        dt = info if isinstance(info, float) else info
        print(f"[并发] {q[:50]!r} -> {len(ids)} arxiv_ids, 耗时 {dt}s")
        print(f"        ids: {ids[:5]}")
        if ids:
            ok += 1
    print(f"并发成功率: {ok}/{len(TEST_QUERIES)}")


if __name__ == "__main__":
    test_serial()
    print()
    test_concurrent()
