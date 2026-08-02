# -*- coding: utf-8 -*-
"""测试阿里云百炼 qwen3-max 连接（支持直接运行与 pytest）"""
import os
import sys
import time
import requests

# 从 global_config 读取密钥（兜底）
sys.path.insert(0, os.path.dirname(__file__))
from global_config import API_KEY, ENDPOINT

# 也尝试从环境变量读取
api_key = os.getenv("DASHSCOPE_API_KEY", API_KEY)
base_url = os.getenv("DASHSCOPE_ENDPOINT", ENDPOINT)


def mask_key(key: str) -> str:
    """掩码显示 API Key，避免完整泄露"""
    if len(key) > 30:
        return f"{key[:20]}...{key[-10:]}"
    return key


def main() -> int:
    """执行全部连接测试，全部通过返回 0，否则返回 1"""
    passed = 0
    failed = 0

    def report(name: str, ok: bool, detail: str, elapsed: float):
        nonlocal passed, failed
        status = "✅ 成功" if ok else "❌ 失败"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name} ({elapsed:.2f}s)")
        print(f"         {detail}")

    print("=" * 60)
    print("阿里云百炼 DashScope 连接测试")
    print("=" * 60)
    print(f"Endpoint : {base_url}")
    print(f"API Key  : {mask_key(api_key)}")
    print()

    # 代理环境检查（百炼在部分网络环境需要代理才能访问）
    print(f"HTTP_PROXY  : {os.getenv('HTTP_PROXY', '未设置')}")
    print(f"HTTPS_PROXY : {os.getenv('HTTPS_PROXY', '未设置')}")
    print()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 方式0: 校验 API Key 是否有效（列出可用模型）
    print("[0/3] 校验 API Key 并列出可用模型...")
    t0 = time.time()
    try:
        resp = requests.get(f"{base_url}/models", headers=headers, timeout=30)
        if resp.status_code == 200:
            models = [m["id"] for m in resp.json().get("data", [])]
            report("获取模型列表", True, f"可用模型: {models}", time.time() - t0)
        else:
            report("获取模型列表", False, f"HTTP {resp.status_code}: {resp.text[:300]}", time.time() - t0)
    except Exception as e:
        report("获取模型列表", False, f"异常: {e}", time.time() - t0)

    print()

    # 方式1: 原生 HTTP 请求
    print("[1/3] 原生 HTTP 请求测试...")
    t0 = time.time()
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": "qwen3.5-plus",
                "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己"}],
                "max_tokens": 100,
                "temperature": 0.7,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            report("HTTP 对话", True, f"回复: {content}", time.time() - t0)
        else:
            report("HTTP 对话", False, f"HTTP {resp.status_code}: {resp.text[:300]}", time.time() - t0)
    except Exception as e:
        report("HTTP 对话", False, f"异常: {e}", time.time() - t0)

    print()

    # 方式2: OpenAI SDK
    print("[2/3] OpenAI SDK 测试...")
    t0 = time.time()
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model="qwen3.5-plus",
            messages=[{"role": "user", "content": "你好，请用一句话介绍你自己"}],
            max_tokens=100,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        report("SDK 对话", True, f"回复: {content}", time.time() - t0)
    except Exception as e:
        report("SDK 对话", False, f"异常: {e}", time.time() - t0)

    print()
    print("=" * 60)
    print(f"结果汇总: 通过 {passed} 项, 失败 {failed} 项")
    if failed == 0:
        print("🎉 连接正常，可以继续使用 SPAR 检索流程")
    else:
        print("⚠️  存在失败项，请检查 API Key、Endpoint 或网络代理")
    print("=" * 60)

    return 0 if failed == 0 else 1


def test_bailian_connection():
    """pytest 入口：验证阿里云百炼连接"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
