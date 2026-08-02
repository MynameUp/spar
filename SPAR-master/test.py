import json
import requests

key = "33aacf78ed46627903060d7efea21d3692e58687"
url = "https://google.serper.dev/search"
headers = {"X-API-KEY": key, "Content-Type": "application/json"}

cases = [
    {"q": "test"},
    {"q": "test", "num": 10, "page": 1},
    {"q": "synthetic data site:arxiv.org"},
    {"q": "synthetic data site:arxiv.org", "num": 10, "page": 1},
    {"q": "Literature review on synthetic data applications in NLP and vision SFT site:arxiv.org", "num": 10, "page": 1},
    {"q": "Provide me with some top-tier journal papers to expand my ideas on using synthetic data to augment supervised fine-tuning (SFT) while ensuring data quality and diversity, maintaining a balance between the two. site:arxiv.org", "num": 10, "page": 1},
]

for i, payload in enumerate(cases, 1):
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
        print(f"[{i}] q={payload['q'][:50]!r} num={payload.get('num')} page={payload.get('page')}")
        print(f"    -> {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"[{i}] 异常: {type(e).__name__}: {e}")
