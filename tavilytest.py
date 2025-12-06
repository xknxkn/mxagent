# To install: pip install tavily-python
import sys, io
from tavily import TavilyClient

# Ensure stdout can output UTF-8 on Windows consoles to avoid 'gbk' encode errors
try:
    if getattr(sys.stdout, "reconfigure", None):
        sys.stdout.reconfigure(encoding="utf-8")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    # fallback: ignore if we cannot reconfigure
    pass

client = TavilyClient("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")
response = client.search(
    query="上海今天天气\n",
    max_results=1
)
print(response)
print(response['results'][0]["content"])