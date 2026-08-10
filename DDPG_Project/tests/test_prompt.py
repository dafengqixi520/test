"""Test simplified prompts with DeepSeek V4"""
import os
os.environ['NO_PROXY'] = '*'
from openai import OpenAI

client = OpenAI(
    api_key='sk-30f21a01513144dcb190257ea43daec4',
    base_url='https://api.deepseek.com'
)

# Test 1: simplified short prompt
short = """You are a task scheduling expert for edge computing.

Node 1 status: CPU 36G, queue 0%, failure rate 2%.
Task: size 1536KB, complexity 2G cycles, deadline 3s, hops 0/3.
Neighbor node 0: CPU 84G, queue 0%, failure rate 1%.

Options:
Option 1: Local execution on node 1
Option 2: Forward to node 0

Choose the best option. Output ONLY one line: either "Local execution" or "Forward to node 0"."""

print(f"Test 1 (simple, {len(short)} chars):")
r = client.chat.completions.create(
    model='deepseek-v4-pro',
    messages=[{'role': 'user', 'content': short}],
    temperature=0.2,
    max_tokens=30
)
print(f"  Content: {repr(r.choices[0].message.content)}")

# Test 2: Use deepseek-chat model instead (no thinking mode)
print(f"\nTest 2 (deepseek-chat model):")
r2 = client.chat.completions.create(
    model='deepseek-chat',
    messages=[{'role': 'user', 'content': short}],
    temperature=0.1,
    max_tokens=20
)
print(f"  Content: {repr(r2.choices[0].message.content)}")

# Test 3: Full long prompt with deepseek-chat
full = """You are a task scheduling expert for edge computing.

Node 1: active, CPU 36G (12 cores), queue 0%, exec failure 2%, node failure 1%.
Task on node 1: size 1536KB, complexity 2G cycles, deadline 3s, hops 0/3.
Neighbor node 0: CPU 84G (28 cores), queue 0%, exec failure 1%, transmission rate 30MB/s, tx failure 1%.

Available actions:
1. Local execution on node 1
2. Forward to node 0

Consider: reliability, load balance, deadline proximity. Output ONLY the chosen action:
- Local execution
- Forward to node 0"""

print(f"\nTest 3 (deepseek-chat, long prompt {len(full)} chars):")
r3 = client.chat.completions.create(
    model='deepseek-chat',
    messages=[{'role': 'user', 'content': full}],
    temperature=0.1,
    max_tokens=20
)
print(f"  Content: {repr(r3.choices[0].message.content)}")
