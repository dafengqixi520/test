# Test scope

The current paper chapter-3 implementation is covered by:

- `test_paper_alignment.py`: formula, topology, resource-capacity, and real training checks.
- `test_per_ddpg.py`: lightweight integration smoke test.

`test_env.py`, `test_full_llm.py`, `test_llm.py`, `test_prompt.py`, and most of
`test_frontend.py` refer to an older LeDRL/LLM project API. They are retained as
historical files and are not evidence that this PER-DDPG implementation passes.

Run the current regression suite with:

```powershell
python -m unittest tests.test_paper_alignment -v
```
