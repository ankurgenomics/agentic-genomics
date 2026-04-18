"""Bio-database tools used by the variant interpreter agent.

Each tool wraps a single public API, caches aggressively, and returns
typed Pydantic models. No tool here invokes an LLM — tools are for
deterministic fact retrieval.
"""
