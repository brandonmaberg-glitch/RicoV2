# RicoV2

## Local Memory System

Rico now uses a fully-local SQLite-backed memory service.

### What is stored
- **Conversation log**: every turn in `conversation_log`.
- **STM**: short-term memory in `short_term_memory` with TTL expiry sweep (default 48h).
- **LTM**: persistent memory in `long_term_memory` with `frequency`, `last_accessed_at`, and optional `pinned`.
- **Rolling summary**: compact context in `summary_state`.

### Retrieval and scoring
Memory retrieval is hybrid:
1. FTS5 keyword retrieval from `memory_fts`.
2. Embedding similarity retrieval.
3. Merge + rerank with weighted score:

`score = 0.55*similarity + 0.20*importance + 0.15*recency + 0.10*frequency + pinned_bonus`

Adjust in `Config.memory_scoring_weights` and `retrieval/rerank.py`.

### Classifier behavior
- Rules-first classifier in `memory/classifier/rules.py`.
- Optional local LLM judge fallback in `memory/classifier/llm_judge.py` (toggle `memory_use_llm_judge`).
- Labels: `none`, `stm_task`, `stm_thread`, `ltm_fact`, `ltm_preference`, `ltm_profile`, `pin`.

### Embeddings backend
Switch with `Config.embeddings_backend`:
- `sbert` (default): `sentence-transformers/all-MiniLM-L6-v2`
- `ollama`: local Ollama embeddings endpoint

### Debugging top memories
Use `MemoryService.debug_top_memories(query)` to inspect ranked memory candidates.
