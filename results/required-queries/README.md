# Required-query results

## Recorded configuration

| Setting | Value |
|---|---|
| embedding_model | `Qwen3-Embedding-8B` |
| llm_model | `Qwen3.8-Flash-Next` |
| collection | `food_safety_corpus` |
| chunk_size_chars | `1200` |
| chunk_overlap_chars | `200` |
| retrieval_top_k | `5` |
| max_steps | `10` |
| temperature | `0.0` |
| tool_choice | `auto` |
| query_instruction | `Given a question about food safety, retrieve passages from food-safety documents that answer it` |
| index_chunk_count | `3030` |
| index_built_at | `2026-10-27T00:42:25+00:00` |

## Outcomes

| Query | Expected | Terminal reason | Steps | Citations | Match |
|---|---|---|---:|---:|---|
| [Q1](Q1.md) | answer | answer | 6 | 6 | yes |
| [Q2](Q2.md) | answer | answer | 5 | 10 | yes |
| [Q3](Q3.md) | answer | answer | 6 | 7 | yes |
| [Q4](Q4.md) | answer | answer | 2 | 4 | yes |
| [Q5](Q5.md) | answer | answer | 6 | 11 | yes |
| [Q6](Q6.md) | answer | answer | 8 | 23 | yes |
| [Q7](Q7.md) | answer | answer | 8 | 15 | yes |
| [Q8](Q8.md) | insufficient_evidence | insufficient_evidence | 5 | 5 | yes |
| [Q9](Q9.md) | insufficient_evidence | insufficient_evidence | 2 | 2 | yes |
