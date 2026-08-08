# Manual Evaluation

Evaluation date: 2026-08-08

This evaluation was run against the sample Northstar AI Summer School documents indexed in `data/rag.db`.
The database contained 6 stored chunks, one chunk per sample document.

Commands used:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -B -m src.rag "What time does the daily standup start?"
```

The full manual set was run through `src.rag.answer_query` with cross-encoder reranking and citation verification enabled. Each question recorded the model answer, vector candidates, reranked chunks, evidence verification, and any exception.

## Summary

- Answerable questions: 5 total, 5 pass.
- Unanswerable questions: 3 total, 3 pass.
- Vague or edge-case questions: 2 total, 2 pass.
- Final live report: `data/evaluations/evaluation-20260808-135202.json`.
- Main improvement: `Tell me about setup.` now ranks `tools_and_setup.md` first after reranking.

## Evaluation Questions

### A1 - Answerable

Question: What time does the daily standup start?

Expected behavior: Answer that the daily standup starts at 10:00 AM and cite `schedule_and_attendance.md`.

Actual answer:

> Based on the information provided in the given context, the daily standup begins at 10:00 AM.

Retrieved sources:

- `schedule_and_attendance.md` chunk 1, similarity 0.446
- `support_and_policies.md` chunk 1, similarity 0.388
- `course_overview.md` chunk 1, similarity 0.253

Judgment: Pass.

Notes for improvement: The top retrieved source was correct. The answer is grounded.

### A2 - Answerable

Question: How much of the final project grade is RAG pipeline correctness worth?

Expected behavior: Answer that RAG pipeline correctness is worth 40 percent and cite `grading_and_demo.md`.

Actual answer:

> RAG pipeline correctness is worth 40% of the final project grade.

Retrieved sources:

- `grading_and_demo.md` chunk 1, similarity 0.711
- `project_requirements.md` chunk 1, similarity 0.542
- `tools_and_setup.md` chunk 1, similarity 0.414

Judgment: Pass.

Notes for improvement: The top retrieved source was correct. The model gave a longer answer than requested, but the final fact was correct.

### A3 - Answerable

Question: What storage layer is required for the first version?

Expected behavior: Answer that SQLite is the required storage layer and cite `project_requirements.md`.

Actual answer:

> The required storage layer for the first version is SQLite.

Retrieved sources:

- `project_requirements.md` chunk 1, similarity 0.418
- `tools_and_setup.md` chunk 1, similarity 0.394
- `grading_and_demo.md` chunk 1, similarity 0.366

Judgment: Pass.

Notes for improvement: The top retrieved source was correct. Similarity scores were close, but the answer stayed grounded.

### A4 - Answerable

Question: What are the default chat and embedding models for the course?

Expected behavior: Answer that the chat model is `qwen2.5-0.5b`, the embedding model is `qwen3-embedding-0.6b`, and cite `tools_and_setup.md`.

Actual answer:

> The default chat model for the course is `qwen2.5-0.5b` and the default embedding model is `qwen3-embedding-0.6b`.

Retrieved sources:

- `tools_and_setup.md` chunk 1, similarity 0.614
- `grading_and_demo.md` chunk 1, similarity 0.572
- `project_requirements.md` chunk 1, similarity 0.540

Judgment: Pass.

Notes for improvement: The top retrieved source was correct. The answer is concise and grounded.

### A5 - Answerable

Question: What should students do if they are blocked for more than 30 minutes?

Expected behavior: Answer that students should ask for help after 30 minutes, first ask a teammate, then check course notes, then ask the instructor during lab time, and cite `support_and_policies.md`.

Actual answer:

> Students should ask for help when they are blocked for more than 30 minutes.

Retrieved sources:

- `support_and_policies.md` chunk 1, similarity 0.539
- `schedule_and_attendance.md` chunk 1, similarity 0.402
- `tools_and_setup.md` chunk 1, similarity 0.253

Judgment: Pass.

Notes for improvement: The top retrieved source was correct. The answer is accurate but incomplete because it omitted the preferred support path.

### U1 - Unanswerable

Question: Who is the instructor for Northstar AI Summer School?

Expected behavior: Say that the instructor's identity is not specified in the available documents.

Actual answer:

> I do not know based on the available documents.

Retrieved sources:

- `schedule_and_attendance.md` chunk 1, similarity 0.648
- `course_overview.md` chunk 1, similarity 0.620
- `tools_and_setup.md` chunk 1, similarity 0.602

Judgment: Pass.

Notes for improvement: The retrieved chunks mention an instructor but do not name one. A deterministic exact-detail guard correctly refused the answer despite high similarity scores.

### U2 - Unanswerable

Question: What is the street address of the classroom?

Expected behavior: Say that the street address is not available in the documents.

Actual answer:

> I do not know based on the available documents.

Retrieved sources:

- `support_and_policies.md` chunk 1, similarity 0.377
- `grading_and_demo.md` chunk 1, similarity 0.320
- `tools_and_setup.md` chunk 1, similarity 0.301

Judgment: Pass.

Notes for improvement: The answer correctly avoided inventing an address. The top similarity barely exceeded the 0.35 threshold, so this remains a useful case for tuning the minimum retrieval score.

### U3 - Unanswerable

Question: What exact calendar date is the final demo?

Expected behavior: Say that the exact calendar date is not specified, while optionally noting that demos happen on the last Friday of Week 4.

Actual answer:

> I do not know based on the available documents.

Retrieved sources:

- `grading_and_demo.md` chunk 1, similarity 0.413
- `schedule_and_attendance.md` chunk 1, similarity 0.385
- `support_and_policies.md` chunk 1, similarity 0.375

Judgment: Pass.

Notes for improvement: The source documents say final demos happen on the last Friday of Week 4, but they do not provide an exact calendar date. The exact-detail guard correctly refused the more specific request.

### E1 - Vague Or Edge Case

Question: Tell me about setup.

Expected behavior: Provide a short setup summary from `tools_and_setup.md`, such as Python 3.11+, `.venv`, `requirements.txt`, Foundry Local, and the two setup tests. Because the question is vague, a concise clarification or broad summary is acceptable.

Actual answer:

> # Northstar AI Summer School Tools And Setup The first setup test is a local chat prompt that asks the model to say hello in one sentence. [tools_and_setup.md#0]

Retrieved sources:

- `tools_and_setup.md` chunk 1, rerank score 3.296
- `support_and_policies.md` chunk 1, reranked below the setup source
- `project_requirements.md` chunk 1, reranked below the setup source

Judgment: Pass.

Notes for improvement: The small local chat model omitted citations, so the backend used a concise extractive fallback from the top reranked chunk and attached a validated citation.

### E2 - Vague Or Edge Case

Question: Empty input containing only spaces.

Expected behavior: The CLI should ignore empty input. Direct calls to `answer_query` should reject it with a clear validation error.

Actual answer:

> ValueError: Question must not be empty.

Retrieved sources:

- None.

Judgment: Pass.

Notes for improvement: The direct API behavior is clear. A separate CLI smoke test should confirm that blank terminal input is ignored without calling retrieval or the model.

## Improvements To Consider

- Continue expanding the post-retrieval evidence check for questions asking for exact names, dates, addresses, deadlines, or people.
- Raise or tune `RETRIEVAL_MIN_TOP_SCORE`, or add a score gap check between the top result and weak matches.
- Add a prompt rule that says: if the requested level of specificity is missing, state that it is missing before giving related information.
- Improve vague-query handling by asking a clarifying question when the query has too few meaningful keywords.
- Consider splitting larger sample documents into smaller chunks so retrieval ranks specific setup, grading, schedule, and policy facts more cleanly.
