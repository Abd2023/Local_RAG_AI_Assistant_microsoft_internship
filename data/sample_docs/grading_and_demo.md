# Northstar AI Summer School Grading And Demo

The final project grade is based on four categories. RAG pipeline correctness is worth 40 percent, code quality is worth 25 percent, evaluation and responsible behavior are worth 20 percent, and the final demo is worth 15 percent.

For pipeline correctness, the assistant must ingest documents, create embeddings, store chunks in SQLite, retrieve relevant context, and generate an answer from that context. A project that only sends the user question directly to the model without retrieval does not meet the core requirement.

For code quality, the project should have clear modules, readable function names, and a README with accurate run instructions. Debug print statements should be removed or hidden behind a clear debug mode before the final demo.

For evaluation, each team must test at least ten questions. The set should include five answerable questions, three unanswerable questions, and two vague or edge-case questions. Teams should record the retrieved sources and judge whether each answer was acceptable.

The final demo is five minutes long. Teams should briefly explain the problem, show the assistant running, ask one answerable question, ask one unanswerable question, and explain what they would improve next. Every team member should speak during the demo.

Bonus credit is available for useful improvements after the required CLI version works. Examples include a simple Streamlit interface, PDF ingestion, better source citations, or a comparison between the direct Python implementation and a LangChain implementation.

