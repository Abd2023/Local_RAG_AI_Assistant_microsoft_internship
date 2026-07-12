# Northstar AI Summer School Project Requirements

Each team must build a local RAG assistant that runs on a student laptop. The assistant must use a local embedding model to convert document chunks into vectors and a local chat model to generate answers. Internet access should not be required after the required models and packages have been downloaded.

The first version must use a command-line interface. The CLI should accept a user question, retrieve relevant chunks, generate an answer, and print the source names used. Streamlit, Gradio, Flask, or a full web interface are optional stretch goals only after the CLI version works.

The knowledge base must contain at least five small documents on one controlled topic. Documents should be split into readable chunks before embedding. Chunks should preserve their source file name and chunk index so answers can cite where the information came from.

SQLite is the required storage layer for the first version. The database should store chunk text, source metadata, and JSON-serialized embedding vectors. Teams should rebuild the database when ingestion is run again, so duplicate chunks do not accumulate during testing.

Retrieval must use cosine similarity for the first version. For the small sample dataset, it is acceptable to load all stored embeddings into memory and rank them in Python. Specialized vector databases are not required for the first project milestone.

The answer prompt must instruct the model to use only the retrieved context. If the retrieved context does not contain the answer, the assistant should say that it does not know based on the available documents. The assistant should not invent deadlines, policies, people, or technical details.

