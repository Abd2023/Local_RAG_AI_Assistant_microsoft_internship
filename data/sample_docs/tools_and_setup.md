# Northstar AI Summer School Tools And Setup

Students use Python 3.11 or newer for the project. The recommended editor is Visual Studio Code, but any editor is allowed if the student can run commands from the project directory and inspect files easily.

The project should use a virtual environment named `.venv`. Dependencies should be listed in `requirements.txt` so another student can recreate the environment. The baseline dependencies are the Foundry Local SDK for Windows, OpenAI client types, NumPy, Pytest, and Rich.

Microsoft Foundry Local is used to run models on the student device. The default chat model for the course is `qwen2.5-0.5b`, and the default embedding model is `qwen3-embedding-0.6b`. On machines with a supported NVIDIA GPU, teams should prefer CUDA GPU variants when available.

The first setup test is a local chat prompt that asks the model to say hello in one sentence. The second setup test is an embedding call that converts a sentence into a numeric vector. Both tests must run through Foundry Local rather than a cloud API.

Teams should keep generated databases and virtual environments out of Git. The `.venv` folder, Python cache folders, and `data/rag.db` should be ignored. Source code, documentation, sample documents, and tests should be committed.

If model download fails, students should capture the exact error message, confirm network access for the initial download, and check whether the configured model alias exists in the Foundry Local catalog. After the model is cached, normal demo runs should work offline.

