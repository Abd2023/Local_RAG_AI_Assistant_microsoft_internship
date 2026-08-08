import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "unavailable" }));
  }, []);

  async function submitQuestion(event) {
    event.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail?.message || payload.detail || "The query failed.");
      }
      setResult(payload);
    } catch (caught) {
      setError(caught.message);
    } finally {
      setLoading(false);
    }
  }

  async function uploadDocument(event) {
    const file = event.target.files?.[0];
    if (!file || uploading) return;

    setUploading(true);
    setError("");
    setUploadMessage("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "The document upload failed.");
      }

      const index = payload.index || {};
      setUploadMessage(
        `${payload.document_name} is ready: ${index.chunks ?? 0} chunks indexed.`,
      );
      const healthResponse = await fetch("/api/health");
      if (healthResponse.ok) setHealth(await healthResponse.json());
    } catch (caught) {
      setError(caught.message);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Offline knowledge assistant</p>
          <h1>Local RAG Assistant</h1>
        </div>
        <div className="status" data-state={health?.status || "loading"}>
          <span className="status-dot" />
          {health?.status === "ok" ? `${health.provider} / ${health.vector_backend}` : "Checking service"}
        </div>
      </header>

      <section className="workspace">
        <form className="question-panel" onSubmit={submitQuestion}>
          <label htmlFor="question">Ask your documents</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What time does the daily standup start?"
            rows="5"
          />
          <div className="form-footer">
            <span>{health ? `${health.chunks ?? 0} chunks indexed` : "Local index loading"}</span>
            <button type="submit" disabled={!question.trim() || loading}>
              {loading ? "Searching..." : "Ask"}
            </button>
          </div>
          <div className="upload-section">
            <label htmlFor="document-upload">Add a document</label>
            <input
              id="document-upload"
              type="file"
              accept=".md,.txt,.pdf,.docx"
              onChange={uploadDocument}
              disabled={uploading}
            />
            {uploading && <small>Extracting, embedding, and indexing...</small>}
            {uploadMessage && <small className="success-text">{uploadMessage}</small>}
          </div>
        </form>

        <section className="answer-panel" aria-live="polite">
          {error && <div className="notice error">{error}</div>}
          {!error && !result && <div className="empty-state">Your grounded answer will appear here.</div>}
          {result && (
            <>
              <div className="answer-heading">
                <div>
                  <p className="eyebrow">Verified response</p>
                  <h2>Answer</h2>
                </div>
                <span className={result.verification?.verified ? "badge verified" : "badge failed"}>
                  {result.verification?.verified ? "Citations verified" : "Verification failed"}
                </span>
              </div>
              <p className="answer-text">{result.answer}</p>

              <div className="metadata-grid">
                <div><span>Trace</span><strong>{result.trace_id}</strong></div>
                <div><span>Ranking</span><strong>{result.reranker_status}</strong></div>
                <div><span>Claims</span><strong>{result.verification?.claims?.length ?? 0}</strong></div>
              </div>

              <h3>Retrieved evidence</h3>
              <div className="sources">
                {(result.retrieved_chunks || []).map((chunk) => (
                  <article className="source" key={chunk.chunk_uid || `${chunk.source_name}-${chunk.chunk_index}`}>
                    <div className="source-topline">
                      <strong>{chunk.source_name}</strong>
                      <span>{chunk.rerank_score ?? chunk.similarity}</span>
                    </div>
                    <p>{chunk.preview}</p>
                    <small>{chunk.ranking_method} · chunk {chunk.chunk_number}</small>
                  </article>
                ))}
              </div>
            </>
          )}
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
