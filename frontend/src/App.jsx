
import { useState } from "react";
import { ShieldCheck, Search, Newspaper, RotateCcw, LoaderCircle, Sparkles } from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [newsText, setNewsText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (!newsText.trim()) {
      setError("Please enter some news text to analyse.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: newsText }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed. Please try again.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setNewsText("");
    setResult(null);
    setError("");
  }

  const isFake = result?.veracity === "Fake";
  const isAI = result?.authorship === "AI";

  return (
    <div className="app">
      <header className="navbar">
        <a className="brand" href="#">
          <span className="brand-icon"><ShieldCheck size={23} /></span>
          <span>VeriText<span className="brand-dot">.</span></span>
        </a>
        <span className="nav-label">RESEARCH PROTOTYPE</span>
      </header>

      <main>
        <section className="hero">
          <h1>Look beyond the <span>headline.</span></h1>
          <p className="hero-description">
            Analyse news content to estimate its veracity and authorship origin
            using linguistic and statistical text features.
          </p>
        </section>

        <section className="workspace">
          <div className="section-heading">
            <div>
              <h2>News analyser</h2>
              <p>Paste an article or a news excerpt below.</p>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <label htmlFor="news-text">NEWS CONTENT</label>
            <textarea
              id="news-text"
              value={newsText}
              onChange={(event) => setNewsText(event.target.value)}
              placeholder="Paste the news article you want to analyse..."
              rows={9}
            />
            <div className="input-footer">
              <span>{newsText.trim() ? newsText.trim().split(/\s+/).length : 0} words</span>
              <button type="button" className="text-button" onClick={resetForm}>
                <RotateCcw size={14} /> Clear
              </button>
            </div>
            <button className="analyse-button" type="submit" disabled={loading}>
              {loading ? <LoaderCircle className="spin" size={18} /> : <Search size={18} />}
              {loading ? "Analysing article..." : "Analyse news"}
            </button>
          </form>

          {error && <div className="error-message">{error}</div>}

          {result && (
            <section className="results" aria-live="polite">
              <div className="results-title">
                <div className="result-icon"><Newspaper size={21} /></div>
                <div>
                  <h2>Analysis results</h2>
                  <p>Predictions from the four-class classification model</p>
                </div>
              </div>

              <div className="prediction-banner">
                <span className="result-kicker">PREDICTED CLASS</span>
                <h3>{result.prediction}</h3>
                <p>Estimated class probability: <strong>{result.confidence.toFixed(2)}%</strong></p>
                <div className="confidence-track">
                  <div className="confidence-fill" style={{ width: `${result.confidence}%` }} />
                </div>
              </div>

              <div className="result-grid">
                <article className="result-card">
                  <span className="result-kicker">VERACITY</span>
                  <h3 className={isFake ? "fake" : "real"}>{result.veracity}</h3>
                  <p>{isFake ? "The model classified this article as fake news." : "The model classified this article as real news."}</p>
                </article>
                <article className="result-card">
                  <span className="result-kicker">AUTHORSHIP ORIGIN</span>
                  <h3>{result.authorship}</h3>
                  <p>{isAI ? "The model classified this text as AI-authored." : "The model classified this text as human-authored."}</p>
                </article>
              </div>

              <p className="disclaimer">
                This is a machine-learning prediction, not independent verification
                of the article's claims or definitive proof of its authorship.
              </p>
              <button className="secondary-button" onClick={resetForm}>
                Analyse another article
              </button>
            </section>
          )}
        </section>

        <section className="method-note">
          <div className="method-icon"><ShieldCheck size={20} /></div>
          <div>
            <h3>About this research</h3>
            <p>
              This prototype classifies text into Real/Human, Fake/Human,
              Real/AI, or Fake/AI categories. Its current backend uses a tuned
              TF-IDF and Multinomial Naive Bayes pipeline.
            </p>
          </div>
        </section>
      </main>

      <footer>
        <span>VERITEXT <span className="brand-dot">.</span></span>
        <span>Academic research prototype · Not a fact-checking authority</span>
      </footer>
    </div>
  );
}

export default App;
