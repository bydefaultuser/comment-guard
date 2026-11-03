import { useEffect, useState } from "react";

const EXAMPLES = [
  "Great product, arrived early and works perfectly!",
  "Honestly the worst support I've ever dealt with. Nobody replies.",
  "Oh wonderful, it broke again on day two. Just brilliant.",
  "You are a useless idiot and this app is crap.",
];

const SWEAR_LABELS = ["toxic", "obscene"];

const VERDICTS = {
  clean: { label: "Clean", hint: "Looks fine" },
  negative: { label: "Negative feedback", hint: "Complaint or unhappy tone" },
  abusive: { label: "Abusive", hint: "Hidden from other users" },
};

export default function App() {
  const [health, setHealth] = useState(null);
  const [text, setText] = useState("");
  const [useLlm, setUseLlm] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [comments, setComments] = useState([]);

  useEffect(() => {
    let timer;
    const poll = async () => {
      try {
        const res = await fetch("/api/health");
        const body = await res.json();
        setHealth(body);
        if (body.status !== "ok") timer = setTimeout(poll, 2000);
      } catch {
        setHealth({ status: "offline" });
        timer = setTimeout(poll, 3000);
      }
    };
    poll();
    return () => clearTimeout(timer);
  }, []);

  const post = async (e) => {
    e.preventDefault();
    if (!text.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, use_llm: useLlm && health?.llm_configured }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Request failed");
      setComments((prev) => [{ id: crypto.randomUUID(), ...body }, ...prev]);
      setText("");
    } catch (err) {
      setError(err.message || "Could not reach the backend.");
    } finally {
      setBusy(false);
    }
  };

  const ready = health?.status === "ok";

  return (
    <main className="page">
      <header className="top">
        <div>
          <h1>Comment Guard</h1>
          <p className="muted">Every comment is checked for bad words, toxicity and negative feedback before it is posted.</p>
        </div>
        <StatusChips health={health} />
      </header>

      <form className="composer" onSubmit={post}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Write a comment..."
          rows={3}
          maxLength={5000}
          onKeyDown={(e) => (e.ctrlKey || e.metaKey) && e.key === "Enter" && post(e)}
        />
        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button type="button" key={ex} className="example" onClick={() => setText(ex)}>
              {ex}
            </button>
          ))}
        </div>
        <div className="composer-row">
          <label className={`toggle ${health?.llm_configured ? "" : "disabled"}`}>
            <input
              type="checkbox"
              checked={useLlm && !!health?.llm_configured}
              disabled={!health?.llm_configured}
              onChange={(e) => setUseLlm(e.target.checked)}
            />
            Also ask the LLM {health?.llm_configured ? "" : "(set a Bedrock model and AWS credentials to enable)"}
          </label>
          <button type="submit" className="primary" disabled={!ready || busy || !text.trim()}>
            {busy ? "Checking..." : "Post comment"}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </form>

      <section className="feed">
        {comments.length === 0 ? (
          <p className="muted empty">Posted comments will appear here with their verdict.</p>
        ) : (
          comments.map((c) => <Comment key={c.id} result={c} />)
        )}
      </section>
    </main>
  );
}

function StatusChips({ health }) {
  if (!health) return <div className="chips"><span className="chip">Connecting...</span></div>;
  if (health.status === "offline") return <div className="chips"><span className="chip bad">Backend offline</span></div>;
  return (
    <div className="chips">
      <span className={`chip ${health.status === "ok" ? "good" : ""}`}>
        {health.status === "ok" ? "Models ready" : "Loading models..."}
      </span>
      <span className="chip">{health.device === "cuda" ? `GPU: ${health.gpu_name}` : "CPU"}</span>
      <span className={`chip ${health.llm_configured ? "good" : ""}`}>
        LLM: {health.llm_configured ? health.llm_model : "off"}
      </span>
    </div>
  );
}

const DECIDED_BY = {
  llm: "decided by the LLM",
  "llm+toxicity-override": "toxicity model overruled the LLM",
  rules: "decided by word list + BERT",
};

function Comment({ result }) {
  const [reveal, setReveal] = useState(false);
  const [open, setOpen] = useState(false);
  const v = VERDICTS[result.verdict];
  const profane = result.flags.includes("profanity");
  const censorable = result.verdict === "abusive" || profane;
  const censored = censorable && !reveal;

  return (
    <article className={`comment ${result.verdict}`}>
      <div className="comment-head">
        <span className={`badge ${result.verdict}`}>{v.label}</span>
        {profane && <span className="badge profanity">Profanity</span>}
        <span className="muted small">
          {v.hint} · {DECIDED_BY[result.decided_by]} · {result.elapsed_ms} ms on {result.device.toUpperCase()}
        </span>
      </div>

      <p className={`comment-text ${result.verdict === "abusive" && censored ? "hidden" : ""}`}>
        {censored ? result.profanity.censored_text : result.text}
      </p>

      <ul className="reasons">
        {result.reasons.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>

      <div className="comment-actions">
        {censorable && (
          <button className="link" onClick={() => setReveal(!reveal)}>
            {reveal ? "Hide original" : "Show original"}
          </button>
        )}
        <button className="link" onClick={() => setOpen(!open)}>
          {open ? "Hide details" : "Show details"}
        </button>
      </div>

      {open && <Details result={result} />}
    </article>
  );
}

function Details({ result }) {
  const { profanity, toxicity, sentiment, llm } = result;
  return (
    <div className="details">
      <div className="layer">
        <h3>1 · Bad-word filter</h3>
        <p>{profanity.found ? `Found: ${profanity.words.join(", ") || "yes"}` : "No bad words found."}</p>
      </div>

      <div className="layer">
        <h3>2a · Toxicity model <span className="muted small">unitary/toxic-bert</span></h3>
        <p className="muted small hint">
          Toxic and obscene measure swearing; insult, threat, identity hate and severe toxic measure abuse.
        </p>
        <Bars scores={toxicity.scores} highlight={(s, k) => (s < 0.5 ? "" : SWEAR_LABELS.includes(k) ? "warm" : "hot")} />
      </div>

      <div className="layer">
        <h3>2b · Sentiment model <span className="muted small">twitter-roberta-base-sentiment</span></h3>
        <Bars scores={sentiment.scores} highlight={(s, k) => (k === "negative" && s >= 0.6 ? "warm" : "")} />
      </div>

      <div className="layer">
        <h3>3 · LLM <span className="muted small">{llm.model || ""}</span></h3>
        {llm.status === "ok" ? (
          <>
            <p>
              <strong>{llm.judgement.verdict.replace("_", " ")}</strong>
              {llm.judgement.categories.length > 0 && ` · ${llm.judgement.categories.join(", ")}`}
            </p>
            <p className="muted">{llm.judgement.reason}</p>
          </>
        ) : (
          <p className={llm.status === "error" ? "error" : "muted"}>{llm.message}</p>
        )}
      </div>
    </div>
  );
}

function Bars({ scores, highlight }) {
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  return (
    <div className="bars">
      {entries.map(([label, score]) => (
        <div className="bar-row" key={label}>
          <span className="bar-label">{label.replace("_", " ")}</span>
          <span className="bar-track">
            <span className={`bar-fill ${highlight(score, label)}`} style={{ width: `${score * 100}%` }} />
          </span>
          <span className="bar-value">{Math.round(score * 100)}%</span>
        </div>
      ))}
    </div>
  );
}
