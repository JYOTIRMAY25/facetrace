'use client';

import { useEffect, useRef, useState } from 'react';

interface StageResult { stage: string; status: string; duration_ms: number; }
interface PipelineResponse {
  status?: string; message?: string; error?: string; stages?: StageResult[];
  match?: { post?: { source: string; url: string; title: string; match_score: number } };
  fingerprint?: { hash: string; algorithm: string };
  record?: { network: string; record_id: string; transaction_hash: string; block_number: number };
  verification?: { match: boolean; status: string };
}

const steps = ['UPLOAD', 'DETECT', 'SEARCH', 'MATCH', 'EVIDENCE', 'SHA-256', 'BLOCKCHAIN', 'VERIFIED'];

function BackgroundMotion() {
  const points = [[8, 22], [21, 68], [36, 31], [49, 78], [64, 18], [78, 55], [91, 28], [87, 84], [29, 90]];
  return <div className="motion-background" aria-hidden="true">
    <div className="motion-grid" />
    <svg className="motion-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
      <g className="ring-system">
        <circle className="ring ring-one" cx="76" cy="28" r="17" />
        <circle className="ring ring-two" cx="76" cy="28" r="25" />
        <circle className="ring ring-three" cx="23" cy="76" r="19" />
      </g>
      <g className="signal-paths">
        <path d="M8 22 C18 22 22 68 36 31 S64 18 78 55 S87 84 91 28" />
        <path d="M21 68 L49 78 L87 84" />
        <path className="signal" d="M8 22 C18 22 22 68 36 31 S64 18 78 55 S87 84 91 28" />
      </g>
      <g className="face-vector">
        <rect x="57" y="22" width="25" height="34" rx="1" />
        <path d="M61 28h5M61 28v5M78 28h5M83 28v5M61 50v5M61 55h5M83 50v5M78 55h5" />
        <circle cx="65" cy="36" r="2.2" /><circle cx="75" cy="36" r="2.2" />
        <path d="M70 38l-2 6h4M64 48c4 2 8 2 12 0" />
      </g>
      <g className="motion-points">{points.map(([cx, cy], index) => <circle key={`${cx}-${cy}`} className={`point point-${index + 1}`} cx={cx} cy={cy} r=".45" />)}</g>
    </svg>
    <div className="motion-label label-one">SCAN // 001</div>
    <div className="motion-label label-two">VECTOR ANALYSIS</div>
    <div className="motion-label label-three">NODE ACTIVE</div>
    <div className="motion-label label-four">X: 042.81&nbsp;&nbsp; Y: 018.42</div>
    <div className="motion-label label-five">SIGNAL DETECTED</div>
  </div>;
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null), [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false), [result, setResult] = useState<PipelineResponse | null>(null);
  const [verifying, setVerifying] = useState(false), [verifyResult, setVerifyResult] = useState<{ match?: boolean } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [displayedScore, setDisplayedScore] = useState(0);
  const [showPreloader, setShowPreloader] = useState(true);
  const [startingInvestigation, setStartingInvestigation] = useState(false);
  const [investigationMode, setInvestigationMode] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [resultReady, setResultReady] = useState(false);
  const [resultTransition, setResultTransition] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const selectFile = (selected: File) => {
    if (!selected.type.startsWith('image/')) return;
    setFile(selected); setPreview(URL.createObjectURL(selected)); setResult(null); setVerifyResult(null); setAnalysisError(null);
  };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => { if (e.target.files?.[0]) selectFile(e.target.files[0]); };
  const runInvestigation = async () => {
    if (!file) return; setLoading(true); setAnalysisError(null); setResult(null);
    const formData = new FormData(); formData.append('file', file);
    try {
      const res = await fetch('/api/investigate', { method: 'POST', body: formData });
      const data: PipelineResponse = await res.json();
      if (!res.ok) throw new Error(data.error || data.message || `Request failed (${res.status})`);
      setResult(data);
      const canReveal = Boolean(data.match?.post || data.verification?.match === true);
      setResultTransition(canReveal);
      if (canReveal) {
        window.setTimeout(() => {
          setResultTransition(false);
          setResultReady(true);
        }, 1050);
      } else {
        setResultReady(true);
      }
    } catch (err) {
      setAnalysisError(err instanceof Error && err.message.startsWith('Request failed')
        ? 'The evidence service could not complete this analysis.'
        : 'The evidence service is unavailable.');
    } finally { setLoading(false); }
  };
  const reverifyOnChain = async () => {
    if (!result?.match?.post || !result.record) return; setVerifying(true);
    try {
      const res = await fetch('/api/verify', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ post: result.match.post, record: result.record }) });
      setVerifyResult(await res.json());
    } finally { setVerifying(false); }
  };
  const active = (step: string) => Boolean(result?.stages?.some(s => s.stage === step));
  const isVerified = result?.verification?.match === true || result?.verification?.status?.toUpperCase() === 'VERIFIED' || result?.status?.toUpperCase() === 'VERIFIED';

  useEffect(() => {
    const target = result?.match?.post?.match_score;
    if (target === undefined) {
      setDisplayedScore(0);
      return;
    }
    const end = target * 100;
    const start = performance.now();
    let frame = 0;
    const animate = (now: number) => {
      const progress = Math.min((now - start) / 900, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayedScore(end * eased);
      if (progress < 1) frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [result?.match?.post?.match_score]);

  useEffect(() => {
    const hasLoaded = window.sessionStorage.getItem('facetrace-vector-engine-ready');
    if (hasLoaded) {
      setShowPreloader(false);
      return;
    }
    const timer = window.setTimeout(() => {
      window.sessionStorage.setItem('facetrace-vector-engine-ready', 'true');
      setShowPreloader(false);
    }, 1700);
    return () => window.clearTimeout(timer);
  }, []);

  const startInvestigation = () => {
    setInvestigationMode(true);
    setStartingInvestigation(true);
    window.setTimeout(() => {
      setStartingInvestigation(false);
      document.getElementById('input')?.scrollIntoView({ behavior: 'smooth' });
    }, 850);
  };
  const startNewInvestigation = () => {
    setFile(null); setPreview(null); setResult(null); setVerifyResult(null); setAnalysisError(null); setCopied(false); setResultReady(false); setResultTransition(false); setShowDetails(false);
    setInvestigationMode(true);
    document.getElementById('input')?.scrollIntoView({ behavior: 'smooth' });
  };
  const copyFingerprint = async () => {
    if (!result?.fingerprint?.hash) return;
    await navigator.clipboard.writeText(result.fingerprint.hash);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  };
  const stageStatus = (step: string) => result?.stages?.find(s => s.stage === step)?.status?.toUpperCase();
  const stageActive = (step: string) => loading && !result || active(step);
  const resultMessage = result?.status?.toUpperCase().includes('FACE') ? 'No usable face was detected in the provided source.' : 'No sufficiently supported candidate was found.';

  return <main className="site"><BackgroundMotion /><div className="grain" />
    {showPreloader && <div className="preloader" role="status" aria-label="Loading FaceTrace"><div className="preloader-inner"><div className="brand">FACETRACE AI</div><div className="preloader-status">INITIALIZING VISUAL EVIDENCE</div><div className="loader-track"><span /></div><div className="preloader-status">LOADING VECTOR ENGINE&nbsp;&nbsp; <b>82%</b></div><div className="preloader-ready">EVIDENCE SYSTEM READY</div></div></div>}
    {startingInvestigation && <div className="transition-screen" role="status"><div className="transition-inner"><span>SYSTEM READY</span><span>↓</span><span>INITIALIZING SCAN</span><span>↓</span><span>CAPTURE SOURCE</span><span>↓</span><strong>VECTOR ANALYSIS</strong></div></div>}
    {resultTransition && <div className="result-transition" role="status"><div><span>ANALYSIS COMPLETE</span><b>↓</b><span>EVIDENCE CHAIN</span><b>↓</b><strong>VERIFICATION READY</strong></div></div>}
    <div className="wrap">
    <header className="header"><a className="brand" href="#top">FACETRACE AI</a><nav className="nav" aria-label="Primary navigation"><a href="#input">INVESTIGATE</a><a href="#pipeline">HOW IT WORKS</a><a href="#evidence">EVIDENCE</a></nav><div className="status"><span className="pulse">●</span>SYSTEM ONLINE</div></header>
    <section className={`hero reveal ${investigationMode ? 'investigation-hero' : ''}`} id="top"><div className="eyebrow">{investigationMode ? 'FACETRACE / INVESTIGATION MODE' : 'AI VISUAL INVESTIGATION SYSTEM'}</div><h1><span>TRACE</span><span>THE</span><span>FACE.</span></h1><div className="hero-copy">{investigationMode ? '01 / SOURCE CAPTURE' : <>Find genuine visual evidence.<br />Compare candidates.<br />Verify evidence provenance.</>}<div className="actions"><button className="btn" onClick={startInvestigation}>{investigationMode ? 'CAPTURE SOURCE ↓' : 'START INVESTIGATION →'}</button></div></div></section>
    <section className="section" id="input"><div className="section-head"><div className="number">01</div><div className="section-content"><div className="section-kicker">SOURCE / UPLOAD IMAGE</div><h2 className="section-title">{investigationMode ? 'UPLOAD IMAGE' : 'SELECT FACE EVIDENCE'}</h2><div className={`upload ${dragging ? 'dragging' : ''}`} role="button" tabIndex={0} aria-label="Select visual source" onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); inputRef.current?.click(); } }} onClick={() => inputRef.current?.click()} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]); }}><div className="upload-brackets" aria-hidden="true" /><div className={loading ? 'upload-copy scanning' : 'upload-copy'}><input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp" onChange={handleFileChange} /><strong>{loading ? 'FACE DETECTION ACTIVE' : file ? 'SOURCE CAPTURED' : investigationMode ? 'DROP VISUAL SOURCE TO BEGIN' : 'DROP IMAGE HERE'}</strong><span>{loading ? 'SEARCHING VISUAL SOURCES...' : file ? 'READY TO ANALYZE →' : 'or&nbsp;&nbsp; UPLOAD →'}</span><span>{loading ? 'LANDMARKS ANALYZING / VECTOR ENGINE RUNNING' : 'PNG / JPG / JPEG / WEBP'}</span></div></div>{preview && <div className="preview"><div className={`preview-frame ${loading ? 'is-scanning' : ''}`}><img src={preview} alt="Selected evidence preview" /><span className="detection-brackets" aria-hidden="true" /><span className="image-meta">SOURCE / {file?.name?.slice(0, 24)}</span></div><button className="btn" onClick={runInvestigation}>{loading ? 'ANALYZING...' : 'RUN INVESTIGATION →'}</button>{!loading && result && <span className="detected">FACE DETECTED ✓</span>}</div>}{analysisError && <div className="analysis-error"><b>ERROR // 001</b><strong>ANALYSIS INTERRUPTED</strong><span>{analysisError}</span><button className="btn ghost" onClick={runInvestigation}>RETRY ANALYSIS →</button></div>}</div></div></section>
    {result && <><section className={`result-hero ${resultReady ? 'result-visible' : ''}`}><div className="result-kicker">FACETRACE AI / INVESTIGATION // COMPLETE</div><div className="result-hero-layout"><div><h2>{result.match?.post ? <>MATCH<br /><em>FOUND</em></> : <>ANALYSIS<br /><em>INTERRUPTED</em></>}</h2><p>{isVerified ? 'EVIDENCE VERIFIED' : 'EVIDENCE CHAIN INCOMPLETE'}</p></div><div className="result-badge"><b>● {isVerified ? 'EVIDENCE VERIFIED' : 'VERIFICATION PENDING'}</b><span>FINGERPRINT {result.fingerprint ? 'MATCH' : 'PENDING'}</span><span>BLOCKCHAIN {result.record ? 'CONFIRMED' : 'PENDING'}</span><span>VERIFICATION {isVerified ? 'PASS' : 'INCOMPLETE'}</span></div></div></section><section className="section" id="pipeline"><div className="section-head"><div className="number">02</div><div className="section-content"><div className="section-kicker">INVESTIGATION / LIVE STATE</div><h2 className="section-title">PIPELINE</h2><div className="pipeline">{steps.map((step, i) => <div className={`step ${stageActive(step) ? 'active' : ''}`} key={step}><b>0{i + 1}</b>{step}<small>{stageStatus(step) || (loading ? 'RUNNING' : 'PENDING')}</small></div>)}</div></div></div></section>
      {!result.match?.post && <section className="section"><div className="analysis-error"><b>ERROR // 002</b><strong>{result.status?.toUpperCase() || 'NO MATCH FOUND'}</strong><span>{resultMessage}</span><button className="btn ghost" onClick={startNewInvestigation}>RETRY →</button></div></section>}
      {result.match?.post && <section className="section result-section" id="evidence"><div className="section-head"><div className="number">03</div><div className="section-content"><div className="section-kicker">EVIDENCE / CANDIDATE 001</div><h2 className="section-title">MATCH FOUND</h2><div className="evidence-grid"><div className="card"><div className="card-label">MATCH SCORE</div><div className="score">{displayedScore.toFixed(2)}%</div><div className="evidence-status">EVIDENCE STATUS / VERIFIED</div></div><div className="card"><div className="card-label">SOURCE REFERENCE</div><div className="source">{result.match.post.title || result.match.post.source}</div><div className="url">{result.match.post.url}</div></div></div><button className="details-toggle" onClick={() => setShowDetails(!showDetails)} aria-expanded={showDetails}>{showDetails ? 'HIDE EVIDENCE DETAILS ↑' : 'VIEW EVIDENCE DETAILS ↓'}</button>{showDetails && <div className="details-panel"><span>SOURCE</span><strong>{result.match.post.source}</strong><span>CANDIDATE</span><strong>{result.match.post.title || result.match.post.source}</strong><span>SIMILARITY</span><strong>{(result.match.post.match_score * 100).toFixed(2)}%</strong><span>URL / SOURCE REFERENCE</span><strong className="url">{result.match.post.url}</strong></div>}</div></div></section>}
      {result.fingerprint && <section className="section result-section"><div className="section-head"><div className="number">04</div><div className="section-content"><div className="section-kicker">PROVENANCE</div><h2 className="section-title">SHA-256 / BLOCKCHAIN</h2><div className="provenance"><div className="card hash-card"><div className="card-label">EVIDENCE FINGERPRINT / SHA-256</div><p className="hash hash-reveal">{result.fingerprint.hash}</p><div className="hash-actions"><span>FINGERPRINT STABLE</span><button className="copy-btn" onClick={copyFingerprint}>{copied ? 'COPIED ✓' : 'COPY HASH →'}</button></div></div>{result.record && <div className="card blockchain-card"><div className="card-label">BLOCKCHAIN RECORD</div><div className="chain-flow"><span>SOURCE</span><i /> <span>SHA-256</span><i /> <span>BLOCK</span><i /> <span>READ-BACK</span><i /> <span>VERIFIED</span></div><div className="record"><div><p>NETWORK</p><strong>{result.record.network}</strong></div><div><p>BLOCK</p><strong>{result.record.block_number}</strong></div><div style={{ gridColumn: '1 / -1' }}><p>TRANSACTION</p><strong>{result.record.transaction_hash}</strong></div></div></div>}</div><p className="disclaimer">Verification confirms evidence fingerprint and provenance.<br />It does not establish legal identity.</p></div></div></section>}
      {isVerified ? <section className="final verified"><div className="verification-mark"><svg viewBox="0 0 100 100" aria-label="Evidence verified"><circle cx="50" cy="50" r="43" /><path d="M28 51 43 66 73 34" /></svg></div><h2>EVIDENCE<br />VERIFIED.</h2><div className="final-notes"><span>SHA-256 MATCH</span><span>BLOCKCHAIN RECORD MATCH</span></div>{result.record && <div className="actions" style={{ justifyContent: 'center' }}><button className="btn ghost" onClick={reverifyOnChain}>{verifying ? 'VERIFYING...' : 'RE-VERIFY RECORD →'}</button><button className="btn" onClick={startNewInvestigation}>START NEW INVESTIGATION →</button></div>}{verifyResult && <p className="meta">{verifyResult.match ? 'EVIDENCE VERIFIED / BLOCKCHAIN RECORD CONFIRMED' : 'RECORD CHECK COMPLETE'}</p>}</section> : result.match?.post && <section className="verification-failure"><b>VERIFICATION FAILED</b><strong>EVIDENCE CHAIN INCOMPLETE</strong><span>Verification did not confirm the evidence provenance.</span><button className="btn ghost" onClick={reverifyOnChain}>{verifying ? 'VERIFYING...' : 'RETRY ANALYSIS →'}</button></section>}
    </>}
    <footer className="footer"><div><strong>FACETRACE AI</strong><br />TRACE THE FACE.<br />VERIFY THE EVIDENCE.</div><div>VISUAL INVESTIGATION × CRYPTOGRAPHIC PROVENANCE</div></footer>
  </div></main>;
}
