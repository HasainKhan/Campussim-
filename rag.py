"""Small, dependency-free retrieval layer: chunking, BM25 ranking, rule-conflict check."""
import io, math, re
from collections import Counter

STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with shall may not any all each per".split())

def tok(s):
    return [w for w in re.findall(r"[a-z0-9%]+", s.lower()) if w not in STOP and len(w) > 1]

def chunk(text, size=110, overlap=20):
    out = []
    for p in (x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()):
        w = p.split()
        if len(w) <= size: out.append(p); continue
        i = 0
        while i < len(w):
            out.append(" ".join(w[i:i + size])); i += size - overlap
    return out

def extract_text(filename, data):
    name = filename.lower()
    if name.endswith((".txt", ".md")): return data.decode("utf-8", "replace")
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(data)).pages)
    raise ValueError("unsupported file type")

def search(query, docs, k=4):
    """BM25 over docs = [{id,title,text}]. Returns top-k with a score."""
    if not docs: return []
    toks = [tok(d["text"]) for d in docs]; N = len(docs)
    avg = (sum(len(t) for t in toks) / N) or 1
    df = Counter(w for t in toks for w in set(t)); q = set(tok(query)); scored = []
    for d, t in zip(docs, toks):
        tf, s = Counter(t), 0.0
        for w in q:
            if w in tf:
                s += math.log(1 + (N - df[w] + .5) / (df[w] + .5)) * tf[w] * 2.2 / (tf[w] + 1.2 * (.25 + .75 * len(t) / avg))
        if s > 0: scored.append((s, d))
    return [dict(d, score=round(s, 2)) for s, d in sorted(scored, key=lambda x: -x[0])[:k]]

def pcts(s): return {int(x) for x in re.findall(r"(\d{1,3})\s*%", s)}

def find_conflicts(policy, clauses):
    """Flag clauses on the same topic whose percentages differ from the draft's."""
    mine, out = pcts(policy), []
    for c in clauses:
        theirs = pcts(c["text"])
        if mine and theirs and (theirs - mine) and len(set(tok(policy)) & set(tok(c["text"]))) >= 2:
            out.append(dict(title=c["title"], existing=sorted(theirs), draft=sorted(mine)))
    return out
