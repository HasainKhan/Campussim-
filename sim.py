"""CampusSim engine: persona generation, RAG-grounded agent reactions, a peer-discussion round, scoring.
Backends: Anthropic API (ANTHROPIC_API_KEY), a local LoRA-tuned model (CAMPUSSIM_BACKEND=local), or a
transparent rule engine. The mode is recorded with every run."""
import json, os, random, re, urllib.request
from collections import Counter, defaultdict
import rag

DEPTS = ["MCA", "BCA", "B.Tech CSE", "MBA", "B.Sc", "B.Com"]
LABELS = {
 "attendance_gap": ("Attendance bar is above many students' current record",
                    "Add a medical/hardship condonation window and a one-time grace of 5 percentage points."),
 "commute_attendance": ("Long-distance day scholars struggle with strict attendance",
                    "Offer recorded/hybrid sessions for day scholars travelling more than 15 km."),
 "working_student": ("Working students cannot meet fixed schedules",
                    "Create an evening make-up slot and a documented work-study exemption."),
 "fee_burden": ("Fee changes hit low-income families hardest",
                    "Allow instalments and tie the change to scholarship/fee-waiver review."),
 "penalty": ("Penalties feel punitive without a warning step",
                    "Replace first-instance penalties with a written warning and a support meeting."),
 "early_timing": ("Early timings clash with long commutes",
                    "Shift the earliest session to 9:00 or provide transport-aligned batches."),
 "exam_pressure": ("Exam changes disadvantage students with lower CGPA",
                    "Publish the pattern one semester ahead and add remedial sessions."),
}

def personas(n, seed=7):
    r = random.Random(seed)
    return [dict(id=i, dept=r.choice(DEPTS), year=r.randint(1, 4), cgpa=round(r.uniform(5.5, 9.5), 1),
                 attendance=r.randint(55, 98),
                 income=r.choices(["low", "middle", "high"], [.3, .5, .2])[0],
                 commute=r.choice(["hostel", "<5km", "5-15km", ">15km"]),
                 works=r.random() < .18, lang=r.choice(["English", "Kannada", "Hindi", "Tamil", "Telugu"]))
            for i in range(n)]

def features(text):
    t = text.lower(); m = re.search(r"(\d{2})\s*%", t)
    return dict(att="attendance" in t, thr=int(m.group(1)) if m else None,
                fee=bool(re.search(r"\bfees?\b|payment|charge", t)),
                fine=bool(re.search(r"fine|penalt|debar|detain", t)),
                early=bool(re.search(r"\b(7|8)\s*(:00)?\s*(am|a\.m)|early|before 9", t)),
                exam="exam" in t or "assessment" in t,
                flex=bool(re.search(r"flexib|relax|waive|waiver|extension|support|hybrid|online|remedial|grace", t)))

def heuristic(p, f, rng):
    s, d = 0.0, []
    if f["att"] and f["thr"]:
        gap = f["thr"] - p["attendance"]
        if gap > 0: s -= min(gap / 10, 2); d.append("attendance_gap")
        if p["commute"] == ">15km": s -= .5; d.append("commute_attendance")
        if p["works"]: s -= .7; d.append("working_student")
        if p["attendance"] >= f["thr"] + 10: s += .4
    if f["fee"] and p["income"] == "low": s -= 1.2; d.append("fee_burden")
    if f["fine"]: s -= .6 + (.5 if p["income"] == "low" else 0); d.append("penalty")
    if f["early"] and p["commute"] in ("5-15km", ">15km"): s -= 1; d.append("early_timing")
    if f["exam"] and p["cgpa"] < 6.5: s -= .5; d.append("exam_pressure")
    if f["exam"] and p["cgpa"] > 8.5: s += .3
    if f["flex"]: s += 1.0
    s += rng.gauss(0, .35)
    return max(-2, min(2, round(s))), d

BATCH_SYSTEM = ("You simulate members of a university community reacting to a proposed policy. Stay strictly in each "
    "persona's circumstances. Use the rulebook clauses if given. If a persona has peer_views you may adjust your stance after "
    "reading them. Return ONLY a JSON array: [{\"id\":int,\"stance\":int -2..2,\"reason\":\"max 20 words, first person\"}]")

_local = {}
def local_generate(system, user, max_tokens):
    """Runs a base model, optionally with a LoRA adapter (CAMPUSSIM_ADAPTER). Returns None if unavailable."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if "m" not in _local:
            base = os.environ.get("CAMPUSSIM_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
            m = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
            if os.environ.get("CAMPUSSIM_ADAPTER"):
                from peft import PeftModel
                m = PeftModel.from_pretrained(m, os.environ["CAMPUSSIM_ADAPTER"])
            _local.update(m=m, tok=AutoTokenizer.from_pretrained(base))
        tok, m = _local["tok"], _local["m"]
        ids = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                      add_generation_prompt=True, return_tensors="pt").to(m.device)
        out = m.generate(ids, max_new_tokens=max_tokens, do_sample=False)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    except Exception:
        return None

def backend_tag(): return "local" if os.environ.get("CAMPUSSIM_BACKEND") == "local" else "llm"

def llm(system, user, max_tokens=2000):
    if backend_tag() == "local": return local_generate(system, user, max_tokens)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key: return None
    body = json.dumps({"model": os.environ.get("CAMPUSSIM_MODEL", "claude-sonnet-4-6"), "max_tokens": max_tokens,
                       "system": system, "messages": [{"role": "user", "content": user}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
        {"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return "".join(b.get("text", "") for b in json.load(r)["content"])
    except Exception:
        return None

def llm_batch(policy, batch, context=""):
    prompt = f"POLICY:\n{policy}\n\n" + (f"RULEBOOK CLAUSES:\n{context}\n\n" if context else "") + f"PERSONAS:\n{json.dumps(batch)}"
    out = llm(BATCH_SYSTEM, prompt)
    if not out: return None
    try:
        arr = json.loads(out[out.index("["): out.rindex("]") + 1])
        return {int(a["id"]): (max(-2, min(2, int(a["stance"]))), str(a["reason"])[:160]) for a in arr}
    except Exception:
        return None

def run(policy, n=40, clauses=None):
    clauses = clauses or []
    ctx = "\n".join(f"[{c['title']}] {c['text']}" for c in clauses[:5])
    ps, f, rng, tag = personas(n), features(policy), random.Random(11), backend_tag()
    stats = dict(calls=0, hits=0)
    def ask(batch):
        stats["calls"] += 1; r = llm_batch(policy, batch, ctx); stats["hits"] += bool(r); return r
    agents = []
    for i in range(0, n, 10):                                   # round 1: independent reactions
        batch = ps[i:i + 10]; res = ask(batch)
        for p in batch:
            if res and p["id"] in res: st, why = res[p["id"]]; dr = [why] if st <= -1 else []
            else:
                st, dr = heuristic(p, f, rng); why = "; ".join(LABELS[k][0] for k in dr) or "No major personal impact"
            agents.append(dict(p, stance=st, s1=st, reason=why, drivers=dr))
    by = defaultdict(list)                                      # round 2: agents hear three peers from their department
    for a in agents: by[a["dept"]].append(a)
    for a in agents:
        pool = [b for b in by[a["dept"]] if b["id"] != a["id"]] or [b for b in agents if b["id"] != a["id"]]
        a["peers"] = [dict(stance=b["s1"], reason=b["reason"][:80]) for b in rng.sample(pool, min(3, len(pool)))]
    for i in range(0, n, 10):
        batch = agents[i:i + 10]
        res = ask([{k: a[k] for k in ("id", "dept", "year", "cgpa", "attendance", "income", "commute", "works")} | {"peer_views": a["peers"]} for a in batch])
        for a in batch:
            if res and a["id"] in res: a["stance"], a["reason"] = res[a["id"]]
            else: a["stance"] = max(-2, min(2, round(.7 * a["s1"] + .3 * sum(x["stance"] for x in a["peers"]) / len(a["peers"]))))
            if a["stance"] <= -1 and not a["drivers"]: a["drivers"] = [a["reason"]]
    mode = "rules" if not stats["hits"] else (tag if stats["hits"] == stats["calls"] else tag + "+rules")
    m1, o1 = sum(a["s1"] for a in agents) / n, sum(a["s1"] <= -1 for a in agents) / n
    mean, opp = sum(a["stance"] for a in agents) / n, sum(a["stance"] <= -1 for a in agents) / n
    seg = {}
    for key in ("income", "commute", "dept"):
        g = defaultdict(list)
        for a in agents: g[a[key]].append(a["stance"])
        seg[key] = {k: dict(mean=round(sum(v) / len(v), 2), n=len(v), opposed=round(100 * sum(x <= -1 for x in v) / len(v))) for k, v in sorted(g.items())}
    obj = Counter(d for a in agents if a["stance"] <= -1 for d in a["drivers"])
    voices = sorted(agents, key=lambda a: a["stance"]); voices = voices[:5] + voices[-2:]
    res = dict(mode=mode, n=n, mean=round(mean, 2), opposed=round(100 * opp), mean1=round(m1, 2), opposed1=round(100 * o1),
               risk=round(100 * (.5 * opp + .5 * (1 - (mean + 2) / 4))), segments=seg,
               objections=[(LABELS[k][0] if k in LABELS else k, c) for k, c in obj.most_common(5)],
               voices=[dict(who=f"{a['dept']} Y{a['year']}, {a['income']} income, {a['commute']}", stance=a["stance"], reason=a["reason"]) for a in voices],
               top_drivers=[k for k, _ in obj.most_common(3) if k in LABELS],
               clauses=[dict(title=c["title"], text=c["text"][:260], score=c.get("score")) for c in clauses],
               conflicts=rag.find_conflicts(policy, clauses))
    res["revised"] = revise(policy, res)
    return res

def revise(policy, res):
    out = llm("You are a university policy drafter. Rewrite the policy to address the objections and any rulebook conflicts while "
              "keeping its intent. Return only the revised policy text, plain, under 250 words.",
              f"POLICY:\n{policy}\n\nOBJECTIONS:\n{json.dumps(res['objections'])}\n\nCONFLICTS:\n{json.dumps(res['conflicts'])}\n\nCLAUSES:\n{json.dumps(res['clauses'])}", 900)
    if out: return out.strip()
    lines = [f"- {LABELS[k][1]}" for k in res["top_drivers"]]
    lines += [f"- Reconcile with {c['title']}, which states {', '.join(map(str, c['existing']))}% against {', '.join(map(str, c['draft']))}% here." for c in res["conflicts"]]
    return policy.rstrip() + "\n\nSafeguards (suggested by CampusSim):\n" + "\n".join(lines or ["- No major objections detected; publish with a feedback window."])
