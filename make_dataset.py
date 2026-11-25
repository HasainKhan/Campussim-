"""Build a chat-format JSONL dataset for LoRA fine-tuning, matching the prompt format app/sim.py uses.
  python finetune/make_dataset.py --from-db campussim.db --out finetune/data.jsonl     # real portal feedback
  python finetune/make_dataset.py --bootstrap 400 --out finetune/data.jsonl            # synthetic starter data
Bootstrap labels come from CampusSim's rule engine, so a model trained only on them just imitates the rules.
Prefer real feedback, or LLM-labelled examples reviewed by a human, for anything beyond a demo."""
import argparse, json, os, random, sqlite3, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sim

POLICIES = ["Students need 85% attendance to sit semester exams. Classes begin at 8 AM.",
            "A fine of Rs 1000 applies for late fee payment. Fees rise by 10% next year.",
            "Internal assessments will carry 50% weightage; make-up exams are discontinued.",
            "Attendance requirement is 70% with a grace of 5% and hybrid sessions for day scholars.",
            "Hostel curfew moves to 8 PM; a penalty applies for violations.",
            "Exam pattern changes next semester, with remedial classes and a fee waiver for low-income students."]

def rec(policy, persona, stance, reason):
    return {"messages": [{"role": "system", "content": sim.BATCH_SYSTEM},
        {"role": "user", "content": f"POLICY:\n{policy}\n\nPERSONAS:\n{json.dumps([persona])}"},
        {"role": "assistant", "content": json.dumps([{"id": persona["id"], "stance": stance, "reason": reason}])}]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--from-db"); ap.add_argument("--bootstrap", type=int, default=0); ap.add_argument("--out", default="finetune/data.jsonl")
    a = ap.parse_args(); rows, rng = [], random.Random(3)
    if a.from_db:
        c = sqlite3.connect(a.from_db)
        for body, role, st, com in c.execute("SELECT p.body, u.role, f.stance, f.comment FROM feedback f JOIN policies p ON p.id=f.policy_id JOIN users u ON u.id=f.user_id"):
            if com and com.strip(): rows.append(rec(body, {"id": 0, "role": role}, {"support": 1, "neutral": 0, "oppose": -1}[st], com.strip()[:160]))
    for i in range(a.bootstrap):
        pol = rng.choice(POLICIES); p = sim.personas(1, seed=rng.randint(0, 10**6))[0]
        st, d = sim.heuristic(p, sim.features(pol), rng)
        rows.append(rec(pol, p, st, "; ".join(sim.LABELS[k][0] for k in d) or "No major personal impact"))
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} examples to {a.out}")
main()
