# CampusSim: Multi-Agent LLM Policy Sandbox

Simulate how students, faculty and staff may react to a university policy before rollout. LLM persona agents
(or a built-in rule engine when no API key is set) produce stance, objections, segment impact, a rollout-risk score
and a revised policy draft.

## Run
```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                         # http://localhost:5000
```
Optional LLM mode: `export ANTHROPIC_API_KEY=...` (model via `CAMPUSSIM_MODEL`, default claude-sonnet-4-6).
Without a key the run is labelled "rules" mode so results are never passed off as LLM output.

## What is inside
- **Agents:** 20-100 synthetic personas; round 1 gives independent reactions, round 2 lets each agent hear three peers from its department and update.
- **RAG:** upload rulebook text/PDF under *Rulebook index*; clauses are ranked with BM25 and given to every agent. Percentage mismatches between a draft and the rulebook are flagged as possible conflicts.
- **Backends:** `ANTHROPIC_API_KEY` (API model), `CAMPUSSIM_BACKEND=local` with an optional LoRA adapter (`CAMPUSSIM_ADAPTER`), or the built-in rule engine.
- **LoRA pipeline:** `finetune/make_dataset.py` then `finetune/train_lora.py` (needs a GPU; not trained or run in the build environment).
- **Validation:** the *Validation* page compares predicted objection rates with real ones and reports mean absolute error.

## Roles
| Role | Can do |
|---|---|
| admin | everything, plus create users and assign roles |
| registrar | draft policies, manage the rulebook index, run simulations, publish, adopt revisions, read feedback, record real outcomes |
| faculty / student | read published notices, submit one feedback entry per policy |

Demo logins (change before real use): admin/admin123, registrar/registrar123, faculty/faculty123, student/student123.

## Structure
`app.py` routes, auth, RBAC, CSRF, SQLite | `sim.py` personas, agents, scoring, revision | `rag.py` chunking, BM25, conflict check | `finetune/` LoRA pipeline | `templates/`, `static/` UI

## Production notes
Set `SECRET_KEY`; run `gunicorn app:app` behind HTTPS; replace demo accounts; personas are synthetic, so validate
simulated outcomes against real past policy feedback before relying on them. Next steps: embedding retrieval, SSO,
and a trained adapter evaluated on real feedback.
