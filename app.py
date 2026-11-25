import os, sqlite3, json, secrets, functools
from flask import Flask, g, render_template, request, redirect, url_for, session, flash, abort
from werkzeug.security import generate_password_hash, check_password_hash
import sim, rag

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True, MAX_CONTENT_LENGTH=4 * 1024 * 1024)
DB = os.environ.get("CAMPUSSIM_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "campussim.db"))
ROLES = ("admin", "registrar", "faculty", "student")
STAFF = ("admin", "registrar")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, name TEXT, role TEXT, pw TEXT);
CREATE TABLE IF NOT EXISTS policies(id INTEGER PRIMARY KEY, title TEXT, body TEXT, status TEXT DEFAULT 'draft',
  author_id INTEGER, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS simulations(id INTEGER PRIMARY KEY, policy_id INTEGER, risk INTEGER, mode TEXT,
  result TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY, policy_id INTEGER, user_id INTEGER, stance TEXT,
  comment TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY, title TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc_id INTEGER, text TEXT);
CREATE TABLE IF NOT EXISTS outcomes(policy_id INTEGER PRIMARY KEY, actual_opposed INTEGER, created TEXT DEFAULT CURRENT_TIMESTAMP);
"""
SAMPLE_RULES = """Attendance. A student must maintain at least 75% attendance in each course to be eligible for the semester examination. Shortage of up to 10% may be condoned by the Dean on medical grounds.

Fees. Tuition fee is payable by the 15th of the first month of the semester. A late fee of Rs 200 applies after the due date. Instalments may be approved by the Registrar in cases of financial hardship.

Examinations. Internal assessment carries 40% weightage and the semester examination carries 60%. Students may request revaluation within 7 days of result publication.

Timetable. Classes run from 9:00 AM to 4:30 PM, Monday to Saturday. Changes to timings need Academic Council approval."""
DEMO = [("admin", "System Administrator", "admin", "admin123"), ("registrar", "Dr. Meera Rao (Registrar)", "registrar", "registrar123"),
        ("faculty", "Prof. Arjun Nair", "faculty", "faculty123"), ("student", "Asha Kumar", "student", "student123")]

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d: d.close()

def init_db():
    c = sqlite3.connect(DB); c.executescript(SCHEMA)
    if not c.execute("SELECT 1 FROM users").fetchone():
        for u, n, r, p in DEMO:
            c.execute("INSERT INTO users(username,name,role,pw) VALUES(?,?,?,?)", (u, n, r, generate_password_hash(p)))
        c.execute("INSERT INTO policies(title,body,status,author_id) VALUES(?,?,?,2)", ("Minimum attendance for semester exams",
            "Students must maintain at least 80% attendance to be eligible for semester exams. A fine of Rs 500 applies per "
            "shortage condonation request. Classes begin at 8 AM.", "draft"))
    if not c.execute("SELECT 1 FROM documents").fetchone():
        cur = c.execute("INSERT INTO documents(title) VALUES(?)", ("Sample Academic Regulations (replace with your own)",))
        for t in rag.chunk(SAMPLE_RULES): c.execute("INSERT INTO chunks(doc_id,text) VALUES(?,?)", (cur.lastrowid, t))
    c.commit(); c.close()

@app.before_request
def load_user():
    g.user = None
    if session.get("uid"):
        g.user = db().execute("SELECT * FROM users WHERE id=?", (session["uid"],)).fetchone()
    if request.method == "POST" and request.form.get("csrf") != session.get("csrf"):
        abort(400)

@app.context_processor
def inject():
    return dict(user=g.user, csrf=session.setdefault("csrf", secrets.token_hex(16)))

def roles(*allowed):
    def deco(f):
        @functools.wraps(f)
        def wrap(*a, **k):
            if not g.user: return redirect(url_for("login"))
            if g.user["role"] not in allowed: abort(403)
            return f(*a, **k)
        return wrap
    return deco

ANY = ROLES

@app.route("/")
def home():
    return redirect(url_for("dashboard" if session.get("uid") else "login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db().execute("SELECT * FROM users WHERE username=?", (request.form.get("username", "").strip(),)).fetchone()
        if u and check_password_hash(u["pw"], request.form.get("password", "")):
            session.clear(); session["uid"] = u["id"]; session["csrf"] = secrets.token_hex(16)
            return redirect(url_for("dashboard"))
        flash("Wrong username or password.")
    return render_template("login.html", demo=DEMO)

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))

@app.route("/dashboard")
@roles(*ANY)
def dashboard():
    d = db()
    if g.user["role"] in STAFF:
        stats = dict(total=d.execute("SELECT COUNT(*) FROM policies").fetchone()[0],
                     published=d.execute("SELECT COUNT(*) FROM policies WHERE status='published'").fetchone()[0],
                     simulated=d.execute("SELECT COUNT(DISTINCT policy_id) FROM simulations").fetchone()[0],
                     feedback=d.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])
        rows = d.execute(POL_SQL + " ORDER BY p.id DESC LIMIT 5").fetchall()
    else:
        stats = dict(feedback=d.execute("SELECT COUNT(*) FROM feedback WHERE user_id=?", (g.user["id"],)).fetchone()[0])
        rows = d.execute(POL_SQL + " WHERE p.status='published' ORDER BY p.id DESC LIMIT 5").fetchall()
    return render_template("dashboard.html", stats=stats, rows=rows)

POL_SQL = ("SELECT p.*, (SELECT risk FROM simulations s WHERE s.policy_id=p.id ORDER BY s.id DESC LIMIT 1) AS risk "
           "FROM policies p")

@app.route("/policies")
@roles(*ANY)
def policies():
    q = POL_SQL + ("" if g.user["role"] in STAFF else " WHERE p.status='published'") + " ORDER BY p.id DESC"
    return render_template("policies.html", rows=db().execute(q).fetchall())

@app.route("/policies/new", methods=["GET", "POST"])
@roles(*STAFF)
def policy_new():
    if request.method == "POST":
        t, b = request.form.get("title", "").strip(), request.form.get("body", "").strip()
        if not t or len(b) < 20:
            flash("Add a title and at least 20 characters of policy text.")
        else:
            cur = db().execute("INSERT INTO policies(title,body,author_id) VALUES(?,?,?)", (t, b, g.user["id"]))
            db().commit(); return redirect(url_for("policy_view", pid=cur.lastrowid))
    return render_template("policy_new.html")

def get_policy(pid):
    p = db().execute("SELECT * FROM policies WHERE id=?", (pid,)).fetchone()
    if not p or (g.user["role"] not in STAFF and p["status"] != "published"): abort(404)
    return p

@app.route("/policies/<int:pid>")
@roles(*ANY)
def policy_view(pid):
    p, d, staff = get_policy(pid), db(), g.user["role"] in STAFF
    s = d.execute("SELECT * FROM simulations WHERE policy_id=? ORDER BY id DESC LIMIT 1", (pid,)).fetchone() if staff else None
    fb = d.execute("SELECT f.*, u.name FROM feedback f JOIN users u ON u.id=f.user_id WHERE policy_id=? ORDER BY f.id DESC", (pid,)).fetchall() if staff else []
    mine = d.execute("SELECT * FROM feedback WHERE policy_id=? AND user_id=?", (pid, g.user["id"])).fetchone()
    return render_template("policy_view.html", p=p, sim=json.loads(s["result"]) if s else None, fb=fb, mine=mine)

@app.post("/policies/<int:pid>/simulate")
@roles(*STAFF)
def simulate(pid):
    p = get_policy(pid)
    n = max(10, min(100, int(request.form.get("n", 40) or 40)))
    docs = db().execute("SELECT c.id, c.text, d.title FROM chunks c JOIN documents d ON d.id=c.doc_id").fetchall()
    clauses = rag.search(p["title"] + " " + p["body"], [dict(x) for x in docs], 5)
    r = sim.run(p["body"], n, clauses)
    db().execute("INSERT INTO simulations(policy_id,risk,mode,result) VALUES(?,?,?,?)", (pid, r["risk"], r["mode"], json.dumps(r)))
    db().commit(); flash(f"Simulation finished with {n} agents ({r['mode']} mode).")
    return redirect(url_for("policy_view", pid=pid))

@app.post("/policies/<int:pid>/adopt")
@roles(*STAFF)
def adopt(pid):
    get_policy(pid)
    s = db().execute("SELECT result FROM simulations WHERE policy_id=? ORDER BY id DESC LIMIT 1", (pid,)).fetchone()
    if s:
        db().execute("UPDATE policies SET body=?, status='draft' WHERE id=?", (json.loads(s["result"])["revised"], pid)); db().commit()
        flash("Revised text saved as a draft. Run the simulation again to compare.")
    return redirect(url_for("policy_view", pid=pid))

@app.post("/policies/<int:pid>/publish")
@roles(*STAFF)
def publish(pid):
    p = get_policy(pid)
    db().execute("UPDATE policies SET status=? WHERE id=?", ("draft" if p["status"] == "published" else "published", pid)); db().commit()
    return redirect(url_for("policy_view", pid=pid))

@app.post("/policies/<int:pid>/feedback")
@roles("faculty", "student")
def feedback(pid):
    get_policy(pid); st = request.form.get("stance")
    if st not in ("support", "neutral", "oppose"): abort(400)
    db().execute("DELETE FROM feedback WHERE policy_id=? AND user_id=?", (pid, g.user["id"]))
    db().execute("INSERT INTO feedback(policy_id,user_id,stance,comment) VALUES(?,?,?,?)",
                 (pid, g.user["id"], st, request.form.get("comment", "")[:500])); db().commit()
    flash("Your feedback is saved."); return redirect(url_for("policy_view", pid=pid))

def add_document(title, text):
    d = db(); cur = d.execute("INSERT INTO documents(title) VALUES(?)", (title,))
    for c in rag.chunk(text): d.execute("INSERT INTO chunks(doc_id,text) VALUES(?,?)", (cur.lastrowid, c))
    d.commit()

@app.route("/knowledge", methods=["GET", "POST"])
@roles(*STAFF)
def knowledge():
    if request.method == "POST":
        title, text, f = request.form.get("title", "").strip(), request.form.get("text", "").strip(), request.files.get("file")
        if f and f.filename:
            try: text = rag.extract_text(f.filename, f.read()); title = title or f.filename
            except Exception:
                flash("Could not read that file. Use .txt, .md or a text-based .pdf."); return redirect(url_for("knowledge"))
        if not title or len(text) < 30: flash("Add a title and at least 30 characters of text, or upload a file.")
        else: add_document(title, text); flash("Added to the rulebook index.")
    rows = db().execute("SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.id) AS n FROM documents d ORDER BY d.id DESC").fetchall()
    return render_template("knowledge.html", rows=rows)

@app.post("/knowledge/<int:did>/delete")
@roles(*STAFF)
def knowledge_delete(did):
    db().execute("DELETE FROM chunks WHERE doc_id=?", (did,)); db().execute("DELETE FROM documents WHERE id=?", (did,)); db().commit()
    return redirect(url_for("knowledge"))

@app.route("/validation")
@roles(*STAFF)
def validation():
    rows = db().execute("SELECT p.id, p.title, o.actual_opposed AS actual, "
        "(SELECT result FROM simulations s WHERE s.policy_id=p.id ORDER BY s.id DESC LIMIT 1) AS res, "
        "(SELECT COUNT(*) FROM feedback f WHERE f.policy_id=p.id) AS fb, "
        "(SELECT 100*SUM(f.stance='oppose')/COUNT(*) FROM feedback f WHERE f.policy_id=p.id) AS portal "
        "FROM policies p LEFT JOIN outcomes o ON o.policy_id=p.id ORDER BY p.id").fetchall()
    out, errs = [], []
    for r in rows:
        pred = json.loads(r["res"])["opposed"] if r["res"] else None
        if pred is not None and r["actual"] is not None: errs.append(abs(pred - r["actual"]))
        out.append(dict(id=r["id"], title=r["title"], pred=pred, actual=r["actual"], fb=r["fb"], portal=r["portal"]))
    return render_template("validation.html", rows=out, n=len(errs), mae=round(sum(errs) / len(errs), 1) if errs else None)

@app.post("/validation/<int:pid>")
@roles(*STAFF)
def validation_set(pid):
    get_policy(pid)
    try:
        v = int(request.form["actual"]); assert 0 <= v <= 100
    except Exception:
        flash("Enter a whole number from 0 to 100."); return redirect(url_for("validation"))
    db().execute("INSERT INTO outcomes(policy_id,actual_opposed) VALUES(?,?) ON CONFLICT(policy_id) DO UPDATE SET actual_opposed=excluded.actual_opposed", (pid, v))
    db().commit(); return redirect(url_for("validation"))

@app.route("/users", methods=["GET", "POST"])
@roles("admin")
def users():
    if request.method == "POST":
        f = request.form
        if f.get("role") not in ROLES or len(f.get("password", "")) < 8 or not f.get("username", "").strip():
            flash("Pick a role, a username and a password of 8+ characters.")
        else:
            try:
                db().execute("INSERT INTO users(username,name,role,pw) VALUES(?,?,?,?)", (f["username"].strip(), f.get("name", "").strip() or f["username"], f["role"], generate_password_hash(f["password"])))
                db().commit(); flash("User created.")
            except sqlite3.IntegrityError:
                flash("That username is taken.")
    return render_template("users.html", rows=db().execute("SELECT * FROM users ORDER BY id").fetchall(), roles=ROLES)

@app.errorhandler(403)
def e403(_): return render_template("error.html", code=403, msg="Your role does not have access to this page."), 403
@app.errorhandler(404)
def e404(_): return render_template("error.html", code=404, msg="That page does not exist or is not published."), 404

init_db()
if __name__ == "__main__":
    app.run(debug=os.environ.get("DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
