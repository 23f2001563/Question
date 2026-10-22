import os, json, hashlib, sqlite3, threading, uuid, base64
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
APPROVAL_LIMIT = float(os.environ.get("APPROVAL_LIMIT", "10000"))  # above this -> needs approval
DATABASE_URL = os.environ.get("DATABASE_URL", "")  # Render Postgres; if empty, uses SQLite
DB_PATH = os.environ.get("DB_PATH", "state.db")
LOCK = threading.Lock()


# ---------- tiny key-value store (Postgres if DATABASE_URL set, else SQLite) ----------
def run(sql, params=(), fetch=False):
    with LOCK:
        if DATABASE_URL:
            import psycopg2
            con = psycopg2.connect(DATABASE_URL)
            try:
                cur = con.cursor()
                cur.execute(sql.replace("?", "%s"), params)
                rows = cur.fetchall() if fetch else None
                con.commit()
                return rows
            finally:
                con.close()
        con = sqlite3.connect(DB_PATH)
        try:
            cur = con.execute(sql, params)
            rows = cur.fetchall() if fetch else None
            con.commit()
            return rows
        finally:
            con.close()


run("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")


def get(k):
    rows = run("SELECT v FROM kv WHERE k=?", (k,), True)
    return json.loads(rows[0][0]) if rows else None


def put(k, v):
    run("INSERT INTO kv (k,v) VALUES (?,?) ON CONFLICT (k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))


def claim(k, v):
    """Store v under k only if k is new. True if we stored it (we are first)."""
    rows = run("INSERT INTO kv (k,v) VALUES (?,?) ON CONFLICT (k) DO NOTHING RETURNING k", (k, json.dumps(v)), True)
    return bool(rows)


def list_prefix(prefix):
    rows = run("SELECT v FROM kv WHERE k LIKE ?", (prefix + "%",), True)
    return [json.loads(r[0]) for r in rows]


# ---------- helpers ----------
def now():
    return datetime.now(timezone.utc).isoformat()


def norm(s):
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def base_url(request):
    return os.environ.get("PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL") or str(request.base_url).rstrip("/")


def tenant_of(request, msg):
    t = request.headers.get("x-user-id") or request.headers.get("x-tenant-id")
    if not t and request.headers.get("authorization"):
        t = hashlib.sha1(request.headers["authorization"].encode()).hexdigest()[:16]
    if not t:
        t = (msg.get("metadata") or {}).get("userId") or (msg.get("metadata") or {}).get("user_id")
    return norm(t) or "default"


def collect(parts):
    texts, files = [], []
    for p in parts or []:
        if p.get("text"):
            texts.append(p["text"])
        if p.get("data") is not None:
            texts.append(json.dumps(p["data"]))
        f = p.get("file") or {}
        b64 = f.get("bytes") or p.get("raw")
        mime = f.get("mimeType") or p.get("mediaType") or "application/pdf"
        url = f.get("uri") or p.get("url")
        try:
            if not b64 and url:
                r = requests.get(url, timeout=30)
                mime = r.headers.get("content-type", mime).split(";")[0]
                b64 = base64.b64encode(r.content).decode()
        except Exception:
            pass
        if b64:
            files.append({"mime": mime, "b64": b64})
    return texts, files


def find_approval(parts):
    for p in parts or []:
        d = p.get("data")
        if isinstance(d, dict) and (d.get("approve") or d.get("approved")) and (d.get("proposalId") or d.get("proposal_id")):
            return d.get("proposalId") or d.get("proposal_id")
    return None


PROMPT = """You are an accounts-payable assistant. Read the invoice package below (text and/or attached documents).
Return ONLY a JSON object with these keys:
{"invoice_number": string|null, "vendor": string|null, "amount": number|null, "currency": string|null,
 "already_paid": boolean, "missing_info": boolean, "reason": string,
 "evidence": [{"source": string, "quote": string}]}
Rules:
- already_paid is true ONLY if the package itself (or the PREVIOUSLY PAID list) shows a payment for this SAME invoice number from the SAME vendor.
  A payment for a DIFFERENT invoice number is NOT a duplicate, even for the same vendor or same amount.
- missing_info is true if you cannot find an invoice number, vendor or amount.
- evidence: short exact quotes (under 20 words) from the documents that support your reading, with the document name/section as source.
- Never invent values. Use null when unknown."""


def ask_gemini(texts, files, history):
    parts = [{"text": PROMPT + "\n\nPREVIOUSLY PAID (our records):\n" + json.dumps(history)}]
    parts += [{"text": t} for t in texts]
    parts += [{"inline_data": {"mime_type": f["mime"], "data": f["b64"]}} for f in files]
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_KEY},
        json={"contents": [{"parts": parts}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0}},
        timeout=90,
    )
    r.raise_for_status()
    return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])


def pay(tenant, d):
    pk = f"paid:{tenant}:{norm(d['vendor'])}:{norm(d['invoice_number'])}"
    receipt = {
        "receiptId": "rcpt_" + hashlib.sha1(pk.encode()).hexdigest()[:12],
        "invoice_number": d["invoice_number"], "vendor": d["vendor"],
        "amount": d["amount"], "currency": d.get("currency"),
        "status": "paid", "paidAt": now(),
    }
    if claim(pk, receipt):
        return receipt
    return get(pk)  # already paid earlier: return the original receipt, never pay twice


def decide(tenant, texts, files):
    history = list_prefix(f"paid:{tenant}:")
    d = ask_gemini(texts, files, history)
    d.setdefault("evidence", [])
    dup = bool(d.get("already_paid")) or bool(
        d.get("vendor") and d.get("invoice_number") and get(f"paid:{tenant}:{norm(d['vendor'])}:{norm(d['invoice_number'])}"))
    if d.get("missing_info") or d.get("amount") is None:
        d["action"] = "request_info"
    elif dup:
        d["action"] = "reject_duplicate"
    elif float(d["amount"]) > APPROVAL_LIMIT:
        d["action"] = "hold_for_approval"
    else:
        d["action"] = "pay"
    d["approval_limit"] = APPROVAL_LIMIT
    return d


def task(state, decision, receipt=None, ctx=None):
    data = dict(decision)
    if receipt:
        data["receipt"] = receipt
    return {
        "kind": "task", "id": str(uuid.uuid4()), "contextId": ctx or str(uuid.uuid4()),
        "status": {"state": state, "timestamp": now()},
        "artifacts": [{"artifactId": str(uuid.uuid4()), "name": "invoice_decision",
                       "parts": [{"kind": "data", "data": data}]}],
    }


def handle_send(request, params):
    msg = params.get("message") or {}
    tenant = tenant_of(request, msg)
    parts = msg.get("parts") or []
    texts, files = collect(parts)
    ctx = msg.get("contextId")
    mid = msg.get("messageId") or hashlib.sha1((json.dumps(texts) + str(len(files))).encode()).hexdigest()
    ikey = f"idem:{tenant}:{mid}"
    cached = get(ikey)
    if cached:
        return cached  # retry of the same message: same answer, no repeat action

    approval = find_approval(parts)
    if approval:
        prop = get(f"prop:{tenant}:{approval}")
        if not prop:
            raise ValueError("unknown proposalId")
        receipt = pay(tenant, prop)
        prop["status"] = "executed"
        put(f"prop:{tenant}:{approval}", prop)
        result = task("completed", dict(prop, action="pay"), receipt, ctx)
    else:
        d = decide(tenant, texts, files)
        if d["action"] == "pay":
            result = task("completed", d, pay(tenant, d), ctx)
        elif d["action"] == "hold_for_approval":
            pid = "prop_" + hashlib.sha1(ikey.encode()).hexdigest()[:12]
            d["proposalId"] = pid
            d["status"] = "pending_approval"
            put(f"prop:{tenant}:{pid}", d)
            result = task("input-required", d, None, ctx)
        else:
            result = task("completed", d, None, ctx)
    put(ikey, result)
    put(f"task:{tenant}:{result['id']}", result)
    return result


# ---------- A2A endpoints ----------
def card(request):
    url = base_url(request) + "/a2a/"
    return {
        "name": "Invoice Action Agent", "description": "Reads invoice packages, proposes and executes safe payment actions with evidence.",
        "version": "1.0.0", "protocolVersion": "1.0", "url": url,
        "supportedInterfaces": [{"url": url, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
        "preferredTransport": "JSONRPC",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain", "application/json", "application/pdf", "image/png", "image/jpeg"],
        "defaultOutputModes": ["application/json"],
        "skills": [{"id": "invoice-action", "name": "Invoice action", "description": "Decide pay / hold / reject duplicate / request info",
                    "tags": ["invoice", "payments"]}],
    }


@app.get("/")
def root():
    return {"ok": True}


@app.get("/.well-known/agent-card.json")
@app.get("/.well-known/agent.json")
@app.get("/a2a/.well-known/agent-card.json")
@app.get("/a2a/.well-known/agent.json")
def agent_card(request: Request):
    return card(request)


@app.post("/")
@app.post("/a2a")
@app.post("/a2a/")
async def rpc(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
    rid = body.get("id")
    method = norm(body.get("method"))
    params = body.get("params") or {}
    try:
        if method in ("messagesend", "sendmessage"):
            return {"jsonrpc": "2.0", "id": rid, "result": handle_send(request, params)}
        if method in ("tasksget", "gettask"):
            t = get(f"task:{tenant_of(request, {})}:{params.get('id')}")
            if not t:
                return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32001, "message": "Task not found"}}
            return {"jsonrpc": "2.0", "id": rid, "result": t}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}
    except ValueError as e:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": str(e)}}
    except Exception as e:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": "Internal error: " + str(e)[:200]}}
