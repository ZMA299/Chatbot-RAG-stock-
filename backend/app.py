import os
import re
import json
import time
import hmac
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional, Any, Dict, Tuple

import requests
from fastapi import FastAPI, UploadFile, File, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openai import OpenAI
from supabase import create_client
from langchain_openai import OpenAIEmbeddings


def _env(key: str) -> str:
    return (os.getenv(key) or "").strip()


OPENAI_API_KEY = _env("OPENAI_API_KEY")
SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_KEY = _env("SUPABASE_KEY")
ODOO_URL = _env("ODOO_URL")
ODOO_DB = _env("ODOO_DB")
ODOO_USER = _env("ODOO_USER")
ODOO_PASSWORD = _env("ODOO_PASSWORD")
SYNC_API_KEY = _env("SYNC_API_KEY")
WEBHOOK_SECRET = _env("WEBHOOK_SECRET")

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
LOCAL_CATALOG_TABLE = "catalog_products"
SYNC_LOG_TABLE = "sync_log"
LATENCY_COLUMN = "latency_s"
LEBANON_TZ = ZoneInfo("Asia/Beirut")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")
if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is not set.")
if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY is not set.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings_client = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)


def detect_lang(text: str) -> str:
    if re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text or ""):
        return "ar"
    return "en"


def now_in_lebanon() -> datetime:
    return datetime.now(LEBANON_TZ)


def format_now(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " (Asia/Beirut)"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def parse_latency_to_seconds(latency_str: str) -> Optional[float]:
    try:
        match = re.match(r"^\s*([0-9]*\.?[0-9]+)s", latency_str or "")
        return round(float(match.group(1)), 3) if match else None
    except Exception:
        return None


TERM_NORMALIZATION = {
    "screenas": "screens",
    "screenes": "screens",
    "screns": "screens",
    "screans": "screens",
    "moniters": "monitors",
    "minitors": "monitors",
    "moniter": "monitor",
    "camras": "cameras",
    "camra": "camera",
    "routrs": "routers",
    "swiches": "switches",
}

GENERIC_ALIASES = {
    "screens": "monitor",
    "screen": "monitor",
    "displays": "monitor",
    "display": "monitor",
}


def normalize_terms(text: str) -> str:
    words = normalize_whitespace(text).lower().split()
    fixed = []
    for w in words:
        w = TERM_NORMALIZATION.get(w, w)
        w = GENERIC_ALIASES.get(w, w)
        fixed.append(w)
    return " ".join(fixed)


def extract_search_phrase(question: str) -> str:
    q = normalize_whitespace(question).lower()

    prefix_patterns = [
        r"^go to stock and tell me all\s+",
        r"^go to stock and tell me\s+",
        r"^tell me all\s+",
        r"^show me all\s+",
        r"^show me\s+",
        r"^tell me\s+",
        r"^list all\s+",
        r"^list\s+",
        r"^do you have\s+",
        r"^what do you have\s+",
        r"^i want\s+",
        r"^i need\s+",
        r"^find\s+",
        r"^search for\s+",
        r"^search\s+",
    ]

    for p in prefix_patterns:
        q = re.sub(p, "", q, flags=re.I)

    cleanup_patterns = [
        r"\byou have\b",
        r"\bin stock\b",
        r"\bavailable\b",
        r"\bavailability\b",
        r"\bproducts\b",
        r"\bproduct\b",
        r"\bplease\b",
        r"\bfor me\b",
        r"\bgo to stock\b",
    ]

    for p in cleanup_patterns:
        q = re.sub(p, " ", q, flags=re.I)

    q = re.sub(r"[^\w\s/\-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    q = normalize_terms(q)

    return q or normalize_terms(normalize_whitespace(question))


_SMALLTALK_PHRASES = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "thanks",
    "thank you",
    "thx",
    "ty",
    "ok",
    "okay",
    "sure",
    "alright",
    "wow",
    "nice",
    "cool",
    "great",
    "awesome",
    "amazing",
    "yes",
    "no",
    "yep",
    "nope",
    "yeah",
    "nah",
    "bye",
    "goodbye",
    "see you",
    "good night",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "what's up",
    "whats up",
    "who are you",
    "what are you",
    "what can you do",
    "مرحبا",
    "هلا",
    "السلام عليكم",
    "شكرا",
    "شكرًا",
    "أهلا",
    "أهلاً",
    "مرحباً",
    "صباح الخير",
    "مساء الخير",
    "كيف حالك",
    "كيفك",
    "تمام",
    "حسنا",
    "نعم",
    "لا",
    "مع السلامة",
    "باي",
    "يعطيك العافية",
}

_CONVERSATIONAL_WORDS = {
    "wow",
    "nice",
    "cool",
    "great",
    "awesome",
    "amazing",
    "perfect",
    "ok",
    "okay",
    "sure",
    "alright",
    "yes",
    "no",
    "yep",
    "nope",
    "thanks",
    "thank",
    "please",
    "sorry",
    "help",
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
    "can",
    "could",
    "would",
    "should",
    "will",
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "this",
    "that",
    "it",
    "i",
    "you",
    "we",
    "they",
    "my",
    "your",
    "me",
    "him",
    "her",
    "us",
    "them",
    "not",
    "but",
    "and",
    "or",
    "if",
    "so",
    "too",
    "also",
    "very",
    "really",
    "just",
    "more",
    "much",
    "many",
    "good",
    "bad",
    "need",
    "want",
    "like",
    "know",
    "think",
    "tell",
    "show",
    "give",
    "get",
    "make",
    "go",
    "come",
}

_KNOWN_PRODUCT_HINTS = {
    "lg",
    "samsung",
    "dell",
    "hp",
    "lenovo",
    "asus",
    "acer",
    "ubiquiti",
    "monitor",
    "monitors",
    "screen",
    "screens",
    "display",
    "displays",
    "router",
    "routers",
    "switch",
    "switches",
    "camera",
    "cameras",
    "laptop",
    "laptops",
    "server",
    "servers",
    "phone",
    "phones",
    "cable",
    "cables",
    "fiber",
    "patch",
    "panel",
    "access",
    "point",
    "antenna",
    "battery",
    "adapter",
    "connector",
    "rack",
    "ups",
    "network",
    "ethernet",
    "wireless",
    "wifi",
    "wi-fi",
}

_INVENTORY_PATTERN = re.compile(
    r"\b(stock|inventory|available|in stock|quantity|qty|price|cost|product|products|"
    r"category|categories|do you have|what do you have|setup|install)\b",
    re.I,
)

_STOCK_CHECK_PATTERN = re.compile(
    r"\b(in stock|available|availability|how many|quantity|qty|stock of|stock for)\b",
    re.I,
)


def is_smalltalk(question: str) -> bool:
    return normalize_whitespace(question).lower() in _SMALLTALK_PHRASES


def _is_inventory_request(q: str) -> bool:
    if _INVENTORY_PATTERN.search(q):
        return True

    words = set(re.findall(r"\b[\w\-]+\b", q))

    if words & _KNOWN_PRODUCT_HINTS:
        return True

    if len(words) <= 2:
        non_conversational = words - _CONVERSATIONAL_WORDS
        if non_conversational and (non_conversational & _KNOWN_PRODUCT_HINTS):
            return True
        return False

    return False


def _is_stock_check(q: str) -> bool:
    return bool(_STOCK_CHECK_PATTERN.search(q))


def classify_intent(question: str) -> dict:
    q = normalize_terms(normalize_whitespace(question).lower())

    if is_smalltalk(q):
        return {"intent": "smalltalk"}

    if _is_stock_check(q):
        return {"intent": "inventory"}

    if _is_inventory_request(q):
        return {"intent": "inventory"}

    return {"intent": "rag_only"}


_odoo_uid: Optional[int] = None
_odoo_session: Optional[requests.Session] = None
_odoo_rpc_url: str = ""
_odoo_rpc_id: int = 0


def _jsonrpc(endpoint: str, method: str, *args) -> Any:
    global _odoo_rpc_id

    if _odoo_session is None:
        raise RuntimeError("Odoo session not initialized")

    _odoo_rpc_id += 1

    payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "id": _odoo_rpc_id,
        "params": {
            "service": endpoint,
            "method": method,
            "args": list(args),
        },
    }

    resp = _odoo_session.post(_odoo_rpc_url, json=payload, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    if data.get("error"):
        raise RuntimeError(f"Odoo RPC error: {data['error']}")

    return data["result"]


def _odoo_execute(model: str, method: str, domain, **kwargs) -> Any:
    return _jsonrpc(
        "object",
        "execute_kw",
        ODOO_DB,
        _odoo_uid,
        ODOO_PASSWORD,
        model,
        method,
        [domain],
        kwargs,
    )


if ODOO_URL and ODOO_DB and ODOO_USER and ODOO_PASSWORD:
    try:
        _odoo_rpc_url = f"{ODOO_URL}/jsonrpc"
        _odoo_session = requests.Session()
        _odoo_session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        _odoo_uid = _jsonrpc(
            "common",
            "authenticate",
            ODOO_DB,
            ODOO_USER,
            ODOO_PASSWORD,
            {},
        )
        print(f"[SYNC] Connected to Odoo via JSON-RPC as uid={_odoo_uid}")
    except Exception as e:
        print(f"[SYNC] Odoo connection failed: {type(e).__name__}: {e}")
        _odoo_uid = None
        _odoo_session = None
else:
    print("[SYNC] Odoo credentials missing. Sync features disabled.")


def odoo_is_connected() -> bool:
    return _odoo_uid is not None


_PRODUCT_FIELDS = [
    "id",
    "name",
    "default_code",
    "list_price",
    "qty_available",
    "virtual_available",
    "categ_id",
    "active",
    "write_date",
]


def odoo_fetch_products_by_ids(product_ids: List[int]) -> List[dict]:
    if not odoo_is_connected() or not product_ids:
        return []

    try:
        rows = _odoo_execute(
            "product.template",
            "read",
            product_ids,
            fields=_PRODUCT_FIELDS,
        )
        return rows
    except Exception as e:
        print(f"[SYNC] odoo_fetch_products_by_ids error: {type(e).__name__}: {e}")
        return []


def odoo_fetch_all_products(limit: int = 5000) -> List[dict]:
    if not odoo_is_connected():
        return []

    try:
        t0 = time.perf_counter()
        rows = _odoo_execute(
            "product.template",
            "search_read",
            [["active", "in", [True, False]]],
            fields=_PRODUCT_FIELDS,
            limit=limit,
        )
        print(
            f"[RECONCILE] Fetched {len(rows)} products from Odoo "
            f"in {time.perf_counter() - t0:.2f}s"
        )
        return rows
    except Exception as e:
        print(f"[RECONCILE] odoo_fetch_all_products error: {type(e).__name__}: {e}")
        return []


def _odoo_to_local_row(p: dict) -> dict:
    return {
        "product_id": p.get("id"),
        "name": p.get("name", ""),
        "default_code": p.get("default_code", "") or "",
        "category_name": (p.get("categ_id") or [False, ""])[1],
        "category_id": (p.get("categ_id") or [False])[0] or None,
        "sales_price": p.get("list_price", 0),
        "quantity_on_hand": p.get("qty_available", 0),
        "forecasted_quantity": p.get("virtual_available", 0),
        "in_stock": (p.get("qty_available", 0) or 0) > 0,
        "active": p.get("active", True),
        "search_text": " ".join(filter(None, [
            p.get("name", ""),
            p.get("default_code", ""),
            (p.get("categ_id") or [False, ""])[1],
        ])),
        "last_synced_at": datetime.utcnow().isoformat(),
        "sync_source": "odoo",
    }


def rag_retrieve_chunks(question: str, k: int = 4) -> List[dict]:
    try:
        q_vec = embeddings_client.embed_query(question)
        resp = supabase_client.rpc(
            "match_rag_chunks",
            {
                "query_embedding": q_vec,
                "match_count": k,
            },
        ).execute()
        return resp.data or []
    except Exception as e:
        print(f"rag.retrieve_chunks error: {type(e).__name__}: {e}")
        return []


def rag_build_context(chunks: List[dict], k: int = 4) -> str:
    if not chunks:
        return "(none)"

    return "\n\n".join(
        f"[p{c.get('page', '?')}] {c.get('content', '')}"
        for c in chunks[:k]
    )


def rag_extract_source_pages(chunks: List[dict]) -> List[int]:
    pages = set()

    for c in chunks:
        p = c.get("page")
        if p is not None:
            pages.add(p + 1)

    return sorted(pages)


def _extract_search_keywords(query: str) -> List[str]:
    phrase = extract_search_phrase(query)
    words = phrase.split()

    filler = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "can",
        "you",
        "have",
        "what",
        "how",
        "want",
        "need",
        "wanna",
        "gonna",
        "some",
        "any",
        "all",
        "but",
        "not",
        "are",
        "was",
        "were",
        "been",
        "being",
        "has",
        "had",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "must",
        "very",
        "really",
        "just",
        "also",
        "too",
        "much",
        "many",
        "home",
        "work",
        "setup",
        "remotly",
        "remolty",
        "remotely",
        "make",
        "like",
        "look",
        "tell",
        "show",
        "give",
        "get",
        "iam",
        "hmmm",
        "hmm",
        "umm",
        "hey",
        "please",
        "suggest",
        "recommend",
        "best",
        "good",
    }

    keywords = [w for w in words if len(w) >= 3 and w not in filler]

    return keywords if keywords else words[:3]


def catalog_search_local(query: str, limit: int = 10) -> List[dict]:
    keywords = _extract_search_keywords(query)
    search_phrase = " ".join(keywords) if keywords else extract_search_phrase(query)

    if not search_phrase:
        return []

    try:
        t0 = time.perf_counter()
        resp = supabase_client.rpc(
            "search_catalog",
            {
                "search_query": search_phrase,
                "max_results": limit,
            },
        ).execute()

        elapsed = time.perf_counter() - t0
        rows = resp.data or []

        print(
            f"catalog.search (FTS) '{search_phrase}' "
            f"-> {len(rows)} rows in {elapsed:.3f}s"
        )

        if rows:
            return rows[:limit]

    except Exception as e:
        print(f"catalog.search FTS error: {type(e).__name__}: {e}")

    all_rows = []
    seen_ids = set()

    for kw in keywords[:3]:
        try:
            t0 = time.perf_counter()

            resp = (
                supabase_client.table(LOCAL_CATALOG_TABLE)
                .select(
                    "product_id,name,default_code,category_name,"
                    "category_id,sales_price,quantity_on_hand,"
                    "forecasted_quantity,in_stock,active"
                )
                .or_(
                    f"name.ilike.%{kw}%,"
                    f"category_name.ilike.%{kw}%,"
                    f"search_text.ilike.%{kw}%"
                )
                .eq("active", True)
                .limit(limit)
                .execute()
            )

            elapsed = time.perf_counter() - t0
            print(
                f"catalog.search (ilike) '{kw}' "
                f"-> {len(resp.data or [])} rows in {elapsed:.3f}s"
            )

            for row in resp.data or []:
                pid = row.get("product_id")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_rows.append(row)

        except Exception as e:
            print(f"catalog.search ilike error for '{kw}': {type(e).__name__}: {e}")

        if len(all_rows) >= limit:
            break

    return all_rows[:limit]


def catalog_upsert_rows(rows: List[dict]) -> dict:
    if not rows:
        return {"upserted": 0, "data_count": 0}

    try:
        resp = supabase_client.table(LOCAL_CATALOG_TABLE).upsert(
            rows,
            on_conflict="product_id",
        ).execute()

        return {"upserted": len(rows), "data_count": len(resp.data or [])}
    except Exception as e:
        raise RuntimeError(f"Catalog upsert failed: {type(e).__name__}: {e}")


def catalog_deactivate_products(product_ids: List[int]) -> int:
    if not product_ids:
        return 0

    try:
        resp = (
            supabase_client.table(LOCAL_CATALOG_TABLE)
            .update({
                "active": False,
                "in_stock": False,
                "last_synced_at": datetime.utcnow().isoformat(),
            })
            .in_("product_id", product_ids)
            .execute()
        )
        return len(resp.data or [])
    except Exception as e:
        print(f"catalog.deactivate error: {type(e).__name__}: {e}")
        return 0


def catalog_get_all_product_ids() -> set:
    try:
        resp = supabase_client.table(LOCAL_CATALOG_TABLE).select("product_id").execute()
        return {r["product_id"] for r in resp.data or []}
    except Exception as e:
        print(f"catalog.get_all_product_ids error: {type(e).__name__}: {e}")
        return set()


def _verify_webhook_signature(payload_bytes: bytes, signature: str) -> bool:
    if not WEBHOOK_SECRET:
        return True

    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature or "")


def _log_sync_event(
    source: str,
    event_type: str,
    product_ids: List[int],
    status: str,
    detail: str = "",
):
    try:
        supabase_client.table(SYNC_LOG_TABLE).insert({
            "source": source,
            "event_type": event_type,
            "product_ids": product_ids,
            "status": status,
            "detail": detail,
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        print(f"[SYNC LOG] Failed to log: {type(e).__name__}: {e}")


def _is_odoo19_native_webhook(event: dict) -> bool:
    return "_id" in event and "_model" in event


def _handle_odoo19_native(event: dict) -> dict:
    model = event.get("_model", "")
    record_id = event.get("_id")
    action_name = event.get("_action", "")

    print(f"[WEBHOOK] Odoo 19 native: model={model}, id={record_id}, action={action_name}")

    if not record_id:
        return {"status": "skipped", "reason": "no _id in payload"}

    product_ids = []

    if model == "product.template":
        product_ids = [record_id]

    elif model == "stock.move":
        try:
            moves = _odoo_execute("stock.move", "read", [record_id], fields=["product_id"])

            if moves:
                prod = moves[0].get("product_id")

                if prod:
                    prod_id = prod[0] if isinstance(prod, (list, tuple)) else prod

                    products = _odoo_execute(
                        "product.product",
                        "read",
                        [prod_id],
                        fields=["product_tmpl_id"],
                    )

                    if products:
                        tmpl = products[0].get("product_tmpl_id")
                        tmpl_id = tmpl[0] if isinstance(tmpl, (list, tuple)) else tmpl
                        product_ids = [tmpl_id]

        except Exception as e:
            print(f"[WEBHOOK] stock.move lookup error: {type(e).__name__}: {e}")
            _log_sync_event("webhook", "stock.move.error", [record_id], "error", str(e))
            return {"status": "error", "reason": str(e)}

    else:
        return {"status": "skipped", "reason": f"unhandled model: {model}"}

    if not product_ids:
        return {"status": "skipped", "reason": "could not resolve product_ids"}

    odoo_products = odoo_fetch_products_by_ids(product_ids)

    if not odoo_products:
        _log_sync_event("webhook", f"{model}.sync", product_ids, "warn", "no data from Odoo")
        return {"status": "warn", "reason": "could not fetch products from Odoo"}

    local_rows = [_odoo_to_local_row(p) for p in odoo_products]
    result = catalog_upsert_rows(local_rows)

    _log_sync_event(
        "webhook",
        f"{model}.sync",
        product_ids,
        "ok",
        f"upserted {result['upserted']}",
    )

    return {"status": "ok", **result}


def _handle_custom_format(event: dict) -> dict:
    event_type = event.get("event", "unknown")
    product_ids = event.get("product_ids", [])

    if not product_ids:
        return {"status": "skipped", "reason": "no product_ids"}

    if event_type == "product.unlink":
        count = catalog_deactivate_products(product_ids)
        _log_sync_event("webhook", event_type, product_ids, "ok", f"deactivated {count}")
        return {"status": "ok", "deactivated": count}

    odoo_products = odoo_fetch_products_by_ids(product_ids)

    if not odoo_products:
        _log_sync_event("webhook", event_type, product_ids, "warn", "no data from Odoo")
        return {"status": "warn", "reason": "could not fetch products from Odoo"}

    local_rows = [_odoo_to_local_row(p) for p in odoo_products]
    result = catalog_upsert_rows(local_rows)

    _log_sync_event(
        "webhook",
        event_type,
        product_ids,
        "ok",
        f"upserted {result['upserted']}",
    )

    return {"status": "ok", **result}


def webhook_handle_event(event: dict) -> dict:
    if _is_odoo19_native_webhook(event):
        return _handle_odoo19_native(event)

    return _handle_custom_format(event)


def reconcile_full_sync() -> dict:
    t0 = time.perf_counter()

    odoo_products = odoo_fetch_all_products()

    if not odoo_products:
        _log_sync_event("reconcile", "full_sync", [], "warn", "no products from Odoo")
        return {"status": "warn", "reason": "no products returned from Odoo"}

    local_rows = [_odoo_to_local_row(p) for p in odoo_products]
    result = catalog_upsert_rows(local_rows)

    odoo_ids = {p["id"] for p in odoo_products}
    local_ids = catalog_get_all_product_ids()
    orphaned_ids = list(local_ids - odoo_ids)

    deactivated = 0

    if orphaned_ids:
        deactivated = catalog_deactivate_products(orphaned_ids)

    elapsed = time.perf_counter() - t0

    summary = (
        f"synced {result['upserted']} products, "
        f"deactivated {deactivated} orphans in {elapsed:.2f}s"
    )

    _log_sync_event("reconcile", "full_sync", [], "ok", summary)

    return {
        "status": "ok",
        "synced": result["upserted"],
        "deactivated": deactivated,
        "elapsed_s": round(elapsed, 2),
    }


def llm_build_system_prompt(lang: str) -> str:
    now_str = format_now(now_in_lebanon())

    return f"""You are an AI Receptionist and Customer Support Assistant for Axion.

CAPABILITIES:
1. Product catalog with live-synced prices and stock levels.
2. PDF Knowledge Base from Supabase RAG.
3. Voice and text support.

DATA FRESHNESS:
- Product data is synced from Odoo via webhooks and nightly reconciliation.
- Prices and stock reflect the latest sync.

RESPONSE RULES:
- Answer in the user's language, Arabic or English.
- Be accurate, brief, and professional.
- For inventory questions, use only the product data provided.
- Do not invent stock, prices, products, services, policies, or certifications.
- For company/policy questions, use only the RAG context when available.
- If the context does not specify the answer, say that the document does not specify this information.

Current date/time: {now_str}
"""


def _format_products_for_llm(products: List[dict]) -> str:
    if not products:
        return "(no matching products found in catalog)"

    lines = []

    for p in products:
        name = p.get("name", "")
        code = p.get("default_code", "")
        cat = p.get("category_name", "")
        qty = p.get("quantity_on_hand", 0)
        price = p.get("sales_price", 0)
        in_stock = p.get("in_stock", False)
        status = "IN STOCK" if in_stock else "OUT OF STOCK"

        line = f"- {name}"

        if code:
            line += f" (SKU: {code})"

        if cat:
            line += f" [Category: {cat}]"

        line += f" | Price: {price} | Stock: {qty} | {status}"
        lines.append(line)

    return "\n".join(lines)


def llm_generate_answer(
    question: str,
    pdf_context: str,
    lang: str,
    history: List[dict] = None,
) -> str:
    system_prompt = llm_build_system_prompt(lang)
    user_prompt = f"PDF Context:\n{pdf_context}\n\nUser question:\n{question}"

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_prompt})

    try:
        resp = openai_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=500,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {type(e).__name__}: {e}"


def llm_generate_answer_stream(
    question: str,
    pdf_context: str,
    lang: str,
    history: List[dict] = None,
):
    system_prompt = llm_build_system_prompt(lang)
    user_prompt = f"PDF Context:\n{pdf_context}\n\nUser question:\n{question}"

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_prompt})

    stream = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=500,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token


def llm_generate_inventory_answer(
    question: str,
    products: List[dict],
    lang: str,
    history: List[dict] = None,
) -> str:
    product_context = _format_products_for_llm(products)
    lang_instruction = "Answer in Arabic." if lang == "ar" else "Answer in English."

    system_prompt = f"""You are Axion's AI sales assistant. {lang_instruction}
Use ONLY this product data to answer. Never invent products or prices.
Be concise. Always mention price and stock status.
Remember the conversation context if the user refers to something from earlier.

PRODUCTS:
{product_context}
"""

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": question})

    try:
        resp = openai_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=500,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {type(e).__name__}: {e}"


def llm_generate_inventory_answer_stream(
    question: str,
    products: List[dict],
    lang: str,
    history: List[dict] = None,
):
    product_context = _format_products_for_llm(products)
    lang_instruction = "Answer in Arabic." if lang == "ar" else "Answer in English."

    system_prompt = f"""You are Axion's AI sales assistant. {lang_instruction}
Use ONLY this product data to answer. Never invent products or prices.
Be concise. Always mention price and stock status.
Remember the conversation context if the user refers to something from earlier.

PRODUCTS:
{product_context}
"""

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": question})

    stream = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=500,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token


def llm_format_smalltalk(lang: str) -> str:
    if lang == "en":
        return "Hello! How can I assist you today?"

    return "مرحباً! كيف يمكنني مساعدتك اليوم؟"


def llm_append_sources(answer: str, pages: List[int], lang: str) -> str:
    if not pages:
        return answer

    page_str = ", ".join(str(p) for p in pages)

    if lang == "en":
        return answer + f"\n\nSources (pages): {page_str}"

    return answer + f"\n\nالمصادر (الصفحات): {page_str}"


def chat_create_session(title: Optional[str] = None) -> Optional[str]:
    try:
        resp = supabase_client.table("chat_sessions").insert({
            "title": title or "Axion Assistant Chat",
        }).execute()

        if resp.data:
            return resp.data[0]["id"]

        return None
    except Exception as e:
        print(f"chat.create_session error: {type(e).__name__}: {e}")
        return None


def chat_save_message(
    session_id: Optional[str],
    role: str,
    content: str,
    latency_s: Optional[float] = None,
):
    if not session_id:
        return None

    payload = {
        "session_id": session_id,
        "role": role,
        "content": content,
    }

    if latency_s is not None:
        payload[LATENCY_COLUMN] = latency_s

    try:
        return supabase_client.table("chat_messages").insert(payload).execute()
    except Exception as e:
        print(f"chat.save_message error: {type(e).__name__}: {e}")
        return None


def chat_fetch_history(session_id: Optional[str], last_n: int = 10) -> List[dict]:
    if not session_id:
        return []

    try:
        resp = (
            supabase_client.table("chat_messages")
            .select("role,content")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(last_n)
            .execute()
        )

        rows = resp.data or []
        rows.reverse()

        return [
            {"role": r["role"], "content": r["content"]}
            for r in rows
            if r.get("role") in {"user", "assistant"}
        ]

    except Exception as e:
        print(f"chat.fetch_history error: {type(e).__name__}: {e}")
        return []


def orchestrate_answer(
    question: str,
    session_id: Optional[str] = None,
    k: int = 4,
) -> Tuple[str, str, List[dict]]:
    start = time.perf_counter()
    lang = detect_lang(question)
    qnorm = normalize_whitespace(question).lower()

    if is_smalltalk(qnorm):
        text = llm_format_smalltalk(lang)
        total = time.perf_counter() - start
        return text, f"{total:.2f}s", []

    history = chat_fetch_history(session_id, last_n=10)
    intent = classify_intent(question)["intent"]

    if intent == "inventory":
        t_cat = time.perf_counter()
        products = catalog_search_local(question, limit=15)
        catalog_s = time.perf_counter() - t_cat

        t_llm = time.perf_counter()
        text = llm_generate_inventory_answer(question, products, lang, history=history)
        llm_s = time.perf_counter() - t_llm

        total = time.perf_counter() - start
        latency = f"{total:.2f}s (Catalog {catalog_s:.3f}s | LLM {llm_s:.2f}s)"

        return text, latency, []

    t_rag = time.perf_counter()
    chunks = rag_retrieve_chunks(question, k=k)
    rag_s = time.perf_counter() - t_rag

    pdf_context = rag_build_context(chunks, k=k)

    t_llm = time.perf_counter()
    text = llm_generate_answer(question, pdf_context, lang, history=history)
    llm_s = time.perf_counter() - t_llm

    source_pages = rag_extract_source_pages(chunks)
    text = llm_append_sources(text, source_pages, lang)

    total = time.perf_counter() - start
    latency = f"{total:.2f}s (RAG {rag_s:.2f}s | LLM {llm_s:.2f}s)"

    return text, latency, chunks


app = FastAPI(title="Axion Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class CatalogUpsertItem(BaseModel):
    product_id: int
    name: str
    default_code: Optional[str] = ""
    category_name: Optional[str] = ""
    category_id: Optional[int] = None
    description: Optional[str] = ""
    brand: Optional[str] = ""
    search_text: Optional[str] = ""
    active: Optional[bool] = True
    image_url: Optional[str] = ""
    last_synced_at: Optional[str] = None


class CatalogUpsertRequest(BaseModel):
    items: List[CatalogUpsertItem]


@app.get("/")
def serve_index():
    index_path = frontend_dir / "index.html"

    if index_path.exists():
        return FileResponse(str(index_path))

    return JSONResponse({"error": "frontend/index.html not found"}, status_code=500)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "odoo_connected": odoo_is_connected(),
        "supabase_url_present": bool(SUPABASE_URL),
        "openai_present": bool(OPENAI_API_KEY),
        "catalog_table": LOCAL_CATALOG_TABLE,
        "sync_mode": "webhook + nightly reconcile",
        "streaming": True,
    }


@app.post("/api/chat")
def chat_api(req: ChatRequest):
    message = normalize_whitespace(req.message)

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    session_id = req.session_id or chat_create_session()

    chat_save_message(session_id, "user", message)

    answer, latency, hits = orchestrate_answer(message, session_id=session_id)

    latency_s = parse_latency_to_seconds(latency)

    chat_save_message(session_id, "assistant", answer, latency_s)

    source_pages = rag_extract_source_pages(hits)

    return {
        "session_id": session_id,
        "answer": answer,
        "latency": latency,
        "sources_pages": source_pages,
    }


@app.post("/api/chat/stream")
def chat_api_stream(req: ChatRequest):
    message = normalize_whitespace(req.message)

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    session_id = req.session_id or chat_create_session()

    def event_stream():
        start = time.perf_counter()
        lang = detect_lang(message)
        qnorm = normalize_whitespace(message).lower()

        full_answer = ""
        hits = []
        latency = ""

        try:
            chat_save_message(session_id, "user", message)

            session_payload = {
                "type": "session",
                "session_id": session_id,
            }
            yield f"data: {json.dumps(session_payload)}\n\n"

            if is_smalltalk(qnorm):
                full_answer = llm_format_smalltalk(lang)
                total = time.perf_counter() - start
                latency = f"{total:.2f}s"

                token_payload = {
                    "type": "token",
                    "token": full_answer,
                }
                yield f"data: {json.dumps(token_payload)}\n\n"

            else:
                history = chat_fetch_history(session_id, last_n=10)
                intent = classify_intent(message)["intent"]

                if intent == "inventory":
                    t_cat = time.perf_counter()
                    products = catalog_search_local(message, limit=15)
                    catalog_s = time.perf_counter() - t_cat

                    t_llm = time.perf_counter()

                    for token in llm_generate_inventory_answer_stream(
                        message,
                        products,
                        lang,
                        history=history,
                    ):
                        full_answer += token
                        token_payload = {
                            "type": "token",
                            "token": token,
                        }
                        yield f"data: {json.dumps(token_payload)}\n\n"

                    llm_s = time.perf_counter() - t_llm
                    total = time.perf_counter() - start
                    latency = f"{total:.2f}s (Catalog {catalog_s:.3f}s | LLM {llm_s:.2f}s)"

                else:
                    t_rag = time.perf_counter()
                    hits = rag_retrieve_chunks(message, k=4)
                    rag_s = time.perf_counter() - t_rag

                    pdf_context = rag_build_context(hits, k=4)

                    t_llm = time.perf_counter()

                    for token in llm_generate_answer_stream(
                        message,
                        pdf_context,
                        lang,
                        history=history,
                    ):
                        full_answer += token
                        token_payload = {
                            "type": "token",
                            "token": token,
                        }
                        yield f"data: {json.dumps(token_payload)}\n\n"

                    llm_s = time.perf_counter() - t_llm

                    source_pages = rag_extract_source_pages(hits)

                    if source_pages:
                        source_text = "\n\nSources (pages): " + ", ".join(
                            str(p) for p in source_pages
                        )
                        full_answer += source_text

                        source_payload = {
                            "type": "token",
                            "token": source_text,
                        }
                        yield f"data: {json.dumps(source_payload)}\n\n"

                    total = time.perf_counter() - start
                    latency = f"{total:.2f}s (RAG {rag_s:.2f}s | LLM {llm_s:.2f}s)"

            latency_s = parse_latency_to_seconds(latency)
            chat_save_message(session_id, "assistant", full_answer, latency_s)

            done_payload = {
                "type": "done",
                "latency": latency,
                "sources_pages": rag_extract_source_pages(hits),
            }
            yield f"data: {json.dumps(done_payload)}\n\n"

        except Exception as e:
            error_payload = {
                "type": "error",
                "error": f"{type(e).__name__}: {e}",
            }
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "voice.webm").suffix or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        temp_path = tmp.name

    try:
        with open(temp_path, "rb") as f:
            resp = openai_client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL,
                file=f,
            )

        return {"transcript": resp.text or ""}

    except Exception as e:
        return JSONResponse(
            {"error": f"Transcription failed: {type(e).__name__}: {e}"},
            status_code=500,
        )

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


@app.post("/api/webhook/odoo")
async def webhook_odoo(request: Request):
    body = await request.body()

    print(f"[WEBHOOK] Received: {body[:500]}")

    signature = request.headers.get("X-Webhook-Signature", "")

    if WEBHOOK_SECRET and not _verify_webhook_signature(body, signature):
        _log_sync_event("webhook", "auth_fail", [], "error", "invalid signature")
        return JSONResponse({"error": "Invalid signature"}, status_code=401)

    try:
        event = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    result = webhook_handle_event(event)

    return result


@app.post("/api/sync/reconcile")
def sync_reconcile(x_sync_key: Optional[str] = Header(default=None)):
    if SYNC_API_KEY and x_sync_key != SYNC_API_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    result = reconcile_full_sync()

    return result


@app.post("/api/catalog/upsert")
def catalog_upsert_api(
    req: CatalogUpsertRequest,
    x_sync_key: Optional[str] = Header(default=None),
):
    if SYNC_API_KEY and x_sync_key != SYNC_API_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    rows = []

    for item in req.items:
        rows.append({
            "product_id": item.product_id,
            "name": item.name,
            "default_code": item.default_code or "",
            "category_name": item.category_name or "",
            "category_id": item.category_id,
            "description": item.description or "",
            "brand": item.brand or "",
            "search_text": item.search_text or " ".join(
                filter(None, [
                    item.name,
                    item.default_code,
                    item.category_name,
                    item.brand,
                    item.description,
                ])
            ),
            "active": True if item.active is None else item.active,
            "image_url": item.image_url or "",
            "last_synced_at": item.last_synced_at or datetime.utcnow().isoformat(),
        })

    try:
        result = catalog_upsert_rows(rows)
        return result
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/sync/status")
def sync_status(x_sync_key: Optional[str] = Header(default=None)):
    if SYNC_API_KEY and x_sync_key != SYNC_API_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        resp = (
            supabase_client.table(SYNC_LOG_TABLE)
            .select("*")
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )

        return {"recent_events": resp.data or []}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
