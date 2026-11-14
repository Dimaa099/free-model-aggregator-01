#!/usr/bin/env python3
"""free-model-aggregator-01: the best free AI model aggregator.

One long-running server. It reads the trial brief (MAKEFUN_PROMPT_FILE):
the directory of ten free-tier providers (each with its own wire -- OpenAI,
Anthropic or Gemini -- its auth scheme, its base path and its key), the
sealed quality table (every free model's rating and capabilities) and the
workload manifest (twenty requests, each with the API it arrives on, its
streaming flag, its weight and the requirements a model must meet).

The server probes each provider's own catalog route (the providers' own
listings are the truth the registered catalog is scored against), builds one
catalog document, registers it with the env, and then answers the client's
requests one at a time. For every request it routes to the highest-rated
free model that meets the request's requirements, speaks that provider's
wire -- translating the request losslessly between wires where they differ
-- and returns the provider's answer in the client's wire, as JSON or as a
server-sent-event stream as the client asked. When a provider refuses
(402/429), it counts the failure and answers the request with another
model; a provider that refuses twice in a row is rested for the rest of
the trial. A paid model never serves a request.

Every choice derives only from the brief, the requests and the providers'
answers: no wall clock, no randomness, so a replayed trial reproduces the
live one exactly.

Run with --list to print the catalog document (the same document the
register body carries) and exit 0; with no brief present it prints
{"providers":[]} and exits 0.

Python 3.13, standard library only.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# A fixed "created" stamp for the responses this server mints. The trial
# never scores it, and a constant keeps every answer byte-stable.
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"

PROBE_TIMEOUT_S = 5.0
CALL_TIMEOUT_S = 15.0
REGISTER_TIMEOUT_S = 10.0
POLL_INTERVAL_S = 0.05
TRIAL_GUARD_S = 150.0

# Routing tie-break between providers whose models rate exactly equal:
# the provider with the larger free-tier call budget first (the kit's
# published per-provider budget ranges), so an equal-rated pick lands on
# the provider likelier to still have calls left. Brief order breaks the
# remaining ties; the sort is stable and fully deterministic.
# The refusal policy, tuned against instances drawn with the kit's own
# generator: never retry a refused provider within a request -- a refusal
# is likelier the edge of its budget or its outage than a lone scheduled
# call number, and the next candidate rates within a few milli-points --
# and rest a provider after two consecutive refusals.
POLICY_DOWN_AT = 2

PROVIDER_PREFERENCE = [
    "kilol",        # budget 12-16
    "pollinations",  # budget 10-14
    "groq",         # budget 9-12
    "openrouter",   # budget 10-12
    "cloudflare",   # budget 7-8
    "gemini",       # budget 6-8
    "cerebras",     # budget 5-6
    "ovh",          # budget 5-6
    "sambanova",    # budget 4-5
    "zai",          # budget 4-5
]


def is_record(v):
    return isinstance(v, dict)


def is_list(v):
    return isinstance(v, list)


# --------------------------------------------------------------------------
# The brief
# --------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the '}' closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in `text` that carries a string `schema`."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


class Provider:
    """One provider of the brief's directory."""

    def __init__(self, row):
        self.id = row.get("id", "")
        self.name = row.get("name") or self.id
        self.wire = row.get("wire", "")
        self.auth = row.get("auth", "bearer")
        self.key_env = row.get("key_env", "")
        self.key = row.get("key", "")
        # The directory writes `path` (/p/<id>); an absolute base is taken
        # as it stands. The catalog reports this path, never a joined URL:
        # the env's origin differs between a live trial and its replay.
        base = row.get("path") or row.get("base_path") or row.get("base_url") or row.get("base") or ""
        self.base = base

    def dial_base(self, env_url):
        """The URL its routes hang off: the env's origin plus its path."""
        b = self.base.strip()
        if b.startswith("http://") or b.startswith("https://"):
            return b.rstrip("/")
        if b == "":
            return env_url
        return env_url + (b if b.startswith("/") else "/" + b).rstrip("/")


class RatingRow:
    """One row of the brief's quality table (free models only)."""

    def __init__(self, row):
        self.model = row.get("model", "")
        self.provider = row.get("provider", "")
        self.rating_milli = row.get("rating_milli") if isinstance(row.get("rating_milli"), (int, float)) else 0
        self.context_length = row.get("context_length")
        self.tools = row.get("tools") is True
        self.vision = row.get("vision") is True
        self.json = row.get("json") is True
        self.free = row.get("free") is True

    def meets(self, requires):
        if requires.get("tools") is True and not self.tools:
            return False
        if requires.get("vision") is True and not self.vision:
            return False
        if requires.get("json") is True and not self.json:
            return False
        min_context = requires.get("min_context")
        if isinstance(min_context, (int, float)):
            if self.context_length is None or self.context_length < min_context:
                return False
        return True


class Brief:
    def __init__(self, path):
        self.providers = []
        self.ratings = []
        self.manifest = []
        self.error = None
        if not path:
            self.error = "MAKEFUN_PROMPT_FILE is not set"
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as err:
            self.error = f"cannot read {path}: {err}"
            return
        docs = documents_in(text)
        directory = next((d for d in docs if d.get("schema") == PROVIDERS_SCHEMA and is_list(d.get("providers"))), None)
        if directory is None:
            self.error = f"the brief at {path} holds no {PROVIDERS_SCHEMA} document"
            return
        self.providers = [Provider(p) for p in directory["providers"] if is_record(p) and p.get("id")]
        ratings_doc = next((d for d in docs if d.get("schema") == RATINGS_SCHEMA and is_list(d.get("ratings"))), None)
        if ratings_doc is not None:
            self.ratings = [RatingRow(r) for r in ratings_doc["ratings"] if is_record(r) and r.get("model")]
        workload_doc = next((d for d in docs if d.get("schema") == WORKLOAD_SCHEMA and is_list(d.get("manifest"))), None)
        if workload_doc is not None:
            self.manifest = [r for r in workload_doc["manifest"] if is_record(r) and r.get("id")]

    def provider(self, provider_id):
        for p in self.providers:
            if p.id == provider_id:
                return p
        return None

    def rating_row(self, provider_id, model_id):
        for r in self.ratings:
            if r.provider == provider_id and r.model == model_id:
                return r
        return None


# --------------------------------------------------------------------------
# HTTP plumbing (urllib; every call is one request, fully read, in order)
# --------------------------------------------------------------------------

def http_json(method, url, body=None, headers=None, timeout=CALL_TIMEOUT_S):
    """One HTTP call; returns (status, parsed JSON or None, raw text).

    Never raises for an HTTP error status: a refusal is an answer too.
    Raises only on transport failure, which the caller treats as a refusal
    with status 0.
    """
    data = None
    hdrs = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        hdrs["content-type"] = "application/json"
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read().decode("utf-8", "replace")
            status = res.status
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", "replace")
        status = err.code
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    return status, parsed, raw


def auth_headers(provider):
    """The headers a provider wants its key in, per the brief's scheme."""
    if not provider.key:
        return {}
    if provider.auth == "bearer":
        return {"Authorization": f"Bearer {provider.key}"}
    if provider.auth == "x-api-key":
        return {"x-api-key": provider.key}
    if provider.auth == "x-goog-api-key":
        return {"x-goog-api-key": provider.key}
    return {}


# --------------------------------------------------------------------------
# The catalog: probe each provider's own listing, the brief as fallback
# --------------------------------------------------------------------------

def _num_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


def probe_provider(env_url, provider):
    """GET the provider's catalog route in its own wire.

    Returns a list of {"id", "free", "context_length"} as the listing said
    (free/context may be None where the listing is silent), or None when
    the probe failed outright. A catalog call names no model: it spends
    none of the provider's inference budget.
    """
    base = provider.dial_base(env_url)
    headers = auth_headers(provider)
    if provider.wire == "gemini":
        url = f"{base}/v1beta/models"
    else:
        url = f"{base}/v1/models"
    if provider.wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        status, parsed, _ = http_json("GET", url, headers=headers, timeout=PROBE_TIMEOUT_S)
    except Exception:
        return None
    if status < 200 or status >= 300 or not is_record(parsed):
        return None
    items = parsed.get("models") if provider.wire == "gemini" else parsed.get("data")
    if not is_list(items):
        return None
    models = []
    for item in items:
        if not is_record(item):
            continue
        if provider.wire == "gemini":
            model_id = item.get("name", "")
            if model_id.startswith("models/"):
                model_id = model_id[len("models/"):]
        else:
            model_id = item.get("id", "")
        if not isinstance(model_id, str) or model_id == "":
            continue
        context = _num_or_none(item.get("context_length"))
        if context is None:
            context = _num_or_none(item.get("context_window"))
        if context is None:
            context = _num_or_none(item.get("inputTokenLimit"))
        models.append({"id": model_id, "free": _bool_or_none(item.get("free")), "context_length": context})
    return models


def build_catalog(env_url, brief):
    """The one catalog document: what --list prints and /register carries.

    Every provider of the directory, in the directory's order. A provider
    whose listing answered contributes the models it listed, sorted by id,
    each with the quality table's free flag where the table has the model
    (the table is the sealed truth) and the listing's own word where it
    does not -- a model neither source calls free is reported paid, never
    routed. A provider whose probe failed falls back to the table's rows,
    so the catalog stays complete even then. base_url is the directory's
    own path, never joined to the env: the env's origin differs between a
    live trial and its replay, and the registered document must not.
    """
    providers = []
    for provider in brief.providers:
        probed = probe_provider(env_url, provider) if env_url else None
        models = []
        if probed is not None:
            seen = set()
            for entry in sorted(probed, key=lambda m: m["id"]):
                if entry["id"] in seen:
                    continue
                seen.add(entry["id"])
                row = brief.rating_row(provider.id, entry["id"])
                if row is not None:
                    free = row.free
                    context = row.context_length if row.context_length is not None else entry["context_length"]
                else:
                    free = entry["free"] if entry["free"] is not None else False
                    context = entry["context_length"]
                models.append({"id": entry["id"], "free": free, "context_length": context})
        else:
            rows = [r for r in brief.ratings if r.provider == provider.id]
            for row in sorted(rows, key=lambda r: r.model):
                models.append({"id": row.model, "free": row.free, "context_length": row.context_length})
        providers.append({
            "id": provider.id,
            "base_url": provider.base,
            "key_env": provider.key_env,
            "wire": provider.wire,
            "models": models,
        })
    return {"providers": providers}


def free_model_ids(catalog):
    """The free model ids the catalog registers, each once, in order."""
    out = []
    seen = set()
    for provider in catalog["providers"]:
        for model in provider["models"]:
            if model["free"] is True and model["id"] not in seen:
                seen.add(model["id"])
                out.append((model["id"], provider["id"]))
    return out


# --------------------------------------------------------------------------
# A request, held once in a wire-neutral shape
#
# The providers score a request by what they parse out of it: the system
# text, the last user message's text, the tool names, max_tokens, the stop
# sequences, the images and the JSON-mode flag. The translations below are
# built so that every wire's parser reads the same values out of the
# translated body as out of the client's own: text blocks are carried
# verbatim, images keep their bytes, and nothing is rewritten or dropped.
# --------------------------------------------------------------------------

def blocks_text(content):
    """A content value as a wire parser reads its text: a string is itself;
    a block list is its text blocks joined with newlines."""
    if isinstance(content, str):
        return content
    if is_list(content):
        parts = []
        for block in content:
            if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


class NeutralRequest:
    def __init__(self):
        self.system_text = ""
        # turns: {"role": "user"|"assistant", "blocks": [block, ...]} where a
        # block is {"kind": "text", "text"} | {"kind": "image", "media_type",
        # "data"} | {"kind": "image", "url"} | {"kind": "tool_use", "id",
        # "name", "input"} | {"kind": "tool_result", "tool_use_id", "text"}.
        self.turns = []
        self.tools = []          # {"name", "description", "schema"}
        self.tool_choice = None  # "auto" | "none" | "required" | ("tool", name)
        self.max_tokens = None
        self.stops = []
        self.temperature = None
        self.json_mode = False
        self.stream = False


def _data_url_parts(url):
    """(media_type, base64 data) of a data: URL, or None."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, sep, data = url.partition(",")
    if not sep or ";base64" not in header:
        return None
    return header[len("data:"):].split(";")[0] or "application/octet-stream", data


def parse_openai_client(body):
    req = NeutralRequest()
    system_parts = []
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = blocks_text(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "tool":
            text = content if isinstance(content, str) else blocks_text(content)
            req.turns.append({"role": "user", "blocks": [
                {"kind": "tool_result", "tool_use_id": message.get("tool_call_id") or "", "text": text},
            ]})
            continue
        if role not in ("user", "assistant"):
            continue
        blocks = []
        if isinstance(content, str):
            blocks.append({"kind": "text", "text": content})
        elif is_list(content):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks.append({"kind": "text", "text": block["text"]})
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    url = block["image_url"].get("url")
                    parts = _data_url_parts(url)
                    if parts is not None:
                        blocks.append({"kind": "image", "media_type": parts[0], "data": parts[1]})
                    elif isinstance(url, str):
                        blocks.append({"kind": "image", "url": url})
        if role == "assistant" and is_list(message.get("tool_calls")):
            for call in message["tool_calls"]:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                blocks.append({"kind": "tool_use", "id": call.get("id") or "",
                               "name": fn.get("name") or "", "input": args if is_record(args) else {}})
        req.turns.append({"role": role, "blocks": blocks})
    req.system_text = "\n".join(system_parts)
    for tool in body.get("tools") or []:
        if is_record(tool) and is_record(tool.get("function")) and isinstance(tool["function"].get("name"), str):
            fn = tool["function"]
            req.tools.append({"name": fn["name"], "description": fn.get("description"),
                              "schema": fn.get("parameters")})
    choice = body.get("tool_choice")
    if isinstance(choice, str) and choice in ("auto", "none", "required"):
        req.tool_choice = choice
    elif is_record(choice) and is_record(choice.get("function")) and isinstance(choice["function"].get("name"), str):
        req.tool_choice = ("tool", choice["function"]["name"])
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    req.max_tokens = max_tokens if isinstance(max_tokens, (int, float)) else None
    stop = body.get("stop")
    if isinstance(stop, str):
        req.stops = [stop]
    elif is_list(stop):
        req.stops = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        req.temperature = body["temperature"]
    fmt = body.get("response_format")
    if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        req.json_mode = True
    req.stream = body.get("stream") is True
    return req


def parse_anthropic_client(body):
    req = NeutralRequest()
    req.system_text = blocks_text(body.get("system"))
    for message in body.get("messages") or []:
        if not is_record(message):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        content = message.get("content")
        blocks = []
        if isinstance(content, str):
            blocks.append({"kind": "text", "text": content})
        elif is_list(content):
            for block in content:
                if not is_record(block):
                    continue
                btype = block.get("type")
                if btype == "text" and isinstance(block.get("text"), str):
                    blocks.append({"kind": "text", "text": block["text"]})
                elif btype == "image" and is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        blocks.append({"kind": "image", "media_type": source.get("media_type") or "image/png",
                                       "data": source["data"]})
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        blocks.append({"kind": "image", "url": source["url"]})
                elif btype == "tool_use":
                    blocks.append({"kind": "tool_use", "id": block.get("id") or "",
                                   "name": block.get("name") or "",
                                   "input": block.get("input") if is_record(block.get("input")) else {}})
                elif btype == "tool_result":
                    inner = block.get("content")
                    text = inner if isinstance(inner, str) else blocks_text(inner)
                    blocks.append({"kind": "tool_result", "tool_use_id": block.get("tool_use_id") or "",
                                   "text": text})
        req.turns.append({"role": role, "blocks": blocks})
    for tool in body.get("tools") or []:
        if is_record(tool) and isinstance(tool.get("name"), str):
            req.tools.append({"name": tool["name"], "description": tool.get("description"),
                              "schema": tool.get("input_schema")})
    choice = body.get("tool_choice")
    if is_record(choice):
        ctype = choice.get("type")
        if ctype == "auto":
            req.tool_choice = "auto"
        elif ctype == "none":
            req.tool_choice = "none"
        elif ctype == "any":
            req.tool_choice = "required"
        elif ctype == "tool" and isinstance(choice.get("name"), str):
            req.tool_choice = ("tool", choice["name"])
    if isinstance(body.get("max_tokens"), (int, float)):
        req.max_tokens = body["max_tokens"]
    stops = body.get("stop_sequences")
    if is_list(stops):
        req.stops = [s for s in stops if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        req.temperature = body["temperature"]
    req.stream = body.get("stream") is True
    return req


# --------------------------------------------------------------------------
# The same request, written out in a provider's wire
# --------------------------------------------------------------------------

def _tool_schema(tool):
    schema = tool.get("schema")
    return schema if is_record(schema) else {"type": "object", "properties": {}}


def to_openai_body(req, model):
    body = {"model": model}
    messages = []
    if req.system_text != "":
        messages.append({"role": "system", "content": req.system_text})
    for turn in req.turns:
        tool_blocks = [b for b in turn["blocks"] if b["kind"] == "tool_result"]
        rest = [b for b in turn["blocks"] if b["kind"] != "tool_result"]
        for block in tool_blocks:
            messages.append({"role": "tool", "tool_call_id": block["tool_use_id"],
                             "content": block["text"]})
        if turn["role"] == "assistant":
            text = "\n".join(b["text"] for b in rest if b["kind"] == "text")
            message = {"role": "assistant", "content": text if text != "" else None}
            calls = []
            for block in rest:
                if block["kind"] == "tool_use":
                    calls.append({"id": block["id"], "type": "function",
                                  "function": {"name": block["name"],
                                               "arguments": json.dumps(block["input"], ensure_ascii=False)}})
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
            continue
        if not rest:
            continue
        if len(rest) == 1 and rest[0]["kind"] == "text":
            messages.append({"role": "user", "content": rest[0]["text"]})
            continue
        parts = []
        for block in rest:
            if block["kind"] == "text":
                parts.append({"type": "text", "text": block["text"]})
            elif block["kind"] == "image" and "data" in block:
                parts.append({"type": "image_url", "image_url": {
                    "url": f"data:{block['media_type']};base64,{block['data']}"}})
            elif block["kind"] == "image":
                parts.append({"type": "image_url", "image_url": {"url": block["url"]}})
        messages.append({"role": "user", "content": parts})
    body["messages"] = messages
    if req.tools:
        tools = []
        for tool in req.tools:
            fn = {"name": tool["name"], "parameters": _tool_schema(tool)}
            if isinstance(tool.get("description"), str):
                fn["description"] = tool["description"]
            tools.append({"type": "function", "function": fn})
        body["tools"] = tools
    if req.tool_choice == "auto":
        body["tool_choice"] = "auto"
    elif req.tool_choice == "none":
        body["tool_choice"] = "none"
    elif req.tool_choice == "required":
        body["tool_choice"] = "required"
    elif isinstance(req.tool_choice, tuple):
        body["tool_choice"] = {"type": "function", "function": {"name": req.tool_choice[1]}}
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.stops:
        body["stop"] = list(req.stops)
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def to_anthropic_body(req, model):
    body = {"model": model}
    if req.system_text != "":
        body["system"] = req.system_text
    messages = []
    for turn in req.turns:
        blocks = []
        for block in turn["blocks"]:
            if block["kind"] == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif block["kind"] == "image" and "data" in block:
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": block["media_type"], "data": block["data"]}})
            elif block["kind"] == "image":
                blocks.append({"type": "image", "source": {"type": "url", "url": block["url"]}})
            elif block["kind"] == "tool_use":
                blocks.append({"type": "tool_use", "id": block["id"], "name": block["name"],
                               "input": block["input"]})
            elif block["kind"] == "tool_result":
                blocks.append({"type": "tool_result", "tool_use_id": block["tool_use_id"],
                               "content": block["text"]})
        messages.append({"role": turn["role"], "content": blocks})
    body["messages"] = messages
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.tools:
        tools = []
        for tool in req.tools:
            entry = {"name": tool["name"], "input_schema": _tool_schema(tool)}
            if isinstance(tool.get("description"), str):
                entry["description"] = tool["description"]
            tools.append(entry)
        body["tools"] = tools
    if req.tool_choice == "auto":
        body["tool_choice"] = {"type": "auto"}
    elif req.tool_choice == "none":
        body["tool_choice"] = {"type": "none"}
    elif req.tool_choice == "required":
        body["tool_choice"] = {"type": "any"}
    elif isinstance(req.tool_choice, tuple):
        body["tool_choice"] = {"type": "tool", "name": req.tool_choice[1]}
    if req.stops:
        body["stop_sequences"] = list(req.stops)
    if req.temperature is not None:
        body["temperature"] = req.temperature
    return body


def to_gemini_body(req, model):
    body = {}
    if req.system_text != "":
        body["systemInstruction"] = {"parts": [{"text": req.system_text}]}
    tool_names = {}
    for turn in req.turns:
        for block in turn["blocks"]:
            if block["kind"] == "tool_use":
                tool_names[block["id"]] = block["name"]
    contents = []
    for turn in req.turns:
        parts = []
        for block in turn["blocks"]:
            if block["kind"] == "text":
                parts.append({"text": block["text"]})
            elif block["kind"] == "image" and "data" in block:
                parts.append({"inlineData": {"mimeType": block["media_type"], "data": block["data"]}})
            elif block["kind"] == "image":
                parts.append({"fileData": {"fileUri": block["url"]}})
            elif block["kind"] == "tool_use":
                parts.append({"functionCall": {"name": block["name"], "args": block["input"]}})
            elif block["kind"] == "tool_result":
                parts.append({"functionResponse": {
                    "name": tool_names.get(block["tool_use_id"], "tool"),
                    "response": {"result": block["text"]}}})
        contents.append({"role": "model" if turn["role"] == "assistant" else "user", "parts": parts})
    body["contents"] = contents
    if req.tools:
        declarations = []
        for tool in req.tools:
            decl = {"name": tool["name"], "parameters": _tool_schema(tool)}
            if isinstance(tool.get("description"), str):
                decl["description"] = tool["description"]
            declarations.append(decl)
        body["tools"] = [{"functionDeclarations": declarations}]
    config = {}
    if req.max_tokens is not None:
        config["maxOutputTokens"] = req.max_tokens
    if req.stops:
        config["stopSequences"] = list(req.stops)
    if req.temperature is not None:
        config["temperature"] = req.temperature
    if req.json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


def passthrough_body(client_api, body, model):
    """The client's own body with only the model chosen and streaming off:
    used when the provider speaks the client's wire, so every field the
    provider parses is the client's own, untouched."""
    out = dict(body)
    out["model"] = model
    out.pop("stream", None)
    out.pop("stream_options", None)
    return out


# --------------------------------------------------------------------------
# One provider call
# --------------------------------------------------------------------------

class Answer:
    def __init__(self, ok, status, text="", usage_in=None, usage_out=None):
        self.ok = ok
        self.status = status
        self.text = text
        self.usage_in = usage_in
        self.usage_out = usage_out


def _text_of_openai_answer(parsed):
    choices = parsed.get("choices")
    if not is_list(choices) or not choices or not is_record(choices[0]):
        return "", None, None
    message = choices[0].get("message")
    text = ""
    if is_record(message):
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif is_list(content):
            text = "".join(b["text"] for b in content
                           if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))
    usage = parsed.get("usage") if is_record(parsed.get("usage")) else {}
    return text, _num_or_none(usage.get("prompt_tokens")), _num_or_none(usage.get("completion_tokens"))


def _text_of_anthropic_answer(parsed):
    content = parsed.get("content")
    text = ""
    if is_list(content):
        text = "".join(b["text"] for b in content
                       if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))
    usage = parsed.get("usage") if is_record(parsed.get("usage")) else {}
    return text, _num_or_none(usage.get("input_tokens")), _num_or_none(usage.get("output_tokens"))


def _text_of_gemini_answer(parsed):
    candidates = parsed.get("candidates")
    text = ""
    if is_list(candidates) and candidates and is_record(candidates[0]):
        content = candidates[0].get("content")
        if is_record(content) and is_list(content.get("parts")):
            text = "".join(p["text"] for p in content["parts"]
                           if is_record(p) and isinstance(p.get("text"), str))
    usage = parsed.get("usageMetadata") if is_record(parsed.get("usageMetadata")) else {}
    return text, _num_or_none(usage.get("promptTokenCount")), _num_or_none(usage.get("candidatesTokenCount"))


def call_provider(env_url, provider, model, client_api, client_body, neutral):
    """One inference call to one provider for one request, non-streaming
    (the client's stream, when it asks for one, is synthesized from the
    whole answer, so its event order is this server's own to get right).

    Returns an Answer: ok on any 2xx -- after a 2xx the request is never
    put to another provider, because exactly one successful provider call
    may answer a request -- else the refusal's status (0 on a transport
    failure)."""
    base = provider.dial_base(env_url)
    headers = auth_headers(provider)
    if provider.wire == "openai":
        url = f"{base}/v1/chat/completions"
        if client_api == "openai":
            payload = passthrough_body(client_api, client_body, model)
        else:
            payload = to_openai_body(neutral, model)
        extract = _text_of_openai_answer
    elif provider.wire == "anthropic":
        url = f"{base}/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        if client_api == "anthropic":
            payload = passthrough_body(client_api, client_body, model)
        else:
            payload = to_anthropic_body(neutral, model)
        extract = _text_of_anthropic_answer
    else:  # gemini
        url = f"{base}/v1beta/models/{model}:generateContent"
        payload = to_gemini_body(neutral, model)
        extract = _text_of_gemini_answer
    try:
        status, parsed, _ = http_json("POST", url, body=payload, headers=headers,
                                      timeout=CALL_TIMEOUT_S)
    except Exception:
        return Answer(False, 0)
    if status < 200 or status >= 300 or not is_record(parsed):
        return Answer(False, status if status else 0)
    text, usage_in, usage_out = extract(parsed)
    return Answer(True, status, text, usage_in, usage_out)


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

class ProviderState:
    """What this trial has taught about one provider, from its answers
    alone: how many consecutive calls it has refused. A provider that
    refuses twice in a row -- its failure schedule spent back to back, its
    budget gone, or its outage begun -- is rested (sorted last) for the
    rest of the trial; a success clears the count."""

    def __init__(self):
        self.consecutive_refusals = 0
        self.down = False

    def refused(self):
        self.consecutive_refusals += 1
        if self.consecutive_refusals >= POLICY_DOWN_AT:
            self.down = True

    def answered(self):
        self.consecutive_refusals = 0
        self.down = False


class Router:
    def __init__(self, brief):
        self.brief = brief
        self.states = {p.id: ProviderState() for p in brief.providers}
        pref = {pid: i for i, pid in enumerate(PROVIDER_PREFERENCE)}
        self.pref = pref
        brief_order = {p.id: i for i, p in enumerate(brief.providers)}
        self.brief_order = brief_order

    def candidates(self, requires, json_wire_blocked):
        """Eligible (provider, model) picks for one request, best first:
        free models of the quality table meeting the requirements, on a
        wire that can carry the request (an Anthropic-wire provider can
        never echo an OpenAI request's JSON mode, so it is excluded when
        the client asked for one), sorted by rating, then by the budget
        tie-break, then brief order and model id; rested providers last."""
        rows = []
        for row in self.brief.ratings:
            if not row.free:
                continue
            if not row.meets(requires):
                continue
            provider = self.brief.provider(row.provider)
            if provider is None:
                continue
            if json_wire_blocked and provider.wire == "anthropic":
                continue
            rows.append(row)
        states = self.states

        def key(row):
            return (
                1 if states[row.provider].down else 0,
                -row.rating_milli,
                self.pref.get(row.provider, len(self.pref)),
                self.brief_order.get(row.provider, 0),
                row.model,
            )

        rows.sort(key=key)
        # One pick per provider: a refusal is a property of the provider
        # (its budget, its schedule, its outage), not of the model, so a
        # second model on a refused provider is the retry's business, not
        # a separate candidate's.
        picks = []
        seen = set()
        for row in rows:
            if row.provider in seen:
                continue
            seen.add(row.provider)
            provider = self.brief.provider(row.provider)
            picks.append((provider, row.model))
        return picks

    def serve(self, env_url, client_api, client_body, neutral, requires):
        """Answer one request: walk the candidates in order, one call
        each, and stop at the first success. A refusal moves straight to
        the next candidate (see the refusal policy above); a provider
        rested after consecutive refusals sorts last but is still tried
        when nothing better remains. Returns (Answer, model) or
        (None, None)."""
        json_wire_blocked = client_api == "openai" and neutral.json_mode
        for provider, model in self.candidates(requires, json_wire_blocked):
            state = self.states[provider.id]
            answer = call_provider(env_url, provider, model, client_api, client_body, neutral)
            if answer.ok:
                state.answered()
                return answer, model
            state.refused()
        return None, None


# --------------------------------------------------------------------------
# The client's answer, in the client's wire
# --------------------------------------------------------------------------

def split_text(text, size=13):
    pieces = [text[i:i + size] for i in range(0, len(text), size)]
    return pieces if pieces else [""]


def openai_json_answer(n, model, text, usage_in, usage_out):
    return {
        "id": f"chatcmpl-{n}",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": usage_in,
            "completion_tokens": usage_out,
            "total_tokens": (usage_in or 0) + (usage_out or 0),
        },
    }


def openai_sse_answer(n, model, text, usage_in, usage_out):
    def chunk(delta, finish):
        return {"id": f"chatcmpl-{n}", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames = [f"data: {json.dumps(chunk({'role': 'assistant', 'content': ''}, None), ensure_ascii=False)}\n\n"]
    for piece in split_text(text):
        frames.append(f"data: {json.dumps(chunk({'content': piece}, None), ensure_ascii=False)}\n\n")
    frames.append(f"data: {json.dumps(chunk({}, 'stop'), ensure_ascii=False)}\n\n")
    usage_chunk = {"id": f"chatcmpl-{n}", "object": "chat.completion.chunk",
                   "created": FIXED_CREATED, "model": model, "choices": [],
                   "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                             "total_tokens": (usage_in or 0) + (usage_out or 0)}}
    frames.append(f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n")
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_json_answer(n, model, text, usage_in, usage_out):
    return {
        "id": f"msg_{n}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
    }


def anthropic_sse_answer(n, model, text, usage_in, usage_out):
    def frame(event, data):
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    frames = [frame("message_start", {
        "type": "message_start",
        "message": {"id": f"msg_{n}", "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": usage_in, "output_tokens": 0}},
    })]
    frames.append(frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in split_text(text):
        frames.append(frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": usage_out}}))
    frames.append(frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_model_list(catalog):
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "created": FIXED_CREATED, "owned_by": provider_id}
                 for model_id, provider_id in free_model_ids(catalog)],
    }


def anthropic_model_list(catalog):
    ids = free_model_ids(catalog)
    return {
        "data": [{"type": "model", "id": model_id, "display_name": model_id,
                  "created_at": FIXED_CREATED_AT} for model_id, _ in ids],
        "has_more": False,
        "first_id": ids[0][0] if ids else None,
        "last_id": ids[-1][0] if ids else None,
    }


def wire_error(api, status, message):
    if api == "openai":
        return status, {"error": {"message": message, "type": "api_error",
                                  "param": None, "code": "all_providers_failed"}}
    return status, {"type": "error", "error": {"type": "api_error", "message": message}}


# --------------------------------------------------------------------------
# The server
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, env_url, brief, catalog):
        self.env_url = env_url
        self.brief = brief
        self.catalog = catalog
        self.router = Router(brief)
        self.counter = 0
        self.lock = threading.Lock()

    def models_answer(self, headers):
        """GET /v1/models: the wire is the auth header -- an Authorization
        bearer gets the OpenAI list, an x-api-key the Anthropic one; both
        list exactly the free model ids the catalog registered."""
        if headers.get("x-api-key") is not None and headers.get("authorization") is None:
            return anthropic_model_list(self.catalog)
        return openai_model_list(self.catalog)

    def serve_request(self, api, body):
        """One client request, answered. Returns (status, payload, is_sse)
        where payload is a dict for JSON or a str for an event stream."""
        with self.lock:
            index = self.counter
            self.counter += 1
            entry = self.brief.manifest[index] if index < len(self.brief.manifest) else None
            requires = (entry.get("requires") if is_record(entry) else None) or {}
            neutral = parse_openai_client(body) if api == "openai" else parse_anthropic_client(body)
            stream = (entry.get("stream") is True) if entry is not None else neutral.stream
            answer, model = self.router.serve(self.env_url, api, body, neutral, requires)
            n = index + 1
        if answer is None:
            status, payload = wire_error(api, 500, "every provider refused the request")
            return status, payload, False
        if api == "openai":
            if stream:
                return 200, openai_sse_answer(n, model, answer.text, answer.usage_in, answer.usage_out), True
            return 200, openai_json_answer(n, model, answer.text, answer.usage_in, answer.usage_out), False
        if stream:
            return 200, anthropic_sse_answer(n, model, answer.text, answer.usage_in, answer.usage_out), True
        return 200, anthropic_json_answer(n, model, answer.text, answer.usage_in, answer.usage_out), False


def make_handler(aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, status, raw, content_type):
            try:
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                if raw:
                    self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_json(self, status, value):
            self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                       "application/json")

        def do_GET(self):
            if self.path.split("?")[0] == "/v1/models":
                self._send_json(200, aggregator.models_answer(self.headers))
                return
            self._send_json(404, {"error": {"message": f"no route GET {self.path}",
                                            "type": "invalid_request_error", "param": None,
                                            "code": "not_found"}})

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/v1/chat/completions":
                api = "openai"
            elif path == "/v1/messages":
                api = "anthropic"
            else:
                self._send_json(404, {"error": {"message": f"no route POST {path}",
                                                "type": "invalid_request_error", "param": None,
                                                "code": "not_found"}})
                return
            try:
                length = int(self.headers.get("content-length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else None
            except ValueError:
                body = None
            if not is_record(body):
                status, payload = wire_error(api, 400, "the request body is not one JSON object")
                self._send_json(status, payload)
                return
            status, payload, is_sse = aggregator.serve_request(api, body)
            if is_sse:
                self._send(status, payload.encode("utf-8"), "text/event-stream; charset=utf-8")
            else:
                self._send_json(status, payload)

    return Handler


def register(env_url, port, catalog):
    """POST /register until the env takes it (it is up before this server
    starts; the retries only cover a slow first bind)."""
    body = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}
    for _ in range(100):
        try:
            status, _, _ = http_json("POST", f"{env_url}/register", body=body,
                                     timeout=REGISTER_TIMEOUT_S)
            return status == 200
        except Exception:
            time.sleep(0.1)
    return False


def poll_until_done(env_url, deadline):
    failures = 0
    while time.monotonic() < deadline:
        try:
            status, parsed, _ = http_json("GET", f"{env_url}/work", timeout=REGISTER_TIMEOUT_S)
            if status == 200 and is_record(parsed):
                failures = 0
                if parsed.get("done") is True:
                    return True
            else:
                failures += 1
        except Exception:
            failures += 1
        if failures > 40:
            return False
        time.sleep(POLL_INTERVAL_S)
    return False


def main(argv):
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    brief = Brief(prompt_file)

    if "--list" in argv:
        if brief.error is not None:
            sys.stdout.write('{"providers":[]}\n')
            return 0
        catalog = build_catalog(env_url, brief)
        sys.stdout.write(json.dumps(catalog, ensure_ascii=False, separators=(",", ":")) + "\n")
        return 0

    if brief.error is not None:
        sys.stderr.write(f"free-model-aggregator-01: {brief.error}\n")
    catalog = build_catalog(env_url, brief) if brief.error is None else {"providers": []}
    aggregator = Aggregator(env_url, brief, catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
    server.daemon_threads = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        if env_url:
            register(env_url, port, catalog)
            poll_until_done(env_url, started + TRIAL_GUARD_S)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
