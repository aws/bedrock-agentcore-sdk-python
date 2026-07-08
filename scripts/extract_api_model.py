#!/usr/bin/env python3
# =============================================================================
# extract_api_model.py  —  Python SDK -> doc-model JSON
# =============================================================================
# Introspects the installed `bedrock-agentcore` package and emits the shared
# doc-model JSON (schema v1, see _shared/render_adoc.py) that the shared
# renderer turns into .adoc.
#
# Runs on a GitHub-hosted runner AFTER `pip install bedrock-agentcore`, so the
# import below resolves against the real published wheel.
#
# NOTE: if these workflows are later consolidated into a single shared reusable
# workflow, this script is vendored there and selected via a `language: python`
# input. It lives here now only so the draft is self-contained and runnable in
# isolation.
#
# Docstring style: the SDK uses Google-style docstrings. We do a light parse
# (Args/Returns/Raises/Example sections). We intentionally keep the parser
# small; anything we can't classify falls through into `description` verbatim.
# =============================================================================

import argparse
import importlib
import inspect
import json
import re
import sys

PACKAGE = "bedrock_agentcore"

# All public modules to document (decision: include ALL of them).
# Each maps to one group == one .adoc file.
GROUPS = [
    ("runtime", "Runtime", "bedrock_agentcore.runtime"),
    ("memory", "Memory", "bedrock_agentcore.memory"),
    ("identity", "Identity", "bedrock_agentcore.identity"),
    ("tools", "Built-in Tools", "bedrock_agentcore.tools"),
    ("gateway", "Gateway", "bedrock_agentcore.gateway"),
    ("policy", "Policy", "bedrock_agentcore.policy"),
    ("evaluation", "Evaluation", "bedrock_agentcore.evaluation"),
    ("config-bundle", "Configuration Bundles", "bedrock_agentcore.config_bundle"),
    ("payments", "Payments", "bedrock_agentcore.payments"),
    ("knowledge-base", "Knowledge Base", "bedrock_agentcore.knowledge_base"),
]

_SECTION_RE = re.compile(r"^\s*(Args|Arguments|Returns|Raises|Example|Examples):\s*$")
_ARG_RE = re.compile(r"^\s+(\w+)\s*(?:\(([^)]+)\))?:\s*(.*)$")

# The SDK mixes Google-style ("Args:") and reST-style (":param x:") docstrings,
# so we also recognize the reST field forms and pull them out of the prose.
_REST_PARAM_RE = re.compile(r"^\s*:param\s+(\w+):\s*(.*)$")
_REST_RETURNS_RE = re.compile(r"^\s*:returns?:\s*(.*)$")
_REST_RAISES_RE = re.compile(r"^\s*:raises?\s+([\w.]+):\s*(.*)$")


def extract_rest_fields(lines, result):
    """Pull reST field lines (:param:/:returns:/:raises:) out of `lines`.

    Returns the remaining (non-field) lines so they can form the description.
    Mutates `result` in place, matching the Google parser's output shape.
    """
    kept = []
    for line in lines:
        m = _REST_PARAM_RE.match(line)
        if m:
            result["params"].append(
                {
                    "name": m.group(1),
                    "type": None,
                    "required": True,
                    "description": m.group(2).strip(),
                }
            )
            continue
        m = _REST_RETURNS_RE.match(line)
        if m:
            result["returns"] = {"type": None, "description": m.group(1).strip()}
            continue
        m = _REST_RAISES_RE.match(line)
        if m:
            result["raises"].append({"type": m.group(1), "description": m.group(2).strip()})
            continue
        kept.append(line)
    return kept


def parse_google_docstring(doc):
    """Very small Google-style docstring parser -> structured dict."""
    result = {"summary": "", "description": "", "params": [], "returns": None, "raises": [], "examples": []}
    if not doc:
        return result
    lines = inspect.cleandoc(doc).splitlines()

    # summary = first paragraph
    i = 0
    summary = []
    while i < len(lines) and lines[i].strip():
        summary.append(lines[i].strip())
        i += 1
    result["summary"] = " ".join(summary)

    section = "description"
    desc, example_buf = [], []
    while i < len(lines):
        line = lines[i]
        m = _SECTION_RE.match(line)
        if m:
            name = m.group(1).lower()
            section = {
                "args": "args",
                "arguments": "args",
                "returns": "returns",
                "raises": "raises",
                "example": "example",
                "examples": "example",
            }[name]
            i += 1
            continue
        if section == "description":
            desc.append(line)
        elif section == "args":
            am = _ARG_RE.match(line)
            if am:
                result["params"].append(
                    {
                        "name": am.group(1),
                        "type": (am.group(2) or "").strip() or None,
                        "required": "optional" not in (am.group(2) or "").lower(),
                        "description": am.group(3).strip(),
                    }
                )
            elif result["params"] and line.strip():
                result["params"][-1]["description"] += " " + line.strip()
        elif section == "returns":
            if line.strip():
                if result["returns"] is None:
                    result["returns"] = {"type": None, "description": line.strip()}
                else:
                    result["returns"]["description"] += " " + line.strip()
        elif section == "raises":
            am = _ARG_RE.match(line)
            if am:
                result["raises"].append({"type": am.group(1), "description": am.group(3).strip()})
        elif section == "example":
            example_buf.append(line)
        i += 1  # always advance — non-header branches above don't, else infinite loop

    # Second pass: some docstrings use reST fields (:param:/:returns:/:raises:)
    # instead of, or mixed with, Google sections. Pull those out of the prose.
    desc = extract_rest_fields(desc, result)
    result["description"] = "\n".join(desc).strip()
    if example_buf:
        code = "\n".join(example_buf).strip()
        # strip a leading ```python fence if the docstring used one
        code = re.sub(r"^```\w*\n?|\n?```$", "", code).strip()
        if code:
            result["examples"].append({"lang": "python", "code": code})
    return result


def _own_docstring(obj):
    """Return obj's docstring, but suppress ones merely inherited from `object`.

    Classes/methods that don't define their own docstring inherit boilerplate
    like "Initialize self. See help(type(self))..." from object.__init__ /
    object.__new__ — noise we don't want in the reference.
    """
    doc = inspect.getdoc(obj)
    if not doc:
        return None
    for base in (object.__init__, object.__new__, object):
        if doc == inspect.getdoc(base):
            return None
    return doc


def entry_from_object(name, obj):
    """Build a doc-model entry for a class or function."""
    doc = parse_google_docstring(_own_docstring(obj))
    try:
        signature = inspect.signature(obj)
        # Drop the implicit `self`/`cls` receiver from method signatures.
        params = [p for p in signature.parameters.values() if p.name not in ("self", "cls")]
        signature = signature.replace(parameters=params)
        sig = f"{name}{signature}"
    except (ValueError, TypeError):
        sig = name

    kind = "class" if inspect.isclass(obj) else "function"
    entry = {
        "kind": kind,
        "name": name,
        "signature": sig,
        "summary": doc["summary"],
        "description": doc["description"],
        "params": doc["params"],
        "returns": doc["returns"],
        "raises": doc["raises"],
        "examples": doc["examples"],
        "members": [],
    }

    if kind == "class":
        for mname, mobj in inspect.getmembers(obj, predicate=inspect.isfunction):
            if mname.startswith("_") and mname != "__init__":
                continue
            if mobj.__qualname__.split(".")[0] != obj.__name__:
                continue  # skip inherited members
            entry["members"].append(entry_from_object(mname, mobj))
    return entry


def collect_public_names(module):
    """Public API of a module = its __all__, else non-underscore attrs."""
    names = getattr(module, "__all__", None)
    if names:
        return list(names)
    return [n for n in dir(module) if not n.startswith("_")]


def build_group(gid, title, modname):
    try:
        module = importlib.import_module(modname)
    except Exception as e:  # noqa: BLE001 — module may not exist in a given version
        print(f"  skip {modname}: {e}", file=sys.stderr)
        return None

    summary = (inspect.getdoc(module) or "").split("\n\n")[0]
    entries = []
    for name in collect_public_names(module):
        # Some modules (e.g. evaluation) expose symbols via a lazy __getattr__
        # that RAISES ImportError for optional extras rather than returning a
        # default — so we can't rely on getattr's default and must catch.
        try:
            obj = getattr(module, name, None)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {modname}.{name}: {type(e).__name__}", file=sys.stderr)
            continue
        if inspect.isclass(obj) or inspect.isfunction(obj):
            # only document objects actually defined in this package
            if getattr(obj, "__module__", "").startswith(PACKAGE):
                entries.append(entry_from_object(name, obj))
    if not entries:
        return None
    return {"id": gid, "title": title, "summary": summary, "entries": entries}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output doc-model JSON path")
    args = ap.parse_args()

    importlib.import_module(PACKAGE)
    # The package exposes its version via distribution metadata, not a
    # __version__ attribute, so read it from there (fall back gracefully).
    try:
        from importlib.metadata import version as _dist_version

        version = _dist_version("bedrock-agentcore")
    except Exception:  # noqa: BLE001
        version = "unknown"

    groups = []
    for gid, title, modname in GROUPS:
        g = build_group(gid, title, modname)
        if g:
            groups.append(g)

    model = {
        "source": "python-sdk",
        "package": "bedrock-agentcore",
        "version": version,
        "language": "python",
        "groups": groups,
    }
    with open(args.out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Wrote doc-model: {len(groups)} groups, version {version}", file=sys.stderr)


if __name__ == "__main__":
    main()
