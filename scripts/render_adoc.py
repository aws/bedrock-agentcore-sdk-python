#!/usr/bin/env python3
# =============================================================================
# render_adoc.py  —  shared doc-model -> AsciiDoc renderer
# =============================================================================
# Renders a normalized "doc-model" JSON (produced by any of the three
# extractors) into .adoc files for the documentation repository.
#
# It is deliberately source-agnostic: the Python extractor, the TypeScript
# TypeDoc extractor, and the CLI --help extractor all emit the SAME doc-model
# schema, so this one renderer produces consistent output for all three.
#
# NOTE: if these workflows are later consolidated into a single shared reusable
# workflow, this file is vendored there ONCE and every source repo's caller
# invokes it. Keep it dependency-free (stdlib only) so it drops cleanly into any
# runner.
#
# -----------------------------------------------------------------------------
# doc-model schema (v1) — the contract every extractor must emit:
#   {
#     "source":   "python-sdk" | "ts-sdk" | "cli",
#     "package":  "bedrock-agentcore",
#     "version":  "1.16.0",
#     "language": "python" | "typescript" | "cli",
#     "groups": [                      # a group -> one .adoc file
#       {
#         "id":       "runtime",       # -> <prefix>-runtime.adoc, and anchor id
#         "title":    "Runtime",
#         "summary":  "Runtime management and application context.",
#         "entries": [                 # classes / functions / commands
#           {
#             "kind":      "class" | "function" | "command",
#             "name":      "BedrockAgentCoreApp",
#             "signature": "BedrockAgentCoreApp(params)",
#             "summary":   "one-line summary",
#             "description": "longer prose (optional)",
#             "params": [ {"name","type","required","description"} ],
#             "returns":  {"type","description"} | null,
#             "raises":   [ {"type","description"} ],
#             "examples": [ {"lang","code"} ],
#             "members":  [ <entry>, ... ]   # methods on a class (recursive)
#           }
#         ]
#       }
#     ]
#   }
# -----------------------------------------------------------------------------

import argparse
import json
import os
import re
import sys
import textwrap

SCHEMA_VERSION = 1


def esc(text):
    """Escape AsciiDoc-significant characters in inline text."""
    if not text:
        return ""
    # Guard the couple of chars that start AsciiDoc markup in running prose.
    return text.replace("|", "\\|").replace("{", "\\{")


# Match markdown code fences that may be indented (reST/Google docstrings often
# indent example blocks). `re.MULTILINE` lets ^ match each line start; the
# leading-whitespace groups are stripped from the captured code.
_FENCE_RE = re.compile(r"^[ \t]*```(\w*)[ \t]*\n(.*?)\n[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)


def render_prose(text):
    """Render description prose that may contain markdown ``` code fences.

    Docstrings/TSDoc frequently embed fenced code blocks in the description
    (not just in @example). Left alone they leak literal backticks into the
    AsciiDoc. Split the prose on fences: escape the prose spans, and convert
    each fenced block into an AsciiDoc [source] block (verbatim, not escaped).
    """
    if not text:
        return []
    out = []
    pos = 0
    for m in _FENCE_RE.finditer(text):
        before = text[pos : m.start()].strip()
        if before:
            out.append(esc(before))
            out.append("")
        lang = m.group(1) or ""
        out.append(f"[source,{lang}]" if lang else "[source]")
        out.append("----")
        # Dedent the captured code (fences are often indented in docstrings).
        out.append(textwrap.dedent(m.group(2)).rstrip())
        out.append("----")
        out.append("")
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        out.append(esc(tail))
    return out


def block(lines):
    return "\n".join(lines)


def render_params(params, out):
    if not params:
        return
    out.append("*Parameters*")
    out.append("")
    for p in params:
        req = "" if p.get("required") else " _(optional)_"
        typ = f"`{p['type']}`" if p.get("type") else ""
        out.append(f"`{p['name']}`{req} {typ}::")
        out.append(esc(p.get("description", "")) or "_No description._")
    out.append("")


def render_returns(returns, out):
    if not returns:
        return
    typ = f"`{returns['type']}` — " if returns.get("type") else ""
    out.append("*Returns*")
    out.append("")
    out.append(f"{typ}{esc(returns.get('description', ''))}".strip())
    out.append("")


def render_raises(raises, out):
    if not raises:
        return
    out.append("*Raises*")
    out.append("")
    for r in raises:
        out.append(f"`{r.get('type', 'Error')}`:: {esc(r.get('description', ''))}")
    out.append("")


def clean_example_code(code):
    """Strip stray markdown fence lines from example code.

    Extractors try to remove ``` fences, but real docstrings put them mid-buffer
    (e.g. a fence followed by extra "Notes:"/"Thread Safety:" prose swept into
    the example). Since this text is already going inside an AsciiDoc [source]
    block, any line that is just a fence is spurious — drop it, and drop trailing
    non-code prose that follows a closing fence.
    """
    lines = code.split("\n")
    kept = []
    closed = False
    for line in lines:
        if re.match(r"^[ \t]*```", line):
            # A closing fence marks the end of the real code; ignore the fence
            # line itself and stop taking subsequent prose.
            if kept:
                closed = True
            continue
        if closed and line.strip() == "":
            continue
        if closed and line.strip():
            break  # prose after the closing fence — not part of the example
        kept.append(line)
    return textwrap.dedent("\n".join(kept)).strip()


def render_examples(examples, out):
    for ex in examples or []:
        lang = ex.get("lang", "text")
        code = clean_example_code(ex.get("code", ""))
        if not code:
            continue
        out.append(f"[source,{lang}]")
        out.append("----")
        out.append(code)
        out.append("----")
        out.append("")


def render_entry(entry, level, out):
    """Render a single class/function/command as an AsciiDoc section."""
    heading = "=" * level
    name = entry.get("name", "")
    out.append(f"{heading} {name}")
    out.append("")

    sig = entry.get("signature")
    if sig:
        out.append("[source]")
        out.append("----")
        out.append(sig)
        out.append("----")
        out.append("")

    if entry.get("summary"):
        out.extend(render_prose(entry["summary"]))
        out.append("")
    if entry.get("description"):
        out.extend(render_prose(entry["description"]))
        out.append("")

    render_params(entry.get("params"), out)
    render_returns(entry.get("returns"), out)
    render_raises(entry.get("raises"), out)
    render_examples(entry.get("examples"), out)

    # methods / subcommands nest one heading level deeper
    for member in entry.get("members", []):
        render_entry(member, level + 1, out)


def render_group(group, model, prefix):
    """Render one group -> one .adoc file body (string)."""
    out = []
    gid = group["id"]
    # AsciiDoc anchor so the TOC / other pages can xref into it.
    out.append(f"[[{prefix}-{gid}]]")
    out.append(f"= {group['title']}")
    out.append("")
    out.append(f"_Auto-generated from `{model['package']}` v{model['version']} — do not edit by hand._")
    out.append("")
    if group.get("summary"):
        out.extend(render_prose(group["summary"]))
        out.append("")

    for entry in group.get("entries", []):
        render_entry(entry, level=2, out=out)

    return block(out).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description="Render doc-model JSON to .adoc")
    ap.add_argument("--model", required=True, help="path to doc-model JSON")
    ap.add_argument("--out-dir", required=True, help="output dir for .adoc files")
    ap.add_argument(
        "--prefix",
        required=True,
        help="filename + anchor prefix, e.g. 'python-sdk', 'ts-sdk', 'cli'",
    )
    args = ap.parse_args()

    with open(args.model) as f:
        model = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for group in model.get("groups", []):
        body = render_group(group, model, args.prefix)
        fname = f"{args.prefix}-{group['id']}.adoc"
        path = os.path.join(args.out_dir, fname)
        with open(path, "w") as f:
            f.write(body)
        written.append(fname)

    # Emit the list of includes the caller can splice into the docs TOC file.
    manifest = os.path.join(args.out_dir, f"{args.prefix}-includes.txt")
    with open(manifest, "w") as f:
        for fname in written:
            f.write(f"include::{args.prefix}/{fname}[leveloffset=+1]\n")

    print(f"Rendered {len(written)} .adoc files to {args.out_dir}", file=sys.stderr)
    for fname in written:
        print(f"  - {fname}", file=sys.stderr)


if __name__ == "__main__":
    main()
