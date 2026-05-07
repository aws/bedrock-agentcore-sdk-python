"""Lint guard: ensure correct instrument type naming is used.

Payment instruments use 'embeddedCryptoWallet' / 'EMBEDDED_CRYPTO_WALLET'.
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SCAN_DIRS = [
    REPO_ROOT / "src" / "bedrock_agentcore" / "payments",
    REPO_ROOT / "tests" / "bedrock_agentcore" / "payments",
    REPO_ROOT / "tests_integ" / "payments",
]

BANNED_PATTERNS = [
    re.compile(r"\bcryptoWallet\b(?!\.)", re.IGNORECASE),
    re.compile(r"\bCRYPTO_WALLET\b"),
    re.compile(r"\bcrypto_wallet\b"),
]

ALLOWED_PATTERNS = [
    re.compile(r"cryptoX402", re.IGNORECASE),
    re.compile(r"CryptoX402"),
    re.compile(r"CRYPTO_X402"),
    re.compile(r"embeddedCryptoWallet", re.IGNORECASE),
    re.compile(r"EMBEDDED_CRYPTO_WALLET"),
]


def _is_allowed(line: str, match: re.Match) -> bool:
    start = match.start()
    for pattern in ALLOWED_PATTERNS:
        m = pattern.search(line)
        if m and m.start() <= start < m.end():
            return True
    return False


def test_no_deprecated_crypto_wallet_in_source_and_tests():
    """Fail if any payments source or test file uses the old cryptoWallet naming."""
    violations = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            if py_file.name == "test_deprecated_naming.py":
                continue
            for i, line in enumerate(py_file.read_text().splitlines(), 1):
                for pattern in BANNED_PATTERNS:
                    for match in pattern.finditer(line):
                        if not _is_allowed(line, match):
                            rel = py_file.relative_to(REPO_ROOT)
                            violations.append(f"  {rel}:{i}: {line.strip()}")

    assert not violations, (
        "Found 'cryptoWallet' / 'CRYPTO_WALLET'. "
        "Use 'embeddedCryptoWallet' / 'EMBEDDED_CRYPTO_WALLET' instead.\n" + "\n".join(violations)
    )
