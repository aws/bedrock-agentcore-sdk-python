import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    content = Path("pyproject.toml").read_text()
    pattern = r'(?:^\[project\]|\[tool\.poetry\])[\s\S]*?^version\s*=\s*"([^"]+)"'
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        raise ValueError("Version not found in pyproject.toml under [project] or [tool.poetry]")
    return match.group(1)


def parse_version(version: str) -> Tuple[int, int, int, Optional[str]]:
    """Parse semantic version string."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)(?:-(.+))?", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")

    major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
    pre_release = match.group(4)
    return major, minor, patch, pre_release


def bump_version(current: str, bump_type: str) -> str:
    """Bump version based on type."""
    major, minor, patch, pre_release = parse_version(current)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif bump_type == "pre":
        if pre_release:
            match = re.match(r"(.+?)(\d+)$", pre_release)
            if match:
                prefix, num = match.groups()
                return f"{major}.{minor}.{patch}-{prefix}{int(num) + 1}"
        return f"{major}.{minor}.{patch + 1}-rc1"
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")


def update_version_in_file(file_path: Path, old_version: str, new_version: str) -> bool:
    """Update version in a file."""
    if not file_path.exists():
        return False

    content = file_path.read_text()

    # Fix: Use re.sub with a function to avoid group reference issues
    pattern = rf'^(__version__\s*=\s*["\'])({re.escape(old_version)})(["\'])'

    def replacer(match):
        return f"{match.group(1)}{new_version}{match.group(3)}"

    new_content = re.sub(pattern, replacer, content, flags=re.MULTILINE)

    if new_content != content:
        file_path.write_text(new_content)
        return True
    return False


def update_all_versions(old_version: str, new_version: str):
    """Update version in all relevant files."""
    # Update pyproject.toml - use simple string replacement to avoid regex issues
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text()

    # Simple string replacement instead of regex
    old_version_line = f'version = "{old_version}"'
    new_version_line = f'version = "{new_version}"'

    if old_version_line in content:
        content = content.replace(old_version_line, new_version_line, 1)
        pyproject.write_text(content)
        print("✓ Updated pyproject.toml")
    else:
        raise ValueError(f'Could not find version = "{old_version}" in pyproject.toml')

    # Update __init__.py files that contain version
    init_file = Path("src/bedrock_agentcore/__init__.py")
    if init_file.exists() and update_version_in_file(init_file, old_version, new_version):
        print(f"✓ Updated {init_file}")


def format_git_log(git_log: str) -> str:
    """Format git log entries for changelog."""
    if not git_log.strip():
        return ""

    fixes = []
    features = []
    docs = []
    other = []

    for line in git_log.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("-"):
            continue

        commit_msg = line[2:].strip()

        if commit_msg.startswith("fix:") or commit_msg.startswith("bugfix:"):
            fixes.append(commit_msg)
        elif commit_msg.startswith("feat:") or commit_msg.startswith("feature:"):
            features.append(commit_msg)
        elif commit_msg.startswith("docs:") or commit_msg.startswith("doc:"):
            docs.append(commit_msg)
        else:
            other.append(commit_msg)

    sections = []

    if features:
        sections.append("### Added\n" + "\n".join(f"- {msg}" for msg in features))

    if fixes:
        sections.append("### Fixed\n" + "\n".join(f"- {msg}" for msg in fixes))

    if docs:
        sections.append("### Documentation\n" + "\n".join(f"- {msg}" for msg in docs))

    if other:
        sections.append("### Other Changes\n" + "\n".join(f"- {msg}" for msg in other))

    return "\n\n".join(sections)


def get_git_log(since_tag: Optional[str] = None) -> str:
    """Get git commit messages since last tag."""
    cmd = ["git", "log", "--pretty=format:- %s (%h)"]
    if since_tag:
        cmd.append(f"{since_tag}..HEAD")
    else:
        try:
            last_tag = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True, check=True
            ).stdout.strip()
            cmd.append(f"{last_tag}..HEAD")
        except subprocess.CalledProcessError:
            cmd.extend(["-n", "20"])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def update_changelog(new_version: str, changes: str = None):
    """Update CHANGELOG.md with new version."""
    changelog_path = Path("CHANGELOG.md")

    if not changelog_path.exists():
        content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
    else:
        content = changelog_path.read_text()

    date = datetime.now().strftime("%Y-%m-%d")
    entry = f"\n## [{new_version}] - {date}\n\n"

    if changes:
        entry += "### Changes\n\n"
        entry += changes + "\n"
    else:
        print("\n⚠️  No changelog provided. Auto-generating from commits.")
        print("💡 Tip: Use --changelog to provide meaningful release notes")

        git_log = get_git_log()
        if git_log:
            formatted_log = format_git_log(git_log)
            if formatted_log:
                entry += formatted_log + "\n"
            else:
                entry += "### Changes\n\n"
                entry += git_log + "\n"

    # Insert after header
    if "# Changelog" in content:
        parts = content.split("\n", 2)
        content = parts[0] + "\n" + entry + "\n" + (parts[2] if len(parts) > 2 else "")
    else:
        content = "# Changelog\n" + entry + "\n" + content

    changelog_path.write_text(content)
    print("✓ Updated CHANGELOG.md")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bump SDK version")
    parser.add_argument("bump_type", choices=["major", "minor", "patch", "pre"], help="Type of version bump")
    parser.add_argument("--changelog", help="Custom changelog entry")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")

    args = parser.parse_args()

    try:
        current = get_current_version()
        new = bump_version(current, args.bump_type)

        print(f"Current version: {current}")
        print(f"New version: {new}")

        if args.dry_run:
            print("\nDry run - no changes made")
            return

        update_all_versions(current, new)
        update_changelog(new, args.changelog)

        print(f"\n✓ Version bumped from {current} to {new}")
        print("\nNext steps:")
        print("1. Review changes: git diff")
        print("2. Commit: git add -A && git commit -m 'chore: bump version to {}'".format(new))
        print("3. Create PR or push to trigger release workflow")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
