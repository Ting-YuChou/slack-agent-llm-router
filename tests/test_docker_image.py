from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILES = [
    ROOT / "docker/Dockerfile",
    ROOT / "docker/flink/Dockerfile",
    *sorted((ROOT / "agent-runtime").glob("Dockerfile.*")),
]


def _context_sources(dockerfile):
    # Sources of COPY/ADD that read the build context (not another stage or image).
    sources = []
    for line in dockerfile.read_text().replace("\\\n", " ").splitlines():
        parts = line.split()
        if not parts or parts[0] not in ("COPY", "ADD"):
            continue
        if any(part.startswith("--from=") for part in parts):
            continue
        args = [part for part in parts[1:] if not part.startswith("--")]
        sources.extend(args[:-1])
    return sources


def _dockerignore_patterns():
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _pattern_regex(pattern):
    regex = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            regex, i = regex + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            regex, i = regex + ".*", i + 2
        elif pattern[i] == "*":
            regex, i = regex + "[^/]*", i + 1
        elif pattern[i] == "?":
            regex, i = regex + "[^/]", i + 1
        else:
            regex, i = regex + re.escape(pattern[i]), i + 1
    # A pattern that names a directory excludes everything under it.
    return re.compile(f"^{regex}(?:/.*)?$")


def _ignored(path):
    # Docker semantics: the last matching pattern wins, `!` re-includes.
    ignored = False
    for pattern in _dockerignore_patterns():
        negated = pattern.startswith("!")
        if _pattern_regex(pattern.lstrip("!").lstrip("/")).match(path):
            ignored = not negated
    return ignored


def test_no_dockerfile_copies_the_whole_build_context():
    for dockerfile in DOCKERFILES:
        for source in _context_sources(dockerfile):
            assert source not in (".", "./", "*"), f"{dockerfile.name} copies {source}"


def test_app_image_copies_only_what_the_app_runs():
    sources = _context_sources(ROOT / "docker/Dockerfile")

    assert sorted(sources) == [
        "config",
        "docker/requirements-runtime.txt",
        "main.py",
        "slack",
        "src",
    ]


def test_dockerignore_keeps_local_secrets_out_of_build_contexts():
    for path in [
        ".env",
        ".env.demo",
        ".env.demo.bak",
        ".env.example.local",
        "config/local.env",
        "src/.env",
        "slack/credentials/bot-token.json",
        "certs/server.pem",
        "certs/server.key",
        ".claude/worktrees/feature/.env",
        ".pi-agent-runtime/runtime-state.json",
        ".pi-agent-worktrees/run/.env.local",
        "agent-runtime/node_modules/undici/package.json",
    ]:
        assert _ignored(path), path


def test_dockerignore_keeps_what_the_images_copy():
    for path in [
        "main.py",
        "src/llm_router_part1_router.py",
        "slack/bot_real.py",
        "config/config.compose.yaml",
        "config/demo.env.example",
        "docker/requirements-runtime.txt",
        "agent-runtime/src/server.ts",
        "agent-runtime/requirements-ci.lock",
    ]:
        assert not _ignored(path), path


def _final_stage(dockerfile: Path):
    text = dockerfile.read_text().replace("\\\n", " ")
    stages = re.split(r"(?m)^FROM ", text)
    return stages[-1].splitlines()


def test_app_image_prefetches_tiktoken_encodings_for_offline_use():
    lines = _final_stage(ROOT / "docker/Dockerfile")
    env_index = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("ENV ") and "TIKTOKEN_CACHE_DIR=" in line
    )
    cache_dir = re.search(r"TIKTOKEN_CACHE_DIR=(\S+)", lines[env_index]).group(1)
    fetch_index = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("RUN ")
        and "cl100k_base" in line
        and "o200k_base" in line
        and "get_encoding" in line
    )
    user_index = next(i for i, line in enumerate(lines) if line.startswith("USER "))

    assert cache_dir.startswith("/")
    assert env_index < fetch_index < user_index
    fetch = lines[fetch_index]
    # The runtime user must be able to read the files root fetched.
    assert cache_dir in fetch and ("a+rX" in fetch or "chown" in fetch)
