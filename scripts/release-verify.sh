#!/usr/bin/env bash
# Post-release verification. Run it after pushing the tags (the registries and wheel builds can
# take a few minutes to catch up, so re-run it until it is green):
#
#   scripts/release-verify.sh 0.3.0
#
# Checks that every package is on its registry, that each GitHub release exists and carries
# real notes, and that a clean `pip install ANNexDB==<version>` works.
set -uo pipefail

V="${1:?usage: scripts/release-verify.sh <version>   e.g. 0.3.0}"
REPO="rosharma719/ANNex"
fail=0; warn=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; fail=1; }
note() { printf '  \033[33m!\033[0m %s\n' "$*"; warn=1; }
http() { curl -s -m 25 -A "annex-release-verify" "$@"; }

echo "Verifying release $V"

echo; echo "1. crates.io"
for c in annex annex-multivector annex-server; do
  state=$(http "https://crates.io/api/v1/crates/$c" | python3 -c '
import sys, json
try: vs = {v["num"]: v for v in json.load(sys.stdin)["versions"]}
except Exception: print("unreadable"); raise SystemExit
v = vs.get(sys.argv[1])
print("missing" if v is None else ("yanked" if v.get("yanked") else "ok"))' "$V" 2>/dev/null)
  case "$state" in ok) ok "$c $V is published";; missing) bad "$c $V is not on crates.io yet";; yanked) bad "$c $V is yanked";; *) note "$c: could not read crates.io";; esac
done

echo; echo "2. PyPI (ANNexDB)"
files=$(http "https://pypi.org/pypi/ANNexDB/$V/json" | python3 -c '
import sys, json
try: d = json.load(sys.stdin)
except Exception: raise SystemExit
for f in d.get("urls", []): print(f["filename"])' 2>/dev/null)
if [ -z "$files" ]; then bad "ANNexDB $V is not on PyPI yet"; else
  n=$(printf '%s\n' "$files" | grep -c '\.whl$' || true)
  ok "ANNexDB $V is published ($n wheel(s))"
  printf '%s\n' "$files" | sed 's/^/      /'
  [ "${n:-0}" -ge 4 ] || note "expected 4 wheels (linux x86_64 and aarch64, macOS universal2, windows x86_64); the build may still be finishing"
fi

echo; echo "3. GitHub releases"
for t in "annex-v$V" "annex-multivector-v$V" "annex-server-v$V" "annex-py-v$V"; do
  res=$(http "https://api.github.com/repos/$REPO/releases/tags/$t" | python3 -c '
import sys, json
try: d = json.load(sys.stdin)
except Exception: print("unreadable"); raise SystemExit
if d.get("message") == "Not Found": print("missing"); raise SystemExit
body = (d.get("body") or "").strip()
if not body or "No changelog entry found" in body: print("empty")
elif d.get("draft"): print("draft")
else: print("ok:%d" % len(body.splitlines()))' 2>/dev/null)
  case "$res" in
    ok:*)  ok "$t: published, notes are ${res#ok:} lines";;
    missing) bad "$t: no GitHub release (the Release workflow may still be running or failed)";;
    empty) bad "$t: release has no real notes (the changelog section is missing?)";;
    draft) bad "$t: release is still a draft";;
    *)     note "$t: could not read GitHub";;
  esac
done

echo; echo "4. Clean install and smoke test of ANNexDB==$V"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
if python3 -m venv "$tmp/venv" >/dev/null 2>&1 && "$tmp/venv/bin/pip" install -q "ANNexDB==$V" numpy >"$tmp/pip.log" 2>&1; then
  out=$("$tmp/venv/bin/python" - "$V" <<'PYEOF' 2>&1
import sys, os, tempfile, importlib.metadata as md
import numpy as np, annexdb
want = sys.argv[1]
got = md.version("ANNexDB")
assert got == want, f"installed {got}, expected {want}"
if not hasattr(annexdb.Index, "build"):
    print(f"installed {got}; this version has no Index.build (load/search only)")
else:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 8)).astype(np.float32)
    idx = annexdb.Index.build(x, metric="cosine")
    ids, _ = idx.search(x[7], k=1, ef=200)
    assert int(ids[0]) == 7, f"self-search returned {ids[0]}"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "i.bin"); idx.save(p)
        ids2, _ = annexdb.Index(p).search(x[7], k=1, ef=200)
        assert int(ids2[0]) == 7, "reloaded index disagrees"
    print(f"installed {got}; build, search, save and reload work")
PYEOF
  ) && ok "$out" || { bad "smoke test failed"; printf '%s\n' "$out" | tail -5 | sed 's/^/      /'; }
else
  bad "pip install ANNexDB==$V failed (not on PyPI yet, or no wheel for this platform)"; grep -E "ERROR|No matching" "$tmp/pip.log" 2>/dev/null | head -2 | sed 's/^/      /'
fi

echo
if [ $fail -ne 0 ]; then echo "NOT COMPLETE: re-run in a few minutes; if an item stays red, check the Release workflow run."; exit 1; fi
[ $warn -ne 0 ] && echo "Complete, with the ! warnings above."
[ $warn -eq 0 ] && echo "Release $V verified."
cat <<EOT

Remaining by hand:
  - The site's /changelog and version only update on its next build (see docs/releasing.md).
  - Update the site quickstart to match this release's features.
EOT
