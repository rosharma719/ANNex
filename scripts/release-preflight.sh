#!/usr/bin/env bash
# Pre-flight for a release. Run it on the commit you intend to tag, BEFORE pushing any tag:
# tags publish to crates.io and PyPI, and a published version can never be re-uploaded.
#
#   scripts/release-preflight.sh                 # checks origin/master
#   scripts/release-preflight.sh <commit-ish>    # checks that commit
#   SKIP_DRY_RUN=1 scripts/release-preflight.sh  # skip the (slow) cargo publish dry run
#
# It checks that the commit is on master, every manifest agrees on one version, the changelog
# has a section for it (the release workflow publishes that section as the GitHub release
# notes), nothing at that version is already tagged or published, and CI is green on the
# commit; then it dry-runs `cargo publish` for the core crate from a clean checkout of the
# commit. Finally it prints the tag commands, pinned to the full commit hash.
set -uo pipefail

REPO="rosharma719/ANNex"
fail=0; warn=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; fail=1; }
note() { printf '  \033[33m!\033[0m %s\n' "$*"; warn=1; }
http() { curl -s -m 25 -A "annex-release-preflight" "$@"; }

cd "$(git rev-parse --show-toplevel)" || exit 2
git fetch -q origin master --tags || { echo "cannot reach origin"; exit 2; }
SHA=$(git rev-parse --verify "${1:-origin/master}^{commit}" 2>/dev/null) || { echo "cannot resolve '${1:-origin/master}'"; exit 2; }
echo "Commit under test: $SHA"
git log -1 --format='  %s' "$SHA"

echo; echo "1. Commit"
if git merge-base --is-ancestor "$SHA" origin/master; then ok "reachable from origin/master"; else bad "not on origin/master: tags must point at a merged commit"; fi
[ "$SHA" = "$(git rev-parse origin/master)" ] && ok "is the tip of origin/master" || note "origin/master has moved past this commit"

echo; echo "2. Versions"
version_of() {  # file, table
  git show "$SHA:$1" 2>/dev/null | python3 -c 'import sys,tomllib; print(tomllib.loads(sys.stdin.read())[sys.argv[1]]["version"])' "$2" 2>/dev/null
}
V=$(version_of crates/annex-core/Cargo.toml package)
[ -n "$V" ] || { bad "cannot read the annex version"; exit 1; }
ok "annex = $V"
for spec in "annex-multivector crates/annex-multivector/Cargo.toml package" "annex-server crates/annex-server/Cargo.toml package" "annex-py python/annex-py/Cargo.toml package" "annex-py(pyproject) python/annex-py/pyproject.toml project"; do
  set -- $spec; got=$(version_of "$2" "$3")
  [ "$got" = "$V" ] && ok "$1 = $got" || bad "$1 = ${got:-?}, expected $V (the release workflow rejects a tag that does not match its manifest)"
done

echo; echo "3. Changelog"
lines=$(git show "$SHA:CHANGELOG.md" | awk -v v="$V" 'index($0,"## [" v "]")==1{f=1;next} f&&/^## \[/{exit} f&&NF{n++} END{print n+0}')
[ "${lines:-0}" -gt 0 ] && ok "CHANGELOG.md has a [$V] section ($lines lines): it becomes the GitHub release notes" || bad "no [$V] section in CHANGELOG.md: the release would say \"No changelog entry found\""

echo; echo "4. Already tagged or published?"
tags=$(git ls-remote --tags origin "refs/tags/*-v$V" | wc -l)
[ "$tags" -eq 0 ] && ok "no *-v$V tags on origin" || bad "$tags tag(s) already exist for $V"
for c in annex annex-multivector annex-server; do
  vers=$(http "https://crates.io/api/v1/crates/$c" | python3 -c 'import sys,json; print(" ".join(v["num"] for v in json.load(sys.stdin)["versions"]))' 2>/dev/null)
  [ -n "$vers" ] || { note "crates.io/$c: could not read (skipped)"; continue; }
  case " $vers " in *" $V "*) bad "crates.io already has $c $V";; *) ok "crates.io/$c has no $V (has: $vers)";; esac
done
pyvers=$(http "https://pypi.org/pypi/ANNexDB/json" | python3 -c 'import sys,json; print(" ".join(json.load(sys.stdin)["releases"]))' 2>/dev/null)
if [ -n "$pyvers" ]; then case " $pyvers " in *" $V "*) bad "PyPI already has ANNexDB $V";; *) ok "PyPI/ANNexDB has no $V (has: $pyvers)";; esac; else note "PyPI: could not read (skipped)"; fi

echo; echo "5. CI on this commit"
ci=$(http "https://api.github.com/repos/$REPO/commits/$SHA/check-runs?per_page=100" | python3 -c '
import sys, json
try: runs = json.load(sys.stdin)["check_runs"]
except Exception: print("unreadable"); raise SystemExit
bad = [r["name"] + " " + str(r["conclusion"] or r["status"]) for r in runs if r["status"] != "completed" or r["conclusion"] not in ("success", "skipped", "neutral")]
print("none" if not runs else ("fail:" + "; ".join(bad) if bad else "ok:%d" % len(runs)))' 2>/dev/null)
case "$ci" in
  ok:*)  ok "all ${ci#ok:} check runs succeeded";;
  none)  bad "no check runs found for this commit (not run yet?)";;
  fail:*) bad "checks not green: ${ci#fail:}";;
  *)     note "could not read CI status (rate limited?): check the commit on GitHub";;
esac

echo; echo "6. cargo publish --dry-run (annex), from a clean checkout of the commit"
if [ "${SKIP_DRY_RUN:-0}" = 1 ]; then note "skipped (SKIP_DRY_RUN=1)"; else
  tmp=$(mktemp -d); git worktree add -q --detach "$tmp" "$SHA" 2>/dev/null
  out=$(cd "$tmp" && CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/../target-preflight}" cargo publish --dry-run --locked -p annex 2>&1); st=$?
  git worktree remove --force "$tmp" 2>/dev/null
  [ $st -eq 0 ] && ok "$(printf '%s\n' "$out" | grep -E 'Packaged' | head -1 | sed 's/^ *//')" || { bad "dry run failed"; printf '%s\n' "$out" | tail -8 | sed 's/^/      /'; }
fi

echo
if [ $fail -ne 0 ]; then echo "NOT READY: fix the ✗ items above. Do not tag."; exit 1; fi
[ $warn -ne 0 ] && echo "Ready, with the ! warnings above to check by hand."
[ $warn -eq 0 ] && echo "Ready."
cat <<EOT

Tag in this order. The three dependants need annex on crates.io first (about 5 minutes).

  git fetch origin
  git tag -a annex-v$V -m "annex $V" $SHA
  git push origin annex-v$V
  # wait for the Release workflow to go green and annex $V to appear on crates.io, then:
  for c in annex-multivector annex-server annex-py; do git tag -a \$c-v$V -m "\$c $V" $SHA; done
  git push origin annex-multivector-v$V annex-server-v$V annex-py-v$V

Then run: scripts/release-verify.sh $V
EOT
