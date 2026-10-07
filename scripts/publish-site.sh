#!/usr/bin/env bash
# Mirror only site/ into the public Pages repository. Preserve its Git history.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
destination=$(mktemp -d "${TMPDIR:-/tmp}/soul-pages.XXXXXX")
trap 'rm -rf "$destination"' EXIT

test -f "$root/site/index.html"
git clone --quiet --depth 1 git@github.com:mastrogiovanni/soul-community.git "$destination"
git -C "$destination" checkout -B main
rsync -a --delete --exclude='.git' "$root/site/" "$destination/"
git -C "$destination" add --all
if git -C "$destination" diff --cached --quiet; then
    printf 'The public website is already up to date.\n'
    exit 0
fi

source_commit=$(git -C "$root" rev-parse --short HEAD)
git -C "$destination" -c user.name='Soul website publisher' \
    -c user.email='website@users.noreply.github.com' \
    commit --quiet -m "Publish Soul community website ($source_commit)"
git -C "$destination" push origin main
printf 'Published: https://mastrogiovanni.github.io/soul-community/\n'
