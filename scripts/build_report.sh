#!/usr/bin/env bash
# Print report/report.html to a tagged, text-selectable PDF with headless Chrome.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-report/lightheart-jonny-vector-retrieval.pdf}"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --generate-pdf-document-outline \
  --export-tagged-pdf --print-to-pdf="$OUT" "file://$PWD/report/report.html" 2>/dev/null
echo "wrote $OUT"
