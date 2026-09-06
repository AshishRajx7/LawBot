#!/usr/bin/env bash
# render_build.sh - Production build script for LawBot on Render
set -e

echo "=== [1/4] Upgrading pip ==="
pip install --upgrade pip

echo "=== [2/4] Installing production dependencies ==="
pip install -r requirements.txt

echo "=== [3/4] Pre-caching FlashRank cross-encoder model ==="
python -c "
try:
    from flashrank import Ranker
    print('Downloading & caching ms-marco-TinyBERT-L-2-v2...')
    Ranker(model_name='ms-marco-TinyBERT-L-2-v2', max_length=256)
    print('FlashRank model successfully cached.')
except Exception as e:
    print(f'Notice: FlashRank pre-cache notice (will initialize on demand): {e}')
"

echo "=== [4/4] Verifying production database assets ==="
python -c "
import os
assert os.path.exists('data/bm25_index.json'), 'Missing data/bm25_index.json index file!'
assert os.path.exists('data/chroma_db'), 'Missing data/chroma_db persistent vector directory!'
assert os.path.exists('data/legal_knowledge_graph.json'), 'Missing data/legal_knowledge_graph.json file!'
print('All core data indexes verified successfully.')
"

echo "=== Build Completed Successfully ==="
