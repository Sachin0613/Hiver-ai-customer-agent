# Phase 8 — Historical Retrieval

The retrieval corpus contains only customer/company response pairs from the
Phase 4 retrieval split. The golden pool is checked by conversation ID before
index construction and is never read as retrieval evidence.

Each indexed document combines:

```text
CUSTOMER: <historical customer message>
BRAND: <historical company response>
```

Documents are embedded with `sentence-transformers/all-MiniLM-L6-v2`, normalized,
and stored in a FAISS `IndexFlatIP` index. Inner product on normalized vectors
is cosine similarity. Metadata remains in a CSV with the same row order as the
FAISS vectors.

Build the index with:

```powershell
python -m pip install -r requirements.txt
python scripts/build_index.py
```

Outputs are written to `data/indexes/` and should remain out of Git. The first
run may download the Sentence Transformer model. No intent labels, golden data,
LLM, or FAISS-based evaluation are used in this phase.

On Windows, the project pins a compatible CPU embedding stack and includes
`msvc-runtime`; Torch otherwise fails to load `c10.dll` with WinError 126 when
the Microsoft C++ runtime is absent.