from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

# The embedding provider (Voyage AI) has no local/offline tokenizer, so max_tokens/
# overlap_tokens budgeting during chunking uses a local approximate proxy instead -- this is
# docling_core's HybridChunker's own default tokenizer. Chunk.token_count is documented as an
# approximation, not a Voyage-exact count; calling a remote tokenization API per
# chunk-boundary decision would be far too slow/costly for chunking (many calls per doc).
PROXY_TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_cache: dict[int, HuggingFaceTokenizer] = {}


def build_tokenizer(max_tokens: int) -> HuggingFaceTokenizer:
    if max_tokens not in _cache:
        _cache[max_tokens] = HuggingFaceTokenizer.from_pretrained(
            PROXY_TOKENIZER_MODEL, max_tokens=max_tokens
        )
    return _cache[max_tokens]


def token_windows(
    text: str, max_tokens: int, overlap_tokens: int, tokenizer: HuggingFaceTokenizer
) -> list[str]:
    """Splits text into token-budgeted windows using real offset-mapping from the fast
    tokenizer backing `tokenizer`, so windows are exact slices of the ORIGINAL string --
    never encode-then-decode, which would mangle text (BERT's WordPiece decode lowercases
    and destroys original spacing/punctuation).
    """
    if overlap_tokens >= max_tokens:
        raise ValueError(f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})")

    hf = tokenizer.get_tokenizer()
    encoding = hf(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    n = len(offsets)
    if n == 0:
        return []

    step = max_tokens - overlap_tokens
    windows: list[str] = []
    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        windows.append(text[char_start:char_end])
        if end == n:
            break
        start += step
    return windows
