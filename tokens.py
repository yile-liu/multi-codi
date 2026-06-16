"""CWM trace special tokens + tokenizer/embedding setup for a non-CWM base."""

# Trace-format tokens (mirrors data/trace_format.py) + latent delimiters.
TRACE_TOKENS = [
    "<|trace_context_start|>",
    "<|call_sep|>", "<|line_sep|>", "<|return_sep|>", "<|exception_sep|>",
    "<|action_sep|>", "<|arg_sep|>", "<|frame_sep|>", "<|end_of_text|>",
    "<|latent_start|>", "<|latent_end|>",
]


def add_trace_tokens(tokenizer) -> int:
    """Add the trace tokens as special tokens. Returns the count newly added."""
    return tokenizer.add_tokens(TRACE_TOKENS, special_tokens=True)


def resize_and_init(model, tokenizer, n_added: int) -> None:
    """Resize embeddings to the tokenizer; init new rows to the existing mean."""
    old = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    if n_added <= 0:
        return
    seen = set()
    for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
        if emb is None or id(emb) in seen:  # tied embeddings: resize once
            continue
        seen.add(id(emb))
        w = emb.weight.data
        w[old:] = w[:old].mean(dim=0, keepdim=True)


def token_ids(tokenizer) -> dict[str, int]:
    """Map each trace token to its single id (asserts single-token encoding)."""
    ids = {}
    for t in TRACE_TOKENS:
        enc = tokenizer.encode(t, add_special_tokens=False)
        assert len(enc) == 1, f"{t!r} did not encode to a single id: {enc}"
        ids[t] = enc[0]
    return ids
