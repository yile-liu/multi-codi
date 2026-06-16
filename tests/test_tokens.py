"""Stage-0 checks. Set CODI_BASE to a tokenizer/model path to run; else skip."""

import os

import pytest

from tokens import TRACE_TOKENS, add_trace_tokens, token_ids

BASE = os.environ.get("CODI_BASE")


@pytest.mark.skipif(not BASE, reason="set CODI_BASE to a tokenizer path")
def test_each_token_is_single_id():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    add_trace_tokens(tok)
    ids = token_ids(tok)
    assert len(set(ids.values())) == len(TRACE_TOKENS)
