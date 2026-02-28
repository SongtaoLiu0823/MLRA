# Copyright (c) 2025, Ted Zadouri, Tri Dao.

# We recommend locking GPU clocks before running the benchmark to ensure consistent results.
# This can be done using the following commands (1830 MHz is the clock for H100):
# sudo nvidia-smi -i 0 -pm 1
# sudo nvidia-smi -i 0 --lock-gpu-clocks 1830,1830
# See more here: https://github.com/triton-lang/triton/blob/d9f10ebdc5da53f73eb852fde73d8d7d80b679d1/python/triton/testing.py#L487

import time
import torch

from triton.testing import do_bench, do_bench_cudagraph
from einops import rearrange
from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata
from flash_mla import flash_mla_with_kvcache, get_mla_metadata


device = "cuda"
dtype = torch.bfloat16
use_bench_cudagraph = False

attn_variants = ["gqa", "mla", "gla", "mlra"]
seqlen_q = 1
nheads_kv = 1
batch_size = 1

# GQA (TP-8), MLA (DP), GLA (TP-2), MLRA (TP-4)
for attn_variant in attn_variants:    
    if attn_variant == "mla" or attn_variant == "mlra":
        nheads_q = 64
    elif attn_variant == "gla":
        nheads_q = 32
    elif attn_variant == "gqa":
        nheads_q = 8

    headdim = 64 if attn_variant in ["mla", "gla", "mlra"] else 128

    if attn_variant == "mla":
        headdim_v = 512
    elif attn_variant == "gla":
        headdim_v = 256
    elif attn_variant == "mlra":
        headdim_v = 128
    else:
        headdim_v = headdim
    
    has_qv = True if attn_variant in ["mla", "gla", "mlra"] else False


    # page_size = None
    page_size = 64 if attn_variant in ["mla", "gla", "mlra"] else 128

    should_run_flashmla = attn_variant == "mla" and page_size == 64

    torch.manual_seed(0)

    cache_seqlens = None

    print(f"\n{attn_variant.upper()}, nheads_q = {nheads_q}, nheads_kv = {nheads_kv}, headdim = {headdim}, headdim_v = {headdim_v}, page_size = {page_size}")

    for seqlen in [s * 1024 for s in [2048, 1024, 512, 256, 128]]:
        cache_seqlens = torch.tensor([seqlen] * batch_size, device=device, dtype=torch.int)
        num_splits = 0
        q = torch.randn(batch_size, seqlen_q, nheads_q, headdim, dtype=dtype, device=device)
        try:
            k_cache = torch.randn(batch_size, seqlen, nheads_kv, headdim, dtype=dtype, device=device)
            v_cache = torch.randn(batch_size, seqlen, nheads_kv, headdim_v, dtype=dtype, device=device)
            if page_size is not None:
                assert seqlen % page_size == 0
                k_cache, v_cache = [rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) for x in [k_cache, v_cache]]
                page_table = rearrange(torch.arange(batch_size * seqlen // page_size, device=device, dtype=torch.int32),
                                    "(b s) -> b s", s=seqlen // page_size)
            else:
                page_table = None
        except torch.OutOfMemoryError:
            continue
        qv = torch.randn(batch_size, seqlen_q, nheads_q, headdim_v, dtype=dtype, device=device) if has_qv else None

        if not should_run_flashmla:
            # Precomputing this saves ~2us
            scheduler_metadata = get_scheduler_metadata(
                batch_size, seqlen_q, seqlen, nheads_q, nheads_kv, headdim,
                cache_seqlens, q.dtype, headdim_v=headdim_v, page_size=page_size, causal=True
            )
            fn0 = lambda: flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens, num_splits=num_splits, qv=qv, page_table=page_table, causal=True, scheduler_metadata=scheduler_metadata)
            time.sleep(1)  # to avoid power throttling
            # Time in ms
            if not use_bench_cudagraph:
                t0 = do_bench(fn0, warmup=1, rep=10)
            else:
                torch.cuda.synchronize()  # Gotta wait, otherwise e.g. k_cache might not be ready
                with torch.cuda.stream(torch.cuda.Stream()):
                    t0 = do_bench_cudagraph(fn0, rep=10)
        else:
            # Separate out the preprocessing since this can be done once and reused for all layers
            mla_metadata = get_mla_metadata(cache_seqlens, seqlen_q * nheads_q // nheads_kv, nheads_kv)
            q_concat = torch.concat([q, qv], dim=-1) if has_qv else q
            kv_cache_concat = torch.concat([v_cache, k_cache], dim=-1)
            fn1 = lambda: flash_mla_with_kvcache(q_concat, kv_cache_concat, page_table, cache_seqlens, headdim_v, *mla_metadata, causal=True)
            time.sleep(1)  # to avoid power throttling
            if not use_bench_cudagraph:
                t1 = do_bench(fn1, warmup=1, rep=10)
            else:
                torch.cuda.synchronize()  # Gotta wait, otherwise e.g. k_cache might not be ready
                with torch.cuda.stream(torch.cuda.Stream()):
                    t1 = do_bench_cudagraph(fn1, rep=10)

        total_seqlen = seqlen * batch_size if cache_seqlens is None else cache_seqlens.sum().item()
        mem_io = total_seqlen * nheads_kv * (headdim + headdim_v) * 2 + q.numel() * 2 + (qv.numel() * 2 if has_qv else 0) + q.numel() * headdim_v // headdim * 2  # last term is for the output
        ideal_kv_load_time = mem_io / 3.35e12 * 1e6
        if should_run_flashmla:
            print(f"Seqlen = {seqlen}, FlashMLA time{'' if not use_bench_cudagraph else ' w CUDA Graph'}: {t1 * 1e3:.1f} us")
        else:
            print(f"Seqlen = {seqlen}, FA3 time{'' if not use_bench_cudagraph else ' w CUDA Graph'}: {t0 * 1e3:.1f} us")
        print(f"Ideal KV load time: {ideal_kv_load_time:.0f} us")
