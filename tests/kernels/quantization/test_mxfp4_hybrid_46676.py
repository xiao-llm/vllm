# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEST ONLY: hybrid W4A8 GEMV (M <= 8) + #46676 WMMA (M > 8) vs the fallback."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm()
    or not hasattr(torch.ops._rocm_C, "mxfp4_gemm_rdna3"),
    reason="needs the #46676 op built for this GPU",
)


@pytest.mark.parametrize("M", [9, 16, 64, 512])
@pytest.mark.parametrize("N,K", [(256, 512), (5120, 5120), (17408, 5120)])
def test_hybrid_matches_fallback(M, N, K):
    from vllm.model_executor.kernels.linear.mxfp4.rdna_w4a8 import (
        _rdna_mxfp4_hybrid_apply_impl,
        _rdna_mxfp4_w4a8_apply_impl,
    )

    torch.manual_seed(0)
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(118, 128, (N, K // 32), dtype=torch.uint8, device="cuda")
    w46 = w.view(torch.int32).t().contiguous()
    s46 = s.t().contiguous()
    x = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)

    ref = _rdna_mxfp4_w4a8_apply_impl(x, w, s, None).float()
    out = _rdna_mxfp4_hybrid_apply_impl(x, w, s, w46, s46, None).float()
    rel = (out - ref).norm() / ref.norm()
    assert rel < 1e-2, f"relative error {rel.item():.3e}"
