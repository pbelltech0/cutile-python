# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct
import torch
from cuda.tile._ir.op_impl import impl


def raise_error(*args): ...


@impl(raise_error)
def raise_error_impl(args):
    msg = " ".join(str(x.get_constant()) for x in args)
    raise AssertionError(msg)


@ct.kernel
def array_attr_kernel(X, out):
    ndim = X.ndim
    shape = X.shape
    strides = X.strides
    if ndim != 3:
        raise_error("Expect ndim 3, got", ndim)
    if len(shape) != ndim:
        raise_error("Expect shape len 3, got", len(shape))
    if len(strides) != ndim:
        raise_error("Expect stride len 3, got", len(strides))

    ct.store(out, (0,), shape[0])
    ct.store(out, (1,), shape[1])
    ct.store(out, (2,), shape[2])
    ct.store(out, (3,), strides[0])
    ct.store(out, (4,), strides[1])
    ct.store(out, (5,), strides[2])


def test_array_attr():
    x = torch.zeros((2, 3, 4), device='cuda')
    out = torch.zeros(6, device='cuda', dtype=torch.int64)
    ct.launch(torch.cuda.current_stream(),
              (1,),
              array_attr_kernel, (x, out))
    assert list(out[0:3]) == list(x.shape)
    assert list(out[3:6]) == list(x.stride())
