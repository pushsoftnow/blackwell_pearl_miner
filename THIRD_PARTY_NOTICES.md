# Third-party notices

This repository's own code is licensed under the ISC license (see LICENSE). It builds on the
following components, which keep their own licenses.

## Pearl (git submodule `third_party/pearl`)

https://github.com/pearl-research-labs/pearl, pinned commit 82e6eee7076aafe10123e08db78f100d270c0582.
Used: the `zk-pow` verifier and `pearl-blake3` crates (Rust dependencies) and CUDA headers from
`miner/pearl-gemm/csrc` (BLAKE3, noise generation, commitment hash, proof-of-work utilities) that the
kernels include. Its `plonky2` dependency is licensed MIT OR Apache-2.0.

```
ISC License

Copyright (c) 2025-2026 Pearl Research Labs
Copyright (c) 2015-2016 The Decred developers

Permission to use, copy, modify, and distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

---

This monorepo contains multiple sub-projects, each with its own LICENSE
file. See the LICENSE in each sub-directory for details:

  node/             ISC                (Pearl Research Labs)
  wallet/           ISC                (Pearl Research Labs)
  spv/              MIT                (Lightning Labs)
  plonky2/          MIT OR Apache-2.0  (The Plonky2 Authors, Pearl Research Labs)
  xmss/external/    CC0-1.0
  miner/pearl-gemm/third_party/cutlass/  BSD-3-Clause (NVIDIA)
```

## CUTLASS (git submodule of Pearl, `miner/pearl-gemm/third_party/cutlass`)

https://github.com/NVIDIA/cutlass, header-only, used by the Pearl CUDA headers.

```
Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Certain files within this repository are subject to separate licensing terms:

- The files located in the `python/CuTeDSL` directory are licensed under the
  NVIDIA End User License Agreement (EULA). Please refer to
  https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
  for the full terms.
```

## Rust crates

Fetched by cargo (see Cargo.lock) under their own licenses: MIT, Apache-2.0, BSD-2/3-Clause,
ISC, Zlib, CC0-1.0, Unlicense, Unicode-3.0 (in dual- or multi-licensed form). None is copyleft-only.

## NVIDIA CUDA

Building requires the NVIDIA CUDA Toolkit. The kernel library embedded in the compiled binary
statically links the CUDA runtime (`libcudart_static`), which NVIDIA licenses as a redistributable
component under the CUDA Toolkit End User License Agreement
(https://docs.nvidia.com/cuda/eula/). Distributors of binaries must comply with that agreement.
