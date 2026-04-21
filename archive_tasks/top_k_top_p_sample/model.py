import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, eps: float = 1e-8, top_k_guess: int = 32) -> None:
        super().__init__()
        self.eps = float(eps)
        self.top_k_guess = int(top_k_guess)

    def forward(self, logits, top_ks, top_ps, q, is_need_logits=False):
        logits_2d = logits.reshape(-1, logits.shape[-1]).to(torch.float32)
        top_ks_1d = top_ks.reshape(-1).to(torch.int64)
        top_ps_1d = top_ps.reshape(-1).to(torch.float32)
        q_2d = q.reshape(-1, logits.shape[-1]).to(torch.float32)

        batch = logits_2d.shape[0]
        vocab = logits_2d.shape[1]
        output_idx = []
        output_logits = []

        for row in range(batch):
            row_logits = logits_2d[row]
            row_top_k = int(top_ks_1d[row].item())
            row_top_p = float(top_ps_1d[row].item())
            row_q = q_2d[row]

            sorted_logits, sorted_indices = torch.sort(row_logits, dim=-1, descending=True, stable=True)
            top_k = max(1, min(row_top_k, vocab, 1024))
            logits_top_k = sorted_logits[:top_k]
            indices_top_k = sorted_indices[:top_k]

            probs_top_k = torch.softmax(logits_top_k, dim=-1)
            if 0.0 < row_top_p < 1.0:
                cumulative = probs_top_k.cumsum(dim=-1)
                top_p_hits = torch.nonzero(cumulative > row_top_p, as_tuple=False)
                if top_p_hits.numel() > 0:
                    top_p_num = int(top_p_hits[0].item()) + 1
                else:
                    top_p_num = top_k
            else:
                top_p_num = top_k

            kept_logits = logits_top_k[:top_p_num]
            kept_indices = indices_top_k[:top_p_num]
            kept_probs = torch.softmax(kept_logits, dim=-1)
            q_prefix = row_q[:top_p_num].abs().add(self.eps)
            sample_scores = kept_probs / q_prefix
            sample_index = int(sample_scores.argmax(dim=-1).item())

            selected_idx = kept_indices[sample_index].to(torch.int64)
            selected_logits = torch.full(
                (vocab,),
                -float("inf"),
                dtype=torch.float32,
                device=logits.device,
            )
            if is_need_logits:
                selected_logits.scatter_(0, kept_indices.to(torch.int64), kept_logits)

            output_idx.append(selected_idx)
            output_logits.append(selected_logits)

        logits_select_idx = torch.stack(output_idx, dim=0).reshape(top_ks.shape)
        logits_top_kp_select = torch.stack(output_logits, dim=0).reshape(logits.shape)
        return logits_select_idx, logits_top_kp_select


TOP_K_TOP_P_SAMPLE_CASES = [
    {"shape": [8, 64], "dtype": torch.float16, "is_need_logits": False, "seed": 2026},
    {"shape": [8, 64], "dtype": torch.bfloat16, "is_need_logits": False, "seed": 2027},
    {"shape": [4, 255], "dtype": torch.float16, "is_need_logits": False, "seed": 2030},
    {"shape": [4, 255], "dtype": torch.bfloat16, "is_need_logits": False, "seed": 2031},
    {"shape": [1, 258], "dtype": torch.float16, "is_need_logits": False, "seed": 2032},
    {"shape": [1, 258], "dtype": torch.bfloat16, "is_need_logits": False, "seed": 2033},
    {"shape": [1, 272], "dtype": torch.float16, "is_need_logits": False, "seed": 2034},
    {"shape": [1, 272], "dtype": torch.bfloat16, "is_need_logits": False, "seed": 2035},
    {"shape": [1, 288], "dtype": torch.float16, "is_need_logits": False, "seed": 2036},
    {"shape": [1, 288], "dtype": torch.bfloat16, "is_need_logits": False, "seed": 2037},
    {"shape": [8, 64], "dtype": torch.float16, "is_need_logits": True, "seed": 2100},
    {"shape": [8, 64], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2101},
    {"shape": [4, 255], "dtype": torch.float16, "is_need_logits": True, "seed": 2102},
    {"shape": [4, 255], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2103},
    {"shape": [2, 384], "dtype": torch.float16, "is_need_logits": True, "seed": 2040},
    {"shape": [2, 384], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2041},
    {"shape": [1, 513], "dtype": torch.float16, "is_need_logits": True, "seed": 2042},
    {"shape": [1, 513], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2043},
    {"shape": [1, 768], "dtype": torch.float16, "is_need_logits": True, "seed": 2044},
    {"shape": [1, 768], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2045},
    {"shape": [1, 1024], "dtype": torch.float16, "is_need_logits": False, "seed": 2050},
    {"shape": [1, 1024], "dtype": torch.bfloat16, "is_need_logits": False, "seed": 2051},
    {"shape": [1, 1536], "dtype": torch.float16, "is_need_logits": False, "seed": 2052},
    {"shape": [1, 1536], "dtype": torch.bfloat16, "is_need_logits": False, "seed": 2053},
    {"shape": [1, 1024], "dtype": torch.float16, "is_need_logits": True, "seed": 2200},
    {"shape": [1, 1024], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2201},
    {"shape": [1, 1536], "dtype": torch.float16, "is_need_logits": True, "seed": 2202},
    {"shape": [1, 1536], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2203},
    {"shape": [1, 2048], "dtype": torch.float16, "is_need_logits": True, "seed": 2204},
    {"shape": [1, 2048], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2205},
    {"shape": [2, 1024], "dtype": torch.float16, "is_need_logits": True, "seed": 2206},
    {"shape": [2, 1024], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2207},
    {"shape": [64, 64], "dtype": torch.float16, "is_need_logits": False, "seed": 2300},
    {"shape": [64, 64], "dtype": torch.bfloat16, "is_need_logits": False, "seed": 2301},
    {"shape": [96, 64], "dtype": torch.float16, "is_need_logits": False, "seed": 2302},
    {"shape": [96, 64], "dtype": torch.bfloat16, "is_need_logits": False, "seed": 2303},
    {"shape": [64, 255], "dtype": torch.float16, "is_need_logits": False, "seed": 2304},
    {"shape": [64, 255], "dtype": torch.bfloat16, "is_need_logits": False, "seed": 2305},
    {"shape": [64, 64], "dtype": torch.float16, "is_need_logits": True, "seed": 2310},
    {"shape": [64, 64], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2311},
    {"shape": [96, 64], "dtype": torch.float16, "is_need_logits": True, "seed": 2312},
    {"shape": [96, 64], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2313},
    {"shape": [64, 255], "dtype": torch.float16, "is_need_logits": True, "seed": 2314},
    {"shape": [64, 255], "dtype": torch.bfloat16, "is_need_logits": True, "seed": 2315},
]


def _make_case(shape, dtype, seed, is_need_logits):
    batch, vocab = shape
    logits_gen = torch.Generator().manual_seed(seed)
    q_gen = torch.Generator().manual_seed(seed + 1000)
    topk_gen = torch.Generator().manual_seed(seed + 2000)
    topp_gen = torch.Generator().manual_seed(seed + 3000)

    logits = torch.randn(batch, vocab, generator=logits_gen, dtype=torch.float32).to(dtype)
    q = (torch.rand(batch, vocab, generator=q_gen, dtype=torch.float32) * 1.9 + 0.1).to(torch.float32)
    top_ks = torch.randint(1, min(32, vocab) + 1, (batch,), generator=topk_gen, dtype=torch.int32)
    top_ps = (torch.rand(batch, generator=topp_gen, dtype=torch.float32) * 0.45 + 0.5).to(dtype)
    return [logits, top_ks, top_ps, q, is_need_logits]


def get_input_groups():
    return [
        _make_case(case["shape"], case["dtype"], case["seed"], case["is_need_logits"])
        for case in TOP_K_TOP_P_SAMPLE_CASES
    ]


def get_init_inputs():
    return []
