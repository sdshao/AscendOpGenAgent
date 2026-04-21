"""Block-level TileLang design for top_k_top_p_sample.

This file only keeps the block/task partition, workspace ownership, and the
cross-stage pipeline skeleton. All concrete math is intentionally deferred to
`design/tile_level/top_k_top_p_sample.py`.

The scheduling mirrors the current AscendC structure and follows the
`archive_tasks/rms_norm/design` style:
- `small_row_fastpath` for `row_len <= 1024`
- `medium_row_streaming` for `1024 < row_len <= 4096`
- `large_row_segmented` for `row_len > 4096`
"""

import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[4, 5], workspace_idx=6, pass_configs=pass_configs)
def top_k_top_p_sample(
    row_num,
    row_len,
    is_need_logits=False,
    top_k_guess=32,
    eps=1e-8,
    dtype="float16",
):
    num_physical_cores = 40
    used_core_num = min(num_physical_cores, max(1, row_num))
    tasks_per_core = T.ceildiv(row_num, used_core_num)
    topk_limit = min(row_len, 1024)
    topk_pad = ((topk_limit + 7) // 8) * 8
    softmax_tile = min(row_len, 8192)
    softmax_loops = T.ceildiv(row_len, softmax_tile)
    sort_tile = min(row_len, 1024)
    sort_loops = T.ceildiv(row_len, sort_tile)
    guess_k = min(top_k_guess, row_len, 1024)

    @T.prim_func
    def small_row_fastpath(
        Logits: T.Tensor((row_num, row_len), dtype),
        TopKs: T.Tensor((row_num,), "int32"),
        TopPs: T.Tensor((row_num,), dtype),
        Q: T.Tensor((row_num, row_len), "float32"),
        SelectedIdx: T.Tensor((row_num,), "int64"),
        SelectedLogits: T.Tensor((row_num, row_len), "float32"),
        Workspace: T.Tensor((6, row_num, row_len), "float32"),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            core_idx = cid
            _ = vid

            # Workspace ownership follows the AscendC host-side layout:
            #   Workspace[0] -> logits / softmax scratch
            #   Workspace[1] -> top-k or partial-sort buffer
            #   Workspace[2] -> merged-sort buffer
            #   Workspace[3] -> kept logits buffer after top-p cutoff
            #   Workspace[4] -> q-sampling score scratch
            #   Workspace[5] -> optional scatter-output scratch
            for local_idx in T.serial(tasks_per_core):
                row_id = core_idx * tasks_per_core + local_idx
                with T.Scope("V"):
                    if row_id < row_num:
                        # TODO(tile-level):
                        # - load the full logits row and q row into UB
                        # - read per-row top_k / top_p and clamp top_k to [1, min(row_len, 1024)]
                        # - stable descending top-k selection for the kept candidate window
                        # - softmax only over the kept top-k candidates
                        # - prefix-scan probabilities until cumulative > top_p when 0 < top_p < 1
                        # - optionally build q-rescore = prob / (abs(q) + eps)
                        # - argmax the final scores and write SelectedIdx[row_id]
                        # - when is_need_logits, scatter preserved logits to SelectedLogits[row_id, :]
                        _ = Logits
                        _ = TopKs
                        _ = TopPs
                        _ = Q
                        _ = SelectedIdx
                        _ = SelectedLogits
                        _ = Workspace
                        _ = row_id
                        _ = topk_limit
                        _ = topk_pad
                        _ = guess_k
                        _ = eps
                        _ = is_need_logits

    @T.prim_func
    def medium_row_streaming(
        Logits: T.Tensor((row_num, row_len), dtype),
        TopKs: T.Tensor((row_num,), "int32"),
        TopPs: T.Tensor((row_num,), dtype),
        Q: T.Tensor((row_num, row_len), "float32"),
        SelectedIdx: T.Tensor((row_num,), "int64"),
        SelectedLogits: T.Tensor((row_num, row_len), "float32"),
        Workspace: T.Tensor((6, row_num, row_len), "float32"),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            core_idx = cid
            _ = vid

            for local_idx in T.serial(tasks_per_core):
                row_id = core_idx * tasks_per_core + local_idx
                with T.Scope("V"):
                    if row_id < row_num:
                        # TODO(tile-level):
                        # - stage the full row to Workspace[0] tile by tile
                        # - compute full-row softmax through multi-pass row-max / exp / normalize
                        # - if top_k <= 1024, reuse the top-k path on the normalized candidate window
                        # - otherwise try the guessed-top-k prefix first for top-p cutoff
                        # - if the guessed prefix is insufficient, run segmented sort/merge via Workspace[1:3]
                        # - optionally compute q-rescore in Workspace[4]
                        # - write SelectedIdx[row_id]
                        # - when is_need_logits, scatter preserved logits to SelectedLogits[row_id, :]
                        _ = Logits
                        _ = TopKs
                        _ = TopPs
                        _ = Q
                        _ = SelectedIdx
                        _ = SelectedLogits
                        _ = Workspace
                        _ = row_id
                        _ = softmax_tile
                        _ = softmax_loops
                        _ = sort_tile
                        _ = sort_loops
                        _ = guess_k
                        _ = eps
                        _ = is_need_logits

    @T.prim_func
    def large_row_segmented(
        Logits: T.Tensor((row_num, row_len), dtype),
        TopKs: T.Tensor((row_num,), "int32"),
        TopPs: T.Tensor((row_num,), dtype),
        Q: T.Tensor((row_num, row_len), "float32"),
        SelectedIdx: T.Tensor((row_num,), "int64"),
        SelectedLogits: T.Tensor((row_num, row_len), "float32"),
        Workspace: T.Tensor((6, row_num, row_len), "float32"),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            core_idx = cid
            _ = vid

            for local_idx in T.serial(tasks_per_core):
                row_id = core_idx * tasks_per_core + local_idx
                with T.Scope("V"):
                    if row_id < row_num:
                        # TODO(tile-level):
                        # - stream the row by `sort_tile` segments into Workspace[0]
                        # - compute segmented row-max / exp / normalize over the whole row
                        # - produce partial descending runs in Workspace[1]
                        # - merge the runs into Workspace[2]
                        # - apply top-p cutoff and gather kept logits into Workspace[3]
                        # - optionally compute q-rescore into Workspace[4]
                        # - write SelectedIdx[row_id]
                        # - when is_need_logits, scatter preserved logits to SelectedLogits[row_id, :]
                        _ = Logits
                        _ = TopKs
                        _ = TopPs
                        _ = Q
                        _ = SelectedIdx
                        _ = SelectedLogits
                        _ = Workspace
                        _ = row_id
                        _ = sort_tile
                        _ = sort_loops
                        _ = guess_k
                        _ = eps
                        _ = is_need_logits

    if row_len <= 1024:
        return small_row_fastpath
    if row_len <= 4096:
        return medium_row_streaming
    return large_row_segmented
