# 算术优化（乘法替代除法、常量预计算）

## 1. 乘法替代除法

### 问题描述

**问题：** 除法比乘法慢 3-10 倍。

```python
# 问题代码：使用除法
scale = absmax / 127.0  # 每次都要做除法
x_q = x / absmax * 127  # 除法 + 乘法
```

### 优化方案

```python
# 优化代码：预计算倒数，用乘法替代
inv_127 = 1.0 / 127.0  # 预计算或在 host 端传入
scale = absmax * inv_127  # 仅一次乘法
inv_absmax = 1.0 / absmax
x_q = x * inv_absmax * 127  # 两次乘法，无除法
```

### 性能对比

| 操作 | 延迟 (cycles) |
|-----|--------------|
| 乘法 | ~4 |
| 除法 | ~15-40 |
| 收益 | **5-15%** |

## 2. 常量预计算

### 问题描述

**问题：** 在 kernel 内重复计算常量。

```python
# 问题代码：每次调用都计算常量
@triton.jit
def kernel(...):
    inv_max = 1.0 / 127.0  # 每次调用都计算
    scale = x * inv_max
```

### 优化方案

```python
# 优化代码：host 端预计算后传入
def host_function(...):
    inv_int8_max = 1.0 / 127.0  # 一次计算
    kernel[grid](..., inv_int8_max=inv_int8_max)

@triton.jit
def kernel(..., inv_int8_max, ...):
    scale = x * inv_int8_max  # 直接使用
```

### 常用预计算常量

| 常量 | 预计算值 | 用途 |
|-----|---------|------|
| `inv_127` | `1.0 / 127.0` | int8 量化 scale |
| `inv_255` | `1.0 / 255.0` | uint8 归一化 |
| `inv_sqrt_2pi` | `1.0 / sqrt(2 * pi)` | 高斯分布 |
| `log2_e` | `log2(e) ≈ 1.4427` | 指数变换 |

### 性能收益

- 避免每次 kernel 调用都计算常量
- 典型收益 **2-5%**

## 完整示例

```python
@triton.jit
def quant_kernel(
    x_ptr, xq_ptr, scale_ptr,
    stride, M, N, eps,
    int8_min, int8_max,
    inv_int8_max,  # 预传入 1/127，关键优化！
    BLOCK: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    # ...
    absmax = tl.max(tl.abs(x), axis=1, keep_dims=True)
    
    # 关键：用乘法替代除法
    scale = absmax * inv_int8_max  # 而非 absmax / int8_max
    x_q = tl.clamp((x / absmax) * int8_max, int8_min, int8_max)
    # ...

def host_function(x, xq, scale):
    inv_int8_max = 1.0 / 127.0  # 预计算
    quant_kernel[grid](..., inv_int8_max=inv_int8_max)
```

## 常见错误

### 错误 1：忘记预传入常量

```python
# 错误：在 kernel 内计算
inv_max = 1.0 / 127.0

# 正确：host 预传入
kernel[grid](..., inv_int8_max=1.0/127.0)
```

### 错误 2：精度问题

```python
# 注意：预计算可能影响精度
inv_127 = 1.0 / 127.0  # float32 精度
# 如果需要更高精度，用 float64 预计算后传入
```

## 案例: Per-Token Int8 Quantization

### 原始实现问题

```python
# 原始：使用除法
@triton.jit
def quant_kernel(...):
    x = tl.load(...)
    absmax = tl.max(tl.abs(x))
    scale = absmax / 127  # 除法
    x_q = round(x * 127 / absmax)  # 又一次除法
    ...
```

### 优化后实现

```python
@triton.jit
def quant_kernel_optimized(
    x_ptr, xq_ptr, scale_ptr,
    stride, M, N, eps,
    int8_min, int8_max,
    inv_int8_max,  # 预传入 1/127
    BLOCK: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offs = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < M
    col_offs = tl.arange(0, BLOCK)
    col_mask = col_offs < N
    mask_2d = row_mask[:, None] & col_mask[None, :]

    # 预计算指针
    x_ptr = x_ptr + row_offs[:, None] * stride + col_offs[None, :]
    xq_ptr = xq_ptr + row_offs[:, None] * stride + col_offs[None, :]
    scale_ptr = scale_ptr + row_offs[:, None]

    # 2D 加载
    x = tl.load(x_ptr, mask=mask_2d, other=0.0).to(tl.float32)

    # 向量化 reduce
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1, keep_dims=True), eps)

    # 乘法替代除法
    scale = absmax * inv_int8_max  # 而非 absmax / 127
    tl.store(scale_ptr, scale)

    # 量化
    x_q = tl.clamp((x / absmax) * int8_max, int8_min, int8_max).to(tl.int8)
    tl.store(xq_ptr, x_q, mask=mask_2d)
```

### Host 调用示例

```python
def per_token_quant_host(x, xq, scale):
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    ROWS_PER_BLOCK = 8 if N <= 256 else 4

    # 预计算常量
    inv_int8_max = 1.0 / 127.0

    grid = (triton.cdiv(M, ROWS_PER_BLOCK),)
    quant_kernel_optimized[grid](
        x, xq, scale, x.stride(0), M, N,
        eps=1e-10, int8_min=-128, int8_max=127,
        inv_int8_max=inv_int8_max,  # 预传入
        BLOCK=BLOCK, ROWS_PER_BLOCK=ROWS_PER_BLOCK,
    )
```

### 优化点总结

| 优化点 | 代码示例 | 性能收益 |
|-------|---------|---------|
| 乘法替代除法 | `scale = absmax * inv_int8_max` | 5-15% |
| 指针预计算 | `x_ptr = x_ptr + ...` | 3-8% |
| 预传常量 | `inv_int8_max` 预计算 | 2-5% |
| 多行并行 | `ROWS_PER_BLOCK=8` | 10-30x |
| 2D 向量化 | `row_offs[:, None]` | 2-4x |
