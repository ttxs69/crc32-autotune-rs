# CRC32 Auto-Optimization (Rust)

让 AI Agent 自主优化 CRC32 性能的项目。

## 项目结构

| 文件 | 作用 | 可修改 |
|------|------|--------|
| `benches/benchmark.rs` | 基准测试框架 | ❌ 固定 |
| `src/crc32.rs` | CRC32 实现 | ✅ Agent 修改 |
| `AUTOTUNE.md` | Agent 指令 | 人类修改 |

## 快速开始

```bash
# 运行基准测试
cargo bench

# 运行测试
cargo test

# 检查正确性
cargo test --lib
```

## 实验循环

LOOP FOREVER:

1. 查看 git 状态
2. 修改 `src/crc32.rs`，尝试优化
3. 运行测试：`cargo test`
4. 如果测试失败，回退
5. 运行基准测试：`cargo bench > bench.log 2>&1`
6. 读取结果：`grep "throughput" bench.log`
7. 如果吞吐量提升，保留；否则回退
8. 记录到 `results.tsv`

## 目标

最大化吞吐量（MB/s），同时保持正确性。

### 基线性能

- 基础查表法：~200-500 MB/s
- 目标：> 5000 MB/s

## 优化方向

### 1. Slice-by-N

一次处理多个字节，减少循环开销：

```rust
// Slice-by-4: 一次处理 4 字节
let chunk = u32::from_le_bytes([data[i], data[i+1], data[i+2], data[i+3]]);
let index1 = ((crc ^ chunk) & 0xFF) as usize;
let index2 = (((crc >> 8) ^ (chunk >> 8)) & 0xFF) as usize;
// ... 4 次查表
```

### 2. 更大的查找表

- 256 条目（基础）
- 8K 条目（slice-by-4）
- 64K 条目（slice-by-8）

### 3. SIMD 加速

使用 `std::arch::x86_64` 的 SIMD 指令：

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn crc32_avx2(data: &[u8]) -> u32 {
    // SIMD 实现
}
```

### 4. 多线程并行

分块计算，然后合并结果：

```rust
use std::thread;

fn crc32_parallel(data: &[u8]) -> u32 {
    let chunks: Vec<_> = data.chunks(data.len() / 4).collect();
    // 每个线程计算一个块
    // 合并部分 CRC
}
```

### 5. 硬件加速

如果可用，使用 CRC32 指令：

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_crc32_u64;
```

## 结果记录

`results.tsv` 格式：

```
commit	throughput_mb_s	status	description
a1b2c3d	450.0	keep	baseline table
b2c3d4e	1200.0	keep	slice-by-4
c3d4e5f	1100.0	discard	larger table
d4e5f6g	0.0	crash	incorrect results
```

## 约束

**可以修改：**
- `src/crc32.rs` 中的任何内容

**不能修改：**
- `benches/benchmark.rs`
- `Cargo.toml` 中的依赖
- 测试数据和种子

**正确性要求：**
- 输出必须与 `crc32fast` crate 一致

## 永不停止

一旦开始，持续运行直到手动中断。

## 禁止事项

**绝对不要：**
- 使用 `unsafe` 块调用外部 C 代码
- 使用 `crc32fast` 或其他现成库作为实现
- 修改测试用例

**可以：**
- 使用 `unsafe` 块进行 SIMD 操作
- 使用 `std::arch` 内联汇编
- 使用 Rust 标准库的所有功能