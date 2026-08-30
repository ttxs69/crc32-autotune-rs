# CRC32 Auto-Tune (Rust)

让 AI Agent 自主优化 CRC32 性能的项目。

灵感来自 [Karpathy 的 autoresearch](https://github.com/karpathy/autoresearch)。

## 性能（Apple M1 实测，受控测量）

| 输入大小 | 单线程 | 说明 |
|---|---|---|
| < 192 B | ~5 GiB/s | 硬件 CRC32 串行链 |
| ≥ 192 B | 16–33 GiB/s | NEON PMULL 折叠（双 fold-by-4 组 + 裸指针装载） |

- 多线程（rayon，> 1 MiB）：1MB ~34 GiB/s，10MB ~70 GiB/s，100MB ~45 GiB/s（DRAM 受限）
- 对比 crc32fast：单线程 2–4×，多线程最高 13×（1GB）
- x86_64 上另有 PCLMULQDQ（SSE）与 AVX-512 VPCLMULQDQ 路径（结构经标量 oracle 验证）
- 100MB 档受 DRAM 带宽限制；>32MiB 输入自动减少并发流数以降低 DRAM 争用

## 快速开始

```bash
# 运行基准测试
cargo bench

# 运行测试
cargo test

# CLI 使用
cargo run --release -- <file>
cargo run --release -- --bench 100

# 与 crc32fast 对比
cargo run --release --example compare
```

## 项目结构

| 文件 | 作用 | 可修改 |
|------|------|--------|
| `benches/benchmark.rs` | 基准测试框架 | ❌ 固定 |
| `src/crc32.rs` | CRC32 实现 | ✅ Agent 修改 |
| `AUTOTUNE.md` | Agent 指令 | 人类修改 |

## 目标

最大化吞吐量（MB/s），同时保持正确性。

- 基线（查表法）：~200-500 MB/s ✅ 已远超
- 当前：单线程最高 ~22 GiB/s，多线程最高 ~54 GiB/s

## 测量注意事项

criterion 基准对纯函数的小尺寸用例存在编译器提升（LICM/CSE）导致的数字通胀
（例如 1KB 档显示超物理上限的值）。评估小尺寸优化时请使用受控测量：
不透明随机数据 + 每次调用不同输入。详见 `results.tsv` 备注。

## Agent 使用方式

1. 在 AI 工具中打开项目
2. 让 Agent 读取 `AUTOTUNE.md`
3. Agent 开始自主优化循环

## License

MIT
