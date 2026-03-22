# CRC32 Auto-Tune (Rust)

让 AI Agent 自主优化 CRC32 性能的项目。

灵感来自 [Karpathy 的 autoresearch](https://github.com/karpathy/autoresearch)。

## 快速开始

```bash
# 运行基准测试
cargo bench

# 运行测试
cargo test

# CLI 使用
cargo run --release -- <file>
cargo run --release -- --bench 100
```

## 项目结构

| 文件 | 作用 | 可修改 |
|------|------|--------|
| `benches/benchmark.rs` | 基准测试框架 | ❌ 固定 |
| `src/crc32.rs` | CRC32 实现 | ✅ Agent 修改 |
| `AUTOTUNE.md` | Agent 指令 | 人类修改 |

## 目标

最大化吞吐量（MB/s），同时保持正确性。

### 基线

- 基础查表法：~200-500 MB/s
- 目标：> 5000 MB/s

## Agent 使用方式

1. 在 AI 工具中打开项目
2. 让 Agent 读取 `AUTOTUNE.md`
3. Agent 开始自主优化循环

## License

MIT