# Timeline simulator 设计

## 0. 这是干什么的

手上有一堆某个 harness 吐出来的 trace、一份训练 recipe、一堆 config，但没有 profile。simulator
要回答两件事：

- 一步 GRPO 端到端要多久；
- 这一步的时间线长什么样：什么时候在 decode，什么时候在等 tool，什么时候在等打分，什么时候在训练。

它是一个 trace-driven 的 discrete-event simulator。trace 告诉它「要干哪些活、每段活有多长」，
它自己算「这些活放到这套系统上，各自什么时候发生」。

trace 没有固定格式，harness 当时怎么高兴怎么写。所以 trace 里有什么就用什么，缺的只能估，
**估出来的数必须明着标出来**。

simulator 出错基本都是静默的，最常见的三种：

1. 把答案抄进了输入（比如直接用 trace 里的到达时刻），看起来很准，其实什么都没预测；
2. 参数是在这一条 trace 上拟合的，换个 workload 就错；
3. 某个数是 adapter 编的，却被当成实测值用了。

后面大部分规矩都是在堵这三个口子。

---

## 1. 能回答什么，不能回答什么

**能回答：改系统，不改模型的行为。** 改 TP、replica 数、concurrency cap、KV dtype、换硬件——
模型写出来的 token 不变，回放 trace 里的 token 流照样成立。

**不能回答，必须认出来并拒绝：**

- 改 turn cap、工具集、prompt 模板、换模型。trajectory 本身变了，旧 trace 代表不了。
- staleness、curriculum、sampling temperature。同理，它们改的是 policy 写什么。
- 一个模型副本拆到两台以上机器上跑。一台机器里 GPU 之间的通信又快又稳，测 cost model 的时候
  顺带就测进去了；跨机器要走网卡，又慢，耗时还要看网络上同时有谁在抢，cost model 测不出来。
  要算这个就得模拟网络本身，[simllm](https://github.com/openfabric-systems/simllm) 做的就是这个，
  我们不重做。多个副本各占一台机器没问题，副本之间不通信。

**两种时候，别混：**

| | 手上有什么 | 干什么 |
| --- | --- | --- |
| 开发时 | 我们自己的 run：trace + 真实 timeline | 拿真实 timeline 对答案，证明 simulator 是对的 |
| 使用时 | 别人的 trace + config，**没有 timeline** | 报一个带 interval 的 e2e。没人能告诉你对不对 |

真正要交付的是使用时那一种。在自家 trace 上对上 `analysis/<run>/timeline/*.jsonl` 只是里程碑：
证明了 simulator 有资格去给没有答案的 trace 报数。

---

## 2. 整体长什么样

```
trace  ──adapter──>  IR（要干哪些活）  ──┐
                                          ├──>  engine  ──>  预测 timeline
config ────────────> config bundle     ──┘                  汇总表（e2e interval 等）
                                                             provenance 报告（每个数从哪来）
```

engine 里面：

```
虚拟时钟 + 事件队列
├─ router        照抄 verl，决定请求发去哪个 replica
├─ scheduler     每个 replica 一个真的 SGLang Scheduler，model worker 换成 stub（§7）
├─ cost model    scheduler 每跑一步，告诉它这一步花了多久（§8）
├─ 资源池        container、打分用的 eval container，按 semaphore 排队
└─ 编排          group barrier；训练各段按给定时长占住时间
```

一个 episode 在里面走一遍：放出来 → 拿到 container → 发第一轮请求 → router 选 replica →
scheduler 把它排进 batch，一步一步 decode，每一步的耗时由 cost model 给 → 生成完 → 跑 tool
（耗时从 trace 回放）→ 发下一轮请求 → …… → 提交，抢 eval container 打分 → 整个 batch 都打完分 →
训练各段 → 下一步。

注意：第二轮以后的请求**什么时候到**，是 simulator 自己算的——上一轮什么时候生成完（simulator 算）
加上 tool 跑多久（trace 给）。

---

## 3. 最重要的规矩：答案不能当输入

> 凡是在真实 timeline 里能看到的量，都是 simulator 要算的，不能喂给它。

两个理由：

- 喂进去等于把答案抄进题面。simulator 永远不会错，也永远告诉不了你任何事。
- 更硬的理由：使用时根本没有 timeline，这些量想喂也拿不到。所以这不是自不自觉的问题，
  不这么做就跑不起来。

怎么分？看它换一套系统配置会不会变：

| 可以喂：活本身有多长、有多少 | 不能喂：活放到这套系统上什么时候发生、结果如何 |
| --- | --- |
| 容器启动、tool、打分各要多久 | 每一轮请求什么时候到 server |
| 每一轮的 `prompt_tokens` / `completion_tokens` | 排队多久、TTFT、TPOT、batch 里当时有谁 |
| 这条请求接在哪条后面，结构上最多能复用多少 token | `cached_tokens`：实际从 cache 里省了多少 |

左边这些，改 TP、改 KV dtype、改并发上限都不会变；右边这些正是这些改动会改变的。

**例子：prefix hit rate 不是输入。** trace 能给的是结构：这条请求接在哪条后面、共享多少 token。
实际命中多少由 radix cache、LRU eviction、`page_size` 和 router 一起决定，是 simulator 要算的。
trace 里记的 `cached_tokens` 拿来对答案，而且很好用：

```
run 20260912-072938，95,403 对相邻 turn
  prompt_{k+1} < prompt_k + completion_k        0 例   -> 下一轮 prompt 就是上一轮整段再加新内容
  cached_{k+1} - (prompt_k + completion_k)      p50 -113，p90 -69，全部为负
  cached_{k+1} == floor(prompt_k / 64) * 64     99.62%（95,359 对里 94,999 对；其余多一整页）
```

少掉的那 ~113 个 token 是上一轮的整段输出，加上 prompt 不满一页的零头。SGLang 0.5.8 请求结束时往
radix tree 里插的是 `req.fill_ids`，也就是最后一次 prefill 时的 prompt；decode 路径从不更新它，输出
的 KV 直接释放。所以下一轮只能命中上一轮的 prompt，按页向下取整，上一轮写的东西每轮都要重新
prefill 一遍。（这里原来写的是「页对齐加上扣下的最后一页」，是猜的，2026-10-07 拿 stub 跑真
scheduler 时才发现不对。）一个照「输出也进 cache」去抄 SGLang 的 simulator 会预测差值不到一页，
马上露馅；跑真 scheduler 不用知道这件事，自动就对（§7）。

同样不能喂的还有：prefill / decode 怎么交错、队列深度、KV token usage、retraction 次数、
ramp / plateau / drain 的形状。

**唯一的例外**是 `Session.release_offset_s`：每个 session 相对 t=0 什么时候放出来，sync GRPO 下
全是 0。它来自 recipe，不来自 trace——别人 trace 里的到达节奏是**那套系统当时的行为**，不能照抄。

**最容易踩的坑：时间戳相减得到的耗时。** 很多 trace 只有时间戳，tool 耗时只能这么算：
`下一个请求发出的时刻 − 上一个回复到达的时刻`。这个差里混着排队、client 开销、harness 自己的处理
时间，还有采集那次运行有多挤。拿它当 tool 耗时喂进去，等于把「系统当时等了多久」塞进了「活本身
有多久」，simulator 的 e2e 会显得很准，因为答案是抄来的。所以这种耗时要么当成估的数、只当
upper bound 用，要么先证明采集那次系统没跑满。当场就能做的检查：把 config 里的 replica 数翻倍，
看预测的 e2e 动不动。不动，就是被污染了。

开发时对答案（§12），会分级把真实的到达时刻喂回去。那是对答案用的脚手架，只在开发时存在，
千万别接进使用时的流程。

---

## 4. 输入一：IR

IR 借的是编译器的说法：adapter 把各种 harness 的 trace 翻成 IR，simulator 只认 IR，不知道 trace
原来长什么样。定义在 [src/simulator/ir.py](src/simulator/ir.py)：

| class | 是什么 | 关键字段 |
| --- | --- | --- |
| `Workload` | 一整份 workload | `sessions`、`prefixes`、`source`（从哪来） |
| `Session` | 一串相关的活，比如一个 episode | `requests`、`tasks`、`leases`、`group`、`release_offset_s`、`spawned_by`、`problem` |
| `Request` | 一次 inference 调用 | `prompt_tokens`、`completion_tokens`、`prefix_parent`、`shared_prefix_tokens`、`max_new_tokens`、`after` |
| `Task` | 不是 inference 的活：tool、起容器、打分 | `duration_s`、`pool`、`after` |
| `Lease` | 跨好几个活一直占着的资源，比如一个 episode 从头占到尾的 container | `pool`、`acquire_before`、`release_after` |
| `Prefix` | 好几个 session 共享的一段内容，比如 system prompt、题面 | `tokens`、`parent` |
| `Unknown` | 这个字段没人量过，要从哪个分布抽 | `dist`、`knob`、`scale` |

`Request`、`Task`、`Prefix` 还各带两张小表：`prov` 和 `unknown`，见下面「IR 怎么说不知道」。

`Session.problem` 是这个 session 在做哪道题，比如 SWE-bench 的 instance id。engine 不读它，只有 prior
读：有些活多长是题目决定的，跟 run 无关。同一个 instance 的测试跑 15 秒还是 770 秒，换谁写 patch
都差不多。所以 eval 的 prior 按 instance 查表（§5）。

一个 Session 是一张 DAG：node 是 Request 或 Task，edge 是 `after`（「等它做完我才能开始」）。
一个 SWE-bench episode 和一条 open-loop serving 负载都要能用这同一套 class 写出来，
[tests/test_simulator_ir.py](tests/test_simulator_ir.py) 开头两个 test 就是这两种。

**IR 里故意没有时间。** 没有时间戳、没有 `cached_tokens`、没有 TTFT。§3 那条规矩不靠自觉，靠 IR
里根本没有这些字段。真实系统的这些数放在 [src/simulator/observed.py](src/simulator/observed.py)，
只用来对答案。

**叫 `Session` 不叫 `Episode`，是故意的。** engine 需要知道哪些请求是一伙的（router 靠它把同一个
session 的请求发到同一个 replica），但不许知道它们为什么是一伙的。engine 一旦知道这是 agent 的
episode，就可能被我们这个 workload 带偏。

**`Prefix`：第一轮请求不是冷的。** 没有 `Prefix` 的时候，第一轮请求只能写成「不接任何东西」，
等于说它要从头 prefill。实测完全不是：

```
run 20260912-072938，4,243 个第一轮请求
  cached / prompt        p10 0.39   p50 0.983   p90 0.996
  合计                    9.24M / 11.19M = 82.6%
  命中不到 50% 的         1,118 (26%)
```

同一个 group 的 8 条 rollout 共享 system prompt 和题面。那 26% 也不是噪声：530 个 group × 2 个
replica = 1,060，跟 1,118 对得上——每个 group 在每个 replica 上冷一次，其余都命中。这个数正好
可以拿来验 router。

`Prefix` 只写长度和接在谁后面，不写内容。第一轮的 `prefix_parent` 指向一个 `Prefix`，后面的轮次
指向上一轮的 Request。共享内容不等于要等它，所以 `after` 不能指向 `Prefix`，也不能跨 session——
谁先到谁付 prefill，这由 simulator 算。按字节算这一项不大（只占全部 cached token 的 0.7%），
但换成大量请求共享 system prompt 的 serving workload，它就是全部。

**`max_new_tokens` 是个坑。** SGLang 接纳新请求时，会给每个在跑的请求预留
`min(max_new_tokens - 已生成, 4096) × new_token_ratio` 个 token 的 KV（`schedule_policy.py:433`）。
真 server 只知道 cap，不知道这条请求最后会写多少。所以 simulator 让请求停下的办法只能是：在第
`completion_tokens` 个位置吐 EOS。**绝对不能**图省事把 `max_new_tokens` 设成 `completion_tokens`——
那样预留就从「按 cap 估」变成「按答案留」，retraction 永远不会发生，而且不报任何错。

陌生 trace 多半没记 cap，所以这个字段可以是空的，由 config bundle 填。engine 门口的
`require_runnable` 拦住还空着的，比让 adapter 随手编一个强。

**IR 怎么说「不知道」。** 陌生 harness 的 trace 一定缺字段。要是每个字段都必须填一个数，adapter
除了编没有别的办法，编出来的还跟实测的长得一模一样——这就是 §0 那三种静默出错的第三种。所以
`Request`、`Task`、`Prefix` 各带两张小表：

- `prov`：每个字段的值哪来的。trace 读的、config 给的、推出来的（tokenizer 配模板重建、拿相邻
  turn 相减）、估的。表里没有的字段就是没说，报告里照实列「没说」，不替它猜。
- `unknown`：这个字段没人量过，要从哪个分布抽。

字段本身还是裸的 `int` / `float`，engine 一行都不用改。一条规矩守着：`unknown` 里有的字段，`prov`
必须写着「估的」。想挂个分布又想说这是实测值，validate 当场拒。

**抽样放在 engine 前面：`realise`。** [src/simulator/realise.py](src/simulator/realise.py) 吃一个带
分布的 workload 加一个 seed，吐一个全是具体数的 workload，engine 只认后者。这样 §6 那句「engine
不许根据 provenance 改变行为」就不用靠自觉：到 engine 手里的时候，估的数和实测的数就是同一个
float，想区别对待也没得区别。`prov` 跟着 workload 一路走进报告，engine 从不读它。

**lease 两端的 task 不再自己要 slot。** 起容器的 task 是在 session 的 container lease 里跑的，
`pool` 必须是 None。两个 adapter 原来都给它写了 `pool="container"`，等于一个 episode 要两个 slot：
session 比 slot 多的时候，每个 lease 拿着一个、等第二个，谁也跑不了。64 个 session 时看不出来，
喂 0912 step 1 的 2,048 个 session 时卡死了。现在 `validate` 拒这种写法。

**一个 node 可以等好几个 node。** `after` 是一个 list。一个 assistant turn 并行发三个 tool call、
下一轮等三个都跑完，这在别的 harness 里是常态；串成一条链就是凭空加依赖，把 session 拉长。

**subagent 用 `spawned_by`。** 子 agent 是独立的 session，由父 session 的某个 task 拉起，那个 task
要等子 session 整个跑完才算完。这是唯一允许跨 session 的依赖：两个平级 session 互相等是编排，
父子相等是 workload 本身的形状。子 agent 单独一个 session，也就单独一个 routing key，这正是想要
的——它的 prompt 跟父亲几乎不共享，塞进父 session 会让 router 以为它们该在一台机器上。

**还缺的：`Workload.source` 只有一个字符串**，几个来源混在一份 workload 里就不够用了。

**还缺的：unknown 之间不能有关系。** 每个 unknown 字段各抽各的。可 token 数不是这样：下一轮的
`prompt_tokens` 等于上一轮 prompt 加 completion 再加新内容，`shared_prefix_tokens` 也由它们算出来。
一份连 token 数都没记的 trace，adapter 现在只能拒绝，因为分开抽出来的三个数一定对不上 `validate`
的算术。要支持它，IR 得允许「这个字段 = 那几个字段的和」这类写法。只有对话的 trace 不受影响：
用 tokenizer 重建出来的 token 数是 derived，天然自洽。

`Task.duration_is_cap`（这次 tool 是不是超时被杀了）engine 现在不读。留着它，是因为当时不记，
事后就补不回来了。

---

## 5. 输入二：config bundle

IR 只装 workload，不装系统配置：`Lease` 只说要哪个 pool，不说 pool 多大。系统配置走另一个输入：

| 从哪来 | 装什么 | 缺了怎么办 |
| --- | --- | --- |
| recipe | batch、`n`、group barrier、turn cap、worker 数、每个 worker 的 container 上限、eval pool 大小、session 的释放节奏 | 拒绝跑 |
| SGLang server args | SGLang 版本；TP、replica 数、`page_size`、`chunked_prefill_size`、`max_running_requests`、`new_token_ratio` 的三个参数、`max_total_num_tokens`、overlap 开没开 | 有活着的 server 就直接读；没有就跑 SGLang 自己算默认值的那段代码 |
| 模型 `config.json` + 硬件规格 | layer 数、KV head 数、head dim、active expert、dtype 宽度、HBM 带宽、`cuda_graph_max_bs` | 拒绝跑 |
| calibration 结果 | cost model 的三个拟合量（§8） | 本机没测过，就用别的机器上最近的测量点，并打印离它多远 |
| prior（trace 没记的耗时） | 每个 knob 一个数：中位数要几秒；拿不准就给 `[lo, hi]` | 用我们自己 run 量的分布，报告里标「默认」 |

**一项都不许拿「常见默认值」顶。** SGLang 很多默认值是按 GPU 显存分支的，照抄就是在编。整个
bundle 要能原样存成一个文件、跟着预测结果走，不然这份预测没法复现。

**落地了的部分（2026-10-13）。** 读文件、查齐全在 `src/simulator/bundle.py`；拿 SGLang 和 calibration
核对、建出 replica 在 `src/simulator/engine/system.py`。例子是 `configs/simulator/example_bundle.yaml`，
0912 自己的是 `configs/simulator/run-20260912-072938.yaml`。一个文件四段：

- `recipe`：worker 数；每个 pool 每个 worker 几个 slot，pool 名字用 workload 里的；`request_cap`，trace
  没记 `max_new_tokens` 的请求按 `min(max_new_tokens, max_sequence_tokens − prompt, context_length − 1 − prompt)`
  填（verl 的规则写成通用的样子），prov 记 `config`。填完要是有请求写得比 cap 还多，说明这条 trace 不是在
  这份配置下跑的，拒。
- `server`：SGLang 版本、replica 数、`args`、`env`。`args` 用 SGLang 自己的字段名，和活 server 的
  `/get_server_info` 一样；必填的是 scheduler 做决定要用的那些，加上 cost model 要对上的几项。不是
  `ServerArgs` 字段的名字直接拒，拼错了不会悄悄用默认值。`env` 是 `new_token_ratio` 那三个环境变量，
  server 没设也要把 SGLang 的默认值写上。
- `calibration`：拟合好的 `calibration.json`。路径相对 bundle 文件，不相对当前目录。
- `priors`：同上，由 `load_priors` 读。

建 system 时查三件事，哪件对不上都拒：SGLang 版本和 stub 的一样；`tp_size`、`attention_backend`、
`moe_runner_backend`、`kv_cache_dtype`、`disable_cuda_graph` 和 calibration 那份 grid 的 `meta.json` 一样；
calibration 的 TP 和 server 的一样。030010 用的是 `flashinfer_trtllm` MoE，grid 是 `triton`，这一条会直接拒。

还没做：从活 server 自动读 `args`；没有活 server 时跑 SGLang 算默认值的那段代码；group barrier 和训练段
要的 batch、`n`、训练时长。

**每个 worker 各一份 pool（2026-10-13）。** verl 把一个 batch 切成连续的几块，一块给一个 agent-loop
worker（`AgentLoopManager.generate_sequences` 里的 `DataProto.chunk`），semaphore 是每个进程自己的。所以
engine 把 workload 里的 session 按顺序切成 worker 数那么多块，每块的 session 只用这块自己的 slot
（`engine.split`）。块长是 `ceil(n / workers)`，和 `torch.chunk` 一样。GRPO 一个 group 的 n 条在 batch 里
挨着，块长是 n 的倍数时一个 group 不会跨 worker。

这样 session 的顺序就有了意思：它是 harness 拿到 batch 的顺序（见 `Workload.sessions` 的注释）。worker
数和怎么切是 recipe 的事，顺序是 workload 的事。

0912 的 timeline 每个 worker 一份文件，正好拿来验这条规则：step 1 八个 worker 各 256 个 session、32 个整
group，没有一个 group 跨 worker。adapter 现在一份文件一份文件地排，同一份里按开始等 container 的时刻排
（`admission_wait` 的 `t`，不是它等了多久）。这就是那个 semaphore 先来先得的顺序：八个 worker 里，按这个
顺序排在前 33 个的都马上拿到了 container，后面的一个都没有。它和 batch 顺序差一点：verl 在这之前 await
了一次 chat template，相邻的偶尔对调，所以一个 group 中间有时插进别的 group 的一两条（32 个 group 排出来
是 32 到 36 段）。

**prior 是唯一有默认的一项。** 前几行是系统配置，填错了预测就错，而且看不出来。prior 本来就是
「估的」：它已经带着 interval，sensitivity table 会单独给它一行，报告会写它从哪来。用我们的 run
顶上，最坏是那一行的 interval 偏了，用户看得见，也知道该补哪个数。拒绝跑反而让只有对话的 trace
一个数都拿不到。

默认值是一份存进 repo 的文件 `src/simulator/adapters/default_priors.json`，由
`scripts/simulator_dev/build_default_priors.py` 生成。分布取 run 20260912-072938 step 1 的全部实测耗时。
另外七个 run 的 14 个 step，加上 0912 自己的 step 2，一共 15 个 step（0903 和 030010 各三个、0906 三个、
0908 两个 run 共三个、1005 两个 run 各一个、0912 step 2），做两件事：

- **eval 按 instance 建表。** 量过至少 4 次的 instance（482 个）有自己的一组值，抽的时候从这组值里
  原样抽一个，不插值。插值会让最大值只算半个，8 个值里有一个 916 秒的话，这个 instance 的均值就
  少四成。没量过的 instance 用整体分布。
- **scale 跨 step 量，不只跨 run。** 每个 run 同一个 step 抽到的题是一样的，只拿 step 1 比，量到的
  只是同一批题上的噪声。对每个 step 算「实测总量 / prior 预测的总量」，预测时把这个 step 所在的
  run 从表里拿掉（不然表里有答案，比值永远是 1），再把 1.0 包进去。没建表的 knob 就是均值之比。

**用户怎么配。** 在 bundle 的 YAML 里写一个 `priors` 段，例子在
`configs/simulator/example_bundle.yaml`：

```yaml
priors:
  eval_duration: 45            # 我们的 harness 打分大概 45 秒
  tool_duration: [0.5, 2]      # 拿不准，0.5 到 2 秒之间
  # 没写的 container_start、patch_recover、cleanup 用我们的
```

写完跑 `python scripts/show_priors.py <bundle>`，每个 knob 打一行：中位数、范围、用的是用户的数还是我们
的。代码里是 `load_priors(path)`，返回的 `Priors` 直接交给 adapter。

**不知道要填什么。** 跑 `python scripts/priors_template.py`，它把所有 knob 列出来，每个都注释掉、写着
我们的数和它在我们 run 里占多少 task 时间：

```yaml
priors:
  # eval_duration: 21.7        # running the tests that score one submitted patch. 66.3% of task time, our runs drifted 0.90-1.10x; 482 problems have their own values, moved with yours
  # tool_duration: 0.22        # one tool call (bash, edit, submit, ...). 29.4% of task time, our runs drifted 0.71-1.04x
  # container_start: 0.377     # starting the container one episode runs in. 3.7% of task time, our runs drifted 0.79-1.09x
  # patch_recover: 0.737       # pulling the diff out of an episode that never submitted. 0.6% of task time, our runs drifted 0.88-1.02x
  # cleanup: 0.000697          # removing the container when an episode ends. 0.0% of task time, our runs drifted 0.77-1.00x
```

按占比排，所以实际上只有头两行值得填：打分和 tool 加起来 96%。哪行都不填也能跑，全用我们的。

只有两种写法：

- `eval_duration: 40`：中位数 40 秒。把我们的分布整体挪过去，让中位数落在 40，一次和一次之间差多少
  还用我们量的。按 instance 的表也乘同一个倍数：哪道题慢还是我们量的，整体多慢听用户的。别人一般说得出「打分大概要多久」，说不出它的尾巴长什么样。给一个数等于声明这个
  中位数是对的，这一行没有 scale，interval 不会因为它变宽。
- `tool_duration: [0.5, 2]`：拿不准，中位数在 0.5 到 2 秒之间。这个范围就是 scale，sensitivity table
  扫的就是它。

不管哪种，**我们自己的 scale 都不带过去**：它量的是我们几个 run 之间差多少，跟别人写的数准不准没
关系。knob 名字写错直接拒绝，不然默认值原样留着，用户还以为自己改了。

默认值不能拿来转它自己量过的 run：上面八个 run 的任何一个 step、包括对话 dump，`check` 都会拒，
用户给了中位数也一样，因为分布形状还是那个 run 的。要转这几个 run，就用别的 run 现建 prior（削
trace 实验就是这么做的）。

训练各段的时长还没有 knob，默认值里也没有它。engine 有了训练段以后再加。

---

## 6. 每个数都要说清 provenance

simulator 用到的每个数，属于下面五种之一：

| 从哪来 | 例子 | 注意 |
| --- | --- | --- |
| trace 给的 | token 数；tool 耗时（如果 trace 自己记了） | 同一个字段在不同 trace 里来路可能不同，见下 |
| config 读的 | TP、`page_size`、HBM 带宽 | 从活 server 读的，比从默认值代码解出来的可靠 |
| 实测拟合的 | 只有 cost model 的三个量 | 只在专门设计的测量 grid 上拟合，绝不在 run trace 上拟合（§8） |
| simulator 算的 | 到达时刻、batch 组成、TTFT、`cached_tokens` | 永远是输出；开发时拿来对答案 |
| 估的 | trace 里没有，只能从某个分布里抽 | 必须说清从哪个分布抽、那个分布又是哪来的 |

我们自己的 run 里，有些量现在没记、但加一行 log 就能有（比如每个请求落在哪个 replica），这叫
instrumentation gap。别人的 trace 补不了，只能算「估的」。

**provenance 跟着数据走，不跟着字段走。** 同一个 `prompt_tokens`：trace 有 API usage 字段时是
trace 给的；只有对话时要靠 tokenizer 加完全一致的模板重建；只记了轮数时就只能估。所以 provenance
要写进 IR、跟着数据走，不能只写在这份文档的表里。engine 不许根据 provenance 改变行为——这一条
靠 `realise`（§4）落实：抽样在 engine 之前做完，engine 拿到的全是裸 float，想区别对待也没得区别。

**有估的数，就必须这样报：**

- e2e 报一个 interval，不报一个点。输入是分布，输出就不可能是一个点。
- 出一张 sensitivity table，说清是哪个估的数把 interval 撑开的。只说「1.4–2.2h」，用户不知道
  该去补什么。
- 随机抽样用固定 seed。不然对答案时分不清是模型错了，还是这次抽的数不一样。

sensitivity table 的算法：每次只动一个估的数，从它分布的 p10 扫到 p90，其余固定在 p50，看 e2e 动
多少，按动的大小排序。下表的数字只是示意：

| 估的数 | 分布从哪来 | e2e 动了多少 |
| --- | --- | --- |
| 训练各段时长 | 没有模型，只有一个范围 | ±38 min |
| tool 耗时 | cap 16 / cap 40 两个 run 的分布 | ±6 min |
| 打分耗时 | `score` p50 21.6s | ±4 min |
| 容器启动 | p50 3.3s | ±1 min |

第一行就是要交给用户的那句话：「去把训练各段的时长记下来，interval 能缩掉一大半。」因为每次
只动一个，各行加起来**不等于** interval 的总宽度，两个数都要报。

**「一个估的数」在代码里叫 knob。** 每个 unknown 字段挂一个 knob 名，一个 knob 底下可能只有一个
字段（训练段），也可能有十万个（tool 耗时），table 一行就是一个 knob，还要报它管着多少个字段：
让用户去补一个数和去补十万个数，是完全不同的两件事。扫的时候 seed 不变，所以每个 node 自己那
一抽在整条曲线上是同一个数，e2e 动的那点只能是这个 knob 带来的。

**借来的分布要自己说清可能偏多少。** 一个 unknown 带两样东西：`dist` 说单个值长什么样，`scale`
说这整条分布可能整体偏多少倍。两件事推不出彼此——tool 和 tool 之间差多少，跟 tool 耗时的中位数
估错多少，根本没关系；拿前者当后者，sensitivity table 就会永远指着尾巴最重的那个量。`scale` 也
正是没有就该说没有的东西：少了它，每个 node 各抽各的，几万个 node 一平均就什么都不剩。两万个
tool task 抽 64 次，预测总时长只晃 2.1%，比两个 replica 之间 4.0%–16.8% 的噪声还小。那不叫预测。

**目前设计里最大的洞是训练段。** simulator 不给 FSDP 建模，训练各段按给定时长当成一整块。没有
profile 时它们只能估，而 `old_log_prob`、`ref`、`adv`、`update_actor`、`update_weights` 加起来
占一步的 52%。所以只有对话的 trace，e2e 的 interval 会被这一个估的数主导，除非另外从 recipe
给训练段建个模型。

**trace 里可能有什么。** 按信息从少到多：

| trace 里有 | 典型来源 | 于是 |
| --- | --- | --- |
| 只有对话 | `.traj`、消息 jsonl、各种日志 | 配 tokenizer 和 prompt 模板，能重建 token 数、prefix 结构、轮数、退出原因。所有耗时都只能估 |
| + 每条事件的时间戳 | 大多数 harness | 耗时只能相减，是被污染的（§3） |
| + API 的 usage 字段 | 记了 usage 的 harness | token 数精确。里面的 `cached_tokens` 是答案，adapter 要把它放进 `observed`，不进 IR |
| + 每段活自己记了耗时 | 我们的 harness（`tool_called` 带 `elapsed`） | 耗时是直接量的，可以用 |

这几行不是严格的高低：一条 trace 完全可能记了每段耗时却没有 usage 字段。所以 adapter 是**逐字段**
判断来路，不是给整份文件打一个等级。

最差的情况（只有对话）其实也不坏：几乎所有 harness 都给得出 token 流，而 token 流正是 cost model
最需要的（KV 读多少字节由 context 长度决定）。缺的是非 GPU 部分的耗时，而 tool 只占一个 episode
的 3.1%。所以只有对话加一份 config，就够把占大头的 decode 算出来，剩下的用 interval 兜住。
trace 越差，interval 越宽，但不会突然算不出来。

**我们自己的 run 有真实 timeline，但它不算「最好的 trace」。** timeline 是答案，不是输入。自家 run
的 adapter 要把同一份文件劈成两半：换个 harness 也会有的部分（token 数、prefix 结构、记下来的
耗时）进 IR；只有这套系统当时的行为才有的部分（到达时刻、`cached_tokens`、TTFT、排队时长）进
`observed.py`。劈完之后，就当 timeline 不存在。

---

## 7. Scheduler：直接跑真的 SGLang，不自己抄

**决定：** batch 怎么组、KV 怎么分、radix cache 怎么命中和 evict，都不自己写，直接跑 SGLang 0.5.8
自己的 `Scheduler`，只把真正跑模型的 worker 换成一个什么都不算的 stub。这个做法借自 simllm 的
`SimTpModelWorker`。

**为什么不照源码抄一份。** 抄看起来没有参数要拟合，可抄的过程处处是隐藏参数：page 怎么对齐、
扣下哪一页、`new_token_ratio` 什么时候衰减、chunked prefill 怎么切、retraction 先踢谁。哪一处抄错
都不报错，只会让结果差几个百分点，然后被当成 cost model 的误差去调。跑真的，这类错误就不存在了。
§3 那个 `cached_tokens` 少 113 个 token 的现象就是例子：我们原来以为是页对齐，其实是 SGLang 根本不缓存
输出。stub 化的真 scheduler 第一次跑就复现了，照源码抄的版本几乎一定会抄错。

**怎么接**（0.5.8 没有 plugin 机制，只能在进程里直接 patch）：

| 改哪里 | 怎么改 | 为什么 |
| --- | --- | --- |
| `Scheduler.init_tp_model_worker`（`scheduler.py:494`） | 子类覆盖它，构造 `StubWorker`；顺带建一次单 rank 的 gloo process group | worker 只在这里构造 |
| KV pool | 在 CPU 上建 `ReqToTokenPool` 和分页 allocator，不分配真的 KV tensor，大小取 config 里的 `max_total_num_tokens`。allocator 的 extend 用 SGLang 自带的纯 torch 版本（NPU 后端那份 `_alloc_extend_naive`），页数照样用 `get_num_new_pages` 算 | eviction 和 retraction 在真实的容量边界上发生；CUDA 版是 triton kernel，CPU 上跑不了 |
| `attention_backend` | 换成 `torch_native` | scheduler 路径上它只决定 `mem_cache/common.py` 里两个 KV 下标的小函数走 triton 还是 torch，不影响任何决定。`ServerArgs` 会按 backend 推默认值，所以 config bundle 要先用真实 backend 把 `page_size`、`chunked_prefill_size` 这些算出来，再换 |
| `forward_batch_generation` | 不算任何东西，每条请求吐一个 token。没到 `completion_tokens` 时吐一个由 (请求 id, 位置) hash 出来的 token（避开 EOS 和 stop token），到了就吐 EOS。`stop="length"` 的请求不吐 EOS，让 SGLang 自己按 cap 停 | 什么时候停来自 trace；scheduler 看到的仍然只有 cap（§4） |
| 主循环 | 不调 `event_loop_normal`（`scheduler.py:1068`，死循环 + ZMQ）。engine 自己一步一步调同样那几个函数：`process_input_requests` → `get_next_batch_to_run` → `run_batch` → cost model 给出这一步耗时、虚拟时钟往前走 → `process_batch_result`。overlap 模式下 `process_batch_result` 晚一步调，照 `event_loop_overlap` 的做法 | 时钟归 engine 管，SGLang 只负责做决定 |
| 输出 | 截住 `send_to_tokenizer`，把「哪条请求在哪一步结束」交回 engine；每一步记下 `forward_mode`、`extend_lens`、`seq_lens`、`cached_tokens` | 预测 timeline 就从这里来 |

**token id 从哪来。** radix cache 按内容匹配，不按长度，所以要从 IR 的结构造出 token id：`Prefix`
和每条请求自己新加的那段，用 hash(id, 位置) 生成，保证彼此不撞；接在上一轮后面的那段，直接拿上一轮
的 prompt id 加 stub 实际吐出的 output id，截到 `shared_prefix_tokens`。这件事在 workload 那一侧做，
SGLang 看到的只是 token 序列。

代码在 [src/simulator/tokens.py](src/simulator/tokens.py)：新的一段由 (类别, id) 当种子生成，`Prefix`
是父亲的 id 接上自己那段，请求的 prompt 是父亲内容的前 `shared_prefix_tokens` 个接上自己那段。父亲的
内容要等它跑完才知道，所以 stub 的 `Finished` 带着 SGLang 实际发回来的 output id（含 EOS），由
driver 交回来。一条请求的 id 只留到它最后一个孩子造完。

**拿 0912 step 1 验过（2026-10-07）。** 脚本 `scripts/simulator_dev/feed_workload.py`，driver 在
`src/simulator/dev/ir_driver.py`（不是 engine：每步耗时是常数，pool 只是全局计数器，router 照抄
verl）。48,258 个请求、两个 replica，一个进程 389 秒跑完，没有 retraction。逐个请求比 `cached_tokens`：

```
                       请求数    和 run 一样    差不超过一页    总命中率 stub / run
后续轮次               46,210    99.63%         100%            0.957 / 0.957
第一轮，没有共享根      2,038    63.1%                          0.719 / 0.839
第一轮，有共享根        2,038    78.1%                          0.819 / 0.839
  其中两边都命中        1,276    100%
```

- 后续轮次就是 §3 那条规则：run 里有 0.37% 多命中一整页，stub 一次都没有。原因还不知道。
- 第一轮差的那一大块是 system prompt。timeline 没有文本，adapter 不知道所有 group 共享开头的
  1,045 个 token（system prompt 加 tool schema），每个 group 的 `Prefix` 都是一整段独立的。run 里
  514 个冷的第一轮全部恰好命中 1,024 个 token，就是它按页取整。1,045 这个数是拿 run 当时的 commit
  （b2f97c8）的 harness 模板配 Qwen3 tokenizer、渲染两道不同的题、取公共前缀算出来的，没有看答案。
  `--shared-root 1045` 在所有 group 的 `Prefix` 上面挂一个共享的根。
- 加了根以后，剩下的差别都取决于时间：同一个 group 的哪一条先到哪个 replica，前一条 prefill
  完了没有。stub 里冷的第一轮 572 个，run 里 514 个；stub 有 509 个 (group, replica) 组合，多出来的
  是几乎同时到的两条都没命中。每步耗时现在是常数，这一项要等 5(d) 接上 cost model 再对。另外
  stub 里有 14 个第一轮连 system prompt 都没命中，run 里一个都没有：run 的 step 1 之前跑过
  validation，cache 已经是热的。

**SGLang 会不会偷看真实时钟？** 查过源码，scheduler 路径上会影响行为的只有三处：

- `SGLANG_QUEUED_TIMEOUT_MS`（默认关，`scheduler.py:1708`）：开着的话会按真实时间踢掉排队太久的
  请求。engine 启动时要断言它是关的。
- `empty_cache_interval`：只调 `torch.cuda.empty_cache`，不影响调度。
- radix 节点的 `last_access_time`（`time.monotonic()`）：这是 LRU 的排序依据，但 `evict_policy.py`
  只比先后、不比差值。simulator 按事件顺序调用，先后就是对的。

所以不用 patch `time`。

**router 不在 SGLang 里。** 它是 verl 的 `GlobalRequestLoadBalancer`（`src/agentic_grpo/agent_loop.py:322`）：
同一个 id 来过就发回同一台，没来过就挑当前 load 最小的那台。几十行，照抄进 engine。

**0912 的 `chunked_prefill_size` 是 16384，不是 8192。** verl 不把 `max_num_batched_tokens` 传给
SGLang，所以用的是 SGLang 在 B200 上的默认值。5(b) 的 bench 写的是 8192，那 385 秒照旧成立，但
喂真 workload 时要用 16384。

**要从 config 读的 scheduler 参数：** `schedule_policy`（我们是 `fcfs`）、`schedule_conservativeness`
（1.0）、`chunked_prefill_size`、`max_running_requests`、`page_size`（64）、`new_token_ratio` 的
初值 / 衰减 / 下限、overlap 开没开。我们的 run 没传 `disable_overlap_schedule`，按默认是开的；
要确认就看一次 server log。

**代价：**

- 版本绑定：simulator 里跑的 SGLang 必须和 config 里 server 的版本一致，否则拒绝。每升一次 SGLang，
  stub 都要重新对一遍。
- 速度：见下面「量过的」。**不许为了快换回自己抄的版本**。

**量过的（2026-10-07，代码在 `src/simulator/engine/sglang_stub.py`）：**

- **一个进程能起多个 `Scheduler`。** ZMQ socket 只 connect 不 bind，输出截住就不会发；tokenizer 用
  `skip_tokenizer_init` 跳过；torch.distributed 建一次单 rank 的 gloo group，所有 replica 共用。
  它们共用的进程级状态只有这些 group 和 `get_global_server_args()`，所以一个进程里的 replica 必须
  配置一样，配置不同就各开一个进程（构造时会拒）。验法：同一份 workload，一个 replica 单独跑，
  和另一个 replica 交替跑，每一步的 forward mode、batch 里有哪些请求、extend 长度逐步一样，overlap
  开和关都一样（`tests/test_simulator_sglang_stub.py`）。
- **stub 停得对。** 每条请求的 completion 数都等于 trace 给的数，按 cap 停的请求由 SGLang 按 length
  停；跑完 KV 守恒（free + evictable = 总量）。overlap 下有一个坑：chunked prefill 的最后一块发出去
  的时候，上一块的结果还没处理，`is_chunked` 还是 1。所以 stub 不看 `is_chunked`，看这一块有没有
  走到序列末尾。
- **一次完整模拟要多久：** 按 0912 step 1 的形状造的合成负载（2,048 条链、46,864 个请求、758 万输出
  token，分到两个 replica），一个进程跑完 385 秒墙钟，119,026 次 forward，平均每次 3.24 ms（两个
  replica 各约 6 万次，和 §7 原来估的 5.5 万对得上）。EXTEND 比 DECODE 多：每一轮请求到达都单独触发
  一次 prefill batch，这一类每次约 3.3 ms，是最贵的。所以 **N=64 撑得住**：样本之间互不相关，一个样本
  一个进程，机器有 192 核，64 个并行大约 6.5 分钟；串行是 6.8 小时。不用降 N，也不用优化 stub。
  脚本是 `scripts/simulator_dev/stub_scheduler_bench.py`。

所以 engine 走单进程：一个样本的所有 replica 放在一个进程里，由同一个虚拟时钟推进；并行发生在
样本之间，不在 replica 之间。

**engine 的时钟（2026-10-13）。** 代码在 `src/simulator/engine/engine.py`。比 5(c) 的 driver 多的是时钟：
每次 forward 多久由一个 price 函数给（cost model，或者常数），每个 replica 的时间照 SGLang 自己的 event
loop 走。CPU 时间算 0，因为 standalone server 上量过，overlap 把它盖住了（§8）。

- 不开 overlap：一轮在 T 组 batch、跑、处理结果。batch 在 T 加上它的耗时结束，跑完的请求这时离开，下一轮
  这时开始。
- 开 overlap（我们的 run 都开着）：一轮在 T 组 batch i、发出去，再处理 batch i−1 的结果，这要等 i−1 跑完。
  batch i 在 max(T, i−1 结束) 开始；i−1 里跑完的请求也在这一刻离开，下一轮也在这一刻开始。所以每个 batch
  都是上一个还在跑的时候组的：batch i 跑的时候到的请求，要到 batch i+2 才进得去，比不开 overlap 多等一个
  batch。请求自己的 token 不受影响：一条请求单独跑、写 n 个 token，两种模式都是 n 个 batch 的时间，overlap
  只是多跑一次用不上的 decode（EOS 还没回来）。
- 第一个 token 的时刻，是 prefill 完 prompt 最后一段的那次 forward 的结果被处理的时刻。SGLang 在那里打
  `prefill_finished_ts`。

5(c) 的 driver 在 overlap 下把跑完的请求晚记了一个 batch，也没有「提前一个 batch 组」这件事。不开 overlap
时两者应该一样，也确实一样：同一份 workload、常数耗时，每一步的 batch、每个请求的结果和 driver 一字
不差。overlap 那几条各有一个 test，真 scheduler 跑出来的数和上面推的一样（`tests/test_simulator_engine.py`）。

要是 run 里 decode 慢的原因是 overlap 没盖住的 CPU，就在上面两个 max 里加一项每轮的 CPU 时间。

**拿 0912 step 1 跑了一遍（2026-10-13）。** 脚本 `scripts/simulator_dev/simulate_step.py`，用 0912 的 bundle，
`--shared-root 1045`，结果在 `analysis/20260912-072938/engine/`。pool 还是所有 worker 合成一个的（container
264、eval 32），session 什么时候拿到 container、eval 排多久都和 run 不一样，所以跟时间有关的数只能当测量看。

```
                                   cost model     每步常数 44 ms     run
一次模拟的墙钟（一个进程）          464 s          450 s
后续轮次 cached_tokens 一字不差      99.63%         99.63%
第一轮没命中自己 group 的 prefix     571            573               514
TTFT p50                           0.073 s        0.108 s           0.130 s
每个请求 decode 每 token p50         0.038 s        0.072 s           0.044 s
session 结束 p50                     653 s          1,109 s           680 s
rollout 总长                        1,648 s        4,225 s           2,418 s
```

- 时钟不动结构：后续轮次的命中和 5(c) 一样。
- 第一轮冷的多出来 57 个，换成 cost model 也没变（5(c) 是 572），所以不是每步多久的问题。剩下的嫌疑是
  run 的 step 1 之前跑过 validation、cache 是热的，和同一个 group 的几条什么时候到（container 起得多快）。
- 每个请求的 decode 每 token 只比 run 快 14%（p50），不是第一级说的 1.5 倍。两边 batch 的组成不一样（模拟
  跑得快，同一时刻在跑的请求不同），这个数里还含着被别的请求 prefill 打断的时间，所以和第一级不矛盾，但
  也说明 1.5 倍得在同样的 batch 上比才看得出来，第一级就是这么比的。
- 总长 0.68 里不全是 cost model：eval 现在 32 个 slot 谁都能用，run 里每个 worker 只有 4 个，尾巴上是少数
  几个 worker 的 eval 在排队。下一步每个 worker 各一份 pool，先把这一块分出来。
- 常数 44 ms 慢 1.75 倍：44 ms 是 0912 稳态一步的中位数，拿来算尾巴的小 batch 太贵。
- 一个样本 464 秒，比 5(b) 的 385 秒多两成，是 cost model 每步算价的开销。N=64 并行约 8 分钟。

**每个 worker 各一份 pool 以后再跑（2026-10-13）。** 同一个脚本、同一份 bundle，pool 改成 8 个 worker 各
33 个 container、4 个 eval（§5），session 按 adapter 排的顺序切块。结果在
`analysis/20260912-072938/engine/step1-model-perworker.*`。

```
                                   pool 合成一个    每个 worker 一份    run
session 落在 run 的那个 worker 上                    2,048 / 2,048
后续轮次 cached_tokens 一字不差      99.63%          99.63%
第一轮没命中自己 group 的 prefix     571             509                514
decode 每 token，sim/run p10/p50/p90 0.67/0.86/1.14  0.73/0.85/1.01
等 container p50/p90/max                             438/942/1,089 s    502/1,101/1,280 s
等 eval p50/p90/max                                  42/258/790 s       17/163/755 s
session 结束，sim/run p10/p50/p90    0.21/0.92/5.57  0.87/0.99/1.32
rollout 总长                        1,648 s         2,301 s            2,418 s
一次模拟的墙钟                       464 s           368 s
```

- 切块规则把 2,048 个 session 全放回了 run 里的那个 worker。
- rollout 总长从 0.68 到 0.95。原来差的那一大截主要是 pool：32 个 eval slot 谁都能用的时候，尾巴上没人
  排队；每个 worker 只有 4 个，尾巴就是少数几个 worker 的 eval 在排队，和 run 一样。
- 第一轮冷的从 571 变成 509，run 是 514。上面猜的「同一个 group 的几条什么时候到」就是这个：session
  现在按 semaphore 放行的顺序拿 container，同一个 group 的几条差不多同时开始，谁先到哪个 replica 也就和
  run 一样了。多出来的 57 个不用 cache 热来解释了。
- 等 container 比 run 少 14%，和 decode 快 15% 对得上：前面的 session 跑得快，slot 放得早。
- 等 eval 反而是 run 的 1.57 倍（p50 42 s 对 17 s）。session 跑得快、container 放得早，到 eval 的时刻更挤
  在一起，是一种解释，还没验。
- session 结束的 p10 和 p90 从 0.21、5.57 收到 0.87、1.32：原来两头歪是因为 container 和 eval 排的队不对。
- 剩下的 5%，方向和 decode 快 15% 一致。总长对上了不等于 cost model 对了：decode 快和 eval 多等，两边
  正好抵掉一部分。在 decode 慢 1.5 倍的原因找到之前，这些仍然只是测量。

---

## 8. Cost model：每一步要多久

一个 step 是 scheduler 的一次 forward。最早这里只有一条式子，`T = (W_bytes + KV_bytes) / (BW × eta(B,
ctx, prefill_len)) + overhead`，拿 batch 大小和平均 context 去查一张 `eta` 表。2026-10-07 在 grid 上量完、
又拿 run 0912 对过答案（§12），这条式子只解释了 run 31% 的时间。现在的形式按 kernel 实际怎么跑来写，
代码在 [src/simulator/engine/cost_model.py](src/simulator/engine/cost_model.py)：

```
DECODE  T = routing(B) × base(B) + attention(ctx_1 … ctx_B)
EXTEND  T = max( floor,  a + b × 新 token 数 + c × Σ 新 token × (prefix + 新 token / 2),  d + e × 最长那条 )
```

- `base(B)`：不算读 context 的那部分，权重、MoE、其余。每个量过的 B 一个数，中间插值。
- `attention`：trtllm-gen 的 decode kernel 给每条序列、每个 KV head 分同样多的 CTA，份数是
  `max(1, SM 数 // (B × 每卡 KV head 数))`，再以 `ceil(最长 context / 256)` 为上限（flashinfer
  `fmhaKernels.cuh` 的 `numCtasPerSeqKv`）。这些 CTA 按 batch 顺序上 SM，一个 SM 一个，做完一个补一个；
  同时在跑的平分 HBM 带宽（`kv`，每 token 多少秒），但每个都快不过一个 CTA 自己的速度（`cta`，每个
  head 每 token 多少秒）。份数照 kernel 的规则算，SM 数从硬件读，不拟合。
- `routing(B)`：grid 的点不真跑 prefill，KV 是填进去的，MoE 路由和真文本不一样；这是真文本 prefill
  和填 KV 的耗时之比，在能真跑的小点上量。
- EXTEND 三项：`floor` 是 eager forward 本身（decode 走 CUDA graph，extend 一个一个 launch 约 800 个
  kernel）；中间一项是算力；最后一项是 context attention 的 critical path：新 token 少时一条序列只有
  很少几个 CTA，每个都要读完整个 prefix。
- `overhead`（scheduler 的 CPU）不在式子里：在 standalone server 上量过，overlap 调度下它被 GPU 那一步
  盖住了（见下）。

拟合的量：`base(B)` 的表、`kv`、`cta`、`routing(B)` 的表、EXTEND 的六个数。都是机器加 engine 的属性，
**按 TP 配置各拟一份**：单机内的 allreduce、kernel 没跑满、launch 开销都折在里面，TP=4 的不能拿去算
TP=8。`scripts/fit_cost_model.py` 打印对应的 `eta`（`kv` 折算的 KV 带宽、`base` 折算的权重带宽），让数
有个物理上的参照；拟合的是时间，不是 `eta`，因为 `attention` 那部分不是「字节除以带宽」能写出来的。

**只在专门设计的 grid 上测，绝不在 run trace 上拟合。** 原因是 run trace 里 batch 大小和 context
长度几乎完全反相关（`corr = -0.845`）：rollout 尾巴上还没跑完的，恰好是最长的那些 episode。

```
B   1- 16   ctx p50 51784      <- 尾巴
B  16- 48   ctx p50 28903
B  48- 80   ctx p50 21255
B  80-112   ctx p50 18671
B 112-140   ctx p50 15999      <- 平台期
```

trace 只覆盖了二维空间里的一条对角线。在这上面拟合，分不清耗时是随 batch 涨还是随 context 涨，
换一个不在对角线上的 workload 就是 extrapolate。所以 grid 要在 `batch × context × prefill 长度`
三个方向上各自独立地铺开，范围按 engine 能跑的来，不按这条 trace 碰到的来。

**grid 外面不 extrapolate。** `base(B)` 这类表只在量过的范围里插值。落在 grid 外的 step 不给估计，标出来，
并报离最近的测量点多远、在哪个方向出界。这种 step 超过一定比例（默认 5%）就整份拒绝报数。
借别的机器上的测量结果也一样，距离必须打印出来——没在这台机器上验过的 `eta` 算出来的 e2e，
跟编的没区别。

**测过的（2026-10-07）。** grid 脚本 `scripts/measure_cost_grid.py`：standalone 的 `ModelRunner`，
server args 照 run 0912 的（`trtllm_mha`、triton MoE、page 64、max running 256），其余交给 SGLang 在这块
卡上自己算默认值。每个点的 prefix KV 直接从 allocator 里分，不跑长 prefill，然后计时一次 EXTEND 和 16 步
DECODE。TP=4、2、1 各一份，在 `analysis/costgrid/b200-tp<N>/`，拟合结果在同目录的 `calibration.json`。
GPU 7 当时不在 `nvidia-smi -L` 里，TP=8 没测。

- **噪声。** 同一个 grid 跑两次（两个进程），每个点之比 p50 +2%、p90 +6%。旁边几张卡同时跑别的 grid 时
  慢了 7%，所以 grid 一次只跑一个。一次 26 万 token 的 EXTEND 让 TP=4 四个 rank 全卡死在 0% GPU，
  真 prefill 的对照点控制在 13 万 token 以内。
- **整齐的 batch，KV 是满带宽读的。** TP=4 拟出 `kv` = 3.25 ns/token，每卡 24 KiB/token，折合 7.56 TB/s。
  `eta` 低是 `base` 那部分：B=224 时权重只跑到 0.23。TP=2、1 折出来超过 8 TB/s，是短 context 的几个点
  不随 context 单调（像是 kernel 换了变体）把斜率带歪了，按 ctx≥4k 算是 7.4 TB/s 左右。
- **最长那条请求决定一步。** TP=4 下 127 条 2k 加一条 64k，一步从 7.5 ms 涨到 20.1 ms，多读的 KV 按带宽
  只该多 0.2 ms。B≥75 时每条序列只有一个 CTA，一个 CTA 只能读约 125 GB/s（`cta` = 197 ns/token，
  TP=2、1 也是 180 到 200）。上面的 `attention` 只用整齐 batch 和「一条长的」两组点拟合，lognormal 的
  90 个点完全留着检验：平均误差 4.0%，p10 -8%。只取 `max(带宽项, 最长那条项)` 的写法 p10 是 -18%：
  B 超过 SM 数时 CTA 要分波，长的那个可能排在第二波。
- **EXTEND 的下限。** 新 token 不超过 1,024 时 TP=4 约 34 ms，TP=1 约 27 ms。profile 一次 16 个 token、
  16k prefix 的 EXTEND：818 个 kernel，TP=4 下 9.5 到 27.5 ms 是 lamport allreduce 在 GPU 上自旋等别的
  rank；TP=1 没有 allreduce 也有下限，所以主要是 eager forward 本身慢（`enable_piecewise_cuda_graph`
  默认关）。prefix 长到 49k、64k 时下限也压不住了：64k prefix 上 256 个新 token 要 45 ms，按最长那条
  每 token 0.54 µs 涨。EXTEND 拟合在 435 个点上平均误差 8.6%，小 extend 自己重复测量就差 11%。
- **路由。** 一开始的对照是「真 prefill 随机 token」，它比填 KV 快 10% 到 35%，于是以为是填 KV 偏了。
  拿 030010 对话 dump 重建的真 token（`scripts/simulator_dev/dump_real_token_ids.py`）再测，真文本
  prefill、从真文本复制过来的 KV、随机填的 KV 三者相差不多，偏的是随机 token 那个参照。剩下的差别
  （真文本比填 KV 慢）TP=4 下最多 5.5%，TP=1 最多 13%，就是 `routing(B)`，三个 seed 平均。
- **scheduler 的开销。** `scripts/measure_server_overhead.py` 用 run 的参数起一个真 server，读它自己每
  40 步打的 decode 日志：B 从 1 到 224、ctx 2k 和 16k，每步只比 grid 预测的 GPU 时间多 0 到 5%，
  `return_logprob` 开不开差不到 1%。overlap 调度把 CPU 那份全盖住了，所以 `overhead` 不进式子。

---

---

## 9. 其他部分

**router**：见 §7。每个请求实际落在哪个 replica，2026-10-04 起 `generate` 事件里有 `replica` 这个字段，
030010 有，0912 和更早的 run 没有。§12 的第二级要逐请求对答案，得用带这个字段的 run。

**KV pool**：TP=4 下每个 replica 的 `max_total_num_tokens` 是 4,560,512（训练进程和它共用显存；
同样的 args 单独起 server 是 5,145,984，所以这个数要从 run 读，不能自己起一个算）。实测 KV 占用 p50 0.38、
p99 0.47，这是 simulator 要算出来的。

**非 GPU 资源**：

- worker 数、每个 worker 的 container 上限，从 recipe 读。
- 打分用的 eval container 是每个 worker 一个 semaphore，上限 4。`eval_wait` p90 248s、最长 755s，
  直接原因就是它。
- 容器启动 p50 3.3s。容器启动和 tool 耗时，我们的 trace 直接记了，别人的 trace 多半只能估。
- 两条假设：tool 耗时不随 container 并发数变化（中等风险）；worker 之间没有 CPU / network 争抢。
- 各个 pool 的排队时间是 simulator 要算的。

**编排（sync GRPO）**：batch 256 × n=8，group barrier。训练各段按给定时长当成一整块。weight sync
的语义来自 config，时长来自 trace。ramp / plateau / drain 的形状是 simulator 要算的，是第三级
对答案的主要目标。

**client 开销**：当常数。实测 client→server p50 4.3ms，server→client p50 1.9ms，加起来占一个请求的 0.2%。

---

## 10. 这条 trace 用不到、但必须能对的机制

在 SWE-bench trace 上，下面几样都能证明可以忽略。可要是真忽略了，这个 workload 的形状就焊死在
simulator 里了，而且 code review 看不出来。

跑真 scheduler 之后，前三行的机制 SGLang 本来就有，不用自己写，但**验证照样要做**。风险变成了
「stub 或 engine 不小心把它绕过去了」：KV pool 建小了、虚拟时钟没让队列真的积起来、`max_new_tokens`
被换成了答案。这些都不报错，只有专门的 workload 打得出来。

| 想省掉的 | 在这条 trace 上为什么能省 | 换什么 workload 会出事 | 用来验它的 workload |
| --- | --- | --- | --- |
| 不建等待队列 | `num_queue_reqs` 一直是 0 | 任何 open-loop 到达的负载，排队就是 latency 本身 | open-loop Poisson，短 prompt，不复用 |
| prefill 可以忽略 | cache 命中 96.2%，真要 prefill 的只有 ~450 token | 没有 prefix 复用时，prefill 变主导 | 长 prompt，只生成 1 个 token |
| 不管 KV 容量和 retraction | `num_retracted_reqs` 一直是 0，KV 占用 p50 0.38 | 长 context 高并发，throughput 会断崖 | 超长 context，顶到 KV 上限 |
| 只把 decode 做准 | decode 占一个 episode 的 94.8% | tool 很慢的 workload | closed-loop，每轮 tool 故意很慢 |

最后一列就是用来证明 simulator 没被这个 workload 带偏的 held-out set：每一条「这条 trace 上可以省」，
都配一个专门打它的 workload。另外还要跑两个：batch 大、context 短（trace 对角线从来不去的角落）；
以及 repo 里现成的两组对比（context 28672 vs 61440、只有 bash vs 加了 edit tool）。

这些机制在我们的 trace 上**一次都不触发**，这本身也是一项要对的检查。

---

## 11. 输出

使用时（没有答案），每次输出三样东西，缺一样都不算完：

1. **预测 timeline**，格式和 `analysis/<run>/timeline/*.jsonl` 完全一样（同样的 `cat` / `name` /
   `t` / `t_end` 事件）。现成的 report 工具直接能画；对答案也就是比两份同格式的文件。
2. **汇总表**：e2e 的 interval 和中位数、ramp / plateau / drain、每分钟完成多少 episode、每个
   replica 的 step 耗时分布、各个 pool 的等待时间分位数。
3. **provenance 报告**：每个参与计算的数从哪来（§6）；cost model 是本机测的还是借来的、离最近的
   测量点多远；每个估的数从哪个分布抽的；再加上 sensitivity table。没有这一份，用户分不出哪些数
   是编的。

**interval 怎么算**：对所有估的数一起抽样 N 次（默认 64），seed 固定并写进报告；取 p10–p90 当
interval，p50 当点估计。

**可复现**：同样的 IR、config 和 seed，结果必须一位不差。随机性只能来自那个 seed。

**宁可不报，也不报错的数。** 下面几种情况直接拒绝、非零退出，并写清缺什么、去哪补：

| 情况 | 为什么不能凑合 |
| --- | --- |
| trace 和 config 对不上（这份 trace 不是在这套配置下跑出来的） | 那等于在问「换了 policy 会怎样」，回答不了（§1） |
| config 里有必填项缺了 | 默认值按显存分支，照抄就是编 |
| grid 外的 step 超过阈值 | extrapolate 出来的 `eta` 没有任何保证 |
| 对答案时，被对的那个量依赖了估的数 | 那验的是自己的估计，不是模型 |
| 一个模型副本拆到多台机器上 | 跨机器通信 `eta` 测不出来（§1） |
| simulator 里的 SGLang 版本和 config 里的不一致 | 跑的是另一个 scheduler |

---

## 12. 怎么对答案（只在开发时）

要有一份 trace 配一份真实 timeline 才能做，使用时一份都没有。这三级都过了，才有资格去给没有答案
的 trace 报数。

分三级，每一级多放开一个 feedback loop。只看最外层对不上时，你不知道错在哪，所以要有里面两级。

- **第一级：只验 cost model。** 输入真实的每秒 `(batch 大小, 总 context 长度)` 序列，预测每一步
  耗时，跟真实的 `batch / gen_throughput` 比。不需要知道请求落在哪个 replica，现有数据就能做。
- **第二级：验 scheduler。** 输入真实的请求到达时刻和每个请求的 token 数，simulator 自己组 batch、
  自己决定什么时候完成。**逐请求**比 TTFT 和 decode 时长，共 99,646 个请求。前提是先补上
  「请求落在哪个 replica」。
- **第三级：全闭环。** 只给 episode 什么时候放出来，其余全靠 simulator。比整体形状：ramp 365s /
  plateau 1238s / drain 463s、每分钟完成的 episode 数、生成阶段总时长、一步的 wall clock、每秒的
  并发曲线。

**误差会不会滚雪球。** 在第二、三级加一个开关：每隔 k 轮把时间对齐回真实值一次，画误差随 k 的
变化。k=1 准、k=∞ 不准，说明一个小偏差在 feedback loop 里被放大了；k=1 就不准，说明单步模型本身错了。

**容差从 noise floor 来，不凭感觉。** 我们的 run 有两个独立的 replica，负载统计上一样，两者之间的
差就是去不掉的噪声（2,862 个两边都忙的采样点）：

```
num_running_reqs   平均相对差    4.0%   (corr 0.979)
gen_throughput     平均相对差   16.8%
step 耗时中位数    44.5ms vs 43.3ms -> 2.7%
```

所以按秒采样的曲线上，要求比 ~17% 更准没有意义；中位数上 ~3% 就是底。验收标准定在这里。

**第一级做过了（2026-10-07 做，2026-10-13 改正）。** 脚本 `scripts/simulator_dev/validate_cost_model.py`，
run 0912 step 1 和 030010 step 1–3。不是模拟：排程用真实的，只有每次 forward 的耗时是 cost model 给的。

每条请求落在哪个 replica：030010 记了（`generate` 事件 2026-10-04 起带 `replica`），直接用，running 数和
srv-metrics 比中位数差 1.1%、p90 差 2.8%。0912 没记，但 verl 的 router 是确定的（同一个 session 回原来
那台，否则挑 in-flight 最少的），照 timeline 里请求开始和结束的先后重放出来，中位数差 3.1%，p90 差 26%。

然后在两次 srv 采样之间（约 1 秒）把这台 replica 实际做的活算一遍价：哪些请求在 decode、各自 context
多长、出了多少 token，以及哪些 EXTEND（按 `chunked_prefill_size` 切块）。算出来的时间除以窗口的实际长度。

**哪些请求是同一次 EXTEND。** SGLang 在处理结果的循环里一条一条地打 `prefill_finished_ts`，中间夹着
radix insert，同一次 forward 的几条请求时间戳能差出 10 ms。原来按 `decode_start` 取到毫秒分组，一次
forward 被拆成好几批，每批都付一次 34 ms 的下限。030010 相邻时间戳的间隔分成干净的两团：0.1 到 12 ms，
然后直到 34 ms 才又有，中间几乎是空的。所以现在间隔小于 20 ms 的连成一次 forward（`--same-forward-ms`）。
原来那份 1.00 就是拆批拆出来的。

```
priced / actual 的中位数        稳态（B≥100）              尾巴（B<100）
0912 step 1，原来的分法          1.00                       0.72
0912 step 1                     0.75（门槛 12 ms 时 0.79）  0.69
030010 step 1 / 2 / 3           1.09 / 1.09 / 1.06          0.66 / 0.76 / 0.65

0912 稳态：时间里 decode 0.42、extend 0.33；B 122，ctx 均值 16.9k、最长 54.6k，每秒 8.8 次 extend（原来数成 16.6）
0912 尾巴：decode 0.52、extend 0.13；B 17，ctx 均值 42.7k
030010 稳态：decode 0.22、extend 0.84；B 215
```

- **0912 稳态没对上，只解释了 75% 到 79%。** 原来 extend 多算了将近一倍，正好填上 decode 少算的那块。
  所以「server 59% 的时间在做 EXTEND」也不对，是 33% 到 36%。给 extend 开 piecewise CUDA graph 还是
  值得试，收益要按这个数估。门槛从 1 ms 往上，结果是 0.97、0.86、0.79，到 12 ms 以上才稳。0912 的
  replica 是重放的，门槛越大越容易把另一台的请求并进来，所以报 0.75 到 0.79 这一段。
- **尾巴的差距跟 router 重放没关系。** 030010 只有一个 replica，routing 一点都不用猜，尾巴照样是 0.65 到 0.76。
- **两份 run 都指向 decode 少算了。** grid 上 8k 到 16k 新 token 的 extend 误差在 2% 以内，假设 extend 的价
  是对的：0912 稳态里 decode 实际约占 1 − 0.36 = 0.64，模型给 0.42；尾巴里实际约 0.87，模型给 0.52。两处
  都是 run 里每步 decode 比 grid 慢 1.5 到 1.65 倍。可用同样的 args 单独起 server，decode 每步只比 grid 多
  0 到 5%（§8）。所以是 run 里有什么东西让 decode 变慢，cost model 在 standalone 上是对的。配置查过：
  rollout 的 `enforce_eager` 是 false（config 里两处 true 在 reward 段），CUDA graph 开着。剩下的嫌疑都是
  standalone 那次没有的：机器上几十个 container 和 eval 抢 CPU，B 小时一步 GPU 只有 5 ms，overlap 盖不住
  scheduler 的 CPU；verl 开的 `enable_memory_saver`；prometheus middleware 和每个 batch 的 stats loop。
  验法是在 standalone server 上把这几样一样一样加回去，看 decode 每步多出多少。要 GPU，不用跑训练。
- **0912 的 attention backend 没法确定。** 写 0912 的 bundle 时发现：run 当时的 commit（b2f97c8）里
  config 写的是 `attention_backend: flashinfer`，没写 `page_size`，这样 SGLang 的 page 是 1；可 run 的
  `cached_tokens` 是按 64 取整的，所以 server 实际用的不是 commit 里那份。`trtllm_mha` 是 2026-09-15
  （eca206f）才写进 config 的，grid 和 standalone server 都用它。要是 0912 的 decode 其实走的是 flashinfer
  的 kernel，1.5 倍里就有一部分是 calibration 配错了 server，不是 run 里多了什么。要查：0912 有没有留下
  server 的启动 log。030010 确定是 `trtllm_mha`，它的尾巴也差，但它的 MoE 又和 grid 不一样。
- **030010 稳态多算的 6% 到 9% 不作数。** 它跑在另一种状态里：一个 replica 扛 2,048 个 session，prefix
  cache 装不下，51% 的后续轮次被 evict（中位数丢 10k token），三个 step 重新 prefill 了约 10 亿 token
  （0912 是 6,200 万），稳态时间八成在 extend 上。它的 MoE 又是 `flashinfer_trtllm`，grid 量的是 `triton`，
  calibration 本来就对不上。它只能用来证明尾巴那一条。
- **想直接从 run 里量出一次 extend 多贵，没量成。** 办法是看已经在排队、紧跟着上一次 forward 跑完的
  extend 隔了多久。小 extend 隔 39 到 54 ms（价是 34 ms），中间可能夹了一次 decode；大的被 chunked
  prefill 切开，前几块在别的 forward 里跑过了，间隔偏短。这要等 scheduler 的 decode 日志。

**每一步的 latency 目前没有真值可对。** 存档的 trace 最细只到每个请求。SGLang 其实每隔
`decode_log_interval` 步会打印 batch 组成和 throughput，但写在 Ray actor 的 log 里，run
20260912-072938 的 Ray session 已经没了。把这几行 log 存进 timeline 目录几乎没成本，也是验证每步
预测的唯一办法。

---

## 13. 代码分层和 test 守的规矩

| 层 | 在哪 | 允许知道什么 | 允许 import |
| --- | --- | --- | --- |
| adapter | `src/simulator/adapters/`：接口在 `base.py`；只有一个 adapter（`generic.py`），`shapes.py` 认 trace 的长相，`trace.py` 逐字段判断来路；`agentic_messages.py` 是自家 harness 的 profile；`priors.py` 读默认 prior 和 config bundle | trace 的格式；逐字段判断来路 | `simulator.ir`、`simulator.observed` |
| IR | `src/simulator/ir.py` | 只有 workload | — |
| realise | `src/simulator/realise.py` | IR，加一个 seed | `simulator.ir` |
| token id | `src/simulator/tokens.py`：从 IR 的结构造 token id（§7） | 抽样过的 IR，加上 stub 吐出的 output id | `simulator.ir` |
| observed | `src/simulator/observed.py` | 真实系统的结果，也就是答案 | `simulator.ir`（只借 id 类型） |
| config bundle | `src/simulator/bundle.py`：读 bundle、查齐全、给请求填 cap（§5） | 系统配置 | `simulator.ir` |
| engine | `src/simulator/engine/`：`sglang_stub.py`（stub 化的真 scheduler，§7）、`cost_model.py`（§8，含拟合）、`router.py`（照抄 verl）、`engine.py`（虚拟时钟、每个 worker 各一份的资源池）、`system.py`（从 bundle 建出 replica、worker 和 pool、cost model）；group barrier、训练段待建 | 抽样过的 IR + config | `simulator.ir`、`simulator.bundle`、SGLang |
| 开发专用 | `src/simulator/dev/`：`agentic_timeline.py` 读自家 timeline（也是 adapter，守 adapter 那一行的 import 规矩），`priors.py` 从自家 run 量 prior、写 `default_priors.json`，`stub_driver.py` 是 5(a)/(b) 用的合成闭环负载，`ir_driver.py` 是 5(c) 用的、把真 IR 喂进 stub 的闭环；脚本在 `scripts/simulator_dev/` | 自家 run，连同答案 | 全部 |
| 对答案 | 待建，放进 `src/simulator/dev/` | 两边都知道 | 全部 |

规矩：

1. **engine 不许 import `simulator.observed`；任何 simulator 模块不许 import `agentic_grpo`。**
   这两条已经有 test 守着（[tests/test_simulator_ir.py](tests/test_simulator_ir.py)）。
2. **engine 里不许出现 `episode`、`turn`、`tool`、`swebench` 这几个词。** engine 和 workload 之间
   只有一个请求流：每个到达事件带着 `(prompt_tokens, prefix_parent, shared_len, completion_tokens)`。
   「第 k+1 轮要等第 k 轮的 tool 跑完」这种逻辑属于 workload 那一侧，到 engine 这里只剩一个到达时刻。
   engine 分不出输入是谁生成的，就不会偏向谁。
3. **关于 trace 格式的知识只能待在 adapter 里。** 进来的 trace 本来就来自不认识的 harness。
4. **engine 不许根据 `Task.kind` 或 provenance 做分支。** `kind` 只是报表上的标签（trace 里的
   `bash` / `edit` / `submit`，进 IR 后都是 `"tool"`）。provenance 这一半由 `realise` 兜住：
   engine 只收抽样过的 workload，`require_runnable` 在门口拦下还带着分布的字段。
5. **cost model 不许在 run trace 上拟合**（§8）。
6. **没有答案也必须能跑完。** 使用时没有 timeline，任何「需要 `observed` 才走得通」的代码路径，
   在真正该用的场合就是死路，而且开发时永远发现不了。检查办法：拿一份只有 trace、没有 timeline
   目录的输入跑一遍完整流程，`Observations` 全空也要得出数。
7. **使用时的代码不许 import `simulator.dev`。** 使用时的代码指 `src/simulator/` 下 `dev/` 以外的
   所有文件，加上 `scripts/` 顶层的脚本。`dev` 里的东西只在我们自己的 run 上走得通，使用时调到它
   就是第 6 条那种死路。默认的 `ADAPTERS` 里也没有读自家 timeline 的那个 adapter，开发时要显式传。
   有 test 守着（[tests/test_simulator_ir.py](tests/test_simulator_ir.py)）。

---

## 14. 假设

| 假设 | 风险 | 怎么证伪 |
| --- | --- | --- |
| trace 和 config 是一套：这份 token 流就是在这份 config 下跑出来的 | **高** | 比对 turn cap、模型名、工具集、prompt 模板指纹。对不上就拒绝报数 |
| 时间戳相减的耗时可以当活本身的耗时 | **高** | 同一个 harness 空载重跑一遍比分布；当场能做的是把 replica 数翻倍，看 e2e 动不动 |
| 只有对话时，tokenizer + 模板重建的 token 数等于当时真实发出的 | **已量**（run 20261005-113503，144 条 episode）：prompt 93% 一字不差，completion 98.7%，其余差 +1 到 +9 个 token，来自模型采样出非标准的切分，重编码补不回来。另外第一轮 prompt 超过 `prompt_length`（这里 4096）会被从头截掉，文本里看不出来，所以这个数必须从 config bundle 拿 | `scripts/simulator_dev/check_messages_reconstruction.py` |
| tool 耗时不随 container 并发数变化 | **中** | 比较 cap 16 和 cap 40 两个 run 的 tool 耗时分布 |
| 下一轮 prompt 就是上一轮整段再加新内容 | **已验证**：95,403 对里 0 例外 | — |
| stub 吐什么 token 不影响调度，SGLang 只看长度和 id 是否相同 | 低 | 用两个不同 seed 的 hash 跑同一份 IR，每一步的记录必须完全一样 |
| scheduler 不靠真实时钟做决定 | **已查源码**（§7） | queued timeout 开着就不成立，engine 启动时断言 |
| 单机：TP 的 allreduce 可以折进 `eta` | 低（这台机器） | 跨机器就拒绝（§1），而不是放宽这条 |
| worker 之间没有 CPU / network 争抢 | 低 | |
| client 和 server 之间的开销是常数 | 低 | 已量，占一个请求的 0.2% |
| 两个 replica 可以互换 | 低 | 已量，就是 §12 的 noise floor |
| 没有失败和重试 | 低 | |

---

## 15. 做的顺序

每一步都能单独交付、单独验。用 ML 的话说：第 2 步的测量 grid 是 training set，也是整个 simulator
唯一拟合参数的地方；自家 timeline 是 validation set；第 6 步的合成 workload 是第二个 held-out set，
专门证明没被带偏。我们的 trace 在第 1 步只当格式样本看，第一次参与计算是第 3 步，当 validation set
用，永远不碰参数。

1. **adapter。** 陌生 trace 进来，逐字段判断来路，输出 IR 和 provenance。用户真正碰到的就是这一层，
   所以排第一。交付：一条只有对话的 trace 能造出合法的 workload，每个编出来的字段都带标记。IR
   这边的准备已经做完了（§4：`prov`、`unknown`、`realise`）。别照着自家 trace 写一个
   `trace_to_ir.py`，也别给每种 trace 各写一个 adapter：使用时进来的 trace 来自没见过的 harness，
   专门为某种格式写的代码到时候一行都用不上。

   所以只有一个 adapter，分两层。下层认长相：一行是一串 chat message、一组带 `gen_ai.*` 的 OTel
   span、一串每轮的 token 数，还是一行就是一个请求（serving 日志，CSV 也行；有 conversation id 就按
   它拼成多轮）。列名不分大小写和空格，`Request tokens`、`ContextToken`、`input_length` 是同一个东西。认的是通用写法（`prompt_tokens` / `input_tokens` / `isl` 这些常见
   拼法、OpenAI 和 Anthropic 的 usage），不是数据集的名字。读出来是同一种每轮记录，没记的字段就是空。
   上层对每个 IR 字段走同一条梯子：

   | 字段 | 梯子 |
   | --- | --- |
   | `prompt_tokens` | usage 或 count（traced）→ tokenizer 配模板渲染消息（derived）→ 下一轮的 cache read 按这份 trace 自己量出的偏移倒推（derived）→ 拒 |
   | `completion_tokens` | usage 或 count（traced）→ tokenizer 数输出文本（derived）→ 拒 |
   | prefix parent | token 级的 loop：上一轮整段 → 有消息：这一轮的消息接在哪一轮后面 → 只有 count：prompt 够长就当它接在上一轮整段后面，并计数 → 有块哈希（Mooncake 的 `hash_ids`）：跟别的请求从头开始有几块一样，就挂在那一段共享的 `Prefix` 上，取共享更多的那个 |
   | 每轮几次 tool | 回应它的 tool 消息（回应它的是 user 消息就是 0 次）→ 输出里写的 call → 什么都不放，写进 note；profile 可以设 `calls_per_turn` 假设几次 |
   | tool 耗时 | trace 自己记的（traced）→ 上一个回复到下一个请求的间隔（derived，里面有 harness 时间，只当 upper bound）→ prior |

   块哈希先过一道检查：块大小取让 `ceil(长度 / 块大小) = 哈希个数` 在 99% 以上请求上成立的那个，
   长度优先用 trace 单独记的「哈希覆盖的长度」，没有就用 prompt。对不上就不用哈希。哈希串不串联都行：
   比的是从头开始的哈希序列，Mooncake 那种串起来的只是特例。

   时间戳先过一道检查：请求的时长中位数是 0、或者有倒着走的，说明时钟是事后重建的，间隔不用。
   间隔为负（请求重叠）的那一段也退回 prior。

   一个 harness 的语义不是格式：loop 是拼 token 还是每轮重新渲染、哪些 tool 消息是 observation、
   哪些是 nudge、episode 外面还包着 container 和打分。这些写成 profile，是一份数据，不是另一个
   reader。自家对话 dump 就是一份 profile（`agentic_messages.py`），在 run 20261005-113503 的
   2,483 条 episode 上，和改造前的专用 adapter 逐字段一样，顺序也一样。

   接口在 [src/simulator/adapters/base.py](src/simulator/adapters/base.py)。每个 adapter
   两个方法：`claims(path)` 说这份文件我认不认识，`convert(path, priors)` 读它。读完交出三样：IR、
   observed、几句话说明哪些字段没找到。`load` 挑出唯一认领的那个 adapter，跑完再查四条：IR 合法；
   两边的 `source` 一样；observed 里每个 id 都能在 IR 里找到对应的 request 或 session；每个「估的」
   字段用的都是调用方给的 prior，而且那个 prior 不是在正在转的这份 trace 上量的。最后一条是为了
   下面的削 trace 实验：删掉一个字段再用同一份 trace 的分布填回去，实验就白做了。

   验收这一层只有一个办法有说服力：**把自家最全的那条 trace 一级一级削**。先删每段活自己记的
   耗时，再删 usage 字段，再删时间戳，最后只剩对话。每一级跑一遍 adapter，看三件事：还能不能出
   合法 workload；interval 是不是一级比一级宽；每一级的 interval 盖不盖得住信息最全那一级的点
   估计。这是唯一能把「为了通用牺牲了多少精度」量出来的实验，而且只要一条 trace 就能做。
2. **测量 grid。** 在 standalone server 上测出 `eta`。顺带把 3.7x 的 gap 变成一张按
   `(batch, context)` 排的残差表。
3. **第一级对答案。** 单独接受或否掉 cost model。
4. **补 instrumentation。** `generate` 事件加上 replica id（2026-10-04 加了）；确认 overlap 开没开；把
   scheduler 的 decode batch log 存档（还没做）。
5. **接上真 scheduler，再搭 engine**（§7），按这个顺序：
   - (a) 一个进程里能起几个 stub 化的 `Scheduler`，起不了就走多进程；
   - (b) 量一次完整模拟要多久，定下 N；
   - (c) 从 IR 造出 token id，包括 `Prefix`；
   - (d) engine：虚拟时钟、router、资源池、group barrier、训练段。

   §10 那四样机制 SGLang 本来就有，这一步要证明的是 stub 没把它们绕过去。
6. **合成的 held-out workload。** 证明没被带偏（§10）。
7. **第二、三级对答案。** 到这里 simulator 才算验过。
8. **sensitivity table 和带 interval 的 e2e。** 这一步才是交付物，也是第一个完全没有答案的步骤：
   吃一条陌生 trace 加一份 config，出一个带 interval 的 e2e，外加一张表说明哪个估的数撑开了
   interval、去补哪项 instrumentation 最划算。前七步是拿答案换信心，到这一步没答案可换了，靠的
   全是前七步攒下的。

**现在到哪了（2026-10-07）：** 第 1 步做完了。5(a)、5(b) 提到 grid 前面先做了，因为它们不碰 `eta`，
又决定 engine 是单进程还是多进程：一个进程能起多个 stub 化的 `Scheduler`，互不干扰；一次完整模拟
385 秒，N=64 并行约 6.5 分钟（§7「量过的」）。顺带发现 SGLang 不缓存输出（§3）。

5(c) 也做完了：从 IR 造 token id，0912 step 1 的后续轮次 `cached_tokens` 和 run 99.63% 一样（§7）。
第 2 步做完了：grid（TP=4、2、1）、按 kernel 行为改写的式子、拟合（§8）；TP=8 的 grid 2026-10-13 开始测。
第 3 步，第一级对答案，2026-10-13 改正以后：0912 step 1 稳态只解释了 75% 到 79%，尾巴 69%；030010 有记下
的 replica，尾巴照样差，所以不是 router 重放的问题（§12）。两处都指向 run 里的 decode 比 standalone 慢
1.5 倍上下。第 4 步里 replica id 早就加了（2026-10-04），还差 decode 日志存档。下一步：在 standalone server
上把 run 才有的东西（抢 CPU、`enable_memory_saver`、stats loop）一样一样加回去，找出 decode 慢在哪；
然后 5(d) 的 engine。

5(d) 先开工了（2026-10-13），因为 8 张卡凑不齐，而 engine 只用 CPU。按这个顺序：时钟、config bundle、每个
worker 各自的 pool、group barrier 和训练段、输出和 subagent。前三样做完了（§5、§7）：每个 worker 各一份
pool 以后，0912 step 1 的 rollout 总长从 run 的 0.68 到 0.95。decode 慢 1.5 倍的原因
找到之前，engine 的时间对答案只能当测量看，不当验收。下面是第 1 步的经过。

schema 加抽样那一步——`src/simulator/ir.py`、
`src/simulator/realise.py`、`src/simulator/observed.py`、`tests/test_simulator_ir.py`。IR 现在能
说「不知道」，能写并行 tool call，能写 subagent，`max_new_tokens` 可以空着让 config 填（§4）。
第 1 步最早的样子：接口在 `adapters/base.py`，读自家 timeline 的 adapter 在
`dev/agentic_timeline.py`，一份 workload 对应一个训练 step。在 run 20260912-072938 的 step 1 上
跑过：2,048 个 session、256 个 group 全部合法，1.8 秒转完；它算出的 structural share 减去 observed
里的 `cached_tokens`，p50 是 114，第一轮的命中率 p10 0.396、p50 0.984，和 §3、§4 的数字对得上。
削 trace 实验已经整条跑完，脚本是 `scripts/simulator_dev/degrade_trace.py`，数据是 run 20261006-030010 的 step 1
（timeline 和对话全文都有，2,037 条 episode；还有 11 条的 event 在 TimelineWriter 的 400 行 buffer 里没
落盘），prior 取 run 20261005-113503 的 step 1，scale 用 20261005-143426 和 20260912-072938 每个 knob 的
均值相对 prior 的比值撑开。结果：

- 全量 timeline、删掉每段活的耗时、只剩对话，这三级都出合法 workload。中间两级（再删 usage、再删
  时间戳）被 adapter 拒掉：token 数估不了，因为相邻 turn 的 prompt 是绑在一起的；没有 trainer 的 gen
  span 就放不进 step。拒绝是如实的，不是算出一堆错数。
- 只剩对话这一级，DAG 形状 2,037 条全部和 timeline 一样，completion 97.8% 一字不差、总量差 0.01%，
  prompt 88% 一字不差（差的全是 completion 误差的累加），第一轮 prompt 和 group prefix 全对。文本
  读不出来的还是那几样：属于哪个 step（对话 dump 里有 2,208 条，160 条 validation 得靠 timeline 才
  剔得掉）、cap、停止原因、tool 有没有超时；observed 那半边是空的，所以这一级没法自己对答案。
- interval 的检查没有全过。带 scale 时 task 总耗时、eval、per-session p90 盖住了全量那级的点，但
  tool 和 container_start 的 interval 整个在点的上方：这次 run 的 tool 和 eval 比之前三个 run 都快，
  container_start 和最快的那个持平（均值比 0.78 到 0.97），而 scale 是拿那几个 run 撑出来的，下沿
  要么没够到要么正卡在边上。不带 scale 的 control 一个总量都盖不住，证实了只靠逐节点抽样的
  interval 不是预测。
- 删掉耗时那级和只剩对话那级的 interval 一样宽：文本把结构读得一字不差，所以再往下削掉的不是
  精度，是 step 归属、答案和 timeout 标记。
- 逐节点独立抽样丢掉了 session 内部的相关性（同一个 instance 的 eval 和 tool 是一起长一起短的），
  所以总量盖住了，per-session 的 p50 还是偏低、max 偏高。

030010 跑完三个 step 之后又跑了一遍（2026-10-06），step 1、2、3 各一次，prior 不变，scale 分两组：
原来那两个 run，和再加上 0903、0906、0908 两个 run 一共六个。report 在
`analysis/20261006-030010/degrade/step<N>/scale-<2|6>runs/report.md`；最早那份 2,037 条的挪到了
`step1-partial-2037/`，step 1 现在是完整的 2,048 条。

- 结构那一半三个 step 结论一样：中间两级照样拒，只剩对话那级 DAG 形状 2,048/2,048 全对，completion
  97.6% 到 97.9% 一字不差、总量差 0.011%，prompt 88% 左右一字不差，第一轮 prompt 和 group prefix 全对。
  对话 dump 一共 6,464 条（三个 step 各 2,048，加 320 条 validation），文本自己分不出 step。
- 六个 run 撑出来的 scale 在 step 1 上把每个总量都盖住了，tool 和 container_start 原来没盖住的这回
  也盖住了：这两个是 run 级别的旋钮（机器、docker 快不快），多几个 run 就够。
- step 2、3 上 eval 没盖住，六个 run 也不行：interval 整个在点的下方（step 3 差 9% 到 25%）。030010 的
  eval 均值从 step 1 到 step 3 是 19.1、21.9、23.3 秒，step 3 比任何一个 run 任何一个 step 都高。原因不是
  run 变慢了，也不是 repo 比例变了（按 repo 加权 prior 只预测出 21.2、21.9）。同一个 instance 去掉超过
  120 秒的 eval 之后，step 1 到 step 3 的比值是 0.98，同时跑的 eval 多几个也不影响耗时。多出来的
  全在 tail：每个 step 有 19 到 28 个 eval 跑 300 到 900 秒，只占 1%，却占了 eval 总秒数的 17% 到
  24%，集中在少数几个 instance 上（psf__requests-2317、sphinx-doc__sphinx-7985/8475、
  sympy__sympy-15875、matplotlib__matplotlib-24149/24177）。
- 所以 scale 量错了东西：每个 run 的 step 1 用的是同一组 256 个 instance，跨 run 的比值只量到同一批
  task 上的噪声。step 2 和 step 1 只有 145 个 instance 重合，换一批 task，eval 总量能差 20% 以上，
  而这个差是 task 带来的，按均值乘一个系数表示不出来。prior 现在不认 instance，同一个 instance 的
  8 条 rollout 当成 8 次独立抽样，长 eval 的 instance 在不在这一批就丢了。

按这个结论改了 prior（2026-10-06）。IR 加了 `Session.problem`，两个 adapter 都把 instance id 填进去。
eval 的 prior 按 instance 查表。梯子用的表来自 prior run 113503 的 step 1，加上 143426、0912、0903、0906、
0908 两个 run 的 12 个 step；scale 用这 12 个 step 量（§5 的做法，只是 030010 不在里面）。默认 prior 也按同样的方法重建了。再跑一遍梯子，三组 prior 对比，
report 在 `step<N>/pooled-crossstep/` 和 `step<N>/keyed-crossstep/`：

| R1 的 interval 盖不盖得住 R0 的点 | step 1 | step 2 | step 3 |
| --- | --- | --- | --- |
| 六个 run 的 step 1，不分 instance（上面那组） | eval、总量都盖住 | eval 低 3% 到 17%，没盖住 | eval 低 9% 到 25%，没盖住 |
| 跨 step 量 scale，不分 instance | 盖住 | 低 4% 到 20%，没盖住 | 低 10% 到 27%，没盖住 |
| 跨 step 量 scale，eval 按 instance 查表 | 盖住 | 盖住（-13% 到 +4%） | 差 1% 没盖住（-13% 到 -1%）；R4 盖住 |

- 光把 scale 跨 step 量没用，scale 只是往下多撑了一点。起作用的是按 instance 查表：step 2 盖住了，
  step 3 只差 1%。
- 查表以后，030010 三个 step 的 eval 实测和预测之比是 0.928、1.039、1.077，都在 scale [0.885, 1.103]
  里面。step 3 落在 scale 的上沿附近，p10 到 p90 只取中间 80%，所以差一点。剩下的误差主要是一个
  instance：psf__requests-2317 的 eval 要么 340 秒左右、要么 770 秒左右，看 patch 是什么样。别的
  run 两种都有，030010 几乎每次都是 770 秒，一个 step 就多出 3,000 多秒。这只能靠 scale 兜，prior
  预测不了。
- per-session 的 p50 第一次三个 step 都盖住了。原来偏低是因为逐节点独立抽样丢了 session 内部的相关性，
  现在看来这个相关性大半就是 instance 本身。
- 只剩对话那级（R4）也用上了表：step 1、2 的 2,048 个 eval 全部查到了自己 instance 的值，step 3 是
  1,976 个。eval 和总量三个 step 都盖住了。

中间两级的混合 adapter 仍然不做。config bundle 后来补齐了（§5）。

2026-10-07 拿 HuggingFace 上的别人的 trace 试了这个通用 adapter，没有为它们改一行格式代码，都用默认的
`load` 读进来，`check` 和 `realise` 都过：

| trace | 长相 | 进来的样子 |
| --- | --- | --- |
| `netpreme/coding_agent_traces` opus-pro（Claude Code 跑 SWE-bench Pro，731 条） | 每轮只有 `isl / osl / isl_new` | token 数 traced；第一轮全都缺 `isl`，按下一轮的 cache read 倒推（这份 trace 上偏移是 -1，p1 到 p99 在 -6 到 -1 之间）；4% 的轮次 prompt 变短，只共享上一轮的 prompt；两轮之间发生了什么 trace 没说，默认什么都不放，要估 tool 就用 `calls_per_turn=1` 的 profile |
| 同上 gpt-oss-120b-verified 的一个分片（72 条，vLLM） | count 加文本 | tool 次数从输出文本里的 `[tool_use:` 数出来；第一轮是冷的，cache 偏移很散（p99 619），所以 72 个第一轮里有 32 个落到了 `isl_new` 下限 |
| `Exgentic/agent-llm-traces-v2` 的 SWE-bench 部分（1,959 条，5 个模型 × 4 个 harness） | OTel span：时间戳、usage、全部消息 | token 数全部 traced；prefix parent 靠比消息找；973 条的时钟是重建的（opus、gemini、gpt-5.2 的请求时长都是 0），间隔不用；能用的间隔 47,872 段，标 derived |

后来又加了几份普通 inference 的 trace，单轮的、多轮聊天的都有。为它们加了一种通用的长相（一行一个请求）和块哈希的规则，没有哪一行是为某个数据集写的：

| trace | 长相 | 进来的样子 |
| --- | --- | --- |
| `lzzmm/BurstGPT`（Azure OpenAI 的 serving 日志，CSV，140 万行） | 每行一个请求，只有 token 数 | 140 万个单请求 session，token 数 traced，没有共享结构可读 |
| `valeriol29/mooncake-traces` 四份（mooncake、conversation、toolagent、synthetic） | 每行一个请求，带 512 token 一块的 `hash_ids` | 块大小 100% 推得出来；结构复用 53% 到 80%，靠哈希挂到共享 `Prefix` 上 |
| `eth-easl/swissai-serving-trace` `data.jsonl`（52 万行） | 每行一个请求，有开始和结束时间 | 6,248 行 token 数是负的，当坏记录跳过；11.6 万行失败请求没有 token 数，也跳过 |
| 同上 `qwen3-32b-buckets.jsonl` 的前 80MB（10 万行） | 每行一个请求，带 16 token 一块的 `bucket_ids` | 哈希不串联，而且覆盖的是 `token_count`（比 prompt 少 3 个模板 token）；按从头比的规则读出 73.5% 的结构复用 |
| `ST-TraceWeaver/LLM-Inference-Traces-NYC`（合成的，取 2 万个对话） | 每行一轮，靠 `ConversationID` 和 `Turn` 拼成多轮 | 34,893 轮，token 数 traced，没有一轮的 prompt 比上一轮整段短 |
| `allenai/WildChat-1M` 一个分片、`ShareGPT_V3`（各取 3,000 个对话） | 只有对话 | 用 Qwen3 的 tokenizer 和它自带的模板数 token，note 里写明 trace 没说用的哪个模板；prefix parent 全部从消息上找到；user 的思考时间不建模 |

最开始那几份 agent trace 都是评测跑出来的，每道题一条，没有 group，也没有打分和 container 的事件，所以 session 里只有
请求和 tool。Claude Code 会在主对话旁边插几条独立的请求，现在按开始时间串成一条链，跟主链重叠的
那几段间隔是负的，退回了 prior。

---

## 16. 数据出处

上面所有实测数字都来自 run `20260912-072938`（Qwen3-30B-A3B，TP=4 × 2 个 replica，8× B200），
算自 `/data0/shared/kewen.liu/agentic-timelines/20260912-072938/`：

- `timeline-13267*.jsonl`：99,646 个 `generate` 事件，95,403 对相邻 turn。第一轮的 prefix 数字（§4）
  来自其中 `turn == 1` 的 4,243 个事件。
- `srv-metrics-1312837.jsonl`：7,193 个有 running batch 的采样点。
- 一步的时间分配来自 `cat="train"` 的 span：step 5072s / 4377s，`gen` 2418s / 1824s，
  `update_actor` 1796s / 1718s，`update_weights` 59s / 60s。

`eta ≈ 0.12` 是推出来的，不是直接测的：每个 replica 1.84M token 的 KV，每 token 96KB，分到 4 张卡上
每张约 43GB，按 8TB/s 读要 5.4ms，实测一步 44.5ms。

顺手查清的一件事：`is_cuda_graph` 这个 gauge 在 7,193 个采样里全是 0，但 SGLang 0.5.8 里根本没有
地方给 `stats.is_cuda_graph` 赋值——它是个死 gauge，不是证据。CUDA graph 是开着的，5.4ms 和 44.5ms
之间那 8 倍的差距不是它造成的。
