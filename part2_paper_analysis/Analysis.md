Efficient Memory Management for Large Language Model Serving with PagedAttention

Conference: SOSP 2023
Authors: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica

1. Summary (1 Page)

What problem does the paper address?

The paper addresses the significant problem of inefficient GPU memory utilization during the serving (inference) of Large Language Models (LLMs), particularly within the Key-Value (KV) cache used by the Transformer's self-attention mechanism. The KV cache is crucial for autoregressive generation but dynamically grows and shrinks, and existing serving systems struggle to manage this dynamic memory efficiently, leading to high memory fragmentation and redundant duplication. This inefficiency drastically limits the maximum batch size of concurrent requests, making LLM serving expensive and throughput-limited.

Why is this problem important?

The problem is critical because LLMs are computationally expensive, and serving costs are a major bottleneck for widespread commercial and research deployment. Throughput is proportional to the number of requests that can be batched together. If memory is wasted, the batch size is limited, and the expensive GPU compute resources are underutilized (the workload becomes memory-bound). As GPU compute power grows faster than GPU memory capacity, this memory bottleneck is becoming increasingly severe. Improving memory efficiency directly translates to 2-4x higher throughput, significantly reducing the cost per request.

What is the key insight or main contribution?

The key insight is that the memory management challenges in LLM KV cache—dynamic size, need for contiguous access, and potential for sharing—are analogous to the challenges faced by operating systems (OS) when managing memory for processes.

The main contribution is PagedAttention, an attention algorithm inspired by the classical OS concepts of virtual memory and paging. PagedAttention allows the KV cache for a sequence to be stored in non-contiguous physical memory blocks, enabling fine-grained, block-level memory management and sharing.

Brief overview of the proposed solution

The proposed solution is the vLLM system, built on PagedAttention.

PagedAttention: It partitions the KV cache of a sequence into fixed-size "KV blocks" (analogous to OS pages). The attention kernel is modified to fetch and process these non-contiguous blocks via a Block Table (analogous to an OS page table).

KV Cache Manager: This component dynamically allocates physical blocks on demand for a request's logical blocks. Because blocks are fixed-size and allocated on demand, internal and external fragmentation are nearly eliminated.

Memory Sharing: The block table structure enables memory sharing. For complex decoding algorithms like parallel sampling and beam search, the system uses a reference counting mechanism on physical blocks and implements a copy-on-write policy, allowing multiple sequences to share common KV cache blocks (e.g., for shared prompts or beam search prefixes).

Scheduling and Preemption: The system uses a centralized scheduler and implements a block-level swap-out mechanism (to CPU RAM) or recomputation to handle memory exhaustion, enabling effective preemption of low-priority requests without losing all progress.

2. Technical Understanding (2-3 Pages)

a) Problem Analysis

Detailed explanation of the problem

LLM serving is characterized by autoregressive generation, where the generation of the current token depends on all previous tokens in the sequence. This dependency is maintained using the KV cache, which stores the key ($K$) and value ($V$) vectors for every previously generated token.

The core issue arises from two characteristics of the KV cache:

Dynamic and Unpredictable Size: The output length of an LLM request is unknown a priori. It grows one token at a time until termination.

Contiguous Memory Requirement: Existing deep learning frameworks and specialized serving systems (like Faster Transformer) require the KV cache tensor for a single sequence to be stored in a contiguous chunk of GPU memory.

Why existing solutions are inadequate

Existing solutions, exemplified by Orca (a state-of-the-art system) and Faster Transformer, are inadequate because they rely on statically or over-provisionally allocating a contiguous memory chunk based on the request's maximum possible sequence length (e.g., 2048 tokens). This approach leads to three main types of memory waste (as quantified in Figure 2):

Internal Fragmentation: Memory reserved for the request's maximum possible length that is never actually used if the generated sequence is short. This can be up to $57.3\%$ of the KV cache memory (Orca Max).

Reserved Slots: Memory slots reserved for tokens that will be generated in the future. Although eventually used, reserving this space for the entire request lifetime prevents other requests from using it, limiting concurrent batching.

External Fragmentation: Since the required contiguous chunk sizes vary between requests, a contiguous memory allocator (like the buddy allocator used by Orca) cannot efficiently pack the chunks, leaving unusable gaps.

Furthermore, the requirement for a contiguous chunk prohibits memory sharing. In complex decoding (like beam search or parallel sampling), multiple output sequences often share a common prompt or prefix. Contiguous allocation forces a full copy for each sequence, duplicating the shared prefix cache and wasting significant memory (Figure 8).

Motivating examples or workloads

The inadequacy is clearly motivated by workloads with high variance in sequence length, such as the ShareGPT dataset, which has long inputs and outputs (mean input: 161 tokens, mean output: 338 tokens) and high variance.

The problem is exacerbated by complex decoding algorithms:

Parallel Sampling: Generates multiple independent sequences from a single prompt. The KV cache for the (often long) prompt is duplicated across all samples.

Beam Search: Generates multiple candidate sequences, which share large common prefixes that change dynamically. Duplicating or frequently copying these large prefixes results in high memory/copy overhead.

b) Proposed Solution

System design and architecture

The vLLM system (Figure 4) adopts a distributed architecture with a centralized Scheduler and multiple distributed GPU Workers. The key innovation is the KV Cache Manager, which manages memory in a paged fashion, enabled by the new attention algorithm.

Key algorithms or techniques

PagedAttention (Paging for LLMs): This is the core modification to the attention mechanism.

The KV cache for a sequence is divided into Logical KV Blocks.

The system maintains a Block Table for each sequence, mapping its Logical KV Blocks to non-contiguous Physical KV Blocks stored in GPU DRAM.

The PagedAttention kernel is modified to read the key and value vectors block-wise using this block table mapping (Figure 5). This non-contiguous access decouples the logical view of the sequence from the physical memory layout, eliminating the contiguous memory requirement.

Dynamic Block Allocation:

Physical blocks are allocated dynamically and on demand as new tokens are generated. Since all physical blocks are fixed-size, external fragmentation is eliminated, and internal fragmentation is minimized to, at most, one KV block per sequence.

Copy-on-Write (CoW) for Sharing:

For parallel sampling or beam search, sequences share the same physical blocks for their common prefixes (e.g., the input prompt blocks).

Each physical block has a reference count (Figure 8).

When a sequence needs to write a new token to a shared physical block, the system detects the reference count is greater than 1, allocates a new physical block, copies the data from the old shared block (CoW), updates the sequence's block table entry to point to the new block, and decrements the old block's reference count. This achieves memory sharing with minimal overhead.

Preemption and Recovery:

When GPU memory runs out, the scheduler preempts a sequence group (like beam candidates).

The recovery method is either:

Swapping: The KV blocks are copied from GPU memory to CPU RAM via the CPU Block Allocator.

Recomputation: The blocks are simply discarded, and when the request is rescheduled, the model recomputes the necessary KV cache by treating the previously generated sequence as a new prompt.

c) Evaluation

Experimental setup

Models: OPT-13B, OPT-66B, OPT-175B, and LLAMA-13B, run on NVIDIA A100 GPUs (1 to 8 GPUs depending on model size).

Workloads: Synthesized requests based on real-world datasets with different length characteristics:

ShareGPT: Long sequences with high variance.

Alpaca: Shorter sequences with lower variance.

Baselines:

Faster Transformer (FT): Distributed, latency-optimized engine using basic memory allocation (similar to Orca Max).

Orca (State-of-the-Art Throughput): Custom-implemented with three memory management policies for comparison:

Orca (Max): Reserves max sequence length (worst-case fragmentation).

Orca (Pow2): Reserves power-of-2 size, limiting over-reservation to 2x (better, but still fragmentation).

Orca (Oracle): Assumes perfect knowledge of final output length (infeasible upper bound for Orca's approach).

Key Metric: Normalized latency (seconds/token) vs. Request rate (req/s). A better system maintains low normalized latency at higher request rates.

Key results and metrics

Basic Sampling (Figure 12): On the memory-intensive ShareGPT workload, vLLM sustains 1.7x-2.7x higher request rates compared to the infeasible upper-bound Orca (Oracle), and up to 8x higher than Orca (Max). This is because vLLM batches 2.2x to 4.3x more requests simultaneously by reclaiming wasted memory (Figure 13).

Complex Decoding (Figure 14 & 15): The benefits of sharing are profound:

In Beam Search (width=6), vLLM's throughput advantage over Orca (Oracle) increases from 1.3x (basic sampling) to 2.3x.

This is due to high memory saving: $37.6\%$ to $55.2\%$ for beam search, achieved via CoW and reference counting.

Shared Prefix (Figure 16): For workloads with long shared context (few-shot prefix prompt), vLLM achieves 3.58x higher throughput than Orca (Oracle) by caching and sharing the prefix KV blocks.

Ablation Study (Figure 18):

The PagedAttention kernel introduces a small overhead (20-26% higher latency) compared to the highly optimized contiguous attention in Faster Transformer, but this micro-overhead is negligible in the overall end-to-end throughput gain.

Optimal Block Size is around 16-32 tokens, balancing GPU utilization/parallelism (larger size preferred) against internal fragmentation (smaller size preferred).

How results support the claims

The results strongly support the claims:

Near-zero waste: Figure 2 shows vLLM using $96.3\%$ of the KV cache for actual token states, compared to $20.4\% - 38.2\%$ for Orca baselines, proving the effectiveness of the paging approach in reducing fragmentation.

Flexible sharing: The superior performance in beam search and parallel sampling (Figure 14) and the quantified memory savings (Figure 15) demonstrate that the CoW-enabled block-level sharing is highly effective in practice, especially for workloads with common prefixes or shared tree structures.

High throughput: The consistent 2x-4x throughput improvements across different models and workloads (Figure 12) prove that solving the memory bottleneck via paging is the key to maximizing GPU utility in LLM serving.

3. Critical Analysis (2-3 Pages)

Strengths

Novel Insight and Conceptual Elegance: The application of virtual memory and paging—a 60-year-old OS concept—to modern GPU memory management for LLM serving is a highly novel and elegant solution. It successfully reframes the dynamically growing KV cache as a "virtual memory space" problem, demonstrating fundamental computer science principles transcend hardware generations.

Comprehensive Fragmentation Solution: PagedAttention effectively addresses all three major forms of memory waste identified in existing systems (internal, reserved, and external fragmentation) with a single mechanism. The achievement of $96.3\%$ effective utilization is a strong technical achievement.

Effective Handling of Complex Workloads: The introduction of Copy-on-Write (CoW) and reference counting is crucial. This not only makes basic serving efficient but also unlocks huge performance gains (up to 2.3x) for complex, real-world decoding algorithms like beam search, which were previously burdened by high memory duplication/copying overhead.

Robust Evaluation against Strong Baselines: The paper uses comprehensive and realistic baselines, including three versions of Orca (Max, Pow2, and the infeasible Oracle upper-bound). Beating the Orca (Oracle) baseline decisively proves that the system design (paged memory) is fundamentally superior to the contiguous allocation paradigm, even when the latter has perfect a priori knowledge.

Weaknesses

Kernel Overhead and Optimality: While the paper claims the 20-26% higher attention kernel latency for PagedAttention (Figure 18a) is small, this overhead is fundamental to the non-contiguous memory access. As models get smaller or token generation latency becomes dominated by compute rather than memory (e.g., with very fast future GPUs), this overhead could become the dominant factor, limiting the theoretical peak performance of the GPU. The design is optimized for memory-bound scenarios, which may not always hold true.

Increased Complexity and Implementation Burden: Introducing a block table, a two-level memory management hierarchy (logical-to-physical), and complex CoW/reference counting significantly complicates the core attention kernel and the scheduler. Maintaining this complexity, especially in CUDA kernels, is a high engineering cost compared to the simpler, contiguous buffer approach. Future optimizations to the underlying deep learning framework's attention kernel (like FlashAttention) may be harder to integrate.

Swapping vs. Recomputation Trade-off: The discussion and evaluation of swapping (to CPU RAM) vs. recomputation (Figure 19) are important but show that the optimal recovery mechanism is highly dependent on the block size and the PCIe bandwidth. Given that recomputation latency is never much worse than swapping and is much simpler, the argument for including swapping as a feature is somewhat weak, especially since it requires the complexity of a separate CPU block allocator and memory transfer management.

Specific Critiques (Choose at least 3)

1. Are the evaluation metrics appropriate? (Critique: Mostly Appropriate, but one key metric is missing)

The use of Normalized Latency (s/token) vs. Request Rate (req/s) is highly appropriate for measuring serving throughput and system capacity. However, a key metric for memory systems, Jitter/Tail Latency, is largely missing. While mean normalized latency is measured, LLM services often have strict Service Level Objectives (SLOs) on 95th or 99th percentile latency. The preemption/swapping mechanism, while good for overall throughput, can introduce high latency spikes (jitter) for the preempted requests. A more complete evaluation of $p95/p99$ latency, especially for workloads that trigger preemption, would be necessary to fully assess the system's suitability for production environments.

2. Is the baseline comparison fair? (Critique: Extremely Fair and Effective)

The baseline comparison is outstandingly fair. By implementing three versions of Orca, particularly the Orca (Oracle) baseline, the authors move beyond simply comparing against existing performance and instead benchmark against the theoretical upper bound of the contiguous-allocation design. Demonstrating that vLLM still outperforms the Oracle baseline by up to 2.7x (Figure 12) is the most compelling piece of evidence that the paging approach is fundamentally superior for this class of problem, regardless of how well the contiguous approach is optimized.

3. Does the solution generalize beyond the tested scenarios? (Critique: High Generalizability, but limited to KV Cache/Autoregression)

The core PagedAttention concept generalizes well to any autoregressive Transformer-based model (not just OPT/LLAMA) and any task that involves shared state (chatbots, complex sampling, prefix tuning). The solution's power lies in addressing a memory bottleneck common to all sequential, token-by-token generation. However, the solution is specifically limited to the KV cache and autoregressive decoding. As the authors acknowledge, it is not directly applicable to traditional DNN training or serving non-autoregressive models, where static tensor shapes and compute-bound nature make the benefits of memory indirection negligible or detrimental.

4. Are the claimed contributions novel? (Critique: Novel in Application, not in Principle)

The claimed contributions are highly novel in their application and integration into a high-performance LLM serving system, but the core principle (paging, block table, copy-on-write) is a direct, acknowledged adaptation of operating system concepts dating back to the 1960s (Kilburn et al., 1962). The true novelty lies in identifying the structural analogy between OS processes/memory and LLM sequences/KV cache, and then successfully engineering the attention kernel and memory manager to realize these principles on modern GPU hardware.

4. Personal Reflection (1 Page)

What did you learn?

I learned a profound lesson about the cyclical nature of computer science problems. The core challenges of modern, bleeding-edge AI infrastructure—dynamic resource allocation, fragmentation, and state sharing—are conceptually identical to the fundamental memory management problems that operating systems designers solved decades ago. Specifically, I realized that the Transformer's KV cache is essentially a process's dynamically growing stack/heap space that needs to be accessed contiguously in a logical sense, but must be managed non-contiguously in a physical sense to maximize memory density. I also gained a much deeper appreciation for the high cost of memory fragmentation in modern, high-throughput systems.

What surprised you?

What surprised me most was the magnitude of the performance gain over the Orca (Oracle) baseline. Prior to reading the evaluation, I would have assumed that an "Oracle" system (knowing the exact sequence length beforehand) would be nearly unbeatable. The fact that vLLM's system design allowed it to outperform the contiguous-allocation system's theoretical maximum by 1.7x to 2.7x shows that the memory inefficiency is not just about prediction error (the Orca Max/Pow2 issue), but a fundamental design flaw in using contiguous tensors for dynamically sized objects in a batching environment. The elegance of using Copy-on-Write for shared state in beam search was also a delightful surprise—it's a perfect fit.

How does this relate to other systems concepts you know?

This work directly relates to several core systems concepts:

Operating Systems (OS): The entire system is an OS-analogy: Logical Block $\approx$ Virtual Page, Physical Block $\approx$ Physical Frame, Block Table $\approx$ Page Table, Sequence $\approx$ Process, and Copy-on-Write $\approx$ Forking a Process.

Database/Storage Systems: The dynamic memory management and eviction policies (swapping vs. recomputation) mirror techniques used in cache management and transaction processing. The all-or-nothing eviction policy is similar to coarse-grained locking or group-based eviction strategies used in some storage systems to simplify recovery or consistency.

Compiler/Runtime Systems: The use of kernel fusion (fusing block read/write with attention) to mitigate memory indirection overhead is a classic compiler/runtime optimization technique, ensuring that abstract/flexible structures (like the block table) don't incur excessive runtime costs.

Would you have designed it differently? How?

Given the results, the vLLM design is highly effective, but if I were designing it, I would have focused the recovery mechanism purely on Recomputation and eliminated the complexity of the Swapping feature.

Why Recomputation over Swapping?

Complexity: Swapping requires a full CPU-side memory management layer and reliable, high-bandwidth PCIe transfers, which are system-dependent and prone to errors (Fig 19a shows high overhead for small blocks).

LLM Specific Semantic: Recomputation is a luxury only available because the KV cache can be quickly regenerated in a single, parallelized "prompt phase" pass once the partially completed sequence is known (Section 4.5). This is a domain-specific advantage that traditional OS cannot use. The performance difference is minor for medium block sizes (Fig 19b), and the simplicity of recomputation makes the overall system more robust and easier to maintain.

Therefore, a simplified design would have been: PagedAttention + Dynamic Block Allocation + Copy-on-Write + Recomputation-Only Preemption. This maximizes the benefit-to-complexity ratio.