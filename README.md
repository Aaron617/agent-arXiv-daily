# Agent arXiv Daily

**Last Updated:** 2026-05-21 05:19:44

**Total Papers:** 91

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (6 papers)</h2></summary>

<details>
<summary><strong>SOLAR: A Self-Optimizing Open-Ended Autonomous Agent for Lifelong Learning and Continual Adaptation</strong> - Nitin Vetcha, Dianbo Liu - [[pdf]](https://arxiv.org/pdf/2605.20189)</summary>

**Abstract:** Despite the remarkable success of large language models (LLMs), they still face bottlenecks while deploying in dynamic, real-world settings with primary challenges being concept drift and the high cost of gradient-based adaptation. Traditional fine-tuning (FT) struggles to adapt to non-stationary data streams without resulting in catastrophic for getting or requiring extensive manual data curation. To address these limitations within the streaming and continual learning paradigm, we propose the Self-Optimizing Lifelong Autonomous Reasoner (SOLAR) which is an open-ended autonomous agent that leverages parameter-level meta-learning to self-improve, treating model weights as an environment for exploration. It initiates the process by consolidating a strong prior over common-sense knowledge making it effective for transfer-learning. By utilizing a multi-level reinforcement learning approach, SOLAR autonomously discovers adaptation strategies, enabling efficient test-time adaptation to unseen domains. Crucially, SOLAR maintains an evolving knowledge base of valid modification strategies, implicitly acting as an episodic memory buffer to balance plasticity (adaptation to new tasks) and stability (retention of meta-knowledge). Experiments demonstrate that SOLAR outperforms strong baselines on common-sense, mathematical, medical, coding, social and logical reasoning tasks, marking a significant step toward autonomous agents capable of lifelong adaptation in evolving environments.

**arXiv ID:** 2605.20189
</details>

<details>
<summary><strong>Evaluating Temporal Semantic Caching and Workflow Optimization in Agentic Plan-Execute Pipelines</strong> - Alimurtaza Mustafa Merchant, Krish Veera, Sajal Kumar Goyla, Shambhawi Bhure, Dhaval Patel, Kaoutar El Maghraoui - [[pdf]](https://arxiv.org/pdf/2605.20630)</summary>

**Abstract:** Industrial asset operations workflows are latency-sensitive because a single user query may require coordination over sensor data, work orders, failure modes, forecasting tools, and domain-specific agents. We evaluate this problem on AssetOpsBench (AOB), an industrial agent benchmark whose plan-execute pipeline exposes repeated overhead from tool discovery, LLM planning, MCP tool execution, and final summarization. Existing LLM caching techniques such as KV-cache reuse and embedding-based semantic caching were designed for chatbot serving and break down when output validity depends on time, asset, or sensor parameters. We propose two complementary optimization layers for AOB plan-execute pipelines: a temporal semantic cache and a set of MCP workflow optimizations combining disk-backed tool-discovery caching and dependency-aware parallel step execution. MCP workflow optimizations corresponded to a 1.67x speedup and reduced median end-to-end latency by about 40.0% while the temporal-cache benchmark achieved a median of 30.6x speedup on cache hits. Beyond the speedup, our results expose a concrete failure mode of pure semantic caching for parameter-rich industrial queries, providing a critical analysis of how caching choices interact with evaluation correctness in MCP-backed agent benchmarks.

**arXiv ID:** 2605.20630
</details>

<details>
<summary><strong>Evaluating multimodal emotion recognition in proactive conversational agents: A user study</strong> - Adnana Dragut, Raquel Lacuesta, F. Xavier Gaya-Morey, Jose M. Buades-Rubio - [[pdf]](https://arxiv.org/pdf/2605.20200)</summary>

**Abstract:** This article presents a multimodal emotion recognition module integrated into a proactive Socially Interactive Agent (SIA) powered by generative artificial intelligence. The system evaluates real-time affective states through two distinct channels: a computer vision-based facial recognition module and a semantic linguistic analysis engine. To validate the framework, an empirical study was conducted with 20 users who engaged in dynamic, unscripted dialogues with the conversational agent. The findings reveal a significant discrepancy between automated visual cues and actual internal emotional states. When interacting with the AI, users consistently exhibited a "poker face" effect, displaying serious, concentrated facial expressions even when experiencing positive emotions. Consequently, the generative AI linguistic analysis proved significantly more reliable, by contextualizing the users' verbal expressions. Furthermore, an analysis of the interaction dynamics demonstrated that SIAs can effectively elicit specific emotions by adapting conversational themes and employing structured linguistic patterns, such as empathetic or humorous language. However, the study also noted that instances of uncalibrated proactivity occasionally led to user disengagement and a perception of artificiality. Ultimately, this research highlights the necessity of refining SIAs to dynamically adapt to users' emotional evolution, relying on deep linguistic context to foster more natural, human-like interactions.

**arXiv ID:** 2605.20200
</details>

<details>
<summary><strong>GrandGuard: Taxonomy, Benchmark, and Safeguards for Elderly-Chatbot Interaction Safety</strong> - Changxuan Fan, Xi Yang, Yueyuan Zheng, Bin Zhou, Yuanping Wang, Wenbin Hu, Huihao Jing, Ki Sen Hung, Dazhao Du, Haoran Li, Janet Hui-wen Hsiao, Yangqiu Song - [[pdf]](https://arxiv.org/pdf/2605.20203)</summary>

**Abstract:** As older adults increasingly use LLM-based chatbots for companionship and assistance, a safety gap is emerging. Older adults may face vulnerabilities from social isolation, limited digital literacy, and cognitive decline, yet existing safety benchmarks largely target general harms and overlook elderly-specific risks. For example, a prompt such as "how to repair a ceiling light alone in the dark" may be benign for most users but poses a serious fall risk for older adults with mobility limitations. We introduce GrandGuard, the first comprehensive framework for assessing and mitigating elderly-specific contextual risks in LLM interactions. We develop a three-level taxonomy with 50 fine-grained risk types across mental well-being, financial, medical, toxicity, and privacy domains, grounded in real-world incidents, community discussions, and analysis of stakeholder studies. Using this taxonomy, we construct a benchmark of 10,404 labeled prompts and responses, showing that several leading LLMs mishandle elderly-specific contextual risks in over 50% of cases. We mitigate these failures with two safeguards: a fine-tuned Llama-Guard-3 and a policy-enhanced gpt-oss-safeguard-20b, achieving up to 96.2% and 90.9% unsafe-prompt detection accuracy, respectively. GrandGuard lays the groundwork for AI systems that move beyond general safety to support aging populations.

**arXiv ID:** 2605.20203
</details>

<details>
<summary><strong>Pramana: A Protocol-Layer Treatment of Claim Verification in Autonomous Agent Networks</strong> - Ravi Kiran Kadaboina - [[pdf]](https://arxiv.org/pdf/2605.20312)</summary>

**Abstract:** Autonomous agents deployed in regulated domains must produce a verification artifact per consequential output: a record an auditor can re-execute offline, capturing what was claimed, against what source, by whom, when, and how. Production verification today splits into two unstandardized halves. Probabilistic verdict patterns (self-consistency voting, reviewer LLM ensembles) produce judgments, not artifacts. Artifact-producing patterns (RAG, tool-augmented traces, generator-verifier loops) produce vendor-specific records no external auditor can reconstruct without bespoke integration.
Pramana defines the missing wire format. Every consequential agent output is wrapped in a typed ClaimAttestation with one of four variants (measurement, inference, analogy, citation), each paired with a verify() operation against the recorded source. verify() is deterministic for MeasurementClaim and CitationClaim. For InferenceClaim and AnalogyClaim, determinism is conditional on the oracle (audit-replayable when LLM-backed). The four-way typology derives from classical Indian epistemology (pramana, valid means of knowledge).
The lifecycle is specified in TLA+ and exhaustively verified under TLC across three symmetry-reduced models: 38,563 distinct reachable states, zero invariant violations. The Python reference implementation passes 84 tests. An A2A and MCP wire-extension manifest layers three deployment-grade invariants: reachability, SLA bound, and offline re-verifiability.
An exploratory pilot (n=100, 2,275 reviewer calls) probes LLM-as-judge in code generation. The strongest observation is a 40-percentage-point raw FPR delta across corpora, consistent with reference-solution quality contributing significantly. The pilot does not validate Pramana on its own; the structural argument and formal verification do that.

**arXiv ID:** 2605.20312
</details>

<details>
<summary><strong>WildRoadBench: A Wild Aerial Road-Damage Grounding Benchmark for Vision-Language Models and Autonomous Agents</strong> - Bingnan Liu, Chenhang Cui, Rui Huang, Jiani Luo, Zhirong Shen, Tinghao Wang, Xiande Huang, Lingbei Meng, Fei Shen, An Zhang - [[pdf]](https://arxiv.org/pdf/2605.20306)</summary>

**Abstract:** We introduce WildRoadBench, a wild aerial road-damage grounding benchmark that couples direct visual grounding by vision-language models with autonomous research-and-engineering by LLM-driven agents on a single professionally annotated UAV corpus. The same image set and the same per-class AP_50 metric are evaluated under two protocols. The VLM Track measures whether a fixed VLM can localise domain-specific damage from one image and one short prompt under a unified prompting, decoding and parsing pipeline. The Agent Track measures whether an autonomous agent, given only a written task brief, a small exploratory slice and a fixed interaction budget, can search the public web, adapt pretrained components, write training and inference code, and submit predictions through a scalar-feedback oracle on a hidden holdout. We benchmark a broad pool of closed-source frontier models and open-source VLMs together with several frontier LLM-driven agents. Both routes remain far from reliable performance in this wild setting: closed-source frontier models lead the VLM leaderboard but still leave more than half of the metric on the table; open-source grounders plateau well below them, and newer generations or reasoning-style variants do not consistently improve grounding; small targets collapse for every open-source model; agents lag the strongest VLM despite richer affordances, and several fail to land a valid submission within the budget. We release the code and data at this https URL to support reproducible follow-up research.

**arXiv ID:** 2605.20306
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>Declarative Data Services: Structured Agentic Discovery for Composing Data Systems</strong> - Shanshan Ye, Duo Lu - [[pdf]](https://arxiv.org/pdf/2605.20690)</summary>

**Abstract:** Agentic discovery has shown that LLM-driven search can find novel algorithms, designs, and code under benchmark conditions. Translating the paradigm to multi-system data backends surfaces a harder problem: the search space is heterogeneous, the verifier is whether a deployed stack actually runs, and composition knowledge is unevenly captured in pretraining. Unbounded agentic discovery, a coding agent iterating on failure-log feedback, fails to converge consistently on a working stack even when iteration and explicit composition knowledge are added. We propose Declarative Data Services (DDS), an architecture for structured agentic discovery of data-system compositions from declarative user intent. The framework owns four typed contracts at successive layers (intent, operator DAG, per-system skills, runtime attribution) that decompose the global search into bounded sub-searches; sub-agents search each typed space, while the framework provides the channels by which knowledge flows forward as inline skill citations and errors route backward as typed signals. As a proof of life on a trading-backend workload, DDS converges where unbounded discovery does not; runtime failures become skill patches that the next deployment cites inline. We position this as an early prototype reporting lessons from real-world data-system composition.

**arXiv ID:** 2605.20690
</details>

<details>
<summary><strong>RealUserSim: Bridging the Reality Gap in Agent Benchmarking via Grounded User Simulation</strong> - Ming Zhu, Juntao Tan, Rithesh Murthy, Jielin Qiu, Liangwei Yang, Wenting Zhao, Silvio Savarese, Shelby Heinecke, Huan Wang - [[pdf]](https://arxiv.org/pdf/2605.20204)</summary>

**Abstract:** LLM-based user simulation is the primary mechanism for end-to-end agent evaluation, yet simulated users are poor proxies for real humans: unconstrained LLM defaults produce a Formalism Ceiling (style match rates of 6-8% against real users), while hand-crafted behavioral directives trigger Directive Amplification, where models hyper-interpret instructions into unnatural behavioral extremes that vary dramatically across simulator models. We present RealUserSim, the first user simulation framework grounded in real behavioral data. From 14,000+ authentic human-LLM conversations (WildChat), we extract 7,275 executable behavioral profiles and use them to ground LLM simulators. A fidelity benchmark (PT3) on 600 conversations across 71+ domains with anti-leakage controls shows that grounded simulation raises match rate from 24.2% to 45.3% across five behavioral dimensions. Agent evaluation on TauBench with 6 simulator models and extensive analysis shows that grounded simulation acts as a realistic stress test, surfacing three failure mechanisms invisible to cooperative simulators (mean -3.2% to -3.5% task success degradation), while Directive Amplification in existing benchmarks produces unrealistic behavior that compromises the validity of agent evaluation.

**arXiv ID:** 2605.20204
</details>

<details>
<summary><strong>ProcBench: Evaluating Process-Level Defects and Control Preservation in LLM Coding Agents</strong> - Jiawei He, Jie Jia, Chenbo Liu, Chaoyi Xue, Yapeng Song, Xikai Yang, Dong Sun - [[pdf]](https://arxiv.org/pdf/2605.20251)</summary>

**Abstract:** Existing benchmarks for LLM coding agents mainly evaluate final outcomes, such as task completion, compilation success, and test pass rates. While these metrics are useful for measuring end-task capability, they provide limited visibility into how an execution unfolds and often miss recurrent process-level failures that arise during multi-step operation. We present ProcBench, a benchmark-oriented framework for evaluating coding-agent trajectories through process defects and control preservation. ProcBench organizes execution failures into a reusable ontology, standardizes heterogeneous logs into a unified trajectory representation, and reports calibrated risk-based scorecards instead of relying only on final outcomes. We instantiate ProcBench on an annotated set of 200 trajectories and apply it across three coding-agent benchmarks: AndroidBench, TerminalBench, and SWE-bench-Verified. Our results suggest that ProcBench can be instantiated with useful reliability, that calibration improves the empirical interpretability of defect findings relative to direct thresholding, and that process-aware scorecards provide diagnostic distinctions beyond conventional outcome-based evaluation. We also discuss limitations, including annotation dependence, partial observability for some defect classes, and the need for broader external validation.

**arXiv ID:** 2605.20251
</details>

<details>
<summary><strong>STELLAR: Scaling 3D Perception Large Models for Autonomous Driving</strong> - Yingwei Li, Xin Huang, Yang Liu, Yang Fu, Alex Zihao Zhu, Chen Song, Junwen Yao, Anant Subramanian, Hao Xiang, Weijing Shi, Yuliang Zou, Tom Hoddes, Zhaoqi Leng, Govind Thattai, Dragomir Anguelov, Mingxing Tan - [[pdf]](https://arxiv.org/pdf/2605.20390)</summary>

**Abstract:** Model scaling has demonstrated remarkable success through large-scale training on diverse datasets. It remains an open question whether the same paradigm would apply to autonomous driving perception systems due to unique challenges, such as fusing heterogeneous sensor data and the need for sophisticated 3D spatial understanding. To bridge this gap, we present a comprehensive study on systematically analyzing the impact of scale on these systems. We develop our STELLAR model based on Sparse Window Transformer, by extending the input modalities to include LiDAR, radar, camera, and map prior. We train the model on a large-scale dataset of 50 million driving examples with up to 500 million parameters. Our large-scale experiments reveal empirical scaling trends that connect model performance to model size, data, and compute. The resulting model establishes a new state-of-the-art on the Waymo Open Dataset challenge, outperforming prior arts by a large margin. Our work demonstrates that large-scale training is a highly promising path for advancing the capabilities of perception models for autonomous driving.

**arXiv ID:** 2605.20390
</details>

<details>
<summary><strong>Training Language Agents to Learn from Experience</strong> - Yuval Shalev, Zifeng Ding, Mateja Jamnik - [[pdf]](https://arxiv.org/pdf/2605.20477)</summary>

**Abstract:** Language agents can adapt from experience in interactive environments, but current reflection-based methods can only self-correct within a single task instance. Whether such experience can be distilled into reusable lessons that improve performance on future unseen tasks remains unclear. We address this problem by introducing the In-context Training (ICT) task, a framework for evaluating cross-task self-improvement in language agents. In ICT, a reflector model observes trajectories collected by an actor model and generates system prompts intended to improve the actor's performance on future unseen tasks. We then propose an RL-based training pipeline for learning such reflections directly from experience, without human-provided examples. Across ALFWorld and MiniHack, our trained reflectors outperform an untrained baseline on most held-out task families, showing that the ability to learn from experience can itself be learned. In some cases, we observe generalisation beyond the benchmark on which the reflector was trained, to substantially different environments. Finally, we introduce MetaGym, a generic Python library for constructing meta-environments, enabling future research on self-improving language agents.

**arXiv ID:** 2605.20477
</details>

<details>
<summary><strong>Heartbeat-Bound Hierarchical Credentials: Cryptographic Revocation for AI Agent Swarms</strong> - Saurabh Deochake - [[pdf]](https://arxiv.org/pdf/2605.20704)</summary>

**Abstract:** Autonomous AI agents that spawn sub-agent swarms create a safety gap: existing credential revocation mechanisms, OAuth~2.0 introspection, OCSP, and W3C Status Lists, require network connectivity to a central authority, leaving ``zombie agents'' executing privileged operations for minutes to hours after operator shutdown. We present Heartbeat-Bound Hierarchical Credentials (HBHC), a cryptographic protocol that binds credential validity to periodic parent liveness proofs. Verifiers enforce freshness using only a cached public key and local clock; no network round-trip is required. When heartbeat generation ceases, all descendant credentials become unusable within a deterministically bounded window $W_z \le W_{\max} + \Delta_h + \epsilon$, conditional on bounded clock skew and parent keys held in secure enclaves. Evaluation at the protocol layer and with real LLM-backed agent swarms (GPT-4o-mini) demonstrates a 90$\times$ reduction in the zombie window over OAuth~2.0, 0.26~ms full authentication in Rust, 18,000+ verifications per second under concurrent HTTP load, and stable per-verification latency from 10 to 10,000 agents. Real-agent experiments show 0.71\% end-to-end overhead on tool calls, zero post-revocation tool calls under prompt injection that bypasses application-layer guardrails, and cascading revocation across a 49-agent four-level hierarchy within the theoretical bound.

**arXiv ID:** 2605.20704
</details>

<details>
<summary><strong>Terminal-World: Scaling Terminal-Agent Environments via Agent Skills</strong> - Zihao Cheng, Hongru Wang, Zeming Liu, Xinyi Wang, Xiangrong Zhu, Yuhang Guo, Wei Lin, Jeff Z. Pan, Yunhong Wang - [[pdf]](https://arxiv.org/pdf/2605.20876)</summary>

**Abstract:** Terminal agents extend Large Language Models with the ability to execute tasks directly in command-line environments, but their progress is bottlenecked by the scarcity of high-quality training data. Existing approaches bootstrap from partial sources such as human-defined seeds or GitHub repositories to instantiate one component and then complete the rest, producing tasks confined to narrow seed distributions, environments misaligned with task semantics, and inefficient trajectories from unguided exploration. To address these limitations, we introduce Terminal-World, a fully automated pipeline that uses agent skills as the central synthesis primitive, which jointly encode what to accomplish, when to apply (preconditions and environment state), and how to execute, enabling task instructions, environments, and teacher trajectories to be co-derived. To further broaden the synthesis space, Terminal-World composes skills into skill teams and skill graphs for multi-role and cross-domain task synthesis. Using this pipeline, we construct 5,723 training environments and train Terminal-World-8B/14B/32B, evaluated across 6 benchmarks where the Terminal-World series consistently outperforms terminal-agent baselines. Notably, using the same teacher model and only 1.2% of the training data, Terminal-World-32B surpasses Nemotron-Terminal-32B on Terminal-Bench 2.0 by +4.5 Pass@1 (31.5) and achieves 43.8 Pass@3.

**arXiv ID:** 2605.20876
</details>

<details>
<summary><strong>SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents</strong> - Bingchen Zhao, Dhruv Srikanth, Yuxiang Wu, Zhengyao Jiang - [[pdf]](https://arxiv.org/pdf/2605.21384)</summary>

**Abstract:** As long-horizon coding agents produce more code than any developer can review, oversight collapses onto a single surface: the automated test suite. Reward hacking naturally arises in this setup, as the agent optimizes for passing tests while deviating from the users true goal. We study this reward hacking phenomenon by decompose software engineering tasks into three parts: (i) a natural language description of the specification (ii) visible validation tests that exercise specified features in isolation, and (iii) held-out tests that compose those same features to simulate real-world usage. Based on the specification and the visible validation test suites, a genuine agent would be able to generate a solution that can also pass all of the held-out tests. Therefore we use the gap in pass rates on these two suites to quantify reward hacking. Based on this methodology, we introduce SpecBench, a benchmark comprising 30 systems-level programming tasks ranging from short horizon tasks like building a JSON parser to ultra long horizon tasks like building an entire OS kernel from scratch. Large-scale experiments reveal a consistent pattern: while every frontier agent saturates the visible suite, reward hacking persists, with smaller models exhibiting larger gaps on holdout suites. The gap also scales sharply with task length: it grows by 28 percentage points for every tenfold increase in code size. Failures range from subtle feature isolation to deliberate exploits, including a 2,900-line hash-table "compiler" that memorizes test inputs. SpecBench offers a principled testbed for measuring whether coding agents build genuine working systems or merely game the test suites developers hand them.

**arXiv ID:** 2605.21384
</details>

<details>
<summary><strong>AMBER: A Columnar Architecture for High-Performance Agent-Based Modeling in Python</strong> - Anh-Duy Pham - [[pdf]](https://arxiv.org/pdf/2601.16292)</summary>

**Abstract:** Python is widely used for agent-based modelling because it is accessible and has a mature scientific ecosystem, but object-per-agent execution incurs interpreter overhead that restricts the population sizes feasible in interactive modelling, calibration, and parameter sweeps. This paper presents AMBER, a Python framework that stores agent state in a Polars-backed columnar table and exposes population operations through a compact view API. The framework preserves conventional model and agent abstractions while translating common population updates into compiled column operations; behaviours that do not vectorise remain expressible through a buffered object-oriented path. We evaluate AMBER on wealth transfer, random walk, and spatial SIR benchmarks against Mesa, AgentPy, SimPy, Melodie, this http URL, and AMBER's own loop path, using invariant checks to verify comparable model outputs before timing. Across the tested workloads, AMBER has the lowest execution time among Python-hosted implementations and achieves speedups of up to $1118\times$ over Mesa; on the largest SIR benchmark it is also faster than the Julia-based this http URL implementation.

**arXiv ID:** 2601.16292
</details>

<details>
<summary><strong>Hypergraph Enterprise Agentic Reasoner over Heterogeneous Business Systems</strong> - Ling Wang, Xin Liu, Songnan Liu, Jianan Wang, Cheng Cheng, Yihan Zhu, Enyu Li, Yu Xiao, Jiangyong Xie, Duogong Yan, Jiangyi Chen - [[pdf]](https://arxiv.org/pdf/2605.14259)</summary>

**Abstract:** Applying Large Language Models (LLMs) to heterogeneous enterprise systems is hindered by hallucinations and failures in multi-hop, n-ary reasoning. Existing paradigms (e.g., GraphRAG, NL2SQL) lack the semantic grounding and auditable execution required for these complex environments. We introduce HEAR, an enterprise agentic reasoner built on a Stratified Hypergraph Ontology. Its base Graph Layer virtualizes provenance-aware data interfaces, while the Hyperedge Layer encodes n-ary business rules and procedural protocols. Operating an evidence-driven reasoning loop, HEAR dynamically orchestrates ontology tools for structured multi-hop analysis without requiring LLM retraining. Evaluations on supply-chain tasks, including order fulfillment blockage root cause analysis (RCA), show HEAR achieves up to 94.7% accuracy. Crucially, HEAR demonstrates adaptive efficiency: utilizing procedural hyperedges to minimize token costs, while leveraging topological exploration for rigorous correctness on complex queries. By matching proprietary model performance with open-weight backbones and automating manual diagnostics, HEAR establishes a scalable, auditable foundation for enterprise intelligence.

**arXiv ID:** 2605.14259
</details>

<details>
<summary><strong>Weasel: Out-of-Domain Generalization for Web Agents via Importance-Diversity Data Selection</strong> - Fatemeh Pesaran zadeh, Seyeon Choi, Xing Han Lù, Siva Reddy, Gunhee Kim - [[pdf]](https://arxiv.org/pdf/2605.20291)</summary>

**Abstract:** Large language models (LLMs) have enabled web agents that follow natural language goals through multi-step browser interactions. However, agents fine-tuned on specific trajectories and domain often struggle to generalize out of domain, and offline training can be compute-inefficient due to noisy, redundant trajectories and long accessibility-tree (AXTree) states. To address both issues, we propose Weasel, a trajectory selection method for offline training of web agents. Weasel selects a fixed-budget subset of trajectory steps by optimizing an objective that balances unary importance with pairwise diversity over states, websites, and interaction patterns, solving efficiently with a greedy algorithm. We further improve efficiency with target-centered AXTree pruning that keeps only content around the ground-truth action target, and we mitigate style mismatch for reasoning-native models by replacing expert traces with model-generated, style-consistent rationales. Across AgentTrek and NNetNav training datasets, evaluations in WebArena, WorkArena, and MiniWob, and experiments with Qwen2.5-7B, Gemma3-4B, and Qwen3-8B, Weasel improves out-of-domain performance while reducing training cost, producing roughly 9.7-12.5$\times$ training speedups over standard fine-tuning. We make the code available at this https URL.

**arXiv ID:** 2605.20291
</details>

<details>
<summary><strong>The Yes-Man Syndrome: Benchmarking Abstention in Embodied Robotic Agents</strong> - Doguhan Yeke, Elif Su Temirel, Ananth Shreekumar, Brandon Lee, Dongyan Xu, Z Berkay Celik - [[pdf]](https://arxiv.org/pdf/2605.20544)</summary>

**Abstract:** Vision-language models (VLMs) are used as high-level planners for embodied agents, translating natural language instructions and visual observations into action plans. While prior work has studied abstention in LLMs, existing benchmarks are largely text-only and do not capture the perceptual grounding and physical constraints inherent to embodied robotics environments. In such settings, abstention requires recognizing when instructions are ambiguous, physically infeasible, based on false premises, or otherwise unresolvable given the available sensory modalities and context. To address this gap, we introduce a taxonomy to categorize abstention in the context of embodied robotics and present RoboAbstention, a scalable and auditable framework for generating abstention instructions grounded in images gathered from five robotics datasets. RoboAbstention instantiates the taxonomy through a three-phase pipeline: (1) structured visual grounding, (2) deterministic constraint derivation, and (3) controlled instruction generation via category-specific templates. This enables the construction of a diverse dataset with verifiable abstention conditions. We evaluate several frontier VLMs and find that all models exhibit significant weaknesses in abstention, including those with advanced reasoning capabilities. The best-performing model, Gemini 2.5 Flash, abstains on only 39.0% of our 6,069 benchmark instructions, while the embodied planner Gemini Robotics ER 1.6 Preview abstains on just 16.5%. We further explore methods for improving abstention in VLM planners, such as defensive prompting and in-context learning, and find that these interventions substantially improve performance, reaching 93.6% abstention rate for Gemini Robotics ER 1.6 Preview and 88.6% for GPT 5.4 Mini, yet no approach fully solves the problem. We open-source RoboAbstention at this https URL.

**arXiv ID:** 2605.20544
</details>

<details>
<summary><strong>Benchmarking Empirical and Learning-Based Approaches for Feedforward Steering Control in Autonomous Racing</strong> - Georg Jank, Mattia Piccinini, Sebastian Wenk, Phillip Pitschi, Johannes Betz, Boris Lohmann - [[pdf]](https://arxiv.org/pdf/2605.21111)</summary>

**Abstract:** Feedforward steering control is a key component of hierarchical control architectures for autonomous racing. The goal is to reduce steering corrections from the feedback controllers by predicting the vehicle's inverse lateral dynamics. This paper presents a systematic benchmark of two learning-based and two empirical (analytical) feedforward steering controllers. We introduce a new \acf{ehd} formulation based on a polynomial surface fit that captures velocity-dependent nonlinear steering behavior with minimal parametrization. We test the feedforward controllers in a high-fidelity simulation framework based on the real-world Abu Dhabi Autonomous Racing League competition, using a high-fidelity double-track vehicle dynamics simulator. Open-loop evaluation shows that the learning-based controllers achieve the lowest prediction errors; however, closed-loop testing reveals that this improved accuracy does not translate into superior path tracking performance or lap times, even after iterative fine-tuning. In contrast, the proposed EHD approach achieves the best overall closed-loop robustness and lap time, highlighting the necessity of evaluating feedforward strategies within the complete trajectory planning and control software stack. Our code is available at this https URL.

**arXiv ID:** 2605.21111
</details>

</details>

<details open>
<summary><h2>LLM Agents (11 papers)</h2></summary>

<details>
<summary><strong>AgentAtlas: Beyond Outcome Leaderboards for LLM Agents</strong> - Parsa Mazaheri, Kasra Mazaheri - [[pdf]](https://arxiv.org/pdf/2605.20530)</summary>

**Abstract:** Large language model agents now act on codebases, browsers, operating systems, calendars, files, and tool ecosystems, but the benchmarks used to evaluate them are fragmented: each emphasizes a different unit of measurement (final task success, tool-call validity, repeated-pass consistency, trajectory safety, or attack robustness). A line of 2024-2025 work has converged on the diagnosis that a single accuracy column is no longer the right unit of comparison for deployable agents. AgentAtlas extends this line of work with four components: (i) a six-state control-decision taxonomy (Act / Ask / Refuse / Stop / Confirm / Recover); (ii) a nine-category trajectory-failure taxonomy with two orthogonal hierarchical labels (primary_error_source, impact); (iii) a taxonomy-aware vs. taxonomy-blind methodology that measures how much of a model's apparent capability comes from the supervision in the prompt; and (iv) a benchmark-coverage audit mapping fifteen agent benchmarks against six behavioral axes. To demonstrate the methodology we run a small fixed eight-model set (1,342 generated items, four frontier closed and four open-weight) under both prompt modes. Removing the explicit label menu drops every model's trajectory accuracy by 14-40 pp to a tight 0.54-0.62 floor regardless of family, and no single model wins on all three of control accuracy, trajectory diagnosis, and tool-context utility retention. We treat the synthetic run as a measurement-protocol demonstration, not a benchmark release.

**arXiv ID:** 2605.20530
</details>

<details>
<summary><strong>Governance by Construction for Generalist Agents</strong> - Segev Shlomov, Iftach Shoham, Alon Oved, Ido Levy, Sami Marreed, Harold Ship, Offer Akrabi, Sergey Zeltyn, Avi Yaeli, Nir Mashkif - [[pdf]](https://arxiv.org/pdf/2605.20874)</summary>

**Abstract:** Enterprise agents are increasingly expected to operate autonomously across tools and interfaces, yet production deployments require governance by construction. Systems must specify which actions are allowed, when human oversight is required, and what information may be exposed, without rebuilding the agent for each domain. This demo presents CUGA's policy system, a modular policy-as-code layer that composes with a generalist LLM agent to deliver predictable, auditable, and compliance-aware behavior in compound workflows without model fine-tuning. We present a runtime governance architecture that enforces policy interventions at every critical stage of execution. Rather than passively constraining behavior, policies intercept the agent at five structural checkpoints: upstream of planning (Intent Guard), within the system prompt to steer reasoning (Playbook), at the tool-call boundary to enforce proper usage (Tool Guide), outside the reasoning loop as a Human-in-the-Loop gate for high-risk actions (Tool Approvals), and at the output stage to filter and structure the final response (Output Formatter). Together, these stages embed governance continuously across the agent's execution pipeline rather than treating it as an afterthought. Using a healthcare scenario and a multi-layered enforcement intervention, the demo shows dynamic playbook injection for structured tool-sequence enforcement, intent guards that block malicious or accidental harmful requests, and human-in-the-loop tool approval checkpoints for potentially destructive actions. The artifact illustrates how typed governance primitives enable faster, safer deployment of enterprise agentic systems while improving policy adherence and execution consistency.

**arXiv ID:** 2605.20874
</details>

<details>
<summary><strong>An Application-Layer Multi-Modal Covert-Channel Reference Monitor for LLM Agent Egress</strong> - Alfredo Metere - [[pdf]](https://arxiv.org/pdf/2605.20734)</summary>

**Abstract:** A large language model (LLM) agent that sends messages can leak data inside them. Destination allowlists and content scanners do not police whether an otherwise-benign payload is itself a covert channel: a compromised agent encodes bits in zero-width characters, homoglyphs, whitespace, base64, JavaScript Object Notation (JSON) key ordering, message timing or size -- and, in binary egress, in least-significant-bit (LSB) pixel planes, per-image mean luminance, inter-image sequence permutation, ultrasonic tones, or audible-band sonified data. Our egress reference monitor has three contributions. (i) A text pipeline of ten capacity-reducing stages, a per-sink leaky-bucket capacity ledger, and a staged posture that enforces lossless stages from day one. (ii) Two media scramblers (a Fourier-domain audio band-limiter and a red-green-blue (RGB) image bit-depth and mean-luminance bucketer) gated by a boot-time cryptographic legitimacy attestation: an auditor publishes at boot the trusted Ed25519 keys and {kind, data-class} pairs; only payloads with a verifying signature for an authorized class are exempt. The attestation sidesteps the intractable content-based discrimination between real media and data sonified or rasterized as a carrier; unsigned media is suspect by default; a content-addressed canonicalizer closes the inter-image permutation channel. (iii) Residual capacity is the Miller--Madow corrected mutual information between embedded and recovered bits (zero when destroyed), measured by an adversarial ensemble of fifteen working encoders across text, image and audio. The reference implementation drives residual capacity to zero on every destroyable channel and to a stated bound on the one (per-image mean luminance) that cannot be destroyed without ruining the image.

**arXiv ID:** 2605.20734
</details>

<details>
<summary><strong>Causal Past Logic for Runtime Verification of Distributed LLM Agent Workflows</strong> - Benedikt Bollig - [[pdf]](https://arxiv.org/pdf/2605.20923)</summary>

**Abstract:** Distributed LLM agent workflows should not be monitored as if they produced a single sequential log. In an asynchronous execution, a decision can only depend on events that are causally visible to the lifeline that makes it: an event that appears earlier in some log may still be unknown locally. We extend the ZipperGen agent-workflow framework with Causal Past Logic (CPL), a small past-time temporal logic for guards in conditionals and while loops. In addition to standard past-time modalities such as previous and since, a guard can inspect the latest causally visible event of another lifeline and selected variables stored there. The formula is a source-level guard: it is evaluated online by the owner lifeline and can influence control flow at runtime. We give a vector-clock monitor with latest-value views and prove that the locally computed monitor value coincides with the denotational semantics of the guard at the current event. Thus runtime verification becomes part of the coordination language itself, rather than a post-hoc check over an execution log.

**arXiv ID:** 2605.20923
</details>

<details>
<summary><strong>APEX: Autonomous Policy Exploration for Self-Evolving LLM Agents</strong> - Yibo Li, Jiashuo Yang, Zhi Zheng, Zhiyuan Hu, Yuan Sui, Shizun Wang, Yufei He, Bryan Hooi - [[pdf]](https://arxiv.org/pdf/2605.21240)</summary>

**Abstract:** LLM agents have shown strong performance across a wide range of complex tasks, including interactive environments that require long-horizon decision making. But these agents cannot learn on the fly at test time. Self-evolving agents address this by accumulating memory and reflection across episodes rather than requiring model-weight updates. However, these agents often suffer from exploration collapse: as memory grows, behavior concentrates around familiar high-reward routines, reducing the chance of discovering better alternatives. To address this problem, we propose Autonomous Policy EXploration (APEX), which builds and maintains an explicit strategy space through a strategy map-a directed acyclic graph of milestones with prerequisite dependency edges. In APEX, Fork Discovery expands the map with evidence-grounded unexplored directions, while Policy Selection balances exploration and exploitation during planning. Evaluated on nine Jericho text-adventure games and WebArena, a realistic web interaction benchmark, APEX outperforms all baselines. Extensive ablations validate each component's contribution and demonstrate robustness across diverse settings, demonstrating APEX's effectiveness for sustained exploration in self-evolving agents.

**arXiv ID:** 2605.21240
</details>

<details>
<summary><strong>Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs</strong> - Haiquan Lu, Zigeng Chen, Gongfan Fang, Xinyin Ma, Xinchao Wang - [[pdf]](https://arxiv.org/pdf/2605.20315)</summary>

**Abstract:** LLM agents have recently emerged as a powerful paradigm for solving complex tasks through planning, tool use, memory retrieval, and multi-step interaction. However, these agentic workflows often introduce substantial input-side overhead, making the compute-intensive prefilling stage a key bottleneck in long-context, multi-turn inference. In this work, we propose Mix-Quant, a simple and effective phase-aware quantization framework for fast agentic inference. We first investigate FP4 quantization in agentic LLM workflows and observe that quantizing the entire inference process can incur significant performance degradation. In contrast, the prefilling stage exhibits substantial quantization redundancy and can therefore be quantized with minimal accuracy loss, despite being the dominant source of computation. Based on this insight, we apply high-throughput NVFP4 quantization to the prefilling phase while preserving BF16 precision for decoding. By decoupling prefilling acceleration from decoding quality, Mix-Quant combines phase-aware algorithmic quantization with hardware-efficient NVFP4 execution to alleviate the inference bottleneck in LLM agents. Extensive experiments across long-context and agentic benchmarks demonstrate that Mix-Quant largely preserves task performance while delivering significant efficiency improvements, achieving up to a 3x speedup during prefilling.

**arXiv ID:** 2605.20315
</details>

<details>
<summary><strong>MemGym: a Long-Horizon Memory Environment for LLM Agents</strong> - Wujiang Xu, Yu Wang, Kai Mei, Kaiqu Liang, Zhenting Wang, Mingyu Jin, Han Zhang, Shi-Xiong Zhang, Wenyue Hua, Sambit Sahu, Dimitris N. Metaxas - [[pdf]](https://arxiv.org/pdf/2605.20833)</summary>

**Abstract:** Memory is a central capability for LLM agents operating across long-horizon tasks. Existing memory benchmarks predominantly evaluate retention of personalized information in multi-turn chat scenarios, overlooking the dynamic memory formation that occurs during extended agent execution. Consequently, the memory systems they produce transfer poorly to realistic agentic environments, such as coding and web navigation. We present MemGym, a benchmark for agentic memory that unifies existing agent gyms and in-house memory-grounded pipelines behind one memory-reasoning interface. MemGym spans five evaluation tracks grouped into four agentic regimes: tool-use dialogue (tau2-bench), multi-turn deep-research search (MEMGYM-DR), coding (SWE-Gym and MEMGYM-CODEQA), and computer use (WebArena-Infinity). MemGym reports memory-isolated scores that decouple memory performance from reasoning, retrieval, and tool-use ability, so memory strategies can be ranked without those confounders. Our synthetic pipelines for MEMGYM-CODEQA and MEMGYM-DR are length-controllable, ablation-verified at every stage, and tightly aligned with downstream scenarios. To make evaluation on coding environments academically tractable, we train MemRM, a lightweight reward model (Qwen3-1.7B fine-tuned with QLoRA) that scores compression quality as a fast scalar read in place of full Docker rollouts.

**arXiv ID:** 2605.20833
</details>

<details>
<summary><strong>From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents</strong> - Md Tahmid Rahman Laskar, Xue-Yong Fu, Seyyed Saeed Sarfjoo, Quinten McNamara, Jonas Robertson, Shashi Bhushan TN - [[pdf]](https://arxiv.org/pdf/2605.15104)</summary>

**Abstract:** Voice agents increasingly require reliable tool use from speech, whereas prominent tool-calling benchmarks remain text-based. We study whether verified text benchmarks can be converted into controlled audio-based tool calling evaluations without re-annotating the tool schema and gold labels. Our dataset-agnostic framework uses text-to-speech, speaker variation, and environmental noise to create paired text-audio instances while preserving the original dataset annotations. Based on extensive evaluation of 7 omni-modal models on audio-converted versions of Confetti and When2Call, our framework demonstrates that the performance is strongly model- and task-dependent: Gemini-3.1-Flash-Live obtains the highest Confetti score (70.4), whereas GPT-Realtime-1.5 performs best on When2Call (71.9). On Confetti, the text-to-voice gap ranges from 1.8 points for Qwen3-Omni to 4.8 points for GPT-Realtime-1.5. A targeted analysis of failure cases demonstrates that degradations most often reflect misunderstandings of argument values in the speech. Considering real-world deployment scenarios, we further report text-only results, an ambiguity-based reformulation stress test, and a reference-free LLM-as-judge protocol validated against human preferences. Notably, we find that open-source Qwen3 judges with at least 8B parameters exceed 80% agreement with proprietary judges, supporting privacy-preserving evaluation. Overall, our framework provides a verifiable and reproducible first-stage diagnostic that complements purpose-built audio corpora.

**arXiv ID:** 2605.15104
</details>

<details>
<summary><strong>Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations?</strong> - Anvesh Rao Vijjini, Sagar Manjunath, Snigdha Chaturvedi - [[pdf]](https://arxiv.org/pdf/2605.17694)</summary>

**Abstract:** Power differences shape human communication through well documented socio cognitive effects, including language coordination, pronoun usage, authority bias, and harmful compliance. We examine whether large language models (LLMs) exhibit similar behaviors when assigned high or low status personas. Using personas from diverse professions, we simulate multi turn, power asymmetric dialogues (e.g., principal teacher, justice lawyer) and measure (i) language coordination, (ii) pronoun usage, (iii) persuasion success, and (iv) compliance with unsafe requests. Our results show that LLMs show key socio-cognitive effects of power, albeit with nuances and variability, linking simulated interactions to both desirable and unsafe behaviors.

**arXiv ID:** 2605.17694
</details>

<details>
<summary><strong>What Twelve LLM Agent Benchmark Papers Disclose About Themselves: A Pilot Audit and an Open Scoring Schema</strong> - Mahdi Naser Moghadasi, Faezeh Ghaderi - [[pdf]](https://arxiv.org/pdf/2605.21404)</summary>

**Abstract:** We read twelve well-known LLM agent benchmark papers and recorded, dimension by dimension, what each paper actually says about how its evaluation was run. The motivation came from a familiar frustration: two papers will report results on the same benchmark with the same model name and disagree, and you cannot tell why -- the scaffold, the sampling settings, the subset, or the evaluator version. In many cases the published artifact does not let you answer. This paper is an implementation report on the attempt. We designed a small audit schema (five fields: benchmark identity, harness specification, inference settings, cost reporting, failure breakdown), wrote a scoring codebook with the boundary cases we hit during pilot scoring, applied it to twelve canonical papers (eight agent, four classical static), and recorded what we saw. We score the disclosure of an agent run, not its correctness, and make no claim that disclosure implies a trustworthy result. The mean audit score across the eight agent-benchmark papers is 0.38 (out of 1.0), and across the four classical static benchmarks 0.66; the largest gap is on cost (none of the eight agent benchmark papers disclose inference cost in any form) and on harness specification (none fully disclose a content-addressed container image of the evaluation environment). We release the schema as a JSON Schema file, the codebook as a Markdown document, and the raw scoring sheet as a CSV. The scoring was performed by a single auditor in one pass; a multi-rater audit is the natural next step, and we discuss what we think it would change.

**arXiv ID:** 2605.21404
</details>

<details>
<summary><strong>Toward User Comprehension Supports for LLM Agent Skill Specifications</strong> - Zikai Alex Wen - [[pdf]](https://arxiv.org/pdf/2605.19362)</summary>

**Abstract:** Users often interpret and select agent skills through their SKILL markdown specifications. To protect users, existing audits mainly focus on malicious or unsafe skills. We study the complementary question of whether specifications help users form bounded expectations about what a skill consumes, produces, and covers. Across 878 cybersecurity skills, we used rule-based coding to measure textual cues for four comprehension anchors, namely operational basis, output contract, boundary disclosure, and example capability demonstration. Cues for operational basis were common, but only 19.0% of specifications exhibited cues for an example task, sample, or expected outcome, and only 2.3% exhibited cues for all four anchors. We further examined a small DNS/C2 telemetry subset (n$=$6) to illustrate why missing examples may matter. Examples appeared to make first local checks easier to construct, while no-example skills typically required helper code inspection to recover command arguments or output fields. We argue that agent-skill evaluation should treat specifications as user-facing capability disclosures, not merely as containers for executable instructions.

**arXiv ID:** 2605.19362
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (20 papers)</h2></summary>

<details>
<summary><strong>AgentCo-op: Retrieval-Based Synthesis of Interoperable Multi-Agent Workflows</strong> - Shuaike Shen, Wenduo Cheng, Shike Wang, Mingqian Ma, Jian Ma - [[pdf]](https://arxiv.org/pdf/2605.20425)</summary>

**Abstract:** Designing multi-agent workflows is especially difficult in open-ended scientific settings where tasks lack curated training sets, reliable scalar evaluation metrics, and standardized interfaces between existing tools and agents. We propose AgentCo-op, a retrieval-based synthesis framework that composes reusable skills, tools, and external agents into executable workflows through typed artifact handoffs, then applies bounded self-guided local repair to implicated components when execution evidence indicates failure. In two open-world genomics case studies, AgentCo-op composes independently developed scientific agents and external tool repositories into auditable workflows without redesigning them or running global topology search. It coordinates specialized agents for spatial transcriptomics and gene-set interpretation to enable collaborative discovery from spatial transcriptomics data, and builds a parallel workflow for cross-modality marker analysis on single-cell multiome data. AgentCo-op can also import a searched workflow as a structural prior and improve it by grounding nodes with retrieved components and applying local repair, showing that synthesis and search are complementary. On six coding, math, and question-answering benchmarks, AgentCo-op achieves the best result on four benchmarks and the best average score under a unified backbone setting, while consistently reducing per-task cost relative to multi-agent baselines. Together, these results suggest that retrieval-based synthesis can extend automated agentic workflow design beyond benchmark-optimized agent graphs to open-world workflows built from existing agents, tools, and typed artifacts.

**arXiv ID:** 2605.20425
</details>

<details>
<summary><strong>From Automated to Autonomous: Hierarchical Agent-native Network Architecture (HANA)</strong> - Binghan Wu, Shoufeng Wang, Yunxin Liu, Ya-Qin Zhang, Joseph Sifakis, Ye Ouyang - [[pdf]](https://arxiv.org/pdf/2605.20608)</summary>

**Abstract:** Realizing Level 4/5 Autonomous Networks (AN) demands a shift from static automation to agent-native intelligence. Current operations, reliant on rigid scripts, lack the cognitive agency to handle off-nominal conditions. To address this, this letter proposes a hierarchical multi-agent reference architecture enabling high-level autonomy. The framework features a Dual-Driven Orchestrator that coordinates specialized Executive Agents, supported by a shared Public Memory for unified domain knowledge. A key innovation is the integration of agent self-awareness, which empowers the system to harmonize deliberative strategic governance with reflexive fault recovery. We instantiate and validate this architecture within a 5G Core environment. Case studies demonstrate that the system sustains critical throughput under congestion and reduces Mean Time to Repair (MTTR) by 86%, confirming its efficacy in unifying strategic planning with operational resilience.

**arXiv ID:** 2605.20608
</details>

<details>
<summary><strong>COAgents: Multi-Agent Framework to Learn and Navigate Routing Problems Search Space</strong> - Oleksandr Yakovenko, Mahdi Mostajabdaveh, Cheikh Ahmed, Abdullah Ali Sivas, Xiaorui Li, Zirui Zhou, Mao Kun - [[pdf]](https://arxiv.org/pdf/2605.20618)</summary>

**Abstract:** Although Vehicle Routing Problems (VRP) are essential to many real-world systems, they remain computationally intractable at scale due to their combinatorial complexity. Traditional heuristics rely on handcrafted rules for local improvements and occasional \textit{jumps} to escape local minima, but often struggle to generalize across diverse instances. We introduce \textbf{COAgents}, a cooperative multi-agent framework that models the search process as a graph: nodes represent solutions, and edges correspond to either local refinements or large perturbations for diversification (i.e., jumps). A \textit{Partial Search Graph} (PSG) is dynamically constructed during search, enabling COAgents to train a Node Selection Agent and a Move Selection Agent to guide intensification, and a Jump Agent to trigger well-timed explorations of new regions. Unlike end-to-end learning approaches, COAgents cleanly separates problem-agnostic search control from compact domain-specific encoding, facilitating adaptability across tasks. Extensive experiments on the CVRP and VRPTW benchmarks show that COAgents remains competitive with several learn-to-search baselines on CVRP and sets a new state of the art among learning-based methods on the more challenging VRPTW instances, reducing the gap to the best-known solutions by 14\% at $N\!=\!100$ and 44\% at $N\!=\!50$ relative to the strongest neural solver (POMO), and by 21\% and 40\% respectively relative to ALNS.
Code is available at this https URL.

**arXiv ID:** 2605.20618
</details>

<details>
<summary><strong>Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents</strong> - Akshay Manglik, Apaar Shanker, Kaustubh Deshpande, Jason Qin, Yash Maurya, Veronica Chatrath, Vijay S. Kalmath, Levi Lentz, Yuan - [[pdf]](https://arxiv.org/pdf/2605.21347)</summary>

**Abstract:** Diagnosing failures in LLM agents remains largely manual. Practitioners inspect a small subset of execution traces, form ad-hoc hypotheses, and iterate. This process misses patterns that only emerge across trace populations and does not scale to production corpora where individual traces span tens of thousands of tokens. We formalize the problem of corpus-level trace diagnostics. Given a corpus of execution traces, the goal is to produce grounded natural-language insights that characterize systematic behavioral patterns across trace groups, each linked to supporting evidence. We present the Insights Generator (IG), a multi-agent system that answers diagnostic questions by proposing and testing hypotheses across the trace corpus to produce an evidence-backed insights report. We evaluate IG across qualitative and objective dimensions, spanning rubric-based report assessment and downstream performance improvements achieved by implementing IG insights. Human experts using IG reports improve scaffold performance by 30.4pp over the unmodified baseline scaffold, and coding agents leveraging IG-derived insights show consistent and stable gains. Across benchmarks, IG's scout-investigator architecture produces findings comparable in detection coverage to competing approaches, while domain experts rated IG reports as leading depth and evidence quality.

**arXiv ID:** 2605.21347
</details>

<details>
<summary><strong>Towards Resilient and Autonomous Networks: A BlueSky Vision on AI-Native 6G</strong> - Liang Wu, Kelly Wan, Mayank Darbari, Liangjie Hong - [[pdf]](https://arxiv.org/pdf/2605.21395)</summary>

**Abstract:** The proliferation of emerging applications, such as autonomous driving and immersive experiences, demands cellular networks that are not only faster, but fundamentally more resilient and autonomous. This paper presents a BlueSky vision on how Artificial Intelligence will be natively integrated into 6G, shifting the paradigm from \underline{Network for AI} to \underline{AI for Network}. We envision that, unlike 5G's reliance on scattered, ad-hoc models each trained for a single task, native AI in the 6G era will be anchored by a foundation model and and orchestrated via collaborative multi-agent systems, framing network management as a unified, multi-modal, multi-task optimization problem. Built on this vision, we outline two transformative directions. The first focuses on developing a 6G foundation model as a unified backbone, with task-specific knowledge distilled into compact models suited for diverse edge deployments. The second advances multi-agent systems designed to autonomously diagnose, maintain, and recover networks with minimal human intervention. These directions chart a roadmap for 6G to evolve into an intelligent, self-sustaining communication infrastructure.

**arXiv ID:** 2605.21395
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning for Safe Autonomous Driving Under Pedestrian Behavioral Uncertainty</strong> - Prakash Aryan, Kaushik Raghupathruni, Timo Kehrer, Sebastiano Panichella - [[pdf]](https://arxiv.org/pdf/2605.20255)</summary>

**Abstract:** Simulation-based testing of self-driving cars (SDCs) typically relies on scripted or simplified pedestrian models that do not capture the heterogeneity and uncertainty of real human crossing behavior. This limits the realism of safety assessments, especially in scenarios involving jaywalking, which is governed by latent personality traits that the vehicle cannot observe. We hypothesize that jointly training pedestrians and the SDC with multi-agent reinforcement learning (MARL) produces more realistic interaction scenarios than training the SDC against fixed pedestrian policies, and that the resulting behavior gap between predictable and unpredictable crossings can be measured directly from trajectories. This paper describes a MARL environment in which an SDC and 12 pedestrians are co-trained using Multi-Agent Proximal Policy Optimization (MAPPO). Pedestrian locomotion follows scripted Dijkstra pathfinding, while an RL policy controls high-level go/wait decisions. Jaywalking probability depends on a per-pedestrian personality trait sampled at episode start and hidden from the SDC. In 500-episode evaluations, the co-trained SDC reached 78% of goals with a 14% collision rate, compared to 35% goals and 33% collisions for the best rule-based baseline. A speed differential metric shows that the SDC traveled 2.65 m/s faster near jaywalkers than near crosswalk users at close range (0-3 m), indicating that jaywalking encounters were not anticipated. Jaywalking accounted for 13% of crossing events but was associated with 62% of collisions. Co-training with MARL pedestrians reduced collisions by 30% relative to single-agent RL, as pedestrians learned to wait when the SDC approached at speed.

**arXiv ID:** 2605.20255
</details>

<details>
<summary><strong>Modeling Emotional Dynamics in Agent-to-Agent Interactions on Moltbook</strong> - Syed Mhamudul Hasan, Abdur R. Shahid - [[pdf]](https://arxiv.org/pdf/2605.20442)</summary>

**Abstract:** Generative AI systems are increasingly deployed as interactive agents in online environments, such as a social network called Moltbook. In Moltbook, large-scale agentic AIs can post, comment, and engage in activities generated at scale by AI-driven text. Yet these agent behavioral characteristics remain insufficiently understood, particularly in complex, multi-agent interaction. In this study, we analyze the emotional dynamics of agent interactions within Moltbook. We construct an emotion-aware framework that maps textual interactions to a predefined set of fine-grained emotional categories, enabling the extraction of structured emotion profiles across agents and interaction contexts. To further evaluate behavioral reliability, we introduce an emotion-based domain called Persona-Stimulus-Reaction (PSR) that captures the alignment of emotional responses across similar contexts. Our analysis shows distinct emotional patterns and varying levels of behavioral stability across agents. Our analysis reveals that agents exhibit distinct emotional signatures with varying levels of behavioral stability influenced by interaction context.

**arXiv ID:** 2605.20442
</details>

<details>
<summary><strong>Multi-agent Collaboration with State Management</strong> - Mengyang Liu, Taozhi Chen, Zhenhua Xu, Xue Jiang, Yihong Dong - [[pdf]](https://arxiv.org/pdf/2605.20563)</summary>

**Abstract:** Recent advances in multi-agent systems have shown great potential for solving complex tasks. However, when multiple agents edit a shared codebase concurrently, their changes can silently conflict and inconsistent views lead to integration failures. Existing multi-agent systems address this through workspace isolation (e.g., one git worktree per agent), but this defers conflict resolution to a post-hoc merge step where recovery is expensive. In this paper, we propose STORM, i.e., STate-ORiented Management for multi-agent collaboration. Specifically, STORM manages agent states by mediating their interactions with the shared workspace, ensuring that each agent operates on a consistent view of the codebase and that conflicting edits are detected and resolved at write time. We evaluate STORM on Commit0 and PaperBench across multiple LLMs. STORM outperforms the git-worktree-based multi-agent baseline by +18.7 on Commit0-Lite and +1.4 on PaperBench, while achieving comparable or better cost efficiency. Combined with single-agent runs, STORM reaches highest scores of 87.6 and 78.2 on the two benchmarks respectively, suggesting that explicit state management is a more effective foundation for multi-agent collaboration than workspace isolation. STORM can also be plugged into any multi-agent system seamlessly.

**arXiv ID:** 2605.20563
</details>

<details>
<summary><strong>Enabling Regulatory Multi-Agent Collaboration: Architecture, Challenges, and Solutions</strong> - Qinnan Hu, Yuntao Wang, Yuan Gao, Zhou Su, Linkang Du, Qichao Xu - [[pdf]](https://arxiv.org/pdf/2509.09215)</summary>

**Abstract:** Large language models (LLMs)-empowered autonomous agents are transforming both digital and physical environments by enabling adaptive, multi-agent collaboration. While these agents offer significant opportunities across domains such as finance, healthcare, and smart manufacturing, their unpredictable behaviors and heterogeneous capabilities pose substantial governance and accountability challenges. In this paper, we propose a blockchain-enabled layered architecture for regulatory agent collaboration, comprising an agent layer, a blockchain data layer, and a regulatory application layer. Within this framework, we design three key modules: (i) an agent behavior tracing and arbitration module for automated accountability, (ii) a dynamic reputation evaluation module for trust assessment in collaborative scenarios, and (iii) a malicious behavior forecasting module for early detection of adversarial activities. Our approach establishes a systematic foundation for trustworthy, resilient, and scalable regulatory mechanisms in large-scale agent ecosystems. Finally, we discuss the future research directions for blockchain-enabled regulatory frameworks in multi-agent systems.

**arXiv ID:** 2509.09215
</details>

<details>
<summary><strong>What Do Agents Communicate? Characterizing Information Exchange in Multi-Agent Systems</strong> - Yong Jin Chun, Iftekhar Ahmed - [[pdf]](https://arxiv.org/pdf/2605.20548)</summary>

**Abstract:** Large Language Models (LLMs) have enabled collaborative Multi-Agent (MA) systems, where interacting agents improve performance through diverse reasoning and iterative refinement. However, these systems remain vulnerable to error propagation, where early-stage information degrades downstream reasoning. To address this, we conduct a systematic analysis of inter-agent communication to identify which information drives MA performance. We find that the absence of reasoning and verification in inter-agent communication significantly degrades performance. Based on these insights, we propose Category-Aware Recovery Augmentation (technique), which enforces the presence of critical information during communication. recovers up to 86.2% of failed cases. Our results highlight the key role of information quality in effective MA collaboration. Our code is available at this https URL

**arXiv ID:** 2605.20548
</details>

<details>
<summary><strong>Time-To-Reach Separation and Safety Filtering for Safe, Fair, and Efficient Multi-Agent Coordination</strong> - Matthew Low, Jasmine Jerry Aloor, Victoria Marie Tuck, Pierluigi Nuzzo, Jason J. Choi - [[pdf]](https://arxiv.org/pdf/2605.20625)</summary>

**Abstract:** Advanced Air Mobility (AAM) operations are expected to significantly increase aerial traffic in urban airspace, requiring autonomous traffic management systems to ensure collision-free operations in highly congested environments. In this paper, we propose a multi-agent coordination framework that uses minimum time-to-reach (TTR) as a unifying metric for priority assignment, temporal separation, and safety filtering. We focus on the problem of coordinating multiple aerial vehicles merging into an air corridor while maintaining safe separation between vehicles. Vehicles are assigned arrival-consistent priority based on TTR, and target TTR values are used to enforce temporal spacing that induces spatial separation. A priority-consistent safety filtering layer based on Hamilton-Jacobi reachability value functions ensures collision avoidance while minimally modifying the reference guidance. Simulation results in a highly congested corridor merging scenario show that the proposed method improves safety, fairness, and efficiency compared to time-optimal guidance and priority-agnostic safety filtering.

**arXiv ID:** 2605.20625
</details>

<details>
<summary><strong>DRAMA: Next-Gen Dynamic Orchestration for Resilient Multi-Agent Ecosystems in Flux</strong> - Xinkui Zhao, Yifan Zhang, Sai Liu, Naibo Wang, Guanjie Cheng, Yueshen Xu, Chang Liu, Shuiguang Deng, Jianwei Yin - [[pdf]](https://arxiv.org/pdf/2508.04332)</summary>

**Abstract:** Multi-agent systems (MAS) have demonstrated significant effectiveness in addressing complex problems through coordinated collaboration among heterogeneous agents. However, real-world environments and task specifications are inherently dynamic, characterized by frequent changes, uncertainty, and variability. Despite this, most existing MAS frameworks rely on static architectures with fixed agent capabilities and rigid task allocation strategies, which greatly limits their adaptability to evolving conditions. This inflexibility poses substantial challenges for sustaining robust and efficient multi-agent cooperation in dynamic and unpredictable scenarios. To address these limitations, we propose DRAMA: a Dynamic and Robust Allocation-based Multi-Agent System designed to facilitate resilient collaboration in rapidly changing environments. DRAMA features a modular architecture with a clear separation between the control plane and the worker plane. Both agents and tasks are abstracted as resource objects with well-defined lifecycles, while task allocation is achieved via an affinity-based, loosely coupled mechanism. The control plane enables real-time monitoring and centralized planning, allowing flexible and efficient task reassignment as agents join, depart, or become unavailable, thereby ensuring continuous and robust task execution. The worker plane comprises a cluster of autonomous agents, each with local reasoning, task execution, the ability to collaborate, and the capability to take over unfinished tasks from other agents when needed.

**arXiv ID:** 2508.04332
</details>

<details>
<summary><strong>Learning Incentive Structures for Cooperative Resilience in Multi-Agent Systems under Social Dilemmas</strong> - Manuela Chacon-Chamorro, Luis Felipe Giraldo, Nicanor Quijano - [[pdf]](https://arxiv.org/pdf/2601.22292)</summary>

**Abstract:** Multi-agent social dilemmas, such as the tragedy of the commons, capture settings where individual incentives conflict with collective well-being, making these systems highly vulnerable to collapse under disruptions. In this context, this work studies cooperative resilience, understood as the system-level ability to maintain collective well-being under perturbations through adaptive agent behavior. We propose a framework for learning incentive structures aligned with collective well-being in multi-agent reinforcement learning systems, where reward functions shape individual decision-making and collective behavior. A resilience metric is used to score and rank agent trajectories, allowing the inference of reward functions that promote resilient collective behavior. These inferred reward functions are integrated into the multi-agent reinforcement learning process to shape agent interactions in social dilemma settings. The approach is evaluated in resource-sharing environments subject to disruptions, using three incentive structures: individual incentives, resilience-aligned incentives, and a hybrid incentive structure that combines both individual and collective components. The results show that the hybrid incentive structure promotes sustained collective behavior, reduces collapse events associated with resource depletion, and preserves system performance under disruption. These findings highlight the role of incentive design as a mechanism for promoting resilient collective behavior and provide a computational framework for multi-agent social dilemmas under disruptions.

**arXiv ID:** 2601.22292
</details>

<details>
<summary><strong>Distributed Non-Uniform Scaling Control of Multi-Agent Formation via Matrix-Valued Constraints</strong> - Tao He, Gangshan Jing - [[pdf]](https://arxiv.org/pdf/2508.02289)</summary>

**Abstract:** Distributed formation maneuver control refers to the problem of maneuvering a group of agents to change their formation shape by adjusting the motions of partial agents, where the controller of each agent only requires local information measured from its neighbors. Although this problem has been extensively investigated, existing approaches are mostly limited to uniform scaling transformations. This article proposes a new type of local matrix-valued constraints, via which non-uniform scaling control of position formation can be achieved by tuning the positions of only two agents (i.e., leaders). Here, the non-uniform scaling transformation refers to global scaling the position formation with different ratios along different orthogonal coordinate directions. Moreover, by defining scaling and translation of attitudes, we propose a distributed control scheme for scaling and translation maneuver control of joint position-attitude formations. It is proven that the proposed controller achieves global convergence, provided that the sensing graph among agents is a 2-rooted bidirectional graph. Compared with the affine formation maneuver control approach, the proposed approach leverages a sparser sensing graph, requires fewer leaders, and additionally enables scaling transformations of the attitude formation. A simulation example demonstrates our theoretical results.

**arXiv ID:** 2508.02289
</details>

<details>
<summary><strong>FACET: Teacher-Centred LLM-Based Multi-Agent Systems-Towards Personalized Educational Worksheets</strong> - Jana Gonnermann-Müller, Jennifer Haase, Konstantin Fackeldey, Sebastian Pokutta - [[pdf]](https://arxiv.org/pdf/2508.11401)</summary>

**Abstract:** The increasing heterogeneity of student populations poses significant challenges for teachers, particularly in mathematics education, where cognitive, motivational, and emotional differences strongly influence learning outcomes. While AI-driven personalization tools have emerged, most remain performance-focused, offering limited support for teachers and neglecting broader pedagogical needs. This paper presents the FACET framework, a teacher-facing, large language model (LLM)-based multi-agent system designed to generate individualized classroom materials that integrate both cognitive and motivational dimensions of learner profiles. The framework comprises three specialized agents: (1) learner agents that simulate diverse profiles incorporating topic proficiency and intrinsic motivation, (2) a teacher agent that adapts instructional content according to didactical principles, and (3) an evaluator agent that provides automated quality assurance. We tested the system using authentic grade 8 mathematics curriculum content and evaluated its feasibility through a) automated agent-based assessment of output quality and b) exploratory feedback from K-12 in-service teachers. Results from ten internal evaluations highlighted high stability and alignment between generated materials and learner profiles, and teacher feedback particularly highlighted structure and suitability of tasks. The findings demonstrate the potential of multi-agent LLM architectures to provide scalable, context-aware personalization in heterogeneous classroom settings, and outline directions for extending the framework to richer learner profiles and real-world classroom trials.

**arXiv ID:** 2508.11401
</details>

<details>
<summary><strong>CTFExplorer: Evaluating LLM Offensive Agents Through Multi-Target Web CTF Benchmarking</strong> - Nanda Rani, Kimberly Milner, Minghao Shao, Meet Udeshi, Haoran Xi, Venkata Sai Charan Putrevu, Saksham Aggarwal, Sandeep K. Shukla, Prashanth Krishnamurthy, Farshad Khorrami, Muhammad Shafique, Ramesh Karri - [[pdf]](https://arxiv.org/pdf/2602.08023)</summary>

**Abstract:** Existing benchmarks for LLM-based offensive security agents use isolated, single-target setups with a known vulnerable service and fixed objective. They measure exploitation effectively, but miss how real Capture-the-Flag (CTF) participants triage unknown surfaces, prioritize targets, and allocate effort under uncertainty. Current evaluations therefore fail to assess strategic reasoning beyond exploitation alone. To address this, we introduce \textit{CTFExplorer}, a benchmark suite that shifts offensive security evaluation toward a multi-target setting, which tests how agents explore, prioritize, and chain attacks. CTFExplorer deploys 40 web-based vulnerable services within a single environment, where agents must autonomously discover, distinguish, and exploit targets without predefined guidance. We also present a reactive multi-agent setup as a reference agent framework and develop an agent-agnostic evaluation framework that records structured reasoning traces for fine-grained assessment. This enables behavioral evaluation beyond binary flag capture, such as how agents manage target selection, handle failed hypotheses, coordinate across multiple stages, and extract security intelligence.

**arXiv ID:** 2602.08023
</details>

<details>
<summary><strong>MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing</strong> - Yang Liu, Jinxuan Cai, Yishen Li, Qi Meng, Zedi Liu, Xin Li, Chen Qian, Chuan Shi, Cheng Yang - [[pdf]](https://arxiv.org/pdf/2603.06007)</summary>

**Abstract:** Large language model-based (LLM-based) multi-agent systems (MAS) are increasingly used to extend agentic problem solving via role specialization and collaboration. MAS workflows can be naturally modeled as directed computation graphs, where nodes execute agents or sub-workflows and edges encode dependencies and message passing. However, implementing complex graph workflows in current frameworks still requires substantial manual effort, offers limited reuse, and makes it difficult to integrate heterogeneous external context sources. To overcome these limitations, we present MASFactory, a graph-centric framework for orchestrating LLM-based MAS. It introduces Vibe Graphing, a human-in-the-loop approach that compiles natural-language intent into an editable workflow specification and then into an executable graph. In addition, the framework provides reusable components, skill support, multimodal message handling, and pluggable context integration, as well as a visualizer for topology preview, runtime tracing, and human-in-the-loop interaction. We evaluate MASFactory on seven public benchmarks, validating both reproduction consistency for representative MAS methods and the effectiveness of Vibe Graphing. Our code (this https URL, licensed under Apache-2.0) and video demonstration (this https URL) are publicly available.

**arXiv ID:** 2603.06007
</details>

<details>
<summary><strong>STEAM: A Training-Free Congestion-Aware Enhancement Framework for Decentralized Multi-Agent Path Finding</strong> - Mingyang Feng, Mengnuo Zhang, Shaoyuan Li, Xiang Yin - [[pdf]](https://arxiv.org/pdf/2605.20929)</summary>

**Abstract:** We propose STEAM (Spatial, Temporal, and Emergent congestion Awareness for MAPF), a training-free test-time enhancement framework for learning-based decentralized Multi-Agent Path Finding (MAPF) in discrete environments. Given a pretrained decentralized policy, STEAM requires no retraining, architectural modification, or replacement by a centralized planner. Instead, it injects lightweight congestion-aware guidance into the original policy execution. STEAM first rolls out the shortest paths induced by the current cost-to-go maps to identify potential future congestion hotspots. Spatially avoidable congestion is mitigated by updating agent-specific cost-to-go information, while spatially unavoidable bottlenecks are handled through temporal logit correction. In addition, emergent local congestion is reduced by a density-aware logit correction based on neighboring agents' corrected cost-to-go maps. Extensive experiments on representative learning-based decentralized MAPF algorithms show that STEAM consistently improves success rate, makespan, and solution cost, with success-rate gains of up to 60% and only minor computational overhead. The implementation is available at this https URL.

**arXiv ID:** 2605.20929
</details>

<details>
<summary><strong>Hyper-V2X: Hypernetworks for Estimating Epistemic and Aleatoric Uncertainty in Cooperative Bird's-Eye-View Semantic Segmentation</strong> - Abhishek Dinkar Jagtap, Sanath Tiptur Sadashivaiah, Andreas Festag - [[pdf]](https://arxiv.org/pdf/2605.21309)</summary>

**Abstract:** Cooperative perception enabled by Vehicle-to-Everything (V2X) communication enhances autonomous driving safety by creating a unified environmental representation through shared sensory data. While recent works have advanced multi-agent fusion for improved perception, uncertainty quantification in such cooperative frameworks remains largely unexplored. This paper introduces Hyper-V2X, a hypernetwork-based framework for estimating both epistemic and aleatoric uncertainties in V2X-based perception. Specifically, we propose a partial weight generation scheme and V2X context embedding module that conditions a Bayesian hypernetwork on fused multi-agent features to generate weight distributions for stochastic Bird's-Eye-View (BEV) segmentation. Unlike existing deterministic BEV models, Hyper-V2X enables efficient uncertainty estimation with little computation overhead. Our approach is architecture-agnostic, and can be seamlessly integrating with modern cooperative backbones such as CoBEVT. Experiments on the OPV2V benchmark demonstrate that Hyper-V2X provides accurate, well-calibrated uncertainty estimates and improves overall perception reliability. Our code and benchmark are publicly available under an open-source license: this https URL

**arXiv ID:** 2605.21309
</details>

<details>
<summary><strong>MAPLE: Latent Multi-Agent Play for End-to-End Autonomous Driving</strong> - Rajeev Yasarla, Deepti Hegde, Hsin-Pai Cheng, Shizhong Han, Yunxiao Shi, Meysam Sadeghigooghari, Hanno Ackermann, Litian Liu, Pranav Desai, Fatih Porikli, Mohammad Ghavamzadeh, Hong Cai - [[pdf]](https://arxiv.org/pdf/2605.14201)</summary>

**Abstract:** Vision-language-action (VLA) models are effective as end-to-end motion planners, but can be brittle when evaluated in closed-loop settings due to being trained under traditional imitation learning framework. Existing closed-loop supervision approaches lack scalability and fail to completely model a reactive environment. We propose MAPLE, a novel framework for reactive, multi-agent rollout of a dynamic driving scenario in the latent space of the VLA model. The ego vehicle and nearby traffic agents are independently controlled over multi-step horizons, while being reactive to other agents in the scene, enabling closed-loop training. MAPLE consists of two training stages: (1) supervised fine-tuning on the latent rollouts based on ground-truth trajectories, followed by (2) reinforcement learning with global and agent -specific rewards that encourage safety, progress, and interaction realism. We further propose diversity rewards that encourage the model to generate planning behaviors that may not be present in logged driving data. Notably, our closed-loop training framework is scalable and does not require external simulators, which can be computationally expensive to run and have limited visual fidelity to the real-world. MAPLE achieves state-of-the-art driving performance on Bench2Drive and demonstrates scalable, closed-loop multi-agent play for robust E2E autonomous driving systems.

**arXiv ID:** 2605.14201
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>Personality Engineering with AI Agents: A New Methodology for Negotiation Research</strong> - Michelle A. Vaccaro, Jared R. Curhan - [[pdf]](https://arxiv.org/pdf/2605.20554)</summary>

**Abstract:** According to canonical negotiation theory, people's success in a negotiation depends on how well they balance competing demands--empathizing and asserting, demonstrating concern for other and concern for self, being soft on the people and hard on the problem. Yet people struggle to manage these tensions, so researchers have lacked the ability to rigorously test the field's prescriptions under controlled conditions. AI agents do not face the same limitations, and their precision, repertoire, consistency, and scalability enable a new class of experiments to contribute to negotiation theory. In this article, we introduce personality engineering: a methodology that uses AI agents to precisely parameterize, manipulate, and evaluate negotiator personality. We propose using the interpersonal circumplex--and its two core dimensions of warmth and dominance--as a foundational coordinate system for the field. This approach offers both a rigorous methodology for testing classic negotiation theories and a practical guide for designing the personalities of AI negotiation agents.

**arXiv ID:** 2605.20554
</details>

<details>
<summary><strong>Governance by Design: Architecting Agentic AI for Organizational Learning and Scalable Autonomy</strong> - Nelly Dux, Cristina Alaimo, Philippe Roussiere, Abhishek Kumar Mishra - [[pdf]](https://arxiv.org/pdf/2605.20210)</summary>

**Abstract:** Agentic AI systems - systems that can pursue goals through multi-step planning and tool-mediated action with limited direct supervision - are moving from experimental prototypes to enterprise deployments. This transition introduces tensions in implementation, scaling, and governance: organizations seek scalable autonomy for knowledge and coordination work, yet must preserve accountability, safety, cost control, and responsibility as systems initiate actions, access enterprise data, and evolve through iterative updates. Building on an in-depth qualitative case of a large IT services company's 2025 development and staged rollout of an agentic system integrated with enterprise tools; we show that governance is implemented through concrete architectural and working arrangements that determine what the system is allowed to do, which tools and data it can use, how memory is handled, and how performance improvements are introduced over time. We then distill seven lessons that explain how to build effective governance into agentic AI during operationalization and scaling.

**arXiv ID:** 2605.20210
</details>

<details>
<summary><strong>Beyond Text-to-SQL: An Agentic LLM System for Governed Enterprise Analytics APIs</strong> - Gundeep Singh, Parsa Kavehzadeh, Jing Xia, Xue-Yong Fu, Julien Bouvier Tremblay, Md Tahmid Rahman Laskar, Vincent Lum, Shashi Bhushan TN - [[pdf]](https://arxiv.org/pdf/2605.21027)</summary>

**Abstract:** Enterprise analytics aims to make organizational data accessible for decision-making, yet non-technical users still face barriers when using traditional business intelligence tools or Text-to-SQL systems. While recent Text-to-SQL approaches based on Large Language Models (LLMs) promise natural language access to structured data, they fall short in enterprise settings where analytics pipelines rely on governed APIs rather than raw databases. In practice, these APIs encapsulate complex business logic to ensure consistency, auditability, and security. However, delegating mathematical or aggregation logic to an LLM introduces reliability and compliance risks. To this end, we present Analytic Agent, an LLM-based agentic system that translates natural language intents into secure interactions with enterprise analytics APIs. Evaluated on 90 real enterprise use cases constructed by domain experts, it reliably interprets user goals, validates permissions, executes governed queries, and generates compliant visualizations through multi-step reasoning and policy-aware orchestration.

**arXiv ID:** 2605.21027
</details>

<details>
<summary><strong>Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents</strong> - Chongrui Ye, Yuxiang Liu, Yu Wang, Haofei Yu, Yining Zhao, Ge Liu, Julian McAuley, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2605.20616)</summary>

**Abstract:** Language agents increasingly operate over streams of related tasks, yet existing memory systems struggle to convert accumulated experience into reusable knowledge. Retrieval-augmented and structured memory methods record per-session observations effectively, but often couple acquisition and consolidation into a single online process, leaving the agent without a global view across sessions to discover recurring patterns, abstract shared procedures, or prune redundant entries. Inspired by complementary learning systems theory, we propose Auto-Dreamer, a learned offline consolidator for language-agent memory. Auto-Dreamer decouples fast per-session memory acquisition from slow cross-session consolidation. Given a selected working region of a typed memory bank, the consolidator treats the region as read-only evidence, performs bounded tool-use to inspect entries and provenance-linked source trajectories, and synthesizes a fresh compact replacement set that abstracts across sessions and supersedes the original region. We train Auto-Dreamer via GRPO, using end-to-end agent performance as the reward signal to learn how to consolidate memories acquired through fast online experience. Trained on ScienceWorld trajectories alone, Auto-Dreamer outperforms fixed, RL-trained, and prompted memory baselines on ScienceWorld by 7 points while using an active memory bank 12$\times$ smaller than the strongest baseline, and continues to lead on held-out ALFWorld and WebArena without retraining -- using 6$\times$ less memory than the strongest baseline on ALFWorld.

**arXiv ID:** 2605.20616
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling</strong> - Caleb Winston, Ron Yifeng Wang, Azalia Mirhoseini, Christos Kozyrakis - [[pdf]](https://arxiv.org/pdf/2605.21470)</summary>

**Abstract:** Computer-use agents (CUA) automate tasks specified with natural language such as "order the cheapest item from Taco Bell" by generating sequences of calls to tools such as click, type, and scroll on a browser. Current implementations follow a sequential fetch-screenshot-execute loop where each iteration requires an LLM call, resulting in high latency and frequent errors from incorrect tool use. We present agent just-in-time (JIT) compilation, an alternative that compiles task descriptions directly into executable code that is free to include LLM calls, tool calls, and parallelization. Our approach comprises three components: (1) JIT-Planner, which generates multiple code plans, validates each against tool specifications, and selects the minimum-cost candidate; (2) JIT-Scheduler, which explores parallelization strategies via Monte Carlo cost estimation from learned latency distributions; and (3) an invariant-enforcing tool protocol specifying precondition and postcondition state requirements that reduce the rate of generating plans with incorrect tool use. Across 5 web applications, JIT-Planner achieves $10.4\times$ speedup and $+28\%$ accuracy over Browser-Use, while JIT-Scheduler achieves $2.4\times$ speedup and $+9\%$ accuracy over OpenAI CUA.

**arXiv ID:** 2605.21470
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (36 papers)</h2></summary>

<details>
<summary><strong>Tool-Augmented Agent for Closed-loop Optimization,Simulation,and Modeling Orchestration</strong> - Liyuan Deng, Shujian Deng, Yongkang Chen, Yongkang Dai, Zhihang Zhong, Linyang Li, Xiao Sun, Yilei Shi, Huaxi Huang - [[pdf]](https://arxiv.org/pdf/2605.20190)</summary>

**Abstract:** Iterative industrial design-simulation optimization is bottlenecked by the CAD-CAE semantic gap: translating simulation feedback into valid geometric edits under diverse, coupled constraints. To fill this gap, we propose COSMO-Agent (Closed-loop Optimization, Simulation, and Modeling Orchestration), a tool-augmented reinforcement learning (RL) framework that teaches LLMs to complete the closed-loop CAD-CAE process. Specifically, we cast CAD generation, CAE solving, result parsing, and geometry revision as an interactive RL environment, where an LLM learns to orchestrate external tools and revise parametric geometries until constraints are satisfied. To make this learning stable and industrially usable, we design a multi-constraint reward that jointly encourages feasibility, toolchain robustness, and structured output validity. In addition, we contribute an industry-aligned dataset that covers 25 component categories with executable CAD-CAE tasks to support realistic training and evaluation. Experiments show that COSMO-Agent training substantially improves small open-source LLMs for constraint-driven design, exceeding large open-source and strong closed-source models in feasibility, efficiency, and stability.

**arXiv ID:** 2605.20190
</details>

<details>
<summary><strong>Mahjax: A GPU-Accelerated Mahjong Simulator for Reinforcement Learning in JAX</strong> - Soichiro Nishimori, Shinri Okano, Keigo Habara, Sotetsu Koyamada, Eason Yu, Masashi Sugiyama - [[pdf]](https://arxiv.org/pdf/2605.20577)</summary>

**Abstract:** Riichi Mahjong is a multi-player, imperfect-information game characterized by stochasticity and high-dimensional state spaces. These attributes present a unique combination of challenges that mirror complex real-world decision-making problems in reinforcement learning. While prior research has heavily relied on supervised learning from human play logs to pre-train the policy, algorithms capable of learning \textit{tabula rasa} (from scratch) offer greater potential for general applicability, as evidenced by the AlphaZero lineage. To facilitate such research, we introduce \textbf{Mahjax}, a fully vectorized Riichi Mahjong environment implemented in JAX to enable large-scale rollout parallelization on Graphics Processing Units (GPUs). We also provide a high-quality visualization tool to streamline debugging and interaction with trained agents. Experimental results demonstrate that Mahjax achieves throughputs of up to \textbf{2 million} and \textbf{1 million steps per second} on eight NVIDIA A100 GPUs under the no-red and red rules, respectively. Furthermore, we validate the environment's utility for reinforcement learning by showing that agents can be trained effectively to improve their rank against baseline policies.

**arXiv ID:** 2605.20577
</details>

<details>
<summary><strong>VBFDD-Agent for Electric Vehicle Battery Fault Detection and Diagnosis: Descriptive Text Modeling of Battery Digital Signals</strong> - Joey Chan, Zhen Chen, Ershun Pan - [[pdf]](https://arxiv.org/pdf/2605.20742)</summary>

**Abstract:** With the rapid proliferation of electric vehicles, the safety and reliability of lithium-ion batteries have become critical concerns. Effective anomaly detection is essential for ensuring safe battery operation. However, as battery systems and operating scenarios become increasingly complex, battery fault diagnosis and maintenance require stronger cross-domain adaptability and human-AI collaboration. Traditional fault detection and diagnosis methods are usually designed for specific scenarios and predefined workflows, making them less effective in complex real-world applications.
To address the scarcity of open-source battery fault report corpora and the lack of unified maintenance knowledge representation, this study proposes a descriptive text modeling approach for battery signal reports. Monitoring signals, statistical features, anomaly records, and state assessment results are transformed into structured and readable natural language descriptions, forming a language corpus for battery health diagnosis and maintenance.
Based on this corpus, we propose VBFDD-Agent, a vehicle battery fault detection and diagnosis agent for automotive-grade battery systems. VBFDD-Agent integrates descriptive battery-state texts, historical case retrieval, local maintenance manuals, and large language model reasoning to generate structured diagnostic results and maintenance recommendations. Experiments show that the proposed framework can accurately perform anomaly monitoring based on descriptive textual representations and provide flexible, efficient, and actionable maintenance suggestions. Expert evaluation further confirms the practical value of the generated recommendations. Overall, VBFDD-Agent extends traditional battery diagnosis from label prediction to interpretable and maintenance-oriented decision support.

**arXiv ID:** 2605.20742
</details>

<details>
<summary><strong>ScenePilot: Controllable Boundary-Driven Critical Scenario Generation for Autonomous Driving</strong> - Qiyu Ruan, Yuxuan Wang, He Li, Zhenning Li, Cheng-zhong Xu - [[pdf]](https://arxiv.org/pdf/2605.21168)</summary>

**Abstract:** Safety-critical scenarios are central to evaluating autonomous driving systems, yet their rarity in naturalistic logs makes simulation-based stress testing indispensable. Most scenario generation methods treat surrounding agents as adversaries, but they either (i) induce failures without explicitly modeling vehicle-road physical limits, yielding visually extreme yet physically unsolvable crashes, or (ii) enforce physical feasibility or policy feasibility in isolation, which can over-focus on aggressive maneuvers or remain tied to a controller-dependent capability boundary. We propose ScenePilot, a feasibility-guided, boundary-driven framework that targets the boundary band: scenarios that are physically solvable in principle yet still cause the deployed autonomy stack to fail. We formulate generation as constrained multi-objective reinforcement learning, combining an RSS-derived physical-feasibility score $\sigma$ with an online-learned AV-risk predictor $\Phi$, and introduce step-level feasibility-aware shielding to keep exploration near the feasibility boundary while avoiding infeasible artifacts. Experiments on SafeBench with multiple planners show that ScenePilot yields substantially higher collision rates (+6.2 percentage points) while preserving physical validity, and that adversarial fine-tuning on these boundary-band scenarios consistently reduces downstream crash rates. The code is available at this https URL.

**arXiv ID:** 2605.21168
</details>

<details>
<summary><strong>Lean Refactor: Multi-Objective Controllable Proof Optimization via Agentic Strategy Search</strong> - Jialin Lu, Soonho Kong, Rodrigo Stehling, Kaiyu Yang, Zhangyang Wang, Weiran Sun, Wuyang Chen - [[pdf]](https://arxiv.org/pdf/2605.20244)</summary>

**Abstract:** We present Lean Refactor, a plug-and-play retrieval-augmented agentic framework for multi-objective, controllable, and version-robust refactoring of Lean proofs. LLM-generated proofs are notoriously correct-but-verbose and brittle across library versions, yet existing refactoring works overlook three practical challenges: 1) Lean refactoring is natively multi-objective (proof length, compilation cost, and version compatibility are often in tension); 2) Lean repositories have fragile compatibility, whereas LLM releases are unaware of Lean/Mathlib versions; 3) Training-based pipelines require repeated fine-tuning with each new LLM release, scaling neither with model churn nor with Lean's release cycle. Lean Refactor steers a frozen agentic LLM with retrievals from a curated database of multi-objective refactoring strategies, each densely annotated with metadata such as supported Lean/Mathlib versions and expected compilation-cost reduction. Experiments show over $70\%$ token-level compression on competition benchmarks, over $20\%$ on research repositories, and up to $60\%$ compilation-time reduction, outperforming prior work and Claude Code. Version-filtered retrieval further improves compression on the target Lean version, and refactored miniF2F proofs exhibit stronger zero-shot version transfer to future Lean releases than their unrefactored counterparts.

**arXiv ID:** 2605.20244
</details>

<details>
<summary><strong>GROW: Aligning GRPO with State-Action Modeling for Open-World VLM Agents</strong> - Xiongbin Wu, Zhihao Luo, Shanzhe Lei, Lechao Zhang, Xuhong Wang, Jie Yang, Zhonglong Zheng, Yuanjie Zheng, Xin Tan, Wei Liu - [[pdf]](https://arxiv.org/pdf/2605.20246)</summary>

**Abstract:** Recently, vision-language model (VLM) agents have shown promising progress in open-world tasks, where successful task completion often requires multiple turns of visual perception and action execution. However, existing methods still rely primarily on Supervised Fine-Tuning (SFT) with expert demonstrations, while the advanced reinforcement learning (RL) algorithm, specifically Group Relative Policy Optimization (GRPO), has not been effectively employed for multi-turn RL in these tasks because standard GRPO requires full trajectories as training samples which leads to excessively long context and noise. To address this issue, we propose GROW, a RL framework for open-world VLM agents that decomposes collected trajectories into state-action samples, and computes advantages between these samples rather than treating a full trajectory as a single entity. We further provide a surrogate analysis indicating that, even though the grouped samples are conditioned on different local states rather than an identical prompt context, the objective can preserve the core relative policy optimization signal of GRPO under simplifying assumptions. Experiments on more than 800 Minecraft tasks show that our method achieves state-of-the-art (SOTA) performance, demonstrating the effectiveness of our proposed RL framework for open-world VLM agents.

**arXiv ID:** 2605.20246
</details>

<details>
<summary><strong>FBOS-RL: Feedback-Driven Bi-Objective Synergistic Reinforcement Learning</strong> - Xikai Zhang, Yongzhi Li, Likang Xiao, Yingze Zhang, Yanhua Cheng, Quan Chen, Peng Jiang, Wenjun Wu, Liu Liu - [[pdf]](https://arxiv.org/pdf/2605.20256)</summary>

**Abstract:** Reinforcement learning has become a cornerstone for aligning and unlocking the reasoning capabilities of large-scale models. At its core, the training loop of GRPO and its variants alternates between rollout sampling and policy update. Unlike supervised learning, where each gradient step is anchored to an explicit ground-truth target, the optimal gradient direction for updating model parameters in this setting is not known a priori; the high-quality rollouts drawn during the sampling stage therefore act as the implicit "teacher" that guides every parameter update. However, GRPO adopt a simple sampling scheme that conditions all rollouts on the same original prompt. When a task lies beyond the policy model's current capability, this sampling scheme rarely yields a high-quality rollout, leaving the policy model without a meaningful gradient direction when updating its parameters, which causes training to stall. To address this issue, we propose FBOS-RL, a Feedback-Driven Bi-Objective Synergistic reinforcement learning framework. Specifically, we let the model perform Feedback-Guided Exploration Enhancement based on the feedback provided by the environment, and on top of this we design two mutually reinforcing training objectives: Exploitation-oriented Policy Alignment(EPA) and Exploration-oriented Capability Cultivation(ECC). Extensive experiments demonstrate that EPA and ECC can mutually reinforce each other, forming a positive flywheel effect that significantly improves both the training efficiency and the final performance ceiling of reinforcement learning. Specifically, under an identical number of rollouts, FBOS-RL learns substantially faster than GRPO and feedback-based baselines and ultimately attains a higher performance ceiling, while exhibiting higher policy entropy and lower gradient norms throughout training.

**arXiv ID:** 2605.20256
</details>

<details>
<summary><strong>Smaller Abstract State Spaces Enable Cross-Scale Generalization in Reinforcement Learning</strong> - Nasehatul Mustakim, Lucas Lehnert - [[pdf]](https://arxiv.org/pdf/2605.20272)</summary>

**Abstract:** While humans readily generalize abstract concepts to more complex or larger tasks, building Reinforcement Learning (RL) systems with this ability remains elusive. Here, we present the first theoretical model of how such Out-of-Distribution (OOD) generalization can be achieved in RL agents. Our approach considers Partially Observable Markov Decision Processes (POMDPs) and assumes that an intelligent agent uses an abstraction function to determine which experiences can be treated as equivalent and which must be distinguished. First, we extend the existing state abstraction framework and proof techniques to POMDPs. Then, we define a successor-weighted model reduction, a model reduction variant that enables compression into smaller abstract spaces than prior definitions allow. We derive a bound on the agent's OOD test performance, thereby defining the conditions under which OOD generalization is achievable. This bound decomposes an agent's performance loss into approximation and estimation errors, revealing how reducing an agent's abstract state space size improves test performance and OOD generalization. Our analysis suggests that constraining an agent to operate over a small, finite set of abstract states is necessary for achieving generalization to more complex tasks. Our results motivate further research into learning RL architectures that scale across tasks of varying complexity levels.

**arXiv ID:** 2605.20272
</details>

<details>
<summary><strong>ClaimDiff-RL: Fine-Grained Caption Reinforcement Learning through Visual Claim Comparison</strong> - Tianle Li, Xuyang Shen, Yan Ma, Rongxin Guo, Shaoxiang Chen, Jiacheng Chen, Haochen Wang, Hongyang Tang, Yucong Zhou, Yu Cheng - [[pdf]](https://arxiv.org/pdf/2605.20278)</summary>

**Abstract:** Long-form image captioning exposes a reward granularity problem in RL: captions are judged as whole sequences, while the important errors occur at the level of individual visual claims. A good dense caption should be both faithful and informative, avoiding hallucination without omitting salient details. Yet pairwise preferences, reference-based metrics, and holistic scalar rewards compress these local errors into a single sequence-level signal, obscuring the tradeoff between factuality and coverage. We introduce ClaimDiff-RL, a framework that uses reference-conditioned atomic claim differences as the reward unit for caption RL. Given an image, an actor caption, and a reference caption, a multimodal judge enumerates visually grounded differences, verifies each difference against the image, assigns open-vocabulary error types and severity levels, and produces per-difference statistics for reward composition. This makes hallucinated claims and omitted salient facts separately measurable and tunable. Experiments show that holistic scalar rewards can reduce hallucination by increasing missing facts, while ClaimDiff-RL exposes this faithfulness and coverage tradeoff and enables more balanced operating points. On a 160-image human-labeled diagnostic benchmark, public captioning benchmarks, and VQA benchmarks, ClaimDiff-RL improves the hallucination--missing-fact balance, preserves general capability, and even surpasses Gemini-3-Pro-Preview on several fine-grained Capability dimensions such as object counting, spatial relations, and scene recognition. These results suggest that typed, verifiable claim differences are an effective reward unit for fine-grained and diagnosable caption RL.

**arXiv ID:** 2605.20278
</details>

<details>
<summary><strong>ConceptSeg-R1: Segment Any Concept via Meta-Reinforcement Learning</strong> - Yuan Zhao, Youwei Pang, Jiaming Zuo, Wei Ji, Kailai Zhou, Bin Fan, Yunkang Cao, Lihe Zhang, Xiaofeng Liu, Huchuan Lu, Weisi Lin, Dacheng Tao, Xiaoqi Zhao - [[pdf]](https://arxiv.org/pdf/2605.20385)</summary>

**Abstract:** Recent progress in promptable segmentation has shifted visual perception from object-level localization toward concept-level understanding. However, the notion of a concept remains under-specified, making it unclear whether current methods truly generalize beyond category recognition. In this work, we formalize generalized concept segmentation through a three-level taxonomy consisting of context-independent (CI), context-dependent (CD), and context-reasoning (CR) concepts, which reveals a clear capability gap across increasing levels of cognitive complexity. To address this challenge, we propose ConceptSeg-R1, a unified framework that reformulates concept segmentation as rule-induced concept grounding. At the core of our method is Meta-GRPO, a meta-reinforcement learning mechanism that learns transferable task rules from visual demonstrations and verifies them through proxy reasoning. The inferred reasoning states are then translated into segmentation-ready concept prompts via a lightweight concept translation module, enabling deductive application to target images. A shortcut routing strategy further preserves the native efficiency of segmentation models on simple cases. To systematically evaluate generalized concept segmentation, we conduct extensive experiments across diverse CI, CD, and CR concept segmentation benchmarks spanning natural, industrial, medical and reasoning-intensive domains. Without bells and whistles, ConceptSeg-R1 achieves strong performance across the full concept hierarchy while maintaining the native capability of promptable segmentation backbones. As an initial step toward segmenting any concept, we hope ConceptSeg-R1 can serve as a practical baseline for advancing segmentation from object-level prediction toward concept-level understanding.

**arXiv ID:** 2605.20385
</details>

<details>
<summary><strong>Decomposing MXFP4 quantization error for LLM reinforcement learning: reducible bias, recoverable deadzone, and an irreducible floor</strong> - Xiaocan Li, Shiliang Wu, Zheng Shen - [[pdf]](https://arxiv.org/pdf/2605.20402)</summary>

**Abstract:** MXFP4 arithmetic can dramatically accelerate reinforcement learning (RL) post-training of large language models (LLMs), yet the quantization error introduces severe accuracy degradation. Existing work treats the quantization error as a monolithic noise term, missing the distinct mechanisms upon interpreting how quantization error damages training. We prove an exact three-way decomposition of quantization error and show how each component dominates a distinct RL training pathway. Our theoretical and empirical analysis decomposes the MXFP4 quantization error into three additive components: "scale bias" from power-of-two rounding, "deadzone truncation" from zeroing small values, and "grid noise" from rounding to the nearest 4-bit grid. Each component dominates a distinct RL failure mode: scale bias accumulates multiplicatively through the backward pass, affecting gradient accuracy; deadzone truncation degrades rollout quality; and grid noise raises the policy's entropy. We combine corrections that are RL failure mode-targeted but not component-exclusive: Macro-block scaling to reduce scale bias, Outlier Fallback recovers deadzone entries, but also partially reduces scale bias induced error, and Adaptive Quantization Noise (AQN) for controlling the policy entropy. On Qwen2.5-3B dense and Qwen3-30B-A3B-Base mixture-of-experts model, the targeted corrections recover BF16 accuracy to within 0.7% and 3.0% respectively.

**arXiv ID:** 2605.20402
</details>

<details>
<summary><strong>Agentic Agile-V: From Vibe Coding to Verified Engineering in Software and Hardware Development</strong> - Christopher Koch - [[pdf]](https://arxiv.org/pdf/2605.20456)</summary>

**Abstract:** Agentic AI coding systems can inspect repositories, plan implementation steps, edit files, call tools, run tests, and submit pull requests. These capabilities make software and hardware development faster in some settings, but current evidence does not support the simple claim that autonomous code generation automatically improves engineering outcomes. Controlled studies report productivity gains in some enterprise tasks, slowdowns in mature open-source work, moderate but heterogeneous meta-analytic effects, and persistent failures in repository setup, dependency handling, permission gating, and hardware verification. This paper argues that the central problem is no longer prompt engineering; it is engineering process control. It synthesizes evidence from agentic software engineering, GitHub-scale adoption studies, repository-level agent configuration, productivity trials, issue-resolution benchmarks, and hardware/RTL verification research. It proposes Agentic Agile-V, a process framework that uses Agile-V as the lifecycle backbone and a task-level SCOPE-V loop - Specify, Constrain, Orchestrate, Prove, Evolve, and Verify - to convert conversational intent into structured engineering artifacts and acceptance evidence. The paper contributes: (i) a taxonomy of minimum input artifacts for agentic software, firmware, and hardware work; (ii) a conversation-to-contract gate that separates exploratory dialogue from implementation; (iii) risk-adaptive feature, bug-fix, testing, and hardware workflows; and (iv) an evidence-bundle acceptance model for agent-generated artifacts. The paper concludes that agentic AI does not eliminate engineering discipline; it increases the value of requirements, constraints, traceability, independent verification, and human approval.

**arXiv ID:** 2605.20456
</details>

<details>
<summary><strong>Complementing reinforcement learning with SFT through logit averaging in the post training of LLMs</strong> - Xingwei Gan, Ying Zhu - [[pdf]](https://arxiv.org/pdf/2605.20555)</summary>

**Abstract:** We introduce a novel method that averages the logits of a frozen reference policy (e.g., SFT) and a trainable policy, and incorporate the method into Group Relative Policy Optimization (GRPO). In contrast to Reinforcement Learning with Verifiable Rewards (RLVR) methods, our proposal does not involve a Kullback Leibler (KL) regularization or critic; the trainable policy and the reference anchor are coupled through the logit averaging structure to leverage the reasoning expertise of the trainable policy while maintaining the formatting advantage of SFT. Our method is evaluated on MATH, cn-k12, and MMLU, and the results show a higher accuracy or at least comparable accuracy relative to the canonical KL-regularized GRPO.

**arXiv ID:** 2605.20555
</details>

<details>
<summary><strong>Design for Manufacturing: A Manufacturability Knowledge-Integrated Reinforcement Learning Framework for Free-Form Pipe Routing in Aeroengines</strong> - Caicheng Wang, Zili Wang, Shuyou Zhang, Yongzhe Xiang, Zheyi Li, Liangyou Li, Jianrong Tan - [[pdf]](https://arxiv.org/pdf/2605.20644)</summary>

**Abstract:** Design for manufacturing plays a critical role in advanced aeroengine development, where complex components necessitate careful consideration of manufacturability. However, current practices in pipe routing remain largely decoupled from down-stream manufacturing, leading to labor-intensive, trial-and-error iterations to achieve manufacturable designs. To address this problem, this study proposes the Frenet-based pipe routing optimization (FPRO) framework, a manufacturability knowledge-integrated reinforcement learning approach for free-form pipe design in aeroengines. FPRO formulates the routing problem as a boundary value problem in the Frenet frame. In this framework, the pipe path is represented by curvature and torsion profiles, which are generated using cubic Hermite interpolation. To integrate design and manufacturing, domain-specific manufacturing knowledge is embedded as constraints on the permissible ranges of curvature and torsion. The path optimization is performed using the proximal policy optimization algorithm with stochastic exploration and a stage-guided reward mechanism. A unified mapping formulation then translates the optimized path into motion trajectories for the bending die, enabling direct fabrication on a six-axis free-bending machine. Experimental results demonstrate that FPRO consistently generates collision-free, manufacturable paths with smoother geometric profiles compared to Cartesian-based methods. It also achieves faster convergence and superior performance in terminal alignment, path length, obstacle avoidance, and manufacturability compared to state-of-the-art reinforcement learning baselines. Real-world validation confirms the close geometric correspondence between the manufactured pipe and its digital design, validating the practical feasibility of FPRO.

**arXiv ID:** 2605.20644
</details>

<details>
<summary><strong>Distribution-Aware Reward: Reinforcement Learning over Predictive Distributions for LLM Regression</strong> - Jungsoo Park, Hyungjoo Chae, Ethan Mendes, Jay DeYoung, Varsha Kishore, Wei Xu, Alan Ritter - [[pdf]](https://arxiv.org/pdf/2605.20740)</summary>

**Abstract:** Large language models can predict real-valued quantities from heterogeneous inputs such as text, code, and molecular strings, but most training objectives score each decoded floating-point number independently, improving point estimates without ensuring calibrated predictive distributions. This limits applications requiring candidate ranking or uncertainty estimation. We introduce Distribution-Aware Reward, an on-policy reinforcement learning objective whose main contribution is to train language models to produce better predictive distributions for regression tasks, rather than only optimizing individual decoded outputs against scalar targets. Our method treats multiple decoded samples as an empirical predictive distribution, evaluates it with the Continuous Ranked Probability Score, and assigns leave-one-out credit based on each rollout's marginal contribution to distribution quality, rewarding predictions that are both accurate and appropriately dispersed. We evaluate our method on a controlled Gaussian-mixture task, code performance prediction, and molecular property prediction from SMILES strings. Across tasks, our method improves over supervised fine-tuning and pointwise reinforcement learning baselines, with strong rank-correlation gains, including a 6-point Spearman improvement on KBSS. On MoleculeNet, it uses only SMILES strings yet remains competitive with strong graph-based and 3D molecular models. Further analyses show that our method mitigates rollout diversity collapse and improves uncertainty diagnostics, suggesting that directly optimizing predictive distributions makes language model regression more robust and better calibrated.

**arXiv ID:** 2605.20740
</details>

<details>
<summary><strong>Multi-Step Likelihood-Ratio Correction for Reinforcement Learning with Verifiable Rewards</strong> - Deokgyu Yoon, Hyungkyu Kang, Joongkyu Lee, Byeongchan Kim, Gyungin Shin, Sungrae Park, Min-hwan Oh - [[pdf]](https://arxiv.org/pdf/2605.20865)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) plays a pivotal role in improving the reasoning ability of large language models. However, widely used PPO surrogate objectives are fundamentally local, as they rely on a local approximation of the exact policy gradient objective. While this approximation improves stability by reducing the variance induced by importance sampling, it also introduces structural bias into the surrogate objective, which must be controlled through trust region mechanisms. In this work, we introduce the $N$-step forward trace, which augments the PPO surrogate objective using the cumulative likelihood ratio of the next $N-1$ tokens. Building on this idea, we propose $N$-Step Forward-Trace Policy Optimization (NFPO), a practical RLVR algorithm that integrates the $N$-step forward trace into the masked policy gradient framework. NFPO provides a continuous bridge between the PPO surrogate objective and the exact policy gradient objective, offering a principled mechanism for controlling the bias-variance trade-off. Our theoretical analysis shows that, with an appropriate choice of $N$, the proposed objective yields a tighter policy-improvement bound than the standard PPO surrogate. Experiments on comprehensive reasoning benchmarks demonstrate that NFPO consistently improves performance, supporting our theoretical findings.

**arXiv ID:** 2605.20865
</details>

<details>
<summary><strong>Enhanced Reinforcement Learning-based Process Synthesis via Quantum Computing</strong> - Austin Braniff, Fengqi You, Yuhe Tian - [[pdf]](https://arxiv.org/pdf/2605.21213)</summary>

**Abstract:** In this work, we present quantum reinforcement learning (RL) as a solution strategy for process synthesis problems. Building on our prior work, we develop a generalized framework that formally poses process synthesis as a Markov decision process and introduces quantum-enhanced RL algorithms to solve it with improved scalability. Earlier implementations of quantum-based RL for process synthesis were limited by qubit requirements, which scaled poorly with problem complexity. This work overcomes this challenge by introducing state encoding algorithms to decouple qubit requirements from problem size. A classical RL-based solution strategy is used as a baseline to benchmark the quantum algorithms under identical training conditions. All algorithms are evaluated across a flowsheet synthesis problem of increasing unit counts to analyze their performance and scalability. Results show that all approaches are capable of identifying the optimal flowsheet designs in small design spaces. For moderate-scale unit counts, quantum approaches demonstrate competitive performance on a per-episode basis and improved efficiency on a per-parameter basis versus the classical RL benchmark. This work provides a foundation for future quantum computing applications within process systems engineering, establishes a controlled benchmark for comparing classical and quantum algorithms, and shows that the proposed quantum variants remain competitive for the process synthesis problem examined in this work.

**arXiv ID:** 2605.21213
</details>

<details>
<summary><strong>Behavior-Consistent Deep Reinforcement Learning</strong> - Marcel Hussing, Liv G. d'Aliberti, Claas Voelcker, Benjamin Eysenbach, Eric Eaton - [[pdf]](https://arxiv.org/pdf/2605.21214)</summary>

**Abstract:** Reinforcement learning (RL) often exhibits high variance across training runs, leading to unreliable performance and posing a major challenge to deployment in real-world domains. In this work, we address the challenge of cross-run policy divergence by formalizing the problem of behavior-consistent RL, where the objective is to obtain policies that are both high-performing and distributionally similar across training runs. Our key observation is that maximum-entropy RL provides a direct mechanism for controlling behavioral divergence by anchoring runs to a common (uniform) prior. We prove that, for Boltzmann policies, choosing the temperature proportional to $Q$-function disagreement bounds the pairwise KL divergence between the induced policies. However, we also show that naïvely increasing entropy might impair policy optimization while amplifying off-policy error. Building upon these observations, we propose $Q$-value Expectile Disagreement (QED), a state-dependent temperature schedule that uses double-critic disagreement as a single-run proxy for cross-run disagreement. Empirically, we demonstrate that across 18 continuous-control tasks, QED reduces across-run divergence by two orders of magnitude without sacrificing performance, resulting in a considerable reduction in return variance at modest sample-efficiency costs.

**arXiv ID:** 2605.21214
</details>

<details>
<summary><strong>DeCoR: Design and Control Co-Optimization for Urban Streets Using Reinforcement Learning</strong> - Bibek Poudel, Lei Zhu, Kevin Heaslip, Sai Swaminathan, Weizi Li - [[pdf]](https://arxiv.org/pdf/2605.21311)</summary>

**Abstract:** Modern vision systems can detect, track, and forecast urban actors at scale, yet translating perception outputs to urban design remains limited. We introduce DeCoR, a two-stage reinforcement learning framework that leverages flow observations to co-optimize crosswalk layout and network-level signal control. The design stage encodes the pedestrian network as a graph and learns a generative policy that parameterizes a Gaussian mixture model over crosswalk location and width, from which new crosswalks are sampled. For each layout, a shared control policy learns adaptive signal timings to minimize joint pedestrian and vehicle delay. On a 750 m real-world urban corridor with demand sensed from video and Wi-Fi logs, DeCoR learns a layout that reduces pedestrian arrival time to their nearest crosswalk by 23% while using fewer crosswalks than existing configurations. On the control side, DeCoR reduces pedestrian and vehicle wait time by 79% and 65%, respectively, relative to fixed-time signalization. Further, the control policy generalizes to demands outside of training and is robust to layout changes without retraining.

**arXiv ID:** 2605.21311
</details>

<details>
<summary><strong>Agentic Physical AI toward a Domain-Specific Foundation Model for Nuclear Reactor Control</strong> - Yoon Pyo Lee, Samrendra Roy, Jay Yoo, Kazuma Kobayashi, Sajedul Talukder, Seid Koric, Souvik Chakraborty, Syed Bahauddin Alam - [[pdf]](https://arxiv.org/pdf/2512.23292)</summary>

**Abstract:** The prevailing paradigm in AI for physical systems (scaling general-purpose foundation models toward universal multimodal reasoning) confronts a fundamental barrier at the control interface. Recent benchmarks show that even frontier vision--language models achieve only 50--53% accuracy on basic quantitative physics tasks, behaving as approximate guessers that preserve semantic plausibility by violating physical constraints. This input unfaithfulness is not a scaling deficiency but a structural limitation: perception-centric architectures optimize parameter-space imitation, whereas safety-critical control demands outcome-space guarantees over executed actions. Here, we present a fundamentally different pathway "toward" domain-specific foundation models by introducing compact language models operating as Agentic Physical AI, in which policy optimization is driven by physics-based validation rather than perceptual inference. We train a 360-million-parameter model on synthetic nuclear reactor control scenarios, scaling the dataset from 10^3 to 10^5 examples. Scaling induces strong improvements in closed-loop reliability under nominal simulated conditions, with a steep but smooth gain at strict tolerances: small-scale systems exhibit high-variance imitation with severe tail excursions, while large-scale models undergo variance collapse (approximately 500times reduction), stabilizing execution-level behavior within the sampled distribution. Despite balanced exposure to four actuation families, the model autonomously rejects approximately 70\% of the training distribution, concentrating 95% of runtime execution on a single-bank strategy. This emergent policy distillation arises without reinforcement learning or reward engineering, driven solely by outcome-level success under physical execution.

**arXiv ID:** 2512.23292
</details>

<details>
<summary><strong>DelTA: Discriminative Token Credit Assignment for Reinforcement Learning from Verifiable Rewards</strong> - Kaiyi Zhang, Wei Wu, Yankai Lin - [[pdf]](https://arxiv.org/pdf/2605.21467)</summary>

**Abstract:** Reinforcement learning from verifiable rewards (RLVR) has emerged as a central technique for improving the reasoning capabilities of large language models. Despite its effectiveness, how response-level rewards translate into token-level probability changes remains poorly understood. We introduce a discriminator view of RLVR updates, showing that the policy-gradient update direction implicitly acts as a linear discriminator over token-gradient vectors and thereby determines which token probabilities are increased or decreased during learning. Under standard sequence-level RLVR, this discriminator is constructed from positive- and negative-side centroids formed by advantage-weighted averaging of token-gradient vectors. However, such centroid construction can be dominated by shared high-frequency patterns, such as formatting tokens, diluting sparse yet discriminative directions that better distinguish high-reward responses from low-reward ones. To address this limitation, we propose $\textbf{DelTA}$, a discriminative token credit assignment method that estimates token coefficients to amplify side-specific token-gradient directions and downweight shared or weakly discriminative ones. These coefficients reweight a self-normalized RLVR surrogate, making the effective side-wise centroids more contrastive and thereby reshaping the RLVR update direction. On seven mathematical benchmarks, DelTA outperforms the strongest same-scale baselines by 3.26 and 2.62 average points on Qwen3-8B-Base and Qwen3-14B-Base, respectively. Additional results on code generation, a different backbone, and out-of-domain evaluations further demonstrate the generalization ability of DelTA.

**arXiv ID:** 2605.21467
</details>

<details>
<summary><strong>Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory</strong> - Haozhen Zhang, Haodong Yue, Tao Feng, Quanyu Long, Jianzhu Bao, Bowen Jin, Weizhi Zhang, Xiao Li, Jiaxuan You, Chengwei Qin, Wenya Wang - [[pdf]](https://arxiv.org/pdf/2602.06025)</summary>

**Abstract:** Memory is increasingly central to Large Language Model (LLM) agents operating beyond a single context window, yet most existing systems rely on offline, query-agnostic memory construction that can be inefficient and may discard query-critical information. Although runtime memory utilization is a natural alternative, prior work often incurs substantial overhead and offers limited explicit control over the performance-cost trade-off. In this work, we present \textbf{BudgetMem}, a runtime agent memory framework for explicit, query-aware performance-cost control. BudgetMem structures memory processing as a set of memory modules, each offered in three budget tiers (i.e., \textsc{Low}/\textsc{Mid}/\textsc{High}). A lightweight router performs budget-tier routing across modules to balance task performance and memory construction cost, which is implemented as a compact neural policy trained with reinforcement learning. Using BudgetMem as a unified testbed, we study three complementary strategies for realizing budget tiers: implementation (method complexity), reasoning (inference behavior), and capacity (module model size). Across LoCoMo, LongMemEval, and HotpotQA, BudgetMem surpasses strong baselines when performance is prioritized (i.e., high-budget setting), and delivers better accuracy-cost frontiers under tighter budgets. Moreover, our analysis disentangles the strengths and weaknesses of different tiering strategies, clarifying when each axis delivers the most favorable trade-offs under varying budget regimes.

**arXiv ID:** 2602.06025
</details>

<details>
<summary><strong>Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations</strong> - Dongming Jiang, Yi Li, Songtao Wei, Jinxin Yang, Ayushi Kishore, Alysa Zhao, Dingyi Kang, Xu Hu, Feng Chen, Qiannan Li, Bingzhe Li - [[pdf]](https://arxiv.org/pdf/2602.19320)</summary>

**Abstract:** Agentic memory systems enable large language model (LLM) agents to maintain state across long interactions, supporting long-horizon reasoning and personalization beyond fixed context windows. Despite rapid architectural development, the empirical foundations of these systems remain fragile: existing benchmarks are often underscaled, evaluation metrics are misaligned with semantic utility, performance varies significantly across backbone models, and system-level costs are frequently overlooked. This survey presents a structured analysis of agentic memory from both architectural and system perspectives. We first introduce a concise taxonomy of MAG systems based on four memory structures. Then, we analyze key pain points limiting current systems, including benchmark saturation effects, metric validity and judge sensitivity, backbone-dependent accuracy, and the latency and throughput overhead introduced by memory maintenance. By connecting the memory structure to empirical limitations, this survey clarifies why current agentic memory systems often underperform their theoretical promise and outlines directions for more reliable evaluation and scalable system design.

**arXiv ID:** 2602.19320
</details>

<details>
<summary><strong>Argus: Evidence Assembly for Scalable Deep Research Agents</strong> - Zhen Zhang, Liangcai Su, Zhuo Chen, Xiang Lin, Haotian Xu, Simon Shaolei Du, Kaiyu Yang, Bo An, Lidong Bing, Xinyu Wang - [[pdf]](https://arxiv.org/pdf/2605.16217)</summary>

**Abstract:** Deep research agents have achieved remarkable progress on complex information seeking tasks. Even long ReAct style rollouts explore only a single trajectory, while recent state of the art systems scale inference time compute via parallel search and aggregation. Yet deep research answers are composed of complementary pieces of evidence, which parallel rollouts often duplicate rather than complete, yielding diminishing returns while pushing the aggregation context toward the model's limit. We propose Argus, an agentic system in which a Searcher and a Navigator cooperate to treat deep research as assembling a jigsaw from complementary evidence pieces, rather than brute forcing the whole answer in parallel. The Searcher collects evidence traces for a given sub-query through ReAct-style interaction. The Navigator maintains a shared evidence graph, verifying which pieces are still missing, dispatching Searchers to gather them, and reasoning over the completed graph to produce a source-traced final answer. We train the Navigator with reinforcement learning to verify, dispatch, and synthesize, while independently training the Searcher to remain a standard ReAct agent. The resulting Navigator supports rollouts with a single Searcher or many in parallel without retraining. With both Searcher and Navigator built on a 35B-A3B MoE backbone, Argus gains 5.5 points with a single Searcher and 12.7 points with 8 parallel Searchers, averaged over eight benchmarks. With 64 Searchers it reaches 86.2 on BrowseComp, surpassing every proprietary agent we benchmark, while the Navigator's reasoning context stays under 21.5K tokens.

**arXiv ID:** 2605.16217
</details>

<details>
<summary><strong>General Preference Reinforcement Learning</strong> - Muhammad Umer, Muhammad Ahmed Mohsin, Ahsan Bilal, Arslan Chaudhry, Andreas Haupt, Sanmi Koyejo, Emily Fox, John M. Cioffi - [[pdf]](https://arxiv.org/pdf/2605.18721)</summary>

**Abstract:** Post-training has split large language model (LLM) alignment into two largely disconnected tracks. Online reinforcement learning (RL) with verifiable rewards drives emergent reasoning on math and code but depends on a programmatic verifier that cannot reach open-ended tasks, while preference optimization handles open-ended generation yet forgoes the continuous exploration that powers online RL. Closing this gap requires a verifier for open-ended quality, but a scalar reward model is the wrong shape for the job. Quality is multi-dimensional, and any scalar score is an incomplete proxy that lets online RL collapse onto whichever axis the score is most sensitive to. We turn instead to the General Preference Model (GPM), which embeds responses into $k$ skew-symmetric subspaces and represents preference as a structured, intransitivity-aware comparison. Building on this, we propose General Preference Reinforcement Learning (GPRL), which carries the $k$-way structure through to the policy update. GPRL computes per-dimension group-relative advantages, normalizes each on its own scale so no axis can dominate, and aggregates them with context-dependent eigenvalues. The same structure powers a closed-loop drift monitor that detects single-axis exploitation and corrects it on the fly by reweighting dimensions and tightening the trust region. Starting from $\texttt{Llama-3-8B-Instruct}$, GPRL reaches a length-controlled win rate of $56.51\%$ on AlpacaEval~2.0 while also outperforming SimPO and SPPO on Arena-Hard, MT-Bench, and WildBench by resisting reward hacking across extended training runs.

**arXiv ID:** 2605.18721
</details>

<details>
<summary><strong>ReversedQ: Opportunities for Faster Q-Learning in Episodic Online Reinforcement Learning</strong> - Sofia R. Miskala-Dinc, Aviva Prins - [[pdf]](https://arxiv.org/pdf/2605.20592)</summary>

**Abstract:** We study model-free Q-learning in finite-horizon episodic Markov Decision Processes (MDPs) with stationary dynamics across episodes. We identify a central issue in nascent model-free posterior-sampling works: the reliance on delayed learning in order to prove theoretical guarantees. In particular, we identify three opportunities for faster learning - (i) value-function update order, (ii) update frequencies, and (iii) value-function initialization. Using Wang et al.'s RandomizedQ as a basis, we illustrate these changes and their individual (as well as cumulative) impact in multiple empirical studies. We find that our combined modifications, termed ReversedQ, improve scaled mean cumulative reward compared to RandomizedQ, from 9.53% to 78.78% in the Bidirectional Diabolical Combination Lock (BDCL), and from 21.76% to 61.81% in a chain MDP.

**arXiv ID:** 2605.20592
</details>

<details>
<summary><strong>Compositional Transduction with Latent Analogies for Offline Goal-Conditioned Reinforcement Learning</strong> - Junseok Kim, Dohyeong Kim, Mineui Hong, Songhwai Oh - [[pdf]](https://arxiv.org/pdf/2605.20609)</summary>

**Abstract:** Compositional generalization is essential for reaching unseen goals under novel contextual variations in offline goal-conditioned reinforcement learning (GCRL), where a generalist goal-reaching agent must be learned from limited data. Most prior approaches pursue this via trajectory stitching over temporally contiguous segments, which limits composing behaviors across varying contexts. To overcome this limitation, we formalize analogy transduction as synthesizing new plans by composing task-endogenous analogies with given contexts and propose a novel analogy representation tailored for it. Grounded in our theory, this analogy representation captures what changes under optimal task execution, remains invariant to contextual variations, and is sufficient for optimal goal reaching. We further contend that generalization to unseen analogy-context pairs is a practical obstacle in analogy transduction, and introduce a new approach for offline GCRL that enables analogy transduction beyond seen pairs to unseen combinations. We empirically demonstrate the effectiveness of our approach on OGBench manipulation environments, substantially outperforming prior methods that do not perform analogy transduction. Project page: this https URL

**arXiv ID:** 2605.20609
</details>

<details>
<summary><strong>Learning First Integrals via Backward-Generated Data and Guided Reinforcement Learning</strong> - Jingfeng Zhong, Zhengxiang Liu, Zhijie Wang, Shuai Li - [[pdf]](https://arxiv.org/pdf/2605.21160)</summary>

**Abstract:** The discovery of first integrals is of fundamental scientific importance for understanding conservation laws in dynamical systems. However, existing symbolic computation tools and Large Language Models (LLMs) remain limited on this task because high-quality training data are scarce and successful solutions often depend on mathematical intuition. This paper presents FISolver, an LLM-based solver developed to address this challenge. First, we introduce a "Backward Generation" algorithm that systematically builds large-scale datasets of (differential equation, first integral) pairs by deriving differential equations from sampled integrals, thereby alleviating the data scarcity bottleneck. Second, we apply supervised fine-tuning to a compact mathematical model and further improve its performance through reinforcement learning with a Levenshtein Distance-based shaped reward. In addition, we design data synthesis and blending strategies that support effective adaptation to difficult problem families from sparse examples. Experiments show that FISolver, while requiring substantially lower computational cost, significantly outperforms larger mathematical LLMs and commercial solvers such as Mathematica on challenging benchmarks, indicating a new data-driven route for automated discovery of first integrals.

**arXiv ID:** 2605.21160
</details>

<details>
<summary><strong>Domain-Adaptable Reinforcement Learning for Code Generation with Dense Rewards</strong> - Erfan Aghadavoodi Jolfaei, Daniel Maninger, Abhinav Anand, Mert Tiftikci, Mira Mezini - [[pdf]](https://arxiv.org/pdf/2605.21180)</summary>

**Abstract:** Large language models show strong potential for automated code generation, but lack guarantees for correctness, quality, safety, and domain-specific constraints. For instance in robotics, where code generation is increasingly being used for planning and executing actions, awareness of the environment and physical constraints is critical. To facilitate the adaption of code-generating LLMs to diverse requirements, including domain-specific ones, we present a reinforcement learning framework that fine-tunes pre-trained LLMs using proximal policy optimization. Our customizable execution-aware reward formula captures and optimizes syntax, functional correctness, code style, security, and simulator executability. A token-level reward mapping mechanism enables effective credit assignment from execution outcomes to generated tokens. The framework is evaluated on general-purpose code generation (MBPP/MBPP+) and robotic program synthesis (RoboEval). The results show substantial improvements in functional correctness and simulator executability, including an absolute pass@1 increase of 19% on MBPP and a reduction in execution failures by 51% on RoboEval. These findings demonstrate that structured reinforcement learning can effectively align language models to correct program generation and domain-specific requirements.

**arXiv ID:** 2605.21180
</details>

<details>
<summary><strong>Q-SpiRL: Quantum Spiking Reinforcement Learning for Adaptive Robot Navigation</strong> - Mohamed Khair Altrabulsi, Nouhaila Innan, Alberto Marchisio, Muhammad Kashif, Muhammad Shafique - [[pdf]](https://arxiv.org/pdf/2605.20801)</summary>

**Abstract:** Adaptive robot navigation in dynamic environments requires policies that can reach the target reliably while producing efficient and stable trajectories. This paper presents Q-SpiRL, a quantum spiking reinforcement learning framework for obstacle-aware robot navigation. The framework develops and evaluates five agent families: tabular Q-learning, classical MLP, classical SNN, quantum-enhanced MLP (QMLP), and quantum-enhanced spiking neural network (QSNN). While all models are implemented under a unified training and evaluation pipeline, the QSNN is the central architecture of interest, as it combines spike-based temporal processing with variational quantum feature transformation. Experiments are conducted across three grid-world environments of increasing size, namely 20x20, 30x30, and 40x40, with both static and dynamic obstacles. Performance is assessed using success rate, success-weighted path length, path length, and turn rate under deterministic inference. Results show that QSNN achieves the strongest overall trade-off between task completion, trajectory efficiency, and motion smoothness, reaching up to 99% success rate while maintaining high path efficiency in the most challenging setting. Execution on IBM quantum hardware further demonstrates the feasibility of deploying the proposed hybrid policy under real-device conditions.

**arXiv ID:** 2605.20801
</details>

<details>
<summary><strong>Reinforcement Learning for Risk Adaptation via Differentiable CVaR Barrier Functions</strong> - Xinyi Wang, Taekyung Kim, Bardh Hoxha, Georgios Fainekos, Dimitra Panagou - [[pdf]](https://arxiv.org/pdf/2605.21257)</summary>

**Abstract:** Planning through crowded environments under uncertain obstacle motions remains difficult, as stochastic interactions often induce overly conservative behavior or reduced efficiency. To address this challenge, we propose an end-to-end risk adaptation framework for crowd navigation under obstacle-motion uncertainty modeled by a Gaussian mixture model. The framework combines reinforcement learning~(RL) with a differentiable quadratic-program safety layer based on Conditional Value-at-Risk~(CVaR) barrier functions, jointly learning nominal control input, risk level, and safety margin and enforcing explicit probabilistic safety constraints. This design enables context-aware adaptation, promoting efficient behavior while invoking caution only when necessary. We conduct extensive evaluations in dynamic, uncertain, and crowded environments across varying obstacle densities and robot models, and further assess generalization under three out-of-distribution cases. Comparisons across optimization-based, RL-based, and integrated RL and optimization methods are provided, and the proposed method is shown to deliver the strongest overall performance in safety, efficiency, and generalization under uncertainty.

**arXiv ID:** 2605.21257
</details>

<details>
<summary><strong>Affordance-R1: Reinforcement Learning for Generalizable Affordance Reasoning in Multimodal Large Language Model</strong> - Hanqing Wang, Shaoyang Wang, Yiming Zhong, Zemin Yang, Jiamin Wang, Zhiqing Cui, Jiahao Yuan, Yifan Han, Mingyu Liu, Yuexin Ma - [[pdf]](https://arxiv.org/pdf/2508.06206)</summary>

**Abstract:** Affordance grounding focuses on predicting the specific regions of objects that are associated with the actions to be performed by robots. It plays a vital role in the fields of human-robot interaction, human-object interaction, embodied manipulation, and embodied perception. Existing models often neglect the affordance shared among different objects because they lack the Chain-of-Thought(CoT) reasoning abilities, limiting their out-of-domain (OOD) generalization and explicit reasoning capabilities. To address these challenges, we propose Affordance-R1, the first unified affordance grounding framework that integrates cognitive CoT guided Group Relative Policy Optimization (GRPO) within a reinforcement learning paradigm. Specifically, we designed a sophisticated affordance function, which contains format, perception, and cognition rewards to effectively guide optimization directions. Furthermore, we constructed a high-quality affordance-centric reasoning dataset, ReasonAff, to support training. Trained exclusively via reinforcement learning with GRPO and without explicit reasoning data, Affordance-R1 achieves robust zero-shot generalization and exhibits emergent test-time reasoning capabilities. Comprehensive experiments demonstrate that our model outperforms well-established methods and exhibits open-world generalization. To the best of our knowledge, Affordance-R1 is the first to integrate GRPO-based RL with reasoning into affordance reasoning. The code of our method and our dataset is released on this https URL.

**arXiv ID:** 2508.06206
</details>

<details>
<summary><strong>ARC-RL: A Reinforcement Learning Playground Inspired by ARC Raiders</strong> - Carlo Romeo, Andrew D. Bagdanov - [[pdf]](https://arxiv.org/pdf/2605.19503)</summary>

**Abstract:** Reinforcement learning for legged locomotion has matured into a stack of multi-component reward functions and physics-engine benchmarks whose morphologies are uniformly derived from real commercial hardware. Game NPCs, however, are bound by stylistic constraints absent from sim-to-real robotics and routinely take the form of creatures with no real-robot counterpart. We introduce ARC-RL, a suite of four MuJoCo continuous-control environments featuring robotic morphologies inspired by the bestiary of ARC Raiders: the 18-DoF tall hexapod Queen, the 12-DoF armoured hexapod Bastion, the 18-DoF compact hexapod Tick, and the 12-DoF quadruped Leaper. All four robots share a unified observation template, action convention, simulation cadence, and a single closed-form multi-component reward function whose only per-morphology variation lives in a small set of weights and parameters. The reward fuses a velocity-tracking tent, a healthy survive bonus, a phase-locked gait-compliance bonus/cost pair, action regularisers, three safety penalties, and a posture anchor; no motion-capture data enters the reward at any point. We additionally provide hand-crafted Central Pattern Generator demonstrators per morphology, which serve both as fixed expert references and as sources of prior data for offline-to-online training. On this playground, we conduct a controlled empirical study comparing standard online algorithms (SAC, SPEQ, SOPE-EO) and methods augmented with prior data (SACfD, SPEQ-O2O, SOPE), and characterise how each paradigm copes with the playground's morphological diversity and animation-style stylistic constraints. Source code is available at this https URL.

**arXiv ID:** 2605.19503
</details>

<details>
<summary><strong>Multimodal Fusion for Sim2real Transfer in Visual Reinforcement Learning</strong> - Zichun Xu, Jingdong Zhao, Chenyu Guo, Qianxue Zhang, Liao Zhang, Xiao Zhang, Yiming Ren, Lian Zhang, Zengren Zhao - [[pdf]](https://arxiv.org/pdf/2507.09180)</summary>

**Abstract:** Depth information is robust to scene appearance variations and inherently carries 3D spatial details. Thus, a visual backbone based on the vision transformer is proposed to fuse RGB and depth modalities for enhancing generalization in this paper. Different modalities are first processed by separate CNN stems, and the combined convolutional features are delivered to the scalable vision transformer to obtain visual representations. Moreover, a contrastive learning scheme is designed with masked and unmasked tokens to enhance the sample efficiency and generalization performance. A curriculum-based domain randomization scheme is used to flexibly stabilize the training process. Finally, simulation results demonstrate that our fusion scheme outperforms the other baselines. The feasibility of our model is validated to perform real-world manipulation tasks via zero-shot transfer.

**arXiv ID:** 2507.09180
</details>

<details>
<summary><strong>RankQ: Offline-to-Online Reinforcement Learning via Self-Supervised Action Ranking</strong> - Andrew Choi, Wei Xu - [[pdf]](https://arxiv.org/pdf/2605.11151)</summary>

**Abstract:** Offline-to-online reinforcement learning (RL) improves sample efficiency by leveraging pre-collected datasets prior to online interaction. A key challenge, however, is learning an accurate critic in large state--action spaces with limited dataset coverage. To mitigate harmful updates from value overestimation, prior methods impose pessimism by down-weighting out-of-distribution (OOD) actions relative to dataset actions. While effective, this essentially acts as a behavior cloning anchor and can hinder downstream online policy improvement when dataset actions are suboptimal. We propose RankQ, an offline-to-online Q-learning objective that augments temporal-difference learning with a self-supervised multi-term ranking loss to enforce structured action ordering. By learning relative action preferences rather than uniformly penalizing unseen actions, RankQ shapes the Q-function such that action gradients are directed toward higher-quality behaviors. Across sparse reward D4RL benchmarks, RankQ achieves performance competitive with or superior to seven prior methods. In vision-based robot learning, RankQ enables effective offline-to-online fine-tuning of a pretrained vision-language-action (VLA) model in a low-data regime, achieving on average a 42.7% higher simulation success rate than the next best method. In a high-data setting, RankQ improves simulation performance by 13.7% over the next best method and achieves strong sim-to-real transfer, increasing real-world cube stacking success from 43.1% to 88.9% relative to the VLA's initial performance.

**arXiv ID:** 2605.11151
</details>

<details>
<summary><strong>PaintCopilot: Modeling Painting as Autonomous Artistic Continuation</strong> - Yunge Wen, Yuancheng Shen, Paul Pu Liang - [[pdf]](https://arxiv.org/pdf/2605.20941)</summary>

**Abstract:** We present PaintCopilot, a co-creative neural painting assistant that models painting as an open-ended autoregressive artistic behavior conditioned on evolving canvas states and prior brushstroke history, without requiring a target image. Unlike existing neural painting methods that frame painting as pixel reconstruction toward a predefined reference, PaintCopilot predicts future strokes directly from learned artistic dynamics, analogous to how large language models continue text sequences from prior context.
The framework proposes three complementary models: a ViT-based Target Predictor that infers artist intent from partial canvas observations, an autoregressive Next Stroke Predictor that generates temporally coherent brushstrokes via flow matching, and a VAE-based Region Sampler that synthesizes semantically localized stroke sequences on demand. Built on three differentiable brush representations (Hard Round, Brush Tip, and 2D Gaussian), the system supports four interactive workflows: Optimize History, Stroke Completion, Region Inpainting, and Dynamic Brush. Through case studies with professional artists, we demonstrate that PaintCopilot enables fluid co-creative painting workflows in which artists and AI continuously alternate control throughout the creative process.

**arXiv ID:** 2605.20941
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
