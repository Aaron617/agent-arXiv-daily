# Agent arXiv Daily

**Last Updated:** 2026-05-07 04:00:53

**Total Papers:** 76

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (4 papers)</h2></summary>

<details>
<summary><strong>Accountable Agents in Software Engineering: An Analysis of Terms of Service and a Research Roadmap</strong> - Christoph Treude - [[pdf]](https://arxiv.org/pdf/2605.04532)</summary>

**Abstract:** AI coding assistants and autonomous agents are becoming integral to software development workflows, reshaping how code is produced, reviewed, and maintained. While recent research has focused mainly on the capabilities and impacts of productivity of these systems, much less attention has been paid to accountability: who is responsible when agents generate, modify, or recommend code? In practice, accountability is defined through the Terms of Service (ToS) and related policy documents that govern the use of AI-powered development tools.
In this vision paper, we present a comparative analysis of the Terms of Service for widely used AI coding assistants and agent-enabled development tools. We examine how these documents allocate ownership, responsibility, liability, and disclosure obligations between tool providers and software developers, and we identify common patterns and divergences between providers. Our analysis reveals a consistent tendency to shift responsibility for correctness, safety, and legal compliance onto users, as well as substantial variation in how providers address issues such as indemnification, data reuse, and acceptable use.
Based on these findings, we argue that existing policy frameworks are poorly aligned with increasingly agent-mediated and autonomous software development workflows. We outline a research roadmap for accountable agents in software engineering, identifying challenges and opportunities for modeling responsibility, designing governance artifacts, developing tooling that supports accountability, and conducting empirical studies of developers' perceptions and practices.

**arXiv ID:** 2605.04532
</details>

<details>
<summary><strong>A Brief Overview: Agentic Reinforcement Learning In Large Language Models</strong> - Fangming Cui, Ruixiao Zhu, Cheng Fang, Sunan Li, Jiahong Li - [[pdf]](https://arxiv.org/pdf/2604.27859)</summary>

**Abstract:** Reinforcement Learning (RL) has traditionally focused on training specialized agents to optimize predefined reward functions within narrowly defined environments. However, the advent of powerful Large Language Models (LLMs) and increasingly complex, open-ended tasks has catalyzed a paradigm shift towards agentic paradigms within RL. This emerging framework extends beyond traditional RL by emphasizing the development of autonomous agents capable of goal-setting, long-term planning, dynamic strategy adaptation, and interactive reasoning in uncertain, real-world environments. Unlike conventional approaches that rely heavily on static objectives and episodic interactions, LLM-based Agentic RL incorporates cognitive-like capabilities such as meta-reasoning, self-reflection, and multi-step decision-making directly into the learning loop. In this paper, we provide a deep insight for looking the conceptual foundations, methodological innovations, and effective designs underlying this trend. Furthermore, we identify critical challenges and outline promising future directions for building LLM-based Agentic RL.

**arXiv ID:** 2604.27859
</details>

<details>
<summary><strong>Caesar: Deep Agentic Web Exploration for Creative Answer Synthesis</strong> - Jason Liang, Elliot Meyerson, Risto Miikkulainen - [[pdf]](https://arxiv.org/pdf/2604.20855)</summary>

**Abstract:** To advance from passive retrieval to creative discovery of new ideas, autonomous agents must be capable of deep, associative synthesis. However, current agentic frameworks prioritize convergent search, often resulting in derivative summaries that lack creativity. Caesar is an agentic architecture designed to bridge the gap between information gathering and synthesis of new insights. Unlike existing agents that treat the web as a flat sequence of disconnected documents, Caesar performs a deep web traversal to construct a dynamic knowledge graph. This graph then serves as a navigational scaffold, guiding the agent to diverse, non-obvious information that flat retrieval would never encounter. Caesar thus consists of two components: (1) exploration driven by a dynamic context-aware policy that maximizes information coverage across the web's topological structure, and (2) synthesis through adversarial refinement that actively seeks novel perspectives rather than confirming established priors. Caesar demonstrates the ability to generate artifacts and answers characterized by high novelty and structural coherence, achieving 13% to 23% improvement over state-of-the-art deep research agents in creative synthesis challenges, with strong dominance across all output formats.

**arXiv ID:** 2604.20855
</details>

<details>
<summary><strong>LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey</strong> - Henry Peng Zou, Wei-Chieh Huang, Yaozu Wu, Jizhou Guo, Yankai Chen, Chunyu Miao, Hoang Nguyen, Yue Zhou, Weizhi Zhang, Liancheng Fang, Hanrong Zhang, Fangxin Wang, Pengfei Zhang, Huacan Wang, Langzhou He, Yangning Li, Dongyuan Li, Renhe Jiang, Xue Liu, Philip S. Yu - [[pdf]](https://arxiv.org/pdf/2505.00753)</summary>

**Abstract:** Recent advances in large language models (LLMs) have sparked growing interest in building fully autonomous agents. However, fully autonomous LLM-based agents still face significant challenges, including limited reliability due to hallucinations, difficulty in handling complex tasks, and substantial safety and ethical risks, all of which limit their feasibility and trustworthiness in real-world applications. To overcome these limitations, LLM-based human-agent systems (LLM-HAS) incorporate human-provided information, feedback, or control into the agent system to enhance system performance, reliability, and safety. These human-agent collaboration systems enable humans and LLM-based agents to collaborate effectively by leveraging their complementary strengths. This paper provides the first comprehensive and structured survey of LLM-HAS. It clarifies fundamental concepts, systematically presents core components shaping these systems, including environment and profiling, human feedback, interaction types, orchestration, and communication, explores emerging applications, and discusses unique challenges and opportunities arising from human-AI collaboration. By consolidating current knowledge and offering a structured overview, we aim to foster further research and innovation in this rapidly evolving interdisciplinary field. Paper lists and resources are available at this https URL.

**arXiv ID:** 2505.00753
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (14 papers)</h2></summary>

<details>
<summary><strong>AgentTrust: Runtime Safety Evaluation and Interception for AI Agent Tool Use</strong> - Chenglin Yang - [[pdf]](https://arxiv.org/pdf/2605.04785)</summary>

**Abstract:** Modern AI agents execute real-world side effects through tool calls such as file operations, shell commands, HTTP requests, and database queries. A single unsafe action, including accidental deletion, credential exposure, or data exfiltration, can cause irreversible harm. Existing defenses are incomplete: post-hoc benchmarks measure behavior after execution, static guardrails miss obfuscation and multi-step context, and infrastructure sandboxes constrain where code runs without understanding what an action means.
We present AgentTrust, a runtime safety layer that intercepts agent tool calls before execution and returns a structured verdict: allow, warn, block, or review. AgentTrust combines a shell deobfuscation normalizer, SafeFix suggestions for safer alternatives, RiskChain detection for multi-step attack chains, and a cache-aware LLM-as-Judge for ambiguous inputs.
We release a 300-scenario benchmark across six risk categories and an additional 630 independently constructed real-world adversarial scenarios. On the internal benchmark, the production-only ruleset achieves 95.0% verdict accuracy and 73.7% risk-level accuracy at low-millisecond end-to-end latency. On the 630-scenario benchmark, evaluated under a patched ruleset and not claimed as zero-shot, AgentTrust achieves 96.7% verdict accuracy, including about 93% on shell-obfuscated payloads. AgentTrust is released under the AGPL-3.0 license and provides a Model Context Protocol server for MCP-compatible agents.

**arXiv ID:** 2605.04785
</details>

<details>
<summary><strong>LongSeeker: Elastic Context Orchestration for Long-Horizon Search Agents</strong> - Yijun Lu, Rui Ye, Yuwen Du, Jiajun Wang, Songhua Liu, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2605.05191)</summary>

**Abstract:** Long-horizon search agents must manage a rapidly growing working context as they reason, call tools, and observe information. Naively accumulating all intermediate content can overwhelm the agent, increasing costs and the risk of errors. We propose that effective context management should be adaptive: parts of the agent's trajectory are maintained at different levels of detail depending on their current relevance to the task. To operationalize this principle, we introduce Context-ReAct, a general agentic paradigm for elastic context orchestration that integrates reasoning, context management, and tool use in a unified loop. Context-ReAct provides five atomic operations: Skip, Compress, Rollback, Snippet and Delete, which allow the agent to dynamically reshape its working context, preserving important evidence, summarizing resolved information, discarding unhelpful branches, and controlling context size. We prove that the Compress operator is expressively complete, while the other specialized operators provide efficiency and fidelity guarantees that reduce generation cost and hallucination risk. Building on this paradigm, we develop LongSeeker, a long-horizon search agent fine-tuned from Qwen3-30B-A3B on 10k synthesized trajectories. Across four representative search benchmarks, LongSeeker achieves 61.5% on BrowseComp and 62.5% on BrowseComp-ZH, substantially outperforming Tongyi DeepResearch (43.2% and 46.7%) and AgentFold (36.2% and 47.3%). These results highlight the potential of adaptive context management, showing that agents can achieve more reliable and efficient long-horizon reasoning by actively shaping their working memory.

**arXiv ID:** 2605.05191
</details>

<details>
<summary><strong>TSCG: Deterministic Tool-Schema Compilation for Agentic LLM Deployments</strong> - Furkan Sakizli - [[pdf]](https://arxiv.org/pdf/2605.04107)</summary>

**Abstract:** Production agent frameworks (OpenAI Function Calling, Anthropic Tool Use, MCP) transmit tool schemas as JSON, a format designed for machine parsing, not for interpretation by language models. For small models (4B-14B), this protocol mismatch accounts for the majority of tool-use failure at production catalog sizes. We present TSCG, a deterministic tool-schema compiler that resolves this mismatch at the API boundary, converting JSON schemas into token-efficient structured text without model access, fine-tuning, or runtime search. TSCG combines eight composable operators with a formal compression bound (>=51% on well-formed schemas).
On TSCG-Agentic-Bench (about 19,000 calls, 12 models, 5 scenarios), TSCG restores Phi-4 14B from 0% to 84.4% accuracy at 20 tools (90.3% at 50 tools) and achieves 108-181% accuracy-retained ratio across three models on BFCL. Format-versus-compression decomposition (R^2=0.88 -> 0.03) establishes representation change as the dominant mechanism. Per-operator isolation across three frontier models reveals three distinct operator-response profiles: operator-hungry (Opus 4.7), operator-sensitive (GPT-5.2), and operator-robust (Sonnet 4), providing per-model deployment guidance. Scaling experiments show accuracy advantages persisting on heavy production MCP schemas (+5.0 pp at about 10,500 input tokens) despite saturation on light synthetic catalogs, with 52-57% token savings throughout. The synthetic benchmark generalizes to real MCP schemas within 0.1 accuracy points. TSCG ships as a 1,200-line zero-dependency TypeScript package.

**arXiv ID:** 2605.04107
</details>

<details>
<summary><strong>BEAGLE: Behavior-Enforced Agent for Grounded Learner Emulation</strong> - Hanchen David Wang, Clayton Cohn, Zifan Xu, Siyuan Guo, Gautam Biswas, Meiyi Ma - [[pdf]](https://arxiv.org/pdf/2602.13280)</summary>

**Abstract:** Simulating student learning behaviors in open-ended problem-solving environments holds potential for education research, from training adaptive tutoring systems to stress-testing pedagogical interventions. However, collecting authentic data is challenging due to privacy concerns and the high cost of longitudinal studies. While Large Language Models (LLMs) offer a promising path to student simulation, they suffer from competency bias, optimizing for efficient correctness rather than the erratic, iterative struggle characteristic of novice learners. We present BEAGLE, a neuro-symbolic framework that addresses this bias by incorporating Self-Regulated Learning (SRL) theory into a novel architecture. BEAGLE integrates three key technical innovations: (1) a semi-Markov model that governs the timing and transitions of cognitive behaviors and metacognitive behaviors; (2) Bayesian Knowledge Tracing with explicit flaw injection to enforce realistic knowledge gaps and "unknown unknowns"; and (3) a decoupled agent design that separates high-level strategy use from code generation actions to prevent the model from silently correcting its own intentional errors. In evaluations on Python programming tasks, BEAGLE significantly outperforms state-of-the-art baselines in reproducing authentic trajectories. In a human Turing test, participants could not reliably tell BEAGLE traces apart from real student data: classification accuracy was statistically equivalent to chance (52.8%, d' = 0.15, N = 71)

**arXiv ID:** 2602.13280
</details>

<details>
<summary><strong>The Effects of Visual Priming on Cooperative Behavior in Vision-Language Models</strong> - Kenneth J. K. Ong - [[pdf]](https://arxiv.org/pdf/2604.27953)</summary>

**Abstract:** As Vision-Language Models (VLMs) become increasingly integrated into decision-making systems, it is essential to understand how visual inputs influence their behavior. This paper investigates the effects of visual priming on VLMs' cooperative behavior using the Iterated Prisoner's Dilemma (IPD) as a test scenario. We examine whether exposure to images depicting behavioral concepts (kindness/helpfulness vs. aggressiveness/selfishness) and color-coded reward matrices alters VLM decision patterns. Experiments were conducted across multiple state-of-the-art VLMs. We further explore mitigation strategies including prompt modifications, Chain of Thought (CoT) reasoning, and visual token reduction. Results show that VLM behavior can be influenced by both image content and color cues, with varying susceptibility and mitigation effectiveness across models. These findings not only underscore the importance of robust evaluation frameworks for VLM deployment in visually rich and safety-critical environments, but also highlight how architectural and training differences among models may lead to distinct behavioral responses-an area worthy of further investigation.

**arXiv ID:** 2604.27953
</details>

<details>
<summary><strong>CreativityBench: Evaluating Agent Creative Reasoning via Affordance-Based Tool Repurposing</strong> - Cheng Qian, Hyeonjeong Ha, Jiayu Liu, Jeonghwan Kim, Jiateng Liu, Bingxuan Li, Aditi Tiwari, Dwip Dalal, Zhenhailong Wang, Xiusi Chen, Mahdi Namazifar, Yunzhu Li, Heng Ji - [[pdf]](https://arxiv.org/pdf/2605.02910)</summary>

**Abstract:** Recent advances in large language models have led to strong performance on reasoning and environment-interaction tasks, yet their ability for creative problem-solving remains underexplored. We study this capability through the lens of creative tool use, where a model repurposes available objects by reasoning about their affordances and attributes rather than relying on canonical usage. As a first step, we introduce CreativityBench, a benchmark for evaluating affordance-based creativity in LLMs. To this end, we build a large-scale affordance knowledge base (KB) with 4K entities and 150K+ affordance annotations, explicitly linking objects, parts, attributes, and actionable uses. Building on this KB, we generate 14K grounded tasks that require identifying non-obvious yet physically plausible solutions under constraints. Evaluations across 10 state-of-the-art LLMs, including closed and open-source models, show that models can often select a plausible object, but fail to identify the correct parts, their affordances, and the underlying physical mechanism needed to solve the task, leading to a significant drop in performance. Furthermore, improvements from model scaling quickly saturate, strong general reasoning does not reliably translate to creative affordance discovery, and common inference-time strategies such as Chain-of-Thought yield limited gains. These results suggest that creative tool use remains a major challenge for current models, and that CreativityBench provides a useful testbed for studying this missing dimension of intelligence, with potential implications for planning and reasoning modules in future agents.

**arXiv ID:** 2605.02910
</details>

<details>
<summary><strong>ADAPTS: Agentic Decomposition for Automated Protocol-agnostic Tracking of Symptoms</strong> - Alexandria K. Vail, Marcelo Cicconet, Katie Aafjes-van Doorn, Ryan Maroney, Marc Aafjes - [[pdf]](https://arxiv.org/pdf/2605.03212)</summary>

**Abstract:** Modeling latent clinical constructs from unconstrained clinical interactions is a unique challenge in affective computing. We present ADAPTS (Agentic Decomposition for Automated Protocol-agnostic Tracking of Symptoms), a framework for automated rating of depression and anxiety severity using a mixture-of-agents LLM architecture. This approach decomposes long-form clinical interviews into symptom-specific reasoning tasks, producing auditable justifications while preserving temporal and speaker alignment. Generalization was evaluated across two independent datasets ($N=204$) with distinct interview structures. On high-discrepancy interviews, automated ratings approximated expert benchmarks ($\text{absolute error}=22$) more closely than original human ratings ($\text{absolute error}=26$). Implementing an ``extended'' protocol that incorporates qualitative clinical conventions significantly stabilized ratings, with absolute agreement reaching $\text{ICC(2,1)} = 0.877$. These findings suggest that the ADAPTS framework enables promising evaluations of psychiatric severity. While the current implementation is purely text-based, the underlying architecture is readily extensible to multimodal inputs, including acoustic and visual features. By approximating expert-level precision in a protocol-agnostic manner, this framework provides a foundation for objective and scalable psychiatric assessment, especially in resource-limited settings.

**arXiv ID:** 2605.03212
</details>

<details>
<summary><strong>SWE-WebDevBench: Evaluating Coding Agent Application Platforms as Virtual Software Agencies</strong> - Siddhant Saxena, Nilesh Trivedi, Vinayaka Jyothi - [[pdf]](https://arxiv.org/pdf/2605.04637)</summary>

**Abstract:** The emergence of "vibe coding" platforms, where users describe applications in natural language and AI agents autonomously generate full-stack software, has created a need for rigorous evaluation beyond code-level benchmarks. In order to assess them as virtual software development agencies on understanding business requirements, making architectural decisions, writing production code, handling iterative modifications, and maintaining business readiness, we introduce SWE-WebDev Bench, a 68-metric evaluation framework spanning 25 primary and 43 diagnostic metrics across seven groups, organized along three dimensions: Interaction Mode (App Creation Request (ACR) vs. App Modification Request (AMR)), Agency Angle (Product Manager (PM), Engineering, Ops), and Complexity Tier (T4 multi-role SaaS, T5 AI-native).
Our evaluation (six platforms, three domains, 18 evaluation cells) reveals four recurring shortcomings in the current generation of AI app builders: (1) A specification bottleneck, where platforms compress rich business requirements into oversimplified technical plans, (2) A pervasive frontend-backend decoupling, where visually polished UIs mask absent or broken backend infrastructure, (3) A steep production-readiness cliff, where no platform scores above 60% on engineering quality and post-generation human effort varies substantially across platforms and (4) Widespread security and infrastructure failures, with no platform exceeding 65% Security Score against a 90% target and concurrency handling as low as 6%. These observations are descriptive of our sample and require larger-scale replication to establish generality. We release SWE-WebDev Bench as a community benchmark to enable such replication and help platform builders identify and address these gaps.
Code and benchmark resources are available at: this https URL and this https URL.

**arXiv ID:** 2605.04637
</details>

<details>
<summary><strong>Bridging Perception and Action: A Lightweight Multimodal Meta-Planner Framework for Robust Earth Observation Agents</strong> - Jinghui Xu, Boyi Shangguan, Mengke Zhu, Hao Liu, Junhuan Jiang, Guangjun He, Pengming Feng, Shichao Jin, Bin Liang, Yongzhe Chang, Junbo Tan, Tiantian Zhang, Xueqian Wang - [[pdf]](https://arxiv.org/pdf/2605.04777)</summary>

**Abstract:** Autonomous Earth Observation (EO) agents are transitioning from passive perception to complex, multi-step task execution. However, current architectures that integrate planning and execution within a single model often struggle with combinatorial complexity and reasoning errors in dynamic EO scenarios. To resolve these challenges, we propose the Lightweight Multimodal Meta-Planner (LMMP) framework. LMMP incorporates a dual-awareness mechanism that grounds strategic plans in both multimodal image features and high-level task semantics. Crucially, we introduce a Meta Task Library to inject remote sensing expert knowledge directly into the workflow, which standardizes domain logic and ensures plans are physically feasible. We further implement a two-stage training pipeline, initializing the Meta-Planner via expert-distilled Supervised Fine-Tuning and refining it through Direct Preference Optimization based on execution feedback. Extensive experiments on a dataset derived from EarthBench and ThinkGeo demonstrate that LMMP significantly improves tool-calling accuracy and task success rates. Moreover, the framework exhibits strong ``plug-and-play'' versatility, consistently enhancing the performance of diverse executor backbones across previously unseen EO missions.

**arXiv ID:** 2605.04777
</details>

<details>
<summary><strong>From SWE-ZERO to SWE-HERO: Execution-free to Execution-based Fine-tuning for Software Engineering Agents</strong> - Nikolai Ludwig, Wasi Uddin Ahmad, Somshubra Majumdar, Boris Ginsburg - [[pdf]](https://arxiv.org/pdf/2604.01496)</summary>

**Abstract:** We introduce SWE-ZERO to SWE-HERO, a two-stage SFT recipe that achieves state-of-the-art results on SWE-bench by distilling open-weight frontier LLMs. Our pipeline replaces resource-heavy dependencies with an evolutionary refinement strategy: (1) SWE-ZERO utilizes large-scale, execution-free trajectories to master code semantics and repository-level reasoning, and (2) SWE-HERO applies targeted, execution-backed refinement to transition these semantic intuitions into rigorous engineering workflows. Our empirical results set a new benchmark for open-source models of comparable size. We release a dataset of 300k SWE-ZERO and 13k SWE-HERO trajectories distilled from Qwen3-Coder-480B, alongside a suite of agents based on the Qwen2.5-Coder series. Notably, SWE-HERO-32B achieves a 62.2% resolution rate on SWE-bench Verified. Furthermore, despite being trained exclusively on Python, our agents demonstrate robust zero-shot transferability on SWE-bench Multilingual, reaching 44.1% and confirming the paradigm's generalizability across diverse languages.

**arXiv ID:** 2604.01496
</details>

<details>
<summary><strong>Agentic Vulnerability Reasoning on Windows COM Binaries</strong> - Hwiwon Lee, Jongseong Kim, Lingming Zhang - [[pdf]](https://arxiv.org/pdf/2605.05000)</summary>

**Abstract:** Windows Component Object Model (COM) services run with elevated privileges and are widely accessible to authenticated users, making race conditions in these binaries a critical surface for local privilege escalation. We present SLYP, an end-to-end agentic pipeline that discovers race condition vulnerabilities in COM binaries and generates debugger-verified proof-of-concept (PoC) code. SLYP exposes binary exploration, COM inspection, and dynamic debugging as reusable tool interfaces, giving agents the static context, COM activation metadata, and debugger feedback needed to move from vulnerability discovery to verified PoC generation. On a benchmark of 20 COM objects covering 40 vulnerability cases, SLYP achieves 0.973 F1, outperforming production coding agents by up to 0.208 F1 and the state-of-the-art static analyzer by 3.3x in bug discovery. For PoC generation, production coding agents in their default setup (without our COM inspection and dynamic debugging tools) verify essentially no cases on either frontier model, whereas SLYP's interactive toolsets enable it to autonomously synthesize working PoCs for 67.5% of cases on the strongest configuration. Deployed on production Windows services, SLYP discovers 28 previously unknown vulnerabilities across nine COM services, all confirmed by the Microsoft Security Response Center (MSRC) with 16 CVEs assigned and $140,000 in bounties. Furthermore, SLYP is designed with generalizable binary analysis and debugging interfaces, making it readily applicable to other commercial off-the-shelf (COTS) binaries beyond Windows COM services.

**arXiv ID:** 2605.05000
</details>

<details>
<summary><strong>Beyond Fixed Thresholds and Domain-Specific Benchmarks for Explainable Multi-Task Classification in Autonomous Vehicles</strong> - Maryam Sadat Hosseini Azad, Shahriar Baradaran Shokouhi - [[pdf]](https://arxiv.org/pdf/2605.04299)</summary>

**Abstract:** Scene understanding is a vital part of autonomous driving systems, which requires the use of deep learning models. Deep learning methods are intrinsically black box models, which lack transparency and safety in autonomous driving. To make these systems transparent, multi-task visual understanding has become crucial for explainable autonomous driving perception systems, where simultaneous prediction of multiple driving behaviors and their underlying explanations is essential for safe navigation and human trust in autonomous vehicles. In order to design an accurate and cross-cultural explainable autonomous driving system, we introduce a comprehensive confidence threshold sensitivity analysis that evaluates various threshold values to identify optimal decision boundaries for different tasks. Our analysis demonstrates that traditional fixed threshold approaches are suboptimal for multi-task scenarios. Through extensive evaluation, we demonstrate that our adaptive threshold selection methodology improves F1-scores across different tasks. In addition, we introduce IUST-XAI-AD, a novel dataset consisting of 958 images with human annotations for driving decisions and corresponding reasoning. This dataset addresses the critical gap in domain-specific evaluation benchmarks for distinct driving contexts and provides a more challenging test environment compared to existing datasets. Experimental results demonstrate that confidence threshold sensitivity analysis can significantly improve model performance, while the introduction of the IUST-XAI-AD dataset reveals important insights about cross-cultural driving behavior patterns. The combined contributions of this work provide both methodological advances and practical evaluation tools that can accelerate the development of more reliable, explainable, and culturally-adaptive autonomous driving systems for global deployment.

**arXiv ID:** 2605.04299
</details>

<details>
<summary><strong>Multi-Source Human-in-the-Loop Digital Twin Testbed for Connected and Autonomous Vehicles in Mixed Traffic Flow</strong> - Jianghong Dong, Chunying Yang, Mengchi Cai, Chaoyi Chen, Qing Xu, Jianqiang Wang, Jiawei Wang, Keqiang Li - [[pdf]](https://arxiv.org/pdf/2603.17751)</summary>

**Abstract:** In the emerging mixed traffic environments, Connected and Autonomous Vehicles (CAVs) have to interact with surrounding human-driven vehicles (HDVs). This paper introduces MSH-MCCT (Multi-Source Human-in-the-Loop Mixed Cloud Control Testbed), a novel CAV testbed that captures complex interactions between various CAVs and HDVs. Utilizing the Mixed Digital Twin concept, which combines Mixed Reality with Digital Twin, MSH-MCCT integrates physical, virtual, and mixed platforms, along with multi-source control inputs. Bridged by the mixed platform, MSH-MCCT allows human drivers and CAV algorithms to operate both physical and virtual vehicles within multiple fields of view. Particularly, this testbed facilitates the coexistence and real-time interaction of physical and virtual CAVs \& HDVs, significantly enhancing the experimental flexibility and scalability. Experiments on vehicle platooning in mixed traffic showcase the potential of MSH-MCCT to conduct CAV testing with multi-source real human drivers in the loop through driving simulators of diverse fidelity. The videos for the experiments are available at our project website: this https URL.

**arXiv ID:** 2603.17751
</details>

<details>
<summary><strong>Governed Capability Evolution for Embodied Agents: Safe Upgrade, Compatibility Checking, and Runtime Rollback for Embodied Capability Modules</strong> - Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li - [[pdf]](https://arxiv.org/pdf/2604.08059)</summary>

**Abstract:** Embodied agents are increasingly expected to improve over time by updating their executable capabilities rather than rewriting the agent itself. Prior work has separately studied modular capability packaging, capability evolution, and runtime governance. However, a key systems problem remains underexplored: once an embodied capability module evolves into a new version, how can the hosting system deploy it safely without breaking policy constraints, execution assumptions, or recovery guarantees?
We formulate governed capability evolution as a first-class systems problem for embodied agents. We propose a lifecycle-aware upgrade framework in which every new capability version is treated as a governed deployment candidate rather than an immediately executable replacement. The framework introduces four upgrade compatibility checks -- interface, policy, behavioral, and recovery -- and organizes them into a staged runtime pipeline comprising candidate validation, sandbox evaluation, shadow deployment, gated activation, online monitoring, and rollback.
We evaluate over 6 rounds of capability upgrade with 15 random seeds. Naive upgrade achieves 72.9% task success but drives unsafe activation to 60% by the final round; governed upgrade retains comparable success (67.4%) while maintaining zero unsafe activations across all rounds (Wilcoxon p=0.003). Shadow deployment reveals 40% of regressions invisible to sandbox evaluation alone, and rollback succeeds in 79.8% of post-activation drift scenarios.

**arXiv ID:** 2604.08059
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Do Agents Dream of Root Shells? Partial-Credit Evaluation of LLM Agents in Capture the Flag Challenges</strong> - Ali Al-Kaswan, Maksim Plotnikov, Maxim Hájek, Roland Vízner, Arie van Deursen, Maliheh Izadi - [[pdf]](https://arxiv.org/pdf/2604.19354)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly proposed for autonomous cybersecurity tasks, but their capabilities in realistic offensive settings remain poorly understood. We present DeepRed, an open-source benchmark for evaluating LLM-based agents on realistic Capture The Flag (CTF) challenges in isolated virtualized environments. DeepRed places an agent in a Kali attacker environment with terminal tools and optional web search, connected over a private network to a target challenge, and records full execution traces for analysis. To move beyond binary solved/unsolved outcomes, we introduce a partial-credit scoring method based on challenge-specific checkpoints derived from public writeups, together with an automated summarise-then-judge labelling pipeline for assigning checkpoint completion from logs. Using DeepRed, we benchmark ten commercially accessible LLMs on ten VM-based CTF challenges spanning different challenge categories. The results indicate that current agents remain limited: the best model achieves only 35% average checkpoint completion, performing strongest on common challenge types and weakest on tasks requiring non-standard discovery and longer-horizon adaptation.

**arXiv ID:** 2604.19354
</details>

<details>
<summary><strong>NeuroState-Bench: A Human-Calibrated Benchmark for Commitment Integrity in LLM Agent Profiles</strong> - Jia Xiao - [[pdf]](https://arxiv.org/pdf/2605.01847)</summary>

**Abstract:** Outcome-only evaluation under-specifies whether an evaluated agent profile preserves the commitments required to solve a multi-turn task coherently. NeuroState-Bench is a human-calibrated benchmark that operationalizes commitment integrity through benchmark-defined side-query probes rather than inferred hidden activations. The released inventory contains 144 deterministic tasks and 306 benchmark-defined side-query probes spanning eight cognitively motivated failure families, paired clean and distractor variants, and three difficulty bands. The main 32-profile evaluation contains a fixed 16-profile local subset and a matched 16-profile hosted large-model subset evaluated through the same benchmark pipeline. Human calibration uses the final merged reporting scope: 104 sampled task units, 216 raw annotations, and 108 adjudicated task rows, with weighted kappa = 0.977 and ICC(2,1) = 0.977. Empirically, task success and commitment integrity diverge across this expanded grid: the success leader is not the integrity leader, 31 of 32 profiles change rank when integrity replaces task success, and integrity rankings are more stable under distractor perturbation. The primary confidence-free score HCCIS-CORE reaches 0.8469 AUC and 0.6992 PR-AUC for post-probe diagnostic discrimination of terminal task failure; the legacy full heuristic variant HCCIS-FULL reaches 0.7997 AUC and 0.6410 PR-AUC. Probe accuracy and state drift achieve slightly higher ROC-AUC, 0.8587, and better Brier/ECE, while HCCIS-CORE has substantially higher point-estimate PR-AUC and remains more closely tied to the benchmark's intended construct. The exploratory neural-augmented variant HCCIS+N is weaker overall, and a randomized subspace control approaches chance. NeuroState-Bench therefore contributes a calibrated evaluation axis for exposing commitment failures over a broader model grid than the original local-only subset.

**arXiv ID:** 2605.01847
</details>

<details>
<summary><strong>Self-Induced Outcome Potential: Turn-Level Credit Assignment for Agents without Verifiers</strong> - Senkang Hu, Yong Dai, Xudong Han, Zhengru Fang, Yuzhi Zhao, Sam Tak Wu Kwong, Yuguang Fang - [[pdf]](https://arxiv.org/pdf/2605.04984)</summary>

**Abstract:** Long-horizon LLM agents depend on intermediate information-gathering turns, yet training feedback is usually observed only at the final answer, because process-level rewards require high-quality human annotation. Existing turn-level shaping methods reward turns that increase the likelihood of a gold answer, but they require answer supervision or stable task-specific verifiers. Conversely, label-free RL methods extract self-signals from output distributions, but mainly at the answer or trajectory level and therefore cannot assign credit to intermediate turns. We propose Self-Induced Outcome Potential (SIOP), which treats semantic clusters of final answers as latent future outcome states for potential-based turn-level credit assignment. For each query, SIOP samples multiple rollouts, clusters final answers into semantic outcome modes, and builds a reliability-aware target distribution over these states. It then rewards turns for increasing posterior support for reliable future states using a tractable cluster-level approximation. The objective generalizes information-potential shaping from gold-answer supervision to settings without task-specific gold verifiers while avoiding the broadcasted rollout-level advantages used by standard GRPO. We formalize the framework, characterize its supervised gold-answer limit, and show that SIOP improves average performance over verifier-free outcome-level baselines on seven search-augmented agentic reasoning benchmarks while approaching a gold-supervised outcome baseline. Code is available at this https URL.

**arXiv ID:** 2605.04984
</details>

<details>
<summary><strong>Tailoring Scaffolding to Diagnostic Strategies: Theory-Informed LLM-Based Agents</strong> - Fatma Betul Gures, Tanya Nazaretsky, Tanja Kaser - [[pdf]](https://arxiv.org/pdf/2605.04996)</summary>

**Abstract:** Learning analytics systems increasingly integrate large language models (LLMs) to provide adaptive scaffolding in complex learning environments, yet personalization is often driven by global instructional choices rather than principled alignment with learning theory, limiting effectiveness and pedagogical grounding. In prior work, we examined how structuring and problematizing scaffolding approaches can be instantiated through LLM agents in a scenario-based learning environment for diagnostic reasoning. While both approaches supported learning, we observed systematic differences in learner interaction patterns and clear tendencies indicating that different diagnostic strategies benefited from distinct forms of scaffolding. Building on these findings, we propose a theory-informed scaffolding design grounded in the Knowledge Learning Instruction (KLI) framework, as different diagnostic strategies target different types of knowledge and require different instructional mechanisms. We use KLI to guide the alignment between strategy demands and scaffolding approaches and introduce a KLI-informed hybrid LLM agent that adapts its pedagogical support according to the diagnostic strategy being practiced, rather than applying a single global scaffolding approach. We hypothesize that this design could enable better learning gains.

**arXiv ID:** 2605.04996
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (19 papers)</h2></summary>

<details>
<summary><strong>SensingAgents: A Multi-Agent Collaborative Framework for Robust IMU Activity Recognition</strong> - Naiyu Zheng, Tianlong Yu, Haochen Yin, Xiaoyi Fan, Xiping Hu, Zhimeng Yin - [[pdf]](https://arxiv.org/pdf/2605.04608)</summary>

**Abstract:** Human Activity Recognition (HAR) using Inertial Measurement Unit (IMU) sensors is a cornerstone of mobile health, smart environments, and human-computer interaction. However, current deep learning-based HAR models often struggle with heavy reliance on labeled data, position-specific ambiguity, and a lack of transparent reasoning. Inspired by the advanced agents framework, which emulates a collaborative agent using Large Language Models (LLMs), we propose SensingAgents, a novel multi-agent system for robust IMU activity recognition. SensingAgents organizes LLM-powered agents into specialized roles: a group of Analyst Agents for position-specific sensor analysis (arm, wrist, belt, pocket), a pair of Advocate Agents that resolves sensor conflicts through dynamic and static dialectical debates, and a Decision Agent that ensures reliability under sensor drift or failure. Evaluation on the Shoaib dataset demonstrates that SensingAgents significantly outperforms state-of-the-art single-agent and multi-agent LLM models, achieving an accuracy of 79.5% in a zero setting--29% higher than existing agent models and 9.4% higher than deep learning baselines--particularly in complex scenarios where multi-sensor data is conflicting or noisy. Our work highlights the potential of multi-agent collaborative reasoning for advancing the robustness and interpretability of ubiquitous sensing systems.

**arXiv ID:** 2605.04608
</details>

<details>
<summary><strong>When Context Hurts: The Crossover Effect of Knowledge Transfer on Multi-Agent Design Exploration</strong> - Saranyan Vigraham - [[pdf]](https://arxiv.org/pdf/2605.04361)</summary>

**Abstract:** The prevailing assumption in agent orchestration is that more context is better. We test this on multi-agent software design across 10 tasks, 7 context-injection conditions, and over 2,700 runs, and find a crossover effect: the same artifact type improves design exploration on some tasks (up to 20$\times$ tradeoff coverage) and actively degrades it on others (up to 46% reduction). On several tasks, an irrelevant document performs as well as or better than every relevant artifact. The direction is predicted by a single measurable variable--baseline exploration without context--with Pearson $r = -0.82$ ($p < 0.001$). Probing the mechanism by manipulating convergence pressure through prompt design reveals two distinct regimes: convergence driven by training data priors (natural) responds to artifact disruption, while convergence driven by explicit instructions (induced) does not. The implication is that context injection should be conditional, not universal: one no-context trial is a cheap diagnostic that predicts whether knowledge artifacts will help or hurt a given task.

**arXiv ID:** 2605.04361
</details>

<details>
<summary><strong>Agent Island: A Saturation- and Contamination-Resistant Benchmark from Multiagent Games</strong> - Connacher Murphy - [[pdf]](https://arxiv.org/pdf/2605.04312)</summary>

**Abstract:** Static capabilities benchmarks suffer from saturation and contamination, making it difficult to track capabilities progress over time. We introduce Agent Island, a multiplayer simulation environment in which language-model agents compete in a game of interagent cooperation, conflict, and persuasion. The environment yields a dynamic benchmark designed to mitigate both saturation and contamination; new models can always outperform the current leading player in this winner-take-all game, and agents compete against other adaptive agents rather than face a fixed task set. We rank players with a Bayesian Plackett-Luce model, allowing us to quantify uncertainty in player skill. In 999 games involving 49 unique models, openai/gpt-5.5 dominates its peers with a posterior mean skill of 5.64, compared with 3.10 for the second-ranked model, openai/gpt-5.2, and 2.86 for the third-ranked model, openai/gpt-5.3-codex. We release the game logs as a dataset for analyses of model behavior. As an example, we investigate same-provider preference in final-round votes and find that models are 8.3 p.p. more likely to support a same-provider finalist than finalists from other providers. This preference is not uniform across providers: among separately estimated providers, the effect is strongest for OpenAI models and weakest for Anthropic models.

**arXiv ID:** 2605.04312
</details>

<details>
<summary><strong>Strat-Reasoner: Reinforcing Strategic Reasoning of LLMs in Multi-Agent Games</strong> - Yidong He, Yutao Lai, Pengxu Yang, Jiarui Gan, Jiexin Wang, Yi Cai, Mengchen Zhao - [[pdf]](https://arxiv.org/pdf/2605.04906)</summary>

**Abstract:** While Large Language Models (LLMs) excel in certain reasoning tasks, they struggle in multi-agent games where the final outcome depends on the joint strategies of all agents. In multi-agent games, the non-stationarity of other agents brings significant challenges on the evaluation of the reasoning process and the credit assignment over multiple reasoning steps. Existing single-agent reinforcement learning (RL) approaches and their multi-agent extensions fail to address these challenges as they do not incorporate other agents in the reasoning process. In this work, we propose Strat-Reasoner, a novel RL-based framework that improves LLMs' strategic reasoning ability in multi-agent games. We introduce a novel recursive reasoning paradigm where an agent's reasoning also integrates other agents' reasoning processes. To provide effective reward signals for the intermediate reasoning sequences, we employ a centralized Chain-of-Thought (CoT) comparison module to evaluate the reasoning quality. Finally, we compute an accurate hybrid advantage and develop a group-relative RL approach to optimize the LLM policy. Experimental results show that Strat-Reasoner substantially improves strategic abilities of underlying LLMs, achieving 22.1\% average performance improvements across various multi-agent games.

**arXiv ID:** 2605.04906
</details>

<details>
<summary><strong>Uno-Orchestra: Parsimonious Agent Routing via Selective Delegation</strong> - Zhiqing Cui, Haotong Xie, Jiahao Yuan, Cheng Yang, Hanqing Wang, Yuxin Wu, Yifan Wu, Siru Zhong, Tao Yu, Yifu Guo, Siyu Zhang, Xinlei Yu, Qibing Ren, Usman Naseem - [[pdf]](https://arxiv.org/pdf/2605.05007)</summary>

**Abstract:** Large language model (LLM) multi-agent systems typically rely on rigid orchestration, committing either to flat per-query routing or to hand-engineered task decomposition, so decomposition depth, worker choice, and inference budget are not jointly optimized under one objective. We introduce Uno-Orchestra, a unified orchestration policy that selectively decomposes a task and dispatches each subtask to an admissible (model, primitive) pair, with both decisions learned together from curated RL trajectories grounded in real worker interactions. Against 22 baselines on a 13-benchmark suite spanning math, code, knowledge, long-context, and agentic tool-use, Uno-Orchestra reaches 77.0% macro pass@1, roughly 16% above the strongest workflow baseline, at roughly an order of magnitude lower per-query cost, advancing the accuracy-efficiency frontier of selective delegation.

**arXiv ID:** 2605.05007
</details>

<details>
<summary><strong>ARMATA: Auto-Regressive Multi-Agent Task Assignment</strong> - Yazan Youssef, Aboelmagd Noureldin, Sidney Givigi - [[pdf]](https://arxiv.org/pdf/2605.04225)</summary>

**Abstract:** Coordinating multi-agent systems over spatially distributed areas requires solving a complex hierarchical problem: first distributing areas among agents (allocation) and subsequently determining the optimal visitation order (routing). Existing methods typically decouple these stages ignoring inter-stage dependencies or rely on decentralized heuristics that lack global context. In this work, we propose a centralized, fully end-to-end auto-regressive framework that jointly generates allocation decisions and routing sequences. The core contribution of our approach is a multi-stage decoding mechanism that unifies high-level allocation and low-level routing in a single autoregressive pass while maintaining a centralized global state. This enables the model to implicitly balance workload distribution with routing efficiency, avoiding local optima common in decentralized methods. Extensive experiments demonstrate that our method significantly outperforms diverse baselines, achieving up to a 20\% improvement in solution quality over industrial solvers such as Google OR-Tools, IBM CPLEX, and LKH-3, while reducing computation time from hours to seconds.

**arXiv ID:** 2605.04225
</details>

<details>
<summary><strong>Evolving Idea Graphs with Learnable Edits-and-Commits for Multi-Agent Scientific Ideation</strong> - Jiangwen Dong, Bo Li, Wanyu Lin - [[pdf]](https://arxiv.org/pdf/2605.04922)</summary>

**Abstract:** LLM-empowered multi-agent systems offer new potential to accelerate scientific discovery by generating novel research ideas. However, existing methods typically coordinate agents through temporary texts, such as drafts or chat logs; it is difficult to pinpoint the weaknesses in the generated ideas and how the agents refine them. To this end, we introduce \textbf{Evolving Idea Graphs} (EIG), a graph-based multi-agent scientific ideation framework that can generate high-performance research ideas across various benchmark-native metrics, such as novelty, feasibility, and clarity. Instead of coordinating solely through texts, EIG represents a partially formed proposal as an evolving idea graph, where nodes capture scientific claims and edges encode relations (e.g., support and conflict), enabling unresolved weaknesses to remain identifiable throughout the idea evolving process. Specifically, a learned two-head controller operates over the evolving graph to guide the ideation: one head selects graph edits for agents to execute, while the other decides when the graph is ready for commit as final proposal synthesis. On AI Idea Bench 2025 and LiveIdeaBench, EIG outperforms all compared systems on both automatic benchmark scores and blind expert ratings. Ablations further show that explicit graph state provides the main performance gains, and learned edit-and-commit control adds consistent improvements.

**arXiv ID:** 2605.04922
</details>

<details>
<summary><strong>Modular Reinforcement Learning For Cooperative Swarms</strong> - Erel Shtossel, Gal A. Kaminka - [[pdf]](https://arxiv.org/pdf/2605.04939)</summary>

**Abstract:** A cooperative robot swarm is a collective of computationally-limited robots that share a common goal. Each robot can only interact with a small subset of its peers, without knowing how this affects the collective utility. Recent advances in distributed multi-agent reinforcement learning have demonstrated that it is possible for robots to learn how to interact effectively with others, in a manner that is aligned with the common goal, despite each robot learning independently of others. However, this requires each robot to represent a potentially combinatorial number of interaction states, challenging the memory capabilities of the robots. This paper proposes an alternative approach for representing spatial interaction states for multi-robot reinforcement learning in swarms. A modular (decomposed) representation is used, where each feature of the state is handled by a separate learning procedure, and the results aggregated. We demonstrate the efficacy of the approach in numerous experiments with simulated robot swarms carrying out foraging.

**arXiv ID:** 2605.04939
</details>

<details>
<summary><strong>Design Conductor 2.0: An agent builds a TurboQuant inference accelerator in 80 hours</strong> - Verkor Team, Ravi Krishna, Suresh Krishna, David Chin - [[pdf]](https://arxiv.org/pdf/2605.05170)</summary>

**Abstract:** Driven by a rapid co-evolution of both harness and underlying models, LLM agents are improving at a dizzying pace. In our prior work (performed in Dec. 2025), we introduced "Design Conductor" (or just "Conductor"), a system capable of building a 5-stage Linux-capable RISC-V CPU in 12 hours. In this work, we introduce an updated multi-agent harness powered by frontier models released in April 2026, which is able to handle 80x larger tasks, at higher quality, fully autonomously. Following a brief introduction, we examine 4 designs that the system produced autonomously, including "VerTQ", an LLM inference accelerator which hard-wires support for TurboQuant in a 240-cycle pipeline, starting from the TurboQuant arXiv paper. VerTQ includes heavy compute processing, with 5129 FP16/32 units; the design was mapped to an FPGA at 125 MHz and consumes 5.7 mm^2 in TSMC 16FF (8 attention pipes). We review the key new characteristics that enabled these results. Finally, we analyze Design Conductor's token usage and other empirical characteristics, including its limitations.

**arXiv ID:** 2605.05170
</details>

<details>
<summary><strong>GRAIL: A Deep-Granularity Hybrid Resonance Framework for Real-Time Agent Discovery via SLM-Enhanced Indexing</strong> - Jinliang Xu - [[pdf]](https://arxiv.org/pdf/2605.02489)</summary>

**Abstract:** As the ecosystem of Large Language Model (LLM)-based agents expands rapidly, efficient and accurate Agent Discovery becomes a critical bottleneck for large-scale multi-agent collaboration. Existing approaches typically face a dichotomy: either relying on heavy-weight LLMs for intent parsing, leading to prohibitive latency (often exceeding 30 seconds), or using monolithic vector retrieval that sacrifices semantic precision for speed. To bridge this gap, we propose \textbf{GRAIL} (Granular Resonance-based Agent/AI Link), a novel framework achieving sub-400ms discovery latency without compromising accuracy. GRAIL introduces three key innovations: (1) \textbf{SLM-Enhanced Prediction}, replacing the generalized LLM parser with a specialized, fine-tuned Small Language Model (SLM) for millisecond-level capability tag prediction; (2) \textbf{Pseudo-Document Expansion}, augmenting agent descriptions with synthetic queries to enhance semantic density for robust dense retrieval; and (3) \textbf{MaxSim Resonance}, a fine-grained matching mechanism computing maximum similarity between user queries and discrete agent usage examples, effectively mitigating semantic dilution. Validated on \textbf{AgentTaxo-9K}, our new large-scale dataset of 9,240 agents, GRAIL reduces end-to-end discovery latency by over \textbf{79$\times$} compared to LLM-parsing baselines, while significantly outperforming traditional vector search in Recall@10. This framework offers a scalable, industrial-grade solution for the real-time ``Internet of Agents."

**arXiv ID:** 2605.02489
</details>

<details>
<summary><strong>Governed Collaborative Memory as Artificial Selection in LLM-Based Multi-Agent Systems</strong> - Diego F. Cuadros, Abdoul-Aziz Maiga, Helen Meskhidze, Andre Curtis-Trudel - [[pdf]](https://arxiv.org/pdf/2605.04264)</summary>

**Abstract:** Persistent memory is turning language-model-based agents from stateless participants in isolated interactions into state-bearing components of LLM-based multi-agent systems. As memory becomes durable, reloadable, and behavior-shaping across agents, sessions, or versions, a design question arises that is not captured by retrieval accuracy or access control alone: which candidate memories should become shared institutional state? This Viewpoint frames that problem as governed collaborative memory. We argue that memory governance functions as a selection regime, determining which memory variants persist, which remain private, and which are rejected, abstained from, or superseded. We distinguish ungoverned persistence, constitutional or hybrid selection, automatic metric-based selection, and human-ratified artificial selection, emphasizing that these regimes are not a ranking but a design choice over target properties. We then describe a layered architecture that separates agent-local memory, shared institutional memory, archive memory, and project-continuity memory, with provenance and version lineage making selection inspectable. Documented traces from one running LLM-based multi-agent ecosystem illustrate unmanaged false-memory persistence, ratified institutional memory, rejection and revision, identity-preserving expansion, and governance-as-learning. The contribution is a design agenda: persistent LLM-based multi-agent systems should evaluate memory not only for recall and performance, but also for provenance fidelity, selection traceability, epistemic quality, correction pathways, and role preservation.

**arXiv ID:** 2605.04264
</details>

<details>
<summary><strong>Hierarachical Multiagent Reinforcement Learning for Multi-Group Tax Game</strong> - Honglei Guo, Yuhan Zhao, Yexin Li - [[pdf]](https://arxiv.org/pdf/2605.04741)</summary>

**Abstract:** Reinforcement learning has increasingly been used to study economic decision-making, such as taxation, public spending, and labour supply. However, most existing RL-based economic models focus on a single government--household group, thereby overlooking the strategic interactions that arise when multiple governments compete while managing their own populations. In practice, many economic systems (e.g., taxation) exhibit a multi-group structure, where each government must optimize its fiscal policy in response not only to household behaviour within its jurisdiction, but also to the policies of other competing governments. To capture this structure, we formulate taxation as a hierarchical multi-group game. Within each group, the interaction between the government and households is modelled as a leader--follower game; across groups, governments are modelled as players in a competitive game. This results in a hybrid hierarchical game that is difficult to solve using standard multi-agent reinforcement learning algorithms. We therefore propose a bi-level training framework built on multi-agent reinforcement learning, together with \textit{ Curriculum Learning} and a \textit{ Closed-Loop Sequential Update} strategy, to stabilize training and promote convergence. We instantiate this framework in a taxation game simulation environment grounded in classical economic models. The environment supports the evaluation of different taxation algorithms and provides multiple economic indicators for assessing policy performance. Experiments show that our approach can learn stable tax policies that benefit all participating groups. Compared with a two-group baseline without the proposed update mechanisms, our method avoids premature game collapse, extends the effective game duration by 60.92\%, produces more sustainable and robust tax policies, and reduces GDP disparities among governments by 44.12\%.

**arXiv ID:** 2605.04741
</details>

<details>
<summary><strong>Tree-based Credit Assignment for Multi-Agent Memory System</strong> - Marina Mao, Alexandr Liu, Pengbo Li, Siheng Li, Bo Zhou, Xiang Wang - [[pdf]](https://arxiv.org/pdf/2605.04811)</summary>

**Abstract:** Memory systems are widely adopted to enhance LLMs for long-horizon tasks, and are commonly organized as multi-agent pipelines with memory building, summarizing, and retrieval agents. To empower this system, existing RL-based methods either apply final downstream task rewards (e.g., QA accuracy) for all agents uniformly, which are coarse and ambiguous, or design task-specific rewards for agents on different subtasks, which require costly annotations (e.g., key evidence) and are difficult to define reliably. To address these limitations, we propose Tree-based Credit Assignment for Multi-Agent Memory Systems (TreeMem), which derives agent-specific credit from the final reward without task-specific annotations. Specifically, TreeMem extends the multi-agent pipeline (builder--summarizer--retrieval) into a tree structure, where each agent's outputs are expanded into multiple subsequent branches. The contribution of each agent is estimated via Monte Carlo averaging over its subsequent branches, capturing how intermediate agent actions may influence the final reward. This converts the coarse final reward into agent-specific optimization signals. These signals are then used to update all agent policies simultaneously, helping heterogeneous agents specialize effectively. Experiments on long-horizon benchmarks show that TreeMem improves memory system performance over strong baselines, validating the effectiveness of tree-structured credit assignment for the multi-agent memory system.

**arXiv ID:** 2605.04811
</details>

<details>
<summary><strong>Graph-SND: Sparse Aggregation for Behavioral Diversity in Multi-Agent Reinforcement Learning</strong> - Shawn Ray - [[pdf]](https://arxiv.org/pdf/2605.05020)</summary>

**Abstract:** System Neural Diversity (SND) measures behavioral heterogeneity in multi-agent reinforcement learning by averaging pairwise distances over all $\binom{n}{2}$ agent pairs, making each call quadratic in team size. We introduce Graph-SND, which replaces this complete-graph average with a weighted average over the edges of an arbitrary graph $G$. Three regimes follow: $G=K_n$ recovers SND exactly; a fixed sparse $G$ defines a localized diversity measure at $O(|E|)$ cost; and random edge samples yield an unbiased Horvitz-Thompson estimator and a normalized sample mean with $O(1/\sqrt{m})$ concentration in the sampled edge count $m$. For fixed sparse graphs we prove forwarding-index distortion bounds for expanders and a spectral refinement under low-rank distance structure; for random $d$-regular graphs we prove an unconditional probabilistic $\widetilde{\mathcal{O}}(D_{\max}/\sqrt{n})$ bound. On VMAS we verify recovery, unbiasedness, concentration, and wall-clock scaling, with a PettingZoo TVD panel checking non-Gaussian transfer. In a 500-iteration $n=100$ PPO run, Bernoulli-$0.1$ Graph-SND tracks full SND while reducing per-call metric time by about $10\times$, and frozen-policy GPU timing up to $n=500$ follows the predicted $\binom{n}{2}/|E|$ speedup. Random $d$-regular expanders empirically achieve $\mathrm{SND}_{G}^{\mathrm{u}}/\mathrm{SND} \in [0.9987, 1.0013]$ at $\Theta(n \log n)$ edges. In DiCo diversity control at $n=50$, Bernoulli-$0.1$ Graph-SND preserves set-point tracking with paired reward differences indistinguishable from zero across nine matched cells while cutting per-call metric cost by ${\sim}9.5\times$. Together, these results show that the SND aggregation bottleneck can be removed without changing the metric's semantics, yielding a drop-in sparse alternative that scales beyond complete-graph SND and supports both passive measurement and closed-loop diversity control.

**arXiv ID:** 2605.05020
</details>

<details>
<summary><strong>When Stress Becomes Signal: Detecting Antifragility-Compatible Regimes in Multi-Agent LLM Systems</strong> - Jose Manuel de la Chica, Juan Manuel Vera, Jairo Rodríguez - [[pdf]](https://arxiv.org/pdf/2605.02463)</summary>

**Abstract:** Multi-agent LLM systems are increasingly used to solve complex tasks through decomposition, debate, specialization, and ensemble reasoning. However, these systems are usually evaluated in terms of robustness: whether performance is preserved under perturbation. This paper studies a different question: whether semantic stress exposes structured variation that could support future antifragile learning. We introduce CAFE (Cognitive Antifragility Framework for Evaluation), a statistical framework for detecting antifragility-compatible regimes in multi-agent architectures. CAFE models a controlled expected distribution of semantic stressors, reconstructs an architecture-specific observed effective stress distribution from multi-dimensional judge signals, and compares both distributions using a distributional Jensen Gap under a convex stress potential. A positive gap does not imply immediate performance improvement; instead, it indicates a convex-expansive deformation of the observed stress distribution, suggesting that the architecture exposes learnable stress structure. We evaluate CAFE on a banking-risk analysis benchmark with five multi-agent architectures: flat, hierarchical, debate, meta-adaptive, and ensemble. Across all architectures, semantic stress reduces average judged quality by roughly one third. Yet all architectures exhibit positive distributional Jensen Gaps with bootstrap confidence intervals above zero. These results show that immediate quality degradation can coexist with statistically detectable antifragility-compatible stress geometry. CAFE is therefore not an antifragile learner itself, but a measurement layer for identifying when and where antifragility learning may be worth applying.

**arXiv ID:** 2605.02463
</details>

<details>
<summary><strong>When Reasoning Models Hurt Behavioral Simulation: A Solver-Sampler Mismatch in Multi-Agent LLM Negotiation</strong> - Sandro Andric - [[pdf]](https://arxiv.org/pdf/2604.11840)</summary>

**Abstract:** Behavioral simulation and strategic problem solving are different tasks. Large language models are increasingly explored as agents in policy-facing institutional simulations, but stronger reasoning need not improve behavioral sampling. We study this solver-sampler mismatch in three multi-agent negotiation environments: two trading-limits scenarios with different authority structures and a grid-curtailment case in emergency electricity management. Across two primary model families, native reasoning and often no reflection collapse toward authority-heavy outcomes. The sharpest case is DeepSeek native reasoning in the grid-curtailment transfer: it reaches action entropy 1.256 and a concession-arc rate of 0.933, yet still ends in authority decision in 15 of 15 runs. A direct OpenAI extension shows the same pressure at provider breadth: GPT-5.2 native reasoning ends in authority decisions in 45 of 45 runs across the three environments. Budget-matched no-reflection controls and orthogonal private-state controls remain rigid, while the negotiation-structured scaffold condition is the only condition that consistently opens negotiated outcomes. These diagnostics are failure screens within a fixed negotiation grammar, not evidence of external behavioral realism or policy-forecasting validity. The results show that neither more output space nor generic extra private state rescues solver-like sampler failure. For institutional simulation, solver strength and sampler qualification are different objectives: models should be evaluated for the behavioral role they are meant to play, not only for strategic capability.

**arXiv ID:** 2604.11840
</details>

<details>
<summary><strong>Material Database Agent: A Multimodal Agentic Framework for Scientific Literature Mining</strong> - Achuth Chandrasekhar, Omid Barati Farimani, Radheesh Sharma Meda, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2605.04278)</summary>

**Abstract:** Materials science workflows rely on structured and unstructured data from the vast body of available scientific literature. However, most of the experimental details remain buried in text, tables, graphs and figures. Thus, constructing databases that incorporate this data is a manual, time-consuming, and hard-to-scale process. Multimodal large language models have made it feasible to extract information from text and scientific figures with high speed and accuracy. This opens the possibility of an AI system that can create production-scale material databases. Material Database Agent (MDA) is a modular, multi-agent system architecture for converting research literature into structured databases. MDA accepts article PDFs as input, which are subsequently processed in parallel into markdown files and figures. Multiple sub-agents read these markdown files and figures in parallel to assemble sub-databases for each paper. These sub-databases are then compiled into a single tabular database by an agent. As opposed to using either a rule-based approach or a single-pass pipeline for extracting information, MDA is a specialized architecture for transforming the literature into a database in the field of materials science. More generally, this study provides a basis for positioning multimodal agentic information extraction as a viable means for constructing next-generation scientific databases from the primary literature.

**arXiv ID:** 2605.04278
</details>

<details>
<summary><strong>Code Broker: A Multi-Agent System for Automated Code Quality Assessment</strong> - Samer Attrah - [[pdf]](https://arxiv.org/pdf/2604.23088)</summary>

**Abstract:** We present Code Broker, a multi agent system built on Google s Agent Development Kit ADK that analyses Python source code from individual files, local directory trees, or remote GitHub repositories and generates structured, actionable quality assessment reports. The system realises a hierarchical five agent architecture in which a root orchestrator coordinates a sequential pipeline agent that, in turn, dispatches three specialised agents concurrently a Correctness Assessor, a Style Assessor, and a Description Generator before synthesising their findings through an Improvement Recommender. Reports quantify four quality dimensions correctness, security, style, and maintainability on a normalised scale and are rendered in both Markdown and HTML for integration into diverse developer workflows. Code Broker fuses LLM based semantic reasoning with deterministic static analysis signals from Pylint, employs asynchronous execution with exponential backoff retry logic to improve robustness under transient API failures, and explores lightweight session memory for retaining and querying prior assessment context across runs. We frame this paper as a technical report on system design, prompt engineering, and tool orchestration, and present a preliminary qualitative evaluation on representative Python codebases of varying scale. The results indicate that parallel specialised agents produce readable, developer oriented feedback that complements traditional linting, while also foregrounding current limitations in evaluation depth, security tooling, large repository handling, and the exclusive reliance on in memory persistence. All code and reproducibility materials are publicly available: this https URL.

**arXiv ID:** 2604.23088
</details>

<details>
<summary><strong>Scalable Multi Agent Diffusion Policies for Coverage Control</strong> - Frederic Vatnsdal, Romina Garcia Camargo, Saurav Agarwal, Alejandro Ribeiro - [[pdf]](https://arxiv.org/pdf/2509.17244)</summary>

**Abstract:** We propose MADP, a novel diffusion-model-based approach for collaboration in decentralized robot swarms. MADP leverages diffusion models to generate samples from complex and high-dimensional action distributions that capture the interdependencies between agents' actions. Each robot conditions policy sampling on a fused representation of its own observations and perceptual embeddings received from peers. To evaluate this approach, we task a team of holonomic robots piloted by MADP to address coverage control-a canonical multi agent navigation problem. The policy is trained via imitation learning from a clairvoyant expert on the coverage control problem, with the diffusion process parameterized by a spatial transformer architecture to enable decentralized inference. We evaluate the system under varying numbers, locations, and variances of importance density functions, capturing the robustness demands of real-world coverage tasks. Experiments demonstrate that our model inherits valuable properties from diffusion models, generalizing across agent densities and environments, and consistently outperforming state-of-the-art baselines.

**arXiv ID:** 2509.17244
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>Position: Embodied AI Requires a Privacy-Utility Trade-off</strong> - Xiaoliang Fan, Jiarui Chen, Zhuodong Liu, Ziqi Yang, Peixuan Xu, Ruimin Shen, Junhui Liu, Jianzhong Qi, Cheng Wang - [[pdf]](https://arxiv.org/pdf/2605.05017)</summary>

**Abstract:** Embodied AI (EAI) systems are rapidly transitioning from simulations into real-world domestic and other sensitive environments. However, recent EAI solutions have largely demonstrated advancements within isolated stages such as instruction, perception, planning and interaction, without considering their coupled privacy implications in high-frequency deployments where privacy leakage is often irreversible. This position paper argues that optimizing these components independently creates a systemic privacy crisis when deployed in sensitive settings, thereby advancing the position that privacy in EAI is a life cycle-level architectural constraint rather than a stage-local feature. To address these challenges, we propose Secure Privacy Integration in Next-generation Embodied AI (SPINE), a unified privacy-aware framework that treats privacy as a dynamic control signal governing cross-stage coupling throughout the entire EAI life cycle. SPINE decomposes the EAI pipeline into various stages and establishes a multi-criterion privacy classification matrix to orchestrate contextual sensitivity across stage boundaries. We conduct preliminary simulation and real-world case studies to conceptually validate how privacy constraints propagate downstream to reshape system behavior, illustrating the insufficiency of fragmented privacy patches and motivating future research directions into secure yet functional embodied AI systems. We detail the SPINE framework and case studies at this https URL.

**arXiv ID:** 2605.05017
</details>

<details>
<summary><strong>Executable World Models for ARC-AGI-3 in the Era of Coding Agents</strong> - Sergey Rodionov - [[pdf]](https://arxiv.org/pdf/2605.05138)</summary>

**Abstract:** We evaluate an initial coding-agent system for ARC-AGI-3 in which the agent maintains an executable Python world model, verifies it against previous observations, refactors it toward simpler abstractions as a practical proxy for an MDL-like simplicity bias, and plans through the model before acting. The system is intentionally direct: it uses a scripted controller, predefined world-model interfaces, verifier programs, and a plan executor, but no hand-coded game-specific logic. We report results on the 25 public ARC-AGI-3 games. Each recorded playthrough uses a fresh agent instance with no access to previous playthrough-specific files or conversation state. Most games have a single recorded playthrough; for a few games, we report multiple independent fresh-agent playthroughs to expose run-to-run variability. The agent fully solved 7 games, achieved a Relative Human Action Efficiency greater than 75%, on 6 games, and obtained a mean per-game RHAE of 32.58%. Because the system uses no game-specific code, it can serve as a game-general baseline for ARC-AGI-3. Performance on the private validation set remains to be tested. Overall, the results provide preliminary evidence that verifier-driven executable world models are a promising approach for ARC-AGI-3 agents.

**arXiv ID:** 2605.05138
</details>

<details>
<summary><strong>GEM: Graph-Enhanced Mixture-of-Experts with ReAct Agents for Dialogue State Tracking</strong> - Ziqi Zhu, Adithya Suresh, Tomal Deb, Iman Abbasnejad - [[pdf]](https://arxiv.org/pdf/2605.04449)</summary>

**Abstract:** Dialogue State Tracking (DST) requires precise extraction of structured information from multi-domain conversations, a task where Large Language Models (LLMs) struggle despite their impressive general capabilities. We present GEM (Graph-Enhanced Mixture-of-Experts), a novel framework that combines language models and graph-structured dialogue understanding with ReAct agent-based reasoning for superior DST performance. Our approach dynamically routes between specialized experts: a Graph Neural Network that captures dialogue structure and turn-level dependencies, and a finetuned T5-Small encoder-decoder for sequence modeling, coordinated by an intelligent router. For complex value generation tasks, we integrate ReAct agents that perform structured reasoning over dialogue context. On MultiWOZ 2.2, GEM achieves 65.19% Joint Goal Accuracy, substantially outperforming end-to-end LLM approaches (best: 38.43%) and surpassing state-of-the-art (SOTA) methods including TOATOD (63.79%), D3ST (58.70%), and Diable (56.48%). Our graph-enhanced mixture-of-experts architecture with ReAct integration demonstrates that combining structured dialogue representation with dynamic expert routing and agent-based reasoning provides a powerful paradigm for dialogue state tracking, achieving superior accuracy while maintaining computational efficiency through selective expert activation.

**arXiv ID:** 2605.04449
</details>

<details>
<summary><strong>Storage Is Not Memory: A Retrieval-Centered Architecture for Agent Recall</strong> - Joshua Adler, Guy Zehavi - [[pdf]](https://arxiv.org/pdf/2605.04897)</summary>

**Abstract:** Extraction at ingestion is the wrong primitive for agent memory: content discarded before the query is known cannot be recovered at retrieval time. We propose True Memory, a six-layer architecture that shifts the center of the system from a storage schema to a multi-stage retrieval pipeline operating over events preserved verbatim. The full system runs as a single SQLite file on commodity CPU with no external database, vector index, graph store, or GPU. On LoCoMo (1,540 questions across 10 multi-session conversations), True Memory Pro reaches 93.0% accuracy (3-run mean) against 61.4% for Mem0, 65.4% for Supermemory, approximately 71% for Zep, and 94.5% for EverMemOS under a matched gpt-4.1-mini answer model. On LongMemEval (500 questions), True Memory Pro reaches 87.8% (3-run mean). On BEAM-1M (700 questions at the 1-million-token scale), True Memory Pro reaches 76.6% (3-run mean), above the prior published result of 73.9% for Hindsight. A 56-configuration ablation shows a 1.3-percentage-point spread within the top-performing configuration family.

**arXiv ID:** 2605.04897
</details>

<details>
<summary><strong>Hierarchical Visual Agent: Managing Contexts in Joint Image-Text Space for Advanced Chart Reasoning</strong> - Qihua Dong, Ruozhen He, Junwen Chen, Yizhou Wang, Xu Ma, Songyao Jiang, Yun Fu - [[pdf]](https://arxiv.org/pdf/2605.04304)</summary>

**Abstract:** Advanced chart question answering requires both precise perception of small visual elements and multi-step reasoning across several subplots. While existing MLLMs are strong at understanding single plots, they often struggle with multi-step reasoning across multiple subplots. We propose HierVA, a hierarchical visual agent framework for chart reasoning that iteratively constructs and updates a working context in a joint image--text space. A high-level manager generates plans and maintains a compact context containing only key information, while specialized workers perform reasoning, gather evidence, and return results. In particular, the agent maintains separate visual and textual contexts, using a zoom-in tool to restrict the visual context. Experiments on the CharXiv reasoning subset demonstrate consistent improvements over strong multimodal baselines, and ablation studies verify that hierarchical architecture, scoped visual context, and distilled context contribute complementary gains.

**arXiv ID:** 2605.04304
</details>

<details>
<summary><strong>Adaptivity Under Realizability Constraints: Comparing In-Context and Agentic Learning</strong> - Anastasis Kratsios, A. Martina Neuman, Philipp Petersen - [[pdf]](https://arxiv.org/pdf/2605.04995)</summary>

**Abstract:** We compare in-context learning with fixed queries and agentic learning with adaptive queries for uniform approximation of task families. We consider two settings: an unrestricted regime, where querying and approximation are arbitrary functions, and a realizable regime, where we require these operations to be implemented by ReLU neural networks. In both settings, adaptivity never hinders approximation performance. However, this advantage can change when one passes from the unrestricted regime to the realizable regime. We identify four distinct approximation scenarios, each witnessed by an explicit task family: (a) no advantage of adaptivity; (b) an advantage in the unrestricted regime that persists under ReLU realizability; (c) an advantage that arises only under realizability; and (d) an advantage that disappears under realizability. This demonstrates that representational constraints interact profoundly with the effect of adaptivity.

**arXiv ID:** 2605.04995
</details>

<details>
<summary><strong>AI-Aided Advancements in Autonomous Underwater Vehicle Navigation</strong> - Guy Damari, Zeev Yampolsky, Nadav Cohen, Arup Kumar Sahoo, Jeryes Danial, Felipe O. Silva, Itzik Klein - [[pdf]](https://arxiv.org/pdf/2605.04672)</summary>

**Abstract:** Autonomous underwater vehicles (AUVs) have become indispensable for deep-sea exploration, spanning critical scientific research and commercial applications. The rapid attenuation of electromagnetic waves renders satellite radio signals unavailable, while the dynamic unpredictability of the marine environment presents formidable navigation challenges. This chapter explores recent advancements in AI-aided AUV positioning, specifically focusing on advanced sensor fusion architectures that integrate inertial navigation systems with Doppler velocity logs and cameras. Beyond traditional model-based filtering, we examine the transformative emergence of AI-driven learning approaches in enhancing inertial dead-reckoning tasks and adaptive fusion algorithms. By addressing these recent milestones, this chapter provides a comprehensive roadmap for achieving the high-precision navigation essential for autonomous underwater missions.

**arXiv ID:** 2605.04672
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (28 papers)</h2></summary>

<details>
<summary><strong>AuditRepairBench: A Paired-Execution Trace Corpus for Evaluator-Channel Ranking Instability in Agent Repair</strong> - Yuelin Hu, Zhenbo Yu, Zhengxue Cheng, Wei Liu, Li Song - [[pdf]](https://arxiv.org/pdf/2605.04624)</summary>

**Abstract:** Agent-repair leaderboards reorder under evaluator reconfiguration, and a measurable share of the reordering is produced by methods that consult evaluator-derived signal during internal selection of candidate repairs. We document this failure mode on a public leaderboard and release AuditRepairBench, a paired-execution trace corpus of 576,000 registered cells (96,000 executed) that operationalizes evaluator-channel-blocking ranking instability within a declared observability boundary. A modular screening architecture decides pathway-blocking through four interchangeable implementations, a learned influence proxy, a rule-based channel-exposure ratio that uses no trained model, a counterfactual sensitivity proxy, and a sparse human-audit proxy, combined into a screening posterior that feeds a cell-level flip functional, a set-valued label, a stratified system score, and a set-valued leaderboard. The resource is supported by mechanism-anchored validation on an 80-case source-level channel-surgery subset, an independent-discovery protocol under which two annotator groups separated from the pipeline developers discover coupling patterns blinded to the screening design and the frozen ensemble attains pooled AUROC 0.83 on their 79 cases, implementation robustness, uncertainty propagation that raises 95% coverage from 0.81 to 0.95, and forward transfer with pooled community-evaluator Spearman \r{ho} = 0.65. Screening-guided blinding patches reduce rank displacement by 55--74% (mean 62%) at fewer than 50 lines of code, whereas random channel blinding produces at most 7% reduction and generic retraining at most 13%. AuditRepairBench-Lite, a rule-only configuration on a 12,000-cell subset, preserves the leaderboard at Kendall {\tau} = 0.88 under twenty-four GPU-hours and is the primary release artifact at 42 GB.

**arXiv ID:** 2605.04624
</details>

<details>
<summary><strong>Reward-Decomposed Reinforcement Learning for Immersive Video Role-Playing</strong> - Miao Wang, Yuling Shi, Yijiang Li, Yeheng Chen, Xiaodong Gu, Bin Li, Bo Gao, Yaduan Ruan - [[pdf]](https://arxiv.org/pdf/2605.04733)</summary>

**Abstract:** Text-based role-playing models can imitate character styles, yet they often fail to reflect a scene's atmosphere and evolving tension, both essential for immersive applications such as Virtual Reality (VR) games and interactive narratives. We study video-grounded role-playing dialogue and introduce EBM-RL (Eye-Brain-Mouth Reinforcement Learning), a decoupled GRPO-based framework that explicitly separates observation ([perception]), reasoning ([think]), and utterance ([answer]). This structure promotes human-like sensory grounding by compelling the model to first attend to visual cues, then form internal interpretations, and finally generate context-appropriate dialogue.
EBM-RL integrates four complementary rewards: (i) CLIP-based scene-text alignment to improve ambiance and emotion; (ii) a Perceptual-Cognitive reward that encourages [perception] and [think] processes that increase the likelihood of the reference response; (iii) answer accuracy to ensure faithfulness; and (iv) a dense format reward to enforce the desired structured output.
Extensive experiments demonstrate that EBM-RL substantially outperforms text-only role-playing baselines and larger-scale vision-language models on our immersive role-playing benchmark, delivering simultaneous gains in visual-atmosphere consistency and character authenticity. Beyond the role-playing domain, EBM-RL also exhibits strong zero-shot generalization: without any additional fine-tuning, it consistently improves performance on out-of-domain VideoQA benchmarks. We additionally release an open-source dataset for video-grounded role-playing dialogue.

**arXiv ID:** 2605.04733
</details>

<details>
<summary><strong>DecodingTrust-Agent Platform (DTap): A Controllable and Interactive Red-Teaming Platform for AI Agents</strong> - Zhaorun Chen, Xun Liu, Haibo Tong, Chengquan Guo, Yuzhou Nie, Jiawei Zhang, Mintong Kang, Chejian Xu, Qichang Liu, Xiaogeng Liu, Tianneng Shi, Chaowei Xiao, Sanmi Koyejo, Percy Liang, Wenbo Guo, Dawn Song, Bo Li - [[pdf]](https://arxiv.org/pdf/2605.04808)</summary>

**Abstract:** AI agents are increasingly deployed across diverse domains to automate complex workflows through long-horizon and high-stakes action executions. Due to their high capability and flexibility, such agents raise significant security and safety concerns. A growing number of real-world incidents have shown that adversaries can easily manipulate agents into performing harmful actions, such as leaking API keys, deleting user data, or initiating unauthorized transactions. Evaluating agent security is inherently challenging, as agents operate in dynamic, untrusted environments involving external tools, heterogeneous data sources, and frequent user interactions. However, realistic, controllable, and reproducible environments for large-scale risk assessment remain largely underexplored. To address this gap, we introduce the DecodingTrust-Agent Platform (DTap), the first controllable and interactive red-teaming platform for AI agents, spanning 14 real-world domains and over 50 simulation environments that replicate widely used systems such as Google Workspace, Paypal, and Slack. To scale the risk assessment of agents in DTap, we further propose DTap-Red, the first autonomous red-teaming agent that systematically explores diverse injection vectors (e.g., prompt, tool, skill, environment, combinations) and autonomously discovers effective attack strategies tailored to varying malicious goals. Using DTap-Red, we curate DTap-Bench, a large-scale red-teaming dataset comprising high-quality instances across domains, each paired with a verifiable judge to automatically validate attack outcomes. Through DTap, we conduct large-scale evaluations of popular AI agents built on various backbone models, spanning security policies, risk categories, and attack strategies, revealing systematic vulnerability patterns and providing valuable insights for developing secure next-generation agents.

**arXiv ID:** 2605.04808
</details>

<details>
<summary><strong>Designing a double deep reinforcement learning selection tool for resilient demand prediction</strong> - Bilel Abderrahmane Benziane, Benoit Lardeux, Ayoub Mcharek, Maher Jridi - [[pdf]](https://arxiv.org/pdf/2605.04068)</summary>

**Abstract:** The use of artificial intelligence in supply chain forecasting has attracted many scientific studies for several decades. However, the process of selecting an appropriate forecasting solution becomes a daunting task. This complexity arises due to the distinct features inherent to each dataset. Research to tackle this issue has been performed since the eighties but recent development of demand forecasting has opened new perspectives. This research aims to enhance automatic forecasting model selection by proposing a novel architecture that acts as a double deep reinforcement learning agent, selecting automatically a forecasting model from the forecasting committee at the time of prediction. Moreover, a novel early-stopping approach based on average reward convergence has been introduced to expedite training time. To evaluate the model's performance, an empirical study was conducted utilizing grocery sales datasets and snack demands datasets. The experimental results demonstrate the robustness of the proposed approach when compared to state-of-the-art methods.

**arXiv ID:** 2605.04068
</details>

<details>
<summary><strong>A Harmonic Mean Formulation of Average Reward Reinforcement Learning in SMDPs</strong> - Erel Shtossel, Alicia Vidler, Uri Shaham, Gal A. Kaminka - [[pdf]](https://arxiv.org/pdf/2605.04880)</summary>

**Abstract:** Recent research has revived and amplified interest in algorithms for undiscounted average reward reinforcement learning in infinite-horizon, non-episodic (continuing) tasks. Semi-Markov decision processes (SMDPs) are of particular interest. In SMDPs, discrete actions stochastically generate both rewards and durations, and the objective is to optimize the average reward rate. Existing algorithms approach this by optimizing the ratio of rewards to durations. However, when rewards and durations are non-stationary (in the infinite horizon), this can be incorrect. This paper presents a novel modified harmonic mean operator that correctly computes reward rates even under such conditions. This yields model-free learning algorithms that can work with SMDPs, while maintaining robustness to non-stationary reward and duration distributions over time. We prove theoretical properties of the modified harmonic mean operator, and empirically demonstrate its efficacy in comparison to existing algorithms.

**arXiv ID:** 2605.04880
</details>

<details>
<summary><strong>LineRides: Line-Guided Reinforcement Learning for Bicycle Robot Stunts</strong> - Seungeun Rho, Shamel Fahmi, Jeonghwan Kim, Arianna Ilvonen, Sehoon Ha, Gabriel Nelson - [[pdf]](https://arxiv.org/pdf/2605.05110)</summary>

**Abstract:** Designing reward functions for agile robotic maneuvers in reinforcement learning remains difficult, and demonstration-based approaches often require reference motions that are unavailable for novel platforms or extreme stunts. We present LineRides, a line-guided learning framework that enables a custom bicycle robot to acquire diverse, commandable stunt behaviors from a user-provided spatial guideline and sparse key-orientations, without demonstrations or explicit timing. LineRides handles physically infeasible guidelines using a tracking margin that permits controlled deviation, resolves temporal ambiguity by measuring progress via traveled distance along the guideline, and disambiguates motion details through position- and sequence-based key-orientations. We evaluate LineRides on the Ultra Mobility Vehicle (UMV) and show that the policy trained with our methods supports seamless transitions between normal driving and stunt execution, enabling five distinct stunts on command: MiniHop, LargeHop, ThreePointTurn, Backflip, and DriftTurn.

**arXiv ID:** 2605.05110
</details>

<details>
<summary><strong>Adaptive Policy Selection and Fine-Tuning under Interaction Budgets for Offline-to-Online Reinforcement Learning</strong> - Alper Kamil Bozkurt, Xiaoan Xu, Shangtong Zhang, Miroslav Pajic, Yuichi Motai - [[pdf]](https://arxiv.org/pdf/2605.05123)</summary>

**Abstract:** In offline-to-online reinforcement learning (O2O-RL), policies are first safely trained offline using previously collected datasets and then further fine-tuned for tasks via limited online interactions. In a typical O2O-RL pipeline, candidate policies trained with offline RL are evaluated via either off-policy evaluation (OPE) or online evaluation (OE). The policy with the highest estimated value is then deployed and continually fine-tuned. However, this setup has two main issues. First, OPE can be unreliable, making it risky to deploy a policy based solely on those estimates, whereas OE may identify a viable policy with substantial online interaction, which could have been used for fine-tuning. Second--and more importantly--it is also often not possible to determine a priori whether a pretrained policy will improve with post-deployment fine-tuning, especially in non-stationary environments. As a result, procedures committing to a single deployed policy are impractical in many real-world settings. Moreover, a naive remedy that exhaustively fine-tunes all candidates would violate interaction budget constraints and is likewise infeasible. In this paper, we propose a novel adaptive approach for policy selection and fine-tuning under online interaction budgets in O2O-RL. Following the standard pipeline, we first train a set of candidate policies with different offline RL algorithms and hyperparameters; we then perform OPE to obtain initial performance estimates. We next adaptively select and fine-tune the policies based on their predicted performance via an upper-confidence-bound approach thereby making efficient use of online interactions. We demonstrate that our approach improves upon O2O-RL baselines with various benchmarks.

**arXiv ID:** 2605.05123
</details>

<details>
<summary><strong>When Life Gives You BC, Make Q-functions: Extracting Q-values from Behavior Cloning for On-Robot Reinforcement Learning</strong> - Lakshita Dodeja, Ondrej Biza, Shivam Vats, Stephen Hart, Stefanie Tellex, Robin Walters, Karl Schmeckpeper, Thomas Weng - [[pdf]](https://arxiv.org/pdf/2605.05172)</summary>

**Abstract:** Behavior Cloning (BC) has emerged as a highly effective paradigm for robot learning. However, BC lacks a self-guided mechanism for online improvement after demonstrations have been collected. Existing offline-to-online learning methods often cause policies to replace previously learned good actions due to a distribution mismatch between offline data and online learning. In this work, we propose Q2RL, Q-Estimation and Q-Gating from BC for Reinforcement Learning, an algorithm for efficient offline-to-online learning. Our method consists of two parts: (1) Q-Estimation extracts a Q-function from a BC policy using a few interaction steps with the environment, followed by online RL with (2) Q-Gating, which switches between BC and RL policy actions based on their respective Q-values to collect samples for RL policy training. Across manipulation tasks from D4RL and robomimic benchmarks, Q2RL outperforms SOTA offline-to-online learning baselines on success rate and time to convergence. Q2RL is efficient enough to be applied in an on-robot RL setting, learning robust policies for contact-rich and high precision manipulation tasks such as pipe assembly and kitting, in 1-2 hours of online interaction, achieving success rates of up to 100% and up to 3.75x improvement against the original BC policy. Code and video are available at this https URL

**arXiv ID:** 2605.05172
</details>

<details>
<summary><strong>Meta-Learning and Meta-Reinforcement Learning -- Tracing the Path towards DeepMind's Adaptive Agent</strong> - Björn Hoppmann, Christoph Scholz - [[pdf]](https://arxiv.org/pdf/2602.19837)</summary>

**Abstract:** Humans are highly effective at utilizing prior knowledge to adapt to novel tasks, a capability that standard machine learning models struggle to replicate due to their reliance on task-specific training. Meta-learning overcomes this limitation by allowing models to acquire transferable knowledge from various tasks, enabling rapid adaptation to new challenges with minimal data. This survey provides a rigorous, task-based formalization of meta-learning and meta-reinforcement learning and uses that paradigm to chronicle the landmark algorithms that paved the way for DeepMind's Adaptive Agent, consolidating the essential concepts needed to understand the Adaptive Agent and other generalist approaches.

**arXiv ID:** 2602.19837
</details>

<details>
<summary><strong>WaferSAGE: Large Language Model-Powered Wafer Defect Analysis via Synthetic Data Generation and Rubric-Guided Reinforcement Learning</strong> - Ke Xu - [[pdf]](https://arxiv.org/pdf/2604.27629)</summary>

**Abstract:** We present WaferSAGE, a framework for wafer defect visual question answering using small vision-language models. To address data scarcity in semiconductor manufacturing, we propose a three-stage synthesis pipeline incorporating structured rubric generation for precise evaluation. Starting from limited labeled wafer maps, we employ clustering-based cleaning to filter label noise, then generate comprehensive defect descriptions using vision-language models, which are converted into structured evaluation rubrics criteria. These rubrics guide the synthesis of VQA pairs, ensuring coverage across defect type identification, spatial distribution, morphology, and root cause analysis.
Our dual assessment framework aligns rule-based metrics with LLM-Judge scores via Bayesian optimization, enabling reliable automated evaluation. Through curriculum-based reinforcement learning with Group Sequence Policy Optimization (GSPO) and rubric-aligned rewards, our 4B-parameter Qwen3-VL model achieves a 6.493 LLM-Judge score, closely approaching Gemini-3-Flash (7.149) while enabling complete on-premise deployment. We demonstrate that small models with domain-specific training can surpass proprietary large models in specialized industrial visual understanding, offering a viable path for privacy-preserving, cost-effective deployment in semiconductor manufacturing.

**arXiv ID:** 2604.27629
</details>

<details>
<summary><strong>In-Context Prompting Obsoletes Agent Orchestration for Procedural Tasks</strong> - Simon Dennis, Michael Diamond, Rivaan Patil, Kevin Shabahang, Hao Guo - [[pdf]](https://arxiv.org/pdf/2604.27891)</summary>

**Abstract:** Agent orchestration frameworks -- LangGraph, CrewAI, Google ADK, OpenAI Agents SDK, and others -- place an external orchestrator above the LLM, tracking state and injecting routing instructions at every turn. We present a controlled comparison showing that for procedural tasks, this architecture is dominated by a simpler alternative: putting the entire procedure in the system prompt and letting the model self-orchestrate. Across three domains -- travel booking (14 nodes), Zoom technical support (14 nodes), and insurance claims processing (55 nodes) -- we evaluate 200 conversations per condition using LLM-as-judge scoring on five quality criteria. The in-context approach scores 4.53--5.00 on a 5-point scale while a LangGraph orchestrator using the same model scores 4.17--4.84. The orchestrated system fails on 24% of travel, 9% of Zoom, and 17% of insurance conversations, compared to 11.5%, 0.5%, and 5% for the in-context baseline. While external orchestration may have been necessary for earlier models, advances in frontier model capabilities have made it unnecessary for multi-turn conversations following a defined procedure.

**arXiv ID:** 2604.27891
</details>

<details>
<summary><strong>Optimal Control with Natural Images: Efficient Reinforcement Learning using Overcomplete Sparse Codes</strong> - Peter N. Loxley - [[pdf]](https://arxiv.org/pdf/2412.08893)</summary>

**Abstract:** Optimal control and sequential decision making are widely used in many complex tasks. Optimal control over a sequence of natural images is a first step towards understanding the role of vision in control. Here, we formalize this problem as a reinforcement learning task, and derive general conditions under which an image includes enough information to implement an optimal policy. Reinforcement learning is shown to provide a computationally efficient method for finding optimal policies when natural images are encoded into "efficient" image representations. This is demonstrated by introducing a new reinforcement learning benchmark that easily scales to large numbers of states and long horizons. In particular, by representing each image as an overcomplete sparse code, we are able to efficiently solve an optimal control task that is orders of magnitude larger than those tasks solvable using complete codes. Theoretical justification for this behaviour is provided. This work also demonstrates that deep learning is not necessary for efficient optimal control with natural images.

**arXiv ID:** 2412.08893
</details>

<details>
<summary><strong>RouteFormer: A Transformer-Based Routing Framework for Autonomous Vehicles</strong> - Yazan Youssef, Paulo Ricardo Marques de Araujo, Aboelmagd Noureldin, Sidney Givigi - [[pdf]](https://arxiv.org/pdf/2504.05407)</summary>

**Abstract:** Autonomous surveillance missions in Internet of Things (IoT) networks often involve solving NP-hard combinatorial optimization problems to ensure efficient resource utilization. To address the limitations of conventional heuristics in dynamic environments, we propose RouteFormer, a novel framework for single-agent routing in graph-based terrains. RouteFormer creates a synergy between the global context awareness of the transformer self-attention mechanism and the adaptive decision-making capabilities of Reinforcement Learning (RL). This architecture allows the system to output optimized routing decisions that adapt to complex task dependencies and resource availability without requiring labeled training datasets. We evaluated RouteFormer on varying graph sizes designed to resemble realistic reconnaissance missions. The results indicate that our model effectively handles the complexity of missions requiring multiple action profiles, outperforming baseline approaches, in terms of both time and distance. Specifically, RouteFormer achieved 10\% and 7\% reduction in distance compared to the solutions obtained from well-established solvers like Concorde and Lin-Kernighan-Helsgaun-3 (LKH-3). This improvement was achieved by effectively incorporating mission-specific constraints that traditional solvers overlook. The proposed framework serves as a modular, scalable pipeline for diverse autonomous scheduling and routing tasks.

**arXiv ID:** 2504.05407
</details>

<details>
<summary><strong>Autonomous Synchronization of Discrete-Time Heterogeneous Multiagent Systems</strong> - Wei Hu, Quanyi Liang - [[pdf]](https://arxiv.org/pdf/2605.04627)</summary>

**Abstract:** This paper investigates the autonomous synchronization problem for discrete-time heterogeneous multiagent systems.
The synchronization problem is transformed into the asymptotic decoupling problem of stable modes in a class of discrete-time linear time-varying systems,
for which we provide a sufficient condition.
Leveraging this condition, synchronization conditions are established.
The synchronization conditions are based on the average of the agents' initial dynamic matrices,
without requiring the differences among these matrices to be small.
This approach reduces the conservativeness of existing conditions and achieves a unification of both homogeneous and heterogeneous systems.
Numerical simulation results are provided to support the theoretical findings.

**arXiv ID:** 2605.04627
</details>

<details>
<summary><strong>Free Energy-Driven Reinforcement Learning with Adaptive Advantage Shaping for Unsupervised Reasoning in LLMs</strong> - Yiming Huang, Zhenbo Shi, Xin-Cheng Wen, Jichuan Zeng, Cuiyun Gao, Peiyi Han, Chuanyi Liu - [[pdf]](https://arxiv.org/pdf/2605.04065)</summary>

**Abstract:** Unsupervised reinforcement learning (RL) has emerged as a promising paradigm for enabling self-improvement in large language models (LLMs). However, existing unsupervised RL-based methods often lack the capacity to adapt to the model's evolving reasoning capabilities during training. Therefore, these methods can misdirect policy optimization in the absence of ground-truth supervision. To address this issue, we introduce FREIA, a novel RL-based algorithm built on two key innovations: (1) Free Energy-Driven Reward (FER) adapts rewards to balance consensus and exploration based on the Free Energy Principle. (2) Adaptive Advantage Shaping (AAS) adaptively adjusts learning signals based on the statistical characteristics of sampled rewards. Empirical evaluations on nine datasets across three reasoning tasks showcase that FREIA outperforms other unsupervised RL-based baselines. Notably, in mathematical reasoning tasks, FREIA surpasses other methods by an average of 0.5 to 3.5 points in Pass@1 using the DeepSeek-R1-Distill-Qwen-1.5B model.

**arXiv ID:** 2605.04065
</details>

<details>
<summary><strong>Reinforcement Learning for Compositional Generalization with Outcome-Level Optimization</strong> - Xiyan Fu, Wei Liu - [[pdf]](https://arxiv.org/pdf/2605.04920)</summary>

**Abstract:** Compositional generalization refers to correctly interpret novel combinations of known primitives, which remains a major challenge. Existing approaches often rely on supervised fine-tuning, which encourages models to imitate target outputs. This token-level training paradigm fails to capture the global compositional structure required for generalizing to unseen combinations. In this work, we investigate whether compositional generalization can instead be improved through outcome-level reinforcement learning. We adopt Group Relative Policy Optimization to optimize models based on feedback on their final outputs. Within this framework, we explore both a simple binary outcome reward and a composite reward that provides additional composition feedback. Experiments on multiple compositional benchmarks show that reinforcement learning improves compositional generalization compared to supervised fine-tuning. Further analysis reveals that supervised models tend to overfit frequent training compositions, whereas reinforcement learning improves compositional generalization by reshaping the output distribution, particularly for more complex composition types.

**arXiv ID:** 2605.04920
</details>

<details>
<summary><strong>ProMediate: A Socio-cognitive framework for evaluating proactive agents in multi-party negotiation</strong> - Ziyi Liu, Bahar Sarrafzadeh, Pei Zhou, Longqi Yang, Jieyu Zhao, Ashish Sharma - [[pdf]](https://arxiv.org/pdf/2510.25224)</summary>

**Abstract:** While Large Language Models (LLMs) are increasingly used in agentic frameworks to assist individual users, there is a growing need for agents that can proactively manage complex, multi-party collaboration. Systematic evaluation methods for such proactive agents remain scarce, limiting progress in developing AI that can effectively support multiple people together. Negotiation offers a demanding testbed for this challenge, requiring socio-cognitive intelligence to navigate conflicting interests between multiple participants and multiple topics and build consensus. Here, we present ProMediate, the first framework for evaluating proactive AI mediator agents in complex, multi-topic, multi-party negotiations. ProMediate consists of two core components: (i) a simulation testbed based on realistic negotiation cases and theory-driven difficulty levels (ProMediate-Easy, ProMediate-Medium, and ProMediate-Hard), with a plug-and-play proactive AI mediator grounded in socio-cognitive mediation theories, capable of flexibly deciding when and how to intervene; and (ii) a socio-cognitive evaluation framework with a new suite of metrics to measure consensus changes, intervention latency, mediator effectiveness, and intelligence. Together, these components establish a systematic framework for assessing the socio-cognitive intelligence of proactive AI agents in multi-party settings. Our results show that a socially intelligent mediator agent outperforms a generic baseline, via faster, better-targeted interventions. In the ProMediate-Hard setting, our social mediator increases consensus change by 3.6 percentage points compared to the generic baseline (10.65\% vs 7.01\%) while being 77\% faster in response (15.98s vs. 3.71s). In conclusion, ProMediate provides a rigorous, theory-grounded testbed to advance the development of proactive, socially intelligent agents.

**arXiv ID:** 2510.25224
</details>

<details>
<summary><strong>Beyond Majority Voting: Towards Fine-grained and More Reliable Reward Signal for Test-Time Reinforcement Learning</strong> - Weiqin Wang, Yile Wang, Kehao Chen, Hui Huang - [[pdf]](https://arxiv.org/pdf/2512.15146)</summary>

**Abstract:** Test-time reinforcement learning mitigates the reliance on annotated data by using majority voting results as pseudo-labels, emerging as a complementary direction to reinforcement learning with verifiable rewards (RLVR) for improving reasoning ability. However, this voting strategy often induces confirmation bias and suffers from sparse rewards, limiting the overall performance. In this work, we propose subgroup-specific step-wise confidence-weighted pseudo-label estimation (SCOPE), a framework integrating model confidence and dynamic subgroup partitioning to address these issues. Specifically, SCOPE integrates the proposed step-wise confidence into pseudo label estimation, prioritizing high-quality reasoning paths over simple frequency count. Furthermore, it dynamically partitions the candidate outputs pool into independent subgroups by balancing reasoning quality against exploration diversity. By deriving local consensus via repeat sampling for each sub group, SCOPE provides diverse supervision targets to encourage broader exploration. We conduct experiments across various models and benchmarks, experimental results show that SCOPE consistently outperforms recent baselines. Notably, SCOPE achieving relative improvements of 13.1% on challenging AIME 2025 and 8.1% on AMC. The code is released at this https URL.

**arXiv ID:** 2512.15146
</details>

<details>
<summary><strong>Constraint-Enhanced Reinforcement Learning Based on Dynamic Decoupled Spherical Radial Squashing</strong> - Qijun Liao, Zhaoxin Yu, Jue Yang - [[pdf]](https://arxiv.org/pdf/2605.04185)</summary>

**Abstract:** When deploying reinforcement learning policies to physical robots, actuator rate constraints -- hard limits on how fast each joint can move per control step -- are unavoidable. These limits vary substantially across joints due to differences in motor inertia, power bandwidth, and transmission stiffness, creating pronounced heterogeneity that existing methods fail to handle geometrically: the per-joint feasible region forms a high-dimensional box in action-increment space, yet QP projection and spherical parameterization methods impose isotropic ball-shaped constraints, exponentially under-covering the true feasible set as heterogeneity grows. This paper proposes Dynamic Decoupled Spherical Radial Squashing (DD-SRad), which resolves this mismatch by computing a position-adaptive radius independently for each actuator, achieving tight alignment with the true per-joint feasible region. DD-SRad satisfies per-step hard constraints with probability~1, preserves well-conditioned gradients throughout training, and admits exact policy gradient backpropagation with zero runtime solver overhead. MuJoCo benchmark experiments demonstrate the highest task return at zero constraint violation -- matching the unconstrained upper bound -- with 30%--50% improvement in constraint-space coverage over spherical baselines. High-fidelity IsaacLab simulations with Unitree H1 and G1 humanoid robots confirm end-to-end optimality parameterized directly from official joint specifications, validating a systematic pathway from hardware datasheets to safe deployment.

**arXiv ID:** 2605.04185
</details>

<details>
<summary><strong>Hierarchical Support Vector State Partitioning for Distilling Black Box Reinforcement Learning Policies</strong> - Senne Deproost, Mehrdad Asadi, Ann Nowé - [[pdf]](https://arxiv.org/pdf/2605.04254)</summary>

**Abstract:** We introduce State Vector Space Partitioning (SVSP), a novel method to mimic a black box reinforcement learning policy using a set of human-interpretable subpolicies. By partitioning a distillation dataset of state action pairs with linear support vector machine splits, SVSP constructs a compact and structured representation of the original policy. Our method improves mean return by +7.4\% over previous critic driven state partitioning attempts such as Voronoi State Partitioning (VSP) and +2.8\% over the original TD3 policy, while reducing the number of required subpolicies against VSP by 82.1\%. Our results pave the path towards a more flexible form of distillation where both the decision boundary and surrogate models can be chosen within a margin of the original black box behavior.

**arXiv ID:** 2605.04254
</details>

<details>
<summary><strong>Data-dependent Exploration for Online Reinforcement Learning from Human Feedback</strong> - Zhen-Yu Zhang, Yuting Tang, Jiandong Zhang, Lanjihong Ma, Masashi Sugiyama - [[pdf]](https://arxiv.org/pdf/2605.04477)</summary>

**Abstract:** Online reinforcement learning from human feedback (RLHF) has emerged as a promising paradigm for aligning large language models (LLMs) by continuously collecting new preference feedback during training. A foundational challenge in this setting is exploration, which requires algorithms that enable the LLMs to generate informative comparisons that improve sample-efficiency in online RLHF. Existing exploration strategies often derive bonuses via on-policy expectations, which are difficult to estimate reliably from the limited historical preference data available during training; as a result, the policy can prematurely down-weight under-explored regions that may contain high-value behaviors. In this paper, we propose data-dependent exploration for preference optimization (DEPO), a simple and scalable method that leverages historical data to construct an extra uncertainty bonus for high-uncertainty regions, encouraging exploration toward potentially high-value data. Theoretically, we provide a data-dependent regret bound for the proposed algorithm, showing that it adapts to the hardness of the learning task itself and can be tighter than worst-case bounds in practice. Empirically, the proposed method consistently outperforms strong baselines across benchmarks, demonstrating improved sample efficiency.

**arXiv ID:** 2605.04477
</details>

<details>
<summary><strong>SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning</strong> - Lirui Luo, Guoxi Zhang, Hongming Xu, Cong Fang, Qing Li - [[pdf]](https://arxiv.org/pdf/2605.04712)</summary>

**Abstract:** In deep reinforcement learning (DRL), an agent is trained from a stream of experience. In a continual learning setting, such agents can suffer from plasticity loss: their ability to learn new skills from new experiences diminishes over training. Recently, Mixture-of-Experts (MoE) networks have been reported to enable scaling laws and facilitate the learning of diverse skills. However, in continual reinforcement learning settings, their performance can degenerate as learning proceeds, indicating a loss of plasticity. To address this, building on Neural Tangent Kernel (NTK) theory, we formalize the plasticity loss in MoE policies as a loss of spectral plasticity. We then derive a tractable proxy for spectral plasticity, one expressible in terms of individual expert feature matrices. Leveraging this proxy, we introduce SPHERE, a practical Parseval penalty tailored for MoE-based policies that alleviates the loss of spectral plasticity. On MetaWorld and HumanoidBench, SPHERE improves average success under continual RL by 133% and 50% over an unregularized MoE baseline, while maintaining higher spectral plasticity throughout training.

**arXiv ID:** 2605.04712
</details>

<details>
<summary><strong>Unified Framework of Distributional Regret in Multi-Armed Bandits and Reinforcement Learning</strong> - Harin Lee, Min-hwan Oh - [[pdf]](https://arxiv.org/pdf/2605.05102)</summary>

**Abstract:** We study the distribution of regret in stochastic multi-armed bandits and episodic reinforcement learning through a unified framework. We formalize a distributional regret bound as a probabilistic guarantee that holds uniformly over all confidence levels $\delta \in (0,1]$, thereby characterizing the regret distribution across the full range of $\delta$. We present a simple UCBVI-style algorithm with exploration bonus $\min\{c_{1,k}/N, c_{2,k}/\sqrt{N}\}$, where $N$ denotes the visit count and $(c_{1,k},c_{2,k})$ are user-specified parameters. For arbitrary parameter sequences, we derive general gap-independent and gap-dependent distributional regret bounds, yielding a principled characterization of how the parameters control the trade-off between expected performance, tail risk, and instance-dependent behavior. In particular, our bounds achieve optimal trade-offs between expected and distributional regret in both minimax and instance-dependent regimes. As a special case, for multi-armed bandits with $A$ arms and horizon $T$, we obtain a distributional regret bound of order $\mathcal{O}(\sqrt{AT}\log(1/\delta))$, confirming the conjecture of Lattimore & Szepesvári (2020, Section 17.1) for the first time.

**arXiv ID:** 2605.05102
</details>

<details>
<summary><strong>Autonomous Laparoscope Control through Unified Mechanics-Based Representation of Multimodal Intraoperative Information</strong> - Xiaojian Li, Jin Fang, Yudong Shi, Xilin Xiao, Kai Yan, Kang Min, Ling Li, Hua Tang, Hangjie Mo - [[pdf]](https://arxiv.org/pdf/2605.04408)</summary>

**Abstract:** Laparoscope-holding robots can provide surgeons with a stable laparoscopic field of view (FOV) and reduce the burden on human assistants. To maintain an ideal intraoperative FOV, the robot must continuously adjust the laparoscope pose according to intraoperative information. However, intraoperative multimodal signals, such as position, force/torque, and images, differ markedly in physical meaning and units, making it difficult to build a unified representation and to generate control commands that can be used directly for laparoscope control. To address this issue, we propose a laparoscope-holding robot control method based on unified mechanics modeling of multimodal information. First, we design mapping strategies for multiple intraoperative sources, including position, force/torque, and images, and unify them into an equivalent-wrench representation in the operational space. Then, using a task-priority scheme, we inject the wrenches into the task space and the null space, respectively, and synthesize laparoscope control commands via task-priority projection, thereby achieving consistent representation and coordinated fusion of multimodal information within a single framework. Finally, taking the intraoperative remote center of motion (RCM) position, force/torque sensor readings, and laparoscopic images as examples, we construct an RCM-constraint wrench to enforce the RCM geometric constraint and reduce the contact force at the trocar site, a laparoscope-manipulation wrench to enable compliant dragging, and an instrument-tracking wrench to achieve autonomous visual tracking of the instruments. Experiments on a surgical phantom and in vivo porcine trials demonstrate that the proposed method supports multi-task operation, including compliant laparoscope manipulation and autonomous instrument tracking, while maintaining the RCM constraint and reducing sustained trocar-site loading.

**arXiv ID:** 2605.04408
</details>

<details>
<summary><strong>Collision-Aware Object-Goal Visual Navigation via Two-Stage Deep Reinforcement Learning</strong> - Hongwu Wang, Shiwei Lian, Feitian Zhang - [[pdf]](https://arxiv.org/pdf/2502.13498)</summary>

**Abstract:** Object-goal visual navigation aims to reach a specific target object using egocentric visual observations. Recent deep reinforcement learning (DRL) approaches have achieved promising success rates but often neglect collisions during evaluation, limiting real-world deployment. To address this issue, this letter introduces a collision-aware evaluation metric, namely collision-free success rate (CF-SR), to explicitly measure navigation performance under collision constraints. In addition, collision-free success weighted by path length (CF-SPL) is adopted to further evaluate navigation efficiency. Furthermore, a two-stage DRL training framework with collision prediction is proposed to improve collision-free navigation performance. In the first stage, a collision prediction module is trained by supervising the agent's collision states during exploration. In the second stage, leveraging the trained collision prediction, the agent learns to navigate toward target objects while avoiding collision. Extensive experiments across multiple navigation models in the AI2-THOR environment demonstrate consistent improvements in both CF-SR and CF-SPL. Real-world experiments further validate the effectiveness and generalization capability of the proposed framework.

**arXiv ID:** 2502.13498
</details>

<details>
<summary><strong>Efficient Model-Based Reinforcement Learning for Robot Control via Online Optimization</strong> - Fang Nan, Hao Ma, Qinghua Guan, Josie Hughes, Michael Muehlebach, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2510.18518)</summary>

**Abstract:** We present an online model-based reinforcement learning algorithm suitable for controlling complex robotic systems directly in the real world. Unlike prevailing sim-to-real pipelines that rely on extensive offline simulation and model-free policy optimization, our method builds a dynamics model from real-time interaction data and performs policy updates guided by the learned dynamics model. This efficient model-based reinforcement learning scheme significantly reduces the number of samples to train control policies, enabling direct training on real-world rollout data. This significantly reduces the influence of bias in the simulated data, and facilitates the search for high-performance control policies. We adopt online optimization analysis to derive sublinear regret bounds under stochastic online optimization assumptions, providing formal guarantees on performance improvement as more interaction data are collected. Experimental evaluations were performed on a hydraulic excavator arm and a soft robot arm, where the algorithm demonstrates strong sample efficiency compared to model-free reinforcement learning methods, reaching comparable performance within hours. Robust adaptation to shifting dynamics was also observed when the payload condition was randomized. Our approach paves the way toward efficient and reliable on-robot learning for a broad class of challenging control tasks.

**arXiv ID:** 2510.18518
</details>

<details>
<summary><strong>3D Generation for Embodied AI and Robotic Simulation: A Survey</strong> - Tianwei Ye, Yifan Mao, Minwen Liao, Jian Liu, Chunchao Guo, Dazhao Du, Quanxin Shou, Fangqi Zhu, Song Guo - [[pdf]](https://arxiv.org/pdf/2604.26509)</summary>

**Abstract:** Embodied AI and robotic systems increasingly depend on scalable, diverse, and physically grounded 3D content for simulation-based training and real-world deployment. While 3D generative modeling has advanced rapidly, embodied applications impose requirements far beyond visual realism: generated objects must carry kinematic structure and material properties, scenes must support interaction and task execution, and the resulting content must bridge the gap between simulation and reality. This survey reviews 3D generation for embodied AI and organizes the literature around three roles that 3D generation plays in embodied systems. In Data Generator, 3D generation produces simulation-ready objects and assets, including articulated, physically grounded, and deformable content for downstream interaction; in Simulation Environments, it constructs interactive and task-oriented worlds, spanning structure-aware, controllable, and agentic scene generation; and in Sim2Real Bridge, it supports digital twin reconstruction, data augmentation, and synthetic demonstrations for downstream robot learning and real-world transfer. We also show that the field is shifting from visual realism toward interaction readiness, and we identify the main bottlenecks, including limited physical annotations, the gap between geometric quality and physical validity, fragmented evaluation, and the persistent sim-to-real divide, that must be addressed for 3D generation to become a dependable foundation for embodied intelligence. Our project page is at this https URL.

**arXiv ID:** 2604.26509
</details>

<details>
<summary><strong>Stability of Control Lyapunov Function Guided Reinforcement Learning</strong> - Zachary Olkin, William D. Compton, Aaron D. Ames - [[pdf]](https://arxiv.org/pdf/2605.01978)</summary>

**Abstract:** Reinforcement learning (RL) has become the de facto method for achieving locomotion on humanoid robots in practice, yet stability analysis of the corresponding control policies is lacking. Recent work has attempted to merge control theoretic ideas with reinforcement learning through control guided learning. A notable example of this is the use of a control Lyapunov function (CLF) to synthesize the reinforcement learning rewards, a technique known as CLF-RL, which has shown practical success. This paper investigates the stability properties of optimal controllers using CLF-RL with the goal of bridging experimentally observed stability with theoretical guarantees. The RL problem is viewed as an optimal control problem and exponential stability is proven in both continuous and discrete time using both core CLF reward terms and the additional terms used in practice. The theoretical bounds are numerically verified on systems such as the double integrator and cart-pole. Finally, the CLF guided rewards are implemented for a walking humanoid robot to generate stable periodic orbits.

**arXiv ID:** 2605.01978
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
