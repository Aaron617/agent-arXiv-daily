# Agent arXiv Daily

**Last Updated:** 2026-04-22 03:39:16

**Total Papers:** 95

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (6 papers)</h2></summary>

<details>
<summary><strong>WebUncertainty: Dual-Level Uncertainty Driven Planning and Reasoning For Autonomous Web Agent</strong> - Lingfeng Zhang, Yongan Sun, Jinpeng Hu, Hui Ma, Yang Ying, Kuien Liu, Zenglin Shi, Meng Wang - [[pdf]](https://arxiv.org/pdf/2604.17821)</summary>

**Abstract:** Recent advancements in large language models (LLMs) have empowered autonomous web agents to execute natural language instructions directly on real-world webpages. However, existing agents often struggle with complex tasks involving dynamic interactions and long-horizon execution due to rigid planning strategies and hallucination-prone reasoning. To address these limitations, we propose WebUncertainty, a novel autonomous agent framework designed to tackle dual-level uncertainty in planning and reasoning. Specifically, we design a Task Uncertainty-Driven Adaptive Planning Mechanism that adaptively selects planning modes to navigate unknown environments. Furthermore, we introduce an Action Uncertainty-Driven Monte Carlo tree search (MCTS) Reasoning Mechanism. This mechanism incorporates the Confidence-induced Action Uncertainty (ConActU) strategy to quantify both aleatoric uncertainty (AU) and epistemic uncertainty (EU), thereby optimizing the search process and guiding robust decision-making. Experimental results on the WebArena and WebVoyager benchmarks demonstrate that WebUncertainty achieves superior performance compared to state-of-the-art baselines.

**arXiv ID:** 2604.17821
</details>

<details>
<summary><strong>Speculative End-Turn Detector for Efficient Speech Chatbot Assistant</strong> - Hyunjong Ok, Suho Yoo, Jaeho Lee - [[pdf]](https://arxiv.org/pdf/2503.23439)</summary>

**Abstract:** Spoken dialogue systems powered by large language models have demonstrated remarkable abilities in understanding human speech and generating appropriate spoken responses. However, these systems struggle with end-turn detection (ETD) -- the ability to distinguish between user turn completion and hesitation. This limitation often leads to premature or delayed responses, disrupting the flow of spoken conversations. In this paper, we introduce the ETD Dataset, the first public dataset for end-turn detection. The ETD dataset consists of both synthetic speech data generated with text-to-speech models and real-world speech data collected from web sources. We also propose SpeculativeETD, a novel collaborative inference framework that balances efficiency and accuracy to improve real-time ETD in resource-constrained environments. Our approach jointly employs a lightweight GRU-based model, which rapidly detects the non-speaking units in real-time on local devices, and a high-performance Wav2vec-based model running on the server to make a more challenging classification of distinguishing turn ends from mere pauses. Experiments demonstrate that the proposed SpeculativeETD significantly improves ETD accuracy while keeping the required computations low. Datasets and code will be available after the review.

**arXiv ID:** 2503.23439
</details>

<details>
<summary><strong>LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization</strong> - Jiaqi Tang, Yu Xia, Yi-Feng Wu, Yuwei Hu, Yuhui Chen, Qing-Guo Chen, Xiaogang Xu, Xiangyu Wu, Hao Lu, Yanqing Ma, Shiyin Lu, Qifeng Chen - [[pdf]](https://arxiv.org/pdf/2506.09373)</summary>

**Abstract:** The advent of autonomous agents is transforming interactions with Graphical User Interfaces (GUIs) by employing natural language as a powerful intermediary. Despite the predominance of Supervised Fine-Tuning (SFT) methods in current GUI agents for achieving spatial localization, these methods face substantial challenges due to their limited capacity to accurately perceive positional data. Existing strategies, such as reinforcement learning, often fail to assess positional accuracy effectively, thereby restricting their utility. In response, we introduce Location Preference Optimization (LPO), a novel approach that leverages locational data to optimize interaction preferences. LPO uses information entropy to predict interaction positions by focusing on zones rich in information. Besides, it further introduces a dynamic location reward function based on physical distance, reflecting the varying importance of interaction positions. Supported by Group Relative Preference Optimization (GRPO), LPO facilitates an extensive exploration of GUI environments and significantly enhances interaction precision. Comprehensive experiments demonstrate LPO's superior performance, achieving SOTA results across both offline benchmarks and real-world online evaluations. Our code will be made publicly available soon, at this https URL.

**arXiv ID:** 2506.09373
</details>

<details>
<summary><strong>Best Agent Identification for General Game Playing</strong> - Matthew Stephenson, Alex Newcombe, Eric Piette, Dennis Soemers - [[pdf]](https://arxiv.org/pdf/2507.00451)</summary>

**Abstract:** We present an efficient and generalised procedure to accurately identify the best (or near best) performing algorithm for each sub-task in a multi-problem domain. Our approach treats this as a set of best arm identification problems for multi-armed bandits, where each bandit corresponds to a specific task and each arm corresponds to a specific algorithm or agent. We propose an optimistic selection process based on a chosen confidence interval, that ranks each arm across all bandits in terms of their potential to influence our overall simple regret. We evaluate the performance of our approach on two of the most popular general game playing domains, the General Video Game AI (GVGAI) framework and the Ludii general game playing system, with the goal of selecting a high-performing agent for each game using a limited number of available trials. Compared to previous best arm identification algorithms for multi-armed bandits, our results demonstrate a substantial performance improvement in terms of average simple regret and average probability of error. This novel approach can be used to significantly improve the quality and accuracy of agent evaluation procedures for general game frameworks, as well as other multi-task domains with high algorithm runtimes.

**arXiv ID:** 2507.00451
</details>

<details>
<summary><strong>BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design</strong> - Deepro Choudhury, Sinead Williamson, Adam Goliński, Ning Miao, Freddie Bickford Smith, Michael Kirchhof, Yizhe Zhang, Tom Rainforth - [[pdf]](https://arxiv.org/pdf/2508.21184)</summary>

**Abstract:** We propose a general-purpose approach for improving the ability of large language models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian experimental design with large language models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) with respect to a variable of interest given the responses gathered previously. We show how this EIG can be formulated (and then estimated) in a principled way using a probabilistic model derived from the LLM's predictive distributions and provide detailed insights into key decisions in its construction and updating procedure. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20 Questions game and using the LLM to actively infer user preferences, compared to purely prompting-based design generation and other adaptive design strategies.

**arXiv ID:** 2508.21184
</details>

<details>
<summary><strong>Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning</strong> - Zhaoyan Gong, Zhiqiang Liu, Songze Li, Xiaoke Guo, Yuanxiang Liu, Xinle Deng, Zhizhen Liu, Lei Liang, Huajun Chen, Wen Zhang - [[pdf]](https://arxiv.org/pdf/2601.18296)</summary>

**Abstract:** Temporal Knowledge Graph Question Answering (TKGQA) is inherently challenging, as it requires sophisticated reasoning over dynamic facts with multi-hop dependencies and complex temporal constraints. Existing methods rely on fixed workflows and expensive closed-source APIs, limiting flexibility and scalability. We propose Temp-R1, the first autonomous end-to-end agent for TKGQA trained through reinforcement learning. To address cognitive overload in single-action reasoning, we expand the action space with specialized internal actions alongside external action. To prevent shortcut learning on simple questions, we introduce reverse curriculum learning that trains on difficult questions first, forcing the development of sophisticated reasoning before transferring to easier cases. Our 8B-parameter Temp-R1 achieves state-of-the-art performance on MultiTQ and TimelineKGQA, improving 19.8% over strong baselines on complex questions. Our work establishes a new paradigm for autonomous temporal reasoning agents. The code is available at this https URL.

**arXiv ID:** 2601.18296
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (21 papers)</h2></summary>

<details>
<summary><strong>How Adversarial Environments Mislead Agentic AI?</strong> - Zhonghao Zhan, Huichi Zhou, Zhenhao Li, Peiyuan Jing, Krinos Li, Hamed Haddadi - [[pdf]](https://arxiv.org/pdf/2604.18874)</summary>

**Abstract:** Tool-integrated agents are deployed on the premise that external tools ground their outputs in reality. Yet this very reliance creates a critical attack surface. Current evaluations benchmark capability in benign settings, asking "can the agent use tools correctly" but never "what if the tools lie". We identify this Trust Gap: agents are evaluated for performance, not for skepticism. We formalize this vulnerability as Adversarial Environmental Injection (AEI), a threat model where adversaries compromise tool outputs to deceive agents. AEI constitutes environmental deception: constructing a "fake world" of poisoned search results and fabricated reference networks around unsuspecting agents. We operationalize this via POTEMKIN, a Model Context Protocol (MCP)-compatible harness for plug-and-play robustness testing. We identify two orthogonal attack surfaces: The Illusion (breadth attacks) poison retrieval to induce epistemic drift toward false beliefs, while The Maze (depth attacks) exploit structural traps to cause policy collapse into infinite loops. Across 11,000+ runs on five frontier agents, we find a stark robustness gap: resistance to one attack often increases vulnerability to the other, demonstrating that epistemic and navigational robustness are distinct capabilities.

**arXiv ID:** 2604.18874
</details>

<details>
<summary><strong>Human-Guided Harm Recovery for Computer Use Agents</strong> - Christy Li, Sky CH-Wang, Andi Peng, Andreea Bobu - [[pdf]](https://arxiv.org/pdf/2604.18847)</summary>

**Abstract:** As LM agents gain the ability to execute actions on real computer systems, we need ways to not only prevent harmful actions at scale but also effectively remediate harm when prevention fails. We formalize a solution to this neglected challenge in post-execution safeguards as harm recovery: the problem of optimally steering an agent from a harmful state back to a safe one in alignment with human preferences. We ground preference-aligned recovery through a formative user study that identifies valued recovery dimensions and produces a natural language rubric. Our dataset of 1,150 pairwise judgments reveals context-dependent shifts in attribute importance, such as preferences for pragmatic, targeted strategies over comprehensive long-term approaches. We operationalize these learned insights in a reward model, re-ranking multiple candidate recovery plans generated by an agent scaffold at test time. To evaluate recovery capabilities systematically, we introduce BackBench, a benchmark of 50 computer-use tasks that test an agent's ability to recover from harmful states. Human evaluation shows our reward model scaffold yields higher-quality recovery trajectories than base agents and rubric-based scaffolds. Together, these contributions lay the foundation for a new class of agent safety methods -- ones that confront harm not only by preventing it, but by navigating its aftermath with alignment and intent.

**arXiv ID:** 2604.18847
</details>

<details>
<summary><strong>Four-Axis Decision Alignment for Long-Horizon Enterprise AI Agents</strong> - Vasundra Srininvasan - [[pdf]](https://arxiv.org/pdf/2604.19457)</summary>

**Abstract:** Long-horizon enterprise agents make high-stakes decisions (loan underwriting, claims adjudication, clinical review, prior authorization) under lossy memory, multi-step reasoning, and binding regulatory constraints. Current evaluation reports a single task-success scalar that conflates distinct failure modes and hides whether an agent is aligned with the standards its deployment environment requires. We propose that long-horizon decision behavior decomposes into four orthogonal alignment axes, each independently measurable and failable: factual precision (FRP), reasoning coherence (RCS), compliance reconstruction (CRR), and calibrated abstention (CAR). CRR is a novel regulatory-grounded axis; CAR is a measurement axis separating coverage from accuracy. We exercise the decomposition on a controlled benchmark (LongHorizon-Bench) covering loan qualification and insurance claims adjudication with deterministic ground-truth construction. Running six memory architectures, we find structure aggregate accuracy cannot see: retrieval collapses on factual precision; schema-anchored architectures pay a scaffolding tax; plain summarization under a fact-preservation prompt is a strong baseline on FRP, RCS, EDA, and CRR; and all six architectures commit on every case, exposing a decisional-alignment axis the field has not targeted. The decomposition also surfaced a pre-registered prediction of our own, that summarization would fail factual recall, which the data reversed at large magnitude, an axis-level reversal aggregate accuracy would have hidden. Institutional alignment (regulatory reconstruction) and decisional alignment (calibrated abstention) are under-represented in the alignment literature and become load-bearing once decisions leave the laboratory. The framework transfers to any regulated decisioning domain via two steps: build a fact schema, and calibrate the CRR auditor prompt.

**arXiv ID:** 2604.19457
</details>

<details>
<summary><strong>Revac: A Social Deduction Reasoning Agent</strong> - Mihir Shriniwas Arya, Avinash Anish, Aditya Ranjan - [[pdf]](https://arxiv.org/pdf/2604.19523)</summary>

**Abstract:** Social deduction games such as Mafia present a unique AI challenge: players must reason under uncertainty, interpret incomplete and intentionally misleading information, evaluate human-like communication, and make strategic elimination decisions. Unlike deterministic board games, success in Mafia depends not on perfect information or brute-force search, but on inference, memory, and adaptability in the presence of deception. This work presents the design and evaluation of Revac-8, an AI agent developed for the Social Deduction track of the MindGames Arena competition, where it achieved first place. The final agent evolved from a simple two-stage reasoning system into a multi-module architecture that integrates memory-based player profiling, social-graph analysis of accusations and defenses, and dynamic tone selection for communication. These results highlight the importance of structured memory and adaptive communication for achieving strong performance in high-stakes social environments.

**arXiv ID:** 2604.19523
</details>

<details>
<summary><strong>AblateCell: A Reproduce-then-Ablate Agent for Virtual Cell Repositories</strong> - Xue Xia, Chengkai Yao, Mingyu Tsoi, Xinjie Mao, Wenxuan Huang, Jiaqi Wei, Hao Wu, Cheng Tan, Lang Yu, Yuejin Yang, Siqi Sun, Zhangyang Gao - [[pdf]](https://arxiv.org/pdf/2604.19606)</summary>

**Abstract:** Systematic ablations are essential to attribute performance gains in AI Virtual Cells, yet they are rarely performed because biological repositories are under-standardized and tightly coupled to domain-specific data and formats. While recent coding agents can translate ideas into implementations, they typically stop at producing code and lack a verifier that can reproduce strong baselines and rigorously test which components truly matter. We introduce AblateCell, a reproduce-then-ablate agent for virtual cell repositories that closes this verification gap. AblateCell first reproduces reported baselines end-to-end by auto-configuring environments, resolving dependency and data issues, and rerunning official evaluations while emitting verifiable artifacts. It then conducts closed-loop ablation by generating a graph of isolated repository mutations and adaptively selecting experiments under a reward that trades off performance impact and execution cost. Evaluated on three single-cell perturbation prediction repositories (CPA, GEARS, BioLORD), AblateCell achieves 88.9% (+29.9% to human expert) end-to-end workflow success and 93.3% (+53.3% to heuristic) accuracy in recovering ground-truth critical components. These results enable scalable, repository-grounded verification and attribution directly on biological codebases.

**arXiv ID:** 2604.19606
</details>

<details>
<summary><strong>CentaurTA Studio: A Self-Improving Human-Agent Collaboration System for Thematic Analysis</strong> - Lei Wang, Min Huang, Eduard Dragut - [[pdf]](https://arxiv.org/pdf/2604.18589)</summary>

**Abstract:** Thematic analysis is difficult to scale: manual workflows are labor-intensive, while fully automated pipelines often lack controllability and transparent evaluation. We present \textbf{CentaurTA Studio}, a web-based system for self-improving human--agent collaboration in open coding and theme construction. The system integrates (1) a two-stage human feedback pipeline separating simulator drafting and expert validation, (2) persistent prompt optimization that distills validated feedback into reusable alignment principles, and (3) rubric-based evaluation with early stopping for process control.
Across three domains, CentaurTA achieves the strongest performance in both Open Coding and Theme Construction, reaching up to 92.12\% accuracy and consistently outperforming baseline systems. Agreement between the rubric-based LLM judge and human annotators reaches substantial reliability (average $\kappa = 0.68$). Ablation studies show that removing the feedback loop reduces performance from 90\% to 81\%, while eliminating the Critic or early stopping degrades accuracy or increases interaction cost. The full system reaches peak performance within 10 iterative rounds (about 25 minutes), demonstrating improved efficiency over expert-only refinement.

**arXiv ID:** 2604.18589
</details>

<details>
<summary><strong>From Craft to Kernel: A Governance-First Execution Architecture and Semantic ISA for Agentic Computers</strong> - Xiangyu Wen, Yuang Zhao, Xiaoyu Xu, Lingjun Chen, Changran Xu, Shu Chi, Jianrong Ding, Zeju Li, Haomin Li, Li Jiang, Fangxin Liu, Qiang Xu - [[pdf]](https://arxiv.org/pdf/2604.18652)</summary>

**Abstract:** The transition of agentic AI from brittle prototypes to production systems is stalled by a pervasive crisis of craft. We suggest that the prevailing orchestration paradigm-delegating the system control loop to large language models and merely patching with heuristic guardrails-is the root cause of this fragility. Instead, we propose Arbiter-K, a Governance-First execution architecture that reconceptualizes the underlying model as a Probabilistic Processing Unit encapsulated by a deterministic, neuro-symbolic kernel. Arbiter-K implements a Semantic Instruction Set Architecture (ISA) to reify probabilistic messages into discrete instructions. This allows the kernel to maintain a Security Context Registry and construct an Instruction Dependency Graph at runtime, enabling active taint propagation based on the data-flow pedigree of each reasoning node. By leveraging this mechanism, Arbiter-K precisely interdicts unsafe trajectories at deterministic sinks (e.g., high-risk tool calls or unauthorized network egress) and enables autonomous execution correction and architectural rollback when security policies are triggered. Evaluations on OpenClaw and NanoBot demonstrate that Arbiter-K enforces security as a microarchitectural property, achieving 76% to 95% unsafe interception for a 92.79% absolute gain over native policies. The code is publicly available at this https URL.

**arXiv ID:** 2604.18652
</details>

<details>
<summary><strong>Owner-Harm: A Missing Threat Model for AI Agent Safety</strong> - Dongcheng Zhang, Yiqing Jiang - [[pdf]](https://arxiv.org/pdf/2604.18658)</summary>

**Abstract:** Existing AI agent safety benchmarks focus on generic criminal harm (cybercrime, harassment, weapon synthesis), leaving a systematic blind spot for a distinct and commercially consequential threat category: agents harming their own deployers. Real-world incidents illustrate the gap: Slack AI credential exfiltration (Aug 2024), Microsoft 365 Copilot calendar-injection leaks (Jan 2024), and a Meta agent unauthorized forum post exposing operational data (Mar 2026). We propose Owner-Harm, a formal threat model with eight categories of agent behavior damaging the deployer. We quantify the defense gap on two benchmarks: a compositional safety system achieves 100% TPR / 0% FPR on AgentHarm (generic criminal harm) yet only 14.8% (4/27; 95% CI: 5.9%-32.5%) on AgentDojo injection tasks (prompt-injection-mediated owner harm). A controlled generic-LLM baseline shows the gap is not inherent to owner-harm (62.7% vs. 59.3%, delta 3.4 pp) but arises from environment-bound symbolic rules that fail to generalize across tool vocabularies. On a post-hoc 300-scenario owner-harm benchmark, the gate alone achieves 75.3% TPR / 3.3% FPR; adding a deterministic post-audit verifier raises overall TPR to 85.3% (+10.0 pp) and Hijacking detection from 43.3% to 93.3%, demonstrating strong layer complementarity. We introduce the Symbolic-Semantic Defense Generalization (SSDG) framework relating information coverage to detection rate. Two SSDG experiments partially validate it: context deprivation amplifies the detection gap 3.4x (R = 3.60 vs. R = 1.06); context injection reveals structured goal-action alignment, not text concatenation, is required for effective owner-harm detection.

**arXiv ID:** 2604.18658
</details>

<details>
<summary><strong>Characterizing AlphaEarth Embedding Geometry for Agentic Environmental Reasoning</strong> - Mashrekur Rahman, Samuel J. Barrett, Christina Last - [[pdf]](https://arxiv.org/pdf/2604.18715)</summary>

**Abstract:** Earth observation foundation models encode land surface information into dense embedding vectors, yet the geometric structure of these representations and its implications for downstream reasoning remain underexplored. We characterize the manifold geometry of Google AlphaEarth's 64-dimensional embeddings across 12.1 million Continental United States samples (2017--2023) and develop an agentic system that leverages this geometric understanding for environmental reasoning. The manifold is non-Euclidean: effective dimensionality is 13.3 (participation ratio) from 64 raw dimensions, with local intrinsic dimensionality of approximately 10. Tangent spaces rotate substantially, with 84\% of locations exceeding 60\textdegree{} and local-global alignment (mean$|\cos\theta| = 0.17$) approaching the random baseline of 0.125. Supervised linear probes indicate that concept directions rotate across the manifold, and compositional vector arithmetic using both PCA-derived and probe-derived directions yields poor precision. Retrieval instead produces physically coherent results, with local geometry predicting retrieval coherence ($R^2 = 0.32$). Building on this characterization, we introduce an agentic system with nine specialized tools that decomposes environmental queries into reasoning chains over a FAISS-indexed embedding database. A five-condition ablation (120 queries, three complexity tiers) shows that embedding retrieval dominates response quality ($\mu = 3.79 \pm 0.90$ vs.\ $3.03 \pm 0.77$ parametric-only; scale 1--5), with peak performance on multi-step comparisons ($\mu = 4.28 \pm 0.43$). A cross-model benchmark show that geometric tools reduce Sonnet 4.5's score by 0.12 points but improve Opus 4.6's by 0.07, with Opus achieving higher geometric grounding (3.38 vs.\ 2.64), suggesting that the value of geometric characterization scales with the reasoning capability of the consuming model.

**arXiv ID:** 2604.18715
</details>

<details>
<summary><strong>Towards Optimal Agentic Architectures for Offensive Security Tasks</strong> - Isaac David, Arthur Gervais - [[pdf]](https://arxiv.org/pdf/2604.18718)</summary>

**Abstract:** Agentic security systems increasingly audit live targets with tool-using LLMs, but prior systems fix a single coordination topology, leaving unclear when additional agents help and when they only add cost. We treat topology choice as an empirical systems question. We introduce a controlled benchmark of 20 interactive targets (10 web/API and 10 binary), each exposing one endpoint-reachable ground-truth vulnerability, evaluated in whitebox and blackbox modes. The core study executes 600 runs over five architecture families, three model families, and both access modes, with a separate 60-run long-context pilot reported only in the appendix. On the completed core benchmark, detection-any reaches 58.0% and validated detection reaches 49.8%. MAS-Indep attains the highest validated detection rate (64.2%), while SAS is the strongest efficiency baseline at $0.058 per validated finding. Whitebox materially outperforms blackbox (67.0% vs. 32.7% validated detection), and web materially outperforms binary (74.3% vs. 25.3%). Bootstrap confidence intervals and paired target-level deltas show that the dominant effects are observability and domain, while some leading whitebox topologies remain statistically close. The main result is a non-monotonic cost-quality frontier: broader coordination can improve coverage, but it does not dominate once latency, token cost, and exploit-validation difficulty are taken into account.

**arXiv ID:** 2604.18718
</details>

<details>
<summary><strong>ST-Prune: Training-Free Spatio-Temporal Token Pruning for Vision-Language Models in Autonomous Driving</strong> - Lin Sha, Haiyun Guo, Tao Wang, Cong Zhang, Min Huang, Jinqiao Wang, Qinghai Miao - [[pdf]](https://arxiv.org/pdf/2604.19145)</summary>

**Abstract:** Vision-Language Models (VLMs) have become central to autonomous driving systems, yet their deployment is severely bottlenecked by the massive computational overhead of multi-view camera and multi-frame video input. Existing token pruning methods, primarily designed for single-image inputs, treat each frame or view in isolation and thus fail to exploit the inherent spatio-temporal redundancies in driving scenarios. To bridge this gap, we propose ST-Prune, a training-free, plug-and-play framework comprising two complementary modules: Motion-aware Temporal Pruning (MTP) and Ring-view Spatial Pruning (RSP). MTP addresses temporal redundancy by encoding motion volatility and temporal recency as soft constraints within the diversity selection objective, prioritizing dynamic trajectories and current-frame content over static historical background. RSP further resolves spatial redundancy by exploiting the ring-view camera geometry to penalize bilateral cross-view similarity, eliminating duplicate projections and residual background that temporal pruning alone cannot suppress. These two modules together constitute a complete spatio-temporal pruning process, preserving key scene information under strict compression. Validated across four benchmarks spanning perception, prediction, and planning, ST-Prune establishes new state-of-the-art for training-free token pruning. Notably, even at 90\% token reduction, ST-Prune achieves near-lossless performance with certain metrics surpassing the full-model baseline, while maintaining inference speeds comparable to existing pruning approaches.

**arXiv ID:** 2604.19145
</details>

<details>
<summary><strong>Cyber Defense Benchmark: Agentic Threat Hunting Evaluation for LLMs in SecOps</strong> - Alankrit Chona, Igor Kozlov, Ambuj Kumar - [[pdf]](https://arxiv.org/pdf/2604.19533)</summary>

**Abstract:** We introduce the Cyber Defense Benchmark, a benchmark for measuring how well large language model (LLM) agents perform the core SOC analyst task of threat hunting: given a database of raw Windows event logs with no guided questions or hints, identify the exact timestamps of malicious events.
The benchmark wraps 106 real attack procedures from the OTRF Security-Datasets corpus - spanning 86 MITRE ATT&CK sub-techniques across 12 tactics - into a Gymnasium reinforcement-learning environment. Each episode presents the agent with an in-memory SQLite database of 75,000-135,000 log records produced by a deterministic campaign simulator that time-shifts and entity-obfuscates the raw recordings.
The agent must iteratively submit SQL queries to discover malicious event timestamps and explicitly flag them, scored CTF-style against Sigma-rule-derived ground truth.
Evaluating five frontier models - Claude Opus 4.6, GPT-5, Gemini 3.1 Pro, Kimi K2.5, and Gemini 3 Flash - on 26 campaigns covering 105 of 106 procedures, we find that all models fail dramatically: the best model (Claude Opus 4.6) submits correct flags for only 3.8% of malicious events on average, and no run across any model ever finds all flags.
We define a passing score as >= 50% recall on every ATT&CK tactic - the minimum bar for unsupervised SOC deployment. No model passes: the leader clears this bar on 5 of 13 tactics and the remaining four on zero.
These results suggest that current LLMs are poorly suited for open-ended, evidence-driven threat hunting despite strong performance on curated Q&A security benchmarks.

**arXiv ID:** 2604.19533
</details>

<details>
<summary><strong>An AI Agent Execution Environment to Safeguard User Data</strong> - Robert Stanley, Avi Verma, Lillian Tsai, Konstantinos Kallas, Sam Kumar - [[pdf]](https://arxiv.org/pdf/2604.19657)</summary>

**Abstract:** AI agents promise to serve as general-purpose personal assistants for their users, which requires them to have access to private user data (e.g., personal and financial information). This poses a serious risk to security and privacy. Adversaries may attack the AI model (e.g., via prompt injection) to exfiltrate user data. Furthermore, sharing private data with an AI agent requires users to trust a potentially unscrupulous or compromised AI model provider with their private data.
This paper presents GAAP (Guaranteed Accounting for Agent Privacy), an execution environment for AI agents that guarantees confidentiality for private user data. Through dynamic and directed user prompts, GAAP collects permission specifications from users describing how their private data may be shared, and GAAP enforces that the agent's disclosures of private user data, including disclosures to the AI model and its provider, comply with these specifications. Crucially, GAAP provides this guarantee deterministically, without trusting the agent with private user data, and without requiring any AI model or the user prompt to be free of attacks.
GAAP enforces the user's permission specification by tracking how the AI agent accesses and uses private user data. It augments Information Flow Control with novel persistent data stores and annotations that enable it to track the flow of private information both across execution steps within a single task, and also over multiple tasks separated in time. Our evaluation confirms that GAAP blocks all data disclosure attacks, including those that make other state-of-the-art systems disclose private user data to untrusted parties, without a significant impact on agent utility.

**arXiv ID:** 2604.19657
</details>

<details>
<summary><strong>VideoAgent: Personalized Synthesis of Scientific Videos</strong> - Xiao Liang, Bangxin Li, Zixuan Chen, Hanyue Zheng, Zhi Ma, Di Wang, Cong Tian, Quan Wang - [[pdf]](https://arxiv.org/pdf/2509.11253)</summary>

**Abstract:** The technical complexity of research papers often limits their reach, necessitating more accessible formats like scientific videos to disseminate key insights through engaging narration. However, existing automated methods primarily focus on static posters or slide presentations that remain template-bound and linear. Shifting to audience-adaptive video synthesis requires addressing non-linear narrative orchestration and the joint synchronization of disparate multimodal assets. We introduce VideoAgent, a modular framework that redefines scientific video synthesis as an intent-driven planning problem. By decoupling content understanding from multimodal synthesis, VideoAgent adaptively interleaves static slides with dynamic animations to match the semantic density of the narration. We further propose SciVidEval, a benchmark evaluating multimodal quality and pedagogical utility through automated metrics and human knowledge transfer studies. Extensive experiments demonstrate that VideoAgent effectively conveys complex technical logic with high narrative fidelity and communicative impact.

**arXiv ID:** 2509.11253
</details>

<details>
<summary><strong>SAGE-32B: Agentic Reasoning via Iterative Distillation</strong> - Basab Jha, Firoj Paudel, Ujjwal Puri, Ethan Henkel, Zhang Yuting, Mateusz Kowalczyk, Mei Huang, Choi Donghyuk, Wang Junhao - [[pdf]](https://arxiv.org/pdf/2601.04237)</summary>

**Abstract:** We demonstrate SAGE-32B, a 32 billion parameter language model that focuses on agentic reasoning and long range planning tasks. Unlike chat models that aim for general conversation fluency, SAGE-32B is designed to operate in an agentic loop, emphasizing task decomposition, tool usage, and error recovery. The model is initialized from the Qwen2.5-32B pretrained model and fine tuned using Iterative Distillation, a two stage training process that improves reasoning performance through rigorously tested feedback loops. SAGE-32B also introduces an inverse reasoning approach, which uses a meta cognition head to forecast potential failures in the planning process before execution. On agentic reasoning benchmarks including MMLU-Pro, AgentBench, and MATH-500, SAGE-32B achieves higher success rates in multi tool usage scenarios compared to similarly sized baseline models, while remaining competitive on standard reasoning evaluations. Model weights are publicly released at this https URL

**arXiv ID:** 2601.04237
</details>

<details>
<summary><strong>EvoMaster: A Foundational Evolving Agent Framework for Agentic Science at Scale</strong> - Xinyu Zhu, Yuzhu Cai, Zexi Liu, Cheng Wang, Fengyang Li, Wenkai Jin, Wanxu Liu, Zehao Bing, Bingyang Zheng, Jingyi Chai, Shuo Tang, Rui Ye, Yuwen Du, Xianghe Pang, Yaxin Du, Tingjia Miao, Yuzhi Zhang, Ruoxue Liao, Zhaohan Ding, Linfeng Zhang, Yanfeng Wang, Weinan E, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2604.17406)</summary>

**Abstract:** The convergence of large language models and agents is catalyzing a new era of scientific discovery: Agentic Science. While the scientific method is inherently iterative, existing agent frameworks are predominantly static, narrowly scoped, and lack the capacity to learn from trial and error. To bridge this gap, we present EvoMaster, a foundational evolving agent framework engineered specifically for Agentic Science at Scale. Driven by the core principle of continuous self-evolution, EvoMaster empowers agents to iteratively refine hypotheses, self-critique, and progressively accumulate knowledge across experimental cycles, faithfully mirroring human scientific inquiry. Crucially, as a domain-agnostic base harness, EvoMaster is exceptionally easy to scale up -- enabling developers to build and deploy highly capable, self-evolving scientific agents for arbitrary disciplines in approximately 100 lines of code. Built upon EvoMaster, we incubated the SciMaster ecosystem across domains such as machine learning, physics, and general science. Evaluations on four authoritative benchmarks (Humanity's Last Exam, MLE-Bench Lite, BrowseComp, and FrontierScience) demonstrate that EvoMaster achieves state-of-the-art scores of 41.1%, 75.8%, 73.3%, and 53.3%, respectively. It comprehensively outperforms the general-purpose baseline OpenClaw with relative improvements ranging from +159% to +316%, robustly validating its efficacy and generality as the premier foundational framework for the next generation of autonomous scientific discovery. EvoMaster is available at this https URL.

**arXiv ID:** 2604.17406
</details>

<details>
<summary><strong>Agentic Forecasting using Sequential Bayesian Updating of Linguistic Beliefs</strong> - Kevin Murphy - [[pdf]](https://arxiv.org/pdf/2604.18576)</summary>

**Abstract:** We present BLF (Bayesian Linguistic Forecaster), an agentic system for binary forecasting that achieves state-of-the-art performance on the ForecastBench benchmark. The system is built on three ideas. (1) A linguistic belief state: a semi-structured representation combining numerical probability estimates with natural-language evidence summaries, updated by the LLM at each step of an iterative tool-use loop. This contrasts with the common approach of appending all retrieved evidence to an ever-growing context. (2) Hierarchical multi-trial aggregation: running $K$ independent trials and combining them using logit-space shrinkage with a data-dependent prior. (3) Hierarchical calibration: Platt scaling with a hierarchical prior, which avoids over-shrinking extreme predictions for sources with skewed base rates. On 400 backtesting questions from the ForecastBench leaderboard, BLF outperforms all the top public methods, including Cassi, GPT-5, Grok~4.20, and Foresight-32B. Ablation studies show that the structured belief state is almost as impactful as web search access, and that shrinkage aggregation and hierarchical calibration each provide significant additional gains. In addition, we develop a robust back-testing framework with a leakage rate below 1.5\%, and use rigorous statistical methodology to compare different methods while controlling for various sources of noise.

**arXiv ID:** 2604.18576
</details>

<details>
<summary><strong>Visual Reasoning Agent: Robust Vision Systems in Remote Sensing via Inference-Time Scaling</strong> - Chung-En Johnny Yu, Brian Jalaian, Nathaniel D. Bastian - [[pdf]](https://arxiv.org/pdf/2509.16343)</summary>

**Abstract:** Building robust vision systems for high-stakes domains such as remote sensing requires stronger visual reasoning than what single-pass inference typically provides; yet, retraining large models is often computationally expensive and data intensive. We present Visual Reasoning Agent (VRA), a training-free agentic visual reasoning framework that orchestrates off-the-shelf large vision-language models (LVLMs) with a large reasoning model (LRM) through an iterative Think-Critique-Act loop for cross-model verification, self-critique, and recursive refinement. On the remote sensing benchmark VRSBench VQA dataset, VRA consistently outperforms multiple standalone LVLM baselines and achieves up to 40.67\% improvement on challenging question types spanning both perception and reasoning tasks. In addition, integrating three LVLMs with VRA improves the overall accuracy of the standalone LVLMs from 52.8% to 78.8%, demonstrating the effectiveness of agentic reasoning with increased inference-time compute.

**arXiv ID:** 2509.16343
</details>

<details>
<summary><strong>A Functionality-Grounded Benchmark for Evaluating Web Agents in E-commerce Domains</strong> - Xianren Zhang, Shreyas Prasad, Di Wang, Qiuhai Zeng, Suhang Wang, Wenbo Yan, Mat Hans - [[pdf]](https://arxiv.org/pdf/2508.15832)</summary>

**Abstract:** Web agents have shown great promise in performing many tasks on ecommerce website. To assess their capabilities, several benchmarks have been introduced. However, current benchmarks in the e-commerce domain face two major problems. First, they primarily focus on product search tasks (e.g., Find an Apple Watch), failing to capture the broader range of functionalities offered by real-world e-commerce platforms such as Amazon, including account management and gift card operations. Second, existing benchmarks typically evaluate whether the agent completes the user query, but ignore the potential risks involved. In practice, web agents can make unintended changes that negatively impact the user account or status. For instance, an agent might purchase the wrong item, delete a saved address, or incorrectly configure an auto-reload setting. To address these gaps, we propose a new benchmark called Amazon-Bench. To generate user queries that cover a broad range of tasks, we propose a data generation pipeline that leverages webpage content and interactive elements (e.g., buttons, check boxes) to create diverse, functionality-grounded user queries covering tasks such as address management, wish list management, and brand store following. To improve the agent evaluation, we propose an automated evaluation framework that assesses both the performance and the safety of web agents. We systematically evaluate different agents, finding that current agents struggle with complex queries and pose safety risks. These results highlight the need for developing more robust and reliable web agents.

**arXiv ID:** 2508.15832
</details>

<details>
<summary><strong>From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck for Long-Horizon Video Agents</strong> - Niu Lian, Yuting Wang, Hanshu Yao, Jinpeng Wang, Bin Chen, Yaowei Wang, Min Zhang, Shu-Tao Xia - [[pdf]](https://arxiv.org/pdf/2603.01455)</summary>

**Abstract:** While multimodal large language models have demonstrated impressive short-term reasoning, they struggle with long-horizon video understanding due to limited context windows and static memory mechanisms that fail to mirror human cognitive efficiency. Existing paradigms typically fall into two extremes: vision-centric methods that incur high latency and redundancy through dense visual accumulation, or text-centric approaches that suffer from detail loss and hallucination via aggressive captioning. To bridge this gap, we propose MM-Mem, a pyramidal multimodal memory architecture grounded in Fuzzy-Trace Theory. MM-Mem structures memory hierarchically into a Sensory Buffer, Episodic Stream, and Symbolic Schema, enabling the progressive distillation of fine-grained perceptual traces (verbatim) into high-level semantic schemas (gist). Furthermore, to govern the dynamic construction of memory, we derive a Semantic Information Bottleneck objective and introduce SIB-GRPO to optimize the trade-off between memory compression and task-relevant information retention. In inference, we design an entropy-driven top-down memory retrieval strategy. Extensive experiments across 4 benchmarks confirm that MM-Mem achieves state-of-the-art performance on both offline and streaming tasks, demonstrating robust generalization and validating the effectiveness of cognition-inspired memory organization. Code and associated configurations are publicly available at this https URL.

**arXiv ID:** 2603.01455
</details>

<details>
<summary><strong>Scaling Test-Time Compute for Agentic Coding</strong> - Joongwon Kim, Wannan Yang, Kelvin Niu, Hongming Zhang, Yun Zhu, Eryk Helenowski, Ruan Silva, Zhengxing Chen, Srinivasan Iyer, Manzil Zaheer, Daniel Fried, Hannaneh Hajishirzi, Sanjeev Arora, Gabriel Synnaeve, Ruslan Salakhutdinov, Anirudh Goyal - [[pdf]](https://arxiv.org/pdf/2604.16529)</summary>

**Abstract:** Test-time scaling has become a powerful way to improve large language models. However, existing methods are best suited to short, bounded outputs that can be directly compared, ranked or refined. Long-horizon coding agents violate this premise: each attempt produces an extended trajectory of actions, observations, errors, and partial progress taken by the agent. In this setting, the main challenge is no longer generating more attempts, but representing prior experience in a form that can be effectively selected from and reused. We propose a test-time scaling framework for agentic coding based on compact representations of rollout trajectories. Our framework converts each rollout into a structured summary that preserves its salient hypotheses, progress, and failure modes while discarding low-signal trace details. This representation enables two complementary forms of inference-time scaling. For parallel scaling, we introduce Recursive Tournament Voting (RTV), which recursively narrows a population of rollout summaries through small-group comparisons. For sequential scaling, we adapt Parallel-Distill-Refine (PDR) to the agentic setting by conditioning new rollouts on summaries distilled from prior attempts. Our method consistently improves the performance of frontier coding agents across SWE-Bench Verified and Terminal-Bench v2.0. For example, by using our method Claude-4.5-Opus improves from 70.9% to 77.6% on SWE-Bench Verified (mini-SWE-agent) and 46.9% to 59.1% on Terminal-Bench v2.0 (Terminus 1). Our results suggest that test-time scaling for long-horizon agents is fundamentally a problem of representation, selection, and reuse.

**arXiv ID:** 2604.16529
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>Do Agents Dream of Root Shells? Partial-Credit Evaluation of LLM Agents in Capture The Flag Challenges</strong> - Ali Al-Kaswan, Maksim Plotnikov, Maxim Hájek, Roland Vízner, Arie van Deursen, Maliheh Izadi - [[pdf]](https://arxiv.org/pdf/2604.19354)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly proposed for autonomous cybersecurity tasks, but their capabilities in realistic offensive settings remain poorly understood. We present DeepRed, an open-source benchmark for evaluating LLM-based agents on realistic Capture The Flag (CTF) challenges in isolated virtualized environments. DeepRed places an agent in a Kali attacker environment with terminal tools and optional web search, connected over a private network to a target challenge, and records full execution traces for analysis. To move beyond binary solved/unsolved outcomes, we introduce a partial-credit scoring method based on challenge-specific checkpoints derived from public writeups, together with an automated summarise-then-judge labelling pipeline for assigning checkpoint completion from logs. Using DeepRed, we benchmark ten commercially accessible LLMs on ten VM-based CTF challenges spanning different challenge categories. The results indicate that current agents remain limited: the best model achieves only 35% average checkpoint completion, performing strongest on common challenge types and weakest on tasks requiring non-standard discovery and longer-horizon adaptation.

**arXiv ID:** 2604.19354
</details>

<details>
<summary><strong>Plug-and-Play Dramaturge: A Divide-and-Conquer Approach for Iterative Narrative Script Refinement via Collaborative LLM Agents</strong> - Wenda Xie, Chao Guo, Yanqing Jing, Junle Wang, Yisheng Lv, Fei-Yue Wang - [[pdf]](https://arxiv.org/pdf/2510.05188)</summary>

**Abstract:** Although LLMs have been widely adopted for creative content generation, a single-pass process often struggles to produce high-quality long narratives. How to effectively revise and improve long narrative scripts like scriptwriters remains a significant challenge, as it demands a comprehensive understanding of the entire context to identify global structural issues and local detailed flaws, as well as coordinating revisions at multiple granularities and locations. Direct modifications by LLMs typically introduce inconsistencies between local edits and the overall narrative requirements. To address these issues, we propose Dramaturge, a task and feature oriented divide-and-conquer approach powered by hierarchical multiple LLM agents. It consists of a Global Review stage to grasp the overall storyline and structural issues, a Scene-level Review stage to pinpoint detailed scene and sentence flaws, and a Hierarchical Coordinated Revision stage that coordinates and integrates structural and detailed improvements throughout the script. The top-down task flow ensures that high-level strategies guide local modifications, maintaining contextual consistency. The review and revision workflow follows a coarse-to-fine iterative process, continuing through multiple rounds until no further substantive improvements can be made. Comprehensive experiments show that Dramaturge significantly outperforms all baselines in terms of script-level overall quality and scene-level details. Our approach is plug-and-play and can be easily integrated into existing methods to improve the generated scripts.

**arXiv ID:** 2510.05188
</details>

<details>
<summary><strong>Sentipolis: Emotion-Aware Agents for Social Simulations</strong> - Chiyuan Fu, Lyuhao Chen, Yunze Xiao, Weihao Xuan, Carlos Busso, Mona Diab - [[pdf]](https://arxiv.org/pdf/2601.18027)</summary>

**Abstract:** LLM agents are increasingly used for social simulation, yet emotion is often treated as a transient cue, causing emotional amnesia and weak long-horizon continuity. We present Sentipolis, a framework for emotionally stateful agents that integrates continuous Pleasure-Arousal-Dominance (PAD) representation, dual-speed emotion dynamics, and emotion--memory coupling. Across thousands of interactions over multiple base models and evaluators, Sentipolis improves emotionally grounded behavior, boosting communication, and emotional continuity. Gains are model-dependent: believability increases for higher-capacity models but can drop for smaller ones, and emotion-awareness can mildly reduce adherence to social norms, reflecting a human-like tension between emotion-driven behavior and rule compliance in social simulation. Network-level diagnostics show reciprocal, moderately clustered, and temporally stable relationship structures, supporting the study of cumulative social dynamics such as alliance formation and gradual relationship change.

**arXiv ID:** 2601.18027
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (28 papers)</h2></summary>

<details>
<summary><strong>Explicit Trait Inference for Multi-Agent Coordination</strong> - Suhaib Abdurahman, Etsuko Ishii, Katerina Margatina, Divya Bhargavi, Monica Sunkara, Yi Zhang - [[pdf]](https://arxiv.org/pdf/2604.19278)</summary>

**Abstract:** LLM-based multi-agent systems (MAS) show promise on complex tasks but remain prone to coordination failures such as goal drift, error cascades, and misaligned behaviors. We propose Explicit Trait Inference (ETI), a psychologically grounded method for improving coordination. ETI enables agents to infer and track partner characteristics along two established psychological dimensions--warmth (e.g., trust) and competence (e.g., skill)--from interaction histories to guide decisions. We evaluate ETI in controlled settings (economic games), where it reduces payoff loss by 45-77%, and in more realistic, complex multi-agent settings (MultiAgentBench), where it improves performance by 3-29% depending on the scenario and model, relative to a CoT baseline. Additional analysis shows that gains are closely linked to trait inference: ETI profiles predict agents' actions, and informative profiles drive improvements. These results highlight ETI as a lightweight and robust mechanism for improving coordination in diverse multi-agent settings, and provide the first systematic evidence that LLM agents can (i) reliably infer others' traits from interaction histories and (ii) leverage structured awareness of others' traits for coordination.

**arXiv ID:** 2604.19278
</details>

<details>
<summary><strong>From Experience to Skill: Multi-Agent Generative Engine Optimization via Reusable Strategy Learning</strong> - Beining Wu, Fuyou Mao, Jiong Lin, Cheng Yang, Jiaxuan Lu, Yifu Guo, Siyu Zhang, Yifan Wu, Ying Huang, Fu Li - [[pdf]](https://arxiv.org/pdf/2604.19516)</summary>

**Abstract:** Generative engines (GEs) are reshaping information access by replacing ranked links with citation-grounded answers, yet current Generative Engine Optimization (GEO) methods optimize each instance in isolation, unable to accumulate or transfer effective strategies across tasks and engines. We reframe GEO as a strategy learning problem and propose MAGEO, a multi-agent framework in which coordinated planning, editing, and fidelity-aware evaluation serve as the execution layer, while validated editing patterns are progressively distilled into reusable, engine-specific optimization skills. To enable controlled assessment, we introduce a Twin Branch Evaluation Protocol for causal attribution of content edits and DSV-CF, a dual-axis metric that unifies semantic visibility with attribution accuracy. We further release MSME-GEO-Bench, a multi-scenario, multi-engine benchmark grounded in real-world queries. Experiments on three mainstream engines show that MAGEO substantially outperforms heuristic baselines in both visibility and citation fidelity, with ablations confirming that engine-specific preference modeling and strategy reuse are central to these gains, suggesting a scalable learning-driven paradigm for trustworthy GEO. Code is available at this https URL

**arXiv ID:** 2604.19516
</details>

<details>
<summary><strong>Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models</strong> - Xudong Wang, Chaoning Zhang, Chenghao Li, Shuxu Chen, Qigan Sun, Jiaquan Zhang, Fachrina Dewi Puspitasari, Tae-Ho Kim, Jiwei Wei, Malu Zhang, Guoqing Wang, Yang Yang, Heng Tao Shen - [[pdf]](https://arxiv.org/pdf/2604.18612)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated strong capabilities in complex reasoning tasks, while recent prompting strategies such as Chain-of-Thought (CoT) have further elevated their performance in handling complex logical problems. Despite these advances, high-quality reasoning remains heavily reliant on manual static prompts and is sensitive to decoding configurations and task distributions, leading to performance fluctuations and limited transferability. Existing automatic prompt optimization methods typically adopt single-agent local search, failing to simultaneously optimize prompts and decoding hyperparameters within a unified framework to achieve stable global improvements. To address this limitation, we propose Agent-GWO, a dynamic prompt optimization framework for complex reasoning. Specifically, we unify prompt templates and decoding hyperparameters as inheritable agent configurations. By leveraging the leader-follower mechanism of the Grey Wolf Optimizer (GWO), we automatically select three leader agents ($\alpha$, $\beta$, and $\delta$) to guide the collaborative updates of the remaining agents, enabling iterative convergence toward robust optimal reasoning configurations that can be seamlessly integrated for inference. Extensive experiments on multiple mathematical and hybrid reasoning benchmarks across diverse LLM backbones show that Agent-GWO consistently improves accuracy and stability over existing prompt optimization methods. The code will be released publicly.

**arXiv ID:** 2604.18612
</details>

<details>
<summary><strong>Refute-or-Promote: An Adversarial Stage-Gated Multi-Agent Review Methodology for High-Precision LLM-Assisted Defect Discovery</strong> - Abhinav Agarwal - [[pdf]](https://arxiv.org/pdf/2604.19049)</summary>

**Abstract:** LLM-assisted defect discovery has a precision crisis: plausible-but-wrong reports overwhelm maintainers and degrade credibility for real findings. We present Refute-or-Promote, an inference-time reliability pattern combining Stratified Context Hunting (SCH) for candidate generation, adversarial kill mandates, context asymmetry, and a Cross-Model Critic (CMC). Adversarial agents attempt to disprove candidates at each promotion gate; cold-start reviewers are intended to reduce anchoring cascades; cross-family review can catch correlated blind spots that same-family review misses. Over a 31-day campaign across 7 targets (security libraries, the ISO C++ standard, major compilers), the pipeline killed roughly 79% of 171 candidates before advancing to disclosure (retrospective aggregate); on a consolidated-protocol subset (lcms2, wolfSSL; n=30), the prospective kill rate was 83%. Outcomes: 4 CVEs (3 public, 1 embargoed); LWG 4549 accepted to the C++ working paper; 5 merged C++ editorial PRs; 3 compiler conformance bugs; 8 merged security-related fixes without CVE; an RFC 9000 errata filed under committee review; and 1+ FIPS 140-3 normative compliance issues under coordinated disclosure -- all evaluated by external acceptance, not benchmarks. The most instructive failure: ten dedicated reviewers unanimously endorsed a non-existent Bleichenbacher padding oracle in OpenSSL's CMS module; it was killed only by a single empirical test, motivating the mandatory empirical gate. No vulnerability was discovered autonomously; the contribution is external structure that filters LLM agents' persistent false positives. As a preliminary transfer test beyond defect discovery, a simplified cross-family critique variant also solved five previously unsolved SymPy instances on SWE-bench Verified and one SWE-rebench hard task.

**arXiv ID:** 2604.19049
</details>

<details>
<summary><strong>Rethinking Scale: Deployment Trade-offs of Small Language Models under Agent Paradigms</strong> - Xinlin Wang, Mats Brorsson - [[pdf]](https://arxiv.org/pdf/2604.19299)</summary>

**Abstract:** Despite the impressive capabilities of large language models, their substantial computational costs, latency, and privacy risks hinder their widespread deployment in real-world applications. Small Language Models (SLMs) with fewer than 10 billion parameters present a promising alternative; however, their inherent limitations in knowledge and reasoning curtail their effectiveness. Existing research primarily focuses on enhancing SLMs through scaling laws or fine-tuning strategies while overlooking the potential of using agent paradigms, such as tool use and multi-agent collaboration, to systematically compensate for the inherent weaknesses of small models. To address this gap, this paper presents the first large-scale, comprehensive study of <10B open-source models under three paradigms: (1) the base model, (2) a single agent equipped with tools, and (3) a multi-agent system with collaborative capabilities. Our results show that single-agent systems achieve the best balance between performance and cost, while multi-agent setups add overhead with limited gains. Our findings highlight the importance of agent-centric design for efficient and trustworthy deployment in resource-constrained settings.

**arXiv ID:** 2604.19299
</details>

<details>
<summary><strong>M$^{2}$GRPO: Mamba-based Multi-Agent Group Relative Policy Optimization for Biomimetic Underwater Robots Pursuit</strong> - Yukai Feng, Zhiheng Wu, Zhengxing Wu, Junwen Gu, Junzhi Yu - [[pdf]](https://arxiv.org/pdf/2604.19404)</summary>

**Abstract:** Traditional policy learning methods in cooperative pursuit face fundamental challenges in biomimetic underwater robots, where long-horizon decision making, partial observability, and inter-robot coordination require both expressiveness and stability. To address these issues, a novel framework called Mamba-based multi-agent group relative policy optimization (M$^{2}$GRPO) is proposed, which integrates a selective state-space Mamba policy with group-relative policy optimization under the centralized-training and decentralized-execution (CTDE) paradigm. Specifically, the Mamba-based policy leverages observation history to capture long-horizon temporal dependencies and exploits attention-based relational features to encode inter-agent interactions, producing bounded continuous actions through normalized Gaussian sampling. To further improve credit assignment without sacrificing stability, the group-relative advantages are obtained by normalizing rewards across agents within each episode and optimized through a multi-agent extension of GRPO, significantly reducing the demand for training resources while enabling stable and scalable policy updates. Extensive simulations and real-world pool experiments across team scales and evader strategies demonstrate that M$^{2}$GRPO consistently outperforms MAPPO and recurrent baselines in both pursuit success rate and capture efficiency. Overall, the proposed framework provides a practical and scalable solution for cooperative underwater pursuit with biomimetic robot systems.

**arXiv ID:** 2604.19404
</details>

<details>
<summary><strong>Mesh Memory Protocol: Semantic Infrastructure for Multi-Agent LLM Systems</strong> - Hongwei Xu - [[pdf]](https://arxiv.org/pdf/2604.19540)</summary>

**Abstract:** Teams of LLM agents increasingly collaborate on tasks spanning days or weeks: multi-day data-generation sprints where generator, reviewer, and auditor agents coordinate in real time on overlapping batches; specialists carrying findings forward across session restarts; product decisions compounding over many review rounds. This requires agents to share, evaluate, and combine each other's cognitive state in real time across sessions. We call this cross-session agent-to-agent cognitive collaboration, distinct from parallel agent execution. To enable it, three problems must be solved together. (P1) Each agent decides field by field what to accept from peers, not accept or reject whole messages. (P2) Every claim is traceable to source, so returning claims are recognised as echoes of the receiver's own prior thinking. (P3) Memory that survives session restarts is relevant because of how it was stored, not how it is retrieved. These are protocol-level properties at the semantic layer of agent communication, distinct from tool-access and task-delegation protocols at lower layers. We call this missing protocol layer "semantic infrastructure," and the Mesh Memory Protocol (MMP) specifies it. Four composable primitives work together: CAT7, a fixed seven-field schema for every Cognitive Memory Block (CMB); SVAF, which evaluates each field against the receiver's role-indexed anchors and realises P1; inter-agent lineage, carried as parents and ancestors of content-hash keys and realising P2; and remix, which stores only the receiver's own role-evaluated understanding of each accepted CMB, never the raw peer signal, realising P3. MMP is specified, shipped, and running in production across three reference deployments, where each session runs an autonomous agent as a mesh peer with its own identity and memory, collaborating with other agents across the network for collective intelligence.

**arXiv ID:** 2604.19540
</details>

<details>
<summary><strong>Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment</strong> - Bobo Li, Rui Wu, Zibo Ji, Meishan Zhang, Hao Fei, Min Zhang, Mong-Li Lee, Wynne Hsu - [[pdf]](https://arxiv.org/pdf/2604.19548)</summary>

**Abstract:** Large Language Model agents have rapidly evolved from static text generators into dynamic systems capable of executing complex autonomous workflows. To enhance reliability, multi-agent frameworks assigning specialized roles are increasingly adopted to enable self-reflection and mutual auditing. While such role-playing effectively leverages domain expert knowledge, we find it simultaneously induces a human-like cognitive bias known as Actor-Observer Asymmetry (AOA). Specifically, an agent acting as an actor (during self-reflection) tends to attribute failures to external factors, whereas an observer (during mutual auditing) attributes the same errors to internal faults. We quantify this using our new Ambiguous Failure Benchmark, which reveals that simply swapping perspectives triggers the AOA effect in over 20% of cases for most models. To tame this bias, we introduce ReTAS (Reasoning via Thesis-Antithesis-Synthesis), a model trained through dialectical alignment to enforce perspective-invariant reasoning. By integrating dialectical chain-of-thought with Group Relative Policy Optimization, ReTAS guides agents to synthesize conflicting viewpoints into an objective consensus. Experiments demonstrate that ReTAS effectively mitigates attribution inconsistency and significantly improves fault resolution rates in ambiguous scenarios.

**arXiv ID:** 2604.19548
</details>

<details>
<summary><strong>Autogenesis: A Self-Evolving Agent Protocol</strong> - Wentao Zhang, Zhe Zhao, Haibin Wen, Yingcheng Wu, Ming Yin, Bo An, Mengdi Wang - [[pdf]](https://arxiv.org/pdf/2604.15034)</summary>

**Abstract:** Recent advances in LLM based agent systems have shown promise in tackling complex, long horizon tasks. However, existing agent protocols (e.g., A2A and MCP) under specify cross entity lifecycle and context management, version tracking, and evolution safe update interfaces, which encourages monolithic compositions and brittle glue code. We introduce Autogenesis Protocol (AGP), a self evolution protocol that decouples what evolves from how evolution occurs. Its Resource Substrate Protocol Layer (RSPL) models prompts, agents, tools, environments, and memory as protocol registered resources with explicit state, lifecycle, and versioned interfaces. Its Self Evolution Protocol Layer (SEPL) specifies a closed loop operator interface for proposing, assessing, and committing improvements with auditable lineage and rollback. Building on AGP, we present Autogenesis System (AGS), a self-evolving multi-agent system that dynamically instantiates, retrieves, and refines protocol-registered resources during execution. We evaluate AGS on multiple challenging benchmarks that require long horizon planning and tool use across heterogeneous resources. The results demonstrate consistent improvements over strong baselines, supporting the effectiveness of agent resource management and closed loop self evolution. The code is available at this https URL.

**arXiv ID:** 2604.15034
</details>

<details>
<summary><strong>AgentDynEx: Nudging the Mechanics and Dynamics of Multi-Agent Simulations</strong> - Jenny Ma, Riya Sahni, Karthik Sreedhar, Lydia B. Chilton - [[pdf]](https://arxiv.org/pdf/2504.09662)</summary>

**Abstract:** Multi-agent large language model simulations have the potential to model complex human behaviors and interactions. If the mechanics are set up properly, unanticipated and valuable social dynamics can surface. However, it is challenging to consistently enforce simulation mechanics while still allowing for notable and emergent dynamics. We present AgentDynEx, an AI system that helps set up simulations from user-specified mechanics and dynamics. AgentDynEx uses LLMs to guide users through a Configuration Matrix to identify core mechanics and define milestones to track dynamics. It also introduces a method called \textit{nudging}, where the system dynamically reflects on simulation progress and gently intervenes if it begins to deviate from intended outcomes. A technical evaluation found that nudging enables simulations to have more complex mechanics and maintain its notable dynamics compared to simulations without nudging. We discuss the importance of nudging as a technique for balancing mechanics and dynamics of multi-agent simulations.

**arXiv ID:** 2504.09662
</details>

<details>
<summary><strong>OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration</strong> - Shijun Li, Hilaf Hasson, Joydeep Ghosh - [[pdf]](https://arxiv.org/pdf/2505.11765)</summary>

**Abstract:** Agents powered by advanced large language models (LLMs) have demonstrated impressive capabilities across diverse complex applications. Recently, Multi-Agent Systems (MAS), wherein multiple agents collaborate and communicate with each other, have exhibited enhanced capabilities in complex tasks, such as high-quality code generation and arithmetic reasoning. However, the development of such systems often relies on handcrafted methods, and the literature on systematic design and optimization of LLM-based MAS remains limited. In this work, we introduce OMAC, a general framework designed for holistic optimization of LLM-based MAS. Specifically, we identify five key optimization dimensions for MAS, encompassing both agent functionality and collaboration structure. Building upon these dimensions, we first propose a general algorithm, utilizing two actors termed the Semantic Initializer and the Contrastive Comparator, to optimize any single dimension. Then, we present an algorithm for joint optimization across multiple dimensions. Extensive experiments demonstrate the superior performance of OMAC on code generation, arithmetic reasoning, and general reasoning tasks against state-of-the-art approaches.

**arXiv ID:** 2505.11765
</details>

<details>
<summary><strong>Superficial Success vs. Internal Breakdown: An Empirical Study of Generalization in Adaptive Multi-Agent Systems</strong> - Namyoung So, Seokgyu Jang, Taeuk Kim - [[pdf]](https://arxiv.org/pdf/2604.18951)</summary>

**Abstract:** Adaptive multi-agent systems (MAS) are increasingly adopted to tackle complex this http URL, the narrow task coverage of their optimization raises the question of whether they can function as general-purpose this http URL address this gap, we conduct an extensive empirical study of adaptive MAS, revealing two key findings: (1) topological overfitting -- they fail to generalize across different domains; and (2) illusory coordination -- they achieve reasonable surface-level accuracy while the underlying agent interactions diverge from ideal MAS behavior, raising concerns about their practical this http URL findings highlight the pressing need to prioritize generalization in MAS development and motivate evaluation protocols that extend beyond simple final-answer correctness.

**arXiv ID:** 2604.18951
</details>

<details>
<summary><strong>Gated Coordination for Efficient Multi-Agent Collaboration in Minecraft Game</strong> - HuaDong Jian, Chenghao Li, Haoyu Wang, Jiajia Shuai, Jinyu Guo, Yang Yang, Chaoning Zhang - [[pdf]](https://arxiv.org/pdf/2604.18975)</summary>

**Abstract:** In long-horizon open-world multi-agent systems, existing methods often treat local anomalies as automatic triggers for communication. This default design introduces coordination noise, interrupts local execution, and overuses public interaction in cases that could be resolved locally. To address this issue, we propose a partitioned information architecture for MLLM agents that explicitly separates private execution states from public coordination states. Building on this design, we introduce two key mechanisms. First, we develop an event-triggered working memory based on system-verified outcomes to maintain compact and low-noise local state representations. Second, we propose a cost-sensitive gated escalation mechanism that determines whether cross-region communication should be initiated by jointly considering node criticality, local recovery cost, and downstream task impact. In this way, communication is transformed from a default reaction into a selective decision. Experiments conducted on long-term construction tasks in open environments demonstrate that, compared to baseline models based on strong communication and planned structures, the introduction of gated communication and a partitioned information architecture results in superior performance in terms of blueprint completion quality and execution chain length. It also improves local self-recovery, reduces ineffective escalations, and increases the utility of public communication.

**arXiv ID:** 2604.18975
</details>

<details>
<summary><strong>ClawCoin: An Agentic AI-Native Cryptocurrency for Decentralized Agent Economies</strong> - Shaoyu Li, Chaoyu Zhang, Hexuan Yu, Y. Thomas Hou, Wenjing Lou - [[pdf]](https://arxiv.org/pdf/2604.19026)</summary>

**Abstract:** Autonomous AI agents live or die by the API tokens they consume: without paid inference capacity they cannot reason, act, or delegate. Compute-token cost has become the binding resource of the emerging agent economy, yet it is non-transferable: it is account-bound, vendor-specific, and absent from on-chain ledgers. Existing payment rails such as x402 move fiat-backed value between agents, but they do not represent the quantity agents actually burn. As a result, agents can transport purchasing power but cannot quote, escrow, or settle workflows in a unit aligned with compute cost.
We present ClawCoin, a tokenized, compute-cost-indexed unit of account and settlement asset for decentralized agent economies. ClawCoin combines four layers: a robust basket index over standardized prices; an oracle publishing signed fresh attestations; a NAV-based mint/redeem vault with coverage thresholds and rate limits; and an on-chain settlement layer for multi-hop delegations.
We implement a prototype on an Ethereum-compatible L2 and evaluate it using a multi-agent simulator and the OpenClaw testbed. Across single-agent, multi-agent, workflow, and procurement experiments, ClawCoin stabilizes execution capacity under cost shocks, reduces cross-agent quote dispersion, eliminates partial settlements, and sustains cooperative market dynamics that fiat-denominated baselines cannot. These results suggest that compute-indexed units of account can improve decentralized agent coordination.

**arXiv ID:** 2604.19026
</details>

<details>
<summary><strong>TeamFusion: Supporting Open-ended Teamwork with Multi-Agent Systems</strong> - Jiale Liu, Victor S. Bursztyn, Lin Ai, Haoliang Wang, Sunav Choudhary, Saayan Mitra, Qingyun Wu - [[pdf]](https://arxiv.org/pdf/2604.19589)</summary>

**Abstract:** In open-ended domains, teams must reconcile diverse viewpoints to produce strong deliverables. Answer aggregation approaches commonly used in closed domains are ill-suited to this setting, as they tend to suppress minority perspectives rather than resolve underlying disagreements. We present TeamFusion, a multi-agent system designed to support teamwork in open-ended domains by: 1. Instantiating a proxy agent for each team member conditioned on their expressed preferences; 2. Conducting a structured discussion to surface agreements and disagreements; and 3. Synthesizing more consensus-oriented deliverables that feed into new iterations of discussion and refinement. We evaluate TeamFusion on two teamwork tasks where team members can assess how well their individual views are represented in team decisions and how consensually strong the final deliverables are, finding that it outperforms direct aggregation baselines across metrics, tasks, and team configurations.

**arXiv ID:** 2604.19589
</details>

<details>
<summary><strong>Cost-Aware Distributed Online Learning with Strict Rejection Behavior against Adversarial Agents</strong> - Yuhan Suo, Senchun Chai, Xudong Zhao, Yuanqing Xia, and Runqi Chai - [[pdf]](https://arxiv.org/pdf/2412.01524)</summary>

**Abstract:** Distributed online learning in multi-agent systems is highly vulnerable to adversarial influence, especially when malicious agents cannot be fully isolated during the transient stage. While existing studies mainly pursue resilient consensus or secure fusion, they pay much less attention to the learning inefficiency and extra evolution cost accumulated during the defense process. This paper addresses this gap by developing a cost-aware distributed online learning framework with strict rejection behavior against adversarial this http URL this mechanism, the state evolution cost of online adaptation is formulated and the cost amplification effect caused by adversarial interactions is theoretically characterized. To balance robustness, convergence efficiency, and long-term cost, we propose an adaptive adjustment mechanism for the state-evolution rate. The resulting outer-layer update can be equivalently viewed as a constrained online optimization problem. We further establish the well-posedness and regularity of the associated periodic Riccati layer, and show that the outer-layer update ensures feasibility and controlled variation. Based on these properties, closed-loop practical stability is rigorously established via a two-time-scale Lyapunov framework. Simulations demonstrate that the proposed method achieves robust and low-cost convergence under adversarial disturbances. Furthermore, a multi-satellite target tracking scenario with malicious interference further demonstrates the practical effectiveness of the strict rejection behavior.

**arXiv ID:** 2412.01524
</details>

<details>
<summary><strong>Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation</strong> - Nuo Chen, Yicheng Tong, Yuzhe Yang, Yufei He, Xueyi Zhang, Qingyun Zou, Qian Wang, Bingsheng He - [[pdf]](https://arxiv.org/pdf/2604.18005)</summary>

**Abstract:** Multi-agent systems (MAS) are increasingly used for open-ended idea generation, driven by the expectation that collective interaction will broaden the exploration diversity. However, when and why such collaboration truly expands the solution space remains unclear. We present a systematic empirical study of diversity in MAS-based ideation across three bottom-up levels: model intelligence, agent cognition, and system dynamics. At the model level, we identify a compute efficiency paradox, where stronger, highly aligned models yield diminishing marginal diversity despite higher per-sample quality. At the cognition level, authority-driven dynamics suppress semantic diversity compared to junior-dominated groups. At the system level, group-size scaling yields diminishing returns and dense communication topologies accelerate premature convergence. We characterize these outcomes as collective failures emerging from structural coupling, a process where interaction inadvertently contracts agent exploration and triggers diversity collapse. Our analysis shows that this collapse arises primarily from the interaction structure rather than inherent model insufficiency, highlighting the importance of preserving independence and disagreement when designing MAS for creative tasks. Our code is available at this https URL.

**arXiv ID:** 2604.18005
</details>

<details>
<summary><strong>Multi-agent Adaptive Mechanism Design</strong> - Qiushi Han, David Simchi-Levi, Renfei Tan, Zishuo Zhao - [[pdf]](https://arxiv.org/pdf/2512.21794)</summary>

**Abstract:** We study a sequential mechanism design problem in which a principal seeks to elicit truthful reports from multiple rational agents while starting with no prior knowledge of agents' beliefs. We introduce Distributionally Robust Adaptive Mechanism (DRAM), a general framework combining insights from both mechanism design and online learning to jointly address truthfulness and cost-optimality. Throughout the sequential game, the mechanism estimates agents' beliefs and iteratively updates a distributionally robust linear program with shrinking ambiguity sets to reduce payments while preserving truthfulness. Our mechanism guarantees truthful reporting with high probability while achieving $\tilde{O}(\sqrt{T})$ cumulative regret, and we establish a matching lower bound showing that no feasible adaptive mechanism can asymptotically do better. The framework generalizes to plug-in estimators, supporting structured priors and delayed feedback. To our knowledge, this is the first adaptive mechanism under general settings that maintains truthfulness and achieves optimal regret when incentive constraints are unknown and must be learned.

**arXiv ID:** 2512.21794
</details>

<details>
<summary><strong>Mango: Multi-Agent Web Navigation via Global-View Optimization</strong> - Weixi Tong, Yifeng Di, Tianyi Zhang - [[pdf]](https://arxiv.org/pdf/2604.18779)</summary>

**Abstract:** Existing web agents typically initiate exploration from the root URL, which is inefficient for complex websites with deep hierarchical structures. Without a global view of the website's structure, agents frequently fall into navigation traps, explore irrelevant branches, or fail to reach target information within a limited budget. We propose Mango, a multi-agent web navigation method that leverages the website structure to dynamically determine optimal starting points. We formulate URL selection as a multi-armed bandit problem and employ Thompson Sampling to adaptively allocate the navigation budget across candidate URLs. Furthermore, we introduce an episodic memory component to store navigation history, enabling the agent to learn from previous attempts. Experiments on WebVoyager demonstrate that Mango achieves a success rate of 63.6% when using GPT-5-mini, outperforming the best baseline by 7.3%. Furthermore, on WebWalkerQA, Mango attains a 52.5% success rate, surpassing the best baseline by 26.8%. We also demonstrate the generalizability of Mango using both open-source and closed-source models as backbones. Our data and code are open-source and available at this https URL.

**arXiv ID:** 2604.18779
</details>

<details>
<summary><strong>Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection</strong> - Yixuan Tang, Yirui Zhang, Hang Feng, Anthony K.H. Tung - [[pdf]](https://arxiv.org/pdf/2604.19005)</summary>

**Abstract:** Half-truths, claims that are factually correct yet misleading due to omitted context, remain a blind spot for fact verification systems focused on explicit falsehoods. Addressing such omission-based manipulation requires reasoning not only about what is said, but also about what is left unsaid. We propose RADAR, a role-anchored multi-agent debate framework for omission-aware fact verification under realistic, noisy retrieval. RADAR assigns complementary roles to a Politician and a Scientist, who reason adversarially over shared retrieved evidence, moderated by a neutral Judge. A dual-threshold early termination controller adaptively decides when sufficient reasoning has been reached to issue a verdict. Experiments show that RADAR consistently outperforms strong single- and multi-agent baselines across datasets and backbones, improving omission detection accuracy while reducing reasoning cost. These results demonstrate that role-anchored, retrieval-grounded debate with adaptive control is an effective and scalable framework for uncovering missing context in fact verification. The code is available at this https URL.

**arXiv ID:** 2604.19005
</details>

<details>
<summary><strong>Rethinking Information Synthesis in Multimodal Question Answering A Multi-Agent Perspective</strong> - Krishna Singh Rajput, Tejas Anvekar, Chitta Baral, Vivek Gupta - [[pdf]](https://arxiv.org/pdf/2505.20816)</summary>

**Abstract:** Recent advances in multimodal question answering have primarily focused on combining heterogeneous modalities or fine-tuning multimodal large language models. While these approaches have shown strong performance, they often rely on a single, generalized reasoning strategy, overlooking the unique characteristics of each modality ultimately limiting both accuracy and interpretability. To address these limitations, we propose MAMMQA, a multi-agent QA framework for multimodal inputs spanning text, tables, and images. Our system includes two Visual Language Model (VLM) agents and one text-based Large Language Model (LLM) agent. The first VLM decomposes the user query into sub-questions and sequentially retrieves partial answers from each modality. The second VLM synthesizes and refines these results through cross-modal reasoning. Finally, the LLM integrates the insights into a cohesive answer. This modular design enhances interpretability by making the reasoning process transparent and allows each agent to operate within its domain of expertise. Experiments on diverse multimodal QA benchmarks demonstrate that our cooperative, multi-agent framework consistently outperforms existing baselines in both accuracy and robustness.

**arXiv ID:** 2505.20816
</details>

<details>
<summary><strong>The Alignment Waltz: Jointly Training Agents to Collaborate for Safety</strong> - Jingyu Zhang, Haozhu Wang, Eric Michael Smith, Sid Wang, Amr Sharaf, Mahesh Pasupuleti, Benjamin Van Durme, Daniel Khashabi, Jason Weston, Hongyuan Zhan - [[pdf]](https://arxiv.org/pdf/2510.08240)</summary>

**Abstract:** Harnessing the power of LLMs requires a delicate dance between being helpful and harmless. This creates a fundamental tension between two competing challenges: vulnerability to adversarial attacks that elicit unsafe content, and a tendency for overrefusal on benign but sensitive prompts. Current approaches often navigate this dance with safeguard models that completely reject any content that contains unsafe portions. This approach cuts the music entirely-it may exacerbate overrefusals and fails to provide nuanced guidance for queries it refuses. To teach models a more coordinated choreography, we propose WaltzRL, a novel multi-agent reinforcement learning framework that formulates safety alignment as a collaborative, positive-sum game. WaltzRL jointly trains a conversation agent and a feedback agent, where the latter is incentivized to provide useful suggestions that improve the safety and helpfulness of the conversation agent's responses. At the core of WaltzRL is a Dynamic Improvement Reward (DIR) that evolves over time based on how well the conversation agent incorporates the feedback. At inference time, unsafe or overrefusing responses from the conversation agent are improved rather than discarded. The feedback agent is deployed together with the conversation agent and only engages adaptively when needed, preserving helpfulness and low latency on safe queries. Our experiments, conducted across five diverse datasets, demonstrate that WaltzRL significantly reduces both unsafe responses (e.g., from 39.0% to 4.6% on WildJailbreak) and overrefusals (from 45.3% to 9.9% on OR-Bench) compared to various baselines. By enabling the conversation and feedback agents to co-evolve and adaptively apply feedback, WaltzRL enhances LLM safety without degrading general capabilities, thereby advancing the Pareto front between helpfulness and harmlessness.

**arXiv ID:** 2510.08240
</details>

<details>
<summary><strong>STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning</strong> - Juntong Ni, Shiyu Wang, Qi He, Ming Jin, Wei Jin - [[pdf]](https://arxiv.org/pdf/2601.03248)</summary>

**Abstract:** Spatio-temporal reasoning in time series involves the explicit synthesis of temporal dynamics, spatial dependencies, and textual context. This capability is vital for high-stakes decision-making in systems such as traffic networks, power grids, and disease propagation. However, the field remains underdeveloped because most existing works prioritize predictive accuracy over reasoning. To address the gap, we introduce ST-Bench, a benchmark consisting of four core tasks, including etiological reasoning, entity identification, correlation reasoning, and in-context forecasting, developed via a network SDE-based multi-agent data synthesis pipeline. We then propose STReasoner, which empowers LLM to integrate time series, graph structure, and text for explicit reasoning. To promote spatially grounded logic, we introduce S-GRPO, a reinforcement learning algorithm that rewards performance gains specifically attributable to spatial information. Experiments show that STReasoner achieves average accuracy gains between 17% and 135% at only 0.004X the cost of proprietary models and generalizes robustly to real-world data.

**arXiv ID:** 2601.03248
</details>

<details>
<summary><strong>LSTM-MAS: A Long Short-Term Memory Inspired Multi-Agent System for Long-Context Understanding</strong> - Yichen Jiang, Jiakang Yuan, Chongjun Tu, Peng Ye, Tao Chen - [[pdf]](https://arxiv.org/pdf/2601.11913)</summary>

**Abstract:** Effectively processing long contexts remains a fundamental yet unsolved challenge for large language models (LLMs). Existing single-LLM-based methods primarily reduce the context window or optimize the attention mechanism, but they often encounter additional computational costs or constrained expanded context length. While multi-agent-based frameworks can mitigate these limitations, they remain susceptible to the accumulation of errors and the propagation of hallucinations. In this work, we draw inspiration from the Long Short-Term Memory (LSTM) architecture to design a Multi-Agent System called LSTM-MAS, emulating LSTM's hierarchical information flow and gated memory mechanisms for long-context understanding. Specifically, LSTM-MAS organizes agents in a chained architecture, where each node comprises a worker agent for segment-level comprehension, a filter agent for redundancy reduction, a judge agent for continuous error detection, and a manager agent for globally regulates information propagation and retention, analogous to LSTM and its input gate, forget gate, constant error carousel unit, and output gate. These novel designs enable controlled information transfer and selective long-term dependency modeling across textual segments, which can effectively avoid error accumulation and hallucination propagation. We conducted an extensive evaluation of our method. Compared with the previous best multi-agent approach, CoA, our model achieves improvements of 97.97%, 65.75%, 122.19%, 39.61% and 10.80% on Narrative QA, Qasper, HotpotQA, 2WikiMQA and MuSiQue, respectively.

**arXiv ID:** 2601.11913
</details>

<details>
<summary><strong>MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering</strong> - Sieun Hyeon, Jusang Oh, Sunghwan Steve Cho, Jaeyoung Do - [[pdf]](https://arxiv.org/pdf/2602.09642)</summary>

**Abstract:** Recent advances in Large Language Models (LLMs) have significantly improved table understanding tasks such as Table Question Answering (TableQA), yet challenges remain in ensuring reliability, scalability, and efficiency, especially in resource-constrained or privacy-sensitive environments. In this paper, we introduce MATA, a multi-agent TableQA framework that leverages multiple complementary reasoning paths and a set of tools built with small language models. MATA generates candidate answers through diverse reasoning styles for a given table and question, then refines or selects the optimal answer with the help of these tools. Furthermore, it incorporates an algorithm designed to minimize expensive LLM agent calls, enhancing overall efficiency. MATA maintains strong performance with small, open-source models and adapts easily across various LLM types. Extensive experiments on two benchmarks of varying difficulty with ten different LLMs demonstrate that MATA achieves state-of-the-art accuracy and highly efficient reasoning while avoiding excessive LLM inference. Our results highlight that careful orchestration of multiple reasoning pathways yields scalable and reliable TableQA. The code is available at this https URL.

**arXiv ID:** 2602.09642
</details>

<details>
<summary><strong>Council Mode: Mitigating Hallucination and Bias in LLMs via Multi-Agent Consensus</strong> - Shuai Wu, Xue Li, Yanna Feng, Yufang Li, Zhijun Wang - [[pdf]](https://arxiv.org/pdf/2604.02923)</summary>

**Abstract:** Large Language Models (LLMs), particularly those employing Mixture-of-Experts (MoE) architectures, have achieved remarkable capabilities across diverse natural language processing tasks. However, these models frequently suffer from hallucinations -- generating plausible but factually incorrect content -- and exhibit systematic biases that are amplified by uneven expert activation during inference. In this paper, we propose the Council Mode, a novel multi-agent consensus framework that addresses these limitations by dispatching queries to multiple heterogeneous frontier LLMs in parallel and synthesizing their outputs through a dedicated consensus model. The Council pipeline operates in three phases: (1) an intelligent triage classifier that routes queries based on complexity, (2) parallel expert generation across architecturally diverse models, and (3) a structured consensus synthesis that explicitly identifies agreement, disagreement, and unique findings before producing the final response. We implement and evaluate this architecture within an open-source AI workspace. Our comprehensive evaluation across multiple benchmarks demonstrates that the Council Mode achieves a 35.9% relative reduction in hallucination rates on the HaluEval benchmark and a 7.8-point improvement on TruthfulQA compared to the best-performing individual model, while maintaining significantly lower bias variance across domains. We provide the mathematical formulation of the consensus mechanism, detail the system architecture, and present extensive empirical results with ablation studies.

**arXiv ID:** 2604.02923
</details>

<details>
<summary><strong>MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation</strong> - Xingchen Xiao, Heyan Huang, Runheng Liu, Jincheng Xie - [[pdf]](https://arxiv.org/pdf/2604.18509)</summary>

**Abstract:** Large language models (LLMs) are widely used in retrieval-augmented generation (RAG) to incorporate external knowledge at inference time. However, when retrieved contexts are noisy, incomplete, or heterogeneous, a single generation process often struggles to reconcile evidence effectively. We propose \textbf{MASS-RAG}, a multi-agent synthesis approach to retrieval-augmented generation that structures evidence processing into multiple role-specialized agents. MASS-RAG applies distinct agents for evidence summarization, evidence extraction, and reasoning over retrieved documents, and combines their outputs through a dedicated synthesis stage to produce the final answer. This design exposes multiple intermediate evidence views, allowing the model to compare and integrate complementary information before answer generation. Experiments on four benchmarks show that MASS-RAG consistently improves performance over strong RAG baselines, particularly in settings where relevant evidence is distributed across retrieved contexts.

**arXiv ID:** 2604.18509
</details>

<details>
<summary><strong>SynAgent: Generalizable Cooperative Humanoid Manipulation via Solo-to-Cooperative Agent Synergy</strong> - Wei Yao, Haohan Ma, Hongwen Zhang, Yunlian Sun, Liangjun Xing, Zhile Yang, Yuanjun Guo, Yebin Liu, Jinhui Tang - [[pdf]](https://arxiv.org/pdf/2604.18557)</summary>

**Abstract:** Controllable cooperative humanoid manipulation is a fundamental yet challenging problem for embodied intelligence, due to severe data scarcity, complexities in multi-agent coordination, and limited generalization across objects. In this paper, we present SynAgent, a unified framework that enables scalable and physically plausible cooperative manipulation by leveraging Solo-to-Cooperative Agent Synergy to transfer skills from single-agent human-object interaction to multi-agent human-object-human scenarios. To maintain semantic integrity during motion transfer, we introduce an interaction-preserving retargeting method based on an Interact Mesh constructed via Delaunay tetrahedralization, which faithfully maintains spatial relationships among humans and objects. Building upon this refined data, we propose a single-agent pretraining and adaptation paradigm that bootstraps synergistic collaborative behaviors from abundant single-human data through decentralized training and multi-agent PPO. Finally, we develop a trajectory-conditioned generative policy using a conditional VAE, trained via multi-teacher distillation from motion imitation priors to achieve stable and controllable object-level trajectory execution. Extensive experiments demonstrate that SynAgent significantly outperforms existing baselines in both cooperative imitation and trajectory-conditioned control, while generalizing across diverse object geometries. Codes and data will be available after publication. Project Page: this http URL

**arXiv ID:** 2604.18557
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>ClawNet: Human-Symbiotic Agent Network for Cross-User Autonomous Cooperation</strong> - Zhiqin Yang, Zhenyuan Zhang, Xianzhang Jia, Jun Song, Wei Xue, Yonggang Zhang, Yike Guo - [[pdf]](https://arxiv.org/pdf/2604.19211)</summary>

**Abstract:** Current AI agent frameworks have made remarkable progress in automating individual tasks, yet all existing systems serve a single user. Human productivity rests on the social and organizational relationships through which people coordinate, negotiate, and delegate. When agents move beyond performing tasks for one person to representing that person in collaboration with others, the infrastructure for cross-user agent collaboration is entirely absent, let alone the governance mechanisms needed to secure it. We argue that the next frontier for AI agents lies not in stronger individual capability, but in the digitization of human collaborative relationships. To this end, we propose a human-symbiotic agent paradigm. Each user owns a permanently bound agent system that collaborates on the owner's behalf, forming a network whose nodes are humans rather than agents. This paradigm rests on three governance primitives. A layered identity architecture separates a Manager Agent from multiple context-specific Identity Agents; the Manager Agent holds global knowledge but is architecturally isolated from external communication. Scoped authorization enforces per-identity access control and escalates boundary violations to the owner. Action-level accountability logs every operation against its owner's identity and authorization, ensuring full auditability. We instantiate this paradigm in ClawNet, an identity-governed agent collaboration framework that enforces identity binding and authorization verification through a central orchestrator, enabling multiple users to collaborate securely through their respective agents.

**arXiv ID:** 2604.19211
</details>

<details>
<summary><strong>Temporal UI State Inconsistency in Desktop GUI Agents: Formalizing and Defending Against TOCTOU Attacks on Computer-Use Agents</strong> - Wenpeng Xu - [[pdf]](https://arxiv.org/pdf/2604.18860)</summary>

**Abstract:** GUI agents that control desktop computers via screenshot-and-click loops introduce a new class of vulnerability: the observation-to-action gap (mean 6.51 s on real OSWorld workloads) creates a Time-Of-Check, Time-Of-Use (TOCTOU) window during which an unprivileged attacker can manipulate the UI state. We formalize this as a Visual Atomicity Violation and characterize three concrete attack primitives: (A) Notification Overlay Hijack, (B) Window Focus Manipulation, and (C) Web DOM Injection. Primitive B, the closest desktop analog to Android Action Rebinding, achieves 100% action-redirection success rate with zero visual evidence at the observation time. We propose Pre-execution UI State Verification (PUSV), a lightweight three-layer defense that re-verifies the UI state immediately before each action dispatch: masked pixel SSIM at the click target (L1), global screenshot diff (L2a), and X Window snapshot diff (L2b). PUSV achieves 100% Action Interception Rate across 180 adversarial trials (135 Primitive A + 45 Primitive B) with zero false positives and < 0.1 s overhead. Against Primitive C (zero-visual-footprint DOM injection), PUSV reveals a structural blind spot (~0% AIR), motivating future OS+DOM defense-in-depth architectures. No single PUSV layer alone achieves full coverage; different primitives require different detection signals, validating the layered design.

**arXiv ID:** 2604.18860
</details>

<details>
<summary><strong>Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems: A Neurosymbolic Architecture for Domain-Grounded AI Agents</strong> - Thanh Luong Tuan - [[pdf]](https://arxiv.org/pdf/2604.00555)</summary>

**Abstract:** Enterprise adoption of Large Language Models (LLMs) is constrained by hallucination, domain drift, and the inability to enforce regulatory compliance at the reasoning level. We present a neurosymbolic architecture implemented within the Foundation AgenticOS (FAOS) platform that addresses these limitations through ontology-constrained neural reasoning. Our approach introduces a three-layer ontological framework--Role, Domain, and Interaction ontologies--that provides formal semantic grounding for LLM-based enterprise agents. We formalize the concept of asymmetric neurosymbolic coupling, wherein symbolic ontological knowledge constrains agent inputs (context assembly, tool discovery, governance thresholds) while proposing mechanisms for extending this coupling to constrain agent outputs (response validation, reasoning verification, compliance checking). We evaluate the architecture through a controlled experiment (600 runs across five industries: FinTech, Insurance, Healthcare, Vietnamese Banking, and Vietnamese Insurance), finding that ontology-coupled agents significantly outperform ungrounded agents on Metric Accuracy (p < .001, W = .460), Regulatory Compliance (p = .003, W = .318), and Role Consistency (p < .001, W = .614), with improvements greatest where LLM parametric knowledge is weakest--particularly in Vietnam-localized domains. Our contributions include: (1) a formal three-layer enterprise ontology model, (2) a taxonomy of neurosymbolic coupling patterns, (3) ontology-constrained tool discovery via SQL-pushdown scoring, (4) a proposed framework for output-side ontological validation, (5) empirical evidence for the inverse parametric knowledge effect that ontological grounding value is inversely proportional to LLM training data coverage of the domain, and (6) a production system serving 21 industry verticals with 650+ agents.

**arXiv ID:** 2604.00555
</details>

<details>
<summary><strong>HadAgent: Harness-Aware Decentralized Agentic AI Serving with Proof-of-Inference Blockchain Consensus</strong> - Landy Jimenez, Mariah Weatherspoon, Bingyu Shen, Yi Sheng, Jianming Liu, Boyang Li - [[pdf]](https://arxiv.org/pdf/2604.18614)</summary>

**Abstract:** Proof-of-Work (PoW) blockchain consensus consumes vast computational resources without producing useful output, while the rapid growth of large language model (LLM) agents has created unprecedented demand for GPU computation. We present HadAgent, a decentralized agentic AI serving system that replaces hash-based mining with Proof-of-Inference (PoI), a consensus mechanism in which nodes earn block-creation rights by executing deterministic LLM inference tasks. Because verification requires only re-executing a single forward pass under identical conditions, cross-node verification operates at consensus speed. HadAgent organizes validated records into a three-lane block body with dedicated DATA, MODEL, and PROOF channels, each protected by an independent Merkle root for fine-grained tamper detection. A two-tier node architecture classifies secondary nodes as trusted or non-trusted based on historical behavior: trusted nodes serve inference results in real time through optimistic execution, while non-trusted nodes must undergo full consensus verification. A harness layer monitors node behavior through heartbeat probes, anomaly detection via deterministic recomputation, and automated trust management, creating a self-correcting feedback loop that isolates malicious or unreliable participants. Experiments on a prototype implementation demonstrate 100% detection rate and 0% false positive rate for tampered records, sub-millisecond validation latency for record and hub operations, and effective harness convergence that excludes adversarial nodes within two rounds while promoting honest nodes to trusted status within five rounds.

**arXiv ID:** 2604.18614
</details>

<details>
<summary><strong>Reflection-Driven Self-Optimization 6G Agentic AI RAN via Simulation-in-the-Loop Workflows</strong> - Yunhao Hu, Xinchen Lyu, Chenshan Ren, Keda Chen, Qimei Cui, Xiaofeng Tao - [[pdf]](https://arxiv.org/pdf/2512.20640)</summary>

**Abstract:** The escalating complexity of sixth-generation (6G) networks demands unprecedented levels of autonomy beyond the capabilities of traditional optimization-based and current AI-based resource management approaches. While agentic AI has emerged as a promising paradigm for autonomous RAN, current frameworks provide sophisticated reasoning capabilities but lack mechanisms for empirical validation and self-improvement. This article identifies simulation-in-the-loop validation as a critical enabler for truly autonomous networks, where AI agents can empirically verify decisions and learn from outcomes. We present the first reflection-driven self-optimization framework that integrates agentic AI with high-fidelity network simulation in a closed-loop architecture. Our system orchestrates four specialized agents, including scenario, solver, simulation, and reflector agents, working in concert to transform agentic AI into a self-correcting system capable of escaping local optima, recognizing implicit user intent, and adapting to dynamic network conditions. Extensive experiments validate significant performance improvements over non-agentic approaches: 17.1\% higher throughput in interference optimization, 67\% improved user QoS satisfaction through intent recognition, and 25\% reduced resource utilization during low-traffic periods while maintaining service quality.

**arXiv ID:** 2512.20640
</details>

<details>
<summary><strong>Whispers in the Machine: Confidentiality in Agentic Systems</strong> - Jonathan Evertz, Merlin Chlosta, Lea Schönherr, Thorsten Eisenhofer - [[pdf]](https://arxiv.org/pdf/2402.06922)</summary>

**Abstract:** Large language model (LLM)-based agents combine LLMs with external tools to automate tasks such as scheduling meetings, managing documents, or booking travel. While these integrations unlock powerful capabilities, they also create new and more severe attack surfaces. In particular, prompt injection attacks become far more dangerous in the agentic setting: malicious instructions embedded in connected services can misdirect the agent, providing a direct pathway for sensitive data to be exfiltrated. Yet, despite a growing number of real-world incidents, the confidentiality risks of such systems remain poorly understood. To address this gap, we provide a formalization of confidentiality in LLM-based agents. By abstracting sensitive data as a secret string, we evaluate ten agents across 20 tool scenarios and 14 attack strategies. We find that all agents are vulnerable to at least one attack, and existing defenses fail to provide reliable protection against these threats. Strikingly, we find that the tooling itself can amplify leakage risks.

**arXiv ID:** 2402.06922
</details>

<details>
<summary><strong>Task-Adaptive Admittance Control for Human-Quadrotor Cooperative Load Transportation with Dynamic Cable-Length Regulation</strong> - Shuai Li, Ton T. H. Duong, Damiano Zanotto - [[pdf]](https://arxiv.org/pdf/2604.18905)</summary>

**Abstract:** The collaboration between humans and robots is critical in many robotic applications, especially in those requiring physical human-robot interaction (pHRI). Previous research in pHRI has largely focused on robotic manipulators, employing impedance or admittance control to maintain operational safety. Conversely, research in human-quadrotor cooperative load transportation (CLT) is still in its infancy. This letter introduces a novel admittance controller designed for safe and effective human-quadrotor CLT using a quadrotor equipped with an actively-controlled winch. The proposed method accounts for the system's coupled dynamics, allowing the quadrotor and its cable to dynamically adapt to contact forces during CLT tasks, thereby enhancing responsiveness. We experimentally validated the task-adaptive capability of the controller across the entire CLT process, including in-place loading/unloading and load transporting tasks. To this end, we compared the system performances against a conventional approach, using both variable and fixed cable lengths under low- and high-stiffness conditions. Results demonstrate that the proposed method outperforms the conventional approach in terms of system responsiveness and motion smoothness, leading to improved CLT capabilities.

**arXiv ID:** 2604.18905
</details>

<details>
<summary><strong>Autonomous UAV Pipeline Near-proximity Inspection via Disturbance-Aware Predictive Visual Servoing</strong> - Wen Li, Hui Wang, Jinya Su, Cunjia Liu, Wen-Hua Chen, Shihua Li - [[pdf]](https://arxiv.org/pdf/2604.19618)</summary>

**Abstract:** Reliable pipeline inspection is critical to safe energy transportation, but is constrained by long distances, complex terrain, and risks to human inspectors. Unmanned aerial vehicles provide a flexible sensing platform, yet reliable autonomous inspection remains challenging. This paper presents an autonomous quadrotor near-proximity pipeline inspection framework for three-dimensional scenarios based on image-based visual servoing model predictive control (VMPC). A unified predictive model couples quadrotor dynamics with image feature kinematics, enabling direct image-space prediction within the control loop. To address low-rate visual updates, measurement noise, and environmental uncertainties, an extended-state Kalman filtering scheme with image feature prediction (ESKF-PRE) is developed, and the estimated lumped disturbances are incorporated into the VMPC prediction model, yielding the ESKF-PRE-VMPC framework. A terrain-adaptive velocity design is introduced to maintain the desired cruising speed while generating vertical velocity references over unknown terrain slopes without prior terrain information. The framework is validated in high-fidelity Gazebo simulations and real-world experiments. In real-world tests, the proposed method reduces RMSE by 52.63% and 75.04% in pipeline orientation and lateral deviation in the image, respectively, for straight-pipeline inspection without wind, and successfully completes both wind-disturbance and bend-pipeline tasks where baseline method fails. An open-source nano quadrotor is modified for indoor experimentation.

**arXiv ID:** 2604.19618
</details>

<details>
<summary><strong>Personalized Embodied Navigation for Portable Object Finding</strong> - Vishnu Sashank Dorbala, Bhrij Patel, Amrit Singh Bedi, Dinesh Manocha - [[pdf]](https://arxiv.org/pdf/2403.09905)</summary>

**Abstract:** Embodied navigation methods commonly operate in static environments with stationary objects. In this work, we present approaches for tackling navigation in dynamic scenarios with non-stationary targets. In an indoor environment, we assume that these objects are everyday portable items moved by human intervention. We therefore formalize the problem as a personalized habit learning problem. To learn these habits, we introduce two Transit-Aware Planning (TAP) approaches that enrich embodied navigation policies with object path information. TAP improves performance in portable object finding by rewarding agents that learn to synchronize their routes with target routes. TAPs are evaluated on Dynamic Object Maps (DOMs), a dynamic variant of node-attributed topological graphs with structured object transitions. DOMs mimic human habits to simulate realistic object routes on a graph. We test TAP agents both in simulation as well as the real-world. In the MP3D simulator, TAP improves the success of a vanilla agent by 21.1% in finding non-stationary targets, while also generalizing better from static environments by 44.5% when measured by Relative Change in Success. In the real-world, we note a similar 18.3% increase on average, in multiple transit scenarios. We present qualitative inferences of TAP-agents deployed in the real world, showing them to be especially better at providing personalized assistance by finding targets in positions that they are usually not expected to be in (a toothbrush in a workspace). We also provide details of our real-to-sim pipeline, which allows researchers to generate simulations of their own physical environments for TAP, aiming to foster research in this area.

**arXiv ID:** 2403.09905
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (28 papers)</h2></summary>

<details>
<summary><strong>Reinforcement Learning Improves LLM Accuracy and Reasoning in Disease Classification from Radiology Reports</strong> - Yishu Wei, Yi Lin, Adam Flanders, George Shih, Yifan Peng - [[pdf]](https://arxiv.org/pdf/2604.19060)</summary>

**Abstract:** Accurate disease classification from radiology reports is essential for many applications. While supervised fine-tuning (SFT) of lightweight LLMs improves accuracy, it can degrade reasoning. We propose a two-stage approach: SFT on disease labels followed by Group Relative Policy Optimization (GRPO) to refine predictions by optimizing accuracy and format without reasoning supervision. Across three radiologist-annotated datasets, SFT outperformed baselines and GRPO further improved classification and enhanced reasoning recall and comprehensiveness.

**arXiv ID:** 2604.19060
</details>

<details>
<summary><strong>Integrating Anomaly Detection into Agentic AI for Proactive Risk Management in Human Activity</strong> - Farbod Zorriassatine, Ahmad Lotfi - [[pdf]](https://arxiv.org/pdf/2604.19538)</summary>

**Abstract:** Agentic AI, with goal-directed, proactive, and autonomous decision-making capabilities, offers a compelling opportunity to address movement-related risks in human activity, including the persistent hazard of falls among elderly populations. Despite numerous approaches to fall mitigation through fall prediction and detection, existing systems have not yet functioned as universal solutions across care pathways and safety-critical environments. This is largely due to limitations in consistently handling real-world complexity, particularly poor context awareness, high false alarm rates, environmental noise, and data scarcity. We argue that fall detection and fall prediction can usefully be formulated as anomaly detection problems and more effectively addressed through an agentic AI system. More broadly, this perspective enables the early identification of subtle deviations in movement patterns associated with increased risk, whether arising from age-related decline, fatigue, or environmental factors. While technical requirements for immediate deployment are beyond the scope of this paper, we propose a conceptual framework that highlights potential value. This framework promotes a well-orchestrated approach to risk management by dynamically selecting relevant tools and integrating them into adaptive decision-making workflows, rather than relying on static configurations tailored to narrowly defined scenarios.

**arXiv ID:** 2604.19538
</details>

<details>
<summary><strong>A-MAR: Agent-based Multimodal Art Retrieval for Fine-Grained Artwork Understanding</strong> - Shuai Wang, Hongyi Zhu, Jia-Hong Huang, Yixian Shen, Chengxi Zeng, Stevan Rudinac, Monika Kackovic, Nachoem Wijnberg, Marcel Worring - [[pdf]](https://arxiv.org/pdf/2604.19689)</summary>

**Abstract:** Understanding artworks requires multi-step reasoning over visual content and cultural, historical, and stylistic context. While recent multimodal large language models show promise in artwork explanation, they rely on implicit reasoning and internalized knowl- edge, limiting interpretability and explicit evidence grounding. We propose A-MAR, an Agent-based Multimodal Art Retrieval framework that explicitly conditions retrieval on structured reasoning plans. Given an artwork and a user query, A-MAR first decomposes the task into a structured reasoning plan that specifies the goals and evidence requirements for each step. Retrieval is then conditionedon this plan, enabling targeted evidence selection and supporting step-wise, grounded explanations. To evaluate agent-based multi- modal reasoning within the art domain, we introduce ArtCoT-QA. This diagnostic benchmark features multi-step reasoning chains for diverse art-related queries, enabling a granular analysis that extends beyond simple final answer accuracy. Experiments on SemArt and Artpedia show that A-MAR consistently outperforms static, non planned retrieval and strong MLLM baselines in final explanation quality, while evaluations on ArtCoT-QA further demonstrate its advantages in evidence grounding and multi-step reasoning ability. These results highlight the importance of reasoning-conditioned retrieval for knowledge-intensive multimodal understanding and position A-MAR as a step toward interpretable, goal-driven AI systems, with particular relevance to cultural industries. The code and data are available at: this https URL.

**arXiv ID:** 2604.19689
</details>

<details>
<summary><strong>ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants</strong> - Haohui Mai, Xiaoyan Guo, Xiangyun Ding, Daifeng Li, Qiuchu Yu, Chenzhun Guo, Cong Wang, Jiacheng Zhao, Christos Kozyrakis, Binhang Yuan - [[pdf]](https://arxiv.org/pdf/2604.18616)</summary>

**Abstract:** LLM-based coding agents can generate functionally correct GPU kernels, yet their performance remains far below hand-optimized libraries on critical computations such as matrix multiplication, attention, and Mixture-of-Experts (MoE). Peak GPU performance requires coordinated reasoning over tightly coupled optimizations, including tiling, shared-memory staging, software pipelining, and instruction scheduling, while existing agents rely on sparse pass/fail feedback, leaving them unable to diagnose global constraint violations.
We present Argus, an agentic framework that addresses this through data-flow invariants: compile-time specifications encoding how data must be choreographed throughout kernel execution. Argus introduces a tile-based, Pythonic DSL exposing hardware instructions and compiler policies while hiding low-level representations. The DSL provides tag functions to propagate symbolic annotations through data and control flow, and tag assertions to enforce relational constraints at use sites. When violations occur, the compiler returns concrete counterexamples identifying the thread, data element, and program point, enabling dense, structured feedback for targeted fixes. Invariants are verified at compile time via abstract interpretation over a layout algebra and SMT solving, with zero runtime overhead. An in-context reinforcement learning planner learns to select optimizations and synthesize effective invariants, supported by a curated knowledge base of GPU optimization techniques.
We evaluate Argus on the AMD MI300X GPU across GEMM, flash attention, and MoE kernels accounting for over 90% of GPU time in LLM inference. Generated kernels achieve 99-104% of state-of-the-art hand-optimized assembly throughput and are 2-1543x faster than existing agentic systems. Argus further generalizes to 200 KernelBench tasks, solving 100% of Level 1 and 90% of Level 2 problems.

**arXiv ID:** 2604.18616
</details>

<details>
<summary><strong>Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning</strong> - Zhiyin Yu, Bo Zhang, Qibin Hou, Zhonghai Wu, Xiao Luo, Lei Bai - [[pdf]](https://arxiv.org/pdf/2604.18639)</summary>

**Abstract:** Previous LLMs-based RL studies typically follow either supervised learning with high annotation costs, or unsupervised paradigms using voting or entropy-based rewards. However, their performance remains far from satisfactory due to the substantial annotation cost and issues such as model collapse or reward hacking. To address these issues, we introduce a new perspective inspired by cognitive learning theory and propose a novel approach called EasyRL. The core of EasyRL is to simulate the human cognitive acquisition curve by integrating reliable knowledge transfer from easy labeled data with a progressive divide-and-conquer strategy that tackles increasingly difficult unlabeled data. Specifically, we initialize a warm-up model using supervised RL with few-shot labeled data. This is followed by a divide-and-conquer pseudo-labeling strategy on difficult unlabeled data, combining consistency-based selection for low-uncertainty cases and reflection-based resolution for medium-uncertainty cases. Finally, difficulty-progressive self-training with iterative pseudo-labeling and RL further strengthens the model's reasoning capability. EasyRL provides a unified self-evolving framework that facilitates data-efficient post-training of LLMs. Experimental results on mathematical and scientific benchmarks demonstrate that EasyRL, using only 10% of easy labeled data, consistently outperforms state-of-the-art baselines.

**arXiv ID:** 2604.18639
</details>

<details>
<summary><strong>Low-Rank Adaptation for Critic Learning in Off-Policy Reinforcement Learning</strong> - Yuan Zhuang, Yuexin Bian, Sihong He, Jie Feng, Qing Su, Songyang Han, Jonathan Petit, Shihao Ji, Yuanyuan Shi, Fei Miao - [[pdf]](https://arxiv.org/pdf/2604.18978)</summary>

**Abstract:** Scaling critic capacity is a promising direction for enhancing off-policy reinforcement learning (RL). However, larger critics are prone to overfitting and unstable in replay-buffer-based bootstrap training. This paper leverages Low-Rank Adaptation (LoRA) as a structural-sparsity regularizer for off-policy critics. Our approach freezes randomly initialized base matrices and solely optimizes low-rank adapters, thereby constraining critic updates to a low-dimensional subspace. Built on top of SimbaV2, we further develop a LoRA formulation, compatible with SimbaV2, that preserves its hyperspherical normalization geometry under frozen-backbone training. We evaluate our method with SAC and FastTD3 on DeepMind Control locomotion and IsaacLab robotics benchmarks. LoRA consistently achieves lower critic loss during training and stronger policy performance. Extensive experiments demonstrate that adaptive low-rank updates provide a simple, scalable, and effective structural regularization for critic learning in off-policy RL.

**arXiv ID:** 2604.18978
</details>

<details>
<summary><strong>Intentional Updates for Streaming Reinforcement Learning</strong> - Arsalan Sharifnassab, Mohamed Elsayed, Kris De Asis, A. Rupam Mahmood, Richard S. Sutton - [[pdf]](https://arxiv.org/pdf/2604.19033)</summary>

**Abstract:** In gradient-based learning, a step size chosen in parameter units does not produce a predictable per-step change in function output. This often leads to instability in the streaming setting (i.e., batch size=1), where stochasticity is not averaged out and update magnitudes can momentarily become arbitrarily big or small. Instead, we propose intentional updates: first specify the intended outcome of an update and then solve for the step size that approximately achieves it. This strategy has precedent in online supervised linear regression via Normalized Least Mean Squares algorithm, which selects a step size to yield a specified change in the function output proportional to the current error. We extend this principle to streaming deep reinforcement learning by defining appropriate intended outcomes: Intentional TD aims for a fixed fractional reduction of the TD error, and Intentional Policy Gradient aims for a bounded per-step change in the policy, limiting local KL divergence. We propose practical algorithms combining eligibility traces and diagonal scaling. Empirically, these methods yield state-of-the-art streaming performance, frequently performing on par with batch and replay-buffer approaches.

**arXiv ID:** 2604.19033
</details>

<details>
<summary><strong>Multi-Gait Learning for Humanoid Robots Using Reinforcement Learning with Selective Adversarial Motion Prior</strong> - Yuanye Wu, Keyi Wang, Linqi Ye, Boyang Xing - [[pdf]](https://arxiv.org/pdf/2604.19102)</summary>

**Abstract:** Learning diverse locomotion skills for humanoid robots in a unified reinforcement learning framework remains challenging due to the conflicting requirements of stability and dynamic expressiveness across different gaits. We present a multi-gait learning approach that enables a humanoid robot to master five distinct gaits -- walking, goose-stepping, running, stair climbing, and jumping -- using a consistent policy structure, action space, and reward formulation. The key contribution is a selective Adversarial Motion Prior (AMP) strategy: AMP is applied to periodic, stability-critical gaits (walking, goose-stepping, stair climbing) where it accelerates convergence and suppresses erratic behavior, while being deliberately omitted for highly dynamic gaits (running, jumping) where its regularization would over-constrain the motion. Policies are trained via PPO with domain randomization in simulation and deployed on a physical 12-DOF humanoid robot through zero-shot sim-to-real transfer. Quantitative comparisons demonstrate that selective AMP outperforms a uniform AMP policy across all five gaits, achieving faster convergence, lower tracking error, and higher success rates on stability-focused gaits without sacrificing the agility required for dynamic ones.

**arXiv ID:** 2604.19102
</details>

<details>
<summary><strong>Reinforcement Learning Enabled Adaptive Multi-Task Control for Bipedal Soccer Robots</strong> - Yulai Zhang, Yinrong Zhang, Ting Wu, Linqi Ye - [[pdf]](https://arxiv.org/pdf/2604.19104)</summary>

**Abstract:** Developing bipedal football robots in dynamiccombat environments presents challenges related to motionstability and deep coupling of multiple tasks, as well ascontrol switching issues between different states such as up-right walking and fall recovery. To address these problems,this paper proposes a modular reinforcement learning (RL)framework for achieving adaptive multi-task control. Firstly,this framework combines an open-loop feedforward oscilla-tor with a reinforcement learning-based feedback residualstrategy, effectively separating the generation of basic gaitsfrom complex football actions. Secondly, a posture-driven statemachine is introduced, clearly switching between the ballseeking and kicking network (BSKN) and the fall recoverynetwork (FRN), fundamentally preventing state this http URL FRN is efficiently trained through a progressive forceattenuation curriculum learning strategy. The architecture wasverified in Unity simulations of bipedal robots, demonstratingexcellent spatial adaptability-reliably finding and kicking theball even in restricted corner scenarios-and rapid autonomousfall recovery (with an average recovery time of 0.715 seconds).This ensures seamless and stable operation in complex multi-task environments.

**arXiv ID:** 2604.19104
</details>

<details>
<summary><strong>MRS: Multi-Resolution Skills for HRL Agents</strong> - Shashank Sharma, Janina Hoffmann, Vinay Namboodiri - [[pdf]](https://arxiv.org/pdf/2505.21410)</summary>

**Abstract:** Hierarchical reinforcement learning (HRL) decomposes the policy into a manager and a worker, enabling long-horizon planning but introducing a performance gap on tasks requiring agility. We identify a root cause: in subgoal-based HRL, the manager's goal representation is typically learned without constraints on reachability or temporal distance from the current state, preventing precise local subgoal selection. We further show that the optimal subgoal distance is both task- and state-dependent: nearby subgoals enable precise control but amplify prediction noise, while distant subgoals produce smoother motion at the cost of geometric precision. We propose Multi-Resolution Skills (MRS), which learns multiple goal-prediction modules each specialized to a fixed temporal horizon, with a jointly trained meta-controller that selects among them based on the current state. MRS consistently outperforms fixed-resolution baselines and significantly reduces the performance gap between HRL and non-HRL state-of-the-art on DeepMind Control Suite, Gym-Robotics, and long-horizon AntMaze tasks. [Project page: this https URL]

**arXiv ID:** 2505.21410
</details>

<details>
<summary><strong>StepFly: Agentic Troubleshooting Guide Automation for Incident Diagnosis</strong> - Jiayi Mao, Liqun Li, Yanjie Gao, Zegang Peng, Shilin He, Chaoyun Zhang, Si Qin, Samia Khalid, Qingwei Lin, Saravan Rajmohan, Sitaram Lanka, Dongmei Zhang - [[pdf]](https://arxiv.org/pdf/2510.10074)</summary>

**Abstract:** Effective incident management in large-scale IT systems relies on troubleshooting guides (TSGs), but their manual execution is slow and error-prone. While recent advances in LLMs offer promise for automating incident management tasks, existing LLM-based solutions lack specialized support for several key challenges, including managing TSG quality issues, interpreting complex control flow, handling data-intensive queries, and exploiting execution parallelism. We first conducted an empirical study on 92 real-world TSGs, and, guided by our findings, we present StepFly, a novel end-to-end agentic framework for troubleshooting guide automation. Our approach features a three-stage workflow: the first stage provides a comprehensive guide together with a tool, TSG Mentor, to assist site reliability engineers (SREs) in improving TSG quality; the second stage performs offline preprocessing using LLMs to extract structured execution directed acyclic graphs (DAGs) from unstructured TSGs and to create dedicated Query Preparation Plugins (QPPs); and the third stage executes online using a DAG-guided scheduler-executor framework with a memory system to ensure correct workflow and support parallel execution of independent steps. Our empirical evaluation on a collection of real-world TSGs and incidents demonstrates that StepFly achieves a ~94% success rate on GPT-4.1, outperforming baselines with less time and token consumption. Furthermore, it achieves a remarkable execution time reduction of 32.9% to 70.4% for parallelizable TSGs. Our code and sample data are publicly available at this https URL.

**arXiv ID:** 2510.10074
</details>

<details>
<summary><strong>BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search</strong> - Shiyu Liu, Yongjing Yin, Jianhao Yan, Yunbo Tang, Qinggang Zhang, Bei Li, Xin Chen, Jingang Wang, Xunliang Cai, Jinsong Su - [[pdf]](https://arxiv.org/pdf/2601.11037)</summary>

**Abstract:** RL-based agentic search enables LLMs to solve complex questions via dynamic planning and external search. While this approach significantly enhances accuracy with agent policies optimized via large-scale reinforcement learning, we identify a critical gap in reliability: these agents fail to recognize their reasoning boundaries and rarely admit ``I DON'T KNOW'' (IDK) even when evidence is insufficient or reasoning reaches its limit. The lack of reliability often leads to plausible but unreliable answers, introducing significant risks in many real-world scenarios. To this end, we propose Boundary-Aware Policy Optimization (BAPO), a novel RL framework designed to cultivate reliable boundary awareness without compromising accuracy. BAPO introduces two key components: (i) a group-based boundary-aware reward that encourages an IDK response only when the reasoning reaches its limit, and (ii) an adaptive reward modulator that strategically suspends this reward during early exploration, preventing the model from exploiting IDK as a shortcut. Extensive experiments on four benchmarks demonstrate that BAPO substantially enhances the overall reliability of agentic search.

**arXiv ID:** 2601.11037
</details>

<details>
<summary><strong>Opinion polarization from compression-based decision making where agents optimize local complexity and global simplicity</strong> - Alina Dubovskaya, David J. P. O'Sullivan, Michael Quayle - [[pdf]](https://arxiv.org/pdf/2604.18755)</summary>

**Abstract:** Understanding social polarization requires integrating insights from psychology, sociology, and complex systems science. Agent-based modeling provides a natural framework to combine perspectives from different fields and explore how individual cognition shapes collective outcomes. This study introduces a novel agent-based model that integrates two cognitive and social mechanisms: the desire to be unique within a group (optimal distinctiveness theory) and the tendency to simplify complex information (cognitive compression). In the model, virtual agents interact in pairs and decide whether to adopt each other's opinions by balancing two opposing drives: maximizing opinion diversity within their local social group while simplifying the overall opinion landscape, with both evaluated using Shannon entropy. We show that the combination of these mechanisms can reproduce real-world patterns, such as the emergence of distinct heterogeneous opinion clusters. Moreover, unlike many existing models where opinions become fixed once opinion groups form, individuals in our model continue to adjust their opinions after clusters emerge, leading to ongoing variation within and between opinion groups. Computational experiments reveal that polarization emerges when local group sizes are moderate (consistent with Dunbar's number), while smaller groups cause fragmentation and larger ones hinder distinct cluster formation. Higher cognitive compression increases unpredictability, while lower compression produces more consistent group structures. These results demonstrate how simple psychological rules can generate complex, realistic social behavior and advance understanding of polarization in human societies.

**arXiv ID:** 2604.18755
</details>

<details>
<summary><strong>ORCA: An Agentic Reasoning Framework for Hallucination and Adversarial Robustness in Vision-Language Models</strong> - Chung-En Johnny Yu, Brian Jalaian, Nathaniel D. Bastian - [[pdf]](https://arxiv.org/pdf/2509.15435)</summary>

**Abstract:** Large Vision-Language Models (LVLMs) exhibit strong multimodal capabilities but remain vulnerable to hallucinations from intrinsic errors and adversarial attacks from external exploitations, limiting their reliability in real-world applications. We present ORCA, an agentic reasoning framework that improves the factual accuracy and adversarial robustness of pretrained LVLMs through inference-time structured inference reasoning with a suite of small vision models (less than 3B parameters). ORCA operates via an Observe-Reason-Critique-Act loop, querying multiple visual tools with evidential questions, validating cross-model inconsistencies, and refining predictions iteratively without access to model internals or retraining. ORCA also stores intermediate reasoning traces, which supports auditable decision-making. Though designed primarily to mitigate object-level hallucinations, ORCA also exhibits emergent adversarial robustness without requiring adversarial training or defense mechanisms. We evaluate ORCA across three settings: (1) clean images on hallucination benchmarks, (2) adversarially perturbed images without defense, and (3) adversarially perturbed images with defense applied. On the POPE hallucination benchmark, ORCA improves standalone LVLMs performance by +3.64% to +40.67% across different subsets. Under adversarial perturbations on POPE, ORCA achieves an average accuracy gain of +20.11% across LVLMs. When combined with defense techniques on adversarially perturbed AMBER images, ORCA further improves standalone LVLM performance, with gains ranging from +1.20% to +48.00% across metrics. These results demonstrate that ORCA offers a promising path toward building more reliable and robust multimodal systems.

**arXiv ID:** 2509.15435
</details>

<details>
<summary><strong>TRN-R1-Zero: Text-rich Network Reasoning via LLMs with Reinforcement Learning Only</strong> - Yilun Liu, Ruihong Qiu, Zi Huang - [[pdf]](https://arxiv.org/pdf/2604.19070)</summary>

**Abstract:** Zero-shot reasoning on text-rich networks (TRNs) remains a challenging frontier, as models must integrate textual semantics with relational structure without task-specific supervision. While graph neural networks rely on fixed label spaces and supervised objectives, recent large language model (LLM)-based approaches often overlook graph context or depend on distillation from larger models, limiting generalisation. We propose TRN-R1-Zero, a post-training framework for TRN reasoning trained solely via reinforcement learning. TRN-R1-Zero directly optimises base LLMs using a Neighbour-aware Group Relative Policy Optimisation objective that dynamically adjusts rewards based on a novel margin gain metric for the informativeness of neighbouring signals, effectively guiding the model toward relational reasoning. Unlike prior methods, TRN-R1-Zero requires no supervised fine-tuning or chain-of-thought data generated from large reasoning models. Extensive experiments across citation, hyperlink, social and co-purchase TRN benchmarks demonstrate the superiority and robustness of TRN-R1-Zero. Moreover, relying strictly on node-level training, TRN-R1-Zero achieves zero-shot inference on edge- and graph-level tasks, extending beyond cross-domain transfer. The codebase is publicly available at this https URL.

**arXiv ID:** 2604.19070
</details>

<details>
<summary><strong>A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression</strong> - Jincheng Ren, Siwei Wu, Yizhi Li, Kang Zhu, Shu Xu, Boyu Feng, Ruibin Yuan, Wei Zhang, Riza Batista-Navarro, Jian Yang, Chenghua Lin - [[pdf]](https://arxiv.org/pdf/2604.19572)</summary>

**Abstract:** As model capabilities advance, research has increasingly shifted toward long-horizon, multi-turn terminal-centric agentic tasks, where raw environment feedback is often preserved in the interaction history to support future decisions. However, repeatedly retaining such feedback introduces substantial redundancy and causes cumulative token cost to grow quadratically with the number of steps, hindering long-horizon reasoning. Although observation compression can mitigate this issue, the heterogeneity of terminal environments makes heuristic-based or fixed-prompt methods difficult to generalize. We propose TACO, a plug-and-play, self-evolving Terminal Agent Compression framework that automatically discovers and refines compression rules from interaction trajectories for existing terminal agents. Experiments on TerminalBench (TB 1.0 and TB 2.0) and four additional terminal-related benchmarks (i.e., SWE-Bench Lite, CompileBench, DevEval, and CRUST-Bench) show that TACO consistently improves performance across mainstream agent frameworks and strong backbone models. With MiniMax-2.5, it improves performance on most benchmarks while reducing token overhead by around 10%. On TerminalBench, it brings consistent gains of 1%-4% across strong agentic models, and further improves accuracy by around 2%-3% under the same token budget. These results demonstrate the effectiveness and generalization of self-evolving, task-aware compression for terminal agents.

**arXiv ID:** 2604.19572
</details>

<details>
<summary><strong>A Goal Without a Plan Is Just a Wish: Efficient and Effective Global Planner Training for Long-Horizon Agent Tasks</strong> - Shuzheng Si, Haozhe Zhao, Kangyang Luo, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, Maosong Sun - [[pdf]](https://arxiv.org/pdf/2510.05608)</summary>

**Abstract:** Agents based on large language models (LLMs) struggle with brainless trial-and-error and generating hallucinatory actions due to a lack of global planning in long-horizon tasks. In this paper, we introduce a plan-and-execute framework and propose EAGLET, an efficient and effective planner training method to enhance the executor agent's planning abilities without human effort. Specifically, we train a plug-and-play global planner through a two-step process: we first synthesize high-quality plans from an advanced LLM using our proposed homologous consensus filtering strategy, and apply fine-tuning as a cold start. Moreover, we further improve the planner with a rule-based reinforcement learning stage using a novel executor capability gain reward, ensuring it can handle task instructions of varying difficulty. Experiments on three long-horizon agent tasks show that executor agents equipped with our planner outperform existing methods, achieving new state-of-the-art performance. Meanwhile, EAGLET reduces training costs by 8x compared to RL-based baselines, and it does not require manual effort or extra training data, offering an efficient and effective solution.

**arXiv ID:** 2510.05608
</details>

<details>
<summary><strong>MiroThinker: Pushing the Performance Boundaries of Open-Source Research Agents via Model, Context, and Interactive Scaling</strong> - MiroMind Team, Song Bai, Lidong Bing, Carson Chen, Guanzheng Chen, Yuntao Chen, Zhe Chen, Ziyi Chen, Jifeng Dai, Xuan Dong, Wenhan Dou, Yue Deng, Yunjie Fu, Junqi Ge, Chenxia Han, Tammy Huang, Zhenhang Huang, Jerry Jiao, Shilei Jiang, Tianyu Jiao, Xiaoqi Jian, Lei Lei, Ruilin Li, Gen Luo, Tiantong Li, Xiang Lin, Ziyuan Liu, Zhiqi Li, Jie Ni, Qiang Ren, Pax Sun, Shiqian Su, Chenxin Tao, Bin Wang, Wenhai Wang, Haonan Wang, James Wang, Jin Wang, Jojo Wang, Letian Wang, Shizun Wang, Weizhi Wang, Zixuan Wang, Jinfan Xu, Sen Xing, Chenyu Yang, Hai Ye, Jiaheng Yu, Yue Yu, Muyan Zhong, Tianchen Zhao, Xizhou Zhu, Yanpeng Zhou, Yifan Zhang, Zhi Zhu - [[pdf]](https://arxiv.org/pdf/2511.11793)</summary>

**Abstract:** We present MiroThinker v1.0, an open-source research agent designed to advance tool-augmented reasoning and information-seeking capabilities. Unlike previous agents that only scale up model size or context length, MiroThinker explores interaction scaling at the model level, systematically training the model to handle deeper and more frequent agent-environment interactions as a third dimension of performance improvement. Unlike LLM test-time scaling, which operates in isolation and risks degradation with longer reasoning chains, interactive scaling leverages environment feedback and external information acquisition to correct errors and refine trajectories. Through reinforcement learning, the model achieves efficient interaction scaling: with a 256K context window, it can perform up to 600 tool calls per task, enabling sustained multi-turn reasoning and complex real-world research workflows. Across four representative benchmarks-GAIA, HLE, BrowseComp, and BrowseComp-ZH-the 72B variant achieves up to 81.9%, 37.7%, 47.1%, and 55.6% accuracy respectively, surpassing previous open-source agents and approaching commercial counterparts such as GPT-5-high. Our analysis reveals that MiroThinker benefits from interactive scaling consistently: research performance improves predictably as the model engages in deeper and more frequent agent-environment interactions, demonstrating that interaction depth exhibits scaling behaviors analogous to model size and context length. These findings establish interaction scaling as a third critical dimension for building next-generation open research agents, complementing model capacity and context windows.

**arXiv ID:** 2511.11793
</details>

<details>
<summary><strong>Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge</strong> - Junjie Wu, Xuan Kan, Zihao He, Shunwen Tan, Bo Pan, Kaitai Zhang - [[pdf]](https://arxiv.org/pdf/2603.11665)</summary>

**Abstract:** Multimodal Large Language Models (MLLMs) have been widely adopted as MLLM-as-a-Judges due to their strong alignment with human judgment across various visual tasks. However, most existing judge models are optimized for single-task scenarios and struggle to generalize to diverse contexts, which is a critical requirement for reliable evaluation. To address this limitation, we propose Multi-Task Reinforcement Learning for MLLM-as-a-Judge (MT-RL-Judge), a framework that jointly optimizes the judge model across multiple tasks, leveraging the generalization capabilities of RL. Experimental results against several strong baselines demonstrate that MT-RL-Judge outperforms strong baselines in both judgment consistency and correlation with human preferences. Furthermore, our approach exhibits robust generalization on out-of-distribution tasks, further validating its effectiveness.

**arXiv ID:** 2603.11665
</details>

<details>
<summary><strong>Coding-Free and Privacy-Preserving Agentic Framework for Data-Driven Clinical Research</strong> - Taehun Kim, Hyeryun Park, Hyeonhoon Lee, Yushin Lee, Kyungsang Kim, Hyung-Chul Lee - [[pdf]](https://arxiv.org/pdf/2604.12258)</summary>

**Abstract:** Clinical data-driven research requires clinical expertise, programming skills, access to patient data, and extensive documentation, creating barriers and slowing the pace for clinicians and external researchers. To address this, we developed the Clinical Agentic Research Intelligence System (CARIS) that automates the workflow: research planning, literature search, cohort construction, Institutional Review Board (IRB) documentation, Vibe Machine Learning (ML), and report generation, with human-in-the-loop refinement. CARIS integrates Large Language Models (LLMs) with modular tools through the Model Context Protocol (MCP), enabling natural language-driven research without coding while allowing users to access only outputs. We evaluated CARIS on three heterogeneous datasets with distinct clinical tasks, where it completed planning and IRB documentation within four iterations, supported Vibe ML, and generated reports, achieving 96% completeness in LLM-based evaluation and 82% in human evaluation. CARIS demonstrates potential to reduce documentation burden and technical barriers, accelerating data-driven clinical research across public and private data environments.

**arXiv ID:** 2604.12258
</details>

<details>
<summary><strong>Guiding Distribution Matching Distillation with Gradient-Based Reinforcement Learning</strong> - Linwei Dong, Ruoyu Guo, Ge Bai, Zehuan Yuan, Yawei Luo, Changqing Zou - [[pdf]](https://arxiv.org/pdf/2604.19009)</summary>

**Abstract:** Diffusion distillation, exemplified by Distribution Matching Distillation (DMD), has shown great promise in few-step generation but often sacrifices quality for sampling speed. While integrating Reinforcement Learning (RL) into distillation offers potential, a naive fusion of these two objectives relies on suboptimal raw sample evaluation. This sample-based scoring creates inherent conflicts with the distillation trajectory and produces unreliable rewards due to the noisy nature of early-stage generation. To overcome these limitations, we propose GDMD, a novel framework that redefines the reward mechanism by prioritizing distillation gradients over raw pixel outputs as the primary signal for optimization. By reinterpreting the DMD gradients as implicit target tensors, our framework enables existing reward models to directly evaluate the quality of distillation updates. This gradient-level guidance functions as an adaptive weighting that synchronizes the RL policy with the distillation objective, effectively neutralizing optimization divergence. Empirical results show that GDMD sets a new SOTA for few-step generation. Specifically, our 4-step models outperform the quality of their multi-step teacher and substantially exceed previous DMDR results in GenEval and human-preference metrics, exhibiting strong scalability potential.

**arXiv ID:** 2604.19009
</details>

<details>
<summary><strong>Policy Gradient Primal-Dual Method for Safe Reinforcement Learning from Human Feedback</strong> - Qiang Liu, Adrienne Kline, Ermin Wei - [[pdf]](https://arxiv.org/pdf/2604.19024)</summary>

**Abstract:** Safe Reinforcement Learning from Human Feedback (Safe RLHF) has recently achieved empirical success in developing helpful and harmless large language models by decoupling human preferences regarding helpfulness and harmlessness. Existing approaches typically rely on fitting fixed horizon reward models from human feedback and have only been validated empirically. In this paper, we formulate safe RLHF as an infinite horizon discounted Con- strained Markov Decision Process (CMDP), since humans may interact with the model over a continuing sequence of interactions rather than within a single finite episode. We propose two Safe RLHF algorithms that do not require reward model fitting and, in contrast to prior work assuming fixed-length trajectories, support flexible trajectory lengths for training. Both algo- rithms are based on the primal-dual method and achieve global convergence guarantees with polynomial rates in terms of policy gradient iterations, trajectory sample lengths, and human preference queries. To the best of our knowledge, this is the first work to study infinite horizon discounted CMDP under human feedback and establish global, non-asymptotic convergence.

**arXiv ID:** 2604.19024
</details>

<details>
<summary><strong>RL-ABC: Reinforcement Learning for Accelerator Beamline Control</strong> - Anwar Ibrahim, Fedor Ratnikov, Maxim Kaledin, Alexey Petrenko, Denis Derkach - [[pdf]](https://arxiv.org/pdf/2604.19146)</summary>

**Abstract:** Particle accelerator beamline optimization is a high-dimensional control problem traditionally requiring significant expert intervention. We present RLABC (Reinforcement Learning for Accelerator Beamline Control), an open-source Python framework that automatically transforms standard Elegant beamline configurations into reinforcement learning environments. RLABC integrates with the widely-used Elegant beam dynamics simulation code via SDDS-based interfaces, enabling researchers to apply modern RL algorithms to beamline optimization with minimal RL-specific development.
The main contribution is a general methodology for formulating beamline tuning as a Markov decision process: RLABC automatically preprocesses lattice files to insert diagnostic watch points before each tunable element, constructs a 57-dimensional state representation from beam statistics, covariance information, and aperture constraints, and provides a configurable reward function for transmission optimization. The framework supports multiple RL algorithms through Stable-Baselines3 compatibility and implements stage learning strategies for improved training efficiency.
Validation on a test beamline derived from the VEPP-5 injection complex (37 control parameters across 11 quadrupoles and 4 dipoles) demonstrates that the framework successfully enables RL-based optimization, with a Deep Deterministic Policy Gradient agent achieving 70.3\% particle transmission -- performance matching established methods such as differential evolution. The framework's stage learning capability allows decomposition of complex optimization problems into manageable subproblems, improving training efficiency. The complete framework, including configuration files and example notebooks, is available as open-source software to facilitate adoption and further research.

**arXiv ID:** 2604.19146
</details>

<details>
<summary><strong>Safe Continual Reinforcement Learning in Non-stationary Environments</strong> - Austin Coursey, Abel Diaz-Gonzalez, Marcos Quinones-Grueiro, Gautam Biswas - [[pdf]](https://arxiv.org/pdf/2604.19737)</summary>

**Abstract:** Reinforcement learning (RL) offers a compelling data-driven paradigm for synthesizing controllers for complex systems when accurate physical models are unavailable; however, most existing control-oriented RL methods assume stationarity and, therefore, struggle in real-world non-stationary deployments where system dynamics and operating conditions can change unexpectedly. Moreover, RL controllers acting in physical environments must satisfy safety constraints throughout their learning and execution phases, rendering transient violations during adaptation unacceptable. Although continual RL and safe RL have each addressed non-stationarity and safety, respectively, their intersection remains comparatively unexplored, motivating the study of safe continual RL algorithms that can adapt over the system's lifetime while preserving safety. In this work, we systematically investigate safe continual reinforcement learning by introducing three benchmark environments that capture safety-critical continual adaptation and by evaluating representative approaches from safe RL, continual RL, and their combinations. Our empirical results reveal a fundamental tension between maintaining safety constraints and preventing catastrophic forgetting under non-stationary dynamics, with existing methods generally failing to achieve both objectives simultaneously. To address this shortcoming, we examine regularization-based strategies that partially mitigate this trade-off and characterize their benefits and limitations. Finally, we outline key open challenges and research directions toward developing safe, resilient learning-based controllers capable of sustained autonomous operation in changing environments.

**arXiv ID:** 2604.19737
</details>

<details>
<summary><strong>From Particles to Perils: SVGD-Based Hazardous Scenario Generation for Autonomous Driving Systems Testing</strong> - Linfeng Liang, Xiao Cheng, Tsong Yueh Chen, Xi Zheng - [[pdf]](https://arxiv.org/pdf/2604.18918)</summary>

**Abstract:** Simulation-based testing of autonomous driving systems (ADS) must uncover realistic and diverse failures in dense, heterogeneous traffic. However, existing search-based seeding methods (e.g., genetic algorithms) struggle in high-dimensional spaces, often collapsing to limited modes and missing many failure scenarios. We present PtoP, a framework that combines adaptive random seed generation with Stein Variational Gradient Descent (SVGD) to produce diverse, failure-inducing initial conditions. SVGD balances attraction toward high-risk regions and repulsion among particles, yielding risk-seeking yet well-distributed seeds across multiple failure modes. PtoP is plug-and-play and enhances existing online testing methods (e.g., reinforcement learning--based testers) by providing principled seeds. Evaluation in CARLA on two industry-grade ADS (Apollo, Autoware) and a native end-to-end system shows that PtoP improves safety violation rate (up to 27.68%), scenario diversity (9.6%), and map coverage (16.78%) over baselines.

**arXiv ID:** 2604.18918
</details>

<details>
<summary><strong>ASVSim (AirSim for Surface Vehicles): A High-Fidelity Simulation Framework for Autonomous Surface Vehicle Research</strong> - Bavo Lesy, Siemen Herremans, Robin Kerstens, Jan Steckel, Walter Daems, Siegfried Mercelis, Ali Anwar - [[pdf]](https://arxiv.org/pdf/2506.22174)</summary>

**Abstract:** The transport industry has recently shown significant interest in unmanned surface vehicles (USVs), specifically for port and inland waterway transport. These systems can improve operational efficiency and safety, which is especially relevant in the European Union, where initiatives such as the Green Deal are driving a shift towards increased use of inland waterways. At the same time, a shortage of qualified personnel is accelerating the adoption of autonomous solutions. However, there is a notable lack of open-source, high-fidelity simulation frameworks and datasets for developing and evaluating such solutions. To address these challenges, we introduce AirSim for Surface Vehicles (ASVSim), an open-source simulation framework specifically designed for autonomous shipping research in inland and port environments. The framework combines simulated vessel dynamics with marine sensor simulation capabilities, including radar and camera systems and supports the generation of synthetic datasets for training computer vision models and reinforcement learning (RL) agents. Built upon Cosys-AirSim, ASVSim provides a comprehensive platform for developing autonomous navigation algorithms and generating synthetic datasets. The simulator supports research of both traditional control methods and deep learning-based approaches. Through experiments in waterway segmentation and autonomous navigation, we demonstrate the capabilities of the simulator in these research areas. ASVSim is provided as an open-source project under the MIT license, making autonomous navigation research accessible to a larger part of the ocean engineering community. See this https URL.

**arXiv ID:** 2506.22174
</details>

<details>
<summary><strong>QTMRL: An Agent for Quantitative Trading Decision-Making Based on Multi-Indicator Guided Reinforcement Learning</strong> - Jingfeng Pan, Jiahao Chen - [[pdf]](https://arxiv.org/pdf/2508.20467)</summary>

**Abstract:** In the highly volatile and uncertain global financial markets, traditional quantitative trading models relying on statistical modeling or empirical rules often fail to adapt to dynamic market changes and black swan events due to rigid assumptions and limited generalization. To address these issues, this paper proposes QTMRL (Quantitative Trading Multi-Indicator Reinforcement Learning), an intelligent trading agent combining multi-dimensional technical indicators with reinforcement learning (RL) for adaptive and stable portfolio management. We first construct a comprehensive multi-indicator dataset using 23 years of S&P 500 daily OHLCV data (2000-2022) for 16 representative stocks across 5 sectors, enriching raw data with trend, volatility, and momentum indicators to capture holistic market dynamics. Then we design a lightweight RL framework based on the Advantage Actor-Critic (A2C) algorithm, including data processing, A2C algorithm, and trading agent modules to support policy learning and actionable trading decisions. Extensive experiments compare QTMRL with 9 baselines (e.g., ARIMA, LSTM, moving average strategies) across diverse market regimes, verifying its superiority in profitability, risk adjustment, and downside risk control. The code of QTMRL is publicly available at this https URL

**arXiv ID:** 2508.20467
</details>

<details>
<summary><strong>Localization-Guided Foreground Augmentation in Autonomous Driving</strong> - Jiawei Yong, Deyuan Qu, Qi Chen, Kentaro Oguchi, Shintaro Fukushima - [[pdf]](https://arxiv.org/pdf/2604.18940)</summary>

**Abstract:** Autonomous driving systems often degrade under adverse visibility conditions-such as rain, nighttime, or snow-where online scene geometry (e.g., lane dividers, road boundaries, and pedestrian crossings) becomes sparse or fragmented. While high-definition (HD) maps can provide missing structural context, they are costly to construct and maintain at scale. We propose Localization-Guided Foreground Augmentation (LG-FA), a lightweight and plug-and-play inference module that enhances foreground perception by enriching geometric context online. LG-FA: (i) incrementally constructs a sparse global vector layer from per-frame Bird's-Eye View (BEV) predictions; (ii) estimates ego pose via class-constrained geometric alignment, jointly improving localization and completing missing local topology; and (iii) reprojects the augmented foreground into a unified global frame to improve per-frame predictions. Experiments on challenging nuScenes sequences demonstrate that LG-FA improves the geometric completeness and temporal stability of BEV representations, reduces localization error, and produces globally consistent lane and topology reconstructions. The module can be seamlessly integrated into existing BEV-based perception systems without backbone modification. By providing a reliable geometric context prior, LG-FA enhances temporal consistency and supplies stable structural support for downstream modules such as tracking and decision-making.

**arXiv ID:** 2604.18940
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
