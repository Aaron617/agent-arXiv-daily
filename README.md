# Agent arXiv Daily

**Last Updated:** 2025-09-04 02:34:40

**Total Papers:** 55

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (3 papers)</h2></summary>

<details>
<summary><strong>Situating AI Agents in their World: Aspective Agentic AI for Dynamic Partially Observable Information Systems</strong> - Peter J. Bentley, Soo Ling Lim, Fuyuki Ishikawa - [[pdf]](https://arxiv.org/pdf/2509.03380)</summary>

**Abstract:** Agentic LLM AI agents are often little more than autonomous chatbots: actors following scripts, often controlled by an unreliable director. This work introduces a bottom-up framework that situates AI agents in their environment, with all behaviors triggered by changes in their environments. It introduces the notion of aspects, similar to the idea of umwelt, where sets of agents perceive their environment differently to each other, enabling clearer control of information. We provide an illustrative implementation and show that compared to a typical architecture, which leaks up to 83% of the time, aspective agentic AI enables zero information leakage. We anticipate that this concept of specialist agents working efficiently in their own information niches can provide improvements to both security and efficiency.

**arXiv ID:** 2509.03380
</details>

<details>
<summary><strong>Autonomous Learning From Success and Failure: Goal-Conditioned Supervised Learning with Negative Feedback</strong> - Zeqiang Zhang, Fabian Wurzberger, Gerrit Schmid, Sebastian Gottwald, Daniel A. Braun - [[pdf]](https://arxiv.org/pdf/2509.03206)</summary>

**Abstract:** Reinforcement learning faces significant challenges when applied to tasks characterized by sparse reward structures. Although imitation learning, within the domain of supervised learning, offers faster convergence, it relies heavily on human-generated demonstrations. Recently, Goal-Conditioned Supervised Learning (GCSL) has emerged as a potential solution by enabling self-imitation learning for autonomous systems. By strategically relabelling goals, agents can derive policy insights from their own experiences. Despite the successes of this framework, it presents two notable limitations: (1) Learning exclusively from self-generated experiences can exacerbate the agents' inherent biases; (2) The relabelling strategy allows agents to focus solely on successful outcomes, precluding them from learning from their mistakes. To address these issues, we propose a novel model that integrates contrastive learning principles into the GCSL framework to learn from both success and failure. Through empirical evaluations, we demonstrate that our algorithm overcomes limitations imposed by agents' initial biases and thereby enables more exploratory behavior. This facilitates the identification and adoption of effective policies, leading to superior performance across a variety of challenging environments.

**arXiv ID:** 2509.03206
</details>

<details>
<summary><strong>IL-SLAM: Intelligent Line-assisted SLAM Based on Feature Awareness for Dynamic Environments</strong> - Haolan Zhang, Thanh Nguyen Canh, Chenghao Li, Ruidong Yang, Yonghoon Ji, Nak Young Chong - [[pdf]](https://arxiv.org/pdf/2509.02972)</summary>

**Abstract:** Visual Simultaneous Localization and Mapping (SLAM) plays a crucial role in autonomous systems. Traditional SLAM methods, based on static environment assumptions, struggle to handle complex dynamic environments. Recent dynamic SLAM systems employ geometric constraints and deep learning to remove dynamic features, yet this creates a new challenge: insufficient remaining point features for subsequent SLAM processes. Existing solutions address this by continuously introducing additional line and plane features to supplement point features, achieving robust tracking and pose estimation. However, current methods continuously introduce additional features regardless of necessity, causing two problems: unnecessary computational overhead and potential performance degradation from accumulated low-quality additional features and noise. To address these issues, this paper proposes a feature-aware mechanism that evaluates whether current features are adequate to determine if line feature support should be activated. This decision mechanism enables the system to introduce line features only when necessary, significantly reducing computational complexity of additional features while minimizing the introduction of low-quality features and noise. In subsequent processing, the introduced line features assist in obtaining better initial camera poses through tracking, local mapping, and loop closure, but are excluded from global optimization to avoid potential negative impacts from low-quality additional features in long-term process. Extensive experiments on TUM datasets demonstrate substantial improvements in both ATE and RPE metrics compared to ORB-SLAM3 baseline and superior performance over other dynamic SLAM and multi-feature methods.

**arXiv ID:** 2509.02972
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>app.build: A Production Framework for Scaling Agentic Prompt-to-App Generation with Environment Scaffolding</strong> - Evgenii Kniazev, Arseny Kravchenko, Igor Rekun, James Broadhead, Nikita Shamgunov, Pranav Sah, Pratik Nichite, Ivan Yamshchikov - [[pdf]](https://arxiv.org/pdf/2509.03310)</summary>

**Abstract:** We present this http URL (this https URL), an open-source framework that improves LLM-based application generation through systematic validation and structured environments. Our approach combines multi-layered validation pipelines, stack-specific orchestration, and model-agnostic architecture, implemented across three reference stacks. Through evaluation on 30 generation tasks, we demonstrate that comprehensive validation achieves 73.3% viability rate with 30% reaching perfect quality scores, while open-weights models achieve 80.8% of closed-model performance when provided structured environments. The open-source framework has been adopted by the community, with over 3,000 applications generated to date. This work demonstrates that scaling reliable AI agents requires scaling environments, not just models -- providing empirical insights and complete reference implementations for production-oriented agent systems.

**arXiv ID:** 2509.03310
</details>

<details>
<summary><strong>Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving</strong> - Mingyi Wang, Jingke Wang, Tengju Ye, Junbo Chen, Kaicheng Yu - [[pdf]](https://arxiv.org/pdf/2509.02754)</summary>

**Abstract:** Recent breakthroughs in large language models (LLMs) have not only advanced natural language processing but also inspired their application in domains with structurally similar problems--most notably, autonomous driving motion generation. Both domains involve autoregressive sequence modeling, token-based representations, and context-aware decision making, making the transfer of LLM components a natural and increasingly common practice. However, despite promising early attempts, a systematic understanding of which LLM modules are truly transferable remains lacking. In this paper, we present a comprehensive evaluation of five key LLM modules--tokenizer design, positional embedding, pre-training paradigms, post-training strategies, and test-time computation--within the context of motion generation for autonomous driving. Through extensive experiments on the Waymo Sim Agents benchmark, we demonstrate that, when appropriately adapted, these modules can significantly improve performance for autonomous driving motion generation. In addition, we identify which techniques can be effectively transferred, analyze the potential reasons for the failure of others, and discuss the specific adaptations needed for autonomous driving scenarios. We evaluate our method on the Sim Agents task and achieve competitive results.

**arXiv ID:** 2509.02754
</details>

<details>
<summary><strong>JARVIS: A Neuro-Symbolic Commonsense Reasoning Framework for Conversational Embodied Agents</strong> - Kaizhi Zheng, Kaiwen Zhou, Jing Gu, Yue Fan, Jialu Wang, Zonglin Di, Xuehai He, Xin Eric Wang - [[pdf]](https://arxiv.org/pdf/2208.13266)</summary>

**Abstract:** Building a conversational embodied agent to execute real-life tasks has been a long-standing yet quite challenging research goal, as it requires effective human-agent communication, multi-modal understanding, long-range sequential decision making, etc. Traditional symbolic methods have scaling and generalization issues, while end-to-end deep learning models suffer from data scarcity and high task complexity, and are often hard to explain. To benefit from both worlds, we propose JARVIS, a neuro-symbolic commonsense reasoning framework for modular, generalizable, and interpretable conversational embodied agents. First, it acquires symbolic representations by prompting large language models (LLMs) for language understanding and sub-goal planning, and by constructing semantic maps from visual observations. Then the symbolic module reasons for sub-goal planning and action generation based on task- and action-level common sense. Extensive experiments on the TEACh dataset validate the efficacy and efficiency of our JARVIS framework, which achieves state-of-the-art (SOTA) results on all three dialog-based embodied tasks, including Execution from Dialog History (EDH), Trajectory from Dialog (TfD), and Two-Agent Task Completion (TATC) (e.g., our method boosts the unseen Success Rate on EDH from 6.1\% to 15.8\%). Moreover, we systematically analyze the essential factors that affect the task performance and also demonstrate the superiority of our method in few-shot settings. Our JARVIS model ranks first in the Alexa Prize SimBot Public Benchmark Challenge.

**arXiv ID:** 2208.13266
</details>

<details>
<summary><strong>Securing AI Agents with Information-Flow Control</strong> - Manuel Costa, Boris Köpf, Aashish Kolluri, Andrew Paverd, Mark Russinovich, Ahmed Salem, Shruti Tople, Lukas Wutschitz, Santiago Zanella-Béguelin - [[pdf]](https://arxiv.org/pdf/2505.23643)</summary>

**Abstract:** As AI agents become increasingly autonomous and capable, ensuring their security against vulnerabilities such as prompt injection becomes critical. This paper explores the use of information-flow control (IFC) to provide security guarantees for AI agents. We present a formal model to reason about the security and expressiveness of agent planners. Using this model, we characterize the class of properties enforceable by dynamic taint-tracking and construct a taxonomy of tasks to evaluate security and utility trade-offs of planner designs. Informed by this exploration, we present Fides, a planner that tracks confidentiality and integrity labels, deterministically enforces security policies, and introduces novel primitives for selectively hiding information. Its evaluation in AgentDojo demonstrates that this approach enables us to complete a broad range of tasks with security guarantees. A tutorial to walk readers through the the concepts introduced in the paper can be found at this https URL

**arXiv ID:** 2505.23643
</details>

<details>
<summary><strong>LinkAnchor: An Autonomous LLM-Based Agent for Issue-to-Commit Link Recovery</strong> - Arshia Akhavan, Alireza Hosseinpour, Abbas Heydarnoori, Mehdi Keshani - [[pdf]](https://arxiv.org/pdf/2508.12232)</summary>

**Abstract:** Issue-to-commit link recovery plays an important role in software traceability and improves project management. However, it remains a challenging task. A study on GitHub shows that only 42.2% of the issues are correctly linked to their commits. This highlights the potential for further development and research in this area. Existing studies have employed various AI/ML-based approaches, and with the recent development of large language models, researchers have leveraged LLMs to tackle this problem. These approaches suffer from two main issues. First, LLMs are constrained by limited context windows and cannot ingest all of the available data sources, such as long commit histories, extensive issue comments, and large code repositories. Second, most methods operate on individual issue-commit pairs; that is, given a single issue-commit pair, they determine whether the commit resolves the issue. This quickly becomes impractical in real-world repositories containing tens of thousands of commits. To address these limitations, we present LinkAnchor, the first autonomous LLM-based agent designed for issue-to-commit link recovery. The lazy-access architecture of LinkAnchor enables the underlying LLM to access the rich context of software, spanning commits, issue comments, and code files, without exceeding the token limit by dynamically retrieving only the most relevant contextual data. Additionally, LinkAnchor is able to automatically pinpoint the target commit rather than exhaustively scoring every possible candidate. Our evaluations show that LinkAnchor outperforms state-of-the-art issue-to-commit link recovery approaches by 60-262% in Hit@1 score across all our case study projects. We also publicly release LinkAnchor as a ready-to-use tool, along with our replication package. LinkAnchor is designed and tested for GitHub and Jira, and is easily extendable to other platforms.

**arXiv ID:** 2508.12232
</details>

<details>
<summary><strong>Murakkab: Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms</strong> - Gohar Irfan Chaudhry, Esha Choukse, Haoran Qiu, Íñigo Goiri, Rodrigo Fonseca, Adam Belay, Ricardo Bianchini - [[pdf]](https://arxiv.org/pdf/2508.18298)</summary>

**Abstract:** Agentic workflows commonly coordinate multiple models and tools with complex control logic. They are quickly becoming the dominant paradigm for AI applications. However, serving them remains inefficient with today's frameworks. The key problem is that they expose workflows as opaque sequences of model and tool calls that tightly couple agent logic with model and hardware choices. Often, these workflow components are fragmented across different entities, preventing systems from reasoning about trade-offs across accuracy, latency, energy, and cost. This leads to resource waste and degraded service-level objectives (SLOs).
We present Murakkab, a resource-efficient serving system for agentic workflows. Murakkab introduces a declarative abstraction that decouples workflow specification from execution configuration. A profile-guided optimizer and adaptive runtime jointly manage the full stack: orchestrating workflow components, mapping them to models and hardware, and dynamically reconfiguring execution to satisfy user-defined SLOs. By exposing the internal structure of agentic workflows, Murakkab enables cross-layer optimization that existing frameworks and cloud schedulers cannot achieve.
Our evaluation on diverse workflows shows that Murakkab reduces GPU usage by up to 2.8$\times$, energy consumption by 3.7$\times$, and cost by 4.3$\times$ while maintaining SLOs.

**arXiv ID:** 2508.18298
</details>

<details>
<summary><strong>Network-Level Prompt and Trait Leakage in Local Research Agents</strong> - Hyejun Jeong, Mohammadreza Teymoorianfard, Abhinav Kumar, Amir Houmansadr, Eugene Bagdasarian - [[pdf]](https://arxiv.org/pdf/2508.20282)</summary>

**Abstract:** We show that Web and Research Agents (WRAs) -- language model-based systems that investigate complex topics on the Internet -- are vulnerable to inference attacks by passive network adversaries such as ISPs. These agents could be deployed locally by organizations and individuals for privacy, legal, or financial purposes. Unlike sporadic web browsing by humans, WRAs visit $70{-}140$ domains with distinguishable timing correlations, enabling unique fingerprinting attacks.
Specifically, we demonstrate a novel prompt and user trait leakage attack against WRAs that only leverages their network-level metadata (i.e., visited IP addresses and their timings). We start by building a new dataset of WRA traces based on user search queries and queries generated by synthetic personas. We define a behavioral metric (called OBELS) to comprehensively assess similarity between original and inferred prompts, showing that our attack recovers over 73% of the functional and domain knowledge of user prompts. Extending to a multi-session setting, we recover up to 19 of 32 latent traits with high accuracy. Our attack remains effective under partial observability and noisy conditions. Finally, we discuss mitigation strategies that constrain domain diversity or obfuscate traces, showing negligible utility impact while reducing attack effectiveness by an average of 29%.

**arXiv ID:** 2508.20282
</details>

<details>
<summary><strong>Locus: Agentic Predicate Synthesis for Directed Fuzzing</strong> - Jie Zhu, Chihao Shen, Ziyang Li, Jiahao Yu, Yizheng Chen, Kexin Pei - [[pdf]](https://arxiv.org/pdf/2508.21302)</summary>

**Abstract:** Directed fuzzing aims to find program inputs that lead to specified target program states. It has broad applications, such as debugging system crashes, confirming reported bugs, and generating exploits for potential vulnerabilities. This task is inherently challenging because target states are often deeply nested in the program, while the search space manifested by numerous possible program inputs is prohibitively large. Existing approaches rely on branch distances or manually-specified constraints to guide the search; however, the branches alone are often insufficient to precisely characterize progress toward reaching the target states, while the manually specified constraints are often tailored for specific bug types and thus difficult to generalize to diverse target states and programs.
We present Locus, a novel framework to improve the efficiency of directed fuzzing. Our key insight is to synthesize predicates to capture fuzzing progress as semantically meaningful intermediate states, serving as milestones towards reaching the target states. When used to instrument the program under fuzzing, they can reject executions unlikely to reach the target states, while providing additional coverage guidance. To automate this task and generalize to diverse programs, Locus features an agentic framework with program analysis tools to synthesize and iteratively refine the candidate predicates, while ensuring the predicates strictly relax the target states to prevent false rejections via symbolic execution. Our evaluation shows that Locus substantially improves the efficiency of eight state-of-the-art fuzzers in discovering real-world vulnerabilities, achieving an average speedup of 41.6x. So far, Locus has found eight previously unpatched bugs, with one already acknowledged with a draft patch.

**arXiv ID:** 2508.21302
</details>

<details>
<summary><strong>EvolveSignal: A Large Language Model Powered Coding Agent for Discovering Traffic Signal Control Algorithms</strong> - Leizhen Wang, Peibo Duan, Hao Wang, Yue Wang, Jian Xu, Nan Zheng, Zhenliang Ma - [[pdf]](https://arxiv.org/pdf/2509.03335)</summary>

**Abstract:** In traffic engineering, the fixed-time traffic signal control remains widely used for its low cost, stability, and interpretability. However, its design depends on hand-crafted formulas (e.g., Webster) and manual re-timing by engineers to adapt to demand changes, which is labor-intensive and often yields suboptimal results under heterogeneous or congested conditions. This paper introduces the EvolveSignal, a large language models (LLMs) powered coding agent to automatically discover new traffic signal control algorithms. We formulate the problem as program synthesis, where candidate algorithms are represented as Python functions with fixed input-output structures, and iteratively optimized through external evaluations (e.g., a traffic simulator) and evolutionary search. Experiments on a signalized intersection demonstrate that the discovered algorithms outperform Webster's baseline, reducing average delay by 20.1% and average stops by 47.1%. Beyond performance, ablation and incremental analyses reveal that EvolveSignal modifications-such as adjusting cycle length bounds, incorporating right-turn demand, and rescaling green allocations-can offer practically meaningful insights for traffic engineers. This work opens a new research direction by leveraging AI for algorithm design in traffic signal control, bridging program synthesis with transportation engineering.

**arXiv ID:** 2509.03335
</details>

</details>

<details open>
<summary><h2>LLM Agents (2 papers)</h2></summary>

<details>
<summary><strong>Group-in-Group Policy Optimization for LLM Agent Training</strong> - Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An - [[pdf]](https://arxiv.org/pdf/2505.10978)</summary>

**Abstract:** Recent advances in group-based reinforcement learning (RL) have driven frontier large language models (LLMs) in single-turn tasks like mathematical reasoning. However, their scalability to long-horizon LLM agent training remains limited. Unlike static tasks, agent-environment interactions unfold over many steps and often yield sparse or delayed rewards, making credit assignment across individual steps significantly more challenging. In this work, we propose Group-in-Group Policy Optimization (GiGPO), a novel RL algorithm that achieves fine-grained credit assignment for LLM agents while preserving the appealing properties of group-based RL: critic-free, low memory, and stable convergence. GiGPO introduces a two-level structure for estimating relative advantage: (i) At the episode-level, GiGPO computes macro relative advantages based on groups of complete trajectories; (ii) At the step-level, GiGPO introduces an anchor state grouping mechanism that retroactively constructs step-level groups by identifying repeated environment states across trajectories. Actions stemming from the same state are grouped together, enabling micro relative advantage estimation. This hierarchical structure effectively captures both global trajectory quality and local step effectiveness without relying on auxiliary models or additional rollouts. We evaluate GiGPO on two challenging agent benchmarks, ALFWorld and WebShop, using Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct. Crucially, GiGPO delivers fine-grained per-step credit signals and achieves performance gains of > 12\% on ALFWorld and > 9\% on WebShop over the GRPO baseline: all while maintaining the same GPU memory overhead, identical LLM rollout, and incurring little to no additional time cost.

**arXiv ID:** 2505.10978
</details>

<details>
<summary><strong>Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning</strong> - Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Hinrich Schütze, Volker Tresp, Yunpu Ma - [[pdf]](https://arxiv.org/pdf/2508.19828)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of NLP tasks, but they remain fundamentally stateless, constrained by limited context windows that hinder long-horizon reasoning. Recent efforts to address this limitation often augment LLMs with an external memory bank, yet most existing pipelines are static and heuristic-driven, lacking any learned mechanism for deciding what to store, update, or retrieve. We present Memory-R1, a reinforcement learning (RL) framework that equips LLMs with the ability to actively manage and utilize external memory through two specialized agents: a Memory Manager that learns to perform structured memory operations, including adding, updating, deleting, or taking no operation on memory entries; and an Answer Agent that selects the most relevant entries and reasons over them to produce an answer. Both agents are fine-tuned with outcome-driven RL (PPO and GRPO), enabling adaptive memory management and utilization with minimal supervision. With as few as 152 question-answer pairs and a corresponding temporal memory bank for training, Memory-R1 outperforms the strongest existing baseline and demonstrates strong generalization across diverse question types and LLM backbones. Beyond presenting an effective approach, this work provides insights into how RL can unlock more agentic, memory-aware behavior in LLMs, pointing toward richer, more persistent reasoning systems.

**arXiv ID:** 2508.19828
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (11 papers)</h2></summary>

<details>
<summary><strong>Latent Variable Modeling in Multi-Agent Reinforcement Learning via Expectation-Maximization for UAV-Based Wildlife Protection</strong> - Mazyar Taghavi, Rahman Farnoosh - [[pdf]](https://arxiv.org/pdf/2509.02579)</summary>

**Abstract:** Protecting endangered wildlife from illegal poaching presents a critical challenge, particularly in vast and partially observable environments where real-time response is essential. This paper introduces a novel Expectation-Maximization (EM) based latent variable modeling approach in the context of Multi-Agent Reinforcement Learning (MARL) for Unmanned Aerial Vehicle (UAV) coordination in wildlife protection. By modeling hidden environmental factors and inter-agent dynamics through latent variables, our method enhances exploration and coordination under this http URL implement and evaluate our EM-MARL framework using a custom simulation involving 10 UAVs tasked with patrolling protected habitats of the endangered Iranian leopard. Extensive experimental results demonstrate superior performance in detection accuracy, adaptability, and policy convergence when compared to standard algorithms such as Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG). Our findings underscore the potential of combining EM inference with MARL to improve decentralized decisionmaking in complex, high-stakes conservation scenarios. The full implementation, simulation environment, and training scripts are publicly available on GitHub.

**arXiv ID:** 2509.02579
</details>

<details>
<summary><strong>MorphAgent: Empowering Agents through Self-Evolving Profiles and Decentralized Collaboration</strong> - Siyuan Lu, Jiaqi Shao, Bing Luo, Tao Lin - [[pdf]](https://arxiv.org/pdf/2410.15048)</summary>

**Abstract:** Large Language Model (LLM) based multi-agent systems (MAS) have shown promise in tackling complex tasks, but often rely on predefined roles and centralized coordination, limiting their adaptability to evolving challenges. This paper introduces MorphAgent, a novel Autonomous, Self-Organizing, and Self-Adaptive Multi-Agent System for decentralized agent collaboration that enables agents to dynamically evolve their roles and capabilities. Our approach employs self-evolving agent profiles, optimized through three key metrics, guiding agents in refining their individual expertise while maintaining complementary team dynamics. MorphAgent implements a two-phase process: a Profile Update phase for profile optimization, followed by a Task Execution phase where agents continuously adapt their roles based on task feedback. Our experimental results show that MorphAgent outperforms existing frameworks in terms of task performance and adaptability to changing requirements, paving the way for more robust and versatile multi-agent collaborative systems.

**arXiv ID:** 2410.15048
</details>

<details>
<summary><strong>Deep Research Agents: A Systematic Examination And Roadmap</strong> - Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Huichi Zhou, Meng Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, Jianye Hao, Kun Shao, Jun Wang - [[pdf]](https://arxiv.org/pdf/2506.18096)</summary>

**Abstract:** The rapid progress of Large Language Models (LLMs) has given rise to a new category of autonomous AI systems, referred to as Deep Research (DR) agents. These agents are designed to tackle complex, multi-turn informational research tasks by leveraging a combination of dynamic reasoning, adaptive long-horizon planning, multi-hop information retrieval, iterative tool use, and the generation of structured analytical reports. In this paper, we conduct a detailed analysis of the foundational technologies and architectural components that constitute Deep Research agents. We begin by reviewing information acquisition strategies, contrasting API-based retrieval methods with browser-based exploration. We then examine modular tool-use frameworks, including code execution, multimodal input processing, and the integration of Model Context Protocols (MCPs) to support extensibility and ecosystem development. To systematize existing approaches, we propose a taxonomy that differentiates between static and dynamic workflows, and we classify agent architectures based on planning strategies and agent composition, including single-agent and multi-agent configurations. We also provide a critical evaluation of current benchmarks, highlighting key limitations such as restricted access to external knowledge, sequential execution inefficiencies, and misalignment between evaluation metrics and the practical objectives of DR agents. Finally, we outline open challenges and promising directions for future research. A curated and continuously updated repository of DR agent research is available at: {this https URL}.

**arXiv ID:** 2506.18096
</details>

<details>
<summary><strong>Symbiotic Agents: A Novel Paradigm for Trustworthy AGI-driven Networks</strong> - Ilias Chatzistefanidis, Navid Nikaein - [[pdf]](https://arxiv.org/pdf/2507.17695)</summary>

**Abstract:** Large Language Model (LLM)-based autonomous agents are expected to play a vital role in the evolution of 6G networks, by empowering real-time decision-making related to management and service provisioning to end-users. This shift facilitates the transition from a specialized intelligence approach, where artificial intelligence (AI) algorithms handle isolated tasks, to artificial general intelligence (AGI)-driven networks, where agents possess broader reasoning capabilities and can manage diverse network functions. In this paper, we introduce a novel agentic paradigm that combines LLMs with real-time optimization algorithms towards Trustworthy AI, defined as symbiotic agents. Optimizers at the LLM's input-level provide bounded uncertainty steering for numerically precise tasks, whereas output-level optimizers supervised by the LLM enable adaptive real-time control. We design and implement two novel agent types including: (i) Radio Access Network optimizers, and (ii) multi-agent negotiators for Service-Level Agreements (SLAs). We further propose an end-to-end architecture for AGI networks and evaluate it on a 5G testbed capturing channel fluctuations from moving vehicles. Results show that symbiotic agents reduce decision errors fivefold compared to standalone LLM-based agents, while smaller language models (SLM) achieve similar accuracy with a 99.9% reduction in GPU resource overhead and in near-real-time loops of 82 ms. A multi-agent demonstration for collaborative RAN on the real-world testbed highlights significant flexibility in service-level agreement and resource allocation, reducing RAN over-utilization by approximately 44%. Drawing on our findings and open-source implementations, we introduce the symbiotic paradigm as the foundation for next-generation, AGI-driven networks-systems designed to remain adaptable, efficient, and trustworthy even as LLMs advance.

**arXiv ID:** 2507.17695
</details>

<details>
<summary><strong>L-MARS: Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search</strong> - Ziqi Wang, Boqin Yuan - [[pdf]](https://arxiv.org/pdf/2509.00761)</summary>

**Abstract:** We present L-MARS (Legal Multi-Agent Workflow with Orchestrated Reasoning and Agentic Search), a system that reduces hallucination and uncertainty in legal question answering through coordinated multi-agent reasoning and retrieval. Unlike single-pass retrieval-augmented generation (RAG), L-MARS decomposes queries into subproblems, issues targeted searches across heterogeneous sources (Serper web, local RAG, CourtListener case law), and employs a Judge Agent to verify sufficiency, jurisdiction, and temporal validity before answer synthesis. This iterative reasoning-search-verification loop maintains coherence, filters noisy evidence, and grounds answers in authoritative law. We evaluated L-MARS on LegalSearchQA, a new benchmark of 200 up-to-date multiple choice legal questions in 2025. Results show that L-MARS substantially improves factual accuracy, reduces uncertainty, and achieves higher preference scores from both human experts and LLM-based judges. Our work demonstrates that multi-agent reasoning with agentic search offers a scalable and reproducible blueprint for deploying LLMs in high-stakes domains requiring precise legal retrieval and deliberation.

**arXiv ID:** 2509.00761
</details>

<details>
<summary><strong>Towards Agentic OS: An LLM Agent Framework for Linux Schedulers</strong> - Yusheng Zheng, Yanpeng Hu, Wei Zhang, Andi Quinn - [[pdf]](https://arxiv.org/pdf/2509.01245)</summary>

**Abstract:** Operating system schedulers suffer from a fundamental semantic gap, where kernel policies fail to understand application-specific needs, leading to suboptimal performance. We introduce SchedCP, the first framework that enables fully autonomous Large Language Model (LLM) agents to safely and efficiently optimize Linux schedulers without human involvement. Our core insight is that the challenge is not merely to apply a better LLM, but to architect a decoupled control plane that separates the AI's role of semantic reasoning ("what to optimize") from the system's role of execution ("how to observe and act"). Implemented as Model Context Protocol(MCP) server, SchedCP provides a stable interface with three key services: a Workload Analysis Engine, an evolving Scheduler Policy Repository, and an Execution Verifier that validates all AI-generated code and configure before deployment with static and dynamic analysis.
We demonstrate this architecture's power with sched-agent, a multi-agent system that autonomously analyzes workloads, synthesizes custom eBPF scheduling policies, and deploys them via the sched\_ext infrastructure. Our evaluation shows that SchedCP achieves up to an 1.79x performance improvement, and a 13x cost reduction compared to naive agentic approaches, all while maintaining high success rate. By bridging the semantic gap, SchedCP democratizes expert-level system optimization and represents a step towards creating truly self-optimizing, application-aware operating systems. The code is open-sourced in this https URL

**arXiv ID:** 2509.01245
</details>

<details>
<summary><strong>MF-OML: Online Mean-Field Reinforcement Learning with Occupation Measures for Large Population Games</strong> - Anran Hu, Junzi Zhang - [[pdf]](https://arxiv.org/pdf/2405.00282)</summary>

**Abstract:** Reinforcement learning for multi-agent games has attracted lots of attention recently. However, given the challenge of solving Nash equilibria for large population games, existing works with guaranteed polynomial complexities either focus on variants of zero-sum and potential games, or aim at solving (coarse) correlated equilibria, or require access to simulators, or rely on certain assumptions that are hard to verify. This work proposes MF-OML (Mean-Field Occupation-Measure Learning), an online mean-field reinforcement learning algorithm for computing approximate Nash equilibria of large population sequential symmetric games. MF-OML is the first fully polynomial multi-agent reinforcement learning algorithm for provably solving Nash equilibria (up to mean-field approximation gaps that vanish as the number of players $N$ goes to infinity) beyond variants of zero-sum and potential games. When evaluated by the cumulative deviation from Nash equilibria, the algorithm is shown to achieve a high probability regret bound of $\tilde{O}(M^{3/4}+N^{-1/2}M)$ for games with the strong Lasry-Lions monotonicity condition, and a regret bound of $\tilde{O}(M^{11/12}+N^{- 1/6}M)$ for games with only the Lasry-Lions monotonicity condition, where $M$ is the total number of episodes and $N$ is the number of agents of the game. As a byproduct, we also obtain the first tractable globally convergent computational algorithm for computing approximate Nash equilibria of monotone mean-field games.

**arXiv ID:** 2405.00282
</details>

<details>
<summary><strong>Towards Agentic AI on Particle Accelerators</strong> - Antonin Sulc, Thorsten Hellert, Raimund Kammering, Hayden Hoschouer, Jason St. John - [[pdf]](https://arxiv.org/pdf/2409.06336)</summary>

**Abstract:** As particle accelerators grow in complexity, traditional control methods face increasing challenges in achieving optimal performance. This paper envisions a paradigm shift: a decentralized multi-agent framework for accelerator control, powered by Large Language Models (LLMs) and distributed among autonomous agents. We present a proposition of a self-improving decentralized system where intelligent agents handle high-level tasks and communication and each agent is specialized to control individual accelerator components.
This approach raises some questions: What are the future applications of AI in particle accelerators? How can we implement an autonomous complex system such as a particle accelerator where agents gradually improve through experience and human feedback? What are the implications of integrating a human-in-the-loop component for labeling operational data and providing expert guidance? We show three examples, where we demonstrate the viability of such architecture.

**arXiv ID:** 2409.06336
</details>

<details>
<summary><strong>Q-Learning-Driven Adaptive Rewiring for Cooperative Control in Heterogeneous Networks</strong> - Yi-Ning Weng, Hsuan-Wei Lee - [[pdf]](https://arxiv.org/pdf/2509.01057)</summary>

**Abstract:** Cooperation emergence in multi-agent systems represents a fundamental statistical physics problem where microscopic learning rules drive macroscopic collective behavior transitions. We propose a Q-learning-based variant of adaptive rewiring that builds on mechanisms studied in the literature. This method combines temporal difference learning with network restructuring so that agents can optimize strategies and social connections based on interaction histories. Through neighbor-specific Q-learning, agents develop sophisticated partnership management strategies that enable cooperator cluster formation, creating spatial separation between cooperative and defective regions. Using power-law networks that reflect real-world heterogeneous connectivity patterns, we evaluate emergent behaviors under varying rewiring constraint levels, revealing distinct cooperation patterns across parameter space rather than sharp thermodynamic transitions. Our systematic analysis identifies three behavioral regimes: a permissive regime (low constraints) enabling rapid cooperative cluster formation, an intermediate regime with sensitive dependence on dilemma strength, and a patient regime (high constraints) where strategic accumulation gradually optimizes network structure. Simulation results show that while moderate constraints create transition-like zones that suppress cooperation, fully adaptive rewiring enhances cooperation levels through systematic exploration of favorable network configurations. Quantitative analysis reveals that increased rewiring frequency drives large-scale cluster formation with power-law size distributions. Our results establish a new paradigm for understanding intelligence-driven cooperation pattern formation in complex adaptive systems, revealing how machine learning serves as an alternative driving force for spontaneous organization in multi-agent networks.

**arXiv ID:** 2509.01057
</details>

<details>
<summary><strong>Population-aware Online Mirror Descent for Mean-Field Games with Common Noise by Deep Reinforcement Learning</strong> - Zida Wu, Mathieu Lauriere, Matthieu Geist, Olivier Pietquin, Ankur Mehta - [[pdf]](https://arxiv.org/pdf/2509.03030)</summary>

**Abstract:** Mean Field Games (MFGs) offer a powerful framework for studying large-scale multi-agent systems. Yet, learning Nash equilibria in MFGs remains a challenging problem, particularly when the initial distribution is unknown or when the population is subject to common noise. In this paper, we introduce an efficient deep reinforcement learning (DRL) algorithm designed to achieve population-dependent Nash equilibria without relying on averaging or historical sampling, inspired by Munchausen RL and Online Mirror Descent. The resulting policy is adaptable to various initial distributions and sources of common noise. Through numerical experiments on seven canonical examples, we demonstrate that our algorithm exhibits superior convergence properties compared to state-of-the-art algorithms, particularly a DRL version of Fictitious Play for population-dependent policies. The performance in the presence of common noise underscores the robustness and adaptability of our approach.

**arXiv ID:** 2509.03030
</details>

<details>
<summary><strong>AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?</strong> - Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, Shuicheng Yan - [[pdf]](https://arxiv.org/pdf/2509.03312)</summary>

**Abstract:** Large Language Model (LLM)-based agentic systems, often comprising multiple models, complex tool invocations, and orchestration protocols, substantially outperform monolithic agents. Yet this very sophistication amplifies their fragility, making them more prone to system failure. Pinpointing the specific agent or step responsible for an error within long execution traces defines the task of agentic system failure attribution. Current state-of-the-art reasoning LLMs, however, remain strikingly inadequate for this challenge, with accuracy generally below 10%. To address this gap, we propose AgenTracer, the first automated framework for annotating failed multi-agent trajectories via counterfactual replay and programmed fault injection, producing the curated dataset TracerTraj. Leveraging this resource, we develop AgenTracer-8B, a lightweight failure tracer trained with multi-granular reinforcement learning, capable of efficiently diagnosing errors in verbose multi-agent interactions. On the Who&When benchmark, AgenTracer-8B outperforms giant proprietary LLMs like Gemini-2.5-Pro and Claude-4-Sonnet by up to 18.18%, setting a new standard in LLM agentic failure attribution. More importantly, AgenTracer-8B delivers actionable feedback to off-the-shelf multi-agent systems like MetaGPT and MaAS with 4.8-14.2% performance gains, empowering self-correcting and self-evolving agentic AI.

**arXiv ID:** 2509.03312
</details>

</details>

<details open>
<summary><h2>Other Agent Research (11 papers)</h2></summary>

<details>
<summary><strong>The Basic B*** Effect: The Use of LLM-based Agents Reduces the Distinctiveness and Diversity of People's Choices</strong> - Sandra C. Matz, C. Blaine Horton, Sofie Goethals - [[pdf]](https://arxiv.org/pdf/2509.02910)</summary>

**Abstract:** Large language models (LLMs) increasingly act on people's behalf: they write emails, buy groceries, and book restaurants. While the outsourcing of human decision-making to AI can be both efficient and effective, it raises a fundamental question: how does delegating identity-defining choices to AI reshape who people become? We study the impact of agentic LLMs on two identity-relevant outcomes: interpersonal distinctiveness - how unique a person's choices are relative to others - and intrapersonal diversity - the breadth of a single person's choices over time. Using real choices drawn from social-media behavior of 1,000 U.S. users (110,000 choices in total), we compare a generic and personalized agent to a human baseline. Both agents shift people's choices toward more popular options, reducing the distinctiveness of their behaviors and preferences. While the use of personalized agents tempers this homogenization (compared to the generic AI), it also more strongly compresses the diversity of people's preference portfolios by narrowing what they explore across topics and psychological affinities. Understanding how AI agents might flatten human experience, and how using generic versus personalized agents involves distinctiveness-diversity trade-offs, is critical for designing systems that augment rather than constrain human agency, and for safeguarding diversity in thought, taste, and expression.

**arXiv ID:** 2509.02910
</details>

<details>
<summary><strong>Simulacra Naturae: Generative Ecosystem driven by Agent-Based Simulations and Brain Organoid Collective Intelligence</strong> - Nefeli Manoudaki, Mert Toka, Iason Paterakis, Diarmid Flatley - [[pdf]](https://arxiv.org/pdf/2509.02924)</summary>

**Abstract:** Simulacra Naturae is a data-driven media installation that explores collective care through the entanglement of biological computation, material ecologies, and generative systems. The work translates pre-recorded neural activity from brain organoids, lab-grown three-dimensional clusters of neurons, into a multi-sensory environment composed of generative visuals, spatial audio, living plants, and fabricated clay artifacts. These biosignals, streamed through a real-time system, modulate emergent agent behaviors inspired by natural systems such as termite colonies and slime molds. Rather than using biosignals as direct control inputs, Simulacra Naturae treats organoid activity as a co-creative force, allowing neural rhythms to guide the growth, form, and atmosphere of a generative ecosystem. The installation features computationally fabricated clay prints embedded with solenoids, adding physical sound resonances to the generative surround composition. The spatial environment, filled with live tropical plants and a floor-level projection layer featuring real-time generative AI visuals, invites participants into a sensory field shaped by nonhuman cognition. By grounding abstract data in living materials and embodied experience, Simulacra Naturae reimagines visualization as a practice of care, one that decentralizes human agency and opens new spaces for ethics, empathy, and ecological attunement within hybrid computational systems.

**arXiv ID:** 2509.02924
</details>

<details>
<summary><strong>Automatic Differentiation of Agent-Based Models</strong> - Arnau Quera-Bofarull, Nicholas Bishop, Joel Dyer, Daniel Jarne Ornia, Anisoara Calinescu, Doyne Farmer, Michael Wooldridge - [[pdf]](https://arxiv.org/pdf/2509.03303)</summary>

**Abstract:** Agent-based models (ABMs) simulate complex systems by capturing the bottom-up interactions of individual agents comprising the system. Many complex systems of interest, such as epidemics or financial markets, involve thousands or even millions of agents. Consequently, ABMs often become computationally demanding and rely on the calibration of numerous free parameters, which has significantly hindered their widespread adoption. In this paper, we demonstrate that automatic differentiation (AD) techniques can effectively alleviate these computational burdens. By applying AD to ABMs, the gradients of the simulator become readily available, greatly facilitating essential tasks such as calibration and sensitivity analysis. Specifically, we show how AD enables variational inference (VI) techniques for efficient parameter calibration. Our experiments demonstrate substantial performance improvements and computational savings using VI on three prominent ABMs: Axtell's model of firms; Sugarscape; and the SIR epidemiological model. Our approach thus significantly enhances the practicality and scalability of ABMs for studying complex systems.

**arXiv ID:** 2509.03303
</details>

<details>
<summary><strong>Shutdownable Agents through POST-Agency</strong> - Elliott Thornley - [[pdf]](https://arxiv.org/pdf/2505.20203)</summary>

**Abstract:** Many fear that future artificial agents will resist shutdown. I present an idea - the POST-Agents Proposal - for ensuring that doesn't happen. I propose that we train agents to satisfy Preferences Only Between Same-Length Trajectories (POST). I then prove that POST - together with other conditions - implies Neutrality+: the agent maximizes expected utility, ignoring the probability distribution over trajectory-lengths. I argue that Neutrality+ keeps agents shutdownable and allows them to be useful.

**arXiv ID:** 2505.20203
</details>

<details>
<summary><strong>KIRETT: Knowledge-Graph-Based Smart Treatment Assistant for Intelligent Rescue Operations</strong> - Mubaris Nadeem, Johannes Zenkert, Lisa Bender, Christian Weber, Madjid Fathi - [[pdf]](https://arxiv.org/pdf/2508.07834)</summary>

**Abstract:** Over the years, the need for rescue operations throughout the world has increased rapidly. Demographic changes and the resulting risk of injury or health disorders form the basis for emergency calls. In such scenarios, first responders are in a rush to reach the patient in need, provide first aid, and save lives. In these situations, they must be able to provide personalized and optimized healthcare in the shortest possible time and estimate the patients condition with the help of freshly recorded vital data in an emergency situation. However, in such a timedependent situation, first responders and medical experts cannot fully grasp their knowledge and need assistance and recommendation for further medical treatments. To achieve this, on the spot calculated, evaluated, and processed knowledge must be made available to improve treatments by first responders. The Knowledge Graph presented in this article as a central knowledge representation provides first responders with an innovative knowledge management that enables intelligent treatment recommendations with an artificial intelligence-based pre-recognition of the situation.

**arXiv ID:** 2508.07834
</details>

<details>
<summary><strong>A DbC Inspired Neurosymbolic Layer for Trustworthy Agent Design</strong> - Claudiu Leoveanu-Condrei - [[pdf]](https://arxiv.org/pdf/2508.03665)</summary>

**Abstract:** Generative models, particularly Large Language Models (LLMs), produce fluent outputs yet lack verifiable guarantees. We adapt Design by Contract (DbC) and type-theoretic principles to introduce a contract layer that mediates every LLM call. Contracts stipulate semantic and type requirements on inputs and outputs, coupled with probabilistic remediation to steer generation toward compliance. The layer exposes the dual view of LLMs as semantic parsers and probabilistic black-box components. Contract satisfaction is probabilistic and semantic validation is operationally defined through programmer-specified conditions on well-typed data structures. More broadly, this work postulates that any two agents satisfying the same contracts are \emph{functionally equivalent} with respect to those contracts.

**arXiv ID:** 2508.03665
</details>

<details>
<summary><strong>Unsupervised Learning based Element Resource Allocation for Reconfigurable Intelligent Surfaces in mmWave Network</strong> - Pujitha Mamillapalli, Yoghitha Ramamoorthi, Abhinav Kumar, Tomoki Murakami, Tomoaki Ogawa, Yasushi Takatori - [[pdf]](https://arxiv.org/pdf/2509.03241)</summary>

**Abstract:** The increasing demand for high data rates and seamless connectivity in wireless systems has sparked significant interest in reconfigurable intelligent surfaces (RIS) and artificial intelligence-based wireless applications. RIS typically comprises passive reflective antenna elements that control the wireless propagation environment by adequately tuning the phase of the reflective elements. The allocation of RIS elements to multipleuser equipment (UEs) is crucial for efficiently utilizing RIS. In this work, we formulate a joint optimization problem that optimizes the RIS phase configuration and resource allocation under an $\alpha$-fair scheduling framework and propose an efficient way of allocating RIS elements. Conventional iterative optimization methods, however, suffer from exponentially increasing computational complexity as the number of RIS elements increases and also complicate the generation of training labels for supervised learning. To overcome these challenges, we propose a five-layer fully connected neural network (FNN) combined with a preprocessing technique to significantly reduce input dimensionality, lower computational complexity, and enhance scalability. The simulation results show that our proposed NN-based solution reduces computational overhead while significantly improving system throughput by 6.8% compared to existing RIS element allocation schemes. Furthermore, the proposed system achieves better performance while reducing computational complexity, making it significantly more scalable than the iterative optimization algorithms.

**arXiv ID:** 2509.03241
</details>

<details>
<summary><strong>Lessons Learned from Deploying Adaptive Machine Learning Agents with Limited Data for Real-time Cell Culture Process Monitoring</strong> - Thanh Tung Khuat, Johnny Peng, Robert Bassett, Ellen Otte, Bogdan Gabrys - [[pdf]](https://arxiv.org/pdf/2509.02606)</summary>

**Abstract:** This study explores the deployment of three machine learning (ML) approaches for real-time prediction of glucose, lactate, and ammonium concentrations in cell culture processes, using Raman spectroscopy as input features. The research addresses challenges associated with limited data availability and process variability, providing a comparative analysis of pretrained models, just-in-time learning (JITL), and online learning algorithms. Two industrial case studies are presented to evaluate the impact of varying bioprocess conditions on model performance. The findings highlight the specific conditions under which pretrained models demonstrate superior predictive accuracy and identify scenarios where JITL or online learning approaches are more effective for adaptive process monitoring. This study also highlights the critical importance of updating the deployed models/agents with the latest offline analytical measurements during bioreactor operations to maintain the model performance against the changes in cell growth behaviours and operating conditions throughout the bioreactor run. Additionally, the study confirms the usefulness of a simple mixture-of-experts framework in achieving enhanced accuracy and robustness for real-time predictions of metabolite concentrations based on Raman spectral data. These insights contribute to the development of robust strategies for the efficient deployment of ML models in dynamic and changing biomanufacturing environments.

**arXiv ID:** 2509.02606
</details>

<details>
<summary><strong>Separation of Three or More Autonomous Mobile Models under Hierarchical Schedulers</strong> - Shota Naito, Tsukasa Ninomiya, Koichi Wada - [[pdf]](https://arxiv.org/pdf/2508.19805)</summary>

**Abstract:** Understanding the computational power of mobile robot systems is a fundamental challenge in distributed computing. While prior work has focused on pairwise separations between models, we explore how robot capabilities, light observability, and scheduler synchrony interact in more complex ways.
We first show that the Exponential Times Expansion (ETE) problem is solvable only in the strongest model -- fully-synchronous robots with full mutual lights ($\mathcal{LUMT}^F$). We then introduce the Hexagonal Edge Traversal (HET) and TAR(d)* problems to demonstrate how internal memory and lights interact with synchrony: under weak synchrony, internal memory alone is insufficient, while full synchrony can substitute for both lights and memory.
In the asynchronous setting, we classify problems such as LP-MLCv, VEC, and ZCC to show fine-grained separations between $\mathcal{FSTA}$ and $\mathcal{FCOM}$ robots. We also analyze Vertex Traversal Rendezvous (VTR) and Leave Place Convergence (LP-Cv), illustrating the limitations of internal memory in symmetric settings.
These results extend the known separation map of 14 canonical robot models, revealing structural phenomena only visible through higher-order comparisons. Our work provides new impossibility criteria and deepens the understanding of how observability, memory, and synchrony collectively shape the computational power of mobile robots.

**arXiv ID:** 2508.19805
</details>

<details>
<summary><strong>2nd Place Solution for CVPR2024 E2E Challenge: End-to-End Autonomous Driving Using Vision Language Model</strong> - Zilong Guo, Yi Luo, Long Sha, Dongxu Wang, Panqu Wang, Chenyang Xu, Yi Yang - [[pdf]](https://arxiv.org/pdf/2509.02659)</summary>

**Abstract:** End-to-end autonomous driving has drawn tremendous attention recently. Many works focus on using modular deep neural networks to construct the end-to-end archi-tecture. However, whether using powerful large language models (LLM), especially multi-modality Vision Language Models (VLM) could benefit the end-to-end driving tasks remain a question. In our work, we demonstrate that combining end-to-end architectural design and knowledgeable VLMs yield impressive performance on the driving tasks. It is worth noting that our method only uses a single camera and is the best camera-only solution across the leaderboard, demonstrating the effectiveness of vision-based driving approach and the potential for end-to-end driving tasks.

**arXiv ID:** 2509.02659
</details>

<details>
<summary><strong>A Survey: Learning Embodied Intelligence from Physical Simulators and World Models</strong> - Xiaoxiao Long, Qingrui Zhao, Kaiwen Zhang, Zihao Zhang, Dingrui Wang, Yumeng Liu, Zhengjie Shu, Yi Lu, Shouzheng Wang, Xinzhe Wei, Wei Li, Wei Yin, Yao Yao, Jia Pan, Qiu Shen, Ruigang Yang, Xun Cao, Qionghai Dai - [[pdf]](https://arxiv.org/pdf/2507.00917)</summary>

**Abstract:** The pursuit of artificial general intelligence (AGI) has placed embodied intelligence at the forefront of robotics research. Embodied intelligence focuses on agents capable of perceiving, reasoning, and acting within the physical world. Achieving robust embodied intelligence requires not only advanced perception and control, but also the ability to ground abstract cognition in real-world interactions. Two foundational technologies, physical simulators and world models, have emerged as critical enablers in this quest. Physical simulators provide controlled, high-fidelity environments for training and evaluating robotic agents, allowing safe and efficient development of complex behaviors. In contrast, world models empower robots with internal representations of their surroundings, enabling predictive planning and adaptive decision-making beyond direct sensory input. This survey systematically reviews recent advances in learning embodied AI through the integration of physical simulators and world models. We analyze their complementary roles in enhancing autonomy, adaptability, and generalization in intelligent robots, and discuss the interplay between external simulation and internal modeling in bridging the gap between simulated training and real-world deployment. By synthesizing current progress and identifying open challenges, this survey aims to provide a comprehensive perspective on the path toward more capable and generalizable embodied AI systems. We also maintain an active repository that contains up-to-date literature and open-source projects at this https URL.

**arXiv ID:** 2507.00917
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>Integration of Computer Vision with Adaptive Control for Autonomous Driving Using ADORE</strong> - Abu Shad Ahammed, Md Shahi Amran Hossain, Sayeri Mukherjee, Roman Obermaisser, Md. Ziaur Rahman - [[pdf]](https://arxiv.org/pdf/2508.17985)</summary>

**Abstract:** Ensuring safety in autonomous driving requires a seamless integration of perception and decision making under uncertain conditions. Although computer vision (CV) models such as YOLO achieve high accuracy in detecting traffic signs and obstacles, their performance degrades in drift scenarios caused by weather variations or unseen objects. This work presents a simulated autonomous driving system that combines a context aware CV model with adaptive control using the ADORE framework. The CARLA simulator was integrated with ADORE via the ROS bridge, allowing real-time communication between perception, decision, and control modules. A simulated test case was designed in both clear and drift weather conditions to demonstrate the robust detection performance of the perception model while ADORE successfully adapted vehicle behavior to speed limits and obstacles with low response latency. The findings highlight the potential of coupling deep learning-based perception with rule-based adaptive decision making to improve automotive safety critical system.

**arXiv ID:** 2508.17985
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (18 papers)</h2></summary>

<details>
<summary><strong>Plan Verification for LLM-Based Embodied Task Completion Agents</strong> - Ananth Hariharan, Vardhan Dongre, Dilek Hakkani-Tür, Gokhan Tur - [[pdf]](https://arxiv.org/pdf/2509.02761)</summary>

**Abstract:** Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI.

**arXiv ID:** 2509.02761
</details>

<details>
<summary><strong>VendiRL: A Framework for Self-Supervised Reinforcement Learning of Diversely Diverse Skills</strong> - Erik M. Lintunen - [[pdf]](https://arxiv.org/pdf/2509.02930)</summary>

**Abstract:** In self-supervised reinforcement learning (RL), one of the key challenges is learning a diverse set of skills to prepare agents for unknown future tasks. Despite impressive advances, scalability and evaluation remain prevalent issues. Regarding scalability, the search for meaningful skills can be obscured by high-dimensional feature spaces, where relevant features may vary across downstream task domains. For evaluating skill diversity, defining what constitutes "diversity" typically requires a hard commitment to a specific notion of what it means for skills to be diverse, potentially leading to inconsistencies in how skill diversity is understood, making results across different approaches hard to compare, and leaving many forms of diversity unexplored. To address these issues, we adopt a measure of sample diversity that translates ideas from ecology to machine learning -- the Vendi Score -- allowing the user to specify and evaluate any desired form of diversity. We demonstrate how this metric facilitates skill evaluation and introduce VendiRL, a unified framework for learning diversely diverse sets of skills. Given distinct similarity functions, VendiRL motivates distinct forms of diversity, which could support skill-diversity pretraining in new and richly interactive environments where optimising for various forms of diversity may be desirable.

**arXiv ID:** 2509.02930
</details>

<details>
<summary><strong>A Hierarchical Deep Reinforcement Learning Framework for Traffic Signal Control with Predictable Cycle Planning</strong> - Hankang Gu, Yuli Zhang, Chengming Wang, Ruiyuan Jiang, Ziheng Qiao, Pengfei Fan, Dongyao Jia - [[pdf]](https://arxiv.org/pdf/2509.03118)</summary>

**Abstract:** Deep reinforcement learning (DRL) has become a popular approach in traffic signal control (TSC) due to its ability to learn adaptive policies from complex traffic environments. Within DRL-based TSC methods, two primary control paradigms are ``choose phase" and ``switch" strategies. Although the agent in the choose phase paradigm selects the next active phase adaptively, this paradigm may result in unexpected phase sequences for drivers, disrupting their anticipation and potentially compromising safety at intersections. Meanwhile, the switch paradigm allows the agent to decide whether to switch to the next predefined phase or extend the current phase. While this structure maintains a more predictable order, it can lead to unfair and inefficient phase allocations, as certain movements may be extended disproportionately while others are neglected. In this paper, we propose a DRL model, named Deep Hierarchical Cycle Planner (DHCP), to allocate the traffic signal cycle duration hierarchically. A high-level agent first determines the split of the total cycle time between the North-South (NS) and East-West (EW) directions based on the overall traffic state. Then, a low-level agent further divides the allocated duration within each major direction between straight and left-turn movements, enabling more flexible durations for the two movements. We test our model on both real and synthetic road networks, along with multiple sets of real and synthetic traffic flows. Empirical results show our model achieves the best performance over all datasets against baselines.

**arXiv ID:** 2509.03118
</details>

<details>
<summary><strong>Search-Based Credit Assignment for Offline Preference-Based Reinforcement Learning</strong> - Xiancheng Gao, Yufeng Shi, Wengang Zhou, Houqiang Li - [[pdf]](https://arxiv.org/pdf/2508.15327)</summary>

**Abstract:** Offline reinforcement learning refers to the process of learning policies from fixed datasets, without requiring additional environment interaction. However, it often relies on well-defined reward functions, which are difficult and expensive to design. Human feedback is an appealing alternative, but its two common forms, expert demonstrations and preferences, have complementary limitations. Demonstrations provide stepwise supervision, but they are costly to collect and often reflect limited expert behavior modes. In contrast, preferences are easier to collect, but it is unclear which parts of a behavior contribute most to a trajectory segment, leaving credit assignment unresolved. In this paper, we introduce a Search-Based Preference Weighting (SPW) scheme to unify these two feedback sources. For each transition in a preference labeled trajectory, SPW searches for the most similar state-action pairs from expert demonstrations and directly derives stepwise importance weights based on their similarity scores. These weights are then used to guide standard preference learning, enabling more accurate credit assignment that traditional approaches struggle to achieve. We demonstrate that SPW enables effective joint learning from preferences and demonstrations, outperforming prior methods that leverage both feedback types on challenging robot manipulation tasks.

**arXiv ID:** 2508.15327
</details>

<details>
<summary><strong>Impoola: The Power of Average Pooling for Image-Based Deep Reinforcement Learning</strong> - Raphael Trumpp, Ansgar Schäfftlein, Mirco Theile, Marco Caccamo - [[pdf]](https://arxiv.org/pdf/2503.05546)</summary>

**Abstract:** As image-based deep reinforcement learning tackles more challenging tasks, increasing model size has become an important factor in improving performance. Recent studies achieved this by focusing on the parameter efficiency of scaled networks, typically using Impala-CNN, a 15-layer ResNet-inspired network, as the image encoder. However, while Impala-CNN evidently outperforms older CNN architectures, potential advancements in network design for deep reinforcement learning-specific image encoders remain largely unexplored. We find that replacing the flattening of output feature maps in Impala-CNN with global average pooling leads to a notable performance improvement. This approach outperforms larger and more complex models in the Procgen Benchmark, particularly in terms of generalization. We call our proposed encoder model Impoola-CNN. A decrease in the network's translation sensitivity may be central to this improvement, as we observe the most significant gains in games without agent-centered observations. Our results demonstrate that network scaling is not just about increasing model size - efficient network design is also an essential factor. We make our code available at this https URL.

**arXiv ID:** 2503.05546
</details>

<details>
<summary><strong>When a Reinforcement Learning Agent Encounters Unknown Unknowns</strong> - Juntian Zhu, Miguel de Carvalho, Zhouwang Yang, Fengxiang He - [[pdf]](https://arxiv.org/pdf/2505.13188)</summary>

**Abstract:** An AI agent might surprisingly find she has reached an unknown state which she has never been aware of -- an unknown unknown. We mathematically ground this scenario in reinforcement learning: an agent, after taking an action calculated from value functions $Q$ and $V$ defined on the {\it {aware domain}}, reaches a state out of the domain. To enable the agent to handle this scenario, we propose an {\it episodic Markov decision {process} with growing awareness} (EMDP-GA) model, taking a new {\it noninformative value expansion} (NIVE) approach to expand value functions to newly aware areas: when an agent arrives at an unknown unknown, value functions $Q$ and $V$ whereon are initialised by noninformative beliefs -- the averaged values on the aware domain. This design is out of respect for the complete absence of knowledge in the newly discovered state. The upper confidence bound momentum Q-learning is then adapted to the growing awareness for training the EMDP-GA model. We prove that (1) the regret of our approach is asymptotically consistent with the state of the art (SOTA) without exposure to unknown unknowns in an extremely uncertain environment, and (2) our computational complexity and space complexity are comparable with the SOTA -- these collectively suggest that though an unknown unknown is surprising, it will be asymptotically properly discovered with decent speed and an affordable cost.

**arXiv ID:** 2505.13188
</details>

<details>
<summary><strong>NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning</strong> - Wei Liu, Siya Qi, Xinyu Wang, Chen Qian, Yali Du, Yulan He - [[pdf]](https://arxiv.org/pdf/2505.16022)</summary>

**Abstract:** Recent advances such as DeepSeek R1-Zero highlight the effectiveness of incentive training, a reinforcement learning paradigm that computes rewards solely based on the final answer part of a language model's output, thereby encouraging the generation of intermediate reasoning steps. However, these methods fundamentally rely on external verifiers, which limits their applicability to domains like mathematics and coding where such verifiers are readily available. Although reward models can serve as verifiers, they require high-quality annotated data and are costly to train. In this work, we propose NOVER, NO-VERifier Reinforcement Learning, a general reinforcement learning framework that requires only standard supervised fine-tuning data with no need for an external verifier. NOVER enables incentive training across a wide range of text-to-text tasks and outperforms the model of the same size distilled from large reasoning models such as DeepSeek R1 671B by 7.7 percent. Moreover, the flexibility of NOVER enables new possibilities for optimizing large language models, such as inverse incentive training.

**arXiv ID:** 2505.16022
</details>

<details>
<summary><strong>RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Lifelong Learning in Physical Embodied Systems</strong> - Mingcong Lei, Honghao Cai, Binbin Que, Zezhou Cui, Liangchen Tan, Junkun Hong, Gehan Hu, Shuangyu Zhu, Yimou Wu, Shaohan Jiang, Ge Wang, Zhen Li, Shuguang Cui, Yiming Zhao, Yatong Han - [[pdf]](https://arxiv.org/pdf/2508.01415)</summary>

**Abstract:** We present RoboMemory, a brain-inspired multi-memory framework for lifelong learning in physical embodied systems, addressing critical challenges in real-world environments: continuous learning, multi-module memory latency, task correlation capture, and infinite-loop mitigation in closed-loop planning. Grounded in cognitive neuroscience, it integrates four core modules: the Information Preprocessor (thalamus-like), the Lifelong Embodied Memory System (hippocampus-like), the Closed-Loop Planning Module (prefrontal lobe-like), and the Low-Level Executer (cerebellum-like) to enable long-term planning and cumulative learning. The Lifelong Embodied Memory System, central to the framework, alleviates inference speed issues in complex memory frameworks via parallelized updates/retrieval across Spatial, Temporal, Episodic, and Semantic submodules. It incorporates a dynamic Knowledge Graph (KG) and consistent architectural design to enhance memory consistency and scalability. Evaluations on EmbodiedBench show RoboMemory outperforms the open-source baseline (Qwen2.5-VL-72B-Ins) by 25% in average success rate and surpasses the closed-source State-of-the-Art (SOTA) (Claude3.5-Sonnet) by 5%, establishing new SOTA. Ablation studies validate key components (critic, spatial memory, long-term memory), while real-world deployment confirms its lifelong learning capability with significantly improved success rates across repeated tasks. RoboMemory alleviates high latency challenges with scalability, serving as a foundational reference for integrating multi-modal memory systems in physical robots.

**arXiv ID:** 2508.01415
</details>

<details>
<summary><strong>MagicGUI: A Foundational Mobile GUI Agent with Scalable Data Pipeline and Reinforcement Fine-tuning</strong> - Liujian Tang, Shaokang Dong, Yijia Huang, Minqi Xiang, Hongtao Ruan, Bin Wang, Shuo Li, Zhiheng Xi, Zhihui Cao, Hailiang Pang, Heng Kong, He Yang, Mingxu Chai, Zhilin Gao, Xingyu Liu, Yingnan Fu, Jiaming Liu, Xuanjing Huang, Yu-Gang Jiang, Tao Gui, Qi Zhang, Kang Wang, Yunke Zhang, Yuran Wang - [[pdf]](https://arxiv.org/pdf/2508.03700)</summary>

**Abstract:** This paper presents MagicGUI, a foundational mobile GUI agent designed to address critical challenges in perception, grounding, and reasoning within real-world mobile GUI environments. The framework is underpinned by following six key components: (1) a comprehensive and accurate dataset, constructed via the scalable GUI Data Pipeline, which aggregates the largest and most diverse GUI-centric multimodal data to date from open-source repositories, automated crawling, and targeted manual annotation; (2) enhanced perception and grounding capabilities, facilitating fine-grained multimodal alignment for UI element referencing, grounding, and screen comprehension; (3) a comprehensive and unified action space, encompassing both fundamental UI operations and complex interactive intents to support human-agent interactions; (4) planning-oriented reasoning mechanisms that enable the model to decompose complex user instructions into sequential actions with explicit intermediate meta-paln reasoning; (5) an iterative two-stage training procedure, combining large-scale continue pre-training on 7.8M samples with reinforcement fine-tuning utilizing a spatially enhanced composite reward and dual filtering strategy; and (6) competitive performance on both the proprietary Magic-RICH benchmark and over a dozen public benchmarks, achieving superior performance across GUI perception and agent tasks, while demonstrating robust generalization and real-world deployment potential in practical mobile GUI scenarios, as detailed in Figure 1.

**arXiv ID:** 2508.03700
</details>

<details>
<summary><strong>Embodied AI: Emerging Risks and Opportunities for Policy Action</strong> - Jared Perlo, Alexander Robey, Fazl Barez, Luciano Floridi, Jakob Mökander - [[pdf]](https://arxiv.org/pdf/2509.00117)</summary>

**Abstract:** The field of embodied AI (EAI) is rapidly advancing. Unlike virtual AI, EAI systems can exist in, learn from, reason about, and act in the physical world. With recent advances in AI models and hardware, EAI systems are becoming increasingly capable across wider operational domains. While EAI systems can offer many benefits, they also pose significant risks, including physical harm from malicious use, mass surveillance, as well as economic and societal disruption. These risks require urgent attention from policymakers, as existing policies governing industrial robots and autonomous vehicles are insufficient to address the full range of concerns EAI systems present. To help address this issue, this paper makes three contributions. First, we provide a taxonomy of the physical, informational, economic, and social risks EAI systems pose. Second, we analyze policies in the US, EU, and UK to assess how existing frameworks address these risks and to identify critical gaps. We conclude by offering policy recommendations for the safe and beneficial deployment of EAI systems, such as mandatory testing and certification schemes, clarified liability frameworks, and strategies to manage EAI's potentially transformative economic and societal impacts.

**arXiv ID:** 2509.00117
</details>

<details>
<summary><strong>A Reliable Self-Organized Distributed Complex Network for Communication of Smart Agents</strong> - Mehdi Bakhshipoor, Yousef Azizi, Seyed Ehsan Nedaaee Oskoee - [[pdf]](https://arxiv.org/pdf/2503.07702)</summary>

**Abstract:** Collaboration is a fundamental and essential characteristic of many complex systems, ranging from ant colonies to human societies. Each component within a complex system interacts with others, even at a distance, to accomplish a given task. A network of collaboration can be defined to study the collective behavior of such systems within the framework of complex networks. The nodes in these networks may represent simple organisms or more sophisticated intelligent agents, such as humans. In this study, we utilize intelligent agents (nodes) trained through reinforcement learning techniques to establish connections with their neighbors, ultimately leading to the emergence of a large-scale communication cluster. Notably, there is no centralized administrator; instead, agents must adjust their connections based on information obtained from local observations. The connection strategy is formulated using a physical Hamiltonian, thereby categorizing this intelligent system under the paradigm of "Physics-Guided Machine Learning". The resulting self-organized distributed complex network has numerous industrial applications, including constructing Internet of Things (IoT) networks. The design of such networks often encounters challenges, the most critical of which is ensuring effective connectivity for reliable communication while optimizing energy consumption. IoT networks are inherently dynamic in many real-world applications, such as Vehicle Ad-hoc Networks (VANETs), where nodes are mobile, and the connection topology evolves rapidly over time. These systems require a robust and rapidly self-organizing communication network. Our findings demonstrate that the proposed intelligent agents facilitate the formation of self-organized complex networks capable of maintaining network-wide connectivity across various dynamic scenarios while simultaneously optimizing average electrical power consumption.

**arXiv ID:** 2503.07702
</details>

<details>
<summary><strong>Design and Optimization of Reinforcement Learning-Based Agents in Text-Based Games</strong> - Haonan Wang, Mingjia Zhao, Junfeng Sun, Wei Liu - [[pdf]](https://arxiv.org/pdf/2509.03479)</summary>

**Abstract:** As AI technology advances, research in playing text-based games with agents has becomeprogressively popular. In this paper, a novel approach to agent design and agent learning ispresented with the context of reinforcement learning. A model of deep learning is first applied toprocess game text and build a world model. Next, the agent is learned through a policy gradient-based deep reinforcement learning method to facilitate conversion from state value to optimal this http URL enhanced agent works better in several text-based game experiments and significantlysurpasses previous agents on game completion ratio and win rate. Our study introduces novelunderstanding and empirical ground for using reinforcement learning for text games and sets thestage for developing and optimizing reinforcement learning agents for more general domains andproblems.

**arXiv ID:** 2509.03479
</details>

<details>
<summary><strong>Power Grid Control with Graph-Based Distributed Reinforcement Learning</strong> - Carlo Fabrizio, Gianvito Losapio, Marco Mussi, Alberto Maria Metelli, Marcello Restelli - [[pdf]](https://arxiv.org/pdf/2509.02861)</summary>

**Abstract:** The necessary integration of renewable energy sources, combined with the expanding scale of power networks, presents significant challenges in controlling modern power grids. Traditional control systems, which are human and optimization-based, struggle to adapt and to scale in such an evolving context, motivating the exploration of more dynamic and distributed control strategies. This work advances a graph-based distributed reinforcement learning framework for real-time, scalable grid management. The proposed architecture consists of a network of distributed low-level agents acting on individual power lines and coordinated by a high-level manager agent. A Graph Neural Network (GNN) is employed to encode the network's topological information within the single low-level agent's observation. To accelerate convergence and enhance learning stability, the framework integrates imitation learning and potential-based reward shaping. In contrast to conventional decentralized approaches that decompose only the action space while relying on global observations, this method also decomposes the observation space. Each low-level agent acts based on a structured and informative local view of the environment constructed through the GNN. Experiments on the Grid2Op simulation environment show the effectiveness of the approach, which consistently outperforms the standard baseline commonly adopted in the field. Additionally, the proposed model proves to be much more computationally efficient than the simulation-based Expert method.

**arXiv ID:** 2509.02861
</details>

<details>
<summary><strong>Exploring a Graph-based Approach to Offline Reinforcement Learning for Sepsis Treatment</strong> - Taisiya Khakharova, Lucas Sakizloglou, Leen Lambers - [[pdf]](https://arxiv.org/pdf/2509.03393)</summary>

**Abstract:** Sepsis is a serious, life-threatening condition. When treating sepsis, it is challenging to determine the correct amount of intravenous fluids and vasopressors for a given patient. While automated reinforcement learning (RL)-based methods have been used to support these decisions with promising results, previous studies have relied on relational data. Given the complexity of modern healthcare data, representing data as a graph may provide a more natural and effective approach. This study models patient data from the well-known MIMIC-III dataset as a heterogeneous graph that evolves over time. Subsequently, we explore two Graph Neural Network architectures - GraphSAGE and GATv2 - for learning patient state representations, adopting the approach of decoupling representation learning from policy learning. The encoders are trained to produce latent state representations, jointly with decoders that predict the next patient state. These representations are then used for policy learning with the dBCQ algorithm. The results of our experimental evaluation confirm the potential of a graph-based approach, while highlighting the complexity of representation learning in this domain.

**arXiv ID:** 2509.03393
</details>

<details>
<summary><strong>MPCritic: A plug-and-play MPC architecture for reinforcement learning</strong> - Nathan P. Lawrence, Thomas Banker, Ali Mesbah - [[pdf]](https://arxiv.org/pdf/2504.01086)</summary>

**Abstract:** The reinforcement learning (RL) and model predictive control (MPC) communities have developed vast ecosystems of theoretical approaches and computational tools for solving optimal control problems. Given their conceptual similarities but differing strengths, there has been increasing interest in synergizing RL and MPC. However, existing approaches tend to be limited for various reasons, including computational cost of MPC in an RL algorithm and software hurdles towards seamless integration of MPC and RL tools. These challenges often result in the use of "simple" MPC schemes or RL algorithms, neglecting the state-of-the-art in both areas. This paper presents MPCritic, a machine learning-friendly architecture that interfaces seamlessly with MPC tools. MPCritic utilizes the loss landscape defined by a parameterized MPC problem, focusing on "soft" optimization over batched training steps; thereby updating the MPC parameters while avoiding costly minimization and parametric sensitivities. Since the MPC structure is preserved during training, an MPC agent can be readily used for online deployment, where robust constraint satisfaction is paramount. We demonstrate the versatility of MPCritic, in terms of MPC architectures and RL algorithms that it can accommodate, on classic control benchmarks.

**arXiv ID:** 2504.01086
</details>

<details>
<summary><strong>Convergence of regularized agent-state-based Q-learning in POMDPs</strong> - Amit Sinha, Matthieu Geist, Aditya Mahajan - [[pdf]](https://arxiv.org/pdf/2508.21314)</summary>

**Abstract:** In this paper, we present a framework to understand the convergence of commonly used Q-learning reinforcement learning algorithms in practice. Two salient features of such algorithms are: (i)~the Q-table is recursively updated using an agent state (such as the state of a recurrent neural network) which is not a belief state or an information state and (ii)~policy regularization is often used to encourage exploration and stabilize the learning algorithm. We investigate the simplest form of such Q-learning algorithms which we call regularized agent-state-based Q-learning (RASQL) and show that it converges under mild technical conditions to the fixed point of an appropriately defined regularized MDP, which depends on the stationary distribution induced by the behavioral policy. We also show that a similar analysis continues to work for a variant of RASQL that learns periodic policies. We present numerical examples to illustrate that the empirical convergence behavior matches with the proposed theoretical limit.

**arXiv ID:** 2508.21314
</details>

<details>
<summary><strong>SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning</strong> - Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Xiaosen Zheng, Zejun Ma, Bo An - [[pdf]](https://arxiv.org/pdf/2509.02479)</summary>

**Abstract:** Large Language Models (LLMs) can significantly improve their reasoning capabilities by interacting with external tools, a paradigm known as Tool-Integrated Reasoning (TIR). However, extending TIR to multi-turn scenarios using Reinforcement Learning (RL) is often hindered by training instability and performance collapse. We identify that such instability is primarily caused by a distributional drift from external tool feedback, leading to the generation of low-probability tokens. This issue compounds over successive turns, causing catastrophic gradient norm explosions that derail the training process. To address this challenge, we introduce SimpleTIR , a plug-and-play algorithm that stabilizes multi-turn TIR training. Its core strategy is to identify and filter out trajectories containing void turns, i.e., turns that yield neither a code block nor a final answer. By removing these problematic trajectories from the policy update, SimpleTIR effectively blocks the harmful, high-magnitude gradients, thus stabilizing the learning dynamics. Extensive experiments show that SimpleTIR achieves state-of-the-art performance on challenging math reasoning benchmarks, notably elevating the AIME24 score from a text-only baseline of 22.1 to 50.5 when starting from the Qwen2.5-7B base model. Furthermore, by avoiding the constraints of supervised fine-tuning, SimpleTIR encourages the model to discover diverse and sophisticated reasoning patterns, such as self-correction and cross-validation.

**arXiv ID:** 2509.02479
</details>

<details>
<summary><strong>CTBC: Contact-Triggered Blind Climbing for Wheeled Bipedal Robots with Instruction Learning and Reinforcement Learning</strong> - Rankun Li, Hao Wang, Qi Li, Zhuo Han, Yifei Chu, Linqi Ye, Wende Xie, Wenlong Liao - [[pdf]](https://arxiv.org/pdf/2509.02986)</summary>

**Abstract:** In recent years, wheeled bipedal robots have gained increasing attention due to their advantages in mobility, such as high-speed locomotion on flat terrain. However, their performance on complex environments (e.g., staircases) remains inferior to that of traditional legged robots. To overcome this limitation, we propose a general contact-triggered blind climbing (CTBC) framework for wheeled bipedal robots. Upon detecting wheel-obstacle contact, the robot triggers a leg-lifting motion to overcome the obstacle. By leveraging a strongly-guided feedforward trajectory, our method enables the robot to rapidly acquire agile leg-lifting skills, significantly enhancing its capability to traverse unstructured terrains. The approach has been experimentally validated and successfully deployed on LimX Dynamics' wheeled bipedal robot, Tron1. Real-world tests demonstrate that Tron1 can reliably climb obstacles well beyond its wheel radius using only proprioceptive feedback.

**arXiv ID:** 2509.02986
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
