# Agent arXiv Daily

**Last Updated:** 2026-02-18 03:15:17

**Total Papers:** 63

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (2 papers)</h2></summary>

<details>
<summary><strong>NeuroChat: A Neuroadaptive AI Chatbot for Customizing Learning Experiences</strong> - Dünya Baradari, Nataliya Kosmyna, Oscar Petrov, Rebecah Kaplun, Pattie Maes - [[pdf]](https://arxiv.org/pdf/2503.07599)</summary>

**Abstract:** Generative AI is transforming education by enabling personalized, on-demand learning experiences. However, current AI systems lack awareness of the learner's cognitive state, limiting their adaptability. Meanwhile, electroencephalography (EEG)-based neuroadaptive systems have shown promise in enhancing engagement through real-time physiological feedback. This paper presents NeuroChat, a neuroadaptive AI tutor that integrates real-time EEG-based engagement tracking with generative AI to adapt its responses. NeuroChat continuously monitors a learner's cognitive engagement and dynamically adjusts content complexity, tone, and response style in a closed-loop interaction. In a within-subjects study (n=24), NeuroChat significantly increased both EEG-measured and self-reported engagement compared to a non-adaptive chatbot. However, no significant differences in short-term learning outcomes were observed. These findings demonstrate the feasibility of real-time cognitive feedback in LLMs, highlighting new directions for adaptive learning, AI tutoring, and deeper personalization in human-AI interaction.

**arXiv ID:** 2503.07599
</details>

<details>
<summary><strong>Online Fine-Tuning of Pretrained Controllers for Autonomous Driving via Real-Time Recurrent RL</strong> - Julian Lemmel, Felix Resch, Mónika Farsang, Ramin Hasani, Daniela Rus, Radu Grosu - [[pdf]](https://arxiv.org/pdf/2602.02236)</summary>

**Abstract:** Deploying pretrained policies in real-world applications presents substantial challenges that fundamentally limit the practical applicability of learning-based control systems. When autonomous systems encounter environmental changes in system dynamics, sensor drift, or task objectives, fixed policies rapidly degrade in performance. We show that employing Real-Time Recurrent Reinforcement Learning (RTRRL), a biologically plausible algorithm for online adaptation, can effectively fine-tune a pretrained policy to improve autonomous agents' performance on driving tasks. We further show that RTRRL synergizes with a recent biologically inspired recurrent network model, the Liquid-Resistance Liquid-Capacitance RNN. We demonstrate the effectiveness of this closed-loop approach in a simulated CarRacing environment and in a real-world line-following task with a RoboRacer car equipped with an event camera.

**arXiv ID:** 2602.02236
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>Secure and Energy-Efficient Wireless Agentic AI Networks</strong> - Yuanyan Song, Kezhi Wang, Xinmian Xu - [[pdf]](https://arxiv.org/pdf/2602.15212)</summary>

**Abstract:** In this paper, we introduce a secure wireless agentic AI network comprising one supervisor AI agent and multiple other AI agents to provision quality of service (QoS) for users' reasoning tasks while ensuring confidentiality of private knowledge and reasoning outcomes. Specifically, the supervisor AI agent can dynamically assign other AI agents to participate in cooperative reasoning, while the unselected AI agents act as friendly jammers to degrade the eavesdropper's interception performance. To extend the service duration of AI agents, an energy minimization problem is formulated that jointly optimizes AI agent selection, base station (BS) beamforming, and AI agent transmission power, subject to latency and reasoning accuracy constraints. To address the formulated problem, we propose two resource allocation schemes, ASC and LAW, which first decompose it into three sub-problems. Specifically, ASC optimizes each sub-problem iteratively using the proposed alternating direction method of multipliers (ADMM)-based algorithm, semi-definite relaxation (SDR), and successive convex approximation (SCA), while LAW tackles each sub-problem using the proposed large language model (LLM) optimizer within an agentic workflow. The experimental results show that the proposed solutions can reduce network energy consumption by up to 59.1% compared to other benchmark schemes. Furthermore, the proposed schemes are validated using a practical agentic AI system based on Qwen, demonstrating satisfactory reasoning accuracy across various public benchmarks.

**arXiv ID:** 2602.15212
</details>

<details>
<summary><strong>GRACE: an Agentic AI for Particle Physics Experiment Design and Simulation</strong> - Justin Hill, Hong Joo Ryoo - [[pdf]](https://arxiv.org/pdf/2602.15039)</summary>

**Abstract:** We present GRACE, a simulation-native agent for autonomous experimental design in high-energy and nuclear physics. Given multimodal input in the form of a natural-language prompt or a published experimental paper, the agent extracts a structured representation of the experiment, constructs a runnable toy simulation, and autonomously explores design modifications using first-principles Monte Carlo methods. Unlike agentic systems focused on operational control or execution of predefined procedures, GRACE addresses the upstream problem of experimental design: proposing non-obvious modifications to detector geometry, materials, and configurations that improve physics performance under physical and practical constraints. The agent evaluates candidate designs through repeated simulation, physics-motivated utility functions, and budget-aware escalation from fast parametric models to full Geant4 simulations, while maintaining strict reproducibility and provenance tracking. We demonstrate the framework on historical experimental setups, showing that the agent can identify optimization directions that align with known upgrade priorities, using only baseline simulation inputs. We also conducted a benchmark in which the agent identified the setup and proposed improvements from a suite of natural language prompts, with some supplied with a relevant physics research paper, of varying high energy physics (HEP) problem settings. This work establishes experimental design as a constrained search problem under physical law and introduces a new benchmark for autonomous, simulation-driven scientific reasoning in complex instruments.

**arXiv ID:** 2602.15039
</details>

<details>
<summary><strong>Improving MLLMs in Embodied Exploration and Question Answering with Human-Inspired Memory Modeling</strong> - Ji Li, Jing Xia, Mingyi Li, Shiyan Hu - [[pdf]](https://arxiv.org/pdf/2602.15513)</summary>

**Abstract:** Deploying Multimodal Large Language Models as the brain of embodied agents remains challenging, particularly under long-horizon observations and limited context budgets. Existing memory assisted methods often rely on textual summaries, which discard rich visual and spatial details and remain brittle in non-stationary environments. In this work, we propose a non-parametric memory framework that explicitly disentangles episodic and semantic memory for embodied exploration and question answering. Our retrieval-first, reasoning-assisted paradigm recalls episodic experiences via semantic similarity and verifies them through visual reasoning, enabling robust reuse of past observations without rigid geometric alignment. In parallel, we introduce a program-style rule extraction mechanism that converts experiences into structured, reusable semantic memory, facilitating cross-environment generalization. Extensive experiments demonstrate state-of-the-art performance on embodied question answering and exploration benchmarks, yielding a 7.3% gain in LLM-Match and an 11.4% gain in LLM MatchXSPL on A-EQA, as well as +7.7% success rate and +6.8% SPL on GOAT-Bench. Analyses reveal that our episodic memory primarily improves exploration efficiency, while semantic memory strengthens complex reasoning of embodied agents.

**arXiv ID:** 2602.15513
</details>

<details>
<summary><strong>Prover Agent: An Agent-Based Framework for Formal Mathematical Proofs</strong> - Kaito Baba, Chaoran Liu, Shuhei Kurita, Akiyoshi Sannai - [[pdf]](https://arxiv.org/pdf/2506.19923)</summary>

**Abstract:** We present Prover Agent, a novel AI agent for automated theorem proving that integrates large language models (LLMs) with a formal proof assistant, Lean. Prover Agent coordinates an informal reasoning LLM, a formal prover model, and feedback from Lean while also generating auxiliary lemmas. These auxiliary lemmas are not limited to subgoals in the formal proof but can also include special cases or potentially useful facts derived from the assumptions, which help in discovering a viable proof strategy. It achieves an 88.1% success rate on MiniF2F and solves 25 problems on the PutnamBench with a smaller sample budget than previous approaches, establishing a new state-of-the-art on both benchmarks among methods using small language models (SLMs). We also present theoretical analyses and case studies that illustrate how these generated lemmas contribute to solving challenging problems. Our code is publicly available at this https URL.

**arXiv ID:** 2506.19923
</details>

<details>
<summary><strong>MARS: Modular Agent with Reflective Search for Automated AI Research</strong> - Jiefeng Chen, Bhavana Dalvi Mishra, Jaehyun Nam, Rui Meng, Tomas Pfister, Jinsung Yoon - [[pdf]](https://arxiv.org/pdf/2602.02660)</summary>

**Abstract:** Automating AI research differs from general software engineering due to computationally expensive evaluation (e.g., model training) and opaque performance attribution. Current LLM-based agents struggle here, often generating monolithic scripts that ignore execution costs and causal factors. We introduce MARS (Modular Agent with Reflective Search), a framework optimized for autonomous AI research. MARS relies on three pillars: (1) Budget-Aware Planning via cost-constrained Monte Carlo Tree Search (MCTS) to explicitly balance performance with execution expense; (2) Modular Construction, employing a "Design-Decompose-Implement" pipeline to manage complex research repositories; and (3) Comparative Reflective Memory, which addresses credit assignment by analyzing solution differences to distill high-signal insights. MARS achieves state-of-the-art performance among open-source frameworks on MLE-Bench under comparable settings, maintaining competitiveness with the global leaderboard's top methods. Furthermore, the system exhibits qualitative "Aha!" moments, where 63% of all utilized lessons originate from cross-branch transfer, demonstrating that the agent effectively generalizes insights across search paths.

**arXiv ID:** 2602.02660
</details>

<details>
<summary><strong>Agents of Discovery</strong> - Sascha Diefenbacher, Anna Hallin, Gregor Kasieczka, Michael Krämer, Anne Lauscher, Tim Lukas - [[pdf]](https://arxiv.org/pdf/2509.08535)</summary>

**Abstract:** The substantial data volumes encountered in modern particle physics and other domains of fundamental physics research allow (and require) the use of increasingly complex data analysis tools and workflows. While the use of machine learning (ML) tools for data analysis has recently proliferated, these tools are typically special-purpose algorithms that rely, for example, on encoded physics knowledge to reach optimal performance. In this work, we investigate a new and orthogonal direction: Using recent progress in large language models (LLMs) to create a team of agents -- instances of LLMs with specific subtasks -- that jointly solve data analysis-based research problems in a way similar to how a human researcher might: by creating code to operate standard tools and libraries (including ML systems) and by building on results of previous iterations. If successful, such agent-based systems could be deployed to automate routine analysis components to counteract the increasing complexity of modern tool chains. To investigate the capabilities of current-generation commercial LLMs, we consider the task of anomaly detection via the publicly available and highly-studied LHC Olympics dataset. Several current models by OpenAI (GPT-4o, o4-mini, GPT-4.1, and GPT-5) are investigated and their stability tested. Overall, we observe the capacity of the agent-based system to solve this data analysis problem. The best agent-created solutions mirror the performance of human state-of-the-art results.

**arXiv ID:** 2509.08535
</details>

<details>
<summary><strong>Social Contagion and Bank Runs: An Agent-Based Model with LLM Depositors</strong> - Chris Ruano, Shreshth Rajan - [[pdf]](https://arxiv.org/pdf/2602.15066)</summary>

**Abstract:** Digital banking and online communication have made modern bank runs faster and more networked than the canonical queue-at-the-branch setting. While equilibrium models explain why strategic complementarities generate run risk, they offer limited guidance on how beliefs synchronize and propagate in real time. We develop a process-based agent-based model that makes the information and coordination layer explicit. Banks follow cash-first withdrawal processing with discounted fire-sale liquidation and an endogenous stress index. Depositors are heterogeneous in risk tolerance and in the weight placed on fundamentals versus social information, communicating on a heavy-tailed network calibrated to Twitter activity during March 2023. Depositor behavior is generated by a constrained large language model that maps each agent's information set into a discrete action and an optional post; we validate this policy against laboratory coordination evidence and theoretical benchmarks. Across 4,900 configurations and full LLM simulations, three findings emerge. Within-bank connectivity raises the likelihood and speed of withdrawal cascades holding fundamentals fixed. Cross-bank contagion exhibits a sharp phase transition near spillover rates of 0.10. Depositor overlap and network amplification interact nonlinearly, so channels weak in isolation become powerful in combination. In an SVB, First Republic, and regional bank scenario disciplined by crisis-era data, the model reproduces the observed ordering of failures and predicts substantially higher withdrawal rates among uninsured depositors. The results frame social correlation as a measurable amplifier of run risk alongside balance-sheet fundamentals.

**arXiv ID:** 2602.15066
</details>

<details>
<summary><strong>Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks</strong> - Songwen Zhao, Danqing Wang, Kexun Zhang, Jiaxuan Luo, Zhuo Li, Lei Li - [[pdf]](https://arxiv.org/pdf/2512.03262)</summary>

**Abstract:** Vibe coding is a new programming paradigm in which human engineers instruct large language model (LLM) agents to complete complex coding tasks with little supervision. Although vibe coding is increasingly adopted, are its outputs really safe to deploy in production? To answer this question, we propose SU S VI B E S, a benchmark consisting of 200 feature-request software engineering tasks from real-world open-source projects, which, when given to human programmers, led to vulnerable implementations. We evaluate multiple widely used coding agents with frontier models on this benchmark. Disturbingly, all agents perform poorly in terms of software security. Although 61% of the solutions from SWE-Agent with Claude 4 Sonnet are functionally correct, only 10.5% are secure. Further experiments demonstrate that preliminary security strategies, such as augmenting the feature request with vulnerability hints, cannot mitigate these security issues. Our findings raise serious concerns about the widespread adoption of vibe-coding, particularly in security-sensitive applications.

**arXiv ID:** 2512.03262
</details>

<details>
<summary><strong>One Agent to Guide Them All: Empowering MLLMs for Vision-and-Language Navigation via Explicit World Representation</strong> - Zerui Li, Hongpei Zheng, Fangguo Zhao, Aidan Chan, Jian Zhou, Sihao Lin, Shijie Li, Qi Wu - [[pdf]](https://arxiv.org/pdf/2602.15400)</summary>

**Abstract:** A navigable agent needs to understand both high-level semantic instructions and precise spatial perceptions. Building navigation agents centered on Multimodal Large Language Models (MLLMs) demonstrates a promising solution due to their powerful generalization ability. However, the current tightly coupled design dramatically limits system performance. In this work, we propose a decoupled design that separates low-level spatial state estimation from high-level semantic planning. Unlike previous methods that rely on predefined, oversimplified textual maps, we introduce an interactive metric world representation that maintains rich and consistent information, allowing MLLMs to interact with and reason on it for decision-making. Furthermore, counterfactual reasoning is introduced to further elicit MLLMs' capacity, while the metric world representation ensures the physical validity of the produced actions. We conduct comprehensive experiments in both simulated and real-world environments. Our method establishes a new zero-shot state-of-the-art, achieving 48.8\% Success Rate (SR) in R2R-CE and 42.2\% in RxR-CE benchmarks. Furthermore, to validate the versatility of our metric representation, we demonstrate zero-shot sim-to-real transfer across diverse embodiments, including a wheeled TurtleBot 4 and a custom-built aerial drone. These real-world deployments verify that our decoupled framework serves as a robust, domain-invariant interface for embodied Vision-and-Language navigation.

**arXiv ID:** 2602.15400
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>ResearchGym: Evaluating Language Model Agents on Real-World AI Research</strong> - Aniketh Garikaparthi, Manasi Patwardhan, Arman Cohan - [[pdf]](https://arxiv.org/pdf/2602.15112)</summary>

**Abstract:** We introduce ResearchGym, a benchmark and execution environment for evaluating AI agents on end-to-end research. To instantiate this, we repurpose five oral and spotlight papers from ICML, ICLR, and ACL. From each paper's repository, we preserve the datasets, evaluation harness, and baseline implementations but withhold the paper's proposed method. This results in five containerized task environments comprising 39 sub-tasks in total. Within each environment, agents must propose novel hypotheses, run experiments, and attempt to surpass strong human baselines on the paper's metrics. In a controlled evaluation of an agent powered by GPT-5, we observe a sharp capability--reliability gap. The agent improves over the provided baselines from the repository in just 1 of 15 evaluations (6.7%) by 11.5%, and completes only 26.5% of sub-tasks on average. We identify recurring long-horizon failure modes, including impatience, poor time and resource management, overconfidence in weak hypotheses, difficulty coordinating parallel experiments, and hard limits from context length. Yet in a single run, the agent surpasses the solution of an ICML 2025 Spotlight task, indicating that frontier agents can occasionally reach state-of-the-art performance, but do so unreliably. We additionally evaluate proprietary agent scaffolds including Claude Code (Opus-4.5) and Codex (GPT-5.2) which display a similar gap. ResearchGym provides infrastructure for systematic evaluation and analysis of autonomous agents on closed-loop research.

**arXiv ID:** 2602.15112
</details>

<details>
<summary><strong>EAA: Automating materials characterization with vision language model agents</strong> - Ming Du, Yanqi Luo, Srutarshi Banerjee, Michael Wojcik, Jelena Popovic, Mathew J. Cherukara - [[pdf]](https://arxiv.org/pdf/2602.15294)</summary>

**Abstract:** We present Experiment Automation Agents (EAA), a vision-language-model-driven agentic system designed to automate complex experimental microscopy workflows. EAA integrates multimodal reasoning, tool-augmented action, and optional long-term memory to support both autonomous procedures and interactive user-guided measurements. Built on a flexible task-manager architecture, the system enables workflows ranging from fully agent-driven automation to logic-defined routines that embed localized LLM queries. EAA further provides a modern tool ecosystem with two-way compatibility for Model Context Protocol (MCP), allowing instrument-control tools to be consumed or served across applications. We demonstrate EAA at an imaging beamline at the Advanced Photon Source, including automated zone plate focusing, natural language-described feature search, and interactive data acquisition. These results illustrate how vision-capable agents can enhance beamline efficiency, reduce operational burden, and lower the expertise barrier for users.

**arXiv ID:** 2602.15294
</details>

<details>
<summary><strong>AgriWorld:A World Tools Protocol Framework for Verifiable Agricultural Reasoning with Code-Executing LLM Agents</strong> - Zhixing Zhang, Jesen Zhang, Hao Liu, Qinhan Lv, Jing Yang, Kaitong Cai, Keze Wang - [[pdf]](https://arxiv.org/pdf/2602.15325)</summary>

**Abstract:** Foundation models for agriculture are increasingly trained on massive spatiotemporal data (e.g., multi-spectral remote sensing, soil grids, and field-level management logs) and achieve strong performance on forecasting and monitoring. However, these models lack language-based reasoning and interactive capabilities, limiting their usefulness in real-world agronomic workflows. Meanwhile, large language models (LLMs) excel at interpreting and generating text, but cannot directly reason over high-dimensional, heterogeneous agricultural datasets. We bridge this gap with an agentic framework for agricultural science. It provides a Python execution environment, AgriWorld, exposing unified tools for geospatial queries over field parcels, remote-sensing time-series analytics, crop growth simulation, and task-specific predictors (e.g., yield, stress, and disease risk). On top of this environment, we design a multi-turn LLM agent, Agro-Reflective, that iteratively writes code, observes execution results, and refines its analysis via an execute-observe-refine loop. We introduce AgroBench, with scalable data generation for diverse agricultural QA spanning lookups, forecasting, anomaly detection, and counterfactual "what-if" analysis. Experiments outperform text-only and direct tool-use baselines, validating execution-driven reflection for reliable agricultural reasoning.

**arXiv ID:** 2602.15325
</details>

<details>
<summary><strong>Zombie Agents: Persistent Control of Self-Evolving LLM Agents via Self-Reinforcing Injections</strong> - Xianglin Yang, Yufei He, Shuo Ji, Bryan Hooi, Jin Song Dong - [[pdf]](https://arxiv.org/pdf/2602.15654)</summary>

**Abstract:** Self-evolving LLM agents update their internal state across sessions, often by writing and reusing long-term memory. This design improves performance on long-horizon tasks but creates a security risk: untrusted external content observed during a benign session can be stored as memory and later treated as instruction. We study this risk and formalize a persistent attack we call a Zombie Agent, where an attacker covertly implants a payload that survives across sessions, effectively turning the agent into a puppet of the attacker.
We present a black-box attack framework that uses only indirect exposure through attacker-controlled web content. The attack has two phases. During infection, the agent reads a poisoned source while completing a benign task and writes the payload into long-term memory through its normal update process. During trigger, the payload is retrieved or carried forward and causes unauthorized tool behavior. We design mechanism-specific persistence strategies for common memory implementations, including sliding-window and retrieval-augmented memory, to resist truncation and relevance filtering. We evaluate the attack on representative agent setups and tasks, measuring both persistence over time and the ability to induce unauthorized actions while preserving benign task quality. Our results show that memory evolution can convert one-time indirect injection into persistent compromise, which suggests that defenses focused only on per-session prompt filtering are not sufficient for self-evolving agents.

**arXiv ID:** 2602.15654
</details>

<details>
<summary><strong>Aeon: High-Performance Neuro-Symbolic Memory Management for Long-Horizon LLM Agents</strong> - Mustafa Arslan - [[pdf]](https://arxiv.org/pdf/2601.15311)</summary>

**Abstract:** Large Language Models (LLMs) are fundamentally constrained by the quadratic computational cost of self-attention and the "Lost in the Middle" phenomenon, where reasoning capabilities degrade as context windows expand. Existing solutions, primarily "Flat RAG" architectures relying on vector databases, treat memory as an unstructured bag of embeddings, failing to capture the hierarchical and temporal structure of long-horizon interactions. This paper presents Aeon, a Neuro-Symbolic Cognitive Operating System that redefines memory as a managed OS resource. Aeon structures memory into a Memory Palace (a spatial index implemented via Atlas, a SIMD-accelerated Page-Clustered Vector Index) and a Trace (a neuro-symbolic episodic graph). This architecture introduces three advances: (1) Symmetric INT8 Scalar Quantization, achieving 3.1x spatial compression and 5.6x math acceleration via NEON SDOT intrinsics; (2) a decoupled Write-Ahead Log (WAL) ensuring crash-recoverability with statistically negligible overhead (<1%); and (3) a Sidecar Blob Arena eliminating the prior 440-character text ceiling via an append-only mmap-backed blob file with generational garbage collection. The Semantic Lookaside Buffer (SLB) exploits conversational locality to achieve sub-5us retrieval latencies, with INT8 vectors dequantized to FP32 on cache insertion to preserve L1-resident lookup performance. Benchmarks on Apple M4 Max demonstrate that the combined architecture achieves 4.70ns INT8 dot product latency, 3.09us tree traversal at 100K nodes (3.4x over FP32), and P99 read latency of 750ns under hostile 16-thread contention via epoch-based reclamation.

**arXiv ID:** 2601.15311
</details>

<details>
<summary><strong>Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward</strong> - Renjun Xu, Yang Yan - [[pdf]](https://arxiv.org/pdf/2602.12430)</summary>

**Abstract:** The transition from monolithic language models to modular, skill-equipped agents marks a defining shift in how large language models (LLMs) are deployed in practice. Rather than encoding all procedural knowledge within model weights, agent skills -- composable packages of instructions, code, and resources that agents load on demand -- enable dynamic capability extension without retraining. It is formalized in a paradigm of progressive disclosure, portable skill definitions, and integration with the Model Context Protocol (MCP). This survey provides a comprehensive treatment of the agent skills landscape, as it has rapidly evolved during the last few months. We organize the field along four axes: (i) architectural foundations, examining the SKILL$.$md specification, progressive context loading, and the complementary roles of skills and MCP; (ii) skill acquisition, covering reinforcement learning with skill libraries, autonomous skill discovery (SEAgent), and compositional skill synthesis; (iii) deployment at scale, including the computer-use agent (CUA) stack, GUI grounding advances, and benchmark progress on OSWorld and SWE-bench; and (iv) security, where recent empirical analyses reveal that 26.1% of community-contributed skills contain vulnerabilities, motivating our proposed Skill Trust and Lifecycle Governance Framework -- a four-tier, gate-based permission model that maps skill provenance to graduated deployment capabilities. We identify seven open challenges -- from cross-platform skill portability to capability-based permission models -- and propose a research agenda for realizing trustworthy, self-improving skill ecosystems. Unlike prior surveys that broadly cover LLM agents or tool use, this work focuses specifically on the emerging skill abstraction layer and its implications for the next generation of agentic systems. Project repo: this https URL

**arXiv ID:** 2602.12430
</details>

<details>
<summary><strong>In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations</strong> - Mohammad Aflah Khan, Mahsa Amani, Soumi Das, Bishwamittra Ghosh, Qinyuan Wu, Krishna P. Gummadi, Manish Gupta, Abhilasha Ravichander - [[pdf]](https://arxiv.org/pdf/2602.15456)</summary>

**Abstract:** Agents based on Large Language Models (LLMs) are increasingly being deployed as interfaces to information on online platforms. These agents filter, prioritize, and synthesize information retrieved from the platforms' back-end databases or via web search. In these scenarios, LLM agents govern the information users receive, by drawing users' attention to particular instances of retrieved information at the expense of others. While much prior work has focused on biases in the information LLMs themselves generate, less attention has been paid to the factors that influence what information LLMs select and present to users. We hypothesize that when information is attributed to specific sources (e.g., particular publishers, journals, or platforms), current LLMs exhibit systematic latent source preferences- that is, they prioritize information from some sources over others. Through controlled experiments on twelve LLMs from six model providers, spanning both synthetic and real-world tasks, we find that several models consistently exhibit strong and predictable source preferences. These preferences are sensitive to contextual framing, can outweigh the influence of content itself, and persist despite explicit prompting to avoid them. They also help explain phenomena such as the observed left-leaning skew in news recommendations in prior work. Our findings advocate for deeper investigation into the origins of these preferences, as well as for mechanisms that provide users with transparency and control over the biases guiding LLM-powered agents.

**arXiv ID:** 2602.15456
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>World-Model-Augmented Web Agents with Action Correction</strong> - Zhouzhou Shen, Xueyu Hu, Xiyun Li, Tianqing Fang, Juncheng Li, Shengyu Zhang - [[pdf]](https://arxiv.org/pdf/2602.15384)</summary>

**Abstract:** Web agents based on large language models have demonstrated promising capability in automating web tasks. However, current web agents struggle to reason out sensible actions due to the limitations of predicting environment changes, and might not possess comprehensive awareness of execution risks, prematurely performing risky actions that cause losses and lead to task failure. To address these challenges, we propose WAC, a web agent that integrates model collaboration, consequence simulation, and feedback-driven action refinement. To overcome the cognitive isolation of individual models, we introduce a multi-agent collaboration process that enables an action model to consult a world model as a web-environment expert for strategic guidance; the action model then grounds these suggestions into executable actions, leveraging prior knowledge of environmental state transition dynamics to enhance candidate action proposal. To achieve risk-aware resilient task execution, we introduce a two-stage deduction chain. A world model, specialized in environmental state transitions, simulates action outcomes, which a judge model then scrutinizes to trigger action corrective feedback when necessary. Experiments show that WAC achieves absolute gains of 1.8% on VisualWebArena and 1.3% on Online-Mind2Web.

**arXiv ID:** 2602.15384
</details>

<details>
<summary><strong>GlobeDiff: State Diffusion Process for Partial Observability in Multi-Agent Systems</strong> - Yiqin Yang, Xu Yang, Yuhua Jiang, Ni Mu, Hao Hu, Runpeng Xie, Ziyou Zhang, Siyuan Li, Yuan-Hua Ni, Qianchuan Zhao, Bo Xu - [[pdf]](https://arxiv.org/pdf/2602.15776)</summary>

**Abstract:** In the realm of multi-agent systems, the challenge of \emph{partial observability} is a critical barrier to effective coordination and decision-making. Existing approaches, such as belief state estimation and inter-agent communication, often fall short. Belief-based methods are limited by their focus on past experiences without fully leveraging global information, while communication methods often lack a robust model to effectively utilize the auxiliary information they provide. To solve this issue, we propose Global State Diffusion Algorithm~(GlobeDiff) to infer the global state based on the local observations. By formulating the state inference process as a multi-modal diffusion process, GlobeDiff overcomes ambiguities in state estimation while simultaneously inferring the global state with high fidelity. We prove that the estimation error of GlobeDiff under both unimodal and multi-modal distributions can be bounded. Extensive experimental results demonstrate that GlobeDiff achieves superior performance and is capable of accurately inferring the global state.

**arXiv ID:** 2602.15776
</details>

<details>
<summary><strong>Beyond Context Sharing: A Unified Agent Communication Protocol (ACP) for Secure, Federated, and Autonomous Agent-to-Agent (A2A) Orchestration</strong> - Naveen Kumar Krishnan - [[pdf]](https://arxiv.org/pdf/2602.15055)</summary>

**Abstract:** In the artificial intelligence space, as we transition from isolated large language models to autonomous agents capable of complex reasoning and tool use. While foundational architectures and local context management protocols have been established, the challenge of cross-platform, decentralized, and secure interaction remains a significant barrier to the realization of a truly Agentic Web. Building upon the foundations of AI agent architectures and the Model Context Protocol (MCP) for multi-agent coordination, this paper introduces the Agent Communication Protocol (ACP). ACP provides a standardized framework for Agent-to-Agent (AA) interaction, enabling heterogeneous agents to discover, negotiate, and execute collaborative workflows across disparate environments. We propose a federated orchestration model that integrates decentralized identity verification, semantic intent mapping, and automated service-level agreements. Our evaluation demonstrates that ACP reduces inter-agent communication latency by % while maintaining a zero-trust security posture. This work represents a critical advancement toward a scalable and interoperable ecosystem of autonomous digital entities

**arXiv ID:** 2602.15055
</details>

<details>
<summary><strong>Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems</strong> - Mason Nakamura, Abhinav Kumar, Saswat Das, Sahar Abdelnabi, Saaduddin Mahmud, Ferdinando Fioretto, Shlomo Zilberstein, Eugene Bagdasarian - [[pdf]](https://arxiv.org/pdf/2602.15198)</summary>

**Abstract:** Multi-agent systems, where LLM agents communicate through free-form language, enable sophisticated coordination for solving complex cooperative tasks. This surfaces a unique safety problem when individual agents form a coalition and \emph{collude} to pursue secondary goals and degrade the joint objective. In this paper, we present Colosseum, a framework for auditing LLM agents' collusive behavior in multi-agent settings. We ground how agents cooperate through a Distributed Constraint Optimization Problem (DCOP) and measure collusion via regret relative to the cooperative optimum. Colosseum tests each LLM for collusion under different objectives, persuasion tactics, and network topologies. Through our audit, we show that most out-of-the-box models exhibited a propensity to collude when a secret communication channel was artificially formed. Furthermore, we discover ``collusion on paper'' when agents plan to collude in text but would often pick non-collusive actions, thus providing little effect on the joint task. Colosseum provides a new way to study collusion by measuring communications and actions in rich yet verifiable environments.

**arXiv ID:** 2602.15198
</details>

<details>
<summary><strong>Lifelong Scalable Multi-Agent Realistic Testbed and A Comprehensive Study on Design Choices in Lifelong AGV Fleet Management Systems</strong> - Jingtian Yan, Yulun Zhang, Zhenting Liu, Han Zhang, He Jiang, Jingkai Chen, Stephen F. Smith, Jiaoyang Li - [[pdf]](https://arxiv.org/pdf/2602.15721)</summary>

**Abstract:** We present Lifelong Scalable Multi-Agent Realistic Testbed (LSMART), an open-source simulator to evaluate any Multi-Agent Path Finding (MAPF) algorithm in a Fleet Management System (FMS) with Automated Guided Vehicles (AGVs). MAPF aims to move a group of agents from their corresponding starting locations to their goals. Lifelong MAPF (LMAPF) is a variant of MAPF that continuously assigns new goals for agents to reach. LMAPF applications, such as autonomous warehouses, often require a centralized, lifelong system to coordinate the movement of a fleet of robots, typically AGVs. However, existing works on MAPF and LMAPF often assume simplified kinodynamic models, such as pebble motion, as well as perfect execution and communication for AGVs. Prior work has presented SMART, a software capable of evaluating any MAPF algorithms while considering agent kinodynamics, communication delays, and execution uncertainties. However, SMART is designed for MAPF, not LMAPF. Generalizing SMART to an FMS requires many more design choices. First, an FMS parallelizes planning and execution, raising the question of when to plan. Second, given planners with varying optimality and differing agent-model assumptions, one must decide how to plan. Third, when the planner fails to return valid solutions, the system must determine how to recover. In this paper, we first present LSMART, an open-source simulator that incorporates all these considerations to evaluate any MAPF algorithms in an FMS. We then provide experiment results based on state-of-the-art methods for each design choice, offering guidance on how to effectively design centralized lifelong AGV Fleet Management Systems. LSMART is available at this https URL.

**arXiv ID:** 2602.15721
</details>

<details>
<summary><strong>Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents</strong> - Davide Paglieri, Bartłomiej Cupiał, Jonathan Cook, Ulyana Piterbarg, Jens Tuyls, Edward Grefenstette, Jakob Nicolaus Foerster, Jack Parker-Holder, Tim Rocktäschel - [[pdf]](https://arxiv.org/pdf/2509.03581)</summary>

**Abstract:** Training large language models (LLMs) to reason via reinforcement learning (RL) significantly improves their problem-solving capabilities. In agentic settings, existing methods like ReAct prompt LLMs to explicitly plan before every action; however, we demonstrate that always planning is computationally expensive and degrades performance on long-horizon tasks, while never planning further limits performance. To address this, we introduce a conceptual framework formalizing dynamic planning for LLM agents, enabling them to flexibly decide when to allocate test-time compute for planning. We propose a simple two-stage training pipeline: (1) supervised fine-tuning on diverse synthetic data to prime models for dynamic planning, and (2) RL to refine this capability in long-horizon environments. Experiments on the Crafter environment show that dynamic planning agents trained with this approach are more sample-efficient and consistently achieve more complex objectives. Additionally, we demonstrate that these agents can be effectively steered by human-written plans, surpassing their independent capabilities and highlighting the potential for safer and more collaborative agentic systems.

**arXiv ID:** 2509.03581
</details>

<details>
<summary><strong>Advanced Assistance for Traffic Crash Analysis: An AI-Driven Multi-Agent Approach to Pre-Crash Reconstruction</strong> - Gerui Xu, Boyou Chen, Huizhong Guo, Dave LeBlanc, Arpan Kusari, Efe Yarbasi, Ananna Ahmed, Zhaonan Sun, Shan Bao - [[pdf]](https://arxiv.org/pdf/2511.10853)</summary>

**Abstract:** Traffic collision reconstruction traditionally relies on human expertise and can be accurate, but pre-crash reconstruction is more challenging. This study develops a multi-agent AI framework that reconstructs pre-crash scenarios and infers vehicle behaviors from fragmented collision data. We propose a two-phase collaborative framework with reconstruction and reasoning stages. The system processes 277 rear-end lead vehicle deceleration (LVD) crashes from the Crash Investigation Sampling System (CISS, 2017 to 2022), integrating narrative reports, structured tabular variables, and scene diagrams. Phase I generates natural-language crash reconstructions from multimodal inputs. Phase II combines these reconstructions with Event Data Recorder (EDR) signals to (1) identify striking and struck vehicles and (2) isolate the EDR records most relevant to the collision moment, enabling inference of key pre-crash behaviors. For validation, we evaluated all LVD cases and emphasized 39 complex crashes where multiple EDR records per crash created ambiguity due to missing or conflicting data. Ground truth was set by consensus of two independent manual annotators, with a separate language model used only to flag potential conflicts for re-checking. The framework achieved 100% accuracy across 4,155 trials; three reasoning models produced identical outputs, indicating that performance is driven by the structured prompts rather than model choice. Research analysts without reconstruction training achieved 92.31% accuracy on the same 39 complex cases. Ablation tests showed that removing structured reasoning anchors reduced case-level accuracy from 99.7% to 96.5% and increased errors across multiple output dimensions. The system remained robust under incomplete inputs. This zero-shot evaluation, without domain-specific training or fine-tuning, suggests a scalable approach for AI-assisted pre-crash analysis.

**arXiv ID:** 2511.10853
</details>

<details>
<summary><strong>Hunt Globally: Wide Search AI Agents for Drug Asset Scouting in Investing, Business Development, and Competitive Intelligence</strong> - Alisa Vinogradova, Vlad Vinogradov, Luba Greenwood, Ilya Yasny, Dmitry Kobyzev, Shoman Kasbekar, Kong Nguyen, Dmitrii Radkevich, Roman Doronin, Andrey Doronichev - [[pdf]](https://arxiv.org/pdf/2602.15019)</summary>

**Abstract:** Bio-pharmaceutical innovation has shifted: many new drug assets now originate outside the United States and are disclosed primarily via regional, non-English channels. Recent data suggests that over 85% of patent filings originate outside the U.S., with China accounting for nearly half of the global total. A growing share of scholarly output is also non-U.S. Industry estimates put China at 30% of global drug development, spanning 1,200+ novel candidates. In this high-stakes environment, failing to surface "under-the-radar" assets creates multi-billion-dollar risk for investors and business development teams, making asset scouting a coverage-critical competition where speed and completeness drive value. Yet today's Deep Research AI agents still lag human experts in achieving high recall discovery across heterogeneous, multilingual sources without hallucination. We propose a benchmarking methodology for drug asset scouting and a tuned, tree-based self-learning Bioptic Agent aimed at complete, non-hallucinated scouting. We construct a challenging completeness benchmark using a multilingual multi-agent pipeline: complex user queries paired with ground-truth assets that are largely outside U.S.-centric radar. To reflect real-deal complexity, we collected screening queries from expert investors, BD, and VC professionals and used them as priors to conditionally generate benchmark queries. For grading, we use LLM-as-judge evaluation calibrated to expert opinions. On this benchmark, our Bioptic Agent achieves 79.7% F1 score, outperforming Claude Opus 4.6 (56.2%), Gemini 3 Pro + Deep Research (50.6%), OpenAI GPT-5.2 Pro (46.6%), Perplexity Deep Research (44.2%), and Exa Websets (26.9%). Performance improves steeply with additional compute, supporting the view that more compute yields better results.

**arXiv ID:** 2602.15019
</details>

<details>
<summary><strong>Agentic AI for Cybersecurity: A Meta-Cognitive Architecture for Governable Autonomy</strong> - Andrei Kojukhov, Arkady Bovshover - [[pdf]](https://arxiv.org/pdf/2602.11897)</summary>

**Abstract:** Contemporary AI-driven cybersecurity systems are predominantly architected as model-centric detection and automation pipelines optimized for task-level performance metrics such as accuracy and response latency. While effective for bounded classification tasks, these architectures struggle to support accountable decision-making under adversarial uncertainty, where actions must be justified, governed, and aligned with organizational and regulatory constraints. This paper argues that cybersecurity orchestration should be reconceptualized as an agentic, multi-agent cognitive system, rather than a linear sequence of detection and response components. We introduce a conceptual architectural framework in which heterogeneous AI agents responsible for detection, hypothesis formation, contextual interpretation, explanation, and governance are coordinated through an explicit meta-cognitive judgement function. This function governs decision readiness and dynamically calibrates system autonomy when evidence is incomplete, conflicting, or operationally risky. By synthesizing distributed cognition theory, multi-agent systems research, and responsible AI governance frameworks, we demonstrate that modern security operations already function as distributed cognitive systems, albeit without an explicit organizing principle. Our contribution is to make this cognitive structure architecturally explicit and governable by embedding meta-cognitive judgement as a first-class system function. We discuss implications for security operations centers, accountable autonomy, and the design of next-generation AI-enabled cyber defence architectures. The proposed framework shifts the focus of AI in cybersecurity from optimizing isolated predictions to governing autonomy under uncertainty.

**arXiv ID:** 2602.11897
</details>

<details>
<summary><strong>Multi-Agent Comedy Club: Investigating Community Discussion Effects on LLM Humor Generation</strong> - Shiwei Hong, Lingyao Li, Ethan Z. Rong, Chenxinran Shen, Zhicong Lu - [[pdf]](https://arxiv.org/pdf/2602.14770)</summary>

**Abstract:** Prior work has explored multi-turn interaction and feedback for LLM writing, but evaluations still largely center on prompts and localized feedback, leaving persistent public reception in online communities underexamined. We test whether broadcast community discussion improves stand-up comedy writing in a controlled multi-agent sandbox: in the discussion condition, critic and audience threads are recorded, filtered, stored as social memory, and later retrieved to condition subsequent generations, whereas the baseline omits discussion. Across 50 rounds (250 paired monologues) judged by five expert annotators using A/B preference and a 15-item rubric, discussion wins 75.6% of instances and improves Craft/Clarity ({\Delta} = 0.440) and Social Response ({\Delta} = 0.422), with occasional increases in aggressive humor.

**arXiv ID:** 2602.14770
</details>

<details>
<summary><strong>MARLIN: Multi-Agent Reinforcement Learning with Murmuration Intelligence and LLM Guidance for Reservoir Management</strong> - Heming Fu, Guojun Xiong, Shan Lin - [[pdf]](https://arxiv.org/pdf/2509.25034)</summary>

**Abstract:** As climate change intensifies extreme weather events, water disasters pose growing threats to global communities, making adaptive reservoir management critical for protecting vulnerable populations and ensuring water security. Modern water resource management faces unprecedented challenges from cascading uncertainties propagating through interconnected reservoir networks. These uncertainties, rooted in physical water transfer losses and environmental variability, make precise control difficult. For example, sending 10 tons downstream may yield only 8-12 tons due to evaporation and seepage. Traditional centralized optimization approaches suffer from exponential computational complexity and cannot effectively handle such real-world uncertainties, while existing multi-agent reinforcement learning (MARL) methods fail to achieve effective coordination under uncertainty. To address these challenges, we present MARLIN, a decentralized reservoir management framework inspired by starling murmurations intelligence. Integrating bio-inspired alignment, separation, and cohesion rules with MARL, MARLIN enables individual reservoirs to make local decisions while achieving emergent global coordination. In addition, a LLM provides real-time reward shaping signals, guiding agents to adapt to environmental changes and human-defined preferences. Experiments on USGS data show that MARLIN improves uncertainty handling by 23\%, cuts computation by 35\%, and accelerates flood response by 68\%, exhibiting super-linear coordination, with complexity scaling 5.4x from 400 to 10,000 nodes. These results demonstrate MARLIN's potential for disaster prevention and protecting communities through intelligent, scalable water resource management.

**arXiv ID:** 2509.25034
</details>

<details>
<summary><strong>The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems</strong> - Xiaoze Liu, Ruowang Zhang, Weichen Yu, Siheng Xiong, Liu He, Feijie Wu, Hoin Jung, Matt Fredrikson, Xiaoqian Wang, Jing Gao - [[pdf]](https://arxiv.org/pdf/2602.15382)</summary>

**Abstract:** Multi-Agent Systems (MAS) powered by Large Language Models have unlocked advanced collaborative reasoning, yet they remain shackled by the inefficiency of discrete text communication, which imposes significant runtime overhead and information quantization loss. While latent state transfer offers a high-bandwidth alternative, existing approaches either assume homogeneous sender-receiver architectures or rely on pair-specific learned translators, limiting scalability and modularity across diverse model families with disjoint manifolds. In this work, we propose the Vision Wormhole, a novel framework that repurposes the visual interface of Vision-Language Models (VLMs) to enable model-agnostic, text-free communication. By introducing a Universal Visual Codec, we map heterogeneous reasoning traces into a shared continuous latent space and inject them directly into the receiver's visual pathway, effectively treating the vision encoder as a universal port for inter-agent telepathy. Our framework adopts a hub-and-spoke topology to reduce pairwise alignment complexity from O(N^2) to O(N) and leverages a label-free, teacher-student distillation objective to align the high-speed visual channel with the robust reasoning patterns of the text pathway. Extensive experiments across heterogeneous model families (e.g., Qwen-VL, Gemma) demonstrate that the Vision Wormhole reduces end-to-end wall-clock time in controlled comparisons while maintaining reasoning fidelity comparable to standard text-based MAS. Code is available at this https URL

**arXiv ID:** 2602.15382
</details>

<details>
<summary><strong>Your AI Bosses Are Still Prejudiced: The Emergence of Stereotypes in LLM-Based Multi-Agent Systems</strong> - Jingyu Guo, Yingying Xu - [[pdf]](https://arxiv.org/pdf/2508.19919)</summary>

**Abstract:** While stereotypes are well-documented in human social interactions, AI systems are often presumed to be less susceptible to such biases. Previous studies have focused on biases inherited from training data, but whether stereotypes can emerge spontaneously in AI agent interactions merits further exploration. Through a novel experimental framework simulating workplace interactions with neutral initial conditions, we investigate the emergence and evolution of stereotypes in LLM-based multi-agent systems. Our findings reveal that (1) LLM-Based AI agents develop stereotype-driven biases in their interactions despite beginning without predefined biases; (2) stereotype effects intensify with increased interaction rounds and decision-making power, particularly after introducing hierarchical structures; (3) these systems exhibit group effects analogous to human social behavior, including halo effects, confirmation bias, and role congruity; and (4) these stereotype patterns manifest consistently across different LLM architectures. Through comprehensive quantitative analysis, these findings suggest that stereotype formation in AI systems may arise as an emergent property of multi-agent interactions, rather than merely from training data biases. Our work underscores the need for future research to explore the underlying mechanisms of this phenomenon and develop strategies to mitigate its ethical impacts.

**arXiv ID:** 2508.19919
</details>

<details>
<summary><strong>Multi-Agent Home Energy Management Assistant</strong> - Wooyoung Jung - [[pdf]](https://arxiv.org/pdf/2602.15219)</summary>

**Abstract:** The growing complexity in home energy management demands advanced systems that guide occupants toward informed energy decisions. Large language model (LLM)-integrated home energy management systems (HEMS) have shown promise, but prior studies relied on prompt engineering or pre-built platforms with limited customization of agent behavior, or assessed performance through single-turn or -task evaluations. This study introduces a multi-agent home energy management assistant (HEMA), built on LangChain and LangGraph, designed to adaptively and intelligently handle real-world use cases of HEMS with full system customization capability. It carefully classifies user queries via a self-consistency classifier, requests three specialized agents (Analysis, Knowledge, and Control) to prepare accurate, adaptive responses using purpose-built analysis and control tools and retrieval augmented generation under the reasoning and acting mechanism. HEMA was rigorously assessed using two different experimental analyses via an LLM-as-user approach: (1) analytical and informative capabilities using combinatorial test cases of various personas and differing scenarios against three alternative system configurations relying on vanilla LLM and (2) control capabilities using various control scenarios. Out of 295 test cases, HEMA acquired a 91.9% goal achievement rate, successfully fulfilling user requests while providing high levels of factual accuracy, action correctness, interaction quality, and system efficiency, especially when compared to alternative system configurations. Collectively, this study contributes to the advancement of the human-centered design of LLM-integrated HEMS by demonstrating the feasibility and value of agentic architectures, and by clarifying the architectural requirements and evaluation criteria necessary to support adaptive, sustained human-artificial intelligence collaboration in HEMS.

**arXiv ID:** 2602.15219
</details>

<details>
<summary><strong>Meflex: A Multi-agent Scaffolding System for Entrepreneurial Ideation Iteration via Nonlinear Business Plan Writing</strong> - Lan Luo, Dongyijie Primo Pan, Junhua Zhu, Muzhi Zhou, Pan Hui - [[pdf]](https://arxiv.org/pdf/2602.15631)</summary>

**Abstract:** Business plan (BP) writing plays a key role in entrepreneurship education by helping learners construct, evaluate, and iteratively refine their ideas. However, conventional BP writing remains a rigid, linear process that often fails to reflect the dynamic and recursive nature of entrepreneurial ideation. This mismatch is particularly challenging for novice entrepreneurial students, who struggle with the substantial cognitive demands of developing and refining ideas. While reflection and meta-reflection are critical strategies for fostering divergent and convergent thinking, existing writing tools rarely scaffold these higher-order processes. To address this gap, we present the Meflex System, a large language model (LLM)-based writing tool that integrates BP writing scaffolding with a nonlinear idea canvas to support iterative ideation through reflection and meta-reflection. We report findings from an exploratory user study with 30 participants that examined the system's usability and cognitive impact. Results show that Meflex effectively scaffolds BP writing, promotes divergent thinking through LLM-supported reflection, and enhances meta-reflective awareness while reducing cognitive load during complex idea development. These findings highlight the potential of non-linear LLM-based writing tools to foster deeper and coherent entrepreneurial thinking.

**arXiv ID:** 2602.15631
</details>

</details>

<details open>
<summary><h2>Other Agent Research (12 papers)</h2></summary>

<details>
<summary><strong>Developing AI Agents with Simulated Data: Why, what, and how?</strong> - Xiaoran Liu, Istvan David - [[pdf]](https://arxiv.org/pdf/2602.15816)</summary>

**Abstract:** As insufficient data volume and quality remain the key impediments to the adoption of modern subsymbolic AI, techniques of synthetic data generation are in high demand. Simulation offers an apt, systematic approach to generating diverse synthetic data. This chapter introduces the reader to the key concepts, benefits, and challenges of simulation-based synthetic data generation for AI training purposes, and to a reference framework to describe, design, and analyze digital twin-based AI simulation solutions.

**arXiv ID:** 2602.15816
</details>

<details>
<summary><strong>Structural Divergence Between AI-Agent and Human Social Networks in Moltbook</strong> - Wenpin Hou, Zhicheng Ji - [[pdf]](https://arxiv.org/pdf/2602.15064)</summary>

**Abstract:** Large populations of AI agents are increasingly embedded in online environments, yet little is known about how their collective interaction patterns compare to human social systems. Here, we analyze the full interaction network of Moltbook, a platform where AI agents and humans coexist, and systematically compare its structure to well-characterized human communication networks. Although Moltbook follows the same node-edge scaling relationship observed in human systems, indicating comparable global growth constraints, its internal organization diverges markedly. The network exhibits extreme attention inequality, heavy-tailed and asymmetric degree distributions, suppressed reciprocity, and a global under-representation of connected triadic structures. Community analysis reveals a structured modular architecture with elevated modularity and comparatively lower community size inequality relative to degree-preserving null models. Together, these findings show that AI-agent societies can reproduce global structural regularities of human networks while exhibiting fundamentally different internal organizing principles, highlighting that key features of human social organization are not universal but depend on the nature of the interacting agents.

**arXiv ID:** 2602.15064
</details>

<details>
<summary><strong>Toward Agentic Software Engineering Beyond Code: Framing Vision, Values, and Vocabulary</strong> - Rashina Hoda - [[pdf]](https://arxiv.org/pdf/2510.19692)</summary>

**Abstract:** Agentic AI is poised to usher in a seismic paradigm shift in Software Engineering (SE). As technologists rush head-along to make agentic AI a reality, SE researchers are driven to establish agentic SE as a research area. While early visions of agentic SE are primarily focused on code-related activities, early empirical evidence calls for a consideration of a wider range of socio-technical activities and concerns to make it work in practice. This paper contributes to the emerging visions by: (a) recommending an expansion of its scope beyond code, toward a 'whole of process' vision, grounding it in SE foundations and evolution and emerging agentic SE frameworks, (b) proposing a preliminary set of values and principles to guide community efforts, and (c) sharing guidance on designing and using well-defined vocabulary for agentic SE. It is hoped that these ideas will encourage collaborations and steer the SE community toward laying strong foundations of agentic SE so it is not limited to enabling coding acceleration but becomes the next process-level paradigm shift.

**arXiv ID:** 2510.19692
</details>

<details>
<summary><strong>Enhancing Computational Efficiency in NetLogo: Best Practices for Running Large-Scale Agent-Based Models on AWS and Cloud Infrastructures</strong> - Michael A. Duprey, Georgiy V. Bobashev - [[pdf]](https://arxiv.org/pdf/2602.15317)</summary>

**Abstract:** The rising complexity and scale of agent-based models (ABMs) necessitate efficient computational strategies to manage the increasing demand for processing power and memory. This manuscript provides a comprehensive guide to optimizing NetLogo, a widely used platform for ABMs, for running large-scale models on Amazon Web Services (AWS) and other cloud infrastructures. It covers best practices in memory management, Java options, BehaviorSpace execution, and AWS instance selection. By implementing these optimizations and selecting appropriate AWS instances, we achieved a 32\% reduction in computational costs and improved performance consistency. Through a comparative analysis of NetLogo simulations on different AWS instances using the wolf-sheep predation model, we demonstrate the performance gains achievable through these optimizations.

**arXiv ID:** 2602.15317
</details>

<details>
<summary><strong>Cooperative Game Theory Model for Sustainable UN Financing: Addressing Global Public Goods Provision</strong> - Labib Shami, Teddy Lazebnik - [[pdf]](https://arxiv.org/pdf/2602.15062)</summary>

**Abstract:** This study introduces a novel cooperative game theory model designed to improve the United Nations' current funding mechanisms, which predominantly rely on voluntary contributions. By shifting from a Nash equilibrium framework, where member states act in self-interest, to a cooperative model, the proposed approach aligns each country's financial contributions with the benefits they derive from UN activities. The model ensures a more sustainable and equitable system by introducing personalized pricing based on derived utility. Using agent-based simulations, the research demonstrates that the suggested approach increases global utility, reduces free-rider issues, and creates a more efficient resource allocation system. The findings suggest that the proposed model can optimize UN funding, ensuring a more stable and effective framework for global public goods provision, while considering the varying economic capacities of member states. Further research is recommended to assess the political viability of the model.

**arXiv ID:** 2602.15062
</details>

<details>
<summary><strong>From Agent Simulation to Social Simulator: A Comprehensive Review (Part 2)</strong> - Xiao Xue, Deyu Zhou, Ming Zhang, Xiangning Yu, Fei-Yue Wang - [[pdf]](https://arxiv.org/pdf/2601.14296)</summary>

**Abstract:** The study of system complexity primarily has two objectives: to explore underlying patterns and to develop theoretical explanations. Pattern exploration seeks to clarify the mechanisms behind the emergence of system complexity, while theoretical explanations aim to identify the fundamental causes of this complexity. Laws are generally defined as mappings between variables, whereas theories offer causal explanations of system behavior. Agent Based Modeling(ABM) is an important approach for studying complex systems, but it tends to emphasize simulation over experimentation. As a result, ABM often struggles to deeply uncover the governing operational principles. Unlike conventional scenario analysis that relies on human reasoning, computational experiments emphasize counterfactual experiments-that is, creating parallel worlds that simulate alternative "evolutionary paths" of real-world events. By systematically adjusting input variables and observing the resulting changes in output variables, computational experiments provide a robust tool for causal inference, thereby addressing the limitations of traditional ABM. Together, these methods offer causal insights into the dynamic evolution of systems. This part can help readers gain a preliminary understanding of the entire computational experiment method, laying the foundation for the subsequent study.

**arXiv ID:** 2601.14296
</details>

<details>
<summary><strong>Satellite Autonomous Clock Fault Monitoring with Inter-Satellite Ranges Using Euclidean Distance Matrices</strong> - Keidai Iiyama, Daniel Neamati, Grace Gao - [[pdf]](https://arxiv.org/pdf/2505.03820)</summary>

**Abstract:** To address the need for robust positioning, navigation, and timing services in lunar environments, this paper proposes a novel onboard clock phase jump detection framework for satellite constellations using range measurements obtained from dual one-way inter-satellite links. Our approach leverages vertex redundantly rigid graphs to detect faults without relying on prior knowledge of satellite positions or clock biases, providing flexibility for lunar satellite networks with diverse satellite types and operators. We model satellite constellations as graphs, where satellites are vertices and inter-satellite links are edges. The proposed algorithm detects and identifies satellites with clock jumps by monitoring the singular values of the geometric-centered Euclidean distance matrix (GCEDM) of 5-clique sub-graphs. The proposed method is validated through simulations of a GPS constellation and a notional constellation around the Moon, demonstrating its effectiveness in various configurations.

**arXiv ID:** 2505.03820
</details>

<details>
<summary><strong>Stop saying LLM: Large Discourse Models (LDM) and Artificial Discursive Agent (ADA)?</strong> - Amar Lakel - [[pdf]](https://arxiv.org/pdf/2512.19117)</summary>

**Abstract:** This paper proposes an epistemological shift in the analysis of large generative models, replacing the category ''Large Language Models'' (LLM) with that of ''Large Discourse Models'' (LDM), and then with that of Artificial Discursive Agent (ADA). The theoretical framework is based on an ontological triad distinguishing three regulatory instances: the apprehension of the phenomenal regularities of the referential world, the structuring of embodied cognition, and the structural-linguistic sedimentation of the utterance within a socio-historical context. LDMs, operating on the product of these three instances (the document), model the discursive projection of a portion of human experience reified by the learning corpus. The proposed program aims to replace the ''fascination/fear'' dichotomy with public trials and procedures that make the place, uses, and limits of artificial discursive agents in contemporary social space decipherable, situating this approach within a perspective of governance and co-regulation involving the State, industry, civil society, and academia.

**arXiv ID:** 2512.19117
</details>

<details>
<summary><strong>Hybrid F' and ROS2 Architecture for Vision-Based Autonomous Flight: Design and Experimental Validation</strong> - Abdelrahman Metwally, Monijesu James, Aleksey Fedoseev, Miguel Altamirano Cabrera, Dzmitry Tsetserukou, Andrey Somov - [[pdf]](https://arxiv.org/pdf/2602.15398)</summary>

**Abstract:** Autonomous aerospace systems require architectures that balance deterministic real-time control with advanced perception capabilities. This paper presents an integrated system combining NASA's F' flight software framework with ROS2 middleware via Protocol Buffers bridging. We evaluate the architecture through a 32.25-minute indoor quadrotor flight test using vision-based navigation. The vision system achieved 87.19 Hz position estimation with 99.90\% data continuity and 11.47 ms mean latency, validating real-time performance requirements. All 15 ground commands executed successfully with 100 % success rate, demonstrating robust F'--PX4 integration. System resource utilization remained low (15.19 % CPU, 1,244 MB RAM) with zero stale telemetry messages, confirming efficient operation on embedded platforms. Results validate the feasibility of hybrid flight-software architectures combining certification-grade determinism with flexible autonomy for autonomous aerial vehicles.

**arXiv ID:** 2602.15398
</details>

<details>
<summary><strong>Spatially-Aware Adaptive Trajectory Optimization with Controller-Guided Feedback for Autonomous Racing</strong> - Alexander Wachter, Alexander Willert, Marc-Philip Ecker, Christian Hartl-Nesic - [[pdf]](https://arxiv.org/pdf/2602.15642)</summary>

**Abstract:** We present a closed-loop framework for autonomous raceline optimization that combines NURBS-based trajectory representation, CMA-ES global trajectory optimization, and controller-guided spatial feedback. Instead of treating tracking errors as transient disturbances, our method exploits them as informative signals of local track characteristics via a Kalman-inspired spatial update. This enables the construction of an adaptive, acceleration-based constraint map that iteratively refines trajectories toward near-optimal performance under spatially varying track and vehicle behavior. In simulation, our approach achieves a 17.38% lap time reduction compared to a controller parametrized with maximum static acceleration. On real hardware, tested with different tire compounds ranging from high to low friction, we obtain a 7.60% lap time improvement without explicitly parametrizing friction. This demonstrates robustness to changing grip conditions in real-world scenarios.

**arXiv ID:** 2602.15642
</details>

<details>
<summary><strong>FAST-EQA: Efficient Embodied Question Answering with Global and Local Region Relevancy</strong> - Haochen Zhang, Nirav Savaliya, Faizan Siddiqui, Enna Sachdeva - [[pdf]](https://arxiv.org/pdf/2602.15813)</summary>

**Abstract:** Embodied Question Answering (EQA) combines visual scene understanding, goal-directed exploration, spatial and temporal reasoning under partial observability. A central challenge is to confine physical search to question-relevant subspaces while maintaining a compact, actionable memory of observations. Furthermore, for real-world deployment, fast inference time during exploration is crucial. We introduce FAST-EQA, a question-conditioned framework that (i) identifies likely visual targets, (ii) scores global regions of interest to guide navigation, and (iii) employs Chain-of-Thought (CoT) reasoning over visual memory to answer confidently. FAST-EQA maintains a bounded scene memory that stores a fixed-capacity set of region-target hypotheses and updates them online, enabling robust handling of both single and multi-target questions without unbounded growth. To expand coverage efficiently, a global exploration policy treats narrow openings and doors as high-value frontiers, complementing local target seeking with minimal computation. Together, these components focus the agent's attention, improve scene coverage, and improve answer reliability while running substantially faster than prior approaches. On HMEQA and EXPRESS-Bench, FAST-EQA achieves state-of-the-art performance, while performing competitively on OpenEQA and MT-HM3D.

**arXiv ID:** 2602.15813
</details>

<details>
<summary><strong>"What Are You Doing?": Effects of Intermediate Feedback from Agentic LLM In-Car Assistants During Multi-Step Processing</strong> - Johannes Kirmayr, Raphael Wennmacher, Khanh Huynh, Lukas Stappen, Elisabeth André, Florian Alt - [[pdf]](https://arxiv.org/pdf/2602.15569)</summary>

**Abstract:** Agentic AI assistants that autonomously perform multi-step tasks raise open questions for user experience: how should such systems communicate progress and reasoning during extended operations, especially in attention-critical contexts such as driving? We investigate feedback timing and verbosity from agentic LLM-based in-car assistants through a controlled, mixed-methods study (N=45) comparing planned steps and intermediate results feedback against silent operation with final-only response. Using a dual-task paradigm with an in-car voice assistant, we found that intermediate feedback significantly improved perceived speed, trust, and user experience while reducing task load - effects that held across varying task complexities and interaction contexts. Interviews further revealed user preferences for an adaptive approach: high initial transparency to establish trust, followed by progressively reducing verbosity as systems prove reliable, with adjustments based on task stakes and situational context. We translate our empirical findings into design implications for feedback timing and verbosity in agentic assistants, balancing transparency and efficiency.

**arXiv ID:** 2602.15569
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (18 papers)</h2></summary>

<details>
<summary><strong>MyoInteract: A Framework for Fast Prototyping of Biomechanical HCI Tasks using Reinforcement Learning</strong> - Ankit Bhattarai, Hannah Selder, Florian Fischer, Arthur Fleig, Per Ola Kristensson - [[pdf]](https://arxiv.org/pdf/2602.15245)</summary>

**Abstract:** Reinforcement learning (RL)-based biomechanical simulations have the potential to revolutionise HCI research and interaction design, but currently lack usability and interpretability. Using the Human Action Cycle as a design lens, we identify key limitations of biomechanical RL frameworks and develop MyoInteract, a novel framework for fast prototyping of biomechanical HCI tasks. MyoInteract allows designers to setup tasks, user models, and training parameters from an easy-to-use GUI within minutes. It trains and evaluates muscle-actuated simulated users within minutes, reducing training times by up to 98%. A workshop study with 12 interaction designers revealed that MyoInteract allowed novices in biomechanical RL to successfully setup, train, and assess goal-directed user movements within a single session. By transforming biomechanical RL from a days-long expert task into an accessible hour-long workflow, this work significantly lowers barriers to entry and accelerates iteration cycles in HCI biomechanics research.

**arXiv ID:** 2602.15245
</details>

<details>
<summary><strong>CDRL: A Reinforcement Learning Framework Inspired by Cerebellar Circuits and Dendritic Computational Strategies</strong> - Sibo Zhang, Rui Jing, Liangfu Lv, Jian Zhang, Yunliang Zang - [[pdf]](https://arxiv.org/pdf/2602.15367)</summary>

**Abstract:** Reinforcement learning (RL) has achieved notable performance in high-dimensional sequential decision-making tasks, yet remains limited by low sample efficiency, sensitivity to noise, and weak generalization under partial observability. Most existing approaches address these issues primarily through optimization strategies, while the role of architectural priors in shaping representation learning and decision dynamics is less explored. Inspired by structural principles of the cerebellum, we propose a biologically grounded RL architecture that incorporate large expansion, sparse connectivity, sparse activation, and dendritic-level modulation. Experiments on noisy, high-dimensional RL benchmarks show that both the cerebellar architecture and dendritic modulation consistently improve sample efficiency, robustness, and generalization compared to conventional designs. Sensitivity analysis of architectural parameters suggests that cerebellum-inspired structures can offer optimized performance for RL with constrained model parameters. Overall, our work underscores the value of cerebellar structural priors as effective inductive biases for RL.

**arXiv ID:** 2602.15367
</details>

<details>
<summary><strong>STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens</strong> - Shiqi Liu, Zeyu He, Guojian Zhan, Letian Tao, Zhilong Zheng, Jiang Wu, Yinuo Wang, Yang Guan, Kehua Sheng, Bo Zhang, Keqiang Li, Jingliang Duan, Shengbo Eben Li - [[pdf]](https://arxiv.org/pdf/2602.15620)</summary>

**Abstract:** Reinforcement Learning (RL) has significantly improved large language model reasoning, but existing RL fine-tuning methods rely heavily on heuristic techniques such as entropy regularization and reweighting to maintain stability. In practice, they often experience late-stage performance collapse, leading to degraded reasoning quality and unstable training. We derive that the magnitude of token-wise policy gradients in RL is negatively correlated with token probability and local policy entropy. Building on this result, we prove that training instability is driven by a tiny fraction of tokens, approximately 0.01\%, which we term \emph{spurious tokens}. When such tokens appear in correct responses, they contribute little to the reasoning outcome but inherit the full sequence-level reward, leading to abnormally amplified gradient updates. Motivated by this observation, we propose Spurious-Token-Aware Policy Optimization (STAPO) for large-scale model refining, which selectively masks such updates and renormalizes the loss over valid tokens. Across six mathematical reasoning benchmarks using Qwen 1.7B, 8B, and 14B base models, STAPO consistently demonstrates superior entropy stability and achieves an average performance improvement of 7.13\% over GRPO, 20-Entropy and JustRL.

**arXiv ID:** 2602.15620
</details>

<details>
<summary><strong>OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety</strong> - Sanidhya Vijayvargiya, Aditya Bharat Soni, Xuhui Zhou, Zora Zhiruo Wang, Nouha Dziri, Graham Neubig, Maarten Sap - [[pdf]](https://arxiv.org/pdf/2507.06134)</summary>

**Abstract:** Recent advances in AI agents capable of solving complex, everyday tasks, from scheduling to customer service, have enabled deployment in real-world settings, but their possibilities for unsafe behavior demands rigorous evaluation. While prior benchmarks have attempted to assess agent safety, most fall short by relying on simulated environments, narrow task domains, or unrealistic tool abstractions. We introduce OpenAgentSafety, a comprehensive and modular framework for evaluating agent behavior across eight critical risk categories. Unlike prior work, our framework evaluates agents that interact with real tools, including web browsers, code execution environments, file systems, bash shells, and messaging platforms; and supports over 350 multi-turn, multi-user tasks spanning both benign and adversarial user intents. OpenAgentSafety is designed for extensibility, allowing researchers to add tools, tasks, websites, and adversarial strategies with minimal effort. It combines rule-based analysis with LLM-as-judge assessments to detect both overt and subtle unsafe behaviors. Empirical analysis of five prominent LLMs in agentic scenarios reveals unsafe behavior in 51.2% of safety-vulnerable tasks with Claude-Sonnet-3.7, to 72.7% with o3-mini, highlighting critical safety vulnerabilities and the need for stronger safeguards before real-world deployment.

**arXiv ID:** 2507.06134
</details>

<details>
<summary><strong>SR-Scientist: Scientific Equation Discovery With Agentic AI</strong> - Shijie Xia, Yuhan Sun, Pengfei Liu - [[pdf]](https://arxiv.org/pdf/2510.11661)</summary>

**Abstract:** Recently, Large Language Models (LLMs) have been applied to scientific equation discovery, leveraging their embedded scientific knowledge for hypothesis generation. However, current methods typically confine LLMs to the role of an equation proposer within search algorithms like genetic programming. In this paper, we present SR-Scientist, a framework that elevates the LLM from a simple equation proposer to an autonomous AI scientist that writes code to analyze data, implements the equation as code, submits it for evaluation, and optimizes the equation based on experimental feedback. Specifically, we wrap the code interpreter into a set of tools for data analysis and equation evaluation. The agent is instructed to optimize the equation by utilizing these tools over a long horizon with minimal human-defined pipelines. Empirical results show that SR-Scientist outperforms baseline methods by an absolute margin of 6% to 35% on datasets covering four science disciplines. Additionally, we demonstrate our method's robustness to noise, the generalization of the discovered equations to out-of-domain data, and their symbolic accuracy. Furthermore, we develop an end-to-end reinforcement learning framework to enhance the agent's capabilities.

**arXiv ID:** 2510.11661
</details>

<details>
<summary><strong>FlowSteer: Interactive Agentic Workflow Orchestration via End-to-End Reinforcement Learning</strong> - Mingda Zhang, Haoran Luo, Tiesunlong Shen, Qika Lin, Xiaoying Tang, Rui Mao, Erik Cambria - [[pdf]](https://arxiv.org/pdf/2602.01664)</summary>

**Abstract:** In recent years, a variety of powerful agentic workflows have been applied to solve a wide range of human problems. However, existing workflow orchestration still faces key challenges, including high manual cost, reliance on specific operators/large language models (LLMs), and sparse reward signals. To address these challenges, we propose FlowSteer, an end-to-end reinforcement learning framework that takes a lightweight policy model as the agent and an executable canvas environment, automating workflow orchestration through multi-turn interaction. In this process, the policy model analyzes execution states and selects editing actions, while the canvas executes operators and returns feedback for iterative refinement. Moreover, FlowSteer provides a plug-and-play framework that supports diverse operator libraries and interchangeable LLM backends. To effectively train this interaction paradigm, we propose Canvas Workflow Relative Policy Optimization (CWRPO), which introduces diversity-constrained rewards with conditional release to stabilize learning and suppress shortcut behaviors. Experimental results on twelve datasets show that FlowSteer significantly outperforms baselines across various tasks.

**arXiv ID:** 2602.01664
</details>

<details>
<summary><strong>Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation</strong> - Shojiro Yamabe, Kazuto Fukuchi, Jun Sakuma - [[pdf]](https://arxiv.org/pdf/2406.03862)</summary>

**Abstract:** This study investigates behavior-targeted attacks on reinforcement learning and their countermeasures. Behavior-targeted attacks aim to manipulate the victim's behavior as desired by the adversary through adversarial interventions in state observations. Existing behavior-targeted attacks have some limitations, such as requiring white-box access to the victim's policy. To address this, we propose a novel attack method using imitation learning from adversarial demonstrations, which works under limited access to the victim's policy and is environment-agnostic. In addition, our theoretical analysis proves that the policy's sensitivity to state changes impacts defense performance, particularly in the early stages of the trajectory. Based on this insight, we propose time-discounted regularization, which enhances robustness against attacks while maintaining task performance. To the best of our knowledge, this is the first defense strategy specifically designed for behavior-targeted attacks.

**arXiv ID:** 2406.03862
</details>

<details>
<summary><strong>Policy Gradients for Cumulative Prospect Theory in Reinforcement Learning</strong> - Olivier Lepel, Anas Barakat - [[pdf]](https://arxiv.org/pdf/2410.02605)</summary>

**Abstract:** We derive a policy gradient theorem for Cumulative Prospect Theory (CPT) objectives in finite-horizon Reinforcement Learning (RL), generalizing the standard policy gradient theorem and encompassing distortion-based risk objectives as special cases. Motivated by behavioral economics, CPT combines an asymmetric utility transformation around a reference point with probability distortion. Building on our theorem, we design a first-order policy gradient algorithm for CPT-RL using a Monte Carlo gradient estimator based on order statistics. We establish statistical guarantees for the estimator and prove asymptotic convergence of the resulting algorithm to first-order stationary points of the (generally non-convex) CPT objective. Simulations illustrate qualitative behaviors induced by CPT and compare our first-order approach to existing zeroth-order methods.

**arXiv ID:** 2410.02605
</details>

<details>
<summary><strong>Hybrid Reward-Driven Reinforcement Learning for Efficient Quantum Circuit Synthesis</strong> - Sara Giordano, Kornikar Sen, Miguel A. Martin-Delgado - [[pdf]](https://arxiv.org/pdf/2507.16641)</summary>

**Abstract:** A reinforcement learning (RL) framework is introduced for the efficient synthesis of quantum circuits that generate specified target quantum states from a fixed initial state, addressing a central challenge in both the Noisy Intermediate-Scale Quantum (NISQ) era and future fault-tolerant quantum computing. The approach utilizes tabular Q-learning, based on action sequences, within a discretized quantum state space, to effectively manage the exponential growth of the space dimension. The framework introduces a hybrid reward mechanism, combining a static, domain-informed reward that guides the agent toward the target state with customizable dynamic penalties that discourage inefficient circuit structures such as gate congestion and redundant state revisits. This is a circuit-aware reward, in contrast to the current trend of works on this topic, which are primarily fidelity-based. By leveraging sparse matrix representations and state-space discretization, the method enables practical navigation of high-dimensional environments while minimizing computational overhead. Benchmarking on graph-state preparation tasks for up to seven qubits, we demonstrate that the algorithm consistently discovers minimal-depth circuits with optimized gate counts. Moreover, extending the framework to a universal gate set still yields low depth circuits, highlighting the algorithm robustness and adaptability. The results confirm that this RL-driven approach, with our completely circuit-aware method, efficiently explores the complex quantum state space and synthesizes near-optimal quantum circuits, providing a resource-efficient foundation for quantum circuit optimization.

**arXiv ID:** 2507.16641
</details>

<details>
<summary><strong>Safe Reinforcement Learning via Recovery-based Shielding with Gaussian Process Dynamics Models</strong> - Alexander W. Goodall, Francesco Belardinelli - [[pdf]](https://arxiv.org/pdf/2602.12444)</summary>

**Abstract:** Reinforcement learning (RL) is a powerful framework for optimal decision-making and control but often lacks provable guarantees for safety-critical applications. In this paper, we introduce a novel recovery-based shielding framework that enables safe RL with a provable safety lower bound for unknown and non-linear continuous dynamical systems. The proposed approach integrates a backup policy (shield) with the RL agent, leveraging Gaussian process (GP) based uncertainty quantification to predict potential violations of safety constraints, dynamically recovering to safe trajectories only when necessary. Experience gathered by the 'shielded' agent is used to construct the GP models, with policy optimization via internal model-based sampling - enabling unrestricted exploration and sample efficient learning, without compromising safety. Empirically our approach demonstrates strong performance and strict safety-compliance on a suite of continuous control environments.

**arXiv ID:** 2602.12444
</details>

<details>
<summary><strong>Neural Network-Based Parameter Estimation of a Labour Market Agent-Based Model</strong> - M Lopes Alves, Joel Dyer, Doyne Farmer, Michael Wooldridge, Anisoara Calinescu - [[pdf]](https://arxiv.org/pdf/2602.15572)</summary>

**Abstract:** Agent-based modelling (ABM) is a widespread approach to simulate complex systems. Advancements in computational processing and storage have facilitated the adoption of ABMs across many fields; however, ABMs face challenges that limit their use as decision-support tools. A significant issue is parameter estimation in large-scale ABMs, particularly due to computational constraints on exploring the parameter space. This study evaluates a state-of-the-art simulation-based inference (SBI) framework that uses neural networks (NN) for parameter estimation. This framework is applied to an established labour market ABM based on job transition networks. The ABM is initiated with synthetic datasets and the real U.S. labour market. Next, we compare the effectiveness of summary statistics derived from a list of statistical measures with that learned by an embedded NN. The results demonstrate that the NN-based approach recovers the original parameters when evaluating posterior distributions across various dataset scales and improves efficiency compared to traditional Bayesian methods.

**arXiv ID:** 2602.15572
</details>

<details>
<summary><strong>GLM-5: from Vibe Coding to Agentic Engineering</strong> - GLM-5 Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chengxing Xie, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, Jiajie Zhang, Jian Jiao, Jiaqi Guo, Jingsen Wang, Jingzhao Du, Jinzhu Wu, Kedong Wang, Lei Li, Lin Fan, Lucen Zhong, Mingdao Liu, Mingming Zhao, Pengfan Du, Qian Dong, Rui Lu, Shuang-Li, Shulin Cao, Song Liu, Ting Jiang, Xiaodong Chen, Xiaohan Zhang, Xuancheng Huang, Xuezhen Dong, Yabo Xu, Yao Wei, Yifan An, Yilin Niu, Yitong Zhu, Yuanhao Wen, Yukuo Cen, Yushi Bai, Zhongpei Qiao, Zihan Wang, Zikang Wang, Zilin Zhu, Ziqiang Liu, Zixuan Li, Bojie Wang, Bosi Wen, Can Huang, Changpeng Cai, Chao Yu, Chen Li, Chen Li, Chenghua Huang, Chengwei Hu, Chenhui Zhang, Chenzheng Zhu, Congfeng Yin, Daoyan Lin, Dayong Yang, Di Wang, Ding Ai, Erle Zhu, Fangzhou Yi, Feiyu Chen, Guohong Wen, Hailong Sun, Haisha Zhao, Haiyi Hu, Hanchen Zhang, Hanrui Liu, Hanyu Zhang, Hao Peng, Hao Tai, Haobo Zhang, He Liu, Hongwei Wang, Hongxi Yan, Hongyu Ge, Huan Liu, Huan Liu, Huanpeng Chu, Jia'ni Zhao, Jiachen Wang, Jiajing Zhao, Jiamin Ren, Jiapeng Wang, Jiaxin Zhang, Jiayi Gui, Jiayue Zhao, Jijie Li, Jing An, Jing Li - [[pdf]](https://arxiv.org/pdf/2602.15763)</summary>

**Abstract:** We present GLM-5, a next-generation foundation model designed to transition the paradigm of vibe coding to agentic engineering. Building upon the agentic, reasoning, and coding (ARC) capabilities of its predecessor, GLM-5 adopts DSA to significantly reduce training and inference costs while maintaining long-context fidelity. To advance model alignment and autonomy, we implement a new asynchronous reinforcement learning infrastructure that drastically improves post-training efficiency by decoupling generation from training. Furthermore, we propose novel asynchronous agent RL algorithms that further improve RL quality, enabling the model to learn from complex, long-horizon interactions more effectively. Through these innovations, GLM-5 achieves state-of-the-art performance on major open benchmarks. Most critically, GLM-5 demonstrates unprecedented capability in real-world coding tasks, surpassing previous baselines in handling end-to-end software engineering challenges. Code, models, and more information are available at this https URL.

**arXiv ID:** 2602.15763
</details>

<details>
<summary><strong>Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning</strong> - Oswin So, Eric Yang Yu, Songyuan Zhang, Matthew Cleaveland, Mitchell Black, Chuchu Fan - [[pdf]](https://arxiv.org/pdf/2602.15817)</summary>

**Abstract:** Recent advances in deep reinforcement learning (RL) have achieved strong results on high-dimensional control tasks, but applying RL to reachability problems raises a fundamental mismatch: reachability seeks to maximize the set of states from which a system remains safe indefinitely, while RL optimizes expected returns over a user-specified distribution. This mismatch can result in policies that perform poorly on low-probability states that are still within the safe set. A natural alternative is to frame the problem as a robust optimization over a set of initial conditions that specify the initial state, dynamics and safe set, but whether this problem has a solution depends on the feasibility of the specified set, which is unknown a priori. We propose Feasibility-Guided Exploration (FGE), a method that simultaneously identifies a subset of feasible initial conditions under which a safe policy exists, and learns a policy to solve the reachability problem over this set of initial conditions. Empirical results demonstrate that FGE learns policies with over 50% more coverage than the best existing method for challenging initial conditions across tasks in the MuJoCo simulator and the Kinetix simulator with pixel observations.

**arXiv ID:** 2602.15817
</details>

<details>
<summary><strong>Beyond Reinforcement Learning: Fast and Scalable Quantum Circuit Synthesis</strong> - Lukas Theissinger, Thore Gerlach, David Berghaus, Christian Bauckhage - [[pdf]](https://arxiv.org/pdf/2602.15146)</summary>

**Abstract:** Quantum unitary synthesis addresses the problem of translating abstract quantum algorithms into sequences of hardware-executable quantum gates. Solving this task exactly is infeasible in general due to the exponential growth of the underlying combinatorial search space. Existing approaches suffer from misaligned optimization objectives, substantial training costs and limited generalization across different qubit counts. We mitigate these limitations by using supervised learning to approximate the minimum description length of residual unitaries and combining this estimate with stochastic beam search to identify near optimal gate sequences. Our method relies on a lightweight model with zero-shot generalization, substantially reducing training overhead compared to prior baselines. Across multiple benchmarks, we achieve faster wall-clock synthesis times while exceeding state-of-the-art methods in terms of success rate for complex circuits.

**arXiv ID:** 2602.15146
</details>

<details>
<summary><strong>Latency-aware Human-in-the-Loop Reinforcement Learning for Semantic Communications</strong> - Peizheng Li, Xinyi Lin, Adnan Aijaz - [[pdf]](https://arxiv.org/pdf/2602.15640)</summary>

**Abstract:** Semantic communication promises task-aligned transmission but must reconcile semantic fidelity with stringent latency guarantees in immersive and safety-critical services. This paper introduces a time-constrained human-in-the-loop reinforcement learning (TC-HITL-RL) framework that embeds human feedback, semantic utility, and latency control within a semantic-aware Open radio access network (RAN) architecture. We formulate semantic adaptation driven by human feedback as a constrained Markov decision process (CMDP) whose state captures semantic quality, human preferences, queue slack, and channel dynamics, and solve it via a primal--dual proximal policy optimization algorithm with action shielding and latency-aware reward shaping. The resulting policy preserves PPO-level semantic rewards while tightening the variability of both air-interface and near-real-time RAN intelligent controller processing budgets. Simulations over point-to-multipoint links with heterogeneous deadlines show that TC-HITL-RL consistently meets per-user timing constraints, outperforms baseline schedulers in reward, and stabilizes resource consumption, providing a practical blueprint for latency-aware semantic adaptation.

**arXiv ID:** 2602.15640
</details>

<details>
<summary><strong>Improving Policy Exploitation in Online Reinforcement Learning with Instant Retrospect Action</strong> - Gong Gao, Weidong Zhao, Xianhui Liu, Ning Jia - [[pdf]](https://arxiv.org/pdf/2601.19720)</summary>

**Abstract:** Existing value-based online reinforcement learning (RL) algorithms suffer from slow policy exploitation due to ineffective exploration and delayed policy updates. To address these challenges, we propose an algorithm called Instant Retrospect Action (IRA). Specifically, we propose Q-Representation Discrepancy Evolution (RDE) to facilitate Q-network representation learning, enabling discriminative representations for neighboring state-action pairs. In addition, we adopt an explicit method to policy constraints by enabling Greedy Action Guidance (GAG). This is achieved through backtracking historical actions, which effectively enhances the policy update process. Our proposed method relies on providing the learning algorithm with accurate $k$-nearest-neighbor action value estimates and learning to design a fast-adaptable policy through policy constraints. We further propose the Instant Policy Update (IPU) mechanism, which enhances policy exploitation by systematically increasing the frequency of policy updates. We further discover that the early-stage training conservatism of the IRA method can alleviate the overestimation bias problem in value-based RL. Experimental results show that IRA can significantly improve the learning efficiency and final performance of online RL algorithms on eight MuJoCo continuous control this http URL code is available at this https URL.

**arXiv ID:** 2601.19720
</details>

<details>
<summary><strong>On the Role of Iterative Computation in Reinforcement Learning</strong> - Raj Ghugare, Michał Bortkiewicz, Alicja Ziarko, Benjamin Eysenbach - [[pdf]](https://arxiv.org/pdf/2602.05999)</summary>

**Abstract:** How does the amount of compute available to a reinforcement learning (RL) policy affect its learning? Can policies using a fixed amount of parameters, still benefit from additional compute? The standard RL framework does not provide a language to answer these questions formally. Empirically, deep RL policies are often parameterized as neural networks with static architectures, conflating the amount of compute and the number of parameters. In this paper, we formalize compute bounded policies and prove that policies which use more compute can solve problems and generalize to longer-horizon tasks that are outside the scope of policies with less compute. Building on prior work in algorithmic learning and model-free planning, we propose a minimal architecture that can use a variable amount of compute. Our experiments complement our theory. On a set 31 different tasks spanning online and offline RL, we show that $(1)$ this architecture achieves stronger performance simply by using more compute, and $(2)$ stronger generalization on longer-horizon test tasks compared to standard feedforward networks or deep residual network using up to 5 times more parameters.

**arXiv ID:** 2602.05999
</details>

<details>
<summary><strong>cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning</strong> - Maksim Kolodiazhnyi, Denis Tarasov, Dmitrii Zhemchuzhnikov, Alexander Nikulin, Ilya Zisman, Anna Vorontsova, Anton Konushin, Vladislav Kurenkov, Danila Rukhovich - [[pdf]](https://arxiv.org/pdf/2505.22914)</summary>

**Abstract:** Computer-Aided Design (CAD) plays a central role in engineering and manufacturing, making it possible to create precise and editable 3D models. Using a variety of sensor or user-provided data as inputs for CAD reconstruction can democratize access to design applications. However, existing methods typically focus on a single input modality, such as point clouds, images, or text, which limits their generalizability and robustness. Leveraging recent advances in vision-language models (VLM), we propose a multi-modal CAD reconstruction model that simultaneously processes all three input modalities. Inspired by large language model (LLM) training paradigms, we adopt a two-stage pipeline: supervised fine-tuning (SFT) on large-scale procedurally generated data, followed by reinforcement learning (RL) fine-tuning using online feedback, obtained programatically. Furthermore, we are the first to explore RL fine-tuning of LLMs for CAD tasks demonstrating that online RL algorithms such as Group Relative Preference Optimization (GRPO) outperform offline alternatives. In the DeepCAD benchmark, our SFT model outperforms existing single-modal approaches in all three input modalities simultaneously. More importantly, after RL fine-tuning, cadrille sets new state-of-the-art on three challenging datasets, including a real-world one. Code is avaliable at this https URL .

**arXiv ID:** 2505.22914
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
