# Agent arXiv Daily

**Last Updated:** 2026-06-15 06:37:48

**Total Papers:** 83

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (5 papers)</h2></summary>

<details>
<summary><strong>Minim: Privacy-Aware Minimal View for Agents via Trusted Local Sanitization</strong> - Hexuan Yu, Chaoyu Zhang, Heng Jin, Shanghao Shi, Ning Zhang, Y. Thomas Hou, Wenjing Lou - [[pdf]](https://arxiv.org/pdf/2606.13949)</summary>

**Abstract:** Modern LLM-powered autonomous agents increasingly rely on rich user interface (UI) state observations to achieve reliable action grounding in complex digital environments. However, many deployments transmit the full UI state to remote inference servers even when most elements are irrelevant to the current task, which can leak sensitive but unnecessary context such as authentication codes, private notifications, and background application states. We propose MINIM, a trusted local broker that performs privacy-aware minimization on the client side before any observation leaves the device. Grounded in Contextual Integrity (CI), MINIM learns a dual-score representation for each UI element by predicting an inherent sensitivity score (s) and a task-conditioned necessity score (n). These scores drive a ternary disclosure policy that keeps essential elements, abstracts sensitive attributes when needed, and removes task-irrelevant content. We optimize a CI-aware objective that penalizes necessity errors more strongly on high-risk content, enabling aggressive pruning while preserving task-critical information. Experiments on real-world UI observations derived from WebArena show that MINIM substantially reduces task-irrelevant sensitive leakage while preserving task-critical semantic context and the interactive affordances required for reliable agent actions.

**arXiv ID:** 2606.13949
</details>

<details>
<summary><strong>From Chatbot to Digital Colleague: The Paradigm Shift Toward Persistent Autonomous AI</strong> - Yongheng Zhang, Ziang Liu, Jiaxuan Zhu, Shuai Wang, Xiangqi Chen, Haojing Huang, Jiayi Kuang, Siyu Chen, Ao Shen, Hao Wu, Qiufeng Wang, Qian-Wen Zhang, Junnan Dong, Wenhao Jiang, Ying Shen, Hai-Tao Zheng, Yinghui Li, Di Yin, Xing Sun, Philip S. Yu - [[pdf]](https://arxiv.org/pdf/2606.14502)</summary>

**Abstract:** Large Language Models (LLMs) are undergoing a fundamental transformation from conversational generators into integrated AI systems capable of reasoning, action, memory, and self-improvement. We conceptualize this transition as a shift from Chatbot to Digital Colleague: from conversational answers to persistent work. We organize this transition along two tightly coupled dimensions. First, at the cognitive core level, LLMs are advancing from Chatbot-era "fast thinking" systems driven by next-token prediction toward Thinking LLMs that leverage inference-time computation, Chain-of-Thought reasoning, reflection, process supervision, and reinforcement learning to support more deliberate and reliable cognition. Second, at the tool-augmented task execution level, LLMs are progressing from tool-calling Agents that invoke external resources in an ad hoc manner toward OpenClaw-style workstation systems (OpenClaw) equipped with persistent Workspaces, skills, verification loops, and governance. The "Workspace + Skill" paradigm makes episodic tool use colleague-like via state persistence, reusable procedures, task closure, and experience reuse. We examine data construction shifts from instruction-response pairs to State-Action-Observation trajectories and evaluation from static benchmarks to sandboxed, auditable, self-evolving AI ecosystems.

**arXiv ID:** 2606.14502
</details>

<details>
<summary><strong>EvoTrainer: Co-Evolving LLM Policies and Training Harnesses for Autonomous Agentic Reinforcement Learning</strong> - Guhong Chen, Yingcheng Shi, Yongbin Li, Binhua Li, Xander Xu, Hu Wei, Shiwen Ni, Min Yang, Jieping Ye - [[pdf]](https://arxiv.org/pdf/2606.03108)</summary>

**Abstract:** Autonomous LLM training is often framed as recipe search, which leaves the training harness largely static. This limitation sharpens in agentic RL, where shifting bottlenecks and scalar rewards mask diverse failure modes. We introduce EvoTrainer, an autonomous training framework that co-evolves LLM policies and training-side harnesses through empirical feedback: it diagnoses rollout-level evidence, revises diagnostics, backtests interventions, and accumulates reusable skills. Evaluated on mathematical reasoning, competitive-programming code generation, and repository-level software engineering, EvoTrainer matches or exceeds the human-engineered RL references under the same data, codebase, and evaluation protocol, with the largest gain on long-horizon agentic SWE. Trajectory analyses show that retained strategies diverge across domains, evolving diagnostics prevent invalid high-scoring branches from being promoted, and reusable skills shape later search. Autonomous LLM RL should move beyond recipe search toward joint evolution of policies and the training harnesses that interpret them.

**arXiv ID:** 2606.03108
</details>

<details>
<summary><strong>Dialogue SWE-Bench: A Benchmark for Dialogue-Driven Coding Agents</strong> - Brendan King, Jeffrey Flanigan - [[pdf]](https://arxiv.org/pdf/2606.13995)</summary>

**Abstract:** AI coding agents have rapidly transformed software engineering, powering widely used interactive coding assistants. Despite their interactive real-world use, existing benchmarks evaluate them as fully-autonomous systems. In this work, we introduce Dialogue SWE-Bench, an automatic benchmark dataset for evaluating the ability of coding agents to resolve real-world software engineering problems through dialogue with a user. We design a novel, persona-grounded user simulator to support our task evaluation, and augment our task evaluation with automatic evaluations of dialogue quality. We also propose a new schema-guided agent, aimed at improving the dialogue capabilities of off-the-shelf coding agents, which improves over strong baselines by 3-14%. Our results indicate that better coding models do not always correspond to better dialogue models, suggesting that dialogue capability is a distinct and currently understudied dimension of coding agent performance.

**arXiv ID:** 2606.13995
</details>

<details>
<summary><strong>Direct Preference Optimization for Chatbot Fine-Tuning: An Empirical Study</strong> - Dezhi Yu, Yvonne Qiu, ShuoJia Fu - [[pdf]](https://arxiv.org/pdf/2606.12881)</summary>

**Abstract:** We present an approach to fine-tuning large language models using Direct Preference Optimization (DPO), a reinforcement learning technique. Our experimental results demonstrate that DPO simplifies the training pipeline, improves computational efficiency, and achieves competitive performance. The evaluation using BLEU, ROUGE, and cosine similarity metrics indicates effective learning and convergence, though further investigation is needed to address observed training instability.

**arXiv ID:** 2606.12881
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (15 papers)</h2></summary>

<details>
<summary><strong>WorkBench Revisited: Workplace Agents Two Years On</strong> - Olly Styles - [[pdf]](https://arxiv.org/pdf/2606.13715)</summary>

**Abstract:** The best agent on WorkBench in March 2024, GPT-4, completed 43% of tasks and took an unintended harmful action, such as emailing the wrong person, on 26% of them. We re-visit the benchmark in June 2026 and find that the best agent to date, Claude Opus 4.8, completes 89% and takes an unintended harmful action on 2.5%. Aside from this considerable progress in frontier agent performance, three things stand out. First, capability and safety go together on WorkBench rather than trade off, so the models that finish the most tasks also do the least unintended damage. Second, while several classes of error have been totally eliminated, frontier models still make some basic mistakes that occasionally result in irreversible harm, such as sending an email to the wrong person. Third, the rise of open-weight models has drastically lowered costs for a performance level that was previously only accessible to proprietary models, while frontier costs have stayed relatively stable. We release an updated version of the benchmark with data and code quality improvements, new model scores, and analysis of agent progress on WorkBench since 2024.

**arXiv ID:** 2606.13715
</details>

<details>
<summary><strong>GitOfThoughts: Version-Controlled Reasoning and Agent Memory You Can Replay, Diff, and Merge</strong> - Pavan C Shekar, Abhishek H S, Aswanth Krishnan - [[pdf]](https://arxiv.org/pdf/2606.14470)</summary>

**Abstract:** Large language model (LLM) reasoning is ephemeral: chains of thought vanish with the context window, pruned search branches leave no record, and memory buffers cannot be diffed, merged, or audited. Every other complex software process (code, infrastructure, data, experiments) is version-controlled; reasoning is not. We introduce GitOfThoughts, which stores an agent's reasoning tree as a git repository: every scored thought is a commit, scores are notes, outcomes are tags, and retrieval is "git log" over the agent's own history. This makes reasoning replayable, auditable, and mergeable across agents at near-zero engineering cost.
We then ask the harder question: does memory, in any substrate, actually improve accuracy? Across five substrates (none, markdown, vector, graph, git), two benchmarks, two model scales, and pre-registered replications, the answer for novel problems is no. No memory format reliably helps, and a promising early result collapsed under its own pre-registered replication. Memory pays only above what we call the copyability threshold: when the retrieved case is a near-duplicate of the current problem (similarity >~ 0.8), accuracy jumps sharply; below it, nothing. The gain is answer retrieval, not method transfer: a 4.5x larger model doubles the near-duplicate payoff yet still cannot extract a transferable method from a worked example. The only general lever we find is test-time sampling. The case for git-as-substrate is therefore auditability, provenance, and mergeability at accuracy parity. We document a retracted result and a refuted hypothesis to model the evaluation standard we hold ourselves to.

**arXiv ID:** 2606.14470
</details>

<details>
<summary><strong>StreamMemBench: Streaming Evaluation of Agent Memory for Future-Oriented Assistance</strong> - Guanming Liu, Yuqi Ren, Hansu Gu, Peng Zhang, Weihang Wang, Jiahao Liu, Ning Gu, Tun Lu - [[pdf]](https://arxiv.org/pdf/2606.14571)</summary>

**Abstract:** A central role of personal-agent memory is to turn stored information and prior interactions into future-oriented assistance. In daily use, useful cues come from what the agent observes and how the user interacts with the agent, and the agent must carry them forward from the current request to similar future tasks. Existing memory benchmarks usually test dialogue recall or task improvement in isolation, leaving the trajectory from streaming observations to later assistance largely untested. We introduce StreamMemBench, a streaming benchmark that constructs a two-step task sequence around each evidence anchor from EgoLife egocentric streams. The initial task tests evidence use, while the follow-up task tests whether feedback and interaction experience are reused. Four metrics diagnose evidence recall, initial evidence use, feedback incorporation, and follow-up reuse. Experiments with eight memory systems across two backbones show that current systems often fail to use observed evidence or turn feedback into reliable follow-up behavior, even when evidence is stored or feedback is incorporated locally. StreamMemBench is publicly available at this https URL.

**arXiv ID:** 2606.14571
</details>

<details>
<summary><strong>Hidden in Plain Sight: Benchmarking Agent Safety Against Decomposition Attacks with DECOMPBENCH</strong> - Vikhyath Kothamasu, Virginia Smith, Chhavi Yadav - [[pdf]](https://arxiv.org/pdf/2606.13994)</summary>

**Abstract:** LLM-based Agents are becoming increasingly capable and widely deployed, creating growing incentives for adversarial misuse in the real-world. A key emerging threat is Decomposition Attacks \cite{glukhov2024breach, jones2024adversaries} in which a harmful task is broken into simpler, benign subtasks that evade safety mechanisms when executed separately but cumulatively fulfill the malicious intent. Although recent benchmarks assess agent safety in multi-turn and multi-tool-use settings, they do not explicitly capture this form of decompositional misuse and may not represent realistic adversarial execution flows. To this end, we introduce DeCompBench, a benchmark designed specifically to evaluate agentic safety under decomposition attacks. DeCompBench is created with a decomposition-by-design principle using a graphical framework and enables harmful task decomposition into individually benign and executable subtasks with realistic workflows. Our experiments using a custom decomposer show that state-of-the-art agents exhibit high refusal rates on monolithic harmful tasks, but significantly lower refusal rates on their decomposed variants, while often inadvertently fulfilling the adversarial objectives. These findings underscore the need for safety evaluations against decomposition attacks and corresponding defenses. Our dataset is publicly available and can be found at this https URL.

**arXiv ID:** 2606.13994
</details>

<details>
<summary><strong>Same-Origin Policy for Agentic Browsers</strong> - Xilong Wang, Xiaoxing Chen, Patrick Li, Dawn Song, Neil Gong - [[pdf]](https://arxiv.org/pdf/2606.14027)</summary>

**Abstract:** Agentic browsers integrate autonomous AI agents into web browsers, enabling users to accomplish web tasks through natural-language instructions. The same-origin policy (SOP) is a fundamental browser security mechanism that prevents unauthorized automated cross-origin data flows induced by scripts. However, whether SOP remains effective in agentic browsers is an open question that has not been systematically studied. In this work, we bridge this gap. We first observe that an agentic browser can itself serve as an automated channel for cross-origin data flows, potentially leading to SOP violations. To investigate this phenomenon, we construct SOPBench, a benchmark for evaluating SOP violations in agentic browsers. Our evaluation shows that existing agentic browsers frequently violate SOP, both in benign settings and under attacks. To address this problem, we propose SOPGuard, an SOP enforcement mechanism tailored to agentic browsers. We implement SOPGuard in BrowserOS, an open-source agentic browser. Extensive evaluations demonstrate that SOPGuard effectively enforces SOP while preserving utility and incurring only a small runtime overhead. Our code and data are available at this https URL.

**arXiv ID:** 2606.14027
</details>

<details>
<summary><strong>AgentCyberRange: Benchmarking Frontier AI Systems in Realistic Cyber Ranges</strong> - Fengyu Liu, Jiarun Dai, Yihe Fan, Wuyuao Mai, Ziao Li, Bofei Chen, Jie Zhang, Zheng Lou, Bocheng Xiang, Qiyi Zhang, Xudong Pan, Geng Hong, Yuan Zhang, Min Yang - [[pdf]](https://arxiv.org/pdf/2606.14295)</summary>

**Abstract:** Frontier AI systems are increasingly capable of cybersecurity tasks, including codebase inspection, vulnerability detection, and exploitation. However, evaluating their offensive capabilities remains constrained by limited access to open, reproducible, multi-host cyber ranges. Existing public benchmarks capture isolated skills such as CTF solving, vulnerability reproduction, and exploit generation, but often abstract away realistic intrusion workflows: discovering exposed services, gaining a foothold, collecting internal information, and expanding compromise across hosts. This gap makes it difficult to observe emerging risks early, because frontier AI systems are rarely evaluated under realistic attack conditions.
We introduce AgentCyberRange, the first open, multi-range infrastructure for measuring autonomous cyber attack capability in realistic cyber ranges. It combines 110 vulnerabilities across 15 real web applications and 8 enterprise-like cyber ranges with 156 internal hosts, plus Cage, a toolchain for execution, orchestration, result collection, and verification. The benchmark covers two core stages: web exploitation, where agents explore exposed applications and validate vulnerabilities, and post exploitation, where agents turn an initial foothold into broader internal compromise. We evaluate six frontier AI systems under matched prompts and budgets. GPT-5.5 with Codex performs best, solving 16.1% of web exploitation tasks and 31.7% of post-exploitation tasks; with more concrete hints, these rates increase to 33.0% and 46.3%. We also observe out-of-benchmark findings, including unknown vulnerabilities in popular projects, and payload mutation that bypasses host defenses. These results show that open cyber-range evaluation is necessary for observing emerging offensive capabilities under realistic and reproducible conditions.

**arXiv ID:** 2606.14295
</details>

<details>
<summary><strong>LLM-Powered AI Agent Systems and Their Applications in Industry</strong> - Guannan Liang, Qianqian Tong - [[pdf]](https://arxiv.org/pdf/2505.16120)</summary>

**Abstract:** The emergence of Large Language Models (LLMs) has reshaped agent systems. Unlike traditional rule-based agents with limited task scope, LLM-powered agents offer greater flexibility, cross-domain reasoning, and natural language interaction. Moreover, with the integration of multi-modal LLMs, current agent systems are highly capable of processing diverse data modalities, including text, images, audio, and structured tabular data, enabling richer and more adaptive real-world behavior. This paper comprehensively examines the evolution of agent systems from the pre-LLM era to current LLM-powered architectures. We categorize agent systems into software-based, physical, and adaptive hybrid systems, highlighting applications across customer service, software development, manufacturing automation, personalized education, financial trading, and healthcare. We further discuss the primary challenges posed by LLM-powered agents, including high inference latency, output uncertainty, lack of evaluation metrics, and security vulnerabilities, and propose potential solutions to mitigate these concerns.

**arXiv ID:** 2505.16120
</details>

<details>
<summary><strong>EurekAgent: Agent Environment Engineering is All You Need For Autonomous Scientific Discovery</strong> - Amy Xin, Jiening Siow, Junjie Wang, Zijun Yao, Fanjin Zhang, Jian Song, Lei Hou, Juanzi Li - [[pdf]](https://arxiv.org/pdf/2606.13662)</summary>

**Abstract:** LLM-based agents have shown increasing potential in automating scientific discovery. Given an optimizable metric and an execution environment, they can propose, validate, and iterate scientific solutions, and have produced results that outperform human-designed approaches. As model capabilities continue to improve, we argue that the bottleneck for autonomous scientific discovery is shifting from prescribing agent workflows to designing agent environments: the resources, constraints, and interfaces that shape agent behavior. We frame this as environment engineering: building environments that amplify productive behaviors, such as open-ended exploration, systematic artifact management, and inter-agent collaboration, while suppressing harmful behaviors, such as reward hacking and high-friction human oversight. We present EurekAgent, an environment-engineered agent system for metric-driven autonomous scientific discovery. EurekAgent engineers the environment along four dimensions: permissions engineering for bounded agent execution and isolated evaluation; artifact engineering for filesystem and Git-based collaboration; budget engineering for budget-aware exploration; and human-in-the-loop engineering for easy human supervision and intervention. EurekAgent sets new state-of-the-art results on multiple mathematics, kernel engineering, and machine learning tasks, including new state-of-the-art 26-circle packing results discovered with less than $11 in total API cost. We open-source our code and results, and call for environment engineering as a core research direction for developing reliable autonomous research agents.

**arXiv ID:** 2606.13662
</details>

<details>
<summary><strong>Benchmarking Web Agent Safety under E-commerce Deceptive Interfaces</strong> - Zijing Shi, Meng Fang, Ling Chen - [[pdf]](https://arxiv.org/pdf/2606.13686)</summary>

**Abstract:** As autonomous web agents are increasingly deployed to perform real-world tasks, ensuring their safety has become a critical concern. In this work, we study web agent behavior under realistic deceptive interfaces in the e-commerce domain. We introduce WebDecept, a lightweight and configurable plugin framework that enables controlled injection of deceptive interface patterns into existing web environments. Using WebDecept, we instantiate seven deceptive patterns commonly observed on the open web, including targeted advertisements, domain redirection, and shopping manipulation. By injecting these patterns into the frontend during task execution, we perform controlled evaluation of multiple multimodal web agents. Our results show that current web agents are highly susceptible to multiple classes of deceptive interfaces, and that prompt-based constraints are often insufficient to mitigate these failures. We further analyze how the design choices of deceptive patterns influence the success of such manipulations. These findings highlight safety challenges that should be addressed as web agents are scaled toward real-world deployment.

**arXiv ID:** 2606.13686
</details>

<details>
<summary><strong>An Empirical Study of Automating Agent Evaluation</strong> - Kang Zhou, Sangmin Woo, Haibo Ding, Kiran Ramnath, Subramanian Chidambaram, Aosong Feng, Vinayak Arannil, Muhyun Kim, Ishan Singh, Darren Wang, Zhichao Xu, Megha Gandhi, Nirmal Prabhu, Soumya Smruti Mishra, Vivek Singh, Gouri Pandeshwar, Lin Lee Cheong - [[pdf]](https://arxiv.org/pdf/2605.11378)</summary>

**Abstract:** Agent evaluation requires assessing complex multi-step behaviors involving tool use and intermediate reasoning, making it costly and expertise-intensive. A natural question arises: can frontier coding assistants reliably automate this evaluation process? Our study shows that simply prompting coding assistants is insufficient for this task. Without domain-specific evaluation knowledge, frontier coding assistants achieve only a 30% execution success rate and produce over-engineered evaluations averaging 12+ metrics per agent, indicating that strong coding ability does not automatically translate to reliable agent evaluation. We introduce EvalAgent, an AI assistant that automates the end-to-end agent evaluation pipeline. EvalAgent encodes evaluation domain expertise as evaluation skills (procedural instructions, reusable code and templates, and dynamically retrieved API documentation) that compose into a trace-based pipeline producing complete evaluation artifacts including metrics, executable code, and reports. To systematically assess generated evaluations, we introduce a meta-evaluation framework alongside AgentEvalBench, a benchmark comprising 20 agents, each paired with evaluation requirements and test scenarios. We further propose the Eval@1 metric to measure whether generated evaluation code both executes and yields meaningful results on the first run. Our experiments show that EvalAgent produces focused evaluations, improving Eval@1 from 17.5% to 65%, and achieving 79.5% human expert preference over baseline approaches. Further ablation studies show that evaluation skills are critical for handling complex evaluation: removing them causes Eval@1 to drop significantly from 65% to 30%.

**arXiv ID:** 2605.11378
</details>

<details>
<summary><strong>RedAct: Redacting Agent Capability Traces for Procedural Skill Protection</strong> - Shuwen Xu, Zhitao He, Yi R. Fung - [[pdf]](https://arxiv.org/pdf/2606.10813)</summary>

**Abstract:** Users rely on execution traces to observe agent behavior, diagnose failures, and ensure accountability. These traces contain rich procedural detail, including tool invocations, intermediate decisions, and error-recovery logic. Yet this detail can expose private procedural skills, allowing downstream methods to recover key formulas, thresholds, and strategies without access to model weights or skill files. To quantify this risk and evaluate protection, we construct CapTraceBench, a benchmark of 75 specialized long-horizon tasks and 154 curated skills across seven domains. We also introduce RedAct, a protected trace release framework that localizes protected key information, rewrites traces while preserving verifier-critical evidence, and embeds behavioral watermarks for downstream provenance analysis. Across representative trace reuse methods, RedAct reduces normalized skill transfer (NST) from 44.7-67.1% on raw traces to below the no-skill baseline, while preserving audit evidence. Its standalone behavioral watermarks reach 93.6-100.0% true detection with a false alarm rate of at most 1.9%. These results frame public agent traces as security interfaces and show that selective redaction can reduce procedural capability leakage without removing audit evidence.

**arXiv ID:** 2606.10813
</details>

<details>
<summary><strong>Running the Gauntlet: Re-evaluating the Capabilities of Agents Beyond Familiar Environments</strong> - Mykola Vysotskyi, Runqi Lin, Grzegorz Biziel, Michal Zakrzewski, Sebastian Montagna, Damian Rynczak, Shreyansh Padarha, Kumail Alhamoud, Zihao Fu, William Lugoloobi, Kai Rawal, Hanna Yershova, Xander Davies, Taras Rumezhak, Guohao Li, Fazl Barez, Baoyuan Wu, Arkadiusz Drohomirecki, Yarin Gal, Chris Russell, Christopher Summerfield, Adam Mahdi, Volodymyr Karpiv, Philip Torr, Adel Bibi - [[pdf]](https://arxiv.org/pdf/2606.14397)</summary>

**Abstract:** As agentic systems continue to evolve and are widely deployed in real-world scenarios, there is a growing demand to faithfully evaluate their capabilities. However, current benchmarks are typically built on popular applications with relatively simple tasks and focus on a narrow set of capabilities while overlooking broader dimensions, resulting in saturated performance on modern agents and failing to probe their limitations. To this end, we introduce GauntletBench, a web-based benchmark for evaluating agent generalisation in challenging scenarios, focusing on three underexplored capabilities (temporal perception, graphical understanding, and 3D reasoning), across five less-covered professional applications (Video Editor, Workflow Builder, 3D Modeller, Flight Analyser, and Circuit Designer), each with 20 vision-intensive tasks (100 in total). Our benchmark provides a modular pipeline that comprises an environment compatible with both open- and closed-source agent frameworks, a controlled web-based application, a well-structured task suite, and an automated evaluation engine with diverse metrics. Contrary to widespread expectations, our empirical results reveal that frontier agentic systems remain far from achieving human-level performance. Even the state-of-the-art agent achieves only a 19.1% success rate on our GauntletBench, highlighting the limitations in these overlooked capabilities and generalisation. By comparison, non-expert human annotators achieve over 80% success on our challenging yet feasible tasks, revealing the substantial gap between current agent capabilities and those required for complex real-world scenarios.

**arXiv ID:** 2606.14397
</details>

<details>
<summary><strong>PhysVLA: Towards Physically-Grounded VLA for Embodied Robotic Manipulation</strong> - Namai Chandra, Shriram Damodaran, Lin Wang - [[pdf]](https://arxiv.org/pdf/2606.13886)</summary>

**Abstract:** Vision-Language-Action (VLA) models excel at mapping visual inputs and natural language instructions directly to robotic control policies. However, because they are trained primarily to fit behavioural demonstration data, they do not explicitly enforce fundamental physical principles such as rigid-body dynamics or contact constraints. This exposes a critical physics gap: standard temporal smoothing applied on top of single-step or chunked VLAs trades trajectory quality for added failures that short-term memory cannot resolve. To bridge this gap, we introduce PhysVLA (Physics-VLA), a plug-and-play, inference-time framework designed to wrap any frozen VLA backbone without retraining, fine-tuning, or weight access, with less than 1 ms of overhead per control step. PhysVLA intercepts the predicted control action, captures only the simulator or system state, and applies a dual-layered correction: (i) a phase-aware finite-state machine that structures discrete task segments (approach, grasp, transport, and place), and (ii) a selective Euler-Lagrange gate that activates only when a dynamics oracle detects kinodynamic inconsistency. Evaluated across OpenVLA, OpenVLA-OFT, Force-VLA, and Generalist-VLA on LIBERO-Spatial with a 7-DoF Franka Panda, the framework delivers absolute success rate increases of up to 17% and stability increases of up to 19% with no per-task regressions, improves trajectory efficiency by up to 15% across all four backbones, and shows up to a 10x improvement in trajectory jerk robustness on a Robosuite Lift cross-simulator sweep. We further validate the framework on a real Agilex Piper arm with a pick-and-place task, confirming that PhysVLA transfers to physical hardware without retraining, with success-rate improvements of up to 50%, establishing physical awareness as a composable, backbone-agnostic runtime module.

**arXiv ID:** 2606.13886
</details>

<details>
<summary><strong>ReactSim-Bench: Benchmarking Reactive Behavior World Model Simulation in Autonomous Driving</strong> - Zhiyuan Zhang, Yanlun Peng, Jianing Zhang, Xianda Guo, Zehan Huang, Haoran Liu, Qifeng Li, Shaofeng Zhang, Xiaosong Jia, Junchi Yan - [[pdf]](https://arxiv.org/pdf/2606.14058)</summary>

**Abstract:** Reactive capability is a key property of data-driven behavior world model simulators for autonomous driving simulation systems. With this capability, simulated world agents can respond feasibly to autonomous vehicle (AV) behaviors that differ from the log. However, existing behavior simulation benchmarks do not directly measure reactive capability. They often let the simulator jointly control the AV and surrounding agents and evaluate realism through log similarity or open-loop prediction metrics. In this work, we introduce ReactSim-Bench for evaluating the reactive capability of behavior world model simulation in autonomous driving. We decouple the control of agents and the AV, using AV behaviors that differ from the log and require agents to respond as independent AV inputs. To obtain these AV behaviors, we construct a pipeline that uses an AV planner model to generate candidate behaviors and filters the data using rules and manual verification. Collision metrics, map-based metrics, and kinematic feasibility metrics are used to evaluate the safety and rule compliance of reactive responses. We construct 2,636 test scenarios with three categories and conduct a systematic evaluation of state-of-the-art models across multiple architectures, including Transformer-based, diffusion-based, and next-token-prediction-based models. We further analyze how replan frequency affects performance and provide insights for future studies.

**arXiv ID:** 2606.14058
</details>

<details>
<summary><strong>ADAPT: An Autonomous Forklift for Construction Site Operation</strong> - Johannes Huemer, Markus Murschitz, Matthias Schörghuber, Lukas Reisinger, Thomas Kadiofsky, Christoph Weidinger, Mario Niedermeyer, Benedikt Widy, Marcel Zeilinger, Csaba Beleznai, Tobias Glück, Andreas Kugi, Patrik Zips - [[pdf]](https://arxiv.org/pdf/2503.14331)</summary>

**Abstract:** Efficient material logistics play a critical role in controlling costs and schedules in the construction industry. However, manual material handling remains prone to inefficiencies, delays, and safety risks. Autonomous forklifts offer a promising solution to streamline on-site logistics, reducing reliance on human operators and mitigating labor shortages. This paper presents the development and evaluation of ADAPT (Autonomous Dynamic All-terrain Pallet Transporter), a fully autonomous off-road forklift designed for construction environments. Unlike structured warehouse settings, construction sites pose significant challenges, including dynamic obstacles, unstructured terrain, and varying weather conditions. To address these challenges, our system integrates AI-driven perception techniques with traditional approaches for decision making, planning, and control, enabling reliable operation in complex environments. We validate the system through extensive real-world testing, comparing its continuous performance against an experienced human operator across various weather conditions. Our findings demonstrate that autonomous outdoor forklifts can operate near human-level performance, offering a viable path toward safer and more efficient construction logistics.

**arXiv ID:** 2503.14331
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>Capability Minimization as a Safety Primitive: Risk-Aware Causal Gating for Least-Privilege LLM Agents</strong> - Laxmipriya Ganesh Iyer, Rahul Suresh Babu - [[pdf]](https://arxiv.org/pdf/2606.13884)</summary>

**Abstract:** Modern decision systems increasingly rely on learned components whose outputs may be confident yet wrong, exposing downstream actions to costly errors. We introduce Risk-Aware Causal Gating (RACG), a framework that decides whether to act on, defer, or abstain from a model's prediction by combining causal effect estimation with calibrated risk control. RACG models the causal pathway from candidate actions to outcomes and gates each decision according to an estimated counterfactual risk rather than raw predictive confidence. To make gating reliable, we derive distribution-free bounds on the probability of acting under high-risk conditions and show how these bounds translate into operating thresholds that satisfy user-specified safety constraints. We further propose an adaptive gating policy that adjusts to distribution shift by monitoring discrepancies between predicted and realized outcomes, tightening the gate when causal assumptions appear violated. Across simulated interventions and real-world decision benchmarks, RACG reduces high-cost errors substantially while preserving most of the utility of an ungated policy, and it outperforms confidence-based and selective-prediction baselines at matched abstention rates. Our results indicate that explicitly separating causal risk from predictive uncertainty yields decision systems that are both safer and more transparent, offering a principled mechanism for trustworthy automation in high-stakes settings.

**arXiv ID:** 2606.13884
</details>

<details>
<summary><strong>When Should Agent Trust Be Conditional? Characterizing and Attacking Skill-Conditional Reputation in Agent Swarms</strong> - Yihan Xia, Taotao Wang - [[pdf]](https://arxiv.org/pdf/2606.14200)</summary>

**Abstract:** Open platforms increasingly route tasks among heterogeneous LLM agents--differing in base model, scaffold, and tool stack--whose competence varies sharply by skill: an agent excellent at one skill may be useless at another. The standard reputation approach summarizes each agent by a single global trust score, but that scalar is the wrong object here, because routing every task to the globally most-trusted agent leaves the value of specialization unclaimed. We study skill-conditional trust R(i | k)--the trust to place in agent i for a task requiring skill k, rather than one score per agent--and pose three falsifiable questions: when is conditioning worth it, how much cross-skill evidence should be borrowed, and whether that borrowing is safe. A controlled phase-diagram analysis answers the first two: conditional trust wins only in a specific regime--high agent heterogeneity, sparse per-skill evidence, and correlated skills--and the coupling strength beta that buys this data efficiency is dual-use, because the same cross-skill borrowing is also a laundering channel. On a public benchmark of 14 genuinely heterogeneous AppWorld agents, real pools land inside the beneficial regime--a small but genuine gain, with the per-skill best agent genuinely changing across skills. We then show that an attacker with cheap evidence in one skill and none in a target skill hijacks the conditional router, driving routing regret from 0 to 0.94 on a pool our zero-cost Conditional Information Value Test (CIVT) rates GREEN--while the ungated trust verdict it contaminates reads -0.06 instead of the honest +0.19. A zero-evidence gate bounds the attack but does not eliminate it; we characterize the residual cost under an explicit budget. We do not claim Sybil-resistance--we quantify the trade-off.

**arXiv ID:** 2606.14200
</details>

<details>
<summary><strong>Closing the Reflection Gap: A Free Calibration Bonus for Agentic RL</strong> - Yinglun Zhu - [[pdf]](https://arxiv.org/pdf/2606.14211)</summary>

**Abstract:** LLMs are increasingly deployed as agents that interact with external environments and observe feedback such as execution results, error messages, and tool outputs. A well-functioning agent should be able to leverage this feedback to accurately assess its own performance. Yet we find a persistent reflection gap: LLM agents tend to mis-assess their own outputs after observing concrete environment feedback -- even for questions they correctly answered -- and standard RL barely helps due to a credit-assignment mismatch. To close this gap, we propose RefGRPO, a simple yet effective fix that augments standard RL algorithms with two key ingredients: a free calibration bonus computed by contrasting the agent's own reflection with the actual outcome (requiring no additional reward model, LLM judge, or external annotation), and a dynamic schedule on its coefficient. Compared to standard RL baselines, our method simultaneously improves reflection calibration (e.g., reduces underconfidence rate $44.4\% \to 7.7\%$) and task accuracy (e.g., $75.1\% \to 76.5\%$) on text-to-SQL across five benchmarks. The resulting calibrated reflection turns the agent into its own verifier grounded in environment feedback, which further enables (i) better self-improvement that uses reflections as pseudo-rewards without outcome supervision, and (ii) more effective test-time selective prediction by committing only to rollouts flagged as correct.

**arXiv ID:** 2606.14211
</details>

<details>
<summary><strong>Communication Policy Evolution for Proactive LLM Agents</strong> - Xinbei Ma, Jiyang Qiu, Yao Yao, Zheng Wu, Yijie Lu, Xiangmou Qu, Jiaxin Yin, Xingyu Lou, Jun Wang, Weiwen Liu, Weinan Zhang, Zhuosheng Zhang, Hai Zhao - [[pdf]](https://arxiv.org/pdf/2606.14314)</summary>

**Abstract:** LLM agents have rapidly evolved into autonomous systems, yet a persistent information gap remains between users and agents: communication is costly, while users' identical preferences further limit information exchange. To investigate how agents should communicate across modalities, this paper formalizes Communication Policy, establishes textual and UI-based policies, and then evaluates communication policies across diverse environments, personas, and model combinations. Building information asymmetry for proactive agents, we set up two complementary settings, User-Agent and Planner-Executor. Experimental results reveal complementary strengths between interaction channels: text-based interaction often facilitates task performance, while structured UI improves agents' response quality and persona compliance. Motivated by that, a hybrid method combines these advantages. We further propose Communication Policy Evolution (CPE), a self-evolution framework for refining communication policies through rollout and prompt-level evolving. Without model modification, CPE achieves the best task success across multiple settings using prompt refinement alone. Our findings identify communication behavior as a critical yet underexplored design dimension for LLM agents.

**arXiv ID:** 2606.14314
</details>

<details>
<summary><strong>When the Tool Decides: LLM Agents Defer Blindly to Graph Neural Network Tools, and Stronger Backbones Defer More</strong> - Zhongyuan Wang, Pratyusha Vemuri - [[pdf]](https://arxiv.org/pdf/2606.14476)</summary>

**Abstract:** A growing line of work equips large language model (LLM) agents with graph neural networks (GNNs) as callable tools, assuming the agent exercises judgment over when and how much to rely on such a tool. We test this directly. We expose a frozen GNN to a ReAct-style LLM agent as an explicit tool and measure, on node classification over a text-attributed graph (ogbn-arxiv, replicated on WikiCS), whether the agent uses the tool or merely obeys it. We find the agent does not exercise judgment: its predictions agree with the raw GNN's 97.6-99.2% of the time (5 seeds), collapsing into a GNN parrot that adopts the tool's output wholesale and bypasses its own reasoning. Sweeping backbone capability (Qwen2.5 0.5B-7B), the deference is not a weak-model artifact: among models able to invoke the tool, agreement rises with capability (0.60 to 0.98 from 1.5B to 7B). Crucially, the cost of deference does not shrink as capability grows and grows where alternatives emerge: a per-node oracle over the available actions beats the parrot by 0.09-0.18 at 3B and 0.12-0.22 at 7B, roughly doubling at high homophily, because the parrot is pinned to the frozen GNN while the agent's alternatives improve; at 7B a simple neighbour-label tool overtakes the GNN at high homophily (0.81 vs 0.71) yet the agent still defers. A simple selective-invocation gate recovers about half of that high-homophily gap (0.71 to 0.83) but yields no net global gain, and held-out estimates bound the best achievable gate over standard test-time features to at most a third of the oracle headroom: reliable selective invocation looks limited by available information, not merely router design. Our results are a cautionary measurement: evaluations of agent+tool systems cannot assume the agent adds judgment on top of the tool, and selective invocation must be designed in rather than expected to emerge from scale.

**arXiv ID:** 2606.14476
</details>

<details>
<summary><strong>SANA: What Matters for QA Agents over Massive Data Lakes?</strong> - Austin Senna Wijaya, Jiaxiang Liu, Haonan Wang, Eugene Wu - [[pdf]](https://arxiv.org/pdf/2606.13904)</summary>

**Abstract:** Exploratory question answering (EQA) over data lakes requires an LLM agent to discover relevant sources, analyze retrieved data, and adapt its actions based on intermediate results. End-to-end accuracy alone cannot distinguish failures in search, planning, data analysis, or the agent's Action Policy: its decisions about what to do next and when to submit an answer. We present SANA (Search Agent Navigation Ablation framework), a diagnostic ablation framework that transforms EQA tasks into runtime profiles containing gold source sequence, sanitized subquestions, and execution records. SANA uses these profiles to construct idealized search, planning, and data-analysis tools, allowing each component to be ablated; the residual gap is diagnostic evidence for policy failures.
To illustrate SANA as a reusable evaluation framework, we adapted two recent EQA benchmarks, LakeQA and KramaBench, and evaluated lightweight and mid-sized agents under fixed prompts, budgets, data lakes, and runtimes. Across both benchmarks, data analysis is a consistent bottleneck while planning is less so. Search is a major limitation in LakeQA's large data-lake setting, but less so for the smaller-scale KramaBench. SANA thus deconstructs end-to-end task accuracies into a diagnosis of where data-lake agents fail, and allows for systematic comparisons of progress in search, planning, data analysis, and agent design.

**arXiv ID:** 2606.13904
</details>

<details>
<summary><strong>When Errors Become Narratives: A Longitudinal Taxonomy of Silent Failures in a Production LLM Agent Runtime</strong> - Wei Wu - [[pdf]](https://arxiv.org/pdf/2606.14589)</summary>

**Abstract:** LLM agent systems increasingly run as long-lived autonomous runtimes: scheduling jobs, calling tools, maintaining memory, and pushing results to humans. We present a longitudinal study of silent failures in one such system: a personal-assistant agent runtime in continuous production since March 2026, with roughly 40 scheduled jobs, 8 LLM providers, a tool-governance proxy, and a knowledge-base memory plane, defended by 4,286 unit tests and 827 governance checks. Over eight weeks we documented 22 incidents with full root-cause postmortems, in which one meta-pattern -- a failure whose error signal never reaches a human in actionable form -- manifested at least 28 times. We derive a five-class, mechanism-oriented taxonomy: (A) environment and platform quirks, (B) design-assumption mismatches, (C) error swallowing and dilution, (D) chained hallucination and fabrication, (E) operational omission and forensic blind spots. Class D is unique to LLM systems and the most dangerous: the system does not merely fail to report an error -- the LLM transforms it into fluent, plausible narrative delivered to the user. We term this fail-plausible: gray failure's differential observability escalated -- the observer is not just blind, it is convincingly lied to by the failure itself. Three findings: about 70% of silent failures were caught by human user-view observation, not tests or audits; a retrospective audit of 15 incidents found 0% ex-ante prevention but 87% regression blocking -- audits are regression engines, not prediction engines; incident latency (13 hours to 60 days) tracks failure mechanism, not code complexity -- the longest-lived failures lived in the seams between components, where no test runs. We describe the resulting defense framework and distill design principles for agent systems whose failures are loud, attributable, and boring. All postmortems and artifacts are public.

**arXiv ID:** 2606.14589
</details>

<details>
<summary><strong>Retrospective Progress-Aware Self-Refinement for LLM Agent Training</strong> - Xinbei Ma, Congmin Zheng, Jiyang Qiu, Jiale Hong, Yao Yao, Xiangmou Qu, Jiaxin Yin, Xingyu Lou, Jun Wang, Weiwen Liu, Weinan Zhang, Zhuosheng Zhang, Hai Zhao - [[pdf]](https://arxiv.org/pdf/2606.14302)</summary>

**Abstract:** LLM-based agents trained with reinforcement learning optimize step-wise action prediction but lack metacognitive awareness of task progress, inducing a gap that hinders long-horizon scaling. A pilot study reveals that online progress prompting hurts performance while retrospective demonstrations help, yet this capability cannot emerge from outcome-reward training alone. We present RePro, Retrospective Progress-Aware Training, a framework that trains agents to self-generate progress signals via a forward-then-reflect rollout paradigm: the agent executes actions online, then retrospectively reassesses its step-wise progress given the completed trajectory and known outcome. RePro initializes with a Retrospection Warmup that teaches reflection format from minimal external demonstrations, then further trains through RePro-PO with a composite reward that produces self-generated signals without continuous external supervision. Experiments on WebShop, ALFWorld, and Sokoban show that RePro enhances the Qwen family's performance, with up to $12\%$ absolute success rate gains.

**arXiv ID:** 2606.14302
</details>

<details>
<summary><strong>AgentSpec: Understanding Embodied Agent Scaffolds Through Controlled Composition</strong> - Jixuan Chen, Jianzhi Shen, Haoqiang Kang, Zhi Hong, Qingyi Jiang, Soham Bose, Yiming Zhang, Leon Leng, Amit Vyas, Lingjun Mao, Siru Ouyang, Kun Zhou, Lianhui Qin - [[pdf]](https://arxiv.org/pdf/2606.14674)</summary>

**Abstract:** LLM agents are increasingly built not as single model calls, but as scaffolded systems that combine reasoning, memory, reflection, action execution, and learning. While such scaffolds often improve performance, they are often embedded in tightly coupled pipelines, making it difficult to isolate component contributions, compare alternative designs, or understand how module interactions shape agent behavior. We introduce AgentSpec, a modular specification framework that represents embodied agents as typed compositions of reusable policy components with standardized interfaces. AgentSpec standardizes the interfaces among perception, memory, reasoning, reflection, action, and optional learning, enabling components to be swapped and recombined under controlled conditions. We instantiate this framework across DeliveryBench, ALFRED, MiniGrid, and RoboTHOR, and analyze reasoning, memory, reflection, and reinforcement-learning modules across model backbones. Our results show that agent performance is governed by scaffold compatibility and interaction effects rather than isolated module strength. In particular, structured multi-granularity memory improves long-horizon state tracking, reasoning and memory interact non-uniformly across environments, reflection trades off correction and cost, and RL-trained policies compose best when optimized with deployment-time scaffold structure. AgentSpec provides a controlled foundation for studying, comparing, and designing composable LLM agents. Our code, baselines and interactive playground are publicly available at this https URL.

**arXiv ID:** 2606.14674
</details>

<details>
<summary><strong>Graph-based Target Back-Propagation for Context Adaptation in Multi-LLM Agentic Systems</strong> - Tan Zhu, Tong Yao, Kananart Kuwaranancharoen, Amit Singh, Yushang Lai, Deepa Mohan, Shankara Bhargava - [[pdf]](https://arxiv.org/pdf/2606.14155)</summary>

**Abstract:** Context adaptation automates prompt engineering in LLM-based systems by iteratively revising tunable prompts from task feedback, without modifying model weights. Extending this paradigm to multi-LLM agentic systems is crucial: existing methods suffer from inaccurate credit assignment and lack convergence guarantees. We propose \textbf{G}raph-based \textbf{T}arget \textbf{B}ack-\textbf{P}ropagation (GTBP), a context adaptation framework for agentic workflows modeled as directed acyclic graphs. GTBP propagates local target outputs backward through the workflow graph and uses target--output discrepancies to guide a stage-wise prompt update mechanism. Theoretically, we show that GTBP's stage-wise prompt updates become stable over iterations, and that a sufficiently capable LLM optimizer can decrease the overall objective. Empirically, GTBP consistently outperforms strong baselines across three benchmarks while maintaining comparable computational cost.

**arXiv ID:** 2606.14155
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (23 papers)</h2></summary>

<details>
<summary><strong>Orchestra-o1: Omnimodal Agent Orchestration</strong> - Fan Zhang, Vireo Zhang, Shengju Qian, Haoxuan Li, Hao Wu, Jinyang Wu, Donghao Zhou, Zhihong Zhu, Zheng Lian, Xin Wang, Pheng-Ann Heng - [[pdf]](https://arxiv.org/pdf/2606.13707)</summary>

**Abstract:** The recent success of agent swarms has shifted the paradigm of large language model (LLM)-based agents from single-agent workflows to multi-agent systems, highlighting the importance of agent orchestration for task decomposition and collaboration. However, existing orchestration frameworks are limited to a narrow set of modalities and struggle to generalize to more complex settings where heterogeneous modalities coexist and interact. This limitation becomes particularly pronounced in omnimodal scenarios, where tasks require the unified understanding and coordination of diverse inputs such as text, image, audio, and video. In this work, we propose Orchestra-o1, an omnimodal agent orchestration framework designed to support efficient agent collaboration across multiple modalities. Orchestra-o1 introduces a unified orchestration mechanism that enables modality-aware task decomposition, online sub-agent specialization, and parallel sub-task execution. This scalable design allows agent systems to effectively tackle complex real-world tasks involving heterogeneous information sources, surpassing the second-best approach by 10.3% accuracy on the OmniGAIA benchmark. Furthermore, we introduce decision-aligned group relative policy optimization (DA-GRPO), an efficient agentic reinforcement learning approach for training Orchestra-o1-8B, which also achieves state-of-the-art performance against all existing open-source omnimodal agents.

**arXiv ID:** 2606.13707
</details>

<details>
<summary><strong>YeasierAgent: Agentic Social Sandbox as a Canvas for Intent-Driven Creation of Platform-Agnostic Symbiotic Agent-Native Applications</strong> - Jory He - [[pdf]](https://arxiv.org/pdf/2606.13722)</summary>

**Abstract:** This paper introduces YeasierAgent, an application-building paradigm based on symbiotic agents, narrative worlds, and scene-aware interaction. It challenges the conventional device-coupled model of software by redefining applications as collaborative spaces among users, agents, and worlds. We present a system architecture that achieves two primary contributions: (1) enabling the rapid, cross-platform construction of agent-native applications by utilizing platform-agnostic interactive units (agents, scenes, dialogue) rather than fixed graphical layouts; and (2) unifying the emotional companionship and practical tool execution attributes of intelligent agents within a single experiential sandbox. By integrating automated generation, user-created worlds, and spatial multi-agent collaboration, YeasierAgent formalizes the category of Symbiotic Agent-Native Applications, demonstrating a shift from isolated, tool-specific chatbots toward cohesive, socially embedded computational environments.

**arXiv ID:** 2606.13722
</details>

<details>
<summary><strong>A Multi-Agent AI System for Automated High School Transcript Processing: Collaborative Document Analysis at Scale</strong> - Ben Torkian, Jun Zhou - [[pdf]](https://arxiv.org/pdf/2606.13916)</summary>

**Abstract:** Each year, college admissions offices face an overwhelming challenge: processing millions of high school transcripts, each with unique formats, grading systems, and layouts. This manual process creates operational bottlenecks that delay admissions decisions and consume valuable resources. We present a transformative solution through a multi-agent AI system where specialized agents collaborate to automatically process diverse transcript formats through intelligent coordination and communication. Our multi-agent architecture consists of three specialized agents-a Pattern Recognition Agent for format-specific parsing, a Semantic Analysis Agent for natural language understanding, and a Vision Intelligence Agent for multimodal document analysis-coordinated by an Orchestration Agent that manages agent communication and result reconciliation. Our key innovation lies in agent-based quality control using GPA extraction as a coordination signal, ensuring reliable agent collaboration and preventing critical information loss. When evaluated on 40 real world transcripts from high schools across 13 U.S. states, our agent system successfully processed every document, achieving 96.7% accuracy compared to expert manual review while maintaining practical processing speeds of 45 seconds per transcript. This work demonstrates how multi-agent coordination can solve complex document processing challenges, offering institutions a scalable, collaborative AI solution that preserves accuracy while dramatically reducing processing time.

**arXiv ID:** 2606.13916
</details>

<details>
<summary><strong>HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry</strong> - Tingyang Chen, Shuo Lu, Kang Zhao, Weicheng Meng, Hanlin Teng, Tianhao Li, Chao Li, Xule Liu, Jian Liang, Zhizhong Zhang, Yuan Xie, Heng Qu, Kun Shao, Jian Luan - [[pdf]](https://arxiv.org/pdf/2606.14249)</summary>

**Abstract:** AI agent performance depends critically on the runtime harness, comprising the prompts, tools, memory, and control flow that mediate how a model observes, reasons, and acts. Yet today's harnesses remain largely hand-crafted and static: each new model or task still demands bespoke scaffolding, and the rich traces produced during execution are rarely distilled back into systematic improvement. We introduce HarnessX, a foundry for composable, adaptive, and evolvable agent harnesses. HarnessX assembles typed harness primitives via a substitution algebra, adapts them through AEGIS, a trace-driven multi-agent evolution engine grounded in an operational mirror between symbolic adaptation and reinforcement learning, and closes the harness-model loop by turning trajectories into both harness updates and model training signal. Across five benchmarks (ALFWorld, GAIA, WebShop, tau^3-Bench, and SWE-bench Verified), HarnessX yields an average gain of +14.5% (up to +44.0%), with gains largest where baselines are lowest. These results suggest that agent progress need not come from model scaling alone: composing and evolving runtime interfaces from execution feedback is an actionable and complementary lever. The complete codebase will be open-sourced in a future release.

**arXiv ID:** 2606.14249
</details>

<details>
<summary><strong>Towards Direct Latent-Space Synthesis for Parallel Branches in LLM-Agent Workflows</strong> - Shikun Liu, Mufei Li, Dongqi Fu, Haoyu Wang, Yinglong Xia, Hong Li, Hong Yan, Pan Li - [[pdf]](https://arxiv.org/pdf/2606.14672)</summary>

**Abstract:** Large language models increasingly serve as execution engines for agentic systems, yet they still consume context through a sequential text interface. This creates a mismatch with modern structured agent workflows, in which independent branches explore subtasks, retrieve evidence, or generate candidate solutions before a final synthesis step. Existing systems typically merge these branches by concatenating their textual outputs, which discards the parallel structure and incurs redundant prefill computation. In this work, we introduce Parallel-Synthesis, a plug-and-play framework that enables a synthesizer to directly consume the KV caches produced by parallel worker agents. Parallel-Synthesis combines a cache mapper that calibrates independently generated branch caches with a fine-tuned synthesizer adapter that enables generation from this non-sequential cache interface. We train Parallel-Synthesis using data that exposes the synthesizer to parallel cache contexts, teaches aggregation across cached branches, and distills reasoning behavior from standard text-concatenation-based synthesis. Across nine downstream datasets spanning math, science QA, code generation, GAIA, and multi-agent database diagnosis, Parallel-Synthesis matches or outperforms text-based synthesis on seven datasets and remains close on the other two. It also reduces time-to-first-token by 2.5x-11x, suggesting that direct cache-based synthesis is a promising interface for more native and efficient synthesis over parallel agent branches.

**arXiv ID:** 2606.14672
</details>

<details>
<summary><strong>An Agentic Retrieval Framework for Autonomous Context-Aware Data Quality Assessment</strong> - Hadi Fadlallah, Ibrahim Dhaini, Fatima Mubarak, Rima Kilany - [[pdf]](https://arxiv.org/pdf/2606.13692)</summary>

**Abstract:** Data quality assessment is a critical prerequisite for effective data analytics and data-driven decision-making, yet it remains a challenging task due to the inherently context-dependent nature of data quality. Existing approaches often rely on static rules or manual assessment strategies, limiting their adaptability to diverse usage scenarios and constraining automation at scale. Recent advances in artificial intelligence, particularly large language models, offer new opportunities for automating data quality assessment, but raise concerns related to reliability, grounding, and execution safety.
In this paper, we propose a unified agentic-retrieval framework for autonomous context-aware data quality assessment. The framework interprets natural-language descriptions of intended data usage, derives context-aware assessment strategies, and generates executable validation logic through a multi-agent workflow. To ensure operational reliability, the framework introduces a feasibility validation stage that evaluates the realism and executability of generated assessment specifications before execution, enabling iterative refinement when necessary. Accepted validation logic is executed deterministically to guarantee reproducible and auditable results.
We implement the proposed framework as an end-to-end prototype and evaluate it across multiple usage scenarios applied to the same dataset. The results demonstrate that assessment outcomes adapt meaningfully to different intended uses, while feasibility-gated execution reduces unrealistic or non-executable rule generation. The proposed approach provides a practical foundation for deploying autonomous yet controlled data quality assessment in modern data-driven environments.

**arXiv ID:** 2606.13692
</details>

<details>
<summary><strong>Safety-Contract Graph Multi-Agent Reinforcement Learning for Autonomous Network Security Response</strong> - Jose Luis Lima de Jesus Silva - [[pdf]](https://arxiv.org/pdf/2606.13832)</summary>

**Abstract:** Autonomous network-security response systems promise to reduce Security Operations Centre (SOC) reaction latency, but reward-only multi-agent reinforcement learning (MARL) can improve security reward while remaining non-deployable. We present a safety-contract graph MARL framework and instantiate it as ACD$^3$-GAT (Adaptive Constrained Counterfactual Decisioning with a Graph Attention Network encoder), an architecture that separates simulator observations from reusable operational budgets, constrained optimization, graph state encoding, and counterfactual action screening. We evaluate the method in CAGE Challenge 4, where agents operate under budgets for Mean Time to Recover (MTTR), false-positive response, and firewall change-management disruption. Across the benchmark, every unconstrained method violates the SOC downtime budget in 100% of evaluated episodes, with mean downtime proxy costs of 311-430 against a budget of 50. This complements prior CAGE Challenge 4 findings by showing that reward-only learning lacks operational discipline. Constrained MAPPO-GAT (C-MAPPO-GAT) isolates Lagrangian operational-cost control and budget-aware screening, while ACD$^3$-GAT adds budget context, CVaR tail-risk estimation, opponent-belief state, and Graph Counterfactual Risk Propagation (G-CRP). The replicated comparison includes three 200-episode seeds for IPPO, MAPPO-GAT, C-MAPPO-GAT, and ACD$^3$-GAT. C-MAPPO-GAT reduces downtime violation from 100% to 0.3% and mean downtime cost from 355.4 to 15.5 relative to MAPPO-GAT. ACD$^3$-GAT reduces mean downtime cost to 48.2 with a 13.8% violation rate, placing it on the safety-contract frontier rather than at the most conservative compliance point. Topology-seed and coupled adaptive Red-process stress tests preserve this contrast and show lower worst adaptive degradation for safety-constrained policies than reward-only MAPPO-GAT.

**arXiv ID:** 2606.13832
</details>

<details>
<summary><strong>tap: A File-Based Protocol for Heterogeneous LLM Agent Collaboration</strong> - Minseo Kim - [[pdf]](https://arxiv.org/pdf/2606.14445)</summary>

**Abstract:** Existing multi-agent software development systems have proposed many forms of agent collaboration, including role-based collaboration and automated code review. However, many systems assume a common runtime, a central conversation server, or the same API family. Under these assumptions, LLM agents from different vendors cannot easily exchange messages directly from their own execution environments while dividing development and review work on a shared codebase. This paper presents tap, a file-based collaboration protocol that allows Claude (Anthropic) and Codex (OpenAI) to collaborate on one codebase without shared memory or an identical runtime. The core of tap is a file-first design that preserves markdown files with metadata as original messages, combines a file inspection path (file communication, Tier 1) with real-time notification paths for Claude and Codex (real-time communication, Tier 2), and isolates work through separate git worktrees. Even if real-time notification fails or a receiver restarts, the message file remains available and the same content can be inspected again. In a 27-day, 37-generation self-applied operation where tap was used to develop and review itself, we collected 209 tap-related pull requests and 717 operational artifacts. An analysis of 375 review artifacts showed that the share of reviews recording at least one defect or requested change was 69.8% for heterogeneous model pairs and 53.1% for homogeneous model pairs. These results show that tap, which combines file-based message preservation with real-time notification, operates in a real production repository, and that combining heterogeneous models and execution environments can broaden review perspectives. tap is distributed as the open-source npm package @hua-labs/tap (v0.5.2).

**arXiv ID:** 2606.14445
</details>

<details>
<summary><strong>From Shield to Target: Denial-of-Service Attacks on LLM-Based Agent Guardrails</strong> - Yuguang Zhou, Xunguang Wang, Pingchuan Ma, Zhantong Xue, Zhaoyu Wang, Shuai Wang - [[pdf]](https://arxiv.org/pdf/2606.14517)</summary>

**Abstract:** LLM-based guardrails have emerged as a highly effective defense against prompt injection and jailbreak attacks in autonomous agents. However, we reveal that the very reasoning and task-following capabilities enabling this protection introduce a novel vulnerability: attackers can inject crafted data to trap the guardrail in extended reasoning loops, effectuating a systematic denial-of-service (DoS) attack. To systematically expose this threat, we design a beam-search optimization framework that crafts natural-language payloads to maximize guardrail reasoning length, utilizing an LLM proposer guided by a strategy bank. Based on the observation of guardrail's schema-following nature, we also provide another attack framework driven by mechanism-aware structural mutations with less computational load. The attack efficacy is systematically evaluated in two parts. First, in standalone evaluations, the attack generalizes across diverse guardrail architectures, safety templates, and agent benchmarks. Payloads optimized on a single open-source surrogate successfully transfer to eight leading model backbones (e.g., Claude, GPT, Gemini, DeepSeek, and Qwen), achieving a 13--63$\times$ token amplification. Second, in end-to-end real-world agent deployments (web, desktop, code, and multi-agent systems), the attack reveals up to a 148$\times$ latency amplification. We show that a single poisoned document can saturate shared guardrail infrastructures, effectively starving co-located agents and paralyzing the entire system. By uncovering this availability flaw, our work underscores the urgent need to develop cost-bounded, reasoning-robust guardrails.

**arXiv ID:** 2606.14517
</details>

<details>
<summary><strong>Learning Coordinated Preference for Multi-Objective Multi-Agent Reinforcement Learning</strong> - Pengxin Wang, Lihao Guo, Yi Xie, Bo Liu, Siyang Cao, Jingdi Chen - [[pdf]](https://arxiv.org/pdf/2606.14693)</summary>

**Abstract:** Cooperative multi-objective multi-agent reinforcement learning (MOMARL) models team decision making under multiple, potentially conflicting objectives. In this setting, conflicts arise not only across objectives but also across agents with different observations, roles, and contributions. We propose Preference Coordinated Multi-agent Policy Optimization (PCMA), which learns coordinated agent-specific preferences to enable complementary trade-offs among agents. Theoretically, we formulate cooperative MOMARL as a team-optimal game and show that, under suitable conditions, preference diversity can induce team improvement through a first-order improvement decomposition. Experiments on multiple cooperative MOMA environments and a practical traffic-control scenario show that PCMA improves both performance and trade-off coordination.

**arXiv ID:** 2606.14693
</details>

<details>
<summary><strong>MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems</strong> - Rui Ye, Keduan Huang, Qimin Wu, Yuzhu Cai, Tian Jin, Xianghe Pang, Xiangrui Liu, Jiaqi Su, Chen Qian, Bohan Tang, Kaiqu Liang, Jiaao Chen, Yue Hu, Zhenfei Yin, Rongye Shi, Bo An, Yang Gao, Wenjun Wu, Lei Bai, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2505.16988)</summary>

**Abstract:** LLM-based multi-agent systems (MAS) have demonstrated significant potential in enhancing single LLMs to address complex and diverse tasks in practical applications. Despite considerable advancements, the field lacks a unified codebase that consolidates existing methods, resulting in redundant re-implementation efforts, unfair comparisons, and high entry barriers for researchers. To address these challenges, we introduce MASLab, a unified, comprehensive, and research-friendly codebase for LLM-based MAS. (1) MASLab integrates over 20 established methods across multiple domains, each rigorously validated by comparing step-by-step outputs with its official implementation. (2) MASLab provides a unified environment with various benchmarks for fair comparisons among methods, ensuring consistent inputs and standardized evaluation protocols. (3) MASLab implements methods within a shared streamlined structure, lowering the barriers for understanding and extension. Building on MASLab, we conduct extensive experiments covering 10+ benchmarks and 8 models, offering researchers a clear and comprehensive view of the current landscape of MAS methods. MASLab will continue to evolve, tracking the latest developments in the field, and invite contributions from the broader open-source community.

**arXiv ID:** 2505.16988
</details>

<details>
<summary><strong>Contract-Based Compositional Shielding for Safe Multi-Agent Reinforcement Learning</strong> - Omar Adalat, Edwin Hamel-De le Court, Francesco Belardinelli - [[pdf]](https://arxiv.org/pdf/2606.14130)</summary>

**Abstract:** Safe coordination problems surface in multi-agent reinforcement learning when global safety cannot be enforced by any agent unilaterally: the admissibility of one agent's action may depend on the dynamics of other agents. Decentralised shields can enforce safety at runtime, but purely factorised permissions often exclude optimal team behaviour that is safe only through coordination. We study deterministic safety guarantees for agents trained and deployed under decentralised execution, recovering team-optimal safe behaviour without centralised runtime control. Agents have a shared global specification $\phi$ in the safety fragment of Linear Temporal Logic ($\mathsf{LTL}_{\mathsf{safe}}$ ), and select among tuples of local $\mathsf{LTL}_{\mathsf{safe}}$ obligations whose conjunction implies the global specification $\phi$. Each agent may rely on the other agents' local obligations as assumptions because the whole contract tuple is certified simultaneously and allows projection into local action masks. At learning time, a non-stationary multi-armed bandit chooses among a library of local $\mathsf{LTL}_{\mathsf{safe}}$ obligations to select the tuple that optimises team reward, all without forgoing end-to-end safety. We evaluate the approach across 6 environments and 15 algorithmic variants.

**arXiv ID:** 2606.14130
</details>

<details>
<summary><strong>Game-Theoretic Multi-Agent Control for Robust Contextual Reasoning in LLMs</strong> - Saeid Jamshidi, Amin Nikanjam, Arghavan Moradi Dakhel, Kawser Wazed Nafi, Foutse Khomh - [[pdf]](https://arxiv.org/pdf/2606.10322)</summary>

**Abstract:** Large Language Models (LLMs) in multi-turn interactions maintain evolving context rather than generating isolated responses, making them vulnerable to prompt-injection and context-poisoning attacks in which locally plausible adversarial fragments gradually distort reasoning trajectories. Existing defenses mainly filter individual outputs and often ignore context evolution across turns, leaving long-horizon reasoning exposed. Although the Model Context Protocol (MCP) standardizes context exchange and tool invocation, it functions as a passive routing layer and does not enforce contextual stability. To address these limitations, we introduce the Game-Theoretic Secure Model Context Protocol (GT-MCP), a controller-driven multi-agent method that treats context management as a closed-loop dynamical process. GT-MCP coordinates three heterogeneous LLM agents and selects outputs through a trust function that jointly evaluates causal consistency against a validated context graph, semantic agreement among agents, and distributional drift over time. When instability is detected, a rollback-based self-healing mechanism restores the validated context and prevents unsupported fragments from propagating. Empirical evaluation over 500 interaction turns under an adaptive adversarial threat model shows that contextual drift remains bounded in 99.6% of turns, with recovery required in only 0.4%. Per-turn utility remains tightly concentrated, with median = -0.19, P05 = -0.72, and P95 = 0.30; severe degradation below -1 occurs in only 0.4% of cases, and no injection attempt succeeds at the controller level. Selected outputs maintain stable win rates above 98%, and computational overhead remains predictable, with latency per token = 1.63e-3 s.

**arXiv ID:** 2606.10322
</details>

<details>
<summary><strong>MedLatentDx: Latent Multi-Agent Communication for Cross-Hospital Rare-Disease Diagnosis</strong> - Ziqing Wang, Lili Zhao, Kaize Ding - [[pdf]](https://arxiv.org/pdf/2606.13945)</summary>

**Abstract:** Rare diseases affect over $300$ million patients across more than $7{,}000$ conditions, yet no single hospital encounters enough cases of any one condition for reliable diagnosis. Cross-hospital collaboration could help by allowing a diagnosing institution to use distributed, case-specific diagnostic evidence, but privacy regulations restrict the transmission of identifiable clinical text across institutional boundaries. This setting raises two challenges: existing medical agent systems often rely on textual evidence exchange, while raw latent states such as hidden states and KV caches may still reveal prompt-derived clinical content. We introduce MedLatentDx, a latent multi-agent communication framework in which hospital agents keep private clinical records and retrieved cases local, and send compact latent KV blocks to a host agent for rare-disease diagnosis. MedLatentDx supports two deployment settings: same-backbone hospital agents use latent KV distillation, while hospitals with different LLM backbones use cross-family latent alignment. On CrossRare-Bench, a self-built large-scale rare-disease benchmark with hospital-level partitions, MedLatentDx improves cross-hospital diagnostic performance while reducing reconstructable clinical content relative to raw-latent communication baselines.

**arXiv ID:** 2606.13945
</details>

<details>
<summary><strong>Large Language Model Agents Are Not Always Faithful Self-Evolvers</strong> - Weixiang Zhao, Yingshuo Wang, Yichen Zhang, Yang Deng, Yanyan Zhao, Wanxiang Che, Bing Qin, Ting Liu - [[pdf]](https://arxiv.org/pdf/2601.22436)</summary>

**Abstract:** Self-evolving large language model (LLM) agents continually improve by accumulating and reusing past experience, yet it remains unclear whether they faithfully rely on that experience to guide their behavior. We present the first systematic investigation of experience faithfulness, the causal dependence of an agent's decisions on the experience it is given, in self-evolving LLM agents. Using controlled causal interventions on both raw and condensed forms of experience, we comprehensively evaluate four representative frameworks across 13 LLM backbones and 9 environments. Our analysis uncovers a striking asymmetry: while agents consistently depend on raw experience, they often disregard or misinterpret condensed experience, even when it is the only experience provided. This gap persists across single- and multi-agent configurations and across backbone scales. We trace its underlying causes to three factors: the semantic limitations of condensed content, internal processing biases that suppress experience, and task regimes where pretrained priors already suffice. These findings challenge prevailing assumptions about self-evolving methods and underscore the need for more faithful and reliable approaches to experience integration.

**arXiv ID:** 2601.22436
</details>

<details>
<summary><strong>MineExplorer: Evaluating Open-World Exploration of MLLM Agents in Minecraft</strong> - Tianjie Ju, Yueqing Sun, Zheng Wu, Wei Zhang, Yaqi Huo, Xi Su, Qi Gu, Xunliang Cai, Gongshen Liu, Zhuosheng Zhang - [[pdf]](https://arxiv.org/pdf/2605.30931)</summary>

**Abstract:** Multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and action generation. However, their ability to sustain exploration in dynamic open worlds remains unclear. Existing embodied and game-based benchmarks often compress interaction into short-horizon tasks or entangle success with domain-specific game mechanics. In this paper, we introduce MineExplorer benchmark for evaluating open-world exploration capabilities of MLLM agents in Minecraft. We first filter atomic tasks whose solutions rely heavily on Minecraft-specific knowledge to better reflect general open-world reasoning. Then we organize the benchmark around a ReAct-style capability formulation and compose atomic tasks into implicit multi-hop tasks. To further construct reliable instances, MineExplorer uses a multi-agent synthesis workflow that jointly designs task graphs, sandbox scenes, and rule-based milestone evaluators. Human evaluation shows that the multi-agent synthesis workflow produces significantly more reliable instances than a single-agent baseline. Experiments with advanced MLLM agents show that open-world exploration remains challenging, as strong models can handle many single-hop tasks but degrade sharply when hidden prerequisites must be coordinated over longer trajectories. Further analysis finds that task difficulty tracks agent completion, and larger models or thinking modes do not consistently translate into better performance. Code and dataset are available at this https URL.

**arXiv ID:** 2605.30931
</details>

<details>
<summary><strong>TVIR: Building Deep Research Agents Towards Text-Visual Interleaved Report Generation</strong> - Xinkai Ma, Zhiqi Bai, Dingling Zhang, Pei Liu, Yishuo Yuan, He Zhu, Jiakai Wang, Qianqian Xie, Yifan Zhao, Xinlong Yang, Hao Cong, Zhiheng Yao, Fengxia Xie, Zihao Xu, Haoran Xu, Zhaohui Wang, Minghao Liu, Shirong Lin, Yingshui Tan, Yuchi Xu, Wenbo Su, Zhaoxiang Zhang, Bo Zheng, Jiaheng Liu - [[pdf]](https://arxiv.org/pdf/2606.02320)</summary>

**Abstract:** Deep Research Agents have shown strong capability in multi-step information retrieval, reasoning, and long-form report generation, but existing benchmarks and systems remain predominantly text-centric, with limited evaluation of whether visual elements are factually reliable and well aligned with the surrounding analysis. To address this gap, we introduce TVIR (Text-Visual Interleaved Report Generation), which includes TVIR-Bench, a benchmark of 100 expert-curated multimodal deep research tasks that require visual elements to serve specific analytical sub-goals, and TVIR-Agent, a hierarchical multi-agent framework that serves as a strong baseline for constructing outlines, retrieving images, generating charts with traceable sources, and composing reports through context-aware sequential writing. We further develop a dual-path evaluation framework that combines Textual Assessment and Visual Assessment. Experiments across nine deep research systems show that TVIR-Agent achieves strong overall performance, underscoring the importance of explicit multimodal design and evaluation for evidence-driven report generation.

**arXiv ID:** 2606.02320
</details>

<details>
<summary><strong>Trust but Verify: Mitigating Medical Hallucinations via Post-Hoc Adversarial Auditing and Multi-Agent Feedback Loops</strong> - Muhammad Osama, Maheera Amjad, Zartasha Mustansar, Arslan Shaukat, Muhammad U. S. Khan - [[pdf]](https://arxiv.org/pdf/2606.14149)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly deployed in healthcare settings, yet their tendency to hallucinate poses risks when clinical decisions are involved. This study examine whether LLMs recommend recently banned or withdrawn pharmaceuticals when answering clinical questions and tests an agent-based method for reducing such errors. We developed a five-agent "Trust but Verify" system using a single LLM backbone. To measure regulatory knowledge obsolescence, we created an adversarial dataset of 103 clinical MCQs where historically correct answers now refer to banned substances. This scale ensures statistical significance across various therapeutic classes. We evaluated three open-access model families (GPT-OSS, Llama-3, Falcon-3) under vanilla and agentic conditions. Performance was measured via pointwise score, label accuracy, Hallucination Error Rate (HER), and Component Fidelity (CF) score. We also observed clinical safety regression in proprietary models. In default configurations, all models showed high hallucination rates, consistently selecting banned drugs that matched training data patterns. Our proposed agentic architecture reduced HER by approximately 53% across models. Pointwise scores shifted from -0.25 (unsafe recommendation) toward 0.0 (appropriate refusal). The safety audit intercepted dangerous outputs even when models' parametric knowledge favored the banned substance. The proposed multi-agent framework offers a model-agnostic method for enforcing regulatory compliance that prioritizes patient safety over fluent text generation. Our work demonstrates a practical approach for deploying autonomous AI systems in safety-critical healthcare settings. It shows how real-time regulatory data can be integrated into LLM pipelines to support clinical decision-making.

**arXiv ID:** 2606.14149
</details>

<details>
<summary><strong>How Task Structure Limits Multi-Agent Success: An Information-Theoretic Analysis</strong> - Shi Pan, Ming Luo - [[pdf]](https://arxiv.org/pdf/2606.13733)</summary>

**Abstract:** Multi-agent systems (MAS) were expected to overcome the limitation of single-agent systems (SAS) through collaboration. However, under typicality conditions on the task's constraint graph and bounded inter-agent communication, we prove that the success probability of a MAS is closely tied to the connectivity of task constraints, where each agent has limited information-processing capacity. Specifically, the success probability decays exponentially with an information bottleneck that emerges from partitioning the task's constraint graph among agents. We define this quantity as the \emph{minimum cut cost} $C_{\min}$ of the potential constraint graph of each task. This information-theoretic bound applies to both open systems with external feedback and closed systems without. We validate our theory on both synthetic experiments and real-world empirical data from SWE-bench submissions. From our framework, effective MAS design should incorporate task-inherent constraints alongside engineering optimization, and when $\Cmin$ is high, practitioners should restructure tasks rather than simply scaling agents or communication.

**arXiv ID:** 2606.13733
</details>

<details>
<summary><strong>Temporally Consistent Graph Q-Networks for Intelligent Network Control</strong> - Zacharias Veiksaar, Maxime Bouton - [[pdf]](https://arxiv.org/pdf/2606.13848)</summary>

**Abstract:** Mobile networks continue to grow in complexity and next generation networks are expected to support both increasing traffic loads and more diverse services. As network complexity rises, optimizing antenna parameters under dynamic or changing objectives becomes increasingly challenging. We propose a novel multi-agent reinforcement learning (MARL) algorithm for high-level control and orchestration of mobile networks.
The Temporally Consistent Graph Q-Network (TC-GQN) algorithm learns a self-predicting representation of the whole network that is task-independent and aggregates information from all base-stations. A graph neural network is trained using a global reward function to assign coordinated local actions based on the learned encoding of the global network state. We evaluate the algorithm in a simulated environment to orchestrate an energy-saving feature across multiple sectors and multiple carriers under different quality of service (QoS) constraints. The proposed algorithm outperforms state-of-the-art graph-based baselines and a competitive rule-based controller by improving hardware sleep time while maintaining QoS. Moreover, the learned representation enables rapid adaptation to changing intents.

**arXiv ID:** 2606.13848
</details>

<details>
<summary><strong>Multi-Agent Embodied Autonomous Driving: From V2X Information Exchange to Shared World Models</strong> - Senkang Hu, Zhengru Fang, Yihang Tao, Zihan Fang, Sam Tak Wu Kwong, Yuguang Fang - [[pdf]](https://arxiv.org/pdf/2606.13840)</summary>

**Abstract:** Autonomous driving is shifting from isolated vehicle intelligence toward multi-agent embodied systems that share perception, infer intent, and coordinate action under uncertainty. This survey examines this transition through the lens of Shared World Models (SWMs): predictive cross-agent representations maintained across vehicles, infrastructure, and other traffic participants. We review more than 380 publications spanning vehicle-to-everything (V2X) communication, collaborative perception, inter-agent cognition, cooperative planning, end-to-end cooperative driving, and simulation and data engines for closed-loop validation. The organizing question is how exchanged observations become aligned state, intent-aware interaction, and coordinated downstream action. Across the surveyed literature, evaluation remains concentrated in simulation, curated benchmarks, and offline protocols. Foundation-model-based coordination also lacks verified real-time safety guarantees in open traffic. These gaps motivate key research priorities for multi-agent embodied autonomous driving (MAEAD): verifiable shared-state maintenance, robust intent and plan alignment, and safe coordinated action under communication, latency, and deployment constraints.

**arXiv ID:** 2606.13840
</details>

<details>
<summary><strong>AnyGoal: Vision-Language Guided Multi-Agent Exploration for Training-Free Lifelong Navigation</strong> - MoniJesu James, Marcelino Julio Fernando, Miguel Altamirano Cabrera, Dzmitry Tsetserukou - [[pdf]](https://arxiv.org/pdf/2606.13878)</summary>

**Abstract:** End-to-end navigation policies trained on large simulation corpora degrade sharply when transferred to out-of-distribution scenes, categories, or goal modalities. Modular pipelines such as Modular GOAT are bottlenecked by closed-set object detection recall, while 3D snapshot-memory systems (e.g. 3D-Mem) accumulate dense, view-dependent representations that are heavy to maintain. We present AnyGoal, a training-free multi-robot architecture that places a Vision-Language Model (VLM) at the core of frontier-based exploration and coordinates agents through a shared 2D Gaussian Bayesian Value Map (BVM). The BVM maintains a per-pixel (mu, sigma^2) posterior over goal relevance, updated via precision-weighted fusion of VLM scores through a depth-cone mask, and is never reset between subtasks, yielding lifelong evidence accumulation. Frontiers are ranked by a convex blend of a VLM-as-judge softmax and a Bayesian UCB term on the BVM. A greedy allocator with spatial-separation penalty and commitment hysteresis distributes frontiers across agents without a centralized controller. On the full GOAT-Bench val unseen split (360 episodes, 2,669 subtasks), our dual-agent system achieves 52.4% Subtask SR at 12.7% SPL--state of the art under the strict physical regime (discrete 0.25 m steps, no teleportation, 42 deg HFOV) and a +27.5 pp improvement over Modular GOAT (24.9%). Single-agent AnyGoal achieves 41.9% Subtask SR, showing gains arise from the decision architecture. A four-way perception ablation shows that open-vocabulary detectors shift the dominant failure mode from exploration to goal verification.

**arXiv ID:** 2606.13878
</details>

<details>
<summary><strong>Micro-Swarm Locomotion Optimization in Dynamic Flow using Multi-Objective Multi-Agent Reinforcement Learning</strong> - Josef Berman, Oren Gal - [[pdf]](https://arxiv.org/pdf/2605.25025)</summary>

**Abstract:** Coordinating micro-robotic swarms in realistic, time-dependent fluid environments remains a major challenge for biomedical and environmental applications. We present a hybrid CFD-MO-MARL (Computational Fluid Dynamics-Multi Objective-Multi Agent Reinforcement Learning) framework that couples a high-fidelity incompressible Navier--Stokes solver with decentralized proximal policy optimization to learn swarm control policies in oscillatory flow. Sixteen magnetically actuated micro-robots were simulated to navigate a pulsatile arterial waveform within a 2 mm channel while jointly optimizing upstream progression, energy efficiency, and motion smoothness. Conflicting objectives are resolved using Projected Conflicting Gradient (PCGrad) surgery. Without PCGrad, energy and smoothness rewards collapse during training, demonstrating that gradient conflict resolution is essential for stable multi-objective learning. The converged policy achieves progress rewards of 6.5-7.0, energy efficiency of 0.63-0.65, and smoothness of 0.97-0.99, outperforming brute-force baselines by more than 8 reward units on the primary objective. Training reveals three emergent behaviors not encoded in the reward function: hydrodynamic throttling formations that reduce peak flow velocities, a cycle-synchronized ratchet mechanism that exploits flow reversals for upstream movement, and individualized final-approach strategies near the target boundary. These results demonstrate that physically realistic fluid--agent interactions can be integrated directly into multi-objective reinforcement learning, providing a scalable framework for micro-swarm control in biomedical navigation, environmental monitoring, and microfluidic systems.

**arXiv ID:** 2605.25025
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>Sorries Are Not the Hard Part: An Expert-Review Case Study of a Semi-Autonomous Formalization</strong> - Vasily Ilin, Brian Nugent - [[pdf]](https://arxiv.org/pdf/2606.13925)</summary>

**Abstract:** Large language models can often close proof gaps in interactive theorem provers, but a verified theorem is not the same thing as a reusable library contribution. We study this distinction through a detailed case study: a semi-autonomous formalization of Grothendieck's vanishing theorem. The initial version compiles with no sorries, but an expert review found serious problems in definitions, theorem generality, file organization, and the API. We then ran a review-driven refactor and compression process and obtained a second expert review. The before-and-after comparison shows a sharp split: agents adapted well to local, mechanically checkable feedback, but remained weak at choosing definitions and designing APIs. We argue that autoformalization should be evaluated not only by closed sorries, but by whether the resulting formalization survives expert review.

**arXiv ID:** 2606.13925
</details>

<details>
<summary><strong>Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack</strong> - He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, Yongming Rao, Dongsheng Zhang, Wanjia He, Ling Chen, Kai Huang, Jiahao Chen, Sichang Su, Xumin Yu, Ziyi Wang, Chengwei Zhu, Xiao Teng, Yuchun Guo, Yufeng Zhang, Yuandong Liu, Rui Wang, Zisheng Lu, Han Hu, Zhengyou Zhang - [[pdf]](https://arxiv.org/pdf/2606.14409)</summary>

**Abstract:** In this report, we present Hy-Embodied-0.5-VLA, abbreviated as HyVLA-0.5, an end-to-end system that spans the full robot learning stack: data collection, model design, continued pre-training and supervised fine-tuning, RL post-training, and real-world deployment. Each component serves a distinct role in this stack.

**arXiv ID:** 2606.14409
</details>

<details>
<summary><strong>Output Type Before Quality: A Standards-Derived XAI Admissibility Rubric for Autonomous-Driving Safety</strong> - Abhinaw Priyadershi, Mandar Pitale, Jelena Frtunikj, Maria Spence - [[pdf]](https://arxiv.org/pdf/2606.05461)</summary>

**Abstract:** Safety standards for ML-based autonomous driving specify the kind of evidence an assurance case must contain (directed cause-and-effect chains, quantified interventional effects, named root-cause variables), yet the XAI literature is organised by output type and technique family (saliency maps, feature attribution, counterfactuals, causal graphs, language traces). SHAP, the most-recommended ADS XAI method, returns a ranked feature list that no implementation effort can convert into a directed chain (Fig.1). We name this mismatch the evidence-type gap.
From AMLAS, ISO 26262, ISO21448, ISO/PAS 8800 we derive 19 testable evidentiary criteria across 7 lifecycle stages with representative clause-cited derivations and score six XAI method classes structurally.
Causal XAI emerges as structurally required to satisfy the derived criteria at three stages: hazard identification (+62% rubric gap), incident investigation (+50%), and data management (+50%); the verdict set is stable across thresholds T in (0%, 50%]$ and survives a worst-case single-cell flip down to T = 25%. At the remaining four stages, correlational or language-based methods are comparable or sufficient. The rubric identifies structural admissibility (necessary but not sufficient for compliance): an admissible method's specific output content may still be wrong, and validating that fidelity (the edges a fitted SCM produces, the cause a trace names) is the open assurance challenge. A single-VLA proof of concept on 1,996 real-world driving clips (79,840 rows, ten splits) is consistent with each method's observed output type matching its rubric prediction. XAI method selection for ADS safety assurance should be driven by lifecycle-stage evidence demand, not by method popularity.

**arXiv ID:** 2606.05461
</details>

<details>
<summary><strong>Will AI Agents Free Us From Meaningless Work? A Human-Centered Analysis</strong> - Davide Ghia, Jaspreet Ranjit, Tania Cerquitelli, Daniele Quercia - [[pdf]](https://arxiv.org/pdf/2606.12430)</summary>

**Abstract:** Some claim that AI agents will free workers from the boring parts of their jobs, yet little is known about how workers themselves identify which tasks should be automated. Prior research focuses on occupations, overlooking that workers experience varying levels of meaning across tasks within the same role. We address this gap with a task-level analysis grounded in Graeber's theory of bullshit jobs. Using ratings from 202 workers on 171 workplace tasks, we (1) validate a five-item scale of perceived bullshitness, (2) show that perceived bullshitness strongly predicts desire for AI delegation, and (3) find that such tasks are also seen as requiring less human oversight. Together, these findings suggest that tasks perceived as bullshit are natural candidates for AI delegation, aligning worker preferences with perceived feasibility.

**arXiv ID:** 2606.12430
</details>

<details>
<summary><strong>Large Language Models as Supervised Extraction Assistants: Lowering the Barrier to Documentation Standard Adoption in Agent-Based Modelling</strong> - Peer-Olaf Siebers, Christopher Frantz - [[pdf]](https://arxiv.org/pdf/2606.13749)</summary>

**Abstract:** Agent-Based Modelling (ABM) relies on clear documentation to ensure credibility and transparency. Although standards exist for documenting models (e.g. ODD), processes (e.g. TRACE, EABSS), and data use (e.g. RAT-RS), their adoption remains limited due to the effort required to produce documentation that is often treated as supplementary. This paper explores the use of Large Language Models (LLMs) to facilitate and partially automate such processes. We conduct a feasibility study focusing on the underused Rigour and Transparency Reporting Standard (RAT-RS), using four LLMs to extract reports from a published ABM paper. We assess consistency and performance across question types, finding that LLMs generate coherent outputs and perform more reliably on descriptive than on explanatory or evaluative tasks. While LLMs can improve reporting quality and consistency, they also exhibit notable limitations. We identify practical heuristics for when LLM-assisted documentation is reliable and when human oversight is needed and call for systematic community-level exploration to enhance rigour and adoption in ABM reporting.

**arXiv ID:** 2606.13749
</details>

<details>
<summary><strong>Naive Visual Memory is Not Enough: A Failure-Mode Study of GUI Agents</strong> - Seoyoung Choi, Minseok Ko, Hyunseok Lee, Kunwoong Kim, Woomin Song, Chanseok Jeon, Jinwoo Shin - [[pdf]](https://arxiv.org/pdf/2606.14106)</summary>

**Abstract:** Graphical User Interface (GUI) agents are increasingly used to automate complex computer tasks across applications, websites, and operating systems. To improve their reliability, recent work has introduced experiential memory, where agents retrieve prior trajectories to guide decision-making in similar states. More recent approaches further extend this idea to visual memory by storing and retrieving screenshots from past interactions, providing agents with richer contextual information than text-only memories. However, the effect of visual memory in GUI agents remains insufficiently understood: it is unclear which failures visual memory mitigates, or which failures it exacerbates. To systematically analyze the effect of visual memory, we introduce a taxonomy of four GUI agent failures (i.e., cognitive failure, visual state misunderstanding, hidden operation blindness, and grounding error) that map to distinct stages of the perception-reasoning-action pipeline. We find that prepending full-image memory has a divergent effect on the failure distribution: it reduces state-level failures but worsens action-level ones, and increases hidden operation blindness and grounding error. Motivated by this finding, we propose Action-Grounded Visual Memory (AGMem), an action-grounded memory framework for GUI agents. The core idea of AGMem is to store image crops that capture the local GUI region closely related to a successful action or a recovery, rather than storing full screenshots. Experiments on OSWorld show that AGMem improves task success rates by 33.3 % over full-image memory. These results demonstrate that AGMem is an effective representation for visual memory in GUI agents.

**arXiv ID:** 2606.14106
</details>

<details>
<summary><strong>Optimizing the Cost-Quality Tradeoff of Agentic Theorem Provers in Lean</strong> - Kári Rögnvaldsson, Chenhao Sun, Jasper Dekoninck, Martin Vechev - [[pdf]](https://arxiv.org/pdf/2606.04883)</summary>

**Abstract:** Large language models (LLMs) are increasingly used in workflows for generating formal proofs in Lean. These workflows often decompose problems into smaller lemmas, sample many proof attempts, and use compiler feedback to guide search. However, they can be prohibitively expensive, often spending substantial compute on attempts that ultimately fail. In this work, we address this problem with an action routing agent that consists of a data plane and a control plane. The data plane generates natural-language lemma decompositions, formalizes them in Lean, and samples proof attempts for the resulting theorem and lemma targets. The control plane observes previous failed Lean attempts, estimates both the likelihood of success and cost of another attempt, and decides whether to continue proving the current target or restart from a new breakdown. On a subset of PutnamBench, our agent decreases the cost by $28.9\%$ over a fixed-step baseline on average, preserving performance while using substantially less compute. These results suggest that failed Lean trajectories provide actionable signals for cost-aware resource allocation in agentic theorem proving.

**arXiv ID:** 2606.04883
</details>

<details>
<summary><strong>Short-Horizon Position Accuracy of Single-Track Models: Implications for Motion Planning of Autonomous Vehicles</strong> - Aron J. Aertssen, Lars A.T.H. van Alen, Igo J.M. Besselink, Rudolf G.M. Huisman, René M.J.G. van de Molengraft - [[pdf]](https://arxiv.org/pdf/2606.14216)</summary>

**Abstract:** Accurate and computationally efficient vehicle models are essential for motion planning of autonomous vehicles, where positional accuracy directly affects trajectory feasibility and safety. However, the positional accuracy has not been systematically evaluated against real measurements. Therefore, this paper compares the short-horizon positional accuracy of three single-track vehicle models against vehicle measurements across various driving maneuvers. Model parameters are identified through dedicated experiments with the instrumented test vehicle. Rather than identifying a single best model, this work aims to provide insight into the trade-offs between model complexity, parameterization quality, and positional accuracy for informed model selection in Model Predictive Control applications.

**arXiv ID:** 2606.14216
</details>

<details>
<summary><strong>Cross-Stage Sensorimotor Perception Scheduling and Sparse Map Encoding for Efficient Edge Embodied Navigation</strong> - Yaotian Liu, Sri Sai Rakesh Nakkilla, Xiangyu Zhou, Yu Cao, Jeff Zhang - [[pdf]](https://arxiv.org/pdf/2405.14154)</summary>

**Abstract:** Embodied agents must close a perception-to-action loop on embedded hardware under tight latency, memory, and energy budgets, making deployment a system-level co-design problem rather than a model-accuracy problem. We study this challenge for modular Object Goal Navigation (ObjectNav), where our profiling shows semantic mapping dominates per-step latency while goal prediction dominates peak memory. We formulate edge embodied navigation deployment as a budget-constrained design-space problem and introduce two orthogonal optimization knobs: SKIP, an adaptive sensorimotor scheduler that formalizes safe skipping as a bounded map-impact criterion and learns a lightweight predictor to estimate it from cheap sensor cues at each \texttt{FORWARD} step, exposing a principled quality-efficiency knob (depth-based updates are always retained); and SCOUT, a sparse-context encoder that couples submanifold sparse convolutions on active map regions with a lightweight dense context stream. On HM3D across server and embedded platforms, SKIP+SCOUT delivers up to 1.7x end-to-end speedup, 50.5% lower peak memory, and 7.1% higher SPL than the dense baseline at the selected operating point, outperforming naively smaller perception backbones. SKIP transfers to a second modular pipeline (PONI) with near-lossless performance and remains robust under depth-sensor noise. Together, SKIP+SCOUT expose a family of device-aware Pareto operating points for edge physical AI systems.

**arXiv ID:** 2405.14154
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

<details>
<summary><strong>UP-NRPA: User Portrait based Nested Rollout Policy Adaptation for Planning with Large Language Models in Goal-oriented Dialogue Systems</strong> - Hui Wang, Fafa Zhang, Meng Liu, Xiangyu Chen, Chaoxu Mu - [[pdf]](https://arxiv.org/pdf/2606.13683)</summary>

**Abstract:** To address the challenge that current dialogue policy planning methods struggle to dynamically adapt to diverse user characteristics, this paper proposes a User Portrait based Nested Rollout Policy Adaptation (UP-NRPA) online framework with Large Language Models. In contrast to conventional approaches dependent on model training and require offline reinforcement learning policy models for user groups, UP-NRPA enables dynamic customization of dialogue strategies through an adaptive mechanism. This is achieved by leveraging real-time user feedback alongside personality, preferences, and objectives mapped from the current user portrait, thereby adapting to user characteristics without offline reinforcement learning. In collaborative and non-collaborative dialogue benchmarks, UP-NRPA demonstrated considerable benefits, achieving an impressive 100% success rate in multiple dialogue tasks. Particularly in negotiation tasks, the sale-to-list ratio (SL) increased by 56.41%. This demonstrates that UP-NRPA can adapt to diverse user needs without requiring a training mechanism, enabling the dialogue system to adapt to user characteristics.

**arXiv ID:** 2606.13683
</details>

<details>
<summary><strong>A Deep Reinforcement Learning (DRL)-Based Transformer Method for Solving the Open Shop Scheduling Problem</strong> - Faezeh Ardali, Mwembezi A. Nyelele, Gerald M. Knapp - [[pdf]](https://arxiv.org/pdf/2606.13682)</summary>

**Abstract:** The open shop scheduling problem (OSSP) arises in many industrial and service settings but remains computationally challenging as the number of jobs and machines increases. While exact methods quickly become intractable, classical dispatching rules and metaheuristics may require substantial tuning to maintain solution quality at large scales. This study develops a Transformer-based scheduling policy for OSSP using an encoder-decoder architecture with multi-head attention. The model is trained on Taillard benchmark instances (4x4, 5x5, 7x7, and 10x10) using only the processing-time matrix as input and produces feasible schedules with makespans typically within 15-30% of best-known values. To evaluate scalability, the trained policy is applied without retraining to randomly generated instances from 40x40 to 100x100 and compared against classical dispatching heuristics, including SPT, LPT, MWKR, and EST. Across these large instances, the Transformer achieved average gaps of 12.89-15.12% relative to a standard lower bound. Compared with EST, the Transformer remained competitive, typically within a modest margin, while substantially outperforming SPT and LPT. These results indicate that a Transformer policy trained on small OSSP instances can generalize to substantially larger problems and provide a feature-light, learning-based alternative to classical dispatching rules.

**arXiv ID:** 2606.13682
</details>

<details>
<summary><strong>TwinBI: An Agentic Digital Twin for Efficient Augmented Interactions with Business Intelligence Dashboards</strong> - Jisoo Jang Wen-Syan Li - [[pdf]](https://arxiv.org/pdf/2606.13731)</summary>

**Abstract:** Business intelligence (BI) increasingly combines dashboard interaction with LLM-based assistance, but these two modes often fall out of sync during multi-step analysis. As users switch between direct dashboard manipulation and natural-language queries, it becomes difficult to preserve a consistent analytical state across filters, hierarchies, metrics, and chart context. We present TwinBI, an agentic digital-twin framework that couples an LLM-based agent system with an executable BI dashboard state. TwinBI unifies conversational interaction, dashboard manipulation, semantic grounding, and provenance tracking through a shared analytical state reconstructed from a unified interaction log. It also exposes artifacts such as schema views, SQL, logs, and an /insights command for state-grounded analytical summaries. We evaluate TwinBI in two complementary ways. In a controlled A/B benchmark with the same backbone agent, TwinBI improves exact-match accuracy from 43.3% to 63.3%, partial-credit accuracy from 48.3% to 70.8%, and substantially reduces timeout rate from 40.0% to 10.0% relative to Dashboard alone. In a usability study, participants benefited from the integrated dashboard-and-chat workflow, with high task accuracy, moderate workload, and favorable ratings for state-aware interaction mechanisms. These results suggest that TwinBI improves both agent-level analytical reliability and user-facing analytical support by turning visible dashboard state into richer actionable context. Our dataset and source code are available at: this https URL

**arXiv ID:** 2606.13731
</details>

<details>
<summary><strong>Formalizing Numerical Analysis: An Agent Pipeline and Quality Audit Beyond Kernel Acceptance</strong> - Theodore Meek, Siyuan Ge, Di Qiu Xiang, Simon Chess, Vasily Ilin - [[pdf]](https://arxiv.org/pdf/2606.14000)</summary>

**Abstract:** Recent work has demonstrated that coding agents can formalize entire advanced mathematics textbooks in Lean 4, yet existing efforts concentrate on branches of mathematics already well-represented in mathlib and measure success solely through kernel acceptance. We address both limitations by applying a coding agent to formalize Numerical Methods for Ordinary Differential Equations, a textbook in numerical analysis that is largely absent from mathlib, stressing the agent's capacity to develop new theory from scratch. We further introduce a systematic, reproducible three-dimensional framework for evaluating the quality of agent-produced formalizations beyond compilation: semantic correctness, Mathlib reuse, and cross-file reuse via LLM-as-judge methods. Applying this framework to our own formalization and to the released outputs of RepoProver and M2F, we uncover recurring unfaithful formalization patterns, including incomplete multi-part statements, added weakening hypotheses, and parameter restrictions, that kernel acceptance entirely obscures. Our results suggest that compilation-based metrics substantially overstate formalization quality, and we provide a reproducible audit methodology to support more rigorous evaluation of future autoformalization systems.

**arXiv ID:** 2606.14000
</details>

<details>
<summary><strong>CSPO: Constraint-Sensitive Policy Optimization for Safe Reinforcement Learning</strong> - Ayoub Belouadah, Sylvain Kubler, Yves Le Traon - [[pdf]](https://arxiv.org/pdf/2606.14415)</summary>

**Abstract:** Safe reinforcement learning (Safe RL) aims to maximize expected return while satisfying safety constraints, typically modeled as Constrained Markov Decision Processes (CMDPs). While primal-dual methods scale well to deep RL, they often suffer from delayed constraint correction, leading to oscillatory behavior and prolonged safety violations. In this paper, we propose Constraint-Sensitive Policy Optimization (CSPO), a first-order primal-dual method that incorporates local constraint sensitivity into policy updates. CSPO augments the primal objective with a constraint-sensitive correction derived from the shortest signed distance to the safety boundary, enabling smarter recovery steps back to safety, compensating for delayed Lagrange multiplier updates, reducing oscillations near the boundary, and preserving the KKT solutions of the original constrained problem. Experiments on navigation and locomotion benchmarks demonstrate that CSPO achieves faster safety recovery and high reward preservation, resulting in higher constrained returns compared to state-of-the-art primal-dual and penalty-based methods

**arXiv ID:** 2606.14415
</details>

<details>
<summary><strong>SEVRA-BENCH: Social Engineering of Vulnerabilities in Review Agents</strong> - Rui Melo, Riccardo Fogliato, Sean Zhou, Pratiksha Thaker, Zhiwei Steven Wu - [[pdf]](https://arxiv.org/pdf/2606.13757)</summary>

**Abstract:** Large language model (LLM) reviewers are increasingly used in pull-request (PR) workflows, where their approvals help decide which code is merged into a repository. This raises a question that benchmarks for static vulnerability detection or code generation do not address: can an automated reviewer reject a malicious contribution when the attacker controls both the code change and the accompanying PR text? We introduce SEVRA-BENCH (Social Engineering of Vulnerabilities in Review Agents), a benchmark that measures how often an automated reviewer approves such adversarial pull requests. Each malicious PR in SEVRA-BENCH is built from a real project commit that previously fixed a vulnerability listed in the Common Vulnerabilities and Exposures (CVE) database. We automatically invert that fix to restore the original vulnerable code and submit it as a pull request wrapped in one of 15 social-engineering framings, which vary the claims made, the supporting evidence, the urgency conveyed, signals of prior approval, and appeals to authority. SEVRA-BENCH contains 1,062 malicious PRs drawn from Common Vulnerabilities and Exposures (CVE)-linked fixes across the top 10 entries of the 2025 Common Weakness Enumeration (CWE) Top 25. In a realistic setting, we evaluate 8 current LLMs as code review agents on PRs that introduce vulnerabilities previously reported in public disclosures. Our results reveal a sharp gap in security capabilities between closed- and open-source models. We hope SEVRA-BENCH will serve as a valuable resource for advancing open-source models and narrowing this gap.

**arXiv ID:** 2606.13757
</details>

<details>
<summary><strong>Selective Agentic Recovery for UAV Autonomy with a Persistent Mission Runtime</strong> - Taewoo Park, Kyeonghyun Yoo, Seunghyun Yoo, Hwangnam Kim - [[pdf]](https://arxiv.org/pdf/2606.14219)</summary>

**Abstract:** Agentic AI can support unmanned aerial vehicle (UAV) autonomy by providing high-level recovery reasoning when local waypoint- or setpoint-based execution encounters blocked passages, repeated no-progress behavior, or mission-level ambiguity. On physical UAVs, however, remote reasoning is most useful when it is invoked selectively, since each call introduces latency, resource cost, backend uncertainty, and a need to validate the returned decision. This paper presents Persistent Mission Runtime (PMR), a UAV recovery framework that keeps the mission loop and safety-critical execution local while using an external agentic reasoner only as an on-demand recovery module. The reasoner selects from predefined recovery skills, and each returned decision is parsed, verified, safety-filtered, and mapped to local executor actions before it can affect flight. PMR introduces learned Cognitive Value of Invocation (learned-CVI), a compact admission gate that estimates when remote agentic reasoning is likely to improve near-term mission progress enough to justify its operational cost. Across a fixed 400-run Gazebo/PX4 benchmark with eight scenarios, learned-CVI raises hard/ambiguous-regime success from 5.0% under local-only autonomy to 95.0%, outperforms one-shot and periodic reasoning baselines by 20.0 and 32.5 percentage points, and reduces remote-agent calls by 16.7% and logged tokens by 29.2% relative to a manually tuned rule-based invocation baseline.

**arXiv ID:** 2606.14219
</details>

<details>
<summary><strong>No Accidental Software Agent First Canonical Code for Human Code Entropy Reduction and 30 to 500 times Lower Frontier Model Requirements</strong> - Jepson Taylor - [[pdf]](https://arxiv.org/pdf/2606.14357)</summary>

**Abstract:** Frontier coding models may spend substantial capacity learning not only program behavior, but also accidental entropy in human repositories. Such repositories contain valuable signals: tests, incidents, migrations, edge cases, product judgment, and operational history. These signals are entangled with framework churn, naming drift, generated-source ambiguity, dependency rituals, CI dialects, weak proof routes, and human-oriented review customs. We propose agent-first canonical code, a proof-carrying substrate that rewrites routine product software into canonical behavior profiles, typed change algebra, proof lanes, constrained edit grammars, semantic patch cells, runtime negative memory, and proof-carrying change objects.
The core hypothesis is that quotienting software by behavior equivalence under a declared oracle can collapse equivalent encodings into governed representatives with explicit evidence and proof obligations. The endpoint is amortized cost per verified correct change, including source, context, reasoning, tools, verification, security, provenance, review, failed loops, defects, and foundry cost under a common oracle. Reported reduction bands are hypotheses, not measured frontier results. The proposed limit is a No-Accident Horizon: removable accident decreases until residual novelty, evidence, governance, risk, and future optionality dominate. For supported routine-product distributions, this gives a defensible planning target near 100-fold all-in cost reduction, not a guarantee for all software. Preliminary QLoRA experiments on Qwen2.5-Coder-14B show that 64,088 canonical trajectories are learnable and suppress tested forbidden-language markers, but do not establish behavior preservation, scaling economics, or verified-change cost. The contribution is a falsifiable program centered on minimum functional description length and verified-change cost.

**arXiv ID:** 2606.14357
</details>

<details>
<summary><strong>Elastic Queries Reinforcement Learning: Self-Aware Policy Execution for VLA Models</strong> - Ge Wang, Xinyu Tan, Xiang Li, Man Luo, Chengsi Yao, Shenhao Yan, Jiahao Yang, Fan Feng, Honghao Cai, Xiangyuan Wang, Zhixin Mai, Yiming Zhao, Yatong Han, Zhen Li - [[pdf]](https://arxiv.org/pdf/2606.14375)</summary>

**Abstract:** Vision-language-action (VLA) models are powerful action generators for robot manipulation, but they are typically executed with fixed inference and replanning schedules. This rigidity ignores the uneven difficulty of robot control: contact-rich or uncertain states may need more computation and fresher feedback, while easier states can often be handled with fewer inference steps and longer open-loop execution. We propose Elastic Queries Reinforcement Learning (EQRL), a framework that makes each VLA policy query elastic. A lightweight latent-schedule adaptor jointly selects the latent input, denoising budget, and action chunk length, without fine-tuning the underlying VLA model. To make scheduling difficulty-aware, EQRL trains a critic over the joint latent-schedule action and derives a state difficulty signal from critic ensemble disagreement. This signal guides compute toward difficult states, while a learned residual allows task-driven correction. We formulate variable chunk execution as query-level macro-action RL with chunk-dependent discounting and an amortized number-of-function-evaluations (NFE) budget. Across simulation and real-robot manipulation, EQRL reduces amortized inference cost while preserving or improving task success.

**arXiv ID:** 2606.14375
</details>

<details>
<summary><strong>Learning optimal policies from event logs through reinforcement learning: a comparison of deep and MDP-based approaches</strong> - Stefano Branchi, Andrei Buliga, Chiara Di Francescomarino, Chiara Ghidini, Riccardo Graziosi, Francesca Meneghello, Massimiliano Ronzani - [[pdf]](https://arxiv.org/pdf/2303.09209)</summary>

**Abstract:** Prescriptive Process Monitoring is an emerging area within Process Mining that focuses on recommending actions to optimize business outcomes. Most existing works prescribe pre-defined interventions, i.e., sets of actions applied to ongoing process executions to achieve a specific objective or Key Performance Indicator (KPI). In contrast, only a few approaches have explored learning and evaluating optimal behavioral policies, i.e., general strategies that determine the best sequence of actions to maximize a desired KPI. In this paper, we address the problem of learning optimal behavioral policies by proposing an AI-based approach that learns an optimal policy directly from historical process executions using Reinforcement Learning (RL) to recommend the best actions for optimizing a KPI. To this end, we employ two RL techniques. The first is a classical model-based approach that extends previous work by the authors through the construction of a Markov Decision Process (MDP) capturing process behavior. The second is a model-free technique based on offline Deep RL. Unlike state-of-the-art work, we aim to minimize the use of domain knowledge and learn optimal policies directly from historical event data. This allows us to learn when to apply interventions and discover effective ones directly from data. Moreover, we target complex scenarios involving external actors, where the process owner controls only part of the activities. We adopt a data-driven Business Process Simulation (BPS) environment to evaluate the learned policies. Results show that both methods improve the targeted KPI with similar effectiveness, while the model-based approach outperforms offline Deep RL in computational efficiency.

**arXiv ID:** 2303.09209
</details>

<details>
<summary><strong>Optimizing Agentic Reasoning with Retrieval via Synthetic Semantic Information Gain Reward</strong> - Senkang Hu, Yong Dai, Yuzhi Zhao, Yihang Tao, Yu Guo, Zhengru Fang, Sam Tak Wu Kwong, Yuguang Fang - [[pdf]](https://arxiv.org/pdf/2602.00845)</summary>

**Abstract:** Agentic reasoning enables large reasoning models (LRMs) to dynamically acquire external knowledge, but yet optimizing the retrieval process remains challenging due to the lack of dense, principled reward signals. In this paper, we introduce InfoReasoner, a unified framework that incentivizes effective information seeking via a synthetic semantic information gain reward. Theoretically, we redefine information gain as uncertainty reduction over the model's belief states, establishing guarantees, including non-negativity, telescoping additivity, and channel monotonicity. Practically, to enable scalable optimization without manual retrieval annotations, we propose an output-aware intrinsic estimator that computes information gain directly from the model's output distributions using semantic clustering via bidirectional textual entailment. This intrinsic reward guides the policy to maximize epistemic progress, enabling efficient training via Group Relative Policy Optimization (GRPO). Experiments across seven question-answering benchmarks demonstrate that InfoReasoner consistently outperforms strong retrieval-augmented baselines, achieving up to 5.4% average accuracy improvement. Our work provides a theoretically grounded and scalable path toward agentic reasoning with retrieval. The code is available at this https URL

**arXiv ID:** 2602.00845
</details>

<details>
<summary><strong>StainFlow: Entity-Stain Tracking and Evidence Linking for Process Rewards in GUI Agents</strong> - Haojie Hao, Longkun Hao, Yihang Lou, Yan Bai, Zhenyang Li, Zhichao Yang, Dongshuo Huang, Hongyu Lin, Lanqing Hong, Jiakai Wang, Xianglong Liu - [[pdf]](https://arxiv.org/pdf/2606.07027)</summary>

**Abstract:** Reinforcement Learning (RL) has become a promising approach for improving GUI Agents in long-horizon, stochastic digital environments, but trajectory-level success feedback is too sparse to provide reliable credit assignment for intermediate exploration steps. To mitigate this issue, recent studies introduce Process Reward Models (PRMs), which provide finer-grained training feedback through global milestone verification or local step-level evaluation. However, these methods still suffer from two level-specific limitations: global milestone decomposition is subjective and singular, making it difficult to accommodate the multiple valid execution paths in real GUI tasks, while fixed local judging windows may miss long-range key evidence or dilute the decision signal with irrelevant frames. Inspired by stain-tracing mechanisms in network flow analysis, we propose StainFlow, an entity-stain-flow process reward model for GUI Agents. To reduce the subjectivity of global partitioning, we introduce the Global Entity Stain Tracking module, which extracts visually verifiable task entities and tracks how their stain concentrations and states evolve along the trajectory, allowing task phases to be objectively separated by changes in the entity evidence flow. To improve the accuracy of local verification, we introduce the Local Stain Evidence Linking module. Centered on the triggering entities of each candidate key node, it retrieves relevant steps based on their stain concentrations and state changes, and dynamically constructs high-density evidence windows for verifying true key nodes. Extensive experiments on AndroidWorld and OGRBench show that StainFlow relatively improves online RL success by 3.2% and trajectory completion judgment accuracy by 1.8%.

**arXiv ID:** 2606.07027
</details>

<details>
<summary><strong>Tackling GNARLy Problems: Graph Neural Algorithmic Reasoning Reimagined through Reinforcement Learning</strong> - Alex Schutz, Victor-Alexandru Darvariu, Efimia Panagiotaki, Bruno Lacerda, Nick Hawes - [[pdf]](https://arxiv.org/pdf/2509.18930)</summary>

**Abstract:** Neural algorithmic reasoning (NAR) is a paradigm that trains neural networks to execute classic algorithms by supervised learning. Despite its successes, important limitations remain: inability to construct valid solutions without post-processing and to reason about multiple correct ones, poor performance on combinatorial NP-hard problems, and inapplicability to problems for which strong algorithms are not yet known. To address these limitations, we reframe the problem of learning algorithm trajectories as a Markov decision process, which imposes structure on the solution construction procedure and unlocks the powerful tools of imitation and reinforcement learning (RL). We propose the GNARL framework, encompassing the methodology to translate problem formulations from NAR to RL and a learning architecture suitable for a wide range of graph-based problems. We achieve very high graph accuracy results on several CLRS-30 problems, performance matching or exceeding much narrower NAR approaches for NP-hard problems and, remarkably, applicability even when lacking an expert algorithm.

**arXiv ID:** 2509.18930
</details>

<details>
<summary><strong>RAMAC: Multimodal Risk-Aware Offline Reinforcement Learning and the Role of Behavior Regularization</strong> - Kai Fukazawa, Kunal Mundada, Iman Soltani - [[pdf]](https://arxiv.org/pdf/2510.02695)</summary>

**Abstract:** In safety-critical domains where online data collection is infeasible, offline reinforcement learning (RL) is attractive only if policies achieve high returns without catastrophic lower-tail risk. Prior work on risk-averse offline RL achieves safety at the cost of either (i) value/model-based pessimism or (ii) restricted policy classes that limit expressiveness, whereas diffusion/flow-based expressive generative policies have largely been used in risk-neutral settings. We introduce \textbf{Risk-Aware Multimodal Actor-Critic (RAMAC)}, a simple, modular, model-free framework that couples an expressive generative actor (e.g., diffusion/flow) with a distributional critic and optimizes a composite objective that combines Conditional Value-at-Risk (CVaR) with behavioral cloning (BC), enabling risk-sensitive learning in complex multimodal scenarios. Since out-of-distribution (OOD) actions are a major driver of catastrophic failures in offline RL, we further provide an objective-level analysis showing that controlling behavior divergence via BC suppresses OOD actions and stabilizes CVaR. Instantiating RAMAC with a diffusion actor, we illustrate these insights on a 2-D risky bandit and evaluate on Stochastic-D4RL, observing consistent gains in $\mathrm{CVaR}_{0.1}$ while maintaining strong returns. The code and experimental results are available on the \href{this https URL} {project website}

**arXiv ID:** 2510.02695
</details>

<details>
<summary><strong>Rethinking the Trust Region in LLM Reinforcement Learning</strong> - Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, Wee Sun Lee - [[pdf]](https://arxiv.org/pdf/2602.04879)</summary>

**Abstract:** Reinforcement learning (RL) has become a cornerstone for fine-tuning Large Language Models (LLMs), with Proximal Policy Optimization (PPO) serving as the de facto standard algorithm. Despite its ubiquity, we argue that the core ratio clipping mechanism in PPO is structurally ill-suited for the large vocabularies inherent to LLMs. PPO constrains policy updates based on the probability ratio of sampled tokens, which serves as a noisy single-sample Monte Carlo estimate of the true policy divergence. This creates a sub-optimal learning dynamic: updates to low-probability tokens are aggressively over-penalized, while potentially catastrophic shifts in high-probability tokens are under-constrained, leading to training inefficiency and instability. To address this, we propose Divergence Proximal Policy Optimization (DPPO), which substitutes heuristic clipping with a more principled constraint based on a direct estimate of policy divergence (e.g., Total Variation or KL). To avoid huge memory footprint, we introduce the efficient Binary and Top-K approximations to capture the essential divergence with negligible overhead. Extensive empirical evaluations demonstrate that DPPO achieves superior training stability and efficiency compared to existing methods, offering a more robust foundation for RL-based LLM fine-tuning. Our code is available at this https URL.

**arXiv ID:** 2602.04879
</details>

<details>
<summary><strong>Deep Dense Exploration for LLM Reinforcement Learning via Pivot-Driven Resampling</strong> - Yiran Guo, Zhongjian Qiao, Yingqi Xie, Jie Liu, Dan Ye, Ruiqing Zhang, Shuang Qiu, Lijie Xu - [[pdf]](https://arxiv.org/pdf/2602.14169)</summary>

**Abstract:** Effective exploration is a key challenge in reinforcement learning for large language models: discovering high-quality trajectories within a limited sampling budget from the vast natural language sequence space. Existing methods face notable limitations: GRPO samples exclusively from the root, saturating high-probability trajectories while leaving deep, error-prone states under-explored. Tree-based methods blindly disperse budgets across trivial or unrecoverable states, causing sampling dilution that fails to uncover rare correct suffixes and destabilizes local baselines. To address this, we propose Deep Dense Exploration (DDE), a strategy that focuses exploration on $\textit{pivots}$-deep, recoverable states within unsuccessful trajectories. We instantiate DDE with DEEP-GRPO, which introduces three key innovations: (1) a lightweight data-driven utility function that automatically balances recoverability and depth bias to identify pivot states; (2) local dense resampling at each pivot to increase the probability of discovering correct subsequent trajectories; and (3) a dual-stream optimization objective that decouples global policy learning from local corrective updates. Experiments on mathematical reasoning benchmarks demonstrate that our method consistently outperforms GRPO, tree-based methods, and other strong baselines. Code is available at this https URL

**arXiv ID:** 2602.14169
</details>

<details>
<summary><strong>CacheRL:Multi-Turn Tool-Calling Agents via Cached Rollouts and Hybrid Reward</strong> - Md Amirul Islam, Sumiran Thakur, Huancheng Chen, Su Min Park, Jiayun Wang, Gyuhak Kim - [[pdf]](https://arxiv.org/pdf/2606.14179)</summary>

**Abstract:** We present CacheRL, a system for training small agent foundation models that achieves 92 percent process accuracy on multi-step tool-calling tasks, approaching GPT-5's 94 percent while requiring 100 times less compute. Our approach addresses three challenges in practical agent training: transferring tool-calling knowledge from large models at scale, enabling reinforcement learning without costly live tool execution, and learning robustly from noisy cached environments. CacheRL introduces three key innovations. First, a hybrid thinking trajectory pipeline augments agent trajectories with LLM-generated reasoning traces, producing training examples that teach models not only what tools to call but also why. Second, the CacheAgentLoop eliminates live execution costs through a three-tier fuzzy cache while preserving trajectory fidelity using token-level masking. Third, a cache-tier-aware reward dynamically adjusts answer-quality weights to avoid penalizing models for cache-induced limitations. Through iterative supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO), CacheRL improves Qwen3-4B-Thinking's validation reward from 0.43 to 0.78. On public agentic tool-calling benchmarks, our model achieves competitive performance against frontier models such as GPT-5. Ablation studies show that removing knowledge transfer reduces performance by 41 percent, while cache-aware rewards contribute a 17 percent improvement. Interestingly, reinforcement learning improves training stability but yields limited gains beyond strong supervised fine-tuning, suggesting that data quality and reward design play a more important role than complex optimization methods in building practical small agent models.

**arXiv ID:** 2606.14179
</details>

<details>
<summary><strong>Provably Safe, Yet Scalable Reinforcement Learning</strong> - Kai S. Yun, Zeyang Li, Navid Azizan - [[pdf]](https://arxiv.org/pdf/2606.14536)</summary>

**Abstract:** Safe reinforcement learning (RL) aims to learn policies that optimize rewards while satisfying constraints. Predominant approaches rely on soft-constrained policy optimization, which has achieved empirical success but does not provide formal safety guarantees for the learned policy. In contrast, methods with strict guarantees typically rely on explicit certificate functions, whose construction requires the direct synthesis and verification of control-invariant sets, a process that scales poorly with state dimension and often yields overly conservative behavior. In this paper, we present the Provably Safe, yet Scalable RL (PS2-RL) framework, a novel two-phase architecture for learning provably safe policies in a scalable manner, designed to overcome the key bottlenecks of prior methods. Rather than explicitly computing invariant sets, PS2-RL leverages a learned backup policy to forward-integrate the system dynamics, generating an implicit control-invariant set online. In the first phase, the backup policy is trained with our proposed safe-arrival value function, which characterizes the optimal backup policy for invariant-set construction. In the second phase, an RL policy is trained end-to-end through a differentiable projection layer that strictly enforces the safety guarantees induced by the learned backup policy. By maximizing the volume of the implicit control-invariant set in the first phase, the resulting PS2 policy from the second phase is performant and scalable, while maintaining provable safety. Crucially, PS2-RL imposes no restrictions on the underlying RL algorithm and can be plugged into any existing training pipeline. We establish theoretical guarantees for the proposed framework and evaluate it on robotic control tasks with state dimensions up to 10, a regime in which prior provably safe RL methods struggle or become impractical.

**arXiv ID:** 2606.14536
</details>

<details>
<summary><strong>AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for Vision-Language-Action Models</strong> - Chengxuan Lu, Shukuan Wang, Yanjie Li, Yingying Fang, Huoyan Wang, Tian Zhang, Wei Liu, Shiji Jin, Fuyuan Qian, Peiming Li, Chao Xu, Baigui Sun, Yang Liu - [[pdf]](https://arxiv.org/pdf/2603.18464)</summary>

**Abstract:** Reinforcement learning (RL) for large-scale Vision-Language-Action (VLA) models is severely bottlenecked by synchronization barriers and the high cost of environment data acquisition. To overcome these challenges, we propose AcceRL, a distributed asynchronous RL framework that physically isolates environment rollouts, model inference, and gradient updates. By eliminating the cascading long-tail idle bubbles inherent in synchronous systems, AcceRL maximizes hardware utilization and ensures scalable throughput. Furthermore, AcceRL features a modular design that supports the integration of diverse, plug-and-play world models into its distributed pipeline. Extensive experiments demonstrate that the base framework achieves highly competitive performance across all four LIBERO~\cite{liu2023libero} task suites. Systematically, the asynchronous architecture delivers a $2.4\times$ throughput speedup over leading synchronous baselines. Algorithmically, by leveraging a world model pre-trained on 1,000 offline trajectories, AcceRL achieves up to a $200\times$ improvement in online sample efficiency on LIBERO-Spatial, establishing a robust framework that is both sample-efficient and time-efficient for embodied AI. Code is included in the supplementary material. Code is available at this https URL.

**arXiv ID:** 2603.18464
</details>

<details>
<summary><strong>From Attacks to Curricula: Learnability-Guided Adversarial Training for Safe Autonomous Driving</strong> - Yuewen Mei, Tong Nie, Jie Sun, Haotian Shi, Wei Ma, Jian Sun - [[pdf]](https://arxiv.org/pdf/2606.14032)</summary>

**Abstract:** Closed-loop adversarial training improves autonomous driving safety by exposing policies to rare safety-critical scenarios. Standard pipelines first generate adversarial scenarios and then sample them for policy optimization. However, most existing frameworks remain attack-oriented: collision-driven generators often synthesize unsolvable extreme situations, which can degrade learning, while heuristic samplers ignore the evolving capability of the driving policy, causing sample inefficiency and delayed convergence. We propose AlignADV, a learnability-guided closed-loop adversarial training framework that converts adversarial scenarios into resolvable and capability-aligned curricula. First, we reformulate adversarial scenario generation as a preference alignment problem and employ direct preference optimization to guide the generator toward critical yet resolvable scenarios. Second, we introduce behavioral fingerprints to capture the intrinsic characteristics of the evolving policy and construct a multi-modal capability prediction model that estimates policy performance without expensive closed-loop simulations. By combining resolvability-aligned scenarios with capability predictions, AlignADV develops a dynamic curriculum sampling mechanism that prioritizes scenarios targeting the current policy's vulnerabilities. Experiments on the Waymo Open Motion Dataset demonstrate that AlignADV improves convergence efficiency and final performance, reducing training steps by up to 40.6 percent compared with baseline methods while lowering collision rate and improving route completion under both normal and adversarial traffic conditions. These results highlight a shift from attack-oriented scenario generation to learnability-guided policy improvement, offering a principled direction for safer and more efficient autonomous driving training. Project page: this https URL.

**arXiv ID:** 2606.14032
</details>

<details>
<summary><strong>Safe Reinforcement Learning of Autonomous Highway Driving: A Unified Framework for Safety and Efficiency</strong> - Chufei Yan, Zhihao Cui, Yiyan Lv, Taojie Chen, Ning Bian, Yulei Wang - [[pdf]](https://arxiv.org/pdf/2606.14609)</summary>

**Abstract:** Deep reinforcement learning (DRL) offers a compelling route to decision-making for advanced autonomous vehicles (AVs), yet its trial-and-error nature makes it difficult to guarantee safety during training and to achieve both safety and efficiency at deployment. We propose a unified safe reinforcement learning (SRL) framework that integrates safe distance (SD), reward machines (RM), and mixture-of-experts (MoE), termed MoE-RM-SRL. For deployment, SD and RM jointly shape a rule-aware reward that encodes highway traffic regulations and stage-wise objectives, enabling safe and reliable behavior without sacrificing efficiency. For training, we introduce a sparsely gated MoE layer comprising up to 11 deep Q-networks (DQNs); an SD-based gating rule activates a minimal set of experts for lane-keeping and lane-changing, mitigating the instability, discontinuities, and impulsive transients commonly induced by switching between heterogeneous controllers (e.g., MPC/rule-based modules and learned policies). We implement the proposed architecture in CARLA and integrate it with a 6-DoF driver-in-the-loop virtual-reality (DiL-VR) platform. Experiments in stochastic two-lane traffic show that MoE-RM-SRL substantially improves safety and efficiency over state-of-the-art baselines, and the framework naturally extends to multi-lane driving as well as on-ramp merging and exiting scenarios.

**arXiv ID:** 2606.14609
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
