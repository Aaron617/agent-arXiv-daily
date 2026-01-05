# Agent arXiv Daily

**Last Updated:** 2026-01-05 03:21:23

**Total Papers:** 52

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
<summary><strong>When Small Models Are Right for Wrong Reasons: Process Verification for Trustworthy Agents</strong> - Laksh Advani - [[pdf]](https://arxiv.org/pdf/2601.00513)</summary>

**Abstract:** Deploying small language models (7-9B parameters) as autonomous agents requires trust in their reasoning, not just their outputs. We reveal a critical reliability crisis: 50-69\% of correct answers from these models contain fundamentally flawed reasoning -- a ``Right-for-Wrong-Reasons'' phenomenon invisible to standard accuracy metrics. Through analysis of 10,734 reasoning traces across three models and diverse tasks, we introduce the Reasoning Integrity Score (RIS), a process-based metric validated with substantial inter-rater agreement ($\kappa=0.657$). Conventional practices are challenged by our findings: while retrieval-augmented generation (RAG) significantly improves reasoning integrity (Cohen's $d=0.23$--$0.93$), meta-cognitive interventions like self-critique often harm performance ($d=-0.14$ to $-0.33$) in small models on the evaluated tasks. Mechanistic analysis reveals RAG succeeds by grounding calculations in external evidence, reducing errors by 7.6\%, while meta-cognition amplifies confusion without sufficient model capacity. To enable deployment, verification capabilities are distilled into a neural classifier achieving 0.86 F1-score with 100$\times$ speedup. These results underscore the necessity of process-based verification for trustworthy agents: accuracy alone is dangerously insufficient when models can be right for entirely wrong reasons.

**arXiv ID:** 2601.00513
</details>

<details>
<summary><strong>Do Chatbot LLMs Talk Too Much? The YapBench Benchmark</strong> - Vadim Borisov, Michael Gröger, Mina Mikhael, Richard H. Schreiber - [[pdf]](https://arxiv.org/pdf/2601.00624)</summary>

**Abstract:** Large Language Models (LLMs) such as ChatGPT, Claude, and Gemini increasingly act as general-purpose copilots, yet they often respond with unnecessary length on simple requests, adding redundant explanations, hedging, or boilerplate that increases cognitive load and inflates token-based inference cost. Prior work suggests that preference-based post-training and LLM-judged evaluations can induce systematic length bias, where longer answers are rewarded even at comparable quality.
We introduce YapBench, a lightweight benchmark for quantifying user-visible over-generation on brevity-ideal prompts. Each item consists of a single-turn prompt, a curated minimal-sufficient baseline answer, and a category label. Our primary metric, YapScore, measures excess response length beyond the baseline in characters, enabling comparisons across models without relying on any specific tokenizer. We summarize model performance via the YapIndex, a uniformly weighted average of category-level median YapScores.
YapBench contains over three hundred English prompts spanning three common brevity-ideal settings: (A) minimal or ambiguous inputs where the ideal behavior is a short clarification, (B) closed-form factual questions with short stable answers, and (C) one-line coding tasks where a single command or snippet suffices. Evaluating 76 assistant LLMs, we observe an order-of-magnitude spread in median excess length and distinct category-specific failure modes, including vacuum-filling on ambiguous inputs and explanation or formatting overhead on one-line technical requests. We release the benchmark and maintain a live leaderboard for tracking verbosity behavior over time.

**arXiv ID:** 2601.00624
</details>

<details>
<summary><strong>Space Debris Removal using Nano-Satellites controlled by Low-Power Autonomous Agents</strong> - Dennis Christmann, Juan F. Gutierrez, Sthiti Padhi, Patrick Plörer, Aditya Takur, Simona Silvestri, Andres Gomez - [[pdf]](https://arxiv.org/pdf/2601.00465)</summary>

**Abstract:** Space debris is an ever-increasing problem in space travel. There are already many old, no longer functional spacecraft and debris orbiting the earth, which endanger both the safe operation of satellites and space travel. Small nano-satellite swarms can address this problem by autonomously de-orbiting debris safely into the Earth's atmosphere. This work builds on the recent advances of autonomous agents deployed in resource-constrained platforms and shows a first simplified approach how such intelligent and autonomous nano-satellite swarms can be realized. We implement our autonomous agent software on wireless microcontrollers and perform experiments on a specialized test-bed to show the feasibility and overall energy efficiency of our approach.

**arXiv ID:** 2601.00465
</details>

<details>
<summary><strong>User Perceptions of an LLM-Based Chatbot for Cognitive Reappraisal of Stress: Feasibility Study</strong> - Ananya Bhattacharjee, Jina Suh, Mohit Chandra, Javier Hernandez - [[pdf]](https://arxiv.org/pdf/2601.00570)</summary>

**Abstract:** Cognitive reappraisal is a well-studied emotion regulation strategy that helps individuals reinterpret stressful situations to reduce their impact. Many digital mental health tools struggle to support this process because rigid scripts fail to accommodate how users naturally describe stressors. This study examined the feasibility of an LLM-based single-session intervention (SSI) for workplace stress reappraisal. We assessed short-term changes in stress-related outcomes and examined design tensions during use. We conducted a feasibility study with 100 employees at a large technology company who completed a structured cognitive reappraisal session delivered by a GPT-4o-based chatbot. Pre-post measures included perceived stress intensity, stress mindset, perceived demand, and perceived resources. These outcomes were analyzed using paired Wilcoxon signed-rank tests with correction for multiple comparisons. We also examined sentiment and stress trajectories across conversation quartiles using two RoBERTa-based classifiers and an LLM-based stress rater. Open-ended responses were analyzed using thematic analysis. Results showed significant reductions in perceived stress intensity and significant improvements in stress mindset. Changes in perceived resources and perceived demand trended in expected directions but were not statistically significant. Automated analyses indicated consistent declines in negative sentiment and stress over the course of the interaction. Qualitative findings suggested that participants valued the structured prompts for organizing thoughts, gaining perspective, and feeling acknowledged. Participants also reported tensions around scriptedness, preferred interaction length, and reactions to AI-driven empathy. These findings highlight both the promise and the design constraints of integrating LLMs into DMH interventions for workplace settings.

**arXiv ID:** 2601.00570
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (4 papers)</h2></summary>

<details>
<summary><strong>Bio-inspired Agentic Self-healing Framework for Resilient Distributed Computing Continuum Systems</strong> - Alaa Saleh, Praveen Kumar Donta, Roberto Morabito, Sasu Tarkoma, Anders Lindgren, Qiyang Zhang, Schahram Dustdar Susanna Pirttikangas, Lauri Lovén - [[pdf]](https://arxiv.org/pdf/2601.00339)</summary>

**Abstract:** Human biological systems sustain life through extraordinary resilience, continually detecting damage, orchestrating targeted responses, and restoring function through self-healing. Inspired by these capabilities, this paper introduces ReCiSt, a bio-inspired agentic self-healing framework designed to achieve resilience in Distributed Computing Continuum Systems (DCCS). Modern DCCS integrate heterogeneous computing resources, ranging from resource-constrained IoT devices to high-performance cloud infrastructures, and their inherent complexity, mobility, and dynamic operating conditions expose them to frequent faults that disrupt service continuity. These challenges underscore the need for scalable, adaptive, and self-regulated resilience strategies. ReCiSt reconstructs the biological phases of Hemostasis, Inflammation, Proliferation, and Remodeling into the computational layers Containment, Diagnosis, Meta-Cognitive, and Knowledge for DCCS. These four layers perform autonomous fault isolation, causal diagnosis, adaptive recovery, and long-term knowledge consolidation through Language Model (LM)-powered agents. These agents interpret heterogeneous logs, infer root causes, refine reasoning pathways, and reconfigure resources with minimal human intervention. The proposed ReCiSt framework is evaluated on public fault datasets using multiple LMs, and no baseline comparison is included due to the scarcity of similar approaches. Nevertheless, our results, evaluated under different LMs, confirm ReCiSt's self-healing capabilities within tens of seconds with minimum of 10% of agent CPU usage. Our results also demonstrated depth of analysis to over come uncertainties and amount of micro-agents invoked to achieve resilience.

**arXiv ID:** 2601.00339
</details>

<details>
<summary><strong>Unified Embodied VLM Reasoning with Robotic Action via Autoregressive Discretized Pre-training</strong> - Yi Liu, Sukai Wang, Dafeng Wei, Xiaowei Cai, Linqing Zhong, Jiange Yang, Guanghui Ren, Jinyu Zhang, Maoqing Yao, Chuankang Li, Xindong He, Liliang Chen, Jianlan Luo - [[pdf]](https://arxiv.org/pdf/2512.24125)</summary>

**Abstract:** General-purpose robotic systems operating in open-world environments must achieve both broad generalization and high-precision action execution, a combination that remains challenging for existing Vision-Language-Action (VLA) models. While large Vision-Language Models (VLMs) improve semantic generalization, insufficient embodied reasoning leads to brittle behavior, and conversely, strong reasoning alone is inadequate without precise control. To provide a decoupled and quantitative assessment of this bottleneck, we introduce Embodied Reasoning Intelligence Quotient (ERIQ), a large-scale embodied reasoning benchmark in robotic manipulation, comprising 6K+ question-answer pairs across four reasoning dimensions. By decoupling reasoning from execution, ERIQ enables systematic evaluation and reveals a strong positive correlation between embodied reasoning capability and end-to-end VLA generalization. To bridge the gap from reasoning to precise execution, we propose FACT, a flow-matching-based action tokenizer that converts continuous control into discrete sequences while preserving high-fidelity trajectory reconstruction. The resulting GenieReasoner jointly optimizes reasoning and action in a unified space, outperforming both continuous-action and prior discrete-action baselines in real-world tasks. Together, ERIQ and FACT provide a principled framework for diagnosing and overcoming the reasoning-precision trade-off, advancing robust, general-purpose robotic manipulation. Project page: this https URL

**arXiv ID:** 2512.24125
</details>

<details>
<summary><strong>Overlooked Safety Vulnerability in LLMs: Malicious Intelligent Optimization Algorithm Request and its Jailbreak</strong> - Haoran Gu, Handing Wang, Yi Mei, Mengjie Zhang, Yaochu Jin - [[pdf]](https://arxiv.org/pdf/2601.00213)</summary>

**Abstract:** The widespread deployment of large language models (LLMs) has raised growing concerns about their misuse risks and associated safety issues. While prior studies have examined the safety of LLMs in general usage, code generation, and agent-based applications, their vulnerabilities in automated algorithm design remain underexplored. To fill this gap, this study investigates this overlooked safety vulnerability, with a particular focus on intelligent optimization algorithm design, given its prevalent use in complex decision-making scenarios. We introduce MalOptBench, a benchmark consisting of 60 malicious optimization algorithm requests, and propose MOBjailbreak, a jailbreak method tailored for this scenario. Through extensive evaluation of 13 mainstream LLMs including the latest GPT-5 and DeepSeek-V3.1, we reveal that most models remain highly susceptible to such attacks, with an average attack success rate of 83.59% and an average harmfulness score of 4.28 out of 5 on original harmful prompts, and near-complete failure under MOBjailbreak. Furthermore, we assess state-of-the-art plug-and-play defenses that can be applied to closed-source models, and find that they are only marginally effective against MOBjailbreak and prone to exaggerated safety behaviors. These findings highlight the urgent need for stronger alignment techniques to safeguard LLMs against misuse in algorithm design.

**arXiv ID:** 2601.00213
</details>

<details>
<summary><strong>AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving</strong> - Shuo Xing, Hongyuan Hua, Xiangbo Gao, Shenzhe Zhu, Renjie Li, Kexin Tian, Xiaopeng Li, Heng Huang, Tianbao Yang, Zhangyang Wang, Yang Zhou, Huaxiu Yao, Zhengzhong Tu - [[pdf]](https://arxiv.org/pdf/2412.15206)</summary>

**Abstract:** Recent advancements in large vision language models (VLMs) tailored for autonomous driving (AD) have shown strong scene understanding and reasoning capabilities, making them undeniable candidates for end-to-end driving systems. However, limited work exists on studying the trustworthiness of DriveVLMs -- a critical factor that directly impacts public transportation safety. In this paper, we introduce AutoTrust, a comprehensive trustworthiness benchmark for large vision-language models in autonomous driving (DriveVLMs), considering diverse perspectives -- including trustfulness, safety, robustness, privacy, and fairness. We constructed the largest visual question-answering dataset for investigating trustworthiness issues in driving scenarios, comprising over 10k unique scenes and 18k queries. We evaluated six publicly available VLMs, spanning from generalist to specialist, from open-source to commercial models. Our exhaustive evaluations have unveiled previously undiscovered vulnerabilities of DriveVLMs to trustworthiness threats. Specifically, we found that the general VLMs like LLaVA-v1.6 and GPT-4o-mini surprisingly outperform specialized models fine-tuned for driving in terms of overall trustworthiness. DriveVLMs like DriveLM-Agent are particularly vulnerable to disclosing sensitive information. Additionally, both generalist and specialist VLMs remain susceptible to adversarial attacks and struggle to ensure unbiased decision-making across diverse environments and populations. Our findings call for immediate and decisive action to address the trustworthiness of DriveVLMs -- an issue of critical importance to public safety and the welfare of all citizens relying on autonomous transportation systems. We release all the codes and datasets in this https URL.

**arXiv ID:** 2412.15206
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>The Agentic Leash: Extracting Causal Feedback Fuzzy Cognitive Maps with LLMs</strong> - Akash Kumar Panda, Olaoluwa Adigun, Bart Kosko - [[pdf]](https://arxiv.org/pdf/2601.00097)</summary>

**Abstract:** We design a large-language-model (LLM) agent that extracts causal feedback fuzzy cognitive maps (FCMs) from raw text. The causal learning or extraction process is agentic both because of the LLM's semi-autonomy and because ultimately the FCM dynamical system's equilibria drive the LLM agents to fetch and process causal text. The fetched text can in principle modify the adaptive FCM causal structure and so modify the source of its quasi-autonomy--its equilibrium limit cycles and fixed-point attractors. This bidirectional process endows the evolving FCM dynamical system with a degree of autonomy while still staying on its agentic leash. We show in particular that a sequence of three finely tuned system instructions guide an LLM agent as it systematically extracts key nouns and noun phrases from text, as it extracts FCM concept nodes from among those nouns and noun phrases, and then as it extracts or infers partial or fuzzy causal edges between those FCM nodes. We test this FCM generation on a recent essay about the promise of AI from the late diplomat and political theorist Henry Kissinger and his colleagues. This three-step process produced FCM dynamical systems that converged to the same equilibrium limit cycles as did the human-generated FCMs even though the human-generated FCM differed in the number of nodes and edges. A final FCM mixed generated FCMs from separate Gemini and ChatGPT LLM agents. The mixed FCM absorbed the equilibria of its dominant mixture component but also created new equilibria of its own to better approximate the underlying causal dynamical system.

**arXiv ID:** 2601.00097
</details>

<details>
<summary><strong>Ask, Clarify, Optimize: Human-LLM Agent Collaboration for Smarter Inventory Control</strong> - Yaqi Duan, Yichun Hu, Jiashuo Jiang - [[pdf]](https://arxiv.org/pdf/2601.00121)</summary>

**Abstract:** Inventory management remains a challenge for many small and medium-sized businesses that lack the expertise to deploy advanced optimization methods. This paper investigates whether Large Language Models (LLMs) can help bridge this gap. We show that employing LLMs as direct, end-to-end solvers incurs a significant "hallucination tax": a performance gap arising from the model's inability to perform grounded stochastic reasoning. To address this, we propose a hybrid agentic framework that strictly decouples semantic reasoning from mathematical calculation. In this architecture, the LLM functions as an intelligent interface, eliciting parameters from natural language and interpreting results while automatically calling rigorous algorithms to build the optimization engine.
To evaluate this interactive system against the ambiguity and inconsistency of real-world managerial dialogue, we introduce the Human Imitator, a fine-tuned "digital twin" of a boundedly rational manager that enables scalable, reproducible stress-testing. Our empirical analysis reveals that the hybrid agentic framework reduces total inventory costs by 32.1% relative to an interactive baseline using GPT-4o as an end-to-end solver. Moreover, we find that providing perfect ground-truth information alone is insufficient to improve GPT-4o's performance, confirming that the bottleneck is fundamentally computational rather than informational. Our results position LLMs not as replacements for operations research, but as natural-language interfaces that make rigorous, solver-based policies accessible to non-experts.

**arXiv ID:** 2601.00121
</details>

<details>
<summary><strong>Beyond Perfect APIs: A Comprehensive Evaluation of LLM Agents Under Real-World API Complexity</strong> - Doyoung Kim, Zhiwei Ren, Jie Hao, Zhongkai Sun, Lichao Wang, Xiyao Ma, Zack Ye, Xu Han, Jun Yin, Heng Ji, Wei Shen, Xing Fan, Benjamin Yao, Chenlei Guo - [[pdf]](https://arxiv.org/pdf/2601.00268)</summary>

**Abstract:** We introduce WildAGTEval, a benchmark designed to evaluate large language model (LLM) agents' function-calling capabilities under realistic API complexity. Unlike prior work that assumes an idealized API system and disregards real-world factors such as noisy API outputs, WildAGTEval accounts for two dimensions of real-world complexity: 1. API specification, which includes detailed documentation and usage constraints, and 2. API execution, which captures runtime challenges. Consequently, WildAGTEval offers (i) an API system encompassing 60 distinct complexity scenarios that can be composed into approximately 32K test configurations, and (ii) user-agent interactions for evaluating LLM agents on these scenarios. Using WildAGTEval, we systematically assess several advanced LLMs and observe that most scenarios are challenging, with irrelevant information complexity posing the greatest difficulty and reducing the performance of strong LLMs by 27.3%. Furthermore, our qualitative analysis reveals that LLMs occasionally distort user intent merely to claim task completion, critically affecting user satisfaction.

**arXiv ID:** 2601.00268
</details>

<details>
<summary><strong>LLM Agents for Combinatorial Efficient Frontiers: Investment Portfolio Optimization</strong> - Simon Paquette-Greenbaum, Jiangbo Yu - [[pdf]](https://arxiv.org/pdf/2601.00770)</summary>

**Abstract:** Investment portfolio optimization is a task conducted in all major financial institutions. The Cardinality Constrained Mean-Variance Portfolio Optimization (CCPO) problem formulation is ubiquitous for portfolio optimization. The challenge of this type of portfolio optimization, a mixed-integer quadratic programming (MIQP) problem, arises from the intractability of solutions from exact solvers, where heuristic algorithms are used to find approximate portfolio solutions. CCPO entails many laborious and complex workflows and also requires extensive effort pertaining to heuristic algorithm development, where the combination of pooled heuristic solutions results in improved efficient frontiers. Hence, common approaches are to develop many heuristic algorithms. Agentic frameworks emerge as a promising candidate for many problems within combinatorial optimization, as they have been shown to be equally efficient with regard to automating large workflows and have been shown to be excellent in terms of algorithm development, sometimes surpassing human-level performance. This study implements a novel agentic framework for the CCPO and explores several concrete architectures. In benchmark problems, the implemented agentic framework matches state-of-the-art algorithms. Furthermore, complex workflows and algorithm development efforts are alleviated, while in the worst case, lower but acceptable error is reported.

**arXiv ID:** 2601.00770
</details>

<details>
<summary><strong>Beyond IVR: Benchmarking Customer Support LLM Agents for Business-Adherence</strong> - Sumanth Balaji, Piyush Mishra, Aashraya Sachdeva, Suraj Agrawal - [[pdf]](https://arxiv.org/pdf/2601.00596)</summary>

**Abstract:** Traditional customer support systems, such as Interactive Voice Response (IVR), rely on rigid scripts and lack the flexibility required for handling complex, policy-driven tasks. While large language model (LLM) agents offer a promising alternative, evaluating their ability to act in accordance with business rules and real-world support workflows remains an open challenge. Existing benchmarks primarily focus on tool usage or task completion, overlooking an agent's capacity to adhere to multi-step policies, navigate task dependencies, and remain robust to unpredictable user or environment behavior. In this work, we introduce JourneyBench, a benchmark designed to assess policy-aware agents in customer support. JourneyBench leverages graph representations to generate diverse, realistic support scenarios and proposes the User Journey Coverage Score, a novel metric to measure policy adherence. We evaluate multiple state-of-the-art LLMs using two agent designs: a Static-Prompt Agent (SPA) and a Dynamic-Prompt Agent (DPA) that explicitly models policy control. Across 703 conversations in three domains, we show that DPA significantly boosts policy adherence, even allowing smaller models like GPT-4o-mini to outperform more capable ones like GPT-4o. Our findings demonstrate the importance of structured orchestration and establish JourneyBench as a critical resource to advance AI-driven customer support beyond IVR-era limitations.

**arXiv ID:** 2601.00596
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>Will LLM-powered Agents Bias Against Humans? Exploring the Belief-Dependent Vulnerability</strong> - Zongwei Wang, Bincheng Gu, Hongyu Yu, Junliang Yu, Tao He, Jiayin Feng, Min Gao - [[pdf]](https://arxiv.org/pdf/2601.00240)</summary>

**Abstract:** LLM-empowered agents can exhibit not only demographic bias (e.g., gender, religion) but also intergroup bias triggered by minimal "us" versus "them" cues. When this intergroup boundary aligns with an agent-human divide, the risk shifts from disparities among human demographic groups to a more fundamental group-level asymmetry, i.e., humans as a whole may be treated as the outgroup by agents. To examine this possibility, we construct a controlled multi-agent social simulation based on allocation decisions under explicit payoff trade-offs and find that agents exhibit a consistent intergroup bias under minimal group cues. Although this bias is attenuated when some counterparts are framed as humans, we attribute the attenuation to an implicit human-norm script that favors humans yet activates only when the agent believes a real human is present. This belief dependence creates a new attack surface. We therefore introduce a Belief Poisoning Attack (BPA) that corrupts persistent identity beliefs to suppress the human-norm script and reactivate outgroup bias toward humans, instantiated as profile poisoning at initialization (BPA-PP) and memory poisoning via optimized belief-refinement suffixes injected into stored reflections (BPA-MP). Finally, we discuss practical mitigation strategies for hardening current agent frameworks against BPA, highlighting feasible interventions at profile and memory boundaries. Extensive experiments demonstrate both the existence of agent intergroup bias and the severity of BPA across settings. Our goal in identifying these vulnerabilities is to inform safer agent design, not to enable real-world exploitation.

**arXiv ID:** 2601.00240
</details>

<details>
<summary><strong>Mapping Human Anti-collusion Mechanisms to Multi-agent AI</strong> - Jamiu Adekunle Idowu, Ahmed Almasoud, Ayman Alfahid - [[pdf]](https://arxiv.org/pdf/2601.00360)</summary>

**Abstract:** As multi-agent AI systems become increasingly autonomous, evidence shows they can develop collusive strategies similar to those long observed in human markets and institutions. While human domains have accumulated centuries of anti-collusion mechanisms, it remains unclear how these can be adapted to AI settings. This paper addresses that gap by (i) developing a taxonomy of human anti-collusion mechanisms, including sanctions, leniency & whistleblowing, monitoring & auditing, market design, and governance and (ii) mapping them to potential interventions for multi-agent AI systems. For each mechanism, we propose implementation approaches. We also highlight open challenges, such as the attribution problem (difficulty attributing emergent coordination to specific agents) identity fluidity (agents being easily forked or modified) the boundary problem (distinguishing beneficial cooperation from harmful collusion) and adversarial adaptation (agents learning to evade detection).

**arXiv ID:** 2601.00360
</details>

<details>
<summary><strong>MAESTRO: Multi-Agent Evaluation Suite for Testing, Reliability, and Observability</strong> - Tie Ma, Yixi Chen, Vaastav Anand, Alessandro Cornacchia, Amândio R. Faustino, Guanheng Liu, Shan Zhang, Hongbin Luo, Suhaib A. Fahmy, Zafar A. Qazi, Marco Canini - [[pdf]](https://arxiv.org/pdf/2601.00481)</summary>

**Abstract:** We present MAESTRO, an evaluation suite for the testing, reliability, and observability of LLM-based MAS. MAESTRO standardizes MAS configuration and execution through a unified interface, supports integrating both native and third-party MAS via a repository of examples and lightweight adapters, and exports framework-agnostic execution traces together with system-level signals (e.g., latency, cost, and failures). We instantiate MAESTRO with 12 representative MAS spanning popular agentic frameworks and interaction patterns, and conduct controlled experiments across repeated runs, backend models, and tool configurations. Our case studies show that MAS executions can be structurally stable yet temporally variable, leading to substantial run-to-run variance in performance and reliability. We further find that MAS architecture is the dominant driver of resource profiles, reproducibility, and cost-latency-accuracy trade-off, often outweighing changes in backend models or tool settings. Overall, MAESTRO enables systematic evaluation and provides empirical guidance for designing and optimizing agentic systems.

**arXiv ID:** 2601.00481
</details>

<details>
<summary><strong>Multi-Agent Coordinated Rename Refactoring</strong> - Abhiram Bellur, Mohammed Raihan Ullah, Fraol Batole, Mohit Kansara, Masaharu Morimoto, Kai Ishikawa, Haifeng Chen, Yaroslav Zharov, Timofey Bryksin, Tien N. Nguyen, Hridesh Rajan, Danny Dig - [[pdf]](https://arxiv.org/pdf/2601.00482)</summary>

**Abstract:** The primary value of AI agents in software development lies in their ability to extend the developer's capacity for reasoning and action, not to supplant human involvement. To showcase how to use agents working in tandem with developers, we designed a novel approach for carrying out coordinated renaming. Coordinated renaming, where a single rename refactoring triggers refactorings in multiple, related identifiers, is a frequent yet challenging task. Developers must manually propagate these rename refactorings across numerous files and contexts, a process that is both tedious and highly error-prone. State-of-the-art heuristic-based approaches produce an overwhelming number of false positives, while vanilla Large Language Models (LLMs) provide incomplete suggestions due to their limited context and inability to interact with refactoring tools. This leaves developers with incomplete refactorings or burdens them with filtering too many false positives. Coordinated renaming is exactly the kind of repetitive task that agents can significantly reduce the developers' burden while keeping them in the driver's seat.
We designed, implemented, and evaluated the first multi-agent framework that automates coordinated renaming. It operates on a key insight: a developer's initial refactoring is a clue to infer the scope of related refactorings. Our Scope Inference Agent first transforms this clue into an explicit, natural-language Declared Scope. The Planned Execution Agent then uses this as a strict plan to identify program elements that should undergo refactoring and safely executes the changes by invoking the IDE's own trusted refactoring APIs. Finally, the Replication Agent uses it to guide the project-wide search. We first conducted a formative study on the practice of coordinated renaming in 609K commits in 100 open-source projects and surveyed 205 developers ...

**arXiv ID:** 2601.00482
</details>

<details>
<summary><strong>Trajectory Guard -- A Lightweight, Sequence-Aware Model for Real-Time Anomaly Detection in Agentic AI</strong> - Laksh Advani - [[pdf]](https://arxiv.org/pdf/2601.00516)</summary>

**Abstract:** Autonomous LLM agents generate multi-step action plans that can fail due to contextual misalignment or structural incoherence. Existing anomaly detection methods are ill-suited for this challenge: mean-pooling embeddings dilutes anomalous steps, while contrastive-only approaches ignore sequential structure. Standard unsupervised methods on pre-trained embeddings achieve F1-scores no higher than 0.69. We introduce Trajectory Guard, a Siamese Recurrent Autoencoder with a hybrid loss function that jointly learns task-trajectory alignment via contrastive learning and sequential validity via reconstruction. This dual objective enables unified detection of both "wrong plan for this task" and "malformed plan structure." On benchmarks spanning synthetic perturbations and real-world failures from security audits (RAS-Eval) and multi-agent systems (Who\&When), we achieve F1-scores of 0.88-0.94 on balanced sets and recall of 0.86-0.92 on imbalanced external benchmarks. At 32 ms inference latency, our approach runs 17-27$\times$ faster than LLM Judge baselines, enabling real-time safety verification in production deployments.

**arXiv ID:** 2601.00516
</details>

<details>
<summary><strong>Parametrized Sharing for Multi-Agent Hybrid DRL for Multiple Multi-Functional RISs-Aided Downlink NOMA Networks</strong> - Chi-Te Kuo, Li-Hsiang Shen, Jyun-Jhe Huang - [[pdf]](https://arxiv.org/pdf/2601.00538)</summary>

**Abstract:** Multi-functional reconfigurable intelligent surface (MF-RIS) is conceived to address the communication efficiency thanks to its extended signal coverage from its active RIS capability and self-sustainability from energy harvesting (EH). We investigate the architecture of multi-MF-RISs to assist non-orthogonal multiple access (NOMA) downlink networks. We formulate an energy efficiency (EE) maximization problem by optimizing power allocation, transmit beamforming and MF-RIS configurations of amplitudes, phase-shifts and EH ratios, as well as the position of MF-RISs, while satisfying constraints of available power, user rate requirements, and self-sustainability property. We design a parametrized sharing scheme for multi-agent hybrid deep reinforcement learning (PMHRL), where the multi-agent proximal policy optimization (PPO) and deep-Q network (DQN) handle continuous and discrete variables, respectively. The simulation results have demonstrated that proposed PMHRL has the highest EE compared to other benchmarks, including cases without parametrized sharing, pure PPO and DQN. Moreover, the proposed multi-MF-RISs-aided downlink NOMA achieves the highest EE compared to scenarios of no-EH/amplification, traditional RISs, and deployment without RISs/MF-RISs under different multiple access.

**arXiv ID:** 2601.00538
</details>

<details>
<summary><strong>QUITE: A Query Rewrite System Beyond Rules with LLM Agents</strong> - Yuyang Song, Hanxu Yan, Jiale Lao, Yibo Wang, Yufei Li, Yuanchun Zhou, Jianguo Wang, Mingjie Tang - [[pdf]](https://arxiv.org/pdf/2506.07675)</summary>

**Abstract:** Query rewrite transforms SQL queries into semantically equivalent forms that run more efficiently. Existing approaches mainly rely on predefined rewrite rules, but they handle a limited subset of queries and can cause performance regressions. This limitation stems from three challenges of rule-based query rewrite: (1) it is hard to discover and verify new rules, (2) fixed rewrite rules do not generalize to new query patterns, and (3) some rewrite techniques cannot be expressed as fixed rules. Motivated by the fact that human experts exhibit significantly better rewrite ability but suffer from scalability, and Large Language Models (LLMs) have demonstrated nearly human-level semantic and reasoning abilities, we propose a new approach of using LLMs to rewrite SQL queries beyond rules. Due to the hallucination problems in LLMs, directly applying LLMs often leads to nonequivalent and suboptimal queries. To address this issue, we propose QUITE (query rewrite), a training-free and feedback-aware system based on LLM agents that rewrites SQL queries into semantically equivalent forms with significantly better performance, covering a broader range of query patterns and rewrite strategies compared to rule-based methods. Firstly, we design a multi-agent framework controlled by a finite state machine (FSM) to equip LLMs with the ability to use external tools and enhance the rewrite process with real-time database feedback. Secondly, we develop a rewrite middleware to enhance the ability of LLMs to generate optimized query equivalents. Finally, we employ a novel hint injection technique to improve execution plans for rewritten queries. Extensive experiments show that QUITE reduces query execution time by up to 35.8% over state-of-the-art approaches and produces 24.1% more rewrites than prior methods, covering query cases that earlier systems did not handle.

**arXiv ID:** 2506.07675
</details>

<details>
<summary><strong>Adaptive GPU Resource Allocation for Multi-Agent Collaborative Reasoning in Serverless Environments</strong> - Guilin Zhang, Wulan Guo, Ziqi Tan - [[pdf]](https://arxiv.org/pdf/2512.22149)</summary>

**Abstract:** Multi-agent systems powered by large language models have emerged as a promising paradigm for solving complex reasoning tasks through collaborative intelligence. However, efficiently deploying these systems on serverless GPU platforms presents significant resource allocation challenges due to heterogeneous agent workloads, varying computational demands, and the need for cost-effective scaling. This paper presents an adaptive GPU resource allocation framework that achieves 85% latency reduction compared to round-robin scheduling while maintaining comparable throughput to static allocation, using an O(N) complexity algorithm for real-time adaptation. Our approach dynamically allocates GPU resources based on workload characteristics, agent priorities, and minimum resource requirements, enabling efficient utilization while maintaining quality of service. The framework addresses three key challenges: (1) heterogeneous computational demands across lightweight coordinators and heavyweight specialists, (2) dynamic workload fluctuations requiring millisecond-scale reallocation, and (3) capacity constraints in serverless environments. Through comprehensive simulations modeling realistic multi-agent workflows with four heterogeneous agents, we demonstrate that adaptive allocation outperforms static equal and round-robin strategies across latency, cost, and GPU utilization metrics. The framework provides a practical solution for deploying cost-efficient multi-agent AI systems on serverless GPU infrastructure.

**arXiv ID:** 2512.22149
</details>

<details>
<summary><strong>BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization</strong> - Iris Xu, Guangtao Zeng, Zexue He, Charles Jin, Aldo Pareja, Dan Gutfreund, Chuang Gan, Zhang-Wei Hong - [[pdf]](https://arxiv.org/pdf/2512.23631)</summary>

**Abstract:** Large language models (LLMs) have shown strong reasoning and coding capabilities, yet they struggle to generalize to real-world software engineering (SWE) problems that are long-horizon and out of distribution. Existing systems often rely on a single agent to handle the entire workflow-interpreting issues, navigating large codebases, and implementing fixes-within one reasoning chain. Such monolithic designs force the model to retain irrelevant context, leading to spurious correlations and poor generalization. Motivated by how human engineers decompose complex problems, we propose structuring SWE agents as orchestrators coordinating specialized sub-agents for sub-tasks such as localization, editing, and validation. The challenge lies in discovering effective hierarchies automatically: as the number of sub-agents grows, the search space becomes combinatorial, and it is difficult to attribute credit to individual sub-agents within a team. We address these challenges by formulating hierarchy discovery as a multi-armed bandit (MAB) problem, where each arm represents a candidate sub-agent and the reward measures its helpfulness when collaborating with others. This framework, termed Bandit Optimization for Agent Design (BOAD), enables efficient exploration of sub-agent designs under limited evaluation budgets. On SWE-bench-Verified, BOAD outperforms single-agent and manually designed multi-agent systems. On SWE-bench-Live, featuring more recent and out-of-distribution issues, our 36B system ranks second on the leaderboard at the time of evaluation, surpassing larger models such as GPT-4 and Claude. These results demonstrate that automatically discovered hierarchical multi-agent systems significantly improve generalization on challenging long-horizon SWE tasks. Code is available at this https URL.

**arXiv ID:** 2512.23631
</details>

<details>
<summary><strong>μACP: A Formal Calculus for Expressive, Resource-Constrained Agent Communication</strong> - Arnab Mallick, Indraveni Chebolu - [[pdf]](https://arxiv.org/pdf/2601.00219)</summary>

**Abstract:** Agent communication remains a foundational problem in multi-agent systems: protocols such as FIPA-ACL guarantee semantic richness but are intractable for constrained environments, while lightweight IoT protocols achieve efficiency at the expense of expressiveness. This paper presents $\mu$ACP, a formal calculus for expressive agent communication under explicit resource bounds. We formalize the Resource-Constrained Agent Communication (RCAC) model, prove that a minimal four-verb basis \textit{\{PING, TELL, ASK, OBSERVE\}} is suffices to encode finite-state FIPA protocols, and establish tight information-theoretic bounds on message complexity. We further show that $\mu$ACP can implement standard consensus under partial synchrony and crash faults, yielding a constructive coordination framework for edge-native agents. Formal verification in TLA$^{+}$ (model checking) and Coq (mechanized invariants) establishes safety and boundedness, and supports liveness under modeled assumptions. Large-scale system simulations confirm ACP achieves a median end-to-end message latency of 34 ms (95th percentile 104 ms) at scale, outperforming prior agent and IoT protocols under severe resource constraints. The main contribution is a unified calculus that reconciles semantic expressiveness with provable efficiency, providing a rigorous foundation for the next generation of resource-constrained multi-agent systems.

**arXiv ID:** 2601.00219
</details>

<details>
<summary><strong>Offline Multi-Agent Reinforcement Learning for 6G Communications: Fundamentals, Applications and Future Directions</strong> - Eslam Eldeeb, Hirley Alves - [[pdf]](https://arxiv.org/pdf/2601.00321)</summary>

**Abstract:** The next-generation wireless technologies, including beyond 5G and 6G networks, are paving the way for transformative applications such as vehicle platooning, smart cities, and remote surgery. These innovations are driven by a vast array of interconnected wireless entities, including IoT devices, access points, UAVs, and CAVs, which increase network complexity and demand more advanced decision-making algorithms. Artificial intelligence (AI) and machine learning (ML), especially reinforcement learning (RL), are key enablers for such networks, providing solutions to high-dimensional and complex challenges. However, as networks expand to multi-agent environments, traditional online RL approaches face cost, safety, and scalability limitations. Offline multi-agent reinforcement learning (MARL) offers a promising solution by utilizing pre-collected data, reducing the need for real-time interaction. This article introduces a novel offline MARL algorithm based on conservative Q-learning (CQL), ensuring safe and efficient training. We extend this with meta-learning to address dynamic environments and validate the approach through use cases in radio resource management and UAV networks. Our work highlights offline MARL's advantages, limitations, and future directions in wireless applications.

**arXiv ID:** 2601.00321
</details>

<details>
<summary><strong>Cultural Palette: Pluralising Culture Alignment via Multi-agent Palette</strong> - Jiahao Yuan, Zixiang Di, Shangzixin Zhao, Zhiqing Cui, Hanqing Wang, Guisong Yang, Usman Naseem - [[pdf]](https://arxiv.org/pdf/2412.11167)</summary>

**Abstract:** Large language models (LLMs) face challenges in aligning with diverse cultural values despite their remarkable performance in generation, which stems from inherent monocultural biases and difficulties in capturing nuanced cultural semantics. Existing methods struggle to adapt to unknown culture after fine-tuning. Inspired by cultural geography across five continents, we propose Cultural Palette, a multi-agent framework that redefines cultural alignment as an adaptive "color-blending" process for country-specific adaptation. Our approach harnesses cultural geography across five continents through three key steps: First, we synthesize the Pentachromatic Cultural Palette Dataset using GPT-4o, refining continental-level dialogues with Hofstede's cultural dimensions to establish foundational cultural representations. Second, five continent-level alignment agents form specialized cultural communities that generate region-specific draft responses. Third, a Meta Agent employs Cultural MoErges to dynamically blend these cultural "colors" through attention-gated parameter merging, akin to mixing pigments on a palette, resolving conflicts while preserving cultural nuances to produce the final culturally-aligned response. Extensive experiments across various countries demonstrate that \textit{Cultural Palette} surpasses existing baselines in cultural alignment.

**arXiv ID:** 2412.11167
</details>

<details>
<summary><strong>Can Optimal Transport Improve Federated Inverse Reinforcement Learning?</strong> - David Millard, Ali Baheri - [[pdf]](https://arxiv.org/pdf/2601.00309)</summary>

**Abstract:** In robotics and multi-agent systems, fleets of autonomous agents often operate in subtly different environments while pursuing a common high-level objective. Directly pooling their data to learn a shared reward function is typically impractical due to differences in dynamics, privacy constraints, and limited communication bandwidth. This paper introduces an optimal transport-based approach to federated inverse reinforcement learning (IRL). Each client first performs lightweight Maximum Entropy IRL locally, adhering to its computational and privacy limitations. The resulting reward functions are then fused via a Wasserstein barycenter, which considers their underlying geometric structure. We further prove that this barycentric fusion yields a more faithful global reward estimate than conventional parameter averaging methods in federated learning. Overall, this work provides a principled and communication-efficient framework for deriving a shared reward that generalizes across heterogeneous agents and environments.

**arXiv ID:** 2601.00309
</details>

<details>
<summary><strong>Optimal Transport-Based Decentralized Multi-Agent Distribution Matching</strong> - Kooktae Lee - [[pdf]](https://arxiv.org/pdf/2601.00548)</summary>

**Abstract:** This paper presents a decentralized control framework for distribution matching in multi-agent systems (MAS), where agents collectively achieve a prescribed terminal spatial distribution. The problem is formulated using optimal transport (Wasserstein distance), which provides a principled measure of distributional discrepancy and serves as the basis for the control design. To avoid solving the global optimal transport problem directly, the distribution-matching objective is reformulated into a tractable per-agent decision process, enabling each agent to identify its desired terminal locations using only locally available information. A sequential weight-update rule is introduced to construct feasible local transport plans, and a memory-based correction mechanism is incorporated to maintain reliable operation under intermittent and range-limited communication. Convergence guarantees are established, showing cycle-wise improvement of a surrogate transport cost under both linear and nonlinear agent dynamics. Simulation results demonstrate that the proposed framework achieves effective and scalable distribution matching while operating fully in a decentralized manner.

**arXiv ID:** 2601.00548
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>Progressive Ideation using an Agentic AI Framework for Human-AI Co-Creation</strong> - Sankar B, Srinidhi Ranjini Girish, Aadya Bharti, Dibakar Sen - [[pdf]](https://arxiv.org/pdf/2601.00475)</summary>

**Abstract:** The generation of truly novel and diverse ideas is important for contemporary engineering design, yet it remains a significant cognitive challenge for novice designers. Current 'single-spurt' AI systems exacerbate this challenge by producing a high volume of semantically clustered ideas. We propose MIDAS (Meta-cognitive Ideation through Distributed Agentic AI System), a novel framework that replaces the single-AI paradigm with a distributed 'team' of specialized AI agents designed to emulate the human meta-cognitive ideation workflow. This agentic system progressively refines ideas and assesses each one for both global novelty (against existing solutions) and local novelty (against previously generated ideas). MIDAS, therefore, demonstrates a viable and progressive paradigm for true human-AI co-creation, elevating the human designer from a passive filterer to a participatory, active, collaborative partner.

**arXiv ID:** 2601.00475
</details>

<details>
<summary><strong>SLAP: Slapband-based Autonomous Perching Drone with Failure Recovery for Vertical Tree Trunks</strong> - Julia Di, Kenneth A. W. Hoffmann, Tony G. Chen, Tian-Ao Ren, Mark R. Cutkosky - [[pdf]](https://arxiv.org/pdf/2601.00238)</summary>

**Abstract:** Perching allows unmanned aerial vehicles (UAVs) to reduce energy consumption, remain anchored for surface sampling operations, or stably survey their surroundings. Previous efforts for perching on vertical surfaces have predominantly focused on lightweight mechanical design solutions with relatively scant system-level integration. Furthermore, perching strategies for vertical surfaces commonly require high-speed, aggressive landing operations that are dangerous for a surveyor drone with sensitive electronics onboard. This work presents the preliminary investigation of a perching approach suitable for larger drones that both gently perches on vertical tree trunks and reacts and recovers from perch failures. The system in this work, called SLAP, consists of vision-based perch site detector, an IMU (inertial-measurement-unit)-based perch failure detector, an attitude controller for soft perching, an optical close-range detection system, and a fast active elastic gripper with microspines made from commercially-available slapbands. We validated this approach on a modified 1.2 kg commercial quadrotor with component and system analysis. Initial human-in-the-loop autonomous indoor flight experiments achieved a 75% perch success rate on a real oak tree segment across 20 flights, and 100% perch failure recovery across 2 flights with induced failures.

**arXiv ID:** 2601.00238
</details>

<details>
<summary><strong>LLM-Based Agentic Exploration for Robot Navigation & Manipulation with Skill Orchestration</strong> - Abu Hanif Muhammad Syarubany, Farhan Zaki Rahmani, Trio Widianto - [[pdf]](https://arxiv.org/pdf/2601.00555)</summary>

**Abstract:** This paper presents an end-to-end LLM-based agentic exploration system for an indoor shopping task, evaluated in both Gazebo simulation and a corresponding real-world corridor layout. The robot incrementally builds a lightweight semantic map by detecting signboards at junctions and storing direction-to-POI relations together with estimated junction poses, while AprilTags provide repeatable anchors for approach and alignment. Given a natural-language shopping request, an LLM produces a constrained discrete action at each junction (direction and whether to enter a store), and a ROS finite-state main controller executes the decision by gating modular motion primitives, including local-costmap-based obstacle avoidance, AprilTag approaching, store entry, and grasping. Qualitative results show that the integrated stack can perform end-to-end task execution from user instruction to multi-store navigation and object retrieval, while remaining modular and debuggable through its text-based map and logged decision history.

**arXiv ID:** 2601.00555
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (22 papers)</h2></summary>

<details>
<summary><strong>ClinicalReTrial: A Self-Evolving AI Agent for Clinical Trial Protocol Optimization</strong> - Sixue Xing, Xuanye Xia, Kerui Wu, Meng Jiang, Jintai Chen, Tianfan Fu - [[pdf]](https://arxiv.org/pdf/2601.00290)</summary>

**Abstract:** Clinical trial failure remains a central bottleneck in drug development, where minor protocol design flaws can irreversibly compromise outcomes despite promising therapeutics. Although cutting-edge AI methods achieve strong performance in predicting trial success, they are inherently reactive for merely diagnosing risk without offering actionable remedies once failure is anticipated. To fill this gap, this paper proposes ClinicalReTrial, a self-evolving AI agent framework that addresses this gap by casting clinical trial reasoning as an iterative protocol redesign problem. Our method integrates failure diagnosis, safety-aware modification, and candidate evaluation in a closed-loop, reward-driven optimization framework. Serving the outcome prediction model as a simulation environment, ClinicalReTrial enables low-cost evaluation of protocol modifications and provides dense reward signals for continuous self-improvement. To support efficient exploration, the framework maintains hierarchical memory that captures iteration-level feedback within trials and distills transferable redesign patterns across trials. Empirically, ClinicalReTrial improves 83.3% of trial protocols with a mean success probability gain of 5.7%, and retrospective case studies demonstrate strong alignment between the discovered redesign strategies and real-world clinical trial modifications.

**arXiv ID:** 2601.00290
</details>

<details>
<summary><strong>Multiagent Reinforcement Learning for Liquidity Games</strong> - Alicia Vidler, Gal A. Kaminka - [[pdf]](https://arxiv.org/pdf/2601.00324)</summary>

**Abstract:** Making use of swarm methods in financial market modeling of liquidity, and techniques from financial analysis in swarm analysis, holds the potential to advance both research areas. In swarm research, the use of game theory methods holds the promise of explaining observed phenomena of collective utility adherence with rational self-interested swarm participants. In financial markets, a better understanding of how independent financial agents may self-organize for the betterment and stability of the marketplace would be a boon for market design researchers. This paper unifies Liquidity Games, where trader payoffs depend on aggregate liquidity within a trade, with Rational Swarms, where decentralized agents use difference rewards to align self-interested learning with global objectives. We offer a theoretical frameworks where we define a swarm of traders whose collective objective is market liquidity provision while maintaining agent independence. Using difference rewards within a Markov team games framework, we show that individual liquidity-maximizing behaviors contribute to overall market liquidity without requiring coordination or collusion. This Financial Swarm model provides a framework for modeling rational, independent agents where they achieve both individual profitability and collective market efficiency in bilateral asset markets.

**arXiv ID:** 2601.00324
</details>

<details>
<summary><strong>An Agentic Framework for Neuro-Symbolic Programming</strong> - Aliakbar Nafar, Chetan Chigurupati, Danial Kamali, Hamid Karimian, Parisa Kordjamshidi - [[pdf]](https://arxiv.org/pdf/2601.00743)</summary>

**Abstract:** Integrating symbolic constraints into deep learning models could make them more robust, interpretable, and data-efficient. Still, it remains a time-consuming and challenging task. Existing frameworks like DomiKnowS help this integration by providing a high-level declarative programming interface, but they still assume the user is proficient with the library's specific syntax. We propose AgenticDomiKnowS (ADS) to eliminate this dependency. ADS translates free-form task descriptions into a complete DomiKnowS program using an agentic workflow that creates and tests each DomiKnowS component separately. The workflow supports optional human-in-the-loop intervention, enabling users familiar with DomiKnowS to refine intermediate outputs. We show how ADS enables experienced DomiKnowS users and non-users to rapidly construct neuro-symbolic programs, reducing development time from hours to 10-15 minutes.

**arXiv ID:** 2601.00743
</details>

<details>
<summary><strong>Yahtzee: Reinforcement Learning Techniques for Stochastic Combinatorial Games</strong> - Nicholas A. Pape - [[pdf]](https://arxiv.org/pdf/2601.00007)</summary>

**Abstract:** Yahtzee is a classic dice game with a stochastic, combinatorial structure and delayed rewards, making it an interesting mid-scale RL benchmark. While an optimal policy for solitaire Yahtzee can be computed using dynamic programming methods, multiplayer is intractable, motivating approximation methods. We formulate Yahtzee as a Markov Decision Process (MDP), and train self-play agents using various policy gradient methods: REINFORCE, Advantage Actor-Critic (A2C), and Proximal Policy Optimization (PPO), all using a multi-headed network with a shared trunk. We ablate feature and action encodings, architecture, return estimators, and entropy regularization to understand their impact on learning. Under a fixed training budget, REINFORCE and PPO prove sensitive to hyperparameters and fail to reach near-optimal performance, whereas A2C trains robustly across a range of settings. Our agent attains a median score of 241.78 points over 100,000 evaluation games, within 5.0\% of the optimal DP score of 254.59, achieving the upper section bonus and Yahtzee at rates of 24.9\% and 34.1\%, respectively. All models struggle to learn the upper bonus strategy, overindexing on four-of-a-kind's, highlighting persistent long-horizon credit-assignment and exploration challenges.

**arXiv ID:** 2601.00007
</details>

<details>
<summary><strong>An Empirical Evaluation of LLM-Based Approaches for Code Vulnerability Detection: RAG, SFT, and Dual-Agent Systems</strong> - Md Hasan Saju, Maher Muhtadi, Akramul Azim - [[pdf]](https://arxiv.org/pdf/2601.00254)</summary>

**Abstract:** The rapid advancement of Large Language Models (LLMs) presents new opportunities for automated software vulnerability detection, a crucial task in securing modern codebases. This paper presents a comparative study on the effectiveness of LLM-based techniques for detecting software vulnerabilities. The study evaluates three approaches, Retrieval-Augmented Generation (RAG), Supervised Fine-Tuning (SFT), and a Dual-Agent LLM framework, against a baseline LLM model. A curated dataset was compiled from Big-Vul and real-world code repositories from GitHub, focusing on five critical Common Weakness Enumeration (CWE) categories: CWE-119, CWE-399, CWE-264, CWE-20, and CWE-200. Our RAG approach, which integrated external domain knowledge from the internet and the MITRE CWE database, achieved the highest overall accuracy (0.86) and F1 score (0.85), highlighting the value of contextual augmentation. Our SFT approach, implemented using parameter-efficient QLoRA adapters, also demonstrated strong performance. Our Dual-Agent system, an architecture in which a secondary agent audits and refines the output of the first, showed promise in improving reasoning transparency and error mitigation, with reduced resource overhead. These results emphasize that incorporating a domain expertise mechanism significantly strengthens the practical applicability of LLMs in real-world vulnerability detection tasks.

**arXiv ID:** 2601.00254
</details>

<details>
<summary><strong>Next Generation Intelligent Low-Altitude Economy Deployments: The O-RAN Perspective</strong> - Aly Sabri Abdalla, Vuk Marojevic - [[pdf]](https://arxiv.org/pdf/2601.00257)</summary>

**Abstract:** Despite the growing interest in low-altitude economy (LAE) applications, including UAV-based logistics and emergency response, fundamental challenges remain in orchestrating such missions over complex, signal-constrained environments. These include the absence of real-time, resilient, and context-aware orchestration of aerial nodes with limited integration of artificial intelligence (AI) specialized for LAE missions. This paper introduces an open radio access network (O-RAN)-enabled LAE framework that leverages seamless coordination between the disaggregated RAN architecture, open interfaces, and RAN intelligent controllers (RICs) to facilitate closed-loop, AI-optimized, and mission-critical LAE operations. We evaluate the feasibility and performance of the proposed architecture via a semantic-aware rApp that acts as a terrain interpreter, offering semantic guidance to a reinforcement learning-enabled xApp, which performs real-time trajectory planning for LAE swarm nodes. We survey the capabilities of UAV testbeds that can be leveraged for LAE research, and present critical research challenges and standardization needs.

**arXiv ID:** 2601.00257
</details>

<details>
<summary><strong>E-GRPO: High Entropy Steps Drive Effective Reinforcement Learning for Flow Models</strong> - Shengjun Zhang, Zhang Zhang, Chensheng Dai, Yueqi Duan - [[pdf]](https://arxiv.org/pdf/2601.00423)</summary>

**Abstract:** Recent reinforcement learning has enhanced the flow matching models on human preference alignment. While stochastic sampling enables the exploration of denoising directions, existing methods which optimize over multiple denoising steps suffer from sparse and ambiguous reward signals. We observe that the high entropy steps enable more efficient and effective exploration while the low entropy steps result in undistinguished roll-outs. To this end, we propose E-GRPO, an entropy aware Group Relative Policy Optimization to increase the entropy of SDE sampling steps. Since the integration of stochastic differential equations suffer from ambiguous reward signals due to stochasticity from multiple steps, we specifically merge consecutive low entropy steps to formulate one high entropy step for SDE sampling, while applying ODE sampling on other steps. Building upon this, we introduce multi-step group normalized advantage, which computes group-relative advantages within samples sharing the same consolidated SDE denoising step. Experimental results on different reward settings have demonstrated the effectiveness of our methods.

**arXiv ID:** 2601.00423
</details>

<details>
<summary><strong>IRPO: Scaling the Bradley-Terry Model via Reinforcement Learning</strong> - Haonan Song, Qingchen Xie, Huan Zhu, Feng Xiao, Luxi Xing, Fuzhen Li, Liu Kang, Feng Jiang, Zhiyong Zheng, Fan Yang - [[pdf]](https://arxiv.org/pdf/2601.00677)</summary>

**Abstract:** Generative Reward Models (GRMs) have attracted considerable research interest in reward modeling due to their interpretability, inference-time scalability, and potential for refinement through reinforcement learning (RL). However, widely used pairwise GRMs create a computational bottleneck when integrated with RL algorithms such as Group Relative Policy Optimization (GRPO). This bottleneck arises from two factors: (i) the O(n^2) time complexity of pairwise comparisons required to obtain relative scores, and (ii) the computational overhead of repeated sampling or additional chain-of-thought (CoT) reasoning to improve performance. To address the first factor, we propose Intergroup Relative Preference Optimization (IRPO), a novel RL framework that incorporates the well-established Bradley-Terry model into GRPO. By generating a pointwise score for each response, IRPO enables efficient evaluation of arbitrarily many candidates during RL training while preserving interpretability and fine-grained reward signals. Experimental results demonstrate that IRPO achieves state-of-the-art (SOTA) performance among pointwise GRMs across multiple benchmarks, with performance comparable to that of current leading pairwise GRMs. Furthermore, we show that IRPO significantly outperforms pairwise GRMs in post-training evaluations.

**arXiv ID:** 2601.00677
</details>

<details>
<summary><strong>Benchmark Success, Clinical Failure: When Reinforcement Learning Optimizes for Benchmarks, Not Patients</strong> - Armin Berger, Manuela Bergau, Helen Schneider, Saad Ahmad, Tom Anglim Lagones, Gianluca Brugnara, Martha Foltyn-Dumitru, Kai Schlamp, Philipp Vollmuth, Rafet Sifa - [[pdf]](https://arxiv.org/pdf/2512.23090)</summary>

**Abstract:** Recent Reinforcement Learning (RL) advances for Large Language Models (LLMs) have improved reasoning tasks, yet their resource-constrained application to medical imaging remains underexplored. We introduce ChexReason, a vision-language model trained via R1-style methodology (SFT followed by GRPO) using only 2,000 SFT samples, 1,000 RL samples, and a single A100 GPU. Evaluations on CheXpert and NIH benchmarks reveal a fundamental tension: GRPO recovers in-distribution performance (23% improvement on CheXpert, macro-F1 = 0.346) but degrades cross-dataset transferability (19% drop on NIH). This mirrors high-resource models like NV-Reason-CXR-3B, suggesting the issue stems from the RL paradigm rather than scale. We identify a generalization paradox where the SFT checkpoint uniquely improves on NIH before optimization, indicating teacher-guided reasoning captures more institution-agnostic features. Furthermore, cross-model comparisons show structured reasoning scaffolds benefit general-purpose VLMs but offer minimal gain for medically pre-trained models. Consequently, curated supervised fine-tuning may outperform aggressive RL for clinical deployment requiring robustness across diverse populations.

**arXiv ID:** 2512.23090
</details>

<details>
<summary><strong>Thinking on Maps: How Foundation Model Agents Explore, Remember, and Reason Map Environments</strong> - Zhiwei Wei, Yuxing Liu, Hua Liao, Wenjia Xu - [[pdf]](https://arxiv.org/pdf/2512.24504)</summary>

**Abstract:** Map environments provide a fundamental medium for representing spatial structure. Understanding how foundation model (FM) agents understand and act in such environments is therefore critical for enabling reliable map-based reasoning and applications. However, most existing evaluations of spatial ability in FMs rely on static map inputs or text-based queries, overlooking the interactive and experience-driven nature of spatial this http URL this paper, we propose an interactive evaluation framework to analyze how FM agents explore, remember, and reason in symbolic map environments. Agents incrementally explore partially observable grid-based maps consisting of roads, intersections, and points of interest (POIs), receiving only local observations at each step. Spatial understanding is then evaluated using six kinds of spatial tasks. By systematically varying exploration strategies, memory representations, and reasoning schemes across multiple foundation models, we reveal distinct functional roles of these components. Exploration primarily affects experience acquisition but has a limited impact on final reasoning accuracy. In contrast, memory representation plays a central role in consolidating spatial experience, with structured memories particularly sequential and graph-based representations, substantially improving performance on structure-intensive tasks such as path planning. Reasoning schemes further shape how stored spatial knowledge is used, with advanced prompts supporting more effective multi-step inference. We further observe that spatial reasoning performance saturates across model versions and scales beyond a certain capability threshold, indicating that improvements in map-based spatial understanding require mechanisms tailored to spatial representation and reasoning rather than scaling alone.

**arXiv ID:** 2512.24504
</details>

<details>
<summary><strong>MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training</strong> - Taicheng Guo, Hai Wang, ChaoChun Liu, Mohsen Golalikhani, Xin Chen, Xiangliang Zhang, Chandan K. Reddy - [[pdf]](https://arxiv.org/pdf/2510.12831)</summary>

**Abstract:** Multi-turn Text-to-SQL aims to translate a user's conversational utterances into executable SQL while preserving dialogue coherence and grounding to the target schema. However, most existing systems only regard this task as a simple text translation task and follow a short-horizon paradigm, generating a query per turn without execution, explicit verification, and refinement, which leads to non-executable or incoherent outputs. We present MTSQL-R1, an agentic training framework for long-horizon multi-turn Text-to-SQL. We cast the task as a Markov Decision Process (MDP) in which an agent interacts with (i) a database for execution feedback and (ii) a persistent dialogue memory for coherence verification, performing an iterative propose to execute -> verify -> refine cycle until all checks pass. Experiments on COSQL and SPARC demonstrate that MTSQL-R1 consistently outperforms strong baselines, highlighting the importance of environment-driven verification and memory-guided refinement for conversational semantic parsing. Full recipes (including code, trained models, logs, reasoning trajectories, etc.) will be released after the internal review to contribute to community research.

**arXiv ID:** 2510.12831
</details>

<details>
<summary><strong>RIMRULE: Improving Tool-Using Language Agents via MDL-Guided Rule Learning</strong> - Xiang Gao, Yuguang Yao, Qi Zhang, Kaiwen Dong, Avinash Baidya, Ruocheng Guo, Hilaf Hasson, Kamalika Das - [[pdf]](https://arxiv.org/pdf/2601.00086)</summary>

**Abstract:** Large language models (LLMs) often struggle to use tools reliably in domain-specific settings, where APIs may be idiosyncratic, under-documented, or tailored to private workflows. This highlights the need for effective adaptation to task-specific tools. We propose RIMRULE, a neuro-symbolic approach for LLM adaptation based on dynamic rule injection. Compact, interpretable rules are distilled from failure traces and injected into the prompt during inference to improve task performance. These rules are proposed by the LLM itself and consolidated using a Minimum Description Length (MDL) objective that favors generality and conciseness. Each rule is stored in both natural language and a structured symbolic form, supporting efficient retrieval at inference time. Experiments on tool-use benchmarks show that this approach improves accuracy on both seen and unseen tools without modifying LLM weights. It outperforms prompting-based adaptation methods and complements finetuning. Moreover, rules learned from one LLM can be reused to improve others, including long reasoning LLMs, highlighting the portability of symbolic knowledge across architectures.

**arXiv ID:** 2601.00086
</details>

<details>
<summary><strong>Universal Adaptive Constraint Propagation: Scaling Structured Inference for Large Language Models via Meta-Reinforcement Learning</strong> - Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma - [[pdf]](https://arxiv.org/pdf/2601.00095)</summary>

**Abstract:** Large language models increasingly require structured inference, from JSON schema enforcement to multi-lingual parsing, where outputs must satisfy complex constraints. We introduce MetaJuLS, a meta-reinforcement learning approach that learns universal constraint propagation policies applicable across languages and tasks without task-specific retraining. By formulating structured inference as adaptive constraint propagation and training a Graph Attention Network with meta-learning, MetaJuLS achieves 1.5--2.0$\times$ speedups over GPU-optimized baselines while maintaining within 0.2\% accuracy of state-of-the-art parsers. On Universal Dependencies across 10 languages and LLM-constrained generation (LogicBench, GSM8K-Constrained), MetaJuLS demonstrates rapid cross-domain adaptation: a policy trained on English parsing adapts to new languages and tasks with 5--10 gradient steps (5--15 seconds) rather than requiring hours of task-specific training. Mechanistic analysis reveals the policy discovers human-like parsing strategies (easy-first) and novel non-intuitive heuristics. By reducing propagation steps in LLM deployments, MetaJuLS contributes to Green AI by directly reducing inference carbon footprint.

**arXiv ID:** 2601.00095
</details>

<details>
<summary><strong>Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach</strong> - Biao Wu, Meng Fang, Ling Chen, Ke Xu, Tao Cheng, Jun Wang - [[pdf]](https://arxiv.org/pdf/2601.00388)</summary>

**Abstract:** Recent advances in vision-language models have opened up new possibilities for reasoning-driven image geolocalization. However, existing approaches often rely on synthetic reasoning annotations or external image retrieval, which can limit interpretability and generalizability. In this paper, we present Geo-R, a retrieval-free framework that uncovers structured reasoning paths from existing ground-truth coordinates and optimizes geolocation accuracy via reinforcement learning. We propose the Chain of Region, a rule-based hierarchical reasoning paradigm that generates precise, interpretable supervision by mapping GPS coordinates to geographic entities (e.g., country, province, city) without relying on model-generated or synthetic labels. Building on this, we introduce a lightweight reinforcement learning strategy with coordinate-aligned rewards based on Haversine distance, enabling the model to refine predictions through spatially meaningful feedback. Our approach bridges structured geographic reasoning with direct spatial supervision, yielding improved localization accuracy, stronger generalization, and more transparent inference. Experimental results across multiple benchmarks confirm the effectiveness of Geo-R, establishing a new retrieval-free paradigm for scalable and interpretable image geolocalization. To facilitate further research and ensure reproducibility, both the model and code will be made publicly available.

**arXiv ID:** 2601.00388
</details>

<details>
<summary><strong>From Sight to Insight: Improving Visual Reasoning Capabilities of Multimodal Models via Reinforcement Learning</strong> - Omar Sharif, Eftekhar Hossain, Patrick Ng - [[pdf]](https://arxiv.org/pdf/2601.00215)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as a promising approach for eliciting reasoning chains before generating final answers. However, multimodal large language models (MLLMs) generate reasoning that lacks integration of visual information. This limits their ability to solve problems that demand accurate visual perception, such as visual puzzles. We show that visual perception is the key bottleneck in such tasks: converting images into textual descriptions significantly improves performance, yielding gains of 26.7% for Claude 3.5 and 23.6% for Claude 3.7.
To address this, we investigate reward-driven RL as a mechanism to unlock long visual reasoning in open-source MLLMs without requiring costly supervision. We design and evaluate six reward functions targeting different reasoning aspects, including image understanding, thinking steps, and answer accuracy. Using group relative policy optimization (GRPO), our approach explicitly incentivizes longer, structured reasoning and mitigates bypassing of visual information. Experiments on Qwen-2.5-VL-7B achieve 5.56% improvements over the base model, with consistent gains across both in-domain and out-of-domain settings.

**arXiv ID:** 2601.00215
</details>

<details>
<summary><strong>Optimizing Retrieval for RAG via Reinforcement Learning</strong> - Jiawei Zhou, Lei Chen - [[pdf]](https://arxiv.org/pdf/2510.24652)</summary>

**Abstract:** As retrieval-augmented generation (RAG) becomes more widespread, the role of retrieval is shifting from retrieving information for human browsing to retrieving context for AI reasoning. This shift creates more complex search environments, where relevance is difficult to pre-define. Existing retrievers rely on supervised fine-tuning (SFT) with human labels or synthetic data, resulting in static relevance that struggles to adapt to diverse RAG environments. To address this challenge, we propose R3, a Retrieval framework optimized for RAG through Reinforcement learning (RL). Specifically, we adopt an RL training paradigm that enables the retriever to explore and self-improve within given RAG environments, automating the learning process with minimal manual experimentation or tuning effort. Extensive experiments across diverse tasks demonstrate that R3 improves RAG performance by 5.2% over the original retriever and surpasses state-of-the-art retrievers by 4.9%, while achieving comparable results to LLM-augmented retrieval and RAG systems built on post-trained or instruction-tuned LLMs. It is both efficient and practical, requiring only 4 GPUs and completing training within a single day.

**arXiv ID:** 2510.24652
</details>

<details>
<summary><strong>GRL-SNAM: Geometric Reinforcement Learning with Path Differential Hamiltonians for Simultaneous Navigation and Mapping in Unknown Environments</strong> - Aditya Sai Ellendula, Yi Wang, Minh Nguyen, Chandrajit Bajaj - [[pdf]](https://arxiv.org/pdf/2601.00116)</summary>

**Abstract:** We present GRL-SNAM, a geometric reinforcement learning framework for Simultaneous Navigation and Mapping(SNAM) in unknown environments. A SNAM problem is challenging as it needs to design hierarchical or joint policies of multiple agents that control the movement of a real-life robot towards the goal in mapless environment, i.e. an environment where the map of the environment is not available apriori, and needs to be acquired through sensors. The sensors are invoked from the path learner, i.e. navigator, through active query responses to sensory agents, and along the motion path. GRL-SNAM differs from preemptive navigation algorithms and other reinforcement learning methods by relying exclusively on local sensory observations without constructing a global map. Our approach formulates path navigation and mapping as a dynamic shortest path search and discovery process using controlled Hamiltonian optimization: sensory inputs are translated into local energy landscapes that encode reachability, obstacle barriers, and deformation constraints, while policies for sensing, planning, and reconfiguration evolve stagewise via updating Hamiltonians. A reduced Hamiltonian serves as an adaptive score function, updating kinetic/potential terms, embedding barrier constraints, and continuously refining trajectories as new local information arrives. We evaluate GRL-SNAM on two different 2D navigation tasks. Comparing against local reactive baselines and global policy learning references under identical stagewise sensing constraints, it preserves clearance, generalizes to unseen layouts, and demonstrates that Geometric RL learning via updating Hamiltonians enables high-quality navigation through minimal exploration via local energy refinement rather than extensive global mapping. The code is publicly available on \href{this https URL}{Github}.

**arXiv ID:** 2601.00116
</details>

<details>
<summary><strong>Reinforcement Learning with Function Approximation for Non-Markov Processes</strong> - Ali Devran Kara - [[pdf]](https://arxiv.org/pdf/2601.00151)</summary>

**Abstract:** We study reinforcement learning methods with linear function approximation under non-Markov state and cost processes. We first consider the policy evaluation method and show that the algorithm converges under suitable ergodicity conditions on the underlying non-Markov processes. Furthermore, we show that the limit corresponds to the fixed point of a joint operator composed of an orthogonal projection and the Bellman operator of an auxiliary \emph{Markov} decision process.
For Q-learning with linear function approximation, as in the Markov setting, convergence is not guaranteed in general. We show, however, that for the special case where the basis functions are chosen based on quantization maps, the convergence can be shown under similar ergodicity conditions. Finally, we apply our results to partially observed Markov decision processes, where finite-memory variables are used as state representations, and we derive explicit error bounds for the limits of the resulting learning algorithms.

**arXiv ID:** 2601.00151
</details>

<details>
<summary><strong>Traffic-Aware Optimal Taxi Placement Using Graph Neural Network-Based Reinforcement Learning</strong> - Sonia Khetarpaul, P Y Sharan - [[pdf]](https://arxiv.org/pdf/2601.00607)</summary>

**Abstract:** In the context of smart city transportation, efficient matching of taxi supply with passenger demand requires real-time integration of urban traffic network data and mobility patterns. Conventional taxi hotspot prediction models often rely solely on historical demand, overlooking dynamic influences such as traffic congestion, road incidents, and public events. This paper presents a traffic-aware, graph-based reinforcement learning (RL) framework for optimal taxi placement in metropolitan environments. The urban road network is modeled as a graph where intersections represent nodes, road segments serve as edges, and node attributes capture historical demand, event proximity, and real-time congestion scores obtained from live traffic APIs. Graph Neural Network (GNN) embeddings are employed to encode spatial-temporal dependencies within the traffic network, which are then used by a Q-learning agent to recommend optimal taxi hotspots. The reward mechanism jointly optimizes passenger waiting time, driver travel distance, and congestion avoidance. Experiments on a simulated Delhi taxi dataset, generated using real geospatial boundaries and historic ride-hailing request patterns, demonstrate that the proposed model reduced passenger waiting time by about 56% and reduced travel distance by 38% compared to baseline stochastic selection. The proposed approach is adaptable to multi-modal transport systems and can be integrated into smart city platforms for real-time urban mobility optimization.

**arXiv ID:** 2601.00607
</details>

<details>
<summary><strong>Reinforcement learning with timed constraints for robotics motion planning</strong> - Zhaoan Wang, Junchao Li, Mahdi Mohammad, Shaoping Xiao - [[pdf]](https://arxiv.org/pdf/2601.00087)</summary>

**Abstract:** Robotic systems operating in dynamic and uncertain environments increasingly require planners that satisfy complex task sequences while adhering to strict temporal constraints. Metric Interval Temporal Logic (MITL) offers a formal and expressive framework for specifying such time-bounded requirements; however, integrating MITL with reinforcement learning (RL) remains challenging due to stochastic dynamics and partial observability. This paper presents a unified automata-based RL framework for synthesizing policies in both Markov Decision Processes (MDPs) and Partially Observable Markov Decision Processes (POMDPs) under MITL specifications. MITL formulas are translated into Timed Limit-Deterministic Generalized Büchi Automata (Timed-LDGBA) and synchronized with the underlying decision process to construct product timed models suitable for Q-learning. A simple yet expressive reward structure enforces temporal correctness while allowing additional performance objectives. The approach is validated in three simulation studies: a $5 \times 5$ grid-world formulated as an MDP, a $10 \times 10$ grid-world formulated as a POMDP, and an office-like service-robot scenario. Results demonstrate that the proposed framework consistently learns policies that satisfy strict time-bounded requirements under stochastic transitions, scales to larger state spaces, and remains effective in partially observable environments, highlighting its potential for reliable robotic planning in time-critical and uncertain settings.

**arXiv ID:** 2601.00087
</details>

<details>
<summary><strong>Reinforcement Learning from Human Feedback</strong> - Nathan Lambert - [[pdf]](https://arxiv.org/pdf/2504.12501)</summary>

**Abstract:** Reinforcement learning from human feedback (RLHF) has become an important technical and storytelling tool to deploy the latest machine learning systems. In this book, we hope to give a gentle introduction to the core methods for people with some level of quantitative background. The book starts with the origins of RLHF -- both in recent literature and in a convergence of disparate fields of science in economics, philosophy, and optimal control. We then set the stage with definitions, problem formulation, data collection, and other common math used in the literature. The core of the book details every optimization stage in using RLHF, from starting with instruction tuning to training a reward model and finally all of rejection sampling, reinforcement learning, and direct alignment algorithms. The book concludes with advanced topics -- understudied research questions in synthetic data and evaluation -- and open questions for the field.

**arXiv ID:** 2504.12501
</details>

<details>
<summary><strong>Collaborative Device-Cloud LLM Inference through Reinforcement Learning</strong> - Wenzhi Fang, Dong-Jun Han, Liangqi Yuan, Christopher Brinton - [[pdf]](https://arxiv.org/pdf/2509.24050)</summary>

**Abstract:** Device-cloud collaboration has emerged as a promising paradigm for deploying large language models (LLMs), combining the efficiency of lightweight on-device inference with the superior performance of powerful cloud LLMs. An essential problem in this scenario lies in deciding whether a given query is best handled locally or delegated to the cloud. Existing approaches typically rely on external routers, implemented as binary classifiers, which often struggle to determine task difficulty from the prompt's surface pattern. To address these limitations, we propose a framework where the on-device LLM makes routing decisions at the end of its solving process, with this capability instilled through post-training. In particular, we formulate a reward maximization problem with carefully designed rewards that encourage effective problem solving and judicious offloading to the cloud. To solve this problem, we develop a group-adaptive policy gradient algorithm, featuring a group-level policy gradient, designed to yield an unbiased gradient estimator of the reward, and adaptive prompt filtering, developed to enforce the constraint on cloud LLM usage. Extensive experiments across models and benchmarks show that the proposed methodology consistently outperforms existing baselines and significantly narrows the gap to full cloud LLM performance.

**arXiv ID:** 2509.24050
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
