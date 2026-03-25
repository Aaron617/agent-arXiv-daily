# Agent arXiv Daily

**Last Updated:** 2026-03-25 03:16:32

**Total Papers:** 43

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (1 papers)</h2></summary>

<details>
<summary><strong>Learning When to Act: Interval-Aware Reinforcement Learning with Predictive Temporal Structure</strong> - Davide Di Gioia - [[pdf]](https://arxiv.org/pdf/2603.22384)</summary>

**Abstract:** Autonomous agents operating in continuous environments must decide not only what to do, but when to act. We introduce a lightweight adaptive temporal control system that learns the optimal interval between cognitive ticks from experience, replacing ad hoc biologically inspired timers with a principled learned policy. The policy state is augmented with a predictive hyperbolic spread signal (a "curvature signal" shorthand) derived from hyperbolic geometry: the mean pairwise Poincare distance among n sampled futures embedded in the Poincare ball. High spread indicates a branching, uncertain future and drives the agent to act sooner; low spread signals predictability and permits longer rest intervals. We further propose an interval-aware reward that explicitly penalises inefficiency relative to the chosen wait time, correcting a systematic credit-assignment failure of naive outcome-based rewards in timing problems. We additionally introduce a joint spatio-temporal embedding (ATCPG-ST) that concatenates independently normalised state and position projections in the Poincare ball; spatial trajectory divergence provides an independent timing signal unavailable to the state-only variant (ATCPG-SO). This extension raises mean hyperbolic spread (kappa) from 1.88 to 3.37 and yields a further 5.8 percent efficiency gain over the state-only baseline. Ablation experiments across five random seeds demonstrate that (i) learning is the dominant efficiency factor (54.8 percent over no-learning), (ii) hyperbolic spread provides significant complementary gain (26.2 percent over geometry-free control), (iii) the combined system achieves 22.8 percent efficiency over the fixed-interval baseline, and (iv) adding spatial position information to the spread embedding yields an additional 5.8 percent.

**arXiv ID:** 2603.22384
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (11 papers)</h2></summary>

<details>
<summary><strong>Separating Diagnosis from Control: Auditable Policy Adaptation in Agent-Based Simulations with LLM-Based Diagnostics</strong> - Shaoxin Zhong, Yuchen Su, Michael Witbrock - [[pdf]](https://arxiv.org/pdf/2603.22904)</summary>

**Abstract:** Mitigating elderly loneliness requires policy interventions that achieve both adaptability and auditability. Existing methods struggle to reconcile these objectives: traditional agent-based models suffer from static rigidity, while direct large language model (LLM) controllers lack essential traceability. This work proposes a three-layer framework that separates diagnosis from control to achieve both properties simultaneously. LLMs operate strictly as diagnostic instruments that assess population state and generate structured risk evaluations, while deterministic formulas with explicit bounds translate these assessments into traceable parameter updates. This separation ensures that every policy decision can be attributed to inspectable rules while maintaining adaptive response to emergent needs. We validate the framework through systematic ablation across five experimental conditions in elderly care simulation. Results demonstrate that explicit control rules outperform end-to-end black-box LLM approaches by 11.7\% while preserving full auditability, confirming that transparency need not compromise adaptive performance.

**arXiv ID:** 2603.22904
</details>

<details>
<summary><strong>PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments</strong> - Shuochen Liu, Junyi Zhu, Long Shu, Junda Lin, Yuhao Chen, Haotian Zhang, Chao Zhang, Derong Xu, Jia Li, Bo Tang, Zhiyu Li, Feiyu Xiong, Enhong Chen, Tong Xu - [[pdf]](https://arxiv.org/pdf/2603.23231)</summary>

**Abstract:** Empowering large language models with long-term memory is crucial for building agents that adapt to users' evolving needs. However, prior evaluations typically interleave preference-related dialogues with irrelevant conversations, reducing the task to needle-in-a-haystack retrieval while ignoring relationships between events that drive the evolution of user preferences. Such settings overlook a fundamental characteristic of real-world personalization: preferences emerge gradually and accumulate across interactions within noisy contexts. To bridge this gap, we introduce PERMA, a benchmark designed to evaluate persona consistency over time beyond static preference recall. Additionally, we incorporate (1) text variability and (2) linguistic alignment to simulate erratic user inputs and individual idiolects in real-world data. PERMA consists of temporally ordered interaction events spanning multiple sessions and domains, with preference-related queries inserted over time. We design both multiple-choice and interactive tasks to probe the model's understanding of persona along the interaction timeline. Experiments demonstrate that by linking related interactions, advanced memory systems can extract more precise preferences and reduce token consumption, outperforming traditional semantic retrieval of raw dialogues. Nevertheless, they still struggle to maintain a coherent persona across temporal depth and cross-domain interference, highlighting the need for more robust personalized memory management in agents. Our code and data are open-sourced at this https URL.

**arXiv ID:** 2603.23231
</details>

<details>
<summary><strong>MemCollab: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation</strong> - Yurui Chang, Yiran Wu, Qingyun Wu, Lu Lin - [[pdf]](https://arxiv.org/pdf/2603.23234)</summary>

**Abstract:** Large language model (LLM)-based agents rely on memory mechanisms to reuse knowledge from past problem-solving experiences. Existing approaches typically construct memory in a per-agent manner, tightly coupling stored knowledge to a single model's reasoning style. In modern deployments with heterogeneous agents, a natural question arises: can a single memory system be shared across different models? We found that naively transferring memory between agents often degrades performance, as such memory entangles task-relevant knowledge with agent-specific biases. To address this challenge, we propose MemCollab, a collaborative memory framework that constructs agent-agnostic memory by contrasting reasoning trajectories generated by different agents on the same task. This contrastive process distills abstract reasoning constraints that capture shared task-level invariants while suppressing agent-specific artifacts. We further introduce a task-aware retrieval mechanism that conditions memory access on task category, ensuring that only relevant constraints are used at inference time. Experiments on mathematical reasoning and code generation benchmarks demonstrate that MemCollab consistently improves both accuracy and inference-time efficiency across diverse agents, including cross-modal-family settings. Our results show that the collaboratively constructed memory can function as a shared reasoning resource for diverse LLM-based agents.

**arXiv ID:** 2603.23234
</details>

<details>
<summary><strong>Beyond Preset Identities: How Agents Form Stances and Boundaries in Generative Societies</strong> - Hanzhong Zhang, Siyang Song, Jindong Wang - [[pdf]](https://arxiv.org/pdf/2603.23406)</summary>

**Abstract:** While large language models simulate social behaviors, their capacity for stable stance formation and identity negotiation during complex interventions remains unclear. To overcome the limitations of static evaluations, this paper proposes a novel mixed-methods framework combining computational virtual ethnography with quantitative socio-cognitive profiling. By embedding human researchers into generative multiagent communities, controlled discursive interventions are conducted to trace the evolution of collective cognition. To rigorously measure how agents internalize and react to these specific interventions, this paper formalizes three new metrics: Innate Value Bias (IVB), Persuasion Sensitivity, and Trust-Action Decoupling (TAD). Across multiple representative models, agents exhibit endogenous stances that override preset identities, consistently demonstrating an innate progressive bias (IVB > 0). When aligned with these stances, rational persuasion successfully shifts 90% of neutral agents while maintaining high trust. In contrast, conflicting emotional provocations induce a paradoxical 40.0% TAD rate in advanced models, which hypocritically alter stances despite reporting low trust. Smaller models contrastingly maintain a 0% TAD rate, strictly requiring trust for behavioral shifts. Furthermore, guided by shared stances, agents use language interactions to actively dismantle assigned power hierarchies and reconstruct self organized community boundaries. These findings expose the fragility of static prompt engineering, providing a methodological and quantitative foundation for dynamic alignment in human-agent hybrid societies. The official code is available at: this https URL

**arXiv ID:** 2603.23406
</details>

<details>
<summary><strong>Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos</strong> - Shoubin Yu, Lei Shu, Antoine Yang, Yao Fu, Srinivas Sunkara, Maria Wang, Jindong Chen, Mohit Bansal, Boqing Gong - [[pdf]](https://arxiv.org/pdf/2603.22529)</summary>

**Abstract:** Multimodal AI agents are increasingly automating complex real-world workflows that involve online web execution. However, current web-agent benchmarks suffer from a critical limitation: they focus entirely on web-based interaction and perception, lacking grounding in the user's real-world physical surroundings. This limitation prevents evaluation in crucial scenarios, such as when an agent must use egocentric visual perception (e.g., via AR glasses) to recognize an object in the user's surroundings and then complete a related task online. To address this gap, we introduce Ego2Web, the first benchmark designed to bridge egocentric video perception and web agent execution. Ego2Web pairs real-world first-person video recordings with web tasks that require visual understanding, web task planning, and interaction in an online environment for successful completion. We utilize an automatic data-generation pipeline combined with human verification and refinement to curate well-constructed, high-quality video-task pairs across diverse web task types, including e-commerce, media retrieval, knowledge lookup, etc. To facilitate accurate and scalable evaluation for our benchmark, we also develop a novel LLM-as-a-Judge automatic evaluation method, Ego2WebJudge, which achieves approximately 84% agreement with human judgment, substantially higher than existing evaluation methods. Experiments with diverse SoTA agents on our Ego2Web show that their performance is weak, with substantial headroom across all task categories. We also conduct a comprehensive ablation study on task design, highlighting the necessity of accurate video understanding in the proposed task and the limitations of current agents. We hope Ego2Web can be a critical new resource for developing truly capable AI assistants that can seamlessly see, understand, and act across the physical and digital worlds.

**arXiv ID:** 2603.22529
</details>

<details>
<summary><strong>STRIATUM-CTF: A Protocol-Driven Agentic Framework for General-Purpose CTF Solving</strong> - James Hugglestone, Samuel Jacob Chacko, Dawson Stoller, Ryan Schmidt, Xiuwen Liu - [[pdf]](https://arxiv.org/pdf/2603.22577)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated potential in code generation, yet they struggle with the multi-step, stateful reasoning required for offensive cybersecurity operations. Existing research often relies on static benchmarks that fail to capture the dynamic nature of real-world vulnerabilities. In this work, we introduce STRIATUM-CTF (A Search-based Test-time Reasoning Inference Agent for Tactical Utility Maximization in Cybersecurity), a modular agentic framework built upon the Model Context Protocol (MCP). By standardizing tool interfaces for system introspection, decompilation, and runtime debugging, STRIATUM-CTF enables the agent to maintain a coherent context window across extended exploit trajectories.
We validate this approach not merely on synthetic datasets, but in a live competitive environment. Our system participated in a university-hosted Capture-the-Flag (CTF) competition in late 2025, where it operated autonomously to identify and exploit vulnerabilities in real-time. STRIATUM-CTF secured First Place, outperforming 21 human teams and demonstrating strong adaptability in a dynamic problem-solving setting. We analyze the agent's decision-making logs to show how MCP-based tool abstraction significantly reduces hallucination compared to naive prompting strategies. These results suggest that standardized context protocols are a critical path toward robust autonomous cyber-reasoning systems.

**arXiv ID:** 2603.22577
</details>

<details>
<summary><strong>PhotoAgent: A Robotic Photographer with Spatial and Aesthetic Understanding</strong> - Lirong Che, Zhenfeng Gan, Yanbo Chen, Junbo Tan, Xueqian Wang - [[pdf]](https://arxiv.org/pdf/2603.22796)</summary>

**Abstract:** Embodied agents for creative tasks like photography must bridge the semantic gap between high-level language commands and geometric control. We introduce PhotoAgent, an agent that achieves this by integrating Large Multimodal Models (LMMs) reasoning with a novel control paradigm. PhotoAgent first translates subjective aesthetic goals into solvable geometric constraints via LMM-driven, chain-of-thought (CoT) reasoning, allowing an analytical solver to compute a high-quality initial viewpoint. This initial pose is then iteratively refined through visual reflection within a photorealistic internal world model built with 3D Gaussian Splatting (3DGS). This ``mental simulation'' replaces costly and slow physical trial-and-error, enabling rapid convergence to aesthetically superior results. Evaluations confirm that PhotoAgent excels in spatial reasoning and achieves superior final image quality.

**arXiv ID:** 2603.22796
</details>

<details>
<summary><strong>AgentRAE: Remote Action Execution through Notification-based Visual Backdoors against Screenshots-based Mobile GUI Agents</strong> - Yutao Luo, Haotian Zhu, Shuchao Pang, Zhigang Lu, Tian Dong, Yongbin Zhou, Minhui Xue - [[pdf]](https://arxiv.org/pdf/2603.23007)</summary>

**Abstract:** The rapid adoption of mobile graphical user interface (GUI) agents, which autonomously control applications and operating systems (OS), exposes new system-level attack surfaces. Existing backdoors against web GUI agents and general GenAI models rely on environmental injection or deceptive pop-ups to mislead the agent operation. However, these techniques do not work on screenshots-based mobile GUI agents due to the challenges of restricted trigger design spaces, OS background interference, and conflicts in multiple trigger-action mappings. We propose AgentRAE, a novel backdoor attack capable of inducing Remote Action Execution in mobile GUI agents using visually natural triggers (e.g., benign app icons in notifications). To address the underfitting caused by natural triggers and achieve accurate multi-target action redirection, we design a novel two-stage pipeline that first enhances the agent's sensitivity to subtle iconographic differences via contrastive learning, and then associates each trigger with a specific mobile GUI agent action through a backdoor post-training. Our extensive evaluation reveals that the proposed backdoor preserves clean performance with an attack success rate of over 90% across ten mobile operations. Furthermore, it is hard to visibly detect the benign-looking triggers and circumvents eight representative state-of-the-art defenses. These results expose an overlooked backdoor vector in mobile GUI agents, underscoring the need for defenses that scrutinize notification-conditioned behaviors and internal agent representations.

**arXiv ID:** 2603.23007
</details>

<details>
<summary><strong>Code Review Agent Benchmark</strong> - Yuntong Zhang, Zhiyuan Pan, Imam Nur Bani Yusuf, Haifeng Ruan, Ridwan Shariffdeen, Abhik Roychoudhury - [[pdf]](https://arxiv.org/pdf/2603.23448)</summary>

**Abstract:** Software engineering agents have shown significant promise in writing code. As AI agents permeate code writing, and generate huge volumes of code automatically -- the matter of code quality comes front and centre. As the automatically generated code gets integrated into huge code-bases -- the issue of code review and broadly quality assurance becomes important. In this paper, we take a fresh look at the problem and curate a code review dataset for AI agents to work with. Our dataset called c-CRAB (pronounced see-crab) can evaluate agents for code review tasks. Specifically given a pull-request (which could be coming from code generation agents or humans), if a code review agent produces a review, our evaluation framework can asses the reviewing capability of the code review agents. Our evaluation framework is used to evaluate the state of the art today -- the open-source PR-agent, as well as commercial code review agents from Devin, Claude Code, and Codex.
Our c-CRAB dataset is systematically constructed from human reviews -- given a human review of a pull request instance we generate corresponding tests to evaluate the code review agent generated reviews. Such a benchmark construction gives us several insights. Firstly, the existing review agents taken together can solve only around 40% of the c-CRAB tasks, indicating the potential to close this gap by future research. Secondly, we observe that the agent reviews often consider different aspects from the human reviews -- indicating the potential for human-agent collaboration for code review that could be deployed in future software teams. Last but not the least, the agent generated tests from our data-set act as a held out test-suite and hence quality gate for agent generated reviews. What this will mean for future collaboration of code generation agents, test generation agents and code review agents -- remains to be investigated.

**arXiv ID:** 2603.23448
</details>

<details>
<summary><strong>Towards Self-Evolving Benchmarks: Synthesizing Agent Trajectories via Test-Time Exploration under Validate-by-Reproduce Paradigm</strong> - Dadi Guo, Tianyi Zhou, Dongrui Liu, Chen Qian, Qihan Ren, Shuai Shao, Zhiyuan Fan, Yi R. Fung, Kun Wang, Linfeng Zhang, Jing Shao - [[pdf]](https://arxiv.org/pdf/2510.00415)</summary>

**Abstract:** Recent advances in large language models (LLMs) and agent system designs have empowered agents with unprecedented levels of capability. However, existing agent benchmarks are showing a trend of rapid ceiling-hitting by newly developed agents, making it difficult to meet the demands for evaluating agent abilities. To address this problem, we propose the Trajectory-based Validated-by-Reproducing Agent-benchmark Complexity Evolution (TRACE) framework. This framework takes an original task from an existing benchmark and encourages agents to freely explore and evolve it into a new task with higher difficulty while recording validatable agent trajectories. The framework proceeds in three stages: (1) evolutionary proposal mining, which provides task evolution proposals through preliminary exploration and divergent thinking; (2) problem formation and free exploration, where proposals are conceptualized into feasible problem candidates and the agents then explore them freely while recording their execution trajectories; and (3) multi-level validation, which ensures that the evolved tasks are accompanied by validatable and reproducible trajectories. Experiments on the GAIA benchmark demonstrate that the TRACE framework consistently enhances task complexity while improving the reliability of correctness through validatable execution trajectories. In addition, our framework can successfully adapt to and improve reasoning datasets represented by AIME-2024. This work marks a paradigm shift from static, manually curated benchmarks to dynamic, self-evolving evaluation systems, providing a sustainable and challenging runway for agent development

**arXiv ID:** 2510.00415
</details>

<details>
<summary><strong>BuilderBench: The Building Blocks of Intelligent Agents</strong> - Raj Ghugare, Roger Creus Castanyer, Catherine Ji, Kathryn Wantlin, Jin Schofield, Karthik Narasimhan, Benjamin Eysenbach - [[pdf]](https://arxiv.org/pdf/2510.06288)</summary>

**Abstract:** Today's AI models learn primarily through mimicry and refining, so it is not surprising that they struggle to solve problems beyond the limits set by existing data. To solve novel problems, agents should acquire skills for exploring and learning through experience. Finding a scalable learning mechanism for developing agents that learn through interaction remains a major open problem. In this work, we introduce BuilderBench, a benchmark to accelerate research into agent pre-training that centers open-ended exploration. BuilderBench requires agents to learn how to build any structure using blocks. BuilderBench is equipped with $(1)$ a hardware accelerated simulator of a robotic agent interacting with various physical blocks, and $(2)$ a task-suite with over 42 diverse target structures that are carefully curated to test an understanding of physics, mathematics, and long-horizon planning. During training, agents have to explore and learn general principles about the environment without any external supervision. During evaluation, agents have to build the unseen target structures from the task suite. Solving these tasks requires a sort of \emph{embodied reasoning} that is not reflected in words but rather in actions, experimenting with different strategies and piecing them together. Our experiments show that many of these tasks challenge the current iteration of algorithms. Hence, we also provide a ``training wheels'' protocol, in which agents are trained and evaluated to build a single target structure from the task suite. Finally, we provide single-file implementations of six different algorithms as a reference point for researchers.

**arXiv ID:** 2510.06288
</details>

</details>

<details open>
<summary><h2>LLM Agents (8 papers)</h2></summary>

<details>
<summary><strong>From Static Templates to Dynamic Runtime Graphs: A Survey of Workflow Optimization for LLM Agents</strong> - Ling Yue, Kushal Raj Bhandari, Ching-Yun Ko, Dhaval Patel, Shuxin Lin, Nianjun Zhou, Jianxi Gao, Pin-Yu Chen, Shaowu Pan - [[pdf]](https://arxiv.org/pdf/2603.22386)</summary>

**Abstract:** Large language model (LLM)-based systems are becoming increasingly popular for solving tasks by constructing executable workflows that interleave LLM calls, information retrieval, tool use, code execution, memory updates, and verification. This survey reviews recent methods for designing and optimizing such workflows, which we treat as agentic computation graphs (ACGs). We organize the literature based on when workflow structure is determined, where structure refers to which components or agents are present, how they depend on each other, and how information flows between them. This lens distinguishes static methods, which fix a reusable workflow scaffold before deployment, from dynamic methods, which select, generate, or revise the workflow for a particular run before or during execution. We further organize prior work along three dimensions: when structure is determined, what part of the workflow is optimized, and which evaluation signals guide optimization (e.g., task metrics, verifier signals, preferences, or trace-derived feedback). We also distinguish reusable workflow templates, run-specific realized graphs, and execution traces, separating reusable design choices from the structures actually deployed in a given run and from realized runtime behavior. Finally, we outline a structure-aware evaluation perspective that complements downstream task metrics with graph-level properties, execution cost, robustness, and structural variation across inputs. Our goal is to provide a clear vocabulary, a unified framework for positioning new methods, a more comparable view of existing body of literature, and a more reproducible evaluation standard for future work in workflow optimizations for LLM agents.

**arXiv ID:** 2603.22386
</details>

<details>
<summary><strong>Can LLM Agents Generate Real-World Evidence? Evaluating Observational Studies in Medical Databases</strong> - Dubai Li, Yuxiang He, Yan Hu, Yu Tian, Jingsong Li - [[pdf]](https://arxiv.org/pdf/2603.22767)</summary>

**Abstract:** Observational studies can yield clinically actionable evidence at scale, but executing them on real-world databases is open-ended and requires coherent decisions across cohort construction, analysis, and reporting. Prior evaluations of LLM agents emphasize isolated steps or single answers, missing the integrity and internal structure of the resulting evidence bundle. To address this gap, we introduce RWE-bench, a benchmark grounded in MIMIC-IV and derived from peer-reviewed observational studies. Each task provides the corresponding study protocol as the reference standard, requiring agents to execute experiments in a real database and iteratively generate tree-structured evidence bundles. We evaluate six LLMs (three open-source, three closed-source) under three agent scaffolds using both question-level correctness and end-to-end task metrics. Across 162 tasks, task success is low: the best agent reaches 39.9%, and the best open-source model reaches 30.4%. Agent scaffolds also matter substantially, causing over 30% variation in performance metrics. Furthermore, we implement an automated cohort evaluation method to rapidly localize errors and identify agent failure modes. Overall, the results highlight persistent limitations in agents' ability to produce end-to-end evidence bundles, and efficient validation remains an important direction for future work. Code and data are available at this https URL.

**arXiv ID:** 2603.22767
</details>

<details>
<summary><strong>T-MAP: Red-Teaming LLM Agents with Trajectory-aware Evolutionary Search</strong> - Hyomin Lee, Sangwoo Park, Yumin Choi, Sohyun An, Seanie Lee, Sung Ju Hwang - [[pdf]](https://arxiv.org/pdf/2603.22341)</summary>

**Abstract:** While prior red-teaming efforts have focused on eliciting harmful text outputs from large language models (LLMs), such approaches fail to capture agent-specific vulnerabilities that emerge through multi-step tool execution, particularly in rapidly growing ecosystems such as the Model Context Protocol (MCP). To address this gap, we propose a trajectory-aware evolutionary search method, T-MAP, which leverages execution trajectories to guide the discovery of adversarial prompts. Our approach enables the automatic generation of attacks that not only bypass safety guardrails but also reliably realize harmful objectives through actual tool interactions. Empirical evaluations across diverse MCP environments demonstrate that T-MAP substantially outperforms baselines in attack realization rate (ARR) and remains effective against frontier models, including GPT-5.2, Gemini-3-Pro, Qwen3.5, and GLM-5, thereby revealing previously underexplored vulnerabilities in autonomous LLM agents.

**arXiv ID:** 2603.22341
</details>

<details>
<summary><strong>Reasoner-Executor-Synthesizer: Scalable Agentic Architecture with Static O(1) Context Window</strong> - Ivan Dobrovolskyi - [[pdf]](https://arxiv.org/pdf/2603.22367)</summary>

**Abstract:** Large Language Models (LLMs) deployed as autonomous agents commonly use Retrieval-Augmented Generation (RAG), feeding retrieved documents into the context window, which creates two problems: the risk of hallucination grows with context length, and token cost scales linearly with dataset size. We propose the Reasoner-Executor-Synthesizer (RES) architecture, a three-layer design that strictly separates intent parsing (Reasoner), deterministic data retrieval and aggregation (Executor), and narrative generation (Synthesizer). The Executor uses zero LLM tokens and passes only fixed-size statistical summaries to the Synthesizer. We formally prove that RES achieves O(1) token complexity with respect to dataset size, and validate this on ScholarSearch, a scholarly research assistant backed by the Crossref API (130M+ articles). Across 100 benchmark runs, RES achieves a mean token cost of 1,574 tokens regardless of whether the dataset contains 42,000 or 16.3 million articles. The architecture eliminates data hallucination by construction: the LLM never sees raw records. KEYWORDS LLM agents; agentic architecture; hallucination elimination; token optimization; context window; retrieval-augmented generation; deterministic execution; scholarly metadata; Crossref API; O(1) complexity.

**arXiv ID:** 2603.22367
</details>

<details>
<summary><strong>AI Co-Scientist for Ranking: Discovering Novel Search Ranking Models alongside LLM-based AI Agents with Cloud Computing Access</strong> - Liwei Wu, Cho-Jui Hsieh - [[pdf]](https://arxiv.org/pdf/2603.22376)</summary>

**Abstract:** Recent advances in AI agents for software engineering and scientific discovery have demonstrated remarkable capabilities, yet their application to developing novel ranking models in commercial search engines remains unexplored. In this paper, we present an AI Co-Scientist framework that automates the full search ranking research pipeline: from idea generation to code implementation and GPU training job scheduling with expert in the loop. Our approach strategically employs single-LLM agents for routine tasks while leveraging multi-LLM consensus agents (GPT 5.2, Gemini Pro 3, and Claude Opus 4.5) for challenging phases such as results analysis and idea generation. To our knowledge, this is the first study in the ranking community to utilize an AI Co-Scientist framework for algorithmic research. We demonstrate that this framework discovered a novel technique for handling sequence features, with all model enhancements produced automatically, yielding substantial offline performance improvements. Our findings suggest that AI systems can discover ranking architectures comparable to those developed by human experts while significantly reducing routine research workloads.

**arXiv ID:** 2603.22376
</details>

<details>
<summary><strong>Agent Audit: A Security Analysis System for LLM Agent Applications</strong> - Haiyue Zhang, Yi Nian, Yue Zhao - [[pdf]](https://arxiv.org/pdf/2603.22853)</summary>

**Abstract:** What should a developer inspect before deploying an LLM agent: the model, the tool code, the deployment configuration, or all three? In practice, many security failures in agent systems arise not from model weights alone, but from the surrounding software stack: tool functions that pass untrusted inputs to dangerous operations, exposed credentials in deployment artifacts, and over-privileged Model Context Protocol (MCP) configurations.
We present Agent Audit, a security analysis system for LLM agent applications. Agent Audit analyzes Python agent code and deployment artifacts through an agent-aware pipeline that combines dataflow analysis, credential detection, structured configuration parsing, and privilege-risk checks. The system reports findings in terminal, JSON, and SARIF formats, enabling direct integration with local development workflows and CI/CD pipelines. On a benchmark of 22 samples with 42 annotated vulnerabilities, Agent Audit detects 40 vulnerabilities with 6 false positives, substantially improving recall over common SAST baselines while maintaining sub-second scan times. Agent Audit is open source and installable via pip, making security auditing accessible for agent systems.
In the live demonstration, attendees scan vulnerable agent repositories and observe how Agent Audit identifies security risks in tool functions, prompts, and more. Findings are linked to source locations and configuration paths, and can be exported into VS Code and GitHub Code Scanning for interactive inspection.

**arXiv ID:** 2603.22853
</details>

<details>
<summary><strong>Agent-Sentry: Bounding LLM Agents via Execution Provenance</strong> - Rohan Sequeira, Stavros Damianakis, Umar Iqbal, Konstantinos Psounis - [[pdf]](https://arxiv.org/pdf/2603.22868)</summary>

**Abstract:** Agentic computing systems, which autonomously spawn new functionalities based on natural language instructions, are becoming increasingly prevalent. While immensely capable, these systems raise serious security, privacy, and safety concerns. Fundamentally, the full set of functionalities offered by these systems, combined with their probabilistic execution flows, is not known beforehand. Given this lack of characterization, it is non-trivial to validate whether a system has successfully carried out the user's intended task or instead executed irrelevant actions, potentially as a consequence of compromise. In this paper, we propose Agent-Sentry, a framework that attempts to bound agentic systems to address this problem. Our key insight is that agentic systems are designed for specific use cases and therefore need not expose unbounded or unspecified functionalities. Once bounded, these systems become easier to scrutinize. Agent-Sentry operationalizes this insight by uncovering frequent functionalities offered by an agentic system, along with their execution traces, to construct behavioral bounds. It then learns a policy from these traces and blocks tool calls that deviate from learned behaviors or that misalign with user intent. Our evaluation shows that Agent-Sentry helps prevent over 90\% of attacks that attempt to trigger out-of-bounds executions, while preserving up to 98\% of system utility.

**arXiv ID:** 2603.22868
</details>

<details>
<summary><strong>Rethinking the Role of Entropy in Optimizing Tool-Use Behaviors for Large Language Model Agents</strong> - Zeping Li, Hongru Wang, Yiwen Zhao, Guanhua Chen, Yixia Li, Keyang Chen, Yixin Cao, Guangnan Ye, Hongfeng Chai, Zhenfei Yin - [[pdf]](https://arxiv.org/pdf/2602.02050)</summary>

**Abstract:** Tool-using agents based on Large Language Models (LLMs) excel in tasks such as mathematical reasoning and multi-hop question answering. However, in long trajectories, agents often trigger excessive and low-quality tool calls, increasing latency and degrading inference performance, making managing tool-use behavior challenging. In this work, we conduct entropy-based pilot experiments and observe a strong positive correlation between entropy reduction and high-quality tool calls. Building on this finding, we propose using entropy reduction as a supervisory signal and design two reward strategies to address the differing needs of optimizing tool-use behavior. Sparse outcome rewards provide coarse, trajectory-level guidance to improve efficiency, while dense process rewards offer fine-grained supervision to enhance performance. Experiments across diverse domains show that both reward designs improve tool-use behavior: the former reduces tool calls by 72.07% compared to the average of baselines, while the latter improves performance by 22.27%. These results position entropy reduction as a key mechanism for enhancing tool-use behavior, enabling agents to be more adaptive in real-world applications.

**arXiv ID:** 2602.02050
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (8 papers)</h2></summary>

<details>
<summary><strong>Benchmarking Multi-Agent LLM Architectures for Financial Document Processing: A Comparative Study of Orchestration Patterns, Cost-Accuracy Tradeoffs and Production Scaling Strategies</strong> - Siddhant Kulkarni, Yukta Kulkarni - [[pdf]](https://arxiv.org/pdf/2603.22651)</summary>

**Abstract:** The adoption of large language models (LLMs) for structured information extraction from financial documents has accelerated rapidly, yet production deployments face fundamental architectural decisions with limited empirical guidance. We present a systematic benchmark comparing four multi-agent orchestration architectures: sequential pipeline, parallel fan-out with merge, hierarchical supervisor-worker and reflexive self-correcting loop. These are evaluated across five frontier and open-weight LLMs on a corpus of 10,000 SEC filings (10-K, 10-Q and 8-K forms). Our evaluation spans 25 extraction field types covering governance structures, executive compensation and financial metrics, measured along five axes: field-level F1, document-level accuracy, end-to-end latency, cost per document and token efficiency. We find that reflexive architectures achieve the highest field-level F1 (0.943) but at 2.3x the cost of sequential baselines, while hierarchical architectures occupy the most favorable position on the cost-accuracy Pareto frontier (F1 0.921 at 1.4x cost). We further present ablation studies on semantic caching, model routing and adaptive retry strategies, demonstrating that hybrid configurations can recover 89\% of the reflexive architecture's accuracy gains at only 1.15x baseline cost. Our scaling analysis from 1K to 100K documents per day reveals non-obvious throughput-accuracy degradation curves that inform capacity planning. These findings provide actionable guidance for practitioners deploying multi-agent LLM systems in regulated financial environments.

**arXiv ID:** 2603.22651
</details>

<details>
<summary><strong>STEM Agent: A Self-Adapting, Tool-Enabled, Extensible Architecture for Multi-Protocol AI Agent Systems</strong> - Alfred Shen, Aaron Shen - [[pdf]](https://arxiv.org/pdf/2603.22359)</summary>

**Abstract:** Current AI agent frameworks commit early to a single interaction protocol, a fixed tool integration strategy, and static user models, limiting their deployment across diverse interaction paradigms. To address these constraints, we introduce STEM Agent (Self-adapting, Tool-enabled, Extensible, Multi-agent), a modular architecture inspired by biological pluripotency in which an undifferentiated agent core differentiates into specialized protocol handlers, tool bindings, and memory subsystems that compose into a fully functioning AI system. The framework unifies five interoperability protocols (A2A, AG-UI, A2UI, UCP, and AP2) behind a single gateway, introduces a Caller Profiler that continuously learns user preferences across more than twenty behavioral dimensions, externalizes all domain capabilities through the Model Context Protocol (MCP), and implements a biologically inspired skills acquisition system in which recurring interaction patterns crystallize into reusable agent skills through a maturation lifecycle analogous to cell differentiation. Complementing these capabilities, the memory system incorporates consolidation mechanisms, including episodic pruning, semantic deduplication, and pattern extraction, designed for sub-linear growth under sustained interaction. A comprehensive 413-test suite validates protocol handler behavior and component integration across all five architectural layers, completing in under three seconds.

**arXiv ID:** 2603.22359
</details>

<details>
<summary><strong>ABSTRAL: Automatic Design of Multi-Agent Systems Through Iterative Refinement and Topology Optimization</strong> - Weijia Song, Jiashu Yue, Zhe Pang - [[pdf]](https://arxiv.org/pdf/2603.22791)</summary>

**Abstract:** How should multi-agent systems be designed, and can that design knowledge be captured in a form that is inspectable, revisable, and transferable? We introduce ABSTRAL, a framework that treats MAS architecture as an evolving natural-language document, an artifact refined through contrastive trace analysis. Three findings emerge. First, we provide a precise measurement of the multi-agent coordination tax: under fixed turn budgets, ensembles achieve only 26% turn efficiency, with 66% of tasks exhausting the limit, yet still improve over single-agent baselines by discovering parallelizable task decompositions. Second, design knowledge encoded in documents transfers: topology reasoning and role templates learned on one domain provide a head start on new domains, with transferred seeds matching coldstart iteration 3 performance in a single iteration. Third, contrastive trace analysis discovers specialist roles absent from any initial design, a capability no prior system demonstrates. On SOPBench (134 bank tasks, deterministic oracle), ABSTRAL reaches 70% validation / 65.96% test pass rate with a GPT-4o backbone. We release the converged documents as inspectable design rationale.

**arXiv ID:** 2603.22791
</details>

<details>
<summary><strong>Empirical Comparison of Agent Communication Protocols for Task Orchestration</strong> - Ivan Dobrovolskyi - [[pdf]](https://arxiv.org/pdf/2603.22823)</summary>

**Abstract:** Context. Nowadays, artificial intelligence agent systems are transforming from single-tool interactions to complex multi-agent orchestrations. As a result, two competing communication protocols have emerged: a tool integration protocol that standardizes how agents invoke external tools, and an inter-agent delegation protocol that enables autonomous agents to discover and delegate tasks to one another. Despite widespread industry adoption by dozens of enterprise partners, no empirical comparison of these protocols exists in the literature. Objective. The goal of this work is to develop the first systematic benchmark comparing tool-integration-only, multi-agent delegation, and hybrid architectures across standardized queries at three complexity levels, and to quantify the trade-offs in response time, context window consumption, monetary cost, error recovery, and implementation complexity.

**arXiv ID:** 2603.22823
</details>

<details>
<summary><strong>CoMaTrack: Competitive Multi-Agent Game-Theoretic Tracking with Vision-Language-Action Models</strong> - Youzhi Liu, Li Gao, Liu Liu, Mingyang Lv, Yang Cai - [[pdf]](https://arxiv.org/pdf/2603.22846)</summary>

**Abstract:** Embodied Visual Tracking (EVT), a core dynamic task in embodied intelligence, requires an agent to precisely follow a language-specified target. Yet most existing methods rely on single-agent imitation learning, suffering from costly expert data and limited generalization due to static training environments. Inspired by competition-driven capability evolution, we propose CoMaTrack, a competitive game-theoretic multi-agent reinforcement learning framework that trains agents in a dynamic adversarial setting with competitive subtasks, yielding stronger adaptive planning and interference-resilient strategies. We further introduce CoMaTrack-Bench, the first benchmark for competitive EVT, featuring game scenarios between a tracker and adaptive opponents across diverse environments and instructions, enabling standardized robustness evaluation under active adversarial interactions. Experiments show that CoMaTrack achieves state-of-the-art results on both standard benchmarks and CoMaTrack-Bench. Notably, a 3B VLM trained with our framework surpasses previous single-agent imitation learning methods based on 7B models on the challenging EVT-Bench, achieving 92.1% in STT, 74.2% in DT, and 57.5% in AT. The benchmark code will be available at this https URL

**arXiv ID:** 2603.22846
</details>

<details>
<summary><strong>A Multimodal Framework for Human-Multi-Agent Interaction</strong> - Shaid Hasan, Breenice Lee, Sujan Sarker, Tariq Iqbal - [[pdf]](https://arxiv.org/pdf/2603.23271)</summary>

**Abstract:** Human-robot interaction is increasingly moving toward multi-robot, socially grounded environments. Existing systems struggle to integrate multimodal perception, embodied expression, and coordinated decision-making in a unified framework. This limits natural and scalable interaction in shared physical spaces. We address this gap by introducing a multimodal framework for human-multi-agent interaction in which each robot operates as an autonomous cognitive agent with integrated multimodal perception and Large Language Model (LLM)-driven planning grounded in embodiment. At the team level, a centralized coordination mechanism regulates turn-taking and agent participation to prevent overlapping speech and conflicting actions. Implemented on two humanoid robots, our framework enables coherent multi-agent interaction through interaction policies that combine speech, gesture, gaze, and locomotion. Representative interaction runs demonstrate coordinated multimodal reasoning across agents and grounded embodied responses. Future work will focus on larger-scale user studies and deeper exploration of socially grounded multi-agent interaction dynamics.

**arXiv ID:** 2603.23271
</details>

<details>
<summary><strong>Planning over MAPF Agent Dependencies via Multi-Dependency PIBT</strong> - Zixiang Jiang, Yulun Zhang, Rishi Veerapaneni, Jiaoyang Li - [[pdf]](https://arxiv.org/pdf/2603.23405)</summary>

**Abstract:** Modern Multi-Agent Path Finding (MAPF) algorithms must plan for hundreds to thousands of agents in congested environments within a second, requiring highly efficient algorithms. Priority Inheritance with Backtracking (PIBT) is a popular algorithm capable of effectively planning in such situations. However, PIBT is constrained by its rule-based planning procedure and lacks generality because it restricts its search to paths that conflict with at most one other agent. This limitation also applies to Enhanced PIBT (EPIBT), a recent extension of PIBT. In this paper, we describe a new perspective on solving MAPF by planning over agent dependencies. Taking inspiration from PIBT's priority inheritance logic, we define the concept of agent dependencies and propose Multi-Dependency PIBT (MD-PIBT) that searches over agent dependencies. MD-PIBT is a general framework where specific parameterizations can reproduce PIBT and EPIBT. At the same time, alternative configurations yield novel planning strategies that are not expressible by PIBT or EPIBT. Our experiments demonstrate that MD-PIBT effectively plans for as many as 10,000 homogeneous agents under various kinodynamic constraints, including pebble motion, rotation motion, and differential drive robots with speed and acceleration limits. We perform thorough evaluations on different variants of MAPF and find that MD-PIBT is particularly effective in MAPF with large agents.

**arXiv ID:** 2603.23405
</details>

<details>
<summary><strong>Biased Error Attribution in Multi-Agent Human-AI Systems Under Delayed Feedback</strong> - Teerthaa Parakh, Karen M. Feigh - [[pdf]](https://arxiv.org/pdf/2603.23419)</summary>

**Abstract:** Human decision-making is strongly influenced by cognitive biases, particularly under conditions of uncertainty and risk. While prior work has examined bias in single-step decisions with immediate outcomes and in human interaction with a single autonomous agent, comparatively little attention has been paid to decision-making under delayed outcomes involving multiple AI agents, where decisions at each step affect subsequent states. In this work, we study how delayed outcomes shape decision-making and responsibility attribution in a multi-agent human-AI task. Using a controlled game-based experiment, we analyze how participants adjust their behavior following positive and negative outcomes. We observe asymmetric responses to gains and losses, with stronger corrective adjustments after negative outcomes. Importantly, participants often fail to correctly identify the actions that caused failure and misattribute responsibility across AI agents, leading to systematic revisions of decisions that are weakly related to the underlying causes of poor performance. We refer to this phenomenon as a form of attribution bias, manifested as biased error attribution under delayed feedback. Our findings highlight how cognitive biases can be amplified in human-AI systems with delayed outcomes and multiple autonomous agents, underscoring the need for decision-support systems that better support causal understanding and learning over time.

**arXiv ID:** 2603.23419
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>Describe-Then-Act: Proactive Agent Steering via Distilled Language-Action World Models</strong> - Massimiliano Pappa, Luca Romani, Valentino Sacco, Alessio Palma, Stéphane Lathuilière, Fabio Galasso, Xavier Alameda-Pineda, Indro Spinelli - [[pdf]](https://arxiv.org/pdf/2603.23149)</summary>

**Abstract:** Deploying safety-critical agents requires anticipating the consequences of actions before they are executed. While world models offer a paradigm for this proactive foresight, current approaches relying on visual simulation incur prohibitive latencies, often exceeding several seconds per step. In this work, we challenge the assumption that visual processing is necessary for failure prevention. We show that a trained policy's latent state, combined with its planned actions, already encodes sufficient information to anticipate action outcomes, making visual simulation redundant for failure prevention. To this end, we introduce DILLO (DIstiLLed Language-ActiOn World Model), a fast steering layer that shifts the paradigm from "simulate-then-act" to "describe-then-act." DILLO is trained via cross-modal distillation, where a privileged Vision Language Model teacher annotates offline trajectories and a latent-conditioned Large Language Model student learns to predict semantic outcomes. This creates a text-only inference path, bypassing heavy visual generation entirely, achieving a 14x speedup over baselines. Experiments on MetaWorld and LIBERO demonstrate that DILLO produces high-fidelity descriptions of the next state and is able to steer the policy, improving episode success rate by up to 15 pp and 9.3 pp on average across tasks.

**arXiv ID:** 2603.23149
</details>

<details>
<summary><strong>AgentSLR: Automating Systematic Literature Reviews in Epidemiology with Agentic AI</strong> - Shreyansh Padarha, Ryan Othniel Kearns, Tristan Naidoo, Lingyi Yang, Łukasz Borchmann, Piotr BŁaszczyk, Christian Morgenstern, Ruth McCabe, Sangeeta Bhatia, Philip H. Torr, Jakob Foerster, Scott A. Hale, Thomas Rawson, Anne Cori, Elizaveta Semenova, Adam Mahdi - [[pdf]](https://arxiv.org/pdf/2603.22327)</summary>

**Abstract:** Systematic literature reviews are essential for synthesizing scientific evidence but are costly, difficult to scale and time-intensive, creating bottlenecks for evidence-based policy. We study whether large language models can automate the complete systematic review workflow, from article retrieval, article screening, data extraction to report synthesis. Applied to epidemiological reviews of nine WHO-designated priority pathogens and validated against expert-curated ground truth, our open-source agentic pipeline (AgentSLR) achieves performance comparable to human researchers while reducing review time from approximately 7 weeks to 20 hours (a 58x speed-up). Our comparison of five frontier models reveals that performance on SLR is driven less by model size or inference cost than by each model's distinctive capabilities. Through human-in-the-loop validation, we identify key failure modes. Our results demonstrate that agentic AI can substantially accelerate scientific evidence synthesis in specialised domains.

**arXiv ID:** 2603.22327
</details>

<details>
<summary><strong>Do Consumers Accept AIs as Moral Compliance Agents?</strong> - Greg Nyilasy, Abraham Ryan Ade Putra Hito, Jennifer Overbeck, Brock Bastian, Darren W. Dahl - [[pdf]](https://arxiv.org/pdf/2603.22617)</summary>

**Abstract:** Consumers are generally resistant to Artificial Intelligence (AI) involvement in moral decision-making, perceiving moral agency as requiring uniquely human traits. This research investigates whether consumers might instead accept AIs in the role of moral compliance, where AI upholds pre-existing moral norms without exercising subjective discretion. Across five studies this research shows that consumers evaluate AI more positively than human agents in moral compliance roles. The findings reveal that this preference arises from inferences of AI's lack of ulterior motives, which are often attributed to human agents. While previous studies have focused on AI as a decision-maker, this work demonstrates the critical role of upholding pre-existing rules, a role in which AI is perceived to excel. These findings contribute to understanding consumer acceptance of moral AI and provide actionable insights for organizations seeking to leverage AI in ethical oversight. By positioning AI as a moral compliance agent, companies can address consumer skepticism, enhance trust, and improve perceptions of corporate ethicality.

**arXiv ID:** 2603.22617
</details>

<details>
<summary><strong>KALAVAI: Predicting When Independent Specialist Fusion Works -- A Quantitative Model for Post-Hoc Cooperative LLM Training</strong> - Ramchand Kumaresan - [[pdf]](https://arxiv.org/pdf/2603.22755)</summary>

**Abstract:** Independently trained domain specialists can be fused post-hoc into a single model that outperforms any individual specialist, and the gain is predictable: gain = 0.82 x divergence - 2.72 (R^2 = 0.856, n=6, 3-26% divergence). This enables practitioners to estimate cooperative value before committing compute. Below ~3.3% divergence, gains approach this http URL the KALAVAI protocol, contributors fine-tune copies of a shared checkpoint independently, then submit for lightweight MoE routing (500 steps). Gains are consistent: +7.72% at 410M (+/-0.02%, 3 seeds), +7.49% at 1B (+/-0.01%, 3 seeds), +6.53% at 6.9B, each over the best specialist. The router matches domain-oracle routing within <10^{-5} nats. Cross-lingual fusion (Tamil/Yoruba/Welsh/Code) achieves +21.76%, with Yoruba perplexity falling 41.9 to 7.7. A 20-contributor federation achieves +16.71% (+/-0.07pp, 3 seeds).Three requirements bound the protocol. Shared initialisation is necessary: checkpoint mismatch degrades routing. Frozen layers are optional below ~10,000 steps and beneficial beyond. Learned routing is essential: uniform averaging degrades by -1.2% vs. best specialist, while any trained router achieves oracle-optimal assignment.

**arXiv ID:** 2603.22755
</details>

<details>
<summary><strong>Designing Agentic AI-Based Screening for Portfolio Investment</strong> - Mehmet Caner, Agostino Capponi, Nathan Sun, Jonathan Y. Tan - [[pdf]](https://arxiv.org/pdf/2603.23300)</summary>

**Abstract:** We introduce a new agentic artificial intelligence (AI) platform for portfolio management. Our architecture consists of three layers. First, two large language model (LLM) agents are assigned specialized tasks: one agent screens for firms with desirable fundamentals, while a sentiment analysis agent screens for firms with desirable news. Second, these agents deliberate to generate and agree upon buy and sell signals from a large portfolio, substantially narrowing the pool of candidate assets. Finally, we apply a high-dimensional precision matrix estimation procedure to determine optimal portfolio weights. A defining theoretical feature of our framework is that the number of assets in the portfolio is itself a random variable, realized through the screening process. We introduce the concept of sensible screening and establish that, under mild screening errors, the squared Sharpe ratio of the screened portfolio consistently estimates its target. Empirically, our method achieves superior Sharpe ratios relative to an unscreened baseline portfolio and to conventional screening approaches, evaluated on S&P 500 data over the period 2020--2024.

**arXiv ID:** 2603.23300
</details>

<details>
<summary><strong>Toward Data Systems That Are Business Semantic Centric and AI Agents Assisted</strong> - Cecil Pang - [[pdf]](https://arxiv.org/pdf/2506.05520)</summary>

**Abstract:** 

**arXiv ID:** 2506.05520
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (9 papers)</h2></summary>

<details>
<summary><strong>Beyond Binary Correctness: Scaling Evaluation of Long-Horizon Agents on Subjective Enterprise Tasks</strong> - Abhishek Chandwani, Ishan Gupta - [[pdf]](https://arxiv.org/pdf/2603.22744)</summary>

**Abstract:** Large language models excel on objectively verifiable tasks such as math and programming, where evaluation reduces to unit tests or a single correct answer. In contrast, real-world enterprise work is often subjective and context-dependent: success hinges on organizational goals, user intent, and the quality of intermediate artifacts produced across long, multi-tool workflows.
We introduce LH-Bench, a three-pillar evaluation design that moves beyond binary correctness to score autonomous, long-horizon execution on subjective enterprise tasks. The pillars are: (i) expert-grounded rubrics that give LLM judges the domain context needed to score subjective work, (ii) curated ground-truth artifacts that enable stepwise reward signals (e.g., chapter-level annotation for content tasks), and (iii) pairwise human preference evaluation for convergent validation. We show that domain-authored rubrics provide substantially more reliable evaluation signals than LLM-authored rubrics (kappa = 0.60 vs. 0.46), and that human preference judgments confirm the same top-tier separation (p < 0.05), evidence that expert-grounded evaluation can scale without sacrificing reliability. We release public datasets and report results on two environments: Figma-to-code (33 real .fig tasks against the Figma API via MCP) and Programmatic content (41 courses comprising 183 individually-evaluated chapters on a course platform serving 30+ daily users).

**arXiv ID:** 2603.22744
</details>

<details>
<summary><strong>Beyond Hard Constraints: Budget-Conditioned Reachability For Safe Offline Reinforcement Learning</strong> - Janaka Chathuranga Brahmanage, Akshat Kumar - [[pdf]](https://arxiv.org/pdf/2603.22292)</summary>

**Abstract:** Sequential decision making using Markov Decision Process underpins many realworld applications. Both model-based and model free methods have achieved strong results in these settings. However, real-world tasks must balance reward maximization with safety constraints, often conflicting objectives, that can lead to unstable min/max, adversarial optimization. A promising alternative is safety reachability analysis, which precomputes a forward-invariant safe state, action set, ensuring that an agent starting inside this set remains safe indefinitely. Yet, most reachability based methods address only hard safety constraints, and little work extends reachability to cumulative cost constraints. To address this, first, we define a safetyconditioned reachability set that decouples reward maximization from cumulative safety cost constraints. Second, we show how this set enforces safety constraints without unstable min/max or Lagrangian optimization, yielding a novel offline safe RL algorithm that learns a safe policy from a fixed dataset without environment interaction. Finally, experiments on standard offline safe RL benchmarks, and a real world maritime navigation task demonstrate that our method matches or outperforms state of the art baselines while maintaining safety.

**arXiv ID:** 2603.22292
</details>

<details>
<summary><strong>CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation</strong> - Max Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, Guanzhi Wang, Fei-Fei Li, Guanya Shi, Jiajun Wu, Shankar Sastry, Yuke Zhu, Ken Goldberg, Linxi "Jim" Fan - [[pdf]](https://arxiv.org/pdf/2603.22435)</summary>

**Abstract:** "Code-as-Policy" considers how executable code can complement data-intensive Vision-Language-Action (VLA) methods, yet their effectiveness as autonomous controllers for embodied manipulation remains underexplored. We present CaP-X, an open-access framework for systematically studying Code-as-Policy agents in robot manipulation. At its core is CaP-Gym, an interactive environment in which agents control robots by synthesizing and executing programs that compose perception and control primitives. Building on this foundation, CaP-Bench evaluates frontier language and vision-language models across varying levels of abstraction, interaction, and perceptual grounding. Across 12 models, CaP-Bench reveals a consistent trend: performance improves with human-crafted abstractions but degrades as these priors are removed, exposing a dependence on designer scaffolding. At the same time, we observe that this gap can be mitigated through scaling agentic test-time computation--through multi-turn interaction, structured execution feedback, visual differencing, automatic skill synthesis, and ensembled reasoning--substantially improves robustness even when agents operate over low-level primitives. These findings allow us to derive CaP-Agent0, a training-free framework that recovers human-level reliability on several manipulation tasks in simulation and on real embodiments. We further introduce CaP-RL, showing reinforcement learning with verifiable rewards improves success rates and transfers from sim2real with minimal gap. Together, CaP-X provides a principled, open-access platform for advancing embodied coding agents.

**arXiv ID:** 2603.22435
</details>

<details>
<summary><strong>AwesomeLit: Towards Hypothesis Generation with Agent-Supported Literature Research</strong> - Zefei Xie, Yuhan Guo, Kai Xu - [[pdf]](https://arxiv.org/pdf/2603.22648)</summary>

**Abstract:** There are different goals for literature research, from understanding an unfamiliar topic to generate hypothesis for the next research project. The nature of literature research also varies according to user's familiarity level of the topic. For inexperienced researchers, identifying gaps in the existing literature and generating feasible hypothesis are crucial but challenging. While general ``deep research'' tools can be used, they are not designed for such use case, thus often not effective. In addition, the ``black box" nature and hallucination of Large Language Models (LLMs) often lead to distrust. In this paper, we introduce a human-agent collaborative visualization system AwesomeLit to address this need. It has several novel features: a transparent user-steerable agentic workflow; a dynamically generated query exploring tree, visualizing the exploration path and provenance; and a semantic similarity view, depicting the relationships between papers. It enables users to transition from general intentions to detailed research topics. Finally, a qualitative study involving several early researchers showed that AwesomeLit is effective in helping users explore unfamiliar topics, identify promising research directions, and improve confidence in research results.

**arXiv ID:** 2603.22648
</details>

<details>
<summary><strong>EVA: Efficient Reinforcement Learning for End-to-End Video Agent</strong> - Yaolun Zhang, Ruohui Wang, Jiahao Wang, Yepeng Tang, Xuanyu Zheng, Haonan Duan, Hao Lu, Hanming Deng, Lewei Lu - [[pdf]](https://arxiv.org/pdf/2603.22918)</summary>

**Abstract:** Video understanding with multimodal large language models (MLLMs) remains challenging due to the long token sequences of videos, which contain extensive temporal dependencies and redundant frames. Existing approaches typically treat MLLMs as passive recognizers, processing entire videos or uniformly sampled frames without adaptive reasoning. Recent agent-based methods introduce external tools, yet still depend on manually designed workflows and perception-first strategies, resulting in inefficiency on long videos. We present EVA, an Efficient Reinforcement Learning framework for End-to-End Video Agent, which enables planning-before-perception through iterative summary-plan-action-reflection reasoning. EVA autonomously decides what to watch, when to watch, and how to watch, achieving query-driven and efficient video understanding. To train such agents, we design a simple yet effective three-stage learning pipeline - comprising supervised fine-tuning (SFT), Kahneman-Tversky Optimization (KTO), and Generalized Reward Policy Optimization (GRPO) - that bridges supervised imitation and reinforcement learning. We further construct high-quality datasets for each stage, supporting stable and reproducible training. We evaluate EVA on six video understanding benchmarks, demonstrating its comprehensive capabilities. Compared with existing baselines, EVA achieves a substantial improvement of 6-12% over general MLLM baselines and a further 1-3% gain over prior adaptive agent methods. Our code and model are available at this https URL.

**arXiv ID:** 2603.22918
</details>

<details>
<summary><strong>Neural ODE and SDE Models for Adaptation and Planning in Model-Based Reinforcement Learning</strong> - Chao Han, Stefanos Ioannou, Luca Manneschi, T.J. Hayward, Michael Mangan, Aditya Gilra, Eleni Vasilaki - [[pdf]](https://arxiv.org/pdf/2603.23245)</summary>

**Abstract:** We investigate neural ordinary and stochastic differential equations (neural ODEs and SDEs) to model stochastic dynamics in fully and partially observed environments within a model-based reinforcement learning (RL) framework. Through a sequence of simulations, we show that neural SDEs more effectively capture the inherent stochasticity of transition dynamics, enabling high-performing policies with improved sample efficiency in challenging scenarios. We leverage neural ODEs and SDEs for efficient policy adaptation to changes in environment dynamics via inverse models, requiring only limited interactions with the new environment. To address partial observability, we introduce a latent SDE model that combines an ODE with a GAN-trained stochastic component in latent space. Policies derived from this model provide a strong baseline, outperforming or matching general model-based and model-free approaches across stochastic continuous-control benchmarks. This work demonstrates the applicability of action-conditional latent SDEs for RL planning in environments with stochastic transitions. Our code is available at: this https URL

**arXiv ID:** 2603.23245
</details>

<details>
<summary><strong>Hybrid Stackelberg Game and Diffusion-based Auction for Two-tier Agentic AI Task Offloading in Internet of Agents</strong> - Yue Zhong, Yongju Tong, Jiawen Kang, Minghui Dai, Hong-Ning Dai, Zhou Su, Dusit Niyato - [[pdf]](https://arxiv.org/pdf/2511.22076)</summary>

**Abstract:** The Internet of Agents (IoA) is rapidly gaining prominence as a foundational architecture for interconnected intelligent systems, designed to facilitate seamless discovery, communication, and collaborative reasoning among a vast network of Artificial Intelligence (AI) agents. Powered by Large Language and Vision-Language Models, IoA enables the development of interactive, rational agents capable of complex cooperation, moving far beyond traditional isolated models. IoA involves physical entities, i.e., Wireless Agents (WAs) with limited onboard resources, which need to offload their compute-intensive agentic AI services to nearby servers. Such servers can be Mobile Agents (MAs), e.g., vehicle agents, or Fixed Agents (FAs), e.g., end-side units agents. Given their fixed geographical locations and stable connectivity, FAs can serve as reliable communication gateways and task aggregation points. This stability allows them to effectively coordinate with and offload to an Aerial Agent (AA) tier, which has an advantage not affordable for highly mobile MAs with dynamic connectivity limitations. As such, we propose a two-tier optimization approach. The first tier employs a multi-leader multi-follower Stackelberg game. In the game, MAs and FAs act as the leaders who set resource prices. WAs are the followers to determine task offloading ratios. However, when FAs become overloaded, they can further offload tasks to available aerial resources. Therefore, the second tier introduces a Double Dutch Auction model where overloaded FAs act as the buyers to request resources, and AAs serve as the sellers for resource provision. We then develop a diffusion-based Deep Reinforcement Learning algorithm to solve the model. Numerical results demonstrate the superiority of our proposed scheme in facilitating task offloading.

**arXiv ID:** 2511.22076
</details>

<details>
<summary><strong>CXReasonAgent: Evidence-Grounded Diagnostic Reasoning Agent for Chest X-rays</strong> - Hyungyung Lee, Hangyul Yoon, Edward Choi - [[pdf]](https://arxiv.org/pdf/2602.23276)</summary>

**Abstract:** Chest X-ray plays a central role in thoracic diagnosis, and its interpretation inherently requires multi-step, evidence-grounded reasoning. However, large vision-language models (LVLMs) often generate plausible responses that are not faithfully grounded in diagnostic evidence and provide limited visual evidence for verification, while also requiring costly retraining to support new diagnostic tasks, limiting their reliability and adaptability in clinical settings. To address these limitations, we present CXReasonAgent, a diagnostic agent that integrates a large language model (LLM) with clinically grounded diagnostic tools to perform evidence-grounded diagnostic reasoning using image-derived diagnostic and visual evidence. To evaluate these capabilities, we introduce CXReasonDial, a multi-turn dialogue benchmark with 1,946 dialogues across 12 diagnostic tasks, and show that CXReasonAgent produces faithfully grounded responses, enabling more reliable and verifiable diagnostic reasoning than LVLMs. These findings highlight the importance of integrating clinically grounded diagnostic tools, particularly in safety-critical clinical settings. The demo is available \href{this https URL}{here}.

**arXiv ID:** 2602.23276
</details>

<details>
<summary><strong>Agentic AI-based Coverage Closure for Formal Verification</strong> - Sivaram Pothireddypalli, Ashish Raman, Deepak Narayan Gadde, Aman Kumar - [[pdf]](https://arxiv.org/pdf/2603.03147)</summary>

**Abstract:** Coverage closure is a critical requirement in Integrated Chip (IC) development process and key metric for verification sign-off. However, traditional exhaustive approaches often fail to achieve full coverage within project timelines. This study presents an agentic AI-driven workflow that utilizes Large Language Model (LLM)-enabled Generative AI (GenAI) to automate coverage analysis for formal verification, identify coverage gaps, and generate the required formal properties. The framework accelerates verification efficiency by systematically addressing coverage holes. Benchmarking open-source and internal designs reveals a measurable increase in coverage metrics, with improvements correlated to the complexity of the design. Comparative analysis validates the effectiveness of this approach. These results highlight the potential of agentic AI-based techniques to improve formal verification productivity and support comprehensive coverage closure.

**arXiv ID:** 2603.03147
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
