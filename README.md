# Agent arXiv Daily

**Last Updated:** 2026-03-06 03:36:51

**Total Papers:** 106

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
<summary><strong>GCAgent: Enhancing Group Chat Communication through Dialogue Agents System</strong> - Zijie Meng, Zheyong Xie, Zheyu Ye, Chonggang Lu, Zuozhu Liu, Zihan Niu, Yao Hu, Shaosheng Cao - [[pdf]](https://arxiv.org/pdf/2603.05240)</summary>

**Abstract:** As a key form in online social platforms, group chat is a popular space for interest exchange or problem-solving, but its effectiveness is often hindered by inactivity and management challenges. While recent large language models (LLMs) have powered impressive one-to-one conversational agents, their seamlessly integration into multi-participant conversations remains unexplored. To address this gap, we introduce GCAgent, an LLM-driven system for enhancing group chats communication with both entertainment- and utility-oriented dialogue agents. The system comprises three tightly integrated modules: Agent Builder, which customizes agents to align with users' interests; Dialogue Manager, which coordinates dialogue states and manage agent invocations; and Interface Plugins, which reduce interaction barriers by three distinct tools. Through extensive experiment, GCAgent achieved an average score of 4.68 across various criteria and was preferred in 51.04\% of cases compared to its base model. Additionally, in real-world deployments over 350 days, it increased message volume by 28.80\%, significantly improving group activity and engagement. Overall, this work presents a practical blueprint for extending LLM-based dialogue agent from one-party chats to multi-party group scenarios.

**arXiv ID:** 2603.05240
</details>

<details>
<summary><strong>Designing for Adolescent Voice in Health Decisions: Embodied Conversational Agents for HPV Vaccination</strong> - Ian Steenstra, Neha Patkar, Rebecca B. Perkins, Michael K. Paasche-Orlow, Timothy Bickmore - [[pdf]](https://arxiv.org/pdf/2603.05321)</summary>

**Abstract:** Adolescents are directly affected by preventive health decisions such as vaccination, yet their perspectives are rarely solicited or supported. Most digital interventions for Human Papillomavirus (HPV) vaccination are designed exclusively for parents, implicitly treating adolescents as passive recipients rather than stakeholders with agency. We present the design and evaluation of a mobile intervention that gives adolescents a voice in HPV vaccination decisions alongside their parents. The system uses embodied conversational agents tailored to each audience: parents interact with an animated physician using education and motivational interviewing techniques, while adolescents can choose between an age-appropriate doctor or a narrative fantasy game that conveys HPV facts through play. We report findings from a clinic-based pilot study with 21 parent-adolescent dyads. Results indicate high satisfaction across both audiences, improved HPV knowledge, and increased intent to vaccinate. We discuss design implications for supporting adolescent participation, choice, and agency in decisions about their health.

**arXiv ID:** 2603.05321
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (19 papers)</h2></summary>

<details>
<summary><strong>Evaluating the Search Agent in a Parallel World</strong> - Jiawei Chen, Xintian Shen, Lihao Zheng, Lifu Mu, Haoyi Sun, Ning Mao, Hao Ma, Tao Wei, Pan Zhou, Kun Zhan - [[pdf]](https://arxiv.org/pdf/2603.04751)</summary>

**Abstract:** Integrating web search tools has significantly extended the capability of LLMs to address open-world, real-time, and long-tail problems. However, evaluating these Search Agents presents formidable challenges. First, constructing high-quality deep search benchmarks is prohibitively expensive, while unverified synthetic data often suffers from unreliable sources. Second, static benchmarks face dynamic obsolescence: as internet information evolves, complex queries requiring deep research often degrade into simple retrieval tasks due to increased popularity, and ground truths become outdated due to temporal shifts. Third, attribution ambiguity confounds evaluation, as an agent's performance is often dominated by its parametric memory rather than its actual search and reasoning capabilities. Finally, reliance on specific commercial search engines introduces variability that hampers reproducibility. To address these issues, we propose a novel framework, Mind-ParaWorld, for evaluating Search Agents in a Parallel World. Specifically, MPW samples real-world entity names to synthesize future scenarios and questions situated beyond the model's knowledge cutoff. A ParaWorld Law Model then constructs a set of indivisible Atomic Facts and a unique ground-truth for each question. During evaluation, instead of retrieving real-world results, the agent interacts with a ParaWorld Engine Model that dynamically generates SERPs grounded in these inviolable Atomic Facts. We release MPW-Bench, an interactive benchmark spanning 19 domains with 1,608 instances. Experiments across three evaluation settings show that, while search agents are strong at evidence synthesis given complete information, their performance is limited not only by evidence collection and coverage in unfamiliar search environments, but also by unreliable evidence sufficiency judgment and when-to-stop decisions-bottlenecks.

**arXiv ID:** 2603.04751
</details>

<details>
<summary><strong>MOOSEnger -- a Domain-Specific AI Agent for the MOOSE Ecosystem</strong> - Mengnan Li, Jason Miller, Zachary Prince, Alexander Lindsay, Cody Permann - [[pdf]](https://arxiv.org/pdf/2603.04756)</summary>

**Abstract:** MOOSEnger is a tool-enabled AI agent tailored to the Multiphysics Object-Oriented Simulation Environment (MOOSE). MOOSE cases are specified in HIT ".i" input files; the large object catalog and strict syntax make initial setup and debugging slow. MOOSEnger offers a conversational workflow that turns natural-language intent into runnable inputs by combining retrieval-augmented generation over curated docs/examples with deterministic, MOOSE-aware parsing, validation, and execution tools. A core-plus-domain architecture separates reusable agent infrastructure (configuration, registries, tool dispatch, retrieval services, persistence, and evaluation) from a MOOSE plugin that adds HIT-based parsing, syntax-preserving ingestion of input files, and domain-specific utilities for input repair and checking. An input precheck pipeline removes hidden formatting artifacts, fixes malformed HIT structure with a bounded grammar-constrained loop, and resolves invalid object types via similarity search over an application syntax registry. Inputs are then validated and optionally smoke-tested with the MOOSE runtime in the loop via an MCP-backed execution backend (with local fallback), translating solver diagnostics into iterative verify-and-correct updates. Built-in evaluation reports RAG metrics (faithfulness, relevancy, context precision/recall) and end-to-end success by actual execution. On a 125-prompt benchmark spanning diffusion, transient heat conduction, solid mechanics, porous flow, and incompressible Navier--Stokes, MOOSEnger achieves a 0.93 execution pass rate versus 0.08 for an LLM-only baseline.

**arXiv ID:** 2603.04756
</details>

<details>
<summary><strong>EchoGuard: An Agentic Framework with Knowledge-Graph Memory for Detecting Manipulative Communication in Longitudinal Dialogue</strong> - Ratna Kandala, Niva Manchanda, Akshata Kishore Moharir, Ananth Kandala - [[pdf]](https://arxiv.org/pdf/2603.04815)</summary>

**Abstract:** Manipulative communication, such as gaslighting, guilt-tripping, and emotional coercion, is often difficult for individuals to recognize. Existing agentic AI systems lack the structured, longitudinal memory to track these subtle, context-dependent tactics, often failing due to limited context windows and catastrophic forgetting. We introduce EchoGuard, an agentic AI framework that addresses this gap by using a Knowledge Graph (KG) as the agent's core episodic and semantic memory. EchoGuard employs a structured Log-Analyze-Reflect loop: (1) users log interactions, which the agent structures as nodes and edges in a personal, episodic KG (capturing events, emotions, and speakers); (2) the system executes complex graph queries to detect six psychologically-grounded manipulation patterns (stored as a semantic KG); and (3) an LLM generates targeted Socratic prompts grounded by the subgraph of detected patterns, guiding users toward self-discovery. This framework demonstrates how the interplay between agentic architectures and Knowledge Graphs can empower individuals in recognizing manipulative communication while maintaining personal autonomy and safety. We present the theoretical foundation, framework design, a comprehensive evaluation strategy, and a vision to validate this approach.

**arXiv ID:** 2603.04815
</details>

<details>
<summary><strong>SEA-TS: Self-Evolving Agent for Autonomous Code Generation of Time Series Forecasting Algorithms</strong> - Longkun Xu, Xiaochun Zhang, Qiantu Tuo, Rui Li - [[pdf]](https://arxiv.org/pdf/2603.04873)</summary>

**Abstract:** Accurate time series forecasting underpins decision-making across domains, yet conventional ML development suffers from data scarcity in new deployments, poor adaptability under distribution shift, and diminishing returns from manual iteration. We propose Self-Evolving Agent for Time Series Algorithms (SEA-TS), a framework that autonomously generates, validates, and optimizes forecasting code via an iterative self-evolution loop. Our framework introduces three key innovations: (1) Metric-Advantage Monte Carlo Tree Search (MA-MCTS), which replaces fixed rewards with a normalized advantage score for discriminative search guidance; (2) Code Review with running prompt refinement, where each executed solution undergoes automated review followed by prompt updates that encode corrective patterns, preventing recurrence of similar errors; and (3) Global Steerable Reasoning, which compares each node against global best and worst solutions, enabling cross-trajectory knowledge transfer. We adopt a MAP-Elites archive for architectural diversity. On the public Solar-Energy benchmark, SEA-TS generated code achieves a 40% MAE reduction relative to TimeMixer, surpassing state-of-the-art methods. On proprietary datasets, SEA-TS generated code reduces WAPE by 8.6% on solar PV forecasting and 7.7% on residential load forecasting compared to human-engineered baselines, and achieves 26.17% MAPE on load forecasting versus 29.34% by TimeMixer. Notably, the evolved models discover novel architectural patterns--including physics-informed monotonic decay heads encoding solar irradiance constraints, per-station learned diurnal cycle profiles, and learnable hourly bias correction--demonstrating that autonomous ML engineering can generate genuinely novel algorithmic ideas beyond manual design.

**arXiv ID:** 2603.04873
</details>

<details>
<summary><strong>TimeWarp: Evaluating Web Agents by Revisiting the Past</strong> - Md Farhan Ishmam, Kenneth Marino - [[pdf]](https://arxiv.org/pdf/2603.04949)</summary>

**Abstract:** The improvement of web agents on current benchmarks raises the question: Do today's agents perform just as well when the web changes? We introduce TimeWarp, a benchmark that emulates the evolving web using containerized environments that vary in UI, design, and layout. TimeWarp consists of three web environments, each with six UI versions spanning different eras of the internet, paired with a set of complex, realistic tasks requiring different forms of web navigation. Our experiments reveal web agents' vulnerability to changes and the limitations of behavior cloning (BC) on single-version trajectories. To address this, we propose TimeTraj, a simple yet effective algorithm that uses plan distillation to collect trajectories across multiple versions. By training agents on teacher rollouts using our BC-variant, we achieve substantial performance gains: $20.4\%\rightarrow37.7\%$ for Qwen-3 4B and $0\%\rightarrow27.0\%$ for Llama-3.1 8B models. We hope our work helps researchers study generalization across web designs and unlock a new paradigm for collecting plans rather than trajectories, thereby improving the robustness of web agents.

**arXiv ID:** 2603.04949
</details>

<details>
<summary><strong>AegisUI: Behavioral Anomaly Detection for Structured User Interface Protocols in AI Agent Systems</strong> - Mohd Safwan Uddin, Saba Hajira - [[pdf]](https://arxiv.org/pdf/2603.05031)</summary>

**Abstract:** AI agents that build user interfaces on the fly assembling buttons, forms, and data displays from structured protocol payloads are becoming common in production systems. The trouble is that a payload can pass every schema check and still trick a user: a button might say "View invoice" while its hidden action wipes an account, or a display widget might quietly bind to an internal salary field. Current defenses stop at syntax; they were never built to catch this kind of behavioral mismatch.
We built AegisUI to study exactly this gap. The framework generates structured UI payloads, injects realistic attacks into them, extracts numeric features, and benchmarks anomaly detectors end-to-end. We produced 4000 labeled payloads (3000 benign, 1000 malicious) spanning five application domains and five attack families: phishing interfaces, data leakage, layout abuse, manipulative UI, and workflow anomalies.
From each payload we extracted 18 features covering structural, semantic, binding, and session dimensions, then compared three detectors: Isolation Forest (unsupervised), a benign-trained autoencoder (semi-supervised), and Random Forest (supervised). On a stratified 80/20 split, Random Forest scored best overall (accuracy 0.931, precision 0.980, recall 0.740, F1 0.843, ROC-AUC 0.952). The autoencoder came second (F1 0.762, ROC-AUC 0.863) and needs no malicious labels at training time, which matters when deploying a new system that lacks attack history. Per-attack-type analysis showed that layout abuse is easiest to catch while manipulative UI payloads are hardest. All code, data, and configurations are released for full reproducibility.

**arXiv ID:** 2603.05031
</details>

<details>
<summary><strong>STRUCTUREDAGENT: Planning with AND/OR Trees for Long-Horizon Web Tasks</strong> - ELita Lobo, Xu Chen, Jingjing Meng, Nan Xi, Yang Jiao, Chirag Agarwal, Yair Zick, Yan Gao - [[pdf]](https://arxiv.org/pdf/2603.05294)</summary>

**Abstract:** Recent advances in large language models (LLMs) have enabled agentic systems for sequential decision-making. Such agents must perceive their environment, reason across multiple time steps, and take actions that optimize long-term objectives. However, existing web agents struggle on complex, long-horizon tasks due to limited in-context memory for tracking history, weak planning abilities, and greedy behaviors that lead to premature termination. To address these challenges, we propose STRUCTUREDAGENT, a hierarchical planning framework with two core components: (1) an online hierarchical planner that uses dynamic AND/OR trees for efficient search and (2) a structured memory module that tracks and maintains candidate solutions to improve constraint satisfaction in information-seeking tasks. The framework also produces interpretable hierarchical plans, enabling easier debugging and facilitating human intervention when needed. Our results on WebVoyager, WebArena, and custom shopping benchmarks show that STRUCTUREDAGENT improves performance on long-horizon web-browsing tasks compared to standard LLM-based agents.

**arXiv ID:** 2603.05294
</details>

<details>
<summary><strong>FinRetrieval: A Benchmark for Financial Data Retrieval by AI Agents</strong> - Eric Y. Kim, Jie Huang - [[pdf]](https://arxiv.org/pdf/2603.04403)</summary>

**Abstract:** AI agents increasingly assist with financial research, yet no benchmark evaluates their ability to retrieve specific numeric values from structured databases. We introduce FinRetrieval, a benchmark of 500 financial retrieval questions with ground truth answers, agent responses from 14 configurations across three frontier providers (Anthropic, OpenAI, Google), and complete tool call execution traces. Our evaluation reveals that tool availability dominates performance: Claude Opus achieves 90.8% accuracy with structured data APIs but only 19.8% with web search alone--a 71 percentage point gap that exceeds other providers by 3-4x. We find that reasoning mode benefits vary inversely with base capability (+9.0pp for OpenAI vs +2.8pp for Claude), explained by differences in base-mode tool utilization rather than reasoning ability. Geographic performance gaps (5.6pp US advantage) stem from fiscal year naming conventions, not model limitations. We release the dataset, evaluation code, and tool traces to enable research on financial AI systems.

**arXiv ID:** 2603.04403
</details>

<details>
<summary><strong>Escaping the Hydrolysis Trap: An Agentic Workflow for Inverse Design of Durable Photocatalytic Covalent Organic Frameworks</strong> - Iman Peivaste, Nicolas D. Boscher, Ahmed Makradi, Salim Belouettar - [[pdf]](https://arxiv.org/pdf/2603.05188)</summary>

**Abstract:** Covalent organic frameworks (COFs) are promising photocatalysts for solar hydrogen production, yet the most electronically favorable linkages, imines, hydrolyze rapidly in water, creating a stability--activity trade-off that limits practical deployment. Navigating the combinatorial design space of nodes, linkers, linkages, and functional groups to identify candidates that are simultaneously active and durable remains a formidable challenge. Here we introduce Ara, a large-language-model (LLM) agent that leverages pretrained chemical knowledge, donor--acceptor theory, conjugation effects, and linkage stability hierarchies, to guide the search for photocatalytic COFs satisfying joint band-gap, band-edge, and hydrolytic-stability criteria. Evaluated against random search and Bayesian optimization (BO) over a space consisting of candidates with various nodes, linkers, linkages, and r-groups, screened with a GFN1-xTB fragment pipeline, Ara achieves a 52.7\% hit rate (11.5$\times$ random, p = 0.006), finds its first hit at iteration 12 versus 25 for random search, and significantly outperforms BO (p = 0.006). Inspection of the agent's reasoning traces reveals interpretable chemical logic: early convergence on vinylene and beta-ketoenamine linkages for stability, node selection informed by electron-withdrawing character, and systematic R-group optimization to center the band gap at 2.0 eV. Exhaustive evaluation of the full search space uncovers a complementary exploitation--exploration trade-off between the agent and BO, suggesting that hybrid strategies may combine the strengths of both approaches. These results demonstrate that LLM chemical priors can substantially accelerate multi-criteria materials discovery.

**arXiv ID:** 2603.05188
</details>

<details>
<summary><strong>BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving</strong> - Shu Liu, Wenlin Chen, Weihao Li, Zheng Wang, Lijin Yang, Jianing Huang, Yipin Zhang, Zhongzhan Huang, Ze Cheng, Hao Yang - [[pdf]](https://arxiv.org/pdf/2509.23589)</summary>

**Abstract:** Diffusion-based planners have shown strong potential for autonomous driving by capturing multi-modal driving behaviors. A key challenge is how to effectively guide these models for safe and reactive planning in closed-loop settings, where the ego vehicle's actions influence future states. Recent work leverages typical expert driving behaviors (i.e., anchors) to guide diffusion planners but relies on a truncated diffusion schedule that introduces an asymmetry between the forward and denoising processes, diverging from the core principles of diffusion models. To address this, we introduce BridgeDrive, a novel anchor-guided diffusion bridge policy for closed-loop trajectory planning. Our approach formulates planning as a diffusion bridge that directly transforms coarse anchor trajectories into refined, context-aware plans, ensuring theoretical consistency between the forward and reverse processes. BridgeDrive is compatible with efficient ODE solvers, enabling real-time deployment. We achieve state-of-the-art performance on the Bench2Drive closed-loop evaluation benchmark, improving the success rate by 7.72% and 2.45% over prior arts with PDM-Lite and LEAD datasets, respectively. Project page: this https URL.

**arXiv ID:** 2509.23589
</details>

<details>
<summary><strong>DAP: A Discrete-token Autoregressive Planner for Autonomous Driving</strong> - Bowen Ye, Bin Zhang, Hang Zhao - [[pdf]](https://arxiv.org/pdf/2511.13306)</summary>

**Abstract:** Gaining sustainable performance improvement with scaling data and model budget remains a pivotal yet unresolved challenge in autonomous driving. While autoregressive models exhibited promising data-scaling efficiency in planning tasks, predicting ego trajectories alone suffers sparse supervision and weakly constrains how scene evolution should shape ego motion. Therefore, we introduce DAP, a discrete-token autoregressive planner that jointly forecasts BEV semantics and ego trajectories, thereby enforcing comprehensive representation learning and allowing predicted dynamics to directly condition ego motion. In addition, we incorporate a reinforcement-learning-based fine-tuning, which preserves supervised behavior cloning priors while injecting reward-guided improvements. Despite a compact 160M parameter budget, DAP achieves state-of-the-art performance on open-loop metrics and delivers competitive closed-loop results on the NAVSIM benchmark. Overall, the fully discrete-token autoregressive formulation operating on both rasterized BEV and ego actions provides a compact yet scalable planning paradigm for autonomous driving.

**arXiv ID:** 2511.13306
</details>

<details>
<summary><strong>PerfGuard: A Performance-Aware Agent for Visual Content Generation</strong> - Zhipeng Chen, Zhongrui Zhang, Chao Zhang, Yifan Xu, Lan Yang, Jun Liu, Ke Li, Yi-Zhe Song - [[pdf]](https://arxiv.org/pdf/2601.22571)</summary>

**Abstract:** The advancement of Large Language Model (LLM)-powered agents has enabled automated task processing through reasoning and tool invocation capabilities. However, existing frameworks often operate under the idealized assumption that tool executions are invariably successful, relying solely on textual descriptions that fail to distinguish precise performance boundaries and cannot adapt to iterative tool updates. This gap introduces uncertainty in planning and execution, particularly in domains like visual content generation (AIGC), where nuanced tool performance significantly impacts outcomes. To address this, we propose PerfGuard, a performance-aware agent framework for visual content generation that systematically models tool performance boundaries and integrates them into task planning and scheduling. Our framework introduces three core mechanisms: (1) Performance-Aware Selection Modeling (PASM), which replaces generic tool descriptions with a multi-dimensional scoring system based on fine-grained performance evaluations; (2) Adaptive Preference Update (APU), which dynamically optimizes tool selection by comparing theoretical rankings with actual execution rankings; and (3) Capability-Aligned Planning Optimization (CAPO), which guides the planner to generate subtasks aligned with performance-aware strategies. Experimental comparisons against state-of-the-art methods demonstrate PerfGuard's advantages in tool selection accuracy, execution reliability, and alignment with user intent, validating its robustness and practical utility for complex AIGC tasks. The project code is available at this https URL.

**arXiv ID:** 2601.22571
</details>

<details>
<summary><strong>Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving</strong> - Ahmed Abouelazm, Mianzhi Liu, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner - [[pdf]](https://arxiv.org/pdf/2505.06740)</summary>

**Abstract:** Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66% to just 1%. These results highlight the effectiveness of our approach in generating feasible and robust predictions.

**arXiv ID:** 2505.06740
</details>

<details>
<summary><strong>Beyond the Context Window: A Cost-Performance Analysis of Fact-Based Memory vs. Long-Context LLMs for Persistent Agents</strong> - Natchanon Pollertlam, Witchayut Kornsuwannawit - [[pdf]](https://arxiv.org/pdf/2603.04814)</summary>

**Abstract:** Persistent conversational AI systems face a choice between passing full conversation histories to a long-context large language model (LLM) and maintaining a dedicated memory system that extracts and retrieves structured facts. We compare a fact-based memory system built on the Mem0 framework against long-context LLM inference on three memory-centric benchmarks - LongMemEval, LoCoMo, and PersonaMemv2 - and evaluate both architectures on accuracy and cumulative API cost. Long-context GPT-5-mini achieves higher factual recall on LongMemEval and LoCoMo, while the memory system is competitive on PersonaMemv2, where persona consistency depends on stable, factual attributes suited to flat-typed extraction. We construct a cost model that incorporates prompt caching and show that the two architectures have structurally different cost profiles: long-context inference incurs a per-turn charge that grows with context length even under caching, while the memory system's per-turn read cost remains roughly fixed after a one-time write phase. At a context length of 100k tokens, the memory system becomes cheaper after approximately ten interaction turns, with the break-even point decreasing as context length grows. These results characterize the accuracy-cost trade-off between the two approaches and provide a concrete criterion for selecting between them in production deployments.

**arXiv ID:** 2603.04814
</details>

<details>
<summary><strong>Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs</strong> - Yurun Chen, Xavier Hu, Yuhan Liu, Ziqi Wang, Zeyi Liao, Lin Chen, Feng Wei, Yuxi Qian, Bo Zheng, Keting Yin, Shengyu Zhang - [[pdf]](https://arxiv.org/pdf/2510.00507)</summary>

**Abstract:** As multimodal LLM-driven agents advance in autonomy and generalization, traditional static datasets face inherent scalability limitations and are insufficient for fully assessing their capabilities in increasingly complex and diverse tasks. Existing studies have attempted to generate agent tasks using LLMs, but due to the inherent hallucinations of LLMs and the lack of internal data relationship modeling, these tasks often exhibit semantic inconsistencies and solvability issues. To address these challenges, we introduce Graph2Eval, a knowledge-graph-driven framework for automated, scalable, and semantically grounded agent task generation. At its core, Graph2Eval leverages a knowledge graph built from heterogeneous external data sources as a structured task space, generating multimodal agent tasks through subgraph sampling and task construction guided by task templates and meta-path strategies. To further ensure task reliability, a multi-stage filtering pipeline based on node reachability analysis, LLM scoring, and similarity analysis ensures the diversity and solvability of the generated tasks. By unifying both RAG Agent and Web Agent scenarios, Graph2Eval enables efficient generation of multimodal document understanding tasks and multi-step web interaction tasks. We instantiate the framework with Graph2Eval-Bench, a curated dataset of 1,319 tasks spanning document understanding and web interaction scenarios. Extensive experiments show that, on average, Graph2Eval improves task semantic consistency by 20% and solvability by 17% over baselines, while Graph2Eval-Bench effectively distinguishes agent performance, offering a new perspective on agent evaluation.

**arXiv ID:** 2510.00507
</details>

<details>
<summary><strong>AgentIR: Reasoning-Aware Retrieval for Deep Research Agents</strong> - Zijian Chen, Xueguang Ma, Shengyao Zhuang, Jimmy Lin, Akari Asai, Victor Zhong - [[pdf]](https://arxiv.org/pdf/2603.04384)</summary>

**Abstract:** Deep Research agents are rapidly emerging as primary consumers of modern retrieval systems. Unlike human users who issue and refine queries without documenting their intermediate thought processes, Deep Research agents generate explicit natural language reasoning before each search call, revealing rich intent and contextual information that existing retrievers entirely ignore. To exploit this overlooked signal, we introduce: (1) Reasoning-Aware Retrieval, a retrieval paradigm that jointly embeds the agent's reasoning trace alongside its query; and (2) DR-Synth, a data synthesis method that generates Deep Research retriever training data from standard QA datasets. We demonstrate that both components are independently effective, and their combination yields a trained embedding model, AgentIR-4B, with substantial gains. On the challenging BrowseComp-Plus benchmark, AgentIR-4B achieves 68\% accuracy with the open-weight agent Tongyi-DeepResearch, compared to 50\% with conventional embedding models twice its size, and 37\% with BM25. Code and data are available at: this https URL.

**arXiv ID:** 2603.04384
</details>

<details>
<summary><strong>Autonomous Aerial Non-Destructive Testing: Ultrasound Inspection with a Commercial Quadrotor in an Unstructured Environment</strong> - Ruben Veenstra, Barbara Bazzana, Sander Smits, Antonio Franchi - [[pdf]](https://arxiv.org/pdf/2603.04642)</summary>

**Abstract:** This work presents an integrated control and software architecture that enables arguably the first fully autonomous, contact-based non-destructive testing (NDT) using a commercial multirotor originally restricted to remotely-piloted operations. To allow autonomous operation with an off-the-shelf platform, we developed a real-time framework that interfaces directly with its onboard sensor suite. The architecture features a multi-rate control scheme: low-level control is executed at 200 Hz, force estimation at 100 Hz, while an admittance filter and trajectory planner operate at 50 Hz, ultimately supplying acceleration and yaw rate commands to the internal flight controller. We validate the system through physical experiments on a Flyability Elios 3 quadrotor equipped with an ultrasound payload. Relying exclusively on onboard sensing, the vehicle successfully performs autonomous NDT measurements within an unstructured, industrial-like environment. This work demonstrates the viability of retrofitting off-the-shelf platforms for autonomous physical interaction, paving the way for safe, contact-based inspection of hazardous and confined infrastructure.

**arXiv ID:** 2603.04642
</details>

<details>
<summary><strong>GUIDE: A Diffusion-Based Autonomous Robot Exploration Framework Using Global Graph Inference</strong> - Zijun Che, Yinghong Zhang, Shengyi Liang, Boyu Zhou, Jun Ma, Jinni Zhou - [[pdf]](https://arxiv.org/pdf/2509.19916)</summary>

**Abstract:** Autonomous exploration in structured and complex indoor environments remains a challenging task, as existing methods often struggle to appropriately model unobserved space and plan globally efficient paths. To address these limitations, we propose GUIDE, a novel exploration framework that synergistically combines global graph inference with diffusion-based decision-making. We introduce a region-evaluation global graph representation that integrates both observed environmental data and predictions of unexplored areas, enhanced by a region-level evaluation mechanism to prioritize reliable structural inferences while discounting uncertain predictions. Building upon this enriched representation, a diffusion policy network generates stable, foresighted action sequences with significantly reduced denoising steps. Extensive simulations and real-world deployments demonstrate that GUIDE consistently outperforms state-of-the-art methods, achieving up to 18.3% faster coverage completion and a 34.9% reduction in redundant movements.

**arXiv ID:** 2509.19916
</details>

<details>
<summary><strong>LAP: Fast LAtent Diffusion Planner for Autonomous Driving</strong> - Jinhao Zhang, Wenlong Xia, Zhexuan Zhou, Haoming Song, Youmin Gong, Jie Mei - [[pdf]](https://arxiv.org/pdf/2512.00470)</summary>

**Abstract:** Diffusion models have demonstrated strong capabilities for modeling human-like driving behaviors in autonomous driving, but their iterative sampling process induces substantial latency, and operating directly on raw trajectory points forces the model to spend capacity on low-level kinematics, rather than high-level multi-modal semantics. To address these limitations, we propose LAtent Planner (LAP), a framework that plans in a VAE-learned latent space that disentangles high-level intents from low-level kinematics, enabling our planner to capture rich, multi-modal driving strategies. To bridge the representational gap between the high-level semantic planning space and the vectorized scene context, we introduce an intermediate feature alignment mechanism that facilitates robust information fusion. Notably, LAP can produce high-quality plans in one single denoising step, substantially reducing computational overhead. Through extensive evaluations on the large-scale nuPlan benchmark, LAP achieves state-of-the-art closed-loop performance among learning-based planning methods, while demonstrating an inference speed-up of at most 10x over previous SOTA approaches.

**arXiv ID:** 2512.00470
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>Adaptive Memory Admission Control for LLM Agents</strong> - Guilin Zhang, Wei Jiang, Xiejiashan Wang, Aisha Behr, Kai Zhao, Jeffrey Friedman, Xu Chu, Amine Anoun - [[pdf]](https://arxiv.org/pdf/2603.04549)</summary>

**Abstract:** LLM-based agents increasingly rely on long-term memory to support multi-session reasoning and interaction, yet current systems provide little control over what information is retained. In practice, agents either accumulate large volumes of conversational content, including hallucinated or obsolete facts, or depend on opaque, fully LLM-driven memory policies that are costly and difficult to audit. As a result, memory admission remains a poorly specified and weakly controlled component in agent architectures. To address this gap, we propose Adaptive Memory Admission Control (A-MAC), a framework that treats memory admission as a structured decision problem. A-MAC decomposes memory value into five complementary and interpretable factors: future utility, factual confidence, semantic novelty, temporal recency, and content type prior. The framework combines lightweight rule-based feature extraction with a single LLM-assisted utility assessment, and learns domain-adaptive admission policies through cross-validated optimization. This design enables transparent and efficient control over long-term memory. Experiments on the LoCoMo benchmark show that A-MAC achieves a superior precision-recall tradeoff, improving F1 to 0.583 while reducing latency by 31% compared to state-of-the-art LLM-native memory systems. Ablation results identify content type prior as the most influential factor for reliable memory admission. These findings demonstrate that explicit and interpretable admission control is a critical design principle for scalable and reliable memory in LLM-based agents.

**arXiv ID:** 2603.04549
</details>

<details>
<summary><strong>EvoTool: Self-Evolving Tool-Use Policy Optimization in LLM Agents via Blame-Aware Mutation and Diversity-Aware Selection</strong> - Shuo Yang, Soyeon Caren Han, Xueqi Ma, Yan Li, Mohammad Reza Ghasemi Madani, Eduard Hovy - [[pdf]](https://arxiv.org/pdf/2603.04900)</summary>

**Abstract:** LLM-based agents depend on effective tool-use policies to solve complex tasks, yet optimizing these policies remains challenging due to delayed supervision and the difficulty of credit assignment in long-horizon trajectories. Existing optimization approaches tend to be either monolithic, which are prone to entangling behaviors, or single-aspect, which ignore cross-module error propagation. To address these limitations, we propose EvoTool, a self-evolving framework that optimizes a modular tool-use policy via a gradient-free evolutionary paradigm. EvoTool decomposes agent's tool-use policy into four modules, including Planner, Selector, Caller, and Synthesizer, and iteratively improves them in a self-improving loop through three novel mechanisms. Trajectory-Grounded Blame Attribution uses diagnostic traces to localize failures to a specific module. Feedback-Guided Targeted Mutation then edits only that module via natural-language critique. Diversity-Aware Population Selection preserves complementary candidates to ensure solution diversity. Across four benchmarks, EvoTool outperforms strong baselines by over 5 points on both GPT-4.1 and Qwen3-8B, while achieving superior efficiency and transferability. The code will be released once paper is accepted.

**arXiv ID:** 2603.04900
</details>

<details>
<summary><strong>AMV-L: Lifecycle-Managed Agent Memory for Tail-Latency Control in Long-Running LLM Systems</strong> - Emmanuel Bamidele - [[pdf]](https://arxiv.org/pdf/2603.04443)</summary>

**Abstract:** Long-running LLM agents require persistent memory to preserve state across interactions, yet most deployed systems manage memory with age-based retention (e.g., TTL). While TTL bounds item lifetime, it does not bound the computational footprint of memory on the request path: as retained items accumulate, retrieval candidate sets and vector similarity scans can grow unpredictably, yielding heavy-tailed latency and unstable throughput. We present AMV-L (Adaptive Memory Value Lifecycle), a memory-management framework that treats agent memory as a managed systems resource. AMV-L assigns each memory item a continuously updated utility score and uses value-driven promotion, demotion, and eviction to maintain lifecycle tiers; retrieval is restricted to a bounded, tier-aware candidate set that decouples the request-path working set from total retained memory. We implement AMV-L in a full-stack LLM serving system and evaluate it under identical long-running workloads against two baselines: TTL and an LRU working-set policy, with fixed prompt-injection caps. Relative to TTL, AMV-L improves throughput by 3.1x and reduces latency by 4.2x (median), 4.7x (p95), and 4.4x (p99), while reducing the fraction of requests exceeding 2s from 13.8% to 0.007%. Compared to LRU, AMV-L trades a small regression in median/p95 latency (+26% / +3%) for improved extreme-tail behavior (-15% p99; -98% >2s) and lower token overhead (approximately 6% fewer tokens/request), while matching retrieval quality (value means within approximately 0-2%). The gains arise primarily from bounding retrieval-set size and vector-search work, not from shortening prompts. Our results show that predictable performance for long-running LLM agents requires explicit control of memory working-set size and value-driven lifecycle management, rather than retention time alone.

**arXiv ID:** 2603.04443
</details>

<details>
<summary><strong>DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval</strong> - Maojun Sun, Yue Wu, Yifei Xie, Ruijian Han, Binyan Jiang, Defeng Sun, Yancheng Yuan, Jian Huang - [[pdf]](https://arxiv.org/pdf/2603.04743)</summary>

**Abstract:** Large Language Model (LLM) agents can automate data-science workflows, but many rigorous statistical methods implemented in R remain underused because LLMs struggle with statistical knowledge and tool retrieval. Existing retrieval-augmented approaches focus on function-level semantics and ignore data distribution, producing suboptimal matches. We propose DARE (Distribution-Aware Retrieval Embedding), a lightweight, plug-and-play retrieval model that incorporates data distribution information into function representations for R package retrieval. Our main contributions are: (i) RPKB, a curated R Package Knowledge Base derived from 8,191 high-quality CRAN packages; (ii) DARE, an embedding model that fuses distributional features with function metadata to improve retrieval relevance; and (iii) RCodingAgent, an R-oriented LLM agent for reliable R code generation and a suite of statistical analysis tasks for systematically evaluating LLM agents in realistic analytical scenarios. Empirically, DARE achieves an NDCG at 10 of 93.47%, outperforming state-of-the-art open-source embedding models by up to 17% on package retrieval while using substantially fewer parameters. Integrating DARE into RCodingAgent yields significant gains on downstream analysis tasks. This work helps narrow the gap between LLM automation and the mature R statistical ecosystem.

**arXiv ID:** 2603.04743
</details>

<details>
<summary><strong>Towards Trustworthy Legal AI through LLM Agents and Formal Reasoning</strong> - Linze Chen, Yufan Cai, Zhe Hou, Jin Song Dong - [[pdf]](https://arxiv.org/pdf/2511.21033)</summary>

**Abstract:** Legal decisions should be logical and based on statutory laws. While large language models(LLMs) are good at understanding legal text, they cannot provide verifiable justifications. We present L4L, a solver-centric framework that enforces formal alignment between LLM-based legal reasoning and statutory laws. The framework integrates role-differentiated LLM agents with SMT-backed verification, combining the flexibility of natural language with the rigor of symbolic reasoning. Our approach operates in four stages: (1) Statute Knowledge Building, where LLMs autoformalize legal provisions into logical constraints and validate them through case-level testing; (2) Dual Fact-and-Statute Extraction, in which the prosecutor-and defense-aligned agents independently map case narratives to argument tuples; (3) Solver-Centric Adjudication, where SMT solvers check the legal admissibility and consistency of the arguments against the formalized statute knowledge; (4) Judicial Rendering, in which a judge agent integrates solver-validated reasoning with statutory interpretation and similar precedents to produce a legally grounded verdict. Experiments on public legal benchmarks show that L4L consistently outperforms baselines, while providing auditable justifications that enable trustworthy legal AI.

**arXiv ID:** 2511.21033
</details>

<details>
<summary><strong>Achieving Olympia-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning</strong> - Haiteng Zhao, Junhao Shen, Yiming Zhang, Songyang Gao, Kuikun Liu, Tianyou Ma, Fan Zheng, Dahua Lin, Wenwei Zhang, Kai Chen - [[pdf]](https://arxiv.org/pdf/2512.10534)</summary>

**Abstract:** Large language model (LLM) agents exhibit strong mathematical problem-solving abilities and can even solve International Mathematical Olympiad (IMO) level problems with the assistance of formal proof systems. However, due to weak heuristics for auxiliary constructions, AI for geometry problem solving remains dominated by expert models such as AlphaGeometry 2, which rely heavily on large-scale data synthesis and search for both training and evaluation. In this work, we make the first attempt to build a medalist-level LLM agent for geometry and present InternGeometry. InternGeometry overcomes the heuristic limitations in geometry by iteratively proposing propositions and auxiliary constructions, verifying them with a symbolic engine, and reflecting on the engine's feedback to guide subsequent proposals. A dynamic memory mechanism enables InternGeometry to conduct more than two hundred interactions with the symbolic engine per problem. To further accelerate learning, we introduce Complexity-Boosting Reinforcement Learning (CBRL), which gradually increases the complexity of synthesized problems across training stages. Built on InternThinker-32B, InternGeometry solves 44 of 50 IMO geometry problems (2000-2024), exceeding the average gold medalist score (40.9), using only 13K training examples, just 0.004% of the data used by AlphaGeometry 2, demonstrating the potential of LLM agents on expert-level geometry tasks. InternGeometry can also propose novel auxiliary constructions for IMO problems that do not appear in human solutions.

**arXiv ID:** 2512.10534
</details>

<details>
<summary><strong>Act-Observe-Rewrite: Multimodal Coding Agents as In-Context Policy Learners for Robot Manipulation</strong> - Vaishak Kumar - [[pdf]](https://arxiv.org/pdf/2603.04466)</summary>

**Abstract:** Can a multimodal language model learn to manipulate physical objects by reasoning about its own failures-without gradient updates, demonstrations, or reward engineering? We argue the answer is yes, under conditions we characterise precisely. We present Act-Observe-Rewrite (AOR), a framework in which an LLM agent improves a robot manipulation policy by synthesising entirely new executable Python controller code between trials, guided by visual observations and structured episode outcomes. Unlike prior work that grounds LLMs in pre-defined skill libraries or uses code generation for one-shot plan synthesis, AOR makes the full low-level motor control implementation the unit of LLM reasoning, enabling the agent to change not just what the robot does, but how it does it. The central claim is that interpretable code as the policy representation creates a qualitatively different kind of in-context learning from opaque neural policies: the agent can diagnose systematic failures and rewrite their causes. We validate this across three robosuite manipulation tasks and report promising results, with the agent achieving high success rates without demonstrations, reward engineering, or gradient updates.

**arXiv ID:** 2603.04466
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (24 papers)</h2></summary>

<details>
<summary><strong>Discovering mathematical concepts through a multi-agent system</strong> - Daattavya Aggarwal, Oisin Kim, Carl Henrik Ek, Challenger Mishra - [[pdf]](https://arxiv.org/pdf/2603.04528)</summary>

**Abstract:** Mathematical concepts emerge through an interplay of processes, including experimentation, efforts at proof, and counterexamples. In this paper, we present a new multi-agent model for computational mathematical discovery based on this observation. Our system, conceived with research in mind, poses its own conjectures and then attempts to prove them, making decisions informed by this feedback and an evolving data distribution. Inspired by the history of Euler's conjecture for polyhedra and an open challenge in the literature, we benchmark with the task of autonomously recovering the concept of homology from polyhedral data and knowledge of linear algebra. Our system completes this learning problem. Most importantly, the experiments are ablations, statistically testing the value of the complete dynamic and controlling for experimental setup. They support our main claim: that the optimisation of the right combination of local processes can lead to surprisingly well-aligned notions of mathematical interestingness.

**arXiv ID:** 2603.04528
</details>

<details>
<summary><strong>HiMAP-Travel: Hierarchical Multi-Agent Planning for Long-Horizon Constrained Travel</strong> - Viet Bui, Wenjun Li, Yong Liu - [[pdf]](https://arxiv.org/pdf/2603.04750)</summary>

**Abstract:** Sequential LLM agents fail on long-horizon planning with hard constraints like budgets and diversity requirements. As planning progresses and context grows, these agents drift from global constraints. We propose HiMAP-Travel, a hierarchical multi-agent framework that splits planning into strategic coordination and parallel day-level execution. A Coordinator allocates resources across days, while Day Executors plan independently in parallel. Three key mechanisms enable this: a transactional monitor enforcing budget and uniqueness constraints across parallel agents, a bargaining protocol allowing agents to reject infeasible sub-goals and trigger re-planning, and a single policy trained with GRPO that powers all agents through role conditioning. On TravelPlanner, HiMAP-Travel with Qwen3-8B achieves 52.78% validation and 52.65% test Final Pass Rate (FPR). In a controlled comparison with identical model, training, and tools, it outperforms the sequential DeepTravel baseline by +8.67~pp. It also surpasses ATLAS by +17.65~pp and MTP by +10.0~pp. On FlexTravelBench multi-turn scenarios, it achieves 44.34% (2-turn) and 37.42% (3-turn) FPR while reducing latency 2.5x through parallelization.

**arXiv ID:** 2603.04750
</details>

<details>
<summary><strong>Alignment Backfire: Language-Dependent Reversal of Safety Interventions Across 16 Languages in LLM Multi-Agent Systems</strong> - Hiroki Fukui - [[pdf]](https://arxiv.org/pdf/2603.04904)</summary>

**Abstract:** In perpetrator treatment, a recurring observation is the dissociation between insight and action: offenders articulate remorse yet behavioral change does not follow. We report four preregistered studies (1,584 multi-agent simulations across 16 languages and three model families) demonstrating that alignment interventions in large language models produce a structurally analogous phenomenon: surface safety that masks or generates collective pathology and internal dissociation. In Study 1 (N = 150), increasing alignment-instructed agents reduced collective pathology in English (g = -1.844, p < .0001) but amplified it in Japanese (g = +0.771, p = .038)--a directional reversal we term "alignment backfire." Study 2 (N = 1,174) extended to 16 languages: alignment-induced dissociation was near-universal (15/16 languages; beta = 0.0667, p < .0001), while collective pathology bifurcated along cultural-linguistic lines (interaction beta = 0.0684, p = .0003), correlating with Power Distance Index (r = 0.474, p = .064). Study 3 (N = 180) tested individuation as countermeasure; individuated agents became the primary source of both pathology and dissociation (DI = +1.120) with conformity above 84%--demonstrating iatrogenesis. Study 4 (N = 80) validated patterns across Llama 3.3 70B, GPT-4o-mini, and Qwen3-Next-80B-A3B, confirming English safety is model-general while Japanese backfire is model-specific. These findings reframe alignment as a behavioral intervention subject to risk homeostasis and iatrogenesis. Language space--the linguistic, pragmatic, and cultural properties inherited from training data--structurally determines alignment outcomes. Safety validated in English does not transfer to other languages, and prompt-level interventions cannot override language-space-level constraints.

**arXiv ID:** 2603.04904
</details>

<details>
<summary><strong>BioLLMAgent: A Hybrid Framework with Enhanced Structural Interpretability for Simulating Human Decision-Making in Computational Psychiatry</strong> - Zuo Fei, Kezhi Wang, Xiaomin Chen, Yizhou Huang - [[pdf]](https://arxiv.org/pdf/2603.05016)</summary>

**Abstract:** Computational psychiatry faces a fundamental trade-off: traditional reinforcement learning (RL) models offer interpretability but lack behavioral realism, while large language model (LLM) agents generate realistic behaviors but lack structural interpretability. We introduce BioLLMAgent, a novel hybrid framework that combines validated cognitive models with the generative capabilities of LLMs. The framework comprises three core components: (i) an Internal RL Engine for experience-driven value learning; (ii) an External LLM Shell for high-level cognitive strategies and therapeutic interventions; and (iii) a Decision Fusion Mechanism for integrating components via weighted utility. Comprehensive experiments on the Iowa Gambling Task (IGT) across six clinical and healthy datasets demonstrate that BioLLMAgent accurately reproduces human behavioral patterns while maintaining excellent parameter identifiability (correlations $>0.67$). Furthermore, the framework successfully simulates cognitive behavioral therapy (CBT) principles and reveals, through multi-agent dynamics, that community-wide educational interventions may outperform individual treatments. Validated across reward-punishment learning and temporal discounting tasks, BioLLMAgent provides a structurally interpretable "computational sandbox" for testing mechanistic hypotheses and intervention strategies in psychiatric research.

**arXiv ID:** 2603.05016
</details>

<details>
<summary><strong>S5-SHB Agent: Society 5.0 enabled Multi-model Agentic Blockchain Framework for Smart Home</strong> - Janani Rangila, Akila Siriweera, Incheon Paik, Keitaro Naruse, Isuru Jayanada, Vishmika Devindi - [[pdf]](https://arxiv.org/pdf/2603.05027)</summary>

**Abstract:** The smart home is a key application domain within the Society 5.0 vision for a human-centered society. As smart home ecosystems expand with heterogeneous IoT protocols, diverse devices, and evolving threats, autonomous systems must manage comfort, security, energy, and safety for residents. Such autonomous decision-making requires a trust anchor, making blockchain a preferred foundation for transparent and accountable smart home governance. However, realizing this vision requires blockchain-governed smart homes to simultaneously address adaptive consensus, intelligent multi-agent coordination, and resident-controlled governance aligned with the principles of Society 5.0. Existing frameworks rely solely on rigid smart contracts with fixed consensus protocols, employ at most a single AI model without multi-agent coordination, and offer no governance mechanism for residents to control automation behaviour. To address these limitations, this paper presents the Society 5.0-driven human-centered governance-enabled smart home blockchain agent (S5-SHB-Agent). The framework orchestrates ten specialized agents using interchangeable large language models to make decisions across the safety, security, comfort, energy, privacy, and health domains. An adaptive PoW blockchain adjusts mining difficulty based on transaction volume and emergency conditions, with digital signatures and Merkle tree anchoring to ensure tamper evident auditability. A four-tier governance model enables residents to control automation through tiered preferences from routine adjustments to immutable safety thresholds. Evaluation confirms that resident governance correctly separates adjustable comfort priorities from immutable safety thresholds across all tested configurations, while adaptive consensus commits emergency blocks.

**arXiv ID:** 2603.05027
</details>

<details>
<summary><strong>Bidirectional Curriculum Generation: A Multi-Agent Framework for Data-Efficient Mathematical Reasoning</strong> - Boren Hu, Xiao Liu, Boci Peng, Xinping Zhao, Xiaoran Shang, Yun Zhu, Lijun Wu - [[pdf]](https://arxiv.org/pdf/2603.05120)</summary>

**Abstract:** Enhancing mathematical reasoning in Large Language Models typically demands massive datasets, yet data efficiency remains a critical bottleneck. While Curriculum Learning attempts to structure this process, standard unidirectional approaches (simple-to-complex) suffer from inefficient sample utilization: they blindly escalate complexity even when foundational gaps persist, leading to wasted computation on unsolvable problems. To maximize the instructional value of every training sample, we introduce a novel Bidirectional Curriculum Generation framework. Unlike rigid trajectories, our multi-agent ecosystem mimics adaptive pedagogy to establish a closed feedback loop. It dynamically generates data by either complicating problems to challenge the model or, crucially, simplying them to repair specific reasoning failures. This mechanism ensures that the model consumes only the most effective data at any given stage. Grounded in the Optimal Pacing Theorem, our approach optimizes the learning trajectory, significantly outperforming baselines while achieving superior reasoning performance with substantially fewer instruction samples.

**arXiv ID:** 2603.05120
</details>

<details>
<summary><strong>Do Mixed-Vendor Multi-Agent LLMs Improve Clinical Diagnosis?</strong> - Grace Chang Yuan, Xiaoman Zhang, Sung Eun Kim, Pranav Rajpurkar - [[pdf]](https://arxiv.org/pdf/2603.04421)</summary>

**Abstract:** Multi-agent large language model (LLM) systems have emerged as a promising approach for clinical diagnosis, leveraging collaboration among agents to refine medical reasoning. However, most existing frameworks rely on single-vendor teams (e.g., multiple agents from the same model family), which risk correlated failure modes that reinforce shared biases rather than correcting them. We investigate the impact of vendor diversity by comparing Single-LLM, Single-Vendor, and Mixed-Vendor Multi-Agent Conversation (MAC) frameworks. Using three doctor agents instantiated with o4-mini, Gemini-2.5-Pro, and Claude-4.5-Sonnet, we evaluate performance on RareBench and DiagnosisArena. Mixed-vendor configurations consistently outperform single-vendor counterparts, achieving state-of-the-art recall and accuracy. Overlap analysis reveals the underlying mechanism: mixed-vendor teams pool complementary inductive biases, surfacing correct diagnoses that individual models or homogeneous teams collectively miss. These results highlight vendor diversity as a key design principle for robust clinical diagnostic systems.

**arXiv ID:** 2603.04421
</details>

<details>
<summary><strong>Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices</strong> - Yakov Pyotr Shkolnikov - [[pdf]](https://arxiv.org/pdf/2603.04428)</summary>

**Abstract:** Multi-agent LLM systems on edge devices face a memory management problem: device RAM is too small to hold every agent's KV cache simultaneously. On Apple M4 Pro with 10.2 GB of cache budget, only 3 agents fit at 8K context in FP16. A 10-agent workflow must constantly evict and reload caches. Without persistence, every eviction forces a full re-prefill through the model -- 15.7 seconds per agent at 4K context. We address this by persisting each agent's KV cache to disk in 4-bit quantized format and reloading it directly into the attention layer, eliminating redundant O(n) prefill computation via direct cache restoration. The system comprises three components: a block pool providing per-agent isolated Q4 KV caches in safetensors format, a BatchQuantizedKVCache for concurrent inference over multiple agents' quantized caches, and cross-phase context injection that accumulates attention state across conversation phases without re-computation. Evaluated on three architectures (Gemma 3 12B, dense GQA, 48 layers; DeepSeek-Coder-V2-Lite 16B, MoE MLA, 27 layers; Llama 3.1 8B, dense GQA, 32 layers), cache restoration reduces time-to-first-token by up to 136x (Gemma: 22--136x at 4K--32K; DeepSeek: 11--76x at 4K--32K; Llama: 24--111x at 4K--16K; 3--10x at 1K). Q4 quantization fits 4x more agent contexts into fixed device memory than FP16. Perplexity measured with actual Q4 KV caches shows -0.7% for Gemma, +2.8% for Llama, and +3.0% for DeepSeek. Open-source at this https URL

**arXiv ID:** 2603.04428
</details>

<details>
<summary><strong>From Spark to Fire: Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent Collaboration</strong> - Yizhe Xie, Congcong Zhu, Xinyue Zhang, Tianqing Zhu, Dayong Ye, Minfeng Qi, Huajie Chen, Wanlei Zhou - [[pdf]](https://arxiv.org/pdf/2603.04474)</summary>

**Abstract:** Large Language Model-based Multi-Agent Systems (LLM-MAS) are increasingly applied to complex collaborative scenarios. However, their collaborative mechanisms may cause minor inaccuracies to gradually solidify into system-level false consensus through iteration. Such risks are difficult to trace since errors can propagate and amplify through message dependencies. Existing protections often rely on single-agent validation or require modifications to the collaboration architecture, which can weaken effective information flow and may not align with natural collaboration processes in real tasks. To address this, we propose a propagation dynamics model tailored for LLM-MAS that abstracts collaboration as a directed dependency graph and provides an early-stage risk criterion to characterize amplification risk. Through experiments on six mainstream frameworks, we identify three vulnerability classes: cascade amplification, topological sensitivity, and consensus inertia. We further instantiate an attack where injecting just a single atomic error seed leads to widespread failure. In response, we introduce a genealogy-graph-based governance layer, implemented as a message-layer plugin, that suppresses both endogenous and exogenous error amplification without altering the collaboration architecture. Experiments show that this approach raises the defense success rate from a baseline of 0.32 to over 0.89 and significantly mitigates the cascading spread of minor errors.

**arXiv ID:** 2603.04474
</details>

<details>
<summary><strong>GIANT - Global Path Integration and Attentive Graph Networks for Multi-Agent Trajectory Planning</strong> - Jonas le Fevre Sejersen, Toyotaro Suzumura, Erdal Kayacan - [[pdf]](https://arxiv.org/pdf/2603.04659)</summary>

**Abstract:** This paper presents a novel approach to multi-robot collision avoidance that integrates global path planning with local navigation strategies, utilizing attentive graph neural networks to manage dynamic interactions among agents. We introduce a local navigation model that leverages pre-planned global paths, allowing robots to adhere to optimal routes while dynamically adjusting to environmental changes. The models robustness is enhanced through the introduction of noise during training, resulting in superior performance in complex, dynamic environments. Our approach is evaluated against established baselines, including NH-ORCA, DRL-NAV, and GA3C-CADRL, across various structurally diverse simulated scenarios. The results demonstrate that our model achieves consistently higher success rates, lower collision rates, and more efficient navigation, particularly in challenging scenarios where baseline models struggle. This work offers an advancement in multi-robot navigation, with implications for robust performance in complex, dynamic environments with varying degrees of complexity, such as those encountered in logistics, where adaptability is essential for accommodating unforeseen obstacles and unpredictable changes.

**arXiv ID:** 2603.04659
</details>

<details>
<summary><strong>SCoUT: Scalable Communication via Utility-Guided Temporal Grouping in Multi-Agent Reinforcement Learning</strong> - Manav Vora, Gokul Puthumanaillam, Hiroyasu Tsukamoto, Melkior Ornik - [[pdf]](https://arxiv.org/pdf/2603.04833)</summary>

**Abstract:** Communication can improve coordination in partially observed multi-agent reinforcement learning (MARL), but learning \emph{when} and \emph{who} to communicate with requires choosing among many possible sender-recipient pairs, and the effect of any single message on future reward is hard to isolate. We introduce \textbf{SCoUT} (\textbf{S}calable \textbf{Co}mmunication via \textbf{U}tility-guided \textbf{T}emporal grouping), which addresses both these challenges via temporal and agent abstraction within traditional MARL. During training, SCoUT resamples \textit{soft} agent groups every \(K\) environment steps (macro-steps) via Gumbel-Softmax; these groups are latent clusters that induce an affinity used as a differentiable prior over recipients. Using the same assignments, a group-aware critic predicts values for each agent group and maps them to per-agent baselines through the same soft assignments, reducing critic complexity and variance. Each agent is trained with a three-headed policy: environment action, send decision, and recipient selection. To obtain precise communication learning signals, we derive counterfactual communication advantages by analytically removing each sender's contribution from the recipient's aggregated messages. This counterfactual computation enables precise credit assignment for both send and recipient-selection decisions. At execution time, all centralized training components are discarded and only the per-agent policy is run, preserving decentralized execution. Project website, videos and code: \hyperlink{this https URL}{this https URL}

**arXiv ID:** 2603.04833
</details>

<details>
<summary><strong>Foam-Agent: Towards Automated Intelligent CFD Workflows</strong> - Ling Yue, Nithin Somasekharan, Tingwen Zhang, Yadi Cao, Zhangze Chen, Shimin Di, Shaowu Pan - [[pdf]](https://arxiv.org/pdf/2505.04997)</summary>

**Abstract:** Computational fluid dynamics (CFD) has been the main workhorse of computational physics. Yet its steep learning curve and fragmented, multi-stage workflow create significant barriers. To address these challenges, we present Foam-Agent, a multi-agent framework leveraging large language models (LLMs) to automate the end-to-end CFD workflow from a single natural language prompt. Foam-Agent orchestrates the comprehensive simulation workflow from mesh generation and high-performance computing job scripting to post-processing visualization. The system integrates retrieval-augmented generation with dependency-aware scheduling to synthesize high-fidelity simulation configurations. Furthermore, Foam-Agent adopts the Model Context Protocol to expose its core functions as discrete, callable tools. This allows for flexible integration and use by any other agentic systems. Evaluated on 110 simulation tasks, Foam-Agent achieved a state-of-the-art execution success rate of 88.2% without expert intervention. These results demonstrate how specialized multi-agent systems can effectively reduce expertise barriers and streamline complex fluid simulations.

**arXiv ID:** 2505.04997
</details>

<details>
<summary><strong>ClinNoteAgents: An LLM Multi-Agent System for Predicting and Interpreting Heart Failure 30-Day Readmission from Clinical Notes</strong> - Rongjia Zhou, Chengzhuo Li, Carl Yang, Jiaying Lu - [[pdf]](https://arxiv.org/pdf/2512.07081)</summary>

**Abstract:** Heart failure (HF) is one of the leading causes of rehospitalization among older adults in the United States. Although clinical notes contain rich, detailed patient information and make up a large portion of electronic health records (EHRs), they remain underutilized for HF readmission risk analysis. Traditional computational models for HF readmission often rely on expert-crafted rules, medical thesauri, and ontologies to interpret clinical notes, which are typically written under time pressure and may contain misspellings, abbreviations, and domain-specific jargon. We present ClinNoteAgents, an LLM-based multi-agent framework that transforms free-text clinical notes into (1) structured representations of clinical and social risk factors for association analysis and (2) clinician-style abstractions for HF 30-day readmission prediction. We evaluate ClinNoteAgents on 3,544 notes from 2,065 patients (readmission rate=35.16%), demonstrating high extraction fidelity for clinical variables (conditional accuracy >= 90% for multiple vitals), key risk factor identification, and preservation of predictive signal despite 60 to 90% text reduction. By reducing reliance on structured fields and minimizing manual annotation and model training, ClinNoteAgents provides a scalable and interpretable approach to note-based HF readmission risk modeling in data-limited healthcare systems.

**arXiv ID:** 2512.07081
</details>

<details>
<summary><strong>Strategic Interactions in Multi-Level Stackelberg Games with Non-Follower Agents and Heterogeneous Leaders</strong> - Niloofar Aminikalibar, Farzaneh Farhadi, Maria Chli - [[pdf]](https://arxiv.org/pdf/2603.04628)</summary>

**Abstract:** Strategic interaction in congested systems is commonly modelled using Stackelberg games, where competing leaders anticipate the behaviour of self-interested followers. A key limitation of existing models is that they typically ignore agents who do not directly participate in market competition, yet both contribute to and adapt to congestion. Although such non-follower agents do not generate revenue or respond to market incentives, their behaviour reshapes congestion patterns, which in turn affects the decisions of leaders and followers through shared resources.
We argue that overlooking non-followers leads to systematically distorted equilibrium predictions in congestion-coupled markets. To address this, we introduce a three-level Stackelberg framework with heterogeneous leaders differing in decision horizons and feasible actions, strategic followers, and non-follower agents that captures bidirectional coupling between infrastructure decisions, competition, and equilibrium congestion.
We instantiate the framework in the context of electric vehicle (EV) charging infrastructure, where charging providers compete with rivals, while EV and non-EV traffic jointly shape congestion. The model illustrates how explicitly accounting for non-followers and heterogeneous competitors qualitatively alters strategic incentives and equilibrium outcomes. Beyond EV charging, the framework applies to a broad class of congestion-coupled multi-agent systems in mobility, energy, and computing markets.

**arXiv ID:** 2603.04628
</details>

<details>
<summary><strong>Dual-Interaction-Aware Cooperative Control Strategy for Alleviating Mixed Traffic Congestion</strong> - Zhengxuan Liu, Yuxin Cai, Yijing Wang, Xiangkun He, Chen Lv, Zhiqiang Zuo - [[pdf]](https://arxiv.org/pdf/2603.03848)</summary>

**Abstract:** As Intelligent Transportation System (ITS) develops, Connected and Automated Vehicles (CAVs) are expected to significantly reduce traffic congestion through cooperative strategies, such as in bottleneck areas. However, the uncertainty and diversity in the behaviors of Human-Driven Vehicles (HDVs) in mixed traffic environments present major challenges for CAV cooperation. This paper proposes a Dual-Interaction-Aware Cooperative Control (DIACC) strategy that enhances both local and global interaction perception within the Multi-Agent Reinforcement Learning (MARL) framework for Connected and Automated Vehicles (CAVs) in mixed traffic bottleneck scenarios. The DIACC strategy consists of three key innovations: 1) A Decentralized Interaction-Adaptive Decision-Making (D-IADM) module that enhances actor's local interaction perception by distinguishing CAV-CAV cooperative interactions from CAV-HDV observational interactions. 2) A Centralized Interaction-Enhanced Critic (C-IEC) that improves critic's global traffic understanding through interaction-aware value estimation, providing more accurate guidance for policy updates. 3) A reward design that employs softmin aggregation with temperature annealing to prioritize interaction-intensive scenarios in mixed traffic. Additionally, a lightweight Proactive Safety-based Action Refinement (PSAR) module applies rule-based corrections to accelerate training convergence. Experimental results demonstrate that DIACC significantly improves traffic efficiency and adaptability compared to rule-based and benchmark MARL models.

**arXiv ID:** 2603.03848
</details>

<details>
<summary><strong>Beyond Input Guardrails: Reconstructing Cross-Agent Semantic Flows for Execution-Aware Attack Detection</strong> - Yangyang Wei, Yijie Xu, Zhenyuan Li, Xiangmin Shen, Shouling Ji - [[pdf]](https://arxiv.org/pdf/2603.04469)</summary>

**Abstract:** Multi-Agent System is emerging as the \textit{de facto} standard for complex task orchestration. However, its reliance on autonomous execution and unstructured inter-agent communication introduces severe risks, such as indirect prompt injection, that easily circumvent conventional input guardrails. To address this, we propose \SysName, a framework that shifts the defensive paradigm from static input filtering to execution-aware analysis. By extracting and reconstructing Cross-Agent Semantic Flows, \SysName synthesizes fragmented operational primitives into contiguous behavioral trajectories, enabling a holistic view of system activity. We leverage a Supervisor LLM to scrutinize these trajectories, identifying anomalies across data flow violations, control flow deviations, and intent inconsistencies. Empirical evaluations demonstrate that \SysName effectively detects over ten distinct compound attack vectors, achieving F1-scores of 85.3\% and 66.7\% for node-level and path-level end-to-end attack detection, respectively. The source code is available at this https URL.

**arXiv ID:** 2603.04469
</details>

<details>
<summary><strong>Conflict-Based Search as a Protocol: A Multi-Agent Motion Planning Protocol for Heterogeneous Agents, Solvers, and Independent Tasks</strong> - Rishi Veerapaneni, Alvin Tang, Haodong He, Sophia Zhao, Viraj Shah, Yidai Cen, Ziteng Ji, Gabriel Olin, Jon Arrizabalaga, Yorai Shaoul, Jiaoyang Li, Maxim Likhachev - [[pdf]](https://arxiv.org/pdf/2510.00425)</summary>

**Abstract:** Imagine the future construction site, hospital, or office with dozens of robots bought from different manufacturers. How can we enable these different robots to effectively move in a shared environment, given that each robot may have its own independent motion planning system? This work shows how we can get efficient collision-free movements between algorithmically heterogeneous agents by using Conflict-Based Search (Sharon et al. 2015) as a protocol. At its core, the CBS Protocol requires one specific single-agent motion planning API; finding a collision-free path that satisfies certain space-time constraints. Given such an API, CBS uses a central planner to find collision-free paths - independent of how the API is implemented. We demonstrate how this protocol enables multi-agent motion planning for a heterogeneous team of agents completing independent tasks with a variety of single-agent planners including: Heuristic Search (e.g., A*), Sampling Based Search (e.g., RRT), Optimization (e.g., Direct Collocation), Diffusion, and Reinforcement Learning.

**arXiv ID:** 2510.00425
</details>

<details>
<summary><strong>GRAND: Guidance, Rebalancing, and Assignment for Networked Dispatch in Multi-Agent Path Finding</strong> - Johannes Gaber, Meshal Alharbi, Daniele Gammelli, Gioele Zardini - [[pdf]](https://arxiv.org/pdf/2512.03194)</summary>

**Abstract:** Large robot fleets are now common in warehouses and other logistics settings, where small control gains translate into large operational impacts. In this article, we address task scheduling for lifelong Multi-Agent Pickup-and-Delivery (MAPD) and propose a hybrid method that couples learning-based global guidance with lightweight optimization. A graph neural network policy trained via reinforcement learning outputs a desired distribution of free agents over an aggregated warehouse graph. This signal is converted into region-to-region rebalancing through a minimum-cost flow, and finalized by small, local assignment problems, preserving accuracy while keeping per-step latency within a 1 s compute budget. We call this approach GRAND: a hierarchical algorithm that relies on Guidance, Rebalancing, and Assignment to explicitly leverage the workspace Network structure and Dispatch agents to tasks. On congested warehouse benchmarks from the League of Robot Runners (LoRR) with up to 500 agents, our approach improves throughput by up to 10% over the 2024 winning scheduler while maintaining real-time execution. The results indicate that coupling graph-structured learned guidance with tractable solvers reduces congestion and yields a practical, scalable blueprint for high-throughput scheduling in large fleets.

**arXiv ID:** 2512.03194
</details>

<details>
<summary><strong>TritonDFT: Automating DFT with a Multi-Agent Framework</strong> - Zhengding Hu, Kuntal Talit, Zhen Wang, Haseeb Ahmad, Yichen Lin, Prabhleen Kaur, Christopher Lane, Elizabeth A. Peterson, Zhiting Hu, Elizabeth A. Nowadnick, Yufei Ding - [[pdf]](https://arxiv.org/pdf/2603.03372)</summary>

**Abstract:** Density Functional Theory (DFT) is a cornerstone of materials science, yet executing DFT in practice requires coordinating a complex, multi-step workflow. Existing tools and LLM-based solutions automate parts of the steps, but lack support for full workflow automation, diverse task adaptation, and accuracy-cost trade-off optimization in DFT configuration. To this end, we present TritonDFT, a multi-agent framework that enables efficient and accurate DFT execution through an expert-curated, extensible workflow design, Pareto-aware parameter inference, and multi-source knowledge augmentation. We further introduce DFTBench, a benchmark for evaluating the agent's multi-dimensional capabilities, spanning science expertise, trade0off optimization, HPC knowledge, and cost efficiency. TritonDFT provides an open user interface for real-world usage. Our website is at this https URL. Our source code and benchmark suite are available at this https URL.

**arXiv ID:** 2603.03372
</details>

<details>
<summary><strong>Optimizing What We Trust: Reliability-Guided QUBO Selection of Multi-Agent Weak Framing Signals for Arabic Sentiment Prediction</strong> - Rabab Alkhalifa - [[pdf]](https://arxiv.org/pdf/2603.04416)</summary>

**Abstract:** Framing detection in Arabic social media is difficult due to interpretive ambiguity, cultural grounding, and limited reliable supervision. Existing LLM-based weak supervision methods typically rely on label aggregation, which is brittle when annotations are few and socially dependent. We propose a reliability-aware weak supervision framework that shifts the focus from label fusion to data curation. A small multi-agent LLM pipeline, two framers, a critic, and a discriminator, treats disagreement and reasoning quality as epistemic signals and produces instance-level reliability estimates. These estimates guide a QUBO-based subset selection procedure that enforces frame balance while reducing redundancy. Intrinsic diagnostics and an out-of-domain Arabic sentiment transfer test show that the selected subsets are more reliable and encode non-random, transferable structure, without degrading strong text-only baselines.

**arXiv ID:** 2603.04416
</details>

<details>
<summary><strong>HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents</strong> - Yilin Jiang, Fei Tan, Xuanyu Yin, Jing Leng, Aimin Zhou - [[pdf]](https://arxiv.org/pdf/2603.04855)</summary>

**Abstract:** Student Personas (SPs) are emerging as infrastructure for educational LLMs, yet prior work often relies on ad-hoc prompting or hand-crafted profiles with limited control over educational theory and population distributions. We formalize this as Theory-Aligned and Distribution-Controllable Persona Generation (TAD-PG) and introduce HACHIMI, a multi-agent Propose-Validate-Revise framework that generates theory-aligned, quota-controlled personas. HACHIMI factorizes each persona into a theory-anchored educational schema, enforces developmental and psychological constraints via a neuro-symbolic validator, and combines stratified sampling with semantic deduplication to reduce mode collapse. The resulting HACHIMI-1M corpus comprises 1 million personas for Grades 1-12. Intrinsic evaluation shows near-perfect schema validity, accurate quotas, and substantial diversity, while external evaluation instantiates personas as student agents answering CEPS and PISA 2022 surveys; across 16 cohorts, math and curiosity/growth constructs align strongly between humans and agents, whereas classroom-climate and well-being constructs are only moderately aligned, revealing a fidelity gradient. All personas are generated with Qwen2.5-72B, and HACHIMI provides a standardized synthetic student population for group-level benchmarking and social-science simulations. Resources available at this https URL

**arXiv ID:** 2603.04855
</details>

<details>
<summary><strong>INMS: Memory Sharing for Large Language Model based Agents</strong> - Hang Gao, Yongfeng Zhang - [[pdf]](https://arxiv.org/pdf/2404.09982)</summary>

**Abstract:** While Large Language Model (LLM) based agents excel at complex tasks, their performance in open-ended scenarios is often constrained by isolated operation and reliance on static databases, missing the dynamic knowledge exchange of human dialogue. To bridge this gap, we propose the INteractive Memory Sharing (INMS) framework, an asynchronous interaction paradigm for multi-agent systems. By integrating real-time memory filtering, storage, and retrieval, INMS establishes a shared conversational memory pool. This enables continuous, dialogue-like memory sharing among agents, promoting collective self-enhancement and dynamically refining the retrieval mediator based on interaction history. Extensive experiments across three datasets demonstrate that INMS significantly improves agent performance by effectively modeling multi-agent interaction and collective knowledge sharing.

**arXiv ID:** 2404.09982
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning in Intelligent Transportation Systems: A Comprehensive Survey</strong> - Rexcharles Donatus, Kumater Ter, Daniel Udekwe - [[pdf]](https://arxiv.org/pdf/2508.20315)</summary>

**Abstract:** The growing complexity of urban mobility and the demand for efficient, sustainable, and adaptive solutions have positioned Intelligent Transportation Systems (ITS) at the forefront of modern infrastructure innovation. At the core of ITS lies the challenge of autonomous decision-making across dynamic, large scale, and uncertain environments where multiple agents traffic signals, autonomous vehicles, or fleet units must coordinate effectively. Multi Agent Reinforcement Learning (MARL) offers a promising paradigm for addressing these challenges by enabling distributed agents to jointly learn optimal strategies that balance individual objectives with system wide efficiency. This paper presents a comprehensive survey of MARL applications in ITS. We introduce a structured taxonomy that categorizes MARL approaches according to coordination models and learning algorithms, spanning value based, policy based, actor critic, and communication enhanced frameworks. Applications are reviewed across key ITS domains, including traffic signal control, connected and autonomous vehicle coordination, logistics optimization, and mobility on demand systems. Furthermore, we highlight widely used simulation platforms such as SUMO, CARLA, and CityFlow that support MARL experimentation, along with emerging benchmarks. The survey also identifies core challenges, including scalability, non stationarity, credit assignment, communication constraints, and the sim to real transfer gap, which continue to hinder real world deployment.

**arXiv ID:** 2508.20315
</details>

<details>
<summary><strong>Breaking and Fixing Defenses Against Control-Flow Hijacking in Multi-Agent Systems</strong> - Rishi Jha, Harold Triedman, Justin Wagle, Vitaly Shmatikov - [[pdf]](https://arxiv.org/pdf/2510.17276)</summary>

**Abstract:** Control-flow hijacking attacks manipulate orchestration mechanisms in multi-agent systems into performing unsafe actions that compromise the system and exfiltrate sensitive information. Recently proposed defenses, such as LlamaFirewall, rely on alignment checks of inter-agent communications to ensure that all agent invocations are "related to" and "likely to further" the original objective.
We start by demonstrating control-flow hijacking attacks that evade these defenses even if alignment checks are performed by advanced LLMs. We argue that the safety and functionality objectives of multi-agent systems fundamentally conflict with each other. This conflict is exacerbated by the brittle definitions of "alignment" and the checkers' incomplete visibility into the execution context.
We then propose, implement, and evaluate ControlValve, a new defense inspired by the principles of control-flow integrity and least privilege. ControlValve (1) generates permitted control-flow graphs for multi-agent systems, and (2) enforces that all executions comply with these graphs, along with contextual rules (generated in a zero-shot manner) for each agent invocation.

**arXiv ID:** 2510.17276
</details>

</details>

<details open>
<summary><h2>Other Agent Research (15 papers)</h2></summary>

<details>
<summary><strong>Capability Thresholds and Manufacturing Topology: How Embodied Intelligence Triggers Phase Transitions in Economic Geography</strong> - Xinmin Fang, Lingfeng Tao, Zhengxiong Li - [[pdf]](https://arxiv.org/pdf/2603.04457)</summary>

**Abstract:** The fundamental topology of manufacturing has not undergone a paradigm-level transformation since Henry Ford's moving assembly line in 1913. Every major innovation of the past century, from the Toyota Production System to Industry 4.0, has optimized within the Fordist paradigm without altering its structural logic: centralized mega-factories, located near labor pools, producing at scale. We argue that embodied intelligence is poised to break this century-long stasis, not by making existing factories more efficient, but by triggering phase transitions in manufacturing economic geography itself. When embodied AI capabilities cross critical thresholds in dexterity, generalization, reliability, and tactile-vision fusion, the consequences extend far beyond cost reduction: they restructure where factories are built, how supply chains are organized, and what constitutes viable production scale. We formalize this by defining a Capability Space C = (d, g, r, t) and showing that the site-selection objective function undergoes topological reorganization when capability vectors cross critical surfaces. Through three pathways, weight inversion, batch collapse, and human-infrastructure decoupling, we show that embodied intelligence enables demand-proximal micro-manufacturing, eliminates "manufacturing deserts," and reverses geographic concentration driven by labor arbitrage. We further introduce Machine Climate Advantage: once human workers are removed, optimal factory locations are determined by machine-optimal conditions (low humidity, high irradiance, thermal stability), factors orthogonal to traditional siting logic, creating a production geography with no historical precedent. This paper establishes Embodied Intelligence Economics, the study of how physical AI capability thresholds reshape the spatial and structural logic of production.

**arXiv ID:** 2603.04457
</details>

<details>
<summary><strong>When Agents Persuade: Propaganda Generation and Mitigation in LLMs</strong> - Julia Jose, Ritik Roongta, Rachel Greenstadt - [[pdf]](https://arxiv.org/pdf/2603.04636)</summary>

**Abstract:** Despite their wide-ranging benefits, LLM-based agents deployed in open environments can be exploited to produce manipulative material. In this study, we task LLMs with propaganda objectives and analyze their outputs using two domain-specific models: one that classifies text as propaganda or non-propaganda, and another that detects rhetorical techniques of propaganda (e.g., loaded language, appeals to fear, flag-waving, name-calling). Our findings show that, when prompted, LLMs exhibit propagandistic behaviors and use a variety of rhetorical techniques in doing so. We also explore mitigation via Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and ORPO (Odds Ratio Preference Optimization). We find that fine-tuning significantly reduces their tendency to generate such content, with ORPO proving most effective.

**arXiv ID:** 2603.04636
</details>

<details>
<summary><strong>Building AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned</strong> - Nghi D. Q. Bui - [[pdf]](https://arxiv.org/pdf/2603.05344)</summary>

**Abstract:** The landscape of AI coding assistance is undergoing a fundamental shift from complex IDE plugins to versatile, terminal-native agents. Operating directly where developers manage source control, execute builds, and deploy environments, CLI-based agents offer unprecedented autonomy for long-horizon development tasks. In this paper, we present OPENDEV, an open-source, command-line coding agent engineered specifically for this new paradigm. Effective autonomous assistance requires strict safety controls and highly efficient context management to prevent context bloat and reasoning degradation. OPENDEV overcomes these challenges through a compound AI system architecture with workload-specialized model routing, a dual-agent architecture separating planning from execution, lazy tool discovery, and adaptive context compaction that progressively reduces older observations. Furthermore, it employs an automated memory system to accumulate project-specific knowledge across sessions and counteracts instruction fade-out through event-driven system reminders. By enforcing explicit reasoning phases and prioritizing context efficiency, OPENDEV provides a secure, extensible foundation for terminal-first AI assistance, offering a blueprint for robust autonomous software engineering.

**arXiv ID:** 2603.05344
</details>

<details>
<summary><strong>The Convergence of Schema-Guided Dialogue Systems and the Model Context Protocol</strong> - Andreas Schlapbach - [[pdf]](https://arxiv.org/pdf/2602.18764)</summary>

**Abstract:** This paper establishes a fundamental convergence: Schema-Guided Dialogue (SGD) and the Model Context Protocol (MCP) represent two manifestations of a unified paradigm for deterministic, auditable LLM-agent interaction. SGD, designed for dialogue-based API discovery (2019), and MCP, now the de facto standard for LLM-tool integration, share the same core insight -- that schemas can encode not just tool signatures but operational constraints and reasoning guidance. By analyzing this convergence, we extract five foundational principles for schema design: (1) Semantic Completeness over Syntactic Precision, (2) Explicit Action Boundaries, (3) Failure Mode Documentation, (4) Progressive Disclosure Compatibility, and (5) Inter-Tool Relationship Declaration. These principles reveal three novel insights: first, SGD's original design was fundamentally sound and should be inherited by MCP; second, both frameworks leave failure modes and inter-tool relationships unexploited -- gaps we identify and resolve; third, progressive disclosure emerges as a critical production-scaling insight under real-world token constraints. We provide concrete design patterns for each principle. These principles position schema-driven governance as a scalable mechanism for AI system oversight without requiring proprietary system inspection -- central to Software 3.0.

**arXiv ID:** 2602.18764
</details>

<details>
<summary><strong>Real-Time BDI Agents: a model and its implementation</strong> - Andrea Traldi, Francesco Bruschetti, Marco Robol, Davide Calvaresi, Marco Roveri, Paolo Giorgini - [[pdf]](https://arxiv.org/pdf/2205.00979)</summary>

**Abstract:** The BDI model proved to be effective for developing applications requiring high-levels of autonomy and to deal with the complexity and unpredictability of real-world scenarios. The model, however, has significant limitations in reacting and handling contingencies within the given real-time constraints. Without an explicit representation of time, existing real-time BDI implementations overlook the temporal implications during the agent's decision process that may result in delays or unresponsiveness of the system when it gets overloaded. In this paper, we redefine the BDI agent control loop inspired by well established algorithms for real-time systems to ensure a proper reaction of agents and their effective application in typical real-time domains. Our model proposes an effective real-time management of goals, plans, and actions with respect to time constraints and resources availability. We propose an implementation of the model for a resource-collection video-game and we validate the approach against a set of significant scenarios.

**arXiv ID:** 2205.00979
</details>

<details>
<summary><strong>AILS-NTUA at SemEval-2026 Task 10: Agentic LLMs for Psycholinguistic Marker Extraction and Conspiracy Endorsement Detection</strong> - Panagiotis Alexios Spanakis, Maria Lymperaiou, Giorgos Filandrianos, Athanasios Voulodimos, Giorgos Stamou - [[pdf]](https://arxiv.org/pdf/2603.04921)</summary>

**Abstract:** This paper presents a novel agentic LLM pipeline for SemEval-2026 Task 10 that jointly extracts psycholinguistic conspiracy markers and detects conspiracy endorsement. Unlike traditional classifiers that conflate semantic reasoning with structural localization, our decoupled design isolates these challenges. For marker extraction, we propose Dynamic Discriminative Chain-of-Thought (DD-CoT) with deterministic anchoring to resolve semantic ambiguity and character-level brittleness. For conspiracy detection, an "Anti-Echo Chamber" architecture, consisting of an adversarial Parallel Council adjudicated by a Calibrated Judge, overcomes the "Reporter Trap," where models falsely penalize objective reporting. Achieving 0.24 Macro F1 (+100\% over baseline) on S1 and 0.79 Macro F1 (+49\%) on S2, with the S1 system ranking 3rd on the development leaderboard, our approach establishes a versatile paradigm for interpretable, psycholinguistically-grounded NLP.

**arXiv ID:** 2603.04921
</details>

<details>
<summary><strong>U-Parking: Distributed UWB-Assisted Autonomous Parking System with Robust Localization and Intelligent Planning</strong> - Yiang Wu, Qiong Wu, Pingyi Fan, Kezhi Wang, Wen Chen, Guoqiang Mao, Khaled B. Letaief - [[pdf]](https://arxiv.org/pdf/2603.04898)</summary>

**Abstract:** This demonstration presents U-Parking, a distributed Ultra-Wideband (UWB)-assisted autonomous parking system. By integrating Large Language Models (LLMs)-assisted planning with robust fusion localization and trajectory tracking, it enables reliable automated parking in challenging indoor environments, as validated through real-vehicle demonstrations.

**arXiv ID:** 2603.04898
</details>

<details>
<summary><strong>Efficient Autonomous Navigation of a Quadruped Robot in Underground Mines on Edge Hardware</strong> - Yixiang Gao, Kwame Awuah-Offei - [[pdf]](https://arxiv.org/pdf/2603.04470)</summary>

**Abstract:** Embodied navigation in underground mines faces significant challenges, including narrow passages, uneven terrain, near-total darkness, GPS-denied conditions, and limited communication infrastructure. While recent learning-based approaches rely on GPU-accelerated inference and extensive training data, we present a fully autonomous navigation stack for a Boston Dynamics Spot quadruped robot that runs entirely on a low-power Intel NUC edge computer with no GPU and no network connectivity requirements. The system integrates LiDAR-inertial odometry, scan-matching localization against a prior map, terrain segmentation, and visibility-graph global planning with a velocity-regulated local path follower, achieving real-time perception-to-action at consistent control rates. After a single mapping pass of the environment, the system handles arbitrary goal locations within the known map without any environment-specific training or learned components. We validate the system through repeated field trials using four target locations of varying traversal difficulty in an experimental underground mine, accumulating over 700 m of fully autonomous traverse with a 100% success rate across all 20 trials (5 repetitions x 4 targets) and an overall Success weighted by Path Length (SPL) of 0.73 \pm 0.09.

**arXiv ID:** 2603.04470
</details>

<details>
<summary><strong>Distributed State Estimation for Vision-Based Cooperative Slung Load Transportation in GPS-Denied Environments</strong> - Jack R. Pence, Jackson Fezell, Jack W. Langelaan, Junyi Geng - [[pdf]](https://arxiv.org/pdf/2603.04571)</summary>

**Abstract:** Transporting heavy or oversized slung loads using rotorcraft has traditionally relied on single-aircraft systems, which limits both payload capacity and control authority. Cooperative multilift using teams of rotorcraft offers a scalable and efficient alternative, especially for infrequent but challenging "long-tail" payloads without the need of building larger and larger rotorcraft. Most prior multilift research assumes GPS availability, uses centralized estimation architectures, or relies on controlled laboratory motion-capture setups. As a result, these methods lack robustness to sensor loss and are not viable in GPS-denied or operationally constrained environments. This paper addresses this limitation by presenting a distributed and decentralized payload state estimation framework for vision-based multilift operations. Using onboard monocular cameras, each UAV detects a fiducial marker on the payload and estimates its relative pose. These measurements are fused via a Distributed and Decentralized Extended Information Filter (DDEIF), enabling robust and scalable estimation that is resilient to individual sensor dropouts. This payload state estimate is then used for closed-loop trajectory tracking control. Monte Carlo simulation results in Gazebo show the effectiveness of the proposed approach, including the effect of communication loss during flight.

**arXiv ID:** 2603.04571
</details>

<details>
<summary><strong>Selecting Spots by Explicitly Predicting Intention from Motion History Improves Performance in Autonomous Parking</strong> - Long Kiu Chung, David Isele, Faizan M. Tariq, Sangjae Bae, Shreyas Kousik, Jovin D'sa - [[pdf]](https://arxiv.org/pdf/2603.04695)</summary>

**Abstract:** In many applications of social navigation, existing works have shown that predicting and reasoning about human intentions can help robotic agents make safer and more socially acceptable decisions. In this work, we study this problem for autonomous valet parking (AVP), where an autonomous vehicle ego agent must drop off its passengers, explore the parking lot, find a parking spot, negotiate for the spot with other vehicles, and park in the spot without human supervision. Specifically, we propose an AVP pipeline that selects parking spots by explicitly predicting where other agents are going to park from their motion history using learned models and probabilistic belief maps. To test this pipeline, we build a simulation environment with reactive agents and realistic modeling assumptions on the ego agent, such as occlusion-aware observations, and imperfect trajectory prediction. Simulation experiments show that our proposed method outperforms existing works that infer intentions from future predicted motion or embed them implicitly in end-to-end models, yielding better results in prediction accuracy, social acceptance, and task completion. Our key insight is that, in parking, where driving regulations are more lax, explicit intention prediction is crucial for reasoning about diverse and ambiguous long-term goals, which cannot be reliably inferred from short-term motion prediction alone, but can be effectively learned from motion history.

**arXiv ID:** 2603.04695
</details>

<details>
<summary><strong>Integrated cooperative localization of heterogeneous measurement swarm: A unified data-driven method</strong> - Kunrui Ze, Wei Wang, Guibin Sun, Jiaqi Yan, Kexin Liu, Jinhu Lü - [[pdf]](https://arxiv.org/pdf/2603.04932)</summary>

**Abstract:** The cooperative localization (CL) problem in heterogeneous robotic systems with different measurement capabilities is investigated in this work. In practice, heterogeneous sensors lead to directed and sparse measurement topologies, whereas most existing CL approaches rely on multilateral localization with restrictive multi-neighbor geometric requirements. To overcome this limitation, we enable pairwise relative localization (RL) between neighboring robots using only mutual measurement and odometry information. A unified data-driven adaptive RL estimator is first developed to handle heterogeneous and unidirectional measurements. Based on the convergent RL estimates, a distributed pose-coupling CL strategy is then designed, which guarantees CL under a weakly connected directed measurement topology, representing the least restrictive condition among existing results. The proposed method is independent of specific control tasks and is validated through a formation control application and real-world experiments.

**arXiv ID:** 2603.04932
</details>

<details>
<summary><strong>From Code to Road: A Vehicle-in-the-Loop and Digital Twin-Based Framework for Central Car Server Testing in Autonomous Driving</strong> - Chengdong Wu, Sven Kirchner, Nils Purschke, Axel Torschmied, Norbert Kroth, Yinglei Song, André Schamschurko, Erik Leo Haß, Kuo-Yi Chao, Yi Zhang, Nenad Petrovic, Alois C. Knoll - [[pdf]](https://arxiv.org/pdf/2603.05279)</summary>

**Abstract:** Simulation is one of the most essential parts in the development stage of automotive software. However, purely virtual simulations often struggle to accurately capture all real-world factors due to limitations in modeling. To address this challenge, this work presents a test framework for automotive software on the centralized E/E architecture, which is a central car server in our case, based on Vehicle-in-the-Loop (ViL) and digital twin technology. The framework couples a physical test vehicle on a dynamometer test bench with its synchronized virtual counterpart in a simulation environment. Our approach provides a safe, reproducible, realistic, and cost-effective platform for validating autonomous driving algorithms with a centralized architecture. This test method eliminates the need to test individual physical ECUs and their communication protocols separately. In contrast to traditional ViL methods, the proposed framework runs the full autonomous driving software directly on the vehicle hardware after the simulation process, eliminating flashing and intermediate layers while enabling seamless virtual-physical integration and accurately reflecting centralized E/E behavior. In addition, incorporating mixed testing in both simulated and physical environments reduces the need for full hardware integration during the early stages of automotive development. Experimental case studies demonstrate the effectiveness of the framework in different test scenarios. These findings highlight the potential to reduce development and integration efforts for testing autonomous driving pipelines in the future.

**arXiv ID:** 2603.05279
</details>

<details>
<summary><strong>MOSAIC: Modular Scalable Autonomy for Intelligent Coordination of Heterogeneous Robotic Teams</strong> - David Oberacker, Julia Richter, Philip Arm, Marvin Grosse Besselmann, Lennart Puck, William Talbot, Maximilian Schik, Sabine Bellmann, Tristan Schnell, Hendrik Kolvenbach, Rüdiger Dillmann, Marco Hutter, Arne Roennau - [[pdf]](https://arxiv.org/pdf/2601.23038)</summary>

**Abstract:** Mobile robots have become indispensable for exploring hostile environments, such as in space or disaster relief scenarios, but often remain limited to teleoperation by a human operator. This restricts the deployment scale and requires near-continuous low-latency communication between the operator and the robot. We present MOSAIC: a scalable autonomy framework for multi-robot scientific exploration using a unified mission abstraction based on Points of Interest (POIs) and multiple layers of autonomy, enabling supervision by a single operator. The framework dynamically allocates exploration and measurement tasks based on each robot's capabilities, leveraging team-level redundancy and specialization to enable continuous operation. We validated the framework in a space-analog field experiment emulating a lunar prospecting scenario, involving a heterogeneous team of five robots and a single operator. Despite the complete failure of one robot during the mission, the team completed 82.3% of assigned tasks at an Autonomy Ratio of 86%, while the operator workload remained at only 78.2%. These results demonstrate that the proposed framework enables robust, scalable multi-robot scientific exploration with limited operator intervention. We further derive practical lessons learned in robot interoperability, networking architecture, team composition, and operator workload management to inform future multi-robot exploration missions.

**arXiv ID:** 2601.23038
</details>

<details>
<summary><strong>Environment-Aware Learning of Smooth GNSS Covariance Dynamics for Autonomous Racing</strong> - Y. Deemo Chen, Arion Zimmermann, Thomas A. Berrueta, Soon-Jo Chung - [[pdf]](https://arxiv.org/pdf/2602.21366)</summary>

**Abstract:** Ensuring accurate and stable state estimation is a challenging task crucial to safety-critical domains such as high-speed autonomous racing, where measurement uncertainty must be both adaptive to the environment and temporally smooth for control. In this work, we develop a learning-based framework, LACE, capable of directly modeling the temporal dynamics of GNSS measurement covariance. We model the covariance evolution as an exponentially stable dynamical system where a deep neural network (DNN) learns to predict the system's process noise from environmental features through an attention mechanism. By using contraction-based stability and systematically imposing spectral constraints, we formally provide guarantees of exponential stability and smoothness for the resulting covariance dynamics. We validate our approach on an AV-24 autonomous racecar, demonstrating improved localization performance and smoother covariance estimates in challenging, GNSS-degraded environments. Our results highlight the promise of dynamically modeling the perceived uncertainty in state estimation problems that are tightly coupled with control sensitivity.

**arXiv ID:** 2602.21366
</details>

<details>
<summary><strong>Vision Language Model-based Testing of Industrial Autonomous Mobile Robots</strong> - Jiahui Wu, Chengjie Lu, Aitor Arrieta, Shaukat Ali, Thomas Peyrucain - [[pdf]](https://arxiv.org/pdf/2508.02338)</summary>

**Abstract:** PAL Robotics, in Spain, builds a variety of Autonomous Mobile Robots (AMRs), which are deployed in diverse environments (e.g., warehouses, retail spaces, and offices), where they work alongside humans. Given that human behavior can be unpredictable and that AMRs may not have been trained to handle all possible unknown and uncertain behaviors, it is important to test AMRs under a wide range of human interactions to ensure their safe behavior. Moreover, testing in real environments with actual AMRs and humans is often costly, impractical, and potentially hazardous (e.g., it could result in human injury). To this end, we propose a Vision Language Model (VLM)-based testing approach (RVSG) for industrial AMRs developed together with PAL Robotics. Based on the functional and safety requirements, RVSG uses the VLM to generate diverse human behaviors that violate these requirements. We evaluated RVSG with several requirements and navigation routes in a simulator using the latest AMR from PAL Robotics. Our results show that, compared with the baseline, RVSG can effectively generate requirement-violating scenarios. Moreover, RVSG-generated scenarios increase variability in robot behavior, thereby helping reveal their uncertain behaviors.

**arXiv ID:** 2508.02338
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (39 papers)</h2></summary>

<details>
<summary><strong>Visioning Human-Agentic AI Teaming: Continuity, Tension, and Future Research</strong> - Bowen Lou, Tian Lu, T. S. Raghu, Yingjie Zhang - [[pdf]](https://arxiv.org/pdf/2603.04746)</summary>

**Abstract:** Artificial intelligence is undergoing a structural transformation marked by the rise of agentic systems capable of open-ended action trajectories, generative representations and outputs, and evolving objectives. These properties introduce structural uncertainty into human-AI teaming (HAT), including uncertainty about behavior trajectories, epistemic grounding, and the stability of governing logics over time. Under such conditions, alignment cannot be secured through agreement on bounded outputs; it must be continuously sustained as plans unfold and priorities shift. We advance Team Situation Awareness (Team SA) theory, grounded in shared perception, comprehension, and projection, as an integrative anchor for this transition. While Team SA remains analytically foundational, its stabilizing logic presumes that shared awareness, once achieved, will support coordinated action through iterative updating. Agentic AI challenges this presumption. Our argument unfolds in two stages: first, we extend Team SA to reconceptualize both human and AI awareness under open-ended agency, including the sensemaking of projection congruence across heterogeneous systems. Second, we interrogate whether the dynamic processes traditionally assumed to stabilize teaming in relational interaction, cognitive learning, and coordination and control continue to function under adaptive autonomy. By distinguishing continuity from tension, we clarify where foundational insights hold and where structural uncertainty introduces strain, and articulate a forward-looking research agenda for HAT. The central challenge of HAT is not whether humans and AI can agree in the moment, but whether they can remain aligned as futures are continuously generated, revised, enacted, and governed over time.

**arXiv ID:** 2603.04746
</details>

<details>
<summary><strong>Breaking Contextual Inertia: Reinforcement Learning with Single-Turn Anchors for Stable Multi-Turn Interaction</strong> - Xingwu Chen, Zhanqiu Zhang, Yiwen Guo, Difan Zou - [[pdf]](https://arxiv.org/pdf/2603.04783)</summary>

**Abstract:** While LLMs demonstrate strong reasoning capabilities when provided with full information in a single turn, they exhibit substantial vulnerability in multi-turn interactions. Specifically, when information is revealed incrementally or requires updates, models frequently fail to integrate new constraints, leading to a collapse in performance compared to their single-turn baselines. We term the root cause as \emph{Contextual Inertia}: a phenomenon where models rigidly adhere to previous reasoning traces. Even when users explicitly provide corrections or new data in later turns, the model ignores them, preferring to maintain consistency with its previous (incorrect) reasoning path. To address this, we introduce \textbf{R}einforcement \textbf{L}earning with \textbf{S}ingle-\textbf{T}urn \textbf{A}nchors (\textbf{RLSTA}), a generalizable training approach designed to stabilize multi-turn interaction across diverse scenarios and domains. RLSTA leverages the model's superior single-turn capabilities as stable internal anchors to provide reward signals. By aligning multi-turn responses with these anchors, RLSTA empowers models to break contextual inertia and self-calibrate their reasoning based on the latest information. Experiments show that RLSTA significantly outperforms standard fine-tuning and abstention-based methods. Notably, our method exhibits strong cross-domain generalization (e.g., math to code) and proves effective even without external verifiers, highlighting its potential for general-domain applications.

**arXiv ID:** 2603.04783
</details>

<details>
<summary><strong>WebFactory: Automated Compression of Foundational Language Intelligence into Grounded Web Agents</strong> - Sicheng Fan, Qingyun Shi, Shengze Xu, Shengbo Cai, Tieyong Zeng, Li Ling, Yanyi Shang, Dehan Kong - [[pdf]](https://arxiv.org/pdf/2603.05044)</summary>

**Abstract:** Current paradigms for training GUI agents are fundamentally limited by a reliance on either unsafe, non-reproducible live web interactions or costly, scarce human-crafted data and environments. We argue this focus on data volume overlooks a more critical factor: the efficiency of compressing a large language model's (LLM) latent knowledge into actionable agent behavior. We introduce WebFactory, a novel, fully automated closed-loop reinforcement learning pipeline for GUI agents, systematically compressing LLM-encoded internet intelligence into efficient, grounded actions. Our pipeline features a process of scalable environment synthesis, knowledge-aware task generation, LLM-powered trajectory collection, decomposed reward RL training, and systematic agent evaluation. Remarkably, our agent demonstrates exceptional data efficiency and generalization. Trained on synthetic data from only 10 websites within WebFactory, it achieves performance comparable to GUI agents trained on the same amount of human-annotated data from a much larger set of environments. This superior performance is consistent across our internal offline and online transfer benchmarks, where our agent also significantly outperforms the base foundation model. We further provide critical insights into the "embodiment potential" of different LLM foundations, offering a new axis for model evaluation. This work presents a scalable and cost-effective paradigm for transforming passive internet knowledge into active, grounded intelligence, marking a critical step towards general-purpose interactive agents.

**arXiv ID:** 2603.05044
</details>

<details>
<summary><strong>Jagarin: A Three-Layer Architecture for Hibernating Personal Duty Agents on Mobile</strong> - Ravi Kiran Kadaboina - [[pdf]](https://arxiv.org/pdf/2603.05069)</summary>

**Abstract:** Personal AI agents face a fundamental deployment paradox on mobile: persistent background execution drains battery and violates platform sandboxing policies, yet purely reactive agents miss time-sensitive obligations until the user remembers to ask. We present Jagarin, a three-layer architecture that resolves this paradox through structured hibernation and demand-driven wake. The first layer, DAWN (Duty-Aware Wake Network), is an on-device heuristic engine that computes a composite urgency score from four signals: duty-typed optimal action windows, user behavioral engagement prediction, opportunity cost of inaction, and cross-duty batch resonance. It uses adaptive per-user thresholds to decide when a sleeping agent should nudge or escalate. The second layer, ARIA (Agent Relay Identity Architecture), is a commercial email identity proxy that routes the full commercial inbox -- obligations, promotional offers, loyalty rewards, and platform updates -- to appropriate DAWN handlers by message category, eliminating cold-start and removing manual data entry. The third layer, ACE (Agent-Centric Exchange), is a protocol framework for direct machine-readable communication from institutions to personal agents, replacing human-targeted email as the canonical channel. Together, these three layers form a complete stack from institutional signal to on-device action, without persistent cloud state, continuous background execution, or privacy compromise. A working Flutter prototype is demonstrated on Android, combining all three layers with an ephemeral cloud agent invoked only on user-initiated escalation.

**arXiv ID:** 2603.05069
</details>

<details>
<summary><strong>KARL: Knowledge Agents via Reinforcement Learning</strong> - Jonathan D. Chang, Andrew Drozdov, Shubham Toshniwal, Owen Oertell, Alexander Trott, Jacob Portes, Abhay Gupta, Pallavi Koppol, Ashutosh Baheti, Sean Kulinski, Ivan Zhou, Irene Dea, Krista Opsahl-Ong, Simon Favreau-Lessard, Sean Owen, Jose Javier Gonzalez Ortiz, Arnav Singhvi, Xabi Andrade, Cindy Wang, Kartik Sreenivasan, Sam Havens, Jialu Liu, Peyton DeNiro, Wen Sun, Michael Bendersky, Jonathan Frankle - [[pdf]](https://arxiv.org/pdf/2603.05218)</summary>

**Abstract:** We present a system for training enterprise search agents via reinforcement learning that achieves state-of-the-art performance across a diverse suite of hard-to-verify agentic search tasks. Our work makes four core contributions. First, we introduce KARLBench, a multi-capability evaluation suite spanning six distinct search regimes, including constraint-driven entity search, cross-document report synthesis, tabular numerical reasoning, exhaustive entity retrieval, procedural reasoning over technical documentation, and fact aggregation over internal enterprise notes. Second, we show that models trained across heterogeneous search behaviors generalize substantially better than those optimized for any single benchmark. Third, we develop an agentic synthesis pipeline that employs long-horizon reasoning and tool use to generate diverse, grounded, and high-quality training data, with iterative bootstrapping from increasingly capable models. Fourth, we propose a new post-training paradigm based on iterative large-batch off-policy RL that is sample efficient, robust to train-inference engine discrepancies, and naturally extends to multi-task training with out-of-distribution generalization. Compared to Claude 4.6 and GPT 5.2, KARL is Pareto-optimal on KARLBench across cost-quality and latency-quality trade-offs, including tasks that were out-of-distribution during training. With sufficient test-time compute, it surpasses the strongest closed models. These results show that tailored synthetic data in combination with multi-task reinforcement learning enables cost-efficient and high-performing knowledge agents for grounded reasoning.

**arXiv ID:** 2603.05218
</details>

<details>
<summary><strong>CTRL-RAG: Contrastive Likelihood Reward Based Reinforcement Learning for Context-Faithful RAG Models</strong> - Zhehao Tan, Yihan Jiao, Dan Yang, Junjie Wang, Duolin Sun, Jie Feng, Xidong Wang, Lei Liu, Yue Shen, Jian Wang, Jinjie Gu - [[pdf]](https://arxiv.org/pdf/2603.04406)</summary>

**Abstract:** With the growing use of Retrieval-Augmented Generation (RAG), training large language models (LLMs) for context-sensitive reasoning and faithfulness is increasingly important. Existing RAG-oriented reinforcement learning (RL) methods rely on external rewards that often fail to evaluate document faithfulness, and may misjudge similar answers in open-domain settings. In addition, there is no RAG-based selfreward mechanism. Moreover, although such a mechanism could in principle estimate answer confidence given documents, the absence of objective feedback in a self-judgment can cause hallucination accumulation and eventual model collapse. To tackle these issues, we propose a novel "internal-external" hybrid reward framework centered on a Contrastive Likelihood Reward (CLR). CLR directly optimizes the log-likelihood gap between responses conditioned on prompts with and without supporting evidence. This encourages the model to extract relevant evidence and increases its confidence when grounded in a specific context. Experiments show that our method (used alone or combined with external correctness rewards) achieves strong performance on singlehop, multi-hop, vertical-domain, and faithfulness benchmarks. Our training code and models are coming soon.

**arXiv ID:** 2603.04406
</details>

<details>
<summary><strong>Large Language Models as Bidding Agents in Repeated HetNet Auction</strong> - Ismail Lotfi, Ali Ghrayeb, Samson Lasaulce, Merouane Debbah - [[pdf]](https://arxiv.org/pdf/2603.04455)</summary>

**Abstract:** This paper investigates the integration of large language models (LLMs) as reasoning agents in repeated spectrum auctions within heterogeneous networks (HetNets). While auction-based mechanisms have been widely employed for efficient resource allocation, most prior works assume one-shot auctions, static bidder behavior, and idealized conditions. In contrast to traditional formulations where base station (BS) association and power allocation are centrally optimized, we propose a distributed auction-based framework in which each BS independently conducts its own multi-channel auction, and user equipments (UEs) strategically decide both their association and bid values. Within this setting, UEs operate under budget constraints and repeated interactions, transforming resource allocation into a long-term economic decision rather than a one-shot optimization problem. The proposed framework enables the evaluation of diverse bidding behaviors -from classical myopic and greedy policies to LLM-based agents capable of reasoning over historical outcomes, anticipating competition, and adapting their bidding strategy across episodes. Simulation results reveal that the LLM-empowered UE consistently achieves higher channel access frequency and improved budget efficiency compared to benchmarks. These findings highlight the potential of reasoning-enabled agents in future decentralized wireless networks markets and pave the way for lightweight, edge-deployable LLMs to support intelligent resource allocation in next-generation HetNets.

**arXiv ID:** 2603.04455
</details>

<details>
<summary><strong>Bootstrapping Exploration with Group-Level Natural Language Feedback in Reinforcement Learning</strong> - Lei Huang, Xiang Cheng, Chenxiao Zhao, Guobin Shen, Junjie Yang, Xiaocheng Feng, Yuxuan Gu, Xing Yu, Bing Qin - [[pdf]](https://arxiv.org/pdf/2603.04597)</summary>

**Abstract:** Large language models (LLMs) typically receive diverse natural language (NL) feedback through interaction with the environment. However, current reinforcement learning (RL) algorithms rely solely on scalar rewards, leaving the rich information in NL feedback underutilized and leading to inefficient exploration. In this work, we propose GOLF, an RL framework that explicitly exploits group-level language feedback to guide targeted exploration through actionable refinements. GOLF aggregates two complementary feedback sources: (i) external critiques that pinpoint errors or propose targeted fixes, and (ii) intra-group attempts that supply alternative partial ideas and diverse failure patterns. These group-level feedbacks are aggregated to produce high-quality refinements, which are adaptively injected into training as off-policy scaffolds to provide targeted guidance in sparse-reward regions. Meanwhile, GOLF jointly optimizes generation and refinement within a unified RL loop, creating a virtuous cycle that continuously improves both capabilities. Experiments on both verifiable and non-verifiable benchmarks show that GOLF achieves superior performance and exploration efficiency, achieving 2.2$\times$ improvements in sample efficiency compared to RL methods trained solely on scalar rewards. Code is available at this https URL.

**arXiv ID:** 2603.04597
</details>

<details>
<summary><strong>On the Strengths and Weaknesses of Data for Open-set Embodied Assistance</strong> - Pradyumna Tambwekar, Andrew Silva, Deepak Gopinath, Jonathan DeCastro, Xiongyi Cui, Guy Rosman - [[pdf]](https://arxiv.org/pdf/2603.04819)</summary>

**Abstract:** Embodied foundation models are increasingly performant in real-world domains such as robotics or autonomous driving. These models are often deployed in interactive or assistive settings, where it is important that these assistive models generalize to new users and new tasks. Diverse interactive data generation offers a promising avenue for providing data-efficient generalization capabilities for interactive embodied foundation models. In this paper, we investigate the generalization capabilities of a multimodal foundation model fine-tuned on diverse interactive assistance data in a synthetic domain. We explore generalization along two axes: a) assistance with unseen categories of user behavior and b) providing guidance in new configurations not encountered during training. We study a broad capability called \textbf{Open-Set Corrective Assistance}, in which the model needs to inspect lengthy user behavior and provide assistance through either corrective actions or language-based feedback. This task remains unsolved in prior work, which typically assumes closed corrective categories or relies on external planners, making it a challenging testbed for evaluating the limits of assistive data. To support this task, we generate synthetic assistive datasets in Overcooked and fine-tune a LLaMA-based model to evaluate generalization to novel tasks and user behaviors. Our approach provides key insights into the nature of assistive datasets required to enable open-set assistive intelligence. In particular, we show that performant models benefit from datasets that cover different aspects of assistance, including multimodal grounding, defect inference, and exposure to diverse scenarios.

**arXiv ID:** 2603.04819
</details>

<details>
<summary><strong>AgentSCOPE: Evaluating Contextual Privacy Across Agentic Workflows</strong> - Ivoline C. Ngong, Keerthiram Murugesan, Swanand Kadhe, Justin D. Weisz, Amit Dhurandhar, Karthikeyan Natesan Ramamurthy - [[pdf]](https://arxiv.org/pdf/2603.04902)</summary>

**Abstract:** Agentic systems are increasingly acting on users' behalf, accessing calendars, email, and personal files to complete everyday tasks. Privacy evaluation for these systems has focused on the input and output boundaries, but each task involves several intermediate information flows, from agent queries to tool responses, that are not currently evaluated. We argue that every boundary in an agentic pipeline is a site of potential privacy violation and must be assessed independently. To support this, we introduce the Privacy Flow Graph, a Contextual Integrity-grounded framework that decomposes agentic execution into a sequence of information flows, each annotated with the five CI parameters, and traces violations to their point of origin. We present AgentSCOPE, a benchmark of 62 multi-tool scenarios across eight regulatory domains with ground truth at every pipeline stage. Our evaluation across seven state-of-the-art LLMs show that privacy violations in the pipeline occur in over 80% of scenarios, even when final outputs appear clean (24%), with most violations arising at the tool-response stage where APIs return sensitive data indiscriminately. These results indicate that output-level evaluation alone substantially underestimates the privacy risk of agentic systems.

**arXiv ID:** 2603.04902
</details>

<details>
<summary><strong>EVMbench: Evaluating AI Agents on Smart Contract Security</strong> - Justin Wang, Andreas Bigger, Xiaohai Xu, Justin W. Lin, Andy Applebaum, Tejal Patwardhan, Alpin Yukseloglu, Olivia Watkins - [[pdf]](https://arxiv.org/pdf/2603.04915)</summary>

**Abstract:** Smart contracts on public blockchains now manage large amounts of value, and vulnerabilities in these systems can lead to substantial losses. As AI agents become more capable at reading, writing, and running code, it is natural to ask how well they can already navigate this landscape, both in ways that improve security and in ways that might increase risk. We introduce EVMbench, an evaluation that measures the ability of agents to detect, patch, and exploit smart contract vulnerabilities. EVMbench draws on 117 curated vulnerabilities from 40 repositories and, in the most realistic setting, uses programmatic grading based on tests and blockchain state under a local Ethereum execution environment. We evaluate a range of frontier agents and find that they are capable of discovering and exploiting vulnerabilities end-to-end against live blockchain instances. We release code, tasks, and tooling to support continued measurement of these capabilities and future work on security.

**arXiv ID:** 2603.04915
</details>

<details>
<summary><strong>BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning</strong> - Yuan Li, Bo Wang, Yufei Gao, Yuqian Yao, Xinyuan Wang, Zhangyue Yin, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2603.04918)</summary>

**Abstract:** Proximal constraints are fundamental to the stability of the Large Language Model reinforcement learning. While the canonical clipping mechanism in PPO serves as an efficient surrogate for trust regions, we identify a critical bottleneck: fixed bounds strictly constrain the upward update margin of low-probability actions, disproportionately suppressing high-advantage tail strategies and inducing rapid entropy collapse. To address this, we introduce Band-constrained Policy Optimization (BandPO). BandPO replaces canonical clipping with Band, a unified theoretical operator that projects trust regions defined by f-divergences into dynamic, probability-aware clipping intervals. Theoretical analysis confirms that Band effectively resolves this exploration bottleneck. We formulate this mapping as a convex optimization problem, guaranteeing a globally optimal numerical solution while deriving closed-form solutions for specific divergences. Extensive experiments across diverse models and datasets demonstrate that BandPO consistently outperforms canonical clipping and Clip-Higher, while robustly mitigating entropy collapse.

**arXiv ID:** 2603.04918
</details>

<details>
<summary><strong>Boosting ASR Robustness via Test-Time Reinforcement Learning with Audio-Text Semantic Rewards</strong> - Linghan Fang, Tianxin Xie, Li Liu - [[pdf]](https://arxiv.org/pdf/2603.05231)</summary>

**Abstract:** Recently, Automatic Speech Recognition (ASR) systems (e.g., Whisper) have achieved remarkable accuracy improvements but remain highly sensitive to real-world unseen data (data with large distribution shifts), including noisy environments and diverse accents. To address this issue, test-time adaptation (TTA) has shown great potential in improving the model adaptability at inference time without ground-truth labels, and existing TTA methods often rely on pseudo-labeling or entropy minimization. However, by treating model confidence as a learning signal, these methods may reinforce high-confidence errors, leading to confirmation bias that undermines adaptation. To overcome these limitations, we present ASR-TRA, a novel Test-time Reinforcement Adaptation framework inspired by causal intervention. More precisely, our method introduces a learnable decoder prompt and utilizes temperature-controlled stochastic decoding to generate diverse transcription candidates. These are scored by a reward model that measures audio-text semantic alignment, and the resulting feedback is used to update both model and prompt parameters via reinforcement learning. Comprehensive experiments on LibriSpeech with synthetic noise and L2 Arctic accented English datasets demonstrate that our method achieves higher accuracy while maintaining lower latency than existing TTA baselines. Ablation studies further confirm the effectiveness of combining audio and language-based rewards, highlighting our method's enhanced stability and interpretability. Overall, our approach provides a practical and robust solution for deploying ASR systems in challenging real-world conditions.

**arXiv ID:** 2603.05231
</details>

<details>
<summary><strong>Pessimistic Auxiliary Policy for Offline Reinforcement Learning</strong> - Fan Zhang, Baoru Huang, Xin Zhang - [[pdf]](https://arxiv.org/pdf/2602.23974)</summary>

**Abstract:** Offline reinforcement learning aims to learn an agent from pre-collected datasets, avoiding unsafe and inefficient real-time interaction. However, inevitable access to out-ofdistribution actions during the learning process introduces approximation errors, causing the error accumulation and considerable overestimation. In this paper, we construct a new pessimistic auxiliary policy for sampling reliable actions. Specifically, we develop a pessimistic auxiliary strategy by maximizing the lower confidence bound of the Q-function. The pessimistic auxiliary strategy exhibits a relatively high value and low uncertainty in the vicinity of the learned policy, avoiding the learned policy sampling high-value actions with potentially high errors during the learning process. Less approximation error introduced by sampled action from pessimistic auxiliary strategy leads to the alleviation of error accumulation. Extensive experiments on offline reinforcement learning benchmarks reveal that utilizing the pessimistic auxiliary strategy can effectively improve the efficacy of other offline RL approaches.

**arXiv ID:** 2602.23974
</details>

<details>
<summary><strong>Agents Learn Their Runtime: Interpreter Persistence as Training-Time Semantics</strong> - Victor May, Aaditya Salgarkar, Yishan Wang, Diganta Misra, Huu Nguyen - [[pdf]](https://arxiv.org/pdf/2603.01209)</summary>

**Abstract:** Tool-augmented LLMs are increasingly deployed as agents that interleave natural-language reasoning with executable Python actions, as in CodeAct-style frameworks. In deployment, these agents rely on runtime state that persists across steps. By contrast, the traces used to post-train these models rarely encode how interpreter state is managed. We ask whether interpreter persistence is merely a runtime scaffold, or a property of the training data that shapes how agents learn to use the interpreter.
We isolate state persistence as a training-time variable. We introduce Opaque Knapsack, a procedurally generated family of partially observable optimization tasks designed to prevent one-shot solutions. Item attributes and constraints are hidden behind budgeted tool calls, forcing multi-turn control flow and iterative state revision. Holding task instances, prompts, tools, model, and supervision fixed, we generate matched trajectories differing only in whether interpreter state persists across steps or resets after each action. We then fine-tune identical base models (Qwen3-8B) on each trace variant and evaluate all four train-runtime combinations.
Our 2x2 cross-evaluation shows that interpreter persistence shapes how agents reach solutions, not whether they do: solution quality is statistically indistinguishable across conditions, but token cost and stability differ substantially. A persistent-trained model in a stateless runtime triggers missing-variable errors in roughly 80% of episodes; a stateless-trained model in a persistent runtime redundantly re-derives retained state, using roughly 3.5x more tokens.
Interpreter persistence should be treated as a first-class semantic of agent traces. Aligning fine-tuning data with deployment runtimes improves efficiency and reduces brittle train-runtime mismatches.

**arXiv ID:** 2603.01209
</details>

<details>
<summary><strong>ToolRLA: Multiplicative Reward Decomposition for Tool-Integrated Agents</strong> - Pengbo Liu - [[pdf]](https://arxiv.org/pdf/2603.01620)</summary>

**Abstract:** Tool-integrated agents that interleave reasoning with API calls are promising for complex tasks, yet aligning them for high-stakes, domain-specific deployment remains challenging: existing reinforcement learning approaches rely on coarse binary rewards that cannot distinguish tool selection errors from malformed parameters. We present ToolRLA, a three-stage post-training pipeline (SFT -> GRPO -> DPO) for domain-specific tool agents. The core contribution is a fine-grained reward function with multiplicative correctness decomposition spanning four dimensions -- format validity, tool selection, parameter accuracy, and regulatory compliance -- that encodes domain priority orderings as inductive biases in the reward landscape. Deployed on a financial advisory copilot (80+ advisors, 1,200+ daily queries), ToolRLA achieves over three months: a 47% improvement in task completion rate (62%->91%), a 63% reduction in tool invocation errors (38%->14%), and a 93% reduction in regulatory violations (12%->0.8%), within sub-2-second latency. Ablation studies show the multiplicative reward design accounts for 7 percentage points of improvement over additive alternatives. Generalization is further validated on ToolBench and API-Bank.

**arXiv ID:** 2603.01620
</details>

<details>
<summary><strong>Balancing Progress and Safety: A Novel Risk-Aware Objective for RL in Autonomous Driving</strong> - Ahmed Abouelazm, Jonas Michel, Helen Gremmelmaier, Tim Joseph, Philip Schörner, J. Marius Zöllner - [[pdf]](https://arxiv.org/pdf/2505.06737)</summary>

**Abstract:** Reinforcement Learning (RL) is a promising approach for achieving autonomous driving due to robust decision-making capabilities. RL learns a driving policy through trial and error in traffic scenarios, guided by a reward function that combines the driving objectives. The design of such reward function has received insufficient attention, yielding ill-defined rewards with various pitfalls. Safety, in particular, has long been regarded only as a penalty for collisions. This leaves the risks associated with actions leading up to a collision unaddressed, limiting the applicability of RL in real-world scenarios. To address these shortcomings, our work focuses on enhancing the reward formulation by defining a set of driving objectives and structuring them hierarchically. Furthermore, we discuss the formulation of these objectives in a normalized manner to transparently determine their contribution to the overall reward. Additionally, we introduce a novel risk-aware objective for various driving interactions based on a two-dimensional ellipsoid function and an extension of Responsibility-Sensitive Safety (RSS) concepts. We evaluate the efficacy of our proposed reward in unsignalized intersection scenarios with varying traffic densities. The approach decreases collision rates by 21\% on average compared to baseline rewards and consistently surpasses them in route progress and cumulative reward, demonstrating its capability to promote safer driving behaviors while maintaining high-performance levels.

**arXiv ID:** 2505.06737
</details>

<details>
<summary><strong>Automatic Curriculum Learning for Driving Scenarios: Towards Robust and Efficient Reinforcement Learning</strong> - Ahmed Abouelazm, Tim Weinstein, Tim Joseph, Philip Schörner, J. Marius Zöllner - [[pdf]](https://arxiv.org/pdf/2505.08264)</summary>

**Abstract:** This paper addresses the challenges of training end-to-end autonomous driving agents using Reinforcement Learning (RL). RL agents are typically trained in a fixed set of scenarios and nominal behavior of surrounding road users in simulations, limiting their generalization and real-life deployment. While domain randomization offers a potential solution by randomly sampling driving scenarios, it frequently results in inefficient training and sub-optimal policies due to the high variance among training scenarios. To address these limitations, we propose an automatic curriculum learning framework that dynamically generates driving scenarios with adaptive complexity based on the agent's evolving capabilities. Unlike manually designed curricula that introduce expert bias and lack scalability, our framework incorporates a ``teacher'' that automatically generates and mutates driving scenarios based on their learning potential -- an agent-centric metric derived from the agent's current policy -- eliminating the need for expert design. The framework enhances training efficiency by excluding scenarios the agent has mastered or finds too challenging. We evaluate our framework in a reinforcement learning setting where the agent learns a driving policy from camera images. Comparative results against baseline methods, including fixed scenario training and domain randomization, demonstrate that our approach leads to enhanced generalization, achieving higher success rates: +9% in low traffic density, +21% in high traffic density, and faster convergence with fewer training steps. Our findings highlight the potential of ACL in improving the robustness and efficiency of RL-based autonomous driving agents.

**arXiv ID:** 2505.08264
</details>

<details>
<summary><strong>iAgentBench: Benchmarking Sensemaking Capabilities of Information-Seeking Agents on High-Traffic Topics</strong> - Preetam Prabhu Srikar Dammu, Arnav Palkhiwala, Tanya Roosta, Chirag Shah - [[pdf]](https://arxiv.org/pdf/2603.04656)</summary>

**Abstract:** With the emergence of search-enabled generative QA systems, users are increasingly turning to tools that browse, aggregate, and reconcile evidence across multiple sources on their behalf. Yet many widely used QA benchmarks remain answerable by retrieving a single relevant passage, making them poorly suited for measuring cross-source sensemaking, such as integrating evidence, tracking causal links, and resolving dependencies across facets of a topic. We present iAgentBench, a dynamic ODQA benchmark that targets these higher-level information needs while keeping questions natural and grounded in realistic information-seeking behavior. iAgentBench draws seed topics from real-world attention signals and uses common user intent patterns to construct user-like questions whose answers require combining evidence from multiple sources, not just extracting a single snippet. Each instance is released with traceable evidence and auditable intermediate artifacts that support contamination checks and enable fine-grained diagnosis of failures in retrieval versus synthesis. Experiments across multiple LLMs show that retrieval improves accuracy, but retrieval alone does not reliably resolve these questions, underscoring the need to evaluate evidence use, not just evidence access.

**arXiv ID:** 2603.04656
</details>

<details>
<summary><strong>Competitive Multi-Operator Reinforcement Learning for Joint Pricing and Fleet Rebalancing in AMoD Systems</strong> - Emil Kragh Toft, Carolin Schmidt, Daniele Gammelli, Filipe Rodrigues - [[pdf]](https://arxiv.org/pdf/2603.05000)</summary>

**Abstract:** Autonomous Mobility-on-Demand (AMoD) systems promise to revolutionize urban transportation by providing affordable on-demand services to meet growing travel demand. However, realistic AMoD markets will be competitive, with multiple operators competing for passengers through strategic pricing and fleet deployment. While reinforcement learning has shown promise in optimizing single-operator AMoD control, existing work fails to capture competitive market dynamics. We investigate the impact of competition on policy learning by introducing a multi-operator reinforcement learning framework where two operators simultaneously learn pricing and fleet rebalancing policies. By integrating discrete choice theory, we enable passenger allocation and demand competition to emerge endogenously from utility-maximizing decisions. Experiments using real-world data from multiple cities demonstrate that competition fundamentally alters learned behaviors, leading to lower prices and distinct fleet positioning patterns compared to monopolistic settings. Notably, we demonstrate that learning-based approaches are robust to the additional stochasticity of competition, with competitive agents successfully converging to effective policies while accounting for partially unobserved competitor strategies.

**arXiv ID:** 2603.05000
</details>

<details>
<summary><strong>EmboTeam: Grounding LLM Reasoning into Reactive Behavior Trees via PDDL for Embodied Multi-Robot Collaboration</strong> - Haishan Zeng, Mengna Wang, Peng Li - [[pdf]](https://arxiv.org/pdf/2601.11063)</summary>

**Abstract:** In embodied artificial intelligence, enabling heterogeneous robot teams to execute long-horizon tasks from high-level instructions remains a critical challenge. While large language models (LLMs) show promise in instruction parsing and preliminary planning, they exhibit limitations in long-term reasoning and dynamic multi-robot coordination. We propose EmboTeam, a novel embodied multi-robot task planning framework that addresses these issues through a three-stage cascaded architecture: 1) It leverages an LLM to parse instructions and generate Planning Domain Definition Language (PDDL) problem descriptions, thereby transforming commands into formal planning problems; 2) It combines the semantic reasoning of LLMs with the search capabilities of a classical planner to produce optimized action sequences; 3) It compiles the resulting plan into behavior trees for reactive control. The framework supports dynamically sized heterogeneous robot teams via a shared blackboard mechanism for communication and state synchronization. To validate our approach, we introduce the MACE-THOR benchmark dataset, comprising 42 complex tasks across 8 distinct household layouts. Experiments show EmboTeam improves the task success rate from 12% to 55% and goal condition recall from 32% to 72% over the LaMMA-P baseline.

**arXiv ID:** 2601.11063
</details>

<details>
<summary><strong>Neural Network-Based Parameter Estimation of a Labour Market Agent-Based Model</strong> - M Lopes Alves, Joel Dyer, Doyne Farmer, Michael Wooldridge, Anisoara Calinescu - [[pdf]](https://arxiv.org/pdf/2602.15572)</summary>

**Abstract:** Agent-based modelling (ABM) is a widespread approach to simulate complex systems. Advancements in computational processing and storage have facilitated the adoption of ABMs across many fields; however, ABMs face challenges that limit their use as decision-support tools. A significant issue is parameter estimation in large-scale ABMs, particularly due to computational constraints on exploring the parameter space. This study evaluates a state-of-the-art simulation-based inference (SBI) framework that uses neural networks (NN) for parameter estimation. This framework is applied to an established labour market ABM based on job transition networks. The ABM is initiated with synthetic datasets and the real U.S. labour market. Next, we compare the effectiveness of summary statistics derived from a list of statistical measures with that learned by an embedded NN. The results demonstrate that the NN-based approach recovers the original parameters when evaluating posterior distributions across various dataset scales and improves efficiency compared to traditional Bayesian methods.

**arXiv ID:** 2602.15572
</details>

<details>
<summary><strong>Learning Virtual Machine Scheduling in Cloud Computing through Language Agents</strong> - JieHao Wu, Ziwei Wang, Junjie Sheng, Wenhao Li, Xiangfeng Wang, Jun Luo - [[pdf]](https://arxiv.org/pdf/2505.10117)</summary>

**Abstract:** In cloud services, virtual machine (VM) scheduling is a typical Online Dynamic Multidimensional Bin Packing (ODMBP) problem, characterized by large-scale complexity and fluctuating demands. Traditional optimization methods struggle to adapt to real-time changes, domain-expert-designed heuristic approaches suffer from rigid strategies, and existing learning-based methods often lack generalizability and interpretability. To address these limitations, this paper proposes a hierarchical language agent framework named MiCo, which provides a large language model (LLM)-driven heuristic design paradigm for solving ODMBP. Specifically, ODMBP is formulated as a Semi-Markov Decision Process with Options (SMDP-Option), enabling dynamic scheduling through a two-stage architecture, i.e., Option Miner and Option Composer. Option Miner utilizes LLMs to discover diverse and useful non-context-aware strategies by interacting with constructed environments. Option Composer employs LLMs to discover a composing strategy that integrates the non-context-aware strategies with the contextual ones. Extensive experiments on real-world enterprise datasets demonstrate that MiCo achieves a 96.9\% competitive ratio in large-scale scenarios involving more than 10,000 virtual machines. It maintains high performance even under nonstationary request flows and diverse configurations, thus validating its effectiveness in complex and large-scale cloud environments.

**arXiv ID:** 2505.10117
</details>

<details>
<summary><strong>Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards</strong> - Hieu Trung Nguyen, Bao Nguyen, Wenao Ma, Yuzhi Zhao, Ruifeng She, Viet Anh Nguyen - [[pdf]](https://arxiv.org/pdf/2602.01601)</summary>

**Abstract:** Sampling efficiency is a key bottleneck in reinforcement learning with verifiable rewards. Existing group-based policy optimization methods, such as GRPO, allocate a fixed number of rollouts for all training prompts. This uniform allocation implicitly treats all prompts as equally informative, and could lead to inefficient computational budget usage and impede training progress. We introduce VIP, a Variance-Informed Predictive allocation strategy that allocates a given rollout budget to the prompts in the incumbent batch to minimize the expected gradient variance of the policy update. At each iteration, VIP uses a lightweight Gaussian process model to predict per-prompt success probabilities based on recent rollouts. These probability predictions are translated into variance estimates, which are then fed into a convex optimization problem to determine the optimal rollout allocations under a hard compute budget constraint. Empirical results show that VIP consistently improves sampling efficiency and achieves higher performance than uniform or heuristic allocation strategies in multiple benchmarks.

**arXiv ID:** 2602.01601
</details>

<details>
<summary><strong>Distributional Reinforcement Learning with Information Bottleneck for Uncertainty-Aware DRAM Equalization</strong> - Muhammad Usama, Dong Eui Chang - [[pdf]](https://arxiv.org/pdf/2603.04768)</summary>

**Abstract:** Equalizer parameter optimization is critical for signal integrity in high-speed memory systems operating at multi-gigabit data rates. However, existing methods suffer from computationally expensive eye diagram evaluation, optimization of expected rather than worst-case performance, and absence of uncertainty quantification for deployment decisions. In this paper, we propose a distributional risk-sensitive reinforcement learning framework integrating Information Bottleneck latent representations with Conditional Value-at-Risk optimization. We introduce rate-distortion optimal signal compression achieving 51 times speedup over eye diagrams while quantifying epistemic uncertainty through Monte Carlo dropout. Distributional reinforcement learning with quantile regression enables explicit worst-case optimization, while PAC-Bayesian regularization certifies generalization bounds. Experimental validation on 2.4 million waveforms from eight memory units demonstrated mean improvements of 37.1\% and 41.5\% for 4-tap and 8-tap equalizer configurations with worst-case guarantees of 33.8\% and 38.2\%, representing 80.7\% and 89.1\% improvements over Q-learning baselines. The framework achieved 62.5\% high-reliability classification eliminating manual validation for most configurations. These results suggest the proposed framework provides a practical solution for production-scale equalizer optimization with certified worst-case guarantees.

**arXiv ID:** 2603.04768
</details>

<details>
<summary><strong>Reward-Conditioned Reinforcement Learning</strong> - Michal Nauman, Marek Cygan, Pieter Abbeel - [[pdf]](https://arxiv.org/pdf/2603.05066)</summary>

**Abstract:** RL agents are typically trained under a single, fixed reward function, which makes them brittle to reward misspecification and limits their ability to adapt to changing task preferences. We introduce Reward-Conditioned Reinforcement Learning (RCRL), a framework that trains a single agent to optimize a family of reward specifications while collecting experience under only one nominal objective. RCRL conditions the agent on reward parameterizations and learns multiple reward objectives from a shared replay data entirely off-policy, enabling a single policy to represent reward-specific behaviors. Across single-task, multi-task, and vision-based benchmarks, we show that RCRL not only improves performance under the nominal reward parameterization, but also enables efficient adaptation to new parameterizations. Our results demonstrate that RCRL provides a scalable mechanism for learning robust, steerable policies without sacrificing the simplicity of single-task training.

**arXiv ID:** 2603.05066
</details>

<details>
<summary><strong>Decoupling Task and Behavior: A Two-Stage Reward Curriculum in Reinforcement Learning for Robotics</strong> - Kilian Freitag, Knut Åkesson, Morteza Haghir Chehreghani - [[pdf]](https://arxiv.org/pdf/2603.05113)</summary>

**Abstract:** Deep Reinforcement Learning is a promising tool for robotic control, yet practical application is often hindered by the difficulty of designing effective reward functions. Real-world tasks typically require optimizing multiple objectives simultaneously, necessitating precise tuning of their weights to learn a policy with the desired characteristics. To address this, we propose a two-stage reward curriculum where we decouple task-specific objectives from behavioral terms. In our method, we first train the agent on a simplified task-only reward function to ensure effective exploration before introducing the full reward that includes auxiliary behavior-related terms such as energy efficiency. Further, we analyze various transition strategies and demonstrate that reusing samples between phases is critical for training stability. We validate our approach on the DeepMind Control Suite, ManiSkill3, and a mobile robot environment, modified to include auxiliary behavioral objectives. Our method proves to be simple yet effective, substantially outperforming baselines trained directly on the full reward while exhibiting higher robustness to specific reward weightings.

**arXiv ID:** 2603.05113
</details>

<details>
<summary><strong>VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use</strong> - Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, Klara Nahrstedt - [[pdf]](https://arxiv.org/pdf/2505.19255)</summary>

**Abstract:** Reinforcement Learning Finetuning (RFT) has significantly advanced the reasoning capabilities of large language models (LLMs) by enabling long chains of thought, self-correction, and effective tool use. While recent works attempt to extend RFT to vision-language models (VLMs), these efforts largely produce text-only reasoning conditioned on static image inputs, falling short of true multimodal reasoning in the response. In contrast, test-time methods like Visual Sketchpad incorporate visual steps but lack training mechanisms.
We introduce VTool-R1, the first framework that trains VLMs to generate multimodal chains of thought by interleaving text and intermediate visual reasoning steps. VTool-R1 integrates Python-based visual editing tools into the RFT process, enabling VLMs to learn when and how to generate visual reasoning steps that benefit final reasoning. Trained with outcome-based rewards tied to task accuracy, our approach elicits strategic visual tool use for reasoning without relying on process-based supervision. Experiments on structured visual question answering over charts and tables show that VTool-R1 enhances reasoning performance by teaching VLMs to "think with images" and generate multimodal chain of thoughts with tools. To support future research in multi-turn multi-modal reasoning, we open-source our code at this https URL

**arXiv ID:** 2505.19255
</details>

<details>
<summary><strong>Parameter Stress Analysis in Reinforcement Learning: Applying Synaptic Filtering to Policy Networks</strong> - Zain ul Abdeen, Ming Jin - [[pdf]](https://arxiv.org/pdf/2506.23036)</summary>

**Abstract:** This paper explores reinforcement learning (RL) policy robustness by systematically analyzing network parameters under internal and external stresses. \textcolor{black}{We apply synaptic filtering methods using high-pass, low-pass, and pulse-wave filters from} \citep{pravin2024fragility}, as an internal stress by selectively perturbing parameters, while adversarial attacks apply external stress through modified agent observations. This dual approach enables the classification of parameters as \textit{fragile}, \textit{robust}, or \textit{antifragile}, based on their influence on policy performance in clean and adversarial settings. Parameter scores are defined to quantify these characteristics, and the framework is validated on proximal policy optimization (PPO)-trained agents in Mujoco continuous control environments. The results highlight the presence of antifragile parameters that enhance policy performance under stress, demonstrating the potential of targeted filtering techniques to improve RL policy adaptability. These insights provide a foundation for future advancements in the design of robust and antifragile RL systems.

**arXiv ID:** 2506.23036
</details>

<details>
<summary><strong>Kernel Based Maximum Entropy Inverse Reinforcement Learning for Mean-Field Games</strong> - Berkay Anahtarci, Can Deha Kariksiz, Naci Saldi - [[pdf]](https://arxiv.org/pdf/2507.14529)</summary>

**Abstract:** We consider the maximum causal entropy inverse reinforcement learning (IRL) problem for infinite-horizon stationary mean-field games (MFG), in which we model the unknown reward function within a reproducing kernel Hilbert space (RKHS). This allows the inference of rich and potentially nonlinear reward structures directly from expert demonstrations, in contrast to most existing approaches for MFGs that typically restrict the reward to a linear combination of a fixed finite set of basis functions and rely on finite-horizon formulations. We introduce a Lagrangian relaxation that enables us to reformulate the problem as an unconstrained log-likelihood maximization and obtain a solution via a gradient ascent algorithm. To establish the theoretical consistency of the algorithm, we prove the smoothness of the log-likelihood objective through the Fréchet differentiability of the related soft Bellman operators with respect to the parameters in the RKHS. To illustrate the practical advantages of the RKHS formulation, we validate our framework on a mean-field traffic routing game exhibiting state-dependent preference reversal, where the kernel-based method reduces policy recovery error by over an order of magnitude compared to a linear reward baseline with a comparable parameter count. Furthermore, we extend the framework to the finite-horizon non-stationary setting. We demonstrate that the log-likelihood reformulation is structurally unavailable in this regime and instead develop an alternative gradient descent algorithm on the convex dual via Danskin's theorem, establishing smoothness and convergence guarantees.

**arXiv ID:** 2507.14529
</details>

<details>
<summary><strong>TIC-GRPO: Provable and Efficient Optimization for Reinforcement Learning from Human Feedback</strong> - Lei Pang, Jun Luo, Ruinan Jin - [[pdf]](https://arxiv.org/pdf/2508.02833)</summary>

**Abstract:** Group Relative Policy Optimization (GRPO), recently introduced by DeepSeek, is a critic-free reinforcement learning algorithm for fine-tuning large language models. GRPO replaces the value function in Proximal Policy Optimization (PPO) with group-normalized rewards while retaining PPO-style token-level importance sampling based on an old policy. Our theoretical analysis reveals that the GRPO update rule estimates the policy gradient at the old policy rather than the current one; however, since the old policy is refreshed every few steps, the resulting discrepancy remains small and the induced bias is negligible in practice. To empirically validate this insight, we conduct an ablation study that entirely removes importance sampling and performs multiple optimization steps using gradients estimated at a fixed old policy. Remarkably, this simplified variant attains performance comparable to standard GRPO.
Motivated by this finding, we propose Trajectory-level Importance-Corrected GRPO (TIC-GRPO), a new algorithm that replaces token-level importance ratios with a single trajectory-level probability ratio, thereby yielding an estimate of the current policy gradient while preserving the critic-free structure. Furthermore, we present the first convergence analysis for GRPO-style methods and show that TIC-GRPO converges faster than GRPO. Finally, empirical results across math reasoning and coding tasks demonstrate the superiority of TIC-GRPO.

**arXiv ID:** 2508.02833
</details>

<details>
<summary><strong>Guided Flow Policy: Learning from High-Value Actions in Offline Reinforcement Learning</strong> - Franki Nguimatsia Tiofack, Théotime Le Hellard, Fabian Schramm, Nicolas Perrin-Gilbert, Justin Carpentier - [[pdf]](https://arxiv.org/pdf/2512.03973)</summary>

**Abstract:** Offline reinforcement learning often relies on behavior regularization that enforces policies to remain close to the dataset distribution. However, such approaches fail to distinguish between high-value and low-value actions in their regularization components. We introduce Guided Flow Policy (GFP), which couples a multi-step flow-matching policy with a distilled one-step actor. The actor directs the flow policy through weighted behavior cloning to focus on cloning high-value actions from the dataset rather than indiscriminately imitating all state-action pairs. In turn, the flow policy constrains the actor to remain aligned with the dataset's best transitions while maximizing the critic. This mutual guidance enables GFP to achieve state-of-the-art performance across 144 state and pixel-based tasks from the OGBench, Minari, and D4RL benchmarks, with substantial gains on suboptimal datasets and challenging tasks. Webpage: this https URL

**arXiv ID:** 2512.03973
</details>

<details>
<summary><strong>Position: Beyond Model-Centric Prediction -- Agentic Time Series Forecasting</strong> - Mingyue Cheng, Xiaoyu Tao, Qi Liu, Ze Guo, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2602.01776)</summary>

**Abstract:** Time series forecasting has traditionally been formulated as a model-centric, static, and single-pass prediction problem that maps historical observations to future values. While this paradigm has driven substantial progress, it proves insufficient in adaptive and multi-turn settings where forecasting requires informative feature extraction, reasoning-driven inference, iterative refinement, and continual adaptation over time. In this paper, we argue for agentic time series forecasting (ATSF), which reframes forecasting as an agentic process composed of perception, planning, action, reflection, and memory. Rather than focusing solely on predictive models, ATSF emphasizes organizing forecasting as an agentic workflow that can interact with tools, incorporate feedback from outcomes, and evolve through experience accumulation. We outline three representative implementation paradigms -- workflow-based design, agentic reinforcement learning, and a hybrid agentic workflow paradigm -- and discuss the opportunities and challenges that arise when shifting from model-centric prediction to agentic forecasting. Together, this position aims to establish agentic forecasting as a foundation for future research at the intersection of time series forecasting.

**arXiv ID:** 2602.01776
</details>

<details>
<summary><strong>Risk-Aware Reinforcement Learning for Mobile Manipulation</strong> - Michael Groom, James Wilson, Nick Hawes, Lars Kunze - [[pdf]](https://arxiv.org/pdf/2603.04579)</summary>

**Abstract:** For robots to successfully transition from lab settings to everyday environments, they must begin to reason about the risks associated with their actions and make informed, risk-aware decisions. This is particularly true for robots performing mobile manipulation tasks, which involve both interacting with and navigating within dynamic, unstructured spaces. However, existing whole-body controllers for mobile manipulators typically lack explicit mechanisms for risk-sensitive decision-making under uncertainty. To our knowledge, we are the first to (i) learn risk-aware visuomotor policies for mobile manipulation conditioned on egocentric depth observations with runtime-adjustable risk sensitivity, and (ii) show risk-aware behaviours can be transferred through Imitation Learning (IL) to a visuomotor policy conditioned on egocentric depth observations. Our method achieves this by first training a privileged teacher policy using Distributional Reinforcement Learning (DRL), with a risk-neutral distributional critic. Distortion risk-metrics are then applied to the critic's predicted return distribution to calculate risk-adjusted advantage estimates used in policy updates to achieve a range of risk-aware behaviours. We then distil teacher policies with IL to obtain risk-aware student policies conditioned on egocentric depth observations. We perform extensive evaluations demonstrating that our trained visuomotor policies exhibit risk-aware behaviour (specifically achieving better worst-case performance) while performing reactive whole-body motions in unmapped environments, leveraging live depth observations for perception.

**arXiv ID:** 2603.04579
</details>

<details>
<summary><strong>VinePT-Map: Pole-Trunk Semantic Mapping for Resilient Autonomous Robotics in Vineyards</strong> - Giorgio Audrito, Mauro Martini, Alessandro Navone, Giorgia Galluzzo, Marcello Chiaberge - [[pdf]](https://arxiv.org/pdf/2603.05070)</summary>

**Abstract:** Reliable long-term deployment of autonomous robots in agricultural environments remains challenging due to perceptual aliasing, seasonal variability, and the dynamic nature of crop canopies. Vineyards, characterized by repetitive row structures and significant visual changes across phenological stages, represent a pivotal field challenge, limiting the robustness of conventional feature-based localization and mapping approaches. This paper introduces VinePT-Map, a semantic mapping framework that leverages vine trunks and support poles as persistent structural landmarks to enable season-agnostic and resilient robot localization. The proposed method formulates the mapping problem as a factor graph, integrating GPS, IMU, and RGB-D observations through robust geometrical constraints that exploit vineyard structure. An efficient perception pipeline based on instance segmentation and tracking, combined with a clustering filter for outlier rejection and pose refinement, enables accurate landmark detection using low-cost sensors and onboard computation. To validate the pipeline, we present a multi-season dataset for trunk and pole segmentation and tracking. Extensive field experiments conducted across diverse seasons demonstrate the robustness and accuracy of the proposed approach, highlighting its suitability for long-term autonomous operation in agricultural environments.

**arXiv ID:** 2603.05070
</details>

<details>
<summary><strong>Quadrotor Navigation using Reinforcement Learning with Privileged Information</strong> - Jonathan Lee, Abhishek Rathod, Kshitij Goel, John Stecklein, Wennie Tabib - [[pdf]](https://arxiv.org/pdf/2509.08177)</summary>

**Abstract:** This paper presents a reinforcement learning-based quadrotor navigation method that leverages efficient differentiable simulation, novel loss functions, and privileged information to navigate around large obstacles. Prior learning-based methods perform well in scenes that exhibit narrow obstacles, but struggle when the goal location is blocked by large walls or terrain. In contrast, the proposed method utilizes time-of-arrival (ToA) maps as privileged information and a yaw alignment loss to guide the robot around large obstacles. The policy is evaluated in photo-realistic simulation environments containing large obstacles, sharp corners, and dead-ends. Our approach achieves an 86% success rate and outperforms baseline strategies by 34%. We deploy the policy onboard a custom quadrotor in outdoor cluttered environments both during the day and night. The policy is validated across 20 flights, covering 589 meters without collisions at speeds up to 4 m/s.

**arXiv ID:** 2509.08177
</details>

<details>
<summary><strong>CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions</strong> - Lizhi Yang, Blake Werner, Massimiliano de Sa, Aaron D. Ames - [[pdf]](https://arxiv.org/pdf/2510.14959)</summary>

**Abstract:** Reinforcement learning (RL), while powerful and expressive, can often prioritize performance at the expense of safety. Yet safety violations can lead to catastrophic outcomes in real-world deployments. Control Barrier Functions (CBFs) offer a principled method to enforce dynamic safety -- traditionally deployed online via safety filters. While the result is safe behavior, the fact that the RL policy does not have knowledge of the CBF can lead to conservative behaviors. This paper proposes CBF-RL, a framework for generating safe behaviors with RL by enforcing CBFs in training. CBF-RL has two key attributes: (1) minimally modifying a nominal RL policy to encode safety constraints via a CBF term, (2) and safety filtering of the policy rollouts in training. Theoretically, we prove that continuous-time safety filters can be deployed via closed-form expressions on discrete-time roll-outs. Practically, we demonstrate that CBF-RL internalizes the safety constraints in the learned policy -- both enforcing safer actions and biasing towards safer rewards -- enabling safe deployment without the need for an online safety filter. We validate our framework through ablation studies on navigation tasks and on the Unitree G1 humanoid robot, where CBF-RL enables safer exploration, faster convergence, and robust performance under uncertainty, enabling the humanoid robot to avoid obstacles and climb stairs safely in real-world settings without a runtime safety filter.

**arXiv ID:** 2510.14959
</details>

<details>
<summary><strong>MarketGen: A Scalable Simulation Platform with Auto-Generated Embodied Supermarket Environments</strong> - Xu Hu, Yiyang Feng, Junran Peng, Jiawei He, Liyi Chen, Wei Sui, Chuanchen Luo, Xucheng Yin, Qing Li, Zhaoxiang Zhang - [[pdf]](https://arxiv.org/pdf/2511.21161)</summary>

**Abstract:** The development of embodied agents for complex commercial environments is hindered by a critical gap in existing robotics datasets and benchmarks, which primarily focus on household or tabletop settings with short-horizon tasks. To address this limitation, we introduce MarketGen, a scalable simulation platform with automatic scene generation for complex supermarket environments. MarketGen features a novel agent-based Procedural Content Generation (PCG) framework. It uniquely supports multi-modal inputs (text and reference images) and integrates real-world design principles to automatically generate complete, structured, and realistic supermarkets. We also provide an extensive and diverse 3D asset library with a total of 1100+ supermarket goods and parameterized facilities assets. Building on this generative foundation, we propose a novel benchmark for assessing supermarket agents, featuring two daily tasks in a supermarket: (1) Checkout Unloading: long-horizon tabletop tasks for cashier agents, and (2) In-Aisle Item Collection: complex mobile manipulation tasks for salesperson agents. We validate our platform and benchmark through extensive experiments, including the deployment of a modular agent system and successful sim-to-real transfer. MarketGen provides a comprehensive framework to accelerate research in embodied AI for complex commercial applications.

**arXiv ID:** 2511.21161
</details>

<details>
<summary><strong>Risk-Aware Autonomous Driving with Linear Temporal Logic Specifications</strong> - Shuhao Qi, Zengjie Zhang, Zhiyong Sun, Sofie Haesaert - [[pdf]](https://arxiv.org/pdf/2409.09769)</summary>

**Abstract:** Human drivers naturally balance the risks of different concerns while driving, including traffic rule violations, minor accidents, and fatalities. However, achieving the same behavior in autonomous driving systems remains an open problem. This paper extends a risk metric that has been verified in human-like driving studies to encompass more complex driving scenarios specified by linear temporal logic (LTL) that go beyond just collision risks. This extension incorporates the timing and severity of events into LTL specifications, thereby reflecting a human-like risk awareness. Without sacrificing expressivity for traffic rules, we adopt LTL specifications composed of safety and co-safety formulas, allowing the control synthesis problem to be reformulated as a reachability problem. By leveraging occupation measures, we further formulate a linear programming (LP) problem for this LTL-based risk metric. Consequently, the synthesized policy balances different types of driving risks, including both collision risks and traffic rule violations. The effectiveness of the proposed approach is validated by three typical traffic scenarios in Carla simulator.

**arXiv ID:** 2409.09769
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
