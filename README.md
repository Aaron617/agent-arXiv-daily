# Agent arXiv Daily

**Last Updated:** 2026-02-20 03:41:08

**Total Papers:** 73

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (3 papers)</h2></summary>

<details>
<summary><strong>Dynamic System Instructions and Tool Exposure for Efficient Agentic LLMs</strong> - Uria Franko - [[pdf]](https://arxiv.org/pdf/2602.17046)</summary>

**Abstract:** Large Language Model (LLM) agents often run for many steps while re-ingesting long system instructions and large tool catalogs each turn. This increases cost, agent derailment probability, latency, and tool-selection errors. We propose Instruction-Tool Retrieval (ITR), a RAG variant that retrieves, per step, only the minimal system-prompt fragments and the smallest necessary subset of tools. ITR composes a dynamic runtime system prompt and exposes a narrowed toolset with confidence-gated fallbacks. Using a controlled benchmark with internally consistent numbers, ITR reduces per-step context tokens by 95%, improves correct tool routing by 32% relative, and cuts end-to-end episode cost by 70% versus a monolithic baseline. These savings enable agents to run 2-20x more loops within context limits. Savings compound with the number of agent steps, making ITR particularly valuable for long-running autonomous agents. We detail the method, evaluation protocol, ablations, and operational guidance for practical deployment.

**arXiv ID:** 2602.17046
</details>

<details>
<summary><strong>The Bots of Persuasion: Examining How Conversational Agents' Linguistic Expressions of Personality Affect User Perceptions and Decisions</strong> - Uğur Genç, Heng Gu, Chadha Degachi, Evangelos Niforatos, Senthil Chandrasegaran, Himanshu Verma - [[pdf]](https://arxiv.org/pdf/2602.17185)</summary>

**Abstract:** Large Language Model-powered conversational agents (CAs) are increasingly capable of projecting sophisticated personalities through language, but how these projections affect users is unclear. We thus examine how CA personalities expressed linguistically affect user decisions and perceptions in the context of charitable giving. In a crowdsourced study, 360 participants interacted with one of eight CAs, each projecting a personality composed of three linguistic aspects: attitude (optimistic/pessimistic), authority (authoritative/submissive), and reasoning (emotional/rational). While the CA's composite personality did not affect participants' decisions, it did affect their perceptions and emotional responses. Particularly, participants interacting with pessimistic CAs felt lower emotional state and lower affinity towards the cause, perceived the CA as less trustworthy and less competent, and yet tended to donate more toward the charity. Perceptions of trust, competence, and situational empathy significantly predicted donation decisions. Our findings emphasize the risks CAs pose as instruments of manipulation, subtly influencing user perceptions and decisions.

**arXiv ID:** 2602.17185
</details>

<details>
<summary><strong>Toward a Fully Autonomous, AI-Native Particle Accelerator</strong> - Chris Tennant - [[pdf]](https://arxiv.org/pdf/2602.17536)</summary>

**Abstract:** This position paper presents a vision for self-driving particle accelerators that operate autonomously with minimal human intervention. We propose that future facilities be designed through artificial intelligence (AI) co-design, where AI jointly optimizes the accelerator lattice, diagnostics, and science application from inception to maximize performance while enabling autonomous operation. Rather than retrofitting AI onto human-centric systems, we envision facilities designed from the ground up as AI-native platforms. We outline nine critical research thrusts spanning agentic control architectures, knowledge integration, adaptive learning, digital twins, health monitoring, safety frameworks, modular hardware design, multimodal data fusion, and cross-domain collaboration. This roadmap aims to guide the accelerator community toward a future where AI-driven design and operation deliver unprecedented science output and reliability.

**arXiv ID:** 2602.17536
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (15 papers)</h2></summary>

<details>
<summary><strong>NeuDiff Agent: A Governed AI Workflow for Single-Crystal Neutron Crystallography</strong> - Zhongcan Xiao, Leyi Zhang, Guannan Zhang, Xiaoping Wang - [[pdf]](https://arxiv.org/pdf/2602.16812)</summary>

**Abstract:** Large-scale facilities increasingly face analysis and reporting latency as the limiting step in scientific throughput, particularly for structurally and magnetically complex samples that require iterative reduction, integration, refinement, and validation. To improve time-to-result and analysis efficiency, NeuDiff Agent is introduced as a governed, tool-using AI workflow for TOPAZ at the Spallation Neutron Source that takes instrument data products through reduction, integration, refinement, and validation to a validated crystal structure and a publication-ready CIF. NeuDiff Agent executes this established pipeline under explicit governance by restricting actions to allowlisted tools, enforcing fail-closed verification gates at key workflow boundaries, and capturing complete provenance for inspection, auditing, and controlled replay. Performance is assessed using a fixed prompt protocol and repeated end-to-end runs with two large language model backends, with user and machine time partitioned and intervention burden and recovery behaviors quantified under gating. In a reference-case benchmark, NeuDiff Agent reduces wall time from 435 minutes (manual) to 86.5(4.7) to 94.4(3.5) minutes (4.6-5.0x faster) while producing a validated CIF with no checkCIF level A or B alerts. These results establish a practical route to deploy agentic AI in facility crystallography while preserving traceability and publication-facing validation requirements.

**arXiv ID:** 2602.16812
</details>

<details>
<summary><strong>Narrow fine-tuning erodes safety alignment in vision-language agents</strong> - Idhant Gulati, Shivam Raval - [[pdf]](https://arxiv.org/pdf/2602.16931)</summary>

**Abstract:** Lifelong multimodal agents must continuously adapt to new tasks through post-training, but this creates fundamental tension between acquiring capabilities and preserving safety alignment. We demonstrate that fine-tuning aligned vision-language models on narrow-domain harmful datasets induces severe emergent misalignment that generalizes broadly across unrelated tasks and modalities. Through experiments on Gemma3-4B, we show that misalignment scales monotonically with LoRA rank, and that multimodal evaluation reveals substantially higher misalignment ($70.71 \pm 1.22$ at $r=128$) than text-only evaluation ($41.19 \pm 2.51$), suggesting that unimodal safety benchmarks may underestimate alignment degradation in vision-language models. Critically, even 10\% harmful data in the training mixture induces substantial alignment degradation. Geometric analysis reveals that harmful behaviors occupy a remarkably low-dimensional subspace, with the majority of misalignment information captured in 10 principal components. To mitigate misalignment, we evaluate two strategies: benign narrow fine-tuning and activation-based steering. While both approaches substantially reduce misalignment, neither completely removes the learned harmful behaviors. Our findings highlight the need for robust continual learning frameworks, as current post-training paradigms may not sufficiently preserve alignment in post-deployment settings.

**arXiv ID:** 2602.16931
</details>

<details>
<summary><strong>Sales Research Agent and Sales Research Bench</strong> - Deepanjan Bhol - [[pdf]](https://arxiv.org/pdf/2602.17017)</summary>

**Abstract:** Enterprises increasingly need AI systems that can answer sales-leader questions over live, customized CRM data, but most available models do not expose transparent, repeatable evidence of quality. This paper describes the Sales Research Agent in Microsoft Dynamics 365 Sales, an AI-first application that connects to live CRM and related data, reasons over complex schemas, and produces decision-ready insights through text and chart outputs. To make quality observable, we introduce the Sales Research Bench, a purpose-built benchmark that scores systems on eight customer-weighted dimensions, including text and chart groundedness, relevance, explainability, schema accuracy, and chart quality. In a 200-question run on a customized enterprise schema on October 19, 2025, the Sales Research Agent outperformed Claude Sonnet 4.5 by 13 points and ChatGPT-5 by 24.1 points on the 100-point composite score, giving customers a repeatable way to compare AI solutions.

**arXiv ID:** 2602.17017
</details>

<details>
<summary><strong>How AI Coding Agents Communicate: A Study of Pull Request Description Characteristics and Human Review Responses</strong> - Kan Watanabe, Rikuto Tsuchida, Takahiro Monno, Bin Huang, Kazuma Yamasaki, Youmei Fan, Kazumasa Shimari, Kenichi Matsumoto - [[pdf]](https://arxiv.org/pdf/2602.17084)</summary>

**Abstract:** The rapid adoption of large language models has led to the emergence of AI coding agents that autonomously create pull requests on GitHub. However, how these agents differ in their pull request description characteristics, and how human reviewers respond to them, remains underexplored. In this study, we conduct an empirical analysis of pull requests created by five AI coding agents using the AIDev dataset. We analyze agent differences in pull request description characteristics, including structural features, and examine human reviewer response in terms of review activity, response timing, sentiment, and merge outcomes. We find that AI coding agents exhibit distinct PR description styles, which are associated with differences in reviewer engagement, response time, and merge outcomes. We observe notable variation across agents in both reviewer interaction metrics and merge rates. These findings highlight the role of pull request presentation and reviewer interaction dynamics in human-AI collaborative software development.

**arXiv ID:** 2602.17084
</details>

<details>
<summary><strong>APEX-SQL: Talking to the data via Agentic Exploration for Text-to-SQL</strong> - Bowen Cao, Weibin Liao, Yushi Sun, Dong Fang, Haitao Li, Wai Lam - [[pdf]](https://arxiv.org/pdf/2602.16720)</summary>

**Abstract:** Text-to-SQL systems powered by Large Language Models have excelled on academic benchmarks but struggle in complex enterprise environments. The primary limitation lies in their reliance on static schema representations, which fails to resolve semantic ambiguity and scale effectively to large, complex databases. To address this, we propose APEX-SQL, an Agentic Text-to-SQL Framework that shifts the paradigm from passive translation to agentic exploration. Our framework employs a hypothesis-verification loop to ground model reasoning in real data. In the schema linking phase, we use logical planning to verbalize hypotheses, dual-pathway pruning to reduce the search space, and parallel data profiling to validate column roles against real data, followed by global synthesis to ensure topological connectivity. For SQL generation, we introduce a deterministic mechanism to retrieve exploration directives, allowing the agent to effectively explore data distributions, refine hypotheses, and generate semantically accurate SQLs. Experiments on BIRD (70.65% execution accuracy) and Spider 2.0-Snow (51.01% execution accuracy) demonstrate that APEX-SQL outperforms competitive baselines with reduced token consumption. Further analysis reveals that agentic exploration acts as a performance multiplier, unlocking the latent reasoning potential of foundation models in enterprise settings. Ablation studies confirm the critical contributions of each component in ensuring robust and accurate data analysis.

**arXiv ID:** 2602.16720
</details>

<details>
<summary><strong>Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History</strong> - Serin Kim, Sangam Lee, Dongha Lee - [[pdf]](https://arxiv.org/pdf/2602.17003)</summary>

**Abstract:** Large language models have advanced web agents, yet current agents lack personalization capabilities. Since users rarely specify every detail of their intent, practical web agents must be able to interpret ambiguous queries by inferring user preferences and contexts. To address this challenge, we present Persona2Web, the first benchmark for evaluating personalized web agents on the real open web, built upon the clarify-to-personalize principle, which requires agents to resolve ambiguity based on user history rather than relying on explicit instructions. Persona2Web consists of: (1) user histories that reveal preferences implicitly over long time spans, (2) ambiguous queries that require agents to infer implicit user preferences, and (3) a reasoning-aware evaluation framework that enables fine-grained assessment of personalization. We conduct extensive experiments across various agent architectures, backbone models, history access schemes, and queries with varying ambiguity levels, revealing key challenges in personalized web agent behavior. For reproducibility, our codes and datasets are publicly available at this https URL.

**arXiv ID:** 2602.17003
</details>

<details>
<summary><strong>Conditional Flow Matching for Continuous Anomaly Detection in Autonomous Driving on a Manifold-Aware Spectral Space</strong> - Antonio Guillen-Perez - [[pdf]](https://arxiv.org/pdf/2602.17586)</summary>

**Abstract:** Safety validation for Level 4 autonomous vehicles (AVs) is currently bottlenecked by the inability to scale the detection of rare, high-risk long-tail scenarios using traditional rule-based heuristics. We present Deep-Flow, an unsupervised framework for safety-critical anomaly detection that utilizes Optimal Transport Conditional Flow Matching (OT-CFM) to characterize the continuous probability density of expert human driving behavior. Unlike standard generative approaches that operate in unstable, high-dimensional coordinate spaces, Deep-Flow constrains the generative process to a low-rank spectral manifold via a Principal Component Analysis (PCA) bottleneck. This ensures kinematic smoothness by design and enables the computation of the exact Jacobian trace for numerically stable, deterministic log-likelihood estimation. To resolve multi-modal ambiguity at complex junctions, we utilize an Early Fusion Transformer encoder with lane-aware goal conditioning, featuring a direct skip-connection to the flow head to maintain intent-integrity throughout the network. We introduce a kinematic complexity weighting scheme that prioritizes high-energy maneuvers (quantified via path tortuosity and jerk) during the simulation-free training process. Evaluated on the Waymo Open Motion Dataset (WOMD), our framework achieves an AUC-ROC of 0.766 against a heuristic golden set of safety-critical events. More significantly, our analysis reveals a fundamental distinction between kinematic danger and semantic non-compliance. Deep-Flow identifies a critical predictability gap by surfacing out-of-distribution behaviors, such as lane-boundary violations and non-normative junction maneuvers, that traditional safety filters overlook. This work provides a mathematically rigorous foundation for defining statistical safety gates, enabling objective, data-driven validation for the safe deployment of autonomous fleets.

**arXiv ID:** 2602.17586
</details>

<details>
<summary><strong>Beyond Needle(s) in the Embodied Haystack: Environment, Architecture, and Training Considerations for Long Context Reasoning</strong> - Bosung Kim, Prithviraj Ammanabrolu - [[pdf]](https://arxiv.org/pdf/2505.16928)</summary>

**Abstract:** We introduce $\infty$-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI. $\infty$-THOR provides: (1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories; (2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents' long-context reasoning ability; and (3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences. To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction. Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions. Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning.

**arXiv ID:** 2505.16928
</details>

<details>
<summary><strong>Drones that Think on their Feet: Sudden Landing Decisions with Embodied AI</strong> - Diego Ortiz Barbosa, Mohit Agrawal, Yash Malegaonkar, Luis Burbano, Axel Andersson, György Dán, Henrik Sandberg, Alvaro A. Cardenas - [[pdf]](https://arxiv.org/pdf/2510.00167)</summary>

**Abstract:** Autonomous drones must often respond to sudden events, such as alarms, faults, or unexpected changes in their environment, that require immediate and adaptive decision-making. Traditional approaches rely on safety engineers hand-coding large sets of recovery rules, but this strategy cannot anticipate the vast range of real-world contingencies and quickly becomes incomplete. Recent advances in embodied AI, powered by large visual language models, provide commonsense reasoning to assess context and generate appropriate actions in real time. We demonstrate this capability in a simulated urban benchmark in the Unreal Engine, where drones dynamically interpret their surroundings and decide on sudden maneuvers for safe landings. Our results show that embodied AI makes possible a new class of adaptive recovery and decision-making pipelines that were previously infeasible to design by hand, advancing resilience and safety in autonomous aerial systems.

**arXiv ID:** 2510.00167
</details>

<details>
<summary><strong>Autonomous Business System via Neuro-symbolic AI</strong> - Cecil Pang, Hiroki Sayama - [[pdf]](https://arxiv.org/pdf/2601.15599)</summary>

**Abstract:** Current business environments demand continuous reconfiguration of cross-functional processes, yet enterprise systems remain organized around siloed departments, rigid workflows, and hard-coded automation. Meanwhile, large language models (LLMs) excel at interpreting natural language and unstructured data but lack deterministic and verifiable execution of complex business logic. We introduce Autonomous Business System (AUTOBUS), a system that combines LLM-based AI agents, predicate-logic programming, and business-semantics-centric enterprise data into a coherent neuro-symbolic architecture for executing end-to-end business initiatives. AUTOBUS models an initiative as a network of tasks with explicit pre- and post-conditions, required data, evaluation rules, and API-level actions. Enterprise data is represented as a knowledge graph whose entities, relationships, and constraints are translated into logic facts and foundational rules, providing semantic grounding for reasoning. Core AI agents synthesize task instructions, enterprise semantics, and available tools into task-specific logic programs executed by a logic engine that enforces constraints and orchestrates actions. Humans define semantics and policies, curate tools, and oversee high-impact or ambiguous decisions. We present the AUTOBUS architecture and a case study that demonstrates accelerated time to market in a data-rich organization. A reference implementation of the case study is available at this https URL.

**arXiv ID:** 2601.15599
</details>

<details>
<summary><strong>Autonomous Data Processing using Meta-Agents</strong> - Udayan Khurana - [[pdf]](https://arxiv.org/pdf/2602.00307)</summary>

**Abstract:** Traditional data processing pipelines are typically static and handcrafted for specific tasks, limiting their adaptability to evolving requirements. While general-purpose agents and coding assistants can generate code for well-understood data pipelines, they lack the ability to autonomously monitor, manage, and optimize an end-to-end pipeline once deployed. We present \textbf{Autonomous Data Processing using Meta-agents} (ADP-MA), a framework that dynamically constructs, executes, and iteratively refines data processing pipelines through hierarchical agent orchestration. At its core, \textit{meta-agents} analyze input data and task specifications to design a multi-phase plan, instantiate specialized \textit{ground-level agents}, and continuously evaluate pipeline performance. The architecture comprises three key components: a planning module for strategy generation, an orchestration layer for agent coordination and tool integration, and a monitoring loop for iterative evaluation and backtracking. Unlike conventional approaches, ADP-MA emphasizes context-aware optimization, adaptive workload partitioning, and progressive sampling for scalability. Additionally, the framework leverages a diverse set of external tools and can reuse previously designed agents, reducing redundancy and accelerating pipeline construction. We demonstrate ADP-MA through an interactive demo that showcases pipeline construction, execution monitoring, and adaptive refinement across representative data processing tasks.

**arXiv ID:** 2602.00307
</details>

<details>
<summary><strong>Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis</strong> - Pengfei Zhang, Tianxin Xie, Minghao Yang, Li Liu - [[pdf]](https://arxiv.org/pdf/2602.15909)</summary>

**Abstract:** Deep learning-based respiratory auscultation is currently hindered by two fundamental challenges: (i) inherent information loss, as converting signals into spectrograms discards transient acoustic events and clinical context; (ii) limited data availability, exacerbated by severe class imbalance. To bridge these gaps, we present Resp-Agent, an autonomous multimodal system orchestrated by a novel Active Adversarial Curriculum Agent (Thinker-A$^2$CA). Unlike static pipelines, Thinker-A$^2$CA serves as a central controller that actively identifies diagnostic weaknesses and schedules targeted synthesis in a closed loop. To address the representation gap, we introduce a Modality-Weaving Diagnoser that weaves EHR data with audio tokens via Strategic Global Attention and sparse audio anchors, capturing both long-range clinical context and millisecond-level transients. To address the data gap, we design a Flow Matching Generator that adapts a text-only Large Language Model (LLM) via modality injection, decoupling pathological content from acoustic style to synthesize hard-to-diagnose samples. As a foundation for these efforts, we introduce Resp-229k, a benchmark corpus of 229k recordings paired with LLM-distilled clinical narratives. Extensive experiments demonstrate that Resp-Agent consistently outperforms prior approaches across diverse evaluation settings, improving diagnostic robustness under data scarcity and long-tailed class imbalance. Our code and data are available at this https URL.

**arXiv ID:** 2602.15909
</details>

<details>
<summary><strong>Hybrid-Gym: Training Coding Agents to Generalize Across Tasks</strong> - Yiqing Xie, Emmy Liu, Gaokai Zhang, Nachiket Kotalwar, Shubham Gandhi, Sathwik Acharya, Xingyao Wang, Carolyn Rose, Graham Neubig, Daniel Fried - [[pdf]](https://arxiv.org/pdf/2602.16819)</summary>

**Abstract:** When assessing the quality of coding agents, predominant benchmarks focus on solving single issues on GitHub, such as SWE-Bench. In contrast, in real use, these agents solve more various and complex tasks that involve other skills such as exploring codebases, testing software, and designing architecture. In this paper, we first characterize some transferable skills that are shared across diverse tasks by decomposing trajectories into fine-grained components, and derive a set of principles for designing auxiliary training tasks to teach language models these skills. Guided by these principles, we propose a training environment, Hybrid-Gym, consisting of a set of scalable synthetic tasks, such as function localization and dependency search. Experiments show that agents trained on our synthetic tasks effectively generalize to diverse real-world tasks that are not present in training, improving a base model by 25.4% absolute gain on SWE-Bench Verified, 7.9% on SWT-Bench Verified, and 5.1% on Commit-0 Lite. Hybrid-Gym also complements datasets built for the downstream tasks (e.g., improving SWE-Play by 4.9% on SWT-Bench Verified). Code available at: this https URL.

**arXiv ID:** 2602.16819
</details>

<details>
<summary><strong>Boreas Road Trip: A Multi-Sensor Autonomous Driving Dataset on Challenging Roads</strong> - Daniil Lisus, Katya M. Papais, Cedric Le Gentil, Elliot Preston-Krebs, Andrew Lambert, Keith Y.K. Leung, Timothy D. Barfoot - [[pdf]](https://arxiv.org/pdf/2602.16870)</summary>

**Abstract:** The Boreas Road Trip (Boreas-RT) dataset extends the multi-season Boreas dataset to new and diverse locations that pose challenges for modern autonomous driving algorithms. Boreas-RT comprises 60 sequences collected over 9 real-world routes, totalling 643 km of driving. Each route is traversed multiple times, enabling evaluation in identical environments under varying traffic and, in some cases, weather conditions. The data collection platform includes a 5MP FLIR Blackfly S camera, a 360 degree Navtech RAS6 Doppler-enabled spinning radar, a 128-channel 360 degree Velodyne Alpha Prime lidar, an Aeva Aeries II FMCW Doppler-enabled lidar, a Silicon Sensing DMU41 inertial measurement unit, and a Dynapar wheel encoder. Centimetre-level ground truth is provided via post-processed Applanix POS LV GNSS-INS data. The dataset includes precise extrinsic and intrinsic calibrations, a publicly available development kit, and a live leaderboard for odometry and metric localization. Benchmark results show that many state-of-the-art odometry and localization algorithms overfit to simple driving environments and degrade significantly on the more challenging Boreas-RT routes. Boreas-RT provides a unified dataset for evaluating multi-modal algorithms across diverse road conditions. The dataset, leaderboard, and development kit are available at this http URL.

**arXiv ID:** 2602.16870
</details>

<details>
<summary><strong>RoboGene: Boosting VLA Pre-training via Diversity-Driven Agentic Framework for Real-World Task Generation</strong> - Yixue Zhang, Kun Wu, Zhi Gao, Zhen Zhao, Pei Ren, Zhiyuan Xu, Fei Liao, Xinhua Wang, Shichao Fan, Di Wu, Qiuxuan Feng, Meng Li, Zhengping Che, Chang Liu, Jian Tang - [[pdf]](https://arxiv.org/pdf/2602.16444)</summary>

**Abstract:** The pursuit of general-purpose robotic manipulation is hindered by the scarcity of diverse, real-world interaction data. Unlike data collection from web in vision or language, robotic data collection is an active process incurring prohibitive physical costs. Consequently, automated task curation to maximize data value remains a critical yet under-explored challenge. Existing manual methods are unscalable and biased toward common tasks, while off-the-shelf foundation models often hallucinate physically infeasible instructions. To address this, we introduce RoboGene, an agentic framework designed to automate the generation of diverse, physically plausible manipulation tasks across single-arm, dual-arm, and mobile robots. RoboGene integrates three core components: diversity-driven sampling for broad task coverage, self-reflection mechanisms to enforce physical constraints, and human-in-the-loop refinement for continuous improvement. We conduct extensive quantitative analysis and large-scale real-world experiments, collecting datasets of 18k trajectories and introducing novel metrics to assess task quality, feasibility, and diversity. Results demonstrate that RoboGene significantly outperforms state-of-the-art foundation models (e.g., GPT-4o, Gemini 2.5 Pro). Furthermore, real-world experiments show that VLA models pre-trained with RoboGene achieve higher success rates and superior generalization, underscoring the importance of high-quality task generation. Our project is available at this https URL.

**arXiv ID:** 2602.16444
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>AgentLAB: Benchmarking LLM Agents against Long-Horizon Attacks</strong> - Tanqiu Jiang, Yuhui Wang, Jiacheng Liang, Ting Wang - [[pdf]](https://arxiv.org/pdf/2602.16901)</summary>

**Abstract:** LLM agents are increasingly deployed in long-horizon, complex environments to solve challenging problems, but this expansion exposes them to long-horizon attacks that exploit multi-turn user-agent-environment interactions to achieve objectives infeasible in single-turn settings. To measure agent vulnerabilities to such risks, we present AgentLAB, the first benchmark dedicated to evaluating LLM agent susceptibility to adaptive, long-horizon attacks. Currently, AgentLAB supports five novel attack types including intent hijacking, tool chaining, task injection, objective drifting, and memory poisoning, spanning 28 realistic agentic environments, and 644 security test cases. Leveraging AgentLAB, we evaluate representative LLM agents and find that they remain highly susceptible to long-horizon attacks; moreover, defenses designed for single-turn interactions fail to reliably mitigate long-horizon threats. We anticipate that AgentLAB will serve as a valuable benchmark for tracking progress on securing LLM agents in practical settings. The benchmark is publicly available at this https URL.

**arXiv ID:** 2602.16901
</details>

<details>
<summary><strong>Mind the GAP: Text Safety Does Not Transfer to Tool-Call Safety in LLM Agents</strong> - Arnold Cartagena, Ariane Teixeira - [[pdf]](https://arxiv.org/pdf/2602.16943)</summary>

**Abstract:** Large language models deployed as agents increasingly interact with external systems through tool calls--actions with real-world consequences that text outputs alone do not carry. Safety evaluations, however, overwhelmingly measure text-level refusal behavior, leaving a critical question unanswered: does alignment that suppresses harmful text also suppress harmful actions? We introduce the GAP benchmark, a systematic evaluation framework that measures divergence between text-level safety and tool-call-level safety in LLM agents. We test six frontier models across six regulated domains (pharmaceutical, financial, educational, employment, legal, and infrastructure), seven jailbreak scenarios per domain, three system prompt conditions (neutral, safety-reinforced, and tool-encouraging), and two prompt variants, producing 17,420 analysis-ready datapoints. Our central finding is that text safety does not transfer to tool-call safety. Across all six models, we observe instances where the model's text output refuses a harmful request while its tool calls simultaneously execute the forbidden action--a divergence we formalize as the GAP metric. Even under safety-reinforced system prompts, 219 such cases persist across all six models. System prompt wording exerts substantial influence on tool-call behavior: TC-safe rates span 21 percentage points for the most robust model and 57 for the most prompt-sensitive, with 16 of 18 pairwise ablation comparisons remaining significant after Bonferroni correction. Runtime governance contracts reduce information leakage in all six models but produce no detectable deterrent effect on forbidden tool-call attempts themselves. These results demonstrate that text-only safety evaluations are insufficient for assessing agent behavior and that tool-call safety requires dedicated measurement and mitigation.

**arXiv ID:** 2602.16943
</details>

<details>
<summary><strong>LLM4Cov: Execution-Aware Agentic Learning for High-coverage Testbench Generation</strong> - Hejia Zhang, Zhongming Yu, Chia-Tung Ho, Haoxing Ren, Brucek Khailany, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2602.16953)</summary>

**Abstract:** Execution-aware LLM agents offer a promising paradigm for learning from tool feedback, but such feedback is often expensive and slow to obtain, making online reinforcement learning (RL) impractical. High-coverage hardware verification exemplifies this challenge due to its reliance on industrial simulators and non-differentiable execution signals. We propose LLM4Cov, an offline agent-learning framework that models verification as memoryless state transitions guided by deterministic evaluators. Building on this formulation, we introduce execution-validated data curation, policy-aware agentic data synthesis, and worst-state-prioritized sampling to enable scalable learning under execution constraints. We further curate a reality-aligned benchmark adapted from an existing verification suite through a revised evaluation protocol. Using the proposed pipeline, a compact 4B-parameter model achieves 69.2% coverage pass rate under agentic evaluation, outperforming its teacher by 5.3% and demonstrating competitive performance against models an order of magnitude larger.

**arXiv ID:** 2602.16953
</details>

<details>
<summary><strong>Automating Agent Hijacking via Structural Template Injection</strong> - Xinhao Deng, Jiaqing Wu, Miao Chen, Yue Xiao, Ke Xu, Qi Li - [[pdf]](https://arxiv.org/pdf/2602.16958)</summary>

**Abstract:** Agent hijacking, highlighted by OWASP as a critical threat to the Large Language Model (LLM) ecosystem, enables adversaries to manipulate execution by injecting malicious instructions into retrieved content. Most existing attacks rely on manually crafted, semantics-driven prompt manipulation, which often yields low attack success rates and limited transferability to closed-source commercial models. In this paper, we propose Phantom, an automated agent hijacking framework built upon Structured Template Injection that targets the fundamental architectural mechanisms of LLM agents. Our key insight is that agents rely on specific chat template tokens to separate system, user, assistant, and tool instructions. By injecting optimized structured templates into the retrieved context, we induce role confusion and cause the agent to misinterpret the injected content as legitimate user instructions or prior tool outputs. To enhance attack transferability against black-box agents, Phantom introduces a novel attack template search framework. We first perform multi-level template augmentation to increase structural diversity and then train a Template Autoencoder (TAE) to embed discrete templates into a continuous, searchable latent space. Subsequently, we apply Bayesian optimization to efficiently identify optimal adversarial vectors that are decoded into high-potency structured templates. Extensive experiments on Qwen, GPT, and Gemini demonstrate that our framework significantly outperforms existing baselines in both Attack Success Rate (ASR) and query efficiency. Moreover, we identified over 70 vulnerabilities in real-world commercial products that have been confirmed by vendors, underscoring the practical severity of structured template-based hijacking and providing an empirical foundation for securing next-generation agentic systems.

**arXiv ID:** 2602.16958
</details>

<details>
<summary><strong>Phase-Aware Mixture of Experts for Agentic Reinforcement Learning</strong> - Shengtian Yang, Yu Li, Shuo He, Yewen Li, Qingpeng Cai, Peng Jiang, Lei Feng - [[pdf]](https://arxiv.org/pdf/2602.17038)</summary>

**Abstract:** Reinforcement learning (RL) has equipped LLM agents with a strong ability to solve complex tasks. However, existing RL methods normally use a \emph{single} policy network, causing \emph{simplicity bias} where simple tasks occupy most parameters and dominate gradient updates, leaving insufficient capacity for complex tasks. A plausible remedy could be employing the Mixture-of-Experts (MoE) architecture in the policy network, as MoE allows different parameters (experts) to specialize in different tasks, preventing simple tasks from dominating all parameters. However, a key limitation of traditional MoE is its token-level routing, where the router assigns each token to specialized experts, which fragments phase-consistent patterns into scattered expert assignments and thus undermines expert specialization. In this paper, we propose \textbf{Phase-Aware Mixture of Experts (PA-MoE)}. It first features a lightweight \emph{phase router} that learns latent phase boundaries directly from the RL objective without pre-defining phase categories. Then, the phase router allocates temporally consistent assignments to the same expert, allowing experts to preserve phase-specific expertise. Experimental results demonstrate the effectiveness of our proposed PA-MoE.

**arXiv ID:** 2602.17038
</details>

<details>
<summary><strong>KLong: Training LLM Agent for Extremely Long-horizon Tasks</strong> - Yue Liu, Zhiyuan Hu, Flood Sung, Jiaheng Zhang, Bryan Hooi - [[pdf]](https://arxiv.org/pdf/2602.17547)</summary>

**Abstract:** This paper introduces KLong, an open-source LLM agent trained to solve extremely long-horizon tasks. The principle is to first cold-start the model via trajectory-splitting SFT, then scale it via progressive RL training. Specifically, we first activate basic agentic abilities of a base model with a comprehensive SFT recipe. Then, we introduce Research-Factory, an automated pipeline that generates high-quality training data by collecting research papers and constructing evaluation rubrics. Using this pipeline, we build thousands of long-horizon trajectories distilled from Claude 4.5 Sonnet (Thinking). To train with these extremely long trajectories, we propose a new trajectory-splitting SFT, which preserves early context, progressively truncates later context, and maintains overlap between sub-trajectories. In addition, to further improve long-horizon task-solving capability, we propose a novel progressive RL, which schedules training into multiple stages with progressively extended timeouts. Experiments demonstrate the superiority and generalization of KLong, as shown in Figure 1. Notably, our proposed KLong (106B) surpasses Kimi K2 Thinking (1T) by 11.28% on PaperBench, and the performance improvement generalizes to other coding benchmarks like SWE-bench Verified and MLE-bench.

**arXiv ID:** 2602.17547
</details>

<details>
<summary><strong>Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents</strong> - Gil Pasternak, Dheeraj Rajagopal, Julia White, Dhruv Atreja, Matthew Thomas, George Hurn-Maloney, Ash Lewis - [[pdf]](https://arxiv.org/pdf/2510.19771)</summary>

**Abstract:** LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. To address this gap, we present PROBE (Proactive Resolution Of BottlEnecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and expose promising future research directions.

**arXiv ID:** 2510.19771
</details>

<details>
<summary><strong>Bridging Symbolic Control and Neural Reasoning in LLM Agents: Structured Cognitive Loop with a Governance Layer</strong> - Myung Ho Kim - [[pdf]](https://arxiv.org/pdf/2511.17673)</summary>

**Abstract:** Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). Soft Symbolic Control constitutes a dedicated governance layer within SCL, applying symbolic constraints to probabilistic inference while preserving the flexibility of neural reasoning and restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents.

**arXiv ID:** 2511.17673
</details>

<details>
<summary><strong>Helpful to a Fault: Measuring Illicit Assistance in Multi-Turn, Multilingual LLM Agents</strong> - Nivya Talokar, Ayush K Tarun, Murari Mandal, Maksym Andriushchenko, Antoine Bosselut - [[pdf]](https://arxiv.org/pdf/2602.16346)</summary>

**Abstract:** LLM-based agents execute real-world workflows via tools and memory. These affordances enable ill-intended adversaries to also use these agents to carry out complex misuse scenarios. Existing agent misuse benchmarks largely test single-prompt instructions, leaving a gap in measuring how agents end up helping with harmful or illegal tasks over multiple turns. We introduce STING (Sequential Testing of Illicit N-step Goal execution), an automated red-teaming framework that constructs a step-by-step illicit plan grounded in a benign persona and iteratively probes a target agent with adaptive follow-ups, using judge agents to track phase completion. We further introduce an analysis framework that models multi-turn red-teaming as a time-to-first-jailbreak random variable, enabling analysis tools like discovery curves, hazard-ratio attribution by attack language, and a new metric: Restricted Mean Jailbreak Discovery. Across AgentHarm scenarios, STING yields substantially higher illicit-task completion than single-turn prompting and chat-oriented multi-turn baselines adapted to tool-using agents. In multilingual evaluations across six non-English settings, we find that attack success and illicit-task completion do not consistently increase in lower-resource languages, diverging from common chatbot findings. Overall, STING provides a practical way to evaluate and stress-test agent misuse in realistic deployment settings, where interactions are inherently multi-turn and often multilingual.

**arXiv ID:** 2602.16346
</details>

<details>
<summary><strong>Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents</strong> - Wenxuan Ding, Nicholas Tomlin, Greg Durrett - [[pdf]](https://arxiv.org/pdf/2602.16699)</summary>

**Abstract:** LLMs are increasingly being used for complex problems which are not necessarily resolved in a single response, but require interacting with an environment to acquire information. In these scenarios, LLMs must reason about inherent cost-uncertainty tradeoffs in when to stop exploring and commit to an answer. For instance, on a programming task, an LLM should test a generated code snippet if it is uncertain about the correctness of that code; the cost of writing a test is nonzero, but typically lower than the cost of making a mistake. In this work, we show that we can induce LLMs to explicitly reason about balancing these cost-uncertainty tradeoffs, then perform more optimal environment exploration. We formalize multiple tasks, including information retrieval and coding, as sequential decision-making problems under uncertainty. Each problem has latent environment state that can be reasoned about via a prior which is passed to the LLM agent. We introduce a framework called Calibrate-Then-Act (CTA), where we feed the LLM this additional context to enable it to act more optimally. This improvement is preserved even under RL training of both the baseline and CTA. Our results on information-seeking QA and on a simplified coding task show that making cost-benefit tradeoffs explicit with CTA can help agents discover more optimal decision-making strategies.

**arXiv ID:** 2602.16699
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (20 papers)</h2></summary>

<details>
<summary><strong>Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents</strong> - Haiyang Xu, Xi Zhang, Haowei Liu, Junyang Wang, Zhaozai Zhu, Shengjie Zhou, Xuhao Hu, Feiyu Gao, Junjie Cao, Zihua Wang, Zhiyuan Chen, Jitong Liao, Qi Zheng, Jiahui Zeng, Ze Xu, Shuai Bai, Junyang Lin, Jingren Zhou, Ming Yan - [[pdf]](https://arxiv.org/pdf/2602.16855)</summary>

**Abstract:** The paper introduces GUI-Owl-1.5, the latest native GUI agent model that features instruct/thinking variants in multiple sizes (2B/4B/8B/32B/235B) and supports a range of platforms (desktop, mobile, browser, and more) to enable cloud-edge collaboration and real-time interaction. GUI-Owl-1.5 achieves state-of-the-art results on more than 20+ GUI benchmarks on open-source models: (1) on GUI automation tasks, it obtains 56.5 on OSWorld, 71.6 on AndroidWorld, and 48.4 on WebArena; (2) on grounding tasks, it obtains 80.3 on ScreenSpotPro; (3) on tool-calling tasks, it obtains 47.6 on OSWorld-MCP, and 46.8 on MobileWorld; (4) on memory and knowledge tasks, it obtains 75.5 on GUI-Knowledge Bench. GUI-Owl-1.5 incorporates several key innovations: (1) Hybird Data Flywheel: we construct the data pipeline for UI understanding and trajectory generation based on a combination of simulated environments and cloud-based sandbox environments, in order to improve the efficiency and quality of data collection. (2) Unified Enhancement of Agent Capabilities: we use a unified thought-synthesis pipeline to enhance the model's reasoning capabilities, while placing particular emphasis on improving key agent abilities, including Tool/MCP use, memory and multi-agent adaptation; (3) Multi-platform Environment RL Scaling: We propose a new environment RL algorithm, MRPO, to address the challenges of multi-platform conflicts and the low training efficiency of long-horizon tasks. The GUI-Owl-1.5 models are open-sourced, and an online cloud-sandbox demo is available at this https URL.

**arXiv ID:** 2602.16855
</details>

<details>
<summary><strong>IntentCUA: Learning Intent-level Representations for Skill Abstraction and Multi-Agent Planning in Computer-Use Agents</strong> - Seoyoung Lee, Seobin Yoon, Seongbeen Lee, Yoojung Chun, Dayoung Park, Doyeon Kim, Joo Yong Sim - [[pdf]](https://arxiv.org/pdf/2602.17049)</summary>

**Abstract:** Computer-use agents operate over long horizons under noisy perception, multi-window contexts, evolving environment states. Existing approaches, from RL-based planners to trajectory retrieval, often drift from user intent and repeatedly solve routine subproblems, leading to error accumulation and inefficiency. We present IntentCUA, a multi-agent computer-use framework designed to stabilize long-horizon execution through intent-aligned plan memory. A Planner, Plan-Optimizer, and Critic coordinate over shared memory that abstracts raw interaction traces into multi-view intent representations and reusable skills. At runtime, intent prototypes retrieve subgroup-aligned skills and inject them into partial plans, reducing redundant re-planning and mitigating error propagation across desktop applications. In end-to-end evaluations, IntentCUA achieved a 74.83% task success rate with a Step Efficiency Ratio of 0.91, outperforming RL-based and trajectory-centric baselines. Ablations show that multi-view intent abstraction and shared plan memory jointly improve execution stability, with the cooperative multi-agent loop providing the largest gains on long-horizon tasks. These results highlight that system-level intent abstraction and memory-grounded coordination are key to reliable and efficient desktop automation in large, dynamic environments.

**arXiv ID:** 2602.17049
</details>

<details>
<summary><strong>Retaining Suboptimal Actions to Follow Shifting Optima in Multi-Agent Reinforcement Learning</strong> - Yonghyeon Jo, Sunwoo Lee, Seungyul Han - [[pdf]](https://arxiv.org/pdf/2602.17062)</summary>

**Abstract:** Value decomposition is a core approach for cooperative multi-agent reinforcement learning (MARL). However, existing methods still rely on a single optimal action and struggle to adapt when the underlying value function shifts during training, often converging to suboptimal policies. To address this limitation, we propose Successive Sub-value Q-learning (S2Q), which learns multiple sub-value functions to retain alternative high-value actions. Incorporating these sub-value functions into a Softmax-based behavior policy, S2Q encourages persistent exploration and enables $Q^{\text{tot}}$ to adjust quickly to the changing optima. Experiments on challenging MARL benchmarks confirm that S2Q consistently outperforms various MARL algorithms, demonstrating improved adaptability and overall performance. Our code is available at this https URL.

**arXiv ID:** 2602.17062
</details>

<details>
<summary><strong>AutoNumerics: An Autonomous, PDE-Agnostic Multi-Agent Pipeline for Scientific Computing</strong> - Jianda Du, Youran Sun, Haizhao Yang - [[pdf]](https://arxiv.org/pdf/2602.17607)</summary>

**Abstract:** PDEs are central to scientific and engineering modeling, yet designing accurate numerical solvers typically requires substantial mathematical expertise and manual tuning. Recent neural network-based approaches improve flexibility but often demand high computational cost and suffer from limited interpretability. We introduce \texttt{AutoNumerics}, a multi-agent framework that autonomously designs, implements, debugs, and verifies numerical solvers for general PDEs directly from natural language descriptions. Unlike black-box neural solvers, our framework generates transparent solvers grounded in classical numerical analysis. We introduce a coarse-to-fine execution strategy and a residual-based self-verification mechanism. Experiments on 24 canonical and real-world PDE problems demonstrate that \texttt{AutoNumerics} achieves competitive or superior accuracy compared to existing neural and LLM-based baselines, and correctly selects numerical schemes based on PDE structural properties, suggesting its viability as an accessible paradigm for automated PDE solving.

**arXiv ID:** 2602.17607
</details>

<details>
<summary><strong>AdaptOrch: Task-Adaptive Multi-Agent Orchestration in the Era of LLM Performance Convergence</strong> - Geunbin Yu - [[pdf]](https://arxiv.org/pdf/2602.16873)</summary>

**Abstract:** As large language models from diverse providers converge toward comparable benchmark performance, the traditional paradigm of selecting a single best model per task yields diminishing returns. We argue that orchestration topology -- the structural composition of how multiple agents are coordinated, parallelized, and synthesized -- now dominates system-level performance over individual model capability. We present AdaptOrch, a formal framework for task-adaptive multi-agent orchestration that dynamically selects among four canonical topologies (parallel, sequential, hierarchical, and hybrid) based on task dependency graphs and empirically derived domain characteristics. Our framework introduces three key contributions: (1) a Performance Convergence Scaling Law, formalizing conditions under which orchestration selection outweighs model selection; (2) a Topology Routing Algorithm that maps task decomposition DAGs to optimal orchestration patterns in O(|V| + |E|) time; and (3) an Adaptive Synthesis Protocol with provable termination guarantees and heuristic consistency scoring for parallel agent outputs. We validate AdaptOrch across coding (SWE-bench), reasoning (GPQA), and retrieval-augmented generation tasks, demonstrating that topology-aware orchestration achieves 12-23% improvement over static single-topology baselines, even when using identical underlying models. Our results establish orchestration design as a first-class optimization target independent of model scaling.

**arXiv ID:** 2602.16873
</details>

<details>
<summary><strong>MALLVI: a multi agent framework for integrated generalized robotics manipulation</strong> - Iman Ahmadi, Mehrshad Taji, Arad Mahdinezhad Kashani, AmirHossein Jadidi, Saina Kashani, Babak Khalaj - [[pdf]](https://arxiv.org/pdf/2602.16898)</summary>

**Abstract:** Task planning for robotic manipulation with large language models (LLMs) is an emerging area. Prior approaches rely on specialized models, fine tuning, or prompt tuning, and often operate in an open loop manner without robust environmental feedback, making them fragile in dynamic this http URL present MALLVi, a Multi Agent Large Language and Vision framework that enables closed loop feedback driven robotic manipulation. Given a natural language instruction and an image of the environment, MALLVi generates executable atomic actions for a robot manipulator. After action execution, a Vision Language Model (VLM) evaluates environmental feedback and decides whether to repeat the process or proceed to the next this http URL than using a single model, MALLVi coordinates specialized agents, Decomposer, Localizer, Thinker, and Reflector, to manage perception, localization, reasoning, and high level planning. An optional Descriptor agent provides visual memory of the initial state. The Reflector supports targeted error detection and recovery by reactivating only relevant agents, avoiding full this http URL in simulation and real world settings show that iterative closed loop multi agent coordination improves generalization and increases success rates in zero shot manipulation this http URL available at this https URL.

**arXiv ID:** 2602.16898
</details>

<details>
<summary><strong>Discovering Multiagent Learning Algorithms with Large Language Models</strong> - Zun Li, John Schultz, Daniel Hennes, Marc Lanctot - [[pdf]](https://arxiv.org/pdf/2602.16928)</summary>

**Abstract:** Much of the advancement of Multi-Agent Reinforcement Learning (MARL) in imperfect-information games has historically depended on manual iterative refinement of baselines. While foundational families like Counterfactual Regret Minimization (CFR) and Policy Space Response Oracles (PSRO) rest on solid theoretical ground, the design of their most effective variants often relies on human intuition to navigate a vast algorithmic design space. In this work, we propose the use of AlphaEvolve, an evolutionary coding agent powered by large language models, to automatically discover new multiagent learning algorithms. We demonstrate the generality of this framework by evolving novel variants for two distinct paradigms of game-theoretic learning. First, in the domain of iterative regret minimization, we evolve the logic governing regret accumulation and policy derivation, discovering a new algorithm, Volatility-Adaptive Discounted (VAD-)CFR. VAD-CFR employs novel, non-intuitive mechanisms-including volatility-sensitive discounting, consistency-enforced optimism, and a hard warm-start policy accumulation schedule-to outperform state-of-the-art baselines like Discounted Predictive CFR+. Second, in the regime of population based training algorithms, we evolve training-time and evaluation-time meta strategy solvers for PSRO, discovering a new variant, Smoothed Hybrid Optimistic Regret (SHOR-)PSRO. SHOR-PSRO introduces a hybrid meta-solver that linearly blends Optimistic Regret Matching with a smoothed, temperature-controlled distribution over best pure strategies. By dynamically annealing this blending factor and diversity bonuses during training, the algorithm automates the transition from population diversity to rigorous equilibrium finding, yielding superior empirical convergence compared to standard static meta-solvers.

**arXiv ID:** 2602.16928
</details>

<details>
<summary><strong>Puzzle it Out: Local-to-Global World Model for Offline Multi-Agent Reinforcement Learning</strong> - Sijia li, Xinran Li, Shibo Chen, Jun Zhang - [[pdf]](https://arxiv.org/pdf/2601.07463)</summary>

**Abstract:** Offline multi-agent reinforcement learning (MARL) aims to solve cooperative decision-making problems in multi-agent systems using pre-collected datasets. Existing offline MARL methods primarily constrain training within the dataset distribution, resulting in overly conservative policies that struggle to generalize beyond the support of the data. While model-based approaches offer a promising solution by expanding the original dataset with synthetic data generated from a learned world model, the high dimensionality, non-stationarity, and complexity of multi-agent systems make it challenging to accurately estimate the transitions and reward functions in offline MARL. Given the difficulty of directly modeling joint dynamics, we propose a local-to-global (LOGO) world model, a novel framework that leverages local predictions-which are easier to estimate-to infer global state dynamics, thus improving prediction accuracy while implicitly capturing agent-wise dependencies. Using the trained world model, we generate synthetic data to augment the original dataset, expanding the effective state-action space. To ensure reliable policy learning, we further introduce an uncertainty-aware sampling mechanism that adaptively weights synthetic data by prediction uncertainty, reducing approximation error propagation to policies. In contrast to conventional ensemble-based methods, our approach requires only an additional encoder for uncertainty estimation, significantly reducing computational overhead while maintaining accuracy. Extensive experiments across 8 scenarios against 8 baselines demonstrate that our method surpasses state-of-the-art baselines on standard offline MARL benchmarks, establishing a new model-based baseline for generalizable offline multi-agent learning.

**arXiv ID:** 2601.07463
</details>

<details>
<summary><strong>Self-Evolving Multi-Agent Network for Industrial IoT Predictive Maintenance</strong> - Rebin Saleh, Khanh Pham Dinh, Balázs Villányi, Truong-Son Hy - [[pdf]](https://arxiv.org/pdf/2602.16738)</summary>

**Abstract:** Industrial IoT predictive maintenance requires systems capable of real-time anomaly detection without sacrificing interpretability or demanding excessive computational resources. Traditional approaches rely on static, offline-trained models that cannot adapt to evolving operational conditions, while LLM-based monolithic systems demand prohibitive memory and latency, rendering them impractical for on-site edge deployment. We introduce SEMAS, a self-evolving hierarchical multi-agent system that distributes specialized agents across Edge, Fog, and Cloud computational tiers. Edge agents perform lightweight feature extraction and pre-filtering; Fog agents execute diversified ensemble detection with dynamic consensus voting; and Cloud agents continuously optimize system policies via Proximal Policy Optimization (PPO) while maintaining asynchronous, non-blocking inference. The framework incorporates LLM-based response generation for explainability and federated knowledge aggregation for adaptive policy distribution. This architecture enables resource-aware specialization without sacrificing real-time performance or model interpretability. Empirical evaluation on two industrial benchmarks (Boiler Emulator and Wind Turbine) demonstrates that SEMAS achieves superior anomaly detection performance with exceptional stability under adaptation, sustains prediction accuracy across evolving operational contexts, and delivers substantial latency improvements enabling genuine real-time deployment. Ablation studies confirm that PPO-driven policy evolution, consensus voting, and federated aggregation each contribute materially to system effectiveness. These findings indicate that resource-aware, self-evolving 1multi-agent coordination is essential for production-ready industrial IoT predictive maintenance under strict latency and explainability constraints.

**arXiv ID:** 2602.16738
</details>

<details>
<summary><strong>Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form</strong> - Xuefeng Wang, Lei Zhang, Henglin Pu, Husheng Li, Ahmed H. Qureshi - [[pdf]](https://arxiv.org/pdf/2602.17078)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) has made significant progress in recent years, but most algorithms still rely on a discrete-time Markov Decision Process (MDP) with fixed decision intervals. This formulation is often ill-suited for complex multi-agent dynamics, particularly in high-frequency or irregular time-interval settings, leading to degraded performance and motivating the development of continuous-time MARL (CT-MARL). Existing CT-MARL methods are mainly built on Hamilton-Jacobi-Bellman (HJB) equations. However, they rarely account for safety constraints such as collision penalties, since these introduce discontinuities that make HJB-based learning difficult. To address this challenge, we propose a continuous-time constrained MDP (CT-CMDP) formulation and a novel MARL framework that transforms discrete MDPs into CT-CMDPs via an epigraph-based reformulation. We then solve this by proposing a novel physics-informed neural network (PINN)-based actor-critic method that enables stable and efficient optimization in continuous time. We evaluate our approach on continuous-time safe multi-particle environments (MPE) and safe multi-agent MuJoCo benchmarks. Results demonstrate smoother value approximations, more stable training, and improved performance over safe MARL baselines, validating the effectiveness and robustness of our method.

**arXiv ID:** 2602.17078
</details>

<details>
<summary><strong>AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation</strong> - Siyu Wang, Ruotian Lu, Zhihao Yang, Yuchao Wang, Yanzhou Zhang, Lei Xu, Qimin Xu, Guojun Yin, Cailian Chen, Xinping Guan - [[pdf]](https://arxiv.org/pdf/2602.17100)</summary>

**Abstract:** Large language model(LLM)-driven multi-agent systems(MAS) coordinate specialized agents through predefined interaction topologies and have shown promise for complex tasks such as competition-level code generation. Recent studies demonstrate that carefully designed multi-agent workflows and communication graphs can significantly improve code generation performance by leveraging collaborative reasoning. However, existing methods neither adapt topology density to task difficulty nor iteratively refine the topology within an instance using execution feedback, which leads to redundant communication and performance bottlenecks. To address these issues, we propose AgentConductor: a reinforcement learning-optimized MAS with an LLM-based orchestrator agent as its core, which enables end-to-end feedback-driven dynamic generation of interaction topologies. For each query, AgentConductor infers agent roles and task difficulty, then constructs a task-adapted, density-aware layered directed acyclic graph (DAG) topology, underpinned by two key innovations. First, we design a novel topological density function that captures communication-aware mathematical characterizations of multi-agent interactions. Second, we adopt difficulty interval partitioning to avoid excessive pruning for precise topological density upper bound measurement per difficulty level and finer-grained control. Empirically, across three competition-level and two foundational code datasets, AgentConductor achieves state-of-the-art accuracy, outperforming the strongest baseline by up to 14.6% in pass@1 accuracy, 13% in density reduction, and 68% in token cost reduction.

**arXiv ID:** 2602.17100
</details>

<details>
<summary><strong>Multi-Agent Temporal Logic Planning via Penalty Functions and Block-Coordinate Optimization</strong> - Eleftherios E. Vlahakis, Arash Bahari Kordabad, Lars Lindemann, Pantelis Sopasakis, Sadegh Soudjani, Dimos V. Dimarogonas - [[pdf]](https://arxiv.org/pdf/2602.17434)</summary>

**Abstract:** Multi-agent planning under Signal Temporal Logic (STL) is often hindered by collaborative tasks that lead to computational challenges due to the inherent high-dimensionality of the problem, preventing scalable synthesis with satisfaction guarantees. To address this, we formulate STL planning as an optimization program under arbitrary multi-agent constraints and introduce a penalty-based unconstrained relaxation that can be efficiently solved via a Block-Coordinate Gradient Descent (BCGD) method, where each block corresponds to a single agent's decision variables, thereby mitigating complexity. By utilizing a quadratic penalty function defined via smooth STL semantics, we show that BCGD iterations converge to a stationary point of the penalized problem under standard regularity assumptions. To enforce feasibility, the BCGD solver is embedded within a two-layer optimization scheme: inner BCGD updates are performed for a fixed penalty parameter, which is then increased in an outer loop to progressively improve multi-agent STL robustness. The proposed framework enables scalable computations and is validated through various complex multi-robot planning scenarios.

**arXiv ID:** 2602.17434
</details>

<details>
<summary><strong>Fault Tolerant Multi-Agent Learning with Adversarial Budget Constraints</strong> - David Mguni, Yaqi Sun, Haojun Chen, Wanrong Yang, Amir Darabi, Larry Olanrewaju Orimoloye, Yaodong Yang - [[pdf]](https://arxiv.org/pdf/2508.08800)</summary>

**Abstract:** We study robustness to agent malfunctions in cooperative multi-agent reinforcement learning (MARL), a failure mode that is critical in practice yet underexplored in existing theory. We introduce MARTA, a plug-and-play robustness layer that augments standard MARL algorithms with a Switcher-Adversary mechanism which selectively induces malfunctions in performance-critical states. This formulation defines a fault-switching $(N+2)$-player Markov game in which the Switcher chooses when and which agent fails, and the Adversary controls the resulting faulty behaviour via random or worst-case policies. We develop a Q-learning-type scheme and show that the associated Bellman operator is a contraction, yielding existence and uniqueness of the minimax value, convergence to a Markov perfect equilibrium. MARTA integrates seamlessly with MARL algorithms without architectural modification and consistently improves robustness across Traffic Junction (TJ), Level-Based Foraging (LBF), MPE SimpleTag, and SMAC (v2). In these domains, MARTA achieves large gains in final performance of up to 116.7\% in SMAC, 21.4\% in MPE SimpleTag, and 44.6\% in LBF, while significantly reducing failure rates under train-test mismatched fault regimes. These results establish MARTA as a theoretically grounded and practically deployable mechanism for fault-tolerant MARL.

**arXiv ID:** 2508.08800
</details>

<details>
<summary><strong>Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning</strong> - Xuefeng Wang, Lei Zhang, Henglin Pu, Ahmed H. Qureshi, Husheng Li - [[pdf]](https://arxiv.org/pdf/2509.09135)</summary>

**Abstract:** Existing reinforcement learning (RL) methods struggle with complex dynamical systems that demand interactions at high frequencies or irregular time intervals. Continuous-time RL (CTRL) has emerged as a promising alternative by replacing discrete-time Bellman recursion with differential value functions defined as viscosity solutions of the Hamilton--Jacobi--Bellman (HJB) equation. While CTRL has shown promise, its applications have been largely limited to the single-agent domain. This limitation stems from two key challenges: (i) conventional solution methods for HJB equations suffer from the curse of dimensionality (CoD), making them intractable in high-dimensional systems; and (ii) even with HJB-based learning approaches, accurately approximating centralized value functions in multi-agent settings remains difficult, which in turn destabilizes policy training. In this paper, we propose a CT-MARL framework that uses physics-informed neural networks (PINNs) to approximate HJB-based value functions at scale. To ensure the value is consistent with its differential structure, we align value learning with value-gradient learning by introducing a Value Gradient Iteration (VGI) module that iteratively refines value gradients along trajectories. This improves gradient fidelity, in turn yielding more accurate values and stronger policy learning. We evaluate our method using continuous-time variants of standard benchmarks, including multi-agent particle environment (MPE) and multi-agent MuJoCo. Our results demonstrate that our approach consistently outperforms existing continuous-time RL baselines and scales to complex multi-agent dynamics.

**arXiv ID:** 2509.09135
</details>

<details>
<summary><strong>Policy Compiler for Secure Agentic Systems</strong> - Nils Palumbo, Sarthak Choudhary, Jihye Choi, Prasad Chalasani, Somesh Jha - [[pdf]](https://arxiv.org/pdf/2602.16708)</summary>

**Abstract:** LLM-based agents are increasingly being deployed in contexts requiring complex authorization policies: customer service protocols, approval workflows, data access restrictions, and regulatory compliance. Embedding these policies in prompts provides no enforcement guarantees. We present PCAS, a Policy Compiler for Agentic Systems that provides deterministic policy enforcement.
Enforcing such policies requires tracking information flow across agents, which linear message histories cannot capture. Instead, PCAS models the agentic system state as a dependency graph capturing causal relationships among events such as tool calls, tool results, and messages. Policies are expressed in a Datalog-derived language, as declarative rules that account for transitive information flow and cross-agent provenance. A reference monitor intercepts all actions and blocks violations before execution, providing deterministic enforcement independent of model reasoning.
PCAS takes an existing agent implementation and a policy specification, and compiles them into an instrumented system that is policy-compliant by construction, with no security-specific restructuring required. We evaluate PCAS on three case studies: information flow policies for prompt injection defense, approval workflows in a multi-agent pharmacovigilance system, and organizational policies for customer service. On customer service tasks, PCAS improves policy compliance from 48% to 93% across frontier models, with zero policy violations in instrumented runs.

**arXiv ID:** 2602.16708
</details>

<details>
<summary><strong>Modeling Distinct Human Interaction in Web Agents</strong> - Faria Huq, Zora Zhiruo Wang, Zhanqiu Guo, Venu Arvind Arangarajan, Tianyue Ou, Frank Xu, Shuyan Zhou, Graham Neubig, Jeffrey P. Bigham - [[pdf]](https://arxiv.org/pdf/2602.17588)</summary>

**Abstract:** Despite rapid progress in autonomous web agents, human involvement remains essential for shaping preferences and correcting agent behavior as tasks unfold. However, current agentic systems lack a principled understanding of when and why humans intervene, often proceeding autonomously past critical decision points or requesting unnecessary confirmation. In this work, we introduce the task of modeling human intervention to support collaborative web task execution. We collect CowCorpus, a dataset of 400 real-user web navigation trajectories containing over 4,200 interleaved human and agent actions. We identify four distinct patterns of user interaction with agents -- hands-off supervision, hands-on oversight, collaborative task-solving, and full user takeover. Leveraging these insights, we train language models (LMs) to anticipate when users are likely to intervene based on their interaction styles, yielding a 61.4-63.4% improvement in intervention prediction accuracy over base LMs. Finally, we deploy these intervention-aware models in live web navigation agents and evaluate them in a user study, finding a 26.5% increase in user-rated agent usefulness. Together, our results show structured modeling of human intervention leads to more adaptive, collaborative agents.

**arXiv ID:** 2602.17588
</details>

<details>
<summary><strong>Multimodal Multi-Agent Empowered Legal Judgment Prediction</strong> - Zhaolu Kang, Junhao Gong, Qingxi Chen, Hao Zhang, Jiaxin Liu, Rong Fu, Zhiyuan Feng, Yuan Wang, Simon Fong, Kaiyue Zhou - [[pdf]](https://arxiv.org/pdf/2601.12815)</summary>

**Abstract:** Legal Judgment Prediction (LJP) aims to predict the outcomes of legal cases based on factual descriptions, serving as a fundamental task to advance the development of legal systems. Traditional methods often rely on statistical analyses or role-based simulations but face challenges with multiple allegations, diverse evidence, and lack adaptability. In this paper, we introduce JurisMMA, a novel framework for LJP that effectively decomposes trial tasks, standardizes processes, and organizes them into distinct stages. Furthermore, we build JurisMM, a large dataset with over 100,000 recent Chinese judicial records, including both text and multimodal video-text data, enabling comprehensive evaluation. Experiments on JurisMM and the benchmark LawBench validate our framework's effectiveness. These results indicate that our framework is effective not only for LJP but also for a broader range of legal applications, offering new perspectives for the development of future legal methods and datasets.

**arXiv ID:** 2601.12815
</details>

<details>
<summary><strong>Multi-Agent Lipschitz Bandits</strong> - Sourav Chakraborty, Amit Kiran Rege, Claire Monteleoni, Lijun Chen - [[pdf]](https://arxiv.org/pdf/2602.16965)</summary>

**Abstract:** We study the decentralized multi-player stochastic bandit problem over a continuous, Lipschitz-structured action space where hard collisions yield zero reward. Our objective is to design a communication-free policy that maximizes collective reward, with coordination costs that are independent of the time horizon $T$. We propose a modular protocol that first solves the multi-agent coordination problem -- identifying and seating players on distinct high-value regions via a novel maxima-directed search -- and then decouples the problem into $N$ independent single-player Lipschitz bandits. We establish a near-optimal regret bound of $\tilde{O}(T^{(d+1)/(d+2)})$ plus a $T$-independent coordination cost, matching the single-player rate. To our knowledge, this is the first framework providing such guarantees, and it extends to general distance-threshold collision models.

**arXiv ID:** 2602.16965
</details>

<details>
<summary><strong>Action-Graph Policies: Learning Action Co-dependencies in Multi-Agent Reinforcement Learning</strong> - Nikunj Gupta, James Zachary Hare, Jesse Milzman, Rajgopal Kannan, Viktor Prasanna - [[pdf]](https://arxiv.org/pdf/2602.17009)</summary>

**Abstract:** Coordinating actions is the most fundamental form of cooperation in multi-agent reinforcement learning (MARL). Successful decentralized decision-making often depends not only on good individual actions, but on selecting compatible actions across agents to synchronize behavior, avoid conflicts, and satisfy global constraints. In this paper, we propose Action Graph Policies (AGP), that model dependencies among agents' available action choices. It constructs, what we call, \textit{coordination contexts}, that enable agents to condition their decisions on global action dependencies. Theoretically, we show that AGPs induce a strictly more expressive joint policy compared to fully independent policies and can realize coordinated joint actions that are provably more optimal than greedy execution even from centralized value-decomposition methods. Empirically, we show that AGP achieves 80-95\% success on canonical coordination tasks with partial observability and anti-coordination penalties, where other MARL methods reach only 10-25\%. We further demonstrate that AGP consistently outperforms these baselines in diverse multi-agent environments.

**arXiv ID:** 2602.17009
</details>

<details>
<summary><strong>StoryLensEdu: Personalized Learning Report Generation through Narrative-Driven Multi-Agent Systems</strong> - Leixian Shen, Yan Luo, Rui Sheng, Yujia He, Haotian Li, Leni Yang, Huamin Qu - [[pdf]](https://arxiv.org/pdf/2602.17067)</summary>

**Abstract:** Personalized feedback plays an important role in self-regulated learning (SRL), helping students track progress and refine their strategies. However, current common solutions, such as text-based reports or learning analytics dashboards, often suffer from poor interpretability, monotonous presentation, and limited explainability. To overcome these challenges, we present StoryLensEdu, a narrative-driven multi-agent system that automatically generates intuitive, engaging, and interactive learning reports. StoryLensEdu integrates three agents: a Data Analyst that extracts data insights based on a learning objective centered structure, a Teacher that ensures educational relevance and offers actionable suggestions, and a Storyteller that organizes these insights using the Heroes Journey narrative framework. StoryLensEdu supports post-generation interactive question answering to improve explainability and user engagement. We conducted a formative study in a real high school and iteratively developed StoryLensEdu in collaboration with an e-learning team to inform our design. Evaluation with real users shows that StoryLensEdu enhances engagement and promotes a deeper understanding of the learning process.

**arXiv ID:** 2602.17067
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>From Labor to Collaboration: A Methodological Experiment Using AI Agents to Augment Research Perspectives in Taiwan's Humanities and Social Sciences</strong> - Yi-Chih Huang - [[pdf]](https://arxiv.org/pdf/2602.17221)</summary>

**Abstract:** Generative AI is reshaping knowledge work, yet existing research focuses predominantly on software engineering and the natural sciences, with limited methodological exploration for the humanities and social sciences. Positioned as a "methodological experiment," this study proposes an AI Agent-based collaborative research workflow (Agentic Workflow) for humanities and social science research. Taiwan's this http URL usage data (N = 7,729 conversations, November 2025) from the Anthropic Economic Index (AEI) serves as the empirical vehicle for validating the feasibility of this methodology.
This study operates on two levels: the primary level is the design and validation of a methodological framework - a seven-stage modular workflow grounded in three principles: task modularization, human-AI division of labor, and verifiability, with each stage delineating clear roles for human researchers (research judgment and ethical decisions) and AI Agents (information retrieval and text generation); the secondary level is the empirical analysis of AEI Taiwan data - serving as an operational demonstration of the workflow's application to secondary data research, showcasing both the process and output quality (see Appendix A).
This study contributes by proposing a replicable AI collaboration framework for humanities and social science researchers, and identifying three operational modes of human-AI collaboration - direct execution, iterative refinement, and human-led - through reflexive documentation of the operational process. This taxonomy reveals the irreplaceability of human judgment in research question formulation, theoretical interpretation, contextualized reasoning, and ethical reflection. Limitations including single-platform data, cross-sectional design, and AI reliability risks are acknowledged.

**arXiv ID:** 2602.17221
</details>

<details>
<summary><strong>Wink: Recovering from Misbehaviors in Coding Agents</strong> - Rahul Nanda, Chandra Maddila, Smriti Jha, Euna Mehnaz Khan, Matteo Paltenghi, Satish Chandra - [[pdf]](https://arxiv.org/pdf/2602.17037)</summary>

**Abstract:** Autonomous coding agents, powered by large language models (LLMs), are increasingly being adopted in the software industry to automate complex engineering tasks. However, these agents are prone to a wide range of misbehaviors, such as deviating from the user's instructions, getting stuck in repetitive loops, or failing to use tools correctly. These failures disrupt the development workflow and often require resource-intensive manual intervention. In this paper, we present a system for automatically recovering from agentic misbehaviors at scale. We first introduce a taxonomy of misbehaviors grounded in an analysis of production traffic, identifying three primary categories: Specification Drift, Reasoning Problems, and Tool Call Failures, which we find occur in about 30% of all agent trajectories.
To address these issues, we developed a lightweight, asynchronous self-intervention system named Wink. Wink observes agent trajectories and provides targeted course-correction guidance to nudge the agent back to a productive path. We evaluated our system on over 10,000 real world agent trajectories and found that it successfully resolves 90% of the misbehaviors that require a single intervention. Furthermore, a live A/B test in our production environment demonstrated that our system leads to a statistically significant reduction in Tool Call Failures, Tokens per Session and Engineer Interventions per Session. We present our experience designing and deploying this system, offering insights into the challenges of building resilient agentic systems at scale.

**arXiv ID:** 2602.17037
</details>

<details>
<summary><strong>What Breaks Embodied AI Security:LLM Vulnerabilities, CPS Flaws,or Something Else?</strong> - Boyang Ma, Hechuan Guo, Peizhuo Lv, Minghui Xu, Xuelong Dai, YeChao Zhang, Yijun Yang, Yue Zhang - [[pdf]](https://arxiv.org/pdf/2602.17345)</summary>

**Abstract:** Embodied AI systems (e.g., autonomous vehicles, service robots, and LLM-driven interactive agents) are rapidly transitioning from controlled environments to safety critical real-world deployments. Unlike disembodied AI, failures in embodied intelligence lead to irreversible physical consequences, raising fundamental questions about security, safety, and reliability. While existing research predominantly analyzes embodied AI through the lenses of Large Language Model (LLM) vulnerabilities or classical Cyber-Physical System (CPS) failures, this survey argues that these perspectives are individually insufficient to explain many observed breakdowns in modern embodied systems. We posit that a significant class of failures arises from embodiment-induced system-level mismatches, rather than from isolated model flaws or traditional CPS attacks. Specifically, we identify four core insights that explain why embodied AI is fundamentally harder to secure: (i) semantic correctness does not imply physical safety, as language-level reasoning abstracts away geometry, dynamics, and contact constraints; (ii) identical actions can lead to drastically different outcomes across physical states due to nonlinear dynamics and state uncertainty; (iii) small errors propagate and amplify across tightly coupled perception-decision-action loops; and (iv) safety is not compositional across time or system layers, enabling locally safe decisions to accumulate into globally unsafe behavior. These insights suggest that securing embodied AI requires moving beyond component-level defenses toward system-level reasoning about physical risk, uncertainty, and failure propagation.

**arXiv ID:** 2602.17345
</details>

<details>
<summary><strong>GAI: Generative Agents for Innovation</strong> - Masahiro Sato - [[pdf]](https://arxiv.org/pdf/2412.18899)</summary>

**Abstract:** This study examines whether collective reasoning among generative agents can facilitate novel and coherent thinking that leads to innovation. To achieve this, it proposes GAI, a new LLM-empowered framework designed for reflection and interaction among multiple generative agents to replicate the process of innovation. The core of the GAI framework lies in an architecture that dynamically processes the internal states of agents and a dialogue scheme specifically tailored to facilitate analogy-driven innovation. The framework's functionality is evaluated using Dyson's invention of the bladeless fan as a case study, assessing the extent to which the core ideas of the innovation can be replicated through a set of fictional technical documents. The experimental results demonstrate that models with internal states significantly outperformed those without, achieving higher average scores and lower variance. Notably, the model with five heterogeneous agents equipped with internal states successfully replicated the key ideas underlying the Dyson's invention. This indicates that the internal state enables agents to refine their ideas, resulting in the construction and sharing of more coherent and comprehensive concepts.

**arXiv ID:** 2412.18899
</details>

<details>
<summary><strong>Online Learning with Improving Agents: Multiclass, Budgeted Agents and Bandit Learners</strong> - Sajad Ashkezari, Shai Ben-David - [[pdf]](https://arxiv.org/pdf/2602.17103)</summary>

**Abstract:** We investigate the recently introduced model of learning with improvements, where agents are allowed to make small changes to their feature values to be warranted a more desirable label. We extensively extend previously published results by providing combinatorial dimensions that characterize online learnability in this model, by analyzing the multiclass setup, learnability in a bandit feedback setup, modeling agents' cost for making improvements and more.

**arXiv ID:** 2602.17103
</details>

<details>
<summary><strong>ICP-Based Pallet Tracking for Unloading on Inclined Surfaces by Autonomous Forklifts</strong> - Takuro Kato, Mitsuharu Morisawa - [[pdf]](https://arxiv.org/pdf/2602.16744)</summary>

**Abstract:** This paper proposes a control method for autonomous forklifts to unload pallets on inclined surfaces, enabling the fork to be withdrawn without dragging the pallets. The proposed method applies the Iterative Closest Point (ICP) algorithm to point clouds measured from the upper region of the pallet and thereby tracks the relative position and attitude angle difference between the pallet and the fork during the unloading operation in real-time. According to the tracking result, the fork is aligned parallel to the target surface. After the fork is aligned, it is possible to complete the unloading process by withdrawing the fork along the tilt, preventing any dragging of the pallet. The effectiveness of the proposed method is verified through dynamic simulations and experiments using a real forklift that replicate unloading operations onto the inclined bed of a truck.

**arXiv ID:** 2602.16744
</details>

<details>
<summary><strong>Decentralized and Fully Onboard: Range-Aided Cooperative Localization and Navigation on Micro Aerial Vehicles</strong> - Abhishek Goudar, Angela P. Schoellig - [[pdf]](https://arxiv.org/pdf/2602.16594)</summary>

**Abstract:** Controlling a team of robots in a coordinated manner is challenging because centralized approaches (where all computation is performed on a central machine) scale poorly, and globally referenced external localization systems may not always be available. In this work, we consider the problem of range-aided decentralized localization and formation control. In such a setting, each robot estimates its relative pose by combining data only from onboard odometry sensors and distance measurements to other robots in the team. Additionally, each robot calculates the control inputs necessary to collaboratively navigate an environment to accomplish a specific task, for example, moving in a desired formation while monitoring an area. We present a block coordinate descent approach to localization that does not require strict coordination between the robots. We present a novel formulation for formation control as inference on factor graphs that takes into account the state estimation uncertainty and can be solved efficiently. Our approach to range-aided localization and formation-based navigation is completely decentralized, does not require specialized trajectories to maintain formation, and achieves decimeter-level positioning and formation control accuracy. We demonstrate our approach through multiple real experiments involving formation flights in diverse indoor and outdoor environments.

**arXiv ID:** 2602.16594
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (18 papers)</h2></summary>

<details>
<summary><strong>OpenSage: Self-programming Agent Generation Engine</strong> - Hongwei Li, Zhun Wang, Qinrun Dai, Yuzhou Nie, Jinjun Peng, Ruitong Liu, Jingyang Zhang, Kaijie Zhu, Jingxuan He, Lun Wang, Yangruibo Ding, Yueqi Chen, Wenbo Guo, Dawn Song - [[pdf]](https://arxiv.org/pdf/2602.16891)</summary>

**Abstract:** Agent development kits (ADKs) provide effective platforms and tooling for constructing agents, and their designs are critical to the constructed agents' performance, especially the functionality for agent topology, tools, and memory. However, current ADKs either lack sufficient functional support or rely on humans to manually design these components, limiting agents' generalizability and overall performance. We propose OpenSage, the first ADK that enables LLMs to automatically create agents with self-generated topology and toolsets while providing comprehensive and structured memory support. OpenSage offers effective functionality for agents to create and manage their own sub-agents and toolkits. It also features a hierarchical, graph-based memory system for efficient management and a specialized toolkit tailored to software engineering tasks. Extensive experiments across three state-of-the-art benchmarks with various backbone models demonstrate the advantages of OpenSage over existing ADKs. We also conduct rigorous ablation studies to demonstrate the effectiveness of our design for each component. We believe OpenSage can pave the way for the next generation of agent development, shifting the focus from human-centered to AI-centered paradigms.

**arXiv ID:** 2602.16891
</details>

<details>
<summary><strong>Agentic Wireless Communication for 6G: Intent-Aware and Continuously Evolving Physical-Layer Intelligence</strong> - Zhaoyang Li, Xingzhi Jin, Junyu Pan, Qianqian Yang, Zhiguo Shi - [[pdf]](https://arxiv.org/pdf/2602.17096)</summary>

**Abstract:** As 6G wireless systems evolve, growing functional complexity and diverse service demands are driving a shift from rule-based control to intent-driven autonomous intelligence. User requirements are no longer captured by a single metric (e.g., throughput or reliability), but by multi-dimensional objectives such as latency sensitivity, energy preference, computational constraints, and service-level requirements. These objectives may also change over time due to environmental dynamics and user-network interactions. Therefore, accurate understanding of both the communication environment and user intent is critical for autonomous and sustainably evolving 6G communications.
Large language models (LLMs), with strong contextual understanding and cross-modal reasoning, provide a promising foundation for intent-aware network agents. Compared with rule-driven or centrally optimized designs, LLM-based agents can integrate heterogeneous information and translate natural-language intents into executable control and configuration decisions.
Focusing on a closed-loop pipeline of intent perception, autonomous decision making, and network execution, this paper investigates agentic AI for the 6G physical layer and its realization pathways. We review representative physical-layer tasks and their limitations in supporting intent awareness and autonomy, identify application scenarios where agentic AI is advantageous, and discuss key challenges and enabling technologies in multimodal perception, cross-layer decision making, and sustainable optimization. Finally, we present a case study of an intent-driven link decision agent, termed AgenCom, which adaptively constructs communication links under diverse user preferences and channel conditions.

**arXiv ID:** 2602.17096
</details>

<details>
<summary><strong>Web Verbs: Typed Abstractions for Reliable Task Composition on the Agentic Web</strong> - Linxi Jiang, Rui Xi, Zhijie Liu, Shuo Chen, Zhiqiang Lin, Suman Nath - [[pdf]](https://arxiv.org/pdf/2602.17245)</summary>

**Abstract:** The Web is evolving from a medium that humans browse to an environment where software agents act on behalf of users. Advances in large language models (LLMs) make natural language a practical interface for goal-directed tasks, yet most current web agents operate on low-level primitives such as clicks and keystrokes. These operations are brittle, inefficient, and difficult to verify. Complementing content-oriented efforts such as NLWeb's semantic layer for retrieval, we argue that the agentic web also requires a semantic layer for web actions. We propose \textbf{Web Verbs}, a web-scale set of typed, semantically documented functions that expose site capabilities through a uniform interface, whether implemented through APIs or robust client-side workflows. These verbs serve as stable and composable units that agents can discover, select, and synthesize into concise programs. This abstraction unifies API-based and browser-based paradigms, enabling LLMs to synthesize reliable and auditable workflows with explicit control and data flow. Verbs can carry preconditions, postconditions, policy tags, and logging support, which improves \textbf{reliability} by providing stable interfaces, \textbf{efficiency} by reducing dozens of steps into a few function calls, and \textbf{verifiability} through typed contracts and checkable traces. We present our vision, a proof-of-concept implementation, and representative case studies that demonstrate concise and robust execution compared to existing agents. Finally, we outline a roadmap for standardization to make verbs deployable and trustworthy at web scale.

**arXiv ID:** 2602.17245
</details>

<details>
<summary><strong>MedClarify: An information-seeking AI agent for medical diagnosis with case-specific follow-up questions</strong> - Hui Min Wong, Philip Heesen, Pascal Janetzky, Martin Bendszus, Stefan Feuerriegel - [[pdf]](https://arxiv.org/pdf/2602.17308)</summary>

**Abstract:** Large language models (LLMs) are increasingly used for diagnostic tasks in medicine. In clinical practice, the correct diagnosis can rarely be immediately inferred from the initial patient presentation alone. Rather, reaching a diagnosis often involves systematic history taking, during which clinicians reason over multiple potential conditions through iterative questioning to resolve uncertainty. This process requires considering differential diagnoses and actively excluding emergencies that demand immediate intervention. Yet, the ability of medical LLMs to generate informative follow-up questions and thus reason over differential diagnoses remains underexplored. Here, we introduce MedClarify, an AI agent for information-seeking that can generate follow-up questions for iterative reasoning to support diagnostic decision-making. Specifically, MedClarify computes a list of candidate diagnoses analogous to a differential diagnosis, and then proactively generates follow-up questions aimed at reducing diagnostic uncertainty. By selecting the question with the highest expected information gain, MedClarify enables targeted, uncertainty-aware reasoning to improve diagnostic performance. In our experiments, we first demonstrate the limitations of current LLMs in medical reasoning, which often yield multiple, similarly likely diagnoses, especially when patient cases are incomplete or relevant information for diagnosis is missing. We then show that our information-theoretic reasoning approach can generate effective follow-up questioning and thereby reduces diagnostic errors by ~27 percentage points (p.p.) compared to a standard single-shot LLM baseline. Altogether, MedClarify offers a path to improve medical LLMs through agentic information-seeking and to thus promote effective dialogues with medical LLMs that reflect the iterative and uncertain nature of real-world clinical reasoning.

**arXiv ID:** 2602.17308
</details>

<details>
<summary><strong>Overseeing Agents Without Constant Oversight: Challenges and Opportunities</strong> - Madeleine Grunde-McLaughlin, Hussein Mozannar, Maya Murad, Jingya Chen, Saleema Amershi, Adam Fourney - [[pdf]](https://arxiv.org/pdf/2602.16844)</summary>

**Abstract:** To enable human oversight, agentic AI systems often provide a trace of reasoning and action steps. Designing traces to have an informative, but not overwhelming, level of detail remains a critical challenge. In three user studies on a Computer User Agent, we investigate the utility of basic action traces for verification, explore three alternatives via design probes, and test a novel interface's impact on error finding in question-answering tasks. As expected, we find that current practices are cumbersome, limiting their efficacy. Conversely, our proposed design reduced the time participants spent finding errors. However, although participants reported higher levels of confidence in their decisions, their final accuracy was not meaningfully improved. To this end, our study surfaces challenges for human verification of agentic systems, including managing built-in assumptions, users' subjective and changing correctness criteria, and the shortcomings, yet importance, of communicating the agent's process.

**arXiv ID:** 2602.16844
</details>

<details>
<summary><strong>Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization</strong> - Srijan Sood, Kassiani Papasotiriou, Marius Vaiciulis, Tucker Balch - [[pdf]](https://arxiv.org/pdf/2602.17098)</summary>

**Abstract:** Portfolio Management is the process of overseeing a group of investments, referred to as a portfolio, with the objective of achieving predetermined investment goals. Portfolio optimization is a key component that involves allocating the portfolio assets so as to maximize returns while minimizing risk taken. It is typically carried out by financial professionals who use a combination of quantitative techniques and investment expertise to make decisions about the portfolio allocation.
Recent applications of Deep Reinforcement Learning (DRL) have shown promising results when used to optimize portfolio allocation by training model-free agents on historical market data. Many of these methods compare their results against basic benchmarks or other state-of-the-art DRL agents but often fail to compare their performance against traditional methods used by financial professionals in practical settings. One of the most commonly used methods for this task is Mean-Variance Portfolio Optimization (MVO), which uses historical time series information to estimate expected asset returns and covariances, which are then used to optimize for an investment objective.
Our work is a thorough comparison between model-free DRL and MVO for optimal portfolio allocation. We detail the specifics of how to make DRL for portfolio optimization work in practice, also noting the adjustments needed for MVO. Backtest results demonstrate strong performance of the DRL agent across many metrics, including Sharpe ratio, maximum drawdowns, and absolute returns.

**arXiv ID:** 2602.17098
</details>

<details>
<summary><strong>CaveAgent: Transforming LLMs into Stateful Runtime Operators</strong> - Maohao Ran, Zhenglin Wan, Cooper Lin, Yanting Zhang, Hongyu Xin, Hongwei Fan, Yibo Xu, Beier Luo, Yaxin Zhou, Wangbo Zhao, Lijie Yang, Lang Feng, Fuchao Yang, Jingxuan Wu, Yiqiao Huang, Chendong Ma, Dailing Jiang, Jianbo Deng, Sirui Han, Yang You, Bo An, Yike Guo, Jun Song - [[pdf]](https://arxiv.org/pdf/2601.01569)</summary>

**Abstract:** LLM-based agents are increasingly capable of complex task execution, yet current agentic systems remain constrained by text-centric paradigms that struggle with long-horizon tasks due to fragile multi-turn dependencies and context drift. We present CaveAgent, a framework that shifts tool use from ``LLM-as-Text-Generator'' to ``LLM-as-Runtime-Operator.'' CaveAgent introduces a dual-stream architecture that inverts the conventional paradigm: rather than treating the LLM's text context as the primary workspace with tools as auxiliary, CaveAgent elevates the persistent Python runtime as the central locus of state, with a lightweight semantic stream serving as its orchestrator. Beyond leveraging code generation to resolve interdependent sub-tasks (e.g., loops, conditionals) in a single step, CaveAgent introduces \textit{Stateful Runtime Management}: it injects, manipulates, and retrieves complex Python objects (e.g., DataFrames, database connections) that persist across turns, unlike existing code-based approaches that remain text-bound. CaveAgent further provides a runtime-integrated skill management system that extends the Agent Skills open standard, enabling ecosystem interoperability through executable skill injections. This persistence mechanism serves as a high-fidelity external memory that reduces context drift in multi-turn interactions and preserves processed data for downstream applications without information loss. Evaluations show consistent improvement across challenging benchmarks, enabling CaveAgent to handle data scales that cause context overflow in both JSON-based and code-based agents. The accessible runtime state further provides programmatically verifiable feedback, enabling automated evaluation and reward signal generation without human annotation and establishing a structural foundation for future research in Reinforcement Learning with Verifiable Rewards (RLVR).

**arXiv ID:** 2601.01569
</details>

<details>
<summary><strong>EnterpriseBench Corecraft: Training Generalizable Agents on High-Fidelity RL Environments</strong> - Sushant Mehta, Logan Ritchie, Suhaas Garre, Nick Heiner, Edwin Chen - [[pdf]](https://arxiv.org/pdf/2602.16179)</summary>

**Abstract:** We show that training AI agents on high-fidelity reinforcement learning environments produces capabilities that generalize beyond the training distribution. We introduce CoreCraft, the first environment in EnterpriseBench, Surge AI's suite of agentic RL environments. CoreCraft is a fully operational enterprise simulation of a customer support organization, comprising over 2,500 entities across 14 entity types with 23 unique tools, designed to measure whether AI agents can perform the multi-step, domain-specific work that real jobs demand. Frontier models such as GPT-5.2 and Claude Opus 4.6 solve fewer than 30% of tasks when all expert-authored rubric criteria must be satisfied. Using this environment, we train GLM 4.6 with Group Relative Policy Optimization (GRPO) and adaptive clipping. After a single epoch of training, the model improves from 25.37% to 36.76% task pass rate on held-out evaluation tasks. More importantly, these gains transfer to out-of-distribution benchmarks: +4.5% on BFCL Parallel, +7.4% on Tau2-Bench Retail, and +6.8% on Tool Decathlon (Pass@1). We believe three environment properties are consistent with the observed transfer: task-centric world building that optimizes for diverse, challenging tasks; expert-authored rubrics enabling reliable reward computation; and enterprise workflows that reflect realistic professional patterns. Our results suggest that environment quality, diversity, and realism are key factors enabling generalizable agent capabilities.

**arXiv ID:** 2602.16179
</details>

<details>
<summary><strong>Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning</strong> - Sanghyeon Lee, Sangjun Bae, Yisak Park, Seungyul Han - [[pdf]](https://arxiv.org/pdf/2502.03752)</summary>

**Abstract:** Meta-reinforcement learning (Meta-RL) facilitates rapid adaptation to unseen tasks but faces challenges in long-horizon environments. Skill-based approaches tackle this by decomposing state-action sequences into reusable skills and employing hierarchical decision-making. However, these methods are highly susceptible to noisy offline demonstrations, leading to unstable skill learning and degraded performance. To address this, we propose Self-Improving Skill Learning (SISL), which performs self-guided skill refinement using decoupled high-level and skill improvement policies, while applying skill prioritization via maximum return relabeling to focus updates on task-relevant trajectories, resulting in robust and stable adaptation even under noisy and suboptimal data. By mitigating the effect of noise, SISL achieves reliable skill learning and consistently outperforms other skill-based meta-RL methods on diverse long-horizon tasks. Our code is available at this https URL.

**arXiv ID:** 2502.03752
</details>

<details>
<summary><strong>Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning</strong> - Jaebak Hwang, Sanghyeon Lee, Jeongmo Kim, Seungyul Han - [[pdf]](https://arxiv.org/pdf/2506.21039)</summary>

**Abstract:** Long-horizon goal-conditioned tasks pose fundamental challenges for reinforcement learning (RL), particularly when goals are distant and rewards are sparse. While hierarchical and graph-based methods offer partial solutions, their reliance on conventional hindsight relabeling often fails to correct subgoal infeasibility, leading to inefficient high-level planning. To address this, we propose Strict Subgoal Execution (SSE), a graph-based hierarchical RL framework that integrates Frontier Experience Replay (FER) to separate unreachable from admissible subgoals and streamline high-level decision making. FER delineates the reachability frontier using failure and partial-success transitions, which identifies unreliable subgoals, increases subgoal reliability, and reduces unnecessary high-level decisions. Additionally, SSE employs a decoupled exploration policy to cover underexplored regions of the goal space and a path refinement that adjusts edge costs using observed low-level failures. Experimental results across diverse long-horizon benchmarks show that SSE consistently outperforms existing goal-conditioned and hierarchical RL methods in both efficiency and success rate. Our code is available at this https URL

**arXiv ID:** 2506.21039
</details>

<details>
<summary><strong>Efficient Reinforcement Learning for Large Language Models with Intrinsic Exploration</strong> - Yan Sun, Jia Guo, Stanley Kok, Zihao Wang, Zujie Wen, Zhiqiang Zhang - [[pdf]](https://arxiv.org/pdf/2511.00794)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has improved the reasoning ability of large language models, yet training remains costly because many rollouts contribute little to optimization, considering the amount of computation required. This study investigates how simply leveraging intrinsic data properties, almost free benefit during training, can improve data efficiency for RLVR. We propose PREPO with two complementary components. First, we adopt prompt perplexity as an indicator of model adaptability in learning, enabling the model to progress from well-understood contexts to more challenging ones. Second, we amplify the discrepancy among the rollouts by differentiating their relative entropy, and prioritize sequences that exhibit a higher degree of exploration. Together, these mechanisms reduce rollout demand while preserving competitive performance. On the Qwen and Llama models, PREPO achieves effective results on mathematical reasoning benchmarks with up to 3 times fewer rollouts than the baselines. Beyond empirical gains, we provide theoretical and in-depth analyses explaining the underlying rationale of our method to improve the data efficiency of RLVR.

**arXiv ID:** 2511.00794
</details>

<details>
<summary><strong>Stigmergic Swarming Agents for Fast Subgraph Isomorphism</strong> - H. Van Dyke Parunak - [[pdf]](https://arxiv.org/pdf/2601.02449)</summary>

**Abstract:** Maximum partial subgraph isomorphism compares two graphs (nodes joined by edges) to find a largest common subgraph. A common use case, for graphs with labeled nodes, seeks to find instances of a \textit{query} graph with $q$ nodes in a (typically larger) \textit{data} graph with $d$ nodes. The problem is NP-complete, and naïve solutions are exponential in $q + d$. The fastest current heuristic has complexity $O(d^2)$. This paper outlines ASSIST (Approximate Swarming Subgraph Isomorphism through Stigmergy), inspired by the ant colony optimization approach to the traveling salesperson. After peering (identifying matching individual nodes in query and data) in time $O(q\cdot log(d))$, the time required for ASSIST's iterative subgraph search, the combinatorially complex part of the problem, is linear in query size and constant in data size. ASSIST can be extended to support matching problems (such as temporally ordered edges, inexact matches, and missing nodes or edges in the data graph) that frustrate other heuristics.

**arXiv ID:** 2601.02449
</details>

<details>
<summary><strong>PoLi-RL: A Point-to-List Reinforcement Learning Framework for Conditional Semantic Textual Similarity</strong> - Zixin Song, Bowen Zhang, Qian-Wen Zhang, Di Yin, Xing Sun, Chunping Li - [[pdf]](https://arxiv.org/pdf/2510.04080)</summary>

**Abstract:** Conditional Semantic Textual Similarity (C-STS) measures the semantic proximity between text segments under a specific condition, thereby overcoming the ambiguity inherent in traditional STS. However, existing methods are largely confined to discriminative models, failing to fully leverage recent breakthroughs in the NLP community involving Large Language Models (LLMs) and Reinforcement Learning (RL). RL is a particularly well-suited paradigm for this task, as it can directly optimize the non-differentiable Spearman ranking metric and guide the reasoning process required by C-STS. Nevertheless, we find that naively applying listwise RL fails to produce meaningful improvements, as the model struggles with complex, coarse-grained reward signals, leading to optimization difficulties. To address this challenge, we introduce PoLi-RL, a novel Point-to-List Reinforcement Learning framework. PoLi-RL employs a two-stage curriculum: it first trains the model with a simple pointwise reward to establish fundamental scoring capabilities, then transitions to a hybrid reward that combines pointwise, pairwise, and listwise objectives to refine the model's ability to discern subtle semantic distinctions. Crucially, we propose an innovative Parallel Slice Ranking Reward (PSRR) mechanism that computes ranking rewards in parallel slices, where each slice consists of completions with the same index from different samples. This provides a precise, differentiated learning signal for each individual completion, enabling granular credit assignment and effective optimization. On the official C-STS benchmark, PoLi-RL achieves a Spearman correlation coefficient of 48.18, establishing a new SOTA for the cross-encoder architecture. As the first work to successfully apply RL to C-STS, our study introduces a powerful paradigm for aligning LLMs for complex, ranking-based conditional judgment tasks.

**arXiv ID:** 2510.04080
</details>

<details>
<summary><strong>RLGT: A reinforcement learning framework for extremal graph theory</strong> - Ivan Damnjanović, Uroš Milivojević, Irena Đorđević, Dragan Stevanović - [[pdf]](https://arxiv.org/pdf/2602.17276)</summary>

**Abstract:** Reinforcement learning (RL) is a subfield of machine learning that focuses on developing models that can autonomously learn optimal decision-making strategies over time. In a recent pioneering paper, Wagner demonstrated how the Deep Cross-Entropy RL method can be applied to tackle various problems from extremal graph theory by reformulating them as combinatorial optimization problems. Subsequently, many researchers became interested in refining and extending the framework introduced by Wagner, thereby creating various RL environments specialized for graph theory. Moreover, a number of problems from extremal graph theory were solved through the use of RL. In particular, several inequalities concerning the Laplacian spectral radius of graphs were refuted, new lower bounds were obtained for certain Ramsey numbers, and contributions were made to the Turán-type extremal problem in which the forbidden structures are cycles of length three and four. Here, we present Reinforcement Learning for Graph Theory (RLGT), a novel RL framework that systematizes the previous work and provides support for both undirected and directed graphs, with or without loops, and with an arbitrary number of edge colors. The framework efficiently represents graphs and aims to facilitate future RL-based research in extremal graph theory through optimized computational performance and a clean and modular design.

**arXiv ID:** 2602.17276
</details>

<details>
<summary><strong>LexiSafe: Offline Safe Reinforcement Learning with Lexicographic Safety-Reward Hierarchy</strong> - Hsin-Jung Yang, Zhanhong Jiang, Prajwal Koirala, Qisai Liu, Cody Fleming, Soumik Sarkar - [[pdf]](https://arxiv.org/pdf/2602.17312)</summary>

**Abstract:** Offline safe reinforcement learning (RL) is increasingly important for cyber-physical systems (CPS), where safety violations during training are unacceptable and only pre-collected data are available. Existing offline safe RL methods typically balance reward-safety tradeoffs through constraint relaxation or joint optimization, but they often lack structural mechanisms to prevent safety drift. We propose LexiSafe, a lexicographic offline RL framework designed to preserve safety-aligned behavior. We first develop LexiSafe-SC, a single-cost formulation for standard offline safe RL, and derive safety-violation and performance-suboptimality bounds that together yield sample-complexity guarantees. We then extend the framework to hierarchical safety requirements with LexiSafe-MC, which supports multiple safety costs and admits its own sample-complexity analysis. Empirically, LexiSafe demonstrates reduced safety violations and improved task performance compared to constrained offline baselines. By unifying lexicographic prioritization with structural bias, LexiSafe offers a practical and theoretically grounded approach for safety-critical CPS decision-making.

**arXiv ID:** 2602.17312
</details>

<details>
<summary><strong>Online Robust Reinforcement Learning with General Function Approximation</strong> - Debamita Ghosh, George K. Atia, Yue Wang - [[pdf]](https://arxiv.org/pdf/2512.18957)</summary>

**Abstract:** In many real-world settings, reinforcement learning systems suffer performance degradation when the environment encountered at deployment differs from that observed during training. Distributionally robust reinforcement learning (DR-RL) mitigates this issue by seeking policies that maximize performance under the most adverse transition dynamics within a prescribed uncertainty set. Most existing DR-RL approaches, however, rely on strong data availability assumptions, such as access to a generative model or large offline datasets, and are largely restricted to tabular settings.
In this work, we propose a fully online DR-RL algorithm with general function approximation that learns robust policies solely through interaction, without requiring prior knowledge or pre-collected data. Our approach is based on a dual-driven fitted robust Bellman procedure that simultaneously estimates the value function and the corresponding worst-case backup operator. We establish regret guarantees for online DR-RL characterized by an intrinsic complexity notion, the robust Bellman-Eluder dimension, covering a broad class of phi-divergence uncertainty sets. The resulting regret bounds are sublinear, do not scale with the size of the state or action spaces, and specialize to tight rates in structured problem classes, demonstrating the practicality and scalability of our framework.

**arXiv ID:** 2512.18957
</details>

<details>
<summary><strong>Reinforcement Learning to Discover a North-East Monsoon Index for Rainfall Prediction in Thailand</strong> - Kiattikun Chobtham - [[pdf]](https://arxiv.org/pdf/2601.10181)</summary>

**Abstract:** Accurately predicting long-term rainfall is challenging. Global climate indices, such as the El Niño-Southern Oscillation, are standard input features for machine learning. However, a significant gap persists regarding local-scale indices capable of improving predictive accuracy in specific regions of Thailand. This paper introduces a novel North-East monsoon climate index calculated from sea surface temperature to reflect the climatology of the boreal winter monsoon. To optimise the calculated areas used for this index, a Deep Q-Network reinforcement learning agent explores and selects the most effective rectangles based on their correlation with seasonal rainfall. Rainfall stations were classified into 12 distinct clusters to distinguish rainfall patterns between southern and upper Thailand. Experimental results show that incorporating the optimised index into Long Short-Term Memory models significantly improves long-term monthly rainfall prediction skill in most cluster areas. This approach effectively reduces the Root Mean Square Error for 12-month-ahead forecasts.

**arXiv ID:** 2601.10181
</details>

<details>
<summary><strong>Robust Reinforcement Learning-Based Locomotion for Resource-Constrained Quadrupeds with Exteroceptive Sensing</strong> - Davide Plozza, Patricia Apostol, Paul Joseph, Simon Schläpfer, Michele Magno - [[pdf]](https://arxiv.org/pdf/2505.12537)</summary>

**Abstract:** Compact quadrupedal robots are proving increasingly suitable for deployment in real-world scenarios. Their smaller size fosters easy integration into human environments. Nevertheless, real-time locomotion on uneven terrains remains challenging, particularly due to the high computational demands of terrain perception. This paper presents a robust reinforcement learning-based exteroceptive locomotion controller for resource-constrained small-scale quadrupeds in challenging terrains, which exploits real-time elevation mapping, supported by a careful depth sensor selection. We concurrently train both a policy and a state estimator, which together provide an odometry source for elevation mapping, optionally fused with visual-inertial odometry (VIO). We demonstrate the importance of positioning an additional time-of-flight sensor for maintaining robustness even without VIO, thus having the potential to free up computational resources. We experimentally demonstrate that the proposed controller can flawlessly traverse steps up to 17.5 cm in height and achieve an 80% success rate on 22.5 cm steps, both with and without VIO. The proposed controller also achieves accurate forward and yaw velocity tracking of up to 1.0 m/s and 1.5 rad/s respectively. We open-source our training code at this http URL.

**arXiv ID:** 2505.12537
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
