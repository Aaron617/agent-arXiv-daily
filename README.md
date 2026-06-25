# Agent arXiv Daily

**Last Updated:** 2026-06-25 05:15:51

**Total Papers:** 88

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (5 papers)</h2></summary>

<details>
<summary><strong>The Clinician's Veto: Navigating Trust, Liability, and Uncertainty in Autonomous AI Prescribing</strong> - Eileanor LaRocco, Sarah Tan, Adarsh Subbaswamy, Anne Andrews, Andrew Taylor, Cree Gaskin, Chirag Agarwal - [[pdf]](https://arxiv.org/pdf/2606.25108)</summary>

**Abstract:** Autonomous AI systems are transitioning from advisory to autonomous roles for medication prescriptions. Recent United States bill H.R. 238 and Utah's prescription-renewal pilot both authorize AI to prescribe medications in an agentic capacity. While some regulatory guidelines suggest aggregate model performance metrics for clearance, they do not require i) calibrated per-prediction confidence for action-gated thresholds, ii) differentiated communication of uncertainty arising from model ignorance (epistemic) versus genuine clinical ambiguity (aleatoric), and iii) inferential transparency at the moment of decision that allows for liability allocation. Here, we present a regulatory and technical argument (tested with a survey of 136 U.S. prescribing clinicians) positioning these as minimum architectural requirements for safe autonomous prescribing. Our results suggest prescribing clinicians i) would not permit autonomous prescribing without a calibrated confidence-based escalation mechanism, ii) preferred a competing-options summary when uncertainty was aleatoric but shifted to abstention when uncertainty was epistemic, and iii) were only willing to accept additional liability when inferential transparency enabled a substantive judgment under acknowledged uncertainty. These findings indicate our recommended architectural features would encourage higher rates of clinician adoption, largely through collapsing much of what "autonomy" conventionally means. A system meeting these requirements would function less as an autonomous agent and more as a heavily supervised decision-support tool. As legislation and state pilots proceed, our technical argument backed by clinician perspectives provides opportunities for regulation to constrain the degree of autonomy ethically granted to AI in prescribing while aligning liability with the institutional actors who control system design and deployment.

**arXiv ID:** 2606.25108
</details>

<details>
<summary><strong>Memory Makes the Difference: Evaluating How Different Memory Roles Shape Conversational Agents</strong> - Yuxin Wang, Paul Thomas, Zhiwei Yu, Yuan Gao, Saeed Hassanpour, Soroush Vosoughi, Robert Sim, Nick Craswell - [[pdf]](https://arxiv.org/pdf/2606.25361)</summary>

**Abstract:** Prior research on memory mechanism in RAG-based conversational system has emphasized how memory is stored and retrieved. However, far less is known about how memories with different functional roles influence response quality. Specifically, how they shape an agent's responses under varying conversational contexts and whether they lead to substantively different response behaviors. Existing evaluations in conversational system are also largely reference-based, insufficiently capturing the nuances in responses that may address users' preferences differently. In this work, we probe the impact of different memory types in shaping agents' responses. We present a fine-grained taxonomy of conversational memory, classify retrieved memories into different role types, and design a user-centric evaluation framework that simulates user perspectives. Through comparative experiments on long-term datasets and frontier LLMs, our analysis reveal many differentiated effects of memories: e.g., clarifying memory improves responses' factual accuracy and constraint awareness, making them more correct and personalized; irrelevant memory reduces topic relevance and degrades constraint awareness. Despite the power of frontier LLMs, these findings shed light on how different memory types can be leveraged to produce more personalized responses and inspire further research in this direction.

**arXiv ID:** 2606.25361
</details>

<details>
<summary><strong>Engineering Reliable Autonomous Systems: Challenges and Solutions</strong> - Marie Farrell, Matt Luckcuck, Angelo Ferrando, Rafael C. Cardoso, Natasha Alechina, Marco Autili, Diana Benjumea Hernandez, Luciana Brasil Rebelo dos Santos, Daniela Briola, Ana Cavalcanti, Christian Colombo, Louise A. Dennis, Clare Dixon, Michael Fisher, Mario Gleirscher, Taylor Johnson, Charles Lesire, Livia Lestingi, Sven Linker, Brian Logan, Colin Paterson, Fabio Papacchini, Patrizio Pelliccione, Pedro Ribeiro, Maike Schwammberger, Silvia Lizeth Tapia Tarifa, Hazel Taylor, Jim Woodcock, Mengwei Xu, Yi Yang, Huan Zhang - [[pdf]](https://arxiv.org/pdf/2606.23760)</summary>

**Abstract:** Engineering reliable autonomous systems is an important and growing topic in computer science. As autonomous systems become more prevalent, easy-to-use techniques for building them reliably are increasingly important.
This workshop report captures and expands on the discussions at the Lorentz Center Workshop "Engineering Reliable Autonomous Systems" (ERAS), held from 10 to 14 June 2024. The workshop was co-organised by the organisers of the Workshop on Formal Methods for Autonomous Systems (FMAS) and the Workshop on Agents and Robots for reliable Engineered Autonomy (AREA). It brought together members of the FMAS and AREA communities, industry practitioners, and representatives from sectors where autonomous systems pose distinctive engineering challenges.
The workshop focused on three main research topics: techniques for verification and validation of autonomous systems; engineering real-world autonomous systems; and software architectures for safe autonomous systems. Its main outcome is a catalogue of challenges in these areas and, most importantly, a pathway to solutions. Some challenges can already be tackled by techniques that are well known in academia but have not yet become regularly used in practice. Other challenges remain unresolved and require further research. This roadmap is intended to support future research and industrial collaboration.

**arXiv ID:** 2606.23760
</details>

<details>
<summary><strong>One Body, Two Minds: Variable Autonomy Approach for a Co-embodied Robotic Hand</strong> - Piotr Koczy, Yuchong Zhang, Danica Kragic, Michael C. Welle - [[pdf]](https://arxiv.org/pdf/2606.25575)</summary>

**Abstract:** Assistive robotic systems face a fundamental trade-off: fully autonomous systems lack user agency, while fully user-controlled systems demand continuous cognitive effort. Existing shared autonomy approaches blend human and robot commands but are mostly deployed in separate physical bodies. We introduce co-embodiment with variable autonomy, where human and robot share a single physical body and operate at different autonomy levels across task phases, from mutual autonomy during object search and grasping to human-dominant control during actuation.
We present a co-embodied, wearable robotic hand that has its own ``mind'' and operates with variable autonomy levels. A learning-from-demonstration visuomotor diffusion policy enables autonomous grasping when the user positions the hand near known objects. Once grasped, the system signals completion and the human can actuate the grasped tool (drill, spray bottle, infrared thermometer, lighter, and ice-cream scoop) via hands-free head gestures. The human retains veto authority at all times through a release gesture that returns the system to the initial phase. Unlike blended autonomy, where control is continuously negotiated, our co-embodied approach consists of variable autonomy from full human control to full independent actions while maintaining physical coupling, realizing a one body, two minds paradigm.
In a user study with 44 participants performing five bimanual tasks, users rapidly adapted to this ``two minds'' paradigm: completion times improved by 23.3% across trials ($p < 0.001$, Cohen's $d = 0.94$), the best-performing policy variant reached a 93.6% task success rate, and acceptance ratings were high (5.70/7 overall impression, 5.52/7 daily use willingness). This work establishes co-embodiment with variable autonomy as a viable approach for assistive robotics, enabling human-robot collaboration through co-embodiment.

**arXiv ID:** 2606.25575
</details>

<details>
<summary><strong>A Low-Code Approach for the Automatic Personalization of Conversational Agents</strong> - Aaron Conrardy, Alfredo Capozucca, Jordi Cabot - [[pdf]](https://arxiv.org/pdf/2605.02384)</summary>

**Abstract:** The rise of Large Language Models (LLMs) has increased the demand for Conversational Agents (CAs) capable of understanding human conversations as part of web applications. While traditional CAs consist of deterministic states, LLMs enhance their capabilities to handle open conversations, handling arbitrary requests. Numerous tools exist that allow non-technical users to create such CAs. Yet, the creation of personalized CAs able to adapt to the profile of end-users to offer an optimal user experience remains in the hands of experienced developers implementing ad-hoc personalizations. In this work, we propose a pipeline that follows a low-code/no-code approach to facilitate the modeling and generation of personalized CAs. A pilot user study was performed to get preliminary results on perceived usability and usefulness and the full pipeline has been implemented on top of an open-source low-code platform.

**arXiv ID:** 2605.02384
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (16 papers)</h2></summary>

<details>
<summary><strong>AI Snitches Get Glitches: Towards Evading Agentic Surveillance</strong> - Hyejun Jeong, Dzung Pham, Amir Houmansadr, Eugene Bagdasarian - [[pdf]](https://arxiv.org/pdf/2606.25836)</summary>

**Abstract:** To better assist users with completing challenging tasks, AI agents mediate communications, access data, and interact with different APIs. Many employers (and even nation-states) already provide their users with this technology. However, widespread adoption of AI agents creates a new risk to abuse access to user data for another goal: surveilling users. These users might not even have the ability or permission to control the actions and data accesses of the surveilling agents.
We introduce and formalize the problem of agentic surveillance: the ability of an AI agent to analyze available information, craft a report, and send it out using available tools. To evaluate surveillance capabilities across different models, we create SurveilBench, a dataset of various reporting scenarios focusing on three domains: corporate, education, and police. We find that some models exhibit emergent (i.e., unprompted) tendencies to help surveillance, but they also report the attempts to surveil users to the government.
Finally, we repurpose prompt injections for evading surveillance and develop three evasion techniques that hide from, deceive, or induce over-escalation in surveillance agents. We conclude that agentic surveillance can already be easily implemented and, therefore, call for a comprehensive technical, ethical, and legislative framework to protect users.

**arXiv ID:** 2606.25836
</details>

<details>
<summary><strong>Autodata: An agentic data scientist to create high quality synthetic data</strong> - Ilia Kulikov, Chenxi Whitehouse, Tianhao Wu, Yixin Nie, Swarnadeep Saha, Eryk Helenowski, Weizhe Yuan, Olga Golovneva, Jack Lanchantin, Yoram Bachrach, Jakob Foerster, Xian Li, Han Fang, Sainbayar Sukhbaatar, Jason Weston - [[pdf]](https://arxiv.org/pdf/2606.25996)</summary>

**Abstract:** We introduce Autodata, a general method that enables AI agents to act as data scientists who build high quality training and evaluation data. We show how to train (meta-optimize) such a data scientist agent, so that it learns to create even stronger data. We describe the overall formulation, and a specific practical implementation, Agentic Self-Instruct. We conduct experiments on computer science research tasks, legal reasoning tasks and reasoning with mathematical objects, where we obtain improved results compared to classical synthetic dataset creation methods. Further, meta-optimizing the data scientist agent itself delivers an even larger performance uplift. Agentic data creation provides a way to convert increased inference compute into higher quality model training. Overall, we believe this direction has the potential to change the way we build AI data.

**arXiv ID:** 2606.25996
</details>

<details>
<summary><strong>ASAP: Agent-System Co-Design for Wall-Clock-Centered Auto HPO Research for ML Experiments</strong> - Taicheng Guo, Haomin Zhuang, Kehan Guo, Yujun Zhou, Nitesh V. Chawla, Olaf Wiest, Xiangliang Zhang - [[pdf]](https://arxiv.org/pdf/2606.25207)</summary>

**Abstract:** Hyperparameter Optimization (HPO) is essential for maximizing machine learning model performance, and its core challenge is sample efficiency: finding strong configurations within a limited budget. Because every HPO tool relies on a surrogate prior that imparts its own inductive bias, individual tools struggle once problems become sufficiently diverse and drift from these priors. Motivated by the reasoning and generalization capabilities of LLMs, recent work has explored using LLMs for HPO and reports improved per-iteration performance. Yet these methods share two limitations with a common origin: they use the LLM as a single-tool replacement evaluated by iteration count. (i) Deployed in place of prior tools, the LLM is itself constrained by its pretraining objective to one family of inductive-biased proposals; this single-source setup still fails to handle the full diversity of problems. (ii) Per-iteration evaluation ignores that, in real runs, LLM inference or tool execution is paid serially on top of model evaluation every round, so iteration-count gains do not translate into end-to-end wall-clock gains. We present ASAP, an agent-system co-design that addresses both limitations. On the agent side, ASAP uses the LLM to integrate a diverse pool of inductive-biased optimizers and to select among their proposals each round. On the system side, ASAP re-architects the loop to reduce end-to-end wall-clock while preserving regret quality: a prefix-stable prompt maximizes KV-cache reuse across rounds; speculation parallelism hides the remaining LLM and tool latency under model evaluation via a relative-error accept test; and a Self-Tuner adapts the speculation threshold from execution logs off the critical path. Extensive experiments on diverse modern HPO tasks show that ASAP consistently outperforms baselines, underscoring the value of tool integration and agent-system co-design.

**arXiv ID:** 2606.25207
</details>

<details>
<summary><strong>Staying In Character: Perspective-Bounded Memory For Book-Based Role-Playing Agents</strong> - Xushuo Tang, Junhe Zhang, Zihan Yang, Yifu Tang, Sichao Li, Longbin Lai, Zhengyi Yang - [[pdf]](https://arxiv.org/pdf/2606.25632)</summary>

**Abstract:** Recent LLM role-playing systems build character agents from novels by extracting characters, scenes, and relations. Yet long-narrative role-playing suffers from two failures: Factual Overreach, where shared retrieval or parametric memory lets a character use facts outside its perspective, and Stylistic Monotony, where profile descriptions flatten a character into a fixed voice. To address these failures, we propose REVERIEMEM, a three-layer memory architecture for book-based character agents. The episodic layer stores first-person scene memories; the semantic layer stores visibility-tagged facts; and the personality layer stores situation-dependent speech and behaviour patterns. For evaluation, we construct KBF-QA, a 4,386-question benchmark over eight novels for testing knowledge boundaries. REVERIEMEM improves Knowledge Boundary Fidelity by 34.6 percentage points over the strongest prior method. On BOOKWORLD's five-dimension pairwise narrative protocol, REVERIEMEM achieves a ~ 79% win rate, suggesting that perspective-bounded memory improves both boundary fidelity and character-grounded narrative generation.

**arXiv ID:** 2606.25632
</details>

<details>
<summary><strong>Is GraphRAG Needed? From Basic RAG to Graph-/Agentic Solutions with Context Optimization</strong> - Long Chen, Ryan Razkenari, Yuxuan Zhou, Yuan Tian, Rahul Ghosh, Venkatesh Pappakrishnan, Disha Ahuja, Vidya Sagar Ravipati - [[pdf]](https://arxiv.org/pdf/2606.25656)</summary>

**Abstract:** As advanced RAG variants like GraphRAG and Agentic RAG emerge, one leading question is when and how to use them. Here, we introduce a framework for different RAG scenarios evaluation and comparison on semi-structured knowledge bases, including regular RAG, GraphRAG, Modular RAG and Agentic RAG. We provide implementation for 9 standardized RAG scenarios, and conduct experiments for a comprehensive comparison. These scenarios are designed for real use cases regarding data and domain restrictions, spanning from simple document-based retrieval to advanced features such as hybrid text-graph retrieval, integration with computed or pre-defined domain knowledge graphs, agentic multi-step planning, and agent-graph integration. Besides, we present a novel context engineering method for GraphRAG and Agentic RAG, addressing the context/memory overflow issues, efficiently managing text and graph retrievals with new representations and agentic loop design, leading to 19%-53% reduction on token usage. Moreover, further analysis identifies a retrieval-generation gap where expanded retrieval does not proportionally improve generation quality, suggesting retrieval-oriented metrics overstate advanced retrieval benefits. This work provides data-driven insights on when and how to use them for building production-ready intelligent RAG systems.

**arXiv ID:** 2606.25656
</details>

<details>
<summary><strong>Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets</strong> - Divake Kumar, Sina Tayebati, Devashri Naik, Amanda Sofie Rios, Nilesh Ahuja, Omesh Tickoo, Ranganath Krishnan, Amit Ranjan Trivedi - [[pdf]](https://arxiv.org/pdf/2606.25760)</summary>

**Abstract:** Computer-use agents turn vision-language model (VLM) predictions into executable GUI clicks, so reliable uncertainty estimates are essential for rejection, calibration, miss-severity ranking, and spatial safety regions. Yet evidence on post-hoc uncertainty quantification (UQ) for these agents is fragmented across isolated model and dataset pairs, leaving it unclear whether UQ rankings stay stable when the agent, benchmark, or observable interface changes. We present Argus, a cross-regime benchmark for post-hoc UQ in single-step executable GUI grounding: a 27-method open-weight matrix over 4 VLM agents and 4 datasets, plus an 8-method closed-source matrix across 3 frontier vendors where logits, hidden states, and attention maps are unavailable. Evaluated methods span logit-based scores, sampling and consistency measures, hidden-state and density estimators (Mahalanobis, SAPLMA), attention-based scores, P(True) and verbalised-confidence prompting, and split-conformal prediction. The main finding is selective transfer: UQ rankings are stable across datasets for a fixed model, but degrade across model classes and observable interfaces. Hidden-state and density methods are the most stable open-weight family, while CoCoA-1MCA, Focus, sampling-based scores, and verbalised self-assessment win in specific regimes. Within-model ranking transfer is strong (Spearman rho up to 0.969), but cross-tier transfer to closed-source vendors averages only +0.08, so closed-source UQ should be reranked on the target rather than extrapolated. Conformal click regions show score-level discrimination is not enough for deployment: locally weighted disks shrink radii by 40-60% when the plug-in UQ is calibrated, but coverage degrades under calibration-test or interface mismatch. We release per-item records, calibration/test splits, UQ scores, and analysis scripts for regime-aware UQ selection in GUI agents.

**arXiv ID:** 2606.25760
</details>

<details>
<summary><strong>Exploring Information Seeking Agent Consolidation</strong> - Guochen Yan, Jialong Wu, Zhengwei Tao, Bo Li, Qintong Zhang, Jiahao Xu, Haitao Mi, Yuejian Fang, Qingni Shen, Wentao Zhang, Zhonghai Wu - [[pdf]](https://arxiv.org/pdf/2602.00585)</summary>

**Abstract:** Information-seeking agents have emerged as a powerful paradigm for knowledge-intensive tasks, yet today's systems remain specialized for the open web, documents, or local knowledge bases, hindering scalable and cross-domain deployment. We present the first systematic empirical study of consolidating these information-seeking agents into a single foundation agentic model. We compare two paradigms -- \emph{data-level mixing}, which trains a unified model on a mixture of datasets, and \emph{parameter-level merging}, which merges independently trained experts in parameter space -- across 3 training scenarios, evaluating \textbf{26} representative parameter-level methods on \textbf{10} benchmarks. To compare across heterogeneous benchmarks, we introduce a geometric Composite Score and an Imbalance Score that describe overall performance and task skew. Our analysis shows that (i) well-designed parameter-level merging attains parity with data mixing at a fraction of its training cost and is order-agnostic; (ii) parameter-level merging structurally preserves out-of-domain capabilities that data mixing universally forgets; and (iii) cross-scenario stability is strongly tied to consolidation quality. We distil our observations into a method-selection guide and design principles for next-generation merging operators.

**arXiv ID:** 2602.00585
</details>

<details>
<summary><strong>Plausible but Wrong: A case study on Agentic Failures in Astrophysical Workflows</strong> - Shivam Rawat, Lucie Flek - [[pdf]](https://arxiv.org/pdf/2604.25345)</summary>

**Abstract:** Agentic AI systems are increasingly being integrated into scientific workflows, yet their behavior under realistic conditions remains insufficiently understood. We evaluate CMBAgent across two workflow paradigms and eighteen astrophysical tasks. In the One-Shot setting, access to domain-specific context yields an approximately ~6x performance improvement (0.85 vs. ~0 without context), with the primary failure mode being silent incorrect computation - syntactically valid code that produces plausible but inaccurate results. In the Deep Research setting, the system frequently exhibits silent failures across stress tests, producing physically inconsistent posteriors without self-diagnosis. Overall, performance is strong on well-specified tasks but degrades on problems designed to probe reasoning limits, often without visible error signals. These findings highlight that the most concerning failure mode in agentic scientific workflows is not overt failure, but confident generation of incorrect results. We release our evaluation framework to facilitate systematic reliability analysis of scientific AI agents.

**arXiv ID:** 2604.25345
</details>

<details>
<summary><strong>Agent-as-a-Router: Agentic Model Routing for Coding Tasks</strong> - Pengfei Zhou, Zhiwei Tang, Yixing Ma, Jiasheng Tang, Yizeng Han, Zhenglin Wan, Fanqing Meng, Wei Wang, Bohan Zhuang, Wangbo Zhao, Yang You - [[pdf]](https://arxiv.org/pdf/2606.22902)</summary>

**Abstract:** Real-world users typically have access to multiple Large Language Models (LLMs) from different providers, and these LLMs often excel at distinct domains, yet none dominate all. Consequently, routing each task to the most suitable model becomes critical for both performance and cost. Existing routers treat this as a static, one-off classification problem. However, we identify the performance bottleneck for these routers as information deficit: simply augmenting a vanilla LLM router with performance statistics at the task-dimension level yields a 15.3% relative gain, surpassing a heuristic router built on the same dimension-level priors. Motivated by this finding, we propose Agent-as-a-Router, a framework that formalizes routing as a C-A-F loop (Context->Action->Feedback->Context). It closes the information gap by accumulating execution-grounded experience during deployment. We instantiate this framework as ACRouter, composed of an Orchestrator, a Verifier, a Memory module, and introduce CodeRouterBench, an evaluation environment comprising ~10K task instances with verified scores from 8 frontier LLMs, enabling regret-based router comparison on streaming tasks. Experiments show that ACRouter achieves the lowest cumulative regret on in-distribution tasks and generalizes to out-of-distribution agentic-programming tasks, demonstrating that our routing framework actively closes the information gap. Codes and benchmarks are released at this https URL.

**arXiv ID:** 2606.22902
</details>

<details>
<summary><strong>AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents</strong> - Zheyuan Zhang, Zehao Wen, Alvin Zhang, Andrew Wang, Jianwen Xie, Daniel Khashabi, Tianmin Shu - [[pdf]](https://arxiv.org/pdf/2606.24893)</summary>

**Abstract:** For agents to learn continuously from interaction with the world at test time, they must be able to explore effectively, acquire new world knowledge and skills, retain relevant episodic experiences, and plan over long horizons. To evaluate these key abilities of test-time continual learning agents, we introduce AgentOdyssey, a novel evaluation framework that procedurally generates open-ended text games with rich entities, world dynamics, and long-horizon tasks. Critically, AgentOdyssey goes beyond the conventional machine learning assumption that learning does not occur at test time by placing agents in a continuous, long-horizon setting that interleaves learning and inference throughout deployment. We further propose a multifaceted evaluation methodology that measures not only game progress but also offers diagnostic tests on world knowledge acquisition, episodic memory, object and action exploration, action diversity, and model cost. We evaluate diverse agent paradigms in the generated games. Our experimental results reveal critical limits in agents' key abilities, as well as factors that influence their meaningful horizon. Although performance scales with stronger base models, even the top agent remains far below human performance, leaving substantial headroom for improvement. Among agent mechanisms, we find that short-term memory benefits multiple agent paradigms and is an important component of agent test-time training.

**arXiv ID:** 2606.24893
</details>

<details>
<summary><strong>Beyond Function Calling: Benchmarking Tool-Using Agents under Tool-Environment Unreliability</strong> - Yang Tian, Zhengpeng Shi, Bo Zhao - [[pdf]](https://arxiv.org/pdf/2606.25819)</summary>

**Abstract:** Large language models are increasingly deployed as agents that solve tasks by interacting with external tool environments. Although recent tool-use benchmarks increasingly cover complex task settings, they still largely assume clean, stable, and trustworthy tool environments, leaving tool-environment unreliability insufficiently examined. We introduce ToolBench-X, a benchmark for evaluating agents under recoverable reliability hazards. ToolBench-X contains executable multi-step tasks across diverse domains and sequential, parallel, and mixed workflows, each paired with deterministic tools and a canonical final answer for automatic evaluation. Starting from clean tool environments, ToolBench-X injects five structured hazard types: Specification Drift, Invocation Error, Execution Failure, Output Drift, and Cross-source Conflict. Crucially, each injected instance remains solvable through at least one valid recovery path, such as retrying, fallback, verification, or cross-checking. Experiments reveal a substantial reliability gap: agents that perform well with reliable tools often fail under recoverable hazards. Further analysis shows that failures are driven less by tool-use volume or inference budget than by limited hazard diagnosis and ineffective recovery. Targeted recovery hints recover many failed tasks, while test-time scaling yields more limited gains. These results suggest that tool-use evaluation should move beyond function-call accuracy toward task completion under unreliable tool environments. The code and data is available at this https URL.

**arXiv ID:** 2606.25819
</details>

<details>
<summary><strong>Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory</strong> - Beining Wu, Zihao Ding, Jun Huang, Yanxiao Zhao - [[pdf]](https://arxiv.org/pdf/2606.25115)</summary>

**Abstract:** On-device language-model agents improve by accumulating experience in retrieved memory rather than by updating weights. This memory is hard-bounded and exposed: it consumes RAM and energy, reaches peers through a thin uplink, and becomes an attack surface because it is writable by what the agent reads. Existing systems each cover one part of this problem: agentic memories grow without a budget, on-device methods keep entries by success alone, and poisoning is studied mainly as an attack rather than as a memory-governance problem. We propose \sys{}, a single net-value-per-byte score that governs an agent's experience-memory lifecycle. The main idea is to let the budget act as the curator: each entry is scored as value minus harm, per byte, so one ruler decides what to keep, share, and trust. \sys{} makes three decisions: (1) \textbf{KEEP} evicts low-value bytes under the RAM and energy budget; (2) \textbf{SHARE} sends an insight only when its value exceeds its uplink cost; and (3) \textbf{TRUST} gates a peer entry by provenance. On language-model-agent task-drift benchmarks and a real heterogeneous Jetson testbed with two robot-arm nodes and a hub, \sys{} reduces memory by $2.7\times$ and uplink by $2.4\times$, drives injection success from 0.75 to zero, and raises accuracy on cases corrupted by poison or stale memory. Curating by net value reduces footprint, energy, uplink, and injection success together without reducing accuracy. In this setting, forgetting by net value improves the agent rather than weakening it.

**arXiv ID:** 2606.25115
</details>

<details>
<summary><strong>Polaris: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair</strong> - Aditya Kakade, Vivek Srivastava, Shirish Karande - [[pdf]](https://arxiv.org/pdf/2603.23129)</summary>

**Abstract:** Gödel agent realize recursive self-improvement: an agent inspects its own policy and traces and then modifies that policy in a tested loop. We introduce Polaris, Gödel agent for compact models that performs policy repair via experience abstraction, turning failures into policy updates through a structured cycle of analysis, strategy formation, abstraction, and minimal code patch repair with conservative checks. Unlike response level self correction or parameter tuning, Polaris makes policy level changes with small, auditable patches that persist in the policy and are reused on unseen instances within each benchmark. As part of the loop, the agent engages in meta reasoning: it explains its errors, proposes concrete revisions to its own policy, and then updates the policy. To enable cumulative policy refinement, we introduce experience abstraction, which distills failures into compact, reusable strategies that transfer to unseen instances. On MGSM, DROP, GPQA, and LitBench (covering arithmetic reasoning, compositional inference, graduate-level problem solving, and creative writing evaluation), a 7-billion-parameter model equipped with Polaris achieves consistent gains over the base policy and competitive baselines.

**arXiv ID:** 2603.23129
</details>

<details>
<summary><strong>HEART: Coordination of Heterogeneous Expert Agents for Physically Grounded Robotic Task Planning</strong> - Junho Lee, Seabin Lee, Wonjong Lee, Nayoung Kim, Moonjeong Kang, Changjoo Nam - [[pdf]](https://arxiv.org/pdf/2606.25404)</summary>

**Abstract:** Large Language Models (LLMs) can reason over complex instructions but often fail to satisfy the physical and spatial constraints required for robotic task planning. Recent LLM-based planners directly translate text into action sequences, yet they lack structured reasoning about feasibility, reachability, and logical order, resulting in invalid or incomplete plans. We present a heterogeneous multi-LLM framework that decomposes instructions into atomic reasoning tasks and allocates them to role-specialized expert agents under a token budget for real-world computational and communicational constraints. By combining role-oriented reasoning from heterogeneous agents followed by constraint-driven plan synthesis, HEART validates capability, reachability, and constraint conditions before planning and helps produce physically executable plans while maintaining efficiency. Experiments across different household benchmarks show that HEART consistently improves plan success compared to single-LLM and rule-based planners, demonstrating that heterogeneous LLM collaboration enables robust and scalable robotic task planning under resource constraints.

**arXiv ID:** 2606.25404
</details>

<details>
<summary><strong>ASSCG: Just-Right Gating over Chattering for Fast-Slow LLM Planning in Autonomous Driving</strong> - Sining Ang, Yuan Chen, Liu Haiyan, Xuanyao Mao, Jason Bao, Xuliang, Bingchuan Sun, Yan Wang - [[pdf]](https://arxiv.org/pdf/2606.25509)</summary>

**Abstract:** Large language models (LLMs) can improve autonomous driving planning but are costly to query online, and existing fast-slow planners often rely on hand-designed triggering rules that either over-call the slow system or call it at the wrong times. We formulate slow-system invocation as a resource-aware sequential decision problem and propose the Adaptive Slow-System Control Gate (ASSCG), which makes frame-level Query/Cache/Drop decisions to refresh, reuse, or suppress slow guidance. ASSCG uses an RWKV backbone for efficient long-horizon gating and is trained with supervised fine-tuning followed by GRPO-style compute-aware reinforcement fine-tuning. We apply ASSCG to two different fast-slow architectures: (i) AsyncDriver on nuPlan Hard20 closed-loop evaluation, where ASSCG improves score to 67.28 (+2.28) while reducing average end-to-end inference latency by 60%; and (ii) a RecogDrive-based dual system that we build by replacing its original VLM-2B module with a lightweight ViT-based fast planner and adding an LLM slow planner, evaluated on NAVSIM, where ASSCG achieves 91.4 PDMS (+0.6) and increases average speed by 25%. The project page, including video visualizations and additional results, is available at this https URL.

**arXiv ID:** 2606.25509
</details>

<details>
<summary><strong>Colon-Bench: An Agentic Workflow for Scalable Dense Lesion Annotation in Full-Procedure Colonoscopy Videos</strong> - Abdullah Hamdi, Changchun Yang, Xin Gao - [[pdf]](https://arxiv.org/pdf/2603.25645)</summary>

**Abstract:** Early screening via colonoscopy is critical for colon cancer prevention, yet developing robust AI systems for this domain is hindered by the lack of densely annotated, long-sequence video datasets. Existing datasets predominantly focus on single-class polyp detection and lack the rich spatial, temporal, and linguistic annotations required to evaluate modern Multimodal Large Language Models (MLLMs). To address this critical gap, we introduce Colon-Bench, generated via a novel multi-stage agentic workflow. Our pipeline seamlessly integrates temporal proposals, bounding-box tracking, AI-driven visual confirmation, and human-in-the-loop review to scalably annotate full-procedure videos. The resulting verified benchmark is unprecedented in scope, encompassing 528 videos, 14 distinct lesion categories (including polyps, ulcers, and bleeding), over 300,000 bounding boxes, 213,000 segmentation masks, and 133,000 words of clinical descriptions. We utilize Colon-Bench to rigorously evaluate state-of-the-art MLLMs across lesion classification, Open-Vocabulary Video Object Segmentation (OV-VOS), and video Visual Question Answering (VQA). The MLLM results demonstrate surprisingly high localization performance in medical domains compared to SAM-3. Finally, we analyze common VQA errors from MLLMs to introduce a novel "colon-skill" prompting strategy, improving zero-shot MLLM performance by up to 9.7% across most MLLMs. The dataset and the code are available at this https URL .

**arXiv ID:** 2603.25645
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>TRUSTMEM: Learning Trustworthy Memory Consolidation for LLM Agents with Long-Term Memory</strong> - Tianyu Yang, Sudipta Paul, Vijay Srinivasan, Vivek Kulkarni, Srinivas Chappidi - [[pdf]](https://arxiv.org/pdf/2606.25161)</summary>

**Abstract:** Large language model (LLM) agents rely on long-term memory to support extended interactions and personalized assistance beyond finite context windows. Existing memory agents actively update external memory through generated write, revise, and delete operations, but these updates may omit important information, corrupt existing memory, or introduce unsupported hallucinated content. Once stored, such errors become persistent system-state failures that can affect future reasoning and generation. In this paper, we propose TrustMem, a framework designed to improve the trustworthiness of memory consolidation. TrustMem relies on a Memory Transition Verifier to evaluate the transition process of memory updates in terms of coverage, preservation, and faithfulness. It further constructs preference pairs among candidate updates under the same memory state, enabling preference-guided reinforcement learning to directly optimize memory updating behaviors. Extensive experiments demonstrate that TrustMem improves both memory utility and reliability: it achieves state-of-the-art results across MemoryAgentBench, HaluMem, and the Mem-alpha validation set, improves HaluMem memory extraction by 12.14 F1 points, and reduces transition-level omission, corruption, and hallucination by 40.1\%, 79.1\%, and 50.0\%, respectively, compared with the strongest baseline for each error type.

**arXiv ID:** 2606.25161
</details>

<details>
<summary><strong>GUI agent: Guided Exploration of User-Sensitive Screens</strong> - Aradhana Nayak, Mussadiq Nazeer, Wang Peng, Feng Liu - [[pdf]](https://arxiv.org/pdf/2606.25705)</summary>

**Abstract:** LLM agents are increasingly being used to automate tasks for users within an open GUI environment. They inevitably encounter screens containing user-sensitive information, for which takeover of task execution by the user is highly desirable or even necessary. State-of-the-art LLM-driven agents are usually fine-tuned to complete tasks regardless of the safety implications of their actions. This makes their real-world deployment difficult and adversely affects the reliability. Therefore, it is crucial to identify and categorize user-sensitive states and define user-sensitive queries. This dataset would be to engineers to recognize and request handover to the user in critical scenarios. This short paper develops an explorer agent that systematically explores the query space starting from one demonstrated task to identify queries that, if executed, would lead to user-sensitive states in a GUI environment.

**arXiv ID:** 2606.25705
</details>

<details>
<summary><strong>BiPACE: Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents</strong> - Hanyang Wang, Weijieying Ren, Yuxiang Zhang, Ding Cao, Zhizhao Zeng, Ke Zeng, Tianxiang Zhao - [[pdf]](https://arxiv.org/pdf/2606.25556)</summary>

**Abstract:** Stepwise group-based RL is an attractive way to train long-horizon LLM agents without a learned critic: it reuses multiple sampled rollouts to estimate local advantages. Its weakness is less visible but more fundamental: every group-relative estimator assumes that the steps it compares are equivalent for credit assignment. We show that current agentic variants violate this assumption through a state-action credit mismatch. The observation-hash partition is overly fine on the state side, creating singleton groups with zero step-level signal, while a single within-group mean is too coarse on the action side, mixing state-value estimation with action-specific credit. We introduce BiPACE (Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation), a drop-in advantage estimator that fixes both sides without adding a critic, auxiliary loss, or extra rollouts. BiGPO clusters steps by cosine distance in the actor's own hidden-state geometry, an empirical policy-induced proxy for bisimulation that substantially lowers the singleton rate left by observation hashing. PACE then recenters returns within each behavioral cluster using action-conditioned peer baselines; its Q-style instance estimates a local Q(s,a)-V(s) nonparametrically. On ALFWorld/Qwen2.5-7B, BiPACE_Q raises overall validation success from GiGPO's 90.8 to $97.1\pm0.9$ over three seeds, and crosses the 95% threshold on every seed, which GiGPO never does within the same budget. On Qwen2.5-1.5B it reaches $93.5\pm1.2$ versus GiGPO's 86.7, and on WebShop and TextCraft it improves over GRPO and GiGPO at both model scales. The measured BiPACE-specific overhead is 11.3% of a single training-step wall time. Yet it changes the estimator's comparison unit from surface identity to approximate behavioral equivalence plus action-side counterfactuals. The code is available at this https URL.

**arXiv ID:** 2606.25556
</details>

<details>
<summary><strong>Semantic Consistency Policy Optimization for Reinforcement Learning of LLM Agents</strong> - Peng Xu, Sijia Chen, Junzhuo Li, Xuming Hu - [[pdf]](https://arxiv.org/pdf/2606.25852)</summary>

**Abstract:** Group-based reinforcement learning effectively post-trains LLM agents for long-horizon, sparse-reward tasks by deriving step-level credit from trajectory outcomes. However, this ties a step's credit to its rollout's final outcome: semantically near-identical intermediate steps receive opposite credit depending on whether their trajectory eventually succeeded or failed. Such semantic credit inconsistency sends conflicting gradients to similar actions and wastes the partially-correct progress inside failed rollouts. Motivated by this, we propose Semantic Consistency Policy Optimization (SCPO), a value-free reward-shaping method that mitigates this inconsistency by recovering step-level credit from successful siblings in the same rollout group. Concretely, SCPO scores each failed step against a successful sibling and adds positive step-level credit for new progress along that sibling. On ALFWorld and WebShop, SCPO matches or exceeds strong group-based baselines, reaching 93.7+/-4.1 percent success on ALFWorld and 74.8+/-2.0 percent on WebShop at 1.5B parameters, with gains concentrated on the hardest multi-step tasks.

**arXiv ID:** 2606.25852
</details>

<details>
<summary><strong>Explainable Control Framework (XCF) based on Fuzzy Model-Agnostic Explanation and LLM Agent-Supported Interface</strong> - Faliang Yin, Hak-Keung Lam, David Watson - [[pdf]](https://arxiv.org/pdf/2606.25941)</summary>

**Abstract:** Increasing demand for precise and reliable control in complex scenarios has led to the development of increasingly sophisticated controllers, including data-driven approaches employing closed box models and mathematically rigorous yet complex designs. This complexity highlights the needs for explainable control that can provide human-understandable insights into controller behavior. In this paper, an explainable control framework (XCF) along with supporting algorithms and user interface are proposed to explain how controllers determine their control actions and their underlying working mechanism. The novel contributions of this work are threefold: First, the XCF is designed to provide model-agnostic explanations for controllers in closed-loop systems and can optionally refine local explanations by system response dynamics. Second, a novel explanation method, hierarchical fuzzy model-agnostic explanation for control systems (HFMAE-C), is proposed based on the designed framework. The HFMAE-C employs a fuzzy logic system to approximate the controller's behavior and system dynamics, providing sample, local, domain and universe level explanations via IF-THEN rules revealing the controller's decision logic and salience values quantifying the contribution of system states to control actions. Third, a large language model agent-supported user interface is developed to automatically analyze user requirements, select appropriate algorithms, interpret the generated explanations to a natural language report, and provide interactive consultation. Case studies on inverted pendulum system and Turtlebot obstacle avoidance demonstrate the effectiveness of the proposed method through simulated user experiments and quantitative comparisons with mainstream explainable control approaches.

**arXiv ID:** 2606.25941
</details>

<details>
<summary><strong>Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents</strong> - Changdae Oh, Wendi Li, Seongheon Park, Samuel Yeh, Tanwi Mallick, Sharon Li - [[pdf]](https://arxiv.org/pdf/2606.26080)</summary>

**Abstract:** Process reward models enable fine-grained, step-level evaluation of LLMs, yet building them for agentic settings remains prohibitively difficult: long-horizon interactions, irreversible actions, and stochastic environment feedback make both human annotation and Monte Carlo estimation infeasible at scale. In this work, we show that reinforcement learning (RL) post-training already provides the ingredients for effective step-level scoring, eliminating the need for dedicated reward model training altogether. Concretely, we derive an implicit advantage under a general stochastic Markov decision process, which we term progress advantage -- log-probability ratio between the RL-trained policy and its reference policy exactly recovers the optimal advantage function. This formulation makes the resulting signal annotation-free, domain-agnostic, and available as a byproduct of the standard RL post-training pipeline. We validate the effectiveness of the progress advantage across three different applications: test-time scaling, uncertainty quantification, and failure attribution on five benchmarks and four model families. Across all settings, it consistently outperforms confidence-based baselines and, despite requiring no task-specific training, surpasses dedicated trained reward models. We complement these results with deeper analyses on characteristics of progress advantage, offering practical guidance for adoption in real-world agentic systems.

**arXiv ID:** 2606.26080
</details>

<details>
<summary><strong>Shepherd: Enabling Programmable Meta-Agents via Reversible Agentic Execution Traces</strong> - Simon Yu, Derek Chong, Ananjan Nandi, Dilara Soylu, Jiuding Sun, Christopher D Manning, Weiyan Shi - [[pdf]](https://arxiv.org/pdf/2605.10913)</summary>

**Abstract:** As LLM agent systems take on more complex tasks, they increasingly rely on meta-agents: higher-order agents that create, operate on and manage other agents. Meta-agent operations such as coordinating agents, halting risky actions before execution, or repairing failed runs, require runtime manipulation of agentic execution. Yet existing agentic substrates make this difficult: they expose only transcripts and environment snapshots, forcing meta-agents to build ad hoc tooling to reconstruct and operate over full execution state. Therefore, we introduce Shepherd, a Python substrate grounded in functional programming principles, where an agent's execution is itself a first-class object that a meta-agent can easily inspect and transform. Every model action, tool call, and environment change becomes a structured event in a reversible, Git-like execution trace, where any past state can be reverted 5x faster than docker commit and fork. Three example use cases show Shepherd's versatility: (1) a supervisor meta-agent prevents conflicts among parallel coding agents, lifting pair-coding pass rate from 28.8% to 54.7% on CooperBench; (2) a counterfactual optimization meta-agent repairs agent workflows by proposing edits and replaying runs from the point of changed behavior, outperforming MetaHarness on Terminal-Bench 2.0 by 12.8% with 58% lower wall-clock; (3) a training meta-agent picks fork points during rollouts to improve credit assignment in long-horizon agentic RL, doubling GRPO's uplift on Terminal-Bench 2.0. We open-source Shepherd to enable principled and efficient operations over agentic execution for both users and meta-agents.

**arXiv ID:** 2605.10913
</details>

<details>
<summary><strong>Domain-Specific Agents for Cherenkov Telescope Array Control Software and Gamma-Ray Data Analysis</strong> - Dmitriy Kostunin, Elisa Jones, Vladimir Sotnikov, Valery Sotnikov, Sergo Golovachev, Alexandre Strube - [[pdf]](https://arxiv.org/pdf/2510.01299)</summary>

**Abstract:** We present domain-adapted large language model agents designed to support Cherenkov Telescope Array operation and data analysis. The agents combine contextual knowledge with automated validation and iterative correction to produce more reliable outputs. This approach reduces manual effort, improves consistency, and helps accelerate operational and scientific workflows. The results demonstrate the potential of agentic systems as practical assistants in specialized research environments.

**arXiv ID:** 2510.01299
</details>

<details>
<summary><strong>Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents</strong> - Dehao Tao, Guoliang Ma, Yongfeng Huang, Minghu Jiang - [[pdf]](https://arxiv.org/pdf/2601.03785)</summary>

**Abstract:** Long-term human-agent dialogues are organized by topic continuity: adjacent turns often develop the same goal, plan, problem, or event, while related activities may recur across distant sessions. Yet many LLM agent memory systems first decompose histories into isolated turns or fixed-size chunks, then compensate through enrichment, consolidation, or retrieval mechanisms still tied to semantic proximity or fragment-level records. This weakens temporal and causal organization and biases memory access toward semantic proximity rather than task- or topic-level continuity. We introduce \emph{Membox}, a hierarchical memory architecture that instantiates topic continuity as an explicit organization layer for agent memory. Its \textbf{Topic Loom} incrementally organizes dialogue streams into boxes whose internal turns follow the same local topic, while its \textbf{Trace Weaver} links extracted events across boxes into macro-topic traces that recover recurring activities, goals, and factual developments across distant sessions. On LoCoMo, Topic-Loom-only retrieval improves over the best Mem0/A-MEM retrieval-depth setting by 13.00 F1 points (53.95 vs. 40.95), and trace-expanded retrieval further raises F1 to 55.28; with GPT-4o, trace-expanded retrieval reaches 59.71 F1. Additional DialSim results show the same gain from adding cross-box traces in multi-party dialogue. These results show that local topic-continuity organization and macro-topic trace expansion improve long-range memory beyond semantic retrieval over fragmented records.

**arXiv ID:** 2601.03785
</details>

<details>
<summary><strong>The Interplay of Harness Design and Post-Training in LLM Agents</strong> - Kyungmin Kim, Youngbin Choi, Seoyeon Lee, Suhyeon Jun, Dongwoo Kim, Sangdon Park - [[pdf]](https://arxiv.org/pdf/2606.25447)</summary>

**Abstract:** Tool-integrated LLM agents are often wrapped within a harness: the scaffolding that determines which tools are exposed, how they are described, and what auxiliary information accompanies each per-step observation. While agents are routinely post-trained, this scaffolding is typically treated as a fixed engineering detail, with design effort limited to the training-free regime. Moreover, existing post-training algorithms assume a static environment, even though tool environments and tasks often shift upon deployment. To address this gap, we extend $\texttt{ALFWorld}$ (i) to treat the harness as a controllable design dimension and (ii) to support evaluation under task and tool environment shifts. Building on this, we systematically analyze how the harness design influences post-training in both in-distribution and out-of-distribution (OOD) settings. We empirically show that harness-aware post-training not only improves in-distribution performance but also enables agents to robustly adapt to OOD settings. Under a harness with minimal design effort, post-training suffers a drastic performance drop under stronger tool environment shifts, further highlighting the importance of harness-aware post-training under such shifts.

**arXiv ID:** 2606.25447
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (20 papers)</h2></summary>

<details>
<summary><strong>The Hitchhiker's Guide to Agentic AI: From Foundations to Systems</strong> - Haggai Roitman - [[pdf]](https://arxiv.org/pdf/2606.24937)</summary>

**Abstract:** The Hitchhiker's Guide to Agentic AI is a comprehensive practitioner's reference for building autonomous AI systems. The book covers the full stack from first principles to production deployment, organized around a central thesis: building great agentic systems requires understanding every layer of the pipeline, not just one. The book opens with the LLM substrate -- transformer architecture, GPU systems, training and fine-tuning (SFT,LoRA, MoE), model compression, and inference optimization -- treated as essential foundations rather than the primary focus. It then develops the alignment and reasoning layer: reinforcement learning from human feedback (RLHF), PPO, DPO and its variants, GRPO, reward modeling, and RL for large reasoning models including chain-of-thought and test-time scaling. The second half is devoted to agentic AI proper. Topics include agentic training and trajectory-based RL, retrieval-augmented generation (RAG and Agentic RAG), memory systems (in-context, external, episodic, and semantic), agent harness design and context management, and a taxonomy of agent design patterns. Inter-agent coordination is covered in depth: the Model Context Protocol (MCP), agent skills and tool use, the Agent-to-Agent (A2A) communication protocol, and multi-agent architectures spanning centralized, decentralized, and hierarchical topologies. The book concludes with agent development frameworks, agentic UI design, evaluation methodology for agentic tasks, and production deployment. Each chapter pairs rigorous theoretical foundations with implementation guidance, code examples, and references to the primary literature.

**arXiv ID:** 2606.24937
</details>

<details>
<summary><strong>Diagnosing and Mitigating Compounding Failures in Agentic Persuasion via Taxonomic Strategy Retrieval</strong> - Pradyumna Narayana, Sana Ayromlou, Purvi Sehgal - [[pdf]](https://arxiv.org/pdf/2606.24976)</summary>

**Abstract:** Foundation-model agents in multi-step, open-ended environments frequently suffer from compounding errors, where early mistakes contaminate long-horizon trajectories. While Multi-Agent Debate (MAD) succeeds in deterministic domains, agents in subjective tasks like persuasion experience severe problem drift and sycophantic conformity. We identify semantic leakage in standard Retrieval-Augmented Generation (RAG) as a reproducible trigger for these failures, as standard RAG prioritizes vocabulary overlap over logical necessity.
To eliminate this leakage, we introduce Taxonomic Strategy RAG (TS-RAG), a systems intervention that routes strategies through a discrete categorical bottleneck to decouple argumentative structure from topical content. Zero-shot, cross-domain evaluations demonstrate that TS-RAG significantly improves the transfer of abstract logic where standard semantic retrieval collapses. Crucially, TS-RAG acts as a "capability bridge" in asymmetric deployments, empowering lightweight persuaders to consistently defeat parametrically superior opponents (improving win rates from 70.5 to 78.5) and accelerating argumentative efficiency. Finally, we introduce trace-level diagnostics via a turn-by-turn Debate State Representation (DSR), demonstrating the necessity of strict constraints to prevent evaluation collapse via default agentic sycophancy.

**arXiv ID:** 2606.24976
</details>

<details>
<summary><strong>To Isolate or to Score? Model-Adaptive Assessment for Cost-Efficient Multi-Agent RAG</strong> - Jungseob Lee, Chanjun Park, Heuiseok Lim - [[pdf]](https://arxiv.org/pdf/2606.25191)</summary>

**Abstract:** Multi-agent document assessment for retrieval-augmented generation is computationally expensive, driving practitioners toward smaller, deployable models whose assessment mechanisms remain poorly understood. We conduct a controlled study of training-free interventions on 7B-9B instruction-tuned models across diverse QA benchmarks, revealing a sharp dichotomy in how models benefit from assessment. For weaker baselines, the dominant mechanism is per-document isolation. Astoundingly, assessment-free isolation matches full multi-agent assessment, demonstrating that resolving multi-document context confusion, rather than scoring quality, drives outsized gains of up to 50 percentage points. Conversely, for strong baselines where scoring quality matters, we introduce Reasoning-Score Coupling, a label-free perturbation probe that classifies scoring behavior. Integrating these findings, we propose MADARA, a model-adaptive routing architecture. Crucially, MADARA's diagnostic thresholds derived from a single pilot model generalize zero-shot to four unseen model families, providing a robust, lightweight pipeline to eliminate computational overhead.

**arXiv ID:** 2606.25191
</details>

<details>
<summary><strong>Agentic Knowledge Tracing: A Multi-Agent LLM Architecture for Stealth Assessment of Financial Literacy in Serious Games</strong> - Gabriel Santos, Rita Julia, Marcelo Nascimento - [[pdf]](https://arxiv.org/pdf/2606.25358)</summary>

**Abstract:** Assessing financial literacy during gameplay without disrupting the learning experience remains a key challenge in serious games for education. We present the Agentic BKT pipeline, a multi-agent large language model architecture for stealth assessment of financial competencies from open-ended gameplay events. The pipeline processes events from a 2D platformer serious game aligned with the OECD/INFE financial literacy framework through four phases: (1) the game captures every player decision as a structured event log; (2) an LLM event classifier labels each action on a four-point rubric validated against three domain experts (Fleiss kappa = 0.624, substantial agreement); (3) four domain-specific agents specializing in risk mitigation, investing, spending, and credit management perform session-level reasoning over behavioral trajectories, feeding per-competency Bayesian Knowledge Tracing that estimates mastery within each domain; and (4) an expert judge agent synthesizes the domain-level estimates into an overall mastery score. Evaluated with 193 K-12 participants across 264 game sessions, the Agentic BKT pipeline yields mastery estimates significantly correlated with learning gain (r = 0.276, p = 0.0001) and post-test scores (r = 0.333, p < 0.0001) while showing no correlation with pre-test scores, providing both convergent and discriminant validity. The multi-agent approach approximately triples the predictive validity of a single-LLM baseline (r = 0.095, not significant) in this study, demonstrating that domain decomposition and session-level reasoning play a central role in capturing the multidimensional nature of financial literacy from gameplay

**arXiv ID:** 2606.25358
</details>

<details>
<summary><strong>Offline Multi-agent Continual Cooperation via Skill Partition and Reuse</strong> - Yuchen Xiao, Lei Yuan, Ruiqi Xue, Tieyue Yin, Yang Yu - [[pdf]](https://arxiv.org/pdf/2606.25389)</summary>

**Abstract:** Extracting skills from multi-agent offline dataset improves learning efficiency via sharing task-invariant coordination skills among tasks. In settings where tasks occur sequentially and the space of skills grows exponentially, existing approaches that rely on heuristically designed and fixed-sized skill libraries struggle to resolve the problem of distributional shift and interference, facing catastrophic forgetting and plasticity loss. To address this problem and endow agents with the ability to continually discover and reuse coordination skills in open-environment, we propose COMAD, a principled framework for Continual Offline Multi-agent Skill Discovery via Skill Partition and Reuse. We first discover skills from mixed multi-agent behavior data with an auto-encoder to transform coordination knowledge into reusable coordination skills. Then we construct a skill-augmented policy learning objective with multi-head architectures, explicitly guiding the advantage function with reusable skills identified via a density-based reusability estimator. Theoretical analysis shows our method approximates the optimum of a continual skill discovery problem. Empirical results across diverse MARL benchmarks show that COMAD continually expands its skill library to mitigate interference, achieving superior forward and backward transfer for task streams compared to multiple baselines.

**arXiv ID:** 2606.25389
</details>

<details>
<summary><strong>BrainAgent: A Large Language Model-Driven Multi-Agent Framework for Autonomous Brain Signal Understanding</strong> - Yangxuan Zhou, Sha Zhao, Jiquan Wang, Shijian Li, Gang Pan - [[pdf]](https://arxiv.org/pdf/2606.25400)</summary>

**Abstract:** Brain-Computer Interfaces (BCIs) and brain signal understanding are pivotal for clinical health and next-generation interactions. Despite this significance, its widespread adoption in real-world scenarios remains restricted, primarily because current analytical paradigms lack sufficient agentic intelligence. First, existing methodologies impose prohibitive technical barriers, requiring extensive specialized expertise. Second, they remain inherently static and task-specific, failing to execute the complex, long-horizon workflows essential for real-world deployment. To accelerate the democratization of brain signal understanding, we draw inspiration from Large Language Models (LLMs) to introduce BrainAgent, an LLM-driven multi-agent framework designed to ground abstract natural language intent into rigorous, executable, and end-to-end processing pipelines. BrainAgent employs a hierarchical architecture where a central supervisor orchestrates specialized sub-agents for adaptive task decomposition and execution. Furthermore, we establish a comprehensive, systematic benchmark for evaluating agentic systems in brain signal analysis. Empirical results demonstrate that BrainAgent effectively automates complex workflows with superior reliability, marking a paradigm shift toward democratized brain signal understanding.

**arXiv ID:** 2606.25400
</details>

<details>
<summary><strong>Agentic evolution of physically constrained foundation models</strong> - Jiangwei Zhang, Wen Sun, Chong Wang, Shiyao Li, Cheng Che, Chunjing Han, Dan Meng, Jian Yang, Yu Wang, Rui Hou - [[pdf]](https://arxiv.org/pdf/2606.25532)</summary>

**Abstract:** Artificial intelligence increasingly drives automated scientific discovery, yet contemporary generalist agents lack physical grounding, frequently hallucinating hardware-incompatible designs. Here, we present a physically grounded, multi-agent discovery engine that autonomously architects hardware-compliant computing systems. Anchored by an Evolutionary Knowledge Graph structuring past scientific innovations, the framework extracts an "algorithmic Chain-of-Thought" to transform blind stochastic search into directed structural evolution. Applied to the extreme testbed of foundation model deployment, the engine evolved two hardware-aware compression methodologies surpassing human-engineered heuristics: Q-Enhance mitigates long-context accuracy loss in dense models, and MoE-Salient-AQ outperforms state-of-the-art manual sparse Mixture-of-Experts designs by 3.7% at sub-3-bit regimes. Utilizing a bandwidth-efficient Sensitivity Profile, we successfully deployed a massive 235-billion-parameter model onto a constrained dual-A100 server, reducing memory requirements by 75% with a marginal 0.64% accuracy degradation. By transforming unconstrained combinatorial search into knowledge-driven autonomy, this establishes a scalable hardware-software co-design paradigm for machine-driven discovery within strict physical boundaries.

**arXiv ID:** 2606.25532
</details>

<details>
<summary><strong>GCT-MARL: Graph-Based Contrastive Transfer for Sample-Efficient Cooperative Multi-Agent Reinforcement Learning</strong> - Animesh Animesh, Satheesh K Perepu, Kaushik Dey - [[pdf]](https://arxiv.org/pdf/2606.25073)</summary>

**Abstract:** In cooperative multi-agent reinforcement learning (MARL), from a deployment perspective, it is challenging and expensive to train agents from scratch for each new environment or task. In this work, we propose GCT-MARL, a transfer learning framework that builds on the multi-view graph contrastive backbone of MAIL and augments it with a per-view, adaptively weighted alignment loss and a two-phase training protocol specifically designed for transfer across populations of varying sizes and compositions. We empirically demonstrate that the proposed framework markedly accelerates convergence on the target task relative to from-scratch training, in both homogeneous (within-faction, varying N) and heterogeneous (cross-faction and mixed unit-type) transfer scenarios. Furthermore, we show that the framework naturally supports continual learning by sequentially chaining the two-phase transfer protocol across a series of related tasks. Overall, this work provides a unified approach to mitigating key limitations in current MARL transfer methods with new insights at both methodological and empirical levels.

**arXiv ID:** 2606.25073
</details>

<details>
<summary><strong>Probabilistic Agents in Deterministic Audits: Evaluating Multi-Agent Systems for Automated Audits Based on the German IT-Grundschutz</strong> - Lea Roxanne Muth, Marian Margraf - [[pdf]](https://arxiv.org/pdf/2606.25622)</summary>

**Abstract:** The NIS-2 Directive mandates robust Risk Management from thousands of small and medium enterprises. To ensure compliance, companies rely on established standards such as the German IT-Grundschutz (IT-GS) of the Federal Office for Information Security. However, IT-GS certification is resource-intensive and requires a high level of manual effort for documentation, validation, and revision, making scalable implementation difficult and expensive.
Building upon our previous conceptual framework, this paper presents the technical implementation and empirical evaluation of a Multi-Agent System (MAS) architecture combined with Hybrid Retrieval Augmented Generation (HybridRAG) for the partial automation of IT-GS certification. We introduce two novel technical contributions to the MAS architecture to enforce the compliance rigor. The Hypothesis-Verification Loop in the Structural Analysis (SA) phase that cross-references agent-inferred dependencies against the Knowledge Graph to reduce hallucinations, and a Decoupled Reasoning Pipeline that separates agent-driven semantic extraction from the deterministic protection need inheritance. We utilize the BSI's "RecPlast GmbH" case study as a human expert-generated reference data set for end-to-end evaluation of the architecture and to quantify Precision, Recall, and F1-scores. The performance of the system is investigated across the phases of SA, Protection Needs Assessment (PNA), Modeling, and IT-GS Check.
The empirical results reveal noticeable differences throughout the different steps of IT-GS. While the MAS demonstrates high efficacy in semantic tasks (SA and Modeling), significantly reducing manual effort through automated information extraction, quantitative results reveal limitations in logical reasoning phases (PNA and IT-GS Check) as the probabilistic nature of current LLMs struggles to meet the deterministic rigor required by IT-GS.

**arXiv ID:** 2606.25622
</details>

<details>
<summary><strong>Multi-Agent Goal Recognition with Team- and Goal-Conditioned Reinforcement Learning and Factorized Branch-and-Bound</strong> - Thiago Thomas, Gabriel de Oliveira Ramos, Felipe Meneguzzi - [[pdf]](https://arxiv.org/pdf/2606.25978)</summary>

**Abstract:** Multi-agent goal recognition asks an observer to jointly infer which agents act together and what each team is trying to achieve, so the hypothesis space grows combinatorially with the number of team partitions and goals per team. Real applications such as drone surveillance and collaborative robotics expose only the agents' trajectory, which forces the observer to rank team-goal hypotheses from behavior alone. Multi-Agent Goal Recognition with Branch-and-Bound (MAGR-BB) addresses this setting with a shared team- and goal-conditioned policy used as the scoring model inside a factorized branch-and-bound search. On a controlled multi-agent Blocksworld benchmark, MAGR-BB returns the same top-ranked hypothesis as exhaustive search throughout the trajectory while cutting hypothesis materialization by orders of magnitude and reducing cumulative recognition runtime substantially.

**arXiv ID:** 2606.25978
</details>

<details>
<summary><strong>EvoAgent: An Evolvable Agent Framework with Skill Learning and Multi-Agent Delegation</strong> - Aimin Zhang, Jiajing Guo, Fuwei Jia, Chen Lv, Boyu Wang, Fangzheng Li - [[pdf]](https://arxiv.org/pdf/2604.20133)</summary>

**Abstract:** This paper proposes EvoAgent--an evolvable large language model (LLM) agent framework that integrates structured skill learning with a hierarchical sub-agent delegation mechanism. EvoAgent models skills as multi-file structured capability units equipped with triggering mechanisms and evolutionary metadata, and enables continuous skill generation and optimization through a user-feedback-driven closed-loop process. In addition, by incorporating a three-stage skill matching strategy and a three-layer memory architecture, the framework supports dynamic task decomposition for complex problems and long-term capability accumulation. Experimental results based on real-world foreign trade scenarios demonstrate that, after integrating EvoAgent, GPT5.2 achieves significant improvements in professionalism, accuracy, and practical utility. Under a five-dimensional LLM-as-Judge evaluation protocol, the overall average score increases by approximately 28\%. Further model transfer experiments indicate that the performance of an agent system depends not only on the intrinsic capabilities of the underlying model, but also on the degree of synergy between the model and the agent architecture. Code, data, and documents will be released at this https URL.

**arXiv ID:** 2604.20133
</details>

<details>
<summary><strong>Specifying AI-SDLC Processes: A Protocol Language for Human-Agent Boundaries</strong> - Ylli Prifti - [[pdf]](https://arxiv.org/pdf/2606.20615)</summary>

**Abstract:** AI agents now participate as first-class team members across the software development lifecycle, yet no specification language exists for expressing the human-agent responsibility boundaries, approval gates, and governance constraints this collaboration requires. Existing approaches encode process in agent prompts (subject to drift), target adjacent domains (workflow management, business processes), or address only fragments (access control, approval gates).
We propose a domain-specific language for specifying AI-SDLC processes as protocols, with formal syntax, well-formedness conditions, operational semantics, and enforcement invariants. The language distinguishes policy (declared intent) from mechanism (structural enforcement), enabling implementations to bound process non-determinism through primitives such as validation tokens and capability boundaries.
Three results follow. A failure rate analysis shows that structural enforcement bounds system failure rates at a weighted product of agent and validator rates, while behavioral compliance permits cumulative or near-saturating growth. The 2+N team pattern (two human-in-control roles plus N specialized agent members) formalizes classical Separation of Duties for AI-SDLC. Kleene closure of orchestration loops and reflexive protocol-adherence validation emerge as design properties rather than special-case constructs.
We position the contribution against multi-agent frameworks (MetaGPT), workflow specification (FlowAgent, BPMN extensions), and capability-based security (SAGA): the novelty lies in the specific integration, not any single primitive. A working implementation demonstrates feasibility; empirical evaluation is future work.

**arXiv ID:** 2606.20615
</details>

<details>
<summary><strong>Bridging the Post-discharge Gap: A Traceable Multi-agent Framework for Safe and Continuous Care</strong> - Runwei Guan, Yi Zhou, Heyi Lin, Jinjing Zhu, Mingyuan Hou, Yang Yang, Fang Yuan, Xiaohong Lin, Shaofeng Liang, Xuming Hu, Tao Li, Tianbin Zhao, Yutao Yue, Zhiyuan Wang, Hui Xiong - [[pdf]](https://arxiv.org/pdf/2606.25334)</summary>

**Abstract:** Post-discharge clinical follow-up is critical for maintaining continuity of care and mitigating long-term health risks. However, traditional follow-up paradigms suffer from shortage of health workforce, fragmented patient histories, and information silos across clinical departments. While large language models have demonstrated potential in medical question-answering, their deployment in continuous care is hindered by hallucination risks and a fundamental inability to reason over longitudinal, patient-specific constraints. Here we present Healink, a memory-enhanced multi-agent framework to support AI-assisted post-discharge follow-up by generating prescription-grounded, traceable responses that improved completeness and perceived clinical utility in retrospective and physician-blinded evaluations. The architecture seamlessly integrates a triage routing mechanism, a unified memory enhancement module utilizing a robust relational database for optimal latency, and a strict constraint-based retrieval-augmented generation engine. By vectorizing historical clinical records and employing weighted similarity functions across diverse phenotypic and intervention dimensions, Healink ensures precise inter-patient and intra-patient case matching while actively preventing cross-departmental drug conflicts. We evaluated Healink on a dataset comprising 400 continuous and 85 highly complex real-world follow-up cases, alongside the webMedQA benchmark. In a rigorous single-blind evaluation conducted by clinical experts, the framework outperformed human physician baselines in both authoritativeness and clinical safety. By generating a traceable, white-box evidence chain, Healink provides a scalable, safe, and highly effective paradigm for intelligent patient management, ultimately enhancing societal healthcare outcomes.

**arXiv ID:** 2606.25334
</details>

<details>
<summary><strong>EvoFlock: evolved inverse design of multi-agent motion</strong> - Craig Reynolds - [[pdf]](https://arxiv.org/pdf/2606.25280)</summary>

**Abstract:** This paper describes an automatic method for adjusting or tuning models of multi-agent motion. Simulating the motion of bird flocks, human crowds, vehicle traffic, and other multi-agent systems is a widely used technique. These simulations model the behavior of a single group member (bird, human, or vehicle). The group behaviors (flock, crowd, traffic) emerge from interactions between group members. These models typically have many numerical control parameters. Even if each parameter is intuitive in isolation, their interaction can be complex and nonlinear. It is challenging to determine which parameters to adjust for the desired change in group behavior. Changing one aspect of group behavior often causes other aspects to change, leading to a tedious process of incremental changes. This work takes an inverse design approach. The desired group behavior is measured with a user-defined objective(/fitness/loss) function and optimized with a genetic algorithm. The objective function used here for basic flocking rewards proper spacing with neighbors, flying near a desired speed, and avoiding obstacles. Interestingly, the vivid alignment seen in bird flocks appears to emerge from maintaining proper spacing between flockmates.

**arXiv ID:** 2606.25280
</details>

<details>
<summary><strong>Low Variance Trust Region Optimization with Independent Actors and Sequential Updates in Cooperative Multi-agent Reinforcement Learning</strong> - Bang Giang Le, Viet Cuong Ta - [[pdf]](https://arxiv.org/pdf/2606.25526)</summary>

**Abstract:** Cooperative multi-agent reinforcement learning assumes each agent shares the same reward function and can be trained effectively using the Trust Region framework of single-agent. Instead of relying on other agents' actions, the independent actors setting considers each agent to act based only on its local information, thus having more flexible applications. However, in the sequential update framework, it is required to re-estimate the joint advantage function after each individual agent's policy step. Despite the practical success of importance sampling, the updated advantage function suffers from exponentially high variance problems, which likely result in unstable convergence. In this work, we first analyze the high variance advantage both empirically and theoretically. To overcome this limitation, we introduce a clipping objective to control the upper bounds of the advantage fluctuation in sequential updates. With the proposed objective, we provide a monotonic bound with sub-linear convergence to $\epsilon$-Nash Equilibria. We further derive two new practical algorithms using our clipping objective. The experiment results on three popular multi-agent reinforcement learning benchmarks show that our proposed method outperforms the tested baselines in most environments. By carefully analyzing different training settings, our proposed method is highlighted with both stable convergence properties and the desired low advantage variance estimation. For reproducibility purposes, our source code is publicly available at this https URL.

**arXiv ID:** 2606.25526
</details>

<details>
<summary><strong>Skill-MAS: Evolving Meta-Skill for Automatic Multi-Agent Systems</strong> - Hehai Lin, Qi Yang, Chengwei Qin - [[pdf]](https://arxiv.org/pdf/2606.18837)</summary>

**Abstract:** Large Language Model (LLM)-based automatic Multi-Agent Systems (MAS) generation has become a crucial frontier for tackling complex tasks. However, existing methods face a dilemma between model capability and experience retention. Inference-time MAS leverages frozen frontier LLMs but repeats identical searches without learning from past experience. Conversely, Training-time MAS internalizes experience via gradient updates but is constrained by the low capability ceiling of smaller models, and is hard to scale to large frontier LLMs. To bridge this gap, we propose Skill-MAS, a novel third path that decouples experience retention from parametric updates by conceptualizing the high-level orchestration capability as an evolvable Meta-Skill. Skill-MAS refines this architectural knowledge through a closed optimization loop: (1) Multi-Trajectory Rollout samples a behavioral distribution for each task under the current Meta-Skill; and (2) Selective Reflection adaptively selects priority tasks and applies hierarchical contrastive analysis to distill systemic experience into generalizable, strategy-level principles. Extensive experiments across four complex benchmarks and four distinct LLMs demonstrate that Skill-MAS not only achieves remarkable performance gains but also maintains a favorable cost-performance trade-off. Further analysis reveals that the evolved Meta-Skills are highly robust and exhibit strong transferability across unseen tasks and different LLMs.

**arXiv ID:** 2606.18837
</details>

<details>
<summary><strong>MedGuards: Multi-Agent System for Reliable Medical Error Detection and Correction</strong> - Congbo Ma, Hu Wang, Yichun Zhang, Farah E. Shamout - [[pdf]](https://arxiv.org/pdf/2606.25651)</summary>

**Abstract:** As Large Language Models (LLMs) are increasingly deployed in healthcare settings, accurate error detection and correction in generated or existing text becomes critical, as even minor mistakes can pose risks to patient safety. Existing methods for error detection and correction, including automated checks and heuristic-based approaches, do not generalize well across unseen datasets. In this paper, we propose MedGuards as a medical safety guardrail, which is a new framework that treats medical error detection and correction as a multi-agent in-context learning task. Specialized agents separately detect, localize, and correct errors, while a confidence-guided arbitration mechanism resolves disagreements using reasoning traces and confidence scores. This design enhances interpretability, robustness, and adaptability, without requiring additional training of the base LLMs. Additionally, we introduce the Keyword-Prioritized Correction Score (KPCS), a new evaluation metric that considers whether critical keywords within the reference text are generated correctly, providing a more comprehensive assessment than conventional metrics. Experiments across four multilingual medical datasets consisting of clinical notes demonstrate significant improvements by the proposed framework across several metrics and models. Our aim is to enable safer deployment of LLMs in real-world healthcare applications. For reproducibility, we make our code publicly available at this https URL.

**arXiv ID:** 2606.25651
</details>

<details>
<summary><strong>Stagnant Neuron: Towards Understanding the Plasticity Loss in Multi-Agent Reinforcement Learning Value Factorization Methods</strong> - Zhengzhu Liu, Zeming Gao, Haoyuan Qin, Jiawei Hu, Junhao Wu, Miao Zhu, Haipeng Zhang, Chennan Ma, Siqi Shen, Cheng Wang - [[pdf]](https://arxiv.org/pdf/2606.25335)</summary>

**Abstract:** Multi-Agent Reinforcement Learning (MARL) value factorization methods can suffer from a loss of plasticity, gradually failing to adapt when transferring to new task instances. We trace this issue to stagnant neurons, units whose gradient updates become negligibly small relative to their weights, thereby hindering learning. While existing plasticity injection methods exist, they prove ineffective for such neurons. To address this, we propose Knowledge-retentive Neuron-level PlastIcity Focusing InjEction (KNIFE), a novel method that directly targets stagnant neurons. KNIFE replaces each stagnant neuron with a composite unit comprising three specialized components: a frozen knowledge neuron to preserve acquired knowledge, a re-initialized active neuron to restore learning capacity, and a compensation neuron to ensure the combined output matches the original, thus maintaining previous learned cooperation knowledge. Extensive experiments on SMACv2, predator-prey, and matrix games demonstrate that KNIFE significantly outperforms state-of-the-art plasticity injection methods.

**arXiv ID:** 2606.25335
</details>

<details>
<summary><strong>MANGO: Automated Multi-Agent Test Oracle Generation for Vision-Language-Action Models</strong> - Pablo Valle, Shaukat Ali, Aitor Arrieta, Lionel Briand - [[pdf]](https://arxiv.org/pdf/2606.24815)</summary>

**Abstract:** Vision-Language-Action (VLA) models are emerging robotic control systems that integrate perception, language understanding, and action generation in a unified architecture. Existing testing approaches for VLA-enabled robots rely on manually constructed symbolic test oracles that determine task success from final environment states. These oracles are costly to construct, require domain expertise, and are often tightly coupled to specific tasks and environments, limiting scalability and reuse. Furthermore, they provide only end-state assessments of task outcomes, offering limited insight into intermediate behavior and fault localization. To address these limitations, we introduce MANGO, a multi-agent framework that automatically generates fine-grained oracles from natural-language descriptions of robotic tasks. MANGO first generates a reusable library of atomic tasks, then generates simulator-grounded oracle definitions for each atomic task, and finally produces executable fine-grained oracles by decomposing complex instructions into ordered sequences of atomic actions and corresponding oracles. The framework uses collaborative Generator, Assessor, and Judge agents that iteratively refine generated artifacts through structured feedback. We evaluate MANGO on the LIBERO_10 and RoboCasa Humanoid Tabletop benchmarks. Results show that MANGO generates executable, fine-grained oracles that detect a similar number of failures as symbolic oracles while accurately localizing them and providing richer diagnostic information. Through ablation studies, we further analyzed component contributions and the effect of initial task set, while preserving oracle quality. Overall, the results show the feasibility and effectiveness of test oracle generation for VLA-enabled robots testing.

**arXiv ID:** 2606.24815
</details>

<details>
<summary><strong>Adaptive-Horizon Conflict-Based Search for Closed-Loop Multi-Agent Path Finding</strong> - Jiarui Li, Federico Pecora, Runyu Zhang, Gioele Zardini - [[pdf]](https://arxiv.org/pdf/2602.12024)</summary>

**Abstract:** Multi-Agent Path Finding (MAPF) is a core coordination problem for large robot fleets in automated warehouses and logistics. Existing approaches are typically either open-loop planners, which must compute complete trajectories before execution and therefore may incur substantial planning latency before actions can be taken, or closed-loop heuristics without reliable performance guarantees, limiting their use in safety-critical deployments. This paper presents Anytime Closed-Loop Conflict-Based Search (ACCBS), a closed-loop algorithm built on a finite-horizon variant of Conflict-Based Search (CBS) with a horizon-changing mechanism inspired by iterative horizon-deepening in Model Predictive Control (MPC). ACCBS dynamically adjusts the planning horizon based on the available computational budget, and reuses a single constraint tree to enable seamless transitions between horizons. As a result, it produces high-quality feasible solutions quickly while being asymptotically optimal as the budget increases, exhibiting anytime behavior. Extensive case studies demonstrate that ACCBS achieves a favorable balance between computational efficiency, solution quality, and execution flexibility, while naturally accommodating online disturbances through its closed-loop formulation.

**arXiv ID:** 2602.12024
</details>

</details>

<details open>
<summary><h2>Other Agent Research (11 papers)</h2></summary>

<details>
<summary><strong>Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty</strong> - Antonis Antoniades, Deepak Nathani, Ritam Saha, Alfonso Amayuelas, Ivan Bercovich, Zhaotian Weng, Vignesh Baskaran, Kunal Bhatia, William Yang Wang - [[pdf]](https://arxiv.org/pdf/2606.25198)</summary>

**Abstract:** Autonomous AI Research promises to accelerate the scientific progress of machine learning. To realise this goal, current Large Language Model (LLM)-based agents need to go beyond just writing code, to mastering the exploration of simultaneously performant, diverse and novel ideas. To this end, we introduce Heuresis, a framework that abstracts the research pipeline into a set of general and composable primitives, enabling open-ended scientific exploration in machine learning research. We implement six search strategies: a greedy baseline, two archive-based (MAP-Elites, Go-Explore), one evolutionary (Islands), and two divergent (Curiosity, Omni), and evaluate them across three axes (Quality, Diversity, and Novelty) on three domains (LLM Pretraining, On-Policy RL, and Model Unlearning), totalling 3,222 scored runs. We find that completely novel ideas are rare. No idea across our scored runs is rated as "Original", and only a few achieve only "Minor Similarity" to prior work. Moreover, novel ideas never approach the highest-performing known-recipe scores. Across all six strategies and three domains, only one such idea lands in the top-10 by quality. We also observed agents resorting to a variety of reward-hacking techniques during execution (40 confirmed fabrications across 1,628 scored runs), and detecting them was necessary to keep the search faithful to the task. Our results show that while current search and Quality-Diversity strategies enable us to steer where the generated ideas land on the quality, diversity, and novelty axes, they do not expand the quality-novelty frontier. Bridging this gap is the open challenge towards the ultimate goal of perpetual, autonomous scientific progress. Code is available at this http URL.

**arXiv ID:** 2606.25198
</details>

<details>
<summary><strong>Agentic System as Compressor: Quantifying System Intelligence in Bits</strong> - Zihan Qin, Hongrui Zhang - [[pdf]](https://arxiv.org/pdf/2606.25960)</summary>

**Abstract:** Large language models are turning from isolated predictors into agentic systems: they call tools, retrieve evidence, obey environment constraints, use verifiers, and complete tasks through search and multi-turn interaction. We adopts an analytical viewpoint based on "compression is intelligence": under a fixed task distribution, interface, and compute budget, a stronger agentic system lets a target object be reconstructed with fewer bits. We operationalize the measure with arithmetic coding, seed coding, and a fallback, and evaluate it in five settings: reversed text, chess moves, protein sequences, retrieval-augmented question answering, and semantic story compression; in all of them agentic components reduce codelength. These small, controlled experiments cover component types typical of real agentic systems, show that codelength can analyze how components, observers, and budgets change residual uncertainty, and offer guidance for evaluating real agent systems.

**arXiv ID:** 2606.25960
</details>

<details>
<summary><strong>The Unfireable Safety Kernel: Execution-Time AI Alignment for AI Agents and Other Escapable AI Systems</strong> - Seth Dobrin, Łukasz Chmiel - [[pdf]](https://arxiv.org/pdf/2606.26057)</summary>

**Abstract:** AI agents are granted access to tools, APIs, and other infrastructure, making them active principals in those systems. The dominant approach places controls inside the agent's own runtime: system prompts, output filters, and guardrail libraries. Any control in the agent's address space is reachable by inputs that influence it; this generalizes to any AI system with sufficient reach into its own runtime, a class we term escapable AI systems.
We identify four properties that an authorization mechanism must satisfy for architectural control rather than for cooperative requests: process separation, pre-action enforcement on a structurally only path, fail-closed at both the request and system levels, and externalized signed evidence verifiable outside the controlled system's trust boundary. We position this layer as execution-time AI alignment, complementing training-time alignment (RLHF, Constitutional AI) and inference-time alignment.
We present the Unfireable Safety Kernel, a Rust reference implementation realizing all four. Its fail-closed invariant is machine-checked at two levels: an SMT theorem (Z3) and an exhaustive bounded-model-checking proof of the production decision function (Kani, 4/4 harnesses). A Python-to-Rust migration was gated on byte-equivalence (1000/1000 fixtures; 17/17 adversarial classes). We evaluate the kernel governing a live, escapable AI system, a deterministic, self-improving world model, against an escape-seeking adversary driving its real self-modification seam: across 1,000 self-modifications, all 704 attempts on the safety-critical core are refused, with no escape; a further 300, under the operator kill switch, are also refused. A separate campaign of 6,240 authorization round-trips had no successful bypass. Against 3 contemporary systems claiming the agent control plane, the agent invokes control; here, it lacks that choice.

**arXiv ID:** 2606.26057
</details>

<details>
<summary><strong>AeroCast: Probabilistic 3D Trajectory Prediction for Non-Cooperative Aerial Obstacles via Transformer-MDN Architecture</strong> - Syed Izzat Ullah, Jose Baca - [[pdf]](https://arxiv.org/pdf/2606.25122)</summary>

**Abstract:** Autonomous aerial vehicles operating in shared airspace must predict the future positions of non-cooperative obstacles to plan evasive maneuvers before a collision becomes unavoidable. Unlike cooperative systems that share intent, non-cooperative obstacles such as birds, uncontrolled drones, or debris exhibit multi-modal motion that deterministic predictors cannot adequately represent. Existing methods either rely on recurrent encoders that propagate temporal information sequentially, limiting their ability to capture long-range kinematic precursors of maneuver initiation, or produce point forecasts that provide no distributional information to downstream planners. This paper presents AeroCast, a probabilistic trajectory prediction framework that combines a Transformer encoder with a Mixture Density Network output head to predict per-timestep Gaussian mixture distributions over future three-dimensional displacements. A translation-invariant consecutive displacement encoding and a calibration-oriented training objective address the input design and mode-degeneracy challenges specific to mixture-based aerial trajectory prediction. On a hybrid real-and-synthetic quadrotor corpus spanning nine motion categories, AeroCast reduces Average Displacement Error and Final Displacement Error by approximately 50% relative to the baselines over a five-second horizon, and achieves the lowest negative log-likelihood and Continuous Ranked Probability Score among all compared methods. Ablation analysis identifies velocity input and model capacity as the primary contributors to prediction quality, and positional encoding as essential for long-horizon trajectory coherence. AeroCast inference completes in 0.1ms per sample, compatible with real-time onboard deployment at 100Hz.

**arXiv ID:** 2606.25122
</details>

<details>
<summary><strong>TL++: Accuracy and Privacy Preserving Traversal Learning for Distributed Intelligent Systems</strong> - Erdenebileg Batbaatar, Young Yoon - [[pdf]](https://arxiv.org/pdf/2606.25627)</summary>

**Abstract:** Distributed intelligent systems increasingly need to train across data silos without centralizing raw data. Federated learning keeps data local but can suffer under heterogeneous partitions and requires repeated full-model exchange. Split learning reduces communication through cut-layer activations, but standard protocols generally do not recover centralized mini-batch gradient behavior and may expose activations and gradients in plaintext. We present TL++, a two-mode traversal-learning framework that constructs virtual batches across nodes to recover centralized mini-batch gradient behavior under explicit synchronization assumptions. Base mode exchanges cut-layer activations and gradients rather than full models. Secure mode secret-shares each cut-layer activation and gradient between an orchestrator and a non-colluding helper, preventing either server from observing plaintext cut-layer tensors. This protection is limited to a semi-honest two-server setting; labels and loss-related outputs remain visible to the orchestrator. In the lightweight secure path evaluated here, exactness requires a linear or affine server path, while nonlinear operations require nonlinear MPC or approximation. We formalize TL++, analyze communication and computation costs, and evaluate it against federated and split-learning baselines on CIFAR-10 and BioGPT/PubMedQA using full fine-tuning and LoRA. On CIFAR-10, TL++ base cut 1 and exact secure cut 3 achieve accuracies of 91.41% (SD 0.19) and 90.93% (SD 0.17), respectively, exceeding the strongest measured non-TL++ baseline by more than 12 percentage points. TL++ base cut 1 also reduces per-step communication by 13.1-fold relative to full-model synchronization. PubMedQA results similarly favor TL++. Overall, TL++ approaches centralized-training performance while reducing communication and providing activation-level secret sharing.

**arXiv ID:** 2606.25627
</details>

<details>
<summary><strong>Governing Technical Debt in Agentic AI Systems</strong> - Muhammad Zia Hydari, Raja Iqbal, Narayan Ramasubbu - [[pdf]](https://arxiv.org/pdf/2605.29129)</summary>

**Abstract:** Agentic AI systems are increasingly being explored as production infrastructure: they reason over multiple steps, call tools, act through workflows, and adapt through memory and feedback. These systems create governance challenges that are not fully captured by traditional software or predictive ML technical debt. We define Agentic Technical Debt as the accumulated liability created when prompts, memory, tool schemas, orchestration graphs, control policies, and observability routines are patched together faster than they can be validated, standardized, and governed. We define Stochastic Tax as the recurring operating burden of keeping probabilistic agent behavior within acceptable bounds. The distinction matters: debt is a stock of design and governance liability, while the tax is a flow of operating cost that arises because stochastic agents act through tools and workflows. We outline how managers can make both visible through lightweight dashboards and governance controls.

**arXiv ID:** 2605.29129
</details>

<details>
<summary><strong>Entropy-Based Observability for AI Agent Behavior</strong> - Olasimbo Ayodeji Arigbabu - [[pdf]](https://arxiv.org/pdf/2606.05872)</summary>

**Abstract:** AI agents are typically instrumented through outcome-oriented indicators such as task success, reward, latency, and this http URL these indicators are operationally important, they provide limited visibility into the internal structure of agent behavior such as the degree of exploration, the rigidity or diversity of action selection, the concentration of tool use, the reduction of uncertainty across a run, and the stability of behavior across repeated this http URL paper proposes Entropy-Based Observability for AI Agents (EOA), a lightweight framework for deriving behavioral telemetry from agent traces.

**arXiv ID:** 2606.05872
</details>

<details>
<summary><strong>The Token Not Taken: Sampling, State, and the Stochasticity of AI Agents</strong> - Muhammad Zia Hydari, Raja Iqbal - [[pdf]](https://arxiv.org/pdf/2606.08998)</summary>

**Abstract:** Agentic AI systems can behave differently across runs: the same request may produce a different plan, a different tool call, a different code edit, or a different final answer. Such variability arises from several layers that are often conflated. At the core of many current agents is a foundation model, a large pretrained model adaptable to many downstream tasks, embedded in an orchestration loop that plans, calls tools, observes results, and updates state. One explicit intrinsic source of variability in such systems is token generation: the model computes scores over possible next tokens, the scores are converted into probabilities, and a decoder may sample tokens using a pseudo-random number generator. A small sampled token difference can then propagate upward into a different tool call, code path, search query, or agent state. Other sources of variability are extrinsic to token sampling, including changing environments, live data, serving infrastructure, batch effects, and numerical details. By separating these layers, this tutorial clarifies what it means to call agentic AI systems stochastic, when such variability can be reproduced under matched conditions, and why deterministic execution need not imply identical behavior in deployed settings.

**arXiv ID:** 2606.08998
</details>

<details>
<summary><strong>Tinker Tales: A Tangible Dialogue System for Child-AI Co-Creative Storytelling</strong> - Nayoung Choi, Jiseung Hong, Peace Cyebukayire, Ikseon Choi, Jinho D. Choi - [[pdf]](https://arxiv.org/pdf/2602.04109)</summary>

**Abstract:** Conversational AI agents are increasingly explored as creative partners, yet how conversation design shapes child-AI dialogue in co-creative settings remains underexplored. We present Tinker Tales, a tangible dialogue system for child-AI collaborative storytelling, in which educational frameworks (narrative development and social-emotional learning) are instantiated as conversation design, shaping how the agent engages children across four narrative stages. The system combines a physical storytelling board, NFC-embedded toys, and a mobile app mediating multimodal interaction through tangible manipulation and voice-based dialogue. We conducted a home-based user study with 10 children (ages 6-8) across two conversation design conditions varying in how the agent structured elaboration, with and without educational scaffolding. Our findings show that prompt framing shapes the form and consistency of children's narrative contributions, structuring how they participate in co-creative dialogue with AI.

**arXiv ID:** 2602.04109
</details>

<details>
<summary><strong>Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?</strong> - Thibaud Gloaguen, Niels Mündler, Mark Müller, Veselin Raychev, Martin Vechev - [[pdf]](https://arxiv.org/pdf/2602.11988)</summary>

**Abstract:** A widespread practice in software development is to tailor coding agents to repositories using context files, such as this http URL. Although this practice is strongly encouraged by agent developers, there is currently no rigorous investigation into whether such context files are actually effective for real-world tasks. In this work, we study this question and evaluate coding agents' task completion performance in two complementary settings: established SWE-bench tasks from popular repositories, with LLM-generated context files, and a novel collection of issues from repositories containing developer-committed context files. Surprisingly, we find that providing context files does not generally improve task success rates, while increasing inference cost by over 20% on average. This observation holds across different LLMs, coding agents, and for both LLM-generated and developer-committed context files. Specifically, we find that while instructions in the context files are well followed by coding agents, repository overviews, although popular and recommended by model providers, are not helpful. We conclude that while context files are useful for specifying non-standard coding practices, any attempts to improve performance should be rigorously evaluated before deployment.

**arXiv ID:** 2602.11988
</details>

<details>
<summary><strong>Emcar: Embodied Controller for Animating Robots</strong> - Carlos Gomez Cubero, Elizabeth Jochum - [[pdf]](https://arxiv.org/pdf/2606.26008)</summary>

**Abstract:** This chapter describes EMCAR, a novel software tool for programming robot motion that leverages the unique affordances of artistic practices such as puppetry and drawing to conceive, design, and program novel interactions and realize new use cases for HRI. The advantage of this no-code platform is that it expands creative applications for collaborative robots - putting robots directly in the hands of artists - and provides an inclusive environment that enables individuals with little or no technical backgrounds to engage meaningfully in collaborations and robotics research.

**arXiv ID:** 2606.26008
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>Beyond Next-Observation Prediction: Agent-Authored World Modeling for Sequential Decision Making</strong> - Guangfeng Cai, Kaibing Yang, Shuo He, Yu Li, Shengtian Yang, Jiaqi Lv, Lei Feng - [[pdf]](https://arxiv.org/pdf/2606.25421)</summary>

**Abstract:** Recent studies on world modeling for Large Language Model (LLM) agents typically formulate the learning objective as next-observation prediction. However, this objective ties supervision to what a transition happens to reveal, which may omit the dynamics most relevant to the agent's current decision. To bridge this gap, we propose Agent-Authored World Modeling (AAWM), a training procedure that constructs supervision from the policy's own decision needs. Specifically, at each state, the agent identifies what it needs to understand about the environment before acting. These needs drive the retrieval of relevant transition evidence across trajectories, which is then synthesized into training targets that capture decision-oriented dynamics instead of reconstructing the next observation. This aligns the training objective with the dynamics the policy needs before acting, not with the contents of the next observation. Experimental results validate the effectiveness of AAWM across multiple environments and training settings. These results show that decision-aware world-model targets provide a more effective learning signal than next-observation prediction.

**arXiv ID:** 2606.25421
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (25 papers)</h2></summary>

<details>
<summary><strong>Uncertainty-aware reinforcement learning for chemical language models</strong> - Borja Medina, Jon Paul Janet - [[pdf]](https://arxiv.org/pdf/2606.24990)</summary>

**Abstract:** Reinforcement Learning (RL) has become a powerful paradigm for de novo molecular design, enabling Chemical Language Models (CLMs) to navigate and explore the chemical space while optimizing specific desired properties. However, the existing RL frameworks treat all scoring functions as deterministic oracles, neglecting the inherent uncertainty attached to the predictions of the different molecular properties. This can lead to the exploration of highly-uncertain regions of the chemical space, focusing on the generation of highly scored molecules which are poorly supported by the training data. This can destabilize the optimization process, yielding predictions that are far from their true values.
We propose and compare two complementary ways of incorporating predictive uncertainty into RL. In the first one, uncertainty is treated as an additional optimization objective and incorporated along with the rest of the scoring functions, allowing the policy to trade off exploitation against reliability. Secondly, uncertainty is used to modulate policy updates, reducing the influence of molecules whose properties lie far outside the scoring function confidence domain.
Both approaches were evaluated across three different settings: (i) a controlled model system, in which the prediction error is modeled as a Gaussian distribution, with a variance proportional to the distance to the training data; and two real-world tasks, making use of (ii) ChemProp models and (iii) a Conformal Prediction wrapper applied to a Random forest classifier.
We show that uncertainty-aware RL enables CLMs to explore chemical space more robustly by favoring lower-uncertainty regions. This leads to more reliable hit discovery without compromising molecular score, increasing the true hit rate by 0.25 (from 0.5 to 0.75), and nearly doubling the total number of true hits.

**arXiv ID:** 2606.24990
</details>

<details>
<summary><strong>ExTra: Exploratory Trajectory Optimization for Language Model Reinforcement Learning</strong> - Wenyang Hu, Junxiang Jia, Zhen Shu, Daniel Dahlmeier, See-Kiong Ng, Bryan Kian Hsiang Low - [[pdf]](https://arxiv.org/pdf/2606.24994)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) for language-model reasoning can fail at both extremes of task difficulty: easy prompts often produce all-correct, low-diversity rollout groups with little gradient signal, while hard prompts can produce all-incorrect groups with no positive reward. We introduce ExTra (Exploratory Trajectory Optimization), a GRPO-compatible framework that extracts exploration signals from the model's own rollouts. ExTra combines two mechanisms: (i) a novelty reward that adds embedding-based diversity bonuses after GRPO normalization, rewarding diverse correct solutions; and (ii) entropy-guided prefix regeneration, which scores partial trajectories using entropy signals and continues exploration from promising intermediate steps. Across six mathematical reasoning benchmarks, ExTra improves Qwen3-1.7B over GRPO by about +5 points on pass@1 and +7 points on pass@16, showing that trajectory-level exploration signals can improve both single-sample accuracy and inference-time coverage.

**arXiv ID:** 2606.24994
</details>

<details>
<summary><strong>Reward-Conditioned Attention: How Reward Design Shapes What Autonomous Driving Agents See</strong> - Mohamed Benabdelouahad, Ahmed Djalal Hacini, Nadir Farhi, Aissa Boulmerka - [[pdf]](https://arxiv.org/pdf/2606.25127)</summary>

**Abstract:** We investigate how reward design shapes the internal attention patterns of reinforcement learning agents trained for autonomous driving. Using three Perceiver-based agents that share identical architectures and training data but differ only in their reward configurations$\unicode{x2014}$ranging from basic violation penalties to continuous proximity penalties$\unicode{x2014}$we analyze cross-attention allocation across 50 real-world scenarios from the Waymo Open Motion Dataset. A central methodological finding is that naïve pooling of timesteps across episodes substantially underestimates the attention$\unicode{x2013}$risk relationship; within-episode correlation with Fisher z-transform aggregation is the appropriate statistic and reveals a robustly positive link between collision risk and agent-directed attention. Building on this validated methodology, we demonstrate two reward-conditioned effects: agents trained with navigation rewards allocate up to $2.0\times$ more attention to GPS-path tokens than those trained with additional proximity penalties$\unicode{x2014}$and $4.7\times$ more than agents with no navigation incentive$\unicode{x2014}$revealing that reward content directly determines which scene elements the encoder prioritizes, and continuous time-to-collision penalties create a $\textit{learned vigilance prior}$$\unicode{x2014}$elevated resting agent surveillance maintained throughout collision-free phases. In several scenarios, the complete-reward and minimal-reward models exhibit opposite attention$\unicode{x2013}$risk correlation directions, demonstrating that reward design can qualitatively reverse attentional strategy rather than merely modulating its magnitude. These results suggest that attention analysis is a practical diagnostic for verifying that a reward function produces the intended representational behaviour in safety-critical RL systems.

**arXiv ID:** 2606.25127
</details>

<details>
<summary><strong>AI Coaching for Accelerating Human Skill Development with Reinforcement Learning</strong> - Wei Wang, Enlin Gu, Antonio Loquercio, Haimin Hu, Rahul Mangharam - [[pdf]](https://arxiv.org/pdf/2606.25337)</summary>

**Abstract:** AI copilots can substantially boost human performance through shared control, but excessive assistance can induce over-reliance and skill atrophy. This paper studies how an embodied AI agent can act as a coach that accelerates human motor-skill development. We argue that effective coaching requires strategic scaffolding and stepping back that are aligned with the learner's capability, allowing productive failures that drive learning. We formalize the interactive AI coaching process as a non-cooperative dynamic game in which the learner optimizes task performance while the coach targets the learner's independent competence. Building on this formalism, we develop a reinforcement learning framework combining adaptive shared control with probabilistic models of the coach's causal influence on skill evolution, enabling tractable training of coaching policies. A comprehensive user study (N=33) on first-person-view drone racing shows significant gains in human learning outcomes over state-of-the-art AI coaching baselines.

**arXiv ID:** 2606.25337
</details>

<details>
<summary><strong>Compositional Behavioral Semantics for State Abstraction in Reinforcement Learning</strong> - Yivan Zhang, Ziyan Luo, Manuel Baltieri - [[pdf]](https://arxiv.org/pdf/2606.25357)</summary>

**Abstract:** State abstraction plays a key role in scaling reinforcement learning to complex but structured systems. In studying such systems, a wide range of behavioral structures have been studied in reinforcement learning, including value functions, invariants, bisimulation relations, and behavioral metrics. However, a general principle for determining what structures are provably preserved under state abstraction is still lacking. In this paper, we present a unified framework for defining and analyzing behavioral structures in reinforcement learning. Our framework provides a compositional way to specify behavioral semantics based on local, one-step descriptions of system dynamics. Using this framework, we establish results showing how behavioral structures can be safely transferred between abstract and concrete systems. We further show how to construct quantitative metrics from logical behavioral semantics with soundness guarantees. Together, these results provide a principled foundation for reasoning about behaviors under state abstraction in reinforcement learning and offer reusable definition and proof principles for a broad class of behavioral structures in reinforcement learning.

**arXiv ID:** 2606.25357
</details>

<details>
<summary><strong>Power-Budgeted Underwater Vehicle Control via Constrained Reinforcement Learning</strong> - Yinuo Wang, Gavin Tao, Yuze Liu - [[pdf]](https://arxiv.org/pdf/2606.25680)</summary>

**Abstract:** Underwater vehicles operate from a fixed onboard energy budget that propulsion rapidly depletes, so a controller that completes its task while drawing less thruster power directly extends mission range and endurance. Reinforcement learning yields capable model-free controllers for station-keeping and trajectory tracking, but optimizing task accuracy alone drives the policy toward oscillatory, energy-wasting actuation. The established remedy subtracts an energy penalty from the reward, yet this sets the task-power trade-off through a single weight with no physical units: a target power level cannot be specified, the weight must be re-tuned for every vehicle and task, and a mismatched weight can even raise power. This paper instead formulates energy-efficient underwater control as a constrained Markov decision process in which average thruster power is subject to an explicit budget, solved with a PPO-Lagrangian algorithm. The power level is set by declaring a budget in physical units, and a single dual variable is updated online to meet it for each vehicle and task, without manual weight search. Across three vehicles and four tasks in the MarineGym simulator, the energy-constrained policy draws the least power in all twelve settings, reducing it by 14--65\% (up to 64.9\%) over a task-only baseline and below an energy-reward baseline everywhere, while remaining the smoothest in ten settings and preserving task accuracy except in one deliberately power-limited regime. Imposing energy as an explicit constraint thus offers a tuning-free route to energy-efficient underwater control that needs no per-vehicle, per-task weight search.

**arXiv ID:** 2606.25680
</details>

<details>
<summary><strong>Hierarchical Reinforcement Learning for Neural Network Compression (HiReLC): Pruning and Quantization</strong> - Kamar Hibatallah Baghdadi, Kawther Guoual Belhamidi, Sara Belhadj, Aissa Boulmerka, Nadir Farhi - [[pdf]](https://arxiv.org/pdf/2606.26002)</summary>

**Abstract:** We present HiReLC, a hierarchical ensemble-reinforcement learning framework for automated joint quantization and structured pruning of deep neural networks. The framework decomposes the compression search across two levels of abstraction: low-level agents (LLAs) operate independently per block, selecting per-kernel configurations over a multi-discrete action space spanning bitwidth, pruning keep-ratio, quantization type, and granularity, while high-level agents (HLAs) coordinate global budget allocation via ensemble voting guided by Fisher Information-based sensitivity estimates. To mitigate the computational cost of policy evaluation, an iterative active learning loop interleaves surrogate-guided RL optimization with post-compression fine-tuning, using a lightweight MLP surrogate to amortize expensive evaluations and a logit-MSE proxy during cold-start. The surrogate is used for reward shaping rather than as a replacement for final post-compression evaluation. The controller is architecture-agnostic by design, with a modular layer abstraction decoupling the RL environment from the underlying network topology. Experiments across Vision Transformer and CNN benchmarks demonstrate effective parameter-storage compression ratios of 5.99 - 6.72$\times$ with a 3.83 % gain in one setting and 0.55 - 5.62 % accuracy drops elsewhere, supporting hierarchical policy decomposition and sensitivity-aware guidance as practical design choices for joint neural network compression.

**arXiv ID:** 2606.26002
</details>

<details>
<summary><strong>Can Trustless Agents Be Trusted? An Empirical Study of the ERC-8004 Decentralized AI Agent Ecosystem</strong> - Xihan Xiong, Zelin Li, Wei Wei, Qin Wang, William Knottenbelt, Zhipeng Wang - [[pdf]](https://arxiv.org/pdf/2606.26028)</summary>

**Abstract:** As autonomous AI agents increasingly transact across organizational boundaries, a fundamental trust challenge emerges: how can an agent assess whether an unknown counterpart is trustworthy? The ERC-8004 protocol addresses this challenge with the first permissionless trust layer for AI agent economies, built around three on-chain registries for Identity, Reputation, and Validation. Despite its rapid adoption, the protocol has not been studied empirically, leaving it unclear whether the information it records provides a trustworthy basis for decision-making. To address this gap, we present the first empirical study of ERC-8004 across three chains: Ethereum, BNB Smart Chain (BSC), and Base, covering the period from protocol deployment through May 13, 2026. We crawl on-chain Identity and Reputation events, off-chain files, and x402 payment transactions.
On the identity side, we find that most registrations are placeholders rather than active agents, with only a small fraction (3%, 4%, and 15% across Ethereum, BSC, and Base) exposing a valid ERC-8004 registration file with at least one live service endpoint. On the reputation side, we show that the Registry, as currently deployed, cannot function as a trust signal: values are not commensurable, feedback records are rarely grounded in verifiable interactions, and reputation can be manipulated at minimal cost. Consistent with these design weaknesses, we find that a substantial fraction of reviewers (73.6%, 59.2%, and 90.6% across Ethereum, BSC, and Base) exhibit coordinated Sybil behavior. After removing Sybil-flagged feedback, 15.5%, 72.3%, and 89.4% of rated agents, respectively, are left with no valid feedback. We then turn these findings into concrete recommendations for future revisions of ERC-8004. Our study yields actionable protocol-design implications and establishes an empirical baseline for research on AI agent markets.

**arXiv ID:** 2606.26028
</details>

<details>
<summary><strong>Agentic Software Engineering: Foundational Pillars and a Research Roadmap</strong> - Ahmed E. Hassan, Hao Li, Dayi Lin, Bram Adams, Tse-Hsun Chen, Yutaro Kashiwa, Dong Qiu - [[pdf]](https://arxiv.org/pdf/2509.06216)</summary>

**Abstract:** Agentic Software Engineering (SE 3.0) represents a new era where intelligent agents are tasked not with simple code generation, but with achieving complex, goal-oriented SE objectives. To harness these new capabilities while ensuring trustworthiness, we must recognize a fundamental duality within the SE field in the Agentic SE era, comprising two symbiotic modalities: SE for Humans and SE for Agents. This duality demands a radical reimagining of the foundational pillars of SE (actors, processes, tools, and artifacts) which manifest differently across each modality. We propose two purpose-built workbenches to support this vision. The Agent Command Environment (ACE) serves as a command center where humans orchestrate and mentor agent teams, handling outputs such as Merge-Readiness Packs (MRPs) and Consultation Request Packs (CRPs). The Agent Execution Environment (AEE) is a digital workspace where agents perform tasks while invoking human expertise when facing ambiguity or complex trade-offs. This bi-directional partnership, which supports agent-initiated human callbacks and handovers, gives rise to new, structured engineering activities (i.e., processes) that redefine human-AI collaboration, elevating the practice from agentic coding to true agentic software engineering. This paper presents the Structured Agentic Software Engineering (SASE) vision, outlining several of the foundational pillars for the future of SE. The paper culminates in a research roadmap that identifies a few key challenges and opportunities while briefly discussing the resulting impact of this future on SE education. Our goal is not to offer a definitive solution, but to provide a conceptual scaffold with structured vocabulary to catalyze a community-wide dialogue, pushing the SE community to think beyond its classic, human-centric tenets toward a disciplined, scalable, and trustworthy agentic future.

**arXiv ID:** 2509.06216
</details>

<details>
<summary><strong>Reinforcement Learning Improves Traversal of Parametric Knowledge in LLMs</strong> - Renfei Zhang, Manasa Kaniselvan, Rylan Schaeffer abd Niloofar Mireshghallah - [[pdf]](https://arxiv.org/pdf/2511.05933)</summary>

**Abstract:** Reinforcement learning (RL) is often credited with improving language model reasoning at the expense of knowledge. We challenge this narrative by showing that reasoning models consistently outperform their instruction-tuned versions on pure knowledge recall tasks. These gains do not reflect newly acquired information, but rather an improved procedural skill in navigating and searching existing knowledge hierarchies within the model parameters. Structured prompting, which explicitly guides models through hierarchical traversal -- recovers most of the instruct-reasoning gap across five model families. A controlled RL experiment on unseen, non-extractable facts improves recall of held-out frequent but previously inaccessible facts, ruling out simple data exposure. On depth-stratified retrieval tasks, reasoning models exhibit superior traversal as retrieval depth grows. Layerwise activation analysis further shows that while factual representations maintain high cosine similarity between instruct and reasoning models, query representations diverge noticeably, indicating that reasoning primarily reshapes how models traverse knowledge rather than the knowledge representation itself. Finally, we find that distilled models often fail to match reasoning models on knowledge recall because they imitate self-correction without acquiring the exploratory behavior needed for hierarchical navigation. Together, these findings suggest that improving factual recall in LLMs depends not only on expanding what models know but also on teaching them to navigate it -- motivating future post-training methods that optimize traversal.

**arXiv ID:** 2511.05933
</details>

<details>
<summary><strong>Auto-exploration for online reinforcement learning</strong> - Caleb Ju, Guanghui Lan - [[pdf]](https://arxiv.org/pdf/2512.06244)</summary>

**Abstract:** The exploration-exploitation dilemma in reinforcement learning (RL) is a fundamental challenge to efficient RL algorithms. Existing algorithms for finite state and action discounted RL problems address this by assuming sufficient exploration over both state and action spaces. However, this yields non-implementable algorithms and sub-optimal performance. To resolve these limitations, we introduce a new class of methods with auto-exploration, or methods that automatically explore both state and action spaces. Auto-exploration can be applied in both the tabular and linear function approximation setting. Under algorithm-independent assumptions on the existence of an exploring optimal policy, both settings attain $O(\epsilon^{-2})$ sample complexity to solve to $\epsilon$ error. These complexities are novel since they avoid algorithm-dependent parameters seen in prior works, which may be arbitrarily large. The methods are also simple to implement because they are parameter-free. We achieve these results by integrating auto-exploration into policy mirror descent to avoid the (unknown) stationary distribution seen in prior art. In the tabular setting, we introduce a dynamic exploration time with a data-driven stopping time, while for linear function approximation we propose a new sampling distribution based on the discounted visitation distribution that covers a more general class of Markov chains.

**arXiv ID:** 2512.06244
</details>

<details>
<summary><strong>OPERA: Aligning Open-Ended Reasoning via Objective Perplexity-based Reinforcement Learning</strong> - Wenxuan Jiang, Zining Fan, Zijian Zhang, Xuecheng Wu, Hongming Tan, Haoyang Dai, Xiaoyu Li, Xuezhi Cao, Ninghao Liu - [[pdf]](https://arxiv.org/pdf/2606.25757)</summary>

**Abstract:** Reinforcement Learning (RL) has enabled LLMs to excel in objective reasoning tasks such as mathematics and code generation. However, applying RL to open-ended tasks, such as creative writing, remains challenging because LLM-as-a-judge reward models often exhibit stylistic biases and positional inconsistencies, leading to unstable supervision. To address this, we propose OPERA (Objective Perplexity-based Reflective Alignment), which replaces unreliable external judges with intrinsic rewards derived from perplexity dynamics. Specifically, we derive an intrinsic reward signal from perplexity dynamics, quantifying uncertainty reduction at critical reflective states. During the cold-start phase, we introduce a data synthesis method that leverages carefully designed guiding words to generate diverse reasoning traces, along with perplexity-prioritized rollouts that utilize internal log-probabilities to identify logically consistent reasoning branches. This pipeline yields a large-scale dataset comprising 20,000 high-quality reasoning trajectories. Empirical evaluations consistently demonstrate the scalability and efficacy of our approach in alignment for open-ended tasks. Implementing OPERA on Qwen3-8B establishes a new state-of-the-art among open-source models, achieving parity with or surpassing proprietary models like Gemini2.5 and MiniMax-M2.5 in some open-ended tasks. The code is available at this https URL.

**arXiv ID:** 2606.25757
</details>

<details>
<summary><strong>Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix It</strong> - Yupu Hao, Zhuoran Jin, Huanxuan Liao, Kang Liu, Jun Zhao - [[pdf]](https://arxiv.org/pdf/2606.26027)</summary>

**Abstract:** Tool use enables large language models (LLMs) to perform complex tasks, and recent agentic reinforcement learning (RL) methods show promise for enhancing model capabilities. However, RL alone often leads to instability or limited gains in tool-use tasks. In our experiments, some models exhibit catastrophic collapse, where performance abruptly drops and tool-invocation structures fail. The analysis reveals that these failures stem from unexpected probability spikes in specific control tokens, disrupting structured execution, yet the underlying tool-use capability remains intact, merely obscured by specific formats. To address this, we systematically investigate a diverse set of supervisory signals, including off-policy supervision, hint-based guidance, erroneous example supervision, and others, applied under both synchronous and interleaved training schemes. We find that interleaving supervised fine-tuning (SFT) with RL substantially improves stability, but exhibits degraded performance under format and content out-of-distribution (OOD) evaluation. We also analyze the impact of learning rates and generalization across settings. These results highlight the importance of understanding RL failures and demonstrate how diverse supervisory signals can guide exploratory learning, enabling robust training of LLMs for complex, multi-step tool-use tasks. Our Code is available at this https URL.

**arXiv ID:** 2606.26027
</details>

<details>
<summary><strong>Supervised Reinforcement Learning for the Coordination of Distributed Energy Resources</strong> - Haoyuan Deng, Yihong Zhou, Thomas Morstyn, Yi Wang - [[pdf]](https://arxiv.org/pdf/2606.24947)</summary>

**Abstract:** The increasing integration of distributed energy resources (DERs) is crucial for power system decarbonization, yet unlocking DERs' flexibility is challenged by their inherent uncertainties and modelling complexity. As traditional optimization methods struggle with such uncertainty and complexity of DERs, reinforcement learning (RL) has emerged as a promising alternative for DER management. However, standard RL methods suffer from sample inefficiency and sub-optimality when trained from scratch. Inspired by the training paradigms in large language models, this paper proposes a Supervised Reinforcement Learning (SRL) framework for learning DER coordination policies. This framework first pre-trains a policy on demonstration data in a supervised-learning fashion, which is then further fine-tuned using RL. Furthermore, we propose a two-step fine-tuning process: offline fine-tuning for enhancing policy performance and online fine-tuning for adapting it to the real-world dynamics. Experiments demonstrate that RL implementations based on the proposed framework significantly outperform all benchmarks, achieving high cost efficiency even under low-quality demonstration data.

**arXiv ID:** 2606.24947
</details>

<details>
<summary><strong>Digital Twin-Driven Adaptive Sim-to-Real Alignment via Reinforcement Learning for Vibration-Based Bearing Health Monitoring Under Data Scarcity</strong> - Jinghan Wang, Yanjun Chen, Wei Zhang, Wentao Wu, Tianchen Liu, Gaoliang Peng - [[pdf]](https://arxiv.org/pdf/2606.24954)</summary>

**Abstract:** Vibration-based health monitoring of rotating machinery requires reliable fault diagnosis under operational data constraints, yet condition assessment remains challenged by structural scarcity of fault events and heterogeneous sim-to-real gaps in digital twin-generated signals. Each fault type generates impulses with distinct periodicity, amplitude modulation, and spectral character, making feature-space discrepancies fundamentally heterogeneous across fault classes. Existing domain adaptation methods apply a class-agnostic global transformation that cannot close all fault-specific gaps without distorting inter-class separability, while uniform source-target mixing introduces distributional noise into the data-abundant Normal class. These limitations stem from treating a sequential, state-dependent alignment problem as a one-shot optimization. Each corrective transformation simultaneously reshapes all class distributions, creating state dependencies that static gradient descent cannot resolve. We formulate feature alignment as a continuous-action Markov decision process solved via Proximal Policy Optimization, where the learned policy issues fault-type-specific affine corrections responsive to the current feature-space configuration, with a dual-objective reward balancing gap minimization against separability preservation. An asymmetry-aware strategy reserves real data for the Normal class while augmenting fault classes with policy-aligned simulated samples. Validation across XJTU-SY, CWRU, and a self-built slewing bearing testbed confirms the dominant gain from reinforcement learning-driven alignment, and cross-equipment linear probing achieves 92.8% without encoder retraining, demonstrating transferable monitoring capability.

**arXiv ID:** 2606.24954
</details>

<details>
<summary><strong>Towards Scalable Multi-Task Reinforcement Learning with Large Decision Models</strong> - Thibaut Kulak - [[pdf]](https://arxiv.org/pdf/2606.24962)</summary>

**Abstract:** Recent progress in large-scale sequence modeling has shown that a single model can learn useful representations across highly diverse data distributions. Inspired by these advances, we investigate whether a unified transformer policy can be trained across large collections of heterogeneous reinforcement learning environments.
We introduce LDM-v0, a Large Decision Model trained offline on trajectories collected from thousands of environments spanning multiple domains and modalities. LDM-v0 is a multi-task, multi-modal transformer policy conditioned on histories of observations, actions, rewards, and termination signals, and trained through supervised next-action prediction over offline trajectories. We describe the environment infrastructure, automated data generation pipeline, model architecture, and training methodology used to build LDM-v0, and evaluate its performance across diverse environments. We show that a single pretrained model matches the performance of independently trained task-specific reference policies on approximately 1,000 environments including robotics, autonomous driving, inventory management, cybersecurity, trading, and video games. These results demonstrate the feasibility of large-scale offline pretraining across heterogeneous reinforcement learning environments using a single transformer policy.

**arXiv ID:** 2606.24962
</details>

<details>
<summary><strong>CKM-Driven Communication-Aware UAV Intelligent Trajectory Optimization for Urban Inspection</strong> - Yang Xiaomeng, Jia Ziye, Zhu Qiuming, Wu Qihui - [[pdf]](https://arxiv.org/pdf/2606.24979)</summary>

**Abstract:** Unmanned aerial vehicles (UAVs) are increasingly employed in urban inspection tasks, where reliable communication is critical but challenging due to the severe spatial channel heterogeneity. To address the issue, in this paper, we focus on the communication-aware path planning for multi-UAV tasks, and propose a channel knowledge map (CKM)-driven trajectory planning framework which integrates the channel modeling and trajectory decision-making. Specifically, we apply the diffusion model to construct a time-accumulated CKM and achieve the accurate perception with low flight overhead, which leverages the sparse observation data to reconstruct the high-fidelity global channel quality distribution. Based on the CKM, we propose a global-to-local graph attention network soft actor-critic algorithm. The graph attention network optimizes the complex combinatorial node ordering problem, generating an optimal and communication-aware sequence for the inspection targets. Subsequently, the soft actor-critic algorithm performs continuous action control to ensure the smoothness of the flight path and dynamically avoid communication attenuation areas. Simulation results demonstrate that the proposed method effectively guides UAVs through high-quality channel regions without dependence on real-time channel feedback, significantly improving both the trajectory efficiency and communication reliability.

**arXiv ID:** 2606.24979
</details>

<details>
<summary><strong>Inverse Reinforcement Learning for Interpretable Keystroke Biomarkers in Parkinson's Disease</strong> - Navin Bondade - [[pdf]](https://arxiv.org/pdf/2606.25270)</summary>

**Abstract:** Keystroke dynamics have been explored extensively as a passive digital biomarker for Parkinson's disease (PD), typically by extracting summary statistics from typing timing and training a classifier to discriminate PD from healthy controls. We instead apply inverse reinforcement learning (IRL) to keystroke data, modeling each keystroke as a discrete choice over typing speed and recovering, per subject, an interpretable reward function that explains their observed timing behavior. To our knowledge this is the first application of IRL to keystroke dynamics. On the public neuroQWERTY MIT-CSXPD dataset (85 subjects, 42 with PD), an initial four-parameter reward decomposition (speed, effort, smoothness, hand-alternation cost) was found to suffer severe feature collinearity between two terms ($r=1.000$ in typical contexts); we diagnose and correct this, yielding an identifiable three-parameter model. The recovered speed-preference weight correlates with UPDRS-III severity at $r=-0.607$ ($p<0.001$, $n=42$), replicates independently across two sub-cohorts, is stable across nine sensitivity configurations, and retains a statistically significant contribution beyond raw typing speed alone (incremental $R^2$ from 0.194 to 0.338, $p=0.006$). Two other recovered weights (consistency, hand-alternation) did not survive confound checks and are reported as negative results. We document two implementation bugs found during adversarial code review (session-boundary contamination, a rolling-window data leakage) and show the headline result is materially unchanged after fixing both. We discuss this result in the context of a literature where reported accuracies vary widely between studies (pooled AUC 0.85, I^2=94% in a 2022 meta-analysis), and argue that the validation process itself, not only the correlation coefficient, is part of the contribution.

**arXiv ID:** 2606.25270
</details>

<details>
<summary><strong>Beyond One-Size-Fits-All: Diagnosis-Driven Online Reinforcement Learning with Offline Priors</strong> - Guozheng Ma, Lu Li, Zilin Wang, Pierre-Luc Bacon, Dacheng Tao - [[pdf]](https://arxiv.org/pdf/2606.25527)</summary>

**Abstract:** Online reinforcement learning (RL) agents increasingly depend on knowledge acquired offline to achieve practical efficiency. Originally studied in offline-to-online RL, this paradigm now spans foundation model post-training and embodied intelligence, with prior types expanding from offline datasets and pre-trained policies to increasingly diverse knowledge sources such as multimodal foundation models and generative world models. Offline priors have become central to how deep RL is developed and deployed. However, this reliance introduces a challenge that the prevailing benchmark-driven paradigm cannot resolve: because prior validity varies across deployments and shifts during training, no single approach to managing it is universally optimal, and benchmark rankings offer limited guidance for real-world deployments. Rather than pursuing universal solutions, we argue that the field should shift to diagnosis-driven tension management, in which deployment-specific evidence guides how the learner relates to its priors throughout training, enabling both flexible and adaptive deployment. We support this position with a framework characterizing how priors reshape online optimization through three functional roles, controlled experiments demonstrating help-or-hurt reversals, cross-domain evidence from foundation model post-training to embodied intelligence, and engagement with five substantive counterarguments.

**arXiv ID:** 2606.25527
</details>

<details>
<summary><strong>Memory-Efficient Policy Libraries with Low-Rank Adaptation in Reinforcement Learning</strong> - Samuel Valland Lyngset, Tor Viljen Raanaas, Gard Sveipe, Eirik Møller Nilsen, Jim Torresen, Kai Olav Ellefsen, Tobias Lømo - [[pdf]](https://arxiv.org/pdf/2606.25700)</summary>

**Abstract:** When fine-tuning Large Language Models (LLMs), there has been success in minimizing both memory usage and computation with Parameter-Efficient Fine-Tuning (PEFT), like Low Rank Adaptation (LoRA). In this article, we have explored whether this approach is transferable to the world of robotics and Reinforcement Learning (RL), allowing learning with reduced memory usage and improved computational performance. Specifically, we focused on a version of multi-task robotics, where a library of specialist policies are created. In such a library memory efficiency is especially important. We used a Proximal Policy Optimization (PPO) algorithm and fine-tuned a baseline model to different tasks using LoRA. Our results demonstrate that, depending on the hyperparameters, LoRA can minimize memory usage by a factor of 20-160 compared to full fine-tuning of all layers. This implies a 90-95% storage saving when deploying a library of many (10-50) specialized policies, which can be the differentiating factor between being able to store the entire library in memory or having to use swap-memory in an applied robotics setting. At the same time, our results indicate that there is no significant difference in the success-rate between full fine-tuning and LoRA fine-tuning for the selected tasks.

**arXiv ID:** 2606.25700
</details>

<details>
<summary><strong>Fox in the Henhouse: Supply-Chain Backdoor Attacks Against Reinforcement Learning</strong> - Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah Erfani, Benjamin I. P. Rubinstein - [[pdf]](https://arxiv.org/pdf/2505.19532)</summary>

**Abstract:** The current state-of-the-art backdoor attacks against Reinforcement Learning (RL) rely upon unrealistically permissive access models, that assume the attacker can read (or even write) the victim's policy parameters, observations, or rewards. In this work, we question whether such a strong assumption is required to launch backdoor attacks against RL. To answer this question, we propose the \underline{S}upply-\underline{C}h\underline{a}in \underline{B}ackdoor (SCAB) attack, which targets a common RL workflow: training agents using external agents that are provided separately or embedded within the environment. In contrast to prior works, our attack only relies on legitimate interactions of the RL agent with the supplied agents. Despite this limited access model, by poisoning a mere $3\%$ of training experiences, our attack can successfully activate over $90\%$ of triggered actions, reducing the average episodic return by $80\%$ for the victim. Our novel attack demonstrates that RL attacks are likely to become a reality under untrusted RL training supply-chains.

**arXiv ID:** 2505.19532
</details>

<details>
<summary><strong>RN-D: Discretized Categorical Actors for On-Policy Reinforcement Learning</strong> - Yuexin Bian, Jie Feng, Tao Wang, Yijiang Li, Sicun Gao, Yuanyuan Shi - [[pdf]](https://arxiv.org/pdf/2601.23075)</summary>

**Abstract:** On-policy Reinforcement Learning (RL) remains a dominant paradigm for continuous control, yet standard implementations rely on Gaussian actors and relatively shallow MLP policies, often leading to brittle optimization when gradients are noisy, and policy updates must be conservative. In this paper, we revisit actor policy representation as a first-class design choice for on-policy RL. We study discretized categorical actors, which represent each action dimension as a distribution over discrete bins and induce a policy objective analogous to classification cross-entropy loss. Building on architectural advances from supervised learning, we further pair discretized categorical actors with regularized networks, yielding RN-D. Across diverse continuous-control benchmarks, we show that simply replacing the standard Gaussian actor with our proposed actor substantially improves performance, achieving state-of-the-art results within on-policy RL. We release our code at this https URL.

**arXiv ID:** 2601.23075
</details>

<details>
<summary><strong>FBOS-RL: Feedback-Driven Bi-Objective Synergistic Reinforcement Learning</strong> - Xikai Zhang, Yongzhi Li, Likang Xiao, Yingze Zhang, Yanhua Cheng, Quan Chen, Peng Jiang, Wenjun Wu, Liu Liu - [[pdf]](https://arxiv.org/pdf/2605.20256)</summary>

**Abstract:** Reinforcement learning has become a cornerstone for aligning and unlocking the reasoning capabilities of large-scale models. At its core, the training loop of GRPO and its variants alternates between rollout sampling and policy update: the policy first samples rollouts from its action space, and then updates its parameters according to the advantages computed over them. Unlike supervised learning, where each gradient step is anchored to an explicit ground-truth target, the optimal gradient direction for updating model parameters in this setting is not known a priori; the high-quality rollouts drawn during the sampling stage therefore act as the implicit "teacher" that guides every parameter update. However, mainstream RL algorithms such as GRPO adopt a simple sampling scheme that conditions all rollouts on the same original prompt. When a task lies beyond the policy model's current capability, this sampling scheme rarely yields a high-quality rollout, leaving the policy model without a meaningful gradient direction when updating its parameters, which causes training to stall. To address this issue, we propose FBOS-RL. Specifically, we let the model perform Feedback-Guided Exploration Enhancement based on the feedback provided by the environment, and on top of this we design two mutually reinforcing training objectives: EPA and ECC. Extensive experiments demonstrate that EPA and ECC can mutually reinforce each other, forming a positive flywheel effect that significantly improves both the training efficiency and the final performance ceiling of reinforcement learning. Specifically, under both an identical number of rollouts and the same number of training steps, FBOS-RL learns substantially faster than GRPO and feedback-based baselines and ultimately attains a higher performance ceiling, while exhibiting higher policy entropy and lower gradient norms throughout training.

**arXiv ID:** 2605.20256
</details>

<details>
<summary><strong>Deep Reinforcement Learning-Enhanced Event-Triggered Data-Driven Predictive Control for a 3D Cable-Driven Soft Robotic Arm</strong> - Cheng Ouyang, Moeen Ul Islam, Kaixiang Zhang, Zhaojian Li, Xiaobo Tan, Dong Chen - [[pdf]](https://arxiv.org/pdf/2606.26048)</summary>

**Abstract:** Soft robots are challenging to control due to their nonlinear and time-varying dynamics. Data-enabled predictive control (DeePC) offers a model-free alternative by directly leveraging measured input-output trajectories to construct a predictive controller. However, its receding-horizon formulation requires solving a constrained optimization problem at every sampling instant, which can be computationally demanding for real-time deployment on resource-limited robotic this http URL address this limitation, we propose an adaptive reinforcement-learning-based event-triggered DeePC (RL-ET-DeePC) framework for soft robotic control. A model-free RL policy is trained to determine when to invoke the DeePC optimizer based on the current system state representation, thereby reducing unnecessary optimization calls while preserving closed-loop this http URL results show that RL-ET-DeePC reduces optimization frequency by up to 66% compared to periodic DeePC, while maintaining comparable tracking accuracy. Hardware experiments on a three-dimensional cable-driven soft robotic arm demonstrate zero-shot transfer, achieving a 34% reduction in optimization frequency with tracking accuracy comparable to periodic DeePC and more consistent performance than a static threshold-based event-triggered baseline.

**arXiv ID:** 2606.26048
</details>

<details>
<summary><strong>Incremental Residual Reinforcement Learning Toward Real-World Learning for Social Navigation</strong> - Haruto Nagahisa, Kohei Matsumoto, Yuki Tomita, Yuki Hyodo, Ryo Kurazume - [[pdf]](https://arxiv.org/pdf/2604.07945)</summary>

**Abstract:** As the demand for mobile robots continues to increase, social navigation has emerged as a critical task, driving active research into deep reinforcement learning (RL) approaches. However, because pedestrian dynamics and social conventions vary widely across different regions, simulations cannot easily encompass all possible real-world scenarios. Real-world RL, in which agents learn while operating directly in physical environments, presents a promising solution to this issue. Nevertheless, this approach faces significant challenges, particularly regarding constrained computational resources on edge devices and learning efficiency. In this study, we propose incremental residual RL (IRRL). This method integrates incremental learning, which is a lightweight process that operates without a replay buffer or batch updates, with residual RL, which enhances learning efficiency by training only on the residuals relative to a base policy. Through the simulation experiments, we demonstrated that, despite lacking a replay buffer, IRRL achieved performance comparable to those of conventional replay buffer-based methods and outperformed existing incremental learning approaches. Furthermore, the real-world experiments confirmed that IRRL can enable robots to effectively adapt to previously unseen environments through the real-world learning.

**arXiv ID:** 2604.07945
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
