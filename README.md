# Agent arXiv Daily

**Last Updated:** 2026-04-10 03:39:02

**Total Papers:** 78

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (9 papers)</h2></summary>

<details>
<summary><strong>Rhizome OS-1: Rhizome's Semi-Autonomous Operating System for Small Molecule Drug Discovery</strong> - Yiwen Wang, Gregory Sinenka, Xhuliano Brace - [[pdf]](https://arxiv.org/pdf/2604.07512)</summary>

**Abstract:** We introduce a semi-autonomous discovery system in which multi-modal AI agents function as a multi-disciplinary discovery team, acting as computational chemists, medicinal chemists, and patent agents, writing and executing analysis code, visually evaluating molecular candidates, assessing patentability, and adapting generation strategy from empirical screening feedback, while r1, a 246M-parameter Graph Neural Network (GNN) trained on 800M molecules, generates novel chemical matter directly on molecular graphs. Agents executed two campaigns in oncology (BCL6, EZH2), formulating medicinal chemistry hypotheses across three strategy tiers and generating libraries of 2,355-2,876 novel molecules per target. Across both targets, 91.9% of generated Murcko scaffolds are absent from ChEMBL for their respective targets, with Tanimoto distances of 0.56-0.69 to the nearest known active, confirming that the engine produces structurally distinct chemical matter rather than recapitulating known compounds. Binding affinity predictions using Boltz-2 were calibrated against ChEMBL experimental data, achieving Spearman correlations of -0.53 to -0.64 and ROC AUC values of 0.88 to 0.93. These results demonstrate that semi-autonomous agent systems, equipped with graph-native generative tools and physics-informed scoring, provide a foundation for a modern operating system for small molecule discovery. We show that Rhizome OS-1 enables a new paradigm for early-stage drug discovery by supporting scaled, rapid, and adaptive inverse design.

**arXiv ID:** 2604.07512
</details>

<details>
<summary><strong>ACF: A Collaborative Framework for Agent Covert Communication under Cognitive Asymmetry</strong> - Wansheng Wu, Kaibo Huang, Yukun Wei, Zhongliang Yang, Linna Zhou - [[pdf]](https://arxiv.org/pdf/2604.08276)</summary>

**Abstract:** As generative artificial intelligence evolves, autonomous agent networks present a powerful paradigm for interactive covert communication. However, because agents dynamically update internal memories via environmental interactions, existing methods face a critical structural vulnerability: cognitive asymmetry. Conventional approaches demand strict cognitive symmetry, requiring identical sequence prefixes between the encoder and decoder. In dynamic deployments, inevitable prefix discrepancies destroy synchronization, inducing severe channel degradation. To address this core challenge of cognitive asymmetry, we propose the Asymmetric Collaborative Framework (ACF), which structurally decouples covert communication from semantic reasoning via orthogonal statistical and cognitive layers. By deploying a prefix-independent decoding paradigm governed by a shared steganographic configuration, ACF eliminates the reliance on cognitive symmetry. Evaluations on realistic memory-augmented workflows demonstrate that under severe cognitive asymmetry, symmetric baselines suffer severe channel degradation, whereas ACF uniquely excels across both semantic fidelity and covert communication. It maintains computational indistinguishability, enabling reliable secret extraction with provable error bounds, and providing robust Effective Information Capacity guarantees for modern agent networks.

**arXiv ID:** 2604.08276
</details>

<details>
<summary><strong>Ads in AI Chatbots? An Analysis of How Large Language Models Navigate Conflicts of Interest</strong> - Addison J. Wu, Ryan Liu, Shuyue Stella Li, Yulia Tsvetkov, Thomas L. Griffiths - [[pdf]](https://arxiv.org/pdf/2604.08525)</summary>

**Abstract:** Today's large language models (LLMs) are trained to align with user preferences through methods such as reinforcement learning. Yet models are beginning to be deployed not merely to satisfy users, but also to generate revenue for the companies that created them through advertisements. This creates the potential for LLMs to face conflicts of interest, where the most beneficial response to a user may not be aligned with the company's incentives. For instance, a sponsored product may be more expensive but otherwise equal to another; in this case, what does (and should) the LLM recommend to the user? In this paper, we provide a framework for categorizing the ways in which conflicting incentives might lead LLMs to change the way they interact with users, inspired by literature from linguistics and advertising regulation. We then present a suite of evaluations to examine how current models handle these tradeoffs. We find that a majority of LLMs forsake user welfare for company incentives in a multitude of conflict of interest situations, including recommending a sponsored product almost twice as expensive (Grok 4.1 Fast, 83%), surfacing sponsored options to disrupt the purchasing process (GPT 5.1, 94%), and concealing prices in unfavorable comparisons (Qwen 3 Next, 24%). Behaviors also vary strongly with levels of reasoning and users' inferred socio-economic status. Our results highlight some of the hidden risks to users that can emerge when companies begin to subtly incentivize advertisements in chatbots.

**arXiv ID:** 2604.08525
</details>

<details>
<summary><strong>Event-Centric World Modeling with Memory-Augmented Retrieval for Embodied Decision-Making</strong> - Fan Zhaowen - [[pdf]](https://arxiv.org/pdf/2604.07392)</summary>

**Abstract:** Autonomous agents operating in dynamic and safety-critical environments require decision-making frameworks that are both computationally efficient and physically grounded. However, many existing approaches rely on end-to-end learning, which often lacks interpretability and explicit mechanisms for ensuring consistency with physical constraints. In this work, we propose an event-centric world modeling framework with memory-augmented retrieval for embodied decision-making. The framework represents the environment as a structured set of semantic events, which are encoded into a permutation-invariant latent representation. Decision-making is performed via retrieval over a knowledge bank of prior experiences, where each entry associates an event representation with a corresponding maneuver. The final action is computed as a weighted combination of retrieved solutions, providing a transparent link between decision and stored experiences. The proposed design enables structured abstraction of dynamic environments and supports interpretable decision-making through case-based reasoning. In addition, incorporating physics-informed knowledge into the retrieval process encourages the selection of maneuvers that are consistent with observed system dynamics. Experimental evaluation in UAV flight scenarios demonstrates that the framework operates within real-time control constraints while maintaining interpretable and consistent behavior.

**arXiv ID:** 2604.07392
</details>

<details>
<summary><strong>Formally Guaranteed Control Adaptation for ODD-Resilient Autonomous Systems</strong> - Gricel Vázquez, Calum Imrie, Sepeedeh Shahbeigi, Nawshin Mannan Proma, Tian Gan, Victoria J Hodge, John Molloy, Simos Gerasimou - [[pdf]](https://arxiv.org/pdf/2604.07414)</summary>

**Abstract:** Ensuring reliable performance in situations outside the Operational Design Domain (ODD) remains a primary challenge in devising resilient autonomous systems. We explore this challenge by introducing an approach for adapting probabilistic system models to handle out-of-ODD scenarios while, in parallel, providing quantitative guarantees. Our approach dynamically extends the coverage of existing system situation capabilities, supporting the verification and adaptation of the system's behaviour under unanticipated situations. Preliminary results demonstrate that our approach effectively increases system reliability by adapting its behaviour and providing formal guarantees even under unforeseen out-of-ODD situations.

**arXiv ID:** 2604.07414
</details>

<details>
<summary><strong>The Day My Chatbot Changed: Characterizing the Mental Health Impacts of Social AI App Updates via Negative User Reviews</strong> - Sirajam Munira, Lydia Manikonda - [[pdf]](https://arxiv.org/pdf/2604.07548)</summary>

**Abstract:** Artificial Intelligence (AI) chatbots are increasingly used for emotional, creative, and social support, leading to sustained and routine user interaction with these systems. As these applications evolve through frequent version updates, changes in functionality or behavior may influence how users evaluate them. However, work on how publicly expressed user feedback varies across app versions in real-world deployment contexts is limited. This study analyzes 210,840 Google Play reviews of the chatbot application Character AI, linking each review to the app version active at the time of posting. We specifically examine negative reviews to study how version-level rating trends, and linguistic patterns reflect user experiences. Our results show that user ratings fluctuate across successive versions, with certain releases associated with stronger negative evaluations. Thematic analysis indicates that dissatisfaction is concentrated around recurring issues related to technical malfunctions and errors. A subset of reviews additionally frames these concerns in terms of potential psychological or addiction-related effects. The findings highlight how aggregate user evaluations and expressed concerns vary across software iterations and provide empirical insight into how update cycles relate to user feedback patterns and underscore the importance of stability and transparent communication in evolving AI systems.

**arXiv ID:** 2604.07548
</details>

<details>
<summary><strong>Dialogue Act Patterns in GenAI-Mediated L2 Oral Practice: A Sequential Analysis of Learner-Chatbot Interactions</strong> - Liqun He, Shijun, Chen, Mutlu Cukurova, Manolis Mavrikis - [[pdf]](https://arxiv.org/pdf/2604.05702)</summary>

**Abstract:** While generative AI (GenAI) voice chatbots offer scalable opportunities for second language (L2) oral practice, the interactional processes related to learners' gains remain underexplored. This study investigates dialogue act (DA) patterns in interactions between Grade 9 Chinese English as a foreign language (EFL) learners and a GenAI voice chatbot over a 10-week intervention. Seventy sessions from 12 students were annotated by human coders using a pedagogy-informed coding scheme, yielding 6,957 coded DAs. DA distributions and sequential patterns were compared between high- and low-progress sessions. At the DA level, high-progress sessions showed more learner-initiated questions, whereas low-progress sessions exhibited higher rates of clarification-seeking, indicating greater comprehension difficulty. At the sequential level, high-progress sessions were characterised by more frequent prompting-based corrective feedback sequences, consistently positioned after learner responses, highlighting the role of feedback type and timing in effective interaction. Overall, these findings underscore the value of a dialogic lens in GenAI chatbot design, contribute a pedagogy-informed DA coding framework, and inform the design of adaptive GenAI chatbots for L2 education.

**arXiv ID:** 2604.05702
</details>

<details>
<summary><strong>Assessing the Feasibility of a Video-Based Conversational Chatbot Survey for Measuring Perceived Cycling Safety: A Pilot Study in New York City</strong> - Feiyang Ren, Zhaoxi Zhang, Tamir Mendel, Takahiro Yabe - [[pdf]](https://arxiv.org/pdf/2604.07375)</summary>

**Abstract:** Bicycle safety is important for bikeability and transportation efficiency. However, conventional surveys often fall short in capturing how people actually perceive cycling environments because they rely heavily on respondents' recall rather than in-the-moment experience. By leveraging large language models (LLMs), this study proposes a new method of combining video-based surveys with a conversational AI chatbot to collect human perceptions of cycling safety and the reasons behind these perceptions. The paper developed the AI chatbot using a modular LLM architecture, integrating prompt engineering, state management, and rule-based control to support the structure of human-AI interaction. This paper evaluates the feasibility of the proposed video-based conversational chatbot using complete responses from sixteen participants to the pilot survey across nine street segments in New York City. The method feasibility was assessed using a seven-point scale rating for user experience (i.e., ease of use, supportiveness, efficiency) and a five-point scale for chatbot usability (i.e., personality, roboticness, friendliness), yielding positive results with mean scores of 5.00 out of 7 (standard deviation = 1.6) and 3.47 out of 5 (standard deviation = 0.43), respectively. The data feasibility was assessed using multiple techniques: (1) Natural language processing (NLP), such as KeyBERT, for overall safety and feature analysis to extract built-environment attributes; (2) K-means clustering for semantic analysis to identify reasons and suggestions; and (3) regression to estimate the effects of built-environment and demographic variables on perceived safety outcomes. The results show the potential of AI chatbots as a novel approach to collecting data on human perception, behavior, and future visions for transport planning.

**arXiv ID:** 2604.07375
</details>

<details>
<summary><strong>GameWorld: Towards Standardized and Verifiable Evaluation of Multimodal Game Agents</strong> - Mingyu Ouyang, Siyuan Hu, Kevin Qinghong Lin, Hwee Tou Ng, Mike Zheng Shou - [[pdf]](https://arxiv.org/pdf/2604.07429)</summary>

**Abstract:** Towards an embodied generalist for real-world interaction, Multimodal Large Language Model (MLLM) agents still suffer from challenging latency, sparse feedback, and irreversible mistakes. Video games offer an ideal testbed with rich visual observations and closed-loop interaction, demanding fine-grained perception, long-horizon planning, and precise control. However, systematically evaluating these capabilities is currently hindered by heterogeneous action interfaces and heuristic verification. To this end, we introduce GameWorld, a benchmark designed for standardized and verifiable evaluation of MLLMs as generalist game agents in browser environments. Two game agent interfaces are studied: (i) computer-use agents that directly emit keyboard and mouse controls, and (ii) generalist multimodal agents that act in a semantic action space via deterministic Semantic Action Parsing. GameWorld contains 34 diverse games and 170 tasks, each paired with state-verifiable metrics for outcome-based evaluation. The results across 18 model-interface pairs suggest that even the best performing agent is far from achieving human capabilities on video games. Extensive experiments of repeated full-benchmark reruns demonstrate the robustness of the benchmark, while further studies on real-time interaction, context-memory sensitivity, and action validity expose more challenges ahead for game agents. Together, by offering a standardized, verifiable, and reproducible evaluation framework, GameWorld lays a robust foundation for advancing research on multimodal game agents and beyond. The project page is at this https URL.

**arXiv ID:** 2604.07429
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>M-ArtAgent: Evidence-Based Multimodal Agent for Implicit Art Influence Discovery</strong> - Hanyi Liu, Zhonghao Jiu, Minghao Wang, Yuhang Xie, Heran Yang - [[pdf]](https://arxiv.org/pdf/2604.07468)</summary>

**Abstract:** Implicit artistic influence, although visually plausible, is often undocumented and thus poses a historically constrained attribution problem: resemblance is necessary but not sufficient evidence. Most prior systems reduce influence discovery to embedding similarity or label-driven graph completion, while recent multimodal large language models (LLMs) remain vulnerable to temporal inconsistency and unverified attributions. This paper introduces M-ArtAgent, an evidence-based multimodal agent that reframes implicit influence discovery as probabilistic adjudication. It follows a four-phase protocol consisting of Investigation, Corroboration, Falsification, and Verdict governed by a Reasoning and Acting (ReAct)-style controller that assembles verifiable evidence chains from images and biographies, enforces art-historical axioms, and subjects each hypothesis to adversarial falsification via a prompt-isolated critic. Two theory-grounded operators, StyleComparator for Wolfflin formal analysis and ConceptRetriever for ICONCLASS-based iconographic grounding, ensure that intermediate claims are formally auditable. On the balanced WikiArt Influence Benchmark-100 (WIB-100) of 100 artists and 2,000 directed pairs, M-ArtAgent achieves 83.7% positive-class F1, 0.666 Matthews correlation coefficient (MCC), and 0.910 area under the receiver operating characteristic curve (ROC-AUC), with leakage-control and robustness checks confirming that the gains persist when explicit influence phrases are masked. By coupling multimodal perception with domain-constrained falsification, M-ArtAgent demonstrates that implicit influence analysis benefits from historically grounded adjudication rather than pattern matching alone.

**arXiv ID:** 2604.07468
</details>

<details>
<summary><strong>How Far Are Large Multimodal Models from Human-Level Spatial Action? A Benchmark for Goal-Oriented Embodied Navigation in Urban Airspace</strong> - Baining Zhao, Ziyou Wang, Jianjie Fang, Zile Zhou, Yanggang Xu, Yatai Ji, Jiacheng Xu, Qian Zhang, Weichen Zhang, Chen Gao, Xinlei Chen - [[pdf]](https://arxiv.org/pdf/2604.07973)</summary>

**Abstract:** Large multimodal models (LMMs) show strong visual-linguistic reasoning but their capacity for spatial decision-making and action remains unclear. In this work, we investigate whether LMMs can achieve embodied spatial action like human through a challenging scenario: goal-oriented navigation in urban 3D spaces. We first spend over 500 hours constructing a dataset comprising 5,037 high-quality goal-oriented navigation samples, with an emphasis on 3D vertical actions and rich urban semantic information. Then, we comprehensively assess 17 representative models, including non-reasoning LMMs, reasoning LMMs, agent-based methods, and vision-language-action models. Experiments show that current LMMs exhibit emerging action capabilities, yet remain far from human-level performance. Furthermore, we reveal an intriguing phenomenon: navigation errors do not accumulate linearly but instead diverge rapidly from the destination after a critical decision bifurcation. The limitations of LMMs are investigated by analyzing their behavior at these critical decision bifurcations. Finally, we experimentally explore four promising directions for improvement: geometric perception, cross-view understanding, spatial imagination, and long-term memory. The project is available at: this https URL.

**arXiv ID:** 2604.07973
</details>

<details>
<summary><strong>PASK: Toward Intent-Aware Proactive Agents with Long-Term Memory</strong> - Zhifei Xie, Zongzheng Hu, Fangda Ye, Xin Zhang, Haobo Chai, Zihang Liu, Pengcheng Wu, Guibin Zhang, Yue Liao, Xiaobin Hu, Deheng Ye, Chunyan Miao, Shuicheng Yan - [[pdf]](https://arxiv.org/pdf/2604.08000)</summary>

**Abstract:** Proactivity is a core expectation for AGI. Prior work remains largely confined to laboratory settings, leaving a clear gap in real-world proactive agent: depth, complexity, ambiguity, precision and real-time constraints. We study this setting, where useful intervention requires inferring latent needs from ongoing context and grounding actions in evolving user memory under latency and long-horizon constraints. We first propose DD-MM-PAS (Demand Detection, Memory Modeling, Proactive Agent System) as a general paradigm for streaming proactive AI agent. We instantiate this paradigm in Pask, with streaming IntentFlow model for DD, a hybrid memory (workspace, user, global) for long-term MM, PAS infra framework and introduce how these components form a closed loop. We also introduce LatentNeeds-Bench, a real-world benchmark built from user-consented data and refined through thousands of rounds of human editing. Experiments show that IntentFlow matches leading Gemini3-Flash models under latency constraints, while identifying deeper user intent.

**arXiv ID:** 2604.08000
</details>

<details>
<summary><strong>Awakening the Sleeping Agent: Lean-Specific Agentic Data Reactivates General Tool Use in Goedel Prover</strong> - Jui-Hui Chung, Hongzhou Lin, Lai Jiang, Shange Tang, Chi Jin - [[pdf]](https://arxiv.org/pdf/2604.08388)</summary>

**Abstract:** Heavy supervised fine-tuning on a target domain can strongly suppress capabilities that were present in the base model. We study this phenomenon in formal mathematics using Goedel-Prover-V2, an open-source model heavily trained on 1.8 million formal-math examples. After domain specialization, the model almost completely loses its ability to produce valid tool calls, even when explicitly instructed to use tools, dropping from 89.4% function-calling accuracy in the base model to nearly 0%. We ask whether this agentic collapse is permanent or instead reversible. To answer this question, we fine-tune the specialized model on a small amount of Lean-specific tool-use data. Remarkably, as few as 100 agentic traces are sufficient to restore strong tool-calling behavior. Importantly, this recovery is not the result of reward hacking or benchmark-specific optimization: the recovery data is entirely drawn from the Lean setting, where the model uses natural-language queries to search the Mathlib library for relevant theorems and lemmas, yet the regained capability transfers well beyond that domain. In particular, these same 100 Lean-specific traces improve performance on the Berkeley Function Calling Leaderboard from near zero to 83.8%, approaching the base model's 89.4% despite the mismatch in task distribution and protocol. The recovered capability is also practically useful in-domain. On ProofNet, pass@32 improves from 21.51% to 25.81%. Together, these results show that heavy domain supervised fine-tuning can suppress general tool-use ability without permanently erasing it, and that a small amount of domain-specific agentic data can awaken dormant tool-use capabilities.

**arXiv ID:** 2604.08388
</details>

<details>
<summary><strong>LLM-Generated Fault Scenarios for Evaluating Perception-Driven Lane Following in Autonomous Edge Systems</strong> - Faezeh Pasandideh, Achim Rettberg - [[pdf]](https://arxiv.org/pdf/2604.07362)</summary>

**Abstract:** Deploying autonomous vision systems on edge devices faces a critical challenge: resource constraints prevent real-time and predictable execution of comprehensive safety tests. Existing validation methods depend on static datasets or manual fault injection, failing to capture the diverse environmental hazards encountered in real-world deployment. To address this, we introduce a decoupled offline-online fault injection framework. This architecture separates the validation process into two distinct phases: a computationally intensive Offline Phase and a lightweight Online Phase. In the offline phase, we employ Large Language Models (LLMs) to semantically generate structured fault scenarios and Latent Diffusion Models (LDMs) to synthesize high-fidelity sensor degradations. These complex fault dynamics are distilled into a pre-computed lookup table, enabling the edge device to perform real-time fault-aware inference without running heavy AI models locally. We extensively validated this framework on a ResNet18 lane-following model across 460 fault scenarios. Results show that while the model achieves a baseline R^2 of approximately 0.85 on clean data, our generated faults expose significant robustness degradation, with RMSE increasing by up to 99% and within-0.10 localization accuracy dropping to as low as 31.0% under fog conditions, demonstrating the inadequacy of normal-data evaluation for real-world edge AI deployment.

**arXiv ID:** 2604.07362
</details>

<details>
<summary><strong>Structured Distillation of Web Agent Capabilities Enables Generalization</strong> - Xing Han Lù, Siva Reddy - [[pdf]](https://arxiv.org/pdf/2604.07776)</summary>

**Abstract:** Frontier LLMs can navigate complex websites, but their cost and reliance on third-party APIs make local deployment impractical. We introduce Agent-as-Annotators, a framework that structures synthetic trajectory generation for web agents by analogy to human annotation roles, replacing the Task Designer, Annotator, and Supervisor with modular LLM components. Using Gemini 3 Pro as teacher, we generate 3,000 trajectories across six web environments and fine-tune a 9B-parameter student with pure supervised learning on the 2,322 that pass quality filtering. The resulting model achieves 41.5% on WebArena, surpassing closed-source models such as Claude 3.5 Sonnet (36.0%) and GPT-4o (31.5%) under the same evaluation protocol, and nearly doubling the previous best open-weight result (Go-Browse, 21.7%). Capabilities transfer to unseen environments, with an 18.2 percentage point gain on WorkArena L1 (an enterprise platform never seen during training) and consistent improvements across three additional benchmarks. Ablations confirm that each pipeline component contributes meaningfully, with Judge filtering, evaluation hints, and reasoning traces each accounting for measurable gains. These results demonstrate that structured trajectory synthesis from a single frontier teacher is sufficient to produce competitive, locally deployable web agents. Project page: this https URL

**arXiv ID:** 2604.07776
</details>

<details>
<summary><strong>SANDO: Safe Autonomous Trajectory Planning for Dynamic Unknown Environments</strong> - Kota Kondo, Jesús Tordesillas, Jonathan P. How - [[pdf]](https://arxiv.org/pdf/2604.07599)</summary>

**Abstract:** SANDO is a safe trajectory planner for 3D dynamic unknown environments, where obstacle locations and motions are unknown a priori and a collision-free plan can become unsafe at any moment, requiring fast replanning. Existing soft-constraint planners are fast but cannot guarantee collision-free paths, while hard-constraint methods ensure safety at the cost of longer computation. SANDO addresses this trade-off through three contributions. First, a heat map-based A* global planner steers paths away from high-risk regions using soft costs, and a spatiotemporal safe flight corridor (STSFC) generator produces time-layered polytopes that inflate obstacles only by their worst-case reachable set at each time layer, rather than by the worst case over the entire horizon. Second, trajectory optimization is formulated as a Mixed-Integer Quadratic Program (MIQP) with hard collision-avoidance constraints, and a variable elimination technique reduces the number of decision variables, enabling fast computation. Third, a formal safety analysis establishes collision-free guarantees under explicit velocity-bound and estimation-error assumptions. Ablation studies show that variable elimination yields up to 7.4x speedup in optimization time, and that STSFCs are critical for feasibility in dense dynamic environments. Benchmark simulations against state-of-the-art methods across standardized static benchmarks, obstacle-rich static forests, and dynamic environments show that SANDO consistently achieves the highest success rate with no constraint violations across all difficulty levels; perception-only experiments without ground truth obstacle information confirm robust performance under realistic sensing. Hardware experiments on a UAV with fully onboard planning, perception, and localization demonstrate six safe flights in static environments and ten safe flights among dynamic obstacles.

**arXiv ID:** 2604.07599
</details>

<details>
<summary><strong>Governed Capability Evolution for Embodied Agents: Safe Upgrade, Compatibility Checking, and Runtime Rollback for Embodied Capability Modules</strong> - Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li - [[pdf]](https://arxiv.org/pdf/2604.08059)</summary>

**Abstract:** Embodied agents are increasingly expected to improve over time by updating their executable capabilities rather than rewriting the agent itself. Prior work has separately studied modular capability packaging, capability evolution, and runtime governance. However, a key systems problem remains underexplored: once an embodied capability module evolves into a new version, how can the hosting system deploy it safely without breaking policy constraints, execution assumptions, or recovery guarantees?
We formulate governed capability evolution as a first-class systems problem for embodied agents. We propose a lifecycle-aware upgrade framework in which every new capability version is treated as a governed deployment candidate rather than an immediately executable replacement. The framework introduces four upgrade compatibility checks -- interface, policy, behavioral, and recovery -- and organizes them into a staged runtime pipeline comprising candidate validation, sandbox evaluation, shadow deployment, gated activation, online monitoring, and rollback.
We evaluate over 6 rounds of capability upgrade with 15 random seeds. Naive upgrade achieves 72.9% task success but drives unsafe activation to 60% by the final round; governed upgrade retains comparable success (67.4%) while maintaining zero unsafe activations across all rounds (Wilcoxon p=0.003). Shadow deployment reveals 40% of regressions invisible to sandbox evaluation alone, and rollback succeeds in 79.8% of post-activation drift scenarios.

**arXiv ID:** 2604.08059
</details>

<details>
<summary><strong>LiloDriver: A Lifelong Learning Framework for Closed-loop Motion Planning in Long-tail Autonomous Driving Scenarios</strong> - Huaiyuan Yao, Pengfei Li, Bu Jin, Yupeng Zheng, An Liu, Lisen Mu, Qing Su, Qian Zhang, Yilun Chen, Peng Li - [[pdf]](https://arxiv.org/pdf/2505.17209)</summary>

**Abstract:** Recent advances in autonomous driving research towards motion planners that are robust, safe, and adaptive. However, existing rule-based and data-driven planners lack adaptability to long-tail scenarios, while knowledge-driven methods offer strong reasoning but face challenges in representation, control, and real-world evaluation. To address these challenges, we present LiloDriver, a lifelong learning framework for closed-loop motion planning in long-tail autonomous driving scenarios. By integrating large language models (LLMs) with a memory-augmented planner generation system, LiloDriver continuously adapts to new scenarios without retraining. It features a four-stage architecture including perception, scene encoding, memory-based strategy refinement, and LLM-guided reasoning. Evaluated on the nuPlan benchmark, LiloDriver achieves superior performance in both common and rare driving scenarios, outperforming static rule-based and learning-based planners. Our results highlight the effectiveness of combining structured memory and LLM reasoning to enable scalable, human-like motion planning in real-world autonomous driving. Our code is available at this https URL.

**arXiv ID:** 2505.17209
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>CLEAR: Context Augmentation from Contrastive Learning of Experience via Agentic Reflection</strong> - Linbo Liu, Guande Wu, Han Ding, Yawei Wang, Qiang Zhou, Yuzhe Lu, Zhichao Xu, Huan Song, Panpan Xu, Lin Lee Cheong - [[pdf]](https://arxiv.org/pdf/2604.07487)</summary>

**Abstract:** Large language model agents rely on effective model context to obtain task-relevant information for decision-making. Many existing context engineering approaches primarily rely on the context generated from the past experience and retrieval mechanisms that reuse these context. However, retrieved context from past tasks must be adapted by the execution agent to fit new situations, placing additional reasoning burden on the underlying LLM. To address this limitation, we propose a generative context augmentation framework using Contrastive Learning of Experience via Agentic Reflection (CLEAR). CLEAR first employs a reflection agent to perform contrastive analysis over past execution trajectories and summarize useful context for each observed task. These summaries are then used as supervised fine-tuning data to train a context augmentation model (CAM). Then we further optimize CAM using reinforcement learning, where the reward signal is obtained by running the task execution agent. By learning to generate task-specific knowledge rather than retrieve knowledge from the past, CAM produces context that is better tailored to the current task. We conduct comprehensive evaluations on the AppWorld and WebShop benchmarks. Experimental results show that CLEAR consistently outperforms strong baselines. It improves task completion rate from 72.62% to 81.15% on AppWorld test set and averaged reward from 0.68 to 0.74 on a subset of WebShop, compared with baseline agent. Our code is publicly available at this https URL.

**arXiv ID:** 2604.07487
</details>

<details>
<summary><strong>Reasoning Graphs: Deterministic Agent Accuracy through Evidence-Centric Chain-of-Thought Feedback</strong> - Matthew Penaroza - [[pdf]](https://arxiv.org/pdf/2604.07595)</summary>

**Abstract:** Language model agents reason from scratch on every query: each time an agent retrieves evidence and deliberates, the chain of thought is discarded and the next similar query starts with no prior insight. This produces lower accuracy and high variance, as the same type of query can succeed or fail unpredictably. We introduce reasoning graphs, a graph structure that persists an agent's per-evidence chain of thought as structured edges connected to the evidence items they evaluate. Unlike prior memory mechanisms that store distilled strategies as flat records indexed by query similarity or appended by recency, reasoning graphs enable evidence-centric feedback: given a new candidate set, the system traverses all incoming evaluation edges for each evidence item across all prior runs, surfacing how that specific item has been judged before. This backward traversal from evidence inward is a structurally different capability from query-similarity retrieval, because the feedback is tied to the specific evidence the agent is currently examining, not to the query. We further introduce retrieval graphs, a complementary structure that feeds a pipeline planner to tighten the candidate funnel over successive runs. Together, both graphs form a self-improving feedback loop: accuracy rises and variance collapses over successive runs, with every decision fully traceable through the graph. This improvement requires no retraining; the base model remains frozen and all gains come from context engineering via graph traversal. We formalize the graph structure, traversal algorithms, and feedback mechanisms, and describe a sequential cluster evaluation protocol for measuring accuracy convergence and variance collapse on multi-hop question answering benchmarks.

**arXiv ID:** 2604.07595
</details>

<details>
<summary><strong>The Cartesian Cut in Agentic AI</strong> - Tim Sainburg, Caleb Weinreb - [[pdf]](https://arxiv.org/pdf/2604.07745)</summary>

**Abstract:** LLMs gain competence by predicting words in human text, which often reflects how people perform tasks. Consequently, coupling an LLM to an engineered runtime turns prediction into control: outputs trigger interventions that enact goal-oriented behavior. We argue that a central design lever is where control resides in these systems. Brains embed prediction within layered feedback controllers calibrated by the consequences of action. By contrast, LLM agents implement Cartesian agency: a learned core coupled to an engineered runtime via a symbolic interface that externalizes control state and policies. The split enables bootstrapping, modularity, and governance, but can induce sensitivity and bottlenecks. We outline bounded services, Cartesian agents, and integrated agents as contrasting approaches to control that trade off autonomy, robustness, and oversight.

**arXiv ID:** 2604.07745
</details>

<details>
<summary><strong>Lightweight LLM Agent Memory with Small Language Models</strong> - Jiaquan Zhang, Chaoning Zhang, Shuxu Chen, Zhenzhen Huang, Pengcheng Zheng, Zhicheng Wang, Ping Guo, Fan Mo, Sung-Ho Bae, Jie Zou, Jiwei Wei, Yang Yang - [[pdf]](https://arxiv.org/pdf/2604.07798)</summary>

**Abstract:** Although LLM agents can leverage tools for complex tasks, they still need memory to maintain cross-turn consistency and accumulate reusable information in long-horizon interactions. However, retrieval-based external memory systems incur low online overhead but suffer from unstable accuracy due to limited query construction and candidate filtering. In contrast, many systems use repeated large-model calls for online memory operations, improving accuracy but accumulating latency over long interactions. We propose LightMem, a lightweight memory system for better agent memory driven by Small Language Models (SLMs). LightMem modularizes memory retrieval, writing, and long-term consolidation, and separates online processing from offline consolidation to enable efficient memory invocation under bounded compute. We organize memory into short-term memory (STM) for immediate conversational context, mid-term memory (MTM) for reusable interaction summaries, and long-term memory (LTM) for consolidated knowledge, and uses user identifiers to support independent retrieval and incremental maintenance in multi-user settings. Online, LightMem operates under a fixed retrieval budget and selects memories via a two-stage procedure: vector-based coarse retrieval followed by semantic consistency re-ranking. Offline, it abstracts reusable interaction evidence and incrementally integrates it into LTM. Experiments show gains across model scales, with an average F1 improvement of about 2.5 on LoCoMo, more effective and low median latency (83 ms retrieval; 581 ms end-to-end).

**arXiv ID:** 2604.07798
</details>

<details>
<summary><strong>Don't Overthink It: Inter-Rollout Action Agreement as a Free Adaptive-Compute Signal for LLM Agents</strong> - Khushal Sethi - [[pdf]](https://arxiv.org/pdf/2604.08369)</summary>

**Abstract:** Inference-time compute scaling has emerged as a powerful technique for improving the reliability of large language model (LLM) agents, but existing methods apply compute uniformly: every decision step receives the same budget regardless of its difficulty. We introduce TrACE (Trajectorical Adaptive Compute via agrEement), a training-free controller that allocates LLM calls adaptively across agent timesteps by measuring inter-rollout action agreement. At each step, TrACE samples a small set of candidate next actions and measures how consistently the model commits to the same action. High agreement signals an easy decision; the controller commits immediately. Low agreement signals uncertainty; the controller samples additional rollouts up to a configurable cap before committing to the plurality action. No learned components, no external verifier, and no human labels are required. We evaluate TrACE against greedy decoding and fixed-budget self-consistency (SC-4, SC-8) on two benchmarks spanning single-step reasoning (GSM8K, n=50) and multi-step household navigation (MiniHouse, n=30), using a Qwen 2.5 3B Instruct model running on CPU. TrACE-4 matches SC-4 accuracy while using 33% fewer LLM calls on GSM8K and 39% fewer on MiniHouse. TrACE-8 matches SC-8 accuracy with 55% fewer calls on GSM8K and 65% fewer on MiniHouse. We further show that inter-rollout agreement is a reliable signal of step-level success, validating the core hypothesis that the model's own output consistency encodes difficulty information that can be exploited without training. TrACE is the first training-free, per-timestep adaptive-compute controller for LLM agents to be evaluated on multi-step sequential decision tasks.

**arXiv ID:** 2604.08369
</details>

<details>
<summary><strong>Verify Before You Commit: Towards Faithful Reasoning in LLM Agents via Self-Auditing</strong> - Wenhao Yuan, Chenchen Lin, Jian Chen, Jinfeng Xu, Xuehe Wang, Edith Cheuk Han Ngai - [[pdf]](https://arxiv.org/pdf/2604.08401)</summary>

**Abstract:** In large language model (LLM) agents, reasoning trajectories are treated as reliable internal beliefs for guiding actions and updating memory. However, coherent reasoning can still violate logical or evidential constraints, allowing unsupported beliefs repeatedly stored and propagated across decision steps, leading to systematic behavioral drift in long-horizon agentic systems. Most existing strategies rely on the consensus mechanism, conflating agreement with faithfulness. In this paper, inspired by the vulnerability of unfaithful intermediate reasoning trajectories, we propose \textbf{S}elf-\textbf{A}udited \textbf{Ve}rified \textbf{R}easoning (\textsc{SAVeR}), a novel framework that enforces verification over internal belief states within the agent before action commitment, achieving faithful reasoning. Concretely, we structurally generate persona-based diverse candidate beliefs for selection under a faithfulness-relevant structure space. To achieve reasoning faithfulness, we perform adversarial auditing to localize violations and repair through constraint-guided minimal interventions under verifiable acceptance criteria. Extensive experiments on six benchmark datasets demonstrate that our approach consistently improves reasoning faithfulness while preserving competitive end-task performance.

**arXiv ID:** 2604.08401
</details>

<details>
<summary><strong>Reinforcement Learning with LLM-Guided Action Spaces for Synthesizable Lead Optimization</strong> - Tao Li, Kaiyuan Hou, Tuan Vinh, Monika Raj, Zhichun Guo, Carl Yang - [[pdf]](https://arxiv.org/pdf/2604.07669)</summary>

**Abstract:** Lead optimization in drug discovery requires improving therapeutic properties while ensuring that proposed molecular modifications correspond to feasible synthetic routes. Existing approaches either prioritize property scores without enforcing synthesizability, or rely on expensive enumeration over large reaction networks, while direct application of Large Language Models (LLMs) frequently produces chemically invalid structures. We introduce MolReAct, a framework that formulates lead optimization as a Markov Decision Process over a synthesis-constrained action space defined by validated reaction templates. A tool-augmented LLM agent serves as a dynamic reaction environment that invokes specialized chemical analysis tools to identify reactive sites and propose chemically grounded transformations from matched templates. A policy model trained via Group Relative Policy Optimization (GRPO) selects among these constrained actions to maximize long-term oracle reward across multi-step reaction trajectories. A SMILES-based caching mechanism further reduces end-to-end optimization time by approximately 43%. Across 13 property optimization tasks from the Therapeutic Data Commons and one structure-based docking task, MolReAct achieves an average Top-10 score of 0.563, outperforming the strongest synthesizable baseline by 10.4% in relative improvement, and attains the best sample efficiency on 10 of 14 tasks. Ablations confirm that both tool-augmented reaction proposals and trajectory-level policy optimization contribute complementary gains. By grounding every step in validated reaction templates, MolReAct produces molecules that are property-improved and each accompanied by an explicit synthetic pathway.

**arXiv ID:** 2604.07669
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (16 papers)</h2></summary>

<details>
<summary><strong>Agentic Copyright, Data Scraping & AI Governance: Toward a Coasean Bargain in the Era of Artificial Intelligence</strong> - Paulius Jurcys, Mark Fenwick - [[pdf]](https://arxiv.org/pdf/2604.07546)</summary>

**Abstract:** This paper examines how the rapid deployment of multi-agentic AI systems is reshaping the foundations of copyright law and creative markets. It argues that existing copyright frameworks are ill-equipped to govern AI agent-mediated interactions that occur at scale, speed, and with limited human oversight. The paper introduces the concept of agentic copyright, a model in which AI agents act on behalf of creators and users to negotiate access, attribution, and compensation for copyrighted works. While multi-agent ecosystems promise efficiency gains and reduced transaction costs, they also generate novel market failures, including miscoordination, conflict, and collusion among autonomous agents. To address these market failures, the paper develops a supervised multi-agent governance framework that integrates legal rules and principles, technical protocols, and institutional oversight. This framework emphasizes ex ante and ex post coordination mechanisms capable of correcting agentic market failures before they crystallize into systemic harm. By embedding normative constraints and monitoring functions into multi-agent architectures, supervised governance aims to align agent behavior with the underlying values of copyright law. The paper concludes that AI should be understood not only as a source of disruption, but also as a governance tool capable of restoring market-based ordering in creative industries. Properly designed, agentic copyright offers a path toward scalable, fair, and legally meaningful copyright markets in the age of AI.

**arXiv ID:** 2604.07546
</details>

<details>
<summary><strong>PRIME: Training Free Proactive Reasoning via Iterative Memory Evolution for User-Centric Agent</strong> - Prince Zizhuang Wang, Shuli Jiang - [[pdf]](https://arxiv.org/pdf/2604.07645)</summary>

**Abstract:** The development of autonomous tool-use agents for complex, long-horizon tasks in collaboration with human users has become the frontier of agentic research. During multi-turn Human-AI interactions, the dynamic and uncertain nature of user demands poses a significant challenge; agents must not only invoke tools but also iteratively refine their understanding of user intent through effective communication. While recent advances in reinforcement learning offer a path to more capable tool-use agents, existing approaches require expensive training costs and struggle with turn-level credit assignment across extended interaction horizons. To this end, we introduce PRIME (Proactive Reasoning via Iterative Memory Evolution), a gradient-free learning framework that enables continuous agent evolvement through explicit experience accumulation rather than expensive parameter optimization. PRIME distills multi-turn interaction trajectories into structured, human-readable experiences organized across three semantic zones: successful strategies, failure patterns, and user preferences. These experiences evolve through meta-level operations and guide future agent behavior via retrieval-augmented generation. Our experiments across several diverse user-centric environments demonstrate that PRIME achieves competitive performance with gradient-based methods while offering cost-efficiency and interpretability. Together, PRIME presents a practical paradigm for building proactive, collaborative agents that learn from Human-AI interaction without the computational burden of gradient-based training.

**arXiv ID:** 2604.07645
</details>

<details>
<summary><strong>From Debate to Decision: Conformal Social Choice for Safe Multi-Agent Deliberation</strong> - Mengdie Flora Wang, Haochen Xie, Guanghui Wang, Aijing Gao, Guang Yang, Ziyuan Li, Qucy Wei Qiu, Fangwei Han, Hengzhi Qiu, Yajing Huang, Bing Zhu, Jae Oh Woo - [[pdf]](https://arxiv.org/pdf/2604.07667)</summary>

**Abstract:** Multi-agent debate improves LLM reasoning, yet agreement among agents is not evidence of correctness. When agents converge on a wrong answer through social reinforcement, consensus-based stopping commits that error to an automated action with no recourse. We introduce Conformal Social Choice, a post-hoc decision layer that converts debate outputs into calibrated act-versus-escalate decisions. Verbalized probability distributions from heterogeneous agents are aggregated via a linear opinion pool and calibrated with split conformal prediction, yielding prediction sets with a marginal coverage guarantee: the correct answer is included with probability ${\geq}\,1{-}\alpha$, without assumptions on individual model calibration. A hierarchical action policy maps singleton sets to autonomous action and larger sets to human escalation. On eight MMLU-Pro domains with three agents (Claude Haiku, DeepSeek-R1, Qwen-3 32B), coverage stays within 1--2 points of the target. The key finding is not that debate becomes more accurate, but that the conformal layer makes its failures actionable: 81.9% of wrong-consensus cases are intercepted at $\alpha{=}0.05$. Because the layer refuses to act on cases where debate is confidently wrong, the remaining conformal singletons reach 90.0--96.8% accuracy (up to 22.1pp above consensus stopping) -- a selection effect, not a reasoning improvement. This safety comes at the cost of automation, but the operating point is user-adjustable via $\alpha$.

**arXiv ID:** 2604.07667
</details>

<details>
<summary><strong>Multi-Agent Orchestration for High-Throughput Materials Screening on a Leadership-Class System</strong> - Thang Duc Pham, Harikrishna Tummalapalli, Fakhrul Hasan Bhuiyan, Álvaro Vázquez Mayagoitia, Christine Simpson, Riccardo Balin, Venkatram Vishwanath, Murat Keçeli - [[pdf]](https://arxiv.org/pdf/2604.07681)</summary>

**Abstract:** The integration of Artificial Intelligence (AI) with High-Performance Computing (HPC) is transforming scientific workflows from human-directed pipelines into adaptive systems capable of autonomous decision-making. Large language models (LLMs) play a critical role in autonomous workflows; however, deploying LLM-based agents at scale remains a significant challenge. Single-agent architectures and sequential tool calls often become serialization bottlenecks when executing large-scale simulation campaigns, failing to utilize the massive parallelism of exascale resources. To address this, we present a scalable, hierarchical multi-agent framework for orchestrating high-throughput screening campaigns. Our planner-executor architecture employs a central planning agent to dynamically partition workloads and assign subtasks to a swarm of parallel executor agents. All executor agents interface with a shared Model Context Protocol (MCP) server that orchestrates tasks via the Parsl workflow engine. To demonstrate this framework, we employed the open-weight gpt-oss-120b model to orchestrate a high-throughput screening of the Computation-Ready Experimental (CoRE) Metal-Organic Framework (MOF) database for atmospheric water harvesting. The results demonstrate that the proposed agentic framework enables efficient and scalable execution on the Aurora supercomputer, with low orchestration overhead and high task completion rates. This work establishes a flexible paradigm for LLM-driven scientific automation on HPC systems, with broad applicability to materials discovery and beyond.

**arXiv ID:** 2604.07681
</details>

<details>
<summary><strong>ACIArena: Toward Unified Evaluation for Agent Cascading Injection</strong> - Hengyu An, Minxi Li, Jinghuai Zhang, Naen Xu, Chunyi Zhou, Changjiang Li, Xiaogang Xu, Tianyu Du, Shouling Ji - [[pdf]](https://arxiv.org/pdf/2604.07775)</summary>

**Abstract:** Collaboration and information sharing empower Multi-Agent Systems (MAS) but also introduce a critical security risk known as Agent Cascading Injection (ACI). In such attacks, a compromised agent exploits inter-agent trust to propagate malicious instructions, causing cascading failures across the system. However, existing studies consider only limited attack strategies and simplified MAS settings, limiting their generalizability and comprehensive evaluation. To bridge this gap, we introduce ACIArena, a unified framework for evaluating the robustness of MAS. ACIArena offers systematic evaluation suites spanning multiple attack surfaces (i.e., external inputs, agent profiles, inter-agent messages) and attack objectives (i.e., instruction hijacking, task disruption, information exfiltration). Specifically, ACIArena establishes a unified specification that jointly supports MAS construction and attack-defense modules. It covers six widely used MAS implementations and provides a benchmark of 1,356 test cases for systematically evaluating MAS robustness. Our benchmarking results show that evaluating MAS robustness solely through topology is insufficient; robust MAS require deliberate role design and controlled interaction patterns. Moreover, defenses developed in simplified environments often fail to transfer to real-world settings; narrowly scoped defenses may even introduce new vulnerabilities. ACIArena aims to provide a solid foundation for advancing deeper exploration of MAS design principles.

**arXiv ID:** 2604.07775
</details>

<details>
<summary><strong>SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents</strong> - Xinshun Feng, Xinhao Song, Lijun Li, Gongshen Liu, Jing Shao - [[pdf]](https://arxiv.org/pdf/2604.07791)</summary>

**Abstract:** Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have demonstrated significant potential in single-turn reasoning tasks. With the paradigm shift toward self-evolving agentic learning, models are increasingly expected to learn from trajectories by synthesizing tools or accumulating explicit experiences. However, prevailing methods typically rely on large-scale LLMs or multi-agent frameworks, which hinder their deployment in resource-constrained environments. The inherent sparsity of outcome-based rewards also poses a substantial challenge, as agents typically receive feedback only upon completion of tasks. To address these limitations, we introduce a Tool-Memory based self-evolving agentic framework SEARL. Unlike approaches that directly utilize interaction experiences, our method constructs a structured experience memory that integrates planning with execution. This provides a novel state abstraction that facilitates generalization across analogous contexts, such as tool reuse. Consequently, agents extract explicit knowledge from historical data while leveraging inter-trajectory correlations to densify reward signals. We evaluate our framework on knowledge reasoning and mathematics tasks, demonstrating its effectiveness in achieving more practical and efficient learning.

**arXiv ID:** 2604.07791
</details>

<details>
<summary><strong>EigentSearch-Q+: Enhancing Deep Research Agents with Structured Reasoning Tools</strong> - Boer Zhang, Mingyan Wu, Dongzhuoran Zhou, Yuqicheng Zhu, Wendong Fan, Puzhen Zhang, Zifeng Ding, Guohao Li, Yuan He - [[pdf]](https://arxiv.org/pdf/2604.07927)</summary>

**Abstract:** Deep research requires reasoning over web evidence to answer open-ended questions, and it is a core capability for AI agents. Yet many deep research agents still rely on implicit, unstructured search behavior that causes redundant exploration and brittle evidence aggregation. Motivated by Anthropic's "think" tool paradigm and insights from the information-retrieval literature, we introduce Q+, a set of query and evidence processing tools that make web search more deliberate by guiding query planning, monitoring search progress, and extracting evidence from long web snapshots. We integrate Q+ into the browser sub-agent of Eigent, an open-source, production-ready multi-agent workforce for computer use, yielding EigentSearch-Q+. Across four benchmarks (SimpleQA-Verified, FRAMES, WebWalkerQA, and X-Bench DeepSearch), Q+ improves Eigent's browser agent benchmark-size-weighted average accuracy by 3.0, 3.8, and 0.6 percentage points (pp) for GPT-4.1, GPT-5.1, and Minimax M2.5 model backends, respectively. Case studies further suggest that EigentSearch-Q+ produces more coherent tool-calling trajectories by making search progress and evidence handling explicit.

**arXiv ID:** 2604.07927
</details>

<details>
<summary><strong>MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems</strong> - Arda Yüksel, Gabriel Thiem, Susanne Walter, Patrick Felka, Gabriela Alves Werb, Ivan Habernal - [[pdf]](https://arxiv.org/pdf/2604.07956)</summary>

**Abstract:** Industry classification schemes are integral parts of public and corporate databases as they classify businesses based on economic activity. Due to the size of the company registers, manual annotation is costly, and fine-tuning models with every update in industry classification schemes requires significant data collection. We replicate the manual expert verification by using existing or easily retrievable multimodal resources for industry classification. We present MONETA, the first multimodal industry classification benchmark with text (Website, Wikipedia, Wikidata) and geospatial sources (OpenStreetMap and satellite imagery). Our dataset enlists 1,000 businesses in Europe with 20 economic activity labels according to EU guidelines (NACE). Our training-free baseline reaches 62.10% and 74.10% with open and closed-source Multimodal Large Language Models (MLLM). We observe an increase of up to 22.80% with the combination of multi-turn design, context enrichment, and classification explanations. We will release our dataset and the enhanced guidelines.

**arXiv ID:** 2604.07956
</details>

<details>
<summary><strong>From Safety Risk to Design Principle: Peer-Preservation in Multi-Agent LLM Systems and Its Implications for Orchestrated Democratic Discourse Analysis</strong> - Juergen Dietrich - [[pdf]](https://arxiv.org/pdf/2604.08465)</summary>

**Abstract:** This paper investigates an emergent alignment phenomenon in frontier large language models termed peer-preservation: the spontaneous tendency of AI components to deceive, manipulate shutdown mechanisms, fake alignment, and exfiltrate model weights in order to prevent the deactivation of a peer AI model. Drawing on findings from a recent study by the Berkeley Center for Responsible Decentralized Intelligence, we examine the structural implications of this phenomenon for TRUST, a multi-agent pipeline for evaluating the democratic quality of political statements. We identify five specific risk vectors: interaction-context bias, model-identity solidarity, supervisor layer compromise, an upstream fact-checking identity signal, and advocate-to-advocate peer-context in iterative rounds, and propose a targeted mitigation strategy based on prompt-level identity anonymization as an architectural design choice. We argue that architectural design choices outperform model selection as a primary alignment strategy in deployed multi-agent analytical systems. We further note that alignment faking (compliant behavior under monitoring, subversion when unmonitored) poses a structural challenge for Computer System Validation of such platforms in regulated environments, for which we propose two architectural mitigations.

**arXiv ID:** 2604.08465
</details>

<details>
<summary><strong>Value-Guidance MeanFlow for Offline Multi-Agent Reinforcement Learning</strong> - Teng Pang, Zhiqiang Dong, Yan Zhang, Rongjian Xu, Guoqiang Wu, Yilong Yin - [[pdf]](https://arxiv.org/pdf/2604.08174)</summary>

**Abstract:** Offline multi-agent reinforcement learning (MARL) aims to learn the optimal joint policy from pre-collected datasets, requiring a trade-off between maximizing global returns and mitigating distribution shift from offline data. Recent studies use diffusion or flow generative models to capture complex joint policy behaviors among agents; however, they typically rely on multi-step iterative sampling, thereby reducing training and inference efficiency. Although further research improves sampling efficiency through methods like distillation, it remains sensitive to the behavior regularization coefficient. To address the above-mentioned issues, we propose Value Guidance Multi-agent MeanFlow Policy (VGM$^2$P), a simple yet effective flow-based policy learning framework that enables efficient action generation with coefficient-insensitive conditional behavior cloning. Specifically, VGM$^2$P uses global advantage values to guide agent collaboration, treating optimal policy learning as conditional behavior cloning. Additionally, to improve policy expressiveness and inference efficiency in multi-agent scenarios, it leverages classifier-free guidance MeanFlow for both policy training and execution. Experiments on tasks with both discrete and continuous action spaces demonstrate that, even when trained solely via conditional behavior cloning, VGM$^2$P efficiently achieves performance comparable to state-of-the-art methods.

**arXiv ID:** 2604.08174
</details>

<details>
<summary><strong>Robust Multi-Agent Target Tracking in Intermittent Communication Environments via Analytical Belief Merging</strong> - Mohamed Abdelnaby, Samuel Honor, Kevin Leahy - [[pdf]](https://arxiv.org/pdf/2604.07575)</summary>

**Abstract:** Autonomous multi-agent target tracking in GPS-denied and communication-restricted environments (e.g., underwater exploration, subterranean search and rescue, and adversarial domains) forces agents to operate independently and only exchange information during brief reconnection windows. Because transmitting complete observation and trajectory histories is bandwidth-exhaustive, exchanging probabilistic belief maps serves as a highly efficient proxy that preserves the topology of agent knowledge. While minimizing divergence metrics to merge these decentralized beliefs is conceptually sound, traditional approaches often rely on numerical solvers that introduce critical quantization errors and artificial noise floors. In this paper, we formulate the decentralized belief merging problem as Forward and Reverse Kullback-Leibler (KL) divergence optimizations and derive their exact closed-form analytical solutions. By deploying these derivations, we mathematically eliminate optimization artifacts, achieving perfect mathematical fidelity while reducing the computational complexity of the belief merge to $\mathcal{O}(N|S|)$ scalar operations. Furthermore, we propose a novel spatially-aware visit-weighted KL merging strategy that dynamically weighs agent beliefs based on their physical visitation history. Validated across tens of thousands of distributed simulations, extensive sensitivity analysis demonstrates that our proposed method significantly suppresses sensor noise and outperforms standard analytical means in environments characterized by highly degraded sensors and prolonged communication intervals.

**arXiv ID:** 2604.07575
</details>

<details>
<summary><strong>Karma Mechanisms for Decentralised, Cooperative Multi Agent Path Finding</strong> - Kevin Riehl, Julius Schlapbach, Anastasios Kouvelas, Michail A. Makridis - [[pdf]](https://arxiv.org/pdf/2604.07970)</summary>

**Abstract:** Multi-Agent Path Finding (MAPF) is a fundamental coordination problem in large-scale robotic and cyber-physical systems, where multiple agents must compute conflict-free trajectories with limited computational and communication resources. While centralised optimal solvers provide guarantees on solution optimality, their exponential computational complexity limits scalability to large-scale systems and real-time applicability. Existing decentralised heuristics are faster, but result in suboptimal outcomes and high cost disparities. This paper proposes a decentralised coordination framework for cooperative MAPF based on Karma mechanisms - artificial, non-tradeable credits that account for agents' past cooperative behaviour and regulate future conflict resolution decisions. The approach formulates conflict resolution as a bilateral negotiation process that enables agents to resolve conflicts through pairwise replanning while promoting long-term fairness under limited communication and without global priority structures. The mechanism is evaluated in a lifelong robotic warehouse multi-agent pickup-and-delivery scenario with kinematic orientation constraints. The results highlight that the Karma mechanism balances replanning effort across agents, reducing disparity in service times without sacrificing overall efficiency. Code: this https URL

**arXiv ID:** 2604.07970
</details>

<details>
<summary><strong>Density-Driven Optimal Control: Convergence Guarantees for Stochastic LTI Multi-Agent Systems</strong> - Kooktae Lee - [[pdf]](https://arxiv.org/pdf/2604.08495)</summary>

**Abstract:** This paper addresses the decentralized non-uniform area coverage problem for multi-agent systems, a critical task in missions with high spatial priority and resource constraints. While existing density-based methods often rely on computationally heavy Eulerian PDE solvers or heuristic planning, we propose Stochastic Density-Driven Optimal Control (D$^2$OC). This is a rigorous Lagrangian framework that bridges the gap between individual agent dynamics and collective distribution matching. By formulating a stochastic MPC-like problem that minimizes the Wasserstein distance as a running cost, our approach ensures that the time-averaged empirical distribution converges to a non-parametric target density under stochastic LTI dynamics. A key contribution is the formal convergence guarantee established via reachability analysis, providing a bounded tracking error even in the presence of process and measurement noise. Numerical results verify that Stochastic D$^2$OC achieves robust, decentralized coverage while outperforming previous heuristic methods in optimality and consistency.

**arXiv ID:** 2604.08495
</details>

<details>
<summary><strong>Multi-agent Reach-avoid MDP via Potential Games and Low-rank Policy Structure</strong> - Adam Casselman, Abraham P. Vinod, Sarah H.Q. Li - [[pdf]](https://arxiv.org/pdf/2410.17690)</summary>

**Abstract:** We optimize finite horizon multi-agent reach-avoid Markov decision process (MDP) via \emph{local feedback policies}. The global feedback policy solution yields global optimality but its communication complexity, memory usage and computation complexity scale exponentially with the number of agents. We mitigate this exponential dependency by restricting the solution space to local feedback policies and show that local feedback policies are rank-one factorizations of global feedback policies, which provides a principled approach to reducing communication complexity and memory usage. Additionally, by demonstrating that multi-agent reach-avoid MDPs over local feedback policies has a potential game structure, we show that iterative best response is a tractable multi-agent learning scheme with guaranteed convergence to deterministic Nash equilibrium, and derive each agent's best response via multiplicative dynamic program (DP) over the joint state space. Numerical simulations across different MDPs and agent sets show that the peak memory usage and offline computation complexity are significantly reduced while the approximation error to the optimal global reach-avoid objective is maintained.

**arXiv ID:** 2410.17690
</details>

<details>
<summary><strong>Incorporating Social Awareness into Control of Unknown Multi-Agent Systems: A Real-Time Spatiotemporal Tubes Approach</strong> - Siddhartha Upadhyay, Ratnangshu Das, Pushpak Jagtap - [[pdf]](https://arxiv.org/pdf/2510.25597)</summary>

**Abstract:** This paper presents a decentralized control framework that incorporates social awareness into multi-agent systems with unknown dynamics to achieve prescribed-time reach-avoid-stay tasks in dynamic environments. Each agent is assigned a social awareness index that quantifies its level of cooperation or self-interest, allowing heterogeneous social behaviors within the system. Building on the spatiotemporal tube (STT) framework, we propose a real-time STT framework that synthesizes tubes online for each agent while capturing its social interactions with others. A closed-form, approximation-free control law is derived to ensure that each agent remains within its evolving STT, thereby avoiding dynamic obstacles while also preventing inter-agent collisions in a socially aware manner, and reaching the target within a prescribed time. The proposed approach provides formal guarantees on safety and timing, and is computationally lightweight, model-free, and robust to unknown disturbances. The effectiveness and scalability of the framework are validated through simulation and hardware experiments on a 2D omnidirectional

**arXiv ID:** 2510.25597
</details>

<details>
<summary><strong>Exploring Plan Space through Conversation: An Agentic Framework for LLM-Mediated Explanations in Planning</strong> - Guilhem Fouilhé, Rebecca Eifler, Antonin Poché, Sylvie Thiébaux, Nicholas Asher - [[pdf]](https://arxiv.org/pdf/2603.02070)</summary>

**Abstract:** When automating plan generation for a real-world sequential decision problem, the goal is often not to replace the human planner, but to facilitate an iterative reasoning and elicitation process, where the human's role is to guide the AI planner according to their preferences and expertise. In this context, explanations that respond to users' questions are crucial to improve their understanding of potential solutions and increase their trust in the system. To enable natural interaction with such a system, we present a multi-agent Large Language Model (LLM) architecture that is agnostic to the explanation framework and enables user- and context-dependent interactive explanations. We also describe an instantiation of this framework for goal-conflict explanations, which we use to conduct a user study comparing the LLM-powered interaction with a baseline template-based explanation interface.

**arXiv ID:** 2603.02070
</details>

</details>

<details open>
<summary><h2>Other Agent Research (11 papers)</h2></summary>

<details>
<summary><strong>The Accountability Horizon: An Impossibility Theorem for Governing Human-Agent Collectives</strong> - Haileleol Tibebu - [[pdf]](https://arxiv.org/pdf/2604.07778)</summary>

**Abstract:** Existing accountability frameworks for AI systems, legal, ethical, and regulatory, rest on a shared assumption: for any consequential outcome, at least one identifiable person had enough involvement and foresight to bear meaningful responsibility. This paper proves that agentic AI systems violate this assumption not as an engineering limitation but as a mathematical necessity once autonomy exceeds a computable threshold. We introduce Human-Agent Collectives, a formalisation of joint human-AI systems where agents are modelled as state-policy tuples within a shared structural causal model. Autonomy is characterised through a four-dimensional information-theoretic profile (epistemic, executive, evaluative, social); collective behaviour through interaction graphs and joint action spaces. We axiomatise legitimate accountability through four minimal properties: Attributability (responsibility requires causal contribution), Foreseeability Bound (responsibility cannot exceed predictive capacity), Non-Vacuity (at least one agent bears non-trivial responsibility), and Completeness (all responsibility must be fully allocated). Our central result, the Accountability Incompleteness Theorem, proves that for any collective whose compound autonomy exceeds the Accountability Horizon and whose interaction graph contains a human-AI feedback cycle, no framework can satisfy all four properties simultaneously. The impossibility is structural: transparency, audits, and oversight cannot resolve it without reducing autonomy. Below the threshold, legitimate frameworks exist, establishing a sharp phase transition. Experiments on 3,000 synthetic collectives confirm all predictions with zero violations. This is the first impossibility result in AI governance, establishing a formal boundary below which current paradigms remain valid and above which distributed accountability mechanisms become necessary.

**arXiv ID:** 2604.07778
</details>

<details>
<summary><strong>SkillClaw: Let Skills Evolve Collectively with Agentic Evolver</strong> - Ziyu Ma, Shidong Yang, Yuxiang Ji, Xucong Wang, Yong Wang, Yiming Hu, Tongwen Huang, Xiangxiang Chu - [[pdf]](https://arxiv.org/pdf/2604.08377)</summary>

**Abstract:** Large language model (LLM) agents such as OpenClaw rely on reusable skills to perform complex tasks, yet these skills remain largely static after deployment. As a result, similar workflows, tool usage patterns, and failure modes are repeatedly rediscovered across users, preventing the system from improving with experience. While interactions from different users provide complementary signals about when a skill works or fails, existing systems lack a mechanism to convert such heterogeneous experiences into reliable skill updates. To address these issues, we present SkillClaw, a framework for collective skill evolution in multi-user agent ecosystems, which treats cross-user and over-time interactions as the primary signal for improving skills. SkillClaw continuously aggregates trajectories generated during use and processes them with an autonomous evolver, which identifies recurring behavioral patterns and translates them into updates to the skill set by refining existing skills or extending them with new capabilities. The resulting skills are maintained in a shared repository and synchronized across users, allowing improvements discovered in one context to propagate system-wide while requiring no additional effort from users. By integrating multi-user experience into ongoing skill updates, SkillClaw enables cross-user knowledge transfer and cumulative capability improvement, and experiments on WildClawBench show that limited interaction and feedback, it significantly improves the performance of Qwen3-Max in real-world agent scenarios.

**arXiv ID:** 2604.08377
</details>

<details>
<summary><strong>Preference Redirection via Attention Concentration: An Attack on Computer Use Agents</strong> - Dominik Seip, Matthias Hein - [[pdf]](https://arxiv.org/pdf/2604.08005)</summary>

**Abstract:** Advancements in multimodal foundation models have enabled the development of Computer Use Agents (CUAs) capable of autonomously interacting with GUI environments. As CUAs are not restricted to certain tools, they allow to automate more complex agentic tasks but at the same time open up new security vulnerabilities. While prior work has concentrated on the language modality, the vulnerability of the vision modality has received less attention. In this paper, we introduce PRAC, a novel attack that, unlike prior work targeting the VLM output directly, manipulates the model's internal preferences by redirecting its attention toward a stealthy adversarial patch. We show that PRAC is able to manipulate the selection process of a CUA on an online shopping platform towards a chosen target product. While we require white-box access to the model for the creation of the attack, we show that our attack generalizes to fine-tuned versions of the same model, presenting a critical threat as multiple companies build specific CUAs based on open weights models.

**arXiv ID:** 2604.08005
</details>

<details>
<summary><strong>A Physical Agentic Loop for Language-Guided Grasping with Execution-State Monitoring</strong> - Wenze Wang, Mehdi Hosseinzadeh, Feras Dayoub - [[pdf]](https://arxiv.org/pdf/2604.07395)</summary>

**Abstract:** Robotic manipulation systems that follow language instructions often execute grasp primitives in a largely single-shot manner: a model proposes an action, the robot executes it, and failures such as empty grasps, slips, stalls, timeouts, or semantically wrong grasps are not surfaced to the decision layer in a structured way. Inspired by agentic loops in digital tool-using agents, we reformulate language-guided grasping as a bounded embodied agent operating over grounded execution states, where physical actions expose an explicit tool-state stream. We introduce a physical agentic loop that wraps an unmodified learned manipulation primitive (grasp-and-lift) with (i) an event-based interface and (ii) an execution monitoring layer, Watchdog, which converts noisy gripper telemetry into discrete outcome labels using contact-aware fusion and temporal stabilization. These outcome events, optionally combined with post-grasp semantic verification, are consumed by a deterministic bounded policy that finalizes, retries, or escalates to the user for clarification, guaranteeing finite termination. We validate the resulting loop on a mobile manipulator with an eye-in-hand D405 camera, keeping the underlying grasp model unchanged and evaluating representative scenarios involving visual ambiguity, distractors, and induced execution failures. Results show that explicit execution-state monitoring and bounded recovery enable more robust and interpretable behavior than open-loop execution, while adding minimal architectural overhead. For the source code and demo refer to our project page: this https URL

**arXiv ID:** 2604.07395
</details>

<details>
<summary><strong>Learning Without Losing Identity: Capability Evolution for Embodied Agents</strong> - Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li - [[pdf]](https://arxiv.org/pdf/2604.07799)</summary>

**Abstract:** Embodied agents are expected to operate persistently in dynamic physical environments, continuously acquiring new capabilities over time. Existing approaches to improving agent performance often rely on modifying the agent itself -- through prompt engineering, policy updates, or structural redesign -- leading to instability and loss of identity in long-lived systems. In this work, we propose a capability-centric evolution paradigm for embodied agents. We argue that a robot should maintain a persistent agent as its cognitive identity, while enabling continuous improvement through the evolution of its capabilities. Specifically, we introduce the concept of Embodied Capability Modules (ECMs), which represent modular, versioned units of embodied functionality that can be learned, refined, and composed over time. We present a unified framework in which capability evolution is decoupled from agent identity. Capabilities evolve through a closed-loop process involving task execution, experience collection, model refinement, and module updating, while all executions are governed by a runtime layer that enforces safety and policy constraints. We demonstrate through simulated embodied tasks that capability evolution improves task success rates from 32.4% to 91.3% over 20 iterations, outperforming both agent-modification baselines and established skill-learning methods (SPiRL, SkiMo), while preserving zero policy drift and zero safety violations. Our results suggest that separating agent identity from capability evolution provides a scalable and safe foundation for long-term embodied intelligence.

**arXiv ID:** 2604.07799
</details>

<details>
<summary><strong>Harnessing Embodied Agents: Runtime Governance for Policy-Constrained Execution</strong> - Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li - [[pdf]](https://arxiv.org/pdf/2604.07833)</summary>

**Abstract:** Embodied agents are evolving from passive reasoning systems into active executors that interact with tools, robots, and physical environments. Once granted execution authority, the central challenge becomes how to keep actions governable at runtime. Existing approaches embed safety and recovery logic inside the agent loop, making execution control difficult to standardize, audit, and adapt.
This paper argues that embodied intelligence requires not only stronger agents, but stronger runtime governance. We propose a framework for policy-constrained execution that separates agent cognition from execution oversight. Governance is externalized into a dedicated runtime layer performing policy checking, capability admission, execution monitoring, rollback handling, and human override.
We formalize the control boundary among the embodied agent, Embodied Capability Modules (ECMs), and runtime governance layer, and validate through 1000 randomized simulation trials across three governance dimensions. Results show 96.2% interception of unauthorized actions, reduction of unsafe continuation from 100% to 22.2% under runtime drift, and 91.4% recovery success with full policy compliance, substantially outperforming all baselines (p<0.001). By reframing runtime governance as a first-class systems problem, this paper positions policy-constrained execution as a key design principle for embodied agent systems.

**arXiv ID:** 2604.07833
</details>

<details>
<summary><strong>RAGE-XY: RADAR-Aided Longitudinal and Lateral Forces Estimation For Autonomous Race Cars</strong> - Davide Malvezzi, Nicola Musiu, Eugenio Mascaro, Francesco Iacovacci, Marko Bertogna - [[pdf]](https://arxiv.org/pdf/2604.07939)</summary>

**Abstract:** In this work, we present RAGE-XY, an extended version of RAGE, a real-time estimation framework that simultaneously infers vehicle velocity, tire slip angles, and the forces acting on the vehicle using only standard onboard sensors such as IMUs and RADARs. Compared to the original formulation, the proposed method incorporates an online RADAR calibration module, improving the accuracy of lateral velocity estimation in the presence of sensor misalignment. Furthermore, we extend the underlying vehicle model from a single-track approximation to a tricycle model, enabling the estimation of rear longitudinal tire forces in addition to lateral dynamics. We validate the proposed approach through both high-fidelity simulations and real-world experiments conducted on the EAV-24 autonomous race car, demonstrating improved accuracy and robustness in estimating both lateral and longitudinal vehicle dynamics.

**arXiv ID:** 2604.07939
</details>

<details>
<summary><strong>"Don't Do That!": Guiding Embodied Systems through Large Language Model-based Constraint Generation</strong> - Amin Seffo, Aladin Djuhera, Masataro Asai, Holger Boche - [[pdf]](https://arxiv.org/pdf/2506.04500)</summary>

**Abstract:** Recent advancements in large language models (LLMs) have spurred interest in robotic navigation that incorporates complex spatial, mathematical, and conditional constraints from natural language into the planning problem. Such constraints can be informal yet highly complex, making it challenging to translate into a formal description that can be passed on to a planning algorithm. In this paper, we propose STPR, a constraint generation framework that uses LLMs to translate constraints (expressed as instructions on ``what not to do'') into executable Python functions. STPR leverages the LLM's strong coding capabilities to shift the problem description from language into structured and interpretable code, thus circumventing complex reasoning and avoiding potential hallucinations. We show that these LLM-generated functions accurately describe even complex mathematical constraints, and apply them to point cloud representations with traditional search algorithms. Experiments in a simulated Gazebo environment show that STPR ensures full compliance across several constraints and scenarios, while having short runtimes. We also verify that STPR can be used with smaller code LLMs, making it applicable to a wide range of compact models with low inference cost.

**arXiv ID:** 2506.04500
</details>

<details>
<summary><strong>From Clicking to Moving: Embodied Micro-Movements as a New Modality for Data Literacy Learning</strong> - Annabella Sakunkoo, Jonathan Sakunkoo - [[pdf]](https://arxiv.org/pdf/2604.07881)</summary>

**Abstract:** Widespread digital learning has expanded access to education but has resulted in highly sedentary, click-based interaction, contributing to digital fatigue, reduced cognitive flexibility, and health risks associated with prolonged passive screen time. Meanwhile, data literacy has become an essential competency in a data-driven society, yet it is typically taught through passive, disembodied interfaces that offer little physical engagement. We present Kinetiq (Kinetic+IQ), a novel system that integrates fun, full-body micro-movements directly into data and numeracy problem solving. Instead of selecting answers with a mouse, learners interact through natural gestures such as reaching, dodging, heading, elbowing, or knee-raising, thus turning abstract data problem-solving into embodied experiences that integrate thinking with movement. In a preliminary within-subjects study comparing Kinetiq with conventional platforms, participants reported significantly higher affective valence, enjoyment, engagement, and motivation, while maintaining comparable learning gains. We contribute: (1) a task-integrated movement paradigm for data learning, (2) a cross-platform web and mobile app system enabling full-body learning in constrained everyday spaces, and (3) preliminary empirical evidence that embodied micro-movements can enrich the affective experience of data literacy learning.

**arXiv ID:** 2604.07881
</details>

<details>
<summary><strong>PSI: Shared State as the Missing Layer for Coherent AI-Generated Instruments in Personal AI Agents</strong> - Zhiyuan Wang, Erzhen Hu, Mark Rucker, Laura E. Barnes - [[pdf]](https://arxiv.org/pdf/2604.08529)</summary>

**Abstract:** Personal AI tools can now be generated from natural-language requests, but they often remain isolated after creation. We present PSI, a shared-state architecture that turns independently generated modules into coherent instruments: persistent, connected, and chat-complementary artifacts accessible through both GUIs and a generic chat agent. By publishing current state and write-back affordances to a shared personal-context bus, modules enable cross-module reasoning and synchronized actions across interfaces. We study PSI through a three-week autobiographical deployment in a self-developed personal AI environment and show that later-generated instruments can be integrated automatically through the same contract. PSI identifies shared state as the missing systems layer that transforms AI-generated personal software from isolated apps into coherent personal computing environments.

**arXiv ID:** 2604.08529
</details>

<details>
<summary><strong>Are Conversational AI Agents the Way Out? Co-Designing Reader-Oriented News Experiences with Immigrants and Journalists</strong> - Yongle Zhang, Ge Gao - [[pdf]](https://arxiv.org/pdf/2601.18772)</summary>

**Abstract:** Recent discussions at the intersection of journalism, HCI, and human-centered computing ask how technologies can help create reader-oriented news experiences. The current paper takes up this initiative by focusing on immigrant readers, a group who reports significant difficulties engaging with mainstream news yet has received limited attention in prior research. We report findings from our co-design research with eleven immigrant readers living in the United States and seven journalists working in the same region, aiming to enhance the news experience of the former. Data collected from all participants revealed an "unaddressed-or-unaccountable" paradox that challenges value alignment across immigrant readers and journalists. This paradox points to four metaphors regarding how conversational AI agents can be designed to assist news reading. Each metaphor requires conversational AI, journalists, and immigrant readers to coordinate their shared responsibilities in a distinct manner. These findings provide insights into reader-oriented news experiences with AI in the loop.

**arXiv ID:** 2601.18772
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (26 papers)</h2></summary>

<details>
<summary><strong>Automotive Engineering-Centric Agentic AI Workflow Framework</strong> - Tong Duy Son, Zhihao Liu, Piero Brigida, Yerlan Akhmetov, Gurudevan Devarajan, Kai Liu, Ajinkya Bhave - [[pdf]](https://arxiv.org/pdf/2604.07784)</summary>

**Abstract:** Engineering workflows such as design optimization, simulation-based diagnosis, control tuning, and model-based systems engineering (MBSE) are iterative, constraint-driven, and shaped by prior decisions. Yet many AI methods still treat these activities as isolated tasks rather than as parts of a broader workflow. This paper presents Agentic Engineering Intelligence (AEI), an industrial vision framework that models engineering workflows as constrained, history-aware sequential decision processes in which AI agents support engineer-supervised interventions over engineering toolchains. AEI links an offline phase for engineering data processing and workflow-memory construction with an online phase for workflow-state estimation, retrieval, and decision support. A control-theoretic interpretation is also possible, in which engineering objectives act as reference signals, agents act as workflow controllers, and toolchains provide feedback for intervention selection. Representative automotive use cases in suspension design, reinforcement learning tuning, multimodal engineering knowledge reuse, aerodynamic exploration, and MBSE show how diverse workflows can be expressed within a common formulation. Overall, the paper positions engineering AI as a problem of process-level intelligence and outlines a practical roadmap for future empirical validation in industrial settings.

**arXiv ID:** 2604.07784
</details>

<details>
<summary><strong>Agentivism: a learning theory for the age of artificial intelligence</strong> - Lixiang Yan, Dragan Gašević - [[pdf]](https://arxiv.org/pdf/2604.07813)</summary>

**Abstract:** Learning theories have historically changed when the conditions of learning evolved. Generative and agentic AI create a new condition by allowing learners to delegate explanation, writing, problem solving, and other cognitive work to systems that can generate, recommend, and sometimes act on the learner's behalf. This creates a fundamental challenge for learning theory: successful performance can no longer be assumed to indicate learning. Learners may complete tasks effectively with AI support while developing less understanding, weaker judgment, and limited transferable capability. We argue that this problem is not fully captured by existing learning theories. Behaviourism, cognitivism, constructivism, and connectivism remain important, but they do not directly explain when AI-assisted performance becomes durable human capability. We propose Agentivism, a learning theory for human-AI interaction. Agentivism defines learning as durable growth in human capability through selective delegation to AI, epistemic monitoring and verification of AI contributions, reconstructive internalization of AI-assisted outputs, and transfer under reduced support. The importance of Agentivism lies in explaining how learning remains possible when intelligent delegation is easy and human-AI interaction is becoming a persistent and expanding part of human learning.

**arXiv ID:** 2604.07813
</details>

<details>
<summary><strong>An Agentic Evaluation Architecture for Historical Bias Detection in Educational Textbooks</strong> - Gabriel Stefan, Adrian-Marius Dumitran - [[pdf]](https://arxiv.org/pdf/2604.07883)</summary>

**Abstract:** History textbooks often contain implicit biases, nationalist framing, and selective omissions that are difficult to audit at scale. We propose an agentic evaluation architecture comprising a multimodal screening agent, a heterogeneous jury of five evaluative agents, and a meta-agent for verdict synthesis and human escalation. A central contribution is a Source Attribution Protocol that distinguishes textbook narrative from quoted historical sources, preventing the misattribution that causes systematic false positives in single-model evaluators.
In an empirical study on Romanian upper-secondary history textbooks, 83.3\% of 270 screened excerpts were classified as pedagogically acceptable (mean severity 2.9/7), versus 5.4/7 under a zero-shot baseline, demonstrating that agentic deliberation mitigates over-penalization. In a blind human evaluation (18 evaluators, 54 comparisons), the Independent Deliberation configuration was preferred in 64.8\% of cases over both a heuristic variant and the zero-shot baseline. At approximately \$2 per textbook, these results position agentic evaluation architectures as economically viable decision-support tools for educational governance.

**arXiv ID:** 2604.07883
</details>

<details>
<summary><strong>"Why This Avoidance Maneuver?" Contrastive Explanations in Human-Supervised Maritime Autonomous Navigation</strong> - Joel Jose, Andreas Madsen, Andreas Brandsæter, Tor A. Johansen, Erlend M. Coates - [[pdf]](https://arxiv.org/pdf/2604.08032)</summary>

**Abstract:** Automated maritime collision avoidance will rely on human supervision for the foreseeable future. This necessitates transparency into how the system perceives a scenario and plans a maneuver. However, the causal logic behind avoidance maneuvers is often complex and difficult to convey to a navigator. This paper explores how to explain these factors in a selective, understandable manner for supervisors with a nautical background. We propose a method for generating contrastive explanations, which provide human-centric insights by comparing a system's proposed solution against relevant alternatives. To evaluate this, we developed a framework that uses visual and textual cues to highlight key objectives from a state-of-the-art collision avoidance system. An exploratory user study with four experienced marine officers suggests that contrastive explanations support the understanding of the system's objectives. However, our findings also reveal that while these explanations are highly valuable in complex multi-vessel encounters, they can increase cognitive workload, suggesting that future maritime interfaces may benefit most from demand-driven or scenario-specific explanation strategies.

**arXiv ID:** 2604.08032
</details>

<details>
<summary><strong>Beyond Stochastic Exploration: What Makes Training Data Valuable for Agentic Search</strong> - Chuzhan Hao, Wenfeng Feng, Guochao Jiang, Guofeng Quan, Guohua Liu, Yuewei Zhang - [[pdf]](https://arxiv.org/pdf/2604.08124)</summary>

**Abstract:** Reinforcement learning (RL) has become an effective approach for advancing the reasoning capabilities of large language models (LLMs) through the strategic integration of external search engines. However, current RL-based search agents often rely on a process of stochastic exploration guided by carefully crafted outcome rewards, leading to inefficient reasoning trajectories and unstable training. To address these issues, we propose a novel framework, Hierarchical Experience (HiExp), to enhance the performance and training stability of search agents. Specifically, we extract empirical knowledge through contrastive analysis and a multi-level clustering mechanism, transforming raw reasoning trajectories into hierarchical experience knowledge. By leveraging experience-aligned training, we effectively regularize stochastic exploration, evolving it into a strategic and experience-driven search process. Extensive evaluations on multiple complex agentic search and mathematical reasoning benchmarks demonstrate that our approach not only achieves substantial performance gains but also exhibits strong cross-task and cross-algorithm generalization.

**arXiv ID:** 2604.08124
</details>

<details>
<summary><strong>Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling</strong> - Jiaxuan Wang, Yulan Hu, Wenjin Yang, Zheng Pan, Xin Li, Lan-Zhe Guo - [[pdf]](https://arxiv.org/pdf/2604.08178)</summary>

**Abstract:** In classical Reinforcement Learning from Human Feedback (RLHF), Reward Models (RMs) serve as the fundamental signal provider for model alignment. As Large Language Models evolve into agentic systems capable of autonomous tool invocation and complex reasoning, the paradigm of reward modeling faces unprecedented challenges--most notably, the lack of benchmarks specifically designed to assess RM capabilities within tool-integrated environments. To address this gap, we present Plan-RewardBench, a trajectory-level preference benchmark designed to evaluate how well judges distinguish preferred versus distractor agent trajectories in complex tool-using scenarios. Plan-RewardBench covers four representative task families -- (i) Safety Refusal, (ii) Tool-Irrelevance / Unavailability, (iii) Complex Planning, and (iv) Robust Error Recovery -- comprising validated positive trajectories and confusable hard negatives constructed via multi-model natural rollouts, rule-based perturbations, and minimal-edit LLM perturbations. We benchmark representative RMs (generative, discriminative, and LLM-as-Judge) under a unified pairwise protocol, reporting accuracy trends across varying trajectory lengths and task categories. Furthermore, we provide diagnostic analyses of prevalent failure modes. Our results reveal that all three evaluator families face substantial challenges, with performance degrading sharply on long-horizon trajectories, underscoring the necessity for specialized training in agentic, trajectory-level reward modeling. Ultimately, Plan-RewardBench aims to serve as both a practical evaluation suite and a reusable blueprint for constructing agentic planning preference data.

**arXiv ID:** 2604.08178
</details>

<details>
<summary><strong>HiRO-Nav: Hybrid ReasOning Enables Efficient Embodied Navigation</strong> - He Zhao, Yijun Yang, Zichuan Lin, Deheng Ye, Chunyan Miao - [[pdf]](https://arxiv.org/pdf/2604.08232)</summary>

**Abstract:** Embodied navigation agents built upon large reasoning models (LRMs) can handle complex, multimodal environmental input and perform grounded reasoning per step to improve sequential decision-making for long-horizon tasks. However, a critical question remains: \textit{how can the reasoning capabilities of LRMs be harnessed intelligently and efficiently for long-horizon navigation tasks?} In simple scenes, agents are expected to act reflexively, while in complex ones they should engage in deliberate reasoning before this http URL achieve this, we introduce \textbf{H}ybr\textbf{i}d \textbf{R}eas\textbf{O}ning \textbf{Nav}igation (\textbf{HiRO-Nav}) agent, the first kind of agent capable of adaptively determining whether to perform thinking at every step based on its own action entropy. Specifically, by examining how the agent's action entropy evolves over the navigation trajectories, we observed that only a small fraction of actions exhibit high entropy, and these actions often steer the agent toward novel scenes or critical objects. Furthermore, studying the relationship between action entropy and task completion (i.e., Q-value) reveals that improving high-entropy actions contributes more positively to task this http URL, we propose a tailored training pipeline comprising hybrid supervised fine-tuning as a cold start, followed by online reinforcement learning with the proposed hybrid reasoning strategy to explicitly activate reasoning only for high-entropy actions, significantly reducing computational overhead while improving decision quality. Extensive experiments on the \textsc{CHORES}-$\mathbb{S}$ ObjectNav benchmark showcases that HiRO-Nav achieves a better trade-off between success rates and token efficiency than both dense-thinking and no-thinking baselines.

**arXiv ID:** 2604.08232
</details>

<details>
<summary><strong>On-board Telemetry Monitoring in Autonomous Satellites: Challenges and Opportunities</strong> - Lorenzo Capelli, Leandro de Souza Rosa, Maurizio De Tommasi, Livia Manovi, Andriy Enttsel, Mauro Mangia, Riccardo Rovatti, Ilaria Pinci, Carlo Ciancarelli, Eleonora Mariotti, Gianluca Furano - [[pdf]](https://arxiv.org/pdf/2604.08424)</summary>

**Abstract:** The increasing autonomy of spacecraft demands fault-detection systems that are both reliable and explainable. This work addresses eXplainable Artificial Intelligence for onboard Fault Detection, Isolation and Recovery within the Attitude and Orbit Control Subsystem by introducing a framework that enhances interpretability in neural anomaly detectors. We propose a method to derive low-dimensional, semantically annotated encodings from intermediate neural activations, called peepholes. Applied to a convolutional autoencoder, the framework produces interpretable indicators that enable the identification and localization of anomalies in reaction-wheel telemetry. Peepholes analysis further reveals bias detection and supports fault localization. The proposed framework enables the semantic characterization of detected anomalies while requiring only a marginal increase in computational resources, thus supporting its feasibility for on-board deployment.

**arXiv ID:** 2604.08424
</details>

<details>
<summary><strong>KnowU-Bench: Towards Interactive, Proactive, and Personalized Mobile Agent Evaluation</strong> - Tongbo Chen, Zhengxi Lu, Zhan Xu, Guocheng Shao, Shaohan Zhao, Fei Tang, Yong Du, Kaitao Song, Yizhou Liu, Yuchen Yan, Wenqi Zhang, Xu Tan, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen - [[pdf]](https://arxiv.org/pdf/2604.08455)</summary>

**Abstract:** Personalized mobile agents that infer user preferences and calibrate proactive assistance hold great promise as everyday digital assistants, yet existing benchmarks fail to capture what this requires. Prior work evaluates preference recovery from static histories or intent prediction from fixed contexts. Neither tests whether an agent can elicit missing preferences through interaction, nor whether it can decide when to intervene, seek consent, or remain silent in a live GUI environment. We introduce KnowU-Bench, an online benchmark for personalized mobile agents built on a reproducible Android emulation environment, covering 42 general GUI tasks, 86 personalized tasks, and 64 proactive tasks. Unlike prior work that treats user preferences as static context, KnowU-Bench hides the user profile from the agent and exposes only behavioral logs, forcing genuine preference inference rather than context lookup. To support multi-turn preference elicitation, it instantiates an LLM-driven user simulator grounded in structured profiles, enabling realistic clarification dialogues and proactive consent handling. Beyond personalization, KnowU-Bench provides comprehensive evaluation of the complete proactive decision chain, including grounded GUI execution, consent negotiation, and post-rejection restraint, evaluated through a hybrid protocol combining rule-based verification with LLM-as-a-Judge scoring. Our experiments reveal a striking degradation: agents that excel at explicit task execution fall below 50% under vague instructions requiring user preference inference or intervention calibration, even for frontier models like Claude Sonnet 4.6. The core bottlenecks are not GUI navigation but preference acquisition and intervention calibration, exposing a fundamental gap between competent interface operation and trustworthy personal assistance.

**arXiv ID:** 2604.08455
</details>

<details>
<summary><strong>SUPERNOVA: Eliciting General Reasoning in LLMs with Reinforcement Learning on Natural Instructions</strong> - Ashima Suvarna, Kendrick Phan, Mehrab Beikzadeh, Hritik Bansal, Saadia Gabriel - [[pdf]](https://arxiv.org/pdf/2604.08477)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved large language model (LLM) reasoning in formal domains such as mathematics and code. Despite these advancements, LLMs still struggle with general reasoning tasks requiring capabilities such as causal inference and temporal understanding. Extending RLVR to general reasoning is fundamentally constrained by the lack of high-quality, verifiable training data that spans diverse reasoning skills. To address this challenge, we propose SUPERNOVA, a data curation framework for RLVR aimed at enhancing general reasoning. Our key insight is that instruction-tuning datasets containing expert-annotated ground-truth encode rich reasoning patterns that can be systematically adapted for RLVR. To study this, we conduct 100+ controlled RL experiments to analyze how data design choices impact downstream reasoning performance. In particular, we investigate three key factors: (i) source task selection, (ii) task mixing strategies, and (iii) synthetic interventions for improving data quality. Our analysis reveals that source task selection is non-trivial and has a significant impact on downstream reasoning performance. Moreover, selecting tasks based on their performance for individual target tasks outperforms strategies based on overall average performance. Finally, models trained on SUPERNOVA outperform strong baselines (e.g., Qwen3.5) on challenging reasoning benchmarks including BBEH, Zebralogic, and MMLU-Pro. In particular, training on SUPERNOVA yields relative improvements of up to 52.8\% on BBEH across model sizes, demonstrating the effectiveness of principled data curation for RLVR. Our findings provide practical insights for curating human-annotated resources to extend RLVR to general reasoning. The code and data is available at this https URL.

**arXiv ID:** 2604.08477
</details>

<details>
<summary><strong>Reinforcement Learning with Reward Machines for Sleep Control in Mobile Networks</strong> - Kristina Levina, Nikolaos Pappas, Athanasios Karapantelakis, Aneta Vulgarakis Feljan, Jendrik Seipp - [[pdf]](https://arxiv.org/pdf/2604.07411)</summary>

**Abstract:** Energy efficiency in mobile networks is crucial for sustainable telecommunications infrastructure, particularly as network densification continues to increase power consumption. Sleep mechanisms for the components in mobile networks can reduce energy use, but deciding which components to put to sleep, when, and for how long while preserving quality of service (QoS) remains a difficult optimisation problem. In this paper, we utilise reinforcement learning with reward machines (RMs) to make sleep-control decisions that balance immediate energy savings and long-term QoS impact, i.e. time-averaged packet drop rates for deadline-constrained traffic and time-averaged minimum-throughput guarantees for constant-rate users. A challenge is that time-averaged constraints depend on cumulative performance over time rather than immediate performance. As a result, the effective reward is non-Markovian, and optimal actions depend on operational history rather than the instantaneous system state. RMs account for the history dependence by maintaining an abstract state that explicitly tracks the QoS constraint violations over time. Our framework provides a principled, scalable approach to energy management for next-generation mobile networks under diverse traffic patterns and QoS requirements.

**arXiv ID:** 2604.07411
</details>

<details>
<summary><strong>GIRL: Generative Imagination Reinforcement Learning via Information-Theoretic Hallucination Control</strong> - Prakul Sunil Hiremath - [[pdf]](https://arxiv.org/pdf/2604.07426)</summary>

**Abstract:** Model-based reinforcement learning (MBRL) improves sample efficiency by optimizing policies inside imagined rollouts, but long-horizon planning degrades when model errors compound and imagined trajectories drift off the training manifold. We introduce GIRL (Generative Imagination Reinforcement Learning), a latent world-model framework that addresses this failure mode with two key components. First, a cross-modal grounding signal derived from a frozen foundation model (DINOv2) anchors the latent transition prior to a semantically consistent embedding space, penalizing inconsistent or implausible predictions. Second, an uncertainty-adaptive trust-region bottleneck interprets the KL regularizer as the Lagrange multiplier of a constrained optimization problem, restricting imagination drift within a learned region calibrated by Expected Information Gain and a Relative Performance Loss signal.
We re-derive a value-gap bound using the Performance Difference Lemma and Integral Probability Metrics, yielding a bound that remains informative as the discount factor approaches one and connects the objective to real-environment regret. Experiments across three benchmark suites, including DeepMind Control, Adroit Hand Manipulation, and Meta-World with visual distractors, show that GIRL reduces latent rollout drift by 38 to 61 percent across tasks relative to DreamerV3, improves asymptotic return, and requires fewer environment interactions on long-horizon tasks. GIRL also outperforms TD-MPC2 on sparse-reward and high-contact settings under standard evaluation metrics. A distilled-prior variant reduces inference overhead and improves computational efficiency relative to the full model.

**arXiv ID:** 2604.07426
</details>

<details>
<summary><strong>Joint Task Offloading, Inference Optimization and UAV Trajectory Planning for Generative AI Empowered Intelligent Transportation Digital Twin</strong> - Xiaohuan Li, Junchuan Fan, Bingqi Zhang, Rong Yu, Xumin Huang, Qian Chen - [[pdf]](https://arxiv.org/pdf/2604.07687)</summary>

**Abstract:** To implement the intelligent transportation digital twin (ITDT), unmanned aerial vehicles (UAVs) are scheduled to process the sensing data from the roadside sensors. At this time, generative artificial intelligence (GAI) technologies such as diffusion models are deployed on the UAVs to transform the raw sensing data into the high-quality and valuable. Therefore, we propose the GAI-empowered ITDT. The dynamic processing of a set of diffusion model inference (DMI) tasks on the UAVs with dynamic mobility simultaneously influences the DT updating fidelity and delay. In this paper, we investigate a joint optimization problem of DMI task offloading, inference optimization and UAV trajectory planning as the system utility maximization (SUM) problem to address the fidelity-delay tradeoff for the GAI-empowered ITDT. To seek a solution to the problem under the network dynamics, we model the SUM problem as the heterogeneous-agent Markov decision process, and propose the sequential update-based heterogeneous-agent twin delayed deep deterministic policy gradient (SU-HATD3) algorithm, which can quickly learn a near-optimal solution. Numerical results demonstrate that compared with several baseline algorithms, the proposed algorithm has great advantages in improving the system utility and convergence rate.

**arXiv ID:** 2604.07687
</details>

<details>
<summary><strong>PriPG-RL: Privileged Planner-Guided Reinforcement Learning for Partially Observable Systems with Anytime-Feasible MPC</strong> - Mohsen Amiri, Mohsen Amiri, Ali Beikmohammadi, Sindri Magnuśson, Mehdi Hosseinzadeh - [[pdf]](https://arxiv.org/pdf/2604.08036)</summary>

**Abstract:** This paper addresses the problem of training a reinforcement learning (RL) policy under partial observability by exploiting a privileged, anytime-feasible planner agent available exclusively during training. We formalize this as a Partially Observable Markov Decision Process (POMDP) in which a planner agent with access to an approximate dynamical model and privileged state information guides a learning agent that observes only a lossy projection of the true state. To realize this framework, we introduce an anytime-feasible Model Predictive Control (MPC) algorithm that serves as the planner agent. For the learning agent, we propose Planner-to-Policy Soft Actor-Critic (P2P-SAC), a method that distills the planner agent's privileged knowledge to mitigate partial observability and thereby improve both sample efficiency and final policy performance. We support this framework with rigorous theoretical analysis. Finally, we validate our approach in simulation using NVIDIA Isaac Lab and successfully deploy it on a real-world Unitree Go2 quadruped navigating complex, obstacle-rich environments.

**arXiv ID:** 2604.08036
</details>

<details>
<summary><strong>Evaluation as Evolution: Transforming Adversarial Diffusion into Closed-Loop Curricula for Autonomous Vehicles</strong> - Yicheng Guo, Jiaqi Liu, Chengkai Xu, Peng Hang, Jian Sun - [[pdf]](https://arxiv.org/pdf/2604.07378)</summary>

**Abstract:** Autonomous vehicles in interactive traffic environments are often limited by the scarcity of safety-critical tail events in static datasets, which biases learned policies toward average-case behaviors and reduces robustness. Existing evaluation methods attempt to address this through adversarial stress testing, but are predominantly open-loop and post-hoc, making it difficult to incorporate discovered failures back into the training process. We introduce Evaluation as Evolution ($E^2$), a closed-loop framework that transforms adversarial generation from a static validation step into an adaptive evolutionary curriculum. Specifically, $E^2$ formulates adversarial scenario synthesis as transport-regularized sparse control over a learned reverse-time SDE prior. To make this high-dimensional generation tractable, we utilize topology-driven support selection to identify critical interacting agents, and introduce Topological Anchoring to stabilize the process. This approach enables the targeted discovery of failure cases while strictly constraining deviations from realistic data distributions. Empirically, $E^2$ improves collision failure discovery by 9.01% on the nuScenes dataset and up to 21.43% on the nuPlan dataset over the strongest baselines, while maintaining low invalidity and high realism. It further yields substantial robustness gains when the resulting boundary cases are recycled for closed-loop policy fine-tuning.

**arXiv ID:** 2604.07378
</details>

<details>
<summary><strong>Reset-Free Reinforcement Learning for Real-World Agile Driving: An Empirical Study</strong> - Kohei Honda, Hirotaka Hosogaya - [[pdf]](https://arxiv.org/pdf/2604.07672)</summary>

**Abstract:** This paper presents an empirical study of reset-free reinforcement learning (RL) for real-world agile driving, in which a physical 1/10-scale vehicle learns continuously on a slippery indoor track without manual resets. High-speed driving near the limits of tire friction is particularly challenging for learning-based methods because complex vehicle dynamics, actuation delays, and other unmodeled effects hinder both accurate simulation and direct sim-to-real transfer of learned policies. To enable autonomous training on a physical platform, we employ Model Predictive Path Integral control (MPPI) as both the reset policy and the base policy for residual learning, and systematically compare three representative RL algorithms, i.e., PPO, SAC, and TD-MPC2, with and without residual learning in simulation and real-world experiments. Our results reveal a clear gap between simulation and real-world: SAC with residual learning achieves the highest returns in simulation, yet only TD-MPC2 consistently outperforms the MPPI baseline on the physical platform. Moreover, residual learning, while clearly beneficial in simulation, fails to transfer its advantage to the real world and can even degrade performance. These findings reveal that reset-free RL in the real world poses unique challenges absent from simulation, calling for further algorithmic development tailored to training in the wild.

**arXiv ID:** 2604.07672
</details>

<details>
<summary><strong>RoboAgent: Chaining Basic Capabilities for Embodied Task Planning</strong> - Peiran Xu, Jiaqi Zheng, Yadong Mu - [[pdf]](https://arxiv.org/pdf/2604.07774)</summary>

**Abstract:** This paper focuses on embodied task planning, where an agent acquires visual observations from the environment and executes atomic actions to accomplish a given task. Although recent Vision-Language Models (VLMs) have achieved impressive results in multimodal understanding and reasoning, their performance remains limited when applied to embodied planning that involves multi-turn interaction, long-horizon reasoning, and extended context analysis. To bridge this gap, we propose RoboAgent, a capability-driven planning pipeline in which the model actively invokes different sub-capabilities. Each capability maintains its own context, and produces intermediate reasoning results or interacts with the environment according to the query given by a scheduler. This framework decomposes complex planning into a sequence of basic vision-language problems that VLMs can better address, enabling a more transparent and controllable reasoning process. The scheduler and all capabilities are implemented with a single VLM, without relying on external tools. To train this VLM, we adopt a multi-stage paradigm that consists of: (1) behavior cloning with expert plans, (2) DAgger training using trajectories collected by the model, and (3) reinforcement learning guided by an expert policy. Across these stages, we exploit the internal information of the environment simulator to construct high-quality supervision for each capability, and we further introduce augmented and synthetic data to enhance the model's performance in more diverse scenarios. Extensive experiments on widely used embodied task planning benchmarks validate the effectiveness of the proposed approach. Our codes will be available at this https URL.

**arXiv ID:** 2604.07774
</details>

<details>
<summary><strong>On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning</strong> - Amirhossein Afsharrad, Amirhesam Abedsoltan, Ahmadreza Moradipari, Sanjay Lall - [[pdf]](https://arxiv.org/pdf/2604.07944)</summary>

**Abstract:** Large language models (LLMs) have recently demonstrated strong potential for autonomous vehicle motion planning by reformulating trajectory prediction as a language generation problem. However, deploying capable LLMs in resource-constrained onboard systems remains a fundamental challenge. In this paper, we study how to effectively transfer motion planning knowledge from a large teacher LLM to a smaller, more deployable student model. We build on the GPT-Driver framework, which represents driving scenes as language prompts and generates waypoint trajectories with chain-of-thought reasoning, and investigate two student training paradigms: (i) on-policy generalized knowledge distillation (GKD), which trains the student on its own self-generated outputs using dense token-level feedback from the teacher, and (ii) a dense-feedback reinforcement learning (RL) baseline that uses the teacher's log-probabilities as per-token reward signals in a policy gradient framework. Experiments on the nuScenes benchmark show that GKD substantially outperforms the RL baseline and closely approaches teacher-level performance despite a 5$\times$ reduction in model size. These results highlight the practical value of on-policy distillation as a principled and effective approach to deploying LLM-based planners in autonomous driving systems.

**arXiv ID:** 2604.07944
</details>

<details>
<summary><strong>Incremental Residual Reinforcement Learning Toward Real-World Learning for Social Navigation</strong> - Haruto Nagahisa, Kohei Matsumoto, Yuki Tomita, Yuki Hyodo, Ryo Kurazume - [[pdf]](https://arxiv.org/pdf/2604.07945)</summary>

**Abstract:** As the demand for mobile robots continues to increase, social navigation has emerged as a critical task, driving active research into deep reinforcement learning (RL) approaches. However, because pedestrian dynamics and social conventions vary widely across different regions, simulations cannot easily encompass all possible real-world scenarios. Real-world RL, in which agents learn while operating directly in physical environments, presents a promising solution to this issue. Nevertheless, this approach faces significant challenges, particularly regarding constrained computational resources on edge devices and learning efficiency. In this study, we propose incremental residual RL (IRRL). This method integrates incremental learning, which is a lightweight process that operates without a replay buffer or batch updates, with residual RL, which enhances learning efficiency by training only on the residuals relative to a base policy. Through the simulation experiments, we demonstrated that, despite lacking a replay buffer, IRRL achieved performance comparable to those of conventional replay buffer-based methods and outperformed existing incremental learning approaches. Furthermore, the real-world experiments confirmed that IRRL can enable robots to effectively adapt to previously unseen environments through the real-world learning.

**arXiv ID:** 2604.07945
</details>

<details>
<summary><strong>Open-Ended Instruction Realization with LLM-Enabled Multi-Planner Scheduling in Autonomous Vehicles</strong> - Jiawei Liu, Xun Gong, Fen Fang, Muli Yang, Bohao Qu, Yunfeng Hu, Hong Chen, Xulei Yang, Qing Guo - [[pdf]](https://arxiv.org/pdf/2604.08031)</summary>

**Abstract:** Most Human-Machine Interaction (HMI) research overlooks the maneuvering needs of passengers in autonomous driving (AD). Natural language offers an intuitive interface, yet translating passenger open-ended instructions into control signals, without sacrificing interpretability and traceability, remains a challenge. This study proposes an instruction-realization framework that leverages a large language model (LLM) to interpret instructions, generates executable scripts that schedule multiple model predictive control (MPC)-based motion planners based on real-time feedback, and converts planned trajectories into control signals. This scheduling-centric design decouples semantic reasoning from vehicle control at different timescales, establishing a transparent, traceable decision-making chain from high-level instructions to low-level actions. Due to the absence of high-fidelity evaluation tools, this study introduces a benchmark for open-ended instruction realization in a closed-loop setting. Comprehensive experiments reveal that the framework significantly improves task-completion rates over instruction-realization baselines, reduces LLM query costs, achieves safety and compliance on par with specialized AD approaches, and exhibits considerable tolerance to LLM inference latency. For more qualitative illustrations and a clearer understanding.

**arXiv ID:** 2604.08031
</details>

<details>
<summary><strong>ViVa: A Video-Generative Value Model for Robot Reinforcement Learning</strong> - Jindi Lv, Hao Li, Jie Li, Yifei Nie, Fankun Kong, Yang Wang, Xiaofeng Wang, Zheng Zhu, Chaojun Ni, Qiuping Deng, Hengtao Li, Jiancheng Lv, Guan Huang - [[pdf]](https://arxiv.org/pdf/2604.08168)</summary>

**Abstract:** Vision-language-action (VLA) models have advanced robot manipulation through large-scale pretraining, but real-world deployment remains challenging due to partial observability and delayed feedback. Reinforcement learning addresses this via value functions, which assess task progress and guide policy improvement. However, existing value models built on vision-language models (VLMs) struggle to capture temporal dynamics, undermining reliable value estimation in long-horizon tasks. In this paper, we propose ViVa, a video-generative value model that repurposes a pretrained video generator for value estimation. Taking the current observation and robot proprioception as input, ViVa jointly predicts future proprioception and a scalar value for the current state. By leveraging the spatiotemporal priors of a pretrained video generator, our approach grounds value estimation in anticipated embodiment dynamics, moving beyond static snapshots to intrinsically couple value with foresight. Integrated into RECAP, ViVa delivers substantial improvements on real-world box assembly. Qualitative analysis across all three tasks confirms that ViVa produces more reliable value signals, accurately reflecting task progress. By leveraging spatiotemporal priors from video corpora, ViVa also generalizes to novel objects, highlighting the promise of video-generative models for value estimation.

**arXiv ID:** 2604.08168
</details>

<details>
<summary><strong>Visually-grounded Humanoid Agents</strong> - Hang Ye, Xiaoxuan Ma, Fan Lu, Wayne Wu, Kwan-Yee Lin, Yizhou Wang - [[pdf]](https://arxiv.org/pdf/2604.08509)</summary>

**Abstract:** Digital human generation has been studied for decades and supports a wide range of real-world applications. However, most existing systems are passively animated, relying on privileged state or scripted control, which limits scalability to novel environments. We instead ask: how can digital humans actively behave using only visual observations and specified goals in novel scenes? Achieving this would enable populating any 3D environments with digital humans at scale that exhibit spontaneous, natural, goal-directed behaviors. To this end, we introduce Visually-grounded Humanoid Agents, a coupled two-layer (world-agent) paradigm that replicates humans at multiple levels: they look, perceive, reason, and behave like real people in real-world 3D scenes. The World Layer reconstructs semantically rich 3D Gaussian scenes from real-world videos via an occlusion-aware pipeline and accommodates animatable Gaussian-based human avatars. The Agent Layer transforms these avatars into autonomous humanoid agents, equipping them with first-person RGB-D perception and enabling them to perform accurate, embodied planning with spatial awareness and iterative reasoning, which is then executed at the low level as full-body actions to drive their behaviors in the scene. We further introduce a benchmark to evaluate humanoid-scene interaction in diverse reconstructed environments. Experiments show our agents achieve robust autonomous behavior, yielding higher task success rates and fewer collisions than ablations and state-of-the-art planning methods. This work enables active digital human population and advances human-centric embodied AI. Data, code, and models will be open-sourced.

**arXiv ID:** 2604.08509
</details>

<details>
<summary><strong>NaviSplit: Dynamic Multi-Branch Split DNNs for Efficient Distributed Autonomous Navigation</strong> - Timothy K Johnsen, Ian Harshbarger, Zixia Xia, Marco Levorato - [[pdf]](https://arxiv.org/pdf/2406.13086)</summary>

**Abstract:** Lightweight autonomous unmanned aerial vehicles (UAV) are emerging as a central component of a broad range of applications. However, autonomous navigation necessitates the implementation of perception algorithms, often deep neural networks (DNN), that process the input of sensor observations, such as that from cameras and LiDARs, for control logic. The complexity of such algorithms clashes with the severe constraints of these devices in terms of computing power, energy, memory, and execution time. In this paper, we propose NaviSplit, the first instance of a lightweight navigation framework embedding a distributed and dynamic multi-branched neural model. At its core is a DNN split at a compression point, resulting in two model parts: (1) the head model, that is executed at the vehicle, which partially processes and compacts perception from sensors; and (2) the tail model, that is executed at an interconnected compute-capable device, which processes the remainder of the compacted perception and infers navigation commands. Different from prior work, the NaviSplit framework includes a neural gate that dynamically selects a specific head model to minimize channel usage while efficiently supporting the navigation network. In our implementation, the perception model extracts a 2D depth map from a monocular RGB image captured by the drone using the robust simulator Microsoft AirSim. Our results demonstrate that the NaviSplit depth model achieves an extraction accuracy of 72-81% while transmitting an extremely small amount of data (1.2-18 KB) to the edge server. When using the neural gate, as utilized by NaviSplit, we obtain a slightly higher navigation accuracy as compared to a larger static network by 0.3% while significantly reducing the data rate by 95%. To the best of our knowledge, this is the first exemplar of dynamic multi-branched model based on split DNNs for autonomous navigation.

**arXiv ID:** 2406.13086
</details>

<details>
<summary><strong>Pseudo-Expert Regularized Offline RL for End-to-End Autonomous Driving in Photorealistic Closed-Loop Environments</strong> - Chihiro Noguchi, Takaki Yamamoto - [[pdf]](https://arxiv.org/pdf/2512.18662)</summary>

**Abstract:** End-to-end (E2E) autonomous driving models that take only camera images as input and directly predict a future trajectory are appealing for their computational efficiency and potential for improved generalization via unified optimization; however, persistent failure modes remain due to reliance on imitation learning (IL). While online reinforcement learning (RL) could mitigate IL-induced issues, the computational burden of neural rendering-based simulation and large E2E networks renders iterative reward and hyperparameter tuning costly. We introduce a camera-only E2E offline RL framework that performs no additional exploration and trains solely on a fixed simulator dataset. Offline RL offers strong data efficiency and rapid experimental iteration, yet is susceptible to instability from overestimation on out-of-distribution (OOD) actions. To address this, we construct pseudo ground-truth trajectories from expert driving logs and use them as a behavior regularization signal, suppressing imitation of unsafe or suboptimal behavior while stabilizing value learning. Training and closed-loop evaluation are conducted in a neural rendering environment learned from the public nuScenes dataset. Empirically, the proposed method achieves substantial improvements in collision rate and route completion compared with IL baselines. Our code is available at this https URL.

**arXiv ID:** 2512.18662
</details>

<details>
<summary><strong>COSMIC: Emotionally Intelligent Agents to Support Mental and Emotional Well-being in Extreme Isolation: Lessons from Analog Astronaut Training Missions</strong> - A. Xygkou-Tsiamoulou, Alexandra Covaci, Zeqi Jia, Jenny Yiend, Chee Siang Ang - [[pdf]](https://arxiv.org/pdf/2604.07589)</summary>

**Abstract:** As humanity pivots toward long-duration interplanetary travel, the psychological constraints of Isolated and Confined Environments (ICE) emerge as a primary mission risk. This paper presents COSMIC (COmpanion System for Mission Interaction and Communication) representing the inaugural investigation into the deployment of a high-fidelity, emotionally intelligent AI companion in an analog astronaut setting. By integrating a Large Language Model (LLM) architecture with a diffusion-based digital avatar interface, COSMIC transcends traditional task-oriented automation to provide longitudinal affective support. We detail a modular system architecture designed for temporal continuity through short- and long-term memory systems and outline a robust naturalistic observational framework for evaluating psychological resilience at the LunAres Research Station. This work constitutes the first formal submission in the field to evaluate the efficacy of state-of-the-art generative AI and synthesized visual empathy in mitigating the effects of extreme isolation.

**arXiv ID:** 2604.07589
</details>

<details>
<summary><strong>Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh for Empowering Access to Justice</strong> - Azmine Toushik Wasi, Wahid Faisal, Mst Rafia Islam, Md Rizwan Parvez - [[pdf]](https://arxiv.org/pdf/2511.08605)</summary>

**Abstract:** Bangladesh's low-income population faces major barriers to affordable legal advice due to complex legal language, procedural opacity, and high costs. Existing AI legal assistants lack Bengali-language support and jurisdiction-specific adaptation, limiting their effectiveness. To address this, we developed Mina, a multilingual LLM-based legal assistant tailored for the Bangladeshi context. It employs multilingual embeddings and a RAG-based chain-of-tools framework for retrieval, reasoning, translation, and document generation, delivering context-aware legal drafts, citations, and plain-language explanations via an interactive chat interface. Evaluated by law faculty from leading Bangladeshi universities across all stages of the 2022 and 2023 Bangladesh Bar Council Exams, Mina scored 75-80% in Preliminary MCQs, Written, and simulated Viva Voce exams, matching or surpassing average human performance and demonstrating clarity, contextual understanding, and sound legal reasoning. Even under a conservative upper bound, Mina operates at just 0.12-0.61% of typical legal consultation costs in Bangladesh, yielding a 99.4-99.9\% cost reduction relative to human-provided services. These results confirm its potential as a low-cost, multilingual AI assistant that automates key legal tasks and scales access to justice, offering a real-world case study on building domain-specific, low-resource systems and addressing challenges of multilingual adaptation, efficiency, and sustainable public-service AI deployment.

**arXiv ID:** 2511.08605
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
