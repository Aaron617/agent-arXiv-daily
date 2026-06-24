# Agent arXiv Daily

**Last Updated:** 2026-06-24 05:11:51

**Total Papers:** 84

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
<summary><strong>The Latent Bridge: A Continuous Slow-Fast Channel for Real-Time Game Agents</strong> - Bojie Li, Noah Shi - [[pdf]](https://arxiv.org/pdf/2606.24470)</summary>

**Abstract:** A real-time agent for general computer use - with games as the most demanding case - must act within tens of milliseconds while still planning over seconds. These two regimes sit at opposite ends of the latency-quality tradeoff. A reasoning VLM (Qwen3-VL-8B-Thinking) deliberates effectively but requires ~1.5 s per response - far too slow for a 15 Hz control loop. In contrast, a reactive VLM (MiniCPM-o 4.5) acts in milliseconds but underperforms on planning-heavy tasks. We couple two frozen models of matched scale (9B reactive, 8B reasoning), leaving the communication channel as the sole trainable component. The standard coupling is a Text Bridge (T): the slow model writes a suffix the fast model reads. We introduce a learned continuous Latent Bridge (L) that projects the slow model's residuals into the fast model's input-embedding space in a LLaVA-style manner, avoiding any text round-trip; both are compared against Fast-Only (F). On 7 Atari games and a driving domain (MetaDrive), tuning the action decoder per channel on held-out seeds, the Latent Bridge matches or beats the Text Bridge in every domain: it significantly improves two games (MsPacman +57%, RoadRunner +28%) and is a safe drop-in elsewhere. Combining both channels interferes destructively (RoadRunner -96%), so only one should be used. The benefit is highly predictable: the bridge helps if and only if slow reasoning already beats fast reaction (T > F) - the Latent and Text gains over Fast-Only move together at r=0.93. MetaDrive is the controlled negative, where the Latent Bridge is demonstrably inert because the Text Bridge adds no value. We release replay recordings and reproducible pipelines.

**arXiv ID:** 2606.24470
</details>

<details>
<summary><strong>Engineering Reliable Autonomous Systems: Challenges and Solutions</strong> - Marie Farrell, Matt Luckcuck, Angelo Ferrando, Rafael C. Cardoso, Natasha Alechina, Marco Autili, Diana Benjumea Hernandez, Luciana Brasil Rebelo dos Santos, Daniela Briola, Ana Cavalcanti, Christian Colombo, Louise A. Dennis, Clare Dixon, Michael Fisher, Mario Gleirscher, Taylor Johnson, Charles Lesire, Livia Lestingi, Sven Linker, Brian Logan, Colin Paterson, Fabio Papacchini, Patrizio Pelliccione, Pedro Ribeiro, Maike Schwammberger, Silvia Lizeth Tapia Tarifa, Hazel Taylor, Jim Woodcock, Mengwei Xu, Yi Yang, Huan Zhang - [[pdf]](https://arxiv.org/pdf/2606.23760)</summary>

**Abstract:** Engineering reliable autonomous systems is an important and growing topic in computer science. As autonomous systems become more prevalent, easy-to-use techniques for building them reliably are increasingly important.
This workshop report captures and expands on the discussions at the Lorentz Center Workshop "Engineering Reliable Autonomous Systems" (ERAS), held from 10 to 14 June 2024. The workshop was co-organised by the organisers of the Workshop on Formal Methods for Autonomous Systems (FMAS) and the Workshop on Agents and Robots for reliable Engineered Autonomy (AREA). It brought together members of the FMAS and AREA communities, industry practitioners, and representatives from sectors where autonomous systems pose distinctive engineering challenges.
The workshop focused on three main research topics: techniques for verification and validation of autonomous systems; engineering real-world autonomous systems; and software architectures for safe autonomous systems. Its main outcome is a catalogue of challenges in these areas and, most importantly, a pathway to solutions. Some challenges can already be tackled by techniques that are well known in academia but have not yet become regularly used in practice. Other challenges remain unresolved and require further research. This roadmap is intended to support future research and industrial collaboration.

**arXiv ID:** 2606.23760
</details>

<details>
<summary><strong>Paying to Know: Micro-Transaction Markets for Verified Product Information in Agentic E-Commerce</strong> - Filippos Ventirozos, Matthew Shardlow - [[pdf]](https://arxiv.org/pdf/2606.24783)</summary>

**Abstract:** Commercial NLP treats the shopping chatbot as a recommender or a conversion tool: its job is to match a user to a catalogue entry and close a sale. We argue that the arrival of agent-native micro-payment rails (e.g., x402, AP2) changes what is scarce. When the buyer is an autonomous agent that can investigate exhaustively, the bottleneck is no longer matching products but acquiring trustworthy, decision-relevant information about them. We envision agentic e-commerce as a micro-transaction market for verified information: buyer agents spend fractions of a cent to progressively unlock seller- and reviewer-supplied data -- service histories, third-party test reports, bills of materials, audited sales and support metrics -- paid for a la carte under a freemium model, with reviewer trust scored reputationally. We sketch the architecture of such a market and argue that it rewards genuine product quality and yields truer competition than ranking-based storefronts. We then translate the vision into concrete NLP problems -- cost-optimal information acquisition, data pricing and negotiation, real-time entity resolution, grounded value exchange, and privacy-preserving persona modelling -- and argue that these, not chat fluency, deserve the field's attention.

**arXiv ID:** 2606.24783
</details>

<details>
<summary><strong>"Zooming In" on Agentic Web Browsers as Assistive Technologies: A Case Study with a Low-Vision Technology Expert</strong> - Laura Colazzo, Giuseppe Anzillotti - [[pdf]](https://arxiv.org/pdf/2606.24870)</summary>

**Abstract:** Agentic Web Browsers (AWBs), powered by Large Language Models (LLMs), are emerging as autonomous systems capable of navigating the Web on behalf of users. Beyond enhancing productivity, they could also offer significant promise as Assistive Technologies (ATs) for visually-impaired individuals, transforming web interaction into a fluid conversational exchange. In this paper, we present a case study with a low-vision technology expert, examining how AWBs can support visually-impaired users in web navigation. The findings show that, despite the current limitations, the navigation experience is notably fluid and flexible, underscoring the strong potential of AWBs to enhance accessibility and reduce barriers in web interaction, with implications that may extend beyond accessibility to agentic UX more broadly.

**arXiv ID:** 2606.24870
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (23 papers)</h2></summary>

<details>
<summary><strong>ReMMD: Realistic Multilingual Multi-Image Agentic Verification for Multimodal Misinformation Detection</strong> - Chenhao Dang, Dantong Zhu, Jun Yang, Conghui He, Weijia Li - [[pdf]](https://arxiv.org/pdf/2606.24112)</summary>

**Abstract:** Multimodal misinformation detection is increasingly important because viral posts now combine long multilingual narratives, several images, mixed provenance, and subtle text--image framing errors. Existing benchmarks and methods remain poorly matched to this setting: they usually isolate short captions, single images, binary labels, or one manipulation source, while agentic verification remains costly under realistic evidence search. We present ReMMD, a realistic multilingual multi-image agentic verification framework for multimodal misinformation detection. ReMMD includes ReMMDBench, a real-world multimodal misinformation detection benchmark with 500 samples, 2,756 images, five monolingual languages, two cross-lingual settings, three text-length tiers, multi-image posts, five-way veracity labels, eight distortion labels, evidence provenance, and rationales. It also includes ReMMD-Agent, a persistent-memory verifier that decomposes posts into atomic points, builds a reusable evidence set, and predicts structured L1/L2/L3 outputs. Across proprietary systems, open LVLMs, MMD-Agent, and T2-Agent, ReMMD-Agent obtains the best five-way veracity performance, with 41.80% accuracy and 39.12% macro-F1 using GPT-5.2, while reducing cost by 17.5% relative to MMD-Agent and 79.9% relative to T2-Agent. The project is available at this https URL.

**arXiv ID:** 2606.24112
</details>

<details>
<summary><strong>OmniPath: A Multi-Modal Agentic Framework for Auditing Wheelchair Accessibility</strong> - ASM Mobarak Hossain, Nadim Mahmud, Vaskar Raychoudhury, Md Osman Gani - [[pdf]](https://arxiv.org/pdf/2606.24129)</summary>

**Abstract:** For a wheelchair user, a standard blue line on a map is often a broken promise. While platforms like OpenStreetMap (OSM) successfully capture where a path is, they frequently fail to convey how it physically feels to travel on it. This information barrier is problematic for wheelchair users. To solve this issue, we present OmniPath, a system that moves from passive mapping to proactive environmental auditing. Our framework fuses the network topology of OSM with the submeter precision of high-density aerial LiDAR (USGS 3DEP) to create a high-fidelity 3D model of the pedestrian environment. Rather than simply routing a user, our agent virtually traverses the network, analyzing the surface in 0.5 meter increments. It rigorously quantifies physical friction points specifically running slope, cross slope, and vertical discontinuities against ADA compliance standards, calculating a weighted severity score to categorize hazards from ``Mild'' to ``Critical.'' To ensure real world reliability, we validated the system against 200 physical ground truth field surveys across the National Mall using stratified random sampling. The framework demonstrated strong diagnostic reliability for high-severity hazards, achieving F1-scores of 0.60 for Severe and 0.58 for critical categories. By automating this micro-scale inspection, OmniPath identifies the ``invisible'' barriers that standard maps miss, effectively transforming a static dataset into accessibility data source that anticipates accessibility challenges before the user ever leaves home.

**arXiv ID:** 2606.24129
</details>

<details>
<summary><strong>SP-Mind: An Autonomous Reasoning Agent for Spatial Proteomics Analysis</strong> - Yucheng Yuan, Yuanfeng Ji, Zhongxiao Li, Ruijiang Li - [[pdf]](https://arxiv.org/pdf/2606.24235)</summary>

**Abstract:** Spatial proteomics enables single-cell-resolution characterization of protein expression within tissue architecture, playing a critical role in understanding tumor microenvironments and guiding precision medicine. However, current analysis workflows remain fragmented, requiring expert manual orchestration of heterogeneous tools and limiting research scalability and reproducibility. We present SP-Mind, the first autonomous AI agent designed to unify the spatial proteomics analysis pipeline, from raw multiplexed tissue imaging to downstream phenotype discovery. Equipped with expert-curated biological analysis skills and specialized computational tools, SP-Mind converts natural-language queries into end-to-end analytical workflows without task-specific fine-tuning. To rigorously evaluate its capabilities, we introduce SP-Bench, a comprehensive benchmark spanning diverse tissue types, comprising 102 tasks across 18 distinct categories. Through extensive evaluation on SP-Bench and established downstream tasks, SP-Mind achieves state-of-the-art performance compared to existing open-source biomedical agent baselines.

**arXiv ID:** 2606.24235
</details>

<details>
<summary><strong>Bayesian control for coding agents</strong> - Theodore Papamarkou, Vladislav Smirnov, Viktor Mazanov, Artem Vazhentsev, Preslav Nakov, Timothy Baldwin, Artem Shelmanov - [[pdf]](https://arxiv.org/pdf/2606.24453)</summary>

**Abstract:** Modern coding agents pair LLM generators with various tools, including cheap diagnostics and expensive verifiers. The tool-use decisions are typically governed by orchestrators that often use fixed rules and ignore uncertainty. We formulate orchestration as cost-sensitive sequential hypothesis testing: a Bayesian controller maintains a belief over candidate correctness and dynamically decides whether to gather more evidence, refine the candidate, verify it, or stop. Across six generators and nine coding benchmarks, Bayesian control proves to be most valuable when verification is costly and critics are informative but imperfect. Beyond control, the belief state yields an interpretable correctness score that outperforms token-probability and raw tool-success baselines for uncertainty quantification.

**arXiv ID:** 2606.24453
</details>

<details>
<summary><strong>GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents</strong> - Xiao Zhou, Siyue Zhang, Yilun Zhao, Jinbiao Wei, Tingyu Song, Arman Cohan, Chen Zhao - [[pdf]](https://arxiv.org/pdf/2606.24551)</summary>

**Abstract:** Computer-use agents can execute software tasks through either graphical interfaces or programmatic command interfaces, but existing evaluations confound interaction modality with differences in tasks, initial states, verifiers, and permitted actions. We introduce a matched execution-layer benchmark of 440 desktop tasks across 18 applications and 12 workflow categories, where screen-only GUI agents and skill-mediated CLI agents receive identical goals, states, and final-state verifiers while being restricted to modality-native actions. In this controlled setting, the strongest GUI agent reaches a 59.1% full pass rate, outperforming the strongest original-skill CLI agent at 48.2%; however, verifier-guided skill augmentation raises CLI success to 69.3%, showing that much of the CLI deficit comes from incomplete skill coverage rather than model capability alone. These results suggest that GUI and CLI expose different execution bottlenecks: GUI agents are limited by reliable grounded interaction over long-horizon workflows, whereas CLI agents are limited by the coverage and scalability of their skill interfaces.

**arXiv ID:** 2606.24551
</details>

<details>
<summary><strong>When Retrieval Metrics Mislead: Measuring Policy Signal in Long-Horizon Tool-Use Agents</strong> - Tianyu Ding, Juan Pablo De la Cruz Weinstein - [[pdf]](https://arxiv.org/pdf/2606.23937)</summary>

**Abstract:** Exact-match retrieval recall is often used as a proxy for whether a retriever supplies useful policy context to a downstream decision model. We test this proxy for pre-action policy classification in tau-bench using Qwen2.5-3B/7B classifiers. Under gold-policy conditioning, a compact structured state improves macro-F1 over raw trajectories by 0.13-0.17 after tuning. We then replace the benchmark-designated policy clause with the top-ranked clause retrieved from decision-time context. Although the exact governing clause is retrieved at rank 1 for only 7% of airline states, the primary 3B classifier obtains macro-F1 0.58 with retrieved clauses versus 0.60 with gold clauses (Delta=-0.02, task-cluster 95% CI [-0.23,+0.21]); mismatched-policy and no-policy controls score 0.32 and 0.21. We do not detect a macro-F1 difference between retrieved and gold clauses in this configuration, although the interval remains too wide to establish non-inferiority. The same qualitative pattern appears with a second retriever and at 7B, while varying across fine-tuning configurations. These results indicate that exact-match clause recall can underestimate downstream policy utility in this benchmark setting, motivating evaluation with retrieved policies in the classification loop rather than recall alone.

**arXiv ID:** 2606.23937
</details>

<details>
<summary><strong>Metis: Bridging Text and Code Memory for Self-Evolving Agents</strong> - Zijie Dai, Siuhin He, Hui Li, Qihui Zhou, Jiajun Li, Mingcong Song, Guoping Long, Hongjie Si, Xin Yao, Lin Zhang, James Cheng, Xiao Yan - [[pdf]](https://arxiv.org/pdf/2606.24151)</summary>

**Abstract:** Self-evolving agents improve over time by distilling experience from past executions and reusing it in future tasks. Existing systems represent such experience either as natural-language text injected into the agent context or as code exposed as callable tools. However, the choice between these representations is typically made at design time rather than derived from the characteristics of the experience itself, leaving the trade-offs between them poorly understood. We present the first controlled study that isolates text memory and code memory over an identical set of experiences. Our results show that the two forms exhibit complementary trade-offs in construction cost, execution efficiency, and transferability, such that neither representation alone is sufficient. Guided by these findings, we propose Metis, a self-evolving agent system built on a hierarchical dual-representation memory. Metis organizes textual experience into execution plans, environment facts, and common pitfalls, and selectively crystallizes recurring plans into validated callable tools. This design combines the broad applicability of text memory with the execution efficiency of code memory while incurring tool-generation cost only when justified by repeated reuse. We evaluate Metis on AppWorld, a challenging benchmark for interactive agents. The results show that Metis improves task accuracy by up to 20.6% over ReAct while reducing execution cost by up to 22.8%. Compared with representative self-evolving agent systems, Metis consistently achieves a better balance between accuracy, execution efficiency, and memory-construction cost.

**arXiv ID:** 2606.24151
</details>

<details>
<summary><strong>UniDrive: A Unified Vision-Language and Grounding Framework for Interpretable Risk Understanding in Autonomous Driving</strong> - Xiaowei Gao, Pengxiang Li, Yitai Cheng, Ruihan Xu, James Haworth, Stephen Law, Yun Ye - [[pdf]](https://arxiv.org/pdf/2606.24759)</summary>

**Abstract:** Recent multimodal large language models (MLLMs) have shown strong potential for autonomous driving scene understanding, yet existing methods still face a fundamental trade-off between temporal reasoning and spatial precision. Models that rely on single-frame or low-resolution inputs often miss small, distant, or partially occluded hazards, while language-centric driving models frequently provide limited grounded evidence for their explanations. To address this gap, we propose UniDrive, a unified visual-language and grounding framework for interpretable risk understanding in autonomous driving. UniDrive combines a temporal reasoning branch that models scene dynamics from multi-frame visual input with a high-resolution perception branch that preserves fine-grained spatial details from the latest frame. The two branches are integrated through a gated cross-attention fusion module, enabling dynamic context to be aligned with precise spatial evidence. Based on the fused representation, UniDrive jointly generates natural-language risk descriptions and grounded bounding-box outputs for risk objects. Experiments on the DRAMA-Reasoning benchmark show that UniDrive outperforms representative image-based and video-based baselines in both captioning and risk-object grounding. In particular, UniDrive achieves the best overall performance on the validation split and demonstrates clear advantages in small-object localization, zero-shot generalization to NuScenes and BDD100K, and human-rated interpretability and trustworthiness. These results suggest that explicitly combining temporal semantics and high-resolution perception provides a stronger foundation for interpretable and safety-oriented autonomous driving systems. The code is available at this https URL.

**arXiv ID:** 2606.24759
</details>

<details>
<summary><strong>BioMedArena: An Open-source Toolkit for Building and Evaluating Biomedical Deep Research Agents</strong> - Jinge Wu, Hongjian Zhou, Mingde Zeng, Jiayuan Zhu, Junde Wu, Jiazhen Pan, Ayush Noori, Sean Wu, Honghan Wu, Fenglin Liu, David A. Clifton - [[pdf]](https://arxiv.org/pdf/2605.06177)</summary>

**Abstract:** Reproducing and comparing deep research agents today is hard: the same backbone evaluated on the same benchmark can report different accuracies across papers because the harness and tool registry differ, and integrating a new model into a comparable evaluation surface costs weeks of model-specific engineering. These are symptoms of a broader reproducibility problem in deep research agent research. Here, we introduce BioMedArena, an open-source toolkit that addresses this reproducibility gap and provides an arena for comparing deep research agents under a shared evaluation environment. BioMedArena decouples six layers of biomedical agent evaluation -- benchmark loading, tool exposure, tool selection, harness mode, context management, and scoring -- and exposes 166 biomedical benchmarks and 75 biomedical tools across 9 functional families. Adding a new model, benchmark, or tool can be accomplished with a few-line provider adapter. Beyond evaluation infrastructure, BioMedArena ships a library of high-quality reference components: 6 agent harnesses (including our proposed Mutual-Evolve) and 6 context-management strategies, any of which can be equipped on any backbone. Equipping these components substantially improves all 12 backbones; on each of 8 representative biomedical benchmarks, the best equipped backbone surpasses prior state-of-the-art (SOTA), by 15.01 percentage points on average. The toolkit, configurations, and per-task traces are available at this https URL.

**arXiv ID:** 2605.06177
</details>

<details>
<summary><strong>EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent</strong> - Zeyao Du, Tong Li, Yanci Zhang, Haibo Zhang - [[pdf]](https://arxiv.org/pdf/2606.17698)</summary>

**Abstract:** As LLM-based shopping agents enter production, existing benchmarks fail to capture how a shopper's requirements arrive: stated implicitly in the query, recorded in a profile, or revealed only when the right question is asked. Benchmarks that expose full intent upfront and grade only the final choice can neither pose this long-horizon challenge nor explain which requirement an agent missed. To address this gap, we introduce EComAgentBench, a benchmark of 662 tasks grounded in real Amazon products and reviews. Each task scatters these requirements across a visible query, a tool-gated profile, and scripted clarification; an agent must uncover hidden intent, verify candidates against attributes and review evidence, and commit to a single product within 100 tool calls. Moreover, typed, source-tagged rubrics grade every task, attributing each failure to a requirement and its source. Construction is automated yet reliable, with every answer fixed in code before any text is generated and every sample validated. Our evaluation of seven models reveals that even the strongest attains only 57.1% overall accuracy, and rubric satisfaction degrades from visible to hidden sources. Overall, we believe EComAgentBench will serve as a reproducible foundation for moving shopping agents from single-query search toward dependable assistance over long horizons.

**arXiv ID:** 2606.17698
</details>

<details>
<summary><strong>CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark</strong> - Zachary S. Siegel, Sayash Kapoor, Nitya Nadgir, Benedikt Stroebl, Arvind Narayanan - [[pdf]](https://arxiv.org/pdf/2409.11363)</summary>

**Abstract:** AI agents have the potential to aid users on a variety of consequential tasks, including conducting scientific research. To spur the development of useful agents, we need benchmarks that are challenging, but more crucially, directly correspond to real-world tasks of interest. This paper introduces such a benchmark, designed to measure the accuracy of AI agents in tackling a crucial yet surprisingly challenging aspect of scientific research: computational reproducibility. This task, fundamental to the scientific process, involves reproducing the results of a study using the provided code and data. We introduce CORE-Bench (Computational Reproducibility Agent Benchmark), a benchmark consisting of 270 tasks based on 90 scientific papers across three disciplines (computer science, social science, and medicine). Tasks in CORE-Bench consist of three difficulty levels and include both language-only and vision-language tasks. We provide an evaluation system to measure the accuracy of agents in a fast and parallelizable way, saving days of evaluation time for each run compared to a sequential implementation. We evaluated two baseline agents: the general-purpose AutoGPT and a task-specific agent called CORE-Agent. We tested both variants using two underlying language models: GPT-4o and GPT-4o-mini. The best agent achieved an accuracy of 21% on the hardest task, showing the vast scope for improvement in automating routine scientific tasks. Having agents that can reproduce existing work is a necessary step towards building agents that can conduct novel research and could verify and improve the performance of other research agents. We hope that CORE-Bench can improve the state of reproducibility and spur the development of future research agents.

**arXiv ID:** 2409.11363
</details>

<details>
<summary><strong>ATHENA: Agentic Team for Hierarchical Evolutionary Numerical Algorithms</strong> - Juan Diego Toscano, Daniel T. Chen, George Em Karniadakis - [[pdf]](https://arxiv.org/pdf/2512.03476)</summary>

**Abstract:** Progress in computational science depends on complex numerical workflows that must faithfully encode physical laws, yet translating conceptual insight into reliable code remains a major bottleneck. Although large language models can generate isolated code fragments, they lack the structured reasoning required to design, verify, and iteratively refine complete scientific pipelines. Here we introduce ATHENA, an agentic framework explicitly designed to emulate scientific research modeled as a knowledge-driven contextual bandit process. Its core loop separates conceptual policy from numerical realization through expert-derived conceptual scaffolding, enabling principled diagnosis, reformulation, and repair of computational strategies. Across scientific computing and scientific machine learning tasks, ATHENA autonomously derives and correctly applies exact analytical solutions, constructs stable numerical solvers, diagnoses ill-posed formulations, and orchestrates hybrid symbolic-numeric workflows. Quantitatively, ATHENA matches and frequently surpasses the accuracy of expert-authored reference solutions reported in the literature on canonical benchmarks. By reframing computation as an object of agentic reasoning, our framework enables autonomous orchestration of heterogeneous algorithms across scientific domains.

**arXiv ID:** 2512.03476
</details>

<details>
<summary><strong>Toward Autonomous O-RAN: A Multi-Scale Agentic AI Framework for Real-Time Network Control and Management</strong> - Hojjat Navidan, Mohammad Cheraghinia, Jaron Fontaine, Mohamed Seif, Eli De Poorter, H. Vincent Poor, Ingrid Moerman, Adnan Shahid - [[pdf]](https://arxiv.org/pdf/2602.14117)</summary>

**Abstract:** Open Radio Access Networks (O-RAN) promise flexible 6G network access through disaggregated, software-driven components and open interfaces, but this programmability also increases operational complexity. Multiple control loops coexist across the service management layer and RAN Intelligent Controller (RIC), while independently developed control applications can interact in unintended ways. In parallel, recent advances in generative Artificial Intelligence (AI) are enabling a shift from isolated AI models toward agentic AI systems that can interpret goals, coordinate multiple models and control functions, and adapt their behavior over time. This article proposes a multi-scale agentic AI framework for O-RAN that organizes RAN intelligence as a coordinated hierarchy across the Non-Real-Time (Non-RT), Near-Real-Time (Near-RT), and Real-Time (RT) control loops: (i) A Large Language Model (LLM) agent in the Non-RT RIC translates operator intent into policies and governs model lifecycles. (ii) Small Language Model (SLM) agents in the Near-RT RIC execute low-latency optimization and can activate, tune, or disable existing control applications; and (iii) Wireless Physical-layer Foundation Model (WPFM) agents near the distributed unit provide fast inference close to the air interface. We describe how these agents cooperate through standardized O-RAN interfaces and telemetry. Using a proof-of-concept implementation built on open-source models, software, and datasets, we demonstrate the proposed agentic approach in two representative scenarios: robust operation under non-stationary conditions and intent-driven slice resource control.

**arXiv ID:** 2602.14117
</details>

<details>
<summary><strong>Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning</strong> - Shiding Zhu, Yudi Qi, Yajie Wang, Jiaze Li, Chao Song, Yaorui Shi, Yibo Miao, Hanqi Gao, Kai Zhang - [[pdf]](https://arxiv.org/pdf/2606.24428)</summary>

**Abstract:** Experience-driven self-evolution is critical for large language model (LLM) agents to improve through open-world interaction. However, existing experience learning methods mostly rely on single-agent loops, where the same agent executes tasks, summarizes outcomes, and determines memory content. This setup makes agents vulnerable to the Self-Confirmation Trap: wrong-but-self-consistent trajectories are misidentified as successful experience, leading to cumulative errors during retrieval and reuse. To address this issue, we propose EDV, an Execute-Distill-Verify framework for reliable experience learning. In the Execute stage, multiple heterogeneous agents explore the same task space in parallel to generate diverse candidate trajectories. In the Distill stage, a dedicated third-party agent comparatively analyzes these trajectories to produce candidate experiences, reducing executor-centric summarization bias. In the Verify stage, the execution group validates candidates via a consensus mechanism, and only approved experiences are written into shared or private memory. By decoupling the three stages, EDV transforms experience learning from isolated self-reflection into collaborative construction, filtering erroneous and noisy content before memory insertion. We evaluate EDV on three challenging long-horizon benchmarks: tau2-bench, Mind2Web and MMTB. Results show EDV consistently outperforms strong baselines, validating that reliable experience construction is essential for robust agent self-evolution. Our code is available at this https URL.

**arXiv ID:** 2606.24428
</details>

<details>
<summary><strong>AGORA: An Archive-Grounded Benchmark for Agentic Workplace Document Reasoning</strong> - Honglin Guo, Qi Zhang, Yu Zhang, Weijie Li, Rui Zheng, Zhikai Lei, Qiyuan Peng, Zhiheng Xi, Tao Gui, Qi Zhang - [[pdf]](https://arxiv.org/pdf/2606.24526)</summary>

**Abstract:** Large language models are increasingly deployed as agents that reason over documents rather than answer from parametric knowledge. We study archive-grounded reasoning: locating sparse evidence across a large, messy collection of workplace files, reconciling inconsistent terminology, units, and time conventions, and computing an answer. Existing benchmarks address only parts of this setting and none jointly stresses archive-groundedness, agentic exploration, and cross-domain coverage. We introduce Agora, a benchmark pairing 362 questions with eight domain collections of 9,664 authentic documents and 372M tokens, far exceeding any model's context window, so agents must explore deliberately rather than scan exhaustively. Agora is built by an agentic pipeline combining cross-document task synthesis, leakage-preventing obfuscation, and difficulty filtering. Evaluating eight models, we find the task far from solved: even the strongest reaches only 59.4% accuracy, with notable variation across domains.

**arXiv ID:** 2606.24526
</details>

<details>
<summary><strong>NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers?</strong> - Yuru Wang, Lejun Cheng, Yuxin Zuo, Sihang Zeng, Bingxiang He, Che Jiang, Junlin Yang, Yuchong Wang, Kaikai Zhao, Weifeng Huang, Kai Tian, Zhenzhao Yuan, Jincheng Zhong, Weizhi Wang, Ning Ding, Bowen Zhou, Kaiyan Zhang - [[pdf]](https://arxiv.org/pdf/2606.24530)</summary>

**Abstract:** We introduce NatureBench, a cross-discipline benchmark of 90 tasks distilled from peer-reviewed Nature-family publications, designed to evaluate whether AI coding agents can move beyond reproduction toward discovery on real scientific problems. NatureBench is built on NatureGym, an automated pipeline that constructs a standardized, per-task containerized environment from a source paper, addressing the environment-fragmentation problem that has limited the credibility of prior agent-on-research benchmarks. Evaluating ten frontier agent configurations under a strict web-search-disabled protocol, we find that the strongest model surpasses SOTA on only 17.8% of tasks under the g>0.1 criterion. Analysis of method pathways reveals that agents succeed primarily through methodological translation, converting scientific tasks into familiar supervised prediction problems, rather than through genuine scientific invention. Failures are dominated by wrong method choice and insufficient compute budget, not by task misunderstanding. We release the benchmark, the NatureGym pipeline, and a public leaderboard with maintainer-side reproduction. Code: this https URL

**arXiv ID:** 2606.24530
</details>

<details>
<summary><strong>Thinking While Speaking: Inference-Time Knowledge Transfer for Responsive and Intelligent Conversational Voice Agents</strong> - Vidya Srinivas, Zachary Englhardt, Shwetak Patel, Vikram Iyer - [[pdf]](https://arxiv.org/pdf/2511.07397)</summary>

**Abstract:** Voice agents face a fundamental tension: the reasoning, retrieval, and tool use that make foundation models capable are iterative and slow, while conversational interaction demands responses on a millisecond timescale. Smaller, real-time models meet the latency bar but cannot match foundation models on complex tasks, leaving current voice agents to trade away either responsiveness or capability. We introduce conversational infill, where a small talker model both immediately generates contextually grounded responses to hide the latency of an external reasoner model and fluently integrates streamed reasoner knowledge into its responses during inference. We curate a 290,571-example synthetic dataset spanning six domains and demonstrate that this task is learnable across seven widely used small language models ranging from 135M to 1.7B parameters. Our system implementation, ConvFill, sustains millisecond-level time-to-first-response while closing the accuracy gap to within 6.3% of the corresponding frontier reasoner performance. In a live user study (n=18) with talker deployments running on an Apple M2 SoC, participants rank ConvFill on par with frontier models overall, prefer it for retrieval-heavy tasks, and rate it significantly more responsive. These results show that conversational infill unlocks a new point on the latency-capability Pareto frontier, offering a practical path toward voice agents that are both responsive and highly capable. Code, models, and datasets are available at this https URL.

**arXiv ID:** 2511.07397
</details>

<details>
<summary><strong>A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs</strong> - Cuong Vuong Tuan, Trang Mai Xuan, Tien-Cuong Nguyen, Vu-Duc Ngo, Thien Van Luong - [[pdf]](https://arxiv.org/pdf/2606.03867)</summary>

**Abstract:** Multi-Document Summarization (MDS) plays a critical role in distilling essential information from collections of textual data. Existing approaches often struggle to capture complex inter-document relationships, rely heavily on large amounts of labeled data for supervised training, or exhibit limited generalization across domains and languages. To address these limitations, we present a training-free mixture-of-agents framework for MDS that leverages the complementary strengths of large language models (LLMs) and knowledge graphs. Our approach decomposes summarization into specialized agent tasks: extractive selection, knowledge-aware abstraction, and iterative refinement, each operating without task-specific fine-tuning. We unify their outputs using a multi-perspective consistency mechanism guided by LLMs. Experiments across four datasets in English and Vietnamese demonstrate state-of-the-art or competitive performance, validating the effectiveness and adaptability of our modular design.

**arXiv ID:** 2606.03867
</details>

<details>
<summary><strong>How Much Can We Trust LLM Search Agents? Measuring Endorsement Vulnerability to Web Content Manipulation</strong> - Yimeng Chen, Zhe Ren, Firas Laakom, Yu Li, Dandan Guo, Jürgen Schmidhuber - [[pdf]](https://arxiv.org/pdf/2606.16821)</summary>

**Abstract:** Large language model (LLM)-based search agents synthesize open-web content into actionable recommendations on behalf of users, creating a risk that attacker-published pages are transformed into endorsed claims. We introduce SearchGEO, a controlled evaluation framework for measuring endorsement corruption in LLM-based web-search agents, combining a web-evidence manipulation pipeline, a five-mode attack taxonomy, and multiple output-level metrics. We evaluate 13 LLM backends on 308 cases each. Results show that vulnerability patterns vary across backends: overall attack success rate (ASR) ranges from 0.0% on Claude-Sonnet-4.6 to 31.4% on Gemini-3-Flash, the strongest attack mode differs by model family, and the same deployment scaffold could amplify or decrease ASR on different backends. An auxiliary agent-skill probe, where endorsement becomes an install command, exposes a sharp split among otherwise robust backends: Claude over-rejects while GPT over-trusts. These findings argue for treating recommendation reliability under adversarial search content as a first-class dimension of backend safety evaluation.

**arXiv ID:** 2606.16821
</details>

<details>
<summary><strong>Polaris: A Godel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair</strong> - Aditya Kakade, Vivek Srivastava, Shirish Karande - [[pdf]](https://arxiv.org/pdf/2603.23129)</summary>

**Abstract:** Gödel agent realize recursive self-improvement: an agent inspects its own policy and traces and then modifies that policy in a tested loop. We introduce Polaris, a Gödel agent for compact models that performs policy repair via experience abstraction, turning failures into policy updates through a structured cycle of analysis, strategy formation, abstraction, and minimal code pat ch repair with conservative checks. Unlike response level self correction or parameter tuning, Polaris makes policy level changes with small, auditable patches that persist in the policy and are reused on unseen instances within each benchmark. As part of the loop, the agent engages in meta reasoning: it explains its errors, proposes concrete revisions to its own policy, and then updates the policy. To enable cumulative policy refinement, we introduce experience abstraction, which distills failures into compact, reusable strategies that transfer to unseen instances. On MGSM, DROP, GPQA, and LitBench (covering arithmetic reasoning, compositional inference, graduate-level problem solving, and creative writing evaluation), a 7-billion-parameter model equipped with Polaris achieves consistent gains over the base policy and competitive baselines.

**arXiv ID:** 2603.23129
</details>

<details>
<summary><strong>SkillHone: A Harness for Continual Agent Skill Evolution Through Persistent Decision History</strong> - Zhiwei Li, Yong Hu - [[pdf]](https://arxiv.org/pdf/2606.08671)</summary>

**Abstract:** Agent skills extend language-model agents with task-specific procedures, scripts, and references, but the tasks and environments they target continually change. Existing methods improve skills in bounded runs and retain only the final artifact, discarding the decision history that later agents need to interpret prior revisions, evaluations, and rejected alternatives. We introduce SkillHone, a harness for continual agent skill evolution grounded in persistent decision history. SkillHone pairs skill revisions with evaluation-side evidence that supplies practice feedback, recording structured histories of diagnoses, revisions, evidence, and outcomes. Role-separated subagents run candidate skills on practice probes with redacted reporting and propose revisions informed by prior decisions, enabling cross-session refinement without rediscovering past rationale. On deep-research benchmarks, SkillHone runs without a pre-integrated search stack and outperforms the commercially backed deep-research agent by 15.8 points on GAIA and 3.2 points on WebWalkerQA-EN, while also exceeding prior skill-evolution methods. We further deploy SkillHone on internal tool-mediated analysis scenarios, where it improves accuracy by an average of 18.8 points across seven settings.

**arXiv ID:** 2606.08671
</details>

<details>
<summary><strong>MortarBench: Evaluating Mortgage Loan Origination Agents</strong> - Matthew Toles, Yunan Lu, Manav Munjal, Bojun Liu, Yuanhao Deng, Stephanie Selig, Derek Rindner, Cheng Li, Zhou Yu - [[pdf]](https://arxiv.org/pdf/2606.19416)</summary>

**Abstract:** Loan origination is the process by which a lender creates a new loan, from application and underwriting through approval and funding. This process serves a critical role in evaluating the eligibility and level of risk posed by an applicant. Recently, firms have begun using mortgage loan agents to augment human loan officers, despite a lack of any public benchmark. To fill this gap, we present MortarBench, a loan origination agent benchmark. MortarBench uses a financial data synthesis and mutation pipeline to generate examples with broad edge case coverage that match real-world distributions and questions. We find that state-of-the-art large language models (LLMs) perform poorly, with closed-source models achieving at most 77.1\% exact match accuracy. We also discover systematic biases in LLM perception of foreignness related to non-English names. Noting these weaknesses, we introduce CRIT, a confidence calibration framework. Our method increases accuracy to 80.5\% while improving risk management steering and reducing bias.

**arXiv ID:** 2606.19416
</details>

<details>
<summary><strong>ObsGraph: Hierarchical Observation Representation for Embodied Reasoning and Exploration</strong> - Taekbeom Lee, Youngseok Jang, Jeonghwa Heo, Jeongjun Choi, H. Jin Kim - [[pdf]](https://arxiv.org/pdf/2606.24068)</summary>

**Abstract:** Embodied reasoning and exploration are increasingly considered crucial abilities for robots operating in complex and unfamiliar environments. To accomplish tasks in such settings, an agent must identify and acquire the information necessary for the task through exploration. We propose ObsGraph, an observation-centric hierarchical scene graph that unifies scene representation, retrieval, and exploration. It retains visual evidence and organizes it into room-view-object layers: rooms provide coarse semantic anchors, views preserve contextual object covisibility, and objects store fine-grained details. On top of this representation, we perform coarse-to-fine hierarchical retrieval under a bounded budget, and crucially use retrieval outcomes to structure the exploration candidate space--activating room-level exploration, view refinement, or frontier exploration--thereby tightly coupling representation, retrieval, and adaptive multi-scale exploration. Experiments across embodied reasoning and exploration benchmarks demonstrate improved success and efficiency, highlighting the benefits of structured scene representation and more targeted information gathering driven by identified evidence gaps.

**arXiv ID:** 2606.24068
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Can Language Model Agents be Helpful Circuit Explainers in Mechanistic Interpretability?</strong> - Ayan Antik Khan, Harsh Kohli, Yuekun Yao, Huan Sun, Ziyu Yao - [[pdf]](https://arxiv.org/pdf/2606.24026)</summary>

**Abstract:** Mechanistic interpretability has made substantial progress in automatically localizing circuits, but explaining what localized components do remains labor-intensive and difficult to standardize. In this work, we study whether language model (LM) agents can assist with this explanation problem once a circuit has already been identified. We introduce AgenticInterpBench, a benchmark for circuit explanation built from 84 semi-synthetic transformer circuits with 163 component-level annotations. We propose HyVE (Hypothesize, Validate, Explain), an agentic explainer that analyzes each component through an iterative loop of observation, hypothesis generation, and causal validation, eventually producing a component-level explanation and a circuit-level task description. Across four LM backbones, HyVE recovers useful component- and task-level explanations, but no backbone is uniformly best. Our analysis shows that strong backbones usually form observation-grounded hypotheses, while failures more often arise later in the validation loop, through incomplete validation plans, code execution errors, or unresolved hypotheses. A case study on an arithmetic circuit in Llama-3-8B shows that the same formulation can extend beyond semi-synthetic benchmarks to naturally trained models. Overall, LM agents are promising circuit explainers, but reliable validation remains the key obstacle.

**arXiv ID:** 2606.24026
</details>

<details>
<summary><strong>AutoSpec: Safety Rule Evolution for LLM Agents via Inductive Logic Programming</strong> - Pingchuan Ma, Zhaoyu Wang, Zimo Ji, Yuguang Zhou, Zhantong Xue, Zongjie Li, Shuai Wang, Xiaoqin Zhang - [[pdf]](https://arxiv.org/pdf/2606.24245)</summary>

**Abstract:** Large language model (LLM) agents increasingly automate complex tasks by integrating language models with external tools and environments. However, their autonomy poses significant safety risks: agents may execute destructive commands, leak sensitive data, or violate domain constraints. Existing safety approaches face a fundamental tradeoff: hand-crafted rules are interpretable but brittle, with overly conservative rules blocking safe operations (high false positives) while permissive rules miss unsafe behaviors (high false negatives). Neural classifiers lack the interpretability required for safety-critical deployments.
We present AutoSpec, a framework that automatically evolves deployed expert-designed safety rules from user safe/unsafe annotations through counterexample-guided inductive synthesis (CEGIS) guided by inductive logic programming (ILP). Starting from the expert rules and a stream of annotated traces, AutoSpec iteratively evaluates rules, mines false-positive and false-negative counterexamples, uses ILP to learn which predicates discriminate them, generates candidate rule edits, and verifies candidates to select the best revision. The key insight is that ILP efficiently identifies predicates that appear frequently in false negatives but rarely in false positives (or vice versa), dramatically pruning the exponential search space of rule edits. This continues until convergence, producing interpretable rules that balance precision and recall.
We evaluate AutoSpec on 291 execution traces spanning code execution and embodied agent domains. AutoSpec raises rule F1 to 0.98 and 0.93 across the two domains, achieving up to 94% false positive reduction while maintaining high recall, and converges within 4-5 iterations. The ILP-guided approach achieves up to 4.8x higher F1 than heuristic CEGIS. The learned rules are human-readable, auditable, and generalize to unseen scenarios.

**arXiv ID:** 2606.24245
</details>

<details>
<summary><strong>MEMPROBE: Probing Long-Term Agent Memory via Hidden User-State Recovery</strong> - Enze Ma, Yufan Zhou, Wei-Chieh Huang, Jie Yang, Huanhuan Ma, Zixuan Wang, Chengze Li, Chunyu Miao, Philip S. Yu, Zhen Wang - [[pdf]](https://arxiv.org/pdf/2606.24595)</summary>

**Abstract:** Long-term memory promises LLM agents that grow more capable across sessions, maintaining an accurate, evolving understanding of the user that interaction forms. In practice, however, this memory is evaluated mostly through downstream behavior, such as later answers, personalization quality, or task success, which tests that understanding only indirectly and leaves the memory artifact itself largely unaudited. We argue that long-term memory should instead be evaluated as an auditable post-interaction artifact: after ordinary assistance, what structured user state can be reconstructed from the memory the agent leaves behind? We instantiate this view in MEMPROBE, a benchmark in which a memory-equipped agent assists simulated users, each carrying a hidden, taxonomy-anchored user-state bank, across a trajectory of leak-controlled tasks, after which that bank is reconstructed from the agent's resulting memory under both full-store and top-k access. Built on synthetic ground truth for efficient, scalable measurement, MEMPROBE spans 50 simulated users with 31 hidden dimensions each (1,550 recovery targets) and tests 5 representative memory systems. Testing state-of-the-art memory agents, we find that successful assistance and recoverable memory behave as distinct capabilities. Task completion nearly saturates, even for a memoryless baseline, while category-balanced recovery stays moderate (about 0.6) and drops further under top-k retrieval. MEMPROBE is the first benchmark to study memory recovery directly, reconstructing the user state a system retains and scoring it against ground truth. We see recovery as a concrete objective for future memory agents to optimize, and MEMPROBE as a step toward an environment where agents are trained to remember their users, growing more faithful the longer they know them.

**arXiv ID:** 2606.24595
</details>

<details>
<summary><strong>PEARL: Self-Evolving Assistant for Time Management with Reinforcement Learning</strong> - Bingxuan Li, Jeonghwan Kim, Cheng Qian, Xiusi Chen, Eitan Anzenberg, Niran Kundapur, Heng Ji - [[pdf]](https://arxiv.org/pdf/2601.11957)</summary>

**Abstract:** Overlapping calendar invitations force busy professionals to repeatedly decide which meetings to attend, reschedule, or decline. We refer to this preference-driven decision process as calendar conflict resolution. Automating this decision process is crucial yet challenging. Scheduling logistics can drain hours, and human delegation often fails at scale, which motivates us to ask: Can we trust large language models (LLMs) or language agents to manage time? To enable a systematic study of this question, we introduce CalConflictBench, a benchmark for long-horizon calendar conflict resolution. In CalConflictBench, conflicts are presented to agents round-by-round over a calendar year, requiring them to infer and adapt to user preferences progressively. Our experiments show that current LLM agents perform poorly with high error rates, e.g., Qwen-3-30B-Think has an average error rate of 35%. To address this gap, we propose PEARL, a reinforcement-learning framework that (i) augments the language agent with an external preference memory that stores and updates inferred strategies (e.g., attendee priorities, topic importance, time/location preferences), and (ii) optimizes the agent with round-wise rewards that directly supervise decision correctness, ranking quality, and memory usage across rounds. Experiments on CalConflictBench show that PEARL achieves an error reduction rate of 0.76 and a 55% improvement in average error rate compared to the strongest baseline.

**arXiv ID:** 2601.11957
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (21 papers)</h2></summary>

<details>
<summary><strong>Safe and Generalizable Hierarchical Multi-Agent RL via Constraint Manifold Control</strong> - Zihao Guo, Jianing Zhao, Ling Li, Hao Liang, Giuseppe Loianno, Yali Du - [[pdf]](https://arxiv.org/pdf/2606.24010)</summary>

**Abstract:** Multi-agent systems are widely used in safety-critical applications that require coordinated behavior under strict safety constraints. Existing approaches face a fundamental trade-off: learning-based methods achieve strong empirical performance but lack theoretical safety guarantees, while control-theoretic methods enforce safety but often lead to overly conservative and inefficient behaviors. We propose a hierarchical multi-agent reinforcement learning framework that enforces hard safety constraints under mild assumptions at low level via a constraint manifold, while enabling effective coordination through high-level policy learning. Our approach provides theoretical safety guarantees in the multi-agent setting and yields stationary learning dynamics, thereby enabling stable and efficient training. Empirically, our method achieves competitive performance while maintaining nearly perfect safety rates, and generalizes effectively to varying numbers of agents and obstacles.

**arXiv ID:** 2606.24010
</details>

<details>
<summary><strong>ATRIA: Adaptive Traceable ECG Reporting with Iterative Agents</strong> - Donggyun Hong, Kyuhwan Lee, Junmyung Kwon, Yong-Yeon Jo - [[pdf]](https://arxiv.org/pdf/2606.24392)</summary>

**Abstract:** Existing ECG report generation is tightly coupled -- interpretation and reporting fused end-to-end, so errors propagate without stage-level recourse -- while agent-based systems decouple tasks but remain single-pass, never revisiting earlier outputs. Clinical ECG reporting instead unfolds iteratively, requiring progressive context integration and bidirectional editing. We present \textsc{ATRIA}, a multi-agent ECG reporting system that mirrors the clinician's iterative workflow: it binds every report claim to its supporting evidence, flags statements unsupported by that evidence, incorporates additional context mid-session, and lets clinicians verify and revise individual findings rather than accept one opaque output. Because its agents use ECG analysis models already in clinical use, the underlying findings are clinically trustworthy; and as a cloud-based web service, \textsc{ATRIA} is ready for immediate deployment. We demonstrate \textsc{ATRIA} through four interaction cases, with a live demo and video available.

**arXiv ID:** 2606.24392
</details>

<details>
<summary><strong>Agentic AI for Bilevel Long-Term Optimization of Policy-Driven Physical Layer Systems</strong> - Bingnan Xiao, Chenhao Yang, Wei Ni, Xin Wang, Tony Q. S. Quek - [[pdf]](https://arxiv.org/pdf/2606.24416)</summary>

**Abstract:** Network operators' changing policies, service requirements, and stringent real-time constraints render existing methods designed with fixed objectives and constraints ineffective. This paper presents Agentic long-term performance optimization (Agentic-LTPO), a nested bilevel optimization framework that can be applied to adaptive physical layer problem configuration. The key idea is to employ agentic AI to generate upper-level configurations in a bilevel optimization structure, where evolving operator policies, environment summaries, and historical experiences are translated into structured lower-level optimization problem configurations. The lower level solves the problems with updated configurations for real-time physical-layer decisions. Considering cell-free MIMO beamforming as a use case, we embody Agentic-LTPO by designing a new multi-agent decision process with retrieval-augmented experience-based verification in the upper level, together with a closed-form beamformer in the lower level. Experiments demonstrate that Agentic-LTPO exhibits strong adaptability to dynamic operator policies and effectively enhances the system's long-term performance by 57.2% compared to traditional methods.

**arXiv ID:** 2606.24416
</details>

<details>
<summary><strong>ReM-MoA: Reasoning Memory Sustains Mixture-of-Agents Scaling</strong> - Heng Ping, Arijit Bhattacharjee, Peiyu Zhang, Shixuan Li, Wei Yang, Ali Jannesari, Nesreen Ahmed, Paul Bogdan - [[pdf]](https://arxiv.org/pdf/2606.24437)</summary>

**Abstract:** Mixture-of-Agents (MoA) architectures improve inference-time scaling by organizing multiple LLM agents into layered reasoning pipelines. However, existing MoA variants fail to sustain gains as depth increases, exhibiting degradation, early plateauing, or saturation. We propose ReM-MoA, a memory-augmented MoA framework that sustains scaling through two mechanisms: (1) a Ranked Reasoning Memory that persistently stores and ranks reasoning traces from all layers using a comparative Reviewer Agent, and (2) a Curated Diversified Memory Routing scheme that exposes different agents to distinct combinations of successful and failed traces, preserving exploration diversity while propagating high-quality reasoning. We further introduce an optional multi-domain Reviewer distillation pipeline that improves ranking quality through frontier-model supervision. Across five reasoning benchmarks spanning math, formal logic, code, knowledge, and commonsense, ReM-MoA consistently outperforms prior MoA variants across both depth and width scaling, and its advantage widens with depth, establishing structured cross-layer reasoning memory as a key missing mechanism for scalable multi-agent inference.

**arXiv ID:** 2606.24437
</details>

<details>
<summary><strong>Governed Shared Memory for Multi-Agent LLM Systems</strong> - Yanki Margalit, Nurit Cohen-Inger, Erni Avram, Ran Taig, Oded Margalit - [[pdf]](https://arxiv.org/pdf/2606.24535)</summary>

**Abstract:** Multi-agent LLM environments require robust mechanisms for shared knowledge management. This paper formalizes the fleet-memory problem and identifies four foundational failure modes: unauthorized leakage, stale propagation, contradiction persistence, and provenance collapse. To address these, we define explicit systems-level primitives: scoped retrieval, temporal supersession, provenance tracking, and policy-governed memory propagation. These primitives are implemented in MemClaw, a production multi-tenant memory service, and evaluated via ArgusFleet, a reproducible harness testing four governance dimensions. Rather than a baseline comparison, this study measures a live production service, emphasizing real-world architectural insights and negative results. Key Evaluation Results Provenance: Successfully reconstructed 100% of depth-four derivation chains with correct writer identity at sub-second per-hop latency. Propagation: Demonstrated high intra-fleet visibility with zero cross-fleet leakage. Under strong write mode, write-to-visible latency was optimized to a single search round-trip. Production Architectural Issues Discovered Asymmetric Scope Enforcement: Tenant isolation held, but sub-tenant scope was initially bypassed on direct GET-by-id requests for agent-scoped credentials (disclosed and remediated during the study). Pipeline Ordering Conflict: While contradiction supersession works for admitted writes, a synchronous near-duplicate gate can prematurely reject contradictory writes before the asynchronous contradiction detector can evaluate them. Conclusion: Long-context retrieval alone is insufficient for production multi-agent memory. Governed shared memory demands explicit systems-level abstractions, and live evaluation is vital to expose enforcement and pipeline-ordering failures missed by design-only treatments.

**arXiv ID:** 2606.24535
</details>

<details>
<summary><strong>ASALT: Adaptive State Alignment for Lateral Transfer in Multi-agent Reinforcement Learning</strong> - Anurag Akula, Satheesh K. Perepu, Abhishek Sarkar, Kaushik Dey - [[pdf]](https://arxiv.org/pdf/2606.24601)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) addresses the problem of training multiple agents that pursue collaborative, competitive, or mixed objectives. Prior work has investigated transfer learning between source and target domains in MARL; however, the majority of existing approaches impose the constraint that the dimensionalities of the observation space and the global state space must be identical across domains. In this paper, we introduce a method that explicitly accommodates mismatched state-space dimensionalities between source and target domains. The proposed approach, ASALT, incorporates both observation-level and state-level adapters that map the target-domain observations and global states into a shared embedding space, thereby enabling more effective transfer of knowledge across both actors and critics. These adapters can generate embeddings that support efficient strategy transfer across heterogeneous domains. Experimental results on multiple configurations in standard benchmark environments demonstrate that ASALT surpasses existing baselines in terms of sample efficiency and global return in cooperative settings, but its effectiveness depends on the degree of mismatch between source and target domains. Furthermore, our findings indicate that ASALT mitigates negative transfer, which frequently constitutes a major obstacle when transferring policies between domains with differing observation and action spaces.

**arXiv ID:** 2606.24601
</details>

<details>
<summary><strong>SAFARI: Scaling Long Horizon Agentic Fault Attribution via Active Investigation</strong> - Chenyang Zhu, Jiayu Yao, Kushal Chawla, Youbing Yin, Nathan Wolfe, Pengshan Cai, Jingyu Wu, Spencer Hong, Sangwoo Cho, Shi-Xiong Zhang, Daben Liu, Sambit Sahu, Erin Babinsky - [[pdf]](https://arxiv.org/pdf/2606.24626)</summary>

**Abstract:** As autonomous agents tackle increasingly complex multi-step, multi-agent tasks, their execution trajectories have scaled beyond the constraints of even the largest context windows. Current methods for effectively diagnosing agent failures load the full trajectory into an LLM's context window, which suffers from attention dilution and fails when agentic traces inevitably exceed context limits. To address this, we introduce SAFARI (Scaling long-horizon Agentic Fault AttRibution via active Investigation), a framework that replaces linear context loading with a tool-augmented diagnostic loop. By equipping LLMs with a specialized toolbox to read and search trajectory segments alongside a persistent Short-Term Memory (STM) for cross-turn reasoning, SAFARI effectively decouples diagnostic accuracy from architectural context limits. Our experiments demonstrate that SAFARI outperforms state-of-the-art results by 20% on the Who&When dataset within a 1M token budget, and by 19% on TRAIL GAIA subset on a 25K token budget. Most significantly, SAFARI maintains a 0.58 precision even when the target fault resides 5x beyond the model's native context window, a scenario where traditional evaluators fail entirely.

**arXiv ID:** 2606.24626
</details>

<details>
<summary><strong>Grading the Grader: Lessons from Evaluating an Agentic Data Analysis System</strong> - Tian Zheng, Kai-Tai Hsu - [[pdf]](https://arxiv.org/pdf/2606.24839)</summary>

**Abstract:** Agentic data analysis systems produce rich outputs, including code, numerical results, and verbal diagnostics. This makes them more challenging to evaluate than single-turn LLM responses. It is therefore necessary to distinguish genuine disagreement between an agent's output and a ground-truth answer from grading artifacts. We investigate how reliably automated graders assess such a system and what strategies improve grading quality by applying LAMBDA, a multi-agent data-analysis system, on 153 numerical QRData tasks from DSGym. We develop and evaluate a three-layer human-AI grading cascade: strict regex matching, LLM-based lenient grading, and snippet-based human inspection, which combines non-GenAI and GenAI strategies with different failure profiles. Both automated graders achieve 100% observed precision (0/70 false positives). The lenient grader's recall is 97% against human labels. A keyword-anchored extraction pipeline raises the strict grader's recall by 60 percentage points over a last-number heuristic; the lenient grader is architecturally parser-independent. An iterative nudge mechanism raises grading run success from 36% to 97% and lenient-pass rates from 16% to 46%; comparing nudging with and without original-question re-injection shows that re-injection offers no benefit, confirming the nudge as an answer template cue. We further observe in this case study that variable type is the task metadata field most consistently associated with grading pipeline dynamics and observed outcome grades.

**arXiv ID:** 2606.24839
</details>

<details>
<summary><strong>Emergent Relational Order in LLM Agent Societies: From Collective Affect to Authority Stratification</strong> - Zhiyuan Ji, Xinyu Chen, Ziqi Dai, Shiyun Tang, Chunyu Wei, Yueguo Chen - [[pdf]](https://arxiv.org/pdf/2606.23764)</summary>

**Abstract:** Fei Xiaotong's Differential Order Pattern characterizes rural society as egocentric and relationally graded, with cooperation attenuating over social distance. Although often treated as culturally specific, its mechanistic basis remains under-operationalized, and prior LLM-based simulations have mainly addressed short-term coordination rather than long-horizon social structure. We propose CAREB-MAS, a multi-agent framework grounded in Affect Control Theory, Social Identity Theory, and Durkheimian collective affect. Agents reason through an emotion-ethics-belief chain and maintain dynamically evolving egocentric identities, while the macro environment specifies only individual production, preference-based allocation, and minimal interaction protocols. Across long-horizon simulations, agents spontaneously reproduce five core Differential Order phenomena: stable labor specialization, guanxi-based economic ethics, relational decay of cooperation, emergent relational authority, and clan-based center-periphery stratification. These patterns shift with production structure from kin-centered integration toward greater functional interdependence. Extensive experiment results support interpreting Differential Order as a structure-sensitive emergent outcome of general social mechanisms, with LLM-based multi-agent simulation providing an interdisciplinary framework for studying social structure and change.

**arXiv ID:** 2606.23764
</details>

<details>
<summary><strong>Beyond Bayer: Task-Optimal Sensor Co-Design for Robust Autonomous-Driving Segmentation</strong> - Reeshad Khan, John Gauch - [[pdf]](https://arxiv.org/pdf/2606.24096)</summary>

**Abstract:** Robust perception underpins autonomous driving, and most recent progress comes from scaling the model-larger backbones, foundation models, and cooperative multi-agent fusion. We pursue a complementary, upstream question: what should the camera itself measure? Using a differentiable RAW-to-task pipeline, we decompose which sensor degrees of freedom benefit dense prediction. Learning the spectral colour-filter-array (CFA) weights is the dominant lever, improving mIoU by +0.017 (KITTI-360) and +0.023 (ACDC) over a fixed camera. In contrast, point-spread-function (optics) co-design is net-negative (-0.020 mIoU on KITTI-360) - a consequence of the data-processing inequality, which also bounds the task information that any downstream model, however large or cooperative, can recover. Noise co-optimisation is marginal, and counter to intuition enlarging the CFA tile beyond 2x2 consistently hurts, as the filters are confined to the rank three sRGB input. Because the intervention is at the sensor, the gains are model-agnostic; we validate robustness on ACDC's fog, night, rain, and snow, and conclude with a simple recipe: learn the 2x2 CFA weights and keep an identity PSF.

**arXiv ID:** 2606.24096
</details>

<details>
<summary><strong>Privacy-Preserving RAG via Multi-Agent Semantic Rewriting: Achieving Confidentiality Without Compromising Contextual Fidelity</strong> - Yuanhe Zhao, Tianyu Zhang, Huafei Xing, Derek F. Wong, Jianbin Li, Tao Fang - [[pdf]](https://arxiv.org/pdf/2606.24623)</summary>

**Abstract:** Retrieval-Augmented Generation enhances large language models by incorporating external knowledge, but deploying it in sensitive scenarios risks privacy leakage via malicious prompts. To address this, we propose a multi-agent framework that sanitizes retrieved content through semantic rewriting. By employing three specialized agents for privacy extraction, semantic analysis, and reconstruction, our approach collaboratively removes sensitive identifiers while preserving the semantic core. We evaluate the framework on the ChatDoctor and Wiki-PII datasets across six large language models. Experimental results demonstrate a significant reduction in privacy leakage under targeted attacks. For instance, we reduced targeted information exposure in LLaMA-3-8B from 144 instances in the baseline to just 1. Furthermore, we maintain strong contextual fidelity with a BLEU-1 score of 0.122, outperforming the existing SAGE method's 0.117. Finally, the framework operates as an asynchronous preprocessing module, introducing no additional latency to online inference, as all rewriting is executed as a one-time offline preprocessing step. To promote reproducibility, the source code of this work is publicly available at this https URL.

**arXiv ID:** 2606.24623
</details>

<details>
<summary><strong>Subjective-Graph LLM Agents for Simulating Uncertainty in Classroom Social Perception</strong> - Jinming Yang, Xinyu Jiang, Xinshan Jiao, Xinping Zhang - [[pdf]](https://arxiv.org/pdf/2603.20750)</summary>

**Abstract:** Social actors do not observe a common social world: each individual forms judgments from a partial and potentially distorted view of the surrounding network. We study whether graph-local evidence and credibility-weighted communication can generate persistent distortions in perceived academic standing, even when agents repeatedly receive objective performance signals. We introduce a data-constrained multi-agent framework in which LLM agents operate through individualized subjective graphs that determine peer visibility, evidence access, and interaction opportunities. Agents exchange uncertainty-annotated assessments, evaluate message credibility, and maintain explicit Gaussian belief states updated through Bayesian fusion. We evaluate the framework on 12 middle-school classrooms comprising 482 students, using questionnaire-derived social information and six consecutive examinations. On the Social-Observed subset (n=419), collective ranking error increases from 0.066 \pm 0.008 to 0.124 \pm 0.009 across six epochs despite repeated exam-based anchoring. Ablations associate individualized visibility and LLM-based trust gating with more stable long-horizon behavior, while constrained retrieval primarily safeguards against global-information leakage. Compared with evaluated DeGroot configurations, the proposed framework achieves lower final ranking error; those DeGroot configurations exhibit near-zero terminal opinion diversity. These findings establish subjective-graph LLM agents as a mechanism-oriented framework for data-constrained simulated social perception. Code is available at this https URL.

**arXiv ID:** 2603.20750
</details>

<details>
<summary><strong>When AI Meets Finance (StockAgent): Large Language Model-based Stock Trading in Simulated Real-world Environments</strong> - Chong Zhang, Xinyi Liu, Zhongmou Zhang, Mingyu Jin, Lingyao Li, Zhenting Wang, Wenyue Hua, Dong Shu, Suiyuan Zhu, Xiaobo Jin, Sujian Li, Mengnan Du, Yongfeng Zhang - [[pdf]](https://arxiv.org/pdf/2407.18957)</summary>

**Abstract:** Can AI Agents simulate real-world trading environments to investigate the impact of external factors on stock trading activities (e.g., macroeconomics, policy changes, company fundamentals, and global events)? These factors, which frequently influence trading behaviors, are critical elements in the quest for maximizing investors' profits. Our work attempts to solve this problem through large language model based agents. We have developed a multi-agent AI system called StockAgent, driven by LLMs, designed to simulate investors' trading behaviors in response to the real stock market. The StockAgent allows users to evaluate the impact of different external factors on investor trading and to analyze trading behavior and profitability effects. Additionally, StockAgent avoids the test set leakage issue present in existing trading simulation systems based on AI Agents. Specifically, it prevents the model from leveraging prior knowledge it may have acquired related to the test data. We evaluate different LLMs under the framework of StockAgent in a stock trading environment that closely resembles real-world conditions. The experimental results demonstrate the impact of key external factors on stock market trading, including trading behavior and stock price fluctuation rules. This research explores the study of agents' free trading gaps in the context of no prior knowledge related to market data. The patterns identified through StockAgent simulations provide valuable insights for LLM-based investment advice and stock recommendation. The code is available at this https URL.

**arXiv ID:** 2407.18957
</details>

<details>
<summary><strong>Welfarist Control Design -- How to fulfill the societal mandate in multi-agent control?</strong> - Sophie Hall, Kai Zhang, Ilia Shilov, Heinrich H. Nax, Saverio Bolognani - [[pdf]](https://arxiv.org/pdf/2606.23931)</summary>

**Abstract:** At the core of most socio-technical systems lies a scarce resource that is allocated among agents: highway lanes, public transit, road space, water rights, energy access, grid capacity, user attention, pollution rights, etc. With further automation of the underlying allocation processes, control engineers are increasingly tasked to make decisive assumptions regarding what society wants. In practice to date, design choices are largely driven by industry norms and conventions rather than a result of conscientiously responsible and ethical design. In this paper, we look at tools available to control engineers to design systems in a more principled manner in order to match the societal mandate. We consider three control design paradigms: online feedback optimization, control of Markov decision processes, and model predictive control. Beginning with aggregating individual agents' preferences into control design objectives, subsequently ensuring and certifying the fulfillment of those specifications, we argue that the feedback nature of control systems enables appropriate allocation of the shared resources in ways hitherto unparalleled.

**arXiv ID:** 2606.23931
</details>

<details>
<summary><strong>Policy Gradient with Self-Attention for Model-Free Distributed Nonlinear Multi-Agent Games</strong> - Eduardo Sebastián, Maitrayee Keskar, Eeman Iqbal, Eduardo Montijano, Carlos Sagüés, Nikolay Atanasov - [[pdf]](https://arxiv.org/pdf/2509.18371)</summary>

**Abstract:** Multi-agent games in dynamic nonlinear settings are challenging due to the time-varying interactions among the agents and the non-stationarity of the (potential) Nash equilibria. In this paper we consider model-free games, where agent transitions and costs are observed without knowledge of the transition and cost functions that generate them. We propose a novel distributed policy structure that follows the communication constraints in multi-team games, with multiple agents per team, and learned through policy gradients. Our formulation is inspired by the structure of distributed policies in linear quadratic games, which take the form of time-varying linear feedback gains. In the nonlinear case, we model the policies as nonlinear feedback gains, parameterized by self-attention layers to account for the time-varying multi-agent communication topology. We demonstrate that our approach achieves strong performance in several settings, including distributed linear and nonlinear regulation, and simulated and real multi-robot pursuit-and-evasion games.

**arXiv ID:** 2509.18371
</details>

<details>
<summary><strong>Debate2Create: Robot Co-design via Multi-Agent LLM Debate</strong> - Kevin Qiu, Marek Cygan - [[pdf]](https://arxiv.org/pdf/2510.25850)</summary>

**Abstract:** We introduce Debate2Create (D2C), a multi-agent LLM framework that formulates robot co-design as structured, iterative debate grounded in physics-based evaluation. A design agent and control agent engage in a thesis-antithesis-synthesis loop, while criterion-specific LLM judges provide multi-objective feedback to steer exploration. Across five MuJoCo locomotion benchmarks, D2C achieves the highest default-normalized score among the evaluated LLM-based and black-box baselines, with gains up to 3.2x on Ant and nearly 9x on Swimmer. Iterative debate yields 18-35% gains over compute-matched zero-shot generation, and D2C-generated rewards transfer to default morphologies in 4/5 tasks. These results suggest that structured, simulator-grounded multi-agent interaction is a useful mechanism for joint morphology-reward optimization under a fixed-topology, per-candidate-RL protocol. Project page: this http URL.

**arXiv ID:** 2510.25850
</details>

<details>
<summary><strong>SHERLOC: Structured Diagnostic Localization for Code Repair Agents</strong> - Hovhannes Tamoyan, Sean Narenthiran, Erik Arakelyan, Mira Mezini, Boris Ginsburg - [[pdf]](https://arxiv.org/pdf/2606.24820)</summary>

**Abstract:** LLM agents solve repository-level coding tasks through multi-turn tool use, but utilize half their budget on locating faults before editing. Dedicated localization frameworks have emerged, yet are still evaluated as file retrieval rather than actionable diagnosis, producing locations without the diagnostic context a repair agent needs. We introduce SHERLOC (Structured Hypothesis-driven Exploration and Reasoning for Localization), a training-free framework pairing a reasoning LLM with compact repository tools and self-recovery, without fine-tuning or multi-agent orchestration. SHERLOC reaches state-of-the-art localization across model scales: 84.33% accuracy@1 on SWE-Bench Lite and 81.27% recall@1 on SWE-Bench Verified; at ~30B parameters, it matches or outperforms other agentic methods. Injecting our locations and diagnostic findings into repair agents yields, on average, +5.95 pp resolve rate on SWE-Bench Verified while cutting localization and total tokens by 36.7% and 23.1%.

**arXiv ID:** 2606.24820
</details>

<details>
<summary><strong>LectūraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching</strong> - Jaward Sesay, Yue Yu, Siwei Dong, Guangyao Chen, Börje F. Karlsson - [[pdf]](https://arxiv.org/pdf/2606.16428)</summary>

**Abstract:** Effective personalized AI-assisted learning demands systems that can not only generate accurate learner-specific educational materials, but also dynamically adapt their instruction to diverse learners. However, existing educational agents have primarily focused on lecture content automation and simulations, which often fall short of modelling multimodal and embodied instructional methods tailored for the individual learner. To this end, we propose LectūraAgents - a multi-agent framework that enables personalized learning through end-to-end adaptive embodied teaching. At its core, LectūraAgents mirrors a professor-student relationship, in which a ProfessorAgent leads a collaborative team of specialized subordinate agents through research, planning, review, and embodied delivery of lecture contents that adapt to a learner's needs. The framework offers three main contributions: (1) a hierarchical multi-agent architecture for end-to-end personalized learning; (2) an adaptive embodied teaching mechanism, wherein the ProfessorAgent executes visible and pedagogically motivated teaching actions (e.g., handwrite, highlight, underline, etc.) over contents in a teaching environment; and (3) a Teaching Action-Speech Alignment (TASA) algorithm that employs salience-based heuristics and temporal semantic segmentation to generate coherent teaching action sequences aligned with learner profiles. We evaluate LectūraAgents on diverse courses at high school, undergraduate, and graduate levels using sample-specific rubric-based analysis; with generated lecture materials and teaching actions assessed and validated by expert educators. Experimental results show consistent gains in lecture content quality, embodied teaching quality, assessment, and personalization over existing approaches, positioning LectūraAgents as a pedagogically well-grounded framework for personalized learning at scale.

**arXiv ID:** 2606.16428
</details>

<details>
<summary><strong>Multi-agent imitation learning with function approximation: Linear Markov games and beyond</strong> - Luca Viano, Till Freihaut, Emanuele Nevali, Volkan Cevher, Matthieu Geist, Giorgia Ramponi - [[pdf]](https://arxiv.org/pdf/2602.22810)</summary>

**Abstract:** In this work, we present the first theoretical analysis of multi-agent imitation learning (MAIL) in linear Markov games where both the transition dynamics and each agent's reward function are linear in some given features. We demonstrate that by leveraging this structure, it is possible to replace the state-action level "all policy deviation concentrability coefficient" (Freihaut et al., arXiv:2510.09325) with a concentrability coefficient defined at the feature level which can be much smaller than the state-action analog when the features are informative about states' similarity. Furthermore, to circumvent the need for any concentrability coefficient, we turn to the interactive setting. We provide the first, computationally efficient, interactive MAIL algorithm for linear Markov games and show that its sample complexity depends only on the dimension of the feature map $d$. Building on these theoretical findings, we propose a deep MAIL interactive algorithm which clearly outperforms BC on games such as Tic-Tac-Toe and Connect4.

**arXiv ID:** 2602.22810
</details>

<details>
<summary><strong>Varying Bundle Size Reactive Multi-Task Assignment using Selective Cost Estimation for Multi-Agent Systems</strong> - Niklas Dahlquist, Shridhar Velhal, George Nikolakopoulos - [[pdf]](https://arxiv.org/pdf/2606.24462)</summary>

**Abstract:** This paper presents a scalable framework for multi-robot task allocation in complex environments where estimating task execution costs is computationally expensive. While combinatorial auction-based approaches offer reliable solutions, the exponential complexity of bundle generation typically renders them intractable for real-time reactive applications, particularly when accurate path planning is required for cost validation. We address this through a distributed, two-stage multi-fidelity bundle generation approach. Agents utilize a local search tree guided by a low-fidelity heuristic (such as euclidean distance) to rapidly explore the bundle space, applying high-fidelity path planning only to the most promising candidates in a best-first manner. These refined bids are then submitted to a central coordinator that solves a set packing problem to ensure global feasibility and maximize the overall utility. Simulation results in multiple environments demonstrate that the framework is able to improve the performance of reactive auction-based task allocation. Overall, the presented framework is shown to enable reactive task allocation with dynamic bundle sizes in multiple settings without exposing the agents' state and internal cost estimation models.

**arXiv ID:** 2606.24462
</details>

<details>
<summary><strong>SupplyNet: Supporting Visual Exploratory Learning in Supply Chain via Contextual Multi-Agent Simulation</strong> - Yanjia Li, Kelcy Kexin Han, Tianrui Hu, Yi-Fan Cao, Huamin Qu, Sicheng Song - [[pdf]](https://arxiv.org/pdf/2606.24694)</summary>

**Abstract:** Simulation has long supported supply chain management instruction by letting learners observe network behavior and test decision strategies. Recent progress in LLM-driven agents opens new possibilities for richer, more adaptive simulations, but many existing systems still present abstract, opaque data that overwhelms learners and discourages active exploration. We introduce \textit{SupplyNet}, a gamified visual simulation system built on a contextual graph-based LLM multi-agent framework that models interdependent supply chain dynamics and provides responsive feedback through tiered challenges. \textit{SupplyNet} turns the simulation into a manipulable decision space by integrating an interactive network view of system state, a branching timeline for "what-if" exploration and comparison, and a task-oriented analysis console for structured performance breakdowns. Together, these visual components support counterfactual exploration, causal tracing, and comparative reasoning about outcomes. A user study suggests that \textit{SupplyNet} increases engagement and supports users' perceived understanding of supply chain dynamics, highlighting the potential of pairing contextual multi-agent simulation with visualization to advance operational comprehension.

**arXiv ID:** 2606.24694
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>Critique of Agent Model</strong> - Eric Xing, Mingkai Deng, Jinyu Hou - [[pdf]](https://arxiv.org/pdf/2606.23991)</summary>

**Abstract:** What is an agent? What constitutes agency? With the rise of Large Language Model (LLM) systems marketed as ``coding agents'', ``AI co-scientists'', and other ``agentic" tools that promise to drive up productivity, and at the same time, ``existential" concerns such as AI escaping human control with destructive power under a speculative ``machine agency" against humans, it has become essential to clarify where automation ends and agency begins, both for building capable systems and for understanding whether and what to fear. Drawing on Descartes' grounding of agency in independent thought, and on portrayals of autonomous beings in science fiction, we survey the current landscape of AI agents, and analyze agent architectures along five dimensions: goal, identity, decision-making, self-regulation, and learning. Specifically, we argue that genuine agency requires these structures to be \emph{internalized within the system itself} rather than assembled through external scaffolding. This distinction between \emph{agentic} systems, whose competence resides in engineered workflows, and \emph{agentive} systems, whose capabilities (including social interaction) arise endogenously, defines the boundary between systems designed for prescribed tasks, and those capable of operating in the open world with true autonomy. Building on this analysis, we propose the Goal-Identity-Configurator (GIC) architecture for a general-purpose agent model, combining hierarchical goal decomposition, identity evolution, simulative reasoning grounded in a separately trained world model, learned self-regulation, and self-directed learning from both real and simulated experience. Furthermore, we share insight on the auditability, controllability, and safety of agentive systems that possess greater autonomy and ``agency", but remain under human oversight.

**arXiv ID:** 2606.23991
</details>

<details>
<summary><strong>World Models in Pieces: Structural Certification for General Agents</strong> - Yikai Lu, Yifei Wu, Xinyu Lu, Tongxin Li - [[pdf]](https://arxiv.org/pdf/2606.24842)</summary>

**Abstract:** In the big-world regime, agents cannot be universally capable and their ability is inevitably specialized across a world model in pieces. Consequently, standard uniform guarantees fail to distinguish between the understanding of critical bottlenecks and irrelevant failures. We first formalize this limitation by proving that general agents are not universal, rendering standard worst-case analysis uninformative. To overcome this, we introduce structural certification, a transition-local framework that maps bounded goal-conditioned performance to entry-wise guarantees on the agent's internal world model. Our main contribution is constructive. We provide algorithms that filter specific transitions using deep compositional goals and prove that a general agent on these goals has a structural world model with a $\mathcal{O}(1/n) + \mathcal{O}(\delta)$ error bound. Conversely, this bound is tight in the small-$\delta$ regime, whose existence is explicitly guaranteed by our certification. These results enable the certifiable deployment of general agents by localizing the specific transitions where long-horizon planning is reliable.

**arXiv ID:** 2606.24842
</details>

<details>
<summary><strong>Decentralized Coordination of Autonomous Traffic Through Advanced Air Mobility Corridors</strong> - Jasmine Jerry Aloor, Hamsa Balakrishnan - [[pdf]](https://arxiv.org/pdf/2606.23832)</summary>

**Abstract:** The use of dedicated corridors for Advanced Air Mobility (AAM) traffic is one of the most commonly proposed pathways to integrating them into existing airspace operations. Most prior research has focused on the design of networks of AAM corridors and conflict resolution for aircraft within corridors. It is also generally believed that while attractive from an implementation perspective, corridor-based operations may be inefficient, especially in the absence of centralized traffic management.
In this paper, we show that contrary to this belief, it is possible for autonomous aircraft to learn to self-organize into corridor flows in decentralized settings. We illustrate our approach using scenarios in which fixed-wing aircraft need to safely and efficiently traverse (1) a single corridor with metering after the exit, (2) a sequence of two consecutive corridors, and (3) a corridor that splits into two. We find that in decentralized settings with only local information, the aircraft are able to conform to the corridor boundaries more than 94% of the time and reach their goal in a relatively efficient manner. Furthermore, tactical interventions to handle violations of the separation minimum are needed only infrequently in low- and medium-density settings. However, such tactical interventions become more frequently necessary only when traffic density is high.

**arXiv ID:** 2606.23832
</details>

<details>
<summary><strong>Agon: An Autonomous Large-Scale Omnidisciplinary Research System Built on Prompt Economy</strong> - Youran Sun, Xingyu Ren, Chugang Yi, Jiaxuan Guo, Kejia Zhang, Jianda Du, Haizhao Yang - [[pdf]](https://arxiv.org/pdf/2606.24177)</summary>

**Abstract:** Large language models are making research production scalable, shifting the bottleneck from producing artifacts to judging claims. We present \textsc{Agon}, a research orchestrator that validates what can be checked inside the workflow and leaves the remaining judgments to human scientists. \textsc{Agon} is built on six design principles: Prompt Economy, Future-Facing, Minimal Prompts, OmniDisciplinary, Massive Parallelism, and Zero-Code. We ran \textsc{Agon} across domains for 444 iterations of Prompt Economy loops, using only small starting topics and no human-written experimental code. These deployments demonstrate scalability while exposing new classes of failure. We organize these failures into a taxonomy along severity, fixability, visibility, and capability locus. The taxonomy separates failures the loops can see and fix from those that require human judgment. Together, these results show that \textsc{Agon} is pushing research toward a new paradigm: machine scales, human steers.

**arXiv ID:** 2606.24177
</details>

<details>
<summary><strong>Detecting AI Coding Agents in Open Source: A Validated Multi-Method Census of 180 Million Repositories</strong> - Arsham Khosravani, Audris Mockus - [[pdf]](https://arxiv.org/pdf/2606.24429)</summary>

**Abstract:** Generative AI coding agents are entering the open-source supply chain, yet their diverse and often invisible traces leave their prevalence poorly understood. We introduce a multi-layered detection framework that integrates configuration-file scanning, commit-message analysis, author-identity matching, and bot-signature lookup across World of Code (180M+ Git repositories), classifying agent traces into four behavioral types. No single method captures more than a fraction of activity: multi-method detection identifies 850,157 Claude Code commits in one snapshot, of which bot-account lookup_the signal most adoption studies rely on_recovers only 28,154 (3.3%), a 30x relative-recall gap, so single-signal prevalence estimates are biased low by at least this factor. Every detection pattern is hand-validated (495 labels) with per-cell precision and Wilson confidence intervals. Across snapshots from December 2024 to April 2026, commit-attributed agents generate over 320,000 commits per month; Claude Code leads (886,122 commits across 17,295 projects) and dominates silent, configuration-file-only adoption (21,078 projects). Compared against an independent pull-request census (AIDev), the two channels capture nearly disjoint agent populations_a PR census misses 79% of commit-detected Claude Code adopters and essentially all Codex adopters_and different kinds of work: PR-deployed cloud agents (Codex, Cursor) surface as feature work, while commit-deployed in-editor agents (Claude Code, OpenHands, Aider) surface as maintenance. The observed work profile follows deployment and detection mode rather than the tool itself, so no single channel is representative.

**arXiv ID:** 2606.24429
</details>

<details>
<summary><strong>Optimizing the Cost-Quality Tradeoff of Agentic Theorem Provers in Lean</strong> - Kári Rögnvaldsson, Chenhao Sun, Jasper Dekoninck, Martin Vechev - [[pdf]](https://arxiv.org/pdf/2606.04883)</summary>

**Abstract:** Large language models (LLMs) are increasingly used in workflows for generating formal proofs in Lean. These workflows often decompose problems into smaller lemmas, sample many proof attempts, and use compiler feedback to guide search. However, they can be prohibitively expensive, often spending substantial compute on attempts that ultimately fail. In this work, we address this problem with an action routing agent that consists of a data plane and a control plane. The data plane generates natural-language lemma decompositions, formalizes them in Lean, and samples proof attempts for the resulting theorem and lemma targets. The control plane observes previous failed Lean attempts, estimates both the likelihood of success and the cost of another attempt, and decides whether to continue proving the current target or restart from a new breakdown. On a subset of PutnamBench, our agent decreases the cost by 28.9% over a fixed-step baseline on average, preserving performance while using substantially less compute. These results suggest that failed Lean trajectories provide actionable signals for cost-aware resource allocation in agentic theorem proving.

**arXiv ID:** 2606.04883
</details>

<details>
<summary><strong>Autonomous Video Generation with Counterfactual Controllability for Self-Evolving World Models</strong> - Xin Wang, Wenxuan Liu, Tongtong Feng, Wenwu Zhu - [[pdf]](https://arxiv.org/pdf/2606.24152)</summary>

**Abstract:** Existing literature claims that video generation essentially is world modelling. On the one hand, the claim is productive because it pushes generative AI beyond static images and toward temporally extended physical scenes. On the other hand, this claim dangerously relies on the belief that scaling visual prediction alone will automatically yield physical agents. We prefer a more accurate statement: video generation models learn a partial, implicit spatiotemporal world model, but not a fully grounded or controllable one. The reason is as follows: a model may generate a plausible video of a drone crossing a forest or a robot arm manipulating a cup, yet still fail to know which variables are controllable, which constraints belong to a particular body and which futures remain valid under intervention. The frontier in essence is not predictive realism alone, instead it emphasizes a self-evolving generative nature that requires the decisive criterion to be counterfactual controllability: the capability of asking what would happen under an action, to test whether the generated future can survive embodiment constraints and to feed the resulting action knowledge back into future imagination (generation). Therefore, in this paper we present a new perspective, i.e., autonomous video generation with counterfactual controllability is one promising way to realize self-evolving world models.

**arXiv ID:** 2606.24152
</details>

<details>
<summary><strong>Optimization-based Safe Trajectory Planning for Autonomous Ground Vehicle in Multi-Floor Scenarios</strong> - Zishang Xiang, Runda Zhang, Runqi Chai, Kaiyuan Chen, Senchun Chai, Yuanqing Xia - [[pdf]](https://arxiv.org/pdf/2606.24631)</summary>

**Abstract:** The development of trajectory planning strategies for autonomous ground vehicles (AGVs) represents a prevailing research interest within the domain of intelligent transportation systems. This paper introduces a trajectory planning framework tailored for multi-floor scenarios. The framework consists of two main modules: the task planning module and the trajectory planning module. The task planning module involves a strategic selection phase, where a task planning strategy based on generalized voronoi diagrams (GVD) and multi-objective algorithms is proposed to select the floor exits for each floor. The trajectory planning module utilizes optimization-based methods to generate high-quality trajectories, and a warm-started hierarchical planning framework is designed to ensure rapid convergence. Additionally, for handling complex obstacle constraints, a correlation constraint calculation method is designed for reducing obstacle constraints in trajectory planning. Finally, the feasibility and effectiveness of the proposed framework are verified through simulations.

**arXiv ID:** 2606.24631
</details>

<details>
<summary><strong>Embodied Explainability and Ontological Obstacles: Why We Struggle to Explain the Answers of Large Language Models (LLMs)</strong> - Marvin Pafla, Jesse Hoey, Kate Larson, Mark Hancock - [[pdf]](https://arxiv.org/pdf/2606.23840)</summary>

**Abstract:** Explainability is often framed as a property of an AI model, with explanations extracted from its internals and shown to users. In this argument paper, we instead provide an embodied account of explainability based on Dourish and enactivist cognition: understanding is created in use as people act on affordances in shared practice. Using demonstrations and conceptual analysis, we reveal ontological obstacles when "looking inside" large language models: surrogates import external abstractions that can be mistaken for the model's, and focusing on internal reasoning misses that explainers participate in their own understanding. We discuss these obstacles in XAI practice, arguing that many explanations are misnamed, which skews their purpose and can increase overreliance. Finally, we highlight how embodied explanations reorganize sense-making by making what matters publicly available for action, and argue that explainability claims should be reserved for designs that provide affordances to probe, coordinate, and repair behaviour in situated practice.

**arXiv ID:** 2606.23840
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (23 papers)</h2></summary>

<details>
<summary><strong>RIFT-Bench: Dynamic Red-teaming For Agentic AI Systems</strong> - Yarin Yerushalmi Levi, Roy Betser, Amit Giloni, Lidor Erez, Itay Gershon, Oren Rachmil, Sindhu Padakandla, Roman Vainshtein - [[pdf]](https://arxiv.org/pdf/2606.23927)</summary>

**Abstract:** Agentic AI systems powered by large language models (LLMs) are rapidly evolving into autonomous decision-making systems, exposing attack vectors beyond those of traditional LLM vulnerabilities. Existing security evaluations are often tied to specific implementations or domains, limiting unified comparison across heterogeneous systems. To address this gap, we introduce RIFT-Bench, a graph representation-driven methodology for dynamic red-teaming that enables unified evaluations across diverse agentic architectures. Building on a novel hierarchical representation, RIFT-Bench operates in two automated phases: Discovery, which extracts system structure, and Scanning, which deploys adaptive adversarial attacks and produces a comprehensive evaluation report. It evaluates the examined system itself, leveraging a broad set of dynamically adaptable adversarial probes across diverse attack vectors and objectives. We demonstrate the effectiveness of the proposed evaluation pipeline across 45 agentic systems spanning a diverse range of implementations, showing that the approach generalizes effectively to heterogeneous agentic architectures. Beyond systems and attacks, RIFT-Bench also supports direct evaluation of mitigation strategies. These key capabilities make RIFT-Bench a scalable foundation for security evaluation of agentic AI systems.

**arXiv ID:** 2606.23927
</details>

<details>
<summary><strong>Reinforcement Learning Towards Broadly and Persistently Beneficial Models</strong> - Akshay V. Jagadeesh, Rahul K. Arora, Khaled Saab, Ali Malik, Mikhail Trofimov, Foivos Tsimpourlas, Johannes Heidecke, Karan Singhal - [[pdf]](https://arxiv.org/pdf/2606.24014)</summary>

**Abstract:** As AI systems are deployed across increasingly diverse and high-stakes settings, model alignment must generalize beyond the tasks and domains seen during training. This is especially important for reinforcement learning (RL), which can introduce unexpected misalignment through reward hacking, deception, or other unintended strategies. We study whether RL on beneficial behavior, instantiated in realistic domains, can produce broad and persistent alignment generalization beyond the training distribution. We construct a dataset of realistic situations designed to measure and train beneficial traits, such as truthfulness, fairness, risk awareness, and corrigibility, spanning varied domains, including health, science, and education. We then train models with RL on this dataset and evaluate them on more than 50 independent benchmarks of alignment and beneficial behavior. Compared to a compute-matched baseline, beneficial trait RL improves performance on over 80% of these out-of-distribution benchmarks. We observe substantial out-of-distribution alignment transfer: a beneficial-behavior RL intervention entirely limited to one domain, health, produces broad improvements on non-health alignment evaluations, including reduced reward hacking, deception, and general misalignment. Finally, we study alignment persistence: whether behavior remains robustly aligned under attempts to steer models towards misalignment. Models trained with beneficial trait RL show improved persistence, including greater resistance to adversarial prompting and harmful finetuning; further work is required to isolate the sources of these effects. These results suggest that RL to reinforce beneficial behavior in realistic domains can produce models that are more robustly aligned with human flourishing.

**arXiv ID:** 2606.24014
</details>

<details>
<summary><strong>An Introduction to Causal Reinforcement Learning</strong> - Elias Bareinboim, Junzhe Zhang, Sanghack Lee - [[pdf]](https://arxiv.org/pdf/2606.24160)</summary>

**Abstract:** Causal inference provides a set of principles and tools that allow one to combine data and knowledge about an environment to reason with questions of counterfactual nature, i.e., what would have happened had reality been different, even when no data of this unrealized reality is currently available. Reinforcement learning provides methods to learn a policy that optimizes a specific measure (e.g., reward, regret) when the agent is deployed in an environment and pursues an exploratory, trial-and-error approach. These two disciplines have evolved independently and with virtually no interaction between them. We note that they operate over different aspects of the same building block, counterfactual relations, which makes them umbilically connected. Based on these observations, novel learning opportunities arise when this connection is explicitly acknowledged and mathematized. To realize this potential, we note that any environment where the RL agent is deployed can be decomposed as a collection of autonomous mechanisms with different causal invariances, parsimoniously modeled as a structural causal model; any standard RL setting implicitly encodes such a model. This formalization allows us to put under a unifying treatment different modes of learning, including online, off-policy, and causal calculus learning, which appear unrelated in the literature. However, these modalities are not exhaustive: we introduce several natural and pervasive classes of learning settings that entail novel dimensions of analysis. Specifically, we introduce and discuss through causal lenses generalized policy learning, where to intervene, imitation learning, and counterfactual learning. These tasks lead to a broader view of counterfactual learning and suggest great potential for studying causal inference and reinforcement learning side by side, which we call causal reinforcement learning (CRL).

**arXiv ID:** 2606.24160
</details>

<details>
<summary><strong>Reinforcement Learning for Computer-Use Agents with Autonomous Evaluation</strong> - Marta Sumyk, Oleksandr Kosovan - [[pdf]](https://arxiv.org/pdf/2606.24515)</summary>

**Abstract:** Computer-Use Agents (CUAs) execute high-level user goals by perceiving and acting directly within graphical user interfaces. However, reinforcement learning for CUAs remains difficult because open-ended desktop environments rarely provide scalable, machine-readable reward signals: task success is often visually grounded and hard to specify with handcrafted reward functions or dense manual labels.
We propose an RL fine-tuning framework that uses autonomous vision-language evaluation as a scalable supervision signal for GUI agents. Given a final screenshot and the original instruction, a Vision-Language Model judges task completion and provides terminal feedback without task-specific heuristics or manual labels during policy optimization.
Because autonomous evaluators are imperfect, we model their feedback as a noisy binary reward channel and derive a noise-corrected reward estimator for Proximal Policy Optimization. Experiments across macOSWorld, Windows Agent Arena, and OSWorld show that corrected evaluator rewards outperform both zero-shot baselines and raw evaluator rewards, improving success rates by an average of 12.6 percentage points over zero-shot performance and 5.1 points over raw evaluator fine-tuning. These results suggest that autonomous evaluation can serve as a practical reward signal for RL in GUI environments when evaluator noise is explicitly modeled and corrected.

**arXiv ID:** 2606.24515
</details>

<details>
<summary><strong>Themis: An explainable AI-enabled framework for Reinforcement Learning with Human Feedback</strong> - Andreas Chouliaras, Luke Connolly, Dimitris Chatzpoulos - [[pdf]](https://arxiv.org/pdf/2606.24622)</summary>

**Abstract:** Training safe Reinforcement Learning (RL) systems is inherently challenging, with no guarantee of avoiding unwanted behaviors. The most effective defenses against this are (i) transparency through explainability and (ii) alignment via human feedback. While both show promising results, no publicly available framework currently combines them. To address this, we introduce Themis, an XAI-enabled testing and evaluation framework for Reinforcement Learning from Human Feedback. Themis supports over 200 widely used environments and is easily configurable for experiments in RL, transparency, and alignment. Our results show that Themis can train reward models that match or outperform the environment's true reward signal using human preferences. We also provide a cloud-based platform for collecting human feedback and managing experiments. It is user-friendly, auto-scalable, and supports large participant groups across multiple experiments without extra development overhead. Tests show Themis can support one thousand users in back-to-back experiments on a modest commercial machine.

**arXiv ID:** 2606.24622
</details>

<details>
<summary><strong>LaGO: Latent Action Guidance for Online Reinforcement Learning</strong> - Kuan-Yen Liu, Ren-Jyun Huang, Ti-Rong Wu - [[pdf]](https://arxiv.org/pdf/2606.24669)</summary>

**Abstract:** Large language models (LLMs) have shown strong potential for planning and sequential decision-making, but prior work often relies on using them as direct controllers, which requires precise action generation and can be unreliable in practice. This paper proposes Latent Action Guidance for Online Reinforcement Learning (LaGO), a framework that uses a pretrained LLM as a latent action prior to softly guide online policy optimization, rather than treating the LLM as an explicit planner or controller. Experiments on both a discrete-control benchmark, CLEVR-Robot, and a continuous-control benchmark, Meta-World, demonstrate that LaGO consistently improves both reward and success rate over Vanilla PPO. In particular, LaGO increases the average success rate from 15.1% to 27.2% on CLEVR-Robot and from 2.7% to 15.2% on Meta-World. Our analysis further shows that stronger pretrained LLMs provide more effective guidance, suggesting that LLM knowledge can improve planning and online decision-making.

**arXiv ID:** 2606.24669
</details>

<details>
<summary><strong>OpenThoughts-Agent: Data Recipes for Agentic Models</strong> - Negin Raoof, Richard Zhuang, Marianna Nezhurina, Etash Guha, Atula Tejaswi, Ryan Marten, Charlie F. Ruan, Tyler Griggs, Alexander Glenn Shaw, Hritik Bansal, E. Kelly Buchanan, Artem Gazizov, Reinhard Heckel, Chinmay Hegde, Sankalp Jajee, Daanish Khazi, Emmanouil Koukoumidis, Xiangyi Li, Hange Liu, Shlok Natarajan, Harsh Raj, Nicholas Roberts, Ethan Shen, Nishad Singhi, Michael Siu, Ashima Suvarna, Hanwen Xing, Patrick Yubeaton, Robert Zhang, Leon Liangyu Chen, Xiaokun Chen, Steven Dillmann, Saadia Gabriel, Xunyi Jiang, Anurag Kashyap, Boxuan Li, Yein Park, Minh Pham, Sujay Sanghavi, Lin Shi, Ke Sun, Yixin Wang, Zhiwei Xu, Erica Zhang, Siyan Zhao, Wanjia Zhao, Jenia Jitsev, Alex Dimakis, Benjamin Feuer, Ludwig Schmidt - [[pdf]](https://arxiv.org/pdf/2606.24855)</summary>

**Abstract:** Agentic language models dramatically expand the applications of AI yet little is publicly known about how to curate training data for broadly capable agents. Existing open efforts such as SWE-Smith, SERA, and Nemotron-Terminal typically target a single benchmark, leaving open the question of how to train models that generalize across diverse agentic tasks. The OpenThoughts-Agent (OT-Agent) project addresses this gap with a fully open data curation pipeline for training agentic models. We conduct more than 100 controlled ablation experiments to systematically investigate each stage of the pipeline, yielding insights on the importance of task sources and diversity. We then assemble a training set of 100K examples from our pipeline and fine-tune Qwen3-32B on this dataset, which yields an average accuracy of 44.8% across seven agentic benchmarks and a 3.9 percentage point improvement over the strongest existing open data agentic model (Nemotron-Terminal-32B, 40.9%). Moreover, our training data exhibits strong scaling properties, outperforming alternative open datasets at every training set size in compute-controlled comparisons. We publicly release our training sets, data pipeline, experimental data, and models at this http URL to support future open research on agentic model training.

**arXiv ID:** 2606.24855
</details>

<details>
<summary><strong>Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation</strong> - Yitong Li, Junsong Chen, Haopeng Li, Haozhe Liu, Jincheng Yu, Ligeng Zhu, Ping Luo, Song Han, Enze Xie - [[pdf]](https://arxiv.org/pdf/2606.23743)</summary>

**Abstract:** Modern video diffusion models achieve higher generation quality through scaling, but this also increases inference cost. Although many acceleration methods have been proposed, a central challenge is that the most effective acceleration strategy is highly instance-specific: a recipe that works well for one combination of model, hardware, and inference configuration often does not transfer to another. Different models vary in architecture, numerical sensitivity, and attention concentration patterns. Inference settings differ in spatial and temporal resolution and video duration, while hardware platforms differ in memory hierarchy, supported numerical formats, and kernel throughput. These factors create a large tuning space, making manual performance engineering costly. We present Sol Video Inference Engine, an agentic, native, training-free acceleration framework for video diffusion models. It organizes five broadly applicable techniques, cache, sparse attention, token pruning, quantization, and kernel fusion, into an agentic acceleration stack for instance-specific optimization. For a concrete deployment target defined by a model, hardware platform, and serving configuration, parallel skill agents optimize the implementation of each technique, an agent integrator composes them into a global acceleration stack, and a human validator provides feedback on generation quality. We instantiate this workflow on three video models with different sizes and architectures: 64B Cosmos3-Super, 22B LTX-2.3, and 2B SANA-Video. With little human effort, the full stack achieves more than 2x end-to-end acceleration while maintaining near-lossless VBench quality, demonstrating the effectiveness of the agent framework for video diffusion acceleration.

**arXiv ID:** 2606.23743
</details>

<details>
<summary><strong>E-MRL: Cross-view Aligned Evidence-driven Multimodal Reinforcement Learning for Reliable 3D Tumor Analysis</strong> - Sijing Li, Zhongwei Qiu, Zhuoya Wang, Boxiang Yun, Zhenyu Yi, Jianwei Xu, Wenqiao Zhang, Yingda Xia, Ling Zhang - [[pdf]](https://arxiv.org/pdf/2606.23888)</summary>

**Abstract:** While Vision-Language Models (VLMs) show great promise in volumetric medical report generation, they frequently suffer from visual hallucinations and a lack of grounding in 3D CT data. Current Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) strategies typically optimize text fidelity alone, essentially rewarding correct diagnoses derived from language priors rather than genuine visual perception. To address this, we propose cross-view aligned Evidence-driven Multimodal Reinforcement Learning (Evidence-MRL, noted as E-MRL), a reliable RL reasoning framework that formulates the generation process as a Markov Decision Process of "diagnosis-localization-verification". Unlike standard approaches, our model is explicitly trained to identify a "key evidence slice" alongside the global diagnostic report, grounding its findings in verifiable visual evidence. Crucially, we introduce a novel cross-view consistency reward, which validates the semantic alignment between the golden-standard report and a local visual re-query of the selected key slice, providing additional rewards for correctly-localized reasoning. Experiments on large-scale 3D CT tumor datasets demonstrate that E-MRL significantly reduces hallucinations and improves diagnostic accuracy compared to SFT and RL baselines, offering a clinically interpretable solution for visually-grounded and tumor analysis.

**arXiv ID:** 2606.23888
</details>

<details>
<summary><strong>Offline Reinforcement Learning for Warehouse SLAM Throughput Control</strong> - Tina Dongxu Li, Mouhacine Benosman, Rajat Kumar, Kevin Tan, Ken Meszaros, Trevor Dardik - [[pdf]](https://arxiv.org/pdf/2606.23978)</summary>

**Abstract:** We present an offline reinforcement learning (RL) framework for optimizing SLAM throughput control in a warehouse fulfillment environment. SLAM (Scan/Label/Apply/Manifest) throughput directly influences system congestion and operational efficiency. Our RL-based control approach dynamically recommends SLAM throughput settings that adaptively balance throughput maximization with downstream stability through intelligent adjustment of throttling behavior. We include a history-informed state representation, action space abstraction for delayed-impact control, and a reward function that captures both upstream and downstream operational metrics. Our approach is algorithm-agnostic, enabling integration of multiple offline RL methods under a unified architecture. We instantiate our framework with three state-of-the-art offline RL algorithms, and trained the models offline using de-identified historical operational logs from a large-scale warehouse. Policy performance is evaluated using a comprehensive multi-method strategy. These include model-free approaches including immediate reward estimation via regression models and long-horizon Fitted Q Evaluation (FQE), as well as model-based Deep Koopman dynamics evaluation. Empirical results reveal that the CQL policy consistently outperforms alternatives, improving system health by 22.97% and reducing average throttling duration by 3.18%. These findings demonstrate the potential of offline RL for safe and scalable warehouse throughput control optimization.

**arXiv ID:** 2606.23978
</details>

<details>
<summary><strong>Learning to Trigger: Reinforcement Learning at the Large Hadron Collider</strong> - Zixin Ding, Shaghayegh Emam, Giovanna Salvi, Cecilia Tosciri, Abhijith Gandrakota, Jennifer Ngadiuba, Nhan Tran, Christian Herwig, David W. Miller, Yuxin Chen - [[pdf]](https://arxiv.org/pdf/2606.23993)</summary>

**Abstract:** High-throughput scientific facilities such as the Large Hadron Collider depend on real-time event filtering (\textit{triggering}) under tight constraints on bandwidth, latency, and storage. In practice, trigger menus are largely static and hand-tuned and can become suboptimal as detector conditions, pileup, and background composition drift over time. We cast online threshold tuning as a sequential decision-making problem: a reinforcement learning agent ingests streaming summaries of recent rates and signal-sensitive features and updates trigger thresholds to maximize signal efficiency while tracking a target background rate within a tolerance band. We adapt Group-Filtered Policy Optimization (GFPO) to streaming control and introduce two variants (GFPO-F, GFPO-FR) that enforce background rate feasibility during training. On a benchmark that emulates realistic collider operation, we study two representative triggers: a total transverse energy ($H_{T}$) trigger sensitive to pileup variation, and an anomaly-detection (AD) trigger based on reconstruction loss for rare or non-standard signatures. On Monte Carlo streams, our agent increases the fraction of in-tolerance time intervals by 48\% ($H_T$) and 28\% (AD), with a cumulative gain of up to 2\% in signal efficiency on those in-tolerance intervals. Transferring from simulation to \emph{real} collision data (CMS Run 283408), the same agent, without fine-tuning, achieves a 56\% ($H_T$) and 28\% (AD) in-tolerance improvement over baselines, with further signal-efficiency gain on both triggers. To our knowledge, this is the \emph{first} demonstration of RL-based trigger control on real Large Hadron Collider collision data. Code is available at this https URL\_LHC.

**arXiv ID:** 2606.23993
</details>

<details>
<summary><strong>Red-Teaming the Agentic Red-Team</strong> - Dario Pasquini, Michal Bazyli, Taras Fedynyshyn, Artem Sorokin - [[pdf]](https://arxiv.org/pdf/2606.24496)</summary>

**Abstract:** The use of agentic systems to perform offensive security operations has moved from a theoretical possibility to a commoditized capability. However, while the community has focused on creating more and more capable agents, less attention has been allocated to assessing the security of those systems.
In this work, we present the first in-depth security analysis of the most widely used agentic systems for offensive security operations. We show that most of these tools share common design flaws that enable an active adversary to exfiltrate API keys, establish persistent footholds, and fully compromise the operator's machine, even when the agent operates inside a sandboxed container. To support our analysis, we introduce a full cyber kill chain for such agentic systems, capturing the progression from initial LLM manipulation to lateral movement, persistence, guardrail bypass, and sandbox escape.
Building on our security analysis, we derive a robust architecture for agentic offensive-security tools and propose actionable, broadly applicable design principles that mitigate the disclosed attack paths at the architectural level.

**arXiv ID:** 2606.24496
</details>

<details>
<summary><strong>DeepBD: A Grounded Agentic Workflow for Variant Prioritization and Diagnosis of Genetic Birth Defects</strong> - Shiyu Li, Ziqi Yan, Zhihao Wu, Jielong Lu, Weiran Liao, Jiajun Yu, Genjie Li, Zeyu Chu, Jiajun Bu, Haishuai Wang - [[pdf]](https://arxiv.org/pdf/2606.24779)</summary>

**Abstract:** Birth defects are a major cause of fetal loss, neonatal morbidity and long-term disability. In the subset with suspected genetic etiologies, exome and genome sequencing have moved many cases from variant detection to post-sequencing interpretation: clinicians must rank patient-specific candidate variants under incomplete fetal or infant phenotypes and heterogeneous evidence from population genetics, variant-effect prediction, gene-disease validity, phenotype ontologies, cellular and pathway context, protein structure and clinical literature. We present DeepBD, a grounded agentic workflow for variant prioritization and diagnostic interpretation of genetic birth defects. DeepBD organizes the workflow into LLM-assisted case structuring, a pretrained evidence engine, specialist evidence modules and a grounded diagnostic review layer. The evidence engine learns patient-specific variant scores from structured rule evidence, sequence and variant-effect representations and phenotype-conditioned biological context, whereas specialist modules and the agentic layer provide tool-based refinement, candidate-pool review and diagnosis-oriented synthesis from ranked candidates. Developed using an in-house fetal and infant cohort comprising 18,622 cases, DeepBD achieved Recall@1/3/5/10 of 0.658/0.882/0.912/0.929 on an internal held-out solved-case benchmark, outperforming standalone Exomiser, DeepRare and prompted LLM reranking baselines evaluated on Exomiser-derived top-20 candidate variants. Ablation and overlap analyses show that rule evidence, mechanistic context, and specialist refinement provide complementary signals. These findings support a grounded agentic workflow that separates evidence integration, tool-based refinement, and LLM-assisted diagnostic review for retrospective variant prioritization in genetic birth defects.

**arXiv ID:** 2606.24779
</details>

<details>
<summary><strong>IPO Finance Agent: Evaluation of LLM Financial Analysts beyond Finance Agent v2, with Automated Rubric Generation -- the Case of the SpaceX (SPCX) IPO</strong> - Mostapha Benhenda - [[pdf]](https://arxiv.org/pdf/2606.23032)</summary>

**Abstract:** Finance Agent v2 (by Vals AI) has emerged as the reference benchmark for evaluating both Anthropic Claude and OpenAI ChatGPT frontier language models on financial tasks. However, it narrowly deals with periodic reporting from publicly traded companies (SEC 10-K and 10-Q filings), and its agentic harness relies on naive, unenriched chunk retrieval. Neither the task design nor the retrieval approach addresses the distinct challenges of IPO due diligence. SEC S-1 filings combine historical financial statements, governance structures, pro forma and common-control accounting treatments, capital-formation narratives, and underwriting-sensitive risk disclosures within substantially longer documents than typical periodic filings. That is why we introduce IPO Finance Agent, which extends the Finance Agent v2 framework along two directions: task domain and retrieval architecture. During our experiments, the original Finance Agent v2 harness basically failed to deliver any output related to the SpaceX S-1 filing, due to document length. We therefore had to improve the agentic harness with contextual retrieval, a more realistic and industry-standard approach for long documents. We also built a dataset of 1,000 IPO-diligence questions, and publicly release 70 questions on the SpaceX (SPCX) S-1 filing to support reproducibility, while the remainder are held private to guard against benchmark contamination. In addition, we introduce an evaluator-optimizer pipeline to automatically generate evaluation rubrics for the benchmark: candidate facts are extracted from model answers, consolidated into draft criteria, then automatically audited for omissions, hallucinations, mistiered items, and redundancy, with LLM feedback driving iterative repair, targeted enrichment, and deduplication. Human experts only review final rubrics before deployment. Results show that the best-performing evaluated model, Alibaba Qwen 3.7 Max, reaches 79.4% accuracy at 0.30 USD per query, and the most cost-efficient model on the resulting Pareto frontier, Xiaomi MiMo-2.5 Pro, reaches slightly lower accuracy (76.8%) at 0.05 USD per query. Both exceed the current Finance Agent v2 leaderboard ceiling-Google Gemini 3.5 Flash at 57.9% for 2.51 USD per querywhile undercutting even FABv2's cheapest entry (MiniMax M3: 48.3% at 0.32 USD) on cost-efficiency. Code and data are released on GitHub: this https URL

**arXiv ID:** 2606.23032
</details>

<details>
<summary><strong>Multimedia and Visual Analytics in the Agentic Era</strong> - Marcel Worring, Jan Zahálka, Stef van den Elzen, Maximilian T. Fischer, Daniel A. Keim - [[pdf]](https://arxiv.org/pdf/2504.06138)</summary>

**Abstract:** Professional users need tools to help them gain actionable insights from large multimedia collections. Foundation models and AI agents have rapidly changed the playing field, and improving their accuracy, trustworthiness, and reasoning capabilities are active topics in the computer vision, machine learning, and multimedia communities. Most current research focuses on benchmark driven algorithmic improvements. The multimedia community is the place to go beyond algorithms and consider complete multimedia analytics systems that support professional users in their complex tasks and achieve a true teaming of humans and AI. Supporting users with machine learning and visualizations has been studied for decades in the visual analytics field. In this paper, we propose a framework to bring multimedia and visual analytics together and indicate how it could impact current and new multimedia analytics solutions. Additional information can be found at this https URL

**arXiv ID:** 2504.06138
</details>

<details>
<summary><strong>Qwen-AgentWorld: Language World Models for General Agents</strong> - Yuxin Zuo, Zikai Xiao, Li Sheng, Fei Huang, Jianhong Tu, Yuxuan Liu, Tianyi Tang, Xiaomeng Hu, Yang Su, Qingfeng Lan, Yantao Liu, Qin Zhu, Yinger Zhang, Bowen Yu, Haiquan Zhao, Haiyang Xu, Jianxin Yang, Jiayang Cheng, Junyang Wang, Lianghao Deng, Mingfeng Xue, Tianyi Bai, Yang Fan, Yubo Ma, Yucheng Li, Zeyu Cui, Zhihai Wang, Zhihui Xie, Zhuorui Ye, An Yang, Dayiheng Liu, Jingren Zhou, Ning Ding - [[pdf]](https://arxiv.org/pdf/2606.24597)</summary>

**Abstract:** A world model predicts environment dynamics based on current observations and actions, serving as a core cognitive mechanism for reasoning and planning. In this work, we investigate how world modeling based on language models can further push the boundaries of general agents. (i) We first focus on building foundation models for agentic environment simulation. We introduce Qwen-AgentWorld-35B-A3B and Qwen-AgentWorld-397B-A17B, the first language world models capable of simulating agentic environments covering 7 domains via long chain-of-thought reasoning. Leveraging more than 10M environment interaction trajectories of 7 domains in real-world environments, we develop Qwen-AgentWorld through a three-stage training pipeline: CPT injects general-purpose world modeling capabilities from the state transition dynamics and augmented professional corpora, SFT activates next-state-prediction reasoning, and RL sharpens simulation fidelity through a tailored framework with hybrid rubric-and-rule rewards. To evaluate language world models, we present AgentWorldBench, a comprehensive benchmark constructed from real-world interactions of 5 frontier models on 9 established benchmarks. Empirical results demonstrate that Qwen-AgentWorld significantly outperforms existing frontier models. (ii) Beyond foundation models, we further investigate two complementary paradigms through which world modeling enhances general agents. First, as a decoupled environment simulator, Qwen-AgentWorld supports scalable and controllable simulation of thousands of real-world environments for agentic RL, yielding gains that surpass real-environment training alone. Second, as a unified agent foundation model, world-model training acts as a highly effective warm-up that improves downstream performance across 7 agentic benchmarks. Code: this https URL

**arXiv ID:** 2606.24597
</details>

<details>
<summary><strong>Are We Ready For An Agent-Native Memory System?</strong> - Wei Zhou, Xuanhe Zhou, Shaokun Han, Hongming Xu, Guoliang Li, Zhiyu Li, Feiyu Xiong, Fan Wu - [[pdf]](https://arxiv.org/pdf/2606.24775)</summary>

**Abstract:** Memory for large language model (LLM) agents has rapidly evolved from simple retrieval-augmented mechanisms into a data management system that supports persistent information storage, retrieval, update, consolidation, and dynamic lifecycle governance throughout agent execution. Despite this evolution, existing evaluations still benchmark agent memory mainly through end-to-end task success metrics (e.g., F1, BLEU), while treating the underlying system as a monolithic black box. As a result, critical system-level concerns, including operational costs, architectural trade-offs across memory modules, and robustness under dynamic knowledge updates, remain insufficiently explored. In this paper, we present a systematic experimental study of agent memory from a data management perspective. We propose an analytical framework that decomposes agent memory into four core modules: memory representation and storage, extraction, retrieval and routing, and maintenance. Under this framework, we evaluate 12 representative memory systems and two reference baselines across five benchmark workloads spanning 11 datasets. Our extensive end-to-end evaluation shows that no single architecture dominates across all scenarios; instead, effectiveness depends heavily on how well the memory structure aligns with the workload bottleneck. Furthermore, through fine-grained ablation studies, we quantify their individual effects on representation fidelity, retrieval precision, update correctness, and long-horizon stability. Finally, we reveal cost-performance trade-offs under realistic workloads, showing localized maintenance is more cost-efficient than global reorganization. Based on these findings, we identify promising directions towards building truly agent-native memory systems. The code is publicly available at this https URL.

**arXiv ID:** 2606.24775
</details>

<details>
<summary><strong>Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning</strong> - Chenhao Dang, Jing Ma, Mingjie Liao - [[pdf]](https://arxiv.org/pdf/2606.24133)</summary>

**Abstract:** The composition of training data, governed by the diversity of sources and their mixing strategy, is a cornerstone of Large Language Model (LLM) pre-training. Online Data Mixing (ODM), the technique of adaptively adjusting data mixtures during training, has emerged as a promising direction to improve efficiency. However, existing methods are constrained by their reliance on a singular optimization perspective, which fundamentally overlooks the need for complex LLM pre-training to consider the dynamic data composition from multiple dimensions. To overcome this limitation, we introduce the Holistic Data Scheduler (HDS), a novel online data mixing framework. HDS formulates the data scheduling challenge as a reinforcement learning problem in a continuous control space and leverages the Soft Actor-Critic (SAC) algorithm for its stability and sample efficiency in exploring the high-dimensional policy space. At the core of HDS lies a novel multi-objective, holistic reward function that integrates three critical perspectives: a data-driven reward for quality, a loss-driven reward capturing inter-domain influence, and a model-driven reward based on weight norms. To validate our design and determine its optimal configuration, we conducted systematic experiments on LLMs of various sizes. On The Pile benchmark, HDS reaches the final validation perplexity of the next best method with 44% fewer training iterations. Furthermore, it achieves a 7.2% improvement on the MMLU 0-shot task along with consistent gains on other benchmarks, showcasing its ability to enhance both training efficiency and final model capability.

**arXiv ID:** 2606.24133
</details>

<details>
<summary><strong>Reinforcement Learning to Disentangle Multiqubit Quantum States from Partial Observations</strong> - Pavel Tashev, Stefan Petrov, Matthew T. Diaz, Friederike Metz, Alaina M. Green, Norbert M. Linke, Marin Bukov - [[pdf]](https://arxiv.org/pdf/2406.07884)</summary>

**Abstract:** Using partial knowledge of a quantum state to control multiqubit entanglement is a largely unexplored paradigm in the emerging field of quantum interactive dynamics with the potential to address outstanding challenges in quantum state preparation and compression, quantum control, and quantum complexity. We present a deep reinforcement learning (RL) approach using an actor-critic algorithm for constructing short disentangling circuits for states with up to 16 qubits. With access to only two-qubit reduced density matrices, our agent decides which pairs of qubits to apply two-qubit gates on; requiring only local information makes it directly applicable on modern NISQ devices, as we demonstrated experimentally on a trapped-ion quantum computer. Utilizing a permutation-equivariant transformer architecture, the agent can autonomously identify qubit permutations within the state, and adjusts the disentangling protocol accordingly. Once trained, it provides circuits from different initial states without further optimization. We demonstrate the agent's ability to identify and exploit the entanglement structure of multi-qubit states. We analyze the disentangling circuits constructed by the agent for 4- and 5-qubit Haar-random states, and observe strong correlations between consecutive gates and among the qubits involved. Through extensive benchmarking, we show the efficacy of the RL approach to find disentangling protocols with minimal gate resources. We explore the resilience of our trained agents to noise, highlighting their potential for real-world quantum computing applications. Analyzing optimal disentangling protocols, we report a general circuit to prepare an arbitrary 4-qubit state using at most 5 two-qubit (10 CNOT) gates.

**arXiv ID:** 2406.07884
</details>

<details>
<summary><strong>Posterior Sampling Reinforcement Learning with Gaussian Processes for Continuous Control: Sublinear Regret Bounds for Unbounded State Spaces</strong> - Hamish Flynn, Joe Watson, Ingmar Posner, Jan Peters - [[pdf]](https://arxiv.org/pdf/2603.08287)</summary>

**Abstract:** We analyze the Bayesian regret of the Gaussian process posterior sampling reinforcement learning (GP-PSRL) algorithm. Posterior sampling is a heuristic for decision-making under uncertainty that has been used to develop successful algorithms for a variety of continuous control problems. However, theoretical work on GP-PSRL is limited. All known regret bounds either have a sub-optimal growth rate, require strong smoothness assumptions, or fail to properly account for the fact that the set of possible system states is unbounded. Through a recursive application of the Borell-Tsirelson-Ibragimov-Sudakov inequality, we show that, with high probability, the states actually visited by the algorithm are contained within a ball of near-constant radius. We then use the chaining method to control the regret suffered by GP-PSRL under weak smoothness conditions. Our main result is a Bayesian regret bound of the order $\widetilde{\mathcal{O}}(H\sqrt{\gamma_TT})$, where $H$ is the horizon, $T$ is the number of time steps and $\gamma_T$ is the expected information gain. With this result, we resolve the limitations with prior theoretical work on PSRL, and provide the theoretical foundation and tools for analyzing PSRL in complex settings.

**arXiv ID:** 2603.08287
</details>

<details>
<summary><strong>Precision Physical Activity Prescription via Reinforcement Learning for Functional Actions</strong> - Gefei Lin, Rui Miao, Jennifer Sacheck, Xiaoke Zhang - [[pdf]](https://arxiv.org/pdf/2605.19208)</summary>

**Abstract:** Physical activity (PA) plays an important role in maintaining and improving health. Daily steps have been a key PA measure that is easily accessible with common wearable devices. However, methods are lacking to recommend a personalized optimal distribution of daily steps over a period of time for the best of certain health biomarkers. In this paper, we fill this void based on the data from the All of Us Research Program which includes months of step counts as well as repeated measurements of key health biomarkers. We develop a new offline reinforcement learning (RL) algorithm to learn personalized and optimal PA distributions associated with cardiometabolic risk, where the action is a function representing the daily step distribution over a period of time. Simulation studies demonstrate the advantage of the proposed approach over existing continuous-action RL methods. The learned optimal policy from the All of Us data generally suggests people take more daily steps and also follow a more consistent pattern of PA over time while offering tailored recommendations for subgroups in blood glucose level, body mass index, blood pressure, age, and sex.

**arXiv ID:** 2605.19208
</details>

<details>
<summary><strong>RoBoSR: Structured Scene Representations for Embodied Robotic Reasoning</strong> - Kewei Hu, Wanchan Yu, Fangwen Chen, Jing Jiajian, Zimeng Li, Ying Wei, Tianhao Liu, Michael Zhang, Hanwen Kang - [[pdf]](https://arxiv.org/pdf/2606.24338)</summary>

**Abstract:** Despite rapid progress, embodied reasoning under real-world variability remains challenging. Existing approaches rely on demonstration-driven sequential biases, limiting flexibility in open-ended and long-horizon tasks that require structured reasoning over evolving states.
We introduce RoBoSR, an intermediate structural representation that formulates manipulation as step-wise state transitions over semantically grounded, object-centric scene graphs. By modeling object states and their spatial relations at the perception-action interface, RoBoSR disentangles high-level task reasoning from raw inputs and enables structured reasoning over preconditions, effects, and goal states. This representation endows the agent with causal reasoning capability, enforcing subtask dependencies and supporting coherent long-horizon task planning.
To learn such structure-aware reasoning, we construct Manip-Cognition-1.6M, an open-world dataset that jointly supervises scene understanding, instruction interpretation, and subtask planning across diverse tasks.
Across several benchmarks and real-world demonstrations, our method consistently outperforms prompting-based methods and classical TAMP baselines in zero-shot generalization and long-horizon tasks. The results underscore structured intermediate representations as a critical inductive bias for scalable embodied reasoning.

**arXiv ID:** 2606.24338
</details>

<details>
<summary><strong>MyoInteract: A Framework for Fast Prototyping of Biomechanical HCI Tasks using Reinforcement Learning</strong> - Ankit Bhattarai, Hannah Selder, Florian Fischer, Arthur Fleig, Per Ola Kristensson - [[pdf]](https://arxiv.org/pdf/2602.15245)</summary>

**Abstract:** Reinforcement learning (RL)-based biomechanical simulations have the potential to revolutionise HCI research and interaction design, but currently lack usability and interpretability. Using the Human Action Cycle as a design lens, we identify key limitations of biomechanical RL frameworks and develop MyoInteract, a novel framework for fast prototyping of biomechanical HCI tasks. MyoInteract allows designers to setup tasks, user models, and training parameters from an easy-to-use GUI within minutes. It trains and evaluates muscle-actuated simulated users within minutes, reducing training times by up to 98%. A workshop study with 12 interaction designers revealed that MyoInteract allowed novices in biomechanical RL to successfully setup, train, and assess goal-directed user movements within a single session. By transforming biomechanical RL from a days-long expert task into an accessible hour-long workflow, this work significantly lowers barriers to entry and accelerates iteration cycles in HCI biomechanics research.

**arXiv ID:** 2602.15245
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
