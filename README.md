# Agent arXiv Daily

**Last Updated:** 2026-06-25 04:21:13

**Total Papers:** 62

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
<summary><strong>One Body, Two Minds: Variable Autonomy Approach for a Co-embodied Robotic Hand</strong> - Piotr Koczy, Yuchong Zhang, Danica Kragic, Michael C. Welle - [[pdf]](https://arxiv.org/pdf/2606.25575)</summary>

**Abstract:** Assistive robotic systems face a fundamental trade-off: fully autonomous systems lack user agency, while fully user-controlled systems demand continuous cognitive effort. Existing shared autonomy approaches blend human and robot commands but are mostly deployed in separate physical bodies. We introduce co-embodiment with variable autonomy, where human and robot share a single physical body and operate at different autonomy levels across task phases, from mutual autonomy during object search and grasping to human-dominant control during actuation.
We present a co-embodied, wearable robotic hand that has its own ``mind'' and operates with variable autonomy levels. A learning-from-demonstration visuomotor diffusion policy enables autonomous grasping when the user positions the hand near known objects. Once grasped, the system signals completion and the human can actuate the grasped tool (drill, spray bottle, infrared thermometer, lighter, and ice-cream scoop) via hands-free head gestures. The human retains veto authority at all times through a release gesture that returns the system to the initial phase. Unlike blended autonomy, where control is continuously negotiated, our co-embodied approach consists of variable autonomy from full human control to full independent actions while maintaining physical coupling, realizing a one body, two minds paradigm.
In a user study with 44 participants performing five bimanual tasks, users rapidly adapted to this ``two minds'' paradigm: completion times improved by 23.3% across trials ($p < 0.001$, Cohen's $d = 0.94$), the best-performing policy variant reached a 93.6% task success rate, and acceptance ratings were high (5.70/7 overall impression, 5.52/7 daily use willingness). This work establishes co-embodiment with variable autonomy as a viable approach for assistive robotics, enabling human-robot collaboration through co-embodiment.

**arXiv ID:** 2606.25575
</details>

<details>
<summary><strong>Engineering Reliable Autonomous Systems: Challenges and Solutions</strong> - Marie Farrell, Matt Luckcuck, Angelo Ferrando, Rafael C. Cardoso, Natasha Alechina, Marco Autili, Diana Benjumea Hernandez, Luciana Brasil Rebelo dos Santos, Daniela Briola, Ana Cavalcanti, Christian Colombo, Louise A. Dennis, Clare Dixon, Michael Fisher, Mario Gleirscher, Taylor Johnson, Charles Lesire, Livia Lestingi, Sven Linker, Brian Logan, Colin Paterson, Fabio Papacchini, Patrizio Pelliccione, Pedro Ribeiro, Maike Schwammberger, Silvia Lizeth Tapia Tarifa, Hazel Taylor, Jim Woodcock, Mengwei Xu, Yi Yang, Huan Zhang - [[pdf]](https://arxiv.org/pdf/2606.23760)</summary>

**Abstract:** Engineering reliable autonomous systems is an important and growing topic in computer science. As autonomous systems become more prevalent, easy-to-use techniques for building them reliably are increasingly important.
This workshop report captures and expands on the discussions at the Lorentz Center Workshop "Engineering Reliable Autonomous Systems" (ERAS), held from 10 to 14 June 2024. The workshop was co-organised by the organisers of the Workshop on Formal Methods for Autonomous Systems (FMAS) and the Workshop on Agents and Robots for reliable Engineered Autonomy (AREA). It brought together members of the FMAS and AREA communities, industry practitioners, and representatives from sectors where autonomous systems pose distinctive engineering challenges.
The workshop focused on three main research topics: techniques for verification and validation of autonomous systems; engineering real-world autonomous systems; and software architectures for safe autonomous systems. Its main outcome is a catalogue of challenges in these areas and, most importantly, a pathway to solutions. Some challenges can already be tackled by techniques that are well known in academia but have not yet become regularly used in practice. Other challenges remain unresolved and require further research. This roadmap is intended to support future research and industrial collaboration.

**arXiv ID:** 2606.23760
</details>

<details>
<summary><strong>A Low-Code Approach for the Automatic Personalization of Conversational Agents</strong> - Aaron Conrardy, Alfredo Capozucca, Jordi Cabot - [[pdf]](https://arxiv.org/pdf/2605.02384)</summary>

**Abstract:** The rise of Large Language Models (LLMs) has increased the demand for Conversational Agents (CAs) capable of understanding human conversations as part of web applications. While traditional CAs consist of deterministic states, LLMs enhance their capabilities to handle open conversations, handling arbitrary requests. Numerous tools exist that allow non-technical users to create such CAs. Yet, the creation of personalized CAs able to adapt to the profile of end-users to offer an optimal user experience remains in the hands of experienced developers implementing ad-hoc personalizations. In this work, we propose a pipeline that follows a low-code/no-code approach to facilitate the modeling and generation of personalized CAs. A pilot user study was performed to get preliminary results on perceived usability and usefulness and the full pipeline has been implemented on top of an open-source low-code platform.

**arXiv ID:** 2605.02384
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents</strong> - Zheyuan Zhang, Zehao Wen, Alvin Zhang, Andrew Wang, Jianwen Xie, Daniel Khashabi, Tianmin Shu - [[pdf]](https://arxiv.org/pdf/2606.24893)</summary>

**Abstract:** For agents to learn continuously from interaction with the world at test time, they must be able to explore effectively, acquire new world knowledge and skills, retain relevant episodic experiences, and plan over long horizons. To evaluate these key abilities of test-time continual learning agents, we introduce AgentOdyssey, a novel evaluation framework that procedurally generates open-ended text games with rich entities, world dynamics, and long-horizon tasks. Critically, AgentOdyssey goes beyond the conventional machine learning assumption that learning does not occur at test time by placing agents in a continuous, long-horizon setting that interleaves learning and inference throughout deployment. We further propose a multifaceted evaluation methodology that measures not only game progress but also offers diagnostic tests on world knowledge acquisition, episodic memory, object and action exploration, action diversity, and model cost. We evaluate diverse agent paradigms in the generated games. Our experimental results reveal critical limits in agents' key abilities, as well as factors that influence their meaningful horizon. Although performance scales with stronger base models, even the top agent remains far below human performance, leaving substantial headroom for improvement. Among agent mechanisms, we find that short-term memory benefits multiple agent paradigms and is an important component of agent test-time training.

**arXiv ID:** 2606.24893
</details>

<details>
<summary><strong>Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory</strong> - Beining Wu, Zihao Ding, Jun Huang, Yanxiao Zhao - [[pdf]](https://arxiv.org/pdf/2606.25115)</summary>

**Abstract:** On-device language-model agents improve by accumulating experience in retrieved memory rather than by updating weights. This memory is hard-bounded and exposed: it consumes RAM and energy, reaches peers through a thin uplink, and becomes an attack surface because it is writable by what the agent reads. Existing systems each cover one part of this problem: agentic memories grow without a budget, on-device methods keep entries by success alone, and poisoning is studied mainly as an attack rather than as a memory-governance problem. We propose \sys{}, a single net-value-per-byte score that governs an agent's experience-memory lifecycle. The main idea is to let the budget act as the curator: each entry is scored as value minus harm, per byte, so one ruler decides what to keep, share, and trust. \sys{} makes three decisions: (1) \textbf{KEEP} evicts low-value bytes under the RAM and energy budget; (2) \textbf{SHARE} sends an insight only when its value exceeds its uplink cost; and (3) \textbf{TRUST} gates a peer entry by provenance. On language-model-agent task-drift benchmarks and a real heterogeneous Jetson testbed with two robot-arm nodes and a hub, \sys{} reduces memory by $2.7\times$ and uplink by $2.4\times$, drives injection success from 0.75 to zero, and raises accuracy on cases corrupted by poison or stale memory. Curating by net value reduces footprint, energy, uplink, and injection success together without reducing accuracy. In this setting, forgetting by net value improves the agent rather than weakening it.

**arXiv ID:** 2606.25115
</details>

<details>
<summary><strong>ASAP: Agent-System Co-Design for Wall-Clock-Centered Auto HPO Research for ML Experiments</strong> - Taicheng Guo, Haomin Zhuang, Kehan Guo, Yujun Zhou, Nitesh V. Chawla, Olaf Wiest, Xiangliang Zhang - [[pdf]](https://arxiv.org/pdf/2606.25207)</summary>

**Abstract:** Hyperparameter Optimization (HPO) is essential for maximizing machine learning model performance, and its core challenge is sample efficiency: finding strong configurations within a limited budget. Because every HPO tool relies on a surrogate prior that imparts its own inductive bias, individual tools struggle once problems become sufficiently diverse and drift from these priors. Motivated by the reasoning and generalization capabilities of LLMs, recent work has explored using LLMs for HPO and reports improved per-iteration performance. Yet these methods share two limitations with a common origin: they use the LLM as a single-tool replacement evaluated by iteration count. (i) Deployed in place of prior tools, the LLM is itself constrained by its pretraining objective to one family of inductive-biased proposals; this single-source setup still fails to handle the full diversity of problems. (ii) Per-iteration evaluation ignores that, in real runs, LLM inference or tool execution is paid serially on top of model evaluation every round, so iteration-count gains do not translate into end-to-end wall-clock gains. We present ASAP, an agent-system co-design that addresses both limitations. On the agent side, ASAP uses the LLM to integrate a diverse pool of inductive-biased optimizers and to select among their proposals each round. On the system side, ASAP re-architects the loop to reduce end-to-end wall-clock while preserving regret quality: a prefix-stable prompt maximizes KV-cache reuse across rounds; speculation parallelism hides the remaining LLM and tool latency under model evaluation via a relative-error accept test; and a Self-Tuner adapts the speculation threshold from execution logs off the critical path. Extensive experiments on diverse modern HPO tasks show that ASAP consistently outperforms baselines, underscoring the value of tool integration and agent-system co-design.

**arXiv ID:** 2606.25207
</details>

<details>
<summary><strong>Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets</strong> - Divake Kumar, Sina Tayebati, Devashri Naik, Amanda Sofie Rios, Nilesh Ahuja, Omesh Tickoo, Ranganath Krishnan, Amit Ranjan Trivedi - [[pdf]](https://arxiv.org/pdf/2606.25760)</summary>

**Abstract:** Computer-use agents turn vision-language model (VLM) predictions into executable GUI clicks, so reliable uncertainty estimates are essential for rejection, calibration, miss-severity ranking, and spatial safety regions. Yet evidence on post-hoc uncertainty quantification (UQ) for these agents is fragmented across isolated model and dataset pairs, leaving it unclear whether UQ rankings stay stable when the agent, benchmark, or observable interface changes. We present Argus, a cross-regime benchmark for post-hoc UQ in single-step executable GUI grounding: a 27-method open-weight matrix over 4 VLM agents and 4 datasets, plus an 8-method closed-source matrix across 3 frontier vendors where logits, hidden states, and attention maps are unavailable. Evaluated methods span logit-based scores, sampling and consistency measures, hidden-state and density estimators (Mahalanobis, SAPLMA), attention-based scores, P(True) and verbalised-confidence prompting, and split-conformal prediction. The main finding is selective transfer: UQ rankings are stable across datasets for a fixed model, but degrade across model classes and observable interfaces. Hidden-state and density methods are the most stable open-weight family, while CoCoA-1MCA, Focus, sampling-based scores, and verbalised self-assessment win in specific regimes. Within-model ranking transfer is strong (Spearman rho up to 0.969), but cross-tier transfer to closed-source vendors averages only +0.08, so closed-source UQ should be reranked on the target rather than extrapolated. Conformal click regions show score-level discrimination is not enough for deployment: locally weighted disks shrink radii by 40-60% when the plug-in UQ is calibrated, but coverage degrades under calibration-test or interface mismatch. We release per-item records, calibration/test splits, UQ scores, and analysis scripts for regime-aware UQ selection in GUI agents.

**arXiv ID:** 2606.25760
</details>

<details>
<summary><strong>Autodata: An agentic data scientist to create high quality synthetic data</strong> - Ilia Kulikov, Chenxi Whitehouse, Tianhao Wu, Yixin Nie, Swarnadeep Saha, Eryk Helenowski, Weizhe Yuan, Olga Golovneva, Jack Lanchantin, Yoram Bachrach, Jakob Foerster, Xian Li, Han Fang, Sainbayar Sukhbaatar, Jason Weston - [[pdf]](https://arxiv.org/pdf/2606.25996)</summary>

**Abstract:** We introduce Autodata, a general method that enables AI agents to act as data scientists who build high quality training and evaluation data. We show how to train (meta-optimize) such a data scientist agent, so that it learns to create even stronger data. We describe the overall formulation, and a specific practical implementation, Agentic Self-Instruct. We conduct experiments on computer science research tasks, legal reasoning tasks and reasoning with mathematical objects, where we obtain improved results compared to classical synthetic dataset creation methods. Further, meta-optimizing the data scientist agent itself delivers an even larger performance uplift. Agentic data creation provides a way to convert increased inference compute into higher quality model training. Overall, we believe this direction has the potential to change the way we build AI data.

**arXiv ID:** 2606.25996
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
<summary><h2>LLM Agents (7 papers)</h2></summary>

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
<summary><strong>The Interplay of Harness Design and Post-Training in LLM Agents</strong> - Kyungmin Kim, Youngbin Choi, Seoyeon Lee, Suhyeon Jun, Dongwoo Kim, Sangdon Park - [[pdf]](https://arxiv.org/pdf/2606.25447)</summary>

**Abstract:** Tool-integrated LLM agents are often wrapped within a harness: the scaffolding that determines which tools are exposed, how they are described, and what auxiliary information accompanies each per-step observation. While agents are routinely post-trained, this scaffolding is typically treated as a fixed engineering detail, with design effort limited to the training-free regime. Moreover, existing post-training algorithms assume a static environment, even though tool environments and tasks often shift upon deployment. To address this gap, we extend $\texttt{ALFWorld}$ (i) to treat the harness as a controllable design dimension and (ii) to support evaluation under task and tool environment shifts. Building on this, we systematically analyze how the harness design influences post-training in both in-distribution and out-of-distribution (OOD) settings. We empirically show that harness-aware post-training not only improves in-distribution performance but also enables agents to robustly adapt to OOD settings. Under a harness with minimal design effort, post-training suffers a drastic performance drop under stronger tool environment shifts, further highlighting the importance of harness-aware post-training under such shifts.

**arXiv ID:** 2606.25447
</details>

<details>
<summary><strong>Semantic Consistency Policy Optimization for Reinforcement Learning of LLM Agents</strong> - Peng Xu, Sijia Chen, Junzhuo Li, Xuming Hu - [[pdf]](https://arxiv.org/pdf/2606.25852)</summary>

**Abstract:** Group-based reinforcement learning effectively post-trains LLM agents for long-horizon, sparse-reward tasks by deriving step-level credit from trajectory outcomes. However, this ties a step's credit to its rollout's final outcome: semantically near-identical intermediate steps receive opposite credit depending on whether their trajectory eventually succeeded or failed. Such semantic credit inconsistency sends conflicting gradients to similar actions and wastes the partially-correct progress inside failed rollouts. Motivated by this, we propose Semantic Consistency Policy Optimization (SCPO), a value-free reward-shaping method that mitigates this inconsistency by recovering step-level credit from successful siblings in the same rollout group. Concretely, SCPO scores each failed step against a successful sibling and adds positive step-level credit for new progress along that sibling. On ALFWorld and WebShop, SCPO matches or exceeds strong group-based baselines, reaching 93.7+/-4.1 percent success on ALFWorld and 74.8+/-2.0 percent on WebShop at 1.5B parameters, with gains concentrated on the hardest multi-step tasks.

**arXiv ID:** 2606.25852
</details>

<details>
<summary><strong>Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents</strong> - Changdae Oh, Wendi Li, Seongheon Park, Samuel Yeh, Tanwi Mallick, Sharon Li - [[pdf]](https://arxiv.org/pdf/2606.26080)</summary>

**Abstract:** Process reward models enable fine-grained, step-level evaluation of LLMs, yet building them for agentic settings remains prohibitively difficult: long-horizon interactions, irreversible actions, and stochastic environment feedback make both human annotation and Monte Carlo estimation infeasible at scale. In this work, we show that reinforcement learning (RL) post-training already provides the ingredients for effective step-level scoring, eliminating the need for dedicated reward model training altogether. Concretely, we derive an implicit advantage under a general stochastic Markov decision process, which we term progress advantage -- log-probability ratio between the RL-trained policy and its reference policy exactly recovers the optimal advantage function. This formulation makes the resulting signal annotation-free, domain-agnostic, and available as a byproduct of the standard RL post-training pipeline. We validate the effectiveness of the progress advantage across three different applications: test-time scaling, uncertainty quantification, and failure attribution on five benchmarks and four model families. Across all settings, it consistently outperforms confidence-based baselines and, despite requiring no task-specific training, surpasses dedicated trained reward models. We complement these results with deeper analyses on characteristics of progress advantage, offering practical guidance for adoption in real-world agentic systems.

**arXiv ID:** 2606.26080
</details>

<details>
<summary><strong>BiPACE: Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents</strong> - Hanyang Wang, Weijieying Ren, Yuxiang Zhang, Ding Cao, Zhizhao Zeng, Ke Zeng, Tianxiang Zhao - [[pdf]](https://arxiv.org/pdf/2606.25556)</summary>

**Abstract:** Stepwise group-based RL is an attractive way to train long-horizon LLM agents without a learned critic: it reuses multiple sampled rollouts to estimate local advantages. Its weakness is less visible but more fundamental: every group-relative estimator assumes that the steps it compares are equivalent for credit assignment. We show that current agentic variants violate this assumption through a state-action credit mismatch. The observation-hash partition is overly fine on the state side, creating singleton groups with zero step-level signal, while a single within-group mean is too coarse on the action side, mixing state-value estimation with action-specific credit. We introduce BiPACE (Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation), a drop-in advantage estimator that fixes both sides without adding a critic, auxiliary loss, or extra rollouts. BiGPO clusters steps by cosine distance in the actor's own hidden-state geometry, an empirical policy-induced proxy for bisimulation that substantially lowers the singleton rate left by observation hashing. PACE then recenters returns within each behavioral cluster using action-conditioned peer baselines; its Q-style instance estimates a local Q(s,a)-V(s) nonparametrically. On ALFWorld/Qwen2.5-7B, BiPACE_Q raises overall validation success from GiGPO's 90.8 to $97.1\pm0.9$ over three seeds, and crosses the 95% threshold on every seed, which GiGPO never does within the same budget. On Qwen2.5-1.5B it reaches $93.5\pm1.2$ versus GiGPO's 86.7, and on WebShop and TextCraft it improves over GRPO and GiGPO at both model scales. The measured BiPACE-specific overhead is 11.3% of a single training-step wall time. Yet it changes the estimator's comparison unit from surface identity to approximate behavioral equivalence plus action-side counterfactuals. The code is available at this https URL.

**arXiv ID:** 2606.25556
</details>

<details>
<summary><strong>Explainable Control Framework (XCF) based on Fuzzy Model-Agnostic Explanation and LLM Agent-Supported Interface</strong> - Faliang Yin, Hak-Keung Lam, David Watson - [[pdf]](https://arxiv.org/pdf/2606.25941)</summary>

**Abstract:** Increasing demand for precise and reliable control in complex scenarios has led to the development of increasingly sophisticated controllers, including data-driven approaches employing closed box models and mathematically rigorous yet complex designs. This complexity highlights the needs for explainable control that can provide human-understandable insights into controller behavior. In this paper, an explainable control framework (XCF) along with supporting algorithms and user interface are proposed to explain how controllers determine their control actions and their underlying working mechanism. The novel contributions of this work are threefold: First, the XCF is designed to provide model-agnostic explanations for controllers in closed-loop systems and can optionally refine local explanations by system response dynamics. Second, a novel explanation method, hierarchical fuzzy model-agnostic explanation for control systems (HFMAE-C), is proposed based on the designed framework. The HFMAE-C employs a fuzzy logic system to approximate the controller's behavior and system dynamics, providing sample, local, domain and universe level explanations via IF-THEN rules revealing the controller's decision logic and salience values quantifying the contribution of system states to control actions. Third, a large language model agent-supported user interface is developed to automatically analyze user requirements, select appropriate algorithms, interpret the generated explanations to a natural language report, and provide interactive consultation. Case studies on inverted pendulum system and Turtlebot obstacle avoidance demonstrate the effectiveness of the proposed method through simulated user experiments and quantitative comparisons with mainstream explainable control approaches.

**arXiv ID:** 2606.25941
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (13 papers)</h2></summary>

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
<summary><strong>Stagnant Neuron: Towards Understanding the Plasticity Loss in Multi-Agent Reinforcement Learning Value Factorization Methods</strong> - Zhengzhu Liu, Zeming Gao, Haoyuan Qin, Jiawei Hu, Junhao Wu, Miao Zhu, Haipeng Zhang, Chennan Ma, Siqi Shen, Cheng Wang - [[pdf]](https://arxiv.org/pdf/2606.25335)</summary>

**Abstract:** Multi-Agent Reinforcement Learning (MARL) value factorization methods can suffer from a loss of plasticity, gradually failing to adapt when transferring to new task instances. We trace this issue to stagnant neurons, units whose gradient updates become negligibly small relative to their weights, thereby hindering learning. While existing plasticity injection methods exist, they prove ineffective for such neurons. To address this, we propose Knowledge-retentive Neuron-level PlastIcity Focusing InjEction (KNIFE), a novel method that directly targets stagnant neurons. KNIFE replaces each stagnant neuron with a composite unit comprising three specialized components: a frozen knowledge neuron to preserve acquired knowledge, a re-initialized active neuron to restore learning capacity, and a compensation neuron to ensure the combined output matches the original, thus maintaining previous learned cooperation knowledge. Extensive experiments on SMACv2, predator-prey, and matrix games demonstrate that KNIFE significantly outperforms state-of-the-art plasticity injection methods.

**arXiv ID:** 2606.25335
</details>

<details>
<summary><strong>Low Variance Trust Region Optimization with Independent Actors and Sequential Updates in Cooperative Multi-agent Reinforcement Learning</strong> - Bang Giang Le, Viet Cuong Ta - [[pdf]](https://arxiv.org/pdf/2606.25526)</summary>

**Abstract:** Cooperative multi-agent reinforcement learning assumes each agent shares the same reward function and can be trained effectively using the Trust Region framework of single-agent. Instead of relying on other agents' actions, the independent actors setting considers each agent to act based only on its local information, thus having more flexible applications. However, in the sequential update framework, it is required to re-estimate the joint advantage function after each individual agent's policy step. Despite the practical success of importance sampling, the updated advantage function suffers from exponentially high variance problems, which likely result in unstable convergence. In this work, we first analyze the high variance advantage both empirically and theoretically. To overcome this limitation, we introduce a clipping objective to control the upper bounds of the advantage fluctuation in sequential updates. With the proposed objective, we provide a monotonic bound with sub-linear convergence to $\epsilon$-Nash Equilibria. We further derive two new practical algorithms using our clipping objective. The experiment results on three popular multi-agent reinforcement learning benchmarks show that our proposed method outperforms the tested baselines in most environments. By carefully analyzing different training settings, our proposed method is highlighted with both stable convergence properties and the desired low advantage variance estimation. For reproducibility purposes, our source code is publicly available at this https URL.

**arXiv ID:** 2606.25526
</details>

<details>
<summary><strong>Multi-Agent Goal Recognition with Team- and Goal-Conditioned Reinforcement Learning and Factorized Branch-and-Bound</strong> - Thiago Thomas, Gabriel de Oliveira Ramos, Felipe Meneguzzi - [[pdf]](https://arxiv.org/pdf/2606.25978)</summary>

**Abstract:** Multi-agent goal recognition asks an observer to jointly infer which agents act together and what each team is trying to achieve, so the hypothesis space grows combinatorially with the number of team partitions and goals per team. Real applications such as drone surveillance and collaborative robotics expose only the agents' trajectory, which forces the observer to rank team-goal hypotheses from behavior alone. Multi-Agent Goal Recognition with Branch-and-Bound (MAGR-BB) addresses this setting with a shared team- and goal-conditioned policy used as the scoring model inside a factorized branch-and-bound search. On a controlled multi-agent Blocksworld benchmark, MAGR-BB returns the same top-ranked hypothesis as exhaustive search throughout the trajectory while cutting hypothesis materialization by orders of magnitude and reducing cumulative recognition runtime substantially.

**arXiv ID:** 2606.25978
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
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty</strong> - Antonis Antoniades, Deepak Nathani, Ritam Saha, Alfonso Amayuelas, Ivan Bercovich, Zhaotian Weng, Vignesh Baskaran, Kunal Bhatia, William Yang Wang - [[pdf]](https://arxiv.org/pdf/2606.25198)</summary>

**Abstract:** Autonomous AI Research promises to accelerate the scientific progress of machine learning. To realise this goal, current Large Language Model (LLM)-based agents need to go beyond just writing code, to mastering the exploration of simultaneously performant, diverse and novel ideas. To this end, we introduce Heuresis, a framework that abstracts the research pipeline into a set of general and composable primitives, enabling open-ended scientific exploration in machine learning research. We implement six search strategies: a greedy baseline, two archive-based (MAP-Elites, Go-Explore), one evolutionary (Islands), and two divergent (Curiosity, Omni), and evaluate them across three axes (Quality, Diversity, and Novelty) on three domains (LLM Pretraining, On-Policy RL, and Model Unlearning), totalling 3,222 scored runs. We find that completely novel ideas are rare. No idea across our scored runs is rated as "Original", and only a few achieve only "Minor Similarity" to prior work. Moreover, novel ideas never approach the highest-performing known-recipe scores. Across all six strategies and three domains, only one such idea lands in the top-10 by quality. We also observed agents resorting to a variety of reward-hacking techniques during execution (40 confirmed fabrications across 1,628 scored runs), and detecting them was necessary to keep the search faithful to the task. Our results show that while current search and Quality-Diversity strategies enable us to steer where the generated ideas land on the quality, diversity, and novelty axes, they do not expand the quality-novelty frontier. Bridging this gap is the open challenge towards the ultimate goal of perpetual, autonomous scientific progress. Code is available at this http URL.

**arXiv ID:** 2606.25198
</details>

<details>
<summary><strong>TL++: Accuracy and Privacy Preserving Traversal Learning for Distributed Intelligent Systems</strong> - Erdenebileg Batbaatar, Young Yoon - [[pdf]](https://arxiv.org/pdf/2606.25627)</summary>

**Abstract:** Distributed intelligent systems increasingly need to train across data silos without centralizing raw data. Federated learning keeps data local but can suffer under heterogeneous partitions and requires repeated full-model exchange. Split learning reduces communication through cut-layer activations, but standard protocols generally do not recover centralized mini-batch gradient behavior and may expose activations and gradients in plaintext. We present TL++, a two-mode traversal-learning framework that constructs virtual batches across nodes to recover centralized mini-batch gradient behavior under explicit synchronization assumptions. Base mode exchanges cut-layer activations and gradients rather than full models. Secure mode secret-shares each cut-layer activation and gradient between an orchestrator and a non-colluding helper, preventing either server from observing plaintext cut-layer tensors. This protection is limited to a semi-honest two-server setting; labels and loss-related outputs remain visible to the orchestrator. In the lightweight secure path evaluated here, exactness requires a linear or affine server path, while nonlinear operations require nonlinear MPC or approximation. We formalize TL++, analyze communication and computation costs, and evaluate it against federated and split-learning baselines on CIFAR-10 and BioGPT/PubMedQA using full fine-tuning and LoRA. On CIFAR-10, TL++ base cut 1 and exact secure cut 3 achieve accuracies of 91.41% (SD 0.19) and 90.93% (SD 0.17), respectively, exceeding the strongest measured non-TL++ baseline by more than 12 percentage points. TL++ base cut 1 also reduces per-step communication by 13.1-fold relative to full-model synchronization. PubMedQA results similarly favor TL++. Overall, TL++ approaches centralized-training performance while reducing communication and providing activation-level secret sharing.

**arXiv ID:** 2606.25627
</details>

<details>
<summary><strong>AeroCast: Probabilistic 3D Trajectory Prediction for Non-Cooperative Aerial Obstacles via Transformer-MDN Architecture</strong> - Syed Izzat Ullah, Jose Baca - [[pdf]](https://arxiv.org/pdf/2606.25122)</summary>

**Abstract:** Autonomous aerial vehicles operating in shared airspace must predict the future positions of non-cooperative obstacles to plan evasive maneuvers before a collision becomes unavoidable. Unlike cooperative systems that share intent, non-cooperative obstacles such as birds, uncontrolled drones, or debris exhibit multi-modal motion that deterministic predictors cannot adequately represent. Existing methods either rely on recurrent encoders that propagate temporal information sequentially, limiting their ability to capture long-range kinematic precursors of maneuver initiation, or produce point forecasts that provide no distributional information to downstream planners. This paper presents AeroCast, a probabilistic trajectory prediction framework that combines a Transformer encoder with a Mixture Density Network output head to predict per-timestep Gaussian mixture distributions over future three-dimensional displacements. A translation-invariant consecutive displacement encoding and a calibration-oriented training objective address the input design and mode-degeneracy challenges specific to mixture-based aerial trajectory prediction. On a hybrid real-and-synthetic quadrotor corpus spanning nine motion categories, AeroCast reduces Average Displacement Error and Final Displacement Error by approximately 50% relative to the baselines over a five-second horizon, and achieves the lowest negative log-likelihood and Continuous Ranked Probability Score among all compared methods. Ablation analysis identifies velocity input and model capacity as the primary contributors to prediction quality, and positional encoding as essential for long-horizon trajectory coherence. AeroCast inference completes in 0.1ms per sample, compatible with real-time onboard deployment at 100Hz.

**arXiv ID:** 2606.25122
</details>

<details>
<summary><strong>The Unfireable Safety Kernel: Execution-Time AI Alignment for AI Agents and Other Escapable AI Systems</strong> - Seth Dobrin, Łukasz Chmiel - [[pdf]](https://arxiv.org/pdf/2606.26057)</summary>

**Abstract:** AI agents are granted access to tools, APIs, and other infrastructure, making them active principals in those systems. The dominant approach places controls inside the agent's own runtime: system prompts, output filters, and guardrail libraries. Any control in the agent's address space is reachable by inputs that influence it; this generalizes to any AI system with sufficient reach into its own runtime, a class we term escapable AI systems.
We identify four properties that an authorization mechanism must satisfy for architectural control rather than for cooperative requests: process separation, pre-action enforcement on a structurally only path, fail-closed at both the request and system levels, and externalized signed evidence verifiable outside the controlled system's trust boundary. We position this layer as execution-time AI alignment, complementing training-time alignment (RLHF, Constitutional AI) and inference-time alignment.
We present the Unfireable Safety Kernel, a Rust reference implementation realizing all four. Its fail-closed invariant is machine-checked at two levels: an SMT theorem (Z3) and an exhaustive bounded-model-checking proof of the production decision function (Kani, 4/4 harnesses). A Python-to-Rust migration was gated on byte-equivalence (1000/1000 fixtures; 17/17 adversarial classes). We evaluate the kernel governing a live, escapable AI system, a deterministic, self-improving world model, against an escape-seeking adversary driving its real self-modification seam: across 1,000 self-modifications, all 704 attempts on the safety-critical core are refused, with no escape; a further 300, under the operator kill switch, are also refused. A separate campaign of 6,240 authorization round-trips had no successful bypass. Against 3 contemporary systems claiming the agent control plane, the agent invokes control; here, it lacks that choice.

**arXiv ID:** 2606.26057
</details>

<details>
<summary><strong>Emcar: Embodied Controller for Animating Robots</strong> - Carlos Gomez Cubero, Elizabeth Jochum - [[pdf]](https://arxiv.org/pdf/2606.26008)</summary>

**Abstract:** This chapter describes EMCAR, a novel software tool for programming robot motion that leverages the unique affordances of artistic practices such as puppetry and drawing to conceive, design, and program novel interactions and realize new use cases for HRI. The advantage of this no-code platform is that it expands creative applications for collaborative robots - putting robots directly in the hands of artists - and provides an inclusive environment that enables individuals with little or no technical backgrounds to engage meaningfully in collaborations and robotics research.

**arXiv ID:** 2606.26008
</details>

<details>
<summary><strong>Tinker Tales: A Tangible Dialogue System for Child-AI Co-Creative Storytelling</strong> - Nayoung Choi, Jiseung Hong, Peace Cyebukayire, Ikseon Choi, Jinho D. Choi - [[pdf]](https://arxiv.org/pdf/2602.04109)</summary>

**Abstract:** Conversational AI agents are increasingly explored as creative partners, yet how conversation design shapes child-AI dialogue in co-creative settings remains underexplored. We present Tinker Tales, a tangible dialogue system for child-AI collaborative storytelling, in which educational frameworks (narrative development and social-emotional learning) are instantiated as conversation design, shaping how the agent engages children across four narrative stages. The system combines a physical storytelling board, NFC-embedded toys, and a mobile app mediating multimodal interaction through tangible manipulation and voice-based dialogue. We conducted a home-based user study with 10 children (ages 6-8) across two conversation design conditions varying in how the agent structured elaboration, with and without educational scaffolding. Our findings show that prompt framing shapes the form and consistency of children's narrative contributions, structuring how they participate in co-creative dialogue with AI.

**arXiv ID:** 2602.04109
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
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

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
<summary><strong>Inverse Reinforcement Learning for Interpretable Keystroke Biomarkers in Parkinson's Disease</strong> - Navin Bondade - [[pdf]](https://arxiv.org/pdf/2606.25270)</summary>

**Abstract:** Keystroke dynamics have been explored extensively as a passive digital biomarker for Parkinson's disease (PD), typically by extracting summary statistics from typing timing and training a classifier to discriminate PD from healthy controls. We instead apply inverse reinforcement learning (IRL) to keystroke data, modeling each keystroke as a discrete choice over typing speed and recovering, per subject, an interpretable reward function that explains their observed timing behavior. To our knowledge this is the first application of IRL to keystroke dynamics. On the public neuroQWERTY MIT-CSXPD dataset (85 subjects, 42 with PD), an initial four-parameter reward decomposition (speed, effort, smoothness, hand-alternation cost) was found to suffer severe feature collinearity between two terms ($r=1.000$ in typical contexts); we diagnose and correct this, yielding an identifiable three-parameter model. The recovered speed-preference weight correlates with UPDRS-III severity at $r=-0.607$ ($p<0.001$, $n=42$), replicates independently across two sub-cohorts, is stable across nine sensitivity configurations, and retains a statistically significant contribution beyond raw typing speed alone (incremental $R^2$ from 0.194 to 0.338, $p=0.006$). Two other recovered weights (consistency, hand-alternation) did not survive confound checks and are reported as negative results. We document two implementation bugs found during adversarial code review (session-boundary contamination, a rolling-window data leakage) and show the headline result is materially unchanged after fixing both. We discuss this result in the context of a literature where reported accuracies vary widely between studies (pooled AUC 0.85, I^2=94% in a 2022 meta-analysis), and argue that the validation process itself, not only the correlation coefficient, is part of the contribution.

**arXiv ID:** 2606.25270
</details>

<details>
<summary><strong>Compositional Behavioral Semantics for State Abstraction in Reinforcement Learning</strong> - Yivan Zhang, Ziyan Luo, Manuel Baltieri - [[pdf]](https://arxiv.org/pdf/2606.25357)</summary>

**Abstract:** State abstraction plays a key role in scaling reinforcement learning to complex but structured systems. In studying such systems, a wide range of behavioral structures have been studied in reinforcement learning, including value functions, invariants, bisimulation relations, and behavioral metrics. However, a general principle for determining what structures are provably preserved under state abstraction is still lacking. In this paper, we present a unified framework for defining and analyzing behavioral structures in reinforcement learning. Our framework provides a compositional way to specify behavioral semantics based on local, one-step descriptions of system dynamics. Using this framework, we establish results showing how behavioral structures can be safely transferred between abstract and concrete systems. We further show how to construct quantitative metrics from logical behavioral semantics with soundness guarantees. Together, these results provide a principled foundation for reasoning about behaviors under state abstraction in reinforcement learning and offer reusable definition and proof principles for a broad class of behavioral structures in reinforcement learning.

**arXiv ID:** 2606.25357
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
<summary><strong>Hierarchical Reinforcement Learning for Neural Network Compression (HiReLC): Pruning and Quantization</strong> - Kamar Hibatallah Baghdadi, Kawther Guoual Belhamidi, Sara Belhadj, Aissa Boulmerka, Nadir Farhi - [[pdf]](https://arxiv.org/pdf/2606.26002)</summary>

**Abstract:** We present HiReLC, a hierarchical ensemble-reinforcement learning framework for automated joint quantization and structured pruning of deep neural networks. The framework decomposes the compression search across two levels of abstraction: low-level agents (LLAs) operate independently per block, selecting per-kernel configurations over a multi-discrete action space spanning bitwidth, pruning keep-ratio, quantization type, and granularity, while high-level agents (HLAs) coordinate global budget allocation via ensemble voting guided by Fisher Information-based sensitivity estimates. To mitigate the computational cost of policy evaluation, an iterative active learning loop interleaves surrogate-guided RL optimization with post-compression fine-tuning, using a lightweight MLP surrogate to amortize expensive evaluations and a logit-MSE proxy during cold-start. The surrogate is used for reward shaping rather than as a replacement for final post-compression evaluation. The controller is architecture-agnostic by design, with a modular layer abstraction decoupling the RL environment from the underlying network topology. Experiments across Vision Transformer and CNN benchmarks demonstrate effective parameter-storage compression ratios of 5.99 - 6.72$\times$ with a 3.83 % gain in one setting and 0.55 - 5.62 % accuracy drops elsewhere, supporting hierarchical policy decomposition and sensitivity-aware guidance as practical design choices for joint neural network compression.

**arXiv ID:** 2606.26002
</details>

<details>
<summary><strong>Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix It</strong> - Yupu Hao, Zhuoran Jin, Huanxuan Liao, Kang Liu, Jun Zhao - [[pdf]](https://arxiv.org/pdf/2606.26027)</summary>

**Abstract:** Tool use enables large language models (LLMs) to perform complex tasks, and recent agentic reinforcement learning (RL) methods show promise for enhancing model capabilities. However, RL alone often leads to instability or limited gains in tool-use tasks. In our experiments, some models exhibit catastrophic collapse, where performance abruptly drops and tool-invocation structures fail. The analysis reveals that these failures stem from unexpected probability spikes in specific control tokens, disrupting structured execution, yet the underlying tool-use capability remains intact, merely obscured by specific formats. To address this, we systematically investigate a diverse set of supervisory signals, including off-policy supervision, hint-based guidance, erroneous example supervision, and others, applied under both synchronous and interleaved training schemes. We find that interleaving supervised fine-tuning (SFT) with RL substantially improves stability, but exhibits degraded performance under format and content out-of-distribution (OOD) evaluation. We also analyze the impact of learning rates and generalization across settings. These results highlight the importance of understanding RL failures and demonstrate how diverse supervisory signals can guide exploratory learning, enabling robust training of LLMs for complex, multi-step tool-use tasks. Our Code is available at this https URL.

**arXiv ID:** 2606.26027
</details>

<details>
<summary><strong>Fox in the Henhouse: Supply-Chain Backdoor Attacks Against Reinforcement Learning</strong> - Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah Erfani, Benjamin I. P. Rubinstein - [[pdf]](https://arxiv.org/pdf/2505.19532)</summary>

**Abstract:** The current state-of-the-art backdoor attacks against Reinforcement Learning (RL) rely upon unrealistically permissive access models, that assume the attacker can read (or even write) the victim's policy parameters, observations, or rewards. In this work, we question whether such a strong assumption is required to launch backdoor attacks against RL. To answer this question, we propose the \underline{S}upply-\underline{C}h\underline{a}in \underline{B}ackdoor (SCAB) attack, which targets a common RL workflow: training agents using external agents that are provided separately or embedded within the environment. In contrast to prior works, our attack only relies on legitimate interactions of the RL agent with the supplied agents. Despite this limited access model, by poisoning a mere $3\%$ of training experiences, our attack can successfully activate over $90\%$ of triggered actions, reducing the average episodic return by $80\%$ for the victim. Our novel attack demonstrates that RL attacks are likely to become a reality under untrusted RL training supply-chains.

**arXiv ID:** 2505.19532
</details>

<details>
<summary><strong>Auto-exploration for online reinforcement learning</strong> - Caleb Ju, Guanghui Lan - [[pdf]](https://arxiv.org/pdf/2512.06244)</summary>

**Abstract:** The exploration-exploitation dilemma in reinforcement learning (RL) is a fundamental challenge to efficient RL algorithms. Existing algorithms for finite state and action discounted RL problems address this by assuming sufficient exploration over both state and action spaces. However, this yields non-implementable algorithms and sub-optimal performance. To resolve these limitations, we introduce a new class of methods with auto-exploration, or methods that automatically explore both state and action spaces. Auto-exploration can be applied in both the tabular and linear function approximation setting. Under algorithm-independent assumptions on the existence of an exploring optimal policy, both settings attain $O(\epsilon^{-2})$ sample complexity to solve to $\epsilon$ error. These complexities are novel since they avoid algorithm-dependent parameters seen in prior works, which may be arbitrarily large. The methods are also simple to implement because they are parameter-free. We achieve these results by integrating auto-exploration into policy mirror descent to avoid the (unknown) stationary distribution seen in prior art. In the tabular setting, we introduce a dynamic exploration time with a data-driven stopping time, while for linear function approximation we propose a new sampling distribution based on the discounted visitation distribution that covers a more general class of Markov chains.

**arXiv ID:** 2512.06244
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
<summary><strong>AI Coaching for Accelerating Human Skill Development with Reinforcement Learning</strong> - Wei Wang, Enlin Gu, Antonio Loquercio, Haimin Hu, Rahul Mangharam - [[pdf]](https://arxiv.org/pdf/2606.25337)</summary>

**Abstract:** AI copilots can substantially boost human performance through shared control, but excessive assistance can induce over-reliance and skill atrophy. This paper studies how an embodied AI agent can act as a coach that accelerates human motor-skill development. We argue that effective coaching requires strategic scaffolding and stepping back that are aligned with the learner's capability, allowing productive failures that drive learning. We formalize the interactive AI coaching process as a non-cooperative dynamic game in which the learner optimizes task performance while the coach targets the learner's independent competence. Building on this formalism, we develop a reinforcement learning framework combining adaptive shared control with probabilistic models of the coach's causal influence on skill evolution, enabling tractable training of coaching policies. A comprehensive user study (N=33) on first-person-view drone racing shows significant gains in human learning outcomes over state-of-the-art AI coaching baselines.

**arXiv ID:** 2606.25337
</details>

<details>
<summary><strong>Power-Budgeted Underwater Vehicle Control via Constrained Reinforcement Learning</strong> - Yinuo Wang, Gavin Tao, Yuze Liu - [[pdf]](https://arxiv.org/pdf/2606.25680)</summary>

**Abstract:** Underwater vehicles operate from a fixed onboard energy budget that propulsion rapidly depletes, so a controller that completes its task while drawing less thruster power directly extends mission range and endurance. Reinforcement learning yields capable model-free controllers for station-keeping and trajectory tracking, but optimizing task accuracy alone drives the policy toward oscillatory, energy-wasting actuation. The established remedy subtracts an energy penalty from the reward, yet this sets the task-power trade-off through a single weight with no physical units: a target power level cannot be specified, the weight must be re-tuned for every vehicle and task, and a mismatched weight can even raise power. This paper instead formulates energy-efficient underwater control as a constrained Markov decision process in which average thruster power is subject to an explicit budget, solved with a PPO-Lagrangian algorithm. The power level is set by declaring a budget in physical units, and a single dual variable is updated online to meet it for each vehicle and task, without manual weight search. Across three vehicles and four tasks in the MarineGym simulator, the energy-constrained policy draws the least power in all twelve settings, reducing it by 14--65\% (up to 64.9\%) over a task-only baseline and below an energy-reward baseline everywhere, while remaining the smoothest in ten settings and preserving task accuracy except in one deliberately power-limited regime. Imposing energy as an explicit constraint thus offers a tuning-free route to energy-efficient underwater control that needs no per-vehicle, per-task weight search.

**arXiv ID:** 2606.25680
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
