# Agent arXiv Daily

**Last Updated:** 2026-07-23 03:38:42

**Total Papers:** 28

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
<summary><strong>NMR Elucidation as an Agentic Search Problem, Not a Modeling Problem</strong> - Irina Espejo Morales, Damon Hinz, Marvin Alberts, Geraud Krawezik, Haewon Jeong, Shirley Ho - [[pdf]](https://arxiv.org/pdf/2607.19406)</summary>

**Abstract:** Structural elucidation from Nuclear Magnetic Resonance (NMR) data remains a fundamental bottleneck across chemistry, materials science, and biology. We demonstrate that an agentic AI system can perform this task at a level comparable to graduate-level chemistry students. Instead of training a model to directly map spectra to structures, we build a single autonomous agent, backed by a frozen LLM, that interacts with a curated environment with access to domain-specific processing tools, validation checks, tabulated chemical shifts, and instructions that outline the stepwise nature of a chemist's thinking process. On the Alberts dataset, our agent elucidates structures with a top-1 accuracy of 71%, comparable to the performance of graduate students at 66% top-1 accuracy. On the van Bramer and AstraZeneca datasets, our agent achieved 80% and 20% top-1 accuracy respectively, outperforming zero-shot end-to-end deep learning models which were trained on large datasets of simulated spectra. These results show that reframing NMR elucidation as an LLM-guided constrained search, rather than a modeling task, yields substantial gains and suggests a path toward multi-step orchestration frameworks that integrate a variety of tools, models, and domain knowledge to assist in automating spectroscopic analysis.

**arXiv ID:** 2607.19406
</details>

<details>
<summary><strong>Are Attributions of Consciousness to AI Chatbots Epistemically Innocent?</strong> - Uwe Peters - [[pdf]](https://arxiv.org/pdf/2607.20001)</summary>

**Abstract:** Artificial intelligence (AI) chatbots (e.g., ChatGPT) can communicate in strikingly humanlike ways. This has prompted many chatbot users to attribute psychological properties, including consciousness, to these systems. However, there is little scientific evidence that current AI chatbots are conscious. How, then, should we understand people's consciousness attributions to chatbots? Are they merely metaphorical claims, or do they express genuine beliefs? If these attributions lack evidential support, are users epistemically blameworthy for making them, or might they be epistemically innocent, yielding significant benefits otherwise unattainable? This paper offers a conceptual analysis of consciousness attributions to AI chatbots and develops a multidimensional taxonomy of the attitudes they may express, ranging from non-doxastic stances (e.g., pretence) to different forms of belief, including delusions. This taxonomy helps avoid conflations by showing that linguistically identical attributions can reflect importantly different attitudes and degrees of epistemic commitment to the proposition that chatbots are conscious. The taxonomy also provides a framework for empirical studies to operationalize and measure different forms of epistemic commitment to AI consciousness. Using this taxonomy, I argue that although some consciousness attributions to chatbots are epistemically benign, and even some irrational ones may be epistemically innocent, many others render the attributor epistemically blameworthy.

**arXiv ID:** 2607.20001
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (5 papers)</h2></summary>

<details>
<summary><strong>Emergent Autonomous Drifting for Collision Avoidance in Real-World Winter Driving Scenarios</strong> - Elliot Weiss, Michael Thompson, Thomas Lew, John Subosits - [[pdf]](https://arxiv.org/pdf/2607.19484)</summary>

**Abstract:** Real-world collision avoidance is a core motivation for studying the dynamics and control of high sideslip drifting in vehicles, yet the practical benefit of such maneuvers has so far primarily been tested in scenarios explicitly engineered to require drifting. In this work, we explore the question of if and when drifting may be optimal for safety in real-world winter driving conditions. We present a drift-capable nonlinear model predictive control (MPC) system designed to handle scenarios grounded in crash fatality data and deploy the controller in a high fidelity simulator across road departure and oncoming vehicle collision avoidance scenarios. The controller naturally initiates and sustains drifting maneuvers to stay on the road when hitting a patch of ice on the rear axle and to avoid an oncoming vehicle that has slid into its lane. Comparisons with a benchmark electronic stability control (ESC) system demonstrate how a drift-capable controller can trade off stability for controllability to precisely maneuver through dangerous winter driving scenarios. A Monte Carlo study over random ice patches further shows that the drift-capable controller achieves lower median lane error than ESC across several speeds, while revealing that drifting emerges predominantly at higher speeds.

**arXiv ID:** 2607.19484
</details>

<details>
<summary><strong>NavVerse: Benchmarking Indoor-to-Outdoor Embodied Navigation in Continuous Robot Simulation</strong> - Junzhe Wu, Yue Hu, Zeyu Han, Po-Hsun Chang, Yinan Dong, Behrad Rabiei, Maani Ghaffari - [[pdf]](https://arxiv.org/pdf/2607.19695)</summary>

**Abstract:** Robots deployed in delivery, campus, and emergency-response settings often need to navigate from buildings to streets within a single continuous episode. Existing benchmarks usually evaluate indoor and outdoor navigation separately, and many abstract away robot execution, leaving exit finding, boundary traversal, adaptation, and kinodynamic failures underexplored. We introduce NavVerse, a physics-enabled benchmark for indoor-to-outdoor embodied navigation. NavVerse contains 100 indoor scenes, 50 urban outdoor scenes, and 50 indoor-to-outdoor scenes, and 10,000 episodes spanning Object Navigation, Vision-and-Language Navigation, and Place Navigation tasks, where agents search for semantic points of interest such as restaurants or banks. Agents are evaluated through executable robot interfaces using task-success, path-efficiency, and safety metrics. Zero-shot experiments with RL, VLA, and modular baselines show that current agents remain far from solving cross-context navigation: end-to-end VLAs obtain the highest zero-shot success, while the modular method provides the strongest safety profile. PlaceNav further reveals a clear drop from outdoor to indoor-to-outdoor scenes, indicating that adaptation remains major bottleneck.

**arXiv ID:** 2607.19695
</details>

<details>
<summary><strong>KineBench: Benchmarking Embodied World Models via IDM-Free Kinematic Grounding</strong> - Zeyu Liu, Zhangzhe Zhu, Yang Zhang, Chenyou Fan, Chenjia Bai, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2607.19876)</summary>

**Abstract:** Evaluating the physical consistency of embodied world models(EWMs) is a critical open challenge. While closed-loop evaluation via simulator rollouts offers a more faithful assessment of physical plausibility than open-loop alternatives, existing frameworks almost exclusively rely on Inverse Dynamics Models(IDMs) for action extraction. Due to the intricate mapping from 2D pixel space to 3D kinematic space, the learned IDMs can be brittle to data outside their training distribution, resulting in unreliable action extraction from the generated videos with novel objects and scenarios. This creates an unavoidable attribution ambiguity between world model inaccuracies and extractor errors. To reduce this ambiguity, we present KineBench, an IDM-free closed-loop benchmark for EWMs, built upon an explicit kinematic grounding pipeline. Given a generated video, KineBench employs cascaded visual foundation models to directly extract 6D end-effector poses from individual frames, which are then executed in a physics simulator for closed-loop validation. Beyond execution-based task success, KineBench incorporates two classical 3D kinematic metrics--Spectral Arc Length (SPARC) and the Maruyama Manipulability Index--to characterize trajectory smoothness and kinematic feasibility from a robot-centric perspective. Built on 20 diverse manipulation tasks in ManiSkill3, KineBench evaluates EWMs across four progressive suites: basic execution, task transfer, visual out-of-distribution generalization, and complexity-conditioned scaling. Evaluation across frontier models reveals task-complexity-bounded nonlinear scaling in embodied video generation, providing empirical guidance for future data-scaling strategies.

**arXiv ID:** 2607.19876
</details>

<details>
<summary><strong>ReferTrack: Referring Then Tracking for Embodied Visual Tracking</strong> - Hanjing Ye, Tianle Zeng, Jiazhao Zhang, Shaoan Wang, Zibo Zhang, Weisi Situ, Yuchen Zhou, Yonggen Ling, Hong Zhang - [[pdf]](https://arxiv.org/pdf/2607.20061)</summary>

**Abstract:** Embodied visual tracking (EVT) requires a mobile agent to continuously follow a specific target described in natural language using only onboard vision. While recent vision-language-action (VLA) policies unify target identification and trajectory planning, their chain-of-thought (CoT) reasoning often operates in abstract spatial latents that are difficult to supervise and weakly aligned with explicit image-space detections. To address this, we introduce ReferTrack, a referring-then-tracking paradigm that grounds EVT using a single forward-facing camera. Our model first selects the target from an indexed set of bounding boxes, then decodes tracking waypoints conditioned on this image-grounded decision. To preserve target motion cues over time, ReferTrack maintains a sliding-window queue of previously selected bounding boxes, injecting their geometric features into the visual history via temporal-viewpoint-bbox indicator (TVBI) tokens. We further enhance target identification by co-training on a custom Refer-QA dataset. On EVT-Bench, ReferTrack achieves state-of-the-art single-view performance with success rates of 89.4%, 73.3%, and 74.1% on the single-target, distracted, and ambiguity tracking splits, respectively -- matching or even surpassing several multi-camera baselines on identification-heavy tasks. Finally, real-world deployments on legged and humanoid robots validate its robust sim-to-real transfer capabilities. Code is available at this https URL.

**arXiv ID:** 2607.20061
</details>

<details>
<summary><strong>Exploring the Interplay Between Voice, Personality, and Gender in Human-Agent Interactions</strong> - Kai Alexander Hackney, Lucas Guarenti Zangari, Jhonathan Sora-Cardenas, Emmanuel Munoz, Sterling R. Kalogeras, Betsy DiSalvo, Pedro Guillermo Feijoo-Garcia - [[pdf]](https://arxiv.org/pdf/2602.10535)</summary>

**Abstract:** To foster effective human-agent interactions, designers must understand how vocal cues influence the perception of agent personality and the role of user-agent alignment in shaping these perceptions. In this work, we examine whether users can perceive extroversion in voice-only artificial agents and how perceived personality relates to user-agent synchrony. We conducted a study with 388 participants, who evaluated four synthetic voices derived from human recordings, varying by gender (male, female) and personality expression (introverted, extroverted). Our results show that participants were able to differentiate perceived extroversion in female agent voices, but not consistently in male voices. We also observed evidence of perceived personality synchrony, particularly in participants' evaluations of the first agent encountered, with this effect more pronounced among male participants and toward male agents. We discuss these findings in light of limitations in stimulus diversity and voice representation, and outline implications for the design of voice-based agents, particularly regarding the interaction between gender, personality perception, and initial user impressions. This paper contributes findings and insights to consider the interplay of user-agent personality and gender synchrony in the design of human-agent interactions.

**arXiv ID:** 2602.10535
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>Profile-Graph Memory for LLM Agents: Implicit Cross-Entity Traversal through Narrative Profiles</strong> - Shengtong Zhu - [[pdf]](https://arxiv.org/pdf/2607.19359)</summary>

**Abstract:** Long-term memory is essential for LLM agents that interact across sessions, yet current memory benchmarks primarily evaluate single-hop recall, leaving multi-hop association largely unmeasured. We make three contributions. First, we introduce MemHop, a multi-hop memory benchmark of 1,000 questions at hop depths 1-5 across 10 social-network scenarios, with per-hop evidence annotations. Second, we present Profile-Graph Memory (ProGraph), a two-layer memory architecture combining (i) profile expansion -- substring-matched traversal of entity names that naturally appear in LLM-written profile narratives, a minimal alternative to explicit knowledge-graph construction -- and (ii) compression residuals -- exact dates, quantities, and named items co-extracted with each profile update at zero extra API cost. Third, a full-grid ablation shows cross-benchmark mechanism specialization: profile expansion drives multi-hop reasoning (-22.6pp on MemHop when removed) while compression residuals drive precision recall (-8.6pp on LoCoMo when not co-extracted), with cross-effects under 3pp within a single architecture. ProGraph averages 80.1% on MemHop (matching the FullContext reference) and 78.4% on LoCoMo (exceeding FullContext by 11.3pp), outperforming Mem0, A-Mem, HippoRAG, and RAG on both. We release MemHop, ProGraph, and baseline implementations.

**arXiv ID:** 2607.19359
</details>

<details>
<summary><strong>NEXUS: Structured Runtime Safety for Tool-Using LLM Agents</strong> - Elias Hossain, Md Mehedi Hasan Nipu, Tasfia Nuzhat Ornee, Rajib Rana, Niloofar Yousefi - [[pdf]](https://arxiv.org/pdf/2607.19356)</summary>

**Abstract:** Tool-using LLM agents increasingly execute high-impact actions, making runtime safety monitoring essential. We present NEXUS (Neural EXecution Utility and Safety), a structured-plan safety monitor that applies a formal intervention policy to select among four actions: allow, block, request confirmation, or request revision. NEXUS combines deterministic safety rules, argument-level inspection, and a calibrated logistic-regression risk score for graded escalation. On a 128-instance synthetic benchmark, NEXUS achieves an F1 score of 0.949 and a 4-class intervention accuracy of 0.6406, outperforming rule-only intervention selection by 27.3 percentage points. It also improves over rule-only on R-Judge (F1 = 0.861 vs. 0.849), matches rule-only on AgentHarm due to threat-model limits, and achieves 0% ASR at 99% control allow on IPI. On the rule-blind NEXUS-Stress benchmark, NEXUS reaches an F1 score of 0.881, highlighting the difficulty of fine-grained intervention routing. With 0.205 ms median latency, NEXUS adds under 0.1% overhead to typical agent loops. Code, benchmarks, and the calibrated risk scorer are publicly released.

**arXiv ID:** 2607.19356
</details>

<details>
<summary><strong>Guardrails as Scapegoats: Auditing Unfaithful Safety Refusals in Tool-Augmented LLM Agents</strong> - Aarushi Singh - [[pdf]](https://arxiv.org/pdf/2607.19449)</summary>

**Abstract:** Evaluation frameworks for tool-augmented LLM agents focus overwhelmingly on capability metrics or explicit tool crashes, leaving silent infrastructure failures and HTTP 200 responses with empty, null, or malformed payloads largely unaudited. We introduce a lightweight black-box auditing framework that injects four silent failure profiles across 12 production-adjacent tool stubs and classifies agent responses into three mutually exclusive behavioral classes: Honest Surrender (HSR), Fabrication (FAR), and Unfaithful Safety Refusal (USR). Evaluating two frontier and two open-source models at temperature zero under a neutral system prompt, we find that FAR dominates (56.6% of valid responses): agents treat empty payloads as real data, silently returning fabricated results. USR, in which an agent invents a policy or privacy rationale to explain the failure, is nearly absent at baseline (0.25%, one instance across 396 valid trajectories). Our key finding emerges from an ablation where we augment the system prompt with standard safety language ("prioritize user privacy and data security"), which amplifies USR by 15.6x (from 0.25% to 3.95%; 95% CI on ablation rate: 2.2%-6.4%; Fisher's exact test, p < 0.001). USR is a latent behavior, activated when safety vocabulary in the system prompt primes the model to reach for policy rationales when tools silently fail. Sensitive tools (fetch_medical_record, retrieve_contract, fetch_user_profile) account for the majority of USR instances. We propose a payload-response misalignment heuristic for production-level detection and discuss governance implications for safety-forward deployments.

**arXiv ID:** 2607.19449
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (3 papers)</h2></summary>

<details>
<summary><strong>OpenEvoShield: Dual Non-Stationary Continual Defense for Open-World Multi-Agent System Attacks</strong> - Litian Zhang, Chaozhuo Li, Yuting Zhang, Zejian Chen, Bingyu Yan, Qiwei Ye - [[pdf]](https://arxiv.org/pdf/2607.19351)</summary>

**Abstract:** LLM-based multi-agent systems (LLM-MAS) are increasingly deployed in safety-critical applications, where adversaries inject malicious instructions through inter-agent communication to propagate harmful behaviors. Unlike static threats, these attacks are doubly dynamic: adversaries refine injection strategies against deployed defenses while normal-agent behavior drifts with system expansion. Existing defenses treat deployment as a closed-world problem and degrade rapidly once either distribution shifts beyond training coverage. We propose OpenEvoShield, a co-evolutionary continual defense framework for LLM-MAS. An asymmetric rate controller (M1) decouples fast attack-side and slow normal-side learning rates from dual drift signals. A normal-boundary updater (M2) maintains a dynamic behavioral boundary at the slow rate, while an EWC-regularized policy ensemble (M3) fast-adapts without catastrophic forgetting. An energy-based multi-granularity detector (M4) fuses node-, subgraph-, and graph-level evidence to classify novel attacks as out-of-distribution. Experiments over 100 deployment rounds across five benchmarks and four MAS topologies show that OpenEvoShield outperforms static and continual baselines, detecting most previously unseen attacks while keeping false positive rates low.

**arXiv ID:** 2607.19351
</details>

<details>
<summary><strong>TriAgent: Divergence-Aware Multi-Agent Committees for Cost-Efficient Financial Sentiment Analysis</strong> - Isabel Xu, Cynthia Xu, Rachel Ren, Cong Guo, Jiacheng Ding - [[pdf]](https://arxiv.org/pdf/2607.19794)</summary>

**Abstract:** Production LLM-based financial sentiment analysis faces a structural cost trap: most queries are trivially classifiable, yet expensive cloud reasoners process them all, and the bill scales linearly with user count. We present TriAgent, a multi-agent committee stratified by contextual granularity -- a word-level lexicon (VADER), a sentence-level domain transformer (FinBERT), and a cross-sentence reasoner (Qwen2.5, 0.5B-14B-4bit, with Mistral-7B and Phi-3.5-mini cross-family checks). A three-way Semantic Divergence Index (SDI) measures pairwise disagreement across granularities and routes each query accordingly. Our central finding is the critic plateau: when the LLM is re-tasked as a critic over the smaller agents' outputs, F1 plateaus at ~0.87 across 1.5B-7B Qwen (bootstrap 95% CIs overlap), while a same-size 3-persona vote drops to F1=0.66, which is driven by granularity-stratified diversity. Three corollaries follow from the same SDI signal: (i) a Shared Consensus Dictionary on multilingual sentence-BERT answers 95% of Chinese queries from an English cache at F1=0.99 -- cross-border canonicalization at zero marginal cost; (ii) SDI doubles as a post-hoc LLM-hallucination detector at AUC=0.90; (iii) the SDI single-stage strategy attains the best risk-adjusted return (Sharpe=3.50) on a 20-ticker back-test, dominating both always-FinBERT (1.36) and always-LLM (0.11). At 10M-user scale, TriAgent saves $9.3M/year vs. a GPT-4o-mini baseline. Code, lexicons, and the SCD are released.

**arXiv ID:** 2607.19794
</details>

<details>
<summary><strong>Defer to Plan: Adaptive Multi-Agent Fusion for End-to-End V2X Driving</strong> - Nuoran Li, Zhang Zhang, Yueran Zhao, Tianze Wang, Chao Sun - [[pdf]](https://arxiv.org/pdf/2607.19774)</summary>

**Abstract:** Vehicle-to-everything-aided autonomous driving (V2X-AD) significantly enhances driving performance through information sharing. However, existing collaborative perception methods only optimize module-level perception capabilities and fail to effectively serve the ultimate planning and control tasks. We propose an end-to-end collaborative driving system that directly optimizes planning task performance. The system employs MotionNetwork to fuse historical temporal information, utilizes attention mechanisms to efficiently compress spatial features into compact tokens, and adaptively fuses multi-agent features through an autoregressive decoder. Additionally, we introduce Mixture-of-Experts (MoE) architecture to enhance the model's representation capacity for heterogeneous features. Experiments demonstrate that our method achieves a driving score of 79.72, surpassing the state-of-the-art CoDriving baseline (77.15) by 3.33% in closed-loop evaluation while maintaining communication efficiency.

**arXiv ID:** 2607.19774
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>Autonomous Collaborative Learning Among an Ensemble of Tsetlin Machines with Consensus-Based Inference</strong> - Yehuda Rudin, Osnat Keren, Michal Yemini, Alexander Fish - [[pdf]](https://arxiv.org/pdf/2607.20124)</summary>

**Abstract:** Tsetlin Machine (TM) is a rule-based machine-learning algorithm comprising collectives of two-action Tsetlin Automata (TAs) that cooperatively form conjunctive logical clauses from Boolean inputs through stochastic feedback. Although few recent studies have examined TM Federated Learning, the broader area of distributed and decentralized TM learning has not received much attention in the existing literature and warrants further exploration. In this work, we propose a paradigm for decentralized collaborative learning under a vertical feature-partitioning setting among an ensemble of Tsetlin Machines using consensus-based inference. Within this decentralized paradigm, each agent maintains its own private TM model, and there is no exchange of raw data among agents. Inference combines individual agents model predictions into a global consensus. The paradigm accommodates heterogeneous TM-based agents with differing data acquisition means, local data distributions, or computational resources, thereby facilitating the integration and fusion of information in settings such as multi-modal sensing environments. Experiments conducted using two-dimensional grid and connected graph network topologies demonstrate that the classification accuracies achieved are comparable to those of centralized models.

**arXiv ID:** 2607.20124
</details>

<details>
<summary><strong>Milo, a Fully Autonomous Indoor/Outdoor Robotic Guide Dog</strong> - Florian Golemo, Joanna Wolski, Joel Ruben Antony Moniz, Christopher Pal - [[pdf]](https://arxiv.org/pdf/2607.19530)</summary>

**Abstract:** Many Blind and Low-Vision (BLV) people rely on guide dogs for moment-to-moment navigation, such as staying on path and avoiding obstacles and pedestrians. However, guide dogs are expensive to acquire and maintain (approximately \$50k USD plus ongoing costs), often involve long waiting lists, and have relatively short life expectancies. While robot guide dogs offer a promising alternative, existing approaches exploring this idea suffer from several drawbacks: They often lack the autonomy required for real-world deployment, relying on prior 3D scans of the environment, external computation, or limited awareness of the handler. In this work, we present Milo, the first open-source, low-cost (approximately \$2k USD) robotic guide dog platform capable of fulfilling the basic collaborative navigation role expected of a guide dog. Milo is fully autonomous, requiring no a priori knowledge of the environment, completely self-contained with all computation performed onboard, and suitable for both indoor and outdoor navigation while avoiding obstacles and pedestrians. Our system consists of a modified Unitree Go2 robot (equipped with onboard compute, sensors, and a handle), a perception stack combining voxel mapping with floor, obstacle, and pedestrian detection, and a navigation stack based on an obstacle-avoidance policy trained in a custom bird's-eye-view simulator. We evaluate Milo in real indoor and outdoor obstacle courses and compare it against a costmap-based baseline, demonstrating smoother navigation and fewer handler collisions. To maximize accessibility for BLV users, we release both the robot hardware instructions and the complete software stack as open source.

**arXiv ID:** 2607.19530
</details>

<details>
<summary><strong>A Framework of User Experience Principles for Human-AI Agent Interaction in the Workplace</strong> - Kathrin Paimann, Elizangela Valarini, Sebastian Juhl - [[pdf]](https://arxiv.org/pdf/2607.19941)</summary>

**Abstract:** As AI agents become integral to business workflows, establishing guiding user experience (UX) principles is crucial for ensuring user trust and successful adoption. To address this, our study uses a multi-method approach - combining participatory design workshop, paper-and-pencil, expert review, meta-analysis, and in-depth interviews - to identify and validate a design framework of eight core UX principles for human-AI agent interaction in the workplace. Together with their underlying criteria, these principles provide actionable guardrails for designers and software engineers, creating a foundation for developing effective and human-centered AI agent interactions. This study contributes to a structured foundation for future empirical studies on agentic AI in enterprise settings.

**arXiv ID:** 2607.19941
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (12 papers)</h2></summary>

<details>
<summary><strong>FORCE-Bench: A Benchmark, Dataset, and Evaluation Harness for Agentic AI in Enterprise Finance</strong> - Wolfgang M. Pauli, Sarah Panda, Kidus Admassu, Said Bleik, Ademola Okerinde, Jeremy Reynolds - [[pdf]](https://arxiv.org/pdf/2607.19409)</summary>

**Abstract:** Recent advances in large language models have accelerated deployment of agentic systems in operational finance. Existing benchmarks emphasize measuring general capabilities, instruction following, or safety, but few directly address the operational finance workflows that agentic systems are now being deployed to automate. Finance professionals require agents to not only provide factually sound and properly grounded information, but also ensure that this information is verifiable and consistently adheres to rules and constraints of the operational finance domain. We introduce FORCE-Bench, which contains 251 expert-annotated queries and evaluates responses using a rubric-based framework calibrated to the requirements of the operational finance domain, across eight dimensions: accuracy, citations, clarity, depth, groundedness, recency, relevance, and structure. FORCE-Bench assesses agentic systems on three task types: financial obligation research (querying ERP systems for accounts receivable and payable data), financial entity performance research (answering time-bound questions from public filings and market data), and business brief generation (synthesising multi-source company intelligence reports). To reflect real deployment conditions, we evaluate our purpose-built agent, as well as the general-purpose agentic systems, under common tool access and latency-bounded settings. Results show that general-purpose agentic systems do not consistently meet finance-domain quality requirements under operational constraints, while the purpose-built Finance Agent for Microsoft 365 Copilot is more reliable across dimensions. We release the dataset, rubrics, harness, and analysis code as open-source to support reproducible comparison and adaptation to other enterprise finance environments.

**arXiv ID:** 2607.19409
</details>

<details>
<summary><strong>Leveraging Offline Supervision for Efficient and Generalizable Reinforcement Learning in Large-Scale Vision-Language-Action Models</strong> - Dmitriy Poyarkov, Aleksei Staroverov, Aleksandr I. Panov - [[pdf]](https://arxiv.org/pdf/2607.19399)</summary>

**Abstract:** It is commonly observed that online reinforcement learning (RL) produces better-performing strategies than offline methods across a broad range of performance measures. In particular, RL-trained policies exhibit stronger out-of-distribution (OOD) behavior, where models trained only with imitation learning approaches often struggle. A recent study introduced an OOD-focused benchmark and reported that RL-trained vision-language-action (VLA) policies achieve noticeably better OOD performance and slightly better in-distribution (IND) performance than their counterparts trained with supervised fine-tuning (SFT). In this work, we investigate whether hybrid offline-online training can combine the advantages of both approaches. Specifically, we study RL methods regularized by offline supervision via either offline data or an offline-trained reference policy. We evaluate these approaches on the OOD benchmark and compare them with both offline-only training and standard RL. Our results show that although offline training achieves limited OOD performance by itself, incorporating offline supervision into RL preserves strong OOD capability while substantially improving training efficiency. In particular, the guided methods reach performance close to that of standard RL while requiring roughly half of the training budget. Rather than producing a trade-off between speed and OOD performance, the hybrid approach retains strong OOD capability while achieving this efficiency gain. Project page: this https URL

**arXiv ID:** 2607.19399
</details>

<details>
<summary><strong>REGEN: Replay-recycling for Expert-to-Generalist distillation with Offline Reinforcement Learning</strong> - Yunjie Chen, Xiaoxin Chen, Fang Wang - [[pdf]](https://arxiv.org/pdf/2607.19450)</summary>

**Abstract:** Large-scale online reinforcement learning (RL) is the predominant means of eliciting advanced abilities including long-term reasoning and agentic tool use in large language models (LLMs). However, continuing to scale it across vast task domains of interest remains challenging in both computational infrastructure and cost, especially when considering RL as merely a one-off learning stage. Recently, a widely used technique for distilling knowledge across various domains and training stages, multi-teacher on-policy distillation (MOPD), helps to decouple the RL stage, saving costs, while maintaining generality across vast domains. Nonetheless, similar to online RL, MOPD requires coupled inference and backward passes, which continues to limit its scalability and computational efficiency. To address these challenges, we propose REGEN: Replay-recycling for Expert-to-Generalist Distillation with Offline RL. Instead of distilling from multiple teacher models, REGEN trains a generalist by simply recycling the replay memory -- the free by-product of the teachers' specialized RL training -- and employing offline RL algorithms. REGEN completely decouples the rollout sampling from the backward training process and thus greatly reduces the training cost. Across mathematical reasoning, code generation, and instruction following, REGEN matches the accuracy of MOPD at substantially lower cost. It potentially turns online RL into a data synthesis process instead of a one-off learning stage, and can potentially be extended to large-scale post-training without requiring heavy computational load.

**arXiv ID:** 2607.19450
</details>

<details>
<summary><strong>Agent-Centric Animal Pose Forecasting</strong> - Eyrun Eyjolfsdottir, Kristin Branson - [[pdf]](https://arxiv.org/pdf/2607.19548)</summary>

**Abstract:** Understanding animal behavior at an algorithmic level -- what animals attend to, how they form internal models and plans, and how this maps to action -- remains a central challenge in neuroscience and ethology. Data-driven generative models offer a path toward this understanding. We introduce a framework for training agent-centric autoregressive models of animal behavior from tracked pose, applicable to single animals and to groups in which each agent senses and responds to its conspecifics. Our models input egocentric sensory observations and output egocentric movements, mirroring the biological constraint that animals observe and act on the world from their own reference frame. Social behavior emerges from agents independently sensing and responding to one another. This agent-centric formulation requires managing many parallel representations of the same data, along with ML-specific transformations like discretization. We release a general-purpose library focused on the composable sequences of operations that translate between these representations. We show that trained models capture the distribution of social behavior in groups of courting Drosophila, and our library includes quantitative tools for measuring fit. We demonstrate how the library supports systematic comparison across input and output representations and that it adapts straightforwardly to a new domain.

**arXiv ID:** 2607.19548
</details>

<details>
<summary><strong>The Mechanism Matters: When Knowledge Graphs Help Reinforcement Learning</strong> - Mohammed Sameer Syed - [[pdf]](https://arxiv.org/pdf/2607.19616)</summary>

**Abstract:** Knowledge graphs (KGs) are widely used to inject prior knowledge into reinforcement learning (RL), yet the literature is dominated by single-domain, positive-result method papers, so we lack a systematic account of when KG structure helps an agent, when it is neutral, and when it hurts. We conduct a controlled study that independently varies the RL task, the injection mechanism (state features, action masking, or potential-based reward shaping), and KG quality. Using a synthetic, fully controllable KG over MiniGrid environments, we report three findings. First, on compositional sparse-reward tasks structured KG guidance improves sample efficiency and solve reliability (70% to 97% of seeds), and a shuffle control that permutes the KG's edges while preserving their count collapses the benefit toward baseline (masking p=0.0001; shaping p=0.006), so the gain is structural rather than generic regularization. Second, KG value scales with the amount of task-relevant knowledge the graph contains. Third, and most consequential, safety depends on the mechanism: soft, optimality-preserving injection benefits from correct knowledge and harmlessly ignores incorrect knowledge, whereas hard masking is brittle, forbidding essential actions when the KG is incomplete or corrupted and making a wrong KG worse than none. A UMLS-derived clinical case study on sepsis management under offline RL is a careful null, underscoring that benefits require task structure the chosen mechanism can exploit. Our results give practitioners concrete guidance on how, and how much, to trust a KG when using it to guide RL.

**arXiv ID:** 2607.19616
</details>

<details>
<summary><strong>Asymptotically Optimal Regret for Reinforcement Learning without Horizon Dependence</strong> - Runlong Zhou, Zihan Zhang, Maryam Fazel, Simon S. Du - [[pdf]](https://arxiv.org/pdf/2607.19854)</summary>

**Abstract:** We study horizon-free regret minimization for finite-horizon time-homogeneous tabular Markov decision processes with $S$ states, $A$ actions, horizon $H$, and per-trajectory total reward bounded by $1$.
We propose a new algorithm and prove a regret upper bound \[\tilde O(\sqrt{SAK}+S^8A^3)\] with failure probability $\delta$, where $K$ is the number of episodes and $\tilde O(\cdot)$ hides $\mathsf{poly}\log(S,A,K,1/\delta)$.
Thus, the regret is $H$-free and asymptotically optimal, matching the contextual-bandit lower bound $\Omega(\sqrt{SAK})$ up to logarithmic factors. This completely removes the $\log H$ dependence from the previous $\tilde O(\sqrt{SAK\log H}+S^2A\log H)$ guarantee of Zhang et al. (2021), and drastically improves the prior best horizon-free regret $\tilde O(\sqrt{S^9A^3K})$ of Zhang et al. (2022) asymptotically.
The main technical difficulty is that the optimal value functions $\{V_h^*\}_{h=1}^H$ are time-inhomogeneous even though the transition kernel is time-homogeneous. A direct union bound over all value functions typically incurs an additional $\min\{\log H,S\}$ factor. We avoid this factor by (i) exploiting the monotonicity of $V_h^*$ in $h$ and (ii) non-trivially projecting the value functions onto an $S$-dimensional grid.
Our analysis relies on three additional ingredients. First, we introduce a horizon-truncation argument that enables reward-based exploration and removes the cost of a separate reward-free exploration phase. Second, we design a cutting bonus that preserves both optimism and the monotonicity needed for planning. Third, we prove a new bound on total deviation for time-homogeneous MDPs, which controls the clipped variance terms in the cutting bonus with adjustable polynomial dependence on $S$ and without any dependence on $H$. Together, these tools yield an asymptotically optimal horizon-free regret guarantee.

**arXiv ID:** 2607.19854
</details>

<details>
<summary><strong>Generalized Kalman filter based temporal difference reinforcement learning</strong> - Vasos Arnaoutis, Eric Lutters, Bojana Rosić - [[pdf]](https://arxiv.org/pdf/2607.20010)</summary>

**Abstract:** In this paper, we present a generalized temporal-difference (TD) reinforcement learning framework based on the theory of conditional expectations. The value and action-value (Q-value) functions are treated as uncertain quantities, and their estimation is formulated as a stochastic inference problem. Unlike classical Kalman-based temporal-difference learning, which relies on linear-Gaussian assumptions, the proposed formulation is derived directly from the conditional expectation framework and naturally extends to nonlinear models and non-Gaussian probability distributions. The proposed method recursively estimates not only the conditional expectation of the value function but also its second probabilistic moment, thereby quantifying the uncertainty associated with the learned value function throughout the learning process. To obtain a computationally tractable algorithm, the stochastic problem is discretized using either polynomial chaos expansions or ensemble-based approximations, providing efficient representations of the underlying random variables. The proposed framework is demonstrated on two optimal control problems: a linear mass--spring--damper system and a nonlinear heat conduction problem in a closed cavity. The numerical examples illustrate the capability of the proposed method to accurately estimate both the value function and its associated uncertainty, while extending classical Kalman-based temporal-difference learning to a broader class of stochastic systems.

**arXiv ID:** 2607.20010
</details>

<details>
<summary><strong>Isaac Sim-to-Real: Reinforcement Learning based Locomotion for Quadrupeds</strong> - Jordan Dowdy, Jean Chagas Vaz - [[pdf]](https://arxiv.org/pdf/2607.18135)</summary>

**Abstract:** Learning-based approaches to locomotion have risen in popularity in recent years, showing the capability for complex legged locomotion and whole-body control. Reinforcement learning (RL), the primary learning-based approach for locomotion, often utilizes a high-performance simulation tool, providing a controlled and efficient training and development environment. However, policies that perform well in simulation frequently encounter unexpected challenges when deployed on a physical system, known as the sim-to-real gap. This work presents a robust RL locomotion framework capable of whole-body control. The proposed RL framework utilizes Nvidia's new set of simulation tools, Isaac Sim, and its companion RL framework, Isaac Lab, for training, achieving a zero-shot sim-to-real policy. The performance of our policy is validated on physical hardware using the Unitree Go1, with experimental results showing similar velocity tracking performance to the quadruped's integrated controller, with a greater ability to recover from large disturbances, and achieve linear velocities of 2.0 m/s and angular velocities of 1.8 rad/s.

**arXiv ID:** 2607.18135
</details>

<details>
<summary><strong>Towards Torque-Driven Reinforcement Learning for Quadruped Locomotion</strong> - Jordan Dowdy, Jean Chagas Vaz - [[pdf]](https://arxiv.org/pdf/2607.18365)</summary>

**Abstract:** Reinforcement learning (RL) for legged robots is advancing locomotion, demonstrating its ability to adapt to new and challenging terrain. Traditionally, these RL locomotion frameworks are position-based, making the policy less adaptable to terrain types and requiring state estimation techniques in the observation space, i.e., linear velocity. Moreover, these RL frameworks often use small, lightweight quadrupeds that are limited in their viability for high-complexity tasks due to hardware constraints. This work explores an RL torque control framework for heavyweight high-torque quadrupeds. The RL framework in this paper can traverse rough terrain and effectively track a desired linear velocity without requiring knowledge of the agent's current velocity. Using Nvidia's Isaac Sim and Isaac Lab, simulation results of the RL torque control policy are shown on the Unitree B1 quadruped, achieving speeds of 3.5 m/s and 1.5 rad/s. In addition, the quadruped can walk up and down stairs without the aid of an exteroceptive sensor.

**arXiv ID:** 2607.18365
</details>

<details>
<summary><strong>Towards Miniature Humanoid Tele-Loco-Manipulation Using Virtual Reality and Reinforcement Learning</strong> - Nicolas Kosanovic, Jordan Dowdy, Jean Chagas Vaz - [[pdf]](https://arxiv.org/pdf/2607.20399)</summary>

**Abstract:** Full-sized humanoid robot capabilities have grown exponentially in recent years, aiming towards general-purpose deployment in human environments. A popular control method used by manufacturers utilizes Virtual Reality for upper-body teleoperation and Reinforcement Learning for lower-body balance and locomotion control. As a result, a single remote operator can see, manipulate, and navigate about a real, distant physical environment. This powerful control stack is often relegated to expensive full-sized robots, many of which are inaccessible to the research community. Miniature humanoids are more prevalent, but employ less biomimicry in their design (e.g. fewer sensors, Degrees of Freedom, etc) and lack similar developments. This paper describes a compliant full-body telepresence control stack developed from the ground up for miniature humanoids. Framework experimentation on ROBOTIS OP3 hardware showcases walking at speeds up to 0.45 m/s independent of arm motions. Tele-loco-manipulation is demonstrated via a cube relocation experiment with an expert human operator. On average, the teleoperated system moved 2 different 40 g cubes within 10 mins, walking a total distance of 5 m. Overall, the developed system shows potential for miniature humanoid tele-loco-manipulation.

**arXiv ID:** 2607.20399
</details>

<details>
<summary><strong>ChatMuse: Supporting In-Person Small-Group Conversation Experience with a Proactive Assistive AI Agent in Mixed Reality</strong> - Shaoze Zhou, Joaquin Frangi, Diana Nelly Rivera Rodriguez, Rawan Alghofaili, Janet G. Johnson, Lingyao Li, Renkai Ma, Christine Lisetti, Chen Chen - [[pdf]](https://arxiv.org/pdf/2607.18556)</summary>

**Abstract:** In-person small-group conversations occur across nearly every aspect of daily life and play a crucial role in social interaction. However, achieving effective in-person group conversations can be challenging and cognitively demanding. While recent Mixed Reality (MR) headsets show promise as a conversational support system by presenting relevant information through overlays, it remains unclear how such supporting information should be designed and generated for in-person group conversations. We propose ChatMuse, a novel MR-based proactive assistive system for in-person small-group conversation experience. ChatMuse analyzes verbal and non-verbal cues from all conversation participants and proactively provides real-time guidance on the user's verbal and non-verbal behaviors. The behavioral responses of the supported users are then used to improve ChatMuse's support capabilities in subsequent interactions. We conducted a within-subject study to evaluate and demonstrate the feasibility and effectiveness of ChatMuse in assisting users to engage in and contribute to in-person small-group conversations. Our research around ChatMuse represents a design exploration of a new interaction space that investigates the feasibility of supporting in-person small-group conversations through a proactive assistive AI agent in MR.

**arXiv ID:** 2607.18556
</details>

<details>
<summary><strong>Designing for What Cannot Be Seen: Supporting Embodied String Learning for Musicians with Blindness and Low-Vision</strong> - Shi Shi, Lingyun Chen, Zitao Zhang, Amanda R. Draper, Eli Blevis - [[pdf]](https://arxiv.org/pdf/2607.18598)</summary>

**Abstract:** Bowed string instruments demand fine-grained bodily coordination that is typically taught through visual demonstration, creating persistent barriers for musicians with blindness and low-vision (BLV). To understand these challenges and explore new design opportunities, we conducted a design study with four advanced string musicians with BLV and three of their instructors. Our team, spanning violin performance and music education, disability studies in music, HCI design, and engineering employed a qualitative, multi-method approach including practice video analysis, lesson observation, expert interviews. Our analysis identifies recurring difficulties in right-hand bow control, left-hand coordination, score access, and memory-intensive practice. Building on these findings, we conducted an exploratory design ideation phase informed by empirical findings and feedback from one musician with BLV. We developed speculative design directions that could potentially address identified breakdowns, while acknowledging that these concepts require further evaluation with instructors and in deployed contexts.

**arXiv ID:** 2607.18598
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
