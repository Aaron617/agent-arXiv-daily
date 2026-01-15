# Agent arXiv Daily

**Last Updated:** 2026-01-15 03:06:33

**Total Papers:** 63

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
<summary><strong>Development and Evaluation of HopeBot: an LLM-based chatbot for structured and interactive PHQ-9 depression screening</strong> - Zhijun Guo, Alvina Lai, Julia Ive, Alexandru Petcu, Yutong Wang, Luyuan Qi, Johan H Thygesen, Kezhi Li - [[pdf]](https://arxiv.org/pdf/2507.05984)</summary>

**Abstract:** Static tools like the Patient Health Questionnaire-9 (PHQ-9) effectively screen depression but lack interactivity and adaptability. We developed HopeBot, a chatbot powered by a large language model (LLM) that administers the PHQ-9 using retrieval-augmented generation and real-time clarification. In a within-subject study, 132 adults in the United Kingdom and China completed both self-administered and chatbot versions. Scores demonstrated strong agreement (ICC = 0.91; 45% identical). Among 75 participants providing comparative feedback, 71% reported greater trust in the chatbot, highlighting clearer structure, interpretive guidance, and a supportive tone. Mean ratings (0-10) were 8.4 for comfort, 7.7 for voice clarity, 7.6 for handling sensitive topics, and 7.4 for recommendation helpfulness; the latter varied significantly by employment status and prior mental-health service use (p < 0.05). Overall, 87.1% expressed willingness to reuse or recommend HopeBot. These findings demonstrate voice-based LLM chatbots can feasibly serve as scalable, low-burden adjuncts for routine depression screening.

**arXiv ID:** 2507.05984
</details>

<details>
<summary><strong>Dialogue Telemetry: Turn-Level Instrumentation for Autonomous Information Gathering</strong> - Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo - [[pdf]](https://arxiv.org/pdf/2601.09570)</summary>

**Abstract:** Autonomous systems conducting schema-grounded information-gathering dialogues face an instrumentation gap, lacking turn-level observables for monitoring acquisition efficiency and detecting when questioning becomes unproductive. We introduce Dialogue Telemetry (DT), a measurement framework that produces two model-agnostic signals after each question-answer exchange: (i) a Progress Estimator (PE) quantifying residual information potential per category (with a bits-based variant), and (ii) a Stalling Index (SI) detecting an observable failure signature characterized by repeated category probing with semantically similar, low-marginal-gain responses. SI flags this pattern without requiring causal diagnosis, supporting monitoring in settings where attributing degradation to specific causes may be impractical. We validate DT in controlled search-and-rescue (SAR)-inspired interviews using large language model (LLM)-based simulations, distinguishing efficient from stalled dialogue traces and illustrating downstream utility by integrating DT signals into a reinforcement learning (RL) policy. Across these settings, DT provides interpretable turn-level instrumentation that improves policy performance when stalling carries operational costs.

**arXiv ID:** 2601.09570
</details>

<details>
<summary><strong>Large Multimodal Models for Embodied Intelligent Driving: The Next Frontier in Self-Driving?</strong> - Long Zhang, Yuchen Xia, Bingqing Wei, Zhen Liu, Shiwen Mao, Zhu Han, Mohsen Guizani - [[pdf]](https://arxiv.org/pdf/2601.08434)</summary>

**Abstract:** The advent of Large Multimodal Models (LMMs) offers a promising technology to tackle the limitations of modular design in autonomous driving, which often falters in open-world scenarios requiring sustained environmental understanding and logical reasoning. Besides, embodied artificial intelligence facilitates policy optimization through closed-loop interactions to achieve the continuous learning capability, thereby advancing autonomous driving toward embodied intelligent (El) driving. However, such capability will be constrained by relying solely on LMMs to enhance EI driving without joint decision-making. This article introduces a novel semantics and policy dual-driven hybrid decision framework to tackle this challenge, ensuring continuous learning and joint decision. The framework merges LMMs for semantic understanding and cognitive representation, and deep reinforcement learning (DRL) for real-time policy optimization. We starts by introducing the foundational principles of EI driving and LMMs. Moreover, we examine the emerging opportunities this framework enables, encompassing potential benefits and representative use cases. A case study is conducted experimentally to validate the performance superiority of our framework in completing lane-change planning task. Finally, several future research directions to empower EI driving are identified to guide subsequent work.

**arXiv ID:** 2601.08434
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>The Hierarchy of Agentic Capabilities: Evaluating Frontier Models on Realistic RL Environments</strong> - Logan Ritchie, Sushant Mehta, Nick Heiner, Mason Yu, Edwin Chen - [[pdf]](https://arxiv.org/pdf/2601.09032)</summary>

**Abstract:** The advancement of large language model (LLM) based agents has shifted AI evaluation from single-turn response assessment to multi-step task completion in interactive environments. We present an empirical study evaluating frontier AI models on 150 workplace tasks within a realistic e-commerce RL environment from Surge. Our analysis reveals an empirically-derived \emph{hierarchy of agentic capabilities} that models must master for real-world deployment: (1) tool use, (2) planning and goal formation, (3) adaptability, (4) groundedness, and (5) common-sense reasoning. Even the best-performing models fail approximately 40\% of the tasks, with failures clustering predictably along this hierarchy. Weaker models struggle with fundamental tool use and planning, whereas stronger models primarily fail on tasks requiring contextual inference beyond explicit instructions. We introduce a task-centric design methodology for RL environments that emphasizes diversity and domain expert contributions, provide detailed failure analysis, and discuss implications for agent development. Our findings suggest that while current frontier models can demonstrate coherent multi-step behavior, substantial capability gaps remain before achieving human-level task completion in realistic workplace settings.

**arXiv ID:** 2601.09032
</details>

<details>
<summary><strong>PersonalAlign: Hierarchical Implicit Intent Alignment for Personalized GUI Agent with Long-Term User-Centric Records</strong> - Yibo Lyu, Gongwei Chen, Rui Shao, Weili Guan, Liqiang Nie - [[pdf]](https://arxiv.org/pdf/2601.09636)</summary>

**Abstract:** While GUI agents have shown strong performance under explicit and completion instructions, real-world deployment requires aligning with users' more complex implicit intents. In this work, we highlight Hierarchical Implicit Intent Alignment for Personalized GUI Agent (PersonalAlign), a new agent task that requires agents to leverage long-term user records as persistent context to resolve omitted preferences in vague instructions and anticipate latent routines by user state for proactive assistance. To facilitate this study, we introduce AndroidIntent, a benchmark designed to evaluate agents' ability in resolving vague instructions and providing proactive suggestions through reasoning over long-term user records. We annotated 775 user-specific preferences and 215 routines from 20k long-term records across different users for evaluation. Furthermore, we introduce Hierarchical Intent Memory Agent (HIM-Agent), which maintains a continuously updating personal memory and hierarchically organizes user preferences and routines for personalization. Finally, we evaluate a range of GUI agents on AndroidIntent, including GPT-5, Qwen3-VL, and UI-TARS, further results show that HIM-Agent significantly improves both execution and proactive performance by 15.7% and 7.3%.

**arXiv ID:** 2601.09636
</details>

<details>
<summary><strong>Automating Supply Chain Disruption Monitoring via an Agentic AI Approach</strong> - Sara AlMahri, Liming Xu, Alexandra Brintrup - [[pdf]](https://arxiv.org/pdf/2601.09680)</summary>

**Abstract:** Modern supply chains are increasingly exposed to disruptions from geopolitical events, demand shocks, trade restrictions, to natural disasters. While many of these disruptions originate deep in the supply network, most companies still lack visibility beyond Tier-1 suppliers, leaving upstream vulnerabilities undetected until the impact cascades downstream. To overcome this blind-spot and move from reactive recovery to proactive resilience, we introduce a minimally supervised agentic AI framework that autonomously monitors, analyses, and responds to disruptions across extended supply networks. The architecture comprises seven specialised agents powered by large language models and deterministic tools that jointly detect disruption signals from unstructured news, map them to multi-tier supplier networks, evaluate exposure based on network structure, and recommend mitigations such as alternative sourcing options. \rev{We evaluate the framework across 30 synthesised scenarios covering three automotive manufacturers and five disruption classes. The system achieves high accuracy across core tasks, with F1 scores between 0.962 and 0.991, and performs full end-to-end analyses in a mean of 3.83 minutes at a cost of \$0.0836 per disruption. Relative to industry benchmarks of multi-day, analyst-driven assessments, this represents a reduction of more than three orders of magnitude in response time. A real-world case study of the 2022 Russia-Ukraine conflict further demonstrates operational applicability. This work establishes a foundational step toward building resilient, proactive, and autonomous supply chains capable of managing disruptions across deep-tier networks.

**arXiv ID:** 2601.09680
</details>

<details>
<summary><strong>Companion Agents: A Table-Information Mining Paradigm for Text-to-SQL</strong> - Jiahui Chen, Lei Fu, Jian Cui, Yu Lei, Zhenning Dong - [[pdf]](https://arxiv.org/pdf/2601.08838)</summary>

**Abstract:** Large-scale Text-to-SQL benchmarks such as BIRD typically assume complete and accurate database annotations as well as readily available external knowledge, which fails to reflect common industrial settings where annotations are missing, incomplete, or erroneous. This mismatch substantially limits the real-world applicability of state-of-the-art (SOTA) Text-to-SQL systems. To bridge this gap, we explore a database-centric approach that leverages intrinsic, fine-grained information residing in relational databases to construct missing evidence and improve Text-to-SQL accuracy under annotation-scarce conditions. Our key hypothesis is that when a query requires multi-step reasoning over extensive table information, existing methods often struggle to reliably identify and utilize the truly relevant knowledge. We therefore propose to "cache" query-relevant knowledge on the database side in advance, so that it can be selectively activated at inference time. Based on this idea, we introduce Companion Agents (CA), a new Text-to-SQL paradigm that incorporates a group of agents accompanying database schemas to proactively mine and consolidate hidden inter-table relations, value-domain distributions, statistical regularities, and latent semantic cues before query generation. Experiments on BIRD under the fully missing evidence setting show that CA recovers +4.49 / +4.37 / +14.13 execution accuracy points on RSL-SQL / CHESS / DAIL-SQL, respectively, with larger gains on the Challenging subset +9.65 / +7.58 / +16.71. These improvements stem from CA's automatic database-side mining and evidence construction, suggesting a practical path toward industrial-grade Text-to-SQL deployment without reliance on human-curated evidence.

**arXiv ID:** 2601.08838
</details>

<details>
<summary><strong>Blue Teaming Function-Calling Agents</strong> - Greta Dolcetti, Giulio Zizzo, Sergio Maffeis - [[pdf]](https://arxiv.org/pdf/2601.09292)</summary>

**Abstract:** We present an experimental evaluation that assesses the robustness of four open source LLMs claiming function-calling capabilities against three different attacks, and we measure the effectiveness of eight different defences. Our results show how these models are not safe by default, and how the defences are not yet employable in real-world scenarios.

**arXiv ID:** 2601.09292
</details>

<details>
<summary><strong>Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception</strong> - Zhen Wan, Chao-Han Huck Yang, Jinchuan Tian, Hanrong Ye, Ankita Pasad, Szu-wei Fu, Arushi Goel, Ryo Hachiuma, Shizhe Diao, Kunal Dhawan, Sreyan Ghosh, Yusuke Hirota, Zhehuai Chen, Rafael Valle, Ehsan Hosseini Asl, Chenhui Chu, Shinji Watanabe, Yu-Chiang Frank Wang, Boris Ginsburg - [[pdf]](https://arxiv.org/pdf/2601.09413)</summary>

**Abstract:** We introduce a voice-agentic framework that learns one critical omni-understanding skill: knowing when to trust itself versus when to consult external audio perception. Our work is motivated by a crucial yet counterintuitive finding: naively fine-tuning an omni-model on both speech recognition and external sound understanding tasks often degrades performance, as the model can be easily misled by noisy hypotheses. To address this, our framework, Speech-Hands, recasts the problem as an explicit self-reflection decision. This learnable reflection primitive proves effective in preventing the model from being derailed by flawed external candidates. We show that this agentic action mechanism generalizes naturally from speech recognition to complex, multiple-choice audio reasoning. Across the OpenASR leaderboard, Speech-Hands consistently outperforms strong baselines by 12.1% WER on seven benchmarks. The model also achieves 77.37% accuracy and high F1 on audio QA decisions, showing robust generalization and reliability across diverse audio question answering datasets. By unifying perception and decision-making, our work offers a practical path toward more reliable and resilient audio intelligence.

**arXiv ID:** 2601.09413
</details>

<details>
<summary><strong>AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling</strong> - Zhining Zhang, Chuanyang Jin, Mung Yao Jia, Shunchi Zhang, Tianmin Shu - [[pdf]](https://arxiv.org/pdf/2502.15676)</summary>

**Abstract:** Theory of Mind (ToM), the ability to understand people's minds based on their behavior, is key to developing socially intelligent agents. Current approaches to ToM reasoning either rely on prompting Large Language Models (LLMs), which are prone to systematic errors, or use handcrafted, rigid agent models for model-based inference, which are more robust but fail to generalize across domains. In this work, we introduce AutoToM, an automated agent modeling method for scalable, robust, and interpretable mental inference. Given a ToM problem, AutoToM first proposes an initial agent model and then performs automated Bayesian inverse planning based on this model, leveraging an LLM backend. Guided by inference uncertainty, it iteratively refines the model by introducing additional mental variables and/or incorporating more timesteps in the context. Across five diverse benchmarks, AutoToM outperforms existing ToM methods and even large reasoning models. Additionally, we show that AutoToM can produce human-like confidence estimates and enable online mental inference for embodied decision-making.

**arXiv ID:** 2502.15676
</details>

<details>
<summary><strong>AQ-PCDSys: An Adaptive Quantized Planetary Crater Detection System for Autonomous Space Exploration</strong> - Aditri Paul, Archan Paul - [[pdf]](https://arxiv.org/pdf/2508.18025)</summary>

**Abstract:** Successful autonomous planetary exploration hinges on real-time, high-fidelity environmental perception. However, standard deep learning models usually demand far more memory and computation power than space-qualified, radiation-hardened onboard hardware can provide. This creates a fundamental design challenge of deploying sophisticated detection architectures without saturating the rigid power and memory envelopes of the computation hardware of planetary exploration platforms. We propose the Adaptive Quantized Planetary Crater Detection System to resolve this bottleneck. Our framework integrates a Quantized Neural Network, refined through Quantization Aware Training, with an Adaptive Multi-Sensor Fusion module. By forcing weights into low-precision integer arithmetic, we effectively strip away the floating-point overhead that typically bottlenecks onboard processors and system memory. This yields a leaner model footprint and significantly faster processing while the detection fidelity remains high. Such efficiency enables AMF module to merge high-bandwidth Optical Imagery streams with Digital Elevation Models using an Adaptive Weighting Mechanism to re-balance sensor priority under variable conditions like deep shadows or high albedo. Integrated Multi-Scale Detection Heads then resolve craters across a wide range of diameters, providing a computationally efficient and precise solution for real-time detection, localization of craters and hazard avoidance. This paper establishes the architectural design and theoretical justification of the system. While our methodology is grounded in principles of hybrid computer vision and planetary science, we present this as a blueprint for future empirical validation and hardware benchmarking on integer-arithmetic units. This system provides a capability vital for the next generation of autonomous landing, navigation, and deep space explorations.

**arXiv ID:** 2508.18025
</details>

<details>
<summary><strong>DeepResearchEval: An Automated Framework for Deep Research Task Construction and Agentic Evaluation</strong> - Yibo Wang, Lei Wang, Yue Deng, Keming Wu, Yao Xiao, Huanjin Yao, Liwei Kang, Hai Ye, Yongcheng Jing, Lidong Bing - [[pdf]](https://arxiv.org/pdf/2601.09688)</summary>

**Abstract:** Deep research systems are widely used for multi-step web research, analysis, and cross-source synthesis, yet their evaluation remains challenging. Existing benchmarks often require annotation-intensive task construction, rely on static evaluation dimensions, or fail to reliably verify facts when citations are missing. To bridge these gaps, we introduce DeepResearchEval, an automated framework for deep research task construction and agentic evaluation. For task construction, we propose a persona-driven pipeline generating realistic, complex research tasks anchored in diverse user profiles, applying a two-stage filter Task Qualification and Search Necessity to retain only tasks requiring multi-source evidence integration and external retrieval. For evaluation, we propose an agentic pipeline with two components: an Adaptive Point-wise Quality Evaluation that dynamically derives task-specific evaluation dimensions, criteria, and weights conditioned on each generated task, and an Active Fact-Checking that autonomously extracts and verifies report statements via web search, even when citations are missing.

**arXiv ID:** 2601.09688
</details>

<details>
<summary><strong>AgenticIE: An Adaptive Agent for Information Extraction from Complex Regulatory Documents</strong> - Gaye Colakoglu, Gürkan Solmaz, Jonathan Fürst - [[pdf]](https://arxiv.org/pdf/2509.11773)</summary>

**Abstract:** Declaration of Performance (DoP) documents, mandated by EU regulation, specify characteristics of construction products, such as fire resistance and insulation. While this information is essential for quality control and reducing carbon footprints, it is not easily machine readable. Despite content requirements, DoPs exhibit significant variation in layout, schema, and format, further complicated by their multilingual nature. In this work, we propose DoP Key Information Extraction (KIE) and Question Answering (QA) as new NLP challenges. To address this challenge, we design a domain-specific AgenticIE system based on a planner-executor-corresponder pattern. For evaluation, we introduce a high-density, expert-annotated dataset of complex, multi-page regulatory documents in English and German. Unlike standard IE datasets (e.g., FUNSD, CORD) with sparse annotations, our dataset contains over 15K annotated entities, averaging over 190 annotations per document. Our agentic system outperforms static and multimodal LLM baselines, achieving Exact Match (EM) scores of 0.396 vs. 0.342 (GPT-4o, +16%) and 0.314 (GPT-4o-V, +26%) across the KIE and QA tasks. Our experimental analysis validates the benefits of the agentic system, as well as the challenging nature of our new DoP dataset.

**arXiv ID:** 2509.11773
</details>

<details>
<summary><strong>To Retrieve or To Think? An Agentic Approach for Context Evolution</strong> - Rubing Chen, Jian Wang, Wenjie Li, Xiao-Yong Wei, Qing Li - [[pdf]](https://arxiv.org/pdf/2601.08747)</summary>

**Abstract:** Current context augmentation methods, such as retrieval-augmented generation, are essential for solving knowledge-intensive reasoning tasks. However, they typically adhere to a rigid, brute-force strategy that executes retrieval at every step. This indiscriminate approach not only incurs unnecessary computational costs but also degrades performance by saturating the context with irrelevant noise. To address these limitations, we introduce Agentic Context Evolution (ACE), a framework inspired by human metacognition that dynamically determines whether to seek new evidence or reason with existing knowledge. ACE employs a central orchestrator agent to make decisions strategically via majority voting. It aims to alternate between activating a retriever agent for external retrieval and a reasoner agent for internal analysis and refinement. By eliminating redundant retrieval steps, ACE maintains a concise and evolved context. Extensive experiments on challenging multi-hop QA benchmarks demonstrate that ACE significantly outperforms competitive baselines in accuracy while achieving efficient token consumption. Our work provides valuable insights into advancing context-evolved generation for complex, knowledge-intensive tasks.

**arXiv ID:** 2601.08747
</details>

<details>
<summary><strong>CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion</strong> - Ralf Römer, Yi Zhang, Angela P. Schoellig - [[pdf]](https://arxiv.org/pdf/2601.09512)</summary>

**Abstract:** To teach robots complex manipulation tasks, it is now a common practice to fine-tune a pre-trained vision-language-action model (VLA) on task-specific data. However, since this recipe updates existing representations, it is unsuitable for long-term operation in the real world, where robots must continually adapt to new tasks and environments while retaining the knowledge they have already acquired. Existing continual learning methods for robotics commonly require storing previous data (exemplars), struggle with long task sequences, or rely on task identifiers for deployment. To address these limitations, we propose CLARE, a general, parameter-efficient framework for exemplar-free continual learning with VLAs. CLARE introduces lightweight modular adapters into selected feedforward layers and autonomously expands the model only where necessary when learning a new task, guided by layer-wise feature similarity. During deployment, an autoencoder-based routing mechanism dynamically activates the most relevant adapters without requiring task labels. Through extensive experiments on the LIBERO benchmark, we show that CLARE achieves high performance on new tasks without catastrophic forgetting of earlier tasks, significantly outperforming even exemplar-based methods. Code and data are available at this https URL.

**arXiv ID:** 2601.09512
</details>

<details>
<summary><strong>ReflexDiffusion: Reflection-Enhanced Trajectory Planning for High-lateral-acceleration Scenarios in Autonomous Driving</strong> - Xuemei Yao, Xiao Yang, Jianbin Sun, Liuwei Xie, Xuebin Shao, Xiyu Fang, Hang Su, Kewei Yang - [[pdf]](https://arxiv.org/pdf/2601.09377)</summary>

**Abstract:** Generating safe and reliable trajectories for autonomous vehicles in long-tail scenarios remains a significant challenge, particularly for high-lateral-acceleration maneuvers such as sharp turns, which represent critical safety situations. Existing trajectory planners exhibit systematic failures in these scenarios due to data imbalance. This results in insufficient modelling of vehicle dynamics, road geometry, and environmental constraints in high-risk situations, leading to suboptimal or unsafe trajectory prediction when vehicles operate near their physical limits. In this paper, we introduce ReflexDiffusion, a novel inference-stage framework that enhances diffusion-based trajectory planners through reflective adjustment. Our method introduces a gradient-based adjustment mechanism during the iterative denoising process: after each standard trajectory update, we compute the gradient between the conditional and unconditional noise predictions to explicitly amplify critical conditioning signals, including road curvature and lateral vehicle dynamics. This amplification enforces strict adherence to physical constraints, particularly improving stability during high-lateral-acceleration maneuvers where precise vehicle-road interaction is paramount. Evaluated on the nuPlan Test14-hard benchmark, ReflexDiffusion achieves a 14.1% improvement in driving score for high-lateral-acceleration scenarios over the state-of-the-art (SOTA) methods. This demonstrates that inference-time trajectory optimization can effectively compensate for training data sparsity by dynamically reinforcing safety-critical constraints near handling limits. The framework's architecture-agnostic design enables direct deployment to existing diffusion-based planners, offering a practical solution for improving autonomous vehicle safety in challenging driving conditions.

**arXiv ID:** 2601.09377
</details>

</details>

<details open>
<summary><h2>LLM Agents (9 papers)</h2></summary>

<details>
<summary><strong>MAXS: Meta-Adaptive Exploration with LLM Agents</strong> - Jian Zhang, Zhiyuan Wang, Zhangqi Wang, Yu He, Haoran Luo, li yuan, Lingling Zhang, Rui Mao, Qika Lin, Jun Liu - [[pdf]](https://arxiv.org/pdf/2601.09259)</summary>

**Abstract:** Large Language Model (LLM) Agents exhibit inherent reasoning abilities through the collaboration of multiple tools. However, during agent inference, existing methods often suffer from (i) locally myopic generation, due to the absence of lookahead, and (ii) trajectory instability, where minor early errors can escalate into divergent reasoning paths. These issues make it difficult to balance global effectiveness and computational efficiency. To address these two issues, we propose meta-adaptive exploration with LLM agents this https URL, a meta-adaptive reasoning framework based on LLM Agents that flexibly integrates tool execution and reasoning planning. MAXS employs a lookahead strategy to extend reasoning paths a few steps ahead, estimating the advantage value of tool usage, and combines step consistency variance and inter-step trend slopes to jointly select stable, consistent, and high-value reasoning steps. Additionally, we introduce a trajectory convergence mechanism that controls computational cost by halting further rollouts once path consistency is achieved, enabling a balance between resource efficiency and global effectiveness in multi-tool reasoning. We conduct extensive empirical studies across three base models (MiMo-VL-7B, Qwen2.5-VL-7B, Qwen2.5-VL-32B) and five datasets, demonstrating that MAXS consistently outperforms existing methods in both performance and inference efficiency. Further analysis confirms the effectiveness of our lookahead strategy and tool usage.

**arXiv ID:** 2601.09259
</details>

<details>
<summary><strong>Long-term Task-oriented Agent: Proactive Long-term Intent Maintenance in Dynamic Environments</strong> - Qinglong Shi, Donghai Wang, Hantao Zhou, Jiguo Li, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He - [[pdf]](https://arxiv.org/pdf/2601.09382)</summary>

**Abstract:** Current large language model agents predominantly operate under a reactive paradigm, responding only to immediate user queries within short-term sessions. This limitation hinders their ability to maintain long-term user's intents and dynamically adapt to evolving external environments. In this paper, we propose a novel interaction paradigm for proactive Task-oriented Agents capable of bridging the gap between relatively static user's needs and a dynamic environment. We formalize proactivity through two key capabilities, (i) Intent-Conditioned Monitoring: The agent autonomously formulates trigger conditions based on dialog history; (ii) Event-Triggered Follow-up: The agent actively engages the user upon detecting useful environmental updates. We introduce a high-quality data synthesis pipeline to construct complex, multi-turn dialog data in a dynamic environment. Furthermore, we attempt to address the lack of evaluation criteria of task-oriented interaction in a dynamic environment by proposing a new benchmark, namely ChronosBench. We evaluated some leading close-source and open-source models at present and revealed their flaws in long-term task-oriented interaction. Furthermore, our fine-tuned model trained using synthetic data for supervised learning achieves a task completion rate of 85.19% for complex tasks including shifts in user intent, outperforming other models under test. And the result validated the effectiveness of our data-driven strategy.

**arXiv ID:** 2601.09382
</details>

<details>
<summary><strong>What Do LLM Agents Know About Their World? Task2Quiz: A Paradigm for Studying Environment Understanding</strong> - Siyuan Liu, Hongbang Yuan, Xinze Li, Ziyue Zhu, Yixin Cao, Yu-Gang Jiang - [[pdf]](https://arxiv.org/pdf/2601.09503)</summary>

**Abstract:** Large language model (LLM) agents have demonstrated remarkable capabilities in complex decision-making and tool-use tasks, yet their ability to generalize across varying environments remains a under-examined concern. Current evaluation paradigms predominantly rely on trajectory-based metrics that measure task success, while failing to assess whether agents possess a grounded, transferable model of the environment. To address this gap, we propose Task-to-Quiz (T2Q), a deterministic and automated evaluation paradigm designed to decouple task execution from world-state understanding. We instantiate this paradigm in T2QBench, a suite comprising 30 environments and 1,967 grounded QA pairs across multiple difficulty levels. Our extensive experiments reveal that task success is often a poor proxy for environment understanding, and that current memory machanism can not effectively help agents acquire a grounded model of the environment. These findings identify proactive exploration and fine-grained state representation as primary bottlenecks, offering a robust foundation for developing more generalizable autonomous agents.

**arXiv ID:** 2601.09503
</details>

<details>
<summary><strong>LLMs can Compress LLMs: Adaptive Pruning by Agents</strong> - Sai Varun Kodathala, Rakesh Vunnam - [[pdf]](https://arxiv.org/pdf/2601.09694)</summary>

**Abstract:** As Large Language Models (LLMs) continue to scale, post-training pruning has emerged as a promising approach to reduce computational costs while preserving performance. Existing methods such as SparseGPT and Wanda achieve high sparsity through layer-wise weight reconstruction or activation-aware magnitude pruning, but rely on uniform or hand-crafted heuristics to determine per-layer sparsity ratios. Moreover, recent work has shown that pruned LLMs suffer from severe factual knowledge degradation, with structured pruning methods experiencing near-total collapse in factual question-answering capabilities. We introduce agent-guided pruning, where a foundation model acts as an adaptive pruning agent to intelligently select which layers to prune at each iteration while preserving critical knowledge pathways. Our method constructs layer-wise sensitivity profiles by combining Wanda-inspired weight-activation metrics with gradient importance scores, normalized as z-scores for model-agnostic comparison. These statistics are processed by an LLM agent equipped with self-reflection capabilities, enabling it to learn from previous pruning outcomes and iteratively refine its strategy. A checkpoint rollback mechanism maintains model quality by reverting when perplexity degradation exceeds a threshold. We evaluate our approach on Qwen3 models (4B and 8B parameters) at approximately 45% sparsity, demonstrating substantial improvements over structured pruning baselines: 56% relative improvement in MMLU accuracy, 19x better factual knowledge retention on FreebaseQA, and 69% lower perplexity degradation. Notably, our framework requires no retraining, operates in a model-agnostic manner, and exhibits effective self-correction with only 2-4 rollbacks across 21-40 iterations, demonstrating that foundation models can effectively guide the compression of other foundation models.

**arXiv ID:** 2601.09694
</details>

<details>
<summary><strong>The Agentic Leash: Extracting Causal Feedback Fuzzy Cognitive Maps with LLMs</strong> - Akash Kumar Panda, Olaoluwa Adigun, Bart Kosko - [[pdf]](https://arxiv.org/pdf/2601.00097)</summary>

**Abstract:** We design a large-language-model (LLM) agent that extracts causal feedback fuzzy cognitive maps (FCMs) from raw text. The causal learning or extraction process is agentic both because of the LLM's semi-autonomy and because ultimately the FCM dynamical system's equilibria drive the LLM agents to fetch and process causal text. The fetched text can in principle modify the adaptive FCM causal structure and so modify the source of its quasi-autonomy--its equilibrium limit cycles and fixed-point attractors. This bidirectional process endows the evolving FCM dynamical system with a degree of autonomy while still staying on its agentic leash. We show in particular that a sequence of three finely tuned system instructions guide an LLM agent as it systematically extracts key nouns and noun phrases from text, as it extracts FCM concept nodes from among those nouns and noun phrases, and then as it extracts or infers partial or fuzzy causal edges between those FCM nodes. We test this FCM generation on a recent essay about the promise of AI from the late diplomat and political theorist Henry Kissinger and his colleagues. This three-step process produced FCM dynamical systems that converged to the same equilibrium limit cycles as did the human-generated FCMs even though the human-generated FCM differed in the number of nodes and edges. A final FCM mixed generated FCMs from separate Gemini and ChatGPT LLM agents. The mixed FCM absorbed the equilibria of its dominant mixture component but also created new equilibria of its own to better approximate the underlying causal dynamical system.

**arXiv ID:** 2601.00097
</details>

<details>
<summary><strong>Effects of personality steering on cooperative behavior in Large Language Model agents</strong> - Mizuki Sakai, Mizuki Yokoyama, Wakaba Tateishi, Genki Ichinose - [[pdf]](https://arxiv.org/pdf/2601.05302)</summary>

**Abstract:** Large language models (LLMs) are increasingly used as autonomous agents in strategic and social interactions. Although recent studies suggest that assigning personality traits to LLMs can influence their behavior, how personality steering affects cooperation under controlled conditions remains unclear. In this study, we examine the effects of personality steering on cooperative behavior in LLM agents using repeated Prisoner's Dilemma games. Based on the Big Five framework, we first measure basic personality scores of three models, GPT-3.5-turbo, GPT-4o, and GPT-5, using the Big Five Inventory. We then compare behavior under baseline and personality-informed conditions, and further analyze the effects of independently manipulating each personality dimension to extreme values. Our results show that agreeableness is the dominant factor promoting cooperation across all models, while other personality traits have limited impact. Explicit personality information increases cooperation but can also raise vulnerability to exploitation, particularly in earlier-generation models. In contrast, later-generation models exhibit more selective cooperation. These findings indicate that personality steering acts as a behavioral bias rather than a deterministic control mechanism.

**arXiv ID:** 2601.05302
</details>

<details>
<summary><strong>MPCI-Bench: A Benchmark for Multimodal Pairwise Contextual Integrity Evaluation of Language Model Agents</strong> - Shouju Wang, Haopeng Zhang - [[pdf]](https://arxiv.org/pdf/2601.08235)</summary>

**Abstract:** As language-model agents evolve from passive chatbots into proactive assistants that handle personal data, evaluating their adherence to social norms becomes increasingly critical, often through the lens of Contextual Integrity (CI). However, existing CI benchmarks are largely text-centric and primarily emphasize negative refusal scenarios, overlooking multimodal privacy risks and the fundamental trade-off between privacy and utility. In this paper, we introduce MPCI-Bench, the first Multimodal Pairwise Contextual Integrity benchmark for evaluating privacy behavior in agentic settings. MPCI-Bench consists of paired positive and negative instances derived from the same visual source and instantiated across three tiers: normative Seed judgments, context-rich Story reasoning, and executable agent action Traces. Data quality is ensured through a Tri-Principle Iterative Refinement pipeline. Evaluations of state-of-the-art multimodal models reveal systematic failures to balance privacy and utility and a pronounced modality leakage gap, where sensitive visual information is leaked more frequently than textual information. We will open-source MPCI-Bench to facilitate future research on agentic CI.

**arXiv ID:** 2601.08235
</details>

<details>
<summary><strong>VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit</strong> - Junda Lin, Zhaomeng Zhou, Zhi Zheng, Shuochen Liu, Tong Xu, Yong Chen, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2601.05755)</summary>

**Abstract:** LLM agents operating in open environments face escalating risks from indirect prompt injection, particularly within the tool stream where manipulated metadata and runtime feedback hijack execution flow. Existing defenses encounter a critical dilemma as advanced models prioritize injected rules due to strict alignment while static protection mechanisms sever the feedback loop required for adaptive reasoning. To reconcile this conflict, we propose \textbf{VIGIL}, a framework that shifts the paradigm from restrictive isolation to a verify-before-commit protocol. By facilitating speculative hypothesis generation and enforcing safety through intent-grounded verification, \textbf{VIGIL} preserves reasoning flexibility while ensuring robust control. We further introduce \textbf{SIREN}, a benchmark comprising 959 tool stream injection cases designed to simulate pervasive threats characterized by dynamic dependencies. Extensive experiments demonstrate that \textbf{VIGIL} outperforms state-of-the-art dynamic defenses by reducing the attack success rate by over 22\% while more than doubling the utility under attack compared to static baselines, thereby achieving an optimal balance between security and utility.

**arXiv ID:** 2601.05755
</details>

<details>
<summary><strong>Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning</strong> - Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z. Pan, Hinrich Schütze, Volker Tresp, Yunpu Ma - [[pdf]](https://arxiv.org/pdf/2508.19828)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of NLP tasks, but they remain fundamentally stateless, constrained by limited context windows that hinder long-horizon reasoning. Recent efforts to address this limitation often augment LLMs with an external memory bank, yet most existing pipelines are static and heuristic-driven, lacking a learned mechanism for deciding what to store, update, or retrieve. We present Memory-R1, a reinforcement learning (RL) framework that equips LLMs with the ability to actively manage and utilize external memory through two specialized agents: a Memory Manager that learns structured operations, including ADD, UPDATE, DELETE, and NOOP; and an Answer Agent that pre-selects and reasons over relevant entries. Both agents are fine-tuned with outcome-driven RL (PPO and GRPO), enabling adaptive memory management with minimal supervision. With only 152 training QA pairs, Memory-R1 outperforms strong baselines and generalizes across diverse question types, three benchmarks (LoCoMo, MSC, LongMemEval), and multiple model scales (3B-14B).

**arXiv ID:** 2508.19828
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (10 papers)</h2></summary>

<details>
<summary><strong>Coordinated Pandemic Control with Large Language Model Agents as Policymaking Assistants</strong> - Ziyi Shi, Xusen Guo, Hongliang Lu, Mingxing Peng, Haotian Wang, Zheng Zhu, Zhenning Li, Yuxuan Liang, Xinhu Zheng, Hai Yang - [[pdf]](https://arxiv.org/pdf/2601.09264)</summary>

**Abstract:** Effective pandemic control requires timely and coordinated policymaking across administrative regions that are intrinsically interdependent. However, human-driven responses are often fragmented and reactive, with policies formulated in isolation and adjusted only after outbreaks escalate, undermining proactive intervention and global pandemic mitigation. To address this challenge, here we propose a large language model (LLM) multi-agent policymaking framework that supports coordinated and proactive pandemic control across regions. Within our framework, each administrative region is assigned an LLM agent as an AI policymaking assistant. The agent reasons over region-specific epidemiological dynamics while communicating with other agents to account for cross-regional interdependencies. By integrating real-world data, a pandemic evolution simulator, and structured inter-agent communication, our framework enables agents to jointly explore counterfactual intervention scenarios and synthesize coordinated policy decisions through a closed-loop simulation process. We validate the proposed framework using state-level COVID-19 data from the United States between April and December 2020, together with real-world mobility records and observed policy interventions. Compared with real-world pandemic outcomes, our approach reduces cumulative infections and deaths by up to 63.7% and 40.1%, respectively, at the individual state level, and by 39.0% and 27.0%, respectively, when aggregated across states. These results demonstrate that LLM multi-agent systems can enable more effective pandemic control with coordinated policymaking...

**arXiv ID:** 2601.09264
</details>

<details>
<summary><strong>Collaborative Multi-Agent Test-Time Reinforcement Learning for Reasoning</strong> - Zhiyuan Hu, Yunhai Hu, Juncheng Liu, Shuyue Stella Li, Yucheng Wang, Zhen Xu, See-Kiong Ng, Anh Tuan Luu, Xinxing Xu, Bryan Hooi, Cynthia Breazeal, Hae Won Park - [[pdf]](https://arxiv.org/pdf/2601.09667)</summary>

**Abstract:** Multi-agent systems have evolved into practical LLM-driven collaborators for many applications, gaining robustness from diversity and cross-checking. However, multi-agent RL (MARL) training is resource-intensive and unstable: co-adapting teammates induce non-stationarity, and rewards are often sparse and high-variance. Therefore, we introduce \textbf{Multi-Agent Test-Time Reinforcement Learning (MATTRL)}, a framework that injects structured textual experience into multi-agent deliberation at inference time. MATTRL forms a multi-expert team of specialists for multi-turn discussions, retrieves and integrates test-time experiences, and reaches consensus for final decision-making. We also study credit assignment for constructing a turn-level experience pool, then reinjecting it into the dialogue. Across challenging benchmarks in medicine, math, and education, MATTRL improves accuracy by an average of 3.67\% over a multi-agent baseline, and by 8.67\% over comparable single-agent baselines. Ablation studies examine different credit-assignment schemes and provide a detailed comparison of how they affect training outcomes. MATTRL offers a stable, effective and efficient path to distribution-shift-robust multi-agent reasoning without tuning.

**arXiv ID:** 2601.09667
</details>

<details>
<summary><strong>Improving Implicit Hate Speech Detection via a Community-Driven Multi-Agent Framework</strong> - Ewelina Gajewska, Katarzyna Budzynska, Jarosław A Chudziak - [[pdf]](https://arxiv.org/pdf/2601.09342)</summary>

**Abstract:** This work proposes a contextualised detection framework for implicitly hateful speech, implemented as a multi-agent system comprising a central Moderator Agent and dynamically constructed Community Agents representing specific demographic groups. Our approach explicitly integrates socio-cultural context from publicly available knowledge sources, enabling identity-aware moderation that surpasses state-of-the-art prompting methods (zero-shot prompting, few-shot prompting, chain-of-thought prompting) and alternative approaches on a challenging ToxiGen dataset. We enhance the technical rigour of performance evaluation by incorporating balanced accuracy as a central metric of classification fairness that accounts for the trade-off between true positive and true negative rates. We demonstrate that our community-driven consultative framework significantly improves both classification accuracy and fairness across all target groups.

**arXiv ID:** 2601.09342
</details>

<details>
<summary><strong>When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail</strong> - Xiaoxiao Li - [[pdf]](https://arxiv.org/pdf/2601.04748)</summary>

**Abstract:** Multi-agent AI systems have proven effective for complex reasoning. These systems are compounded by specialized agents, which collaborate through explicit communication, but incur substantial computational overhead. A natural question arises: can we achieve similar modularity benefits with a single agent that selects from a library of skills? We explore this question by viewing skills as internalized agent behaviors. From this perspective, a multi-agent system can be compiled into an equivalent single-agent system, trading inter-agent communication for skill selection. Our preliminary experiments suggest this approach can substantially reduce token usage and latency while maintaining competitive accuracy on reasoning benchmarks. However, this efficiency raises a deeper question that has received little attention: how does skill selection scale as libraries grow?
Drawing on principles from cognitive science, we propose that LLM skill selection exhibits bounded capacity analogous to human decision-making. We investigate the scaling behavior of skill selection and observe a striking pattern. Rather than degrading gradually, selection accuracy remains stable up to a critical library size, then drops sharply, indicating a phase transition reminiscent of capacity limits in human cognition. Furthermore, we find evidence that semantic confusability among similar skills, rather than library size alone, plays a central role in this degradation. This perspective suggests that hierarchical organization, which has long helped humans manage complex choices, may similarly benefit AI systems. Our initial results with hierarchical routing support this hypothesis. This work opens new questions about the fundamental limits of semantic-based skill selection in LLMs and offers a cognitive-grounded framework and practical guidelines for designing scalable skill-based agents.

**arXiv ID:** 2601.04748
</details>

<details>
<summary><strong>Advancing ESG Intelligence: An Expert-level Agent and Comprehensive Benchmark for Sustainable Finance</strong> - Yilei Zhao, Wentao Zhang, Lei Xiao, Yandan Zheng, Mengpu Liu, Wei Yang Bryan Lim - [[pdf]](https://arxiv.org/pdf/2601.08676)</summary>

**Abstract:** Environmental, social, and governance (ESG) criteria are essential for evaluating corporate sustainability and ethical performance. However, professional ESG analysis is hindered by data fragmentation across unstructured sources, and existing large language models (LLMs) often struggle with the complex, multi-step workflows required for rigorous auditing. To address these limitations, we introduce ESGAgent, a hierarchical multi-agent system empowered by a specialized toolset, including retrieval augmentation, web search and domain-specific functions, to generate in-depth ESG analysis. Complementing this agentic system, we present a comprehensive three-level benchmark derived from 310 corporate sustainability reports, designed to evaluate capabilities ranging from atomic common-sense questions to the generation of integrated, in-depth analysis. Empirical evaluations demonstrate that ESGAgent outperforms state-of-the-art closed-source LLMs with an average accuracy of 84.15% on atomic question-answering tasks, and excels in professional report generation by integrating rich charts and verifiable references. These findings confirm the diagnostic value of our benchmark, establishing it as a vital testbed for assessing general and advanced agentic capabilities in high-stakes vertical domains.

**arXiv ID:** 2601.08676
</details>

<details>
<summary><strong>Higher Satisfaction, Lower Cost: A Technical Report on How LLMs Revolutionize Meituan's Intelligent Interaction Systems</strong> - Xuxin Cheng, Ke Zeng, Zhiquan Cao, Linyi Dai, Wenxuan Gao, Fei Han, Ai Jian, Feng Hong, Wenxing Hu, Zihe Huang, Dejian Kong, Jia Leng, Zhuoyuan Liao, Pei Liu, Jiaye Lin, Xing Ma, Jingqing Ruan, Jiaxing Song, Xiaoyu Tan, Ruixuan Xiao, Wenhui Yu, Wenyu Zhan, Haoxing Zhang, Chao Zhou, Hao Zhou, Shaodong Zheng, Ruinian Chen, Siyuan Chen, Ziyang Chen, Yiwen Dong, Yaoyou Fan, Yangyi Fang, Yang Gan, Shiguang Guo, Qi He, Chaowen Hu, Binghui Li, Dailin Li, Xiangyu Li, Yan Li, Chengjian Liu, Xiangfeng Liu, Jiahui Lv, Qiao Ma, Jiang Pan, Cong Qin, Chenxing Sun, Wen Sun, Zhonghui Wang, Abudukelimu Wuerkaixi, Xin Yang, Fangyi Yuan, Yawen Zhu, Tianyi Zhai, Jie Zhang, Runlai Zhang, Yao Xu, Yiran Zhao, Yifan Wang, Xunliang Cai, Yangen Hu, Cao Liu, Lu Pan, Xiaoli Wang, Bo Xiao, Wenyuan Yao, Qianlin Zhou, Benchang Zhu - [[pdf]](https://arxiv.org/pdf/2510.13291)</summary>

**Abstract:** Enhancing customer experience is essential for business success, particularly as service demands grow in scale and complexity. Generative artificial intelligence and Large Language Models (LLMs) have empowered intelligent interaction systems to deliver efficient, personalized, and 24/7 support. In practice, intelligent interaction systems encounter several challenges: (1) Constructing high-quality data for cold-start training is difficult, hindering self-evolution and raising labor costs. (2) Multi-turn dialogue performance remains suboptimal due to inadequate intent understanding, rule compliance, and solution extraction. (3) Frequent evolution of business rules affects system operability and transferability, constraining low-cost expansion and adaptability. (4) Reliance on a single LLM is insufficient in complex scenarios, where the absence of multi-agent frameworks and effective collaboration undermines process completeness and service quality. (5) The open-domain nature of multi-turn dialogues, lacking unified golden answers, hampers quantitative evaluation and continuous optimization. To address these challenges, we introduce WOWService, an intelligent interaction system tailored for industrial applications. With the integration of LLMs and multi-agent architectures, WOWService enables autonomous task management and collaborative problem-solving. Specifically, WOWService focuses on core modules including data construction, general capability enhancement, business scenario adaptation, multi-agent coordination, and automated evaluation. Currently, WOWService is deployed on the Meituan App, achieving significant gains in key metrics, e.g., User Satisfaction Metric 1 (USM 1) -27.53% and User Satisfaction Metric 2 (USM 2) +25.51%, demonstrating its effectiveness in capturing user needs and advancing personalized service.

**arXiv ID:** 2510.13291
</details>

<details>
<summary><strong>MACRO-LLM: LLM-Empowered Multi-Agent Collaborative Reasoning under Spatiotemporal Partial Observability</strong> - Handi Chen, Running Zhao, Xiuzhe Wu, Edith C.H. Ngai - [[pdf]](https://arxiv.org/pdf/2601.09295)</summary>

**Abstract:** Large Language Model (LLM) agents deployed in complex real-world scenarios typically operate as spatially distributed entities. However, this physical dispersion constrains agents to limited local perception and finite temporal horizons. We characterize this bottleneck as spatiotemporal partial observability. Given such fragmented awareness, distributed agents struggle to coordinate efficiently. To bridge this gap, we introduce MACRO-LLM, LLM-empowered multi-agent collaborative reasoning under spatiotemporal partial observability. The architecture addresses spatiotemporal constraints via three modules: (1) the CoProposer mitigates temporal uncertainty by verifying candidate actions via predictive rollouts; (2) the Negotiator overcomes spatial myopia by resolving conflicts through mean-field statistical aggregation; and (3) the Introspector ensures continuous adaptation by analyzing historical experience to refine strategies via semantic gradient descent. Extensive evaluations on two complex long-horizon tasks, cooperative adaptive cruise control and pandemic control, demonstrate that our framework effectively mitigates spatiotemporal partial observability through spatial and temporal strategies, enabling robust coordination.

**arXiv ID:** 2601.09295
</details>

<details>
<summary><strong>SC-MAS: Constructing Cost-Efficient Multi-Agent Systems with Edge-Level Heterogeneous Collaboration</strong> - Di Zhao, Longhui Ma, Siwei Wang, Miao Wang, Yi Kong - [[pdf]](https://arxiv.org/pdf/2601.09434)</summary>

**Abstract:** Large Language Model (LLM)-based Multi-Agent Systems (MAS) enhance complex problem solving through multi-agent collaboration, but often incur substantially higher costs than single-agent systems. Recent MAS routing methods aim to balance performance and overhead by dynamically selecting agent roles and language models. However, these approaches typically rely on a homogeneous collaboration mode, where all agents follow the same interaction pattern, limiting collaboration flexibility across different roles. Motivated by Social Capital Theory, which emphasizes that different roles benefit from distinct forms of collaboration, we propose SC-MAS, a framework for constructing heterogeneous and cost-efficient multi-agent systems. SC-MAS models MAS as directed graphs, where edges explicitly represent pairwise collaboration strategies, allowing different agent pairs to interact through tailored communication patterns. Given an input query, a unified controller progressively constructs an executable MAS by selecting task-relevant agent roles, assigning edge-level collaboration strategies, and allocating appropriate LLM backbones to individual agents. Experiments on multiple benchmarks demonstrate the effectiveness of SC-MAS. In particular, SC-MAS improves accuracy by 3.35% on MMLU while reducing inference cost by 15.38%, and achieves a 3.53% accuracy gain with a 12.13% cost reduction on MBPP. These results validate the feasibility of SC-MAS and highlight the effectiveness of heterogeneous collaboration in multi-agent systems.

**arXiv ID:** 2601.09434
</details>

<details>
<summary><strong>SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams</strong> - Chenglong Wang, Canjia Li, Xingzhao Zhu, Yifu Huo, Huiyu Wang, Weixiong Lin, Yun Yang, Qiaozhi He, Tianhua Zhou, Xiaojia Chang, Jingbo Zhu, Tong Xiao - [[pdf]](https://arxiv.org/pdf/2601.09515)</summary>

**Abstract:** Due to the dynamically evolving nature of real-world query streams, relevance models struggle to generalize to practical search scenarios. A sophisticated solution is self-evolution techniques. However, in large-scale industrial settings with massive query streams, this technique faces two challenges: (1) informative samples are often sparse and difficult to identify, and (2) pseudo-labels generated by the current model could be unreliable. To address these challenges, in this work, we propose a Self-Evolving Relevance Model approach (SERM), which comprises two complementary multi-agent modules: a multi-agent sample miner, designed to detect distributional shifts and identify informative training samples, and a multi-agent relevance annotator, which provides reliable labels through a two-level agreement framework. We evaluate SERM in a large-scale industrial setting, which serves billions of user requests daily. Experimental results demonstrate that SERM can achieve significant performance gains through iterative self-evolution, as validated by extensive offline multilingual evaluations and online testing.

**arXiv ID:** 2601.09515
</details>

<details>
<summary><strong>World Craft: Agentic Framework to Create Visualizable Worlds via Text</strong> - Jianwen Sun, Yukang Feng, Kaining Ying, Chuanhao Li, Zizhen Li, Fanrui Zhang, Jiaxin Ai, Yifan Chang, Yu Dai, Yifei Huang, Kaipeng Zhang - [[pdf]](https://arxiv.org/pdf/2601.09150)</summary>

**Abstract:** Large Language Models (LLMs) motivate generative agent simulation (e.g., AI Town) to create a ``dynamic world'', holding immense value across entertainment and research. However, for non-experts, especially those without programming skills, it isn't easy to customize a visualizable environment by themselves. In this paper, we introduce World Craft, an agentic world creation framework to create an executable and visualizable AI Town via user textual descriptions. It consists of two main modules, World Scaffold and World Guild. World Scaffold is a structured and concise standardization to develop interactive game scenes, serving as an efficient scaffolding for LLMs to customize an executable AI Town-like environment. World Guild is a multi-agent framework to progressively analyze users' intents from rough descriptions, and synthesizes required structured contents (\eg environment layout and assets) for World Scaffold . Moreover, we construct a high-quality error-correction dataset via reverse engineering to enhance spatial knowledge and improve the stability and controllability of layout generation, while reporting multi-dimensional evaluation metrics for further analysis. Extensive experiments demonstrate that our framework significantly outperforms existing commercial code agents (Cursor and Antigravity) and LLMs (Qwen3 and Gemini-3-Pro). in scene construction and narrative intent conveyance, providing a scalable solution for the democratization of environment creation.

**arXiv ID:** 2601.09150
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>Advancing AI Negotiations: A Large-Scale Autonomous Negotiation Competition</strong> - Michelle Vaccaro, Michael Caosun, Harang Ju, Sinan Aral, Jared R. Curhan - [[pdf]](https://arxiv.org/pdf/2503.06416)</summary>

**Abstract:** We conducted an International AI Negotiation Competition in which participants designed and refined prompts for AI negotiation agents. We then facilitated over 180,000 negotiations between these agents across multiple scenarios with diverse characteristics and objectives. Our findings revealed that principles from human negotiation theory remain crucial even in AI-AI contexts. Surprisingly, warmth -- a traditionally human relationship-building trait -- was consistently associated with superior outcomes across all key performance metrics. Dominant agents, meanwhile, were especially effective at claiming value. Our analysis also revealed unique dynamics in AI-AI negotiations not fully explained by existing theory, including AI-specific technical strategies like chain-of-thought reasoning and prompt injection. When we applied natural language processing (NLP) methods to the full transcripts of all negotiations, we found positivity, gratitude, and question-asking (associated with warmth) were strongly associated with reaching deals as well as objective and subjective value, whereas conversation lengths (associated with dominance) were strongly associated with impasses. The results suggest the need to establish a new theory of AI negotiation, which integrates classic negotiation theory with AI-specific negotiation theories to better understand autonomous negotiations and optimize agent performance.

**arXiv ID:** 2503.06416
</details>

<details>
<summary><strong>Beyond One-Size-Fits-All: A Survey of Personalized Affective Computing in Human-Agent Interaction</strong> - Jialin Li, Maha Elgarf, Alia Waleed, Hanan Salam - [[pdf]](https://arxiv.org/pdf/2304.00377)</summary>

**Abstract:** In personalized machine learning, the aim of personalization is to train a model that caters to a specific individual or group of individuals by optimizing one or more performance metrics and adhering to specific constraints. In this paper, we discuss the need for personalization in affective computing and present the first survey of existing approaches for personalization in affective computing. Our review spans training techniques and objectives towards the personalization of affective computing models across various interaction modes and contexts. We develop a taxonomy that clusters existing approaches into Data-level and Model-level approaches. Across the Data-Level and Model-Level broad categories, we group existing approaches into seven sub-categories: (1) User-Specific Models, (2) Group-Specific Models, (3) Weighting-Based Approaches, (4) Feature Augmentation, (5) Generative-Based Models which fall into the Data-Level approaches, (6) Fine-Tuning Approaches, and (7) Multitask Learning Approaches falling under the model-level approaches. We provide a problem formulation for personalized affective computing, and to each of the identified sub-categories. Additionally, we provide a statistical analysis of the surveyed literature, analyzing the prevalence of different affective computing tasks, interaction modes (i.e. Human-Computer Interaction (HCI), Human-Human interaction (HHI), Human-Robot Interaction (HRI)), interaction contexts (e.g. educative, social, gaming, etc.), and the level of personalization among the surveyed works. Based on our analysis, we provide a road-map for researchers interested in exploring this direction.

**arXiv ID:** 2304.00377
</details>

<details>
<summary><strong>QueryIPI: Query-agnostic Indirect Prompt Injection on Coding Agents</strong> - Yuchong Xie, Zesen Liu, Mingyu Luo, Zhixiang Zhang, Kaikai Zhang, Yuanyuan Yuan, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She - [[pdf]](https://arxiv.org/pdf/2510.23675)</summary>

**Abstract:** Modern coding agents integrated into IDEs orchestrate powerful tools and high-privilege system access, creating a high-stakes attack surface. Prior work on Indirect Prompt Injection (IPI) is mainly query-specific, requiring particular user queries as triggers and leading to poor generalizability. We propose query-agnostic IPI, a new attack paradigm that reliably executes malicious payloads under arbitrary user queries. Our key insight is that malicious payloads should leverage the invariant prompt context (i.e., system prompt and tool descriptions) rather than variant user queries. We present QueryIPI, an automated framework that uses tool descriptions as optimizable payloads and refines them via iterative, prompt-based blackbox optimization. QueryIPI leverages system invariants for initial seed generation aligned with agent conventions, and iterative reflection to resolve instruction-following failures and safety refusals. Experiments on five simulated agents show that QueryIPI achieves up to 87% success rate, outperforming the best baseline (50%). Crucially, generated malicious descriptions transfer to real-world coding agents, highlighting a practical security risk.

**arXiv ID:** 2510.23675
</details>

<details>
<summary><strong>Autonomous Robotic Bone Micro-Milling System with Automatic Calibration and 3D Surface Fitting</strong> - Enduo Zhao, Xiaofeng Lin, Yifan Wang, Kanako Harada - [[pdf]](https://arxiv.org/pdf/2503.04038)</summary>

**Abstract:** Automating bone micro-milling using a robotic system presents challenges due to the uncertainties in both the external and internal features of bone tissue. For example, during mouse cranial window creation, a circular path with a radius of 2 to 4 mm needs to be milled on the mouse skull using a microdrill. The uneven surface and non-uniform thickness of the mouse skull make it difficult to fully automate this process, requiring the system to possess advanced perceptual and adaptive capabilities. In this study, we address this challenge by integrating a Microscopic Stereo Camera System (MSCS) into the robotic bone micro-milling system and proposing a novel online pre-measurement pipeline for the target surface. Starting from uncalibrated cameras, the pipeline enables automatic calibration and 3D surface fitting through a convolutional neural network (CNN)-based keypoint detection. Combined with the existing feedback-based system, we develop the world's first autonomous robotic bone micro-milling system capable of rapidly, in real-time perceiving and adapting to surface unevenness and non-uniform thickness, thereby enabling an end-to-end autonomous cranial window creation workflow without human assistance. Validation experiments on euthanized mice demonstrate that the improved system achieves a success rate of 85.7 % and an average milling time of 2.1 minutes, showing not only significant performance improvements over the previous system but also exceptional accuracy, speed, and stability compared to human operators.

**arXiv ID:** 2503.04038
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (24 papers)</h2></summary>

<details>
<summary><strong>ART: Action-based Reasoning Task Benchmarking for Medical AI Agents</strong> - Ananya Mantravadi, Shivali Dalmia, Abhishek Mukherji - [[pdf]](https://arxiv.org/pdf/2601.08988)</summary>

**Abstract:** Reliable clinical decision support requires medical AI agents capable of safe, multi-step reasoning over structured electronic health records (EHRs). While large language models (LLMs) show promise in healthcare, existing benchmarks inadequately assess performance on action-based tasks involving threshold evaluation, temporal aggregation, and conditional logic. We introduce ART, an Action-based Reasoning clinical Task benchmark for medical AI agents, which mines real-world EHR data to create challenging tasks targeting known reasoning weaknesses. Through analysis of existing benchmarks, we identify three dominant error categories: retrieval failures, aggregation errors, and conditional logic misjudgments. Our four-stage pipeline -- scenario identification, task generation, quality audit, and evaluation -- produces diverse, clinically validated tasks grounded in real patient data. Evaluating GPT-4o-mini and Claude 3.5 Sonnet on 600 tasks shows near-perfect retrieval after prompt refinement, but substantial gaps in aggregation (28--64%) and threshold reasoning (32--38%). By exposing failure modes in action-oriented EHR reasoning, ART advances toward more reliable clinical agents, an essential step for AI systems that reduce cognitive load and administrative burden, supporting workforce capacity in high-demand care settings

**arXiv ID:** 2601.08988
</details>

<details>
<summary><strong>Policy-Based Reinforcement Learning with Action Masking for Dynamic Job Shop Scheduling under Uncertainty: Handling Random Arrivals and Machine Failures</strong> - Sofiene Lassoued, Stefan Lier, Andreas Schwung - [[pdf]](https://arxiv.org/pdf/2601.09293)</summary>

**Abstract:** We present a novel framework for solving Dynamic Job Shop Scheduling Problems under uncertainty, addressing the challenges introduced by stochastic job arrivals and unexpected machine breakdowns. Our approach follows a model-based paradigm, using Coloured Timed Petri Nets to represent the scheduling environment, and Maskable Proximal Policy Optimization to enable dynamic decision-making while restricting the agent to feasible actions at each decision point. To simulate realistic industrial conditions, dynamic job arrivals are modeled using a Gamma distribution, which captures complex temporal patterns such as bursts, clustering, and fluctuating workloads. Machine failures are modeled using a Weibull distribution to represent age-dependent degradation and wear-out dynamics. These stochastic models enable the framework to reflect real-world manufacturing scenarios better. In addition, we study two action-masking strategies: a non-gradient approach that overrides the probabilities of invalid actions, and a gradient-based approach that assigns negative gradients to invalid actions within the policy network. We conduct extensive experiments on dynamic JSSP benchmarks, demonstrating that our method consistently outperforms traditional heuristic and rule-based approaches in terms of makespan minimization. The results highlight the strength of combining interpretable Petri-net-based models with adaptive reinforcement learning policies, yielding a resilient, scalable, and explainable framework for real-time scheduling in dynamic and uncertain manufacturing environments.

**arXiv ID:** 2601.09293
</details>

<details>
<summary><strong>Monte-Carlo Tree Search with Neural Network Guidance for Lane-Free Autonomous Driving</strong> - Ioannis Peridis, Dimitrios Troullinos, Georgios Chalkiadakis, Pantelis Giankoulidis, Ioannis Papamichail, Markos Papageorgiou - [[pdf]](https://arxiv.org/pdf/2601.09353)</summary>

**Abstract:** Lane-free traffic environments allow vehicles to better harness the lateral capacity of the road without being restricted to lane-keeping, thereby increasing the traffic flow rates. As such, we have a distinct and more challenging setting for autonomous driving. In this work, we consider a Monte-Carlo Tree Search (MCTS) planning approach for single-agent autonomous driving in lane-free traffic, where the associated Markov Decision Process we formulate is influenced from existing approaches tied to reinforcement learning frameworks. In addition, MCTS is equipped with a pre-trained neural network (NN) that guides the selection phase. This procedure incorporates the predictive capabilities of NNs for a more informed tree search process under computational constraints. In our experimental evaluation, we consider metrics that address both safety (through collision rates) and efficacy (through measured speed). Then, we examine: (a) the influence of isotropic state information for vehicles in a lane-free environment, resulting in nudging behaviour--vehicles' policy reacts due to the presence of faster tailing ones, (b) the acceleration of performance for the NN-guided variant of MCTS, and (c) the trade-off between computational resources and solution quality.

**arXiv ID:** 2601.09353
</details>

<details>
<summary><strong>Reading or Reasoning? Format Decoupled Reinforcement Learning for Document OCR</strong> - Yufeng Zhong, Lei Chen, Zhixiong Zeng, Xuanle Zhao, Deyang Jiang, Liming Zheng, Jing Huang, Haibo Qiu, Peng Shi, Siqi Yang, Lin Ma - [[pdf]](https://arxiv.org/pdf/2601.08834)</summary>

**Abstract:** Reading text from images or scanned documents via OCR models has been a longstanding focus of researchers. Intuitively, text reading is perceived as a straightforward perceptual task, and existing work primarily focuses on constructing enriched data engineering to enhance SFT capabilities. In this work, we observe that even advanced OCR models exhibit significantly higher entropy in formatted text (\emph{e.g.}, formula, table, etc.) compared to plain text, often by an order of magnitude. These statistical patterns reveal that advanced OCR models struggle with high output uncertainty when dealing with format sensitive document, suggesting that reasoning over diverse reading pathways may improve OCR performance. To address this, we propose format decoupled reinforcement learning (FD-RL), which leverages high-entropy patterns for targeted optimization. Our approach employs entropy-based data filtration strategy to identify format-intensive instances, and adopt format decoupled rewards tailored to different format types, enabling format-level validation rather than token-level memorization. FD-RL achieves an average score of 90.41 on OmniDocBench, setting a new record for end-to-end models on this highly popular benchmark. More importantly, we conduct comprehensive ablation studies over data, training, filtering, and rewarding strategies, thoroughly validating their effectiveness.

**arXiv ID:** 2601.08834
</details>

<details>
<summary><strong>Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models</strong> - Youwei Liu, Jian Wang, Hanlin Wang, Beichen Guo, Wenjie Li - [[pdf]](https://arxiv.org/pdf/2601.08955)</summary>

**Abstract:** Recent advances in world models have shown promise for modeling future dynamics of environmental states, enabling agents to reason and act without accessing real environments. Current methods mainly perform single-step or fixed-horizon rollouts, leaving their potential for complex task planning under-exploited. We propose Imagine-then-Plan (\texttt{ITP}), a unified framework for agent learning via lookahead imagination, where an agent's policy model interacts with the learned world model, yielding multi-step ``imagined'' trajectories. Since the imagination horizon may vary by tasks and stages, we introduce a novel adaptive lookahead mechanism by trading off the ultimate goal and task progress. The resulting imagined trajectories provide rich signals about future consequences, such as achieved progress and potential conflicts, which are fused with current observations, formulating a partially \textit{observable} and \textit{imaginable} Markov decision process to guide policy learning. We instantiate \texttt{ITP} with both training-free and reinforcement-trained variants. Extensive experiments across representative agent benchmarks demonstrate that \texttt{ITP} significantly outperforms competitive baselines. Further analyses validate that our adaptive lookahead largely enhances agents' reasoning capability, providing valuable insights into addressing broader, complex tasks.

**arXiv ID:** 2601.08955
</details>

<details>
<summary><strong>DPWriter: Reinforcement Learning with Diverse Planning Branching for Creative Writing</strong> - Qian Cao, Yahui Liu, Wei Bi, Yi Zhao, Ruihua Song, Xiting Wang, Ruiming Tang, Guorui Zhou, Han Li - [[pdf]](https://arxiv.org/pdf/2601.09609)</summary>

**Abstract:** Reinforcement learning (RL)-based enhancement of large language models (LLMs) often leads to reduced output diversity, undermining their utility in open-ended tasks like creative writing. Current methods lack explicit mechanisms for guiding diverse exploration and instead prioritize optimization efficiency and performance over diversity. This paper proposes an RL framework structured around a semi-structured long Chain-of-Thought (CoT), in which the generation process is decomposed into explicitly planned intermediate steps. We introduce a Diverse Planning Branching method that strategically introduces divergence at the planning phase based on diversity variation, alongside a group-aware diversity reward to encourage distinct trajectories. Experimental results on creative writing benchmarks demonstrate that our approach significantly improves output diversity without compromising generation quality, consistently outperforming existing baselines.

**arXiv ID:** 2601.09609
</details>

<details>
<summary><strong>CogRail: Benchmarking VLMs in Cognitive Intrusion Perception for Intelligent Railway Transportation Systems</strong> - Yonglin Tian, Qiyao Zhang, Wei Xu, Yutong Wang, Yihao Wu, Xinyi Li, Xingyuan Dai, Hui Zhang, Zhiyong Cui, Baoqing Guo, Zujun Yu, Yisheng Lv - [[pdf]](https://arxiv.org/pdf/2601.09613)</summary>

**Abstract:** Accurate and early perception of potential intrusion targets is essential for ensuring the safety of railway transportation systems. However, most existing systems focus narrowly on object classification within fixed visual scopes and apply rule-based heuristics to determine intrusion status, often overlooking targets that pose latent intrusion risks. Anticipating such risks requires the cognition of spatial context and temporal dynamics for the object of interest (OOI), which presents challenges for conventional visual models. To facilitate deep intrusion perception, we introduce a novel benchmark, CogRail, which integrates curated open-source datasets with cognitively driven question-answer annotations to support spatio-temporal reasoning and prediction. Building upon this benchmark, we conduct a systematic evaluation of state-of-the-art visual-language models (VLMs) using multimodal prompts to identify their strengths and limitations in this domain. Furthermore, we fine-tune VLMs for better performance and propose a joint fine-tuning framework that integrates three core tasks, position perception, movement prediction, and threat analysis, facilitating effective adaptation of general-purpose foundation models into specialized models tailored for cognitive intrusion perception. Extensive experiments reveal that current large-scale multimodal models struggle with the complex spatial-temporal reasoning required by the cognitive intrusion perception task, underscoring the limitations of existing foundation models in this safety-critical domain. In contrast, our proposed joint fine-tuning framework significantly enhances model performance by enabling targeted adaptation to domain-specific reasoning demands, highlighting the advantages of structured multi-task learning in improving both accuracy and interpretability. Code will be available at this https URL.

**arXiv ID:** 2601.09613
</details>

<details>
<summary><strong>A Curriculum Learning Approach to Reinforcement Learning: Leveraging RAG for Multimodal Question Answering</strong> - Chenliang Zhang, Lin Wang, Yuanyuan Lu, Yusheng Qi, Kexin Wang, Peixu Hou, Wenshi Chen - [[pdf]](https://arxiv.org/pdf/2508.10337)</summary>

**Abstract:** This paper describes the solutions of the Dianping-Trust-Safety team for the META CRAG-MM challenge. The challenge requires building a comprehensive retrieval-augmented generation system capable for multi-modal multi-turn question answering. The competition consists of three tasks: (1) answering questions using structured data retrieved from an image-based mock knowledge graph, (2) synthesizing information from both knowledge graphs and web search results, and (3) handling multi-turn conversations that require context understanding and information aggregation from multiple sources. For Task 1, our solution is based on the vision large language model, enhanced by supervised fine-tuning with knowledge distilled from GPT-4.1. We further applied curriculum learning strategies to guide reinforcement learning, resulting in improved answer accuracy and reduced hallucination. For Task 2 and Task 3, we additionally leveraged web search APIs to incorporate external knowledge, enabling the system to better handle complex queries and multi-turn conversations. Our approach achieved 1st place in Task 1 with a significant lead of 52.38%, and 3rd place in Task 3, demonstrating the effectiveness of the integration of curriculum learning with reinforcement learning in our training pipeline.

**arXiv ID:** 2508.10337
</details>

<details>
<summary><strong>Graph Neural Networks, Deep Reinforcement Learning and Probabilistic Topic Modeling for Strategic Multiagent Settings</strong> - Georgios Chalkiadakis, Charilaos Akasiadis, Gerasimos Koresis, Stergios Plataniotis, Leonidas Bakopoulos - [[pdf]](https://arxiv.org/pdf/2511.10501)</summary>

**Abstract:** This paper provides a comprehensive review of mainly GNN, DRL, and PTM methods with a focus on their potential incorporation in strategic multiagent settings. We draw interest in (i) ML methods currently utilized for uncovering unknown model structures adaptable to the task of strategic opponent modeling, and (ii) the integration of these methods with Game Theoretic concepts that avoid relying on assumptions often invalid in real-world scenarios, such as the Common Prior Assumption (CPA) and the Self-Interest Hypothesis (SIH). We analyze the ability to handle uncertainty and heterogeneity, two characteristics that are very common in real-world application cases, as well as scalability. As a potential answer to effectively modeling relationships and interactions in multiagent settings, we champion the use of GNN. Such approaches are designed to operate upon graph-structured data, and have been shown to be a very powerful tool for performing tasks such as node classification and link prediction. Next, we review the domain of RL, and in particular that of multiagent deep reinforcement learning. Single-agent deep RL has been widely used for decision making in demanding game settings. Its application in multiagent settings though is hindered due to, e.g., varying relationships between agents, and non-stationarity of the environment. We describe existing relevant game theoretic solution concepts, and consider properties such as fairness and stability. Our review comes complete with a note on the literature that utilizes probabilistic topic modeling (PTM) in domains other than that of document analysis and classification. Finally, we identify certain open challenges -- specifically, the need to (i) fit non-stationary environments, (ii) balance the degrees of stability and adaptation, (iii) tackle uncertainty and heterogeneity, (iv) guarantee scalability and solution tractability.

**arXiv ID:** 2511.10501
</details>

<details>
<summary><strong>LSRIF: Logic-Structured Reinforcement Learning for Instruction Following</strong> - Qingyu Ren, Qianyu He, Jingwen Chang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Han Xia, Zeye Sun, Fei Yu - [[pdf]](https://arxiv.org/pdf/2601.06431)</summary>

**Abstract:** Instruction-following is critical for large language models, but real-world instructions often contain logical structures such as sequential dependencies and conditional branching. Existing methods typically construct datasets with parallel constraints and optimize average rewards, ignoring logical dependencies and yielding noisy signals. We propose a logic-structured training framework LSRIF that explicitly models instruction logic. We first construct a dataset LSRInstruct with constraint structures such as parallel, sequential, and conditional types, and then design structure-aware rewarding method LSRIF including average aggregation for parallel structures, failure-penalty propagation for sequential structures, and selective rewards for conditional branches. Experiments show LSRIF brings significant improvements in instruction-following (in-domain and out-of-domain) and general reasoning. Analysis reveals that learning with explicit logic structures brings parameter updates in attention layers and sharpens token-level attention to constraints and logical operators.

**arXiv ID:** 2601.06431
</details>

<details>
<summary><strong>Reinforcement Learning with Exogenous States and Rewards</strong> - George Trimponias, Thomas G. Dietterich - [[pdf]](https://arxiv.org/pdf/2303.12957)</summary>

**Abstract:** Exogenous state variables and rewards can slow reinforcement learning by injecting uncontrolled variation into the reward signal. This paper formalizes exogenous state variables and rewards and shows that if the reward function decomposes additively into endogenous and exogenous components, the MDP can be decomposed into an exogenous Markov Reward Process (based on the exogenous reward) and an endogenous Markov Decision Process (optimizing the endogenous reward). Any optimal policy for the endogenous MDP is also an optimal policy for the original MDP, but because the endogenous reward typically has reduced variance, the endogenous MDP is easier to solve. We study settings where the decomposition of the state space into exogenous and endogenous state spaces is not given but must be discovered. The paper introduces and proves correctness of algorithms for discovering the exogenous and endogenous subspaces of the state space when they are mixed through linear combination. These algorithms can be applied during reinforcement learning to discover the exogenous subspace, remove the exogenous reward, and focus reinforcement learning on the endogenous MDP. Experiments on a variety of challenging synthetic MDPs show that these methods, applied online, discover large exogenous state spaces and produce substantial speedups in reinforcement learning.

**arXiv ID:** 2303.12957
</details>

<details>
<summary><strong>Causality-enhanced Decision-Making for Autonomous Mobile Robots in Dynamic Environments</strong> - Luca Castri, Gloria Beraldo, Nicola Bellotto - [[pdf]](https://arxiv.org/pdf/2504.11901)</summary>

**Abstract:** The growing integration of robots in shared environments - such as warehouses, shopping centres, and hospitals - demands a deep understanding of the underlying dynamics and human behaviours, including how, when, and where individuals engage in various activities and interactions. This knowledge goes beyond simple correlation studies and requires a more comprehensive causal analysis. By leveraging causal inference to model cause-and-effect relationships, we can better anticipate critical environmental factors and enable autonomous robots to plan and execute tasks more effectively. To this end, we propose a novel causality-based decision-making framework that reasons over a learned causal model to assist the robot in deciding when and how to complete a given task. In the examined use case - i.e., a warehouse shared with people - we exploit the causal model to estimate battery usage and human obstructions as factors influencing the robot's task execution. This reasoning framework supports the robot in making informed decisions about task timing and strategy. To achieve this, we developed also PeopleFlow, a new Gazebo-based simulator designed to model context-sensitive human-robot spatial interactions in shared workspaces. PeopleFlow features realistic human and robot trajectories influenced by contextual factors such as time, environment layout, and robot state, and can simulate a large number of agents. While the simulator is general-purpose, in this paper we focus on a warehouse-like environment as a case study, where we conduct an extensive evaluation benchmarking our causal approach against a non-causal baseline. Our findings demonstrate the efficacy of the proposed solutions, highlighting how causal reasoning enables autonomous robots to operate more efficiently and safely in dynamic environments shared with humans.

**arXiv ID:** 2504.11901
</details>

<details>
<summary><strong>Secure and Efficient Access Control for Computer-Use Agents via Context Space</strong> - Haochen Gong, Chenxiao Li, Rui Chang, Wenbo Shen - [[pdf]](https://arxiv.org/pdf/2509.22256)</summary>

**Abstract:** Large language model (LLM)-based computer-use agents represent a convergence of AI and OS capabilities, enabling natural language to control system- and application-level functions. However, due to LLMs' inherent uncertainty issues, granting agents control over computers poses significant security risks. When agent actions deviate from user intentions, they can cause irreversible consequences. Existing mitigation approaches, such as user confirmation and LLM-based dynamic action validation, still suffer from limitations in usability, security, and performance. To address these challenges, we propose CSAgent, a system-level, static policy-based access control framework for computer-use agents. To bridge the gap between static policy and dynamic context and user intent, CSAgent introduces intent- and context-aware policies, and provides an automated toolchain to assist developers in constructing and refining them. CSAgent enforces these policies through an optimized OS service, ensuring that agent actions can only be executed under specific user intents and contexts. CSAgent supports protecting agents that control computers through diverse interfaces, including API, CLI, and GUI. We implement and evaluate CSAgent, which successfully defends against all attacks in the benchmarks while introducing only 1.99% performance overhead and 5.42% utility decrease.

**arXiv ID:** 2509.22256
</details>

<details>
<summary><strong>Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design</strong> - Lianghong Chen, Dongkyu Eugene Kim, Mike Domaratzki, Pingzhao Hu - [[pdf]](https://arxiv.org/pdf/2510.21153)</summary>

**Abstract:** Designing de novo 3D molecules with desirable properties remains a fundamental challenge in drug discovery and molecular engineering. While diffusion models have demonstrated remarkable capabilities in generating high-quality 3D molecular structures, they often struggle to effectively control complex multi-objective constraints critical for real-world applications. In this study, we propose an uncertainty-aware Reinforcement Learning (RL) framework to guide the optimization of 3D molecular diffusion models toward multiple property objectives while enhancing the overall quality of the generated molecules. Our method leverages surrogate models with predictive uncertainty estimation to dynamically shape reward functions, facilitating balance across multiple optimization objectives. We comprehensively evaluate our framework across three benchmark datasets and multiple diffusion model architectures, consistently outperforming baselines for molecular quality and property optimization. Additionally, Molecular Dynamics (MD) simulations and ADMET profiling of top generated candidates indicate promising drug-like behavior and binding stability, comparable to known Epidermal Growth Factor Receptor (EGFR) inhibitors. Our results demonstrate the strong potential of RL-guided generative diffusion models for advancing automated molecular design.

**arXiv ID:** 2510.21153
</details>

<details>
<summary><strong>TeleMem: Building Long-Term and Multimodal Memory for Agentic AI</strong> - Chunliang Chen, Ming Guan, Xiao Lin, Jiaxu Li, Qiyi Wang, Xiangyu Chen, Jixiang Luo, Changzhi Sun, Dell Zhang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2601.06037)</summary>

**Abstract:** Large language models (LLMs) excel at many NLP tasks but struggle to sustain long-term interactions due to limited attention over extended dialogue histories. Retrieval-augmented generation (RAG) mitigates this issue but lacks reliable mechanisms for updating or refining stored memories, leading to schema-driven hallucinations, inefficient write operations, and minimal support for multimodal this http URL address these challenges, we propose TeleMem, a unified long-term and multimodal memory system that maintains coherent user profiles through narrative dynamic extraction, ensuring that only dialogue-grounded information is preserved. TeleMem further introduces a structured writing pipeline that batches, retrieves, clusters, and consolidates memory entries, substantially improving storage efficiency, reducing token usage, and accelerating memory operations. Additionally, a multimodal memory module combined with ReAct-style reasoning equips the system with a closed-loop observe, think, and act process that enables accurate understanding of complex video content in long-term contexts. Experimental results show that TeleMem surpasses the state-of-the-art Mem0 baseline with 19% higher accuracy, 43% fewer tokens, and a 2.1x speedup on the ZH-4O long-term role-play gaming benchmark.

**arXiv ID:** 2601.06037
</details>

<details>
<summary><strong>DiSCo: Making Absence Visible in Intelligent Summarization Interfaces</strong> - Eran Fainman, Hagit Ben Shoshan, Adir Solomon, Osnat Mokryn - [[pdf]](https://arxiv.org/pdf/2601.07229)</summary>

**Abstract:** Intelligent interfaces increasingly use large language models to summarize user-generated content, yet these summaries emphasize what is mentioned while overlooking what is missing. This presence bias can mislead users who rely on summaries to make decisions. We present Domain Informed Summarization through Contrast (DiSCo), an expectation-based computational approach that makes absences visible by comparing each entity's content with domain topical expectations captured in reference distributions of aspects typically discussed in comparable accommodations. This comparison identifies aspects that are either unusually emphasized or missing relative to domain norms and integrates them into the generated text. In a user study across three accommodation domains, namely ski, beach, and city center, DiSCo summaries were rated as more detailed and useful for decision making than baseline large language model summaries, although slightly harder to read. The findings show that modeling expectations reduces presence bias and improves both transparency and decision support in intelligent summarization interfaces.

**arXiv ID:** 2601.07229
</details>

<details>
<summary><strong>Recursive Knowledge Synthesis for Multi-LLM Systems: Stability Analysis and Tri-Agent Audit Framework</strong> - Toshiyuki Shigemura - [[pdf]](https://arxiv.org/pdf/2601.08839)</summary>

**Abstract:** This paper presents a tri-agent cross-validation framework for analyzing stability and explainability in multi-model large language systems. The architecture integrates three heterogeneous LLMs-used for semantic generation, analytical consistency checking, and transparency auditing-into a recursive interaction cycle. This design induces Recursive Knowledge Synthesis (RKS), where intermediate representations are continuously refined through mutually constraining transformations irreducible to single-model behavior. Across 47 controlled trials using public-access LLM deployments (October 2025), we evaluated system stability via four metrics: Reflex Reliability Score (RRS), Transparency Score (TS), Deviation Detection Rate (DDR), and Correction Success Rate (CSR). The system achieved mean RRS = 0.78+-0.06 and maintained TS >= 0.8 in about 68% of trials. Approximately 89% of trials converged, supporting the theoretical prediction that transparency auditing acts as a contraction operator within the composite validation mapping. The contributions are threefold: (1) a structured tri-agent framework for coordinated reasoning across heterogeneous LLMs, (2) a formal RKS model grounded in fixed-point theory, and (3) empirical evaluation of inter-model stability under realistic, non-API public-access conditions. These results provide initial empirical evidence that a safety-preserving, humansupervised multi-LLM architecture can achieve stable recursive knowledge synthesis in realistic, publicly deployed environments.

**arXiv ID:** 2601.08839
</details>

<details>
<summary><strong>UserLM-R1: Modeling Human Reasoning in User Language Models with Multi-Reward Reinforcement Learning</strong> - Feng Zhang, Shijia Li, Chunmao Zhang, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Jingwen Xu, Han Liu - [[pdf]](https://arxiv.org/pdf/2601.09215)</summary>

**Abstract:** User simulators serve as the critical interactive environment for agent post-training, and an ideal user simulator generalizes across domains and proactively engages in negotiation by challenging or bargaining. However, current methods exhibit two issues. They rely on static and context-unaware profiles, necessitating extensive manual redesign for new scenarios, thus limiting generalizability. Moreover, they neglect human strategic thinking, leading to vulnerability to agent manipulation. To address these issues, we propose UserLM-R1, a novel user language model with reasoning capability. Specifically, we first construct comprehensive user profiles with both static roles and dynamic scenario-specific goals for adaptation to diverse scenarios. Then, we propose a goal-driven decision-making policy to generate high-quality rationales before producing responses, and further refine the reasoning and improve strategic capabilities with supervised fine-tuning and multi-reward reinforcement learning. Extensive experimental results demonstrate that UserLM-R1 outperforms competitive baselines, particularly on the more challenging adversarial set.

**arXiv ID:** 2601.09215
</details>

<details>
<summary><strong>Agent Bain vs. Agent McKinsey: A New Text-to-SQL Benchmark for the Business Domain</strong> - Yue Li, Ran Tao, Derek Hommel, Yusuf Denizay Dönder, Sungyong Chang, David Mimno, Unso Eun Seo Jo - [[pdf]](https://arxiv.org/pdf/2510.07309)</summary>

**Abstract:** Text-to-SQL benchmarks have traditionally only tested simple data access as a translation task of natural language to SQL queries. But in reality, users tend to ask diverse questions that require more complex responses including data-driven predictions or recommendations. Using the business domain as a motivating example, we introduce CORGI, a new benchmark that expands text-to-SQL to reflect practical database queries encountered by end users. CORGI is composed of synthetic databases inspired by enterprises such as DoorDash, Airbnb, and Lululemon. It provides questions across four increasingly complicated categories of business queries: descriptive, explanatory, predictive, and recommendational. This challenge calls for causal reasoning, temporal forecasting, and strategic recommendation, reflecting multi-level and multi-step agentic intelligence. We find that LLM performance degrades on higher-level questions as question complexity increases. CORGI also introduces and encourages the text-to-SQL community to consider new automatic methods for evaluating open-ended, qualitative responses in data access tasks. Our experiments show that LLMs exhibit an average 33.12% lower success execution rate (SER) on CORGI compared to existing benchmarks such as BIRD, highlighting the substantially higher complexity of real-world business needs. We release the CORGI dataset, an evaluation framework, and a submission website to support future research.

**arXiv ID:** 2510.07309
</details>

<details>
<summary><strong>ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning</strong> - Ruiyang Zhou, Shuozhe Li, Amy Zhang, Liu Leqi - [[pdf]](https://arxiv.org/pdf/2507.02834)</summary>

**Abstract:** Self-improvement via RL often fails on complex reasoning tasks because GRPO-style post-training methods rely on the model's initial ability to generate positive samples. Without guided exploration, these approaches merely reinforce what the model already knows (distribution-sharpening) rather than enabling the model to solve problems where it initially generates no correct solutions. To unlock reasoning ability in such settings, the model must explore new reasoning trajectories beyond its current output distribution. Such exploration requires access to sufficiently good positive samples to guide the learning. While expert demonstrations seem like a natural solution, we find that they are often ineffective in RL post-training. Instead, we identify two key properties of effective positive samples: they should (1) be likely under the current policy, and (2) increase the model's likelihood of predicting the correct answer. Based on these insights, we propose $\textbf{Self-Explanation Policy Optimization (ExPO)}$-a simple and modular framework that generates such samples by conditioning on the ground-truth answer. It can be integrated with popular RL training methods like GRPO and DPO. ExPO enables efficient exploration and guides the model to produce reasoning trajectories more aligned with its policy than expert-written CoTs, while ensuring higher quality than its own (incorrect) samples. Experiments show that ExPO improves both learning efficiency and final performance on reasoning benchmarks, surpassing expert-demonstration-based methods in challenging settings such as MATH level-5, where the model initially struggles the most. Code is available at this https URL .

**arXiv ID:** 2507.02834
</details>

<details>
<summary><strong>SRT: Accelerating Reinforcement Learning via Speculative Rollout with Tree-Structured Cache</strong> - Chi-Chih Chang, Siqi Zhu, Zhichen Zeng, Haibin Lin, Jiaxuan You, Mohamed S. Abdelfattah, Ziheng Jiang, Xuehai Qian - [[pdf]](https://arxiv.org/pdf/2601.09083)</summary>

**Abstract:** We present Speculative Rollout with Tree-Structured Cache (SRT), a simple, model-free approach to accelerate on-policy reinforcement learning (RL) for language models without sacrificing distributional correctness. SRT exploits the empirical similarity of rollouts for the same prompt across training steps by storing previously generated continuations in a per-prompt tree-structured cache. During generation, the current policy uses this tree as the draft model for performing speculative decoding. To keep the cache fresh and improve draft model quality, SRT updates trees online from ongoing rollouts and proactively performs run-ahead generation during idle GPU bubbles. Integrated into standard RL pipelines (\textit{e.g.}, PPO, GRPO and DAPO) and multi-turn settings, SRT consistently reduces generation and step latency and lowers per-token inference cost, achieving up to 2.08x wall-clock time speedup during rollout.

**arXiv ID:** 2601.09083
</details>

<details>
<summary><strong>Privacy-Preserving in Connected and Autonomous Vehicles Through Vision to Text Transformation</strong> - Abdolazim Rezaei, Mehdi Sookhak, Ahmad Patooghy, Shahab S. Band, Amir Mosavi - [[pdf]](https://arxiv.org/pdf/2506.15854)</summary>

**Abstract:** Intelligent Transportation Systems (ITS) rely on a variety of devices that frequently process privacy-sensitive data. Roadside units are important because they use AI-equipped cameras to detect traffic violations in Connected and Autonomous Vehicles (CAV). However, although the interior of a vehicle is generally considered a private space, the privacy risks associated with captured imagery remain a major concern, as such data can be misused for identity theft, profiling, or unauthorized commercial purposes. Methods like face blurring reduce privacy risks, however individuals' privacy can still be compromised. This paper introduces a novel privacy-preserving framework that leverages feedback-based reinforcement learning (RL) and vision-language models (VLMs) to protect sensitive visual information captured by AIE cameras. The proposed idea transforms images into textual descriptions using an innovative method while the main scene details are preserved and protects privacy. A hierarchical RL strategy is employed to iteratively refine the generated text, enhancing both semantic accuracy and privacy. Unlike prior captioning-based methods, our model incorporates an iterative reinforcement-learning cycle with external knowledge feedback which progressively refines privacy-aware text. In addition to qualitative textual metric evaluations, the privacy-based metrics demonstrate significant improvements in privacy preservation where SSIM, PSNR, MSE, and SRRA values obtained using the proposed method on two different datasets outperform other methods.

**arXiv ID:** 2506.15854
</details>

<details>
<summary><strong>SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling</strong> - Yixian Zhang, Shu'ang Yu, Tonghe Zhang, Mo Guang, Haojia Hui, Kaiwen Long, Yu Wang, Chao Yu, Wenbo Ding - [[pdf]](https://arxiv.org/pdf/2509.25756)</summary>

**Abstract:** Training expressive flow-based policies with off-policy reinforcement learning is notoriously unstable due to gradient pathologies in the multi-step action sampling process. We trace this instability to a fundamental connection: the flow rollout is algebraically equivalent to a residual recurrent computation, making it susceptible to the same vanishing and exploding gradients as RNNs. To address this, we reparameterize the velocity network using principles from modern sequential models, introducing two stable architectures: Flow-G, which incorporates a gated velocity, and Flow-T, which utilizes a decoded velocity. We then develop a practical SAC-based algorithm, enabled by a noise-augmented rollout, that facilitates direct end-to-end training of these policies. Our approach supports both from-scratch and offline-to-online learning and achieves state-of-the-art performance on continuous control and robotic manipulation benchmarks, eliminating the need for common workarounds like policy distillation or surrogate objectives.

**arXiv ID:** 2509.25756
</details>

<details>
<summary><strong>JuggleRL: Mastering Ball Juggling with a Quadrotor via Deep Reinforcement Learning</strong> - Shilong Ji, Yinuo Chen, Chuqi Wang, Jiayu Chen, Ruize Zhang, Feng Gao, Wenhao Tang, Shu'ang Yu, Sirui Xiang, Xinlei Chen, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2509.24892)</summary>

**Abstract:** Aerial robots interacting with objects must perform precise, contact-rich maneuvers under uncertainty. In this paper, we study the problem of aerial ball juggling using a quadrotor equipped with a racket, a task that demands accurate timing, stable control, and continuous adaptation. We propose JuggleRL, the first reinforcement learning-based system for aerial juggling. It learns closed-loop policies in large-scale simulation using systematic calibration of quadrotor and ball dynamics to reduce the sim-to-real gap. The training incorporates reward shaping to encourage racket-centered hits and sustained juggling, as well as domain randomization over ball position and coefficient of restitution to enhance robustness and transferability. The learned policy outputs mid-level commands executed by a low-level controller and is deployed zero-shot on real hardware, where an enhanced perception module with a lightweight communication protocol reduces delays in high-frequency state estimation and ensures real-time control. Experiments show that JuggleRL achieves an average of $311$ hits over $10$ consecutive trials in the real world, with a maximum of $462$ hits observed, far exceeding a model-based baseline that reaches at most $14$ hits with an average of $3.1$. Moreover, the policy generalizes to unseen conditions, successfully juggling a lighter $5$ g ball with an average of $145.9$ hits. This work demonstrates that reinforcement learning can empower aerial robots with robust and stable control in dynamic interaction tasks.

**arXiv ID:** 2509.24892
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
