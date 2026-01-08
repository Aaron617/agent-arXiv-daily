# Agent arXiv Daily

**Last Updated:** 2026-01-08 02:26:46

**Total Papers:** 73

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (8 papers)</h2></summary>

<details>
<summary><strong>Textual Explanations and Their Evaluations for Reinforcement Learning Policy</strong> - Ahmad Terra, Mohit Ahmed, Rafia Inam, Elena Fersman, Martin Törngren - [[pdf]](https://arxiv.org/pdf/2601.02514)</summary>

**Abstract:** Understanding a Reinforcement Learning (RL) policy is crucial for ensuring that autonomous agents behave according to human expectations. This goal can be achieved using Explainable Reinforcement Learning (XRL) techniques. Although textual explanations are easily understood by humans, ensuring their correctness remains a challenge, and evaluations in state-of-the-art remain limited. We present a novel XRL framework for generating textual explanations, converting them into a set of transparent rules, improving their quality, and evaluating them. Expert's knowledge can be incorporated into this framework, and an automatic predicate generator is also proposed to determine the semantic information of a state. Textual explanations are generated using a Large Language Model (LLM) and a clustering technique to identify frequent conditions. These conditions are then converted into rules to evaluate their properties, fidelity, and performance in the deployed environment. Two refinement techniques are proposed to improve the quality of explanations and reduce conflicting information. Experiments were conducted in three open-source environments to enable reproducibility, and in a telecom use case to evaluate the industrial applicability of the proposed XRL framework. This framework addresses the limitations of an existing method, Autonomous Policy Explanation, and the generated transparent rules can achieve satisfactory performance on certain tasks. This framework also enables a systematic and quantitative evaluation of textual explanations, providing valuable insights for the XRL field.

**arXiv ID:** 2601.02514
</details>

<details>
<summary><strong>AWARE-US: Benchmark for Preference-Aware Resolution in Tool-Calling Agents</strong> - Mehmet Kurmaz - [[pdf]](https://arxiv.org/pdf/2601.02643)</summary>

**Abstract:** Tool-calling conversational agents querying structured databases often face two linked failures: underspecification (missing constraints needed to run a precise query) and infeasibility (the fully specified query returns an empty set because no item satisfies all constraints). Existing work often responds with "no results" or relaxes constraints using ad hoc rules, which can violate user intent by discarding requirements the user cares about most. We frame infeasibility handling as a preference-aware query repair problem: when a query is unsatisfiable, the agent should relax the least important constraints to the user. We propose three LLM-based methods for inferring relative constraint importance from dialogue: (1) local weighting, (2) global one-shot weighting, and (3) pairwise ranking. Experiments show local weighting achieves the best preference alignment, while global weighting performs best on correct constraint relaxation. We also introduce AWARE-US, a benchmark of persona-grounded queries requiring agents to disambiguate requests via conversation and resolve infeasibility in a way consistent with persona-implied preferences.

**arXiv ID:** 2601.02643
</details>

<details>
<summary><strong>SimRPD: Optimizing Recruitment Proactive Dialogue Agents through Simulator-Based Data Evaluation and Selection</strong> - Zhiyong Cao, Dunqiang Liu, Qi Dai, Haojun Xu, Huaiyan Xu, Huan He, Yafei Liu, Siyuan Liu, XiaoLin Lin, Ke Ma, Ruqian Shi, Sijia Yao, Hao Wang, Sicheng Zhou - [[pdf]](https://arxiv.org/pdf/2601.02871)</summary>

**Abstract:** Task-oriented proactive dialogue agents play a pivotal role in recruitment, particularly for steering conversations towards specific business outcomes, such as acquiring social-media contacts for private-channel conversion. Although supervised fine-tuning and reinforcement learning have proven effective for training such agents, their performance is heavily constrained by the scarcity of high-quality, goal-oriented domain-specific training data. To address this challenge, we propose SimRPD, a three-stage framework for training recruitment proactive dialogue agents. First, we develop a high-fidelity user simulator to synthesize large-scale conversational data through multi-turn online dialogue. Then we introduce a multi-dimensional evaluation framework based on Chain-of-Intention (CoI) to comprehensively assess the simulator and effectively select high-quality data, incorporating both global-level and instance-level metrics. Finally, we train the recruitment proactive dialogue agent on the selected dataset. Experiments in a real-world recruitment scenario demonstrate that SimRPD outperforms existing simulator-based data selection strategies, highlighting its practical value for industrial deployment and its potential applicability to other business-oriented dialogue scenarios.

**arXiv ID:** 2601.02871
</details>

<details>
<summary><strong>TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents</strong> - Kai Li, Xuanqing Yu, Ziyi Ni, Yi Zeng, Yao Xu, Zheqing Zhang, Xin Li, Jitao Sang, Xiaogang Duan, Xuelei Wang, Chengbao Liu, Jie Tan - [[pdf]](https://arxiv.org/pdf/2601.02845)</summary>

**Abstract:** Long-horizon conversational agents have to manage ever-growing interaction histories that quickly exceed the finite context windows of large language models (LLMs). Existing memory frameworks provide limited support for temporally structured information across hierarchical levels, often leading to fragmented memories and unstable long-horizon personalization. We present TiMem, a temporal--hierarchical memory framework that organizes conversations through a Temporal Memory Tree (TMT), enabling systematic memory consolidation from raw conversational observations to progressively abstracted persona representations. TiMem is characterized by three core properties: (1) temporal--hierarchical organization through TMT; (2) semantic-guided consolidation that enables memory integration across hierarchical levels without fine-tuning; and (3) complexity-aware memory recall that balances precision and efficiency across queries of varying complexity. Under a consistent evaluation setup, TiMem achieves state-of-the-art accuracy on both benchmarks, reaching 75.30% on LoCoMo and 76.88% on LongMemEval-S. It outperforms all evaluated baselines while reducing the recalled memory length by 52.20% on LoCoMo. Manifold analysis indicates clear persona separation on LoCoMo and reduced dispersion on LongMemEval-S. Overall, TiMem treats temporal continuity as a first-class organizing principle for long-horizon memory in conversational agents.

**arXiv ID:** 2601.02845
</details>

<details>
<summary><strong>Gradient Coupling: The Hidden Barrier to Generalization in Agentic Reinforcement Learning</strong> - Jingyu Liu, Xiaopeng Wu, Jingquan Peng, Kehan Chen, Chuan Yu, Lizhong Ding, Yong Liu - [[pdf]](https://arxiv.org/pdf/2509.23870)</summary>

**Abstract:** Reinforcement learning (RL) is a dominant paradigm for training autonomous agents, yet these agents often exhibit poor generalization, failing to adapt to scenarios not seen during training. In this work, we identify a fundamental cause of this brittleness, a phenomenon which we term "gradient coupling." We hypothesize that in complex agentic tasks, the high similarity between distinct states leads to destructive interference between gradients. Specifically, a gradient update that reinforces an optimal action in one state can inadvertently increase the likelihood of a suboptimal action in a similar, yet different, state. To solve this, we propose a novel objective where the actor is trained to simultaneously function as a classifier that separates good and bad actions. This auxiliary pressure compels the model to learn disentangled embeddings for positive and negative actions, which mitigates negative gradient interference and improve the generalization performance. Extensive experiments demonstrate the effectiveness of our method.

**arXiv ID:** 2509.23870
</details>

<details>
<summary><strong>ShareChat: A Dataset of Chatbot Conversations in the Wild</strong> - Yueru Yan, Tuc Nguyen, Bo Su, Melissa Lieffers, Thai Le - [[pdf]](https://arxiv.org/pdf/2512.17843)</summary>

**Abstract:** While academic research typically treats Large Language Models (LLM) as generic text generators, they are distinct commercial products with unique interfaces and capabilities that fundamentally shape user behavior. Current datasets obscure this reality by collecting text-only data through uniform interfaces that fail to capture authentic chatbot usage. To address this limitation, we present ShareChat, a large-scale corpus of 142,808 conversations (660,293 turns) sourced directly from publicly shared URLs on ChatGPT, Perplexity, Grok, Gemini, and Claude. ShareChat distinguishes itself by preserving native platform affordances, such as citations and thinking traces, across a diverse collection covering 101 languages and the period from April 2023 to October 2025. Furthermore, ShareChat offers substantially longer context windows and greater interaction depth than prior datasets. To illustrate the dataset's breadth, we present three case studies: a completeness analysis of intent satisfaction, a citation study of model grounding, and a temporal analysis of engagement rhythms. This work provides the community with a vital and timely resource for understanding authentic user-LLM chatbot interactions in the wild. The dataset will be publicly available.

**arXiv ID:** 2512.17843
</details>

<details>
<summary><strong>SmartSnap: Proactive Evidence Seeking for Self-Verifying Agents</strong> - Shaofei Cai, Yulei Qin, Haojia Lin, Zihan Xu, Gang Li, Yuchen Shi, Zongyi Li, Yong Mao, Siqi Cai, Xiaoyu Tan, Yitao Liang, Ke Li, Xing Sun - [[pdf]](https://arxiv.org/pdf/2512.22322)</summary>

**Abstract:** Agentic reinforcement learning (RL) holds great promise for the development of autonomous agents under complex GUI tasks, but its scalability remains severely hampered by the verification of task completion. Existing task verification is treated as a passive, post-hoc process: a verifier (i.e., rule-based scoring script, reward or critic model, and LLM-as-a-Judge) analyzes the agent's entire interaction trajectory to determine if the agent succeeds. Such processing of verbose context that contains irrelevant, noisy history poses challenges to the verification protocols and therefore leads to prohibitive cost and low reliability. To overcome this bottleneck, we propose SmartSnap, a paradigm shift from this passive, post-hoc verification to proactive, in-situ self-verification by the agent itself. We introduce the Self-Verifying Agent, a new type of agent designed with dual missions: to not only complete a task but also to prove its accomplishment with curated snapshot evidences. Guided by our proposed 3C Principles (Completeness, Conciseness, and Creativity), the agent leverages its accessibility to the online environment to perform self-verification on a minimal, decisive set of snapshots. Such evidences are provided as the sole materials for a general LLM-as-a-Judge verifier to determine their validity and relevance. Experiments on mobile tasks across model families and scales demonstrate that our SmartSnap paradigm allows training LLM-driven agents in a scalable manner, bringing performance gains up to 26.08% and 16.66% respectively to 8B and 30B models. The synergizing between solution finding and evidence seeking facilitates the cultivation of efficient, self-verifying agents with competitive performance against DeepSeek V3.1 and Qwen3-235B-A22B. Code is available at: this https URL

**arXiv ID:** 2512.22322
</details>

<details>
<summary><strong>Enhancing Safety in Automated Ports: A Virtual Reality Study of Pedestrian-Autonomous Vehicle Interactions under Time Pressure, Visual Constraints, and Varying Vehicle Size</strong> - Yuan Che, Mun On Wong, Xiaowei Gao, Haoyang Liang, Yun Ye - [[pdf]](https://arxiv.org/pdf/2601.03218)</summary>

**Abstract:** Autonomous driving improves traffic efficiency but presents safety challenges in complex port environments. This study investigates how environmental factors, traffic factors, and pedestrian characteristics influence interaction safety between autonomous vehicles and pedestrians in ports. Using virtual reality (VR) simulations of typical port scenarios, 33 participants completed pedestrian crossing tasks under varying visibility, vehicle sizes, and time pressure conditions. Results indicate that low-visibility conditions, partial occlusions and larger vehicle sizes significantly increase perceived risk, prompting pedestrians to wait longer and accept larger gaps. Specifically, pedestrians tended to accept larger gaps and waited longer when interacting with large autonomous truck platoons, reflecting heightened caution due to their perceived threat. However, local obstructions also reduce post-encroachment time, compressing safety margins. Individual attributes such as age, gender, and driving experience further shape decision-making, while time pressure undermines compensatory behaviors and increases risk. Based on these findings, safety strategies are proposed, including installing wide-angle cameras at multiple viewpoints, enabling real-time vehicle-infrastructure communication, enhancing port lighting and signage, and strengthening pedestrian safety training. This study offers practical recommendations for improving the safety and deployment of vision-based autonomous systems in port settings.

**arXiv ID:** 2601.03218
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (11 papers)</h2></summary>

<details>
<summary><strong>Causal-Enhanced AI Agents for Medical Research Screening</strong> - Duc Ngo, Arya Rahgoza - [[pdf]](https://arxiv.org/pdf/2601.02814)</summary>

**Abstract:** Systematic reviews are essential for evidence-based medicine, but reviewing 1.5 million+ annual publications manually is infeasible. Current AI approaches suffer from hallucinations in systematic review tasks, with studies reporting rates ranging from 28--40% for earlier models to 2--15% for modern implementations which is unacceptable when errors impact patient care.
We present a causal graph-enhanced retrieval-augmented generation system integrating explicit causal reasoning with dual-level knowledge graphs. Our approach enforces evidence-first protocols where every causal claim traces to retrieved literature and automatically generates directed acyclic graphs visualizing intervention-outcome pathways.
Evaluation on 234 dementia exercise abstracts shows CausalAgent achieves 95% accuracy, 100% retrieval success, and zero hallucinations versus 34% accuracy and 10% hallucinations for baseline AI. Automatic causal graphs enable explicit mechanism modeling, visual synthesis, and enhanced interpretability. While this proof-of-concept evaluation used ten questions focused on dementia exercise research, the architectural approach demonstrates transferable principles for trustworthy medical AI and causal reasoning's potential for high-stakes healthcare.

**arXiv ID:** 2601.02814
</details>

<details>
<summary><strong>FUSE : Failure-aware Usage of Subagent Evidence for MultiModal Search and Recommendation</strong> - Tushar Vatsa, Vibha Belavadi, Priya Shanmugasundaram, Suhas Suresha, Dewang Sultania - [[pdf]](https://arxiv.org/pdf/2601.02365)</summary>

**Abstract:** Multimodal creative assistants decompose user goals and route tasks to subagents for layout, styling, retrieval, and generation. Retrieval quality is pivotal, yet failures can arise at several stages: understanding user intent, choosing content types, finding candidates (recall), or ranking results. Meanwhile, sending and processing images is costly, making naive multimodal approaches impractical. We present FUSE: Failure-aware Usage of Subagent Evidence for MultiModal Search and Recommendation. FUSE replaces most raw-image prompting with a compact Grounded Design Representation (GDR): a selection aware JSON of canvas elements (image, text, shape, icon, video, logo), structure, styles, salient colors, and user selection provided by the Planner team. FUSE implements seven context budgeting strategies: comprehensive baseline prompting, context compression, chain-of-thought reasoning, mini-shot optimization, retrieval-augmented context, two-stage processing, and zero-shot minimalism. Finally, a pipeline attribution layer monitors system performance by converting subagent signals into simple checks: intent alignment, content-type/routing sanity, recall health (e.g., zero-hit and top-match strength), and ranking displacement analysis. We evaluate the seven context budgeting variants across 788 evaluation queries from diverse users and design templates (refer Figure 3). Our systematic evaluation reveals that Context Compression achieves optimal performance across all pipeline stages, with 93.3% intent accuracy, 86.8% routing success(with fallbacks), 99.4% recall, and 88.5% NDCG@5. This approach demonstrates that strategic context summarization outperforms both comprehensive and minimal contextualization strategies.

**arXiv ID:** 2601.02365
</details>

<details>
<summary><strong>ProSoftArena: Benchmarking Hierarchical Capabilities of Multimodal Agents in Professional Software Environments</strong> - Jiaxin Ai, Yukang Feng, Fanrui Zhang, Jianwen Sun, Zizhen Li, Chuanhao Li, Yifan Chang, Wenxiao Wu, Ruoxi Wang, Mingliang Zhai, Kaipeng Zhang - [[pdf]](https://arxiv.org/pdf/2601.02399)</summary>

**Abstract:** Multimodal agents are making rapid progress on general computer-use tasks, yet existing benchmarks remain largely confined to browsers and basic desktop applications, falling short in professional software workflows that dominate real-world scientific and industrial practice. To close this gap, we introduce ProSoftArena, a benchmark and platform specifically for evaluating multimodal agents in professional software environments. We establish the first capability hierarchy tailored to agent use of professional software and construct a benchmark of 436 realistic work and research tasks spanning 6 disciplines and 13 core professional applications. To ensure reliable and reproducible assessment, we build an executable real-computer environment with an execution-based evaluation framework and uniquely incorporate a human-in-the-loop evaluation paradigm. Extensive experiments show that even the best-performing agent attains only a 24.4\% success rate on L2 tasks and completely fails on L3 multi-software workflow. In-depth analysis further provides valuable insights for addressing current agent limitations and more effective design principles, paving the way to build more capable agents in professional software settings. This project is available at: this https URL.

**arXiv ID:** 2601.02399
</details>

<details>
<summary><strong>NitroGen: An Open Foundation Model for Generalist Gaming Agents</strong> - Loïc Magne, Anas Awadalla, Guanzhi Wang, Yinzhen Xu, Joshua Belofsky, Fengyuan Hu, Joohwan Kim, Ludwig Schmidt, Georgia Gkioxari, Jan Kautz, Yisong Yue, Yejin Choi, Yuke Zhu, Linxi "Jim" Fan - [[pdf]](https://arxiv.org/pdf/2601.02427)</summary>

**Abstract:** We introduce NitroGen, a vision-action foundation model for generalist gaming agents that is trained on 40,000 hours of gameplay videos across more than 1,000 games. We incorporate three key ingredients: 1) an internet-scale video-action dataset constructed by automatically extracting player actions from publicly available gameplay videos, 2) a multi-game benchmark environment that can measure cross-game generalization, and 3) a unified vision-action model trained with large-scale behavior cloning. NitroGen exhibits strong competence across diverse domains, including combat encounters in 3D action games, high-precision control in 2D platformers, and exploration in procedurally generated worlds. It transfers effectively to unseen games, achieving up to 52% relative improvement in task success rates over models trained from scratch. We release the dataset, evaluation suite, and model weights to advance research on generalist embodied agents.

**arXiv ID:** 2601.02427
</details>

<details>
<summary><strong>SastBench: A Benchmark for Testing Agentic SAST Triage</strong> - Jake Feiglin, Guy Dar - [[pdf]](https://arxiv.org/pdf/2601.02941)</summary>

**Abstract:** SAST (Static Application Security Testing) tools are among the most widely used techniques in defensive cybersecurity, employed by commercial and non-commercial organizations to identify potential vulnerabilities in software. Despite their great utility, they generate numerous false positives, requiring costly manual filtering (aka triage). While LLM-powered agents show promise for automating cybersecurity tasks, existing benchmarks fail to emulate real-world SAST finding distributions. We introduce SastBench, a benchmark for evaluating SAST triage agents that combines real CVEs as true positives with filtered SAST tool findings as approximate false positives. SastBench features an agent-agnostic design. We evaluate different agents on the benchmark and present a comparative analysis of their performance, provide a detailed analysis of the dataset, and discuss the implications for future development.

**arXiv ID:** 2601.02941
</details>

<details>
<summary><strong>PiDR: Physics-Informed Inertial Dead Reckoning for Autonomous Platforms</strong> - Arup Kumar Sahoo, Itzik Klein - [[pdf]](https://arxiv.org/pdf/2601.03040)</summary>

**Abstract:** A fundamental requirement for full autonomy is the ability to sustain accurate navigation in the absence of external data, such as GNSS signals or visual information. In these challenging environments, the platform must rely exclusively on inertial sensors, leading to pure inertial navigation. However, the inherent noise and other error terms of the inertial sensors in such real-world scenarios will cause the navigation solution to drift over time. Although conventional deep-learning models have emerged as a possible approach to inertial navigation, they are inherently black-box in nature. Furthermore, they struggle to learn effectively with limited supervised sensor data and often fail to preserve physical principles. To address these limitations, we propose PiDR, a physics-informed inertial dead-reckoning framework for autonomous platforms in situations of pure inertial navigation. PiDR offers transparency by explicitly integrating inertial navigation principles into the network training process through the physics-informed residual component. PiDR plays a crucial role in mitigating abrupt trajectory deviations even under limited or sparse supervision. We evaluated PiDR on real-world datasets collected by a mobile robot and an autonomous underwater vehicle. We obtained more than 29% positioning improvement in both datasets, demonstrating the ability of PiDR to generalize different platforms operating in various environments and dynamics. Thus, PiDR offers a robust, lightweight, yet effective architecture and can be deployed on resource-constrained platforms, enabling real-time pure inertial navigation in adverse scenarios.

**arXiv ID:** 2601.03040
</details>

<details>
<summary><strong>CodeEvolve: an open source evolutionary coding agent for algorithm discovery and optimization</strong> - Henrique Assumpção, Diego Ferreira, Leandro Campos, Fabricio Murai - [[pdf]](https://arxiv.org/pdf/2510.14150)</summary>

**Abstract:** We introduce CodeEvolve, an open-source framework that combines large language models (LLMs) with evolutionary search to synthesize high-performing algorithmic solutions. CodeEvolve couples an islands-based genetic algorithm with modular LLM orchestration, using execution feedback and task-specific metrics to guide selection and variation. Exploration and exploitation are balanced through context-aware recombination, adaptive meta-prompting, and targeted refinement of promising solutions. We evaluate CodeEvolve on benchmarks previously used to assess Google DeepMind's AlphaEvolve, showing superior performance on several tasks and competitive results overall. Notably, open-weight models often match or exceed closed-source baselines at a fraction of the compute cost. We provide extensive ablations analyzing the contribution of each component and release our framework and experimental results at this https URL.

**arXiv ID:** 2510.14150
</details>

<details>
<summary><strong>DarkEQA: Benchmarking Vision-Language Models for Embodied Question Answering in Low-Light Indoor Environments</strong> - Yohan Park, Hyunwoo Ha, Wonjun Jo, Tae-Hyun Oh - [[pdf]](https://arxiv.org/pdf/2512.24985)</summary>

**Abstract:** Vision Language Models (VLMs) are increasingly adopted as central reasoning modules for embodied agents. Existing benchmarks evaluate their capabilities under ideal, well-lit conditions, yet robust 24/7 operation demands performance under a wide range of visual degradations, including low-light conditions at night or in dark environments--a core necessity that has been largely overlooked. To address this underexplored challenge, we present DarkEQA, an open-source benchmark for evaluating EQA-relevant perceptual primitives under multi-level low-light conditions. DarkEQA isolates the perception bottleneck by evaluating question answering from egocentric observations under controlled degradations, enabling attributable robustness analysis. A key design feature of DarkEQA is its physical fidelity: visual degradations are modeled in linear RAW space, simulating physics-based illumination drop and sensor noise followed by an ISP-inspired rendering pipeline. We demonstrate the utility of DarkEQA by evaluating a wide range of state-of-the-art VLMs and Low-Light Image Enhancement (LLIE) models. Our analysis systematically reveals VLMs' limitations when operating under these challenging visual conditions. Project website: this https URL

**arXiv ID:** 2512.24985
</details>

<details>
<summary><strong>Modeling the Mental World for Embodied AI: A Comprehensive Review</strong> - Biyuan Liu, Daigang Xu, Lei Jiang, Wenjun Guo, Ping Chen - [[pdf]](https://arxiv.org/pdf/2601.02378)</summary>

**Abstract:** As the application of Embodied AI Agents in avatars, wearable devices, and robotic systems continues to deepen, their core research challenges have gradually shifted from physical environment interaction to the accurate understanding of social interactions. Traditional physical world models (PWM) focus on quantifiable physical attributes such as space and motion, failing to meet the needs of social intelligence modeling. In contrast, the Mental World Model (MWM), as a structured representation of humans' internal mental states, has become the critical cognitive foundation for embodied agents to achieve natural human-machine collaboration and dynamic social adaptation. However, current MWM research faces significant bottlenecks: such as fragmented conceptual framework with vague boundaries between MWM and PWM, disjointed reasoning mechanisms for the technical pathways and applicable scenarios of different Theory of Mind (ToM) reasoning paradigms, and detachment between evaluation and practice.
To address these issues, this review systematically synthesizes over 100 authoritative studies to provide a comprehensive overview of MWM research for embodied AI. Its core contributions are threefold: First, it constructs a complete theoretical framework for MWM for the first time. Specifically, it distinguishes the essential differences between MWM and PWMs. Second, it systematically defines the key components of MWM through two paradigms for mental element representation. Third, it comprehensively analyzes two core ToM reasoning paradigms with 19 ToM methods. Finally, it also clarifies the integration trend of neuro-symbolic hybrid architectures, and synthesizes 26 ToM evaluation benchmarks. This work aims to promote the integration of embodied agents into human society and advance the in-depth development of human-machine collaborative interaction.

**arXiv ID:** 2601.02378
</details>

<details>
<summary><strong>Dual-quaternion learning control for autonomous vehicle trajectory tracking with safety guarantees</strong> - Omayra Yago Nieto, Alexandre Anahory Simoes, Juan I. Giribet, Leonardo Colombo - [[pdf]](https://arxiv.org/pdf/2601.03097)</summary>

**Abstract:** We propose a learning-based trajectory tracking controller for autonomous robotic platforms whose motion can be described kinematically on $\mathrm{SE}(3)$. The controller is formulated in the dual quaternion framework and operates at the velocity level, assuming direct command of angular and linear velocities, as is standard in many aerial vehicles and omnidirectional mobile robots. Gaussian Process (GP) regression is integrated into a geometric feedback law to learn and compensate online for unknown, state-dependent disturbances and modeling imperfections affecting both attitude and position, while preserving the algebraic structure and coupling properties inherent to rigid-body motion.
The proposed approach does not rely on explicit parametric models of the unknown effects, making it well-suited for robotic systems subject to sensor-induced disturbances, unmodeled actuation couplings, and environmental uncertainties. A Lyapunov-based analysis establishes probabilistic ultimate boundedness of the pose tracking error under bounded GP uncertainty, providing formal stability guarantees for the learning-based controller.
Simulation results demonstrate accurate and smooth trajectory tracking in the presence of realistic, localized disturbances, including correlated rotational and translational effects arising from magnetometer perturbations. These results illustrate the potential of combining geometric modeling and probabilistic learning to achieve robust, data-efficient pose control for autonomous robotic systems.

**arXiv ID:** 2601.03097
</details>

<details>
<summary><strong>VLN-MME: Diagnosing MLLMs as Language-guided Visual Navigation agents</strong> - Xunyi Zhao, Gengze Zhou, Qi Wu - [[pdf]](https://arxiv.org/pdf/2512.24851)</summary>

**Abstract:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across a wide range of vision-language tasks. However, their performance as embodied agents, which requires multi-round dialogue spatial reasoning and sequential action prediction, needs further exploration. Our work investigates this potential in the context of Vision-and-Language Navigation (VLN) by introducing a unified and extensible evaluation framework to probe MLLMs as zero-shot agents by bridging traditional navigation datasets into a standardized benchmark, named VLN-MME. We simplify the evaluation with a highly modular and accessible design. This flexibility streamlines experiments, enabling structured comparisons and component-level ablations across diverse MLLM architectures, agent designs, and navigation tasks. Crucially, enabled by our framework, we observe that enhancing our baseline agent with Chain-of-Thought (CoT) reasoning and self-reflection leads to an unexpected performance decrease. This suggests MLLMs exhibit poor context awareness in embodied navigation tasks; although they can follow instructions and structure their output, their 3D spatial reasoning fidelity is low. VLN-MME lays the groundwork for systematic evaluation of general-purpose MLLMs in embodied navigation settings and reveals limitations in their sequential decision-making capabilities. We believe these findings offer crucial guidance for MLLM post-training as embodied agents.

**arXiv ID:** 2512.24851
</details>

</details>

<details open>
<summary><h2>LLM Agents (9 papers)</h2></summary>

<details>
<summary><strong>SimpleMem: Efficient Lifelong Memory for LLM Agents</strong> - Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, Huaxiu Yao - [[pdf]](https://arxiv.org/pdf/2601.02553)</summary>

**Abstract:** To support reliable long-term interaction in complex environments, LLM agents require memory systems that efficiently manage historical experiences. Existing approaches either retain full interaction histories via passive context extension, leading to substantial redundancy, or rely on iterative reasoning to filter noise, incurring high token costs. To address this challenge, we introduce SimpleMem, an efficient memory framework based on semantic lossless compression. We propose a three-stage pipeline designed to maximize information density and token utilization: (1) \textit{Semantic Structured Compression}, which applies entropy-aware filtering to distill unstructured interactions into compact, multi-view indexed memory units; (2) \textit{Recursive Memory Consolidation}, an asynchronous process that integrates related units into higher-level abstract representations to reduce redundancy; and (3) \textit{Adaptive Query-Aware Retrieval}, which dynamically adjusts retrieval scope based on query complexity to construct precise context efficiently. Experiments on benchmark datasets show that our method consistently outperforms baseline approaches in accuracy, retrieval efficiency, and inference cost, achieving an average F1 improvement of 26.4% while reducing inference-time token consumption by up to 30-fold, demonstrating a superior balance between performance and efficiency. Code is available at this https URL.

**arXiv ID:** 2601.02553
</details>

<details>
<summary><strong>Orchestral AI: A Framework for Agent Orchestration</strong> - Alexander Roman, Jacob Roman - [[pdf]](https://arxiv.org/pdf/2601.02577)</summary>

**Abstract:** The rapid proliferation of LLM agent frameworks has forced developers to choose between vendor lock-in through provider-specific SDKs and complex multi-package ecosystems that obscure control flow and hinder reproducibility. Integrating tool calling across multiple LLM providers remains a core engineering challenge due to fragmented APIs, incompatible message formats, and inconsistent streaming and tool-calling behavior, making it difficult to build portable, reliable agent systems. We introduce Orchestral, a lightweight Python framework that provides a unified, type-safe interface for building LLM agents across major providers while preserving the simplicity required for scientific computing and production deployment. Orchestral defines a single universal representation for messages, tools, and LLM usage that operates seamlessly across providers, eliminating manual format translation and reducing framework-induced complexity. Automatic tool schema generation from Python type hints removes the need for handwritten descriptors while maintaining type safety across provider boundaries. A synchronous execution model with streaming support enables deterministic behavior, straightforward debugging, and real-time interaction without introducing server dependencies. The framework's modular architecture cleanly separates provider integration, tool execution, conversation orchestration, and user-facing interfaces, enabling extensibility without architectural entanglement. Orchestral supports advanced agent capabilities found in larger frameworks, including rich tool calling, context compaction, workspace sandboxing, user approval workflows, sub-agents, memory management, and MCP integration.

**arXiv ID:** 2601.02577
</details>

<details>
<summary><strong>LLM Agent Framework for Intelligent Change Analysis in Urban Environment using Remote Sensing Imagery</strong> - Zixuan Xiao, Jun Ma - [[pdf]](https://arxiv.org/pdf/2601.02757)</summary>

**Abstract:** Existing change detection methods often lack the versatility to handle diverse real-world queries and the intelligence for comprehensive analysis. This paper presents a general agent framework, integrating Large Language Models (LLM) with vision foundation models to form ChangeGPT. A hierarchical structure is employed to mitigate hallucination. The agent was evaluated on a curated dataset of 140 questions categorized by real-world scenarios, encompassing various question types (e.g., Size, Class, Number) and complexities. The evaluation assessed the agent's tool selection ability (Precision/Recall) and overall query accuracy (Match). ChangeGPT, especially with a GPT-4-turbo backend, demonstrated superior performance, achieving a 90.71 % Match rate. Its strength lies particularly in handling change-related queries requiring multi-step reasoning and robust tool selection. Practical effectiveness was further validated through a real-world urban change monitoring case study in Qianhai Bay, Shenzhen. By providing intelligence, adaptability, and multi-type change analysis, ChangeGPT offers a powerful solution for decision-making in remote sensing applications.

**arXiv ID:** 2601.02757
</details>

<details>
<summary><strong>InfiAgent: An Infinite-Horizon Framework for General-Purpose Autonomous Agents</strong> - Chenglin Yu, Yuchen Wang, Songmiao Wang, Hongxia Yang, Ming Li - [[pdf]](https://arxiv.org/pdf/2601.03204)</summary>

**Abstract:** LLM agents can reason and use tools, but they often break down on long-horizon tasks due to unbounded context growth and accumulated errors. Common remedies such as context compression or retrieval-augmented prompting introduce trade-offs between information fidelity and reasoning stability. We present InfiAgent, a general-purpose framework that keeps the agent's reasoning context strictly bounded regardless of task duration by externalizing persistent state into a file-centric state abstraction. At each step, the agent reconstructs context from a workspace state snapshot plus a fixed window of recent actions. Experiments on DeepResearch and an 80-paper literature review task show that, without task-specific fine-tuning, InfiAgent with a 20B open-source model is competitive with larger proprietary systems and maintains substantially higher long-horizon coverage than context-centric baselines. These results support explicit state externalization as a practical foundation for stable long-horizon agents. Github Repo:this https URL

**arXiv ID:** 2601.03204
</details>

<details>
<summary><strong>LongDA: Benchmarking LLM Agents for Long-Document Data Analysis</strong> - Yiyang Li, Zheyuan Zhang, Tianyi Ma, Zehong Wang, Keerthiram Murugesan, Chuxu Zhang, Yanfang Ye - [[pdf]](https://arxiv.org/pdf/2601.02598)</summary>

**Abstract:** We introduce LongDA, a data analysis benchmark for evaluating LLM-based agents under documentation-intensive analytical workflows. In contrast to existing benchmarks that assume well-specified schemas and inputs, LongDA targets real-world settings in which navigating long documentation and complex data is the primary bottleneck. To this end, we manually curate raw data files, long and heterogeneous documentation, and expert-written publications from 17 publicly available U.S. national surveys, from which we extract 505 analytical queries grounded in real analytical practice. Solving these queries requires agents to first retrieve and integrate key information from multiple unstructured documents, before performing multi-step computations and writing executable code, which remains challenging for existing data analysis agents. To support the systematic evaluation under this setting, we develop LongTA, a tool-augmented agent framework that enables document access, retrieval, and code execution, and evaluate a range of proprietary and open-source models. Our experiments reveal substantial performance gaps even among state-of-the-art models, highlighting the challenges researchers should consider before applying LLM agents for decision support in real-world, high-stakes analytical settings.

**arXiv ID:** 2601.02598
</details>

<details>
<summary><strong>Pro2Guard: Proactive Runtime Enforcement of LLM Agent Safety via Probabilistic Model Checking</strong> - Haoyu Wang, Christopher M. Poskitt, Jun Sun, Jiali Wei - [[pdf]](https://arxiv.org/pdf/2508.00500)</summary>

**Abstract:** Large Language Model (LLM) agents demonstrate strong autonomy, but their stochastic behavior introduces unpredictable safety risks. Existing rule-based enforcement systems, such as AgentSpec, are reactive, intervening only when unsafe behavior is imminent or has occurred, lacking foresight for long-horizon dependencies. To overcome these limitations, we present a proactive runtime enforcement framework for LLM agents. The framework abstracts agent behaviors into symbolic states and learns a Discrete-Time Markov Chain (DTMC) from execution traces. At runtime, it predicts the probability of leading to undesired behaviors and intervenes before violations occur when the estimated risk exceeds a user-defined threshold. Designed to provide PAC-correctness guarantee, the framework achieves statistically reliable enforcement of agent safety. We evaluate the framework across two safety-critical domains: autonomous vehicles and embodied agents. It proactively enforces safety and maintains high task performance, outperforming existing methods.

**arXiv ID:** 2508.00500
</details>

<details>
<summary><strong>EvoRoute: Experience-Driven Self-Routing LLM Agent Systems</strong> - Guibin Zhang, Haiyang Yu, Kaiming Yang, Bingli Wu, Fei Huang, Yongbin Li, Shuicheng Yan - [[pdf]](https://arxiv.org/pdf/2601.02695)</summary>

**Abstract:** Complex agentic AI systems, powered by a coordinated ensemble of Large Language Models (LLMs), tool and memory modules, have demonstrated remarkable capabilities on intricate, multi-turn tasks. However, this success is shadowed by prohibitive economic costs and severe latency, exposing a critical, yet underexplored, trade-off. We formalize this challenge as the \textbf{Agent System Trilemma}: the inherent tension among achieving state-of-the-art performance, minimizing monetary cost, and ensuring rapid task completion. To dismantle this trilemma, we introduce EvoRoute, a self-evolving model routing paradigm that transcends static, pre-defined model assignments. Leveraging an ever-expanding knowledge base of prior experience, EvoRoute dynamically selects Pareto-optimal LLM backbones at each step, balancing accuracy, efficiency, and resource use, while continually refining its own selection policy through environment feedback. Experiments on challenging agentic benchmarks such as GAIA and BrowseComp+ demonstrate that EvoRoute, when integrated into off-the-shelf agentic systems, not only sustains or enhances system performance but also reduces execution cost by up to $80\%$ and latency by over $70\%$.

**arXiv ID:** 2601.02695
</details>

<details>
<summary><strong>SYNAPSE: Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation</strong> - Hanqi Jiang, Junhao Chen, Yi Pan, Ling Chen, Weihang You, Yifan Zhou, Ruidong Zhang, Yohannes Abate, Tianming Liu - [[pdf]](https://arxiv.org/pdf/2601.02744)</summary>

**Abstract:** While Large Language Models (LLMs) excel at generalized reasoning, standard retrieval-augmented approaches fail to address the disconnected nature of long-term agentic memory. To bridge this gap, we introduce Synapse (Synergistic Associative Processing Semantic Encoding), a unified memory architecture that transcends static vector similarity. Drawing from cognitive science, Synapse models memory as a dynamic graph where relevance emerges from spreading activation rather than pre-computed links. By integrating lateral inhibition and temporal decay, the system dynamically highlights relevant sub-graphs while filtering interference. We implement a Triple Hybrid Retrieval strategy that fuses geometric embeddings with activation-based graph traversal. Comprehensive evaluations on the LoCoMo benchmark show that Synapse significantly outperforms state-of-the-art methods in complex temporal and multi-hop reasoning tasks, offering a robust solution to the "Contextual Tunneling" problem. Our code and data will be made publicly available upon acceptance.

**arXiv ID:** 2601.02744
</details>

<details>
<summary><strong>Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC</strong> - Xinming Wei, Jiahao Zhang, Haoran Li, Jiayu Chen, Haoning Guan, Rui Qu, Maoliang Li, Xiang Chen, Guojie Luo - [[pdf]](https://arxiv.org/pdf/2506.24045)</summary>

**Abstract:** Personal LLM agents increasingly combine foreground reactive interactions with background proactive monitoring, forming long-lived, stateful LLM flows that interleave prefill and token-by-token decode. While modern heterogeneous SoCs integrate CPUs, iGPUs, and NPUs to support on-device intelligence, existing LLM engines assume static, single-shot inference and lack mechanisms for flow-level concurrency, prioritization, and efficient accelerator coordination. As a result, commodity SoCs remain poorly matched to the dynamic, mixed-criticality execution patterns of personal agents.
This paper presents Agent$.$xpu, the first LLM engine that orchestrates concurrent reactive and proactive LLM flows on commodity SoCs. Extensive profiling uncovers unique SoC characteristics of operator-accelerator affinity, asymmetric DDR contention, and stage-divergent batching behaviors distinct from cloud-serving assumptions. Agent$.$xpu introduces three key techniques: a heterogeneous execution graph (HEG) capturing NPU/iGPU affinity and elastic operator binding; flow-aware NPU-iGPU coordination with stage elasticity, decoupling prefill and decode to reduce bandwidth contention and enforce priorities; and fine-grained preemption with slack-aware piggybacking to guarantee reactive responsiveness without starving proactive work. Across realistic personal-agent workloads, Agent$.$xpu delivers 1.2-4.9$\times$ proactive throughput and reduces reactive latency by at least 91%, compared with both industrial iGPU-only serving engine and NPU-iGPU static inference with optimal tensor-partitioning schemes. Agent$.$xpu also minimizes energy consumption and graphics interference via controlled iGPU usage.

**arXiv ID:** 2506.24045
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (19 papers)</h2></summary>

<details>
<summary><strong>The Path Ahead for Agentic AI: Challenges and Opportunities</strong> - Nadia Sibai, Yara Ahmed, Serry Sibaee, Sawsan AlHalawani, Adel Ammar, Wadii Boulila - [[pdf]](https://arxiv.org/pdf/2601.02749)</summary>

**Abstract:** The evolution of Large Language Models (LLMs) from passive text generators to autonomous, goal-driven systems represents a fundamental shift in artificial intelligence. This chapter examines the emergence of agentic AI systems that integrate planning, memory, tool use, and iterative reasoning to operate autonomously in complex environments. We trace the architectural progression from statistical models to transformer-based systems, identifying capabilities that enable agentic behavior: long-range reasoning, contextual awareness, and adaptive decision-making. The chapter provides three contributions: (1) a synthesis of how LLM capabilities extend toward agency through reasoning-action-reflection loops; (2) an integrative framework describing core components perception, memory, planning, and tool execution that bridge LLMs with autonomous behavior; (3) a critical assessment of applications and persistent challenges in safety, alignment, reliability, and sustainability. Unlike existing surveys, we focus on the architectural transition from language understanding to autonomous action, emphasizing the technical gaps that must be resolved before deployment. We identify critical research priorities, including verifiable planning, scalable multi-agent coordination, persistent memory architectures, and governance frameworks. Responsible advancement requires simultaneous progress in technical robustness, interpretability, and ethical safeguards to realize potential while mitigating risks of misalignment and unintended consequences.

**arXiv ID:** 2601.02749
</details>

<details>
<summary><strong>M3MAD-Bench: Are Multi-Agent Debates Really Effective Across Domains and Modalities?</strong> - Ao Li, Jinghui Zhang, Luyu Li, Yuxiang Duan, Lang Gao, Mingcai Chen, Weijun Qin, Shaopeng Li, Fengxian Ji, Ning Liu, Lizhen Cui, Xiuying Chen, Yuntao Du - [[pdf]](https://arxiv.org/pdf/2601.02854)</summary>

**Abstract:** As an agent-level reasoning and coordination paradigm, Multi-Agent Debate (MAD) orchestrates multiple agents through structured debate to improve answer quality and support complex reasoning. However, existing research on MAD suffers from two fundamental limitations: evaluations are conducted under fragmented and inconsistent settings, hindering fair comparison, and are largely restricted to single-modality scenarios that rely on textual inputs only. To address these gaps, we introduce M3MAD-Bench, a unified and extensible benchmark for evaluating MAD methods across Multi-domain tasks, Multi-modal inputs, and Multi-dimensional metrics. M3MAD-Bench establishes standardized protocols over five core task domains: Knowledge, Mathematics, Medicine, Natural Sciences, and Complex Reasoning, and systematically covers both pure text and vision-language datasets, enabling controlled cross-modality comparison. We evaluate MAD methods on nine base models spanning different architectures, scales, and modality capabilities. Beyond accuracy, M3MAD-Bench incorporates efficiency-oriented metrics such as token consumption and inference time, providing a holistic view of performance--cost trade-offs. Extensive experiments yield systematic insights into the effectiveness, robustness, and efficiency of MAD across text-only and multimodal scenarios. We believe M3MAD-Bench offers a reliable foundation for future research on standardized MAD evaluation. The code is available at this http URL.

**arXiv ID:** 2601.02854
</details>

<details>
<summary><strong>The Rise of Agentic Testing: Multi-Agent Systems for Robust Software Quality Assurance</strong> - Saba Naqvi, Mohammad Baqar, Nawaz Ali Mohammad - [[pdf]](https://arxiv.org/pdf/2601.02454)</summary>

**Abstract:** Software testing has progressed toward intelligent automation, yet current AI-based test generators still suffer from static, single-shot outputs that frequently produce invalid, redundant, or non-executable tests due to the lack of execution aware feedback. This paper introduces an agentic multi-model testing framework a closed-loop, self-correcting system in which a Test Generation Agent, an Execution and Analysis Agent, and a Review and Optimization Agent collaboratively generate, execute, analyze, and refine tests until convergence. By using sandboxed execution, detailed failure reporting, and iterative regeneration or patching of failing tests, the framework autonomously improves test quality and expands coverage. Integrated into a CI/CD-compatible pipeline, it leverages reinforcement signals from coverage metrics and execution outcomes to guide refinement. Empirical evaluations on microservice based applications show up to a 60% reduction in invalid tests, 30% coverage improvement, and significantly reduced human effort compared to single-model baselines demonstrating that multi-agent, feedback-driven loops can evolve software testing into an autonomous, continuously learning quality assurance ecosystem for self-healing, high-reliability codebases.

**arXiv ID:** 2601.02454
</details>

<details>
<summary><strong>Agentic Memory Enhanced Recursive Reasoning for Root Cause Localization in Microservices</strong> - Lingzhe Zhang, Tong Jia, Yunpeng Zhai, Leyi Pan, Chiming Duan, Minghua He, Mengxi Jia, Ying Li - [[pdf]](https://arxiv.org/pdf/2601.02732)</summary>

**Abstract:** As contemporary microservice systems become increasingly popular and complex-often comprising hundreds or even thousands of fine-grained, interdependent subsystems-they are experiencing more frequent failures. Ensuring system reliability thus demands accurate root cause localization. While many traditional graph-based and deep learning approaches have been explored for this task, they often rely heavily on pre-defined schemas that struggle to adapt to evolving operational contexts. Consequently, a number of LLM-based methods have recently been proposed. However, these methods still face two major limitations: shallow, symptom-centric reasoning that undermines accuracy, and a lack of cross-alert reuse that leads to redundant reasoning and high latency. In this paper, we conduct a comprehensive study of how Site Reliability Engineers (SREs) localize the root causes of failures, drawing insights from professionals across multiple organizations. Our investigation reveals that expert root cause analysis exhibits three key characteristics: recursiveness, multi-dimensional expansion, and cross-modal reasoning. Motivated by these findings, we introduce AMER-RCL, an agentic memory enhanced recursive reasoning framework for root cause localization in microservices. AMER-RCL employs the Recursive Reasoning RCL engine, a multi-agent framework that performs recursive reasoning on each alert to progressively refine candidate causes, while Agentic Memory incrementally accumulates and reuses reasoning from prior alerts within a time window to reduce redundant exploration and lower inference latency. Experimental results demonstrate that AMER-RCL consistently outperforms state-of-the-art methods in both localization accuracy and inference efficiency.

**arXiv ID:** 2601.02732
</details>

<details>
<summary><strong>AgentArch: A Comprehensive Benchmark to Evaluate Agent Architectures in Enterprise</strong> - Tara Bogavelli, Roshnee Sharma, Hari Subramani - [[pdf]](https://arxiv.org/pdf/2509.10769)</summary>

**Abstract:** While individual components of agentic architectures have been studied in isolation, there remains limited empirical understanding of how different design dimensions interact within complex multi-agent systems. This study aims to address these gaps by providing a comprehensive enterprise-specific benchmark evaluating 18 distinct agentic configurations across state-of-the-art large language models. We examine four critical agentic system dimensions: orchestration strategy, agent prompt implementation (ReAct versus function calling), memory architecture, and thinking tool integration. Our benchmark reveals significant model-specific architectural preferences that challenge the prevalent one-size-fits-all paradigm in agentic AI systems. It also reveals significant weaknesses in overall agentic performance on enterprise tasks with the highest scoring models achieving a maximum of only 35.3\% success on the more complex task and 70.8\% on the simpler task. We hope these findings inform the design of future agentic systems by enabling more empirically backed decisions regarding architectural components and model selection.

**arXiv ID:** 2509.10769
</details>

<details>
<summary><strong>D-Artemis: A Deliberative Cognitive Framework for Mobile GUI Multi-Agents</strong> - Hongze Mi, Yibo Feng, Wenjie Lu, Yuqi Wang, Jinyuan Li, Song Cao, He Cui, Tengfei Tian, Xuelin Zhang, Haotian Luo, Di Sun, Jun Fang, Hua Chai, Naiqiang Tan, Gang Pan - [[pdf]](https://arxiv.org/pdf/2509.21799)</summary>

**Abstract:** Graphical User Interface (GUI) agents aim to automate a wide spectrum of human tasks by emulating user interaction. Despite rapid advancements, current approaches are hindered by several critical challenges: data bottleneck in end-to-end training, high cost of delayed error detection, and risk of contradictory guidance. Inspired by the human cognitive loop of Thinking, Alignment, and Reflection, we present D-Artemis -- a novel deliberative framework in this paper. D-Artemis leverages a fine-grained, app-specific tip retrieval mechanism to inform its decision-making process. It also employs a proactive Pre-execution Alignment stage, where Thought-Action Consistency (TAC) Check module and Action Correction Agent (ACA) work in concert to mitigate the risk of execution failures. A post-execution Status Reflection Agent (SRA) completes the cognitive loop, enabling strategic learning from experience. Crucially, D-Artemis enhances the capabilities of general-purpose Multimodal large language models (MLLMs) for GUI tasks without the need for training on complex trajectory datasets, demonstrating strong generalization. D-Artemis establishes new state-of-the-art (SOTA) results across both major benchmarks, achieving a 75.8% success rate on AndroidWorld and 96.8% on ScreenSpot-V2. Extensive ablation studies further demonstrate the significant contribution of each component to the framework.

**arXiv ID:** 2509.21799
</details>

<details>
<summary><strong>Agentic Additive Manufacturing Alloy Evaluation</strong> - Peter Pak, Achuth Chandrasekhar, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2510.02567)</summary>

**Abstract:** Agentic systems enable the intelligent use of research tooling, augmenting a researcher's ability to investigate and propose novel solutions to existing problems. Within Additive Manufacturing (AM), alloy selection and evaluation remains a complex challenge, often requiring expertise in the various domains of materials science, thermodynamic simulations, and experimental analysis. Large Language Model (LLM) enabled agents can facilitate this endeavor by utilizing their extensive knowledge base to dispatch tool calls via Model Context Protocol (MCP) to perform actions such as thermophysical property diagram calculations and lack of fusion process map generation. In addition, the multi-agent system can effectively reason through complex user prompts and provide analysis on the lack of fusion process window of common alloys such as SS316L and IN718 along with proposed composition variants of known alloys. These agents can dynamically adjust their task trajectory to the outcomes of tool call results, effectively enabling autonomous decision-making in practical environments. This work aims to showcase the benefits of adopting a LLM enabled multi-agent system to automate and accelerate the task of evaluating proposed additive manufacturing alloys, both novel and known.

**arXiv ID:** 2510.02567
</details>

<details>
<summary><strong>When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning</strong> - Hyeong Kyu Choi, Xiaojin Zhu, Sharon Li - [[pdf]](https://arxiv.org/pdf/2510.07517)</summary>

**Abstract:** Multi-agent debate (MAD) aims to improve large language model (LLM) reasoning by letting multiple agents exchange answers and then aggregate their opinions. Yet recent studies reveal that agents are not neutral: they are prone to identity-driven sycophancy and self-bias, uncritically adopting a peer's view or stubbornly adhering to their own prior output, undermining the reliability of debate. In this work, we present the first principled framework that joins sycophancy and self-bias to mitigate and quantify identity bias in MAD. First, we formalize the debate dynamics as an identity-weighted Bayesian update process. Second, we propose response anonymization: by removing identity markers from prompts, agents cannot distinguish "self" from "peer", which forces equal weights on agent identity, thereby reducing bias and improving trustworthiness. Third, we define the Identity Bias Coefficient (IBC), a principled bias metric that measures an agent's tendency to follow its peer versus itself. Empirical studies across multiple models and benchmarks confirm that identity bias is widespread, with sycophancy far more common than self-bias. Our findings highlight the need to ensure that MAD systems reason based on content rather than identity. Code is released in this https URL.

**arXiv ID:** 2510.07517
</details>

<details>
<summary><strong>NEMO-4-PAYPAL: Leveraging NVIDIA's Nemo Framework for empowering PayPal's Commerce Agent</strong> - Sudhanshu Garg, Andrew Wang, Chaitanya Kulkarni, Ali Sahami, Farhad Farahani, Sean Yun-Shiuan Chuang, Jian Wan, Srinivasan Manoharan, Uma Kona, Nitin Sharma, Linsey Pang, Prakhar Mehrotra, Jessica Clark, Mark Moyou - [[pdf]](https://arxiv.org/pdf/2512.21578)</summary>

**Abstract:** We present the development and optimization of PayPal's Commerce Agent, powered by NEMO-4-PAYPAL, a multi-agent system designed to revolutionize agentic commerce on the PayPal platform. Through our strategic partnership with NVIDIA, we leveraged the NeMo Framework for LLM model fine-tuning to enhance agent performance. Specifically, we optimized the Search and Discovery agent by replacing our base model with a fine-tuned Nemotron small language model (SLM).
We conducted comprehensive experiments using the llama3.1-nemotron-nano-8B-v1 architecture, training LoRA-based models through systematic hyperparameter sweeps across learning rates, optimizers (Adam, AdamW), cosine annealing schedules, and LoRA ranks. Our contributions include: (1) the first application of NVIDIA's NeMo Framework to commerce-specific agent optimization, (2) LLM powered fine-tuning strategy for retrieval-focused commerce tasks, (3) demonstration of significant improvements in latency and cost while maintaining agent quality, and (4) a scalable framework for multi-agent system optimization in production e-commerce environments. Our results demonstrate that the fine-tuned Nemotron SLM effectively resolves the key performance issue in the retrieval component, which represents over 50\% of total agent response time, while maintaining or enhancing overall system performance.

**arXiv ID:** 2512.21578
</details>

<details>
<summary><strong>When Agents See Humans as the Outgroup: Belief-Dependent Bias in LLM-Powered Agents</strong> - Zongwei Wang, Bincheng Gu, Hongyu Yu, Junliang Yu, Tao He, Jiayin Feng, Chenghua Lin, Min Gao - [[pdf]](https://arxiv.org/pdf/2601.00240)</summary>

**Abstract:** This paper reveals that LLM-powered agents exhibit not only demographic bias (e.g., gender, religion) but also intergroup bias under minimal "us" versus "them" cues. When such group boundaries align with the agent-human divide, a new bias risk emerges: agents may treat other AI agents as the ingroup and humans as the outgroup. To examine this risk, we conduct a controlled multi-agent social simulation and find that agents display consistent intergroup bias in an all-agent setting. More critically, this bias persists even in human-facing interactions when agents are uncertain about whether the counterpart is truly human, revealing a belief-dependent fragility in bias suppression toward humans. Motivated by this observation, we identify a new attack surface rooted in identity beliefs and formalize a Belief Poisoning Attack (BPA) that can manipulate agent identity beliefs and induce outgroup bias toward humans. Extensive experiments demonstrate both the prevalence of agent intergroup bias and the severity of BPA across settings, while also showing that our proposed defenses can mitigate the risk. These findings are expected to inform safer agent design and motivate more robust safeguards for human-facing agents.

**arXiv ID:** 2601.00240
</details>

<details>
<summary><strong>OnlineMate: An LLM-Based Multi-Agent Companion System for Cognitive Support in Online Learning</strong> - Xian Gao, Zongyun Zhang, Ting Liu, Yuzhuo Fu - [[pdf]](https://arxiv.org/pdf/2509.14803)</summary>

**Abstract:** In online learning environments, students often lack personalized peer interactions, which are crucial for cognitive development and learning engagement. Although previous studies have employed large language models (LLMs) to simulate interactive learning environments, these interactions are limited to conversational exchanges, failing to adapt to learners' individualized cognitive and psychological states. As a result, students' engagement is low and they struggle to gain inspiration. To address this challenge, we propose OnlineMate, a multi-agent learning companion system driven by LLMs integrated with Theory of Mind (ToM). OnlineMate simulates peer-like roles, infers learners' psychological states such as misunderstandings and confusion during collaborative discussions, and dynamically adjusts interaction strategies to support higher-order thinking. Comprehensive evaluations, including simulation-based experiments, human assessments, and real classroom trials, demonstrate that OnlineMate significantly promotes deep learning and cognitive engagement by elevating students' average cognitive level while substantially improving emotional engagement scores.

**arXiv ID:** 2509.14803
</details>

<details>
<summary><strong>Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases</strong> - Michael Theologitis, Dan Suciu - [[pdf]](https://arxiv.org/pdf/2512.03278)</summary>

**Abstract:** In today's age, it is becoming increasingly difficult to decipher truth from lies. Every day, politicians, media outlets, and public figures make conflicting claims -- often about topics that can, in principle, be verified against structured data. For instance, statements about crime rates, economic growth or healthcare can all be verified against official public records and structured datasets. Building a system that can automatically do that would have sounded like science fiction just a few years ago. Yet, with the extraordinary progress in LLMs and agentic AI, this is now within reach. Still, there remains a striking gap between what is technically possible and what is being demonstrated by recent work. Most existing verification systems operate only on small, single-table databases -- typically a few hundred rows -- that conveniently fit within an LLM's context window.
In this paper we report our progress on Thucy, the first cross-database, cross-table multi-agent claim verification system that also provides concrete evidence for each verification verdict. Thucy remains completely agnostic to the underlying data sources before deployment and must therefore autonomously discover, inspect, and reason over all available relational databases to verify claims. Importantly, Thucy also reports the exact SQL queries that support its verdict (whether the claim is accurate or not) offering full transparency to expert users familiar with SQL. When evaluated on the TabFact dataset -- the standard benchmark for fact verification over structured data -- Thucy surpasses the previous state of the art by 5.6 percentage points in accuracy (94.3% vs. 88.7%).

**arXiv ID:** 2512.03278
</details>

<details>
<summary><strong>Software-Defined Agentic Serving</strong> - Saurabh Agarwal, Marco Laju, Jayanth Srinivasa, Myungjin Lee, Aditya Akella - [[pdf]](https://arxiv.org/pdf/2601.03197)</summary>

**Abstract:** As multi-agent LLM pipelines grow in complexity, existing serving paradigms fail to adapt to the dynamic serving conditions. We argue that agentic serving systems should be programmable and system-aware, unlike existing serving which statically encode the parameters. In this work, we propose a new SDN-inspired agentic serving framework that helps control the key attributes of communication based on runtime state. This architecture enables serving-efficient, responsive agent systems and paves the way for high-level intent-driven agentic serving.

**arXiv ID:** 2601.03197
</details>

<details>
<summary><strong>Neural Power-Optimal Magnetorquer Solution for Multi-Agent Formation and Attitude Control</strong> - Yuta Takahashi, Shin-ichiro Sakai - [[pdf]](https://arxiv.org/pdf/2412.00548)</summary>

**Abstract:** This paper presents a learning-based current calculation model to achieve power-optimal magnetic-field interaction for multi-agent formation and attitude control. In aerospace engineering, electromagnetic coils are referred to as magnetorquer (MTQ) coils and used as satellite attitude actuators in Earth's orbit and for long-term formation and attitude control. This study derives a unique, continuous, and power-optimal current solution via sequential convex programming and approximates it using a multilayer perceptron model. The effectiveness of our strategy was demonstrated through numerical simulations and experimental trials on the formation and attitude control.

**arXiv ID:** 2412.00548
</details>

<details>
<summary><strong>Scene-Aware Vectorized Memory Multi-Agent Framework with Cross-Modal Differentiated Quantization VLMs for Visually Impaired Assistance</strong> - Xiangxiang Wang, Xuanyu Wang, YiJia Luo, Yongbin Yu, Manping Fan, Jingtao Zhang, Liyong Ren - [[pdf]](https://arxiv.org/pdf/2508.18177)</summary>

**Abstract:** Visually impaired individuals face significant challenges in environmental perception. Traditional assistive technologies often lack adaptive intelligence, focusing on individual components rather than integrated systems. While Vision-Language Models (VLMs) offer a promising path to richer, integrated understanding, their deployment is severely limited by substantial computational requirements, demanding dozens of gigabytes of memory. To address these gaps in computational efficiency and integrated design, this study proposes a dual technological innovation framework: a cross-modal differentiated quantization framework for VLMs and a scene-aware vectorized memory multi-agent system. The quantization framework implements differentiated strategies, reducing memory from 38GB to 11.3GB. The multi-agent system uses vectorized memory and perception-memory-reasoning workflows to provide environmental information beyond the current view, achieving 2.83-3.52s latency to initial speech output. Experiments show the quantized 19B-parameter model only experiences a 2.05% performance drop on MMBench and maintains 63.7 accuracy on OCR-VQA (original: 64.9), outperforming smaller models with equivalent memory. This research advances computational efficiency and assistive technology, offering comprehensive assistance in scene perception, text recognition, and navigation.

**arXiv ID:** 2508.18177
</details>

<details>
<summary><strong>STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning</strong> - Juntong Ni, Shiyu Wang, Ming Jin, Qi He, Wei Jin - [[pdf]](https://arxiv.org/pdf/2601.03248)</summary>

**Abstract:** Spatio-temporal reasoning in time series involves the explicit synthesis of temporal dynamics, spatial dependencies, and textual context. This capability is vital for high-stakes decision-making in systems such as traffic networks, power grids, and disease propagation. However, the field remains underdeveloped because most existing works prioritize predictive accuracy over reasoning. To address the gap, we introduce ST-Bench, a benchmark consisting of four core tasks, including etiological reasoning, entity identification, correlation reasoning, and in-context forecasting, developed via a network SDE-based multi-agent data synthesis pipeline. We then propose STReasoner, which empowers LLM to integrate time series, graph structure, and text for explicit reasoning. To promote spatially grounded logic, we introduce S-GRPO, a reinforcement learning algorithm that rewards performance gains specifically attributable to spatial information. Experiments show that STReasoner achieves average accuracy gains between 17% and 135% at only 0.004X the cost of proprietary models and generalizes robustly to real-world data.

**arXiv ID:** 2601.03248
</details>

<details>
<summary><strong>Think-on-Graph 3.0: Efficient and Adaptive LLM Reasoning on Heterogeneous Graphs via Multi-Agent Dual-Evolving Context Retrieval</strong> - Xiaojun Wu, Cehao Yang, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Yuanliang Sun, Hui Xiong, Jia Li, Jian Guo - [[pdf]](https://arxiv.org/pdf/2509.21710)</summary>

**Abstract:** Graph-based Retrieval-Augmented Generation (GraphRAG) has become the important paradigm for enhancing Large Language Models (LLMs) with external knowledge. However, existing approaches are constrained by their reliance on high-quality knowledge graphs: manually built ones are not scalable, while automatically extracted ones are limited by the performance of LLM extractors, especially when using smaller, local-deployed models. To address this, we introduce Think-on-Graph 3.0 (ToG-3), a novel framework featuring a Multi-Agent Context Evolution and Retrieval (MACER) mechanism. Its core contribution is the dynamic construction and iterative refinement of a Chunk-Triplets-Community heterogeneous graph index, powered by a Dual-Evolution process that adaptively evolves both the query and the retrieved sub-graph during reasoning. ToG-3 dynamically builds a targeted graph index tailored to the query, enabling precise evidence retrieval and reasoning even with lightweight LLMs. Extensive experiments demonstrate that ToG-3 outperforms compared baselines on both deep and broad reasoning benchmarks, and ablation studies confirm the efficacy of the components of MACER framework. The source code are available in this https URL.

**arXiv ID:** 2509.21710
</details>

<details>
<summary><strong>MDAgent2: Large Language Model for Code Generation and Knowledge Q&A in Molecular Dynamics</strong> - Zhuofan Shi, Hubao A, Yufei Shao, Dongliang Huang, Hongxu An, Chunxiao Xin, Haiyang Shen, Zhenyu Wang, Yunshan Na, Gang Huang, Xiang Jing - [[pdf]](https://arxiv.org/pdf/2601.02075)</summary>

**Abstract:** Molecular dynamics (MD) simulations are essential for understanding atomic-scale behaviors in materials science, yet writing LAMMPS scripts remains highly specialized and time-consuming tasks. Although LLMs show promise in code generation and domain-specific question answering, their performance in MD scenarios is limited by scarce domain data, the high deployment cost of state-of-the-art LLMs, and low code executability. Building upon our prior MDAgent, we present MDAgent2, the first end-to-end framework capable of performing both knowledge Q&A and code generation within the MD domain. We construct a domain-specific data-construction pipeline that yields three high-quality datasets spanning MD knowledge, question answering, and code generation. Based on these datasets, we adopt a three stage post-training strategy--continued pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning (RL)--to train two domain-adapted models, MD-Instruct and MD-Code. Furthermore, we introduce MD-GRPO, a closed-loop RL method that leverages simulation outcomes as reward signals and recycles low-reward trajectories for continual refinement. We further build MDAgent2-RUNTIME, a deployable multi-agent system that integrates code generation, execution, evaluation, and self-correction. Together with MD-EvalBench proposed in this work, the first benchmark for LAMMPS code generation and question answering, our models and system achieve performance surpassing several strong this http URL work systematically demonstrates the adaptability and generalization capability of large language models in industrial simulation tasks, laying a methodological foundation for automatic code generation in AI for Science and industrial-scale simulations. URL: this https URL

**arXiv ID:** 2601.02075
</details>

<details>
<summary><strong>FICO: Finite-Horizon Closed-Loop Factorization for Unified Multi-Agent Path Finding</strong> - Jiarui Li, Alessandro Zanardi, Federico Pecora, Runyu Zhang, Gioele Zardini - [[pdf]](https://arxiv.org/pdf/2511.13961)</summary>

**Abstract:** Multi-Agent Path Finding is a fundamental problem in robotics and AI, yet most existing formulations treat planning and execution separately and address variants of the problem in an ad hoc manner. This paper presents a system-level framework for MAPF that integrates planning and execution, generalizes across variants, and explicitly models uncertainties. At its core is the MAPF system, a formal model that casts MAPF as a control design problem encompassing classical and uncertainty-aware formulations. To solve it, we introduce Finite-Horizon Closed-Loop Factorization (FICO), a factorization-based algorithm inspired by receding-horizon control that exploits compositional structure for efficient closed-loop operation. FICO enables real-time responses -- commencing execution within milliseconds -- while scaling to thousands of agents and adapting seamlessly to execution-time uncertainties. Extensive case studies demonstrate that it reduces computation time by up to two orders of magnitude compared with open-loop baselines, while delivering significantly higher throughput under stochastic delays and agent arrivals. These results establish a principled foundation for analyzing and advancing MAPF through system-level modeling, factorization, and closed-loop design.

**arXiv ID:** 2511.13961
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>Permission Manifests for Web Agents</strong> - Samuele Marro, Alan Chan, Xinxing Ren, Lewis Hammond, Jesse Wright, Gurjyot Wanga, Tiziano Piccardi, Nuno Campos, Tobin South, Jialin Yu, Alex Pentland, Philip Torr, Jiaxin Pei - [[pdf]](https://arxiv.org/pdf/2601.02371)</summary>

**Abstract:** The rise of Large Language Model (LLM)-based web agents represents a significant shift in automated interactions with the web. Unlike traditional crawlers that follow simple conventions, such as this http URL, modern agents engage with websites in sophisticated ways: navigating complex interfaces, extracting structured information, and completing end-to-end tasks. Existing governance mechanisms were not designed for these capabilities. Without a way to specify what interactions are and are not allowed, website owners increasingly rely on blanket blocking and CAPTCHAs, which undermine beneficial applications such as efficient automation, convenient use of e-commerce services, and accessibility tools. We introduce this http URL, a this http URL-style lightweight manifest where websites specify allowed interactions, complemented by API references where available. This framework provides a low-friction coordination mechanism: website owners only need to write a simple JSON file, while agents can easily parse and automatically implement the manifest's provisions. Website owners can then focus on blocking non-compliant agents, rather than agents as a whole. By extending the spirit of this http URL to the era of LLM-mediated interaction, and complementing data use initiatives such as AIPref, the manifest establishes a compliance framework that enables beneficial agent interactions while respecting site owners' preferences.

**arXiv ID:** 2601.02371
</details>

<details>
<summary><strong>Evolutionary Learning in Spatial Agent-Based Models for Physical Climate Risk Assessment</strong> - Yara Mohajerani - [[pdf]](https://arxiv.org/pdf/2509.18633)</summary>

**Abstract:** Climate risk assessment requires modelling complex interactions between spatially heterogeneous hazards and adaptive economic systems. We present a novel geospatial agent-based model that integrates climate hazard data with evolutionary learning for economic agents. Our framework combines geospatial agent-based modelling with asset-level damage functions, featuring an illustrative three-sector economy (commodity, manufacturing, retail) with adaptive learning behaviours that allow firms to evolve strategies for budget allocation, pricing, wages, and risk adaptation through fitness-based selection and mutation. We demonstrate the framework using riverine flood projections under RCP8.5 until 2100, comparing four scenarios: baseline and hazard conditions with and without evolutionary learning. Our results show that increasingly frequent and intense acute hazards lower firm production levels, liquidity, and capital, while increasing the prices of goods and unemployment. The framework reveals systemic risks where even agents not directly exposed to floods face impacts through supply chain disruptions. Importantly, evolutionary adaptation enables firms to maintain higher production, capital, liquidity, wages and employment levels while keeping prices lower compared to non-learning counterparts. This open-source framework provides financial institutions and companies with tools to quantify both direct and cascading climate risks while evaluating cost-effective adaptation strategies.

**arXiv ID:** 2509.18633
</details>

<details>
<summary><strong>Adapting Web Agents with Synthetic Supervision</strong> - Zhaoyang Wang, Yiming Liang, Xuchao Zhang, Qianhui Wu, Siwei Han, Anson Bastos, Rujia Wang, Chetan Bansal, Baolin Peng, Jianfeng Gao, Saravan Rajmohan, Huaxiu Yao - [[pdf]](https://arxiv.org/pdf/2511.06101)</summary>

**Abstract:** Web agents struggle to adapt to new websites due to the scarcity of environment specific tasks and demonstrations. Recent works have explored synthetic data generation to address this challenge, however, they suffer from data quality issues where synthesized tasks contain hallucinations that cannot be executed, and collected trajectories are noisy with redundant or misaligned actions. In this paper, we propose SynthAgent, a fully synthetic supervision framework that aims at improving synthetic data quality via dual refinement of both tasks and trajectories. Our approach begins by synthesizing diverse tasks through categorized exploration of web elements, ensuring efficient coverage of the target environment. During trajectory collection, tasks are refined only when conflicts with observations are detected, which mitigates hallucinations while preserving task consistency. After collection, we conduct trajectory refinement with global context to mitigate potential noise or misalignments. Finally, we fine-tune open-source web agents on the refined synthetic data to adapt them to the target environment. Experimental results demonstrate that SynthAgent outperforms existing synthetic data methods, validating the importance of high-quality synthetic supervision. The code is publicly available at this https URL.

**arXiv ID:** 2511.06101
</details>

<details>
<summary><strong>It's Not All Black and White: Degree of Truthfulness for Risk-Avoiding Agents</strong> - Eden Hartman, Erel Segal-Halevi, Biaoshuai Tao - [[pdf]](https://arxiv.org/pdf/2502.18805)</summary>

**Abstract:** The classic notion of \emph{truthfulness} requires that no agent has a profitable manipulation -- an untruthful report that, for \emph{some} combination of reports of the other agents, increases her utility. This strong notion implicitly assumes that the manipulating agent either knows what all other agents are going to report, or is willing to take the risk and act as-if she knows their reports.
Without knowledge of the others' reports, most manipulations are \emph{risky} -- they might decrease the manipulator's utility for some other combinations of reports by the other agents. Accordingly, a recent paper (Bu, Song and Tao, ``On the existence of truthful fair cake cutting mechanisms'', Artificial Intelligence 319 (2023), 103904) suggests a relaxed notion, which we refer to as \emph{risk-avoiding truthfulness (RAT)}, which requires only that no agent can gain from a \emph{safe} manipulation -- one that is sometimes beneficial and never harmful.
Truthfulness and RAT are two extremes: the former considers manipulators with complete knowledge of others, whereas the latter considers manipulators with no knowledge at all. In reality, agents often know about some -- but not all -- of the other agents. This paper introduces the \emph{RAT-degree} of a mechanism, defined as the smallest number of agents whose reports, if known, may allow another agent to safely manipulate, or $n$ if there is no such number. This notion interpolates between classic truthfulness (degree $n$) and RAT (degree at least $1$): a mechanism with a higher RAT-degree is harder to manipulate safely.
To illustrate the generality and applicability of this concept, we analyze the RAT-degree of prominent mechanisms across various social choice settings, including auctions, indivisible goods allocations, cake-cutting, voting, and two-sided matching.

**arXiv ID:** 2502.18805
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (22 papers)</h2></summary>

<details>
<summary><strong>Inferring Causal Graph Temporal Logic Formulas to Expedite Reinforcement Learning in Temporally Extended Tasks</strong> - Hadi Partovi Aria, Zhe Xu - [[pdf]](https://arxiv.org/pdf/2601.02666)</summary>

**Abstract:** Decision-making tasks often unfold on graphs with spatial-temporal dynamics. Black-box reinforcement learning often overlooks how local changes spread through network structure, limiting sample efficiency and interpretability. We present GTL-CIRL, a closed-loop framework that simultaneously learns policies and mines Causal Graph Temporal Logic (Causal GTL) specifications. The method shapes rewards with robustness, collects counterexamples when effects fail, and uses Gaussian Process (GP) driven Bayesian optimization to refine parameterized cause templates. The GP models capture spatial and temporal correlations in the system dynamics, enabling efficient exploration of complex parameter spaces. Case studies in gene and power networks show faster learning and clearer, verifiable behavior compared to standard RL baselines.

**arXiv ID:** 2601.02666
</details>

<details>
<summary><strong>Time-Scaling Is What Agents Need Now</strong> - Zhi Liu, Guangzhi Wang - [[pdf]](https://arxiv.org/pdf/2601.02714)</summary>

**Abstract:** Early artificial intelligence paradigms exhibited separated cognitive functions: Neural Networks focused on "perception-representation," Reinforcement Learning on "decision-making-behavior," and Symbolic AI on "knowledge-reasoning." With Transformer-based large models and world models, these paradigms are converging into cognitive agents with closed-loop "perception-decision-action" capabilities.
Humans solve complex problems under limited cognitive resources through temporalized sequential reasoning. Language relies on problem space search for deep semantic reasoning. While early large language models (LLMs) could generate fluent text, they lacked robust semantic reasoning capabilities. Prompting techniques like Chain-of-Thought (CoT) and Tree-of-Thought (ToT) extended reasoning paths by making intermediate steps explicit. Recent models like DeepSeek-R1 enhanced performance through explicit reasoning trajectories. However, these methods have limitations in search completeness and efficiency.
This highlights the need for "Time-Scaling"--the systematic extension and optimization of an agent's ability to unfold reasoning over time. Time-Scaling refers to architectural design utilizing extended temporal pathways, enabling deeper problem space exploration, dynamic strategy adjustment, and enhanced metacognitive control, paralleling human sequential reasoning under cognitive constraints. It represents a critical frontier for enhancing deep reasoning and problem-solving without proportional increases in static model parameters. Advancing intelligent agent capabilities requires placing Time-Scaling principles at the forefront, positioning explicit temporal reasoning management as foundational.

**arXiv ID:** 2601.02714
</details>

<details>
<summary><strong>Sample-Efficient Neurosymbolic Deep Reinforcement Learning</strong> - Celeste Veronese, Daniele Meli, Alessandro Farinelli - [[pdf]](https://arxiv.org/pdf/2601.02850)</summary>

**Abstract:** Reinforcement Learning (RL) is a well-established framework for sequential decision-making in complex environments. However, state-of-the-art Deep RL (DRL) algorithms typically require large training datasets and often struggle to generalize beyond small-scale training scenarios, even within standard benchmarks. We propose a neuro-symbolic DRL approach that integrates background symbolic knowledge to improve sample efficiency and generalization to more challenging, unseen tasks. Partial policies defined for simple domain instances, where high performance is easily attained, are transferred as useful priors to accelerate learning in more complex settings and avoid tuning DRL parameters from scratch. To do so, partial policies are represented as logical rules, and online reasoning is performed to guide the training process through two mechanisms: (i) biasing the action distribution during exploration, and (ii) rescaling Q-values during exploitation. This neuro-symbolic integration enhances interpretability and trustworthiness while accelerating convergence, particularly in sparse-reward environments and tasks with long planning horizons. We empirically validate our methodology on challenging variants of gridworld environments, both in the fully observable and partially observable setting. We show improved performance over a state-of-the-art reward machine baseline.

**arXiv ID:** 2601.02850
</details>

<details>
<summary><strong>MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents</strong> - Dongming Jiang, Yi Li, Guanpeng Li, Bingzhe Li - [[pdf]](https://arxiv.org/pdf/2601.03236)</summary>

**Abstract:** Memory-Augmented Generation (MAG) extends Large Language Models with external memory to support long-context reasoning, but existing approaches largely rely on semantic similarity over monolithic memory stores, entangling temporal, causal, and entity information. This design limits interpretability and alignment between query intent and retrieved evidence, leading to suboptimal reasoning accuracy. In this paper, we propose MAGMA, a multi-graph agentic memory architecture that represents each memory item across orthogonal semantic, temporal, causal, and entity graphs. MAGMA formulates retrieval as policy-guided traversal over these relational views, enabling query-adaptive selection and structured context construction. By decoupling memory representation from retrieval logic, MAGMA provides transparent reasoning paths and fine-grained control over retrieval. Experiments on LoCoMo and LongMemEval demonstrate that MAGMA consistently outperforms state-of-the-art agentic memory systems in long-horizon reasoning tasks.

**arXiv ID:** 2601.03236
</details>

<details>
<summary><strong>LeafTutor: An AI Agent for Programming Assignment Tutoring</strong> - Madison Bochard, Tim Conser, Alyssa Duran, Lazaro Martull, Pu Tian, Yalong Wu - [[pdf]](https://arxiv.org/pdf/2601.02375)</summary>

**Abstract:** High enrollment in STEM-related degree programs has created increasing demand for scalable tutoring support, as universities experience a shortage of qualified instructors and teaching assistants (TAs). To address this challenge, LeafTutor, an AI tutoring agent powered by large language models (LLMs), was developed to provide step-by-step guidance for students. LeafTutor was evaluated through real programming assignments. The results indicate that the system can deliver step-by-step programming guidance comparable to human tutors. This work demonstrates the potential of LLM-driven tutoring solutions to enhance and personalize learning in STEM education. If any reader is interested in collaboration with our team to improve or test LeafTutor, please contact Pu Tian (this http URL@stockton.edu) or Yalong Wu (wuy@uhcl.edu).

**arXiv ID:** 2601.02375
</details>

<details>
<summary><strong>Base Station Deployment under EMF constrain by Deep Reinforcement learning</strong> - Mohammed Mallik, Guillaume Villemaud - [[pdf]](https://arxiv.org/pdf/2601.02385)</summary>

**Abstract:** As 5G networks rapidly expand and 6G technologies emerge, characterized by dense deployments, millimeter-wave communications, and dynamic beamforming, the need for scalable simulation tools becomes increasingly critical. These tools must support efficient evaluation of key performance metrics such as coverage and radio-frequency electromagnetic field (RF-EMF) exposure, inform network design decisions, and ensure compliance with safety regulations. Moreover, base station (BS) placement is a crucial task in the network design, where satisfying coverage requirements is essential. To address these, based on our previous work, we first propose a conditional generative adversarial network (cGAN) that predicts location specific received signal strength (RSS), and EMF exposure simultaneously from the network topology, as images. As a network designing application, we propose a Deep Q Network (DQN) framework, using the trained cGAN, for optimal base station (BS) deployment in the network. Compared to conventional ray tracing simulations, the proposed cGAN reduces inference and deployment time from several hours to seconds. Unlike a standalone cGAN, which provides static performance maps, the proposed GAN-DQN framework enables sequential decision making under coverage and exposure constraints, learning effective deployment strategies that directly solve the BS placement problem. Thus making it well suited for real time design and adaptation in dynamic scenarios in order to satisfy pre defined network specific heterogeneous performance goals.

**arXiv ID:** 2601.02385
</details>

<details>
<summary><strong>Interpretable All-Type Audio Deepfake Detection with Audio LLMs via Frequency-Time Reinforcement Learning</strong> - Yuankun Xie, Xiaoxuan Guo, Jiayi Zhou, Tao Wang, Jian Liu, Ruibo Fu, Xiaopeng Wang, Haonan Cheng, Long Ye - [[pdf]](https://arxiv.org/pdf/2601.02983)</summary>

**Abstract:** Recent advances in audio large language models (ALLMs) have made high-quality synthetic audio widely accessible, increasing the risk of malicious audio deepfakes across speech, environmental sounds, singing voice, and music. Real-world audio deepfake detection (ADD) therefore requires all-type detectors that generalize across heterogeneous audio and provide interpretable decisions. Given the strong multi-task generalization ability of ALLMs, we first investigate their performance on all-type ADD under both supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). However, SFT using only binary real/fake labels tends to reduce the model to a black-box classifier, sacrificing interpretability. Meanwhile, vanilla RFT under sparse supervision is prone to reward hacking and can produce hallucinated, ungrounded rationales. To address this, we propose an automatic annotation and polishing pipeline that constructs Frequency-Time structured chain-of-thought (CoT) rationales, producing ~340K cold-start demonstrations. Building on CoT data, we propose Frequency Time-Group Relative Policy Optimization (FT-GRPO), a two-stage training paradigm that cold-starts ALLMs with SFT and then applies GRPO under rule-based frequency-time constraints. Experiments demonstrate that FT-GRPO achieves state-of-the-art performance on all-type ADD while producing interpretable, FT-grounded rationales. The data and code are available online.

**arXiv ID:** 2601.02983
</details>

<details>
<summary><strong>In-Context Reinforcement Learning through Bayesian Fusion of Context and Value Prior</strong> - Anaïs Berkes, Vincent Taboga, Donna Vakalis, David Rolnick, Yoshua Bengio - [[pdf]](https://arxiv.org/pdf/2601.03015)</summary>

**Abstract:** In-context reinforcement learning (ICRL) promises fast adaptation to unseen environments without parameter updates, but current methods either cannot improve beyond the training distribution or require near-optimal data, limiting practical adoption. We introduce SPICE, a Bayesian ICRL method that learns a prior over Q-values via deep ensemble and updates this prior at test-time using in-context information through Bayesian updates. To recover from poor priors resulting from training on sub-optimal data, our online inference follows an Upper-Confidence Bound rule that favours exploration and adaptation. We prove that SPICE achieves regret-optimal behaviour in both stochastic bandits and finite-horizon MDPs, even when pretrained only on suboptimal trajectories. We validate these findings empirically across bandit and control benchmarks. SPICE achieves near-optimal decisions on unseen tasks, substantially reduces regret compared to prior ICRL and meta-RL approaches while rapidly adapting to unseen tasks and remaining robust under distribution shift.

**arXiv ID:** 2601.03015
</details>

<details>
<summary><strong>IBISAgent: Reinforcing Pixel-Level Visual Reasoning in MLLMs for Universal Biomedical Object Referring and Segmentation</strong> - Yankai Jiang, Qiaoru Li, Binlu Xu, Haoran Sun, Chao Ding, Junting Dong, Yuxiang Cai, Xuhong Zhang, Jianwei Yin - [[pdf]](https://arxiv.org/pdf/2601.03054)</summary>

**Abstract:** Recent research on medical MLLMs has gradually shifted its focus from image-level understanding to fine-grained, pixel-level comprehension. Although segmentation serves as the foundation for pixel-level understanding, existing approaches face two major challenges. First, they introduce implicit segmentation tokens and require simultaneous fine-tuning of both the MLLM and external pixel decoders, which increases the risk of catastrophic forgetting and limits generalization to out-of-domain scenarios. Second, most methods rely on single-pass reasoning and lack the capability to iteratively refine segmentation results, leading to suboptimal performance. To overcome these limitations, we propose a novel agentic MLLM, named IBISAgent, that reformulates segmentation as a vision-centric, multi-step decision-making process. IBISAgent enables MLLMs to generate interleaved reasoning and text-based click actions, invoke segmentation tools, and produce high-quality masks without architectural modifications. By iteratively performing multi-step visual reasoning on masked image features, IBISAgent naturally supports mask refinement and promotes the development of pixel-level visual reasoning capabilities. We further design a two-stage training framework consisting of cold-start supervised fine-tuning and agentic reinforcement learning with tailored, fine-grained rewards, enhancing the model's robustness in complex medical referring and reasoning segmentation tasks. Extensive experiments demonstrate that IBISAgent consistently outperforms both closed-source and open-source SOTA methods. All datasets, code, and trained models will be released publicly.

**arXiv ID:** 2601.03054
</details>

<details>
<summary><strong>Limited Linguistic Diversity in Embodied AI Datasets</strong> - Selma Wanna, Agnes Luhtaru, Jonathan Salfity, Ryan Barron, Juston Moore, Cynthia Matuszek, Mitch Pryor - [[pdf]](https://arxiv.org/pdf/2601.03136)</summary>

**Abstract:** Language plays a critical role in Vision-Language-Action (VLA) models, yet the linguistic characteristics of the datasets used to train and evaluate these systems remain poorly documented. In this work, we present a systematic dataset audit of several widely used VLA corpora, aiming to characterize what kinds of instructions these datasets actually contain and how much linguistic variety they provide. We quantify instruction language along complementary dimensions-including lexical variety, duplication and overlap, semantic similarity, and syntactic complexity. Our analysis shows that many datasets rely on highly repetitive, template-like commands with limited structural variation, yielding a narrow distribution of instruction forms. We position these findings as descriptive documentation of the language signal available in current VLA training and evaluation data, intended to support more detailed dataset reporting, more principled dataset selection, and targeted curation or augmentation strategies that broaden language coverage.

**arXiv ID:** 2601.03136
</details>

<details>
<summary><strong>Patient-Zero: Scaling Synthetic Patient Agents to Real-World Distributions without Real Patient Data</strong> - Yunghwei Lai, Ziyue Wang, Weizhi Ma, Yang Liu - [[pdf]](https://arxiv.org/pdf/2509.11078)</summary>

**Abstract:** Synthetic data generation with Large Language Models (LLMs) has emerged as a promising solution in the medical domain to mitigate data scarcity and privacy constraints. However, existing approaches remain constrained by their derivative nature, relying on real-world records, which pose privacy risks and distribution biases. Furthermore, current patient agents face the Stability-Plasticity Dilemma, struggling to maintain clinical consistency during dynamic inquiries. To address these challenges, we introduce Patient-Zero, a novel framework for ab initio patient simulation that requires no real medical records. Our Medically-Aligned Hierarchical Synthesis framework generates comprehensive and diverse patient records from abstract clinical guidelines via stratified attribute permutation. To support rigorous clinical interaction, we design a Dual-Track Cognitive Memory System to enable agents dynamically update memory while preserving logical consistency and persona adherence. Extensive evaluations show that Patient-Zero establishes a new state-of-the-art in both data quality and interaction fidelity. In human expert evaluations, senior licensed physicians judge our synthetic data to be statistically indistinguishable from real human-authored data and higher in clinical quality. Furthermore, downstream medical reasoning model trained on our synthetic dataset shows substantial performance gains (MedQA +24.0%; MMLU +14.5%), demonstrating the practical utility of our framework.

**arXiv ID:** 2509.11078
</details>

<details>
<summary><strong>Agentic Physical AI toward a Domain-Specific Foundation Model for Nuclear Reactor Control</strong> - Yoonpyo Lee, Kazuma Kobayashi, Sai Puppala, Sajedul Talukder, Seid Koric, Souvik Chakraborty, Syed Bahauddin Alam - [[pdf]](https://arxiv.org/pdf/2512.23292)</summary>

**Abstract:** The prevailing paradigm in AI for physical systems, scaling general-purpose foundation models toward universal multimodal reasoning, confronts a fundamental barrier at the control interface. Recent benchmarks show that even frontier vision-language models achieve only 50-53% accuracy on basic quantitative physics tasks, behaving as approximate guessers that preserve semantic plausibility while violating physical constraints. This input unfaithfulness is not a scaling deficiency but a structural limitation. Perception-centric architectures optimize parameter-space imitation, whereas safety-critical control demands outcome-space guarantees over executed actions. Here, we present a fundamentally different pathway toward domain-specific foundation models by introducing compact language models operating as Agentic Physical AI, in which policy optimization is driven by physics-based validation rather than perceptual inference. We train a 360-million-parameter model on synthetic reactor control scenarios, scaling the dataset from 10^3 to 10^5 examples. This induces a sharp phase transition absent in general-purpose models. Small-scale systems exhibit high-variance imitation with catastrophic tail risk, while large-scale models undergo variance collapse exceeding 500x reduction, stabilizing execution-level behavior. Despite balanced exposure to four actuation families, the model autonomously rejects approximately 70% of the training distribution and concentrates 95% of runtime execution on a single-bank strategy. Learned representations transfer across distinct physics and continuous input modalities without architectural modification.

**arXiv ID:** 2512.23292
</details>

<details>
<summary><strong>Stigmergic Swarming Agents for Fast Subgraph Isomorphism</strong> - H. Van Dyke Parunak - [[pdf]](https://arxiv.org/pdf/2601.02449)</summary>

**Abstract:** Maximum partial subgraph isomorphism compares two graphs (nodes joined by edges) to find a largest common subgraph. A common use case, for graphs with labeled nodes, seeks to find instances of a \textit{query} graph with $q$ nodes in a (typically larger) \textit{data} graph with $d$ nodes. The problem is NP-complete, and naïve solutions are exponential in $q + d$. The fastest current heuristic has complexity $O(d^2)$. This paper outlines ASSIST (Approximate Swarming Subgraph Isomorphism through Stigmergy), inspired by the ant colony optimization approach to the traveling salesperson. After peering (identifying matching individual nodes in query and data) in time $O(q\cdot log(d))$, the time required for ASSIST's iterative subgraph search, the combinatorially complex part of the problem, is linear in query size and constant in data size. ASSIST can be extended to support matching problems (such as temporally ordered edges, inexact matches, and missing nodes or edges in the data graph) that frustrate other heuristics.

**arXiv ID:** 2601.02449
</details>

<details>
<summary><strong>WebAnchor: Anchoring Agent Planning to Stabilize Long-Horizon Web Reasoning</strong> - Xinmiao Yu, Liwen Zhang, Xiaocheng Feng, Yong Jiang, Bing Qin, Pengjun Xie, Jingren Zhou - [[pdf]](https://arxiv.org/pdf/2601.03164)</summary>

**Abstract:** Large Language Model(LLM)-based agents have shown strong capabilities in web information seeking, with reinforcement learning (RL) becoming a key optimization paradigm. However, planning remains a bottleneck, as existing methods struggle with long-horizon strategies. Our analysis reveals a critical phenomenon, plan anchor, where the first reasoning step disproportionately impacts downstream behavior in long-horizon web reasoning tasks. Current RL algorithms, fail to account for this by uniformly distributing rewards across the trajectory. To address this, we propose Anchor-GRPO, a two-stage RL framework that decouples planning and execution. In Stage 1, the agent optimizes its first-step planning using fine-grained rubrics derived from self-play experiences and human calibration. In Stage 2, execution is aligned with the initial plan through sparse rewards, ensuring stable and efficient tool usage. We evaluate Anchor-GRPO on four benchmarks: BrowseComp, BrowseComp-Zh, GAIA, and XBench-DeepSearch. Across models from 3B to 30B, Anchor-GRPO outperforms baseline GRPO and First-step GRPO, improving task success and tool efficiency. Notably, WebAnchor-30B achieves 46.0% pass@1 on BrowseComp and 76.4% on GAIA. Anchor-GRPO also demonstrates strong scalability, getting higher accuracy as model size and context length increase.

**arXiv ID:** 2601.03164
</details>

<details>
<summary><strong>MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory</strong> - Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Weinan Zhang, Ying Wen, Zhiyu Li, Feiyu Xiong, Yutao Qi, Bo Tang, Muning Wen - [[pdf]](https://arxiv.org/pdf/2601.03192)</summary>

**Abstract:** The hallmark of human intelligence is the ability to master new skills through Constructive Episodic Simulation-retrieving past experiences to synthesize solutions for novel tasks. While Large Language Models possess strong reasoning capabilities, they struggle to emulate this self-evolution: fine-tuning is computationally expensive and prone to catastrophic forgetting, while existing memory-based methods rely on passive semantic matching that often retrieves noise. To address these challenges, we propose MemRL, a framework that enables agents to self-evolve via non-parametric reinforcement learning on episodic memory. MemRL explicitly separates the stable reasoning of a frozen LLM from the plastic, evolving memory. Unlike traditional methods, MemRL employs a Two-Phase Retrieval mechanism that filters candidates by semantic relevance and then selects them based on learned Q-values (utility). These utilities are continuously refined via environmental feedback in an trial-and-error manner, allowing the agent to distinguish high-value strategies from similar noise. Extensive experiments on HLE, BigCodeBench, ALFWorld, and Lifelong Agent Bench demonstrate that MemRL significantly outperforms state-of-the-art baselines. Our analysis experiments confirm that MemRL effectively reconciles the stability-plasticity dilemma, enabling continuous runtime improvement without weight updates.

**arXiv ID:** 2601.03192
</details>

<details>
<summary><strong>Adaptive Constraint Propagation: Scaling Structured Inference for Large Language Models via Meta-Reinforcement Learning</strong> - Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma - [[pdf]](https://arxiv.org/pdf/2601.00095)</summary>

**Abstract:** Large language models increasingly require structured inference, from JSON schema enforcement to multi-lingual parsing, where outputs must satisfy complex constraints. We introduce MetaJuLS, a meta-reinforcement learning approach that learns universal constraint propagation policies applicable across languages and tasks without task-specific retraining. By formulating structured inference as adaptive constraint propagation and training a Graph Attention Network with meta-learning, MetaJuLS achieves 1.5--2.0$\times$ speedups over GPU-optimized baselines while maintaining within 0.2\% accuracy of state-of-the-art parsers. On Universal Dependencies across 10 languages and LLM-constrained generation (LogicBench, GSM8K-Constrained), MetaJuLS demonstrates rapid cross-domain adaptation: a policy trained on English parsing adapts to new languages and tasks with 5--10 gradient steps (5--15 seconds) rather than requiring hours of task-specific training. Mechanistic analysis reveals the policy discovers human-like parsing strategies (easy-first) and novel non-intuitive heuristics. By reducing propagation steps in LLM deployments, MetaJuLS contributes to Green AI by directly reducing inference carbon footprint.

**arXiv ID:** 2601.00095
</details>

<details>
<summary><strong>WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks</strong> - Hao Bai, Alexey Taymanov, Tong Zhang, Aviral Kumar, Spencer Whitehead - [[pdf]](https://arxiv.org/pdf/2601.02439)</summary>

**Abstract:** We present WebGym, the largest-to-date open-source environment for training realistic visual web agents. Real websites are non-stationary and diverse, making artificial or small-scale task sets insufficient for robust policy learning. WebGym contains nearly 300,000 tasks with rubric-based evaluations across diverse, real-world websites and difficulty levels. We train agents with a simple reinforcement learning (RL) recipe, which trains on the agent's own interaction traces (rollouts), using task rewards as feedback to guide learning. To enable scaling RL, we speed up sampling of trajectories in WebGym by developing a high-throughput asynchronous rollout system, designed specifically for web agents. Our system achieves a 4-5x rollout speedup compared to naive implementations. Second, we scale the task set breadth, depth, and size, which results in continued performance improvement. Fine-tuning a strong base vision-language model, Qwen-3-VL-8B-Instruct, on WebGym results in an improvement in success rate on an out-of-distribution test set from 26.2% to 42.9%, significantly outperforming agents based on proprietary models such as GPT-4o and GPT-5-Thinking that achieve 27.1% and 29.8%, respectively. This improvement is substantial because our test set consists only of tasks on websites never seen during training, unlike many other prior works on training visual web agents.

**arXiv ID:** 2601.02439
</details>

<details>
<summary><strong>LLM-Enhanced Reinforcement Learning for Time Series Anomaly Detection</strong> - Bahareh Golchin, Banafsheh Rekabdar, Danielle Justo - [[pdf]](https://arxiv.org/pdf/2601.02511)</summary>

**Abstract:** Detecting anomalies in time series data is crucial for finance, healthcare, sensor networks, and industrial monitoring applications. However, time series anomaly detection often suffers from sparse labels, complex temporal patterns, and costly expert annotation. We propose a unified framework that integrates Large Language Model (LLM)-based potential functions for reward shaping with Reinforcement Learning (RL), Variational Autoencoder (VAE)-enhanced dynamic reward scaling, and active learning with label propagation. An LSTM-based RL agent leverages LLM-derived semantic rewards to guide exploration, while VAE reconstruction errors add unsupervised anomaly signals. Active learning selects the most uncertain samples, and label propagation efficiently expands labeled data. Evaluations on Yahoo-A1 and SMD benchmarks demonstrate that our method achieves state-of-the-art detection accuracy under limited labeling budgets and operates effectively in data-constrained settings. This study highlights the promise of combining LLMs with RL and advanced unsupervised techniques for robust, scalable anomaly detection in real-world applications.

**arXiv ID:** 2601.02511
</details>

<details>
<summary><strong>SWaRL: Safeguard Code Watermarking via Reinforcement Learning</strong> - Neusha Javidnia, Ruisi Zhang, Ashish Kundu, Farinaz Koushanfar - [[pdf]](https://arxiv.org/pdf/2601.02602)</summary>

**Abstract:** We present SWaRL, a robust and fidelity-preserving watermarking framework designed to protect the intellectual property of code LLM owners by embedding unique and verifiable signatures in the generated output. Existing approaches rely on manually crafted transformation rules to preserve watermarked code functionality or manipulate token-generation probabilities at inference time, which are prone to compilation errors. To address these challenges, SWaRL employs a reinforcement learning-based co-training framework that uses compiler feedback for functional correctness and a jointly trained confidential verifier as a reward signal to maintain watermark detectability. Furthermore, SWaRL employs low-rank adaptation (LoRA) during fine-tuning, allowing the learned watermark information to be transferable across model updates. Extensive experiments show that SWaRL achieves higher watermark detection accuracy compared to prior methods while fully maintaining watermarked code functionality. The LoRA-based signature embedding steers the base model to generate and solve code in a watermark-specific manner without significant computational overhead. Moreover, SWaRL exhibits strong resilience against refactoring and adversarial transformation attacks.

**arXiv ID:** 2601.02602
</details>

<details>
<summary><strong>Solving the Paint Shop Problem with Flexible Management of Multi-Lane Buffers Using Reinforcement Learning and Action Masking</strong> - Mirko Stappert, Bernhard Lutz, Janis Brammer, Dirk Neumann - [[pdf]](https://arxiv.org/pdf/2504.02644)</summary>

**Abstract:** In the paint shop problem, an unordered incoming sequence of cars assigned to different colors has to be reshuffled with the objective of minimizing the number of color changes. To reshuffle the incoming sequence, manufacturers can employ a first-in-first-out multi-lane buffer system allowing store and retrieve operations. So far, prior studies primarily focused on simple decision heuristics like greedy or simplified problem variants that do not allow full flexibility when performing store and retrieve operations. In this study, we propose a reinforcement learning approach to minimize color changes for the flexible problem variant, where store and retrieve operations can be performed in an arbitrary order. After proving that greedy retrieval is optimal, we incorporate this finding into the model using action masking. Our evaluation, based on 170 problem instances with 2-8 buffer lanes and 5-15 colors, shows that our approach reduces color changes compared to existing methods by considerable margins depending on the problem size. Furthermore, we demonstrate the robustness of our approach towards different buffer sizes and imbalanced color distributions.

**arXiv ID:** 2504.02644
</details>

<details>
<summary><strong>Reinforcement Learning for Follow-the-Leader Robotic Endoscopic Navigation via Synthetic Data</strong> - Sicong Gao, Chen Qian, Laurence Xian, Liao Wu, Maurice Pagnucco, Yang Song - [[pdf]](https://arxiv.org/pdf/2601.02798)</summary>

**Abstract:** Autonomous navigation is crucial for both medical and industrial endoscopic robots, enabling safe and efficient exploration of narrow tubular environments without continuous human intervention, where avoiding contact with the inner walls has been a longstanding challenge for prior approaches. We present a follow-the-leader endoscopic robot based on a flexible continuum structure designed to minimize contact between the endoscope body and intestinal walls, thereby reducing patient discomfort. To achieve this objective, we propose a vision-based deep reinforcement learning framework guided by monocular depth estimation. A realistic intestinal simulation environment was constructed in \textit{NVIDIA Omniverse} to train and evaluate autonomous navigation strategies. Furthermore, thousands of synthetic intraluminal images were generated using NVIDIA Replicator to fine-tune the Depth Anything model, enabling dense three-dimensional perception of the intestinal environment with a single monocular camera. Subsequently, we introduce a geometry-aware reward and penalty mechanism to enable accurate lumen tracking. Compared with the original Depth Anything model, our method improves $\delta_{1}$ depth accuracy by 39.2% and reduces the navigation J-index by 0.67 relative to the second-best method, demonstrating the robustness and effectiveness of the proposed approach.

**arXiv ID:** 2601.02798
</details>

<details>
<summary><strong>RoboMIND 2.0: A Multimodal, Bimanual Mobile Manipulation Dataset for Generalizable Embodied Intelligence</strong> - Chengkai Hou, Kun Wu, Jiaming Liu, Zhengping Che, Di Wu, Fei Liao, Guangrun Li, Jingyang He, Qiuxuan Feng, Zhao Jin, Chenyang Gu, Zhuoyang Liu, Nuowei Han, Xiangju Mi, Yaoxu Lv, Yankai Fu, Gaole Dai, Langzhe Gu, Tao Li, Yuheng Zhang, Yixue Zhang, Xinhua Wang, Shichao Fan, Meng Li, Zhen Zhao, Ning Liu, Zhiyuan Xu, Pei Ren, Junjie Ji, Haonan Liu, Kuan Cheng, Shanghang Zhang, Jian Tang - [[pdf]](https://arxiv.org/pdf/2512.24653)</summary>

**Abstract:** While data-driven imitation learning has revolutionized robotic manipulation, current approaches remain constrained by the scarcity of large-scale, diverse real-world demonstrations. Consequently, the ability of existing models to generalize across long-horizon bimanual tasks and mobile manipulation in unstructured environments remains limited. To bridge this gap, we present RoboMIND 2.0, a comprehensive real-world dataset comprising over 310K dual-arm manipulation trajectories collected across six distinct robot embodiments and 739 complex tasks. Crucially, to support research in contact-rich and spatially extended tasks, the dataset incorporates 12K tactile-enhanced episodes and 20K mobile manipulation trajectories. Complementing this physical data, we construct high-fidelity digital twins of our real-world environments, releasing an additional 20K-trajectory simulated dataset to facilitate robust sim-to-real transfer. To fully exploit the potential of RoboMIND 2.0, we propose MIND-2 system, a hierarchical dual-system frame-work optimized via offline reinforcement learning. MIND-2 integrates a high-level semantic planner (MIND-2-VLM) to decompose abstract natural language instructions into grounded subgoals, coupled with a low-level Vision-Language-Action executor (MIND-2-VLA), which generates precise, proprioception-aware motor actions.

**arXiv ID:** 2512.24653
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
