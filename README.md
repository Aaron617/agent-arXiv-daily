# Agent arXiv Daily

**Last Updated:** 2026-04-17 03:41:12

**Total Papers:** 90

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (5 papers)</h2></summary>

<details>
<summary><strong>NuHF Claw: A Risk Constrained Cognitive Agent Framework for Human Centered Procedure Support in Digital Nuclear Control Rooms</strong> - Xingyu Xiao, Jiejuan Tong, Jun Sun, Zhe Sui, Peng Chen, Jingang Liang, Haitao Wang - [[pdf]](https://arxiv.org/pdf/2604.14160)</summary>

**Abstract:** The rapid digitization of nuclear power plant main control rooms has fundamentally reshaped operator interaction patterns, introducing complex soft-control behaviors and elevated cognitive risks that are not adequately addressed by existing human reliability analysis approaches. Although recent advances in large language models and autonomous agents offer new opportunities for intelligent decision support, their deployment in safety critical environments remains constrained by risks of hallucinated reasoning and weakened human authority. This study proposes NuHF Claw, a persistent cognitive-risk agent framework that enables risk governed human centered autonomy for digital nuclear operations. The core methodological innovation lies in the introduction of a risk constrained agent runtime, which tightly couples cognitive state inference with probabilistic safety assessment to regulate autonomous system behavior in real time. By integrating cognitively grounded workload and situational awareness estimation with dynamic human error probability prediction, the framework transforms conventional offline reliability analysis into a proactive intervention mechanism embedded directly within operational workflows. Experimental validation on a high-fidelity digital control room simulator demonstrates that NuHF Claw can anticipate interface induced cognitive degradation, dynamically constrain unsafe autonomous recommendations, and provide risk-aware navigational guidance while preserving human decision authority. The results highlight a fundamental shift from automation-driven operation toward cognition-aware autonomy, offering a principled pathway for the safe integration of intelligent agents into next-generation nuclear control environments.

**arXiv ID:** 2604.14160
</details>

<details>
<summary><strong>CBCL: Safe Self-Extending Agent Communication</strong> - Hugo O'Connor - [[pdf]](https://arxiv.org/pdf/2604.14512)</summary>

**Abstract:** Agent communication languages (ACLs) enable heterogeneous agents to share knowledge and coordinate across diverse domains. This diversity demands extensibility, but expressive extension mechanisms can push the input language beyond the complexity classes where full validation is tractable. We present CBCL (Common Business Communication Language), an agent communication language that constrains all messages, including runtime language extensions, to the deterministic context-free language (DCFL) class. CBCL allows agents to define, transmit, and adopt domain-specific "dialect" extensions as first-class messages; three safety invariants (R1--R3), machine-checked in Lean 4 and enforced in a Rust reference implementation, prevent unbounded expansion, applying declared resource limits, and preserving core vocabulary. We formalize the language and its safety properties in Lean 4, implement a reference parser and dialect engine in Rust with property-based and differential tests, and extract a verified parser binary. Our results demonstrate that homoiconic protocol design, where extension definitions share the same representation as ordinary messages, can be made provably safe. As autonomous agents increasingly extend their own communication capabilities, formally bounding what they can express to each other is a precondition for oversight.

**arXiv ID:** 2604.14512
</details>

<details>
<summary><strong>Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation</strong> - Chen Huang, Zitan Jiang, Changyi Zou, Wenqiang Lei, See-Kiong Ng - [[pdf]](https://arxiv.org/pdf/2604.11077)</summary>

**Abstract:** Customer service chatbots are increasingly expected to serve not merely as reactive support tools for users, but as strategic interfaces for harvesting high-value information and business intelligence. In response, we make three main contributions. 1) We introduce and define a novel task of Proactive Information Probing, which optimizes when to probe users for pre-specified target information while minimizing conversation turns and user friction. 2) We propose PROCHATIP, a proactive chatbot framework featuring a specialized conversation strategy module trained to master the delicate timing of probes. 3) Experiments demonstrate that PROCHATIP significantly outperforms baselines, exhibiting superior capability in both information probing and service quality. We believe that our work effectively redefines the commercial utility of chatbots, positioning them as scalable, cost-effective engines for proactive business intelligence. Our code is available at this https URL.

**arXiv ID:** 2604.11077
</details>

<details>
<summary><strong>From Risk to Rescue: An Agentic Survival Analysis Framework for Liquidation Prevention</strong> - Fernando Spadea, Oshani Seneviratne - [[pdf]](https://arxiv.org/pdf/2604.14583)</summary>

**Abstract:** Decentralized Finance (DeFi) lending protocols like Aave v3 rely on over-collateralization to secure loans, yet users frequently face liquidation due to volatile market conditions. Existing risk management tools utilize static health-factor thresholds, which are reactive and fail to distinguish between administrative "dust" cleanup and genuine insolvency. In this work, we propose an autonomous agent that leverages time-to-event (survival) analysis and moves beyond prediction to execution. Unlike passive risk signals, this agent perceives risk, simulates counterfactual futures, and executes protocol-faithful interventions to proactively prevent liquidations. We introduce a return period metric derived from a numerically stable XGBoost Cox proportional hazards model to normalize risk across transaction types, coupled with a volatility-adjusted trend score to filter transient market noise. To select optimal interventions, we implement a counterfactual optimization loop that simulates potential user actions to find the minimum capital required to mitigate risk. We validate our approach using a high-fidelity, protocol-faithful Aave v3 simulator on a cohort of 4,882 high-risk user profiles. The results demonstrate the agent's ability to prevent liquidations in imminent-risk scenarios where static rules fail, effectively "saving the unsavable" while maintaining a zero worsening rate, providing a critical safety guarantee often missing in autonomous financial agents. Furthermore, the system successfully differentiates between actionable financial risks and negligible dust events, optimizing capital efficiency where static rules fail.

**arXiv ID:** 2604.14583
</details>

<details>
<summary><strong>Dialogue Agents that Share Family Information to Strengthen Grandparent-Grandchild Relationships</strong> - Seiya Mitsuno, Midori Ban, Hiroshi Ishiguro, Yuichiro Yoshikawa - [[pdf]](https://arxiv.org/pdf/2604.12310)</summary>

**Abstract:** Social isolation among older adults has become a critical concern, as reduced opportunities for conversation and weakened family relationships negatively affect mental health. This study proposes a dialogue agent that supports older adults by fostering both a relationship with the agent and a relationship with their grandchild through sharing everyday information. The agent operates on a chatbot platform and engages in daily conversations with older adults and their grandchildren, exchanging information gathered from each party to enhance conversational engagement and social connection. We conducted a ten-day empirical experiment with 52 grandparent-grandchild pairs. The results suggest that older adults became more willing to interact with the proposed agent, which shared information about their grandchildren, and that the psychological connection between grandparents and grandchildren was strengthened. Furthermore, daily interactions with the agent were associated with reduced anxiety in both older adults and their grandchildren. These findings indicate that a dialogue agent that shares personal information can be an effective approach to supporting older adults by simultaneously offering conversational opportunities and promoting family connectedness. Overall, this study provides valuable insights into the design of dialogue agents that effectively address social isolation among older adults.

**arXiv ID:** 2604.12310
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (17 papers)</h2></summary>

<details>
<summary><strong>Simulating Human Cognition: Heartbeat-Driven Autonomous Thinking Activity Scheduling for LLM-based AI systems</strong> - Hong Su - [[pdf]](https://arxiv.org/pdf/2604.14178)</summary>

**Abstract:** Large Language Model (LLM) agents have demonstrated remarkable capabilities in reasoning and tool use, yet they often suffer from rigid, reactive control flows that limit their adaptability and efficiency. Most existing frameworks rely on fixed pipelines or failure-triggered reflection, causing agents to act impulsively or correct errors only after they occur. In this paper, we introduce Heartbeat-Driven Autonomous Thinking Activity Scheduling, a mechanism that enables proactive, adaptive, and continuous self-regulation. Mirroring the natural rhythm of human cognition, our system employs a periodic ``heartbeat'' mechanism to orchestrate a dynamic repertoire of cognitive modules (e.g., Planner, Critic, Recaller, Dreamer). Unlike traditional approaches that rely on hard-coded symbolic rules or immediate reactive triggers, our scheduler learns to determine when to engage specific thinking activities -- such as recalling memories, summarizing experiences, or strategic planning -- based on temporal patterns and historical context. This functional approach allows cognitive modules to be dynamically added or removed without structural reengineering. Meanwhile, we propose a meta-learning strategy for continual policy adaptation, where the scheduler optimizes its cognitive strategy over time using historical interaction logs. Evaluation results demonstrate that our approach effectively learns to schedule cognitive activities based on historical data and can autonomously integrate new thinking modules.

**arXiv ID:** 2604.14178
</details>

<details>
<summary><strong>AIBuildAI: An AI Agent for Automatically Building AI Models</strong> - Ruiyi Zhang, Peijia Qin, Qi Cao, Li Zhang, Pengtao Xie - [[pdf]](https://arxiv.org/pdf/2604.14455)</summary>

**Abstract:** AI models underpin modern intelligent systems, driving advances across science, medicine, finance, and technology. Yet developing high-performing AI models remains a labor-intensive process that requires expert practitioners to iteratively design architectures, engineer representations, implement training pipelines and refine approaches through empirical evaluation. Existing AutoML methods partially alleviate this burden but remain limited to narrow aspects such as hyperparameter optimization and model selection within predefined search spaces, leaving the full development lifecycle largely dependent on human expertise. To address this gap, we introduce AIBuildAI, an AI agent that automatically builds AI models from a task description and training data. AIBuildAI adopts a hierarchical agent architecture in which a manager agent coordinates three specialized sub-agents: a designer for modeling strategy, a coder for implementation and debugging, and a tuner for training and performance optimization. Each sub-agent is itself a large language model (LLM) based agent capable of multi-step reasoning and tool use, enabling end-to-end automation of the AI model development process that goes beyond the scope of existing AutoML approaches. We evaluate AIBuildAI on MLE-Bench, a benchmark of realistic Kaggle-style AI development tasks spanning visual, textual, time-series and tabular modalities. AIBuildAI ranks first on MLE-Bench with a medal rate of 63.1%, outperforming all existing baseline methods and matching the capability of highly experienced AI engineers. These results demonstrate that hierarchical agent systems can automate the full AI model development process from task specification to deployable model, suggesting a pathway toward broadly accessible AI development with minimal human intervention.

**arXiv ID:** 2604.14455
</details>

<details>
<summary><strong>The Agentification of Scientific Research: A Physicist's Perspective</strong> - Xiao-Liang Qi - [[pdf]](https://arxiv.org/pdf/2604.14718)</summary>

**Abstract:** This article argues that the most important significance of the AI revolution, especially the rise of large language models, lies not simply in automation, but in a fundamental change in how complex information and human know-how are carried, replicated, and shared. From this perspective, AI for Science is especially important because it may transform not only the efficiency of research, but also the structure of scientific collaboration, discovery, publishing, and evaluation. The article outlines a gradual path from AI as a research tool to AI as a scientific collaborator, and discusses how AI is likely to fundamentally reshape scientific publication. It also argues that continuous learning and diversity of ideas are essential if AI is to play a meaningful role in original scientific discovery.

**arXiv ID:** 2604.14718
</details>

<details>
<summary><strong>CogEvolution: A Human-like Generative Educational Agent to Simulate Student's Cognitive Evolution</strong> - Wei Zhang, Yihang Cheng, Zhirong Ye, Kezhen Huang - [[pdf]](https://arxiv.org/pdf/2604.14786)</summary>

**Abstract:** Generative Agents, owing to their precise modeling and simulation capabilities of human behavior, have become a pivotal tool in the field of Artificial Intelligence in Education (AIEd) for uncovering complex cognitive processes of learners. However, existing educational agents predominantly rely on static personas to simulate student learning behaviors, neglecting the decisive role of deep cognitive capabilities in learning outcomes during practice interactions. Furthermore, they struggle to characterize the dynamic fluidity of knowledge internalization, transfer, and cognitive state transitions. To overcome this bottleneck, this paper proposes a human-like educational agent capable of simulating student cognitive evolution: CogEvolution. Specifically, we first construct a cognitive depth perceptron based on the Interactive, Constructive, Active, Passive (ICAP) taxonomy from cognitive psychology, achieving precise quantification of learner cognitive engagement. Subsequently, we propose a memory retrieval method based on Item Response Theory (IRT) to simulate the connection and assimilation of new and prior knowledge. Finally, we design a dynamic cognitive update mechanism based on evolutionary algorithms to simulate the real-time integration of student learning behaviors and cognitive evolution processes. Comprehensive evaluations demonstrate that CogEvolution not only significantly outperforms baseline models in behavioral fidelity and learning curve fitting but also uniquely reproduces plausible and robust cognitive evolutionary paths consistent with educational psychology expectations, providing a novel paradigm for constructing highly interpretable educational agents.

**arXiv ID:** 2604.14786
</details>

<details>
<summary><strong>OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis</strong> - Kanzhi Cheng, Zehao Li, Zheng Ma, Nuo Chen, Jialin Cao, Qiushi Sun, Zichen Ding, Fangzhi Xu, Hang Yan, Jiajun Chen, Anh Tuan Luu, Jianbing Zhang, Lewei Lu, Dahua Lin - [[pdf]](https://arxiv.org/pdf/2604.15093)</summary>

**Abstract:** Mobile agents powered by vision-language models have demonstrated impressive capabilities in automating mobile tasks, with recent leading models achieving a marked performance leap, e.g., nearly 70% success on AndroidWorld. However, these systems keep their training data closed and remain opaque about their task and trajectory synthesis recipes. We present OpenMobile, an open-source framework that synthesizes high-quality task instructions and agent trajectories, with two key components: (1) The first is a scalable task synthesis pipeline that constructs a global environment memory from exploration, then leverages it to generate diverse and grounded instructions. and (2) a policy-switching strategy for trajectory rollout. By alternating between learner and expert models, it captures essential error-recovery data often missing in standard imitation learning. Agents trained on our data achieve competitive results across three dynamic mobile agent benchmarks: notably, our fine-tuned Qwen2.5-VL and Qwen3-VL reach 51.7% and 64.7% on AndroidWorld, far surpassing existing open-data approaches. Furthermore, we conduct transparent analyses on the overlap between our synthetic instructions and benchmark test sets, and verify that performance gains stem from broad functionality coverage rather than benchmark overfitting. We release data and code at this https URL to bridge the data gap and facilitate broader mobile agent research.

**arXiv ID:** 2604.15093
</details>

<details>
<summary><strong>Neuro-Oracle: A Trajectory-Aware Agentic RAG Framework for Interpretable Epilepsy Surgical Prognosis</strong> - Aizierjiang Aiersilan, Mohamad Koubeissi - [[pdf]](https://arxiv.org/pdf/2604.14216)</summary>

**Abstract:** Predicting post-surgical seizure outcomes in pharmacoresistant epilepsy is a clinical challenge. Conventional deep-learning approaches operate on static, single-timepoint pre-operative scans, omitting longitudinal morphological changes. We propose \emph{Neuro-Oracle}, a three-stage framework that: (i) distils pre-to-post-operative MRI changes into a compact 512-dimensional trajectory vector using a 3D Siamese contrastive encoder; (ii) retrieves historically similar surgical trajectories from a population archive via nearest-neighbour search; and (iii) synthesises a natural-language prognosis grounded in the retrieved evidence using a quantized Llama-3-8B reasoning agent. Evaluations are conducted on the public EPISURG dataset ($N{=}268$ longitudinally paired cases) using five-fold stratified cross-validation. Since ground-truth seizure-freedom scores are unavailable, we utilize a clinical proxy label based on the resection type. We acknowledge that the network representations may potentially learn the anatomical features of the resection cavities (i.e., temporal versus non-temporal locations) rather than true prognostic morphometry. Our current evaluation thus serves mainly as a proof-of-concept for the trajectory-aware retrieval architecture. Trajectory-based classifiers achieve AUC values between 0.834 and 0.905, compared with 0.793 for a single-timepoint ResNet-50 baseline. The Neuro-Oracle agent (M5) matches the AUC of purely discriminative trajectory classifiers (0.867) while producing structured justifications with zero observed hallucinations under our audit protocol. A Siamese Diversity Ensemble (M6) of trajectory-space classifiers attains an AUC of 0.905 without language-model overhead.

**arXiv ID:** 2604.14216
</details>

<details>
<summary><strong>Knowledge Graph RAG: Agentic Crawling and Graph Construction in Enterprise Documents</strong> - Koushik Chakraborty, Koyel Guha - [[pdf]](https://arxiv.org/pdf/2604.14220)</summary>

**Abstract:** This research paper addresses the limitations of semantic search in complex enterprise document ecosystems. Traditional RAG pipelines often fail to capture hierarchical and interconnected information, leading to retrieval inaccuracies. We propose Agentic Knowledge Graphs featuring Recursive Crawling as a robust solution for navigating superseding logic and multi-hop references. Our benchmark evaluation using the Code of Federal Regulations (CFR) demonstrates that this Knowledge Graph-enhanced approach achieves a 70% accuracy improvement over standard vector-based RAG systems, providing exhaustive and precise answers for complex regulatory queries.

**arXiv ID:** 2604.14220
</details>

<details>
<summary><strong>Evaluation of Agents under Simulated AI Marketplace Dynamics</strong> - To Eun Kim, Alireza Salemi, Hamed Zamani, Fernando Diaz - [[pdf]](https://arxiv.org/pdf/2604.14256)</summary>

**Abstract:** Modern information access ecosystems consist of mixtures of systems, such as retrieval systems and large language models, and increasingly rely on marketplaces to mediate access to models, tools, and data, making competition between systems inherent to deployment. In such settings, outcomes are shaped not only by benchmark quality but also by competitive pressure, including user switching, routing decisions, and operational constraints. Yet evaluation is still largely conducted on static benchmarks with accuracy-focused measures that assume systems operate in isolation. This mismatch makes it difficult to predict post-deployment success and obscures competitive effects such as early-adoption advantages and market dominance. We introduce Marketplace Evaluation, a simulation-based paradigm that evaluates information access systems as participants in a competitive marketplace. By simulating repeated interactions and evolving user and agent preferences, the framework enables longitudinal evaluation and marketplace-level metrics, such as retention and market share, that complement and can extend beyond traditional accuracy-based metrics. We formalize the framework and outline a research agenda, motivated by business and economics, around marketplace simulation, metrics, optimization, and adoption in evaluation campaigns like TREC.

**arXiv ID:** 2604.14256
</details>

<details>
<summary><strong>AD4AD: Benchmarking Visual Anomaly Detection Models for Safer Autonomous Driving</strong> - Fabrizio Genilotti, Arianna Stropeni, Gionata Grotto, Francesco Borsatti, Manuel Barusco, Davide Dalle Pezze, Gian Antonio Susto - [[pdf]](https://arxiv.org/pdf/2604.15291)</summary>

**Abstract:** The reliability of a machine vision system for autonomous driving depends heavily on its training data distribution. When a vehicle encounters significantly different conditions, such as atypical obstacles, its perceptual capabilities can degrade substantially. Unlike many domains where errors carry limited consequences, failures in autonomous driving translate directly into physical risk for passengers, pedestrians, and other road users. To address this challenge, we explore Visual Anomaly Detection (VAD) as a solution. VAD enables the identification of anomalous objects not present during training, allowing the system to alert the driver when an unfamiliar situation is detected. Crucially, VAD models produce pixel-level anomaly maps that can guide driver attention to specific regions of concern without requiring any prior assumptions about the nature or form of the hazard. We benchmark eight state-of-the-art VAD methods on AnoVox, the largest synthetic dataset for anomaly detection in autonomous driving. In particular, we evaluate performance across four backbone architectures spanning from large networks to lightweight ones such as MobileNet and DeiT-Tiny. Our results demonstrate that VAD transfers effectively to road scenes. Notably, Tiny-Dinomaly achieves the best accuracy-efficiency trade-off for edge deployment, matching full-scale localization performance at a fraction of the memory cost. This study represents a concrete step toward safer, more responsible deployment of autonomous vehicles, ultimately improving protection for passengers, pedestrians, and all road users.

**arXiv ID:** 2604.15291
</details>

<details>
<summary><strong>MM-WebAgent: A Hierarchical Multimodal Web Agent for Webpage Generation</strong> - Yan Li, Zezi Zeng, Yifan Yang, Yuqing Yang, Ning Liao, Weiwei Guo, Lili Qiu, Mingxi Cheng, Qi Dai, Zhendong Wang, Zhengyuan Yang, Xue Yang, Ji Li, Lijuan Wang, Chong Luo - [[pdf]](https://arxiv.org/pdf/2604.15309)</summary>

**Abstract:** The rapid progress of Artificial Intelligence Generated Content (AIGC) tools enables images, videos, and visualizations to be created on demand for webpage design, offering a flexible and increasingly adopted paradigm for modern UI/UX. However, directly integrating such tools into automated webpage generation often leads to style inconsistency and poor global coherence, as elements are generated in isolation. We propose MM-WebAgent, a hierarchical agentic framework for multimodal webpage generation that coordinates AIGC-based element generation through hierarchical planning and iterative self-reflection. MM-WebAgent jointly optimizes global layout, local multimodal content, and their integration, producing coherent and visually consistent webpages. We further introduce a benchmark for multimodal webpage generation and a multi-level evaluation protocol for systematic assessment. Experiments demonstrate that MM-WebAgent outperforms code-generation and agent-based baselines, especially on multimodal element generation and integration. Code & Data: this https URL.

**arXiv ID:** 2604.15309
</details>

<details>
<summary><strong>FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems</strong> - Wenhao Wang, Haoting Shi, Mengying Yuan, Yiquan Lin, Panrong Tong, Hanzhang Zhou, Guangyi Liu, Pengxiang Zhao, Yue Wang, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2604.14956)</summary>

**Abstract:** Training GUI agents with traditional centralized methods faces significant cost and scalability challenges. Federated learning (FL) offers a promising solution, yet its potential is hindered by the lack of benchmarks that capture real-world, cross-platform heterogeneity. To bridge this gap, we introduce FedGUI, the first comprehensive benchmark for developing and evaluating federated GUI agents across mobile, web, and desktop platforms. FedGUI provides a suite of six curated datasets to systematically study four crucial types of heterogeneity: cross-platform, cross-device, cross-OS, and cross-source. Extensive experiments reveal several key insights: First, we show that cross-platform collaboration improves performance, extending prior mobile-only federated learning to diverse GUI environments; Second, we demonstrate the presence of distinct heterogeneity dimensions and identify platform and OS as the most influential factors. FedGUI provides a vital foundation for the community to build more scalable and privacy-preserving GUI agents for real-world deployment. Our code and data are publicly available at this https URL..

**arXiv ID:** 2604.14956
</details>

<details>
<summary><strong>AgentOpt v0.1 Technical Report: Client-Side Optimization for LLM-Based Agent</strong> - Wenyue Hua, Sripad Karne, Qian Xie, Armaan Agrawal, Nikos Pagonas, Kostis Kaffes, Tianyi Peng - [[pdf]](https://arxiv.org/pdf/2604.06296)</summary>

**Abstract:** AI agents are increasingly deployed in real-world applications, including systems such as Manus, OpenClaw, and coding agents. Existing research has primarily focused on server-side efficiency, proposing methods such as caching, speculative execution, traffic scheduling, and load balancing to reduce the cost of serving agentic workloads. However, as users increasingly construct agents by composing local tools, remote APIs, and diverse models, an equally important optimization problem arises on the client side. Client-side optimization asks how developers should allocate the resources available to them, including model choice, local tools, and API budget across pipeline stages, subject to application-specific quality, cost, and latency constraints. Because these objectives depend on the task and deployment setting, they cannot be determined by server-side systems alone. We introduce AgentOpt, the first framework-agnostic Python package for client-side agent optimization. We first study model selection, a high-impact optimization lever in multi-step agent pipelines. Given a pipeline and a small evaluation set, the goal is to find the most cost-effective assignment of models to pipeline roles. This problem is consequential in practice: at matched accuracy, the cost gap between the best and worst model combinations can reach 13-32x in our experiments. To efficiently explore the exponentially growing combination space, AgentOpt implements ten search algorithms, including UCB-E, UCB-E with Low-Rank Factorization, Arm Elimination, Epsilon-LUCB, Threshold Successive Elimination, and Bayesian Optimization. Across four benchmarks, UCB-E recovers near-optimal accuracy while reducing evaluation budget by 62-76\% relative to brute-force search. Code and benchmark results available at this https URL.

**arXiv ID:** 2604.06296
</details>

<details>
<summary><strong>IE as Cache: Information Extraction Enhanced Agentic Reasoning</strong> - Hang Lv, Sheng Liang, Hongchao Gu, Wei Guo, Defu Lian, Yong Liu, Hao Wang, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2604.14930)</summary>

**Abstract:** Information Extraction aims to distill structured, decision-relevant information from unstructured text, serving as a foundation for downstream understanding and reasoning. However, it is traditionally treated merely as a terminal objective: once extracted, the resulting structure is often consumed in isolation rather than maintained and reused during multi-step inference. Moving beyond this, we propose \textit{IE-as-Cache}, a framework that repurposes IE as a cognitive cache to enhance agentic reasoning. Drawing inspiration from hierarchical computer memory, our approach combines query-driven extraction with cache-aware reasoning to dynamically maintain compact intermediate information and filter noise. Experiments on challenging benchmarks across diverse LLMs demonstrate significant improvements in reasoning accuracy, indicating that IE can be effectively repurposed as a reusable cognitive resource and offering a promising direction for future research on downstream uses of IE.

**arXiv ID:** 2604.14930
</details>

<details>
<summary><strong>An Intelligent Robotic and Bio-Digestor Framework for Smart Waste Management</strong> - Radhika Khatri, Adit Tewari, Nikhil Sharma, M. B. Srinivas - [[pdf]](https://arxiv.org/pdf/2604.14882)</summary>

**Abstract:** Rapid urbanization and continuous population growth have made municipal solid waste management increasingly challenging. These challenges highlight the need for smarter and automated waste management solutions. This paper presents the design and evaluation of an integrated waste management framework that combines two connected systems, a robotic waste segregation module and an optimized bio-digestor. The robotic waste segregation system uses a MyCobot 280 Jetson Nano robotic arm along with YOLOv8 object detection and robot operating system (ROS)-based path planning to identify and sort waste in real time. It classifies waste into four different categories with high precision, reducing the need for manual intervention. After segregation, the biodegradable waste is transferred to a bio-digestor system equipped with multiple sensors. These sensors continuously monitor key parameters, including temperature, pH, pressure, and motor revolutions per minute. The Particle Swarm Optimization (PSO) algorithm, combined with a regression model, is used to dynamically adjust system parameters. This intelligent optimization approach ensures stable operation and maximizes digestion efficiency under varying environmental conditions. System testing under dynamic conditions demonstrates a sorting accuracy of 98% along with highly efficient biological conversion. The proposed framework offers a scalable, intelligent, and practical solution for modern waste management, making it suitable for both residential and industrial applications.

**arXiv ID:** 2604.14882
</details>

<details>
<summary><strong>Atropos: Improving Cost-Benefit Trade-off of LLM-based Agents under Self-Consistency with Early Termination and Model Hotswap</strong> - Naryeong Kim, Shin Yoo - [[pdf]](https://arxiv.org/pdf/2604.15075)</summary>

**Abstract:** Open-weight Small Language Models(SLMs) can provide faster local inference at lower financial cost, but may not achieve the same performance level as commercial Large Language Models (LLMs) that are orders of magnitudes larger. Consequently, many of the latest applications of LLMs, such as software engineering agents, tend to be evaluated on larger models only, leaving the issue of improving the cost-benefit trade-off of such applications neglected. This paper proposes Atropos, a predictive early-termination analysis and hotswap technique that aims to improve the cost-benefit trade-off for LLM-based agents that use self-consistency. The core component of ATROPOS is a predictive model based on structural properties of LLM inferences: after merging multiple agentic inference paths into a graph representation, ATROPOS uses Graph Convolutional Network (GCN) to predict whether an ongoing inference will eventually succeed or not. If an agentic task instance running on the source LLM is predicted to fail, ATROPOS subsequently performs hotswapping, i.e., migrating the on-going inference context onto the more capable target LLM: this is feasible because LLM contexts are stateless. An empirical evaluation of ATROPOS using three recent LLM-based agents shows that ATROPOS can predict early termination of eventually failing inferences with the accuracy of 0.85 at the midpoint of the inference. Hotswapping LLMs for such inferences can convert up to 27.57% of them to be successful. Consequently, ATROPOS achieves 74.35% of the performance of closed LLMs with as low as only 23.9% of the cost.

**arXiv ID:** 2604.15075
</details>

<details>
<summary><strong>Dual Pose-Graph Semantic Localization for Vision-Based Autonomous Drone Racing</strong> - David Perez-Saura, Miguel Fernandez-Cortizas, Alvaro J. Gaona, Pascual Campoy - [[pdf]](https://arxiv.org/pdf/2604.15168)</summary>

**Abstract:** Autonomous drone racing demands robust real-time localization under extreme conditions: high-speed flight, aggressive maneuvers, and payload-constrained platforms that often rely on a single camera for perception. Existing visual SLAM systems, while effective in general scenarios, struggle with motion blur and feature instability inherent to racing dynamics, and do not exploit the structured nature of racing environments. In this work, we present a dual pose-graph architecture that fuses odometry with semantic detections for robust localization. A temporary graph accumulates multiple gate observations between keyframes and optimizes them into a single refined constraint per landmark, which is then promoted to a persistent main graph. This design preserves the information richness of frequent detections while preventing graph growth from degrading real-time performance. The system is designed to be sensor-agnostic, although in this work we validate it using monocular visual-inertial odometry and visual gate detections. Experimental evaluation on the TII-RATM dataset shows a 56% to 74% reduction in ATE compared to standalone VIO, while an ablation study confirms that the dual-graph architecture achieves 10% to 12% higher accuracy than a single-graph baseline at identical computational cost. Deployment in the A2RL competition demonstrated that the system performs real-time onboard localization during flight, reducing the drift of the odometry baseline by up to 4.2 m per lap.

**arXiv ID:** 2604.15168
</details>

<details>
<summary><strong>Towards a Multi-Embodied Grasping Agent</strong> - Roman Freiberg, Alexander Qualmann, Ngo Anh Vien, Gerhard Neumann - [[pdf]](https://arxiv.org/pdf/2510.27420)</summary>

**Abstract:** Multi-embodiment grasping focuses on developing approaches that exhibit generalist behavior across diverse gripper designs. Existing methods often learn the kinematic structure of the robot implicitly and face challenges due to the difficulty of sourcing the required large-scale data. In this work, we present a data-efficient, flow-based, equivariant grasp synthesis architecture that can handle different gripper types with variable degrees of freedom and successfully exploit the underlying kinematic model, deducing all necessary information solely from the gripper and scene geometry. Unlike previous equivariant grasping methods, we translated all modules from the ground up to JAX and provide a model with batching capabilities over scenes, grippers, and grasps, resulting in smoother learning, improved performance and faster inference time. Our dataset encompasses grippers ranging from humanoid hands to parallel yaw grippers and includes 25,000 scenes and 20 million grasps.

**arXiv ID:** 2510.27420
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>CAMO: An Agentic Framework for Automated Causal Discovery from Micro Behaviors to Macro Emergence in LLM Agent Simulations</strong> - Xiangning Yu, Yuwei Guo, Yuqi Hou, Xiao Xue, Qun Ma - [[pdf]](https://arxiv.org/pdf/2604.14691)</summary>

**Abstract:** LLM-empowered agent simulations are increasingly used to study social emergence, yet the micro-to-macro causal mechanisms behind macro outcomes often remain unclear. This is challenging because emergence arises from intertwined agent interactions and meso-level feedback and nonlinearity, making generative mechanisms hard to disentangle. To this end, we introduce \textbf{\textsc{CAMO}}, an automated \textbf{Ca}usal discovery framework from \textbf{M}icr\textbf{o} behaviors to \textbf{M}acr\textbf{o} Emergence in LLM agent simulations. \textsc{CAMO} converts mechanistic hypotheses into computable factors grounded in simulation records and learns a compact causal representation centered on an emergent target $Y$. \textsc{CAMO} outputs a computable Markov boundary and a minimal upstream explanatory subgraph, yielding interpretable causal chains and actionable intervention levers. It also uses simulator-internal counterfactual probing to orient ambiguous edges and revise hypotheses when evidence contradicts the current view. Experiments across four emergent settings demonstrate the promise of \textsc{CAMO}.

**arXiv ID:** 2604.14691
</details>

<details>
<summary><strong>HWE-Bench: Benchmarking LLM Agents on Real-World Hardware Bug Repair Tasks</strong> - Fan Cui, Hongyuan Hou, Zizhang Luo, Chenyun Yin, Yun Liang - [[pdf]](https://arxiv.org/pdf/2604.14709)</summary>

**Abstract:** Existing benchmarks for hardware design primarily evaluate Large Language Models (LLMs) on isolated, component-level tasks such as generating HDL modules from specifications, leaving repository-scale evaluation unaddressed. We introduce HWE-Bench, the first large-scale, repository-level benchmark for evaluating LLM agents on real-world hardware bug repair tasks. HWE-Bench comprises 417 task instances derived from real historical bug-fix pull requests across six major open-source projects spanning both Verilog/SystemVerilog and Chisel, covering RISC-V cores, SoCs, and security roots-of-trust. Each task is grounded in a fully containerized environment where the agent must resolve a real bug report, with correctness validated through the project's native simulation and regression flows. The benchmark is built through a largely automated pipeline that enables efficient expansion to new repositories. We evaluate seven LLMs with four agent frameworks and find that the best agent resolves 70.7% of tasks overall, with performance exceeding 90% on smaller cores but dropping below 65% on complex SoC-level projects. We observe larger performance gaps across models than commonly reported on software benchmarks, and difficulty is driven by project scope and bug-type distribution rather than code size alone. Our failure analysis traces agent failures to three stages of the debugging process: fault localization, hardware-semantic reasoning, and cross-artifact coordination across RTL, configuration, and verification components, providing concrete directions for developing more capable hardware-aware agents.

**arXiv ID:** 2604.14709
</details>

<details>
<summary><strong>From Reactive to Proactive: Assessing the Proactivity of Voice Agents via ProVoice-Bench</strong> - Ke Xu, Yuhao Wang, Yu Wang - [[pdf]](https://arxiv.org/pdf/2604.15037)</summary>

**Abstract:** Recent advancements in LLM agents are gradually shifting from reactive, text-based paradigms toward proactive, multimodal interaction. However, existing benchmarks primarily focus on reactive responses, overlooking the complexities of proactive intervention and monitoring. To bridge this gap, we introduce ProVoice-Bench, the first evaluation framework specifically designed for proactive voice agents, featuring four novel tasks. By leveraging a multi-stage data synthesis pipeline, we curate 1,182 high-quality samples for rigorous testing. Our evaluation of state-of-the-art Multimodal LLMs reveals a significant performance gap, particularly regarding over-triggering and reasoning capabilities. These findings highlight the limitations of current models and offer a roadmap for developing more natural, context-aware proactive agents.

**arXiv ID:** 2604.15037
</details>

<details>
<summary><strong>Don't Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills for QA and RAG</strong> - Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh - [[pdf]](https://arxiv.org/pdf/2604.14572)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) grounds LLM responses in external evidence but treats the model as a passive consumer of search results: it never sees how the corpus is organized or what it has not yet retrieved, limiting its ability to backtrack or combine scattered evidence. We present Corpus2Skill, which distills a document corpus into a hierarchical skill directory offline and lets an LLM agent navigate it at serve time. The compilation pipeline iteratively clusters documents, generates LLM-written summaries at each level, and materializes the result as a tree of navigable skill files. At serve time, the agent receives a bird's-eye view of the corpus, drills into topic branches via progressively finer summaries, and retrieves full documents by ID. Because the hierarchy is explicitly visible, the agent can reason about where to look, backtrack from unproductive paths, and combine evidence across branches. On WixQA, an enterprise customer-support benchmark for RAG, Corpus2Skill outperforms dense retrieval, RAPTOR, and agentic RAG baselines across all quality metrics.

**arXiv ID:** 2604.14572
</details>

<details>
<summary><strong>Scepsy: Serving Agentic Workflows Using Aggregate LLM Pipelines</strong> - Marcel Wagenländer, Otto White, Britannio Jarrett, Pedro Silvestre, Yanda Tao, Guo Li, Huanzhou Zhu, Llúis Vilanova, Peter Pietzuch - [[pdf]](https://arxiv.org/pdf/2604.15186)</summary>

**Abstract:** Agentic workflows carry out complex tasks by orchestrating multiple large language models (LLMs) and tools. Serving such workflows at a target throughput with low latency is challenging because they can be defined using arbitrary agentic frameworks and exhibit unpredictable execution times: execution may branch, fan-out, or recur in data-dependent ways. Since LLMs in workflows often outnumber available GPUs, their execution also leads to GPU oversubscription.
We describe Scepsy, a new agentic serving system that efficiently schedules arbitrary multi-LLM agentic workflows onto a GPU cluster. Scepsy exploits the insight that, while agentic workflows have unpredictable end-to-end latencies, the shares of each LLM's total execution times are comparatively stable across executions. Scepsy decides on GPU allocations based on these aggregate shares: first, it profiles the LLMs under different parallelism degrees. It then uses these statistics to construct an Aggregate LLM Pipeline, which is a lightweight latency/throughput predictor for allocations. To find a GPU allocation that minimizes latency while achieving a target throughput, Scepsy uses the Aggregate LLM Pipeline to explore a search space over fractional GPU shares, tensor parallelism degrees, and replica counts. It uses a hierarchical heuristic to place the best allocation onto the GPU cluster, minimizing fragmentation, while respecting network topology constraints. Our evaluation on realistic agentic workflows shows that Scepsy achieves up to 2.4x higher throughput and 27x lower latency compared to systems that optimize LLMs independently or rely on user-specified allocations.

**arXiv ID:** 2604.15186
</details>

<details>
<summary><strong>CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas</strong> - Emanuel Tewolde, Xiao Zhang, David Guzman Piedrahita, Vincent Conitzer, Zhijing Jin - [[pdf]](https://arxiv.org/pdf/2604.15267)</summary>

**Abstract:** It is increasingly important that LLM agents interact effectively and safely with other goal-pursuing agents, yet, recent works report the opposite trend: LLMs with stronger reasoning capabilities behave _less_ cooperatively in mixed-motive games such as the prisoner's dilemma and public goods settings. Indeed, our experiments show that recent models -- with or without reasoning enabled -- consistently defect in single-shot social dilemmas.
To tackle this safety concern, we present the first comparative study of game-theoretic mechanisms that are designed to enable cooperative outcomes between rational agents _in equilibrium_. Across four social dilemmas testing distinct components of robust cooperation, we evaluate the following mechanisms: (1) repeating the game for many rounds, (2) reputation systems, (3) third-party mediators to delegate decision making to, and (4) contract agreements for outcome-conditional payments between players. Among our findings, we establish that contracting and mediation are most effective in achieving cooperative outcomes between capable LLM models, and that repetition-induced cooperation deteriorates drastically when co-players vary. Moreover, we demonstrate that these cooperation mechanisms become _more effective_ under evolutionary pressures to maximize individual payoffs.

**arXiv ID:** 2604.15267
</details>

<details>
<summary><strong>What Deserves Memory: Adaptive Memory Distillation for LLM Agents</strong> - Wenquan Ma, Jiayan Nan, Wenlong Wu, Yize Chen - [[pdf]](https://arxiv.org/pdf/2508.03341)</summary>

**Abstract:** Memory systems for LLM agents struggle to determine what information deserves retention. Existing approaches rely on predefined heuristics such as importance scores, emotional tags, or factual templates, encoding designer intuition rather than learning from the data itself. Inspired by cognitive ideas, we propose NEMORI, an adaptive memory distillation framework that casts the assessment of experience's future utility as a matter of predictability. Specifically, NEMORI comprises two cascading modules: Episodic Memory Integration transforms raw interactions into coherent narratives, and Semantic Knowledge Distillation extracts insights via prediction error. Centering on distillation, the framework remains agnostic to downstream management. Extensive experiments confirm that NEMORI achieves strong performance, efficiency, and storage reduction. Our work suggests that observing the intrinsic properties of interaction sequences offers a viable, data-driven alternative to heuristic-based memory design. Code: this https URL.

**arXiv ID:** 2508.03341
</details>

<details>
<summary><strong>Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception</strong> - Yize Cheng, Arshia Soltani Moakhar, Chenrui Fan, Parsa Hosseini, Kazem Faghih, Zahra Sodagar, Wenxiao Wang, Soheil Feizi - [[pdf]](https://arxiv.org/pdf/2510.23853)</summary>

**Abstract:** Large language model (LLM) agents are increasingly used to interact with and execute tasks in dynamic environments. However, a critical yet overlooked limitation of these agents is that they, by default, assume a stationary context, failing to account for the real-world time elapsed between messages. We refer to this as "temporal blindness". This limitation hinders decisions about when to invoke tools, leading agents to either over-rely on stale context and skip needed tool calls, or under-rely on it and redundantly repeat tool calls. To study this challenge, we constructed TicToc, a diverse dataset of multi-turn user-agent message trajectories across 76 scenarios, spanning dynamic environments with high, medium, and low time sensitivity. We collected human preferences between "calling a tool" and "directly answering" on each sample, and evaluated how well LLM tool-calling decisions align with human preferences under varying amounts of elapsed time. Our analysis reveals that existing models display poor alignment with human temporal perception, with no model achieving a normalized alignment rate better than 65% when given time stamp information. We also show that naive, prompt-based alignment techniques have limited effectiveness for most models, but specific post-training alignment can be a viable way to align multi-turn LLM tool use with human temporal perception. Our data and findings provide a first step toward understanding and mitigating temporal blindness, offering insights to foster the development of more time-aware and human-aligned agents.

**arXiv ID:** 2510.23853
</details>

<details>
<summary><strong>AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization</strong> - Genghan Zhang, Shaowei Zhu, Anjiang Wei, Zhenyu Song, Allen Nie, Zhen Jia, Nandita Vijaykumar, Yida Wang, Kunle Olukotun - [[pdf]](https://arxiv.org/pdf/2511.15915)</summary>

**Abstract:** We present AccelOpt, a self-improving large language model (LLM) agentic system that autonomously optimizes kernels for emerging AI acclerators, eliminating the need for expert-provided hardware-specific optimization knowledge. AccelOpt explores the kernel optimization space through iterative generation, informed by an optimization memory that curates experiences and insights from previously encountered slow-fast kernel pairs. We build NKIBench, a new benchmark suite of AWS Trainium accelerator kernels with varying complexity extracted from real-world LLM workloads to evaluate the effectiveness of AccelOpt. Our evaluation confirms that AccelOpt's capability improves over time, boosting the average percentage of peak throughput from $49\%$ to $61\%$ on Trainium 1 and from $45\%$ to $59\%$ on Trainium 2 for NKIBench kernels. Moreover, AccelOpt is highly cost-effective: using open-source models, it matches the kernel improvements of Claude Sonnet 4 while being $26\times$ cheaper. The code is open-sourced at this https URL.

**arXiv ID:** 2511.15915
</details>

<details>
<summary><strong>Does RL Expand the Capability Boundary of LLM Agents? A PASS@(k,T) Analysis</strong> - Zhiyuan Zhai, Wenjing Yan, Xiaodan Shao, Xin Wang - [[pdf]](https://arxiv.org/pdf/2604.14877)</summary>

**Abstract:** Does reinforcement learning genuinely expand what LLM agents can do, or merely make them more reliable? For static reasoning, recent work answers the second: base and RL pass@k curves converge at large k. We ask whether this holds for agentic tool use, where T rounds of interaction enable compositional strategies that re-sampling cannot recover. We introduce PASS@(k,T), a two-dimensional metric that jointly varies sampling budget k and interaction depth T, separating capability expansion from efficiency improvement. Our main finding is that, contrary to the static-reasoning result, tool-use RL genuinely enlarges the capability boundary: the RL agent's pass-curve pulls above the base model's and the gap widens at large k rather than converging. The expansion is specific to compositional, sequential information gathering; on simpler tasks RL behaves as prior work predicts. Under matched training data, supervised fine-tuning regresses the boundary on the same compositional tasks, isolating self-directed exploration as the causal factor. Mechanism analysis shows RL reweights the base strategy distribution toward the subset whose downstream reasoning more often yields a correct answer, with the improvement concentrated on how the agent integrates retrieved information. These results reconcile optimistic and pessimistic readings of RL for LLMs: both are correct, on different task types.

**arXiv ID:** 2604.14877
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (18 papers)</h2></summary>

<details>
<summary><strong>MARS$^2$: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation</strong> - Pengfei Li, Shijie Wang, Fangyuan Li, Yikun Fu, Kaifeng Liu, Kaiyan Zhang, Dazhi Zhang, Yuqiang Li, Biqing Qi, Bowen Zhou - [[pdf]](https://arxiv.org/pdf/2604.14564)</summary>

**Abstract:** Reinforcement learning (RL) paradigms have demonstrated strong performance on reasoning-intensive tasks such as code generation. However, limited trajectory diversity often leads to diminishing returns, which constrains the achievable performance ceiling. Search-enhanced RL alleviates this issue by introducing structured exploration, which remains constrained by the single-agent policy priors. Meanwhile, leveraging multiple interacting policies can acquire more diverse exploratory signals, but existing approaches are typically decoupled from structured search. We propose \textbf{MARS$^2$} (Multi-Agent Reinforced Tree-Search Scaling), a unified RL framework in which multiple independently-optimized agents collaborate within a shared tree-structured search environment. MARS$^2$ models the search tree as a learnable multi-agent interaction environment, enabling heterogeneous agents to collaboratively generate and refine candidate solutions within a shared search topology. To support effective learning, we introduce a path-level group advantage formulation based on tree-consistent reward shaping, which facilitates effective credit assignment across complex search trajectories. Experiments on code generation benchmarks show that MARS$^2$ consistently improves performance across diverse model combinations and training settings, demonstrating the effectiveness of coupling multi-agent collaboration with tree search for enhancing reinforcement learning. Our code is publicly available at this https URL.

**arXiv ID:** 2604.14564
</details>

<details>
<summary><strong>GDPR Auto-Formalization with AI Agents and Human Verification</strong> - Ha Thanh Nguyen, Wachara Fungwacharakorn, Sabine Wehnert, May Myo Zin, Yuntao Kong, Jieying Xue, Michał Araszkiewicz, Randy Goebel, Ken Satoh - [[pdf]](https://arxiv.org/pdf/2604.14607)</summary>

**Abstract:** We study the overall process of automatic formalization of GDPR provisions using large language models, within a human-in-the-loop verification framework. Rather than aiming for full autonomy, we adopt a role-specialized workflow in which LLM-based AI components, operating in a multi-agent setting with iterative feedback, generate legal scenarios, formal rules, and atomic facts. This is coupled with independent verification modules which include human reviewers' assessment of representational, logical, and legal correctness. Using this approach, we construct a high-quality dataset to be used for GDPR auto-formalization, and analyze both successful and problematic cases. Our results show that structured verification and targeted human oversight are essential for reliable legal formalization, especially in the presence of legal nuance and context-sensitive reasoning.

**arXiv ID:** 2604.14607
</details>

<details>
<summary><strong>El Agente Forjador: Task-Driven Agent Generation for Quantum Simulation</strong> - Zijian Zhang, Aiwei Yin, Amaan Baweja, Jiaru Bai, Ignacio Gustin, Varinia Bernales, Alán Aspuru-Guzik - [[pdf]](https://arxiv.org/pdf/2604.14609)</summary>

**Abstract:** AI for science promises to accelerate the discovery process. The advent of large language models (LLMs) and agentic workflows enables the expediting of a growing range of scientific tasks. However, most of the current generation of agentic systems depend on static, hand-curated toolsets that hinder adaptation to new domains and evolving libraries. We present El Agente Forjador, a multi-agent framework in which universal coding agents autonomously forge, validate, and reuse computational tools through a four-stage workflow of tool analysis, tool generation, task execution, and iterative solution evaluation. Evaluated across 24 tasks spanning quantum chemistry and quantum dynamics on five coding agent setups, we compare three operating modes: zero-shot generation of tools per task, reuse of a curriculum-built toolset, and direct problem-solving with the coding agents as the baseline. We find that our tool generation and reuse framework consistently improves accuracy over the baseline. We also show that reusing a toolset built by a stronger coding agent can reduce API cost and substantially raises the solution quality for weaker coding agents. Case studies further demonstrate that tools forged for different domains can be combined to solve hybrid tasks. Taken together, these results show that LLM-based agents can use their scientific knowledge and coding capabilities to autonomously build reusable scientific tools, pointing toward a paradigm in which agent capabilities are defined by the tasks they are designed to solve rather than by explicitly engineered implementations.

**arXiv ID:** 2604.14609
</details>

<details>
<summary><strong>M2-PALE: A Framework for Explaining Multi-Agent MCTS--Minimax Hybrids via Process Mining and LLMs</strong> - Yiyu Qian, Liyuan Zhao, Tim Miller - [[pdf]](https://arxiv.org/pdf/2604.14687)</summary>

**Abstract:** Monte-Carlo Tree Search (MCTS) is a fundamental sampling-based search algorithm widely used for online planning in sequential decision-making domains. Despite its success in driving recent advances in artificial intelligence, understanding the behavior of MCTS agents remains a challenge for both developers and users. This difficulty stems from the complex search trees produced through the simulation of numerous future states and their intricate relationships. A known weakness of standard MCTS is its reliance on highly selective tree construction, which may lead to the omission of crucial moves and a vulnerability to tactical traps. To resolve this, we incorporate shallow, full-width Minimax search into the rollout phase of multi-agent MCTS to enhance strategic depth. Furthermore, to demystify the resulting decision-making logic, we introduce \textsf{M2-PALE} (MCTS--Minimax Process-Aided Linguistic Explanations). This framework employs process mining techniques, specifically the Alpha Miner, iDHM, and Inductive Miner algorithms, to extract underlying behavioral workflows from agent execution traces. These process models are then synthesized by LLMs to generate human-readable causal and distal explanations. We demonstrate the efficacy of our approach in a small-scale checkers environment, establishing a scalable foundation for interpreting hybrid agents in increasingly complex strategic domains.

**arXiv ID:** 2604.14687
</details>

<details>
<summary><strong>Dr.~RTL: Autonomous Agentic RTL Optimization through Tool-Grounded Self-Improvement</strong> - Wenji Fang, Yao Lu, Shang Liu, Jing Wang, Ziyan Guo, Junxian He, Fengbin Tu, Zhiyao Xie - [[pdf]](https://arxiv.org/pdf/2604.14989)</summary>

**Abstract:** Recent advances in large language models (LLMs) have sparked growing interest in automatic RTL optimization for better performance, power, and area (PPA). However, existing methods are still far from realistic RTL optimization. Their evaluation settings are often unrealistic: they are tested on manually degraded, small-scale RTL designs and rely on weak open-source tools. Their optimization methods are also limited, relying on coarse design-level feedback and simple pre-defined rewriting rules. To address these limitations, we present Dr. RTL, an agentic framework for RTL timing optimization in a realistic evaluation environment, with continual self-improvement through reusable optimization skills. We establish a realistic evaluation setting with more challenging RTL designs and an industrial EDA workflow. Within this setting, Dr. RTL performs closed-loop optimization through a multi-agent framework for critical-path analysis, parallel RTL rewriting, and tool-based evaluation. We further introduce group-relative skill learning, which compares parallel RTL rewrites and distills the optimization experience into an interpretable skill library. Currently, this library contains 47 pattern--strategy entries for cross-design reuse to improve PPA and accelerate convergence, and it can continue evolving over time. Evaluated on 20 real-world RTL designs, Dr. RTL achieves average WNS/TNS improvements of 21\%/17\% with a 6\% area reduction over the industry-leading commercial synthesis tool.

**arXiv ID:** 2604.14989
</details>

<details>
<summary><strong>Autogenesis: A Self-Evolving Agent Protocol</strong> - Wentao Zhang - [[pdf]](https://arxiv.org/pdf/2604.15034)</summary>

**Abstract:** Recent advances in LLM based agent systems have shown promise in tackling complex, long horizon tasks. However, existing agent protocols (e.g., A2A and MCP) under specify cross entity lifecycle and context management, version tracking, and evolution safe update interfaces, which encourages monolithic compositions and brittle glue code. We introduce \textbf{\textsc{Autogenesis Protocol (AGP)}}, a self evolution protocol that decouples what evolves from how evolution occurs. Its Resource Substrate Protocol Layer (RSPL) models prompts, agents, tools, environments, and memory as protocol registered resources\footnote{Unless otherwise specified, resources refer to instances of the five RSPL entity types: \emph{prompt}, \emph{agent}, \emph{tool}, \emph{environment}, \emph{memory} with agent \emph{outputs}.} with explicit state, lifecycle, and versioned interfaces. Its Self Evolution Protocol Layer (SEPL) specifies a closed loop operator interface for proposing, assessing, and committing improvements with auditable lineage and rollback. Building on \textbf{\textsc{AGP}}, we present \textbf{\textsc{Autogenesis System (AGS)}}, a self-evolving multi-agent system that dynamically instantiates, retrieves, and refines protocol-registered resources during execution. We evaluate \textbf{\textsc{AGS}} on multiple challenging benchmarks that require long horizon planning and tool use across heterogeneous resources. The results demonstrate consistent improvements over strong baselines, supporting the effectiveness of agent resource management and closed loop self evolution.

**arXiv ID:** 2604.15034
</details>

<details>
<summary><strong>Where are the Humans? A Scoping Review of Fairness in Multi-agent AI Systems</strong> - Simeon Allmendinger, Luca Deck, Lucas Mueller - [[pdf]](https://arxiv.org/pdf/2604.15078)</summary>

**Abstract:** Rapid advances in Generative AI are giving rise to increasingly sophisticated Multi-Agent AI (MAAI) systems. While AI fairness has been extensively studied in traditional predictive scenarios, its examination in MAAI remains nascent and fragmented. This scoping review critically synthesizes existing research on fairness in MAAI systems. Through a qualitative content analysis of 23 selected studies, we identify five archetypal approaches. Our findings reveal that fairness in MAAI systems is often addressed superficially, lacks robust normative foundations, and frequently overlooks the complex dynamics introduced by agent autonomy and system-level interactions. We argue that fairness must be embedded structurally throughout the development lifecycle of MAAI, rather than appended as a post-hoc consideration. Meaningful evaluation requires explicit human oversight, normative clarity, and a precise articulation of fairness objectives and beneficiaries. This review provides a foundation for advancing fairness research in MAAI systems by highlighting critical gaps, exposing prevailing limitations, and suggesting pathways.

**arXiv ID:** 2604.15078
</details>

<details>
<summary><strong>TRACE: A Conversational Framework for Sustainable Tourism Recommendation with Agentic Counterfactual Explanations</strong> - Ashmi Banerjee, Adithi Satish, Wolfgang Wörndl, Yashar Deldjoo - [[pdf]](https://arxiv.org/pdf/2604.14223)</summary>

**Abstract:** Traditional conversational travel recommender systems primarily optimize for user relevance and convenience, often reinforcing popular, overcrowded destinations and carbon-intensive travel choices. To address this, we present TRACE (Tourism Recommendation with Agentic Counterfactual Explanations), a multi-agent, LLM-based framework that promotes sustainable tourism through interactive nudging. TRACE uses a modular orchestrator-worker architecture where specialized agents elicit latent sustainability preferences, construct structured user personas, and generate recommendations that balance relevance with environmental impact.
A key innovation lies in its use of agentic counterfactual explanations and LLM-driven clarifying questions, which together surface greener alternatives and refine understanding of intent, fostering user reflection without coercion. User studies and semantic alignment analyses demonstrate that TRACE effectively supports sustainable decision-making while preserving recommendation quality and interactive responsiveness. TRACE is implemented on Google's Agent Development Kit, with full code, Docker setup, prompts, and a publicly available demo video to ensure reproducibility.
A project summary, including all resources, prompts, and demo access, is available at this https URL.

**arXiv ID:** 2604.14223
</details>

<details>
<summary><strong>ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents</strong> - Zhuofeng Li, Yi Lu, Dongfu Jiang, Haoxiang Zhang, Yuyang Bai, Chuan Li, Yu Wang, Shuiwang Ji, Jianwen Xie, Yu Zhang - [[pdf]](https://arxiv.org/pdf/2604.14261)</summary>

**Abstract:** The rapid rise in AI conference submissions has driven increasing exploration of large language models (LLMs) for peer review support. However, LLM-based reviewers often generate superficial, formulaic comments lacking substantive, evidence-grounded feedback. We attribute this to the underutilization of two key components of human reviewing: explicit rubrics and contextual grounding in existing work. To address this, we introduce REVIEWBENCH, a benchmark evaluating review text according to paper-specific rubrics derived from official guidelines, the paper's content, and human-written reviews. We further propose REVIEWGROUNDER, a rubric-guided, tool-integrated multi-agent framework that decomposes reviewing into drafting and grounding stages, enriching shallow drafts via targeted evidence consolidation. Experiments on REVIEWBENCH show that REVIEWGROUNDER, using a Phi-4-14B-based drafter and a GPT-OSS-120B-based grounding stage, consistently outperforms baselines with substantially stronger/larger backbones (e.g., GPT-4.1 and DeepSeek-R1-670B) in both alignment with human judgments and rubric-based review quality across 8 dimensions. The code is available \href{this https URL}{here}.

**arXiv ID:** 2604.14261
</details>

<details>
<summary><strong>Aerial Multi-Functional RIS in Fluid Antennas-Aided Full-Duplex Networks: A Self-Optimized Hybrid Deep Reinforcement Learning Approach</strong> - Li-Hsiang Shen, Yu-Quan Zheng - [[pdf]](https://arxiv.org/pdf/2604.14309)</summary>

**Abstract:** To address high data traffic demands of sixth-generation (6G) networks, this paper proposes a novel architecture that integrates autonomous aerial vehicles (AAVs) and multi-functional reconfigurable intelligent surfaces (MF-RISs) as AM-RIS in fluid antenna (FA)-assisted full-duplex (FD) networks. The AM-RIS provides hybrid functionalities, including signal reflection, amplification, and energy harvesting (EH), potentially improving both signal coverage and sustainability. Meanwhile, FA facilitates fine-grained spatial adaptability at FD-enabled base station (BS), which complements residual self-interference (SI) suppression. We aim at maximizing the overall energy efficiency (EE) by jointly optimizing transmit DL beamforming at BS, UL user power, configuration of AM-RIS, and positions of the FA and AM-RIS. Owing to the hybrid continuous-discrete parameters and high dimensionality of the intractable problem, we have conceived a self-optimized multi-agent hybrid deep reinforcement learning (DRL) framework (SOHRL), which integrates multi-agent deep Q-networks (DQN) and multi-agent proximal policy optimization (PPO), respectively handling discrete and continuous actions. To enhance self-adaptability, an attention-driven state representation and meta-level hyperparameter optimization are incorporated, enabling multi-agents to autonomously adjust learning hyperparameters. Simulation results validate the effectiveness of the proposed AM-RIS-enabled FA-aided FD networks empowered by SOHRL algorithm. The results reveal that SOHRL outperforms benchmarks of the case without attention mechanism and conventional hybrid/multi-agent/standalone DRL. Moreover, AM-RIS in FD achieves the highest EE compared to half-duplex, conventional rigid antenna arrays, partial EH, and conventional RIS without amplification, highlighting its potential as a compelling solution for EE-aware wireless networks.

**arXiv ID:** 2604.14309
</details>

<details>
<summary><strong>Coalition Formation in LLM Agent Networks: Stability Analysis and Convergence Guarantees</strong> - Dongxin Guo, Jikun Wu, Siu-Ming Yiu - [[pdf]](https://arxiv.org/pdf/2604.14386)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly deployed in multi-agent systems requiring strategic coordination. While recent work has analyzed LLM behavior in two-player games, coalition formation, where $n$ agents dynamically form cooperative groups, remains theoretically uncharacterized. We present the first framework grounding coalition formation in LLM agent networks in hedonic game theory with formal stability guarantees. We introduce the LLM Coalition Formation Game (LCFG), establish sufficient conditions for Nash-stable partitions, and prove complexity results. Our analysis reveals that LLM agents exhibit bounded rationality characterized by $\epsilon$-rational preferences; we provide both deterministic existence guarantees and consistency-driven stability bounds whose predictions are consistent with empirical outcomes. Experiments with GPT-4, Claude-3, and Llama-3 across 2,400 episodes validate our framework: LLM coalitions achieve Nash stability in 73.2% of cases under our Coalition-of-Thought (CoalT) protocol, compared to 58.4% under chain-of-thought and 41.8% under standard prompting ($p < 0.001$). Our framework provides theoretical foundations for designing stable multi-agent LLM systems.

**arXiv ID:** 2604.14386
</details>

<details>
<summary><strong>VeriGraphi: A Multi-Agent Framework of Hierarchical RTL Generation for Large Hardware Designs</strong> - Sazzadul Islam, Tasnim Tabassum, Hao Zheng - [[pdf]](https://arxiv.org/pdf/2604.14550)</summary>

**Abstract:** Generating synthesizable Verilog for large, hierarchical hardware designs remains a significant challenge for large language models (LLMs), which struggle to replicate the structured reasoning that human experts employ when translating complex specifications into RTL. When tasked with producing hierarchical Verilog, LLMs frequently lose context across modules, hallucinate interfaces, fabricate inter-module wiring, and fail to maintain structural coherence - failures that intensify as design complexity grows and specifications involve informal prose, figures, and tables that resist direct operationalization. To address these challenges, we present VeriGraphi, a framework that introduces a spec-anchored Knowledge Graph as the architectural substrate driving the RTL generation pipeline. VeriGraphi constructs a HDA, a structured knowledge graph that explicitly encodes module hierarchy, port-level interfaces, wiring semantics, and inter-module dependencies as first-class graph entities and relations. Built through iterative multi-agent analysis of the specification, this Knowledge Graph provides a deterministic, machine-checkable structural scaffold before code generation. Guided by the KG, a progressive coding module incrementally generates pseudo-code and synthesizable RTL while enforcing interface consistency and dependency correctness at each submodule stage. We evaluate VeriGraphi on a benchmark of three representative specification documents from the National Institute of Standards and Technology and their corresponding implementations, and we present a RV32I processor as a detailed case study to illustrate the full pipeline. The results demonstrate that VeriGraphi enables reliable hierarchical RTL generation with minimal human intervention for RISC-V, marking a significant milestone for LLM-generated hardware design while maintaining strong functional correctness.

**arXiv ID:** 2604.14550
</details>

<details>
<summary><strong>CoGrid & the Multi-User Gymnasium: A Framework for Multi-Agent Experimentation</strong> - Chase McDonald, Cleotilde Gonzalez - [[pdf]](https://arxiv.org/pdf/2604.15044)</summary>

**Abstract:** The increasing integration of artificial intelligence (AI) in everyday life brings with it new challenges and questions for regarding how humans interact with autonomous agents. Multi-agent experiments, where humans and AI act together, can offer important opportunities to study social decision making, but there is a lack of accessible tooling available to researchers to run such experiments. We introduce two tools designed to reduce these barriers. The first, CoGrid, is a multi-agent grid-based simulation library with dual NumPy and JAX backends. The second, Multi-User Gymnasium (MUG), translates such simulation environments directly into interactive web-based experiments. MUG supports interactions with arbitrary numbers of humans and AI, utilizing either server-authoritative or peer-to-peer networking with rollback netcode to account for latency. Together, these tools can enable researchers to deploy studies of human-AI interaction, facilitating inquiry into core questions of psychology, cognition, and decision making and their relationship to human-AI interaction. Both tools are open source and available to the broader research community. Documentation and source code is available at {cogrid, multi-user-gymnasium}.this http URL. This paper details the functionality of these tools and presents several case studies to illustrate their utility in human-AI multi-agent experimentation.

**arXiv ID:** 2604.15044
</details>

<details>
<summary><strong>Autonomous Evolution of EDA Tools: Multi-Agent Self-Evolved ABC</strong> - Cunxi Yu, Haoxing Ren - [[pdf]](https://arxiv.org/pdf/2604.15082)</summary>

**Abstract:** This paper introduces the first \emph{self-evolving} logic synthesis framework, which leverages Large Language Model (LLM) agents to autonomously improve the source code of \textsc{ABC}, the widely adopted logic synthesis system. Our framework operates on the \emph{entire integrated ABC codebase}, and the output repository preserves its single-binary execution model and command interface. In the initial evolution cycle, we bootstrap the system using existing prior open-source synthesis components, covering flow tuning, logic minimization, and technology mapping, but without manually injecting new heuristics. On top of this foundation, a team of LLM-based agents iteratively rewrites and evolves specific sub-components of ABC following our ``programming guidance`` prompts under a unified correctness and QoR-driven evaluation loop. Each evolution cycle proposes code modifications, compiles the integrated binary, validates correctness, and evaluates quality-of-results (QoR) on \emph{multi-suite benchmarks including ISCAS~85/89/99, VTR, EPFL, and IWLS~2005}. Through continuous feedback, the system discovers optimizations beyond human-designed heuristics, effectively \emph{learning new synthesis strategies} that enhance QoR. We detail the architecture of this self-improving system, its integration with \textsc{ABC}, and results demonstrating that the framework can autonomously and progressively improve EDA tool at full million-line scale.

**arXiv ID:** 2604.15082
</details>

<details>
<summary><strong>One Step is Enough: Multi-Agent Reinforcement Learning based on One-Step Policy Optimization for Order Dispatch on Ride-Sharing Platforms</strong> - Zijian Zhao, Sen Li - [[pdf]](https://arxiv.org/pdf/2507.15351)</summary>

**Abstract:** Order dispatch is a critical task in ride-sharing systems with Autonomous Vehicles (AVs), directly influencing efficiency and profits. Recently, Multi-Agent Reinforcement Learning (MARL) has emerged as a promising solution to this problem by decomposing the large state and action spaces among individual agents, effectively addressing the Curse of Dimensionality (CoD) in transportation market, which is caused by the substantial number of vehicles, passengers, and orders. However, conventional MARL-based approaches heavily rely on accurate estimation of the value function, which becomes problematic in large-scale, highly uncertain environments. To address this issue, we propose two novel methods that bypass value function estimation, leveraging the homogeneous property of AV fleets. First, we draw an analogy between AV fleets and groups in Group Relative Policy Optimization (GRPO), adapting it to the order dispatch task. By replacing the Proximal Policy Optimization (PPO) baseline with the group average reward-to-go, GRPO eliminates critic estimation errors and reduces training bias. Inspired by this baseline replacement, we further propose One-Step Policy Optimization (OSPO), demonstrating that the optimal policy can be trained using only one-step group rewards under a homogeneous fleet. Experiments on a real-world ride-hailing dataset show that both GRPO and OSPO achieve promising performance across all scenarios, efficiently optimizing pickup times and the number of served orders using simple Multilayer Perceptron (MLP) networks. Furthermore, OSPO outperforms GRPO in all scenarios, attributed to its elimination of bias caused by the bounded time horizon of GRPO. Our code, trained models, and processed data are provided at this https URL .

**arXiv ID:** 2507.15351
</details>

<details>
<summary><strong>TopoDIM: One-shot Topology Generation of Diverse Interaction Modes for Multi-Agent Systems</strong> - Rui Sun, Jie Ding, Chenghua Gong, Tianjun Gu, Yihang Jiang, Juyuan Zhang, Liming Pan, Linyuan Lü - [[pdf]](https://arxiv.org/pdf/2601.10120)</summary>

**Abstract:** Optimizing communication topology in LLM-based multi-agent system is critical for enabling collective intelligence. Existing methods mainly rely on spatio-temporal interaction paradigms, where the sequential execution of multi-round dialogues incurs high latency and computation. Motivated by the recent insights that evaluation and debate mechanisms can improve problem-solving in multi-agent systems, we propose TopoDIM, a framework for one-shot Topology generation with Diverse Interaction Modes. Designed for decentralized execution to enhance adaptability and privacy, TopoDIM enables agents to autonomously construct heterogeneous communication without iterative coordination, achieving token efficiency and improved task performance. Experiments demonstrate that TopoDIM reduces total token consumption by 46.41% while improving average performance by 1.50% over state-of-the-art methods. Moreover, the framework exhibits strong adaptability in organizing communication among heterogeneous agents. Code is available at: this https URL.

**arXiv ID:** 2601.10120
</details>

<details>
<summary><strong>LLMOrbit: A Circular Taxonomy of Large Language Models -From Scaling Walls to Agentic AI Systems</strong> - Badri N. Patro, Vijay S. Agneeswaran - [[pdf]](https://arxiv.org/pdf/2601.14053)</summary>

**Abstract:** The field of artificial intelligence has undergone a revolution from foundational Transformer architectures to reasoning-capable systems approaching human-level performance. We present LLMOrbit, a comprehensive circular taxonomy navigating the landscape of large language models spanning 2019-2025. This survey examines over 50 models across 15 organizations through eight interconnected orbital dimensions, documenting architectural innovations, training methodologies, and efficiency patterns defining modern LLMs, generative AI, and agentic systems. We identify three critical crises: (1) data scarcity (9-27T tokens depleted by 2026-2028), (2) exponential cost growth ($3M to $300M+ in 5 years), and (3) unsustainable energy consumption (22x increase), establishing the scaling wall limiting brute-force approaches. Our analysis reveals six paradigms breaking this wall: (1) test-time compute (o1, DeepSeek-R1 achieve GPT-4 performance with 10x inference compute), (2) quantization (4-8x compression), (3) distributed edge computing (10x cost reduction), (4) model merging, (5) efficient training (ORPO reduces memory 50%), and (6) small specialized models (Phi-4 14B matches larger models). Three paradigm shifts emerge: (1) post-training gains (RLHF, GRPO, pure RL contribute substantially, DeepSeek-R1 achieving 79.8% MATH), (2) efficiency revolution (MoE routing 18x efficiency, Multi-head Latent Attention 8x KV cache compression enables GPT-4-level performance at $<$$0.30/M tokens), and (3) democratization (open-source Llama 3 88.6% MMLU surpasses GPT-4 86.4%). We provide insights into techniques (RLHF, PPO, DPO, GRPO, ORPO), trace evolution from passive generation to tool-using agents (ReAct, RAG, multi-agent systems), and analyze post-training innovations.

**arXiv ID:** 2601.14053
</details>

<details>
<summary><strong>OccuBench: Evaluating AI Agents on Real-World Professional Tasks via Language Environment Simulation</strong> - Xiaomeng Hu, Yinger Zhang, Fei Huang, Jianhong Tu, Yang Su, Lianghao Deng, Yuxuan Liu, Yantao Liu, Dayiheng Liu, Tsung-Yi Ho - [[pdf]](https://arxiv.org/pdf/2604.10866)</summary>

**Abstract:** AI agents are expected to perform professional work across hundreds of occupational domains (from emergency department triage to nuclear reactor safety monitoring to customs import processing), yet existing benchmarks can only evaluate agents in the few domains where public environments exist. We introduce OccuBench, a benchmark covering 100 real-world professional task scenarios across 10 industry categories and 65 specialized domains, enabled by Language Environment Simulators (LESs) that simulate domain-specific environments through LLM-driven tool response generation. Our multi-agent synthesis pipeline automatically produces evaluation instances with guaranteed solvability, calibrated difficulty, and document-grounded diversity. OccuBench evaluates agents along two complementary dimensions: task completion across professional domains and environmental robustness under controlled fault injection (explicit errors, implicit data degradation, and mixed faults). We evaluate 15 frontier models across 8 model families and find that: (1) no single model dominates all industries, as each has a distinct occupational capability profile; (2) implicit faults (truncated data, missing fields) are harder than both explicit errors (timeouts, 500s) and mixed faults, because they lack overt error signals and require the agent to independently detect data degradation; (3) larger models, newer generations, and higher reasoning effort consistently improve performance. GPT-5.2 improves by 27.5 points from minimal to maximum reasoning effort; and (4) strong agents are not necessarily strong environment simulators. Simulator quality is critical for LES-based evaluation reliability. OccuBench provides the first systematic cross-industry evaluation of AI agents on professional occupational tasks.

**arXiv ID:** 2604.10866
</details>

</details>

<details open>
<summary><h2>Other Agent Research (16 papers)</h2></summary>

<details>
<summary><strong>Layered Mutability: Continuity and Governance in Persistent Self-Modifying Agents</strong> - Krti Tallam - [[pdf]](https://arxiv.org/pdf/2604.14717)</summary>

**Abstract:** Persistent language-model agents increasingly combine tool use, tiered memory, reflective prompting, and runtime adaptation. In such systems, behavior is shaped not only by current prompts but by mutable internal conditions that influence future action. This paper introduces layered mutability, a framework for reasoning about that process across five layers: pretraining, post-training alignment, self-narrative, memory, and weight-level adaptation. The central claim is that governance difficulty rises when mutation is rapid, downstream coupling is strong, reversibility is weak, and observability is low, creating a systematic mismatch between the layers that most affect behavior and the layers humans can most easily inspect. I formalize this intuition with simple drift, governance-load, and hysteresis quantities, connect the framework to recent work on temporal identity in language-model agents, and report a preliminary ratchet experiment in which reverting an agent's visible self-description after memory accumulation fails to restore baseline behavior. In that experiment, the estimated identity hysteresis ratio is 0.68. The main implication is that the salient failure mode for persistent self-modifying agents is not abrupt misalignment but compositional drift: locally reasonable updates that accumulate into a behavioral trajectory that was never explicitly authorized.

**arXiv ID:** 2604.14717
</details>

<details>
<summary><strong>Toward Agentic RAG for Ukrainian</strong> - Marta Sumyk, Oleksandr Kosovan - [[pdf]](https://arxiv.org/pdf/2604.14896)</summary>

**Abstract:** We present an initial investigation into Agentic Retrieval-Augmented Generation (RAG) for Ukrainian, conducted within the UNLP 2026 Shared Task on Multi-Domain Document Understanding. Our system combines two-stage retrieval (BGE-M3 with BGE reranking) with a lightweight agentic layer performing query rephrasing and answer-retry loops on top of Qwen2.5-3B-Instruct. Our analysis reveals that retrieval quality is the primary bottleneck: agentic retry mechanisms improve answer accuracy but the overall score remains constrained by document and page identification. We discuss practical limitations of offline agentic pipelines and outline directions for combining stronger retrieval with more advanced agentic reasoning for Ukrainian.

**arXiv ID:** 2604.14896
</details>

<details>
<summary><strong>Agent-Aided Design for Dynamic CAD Models</strong> - Mitch Adler, Matthew Russo, Michael Cafarella - [[pdf]](https://arxiv.org/pdf/2604.15184)</summary>

**Abstract:** In the past year, researchers have started to create agentic systems that can design real-world CAD-style objects in a training-free setting, a new variety of system that we call Agent-Aided Design. Generally speaking, these systems place an agent in a feedback loop in which it can write code, compile that code to an assembly of CAD model(s), visualize the model, and then iteratively refine its code based on visual and other feedback. Despite rapid progress, a key problem remains: none of these systems can build complex 3D assemblies with moving parts. For example, no existing system can build a piston, a pendulum, or even a pair of scissors. In order for Agent-Aided Design to make a real impact in industrial manufacturing, we need a system that is capable of generating such 3D assemblies. In this paper we present a prototype of AADvark, an agentic system designed for this task. Unlike previous state-of-the-art systems, AADvark captures the dynamic part interactions with one or more degrees-of-freedom. This design decision allows AADvark to reason directly about assemblies with moving parts and can thereby achieve cross-cutting goals, including but not limited to mechanical movements. Unfortunately, current LLMs are imperfect spatial reasoners, a problem that AADvark addresses by incorporating external constraint solver tools with a specialized visual feedback mechanism. We demonstrate that, by modifying the agent's tools (FreeCAD and the assembly solver), we are able to create a strong verification signal which enables our system to build 3D assemblies with movable parts.

**arXiv ID:** 2604.15184
</details>

<details>
<summary><strong>RadAgent: A tool-using AI agent for stepwise interpretation of chest computed tomography</strong> - Mélanie Roschewitz, Kenneth Styppa, Yitian Tao, Jiwoong Sohn, Jean-Benoit Delbrouck, Benjamin Gundersen, Nicolas Deperrois, Christian Bluethgen, Julia Vogt, Bjoern Menze, Farhad Nooralahzadeh, Michael Krauthammer, Michael Moor - [[pdf]](https://arxiv.org/pdf/2604.15231)</summary>

**Abstract:** Vision-language models (VLM) have markedly advanced AI-driven interpretation and reporting of complex medical imaging, such as computed tomography (CT). Yet, existing methods largely relegate clinicians to passive observers of final outputs, offering no interpretable reasoning trace for them to inspect, validate, or refine. To address this, we introduce RadAgent, a tool-using AI agent that generates CT reports through a stepwise and interpretable process. Each resulting report is accompanied by a fully inspectable trace of intermediate decisions and tool interactions, allowing clinicians to examine how the reported findings are derived. In our experiments, we observe that RadAgent improves Chest CT report generation over its 3D VLM counterpart, CT-Chat, across three dimensions. Clinical accuracy improves by 6.0 points (36.4% relative) in macro-F1 and 5.4 points (19.6% relative) in micro-F1. Robustness under adversarial conditions improves by 24.7 points (41.9% relative). Furthermore, RadAgent achieves 37.0% in faithfulness, a new capability entirely absent in its 3D VLM counterpart. By structuring the interpretation of chest CT as an explicit, tool-augmented and iterative reasoning trace, RadAgent brings us closer toward transparent and reliable AI for radiology.

**arXiv ID:** 2604.15231
</details>

<details>
<summary><strong>Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems</strong> - Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, Zhiqiang Shen - [[pdf]](https://arxiv.org/pdf/2604.14228)</summary>

**Abstract:** Claude Code is an agentic coding tool that can run shell commands, edit files, and call external services on behalf of the user. This study describes its comprehensive architecture by analyzing the publicly available TypeScript source code and further comparing it with OpenClaw, an independent open-source AI agent system that answers many of the same design questions from a different deployment context. Our analysis identifies five human values, philosophies, and needs that motivate the architecture (human decision authority, safety and security, reliable execution, capability amplification, and contextual adaptability) and traces them through thirteen design principles to specific implementation choices. The core of the system is a simple while-loop that calls the model, runs tools, and repeats. Most of the code, however, lives in the systems around this loop: a permission system with seven modes and an ML-based classifier, a five-layer compaction pipeline for context management, four extensibility mechanisms (MCP, plugins, skills, and hooks), a subagent delegation mechanism with worktree isolation, and append-oriented session storage. A comparison with OpenClaw, a multi-channel personal assistant gateway, shows that the same recurring design questions produce different architectural answers when the deployment context changes: from per-action safety classification to perimeter-level access control, from a single CLI loop to an embedded runtime within a gateway control plane, and from context-window extensions to gateway-wide capability registration. We finally identify six open design directions for future agent systems, grounded in recent empirical, architectural, and policy literature.

**arXiv ID:** 2604.14228
</details>

<details>
<summary><strong>Challenges and Future Directions in Agentic Reverse Engineering Systems</strong> - Salem Radey, Jack West, Kassem Fawaz - [[pdf]](https://arxiv.org/pdf/2604.14317)</summary>

**Abstract:** Agentic systems built on large language models (LLMs) are increasingly being used for complex security tasks, including binary reverse engineering (RE). Despite recent growth in popularity and capability, these systems continue to face limitations in realistic settings. Cutting-edge systems still fail in complex RE scenarios that involve obfuscation, timing, and unique architecture. In this work, we examine how agentic systems perform reverse engineering tasks with static, dynamic, and hybrid agents. Through an analysis of existing agentic tool usage, we identify several limitations, including token constraints, struggles with obfuscation, and a lack of program guardrails. From these findings, we outline current challenges and position future directions for system designers to overcome from a security perspective.

**arXiv ID:** 2604.14317
</details>

<details>
<summary><strong>APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI</strong> - Pratyay Banerjee, Masud Moshtaghi, Shivashankar Subramanian, Amita Misra, Ankit Chadha - [[pdf]](https://arxiv.org/pdf/2604.14362)</summary>

**Abstract:** Large language models still struggle with reliable long-term conversational memory: simply enlarging context windows or applying naive retrieval often introduces noise and destabilizes responses. We present APEX-MEM, a conversational memory system that combines three key innovations: (1) a property graph which uses domain-agnostic ontology to structure conversations as temporally grounded events in an entity-centric framework, (2) append-only storage that preserves the full temporal evolution of information, and (3) a multi-tool retrieval agent that understands and resolves conflicting or evolving information at query time, producing a compact and contextually relevant memory summary. This retrieval-time resolution preserves the full interaction history while suppressing irrelevant details. APEX-MEM achieves 88.88% accuracy on LOCOMO's Question Answering task and 86.2% on LongMemEval, outperforming state-of-the-art session-aware approaches and demonstrating that structured property graphs enable more temporally coherent long-term conversational reasoning.

**arXiv ID:** 2604.14362
</details>

<details>
<summary><strong>Modular Continual Learning via Zero-Leakage Reconstruction Routing and Autonomous Task Discovery</strong> - Noureddine Kermiche - [[pdf]](https://arxiv.org/pdf/2604.14375)</summary>

**Abstract:** Catastrophic forgetting remains a primary hurdle in sequential task learning for artificial neural networks. We propose a silicon-native modular architecture that achieves structural parameter isolation using Task-Specific Experts and a distributed, outlier-based Gatekeeper. Moving beyond traditional sequential consolidation, our framework utilizes a Simultaneous Pipeline where Teacher learning, Student distillation, and Router manifold acquisition occur in parallel while raw data is present in a localized training session. This approach ensures computational efficiency and complies with privacy mandates like GDPR by deleting raw data as soon as a task is learned. We demonstrate that a Tight-Bottleneck Autoencoder (TB-AE) can effectively distinguish semantically crowded manifolds in high-dimensional latent spaces, overcoming the posterior collapse inherent to standard variational methods. By establishing strict topological boundaries, our TB-AE resolves latent space crowding in 4096-D LLM embeddings to provide a robust, unsupervised novelty signal. Furthermore, we validate an Autonomous Retrieval mechanism that confidently identifies returning manifolds, enabling stable lifelong learning without redundant module instantiation. Empirical results demonstrate that our ``Live Distillation'' approach acts as a natural regularizer, achieving strong retention across computer vision and natural language processing domains without suffering a student fidelity gap.

**arXiv ID:** 2604.14375
</details>

<details>
<summary><strong>SpaceMind: A Modular and Self-Evolving Embodied Vision-Language Agent Framework for Autonomous On-orbit Servicing</strong> - Aodi Wu, Haodong Han, Xubo Luo, Ruisuo Wang, Shan He, Xue Wan - [[pdf]](https://arxiv.org/pdf/2604.14399)</summary>

**Abstract:** Autonomous on-orbit servicing demands embodied agents that perceive through visual sensors, reason about 3D spatial situations, and execute multi-phase tasks over extended horizons. We present SpaceMind, a modular and self-evolving vision-language model (VLM) agent framework that decomposes knowledge, tools, and reasoning into three independently extensible dimensions: skill modules with dynamic routing, Model Context Protocol (MCP) tools with configurable profiles, and injectable reasoning-mode skills. An MCP-Redis interface layer enables the same codebase to operate across simulation and physical hardware without modification, and a Skill Self-Evolution mechanism distills operational experience into persistent skill files without model fine-tuning. We validate SpaceMind through 192 closed-loop runs across five satellites, three task types, and two environments, a UE5 simulation and a physical laboratory, deliberately including degraded conditions to stress-test robustness. Under nominal conditions all modes achieve 90--100% navigation success; under degradation, the Prospective mode uniquely succeeds in search-and-approach tasks where other modes fail. A self-evolution study shows that the agent recovers from failure in four of six groups from a single failed episode, including complete failure to 100% success and inspection scores improving from 12 to 59 out of 100. Real-world validation confirms zero-code-modification transfer to a physical robot with 100% rendezvous success. Code: this https URL

**arXiv ID:** 2604.14399
</details>

<details>
<summary><strong>Agentic Explainability at Scale: Between Corporate Fears and XAI Needs</strong> - Yomna Elsayed, Cecily Jones - [[pdf]](https://arxiv.org/pdf/2604.14984)</summary>

**Abstract:** As companies enter the race for agentic AI adoption, fears surface around agentic autonomy and its subsequent risks. These fears compound as companies scale their agentic AI adoption with low-code applications, without a comparable scaling in their governance processes and expertise resulting in a phenomenon known as "Agent Sprawl". While shadow AI tools can help with agentic discovery and identification, few observability tools offer insights into the agents' configuration and settings or the decision-making process during agent-to-agent communication and orchestration. This paper explores AI governance professionals' concerns in enterprise settings, while offering design-time and runtime explainability techniques as suggested by AI governance experts for addressing those fears. Finally, we provide a preliminary prototype of an Agentic AI Card that can help companies feel at ease deploying agents at scale.

**arXiv ID:** 2604.14984
</details>

<details>
<summary><strong>Agentic Microphysics: A Manifesto for Generative AI Safety</strong> - Federico Pierucci, Matteo Prandi, Marcantonio Bracale Syrnikov, Marcello Galisai, Piercosma Bisconti - [[pdf]](https://arxiv.org/pdf/2604.15236)</summary>

**Abstract:** This paper advances a methodological proposal for safety research in agentic AI. As systems acquire planning, memory, tool use, persistent identity, and sustained interaction, safety can no longer be analysed primarily at the level of the isolated model. Population-level risks arise from structured interaction among agents, through processes of communication, observation, and mutual influence that shape collective behaviour over time. As the object of analysis shifts, a methodological gap emerges. Approaches focused either on single agents or on aggregate outcomes do not identify the interaction-level mechanisms that generate collective risks or the design variables that control them. A framework is required that links local interaction structure to population-level dynamics in a causally explicit way, allowing both explanation and intervention. We introduce two linked concepts. Agentic microphysics defines the level of analysis: local interaction dynamics where one agent's output becomes another's input under specific protocol conditions. Generative safety defines the methodology: growing phenomena and elicit risks from micro-level conditions to identify sufficient mechanisms, detect thresholds, and design effective interventions.

**arXiv ID:** 2604.15236
</details>

<details>
<summary><strong>Implications of zero-growth economics analysed with an agent-based model</strong> - Dylan C. Terry-Doyle, Adam B. Barrett - [[pdf]](https://arxiv.org/pdf/2501.19168)</summary>

**Abstract:** The breaching of planetary boundaries and the potentially catastrophic consequences of climate change are leading researchers to question the endless pursuit of economic growth. Several macroeconomic modelling studies have now examined whether a zero-growth trajectory in a capitalist system with interest-bearing debt can be economically stable, with mixed results. However, stability has not previously been explored at the microeconomic level, where it is important to know the consequences of zero-growth on e.g., distribution of firm sizes, market instability and risk of individual firm bankruptcy. Here we address this by developing an agent-based model incorporating Minskyan financial dynamics, the Post-Growth DYNamic Agent-based MINskyan (PG-DYNAMIN) model, and carrying out simultaneous macro- and microeconomic analyses. Accounting for the fact that growing capitalist economies are unstable and produce crises, we compare the relative stability of growth and zero-growth scenarios. This is achieved by tweaking an exogenous productivity parameter. We find zero-growth scenarios are viable yet exhibit distinct dynamics from growth scenarios. Under zero-growth, GDP was less volatile, there was reduced systemic risk in the credit network, lower unemployment rates, a higher wages share of GDP for workers, lower corporate debt to GDP ratio, and a reduction in market instability. Additionally, there was a higher rate of inflation, lower profit share of GDP for firms, increased market concentration, more economic crises with higher severity, and increased default probabilities for firms during periods of crises.

**arXiv ID:** 2501.19168
</details>

<details>
<summary><strong>Theory of Mind in Action: The Instruction Inference Task in Dynamic Human-Agent Collaboration</strong> - Fardin Saad, Pradeep K. Murukannaiah, Munindar P. Singh - [[pdf]](https://arxiv.org/pdf/2507.02935)</summary>

**Abstract:** Successful human-agent teaming relies on an agent being able to understand instructions given by a (human) principal. In many cases, an instruction may be incomplete or ambiguous. In such cases, the agent must infer the unspoken intentions from their shared context, that is, it must exercise the principal's Theory of Mind (ToM) and infer the mental states of its principal. We consider the prospects of effective human-agent collaboration using large language models (LLMs). To assess ToM in a dynamic, goal-oriented, and collaborative environment, we introduce a novel task, Instruction Inference, in which an agent assists a principal in reaching a goal by interpreting incomplete or ambiguous instructions.
We present Tomcat, an LLM-based agent, designed to exhibit ToM reasoning in interpreting and responding to the principal's this http URL implemented two variants of Tomcat. One, dubbed Fs-CoT (Fs for few-shot, CoT for chain-of-thought), is based on a small number of examples demonstrating the requisite structured reasoning. One, dubbed CP (commonsense prompt), relies on commonsense knowledge and information about the problem. We realized both variants of Tomcat on three leading LLMs, namely, GPT-4o, DeepSeek-R1, and Gemma-3-27B. To evaluate the effectiveness of Tomcat, we conducted a study with 52 human participants in which we provided participants with the same information as the CP variant. We computed intent accuracy, action optimality, and planning optimality to measure the ToM capabilities of Tomcat and our study participants. We found that Tomcat with Fs-CoT, particularly with GPT-4o and DeepSeek-R1, achieves performance comparable to the human participants, underscoring its ToM potential for human-agent collaboration.

**arXiv ID:** 2507.02935
</details>

<details>
<summary><strong>Enabling Agents to Communicate Entirely in Latent Space</strong> - Zhuoyun Du, Runze Wang, Huiyu Bai, Zouying Cao, Xiaoyong Zhu, Yu Cheng, Bo Zheng, Wei Chen, Haochao Ying - [[pdf]](https://arxiv.org/pdf/2511.09149)</summary>

**Abstract:** While natural language is the de facto communication medium for LLM-based agents, it presents a fundamental constraint. The process of downsampling rich, internal latent states into discrete tokens inherently limits the depth and nuance of information that can be transmitted, thereby hindering collaborative problem-solving. Inspired by telepathy, which bypasses symbolic language in communication, we propose Interlat (Inter-agent Latent Space Communication), a paradigm that leverages the continuous last hidden states of an LLM as a representation of its thought for direct communication (termed latent communication). An additional learned compression process further compresses latent communication via latent space reasoning. Experiments demonstrate that Interlat outperforms both fine-tuned chain-of-thought (CoT) prompting and single-agent baselines, even across heterogeneous models, promoting more exploratory behavior and enabling genuine utilization of latent information. Further compression not only substantially accelerates inference by up to 24 times but also maintains competitive performance through an efficient information-preserving mechanism. We position this work as a feasibility study of entirely latent space inter-agent communication, and our results highlight its potential, offering valuable insights for future research. Our code is available at this https URL.

**arXiv ID:** 2511.09149
</details>

<details>
<summary><strong>CooperDrive: Enhancing Driving Decisions Through Cooperative Perception</strong> - Deyuan Qu, Qi Chen, Takayuki Shimizu, Onur Altintas - [[pdf]](https://arxiv.org/pdf/2604.14454)</summary>

**Abstract:** Autonomous vehicles equipped with robust onboard perception, localization, and planning still face limitations in occlusion and non-line-of-sight (NLOS) scenarios, where delayed reactions can increase collision risk. We propose CooperDrive, a cooperative perception framework that augments situational awareness and enables earlier, safer driving decisions. CooperDrive offers two key advantages: (i) each vehicle retains its native perception, localization, and planning stack, and (ii) a lightweight object-level sharing and fusion strategy bridges perception and planning. Specifically, CooperDrive reuses detector Bird's-Eye View (BEV) features to estimate accurate vehicle poses without additional heavy encoders, thereby reconstructing BEV representations and feeding the planner with low latency. On the planning side, CooperDrive leverages the expanded object set to anticipate potential conflicts earlier and adjust speed and trajectory proactively, thereby transforming reactive behaviors into predictive and safer driving decisions. Real-world closed-loop tests at occlusion-heavy NLOS intersections demonstrate that CooperDrive increases reaction lead time, minimum time-to-collision (TTC), and stopping margin, while requiring only 90 kbps bandwidth and maintaining an average end-to-end latency of 89 ms.

**arXiv ID:** 2604.14454
</details>

<details>
<summary><strong>Towards Deploying VLA without Fine-Tuning: Plug-and-Play Inference-Time VLA Policy Steering via Embodied Evolutionary Diffusion</strong> - Zhuo Li, Junjia Liu, Zhipeng Dong, Tao Teng, Quentin Rouxel, Darwin Caldwell, Fei Chen - [[pdf]](https://arxiv.org/pdf/2511.14178)</summary>

**Abstract:** Vision-Language-Action (VLA) models have demonstrated significant potential in real-world robotic manipulation. However, pre-trained VLA policies still suffer from substantial performance degradation during downstream deployment. Although fine-tuning can mitigate this issue, its reliance on costly demonstration collection and intensive computation makes it impractical in real-world settings. In this work, we introduce VLA-Pilot, a plug-and-play inference-time policy steering method for zero-shot deployment of pre-trained VLA without any additional fine-tuning or data collection. We evaluate VLA-Pilot on six real-world downstream manipulation tasks across two distinct robotic embodiments, encompassing both in-distribution and out-of-distribution scenarios. Experimental results demonstrate that VLA-Pilot substantially boosts the success rates of off-the-shelf pre-trained VLA policies, enabling robust zero-shot generalization to diverse tasks and embodiments. Experimental videos and code are available at: this https URL.

**arXiv ID:** 2511.14178
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (24 papers)</h2></summary>

<details>
<summary><strong>Demonstration of Pneuma-Seeker: Agentic System for Reifying and Fulfilling Information Needs on Tabular Data</strong> - Muhammad Imam Luthfi Balaka, Raul Castro Fernandez - [[pdf]](https://arxiv.org/pdf/2604.14422)</summary>

**Abstract:** Data analysts working with relational data often start with vague or underspecified questions and refine them iteratively as they explore the data. To support this iterative process, we demonstrate Pneuma-Seeker, a system that reifies a user's information need as explicit, inspectable relational specifications, enabling iterative refinement of the information need, targeted data discovery, and provenance-aware execution. Through two real-world procurement use cases, we show how Pneuma-Seeker leverages LLMs as transparent, interactive analytical collaborators rather than opaque answer engines.

**arXiv ID:** 2604.14422
</details>

<details>
<summary><strong>Evo-MedAgent: Beyond One-Shot Diagnosis with Agents That Remember, Reflect, and Improve</strong> - Weixiang Shen, Bailiang Jian, Jun Li, Che Liu, Johannes Moll, Xiaobin Hu, Daniel Rueckert, Hongwei Bran Li, Jiazhen Pan - [[pdf]](https://arxiv.org/pdf/2604.14475)</summary>

**Abstract:** Tool-augmented large language model (LLM) agents can orchestrate specialist classifiers, segmentation models, and visual question-answering modules to interpret chest X-rays. However, these agents still solve each case in isolation: they fail to accumulate experience across cases, correct recurrent reasoning mistakes, or adapt their tool-use behavior without expensive reinforcement learning. While a radiologist naturally improves with every case, current agents remain static. In this work, we propose Evo-MedAgent, a self-evolving memory module that equips a medical agent with the capacity for inter-case learning at test time. Our memory comprises three complementary stores: (1)~\emph{Retrospective Clinical Episodes} that retrieve problem-solving experiences from similar past cases, (2)~an \emph{Adaptive Procedural Heuristics} bank curating priority-tagged diagnostic rules that evolves via reflection, much like a physician refining their internal criteria, and (3)~a \emph{Tool Reliability Controller} that tracks per-tool trustworthiness. On ChestAgentBench, Evo-MedAgent raises multiple-choice question (MCQ) accuracy from 0.68 to 0.79 on GPT-5-mini, and from 0.76 to 0.87 on Gemini-3 Flash. With a strong base model, evolving memory improves performance more effectively than orchestrating external tools on qualitative diagnostic tasks. Because Evo-MedAgent requires no training, its per-case overhead is bounded by one additional retrieval pass and a single reflection call, making it deployable on top of any frozen model.

**arXiv ID:** 2604.14475
</details>

<details>
<summary><strong>Targeted Exploration via Unified Entropy Control for Reinforcement Learning</strong> - Chen Wang, Lai Wei, Yanzhi Zhang, Chenyang Shao, Zedong Dan, Weiran Huang, Ge Lan, Yue Wang - [[pdf]](https://arxiv.org/pdf/2604.14646)</summary>

**Abstract:** Recent advances in reinforcement learning (RL) have improved the reasoning capabilities of large language models (LLMs) and vision-language models (VLMs). However, the widely used Group Relative Policy Optimization (GRPO) consistently suffers from entropy collapse, causing the policy to converge prematurely and lose diversity. Existing exploration methods introduce additional bias or variance during exploration, making it difficult to maintain optimization stability. We propose Unified Entropy Control for Reinforcement Learning (UEC-RL), a framework that provides targeted mechanisms for exploration and stabilization. UEC-RL activates more exploration on difficult prompts to search for potential and valuable reasoning trajectories. In parallel, a stabilizer prevents entropy from growing uncontrollably, thereby keeping training stable as the model consolidates reliable behaviors. Together, these components expand the search space when needed while maintaining robust optimization throughout training. Experiments on both LLM and VLM reasoning tasks show consistent gains over RL baselines on both Pass@1 and Pass@$k$. On Geometry3K, UEC-RL achieves a 37.9\% relative improvement over GRPO, indicating that it sustains effective exploration without compromising convergence and underscoring UEC-RL as a key for scaling RL-based reasoning in large models. Our code is available at this https URL.

**arXiv ID:** 2604.14646
</details>

<details>
<summary><strong>AgentGA: Evolving Code Solutions in Agent-Seed Space</strong> - David Y.Y. Tan, Kellie Chin, Jingxian Zhang - [[pdf]](https://arxiv.org/pdf/2604.14655)</summary>

**Abstract:** We present AgentGA, a framework that evolves autonomous code-generation runs by optimizing the agent seed: the task prompt plus optional parent archives that initialize a fresh workspace. The outer loop searches over these reusable starting conditions rather than editing code directly. Each generation launches a fresh autonomous run from a reset workspace, while selected parent archives provide inherited artifacts that descendants can inspect and reuse. AgentGA couples a population-level genetic algorithm with long-horizon agents; selection uses deterministic 1:1 elite tournaments and operator allocation is adapted online with a modified Hedge controller. We instantiate the approach for tabular AutoML on the 16-competition Weco-Kaggle Lite benchmark. On the 10 benchmark runs reported here, AgentGA averages 74.52% Exceeds % of Human versus 54.15% for AIDE. Across 1135 parent-child comparisons, descendants given parent archives outperform runs started from scratch, indicating that inherited artifacts improve later autonomous runs. These findings support agent-seed optimization as a practical design point for autonomous code-search systems.

**arXiv ID:** 2604.14655
</details>

<details>
<summary><strong>Blue Data Intelligence Layer: Streaming Data and Agents for Multi-source Multi-modal Data-Centric Applications</strong> - Moin Aminnaseri, Farima Fatahi Bayat, Nikita Bhutani, Jean-Flavien Bussotti, Kevin Chan, Rafael Li Chen, Yanlin Feng, Jackson Hassell, Estevam Hruschka, Eser Kandogan, Hannah Kim, James Levine, Seiji Maekawa, Jalal Mahmud, Kushan Mitra, Naoki Otani, Pouya Pezeshkpour, Nima Shahbazi, Chen Shen, Dan Zhang - [[pdf]](https://arxiv.org/pdf/2604.15233)</summary>

**Abstract:** NL2SQL systems aim to address the growing need for natural language interaction with data. However, real-world information rarely maps to a single SQL query because (1) users express queries iteratively (2) questions often span multiple data sources beyond the closed-world assumption of a single database, and (3) queries frequently rely on commonsense or external knowledge. Consequently, satisfying realistic data needs require integrating heterogeneous sources, modalities, and contextual data. In this paper, we present Blue's Data Intelligence Layer (DIL) designed to support multi-source, multi-modal, and data-centric applications. Blue is a compound AI system that orchestrates agents and data for enterprise settings. DIL serves as the data intelligence layer for agentic data processing, to bridge the semantic gap between user intent and available information by unifying structured enterprise data, world knowledge accessible through LLMs, and personal context obtained through interaction. At the core of DIL is a data registry that stores metadata for diverse data sources and modalities to enable both native and natural language queries. DIL treats LLMs, the Web, and the User as source 'databases', each with their own query interface, elevating them to first-class data sources. DIL relies on data planners to transform user queries into executable query plans. These plans are declarative abstractions that unify relational operators with other operators spanning multiple modalities. DIL planners support decomposition of complex requests into subqueries, retrieval from diverse sources, and finally reasoning and integration to produce final results. We demonstrate DIL through two interactive scenarios in which user queries dynamically trigger multi-source retrieval, cross-modal reasoning, and result synthesis, illustrating how compound AI systems can move beyond single database NL2SQL.

**arXiv ID:** 2604.15233
</details>

<details>
<summary><strong>Reinforcement Learning via Value Gradient Flow</strong> - Haoran Xu, Kaiwen Hu, Somayeh Sojoudi, Amy Zhang - [[pdf]](https://arxiv.org/pdf/2604.14265)</summary>

**Abstract:** We study behavior-regularized reinforcement learning (RL), where regularization toward a reference distribution (the dataset in offline RL or the base model in LLM RL finetuning) is essential to prevent value over-optimization caused by erroneous out-of-distribution extrapolation. Existing methods either rely on reparameterized policy gradient, which are difficult to scale to large generative models, or on reject sampling, which can be overly conservative when attempting to move beyond the behavior support. In this paper, we propose Value Gradient Flow (VGF), a scalable new paradigm for behavior-regularized RL. VGF casts behavior-regularized RL as an optimal transport problem that maps the reference distribution to the value-induced optimal policy distribution. We solve this transport problem via discrete gradient flow, where value gradients guide particles initialized from the reference distribution. Our analysis shows that VGF imposes regularization implicitly by controlling the transport budget. VGF eliminates explicit policy parameterization while remaining expressive and flexible, this enables adaptive test-time scaling by adjusting the transport budget. Extensive experiments demonstrate that VGF significantly outperforms prior methods, achieving state-of-the-art results on offline RL benchmarks (D4RL, OGBench) and LLM RL tasks. Code and runs can be found at this https URL.

**arXiv ID:** 2604.14265
</details>

<details>
<summary><strong>Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization</strong> - Junzhe Wang, Zhiheng Xi, yajie yang, Hao Luo, Shihan Dou, Tao Gui, Qi Zhang - [[pdf]](https://arxiv.org/pdf/2604.14267)</summary>

**Abstract:** Search agents extend Large Language Models (LLMs) beyond static parametric knowledge by enabling access to up-to-date and long-tail information unavailable during pretraining. While reinforcement learning has been widely adopted for training such agents, existing approaches face key limitations: process supervision often suffers from unstable value estimation, whereas outcome supervision struggles with credit assignment due to sparse, trajectory-level rewards. To bridge this gap, we propose Contribution-Weighted GRPO (CW-GRPO), a framework that integrates process supervision into group relative policy optimization. Instead of directly optimizing process rewards, CW-GRPO employs an LLM judge to assess the retrieval utility and reasoning correctness at each search round, producing per-round contribution scores. These scores are used to rescale outcome-based advantages along the trajectory, enabling fine-grained credit assignment without sacrificing optimization stability. Experiments on multiple knowledge-intensive benchmarks show that CW-GRPO outperforms standard GRPO by 5.0\% on Qwen3-8B and 6.3\% on Qwen3-1.7B, leading to more effective search behaviors. Additional analysis reveals that successful trajectories exhibit concentrated contributions across rounds, providing empirical insight into search agent tasks.

**arXiv ID:** 2604.14267
</details>

<details>
<summary><strong>AgileLog: A Forkable Shared Log for Agents on Data Streams</strong> - Shreesha G. Bhat, Tony Hong, Michael Noguera, Ramnatthan Alagappan, Aishwarya Ganesan - [[pdf]](https://arxiv.org/pdf/2604.14590)</summary>

**Abstract:** In modern data-streaming systems, alongside traditional programs, a new type of entity has emerged that can interact with streaming data: AI agents. Unlike traditional programs, AI agents use LLM reasoning to accomplish high-level tasks specified in natural language over streaming data. Unfortunately, current streaming systems cannot fully support agents: they lack the fundamental mechanisms to avoid the performance interference caused by agentic tasks and to safely handle agentic writes. We argue that the shared log, the core abstraction underlying streaming data, must support creating forks of itself, and that such a forkable shared log serves as a great substrate for agents acting on streaming data. We propose AgileLog, a new shared log abstraction that provides novel forking primitives for agentic use cases. We design Bolt, an implementation of the AgileLog abstraction, that uses novel techniques to make forks cheap, and provide logical and performance isolation.

**arXiv ID:** 2604.14590
</details>

<details>
<summary><strong>AIPC: Agent-Based Automation for AI Model Deployment with Qualcomm AI Runtime</strong> - Jianhao Su, Zhanwei Wu, ShengTing Huang, Weidong Feng - [[pdf]](https://arxiv.org/pdf/2604.14661)</summary>

**Abstract:** Edge AI model deployment is a multi-stage engineering process involving model conversion, operator compatibility handling, quantization calibration, runtime integration, and accuracy validation. In
practice, this workflow is long, failure-prone, and heavily dependent on deployment expertise, particularly when targeting hardware-specific inference runtimes. This technical report presents AIPC (AI
Porting Conversion), an AI agent-driven approach for constrained automation of AI model deployment. AIPC decomposes deployment into standardized, verifiable stages and injects deployment-domain knowledge
into agent execution through Agent Skills, helper scripts, and a stage-wise validation loop. This design reduces both the expertise barrier and the engineering time required for hardware deployment.
Using Qualcomm AI Runtime (QAIRT) as the primary scenario, this report examines automated deployment across representative vision, multimodal, and speech models. In the cases covered here, AIPC can
complete deployment from PyTorch to runnable QNN/SNPE inference within 7-20 minutes for structurally regular vision models, with indicative API costs roughly in the range of USD 0.7-10. For more complex
models involving less-supported operators, dynamic shapes, or autoregressive decoding structures, fully automated deployment may still require further advances, but AIPC already provides practical support
for execution, failure localization, and bounded repair.

**arXiv ID:** 2604.14661
</details>

<details>
<summary><strong>KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality</strong> - Baochang Ren, Shuofei Qiao, Da Zheng, Huajun Chen, Ningyu Zhang - [[pdf]](https://arxiv.org/pdf/2506.19807)</summary>

**Abstract:** Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at this https URL.

**arXiv ID:** 2506.19807
</details>

<details>
<summary><strong>LongAct: Harnessing Intrinsic Activation Patterns for Long-Context Reinforcement Learning</strong> - Bowen Ping, Zijun Chen, Tingfeng Hui, Qize Yu, Chenxuan Li, Junchi Yan, Baobao Chang - [[pdf]](https://arxiv.org/pdf/2604.14922)</summary>

**Abstract:** Reinforcement Learning (RL) has emerged as a critical driver for enhancing the reasoning capabilities of Large Language Models (LLMs). While recent advancements have focused on reward engineering or data synthesis, few studies exploit the model's intrinsic representation characteristics to guide the training process. In this paper, we first observe the presence of high-magnitude activations within the query and key vectors when processing long contexts. Drawing inspiration from model quantization -- which establishes the criticality of such high-magnitude activations -- and the insight that long-context reasoning inherently exhibits a sparse structure, we hypothesize that these weights serve as the pivotal drivers for effective model optimization. Based on this insight, we propose LongAct, a strategy that shifts from uniform to saliency-guided sparse updates. By selectively updating only the weights associated with these significant activations, LongAct achieves an approximate 8% improvement on LongBench v2 and enhances generalization on the RULER benchmark. Furthermore, our method exhibits remarkable universality, consistently boosting performance across diverse RL algorithms such as GRPO and DAPO. Extensive ablation studies suggest that focusing on these salient features is key to unlocking long-context potential.

**arXiv ID:** 2604.14922
</details>

<details>
<summary><strong>One RL to See Them All: Visual Triple Unified Reinforcement Learning</strong> - Yan Ma, Linge Du, Xuyang Shen, Shaoxiang Chen, Pengfei Li, Qibing Ren, Lizhuang Ma, Yuchao Dai, Pengfei Liu, Junjie Yan - [[pdf]](https://arxiv.org/pdf/2505.18129)</summary>

**Abstract:** Reinforcement learning (RL) is becoming an important direction for post-training vision-language models (VLMs), but public training methodologies for unified multimodal RL remain much less mature, especially for heterogeneous reasoning and perception-heavy tasks. We propose V-Triune, a Visual Triple Unified Reinforcement Learning methodology for unified multimodal RL. It organizes training around three coordinated abstractions: Sample-Level Reward Routing, Verifier-Level Outcome Verification, and Source-Level Diagnostics. Within this methodology, Dynamic IoU provides localization-specific reward shaping that avoids reward ambiguity under loose thresholds and reward sparsity under strict ones. Built on V-Triune, we develop Orsta (7B, 32B), a family of models jointly trained on eight reasoning and perception tasks. Under matched budgets, unified training matches or outperforms specialist mixtures. The final Orsta models improve over their backbones on MEGA-Bench, compare favorably with strong multi-task RL-VLM baselines, and transfer these gains to a broad set of downstream benchmarks. These results show that unified RL can improve both reasoning and perception within a single VLM RL this http URL V-Triune system, along with the Orsta models, is publicly available at this https URL.

**arXiv ID:** 2505.18129
</details>

<details>
<summary><strong>ProRank: Prompt Warmup via Reinforcement Learning for Small Language Models Reranking</strong> - Xianming Li, Aamir Shakir, Rui Huang, Tsz-fung Andrew Lee, Julius Lipp, Benjamin Clavié, Jing Li - [[pdf]](https://arxiv.org/pdf/2506.03487)</summary>

**Abstract:** Reranking is fundamental to information retrieval and retrieval-augmented generation, with recent Large Language Models (LLMs) significantly advancing reranking quality. Most current works rely on large-scale LLMs (>7B parameters), presenting high computational costs. Small Language Models (SLMs) offer a promising alternative because of computational efficiency. However, our preliminary quantitative analysis reveals key limitations of SLMs: their representation space is narrow, leading to reduced expressiveness, and they struggle with understanding task prompts without fine-tuning. To address these issues, we introduce a novel two-stage training approach, ProRank, for SLM-based document reranking. We propose using reinforcement learning to improve the understanding of task prompts. Additionally, we introduce fine-grained score learning to enhance representation expressiveness and further improve document reranking quality. Extensive experiments suggest that ProRank consistently outperforms both the most advanced open-source and proprietary reranking models. Notably, our 0.5B ProRank even surpasses powerful LLM reranking models on the BEIR benchmark, establishing that properly trained SLMs can achieve superior document reranking performance while maintaining computational efficiency.

**arXiv ID:** 2506.03487
</details>

<details>
<summary><strong>Wasserstein Formulation of Reinforcement Learning. An Optimal Transport Perspective on Policy Optimization</strong> - Mathias Dus - [[pdf]](https://arxiv.org/pdf/2604.14765)</summary>

**Abstract:** We present a geometric framework for Reinforcement Learning (RL) that views policies as maps into the Wasserstein space of action probabilities. First, we define a Riemannian structure induced by stationary distributions, proving its existence in a general context. We then define the tangent space of policies and characterize the geodesics, specifically addressing the measurability of vector fields mapped from the state space to the tangent space of probability measures over the action space. Next, we formulate a general RL optimization problem and construct a gradient flow using Otto's calculus. We compute the gradient and the Hessian of the energy, providing a formal second-order analysis. Finally, we illustrate the method with numerical examples for low-dimensional problems, computing the gradient directly from our theoretical formalism. For high-dimensional problems, we parameterize the policy using a neural network and optimize it based on an ergodic approximation of the cost.

**arXiv ID:** 2604.14765
</details>

<details>
<summary><strong>RL-STPA: Adapting System-Theoretic Hazard Analysis for Safety-Critical Reinforcement Learning</strong> - Steven A. Senczyszyn, Timothy C. Havens, Nathaniel Rice, Jason E. Summers, Benjamin D. Werner, Benjamin J. Schumeg - [[pdf]](https://arxiv.org/pdf/2604.15201)</summary>

**Abstract:** As reinforcement learning (RL) deployments expand into safety-critical domains, existing evaluation methods fail to systematically identify hazards arising from the black-box nature of neural network enabled policies and distributional shift between training and deployment. This paper introduces Reinforcement Learning System-Theoretic Process Analysis (RL-STPA), a framework that adapts conventional STPA's systematic hazard analysis to address RL's unique challenges through three key contributions: hierarchical subtask decomposition using both temporal phase analysis and domain expertise to capture emergent behaviors, coverage-guided perturbation testing that explores the sensitivity of state-action spaces, and iterative checkpoints that feed identified hazards back into training through reward shaping and curriculum design. We demonstrate RL-STPA in the safety-critical test case of autonomous drone navigation and landing, revealing potential loss scenarios that can be missed by standard RL evaluations. The proposed framework provides practitioners with a toolkit for systematic hazard analysis, quantitative metrics for safety coverage assessment, and actionable guidelines for establishing operational safety bounds. While RL-STPA cannot provide formal guarantees for arbitrary neural policies, it offers a practical methodology for systematically evaluating and improving RL safety and robustness in safety-critical applications where exhaustive verification methods remain intractable.

**arXiv ID:** 2604.15201
</details>

<details>
<summary><strong>Timescale Separation Enables Deep Reinforcement Learning Control of Rotating Detonation Engine Mode Transitions</strong> - Kristian Holme, Jean Rabault, Ricardo Vinuesa, Mikael Mortensen - [[pdf]](https://arxiv.org/pdf/2604.14398)</summary>

**Abstract:** Rotating detonation engines (RDEs) are a promising propulsion concept that may offer higher thermodynamic efficiency and specific impulse than conventional systems, but nonlinear phenomena, including transitions to oscillatory or chaotic propagation modes, can hinder practical operation. Deep Reinforcement Learning (DRL) has emerged as a promising method for controlling complex nonlinear dynamics such as those observed in RDEs. However, the multi-timescale nature of the RDE system makes direct application of DRL challenging. We address this challenge by reformulating the DRL problem in a moving reference frame that follows the detonation-wave pattern, making the wave structure appear quasi-steady to the agent. This reformulation enables scale separation between fast detonation propagation and slower operating-mode dynamics. We train DRL controllers to modulate spatially segmented injection pressure in a one-dimensional reduced-order RDE model and induce rapid transitions between different mode-locked states. Across a range of actuation periods, initial states, and target modes, controllers trained in the moving frame learn more reliably than those trained in a stationary frame and remain effective over a broader range of actuation periods. These results suggest that symmetry-aware moving reference frame formulations may be useful for related multiscale flow-control problems and that scale separation should be exploited whenever possible to enable DRL control of multi-timescale systems.

**arXiv ID:** 2604.14398
</details>

<details>
<summary><strong>Safe Reinforcement Learning using Action Projection: Safeguard the Policy or the Environment?</strong> - Hannah Markgraf, Shambhuraj Sawant, Hanna Krasowski, Lukas Schäfer, Sebastien Gros, Matthias Althoff - [[pdf]](https://arxiv.org/pdf/2509.12833)</summary>

**Abstract:** Projection-based safety filters, which modify unsafe actions by mapping them to the closest safe alternative, are widely used to enforce safety constraints in reinforcement learning (RL). Two integration strategies are commonly considered: Safe environment RL (SE-RL), where the safeguard is treated as part of the environment, and safe policy RL (SP-RL), where it is embedded within the policy through differentiable optimization layers. Despite their practical relevance in safety-critical settings, a formal understanding of their differences is lacking. In this work, we present a theoretical comparison of SE-RL and SP-RL. We identify a key distinction in how each approach is affected by action aliasing, a phenomenon in which multiple unsafe actions are projected to the same safe action, causing information loss in the policy gradients. In SE-RL, this effect is implicitly approximated by the critic, while in SP-RL, it manifests directly as rank-deficient Jacobians during backpropagation through the safeguard. Our contributions are threefold: (i) a unified formalization of SE-RL and SP-RL in the context of actor-critic algorithms, (ii) a theoretical analysis of their respective policy gradient estimates, highlighting the role of action aliasing, and (iii) a comparative study of mitigation strategies, including a novel penalty-based improvement for SP-RL that aligns with established SE-RL practices. Empirical results support our theoretical predictions, showing that action aliasing is more detrimental for SP-RL than for SE-RL. However, with appropriate improvement strategies, SP-RL can match or outperform improved SE-RL across a range of environments. These findings provide actionable insights for choosing and refining projection-based safe RL methods based on task characteristics.

**arXiv ID:** 2509.12833
</details>

<details>
<summary><strong>Model-Based Reinforcement Learning under Random Observation Delays</strong> - Armin Karamzade, Kyungmin Kim, JB Lanier, Davide Corsi, Roy Fox - [[pdf]](https://arxiv.org/pdf/2509.20869)</summary>

**Abstract:** Delays frequently occur in real-world environments, yet standard reinforcement learning (RL) algorithms often assume instantaneous perception of the environment. We study random sensor delays in POMDPs, where observations may arrive out-of-sequence, a setting that has not been previously addressed in RL. We analyze the structure of such delays and demonstrate that naive approaches, such as stacking past observations, are insufficient for reliable performance. To address this, we propose a model-based filtering process that sequentially updates the belief state based on an incoming stream of observations. We then introduce a simple delay-aware framework that incorporates this idea into model-based RL, enabling agents to effectively handle random delays. Applying this framework to the Dreamer world-modeling scheme, our method consistently outperforms delay-aware baselines developed for MDPs and demonstrates robustness to delay distribution shifts during deployment. Additionally, we present experiments on simulated robotic tasks, comparing our method to common practical heuristics and emphasizing the importance of explicitly modeling observation delays.

**arXiv ID:** 2509.20869
</details>

<details>
<summary><strong>Continuous-time reinforcement learning: ellipticity enables model-free value function approximation</strong> - Wenlong Mou - [[pdf]](https://arxiv.org/pdf/2602.06930)</summary>

**Abstract:** We study off-policy reinforcement learning for controlling continuous-time Markov diffusion processes with discrete-time observations and actions. We consider model-free algorithms with function approximation that learn value and advantage functions directly from data, without unrealistic structural assumptions on the dynamics.
Leveraging the ellipticity of the diffusions, we establish a new class of Hilbert-space positive definiteness and boundedness properties for the Bellman operators. Based on these properties, we propose the Sobolev-prox fitted $q$-learning algorithm, which learns value and advantage functions by iteratively solving least-squares regression problems. We derive oracle inequalities for the estimation error, governed by (i) the best approximation error of the function classes, (ii) their localized complexity, (iii) exponentially decaying optimization error, and (iv) numerical discretization error. These results identify ellipticity as a key structural property that renders reinforcement learning with function approximation for Markov diffusions no harder than supervised learning.

**arXiv ID:** 2602.06930
</details>

<details>
<summary><strong>Drowsiness-Aware Adaptive Autonomous Braking System based on Deep Reinforcement Learning for Enhanced Road Safety</strong> - Hossem Eddine Hafidi, Elisabetta De Giovanni, Teodoro Montanaro, Ilaria Sergi, Massimo De Vittorio, Luigi Patrono - [[pdf]](https://arxiv.org/pdf/2604.13878)</summary>

**Abstract:** Driver drowsiness significantly impairs the ability to accurately judge safe braking distances and is estimated to contribute to 10%-20% of road accidents in Europe. Traditional driver-assistance systems lack adaptability to real-time physiological states such as drowsiness. This paper proposes a deep reinforcement learning-based autonomous braking system that integrates vehicle dynamics with driver physiological data. Drowsiness is detected from ECG signals using a Recurrent Neural Network (RNN), selected through an extensive benchmark analysis of 2-minute windows with varying segmentation and overlap configurations. The inferred drowsiness state is incorporated into the observable state space of a Double-Dueling Deep Q-Network (DQN) agent, where driver impairment is modeled as an action delay. The system is implemented and evaluated in a high-fidelity CARLA simulation environment. Experimental results show that the proposed agent achieves a 99.99% success rate in avoiding collisions under both drowsy and non-drowsy conditions. These findings demonstrate the effectiveness of physiology-aware control strategies for enhancing adaptive and intelligent driving safety systems.

**arXiv ID:** 2604.13878
</details>

<details>
<summary><strong>Model-Based Reinforcement Learning Exploits Passive Body Dynamics for High-Performance Biped Robot Locomotion</strong> - Tomoya Kamimura, Haruka Washiyama, Akihito Sano - [[pdf]](https://arxiv.org/pdf/2604.14565)</summary>

**Abstract:** Embodiment is a significant keyword in recent machine learning fields. This study focused on the passive nature of the body of a biped robot to generate walking and running locomotion using model-based deep reinforcement learning. We constructed two models in a simulator, one with passive elements (e.g., springs) and the other, which is similar to general humanoids, without passive elements. The training of the model with passive elements was highly affected by the attractor of the system. This lead that although the trajectories quickly converged to limit cycles, it took a long time to obtain large rewards. However, thanks to the attractor-driven learning, the acquired locomotion was robust and energy-efficient. The results revealed that robots with passive elements could efficiently acquire high-performance locomotion by utilizing stable limit cycles generated through dynamic interaction between the body and ground. This study demonstrates the importance of implementing passive properties in the body for future embodied AI.

**arXiv ID:** 2604.14565
</details>

<details>
<summary><strong>TwinOR: Photorealistic Digital Twins of Dynamic Operating Rooms for Embodied AI Research</strong> - Han Zhang, Yiqing Shen, Roger D. Soberanis-Mukul, Ankita Ghosh, Hao Ding, Lalithkumar Seenivasan, Jose L. Porras, Zhekai Mao, Chenjia Li, Wenjie Xiao, Lonny Yarmus, Angela Christine Argento, Masaru Ishii, Mathias Unberath - [[pdf]](https://arxiv.org/pdf/2511.07412)</summary>

**Abstract:** Developing embodied AI for intelligent surgical systems requires safe, controllable environments for continual learning and evaluation. However, safety regulations and operational constraints in operating rooms (ORs) limit agents from freely perceiving and interacting in realistic settings. Digital twins provide high-fidelity, risk-free environments for exploration and training. How we may create dynamic digital representations of ORs that capture relevant spatial, visual, and behavioral complexity remains an open challenge. We introduce TwinOR, a real-to-sim infrastructure for constructing photorealistic and dynamic digital twins of ORs. The system reconstructs static geometry and continuously models human and equipment motion. The static and dynamic components are fused into an immersive 3D environment that supports controllable simulation and facilitates future embodied exploration. The proposed framework reconstructs complete OR geometry with centimeter-level accuracy while preserving dynamic interaction across surgical workflows. In our experiments, TwinOR synthesizes stereo and monocular RGB streams as well as depth observations for geometry understanding and visual localization tasks. Models such as FoundationStereo and ORB-SLAM3 evaluated on TwinOR-synthesized data achieve performance within their reported accuracy ranges on real-world indoor datasets, demonstrating that TwinOR provides sensor-level realism sufficient for emulating real-world perception and localization challenge. By establishing a perception-grounded real-to-sim pipeline, TwinOR enables the automatic construction of dynamic, photorealistic digital twins of ORs. As a safe and scalable environment for experimentation, TwinOR opens new opportunities for translating embodied intelligence from simulation to real-world clinical environments.

**arXiv ID:** 2511.07412
</details>

<details>
<summary><strong>Beyond Chat and Clicks: GUI Agents for In-Situ Assistance via Live Interface Transformation</strong> - Pan Hao, Rishi Selvakumaran, Jacob Sun, Qianwen Wang - [[pdf]](https://arxiv.org/pdf/2604.14668)</summary>

**Abstract:** Complex visual interfaces are powerful yet have a steep learning curve, as users must navigate feature-rich visual interfaces while reasoning about domain-specific operations. Existing approaches either deliver assistance through a separate chat-based interaction, or require substantial application-specific engineering to build support natively into each interface. To address the gaps, we propose in-situ assistance: a mode of support delivered directly within any live web interface through lightweight, browser-level interventions on the Document Object Model (DOM), without rebuilding the application or modifying its underlying logic. We contribute a design space and a computational pipeline for DOM-mediated in-situ assistance, characterizing how GUI agents can insert, mutate, or recompose web elements to make the interface easier for users to understand, use, and navigate. We instantiate in-situ assistance in DOMSteer, a Chrome extension that interprets a user's help request and live interface context, grounds it to relevant UI elements, and executes reversible DOM manipulations directly on the live page to deliver assistance, including contextual tooltips, control highlighting, layout reorganization. Quantitative evaluations on two complex visual interfaces show that DOMSteer delivers reliable and efficient in-situ assistance. Use cases and a comparative user study with baseline ChatGPTAtlas demonstrate the usability and effectiveness of DOMSteer. Altogether, these findings point to a broader role for GUI agents: not just assisting from the sidelines, but actively reconfiguring live interfaces to support users in the moment.

**arXiv ID:** 2604.14668
</details>

<details>
<summary><strong>A longitudinal health agent framework</strong> - Georgianna "Blue" Lin, Rencong Jiang, Noémie Elhadad, Xuhai "Orson" Xu - [[pdf]](https://arxiv.org/pdf/2604.12019)</summary>

**Abstract:** Although artificial intelligence (AI) agents are increasingly proposed to support potentially longitudinal health tasks, such as symptom management, behavior change, and patient support, most current implementations fall short of facilitating user intent and fostering accountability. This contrasts with prior work on supporting longitudinal needs, where follow-up, coherent reasoning, and sustained alignment with individuals' goals are critical for both effectiveness and safety. In this paper, we draw on established clinical and personal health informatics frameworks to define what it would mean to orchestrate longitudinal health interactions with AI agents. We propose a multi-layer framework and corresponding agent architecture that operationalizes adaptation, coherence, continuity, and agency across repeated interactions. Through representative use cases, we demonstrate how longitudinal agents can maintain meaningful engagement, adapt to evolving goals, and support safe, personalized decision-making over time. Our findings underscore both the promise and the complexity of designing systems capable of supporting health trajectories beyond isolated interactions, and we offer guidance for future research and development in multi-session, user-centered health AI.

**arXiv ID:** 2604.12019
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
