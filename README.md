# Agent arXiv Daily

**Last Updated:** 2025-12-24 02:59:33

**Total Papers:** 49

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
<summary><strong>Conservative Bias in Multi-Teacher Learning: Why Agents Prefer Low-Reward Advisors</strong> - Maher Mesto, Francisco Cruz - [[pdf]](https://arxiv.org/pdf/2512.17180)</summary>

**Abstract:** Interactive reinforcement learning (IRL) has shown promise in enabling autonomous agents and robots to learn complex behaviours from human teachers, yet the dynamics of teacher selection remain poorly understood. This paper reveals an unexpected phenomenon in IRL: when given a choice between teachers with different reward structures, learning agents overwhelmingly prefer conservative, low-reward teachers (93.16% selection rate) over those offering 20x higher rewards. Through 1,250 experimental runs in navigation tasks with multiple expert teachers, we discovered: (1) Conservative bias dominates teacher selection: agents systematically choose the lowest-reward teacher, prioritising consistency over optimality; (2) Critical performance thresholds exist at teacher availability rho >= 0.6 and accuracy omega >= 0.6, below which the framework fails catastrophically; (3) The framework achieves 159% improvement over baseline Q-learning under concept drift. These findings challenge fundamental assumptions about optimal teaching in RL and suggest potential implications for human-robot collaboration, where human preferences for safety and consistency may align with the observed agent selection behaviour, potentially informing training paradigms for safety-critical robotic applications.

**arXiv ID:** 2512.17180
</details>

<details>
<summary><strong>Memory-T1: Reinforcement Learning for Temporal Reasoning in Multi-session Agents</strong> - Yiming Du, Baojun Wang, Yifan Xiang, Zhaowei Wang, Wenyu Huang, Boyang Xue, Bin Liang, Xingshan Zeng, Fei Mi, Haoli Bai, Lifeng Shang, Jeff Z. Pan, Yuxin Jiang, Kam-Fai Wong - [[pdf]](https://arxiv.org/pdf/2512.20092)</summary>

**Abstract:** Temporal reasoning over long, multi-session dialogues is a critical capability for conversational agents. However, existing works and our pilot study have shown that as dialogue histories grow in length and accumulate noise, current long-context models struggle to accurately identify temporally pertinent information, significantly impairing reasoning performance. To address this, we introduce Memory-T1, a framework that learns a time-aware memory selection policy using reinforcement learning (RL). It employs a coarse-to-fine strategy, first pruning the dialogue history into a candidate set using temporal and relevance filters, followed by an RL agent that selects the precise evidence sessions. The RL training is guided by a multi-level reward function optimizing (i) answer accuracy, (ii) evidence grounding, and (iii) temporal consistency. In particular, the temporal consistency reward provides a dense signal by evaluating alignment with the query time scope at both the session-level (chronological proximity) and the utterance-level (chronological fidelity), enabling the agent to resolve subtle chronological ambiguities. On the Time-Dialog benchmark, Memory-T1 boosts a 7B model to an overall score of 67.0\%, establishing a new state-of-the-art performance for open-source models and outperforming a 14B baseline by 10.2\%. Ablation studies show temporal consistency and evidence grounding rewards jointly contribute to a 15.0\% performance gain. Moreover, Memory-T1 maintains robustness up to 128k tokens, where baseline models collapse, proving effectiveness against noise in extensive dialogue histories. The code and datasets are publicly available at this https URL

**arXiv ID:** 2512.20092
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>PhysMaster: Building an Autonomous AI Physicist for Theoretical and Computational Physics Research</strong> - Tingjia Miao, Jiawen Dai, Jingkun Liu, Jinxin Tan, Muhua Zhang, Wenkai Jin, Yuwen Du, Tian Jin, Xianghe Pang, Zexi Liu, Tu Guo, Zhengliang Zhang, Yunjie Huang, Shuo Chen, Rui Ye, Yuzhi Zhang, Linfeng Zhang, Kun Chen, Wei Wang, Weinan E, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2512.19799)</summary>

**Abstract:** Advances in LLMs have produced agents with knowledge and operational capabilities comparable to human scientists, suggesting potential to assist, accelerate, and automate research. However, existing studies mainly evaluate such systems on well-defined benchmarks or general tasks like literature retrieval, limiting their end-to-end problem-solving ability in open scientific scenarios. This is particularly true in physics, which is abstract, mathematically intensive, and requires integrating analytical reasoning with code-based computation. To address this, we propose PhysMaster, an LLM-based agent functioning as an autonomous theoretical and computational physicist. PhysMaster couples absract reasoning with numerical computation and leverages LANDAU, the Layered Academic Data Universe, which preserves retrieved literature, curated prior knowledge, and validated methodological traces, enhancing decision reliability and stability. It also employs an adaptive exploration strategy balancing efficiency and open-ended exploration, enabling robust performance in ultra-long-horizon tasks. We evaluate PhysMaster on problems from high-energy theory, condensed matter theory to astrophysics, including: (i) acceleration, compressing labor-intensive research from months to hours; (ii) automation, autonomously executing hypothesis-driven loops ; and (iii) autonomous discovery, independently exploring open problems.

**arXiv ID:** 2512.19799
</details>

<details>
<summary><strong>Bohrium + SciMaster: Building the Infrastructure and Ecosystem for Agentic Science at Scale</strong> - Linfeng Zhang, Siheng Chen, Yuzhu Cai, Jingyi Chai, Junhan Chang, Kun Chen, Zhi X. Chen, Zhaohan Ding, Yuwen Du, Yuanpeng Gao, Yuan Gao, Jing Gao, Zhifeng Gao, Qiangqiang Gu, Yanhui Hong, Yuan Huang, Xi Fang, Xiaohong Ji, Guolin Ke, Zixing Lei, Xinyu Li, Yongge Li, Ruoxue Liao, Hang Lin, Xiaolu Lin, Yuxiang Liu, Xinzijian Liu, Zexi Liu, Jintan Lu, Tingjia Miao, Haohui Que, Weijie Sun, Yanfeng Wang, Bingyang Wu, Tianju Xue, Rui Ye, Jinzhe Zeng, Duo Zhang, Jiahui Zhang, Linfeng Zhang, Tianhan Zhang, Wenchang Zhang, Yuzhi Zhang, Zezhong Zhang, Hang Zheng, Hui Zhou, Tong Zhu, Xinyu Zhu, Qingguo Zhou, Weinan E - [[pdf]](https://arxiv.org/pdf/2512.20469)</summary>

**Abstract:** AI agents are emerging as a practical way to run multi-step scientific workflows that interleave reasoning with tool use and verification, pointing to a shift from isolated AI-assisted steps toward \emph{agentic science at scale}. This shift is increasingly feasible, as scientific tools and models can be invoked through stable interfaces and verified with recorded execution traces, and increasingly necessary, as AI accelerates scientific output and stresses the peer-review and publication pipeline, raising the bar for traceability and credible evaluation.
However, scaling agentic science remains difficult: workflows are hard to observe and reproduce; many tools and laboratory systems are not agent-ready; execution is hard to trace and govern; and prototype AI Scientist systems are often bespoke, limiting reuse and systematic improvement from real workflow signals.
We argue that scaling agentic science requires an infrastructure-and-ecosystem approach, instantiated in Bohrium+SciMaster. Bohrium acts as a managed, traceable hub for AI4S assets -- akin to a HuggingFace of AI for Science -- that turns diverse scientific data, software, compute, and laboratory systems into agent-ready capabilities. SciMaster orchestrates these capabilities into long-horizon scientific workflows, on which scientific agents can be composed and executed. Between infrastructure and orchestration, a \emph{scientific intelligence substrate} organizes reusable models, knowledge, and components into executable building blocks for workflow reasoning and action, enabling composition, auditability, and improvement through use.
We demonstrate this stack with eleven representative master agents in real workflows, achieving orders-of-magnitude reductions in end-to-end scientific cycle time and generating execution-grounded signals from real workloads at multi-million scale.

**arXiv ID:** 2512.20469
</details>

<details>
<summary><strong>HARMON-E: Hierarchical Agentic Reasoning for Multimodal Oncology Notes to Extract Structured Data</strong> - Shashi Kant Gupta, Arijeet Pramanik, Jerrin John Thomas, Regina Schwind, Lauren Wiener, Avi Raju, Jeremy Kornbluth, Yanshan Wang, Zhaohui Su, Hrituraj Singh - [[pdf]](https://arxiv.org/pdf/2512.19864)</summary>

**Abstract:** Unstructured notes within the electronic health record (EHR) contain rich clinical information vital for cancer treatment decision making and research, yet reliably extracting structured oncology data remains challenging due to extensive variability, specialized terminology, and inconsistent document formats. Manual abstraction, although accurate, is prohibitively costly and unscalable. Existing automated approaches typically address narrow scenarios - either using synthetic datasets, restricting focus to document-level extraction, or isolating specific clinical variables (e.g., staging, biomarkers, histology) - and do not adequately handle patient-level synthesis across the large number of clinical documents containing contradictory information. In this study, we propose an agentic framework that systematically decomposes complex oncology data extraction into modular, adaptive tasks. Specifically, we use large language models (LLMs) as reasoning agents, equipped with context-sensitive retrieval and iterative synthesis capabilities, to exhaustively and comprehensively extract structured clinical variables from real-world oncology notes. Evaluated on a large-scale dataset of over 400,000 unstructured clinical notes and scanned PDF reports spanning 2,250 cancer patients, our method achieves an average F1-score of 0.93, with 100 out of 103 oncology-specific clinical variables exceeding 0.85, and critical variables (e.g., biomarkers and medications) surpassing 0.95. Moreover, integration of the agentic system into a data curation workflow resulted in 0.94 direct manual approval rate, significantly reducing annotation costs. To our knowledge, this constitutes the first exhaustive, end-to-end application of LLM-based agents for structured oncology data extraction at scale

**arXiv ID:** 2512.19864
</details>

<details>
<summary><strong>KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System</strong> - Zhongyu Xia, Wenhao Chen, Yongtao Wang, Ming-Hsuan Yang - [[pdf]](https://arxiv.org/pdf/2512.20299)</summary>

**Abstract:** Visual-language reasoning, driving knowledge, and value alignment are essential for advanced autonomous driving systems. However, existing approaches largely rely on data-driven learning, making it difficult to capture the complex logic underlying decision-making through imitation or limited reinforcement rewards. To address this, we propose KnowVal, a new autonomous driving system that enables visual-language reasoning through the synergistic integration of open-world perception and knowledge retrieval. Specifically, we construct a comprehensive driving knowledge graph that encodes traffic laws, defensive driving principles, and ethical norms, complemented by an efficient LLM-based retrieval mechanism tailored for driving scenarios. Furthermore, we develop a human-preference dataset and train a Value Model to guide interpretable, value-aligned trajectory assessment. Experimental results show that our method substantially improves planning performance while remaining compatible with existing architectures. Notably, KnowVal achieves the lowest collision rate on nuScenes and state-of-the-art results on Bench2Drive.

**arXiv ID:** 2512.20299
</details>

<details>
<summary><strong>Deep Research Comparator: A Platform For Fine-grained Human Annotations of Deep Research Agents</strong> - Prahaladh Chandrahasan, Jiahe Jin, Zhihan Zhang, Tevin Wang, Andy Tang, Lucy Mo, Morteza Ziyadi, Leonardo F.R. Ribeiro, Zimeng Qiu, Markus Dreyer, Akari Asai, Chenyan Xiong - [[pdf]](https://arxiv.org/pdf/2507.05495)</summary>

**Abstract:** Effectively evaluating deep research agents that autonomously search the web, analyze information, and generate reports remains a major challenge, particularly when it comes to assessing long reports and giving detailed feedback on their intermediate steps. To address these gaps, we introduce Deep Research Comparator, a platform that offers a holistic framework for deep research agent hosting, side-by-side comparison, fine-grained human feedback collection, and ranking calculation. Given a user query, our platform displays the final reports from two different agents along with their intermediate steps during generation. Annotators can evaluate the overall quality of final reports based on side-by-side comparison, and also provide detailed feedback separately by assessing intermediate steps or specific text spans within the final report. Furthermore, we develop Simple Deepresearch, an end-to-end agent scaffold. This scaffold serves as a baseline that facilitates the easy integration of various large language models to transform them into deep research agents for evaluation. To demonstrate the platform's utility for deep research agent development, we have collected real user preference data from 17 annotators on three deep research agents. A demo video of our platform can be found at this https URL.

**arXiv ID:** 2507.05495
</details>

<details>
<summary><strong>Trust Semantics Distillation for Collaborator Selection via Memory-Augmented Agentic AI</strong> - Botao Zhu, Jeslyn Wang, Dusit Niyato, Xianbin Wang - [[pdf]](https://arxiv.org/pdf/2509.08151)</summary>

**Abstract:** Offloading computational tasks from resource-constrained devices to resource-abundant peers constitutes a critical paradigm for collaborative computing. Within this context, accurate trust evaluation of potential collaborating devices is essential for the effective execution of complex computing tasks. This trust evaluation process involves collecting diverse trust-related information from every potential collaborator and performing trust inference based on the collected data. However, when each resource-constrained device independently assesses all potential collaborators, frequent data exchange and complex reasoning can incur significant overhead and further degrade the timeliness of trust evaluation. To overcome these challenges, we propose a task-specific trust semantics distillation (TSD) model based on a large AI model (LAM)-enabled teacher-student agent architecture. Specifically, the teacher agent is deployed on a server with powerful computational capabilities and an augmented memory module to perform multidimensional trust-related data collection, task-specific trust semantics extraction, and task-collaborator matching analysis. Upon receiving task-specific evaluation requests from device-side student agents, the teacher agent transfers the trust semantics of potential collaborators to the student agents, enabling rapid and accurate collaborator selection. Experimental results demonstrate that the proposed TSD model can reduce collaborator evaluation time, decrease device resource consumption, and improve the accuracy of collaborator selection.

**arXiv ID:** 2509.08151
</details>

<details>
<summary><strong>Exploring Deep-to-Shallow Transformable Neural Networks for Intelligent Embedded Systems</strong> - Xiangzhong Luo, Weichen Liu - [[pdf]](https://arxiv.org/pdf/2512.19731)</summary>

**Abstract:** Thanks to the evolving network depth, convolutional neural networks (CNNs) have achieved remarkable success across various embedded scenarios, paving the way for ubiquitous embedded intelligence. Despite its promise, the evolving network depth comes at the cost of degraded hardware efficiency. In contrast to deep networks, shallow networks can deliver superior hardware efficiency but often suffer from inferior accuracy. To address this dilemma, we propose Double-Win NAS, a novel deep-to-shallow transformable neural architecture search (NAS) paradigm tailored for resource-constrained intelligent embedded systems. Specifically, Double-Win NAS strives to automatically explore deep networks to first win strong accuracy, which are then equivalently transformed into their shallow counterparts to further win strong hardware efficiency. In addition to search, we also propose two enhanced training techniques, including hybrid transformable training towards better training accuracy and arbitrary-resolution elastic training towards enabling natural network elasticity across arbitrary input resolutions. Extensive experimental results on two popular intelligent embedded systems (i.e., NVIDIA Jetson AGX Xavier and NVIDIA Jetson Nano) and two representative large-scale datasets (i.e., ImageNet and ImageNet-100) clearly demonstrate the superiority of Double-Win NAS over previous state-of-the-art NAS approaches.

**arXiv ID:** 2512.19731
</details>

<details>
<summary><strong>On-device Large Multi-modal Agent for Human Activity Recognition</strong> - Md Shakhrul Iman Siam, Ishtiaque Ahmed Showmik, Guanqun Song, Ting Zhu - [[pdf]](https://arxiv.org/pdf/2512.19742)</summary>

**Abstract:** Human Activity Recognition (HAR) has been an active area of research, with applications ranging from healthcare to smart environments. The recent advancements in Large Language Models (LLMs) have opened new possibilities to leverage their capabilities in HAR, enabling not just activity classification but also interpretability and human-like interaction. In this paper, we present a Large Multi-Modal Agent designed for HAR, which integrates the power of LLMs to enhance both performance and user engagement. The proposed framework not only delivers activity classification but also bridges the gap between technical outputs and user-friendly insights through its reasoning and question-answering capabilities. We conduct extensive evaluations using widely adopted HAR datasets, including HHAR, Shoaib, Motionsense to assess the performance of our framework. The results demonstrate that our model achieves high classification accuracy comparable to state-of-the-art methods while significantly improving interpretability through its reasoning and Q&A capabilities.

**arXiv ID:** 2512.19742
</details>

<details>
<summary><strong>Detecting Non-Optimal Decisions of Embodied Agents via Diversity-Guided Metamorphic Testing</strong> - Wenzhao Wu, Yahui Tang, Mingfei Cheng, Wenbing Tang, Yuan Zhou, Yang Liu - [[pdf]](https://arxiv.org/pdf/2512.20083)</summary>

**Abstract:** As embodied agents advance toward real-world deployment, ensuring optimal decisions becomes critical for resource-constrained applications. Current evaluation methods focus primarily on functional correctness, overlooking the non-functional optimality of generated plans. This gap can lead to significant performance degradation and resource waste. We identify and formalize the problem of Non-optimal Decisions (NoDs), where agents complete tasks successfully but inefficiently. We present NoD-DGMT, a systematic framework for detecting NoDs in embodied agent task planning via diversity-guided metamorphic testing. Our key insight is that optimal planners should exhibit invariant behavioral properties under specific transformations. We design four novel metamorphic relations capturing fundamental optimality properties: position detour suboptimality, action optimality completeness, condition refinement monotonicity, and scene perturbation invariance. To maximize detection efficiency, we introduce a diversity-guided selection strategy that actively selects test cases exploring different violation categories, avoiding redundant evaluations while ensuring comprehensive diversity coverage. Extensive experiments on the AI2-THOR simulator with four state-of-the-art planning models demonstrate that NoD-DGMT achieves violation detection rates of 31.9% on average, with our diversity-guided filter improving rates by 4.3% and diversity scores by 3.3 on average. NoD-DGMT significantly outperforms six baseline methods, with 16.8% relative improvement over the best baseline, and demonstrates consistent superiority across different model architectures and task complexities.

**arXiv ID:** 2512.20083
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>MolAct: An Agentic RL Framework for Molecular Editing and Property Optimization</strong> - Zhuo Yang, Yeyun chen, Jiaqing Xie, Ben Gao, Shuaike Shen, Wanhao Liu, Liujia Yang, Beilun Wang, Tianfan Fu, Yuqiang Li - [[pdf]](https://arxiv.org/pdf/2512.20135)</summary>

**Abstract:** Molecular editing and optimization are multi-step problems that require iteratively improving properties while keeping molecules chemically valid and structurally similar. We frame both tasks as sequential, tool-guided decisions and introduce MolAct, an agentic reinforcement learning framework that employs a two-stage training paradigm: first building editing capability, then optimizing properties while reusing the learned editing behaviors. To the best of our knowledge, this is the first work to formalize molecular design as an Agentic Reinforcement Learning problem, where an LLM agent learns to interleave reasoning, tool-use, and molecular optimization. The framework enables agents to interact in multiple turns, invoking chemical tools for validity checking, property assessment, and similarity control, and leverages their feedback to refine subsequent edits. We instantiate the MolAct framework to train two model families: MolEditAgent for molecular editing tasks and MolOptAgent for molecular optimization tasks. In molecular editing, MolEditAgent-7B delivers 100, 95, and 98 valid add, delete, and substitute edits, outperforming strong closed "thinking" baselines such as DeepSeek-R1; MolEditAgent-3B approaches the performance of much larger open "thinking" models like Qwen3-32B-think. In molecular optimization, MolOptAgent-7B (trained on MolEditAgent-7B) surpasses the best closed "thinking" baseline (e.g., Claude 3.7) on LogP and remains competitive on solubility, while maintaining balanced performance across other objectives. These results highlight that treating molecular design as a multi-step, tool-augmented process is key to reliable and interpretable improvements.

**arXiv ID:** 2512.20135
</details>

<details>
<summary><strong>MemR$^3$: Memory Retrieval via Reflective Reasoning for LLM Agents</strong> - Xingbo Du, Loka Li, Duzhen Zhang, Le Song - [[pdf]](https://arxiv.org/pdf/2512.20237)</summary>

**Abstract:** Memory systems have been designed to leverage past experiences in Large Language Model (LLM) agents. However, many deployed memory systems primarily optimize compression and storage, with comparatively less emphasis on explicit, closed-loop control of memory retrieval. From this observation, we build memory retrieval as an autonomous, accurate, and compatible agent system, named MemR$^3$, which has two core mechanisms: 1) a router that selects among retrieve, reflect, and answer actions to optimize answer quality; 2) a global evidence-gap tracker that explicitly renders the answering process transparent and tracks the evidence collection process. This design departs from the standard retrieve-then-answer pipeline by introducing a closed-loop control mechanism that enables autonomous decision-making. Empirical results on the LoCoMo benchmark demonstrate that MemR$^3$ surpasses strong baselines on LLM-as-a-Judge score, and particularly, it improves existing retrievers across four categories with an overall improvement on RAG (+7.29%) and Zep (+1.94%) using GPT-4.1-mini backend, offering a plug-and-play controller for existing memory stores.

**arXiv ID:** 2512.20237
</details>

<details>
<summary><strong>Automated stereotactic radiosurgery planning using a human-in-the-loop reasoning large language model agent</strong> - Humza Nusrat, Luke Francisco, Bing Luo, Hassan Bagher-Ebadian, Joshua Kim, Karen Chin-Snyder, Salim Siddiqui, Mira Shah, Eric Mellon, Mohammad Ghassemi, Anthony Doemer, Benjamin Movsas, Kundan Thind - [[pdf]](https://arxiv.org/pdf/2512.20586)</summary>

**Abstract:** Stereotactic radiosurgery (SRS) demands precise dose shaping around critical structures, yet black-box AI systems have limited clinical adoption due to opacity concerns. We tested whether chain-of-thought reasoning improves agentic planning in a retrospective cohort of 41 patients with brain metastases treated with 18 Gy single-fraction SRS. We developed SAGE (Secure Agent for Generative Dose Expertise), an LLM-based planning agent for automated SRS treatment planning. Two variants generated plans for each case: one using a non-reasoning model, one using a reasoning model. The reasoning variant showed comparable plan dosimetry relative to human planners on primary endpoints (PTV coverage, maximum dose, conformity index, gradient index; all p > 0.21) while reducing cochlear dose below human baselines (p = 0.022). When prompted to improve conformity, the reasoning model demonstrated systematic planning behaviors including prospective constraint verification (457 instances) and trade-off deliberation (609 instances), while the standard model exhibited none of these deliberative processes (0 and 7 instances, respectively). Content analysis revealed that constraint verification and causal explanation concentrated in the reasoning agent. The optimization traces serve as auditable logs, offering a path toward transparent automated planning.

**arXiv ID:** 2512.20586
</details>

<details>
<summary><strong>A Declarative Language for Building And Orchestrating LLM-Powered Agent Workflows</strong> - Ivan Daunis - [[pdf]](https://arxiv.org/pdf/2512.19769)</summary>

**Abstract:** Building deployment-ready LLM agents requires complex orchestration of tools, data sources, and control flow logic, yet existing systems tightly couple agent logic to specific programming languages and deployment models. We present a declarative system that separates agent workflow specification from implementation, enabling the same pipeline definition to execute across multiple backend languages (Java, Python, Go) and deployment environments (cloud-native, on-premises).
Our key insight is that most agent workflows consist of common patterns -- data serialization, filtering, RAG retrieval, API orchestration -- that can be expressed through a unified DSL rather than imperative code. This approach transforms agent development from application programming to configuration, where adding new tools or fine-tuning agent behaviors requires only pipeline specification changes, not code deployment. Our system natively supports A/B testing of agent strategies, allowing multiple pipeline variants to run on the same backend infrastructure with automatic metric collection and comparison.
We evaluate our approach on real-world e-commerce workflows at PayPal, processing millions of daily interactions. Our results demonstrate 60% reduction in development time, and 3x improvement in deployment velocity compared to imperative implementations. The language's declarative approach enables non-engineers to modify agent behaviors safely, while maintaining sub-100ms orchestration overhead. We show that complex workflows involving product search, personalization, and cart management can be expressed in under 50 lines of DSL compared to 500+ lines of imperative code.

**arXiv ID:** 2512.19769
</details>

<details>
<summary><strong>ABBEL: LLM Agents Acting through Belief Bottlenecks Expressed in Language</strong> - Aly Lidayan, Jakob Bjorner, Satvik Golechha, Kartik Goyal, Alane Suhr - [[pdf]](https://arxiv.org/pdf/2512.20111)</summary>

**Abstract:** As the length of sequential decision-making tasks increases, it becomes computationally impractical to keep full interaction histories in context. We introduce a general framework for LLM agents to maintain concise contexts through multi-step interaction: Acting through Belief Bottlenecks Expressed in Language (ABBEL), and methods to further improve ABBEL agents with RL post-training. ABBEL replaces long multi-step interaction history by a belief state, i.e., a natural language summary of what has been discovered about task-relevant unknowns. Under ABBEL, at each step the agent first updates a prior belief with the most recent observation from the environment to form a posterior belief, then uses only the posterior to select an action. We systematically evaluate frontier models under ABBEL across six diverse multi-step environments, finding that ABBEL supports generating interpretable beliefs while maintaining near-constant memory use over interaction steps. However, bottleneck approaches are generally prone to error propagation, which we observe causing inferior performance when compared to the full context setting due to errors in belief updating. Therefore, we train LLMs to generate and act on beliefs within the ABBEL framework via reinforcement learning (RL). We experiment with belief grading, to reward higher quality beliefs, as well as belief length penalties to reward more compressed beliefs. Our experiments demonstrate the ability of RL to improve ABBEL's performance beyond the full context setting, while using less memory than contemporaneous approaches.

**arXiv ID:** 2512.20111
</details>

<details>
<summary><strong>GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators</strong> - Jiacheng Guo, Ling Yang, Peter Chen, Qixin Xiao, Yinjie Wang, Xinzhe Juan, Jiahao Qiu, Ke Shen, Mengdi Wang - [[pdf]](https://arxiv.org/pdf/2512.19682)</summary>

**Abstract:** Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutionary game between an agent and a scalable, generative environment simulator. Unlike traditional methods that evolve models on static datasets, GenEnv instantiates a dataevolving: the simulator acts as a dynamic curriculum policy, continuously generating tasks specifically tailored to the agent's ``zone of proximal development''. This process is guided by a simple but effective $\alpha$-Curriculum Reward, which aligns task difficulty with the agent's current capabilities. We evaluate GenEnv on five benchmarks, including API-Bank, ALFWorld, BFCL, Bamboogle, and TravelPlanner. Across these tasks, GenEnv improves agent performance by up to \textbf{+40.3\%} over 7B baselines and matches or exceeds the average performance of larger models. Compared to Gemini 2.5 Pro-based offline data augmentation, GenEnv achieves better performance while using 3.3$\times$ less data. By shifting from static supervision to adaptive simulation, GenEnv provides a data-efficient pathway for scaling agent capabilities.

**arXiv ID:** 2512.19682
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (6 papers)</h2></summary>

<details>
<summary><strong>TongSIM: A General Platform for Simulating Intelligent Machines</strong> - Zhe Sun, Kunlun Wu, Chuanjian Fu, Zeming Song, Langyong Shi, Zihe Xue, Bohan Jing, Ying Yang, Xiaomeng Gao, Aijia Li, Tianyu Guo, Huiying Li, Xueyuan Yang, Rongkai Liu, Xinyi He, Yuxi Wang, Yue Li, Mingyuan Liu, Yujie Lu, Hongzhao Xie, Shiyun Zhao, Bo Dai, Wei Wang, Tao Yuan, Song-Chun Zhu, Yujia Peng, Zhenliang Zhang - [[pdf]](https://arxiv.org/pdf/2512.20206)</summary>

**Abstract:** As artificial intelligence (AI) rapidly advances, especially in multimodal large language models (MLLMs), research focus is shifting from single-modality text processing to the more complex domains of multimodal and embodied AI. Embodied intelligence focuses on training agents within realistic simulated environments, leveraging physical interaction and action feedback rather than conventionally labeled datasets. Yet, most existing simulation platforms remain narrowly designed, each tailored to specific tasks. A versatile, general-purpose training environment that can support everything from low-level embodied navigation to high-level composite activities, such as multi-agent social simulation and human-AI collaboration, remains largely unavailable. To bridge this gap, we introduce TongSIM, a high-fidelity, general-purpose platform for training and evaluating embodied agents. TongSIM offers practical advantages by providing over 100 diverse, multi-room indoor scenarios as well as an open-ended, interaction-rich outdoor town simulation, ensuring broad applicability across research needs. Its comprehensive evaluation framework and benchmarks enable precise assessment of agent capabilities, such as perception, cognition, decision-making, human-robot cooperation, and spatial and social reasoning. With features like customized scenes, task-adaptive fidelity, diverse agent types, and dynamic environmental simulation, TongSIM delivers flexibility and scalability for researchers, serving as a unified platform that accelerates training, evaluation, and advancement toward general embodied intelligence.

**arXiv ID:** 2512.20206
</details>

<details>
<summary><strong>LongVideoAgent: Multi-Agent Reasoning with Long Videos</strong> - Runtao Liu, Ziyi Liu, Jiaqi Tang, Yue Ma, Renjie Pi, Jipeng Zhang, Qifeng Chen - [[pdf]](https://arxiv.org/pdf/2512.20618)</summary>

**Abstract:** Recent advances in multimodal LLMs and systems that use tools for long-video QA point to the promise of reasoning over hour-long episodes. However, many methods still compress content into lossy summaries or rely on limited toolsets, weakening temporal grounding and missing fine-grained cues. We propose a multi-agent framework in which a master LLM coordinates a grounding agent to localize question-relevant segments and a vision agent to extract targeted textual observations. The master agent plans with a step limit, and is trained with reinforcement learning to encourage concise, correct, and efficient multi-agent cooperation. This design helps the master agent focus on relevant clips via grounding, complements subtitles with visual detail, and yields interpretable trajectories. On our proposed LongTVQA and LongTVQA+ which are episode-level datasets aggregated from TVQA/TVQA+, our multi-agent system significantly outperforms strong non-agent baselines. Experiments also show reinforcement learning further strengthens reasoning and planning for the trained agent. Code and data will be shared at this https URL.

**arXiv ID:** 2512.20618
</details>

<details>
<summary><strong>Multi-Agent Intelligence for Multidisciplinary Decision-Making in Gastrointestinal Oncology</strong> - Rongzhao Zhang, Junqiao Wang, Shuyun Yang, Mouxiao Bian, Chihao Zhang, Dongyang Wang, Qiujuan Yan, Yun Zhong, Yuwei Bai, Guanxu Zhu, Kangkun Mao, Miao Wang, Chao Ding, Renjie Lu, Lei Wang, Lei Zheng, Tao Zheng, Xi Wang, Zhuo Fan, Bing Han, Meiling Liu, Luyi Jiang, Dongming Shan, Wenzhong Jin, Jiwei Yu, Zheng Wang, Jie Xu, Meng Luo - [[pdf]](https://arxiv.org/pdf/2512.08674)</summary>

**Abstract:** Multimodal clinical reasoning in the field of gastrointestinal (GI) oncology necessitates the integrated interpretation of endoscopic imagery, radiological data, and biochemical markers. Despite the evident potential exhibited by Multimodal Large Language Models (MLLMs), they frequently encounter challenges such as context dilution and hallucination when confronted with intricate, heterogeneous medical histories. In order to address these limitations, a hierarchical Multi-Agent Framework is proposed, which emulates the collaborative workflow of a human Multidisciplinary Team (MDT). The system attained a composite expert evaluation score of 4.60/5.00, thereby demonstrating a substantial improvement over the monolithic baseline. It is noteworthy that the agent-based architecture yielded the most substantial enhancements in reasoning logic and medical accuracy. The findings indicate that mimetic, agent-based collaboration provides a scalable, interpretable, and clinically robust paradigm for automated decision support in oncology.

**arXiv ID:** 2512.08674
</details>

<details>
<summary><strong>cuPilot: A Strategy-Coordinated Multi-agent Framework for CUDA Kernel Evolution</strong> - Jinwu Chen, Qidie Wu, Bin Li, Lin Ma, Xin Si, Yang Hu, Shouyi Yin, Jun Yang - [[pdf]](https://arxiv.org/pdf/2512.16465)</summary>

**Abstract:** Optimizing CUDA kernels is a challenging and labor-intensive task, given the need for hardware-software co-design expertise and the proprietary nature of high-performance kernel libraries. While recent large language models (LLMs) combined with evolutionary algorithms show promise in automatic kernel optimization, existing approaches often fall short in performance due to their suboptimal agent designs and mismatched evolution representations. This work identifies these mismatches and proposes cuPilot, a strategy-coordinated multi-agent framework that introduces strategy as an intermediate semantic representation for kernel evolution. Key contributions include a strategy-coordinated evolution algorithm, roofline-guided prompting, and strategy-level population initialization. Experimental results show that the generated kernels by cuPilot achieve an average speed up of 3.09$\times$ over PyTorch on a benchmark of 100 kernels. On the GEMM tasks, cuPilot showcases sophisticated optimizations and achieves high utilization of critical hardware units. The generated kernels are open-sourced at this https URL.

**arXiv ID:** 2512.16465
</details>

<details>
<summary><strong>A Multi-Agent Retrieval-Augmented Framework for Work-in-Progress Predictio</strong> - Yousef Mehrdad Bibalan, Behrouz Far, Mohammad Moshirpour, Bahareh Ghiyasian - [[pdf]](https://arxiv.org/pdf/2512.19841)</summary>

**Abstract:** Work-in-Progress (WiP) prediction is critical for predictive process monitoring, enabling accurate anticipation of workload fluctuations and optimized operational planning. This paper proposes a retrieval-augmented, multi-agent framework that combines retrieval-augmented generation (RAG) and collaborative multi-agent reasoning for WiP prediction. The narrative generation component transforms structured event logs into semantically rich natural language stories, which are embedded into a semantic vector-based process memory to facilitate dynamic retrieval of historical context during inference. The framework includes predictor agents that independently leverage retrieved historical contexts and a decision-making assistant agent that extracts high-level descriptive signals from recent events. A fusion agent then synthesizes predictions using ReAct-style reasoning over agent outputs and retrieved narratives. We evaluate our framework on two real-world benchmark datasets. Results show that the proposed retrieval-augmented multi-agent approach achieves competitive prediction accuracy, obtaining a Mean Absolute Percentage Error (MAPE) of 1.50\% on one dataset, and surpassing Temporal Convolutional Networks (TCN), Long Short-Term Memory (LSTM), and persistence baselines. The results highlight improved robustness, demonstrating the effectiveness of integrating retrieval mechanisms and multi-agent reasoning in WiP prediction.

**arXiv ID:** 2512.19841
</details>

<details>
<summary><strong>PRISM: A Personality-Driven Multi-Agent Framework for Social Media Simulation</strong> - Zhixiang Lu, Xueyuan Deng, Yiran Liu, Yulong Li, Qiang Yan, Imran Razzak, Jionglong Su - [[pdf]](https://arxiv.org/pdf/2512.19933)</summary>

**Abstract:** Traditional agent-based models (ABMs) of opinion dynamics often fail to capture the psychological heterogeneity driving online polarization due to simplistic homogeneity assumptions. This limitation obscures the critical interplay between individual cognitive biases and information propagation, thereby hindering a mechanistic understanding of how ideological divides are amplified. To address this challenge, we introduce the Personality-Refracted Intelligent Simulation Model (PRISM), a hybrid framework coupling stochastic differential equations (SDE) for continuous emotional evolution with a personality-conditional partially observable Markov decision process (PC-POMDP) for discrete decision-making. In contrast to continuous trait approaches, PRISM assigns distinct Myers-Briggs Type Indicator (MBTI) based cognitive policies to multimodal large language model (MLLM) agents, initialized via data-driven priors from large-scale social media datasets. PRISM achieves superior personality consistency aligned with human ground truth, significantly outperforming standard homogeneous and Big Five benchmarks. This framework effectively replicates emergent phenomena such as rational suppression and affective resonance, offering a robust tool for analyzing complex social media ecosystems.

**arXiv ID:** 2512.19933
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>Drift-Corrected Monocular VIO and Perception-Aware Planning for Autonomous Drone Racing</strong> - Maulana Bisyir Azhari, Donghun Han, Je In You, Sungjun Park, David Hyunchul Shim - [[pdf]](https://arxiv.org/pdf/2512.20475)</summary>

**Abstract:** The Abu Dhabi Autonomous Racing League(A2RL) x Drone Champions League competition(DCL) requires teams to perform high-speed autonomous drone racing using only a single camera and a low-quality inertial measurement unit -- a minimal sensor set that mirrors expert human drone racing pilots. This sensor limitation makes the system susceptible to drift from Visual-Inertial Odometry (VIO), particularly during long and fast flights with aggressive maneuvers. This paper presents the system developed for the championship, which achieved a competitive performance. Our approach corrected VIO drift by fusing its output with global position measurements derived from a YOLO-based gate detector using a Kalman filter. A perception-aware planner generated trajectories that balance speed with the need to keep gates visible for the perception system. The system demonstrated high performance, securing podium finishes across multiple categories: third place in the AI Grand Challenge with top speed of 43.2 km/h, second place in the AI Drag Race with over 59 km/h, and second place in the AI Multi-Drone Race. We detail the complete architecture and present a performance analysis based on experimental data from the competition, contributing our insights on building a successful system for monocular vision-based autonomous drone flight.

**arXiv ID:** 2512.20475
</details>

<details>
<summary><strong>Contingency Model-based Control (CMC) for Communicationless Cooperative Collision Avoidance in Robot Swarms</strong> - Georg Schildbach - [[pdf]](https://arxiv.org/pdf/2512.20391)</summary>

**Abstract:** Cooperative collision avoidance between robots in swarm operations remains an open challenge. Assuming a decentralized architecture, each robot is responsible for making its own control decisions, including motion planning. To this end, most existing approaches mostly rely some form of (wireless) communication between the agents of the swarm. In reality, however, communication is brittle. It may be affected by latency, further delays and packet losses, transmission faults, and is subject to adversarial attacks, such as jamming or spoofing. This paper proposes Contingency Model-based Control (CMC) as a communicationless alternative. It follows the implicit cooperation paradigm, under which the design of the robots is based on consensual (offline) rules, similar to traffic rules. They include the definition of a contingency trajectory for each robot, and a method for construction of mutual collision avoidance constraints. The setup is shown to guarantee the recursive feasibility and collision avoidance between all swarm members in closed-loop operation. Moreover, CMC naturally satisfies the Plug \& Play paradigm, i.e., for new robots entering the swarm. Two numerical examples demonstrate that the collision avoidance guarantee is intact and that the robot swarm operates smoothly under the CMC regime.

**arXiv ID:** 2512.20391
</details>

<details>
<summary><strong>DRO-EDL-MPC: Evidential Deep Learning-Based Distributionally Robust Model Predictive Control for Safe Autonomous Driving</strong> - Hyeongchan Ham, Heejin Ahn - [[pdf]](https://arxiv.org/pdf/2507.05710)</summary>

**Abstract:** Safety is a critical concern in motion planning for autonomous vehicles. Modern autonomous vehicles rely on neural network-based perception, but making control decisions based on these inference results poses significant safety risks due to inherent uncertainties. To address this challenge, we present a distributionally robust optimization (DRO) framework that accounts for both aleatoric and epistemic perception uncertainties using evidential deep learning (EDL). Our approach introduces a novel ambiguity set formulation based on evidential distributions that dynamically adjusts the conservativeness according to perception confidence levels. We integrate this uncertainty-aware constraint into model predictive control (MPC), proposing the DRO-EDL-MPC algorithm with computational tractability for autonomous driving applications. Validation in the CARLA simulator demonstrates that our approach maintains efficiency under high perception confidence while enforcing conservative constraints under low confidence.

**arXiv ID:** 2507.05710
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (23 papers)</h2></summary>

<details>
<summary><strong>Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches</strong> - Chaithra, Kamesh Kadimisetty, Biju R Mohan - [[pdf]](https://arxiv.org/pdf/2512.20082)</summary>

**Abstract:** Financial sentiment analysis plays a crucial role in informing investment decisions, assessing market risk, and predicting stock price trends. Existing works in financial sentiment analysis have not considered the impact of stock prices or market feedback on sentiment analysis. In this paper, we propose an adaptive framework that integrates large language models (LLMs) with real-world stock market feedback to improve sentiment classification in the context of the Indian stock market. The proposed methodology fine-tunes the LLaMA 3.2 3B model using instruction-based learning on the SentiFin dataset. To enhance sentiment predictions, a retrieval-augmented generation (RAG) pipeline is employed that dynamically selects multi-source contextual information based on the cosine similarity of the sentence embeddings. Furthermore, a feedback-driven module is introduced that adjusts the reliability of the source by comparing predicted sentiment with actual next-day stock returns, allowing the system to iteratively adapt to market behavior. To generalize this adaptive mechanism across temporal data, a reinforcement learning agent trained using proximal policy optimization (PPO) is incorporated. The PPO agent learns to optimize source weighting policies based on cumulative reward signals from sentiment-return alignment. Experimental results on NIFTY 50 news headlines collected from 2024 to 2025 demonstrate that the proposed system significantly improves classification accuracy, F1-score, and market alignment over baseline models and static retrieval methods. The results validate the potential of combining instruction-tuned LLMs with dynamic feedback and reinforcement learning for robust, market-aware financial sentiment modeling.

**arXiv ID:** 2512.20082
</details>

<details>
<summary><strong>Scaling Reinforcement Learning for Content Moderation with Large Language Models</strong> - Hamed Firooz, Rui Liu, Yuchen Lu, Zhenyu Hou, Fangzhou Xiong, Xiaoyang Zhang, Changshu Jian, Zhicheng Zhu, Jiayuan Ma, Jacob Tao, Chaitali Gupta, Xiaochang Peng, Shike Mei, Hang Cui, Yang Qin, Shuo Tang, Jason Gaedtke, Arpit Mittal - [[pdf]](https://arxiv.org/pdf/2512.20061)</summary>

**Abstract:** Content moderation at scale remains one of the most pressing challenges in today's digital ecosystem, where billions of user- and AI-generated artifacts must be continuously evaluated for policy violations. Although recent advances in large language models (LLMs) have demonstrated strong potential for policy-grounded moderation, the practical challenges of training these systems to achieve expert-level accuracy in real-world settings remain largely unexplored, particularly in regimes characterized by label sparsity, evolving policy definitions, and the need for nuanced reasoning beyond shallow pattern matching. In this work, we present a comprehensive empirical investigation of scaling reinforcement learning (RL) for content classification, systematically evaluating multiple RL training recipes and reward-shaping strategies-including verifiable rewards and LLM-as-judge frameworks-to transform general-purpose language models into specialized, policy-aligned classifiers across three real-world content moderation tasks. Our findings provide actionable insights for industrial-scale moderation systems, demonstrating that RL exhibits sigmoid-like scaling behavior in which performance improves smoothly with increased training data, rollouts, and optimization steps before gradually saturating. Moreover, we show that RL substantially improves performance on tasks requiring complex policy-grounded reasoning while achieving up to 100x higher data efficiency than supervised fine-tuning, making it particularly effective in domains where expert annotations are scarce or costly.

**arXiv ID:** 2512.20061
</details>

<details>
<summary><strong>Graph-Symbolic Policy Enforcement and Control (G-SPEC): A Neuro-Symbolic Framework for Safe Agentic AI in 5G Autonomous Networks</strong> - Divya Vijay, Vignesh Ethiraj - [[pdf]](https://arxiv.org/pdf/2512.20275)</summary>

**Abstract:** As networks evolve toward 5G Standalone and 6G, operators face orchestration challenges that exceed the limits of static automation and Deep Reinforcement Learning. Although Large Language Model (LLM) agents offer a path toward intent-based networking, they introduce stochastic risks, including topology hallucinations and policy non-compliance. To mitigate this, we propose Graph-Symbolic Policy Enforcement and Control (G-SPEC), a neuro-symbolic framework that constrains probabilistic planning with deterministic verification. The architecture relies on a Governance Triad - a telecom-adapted agent (TSLAM-4B), a Network Knowledge Graph (NKG), and SHACL constraints. We evaluated G-SPEC on a simulated 450-node 5G Core, achieving zero safety violations and a 94.1% remediation success rate, significantly outperforming the 82.4% baseline. Ablation analysis indicates that NKG validation drives the majority of safety gains (68%), followed by SHACL policies (24%). Scalability tests on topologies ranging from 10K to 100K nodes demonstrate that validation latency scales as $O(k^{1.2})$ where $k$ is subgraph size. With a processing overhead of 142ms, G-SPEC is viable for SMO-layer operations.

**arXiv ID:** 2512.20275
</details>

<details>
<summary><strong>QoS-Aware Dynamic CU Selection in O-RAN with Graph-Based Reinforcement Learning</strong> - Sebastian Racedo, Brigitte Jaumard, Oscar Delgado, Meysam Masoudi - [[pdf]](https://arxiv.org/pdf/2512.19696)</summary>

**Abstract:** Open Radio Access Network (O RAN) disaggregates conventional RAN into interoperable components, enabling flexible resource allocation, energy savings, and agile architectural design. In legacy deployments, the binding between logical functions and physical locations is static, which leads to inefficiencies under time varying traffic and resource conditions. We address this limitation by relaxing the fixed mapping and performing dynamic service function chain (SFC) provisioning with on the fly O CU selection. We formulate the problem as a Markov decision process and solve it using GRLDyP, i.e., a graph neural network (GNN) assisted deep reinforcement learning (DRL). The proposed agent jointly selects routes and the O-CU location (from candidate sites) for each incoming service flow to minimize network energy consumption while satisfying quality of service (QoS) constraints. The GNN encodes the instantaneous network topology and resource utilization (e.g., CPU and bandwidth), and the DRL policy learns to balance grade of service, latency, and energy. We perform the evaluation of GRLDyP on a data set with 24-hour traffic traces from the city of Montreal, showing that dynamic O CU selection and routing significantly reduce energy consumption compared to a static mapping baseline, without violating QoS. The results highlight DRL based SFC provisioning as a practical control primitive for energy-aware, resource-adaptive O-RAN deployments.

**arXiv ID:** 2512.19696
</details>

<details>
<summary><strong>Bidirectional human-AI collaboration in brain tumour assessments improves both expert human and AI agent performance</strong> - James K Ruffle, Samia Mohinta, Guilherme Pombo, Asthik Biswas, Alan Campbell, Indran Davagnanam, David Doig, Ahmed Hamman, Harpreet Hyare, Farrah Jabeen, Emma Lim, Dermot Mallon, Stephanie Owen, Sophie Wilkinson, Sebastian Brandner, Parashkev Nachev - [[pdf]](https://arxiv.org/pdf/2512.19707)</summary>

**Abstract:** The benefits of artificial intelligence (AI) human partnerships-evaluating how AI agents enhance expert human performance-are increasingly studied. Though rarely evaluated in healthcare, an inverse approach is possible: AI benefiting from the support of an expert human agent. Here, we investigate both human-AI clinical partnership paradigms in the magnetic resonance imaging-guided characterisation of patients with brain tumours. We reveal that human-AI partnerships improve accuracy and metacognitive ability not only for radiologists supported by AI, but also for AI agents supported by radiologists. Moreover, the greatest patient benefit was evident with an AI agent supported by a human one. Synergistic improvements in agent accuracy, metacognitive performance, and inter-rater agreement suggest that AI can create more capable, confident, and consistent clinical agents, whether human or model-based. Our work suggests that the maximal value of AI in healthcare could emerge not from replacing human intelligence, but from AI agents that routinely leverage and amplify it.

**arXiv ID:** 2512.19707
</details>

<details>
<summary><strong>Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning</strong> - Jiayun Wu, Jiashuo Liu, Zhiyuan Zeng, Tianyang Zhan, Wenhao Huang - [[pdf]](https://arxiv.org/pdf/2512.19920)</summary>

**Abstract:** LLM deployment in critical domains is currently impeded by persistent hallucinations--generating plausible but factually incorrect assertions. While scaling laws drove significant improvements in general capabilities, theoretical frameworks suggest hallucination is not merely stochastic error but a predictable statistical consequence of training objectives prioritizing mimicking data distribution over epistemic honesty. Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero. This paper presents an exhaustive investigation into behavioral calibration, which incentivizes models to stochastically admit uncertainty by abstaining when not confident, aligning model behavior with accuracy. Synthesizing recent advances, we propose and evaluate training interventions optimizing strictly proper scoring rules for models to output a calibrated probability of correctness. Our methods enable models to either abstain from producing a complete response or flag individual claims where uncertainty remains. Utilizing Qwen3-4B-Instruct, empirical analysis reveals behavior-calibrated reinforcement learning allows smaller models to surpass frontier models in uncertainty quantification--a transferable meta-skill decouplable from raw predictive accuracy. Trained on math reasoning tasks, our model's log-scale Accuracy-to-Hallucination Ratio gain (0.806) exceeds GPT-5's (0.207) in a challenging in-domain evaluation (BeyondAIME). Moreover, in cross-domain factual QA (SimpleQA), our 4B LLM achieves zero-shot calibration error on par with frontier models including Grok-4 and Gemini-2.5-Pro, even though its factual accuracy is much lower.

**arXiv ID:** 2512.19920
</details>

<details>
<summary><strong>TableGPT-R1: Advancing Tabular Reasoning Through Reinforcement Learning</strong> - Saisai Yang, Qingyi Huang, Jing Yuan, Liangyu Zha, Kai Tang, Yuhang Yang, Ning Wang, Yucheng Wei, Liyao Li, Wentao Ye, Hao Chen, Tao Zhang, Junlin Zhou, Haobo Wang, Gang Chen, Junbo Zhao - [[pdf]](https://arxiv.org/pdf/2512.20312)</summary>

**Abstract:** Tabular data serves as the backbone of modern data analysis and scientific research. While Large Language Models (LLMs) fine-tuned via Supervised Fine-Tuning (SFT) have significantly improved natural language interaction with such structured data, they often fall short in handling the complex, multi-step reasoning and robust code execution required for real-world table tasks. Reinforcement Learning (RL) offers a promising avenue to enhance these capabilities, yet its application in the tabular domain faces three critical hurdles: the scarcity of high-quality agentic trajectories with closed-loop code execution and environment feedback on diverse table structures, the extreme heterogeneity of feedback signals ranging from rigid SQL execution to open-ended data interpretation, and the risk of catastrophic forgetting of general knowledge during vertical specialization. To overcome these challenges and unlock advanced reasoning on complex tables, we introduce \textbf{TableGPT-R1}, a specialized tabular model built on a systematic RL framework. Our approach integrates a comprehensive data engineering pipeline that synthesizes difficulty-stratified agentic trajectories for both supervised alignment and RL rollouts, a task-adaptive reward system that combines rule-based verification with a criteria-injected reward model and incorporates process-level step reward shaping with behavioral regularization, and a multi-stage training framework that progressively stabilizes reasoning before specializing in table-specific tasks. Extensive evaluations demonstrate that TableGPT-R1 achieves state-of-the-art performance on authoritative benchmarks, significantly outperforming baseline models while retaining robust general capabilities. Our model is available at this https URL.

**arXiv ID:** 2512.20312
</details>

<details>
<summary><strong>Identifying Appropriately-Sized Services with Deep Reinforcement Learning</strong> - Syeda Tasnim Fabiha, Saad Shafiq, Wesley Klewerton Guez Assunção, Nenad Medvidović - [[pdf]](https://arxiv.org/pdf/2512.20381)</summary>

**Abstract:** Service-based architecture (SBA) has gained attention in industry and academia as a means to modernize legacy systems. It refers to a design style that enables systems to be developed as suites of small, loosely coupled, and autonomous components (services) that encapsulate functionality and communicate via language-agnostic APIs. However, defining appropriately sized services that capture cohesive subsets of system functionality remains challenging. Existing work often relies on the availability of documentation, access to project personnel, or a priori knowledge of the target number of services, assumptions that do not hold in many real-world scenarios. Our work addresses these limitations using a deep reinforcement learning-based approach to identify appropriately sized services directly from implementation artifacts. We present Rake, a reinforcement learning-based technique that leverages available system documentation and source code to guide service decomposition at the level of implementation methods. Rake does not require specific documentation or access to project personnel and is language-agnostic. It also supports a customizable objective function that balances modularization quality and business capability alignment, i.e., the degree to which a service covers the targeted business capability. We applied Rake to four open-source legacy projects and compared it with two state-of-the-art techniques. On average, Rake achieved 7-14 percent higher modularization quality and 18-22 percent stronger business capability alignment. Our results further show that optimizing solely for business context can degrade decomposition quality in tightly coupled systems, highlighting the need for balanced objectives.

**arXiv ID:** 2512.20381
</details>

<details>
<summary><strong>Performative Policy Gradient: Optimality in Performative Reinforcement Learning</strong> - Debabrota Basu, Udvas Das, Brahim Driss, Uddalak Mukherjee - [[pdf]](https://arxiv.org/pdf/2512.20576)</summary>

**Abstract:** Post-deployment machine learning algorithms often influence the environments they act in, and thus shift the underlying dynamics that the standard reinforcement learning (RL) methods ignore. While designing optimal algorithms in this performative setting has recently been studied in supervised learning, the RL counterpart remains under-explored. In this paper, we prove the performative counterparts of the performance difference lemma and the policy gradient theorem in RL, and further introduce the Performative Policy Gradient algorithm (PePG). PePG is the first policy gradient algorithm designed to account for performativity in RL. Under softmax parametrisation, and also with and without entropy regularisation, we prove that PePG converges to performatively optimal policies, i.e. policies that remain optimal under the distribution shifts induced by themselves. Thus, PePG significantly extends the prior works in Performative RL that achieves performative stability but not optimality. Furthermore, our empirical analysis on standard performative RL environments validate that PePG outperforms standard policy gradient algorithms and the existing performative RL algorithms aiming for stability.

**arXiv ID:** 2512.20576
</details>

<details>
<summary><strong>Leveraging High-Fidelity Digital Models and Reinforcement Learning for Mission Engineering: A Case Study of Aerial Firefighting Under Perfect Information</strong> - İbrahim Oğuz Çetinkaya, Sajad Khodadadian, Taylan G. Topçu - [[pdf]](https://arxiv.org/pdf/2512.20589)</summary>

**Abstract:** As systems engineering (SE) objectives evolve from design and operation of monolithic systems to complex System of Systems (SoS), the discipline of Mission Engineering (ME) has emerged which is increasingly being accepted as a new line of thinking for the SE community. Moreover, mission environments are uncertain, dynamic, and mission outcomes are a direct function of how the mission assets will interact with this environment. This proves static architectures brittle and calls for analytically rigorous approaches for ME. To that end, this paper proposes an intelligent mission coordination methodology that integrates digital mission models with Reinforcement Learning (RL), that specifically addresses the need for adaptive task allocation and reconfiguration. More specifically, we are leveraging a Digital Engineering (DE) based infrastructure that is composed of a high-fidelity digital mission model and agent-based simulation; and then we formulate the mission tactics management problem as a Markov Decision Process (MDP), and employ an RL agent trained via Proximal Policy Optimization. By leveraging the simulation as a sandbox, we map the system states to actions, refining the policy based on realized mission outcomes. The utility of the RL-based intelligent mission coordinator is demonstrated through an aerial firefighting case study. Our findings indicate that the RL-based intelligent mission coordinator not only surpasses baseline performance but also significantly reduces the variability in mission performance. Thus, this study serves as a proof of concept demonstrating that DE-enabled mission simulations combined with advanced analytical tools offer a mission-agnostic framework for improving ME practice; which can be extended to more complicated fleet design and selection problems in the future from a mission-first perspective.

**arXiv ID:** 2512.20589
</details>

<details>
<summary><strong>Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning</strong> - Seijin Kobayashi, Yanick Schimpf, Maximilian Schlegel, Angelika Steger, Maciej Wolczyk, Johannes von Oswald, Nino Scherre, Kaitlin Maile, Guillaume Lajoie, Blake A. Richards, Rif A. Saurous, James Manyika, Blaise Agüera y Arcas, Alexander Meulemans, João Sacramento - [[pdf]](https://arxiv.org/pdf/2512.20605)</summary>

**Abstract:** Large-scale autoregressive models pretrained on next-token prediction and finetuned with reinforcement learning (RL) have achieved unprecedented success on many problem domains. During RL, these models explore by generating new outputs, one token at a time. However, sampling actions token-by-token can result in highly inefficient learning, particularly when rewards are sparse. Here, we show that it is possible to overcome this problem by acting and exploring within the internal representations of an autoregressive model. Specifically, to discover temporally-abstract actions, we introduce a higher-order, non-causal sequence model whose outputs control the residual stream activations of a base autoregressive model. On grid world and MuJoCo-based tasks with hierarchical structure, we find that the higher-order model learns to compress long activation sequence chunks onto internal controllers. Critically, each controller executes a sequence of behaviorally meaningful actions that unfold over long timescales and are accompanied with a learned termination condition, such that composing multiple controllers over time leads to efficient exploration on novel tasks. We show that direct internal controller reinforcement, a process we term "internal RL", enables learning from sparse rewards in cases where standard RL finetuning fails. Our results demonstrate the benefits of latent action generation and reinforcement in autoregressive models, suggesting internal RL as a promising avenue for realizing hierarchical RL within foundation models.

**arXiv ID:** 2512.20605
</details>

<details>
<summary><strong>CardAIc-Agents: A Multimodal Framework with Hierarchical Adaptation for Cardiac Care Support</strong> - Yuting Zhang, Karina V. Bunting, Asgher Champsi, Xiaoxia Wang, Wenqi Lu, Alexander Thorley, Sandeep S Hothi, Zhaowen Qiu, Baturalp Buyukates, Dipak Kotecha, Jinming Duan - [[pdf]](https://arxiv.org/pdf/2508.13256)</summary>

**Abstract:** Cardiovascular diseases (CVDs) remain the foremost cause of mortality worldwide, a burden worsened by a severe deficit of healthcare workers. Artificial intelligence (AI) agents have shown potential to alleviate this gap through automated detection and proactive screening, yet their clinical application remains limited by: 1) rigid sequential workflows, whereas clinical care often requires adaptive reasoning that select specific tests and, based on their results, guides personalised next steps; 2) reliance solely on intrinsic model capabilities to perform role assignment without domain-specific tool support; 3) general and static knowledge bases without continuous learning capability; and 4) fixed unimodal or bimodal inputs and lack of on-demand visual outputs when clinicians require visual clarification. In response, a multimodal framework, CardAIc-Agents, was proposed to augment models with external tools and adaptively support diverse cardiac tasks. First, a CardiacRAG agent generated task-aware plans from updatable cardiac knowledge, while the Chief agent integrated tools to autonomously execute these plans and deliver decisions. Second, to enable adaptive and case-specific customization, a stepwise update strategy was developed to dynamically refine plans based on preceding execution results, once the task was assessed as complex. Third, a multidisciplinary discussion team was proposed which was automatically invoked to interpret challenging cases, thereby supporting further adaptation. In addition, visual review panels were provided to assist validation when clinicians raised concerns. Experiments across three datasets showed the efficiency of CardAIc-Agents compared to mainstream Vision-Language Models (VLMs) and state-of-the-art agentic systems.

**arXiv ID:** 2508.13256
</details>

<details>
<summary><strong>Reinforcement Learning for Unsupervised Video Summarization with Reward Generator Training</strong> - Mehryar Abbasi, Hadi Hadizadeh, Parvaneh Saeedi - [[pdf]](https://arxiv.org/pdf/2407.04258)</summary>

**Abstract:** This paper presents a novel approach for unsupervised video summarization using reinforcement learning (RL), addressing limitations like unstable adversarial training and reliance on heuristic-based reward functions. The method operates on the principle that reconstruction fidelity serves as a proxy for informativeness, correlating summary quality with reconstruction ability. The summarizer model assigns importance scores to frames to generate the final summary. For training, RL is coupled with a unique reward generation pipeline that incentivizes improved reconstructions. This pipeline uses a generator model to reconstruct the full video from the selected summary frames; the similarity between the original and reconstructed video provides the reward signal. The generator itself is pre-trained self-supervisedly to reconstruct randomly masked frames. This two-stage training process enhances stability compared to adversarial architectures. Experimental results show strong alignment with human judgments and promising F-scores, validating the reconstruction objective.

**arXiv ID:** 2407.04258
</details>

<details>
<summary><strong>Interaction Dataset of Autonomous Vehicles with Traffic Lights and Signs</strong> - Zheng Li, Zhipeng Bao, Haoming Meng, Haotian Shi, Qianwen Li, Handong Yao, Xiaopeng Li - [[pdf]](https://arxiv.org/pdf/2501.12536)</summary>

**Abstract:** This paper presents the development of a comprehensive dataset capturing interactions between Autonomous Vehicles (AVs) and traffic control devices, specifically traffic lights and stop signs. Derived from the Waymo Motion dataset, our work addresses a critical gap in the existing literature by providing real-world trajectory data on how AVs navigate these traffic control devices. We propose a methodology for identifying and extracting relevant interaction trajectory data from the Waymo Motion dataset, incorporating over 37,000 instances with traffic lights and 44,000 with stop signs. Our methodology includes defining rules to identify various interaction types, extracting trajectory data, and applying a wavelet-based denoising method to smooth the acceleration and speed profiles and eliminate anomalous values, thereby enhancing the trajectory quality. Quality assessment metrics indicate that trajectories obtained in this study have anomaly proportions in acceleration and jerk profiles reduced to near-zero levels across all interaction categories. By making this dataset publicly available, we aim to address the current gap in datasets containing AV interaction behaviors with traffic lights and signs. Based on the organized and published dataset, we can gain a more in-depth understanding of AVs' behavior when interacting with traffic lights and signs. This will facilitate research on AV integration into existing transportation infrastructures and networks, supporting the development of more accurate behavioral models and simulation tools.

**arXiv ID:** 2501.12536
</details>

<details>
<summary><strong>Environment Scaling for Interactive Agentic Experience Collection: A Survey</strong> - Yuchen Huang, Sijia Li, Minghao Liu, Wei Liu, Shijue Huang, Zhiyuan Fan, Hou Pong Chan, Yi R. Fung - [[pdf]](https://arxiv.org/pdf/2511.09586)</summary>

**Abstract:** LLM-based agents can autonomously accomplish complex tasks across various domains. However, to further cultivate capabilities such as adaptive behavior and long-term decision-making, training on static datasets built from human-level knowledge is insufficient. These datasets are costly to construct and lack both dynamism and realism. A growing consensus is that agents should instead interact directly with environments and learn from experience through reinforcement learning. We formalize this iterative process as the Generation-Execution-Feedback (GEF) loop, where environments generate tasks to challenge agents, return observations in response to agents' actions during task execution, and provide evaluative feedback on rollouts for subsequent learning. Under this paradigm, environments function as indispensable producers of experiential data, highlighting the need to scale them toward greater complexity, realism, and interactivity. In this survey, we systematically review representative methods for environment scaling from a pioneering environment-centric perspective and organize them along the stages of the GEF loop, namely task generation, task execution, and feedback. We further analyze implementation frameworks, challenges, and applications, consolidating fragmented advances and outlining future research directions for agent intelligence.

**arXiv ID:** 2511.09586
</details>

<details>
<summary><strong>Deep Reinforcement Learning Optimization for Uncertain Nonlinear Systems via Event-Triggered Robust Adaptive Dynamic Programming</strong> - Ningwei Bai, Chi Pui Chan, Qichen Yin, Tengyang Gong, Yunda Yan, Zezhi Tang - [[pdf]](https://arxiv.org/pdf/2512.15735)</summary>

**Abstract:** This work proposes a unified control architecture that couples a Reinforcement Learning (RL)-driven controller with a disturbance-rejection Extended State Observer (ESO), complemented by an Event-Triggered Mechanism (ETM) to limit unnecessary computations. The ESO is utilized to estimate the system states and the lumped disturbance in real time, forming the foundation for effective disturbance compensation. To obtain near-optimal behavior without an accurate system description, a value-iteration-based Adaptive Dynamic Programming (ADP) method is adopted for policy approximation. The inclusion of the ETM ensures that parameter updates of the learning module are executed only when the state deviation surpasses a predefined bound, thereby preventing excessive learning activity and substantially reducing computational load. A Lyapunov-oriented analysis is used to characterize the stability properties of the resulting closed-loop system. Numerical experiments further confirm that the developed approach maintains strong control performance and disturbance tolerance, while achieving a significant reduction in sampling and processing effort compared with standard time-triggered ADP schemes.

**arXiv ID:** 2512.15735
</details>

<details>
<summary><strong>OpComm: A Reinforcement Learning Framework for Adaptive Buffer Control in Warehouse Volume Forecasting</strong> - Wilson Fung, Lu Guo, Drake Hilliard, Alessandro Casadei, Raj Ratan, Sreyoshi Bhaduri, Adi Surve, Nikhil Agarwal, Rohit Malshe, Pavan Mullapudi, Hungjen Wang, Saurabh Doodhwala, Ankush Pole, Arkajit Rakshit - [[pdf]](https://arxiv.org/pdf/2512.19738)</summary>

**Abstract:** Accurate forecasting of package volumes at delivery stations is critical for last-mile logistics, where errors lead to inefficient resource allocation, higher costs, and delivery delays. We propose OpComm, a forecasting and decision-support framework that combines supervised learning with reinforcement learning-based buffer control and a generative AI-driven communication module. A LightGBM regression model generates station-level demand forecasts, which serve as context for a Proximal Policy Optimization (PPO) agent that selects buffer levels from a discrete action set. The reward function penalizes under-buffering more heavily than over-buffering, reflecting real-world trade-offs between unmet demand risks and resource inefficiency. Station outcomes are fed back through a Monte Carlo update mechanism, enabling continual policy adaptation. To enhance interpretability, a generative AI layer produces executive-level summaries and scenario analyses grounded in SHAP-based feature attributions. Across 400+ stations, OpComm reduced Weighted Absolute Percentage Error (WAPE) by 21.65% compared to manual forecasts, while lowering under-buffering incidents and improving transparency for decision-makers. This work shows how contextual reinforcement learning, coupled with predictive modeling, can address operational forecasting challenges and bridge statistical rigor with practical decision-making in high-stakes logistics environments.

**arXiv ID:** 2512.19738
</details>

<details>
<summary><strong>Sample-Efficient Policy Constraint Offline Deep Reinforcement Learning based on Sample Filtering</strong> - Yuanhao Chen, Qi Liu, Pengbin Chen, Zhongjian Qiao, Yanjie Li - [[pdf]](https://arxiv.org/pdf/2512.20115)</summary>

**Abstract:** Offline reinforcement learning (RL) aims to learn a policy that maximizes the expected return using a given static dataset of transitions. However, offline RL faces the distribution shift problem. The policy constraint offline RL method is proposed to solve the distribution shift problem. During the policy constraint offline RL training, it is important to ensure the difference between the learned policy and behavior policy within a given threshold. Thus, the learned policy heavily relies on the quality of the behavior policy. However, a problem exists in existing policy constraint methods: if the dataset contains many low-reward transitions, the learned will be contained with a suboptimal reference policy, leading to slow learning speed, low sample efficiency, and inferior performances. This paper shows that the sampling method in policy constraint offline RL that uses all the transitions in the dataset can be improved. A simple but efficient sample filtering method is proposed to improve the sample efficiency and the final performance. First, we evaluate the score of the transitions by average reward and average discounted reward of episodes in the dataset and extract the transition samples of high scores. Second, the high-score transition samples are used to train the offline RL algorithms. We verify the proposed method in a series of offline RL algorithms and benchmark tasks. Experimental results show that the proposed method outperforms baselines.

**arXiv ID:** 2512.20115
</details>

<details>
<summary><strong>Recurrent Off-Policy Deep Reinforcement Learning Doesn't Have to be Slow</strong> - Tyler Clark, Christine Evers, Jonathon Hare - [[pdf]](https://arxiv.org/pdf/2512.20513)</summary>

**Abstract:** Recurrent off-policy deep reinforcement learning models achieve state-of-the-art performance but are often sidelined due to their high computational demands. In response, we introduce RISE (Recurrent Integration via Simplified Encodings), a novel approach that can leverage recurrent networks in any image-based off-policy RL setting without significant computational overheads via using both learnable and non-learnable encoder layers. When integrating RISE into leading non-recurrent off-policy RL algorithms, we observe a 35.6% human-normalized interquartile mean (IQM) performance improvement across the Atari benchmark. We analyze various implementation strategies to highlight the versatility and potential of our proposed framework.

**arXiv ID:** 2512.20513
</details>

<details>
<summary><strong>Diffusion Self-Weighted Guidance for Offline Reinforcement Learning</strong> - Augusto Tagle, Javier Ruiz-del-Solar, Felipe Tobar - [[pdf]](https://arxiv.org/pdf/2505.18345)</summary>

**Abstract:** Offline reinforcement learning (RL) recovers the optimal policy $\pi$ given historical observations of an agent. In practice, $\pi$ is modeled as a weighted version of the agent's behavior policy $\mu$, using a weight function $w$ working as a critic of the agent's behavior. Though recent approaches to offline RL based on diffusion models have exhibited promising results, the computation of the required scores is challenging due to their dependence on the unknown $w$. In this work, we alleviate this issue by constructing a diffusion over both the actions and the weights. With the proposed setting, the required scores are directly obtained from the diffusion model without learning extra networks. Our main conceptual contribution is a novel guidance method, where guidance (which is a function of $w$) comes from the same diffusion model, therefore, our proposal is termed Self-Weighted Guidance (SWG). We show that SWG generates samples from the desired distribution on toy examples and performs on par with state-of-the-art methods on D4RL's challenging environments, while maintaining a streamlined training pipeline. We further validate SWG through ablation studies on weight formulations and scalability.

**arXiv ID:** 2505.18345
</details>

<details>
<summary><strong>Reinforcement Learning From State and Temporal Differences</strong> - Lex Weaver, Jonathan Baxter - [[pdf]](https://arxiv.org/pdf/2512.08855)</summary>

**Abstract:** TD($\lambda$) with function approximation has proved empirically successful for some complex reinforcement learning problems. For linear approximation, TD($\lambda$) has been shown to minimise the squared error between the approximate value of each state and the true value. However, as far as policy is concerned, it is error in the relative ordering of states that is critical, rather than error in the state values. We illustrate this point, both in simple two-state and three-state systems in which TD($\lambda$)--starting from an optimal policy--converges to a sub-optimal policy, and also in backgammon. We then present a modified form of TD($\lambda$), called STD($\lambda$), in which function approximators are trained with respect to relative state values on binary decision problems. A theoretical analysis, including a proof of monotonic policy improvement for STD($\lambda$) in the context of the two-state system, is presented, along with a comparison with Bertsekas' differential training method [1]. This is followed by successful demonstrations of STD($\lambda$) on the two-state system and a variation on the well known acrobot problem.

**arXiv ID:** 2512.08855
</details>

<details>
<summary><strong>Learning Safe Autonomous Driving Policies Using Predictive Safety Representations</strong> - Mahesh Keswani, Raunak Bhattacharyya - [[pdf]](https://arxiv.org/pdf/2512.17586)</summary>

**Abstract:** Safe reinforcement learning (SafeRL) is a prominent paradigm for autonomous driving, where agents are required to optimize performance under strict safety requirements. This dual objective creates a fundamental tension, as overly conservative policies limit driving efficiency while aggressive exploration risks safety violations. The Safety Representations for Safer Policy Learning (SRPL) framework addresses this challenge by equipping agents with a predictive model of future constraint violations and has shown promise in controlled environments. This paper investigates whether SRPL extends to real-world autonomous driving scenarios. Systematic experiments on the Waymo Open Motion Dataset (WOMD) and NuPlan demonstrate that SRPL can improve the reward-safety tradeoff, achieving statistically significant improvements in success rate (effect sizes r = 0.65-0.86) and cost reduction (effect sizes r = 0.70-0.83), with p < 0.05 for observed improvements. However, its effectiveness depends on the underlying policy optimizer and the dataset distribution. The results further show that predictive safety representations play a critical role in improving robustness to observation noise. Additionally, in zero-shot cross-dataset evaluation, SRPL-augmented agents demonstrate improved generalization compared to non-SRPL methods. These findings collectively demonstrate the potential of predictive safety representations to strengthen SafeRL for autonomous driving.

**arXiv ID:** 2512.17586
</details>

<details>
<summary><strong>UrbanV2X: A Multisensory Vehicle-Infrastructure Dataset for Cooperative Navigation in Urban Areas</strong> - Qijun Qin, Ziqi Zhang, Yihan Zhong, Feng Huang, Xikun Liu, Runzhi Hu, Hang Chen, Wei Hu, Dongzhe Su, Jun Zhang, Hoi-Fung Ng, Weisong Wen - [[pdf]](https://arxiv.org/pdf/2512.20224)</summary>

**Abstract:** Due to the limitations of a single autonomous vehicle, Cellular Vehicle-to-Everything (C-V2X) technology opens a new window for achieving fully autonomous driving through sensor information sharing. However, real-world datasets supporting vehicle-infrastructure cooperative navigation in complex urban environments remain rare. To address this gap, we present UrbanV2X, a comprehensive multisensory dataset collected from vehicles and roadside infrastructure in the Hong Kong C-V2X testbed, designed to support research on smart mobility applications in dense urban areas. Our onboard platform provides synchronized data from multiple industrial cameras, LiDARs, 4D radar, ultra-wideband (UWB), IMU, and high-precision GNSS-RTK/INS navigation systems. Meanwhile, our roadside infrastructure provides LiDAR, GNSS, and UWB measurements. The entire vehicle-infrastructure platform is synchronized using the Precision Time Protocol (PTP), with sensor calibration data provided. We also benchmark various navigation algorithms to evaluate the collected cooperative data. The dataset is publicly available at this https URL.

**arXiv ID:** 2512.20224
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
