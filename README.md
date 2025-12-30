# Agent arXiv Daily

**Last Updated:** 2025-12-30 03:03:37

**Total Papers:** 56

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
<summary><strong>Towards Responsible and Explainable AI Agents with Consensus-Driven Reasoning</strong> - Eranga Bandara, Tharaka Hewa, Ross Gore, Sachin Shetty, Ravi Mukkamala, Peter Foytik, Abdul Rahman, Safdar H. Bouk, Xueping Liang, Amin Hass, Sachini Rajapakse, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan - [[pdf]](https://arxiv.org/pdf/2512.21699)</summary>

**Abstract:** Agentic AI represents a major shift in how autonomous systems reason, plan, and execute multi-step tasks through the coordination of Large Language Models (LLMs), Vision Language Models (VLMs), tools, and external services. While these systems enable powerful new capabilities, increasing autonomy introduces critical challenges related to explainability, accountability, robustness, and governance, especially when agent outputs influence downstream actions or decisions. Existing agentic AI implementations often emphasize functionality and scalability, yet provide limited mechanisms for understanding decision rationale or enforcing responsibility across agent interactions. This paper presents a Responsible(RAI) and Explainable(XAI) AI Agent Architecture for production-grade agentic workflows based on multi-model consensus and reasoning-layer governance. In the proposed design, a consortium of heterogeneous LLM and VLM agents independently generates candidate outputs from a shared input context, explicitly exposing uncertainty, disagreement, and alternative interpretations. A dedicated reasoning agent then performs structured consolidation across these outputs, enforcing safety and policy constraints, mitigating hallucinations and bias, and producing auditable, evidence-backed decisions. Explainability is achieved through explicit cross-model comparison and preserved intermediate outputs, while responsibility is enforced through centralized reasoning-layer control and agent-level constraints. We evaluate the architecture across multiple real-world agentic AI workflows, demonstrating that consensus-driven reasoning improves robustness, transparency, and operational trust across diverse application domains. This work provides practical guidance for designing agentic AI systems that are autonomous and scalable, yet responsible and explainable by construction.

**arXiv ID:** 2512.21699
</details>

<details>
<summary><strong>Context-Aware Intelligent Chatbot Framework Leveraging Mobile Sensing</strong> - Ziyan Zhang, Nan Gao, Zhiqiang Nie, Shantanu Pal, Haining Zhang - [[pdf]](https://arxiv.org/pdf/2512.22032)</summary>

**Abstract:** With the rapid advancement of large language models (LLMs), intelligent conversational assistants have demonstrated remarkable capabilities across various domains. However, they still mainly rely on explicit textual input and do not know the real world behaviors of users. This paper proposes a context-sensitive conversational assistant framework grounded in mobile sensing data. By collecting user behavior and environmental data through smartphones, we abstract these signals into 16 contextual scenarios and translate them into natural language prompts, thus improving the model's understanding of the user's state. We design a structured prompting system to guide the LLM in generating a more personalized and contextually relevant dialogue. This approach integrates mobile sensing with large language models, demonstrating the potential of passive behavioral data in intelligent conversation and offering a viable path toward digital health and personalized interaction.

**arXiv ID:** 2512.22032
</details>

<details>
<summary><strong>Teaching with AI: A Systematic Review of Chatbots, Generative Tools, and Tutoring Systems in Programming Education</strong> - Said Elnaffar, Farzad Rashidi, Abedallah Zaid Abualkishik - [[pdf]](https://arxiv.org/pdf/2510.03884)</summary>

**Abstract:** This review examines the role of artificial intelligence (AI) agents in programming education, focusing on how these tools are being integrated into educational practice and their impact on student learning outcomes. An analysis of fifty-eight peer-reviewed studies published between 2022 and 2025 identified three primary categories of AI agents: chatbots, generative AI (GenAI), and intelligent tutoring systems (ITS), with GenAI being the most frequently studied. The primary instructional objectives reported include enhanced programming support in 94.83% of studies, motivational and emotional benefits in 18.96%, and increased efficiency for educators in 6.90%. Reported benefits include personalized feedback, improved learning outcomes, and time savings. The review also highlights challenges, such as setup barriers documented in 93.10% of studies, overreliance resulting in superficial learning in 65.52%, and concerns regarding AI errors and academic integrity. These findings suggest the need for instructional frameworks that prioritize the development of prompt engineering skills and human oversight to address these issues. This review provides educators and curriculum designers with an evidence-based foundation for the practical and ethical integration of AI in programming education.

**arXiv ID:** 2510.03884
</details>

<details>
<summary><strong>Conversations with AI Chatbots Increase Short-Term Vaccine Intentions But Do Not Outperform Standard Public Health Messaging</strong> - Neil K. R. Sehgal, Sunny Rai, Manuel Tonneau, Anish K. Agarwal, Joseph Cappella, Melanie Kornides, Lyle Ungar, Alison Buttenheim, Sharath Chandra Guntuku - [[pdf]](https://arxiv.org/pdf/2504.20519)</summary>

**Abstract:** Large language model (LLM) based chatbots show promise in persuasive communication, but existing studies often rely on weak controls or focus on belief change rather than behavioral intentions or outcomes. This pre-registered multi-country (US, Canada, UK) randomized controlled trial involving 930 vaccine-hesitant parents evaluated brief (three-minute) multi-turn conversations with LLM-based chatbots against standard public health messaging approaches for increasing human papillomavirus (HPV) vaccine intentions for their children. Participants were randomly assigned to: (1) a weak control (no message), (2) a strong control reflecting the standard of care (reading official public health materials), or (3 and 4) one of two chatbot conditions. One chatbot was prompted to deliver short, conversational responses, while the other used the model's default output style (longer with bullet points). While chatbot interactions significantly increased self-reported vaccination intent (by 7.1-10.3 points on a 100-point scale) compared to no message, they did not outperform standard public health materials, with the conversational chatbot performing significantly worse. Additionally, while the short-term effects of chatbot interactions faded during a 15-day follow-up, the effects of public health material persisted through a 45-day follow-up relative to no message. These findings suggest that while LLMs can effectively shift vaccination intentions in the short-term, their incremental value over existing public health communications is questionable, offering a more tempered view of their persuasive capabilities and highlighting the importance of integrating AI-driven tools alongside, rather than replacing, current public health strategies.

**arXiv ID:** 2504.20519
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>AMS-IO-Bench and AMS-IO-Agent: Benchmarking and Structured Reasoning for Analog and Mixed-Signal Integrated Circuit Input/Output Design</strong> - Zhishuai Zhang, Xintian Li, Shilong Liu, Aodong Zhang, Lu Jie, Nan Sun - [[pdf]](https://arxiv.org/pdf/2512.21613)</summary>

**Abstract:** In this paper, we propose AMS-IO-Agent, a domain-specialized LLM-based agent for structure-aware input/output (I/O) subsystem generation in analog and mixed-signal (AMS) integrated circuits (ICs). The central contribution of this work is a framework that connects natural language design intent with industrial-level AMS IC design deliverables. AMS-IO-Agent integrates two key capabilities: (1) a structured domain knowledge base that captures reusable constraints and design conventions; (2) design intent structuring, which converts ambiguous user intent into verifiable logic steps using JSON and Python as intermediate formats. We further introduce AMS-IO-Bench, a benchmark for wirebond-packaged AMS I/O ring automation. On this benchmark, AMS-IO-Agent achieves over 70\% DRC+LVS pass rate and reduces design turnaround time from hours to minutes, outperforming the baseline LLM. Furthermore, an agent-generated I/O ring was fabricated and validated in a 28 nm CMOS tape-out, demonstrating the practical effectiveness of the approach in real AMS IC design flows. To our knowledge, this is the first reported human-agent collaborative AMS IC design in which an LLM-based agent completes a nontrivial subtask with outputs directly used in silicon.

**arXiv ID:** 2512.21613
</details>

<details>
<summary><strong>SpatialBench: Can Agents Analyze Real-World Spatial Biology Data?</strong> - Kenny Workman, Zhen Yang, Harihara Muralidharan, Hannah Le - [[pdf]](https://arxiv.org/pdf/2512.21907)</summary>

**Abstract:** Spatial transcriptomics assays are rapidly increasing in scale and complexity, making computational analysis a major bottleneck in biological discovery. Although frontier AI agents have improved dramatically at software engineering and general data analysis, it remains unclear whether they can extract biological insight from messy, real-world spatial datasets. We introduce SpatialBench, a benchmark of 146 verifiable problems derived from practical spatial analysis workflows spanning five spatial technologies and seven task categories. Each problem provides a snapshot of experimental data immediately prior to an analysis step and a deterministic grader that evaluates recovery of a key biological result. Benchmark data on frontier models shows that base model accuracy remains low (20-38% across model families), with strong model-task and model-platform interactions. Harness design has a large empirical effect on performance, indicating that tools, prompts, control flow, and execution environment should be evaluated and improved as first-class objects. SpatialBench serves both as a measurement tool and a diagnostic lens for developing agents that can interact with real spatial datasets faithfully, transparently, and reproducibly.

**arXiv ID:** 2512.21907
</details>

<details>
<summary><strong>AInsteinBench: Benchmarking Coding Agents on Scientific Repositories</strong> - Titouan Duston, Shuo Xin, Yang Sun, Daoguang Zan, Aoyan Li, Shulin Xin, Kai Shen, Yixiao Chen, Qiming Sun, Ge Zhang, Jiashuo Liu, Huan Zhou, Jingkai Liu, Zhichen Pu, Yuanheng Wang, Bo-Xuan Ge, Xin Tong, Fei Ye, Zhi-Chao Zhao, Wen-Biao Han, Zhoujian Cao, Yueran Zhao, Weiluo Ren, Qingshen Long, Yuxiao Liu, Anni Huang, Yidi Du, Yuanyuan Rong, Jiahao Peng - [[pdf]](https://arxiv.org/pdf/2512.21373)</summary>

**Abstract:** We introduce AInsteinBench, a large-scale benchmark for evaluating whether large language model (LLM) agents can operate as scientific computing development agents within real research software ecosystems. Unlike existing scientific reasoning benchmarks which focus on conceptual knowledge, or software engineering benchmarks that emphasize generic feature implementation and issue resolving, AInsteinBench evaluates models in end-to-end scientific development settings grounded in production-grade scientific repositories. The benchmark consists of tasks derived from maintainer-authored pull requests across six widely used scientific codebases, spanning quantum chemistry, quantum computing, molecular dynamics, numerical relativity, fluid dynamics, and cheminformatics. All benchmark tasks are carefully curated through multi-stage filtering and expert review to ensure scientific challenge, adequate test coverage, and well-calibrated difficulty. By leveraging evaluation in executable environments, scientifically meaningful failure modes, and test-driven verification, AInsteinBench measures a model's ability to move beyond surface-level code generation toward the core competencies required for computational scientific research.

**arXiv ID:** 2512.21373
</details>

<details>
<summary><strong>Intelligent recognition of GPR road hidden defect images based on feature fusion and attention mechanism</strong> - Haotian Lv, Yuhui Zhang, Jiangbo Dai, Hanli Wu, Jiaji Wang, Dawei Wang - [[pdf]](https://arxiv.org/pdf/2512.21452)</summary>

**Abstract:** Ground Penetrating Radar (GPR) has emerged as a pivotal tool for non-destructive evaluation of subsurface road defects. However, conventional GPR image interpretation remains heavily reliant on subjective expertise, introducing inefficiencies and inaccuracies. This study introduces a comprehensive framework to address these limitations: (1) A DCGAN-based data augmentation strategy synthesizes high-fidelity GPR images to mitigate data scarcity while preserving defect morphology under complex backgrounds; (2) A novel Multi-modal Chain and Global Attention Network (MCGA-Net) is proposed, integrating Multi-modal Chain Feature Fusion (MCFF) for hierarchical multi-scale defect representation and Global Attention Mechanism (GAM) for context-aware feature enhancement; (3) MS COCO transfer learning fine-tunes the backbone network, accelerating convergence and improving generalization. Ablation and comparison experiments validate the framework's efficacy. MCGA-Net achieves Precision (92.8%), Recall (92.5%), and mAP@50 (95.9%). In the detection of Gaussian noise, weak signals and small targets, MCGA-Net maintains robustness and outperforms other models. This work establishes a new paradigm for automated GPR-based defect detection, balancing computational efficiency with high accuracy in complex subsurface environments.

**arXiv ID:** 2512.21452
</details>

<details>
<summary><strong>Comparative Analysis of Deep Learning Models for Perception in Autonomous Vehicles</strong> - Jalal Khan - [[pdf]](https://arxiv.org/pdf/2512.21673)</summary>

**Abstract:** Recently, a plethora of machine learning (ML) and deep learning (DL) algorithms have been proposed to achieve the efficiency, safety, and reliability of autonomous vehicles (AVs). The AVs use a perception system to detect, localize, and identify other vehicles, pedestrians, and road signs to perform safe navigation and decision-making. In this paper, we compare the performance of DL models, including YOLO-NAS and YOLOv8, for a detection-based perception task. We capture a custom dataset and experiment with both DL models using our custom dataset. Our analysis reveals that the YOLOv8s model saves 75% of training time compared to the YOLO-NAS model. In addition, the YOLOv8s model (83%) outperforms the YOLO-NAS model (81%) when the target is to achieve the highest object detection accuracy. These comparative analyses of these new emerging DL models will allow the relevant research community to understand the models' performance under real-world use case scenarios.

**arXiv ID:** 2512.21673
</details>

<details>
<summary><strong>An Information Theoretic Perspective on Agentic System Design</strong> - Shizhe He, Avanika Narayan, Ishan S. Khare, Scott W. Linderman, Christopher Ré, Dan Biderman - [[pdf]](https://arxiv.org/pdf/2512.21720)</summary>

**Abstract:** Agentic language model (LM) systems power modern applications like "Deep Research" and "Claude Code," and leverage multi-LM architectures to overcome context limitations. Beneath their apparent diversity lies a recurring pattern: smaller "compressor" LMs (that can even run locally) distill raw context into compact text that is then consumed by larger "predictor" LMs. Despite their popularity, the design of compressor-predictor systems remains largely ad hoc, with little guidance on how compressor and predictor choices shape downstream performance. In practice, attributing gains to compression versus prediction requires costly, task-specific pairwise sweeps. We argue that these agentic system design questions are, at root, information-theoretic. Viewing the compressor LM as a noisy channel, we introduce a simple estimator of mutual information between the context and its compression to quantify compression quality in a task-independent way. We show that mutual information strongly predicts downstream performance, independent of any specific task. Through an information-theoretic framework, we perform a comprehensive empirical analysis across five datasets and three model families. Results reveal that larger compressors not only are more accurate, but also more token-efficient, conveying more bits of information per token. A 7B Qwen-2.5 compressor, for instance, is $1.6\times$ more accurate, $4.6\times$ more concise, and conveys $5.5\times$ more bits of mutual information per token than its 1.5B sibling. Across datasets, scaling compressors is substantially more effective than scaling predictors, enabling larger on-device compressors to pair with smaller cloud predictors. Applied to a Deep Research system, these principles enable local compressors as small as 3B parameters to recover $99\%$ of frontier-LM accuracy at $26\%$ of API costs.

**arXiv ID:** 2512.21720
</details>

<details>
<summary><strong>Agentic Structured Graph Traversal for Root Cause Analysis of Code-related Incidents in Cloud Applications</strong> - Shengkun Cui, Rahul Krishna, Saurabh Jha, Ravishankar K. Iyer - [[pdf]](https://arxiv.org/pdf/2512.22113)</summary>

**Abstract:** Cloud incidents pose major operational challenges in production, with unresolved production cloud incidents cost on average over $2M per hour. Prior research identifies code- and configuration-related issues as the predominant category of root causes in cloud incidents. This paper introduces PRAXIS, an orchestrator that manages and deploys an agentic workflow for diagnosing code- and configuration-caused cloud incidents. PRAXIS employs an LLM-driven structured traversal over two types of graph: (1) a service dependency graph (SDG) that captures microservice-level dependencies; and (2) a hammock-block program dependence graph (PDG) that captures code-level dependencies for each microservice. Together, these graphs encode microservice- and code-level dependencies and the LLM acts as a traversal policy over these graphs, moving between services and code dependencies to localize and explain failures. Compared to state-of-the-art ReAct baselines, PRAXIS improves RCA accuracy by up to 3.1x while reducing token consumption by 3.8x. PRAXIS is demonstrated on a set of 30 comprehensive real-world incidents that is being compiled into an RCA benchmark.

**arXiv ID:** 2512.22113
</details>

<details>
<summary><strong>Creative Agents: Empowering Agents with Imagination for Creative Tasks</strong> - Penglin Cai, Chi Zhang, Yuhui Fu, Haoqi Yuan, Zongqing Lu - [[pdf]](https://arxiv.org/pdf/2312.02519)</summary>

**Abstract:** We study building embodied agents for open-ended creative tasks. While existing methods build instruction-following agents that can perform diverse open-ended tasks, none of them demonstrates creativity -- the ability to give novel and diverse solutions implicit in the language instructions. This limitation comes from their inability to convert abstract language instructions into concrete goals and perform long-horizon planning for such complicated goals. Given the observation that humans perform creative tasks with imagination, we propose a class of solutions, where the controller is enhanced with an imaginator generating detailed imaginations of task outcomes conditioned on language instructions. We introduce several approaches to implementing the components of creative agents. We implement the imaginator with either a large language model for textual imagination or a diffusion model for visual imagination. The controller can either be a behavior-cloning policy or a pre-trained foundation model generating executable codes in the environment. We benchmark creative tasks with the challenging open-world game Minecraft, where the agents create diverse buildings given free-form language instructions. We propose novel evaluation metrics for open-ended creative tasks utilizing GPT-4V, which holds many advantages over existing metrics. We perform a detailed experimental analysis of creative agents, showing that creative agents are the first AI agents accomplishing diverse building creation in the survival mode of Minecraft. Our benchmark and models are open-source for future research on creative agents (this https URL).

**arXiv ID:** 2312.02519
</details>

<details>
<summary><strong>CP-Agent: Agentic Constraint Programming</strong> - Stefan Szeider - [[pdf]](https://arxiv.org/pdf/2508.07468)</summary>

**Abstract:** Translating natural language into formal constraint models requires expertise in the problem domain and modeling frameworks. To investigate whether constraint modeling benefits from agentic workflows, we introduce CP-Agent, a Python coding agent using the ReAct framework with a persistent IPython kernel. Domain knowledge is provided through a project prompt of under 50 lines. The agent iteratively executes code, observes the solver's feedback, and refines models based on the execution results.
We evaluate CP-Agent on CP-Bench's 101 constraint programming problems. We clarified the benchmark to address systematic ambiguities in problem specifications and errors in ground-truth models. On the clarified benchmark, CP-Agent solves all 101 problems. Ablation studies indicate that minimal guidance outperforms detailed procedural scaffolding, and that explicit task management tools have mixed effects on focused modeling tasks.

**arXiv ID:** 2508.07468
</details>

<details>
<summary><strong>RoboSafe: Safeguarding Embodied Agents via Executable Safety Logic</strong> - Le Wang, Zonghao Ying, Xiao Yang, Quanchen Zou, Zhenfei Yin, Tianlin Li, Jian Yang, Yaodong Yang, Aishan Liu, Xianglong Liu - [[pdf]](https://arxiv.org/pdf/2512.21220)</summary>

**Abstract:** Embodied agents powered by vision-language models (VLMs) are increasingly capable of executing complex real-world tasks, yet they remain vulnerable to hazardous instructions that may trigger unsafe behaviors. Runtime safety guardrails, which intercept hazardous actions during task execution, offer a promising solution due to their flexibility. However, existing defenses often rely on static rule filters or prompt-level control, which struggle to address implicit risks arising in dynamic, temporally dependent, and context-rich environments. To address this, we propose RoboSafe, a hybrid reasoning runtime safeguard for embodied agents through executable predicate-based safety logic. RoboSafe integrates two complementary reasoning processes on a Hybrid Long-Short Safety Memory. We first propose a Backward Reflective Reasoning module that continuously revisits recent trajectories in short-term memory to infer temporal safety predicates and proactively triggers replanning when violations are detected. We then propose a Forward Predictive Reasoning module that anticipates upcoming risks by generating context-aware safety predicates from the long-term safety memory and the agent's multimodal observations. Together, these components form an adaptive, verifiable safety logic that is both interpretable and executable as code. Extensive experiments across multiple agents demonstrate that RoboSafe substantially reduces hazardous actions (-36.8% risk occurrence) compared with leading baselines, while maintaining near-original task performance. Real-world evaluations on physical robotic arms further confirm its practicality. Code will be released upon acceptance.

**arXiv ID:** 2512.21220
</details>

<details>
<summary><strong>HARMON-E: Hierarchical Agentic Reasoning for Multimodal Oncology Notes to Extract Structured Data</strong> - Shashi Kant Gupta, Arijeet Pramanik, Jerrin John Thomas, Regina Schwind, Lauren Wiener, Avi Raju, Jeremy Kornbluth, Yanshan Wang, Zhaohui Su, Hrituraj Singh - [[pdf]](https://arxiv.org/pdf/2512.19864)</summary>

**Abstract:** Unstructured notes within the electronic health record (EHR) contain rich clinical information vital for cancer treatment decision making and research, yet reliably extracting structured oncology data remains challenging due to extensive variability, specialized terminology, and inconsistent document formats. Manual abstraction, although accurate, is prohibitively costly and unscalable. Existing automated approaches typically address narrow scenarios - either using synthetic datasets, restricting focus to document-level extraction, or isolating specific clinical variables (e.g., staging, biomarkers, histology) - and do not adequately handle patient-level synthesis across the large number of clinical documents containing contradictory information. In this study, we propose an agentic framework that systematically decomposes complex oncology data extraction into modular, adaptive tasks. Specifically, we use large language models (LLMs) as reasoning agents, equipped with context-sensitive retrieval and iterative synthesis capabilities, to exhaustively and comprehensively extract structured clinical variables from real-world oncology notes. Evaluated on a large-scale dataset of over 400,000 unstructured clinical notes and scanned PDF reports spanning 2,250 cancer patients, our method achieves an average F1-score of 0.93, with 100 out of 103 oncology-specific clinical variables exceeding 0.85, and critical variables (e.g., biomarkers and medications) surpassing 0.95. Moreover, integration of the agentic system into a data curation workflow resulted in 0.94 direct manual approval rate, significantly reducing annotation costs. To our knowledge, this constitutes the first exhaustive, end-to-end application of LLM-based agents for structured oncology data extraction at scale

**arXiv ID:** 2512.19864
</details>

<details>
<summary><strong>A Plan Reuse Mechanism for LLM-Driven Agent</strong> - Guopeng Li, Ruiqi Wu, Haisheng Tan - [[pdf]](https://arxiv.org/pdf/2512.21309)</summary>

**Abstract:** Integrating large language models (LLMs) into personal assistants, like Xiao Ai and Blue Heart V, effectively enhances their ability to interact with humans, solve complex tasks, and manage IoT devices. Such assistants are also termed LLM-driven agents. Upon receiving user requests, the LLM-driven agent generates plans using an LLM, executes these plans through various tools, and then returns the response to the user. During this process, the latency for generating a plan with an LLM can reach tens of seconds, significantly degrading user experience. Real-world dataset analysis shows that about 30% of the requests received by LLM-driven agents are identical or similar, which allows the reuse of previously generated plans to reduce latency. However, it is difficult to accurately define the similarity between the request texts received by the LLM-driven agent through directly evaluating the original request texts. Moreover, the diverse expressions of natural language and the unstructured format of plan texts make implementing plan reuse challenging. To address these issues, we present and implement a plan reuse mechanism for LLM-driven agents called AgentReuse. AgentReuse leverages the similarities and differences among requests' semantics and uses intent classification to evaluate the similarities between requests and enable the reuse of plans. Experimental results based on a real-world dataset demonstrate that AgentReuse achieves a 93% effective plan reuse rate, an F1 score of 0.9718, and an accuracy of 0.9459 in evaluating request similarities, reducing latency by 93.12% compared with baselines without using the reuse mechanism.

**arXiv ID:** 2512.21309
</details>

<details>
<summary><strong>MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles</strong> - Jing Han, Binwei Yan, Tianyu Guo, Zheyuan Bai, Mengyu Zheng, Hanting Chen, Ying Nie - [[pdf]](https://arxiv.org/pdf/2512.21708)</summary>

**Abstract:** Despite recent advancements of fine-tuning large language models (LLMs) to facilitate agent tasks, parameter-efficient fine-tuning (PEFT) methodologies for agent remain largely unexplored. In this paper, we introduce three key strategies for PEFT in agent tasks: 1) Inspired by the increasingly dominant Reason+Action paradigm, we first decompose the capabilities necessary for the agent tasks into three distinct roles: reasoner, executor, and summarizer. The reasoner is responsible for comprehending the user's query and determining the next role based on the execution trajectory. The executor is tasked with identifying the appropriate functions and parameters to invoke. The summarizer conveys the distilled information from conversations back to the user. 2) We then propose the Mixture-of-Roles (MoR) framework, which comprises three specialized Low-Rank Adaptation (LoRA) groups, each designated to fulfill a distinct role. By focusing on their respective specialized capabilities and engaging in collaborative interactions, these LoRAs collectively accomplish the agent task. 3) To effectively fine-tune the framework, we develop a multi-role data generation pipeline based on publicly available datasets, incorporating role-specific content completion and reliability verification. We conduct extensive experiments and thorough ablation studies on various LLMs and agent benchmarks, demonstrating the effectiveness of the proposed method. This project is publicly available at this https URL.

**arXiv ID:** 2512.21708
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>Accelerating Scientific Discovery with Autonomous Goal-evolving Agents</strong> - Yuanqi Du, Botao Yu, Tianyu Liu, Tony Shen, Junwu Chen, Jan G. Rittig, Kunyang Sun, Yikun Zhang, Zhangde Song, Bo Zhou, Cassandra Masschelein, Yingze Wang, Haorui Wang, Haojun Jia, Chao Zhang, Hongyu Zhao, Martin Ester, Teresa Head-Gordon, Carla P. Gomes, Huan Sun, Chenru Duan, Philippe Schwaller, Wengong Jin - [[pdf]](https://arxiv.org/pdf/2512.21782)</summary>

**Abstract:** There has been unprecedented interest in developing agents that expand the boundary of scientific discovery, primarily by optimizing quantitative objective functions specified by scientists. However, for grand challenges in science , these objectives are only imperfect proxies. We argue that automating objective function design is a central, yet unmet requirement for scientific discovery agents. In this work, we introduce the Scientific Autonomous Goal-evolving Agent (SAGA) to amend this challenge. SAGA employs a bi-level architecture in which an outer loop of LLM agents analyzes optimization outcomes, proposes new objectives, and converts them into computable scoring functions, while an inner loop performs solution optimization under the current objectives. This bi-level design enables systematic exploration of the space of objectives and their trade-offs, rather than treating them as fixed inputs. We demonstrate the framework through a broad spectrum of applications, including antibiotic design, inorganic materials design, functional DNA sequence design, and chemical process design, showing that automating objective formulation can substantially improve the effectiveness of scientific discovery agents.

**arXiv ID:** 2512.21782
</details>

<details>
<summary><strong>CosmoCore-Evo: Evolutionary Dream-Replay Reinforcement Learning for Adaptive Code Generation</strong> - Santhosh Kumar Ravindran - [[pdf]](https://arxiv.org/pdf/2512.21351)</summary>

**Abstract:** Building on the affective dream-replay reinforcement learning framework of CosmoCore, we introduce CosmoCore-Evo, an extension that incorporates evolutionary algorithms to enhance adaptability and novelty in code generation tasks. Inspired by anthropological aspects of human evolution, such as natural selection and adaptation in early hominids, CosmoCore-Evo treats RL trajectories as ``genomes'' that undergo mutation and selection during the nocturnal replay phase. This mechanism allows agents to break free from trained patterns, fostering emergent behaviors and improved performance in distribution-shifted environments, such as changing APIs or novel libraries. We augment the Dream Queue with evolutionary operations, including mutation of high-fitness trajectories and enterprise-tuned fitness functions that incorporate efficiency, compliance, and scalability metrics. Evaluated on extended benchmarks including HumanEval variants with shifts, BigCodeBench, and a custom PySpark pipeline simulation, CosmoCore-Evo achieves up to 35% higher novelty in solutions and 25% faster adaptation compared to the original CosmoCore and baselines like PPO and REAMER. Ablations confirm the role of evolutionary components in bridging the sentient gap for LLM agents. Code for replication, including a toy simulation, is provided.

**arXiv ID:** 2512.21351
</details>

<details>
<summary><strong>One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents</strong> - Zhaoxi Zhang, Yitong Duan, Yanzhi Zhang, Yiming Xu, Jiyan He, Yunfang Wu - [[pdf]](https://arxiv.org/pdf/2512.20957)</summary>

**Abstract:** Locating the files and functions requiring modification in large open-source software (OSS) repositories is challenging due to their scale and structural complexity. Existing large language model (LLM)-based methods typically treat this as a repository-level retrieval task and rely on multiple auxiliary tools, which overlook code execution logic and complicate model control. We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool-jumping to the definition of an invoked symbol. This unified design reflects the actual flow of code execution while simplifying tool manipulation. RepoNavigator is trained end-to-end via Reinforcement Learning (RL) directly from a pretrained model, without any closed-source distillation. Experiments demonstrate that RL-trained RepoNavigator achieves state-of-the-art performance, with the 7B model outperforming 14B baselines, the 14B model surpassing 32B competitors, and even the 32B model exceeding closed-source models such as Claude-3.7. These results confirm that integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization.

**arXiv ID:** 2512.20957
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>From Visual Perception to Deep Empathy: An Automated Assessment Framework for House-Tree-Person Drawings Using Multimodal LLMs and Multi-Agent Collaboration</strong> - Shuide Wen, Yu Sun, Beier Ku, Zhi Gao, Lijun Ma, Yang Yang, Can Jiao - [[pdf]](https://arxiv.org/pdf/2512.21360)</summary>

**Abstract:** Background: The House-Tree-Person (HTP) drawing test, introduced by John Buck in 1948, remains a widely used projective technique in clinical psychology. However, it has long faced challenges such as heterogeneous scoring standards, reliance on examiners subjective experience, and a lack of a unified quantitative coding system.
Results: Quantitative experiments showed that the mean semantic similarity between Multimodal Large Language Model (MLLM) interpretations and human expert interpretations was approximately 0.75 (standard deviation about 0.05). In structurally oriented expert data sets, this similarity rose to 0.85, indicating expert-level baseline comprehension. Qualitative analyses demonstrated that the multi-agent system, by integrating social-psychological perspectives and destigmatizing narratives, effectively corrected visual hallucinations and produced psychological reports with high ecological validity and internal coherence.
Conclusions: The findings confirm the potential of multimodal large models as standardized tools for projective assessment. The proposed multi-agent framework, by dividing roles, decouples feature recognition from psychological inference and offers a new paradigm for digital mental-health services.
Keywords: House-Tree-Person test; multimodal large language model; multi-agent collaboration; cosine similarity; computational psychology; artificial intelligence

**arXiv ID:** 2512.21360
</details>

<details>
<summary><strong>NEMO-4-PAYPAL: Leveraging NVIDIA's Nemo Framework for empowering PayPal's Commerce Agent</strong> - Ali Sahami, Sudhanshu Garg, Andrew Wang, Chaitanya Kulkarni, Farhad Farahani, Sean Yun-Shiuan Chuang, Jian Wan, Srinivasan Manoharan, Uma Kona, Nitin Sharma, Linsey Pang, Prakhar Mehrotra, Jessica Clark, Mark Moyou - [[pdf]](https://arxiv.org/pdf/2512.21578)</summary>

**Abstract:** We present the development and optimization of PayPal's Commerce Agent, powered by NEMO-4-PAYPAL, a multi-agent system designed to revolutionize agentic commerce on the PayPal platform. Through our strategic partnership with NVIDIA, we leveraged the NeMo Framework for LLM model fine-tuning to enhance agent performance. Specifically, we optimized the Search and Discovery agent by replacing our base model with a fine-tuned Nemotron small language model (SLM).
We conducted comprehensive experiments using the llama3.1-nemotron-nano-8B-v1 architecture, training LoRA-based models through systematic hyperparameter sweeps across learning rates, optimizers (Adam, AdamW), cosine annealing schedules, and LoRA ranks. Our contributions include: (1) the first application of NVIDIA's NeMo Framework to commerce-specific agent optimization, (2) LLM powered fine-tuning strategy for retrieval-focused commerce tasks, (3) demonstration of significant improvements in latency and cost while maintaining agent quality, and (4) a scalable framework for multi-agent system optimization in production e-commerce environments. Our results demonstrate that the fine-tuned Nemotron SLM effectively resolves the key performance issue in the retrieval component, which represents over 50\% of total agent response time, while maintaining or enhancing overall system performance.

**arXiv ID:** 2512.21578
</details>

<details>
<summary><strong>Democratizing Drug Discovery with an Orchestrated, Knowledge-Driven Multi-Agent Team for User-Guided Therapeutic Design</strong> - Takahide Suzuki, Kazuki Nakanishi, Takashi Fujiwara, Hideyuki Shimizu - [[pdf]](https://arxiv.org/pdf/2512.21623)</summary>

**Abstract:** Therapeutic discovery remains a formidable challenge, impeded by the fragmentation of specialized domains and the execution gap between computational design and physiological validation. Although generative AI offers promise, current models often function as passive assistants rather than as autonomous executors. Here, we introduce OrchestRA, a human-in-the-loop multi-agent platform that unifies biology, chemistry, and pharmacology into an autonomous discovery engine. Unlike static code generators, our agents actively execute simulations and reason the results to drive iterative optimization. Governed by an Orchestrator, a Biologist Agent leverages deep reasoning over a massive knowledge graph (>10 million associations) to pinpoint high-confidence targets; a Chemist Agent autonomously detects structural pockets for de novo design or drug repositioning; and a Pharmacologist Agent evaluates candidates via rigorous physiologically based pharmacokinetic (PBPK) simulations. This architecture establishes a dynamic feedback loop where pharmacokinetic and toxicity profiles directly trigger structural reoptimization. By seamlessly integrating autonomous execution with human guidance, OrchestRA democratizes therapeutic design, transforming drug discovery from a stochastic search to a programmable evidence-based engineering discipline.

**arXiv ID:** 2512.21623
</details>

<details>
<summary><strong>Multi-Agent LLM Committees for Autonomous Software Beta Testing</strong> - Sumanth Bharadwaj Hachalli Karanam, Dhiwahar Adhithya Kennady - [[pdf]](https://arxiv.org/pdf/2512.21352)</summary>

**Abstract:** Manual software beta testing is costly and time-consuming, while single-agent large language model (LLM) approaches suffer from hallucinations and inconsistent behavior. We propose a multi-agent committee framework in which diverse vision-enabled LLMs collaborate through a three-round voting protocol to reach consensus on testing actions. The framework combines model diversity, persona-driven behavioral variation, and visual user interface understanding to systematically explore web applications. Across 84 experimental runs with 9 testing personas and 4 scenarios, multi-agent committees achieve an 89.5 percent overall task success rate. Configurations with 2 to 4 agents reach 91.7 to 100 percent success, compared to 78.0 percent for single-agent baselines, yielding improvements of 13.7 to 22.0 percentage points. At the action level, the system attains a 93.1 percent success rate with a median per-action latency of 0.71 seconds, enabling real-time and continuous integration testing. Vision-enabled agents successfully identify user interface elements, with navigation and reporting achieving 100 percent success and form filling achieving 99.2 percent success. We evaluate the framework on WebShop and OWASP benchmarks, achieving 74.7 percent success on WebShop compared to a 50.1 percent published GPT-3 baseline, and 82.0 percent success on OWASP Juice Shop security testing with coverage of 8 of the 10 OWASP Top 10 vulnerability categories. Across 20 injected regressions, the committee achieves an F1 score of 0.91 for bug detection, compared to 0.78 for single-agent baselines. The open-source implementation enables reproducible research and practical deployment of LLM-based software testing in CI/CD pipelines.

**arXiv ID:** 2512.21352
</details>

<details>
<summary><strong>Multi-agent Adaptive Mechanism Design</strong> - Qiushi Han, David Simchi-Levi, Renfei Tan, Zishuo Zhao - [[pdf]](https://arxiv.org/pdf/2512.21794)</summary>

**Abstract:** We study a sequential mechanism design problem in which a principal seeks to elicit truthful reports from multiple rational agents while starting with no prior knowledge of agents' beliefs. We introduce Distributionally Robust Adaptive Mechanism (DRAM), a general framework combining insights from both mechanism design and online learning to jointly address truthfulness and cost-optimality. Throughout the sequential game, the mechanism estimates agents' beliefs and iteratively updates a distributionally robust linear program with shrinking ambiguity sets to reduce payments while preserving truthfulness. Our mechanism guarantees truthful reporting with high probability while achieving $\tilde{O}(\sqrt{T})$ cumulative regret, and we establish a matching lower bound showing that no truthful adaptive mechanism can asymptotically do better. The framework generalizes to plug-in estimators, supporting structured priors and delayed feedback. To our knowledge, this is the first adaptive mechanism under general settings that maintains truthfulness and achieves optimal regret when incentive constraints are unknown and must be learned.

**arXiv ID:** 2512.21794
</details>

<details>
<summary><strong>MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting</strong> - Marc S. Montalvo, Hamed Yaghoobian - [[pdf]](https://arxiv.org/pdf/2512.21878)</summary>

**Abstract:** Recent advances in large language models (LLMs) are transforming data-intensive domains, with finance representing a high-stakes environment where transparent and reproducible analysis of heterogeneous signals is essential. Traditional quantitative methods remain vulnerable to survivorship bias, while many AI-driven approaches struggle with signal integration, reproducibility, and computational efficiency. We introduce MASFIN, a modular multi-agent framework that integrates LLMs with structured financial metrics and unstructured news, while embedding explicit bias-mitigation protocols. The system leverages GPT-4.1-nano for reproducability and cost-efficient inference and generates weekly portfolios of 15-30 equities with allocation weights optimized for short-term performance. In an eight-week evaluation, MASFIN delivered a 7.33% cumulative return, outperforming the S&P 500, NASDAQ-100, and Dow Jones benchmarks in six of eight weeks, albeit with higher volatility. These findings demonstrate the promise of bias-aware, generative AI frameworks for financial forecasting and highlight opportunities for modular multi-agent design to advance practical, transparent, and reproducible approaches in quantitative finance.

**arXiv ID:** 2512.21878
</details>

<details>
<summary><strong>A2P-Vis: an Analyzer-to-Presenter Agentic Pipeline for Visual Insights Generation and Reporting</strong> - Shuyu Gan, Renxiang Wang, James Mooney, Dongyeop Kang - [[pdf]](https://arxiv.org/pdf/2512.22101)</summary>

**Abstract:** Automating end-to-end data science pipeline with AI agents still stalls on two gaps: generating insightful, diverse visual evidence and assembling it into a coherent, professional report. We present A2P-Vis, a two-part, multi-agent pipeline that turns raw datasets into a high-quality data-visualization report. The Data Analyzer orchestrates profiling, proposes diverse visualization directions, generates and executes plotting code, filters low-quality figures with a legibility checker, and elicits candidate insights that are automatically scored for depth, correctness, specificity, depth and actionability. The Presenter then orders topics, composes chart-grounded narratives from the top-ranked insights, writes justified transitions, and revises the document for clarity and consistency, yielding a coherent, publication-ready report. Together, these agents convert raw data into curated materials (charts + vetted insights) and into a readable narrative without manual glue work. We claim that by coupling a quality-assured Analyzer with a narrative Presenter, A2P-Vis operationalizes co-analysis end-to-end, improving the real-world usefulness of automated data analysis for practitioners. For the complete dataset report, please see: this https URL.

**arXiv ID:** 2512.22101
</details>

<details>
<summary><strong>Multi-Agent Collaborative Reward Design for Enhancing Reasoning in Reinforcement Learning</strong> - Pei Yang, Ke Zhang, Ji Wang, Xiao Chen, Yuxin Tang, Eric Yang, Lynn Ai, Bill Shi - [[pdf]](https://arxiv.org/pdf/2511.16202)</summary>

**Abstract:** We present CRM (Multi-Agent Collaborative Reward Model), a framework that replaces a single black-box reward model with a coordinated team of specialist evaluators to improve robustness and interpretability in RLHF. Conventional reward models struggle to jointly optimize multiple, sometimes conflicting, preference dimensions (e.g., factuality, helpfulness, safety) and offer limited transparency into why a score is assigned. CRM addresses these issues by decomposing preference evaluation into domain-specific agents that each produce partial signals, alongside global evaluators such as ranker-based and embedding-similarity rewards. A centralized aggregator fuses these signals at each timestep, balancing factors like step-wise correctness, multi-agent agreement, and repetition penalties, yielding a single training reward compatible with standard RL pipelines. The policy is optimized with advantage-based updates (e.g., GAE), while a value model regresses to the aggregated reward, enabling multi-perspective reward shaping without requiring additional human annotations beyond those used to train the evaluators. To support training and assessment, we introduce rewardBench, a benchmark and training suite aligned with the collaborative structure of CRM. Together, CRM and rewardBench provide a practical, modular path to more transparent reward modeling and more stable optimization.

**arXiv ID:** 2511.16202
</details>

<details>
<summary><strong>GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion</strong> - Tongxuan Liu, Xingyu Wang, Weizhe Huang, Wenjiang Xu, Yuting Zeng, Lei Jiang, Hailong Yang, Jing Li - [[pdf]](https://arxiv.org/pdf/2409.14051)</summary>

**Abstract:** In recent years, Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse NLP tasks. Extensive research has explored how to enhance the logical reasoning abilities such as Chain-of-Thought, Chain-of-Thought with Self-Consistency, Tree-Of-Thoughts, and multi-agent debates. In the context of multi-agent debates, significant performance improvements can be achieved with an increasing number of agents and debate rounds. However, the escalation in the number of agents and debate rounds can drastically raise the tokens cost of debates, thereby limiting the scalability of the multi-agent debate technique. To better harness the advantages of multi-agent debates in logical reasoning tasks, this paper proposes a method to significantly reduce token cost in multi-agent debates. This approach involves dividing all agents into multiple debate groups, with agents engaging in debates within their respective groups and sharing interim debate results between groups. Comparative experiments across multiple datasets have demonstrated that this method can reduce the total tokens by up to 51.7% during debates and while potentially enhancing accuracy by as much as 25%. Our method significantly enhances the performance and efficiency of interactions in the multi-agent debate.

**arXiv ID:** 2409.14051
</details>

<details>
<summary><strong>The AI Committee: A Multi-Agent Framework for Automated Validation and Remediation of Web-Sourced Data</strong> - Sunith Vallabhaneni, Thomas Berkane, Maimuna Majumder - [[pdf]](https://arxiv.org/pdf/2512.21481)</summary>

**Abstract:** Many research areas rely on data from the web to gain insights and test their methods. However, collecting comprehensive research datasets often demands manually reviewing many web pages to identify and record relevant data points, which is labor-intensive and susceptible to error. While the emergence of large language models (LLM)-powered web agents has begun to automate parts of this process, they often struggle to ensure the validity of the data they collect. Indeed, these agents exhibit several recurring failure modes - including hallucinating or omitting values, misinterpreting page semantics, and failing to detect invalid information - which are subtle and difficult to detect and correct manually. To address this, we introduce the AI Committee, a novel model-agnostic multi-agent system that automates the process of validating and remediating web-sourced datasets. Each agent is specialized in a distinct task in the data quality assurance pipeline, from source scrutiny and fact-checking to data remediation and integrity validation. The AI Committee leverages various LLM capabilities - including in-context learning for dataset adaptation, chain-of-thought reasoning for complex semantic validation, and a self-correction loop for data remediation - all without task-specific training. We demonstrate the effectiveness of our system by applying it to three real-world datasets, showing that it generalizes across LLMs and significantly outperforms baseline approaches, achieving data completeness up to 78.7% and precision up to 100%. We additionally conduct an ablation study demonstrating the contribution of each agent to the Committee's performance. This work is released as an open-source tool for the research community.

**arXiv ID:** 2512.21481
</details>

<details>
<summary><strong>Analyzing Code Injection Attacks on LLM-based Multi-Agent Systems in Software Development</strong> - Brian Bowers, Smita Khapre, Jugal Kalita - [[pdf]](https://arxiv.org/pdf/2512.21818)</summary>

**Abstract:** Agentic AI and Multi-Agent Systems are poised to dominate industry and society imminently. Powered by goal-driven autonomy, they represent a powerful form of generative AI, marking a transition from reactive content generation into proactive multitasking capabilities. As an exemplar, we propose an architecture of a multi-agent system for the implementation phase of the software engineering process. We also present a comprehensive threat model for the proposed system. We demonstrate that while such systems can generate code quite accurately, they are vulnerable to attacks, including code injection. Due to their autonomous design and lack of humans in the loop, these systems cannot identify and respond to attacks by themselves. This paper analyzes the vulnerability of multi-agent systems and concludes that the coder-reviewer-tester architecture is more resilient than both the coder and coder-tester architectures, but is less efficient at writing code. We find that by adding a security analysis agent, we mitigate the loss in efficiency while achieving even better resiliency. We conclude by demonstrating that the security analysis agent is vulnerable to advanced code injection attacks, showing that embedding poisonous few-shot examples in the injected code can increase the attack success rate from 0% to 71.95%.

**arXiv ID:** 2512.21818
</details>

<details>
<summary><strong>AI Urban Scientist: Multi-Agent Collaborative Automation for Urban Research</strong> - Tong Xia, Jiankun Zhang, Ruiwen You, Ao Xu, Linghao Zhang, Tengyao Tu, Jingzhi Wang, Jinghua Piao, Yunke Zhang, Fengli Xu, Yong Li - [[pdf]](https://arxiv.org/pdf/2512.07849)</summary>

**Abstract:** Urban research aims to understand how cities operate and evolve as complex adaptive systems. With the rapid growth of urban data and analytical methodologies, the central challenge of the field has shifted from data availability to the integration of heterogeneous data into coherent, verifiable urban knowledge through multidisciplinary approaches. Recent advances in AI, particularly the emergence of large language models (LLMs), have enabled the development of AI scientists capable of autonomous reasoning, hypothesis generation, and data-driven experimentation, demonstrating substantial potential for autonomous urban research. However, most general-purpose AI systems remain misaligned with the domain-specific knowledge, methodological conventions, and inferential standards required in urban studies. Here, we introduce the AI Urban Scientist, a knowledge-driven multi-agent framework designed to support autonomous urban research. Grounded in hypotheses, peer-review feedback, datasets, and research methodologies distilled from large-scale prior studies, the system constructs structured domain knowledge that guides LLM-based agents to automatically generate hypotheses, identify and integrate multi-source urban datasets, conduct empirical analyses and simulations, and iteratively refine analytical methods. Through this process, the framework synthesizes new insights in urban science and accelerates the urban research lifecycle.

**arXiv ID:** 2512.07849
</details>

<details>
<summary><strong>A Survey of Freshness-Aware Wireless Networking with Reinforcement Learning</strong> - Alimu Alibotaiken, Suyang Wang, Oluwaseun T. Ajayi, Yu Cheng - [[pdf]](https://arxiv.org/pdf/2512.21412)</summary>

**Abstract:** The age of information (AoI) has become a central measure of data freshness in modern wireless systems, yet existing surveys either focus on classical AoI formulations or provide broad discussions of reinforcement learning (RL) in wireless networks without addressing freshness as a unified learning problem. Motivated by this gap, this survey examines RL specifically through the lens of AoI and generalized freshness optimization. We organize AoI and its variants into native, function-based, and application-oriented families, providing a clearer view of how freshness should be modeled in B5G and 6G systems. Building on this foundation, we introduce a policy-centric taxonomy that reflects the decisions most relevant to freshness, consisting of update-control RL, medium-access RL, risk-sensitive RL, and multi-agent RL. This structure provides a coherent framework for understanding how learning can support sampling, scheduling, trajectory planning, medium access, and distributed coordination. We further synthesize recent progress in RL-driven freshness control and highlight open challenges related to delayed decision processes, stochastic variability, and cross-layer design. The goal is to establish a unified foundation for learning-based freshness optimization in next-generation wireless networks.

**arXiv ID:** 2512.21412
</details>

<details>
<summary><strong>Cerberus: Multi-Agent Reasoning and Coverage-Guided Exploration for Static Detection of Runtime Errors</strong> - Hridya Dhulipala, Xiaokai Rong, Tien N. Nguyen - [[pdf]](https://arxiv.org/pdf/2512.21431)</summary>

**Abstract:** In several software development scenarios, it is desirable to detect runtime errors and exceptions in code snippets without actual execution. A typical example is to detect runtime exceptions in online code snippets before integrating them into a codebase. In this paper, we propose Cerberus, a novel predictive, execution-free coverage-guided testing framework. Cerberus uses LLMs to generate the inputs that trigger runtime errors and to perform code coverage prediction and error detection without code execution. With a two-phase feedback loop, Cerberus first aims to both increasing code coverage and detecting runtime errors, then shifts to focus only detecting runtime errors when the coverage reaches 100% or its maximum, enabling it to perform better than prompting the LLMs for both purposes. Our empirical evaluation demonstrates that Cerberus performs better than conventional and learning-based testing frameworks for (in)complete code snippets by generating high-coverage test cases more efficiently, leading to the discovery of more runtime errors.

**arXiv ID:** 2512.21431
</details>

<details>
<summary><strong>Fuzzwise: Intelligent Initial Corpus Generation for Fuzzing</strong> - Hridya Dhulipala, Xiaokai Rong, Aashish Yadavally, Tien N. Nguyen - [[pdf]](https://arxiv.org/pdf/2512.21440)</summary>

**Abstract:** In mutation-based greybox fuzzing, generating high-quality input seeds for the initial corpus is essential for effective fuzzing. Rather than conducting separate phases for generating a large corpus and subsequently minimizing it, we propose FuzzWise which integrates them into one process to generate the optimal initial corpus of seeds (ICS). FuzzWise leverages a multi-agent framework based on Large Language Models (LLMs). The first LLM agent generates test cases for the target program. The second LLM agent, which functions as a predictive code coverage module, assesses whether each generated test case will enhance the overall coverage of the current corpus. The streamlined process allows each newly generated test seed to be immediately evaluated for its contribution to the overall coverage. FuzzWise employs a predictive approach using an LLM and eliminates the need for actual execution, saving computational resources and time, particularly in scenarios where the execution is not desirable or even impossible. Our empirical evaluation demonstrates that FuzzWise generates significantly fewer test cases than baseline methods. Despite the lower number of test cases, FuzzWise achieves high code coverage and triggers more runtime errors compared to the baselines. Moreover, it is more time-efficient and coverage-efficient in producing an initial corpus catching more errors.

**arXiv ID:** 2512.21440
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>Reflection-Driven Control for Trustworthy Code Agents</strong> - Bin Wang, Jiazheng Quan, Xingrui Yu, Hansen Hu, Yuhao, Ivor Tsang - [[pdf]](https://arxiv.org/pdf/2512.21354)</summary>

**Abstract:** Contemporary large language model (LLM) agents are remarkably capable, but they still lack reliable safety controls and can produce unconstrained, unpredictable, and even actively harmful outputs. To address this, we introduce Reflection-Driven Control, a standardized and pluggable control module that can be seamlessly integrated into general agent architectures. Reflection-Driven Control elevates "self-reflection" from a post hoc patch into an explicit step in the agent's own reasoning process: during generation, the agent continuously runs an internal reflection loop that monitors and evaluates its own decision path. When potential risks are detected, the system retrieves relevant repair examples and secure coding guidelines from an evolving reflective memory, injecting these evidence-based constraints directly into subsequent reasoning steps. We instantiate Reflection-Driven Control in the setting of secure code generation and systematically evaluate it across eight classes of security-critical programming tasks. Empirical results show that Reflection-Driven Control substantially improves the security and policy compliance of generated code while largely preserving functional correctness, with minimal runtime and token overhead. Taken together, these findings indicate that Reflection-Driven Control is a practical path toward trustworthy AI coding agents: it enables designs that are simultaneously autonomous, safer by construction, and auditable.

**arXiv ID:** 2512.21354
</details>

<details>
<summary><strong>HELP: Hierarchical Embodied Language Planner for Household Tasks</strong> - Alexandr V. Korchemnyi, Anatoly O. Onishchenko, Eva A. Bakaeva, Alexey K. Kovalev, Aleksandr I. Panov - [[pdf]](https://arxiv.org/pdf/2512.21723)</summary>

**Abstract:** Embodied agents tasked with complex scenarios, whether in real or simulated environments, rely heavily on robust planning capabilities. When instructions are formulated in natural language, large language models (LLMs) equipped with extensive linguistic knowledge can play this role. However, to effectively exploit the ability of such models to handle linguistic ambiguity, to retrieve information from the environment, and to be based on the available skills of an agent, an appropriate architecture must be designed. We propose a Hierarchical Embodied Language Planner, called HELP, consisting of a set of LLM-based agents, each dedicated to solving a different subtask. We evaluate the proposed approach on a household task and perform real-world experiments with an embodied agent. We also focus on the use of open source LLMs with a relatively small number of parameters, to enable autonomous deployment.

**arXiv ID:** 2512.21723
</details>

<details>
<summary><strong>Beyond Heuristics: A Decision-Theoretic Framework for Agent Memory Management</strong> - Changzhi Sun, Xiangyu Chen, Jixiang Luo, Dell Zhang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2512.21567)</summary>

**Abstract:** External memory is a key component of modern large language model (LLM) systems, enabling long-term interaction and personalization. Despite its importance, memory management is still largely driven by hand-designed heuristics, offering little insight into the long-term and uncertain consequences of memory decisions. In practice, choices about what to read or write shape future retrieval and downstream behavior in ways that are difficult to anticipate. We argue that memory management should be viewed as a sequential decision-making problem under uncertainty, where the utility of memory is delayed and dependent on future interactions. To this end, we propose DAM (Decision-theoretic Agent Memory), a decision-theoretic framework that decomposes memory management into immediate information access and hierarchical storage maintenance. Within this architecture, candidate operations are evaluated via value functions and uncertainty estimators, enabling an aggregate policy to arbitrate decisions based on estimated long-term utility and risk. Our contribution is not a new algorithm, but a principled reframing that clarifies the limitations of heuristic approaches and provides a foundation for future research on uncertainty-aware memory systems.

**arXiv ID:** 2512.21567
</details>

<details>
<summary><strong>Harnessing Data Spaces to Build Intelligent Smart City Infrastructures Across the Cloud-Edge Continuum</strong> - Dimitrios Amaxilatis, Themistoklis Sarantakos, Nikolaos Tsironis, Souvik Sengupta, Kostas Ramantas, Jhofre Ojeda - [[pdf]](https://arxiv.org/pdf/2512.21340)</summary>

**Abstract:** Smart cities are increasingly adopting data-centric architectures to enhance the efficiency, sustainability, and resilience of urban services.

**arXiv ID:** 2512.21340
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (17 papers)</h2></summary>

<details>
<summary><strong>EcoNet: Multiagent Planning and Control Of Household Energy Resources Using Active Inference</strong> - John C. Boik, Kobus Esterhuysen, Jacqueline B. Hynes, Axel Constant, Ines Hipolito, Mahault Albarracin, Alex B. Kiefer, Karl Friston - [[pdf]](https://arxiv.org/pdf/2512.21343)</summary>

**Abstract:** Advances in automated systems afford new opportunities for intelligent management of energy at household, local area, and utility scales. Home Energy Management Systems (HEMS) can play a role by optimizing the schedule and use of household energy devices and resources. One challenge is that the goals of a household can be complex and conflicting. For example, a household might wish to reduce energy costs and grid-associated greenhouse gas emissions, yet keep room temperatures comfortable. Another challenge is that an intelligent HEMS agent must make decisions under uncertainty. An agent must plan actions into the future, but weather and solar generation forecasts, for example, provide inherently uncertain estimates of future conditions. This paper introduces EcoNet, a Bayesian approach to household and neighborhood energy management that is based on active inference. The aim is to improve energy management and coordination, while accommodating uncertainties and taking into account potentially conditional and conflicting goals and preferences. Simulation results are presented and discussed.

**arXiv ID:** 2512.21343
</details>

<details>
<summary><strong>dUltra: Ultra-Fast Diffusion Language Models via Reinforcement Learning</strong> - Shirui Chen, Jiantao Jiao, Lillian J. Ratliff, Banghua Zhu - [[pdf]](https://arxiv.org/pdf/2512.21446)</summary>

**Abstract:** Masked diffusion language models (MDLMs) offer the potential for parallel token generation, but most open-source MDLMs decode fewer than 5 tokens per model forward pass even with sophisticated sampling strategies. As a result, their sampling speeds are often comparable to AR + speculative decoding schemes, limiting their advantage over mainstream autoregressive approaches. Existing distillation-based accelerators (dParallel, d3LLM) finetune MDLMs on trajectories generated by a base model, which can become off-policy during finetuning and restrict performance to the quality of the base model's samples. We propose \texttt{dUltra}, an on-policy reinforcement learning framework based on Group Relative Policy Optimization (GRPO) that learns unmasking strategies for efficient parallel decoding. dUltra introduces an unmasking planner head that predicts per-token unmasking likelihoods under independent Bernoulli distributions. We jointly optimize the base diffusion LLM and the unmasking order planner using reward signals combining verifiable reward, distillation reward, and the number of unmasking steps. Across mathematical reasoning and code generation tasks, dUltra improves the accuracy--efficiency trade-off over state-of-the-art heuristic and distillation baselines, moving towards achieving ``diffusion supremacy'' over autoregressive models.

**arXiv ID:** 2512.21446
</details>

<details>
<summary><strong>How Do Agents Perform Code Optimization? An Empirical Study</strong> - Huiyun Peng, Antonio Zhong, Ricardo Andrés Calvo Méndez, Kelechi G. Kalu, James C. Davis - [[pdf]](https://arxiv.org/pdf/2512.21757)</summary>

**Abstract:** Performance optimization is a critical yet challenging aspect of software development, often requiring a deep understanding of system behavior, algorithmic tradeoffs, and careful code modifications. Although recent advances in AI coding agents have accelerated code generation and bug fixing, little is known about how these agents perform on real-world performance optimization tasks. We present the first empirical study comparing agent- and human-authored performance optimization commits, analyzing 324 agent-generated and 83 human-authored PRs from the AIDev dataset across adoption, maintainability, optimization patterns, and validation practices. We find that AI-authored performance PRs are less likely to include explicit performance validation than human-authored PRs (45.7\% vs. 63.6\%, $p=0.007$). In addition, AI-authored PRs largely use the same optimization patterns as humans. We further discuss limitations and opportunities for advancing agentic code optimization.

**arXiv ID:** 2512.21757
</details>

<details>
<summary><strong>ForestProtector: An IoT Architecture Integrating Machine Vision and Deep Reinforcement Learning for Efficient Wildfire Monitoring</strong> - Kenneth Bonilla-Ormachea, Horacio Cuizaga, Edwin Salcedo, Sebastian Castro, Sergio Fernandez-Testa, Misael Mamani - [[pdf]](https://arxiv.org/pdf/2501.09926)</summary>

**Abstract:** Early detection of forest fires is crucial to minimizing the environmental and socioeconomic damage they cause. Indeed, a fire's duration directly correlates with the difficulty and cost of extinguishing it. For instance, a fire burning for 1 minute might require 1 liter of water to extinguish, while a 2-minute fire could demand 100 liters, and a 10-minute fire might necessitate 1,000 liters. On the other hand, existing fire detection systems based on novel technologies (e.g., remote sensing, PTZ cameras, UAVs) are often expensive and require human intervention, making continuous monitoring of large areas impractical. To address this challenge, this work proposes a low-cost forest fire detection system that utilizes a central gateway device with computer vision capabilities to monitor a 360° field of view for smoke at long distances. A deep reinforcement learning agent enhances surveillance by dynamically controlling the camera's orientation, leveraging real-time sensor data (smoke levels, ambient temperature, and humidity) from distributed IoT devices. This approach enables automated wildfire monitoring across expansive areas while reducing false positives.

**arXiv ID:** 2501.09926
</details>

<details>
<summary><strong>Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning</strong> - Qihao Liu, Luoxin Ye, Wufei Ma, Yu-Cheng Chou, Alan Yuille - [[pdf]](https://arxiv.org/pdf/2512.16917)</summary>

**Abstract:** Large language models (LLMs) with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors, such as incorrect calculations, brittle logic, and superficially plausible but invalid steps. In this paper, we introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality of LLMs. Across various mathematical benchmarks, the method delivers consistent gains over strong baselines with standard RL post-training. Specifically, on AIME24, we improve DeepSeek-R1-Distill-Qwen-7B from 54.0 to 61.3 (+7.3) and DeepSeek-R1-Distill-Llama-8B from 43.7 to 53.7 (+10.0). The modular discriminator also enables flexible reward shaping for objectives such as teacher distillation, preference alignment, and mathematical proof-based reasoning.

**arXiv ID:** 2512.16917
</details>

<details>
<summary><strong>Thinking Forward and Backward: Multi-Objective Reinforcement Learning for Retrieval-Augmented Reasoning</strong> - Wenda Wei, Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Lixin Su, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, Xueqi Cheng - [[pdf]](https://arxiv.org/pdf/2511.09109)</summary>

**Abstract:** Retrieval-augmented generation (RAG) has proven to be effective in mitigating hallucinations in large language models, yet its effectiveness remains limited in complex, multi-step reasoning scenarios. Recent efforts have incorporated search-based interactions into RAG, enabling iterative reasoning with real-time retrieval. Most approaches rely on outcome-based supervision, offering no explicit guidance for intermediate steps. This often leads to reward hacking and degraded response quality. We propose Bi-RAR, a novel retrieval-augmented reasoning framework that evaluates each intermediate step jointly in both forward and backward directions. To assess the information completeness of each step, we introduce a bidirectional information distance grounded in Kolmogorov complexity, approximated via language model generation probabilities. This quantification measures both how far the current reasoning is from the answer and how well it addresses the question. To optimize reasoning under these bidirectional signals, we adopt a multi-objective reinforcement learning framework with a cascading reward structure that emphasizes early trajectory alignment. Empirical results on seven question answering benchmarks demonstrate that Bi-RAR surpasses previous methods and enables efficient interaction and reasoning with the search engine during training and inference.

**arXiv ID:** 2511.09109
</details>

<details>
<summary><strong>Object-Centric World Models for Causality-Aware Reinforcement Learning</strong> - Yosuke Nishimoto, Takashi Matsubara - [[pdf]](https://arxiv.org/pdf/2511.14262)</summary>

**Abstract:** World models have been developed to support sample-efficient deep reinforcement learning agents. However, it remains challenging for world models to accurately replicate environments that are high-dimensional, non-stationary, and composed of multiple objects with rich interactions since most world models learn holistic representations of all environmental components. By contrast, humans perceive the environment by decomposing it into discrete objects, facilitating efficient decision-making. Motivated by this insight, we propose \emph{Slot Transformer Imagination with CAusality-aware reinforcement learning} (STICA), a unified framework in which object-centric Transformers serve as the world model and causality-aware policy and value networks. STICA represents each observation as a set of object-centric tokens, together with tokens for the agent action and the resulting reward, enabling the world model to predict token-level dynamics and interactions. The policy and value networks then estimate token-level cause--effect relations and use them in the attention layers, yielding causality-guided decision-making. Experiments on object-rich benchmarks demonstrate that STICA consistently outperforms state-of-the-art agents in both sample efficiency and final performance.

**arXiv ID:** 2511.14262
</details>

<details>
<summary><strong>Periodic Asynchrony: An Effective Method for Accelerating Reinforcement Learning for Large Language Models</strong> - Jian Lu - [[pdf]](https://arxiv.org/pdf/2511.18871)</summary>

**Abstract:** Since the introduction of the GRPO algorithm, reinforcement learning (RL) has attracted increasing attention, with growing efforts to reproduce and apply it. However, training efficiency remains a critical challenge. In mainstream RL frameworks, inference and training are typically deployed on the same devices. While this approach reduces costs through resource consolidation, its synchronous execution imposes a computational coupling that prevents concurrent inference and training. In this study, we are returning to the strategy of separating inference and training deployment, and by introducing improvements in the data loader, we transform the conventional synchronous architecture into a periodically asynchronous framework, which allows for demand-driven, independent, and elastic scaling of each component, while the accuracy of the algorithm remains completely equivalent to the synchronization method, with both belonging to the on-policy strategy. It is worth emphasizing that we apply a unified tri-model architecture in the training phase, and we also proposed a shared-prompt attention mask to reduce repetitive computation. In practice, these works have achieved at least a threefold overall performance improvement in RL training on NPU platforms, indicating its potential for widespread application.

**arXiv ID:** 2511.18871
</details>

<details>
<summary><strong>Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning</strong> - Jiayun Wu, Jiashuo Liu, Zhiyuan Zeng, Tianyang Zhan, Tianle Cai, Wenhao Huang - [[pdf]](https://arxiv.org/pdf/2512.19920)</summary>

**Abstract:** LLM deployment in critical domains is currently impeded by persistent hallucinations--generating plausible but factually incorrect assertions. While scaling laws drove significant improvements in general capabilities, theoretical frameworks suggest hallucination is not merely stochastic error but a predictable statistical consequence of training objectives prioritizing mimicking data distribution over epistemic honesty. Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero. This paper presents an exhaustive investigation into behavioral calibration, which incentivizes models to stochastically admit uncertainty by abstaining when not confident, aligning model behavior with accuracy. Synthesizing recent advances, we propose and evaluate training interventions optimizing strictly proper scoring rules for models to output a calibrated probability of correctness. Our methods enable models to either abstain from producing a complete response or flag individual claims where uncertainty remains. Utilizing Qwen3-4B-Instruct, empirical analysis reveals behavior-calibrated reinforcement learning allows smaller models to surpass frontier models in uncertainty quantification--a transferable meta-skill decouplable from raw predictive accuracy. Trained on math reasoning tasks, our model's log-scale Accuracy-to-Hallucination Ratio gain (0.806) exceeds GPT-5's (0.207) in a challenging in-domain evaluation (BeyondAIME). Moreover, in cross-domain factual QA (SimpleQA), our 4B LLM achieves zero-shot calibration error on par with frontier models including Grok-4 and Gemini-2.5-Pro, even though its factual accuracy is much lower.

**arXiv ID:** 2512.19920
</details>

<details>
<summary><strong>TableGPT-R1: Advancing Tabular Reasoning Through Reinforcement Learning</strong> - Saisai Yang, Qingyi Huang, Jing Yuan, Liangyu Zha, Kai Tang, Yuhang Yang, Ning Wang, Yucheng Wei, Liyao Li, Wentao Ye, Hao Chen, Tao Zhang, Junlin Zhou, Haobo Wang, Gang Chen, Junbo Zhao - [[pdf]](https://arxiv.org/pdf/2512.20312)</summary>

**Abstract:** Tabular data serves as the backbone of modern data analysis and scientific research. While Large Language Models (LLMs) fine-tuned via Supervised Fine-Tuning (SFT) have significantly improved natural language interaction with such structured data, they often fall short in handling the complex, multi-step reasoning and robust code execution required for real-world table tasks. Reinforcement Learning (RL) offers a promising avenue to enhance these capabilities, yet its application in the tabular domain faces three critical hurdles: the scarcity of high-quality agentic trajectories with closed-loop code execution and environment feedback on diverse table structures, the extreme heterogeneity of feedback signals ranging from rigid SQL execution to open-ended data interpretation, and the risk of catastrophic forgetting of general knowledge during vertical specialization. To overcome these challenges and unlock advanced reasoning on complex tables, we introduce \textbf{TableGPT-R1}, a specialized tabular model built on a systematic RL framework. Our approach integrates a comprehensive data engineering pipeline that synthesizes difficulty-stratified agentic trajectories for both supervised alignment and RL rollouts, a task-adaptive reward system that combines rule-based verification with a criteria-injected reward model and incorporates process-level step reward shaping with behavioral regularization, and a multi-stage training framework that progressively stabilizes reasoning before specializing in table-specific tasks. Extensive evaluations demonstrate that TableGPT-R1 achieves state-of-the-art performance on authoritative benchmarks, significantly outperforming baseline models while retaining robust general capabilities. Our model is available at this https URL.

**arXiv ID:** 2512.20312
</details>

<details>
<summary><strong>Rethinking Sample Polarity in Reinforcement Learning with Verifiable Rewards</strong> - Xinyu Tang, Yuliang Zhan, Zhixun Li, Wayne Xin Zhao, Zhenduo Zhang, Zujie Wen, Zhiqiang Zhang, Jun Zhou - [[pdf]](https://arxiv.org/pdf/2512.21625)</summary>

**Abstract:** Large reasoning models (LRMs) are typically trained using reinforcement learning with verifiable reward (RLVR) to enhance their reasoning abilities. In this paradigm, policies are updated using both positive and negative self-generated rollouts, which correspond to distinct sample polarities. In this paper, we provide a systematic investigation into how these sample polarities affect RLVR training dynamics and behaviors. We find that positive samples sharpen existing correct reasoning patterns, while negative samples encourage exploration of new reasoning paths. We further explore how adjusting the advantage values of positive and negative samples at both the sample level and the token level affects RLVR training. Based on these insights, we propose an Adaptive and Asymmetric token-level Advantage shaping method for Policy Optimization, namely A3PO, that more precisely allocates advantage signals to key tokens across different polarities. Experiments across five reasoning benchmarks demonstrate the effectiveness of our approach.

**arXiv ID:** 2512.21625
</details>

<details>
<summary><strong>SWE-RM: Execution-free Feedback For Software Engineering Agents</strong> - KaShun Shum, Binyuan Hui, Jiawei Chen, Lei Zhang, X. W., Jiaxi Yang, Yuzhen Huang, Junyang Lin, Junxian He - [[pdf]](https://arxiv.org/pdf/2512.21919)</summary>

**Abstract:** Execution-based feedback like unit testing is widely used in the development of coding agents through test-time scaling (TTS) and reinforcement learning (RL). This paradigm requires scalable and reliable collection of unit test cases to provide accurate feedback, and the resulting feedback is often sparse and cannot effectively distinguish between trajectories that are both successful or both unsuccessful. In contrast, execution-free feedback from reward models can provide more fine-grained signals without depending on unit test cases. Despite this potential, execution-free feedback for realistic software engineering (SWE) agents remains underexplored. Aiming to develop versatile reward models that are effective across TTS and RL, however, we observe that two verifiers with nearly identical TTS performance can nevertheless yield very different results in RL. Intuitively, TTS primarily reflects the model's ability to select the best trajectory, but this ability does not necessarily generalize to RL. To address this limitation, we identify two additional aspects that are crucial for RL training: classification accuracy and calibration. We then conduct comprehensive controlled experiments to investigate how to train a robust reward model that performs well across these metrics. In particular, we analyze the impact of various factors such as training data scale, policy mixtures, and data source composition. Guided by these investigations, we introduce SWE-RM, an accurate and robust reward model adopting a mixture-of-experts architecture with 30B total parameters and 3B activated during inference. SWE-RM substantially improves SWE agents on both TTS and RL performance. For example, it increases the accuracy of Qwen3-Coder-Flash from 51.6% to 62.0%, and Qwen3-Coder-Max from 67.0% to 74.6% on SWE-Bench Verified using TTS, achieving new state-of-the-art performance among open-source models.

**arXiv ID:** 2512.21919
</details>

<details>
<summary><strong>Context as a Tool: Context Management for Long-Horizon SWE-Agents</strong> - Shukai Liu, Jian Yang, Bo Jiang, Yizhi Li, Jinyang Guo, Xianglong Liu, Bryan Dai - [[pdf]](https://arxiv.org/pdf/2512.22087)</summary>

**Abstract:** Agents based on large language models have recently shown strong potential on real-world software engineering (SWE) tasks that require long-horizon interaction with repository-scale codebases. However, most existing agents rely on append-only context maintenance or passively triggered compression heuristics, which often lead to context explosion, semantic drift, and degraded reasoning in long-running interactions. We propose CAT, a new context management paradigm that elevates context maintenance to a callable tool integrated into the decision-making process of agents. CAT formalizes a structured context workspace consisting of stable task semantics, condensed long-term memory, and high-fidelity short-term interactions, and enables agents to proactively compress historical trajectories into actionable summaries at appropriate milestones. To support context management for SWE-agents, we propose a trajectory-level supervision framework, CAT-GENERATOR, based on an offline data construction pipeline that injects context-management actions into complete interaction trajectories. Using this framework, we train a context-aware model, SWE-Compressor. Experiments on SWE-Bench-Verified demonstrate that SWE-Compressor reaches a 57.6% solved rate and significantly outperforms ReAct-based agents and static compression baselines, while maintaining stable and scalable long-horizon reasoning under a bounded context budget.

**arXiv ID:** 2512.22087
</details>

<details>
<summary><strong>MobileWorld: Benchmarking Autonomous Mobile Agents in Agent-User Interactive and MCP-Augmented Environments</strong> - Quyu Kong, Xu Zhang, Zhenyu Yang, Nolan Gao, Chen Liu, Panrong Tong, Chenglin Cai, Hanzhang Zhou, Jianan Zhang, Liangyu Chen, Zhidan Liu, Steven Hoi, Yue Wang - [[pdf]](https://arxiv.org/pdf/2512.19432)</summary>

**Abstract:** Among existing online mobile-use benchmarks, AndroidWorld has emerged as the dominant benchmark due to its reproducible environment and deterministic evaluation; however, recent agents achieving over 90% success rates indicate its saturation and motivate the need for a more challenging benchmark. In addition, its environment lacks key application categories, such as e-commerce and enterprise communication, and does not reflect realistic mobile-use scenarios characterized by vague user instructions and hybrid tool usage. We introduce MobileWorld, a substantially more challenging benchmark designed to reflect real-world usage through 201 tasks across 20 applications. MobileWorld derives its difficulty from an emphasis on long-horizon, cross-application workflows, requiring nearly twice as many completion steps on average (27.8 vs. 14.3) and featuring a significantly higher proportion of multi-app tasks (62.2% vs. 9.5%) than AndroidWorld. To overcome the limitations of existing environments, MobileWorld achieves a balance between production-grade utility and reproducible evaluation by utilizing open-source alternatives to industry standards (e.g., Mattermost for Slack). This approach enables a fully observable and controlled environment through source code modification and direct backend database access for precise verification. MobileWorld also introduces novel task categories, including agent-user interaction and Model Context Protocol (MCP)-augmented tasks, for evaluating agents in user-aware, hybrid-tool scenarios. To facilitate evaluation, we develop a planner-executor agentic framework with extended action spaces to support user interactions and MCP calls. Our results reveal a sharp performance drop compared to AndroidWorld, with the best agentic framework and end-to-end model achieving 51.7% and 20.9% success rates, respectively, highlighting ample headroom for future research.

**arXiv ID:** 2512.19432
</details>

<details>
<summary><strong>A Reinforcement Learning Approach to Synthetic Data Generation</strong> - Natalia Espinosa-Dice, Nicholas J. Jackson, Chao Yan, Aaron Lee, Bradley A. Malin - [[pdf]](https://arxiv.org/pdf/2512.21395)</summary>

**Abstract:** Synthetic data generation (SDG) is a promising approach for enabling data sharing in biomedical studies while preserving patient privacy. Yet, state-of-the-art generative models often require large datasets and complex training procedures, limiting their applicability in small-sample settings. In this work, we reframe SDG as a reinforcement learning (RL) problem and introduce RLSyn, a novel framework that models the data generator as a stochastic policy over patient records and optimizes it using Proximal Policy Optimization with discriminator-derived rewards, yielding more stable and data-efficient training. We evaluate RLSyn on two biomedical datasets - AI-READI and MIMIC-IV- and benchmark it against state-of-the-art generative adversarial networks (GANs) and diffusion-based methods across extensive privacy, utility, and fidelity evaluations. RL-Syn performs comparably to diffusion models and outperforms GANs on MIMIC-IV, while outperforming both diffusion models and GANs on the smaller AI-READI dataset. These results demonstrate that reinforcement learning provides a principled and effective alternative for synthetic biomedical data generation, particularly in data-scarce regimes.

**arXiv ID:** 2512.21395
</details>

<details>
<summary><strong>PlanScope: Learning to Plan Within Decision Scope for Urban Autonomous Driving</strong> - Ren Xin, Jie Cheng, Hongji Liu, Jun Ma - [[pdf]](https://arxiv.org/pdf/2411.00476)</summary>

**Abstract:** In the context of urban autonomous driving, imitation learning-based methods have shown remarkable effectiveness, with a typical practice to minimize the discrepancy between expert driving logs and predictive decision sequences. As expert driving logs natively contain future short-term decisions with respect to events, such as sudden obstacles or rapidly changing traffic signals. We believe that unpredictable future events and corresponding expert reactions can introduce reasoning disturbances, negatively affecting the convergence efficiency of planning models. At the same time, long-term decision information, such as maintaining a reference lane or avoiding stationary obstacles, is essential for guiding short-term decisions. Our preliminary experiments on shortening the planning horizon show a rise-and-fall trend in driving performance, supporting these hypotheses. Based on these insights, we present PlanScope, a sequential-decision-learning framework with novel techniques for separating short-term and long-term decisions in decision logs. To identify and extract each decision component, the Wavelet Transform on trajectory profiles is proposed. After that, to enhance the detail-generating ability of Neural Networks, extra Detail Decoders are proposed. Finally, to enable in-scope decision supervision across detail levels, Multi-Scope Supervision strategies are adopted during training. The proposed methods, especially the time-dependent normalization, outperform baseline models in closed-loop evaluations on the nuPlan dataset, offering a plug-and-play solution to enhance existing planning models.

**arXiv ID:** 2411.00476
</details>

<details>
<summary><strong>Think, Act, Learn: A Framework for Autonomous Robotic Agents using Closed-Loop Large Language Models</strong> - Anjali R. Menon, Rohit K. Sharma, Priya Singh, Chengyu Wang, Aurora M. Ferreira, Mateja Novak - [[pdf]](https://arxiv.org/pdf/2507.19854)</summary>

**Abstract:** The integration of Large Language Models (LLMs) into robotics has unlocked unprecedented capabilities in high-level task planning. However, most current systems operate in an open-loop fashion, where LLMs act as one-shot planners, rendering them brittle and unable to adapt to unforeseen circumstances in dynamic physical environments. To overcome this limitation, this paper introduces the "Think, Act, Learn" (T-A-L) framework, a novel architecture that enables an embodied agent to autonomously learn and refine its policies through continuous interaction. Our framework establishes a closed-loop cycle where an LLM first "thinks" by decomposing high-level commands into actionable plans. The robot then "acts" by executing these plans while gathering rich, multimodal sensory feedback. Critically, the "learn" module processes this feedback to facilitate LLM-driven self-reflection, allowing the agent to perform causal analysis on its failures and generate corrective strategies. These insights are stored in an experiential memory to guide future planning cycles. We demonstrate through extensive experiments in both simulation and the real world that our T-A-L agent significantly outperforms baseline methods, including open-loop LLMs, Behavioral Cloning, and traditional Reinforcement Learning. Our framework achieves over a 97% success rate on complex, long-horizon tasks, converges to a stable policy in an average of just 9 trials, and exhibits remarkable generalization to unseen tasks. This work presents a significant step towards developing more robust, adaptive, and truly autonomous robotic agents.

**arXiv ID:** 2507.19854
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
