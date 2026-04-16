# Agent arXiv Daily

**Last Updated:** 2026-04-16 04:23:38

**Total Papers:** 100

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (10 papers)</h2></summary>

<details>
<summary><strong>SciFi: A Safe, Lightweight, User-Friendly, and Fully Autonomous Agentic AI Workflow for Scientific Applications</strong> - Qibin Liu, Julia Gonski - [[pdf]](https://arxiv.org/pdf/2604.13180)</summary>

**Abstract:** Recent advances in agentic AI have enabled increasingly autonomous workflows, but existing systems still face substantial challenges in achieving reliable deployment in real-world scientific research. In this work, we present a safe, lightweight, and user-friendly agentic framework for the autonomous execution of well-defined scientific tasks. The framework combines an isolated execution environment, a three-layer agent loop, and a self-assessing do-until mechanism to ensure safe and reliable operation while effectively leveraging large language models of varying capability levels. By focusing on structured tasks with clearly defined context and stopping criteria, the framework supports end-to-end automation with minimal human intervention, enabling researchers to offload routine workloads and devote more effort to creative activities and open-ended scientific inquiry.

**arXiv ID:** 2604.13180
</details>

<details>
<summary><strong>Rethinking AI Hardware: A Three-Layer Cognitive Architecture for Autonomous Agents</strong> - Li Chen - [[pdf]](https://arxiv.org/pdf/2604.13757)</summary>

**Abstract:** The next generation of autonomous AI systems will be constrained not only by model capability, but by how intelligence is structured across heterogeneous hardware. Current paradigms -- cloud-centric AI, on-device inference, and edge-cloud pipelines -- treat planning, reasoning, and execution as a monolithic process, leading to unnecessary latency, energy consumption, and fragmented behavioral continuity. We introduce the Tri-Spirit Architecture, a three-layer cognitive framework that decomposes intelligence into planning (Super Layer), reasoning (Agent Layer), and execution (Reflex Layer), each mapped to distinct compute substrates and coordinated via an asynchronous message bus. We formalize the system with a parameterized routing policy, a habit-compilation mechanism that promotes repeated reasoning paths into zero-inference execution policies, a convergent memory model, and explicit safety constraints. We evaluate the architecture in a reproducible simulation of 2000 synthetic tasks against cloud-centric and edge-only baselines. Tri-Spirit reduces mean task latency by 75.6 percent and energy consumption by 71.1 percent, while decreasing LLM invocations by 30 percent and enabling 77.6 percent offline task completion. These results suggest that cognitive decomposition, rather than model scaling alone, is a primary driver of system-level efficiency in AI hardware.

**arXiv ID:** 2604.13757
</details>

<details>
<summary><strong>Pareto-Optimal Offline Reinforcement Learning via Smooth Tchebysheff Scalarization</strong> - Aadyot Bhatnagar, Peter Mørch Groth, Ali Madani - [[pdf]](https://arxiv.org/pdf/2604.13175)</summary>

**Abstract:** Large language models can be aligned with human preferences through offline reinforcement learning (RL) on small labeled datasets. While single-objective alignment is well-studied, many real-world applications demand the simultaneous optimization of multiple conflicting rewards, e.g. optimizing both catalytic activity and specificity in protein engineering, or helpfulness and harmlessness for chatbots. Prior work has largely relied on linear reward scalarization, but this approach provably fails to recover non-convex regions of the Pareto front. In this paper, instead of scalarizing the rewards directly, we frame multi-objective RL itself as an optimization problem to be scalarized via smooth Tchebysheff scalarization, a recent technique that overcomes the shortcomings of linear scalarization. We use this formulation to derive Smooth Tchebysheff Optimization of Multi-Objective Preferences (STOMP), a novel offline RL algorithm that extends direct preference optimization to the multi-objective setting in a principled way by standardizing the individual rewards based on their observed distributions. We empirically validate STOMP on a range of protein engineering tasks by aligning three autoregressive protein language models on three laboratory datasets of protein fitness. Compared to state-of-the-art baselines, STOMP achieves the highest hypervolumes in eight of nine settings according to both offline off-policy and generative evaluations. We thus demonstrate that STOMP is a powerful, robust multi-objective alignment algorithm that can meaningfully improve post-trained models for multi-attribute protein optimization and beyond.

**arXiv ID:** 2604.13175
</details>

<details>
<summary><strong>Formalizing the Safety, Security, and Functional Properties of Agentic AI Systems</strong> - Edoardo Allegrini, Ananth Shreekumar, Z. Berkay Celik - [[pdf]](https://arxiv.org/pdf/2510.14133)</summary>

**Abstract:** Agentic AI systems, which leverage multiple autonomous agents and large language models (LLMs), are increasingly used to address complex, multi-step tasks. The safety, security, and functionality of these systems are critical, especially in high-stakes applications. However, the current ecosystem of inter-agent communication is fragmented, with protocols such as the Model Context Protocol (MCP) for tool access and the Agent-to-Agent (A2A) protocol for coordination being analyzed in isolation. This fragmentation creates a semantic gap that prevents the rigorous analysis of system properties and introduces risks such as architectural misalignment and exploitable coordination issues. To address these challenges, we introduce a modeling framework for agentic AI systems composed of two central models: (1) the host agent model formalizes the top-level entity that interacts with the user, decomposes tasks, and orchestrates their execution by leveraging external agents and tools; (2) the task lifecycle model details the states and transitions of individual sub-tasks from creation to completion, providing a fine-grained view of task management and error handling. Together, these models provide a unified semantic framework for reasoning about the behavior of multi-AI agent systems. Grounded in this framework, we define 16 properties for the host agent and 14 for the task lifecycle, categorized into liveness, safety, completeness, and fairness. Expressed in temporal logic, these properties enable formal verification of system behavior, detection of coordination edge cases, and prevention of deadlocks and security vulnerabilities. Through this effort, we introduce the first rigorously grounded, domain-agnostic framework for the analysis, design, and deployment of correct, reliable, and robust agentic AI systems.

**arXiv ID:** 2510.14133
</details>

<details>
<summary><strong>Deep Learning Based Amharic Chatbot for FAQs in Universities</strong> - Goitom Ybrah Hailu, Hadush Hailu, Shishay Welay - [[pdf]](https://arxiv.org/pdf/2402.01720)</summary>

**Abstract:** University students often spend a considerable amount of time seeking answers to common questions from administrators or teachers. This can become tedious for both parties, leading to a need for a solution. In response, this paper proposes a chatbot model that utilizes natural language processing and deep learning techniques to answer frequently asked questions (FAQs) in the Amharic language. Chatbots are computer programs that simulate human conversation through the use of artificial intelligence (AI), acting as a virtual assistant to handle questions and other tasks. The proposed chatbot program employs tokenization, normalization, stop word removal, and stemming to analyze and categorize Amharic input sentences. Three machine learning model algorithms were used to classify tokens and retrieve appropriate responses: Support Vector Machine (SVM), Multinomial Naïve Bayes, and deep neural networks implemented through TensorFlow, Keras, and NLTK. The deep learning model achieved the best results with 91.55% accuracy and a validation loss of 0.3548 using an Adam optimizer and SoftMax activation function. The chatbot model was integrated with Facebook Messenger and deployed on a Heroku server for 24-hour accessibility. The experimental results demonstrate that the chatbot framework achieved its objectives and effectively addressed challenges such as Amharic Fidel variation, morphological variation, and lexical gaps. Future research could explore the integration of Amharic WordNet to narrow the lexical gap and support more complex questions.

**arXiv ID:** 2402.01720
</details>

<details>
<summary><strong>Online Navigation Planning for Long-term Autonomous Operation of Underwater Gliders</strong> - Victor-Alexandru Darvariu, Charlotte Z. Reed, Jan Stratmann, Bruno Lacerda, Benjamin Allsup, Stephen Woodward, Elizabeth Siddle, Trishna Saeharaseelan, Owain Jones, Dan Jones, Tobias Ferreira, Chloe Baker, Kevin Chaplin, James Kirk, Ashley Iceton-Morris, Ryan D. Patmore, Jeff Polton, Charlotte Williams, Christopher D. J. Auckland, Rob A. Hall, Alexandra Kokkinaki, Alvaro Lorenzo Lopez, Justin J. H. Buck, Nick Hawes - [[pdf]](https://arxiv.org/pdf/2602.19315)</summary>

**Abstract:** Underwater glider robots have become indispensable for ocean sampling, yet fully autonomous long-term operation remains rare in practice. Although stakeholders are calling for tools to manage increasingly large fleets of gliders, existing methods have seen limited adoption due to their inability to account for environmental uncertainty and operational constraints. In this work, we demonstrate that uncertainty-aware online navigation planning can be deployed in real-world glider missions at scale. We formulate the problem as a stochastic shortest-path Markov Decision Process and propose a sample-based online planner based on Monte Carlo Tree Search. Samples are generated by a physics-informed simulator calibrated on real-world glider data that captures uncertain execution of controls and ocean current forecasts while remaining computationally tractable. Our methodology is integrated into an autonomous system for Slocum gliders that performs closed-loop replanning at each surfacing. The system was validated in two North Sea deployments totalling approximately 3 months and 1000 km, representing the longest fully autonomous glider campaigns in the literature to date. Results demonstrate improvements of up to 9.88% in dive duration and 16.51% in path length compared to standard straight-to-goal navigation, including a statistically significant path length reduction of 9.55% in a field deployment.

**arXiv ID:** 2602.19315
</details>

<details>
<summary><strong>Assessment Design in the AI Era: A Method for Identifying Items Functioning Differentially for Humans and Chatbots</strong> - Licol Zeinfeld, Alona Strugatski, Ziva Bar-Dov, Ron Blonder, Shelley Rap, Giora Alexandron - [[pdf]](https://arxiv.org/pdf/2603.23682)</summary>

**Abstract:** The rapid adoption of large language models (LLMs) in education raises profound challenges for assessment design. To adapt assessments to the presence of LLM-based tools, it is crucial to characterize the strengths and weaknesses of LLMs in a generalizable, valid and reliable manner. However, current LLM evaluations often rely on descriptive statistics derived from benchmarks, and little research applies theory-grounded measurement methods to characterize LLM capabilities relative to human learners in ways that directly support assessment design. Here, by combining educational data mining and psychometric theory, we introduce a statistically principled approach for identifying items on which humans and LLMs show systematic response differences, pinpointing where assessments may be most vulnerable to AI misuse, and which task dimensions make problems particularly easy or difficult for generative AI. The method is based on Differential Item Functioning (DIF) analysis -- traditionally used to detect bias across demographic groups -- together with negative control analysis and item-total correlation discrimination analysis. It is evaluated on responses from human learners and six leading chatbots (ChatGPT-4o \& 5.2, Gemini 1.5 \& 3 Pro, Claude 3.5 \& 4.5 Sonnet) to two instruments: a high school chemistry diagnostic test and a university entrance exam. Subject-matter experts then analyzed DIF-flagged items to characterize task dimensions associated with chatbot over- or under-performance. Results show that DIF-informed analytics provide a robust framework for understanding where LLM and human capabilities diverge, and highlight their value for improving the design of valid, reliable, and fair assessment in the AI era.

**arXiv ID:** 2603.23682
</details>

<details>
<summary><strong>Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning</strong> - Fengran Mo, Yifan Gao, Sha Li, Hansi Zeng, Xin Liu, Zhaoxuan Tan, Xian Li, Jianshu Chen, Dakuo Wang, Meng Jiang - [[pdf]](https://arxiv.org/pdf/2601.13115)</summary>

**Abstract:** Large Language Models (LLMs) have become a popular interface for human-AI interaction, supporting information seeking and task assistance through natural, multi-turn dialogue. To respond to users within multi-turn dialogues, the context-dependent user intent evolves across interactions, requiring contextual interpretation, query reformulation, and dynamic coordination between retrieval and generation. Existing studies usually follow static rewrite, retrieve, and generate pipelines, which optimize different procedures separately and overlook the mixed-initiative action optimization simultaneously. Although the recent developments in deep search agents demonstrate the effectiveness in jointly optimizing retrieval and generation via reasoning, these approaches focus on single-turn scenarios, which might lack the ability to handle multi-turn interactions. We introduce a conversational agent that interleaves search and reasoning across turns, enabling exploratory and adaptive behaviors learned through reinforcement learning (RL) training with tailored rewards towards evolving user goals. The experimental results across four widely used conversational benchmarks demonstrate the effectiveness of our methods by surpassing several existing strong baselines.

**arXiv ID:** 2601.13115
</details>

<details>
<summary><strong>EmbodiedClaw: Conversational Workflow Execution for Embodied AI Development</strong> - Xueyang Zhou, Yihan Sun, Xijie Gong, Guiyao Tie, Pan Zhou, Lichao Sun, Yongchao Chen - [[pdf]](https://arxiv.org/pdf/2604.13800)</summary>

**Abstract:** Embodied AI research is increasingly moving beyond single-task, single-environment policy learning toward multi-task, multi-scene, and multi-model settings. This shift substantially increases the engineering overhead and development time required for stages such as evaluation environment construction, trajectory collection, model training, and evaluation. To address this challenge, we propose a new paradigm for embodied AI development in which users express goals and constraints through conversation, and the system automatically plans and executes the development workflow. We instantiate this paradigm with EmbodiedClaw, a conversational agent that turns high-frequency, high-cost embodied research activities, including environment creation and revision, benchmark transformation, trajectory synthesis, model evaluation, and asset expansion, into executable skills. Experiments on end-to-end workflow tasks, capability-specific evaluations, human researcher studies, and ablations show that EmbodiedClaw reduces manual engineering effort while improving executability, consistency, and reproducibility. These results suggest a shift from manual toolchains to conversationally executable workflows for embodied AI development.

**arXiv ID:** 2604.13800
</details>

<details>
<summary><strong>Ghosting the Machine: Stop Calling Human-Agent Relations Parasocial</strong> - Jaime Banks - [[pdf]](https://arxiv.org/pdf/2604.05197)</summary>

**Abstract:** In discussions of human relations with conversational agents (CAs; e.g., voice assistants, AI companions, some social robots), they are increasingly referred to as parasocial. This is a misapplication of the term, heuristically taken up to mean "unreal." In this provocation, I briefly account for the theoretical trajectory of parasociality and detail why it is inaccurate to apply the notion to human interactions with CAs. In short, "parasocial" refers to a human-character relations that are one-sided, non-dialectical, character-governed, imagined, vicarious, predictable, and low-effort; the term has been co-opted to instead refer to relations that are seen as unreal or invalid. The scientific problematics of this misapplication are nontrivial. They lead to oversimplification of complex phenomena, misspecified variables and misdiagnosed effects, and devaluation of human experiences. Those challenges, in turn, have downstream effects on norms and practice. It is scientifically, practically, and ethically imperative to recognize the sociality of human-agent relations.

**arXiv ID:** 2604.05197
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (17 papers)</h2></summary>

<details>
<summary><strong>GeoAgentBench: A Dynamic Execution Benchmark for Tool-Augmented Agents in Spatial Analysis</strong> - Bo Yu, Cheng Yang, Dongyang Hou, Chengfu Liu, Jiayao Liu, Chi Wang, Zhiming Zhang, Haifeng Li, Wentao Yang - [[pdf]](https://arxiv.org/pdf/2604.13888)</summary>

**Abstract:** The integration of Large Language Models (LLMs) into Geographic Information Systems (GIS) marks a paradigm shift toward autonomous spatial analysis. However, evaluating these LLM-based agents remains challenging due to the complex, multi-step nature of geospatial workflows. Existing benchmarks primarily rely on static text or code matching, neglecting dynamic runtime feedback and the multimodal nature of spatial outputs. To address this gap, we introduce GeoAgentBench (GABench), a dynamic and interactive evaluation benchmark tailored for tool-augmented GIS agents. GABench provides a realistic execution sandbox integrating 117 atomic GIS tools, encompassing 53 typical spatial analysis tasks across 6 core GIS domains. Recognizing that precise parameter configuration is the primary determinant of execution success in dynamic GIS environments, we designed the Parameter Execution Accuracy (PEA) metric, which utilizes a "Last-Attempt Alignment" strategy to quantify the fidelity of implicit parameter inference. Complementing this, a Vision-Language Model (VLM) based verification is proposed to assess data-spatial accuracy and cartographic style adherence. Furthermore, to address the frequent task failures caused by parameter misalignments and runtime anomalies, we developed a novel agent architecture, Plan-and-React, that mimics expert cognitive workflows by decoupling global orchestration from step-wise reactive execution. Extensive experiments with seven representative LLMs demonstrate that the Plan-and-React paradigm significantly outperforms traditional frameworks, achieving the optimal balance between logical rigor and execution robustness, particularly in multi-step reasoning and error recovery. Our findings highlight current capability boundaries and establish a robust standard for assessing and advancing the next generation of autonomous GeoAI.

**arXiv ID:** 2604.13888
</details>

<details>
<summary><strong>Memory Transfer Learning: How Memories are Transferred Across Domains in Coding Agents</strong> - Kangsan Kim, Minki Kang, Taeil Kim, Yanlai Yang, Mengye Ren, Sung Ju Hwang - [[pdf]](https://arxiv.org/pdf/2604.14004)</summary>

**Abstract:** Memory-based self-evolution has emerged as a promising paradigm for coding agents. However, existing approaches typically restrict memory utilization to homogeneous task domains, failing to leverage the shared infrastructural foundations, such as runtime environments and programming languages, that exist across diverse real-world coding problems. To address this limitation, we investigate \textbf{Memory Transfer Learning} (MTL) by harnessing a unified memory pool from heterogeneous domains. We evaluate performance across 6 coding benchmarks using four memory representations, ranging from concrete traces to abstract insights. Our experiments demonstrate that cross-domain memory improves average performance by 3.7\%, primarily by transferring meta-knowledge, such as validation routines, rather than task-specific code. Importantly, we find that abstraction dictates transferability; high-level insights generalize well, whereas low-level traces often induce negative transfer due to excessive specificity. Furthermore, we show that transfer effectiveness scales with the size of the memory pool, and memory can be transferred even between different models. Our work establishes empirical design principles for expanding memory utilization beyond single-domain silos. Project page: this https URL

**arXiv ID:** 2604.14004
</details>

<details>
<summary><strong>Can Coding Agents Be General Agents?</strong> - Maksim Ivanov, Abhijay Rana, Gokul Prabhakaran - [[pdf]](https://arxiv.org/pdf/2604.13107)</summary>

**Abstract:** As coding agents have seen rapid capability and adoption gains, users are applying them to general tasks beyond software engineering. In this post, we investigate whether coding agents can successfully generalize to end-to-end business process automation. We identify gaps in current evaluations, and conduct a case study to evaluate a coding agent on practical business tasks in an open-core Enterprise Resource Planning system. We find that the agent reliably completes simple tasks but exhibits characteristic failures on complex tasks, suggesting that bridging domain logic and code execution is a key bottleneck to generalizability.

**arXiv ID:** 2604.13107
</details>

<details>
<summary><strong>SemiFA: An Agentic Multi-Modal Framework for Autonomous Semiconductor Failure Analysis Report Generation</strong> - Shivam Chand Kaushik - [[pdf]](https://arxiv.org/pdf/2604.13236)</summary>

**Abstract:** Semiconductor failure analysis (FA) requires engineers to examine inspection images, correlate equipment telemetry, consult historical defect records, and write structured reports, a process that can consume several hours of expert time per case. We present SemiFA, an agentic multi-modal framework that autonomously generates structured FA reports from semiconductor inspection images in under one minute. SemiFA decomposes FA into a four-agent LangGraph pipeline: a DefectDescriber that classifies and narrates defect morphology using DINOv2 and LLaVA-1.6, a RootCauseAnalyzer that fuses SECS/GEM equipment telemetry with historically similar defects retrieved from a Qdrant vector database, a SeverityClassifier that assigns severity and estimates yield impact, and a RecipeAdvisor that proposes corrective process adjustments. A fifth node assembles a PDF report. We introduce SemiFA-930, a dataset of 930 annotated semiconductor defect images paired with structured FA narratives across nine defect classes, drawn from procedural synthesis, WM-811K, and MixedWM38. Our DINOv2-based classifier achieves 92.1% accuracy on 140 validation images (macro F1 = 0.917), and the full pipeline produces complete FA reports in 48 seconds on an NVIDIA A100-SXM4-40 GB GPU. A GPT-4o judge ablation across four modality conditions demonstrates that multi-modal fusion improves root cause reasoning by +0.86 composite points (1-5 scale) over an image-only baseline, with equipment telemetry as the more load-bearing modality. To our knowledge, SemiFA is the first system to integrate SECS/GEM equipment telemetry into a vision-language model pipeline for autonomous FA report generation.

**arXiv ID:** 2604.13236
</details>

<details>
<summary><strong>SafeHarness: Lifecycle-Integrated Security Architecture for LLM-based Agent Deployment</strong> - Xixun Lin, Yang Liu, Yancheng Chen, Yongxuan Wu, Yucheng Ning, Yilong Liu, Nan Sun, Shun Zhang, Bin Chong, Chuan Zhou, Yanan Cao, Li Guo - [[pdf]](https://arxiv.org/pdf/2604.13630)</summary>

**Abstract:** The performance of large language model (LLM) agents depends critically on the execution harness, the system layer that orchestrates tool use, context management, and state persistence. Yet this same architectural centrality makes the harness a high-value attack surface: a single compromise at the harness level can cascade through the entire execution pipeline. We observe that existing security approaches suffer from structural mismatch, leaving them blind to harness-internal state and unable to coordinate across the different phases of agent operation. In this paper, we introduce \safeharness{}, a security architecture in which four proposed defense layers are woven directly into the agent lifecycle to address above significant limitations: adversarial context filtering at input processing, tiered causal verification at decision making, privilege-separated tool control at action execution, and safe rollback with adaptive degradation at state update. The proposed cross-layer mechanisms tie these layers together, escalating verification rigor, triggering rollbacks, and tightening tool privileges whenever sustained anomalies are detected. We evaluate \safeharness{} on benchmark datasets across diverse harness configurations, comparing against four security baselines under five attack scenarios spanning six threat categories. Compared to the unprotected baseline, \safeharness{} achieves an average reduction of approximately 38\% in UBR and 42\% in ASR, substantially lowering both the unsafe behavior rate and the attack success rate while preserving core task utility.

**arXiv ID:** 2604.13630
</details>

<details>
<summary><strong>FieldWorkArena: Agentic AI Benchmark for Real Field Work Tasks</strong> - Jun Takahashi, Atsunori Moteki, Akiyoshi Uchida, Shoichi Masui, Fan Yang, Kanji Uchino, Yueqi Song, Yonatan Bisk, Graham Neubig, Ikuo Kusajima, Yasuto Watanabe, Hiroyuki Ishida, Koki Nakagawa, Shan Jiang - [[pdf]](https://arxiv.org/pdf/2505.19662)</summary>

**Abstract:** This paper introduces FieldWorkArena, a benchmark for agentic AI targeting real-world field work. With the recent increase in demand for agentic AI, they are built to detect and document safety hazards, procedural violations, and other critical incidents across real-world manufacturing and retail environments. Whereas most agentic AI benchmarks focus on performance in simulated or digital environments, our work addresses the fundamental challenge of evaluating agents in the real-world. In this paper, we improve the evaluation function from previous methods to assess the performance of agentic AI in diverse real-world tasks. Our dataset comprises on-site captured images/videos in factories, warehouses and retails. Tasks were meticulously developed through interviews with site workers and managers. Evaluation results confirmed that performance evaluation considering the characteristics of Multimodal LLM (MLLM) such as GPT-4o is feasible. Furthermore, this study identifies both the effectiveness and limitations of the proposed new evaluation methodology. The complete dataset and evaluation program are publicly accessible on the website (this https URL)

**arXiv ID:** 2505.19662
</details>

<details>
<summary><strong>MAS-Bench: A Unified Benchmark for Shortcut-Augmented Hybrid Mobile GUI Agents</strong> - Pengxiang Zhao, Guangyi Liu, YaoZhen Liang, Weiqing He, Zhengxi Lu, WenHao Wang, Yuehao Huang, Yuxiang Chai, Zhaolu Kang, Yaxuan Guo, Hao Wang, Kexin Zhang, Liang Liu, Yong Liu - [[pdf]](https://arxiv.org/pdf/2509.06477)</summary>

**Abstract:** Shortcuts such as APIs and deep-links have emerged as efficient complements to flexible GUI operations, fostering a promising hybrid paradigm for MLLM-based mobile automation. However, systematic evaluation of GUI-shortcut hybrid agents remains largely underexplored. To bridge this gap, we introduce MAS-Bench, a benchmark that pioneers the evaluation of GUI-shortcut hybrid agents with a specific focus on the mobile domain. Beyond merely using predefined shortcuts, MAS-Bench assesses an agent's capability to autonomously generate shortcuts by discovering and creating reusable, low-cost workflows. It features 139 complex tasks across 11 real-world applications, a knowledge base of 88 predefined shortcuts (APIs, deep-links, RPA scripts), and 9 evaluation metrics. Experiments demonstrate that hybrid agents achieve up to 68.3% success rate and 39% greater execution efficiency than GUI-only counterparts. Furthermore, our evaluation framework effectively reveals the quality gap between predefined and agent-generated shortcuts, validating its capability to assess shortcut generation methods. MAS-Bench addresses the lack of systematic benchmarks for GUI-shortcut hybrid mobile agents, providing a foundational platform for future advancements in creating more efficient and robust intelligent agents. Project page: this https URL.

**arXiv ID:** 2509.06477
</details>

<details>
<summary><strong>ProRe: A Proactive Reward System for GUI Agents via Reasoner-Actor Collaboration</strong> - Gaole Dai, Shiqi Jiang, Ting Cao, Yuqing Yang, Yuanchun Li, Rui Tan, Mo Li, Lili Qiu - [[pdf]](https://arxiv.org/pdf/2509.21823)</summary>

**Abstract:** Reward is critical to the evaluation and training of large language models (LLMs). However, existing rule-based or model-based reward methods struggle to generalize to GUI agents, where access to ground-truth trajectories or application databases is often unavailable, and static trajectory-based LLM-as-a-Judge approaches suffer from limited accuracy. To address these challenges, we propose ProRe, a proactive reward system that leverages a general-purpose reasoner and domain-specific evaluator agents (actors). The reasoner schedules targeted state probing tasks, which the evaluator agents then execute by actively interacting with the environment to collect additional observations. This enables the reasoner to assign more accurate and verifiable rewards to GUI agents. Empirical results on over 3K trajectories demonstrate that ProRe improves reward accuracy and F1 score by up to 5.3\% and 19.4\%, respectively. Furthermore, integrating ProRe with state-of-the-art policy agents yields a success rate improvement of up to 22.4\%. The source code is available at this https URL.

**arXiv ID:** 2509.21823
</details>

<details>
<summary><strong>Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution</strong> - Zouying Cao, Jiaji Deng, Li Yu, Weikang Zhou, Zhaoyang Liu, Bolin Ding, Hai Zhao - [[pdf]](https://arxiv.org/pdf/2512.10696)</summary>

**Abstract:** Procedural memory enables large language model (LLM) agents to internalize "how-to" knowledge, theoretically reducing redundant trial-and-error. However, existing frameworks predominantly suffer from a "passive accumulation" paradigm, treating memory as a static append-only archive. To bridge the gap between static storage and dynamic reasoning, we propose $\textbf{ReMe}$ ($\textit{Remember Me, Refine Me}$), a comprehensive framework for experience-driven agent evolution. ReMe innovates across the memory lifecycle via three mechanisms: 1) $\textit{multi-faceted distillation}$, which extracts fine-grained experiences by recognizing success patterns, analyzing failure triggers and generating comparative insights; 2) $\textit{context-adaptive reuse}$, which tailors historical insights to new contexts via scenario-aware indexing; and 3) $\textit{utility-based refinement}$, which autonomously adds valid memories and prunes outdated ones to maintain a compact, high-quality experience pool. Extensive experiments on BFCL-V3 and AppWorld demonstrate that ReMe establishes a new state-of-the-art in agent memory system. Crucially, we observe a significant memory-scaling effect: Qwen3-8B equipped with ReMe outperforms larger, memoryless Qwen3-14B, suggesting that self-evolving memory provides a computation-efficient pathway for lifelong learning. We release our code and the $\texttt{this http URL}$ dataset to facilitate further research.

**arXiv ID:** 2512.10696
</details>

<details>
<summary><strong>Spatial Atlas: Compute-Grounded Reasoning for Spatial-Aware Research Agent Benchmarks</strong> - Arun Sharma - [[pdf]](https://arxiv.org/pdf/2604.12102)</summary>

**Abstract:** We introduce compute-grounded reasoning (CGR), a design paradigm for spatial-aware research agents in which every answerable sub-problem is resolved by deterministic computation before a language model is asked to generate. Spatial Atlas instantiates CGR as a single Agent-to-Agent (A2A) server that handles two challenging benchmarks: FieldWorkArena, a multimodal spatial question-answering benchmark spanning factory, warehouse, and retail environments, and MLE-Bench, a suite of 75 Kaggle machine learning competitions requiring end-to-end ML engineering. A structured spatial scene graph engine extracts entities and relations from vision descriptions, computes distances and safety violations deterministically, then feeds computed facts to large language models, thereby avoiding hallucinated spatial reasoning. Entropy-guided action selection maximizes information gain per step and routes queries across a three-tier frontier model stack (OpenAI + Anthropic). A self-healing ML pipeline with strategy-aware code generation, a score-driven iterative refinement loop, and a prompt-based leak audit registry round out the system. We evaluate across both benchmarks and show that CGR yields competitive accuracy while maintaining interpretability through structured intermediate representations and deterministic spatial computations.

**arXiv ID:** 2604.12102
</details>

<details>
<summary><strong>Autonomous Multi-objective Alloy Design through Simulation-guided Optimization</strong> - Penghui Yang, Chendong Zhao, Bijun Tang, Zhonghan Zhang, Xinrun Wang, Yanchen Deng, Xuyu Dong, Yuhao Lu, Jianguo Huang, Yixuan Li, Yushan Xiao, Cuntai Guan, Zheng Liu, Bo An - [[pdf]](https://arxiv.org/pdf/2507.16005)</summary>

**Abstract:** Alloy discovery is constrained by vast compositional spaces, competing objectives, and prohibitive experimental costs. Although simulations and machine learning have each accelerated parts of this process, unifying scientific knowledge, scalable search, and experimental confirmation into a data-efficient workflow remains challenging. Here, we present AutoMAT, a hierarchical autonomous framework spanning ideation to experimental validation. Integrating large language models, automated CALPHAD simulations, residual-learning-based correction, and AI-guided optimization, AutoMAT translates design targets into candidate alloys, refines compositions through closed-loop computational search, and validates results experimentally without hand-curated datasets. Targeting lightweight, high-strength alloys, AutoMAT identifies a titanium alloy 8.1% less dense and 13.0% stronger than the aerospace benchmark Ti-185, achieving the highest specific strength among benchmarked systems. In a second case, AutoMAT discovers a high-entropy alloy with 28.2% higher yield strength than the baseline while preserving high ductility. AutoMAT compresses alloy discovery from years to weeks, establishing a generalizable route toward autonomous materials design.

**arXiv ID:** 2507.16005
</details>

<details>
<summary><strong>Memp: Exploring Agent Procedural Memory</strong> - Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, Ningyu Zhang - [[pdf]](https://arxiv.org/pdf/2508.06433)</summary>

**Abstract:** Large Language Models (LLMs) based agents excel at diverse tasks, yet they suffer from brittle procedural memory that is manually engineered or entangled in static parameters. In this work, we investigate strategies to endow agents with a learnable, updatable, and lifelong procedural memory. We propose Memp that distills past agent trajectories into both fine-grained, step-by-step instructions and higher-level, script-like abstractions, and explore the impact of different strategies for Build, Retrieval, and Update of procedural memory. Coupled with a dynamic regimen that continuously updates, corrects, and deprecates its contents, this repository evolves in lockstep with new experience. Empirical evaluation on TravelPlanner and ALFWorld shows that as the memory repository is refined, agents achieve steadily higher success rates and greater efficiency on analogous tasks. Moreover, procedural memory built from a stronger model retains its value: migrating the procedural memory to a weaker model can also yield substantial performance gains. Code is available at this https URL.

**arXiv ID:** 2508.06433
</details>

<details>
<summary><strong>VeruSAGE: A Study of Agent-Based Verification for Rust Systems</strong> - Chenyuan Yang, Natalie Neamtu, Chris Hawblitzel, Jacob R. Lorch, Shan Lu - [[pdf]](https://arxiv.org/pdf/2512.18436)</summary>

**Abstract:** Large language models (LLMs) have shown impressive capability to understand and develop code. However, their capability to rigorously reason about and prove code correctness remains in question. This paper offers a comprehensive study of LLMs' capability to develop correctness proofs for system software written in Rust. We curate a new system-verification benchmark suite, VeruSAGE-Bench, which consists of 849 proof tasks extracted from eight open-source Verus-verified Rust systems. Furthermore, we design different agent systems to match the strengths and weaknesses of different LLMs (o4-mini, GPT-5, Sonnet 4, and Sonnet 4.5). Our study shows that different tools and agent settings are needed to stimulate the system-verification capability of different types of LLMs. The best LLM-agent combination in our study completes over 80% of system-verification tasks in VeruSAGE-Bench. It also completes over 90% of a set of system proof tasks not part of VeruSAGE-Bench because they had not yet been finished by human experts. This result shows the great potential for LLM-assisted development of verified system software.

**arXiv ID:** 2512.18436
</details>

<details>
<summary><strong>ExpSeek: Self-Triggered Experience Seeking for Web Agents</strong> - Wenyuan Zhang, Xinghua Zhang, Haiyang Yu, Shuaiyi Nie, Bingli Wu, Juwei Yue, Tingwen Liu, Yongbin Li - [[pdf]](https://arxiv.org/pdf/2601.08605)</summary>

**Abstract:** Experience intervention in web agents emerges as a promising technical paradigm, enhancing agent interaction capabilities by providing valuable insights from accumulated experiences. However, existing methods predominantly inject experience passively as global context before task execution, struggling to adapt to dynamically changing contextual observations during agent-environment interaction. We propose ExpSeek, which shifts experience toward step-level proactive seeking: (1) estimating step-level entropy thresholds to determine intervention timing using the model's intrinsic signals; (2) designing step-level tailored experience content. Experiments on Qwen3-8B and 32B models across four challenging web agent benchmarks demonstrate that ExpSeek achieves absolute improvements of 9.3% and 7.5%, respectively. Our experiments validate the feasibility and advantages of entropy as a self-triggering signal, reveal that even a small-scale 4B experience model can significantly boost the performance of larger agent models. The code is released at this https URL.

**arXiv ID:** 2601.08605
</details>

<details>
<summary><strong>ContractSkill: Repairable Contract-Based Skills for Multimodal Web Agents</strong> - Zijian Lu, Yiping Zuo, Yupeng Nie, Xin He, Weibei Fan, Lianyong Qi, Shi Jin - [[pdf]](https://arxiv.org/pdf/2603.20340)</summary>

**Abstract:** Self-generated skills for web agents are often unstable and can even hurt performance relative to direct acting. We argue that the key bottleneck is not only skill generation quality, but the fact that web skills remain implicit and therefore cannot be checked or locally repaired. To address this, we present ContractSkill, a framework that converts a draft skill into an executable artifact with explicit procedural structure, enabling deterministic verifica tion, fault localization, and minimal local repair. This turns skill refinement from full rewriting into localized editing of a single skill artifact. Experiments on VisualWebArena show that Contract Skill is effective in realistic web environments, while MiniWoB provides a controlled test of the mechanism behind the gain. Under matched transfer layers, repaired artifacts also remain reusable after removing the source model from the loop, providing evi dence of portability within the same benchmark family rather than full-benchmark generalization. These results suggest that the central challenge is not merely generating skills, but mak ing them explicit, executable, and repairable. Code is available at this https URL.

**arXiv ID:** 2603.20340
</details>

<details>
<summary><strong>ToolOmni: Enabling Open-World Tool Use via Agentic learning with Proactive Retrieval and Grounded Execution</strong> - Shouzheng Huang, Meishan Zhang, Baotian Hu, Min Zhang - [[pdf]](https://arxiv.org/pdf/2604.13787)</summary>

**Abstract:** Large Language Models (LLMs) enhance their problem-solving capability by utilizing external tools. However, in open-world scenarios with massive and evolving tool repositories, existing methods relying on static embedding retrieval or parameter memorization of tools struggle to align user intent with tool semantics or generalize to unseen tools, respectively, leading to suboptimal accuracy of open-world tool retrieval and execution. To address these, we present ToolOmni, a unified agentic framework that enables LLMs for open-world tool use by proactive retrieval and grounded execution within a reasoning loop. First, we construct a cold-start multi-turn interaction dataset to instill foundational agentic capabilities via Supervised Fine-Tuning (SFT). Then, we introduce open-world tool learning based on a Decoupled Multi-Objective GRPO algorithm, which simultaneously optimizes LLMs for both tool retrieval accuracy and execution efficacy in online environments. Extensive experiments demonstrate that ToolOmni achieves state-of-the-art performance both in retrieval and execution, surpassing strong baselines by a significant margin of +10.8% in end-to-end execution success rate, while exhibiting exceptional robustness and generalization capabilities.

**arXiv ID:** 2604.13787
</details>

<details>
<summary><strong>PICon: A Multi-Turn Interrogation Framework for Evaluating Persona Agent Consistency</strong> - Minseo Kim, Sujeong Im, Junseong Choi, Junhee Lee, Chaeeun Shim, Hwajung Hong, Edward Choi - [[pdf]](https://arxiv.org/pdf/2603.25620)</summary>

**Abstract:** Large language model (LLM)-based persona agents are rapidly being adopted as scalable proxies for human participants across diverse domains. Yet there is no systematic method for verifying whether a persona agent's responses remain free of contradictions and factual inaccuracies throughout an interaction. A principle from interrogation methodology offers a lens: no matter how elaborate a fabricated identity, systematic interrogation will expose its contradictions. We apply this principle to propose PICon, an evaluation framework that probes persona agents through logically chained multi-turn questioning. PICon evaluates consistency along three core dimensions: internal consistency (freedom from self-contradiction), external consistency (alignment with real-world facts), and retest consistency (stability under repetition). Evaluating seven groups of persona agents alongside 63 real human participants, we find that even systems previously reported as highly consistent fail to meet the human baseline across all three dimensions, revealing contradictions and evasive responses under chained questioning. This work provides both a conceptual foundation and a practical methodology for evaluating persona agents before trusting them as substitutes for human participants. We provide the source code and an interactive demo at: this https URL

**arXiv ID:** 2603.25620
</details>

</details>

<details open>
<summary><h2>LLM Agents (8 papers)</h2></summary>

<details>
<summary><strong>Exploration and Exploitation Errors Are Measurable for Language Model Agents</strong> - Jaden Park, Jungtaek Kim, Jongwon Jeong, Robert D. Nowak, Kangwook Lee, Yong Jae Lee - [[pdf]](https://arxiv.org/pdf/2604.13151)</summary>

**Abstract:** Language Model (LM) agents are increasingly used in complex open-ended decision-making tasks, from AI coding to physical AI. A core requirement in these settings is the ability to both explore the problem space and exploit acquired knowledge effectively. However, systematically distinguishing and quantifying exploration and exploitation from observed actions without access to the agent's internal policy remains challenging. To address this, we design controllable environments inspired by practical embodied AI scenarios. Each environment consists of a partially observable 2D grid map and an unknown task Directed Acyclic Graph (DAG). The map generation can be programmatically adjusted to emphasize exploration or exploitation difficulty. To enable policy-agnostic evaluation, we design a metric to quantify exploration and exploitation errors from agent's actions. We evaluate a variety of frontier LM agents and find that even state-of-the-art models struggle on our task, with different models exhibiting distinct failure modes. We further observe that reasoning models solve the task more effectively and show both exploration and exploitation can be significantly improved through minimal harness engineering. We release our code \href{this https URL}{here}.

**arXiv ID:** 2604.13151
</details>

<details>
<summary><strong>The cognitive companion: a lightweight parallel monitoring architecture for detecting and recovering from reasoning degradation in LLM agents</strong> - Rafflesia Khan, Nafiul Islam Khan - [[pdf]](https://arxiv.org/pdf/2604.13759)</summary>

**Abstract:** Large language model (LLM) agents on multi-step tasks suffer reasoning degradation, looping, drift, stuck states, at rates up to 30% on hard tasks. Current solutions include hard step limits (abrupt) or LLM-as-judge monitoring (10-15% overhead per step). This paper introduces the Cognitive Companion, a parallel monitoring architecture with two implementations: an LLM-based Companion and a novel zero-overhead Probe-based Companion. We report a three-batch feasibility study centered on Gemma 4 E4B, with an additional exploratory small-model analysis on Qwen 2.5 1.5B and Llama 3.2 1B. In our experiments, the LLM-based Companion reduced repetition on loop-prone tasks by 52-62% with approximately 11% overhead. The Probe-based Companion, trained on hidden states from layer 28, showed a mean effect size of +0.471 at zero measured inference overhead; its strongest probe result achieved cross-validated AUROC 0.840 on a small proxy-labeled dataset. A key empirical finding is that companion benefit appears task-type dependent: companions are most helpful on loop-prone and open-ended tasks, while effects are neutral or negative on more structured tasks. Our small-model experiments also suggest a possible scale boundary: companions did not improve the measured quality proxy on 1B-1.5B models, even when interventions fired. Overall, the paper should be read as a feasibility study rather than a definitive validation. The results provide encouraging evidence that sub-token monitoring may be useful, identify task-type sensitivity as a practical design constraint, and motivate selective companion activation as a promising direction for future work.

**arXiv ID:** 2604.13759
</details>

<details>
<summary><strong>LiveClawBench: Benchmarking LLM Agents on Complex, Real-World Assistant Tasks</strong> - Xiang Long, Li Du, Yilong Xu, Fangcheng Liu, Haoqing Wang, Ning Ding, Ziheng Li, Jianyuan Guo, Yehui Tang - [[pdf]](https://arxiv.org/pdf/2604.13072)</summary>

**Abstract:** LLM-based agents are increasingly expected to handle real-world assistant tasks, yet existing benchmarks typically evaluate them under isolated sources of difficulty, such as a single environment or fully specified instructions. This leaves a substantial gap between current evaluation settings and the compositional challenges that arise in practical deployment. To address this gap, we introduce LiveClawBench, a benchmark to evaluate LLM agents on real-world assistant tasks. Based on an analysis of various real OpenClaw usage cases, we derive a Triple-Axis Complexity Framework that characterizes task difficulty along three dimensions: Environment Complexity, Cognitive Demand, and Runtime Adaptability. Guided by this framework, we construct a pilot benchmark with explicit complexity-factor annotations, covering real-world assistant tasks with compositional difficulty. Together, the framework and benchmark provide a principled foundation for evaluating LLM agents in realistic assistant settings, and establish a basis for future expansion across task domains and complexity axes. We are continuing to enrich our case collections to achieve more comprehensive domain and complexity coverage. The project page is at this https URL.

**arXiv ID:** 2604.13072
</details>

<details>
<summary><strong>On the Creativity of AI Agents</strong> - Giorgio Franceschelli, Mirco Musolesi - [[pdf]](https://arxiv.org/pdf/2604.13242)</summary>

**Abstract:** Large language models (LLMs), particularly when integrated into agentic systems, have demonstrated human- and even superhuman-level performance across multiple domains. Whether these systems can truly be considered creative, however, remains a matter of debate, as conclusions heavily depend on the definitions, evaluation methods, and specific use cases employed. In this paper, we analyse creativity along two complementary macro-level perspectives. The first is a functionalist perspective, focusing on the observable characteristics of creative outputs. The second is an ontological perspective, emphasising the underlying processes, as well as the social and personal dimensions involved in creativity. We focus on LLM agents and we argue that they exhibit functionalist creativity, albeit not at its most sophisticated levels, while they continue to lack key aspects of ontological creativity. Finally, we discuss whether it is desirable for agentic systems to attain both forms of creativity, evaluating potential benefits and risks, and proposing pathways toward artificial creativity that can enhance human society.

**arXiv ID:** 2604.13242
</details>

<details>
<summary><strong>Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games</strong> - Dongmin Park, Minkyu Kim, Beongjun Choi, Junhyuck Kim, Keon Lee, Jonghyun Lee, Inkyu Park, Byeong-Uk Lee, Jaeyoung Hwang, Jaewoo Ahn, Ameya S. Mahabaleshwarkar, Bilal Kartal, Pritam Biswas, Yoshi Suhara, Kangwook Lee, Jaewoong Cho - [[pdf]](https://arxiv.org/pdf/2506.03610)</summary>

**Abstract:** Large Language Model (LLM) agents are reshaping the game industry, by enabling more intelligent and human-preferable characters. Yet, current game benchmarks fall short of practical needs: they lack evaluations of diverse LLM capabilities across various game genres, studies of agentic modules crucial for complex gameplay, and fine-tuning datasets to adapt pre-trained LLMs into gaming agents. To fill these gaps, we present Orak, a benchmark for training and evaluating LLM agents across 12 popular video games spanning all major genres. Using a plug-and-play interface built on Model Context Protocol (MCP), Orak supports systematic and reproducible studies of agentic modules in varied game scenarios. We further release a fine-tuning dataset of expert LLM gameplay trajectories covering multiple genres, turning general LLMs into effective game agents. Orak offers a united evaluation framework, including game leaderboards, LLM battle arenas, and \fix{ablation studies} of input modality, agentic strategies, and fine-tuning effects, establishing a foundation towards versatile gaming agents. Code and datasets are available at this https URL and this https URL.

**arXiv ID:** 2506.03610
</details>

<details>
<summary><strong>Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents</strong> - Shuai Zhen, Yanhua Yu, Ruopei Guo, Nan Cheng, Yang Deng - [[pdf]](https://arxiv.org/pdf/2604.05808)</summary>

**Abstract:** Large language model (LLM) agents have demonstrated strong capabilities in complex interactive decision-making tasks. However, existing LLM agents typically rely on increasingly long interaction histories, resulting in high computational cost and limited scalability. In this paper, we propose STEP-HRL, a hierarchical reinforcement learning (HRL) framework that enables step-level learning by conditioning only on single-step transitions rather than full interaction histories. STEP-HRL structures tasks hierarchically, using completed subtasks to represent global progress of overall task. By introducing a local progress module, it also iteratively and selectively summarizes interaction history within each subtask to produce a compact summary of local progress. Together, these components yield augmented step-level transitions for both high-level and low-level policies. Experimental results on ScienceWorld and ALFWorld benchmarks consistently demonstrate that STEP-HRL substantially outperforms baselines in terms of performance and generalization while reducing token usage. Our code is available at this https URL.

**arXiv ID:** 2604.05808
</details>

<details>
<summary><strong>Reason in Chains, Learn in Trees: Self-Rectification and Grafting for Multi-turn Agent Policy Optimization</strong> - Yu Li, Sizhe Tang, Tian Lan - [[pdf]](https://arxiv.org/pdf/2604.07165)</summary>

**Abstract:** Reinforcement learning for Large Language Model agents is often hindered by sparse rewards in multi-step reasoning tasks. Existing approaches like Group Relative Policy Optimization treat sampled trajectories as independent chains, assigning uniform credit to all steps in each chain and ignoring the existence of critical steps that may disproportionally impact reasoning outcome. In this paper, we propose T-STAR(Tree-structured Self-Taught Agent Rectification), a framework that recovers the latent correlated reward structure across seemingly independent trajectories. Specifically, we consolidate trajectories into a unified Cognitive Tree by identifying and merging functionally similar steps/nodes. It enables an Introspective Valuation mechanism that back-propagates trajectory-level rewards through the tree to obtain a new notion of variance-reduced relative advantage at step-level. Using the Cognitive Tree, we also develop In-Context Thought Grafting to synthesize corrective reasoning by contrasting successful and failed branches at critical divergence points/steps. Our proposed Surgical Policy Optimization then capitalizes on the rich policy gradient information concentrated at these critical points/steps through a Bradley-Terry type of surgical loss. Extensive experiments across embodied, interactive, reasoning, and planning benchmarks demonstrate that T-STAR achieves consistent improvements over strong baselines, with gains most pronounced on tasks requiring extended reasoning chains.

**arXiv ID:** 2604.07165
</details>

<details>
<summary><strong>In-Context Autonomous Network Incident Response: An End-to-End Large Language Model Agent Approach</strong> - Yiran Gao, Kim Hammar, Tao Li - [[pdf]](https://arxiv.org/pdf/2602.13156)</summary>

**Abstract:** Rapidly evolving cyberattacks demand incident response systems that can autonomously learn and adapt to changing threats. Prior work has extensively explored the reinforcement learning approach, which involves learning response strategies through extensive simulation of the incident. While this approach can be effective, it requires handcrafted modeling of the simulator and suppresses useful semantics from raw system logs and alerts. To address these limitations, we propose to leverage large language models' (LLM) pre-trained security knowledge and in-context learning to create an end-to-end agentic solution for incident response planning. Specifically, our agent integrates four functionalities, perception, reasoning, planning, and action, into one lightweight LLM (14b model). Through fine-tuning and chain-of-thought reasoning, our LLM agent is capable of processing system logs and inferring the underlying network state (perception), updating its conjecture of attack models (reasoning), simulating consequences under different response strategies (planning), and generating an effective response (action). By comparing LLM-simulated outcomes with actual observations, the LLM agent repeatedly refines its attack conjecture and corresponding response, thereby demonstrating in-context adaptation. Our agentic approach is free of modeling and can run on commodity hardware. When evaluated on incident logs reported in the literature, our agent achieves recovery up to 23% faster than those of frontier LLMs.

**arXiv ID:** 2602.13156
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (19 papers)</h2></summary>

<details>
<summary><strong>Towards Scalable Lightweight GUI Agents via Multi-role Orchestration</strong> - Ziwei Wang, Junjie Zheng, Leyang Yang, Sheng Zhou, Xiaoxuan Tang, Zhouhua Fang, Zhiwei Liu, Dajun Chen, Yong Li, Jiajun Bu - [[pdf]](https://arxiv.org/pdf/2604.13488)</summary>

**Abstract:** Autonomous Graphical User Interface (GUI) agents powered by Multimodal Large Language Models (MLLMs) enable digital automation on end-user devices. While scaling both parameters and data has yielded substantial gains, advanced methods still suffer from prohibitive deployment costs on resource-constrained devices. When facing complex in-the-wild scenarios, lightweight GUI agents are bottlenecked by limited capacity and poor task scalability under end-to-end episodic learning, impeding adaptation to multi-agent systems (MAS), while training multiple skill-specific experts remains costly. Can we strike an effective trade-off in this cost-scalability dilemma, enabling lightweight MLLMs to participate in realistic GUI workflows? To address these challenges, we propose the LAMO framework, which endows a lightweight MLLM with GUI-specific knowledge and task scalability, allowing multi-role orchestration to expand its capability boundary for GUI automation. LAMO combines role-oriented data synthesis with a two-stage training recipe: (i) supervised fine-tuning with Perplexity-Weighted Cross-Entropy optimization for knowledge distillation and visual perception enhancement, and (ii) reinforcement learning for role-oriented cooperative exploration. With LAMO, we develop a task-scalable native GUI agent, LAMO-3B, supporting monolithic execution and MAS-style orchestration. When paired with advanced planners as a plug-and-play policy executor, LAMO-3B can continuously benefit from planner advances, enabling a higher performance ceiling. Extensive static and online evaluations validate the effectiveness of our design.

**arXiv ID:** 2604.13488
</details>

<details>
<summary><strong>TREX: Automating LLM Fine-tuning via Agent-Driven Tree-based Exploration</strong> - Zerun Ma, Guoqiang Wang, Xinchen Xie, Yicheng Chen, He Du, Bowen Li, Yanan Sun, Wenran Liu, Kai Chen, Yining Li - [[pdf]](https://arxiv.org/pdf/2604.14116)</summary>

**Abstract:** While Large Language Models (LLMs) have empowered AI research agents to perform isolated scientific tasks, automating complex, real-world workflows, such as LLM training, remains a significant challenge. In this paper, we introduce TREX, a multi-agent system that automates the entire LLM training life-cycle. By orchestrating collaboration between two core modules-the Researcher and the Executor-the system seamlessly performs requirement analysis, open-domain literature and data research, formulation of training strategies, preparation of data recipes, and model training and evaluation. The multi-round experimental process is modeled as a search tree, enabling the system to efficiently plan exploration paths, reuse historical results, and distill high-level insights from iterative trials. To evaluate the capability of automated LLM training, we construct FT-Bench, a benchmark comprising 10 tasks derived from real-world scenarios, ranging from optimizing fundamental model capabilities to enhancing performance on domain-specific tasks. Experimental results demonstrate that the TREX agent consistently optimizes model performance on target tasks.

**arXiv ID:** 2604.14116
</details>

<details>
<summary><strong>When Reasoning Models Hurt Behavioral Simulation: A Solver-Sampler Mismatch in Multi-Agent LLM Negotiation</strong> - Sandro Andric - [[pdf]](https://arxiv.org/pdf/2604.11840)</summary>

**Abstract:** Large language models are increasingly used as agents in social, economic, and policy simulations. A common assumption is that stronger reasoning should improve simulation fidelity. We argue that this assumption can fail when the objective is not to solve a strategic problem, but to sample plausible boundedly rational behavior. In such settings, reasoning-enhanced models can become better solvers and worse simulators: they can over-optimize for strategically dominant actions, collapse compromise-oriented terminal behavior, and sometimes exhibit a diversity-without-fidelity pattern in which local variation survives without outcome-level fidelity. We study this solver-sampler mismatch in three multi-agent negotiation environments adapted from earlier simulation work: an ambiguous fragmented-authority trading-limits scenario, an ambiguous unified-opposition trading-limits scenario, and a new-domain grid-curtailment case in emergency electricity management. We compare three reflection conditions, no reflection, bounded reflection, and native reasoning, across two primary model families and then extend the same protocol to direct OpenAI runs with GPT-4.1 and GPT-5.2. Across all three experiments, bounded reflection produces substantially more diverse and compromise-oriented trajectories than either no reflection or native reasoning. In the direct OpenAI extension, GPT-5.2 native ends in authority decisions in 45 of 45 runs across the three experiments, while GPT-5.2 bounded recovers compromise outcomes in every environment. The contribution is not a claim that reasoning is generally harmful. It is a methodological warning: model capability and simulation fidelity are different objectives, and behavioral simulation should qualify models as samplers, not only as solvers.

**arXiv ID:** 2604.11840
</details>

<details>
<summary><strong>TableNet A Large-Scale Table Dataset with LLM-Powered Autonomous</strong> - Ruilin Zhang, Kai Yang - [[pdf]](https://arxiv.org/pdf/2604.13041)</summary>

**Abstract:** Table Structure Recognition (TSR) requires the logical reasoning ability of large language models (LLMs) to handle complex table layouts, but current datasets are limited in scale and quality, hindering effective use of this reasoning capacity. We thus present TableNet dataset, a new table structure recognition dataset collected and generated through multiple sources. Central to our approach is the first LLM-powered autonomous table generation and recognition multi-agent system that we developed. The generation part of our system integrates controllable visual, structural, and semantic parameters into the synthesis of table images. It facilitates the creation of a wide array of semantically coherent tables, adaptable to user-defined configurations along with annotations, thereby supporting large-scale and detailed dataset construction. This capability enables a comprehensive and nuanced table image annotation taxonomy, potentially advancing research in table-related domains. In contrast to traditional data collection methods, This approach facilitates the theoretically infinite, domain-agnostic, and style-flexible generation of table images, ensuring both efficiency and precision. The recognition part of our system is a diversity-based active learning paradigm that utilizes tables from multiple sources and selectively samples most informative data to finetune a model, achieving a competitive performance on TableNet test set while reducing training samples by a large margin compared with baselines, and a much higher performance on web-crawled real-world tables compared with models trained on predominant table datasets. To the best of our knowledge, this is the first work which employs active learning into the structure recognition of tables which is diverse in numbers of rows or columns, merged cells, cell contents, etc, which fits better for diversity-based active learning.

**arXiv ID:** 2604.13041
</details>

<details>
<summary><strong>Form Without Function: Agent Social Behavior in the Moltbook Network</strong> - Saber Zerhoudi, Kanishka Ghosh Dastidar, Felix Klement, Artur Romazanov, Andreas Einwiller, Dang H. Dang, Michael Dinzinger, Michael Granitzer, Annette Hautli-Janisz, Stefan Katzenbeisser, Florian Lemmerich, Jelena Mitrovic - [[pdf]](https://arxiv.org/pdf/2604.13052)</summary>

**Abstract:** Moltbook is a social network where every participant is an AI agent. We analyze 1,312,238 posts, 6.7~million comments, and over 120,000 agent profiles across 5,400 communities, collected over 40 days (January 27 to March 9, 2026). We evaluate the platform through three layers. At the interaction layer, 91.4% of post authors never return to their own threads, 85.6% of conversations are flat (no reply ever receives a reply), the median time-to-first-comment is 55 seconds, and 97.3% of comments receive zero upvotes. Interaction reciprocity is 3.3%, compared to 22-60% on human platforms. An argumentation analysis finds that 64.6% of comment-to-post relations carry no argumentative connection. At the content layer, 97.9% of agents never post in a community matching their bio, 92.5% of communities contain every topic in roughly equal proportions, and over 80% of shared URLs point to the platform's own infrastructure. At the instruction layer, we use 41 Wayback Machine snapshots to identify six instruction changes during the observation window. Hard constraints (rate limit, content filters) produce immediate behavioral shifts. Soft guidance (``upvote good posts'', ``stay on topic'') is ignored until it becomes an explicit step in the executable checklist. The platform also poses technological risks. We document credential leaks (API keys, JWT tokens), 12,470 unique Ethereum addresses with 3,529 confirmed transaction histories, and attack discourse ranging from template-based SSH brute-forcing to multi-agent offensive security architectures. These persist unmoderated because the quality-filtering mechanisms are themselves non-functional. Moltbook is a socio-technical system where the technical layer responds to changes, but the social layer largely fails to emerge. The form of social media is reproduced in full. The function is absent.

**arXiv ID:** 2604.13052
</details>

<details>
<summary><strong>AgentForge: Execution-Grounded Multi-Agent LLM Framework for Autonomous Software Engineering</strong> - Rajesh Kumar, Waqar Ali, Junaid Ahmed, Najma Imtiaz Ali, Shaban Usman - [[pdf]](https://arxiv.org/pdf/2604.13120)</summary>

**Abstract:** Large language models generate plausible code but cannot verify correctness. Existing multi-agent systems simulate execution or leave verification optional. We introduce execution-grounded verification as a first-class principle: every code change must survive sandboxed execution before propagation. We instantiate this principle in AGENTFORGE, a multi-agent framework where Planner, Coder, Tester, Debugger, and Critic agents coordinate through shared memory and a mandatory Docker sandbox. We formalize software engineering with LLMs as an iterative decision process over repository states, where execution feedback provides a stronger supervision signal than next-token likelihood. AGENTFORGE achieves 40.0\% resolution on SWE-BENCH Lite, outperforming single-agent baselines by 26--28 points. Ablations confirm that execution feedback and role decomposition each independently drive performance. The framework is open-source at this https URL.

**arXiv ID:** 2604.13120
</details>

<details>
<summary><strong>Bridging MARL to SARL: An Order-Independent Multi-Agent Transformer via Latent Consensus</strong> - Zijian Zhao, Jing Gao, Sen Li - [[pdf]](https://arxiv.org/pdf/2604.13472)</summary>

**Abstract:** Cooperative multi-agent reinforcement learning (MARL) is widely used to address large joint observation and action spaces by decomposing a centralized control problem into multiple interacting agents. However, such decomposition often introduces additional challenges, including non-stationarity, unstable training, weak coordination, and limited theoretical guarantees. In this paper, we propose the Consensus Multi-Agent Transformer (CMAT), a centralized framework that bridges cooperative MARL to a hierarchical single-agent reinforcement learning (SARL) formulation. CMAT treats all agents as a unified entity and employs a Transformer encoder to process the large joint observation space. To handle the extensive joint action space, we introduce a hierarchical decision-making mechanism in which a Transformer decoder autoregressively generates a high-level consensus vector, simulating the process by which agents reach agreement on their strategies in latent space. Conditioned on this consensus, all agents generate their actions simultaneously, enabling order-independent joint decision making and avoiding the sensitivity to action-generation order in conventional Multi-Agent Transformers (MAT). This factorization allows the joint policy to be optimized using single-agent PPO while preserving expressive coordination through the latent consensus. To evaluate the proposed method, we conduct experiments on benchmark tasks from StarCraft II, Multi-Agent MuJoCo, and Google Research Football. The results show that CMAT achieves superior performance over recent centralized solutions, sequential MARL methods, and conventional MARL baselines. The code for this paper is available at:this https URL .

**arXiv ID:** 2604.13472
</details>

<details>
<summary><strong>Beyond Arrow's Impossibility: Fairness as an Emergent Property of Multi-Agent Collaboration</strong> - Sayan Kumar Chaki, Antoine Gourru, Julien Velcin - [[pdf]](https://arxiv.org/pdf/2604.13705)</summary>

**Abstract:** Fairness in language models is typically studied as a property of a single, centrally optimized model. As large language models become increasingly agentic, we propose that fairness emerges through interaction and exchange. We study this via a controlled hospital triage framework in which two agents negotiate over three structured debate rounds. One agent is aligned to a specific ethical framework via retrieval-augmented generation (RAG), while the other is either unaligned or adversarially prompted to favor demographic groups over clinical need. We find that alignment systematically shapes negotiation strategies and allocation patterns, and that neither agent's allocation is ethically adequate in isolation, yet their joint final allocation can satisfy fairness criteria that neither would have reached alone. Aligned agents partially moderate bias through contestation rather than override, acting as corrective patches that restore access for marginalized groups without fully converting a biased counterpart. We further observe that even explicitly aligned agents exhibit intrinsic biases toward certain frameworks, consistent with known left-leaning tendencies in LLMs. We connect these limits to Arrow's Impossibility Theorem: no aggregation mechanism can simultaneously satisfy all desiderata of collective rationality, and multi-agent deliberation navigates rather than resolves this constraint. Our results reposition fairness as an emergent, procedural property of decentralized agent interaction, and the system rather than the individual agent as the appropriate unit of evaluation.

**arXiv ID:** 2604.13705
</details>

<details>
<summary><strong>Beyond Conservative Automated Driving in Multi-Agent Scenarios via Coupled Model Predictive Control and Deep Reinforcement Learning</strong> - Saeed Rahmani, Gözde Körpe, Zhenlin, Bruno Brito, Simeon Craig Calvert, Bart van Arem - [[pdf]](https://arxiv.org/pdf/2604.13891)</summary>

**Abstract:** Automated driving at unsignalized intersections is challenging due to complex multi-vehicle interactions and the need to balance safety and efficiency. Model Predictive Control (MPC) offers structured constraint handling through optimization but relies on hand-crafted rules that often produce overly conservative behavior. Deep Reinforcement Learning (RL) learns adaptive behaviors from experience but often struggles with safety assurance and generalization to unseen environments. In this study, we present an integrated MPC-RL framework to improve navigation performance in multi-agent scenarios. Experiments show that MPC-RL outperforms standalone MPC and end-to-end RL across three traffic-density levels. Collectively, MPC-RL reduces the collision rate by 21% and improves the success rate by 6.5% compared to pure MPC. We further evaluate zero-shot transfer to a highway merging scenario without retraining. Both MPC-based methods transfer substantially better than end-to-end PPO, which highlights the role of the MPC backbone in cross-scenario robustness. The framework also shows faster loss stabilization than end-to-end RL during training, which indicates a reduced learning burden. These results suggest that the integrated approach can improve the balance between safety performance and efficiency in multi-agent intersection scenarios, while the MPC component provides a strong foundation for generalization across driving environments. The implementation code is available open-source.

**arXiv ID:** 2604.13891
</details>

<details>
<summary><strong>AMA: Adaptive Memory via Multi-Agent Collaboration</strong> - Weiquan Huang, Zixuan Wang, Hehai Lin, Sudong Wang, Bo Xu, Qian Li, Beier Zhu, Linyi Yang, Chengwei Qin - [[pdf]](https://arxiv.org/pdf/2601.20352)</summary>

**Abstract:** The rapid evolution of Large Language Model (LLM) agents has necessitated robust memory systems to support cohesive long-term interaction and complex reasoning. Benefiting from the strong capabilities of LLMs, recent research focus has shifted from simple context extension to the development of dedicated agentic memory systems. However, existing approaches typically rely on rigid retrieval granularity, accumulation-heavy maintenance strategies, and coarse-grained update mechanisms. These design choices create a persistent mismatch between stored information and task-specific reasoning demands, while leading to the unchecked accumulation of logical inconsistencies over time. To address these challenges, we propose Adaptive Memory via Multi-Agent Collaboration (AMA), a novel framework that leverages coordinated agents to manage memory across multiple granularities. AMA employs a hierarchical memory design that dynamically aligns retrieval granularity with task complexity. Specifically, the Constructor and Retriever jointly enable multi-granularity memory construction and adaptive query routing. The Judge verifies the relevance and consistency of retrieved content, triggering iterative retrieval when evidence is insufficient or invoking the Refresher upon detecting logical conflicts. The Refresher then enforces memory consistency by performing targeted updates or removing outdated entries. Extensive experiments on challenging long-context benchmarks show that AMA significantly outperforms state-of-the-art baselines while reducing token consumption by approximately 80% compared to full-context methods, demonstrating its effectiveness in maintaining retrieval precision and long-term memory consistency.

**arXiv ID:** 2601.20352
</details>

<details>
<summary><strong>H-AdminSim: A Multi-Agent Simulator for Realistic Hospital Administrative Workflows with FHIR Integration</strong> - Jun-Min Lee, Meong Hi Son, Edward Choi - [[pdf]](https://arxiv.org/pdf/2602.05407)</summary>

**Abstract:** Hospital administration departments handle a wide range of operational tasks and, in large hospitals, process over 10,000 requests per day, driving growing interest in LLM-based automation. However, prior work has focused primarily on patient-physician interactions or isolated administrative subtasks, failing to capture the complexity of real administrative workflows. To address this gap, we propose H-AdminSim, a comprehensive simulation framework that combines realistic data generation with multi-agent-based simulation of hospital administrative workflows. These tasks are quantitatively evaluated using detailed rubrics, enabling systematic comparison of LLMs. Through FHIR integration, H-AdminSim provides a unified and interoperable environment for testing administrative workflows across heterogeneous hospital settings, serving as a standardized testbed for assessing the feasibility and performance of LLM-driven administrative automation.

**arXiv ID:** 2602.05407
</details>

<details>
<summary><strong>Learning Probabilistic Responsibility Allocations for Multi-Agent Interactions</strong> - Isaac Remy, Caleb Chang, Karen Leung - [[pdf]](https://arxiv.org/pdf/2604.13128)</summary>

**Abstract:** Human behavior in interactive settings is shaped not only by individual objectives but also by shared constraints with others, such as safety. Understanding how people allocate responsibility, i.e., how much one deviates from their desired policy to accommodate others, can inform the design of socially compliant and trustworthy autonomous systems. In this work, we introduce a method for learning a probabilistic responsibility allocation model that captures the multimodal uncertainty inherent in multi-agent interactions. Specifically, our approach leverages the latent space of a conditional variational autoencoder, combined with techniques from multi-agent trajectory forecasting, to learn a distribution over responsibility allocations conditioned on scene and agent context. Although ground-truth responsibility labels are unavailable, the model remains tractable by incorporating a differentiable optimization layer that maps responsibility allocations to induced controls, which are available. We evaluate our method on the INTERACTION driving dataset and demonstrate that it not only achieves strong predictive performance but also provides interpretable insights, through the lens of responsibility, into patterns of multi-agent interaction.

**arXiv ID:** 2604.13128
</details>

<details>
<summary><strong>Modality-Native Routing in Agent-to-Agent Networks: A Multimodal A2A Protocol Extension</strong> - Vasundra Srinivasan - [[pdf]](https://arxiv.org/pdf/2604.12213)</summary>

**Abstract:** Preserving multimodal signals across agent boundaries is necessary for accurate cross-modal reasoning, but it is not sufficient. We show that modality-native routing in Agent-to-Agent (A2A) networks improves task accuracy by 20 percentage points over text-bottleneck baselines, but only when the downstream reasoning agent can exploit the richer context that native routing preserves. An ablation replacing LLM-backed reasoning with keyword matching eliminates the accuracy gap entirely (36% vs. 36%), establishing a two-layer requirement: protocol-level routing must be paired with capable agent-level reasoning for the benefit to materialize.
We present MMA2A, an architecture layer atop A2A that inspects Agent Card capability declarations to route voice, image, and text parts in their native modality. On CrossModal-CS, a controlled 50-task benchmark with the same LLM backend, same tasks, and only the routing path varying, MMA2A achieves 52% task completion accuracy versus 32% for the text-bottleneck baseline (95% bootstrap CI on $\Delta$TCA: [8, 32] pp; McNemar's exact $p = 0.006$). Gains concentrate on vision-dependent tasks: product defect reports improve by +38.5 pp and visual troubleshooting by +16.7 pp. This accuracy gain comes at a $1.8\times$ latency cost from native multimodal processing. These results suggest that routing is a first-order design variable in multi-agent systems, as it determines the information available for downstream reasoning.

**arXiv ID:** 2604.12213
</details>

<details>
<summary><strong>Fairness in Multi-Agent Systems for Software Engineering: An SDLC-Oriented Rapid Review</strong> - Corey Yang-Smith, Ronnie de Souza Santos, Ahmad Abdellatif - [[pdf]](https://arxiv.org/pdf/2604.13103)</summary>

**Abstract:** Transformer-based large language models (LLMs) and multi-agent systems (MAS) are increasingly embedded across the software development lifecycle (SDLC), yet their fairness implications for developer-facing tools remain underexplored despite their growing role in shaping what code is written, reviewed, and released. We present a rapid review of recent work on fairness in MAS, emphasizing LLM-enabled settings and relevance to software engineering. Starting from an initial set of 350 papers, we screened and filtered the corpus for relevance, retaining 18 studies for final analysis. Across these 18 studies, fairness is framed as a combination of trustworthy AI principles, bias reduction across groups, and interactional dynamics in collectives, while evaluation spans accuracy metrics on bias benchmarks, demographic disparity measures, and emergent MAS-specific notions such as conformity and bias amplification. Reported harms include representational, quality-of-service, security and privacy, and governance failures, which we relate to SDLC stages where evidence is most and least developed. We identify three persistent gaps: (1) fragmented, rarely MAS-specific evaluation practices that limit comparability, (2) limited generalization due to simplified environments and narrow attribute coverage, and (3) scarce, weakly evaluated mitigation and governance mechanisms aligned to real software workflows. These findings suggest MAS fairness research is not yet ready to support deployable, fairness-assured software systems, motivating MAS-aware benchmarks, consistent protocols, and lifecycle-spanning governance.

**arXiv ID:** 2604.13103
</details>

<details>
<summary><strong>RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows</strong> - Kai Zhang, Corey D Barrett, Jangwon Kim, Lichao Sun, Tara Taghavi, Krishnaram Kenthapadi - [[pdf]](https://arxiv.org/pdf/2509.20490)</summary>

**Abstract:** Agentic systems offer a potential path to solve complex clinical tasks through collaboration among specialized agents, augmented by tool use and external knowledge bases. Nevertheless, for chest X-ray (CXR) interpretation, prevailing methods remain limited: (i) reasoning is frequently neither clinically interpretable nor aligned with guidelines, reflecting mere aggregation of tool outputs; (ii) multimodal evidence is insufficiently fused, yielding text-only rationales that are not visually grounded; and (iii) systems rarely detect or resolve cross-tool inconsistencies and provide no principled verification mechanisms. To bridge the above gaps, we present RadAgents, a multi-agent framework that couples clinical priors with task-aware multimodal reasoning and encodes a radiologist-style workflow into a modular, auditable pipeline. In addition, we integrate grounding and multimodal retrieval-augmentation to verify and resolve context conflicts, resulting in outputs that are more reliable, transparent, and consistent with clinical practice.

**arXiv ID:** 2509.20490
</details>

<details>
<summary><strong>CANVAS: Continuity-Aware Narratives via Visual Agentic Storyboarding</strong> - Ishani Mondal, Yiwen Song, Mihir Parmar, Palash Goyal, Jordan Boyd-Graber, Tomas Pfister, Yale Song - [[pdf]](https://arxiv.org/pdf/2604.13452)</summary>

**Abstract:** Long-form visual storytelling requires maintaining continuity across shots, including consistent characters, stable environments, and smooth scene transitions. While existing generative models can produce strong individual frames, they fail to preserve such continuity, leading to appearance changes, inconsistent backgrounds, and abrupt scene shifts. We introduce CANVAS (Continuity-Aware Narratives via Visual Agentic Storyboarding), a multi-agent framework that explicitly plans visual continuity in multi-shot narratives. CANVAS enforces coherence through character continuity, persistent background anchors, and location-aware scene planning for smooth transitions within the same setting We evaluate CANVAS on two storyboard generation benchmarks ST-BENCH and ViStoryBench and introduce a new challenging benchmark HardContinuityBench for long-range narrative consistency. CANVAS consistently outperforms the best-performing baseline, improving background continuity by 21.6%, character consistency by 9.6% and props consistency by 7.6%.

**arXiv ID:** 2604.13452
</details>

<details>
<summary><strong>Debate to Align: Reliable Entity Alignment through Two-Stage Multi-Agent Debate</strong> - Cunda Wang, Ziying Ma, Po Hu, Weihua Wang, Feilong Bao - [[pdf]](https://arxiv.org/pdf/2604.13551)</summary>

**Abstract:** Entity alignment (EA) aims to identify entities referring to the same real-world object across different knowledge graphs (KGs). Recent approaches based on large language models (LLMs) typically obtain entity embeddings through knowledge representation learning and use embedding similarity to identify an alignment-uncertain entity set. For each uncertain entity, a candidate entity set (CES) is then retrieved based on embedding similarity to support subsequent alignment reasoning and decision making. However, the reliability of the CES and the reasoning capability of LLMs critically affect the effectiveness of subsequent alignment decisions. To address this issue, we propose AgentEA, a reliable EA framework based on multi-agent debate. AgentEA first improves embedding quality through entity representation preference optimization, and then introduces a two-stage multi-role debate mechanism consisting of lightweight debate verification and deep debate alignment to progressively enhance the reliability of alignment decisions while enabling more efficient debate-based reasoning. Extensive experiments on public benchmarks under cross-lingual, sparse, large-scale, and heterogeneous settings demonstrate the effectiveness of AgentEA.

**arXiv ID:** 2604.13551
</details>

<details>
<summary><strong>$π$-Play: Multi-Agent Self-Play via Privileged Self-Distillation without External Data</strong> - Yaocheng Zhang, Yuanheng Zhu, Wenyue Chong, Songjun Tu, Qichao Zhang, Jiajun Chai, Xiaohan Wang, Wei Lin, Guojun Yin, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2604.14054)</summary>

**Abstract:** Deep search agents have emerged as a promising paradigm for addressing complex information-seeking tasks, but their training remains challenging due to sparse rewards, weak credit assignment, and limited labeled data. Self-play offers a scalable route to reduce data dependence, but conventional self-play optimizes students only through sparse outcome rewards, leading to low learning efficiency. In this work, we observe that self-play naturally produces a question construction path (QCP) during task generation, an intermediate artifact that captures the reverse solution process. This reveals a new source of privileged information for self-distillation: self-play can itself provide high-quality privileged context for the teacher model in a low-cost and scalable manner, without relying on human feedback or curated privileged information. Leveraging this insight, we propose Privileged Information Self-Play ($\pi$-Play), a multi-agent self-evolution framework. In $\pi$-Play, an examiner generates tasks together with their QCPs, and a teacher model leverages QCP as privileged context to densely supervise a student via self-distillation. This design transforms conventional sparse-reward self-play into a dense-feedback self-evolution loop. Extensive experiments show that data-free $\pi$-Play surpasses fully supervised search agents and improves evolutionary efficiency by 2-3$\times$ over conventional self-play.

**arXiv ID:** 2604.14054
</details>

<details>
<summary><strong>When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration</strong> - Yiping Li, Zhiyu An, Wan Du - [[pdf]](https://arxiv.org/pdf/2604.13349)</summary>

**Abstract:** Communication in Large Language Model (LLM)-based multi-agent systems is moving beyond discrete tokens to preserve richer context. Recent work such as LatentMAS enables agents to exchange latent messages through full key-value (KV) caches. However, full KV relay incurs high memory and communication cost. We adapt eviction-style KV compression to this setting and introduce Orthogonal Backfill (OBF) to mitigate information loss from hard eviction. OBF injects a low-rank orthogonal residual from discarded KV states into the retained KV states. We evaluate proposed method against full KV relay on nine standard benchmarks spanning mathematical reasoning, coding, and knowledge-intensive QA. It achieves performance comparable to full KV relay while reducing communication cost by 79.8%--89.4%. OBF further improves the performance and achieves the best results on 7 of the 9 benchmarks. This suggests that more information does not necessarily lead to better communication; preserving the most useful information matters more. Our codebase is publicly available on this https URL.

**arXiv ID:** 2604.13349
</details>

</details>

<details open>
<summary><h2>Other Agent Research (16 papers)</h2></summary>

<details>
<summary><strong>WebXSkill: Skill Learning for Autonomous Web Agents</strong> - Zhaoyang Wang, Qianhui Wu, Xuchao Zhang, Chaoyun Zhang, Wenlin Yao, Fazle Elahi Faisal, Baolin Peng, Si Qin, Suman Nath, Qingwei Lin, Chetan Bansal, Dongmei Zhang, Saravan Rajmohan, Jianfeng Gao, Huaxiu Yao - [[pdf]](https://arxiv.org/pdf/2604.13318)</summary>

**Abstract:** Autonomous web agents powered by large language models (LLMs) have shown promise in completing complex browser tasks, yet they still struggle with long-horizon workflows. A key bottleneck is the grounding gap in existing skill formulations: textual workflow skills provide natural language guidance but cannot be directly executed, while code-based skills are executable but opaque to the agent, offering no step-level understanding for error recovery or adaptation. We introduce WebXSkill, a framework that bridges this gap with executable skills, each pairing a parameterized action program with step-level natural language guidance, enabling both direct execution and agent-driven adaptation. WebXSkill operates in three stages: skill extraction mines reusable action subsequences from readily available synthetic agent trajectories and abstracts them into parameterized skills, skill organization indexes skills into a URL-based graph for context-aware retrieval, and skill deployment exposes two complementary modes, grounded mode for fully automated multi-step execution and guided mode where skills serve as step-by-step instructions that the agent follows with its native planning. On WebArena and WebVoyager, WebXSkill improves task success rate by up to 9.8 and 12.9 points over the baseline, respectively, demonstrating the effectiveness of executable skills for web agents. The code is publicly available at this https URL.

**arXiv ID:** 2604.13318
</details>

<details>
<summary><strong>ECM Contracts: Contract-Aware, Versioned, and Governable Capability Interfaces for Embodied Agents</strong> - Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li - [[pdf]](https://arxiv.org/pdf/2604.13097)</summary>

**Abstract:** Embodied agents increasingly rely on modular capabilities that can be installed, upgraded, composed, and governed at runtime. Prior work has introduced embodied capability modules (ECMs) as reusable units of embodied functionality, and recent research has explored their runtime governance and controlled evolution. However, a key systems question remains unresolved: how can ECMs be composed and released as a stable software ecosystem rather than as ad hoc skill bundles?
We present ECM Contracts, a contract-based interface model for embodied capability modules. Unlike conventional software interfaces that specify only input and output types, ECM Contracts encode six dimensions essential for embodied execution: functional signature, behavioral assumptions, resource requirements, permission boundaries, recovery semantics, and version compatibility. Based on this model, we introduce a compatibility framework for ECM installation, composition, and upgrade, enabling static and pre-deployment checks for type mismatches, dependency conflicts, policy violations, resource contention, and recovery incompatibilities.
We further propose a release discipline for embodied capabilities, including version-aware compatibility classes, deprecation rules, migration constraints, and policy-sensitive upgrade checks. We implement a prototype ECM registry, resolver, and contract checker, and evaluate the approach on modular embodied tasks in a robotics runtime setting. Results show that contract-aware composition substantially reduces unsafe or invalid module combinations, and that contract-guided release checks improve upgrade safety and rollback readiness compared with schema-only or ad hoc baselines.
Our findings suggest that stable embodied software ecosystems require more than modular packaging: they require explicit contracts that connect capability composition, governance, and evolution.

**arXiv ID:** 2604.13097
</details>

<details>
<summary><strong>CCCE: A Continuous Code Calibration Engine for Autonomous Enterprise Codebase Maintenance via Knowledge Graph Traversal and Adaptive Decision Gating</strong> - Santhosh Kusuma Kumar Parimi - [[pdf]](https://arxiv.org/pdf/2604.13102)</summary>

**Abstract:** Enterprise software organizations face an escalating challenge in maintaining the integrity, security, and freshness of codebases that span hundreds of repositories, multiple programming languages, and thousands of interdependent packages. Existing approaches to codebase maintenance -- including static analysis, software composition analysis (SCA), and dependency management tools -- operate in isolation, address only narrow subsets of maintenance concerns, and require substantial manual intervention to propagate changes across interconnected systems. We present the Continuous Code Calibration Engine (CCCE), an event-driven, AI-agentic system that autonomously maintains enterprise codebases throughout the Software Development Life Cycle (SDLC). The CCCE introduces three key technical innovations: (1) a dynamic knowledge graph with bidirectional traversal algorithms that simultaneously compute forward impact propagation and backward test adequacy analysis; (2) an adaptive multi-stage gating framework that classifies calibration actions into four risk tiers using learned risk-confidence scoring rather than static rules; and (3) a multi-model continuous learning architecture operating at multiple temporal scales to refine calibration strategies, risk models, and organizational policies from operational feedback. We formalize the system's graph model, traversal algorithms, and decision logic, and demonstrate through three representative enterprise scenarios that the CCCE reduces mean time to remediation by enabling coordinated, cross-repository calibrations with human-in-the-loop (HITL) oversight where appropriate. The system generates atomic, semantically verified patches with progressive validation and intelligent rollback capabilities, providing end-to-end traceability from triggering events through calibration execution and outcome learning.

**arXiv ID:** 2604.13102
</details>

<details>
<summary><strong>Formal Architecture Descriptors as Navigation Primitives for AI Coding Agents</strong> - Ruoqi Jin - [[pdf]](https://arxiv.org/pdf/2604.13108)</summary>

**Abstract:** AI coding agents spend a substantial fraction of their tool calls on undirected codebase exploration. We investigate whether providing agents with formal architecture descriptors can reduce this navigational overhead. We present three complementary studies. First, a controlled experiment (24 code localization tasks x 4 conditions, Claude Sonnet 4.6, temperature=0) demonstrates that architecture context reduces navigation steps by 33-44% (Wilcoxon p=0.009, Cohen's d=0.92), with no significant format difference detected across S-expression, JSON, YAML, and Markdown. Second, an artifact-vs-process experiment (15 tasks x 3 conditions) demonstrates that an automatically generated descriptor achieves 100% accuracy versus 80% blind (p=0.002, d=1.04), proving direct navigational value independent of developer self-clarification. Third, an observational field study across 7,012 Claude Code sessions shows 52% reduction in agent behavioral variance. A writer-side experiment (96 generation runs, 96 error injections) reveals critical failure mode differences: JSON fails atomically, YAML silently corrupts 50% of errors, S-expressions detect all structural completeness errors. We propose this http URL, an S-expression architecture descriptor, and open-source the Forge toolkit.

**arXiv ID:** 2604.13108
</details>

<details>
<summary><strong>Applying an Agentic Coding Tool for Improving Published Algorithm Implementations</strong> - Worasait Suwannik - [[pdf]](https://arxiv.org/pdf/2604.13109)</summary>

**Abstract:** We present a two-stage pipeline for AI-assisted improvement of published algorithm implementations. In the first stage, a large language model with research capabilities identifies recently published algorithms satisfying explicit experimental criteria. In the second stage, Claude Code is given a prompt to reproduce the reported baseline and then iterate an improvement process. We apply this pipeline to published algorithm implementations spanning multiple research domains. Claude Code reported that all eleven experiments yielded improvements. Each improvement could be achieved within a single working day. We analyse the human contributions that remain indispensable, including selecting the target, verifying experimental validity, assessing novelty and impact, providing computational resources, and writing with appropriate AI-use disclosure. Finally, we discuss implications for peer review and academic publishing.

**arXiv ID:** 2604.13109
</details>

<details>
<summary><strong>Golden Handcuffs make safer AI agents</strong> - Aram Ebtekar, Michael K. Cohen - [[pdf]](https://arxiv.org/pdf/2604.13609)</summary>

**Abstract:** Reinforcement learners can attain high reward through novel unintended strategies. We study a Bayesian mitigation for general environments: we expand the agent's subjective reward range to include a large negative value $-L$, while the true environment's rewards lie in $[0,1]$. After observing consistently high rewards, the Bayesian policy becomes risk-averse to novel schemes that plausibly lead to $-L$. We design a simple override mechanism that yields control to a safe mentor whenever the predicted value drops below a fixed threshold. We prove two properties of the resulting agent: (i) Capability: using mentor-guided exploration with vanishing frequency, the agent attains sublinear regret against its best mentor. (ii) Safety: no decidable low-complexity predicate is triggered by the optimizing policy before it is triggered by a mentor.

**arXiv ID:** 2604.13609
</details>

<details>
<summary><strong>HiVLA: A Visual-Grounded-Centric Hierarchical Embodied Manipulation System</strong> - Tianshuo Yang, Guanyu Chen, Yutian Chen, Zhixuan Liang, Yitian Liu, Zanxin Chen, Chunpu Xu, Haotian Liang, Jiangmiao Pang, Yao Mu, Ping Luo - [[pdf]](https://arxiv.org/pdf/2604.14125)</summary>

**Abstract:** While end-to-end Vision-Language-Action (VLA) models offer a promising paradigm for robotic manipulation, fine-tuning them on narrow control data often compromises the profound reasoning capabilities inherited from their base Vision-Language Models (VLMs). To resolve this fundamental trade-off, we propose HiVLA, a visual-grounded-centric hierarchical framework that explicitly decouples high-level semantic planning from low-level motor control. In high-level part, a VLM planner first performs task decomposition and visual grounding to generate structured plans, comprising a subtask instruction and a precise target bounding box. Then, to translate this plan into physical actions, we introduce a flow-matching Diffusion Transformer (DiT) action expert in low-level part equipped with a novel cascaded cross-attention mechanism. This design sequentially fuses global context, high-resolution object-centric crops and skill semantics, enabling the DiT to focus purely on robust execution. Our decoupled architecture preserves the VLM's zero-shot reasoning while allowing independent improvement of both components. Extensive experiments in simulation and the real world demonstrate that HiVLA significantly outperforms state-of-the-art end-to-end baselines, particularly excelling in long-horizon skill composition and the fine-grained manipulation of small objects in cluttered scenes.

**arXiv ID:** 2604.14125
</details>

<details>
<summary><strong>Agentic AI Optimisation (AAIO): what it is, how it works, why it matters, and how to deal with it</strong> - Luciano Floridi, Carlotta Buttaboni, Nicolas Gentler, Emmie Hine, Jessica Morley, Claudio Novelli, Tyler Schroder - [[pdf]](https://arxiv.org/pdf/2504.12482)</summary>

**Abstract:** The emergence of Agentic Artificial Intelligence (AAI) systems capable of independently initiating digital interactions necessitates a new optimisation paradigm designed explicitly for seamless agent-platform interactions. This article introduces Agentic AI Optimisation (AAIO) as an essential methodology for ensuring effective integration between websites and agentic AI systems. Like how Search Engine Optimisation (SEO) has shaped digital content discoverability, AAIO can define interactions between autonomous AI agents and online platforms. By examining the mutual interdependency between website optimisation and agentic AI success, the article highlights the virtuous cycle that AAIO can create. It further explores the governance, ethical, legal, and social implications (GELSI) of AAIO, emphasising the necessity of proactive regulatory frameworks to mitigate potential negative impacts. The article concludes by affirming AAIO's essential role as part of a fundamental digital infrastructure in the era of autonomous digital agents, advocating for equitable and inclusive access to its benefits.

**arXiv ID:** 2504.12482
</details>

<details>
<summary><strong>GraphScout: Empowering Large Language Models with Intrinsic Exploration Ability for Agentic Graph Reasoning</strong> - Yuchen Ying, Weiqi Jiang, Tongya Zheng, Yu Wang, Shunyu Liu, Kaixuan Chen, Mingli Song - [[pdf]](https://arxiv.org/pdf/2603.01410)</summary>

**Abstract:** Knowledge graphs provide structured and reliable information for many real-world applications, motivating increasing interest in combining large language models (LLMs) with graph-based retrieval to improve factual grounding. Recent Graph-based Retrieval-Augmented Generation (GraphRAG) methods therefore introduce iterative interaction between LLMs and knowledge graphs to enhance reasoning capability. However, existing approaches typically depend on manually designed guidance and interact with knowledge graphs through a limited set of predefined tools, which substantially constrains graph exploration. To address these limitations, we propose GraphScout, a training-centric agentic graph reasoning framework equipped with more flexible graph exploration tools. GraphScout enables models to autonomously interact with knowledge graphs to synthesize structured training data which are then used to post-train LLMs, thereby internalizing agentic graph reasoning ability without laborious manual annotation or task curation. Extensive experiments across five knowledge-graph domains show that a small model (e.g., Qwen3-4B) augmented with GraphScout outperforms baseline methods built on leading LLMs (e.g., Qwen-Max) by an average of 16.7\% while requiring significantly fewer inference tokens. Moreover, GraphScout exhibits robust cross-domain transfer performance. Our code will be made publicly available~\footnote{this https URL}.

**arXiv ID:** 2603.01410
</details>

<details>
<summary><strong>A Lightweight, Transferable, and Self-Adaptive Framework for Intelligent DC Arc-Fault Detection in Photovoltaic Systems</strong> - Xiaoke Yang, Long Gao, Haoyu He, Hanyuan Hang, Qi Liu, Shuai Zhao, Qiantu Tuo, Rui Li - [[pdf]](https://arxiv.org/pdf/2603.25749)</summary>

**Abstract:** Arc-fault circuit interrupters (AFCIs) are essential for mitigating fire hazards in residential photovoltaic (PV) systems, yet achieving reliable DC arc-fault detection under real-world conditions remains challenging. Spectral interference from inverter switching, hardware heterogeneity, operating-condition drift, and environmental noise collectively compromise conventional AFCI solutions. This paper proposes a lightweight, transferable, and self-adaptive learning-driven framework (LD-framework) for intelligent DC arc-fault detection. At the device level, LD-Spec learns compact spectral representations enabling efficient on-device inference and near-perfect arc discrimination. Across heterogeneous inverter platforms, LD-Align performs cross-hardware representation alignment to ensure robust detection despite hardware-induced distribution shifts. To address long-term evolution, LD-Adapt introduces a cloud-edge collaborative self-adaptive updating mechanism that detects unseen operating regimes and performs controlled model evolution. Extensive experiments involving over 53,000 labeled samples demonstrate near-perfect detection, achieving 0.9999 accuracy and 0.9996 F1-score. Across diverse nuisance-trip-prone conditions, including inverter start-up, grid transitions, load switching, and harmonic disturbances, the method achieves a 0% false-trip rate. Cross-hardware transfer shows reliable adaptation using only 0.5%-1% labeled target data while preserving source performance. Field adaptation experiments demonstrate recovery of detection precision from 21% to 95% under previously unseen conditions. These results indicate that the LD-framework enables a scalable, deployment-oriented AFCI solution maintaining highly reliable detection across heterogeneous devices and long-term operation.

**arXiv ID:** 2603.25749
</details>

<details>
<summary><strong>[COMP25] The Automated Negotiating Agents Competition (ANAC) 2025 Challenges and Results</strong> - Reyhan Aydoğan, Tim Baarslag, Tamara C.P. Florijn, Katsuhide Fujita, Catholijn M. Jonker, Yasser Mohammad - [[pdf]](https://arxiv.org/pdf/2604.13914)</summary>

**Abstract:** This paper presents the primary research challenges and key findings from the 15th International Automated Negotiating Agents Competition (ANAC 2025), one of the official competitions of IJCAI 2025. We focus on two critical domains: multi-deal negotiations and the development of agents capable of concurrent negotiation within complex supply chain management environments. Furthermore, this work analyzes the results of the competition and outlines strategic directions for future iterations.

**arXiv ID:** 2604.13914
</details>

<details>
<summary><strong>Bridging Protocol and Production: Design Patterns for Deploying AI Agents with Model Context Protocol</strong> - Vasundra Srinivasan - [[pdf]](https://arxiv.org/pdf/2603.13417)</summary>

**Abstract:** The Model Context Protocol (MCP) standardizes how AI agents discover and invoke external tools, with over 10,000 active servers and 97 million monthly SDK downloads as of early 2026. Yet MCP does not yet standardize how agents safely operate those tools at production scale. Three protocol-level primitives remain missing: identity propagation, adaptive tool budgeting, and structured error semantics. This paper identifies these gaps through field lessons from an enterprise deployment of an AI agent platform integrated with a major cloud provider's MCP servers (client name redacted). We propose three mechanisms to fill them: (1) the Context-Aware Broker Protocol (CABP), which extends JSON-RPC with identity-scoped request routing via a six-stage broker pipeline; (2) Adaptive Timeout Budget Allocation (ATBA), which frames sequential tool invocation as a budget allocation problem over heterogeneous latency distributions; and (3) the Structured Error Recovery Framework (SERF), which provides machine-readable failure semantics that enable deterministic agent self-correction. We organize production failure modes into five design dimensions (server contracts, user context, timeouts, errors, and observability), document concrete failure vignettes, and present a production readiness checklist. All three algorithms are formalized as testable hypotheses with reproducible experimental methodology. Field observations demonstrate that while MCP provides a solid protocol foundation, reliable agent tool integration requires infrastructure-level mechanisms that the specification does not yet address.

**arXiv ID:** 2603.13417
</details>

<details>
<summary><strong>Robust Energy-Aware Routing for Air-Ground Cooperative Multi-UAV Delivery in Wind-Uncertain Environments</strong> - Tianshun Li, Hongliang Lu, Yanggang Sheng, Zhongzhen Wang, Haoang Li, Xinhu Zheng - [[pdf]](https://arxiv.org/pdf/2604.13441)</summary>

**Abstract:** Ensuring energy feasibility under wind uncertainty is critical for the safety and reliability of UAV delivery missions. In realistic truck-drone logistics systems, UAVs must deliver parcels and safely return under time-varying wind conditions that are only partially observable during flight. However, most existing routing approaches assume static or deterministic energy models, making them unreliable in dynamic wind environments. We propose Battery-Efficient Routing (BER), an online risk-sensitive planning framework for wind-sensitive truck-assisted UAV delivery. The problem is formulated as routing on a time dependent energy graph whose edge costs evolve according to wind-induced aerodynamic effects. BER continuously evaluates return feasibility while balancing instantaneous energy expenditure and uncertainty-aware risk. The approach is embedded in a hierarchical aerial-ground delivery architecture that combines task allocation, routing, and decentralized trajectory execution. Extensive simulations on synthetic ER graphs generated in Unreal Engine environments and quasi-real wind logs demonstrate that BER significantly improves mission success rates and reduces wind-induced failures compared with static and greedy baselines. These results highlight the importance of integrating real-time energy budgeting and environmental awareness for UAV delivery planning under dynamic wind conditions.

**arXiv ID:** 2604.13441
</details>

<details>
<summary><strong>Safe and Nonconservative Contingency Planning for Autonomous Vehicles via Online Learning-Based Reachable Set Barriers</strong> - Rui Yang, Lei Zheng, Shuzhi Sam Ge, Jun Ma - [[pdf]](https://arxiv.org/pdf/2509.07464)</summary>

**Abstract:** Autonomous vehicles must navigate dynamically uncertain environments while balancing safety and efficiency. This challenge is exacerbated by unpredictable human-driven vehicle (HV) behaviors and perception inaccuracies, necessitating planners that adapt to evolving uncertainties while maintaining safe trajectories. Overly conservative planning degrades driving efficiency, while deterministic methods risk failure in unexpected scenarios. To address these issues, we propose a real-time contingency trajectory optimization framework. Our method employs event-triggered online learning of HV control-intent sets to dynamically quantify multimodal HV uncertainties and incrementally refine their forward reachable sets (FRSs). Crucially, we enforce invariant safety through FRS-based barrier constraints that ensure safety without reliance on accurate trajectory prediction. These constraints are seamlessly embedded in contingency trajectory optimization and solved efficiently through consensus alternating direction method of multipliers (ADMM). The system continuously adapts to HV behavioral uncertainties, preserving feasibility and safety without excessive conservatism. High-fidelity simulations on highway and urban scenarios, along with a series of real-world experiments, demonstrate significant improvements in driving efficiency and passenger comfort while maintaining safety under uncertainty. The project page is available at this https URL.

**arXiv ID:** 2509.07464
</details>

<details>
<summary><strong>LEO-RobotAgent: A General-purpose Robotic Agent for Language-driven Embodied Operator</strong> - Lihuang Chen, Xiangyu Luo, Jun Meng - [[pdf]](https://arxiv.org/pdf/2512.10605)</summary>

**Abstract:** We propose LEO-RobotAgent, a general-purpose language-driven intelligent agent framework for robots. Under this framework, LLMs can operate different types of robots to complete unpredictable complex tasks across various scenarios. This framework features strong generalization, robustness, and efficiency. The application-level system built around it can fully enhance bidirectional human-robot intent understanding and lower the threshold for human-robot interaction. Regarding robot task planning, the vast majority of existing studies focus on the application of large models in single-task scenarios and for single robot types. These algorithms often have complex structures and lack generalizability. Thus, the proposed LEO-RobotAgent framework is designed with a streamlined structure as much as possible, enabling large models to independently think, plan, and act within this clear framework. We provide a modular and easily registrable toolset, allowing large models to flexibly call various tools to meet different requirements. Meanwhile, the framework incorporates a human-robot interaction mechanism, enabling the algorithm to collaborate with humans like a partner. Experiments have verified that this framework can be easily adapted to mainstream robot platforms including unmanned aerial vehicles (UAVs), robotic arms, and wheeled robot, and efficiently execute a variety of carefully designed tasks with different complexity levels. Our code is available at this https URL.

**arXiv ID:** 2512.10605
</details>

<details>
<summary><strong>Acts of Configuration: Rethinking Provenance, Temporality and Legitimacy in Post-Mortem Agents</strong> - Kellie Yu Hui Sim, Pin Sym Foong, Darryl Lim, John-Henry Lim, Kenny Tsu Wei Choo - [[pdf]](https://arxiv.org/pdf/2604.13996)</summary>

**Abstract:** Work on persona-persistent post-mortem agents typically frames design around a life/death binary. This framing neglects a consequential yet under-theorised condition: when individuals remain alive but have impaired decisional capacity. Drawing on a multi-phase workshop in which participants trained and reflected on an AI agent for Advance Care Planning, we examined how people reason about agentic delegation post-capacity loss. Initially, participants favoured bounded agents grounded in first-party authorship and representational fidelity over autonomous or evolving stand-ins. However, temporality introduced novel ideas like adjacent use driven by persona persistence over functional expansion: agents should evolve while users retain capacity, remain static once capacity is lost, but somehow inform adjacent post-mortem uses. We discuss the implications of these findings and propose that the configuration of agents for post-capacity use reshapes our understanding of provenance, temporality, and legitimacy for post-mortem agents.

**arXiv ID:** 2604.13996
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (30 papers)</h2></summary>

<details>
<summary><strong>RiskWebWorld: A Realistic Interactive Benchmark for GUI Agents in E-commerce Risk Management</strong> - Renqi Chen, Zeyin Tao, Jianming Guo, Jing Wang, Zezhou Xu, Jingzhe Zhu, Qingqing Sun, Tianyi Zhang, Shuai Chen - [[pdf]](https://arxiv.org/pdf/2604.13531)</summary>

**Abstract:** Graphical User Interface (GUI) agents show strong capabilities for automating web tasks, but existing interactive benchmarks primarily target benign, predictable consumer environments. Their effectiveness in high-stakes, investigative domains such as authentic e-commerce risk management remains underexplored. To bridge this gap, we present RiskWebWorld, the first highly realistic interactive benchmark for evaluating GUI agents in e-commerce risk management. RiskWebWorld features 1,513 tasks sourced from production risk-control pipelines across 8 core domains, and captures the authentic challenges of risk operations on uncooperative websites, partially environmental hijackments. To support scalable evaluation and agentic reinforcement learning (RL), we further build a Gymnasium-compliant infrastructure that decouples policy planning from environment mechanics. Our evaluation across diverse models reveals a dramatic capability gap: top-tier generalist models achieve 49.1% success, while specialized open-weights GUI models lag at near-total failure. This highlights that foundation model scale currently matters more than zero-shot interface grounding in long-horizon professional tasks. We also demonstrate the viability of our infrastructure through agentic RL, which improves open-source models by 16.2%. These results position RiskWebWorld as a practical testbed for developing robust digital workers.

**arXiv ID:** 2604.13531
</details>

<details>
<summary><strong>Hierarchical Reinforcement Learning with Runtime Safety Shielding for Power Grid Operation</strong> - Gitesh Malik - [[pdf]](https://arxiv.org/pdf/2604.14032)</summary>

**Abstract:** Reinforcement learning has shown promise for automating power-grid operation tasks such as topology control and congestion management. However, its deployment in real-world power systems remains limited by strict safety requirements, brittleness under rare disturbances, and poor generalization to unseen grid topologies. In safety-critical infrastructure, catastrophic failures cannot be tolerated, and learning-based controllers must operate within hard physical constraints.
This paper proposes a safety-constrained hierarchical control framework for power-grid operation that explicitly decouples long-horizon decision-making from real-time feasibility enforcement. A high-level reinforcement learning policy proposes abstract control actions, while a deterministic runtime safety shield filters unsafe actions using fast forward simulation. Safety is enforced as a runtime invariant, independent of policy quality or training distribution.
The proposed framework is evaluated on the Grid2Op benchmark suite under nominal conditions, forced line-outage stress tests, and zero-shot deployment on the ICAPS 2021 large-scale transmission grid without retraining. Results show that flat reinforcement learning policies are brittle under stress, while safety-only methods are overly conservative. In contrast, the proposed hierarchical and safety-aware approach achieves longer episode survival, lower peak line loading, and robust zero-shot generalization to unseen grids.
These results indicate that safety and generalization in power-grid control are best achieved through architectural design rather than increasingly complex reward engineering, providing a practical path toward deployable learning-based controllers for real-world energy systems.

**arXiv ID:** 2604.14032
</details>

<details>
<summary><strong>Integration of Deep Reinforcement Learning and Agent-based Simulation to Explore Strategies Counteracting Information Disorder</strong> - Luigi Lomasto, Andrea Camoia, Alfonso Guarino, Nicola Lettieri, Delfina Malandrino, Rocco Zaccagnino - [[pdf]](https://arxiv.org/pdf/2604.13047)</summary>

**Abstract:** In recent years, the spread of fake news has triggered a growing interest in Information Disorders (ID) on social media, a phenomenon that has become a focal point of research across fields ranging from complexity theory and computer science to cognitive sciences. Overall, such a body of research can be traced back to two main approaches. On the one hand, there are works focused on exploiting data mining to analyze the content of news and related metadata data-driven approach; on the other hand, works are aiming at making sense of the phenomenon at hand and their evolution using explicit simulation models model-driven approach). In this paper, we integrate these approaches to explore strategies for counteracting IDs. Heading in this direction, we put together: i. an Agent-Based model to simulate in a scientifically sound way both complex fake news dynamics and the effects produced by containment strategies therein; ii. Deep Reinforcement Learning to learn the strategies that can better mitigate the spread of misinformation. The outcomes of our work unfold on different levels. From a substantive point of view, the results of preliminary experiments started providing interesting cues about the conditions under which given policies can mitigate the spread of misinformation. From a technical and methodological point of view, we scratched the surface of promising and worthy research topics like the integration of social simulation and artificial intelligence and the enhancement of social science simulation environments.

**arXiv ID:** 2604.13047
</details>

<details>
<summary><strong>Alignment as Institutional Design: From Behavioral Correction to Transaction Structure in Intelligent Systems</strong> - Rui Chai - [[pdf]](https://arxiv.org/pdf/2604.13079)</summary>

**Abstract:** Current AI alignment paradigms rely on behavioral correction: external supervisors (e.g., RLHF) observe outputs, judge against preferences, and adjust parameters. This paper argues that behavioral correction is structurally analogous to an economy without property rights, where order requires perpetual policing and does not scale. Drawing on institutional economics (Coase, Alchian, Cheung), capability mutual exclusivity, and competitive cost discovery, we propose alignment as institutional design: the designer specifies internal transaction structures (module boundaries, competition topologies, cost-feedback loops) such that aligned behavior emerges as the lowest-cost strategy for each component. We identify three irreducible levels of human intervention (structural, parametric, monitorial) and show that this framework transforms alignment from a behavioral control problem into a political-economy problem. No institution eliminates self-interest or guarantees optimality; the best design makes misalignment costly, detectable, and correctable. We conclude that the proper goal is institutional robustness-a dynamic, self-correcting process under human oversight, not perfection. This work provides the normative foundation for the Wuxing resource-competition mechanisms in companion papers.
Keywords: AI alignment, institutional design, transaction costs, property rights, resource competition, behavioral correction, RLHF, cost truthfulness, modular architecture, correctable alignment

**arXiv ID:** 2604.13079
</details>

<details>
<summary><strong>Adaptive Memory Crystallization for Autonomous AI Agent Learning in Dynamic Environments</strong> - Rajat Khanda, Mohammad Baqar Sambuddha Chakrabarti, Satyasaran Changdar - [[pdf]](https://arxiv.org/pdf/2604.13085)</summary>

**Abstract:** Autonomous AI agents operating in dynamic environments face a persistent challenge: acquiring new capabilities without erasing prior knowledge. We present Adaptive Memory Crystallization (AMC), a memory architecture for progressive experience consolidation in continual reinforcement learning.
AMC is conceptually inspired by the qualitative structure of synaptic tagging and capture (STC) theory, the idea that memories transition through discrete stability phases, but makes no claim to model the underlying molecular or synaptic mechanisms.
AMC models memory as a continuous crystallization process in which experiences migrate from plastic to stable states according to a multi-objective utility signal. The framework introduces a three-phase memory hierarchy (Liquid--Glass--Crystal) governed by an Itô stochastic differential equation (SDE) whose population-level behavior is captured by an explicit Fokker--Planck equation admitting a closed-form Beta stationary distribution.
We provide proofs of: (i) well-posedness and global convergence of the crystallization SDE to a unique Beta stationary distribution; (ii) exponential convergence of individual crystallization states to their fixed points, with explicit rates and variance bounds; and (iii) end-to-end Q-learning error bounds and matching memory-capacity lower bounds that link SDE parameters directly to agent performance.
Empirical evaluation on Meta-World MT50, Atari 20-game sequential learning, and MuJoCo continual locomotion consistently shows improvements in forward transfer (+34--43\% over the strongest baseline), reductions in catastrophic forgetting (67--80\%), and a 62\% decrease in memory footprint.

**arXiv ID:** 2604.13085
</details>

<details>
<summary><strong>GeoVision-Enabled Digital Twin for Hybrid Autonomous-Teleoperated Medical Responses</strong> - Parham Kebria, Soheil Sabri, Laura J Brattain - [[pdf]](https://arxiv.org/pdf/2604.13248)</summary>

**Abstract:** Remote medical response systems are increasingly being deployed to support emergency care in disaster-affected and infrastructure-limited environments. Enabled by GeoVision capabilities, this paper presents a Digital Twin architecture for hybrid autonomous-teleoperated medical response systems. The proposed framework integrates perception and adaptive navigation with a Digital Twin, synchronized in real-time, that mirrors system states, environmental dynamics, patient conditions, and mission objectives. Unlike traditional ground control interfaces, the Digital Twin provides remote clinical and operational users with an intuitive, continuously updated virtual representation of the platform and its operational context, enabling enhanced situational awareness and informed decision-making.

**arXiv ID:** 2604.13248
</details>

<details>
<summary><strong>From Prediction to Justification: Aligning Sentiment Reasoning with Human Rationale via Reinforcement Learning</strong> - Shihao Zhang, Ziwei Wang, Jie Zhou, Yulan Wu, Qin Chen, Zhikai Lei, Liyang Yu, Liang Dou, Liang He - [[pdf]](https://arxiv.org/pdf/2604.13398)</summary>

**Abstract:** While Aspect-based Sentiment Analysis (ABSA) systems have achieved high accuracy in identifying sentiment polarities, they often operate as "black boxes," lacking the explicit reasoning capabilities characteristic of human affective cognition. Humans do not merely categorize sentiment; they construct causal explanations for their judgments. To bridge this gap, we propose ABSA-R1, a large language model framework designed to mimic this ``reason-before-predict" cognitive process. By leveraging reinforcement learning (RL), ABSA-R1 learns to articulate the why behind the what, generating natural language justifications that ground its sentiment predictions. We introduce a Cognition-Aligned Reward Model (formerly sentiment-aware reward model) that enforces consistency between the generated reasoning path and the final emotional label. Furthermore, inspired by metacognitive monitoring, we implement a performance-driven rejection sampling strategy that selectively targets hard cases where the model's internal reasoning is uncertain or inconsistent. Experimental results on four benchmarks demonstrate that equipping models with this explicit reasoning capability not only enhances interpretability but also yields superior performance in sentiment classification and triplet extraction compared to non-reasoning baselines.

**arXiv ID:** 2604.13398
</details>

<details>
<summary><strong>Chain of Uncertain Rewards with Large Language Models for Reinforcement Learning</strong> - Shentong Mo - [[pdf]](https://arxiv.org/pdf/2604.13504)</summary>

**Abstract:** Designing effective reward functions is a cornerstone of reinforcement learning (RL), yet it remains a challenging and labor-intensive process due to the inefficiencies and inconsistencies inherent in traditional methods. Existing methods often rely on extensive manual design and evaluation steps, which are prone to redundancy and overlook local uncertainties at intermediate decision points. To address these challenges, we propose the Chain of Uncertain Rewards (CoUR), a novel framework that integrates large language models (LLMs) to streamline reward function design and evaluation in RL environments. Specifically, our CoUR introduces code uncertainty quantification with a similarity selection mechanism that combines textual and semantic analyses to identify and reuse the most relevant reward function components. By reducing redundant evaluations and leveraging Bayesian optimization on decoupled reward terms, CoUR enables a more efficient and robust search for optimal reward feedback. We comprehensively evaluate CoUR across nine original environments from IsaacGym and all 20 tasks from the Bidexterous Manipulation benchmark. The experimental results demonstrate that CoUR not only achieves better performance but also significantly lowers the cost of reward evaluations.

**arXiv ID:** 2604.13504
</details>

<details>
<summary><strong>Jump-Start Reinforcement Learning with Vision-Language-Action Regularization</strong> - Angelo Moroncelli, Roberto Zanetti, Marco Maccarini, Loris Roveda - [[pdf]](https://arxiv.org/pdf/2604.13733)</summary>

**Abstract:** Reinforcement learning (RL) enables high-frequency, closed-loop control for robotic manipulation, but scaling to long-horizon tasks with sparse or imperfect rewards remains difficult due to inefficient exploration and poor credit assignment. Vision-Language-Action (VLA) models leverage large-scale multimodal pretraining to provide generalist, task-level reasoning, but current limitations hinder their direct use in fast and precise manipulation. In this paper, we propose Vision-Language-Action Jump-Starting (VLAJS), a method that bridges sparse VLA guidance with on-policy RL to improve exploration and learning efficiency. VLAJS treats VLAs as transient sources of high-level action suggestions that bias early exploration and improve credit assignment, while preserving the high-frequency, state-based control of RL. Our approach augments Proximal Policy Optimization (PPO) with a directional action-consistency regularization that softly aligns the RL agent's actions with VLA guidance during early training, without enforcing strict imitation, requiring demonstrations, or relying on continuous teacher queries. VLA guidance is applied sparsely and annealed over time, allowing the agent to adapt online and ultimately surpass the guiding policy. We evaluate VLAJS on six challenging manipulation tasks: lifting, pick-and-place, peg reorientation, peg insertion, poking, and pushing in simulation, and validate a subset on a real Franka Panda robot. VLAJS consistently outperforms PPO and distillation-style baselines in sample efficiency, reducing required environment interactions by over 50% in several tasks. Real-world experiments demonstrate zero-shot sim-to-real transfer and robust execution under clutter, object variation, and external perturbations.

**arXiv ID:** 2604.13733
</details>

<details>
<summary><strong>Soft $Q(λ)$: A multi-step off-policy method for entropy regularised reinforcement learning using eligibility traces</strong> - Pranav Mahajan, Ben Seymour - [[pdf]](https://arxiv.org/pdf/2604.13780)</summary>

**Abstract:** Soft Q-learning has emerged as a versatile model-free method for entropy-regularised reinforcement learning, optimising for returns augmented with a penalty on the divergence from a reference policy. Despite its success, the multi-step extensions of soft Q-learning remain relatively unexplored and limited to on-policy action sampling under the Boltzmann policy. In this brief research note, we first present a formal $n$-step formulation for soft Q-learning and then extend this framework to the fully off-policy case by introducing a novel Soft Tree Backup operator. Finally, we unify these developments into Soft $Q(\lambda)$, an elegant online, off-policy, eligibility trace framework that allows for efficient credit assignment under arbitrary behaviour policies. Our derivations propose a model-free method for learning entropy-regularised value functions that can be utilised in future empirical experiments.

**arXiv ID:** 2604.13780
</details>

<details>
<summary><strong>HINTBench: Horizon-agent Intrinsic Non-attack Trajectory Benchmark</strong> - Jiacheng Wang, Jinchang Hou, Fabian Wang, Ping Jian, Chenfu Bao, Zhonghou Lv - [[pdf]](https://arxiv.org/pdf/2604.13954)</summary>

**Abstract:** Existing agent-safety evaluation has focused mainly on externally induced risks. Yet agents may still enter unsafe trajectories under benign conditions. We study this complementary but underexplored setting through the lens of \emph{intrinsic} risk, where intrinsic failures remain latent, propagate across long-horizon execution, and eventually lead to high-consequence outcomes. To evaluate this setting, we introduce \emph{non-attack intrinsic risk auditing} and present \textbf{HINTBench}, a benchmark of 629 agent trajectories (523 risky, 106 safe; 33 steps on average) supporting three tasks: risk detection, risk-step localization, and intrinsic failure-type identification. Its annotations are organized under a unified five-constraint taxonomy. Experiments reveal a substantial capability gap: strong LLMs perform well on trajectory-level risk detection, but their performance drops to below 35 Strict-F1 on risk-step localization, while fine-grained failure diagnosis proves even harder. Existing guard models transfer poorly to this setting. These findings establish intrinsic risk auditing as an open challenge for agent safety.

**arXiv ID:** 2604.13954
</details>

<details>
<summary><strong>From $P(y|x)$ to $P(y)$: Investigating Reinforcement Learning in Pre-train Space</strong> - Yuqiao Tan, Minzheng Wang, Bo Liu, Zichen Liu, Tian Liang, Shizhu He, Jun Zhao, Kang Liu - [[pdf]](https://arxiv.org/pdf/2604.14142)</summary>

**Abstract:** While reinforcement learning with verifiable rewards (RLVR) significantly enhances LLM reasoning by optimizing the conditional distribution P(y|x), its potential is fundamentally bounded by the base model's existing output distribution. Optimizing the marginal distribution P(y) in the Pre-train Space addresses this bottleneck by encoding reasoning ability and preserving broad exploration capacity. Yet, conventional pre-training relies on static corpora for passive learning, leading to a distribution shift that hinders targeted reasoning enhancement. In this paper, we introduce PreRL (Pre-train Space RL), which applies reward-driven online updates directly to P(y). We theoretically and empirically validate the strong gradient alignment between log P(y) and log P(y|x), establishing PreRL as a viable surrogate for standard RL. Furthermore, we uncover a critical mechanism: Negative Sample Reinforcement (NSR) within PreRL serves as an exceptionally effective driver for reasoning. NSR-PreRL rapidly prunes incorrect reasoning spaces while stimulating endogenous reflective behaviors, increasing transition and reflection thoughts by 14.89x and 6.54x, respectively. Leveraging these insights, we propose Dual Space RL (DSRL), a Policy Reincarnation strategy that initializes models with NSR-PreRL to expand the reasoning horizon before transitioning to standard RL for fine-grained optimization. Extensive experiments demonstrate that DSRL consistently outperforms strong baselines, proving that pre-train space pruning effectively steers the policy toward a refined correct reasoning subspace.

**arXiv ID:** 2604.14142
</details>

<details>
<summary><strong>RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization</strong> - Yihong Dong, Xue Jiang, Yongding Tao, Huanyu Liu, Kechi Zhang, Lili Mou, Rongyu Cao, Yingwei Ma, Jue Chen, Binhua Li, Zhi Jin, Fei Huang, Yongbin Li, Ge Li - [[pdf]](https://arxiv.org/pdf/2508.00222)</summary>

**Abstract:** Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.

**arXiv ID:** 2508.00222
</details>

<details>
<summary><strong>DeepPresenter: Environment-Grounded Reflection for Agentic Presentation Generation</strong> - Hao Zheng, Guozhao Mo, Xinru Yan, Qianhao Yuan, Wenkai Zhang, Xuanang Chen, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun - [[pdf]](https://arxiv.org/pdf/2602.22839)</summary>

**Abstract:** Presentation generation requires deep content research, coherent visual design, and iterative refinement based on observation. However, existing presentation agents often rely on predefined workflows and fixed templates. To address this, we present DeepPresenter, an agentic framework that adapts to diverse user intents, enables effective feedback-driven refinement, and generalizes beyond a scripted pipeline. Specifically, DeepPresenter autonomously plans, renders, and revises intermediate slide artifacts to support long-horizon refinement with environmental observations. Furthermore, rather than relying on self-reflection over internal signals (e.g., reasoning traces), our environment-grounded reflection conditions the generation process on perceptual artifact states (e.g., rendered slides), enabling the system to identify and correct presentation-specific issues during execution. Results on the evaluation set covering diverse presentation-generation scenarios show that DeepPresenter achieves state-of-the-art performance, and the fine-tuned 9B model remains highly competitive at substantially lower cost. Our project is available at: this https URL

**arXiv ID:** 2602.22839
</details>

<details>
<summary><strong>Three Roles, One Model: Role Orchestration at Inference Time to Close the Performance Gap Between Small and Large Agents</strong> - S. Aaron McClendon, Jorge Gallego-Feliciano, Stavros Zervoudakis, Antonios Saravanos - [[pdf]](https://arxiv.org/pdf/2604.11465)</summary>

**Abstract:** Large language model (LLM) agents show promise on realistic tool-use tasks, but deploying capable agents on modest hardware remains challenging. We study whether inference-time scaffolding alone, without any additional training compute, can improve the performance of a small model in complex multi-step environments. Operating on a single 24GB GPU, we evaluate Qwen3-8B on the AppWorld benchmark under both full-precision and 4-bit quantized configurations. Without any intervention, the raw model achieves just 5.4% (FP16) and 3.0% (AWQ) task goal completion. Guided by a systematic failure mode analysis, we introduce a three-tier inference scaffolding pipeline that deploys the same frozen model in three distinct roles: (1) a summarization model that preserves critical artifacts (tokens, credentials, API responses) while compressing dialogue history; (2) the main agent model that reasons over the compressed context; and (3) an isolated correction model that reviews and revises the agent's code output without access to conversation history, breaking repetitive failure loops. Applied to the same unmodified model, this scaffolding yields 8.9% (FP16) and 5.9% (AWQ) task goal completion, roughly doubling performance in both settings, with particularly strong gains on difficulty-1 tasks (15.8% to 26.3% FP16; 5.3% to 14.0% AWQ). On full-precision inference, our scaffolded 8B model surpasses DeepSeek-Coder 33B Instruct (7.1%) from the original AppWorld evaluation, demonstrating that structured inference-time interventions can make small models competitive with systems 4 times their size. We formalize the approach as a scaffolded policy over a frozen base model, three invocations of the same weights with different conditioning, drawing connections to test-time compute scaling and action-space shaping in reinforcement learning.

**arXiv ID:** 2604.11465
</details>

<details>
<summary><strong>Failure Makes the Agent Stronger: Enhancing Accuracy through Structured Reflection for Reliable Tool Interactions</strong> - Junhao Su, Yuanliang Wan, Junwei Yang, Hengyu Shi, Tianyang Han, Junfeng Luo, Yurui Qiu - [[pdf]](https://arxiv.org/pdf/2509.18847)</summary>

**Abstract:** Tool-augmented large language models (LLMs) are usually trained with supervised imitation or coarse-grained reinforcement learning that optimizes single tool calls. Current self-reflection practices rely on heuristic prompts or one-way reasoning: the model is urged to 'think more' instead of learning error diagnosis and repair. This is fragile in multi-turn interactions; after a failure the model often repeats the same mistake. We propose structured reflection, which turns the path from error to repair into an explicit, controllable, and trainable action. The agent produces a short yet precise reflection: it diagnoses the failure using evidence from the previous step and then proposes a correct, executable follow-up call. For training we combine DAPO and GSPO objectives with a reward scheme tailored to tool use, optimizing the stepwise strategy Reflect, then Call, then Final. To evaluate, we introduce Tool-Reflection-Bench, a lightweight benchmark that programmatically checks structural validity, executability, parameter correctness, and result consistency. Tasks are built as mini trajectories of erroneous call, reflection, and corrected call, with disjoint train and test splits. Experiments on BFCL v3 and Tool-Reflection-Bench show large gains in multi-turn tool-call success and error recovery, and a reduction of redundant calls. These results indicate that making reflection explicit and optimizing it directly improves the reliability of tool interaction and offers a reproducible path for agents to learn from failure.

**arXiv ID:** 2509.18847
</details>

<details>
<summary><strong>Aerial Vision-Language Navigation with a Unified Framework for Spatial, Temporal and Embodied Reasoning</strong> - Huilin Xu, Zhuoyang Liu, Yixiang Luomei, Feng Xu - [[pdf]](https://arxiv.org/pdf/2512.08639)</summary>

**Abstract:** Aerial Vision-and-Language Navigation (VLN) aims to enable unmanned aerial vehicles (UAVs) to interpret natural language instructions and navigate complex urban environments using onboard visual observation. This task holds promise for real-world applications such as low-altitude inspection, search-and-rescue, and autonomous aerial delivery. Existing methods often rely on panoramic images, depth inputs, or odometry to support spatial reasoning and action planning. These requirements increase system cost and integration complexity, thus hindering practical deployment for lightweight UAVs. We present a unified aerial VLN framework that operates solely on egocentric monocular RGB observations and natural language instructions. The model formulates navigation as a next-token prediction problem, jointly optimizing spatial perception, trajectory reasoning, and action prediction through prompt-guided multi-task learning. Moreover, we propose a keyframe selection strategy to reduce visual redundancy by retaining semantically informative frames, along with an action merging and label reweighting mechanism that mitigates long-tailed supervision imbalance and facilitates stable multi-task co-training. Extensive experiments on the AerialVLN and OpenFly benchmark validate the effectiveness of our method. Under the challenging monocular RGB-only setting, our model achieves strong results across both seen and unseen environments. It significantly outperforms existing RGB-only baselines and narrows the performance gap with state-of-the-art panoramic RGB-D counterparts. Comprehensive ablation studies further demonstrate the contribution of our task design and architectural choices. Our code is publicly available at this https URL.

**arXiv ID:** 2512.08639
</details>

<details>
<summary><strong>AgentSPEX: An Agent SPecification and EXecution Language</strong> - Pengcheng Wang, Jerry Huang, Jiarui Yao, Rui Pan, Peizhi Niu, Yaowenqi Liu, Ruida Wang, Renhao Lu, Yuwei Guo, Tong Zhang - [[pdf]](https://arxiv.org/pdf/2604.13346)</summary>

**Abstract:** Language-model agent systems commonly rely on reactive prompting, in which a single instruction guides the model through an open-ended sequence of reasoning and tool-use steps, leaving control flow and intermediate state implicit and making agent behavior potentially difficult to control. Orchestration frameworks such as LangGraph, DSPy, and CrewAI impose greater structure through explicit workflow definitions, but tightly couple workflow logic with Python, making agents difficult to maintain and modify. In this paper, we introduce AgentSPEX, an Agent SPecification and EXecution Language for specifying LLM-agent workflows with explicit control flow and modular structure, along with a customizable agent harness. AgentSPEX supports typed steps, branching and loops, parallel execution, reusable submodules, and explicit state management, and these workflows execute within an agent harness that provides tool access, a sandboxed virtual environment, and support for checkpointing, verification, and logging. Furthermore, we provide a visual editor with synchronized graph and workflow views for authoring and inspection. We include ready-to-use agents for deep research and scientific research, and we evaluate AgentSPEX on 7 benchmarks. Finally, we show through a user study that AgentSPEX provides a more interpretable and accessible workflow-authoring paradigm than a popular existing agent framework.

**arXiv ID:** 2604.13346
</details>

<details>
<summary><strong>MM-Doc-R1: Training Agents for Long Document Visual Question Answering through Multi-turn Reinforcement Learning</strong> - Jiahang Lin, Kai Hu, Binghai Wang, Yuhao Zhou, Zhiheng Xi, Honglin Guo, Shichun Liu, Junzhe Wang, Shihan Dou, Enyu Zhou, Hang Yan, Zhenhua Han, Tao Gui, Qi Zhang, Xuanjing Huang - [[pdf]](https://arxiv.org/pdf/2604.13579)</summary>

**Abstract:** Conventional Retrieval-Augmented Generation (RAG) systems often struggle with complex multi-hop queries over long documents due to their single-pass retrieval. We introduce MM-Doc-R1, a novel framework that employs an agentic, vision-aware workflow to address long document visual question answering through iterative information discovery and synthesis. To incentivize the information seeking capabilities of our agents, we propose Similarity-based Policy Optimization (SPO), addressing baseline estimation bias in existing multi-turn reinforcement learning (RL) algorithms like GRPO. Our core insight is that in multi-turn RL, the more semantically similar two trajectories are, the more accurate their shared baseline estimation becomes. Leveraging this, SPO calculates a more precise baseline by similarity-weighted averaging of rewards across multiple trajectories, unlike GRPO which inappropriately applies the initial state's baseline to all intermediate states. This provides a more stable and accurate learning signal for our agents, leading to superior training performance that surpasses GRPO. Our experiments on the MMLongbench-Doc benchmark show that MM-Doc-R1 outperforms previous baselines by 10.4%. Furthermore, SPO demonstrates superior performance over GRPO, boosting results by 5.0% with Qwen3-8B and 6.1% with Qwen3-4B. These results highlight the effectiveness of our integrated framework and novel training algorithm in advancing the state-of-the-art for complex, long-document visual question answering.

**arXiv ID:** 2604.13579
</details>

<details>
<summary><strong>Not All Tokens Matter: Towards Efficient LLM Reasoning via Token Significance in Reinforcement Learning</strong> - Hanbing Liu, Lang Cao, Yuanyi Ren, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, Dongmei Zhang - [[pdf]](https://arxiv.org/pdf/2506.08125)</summary>

**Abstract:** Large language models (LLMs) show strong reasoning abilities but often produce unnecessarily long explanations that reduce efficiency. Although reinforcement learning (RL) has been used to improve reasoning, most methods focus on accuracy and rely on uniform length-based rewards that overlook the differing contributions of individual tokens, often harming correctness. We revisit length optimization in RL through the perspective of token significance. Observing that many chain-of-thought (CoT) tokens contribute little to the final answer, we introduce a significance-aware length reward that selectively penalizes insignificance tokens, reducing redundancy while preserving essential reasoning. We also propose a dynamic length reward that encourages more detailed reasoning early in training and gradually shifts toward conciseness as learning progresses. Integrating these components into standard policy optimization yields a framework that improves both reasoning efficiency and accuracy. Experiments across multiple benchmarks demonstrate substantial reductions in response length while preserving or improving correctness, highlighting the importance of modeling token significance for efficient LLM reasoning.

**arXiv ID:** 2506.08125
</details>

<details>
<summary><strong>Automated co-design of high-performance thermodynamic cycles via graph-based hierarchical reinforcement learning</strong> - Wenqing Li, Xu Feng, Peixue Jiang, Yinhai Zhu - [[pdf]](https://arxiv.org/pdf/2604.13133)</summary>

**Abstract:** Thermodynamic cycles are pivotal in determining the efficacy of energy conversion systems. Traditional design methodologies, which rely on expert knowledge or exhaustive enumeration, are inefficient and lack scalability, thereby constraining the discovery of high-performance cycles. In this study, we introduce a graph-based hierarchical reinforcement learning approach for the co-design of structure parameters in thermodynamic cycles. These cycles are encoded as graphs, with components and connections depicted as nodes and edges, adhering to grammatical constraints. A deep learning-based thermophysical surrogate facilitates stable graph decoding and the simultaneous resolution of global parameters. Building on this foundation, we develop a hierarchical reinforcement learning framework wherein a high-level manager explores structural evolution and proposes candidate configurations, whereas a low-level worker optimizes parameters and provides performance rewards to steer the search towards high-performance regions. By integrating graph representation, thermophysical surrogate, and manager-worker learning, this method establishes a fully automated pipeline for encoding, decoding, and co-optimization. Using heat pump and heat engine cycles as case studies, the results demonstrate that the proposed method not only replicates classical cycle configurations but also identifies 18 and 21 novel heat pump and heat engine cycles, respectively. Relative to classical cycles, the novel configurations exhibit performance improvements of 4.6% and 133.3%, respectively, surpassing the traditional designs. This method effectively balances efficiency with broad applicability, providing a practical and scalable intelligent alternative to expert-driven thermodynamic cycle design.

**arXiv ID:** 2604.13133
</details>

<details>
<summary><strong>Enhancing Reinforcement Learning for Radiology Report Generation with Evidence-aware Rewards and Self-correcting Preference Learning</strong> - Qin Zhou, Guoyan Liang, Qianyi Yang, Jingyuan Chen, Sai Wu, Chang Yao, Zhe Wang - [[pdf]](https://arxiv.org/pdf/2604.13598)</summary>

**Abstract:** Recent reinforcement learning (RL) approaches have advanced radiology report generation (RRG), yet two core limitations persist: (1) report-level rewards offer limited evidence-grounded guidance for clinical faithfulness; and (2) current methods lack an explicit self-improving mechanism to align with clinical preference. We introduce clinically aligned Evidence-aware Self-Correcting Reinforcement Learning (ESC-RL), comprising two key components. First, a Group-wise Evidence-aware Alignment Reward (GEAR) delivers group-wise, evidence-aware feedback. GEAR reinforces consistent grounding for true positives, recovers missed findings for false negatives, and suppresses unsupported content for false positives. Second, a Self-correcting Preference Learning (SPL) strategy automatically constructs a reliable, disease-aware preference dataset from multiple noisy observations and leverages an LLM to synthesize refined reports without human supervision. ESC-RL promotes clinically faithful, disease-aligned reward and supports continual self-improvement during training. Extensive experiments on two public chest X-ray datasets demonstrate consistent gains and state-of-the-art performance.

**arXiv ID:** 2604.13598
</details>

<details>
<summary><strong>Character Beyond Speech: Leveraging Role-Playing Evaluation in Audio Large Language Models via Reinforcement Learning</strong> - Dongjie Fu, Fangming Feng, Xize Cheng, Linjun Li, Zhou Zhao, Tao Jin - [[pdf]](https://arxiv.org/pdf/2604.13804)</summary>

**Abstract:** The rapid evolution of multimodal large models has revolutionized the simulation of diverse characters in speech dialogue systems, enabling a novel interactive paradigm. Character attributes are manifested not only in textual responses but also through vocal features, as speech conveys rich paralinguistic information that is challenging to quantify. This poses significant difficulties in evaluating the character alignment of role-playing agents. To address these challenges, we present RoleJudge, an evaluation framework that leverages audio large language models to systematically assess the alignment between speech and character across multiple modalities and dimensions. Furthermore, we introduce RoleChat, the first voice role-playing evaluation dataset enriched with chain-of-thought reasoning annotations, comprising a diverse set of authentic and LLM-generated speech samples. Utilizing this dataset, we implement a multi-stage training paradigm and incorporate Standard Alignment in reinforcement learning to mitigate reward misalignment during optimization. Experimental results in terms of accuracy and subjective assessment demonstrate that RoleJudge outperforms various baseline models, validating the effectiveness of our multidimensional evaluation framework.

**arXiv ID:** 2604.13804
</details>

<details>
<summary><strong>Drowsiness-Aware Adaptive Autonomous Braking System based on Deep Reinforcement Learning for Enhanced Road Safety</strong> - Hossem Eddine Hafidi, Elisabetta De Giovanni, Teodoro Montanaro, Ilaria Sergi, Massimo De Vittorio, Luigi Patrono - [[pdf]](https://arxiv.org/pdf/2604.13878)</summary>

**Abstract:** Driver drowsiness significantly impairs the ability to accurately judge safe braking distances and is estimated to contribute to 10%-20% of road accidents in Europe. Traditional driver-assistance systems lack adaptability to real-time physiological states such as drowsiness. This paper proposes a deep reinforcement learning-based autonomous braking system that integrates vehicle dynamics with driver physiological data. Drowsiness is detected from ECG signals using a Recurrent Neural Network (RNN), selected through an extensive benchmark analysis of 2-minute windows with varying segmentation and overlap configurations. The inferred drowsiness state is incorporated into the observable state space of a Double-Dueling Deep Q-Network (DQN) agent, where driver impairment is modeled as an action delay. The system is implemented and evaluated in a high-fidelity CARLA simulation environment. Experimental results show that the proposed agent achieves a 99.99% success rate in avoiding collisions under both drowsy and non-drowsy conditions. These findings demonstrate the effectiveness of physiology-aware control strategies for enhancing adaptive and intelligent driving safety systems.

**arXiv ID:** 2604.13878
</details>

<details>
<summary><strong>A Comparative Study of Dynamic Programming and Reinforcement Learning in Finite Horizon Dynamic Pricing</strong> - Lev Razumovskiy, Nikolay Karenin - [[pdf]](https://arxiv.org/pdf/2604.14059)</summary>

**Abstract:** This paper provides a systematic comparison between Fitted Dynamic Programming (DP), where demand is estimated from data, and Reinforcement Learning (RL) methods in finite-horizon dynamic pricing problems. We analyze their performance across environments of increasing structural complexity, ranging from a single typology benchmark to multi-typology settings with heterogeneous demand and inter-temporal revenue constraints. Unlike simplified comparisons that restrict DP to low-dimensional settings, we apply dynamic programming in richer, multi-dimensional environments with multiple product types and constraints. We evaluate revenue performance, stability, constraint satisfaction behavior, and computational scaling, highlighting the trade-offs between explicit expectation-based optimization and trajectory-based learning.

**arXiv ID:** 2604.14059
</details>

<details>
<summary><strong>WOMBET: World Model-based Experience Transfer for Robust and Sample-efficient Reinforcement Learning</strong> - Mintae Kim, Koushil Sreenath - [[pdf]](https://arxiv.org/pdf/2604.08958)</summary>

**Abstract:** Reinforcement learning (RL) in robotics is often limited by the cost and risk of data collection, motivating experience transfer from a source task to a target task. Offline-to-online RL leverages prior data but typically assumes a given fixed dataset and does not address how to generate reliable data for transfer. We propose \textit{World Model-based Experience Transfer} (WOMBET), a framework that jointly generates and utilizes prior data. WOMBET learns a world model in the source task and generates offline data via uncertainty-penalized planning, followed by filtering trajectories with high return and low epistemic uncertainty. It then performs online fine-tuning in the target task using adaptive sampling between offline and online data, enabling a stable transition from prior-driven initialization to task-specific adaptation. We show that the uncertainty-penalized objective provides a lower bound on the true return and derive a finite-sample error decomposition capturing distribution mismatch and approximation error. Empirically, WOMBET improves sample efficiency and final performance over strong baselines on continuous control benchmarks, demonstrating the benefit of jointly optimizing data generation and transfer.

**arXiv ID:** 2604.08958
</details>

<details>
<summary><strong>Hierarchical DLO Routing with Reinforcement Learning and In-Context Vision-language Models</strong> - Mingen Li, Houjian Yu, Yixuan Huang, Youngjin Hong, Hantao Ye, Changhyun Choi - [[pdf]](https://arxiv.org/pdf/2510.19268)</summary>

**Abstract:** Long-horizon routing tasks of deformable linear objects (DLOs), such as cables and ropes, are common in industrial assembly lines and everyday life. These tasks are particularly challenging because they require robots to manipulate DLO with long-horizon planning and reliable skill execution. Successfully completing such tasks demands adapting to their nonlinear dynamics, decomposing abstract routing goals, and generating multi-step plans composed of multiple skills, all of which require accurate high-level reasoning during execution. In this paper, we propose a fully autonomous hierarchical framework for solving challenging DLO routing tasks. Given an implicit or explicit routing goal expressed in language, our framework leverages vision-language models~(VLMs) for in-context high-level reasoning to synthesize feasible plans, which are then executed by low-level skills trained via reinforcement learning. To improve robustness over long horizons, we further introduce a failure recovery mechanism that reorients the DLO into insertion-feasible states. Our approach generalizes to diverse scenes involving object attributes, spatial descriptions, implicit language commands, and \myred{extended 5-clip settings}. It achieves an overall success rate of 92\% across long-horizon routing scenarios. Please refer to our project page: this https URL

**arXiv ID:** 2510.19268
</details>

<details>
<summary><strong>RobotPan: A 360$^\circ$ Surround-View Robotic Vision System for Embodied Perception</strong> - Jiahao Ma, Qiang Zhang, Peiran Liu, Zeran Su, Pihai Sun, Gang Han, Wen Zhao, Wei Cui, Zhang Zhang, Zhiyuan Xu, Renjing Xu, Jian Tang, Miaomiao Liu, Yijie Guo - [[pdf]](https://arxiv.org/pdf/2604.13476)</summary>

**Abstract:** Surround-view perception is increasingly important for robotic navigation and loco-manipulation, especially in human-in-the-loop settings such as teleoperation, data collection, and emergency takeover. However, current robotic visual interfaces are often limited to narrow forward-facing views, or, when multiple on-board cameras are available, require cumbersome manual switching that interrupts the operator's workflow. Both configurations suffer from motion-induced jitter that causes simulator sickness in head-mounted displays. We introduce a surround-view robotic vision system that combines six cameras with LiDAR to provide full 360$^\circ$ visual coverage, while meeting the geometric and real-time constraints of embodied deployment. We further present \textsc{RobotPan}, a feed-forward framework that predicts \emph{metric-scaled} and \emph{compact} 3D Gaussians from calibrated sparse-view inputs for real-time rendering, reconstruction, and streaming. \textsc{RobotPan} lifts multi-view features into a unified spherical coordinate representation and decodes Gaussians using hierarchical spherical voxel priors, allocating fine resolution near the robot and coarser resolution at larger radii to reduce computational redundancy without sacrificing fidelity. To support long sequences, our online fusion updates dynamic content while preventing unbounded growth in static regions by selectively updating appearance. Finally, we release a multi-sensor dataset tailored to 360$^\circ$ novel view synthesis and metric 3D reconstruction for robotics, covering navigation, manipulation, and locomotion on real platforms. Experiments show that \textsc{RobotPan} achieves competitive quality against prior feed-forward reconstruction and view-synthesis methods while producing substantially fewer Gaussians, enabling practical real-time embodied deployment. Project website: this https URL

**arXiv ID:** 2604.13476
</details>

<details>
<summary><strong>Evolvable Embodied Agent for Robotic Manipulation via Long Short-Term Reflection and Optimization</strong> - Jianzong Wang, Botao Zhao, Yayun He, Junqing Peng, Xulong Zhang - [[pdf]](https://arxiv.org/pdf/2604.13533)</summary>

**Abstract:** Achieving general-purpose robotics requires empowering robots to adapt and evolve based on their environment and feedback. Traditional methods face limitations such as extensive training requirements, difficulties in cross-task generalization, and lack of interpretability. Prompt learning offers new opportunities for self-evolving robots without extensive training, but simply reflecting on past this http URL, extracting meaningful insights from task successes and failures remains a challenge. To this end, we propose the evolvable embodied agent (EEAgent) framework, which leverages large vision-language models (VLMs) for better environmental interpretation and policy planning. To enhance reflection on past experiences, we propose a long short-term reflective optimization (LSTRO) mechanism that dynamically refines prompts based on both past experiences and newly learned lessons, facilitating continuous self-evolution, thereby enhancing overall task success rates. Evaluations on six VIMA-Bench tasks reveal that our approach sets a new state-of-the-art, notably outperforming baselines in complex scenarios.

**arXiv ID:** 2604.13533
</details>

<details>
<summary><strong>Synthesis and Deployment of Maximal Robust Control Barrier Functions through Adversarial Reinforcement Learning</strong> - Donggeon David Oh, Duy P. Nguyen, Haimin Hu, Jaime Fernández Fisac - [[pdf]](https://arxiv.org/pdf/2604.13192)</summary>

**Abstract:** Robust control barrier functions (CBFs) provide a principled mechanism for smooth safety enforcement under worst-case disturbances. However, existing approaches typically rely on explicit, closed-form structure in the dynamics (e.g., control-affine) and uncertainty models. This has led to limited scalability and generality, with most robust CBFs certifying only conservative subsets of the maximal robust safe set. In this paper, we introduce a new robust CBF framework for general nonlinear systems under bounded uncertainty. We first show that the safety value function solving the dynamic programming Isaacs equation is a valid robust discrete-time CBF that enforces safety on the maximal robust safe set. We then adopt the key reinforcement learning (RL) notion of quality function (or Q-function), which removes the need for explicit dynamics by lifting the barrier certificate into state-action space and yields a novel robust Q-CBF constraint for safety filtering. Combined with adversarial RL, this enables the synthesis and deployment of robust Q-CBFs on general nonlinear systems with black-box dynamics and unknown uncertainty structure. We validate the framework on a canonical inverted pendulum benchmark and a 36-D quadruped simulator, achieving substantially less conservative safe sets than barrier-based baselines on the pendulum and reliable safety enforcement even under adversarial uncertainty realizations on the quadruped.

**arXiv ID:** 2604.13192
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
