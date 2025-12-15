# Agent arXiv Daily

**Last Updated:** 2025-12-15 02:26:56

**Total Papers:** 54

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
<summary><strong>SWEnergy: An Empirical Study on Energy Efficiency in Agentic Issue Resolution Frameworks with SLMs</strong> - Arihant Tripathy, Ch Pavan Harshit, Karthik Vaidhyanathan - [[pdf]](https://arxiv.org/pdf/2512.09543)</summary>

**Abstract:** Context. LLM-based autonomous agents in software engineering rely on large, proprietary models, limiting local deployment. This has spurred interest in Small Language Models (SLMs), but their practical effectiveness and efficiency within complex agentic frameworks for automated issue resolution remain poorly understood.
Goal. We investigate the performance, energy efficiency, and resource consumption of four leading agentic issue resolution frameworks when deliberately constrained to using SLMs. We aim to assess the viability of these systems for this task in resource-limited settings and characterize the resulting trade-offs.
Method. We conduct a controlled evaluation of four leading agentic frameworks (SWE-Agent, OpenHands, Mini SWE Agent, AutoCodeRover) using two SLMs (Gemma-3 4B, Qwen-3 1.7B) on the SWE-bench Verified Mini benchmark. On fixed hardware, we measure energy, duration, token usage, and memory over 150 runs per configuration.
Results. We find that framework architecture is the primary driver of energy consumption. The most energy-intensive framework, AutoCodeRover (Gemma), consumed 9.4x more energy on average than the least energy-intensive, OpenHands (Gemma). However, this energy is largely wasted. Task resolution rates were near-zero, demonstrating that current frameworks, when paired with SLMs, consume significant energy on unproductive reasoning loops. The SLM's limited reasoning was the bottleneck for success, but the framework's design was the bottleneck for efficiency.
Conclusions. Current agentic frameworks, designed for powerful LLMs, fail to operate efficiently with SLMs. We find that framework architecture is the primary driver of energy consumption, but this energy is largely wasted due to the SLMs' limited reasoning. Viable low-energy solutions require shifting from passive orchestration to architectures that actively manage SLM weaknesses.

**arXiv ID:** 2512.09543
</details>

<details>
<summary><strong>Decoding Student Minds: Leveraging Conversational Agents for Psychological and Learning Analysis</strong> - Nour El Houda Ben Chaabene, Hamza Hammami, Laid Kahloul - [[pdf]](https://arxiv.org/pdf/2512.10441)</summary>

**Abstract:** This paper presents a psychologically-aware conversational agent designed to enhance both learning performance and emotional well-being in educational settings. The system combines Large Language Models (LLMs), a knowledge graph-enhanced BERT (KG-BERT), and a bidirectional Long Short-Term Memory (LSTM) with attention to classify students' cognitive and affective states in real time. Unlike prior chatbots limited to either tutoring or affective support, our approach leverages multimodal data-including textual semantics, prosodic speech features, and temporal behavioral trends-to infer engagement, stress, and conceptual understanding. A pilot study with university students demonstrated improved motivation, reduced stress, and moderate academic gains compared to baseline methods. These results underline the promise of integrating semantic reasoning, multimodal fusion, and temporal modeling to support adaptive, student-centered educational interventions.

**arXiv ID:** 2512.10441
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>Echo-CoPilot: A Multi-View, Multi-Task Agent for Echocardiography Interpretation and Reporting</strong> - Moein Heidari, Mohammad Amin Roohi, Armin Khosravi, Ilker Hacihaliloglu - [[pdf]](https://arxiv.org/pdf/2512.09944)</summary>

**Abstract:** Echocardiography is central to contemporary cardiovascular care, but full-study interpretation remains a cognitively demanding, multi-view task that is still performed manually. While recent foundation models for echocardiography can achieve strong performance on individual perceptual subtasks such as view classification, segmentation, or disease prediction, they typically operate in isolation and do not provide a unified, clinically coherent assessment. In this work, we introduce Echo-CoPilot, a multi-view, multi-task agent that uses a large language model to orchestrate a suite of specialized echocardiography tools. Within a ReAct-style loop, the agent decomposes clinician queries, invokes tools for view recognition, cardiac structure segmentation, measurement and disease prediction, and report synthesis, and integrates their outputs into guideline-aware answers and narrative summaries. We evaluate Echo-CoPilot on the public MIMIC-EchoQA benchmark, where it achieves an accuracy of 50.8\%, outperforming both general-purpose and biomedical video vision-language models. Qualitative analyses further show that the agent leverages quantitative measurements and physiologic context to resolve challenging cases near clinical decision thresholds, such as borderline left ventricular hypertrophy or pericardial effusion severity. The code will be released upon acceptance of the paper.

**arXiv ID:** 2512.09944
</details>

<details>
<summary><strong>Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution</strong> - Zouying Cao, Jiaji Deng, Li Yu, Weikang Zhou, Zhaoyang Liu, Bolin Ding, Hai Zhao - [[pdf]](https://arxiv.org/pdf/2512.10696)</summary>

**Abstract:** Procedural memory enables large language model (LLM) agents to internalize "how-to" knowledge, theoretically reducing redundant trial-and-error. However, existing frameworks predominantly suffer from a "passive accumulation" paradigm, treating memory as a static append-only archive. To bridge the gap between static storage and dynamic reasoning, we propose $\textbf{ReMe}$ ($\textit{Remember Me, Refine Me}$), a comprehensive framework for experience-driven agent evolution. ReMe innovates across the memory lifecycle via three mechanisms: 1) $\textit{multi-faceted distillation}$, which extracts fine-grained experiences by recognizing success patterns, analyzing failure triggers and generating comparative insights; 2) $\textit{context-adaptive reuse}$, which tailors historical insights to new contexts via scenario-aware indexing; and 3) $\textit{utility-based refinement}$, which autonomously adds valid memories and prunes outdated ones to maintain a compact, high-quality experience pool. Extensive experiments on BFCL-V3 and AppWorld demonstrate that ReMe establishes a new state-of-the-art in agent memory system. Crucially, we observe a significant memory-scaling effect: Qwen3-8B equipped with ReMe outperforms larger, memoryless Qwen3-14B, suggesting that self-evolving memory provides a computation-efficient pathway for lifelong learning. We release our code and the $\texttt{this http URL}$ dataset to facilitate further research.

**arXiv ID:** 2512.10696
</details>

<details>
<summary><strong>Intelligently Weighting Multiple Reference Models for Direct Preference Optimization of LLMs</strong> - Skyler Wu, Aymen Echarghaoui - [[pdf]](https://arxiv.org/pdf/2512.10040)</summary>

**Abstract:** Fine-tuning is integral for aligning large language models (LLMs) with human preferences. Multiple-Reference Preference Optimization (MRPO) builds on Direct Preference Optimization (DPO) by fine-tuning LLMs on preference datasets while regularizing the policy towards a mixture of reference models to leverage their collective desirable properties. However, current methods for setting the reference weights are ad-hoc and statistically unsound, leading to unreliable performance. To address this, we introduce four new weighting strategies: two offline methods that leverage held-out validation signal; one online method that uses a sliding-window estimator to reduce overfitting; and an online method that treats reference weighting as a $K$-armed bandit via Thompson Sampling. Experiments using Qwen2.5-0.5B as the policy model and seven reference models from the Llama, Mistral, Qwen, Yi, and Phi families (0.5B-14B each) show that all 4 of our strategies outperform the current MRPO weighting methods on UltraFeedback and SafeRLHF in preference accuracy. More thought-provokingly, however, we find that single-reference DPO, using any of 6 out of 7 references, consistently outperforms all tested multiple-reference approaches -- calling into question the practical appeal of multiple-reference approaches.

**arXiv ID:** 2512.10040
</details>

<details>
<summary><strong>Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers</strong> - Youmin Ko, Sungjong Seo, Hyunjoon Kim - [[pdf]](https://arxiv.org/pdf/2512.10422)</summary>

**Abstract:** Since large language models (LLMs) have a tendency to generate factually inaccurate output, retrieval-augmented generation (RAG) has gained significant attention as a key means to mitigate this downside of harnessing only LLMs. However, existing RAG methods for simple and multi-hop question answering (QA) are still prone to incorrect retrievals and hallucinations. To address these limitations, we propose CoopRAG, a novel RAG framework for the question answering task in which a retriever and an LLM work cooperatively with each other by exchanging informative knowledge, and the earlier and later layers of the retriever model work cooperatively with each other to accurately rank the retrieved documents relevant to a given query. In this framework, we (i) unroll a question into sub-questions and a reasoning chain in which uncertain positions are masked, (ii) retrieve the documents relevant to the question augmented with the sub-questions and the reasoning chain, (iii) rerank the documents by contrasting layers of the retriever, and (iv) reconstruct the reasoning chain by filling the masked positions via the LLM. Our experiments demonstrate that CoopRAG consistently outperforms state-of-the-art QA methods on three multi-hop QA datasets as well as a simple QA dataset in terms of both the retrieval and QA performances. Our code is available.\footnote{this https URL}

**arXiv ID:** 2512.10422
</details>

<details>
<summary><strong>ARE: Scaling Up Agent Environments and Evaluations</strong> - Romain Froger, Pierre Andrews, Matteo Bettini, Amar Budhiraja, Ricardo Silveira Cabral, Virginie Do, Emilien Garreau, Jean-Baptiste Gaya, Hugo Laurençon, Maxime Lecanu, Kunal Malkan, Dheeraj Mekala, Pierre Ménard, Gerard Moreno-Torres Bertran, Ulyana Piterbarg, Mikhail Plekhanov, Mathieu Rita, Andrey Rusakov, Vladislav Vorotilov, Mengjue Wang, Ian Yu, Amine Benhalloum, Grégoire Mialon, Thomas Scialom - [[pdf]](https://arxiv.org/pdf/2509.17158)</summary>

**Abstract:** We introduce Meta Agents Research Environments (ARE), a research platform for scalable creation of environments, integration of synthetic or real applications, and execution of agentic orchestrations. ARE provides simple abstractions to build complex and diverse environments, each with their own rules, tools, content, and verifiers, helping to bridge the gap between model development and real-world deployment. We also propose Gaia2, a benchmark built in ARE and designed to measure general agent capabilities. Beyond search and execution, Gaia2 requires agents to handle ambiguities and noise, adapt to dynamic environments, collaborate with other agents, and operate under temporal constraints. Unlike prior benchmarks, Gaia2 runs asynchronously, surfacing new failure modes that are invisible in static settings. Our experiments show that no system dominates across the intelligence spectrum: stronger reasoning often comes at the cost of efficiency, and budget scaling curves plateau, highlighting the need for new architectures and adaptive compute strategies. Perhaps more importantly, ARE abstractions enable continuous extension of Gaia2 to other environments, empowering the community to rapidly create new benchmarks tailored to their domains. In AI's second half, progress increasingly depends on defining meaningful tasks and robust evaluations to drive frontier capabilities forward.

**arXiv ID:** 2509.17158
</details>

<details>
<summary><strong>EcomBench: Towards Holistic Evaluation of Foundation Agents in E-commerce</strong> - Rui Min, Zile Qiao, Ze Xu, Jiawen Zhai, Wenyu Gao, Xuanzhong Chen, Haozhen Sun, Zhen Zhang, Xinyu Wang, Hong Zhou, Wenbiao Yin, Bo Zhang, Xuan Zhou, Ming Yan, Yong Jiang, Haicheng Liu, Liang Ding, Ling Zou, Yi R. Fung, Yalong Li, Pengjun Xie - [[pdf]](https://arxiv.org/pdf/2512.08868)</summary>

**Abstract:** Foundation agents have rapidly advanced in their ability to reason and interact with real environments, making the evaluation of their core capabilities increasingly important. While many benchmarks have been developed to assess agent performance, most concentrate on academic settings or artificially designed scenarios while overlooking the challenges that arise in real applications. To address this issue, we focus on a highly practical real-world setting, the e-commerce domain, which involves a large volume of diverse user interactions, dynamic market conditions, and tasks directly tied to real decision-making processes. To this end, we introduce EcomBench, a holistic E-commerce Benchmark designed to evaluate agent performance in realistic e-commerce environments. EcomBench is built from genuine user demands embedded in leading global e-commerce ecosystems and is carefully curated and annotated through human experts to ensure clarity, accuracy, and domain relevance. It covers multiple task categories within e-commerce scenarios and defines three difficulty levels that evaluate agents on key capabilities such as deep information retrieval, multi-step reasoning, and cross-source knowledge integration. By grounding evaluation in real e-commerce contexts, EcomBench provides a rigorous and dynamic testbed for measuring the practical capabilities of agents in modern e-commerce.

**arXiv ID:** 2512.08868
</details>

<details>
<summary><strong>WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving</strong> - Yifang Xu, Jiahao Cui, Feipeng Cai, Zhihao Zhu, Hanlin Shang, Shan Luan, Mingwang Xu, Neng Zhang, Yaoyi Li, Jia Cai, Siyu Zhu - [[pdf]](https://arxiv.org/pdf/2512.06112)</summary>

**Abstract:** We introduce WAM-Flow, a vision-language-action (VLA) model that casts ego-trajectory planning as discrete flow matching over a structured token space. In contrast to autoregressive decoders, WAM-Flow performs fully parallel, bidirectional denoising, enabling coarse-to-fine refinement with a tunable compute-accuracy trade-off. Specifically, the approach combines a metric-aligned numerical tokenizer that preserves scalar geometry via triplet-margin learning, a geometry-aware flow objective and a simulator-guided GRPO alignment that integrates safety, ego progress, and comfort rewards while retaining parallel generation. A multi-stage adaptation converts a pre-trained auto-regressive backbone (Janus-1.5B) from causal decoding to non-causal flow model and strengthens road-scene competence through continued multimodal pretraining. Thanks to the inherent nature of consistency model training and parallel decoding inference, WAM-Flow achieves superior closed-loop performance against autoregressive and diffusion-based VLA baselines, with 1-step inference attaining 89.1 PDMS and 5-step inference reaching 90.3 PDMS on NAVSIM v1 benchmark. These results establish discrete flow matching as a new promising paradigm for end-to-end autonomous driving. The code will be publicly available soon.

**arXiv ID:** 2512.06112
</details>

<details>
<summary><strong>RoleRMBench & RoleRM: Towards Reward Modeling for Profile-Based Role Play in Dialogue Systems</strong> - Hang Ding, Qiming Feng, Dongqi Liu, Qi Zhao, Tao Yao, Shuo Wang, Dongsheng Chen, Jian Li, Zhenye Gan, Jiangning Zhang, Chengjie Wang, Yabiao Wang - [[pdf]](https://arxiv.org/pdf/2512.10575)</summary>

**Abstract:** Reward modeling has become a cornerstone of aligning large language models (LLMs) with human preferences. Yet, when extended to subjective and open-ended domains such as role play, existing reward models exhibit severe degradation, struggling to capture nuanced and persona-grounded human judgments. To address this gap, we introduce RoleRMBench, the first systematic benchmark for reward modeling in role-playing dialogue, covering seven fine-grained capabilities from narrative management to role consistency and engagement. Evaluation on RoleRMBench reveals large and consistent gaps between general-purpose reward models and human judgment, particularly in narrative and stylistic dimensions. We further propose RoleRM, a reward model trained with Continuous Implicit Preferences (CIP), which reformulates subjective evaluation as continuous consistent pairwise supervision under multiple structuring strategies. Comprehensive experiments show that RoleRM surpasses strong open- and closed-source reward models by over 24% on average, demonstrating substantial gains in narrative coherence and stylistic fidelity. Our findings highlight the importance of continuous preference representation and annotation consistency, establishing a foundation for subjective alignment in human-centered dialogue systems.

**arXiv ID:** 2512.10575
</details>

<details>
<summary><strong>TheMCPCompany: Creating General-purpose Agents with Task-specific Tools</strong> - Reza Esfandiarpoor, Vishwas Suryanarayanan, Stephen H. Bach, Vishal Chowdhary, Anthony Aue - [[pdf]](https://arxiv.org/pdf/2510.19286)</summary>

**Abstract:** Since the introduction of the Model Context Protocol (MCP), the number of available tools for Large Language Models (LLMs) has increased significantly. These task-specific tool sets offer an alternative to general-purpose tools such as web browsers, while being easier to develop and maintain than GUIs. However, current general-purpose agents predominantly rely on web browsers for interacting with the environment. Here, we introduce TheMCPCompany, a benchmark for evaluating tool-calling agents on tasks that involve interacting with various real-world services. We use the REST APIs of these services to create MCP servers, which include over 18,000 tools. We also provide manually annotated ground-truth tools for each task. In our experiments, we use the ground truth tools to show the potential of tool-calling agents for both improving performance and reducing costs assuming perfect tool retrieval. Next, we explore agent performance using tool retrieval to study the real-world practicality of tool-based agents. While all models with tool retrieval perform similarly or better than browser-based agents, smaller models cannot take full advantage of the available tools through retrieval. On the other hand, GPT-5's performance with tool retrieval is very close to its performance with ground-truth tools. Overall, our work shows that the most advanced reasoning models are effective at discovering tools in simpler environments, but seriously struggle with navigating complex enterprise environments. TheMCPCompany reveals that navigating tens of thousands of tools and combining them in non-trivial ways to solve complex problems is still a challenging task for current models and requires both better reasoning and better retrieval models.

**arXiv ID:** 2510.19286
</details>

<details>
<summary><strong>RoboNeuron: A Modular Framework Linking Foundation Models and ROS for Embodied AI</strong> - Weifan Guan, Huasen Xi, Chenxiao Zhang, Aosheng Li, Qinghao Hu, Jian Cheng - [[pdf]](https://arxiv.org/pdf/2512.10394)</summary>

**Abstract:** Current embodied AI systems face severe engineering impediments, primarily characterized by poor cross-scenario adaptability, rigid inter-module coupling, and fragmented inference acceleration. To overcome these limitations, we propose RoboNeuron, a universal deployment framework for embodied intelligence. RoboNeuron is the first framework to deeply integrate the cognitive capabilities of Large Language Models (LLMs) and Vision-Language-Action (VLA) models with the real-time execution backbone of the Robot Operating System (ROS). We utilize the Model Context Protocol (MCP) as a semantic bridge, enabling the LLM to dynamically orchestrate underlying robotic tools. The framework establishes a highly modular architecture that strictly decouples sensing, reasoning, and control by leveraging ROS's unified communication interfaces. Crucially, we introduce an automated tool to translate ROS messages into callable MCP functions, significantly streamlining development. RoboNeuron significantly enhances cross-scenario adaptability and component flexibility, while establishing a systematic platform for horizontal performance benchmarking, laying a robust foundation for scalable real-world embodied applications.

**arXiv ID:** 2512.10394
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation</strong> - Lim Chien Her, Ming Yan, Yunshu Bai, Ruihao Li, Hao Zhang - [[pdf]](https://arxiv.org/pdf/2512.10501)</summary>

**Abstract:** Procedural Content Generation (PCG) offers scalable methods for algorithmically creating complex, customizable worlds. However, controlling these pipelines requires the precise configuration of opaque technical parameters. We propose a training-free architecture that utilizes LLM agents for zero-shot PCG parameter configuration. While Large Language Models (LLMs) promise a natural language interface for PCG tools, off-the-shelf models often fail to bridge the semantic gap between abstract user instructions and strict parameter specifications. Our system pairs an Actor agent with a Critic agent, enabling an iterative workflow where the system autonomously reasons over tool parameters and refines configurations to progressively align with human design preferences. We validate this approach on the generation of various 3D maps, establishing a new benchmark for instruction-following in PCG. Experiments demonstrate that our approach outperforms single-agent baselines, producing diverse and structurally valid environments from natural language descriptions. These results demonstrate that off-the-shelf LLMs can be effectively repurposed as generalized agents for arbitrary PCG tools. By shifting the burden from model training to architectural reasoning, our method offers a scalable framework for mastering complex software without task-specific fine-tuning.

**arXiv ID:** 2512.10501
</details>

<details>
<summary><strong>Achieving Olympia-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning</strong> - Haiteng Zhao, Junhao Shen, Yiming Zhang, Songyang Gao, Kuikun Liu, Tianyou Ma, Fan Zheng, Dahua Lin, Wenwei Zhang, Kai Chen - [[pdf]](https://arxiv.org/pdf/2512.10534)</summary>

**Abstract:** Large language model (LLM) agents exhibit strong mathematical problem-solving abilities and can even solve International Mathematical Olympiad (IMO) level problems with the assistance of formal proof systems. However, due to weak heuristics for auxiliary constructions, AI for geometry problem solving remains dominated by expert models such as AlphaGeometry 2, which rely heavily on large-scale data synthesis and search for both training and evaluation. In this work, we make the first attempt to build a medalist-level LLM agent for geometry and present InternGeometry. InternGeometry overcomes the heuristic limitations in geometry by iteratively proposing propositions and auxiliary constructions, verifying them with a symbolic engine, and reflecting on the engine's feedback to guide subsequent proposals. A dynamic memory mechanism enables InternGeometry to conduct more than two hundred interactions with the symbolic engine per problem. To further accelerate learning, we introduce Complexity-Boosting Reinforcement Learning (CBRL), which gradually increases the complexity of synthesized problems across training stages. Built on InternThinker-32B, InternGeometry solves 44 of 50 IMO geometry problems (2000-2024), exceeding the average gold medalist score (40.9), using only 13K training examples, just 0.004% of the data used by AlphaGeometry 2, demonstrating the potential of LLM agents on expert-level geometry tasks. InternGeometry can also propose novel auxiliary constructions for IMO problems that do not appear in human solutions. We will release the model, data, and symbolic engine to support future research.

**arXiv ID:** 2512.10534
</details>

<details>
<summary><strong>A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents</strong> - Sizhe Zhou, Jiawei Han - [[pdf]](https://arxiv.org/pdf/2511.17208)</summary>

**Abstract:** LLM-based conversational agents still struggle to maintain coherent, personalized interaction over many sessions: fixed context windows limit how much history can be kept in view, and most external memory approaches trade off between coarse retrieval over large chunks and fine-grained but fragmented views of the dialogue. Motivated by neo-Davidsonian event semantics, we propose an event-centric alternative that represents conversational history as short, event-like propositions which bundle together participants, temporal cues, and minimal local context, rather than as independent relation triples or opaque summaries. In contrast to work that aggressively compresses or forgets past content, our design aims to preserve information in a non-compressive form and make it more accessible, rather than more lossy. Concretely, we instruct an LLM to decompose each session into enriched elementary discourse units (EDUs) -- self-contained statements with normalized entities and source turn attributions -- and organize sessions, EDUs, and their arguments in a heterogeneous graph that supports associative recall. On top of this representation we build two simple retrieval-based variants that use dense similarity search and LLM filtering, with an optional graph-based propagation step to connect and aggregate evidence across related EDUs. Experiments on the LoCoMo and LongMemEval$_S$ benchmarks show that these event-centric memories match or surpass strong baselines, while operating with much shorter QA contexts. Our results suggest that structurally simple, event-level memory provides a principled and practical foundation for long-horizon conversational agents. Our code and data will be released at this https URL.

**arXiv ID:** 2511.17208
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (18 papers)</h2></summary>

<details>
<summary><strong>Exploring Health Misinformation Detection with Multi-Agent Debate</strong> - Chih-Han Chen, Chen-Han Tsai, Yu-Shao Peng - [[pdf]](https://arxiv.org/pdf/2512.09935)</summary>

**Abstract:** Fact-checking health-related claims has become increasingly critical as misinformation proliferates online. Effective verification requires both the retrieval of high-quality evidence and rigorous reasoning processes. In this paper, we propose a two-stage framework for health misinformation detection: Agreement Score Prediction followed by Multi-Agent Debate. In the first stage, we employ large language models (LLMs) to independently evaluate retrieved articles and compute an aggregated agreement score that reflects the overall evidence stance. When this score indicates insufficient consensus-falling below a predefined threshold-the system proceeds to a second stage. Multiple agents engage in structured debate to synthesize conflicting evidence and generate well-reasoned verdicts with explicit justifications. Experimental results demonstrate that our two-stage approach achieves superior performance compared to baseline methods, highlighting the value of combining automated scoring with collaborative reasoning for complex verification tasks.

**arXiv ID:** 2512.09935
</details>

<details>
<summary><strong>DynaMate: An Autonomous Agent for Protein-Ligand Molecular Dynamics Simulations</strong> - Salomé Guilbert, Cassandra Masschelein, Jeremy Goumaz, Bohdan Naida, Philippe Schwaller - [[pdf]](https://arxiv.org/pdf/2512.10034)</summary>

**Abstract:** Force field-based molecular dynamics (MD) simulations are indispensable for probing the structure, dynamics, and functions of biomolecular systems, including proteins and protein-ligand complexes. Despite their broad utility in drug discovery and protein engineering, the technical complexity of MD setup, encompassing parameterization, input preparation, and software configuration, remains a major barrier for widespread and efficient usage. Agentic LLMs have demonstrated their capacity to autonomously execute multi-step scientific processes, and to date, they have not successfully been used to automate protein-ligand MD workflows. Here, we present DynaMate, a modular multi-agent framework that autonomously designs and executes complete MD workflows for both protein and protein-ligand systems, and offers free energy binding affinity calculations with the MM/PB(GB)SA method. The framework integrates dynamic tool use, web search, PaperQA, and a self-correcting behavior. DynaMate comprises three specialized modules, interacting to plan the experiment, perform the simulation, and analyze the results. We evaluated its performance across twelve benchmark systems of varying complexity, assessing success rate, efficiency, and adaptability. DynaMate reliably performed full MD simulations, corrected runtime errors through iterative reasoning, and produced meaningful analyses of protein-ligand interactions. This automated framework paves the way toward standardized, scalable, and time-efficient molecular modeling pipelines for future biomolecular and drug design applications.

**arXiv ID:** 2512.10034
</details>

<details>
<summary><strong>EpiPlanAgent: Agentic Automated Epidemic Response Planning</strong> - Kangkun Mao, Fang Xu, Jinru Ding, Yidong Jiang, Yujun Yao, Yirong Chen, Junming Liu, Xiaoqin Wu, Qian Wu, Xiaoyan Huang, Jie Xu - [[pdf]](https://arxiv.org/pdf/2512.10313)</summary>

**Abstract:** Epidemic response planning is essential yet traditionally reliant on labor-intensive manual methods. This study aimed to design and evaluate EpiPlanAgent, an agent-based system using large language models (LLMs) to automate the generation and validation of digital emergency response plans. The multi-agent framework integrated task decomposition, knowledge grounding, and simulation modules. Public health professionals tested the system using real-world outbreak scenarios in a controlled evaluation. Results demonstrated that EpiPlanAgent significantly improved the completeness and guideline alignment of plans while drastically reducing development time compared to manual workflows. Expert evaluation confirmed high consistency between AI-generated and human-authored content. User feedback indicated strong perceived utility. In conclusion, EpiPlanAgent provides an effective, scalable solution for intelligent epidemic response planning, demonstrating the potential of agentic AI to transform public health preparedness.

**arXiv ID:** 2512.10313
</details>

<details>
<summary><strong>On the Dynamics of Multi-Agent LLM Communities Driven by Value Diversity</strong> - Muhua Huang, Qinlin Zhao, Xiaoyuan Yi, Xing Xie - [[pdf]](https://arxiv.org/pdf/2512.10665)</summary>

**Abstract:** As Large Language Models (LLM) based multi-agent systems become increasingly prevalent, the collective behaviors, e.g., collective intelligence, of such artificial communities have drawn growing attention. This work aims to answer a fundamental question: How does diversity of values shape the collective behavior of AI communities? Using naturalistic value elicitation grounded in the prevalent Schwartz's Theory of Basic Human Values, we constructed multi-agent simulations where communities with varying numbers of agents engaged in open-ended interactions and constitution formation. The results show that value diversity enhances value stability, fosters emergent behaviors, and brings more creative principles developed by the agents themselves without external guidance. However, these effects also show diminishing returns: extreme heterogeneity induces instability. This work positions value diversity as a new axis of future AI capability, bridging AI ability and sociological studies of institutional emergence.

**arXiv ID:** 2512.10665
</details>

<details>
<summary><strong>On Decision-Making Agents and Higher-Order Causal Processes</strong> - Matt Wilson - [[pdf]](https://arxiv.org/pdf/2512.10937)</summary>

**Abstract:** We establish a precise correspondence between decision-making agents in partially observable Markov decision processes (POMDPs) and one-input process functions, the classical limit of higher-order quantum operations. In this identification an agent's policy and memory update combine into a process function w that interacts with a POMDP environment via the link product. This suggests a dual interpretation: in the physics view, the process function acts as the environment into which local operations (agent interventions) are inserted, whereas in the AI view it encodes the agent and the inserted functions represent environments. We extend this perspective to multi-agent systems by identifying observation-independent decentralized POMDPs as natural domains for multi-input process functions.

**arXiv ID:** 2512.10937
</details>

<details>
<summary><strong>Norm-Governed Multi-Agent Decision-Making in Simulator-Coupled Environments:The Reinsurance Constrained Multi-Agent Simulation Process (R-CMASP)</strong> - Stella C. Dong - [[pdf]](https://arxiv.org/pdf/2512.09939)</summary>

**Abstract:** Reinsurance decision-making exhibits the core structural properties that motivate multi-agent models: distributed and asymmetric information, partial observability, heterogeneous epistemic responsibilities, simulator-driven environment dynamics, and binding prudential and regulatory constraints. Deterministic workflow automation cannot meet these requirements, as it lacks the epistemic flexibility, cooperative coordination mechanisms, and norm-sensitive behaviour required for institutional risk-transfer.
We propose the Reinsurance Constrained Multi-Agent Simulation Process (R-CMASP), a formal model that extends stochastic games and Dec-POMDPs by adding three missing elements: (i) simulator-coupled transition dynamics grounded in catastrophe, capital, and portfolio engines; (ii) role-specialized agents with structured observability, belief updates, and typed communication; and (iii) a normative feasibility layer encoding solvency, regulatory, and organizational rules as admissibility constraints on joint actions.
Using LLM-based agents with tool access and typed message protocols, we show in a domain-calibrated synthetic environment that governed multi-agent coordination yields more stable, coherent, and norm-adherent behaviour than deterministic automation or monolithic LLM baselines--reducing pricing variance, improving capital efficiency, and increasing clause-interpretation accuracy. Embedding prudential norms as admissibility constraints and structuring communication into typed acts measurably enhances equilibrium stability.
Overall, the results suggest that regulated, simulator-driven decision environments are most naturally modelled as norm-governed, simulator-coupled multi-agent systems.

**arXiv ID:** 2512.09939
</details>

<details>
<summary><strong>Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving</strong> - Songyang Gao, Yuzhe Gu, Zijian Wu, Lingkai Kong, Wenwei Zhang, Zhongrui Cai, Fan Zheng, Tianyou Ma, Junhao Shen, Haiteng Zhao, Duanyang Zhang, Huilun Zhang, Kuikun Liu, Chengqi Lyu, Yanhui Duan, Chiyu Chen, Ningsheng Ma, Jianfei Gao, Han Lyu, Dahua Lin, Kai Chen - [[pdf]](https://arxiv.org/pdf/2512.10739)</summary>

**Abstract:** Large Reasoning Models (LRMs) have expanded the mathematical reasoning frontier through Chain-of-Thought (CoT) techniques and Reinforcement Learning with Verifiable Rewards (RLVR), capable of solving AIME-level problems. However, the performance of LRMs is heavily dependent on the extended reasoning context length. For solving ultra-hard problems like those in the International Mathematical Olympiad (IMO), the required reasoning complexity surpasses the space that an LRM can explore in a single round. Previous works attempt to extend the reasoning context of LRMs but remain prompt-based and built upon proprietary models, lacking systematic structures and training pipelines. Therefore, this paper introduces Intern-S1-MO, a long-horizon math agent that conducts multi-round hierarchical reasoning, composed of an LRM-based multi-agent system including reasoning, summary, and verification. By maintaining a compact memory in the form of lemmas, Intern-S1-MO can more freely explore the lemma-rich reasoning spaces in multiple reasoning stages, thereby breaking through the context constraints for IMO-level math problems. Furthermore, we propose OREAL-H, an RL framework for training the LRM using the online explored trajectories to simultaneously bootstrap the reasoning ability of LRM and elevate the overall performance of Intern-S1-MO. Experiments show that Intern-S1-MO can obtain 26 out of 35 points on the non-geometry problems of IMO2025, matching the performance of silver medalists. It also surpasses the current advanced LRMs on inference benchmarks such as HMMT2025, AIME2025, and CNMO2025. In addition, our agent officially participates in CMO2025 and achieves a score of 102/126 under the judgment of human experts, reaching the gold medal level.

**arXiv ID:** 2512.10739
</details>

<details>
<summary><strong>Multi-Robot Path Planning Combining Heuristics and Multi-Agent Reinforcement Learning</strong> - Shaoming Peng - [[pdf]](https://arxiv.org/pdf/2306.01270)</summary>

**Abstract:** Multi-robot path finding in dynamic environments is a highly challenging classic problem. In the movement process, robots need to avoid collisions with other moving robots while minimizing their travel distance. Previous methods for this problem either continuously replan paths using heuristic search methods to avoid conflicts or choose appropriate collision avoidance strategies based on learning approaches. The former may result in long travel distances due to frequent replanning, while the latter may have low learning efficiency due to low sample exploration and utilization, and causing high training costs for the model. To address these issues, we propose a path planning method, MAPPOHR, which combines heuristic search, empirical rules, and multi-agent reinforcement learning. The method consists of two layers: a real-time planner based on the multi-agent reinforcement learning algorithm, MAPPO, which embeds empirical rules in the action output layer and reward functions, and a heuristic search planner used to create a global guiding path. During movement, the heuristic search planner replans new paths based on the instructions of the real-time planner. We tested our method in 10 different conflict scenarios. The experiments show that the planning performance of MAPPOHR is better than that of existing learning and heuristic methods. Due to the utilization of empirical knowledge and heuristic search, the learning efficiency of MAPPOHR is higher than that of existing learning methods.

**arXiv ID:** 2306.01270
</details>

<details>
<summary><strong>Towards Foundation Models with Native Multi-Agent Intelligence</strong> - Shuyue Hu, Haoyang Yan, Yiqun Zhang, Yang Chen, Dongzhan Zhou, Lei Bai - [[pdf]](https://arxiv.org/pdf/2512.08743)</summary>

**Abstract:** Foundation models (FMs) are increasingly assuming the role of the "brain" of AI agents. While recent efforts have begun to equip FMs with native single-agent abilities -- such as GUI interaction or integrated tool use -- we argue that the next frontier is endowing FMs with native multi-agent intelligence. We identify four core capabilities of FMs in multi-agent contexts: understanding, planning, efficient communication, and adaptation. Contrary to assumptions about the spontaneous emergence of such abilities, we provide extensive empirical evidence across 41 large language models showing that strong single-agent performance alone does not automatically yield robust multi-agent intelligence. To address this gap, we outline key research directions -- spanning dataset construction, evaluation, training paradigms, and safety considerations -- for building FMs with native multi-agent intelligence.

**arXiv ID:** 2512.08743
</details>

<details>
<summary><strong>Risk-Bounded Multi-Agent Visual Navigation via Iterative Risk Allocation</strong> - Viraj Parimi, Brian C. Williams - [[pdf]](https://arxiv.org/pdf/2509.08157)</summary>

**Abstract:** Safe navigation is essential for autonomous systems operating in hazardous environments, especially when multiple agents must coordinate using only high-dimensional visual observations. While recent approaches successfully combine Goal-Conditioned RL (GCRL) for graph construction with Conflict-Based Search (CBS) for planning, they typically rely on static edge pruning to enforce safety. This binary strategy is overly conservative, precluding feasible missions that require traversing high-risk regions, even when the aggregate risk is acceptable. To address this, we introduce a framework for Risk-Bounded Multi-Agent Path Finding (\problem{}), where agents share a user-specified global risk budget ($\Delta$). Rather than permanently discarding edges, our framework dynamically distributes per-agent risk budgets ($\delta_i$) during search via an Iterative Risk Allocation (IRA) layer that integrates with a standard CBS planner. We investigate two distribution strategies: a greedy surplus-deficit scheme for rapid feasibility repair, and a market-inspired mechanism that treats risk as a priced resource to guide improved allocation. This yields a tunable trade-off wherein agents exploit available risk to secure shorter, more efficient paths, but revert to longer, safer detours under tighter budgets. Experiments in complex visual environments show that, our dynamic allocation framework achieves higher success rates than baselines and effectively leverages the available safety budget to reduce travel time.

**arXiv ID:** 2509.08157
</details>

<details>
<summary><strong>Think Before You Drive: World Model-Inspired Multimodal Grounding for Autonomous Vehicles</strong> - Haicheng Liao, Huanming Shen, Bonan Wang, Yongkang Li, Yihong Tang, Chengyue Wang, Dingyi Zhuang, Kehua Chen, Hai Yang, Chengzhong Xu, Zhenning Li - [[pdf]](https://arxiv.org/pdf/2512.03454)</summary>

**Abstract:** Interpreting natural-language commands to localize target objects is critical for autonomous driving (AD). Existing visual grounding (VG) methods for autonomous vehicles (AVs) typically struggle with ambiguous, context-dependent instructions, as they lack reasoning over 3D spatial relations and anticipated scene evolution. Grounded in the principles of world models, we propose ThinkDeeper, a framework that reasons about future spatial states before making grounding decisions. At its core is a Spatial-Aware World Model (SA-WM) that learns to reason ahead by distilling the current scene into a command-aware latent state and rolling out a sequence of future latent states, providing forward-looking cues for disambiguation. Complementing this, a hypergraph-guided decoder then hierarchically fuses these states with the multimodal input, capturing higher-order spatial dependencies for robust localization. In addition, we present DrivePilot, a multi-source VG dataset in AD, featuring semantic annotations generated by a Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT)-prompted LLM pipeline. Extensive evaluations on six benchmarks, ThinkDeeper ranks #1 on the Talk2Car leaderboard and surpasses state-of-the-art baselines on DrivePilot, MoCAD, and RefCOCO/+/g benchmarks. Notably, it shows strong robustness and efficiency in challenging scenes (long-text, multi-agent, ambiguity) and retains superior performance even when trained on 50% of the data.

**arXiv ID:** 2512.03454
</details>

<details>
<summary><strong>Empirical Hardness in Multi-Agent Pathfinding: Research Challenges and Opportunities</strong> - Jingyao Ren, Eric Ewing, T. K. Satish Kumar, Sven Koenig, Nora Ayanian - [[pdf]](https://arxiv.org/pdf/2512.10078)</summary>

**Abstract:** Multi-agent pathfinding (MAPF) is the problem of finding collision-free paths for a team of agents on a map. Although MAPF is NP-hard, the hardness of solving individual instances varies significantly, revealing a gap between theoretical complexity and actual hardness. This paper outlines three key research challenges in MAPF empirical hardness to understand such phenomena. The first challenge, known as algorithm selection, is determining the best-performing algorithms for a given instance. The second challenge is understanding the key instance features that affect MAPF empirical hardness, such as structural properties like phase transition and backbone/backdoor. The third challenge is how to leverage our knowledge of MAPF empirical hardness to effectively generate hard MAPF instances or diverse benchmark datasets. This work establishes a foundation for future empirical hardness research and encourages deeper investigation into these promising and underexplored areas.

**arXiv ID:** 2512.10078
</details>

<details>
<summary><strong>Emergent Collective Memory in Decentralized Multi-Agent AI Systems</strong> - Khushiyant - [[pdf]](https://arxiv.org/pdf/2512.10166)</summary>

**Abstract:** We demonstrate how collective memory emerges in decentralized multi-agent systems through the interplay between individual agent memory and environmental trace communication. Our agents maintain internal memory states while depositing persistent environmental traces, creating a spatially distributed collective memory without centralized control. Comprehensive validation across five environmental conditions (20x20 to 50x50 grids, 5-20 agents, 50 runs per configuration) reveals a critical asymmetry: individual memory alone provides 68.7% performance improvement over no-memory baselines (1563.87 vs 927.23, p < 0.001), while environmental traces without memory fail completely. This demonstrates that memory functions independently but traces require cognitive infrastructure for interpretation. Systematic density-sweep experiments (rho in [0.049, 0.300], up to 625 agents) validate our theoretical phase transition prediction. On realistic large grids (30x30, 50x50), stigmergic coordination dominates above rho ~ 0.20, with traces outperforming memory by 36-41% on composite metrics despite lower food efficiency. The experimental crossover confirms the predicted critical density rho_c = 0.230 within 13% error.

**arXiv ID:** 2512.10166
</details>

<details>
<summary><strong>AutoMedic: An Automated Evaluation Framework for Clinical Conversational Agents with Medical Dataset Grounding</strong> - Gyutaek Oh, Sangjoon Park, Byung-Hoon Kim - [[pdf]](https://arxiv.org/pdf/2512.10195)</summary>

**Abstract:** Evaluating large language models (LLMs) has recently emerged as a critical issue for safe and trustworthy application of LLMs in the medical domain. Although a variety of static medical question-answering (QA) benchmarks have been proposed, many aspects remain underexplored, such as the effectiveness of LLMs in generating responses in dynamic, interactive clinical multi-turn conversation situations and the identification of multi-faceted evaluation strategies beyond simple accuracy. However, formally evaluating a dynamic, interactive clinical situation is hindered by its vast combinatorial space of possible patient states and interaction trajectories, making it difficult to standardize and quantitatively measure such scenarios. Here, we introduce AutoMedic, a multi-agent simulation framework that enables automated evaluation of LLMs as clinical conversational agents. AutoMedic transforms off-the-shelf static QA datasets into virtual patient profiles, enabling realistic and clinically grounded multi-turn clinical dialogues between LLM agents. The performance of various clinical conversational agents is then assessed based on our CARE metric, which provides a multi-faceted evaluation standard of clinical conversational accuracy, efficiency/strategy, empathy, and robustness. Our findings, validated by human experts, demonstrate the validity of AutoMedic as an automated evaluation framework for clinical conversational agents, offering practical guidelines for the effective development of LLMs in conversational medical applications.

**arXiv ID:** 2512.10195
</details>

<details>
<summary><strong>Planning, Living and Judging: A Multi-agent LLM-based Framework for Cyclical Urban Planning</strong> - Hang Ni, Yuzhi Wang, Hao Liu - [[pdf]](https://arxiv.org/pdf/2412.20505)</summary>

**Abstract:** Urban regeneration presents significant challenges within the context of urbanization, requiring adaptive approaches to tackle evolving needs. Leveraging advancements in large language models (LLMs), we propose Cyclical Urban Planning (CUP), a new paradigm that continuously generates, evaluates, and refines urban plans in a closed-loop. Specifically, our multi-agent LLM-based framework consists of three key components: (1) Planning, where LLM agents generate and refine urban plans based on contextual data; (2) Living, where agents simulate the behaviors and interactions of residents, modeling life in the urban environment; and (3) Judging, which involves evaluating plan effectiveness and providing iterative feedback for improvement. The cyclical process enables a dynamic and responsive planning approach. Experiments on the real-world dataset demonstrate the effectiveness of our framework as a continuous and adaptive planning process.

**arXiv ID:** 2412.20505
</details>

<details>
<summary><strong>CompanionCast: A Multi-Agent Conversational AI Framework with Spatial Audio for Social Co-Viewing Experiences</strong> - Yiyang Wang, Chen Chen, Tica Lin, Vishnu Raj, Josh Kimball, Alex Cabral, Josiah Hester - [[pdf]](https://arxiv.org/pdf/2512.10918)</summary>

**Abstract:** Social presence is central to the enjoyment of watching content together, yet modern media consumption is increasingly solitary. We investigate whether multi-agent conversational AI systems can recreate the dynamics of shared viewing experiences across diverse content types. We present CompanionCast, a general framework for orchestrating multiple role-specialized AI agents that respond to video content using multimodal inputs, speech synthesis, and spatial audio. Distinctly, CompanionCast integrates an LLM-as-a-Judge module that iteratively scores and refines conversations across five dimensions (relevance, authenticity, engagement, diversity, personality consistency). We validate this framework through sports viewing, a domain with rich dynamics and strong social traditions, where a pilot study with soccer fans suggests that multi-agent interaction improves perceived social presence compared to solo viewing. We contribute: (1) a generalizable framework for orchestrating multi-agent conversations around multimodal video content, (2) a novel evaluator-agent pipeline for conversation quality control, and (3) exploratory evidence of increased social presence in AI-mediated co-viewing. We discuss challenges and future directions for applying this approach to diverse viewing contexts including entertainment, education, and collaborative watching experiences.

**arXiv ID:** 2512.10918
</details>

<details>
<summary><strong>Learning Controllable and Diverse Player Behaviors in Multi-Agent Environments</strong> - Atahan Cilan, Atay Özgövde - [[pdf]](https://arxiv.org/pdf/2512.10835)</summary>

**Abstract:** This paper introduces a reinforcement learning framework that enables controllable and diverse player behaviors without relying on human gameplay data. Existing approaches often require large-scale player trajectories, train separate models for different player types, or provide no direct mapping between interpretable behavioral parameters and the learned policy, limiting their scalability and controllability. We define player behavior in an N-dimensional continuous space and uniformly sample target behavior vectors from a region that encompasses the subset representing real human styles. During training, each agent receives both its current and target behavior vectors as input, and the reward is based on the normalized reduction in distance between them. This allows the policy to learn how actions influence behavioral statistics, enabling smooth control over attributes such as aggressiveness, mobility, and cooperativeness. A single PPO-based multi-agent policy can reproduce new or unseen play styles without retraining. Experiments conducted in a custom multi-player Unity game show that the proposed framework produces significantly greater behavioral diversity than a win-only baseline and reliably matches specified behavior vectors across diverse targets. The method offers a scalable solution for automated playtesting, game balancing, human-like behavior simulation, and replacing disconnected players in online games.

**arXiv ID:** 2512.10835
</details>

<details>
<summary><strong>FICO: Finite-Horizon Closed-Loop Factorization for Unified Multi-Agent Path Finding</strong> - Jiarui Li, Alessandro Zanardi, Federico Pecora, Runyu Zhang, Gioele Zardini - [[pdf]](https://arxiv.org/pdf/2511.13961)</summary>

**Abstract:** Multi-Agent Path Finding is a fundamental problem in robotics and AI, yet most existing formulations treat planning and execution separately and address variants of the problem in an ad hoc manner. This paper presents a system-level framework for MAPF that integrates planning and execution, generalizes across variants, and explicitly models uncertainties. At its core is the MAPF system, a formal model that casts MAPF as a control design problem encompassing classical and uncertainty-aware formulations. To solve it, we introduce Finite-Horizon Closed-Loop Factorization (FICO), a factorization-based algorithm inspired by receding-horizon control that exploits compositional structure for efficient closed-loop operation. FICO enables real-time responses -- commencing execution within milliseconds -- while scaling to thousands of agents and adapting seamlessly to execution-time uncertainties. Extensive case studies demonstrate that it reduces computation time by up to two orders of magnitude compared with open-loop baselines, while delivering significantly higher throughput under stochastic delays and agent arrivals. These results establish a principled foundation for analyzing and advancing MAPF through system-level modeling, factorization, and closed-loop design.

**arXiv ID:** 2511.13961
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>Suzume-chan: Your Personal Navigator as an Embodied Information Hub</strong> - Maya Grace Torii, Takahito Murakami, Shuka Koseki, Yoichi Ochiai - [[pdf]](https://arxiv.org/pdf/2512.09932)</summary>

**Abstract:** Access to expert knowledge often requires real-time human communication. Digital tools improve access to information but rarely create the sense of connection needed for deep understanding. This study addresses this issue using Social Presence Theory, which explains how a feeling of "being together" enhances communication. An "Embodied Information Hub" is proposed as a new way to share knowledge through physical and conversational interaction. The prototype, Suzume-chan, is a small, soft AI agent running locally with a language model and retrieval-augmented generation (RAG). It learns from spoken explanations and responds through dialogue, reducing psychological distance and making knowledge sharing warmer and more human-centered.

**arXiv ID:** 2512.09932
</details>

<details>
<summary><strong>Detailed balance in large language model-driven agents</strong> - Zhuo-Yang Song, Qing-Hong Cao, Ming-xing Luo, Hua Xing Zhu - [[pdf]](https://arxiv.org/pdf/2512.10047)</summary>

**Abstract:** Large language model (LLM)-driven agents are emerging as a powerful new paradigm for solving complex problems. Despite the empirical success of these practices, a theoretical framework to understand and unify their macroscopic dynamics remains lacking. This Letter proposes a method based on the least action principle to estimate the underlying generative directionality of LLMs embedded within agents. By experimentally measuring the transition probabilities between LLM-generated states, we statistically discover a detailed balance in LLM-generated transitions, indicating that LLM generation may not be achieved by generally learning rule sets and strategies, but rather by implicitly learning a class of underlying potential functions that may transcend different LLM architectures and prompt templates. To our knowledge, this is the first discovery of a macroscopic physical law in LLM generative dynamics that does not depend on specific model details. This work is an attempt to establish a macroscopic dynamics theory of complex AI systems, aiming to elevate the study of AI agents from a collection of engineering practices to a science built on effective measurements that are predictable and quantifiable.

**arXiv ID:** 2512.10047
</details>

<details>
<summary><strong>Dynamics of Agentic Loops in Large Language Models: A Geometric Theory of Trajectories</strong> - Nicolas Tacheny - [[pdf]](https://arxiv.org/pdf/2512.10350)</summary>

**Abstract:** Agentic systems built on large language models operate through recursive feedback loops, where each output becomes the next input. Yet the geometric behavior of these agentic loops (whether they converge, diverge, or exhibit more complex dynamics) remains poorly understood. This paper introduces a geometric framework for analyzing agentic trajectories in semantic embedding space, treating iterative transformations as discrete dynamical systems.
We distinguish the artifact space, where linguistic transformations occur, from the embedding space, where geometric measurements are performed. Because cosine similarity is biased by embedding anisotropy, we introduce an isotonic calibration that eliminates systematic bias and aligns similarities with human semantic judgments while preserving high local stability. This enables rigorous measurement of trajectories, clusters and attractors.
Through controlled experiments on singular agentic loops, we identify two fundamental regimes. A contractive rewriting loop converges toward a stable attractor with decreasing dispersion, while an exploratory summarize and negate loop produces unbounded divergence with no cluster formation. These regimes display qualitatively distinct geometric signatures of contraction and expansion.
Our results show that prompt design directly governs the dynamical regime of an agentic loop, enabling systematic control of convergence, divergence and trajectory structure in iterative LLM transformations.

**arXiv ID:** 2512.10350
</details>

<details>
<summary><strong>Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense</strong> - Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Yuyu Zhao - [[pdf]](https://arxiv.org/pdf/2412.21051)</summary>

**Abstract:** The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided numerous benefits in our daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks such as Denial of Service (DoS). Recent advancements in the large language models (LLMs) offer promising solutions for security intelligence. By exploiting the powerful capabilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel defense architecture that proactively mitigates various DoS threats in cloud networks. LLM-PD can efficiently make decisions through comprehensive data analysis and sequential reasoning, as well as dynamically create and deploy actionable defense mechanisms. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. Our case study on three distinct DoS attacks demonstrates its remarkable ability in terms of defense effectiveness and efficiency when compared with other existing methods.

**arXiv ID:** 2412.21051
</details>

<details>
<summary><strong>LEO-RobotAgent: A General-purpose Robotic Agent for Language-driven Embodied Operator</strong> - Lihuang Chen, Xiangyu Luo, Jun Meng - [[pdf]](https://arxiv.org/pdf/2512.10605)</summary>

**Abstract:** We propose LEO-RobotAgent, a general-purpose language-driven intelligent agent framework for robots. Under this framework, LLMs can operate different types of robots to complete unpredictable complex tasks across various scenarios. This framework features strong generalization, robustness, and efficiency. The application-level system built around it can fully enhance bidirectional human-robot intent understanding and lower the threshold for human-robot interaction. Regarding robot task planning, the vast majority of existing studies focus on the application of large models in single-task scenarios and for single robot types. These algorithms often have complex structures and lack generalizability. Thus, the proposed LEO-RobotAgent framework is designed with a streamlined structure as much as possible, enabling large models to independently think, plan, and act within this clear framework. We provide a modular and easily registrable toolset, allowing large models to flexibly call various tools to meet different requirements. Meanwhile, the framework incorporates a human-robot interaction mechanism, enabling the algorithm to collaborate with humans like a partner. Experiments have verified that this framework can be easily adapted to mainstream robot platforms including unmanned aerial vehicles (UAVs), robotic arms, and wheeled robot, and efficiently execute a variety of carefully designed tasks with different complexity levels. Our code is available at this https URL.

**arXiv ID:** 2512.10605
</details>

<details>
<summary><strong>Development and Testing for Perception Based Autonomous Landing of a Long-Range QuadPlane</strong> - Ashik E Rasul, Humaira Tasnim, Ji Yu Kim, Young Hyun Lim, Scott Schmitz, Bruce W. Jo, Hyung-Jin Yoon - [[pdf]](https://arxiv.org/pdf/2512.09343)</summary>

**Abstract:** QuadPlanes combine the range efficiency of fixed-wing aircraft with the maneuverability of multi-rotor platforms for long-range autonomous missions. In GPS-denied or cluttered urban environments, perception-based landing is vital for reliable operation. Unlike structured landing zones, real-world sites are unstructured and highly variable, requiring strong generalization capabilities from the perception system. Deep neural networks (DNNs) provide a scalable solution for learning landing site features across diverse visual and environmental conditions. While perception-driven landing has been shown in simulation, real-world deployment introduces significant challenges. Payload and volume constraints limit high-performance edge AI devices like the NVIDIA Jetson Orin Nano, which are crucial for real-time detection and control. Accurate pose estimation during descent is necessary, especially in the absence of GPS, and relies on dependable visual-inertial odometry. Achieving this with limited edge AI resources requires careful optimization of the entire deployment framework. The flight characteristics of large QuadPlanes further complicate the problem. These aircraft exhibit high inertia, reduced thrust vectoring, and slow response times further complicate stable landing maneuvers. This work presents a lightweight QuadPlane system for efficient vision-based autonomous landing and visual-inertial odometry, specifically developed for long-range QuadPlane operations such as aerial monitoring. It describes the hardware platform, sensor configuration, and embedded computing architecture designed to meet demanding real-time, physical constraints. This establishes a foundation for deploying autonomous landing in dynamic, unstructured, GPS-denied environments.

**arXiv ID:** 2512.09343
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (15 papers)</h2></summary>

<details>
<summary><strong>An exploration for higher efficiency in multi objective optimisation with reinforcement learning</strong> - Mehmet Emin Aydin - [[pdf]](https://arxiv.org/pdf/2512.10208)</summary>

**Abstract:** Efficiency in optimisation and search processes persists to be one of the challenges, which affects the performance and use of optimisation algorithms. Utilising a pool of operators instead of a single operator to handle move operations within a neighbourhood remains promising, but an optimum or near optimum sequence of operators necessitates further investigation. One of the promising ideas is to generalise experiences and seek how to utilise it. Although numerous works are done around this issue for single objective optimisation, multi-objective cases have not much been touched in this regard. A generalised approach based on multi-objective reinforcement learning approach seems to create remedy for this issue and offer good solutions. This paper overviews a generalisation approach proposed with certain stages completed and phases outstanding that is aimed to help demonstrate the efficiency of using multi-objective reinforcement learning.

**arXiv ID:** 2512.10208
</details>

<details>
<summary><strong>AgentProg: Empowering Long-Horizon GUI Agents with Program-Guided Context Management</strong> - Shizuo Tian, Hao Wen, Yuxuan Chen, Jiacheng Liu, Shanhui Zhao, Guohong Liu, Ju Ren, Yunxin Liu, Yuanchun Li - [[pdf]](https://arxiv.org/pdf/2512.10371)</summary>

**Abstract:** The rapid development of mobile GUI agents has stimulated growing research interest in long-horizon task automation. However, building agents for these tasks faces a critical bottleneck: the reliance on ever-expanding interaction history incurs substantial context overhead. Existing context management and compression techniques often fail to preserve vital semantic information, leading to degraded task performance. We propose AgentProg, a program-guided approach for agent context management that reframes the interaction history as a program with variables and control flow. By organizing information according to the structure of program, this structure provides a principled mechanism to determine which information should be retained and which can be discarded. We further integrate a global belief state mechanism inspired by Belief MDP framework to handle partial observability and adapt to unexpected environmental changes. Experiments on AndroidWorld and our extended long-horizon task suite demonstrate that AgentProg has achieved the state-of-the-art success rates on these benchmarks. More importantly, it maintains robust performance on long-horizon tasks while baseline methods experience catastrophic degradation. Our system is open-sourced at this https URL.

**arXiv ID:** 2512.10371
</details>

<details>
<summary><strong>Enhancing Radiology Report Generation and Visual Grounding using Reinforcement Learning</strong> - Benjamin Gundersen, Nicolas Deperrois, Samuel Ruiperez-Campillo, Thomas M. Sutter, Julia E. Vogt, Michael Moor, Farhad Nooralahzadeh, Michael Krauthammer - [[pdf]](https://arxiv.org/pdf/2512.10691)</summary>

**Abstract:** Recent advances in vision-language models (VLMs) have improved Chest X-ray (CXR) interpretation in multiple aspects. However, many medical VLMs rely solely on supervised fine-tuning (SFT), which optimizes next-token prediction without evaluating answer quality. In contrast, reinforcement learning (RL) can incorporate task-specific feedback, and its combination with explicit intermediate reasoning ("thinking") has demonstrated substantial gains on verifiable math and coding tasks. To investigate the effects of RL and thinking in a CXR VLM, we perform large-scale SFT on CXR data to build an updated RadVLM based on Qwen3-VL, followed by a cold-start SFT stage that equips the model with basic thinking ability. We then apply Group Relative Policy Optimization (GRPO) with clinically grounded, task-specific rewards for report generation and visual grounding, and run matched RL experiments on both domain-specific and general-domain Qwen3-VL variants, with and without thinking. Across these settings, we find that while strong SFT remains crucial for high base performance, RL provides additional gains on both tasks, whereas explicit thinking does not appear to further improve results. Under a unified evaluation pipeline, the RL-optimized RadVLM models outperform their baseline counterparts and reach state-of-the-art performance on both report generation and grounding, highlighting clinically aligned RL as a powerful complement to SFT for medical VLMs.

**arXiv ID:** 2512.10691
</details>

<details>
<summary><strong>Confucius Code Agent: An Open-sourced AI Software Engineer at Industrial Scale</strong> - Zhaodong Wang, Zhenting Qi, Sherman Wong, Nathan Hu, Samuel Lin, Jun Ge, Erwin Gao, Yining Yang, Ben Maurer, Wenlin Chen, David Recordon, Yilun Du, Minlan Yu, Ying Zhang - [[pdf]](https://arxiv.org/pdf/2512.10398)</summary>

**Abstract:** Real-world AI software engineering demands coding agents that can reason over massive repositories, maintain durable memory across and within long sessions, and robustly coordinate complex toolchains at test time. Existing open-source coding agents provide transparency but frequently fall short when pushed to these industrial-scale workloads, while proprietary coding agents offer strong practical performance but limited extensibility, interpretability, and controllability. We present the Confucius Code Agent (CCA), an open-sourced AI software engineer that can operate at an industrial scale. CCA is built atop the Confucius SDK, an open-sourced agent development platform designed around three complementary perspectives: Agent Experience (AX), User Experience (UX), and Developer Experience (DX). The SDK introduces a unified orchestrator with hierarchical working memory for long-context reasoning, a persistent note-taking system for cross-session continual learning, and a modular extension module for robust tool use. Moreover, a meta-agent automates the synthesis, evaluation, and refinement of agent configurations through a build-test-improve loop, enabling rapid agent development on new tasks, environments, and tool stacks. Instantiated on Confucius SDK with these mechanisms, CCA delivers strong performance on real-world software engineering tasks. On SWE-Bench-Pro, CCA achieves a state-of-the-art Resolve@1 performance of 54.3%, substantially improving over prior coding agents. Together, the Confucius SDK and CCA provide a transparent, extensible, and reproducible foundation for AI agents, bridge gaps between research prototypes and production-grade systems, and support agent development and deployment at industrial scale.

**arXiv ID:** 2512.10398
</details>

<details>
<summary><strong>UACER: An Uncertainty-Aware Critic Ensemble Framework for Robust Adversarial Reinforcement Learning</strong> - Jiaxi Wu, Tiantian Zhang, Yuxing Wang, Yongzhe Chang, Xueqian Wang - [[pdf]](https://arxiv.org/pdf/2512.10492)</summary>

**Abstract:** Robust adversarial reinforcement learning has emerged as an effective paradigm for training agents to handle uncertain disturbance in real environments, with critical applications in sequential decision-making domains such as autonomous driving and robotic control. Within this paradigm, agent training is typically formulated as a zero-sum Markov game between a protagonist and an adversary to enhance policy robustness. However, the trainable nature of the adversary inevitably induces non-stationarity in the learning dynamics, leading to exacerbated training instability and convergence difficulties, particularly in high-dimensional complex environments. In this paper, we propose a novel approach, Uncertainty-Aware Critic Ensemble for robust adversarial Reinforcement learning (UACER), which consists of two strategies: 1) Diversified critic ensemble: a diverse set of K critic networks is exploited in parallel to stabilize Q-value estimation rather than conventional single-critic architectures for both variance reduction and robustness enhancement. 2) Time-varying Decay Uncertainty (TDU) mechanism: advancing beyond simple linear combinations, we develop a variance-derived Q-value aggregation strategy that explicitly incorporates epistemic uncertainty to dynamically regulate the exploration-exploitation trade-off while simultaneously stabilizing the training process. Comprehensive experiments across several MuJoCo control problems validate the superior effectiveness of UACER, outperforming state-of-the-art methods in terms of overall performance, stability, and efficiency.

**arXiv ID:** 2512.10492
</details>

<details>
<summary><strong>Adaptive Replay Buffer for Offline-to-Online Reinforcement Learning</strong> - Chihyeon Song, Jaewoo Lee, Jinkyoo Park - [[pdf]](https://arxiv.org/pdf/2512.10510)</summary>

**Abstract:** Offline-to-Online Reinforcement Learning (O2O RL) faces a critical dilemma in balancing the use of a fixed offline dataset with newly collected online experiences. Standard methods, often relying on a fixed data-mixing ratio, struggle to manage the trade-off between early learning stability and asymptotic performance. To overcome this, we introduce the Adaptive Replay Buffer (ARB), a novel approach that dynamically prioritizes data sampling based on a lightweight metric we call 'on-policyness'. Unlike prior methods that rely on complex learning procedures or fixed ratios, ARB is designed to be learning-free and simple to implement, seamlessly integrating into existing O2O RL algorithms. It assesses how closely collected trajectories align with the current policy's behavior and assigns a proportional sampling weight to each transition within that trajectory. This strategy effectively leverages offline data for initial stability while progressively focusing learning on the most relevant, high-rewarding online experiences. Our extensive experiments on D4RL benchmarks demonstrate that ARB consistently mitigates early performance degradation and significantly improves the final performance of various O2O RL algorithms, highlighting the importance of an adaptive, behavior-aware replay buffer design.

**arXiv ID:** 2512.10510
</details>

<details>
<summary><strong>How to Brake? Ethical Emergency Braking with Deep Reinforcement Learning</strong> - Jianbo Wang, Galina Sidorenko, Johan Thunberg - [[pdf]](https://arxiv.org/pdf/2512.10698)</summary>

**Abstract:** Connected and automated vehicles (CAVs) have the potential to enhance driving safety, for example by enabling safe vehicle following and more efficient traffic scheduling. For such future deployments, safety requirements should be addressed, where the primary such are avoidance of vehicle collisions and substantial mitigating of harm when collisions are unavoidable. However, conservative worst-case-based control strategies come at the price of reduced flexibility and may compromise overall performance. In light of this, we investigate how Deep Reinforcement Learning (DRL) can be leveraged to improve safety in multi-vehicle-following scenarios involving emergency braking. Specifically, we investigate how DRL with vehicle-to-vehicle communication can be used to ethically select an emergency breaking profile in scenarios where overall, or collective, three-vehicle harm reduction or collision avoidance shall be obtained instead of single-vehicle such. As an algorithm, we provide a hybrid approach that combines DRL with a previously published method based on analytical expressions for selecting optimal constant deceleration. By combining DRL with the previous method, the proposed hybrid approach increases the reliability compared to standalone DRL, while achieving superior performance in terms of overall harm reduction and collision avoidance.

**arXiv ID:** 2512.10698
</details>

<details>
<summary><strong>MedReasoner: Reinforcement Learning Drives Reasoning Grounding from Clinical Thought to Pixel-Level Precision</strong> - Zhonghao Yan, Muxi Diao, Yuxuan Yang, Ruoyan Jing, Jiayuan Xu, Kaizhou Zhang, Lele Yang, Yanxi Liu, Kongming Liang, Zhanyu Ma - [[pdf]](https://arxiv.org/pdf/2508.08177)</summary>

**Abstract:** Accurately grounding regions of interest (ROIs) is critical for diagnosis and treatment planning in medical imaging. While multimodal large language models (MLLMs) combine visual perception with natural language, current medical-grounding pipelines still rely on supervised fine-tuning with explicit spatial hints, making them ill-equipped to handle the implicit queries common in clinical practice. This work makes three core contributions. We first define Unified Medical Reasoning Grounding (UMRG), a novel vision-language task that demands clinical reasoning and pixel-level grounding. Second, we release U-MRG-14K, a dataset of 14K samples featuring pixel-level masks alongside implicit clinical queries and reasoning traces, spanning 10 modalities, 15 super-categories, and 108 specific categories. Finally, we introduce MedReasoner, a modular framework that distinctly separates reasoning from segmentation: an MLLM reasoner is optimized with reinforcement learning, while a frozen segmentation expert converts spatial prompts into masks, with alignment achieved through format and accuracy rewards. MedReasoner achieves state-of-the-art performance on U-MRG-14K and demonstrates strong generalization to unseen clinical queries, underscoring the significant promise of reinforcement learning for interpretable medical grounding.

**arXiv ID:** 2508.08177
</details>

<details>
<summary><strong>Optimizing the non-Clifford-count in unitary synthesis using Reinforcement Learning</strong> - David Kremer, Ali Javadi-Abhari, Priyanka Mukhopadhyay - [[pdf]](https://arxiv.org/pdf/2509.21709)</summary>

**Abstract:** In this paper we study the potential of using reinforcement learning (RL) in order to synthesize quantum circuits, while optimizing the T-count and CS-count, of unitaries that are exactly implementable by the Clifford+T and Clifford+CS gate sets, respectively. We have designed our RL framework to work with channel representation of unitaries, that enables us to perform matrix operations efficiently, using integers only. We have also incorporated pruning heuristics and a canonicalization of operators, in order to reduce the search complexity. As a result, compared to previous works, we are able to implement significantly larger unitaries, in less time, with much better success rate and improvement factor. Our results for Clifford+T synthesis on two qubit unitaries achieve close-to-optimal decompositions for up to 100 T gates, 5 times more than previous RL algorithms and to the best of our knowledge, the largest instances achieved with any method to date. Our RL algorithm is able to recover previously-known optimal linear complexity algorithm for T-count-optimal decomposition of 1 qubit unitaries. We illustrate significant reduction in the asymptotic T-count estimate of important primitives like controlled cyclic shift (43%), controlled adder (14.3%) and multiplier (14%), without adding any extra ancilla. For 2-qubit Clifford+CS unitaries, our algorithm achieves a linear complexity, something that could only be accomplished by a previous algorithm using SO(6) representation.

**arXiv ID:** 2509.21709
</details>

<details>
<summary><strong>RLAX: Large-Scale, Distributed Reinforcement Learning for Large Language Models on TPUs</strong> - Runlong Zhou, Lefan Zhang, Shang-Chen Wu, Kelvin Zou, Hanzhi Zhou, Ke Ye, Yihao Feng, Dong Yin, Alex Guillen Garcia, Dmytro Babych, Rohit Chatterjee, Matthew Hopkins, Xiang Kong, Chang Lan, Lezhi Li, Yiping Ma, Daniele Molinari, Senyu Tong, Yanchao Sun, Thomas Voice, Jianyu Wang, Chong Wang, Simon Wang, Floris Weers, Yechen Xu, Guolin Yin, Muyang Yu, Yi Zhang, Zheng Zhou, Danyang Zhuo, Ruoming Pang, Cheng Leong - [[pdf]](https://arxiv.org/pdf/2512.06392)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as the de-facto paradigm for improving the reasoning capabilities of large language models (LLMs). We have developed RLAX, a scalable RL framework on TPUs. RLAX employs a parameter-server architecture. A master trainer periodically pushes updated model weights to the parameter server while a fleet of inference workers pull the latest weights and generates new rollouts. We introduce a suite of system techniques to enable scalable and preemptible RL for a diverse set of state-of-art RL algorithms. To accelerate convergence and improve model quality, we have devised new dataset curation and alignment techniques. Large-scale evaluations show that RLAX improves QwQ-32B's pass@8 accuracy by 12.8% in just 12 hours 48 minutes on 1024 v5p TPUs, while remaining robust to preemptions during training.

**arXiv ID:** 2512.06392
</details>

<details>
<summary><strong>UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action</strong> - Yuhao Yang, Zhen Yang, Zi-Yi Dou, Anh Nguyen, Keen You, Omar Attia, Andrew Szot, Michael Feng, Ram Ramrakhya, Alexander Toshev, Chao Huang, Yinfei Yang, Zhe Gan - [[pdf]](https://arxiv.org/pdf/2510.17790)</summary>

**Abstract:** Computer-use agents face a fundamental limitation. They rely exclusively on primitive GUI actions (click, type, scroll), creating brittle execution chains prone to cascading failures. While API-driven agents harness rich capabilities through structured interfaces and tools, computer-use agents remain constrained to low-level visual interactions. We present UltraCUA, a foundation model that transcends this limitation through hybrid action-seamlessly unifying primitive GUI operations with high-level tool execution. Our innovation rests on four critical advances. First, an automated pipeline extracts and scales tool capabilities from software documentation and code repositories. Second, a synthetic data engine produces 17,000+ verifiable tasks capturing real-world computer-use complexity. Third, comprehensive hybrid action trajectory collection incorporates both GUI primitives and strategic tool calls. Fourth, a two-stage training methodology combines supervised fine-tuning with online reinforcement learning, enabling intelligent action selection between GUI and API. Evaluation with our 7B and 32B UltraCUA models reveals transformative performance gains. On OSWorld, UltraCUA achieves 22% relative improvement while executing 11% faster than existing approaches, averagely. Cross-domain validation on WindowsAgentArena demonstrates robust generalization with 21.7% success rate, surpassing Windows-trained baselines. The hybrid action paradigm proves essential, reducing error propagation while improving execution efficiency. This work establishes a scalable paradigm bridging primitive GUI interactions and high-level tool intelligence, enabling more resilient and adaptable computer use agents for diverse environments and complex real-world tasks.

**arXiv ID:** 2510.17790
</details>

<details>
<summary><strong>Digital Twin Supervised Reinforcement Learning Framework for Autonomous Underwater Navigation</strong> - Zamirddine Mari, Mohamad Motasem Nawaf, Pierre Drap - [[pdf]](https://arxiv.org/pdf/2512.10925)</summary>

**Abstract:** Autonomous navigation in underwater environments remains a major challenge due to the absence of GPS, degraded visibility, and the presence of submerged obstacles. This article investigates these issues through the case of the BlueROV2, an open platform widely used for scientific experimentation. We propose a deep reinforcement learning approach based on the Proximal Policy Optimization (PPO) algorithm, using an observation space that combines target-oriented navigation information, a virtual occupancy grid, and ray-casting along the boundaries of the operational area. The learned policy is compared against a reference deterministic kinematic planner, the Dynamic Window Approach (DWA), commonly employed as a robust baseline for obstacle avoidance. The evaluation is conducted in a realistic simulation environment and complemented by validation on a physical BlueROV2 supervised by a 3D digital twin of the test site, helping to reduce risks associated with real-world experimentation. The results show that the PPO policy consistently outperforms DWA in highly cluttered environments, notably thanks to better local adaptation and reduced collisions. Finally, the experiments demonstrate the transferability of the learned behavior from simulation to the real world, confirming the relevance of deep RL for autonomous navigation in underwater robotics.

**arXiv ID:** 2512.10925
</details>

<details>
<summary><strong>TDC-Cache: A Trustworthy Decentralized Cooperative Caching Framework for Web3.0</strong> - Jinyu Chen, Long Shi, Taotao Wang, Jiaheng Wang, Wei Zhang - [[pdf]](https://arxiv.org/pdf/2512.09961)</summary>

**Abstract:** The rapid growth of Web3.0 is transforming the Internet from a centralized structure to decentralized, which empowers users with unprecedented self-sovereignty over their own data. However, in the context of decentralized data access within Web3.0, it is imperative to cope with efficiency concerns caused by the replication of redundant data, as well as security vulnerabilities caused by data inconsistency. To address these challenges, we develop a Trustworthy Decentralized Cooperative Caching (TDC-Cache) framework for Web3.0 to ensure efficient caching and enhance system resilience against adversarial threats. This framework features a two-layer architecture, wherein the Decentralized Oracle Network (DON) layer serves as a trusted intermediary platform for decentralized caching, bridging the contents from decentralized storage and the content requests from users. In light of the complexity of Web3.0 network topologies and data flows, we propose a Deep Reinforcement Learning-Based Decentralized Caching (DRL-DC) for TDC-Cache to dynamically optimize caching strategies of distributed oracles. Furthermore, we develop a Proof of Cooperative Learning (PoCL) consensus to maintain the consistency of decentralized caching decisions within DON. Experimental results show that, compared with existing approaches, the proposed framework reduces average access latency by 20%, increases the cache hit rate by at most 18%, and improves the average success consensus rate by 10%. Overall, this paper serves as a first foray into the investigation of decentralized caching framework and strategy for Web3.0.

**arXiv ID:** 2512.09961
</details>

<details>
<summary><strong>Curriculum-Based Reinforcement Learning for Autonomous UAV Navigation in Unknown Curved Tubular Conduit</strong> - Zamirddine Mari, Jérôme Pasquet, Julien Seinturier - [[pdf]](https://arxiv.org/pdf/2512.10934)</summary>

**Abstract:** Autonomous drone navigation in confined tubular environments remains a major challenge due to the constraining geometry of the conduits, the proximity of the walls, and the perceptual limitations inherent to such scenarios. We propose a reinforcement learning approach enabling a drone to navigate unknown three-dimensional tubes without any prior knowledge of their geometry, relying solely on local observations from LiDAR and a conditional visual detection of the tube center. In contrast, the Pure Pursuit algorithm, used as a deterministic baseline, benefits from explicit access to the centerline, creating an information asymmetry designed to assess the ability of RL to compensate for the absence of a geometric model. The agent is trained through a progressive Curriculum Learning strategy that gradually exposes it to increasingly curved geometries, where the tube center frequently disappears from the visual field. A turning-negotiation mechanism, based on the combination of direct visibility, directional memory, and LiDAR symmetry cues, proves essential for ensuring stable navigation under such partial observability conditions. Experiments show that the PPO policy acquires robust and generalizable behavior, consistently outperforming the deterministic controller despite its limited access to geometric information. Validation in a high-fidelity 3D environment further confirms the transferability of the learned behavior to a continuous physical dynamics.
The proposed approach thus provides a complete framework for autonomous navigation in unknown tubular environments and opens perspectives for industrial, underground, or medical applications where progressing through narrow and weakly perceptive conduits represents a central challenge.

**arXiv ID:** 2512.10934
</details>

<details>
<summary><strong>Task-Oriented Grasping Using Reinforcement Learning with a Contextual Reward Machine</strong> - Hui Li, Akhlak Uz Zaman, Fujian Yan, Hongsheng He - [[pdf]](https://arxiv.org/pdf/2512.10235)</summary>

**Abstract:** This paper presents a reinforcement learning framework that incorporates a Contextual Reward Machine for task-oriented grasping. The Contextual Reward Machine reduces task complexity by decomposing grasping tasks into manageable sub-tasks. Each sub-task is associated with a stage-specific context, including a reward function, an action space, and a state abstraction function. This contextual information enables efficient intra-stage guidance and improves learning efficiency by reducing the state-action space and guiding exploration within clearly defined boundaries. In addition, transition rewards are introduced to encourage or penalize transitions between stages which guides the model toward desirable stage sequences and further accelerates convergence. When integrated with the Proximal Policy Optimization algorithm, the proposed method achieved a 95% success rate across 1,000 simulated grasping tasks encompassing diverse objects, affordances, and grasp topologies. It outperformed the state-of-the-art methods in both learning speed and success rate. The approach was transferred to a real robot, where it achieved a success rate of 83.3% in 60 grasping tasks over six affordances. These experimental results demonstrate superior accuracy, data efficiency, and learning efficiency. They underscore the model's potential to advance task-oriented grasping in both simulated and real-world settings.

**arXiv ID:** 2512.10235
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
