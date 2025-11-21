# Agent arXiv Daily

**Last Updated:** 2025-11-21 02:48:24

**Total Papers:** 53

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
<summary><strong>Secure Autonomous Agent Payments: Verifying Authenticity and Intent in a Trustless Environment</strong> - Vivek Acharya - [[pdf]](https://arxiv.org/pdf/2511.15712)</summary>

**Abstract:** Artificial intelligence (AI) agents are increasingly capable of initiating financial transactions on behalf of users or other agents. This evolution introduces a fundamental challenge: verifying both the authenticity of an autonomous agent and the true intent behind its transactions in a decentralized, trustless environment. Traditional payment systems assume human authorization, but autonomous, agent-led payments remove that safeguard. This paper presents a blockchain-based framework that cryptographically authenticates and verifies the intent of every AI-initiated transaction. The proposed system leverages decentralized identity (DID) standards and verifiable credentials to establish agent identities, on-chain intent proofs to record user authorization, and zero-knowledge proofs (ZKPs) to preserve privacy while ensuring policy compliance. Additionally, secure execution environments (TEE-based attestations) guarantee the integrity of agent reasoning and execution. The hybrid on-chain/off-chain architecture provides an immutable audit trail linking user intent to payment outcome. Through qualitative analysis, the framework demonstrates strong resistance to impersonation, unauthorized transactions, and misalignment of intent. This work lays the foundation for secure, auditable, and intent-aware autonomous economic agents, enabling a future of verifiable trust and accountability in AI-driven financial ecosystems.

**arXiv ID:** 2511.15712
</details>

<details>
<summary><strong>Just Asking Questions: Doing Our Own Research on Conspiratorial Ideation by Generative AI Chatbots</strong> - Katherine M. FitzGerald, Michelle Riedlinger, Axel Bruns, Stephen Harrington, Timothy Graham, Daniel Angus - [[pdf]](https://arxiv.org/pdf/2511.15732)</summary>

**Abstract:** Interactive chat systems that build on artificial intelligence frameworks are increasingly ubiquitous and embedded into search engines, Web browsers, and operating systems, or are available on websites and apps. Researcher efforts have sought to understand the limitations and potential for harm of generative AI, which we contribute to here. Conducting a systematic review of six AI-powered chat systems (ChatGPT 3.5; ChatGPT 4 Mini; Microsoft Copilot in Bing; Google Search AI; Perplexity; and Grok in Twitter/X), this study examines how these leading products respond to questions related to conspiracy theories. This follows the platform policy implementation audit approach established by Glazunova et al. (2023). We select five well-known and comprehensively debunked conspiracy theories and four emerging conspiracy theories that relate to breaking news events at the time of data collection. Our findings demonstrate that the extent of safety guardrails against conspiratorial ideation in generative AI chatbots differs markedly, depending on chatbot model and conspiracy theory. Our observations indicate that safety guardrails in AI chatbots are often very selectively designed: generative AI companies appear to focus especially on ensuring that their products are not seen to be racist; they also appear to pay particular attention to conspiracy theories that address topics of substantial national trauma such as 9/11 or relate to well-established political issues. Future work should include an ongoing effort extended to further platforms, multiple languages, and a range of conspiracy theories extending well beyond the United States.

**arXiv ID:** 2511.15732
</details>

<details>
<summary><strong>A Crowdsourced Study of ChatBot Influence in Value-Driven Decision Making Scenarios</strong> - Anthony Wise, Xinyi Zhou, Martin Reimann, Anind Dey, Leilani Battle - [[pdf]](https://arxiv.org/pdf/2511.15857)</summary>

**Abstract:** Similar to social media bots that shape public opinion, healthcare and financial decisions, LLM-based ChatBots like ChatGPT can persuade users to alter their behavior. Unlike prior work that persuades via overt-partisan bias or misinformation, we test whether framing alone suffices. We conducted a crowdsourced study, where 336 participants interacted with a neutral or one of two value-framed ChatBots while deciding to alter US defense spending. In this single policy domain with controlled content, participants exposed to value-framed ChatBots significantly changed their budget choices relative to the neutral control. When the frame misaligned with their values, some participants reinforced their original preference, revealing a potentially replicable backfire effect, originally considered rare in the literature. These findings suggest that value-framing alone lowers the barrier for manipulative uses of LLMs, revealing risks distinct from overt bias or misinformation, and clarifying risks to countering misinformation.

**arXiv ID:** 2511.15857
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (6 papers)</h2></summary>

<details>
<summary><strong>Graph-Memoized Reasoning: Foundations Structured Workflow Reuse in Intelligent Systems</strong> - Yash Raj Singh - [[pdf]](https://arxiv.org/pdf/2511.15715)</summary>

**Abstract:** Modern large language model-based reasoning systems frequently recompute similar reasoning steps across tasks, wasting computational resources, inflating inference latency, and limiting reproducibility. These inefficiencies underscore the need for persistent reasoning mechanisms that can recall and reuse prior computational traces.
We introduce Graph-Memoized Reasoning, a formal framework for representing, storing, and reusing reasoning workflows as graph-structured memory. By encoding past decision graphs and retrieving them through structural and semantic similarity, our approach enables compositional reuse of subgraphs across new reasoning tasks.
We formulate an optimization objective that minimizes total reasoning cost regularized by inconsistency between stored and generated workflows, providing a theoretical foundation for efficiency-consistency trade-offs in intelligent systems. We outline a conceptual evaluation protocol aligned with the proposed optimization objective.
This framework establishes the groundwork for interpretable, cost-efficient, and self-improving reasoning architectures, offering a step toward persistent memory in large-scale agentic systems.

**arXiv ID:** 2511.15715
</details>

<details>
<summary><strong>An Agent-Based Framework for the Automatic Validation of Mathematical Optimization Models</strong> - Alexander Zadorojniy, Segev Wasserkrug, Eitan Farchi - [[pdf]](https://arxiv.org/pdf/2511.16383)</summary>

**Abstract:** Recently, using Large Language Models (LLMs) to generate optimization models from natural language descriptions has became increasingly popular. However, a major open question is how to validate that the generated models are correct and satisfy the requirements defined in the natural language description. In this work, we propose a novel agent-based method for automatic validation of optimization models that builds upon and extends methods from software testing to address optimization modeling . This method consists of several agents that initially generate a problem-level testing API, then generate tests utilizing this API, and, lastly, generate mutations specific to the optimization model (a well-known software testing technique assessing the fault detection power of the test suite). In this work, we detail this validation framework and show, through experiments, the high quality of validation provided by this agent ensemble in terms of the well-known software testing measure called mutation coverage.

**arXiv ID:** 2511.16383
</details>

<details>
<summary><strong>Practical and Stealthy Touch-Guided Jailbreak Attacks on Deployed Mobile Vision-Language Agents</strong> - Renhua Ding, Xiao Yang, Zhengwei Fang, Jun Luo, Kun He, Jun Zhu - [[pdf]](https://arxiv.org/pdf/2510.07809)</summary>

**Abstract:** Large vision-language models (LVLMs) enable autonomous mobile agents to operate smartphone user interfaces, yet vulnerabilities in their perception and interaction remain critically understudied. Existing research often relies on conspicuous overlays, elevated permissions, or unrealistic threat assumptions, limiting stealth and real-world feasibility. In this paper, we introduce a practical and stealthy jailbreak attack framework, which comprises three key components: (i) non-privileged perception compromise, which injects visual payloads into the application interface without requiring elevated system permissions; (ii) agent-attributable activation, which leverages input attribution signals to distinguish agent from human interactions and limits prompt exposure to transient intervals to preserve stealth from end users; and (iii) efficient one-shot jailbreak, a heuristic iterative deepening search algorithm (HG-IDA*) that performs keyword-level detoxification to bypass built-in safety alignment of LVLMs. Moreover, we developed three representative Android applications and curated a prompt-injection dataset for mobile agents. We evaluated our attack across multiple LVLM backends, including closed-source services and representative open-source models, and observed high planning and execution hijack rates (e.g., GPT-4o: 82.5% planning / 75.0% execution), exposing a fundamental security vulnerability in current mobile agents and underscoring critical implications for autonomous smartphone operation.

**arXiv ID:** 2510.07809
</details>

<details>
<summary><strong>Orion: A Unified Visual Agent for Multimodal Perception, Advanced Visual Reasoning and Execution</strong> - N Dinesh Reddy, Dylan Snyder, Lona Kiragu, Mirajul Mohin, Shahrear Bin Amin, Sudeep Pillai - [[pdf]](https://arxiv.org/pdf/2511.14210)</summary>

**Abstract:** We introduce Orion, a visual agent that integrates vision-based reasoning with tool-augmented execution to achieve powerful, precise, multi-step visual intelligence across images, video, and documents. Unlike traditional vision-language models that generate descriptive outputs, Orion orchestrates a suite of specialized computer vision tools, including object detection, keypoint localization, panoptic segmentation, Optical Character Recognition (OCR), and geometric analysis, to execute complex multi-step visual workflows. The system achieves competitive performance across MMMU, MMBench, DocVQA, and MMLongBench while extending monolithic VLM capabilities to production-grade visual intelligence. Through its agentic, tool-augmented approach, Orion enables autonomous visual reasoning that bridges neural perception with symbolic execution, marking the transition from passive visual understanding to active, tool-driven visual intelligence.
Try Orion for free at: this https URL
Learn more at: this https URL

**arXiv ID:** 2511.14210
</details>

<details>
<summary><strong>MiMo-Embodied: X-Embodied Foundation Model Technical Report</strong> - Xiaoshuai Hao, Lei Zhou, Zhijian Huang, Zhiwen Hou, Yingbo Tang, Lingfeng Zhang, Guang Li, Zheng Lu, Shuhuai Ren, Xianhui Meng, Yuchen Zhang, Jing Wu, Jinghui Lu, Chenxu Dang, Jiayi Guan, Jianhua Wu, Zhiyi Hou, Hanbing Li, Shumeng Xia, Mingliang Zhou, Yinan Zheng, Zihao Yue, Shuhao Gu, Hao Tian, Yuannan Shen, Jianwei Cui, Wen Zhang, Shaoqing Xu, Bing Wang, Haiyang Sun, Zeyu Zhu, Yuncheng Jiang, Zibin Guo, Chuhong Gong, Chaofan Zhang, Wenbo Ding, Kun Ma, Guang Chen, Rui Cai, Diyun Xiang, Heng Qu, Fuli Luo, Hangjun Ye, Long Chen - [[pdf]](https://arxiv.org/pdf/2511.16518)</summary>

**Abstract:** We open-source MiMo-Embodied, the first cross-embodied foundation model to successfully integrate and achieve state-of-the-art performance in both Autonomous Driving and Embodied AI. MiMo-Embodied sets new records across 17 embodied AI benchmarks in Task Planning, Affordance Prediction and Spatial Understanding, while also excelling in 12 autonomous driving benchmarks across Environmental Perception, Status Prediction, and Driving Planning. Across these tasks, MiMo-Embodied significantly outperforms existing open-source, closed-source, and specialized baselines. Our results indicate that through multi-stage learning, curated data construction, and CoT/RL fine-tuning, these two domains exhibit strong positive transfer and mutually reinforce one another. We provide a detailed analysis of our model design and training methodologies to facilitate further research. Code and models are available at this https URL.

**arXiv ID:** 2511.16518
</details>

<details>
<summary><strong>The Shawshank Redemption of Embodied AI: Understanding and Benchmarking Indirect Environmental Jailbreaks</strong> - Chunyang Li, Zifeng Kang, Junwei Zhang, Zhuo Ma, Anda Cheng, Xinghua Li, Jianfeng Ma - [[pdf]](https://arxiv.org/pdf/2511.16347)</summary>

**Abstract:** The adoption of Vision-Language Models (VLMs) in embodied AI agents, while being effective, brings safety concerns such as jailbreaking. Prior work have explored the possibility of directly jailbreaking the embodied agents through elaborated multi-modal prompts. However, no prior work has studied or even reported indirect jailbreaks in embodied AI, where a black-box attacker induces a jailbreak without issuing direct prompts to the embodied agent. In this paper, we propose, for the first time, indirect environmental jailbreak (IEJ), a novel attack to jailbreak embodied AI via indirect prompt injected into the environment, such as malicious instructions written on a wall. Our key insight is that embodied AI does not ''think twice'' about the instructions provided by the environment -- a blind trust that attackers can exploit to jailbreak the embodied agent. We further design and implement open-source prototypes of two fully-automated frameworks: SHAWSHANK, the first automatic attack generation framework for the proposed attack IEJ; and SHAWSHANK-FORGE, the first automatic benchmark generation framework for IEJ. Then, using SHAWSHANK-FORGE, we automatically construct SHAWSHANK-BENCH, the first benchmark for indirectly jailbreaking embodied agents. Together, our two frameworks and one benchmark answer the questions of what content can be used for malicious IEJ instructions, where they should be placed, and how IEJ can be systematically evaluated. Evaluation results show that SHAWSHANK outperforms eleven existing methods across 3,957 task-scene combinations and compromises all six tested VLMs. Furthermore, current defenses only partially mitigate our attack, and we have responsibly disclosed our findings to all affected VLM vendors.

**arXiv ID:** 2511.16347
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent</strong> - Shiyi Cao, Dacheng Li, Fangzhou Zhao, Shuo Yuan, Sumanth R. Hegde, Connor Chen, Charlie Ruan, Tyler Griggs, Shu Liu, Eric Tang, Richard Liaw, Philipp Moritz, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica - [[pdf]](https://arxiv.org/pdf/2511.16108)</summary>

**Abstract:** We introduce SkyRL-Agent, a framework for efficient, multi-turn, long-horizon agent training and evaluation. It provides efficient asynchronous dispatching, lightweight tool integration, and flexible backend interoperability, enabling seamless use with existing RL frameworks such as SkyRL-train, VeRL, and Tinker.
Using SkyRL-Agent, we train SA-SWE-32B, a software engineering agent trained from Qwen3-32B (24.4% Pass@1) purely with reinforcement learning. We introduce two key components: an optimized asynchronous pipeline dispatcher that achieves a 1.55x speedup over naive asynchronous batching, and a tool-enhanced training recipe leveraging an AST-based search tool to facilitate code navigation, boost rollout Pass@K, and improve training efficiency. Together, these optimizations enable SA-SWE-32B to reach 39.4% Pass@1 on SWE-Bench Verified with more than 2x cost reduction compared to prior models reaching similar performance. Despite being trained solely on SWE tasks, SA-SWE-32B generalizes effectively to other agentic tasks, including Terminal-Bench, BrowseComp-Plus, and WebArena. We further demonstrate SkyRL-Agent's extensibility through case studies on deep research, computer use, and memory agents, each trained using a different training backend.

**arXiv ID:** 2511.16108
</details>

<details>
<summary><strong>AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture</strong> - Qiming Guo, Bishal Khatri, Wenbo Sun, Jinwen Tang, Hua Zhang, Wenlu Wang - [[pdf]](https://arxiv.org/pdf/2511.15870)</summary>

**Abstract:** Underground pipeline leaks and infiltrations pose significant threats to water security and environmental safety. Traditional manual inspection methods provide limited coverage and delayed response, often missing critical anomalies. This paper proposes AquaSentinel, a novel physics-informed AI system for real-time anomaly detection in urban underground water pipeline networks. We introduce four key innovations: (1) strategic sparse sensor deployment at high-centrality nodes combined with physics-based state augmentation to achieve network-wide observability from minimal infrastructure; (2) the RTCA (Real-Time Cumulative Anomaly) detection algorithm, which employs dual-threshold monitoring with adaptive statistics to distinguish transient fluctuations from genuine anomalies; (3) a Mixture of Experts (MoE) ensemble of spatiotemporal graph neural networks that provides robust predictions by dynamically weighting model contributions; (4) causal flow-based leak localization that traces anomalies upstream to identify source nodes and affected pipe segments. Our system strategically deploys sensors at critical network junctions and leverages physics-based modeling to propagate measurements to unmonitored nodes, creating virtual sensors that enhance data availability across the entire network. Experimental evaluation using 110 leak scenarios demonstrates that AquaSentinel achieves 100% detection accuracy. This work advances pipeline monitoring by demonstrating that physics-informed sparse sensing can match the performance of dense deployments at a fraction of the cost, providing a practical solution for aging urban infrastructure.

**arXiv ID:** 2511.15870
</details>

<details>
<summary><strong>AskDB: An LLM Agent for Natural Language Interaction with Relational Databases</strong> - Xuan-Quang Phan, Tan-Ha Mai, Thai-Duy Dinh, Minh-Thuan Nguyen, Lam-Son Lê - [[pdf]](https://arxiv.org/pdf/2511.16131)</summary>

**Abstract:** Interacting with relational databases remains challenging for users across different expertise levels, particularly when composing complex analytical queries or performing administrative tasks. Existing systems typically address either natural language querying or narrow aspects of database administration, lacking a unified and intelligent interface for general-purpose database interaction. We introduce AskDB, a large language model powered agent designed to bridge this gap by supporting both data analysis and administrative operations over SQL databases through natural language. Built on Gemini 2, AskDB integrates two key innovations: a dynamic schema-aware prompting mechanism that effectively incorporates database metadata, and a task decomposition framework that enables the agent to plan and execute multi-step actions. These capabilities allow AskDB to autonomously debug derived SQL, retrieve contextual information via real-time web search, and adaptively refine its responses. We evaluate AskDB on a widely used Text-to-SQL benchmark and a curated set of DBA tasks, demonstrating strong performance in both analytical and administrative scenarios. Our results highlight the potential of AskDB as a unified and intelligent agent for relational database systems, offering an intuitive and accessible experience for end users.

**arXiv ID:** 2511.16131
</details>

<details>
<summary><strong>Multi-dimensional Data Analysis and Applications Basing on LLM Agents and Knowledge Graph Interactions</strong> - Xi Wang, Xianyao Ling, Kun Li, Gang Yin, Liang Zhang, Jiang Wu, Jun Xu, Fu Zhang, Wenbo Lei, Annie Wang, Peng Gong - [[pdf]](https://arxiv.org/pdf/2510.15258)</summary>

**Abstract:** In the current era of big data, extracting deep insights from massive, heterogeneous, and complexly associated multi-dimensional data has become a significant challenge. Large Language Models (LLMs) perform well in natural language understanding and generation, but still suffer from "hallucination" issues when processing structured knowledge and are difficult to update in real-time. Although Knowledge Graphs (KGs) can explicitly store structured knowledge, their static nature limits dynamic interaction and analytical capabilities. Therefore, this paper proposes a multi-dimensional data analysis method based on the interactions between LLM agents and KGs, constructing a dynamic, collaborative analytical ecosystem. This method utilizes LLM agents to automatically extract product data from unstructured data, constructs and visualizes the KG in real-time, and supports users in deep exploration and analysis of graph nodes through an interactive platform. Experimental results show that this method has significant advantages in product ecosystem analysis, relationship mining, and user-driven exploratory analysis, providing new ideas and tools for multi-dimensional data analysis.

**arXiv ID:** 2510.15258
</details>

<details>
<summary><strong>AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization</strong> - Genghan Zhang, Shaowei Zhu, Anjiang Wei, Zhenyu Song, Allen Nie, Zhen Jia, Nandita Vijaykumar, Yida Wang, Kunle Olukotun - [[pdf]](https://arxiv.org/pdf/2511.15915)</summary>

**Abstract:** We present AccelOpt, a self-improving large language model (LLM) agentic system that autonomously optimizes kernels for emerging AI acclerators, eliminating the need for expert-provided hardware-specific optimization knowledge. AccelOpt explores the kernel optimization space through iterative generation, informed by an optimization memory that curates experiences and insights from previously encountered slow-fast kernel pairs. We build NKIBench, a new benchmark suite of AWS Trainium accelerator kernels with varying complexity extracted from real-world LLM workloads to evaluate the effectiveness of AccelOpt. Our evaluation confirms that AccelOpt's capability improves over time, boosting the average percentage of peak throughput from $49\%$ to $61\%$ on Trainium 1 and from $45\%$ to $59\%$ on Trainium 2 for NKIBench kernels. Moreover, AccelOpt is highly cost-effective: using open-source models, it matches the kernel improvements of Claude Sonnet 4 while being $26\times$ cheaper.

**arXiv ID:** 2511.15915
</details>

<details>
<summary><strong>AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search</strong> - Yu Li, Lehui Li, Zhihao Wu, Qingmin Liao, Jianye Hao, Kun Shao, Fengli Xu, Yong Li - [[pdf]](https://arxiv.org/pdf/2506.06017)</summary>

**Abstract:** Large language model (LLM) agents have demonstrated strong capabilities across diverse domains, yet automated agent design remains a significant challenge. Current automated agent design approaches are often constrained by limited search spaces that primarily optimize workflows but fail to integrate crucial human-designed components like memory, planning, and tool use. Furthermore, these methods are hampered by high evaluation costs, as evaluating even a single new agent on a benchmark can require tens of dollars. The difficulty of this exploration is further exacerbated by inefficient search strategies that struggle to navigate the large design space effectively, making the discovery of novel agents a slow and resource-intensive process. To address these challenges, we propose AgentSwift, a novel framework for automated agent design. We formalize a hierarchical search space that jointly models agentic workflow and composable functional components. This structure moves beyond optimizing workflows alone by co-optimizing functional components, which enables the discovery of more complex and effective agent architectures. To make exploration within this expansive space feasible, we mitigate high evaluation costs by training a value model on a high-quality dataset, generated via a novel strategy combining combinatorial coverage and balanced Bayesian sampling for low-cost evaluation. Guiding the entire process is a hierarchical MCTS strategy, which is informed by uncertainty to efficiently navigate the search space. Evaluated across a comprehensive set of seven benchmarks spanning embodied, math, web, tool, and game domains, AgentSwift discovers agents that achieve an average performance gain of 8.34\% over both existing automated agent search methods and manually designed agents. Our framework serves as a launchpad for researchers to rapidly discover powerful agent architectures.

**arXiv ID:** 2506.06017
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (18 papers)</h2></summary>

<details>
<summary><strong>MACIE: Multi-Agent Causal Intelligence Explainer for Collective Behavior Understanding</strong> - Abraham Itzhak Weinberg - [[pdf]](https://arxiv.org/pdf/2511.15716)</summary>

**Abstract:** As Multi Agent Reinforcement Learning systems are used in safety critical applications. Understanding why agents make decisions and how they achieve collective behavior is crucial. Existing explainable AI methods struggle in multi agent settings. They fail to attribute collective outcomes to individuals, quantify emergent behaviors, or capture complex interactions. We present MACIE Multi Agent Causal Intelligence Explainer, a framework combining structural causal models, interventional counterfactuals, and Shapley values to provide comprehensive explanations. MACIE addresses three questions. First, each agent's causal contribution using interventional attribution scores. Second, system level emergent intelligence through synergy metrics separating collective effects from individual contributions. Third, actionable explanations using natural language narratives synthesizing causal insights. We evaluate MACIE across four MARL scenarios: cooperative, competitive, and mixed motive. Results show accurate outcome attribution, mean phi_i equals 5.07, standard deviation less than 0.05, detection of positive emergence in cooperative tasks, synergy index up to 0.461, and efficient computation, 0.79 seconds per dataset on CPU. MACIE uniquely combines causal rigor, emergence quantification, and multi agent support while remaining practical for real time use. This represents a step toward interpretable, trustworthy, and accountable multi agent AI.

**arXiv ID:** 2511.15716
</details>

<details>
<summary><strong>Build AI Assistants using Large Language Models and Agents to Enhance the Engineering Education of Biomechanics</strong> - Hanzhi Yan, Qin Lu, Xianqiao Wang, Xiaoming Zhai, Tianming Liu, He Li - [[pdf]](https://arxiv.org/pdf/2511.15752)</summary>

**Abstract:** While large language models (LLMs) have demonstrated remarkable versatility across a wide range of general tasks, their effectiveness often diminishes in domain-specific applications due to inherent knowledge gaps. Moreover, their performance typically declines when addressing complex problems that require multi-step reasoning and analysis. In response to these challenges, we propose leveraging both LLMs and AI agents to develop education assistants aimed at enhancing undergraduate learning in biomechanics courses that focus on analyzing the force and moment in the musculoskeletal system of the human body. To achieve our goal, we construct a dual-module framework to enhance LLM performance in biomechanics educational tasks: 1) we apply Retrieval-Augmented Generation (RAG) to improve the specificity and logical consistency of LLM's responses to the conceptual true/false questions; 2) we build a Multi-Agent System (MAS) to solve calculation-oriented problems involving multi-step reasoning and code execution. Specifically, we evaluate the performance of several LLMs, i.e., Qwen-1.0-32B, Qwen-2.5-32B, and Llama-70B, on a biomechanics dataset comprising 100 true/false conceptual questions and problems requiring equation derivation and calculation. Our results demonstrate that RAG significantly enhances the performance and stability of LLMs in answering conceptual questions, surpassing those of vanilla models. On the other hand, the MAS constructed using multiple LLMs demonstrates its ability to perform multi-step reasoning, derive equations, execute code, and generate explainable solutions for tasks that require calculation. These findings demonstrate the potential of applying RAG and MAS to enhance LLM performance for specialized courses in engineering curricula, providing a promising direction for developing intelligent tutoring in engineering education.

**arXiv ID:** 2511.15752
</details>

<details>
<summary><strong>Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response</strong> - Philip Drammeh - [[pdf]](https://arxiv.org/pdf/2511.15755)</summary>

**Abstract:** Large language models (LLMs) promise to accelerate incident response in production systems, yet single-agent approaches generate vague, unusable recommendations. We present this http URL, a reproducible containerized framework demonstrating that multi-agent orchestration fundamentally transforms LLM-based incident response quality. Through 348 controlled trials comparing single-agent copilot versus multi-agent systems on identical incident scenarios, we find that multi-agent orchestration achieves 100% actionable recommendation rate versus 1.7% for single-agent approaches, an 80 times improvement in action specificity and 140 times improvement in solution correctness. Critically, multi-agent systems exhibit zero quality variance across all trials, enabling production SLA commitments impossible with inconsistent single-agent outputs. Both architectures achieve similar comprehension latency (approx.40s), establishing that the architectural value lies in deterministic quality, not speed. We introduce Decision Quality (DQ), a novel metric capturing validity, specificity, and correctness properties essential for operational deployment that existing LLM metrics do not address. These findings reframe multi-agent orchestration from a performance optimization to a production-readiness requirement for LLM-based incident response. All code, Docker configurations, and trial data are publicly available for reproduction.

**arXiv ID:** 2511.15755
</details>

<details>
<summary><strong>IMACT-CXR - An Interactive Multi-Agent Conversational Tutoring System for Chest X-Ray Interpretation</strong> - Tuan-Anh Le, Anh Mai Vu, David Yang, Akash Awasthi, Hien Van Nguyen - [[pdf]](https://arxiv.org/pdf/2511.15825)</summary>

**Abstract:** IMACT-CXR is an interactive multi-agent conversational tutor that helps trainees interpret chest X-rays by unifying spatial annotation, gaze analysis, knowledge retrieval, and image-grounded reasoning in a single AutoGen-based workflow. The tutor simultaneously ingests learner bounding boxes, gaze samples, and free-text observations. Specialized agents evaluate localization quality, generate Socratic coaching, retrieve PubMed evidence, suggest similar cases from REFLACX, and trigger NV-Reason-CXR-3B for vision-language reasoning when mastery remains low or the learner explicitly asks. Bayesian Knowledge Tracing (BKT) maintains skill-specific mastery estimates that drive both knowledge reinforcement and case similarity retrieval. A lung-lobe segmentation module derived from a TensorFlow U-Net enables anatomically aware gaze feedback, and safety prompts prevent premature disclosure of ground-truth labels. We describe the system architecture, implementation highlights, and integration with the REFLACX dataset for real DICOM cases. IMACT-CXR demonstrates responsive tutoring flows with bounded latency, precise control over answer leakage, and extensibility toward live residency deployment. Preliminary evaluation shows improved localization and diagnostic reasoning compared to baselines.

**arXiv ID:** 2511.15825
</details>

<details>
<summary><strong>Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art</strong> - Noah Bissell, Ethan Paley, Joshua Harrison, Juliano Calil, Myungin Lee - [[pdf]](https://arxiv.org/pdf/2511.15997)</summary>

**Abstract:** Sensorium Arc (AI reflects on climate) is a real-time multimodal interactive AI agent system that personifies the ocean as a poetic speaker and guides users through immersive explorations of complex marine data. Built on a modular multi-agent system and retrieval-augmented large language model (LLM) framework, Sensorium enables natural spoken conversations with AI agents that embodies the ocean's perspective, generating responses that blend scientific insight with ecological poetics. Through keyword detection and semantic parsing, the system dynamically triggers data visualizations and audiovisual playback based on time, location, and thematic cues drawn from the dialogue. Developed in collaboration with the Center for the Study of the Force Majeure and inspired by the eco-aesthetic philosophy of Newton Harrison, Sensorium Arc reimagines ocean data not as an abstract dataset but as a living narrative. The project demonstrates the potential of conversational AI agents to mediate affective, intuitive access to high-dimensional environmental data and proposes a new paradigm for human-machine-ecosystem.

**arXiv ID:** 2511.15997
</details>

<details>
<summary><strong>FOOTPASS: A Multi-Modal Multi-Agent Tactical Context Dataset for Play-by-Play Action Spotting in Soccer Broadcast Videos</strong> - Jeremie Ochin, Raphael Chekroun, Bogdan Stanciulescu, Sotiris Manitsaris - [[pdf]](https://arxiv.org/pdf/2511.16183)</summary>

**Abstract:** Soccer video understanding has motivated the creation of datasets for tasks such as temporal action localization, spatiotemporal action detection (STAD), or multiobject tracking (MOT). The annotation of structured sequences of events (who does what, when, and where) used for soccer analytics requires a holistic approach that integrates both STAD and MOT. However, current action recognition methods remain insufficient for constructing reliable play-by-play data and are typically used to assist rather than fully automate annotation. Parallel research has advanced tactical modeling, trajectory forecasting, and performance analysis, all grounded in game-state and play-by-play data. This motivates leveraging tactical knowledge as a prior to support computer-vision-based predictions, enabling more automated and reliable extraction of play-by-play data. We introduce Footovision Play-by-Play Action Spotting in Soccer Dataset (FOOTPASS), the first benchmark for play-by-play action spotting over entire soccer matches in a multi-modal, multi-agent tactical context. It enables the development of methods for player-centric action spotting that exploit both outputs from computer-vision tasks (e.g., tracking, identification) and prior knowledge of soccer, including its tactical regularities over long time horizons, to generate reliable play-by-play data streams. These streams form an essential input for data-driven sports analytics.

**arXiv ID:** 2511.16183
</details>

<details>
<summary><strong>Multi-Agent Collaborative Reward Design for Enhancing Reasoning in Reinforcement Learning</strong> - Pei Yang, Ke Zhang, Ji Wang, Xiao Chen, Yuxin Tang, Eric Yang, Lynn Ai, Bill Shi - [[pdf]](https://arxiv.org/pdf/2511.16202)</summary>

**Abstract:** We present CRM (Multi-Agent Collaborative Reward Model), a framework that replaces a single black-box reward model with a coordinated team of specialist evaluators to improve robustness and interpretability in RLHF. Conventional reward models struggle to jointly optimize multiple, sometimes conflicting, preference dimensions (e.g., factuality, helpfulness, safety) and offer limited transparency into why a score is assigned. CRM addresses these issues by decomposing preference evaluation into domain-specific agents that each produce partial signals, alongside global evaluators such as ranker-based and embedding-similarity rewards. A centralized aggregator fuses these signals at each timestep, balancing factors like step-wise correctness, multi-agent agreement, and repetition penalties, yielding a single training reward compatible with standard RL pipelines. The policy is optimized with advantage-based updates (e.g., GAE), while a value model regresses to the aggregated reward, enabling multi-perspective reward shaping without requiring additional human annotations beyond those used to train the evaluators. To support training and assessment, we introduce rewardBench, a benchmark and training suite aligned with the collaborative structure of CRM. Together, CRM and rewardBench provide a practical, modular path to more transparent reward modeling and more stable optimization.

**arXiv ID:** 2511.16202
</details>

<details>
<summary><strong>ChemLabs on ChemO: A Multi-Agent System for Multimodal Reasoning on IChO 2025</strong> - Xu Qiang, Shengyuan Bai, Leqing Chen, Zijing Liu, Yu Li - [[pdf]](https://arxiv.org/pdf/2511.16205)</summary>

**Abstract:** Olympiad-level benchmarks in mathematics and physics are crucial testbeds for advanced AI reasoning, but chemistry, with its unique multimodal symbolic language, has remained an open challenge. We introduce ChemO, a new benchmark built from the International Chemistry Olympiad (IChO) 2025. ChemO features two key innovations for automated assessment: Assessment-Equivalent Reformulation (AER), which converts problems requiring visual outputs (e.g., drawing molecules) into computationally tractable formats, and Structured Visual Enhancement (SVE), a diagnostic mechanism to disentangle a model's visual perception capabilities from its core chemical reasoning. To tackle this benchmark, we propose ChemLabs, a hierarchical multi-agent framework that mimics human expert collaboration through specialized agents for problem decomposition, perception, reasoning, and auditing. Experiments on state-of-the-art multimodal models demonstrate that combining SVE with our multi-agent system yields dramatic performance gains. Our top configuration achieves a score of 93.6 out of 100, surpassing an estimated human gold medal threshold and establishing a new state-of-the-art in automated chemical problem-solving. ChemO Dataset: this https URL

**arXiv ID:** 2511.16205
</details>

<details>
<summary><strong>Distributed Agent Reasoning Across Independent Systems With Strict Data Locality</strong> - Daniel Vaughan, Kateřina Vaughan - [[pdf]](https://arxiv.org/pdf/2511.16292)</summary>

**Abstract:** This paper presents a proof-of-concept demonstration of agent-to-agent communication across distributed systems, using only natural-language messages and without shared identifiers, structured schemas, or centralised data exchange. The prototype explores how multiple organisations (represented here as a Clinic, Insurer, and Specialist Network) can cooperate securely via pseudonymised case tokens, local data lookups, and controlled operational boundaries.
The system uses Orpius as the underlying platform for multi-agent orchestration, tool execution, and privacy-preserving communication. All agents communicate through OperationRelay calls, exchanging concise natural-language summaries. Each agent operates on its own data (such as synthetic clinic records, insurance enrolment tables, and clinical guidance extracts), and none receives or reconstructs patient identity. The Clinic computes an HMAC-based pseudonymous token, the Insurer evaluates coverage rules and consults the Specialist agent, and the Specialist returns an appropriateness recommendation.
The goal of this prototype is intentionally limited: to demonstrate feasibility, not to provide a clinically validated, production-ready system. No clinician review was conducted, and no evaluation beyond basic functional runs was performed. The work highlights architectural patterns, privacy considerations, and communication flows that enable distributed reasoning among specialised agents while keeping data local to each organisation. We conclude by outlining opportunities for more rigorous evaluation and future research in decentralised multi-agent systems.

**arXiv ID:** 2511.16292
</details>

<details>
<summary><strong>Hiding in the AI Traffic: Abusing MCP for LLM-Powered Agentic Red Teaming</strong> - Strahinja Janjuesvic, Anna Baron Garcia, Sohrob Kazerounian - [[pdf]](https://arxiv.org/pdf/2511.15998)</summary>

**Abstract:** Generative AI is reshaping offensive cybersecurity by enabling autonomous red team agents that can plan, execute, and adapt during penetration tests. However, existing approaches face trade-offs between generality and specialization, and practical deployments reveal challenges such as hallucinations, context limitations, and ethical concerns. In this work, we introduce a novel command & control (C2) architecture leveraging the Model Context Protocol (MCP) to coordinate distributed, adaptive reconnaissance agents covertly across networks. Notably, we find that our architecture not only improves goal-directed behavior of the system as whole, but also eliminates key host and network artifacts that can be used to detect and prevent command & control behavior altogether. We begin with a comprehensive review of state-of-the-art generative red teaming methods, from fine-tuned specialist models to modular or agentic frameworks, analyzing their automation capabilities against task-specific accuracy. We then detail how our MCP-based C2 can overcome current limitations by enabling asynchronous, parallel operations and real-time intelligence sharing without periodic beaconing. We furthermore explore advanced adversarial capabilities of this architecture, its detection-evasion techniques, and address dual-use ethical implications, proposing defensive measures and controlled evaluation in lab settings. Experimental comparisons with traditional C2 show drastic reductions in manual effort and detection footprint. We conclude with future directions for integrating autonomous exploitation, defensive LLM agents, predictive evasive maneuvers, and multi-agent swarms. The proposed MCP-enabled C2 framework demonstrates a significant step toward realistic, AI-driven red team operations that can simulate advanced persistent threats while informing the development of next-generation defensive systems.

**arXiv ID:** 2511.15998
</details>

<details>
<summary><strong>Securing Smart Contract Languages with a Unified Agentic Framework for Vulnerability Repair in Solidity and Move</strong> - Rabimba Karanjai, Lei Xu, Weidong Shi - [[pdf]](https://arxiv.org/pdf/2502.18515)</summary>

**Abstract:** The rapid growth of the blockchain ecosystem and the increasing value locked in smart contracts necessitate robust security measures. While languages like Solidity and Move aim to improve smart contract security, vulnerabilities persist. This paper presents Smartify, a novel multi-agent framework leveraging Large Language Models (LLMs) to automatically detect and repair vulnerabilities in Solidity and Move smart contracts. Unlike traditional methods that rely solely on vast pre-training datasets, Smartify employs a team of specialized agents working on different specially fine-tuned LLMs to analyze code based on underlying programming concepts and language-specific security principles. We evaluated Smartify on a dataset for Solidity and a curated dataset for Move, demonstrating its effectiveness in fixing a wide range of vulnerabilities. Our results show that Smartify (Gemma2+codegemma) achieves state-of-the-art performance, surpassing existing LLMs and enhancing general-purpose models' capabilities, such as Llama 3.1. Notably, Smartify can incorporate language-specific knowledge, such as the nuances of Move, without requiring massive language-specific pre-training datasets. This work offers a detailed analysis of various LLMs' performance on smart contract repair, highlighting the strengths of our multi-agent approach and providing a blueprint for developing more secure and reliable decentralized applications in the growing blockchain landscape. We also provide a detailed recipe for extending this to other similar use cases.

**arXiv ID:** 2502.18515
</details>

<details>
<summary><strong>Policy Search, Retrieval, and Composition via Task Similarity in Collaborative Agentic Systems</strong> - Saptarshi Nath, Christos Peridis, Eseoghene Benjamin, Xinran Liu, Soheil Kolouri, Peter Kinnell, Zexin Li, Cong Liu, Shirin Dora, Andrea Soltoggio - [[pdf]](https://arxiv.org/pdf/2506.05577)</summary>

**Abstract:** Agentic AI aims to create systems that set their own goals, adapt proactively to change, and refine behavior through continuous experience. Recent advances suggest that, when facing multiple and unforeseen tasks, agents could benefit from sharing machine-learned knowledge and reusing policies that have already been fully or partially learned by other agents. However, how to query, select, and retrieve policies from a pool of agents, and how to integrate such policies remains a largely unexplored area. This study explores how an agent decides what knowledge to select, from whom, and when and how to integrate it in its own policy in order to accelerate its own learning. The proposed algorithm, \emph{Modular Sharing and Composition in Collective Learning} (MOSAIC), improves learning in agentic collectives by combining (1) knowledge selection using performance signals and cosine similarity on Wasserstein task embeddings, (2) modular and transferable neural representations via masks, and (3) policy integration, composition and fine-tuning. MOSAIC outperforms isolated learners and global sharing approaches in both learning speed and overall performance, and in some cases solves tasks that isolated agents cannot. The results also demonstrate that selective, goal-driven reuse leads to less susceptibility to task interference. We also observe the emergence of self-organization, where agents solving simpler tasks accelerate the learning of harder ones through shared knowledge.

**arXiv ID:** 2506.05577
</details>

<details>
<summary><strong>From Static to Adaptive Defense: Federated Multi-Agent Deep Reinforcement Learning-Driven Moving Target Defense Against DoS Attacks in UAV Swarm Networks</strong> - Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Tian Qin, Yuyu Zhao - [[pdf]](https://arxiv.org/pdf/2506.07392)</summary>

**Abstract:** The proliferation of UAVs has enabled a wide range of mission-critical applications and is becoming a cornerstone of low-altitude networks, supporting smart cities, emergency response, and more. However, the open wireless environment, dynamic topology, and resource constraints of UAVs expose low-altitude networks to severe DoS threats. Traditional defense approaches, which rely on fixed configurations or centralized decision-making, cannot effectively respond to the rapidly changing conditions in UAV swarm environments. To address these challenges, we propose a novel federated multi-agent deep reinforcement learning (FMADRL)-driven moving target defense (MTD) framework for proactive DoS mitigation in low-altitude networks. Specifically, we design lightweight and coordinated MTD mechanisms, including leader switching, route mutation, and frequency hopping, to disrupt attacker efforts and enhance network resilience. The defense problem is formulated as a multi-agent partially observable Markov decision process, capturing the uncertain nature of UAV swarms under attack. Each UAV is equipped with a policy agent that autonomously selects MTD actions based on partial observations and local experiences. By employing a policy gradient-based algorithm, UAVs collaboratively optimize their policies via reward-weighted aggregation. Extensive simulations demonstrate that our approach significantly outperforms state-of-the-art baselines, achieving up to a 34.6% improvement in attack mitigation rate, a reduction in average recovery time of up to 94.6%, and decreases in energy consumption and defense cost by as much as 29.3% and 98.3%, respectively, under various DoS attack strategies. These results highlight the potential of intelligent, distributed defense mechanisms to protect low-altitude networks, paving the way for reliable and scalable low-altitude economy.

**arXiv ID:** 2506.07392
</details>

<details>
<summary><strong>Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</strong> - Chengxuan Xia, Qianye Wu, Sixuan Tian, Yilun Hao - [[pdf]](https://arxiv.org/pdf/2507.17061)</summary>

**Abstract:** Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.

**arXiv ID:** 2507.17061
</details>

<details>
<summary><strong>OEMA: Ontology-Enhanced Multi-Agent Collaboration Framework for Zero-Shot Clinical Named Entity Recognition</strong> - Xinli Tao, Xin Dong, Xuezhong Zhou - [[pdf]](https://arxiv.org/pdf/2511.15211)</summary>

**Abstract:** With the rapid expansion of unstructured clinical texts in electronic health records (EHRs), clinical named entity recognition (NER) has become a crucial technique for extracting medical information. However, traditional supervised models such as CRF and BioClinicalBERT suffer from high annotation costs. Although zero-shot NER based on large language models (LLMs) reduces the dependency on labeled data, challenges remain in aligning example selection with task granularity and effectively integrating prompt design with self-improvement frameworks. To address these limitations, we propose OEMA, a novel zero-shot clinical NER framework based on multi-agent collaboration. OEMA consists of three core components: (1) a self-annotator that autonomously generates candidate examples; (2) a discriminator that leverages SNOMED CT to filter token-level examples by clinical relevance; and (3) a predictor that incorporates entity-type descriptions to enhance inference accuracy. Experimental results on two benchmark datasets, MTSamples and VAERS, demonstrate that OEMA achieves state-of-the-art performance under exact-match evaluation. Moreover, under related-match criteria, OEMA performs comparably to the supervised BioClinicalBERT model while significantly outperforming the traditional CRF method. OEMA improves zero-shot clinical NER, achieving near-supervised performance under related-match criteria. Future work will focus on continual learning and open-domain adaptation to expand its applicability in clinical NLP.

**arXiv ID:** 2511.15211
</details>

<details>
<summary><strong>The Subtle Art of Defection: Understanding Uncooperative Behaviors in LLM based Multi-Agent Systems</strong> - Devang Kulshreshtha, Wanyu Du, Raghav Jain, Srikanth Doss, Hang Su, Sandesh Swamy, Yanjun Qi - [[pdf]](https://arxiv.org/pdf/2511.15862)</summary>

**Abstract:** This paper introduces a novel framework for simulating and analyzing how uncooperative behaviors can destabilize or collapse LLM-based multi-agent systems. Our framework includes two key components: (1) a game theory-based taxonomy of uncooperative agent behaviors, addressing a notable gap in the existing literature; and (2) a structured, multi-stage simulation pipeline that dynamically generates and refines uncooperative behaviors as agents' states evolve. We evaluate the framework via a collaborative resource management setting, measuring system stability using metrics such as survival time and resource overuse rate. Empirically, our framework achieves 96.7% accuracy in generating realistic uncooperative behaviors, validated by human evaluations. Our results reveal a striking contrast: cooperative agents maintain perfect system stability (100% survival over 12 rounds with 0% resource overuse), while any uncooperative behavior can trigger rapid system collapse within 1 to 7 rounds. These findings demonstrate that uncooperative agents can significantly degrade collective outcomes, highlighting the need for designing more resilient multi-agent systems.

**arXiv ID:** 2511.15862
</details>

<details>
<summary><strong>SurvAgent: Hierarchical CoT-Enhanced Case Banking and Dichotomy-Based Multi-Agent System for Multimodal Survival Prediction</strong> - Guolin Huang, Wenting Chen, Jiaqi Yang, Xinheng Lyu, Xiaoling Luo, Sen Yang, Xiaohan Xing, Linlin Shen - [[pdf]](https://arxiv.org/pdf/2511.16635)</summary>

**Abstract:** Survival analysis is critical for cancer prognosis and treatment planning, yet existing methods lack the transparency essential for clinical adoption. While recent pathology agents have demonstrated explainability in diagnostic tasks, they face three limitations for survival prediction: inability to integrate multimodal data, ineffective region-of-interest exploration, and failure to leverage experiential learning from historical cases. We introduce SurvAgent, the first hierarchical chain-of-thought (CoT)-enhanced multi-agent system for multimodal survival prediction. SurvAgent consists of two stages: (1) WSI-Gene CoT-Enhanced Case Bank Construction employs hierarchical analysis through Low-Magnification Screening, Cross-Modal Similarity-Aware Patch Mining, and Confidence-Aware Patch Mining for pathology images, while Gene-Stratified analysis processes six functional gene categories. Both generate structured reports with CoT reasoning, storing complete analytical processes for experiential learning. (2) Dichotomy-Based Multi-Expert Agent Inference retrieves similar cases via RAG and integrates multimodal reports with expert predictions through progressive interval refinement. Extensive experiments on five TCGA cohorts demonstrate SurvAgent's superority over conventional methods, proprietary MLLMs, and medical agents, establishing a new paradigm for explainable AI-driven survival prediction in precision oncology.

**arXiv ID:** 2511.16635
</details>

<details>
<summary><strong>Risk Map As Middleware: Towards Interpretable Cooperative End-to-end Autonomous Driving for Risk-Aware Planning</strong> - Mingyue Lei, Zewei Zhou, Hongchen Li, Jiaqi Ma, Jia Hu - [[pdf]](https://arxiv.org/pdf/2508.07686)</summary>

**Abstract:** End-to-end paradigm has emerged as a promising approach to autonomous driving. However, existing single-agent end-to-end pipelines are often constrained by occlusion and limited perception range, resulting in hazardous driving. Furthermore, their black-box nature prevents the interpretability of the driving behavior, leading to an untrustworthiness system. To address these limitations, we introduce Risk Map as Middleware (RiskMM) and propose an interpretable cooperative end-to-end driving framework. The risk map learns directly from the driving data and provides an interpretable spatiotemporal representation of the scenario from the upstream perception and the interactions between the ego vehicle and the surrounding environment for downstream planning. RiskMM first constructs a multi-agent spatiotemporal representation with unified Transformer-based architecture, then derives risk-aware representations by modeling interactions among surrounding environments with attention. These representations are subsequently fed into a learning-based Model Predictive Control (MPC) module. The MPC planner inherently accommodates physical constraints and different vehicle types and can provide interpretation by aligning learned parameters with explicit MPC elements. Evaluations conducted on the real-world V2XPnP-Seq dataset confirm that RiskMM achieves superior and robust performance in risk-aware trajectory planning, significantly enhancing the interpretability of the cooperative end-to-end driving framework. The codebase will be released to facilitate future research in this field.

**arXiv ID:** 2508.07686
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>CorrectHDL: Agentic HDL Design with LLMs Leveraging High-Level Synthesis as Reference</strong> - Kangwei Xu, Grace Li Zhang, Ulf Schlichtmann, Bing Li - [[pdf]](https://arxiv.org/pdf/2511.16395)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable potential in hardware front-end design using hardware description languages (HDLs). However, their inherent tendency toward hallucination often introduces functional errors into the generated HDL designs. To address this issue, we propose the framework CorrectHDL that leverages high-level synthesis (HLS) results as functional references to correct potential errors in LLM-generated HDL this http URL input to the proposed framework is a C/C++ program that specifies the target circuit's functionality. The program is provided to an LLM to directly generate an HDL design, whose syntax errors are repaired using a Retrieval-Augmented Generation (RAG) mechanism. The functional correctness of the LLM-generated circuit is iteratively improved by comparing its simulated behavior with an HLS reference design produced by conventional HLS tools, which ensures the functional correctness of the result but can lead to suboptimal area and power efficiency. Experimental results demonstrate that circuits generated by the proposed framework achieve significantly better area and power efficiency than conventional HLS designs and approach the quality of human-engineered circuits. Meanwhile, the correctness of the resulting HDL implementation is maintained, highlighting the effectiveness and potential of agentic HDL design leveraging the generative capabilities of LLMs and the rigor of traditional correctness-driven IC design flows.

**arXiv ID:** 2511.16395
</details>

<details>
<summary><strong>Semantic Glitch: Agency and Artistry in an Autonomous Pixel Cloud</strong> - Qing Zhang, Jing Huang, Mingyang Xu, Jun Rekimoto - [[pdf]](https://arxiv.org/pdf/2511.16048)</summary>

**Abstract:** While mainstream robotics pursues metric precision and flawless performance, this paper explores the creative potential of a deliberately "lo-fi" approach. We present the "Semantic Glitch," a soft flying robotic art installation whose physical form, a 3D pixel style cloud, is a "physical glitch" derived from digital archaeology. We detail a novel autonomous pipeline that rejects conventional sensors like LiDAR and SLAM, relying solely on the qualitative, semantic understanding of a Multimodal Large Language Model to navigate. By authoring a bio-inspired personality for the robot through a natural language prompt, we create a "narrative mind" that complements the "weak," historically, loaded body. Our analysis begins with a 13-minute autonomous flight log, and a follow-up study statistically validates the framework's robustness for authoring quantifiably distinct personas. The combined analysis reveals emergent behaviors, from landmark-based navigation to a compelling "plan to execution" gap, and a character whose unpredictable, plausible behavior stems from a lack of precise proprioception. This demonstrates a lo-fi framework for creating imperfect companions whose success is measured in character over efficiency.

**arXiv ID:** 2511.16048
</details>

<details>
<summary><strong>Taming Uncertainty via Automation: Observing, Analyzing, and Optimizing Agentic AI Systems</strong> - Dany Moshkovich, Sergey Zeltyn - [[pdf]](https://arxiv.org/pdf/2507.11277)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly deployed within agentic systems - collections of interacting, LLM-powered agents that execute complex, adaptive workflows using memory, tools, and dynamic planning. While enabling powerful new capabilities, these systems also introduce unique forms of uncertainty stemming from probabilistic reasoning, evolving memory states, and fluid execution paths. Traditional software observability and operations practices fall short in addressing these challenges.
This paper presents our vision of AgentOps: a comprehensive framework for observing, analyzing, optimizing, and automating operation of agentic AI systems. We identify distinct needs across four key roles - developers, testers, site reliability engineers (SREs), and business users - each of whom engages with the system at different points in its lifecycle. We present the AgentOps Automation Pipeline, a six-stage process encompassing behavior observation, metric collection, issue detection, root cause analysis, optimized recommendations, and runtime automation. Throughout, we emphasize the critical role of automation in managing uncertainty and enabling self-improving AI systems - not by eliminating uncertainty, but by taming it to ensure safe, adaptive, and effective operation.

**arXiv ID:** 2507.11277
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (17 papers)</h2></summary>

<details>
<summary><strong>Detecting Sleeper Agents in Large Language Models via Semantic Drift Analysis</strong> - Shahin Zanbaghi, Ryan Rostampour, Farhan Abid, Salim Al Jarmakani - [[pdf]](https://arxiv.org/pdf/2511.15992)</summary>

**Abstract:** Large Language Models (LLMs) can be backdoored to exhibit malicious behavior under specific deployment conditions while appearing safe during training a phenomenon known as "sleeper agents." Recent work by Hubinger et al. demonstrated that these backdoors persist through safety training, yet no practical detection methods exist. We present a novel dual-method detection system combining semantic drift analysis with canary baseline comparison to identify backdoored LLMs in real-time. Our approach uses Sentence-BERT embeddings to measure semantic deviation from safe baselines, complemented by injected canary questions that monitor response consistency. Evaluated on the official Cadenza-Labs dolphin-llama3-8B sleeper agent model, our system achieves 92.5% accuracy with 100% precision (zero false positives) and 85% recall. The combined detection method operates in real-time (<1s per query), requires no model modification, and provides the first practical solution to LLM backdoor detection. Our work addresses a critical security gap in AI deployment and demonstrates that embedding-based detection can effectively identify deceptive model behavior without sacrificing deployment efficiency.

**arXiv ID:** 2511.15992
</details>

<details>
<summary><strong>Trustworthy AI in the Agentic Lakehouse: from Concurrency to Governance</strong> - Jacopo Tagliabue, Federico Bianchi, Ciro Greco - [[pdf]](https://arxiv.org/pdf/2511.16402)</summary>

**Abstract:** Even as AI capabilities improve, most enterprises do not consider agents trustworthy enough to work on production data. In this paper, we argue that the path to trustworthy agentic workflows begins with solving the infrastructure problem first: traditional lakehouses are not suited for agent access patterns, but if we design one around transactions, governance follows. In particular, we draw an operational analogy to MVCC in databases and show why a direct transplant fails in a decoupled, multi-language setting. We then propose an agent-first design, Bauplan, that reimplements data and compute isolation in the lakehouse. We conclude by sharing a reference implementation of a self-healing pipeline in Bauplan, which seamlessly couples agent reasoning with all the desired guarantees for correctness and trust.

**arXiv ID:** 2511.16402
</details>

<details>
<summary><strong>D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies</strong> - Sen Chen, Tong Zhao, Yi Bin, Fei Ma, Wenqi Shao, Zheng Wang - [[pdf]](https://arxiv.org/pdf/2511.16590)</summary>

**Abstract:** Developing intelligent agents capable of operating a wide range of Graphical User Interfaces (GUIs) with human-level proficiency is a key milestone on the path toward Artificial General Intelligence. While most existing datasets and benchmarks for training and evaluating GUI agents are static and idealized, failing to reflect the complexity and unpredictability of real-world environments, particularly the presence of anomalies. To bridge this research gap, we propose D-GARA, a dynamic benchmarking framework, to evaluate Android GUI agent robustness in real-world anomalies. D-GARA introduces a diverse set of real-world anomalies that GUI agents commonly face in practice, including interruptions such as permission dialogs, battery warnings, and update prompts. Based on D-GARA framework, we construct and annotate a benchmark featuring commonly used Android applications with embedded anomalies to support broader community research. Comprehensive experiments and results demonstrate substantial performance degradation in state-of-the-art GUI agents when exposed to anomaly-rich environments, highlighting the need for robustness-aware learning. D-GARA is modular and extensible, supporting the seamless integration of new tasks, anomaly types, and interaction scenarios to meet specific evaluation goals.

**arXiv ID:** 2511.16590
</details>

<details>
<summary><strong>Bridging VLMs and Embodied Intelligence with Deliberate Practice Policy Optimization</strong> - Yi Zhang, Che Liu, Xiancong Ren, Hanchu Ni, Yingji Zhang, Shuai Zhang, Zeyuan Ding, Jiayu Hu, Haozhe Shan, Junbo Qi, Yan Bai, Dengjie Li, Jiachen Luo, Yidong Wang, Yong Dai, Zenglin Xu, Bin Shen, Qifan Wang, Jian Tang, Xiaozhu Ju - [[pdf]](https://arxiv.org/pdf/2511.16602)</summary>

**Abstract:** Developing a universal and versatile embodied intelligence system presents two primary challenges: the critical embodied data bottleneck, where real-world data is scarce and expensive, and the algorithmic inefficiency of existing methods, which are resource-prohibitive. To address these limitations, we introduce Deliberate Practice Policy Optimization (DPPO), a metacognitive ``Metaloop'' training framework that dynamically alternates between supervised fine-tuning (competence expansion) and reinforcement learning (skill refinement). This enables automatic weakness identification and targeted resource allocation, specifically designed to maximize learning efficiency from sparse, finite data. Theoretically, DPPO can be formalised as a unified preference-learning framework. Empirically, training a vision-language embodied model with DPPO, referred to as Pelican-VL 1.0, yields a 20.3% performance improvement over the base model and surpasses open-source models at the 100B-parameter scale by 10.6%. We are open-sourcing both the models and code, providing the first systematic framework that alleviates the data and resource bottleneck and enables the community to build versatile embodied agents efficiently.

**arXiv ID:** 2511.16602
</details>

<details>
<summary><strong>Securing AI Agents Against Prompt Injection Attacks</strong> - Badrinath Ramakrishnan, Akshaya Balaji - [[pdf]](https://arxiv.org/pdf/2511.15759)</summary>

**Abstract:** Retrieval-augmented generation (RAG) systems have become widely used for enhancing large language model capabilities, but they introduce significant security vulnerabilities through prompt injection attacks. We present a comprehensive benchmark for evaluating prompt injection risks in RAG-enabled AI agents and propose a multi-layered defense framework. Our benchmark includes 847 adversarial test cases across five attack categories: direct injection, context manipulation, instruction override, data exfiltration, and cross-context contamination. We evaluate three defense mechanisms: content filtering with embedding-based anomaly detection, hierarchical system prompt guardrails, and multi-stage response verification, across seven state-of-the-art language models. Our combined framework reduces successful attack rates from 73.2% to 8.7% while maintaining 94.3% of baseline task performance. We release our benchmark dataset and defense implementation to support future research in AI agent security.

**arXiv ID:** 2511.15759
</details>

<details>
<summary><strong>A Mathematical Framework for Custom Reward Functions in Job Application Evaluation using Reinforcement Learning</strong> - Shreyansh Jain, Madhav Singhvi, Shreya Rahul Jain, Pranav S, Dishaa Lokesh, Naren Chittibabu, Akash Anandhan - [[pdf]](https://arxiv.org/pdf/2511.16073)</summary>

**Abstract:** Conventional Applicant Tracking Systems (ATS) tend to be inflexible keyword-matchers, and deny gifted candidates a role due to a few minor semantic mismatches. This article describes a new two-step process to design a more refined resume evaluation model based on a small language model (<600M parameters) that is finetuned using GRPO on a custom reward function. To begin with, Supervised Fine-Tuning (SFT) was used to build a solid baseline model. Second, this SFT model was also optimized with the help of Reinforcement Learning (RL) through GRPO under the guidance of a new, multi-component reward function that can holistically assess candidates beyond simple keyword matching. We indicate that the RL application presents a critical problem of reward hacking due to the initial experiments of aggressive penalties, which produces faulty, excessively negative model behaviors. We have overcome this challenge by refining the reward function repeatedly and training hyperparameters into a stable "gentle polishing process" of the reward function. Our resulting GRPO-polished model demonstrates significant real-world efficacy, achieving a final accuracy of 91% on unseen test data. The model shows a strong ability to correctly identify qualified candidates (recall of 0.85 for the 'SELECTED' class) while also showing exceptional precision (1.0), confirming its reliability. These results indicate that a properly executed, two-step fine-tuning procedure can indeed effectively refine a small language model to be able to conduct fine-tuned and human-like candidate scoring, overcoming the drawbacks of both traditional ATS and naive RL usage.

**arXiv ID:** 2511.16073
</details>

<details>
<summary><strong>Large Language Model-Based Reward Design for Deep Reinforcement Learning-Driven Autonomous Cyber Defense</strong> - Sayak Mukherjee, Samrat Chatterjee, Emilie Purvine, Ted Fujimoto, Tegan Emerson - [[pdf]](https://arxiv.org/pdf/2511.16483)</summary>

**Abstract:** Designing rewards for autonomous cyber attack and defense learning agents in a complex, dynamic environment is a challenging task for subject matter experts. We propose a large language model (LLM)-based reward design approach to generate autonomous cyber defense policies in a deep reinforcement learning (DRL)-driven experimental simulation environment. Multiple attack and defense agent personas were crafted, reflecting heterogeneity in agent actions, to generate LLM-guided reward designs where the LLM was first provided with contextual cyber simulation environment information. These reward structures were then utilized within a DRL-driven attack-defense simulation environment to learn an ensemble of cyber defense policies. Our results suggest that LLM-guided reward designs can lead to effective defense strategies against diverse adversarial behaviors.

**arXiv ID:** 2511.16483
</details>

<details>
<summary><strong>RAPID: Robust and Agile Planner Using Inverse Reinforcement Learning for Vision-Based Drone Navigation</strong> - Minwoo Kim, Geunsik Bae, Jinwoo Lee, Woojae Shin, Changseung Kim, Myong-Yol Choi, Heejung Shin, Hyondong Oh - [[pdf]](https://arxiv.org/pdf/2502.02054)</summary>

**Abstract:** This paper introduces a learning-based visual planner for agile drone flight in cluttered environments. The proposed planner generates collision-free waypoints in milliseconds, enabling drones to perform agile maneuvers in complex environments without building separate perception, mapping, and planning modules. Learning-based methods, such as behavior cloning (BC) and reinforcement learning (RL), demonstrate promising performance in visual navigation but still face inherent limitations. BC is susceptible to compounding errors due to limited expert imitation, while RL struggles with reward function design and sample inefficiency. To address these limitations, this paper proposes an inverse reinforcement learning (IRL)-based framework for high-speed visual navigation. By leveraging IRL, it is possible to reduce the number of interactions with simulation environments and improve capability to deal with high-dimensional spaces while preserving the robustness of RL policies. A motion primitive-based path planning algorithm collects an expert dataset with privileged map data from diverse environments, ensuring comprehensive scenario coverage. By leveraging both the acquired expert and learner dataset gathered from the agent's interactions with the simulation environments, a robust reward function and policy are learned across diverse states. While the proposed method is trained in a simulation environment only, it can be directly applied to real-world scenarios without additional training or tuning. The performance of the proposed method is validated in both simulation and real-world environments, including forests and various structures. The trained policy achieves an average speed of 7 m/s and a maximum speed of 8.8 m/s in real flight experiments. To the best of our knowledge, this is the first work to successfully apply an IRL framework for high-speed visual navigation of drones.

**arXiv ID:** 2502.02054
</details>

<details>
<summary><strong>PepThink-R1: LLM for Interpretable Cyclic Peptide Optimization with CoT SFT and Reinforcement Learning</strong> - Ruheng Wang, Hang Zhang, Trieu Nguyen, Shasha Feng, Hao-Wei Pang, Xiang Yu, Li Xiao, Peter Zhiping Zhang - [[pdf]](https://arxiv.org/pdf/2508.14765)</summary>

**Abstract:** Designing therapeutic peptides with tailored properties is hindered by the vastness of sequence space, limited experimental data, and poor interpretability of current generative models. To address these challenges, we introduce PepThink-R1, a generative framework that integrates large language models (LLMs) with chain-of-thought (CoT) supervised fine-tuning and reinforcement learning (RL). Unlike prior approaches, PepThink-R1 explicitly reasons about monomer-level modifications during sequence generation, enabling interpretable design choices while optimizing for multiple pharmacological properties. Guided by a tailored reward function balancing chemical validity and property improvements, the model autonomously explores diverse sequence variants. We demonstrate that PepThink-R1 generates cyclic peptides with significantly enhanced lipophilicity, stability, and exposure, outperforming existing general LLMs (e.g., GPT-5) and domain-specific baseline in both optimization success and interpretability. To our knowledge, this is the first LLM-based peptide design framework that combines explicit reasoning with RL-driven property control, marking a step toward reliable and transparent peptide optimization for therapeutic discovery.

**arXiv ID:** 2508.14765
</details>

<details>
<summary><strong>Co-Reinforcement Learning for Unified Multimodal Understanding and Generation</strong> - Jingjing Jiang, Chongjie Si, Jun Luo, Hanwang Zhang, Chao Ma - [[pdf]](https://arxiv.org/pdf/2505.17534)</summary>

**Abstract:** This paper presents a pioneering exploration of reinforcement learning (RL) via group relative policy optimization for unified multimodal large language models (ULMs), aimed at simultaneously reinforcing generation and understanding capabilities. Through systematic pilot studies, we uncover the significant potential of ULMs to enable the synergistic co-evolution of dual capabilities within a shared policy optimization framework. Building on this insight, we introduce CoRL, a co-reinforcement learning framework comprising a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement. With the proposed CoRL, our resulting model, ULM-R1, achieves average improvements of 7% on three text-to-image generation datasets and 23% on nine multimodal understanding benchmarks. These results demonstrate the effectiveness of CoRL and highlight the substantial benefit of reinforcement learning in facilitating cross-task synergy and optimization for ULMs. Code is available at this https URL.

**arXiv ID:** 2505.17534
</details>

<details>
<summary><strong>Machine Learning Epidemic Predictions Using Agent-based Wireless Sensor Network Models</strong> - Chukwunonso Henry Nwokoye, Blessing Oluchi, Sharna Waldron, Peace Ezzeh - [[pdf]](https://arxiv.org/pdf/2511.15982)</summary>

**Abstract:** The lack of epidemiological data in wireless sensor networks (WSNs) is a fundamental difficulty in constructing robust models to forecast and mitigate threats such as viruses and worms. Many studies have examined different epidemic models for WSNs, focusing on how malware infections spread given the network's specific properties, including energy limits and node mobility. In this study, an agent-based implementation of the susceptible-exposed-infected-recovered-vaccinated (SEIRV) mathematical model was employed for machine learning (ML) predictions. Using tools such as NetLogo's BehaviorSpace and Python, two epidemic synthetic datasets were generated and prepared for the application of several ML algorithms. Posed as a regression problem, the infected and recovered nodes were predicted, and the performance of these algorithms is compared using the error metrics of the train and test sets. The predictions performed well, with low error metrics and high R^2 values (0.997, 1.000, 0.999, 1.000), indicating an effective fit to the training set. The validation values were lower (0.992, 0.998, 0.971, and 0.999), as is typical when evaluating model performance on unseen data. Based on the recorded performances, support vector, linear, Lasso, Ridge, and ElasticNet regression were among the worst-performing algorithms, while Random Forest, XGBoost, Decision Trees, and k-nearest neighbors achieved the best results.

**arXiv ID:** 2511.15982
</details>

<details>
<summary><strong>Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning</strong> - Peng Xia, Kaide Zeng, Jiaqi Liu, Can Qin, Fang Wu, Yiyang Zhou, Caiming Xiong, Huaxiu Yao - [[pdf]](https://arxiv.org/pdf/2511.16043)</summary>

**Abstract:** Large Language Model (LLM) Agents, often trained with Reinforcement Learning (RL), are constrained by a dependency on human-curated data, limiting scalability and tethering AI to human knowledge. Existing self-evolution frameworks offer an alternative but are typically restricted by the model's inherent capabilities and single-round interactions, hindering the development of complex curricula involving tool use or dynamic reasoning. We introduce Agent0, a fully autonomous framework that evolves high-performing agents without external data through multi-step co-evolution and seamless tool integration. Agent0 establishes a symbiotic competition between two agents initialized from the same base LLM: a curriculum agent that proposes increasingly challenging frontier tasks, and an executor agent that learns to solve them. We integrate external tools to enhance the executor's problem-solving capacity; this improvement, in turn, pressures the curriculum agent to construct more complex, tool-aware tasks. Through this iterative process, Agent0 establishes a self-reinforcing cycle that continuously produces high-quality curricula. Empirically, Agent0 substantially boosts reasoning capabilities, improving the Qwen3-8B-Base model by 18% on mathematical reasoning and 24% on general reasoning benchmarks. Code is available at this https URL.

**arXiv ID:** 2511.16043
</details>

<details>
<summary><strong>Optimizing Operation Recipes with Reinforcement Learning for Safe and Interpretable Control of Chemical Processes</strong> - Dean Brandner, Sergio Lucia - [[pdf]](https://arxiv.org/pdf/2511.16297)</summary>

**Abstract:** Optimal operation of chemical processes is vital for energy, resource, and cost savings in chemical engineering. The problem of optimal operation can be tackled with reinforcement learning, but traditional reinforcement learning methods face challenges due to hard constraints related to quality and safety that must be strictly satisfied, and the large amount of required training data. Chemical processes often cannot provide sufficient experimental data, and while detailed dynamic models can be an alternative, their complexity makes it computationally intractable to generate the needed data. Optimal control methods, such as model predictive control, also struggle with the complexity of the underlying dynamic models. Consequently, many chemical processes rely on manually defined operation recipes combined with simple linear controllers, leading to suboptimal performance and limited flexibility.
In this work, we propose a novel approach that leverages expert knowledge embedded in operation recipes. By using reinforcement learning to optimize the parameters of these recipes and their underlying linear controllers, we achieve an optimized operation recipe. This method requires significantly less data, handles constraints more effectively, and is more interpretable than traditional reinforcement learning methods due to the structured nature of the recipes. We demonstrate the potential of our approach through simulation results of an industrial batch polymerization reactor, showing that it can approach the performance of optimal controllers while addressing the limitations of existing methods.

**arXiv ID:** 2511.16297
</details>

<details>
<summary><strong>A Comparison Between Decision Transformers and Traditional Offline Reinforcement Learning Algorithms</strong> - Ali Murtaza Caunhye, Asad Jeewa - [[pdf]](https://arxiv.org/pdf/2511.16475)</summary>

**Abstract:** The field of Offline Reinforcement Learning (RL) aims to derive effective policies from pre-collected datasets without active environment interaction. While traditional offline RL algorithms like Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL) have shown promise, they often face challenges in balancing exploration and exploitation, especially in environments with varying reward densities. The recently proposed Decision Transformer (DT) approach, which reframes offline RL as a sequence modelling problem, has demonstrated impressive results across various benchmarks. This paper presents a comparative study evaluating the performance of DT against traditional offline RL algorithms in dense and sparse reward settings for the ANT continous control environment. Our research investigates how these algorithms perform when faced with different reward structures, examining their ability to learn effective policies and generalize across varying levels of feedback. Through empirical analysis in the ANT environment, we found that DTs showed less sensitivity to varying reward density compared to other methods and particularly excelled with medium-expert datasets in sparse reward scenarios. In contrast, traditional value-based methods like IQL showed improved performance in dense reward settings with high-quality data, while CQL offered balanced performance across different data qualities. Additionally, DTs exhibited lower variance in performance but required significantly more computational resources compared to traditional approaches. These findings suggest that sequence modelling approaches may be more suitable for scenarios with uncertain reward structures or mixed-quality data, while value-based methods remain competitive in settings with dense rewards and high-quality demonstrations.

**arXiv ID:** 2511.16475
</details>

<details>
<summary><strong>MagBotSim: Physics-Based Simulation and Reinforcement Learning Environments for Magnetic Robotics</strong> - Lara Bergmann, Cedric Grothues, Klaus Neumann - [[pdf]](https://arxiv.org/pdf/2511.16158)</summary>

**Abstract:** Magnetic levitation is about to revolutionize in-machine material flow in industrial automation. Such systems are flexibly configurable and can include a large number of independently actuated shuttles (movers) that dynamically rebalance production capacity. Beyond their capabilities for dynamic transportation, these systems possess the inherent yet unexploited potential to perform manipulation. By merging the fields of transportation and manipulation into a coordinated swarm of magnetic robots (MagBots), we enable manufacturing systems to achieve significantly higher efficiency, adaptability, and compactness. To support the development of intelligent algorithms for magnetic levitation systems, we introduce MagBotSim (Magnetic Robotics Simulation): a physics-based simulation for magnetic levitation systems. By framing magnetic levitation systems as robot swarms and providing a dedicated simulation, this work lays the foundation for next generation manufacturing systems powered by Magnetic Robotics. MagBotSim's documentation, videos, experiments, and code are available at: this https URL

**arXiv ID:** 2511.16158
</details>

<details>
<summary><strong>Leveraging Reinforcement Learning, Genetic Algorithms and Transformers for background determination in particle physics</strong> - Guillermo Hijano Mendizabal, Davide Lancierini, Alex Marshall, Andrea Mauri, Patrick Haworth Owen, Mitesh Patel, Konstantinos Petridis, Shah Rukh Qasim, Nicola Serra, William Sutcliffe, Hanae Tilquin - [[pdf]](https://arxiv.org/pdf/2509.14894)</summary>

**Abstract:** Experimental studies of beauty hadron decays face significant challenges due to a wide range of backgrounds arising from the numerous possible decay channels with similar final states. For a particular signal decay, the process for ascertaining the most relevant background processes necessitates a detailed analysis of final state particles, potential misidentifications, and kinematic overlaps, which, due to computational limitations, is restricted to the simulation of only the most relevant backgrounds. Moreover, this process typically relies on the physicist's intuition and expertise, as no systematic method exists.
This paper has two primary goals. First, from a particle physics perspective, we present a novel approach that utilises Reinforcement Learning (RL) to overcome the aforementioned challenges by systematically determining the critical backgrounds affecting beauty hadron decay measurements. While beauty hadron physics serves as the case study in this work, the proposed strategy is broadly adaptable to other types of particle physics measurements. Second, from a Machine Learning perspective, we introduce a novel algorithm which exploits the synergy between RL and Genetic Algorithms (GAs) for environments with highly sparse rewards and a large trajectory space. This strategy leverages GAs to efficiently explore the trajectory space and identify successful trajectories, which are used to guide the RL agent's training. Our method also incorporates a transformer architecture for the RL agent to handle token sequences representing decays.

**arXiv ID:** 2509.14894
</details>

<details>
<summary><strong>Safe and Optimal Variable Impedance Control via Certified Reinforcement Learning</strong> - Shreyas Kumar, Ravi Prakash - [[pdf]](https://arxiv.org/pdf/2511.16330)</summary>

**Abstract:** Reinforcement learning (RL) offers a powerful approach for robots to learn complex, collaborative skills by combining Dynamic Movement Primitives (DMPs) for motion and Variable Impedance Control (VIC) for compliant interaction. However, this model-free paradigm often risks instability and unsafe exploration due to the time-varying nature of impedance gains. This work introduces Certified Gaussian Manifold Sampling (C-GMS), a novel trajectory-centric RL framework that learns combined DMP and VIC policies while guaranteeing Lyapunov stability and actuator feasibility by construction. Our approach reframes policy exploration as sampling from a mathematically defined manifold of stable gain schedules. This ensures every policy rollout is guaranteed to be stable and physically realizable, thereby eliminating the need for reward penalties or post-hoc validation. Furthermore, we provide a theoretical guarantee that our approach ensures bounded tracking error even in the presence of bounded model errors and deployment-time uncertainties. We demonstrate the effectiveness of C-GMS in simulation and verify its efficacy on a real robot, paving the way for reliable autonomous interaction in complex environments.

**arXiv ID:** 2511.16330
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
