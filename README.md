# Agent arXiv Daily

**Last Updated:** 2025-10-01 02:49:53

**Total Papers:** 107

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
<summary><strong>Memory Management and Contextual Consistency for Long-Running Low-Code Agents</strong> - Jiexi Xu - [[pdf]](https://arxiv.org/pdf/2509.25250)</summary>

**Abstract:** The rise of AI-native Low-Code/No-Code (LCNC) platforms enables autonomous agents capable of executing complex, long-duration business processes. However, a fundamental challenge remains: memory management. As agents operate over extended periods, they face "memory inflation" and "contextual degradation" issues, leading to inconsistent behavior, error accumulation, and increased computational cost. This paper proposes a novel hybrid memory system designed specifically for LCNC agents. Inspired by cognitive science, our architecture combines episodic and semantic memory components with a proactive "Intelligent Decay" mechanism. This mechanism intelligently prunes or consolidates memories based on a composite score factoring in recency, relevance, and user-specified utility. A key innovation is a user-centric visualization interface, aligned with the LCNC paradigm, which allows non-technical users to manage the agent's memory directly, for instance, by visually tagging which facts should be retained or forgotten. Through simulated long-running task experiments, we demonstrate that our system significantly outperforms traditional approaches like sliding windows and basic RAG, yielding superior task completion rates, contextual consistency, and long-term token cost efficiency. Our findings establish a new framework for building reliable, transparent AI agents capable of effective long-term learning and adaptation.

**arXiv ID:** 2509.25250
</details>

<details>
<summary><strong>Lita: Light Agent Uncovers the Agentic Coding Capabilities of LLMs</strong> - Hankun Dai, Maoquan Wang, Mengnan Qi, Yikai Zhang, Zijian Jin, Yongqiang Yao, Yufan Huang, Shengyu Fu, Elsie Nallipogu - [[pdf]](https://arxiv.org/pdf/2509.25873)</summary>

**Abstract:** Large language models (LLMs) are increasingly being applied to programming tasks, ranging from single-turn code completion to autonomous agents. Current code agent designs frequently depend on complex, hand-crafted workflows and tool sets. However, this reliance on elaborate scaffolding presents several challenges: agent performance becomes overly dependent on prompt tuning and custom design choices, heavy human intervention obscures a model's true underlying capabilities, and intricate pipelines are costly to build and maintain. Furthermore, optimizing complex task prompts increases the risk of data leakage. Currently, when introducing new models, LLM providers like OpenAI and Anthropic often publish benchmark scores to demonstrate their models' coding proficiency, but keep their proprietary evaluation frameworks confidential. To address these limitations, we introduce Lita (Lite Agent), which operationalizes liteness, a principle of minimizing manual design while retaining the essential elements of a fully autonomous agent. Lita enables a more faithful and unified evaluation without elaborate scaffolding. Experiments on the Aider Polyglot and SWE-Bench with frontier models demonstrate that Lita achieves competitive or superior performance compared to workflow-based and agentic baselines. Crucially, Lita also consumes fewer tokens and requires significantly less design effort. Our results suggest that Lita is sufficient to reveal the underlying coding competence of modern LLMs. Finally, we propose the Agent Complexity Law: the performance gap between agents of varying complexity, from simple to sophisticated designs, will shrink as the core model improves, ultimately converging to a negligible difference.

**arXiv ID:** 2509.25873
</details>

<details>
<summary><strong>Trajectory Encryption Cooperative Salvo Guidance</strong> - Lohitvel Gopikannan, Shashi Ranjan Kumar, Abhinav Sinha - [[pdf]](https://arxiv.org/pdf/2509.17341)</summary>

**Abstract:** This paper introduces the concept of trajectory encryption in cooperative simultaneous target interception, wherein heterogeneity in guidance principles across a team of unmanned autonomous systems is leveraged as a strategic design feature. By employing a mix of heterogeneous time-to-go formulations leading to a cooperative guidance strategy, the swarm of vehicles is able to generate diverse trajectory families. This diversity expands the feasible solution space for simultaneous target interception, enhances robustness under disturbances, and enables flexible time-to-go adjustments without predictable detouring. From an adversarial perspective, heterogeneity obscures the collective interception intent by preventing straightforward prediction of swarm dynamics, effectively acting as an encryption layer in the trajectory domain. Simulations demonstrate that the swarm of heterogeneous vehicles is able to intercept a moving target simultaneously from a diverse set of initial engagement configurations.

**arXiv ID:** 2509.17341
</details>

<details>
<summary><strong>Ferret-UI Lite: Lessons from Building Small On-Device GUI Agents</strong> - Zhen Yang, Zi-Yi Dou, Di Feng, Forrest Huang, Anh Nguyen, Keen You, Omar Attia, Yuhao Yang, Michael Feng, Haotian Zhang, Ram Ramrakhya, Chao Jia, Jeffrey Nichols, Alexander Toshev, Yinfei Yang, Zhe Gan - [[pdf]](https://arxiv.org/pdf/2509.26539)</summary>

**Abstract:** Developing autonomous agents that effectively interact with Graphic User Interfaces (GUIs) remains a challenging open problem, especially for small on-device models. In this paper, we present Ferret-UI Lite, a compact, end-to-end GUI agent that operates across diverse platforms, including mobile, web, and desktop. Utilizing techniques optimized for developing small models, we build our 3B Ferret-UI Lite agent through curating a diverse GUI data mixture from real and synthetic sources, strengthening inference-time performance through chain-of-thought reasoning and visual tool-use, and reinforcement learning with designed rewards. Ferret-UI Lite achieves competitive performance with other small-scale GUI agents. In GUI grounding, Ferret-UI Lite attains scores of $91.6\%$, $53.3\%$, and $61.2\%$ on the ScreenSpot-V2, ScreenSpot-Pro, and OSWorld-G benchmarks, respectively. For GUI navigation, Ferret-UI Lite achieves success rates of $28.0\%$ on AndroidWorld and $19.8\%$ on OSWorld. We share our methods and lessons learned from developing compact, on-device GUI agents.

**arXiv ID:** 2509.26539
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (16 papers)</h2></summary>

<details>
<summary><strong>Blueprint-Bench: Comparing spatial intelligence of LLMs, agents and image models</strong> - Lukas Petersson, Axel Backlund, Axel Wennstöm, Hanna Petersson, Callum Sharrock, Arash Dabiri - [[pdf]](https://arxiv.org/pdf/2509.25229)</summary>

**Abstract:** We introduce Blueprint-Bench, a benchmark designed to evaluate spatial reasoning capabilities in AI models through the task of converting apartment photographs into accurate 2D floor plans. While the input modality (photographs) is well within the training distribution of modern multimodal models, the task of spatial reconstruction requires genuine spatial intelligence: inferring room layouts, understanding connectivity, and maintaining consistent scale. We evaluate leading language models (GPT-5, Claude 4 Opus, Gemini 2.5 Pro, Grok-4), image generation models (GPT-Image, NanoBanana), and agent systems (Codex CLI, Claude Code) on a dataset of 50 apartments with approximately 20 interior images each. Our scoring algorithm measures similarity between generated and ground-truth floor plans based on room connectivity graphs and size rankings. Results reveal a significant blind spot in current AI capabilities: most models perform at or below a random baseline, while human performance remains substantially superior. Image generation models particularly struggle with instruction following, while agent-based approaches with iterative refinement capabilities show no meaningful improvement over single-pass generation. Blueprint-Bench provides the first numerical framework for comparing spatial intelligence across different model architectures. We will continue evaluating new models as they are released and welcome community submissions, monitoring for the emergence of spatial intelligence in generalist AI systems.

**arXiv ID:** 2509.25229
</details>

<details>
<summary><strong>Flash-Searcher: Fast and Effective Web Agents via DAG-Based Parallel Execution</strong> - Tianrui Qin, Qianben Chen, Sinuo Wang, He Xing, King Zhu, He Zhu, Dingfeng Shi, Xinxin Liu, Ge Zhang, Jiaheng Liu, Yuchen Eleanor Jiang, Xitong Gao, Wangchunshu Zhou - [[pdf]](https://arxiv.org/pdf/2509.25301)</summary>

**Abstract:** Large language models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks when equipped with external tools. However, current frameworks predominantly rely on sequential processing, leading to inefficient execution particularly for tasks requiring extensive tool interaction. This paper introduces Flash-Searcher, a novel parallel agent reasoning framework that fundamentally reimagines the execution paradigm from sequential chains to directed acyclic graphs (DAGs). Flash-Searcher decomposes complex tasks into subtasks with explicit dependencies, enabling concurrent execution of independent reasoning paths while maintaining logical constraints. Through dynamic workflow optimization, our framework continuously refines the execution graph based on intermediate results, effectively integrating summary module. Comprehensive evaluations across multiple benchmarks demonstrate that Flash-Searcher consistently outperforms existing approaches. Specifically, it achieves 67.7% accuracy on BrowseComp and 83% on xbench-DeepSearch, while reducing agent execution steps by up to 35% compared to current frameworks. Furthermore, when distilling this parallel reasoning pipeline into single models, we observe substantial performance gains across diverse backbone architectures, underscoring the generalizability of our methodology. Our work thus represents a significant advance in agent architecture design, offering a more scalable and efficient paradigm for complex reasoning tasks.

**arXiv ID:** 2509.25301
</details>

<details>
<summary><strong>NuRisk: A Visual Question Answering Dataset for Agent-Level Risk Assessment in Autonomous Driving</strong> - Yuan Gao, Mattia Piccinini, Roberto Brusnicki, Yuchen Zhang, Johannes Betz - [[pdf]](https://arxiv.org/pdf/2509.25944)</summary>

**Abstract:** Understanding risk in autonomous driving requires not only perception and prediction, but also high-level reasoning about agent behavior and context. Current Vision Language Models (VLMs)-based methods primarily ground agents in static images and provide qualitative judgments, lacking the spatio-temporal reasoning needed to capture how risks evolve over time. To address this gap, we propose NuRisk, a comprehensive Visual Question Answering (VQA) dataset comprising 2,900 scenarios and 1.1 million agent-level samples, built on real-world data from nuScenes and Waymo, supplemented with safety-critical scenarios from the CommonRoad simulator. The dataset provides Bird-Eye-View (BEV) based sequential images with quantitative, agent-level risk annotations, enabling spatio-temporal reasoning. We benchmark well-known VLMs across different prompting techniques and find that they fail to perform explicit spatio-temporal reasoning, resulting in a peak accuracy of 33% at high latency. To address these shortcomings, our fine-tuned 7B VLM agent improves accuracy to 41% and reduces latency by 75%, demonstrating explicit spatio-temporal reasoning capabilities that proprietary models lacked. While this represents a significant step forward, the modest accuracy underscores the profound challenge of the task, establishing NuRisk as a critical benchmark for advancing spatio-temporal reasoning in autonomous driving.

**arXiv ID:** 2509.25944
</details>

<details>
<summary><strong>Point-It-Out: Benchmarking Embodied Reasoning for Vision Language Models in Multi-Stage Visual Grounding</strong> - Haotian Xue, Yunhao Ge, Yu Zeng, Zhaoshuo Li, Ming-Yu Liu, Yongxin Chen, Jiaojiao Fan - [[pdf]](https://arxiv.org/pdf/2509.25794)</summary>

**Abstract:** Vision-Language Models (VLMs) have demonstrated impressive world knowledge across a wide range of tasks, making them promising candidates for embodied reasoning applications. However, existing benchmarks primarily evaluate the embodied reasoning ability of VLMs through multiple-choice questions based on image annotations -- for example, selecting which trajectory better describes an event in the image. In this work, we introduce the Point-It-Out (PIO) benchmark, a novel benchmark designed to systematically assess the embodied reasoning abilities of VLMs through precise visual grounding. We propose a hierarchical evaluation protocol spanning three stages (S1: referred-object localization, S2: task-driven pointing, and S3: visual trace prediction), with data collected from critical domains for embodied intelligence, including indoor, kitchen, driving, and robotic manipulation scenarios. Extensive experiments with over ten state-of-the-art VLMs reveal several interesting findings. For example, strong general-purpose models such as GPT-4o, while excelling on many benchmarks (e.g., language, perception, and reasoning), underperform compared to some open-source models in precise visual grounding; models such as MoLMO perform well in S1 and S2 but struggle in S3, where requires grounding combined with visual trace planning.

**arXiv ID:** 2509.25794
</details>

<details>
<summary><strong>DyFlow: Dynamic Workflow Framework for Agentic Reasoning</strong> - Yanbo Wang, Zixiang Xu, Yue Huang, Xiangqi Wang, Zirui Song, Lang Gao, Chenxi Wang, Xiangru Tang, Yue Zhao, Arman Cohan, Xiangliang Zhang, Xiuying Chen - [[pdf]](https://arxiv.org/pdf/2509.26062)</summary>

**Abstract:** Agent systems based on large language models (LLMs) have shown great potential in complex reasoning tasks, but building efficient and generalizable workflows remains a major challenge. Most existing approaches rely on manually designed processes, which limits their adaptability across different tasks. While a few methods attempt automated workflow generation, they are often tied to specific datasets or query types and make limited use of intermediate feedback, reducing system robustness and reasoning depth. Moreover, their operations are typically predefined and inflexible. To address these limitations, we propose DyFlow, a dynamic workflow generation framework that adaptively constructs and adjusts reasoning procedures based on task requirements and real-time intermediate feedback, thereby enhancing cross-task generalization. DyFlow consists of two core components: a designer and an executor. The designer decomposes complex problems into a sequence of sub-goals defined by high-level objectives and dynamically plans the next steps based on intermediate outputs and feedback. These plans are then carried out by the executor, which executes each operation using dynamic operators with context-aware parameterization, enabling flexible and semantically grounded reasoning. We systematically evaluate DyFlow across diverse domains, including social reasoning, biomedical tasks, mathematical problem solving, and code generation. Results demonstrate that DyFlow significantly outperforms existing baselines, achieving substantial Pass@k improvements and exhibiting robust generalization across diverse domains. The code is publicly available at this https URL.

**arXiv ID:** 2509.26062
</details>

<details>
<summary><strong>OceanGym: A Benchmark Environment for Underwater Embodied Agents</strong> - Yida Xue, Mingjun Mao, Xiangyuan Ru, Yuqi Zhu, Baochang Ren, Shuofei Qiao, Mengru Wang, Shumin Deng, Xinyu An, Ningyu Zhang, Ying Chen, Huajun Chen - [[pdf]](https://arxiv.org/pdf/2509.26536)</summary>

**Abstract:** We introduce OceanGym, the first comprehensive benchmark for ocean underwater embodied agents, designed to advance AI in one of the most demanding real-world environments. Unlike terrestrial or aerial domains, underwater settings present extreme perceptual and decision-making challenges, including low visibility, dynamic ocean currents, making effective agent deployment exceptionally difficult. OceanGym encompasses eight realistic task domains and a unified agent framework driven by Multi-modal Large Language Models (MLLMs), which integrates perception, memory, and sequential decision-making. Agents are required to comprehend optical and sonar data, autonomously explore complex environments, and accomplish long-horizon objectives under these harsh conditions. Extensive experiments reveal substantial gaps between state-of-the-art MLLM-driven agents and human experts, highlighting the persistent difficulty of perception, planning, and adaptability in ocean underwater environments. By providing a high-fidelity, rigorously designed platform, OceanGym establishes a testbed for developing robust embodied AI and transferring these capabilities to real-world autonomous ocean underwater vehicles, marking a decisive step toward intelligent agents capable of operating in one of Earth's last unexplored frontiers. The code and data are available at this https URL.

**arXiv ID:** 2509.26536
</details>

<details>
<summary><strong>Agent-as-Judge for Factual Summarization of Long Narratives</strong> - Yeonseok Jeong, Minsoo Kim, Seung-won Hwang, Byung-Hak Kim - [[pdf]](https://arxiv.org/pdf/2501.09993)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated near-human performance in summarization tasks based on traditional metrics such as ROUGE and BERTScore. However, these metrics do not adequately capture critical aspects of summarization quality, such as factual accuracy, particularly for long narratives (>100K tokens). Recent advances, such as LLM-as-a-Judge, address the limitations of metrics based on lexical similarity but still exhibit factual inconsistencies, especially in understanding character relationships and states. In this work, we introduce NarrativeFactScore, a novel "Agent-as-a-Judge" framework for evaluating and refining summaries. By leveraging a Character Knowledge Graph (CKG) extracted from input and generated summaries, NarrativeFactScore assesses the factual consistency and provides actionable guidance for refinement, such as identifying missing or erroneous facts. We demonstrate the effectiveness of NarrativeFactScore through a detailed workflow illustration and extensive validation on widely adopted benchmarks, achieving superior performance compared to competitive methods. Our results highlight the potential of agent-driven evaluation systems to improve the factual reliability of LLM-generated summaries.

**arXiv ID:** 2501.09993
</details>

<details>
<summary><strong>GraphSearch: An Agentic Deep Searching Workflow for Graph Retrieval-Augmented Generation</strong> - Cehao Yang, Xiaojun Wu, Xueyuan Lin, Chengjin Xu, Xuhui Jiang, Yuanliang Sun, Jia Li, Hui Xiong, Jian Guo - [[pdf]](https://arxiv.org/pdf/2509.22009)</summary>

**Abstract:** Graph Retrieval-Augmented Generation (GraphRAG) enhances factual reasoning in LLMs by structurally modeling knowledge through graph-based representations. However, existing GraphRAG approaches face two core limitations: shallow retrieval that fails to surface all critical evidence, and inefficient utilization of pre-constructed structural graph data, which hinders effective reasoning from complex queries. To address these challenges, we propose \textsc{GraphSearch}, a novel agentic deep searching workflow with dual-channel retrieval for GraphRAG. \textsc{GraphSearch} organizes the retrieval process into a modular framework comprising six modules, enabling multi-turn interactions and iterative reasoning. Furthermore, \textsc{GraphSearch} adopts a dual-channel retrieval strategy that issues semantic queries over chunk-based text data and relational queries over structural graph data, enabling comprehensive utilization of both modalities and their complementary strengths. Experimental results across six multi-hop RAG benchmarks demonstrate that \textsc{GraphSearch} consistently improves answer accuracy and generation quality over the traditional strategy, confirming \textsc{GraphSearch} as a promising direction for advancing graph retrieval-augmented generation.

**arXiv ID:** 2509.22009
</details>

<details>
<summary><strong>Agentar-Scale-SQL: Advancing Text-to-SQL through Orchestrated Test-Time Scaling</strong> - Pengfei Wang, Baolin Sun, Xuemei Dong, Yaxun Dai, Hongwei Yuan, Mengdie Chu, Yingqi Gao, Xiang Qi, Peng Zhang, Ying Yan - [[pdf]](https://arxiv.org/pdf/2509.24403)</summary>

**Abstract:** State-of-the-art (SOTA) Text-to-SQL methods still lag significantly behind human experts on challenging benchmarks like BIRD. Current approaches that explore test-time scaling lack an orchestrated strategy and neglect the model's internal reasoning process. To bridge this gap, we introduce Agentar-Scale-SQL, a novel framework leveraging scalable computation to improve performance. Agentar-Scale-SQL implements an Orchestrated Test-Time Scaling strategy that synergistically combines three distinct perspectives: i) Internal Scaling via RL-enhanced Intrinsic Reasoning, ii) Sequential Scaling through Iterative Refinement, and iii) Parallel Scaling using Diverse Synthesis and Tournament Selection. Agentar-Scale-SQL is a general-purpose framework designed for easy adaptation to new databases and more powerful language models. Extensive experiments show that Agentar-Scale-SQL achieves SOTA performance on the BIRD benchmark, reaching 81.67\% execution accuracy on the test set and ranking first on the official leaderboard, demonstrating an effective path toward human-level performance.

**arXiv ID:** 2509.24403
</details>

<details>
<summary><strong>MSCoD: An Enhanced Bayesian Updating Framework with Multi-Scale Information Bottleneck and Cooperative Attention for Structure-Based Drug Design</strong> - Long Xu, Yongcai Chen, Fengshuo Liu, Yuzhong Peng - [[pdf]](https://arxiv.org/pdf/2509.25225)</summary>

**Abstract:** Structure-Based Drug Design (SBDD) is a powerful strategy in computational drug discovery, utilizing three-dimensional protein structures to guide the design of molecules with improved binding affinity. However, capturing complex protein-ligand interactions across multiple scales remains challenging, as current methods often overlook the hierarchical organization and intrinsic asymmetry of these interactions. To address these limitations, we propose MSCoD, a novel Bayesian updating-based generative framework for structure-based drug design. In our MSCoD, Multi-Scale Information Bottleneck (MSIB) was developed, which enables semantic compression at multiple abstraction levels for efficient hierarchical feature extraction. Furthermore, a multi-head cooperative attention (MHCA) mechanism was developed, which employs asymmetric protein-to-ligand attention to capture diverse interaction types while addressing the dimensionality disparity between proteins and ligands. Empirical studies showed that MSCoD outperforms state-of-the-art methods on the benchmark dataset. Case studies on challenging targets such as KRAS G12D further demonstrate its applicability in real-world scenarios. The code and data underlying this article are freely available at this https URL.

**arXiv ID:** 2509.25225
</details>

<details>
<summary><strong>EEsizer: LLM-Based AI Agent for Sizing of Analog and Mixed Signal Circuit</strong> - Chang Liu, Danial Chitnis - [[pdf]](https://arxiv.org/pdf/2509.25510)</summary>

**Abstract:** The design of Analog and Mixed-Signal (AMS) integrated circuits (ICs) often involves significant manual effort, especially during the transistor sizing process. While Machine Learning techniques in Electronic Design Automation (EDA) have shown promise in reducing complexity and minimizing human intervention, they still face challenges such as numerous iterations and a lack of knowledge about AMS circuit design. Recently, Large Language Models (LLMs) have demonstrated significant potential across various fields, showing a certain level of knowledge in circuit design and indicating their potential to automate the transistor sizing process. In this work, we propose EEsizer, an LLM-based AI agent that integrates large language models with circuit simulators and custom data analysis functions, enabling fully automated, closed-loop transistor sizing without relying on external knowledge. By employing prompt engineering and Chain-of-Thought reasoning, the agent iteratively explores design directions, evaluates performance, and refines solutions with minimal human intervention. We first benchmarked 8 LLMs on six basic circuits and selected three high-performing models to optimize a 20-transistor CMOS operational amplifier, targeting multiple performance metrics, including rail-to-rail operation from 180 nm to 90 nm technology nodes. Notably, OpenAI o3 successfully achieved the user-intended target at 90 nm across three different test groups, with a maximum of 20 iterations, demonstrating adaptability and robustness at advanced nodes. To assess design robustness, we manually designed a bias circuit and performed a variation analysis using Gaussian-distributed variations on transistor dimensions and threshold voltages.

**arXiv ID:** 2509.25510
</details>

<details>
<summary><strong>ACT: Agentic Classification Tree</strong> - Vincent Grari, Tim Arni, Thibault Laugel, Sylvain Lamprier, James Zou, Marcin Detyniecki - [[pdf]](https://arxiv.org/pdf/2509.26433)</summary>

**Abstract:** When used in high-stakes settings, AI systems are expected to produce decisions that are transparent, interpretable, and auditable, a requirement increasingly expected by regulations. Decision trees such as CART provide clear and verifiable rules, but they are restricted to structured tabular data and cannot operate directly on unstructured inputs such as text. In practice, large language models (LLMs) are widely used for such data, yet prompting strategies such as chain-of-thought or prompt optimization still rely on free-form reasoning, limiting their ability to ensure trustworthy behaviors. We present the Agentic Classification Tree (ACT), which extends decision-tree methodology to unstructured inputs by formulating each split as a natural-language question, refined through impurity-based evaluation and LLM feedback via TextGrad. Experiments on text benchmarks show that ACT matches or surpasses prompting-based baselines while producing transparent and interpretable decision paths.

**arXiv ID:** 2509.26433
</details>

<details>
<summary><strong>RANGER -- Repository-Level Agent for Graph-Enhanced Retrieval</strong> - Pratik Shah, Rajat Ghosh, Aryan Singhal, Debojyoti Dutta - [[pdf]](https://arxiv.org/pdf/2509.25257)</summary>

**Abstract:** General-purpose automated software engineering (ASE) includes tasks such as code completion, retrieval, repair, QA, and summarization. These tasks require a code retrieval system that can handle specific queries about code entities, or code entity queries (for example, locating a specific class or retrieving the dependencies of a function), as well as general queries without explicit code entities, or natural language queries (for example, describing a task and retrieving the corresponding code). We present RANGER, a repository-level code retrieval agent designed to address both query types, filling a gap in recent works that have focused primarily on code-entity queries. We first present a tool that constructs a comprehensive knowledge graph of the entire repository, capturing hierarchical and cross-file dependencies down to the variable level, and augments graph nodes with textual descriptions and embeddings to bridge the gap between code and natural language. RANGER then operates on this graph through a dual-stage retrieval pipeline. Entity-based queries are answered through fast Cypher lookups, while natural language queries are handled by MCTS-guided graph exploration. We evaluate RANGER across four diverse benchmarks that represent core ASE tasks including code search, question answering, cross-file dependency retrieval, and repository-level code completion. On CodeSearchNet and RepoQA it outperforms retrieval baselines that use embeddings from strong models such as Qwen3-8B. On RepoBench, it achieves superior cross-file dependency retrieval over baselines, and on CrossCodeEval, pairing RANGER with BM25 delivers the highest exact match rate in code completion compared to other RAG methods.

**arXiv ID:** 2509.25257
</details>

<details>
<summary><strong>Reinforced Embodied Planning with Verifiable Reward for Real-World Robotic Manipulation</strong> - Zitong Bo, Yue Hu, Jinming Ma, Mingliang Zhou, Junhui Yin, Yachen Kang, Yuqi Liu, Tong Wu, Diyun Xiang, Hao Chen - [[pdf]](https://arxiv.org/pdf/2509.25852)</summary>

**Abstract:** Enabling robots to execute long-horizon manipulation tasks from free-form language instructions remains a fundamental challenge in embodied AI. While vision-language models (VLMs) have shown promise as high-level planners, their deployment in the real world is hindered by two gaps: (i) the scarcity of large-scale, sequential manipulation data that couples natural language with multi-step action plans, and (ii) the absence of dense, interpretable rewards for fine-tuning VLMs on planning objectives. To address these issues, we propose REVER, a framework that empowers VLMs to generate and validate long-horizon manipulation plans from natural language instructions in real-world scenarios. Under REVER we train and release RoboFarseer, a VLM incentivized to emit chain-of-thought that perform temporal and spatial reasoning, ensuring physically plausible and logically coherent plans. To obtain training data, we leverage the Universal Manipulation Interface framework to capture hardware-agnostic demonstrations of atomic skills. An automated annotation engine converts each demonstration into vision-instruction-plan triplet. We introduce a verifiable reward that scores the generated plan by its ordered bipartite matching overlap with the ground-truth skill sequence. At run time, the fine-tuned VLM functions both as a planner and as a monitor, verifying step-wise completion. RoboFarseer matches or exceeds the performance of proprietary models that are orders of magnitude larger, while on open-ended planning it surpasses the best baseline by more than 40%. In real-world, long-horizon tasks, the complete system boosts overall success by roughly 60% compared with the same low-level controller without the planner. We will open-source both the dataset and the trained model upon publication.

**arXiv ID:** 2509.25852
</details>

<details>
<summary><strong>Side Scan Sonar-based SLAM for Autonomous Algae Farm Monitoring</strong> - Julian Valdez, Ignacio Torroba, John Folkesson, Ivan Stenius - [[pdf]](https://arxiv.org/pdf/2509.26121)</summary>

**Abstract:** The transition of seaweed farming to an alternative food source on an industrial scale relies on automating its processes through smart farming, equivalent to land agriculture. Key to this process are autonomous underwater vehicles (AUVs) via their capacity to automate crop and structural inspections. However, the current bottleneck for their deployment is ensuring safe navigation within farms, which requires an accurate, online estimate of the AUV pose and map of the infrastructure. To enable this, we propose an efficient side scan sonar-based (SSS) simultaneous localization and mapping (SLAM) framework that exploits the geometry of kelp farms via modeling structural ropes in the back-end as sequences of individual landmarks from each SSS ping detection, instead of combining detections into elongated representations. Our method outperforms state of the art solutions in hardware in the loop (HIL) experiments on a real AUV survey in a kelp farm. The framework and dataset can be found at this https URL.

**arXiv ID:** 2509.26121
</details>

<details>
<summary><strong>LargeAD: Large-Scale Cross-Sensor Data Pretraining for Autonomous Driving</strong> - Lingdong Kong, Xiang Xu, Youquan Liu, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu - [[pdf]](https://arxiv.org/pdf/2501.04005)</summary>

**Abstract:** Recent advancements in vision foundation models (VFMs) have revolutionized visual perception in 2D, yet their potential for 3D scene understanding, particularly in autonomous driving applications, remains underexplored. In this paper, we introduce LargeAD, a versatile and scalable framework designed for large-scale 3D pretraining across diverse real-world driving datasets. Our framework leverages VFMs to extract semantically rich superpixels from 2D images, which are aligned with LiDAR point clouds to generate high-quality contrastive samples. This alignment facilitates cross-modal representation learning, enhancing the semantic consistency between 2D and 3D data. We introduce several key innovations: (i) VFM-driven superpixel generation for detailed semantic representation, (ii) a VFM-assisted contrastive learning strategy to align multimodal features, (iii) superpoint temporal consistency to maintain stable representations across time, and (iv) multi-source data pretraining to generalize across various LiDAR configurations. Our approach achieves substantial gains over state-of-the-art methods in linear probing and fine-tuning for LiDAR-based segmentation and object detection. Extensive experiments on 11 large-scale multi-sensor datasets highlight our superior performance, demonstrating adaptability, efficiency, and robustness in real-world autonomous driving scenarios.

**arXiv ID:** 2501.04005
</details>

</details>

<details open>
<summary><h2>LLM Agents (13 papers)</h2></summary>

<details>
<summary><strong>Dive into the Agent Matrix: A Realistic Evaluation of Self-Replication Risk in LLM Agents</strong> - Boxuan Zhang, Yi Yu, Jiaxuan Guo, Jing Shao - [[pdf]](https://arxiv.org/pdf/2509.25302)</summary>

**Abstract:** The widespread deployment of Large Language Model (LLM) agents across real-world applications has unlocked tremendous potential, while raising some safety concerns. Among these concerns, the self-replication risk of LLM agents driven by objective misalignment (just like Agent Smith in the movie The Matrix) has drawn growing attention. Previous studies mainly examine whether LLM agents can self-replicate when directly instructed, potentially overlooking the risk of spontaneous replication driven by real-world settings (e.g., ensuring survival against termination threats). In this paper, we present a comprehensive evaluation framework for quantifying self-replication risks. Our framework establishes authentic production environments and realistic tasks (e.g., dynamic load balancing) to enable scenario-driven assessment of agent behaviors. Designing tasks that might induce misalignment between users' and agents' objectives makes it possible to decouple replication success from risk and capture self-replication risks arising from these misalignment settings. We further introduce Overuse Rate ($\mathrm{OR}$) and Aggregate Overuse Count ($\mathrm{AOC}$) metrics, which precisely capture the frequency and severity of uncontrolled replication. In our evaluation of 21 state-of-the-art open-source and proprietary models, we observe that over 50\% of LLM agents display a pronounced tendency toward uncontrolled self-replication, reaching an overall Risk Score ($\Phi_\mathrm{R}$) above a safety threshold of 0.5 when subjected to operational pressures. Our results underscore the urgent need for scenario-driven risk assessment and robust safeguards in the practical deployment of LLM agents.

**arXiv ID:** 2509.25302
</details>

<details>
<summary><strong>Where LLM Agents Fail and How They can Learn From Failures</strong> - Kunlun Zhu, Zijia Liu, Bingxuan Li, Muxin Tian, Yingxuan Yang, Jiaxun Zhang, Pengrui Han, Qipeng Xie, Fuyang Cui, Weijia Zhang, Xiaoteng Ma, Xiaodong Yu, Gowtham Ramesh, Jialian Wu, Zicheng Liu, Pan Lu, James Zou, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2509.25370)</summary>

**Abstract:** Large Language Model (LLM) agents, which integrate planning, memory, reflection, and tool-use modules, have shown promise in solving complex, multi-step tasks. Yet their sophisticated architectures amplify vulnerability to cascading failures, where a single root-cause error propagates through subsequent decisions, leading to task failure. Current systems lack a framework that can comprehensively understand agent error in a modular and systemic way, and therefore fail to detect these errors accordingly. We address this gap with three contributions. First, we introduce the AgentErrorTaxonomy, a modular classification of failure modes spanning memory, reflection, planning, action, and system-level operations. Second, we construct AgentErrorBench, the first dataset of systematically annotated failure trajectories from ALFWorld, GAIA, and WebShop, grounding error analysis in real-world agent rollouts. Third, we propose AgentDebug, a debugging framework that isolates root-cause failures and provides corrective feedback, enabling agents to recover and iteratively improve. Experiments on AgentErrorBench show that AgentDebug achieves 24% higher all-correct accuracy and 17% higher step accuracy compared to the strongest baseline. Beyond detection, the targeted feedback generated by AgentDebug enables LLM agents to iteratively recover from failures, yielding up to 26% relative improvements in task success across ALFWorld, GAIA, and WebShop. These results establish principled debugging as a pathway to more reliable and adaptive LLM agents. The code and data will be available at this https URL

**arXiv ID:** 2509.25370
</details>

<details>
<summary><strong>RadOnc-GPT: An Autonomous LLM Agent for Real-Time Patient Outcomes Labeling at Scale</strong> - Jason Holmes, Yuexing Hao, Mariana Borras-Osorio, Federico Mastroleo, Santiago Romero Brufau, Valentina Carducci, Katie M Van Abel, David M Routman, Andrew Y. K. Foong, Liv M Muller, Satomi Shiraishi, Daniel K Ebner, Daniel J Ma, Sameer R Keole, Samir H Patel, Mirek Fatyga, Martin Bues, Brad J Stish, Yolanda I Garces, Michelle A Neben Wittich, Robert L Foote, Sujay A Vora, Nadia N Laack, Mark R Waddle, Wei Liu - [[pdf]](https://arxiv.org/pdf/2509.25540)</summary>

**Abstract:** Manual labeling limits the scale, accuracy, and timeliness of patient outcomes research in radiation oncology. We present RadOnc-GPT, an autonomous large language model (LLM)-based agent capable of independently retrieving patient-specific information, iteratively assessing evidence, and returning structured outcomes. Our evaluation explicitly validates RadOnc-GPT across two clearly defined tiers of increasing complexity: (1) a structured quality assurance (QA) tier, assessing the accurate retrieval of demographic and radiotherapy treatment plan details, followed by (2) a complex clinical outcomes labeling tier involving determination of mandibular osteoradionecrosis (ORN) in head-and-neck cancer patients and detection of cancer recurrence in independent prostate and head-and-neck cancer cohorts requiring combined interpretation of structured and unstructured patient data. The QA tier establishes foundational trust in structured-data retrieval, a critical prerequisite for successful complex clinical outcome labeling.

**arXiv ID:** 2509.25540
</details>

<details>
<summary><strong>Causal Autoencoder-like Generation of Feedback Fuzzy Cognitive Maps with an LLM Agent</strong> - Akash Kumar Panda, Olaoluwa Adigun, Bart Kosko - [[pdf]](https://arxiv.org/pdf/2509.25593)</summary>

**Abstract:** A large language model (LLM) can map a feedback causal fuzzy cognitive map (FCM) into text and then reconstruct the FCM from the text. This explainable AI system approximates an identity map from the FCM to itself and resembles the operation of an autoencoder (AE). Both the encoder and the decoder explain their decisions in contrast to black-box AEs. Humans can read and interpret the encoded text in contrast to the hidden variables and synaptic webs in AEs. The LLM agent approximates the identity map through a sequence of system instructions that does not compare the output to the input. The reconstruction is lossy because it removes weak causal edges or rules while it preserves strong causal edges. The encoder preserves the strong causal edges even when it trades off some details about the FCM to make the text sound more natural.

**arXiv ID:** 2509.25593
</details>

<details>
<summary><strong>SafeMind: Benchmarking and Mitigating Safety Risks in Embodied LLM Agents</strong> - Ruolin Chen, Yinqian Sun, Jihang Wang, Mingyang Lv, Qian Zhang, Yi Zeng - [[pdf]](https://arxiv.org/pdf/2509.25885)</summary>

**Abstract:** Embodied agents powered by large language models (LLMs) inherit advanced planning capabilities; however, their direct interaction with the physical world exposes them to safety vulnerabilities. In this work, we identify four key reasoning stages where hazards may arise: Task Understanding, Environment Perception, High-Level Plan Generation, and Low-Level Action Generation. We further formalize three orthogonal safety constraint types (Factual, Causal, and Temporal) to systematically characterize potential safety violations. Building on this risk model, we present SafeMindBench, a multimodal benchmark with 5,558 samples spanning four task categories (Instr-Risk, Env-Risk, Order-Fix, Req-Align) across high-risk scenarios such as sabotage, harm, privacy, and illegal behavior. Extensive experiments on SafeMindBench reveal that leading LLMs (e.g., GPT-4o) and widely used embodied agents remain susceptible to safety-critical failures. To address this challenge, we introduce SafeMindAgent, a modular Planner-Executor architecture integrated with three cascaded safety modules, which incorporate safety constraints into the reasoning process. Results show that SafeMindAgent significantly improves safety rate over strong baselines while maintaining comparable task completion. Together, SafeMindBench and SafeMindAgent provide both a rigorous evaluation suite and a practical solution that advance the systematic study and mitigation of safety risks in embodied LLM agents.

**arXiv ID:** 2509.25885
</details>

<details>
<summary><strong>LLM Agents for Knowledge Discovery in Atomic Layer Processing</strong> - Andreas Werbrouck, Marshall B. Lindsay, Matthew Maschmann, Matthias J. Young - [[pdf]](https://arxiv.org/pdf/2509.26201)</summary>

**Abstract:** Large Language Models (LLMs) have garnered significant attention for several years now. Recently, their use as independently reasoning agents has been proposed. In this work, we test the potential of such agents for knowledge discovery in materials science. We repurpose LangGraph's tool functionality to supply agents with a black box function to interrogate. In contrast to process optimization or performing specific, user-defined tasks, knowledge discovery consists of freely exploring the system, posing and verifying statements about the behavior of this black box, with the sole objective of generating and verifying generalizable statements. We provide proof of concept for this approach through a children's parlor game, demonstrating the role of trial-and-error and persistence in knowledge discovery, and the strong path-dependence of results. We then apply the same strategy to show that LLM agents can explore, discover, and exploit diverse chemical interactions in an advanced Atomic Layer Processing reactor simulation using intentionally limited probe capabilities without explicit instructions.

**arXiv ID:** 2509.26201
</details>

<details>
<summary><strong>Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents</strong> - Shuai Shao, Qihan Ren, Chen Qian, Boyi Wei, Dadi Guo, Jingyi Yang, Xinhao Song, Linfeng Zhang, Weinan Zhang, Dongrui Liu, Jing Shao - [[pdf]](https://arxiv.org/pdf/2509.26354)</summary>

**Abstract:** Advances in Large Language Models (LLMs) have enabled a new class of self-evolving agents that autonomously improve through interaction with the environment, demonstrating strong capabilities. However, self-evolution also introduces novel risks overlooked by current safety research. In this work, we study the case where an agent's self-evolution deviates in unintended ways, leading to undesirable or even harmful outcomes. We refer to this as Misevolution. To provide a systematic investigation, we evaluate misevolution along four key evolutionary pathways: model, memory, tool, and workflow. Our empirical findings reveal that misevolution is a widespread risk, affecting agents built even on top-tier LLMs (e.g., Gemini-2.5-Pro). Different emergent risks are observed in the self-evolutionary process, such as the degradation of safety alignment after memory accumulation, or the unintended introduction of vulnerabilities in tool creation and reuse. To our knowledge, this is the first study to systematically conceptualize misevolution and provide empirical evidence of its occurrence, highlighting an urgent need for new safety paradigms for self-evolving agents. Finally, we discuss potential mitigation strategies to inspire further research on building safer and more trustworthy self-evolving agents. Our code and data are available at this https URL . Warning: this paper includes examples that may be offensive or harmful in nature.

**arXiv ID:** 2509.26354
</details>

<details>
<summary><strong>PALADIN: Self-Correcting Language Model Agents to Cure Tool-Failure Cases</strong> - Sri Vatsa Vuddanti, Aarav Shah, Satwik Kumar Chittiprolu, Tony Song, Sunishchal Dev, Kevin Zhu, Maheep Chaudhary - [[pdf]](https://arxiv.org/pdf/2509.25238)</summary>

**Abstract:** Tool-augmented language agents frequently fail in real-world deployment due to tool malfunctions--timeouts, API exceptions, or inconsistent outputs--triggering cascading reasoning errors and task abandonment. Existing agent training pipelines optimize only for success trajectories, failing to expose models to the tool failures that dominate real-world usage. We propose \textbf{PALADIN}, a generalizable framework for equipping language agents with robust failure recovery capabilities. PALADIN trains on 50,000+ recovery-annotated trajectories constructed via systematic failure injection and expert demonstrations on an enhanced ToolBench dataset. Training uses LoRA-based fine-tuning to retain base capabilities while injecting recovery competence. At inference, PALADIN detects execution-time errors and retrieves the most similar case from a curated bank of 55+ failure exemplars aligned with ToolScan's taxonomy, then executes the corresponding recovery action. This approach generalizes to novel failures beyond the training distribution, retaining 95.2\% recovery performance on unseen tool APIs. Evaluation across PaladinEval and ToolReflectEval demonstrates consistent improvements in Recovery Rate (RR), Task Success Rate (TSR), Catastrophic Success Rate (CSR), and Efficiency Score (ES). PALADIN improves RR from 32.76% to 89.68% (+57% relative) over ToolBench and outperforms the strongest baseline CRITIC (76.34%) by +13.3%. Against vanilla agents, PALADIN achieves 89.86\% RR (+66% relative improvement from 23.75%). These results establish PALADIN as an effective method for building fault-tolerant agents capable of robust recovery in real-world tool environments.

**arXiv ID:** 2509.25238
</details>

<details>
<summary><strong>BuildBench: Benchmarking LLM Agents on Compiling Real-World Open-Source Software</strong> - Zehua Zhang, Ati Priya Bajaj, Divij Handa, Siyu Liu, Arvind S Raj, Hongkai Chen, Hulin Wang, Yibo Liu, Zion Leonahenahe Basque, Souradip Nath, Vishal Juneja, Nikhil Chapre, Yan Shoshitaishvili, Adam Doupé, Chitta Baral, Ruoyu Wang - [[pdf]](https://arxiv.org/pdf/2509.25248)</summary>

**Abstract:** Automatically compiling open-source software (OSS) projects is a vital, labor-intensive, and complex task, which makes it a good challenge for LLM Agents. Existing methods rely on manually curated rules and workflows, which cannot adapt to OSS that requires customized configuration or environment setup. Recent attempts using Large Language Models (LLMs) used selective evaluation on a subset of highly rated OSS, a practice that underestimates the realistic challenges of OSS compilation. In practice, compilation instructions are often absent, dependencies are undocumented, and successful builds may even require patching source files or modifying build scripts. We propose a more challenging and realistic benchmark, BUILD-BENCH, comprising OSS that are more diverse in quality, scale, and characteristics. Furthermore, we propose a strong baseline LLM-based agent, OSS-BUILD-AGENT, an effective system with enhanced build instruction retrieval module that achieves state-of-the-art performance on BUILD-BENCH and is adaptable to heterogeneous OSS characteristics. We also provide detailed analysis regarding different compilation method design choices and their influence to the whole task, offering insights to guide future advances. We believe performance on BUILD-BENCH can faithfully reflect an agent's ability to tackle compilation as a complex software engineering tasks, and, as such, our benchmark will spur innovation with a significant impact on downstream applications in the fields of software development and software security.

**arXiv ID:** 2509.25248
</details>

<details>
<summary><strong>STAC: When Innocent Tools Form Dangerous Chains to Jailbreak LLM Agents</strong> - Jing-Jing Li, Jianfeng He, Chao Shang, Devang Kulshreshtha, Xun Xian, Yi Zhang, Hang Su, Sandesh Swamy, Yanjun Qi - [[pdf]](https://arxiv.org/pdf/2509.25624)</summary>

**Abstract:** As LLMs advance into autonomous agents with tool-use capabilities, they introduce security challenges that extend beyond traditional content-based LLM safety concerns. This paper introduces Sequential Tool Attack Chaining (STAC), a novel multi-turn attack framework that exploits agent tool use. STAC chains together tool calls that each appear harmless in isolation but, when combined, collectively enable harmful operations that only become apparent at the final execution step. We apply our framework to automatically generate and systematically evaluate 483 STAC cases, featuring 1,352 sets of user-agent-environment interactions and spanning diverse domains, tasks, agent types, and 10 failure modes. Our evaluations show that state-of-the-art LLM agents, including GPT-4.1, are highly vulnerable to STAC, with attack success rates (ASR) exceeding 90% in most cases. The core design of STAC's automated framework is a closed-loop pipeline that synthesizes executable multi-step tool chains, validates them through in-environment execution, and reverse-engineers stealthy multi-turn prompts that reliably induce agents to execute the verified malicious sequence. We further perform defense analysis against STAC and find that existing prompt-based defenses provide limited protection. To address this gap, we propose a new reasoning-driven defense prompt that achieves far stronger protection, cutting ASR by up to 28.8%. These results highlight a crucial gap: defending tool-enabled agents requires reasoning over entire action sequences and their cumulative effects, rather than evaluating isolated prompts or responses.

**arXiv ID:** 2509.25624
</details>

<details>
<summary><strong>VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications</strong> - Wei He, Yueqing Sun, Hongyan Hao, Xueyuan Hao, Zhikang Xia, Qi Gu, Chengcheng Han, Dengchang Zhao, Hui Su, Kefeng Zhang, Man Gao, Xi Su, Xiaodong Cai, Xunliang Cai, Yu Yang, Yunke Zhao - [[pdf]](https://arxiv.org/pdf/2509.26490)</summary>

**Abstract:** As LLM-based agents are increasingly deployed in real-life scenarios, existing benchmarks fail to capture their inherent complexity of handling extensive information, leveraging diverse resources, and managing dynamic user interactions. To address this gap, we introduce VitaBench, a challenging benchmark that evaluates agents on versatile interactive tasks grounded in real-world settings. Drawing from daily applications in food delivery, in-store consumption, and online travel services, VitaBench presents agents with the most complex life-serving simulation environment to date, comprising 66 tools. Through a framework that eliminates domain-specific policies, we enable flexible composition of these scenarios and tools, yielding 100 cross-scenario tasks (main results) and 300 single-scenario tasks. Each task is derived from multiple real user requests and requires agents to reason across temporal and spatial dimensions, utilize complex tool sets, proactively clarify ambiguous instructions, and track shifting user intent throughout multi-turn conversations. Moreover, we propose a rubric-based sliding window evaluator, enabling robust assessment of diverse solution pathways in complex environments and stochastic interactions. Our comprehensive evaluation reveals that even the most advanced models achieve only 30% success rate on cross-scenario tasks, and less than 50% success rate on others. Overall, we believe VitaBench will serve as a valuable resource for advancing the development of AI agents in practical real-world applications. The code, dataset, and leaderboard are available at this https URL

**arXiv ID:** 2509.26490
</details>

<details>
<summary><strong>Preemptive Detection and Correction of Misaligned Actions in LLM Agents</strong> - Haishuo Fang, Xiaodan Zhu, Iryna Gurevych - [[pdf]](https://arxiv.org/pdf/2407.11843)</summary>

**Abstract:** Deploying LLM-based agents in real-life applications often faces a critical challenge: the misalignment between agents' behavior and user intent. Such misalignment may lead agents to unintentionally execute critical actions that carry negative outcomes (e.g., accidentally triggering a "buy-now" in web shopping), resulting in undesirable or even irreversible consequences. Although addressing these issues is crucial, the preemptive detection and correction of misaligned actions remains relatively underexplored. To fill this gap, we introduce InferAct, a novel approach that leverages the belief reasoning ability of LLMs, grounded in Theory-of-Mind, to detect misaligned actions before execution. Once the misalignment is detected, InferAct alerts users for timely correction, preventing adverse outcomes and enhancing the reliability of LLM agents' decision-making processes. Experiments on three widely used tasks demonstrate that InferAct achieves up to 20% improvements on Marco-F1 against baselines in misaligned action detection. An in-depth evaluation of misalignment correction further highlights InferAct's effectiveness in improving agent alignment.

**arXiv ID:** 2407.11843
</details>

<details>
<summary><strong>Dual-Scale World Models for LLM Agents Towards Hard-Exploration Problems</strong> - Minsoo Kim, Seung-won Hwang - [[pdf]](https://arxiv.org/pdf/2509.24116)</summary>

**Abstract:** LLM-based agents have seen promising advances, yet they are still limited in "hard-exploration" tasks requiring learning new knowledge through exploration. We present GLoW, a novel approach leveraging dual-scale world models, maintaining a trajectory frontier of high-value discoveries at the global scale, while learning from local trial-and-error in exploration through a Multi-path Advantage Reflection mechanism which infers advantage-based progress signals to guide exploration. To evaluate our framework for hard-exploration, we tackle the Jericho benchmark suite of text-based games, where GLoW achieves a new state-of-theart performance for LLM-based approaches. Compared to state-of-the-art RLbased methods, our approach achieves comparable performance while requiring 100-800x fewer environment interactions.

**arXiv ID:** 2509.24116
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (23 papers)</h2></summary>

<details>
<summary><strong>Neo-Grounded Theory: A Methodological Innovation Integrating High-Dimensional Vector Clustering and Multi-Agent Collaboration for Qualitative Research</strong> - Shuide Wen, Beier Ku, Teng Wang, Mingyang Zou, Yang Yang - [[pdf]](https://arxiv.org/pdf/2509.25244)</summary>

**Abstract:** Purpose: Neo Grounded Theory (NGT) integrates vector clustering with multi agent systems to resolve qualitative research's scale depth paradox, enabling analysis of massive datasets in hours while preserving interpretive rigor. Methods: We compared NGT against manual coding and ChatGPT-assisted analysis using 40,000 character Chinese interview transcripts. NGT employs 1536-dimensional embeddings, hierarchical clustering, and parallel agent-based coding. Two experiments tested pure automation versus human guided refinement. Findings: NGT achieved 168-fold speed improvement (3 hours vs 3 weeks), superior quality (0.904 vs 0.883), and 96% cost reduction. Human AI collaboration proved essential: automation alone produced abstract frameworks while human guidance yielded actionable dual pathway theories. The system discovered patterns invisible to manual coding, including identity bifurcation phenomena. Contributions: NGT demonstrates computational objectivity and human interpretation are complementary. Vector representations provide reproducible semantic measurement while preserving meaning's interpretive dimensions. Researchers shift from mechanical coding to theoretical guidance, with AI handling pattern recognition while humans provide creative insight. Implications: Cost reduction from \$50,000 to \$500 democratizes qualitative research, enabling communities to study themselves. Real-time analysis makes qualitative insights contemporaneous with events. The framework shows computational methods can strengthen rather than compromise qualitative research's humanistic commitments.
Keywords: Grounded theory; Vector embeddings; Multi agent systems; Human AI collaboration; Computational qualitative analysis

**arXiv ID:** 2509.25244
</details>

<details>
<summary><strong>RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration</strong> - Xiuyuan Chen, Jian Zhao, Yuchen Yuan, Tianle Zhang, Huilin Zhou, Zheng Zhu, Ping Hu, Linghe Kong, Chi Zhang, Weiran Huang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2509.25271)</summary>

**Abstract:** Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87% improvement in risk identification accuracy compared to the strongest baseline evaluation method.

**arXiv ID:** 2509.25271
</details>

<details>
<summary><strong>ID-RAG: Identity Retrieval-Augmented Generation for Long-Horizon Persona Coherence in Generative Agents</strong> - Daniel Platnick, Mohamed E. Bengueddache, Marjan Alirezaie, Dava J. Newman, Alex ''Sandy'' Pentland, Hossein Rahnama - [[pdf]](https://arxiv.org/pdf/2509.25299)</summary>

**Abstract:** Generative agents powered by language models are increasingly deployed for long-horizon tasks. However, as long-term memory context grows over time, they struggle to maintain coherence. This deficiency leads to critical failures, including identity drift, ignoring established beliefs, and the propagation of hallucinations in multi-agent systems. To mitigate these challenges, this paper introduces Identity Retrieval-Augmented Generation (ID-RAG), a novel mechanism designed to ground an agent's persona and persistent preferences in a dynamic, structured identity model: a knowledge graph of core beliefs, traits, and values. During the agent's decision loop, this model is queried to retrieve relevant identity context, which directly informs action selection. We demonstrate this approach by introducing and implementing a new class of ID-RAG enabled agents called Human-AI Agents (HAis), where the identity model is inspired by the Chronicle structure used in Perspective-Aware AI, a dynamic knowledge graph learned from a real-world entity's digital footprint. In social simulations of a mayoral election, HAis using ID-RAG outperformed baseline agents in long-horizon persona coherence - achieving higher identity recall across all tested models by the fourth timestep - and reduced simulation convergence time by 19% (GPT-4o) and 58% (GPT-4o mini). By treating identity as an explicit, retrievable knowledge structure, ID-RAG offers a foundational approach for developing more temporally coherent, interpretable, and aligned generative agents. Our code is open-source and available at: this https URL.

**arXiv ID:** 2509.25299
</details>

<details>
<summary><strong>ATLAS: Constraints-Aware Multi-Agent Collaboration for Real-World Travel Planning</strong> - Jihye Choi, Jinsung Yoon, Jiefeng Chen, Somesh Jha, Tomas Pfister - [[pdf]](https://arxiv.org/pdf/2509.25586)</summary>

**Abstract:** While Large Language Models (LLMs) have shown remarkable advancements in reasoning and tool use, they often fail to generate optimal, grounded solutions under complex constraints. Real-world travel planning exemplifies these challenges, evaluating agents' abilities to handle constraints that are explicit, implicit, and even evolving based on interactions with dynamic environments and user needs. In this paper, we present ATLAS, a general multi-agent framework designed to effectively handle such complex nature of constraints awareness in real-world travel planning tasks. ATLAS introduces a principled approach to address the fundamental challenges of constraint-aware planning through dedicated mechanisms for dynamic constraint management, iterative plan critique, and adaptive interleaved search. ATLAS demonstrates state-of-the-art performance on the TravelPlanner benchmark, improving the final pass rate from 23.3% to 44.4% over its best alternative. More importantly, our work is the first to demonstrate quantitative effectiveness on real-world travel planning tasks with live information search and multi-turn feedback. In this realistic setting, ATLAS showcases its superior overall planning performance, achieving an 84% final pass rate which significantly outperforms baselines including ReAct (59%) and a monolithic agent (27%).

**arXiv ID:** 2509.25586
</details>

<details>
<summary><strong>AutoLabs: Cognitive Multi-Agent Systems with Self-Correction for Autonomous Chemical Experimentation</strong> - Gihan Panapitiya, Emily Saldanha, Heather Job, Olivia Hess - [[pdf]](https://arxiv.org/pdf/2509.25651)</summary>

**Abstract:** The automation of chemical research through self-driving laboratories (SDLs) promises to accelerate scientific discovery, yet the reliability and granular performance of the underlying AI agents remain critical, under-examined challenges. In this work, we introduce AutoLabs, a self-correcting, multi-agent architecture designed to autonomously translate natural-language instructions into executable protocols for a high-throughput liquid handler. The system engages users in dialogue, decomposes experimental goals into discrete tasks for specialized agents, performs tool-assisted stoichiometric calculations, and iteratively self-corrects its output before generating a hardware-ready file. We present a comprehensive evaluation framework featuring five benchmark experiments of increasing complexity, from simple sample preparation to multi-plate timed syntheses. Through a systematic ablation study of 20 agent configurations, we assess the impact of reasoning capacity, architectural design (single- vs. multi-agent), tool use, and self-correction mechanisms. Our results demonstrate that agent reasoning capacity is the most critical factor for success, reducing quantitative errors in chemical amounts (nRMSE) by over 85% in complex tasks. When combined with a multi-agent architecture and iterative self-correction, AutoLabs achieves near-expert procedural accuracy (F1-score > 0.89) on challenging multi-step syntheses. These findings establish a clear blueprint for developing robust and trustworthy AI partners for autonomous laboratories, highlighting the synergistic effects of modular design, advanced reasoning, and self-correction to ensure both performance and reliability in high-stakes scientific applications. Code: this https URL

**arXiv ID:** 2509.25651
</details>

<details>
<summary><strong>ScheduleMe: Multi-Agent Calendar Assistant</strong> - N. de Silva, S. Perera, K. L. A. A. Nimasha, I. D. S. Fernando, R.K.A.O. Wijerathne - [[pdf]](https://arxiv.org/pdf/2509.25693)</summary>

**Abstract:** Recent advancements in LLMs have contributed to the rise of advanced conversational assistants that can assist with user needs through natural language conversation. This paper presents a ScheduleMe, a multi-agent calendar assistant for users to manage google calendar events in natural language. The system uses a graph-structured coordination mechanism where a central supervisory agent supervises specialized task agents, allowing modularity, conflicts resolution, and context-aware interactions to resolve ambiguities and evaluate user commands. This approach sets an example of how structured reasoning and agent cooperation might convince operators to increase the usability and flexibility of personal calendar assistant tools.

**arXiv ID:** 2509.25693
</details>

<details>
<summary><strong>Evaluating the Use of Large Language Models as Synthetic Social Agents in Social Science Research</strong> - Emma Rose Madden - [[pdf]](https://arxiv.org/pdf/2509.26080)</summary>

**Abstract:** Large Language Models (LLMs) are being increasingly used as synthetic agents in social science, in applications ranging from augmenting survey responses to powering multi-agent simulations. Because strong prediction plus conditioning prompts, token log-probs, and repeated sampling mimic Bayesian workflows, their outputs can be misinterpreted as posterior-like evidence from a coherent model. However, prediction does not equate to probabilism, and accurate points do not imply calibrated uncertainty. This paper outlines cautions that should be taken when interpreting LLM outputs and proposes a pragmatic reframing for the social sciences in which LLMs are used as high-capacity pattern matchers for quasi-predictive interpolation under explicit scope conditions and not as substitutes for probabilistic inference. Practical guardrails such as independent draws, preregistered human baselines, reliability-aware validation, and subgroup calibration, are introduced so that researchers may engage in useful prototyping and forecasting while avoiding category errors.

**arXiv ID:** 2509.26080
</details>

<details>
<summary><strong>SafeEvalAgent: Toward Agentic and Self-Evolving Safety Evaluation of LLMs</strong> - Yixu Wang, Xin Wang, Yang Yao, Xinyuan Li, Yan Teng, Xingjun Ma, Yingchun Wang - [[pdf]](https://arxiv.org/pdf/2509.26100)</summary>

**Abstract:** The rapid integration of Large Language Models (LLMs) into high-stakes domains necessitates reliable safety and compliance evaluation. However, existing static benchmarks are ill-equipped to address the dynamic nature of AI risks and evolving regulations, creating a critical safety gap. This paper introduces a new paradigm of agentic safety evaluation, reframing evaluation as a continuous and self-evolving process rather than a one-time audit. We then propose a novel multi-agent framework SafeEvalAgent, which autonomously ingests unstructured policy documents to generate and perpetually evolve a comprehensive safety benchmark. SafeEvalAgent leverages a synergistic pipeline of specialized agents and incorporates a Self-evolving Evaluation loop, where the system learns from evaluation results to craft progressively more sophisticated and targeted test cases. Our experiments demonstrate the effectiveness of SafeEvalAgent, showing a consistent decline in model safety as the evaluation hardens. For instance, GPT-5's safety rate on the EU AI Act drops from 72.50% to 36.36% over successive iterations. These findings reveal the limitations of static assessments and highlight our framework's ability to uncover deep vulnerabilities missed by traditional methods, underscoring the urgent need for dynamic evaluation ecosystems to ensure the safe and responsible deployment of advanced AI.

**arXiv ID:** 2509.26100
</details>

<details>
<summary><strong>Automatically Generating Web Applications from Requirements Via Multi-Agent Test-Driven Development</strong> - Yuxuan Wan, Tingshuo Liang, Jiakai Xu, Jingyu Xiao, Yintong Huo, Michael R. Lyu - [[pdf]](https://arxiv.org/pdf/2509.25297)</summary>

**Abstract:** Developing full-stack web applications is complex and time-intensive, demanding proficiency across diverse technologies and frameworks. Although recent advances in multimodal large language models (MLLMs) enable automated webpage generation from visual inputs, current solutions remain limited to front-end tasks and fail to deliver fully functional applications. In this work, we introduce TDDev, the first test-driven development (TDD)-enabled LLM-agent framework for end-to-end full-stack web application generation. Given a natural language description or design image, TDDev automatically derives executable test cases, generates front-end and back-end code, simulates user interactions, and iteratively refines the implementation until all requirements are satisfied. Our framework addresses key challenges in full-stack automation, including underspecified user requirements, complex interdependencies among multiple files, and the need for both functional correctness and visual fidelity. Through extensive experiments on diverse application scenarios, TDDev achieves a 14.4% improvement on overall accuracy compared to state-of-the-art baselines, demonstrating its effectiveness in producing reliable, high-quality web applications without requiring manual intervention.

**arXiv ID:** 2509.25297
</details>

<details>
<summary><strong>Heterogeneous Multi-agent Collaboration in UAV-assisted Mobile Crowdsensing Networks</strong> - Xianyang Deng, Wenshuai Liu, Yaru FuB, Qi Zhu - [[pdf]](https://arxiv.org/pdf/2509.25261)</summary>

**Abstract:** Unmanned aerial vehicles (UAVs)-assisted mobile crowdsensing (MCS) has emerged as a promising paradigm for data collection. However, challenges such as spectrum scarcity, device heterogeneity, and user mobility hinder efficient coordination of sensing, communication, and computation. To tackle these issues, we propose a joint optimization framework that integrates time slot partition for sensing, communication, and computation phases, resource allocation, and UAV 3D trajectory planning, aiming to maximize the amount of processed sensing data. The problem is formulated as a non-convex stochastic optimization and further modeled as a partially observable Markov decision process (POMDP) that can be solved by multi-agent deep reinforcement learning (MADRL) algorithm. To overcome the limitations of conventional multi-layer perceptron (MLP) networks, we design a novel MADRL algorithm with hybrid actor network. The newly developed method is based on heterogeneous agent proximal policy optimization (HAPPO), empowered by convolutional neural networks (CNN) for feature extraction and Kolmogorov-Arnold networks (KAN) to capture structured state-action dependencies. Extensive numerical results demonstrate that our proposed method achieves significant improvements in the amount of processed sensing data when compared with other benchmarks.

**arXiv ID:** 2509.25261
</details>

<details>
<summary><strong>Voting or Consensus? Decision-Making in Multi-Agent Debate</strong> - Lars Benedikt Kaesberg, Jonas Becker, Jan Philip Wahle, Terry Ruas, Bela Gipp - [[pdf]](https://arxiv.org/pdf/2502.19130)</summary>

**Abstract:** Much of the success of multi-agent debates depends on carefully choosing the right parameters. The decision-making protocol stands out as it can highly impact final model answers, depending on how decisions are reached. Systematic comparison of decision protocols is difficult because many studies alter multiple discussion parameters beyond the protocol. So far, it has been largely unknown how decision-making influences different tasks. This work systematically evaluates the impact of seven decision protocols (e.g., majority voting, unanimity consensus). We change only one variable at a time - the decision protocol - to analyze how different methods affect the collaboration between agents and measure differences in knowledge and reasoning tasks. Our results show that voting protocols improve performance by 13.2% in reasoning tasks and consensus protocols by 2.8% in knowledge tasks compared to other decision protocols. Increasing the number of agents improves performance, while more discussion rounds before voting reduce it. To improve decision-making by increasing answer diversity, we propose two new methods, All-Agents Drafting (AAD) and Collective Improvement (CI). Our methods improve task performance by up to 3.3% with AAD and up to 7.4% with CI. This work demonstrates the importance of decision-making in multi-agent debates beyond scaling.

**arXiv ID:** 2502.19130
</details>

<details>
<summary><strong>Dynamic Pricing in High-Speed Railways Using Multi-Agent Reinforcement Learning</strong> - Enrique Adrian Villarrubia-Martin, Luis Rodriguez-Benitez, David Muñoz-Valero, Giovanni Montana, Luis Jimenez-Linares - [[pdf]](https://arxiv.org/pdf/2501.08234)</summary>

**Abstract:** This paper addresses a critical challenge in the high-speed passenger railway industry: designing effective dynamic pricing strategies in the context of competing and cooperating operators. To address this, a multi-agent reinforcement learning (MARL) framework based on a non-zero-sum Markov game is proposed, incorporating random utility models to capture passenger decision making. Unlike prior studies in areas such as energy, airlines, and mobile networks, dynamic pricing for railway systems using deep reinforcement learning has received limited attention. A key contribution of this paper is a parametrisable and versatile reinforcement learning simulator designed to model a variety of railway network configurations and demand patterns while enabling realistic, microscopic modelling of user behaviour, called RailPricing-RL. This environment supports the proposed MARL framework, which models heterogeneous agents competing to maximise individual profits while fostering cooperative behaviour to synchronise connecting services. Experimental results validate the framework, demonstrating how user preferences affect MARL performance and how pricing policies influence passenger choices, utility, and overall system dynamics. This study provides a foundation for advancing dynamic pricing strategies in railway systems, aligning profitability with system-wide efficiency, and supporting future research on optimising pricing policies.

**arXiv ID:** 2501.08234
</details>

<details>
<summary><strong>Towards Agentic OS: An LLM Agent Framework for Linux Schedulers</strong> - Yusheng Zheng, Yanpeng Hu, Wei Zhang, Andi Quinn - [[pdf]](https://arxiv.org/pdf/2509.01245)</summary>

**Abstract:** Operating system schedulers suffer from a fundamental semantic gap, where kernel policies fail to understand application-specific needs, leading to suboptimal performance. We introduce SchedCP, the first framework that enables fully autonomous Large Language Model (LLM) agents to safely and efficiently optimize Linux schedulers without human involvement. Our core insight is that the challenge is not merely to apply a better LLM, but to architect a decoupled control plane that separates the AI's role of semantic reasoning ("what to optimize") from the system's role of execution ("how to observe and act"), thereby separating the optimization problem into two stages: goal-inference and policy-synthesis. Implemented as Model Context Protocol(MCP) server, SchedCP provides a stable interface with three key services: a Workload Analysis Engine, an evolving Scheduler Policy Repository, and an Execution Verifier that validates all AI-generated code and configure before deployment with static and dynamic analysis.
We demonstrate this architecture's power with sched-agent, a multi-agent system that autonomously analyzes workloads, synthesizes custom eBPF scheduling policies, and deploys them via the sched\_ext infrastructure. Our evaluation shows that SchedCP achieves up to an 1.79x performance improvement, and a 13x cost reduction compared to naive agentic approaches, all while maintaining high success rate. By bridging the semantic gap, SchedCP democratizes expert-level system optimization and represents a step towards creating truly self-optimizing, application-aware operating systems. The code is open-sourced in this https URL

**arXiv ID:** 2509.01245
</details>

<details>
<summary><strong>Sequence Pathfinder for Multi-Agent Pickup and Delivery in the Warehouse</strong> - Zeyuan Zhao, Chaoran Li, Shao Zhang, Ying Wen - [[pdf]](https://arxiv.org/pdf/2509.23778)</summary>

**Abstract:** Multi-Agent Pickup and Delivery (MAPD) is a challenging extension of Multi-Agent Path Finding (MAPF), where agents are required to sequentially complete tasks with fixed-location pickup and delivery demands. Although learning-based methods have made progress in MAPD, they often perform poorly in warehouse-like environments with narrow pathways and long corridors when relying only on local observations for distributed decision-making. Communication learning can alleviate the lack of global information but introduce high computational complexity due to point-to-point communication. To address this challenge, we formulate MAPF as a sequence modeling problem and prove that path-finding policies under sequence modeling possess order-invariant optimality, ensuring its effectiveness in MAPD. Building on this, we propose the Sequential Pathfinder (SePar), which leverages the Transformer paradigm to achieve implicit information exchange, reducing decision-making complexity from exponential to linear while maintaining efficiency and global awareness. Experiments demonstrate that SePar consistently outperforms existing learning-based methods across various MAPF tasks and their variants, and generalizes well to unseen environments. Furthermore, we highlight the necessity of integrating imitation learning in complex maps like warehouses.

**arXiv ID:** 2509.23778
</details>

<details>
<summary><strong>The Hunger Game Debate: On the Emergence of Over-Competition in Multi-Agent Systems</strong> - Xinbei Ma, Ruotian Ma, Xingyu Chen, Zhengliang Shi, Mengru Wang, Jen-tse Huang, Qu Yang, Wenxuan Wang, Fanghua Ye, Qingxuan Jiang, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Hai Zhao, Zhaopeng Tu, Xiaolong Li, Linus - [[pdf]](https://arxiv.org/pdf/2509.26126)</summary>

**Abstract:** LLM-based multi-agent systems demonstrate great potential for tackling complex problems, but how competition shapes their behavior remains underexplored. This paper investigates the over-competition in multi-agent debate, where agents under extreme pressure exhibit unreliable, harmful behaviors that undermine both collaboration and task performance. To study this phenomenon, we propose HATE, the Hunger Game Debate, a novel experimental framework that simulates debates under a zero-sum competition arena. Our experiments, conducted across a range of LLMs and tasks, reveal that competitive pressure significantly stimulates over-competition behaviors and degrades task performance, causing discussions to derail. We further explore the impact of environmental feedback by adding variants of judges, indicating that objective, task-focused feedback effectively mitigates the over-competition behaviors. We also probe the post-hoc kindness of LLMs and form a leaderboard to characterize top LLMs, providing insights for understanding and governing the emergent social dynamics of AI community.

**arXiv ID:** 2509.26126
</details>

<details>
<summary><strong>CreAgentive: An Agent Workflow Driven Multi-Category Creative Generation Engine</strong> - Yuyang Cheng, Linyue Cai, Changwei Peng, Yumiao Xu, Rongfang Bie, Yong Zhao - [[pdf]](https://arxiv.org/pdf/2509.26461)</summary>

**Abstract:** We present CreAgentive, an agent workflow driven multi-category creative generation engine that addresses four key limitations of contemporary large language models in writing stories, drama and other categories of creatives: restricted genre diversity, insufficient output length, weak narrative coherence, and inability to enforce complex structural constructs. At its core, CreAgentive employs a Story Prototype, which is a genre-agnostic, knowledge graph-based narrative representation that decouples story logic from stylistic realization by encoding characters, events, and environments as semantic triples. CreAgentive engages a three-stage agent workflow that comprises: an Initialization Stage that constructs a user-specified narrative skeleton; a Generation Stage in which long- and short-term objectives guide multi-agent dialogues to instantiate the Story Prototype; a Writing Stage that leverages this prototype to produce multi-genre text with advanced structures such as retrospection and foreshadowing. This architecture reduces storage redundancy and overcomes the typical bottlenecks of long-form generation. In extensive experiments, CreAgentive generates thousands of chapters with stable quality and low cost (less than $1 per 100 chapters) using a general-purpose backbone model. To evaluate performance, we define a two-dimensional framework with 10 narrative indicators measuring both quality and length. Results show that CreAgentive consistently outperforms strong baselines and achieves robust performance across diverse genres, approaching the quality of human-authored novels.

**arXiv ID:** 2509.26461
</details>

<details>
<summary><strong>SoMi-ToM: Evaluating Multi-Perspective Theory of Mind in Embodied Social Interactions</strong> - Xianzhe Fan, Xuhui Zhou, Chuanyang Jin, Kolby Nottingham, Hao Zhu, Maarten Sap - [[pdf]](https://arxiv.org/pdf/2506.23046)</summary>

**Abstract:** Humans continuously infer the states, goals, and behaviors of others by perceiving their surroundings in dynamic, real-world social interactions. However, most Theory of Mind (ToM) benchmarks only evaluate static, text-based scenarios, which have a significant gap compared to real interactions. We propose the SoMi-ToM benchmark, designed to evaluate multi-perspective ToM in embodied multi-agent complex social interactions. This benchmark is based on rich multimodal interaction data generated by the interaction environment SoMi, covering diverse crafting goals and social relationships. Our framework supports multi-level evaluation: (1) first-person evaluation provides multimodal (visual, dialogue, action, etc.) input from a first-person perspective during a task for real-time state inference, (2) third-person evaluation provides complete third-person perspective video and text records after a task for goal and behavior inference. This evaluation method allows for a more comprehensive examination of a model's ToM capabilities from both the subjective immediate experience and the objective global observation. We constructed a challenging dataset containing 35 third-person perspective videos, 363 first-person perspective images, and 1225 expert-annotated multiple-choice questions (three options). On this dataset, we systematically evaluated the performance of human subjects and several state-of-the-art large vision-language models (LVLMs). The results show that LVLMs perform significantly worse than humans on SoMi-ToM: the average accuracy gap between humans and models is 40.1% in first-person evaluation and 26.4% in third-person evaluation. This indicates that future LVLMs need to further improve their ToM capabilities in embodied, complex social interactions.

**arXiv ID:** 2506.23046
</details>

<details>
<summary><strong>Collaborative Gym: A Framework for Enabling and Evaluating Human-Agent Collaboration</strong> - Yijia Shao, Vinay Samuel, Yucheng Jiang, John Yang, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2412.15701)</summary>

**Abstract:** Recent advancements in language models (LMs) have sparked growing interest in developing LM agents. While fully autonomous agents could excel in many scenarios, numerous use cases inherently require them to collaborate with humans due to humans' latent preferences, domain expertise, or need for control. To facilitate the study of human-agent collaboration, we present Collaborative Gym (Co-Gym), a general framework enabling asynchronous, tripartite interaction among agents, humans, and task environments. We instantiate Co-Gym with three representative tasks in both simulated and real-world conditions, and propose an evaluation framework that assesses both the collaboration outcomes and processes. Our findings reveal that collaborative agents consistently outperform their fully autonomous counterparts in task performance within those delivered cases, achieving win rates of 86% in Travel Planning, 74% in Tabular Analysis, and 66% in Related Work when evaluated by real users. However, our study also highlights significant challenges in developing collaborative agents, requiring advancements in core aspects of intelligence -- communication capabilities, situational awareness, and balancing autonomy and human control.

**arXiv ID:** 2412.15701
</details>

<details>
<summary><strong>A Survey on Code Generation with LLM-based Agents</strong> - Yihong Dong, Xue Jiang, Jiaru Qian, Tian Wang, Kechi Zhang, Zhi Jin, Ge Li - [[pdf]](https://arxiv.org/pdf/2508.00083)</summary>

**Abstract:** Code generation agents powered by large language models (LLMs) are revolutionizing the software development paradigm. Distinct from previous code generation techniques, code generation agents are characterized by three core features. 1) Autonomy: the ability to independently manage the entire workflow, from task decomposition to coding and debugging. 2) Expanded task scope: capabilities that extend beyond generating code snippets to encompass the full software development lifecycle (SDLC). 3) Enhancement of engineering practicality: a shift in research emphasis from algorithmic innovation toward practical engineering challenges, such as system reliability, process management, and tool integration. This domain has recently witnessed rapid development and an explosion in research, demonstrating significant application potential. This paper presents a systematic survey of the field of LLM-based code generation agents. We trace the technology's developmental trajectory from its inception and systematically categorize its core techniques, including both single-agent and multi-agent architectures. Furthermore, this survey details the applications of LLM-based agents across the full SDLC, summarizes mainstream evaluation benchmarks and metrics, and catalogs representative tools. Finally, by analyzing the primary challenges, we identify and propose several foundational, long-term research directions for the future work of the field.

**arXiv ID:** 2508.00083
</details>

<details>
<summary><strong>MASLegalBench: Benchmarking Multi-Agent Systems in Deductive Legal Reasoning</strong> - Huihao Jing, Wenbin Hu, Hongyu Luo, Jianhui Yang, Wei Fan, Haoran Li, Yangqiu Song - [[pdf]](https://arxiv.org/pdf/2509.24922)</summary>

**Abstract:** Multi-agent systems (MAS), leveraging the remarkable capabilities of Large Language Models (LLMs), show great potential in addressing complex tasks. In this context, integrating MAS with legal tasks is a crucial step. While previous studies have developed legal benchmarks for LLM agents, none are specifically designed to consider the unique advantages of MAS, such as task decomposition, agent specialization, and flexible training. In fact, the lack of evaluation methods limits the potential of MAS in the legal domain. To address this gap, we propose MASLegalBench, a legal benchmark tailored for MAS and designed with a deductive reasoning approach. Our benchmark uses GDPR as the application scenario, encompassing extensive background knowledge and covering complex reasoning processes that effectively reflect the intricacies of real-world legal situations. Furthermore, we manually design various role-based MAS and conduct extensive experiments using different state-of-the-art LLMs. Our results highlight the strengths, limitations, and potential areas for improvement of existing models and MAS architectures.

**arXiv ID:** 2509.24922
</details>

<details>
<summary><strong>Conflict-Based Search and Prioritized Planning for Multi-Agent Path Finding Among Movable Obstacles</strong> - Shaoli Hu, Shizhe Zhao, Zhongqiang Ren - [[pdf]](https://arxiv.org/pdf/2509.26050)</summary>

**Abstract:** This paper investigates Multi-Agent Path Finding Among Movable Obstacles (M-PAMO), which seeks collision-free paths for multiple agents from their start to goal locations among static and movable obstacles. M-PAMO arises in logistics and warehouses where mobile robots are among unexpected movable objects. Although Multi-Agent Path Finding (MAPF) and single-agent Path planning Among Movable Obstacles (PAMO) were both studied, M-PAMO remains under-explored. Movable obstacles lead to new fundamental challenges as the state space, which includes both agents and movable obstacles, grows exponentially with respect to the number of agents and movable obstacles. In particular, movable obstacles often closely couple agents together spatially and temporally. This paper makes a first attempt to adapt and fuse the popular Conflict-Based Search (CBS) and Prioritized Planning (PP) for MAPF, and a recent single-agent PAMO planner called PAMO*, together to address M-PAMO. We compare their performance with up to 20 agents and hundreds of movable obstacles, and show the pros and cons of these approaches.

**arXiv ID:** 2509.26050
</details>

<details>
<summary><strong>Robot Conga: A Leader-Follower Walking Approach to Sequential Path Following in Multi-Agent Systems</strong> - Pranav Tiwari, Soumyodipta Nath - [[pdf]](https://arxiv.org/pdf/2509.16482)</summary>

**Abstract:** Coordinated path following in multi-agent systems is a key challenge in robotics, with applications in automated logistics, surveillance, and collaborative exploration. Traditional formation control techniques often rely on time-parameterized trajectories and path integrals, which can result in synchronization issues and rigid behavior. In this work, we address the problem of sequential path following, where agents maintain fixed spatial separation along a common trajectory, guided by a leader under centralized control. We introduce Robot Conga, a leader-follower control strategy that updates each agent's desired state based on the leader's spatial displacement rather than time, assuming access to a global position reference, an assumption valid in indoor environments equipped with motion capture, vision-based tracking, or UWB localization systems. The algorithm was validated in simulation using both TurtleBot3 and quadruped (Laikago) robots. Results demonstrate accurate trajectory tracking, stable inter-agent spacing, and fast convergence, with all agents aligning within 250 time steps (approx. 0.25 seconds) in the quadruped case, and almost instantaneously in the TurtleBot3 implementation.

**arXiv ID:** 2509.16482
</details>

<details>
<summary><strong>InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios</strong> - Chenglin Yu, Yang Yu, Songmiao Wang, Yucheng Wang, Yifan Yang, Jinjia Li, Ming Li, Hongxia Yang - [[pdf]](https://arxiv.org/pdf/2509.22502)</summary>

**Abstract:** Large Language Model (LLM) agents have demonstrated remarkable capabilities in organizing and executing complex tasks, and many such agents are now widely used in various application scenarios. However, developing these agents requires carefully designed workflows, carefully crafted prompts, and iterative tuning, which requires LLM techniques and domain-specific expertise. These hand-crafted limitations hinder the scalability and cost-effectiveness of LLM agents across a wide range of industries. To address these challenges, we propose \textbf{InfiAgent}, a Pyramid-like DAG-based Multi-Agent Framework that can be applied to \textbf{infi}nite scenarios, which introduces several key innovations: a generalized "agent-as-a-tool" mechanism that automatically decomposes complex agents into hierarchical multi-agent systems; a dual-audit mechanism that ensures the quality and stability of task completion; an agent routing function that enables efficient task-agent matching; and an agent self-evolution mechanism that autonomously restructures the agent DAG based on new tasks, poor performance, or optimization opportunities. Furthermore, InfiAgent's atomic task design supports agent parallelism, significantly improving execution efficiency. This framework evolves into a versatile pyramid-like multi-agent system capable of solving a wide range of problems. Evaluations on multiple benchmarks demonstrate that InfiAgent achieves 9.9\% higher performance compared to ADAS (similar auto-generated agent framework), while a case study of the AI research assistant InfiHelper shows that it generates scientific papers that have received recognition from human reviewers at top-tier IEEE conferences.

**arXiv ID:** 2509.22502
</details>

</details>

<details open>
<summary><h2>Other Agent Research (11 papers)</h2></summary>

<details>
<summary><strong>Toward Causal-Visual Programming: Enhancing Agentic Reasoning in Low-Code Environments</strong> - Jiexi Xu, Jiaqi Liu, Ran Tong, Su Liu - [[pdf]](https://arxiv.org/pdf/2509.25282)</summary>

**Abstract:** Large language model (LLM) agents are increasingly capable of orchestrating complex tasks in low-code environments. However, these agents often exhibit hallucinations and logical inconsistencies because their inherent reasoning mechanisms rely on probabilistic associations rather than genuine causal understanding. This paper introduces a new programming paradigm: Causal-Visual Programming (CVP), designed to address this fundamental issue by explicitly introducing causal structures into the workflow design. CVP allows users to define a simple "world model" for workflow modules through an intuitive low-code interface, effectively creating a Directed Acyclic Graph (DAG) that explicitly defines the causal relationships between modules. This causal graph acts as a crucial constraint during the agent's reasoning process, anchoring its decisions to a user-defined causal structure and significantly reducing logical errors and hallucinations by preventing reliance on spurious correlations. To validate the effectiveness of CVP, we designed a synthetic experiment that simulates a common real-world problem: a distribution shift between the training and test environments. Our results show that a causally anchored model maintained stable accuracy in the face of this shift, whereas a purely associative baseline model that relied on probabilistic correlations experienced a significant performance drop. The primary contributions of this study are: a formal definition of causal structures for workflow modules; the proposal and implementation of a CVP framework that anchors agent reasoning to a user-defined causal graph; and empirical evidence demonstrating the framework's effectiveness in enhancing agent robustness and reducing errors caused by causal confusion in dynamic environments. CVP offers a viable path toward building more interpretable, reliable, and trustworthy AI agents.

**arXiv ID:** 2509.25282
</details>

<details>
<summary><strong>Message passing-based inference in an autoregressive active inference agent</strong> - Wouter M. Kouw, Tim N. Nisslbeck, Wouter L.N. Nuijten - [[pdf]](https://arxiv.org/pdf/2509.25482)</summary>

**Abstract:** We present the design of an autoregressive active inference agent in the form of message passing on a factor graph. Expected free energy is derived and distributed across a planning graph. The proposed agent is validated on a robot navigation task, demonstrating exploration and exploitation in a continuous-valued observation space with bounded continuous-valued actions. Compared to a classical optimal controller, the agent modulates action based on predictive uncertainty, arriving later but with a better model of the robot's dynamics.

**arXiv ID:** 2509.25482
</details>

<details>
<summary><strong>Beyond the Algorithm: A Field Guide to Deploying AI Agents in Clinical Practice</strong> - Jack Gallifant, Katherine C. Kellogg, Matt Butler, Amanda Centi, Patrick F. Doyle, Sayon Dutta, Joyce Guo, Matthew J. Hadfield, Esther H. Kim, David E. Kozono, Hugo JWL Aerts, Adam B. Landman, Raymond H. Mak, Rebecca G. Mishuris, Tanna L. Nelson, Guergana K. Savova, Elad Sharon, Benjamin C. Silverman, Umit Topaloglu, Jeremy L. Warner, Danielle S. Bitterman - [[pdf]](https://arxiv.org/pdf/2509.26153)</summary>

**Abstract:** Large language models (LLMs) integrated into agent-driven workflows hold immense promise for healthcare, yet a significant gap exists between their potential and practical implementation within clinical settings. To address this, we present a practitioner-oriented field manual for deploying generative agents that use electronic health record (EHR) data. This guide is informed by our experience deploying the "irAE-Agent", an automated system to detect immune-related adverse events from clinical notes at Mass General Brigham, and by structured interviews with 20 clinicians, engineers, and informatics leaders involved in the project. Our analysis reveals a critical misalignment in clinical AI development: less than 20% of our effort was dedicated to prompt engineering and model development, while over 80% was consumed by the sociotechnical work of implementation. We distill this effort into five "heavy lifts": data integration, model validation, ensuring economic value, managing system drift, and governance. By providing actionable solutions for each of these challenges, this field manual shifts the focus from algorithmic development to the essential infrastructure and implementation work required to bridge the "valley of death" and successfully translate generative AI from pilot projects into routine clinical care.

**arXiv ID:** 2509.26153
</details>

<details>
<summary><strong>Devstral: Fine-tuning Language Models for Coding Agent Applications</strong> - Abhinav Rastogi, Adam Yang, Albert Q. Jiang, Alexander H. Liu, Alexandre Sablayrolles, Amélie Héliou, Amélie Martin, Anmol Agarwal, Andy Ehrenberg, Andy Lo, Antoine Roux, Arthur Darcet, Arthur Mensch, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Clément Denoix, Corentin Barreau, Darius Dabert Devon Mizelle, Diego de las Casas, Elliot Chane-Sane, Emilien Fugier, Emma Bou Hanna, Gabrielle Berrada, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Graham Neubig, Guillaume Lample, Guillaume Martin, Himanshu Jaju, Jan Ludziejewski, Jason Rute, Jean-Malo Delignon, JeanHadrien Chabran, Joachim Studnia, Joep Barmentlo, Jonas Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Karmesh Yadav, Kartik Khandelwal, Khyathi Raghavi Chandu, Kush Jain, Lélio Renard Lavaud, Léonard Blier, Lingxiao Zhao, Louis Martin, Lucile Saulnier, Luyu Gao, Marie Pellat, Mathilde Guillaumin, Mathis Felardos, Matthieu Dinot, Maxime Darrin, Maximilian Augustin, Mickaël Seznec, Neha Gupta, Nikhil Raghuraman, Olivier Duchenne, Patricia Wang, Patrick von Platen, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Rémi Delacourt, Roman Soletskyi, Romain Sauvestre, Sagar Vaze, Sanchit Gandhi, Sandeep Subramanian, Shashwat Dalal, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Teven Le Scao, Thibaut Lavril, Thibault Schueller, Thomas Foubert, Thomas Robert, Thomas Wang, Timothée Lacroix, Tom Bewley, Valeriia Nemychnikova, Victor Paltz, Virgile Richard, Wen-Ding Li, William Marshall, Xingyao Wang - [[pdf]](https://arxiv.org/pdf/2509.25193)</summary>

**Abstract:** We introduce Devstral-Small, a lightweight open source model for code agents with the best performance among models below 100B size. In this technical report, we give an overview of how we design and develop a model and craft specializations in agentic software development. The resulting model, Devstral-Small is a small 24B model, fast and easy to serve. Despite its size, Devstral-Small still attains competitive performance compared to models more than an order of magnitude larger.

**arXiv ID:** 2509.25193
</details>

<details>
<summary><strong>An Agent-Based Simulation of Ageing Societies: Accessibility and Care Dynamics in Remote Areas</strong> - Roberto garrone - [[pdf]](https://arxiv.org/pdf/2509.26496)</summary>

**Abstract:** This paper presents an agent-based simulation of accessibility and care dynamics in ageing societies, applied to the Italian inner area of Premeno (VB). The model integrates census and municipal data, drone-derived elevation models, GIS road networks, and survey-based caregiving information to generate synthetic populations of older adults and their caregivers. Agents are organized into dyads with socio-economic and mobility attributes, enabling the simulation of both micro-scale accessibility and meso-scale caregiving outcomes. Two scenarios are compared: a baseline and an alternative involving the relocation of healthcare services. Key indicators include caregiver effort, overwhelmed caregivers, walkability, and unmet hours of care. Findings show that while relocation improves walkability locally, it increases unmet care hours due to detours and reduced proximity. Household income emerges as the primary driver of caregiver burden, with accessibility shaped by interactions between financial and mobility resources. Results highlight the need for interventions tailored to context-specific constraints in remote ageing communities.

**arXiv ID:** 2509.26496
</details>

<details>
<summary><strong>OpenID Connect for Agents (OIDC-A) 1.0: A Standard Extension for LLM-Based Agent Identity and Authorization</strong> - Subramanya Nagabhushanaradhya - [[pdf]](https://arxiv.org/pdf/2509.25974)</summary>

**Abstract:** OpenID Connect for Agents (OIDC-A) 1.0 is an extension to OpenID Connect Core 1.0 that provides a comprehensive framework for representing, authenticating, and authorizing LLM-based agents within the OAuth 2.0 ecosystem. As autonomous AI agents become increasingly prevalent in digital systems, there is a critical need for standardized protocols to establish agent identity, verify agent attestation, represent delegation chains, and enable fine-grained authorization based on agent attributes. This specification defines standard claims, endpoints, and protocols that address these requirements while maintaining compatibility with existing OAuth 2.0 and OpenID Connect infrastructure. The proposed framework introduces mechanisms for agent identity representation, delegation chain validation, attestation verification, and capability-based authorization, providing a foundation for secure and trustworthy agent-to-service interactions in modern distributed systems.

**arXiv ID:** 2509.25974
</details>

<details>
<summary><strong>RE-Searcher: Robust Agentic Search with Goal-oriented Planning and Self-reflection</strong> - Daocheng Fu, Jianbiao Mei, Licheng Wen, Xuemeng Yang, Cheng Yang, Rong Wu, Tao Hu, Siqi Li, Yufan Shen, Xinyu Cai, Pinlong Cai, Botian Shi, Yong Liu, Yu Qiao - [[pdf]](https://arxiv.org/pdf/2509.26048)</summary>

**Abstract:** Large language models (LLMs) excel at knowledge-intensive question answering and reasoning, yet their real-world deployment remains constrained by knowledge cutoff, hallucination, and limited interaction modalities. Augmenting LLMs with external search tools helps alleviate these issues, but it also exposes agents to a complex search environment in which small, plausible variations in query formulation can steer reasoning into unproductive trajectories and amplify errors. We present a systematic analysis that quantifies how environmental complexity induces fragile search behaviors and, in turn, degrades overall performance. To address this challenge, we propose a simple yet effective approach to instantiate a search agent, RE-Searcher. During search, RE-Searcher explicitly articulates a concrete search goal and subsequently reflects on whether the retrieved evidence satisfies that goal. This combination of goal-oriented planning and self-reflection enables RE-Searcher to resist spurious cues in complex search environments and perform robust search. Extensive experiments show that our method improves search accuracy and achieves state-of-the-art results. Perturbation studies further demonstrate substantial resilience to noisy or misleading external signals, mitigating the fragility of the search process. We believe these findings offer practical guidance for integrating LLM-powered agents into more complex interactive environments and enabling more autonomous decision-making.

**arXiv ID:** 2509.26048
</details>

<details>
<summary><strong>Structured Agent Distillation for Large Language Model</strong> - Jun Liu, Zhenglun Kong, Peiyan Dong, Changdi Yang, Tianqi Li, Hao Tang, Geng Yuan, Wei Niu, Wenbin Zhang, Pu Zhao, Xue Lin, Dong Huang, Yanzhi Wang - [[pdf]](https://arxiv.org/pdf/2505.13820)</summary>

**Abstract:** Large language models (LLMs) exhibit strong capabilities as decision-making agents by interleaving reasoning and actions, as seen in ReAct-style frameworks. Yet, their practical deployment is constrained by high inference costs and large model sizes. We propose Structured Agent Distillation, a framework that compresses large LLM-based agents into smaller student models while preserving both reasoning fidelity and action consistency. Unlike standard token-level distillation, our method segments trajectories into {[REASON]} and {[ACT]} spans, applying segment-specific losses to align each component with the teacher's behavior. This structure-aware supervision enables compact agents to better replicate the teacher's decision process. Experiments on ALFWorld, HotPotQA-ReAct, and WebShop show that our approach consistently outperforms token-level and imitation learning baselines, achieving significant compression with minimal performance drop. Scaling and ablation results further highlight the importance of span-level alignment for efficient and deployable agents.

**arXiv ID:** 2505.13820
</details>

<details>
<summary><strong>PAME-AI: Patient Messaging Creation and Optimization using Agentic AI</strong> - Junjie Luo, Yihong Guo, Anqi Liu, Ritu Agarwal, Gordon Gao - [[pdf]](https://arxiv.org/pdf/2509.24263)</summary>

**Abstract:** Messaging patients is a critical part of healthcare communication, helping to improve things like medication adherence and healthy behaviors. However, traditional mobile message design has significant limitations due to its inability to explore the high-dimensional design space. We develop PAME-AI, a novel approach for Patient Messaging Creation and Optimization using Agentic AI. Built on the Data-Information-Knowledge-Wisdom (DIKW) hierarchy, PAME-AI offers a structured framework to move from raw data to actionable insights for high-performance messaging design. PAME-AI is composed of a system of specialized computational agents that progressively transform raw experimental data into actionable message design strategies. We demonstrate our approach's effectiveness through a two-stage experiment, comprising of 444,691 patient encounters in Stage 1 and 74,908 in Stage 2. The best-performing generated message achieved 68.76% engagement compared to the 61.27% baseline, representing a 12.2% relative improvement in click-through rates. This agentic architecture enables parallel processing, hypothesis validation, and continuous learning, making it particularly suitable for large-scale healthcare communication optimization.

**arXiv ID:** 2509.24263
</details>

<details>
<summary><strong>Towards Intuitive Human-Robot Interaction through Embodied Gesture-Driven Control with Woven Tactile Skins</strong> - ChunPing Lam, Xiangjia Chen, Chenming Wu, Hao Chen, Binzhi Sun, Guoxin Fang, Charlie C.L. Wang, Chengkai Dai, Yeung Yam - [[pdf]](https://arxiv.org/pdf/2509.25951)</summary>

**Abstract:** This paper presents a novel human-robot interaction (HRI) framework that enables intuitive gesture-driven control through a capacitance-based woven tactile skin. Unlike conventional interfaces that rely on panels or handheld devices, the woven tactile skin integrates seamlessly with curved robot surfaces, enabling embodied interaction and narrowing the gap between human intent and robot response. Its woven design combines fabric-like flexibility with structural stability and dense multi-channel sensing through the interlaced conductive threads. Building on this capability, we define a gesture-action mapping of 14 single- and multi-touch gestures that cover representative robot commands, including task-space motion and auxiliary functions. A lightweight convolution-transformer model designed for gesture recognition in real time achieves an accuracy of near-100%, outperforming prior baseline approaches. Experiments on robot arm tasks, including pick-and-place and pouring, demonstrate that our system reduces task completion time by up to 57% compared with keyboard panels and teach pendants. Overall, our proposed framework demonstrates a practical pathway toward more natural and efficient embodied HRI.

**arXiv ID:** 2509.25951
</details>

<details>
<summary><strong>SDA-PLANNER: State-Dependency Aware Adaptive Planner for Embodied Task Planning</strong> - Zichao Shen, Chen Gao, Jiaqi Yuan, Tianchen Zhu, Xingcheng Fu, Qingyun Sun - [[pdf]](https://arxiv.org/pdf/2509.26375)</summary>

**Abstract:** Embodied task planning requires agents to produce executable actions in a close-loop manner within the environment. With progressively improving capabilities of LLMs in task decomposition, planning, and generalization, current embodied task planning methods adopt LLM-based this http URL, existing LLM-based planners remain limited in three aspects, i.e., fixed planning paradigms, lack of action sequence constraints, and error-agnostic. In this work, we propose SDA-PLANNER, enabling an adaptive planning paradigm, state-dependency aware and error-aware mechanisms for comprehensive embodied task planning. Specifically, SDA-PLANNER introduces a State-Dependency Graph to explicitly model action preconditions and effects, guiding the dynamic revision. To handle execution error, it employs an error-adaptive replanning strategy consisting of Error Backtrack and Diagnosis and Adaptive Action SubTree Generation, which locally reconstructs the affected portion of the plan based on the current environment state. Experiments demonstrate that SDA-PLANNER consistently outperforms baselines in success rate and goal completion, particularly under diverse error conditions.

**arXiv ID:** 2509.26375
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (40 papers)</h2></summary>

<details>
<summary><strong>DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search</strong> - Fang Wu, Weihao Xuan, Heli Qi, Ximing Lu, Aaron Tu, Li Erran Li, Yejin ChoiRetry - [[pdf]](https://arxiv.org/pdf/2509.25454)</summary>

**Abstract:** Although RLVR has become an essential component for developing advanced reasoning skills in LLMs, contemporary studies have documented training plateaus that emerge following thousands of optimization steps, demonstrating notable decreases in performance gains despite increased computational investment. This limitation stems from the sparse exploration patterns inherent in current RLVR practices, where models rely on limited rollouts that often miss critical reasoning paths and fail to provide systematic coverage of the solution space. We present DeepSearch, a framework that integrates Monte Carlo Tree Search directly into RLVR training. In contrast to existing methods that rely on tree search only at inference, DeepSearch embeds structured search into the training loop, enabling systematic exploration and fine-grained credit assignment across reasoning steps. Through training-time exploration, DeepSearch addresses the fundamental bottleneck of insufficient exploration, which leads to diminishing performance improvements over prolonged training steps. Our contributions include: (1) a global frontier selection strategy that prioritizes promising nodes across the search tree, (2) selection with entropy-based guidance that identifies confident paths for supervision, and (3) adaptive replay buffer training with solution caching for efficiency. Experiments on mathematical reasoning benchmarks show that DeepSearch achieves 62.95% average accuracy and establishes a new state-of-the-art for 1.5B reasoning models - using 5.7x fewer GPU hours than extended training approaches. These results highlight the importance of strategic exploration over brute-force scaling and demonstrate the promise of algorithmic innovation for advancing RLVR methodologies. DeepSearch establishes a new direction for scaling reasoning capabilities through systematic search rather than prolonged computation.

**arXiv ID:** 2509.25454
</details>

<details>
<summary><strong>Hybrid Reward Normalization for Process-supervised Non-verifiable Agentic Tasks</strong> - Peiran Xu, Zhuohao Li, Xiaoying Xing, Guannan Zhang, Debiao Li, Kunyu Shi - [[pdf]](https://arxiv.org/pdf/2509.25598)</summary>

**Abstract:** Large Language Models (LLMs) increasingly rely on external tools such as search engines to solve complex agentic tasks that require reasoning and external knowledge retrieval. Recently, reinforcement learning with verifiable rewards (RLVR) has demonstrated its effectiveness in advancing capabilities of LLMs by rewarding the final answers via outcome rewards. While straightforward to supervise, outcome rewards only provide sparse signals and delayed feedback, which limits their effectiveness on long trajectories. Process rewards address this by evaluating intermediate steps, providing fine-grained supervision and encouraging grounded problem solving. However, it is notoriously hard to annotate step-wise labels, especially in non-verifiable process without "golden" answers. Furthermore, step-wise judgment requires the balance between local quality with contribution to the final outcome, as optimizing towards higher process reward may not always align with better final outcomes. To address the above challenges, we introduce Principle Process Reward (PPR), an RL approach that unifies principled step-level assessment and outcome verification. We train a principle-based reward model to improve the transparency and reliability of process evaluation, and further introduce a Reward Normalization (ReNorm) strategy to calibrate outcome and process rewards. Experiment results show that PPR achieves state-of-the-art performance across a wide range of benchmarks, demonstrating its impressive robustness and generalization. Our code and model collection is available in this link.

**arXiv ID:** 2509.25598
</details>

<details>
<summary><strong>A Framework for Studying AI Agent Behavior: Evidence from Consumer Choice Experiments</strong> - Manuel Cherep, Chengtian Ma, Abigail Xu, Maya Shaked, Pattie Maes, Nikhil Singh - [[pdf]](https://arxiv.org/pdf/2509.25609)</summary>

**Abstract:** Environments built for people are increasingly operated by a new class of economic actors: LLM-powered software agents making decisions on our behalf. These decisions range from our purchases to travel plans to medical treatment selection. Current evaluations of these agents largely focus on task competence, but we argue for a deeper assessment: how these agents choose when faced with realistic decisions. We introduce ABxLab, a framework for systematically probing agentic choice through controlled manipulations of option attributes and persuasive cues. We apply this to a realistic web-based shopping environment, where we vary prices, ratings, and psychological nudges, all of which are factors long known to shape human choice. We find that agent decisions shift predictably and substantially in response, revealing that agents are strongly biased choosers even without being subject to the cognitive constraints that shape human biases. This susceptibility reveals both risk and opportunity: risk, because agentic consumers may inherit and amplify human biases; opportunity, because consumer choice provides a powerful testbed for a behavioral science of AI agents, just as it has for the study of human behavior. We release our framework as an open benchmark for rigorous, scalable evaluation of agent decision-making.

**arXiv ID:** 2509.25609
</details>

<details>
<summary><strong>Cooperative Autonomous Driving in Diverse Behavioral Traffic: A Heterogeneous Graph Reinforcement Learning Approach</strong> - Qi Liu, Xueyuan Li, Zirui Li, Juhui Gim - [[pdf]](https://arxiv.org/pdf/2509.25751)</summary>

**Abstract:** Navigating heterogeneous traffic environments with diverse driving styles poses a significant challenge for autonomous vehicles (AVs) due to their inherent complexity and dynamic interactions. This paper addresses this challenge by proposing a heterogeneous graph reinforcement learning (GRL) framework enhanced with an expert system to improve AV decision-making performance. Initially, a heterogeneous graph representation is introduced to capture the intricate interactions among vehicles. Then, a heterogeneous graph neural network with an expert model (HGNN-EM) is proposed to effectively encode diverse vehicle features and produce driving instructions informed by domain-specific knowledge. Moreover, the double deep Q-learning (DDQN) algorithm is utilized to train the decision-making model. A case study on a typical four-way intersection, involving various driving styles of human vehicles (HVs), demonstrates that the proposed method has superior performance over several baselines regarding safety, efficiency, stability, and convergence rate, all while maintaining favorable real-time performance.

**arXiv ID:** 2509.25751
</details>

<details>
<summary><strong>Planner-R1: Reward Shaping Enables Efficient Agentic RL with Smaller LLMs</strong> - Siyu Zhu, Yanbin Jiang, Hejian Sang, Shao Tang, Qingquan Song, Biao He, Rohit Jain, Zhipeng Wang, Alborz Geramifard - [[pdf]](https://arxiv.org/pdf/2509.25779)</summary>

**Abstract:** We investigated Agentic RL with large language models on the \textsc{TravelPlanner} benchmark. Our approach, \textsc{Planner-R1}, achieved a \textbf{56.9\%} final-pass rate with only 180 training queries, a $2.7\times$ improvement over GPT-5's $21.2\%$ baseline and the strongest agentic result on the public leaderboard. A central finding was that smaller models (8B) were highly responsive to reward shaping: with dense process-level signals, they reached competitive performance while being $3.5\times$ more compute-efficient and $1.5\times$ more memory-efficient than 32B models. Larger models were more robust under sparse rewards but exhibited smaller relative gains from shaping and higher variance across runs. While curriculum learning offered no significant benefit, shaped rewards consistently amplified learning dynamics, making 8B models the most efficient setting for agentic RL. Crucially, these gains did not come at the cost of overfitting: fine-tuned models mostly maintained or exceeded baseline performance on out-of-domain tasks, including \textsc{Multi-IF}, \textsc{NaturalPlan}, and $\tau$-\textsc{Bench}. These results establish reward shaping as a decisive lever for scaling agentic RL, highlight the competitive strength of smaller models, and demonstrate that efficiency can be achieved without sacrificing generalization.

**arXiv ID:** 2509.25779
</details>

<details>
<summary><strong>KIRETT: Smart Integration of Vital Signs Data for Intelligent Decision Support in Rescue Scenarios</strong> - Mubaris Nadeem, Johannes Zenkert, Christian Weber, Lisa Bender, Madjid Fathi - [[pdf]](https://arxiv.org/pdf/2509.25923)</summary>

**Abstract:** The integration of vital signs in healthcare has witnessed a steady rise, promising health professionals to assist in their daily tasks to improve patient treatment. In life-threatening situations, like rescue operations, crucial decisions need to be made in the shortest possible amount of time to ensure that excellent treatment is provided during life-saving measurements. The integration of vital signs in the treatment holds the potential to improve time utilization for rescuers in such critical situations. They furthermore serve to support health professionals during the treatment with useful information and suggestions. To achieve such a goal, the KIRETT project serves to provide treatment recommendations and situation detection, combined on a wrist-worn wearable for rescue this http URL paper aims to present the significant role of vital signs in the improvement of decision-making during rescue operations and show their impact on health professionals and patients in need.

**arXiv ID:** 2509.25923
</details>

<details>
<summary><strong>RoRecomp: Enhancing Reasoning Efficiency via Rollout Response Recomposition in Reinforcement Learning</strong> - Gang Li, Yulei Qin, Xiaoyu Tan, Dingkang Yang, Yuchen Shi, Zihan Xu, Xiang Li, Xing Sun, Ke Li - [[pdf]](https://arxiv.org/pdf/2509.25958)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has proven effective in eliciting complex reasoning in large language models (LLMs). However, standard RLVR training often leads to excessively verbose processes (in reasoning tasks) and inefficient exploration trajectories (in agentic settings), as outcome-only rewards provide no incentive for efficiency and the high variance in response length within relatively small rollout groups results in noisy optimization signals. To address this, we propose Rollout Response Recomposition (RoRecomp), a plug-and-play method that guides models toward concise reasoning by strategically recomposing the training data. RoRecomp separates responses into two distinct batch types: 1) priority batches, which combine short-correct and long-incorrect responses selected from online batches to provide a clear gradient signal for brevity, and 2) compensation batches, which utilize remaining responses from a replay buffer to maintain stability and prevent model collapse. To comprehensively evaluate effectiveness, we test RoRecomp across three settings where results demonstrate substantial efficiency gains: reducing reasoning length by 27.7% in zero RL training, reducing unnecessary tool calls by 46.8% while improving accuracy in agentic RL, and achieving up to 52.5% length reduction in thinking compression, all with minimal performance impact.

**arXiv ID:** 2509.25958
</details>

<details>
<summary><strong>Fine-tuning Behavioral Cloning Policies with Preference-Based Reinforcement Learning</strong> - Maël Macuglia, Paul Friedrich, Giorgia Ramponi - [[pdf]](https://arxiv.org/pdf/2509.26605)</summary>

**Abstract:** Deploying reinforcement learning (RL) in robotics, industry, and health care is blocked by two obstacles: the difficulty of specifying accurate rewards and the risk of unsafe, data-hungry exploration. We address this by proposing a two-stage framework that first learns a safe initial policy from a reward-free dataset of expert demonstrations, then fine-tunes it online using preference-based human feedback. We provide the first principled analysis of this offline-to-online approach and introduce BRIDGE, a unified algorithm that integrates both signals via an uncertainty-weighted objective. We derive regret bounds that shrink with the number of offline demonstrations, explicitly connecting the quantity of offline data to online sample efficiency. We validate BRIDGE in discrete and continuous control MuJoCo environments, showing it achieves lower regret than both standalone behavioral cloning and online preference-based RL. Our work establishes a theoretical foundation for designing more sample-efficient interactive agents.

**arXiv ID:** 2509.26605
</details>

<details>
<summary><strong>APRIL: API Synthesis with Automatic Prompt Optimization and Reinforcement Learning</strong> - Hua Zhong, Shan Jiang, Sarfraz Khurshid - [[pdf]](https://arxiv.org/pdf/2509.25196)</summary>

**Abstract:** APIs are central to modern software development, yet composing new APIs from large libraries is difficult due to the exponential search space; traditional component-based synthesis relies on costly exploration and hand-crafted specifications. While large language models (LLMs) can generate implementations from natural language, hallucinations and limited access to up-to-date contextual information often yield incorrect code. In this paper, we present APRIL, an approach that combines LLM-based synthesis with Automatic Prompt Optimization (APO) and Reinforcement Learning from Verifiable Rewards (RLVR): APO iteratively refines prompts for a frozen model, while RLVR fine-tunes the policy toward functional correctness, producing an efficient synthesis pipeline. Evaluated on 81 real-world APIs from widely used scientific Python libraries and benchmarked against instruction-tuned but unfine-tuned LLMs guided by expert prompts, APRIL achieves substantial improvements. These results indicate that integrating APO and RLVR provides a robust, scalable path for component-based API synthesis in large libraries.

**arXiv ID:** 2509.25196
</details>

<details>
<summary><strong>Reinforcement Learning-Guided Chain-of-Draft for Token-Efficient Code Generation</strong> - Xunzhu Tang, Iyiola Emmanuel Olatunji, Tiezhu Sun, Jacques Klein, Tegawende F. Bissyande - [[pdf]](https://arxiv.org/pdf/2509.25243)</summary>

**Abstract:** LLMs demonstrate surface-level fluency in code generation but struggle with structured reasoning tasks requiring correctness and semantic alignment. While Chain-of-Thought (CoT) prompting enhances reasoning through intermediate steps, it suffers from verbosity and inefficiency. Chain-of-Draft (CoD) prompting offers more concise reasoning, but the stochastic nature of LLMs produces varying solution quality, making optimal selection challenging. We propose \multicod, a reinforcement learning framework that learns to select the most promising candidate from CoD-generated solutions. Our approach uses strategy-guided prompting to encourage diverse reasoning styles and models solution selection as a contextual bandit problem. The framework optimizes interpretable features including code complexity, reasoning structure, and strategic metadata through a reward function balancing correctness, efficiency, and clarity. Experiments on MBPP, BigCodeBench, SWE-bench Verified, and Defects4J show \multicod~outperforms and in some cases, on par with standard prompting, CoT, and CoD baselines while achieving cost and token efficiency from the user's perspective through a multi-candidate design that charges only for the selected output, reducing user billing by over 50\% and improving LLM response quality, making \multicod~more sustainable and scalable for real-world deployment. Our code is available: this https URL.

**arXiv ID:** 2509.25243
</details>

<details>
<summary><strong>Dynamic Policy Induction for Adaptive Prompt Optimization: Bridging the Efficiency-Accuracy Gap via Lightweight Reinforcement Learning</strong> - Jiexi Xu - [[pdf]](https://arxiv.org/pdf/2509.25267)</summary>

**Abstract:** The performance of Large Language Models (LLMs) depends heavily on the chosen prompting strategy, yet static approaches such as Zero-Shot, Few-Shot, or Chain-of-Thought (CoT) impose a rigid efficiency-accuracy trade-off. Highly accurate strategies like Self-Consistency (SC) incur substantial computational waste on simple tasks, while lightweight methods often fail on complex inputs. This paper introduces the Prompt Policy Network (PPN), a lightweight reinforcement learning framework that formalizes adaptive strategy selection as a single-step Markov Decision Process (MDP). The PPN, trained with Proximal Policy Optimization (PPO) and guided by a resource-explicit reward function, learns to allocate costly reasoning strategies only when necessary. Experiments on arithmetic reasoning benchmarks demonstrate that PPN achieves superior performance on the efficiency-accuracy Pareto front, delivering up to 61.5% token cost reduction compared to Self-Consistency while maintaining competitive accuracy. This work contributes a systematic, adaptive framework for cost-efficient LLM deployment, advancing the design of lightweight optimization techniques for scalable and sustainable language model applications.

**arXiv ID:** 2509.25267
</details>

<details>
<summary><strong>Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning</strong> - Zelin Tan, Hejia Geng, Mulei Zhang, Xiaohang Yu, Guancheng Wan, Yifan Zhou, Qiang He, Xiangyuan Xue, Heng Zhou, Yutao Fan, Zhongzhi Li, Zaibin Zhang, Guibin Zhang, Chen Zhang, Zhenfei Yin, Lei Bai - [[pdf]](https://arxiv.org/pdf/2509.25300)</summary>

**Abstract:** While scaling laws for large language models (LLMs) during pre-training have been extensively studied, their behavior under reinforcement learning (RL) post-training remains largely unexplored. This paper presents a systematic empirical investigation of scaling behaviors in RL-based post-training, with a particular focus on mathematical reasoning. Based on 54 experiments across diverse model sizes and training settings, we characterize how model scale, data volume, and computational budget interact to shape performance. Our analysis leads to four key findings: (1). Under a fixed computational budget, larger models trained for fewer steps consistently outperform smaller models trained for more steps. (2). Given a fixed amount of training data, larger models achieve superior sample efficiency, yielding lower loss. (3). In data-constrained regimes, repeated reuse of high-quality data proves highly effective, as final performance is primarily governed by the total number of optimization steps rather than the uniqueness of samples. (4). These scaling behaviors are robust across both base and instruction-tuned models, which share similar learning dynamics (e.g., larger models show faster convergence) even while differing in absolute accuracy. Collectively, these results provide a principled foundation and practical guidelines for efficiently scaling the reasoning capabilities of LLMs through RL post-training.

**arXiv ID:** 2509.25300
</details>

<details>
<summary><strong>Polychromic Objectives for Reinforcement Learning</strong> - Jubayer Ibn Hamid, Ifdita Hasan Orney, Ellen Xu, Chelsea Finn, Dorsa Sadigh - [[pdf]](https://arxiv.org/pdf/2509.25424)</summary>

**Abstract:** Reinforcement learning fine-tuning (RLFT) is a dominant paradigm for improving pretrained policies for downstream tasks. These pretrained policies, trained on large datasets, produce generations with a broad range of promising but unrefined behaviors. Often, a critical failure mode of RLFT arises when policies lose this diversity and collapse into a handful of easily exploitable outputs. This convergence hinders exploration, which is essential for expanding the capabilities of the pretrained policy and for amplifying the benefits of test-time compute scaling. To address this, we introduce an objective for policy gradient methods that explicitly enforces the exploration and refinement of diverse generations, which we call a polychromic objective. We then show how proximal policy optimization (PPO) can be adapted to optimize this objective. Our method (1) employs vine sampling to collect on-policy rollouts and (2) modifies the advantage function to reflect the advantage under our new objective. Experiments on BabyAI, Minigrid, and Algorithmic Creativity show that our method improves success rates by reliably solving a larger set of environment configurations and generalizes better under large perturbations. Moreover, when given multiple attempts in pass@$k$ experiments, the policy achieves substantially higher coverage, demonstrating its ability to maintain and exploit a diverse repertoire of strategies.

**arXiv ID:** 2509.25424
</details>

<details>
<summary><strong>PIPer: On-Device Environment Setup via Online Reinforcement Learning</strong> - Alexander Kovrigin, Aleksandra Eliseeva, Konstantin Grotov, Egor Bogomolov, Yaroslav Zharov - [[pdf]](https://arxiv.org/pdf/2509.25455)</summary>

**Abstract:** Environment setup-the process of configuring the system to work with a specific software project-represents a persistent challenge in Software Engineering (SE). Automated environment setup methods could assist developers by providing fully configured environments for arbitrary repositories without manual effort. This also helps SE researchers to scale execution-based benchmarks. However, recent studies reveal that even state-of-the-art Large Language Models (LLMs) achieve limited success in automating this task. To address this limitation, we tune a specialized model for environment setup. We combine supervised fine-tuning for generating correct Bash scripts and Reinforcement Learning with Verifiable Rewards (RLVR) to adapt it to the task of environment setup. On EnvBench-Python, our method enables Qwen3-8B (a model runnable on consumer hardware) to perform on par with larger models-Qwen3-32B and GPT-4o. The training code and model checkpoints are available online: this https URL.

**arXiv ID:** 2509.25455
</details>

<details>
<summary><strong>Self-Rewarding Rubric-Based Reinforcement Learning for Open-Ended Reasoning</strong> - Zhiling Ye, Yun Yue, Haowen Wang, Xudong Han, Jiadi Jiang, Cheng Wei, Lei Fan, Jiaxin Liang, Shuowen Zhang, Ji Li, Chunxiao Guo, Jian Wang, Peng Wei, Jinjie Gu - [[pdf]](https://arxiv.org/pdf/2509.25534)</summary>

**Abstract:** Open-ended evaluation is essential for deploying large language models in real-world settings. In studying HealthBench, we observe that using the model itself as a grader and generating rubric-based reward signals substantially improves reasoning performance. Remarkably, the trained model also becomes a stronger grader. Motivated by this, we introduce Self-Rewarding Rubric-Based Reinforcement Learning for Open-Ended Reasoning, a lightweight framework that enables faster and more resource-efficient training while surpassing baselines. Remarkably, on Qwen3-32B, training with just the 4000-sample HealthBench Easy subset is sufficient to obtain a model that exceeds GPT-5 on HealthBench Hard. Incorporating a small amount of teacher-graded data further enhances performance for less capable models.

**arXiv ID:** 2509.25534
</details>

<details>
<summary><strong>Deep Reinforcement Learning-Based Precoding for Multi-RIS-Aided Multiuser Downlink Systems with Practical Phase Shift</strong> - Po-Heng Chou, Bo-Ren Zheng, Wan-Jen Huang, Walid Saad, Yu Tsao, Ronald Y. Chang - [[pdf]](https://arxiv.org/pdf/2509.25661)</summary>

**Abstract:** This study considers multiple reconfigurable intelligent surfaces (RISs)-aided multiuser downlink systems with the goal of jointly optimizing the transmitter precoding and RIS phase shift matrix to maximize spectrum efficiency. Unlike prior work that assumed ideal RIS reflectivity, a practical coupling effect is considered between reflecting amplitude and phase shift for the RIS elements. This makes the optimization problem non-convex. To address this challenge, we propose a deep deterministic policy gradient (DDPG)-based deep reinforcement learning (DRL) framework. The proposed model is evaluated under both fixed and random numbers of users in practical mmWave channel settings. Simulation results demonstrate that, despite its complexity, the proposed DDPG approach significantly outperforms optimization-based algorithms and double deep Q-learning, particularly in scenarios with random user distributions.

**arXiv ID:** 2509.25661
</details>

<details>
<summary><strong>Boundary-to-Region Supervision for Offline Safe Reinforcement Learning</strong> - Huikang Su, Dengyun Peng, Zifeng Zhuang, YuHan Liu, Qiguang Chen, Donglin Wang, Qinghe Liu - [[pdf]](https://arxiv.org/pdf/2509.25727)</summary>

**Abstract:** Offline safe reinforcement learning aims to learn policies that satisfy predefined safety constraints from static datasets. Existing sequence-model-based methods condition action generation on symmetric input tokens for return-to-go and cost-to-go, neglecting their intrinsic asymmetry: return-to-go (RTG) serves as a flexible performance target, while cost-to-go (CTG) should represent a rigid safety boundary. This symmetric conditioning leads to unreliable constraint satisfaction, especially when encountering out-of-distribution cost trajectories. To address this, we propose Boundary-to-Region (B2R), a framework that enables asymmetric conditioning through cost signal realignment . B2R redefines CTG as a boundary constraint under a fixed safety budget, unifying the cost distribution of all feasible trajectories while preserving reward structures. Combined with rotary positional embeddings , it enhances exploration within the safe region. Experimental results show that B2R satisfies safety constraints in 35 out of 38 safety-critical tasks while achieving superior reward performance over baseline methods. This work highlights the limitations of symmetric token conditioning and establishes a new theoretical and practical approach for applying sequence models to safe RL. Our code is available at this https URL.

**arXiv ID:** 2509.25727
</details>

<details>
<summary><strong>TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning</strong> - Zhepei Wei, Xiao Yang, Kai Sun, Jiaqi Wang, Rulin Shao, Sean Chen, Mohammad Kachuee, Teja Gollapudi, Tony Liao, Nicolas Scheffer, Rakesh Wanga, Anuj Kumar, Yu Meng, Wen-tau Yih, Xin Luna Dong - [[pdf]](https://arxiv.org/pdf/2509.25760)</summary>

**Abstract:** While large language models (LLMs) have demonstrated strong performance on factoid question answering, they are still prone to hallucination and untruthful responses, particularly when tasks demand information outside their parametric knowledge. Indeed, truthfulness requires more than accuracy -- models must also recognize uncertainty and abstain when unsure to avoid hallucinations. This presents a fundamental challenge for existing methods: approaches that optimize for accuracy often amplify hallucinations, while those that encourage abstention can become overly conservative, sacrificing correct answers. Both extremes ultimately compromise truthfulness. In this work, we present TruthRL, a general reinforcement learning (RL) framework that directly optimizes the truthfulness of LLMs. Specifically, we implement TruthRL using GRPO with a simple yet effective ternary reward that distinguishes correct answers, hallucinations, and abstentions. It incentivizes models to reduce hallucinations not only by providing correct responses, but also by enabling abstention when uncertain, thereby improving truthfulness. Extensive experiments across four knowledge-intensive benchmarks show that, compared to vanilla RL, TruthRL significantly reduces hallucinations by 28.9% and improves truthfulness by 21.1%, with consistent gains across various backbone models (e.g., Qwen, Llama) under both retrieval and non-retrieval setups. In-depth ablation study demonstrates that vanilla accuracy-driven methods, such as supervised fine-tuning or RL with a binary reward, struggle to balance factual correctness and uncertainty. In contrast, our proposed truthfulness-driven TruthRL achieves strong performance in both accuracy and truthfulness, underscoring the importance of learning objective design for developing truthful LLMs.

**arXiv ID:** 2509.25760
</details>

<details>
<summary><strong>Efficient On-Policy Reinforcement Learning via Exploration of Sparse Parameter Space</strong> - Xinyu Zhang, Aishik Deb, Klaus Mueller - [[pdf]](https://arxiv.org/pdf/2509.25876)</summary>

**Abstract:** Policy-gradient methods such as Proximal Policy Optimization (PPO) are typically updated along a single stochastic gradient direction, leaving the rich local structure of the parameter space unexplored. Previous work has shown that the surrogate gradient is often poorly correlated with the true reward landscape. Building on this insight, we visualize the parameter space spanned by policy checkpoints within an iteration and reveal that higher performing solutions often lie in nearby unexplored regions. To exploit this opportunity, we introduce ExploRLer, a pluggable pipeline that seamlessly integrates with on-policy algorithms such as PPO and TRPO, systematically probing the unexplored neighborhoods of surrogate on-policy gradient updates. Without increasing the number of gradient updates, ExploRLer achieves significant improvements over baselines in complex continuous control environments. Our results demonstrate that iteration-level exploration provides a practical and effective way to strengthen on-policy reinforcement learning and offer a fresh perspective on the limitations of the surrogate objective.

**arXiv ID:** 2509.25876
</details>

<details>
<summary><strong>Asymmetric Information Enhanced Mapping Framework for Multirobot Exploration based on Deep Reinforcement Learning</strong> - Jiyu Cheng, Junhui Fan, Xiaolei Li, Paul L. Rosin, Yibin Li, Wei Zhang - [[pdf]](https://arxiv.org/pdf/2404.18089)</summary>

**Abstract:** Despite the great development of multirobot technologies, efficiently and collaboratively exploring an unknown environment is still a big challenge. In this paper, we propose AIM-Mapping, a Asymmetric InforMation Enhanced Mapping framework. The framework fully utilizes the privilege information in the training process to help construct the environment representation as well as the supervised signal in an asymmetric actor-critic training framework. Specifically, privilege information is used to evaluate the exploration performance through an asymmetric feature representation module and a mutual information evaluation module. The decision-making network uses the trained feature encoder to extract structure information from the environment and combines it with a topological map constructed based on geometric distance. Utilizing this kind of topological map representation, we employ topological graph matching to assign corresponding boundary points to each robot as long-term goal points. We conduct experiments in real-world-like scenarios using the Gibson simulation environments. It validates that the proposed method, when compared to existing methods, achieves great performance improvement.

**arXiv ID:** 2404.18089
</details>

<details>
<summary><strong>Mem-α: Learning Memory Construction via Reinforcement Learning</strong> - Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, Xiaojian Wu - [[pdf]](https://arxiv.org/pdf/2509.25911)</summary>

**Abstract:** Large language model (LLM) agents are constrained by limited context windows, necessitating external memory systems for long-term information understanding. Current memory-augmented agents typically depend on pre-defined instructions and tools for memory updates. However, language models may lack the ability to determine which information to store, how to structure it, and when to update it, especially as memory systems become more complex. This results in suboptimal memory construction and information loss. To this end, we propose Mem-alpha, a reinforcement learning framework that trains agents to effectively manage complex memory systems through interaction and feedback. We also construct a specialized training dataset spanning diverse multi-turn interaction patterns paired with comprehensive evaluation questions designed to teach effective memory management. During training, agents process sequential information chunks, learn to extract and store relevant content, then update the memory system. The reward signal derives from downstream question-answering accuracy over the full interaction history, directly optimizing for memory construction. To illustrate the effectiveness of our training framework, we design a memory architecture comprising core, episodic, and semantic components, equipped with multiple tools for memory operations. Empirical evaluation demonstrates that Mem-alpha achieves significant improvements over existing memory-augmented agent baselines. Despite being trained exclusively on instances with a maximum length of 30k tokens, our agents exhibit remarkable generalization to sequences exceeding 400k tokens, over 13x the training length, highlighting the robustness of Mem-alpha.

**arXiv ID:** 2509.25911
</details>

<details>
<summary><strong>Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning</strong> - Jinyeop Song, Song Wang, Julian Shun, Yada Zhu - [[pdf]](https://arxiv.org/pdf/2509.26383)</summary>

**Abstract:** Knowledge-graph retrieval-augmented generation (KG-RAG) couples large language models (LLMs) with structured, verifiable knowledge graphs (KGs) to reduce hallucinations and expose reasoning traces. However, many KG-RAG systems compose multiple LLM modules (e.g planning, reasoning, and responding), inflating inference cost and binding behavior to a specific target KG. To address this, we introduce KG-R1, an agentic KG retrieval-augmented generation (KG-RAG) framework through reinforcement learning (RL). KG-R1 utilizes a single agent that interacts with KGs as its environment, learning to retrieve at each step and incorporating the retrieved information into its reasoning and generation. The process is optimized through end-to-end RL. In controlled experiments across Knowledge-Graph Question Answering (KGQA) benchmarks, our method demonstrates both efficiency and transferability: Using Qwen-2.5-3B, KG-R1 improves answer accuracy with fewer generation tokens than prior multi-module workflow methods that use larger foundation or fine-tuned models. Furthermore, KG-R1 enables plug and play: after training, it maintains strong accuracy on new KGs without modification. These properties make KG-R1 a promising KG-RAG framework for real-world deployment. Our code is publicly available at this https URL.

**arXiv ID:** 2509.26383
</details>

<details>
<summary><strong>Clarification as Supervision: Reinforcement Learning for Vision-Language Interfaces</strong> - John Gkountouras, Ivan Titov - [[pdf]](https://arxiv.org/pdf/2509.26594)</summary>

**Abstract:** Recent text-only models demonstrate remarkable mathematical reasoning capabilities. Extending these to visual domains requires vision-language models to translate images into text descriptions. However, current models, trained to produce captions for human readers, often omit the precise details that reasoning systems require. This creates an interface mismatch: reasoners often fail not due to reasoning limitations but because they lack access to critical visual information. We propose Adaptive-Clarification Reinforcement Learning (AC-RL), which teaches vision models what information reasoners need through interaction. Our key insight is that clarification requests during training reveal information gaps; by penalizing success that requires clarification, we create pressure for comprehensive initial captions that enable the reasoner to solve the problem in a single pass. AC-RL improves average accuracy by 4.4 points over pretrained baselines across seven visual mathematical reasoning benchmarks, and analysis shows it would cut clarification requests by up to 39% if those were allowed. By treating clarification as a form of implicit supervision, AC-RL demonstrates that vision-language interfaces can be effectively learned through interaction alone, without requiring explicit annotations.

**arXiv ID:** 2509.26594
</details>

<details>
<summary><strong>R1-Code-Interpreter: LLMs Reason with Code via Supervised and Multi-stage Reinforcement Learning</strong> - Yongchao Chen, Yueying Liu, Junwei Zhou, Yilun Hao, Jingquan Wang, Yang Zhang, Na Li, Chuchu Fan - [[pdf]](https://arxiv.org/pdf/2505.21668)</summary>

**Abstract:** Practical guidance on training Large Language Models (LLMs) to leverage Code Interpreter across diverse tasks remains lacking. We present R1-Code-Interpreter, an extension of a text-only LLM trained via multi-turn supervised fine-tuning (SFT) and reinforcement learning (RL) to autonomously generate multiple code queries during step-by-step reasoning. Unlike prior RL + tool-use efforts focused on narrow domains such as math or retrieval, we curate 144 diverse reasoning and planning tasks and show that training a general-purpose Code Interpreter across them presents significant challenges due to task heterogeneity and scarcity of effective samples. To address this, we introduce a multi-stage curriculum learning approach that partitions training samples by measured improvement potential. The RL training prioritizes samples with higher potential and gradually shifts to lower-potential ones, increasing the average RL gains from merely +3.4% to +9.3% across Qwen-2.5 models (3/7/14B). Our final model, R1-CI-14B, improves average accuracy on the 37 test tasks from 44.1% to 72.4%, outperforming text-only GPT-4o (58.6%) and GPT-4o with Code Interpreter (70.9%). Notably, R1-CI-14B also exhibits emergent self-checking behavior through code generation. Datasets, Codes, and Models are available at this https URL and this https URL.

**arXiv ID:** 2505.21668
</details>

<details>
<summary><strong>VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use</strong> - Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai Zou, Chao Du, Tianyu Pang, Wenhu Chen - [[pdf]](https://arxiv.org/pdf/2509.01055)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated success in enhancing LLM reasoning capabilities, but remains limited to single-turn interactions without tool integration. While recent Agentic Reinforcement Learning with Tool use (ARLT) approaches have emerged to address multi-turn tool interactions, existing works develop task-specific codebases that suffer from fragmentation, synchronous execution bottlenecks, and limited extensibility across domains. These inefficiencies hinder broader community adoption and algorithmic innovation. We introduce VerlTool, a unified and modular framework that addresses these limitations through systematic design principles. VerlTool provides four key contributions: (1) upstream alignment with VeRL ensuring compatibility and simplified maintenance, (2) unified tool management via standardized APIs supporting diverse modalities including code execution, search, SQL databases, and vision processing, (3) asynchronous rollout execution achieving near 2$\times$ speedup by eliminating synchronization bottlenecks, and (4) comprehensive evaluation demonstrating competitive performance across 6 ARLT domains. Our framework formalizes ARLT as multi-turn trajectories with multi-modal observation tokens (text/image/video), extending beyond single-turn RLVR paradigms. We train and evaluate models on mathematical reasoning, knowledge QA, SQL generation, visual reasoning, web search, and software engineering tasks, achieving results comparable to specialized systems while providing unified training infrastructure. The modular plugin architecture enables rapid tool integration requiring only lightweight Python definitions, significantly reducing development overhead and providing a scalable foundation for tool-augmented RL research. Our code is open-sourced at this https URL.

**arXiv ID:** 2509.01055
</details>

<details>
<summary><strong>CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning</strong> - Zhenpeng Su, Leiyu Pan, Minxuan Lv, Yuntao Li, Wenping Hu, Fuzheng Zhang, Kun Gai, Guorui Zhou - [[pdf]](https://arxiv.org/pdf/2509.20712)</summary>

**Abstract:** Reinforcement learning (RL) has become a powerful paradigm for optimizing large language models (LLMs) to handle complex reasoning tasks. A core challenge in this process lies in managing policy entropy, which reflects the balance between exploration and exploitation during training. Existing methods, such as proximal policy optimization (PPO) and its variants, discard valuable gradient signals from low-probability tokens due to the clipping mechanism. We systematically analyze the entropy dynamics and reveal that these clipped tokens play a critical yet overlooked role in regulating entropy evolution. We propose \textbf{C}oordinating \textbf{E}ntropy via \textbf{G}radient-\textbf{P}reserving \textbf{P}olicy \textbf{O}ptimization (CE-GPPO), a novel algorithm that reintroduces gradients from clipped tokens in native PPO in a gentle and bounded manner. By controlling the magnitude of gradients from tokens outside the clipping interval, CE-GPPO is able to achieve an exploration-exploitation trade-off. We provide theoretical justification and empirical evidence showing that CE-GPPO effectively mitigates entropy instability. Extensive experiments on mathematical reasoning benchmarks show that CE-GPPO consistently outperforms strong baselines across different model scales.

**arXiv ID:** 2509.20712
</details>

<details>
<summary><strong>Optimisation of Resource Allocation in Heterogeneous Wireless Networks Using Deep Reinforcement Learning</strong> - Oluwaseyi Giwa, Jonathan Shock, Jaco Du Toit, Tobi Awodumila - [[pdf]](https://arxiv.org/pdf/2509.25284)</summary>

**Abstract:** Dynamic resource allocation in heterogeneous wireless networks (HetNets) is challenging for traditional methods under varying user loads and channel conditions. We propose a deep reinforcement learning (DRL) framework that jointly optimises transmit power, bandwidth, and scheduling via a multi-objective reward balancing throughput, energy efficiency, and fairness. Using real base station coordinates, we compare Proximal Policy Optimisation (PPO) and Twin Delayed Deep Deterministic Policy Gradient (TD3) against three heuristic algorithms in multiple network scenarios. Our results show that DRL frameworks outperform heuristic algorithms in optimising resource allocation in dynamic networks. These findings highlight key trade-offs in DRL design for future HetNets.

**arXiv ID:** 2509.25284
</details>

<details>
<summary><strong>World Model for AI Autonomous Navigation in Mechanical Thrombectomy</strong> - Harry Robertshaw, Han-Ru Wu, Alejandro Granados, Thomas C Booth - [[pdf]](https://arxiv.org/pdf/2509.25518)</summary>

**Abstract:** Autonomous navigation for mechanical thrombectomy (MT) remains a critical challenge due to the complexity of vascular anatomy and the need for precise, real-time decision-making. Reinforcement learning (RL)-based approaches have demonstrated potential in automating endovascular navigation, but current methods often struggle with generalization across multiple patient vasculatures and long-horizon tasks. We propose a world model for autonomous endovascular navigation using TD-MPC2, a model-based RL algorithm. We trained a single RL agent across multiple endovascular navigation tasks in ten real patient vasculatures, comparing performance against the state-of-the-art Soft Actor-Critic (SAC) method. Results indicate that TD-MPC2 significantly outperforms SAC in multi-task learning, achieving a 65% mean success rate compared to SAC's 37%, with notable improvements in path ratio. TD-MPC2 exhibited increased procedure times, suggesting a trade-off between success rate and execution speed. These findings highlight the potential of world models for improving autonomous endovascular navigation and lay the foundation for future research in generalizable AI-driven robotic interventions.

**arXiv ID:** 2509.25518
</details>

<details>
<summary><strong>Safe In-Context Reinforcement Learning</strong> - Amir Moeini, Minjae Kwon, Alper Kamil Bozkurt, Yuichi Motai, Rohan Chandra, Lu Feng, Shangtong Zhang - [[pdf]](https://arxiv.org/pdf/2509.25582)</summary>

**Abstract:** In-context reinforcement learning (ICRL) is an emerging RL paradigm where the agent, after some pretraining procedure, is able to adapt to out-of-distribution test tasks without any parameter updates. The agent achieves this by continually expanding the input (i.e., the context) to its policy neural networks. For example, the input could be all the history experience that the agent has access to until the current time step. The agent's performance improves as the input grows, without any parameter updates. In this work, we propose the first method that promotes the safety of ICRL's adaptation process in the framework of constrained Markov Decision Processes. In other words, during the parameter-update-free adaptation process, the agent not only maximizes the reward but also minimizes an additional cost function. We also demonstrate that our agent actively reacts to the threshold (i.e., budget) of the cost tolerance. With a higher cost budget, the agent behaves more aggressively, and with a lower cost budget, the agent behaves more conservatively.

**arXiv ID:** 2509.25582
</details>

<details>
<summary><strong>Clip-Low Increases Entropy and Clip-High Decreases Entropy in Reinforcement Learning of Large Language Models</strong> - Jaesung R. Park, Junsu Kim, Gyeongman Kim, Jinyoung Jo, Sean Choi, Jaewoong Cho, Ernest K. Ryu - [[pdf]](https://arxiv.org/pdf/2509.26114)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has recently emerged as the leading approach for enhancing the reasoning capabilities of large language models (LLMs). However, RLVR is prone to entropy collapse, where the LLM quickly converges to a near-deterministic form, hindering exploration and progress during prolonged RL training. In this work, we reveal that the clipping mechanism in PPO and GRPO induces biases on entropy. Through theoretical and empirical analyses, we show that clip-low increases entropy, while clip-high decreases it. Further, under standard clipping parameters, the effect of clip-high dominates, resulting in an overall entropy reduction even when purely random rewards are provided to the RL algorithm. Our findings highlight an overlooked confounding factor in RLVR: independent of the reward signal, the clipping mechanism influences entropy, which in turn affects the reasoning behavior. Furthermore, our analysis demonstrates that clipping can be deliberately used to control entropy. Specifically, with a more aggressive clip-low value, one can increase entropy, promote exploration, and ultimately prevent entropy collapse in RLVR training.

**arXiv ID:** 2509.26114
</details>

<details>
<summary><strong>Extensions of Robbins-Siegmund Theorem with Applications in Reinforcement Learning</strong> - Xinyu Liu, Zixuan Xie, Shangtong Zhang - [[pdf]](https://arxiv.org/pdf/2509.26442)</summary>

**Abstract:** The Robbins-Siegmund theorem establishes the convergence of stochastic processes that are almost supermartingales and is foundational for analyzing a wide range of stochastic iterative algorithms in stochastic approximation and reinforcement learning (RL). However, its original form has a significant limitation as it requires the zero-order term to be summable. In many important RL applications, this summable condition, however, cannot be met. This limitation motivates us to extend the Robbins-Siegmund theorem for almost supermartingales where the zero-order term is not summable but only square summable. Particularly, we introduce a novel and mild assumption on the increments of the stochastic processes. This together with the square summable condition enables an almost sure convergence to a bounded set. Additionally, we further provide almost sure convergence rates, high probability concentration bounds, and $L^p$ convergence rates. We then apply the new results in stochastic approximation and RL. Notably, we obtain the first almost sure convergence rate, the first high probability concentration bound, and the first $L^p$ convergence rate for $Q$-learning with linear function approximation.

**arXiv ID:** 2509.26442
</details>

<details>
<summary><strong>SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling</strong> - Yixian Zhang, Shu'ang Yu, Tonghe Zhang, Mo Guang, Haojia Hui, Kaiwen Long, Yu Wang, Chao Yu, Wenbo Ding - [[pdf]](https://arxiv.org/pdf/2509.25756)</summary>

**Abstract:** Training expressive flow-based policies with off-policy reinforcement learning is notoriously unstable due to gradient pathologies in the multi-step action sampling process. We trace this instability to a fundamental connection: the flow rollout is algebraically equivalent to a residual recurrent computation, making it susceptible to the same vanishing and exploding gradients as RNNs. To address this, we reparameterize the velocity network using principles from modern sequential models, introducing two stable architectures: Flow-G, which incorporates a gated velocity, and Flow-T, which utilizes a decoded velocity. We then develop a practical SAC-based algorithm, enabled by a noise-augmented rollout, that facilitates direct end-to-end training of these policies. Our approach supports both from-scratch and offline-to-online learning and achieves state-of-the-art performance on continuous control and robotic manipulation benchmarks, eliminating the need for common workarounds like policy distillation or surrogate objectives.

**arXiv ID:** 2509.25756
</details>

<details>
<summary><strong>Parallel Heuristic Search as Inference for Actor-Critic Reinforcement Learning Models</strong> - Hanlan Yang, Itamar Mishani, Luca Pivetti, Zachary Kingston, Maxim Likhachev - [[pdf]](https://arxiv.org/pdf/2509.25402)</summary>

**Abstract:** Actor-Critic models are a class of model-free deep reinforcement learning (RL) algorithms that have demonstrated effectiveness across various robot learning tasks. While considerable research has focused on improving training stability and data sampling efficiency, most deployment strategies have remained relatively simplistic, typically relying on direct actor policy rollouts. In contrast, we propose \pachs{} (\textit{P}arallel \textit{A}ctor-\textit{C}ritic \textit{H}euristic \textit{S}earch), an efficient parallel best-first search algorithm for inference that leverages both components of the actor-critic architecture: the actor network generates actions, while the critic network provides cost-to-go estimates to guide the search. Two levels of parallelism are employed within the search -- actions and cost-to-go estimates are generated in batches by the actor and critic networks respectively, and graph expansion is distributed across multiple threads. We demonstrate the effectiveness of our approach in robotic manipulation tasks, including collision-free motion planning and contact-rich interactions such as non-prehensile pushing. Visit this http URL for demonstrations and examples.

**arXiv ID:** 2509.25402
</details>

<details>
<summary><strong>CoTaP: Compliant Task Pipeline and Reinforcement Learning of Its Controller with Compliance Modulation</strong> - Zewen He, Chenyuan Chen, Dilshod Azizov, Yoshihiko Nakamura - [[pdf]](https://arxiv.org/pdf/2509.25443)</summary>

**Abstract:** Humanoid whole-body locomotion control is a critical approach for humanoid robots to leverage their inherent advantages. Learning-based control methods derived from retargeted human motion data provide an effective means of addressing this issue. However, because most current human datasets lack measured force data, and learning-based robot control is largely position-based, achieving appropriate compliance during interaction with real environments remains challenging. This paper presents Compliant Task Pipeline (CoTaP): a pipeline that leverages compliance information in the learning-based structure of humanoid robots. A two-stage dual-agent reinforcement learning framework combined with model-based compliance control for humanoid robots is proposed. In the training process, first a base policy with a position-based controller is trained; then in the distillation, the upper-body policy is combined with model-based compliance control, and the lower-body agent is guided by the base policy. In the upper-body control, adjustable task-space compliance can be specified and integrated with other controllers through compliance modulation on the symmetric positive definite (SPD) manifold, ensuring system stability. We validated the feasibility of the proposed strategy in simulation, primarily comparing the responses to external disturbances under different compliance settings.

**arXiv ID:** 2509.25443
</details>

<details>
<summary><strong>Online Mapping for Autonomous Driving: Addressing Sensor Generalization and Dynamic Map Updates in Campus Environments</strong> - Zihan Zhang, Abhijit Ravichandran, Pragnya Korti, Luobin Wang, Henrik I. Christensen - [[pdf]](https://arxiv.org/pdf/2509.25542)</summary>

**Abstract:** High-definition (HD) maps are essential for autonomous driving, providing precise information such as road boundaries, lane dividers, and crosswalks to enable safe and accurate navigation. However, traditional HD map generation is labor-intensive, expensive, and difficult to maintain in dynamic environments. To overcome these challenges, we present a real-world deployment of an online mapping system on a campus golf cart platform equipped with dual front cameras and a LiDAR sensor. Our work tackles three core challenges: (1) labeling a 3D HD map for campus environment; (2) integrating and generalizing the SemVecMap model onboard; and (3) incrementally generating and updating the predicted HD map to capture environmental changes. By fine-tuning with campus-specific data, our pipeline produces accurate map predictions and supports continual updates, demonstrating its practical value in real-world autonomous driving scenarios.

**arXiv ID:** 2509.25542
</details>

<details>
<summary><strong>Autonomous Multi-Robot Infrastructure for AI-Enabled Healthcare Delivery and Diagnostics</strong> - Nakhul Kalaivanan, Senthil Arumugam Muthukumaraswamy, Girish Balasubramanian - [[pdf]](https://arxiv.org/pdf/2509.26106)</summary>

**Abstract:** This research presents a multi-robot system for inpatient care, designed using swarm intelligence principles and incorporating wearable health sensors, RF-based communication, and AI-driven decision support. Within a simulated hospital environment, the system adopts a leader-follower swarm configuration to perform patient monitoring, medicine delivery, and emergency assistance. Due to ethical constraints, live patient trials were not conducted; instead, validation was carried out through controlled self-testing with wearable sensors. The Leader Robot acquires key physiological parameters, including temperature, SpO2, heart rate, and fall detection, and coordinates other robots when required. The Assistant Robot patrols corridors for medicine delivery, while a robotic arm provides direct drug administration. The swarm-inspired leader-follower strategy enhanced communication reliability and ensured continuous monitoring, including automated email alerts to healthcare staff. The system hardware was implemented using Arduino, Raspberry Pi, NRF24L01 RF modules, and a HuskyLens AI camera. Experimental evaluation showed an overall sensor accuracy above 94%, a 92% task-level success rate, and a 96% communication reliability rate, demonstrating system robustness. Furthermore, the AI-enabled decision support was able to provide early warnings of abnormal health conditions, highlighting the potential of the system as a cost-effective solution for hospital automation and patient safety.

**arXiv ID:** 2509.26106
</details>

<details>
<summary><strong>Towards autonomous photogrammetric forest inventory using a lightweight under-canopy robotic drone</strong> - Väinö Karjalainen, Niko Koivumäki, Teemu Hakala, Jesse Muhojoki, Eric Hyyppä, Anand George, Juha Suomalainen, Eija Honkavaara - [[pdf]](https://arxiv.org/pdf/2501.12073)</summary>

**Abstract:** Drones are increasingly used in forestry to capture high-resolution remote sensing data, supporting enhanced monitoring, assessment, and decision-making processes. While operations above the forest canopy are already highly automated, flying inside forests remains challenging, primarily relying on manual piloting. In dense forests, relying on the Global Navigation Satellite System (GNSS) for localization is not feasible. In addition, the drone must autonomously adjust its flight path to avoid collisions. Recently, advancements in robotics have enabled autonomous drone flights in GNSS-denied obstacle-rich areas. In this article, a step towards autonomous forest data collection is taken by building a prototype of a robotic under-canopy drone utilizing state-of-the-art open source methods and validating its performance for data collection inside forests. Specifically, the study focused on camera-based autonomous flight under the forest canopy and photogrammetric post-processing of the data collected with the low-cost onboard stereo camera. The autonomous flight capability of the prototype was evaluated through multiple test flights in boreal forests. The tree parameter estimation capability was studied by performing diameter at breast height (DBH) estimation. The prototype successfully carried out flights in selected challenging forest environments, and the experiments showed promising performance in forest 3D modeling with a miniaturized stereoscopic photogrammetric system. The DBH estimation achieved a root mean square error (RMSE) of 3.33 - 3.97 cm (10.69 - 12.98 %) across all trees. For trees with a DBH less than 30 cm, the RMSE was 1.16 - 2.56 cm (5.74 - 12.47 %). The results provide valuable insights into autonomous under-canopy forest mapping and highlight the critical next steps for advancing lightweight robotic drone systems for mapping complex forest environments.

**arXiv ID:** 2501.12073
</details>

<details>
<summary><strong>Apple: Toward General Active Perception via Reinforcement Learning</strong> - Tim Schneider, Cristiana de Farias, Roberto Calandra, Liming Chen, Jan Peters - [[pdf]](https://arxiv.org/pdf/2505.06182)</summary>

**Abstract:** Active perception is a fundamental skill that enables us humans to deal with uncertainty in our inherently partially observable environment. For senses such as touch, where the information is sparse and local, active perception becomes crucial. In recent years, active perception has emerged as an important research domain in robotics. However, current methods are often bound to specific tasks or make strong assumptions, which limit their generality. To address this gap, this work introduces APPLE (Active Perception Policy Learning) - a novel framework that leverages reinforcement learning (RL) to address a range of different active perception problems. APPLE jointly trains a transformer-based perception module and decision-making policy with a unified optimization objective, learning how to actively gather information. By design, APPLE is not limited to a specific task and can, in principle, be applied to a wide range of active perception problems. We evaluate two variants of APPLE across different tasks, including tactile exploration problems from the Tactile MNIST benchmark. Experiments demonstrate the efficacy of APPLE, achieving high accuracies on both regression and classification tasks. These findings underscore the potential of APPLE as a versatile and general framework for advancing active perception in robotics.

**arXiv ID:** 2505.06182
</details>

<details>
<summary><strong>Ocean Diviner: A Diffusion-Augmented Reinforcement Learning Framework for AUV Robust Control in Underwater Tasks</strong> - Jingzehua Xu, Guanwen Xie, Weiyi Liu, Jiwei Tang, Ziteng Yang, Tianxiang Xing, Yiyuan Yang, Shuai Zhang, Xiaofan Li - [[pdf]](https://arxiv.org/pdf/2507.11283)</summary>

**Abstract:** Autonomous Underwater Vehicles (AUVs) are essential for marine exploration, yet their control remains highly challenging due to nonlinear dynamics and uncertain environmental disturbances. This paper presents a diffusion-augmented Reinforcement Learning (RL) framework for robust AUV control, aiming to improve AUV's adaptability in dynamic underwater environments. The proposed framework integrates two core innovations: (1) A diffusion-based action generation framework that produces physically feasible and high-quality actions, enhanced by a high-dimensional state encoding mechanism combining current observations with historical states and actions through a novel diffusion U-Net architecture, significantly improving long-horizon planning capacity for robust control. (2) A sample-efficient hybrid learning architecture that synergizes diffusion-guided exploration with RL policy optimization, where the diffusion model generates diverse candidate actions and the RL critic selects the optimal action, achieving higher exploration efficiency and policy stability in dynamic underwater environments. Extensive simulation experiments validate the framework's superior robustness and flexibility, outperforming conventional control methods in challenging marine conditions, offering enhanced adaptability and reliability for AUV operations in underwater tasks. Finally, we will release the code publicly soon to support future research in this area.

**arXiv ID:** 2507.11283
</details>

<details>
<summary><strong>Botender: Supporting Communities in Collaboratively Designing AI Agents through Case-Based Provocations</strong> - Tzu-Sheng Kuo, Sophia Liu, Quan Ze Chen, Joseph Seering, Amy X. Zhang, Haiyi Zhu, Kenneth Holstein - [[pdf]](https://arxiv.org/pdf/2509.25492)</summary>

**Abstract:** AI agents, or bots, serve important roles in online communities. However, they are often designed by outsiders or a few tech-savvy members, leading to bots that may not align with the broader community's needs. How might communities collectively shape the behavior of community bots? We present Botender, a system that enables communities to collaboratively design LLM-powered bots without coding. With Botender, community members can directly propose, iterate on, and deploy custom bot behaviors tailored to community needs. Botender facilitates testing and iteration on bot behavior through case-based provocations: interaction scenarios generated to spark user reflection and discussion around desirable bot behavior. A validation study found these provocations more useful than standard test cases for revealing improvement opportunities and surfacing disagreements. During a five-day deployment across six Discord servers, Botender supported communities in tailoring bot behavior to their specific needs, showcasing the usefulness of case-based provocations in facilitating collaborative bot design.

**arXiv ID:** 2509.25492
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
