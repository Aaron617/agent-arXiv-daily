# Agent arXiv Daily

**Last Updated:** 2025-10-24 02:38:56

**Total Papers:** 73

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
<summary><strong>AI PB: A Grounded Generative Agent for Personalized Investment Insights</strong> - Daewoo Park, Suho Park, Inseok Hong, Hanwool Lee, Junkyu Park, Sangjun Lee, Jeongman An, Hyunbin Loh - [[pdf]](https://arxiv.org/pdf/2510.20099)</summary>

**Abstract:** We present AI PB, a production-scale generative agent deployed in real retail finance. Unlike reactive chatbots that answer queries passively, AI PB proactively generates grounded, compliant, and user-specific investment insights. It integrates (i) a component-based orchestration layer that deterministically routes between internal and external LLMs based on data sensitivity, (ii) a hybrid retrieval pipeline using OpenSearch and the finance-domain embedding model, and (iii) a multi-stage recommendation mechanism combining rule heuristics, sequential behavioral modeling, and contextual bandits. Operating fully on-premises under Korean financial regulations, the system employs Docker Swarm and vLLM across 24 X NVIDIA H100 GPUs. Through human QA and system metrics, we demonstrate that grounded generation with explicit routing and layered safety can deliver trustworthy AI insights in high-stakes finance.

**arXiv ID:** 2510.20099
</details>

<details>
<summary><strong>GhostEI-Bench: Do Mobile Agents Resilience to Environmental Injection in Dynamic On-Device Environments?</strong> - Chiyu Chen, Xinhao Song, Yunkai Chai, Yang Yao, Haodong Zhao, Lijun Li, Jie Li, Yan Teng, Gongshen Liu, Yingchun Wang - [[pdf]](https://arxiv.org/pdf/2510.20333)</summary>

**Abstract:** Vision-Language Models (VLMs) are increasingly deployed as autonomous agents to navigate mobile graphical user interfaces (GUIs). Operating in dynamic on-device ecosystems, which include notifications, pop-ups, and inter-app interactions, exposes them to a unique and underexplored threat vector: environmental injection. Unlike prompt-based attacks that manipulate textual instructions, environmental injection corrupts an agent's visual perception by inserting adversarial UI elements (for example, deceptive overlays or spoofed notifications) directly into the GUI. This bypasses textual safeguards and can derail execution, causing privacy leakage, financial loss, or irreversible device compromise. To systematically evaluate this threat, we introduce GhostEI-Bench, the first benchmark for assessing mobile agents under environmental injection attacks within dynamic, executable environments. Moving beyond static image-based assessments, GhostEI-Bench injects adversarial events into realistic application workflows inside fully operational Android emulators and evaluates performance across critical risk scenarios. We further propose a judge-LLM protocol that conducts fine-grained failure analysis by reviewing the agent's action trajectory alongside the corresponding screenshot sequence, pinpointing failure in perception, recognition, or reasoning. Comprehensive experiments on state-of-the-art agents reveal pronounced vulnerability to deceptive environmental cues: current models systematically fail to perceive and reason about manipulated UIs. GhostEI-Bench provides a framework for quantifying and mitigating this emerging threat, paving the way toward more robust and secure embodied agents.

**arXiv ID:** 2510.20333
</details>

<details>
<summary><strong>Large Multimodal Models-Empowered Task-Oriented Autonomous Communications: Design Methodology and Implementation Challenges</strong> - Hyun Jong Yang, Hyunsoo Kim, Hyeonho Noh, Seungnyun Kim, Byonghyo Shim - [[pdf]](https://arxiv.org/pdf/2510.20637)</summary>

**Abstract:** Large language models (LLMs) and large multimodal models (LMMs) have achieved unprecedented breakthrough, showcasing remarkable capabilities in natural language understanding, generation, and complex reasoning. This transformative potential has positioned them as key enablers for 6G autonomous communications among machines, vehicles, and humanoids. In this article, we provide an overview of task-oriented autonomous communications with LLMs/LMMs, focusing on multimodal sensing integration, adaptive reconfiguration, and prompt/fine-tuning strategies for wireless tasks. We demonstrate the framework through three case studies: LMM-based traffic control, LLM-based robot scheduling, and LMM-based environment-aware channel estimation. From experimental results, we show that the proposed LLM/LMM-aided autonomous systems significantly outperform conventional and discriminative deep learning (DL) model-based techniques, maintaining robustness under dynamic objectives, varying input parameters, and heterogeneous multimodal conditions where conventional static optimization degrades.

**arXiv ID:** 2510.20637
</details>

<details>
<summary><strong>Designing Intent Communication for Agent-Human Collaboration</strong> - Yi Li, Francesco Chiossi, Helena Anna Frijns, Jan Leusmann, Julian Rasch, Robin Welsch, Philipp Wintersberger, Florian Michahelles, Albrecht Schmidt - [[pdf]](https://arxiv.org/pdf/2510.20409)</summary>

**Abstract:** As autonomous agents, from self-driving cars to virtual assistants, become increasingly present in everyday life, safe and effective collaboration depends on human understanding of agents' intentions. Current intent communication approaches are often rigid, agent-specific, and narrowly scoped, limiting their adaptability across tasks, environments, and user preferences. A key gap remains: existing models of what to communicate are rarely linked to systematic choices of how and when to communicate, preventing the development of generalizable, multi-modal strategies. In this paper, we introduce a multidimensional design space for intent communication structured along three dimensions: Transparency (what is communicated), Abstraction (when), and Modality (how). We apply this design space to three distinct human-agent collaboration scenarios: (a) bystander interaction, (b) cooperative tasks, and (c) shared control, demonstrating its capacity to generate adaptable, scalable, and cross-domain communication strategies. By bridging the gap between intent content and communication implementation, our design space provides a foundation for designing safer, more intuitive, and more transferable agent-human interactions.

**arXiv ID:** 2510.20409
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (8 papers)</h2></summary>

<details>
<summary><strong>Surfer 2: The Next Generation of Cross-Platform Computer Use Agents</strong> - Mathieu Andreux, Märt Bakler, Yanael Barbier, Hamza Ben Chekroun, Emilien Biré, Antoine Bonnet, Riaz Bordie, Nathan Bout, Matthias Brunel, Aleix Cambray, Pierre-Louis Cedoz, Antoine Chassang, Gautier Cloix, Ethan Connelly, Alexandra Constantinou, Ramzi De Coster, Hubert de la Jonquiere, Aurélien Delfosse, Maxime Delpit, Alexis Deprez, Augustin Derupti, Mathieu Diaz, Shannon D'Souza, Julie Dujardin, Abai Edmund, Michael Eickenberg, Armand Fatalot, Wissem Felissi, Isaac Herring, Xavier Koegler, Erwan Le Jumeau de Kergaradec, Aurélien Lac, Maxime Langevin, Corentin Lauverjat, Antonio Loison, Avshalom Manevich, Axel Moyal, Axel Nguyen Kerbel, Marinela Parovic, Julien Revelle, Guillaume Richard, Mats Richter, Ronan Riochet, María Santos, Romain Savidan, Laurent Sifre, Maxime Theillard, Marc Thibault, Ivan Valentini, Tony Wu, Laura Yie, Kai Yuan, Jevgenij Zubovskij - [[pdf]](https://arxiv.org/pdf/2510.19949)</summary>

**Abstract:** Building agents that generalize across web, desktop, and mobile environments remains an open challenge, as prior systems rely on environment-specific interfaces that limit cross-platform deployment. We introduce Surfer 2, a unified architecture operating purely from visual observations that achieves state-of-the-art performance across all three environments. Surfer 2 integrates hierarchical context management, decoupled planning and execution, and self-verification with adaptive recovery, enabling reliable operation over long task horizons. Our system achieves 97.1% accuracy on WebVoyager, 69.6% on WebArena, 60.1% on OSWorld, and 87.1% on AndroidWorld, outperforming all prior systems without task-specific fine-tuning. With multiple attempts, Surfer 2 exceeds human performance on all benchmarks. These results demonstrate that systematic orchestration amplifies foundation model capabilities and enables general-purpose computer control through visual interaction alone, while calling for a next-generation vision language model to achieve Pareto-optimal cost-efficiency.

**arXiv ID:** 2510.19949
</details>

<details>
<summary><strong>Multi-Step Reasoning for Embodied Question Answering via Tool Augmentation</strong> - Mingliang Zhai, Hansheng Liang, Xiaomeng Fan, Zhi Gao, Chuanhao Li, Che Sun, Xu Bin, Yuwei Wu, Yunde Jia - [[pdf]](https://arxiv.org/pdf/2510.20310)</summary>

**Abstract:** Embodied Question Answering (EQA) requires agents to explore 3D environments to obtain observations and answer questions related to the scene. Existing methods leverage VLMs to directly explore the environment and answer questions without explicit thinking or planning, which limits their reasoning ability and results in excessive or inefficient exploration as well as ineffective responses. In this paper, we introduce ToolEQA, an agent that integrates external tools with multi-step reasoning, where external tools can provide more useful information for completing the task, helping the model derive better exploration directions in the next step of reasoning and thus obtaining additional effective information. This enables ToolEQA to generate more accurate responses with a shorter exploration distance. To enhance the model's ability for tool-usage and multi-step reasoning, we further design a novel EQA data generation pipeline that automatically constructs large-scale EQA tasks with reasoning trajectories and corresponding answers. Based on the pipeline, we collect the EQA-RT dataset that contains about 18K tasks, divided into a training set EQA-RT-Train, and two test sets EQA-RT-Seen (scenes overlapping with the training set) and EQA-RT-Unseen (novel scenes). Experiments on EQA-RT-Seen and EQA-RT-Unseen show that ToolEQA improves the success rate by 9.2~20.2% over state-of-the-art baselines, while outperforming the zero-shot ToolEQA by 10% in success rate. In addition, ToolEQA also achieves state-of-the-art performance on the HM-EQA, OpenEQA, and EXPRESS-Bench datasets, demonstrating its generality. Our homepage see this https URL.

**arXiv ID:** 2510.20310
</details>

<details>
<summary><strong>Automated Cloud Infrastructure-as-Code Reconciliation with AI Agents</strong> - Zhenning Yang, Hui Guan, Victor Nicolet, Brandon Paulsen, Joey Dodds, Daniel Kroening, Ang Chen - [[pdf]](https://arxiv.org/pdf/2510.20211)</summary>

**Abstract:** Cloud infrastructure is managed through a mix of interfaces -- traditionally, cloud consoles, command-line interfaces (CLI), and SDKs are the tools of choice. Recently, Infrastructure-as-Code/IaC frameworks (e.g., Terraform) have quickly gained popularity. Unlike conventional tools, IaC~frameworks encode the infrastructure in a "source-of-truth" configuration. They are capable of automatically carrying out modifications to the cloud -- deploying, updating, or destroying resources -- to bring the actual infrastructure into alignment with the IaC configuration. However, when IaC is used alongside consoles, CLIs, or SDKs, it loses visibility into external changes, causing infrastructure drift, where the configuration becomes outdated, and later IaC operations may undo valid updates or trigger errors.
We present NSync, an automated system for IaC reconciliation that propagates out-of-band changes back into the IaC program. Our key insight is that infrastructure changes eventually all occur via cloud API invocations -- the lowest layer for cloud management operations. NSync gleans insights from API traces to detect drift (i.e., non-IaC changes) and reconcile it (i.e., update the IaC configuration to capture the changes). It employs an agentic architecture that leverages LLMs to infer high-level intents from noisy API sequences, synthesize targeted IaC updates using specialized tools, and continually improve through a self-evolving knowledge base of past reconciliations. We further introduce a novel evaluation pipeline for injecting realistic drifts into cloud infrastructure and assessing reconciliation performance. Experiments across five real-world Terraform projects and 372 drift scenarios show that NSync outperforms the baseline both in terms of accuracy (from 0.71 to 0.97 pass@3) and token efficiency (1.47$\times$ improvement).

**arXiv ID:** 2510.20211
</details>

<details>
<summary><strong>Structures generated in a multiagent system performing information fusion in peer-to-peer resource-constrained networks</strong> - Horacio Paggi, Juan A. Lara, Javier Soriano - [[pdf]](https://arxiv.org/pdf/2510.20469)</summary>

**Abstract:** There has recently been a major advance with respect to how information fusion is performed. Information fusion has gone from being conceived as a purely hierarchical procedure, as is the case of traditional military applications, to now being regarded collaboratively, as holonic fusion, which is better suited for civil applications and edge organizations. The above paradigm shift is being boosted as information fusion gains ground in different non-military areas, and human-computer and machine-machine communications, where holarchies, which are more flexible structures than ordinary, static hierarchies, become more widespread. This paper focuses on showing how holonic structures tend to be generated when there are constraints on resources (energy, available messages, time, etc.) for interactions based on a set of fully intercommunicating elements (peers) whose components fuse information as a means of optimizing the impact of vagueness and uncertainty present message exchanges. Holon formation is studied generically based on a multiagent system model, and an example of its possible operation is shown. Holonic structures have a series of advantages, such as adaptability, to sudden changes in the environment or its composition, are somewhat autonomous and are capable of cooperating in order to achieve a common goal. This can be useful when the shortage of resources prevents communications or when the system components start to fail.

**arXiv ID:** 2510.20469
</details>

<details>
<summary><strong>Prover Agent: An Agent-Based Framework for Formal Mathematical Proofs</strong> - Kaito Baba, Chaoran Liu, Shuhei Kurita, Akiyoshi Sannai - [[pdf]](https://arxiv.org/pdf/2506.19923)</summary>

**Abstract:** We present Prover Agent, a novel AI agent for automated theorem proving that integrates large language models (LLMs) with a formal proof assistant, Lean. Prover Agent coordinates an informal reasoning LLM, a formal prover model, and feedback from Lean while also generating auxiliary lemmas. These auxiliary lemmas are not limited to subgoals in the formal proof but can also include special cases or potentially useful facts derived from the assumptions, which help in discovering a viable proof strategy. It achieves an 88.1% success rate on the MiniF2F benchmark, establishing a new state-of-the-art among methods using small language models (SLMs) with a much lower sample budget than previous approaches. We also present theoretical analyses and case studies that illustrate how these generated lemmas contribute to solving challenging problems. Our code is publicly available at: this https URL.

**arXiv ID:** 2506.19923
</details>

<details>
<summary><strong>DeepWideSearch: Benchmarking Depth and Width in Agentic Information Seeking</strong> - Tian Lan, Bin Zhu, Qianghuai Jia, Junyang Ren, Haijun Li, Longyue Wang, Zhao Xu, Weihua Luo, Kaifu Zhang - [[pdf]](https://arxiv.org/pdf/2510.20168)</summary>

**Abstract:** Current search agents fundamentally lack the ability to simultaneously perform \textit{deep} reasoning over multi-hop retrieval and \textit{wide}-scale information collection-a critical deficiency for real-world applications like comprehensive market analysis and business development. To bridge this gap, we introduce DeepWideSearch, the first benchmark explicitly designed to evaluate agents to integrate depth and width in information seeking. In DeepWideSearch, agents must process a large volume of data, each requiring deep reasoning over multi-hop retrieval paths. Specifically, we propose two methods to converse established datasets, resulting in a curated collection of 220 questions spanning 15 diverse domains. Extensive experiments demonstrate that even state-of-the-art agents achieve only 2.39% average success rate on DeepWideSearch, highlighting the substantial challenge of integrating depth and width search in information-seeking tasks. Furthermore, our error analysis reveals four failure modes: lack of reflection, overreliance on internal knowledge, insufficient retrieval, and context overflow-exposing key limitations in current agent architectures. We publicly release DeepWideSearch to catalyze future research on more capable and robust information-seeking agents.

**arXiv ID:** 2510.20168
</details>

<details>
<summary><strong>Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization</strong> - Taeyoon Kwon, Dongwook Choi, Hyojun Kim, Sunghwan Kim, Seungjun Moon, Beong-woo Kwak, Kuan-Hao Huang, Jinyoung Yeo - [[pdf]](https://arxiv.org/pdf/2505.16348)</summary>

**Abstract:** LLM-powered embodied agents have shown success on conventional object-rearrangement tasks, but providing personalized assistance that leverages user-specific knowledge from past interactions presents new challenges. We investigate these challenges through the lens of agents' memory utilization along two critical dimensions: object semantics (identifying objects based on personal meaning) and user patterns (recalling sequences from behavioral routines). To assess these capabilities, we construct MEMENTO, an end-to-end two-stage evaluation framework comprising single-memory and joint-memory tasks. Our experiments reveal that current agents can recall simple object semantics but struggle to apply sequential user patterns to planning. Through in-depth analysis, we identify two critical bottlenecks: information overload and coordination failures when handling multiple memories. Based on these findings, we explore memory architectural approaches to address these challenges. Given our observation that episodic memory provides both personalized knowledge and in-context learning benefits, we design a hierarchical knowledge graph-based user-profile memory module that separately manages personalized knowledge, achieving substantial improvements on both single and joint-memory tasks. Project website: this https URL

**arXiv ID:** 2505.16348
</details>

<details>
<summary><strong>Attention Enhanced Entity Recommendation for Intelligent Monitoring in Cloud Systems</strong> - Fiza Hussain, Anson Bastos, Anjaly Parayil, Ayush Choure, Chetan Bansal, Rujia Wang, Saravan Rajmohan - [[pdf]](https://arxiv.org/pdf/2510.20640)</summary>

**Abstract:** In this paper, we present DiRecGNN, an attention-enhanced entity recommendation framework for monitoring cloud services at Microsoft. We provide insights on the usefulness of this feature as perceived by the cloud service owners and lessons learned from deployment. Specifically, we introduce the problem of recommending the optimal subset of attributes (dimensions) that should be tracked by an automated watchdog (monitor) for cloud services. To begin, we construct the monitor heterogeneous graph at production-scale. The interaction dynamics of these entities are often characterized by limited structural and engagement information, resulting in inferior performance of state-of-the-art approaches. Moreover, traditional methods fail to capture the dependencies between entities spanning a long range due to their homophilic nature. Therefore, we propose an attention-enhanced entity ranking model inspired by transformer architectures. Our model utilizes a multi-head attention mechanism to focus on heterogeneous neighbors and their attributes, and further attends to paths sampled using random walks to capture long-range dependencies. We also employ multi-faceted loss functions to optimize for relevant recommendations while respecting the inherent sparsity of the data. Empirical evaluations demonstrate significant improvements over existing methods, with our model achieving a 43.1% increase in MRR. Furthermore, product teams who consumed these features perceive the feature as useful and rated it 4.5 out of 5.

**arXiv ID:** 2510.20640
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>Learning from Supervision with Semantic and Episodic Memory: A Reflective Approach to Agent Adaptation</strong> - Jackson Hassell, Dan Zhang, Hannah Kim, Tom Mitchell, Estevam Hruschka - [[pdf]](https://arxiv.org/pdf/2510.19897)</summary>

**Abstract:** We investigate how agents built on pretrained large language models can learn target classification functions from labeled examples without parameter updates. While conventional approaches like fine-tuning are often costly, inflexible, and opaque, we propose a memory-augmented framework that leverages both labeled data and LLM-generated critiques. Our framework uses episodic memory to store instance-level critiques-capturing specific past experiences-and semantic memory to distill these into reusable, task-level guidance. Across a diverse set of tasks, incorporating critiques yields up to a 24.8 percent accuracy improvement over retrieval-based (RAG-style) baselines that rely only on labels. Through extensive empirical evaluation, we uncover distinct behavioral differences between OpenAI and opensource models, particularly in how they handle fact-oriented versus preference-based data. To interpret how models respond to different representations of supervision encoded in memory, we introduce a novel metric, suggestibility. This helps explain observed behaviors and illuminates how model characteristics and memory strategies jointly shape learning dynamics. Our findings highlight the promise of memory-driven, reflective learning for building more adaptive and interpretable LLM agents.

**arXiv ID:** 2510.19897
</details>

<details>
<summary><strong>ToolScope: Enhancing LLM Agent Tool Use through Tool Merging and Context-Aware Filtering</strong> - Marianne Menglin Liu, Daniel Garcia, Fjona Parllaku, Vikas Upadhyay, Syed Fahad Allam Shah, Dan Roth - [[pdf]](https://arxiv.org/pdf/2510.20036)</summary>

**Abstract:** Large language model (LLM) agents rely on external tools to solve complex tasks, but real-world toolsets often contain redundant tools with overlapping names and descriptions, introducing ambiguity and reducing selection accuracy. LLMs also face strict input context limits, preventing efficient consideration of large toolsets. To address these challenges, we propose ToolScope, which includes: (1) ToolScopeMerger with Auto-Correction to automatically audit and fix tool merges, reducing redundancy, and (2) ToolScopeRetriever to rank and select only the most relevant tools for each query, compressing toolsets to fit within context limits without sacrificing accuracy. Evaluations on three state-of-the-art LLMs and three open-source tool-use benchmarks show gains of 8.38% to 38.6% in tool selection accuracy, demonstrating ToolScope's effectiveness in enhancing LLM tool use.

**arXiv ID:** 2510.20036
</details>

<details>
<summary><strong>Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design</strong> - Quan Wei, Siliang Zeng, Chenliang Li, William Brown, Oana Frunza, Wei Deng, Anderson Schneider, Yuriy Nevmyvaka, Yang Katie Zhao, Alfredo Garcia, Mingyi Hong - [[pdf]](https://arxiv.org/pdf/2505.11821)</summary>

**Abstract:** This paper investigates Reinforcement Learning (RL) approaches to enhance the reasoning capabilities of Large Language Model (LLM) agents in long-horizon, multi-turn scenarios. Although RL algorithms such as Group Relative Policy Optimization (GRPO) and Proximal Policy Optimization (PPO) have been widely applied to train multi-turn LLM agents, they typically rely only on sparse outcome rewards and lack dense intermediate signals across multiple decision steps, limiting their performance on complex reasoning tasks. To bridge this gap, we present the first systematic study of \textit{turn-level reward design} for multi-turn RL algorithms and agent applications. By integrating turn-level rewards, we extend GRPO and PPO to their respective multi-turn variants, enabling fine-grained credit assignment. We conduct case studies on multi-turn reasoning-augmented search agents, where we carefully design two types of turn-level rewards: verifiable and LLM-as-judge. Our experiments on multi-turn search tasks demonstrate that incorporating well-designed turn-level rewards enables RL algorithms to significantly outperform baseline methods with trajectory-level rewards. Both training and validation reward curves illustrate that our method achieves \textit{greater stability}, \textit{faster convergence}, and \textit{higher accuracy}. Numerical results across diverse question-answering datasets further show that our approach consistently delivers highest answer correctness and 100\% format correctness.

**arXiv ID:** 2505.11821
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (20 papers)</h2></summary>

<details>
<summary><strong>Human-Centered LLM-Agent System for Detecting Anomalous Digital Asset Transactions</strong> - Gyuyeon Na, Minjung Park, Hyeonjeong Cha, Sangmi Chai - [[pdf]](https://arxiv.org/pdf/2510.20102)</summary>

**Abstract:** We present HCLA, a human-centered multi-agent system for anomaly detection in digital asset transactions. The system links three roles: Parsing, Detection, and Explanation, into a conversational workflow that lets non-experts ask questions in natural language, inspect structured analytics, and obtain context-aware rationales. Implemented with an open-source web UI, HCLA translates user intents into a schema for a classical detector (XGBoost in our prototype) and returns narrative explanations grounded in the underlying features. On a labeled Bitcoin mixing dataset (Wasabi Wallet, 2020-2024), the baseline detector reaches strong accuracy, while HCLA adds interpretability and interactive refinement. We describe the architecture, interaction loop, dataset, evaluation protocol, and limitations, and discuss how a human-in-the-loop design improves transparency and trust in financial forensics.

**arXiv ID:** 2510.20102
</details>

<details>
<summary><strong>Mixture-of-Minds: Multi-Agent Reinforcement Learning for Table Understanding</strong> - Yuhang Zhou, Mingrui Zhang, Ke Li, Mingyi Wang, Qiao Liu, Qifei wang, Jiayi Liu, Fei Liu, Serena Li, Weiwi Li, Mingze Gao, Abhishek Kumar, Xiangjun Fan, Zhuokai Zhao, Lizhu Zhang - [[pdf]](https://arxiv.org/pdf/2510.20176)</summary>

**Abstract:** Understanding and reasoning over tables is a critical capability for many real-world applications. Large language models (LLMs) have shown promise on this task, but current approaches remain limited. Fine-tuning based methods strengthen language reasoning; yet they are prone to arithmetic errors and hallucination. In contrast, tool-based methods enable precise table manipulation but rely on rigid schemas and lack semantic understanding. These complementary drawbacks highlight the need for approaches that integrate robust reasoning with reliable table processing. In this work, we propose Mixture-of-Minds, a multi-agent framework that decomposes table reasoning into three specialized roles: planning, coding, and answering. This design enables each agent to focus on a specific aspect of the task while leveraging code execution for precise table manipulation. Building on this workflow, we introduce a self-improvement training framework that employs Monte Carlo Tree Search (MCTS) rollouts to generate pseudo-gold trajectories and optimize agents with reinforcement learning (RL). Extensive experiments show that Mixture-of-Minds delivers substantial gains, reaching 62.13% on TableBench and surpassing OpenAI-o4-mini-high. These results demonstrate the promise of combining structured multi-agent workflows with RL to advance table understanding.

**arXiv ID:** 2510.20176
</details>

<details>
<summary><strong>High-order Interactions Modeling for Interpretable Multi-Agent Q-Learning</strong> - Qinyu Xu, Yuanyang Zhu, Xuefei Wu, Chunlin Chen - [[pdf]](https://arxiv.org/pdf/2510.20218)</summary>

**Abstract:** The ability to model interactions among agents is crucial for effective coordination and understanding their cooperation mechanisms in multi-agent reinforcement learning (MARL). However, previous efforts to model high-order interactions have been primarily hindered by the combinatorial explosion or the opaque nature of their black-box network structures. In this paper, we propose a novel value decomposition framework, called Continued Fraction Q-Learning (QCoFr), which can flexibly capture arbitrary-order agent interactions with only linear complexity $\mathcal{O}\left({n}\right)$ in the number of agents, thus avoiding the combinatorial explosion when modeling rich cooperation. Furthermore, we introduce the variational information bottleneck to extract latent information for estimating credits. This latent information helps agents filter out noisy interactions, thereby significantly enhancing both cooperation and interpretability. Extensive experiments demonstrate that QCoFr not only consistently achieves better performance but also provides interpretability that aligns with our theoretical analysis.

**arXiv ID:** 2510.20218
</details>

<details>
<summary><strong>Balancing Specialization and Centralization: A Multi-Agent Reinforcement Learning Benchmark for Sequential Industrial Control</strong> - Tom Maus, Asma Atamna, Tobias Glasmachers - [[pdf]](https://arxiv.org/pdf/2510.20408)</summary>

**Abstract:** Autonomous control of multi-stage industrial processes requires both local specialization and global coordination. Reinforcement learning (RL) offers a promising approach, but its industrial adoption remains limited due to challenges such as reward design, modularity, and action space management. Many academic benchmarks differ markedly from industrial control problems, limiting their transferability to real-world applications. This study introduces an enhanced industry-inspired benchmark environment that combines tasks from two existing benchmarks, SortingEnv and ContainerGym, into a sequential recycling scenario with sorting and pressing operations. We evaluate two control strategies: a modular architecture with specialized agents and a monolithic agent governing the full system, while also analyzing the impact of action masking. Our experiments show that without action masking, agents struggle to learn effective policies, with the modular architecture performing better. When action masking is applied, both architectures improve substantially, and the performance gap narrows considerably. These results highlight the decisive role of action space constraints and suggest that the advantages of specialization diminish as action complexity is reduced. The proposed benchmark thus provides a valuable testbed for exploring practical and robust multi-agent RL solutions in industrial automation, while contributing to the ongoing debate on centralization versus specialization.

**arXiv ID:** 2510.20408
</details>

<details>
<summary><strong>Co-Designing Quantum Codes with Transversal Diagonal Gates via Multi-Agent Systems</strong> - Xi He, Sirui Lu, Bei Zeng - [[pdf]](https://arxiv.org/pdf/2510.20728)</summary>

**Abstract:** We present a multi-agent, human-in-the-loop workflow that co-designs quantum codes with prescribed transversal diagonal gates. It builds on the Subset-Sum Linear Programming (SSLP) framework (arXiv:2504.20847), which partitions basis strings by modular residues and enforces $Z$-marginal Knill-Laflamme (KL) equalities via small LPs. The workflow is powered by GPT-5 and implemented within TeXRA (this https URL)-a multi-agent research assistant platform that supports an iterative tool-use loop agent and a derivation-then-edit workflow reasoning agent. We work in a LaTeX-Python environment where agents reason, edit documents, execute code, and synchronize their work to Git/Overleaf. Within this workspace, three roles collaborate: a Synthesis Agent formulates the problem; a Search Agent sweeps/screens candidates and exactifies numerics into rationals; and an Audit Agent independently checks all KL equalities and the induced logical action. As a first step we focus on distance $d=2$ with nondegenerate residues. For code dimension $K\in\{2,3,4\}$ and $n\le6$ qubits, systematic sweeps yield certificate-backed tables cataloging attainable cyclic logical groups-all realized by new codes-e.g., for $K=3$ we obtain order $16$ at $n=6$. From verified instances, Synthesis Agent abstracts recurring structures into closed-form families and proves they satisfy the KL equalities for all parameters. It further demonstrates that SSLP accommodates residue degeneracy by exhibiting a new $((6,4,2))$ code implementing the transversal controlled-phase $diag(1,1,1,i)$. Overall, the workflow recasts diagonal-transversal feasibility as an analytical pipeline executed at scale, combining systematic enumeration with exact analytical reconstruction. It yields reproducible code constructions, supports targeted extensions to larger $K$ and higher distances, and leads toward data-driven classification.

**arXiv ID:** 2510.20728
</details>

<details>
<summary><strong>Thought Communication in Multiagent Collaboration</strong> - Yujia Zheng, Zhuokai Zhao, Zijian Li, Yaqi Xie, Mingze Gao, Lizhu Zhang, Kun Zhang - [[pdf]](https://arxiv.org/pdf/2510.20733)</summary>

**Abstract:** Natural language has long enabled human cooperation, but its lossy, ambiguous, and indirect nature limits the potential of collective intelligence. While machines are not subject to these constraints, most LLM-based multi-agent systems still rely solely on natural language, exchanging tokens or their embeddings. To go beyond language, we introduce a new paradigm, thought communication, which enables agents to interact directly mind-to-mind, akin to telepathy. To uncover these latent thoughts in a principled way, we formalize the process as a general latent variable model, where agent states are generated by an unknown function of underlying thoughts. We prove that, in a nonparametric setting without auxiliary information, both shared and private latent thoughts between any pair of agents can be identified. Moreover, the global structure of thought sharing, including which agents share which thoughts and how these relationships are structured, can also be recovered with theoretical guarantees. Guided by the established theory, we develop a framework that extracts latent thoughts from all agents prior to communication and assigns each agent the relevant thoughts, along with their sharing patterns. This paradigm naturally extends beyond LLMs to all modalities, as most observational data arise from hidden generative processes. Experiments on both synthetic and real-world benchmarks validate the theory and demonstrate the collaborative advantages of thought communication. We hope this work illuminates the potential of leveraging the hidden world, as many challenges remain unsolvable through surface-level observation alone, regardless of compute or data scale.

**arXiv ID:** 2510.20733
</details>

<details>
<summary><strong>Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve</strong> - Yuanzhe Liu, Ryan Deng, Tim Kaler, Xuhao Chen, Charles E. Leiserson, Yao Ma, Jie Chen - [[pdf]](https://arxiv.org/pdf/2505.23946)</summary>

**Abstract:** Recent studies show that LLMs possess different skills and specialize in different tasks. In fact, we observe that their varied performance occur in several levels of granularity. For example, in the code optimization task, code LLMs excel at different optimization categories and no one dominates others. This observation prompts the question of how one leverages multiple LLM agents to solve a coding problem without knowing their complementary strengths a priori. We argue that a team of agents can learn from each other's successes and failures so as to improve their own performance. Thus, a lesson is the knowledge produced by an agent and passed on to other agents in the collective solution process. We propose a lesson-based collaboration framework, design the lesson solicitation--banking--selection mechanism, and demonstrate that a team of small LLMs with lessons learned can outperform a much larger LLM and other multi-LLM collaboration methods.

**arXiv ID:** 2505.23946
</details>

<details>
<summary><strong>RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration</strong> - Xiuyuan Chen, Jian Zhao, Yuchen Yuan, Tianle Zhang, Huilin Zhou, Zheng Zhu, Ping Hu, Linghe Kong, Chi Zhang, Weiran Huang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2509.25271)</summary>

**Abstract:** Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87% improvement in risk identification accuracy compared to the strongest baseline evaluation method.

**arXiv ID:** 2509.25271
</details>

<details>
<summary><strong>A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning</strong> - Anjie Liu, Jianhong Wang, Samuel Kaski, Jun Wang, Mengyue Yang - [[pdf]](https://arxiv.org/pdf/2510.17697)</summary>

**Abstract:** Steering cooperative multi-agent reinforcement learning (MARL) towards desired outcomes is challenging, particularly when the global guidance from a human on the whole multi-agent system is impractical in a large-scale MARL. On the other hand, designing external mechanisms (e.g., intrinsic rewards and human feedback) to coordinate agents mostly relies on empirical studies, lacking a easy-to-use research tool. In this work, we employ multi-agent influence diagrams (MAIDs) as a graphical framework to address the above issues. First, we introduce the concept of MARL interaction paradigms, using MAIDs to analyze and visualize both unguided self-organization and global guidance mechanisms in MARL. Then, we design a new MARL interaction paradigm, referred to as the targeted intervention paradigm that is applied to only a single targeted agent, so the problem of global guidance can be mitigated. In our implementation, we introduce a causal inference technique, referred to as Pre-Strategy Intervention (PSI), to realize the targeted intervention paradigm. Since MAIDs can be regarded as a special class of causal diagrams, a composite desired outcome that integrates the primary task goal and an additional desired outcome can be achieved by maximizing the corresponding causal effect through the PSI. Moreover, the bundled relevance graph analysis of MAIDs provides a tool to identify whether an MARL learning paradigm is workable under the design of an MARL interaction paradigm. In experiments, we demonstrate the effectiveness of our proposed targeted intervention, and verify the result of relevance graph analysis.

**arXiv ID:** 2510.17697
</details>

<details>
<summary><strong>Communication to Completion: Modeling Collaborative Workflows with Intelligent Multi-Agent Communication</strong> - Yiming Lu, Xun Wang, Simin Ma, Shujian Liu, Sathish Reddy Indurthi, Song Wang, Haoyun Deng, Fei Liu, Kaiqiang Song - [[pdf]](https://arxiv.org/pdf/2510.19995)</summary>

**Abstract:** Teamwork in workspace for complex tasks requires diverse communication strategies, but current multi-agent LLM systems lack systematic frameworks for task oriented communication. We introduce Communication to Completion (C2C), a scalable framework that addresses this gap through two key innovations: (1) the Alignment Factor (AF), a novel metric quantifying agent task alignment that directly impacts work efficiency, and (2) a Sequential Action Framework that integrates stepwise execution with intelligent communication decisions. C2C enables agents to make cost aware communication choices, dynamically improving task understanding through targeted interactions. We evaluated C2C on realistic coding workflows across three complexity tiers and team sizes from 5 to 17 agents, comparing against no communication and fixed steps baselines. The results show that C2C reduces the task completion time by about 40% with acceptable communication costs. The framework completes all tasks successfully in standard configurations and maintains effectiveness at scale. C2C establishes both a theoretical foundation for measuring communication effectiveness in multi-agent systems and a practical framework for complex collaborative tasks.

**arXiv ID:** 2510.19995
</details>

<details>
<summary><strong>Beyond Static Responses: Multi-Agent LLM Systems as a New Paradigm for Social Science Research</strong> - Jennifer Haase, Sebastian Pokutta - [[pdf]](https://arxiv.org/pdf/2506.01839)</summary>

**Abstract:** As large language models (LLMs) transition from static tools to fully agentic systems, their potential for transforming social science research has become increasingly evident. This paper introduces a structured framework for understanding the diverse applications of LLM-based agents, ranging from simple data processors to complex, multi-agent systems capable of simulating emergent social dynamics. By mapping this developmental continuum across six levels, the paper clarifies the technical and methodological boundaries between different agentic architectures, providing a comprehensive overview of current capabilities and future potential. It highlights how lower-tier systems streamline conventional tasks like text classification and data annotation, while higher-tier systems enable novel forms of inquiry, including the study of group dynamics, norm formation, and large-scale social processes. However, these advancements also introduce significant challenges, including issues of reproducibility, ethical oversight, and the risk of emergent biases. The paper critically examines these concerns, emphasizing the need for robust validation protocols, interdisciplinary collaboration, and standardized evaluation metrics. It argues that while LLM-based agents hold transformative potential for the social sciences, realizing this promise will require careful, context-sensitive deployment and ongoing methodological refinement. The paper concludes with a call for future research that balances technical innovation with ethical responsibility, encouraging the development of agentic systems that not only replicate but also extend the frontiers of social science, offering new insights into the complexities of human behavior.

**arXiv ID:** 2506.01839
</details>

<details>
<summary><strong>Evolution of Cooperation in LLM-Agent Societies: A Preliminary Study Using Different Punishment Strategies</strong> - Kavindu Warnakulasuriya, Prabhash Dissanayake, Navindu De Silva, Stephen Cranefield, Bastin Tony Roy Savarimuthu, Surangika Ranathunga, Nisansa de Silva - [[pdf]](https://arxiv.org/pdf/2504.19487)</summary>

**Abstract:** The evolution of cooperation has been extensively studied using abstract mathematical models and simulations. Recent advances in Large Language Models (LLMs) and the rise of LLM agents have demonstrated their ability to perform social reasoning, thus providing an opportunity to test the emergence of norms in more realistic agent-based simulations with human-like reasoning using natural language. In this research, we investigate whether the cooperation dynamics presented in Boyd and Richerson's model persist in a more realistic simulation of the Diner's Dilemma using LLM agents compared to the abstract mathematical nature in the work of Boyd and Richerson. Our findings indicate that agents follow the strategies defined in the Boyd and Richerson model, and explicit punishment mechanisms drive norm emergence, reinforcing cooperative behaviour even when the agent strategy configuration varies. Our results suggest that LLM-based Multi-Agent System simulations, in fact, can replicate the evolution of cooperation predicted by the traditional mathematical models. Moreover, our simulations extend beyond the mathematical models by integrating natural language-driven reasoning and a pairwise imitation method for strategy adoption, making them a more realistic testbed for cooperative behaviour in MASs.

**arXiv ID:** 2504.19487
</details>

<details>
<summary><strong>SafeDiver: Cooperative AUV-USV Assisted Diver Communication via Multi-agent Reinforcement Learning Approach</strong> - Tinglong Deng, Hang Tao, Xinxiang Wang, Yinyan Wang, Hanjiang Luo - [[pdf]](https://arxiv.org/pdf/2509.11508)</summary>

**Abstract:** As underwater human activities are increasing, the demand for underwater communication service presents a significant challenge. Existing underwater diver communication methods face hurdles due to inherent disadvantages and complex underwater environments. To address this issue, we propose a scheme that utilizes maritime unmanned systems to assist divers with reliable and high-speed communication. Multiple AUVs are equipped with optical and acoustic multimodal communication devices as relay nodes, providing adaptive communication services based on changes in the diver's activity area. By using a multi-agent reinforcement learning (MARL) approach to control the cooperative movement of AUVs, high-speed and reliable data transmission between divers can be achieved. At the same time, utilizing the advantages of on-demand deployment and wide coverage of unmanned surface vehicles (USVs) as surface relay nodes to coordinate and forward information from AUVs, and controlling AUVs to adaptively select relay USV nodes for data transmission, high-quality communication between divers and surface platform can be achieved. Through simulation verification, the proposed scheme can effectively achieve reliable and high-speed communication for divers.

**arXiv ID:** 2509.11508
</details>

<details>
<summary><strong>Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning</strong> - Simin Li, Zihao Mao, Hanxiao Li, Zonglei Jing, Zhuohang bian, Jun Guo, Li Wang, Zhuoran Han, Ruixiao Xu, Xin Yu, Chengdong Ma, Yuqing Ma, Bo An, Yaodong Yang, Weifeng Lv, Xianglong Liu - [[pdf]](https://arxiv.org/pdf/2510.11824)</summary>

**Abstract:** In cooperative Multi-Agent Reinforcement Learning (MARL), it is a common practice to tune hyperparameters in ideal simulated environments to maximize cooperative performance. However, policies tuned for cooperation often fail to maintain robustness and resilience under real-world uncertainties. Building trustworthy MARL systems requires a deep understanding of robustness, which ensures stability under uncertainties, and resilience, the ability to recover from disruptions--a concept extensively studied in control systems but largely overlooked in MARL. In this paper, we present a large-scale empirical study comprising over 82,620 experiments to evaluate cooperation, robustness, and resilience in MARL across 4 real-world environments, 13 uncertainty types, and 15 hyperparameters. Our key findings are: (1) Under mild uncertainty, optimizing cooperation improves robustness and resilience, but this link weakens as perturbations intensify. Robustness and resilience also varies by algorithm and uncertainty type. (2) Robustness and resilience do not generalize across uncertainty modalities or agent scopes: policies robust to action noise for all agents may fail under observation noise on a single agent. (3) Hyperparameter tuning is critical for trustworthy MARL: surprisingly, standard practices like parameter sharing, GAE, and PopArt can hurt robustness, while early stopping, high critic learning rates, and Leaky ReLU consistently help. By optimizing hyperparameters only, we observe substantial improvement in cooperation, robustness and resilience across all MARL backbones, with the phenomenon also generalizing to robust MARL methods across these backbones. Code and results available at this https URL .

**arXiv ID:** 2510.11824
</details>

<details>
<summary><strong>Local Guidance for Configuration-Based Multi-Agent Pathfinding</strong> - Tomoki Arita, Keisuke Okumura - [[pdf]](https://arxiv.org/pdf/2510.19072)</summary>

**Abstract:** Guidance is an emerging concept that improves the empirical performance of real-time, sub-optimal multi-agent pathfinding (MAPF) methods. It offers additional information to MAPF algorithms to mitigate congestion on a global scale by considering the collective behavior of all agents across the entire workspace. This global perspective helps reduce agents' waiting times, thereby improving overall coordination efficiency. In contrast, this study explores an alternative approach: providing local guidance in the vicinity of each agent. While such localized methods involve recomputation as agents move and may appear computationally demanding, we empirically demonstrate that supplying informative spatiotemporal cues to the planner can significantly improve solution quality without exceeding a moderate time budget. When applied to LaCAM, a leading configuration-based solver, this form of guidance establishes a new performance frontier for MAPF.

**arXiv ID:** 2510.19072
</details>

<details>
<summary><strong>Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?</strong> - Hyeong Kyu Choi, Xiaojin Zhu, Sharon Li - [[pdf]](https://arxiv.org/pdf/2508.17536)</summary>

**Abstract:** Multi-Agent Debate~(MAD) has emerged as a promising paradigm for improving the performance of large language models through collaborative reasoning. Despite recent advances, the key factors driving MAD's effectiveness remain unclear. In this work, we disentangle MAD into two key components--Majority Voting and inter-agent Debate--and assess their respective contributions. Through extensive experiments across seven NLP benchmarks, we find that Majority Voting alone accounts for most of the performance gains typically attributed to MAD. To explain this, we propose a theoretical framework that models debate as a stochastic process. We prove that it induces a martingale over agents' belief trajectories, implying that debate alone does not improve expected correctness. Guided by these insights, we demonstrate that targeted interventions, by biasing the belief update toward correction, can meaningfully enhance debate effectiveness. Overall, our findings suggest that while MAD has potential, simple ensembling methods remain strong and more reliable alternatives in many practical settings. Code is released in this https URL.

**arXiv ID:** 2508.17536
</details>

<details>
<summary><strong>Beyond Retrieval-Ranking: A Multi-Agent Cognitive Decision Framework for E-Commerce Search</strong> - Zhouwei Zhai, Mengxiang Chen, Haoyun Xia, Jin Li, Renquan Zhou, Min Yang - [[pdf]](https://arxiv.org/pdf/2510.20567)</summary>

**Abstract:** The retrieval-ranking paradigm has long dominated e-commerce search, but its reliance on query-item matching fundamentally misaligns with multi-stage cognitive decision processes of platform users. This misalignment introduces critical limitations: semantic gaps in complex queries, high decision costs due to cross-platform information foraging, and the absence of professional shopping guidance. To address these issues, we propose a Multi-Agent Cognitive Decision Framework (MACDF), which shifts the paradigm from passive retrieval to proactive decision support. Extensive offline evaluations demonstrate MACDF's significant improvements in recommendation accuracy and user satisfaction, particularly for complex queries involving negation, multi-constraint, or reasoning demands. Online A/B testing on JD search platform confirms its practical efficacy. This work highlights the transformative potential of multi-agent cognitive systems in redefining e-commerce search.

**arXiv ID:** 2510.20567
</details>

<details>
<summary><strong>A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System</strong> - Jiale Guo, Suizhi Huang, Mei Li, Dong Huang, Xingsheng Chen, Regina Zhang, Zhijiang Guo, Han Yu, Siu-Ming Yiu, Pietro Lio, Kwok-Yan Lam - [[pdf]](https://arxiv.org/pdf/2510.09721)</summary>

**Abstract:** The integration of Large Language Models (LLMs) into software engineering has driven a transition from traditional rule-based systems to autonomous agentic systems capable of solving complex problems. However, systematic progress is hindered by a lack of comprehensive understanding of how benchmarks and solutions interconnect. This survey addresses this gap by providing the first holistic analysis of LLM-powered software engineering, offering insights into evaluation methodologies and solution paradigms. We review over 150 recent papers and propose a taxonomy along two key dimensions: (1) Solutions, categorized into prompt-based, fine-tuning-based, and agent-based paradigms, and (2) Benchmarks, including tasks such as code generation, translation, and repair. Our analysis highlights the evolution from simple prompt engineering to sophisticated agentic systems incorporating capabilities like planning, reasoning, memory mechanisms, and tool augmentation. To contextualize this progress, we present a unified pipeline illustrating the workflow from task specification to deliverables, detailing how different solution paradigms address various complexity levels. Unlike prior surveys that focus narrowly on specific aspects, this work connects 50+ benchmarks to their corresponding solution strategies, enabling researchers to identify optimal approaches for diverse evaluation criteria. We also identify critical research gaps and propose future directions, including multi-agent collaboration, self-evolving systems, and formal verification integration. This survey serves as a foundational guide for advancing LLM-driven software engineering. We maintain a GitHub repository that continuously updates the reviewed and related papers at this https URL.

**arXiv ID:** 2510.09721
</details>

<details>
<summary><strong>ComProScanner: A multi-agent based framework for composition-property structured data extraction from scientific literature</strong> - Aritra Roy, Enrico Grisan, John Buckeridge, Chiara Gattinoni - [[pdf]](https://arxiv.org/pdf/2510.20362)</summary>

**Abstract:** Since the advent of various pre-trained large language models, extracting structured knowledge from scientific text has experienced a revolutionary change compared with traditional machine learning or natural language processing techniques. Despite these advances, accessible automated tools that allow users to construct, validate, and visualise datasets from scientific literature extraction remain scarce. We therefore developed ComProScanner, an autonomous multi-agent platform that facilitates the extraction, validation, classification, and visualisation of machine-readable chemical compositions and properties, integrated with synthesis data from journal articles for comprehensive database creation. We evaluated our framework using 100 journal articles against 10 different LLMs, including both open-source and proprietary models, to extract highly complex compositions associated with ceramic piezoelectric materials and corresponding piezoelectric strain coefficients (d33), motivated by the lack of a large dataset for such materials. DeepSeek-V3-0324 outperformed all models with a significant overall accuracy of 0.82. This framework provides a simple, user-friendly, readily-usable package for extracting highly complex experimental data buried in the literature to build machine learning or deep learning datasets.

**arXiv ID:** 2510.20362
</details>

<details>
<summary><strong>Learning Decentralized Routing Policies via Graph Attention-based Multi-Agent Reinforcement Learning in Lunar Delay-Tolerant Networks</strong> - Federico Lozano-Cuadra, Beatriz Soret, Marc Sanchez Net, Abhishek Cauligi, Federico Rossi - [[pdf]](https://arxiv.org/pdf/2510.20436)</summary>

**Abstract:** We present a fully decentralized routing framework for multi-robot exploration missions operating under the constraints of a Lunar Delay-Tolerant Network (LDTN). In this setting, autonomous rovers must relay collected data to a lander under intermittent connectivity and unknown mobility patterns. We formulate the problem as a Partially Observable Markov Decision Problem (POMDP) and propose a Graph Attention-based Multi-Agent Reinforcement Learning (GAT-MARL) policy that performs Centralized Training, Decentralized Execution (CTDE). Our method relies only on local observations and does not require global topology updates or packet replication, unlike classical approaches such as shortest path and controlled flooding-based algorithms. Through Monte Carlo simulations in randomized exploration environments, GAT-MARL provides higher delivery rates, no duplications, and fewer packet losses, and is able to leverage short-term mobility forecasts; offering a scalable solution for future space robotic systems for planetary exploration, as demonstrated by successful generalization to larger rover teams.

**arXiv ID:** 2510.20436
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>CourtGuard: A Local, Multiagent Prompt Injection Classifier</strong> - Isaac Wu, Michael Maslowski - [[pdf]](https://arxiv.org/pdf/2510.19844)</summary>

**Abstract:** As large language models (LLMs) become integrated into various sensitive applications, prompt injection, the use of prompting to induce harmful behaviors from LLMs, poses an ever increasing risk. Prompt injection attacks can cause LLMs to leak sensitive data, spread misinformation, and exhibit harmful behaviors. To defend against these attacks, we propose CourtGuard, a locally-runnable, multiagent prompt injection classifier. In it, prompts are evaluated in a court-like multiagent LLM system, where a "defense attorney" model argues the prompt is benign, a "prosecution attorney" model argues the prompt is a prompt injection, and a "judge" model gives the final classification. CourtGuard has a lower false positive rate than the Direct Detector, an LLM as-a-judge. However, CourtGuard is generally a worse prompt injection detector. Nevertheless, this lower false positive rate highlights the importance of considering both adversarial and benign scenarios for the classification of a prompt. Additionally, the relative performance of CourtGuard in comparison to other prompt injection classifiers advances the use of multiagent systems as a defense against prompt injection attacks. The implementations of CourtGuard and the Direct Detector with full prompts for Gemma-3-12b-it, Llama-3.3-8B, and Phi-4-mini-instruct are available at this https URL.

**arXiv ID:** 2510.19844
</details>

<details>
<summary><strong>A Tutorial on Cognitive Biases in Agentic AI-Driven 6G Autonomous Networks</strong> - Hatim Chergui, Farhad Rezazadeh, Merouane Debbah, Christos Verikoukis - [[pdf]](https://arxiv.org/pdf/2510.19973)</summary>

**Abstract:** The path to higher network autonomy in 6G lies beyond the mere optimization of key performance indicators (KPIs). While KPIs have enabled automation gains under TM Forum Levels 1--3, they remain numerical abstractions that act only as proxies for the real essence of communication networks: seamless connectivity, fairness, adaptability, and resilience. True autonomy requires perceiving and reasoning over the network environment as it is. Such progress can be achieved through \emph{agentic AI}, where large language model (LLM)-powered agents perceive multimodal telemetry, reason with memory, negotiate across domains, and act via APIs to achieve multi-objective goals. However, deploying such agents introduces the challenge of cognitive biases inherited from human design, which can distort reasoning, negotiation, tool use, and actuation. Between neuroscience and AI, this paper provides a tutorial on a selection of well-known biases, including their taxonomy, definition, mathematical formulation, emergence in telecom systems and the commonly impacted agentic components. The tutorial also presents various mitigation strategies tailored to each type of bias. The article finally provides two practical use-cases, which tackle the emergence, impact and mitigation gain of some famous biases in 6G inter-slice and cross-domain management. In particular, anchor randomization, temporal decay and inflection bonus techniques are introduced to specifically address anchoring, temporal and confirmation biases. This avoids that agents stick to the initial high resource allocation proposal or decisions that are recent and/or confirming a prior hypothesis. By grounding decisions in a richer and fairer set of past experiences, the quality and bravery of the agentic agreements in the second use-case, for instance, are leading to $\times 5$ lower latency and around $40\%$ higher energy saving.

**arXiv ID:** 2510.19973
</details>

<details>
<summary><strong>MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning</strong> - Mircea Lică, Ojas Shirekar, Baptiste Colle, Chirag Raman - [[pdf]](https://arxiv.org/pdf/2411.12977)</summary>

**Abstract:** Embodied agents powered by large language models (LLMs), such as Voyager, promise open-ended competence in worlds such as Minecraft. However, when powered by open-weight LLMs they still falter on elementary tasks after domain-specific fine-tuning. We propose MindForge, a generative-agent framework for cultural lifelong learning through explicit perspective taking. We introduce three key innovations: (1) a structured theory of mind representation linking percepts, beliefs, desires, and actions; (2) natural inter-agent communication; and (3) a multi-component memory system. Following the cultural learning framework, we test MindForge in both instructive and collaborative settings within Minecraft. In an instructive setting with GPT-4, MindForge agents powered by open-weight LLMs significantly outperform their Voyager counterparts in basic tasks yielding $3\times$ more tech-tree milestones and collecting $2.3\times$ more unique items than the Voyager baseline. Furthermore, in fully \textit{collaborative} settings, we find that the performance of two underachieving agents improves with more communication rounds, echoing the Condorcet Jury Theorem. MindForge agents demonstrate sophisticated behaviors, including expert-novice knowledge transfer, collaborative problem solving, and adaptation to out-of-distribution tasks through accumulated cultural experiences.

**arXiv ID:** 2411.12977
</details>

<details>
<summary><strong>Adaptive Learning in Spatial Agent-Based Models for Climate Risk Assessment: A Geospatial Framework with Evolutionary Economic Agents</strong> - Yara Mohajerani - [[pdf]](https://arxiv.org/pdf/2509.18633)</summary>

**Abstract:** Climate risk assessment requires modelling complex interactions between spatially heterogeneous hazards and adaptive economic systems. We present a novel geospatial agent-based model that integrates climate hazard data with evolutionary learning for economic agents. Our framework combines Mesa-based spatial modelling with CLIMADA climate impact assessment, introducing adaptive learning behaviours that allow firms to evolve strategies for budget allocation, pricing, wages, and risk adaptation through fitness-based selection and mutation. We demonstrate the framework using riverine flood projections under RCP8.5 until 2100, showing that evolutionary adaptation enables firms to converge with baseline (no hazard) production levels after decades of disruption due to climate stress. Our results reveal systemic risks where even agents that are not directly exposed to floods face impacts through supply chain disruptions, with the end-of-century average price of goods 5.6% higher under RCP8.5 compared to the baseline in our illustrative economic network. This open-source framework provides financial institutions and companies with tools to quantify both direct and cascading climate risks while evaluating cost-effective adaptation strategies.

**arXiv ID:** 2509.18633
</details>

<details>
<summary><strong>From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era</strong> - Wonil Kim, Hyeongseok Wi, Seungsoon Park, Taejun Kim, Sangeun Keum, Keunhyoung Kim, Taewan Kim, Jongmin Jung, Taehyoung Kim, Gaetan Guerrero, Mael Le Goff, Julie Po, Dongjoo Moon, Juhan Nam, Jongpil Lee - [[pdf]](https://arxiv.org/pdf/2510.20276)</summary>

**Abstract:** Generative AI is reshaping music creation, but its rapid growth exposes structural gaps in attribution, rights management, and economic models. Unlike past media shifts, from live performance to recordings, downloads, and streaming, AI transforms the entire lifecycle of music, collapsing boundaries between creation, distribution, and monetization. However, existing streaming systems, with opaque and concentrated royalty flows, are ill-equipped to handle the scale and complexity of AI-driven production. We propose a content-based Music AI Agent architecture that embeds attribution directly into the creative workflow through block-level retrieval and agentic orchestration. Designed for iterative, session-based interaction, the system organizes music into granular components (Blocks) stored in BlockDB; each use triggers an Attribution Layer event for transparent provenance and real-time settlement. This framework reframes AI from a generative tool into infrastructure for a Fair AI Media Platform. By enabling fine-grained attribution, equitable compensation, and participatory engagement, it points toward a post-streaming paradigm where music functions not as a static catalog but as a collaborative and adaptive ecosystem.

**arXiv ID:** 2510.20276
</details>

<details>
<summary><strong>Dino-Diffusion Modular Designs Bridge the Cross-Domain Gap in Autonomous Parking</strong> - Zixuan Wu, Hengyuan Zhang, Ting-Hsuan Chen, Yuliang Guo, David Paz, Xinyu Huang, Liu Ren - [[pdf]](https://arxiv.org/pdf/2510.20335)</summary>

**Abstract:** Parking is a critical pillar of driving safety. While recent end-to-end (E2E) approaches have achieved promising in-domain results, robustness under domain shifts (e.g., weather and lighting changes) remains a key challenge. Rather than relying on additional data, in this paper, we propose Dino-Diffusion Parking (DDP), a domain-agnostic autonomous parking pipeline that integrates visual foundation models with diffusion-based planning to enable generalized perception and robust motion planning under distribution shifts. We train our pipeline in CARLA at regular setting and transfer it to more adversarial settings in a zero-shot fashion. Our model consistently achieves a parking success rate above 90% across all tested out-of-distribution (OOD) scenarios, with ablation studies confirming that both the network architecture and algorithmic design significantly enhance cross-domain performance over existing baselines. Furthermore, testing in a 3D Gaussian splatting (3DGS) environment reconstructed from a real-world parking lot demonstrates promising sim-to-real transfer.

**arXiv ID:** 2510.20335
</details>

<details>
<summary><strong>Degradation-Aware Cooperative Multi-Modal GNSS-Denied Localization Leveraging LiDAR-Based Robot Detections</strong> - Václav Pritzl, Xianjia Yu, Tomi Westerlund, Petr Štěpán, Martin Saska - [[pdf]](https://arxiv.org/pdf/2510.20480)</summary>

**Abstract:** Accurate long-term localization using onboard sensors is crucial for robots operating in Global Navigation Satellite System (GNSS)-denied environments. While complementary sensors mitigate individual degradations, carrying all the available sensor types on a single robot significantly increases the size, weight, and power demands. Distributing sensors across multiple robots enhances the deployability but introduces challenges in fusing asynchronous, multi-modal data from independently moving platforms. We propose a novel adaptive multi-modal multi-robot cooperative localization approach using a factor-graph formulation to fuse asynchronous Visual-Inertial Odometry (VIO), LiDAR-Inertial Odometry (LIO), and 3D inter-robot detections from distinct robots in a loosely-coupled fashion. The approach adapts to changing conditions, leveraging reliable data to assist robots affected by sensory degradations. A novel interpolation-based factor enables fusion of the unsynchronized measurements. LIO degradations are evaluated based on the approximate scan-matching Hessian. A novel approach of weighting odometry data proportionally to the Wasserstein distance between the consecutive VIO outputs is proposed. A theoretical analysis is provided, investigating the cooperative localization problem under various conditions, mainly in the presence of sensory degradations. The proposed method has been extensively evaluated on real-world data gathered with heterogeneous teams of an Unmanned Ground Vehicle (UGV) and Unmanned Aerial Vehicles (UAVs), showing that the approach provides significant improvements in localization accuracy in the presence of various sensory degradations.

**arXiv ID:** 2510.20480
</details>

<details>
<summary><strong>Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey</strong> - Weifan Guan, Qinghao Hu, Aosheng Li, Jian Cheng - [[pdf]](https://arxiv.org/pdf/2510.17111)</summary>

**Abstract:** Vision-Language-Action (VLA) models extend vision-language models to embodied control by mapping natural-language instructions and visual observations to robot actions. Despite their capabilities, VLA systems face significant challenges due to their massive computational and memory demands, which conflict with the constraints of edge platforms such as on-board mobile manipulators that require real-time performance. Addressing this tension has become a central focus of recent research. In light of the growing efforts toward more efficient and scalable VLA systems, this survey provides a systematic review of approaches for improving VLA efficiency, with an emphasis on reducing latency, memory footprint, and training and inference costs. We categorize existing solutions into four dimensions: model architecture, perception feature, action generation, and training/inference strategies, summarizing representative techniques within each category. Finally, we discuss future trends and open challenges, highlighting directions for advancing efficient embodied intelligence.

**arXiv ID:** 2510.17111
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (30 papers)</h2></summary>

<details>
<summary><strong>Integrating Machine Learning into Belief-Desire-Intention Agents: Current Advances and Open Challenges</strong> - Andrea Agiollo, Andrea Omicini - [[pdf]](https://arxiv.org/pdf/2510.20641)</summary>

**Abstract:** Thanks to the remarkable human-like capabilities of machine learning (ML) models in perceptual and cognitive tasks, frameworks integrating ML within rational agent architectures are gaining traction. Yet, the landscape remains fragmented and incoherent, often focusing on embedding ML into generic agent containers while overlooking the expressive power of rational architectures--such as Belief-Desire-Intention (BDI) agents. This paper presents a fine-grained systematisation of existing approaches, using the BDI paradigm as a reference. Our analysis illustrates the fast-evolving literature on rational agents enhanced by ML, and identifies key research opportunities and open challenges for designing effective rational ML agents.

**arXiv ID:** 2510.20641
</details>

<details>
<summary><strong>Plan Then Retrieve: Reinforcement Learning-Guided Complex Reasoning over Knowledge Graphs</strong> - Yanlin Song, Ben Liu, Víctor Gutiérrez-Basulto, Zhiwei Hu, Qianqian Xie, Min Peng, Sophia Ananiadou, Jeff Z. Pan - [[pdf]](https://arxiv.org/pdf/2510.20691)</summary>

**Abstract:** Knowledge Graph Question Answering aims to answer natural language questions by reasoning over structured knowledge graphs. While large language models have advanced KGQA through their strong reasoning capabilities, existing methods continue to struggle to fully exploit both the rich knowledge encoded in KGs and the reasoning capabilities of LLMs, particularly in complex scenarios. They often assume complete KG coverage and lack mechanisms to judge when external information is needed, and their reasoning remains locally myopic, failing to maintain coherent multi-step planning, leading to reasoning failures even when relevant knowledge exists. We propose Graph-RFT, a novel two-stage reinforcement fine-tuning KGQA framework with a 'plan-KGsearch-and-Websearch-during-think' paradigm, that enables LLMs to perform autonomous planning and adaptive retrieval scheduling across KG and web sources under incomplete knowledge conditions. Graph-RFT introduces a chain-of-thought fine-tuning method with a customized plan-retrieval dataset activates structured reasoning and resolves the GRPO cold-start problem. It then introduces a novel plan-retrieval guided reinforcement learning process integrates explicit planning and retrieval actions with a multi-reward design, enabling coverage-aware retrieval scheduling. It employs a Cartesian-inspired planning module to decompose complex questions into ordered subquestions, and logical expression to guide tool invocation for globally consistent multi-step reasoning. This reasoning retrieval process is optimized with a multi-reward combining outcome and retrieval specific signals, enabling the model to learn when and how to combine KG and web retrieval effectively.

**arXiv ID:** 2510.20691
</details>

<details>
<summary><strong>Robust Reinforcement Learning in Finance: Modeling Market Impact with Elliptic Uncertainty Sets</strong> - Shaocong Ma, Heng Huang - [[pdf]](https://arxiv.org/pdf/2510.19950)</summary>

**Abstract:** In financial applications, reinforcement learning (RL) agents are commonly trained on historical data, where their actions do not influence prices. However, during deployment, these agents trade in live markets where their own transactions can shift asset prices, a phenomenon known as market impact. This mismatch between training and deployment environments can significantly degrade performance. Traditional robust RL approaches address this model misspecification by optimizing the worst-case performance over a set of uncertainties, but typically rely on symmetric structures that fail to capture the directional nature of market impact. To address this issue, we develop a novel class of elliptic uncertainty sets. We establish both implicit and explicit closed-form solutions for the worst-case uncertainty under these sets, enabling efficient and tractable robust policy evaluation. Experiments on single-asset and multi-asset trading tasks demonstrate that our method achieves superior Sharpe ratio and remains robust under increasing trade volumes, offering a more faithful and scalable approach to RL in financial markets.

**arXiv ID:** 2510.19950
</details>

<details>
<summary><strong>LyriCAR: A Difficulty-Aware Curriculum Reinforcement Learning Framework For Controllable Lyric Translation</strong> - Le Ren, Xiangjian Zeng, Qingqiang Wu, Ruoxuan Liang - [[pdf]](https://arxiv.org/pdf/2510.19967)</summary>

**Abstract:** Lyric translation is a challenging task that requires balancing multiple musical constraints. Existing methods often rely on hand-crafted rules and sentence-level modeling, which restrict their ability to internalize musical-linguistic patterns and to generalize effectively at the paragraph level, where cross-line coherence and global rhyme are crucial. In this work, we propose LyriCAR, a novel framework for controllable lyric translation that operates in a fully unsupervised manner. LyriCAR introduces a difficulty-aware curriculum designer and an adaptive curriculum strategy, ensuring efficient allocation of training resources, accelerating convergence, and improving overall translation quality by guiding the model with increasingly complex challenges. Extensive experiments on the EN-ZH lyric translation task show that LyriCAR achieves state-of-the-art results across both standard translation metrics and multi-dimensional reward scores, surpassing strong baselines. Notably, the adaptive curriculum strategy reduces training steps by nearly 40% while maintaining superior performance. Code, data and model can be accessed at this https URL.

**arXiv ID:** 2510.19967
</details>

<details>
<summary><strong>Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach</strong> - Woohyeon Byeon, Giseung Park, Jongseong Chae, Amir Leshem, Youngchul Sung - [[pdf]](https://arxiv.org/pdf/2510.20235)</summary>

**Abstract:** In this paper, we propose a provably convergent and practical framework for multi-objective reinforcement learning with max-min criterion. From a game-theoretic perspective, we reformulate max-min multi-objective reinforcement learning as a two-player zero-sum regularized continuous game and introduce an efficient algorithm based on mirror descent. Our approach simplifies the policy update while ensuring global last-iterate convergence. We provide a comprehensive theoretical analysis on our algorithm, including iteration complexity under both exact and approximate policy evaluations, as well as sample complexity bounds. To further enhance performance, we modify the proposed algorithm with adaptive regularization. Our experiments demonstrate the convergence behavior of the proposed algorithm in tabular settings, and our implementation for deep reinforcement learning significantly outperforms previous baselines in many MORL environments.

**arXiv ID:** 2510.20235
</details>

<details>
<summary><strong>Towards AI Agents for Course Instruction in Higher Education: Early Experiences from the Field</strong> - Yogesh Simmhan, Varad Kulkarni - [[pdf]](https://arxiv.org/pdf/2510.20255)</summary>

**Abstract:** This article presents early findings from designing, deploying and evaluating an AI-based educational agent deployed as the primary instructor in a graduate-level Cloud Computing course at IISc. We detail the design of a Large Language Model (LLM)-driven Instructor Agent, and introduce a pedagogical framework that integrates the Instructor Agent into the course workflow for actively interacting with the students for content delivery, supplemented by the human instructor to offer the course structure and undertake question--answer sessions. We also propose an analytical framework that evaluates the Agent--Student interaction transcripts using interpretable engagement metrics of topic coverage, topic depth and turn-level elaboration. We report early experiences on how students interact with the Agent to explore concepts, clarify doubts and sustain inquiry-driven dialogue during live classroom sessions. We also report preliminary analysis on our evaluation metrics applied across two successive instructional modules that reveals patterns of engagement evolution, transitioning from broad conceptual exploration to deeper, focused inquiry. These demonstrate how structured integration of conversational AI agents can foster reflective learning, offer a reproducible methodology for studying engagement in authentic classroom settings, and support scalable, high-quality higher education.

**arXiv ID:** 2510.20255
</details>

<details>
<summary><strong>Enhancing Security in Deep Reinforcement Learning: A Comprehensive Survey on Adversarial Attacks and Defenses</strong> - Wu Yichao, Wang Yirui, Ding Panpan, Wang Hailong, Zhu Bingqian, Liu Chun - [[pdf]](https://arxiv.org/pdf/2510.20314)</summary>

**Abstract:** With the wide application of deep reinforcement learning (DRL) techniques in complex fields such as autonomous driving, intelligent manufacturing, and smart healthcare, how to improve its security and robustness in dynamic and changeable environments has become a core issue in current research. Especially in the face of adversarial attacks, DRL may suffer serious performance degradation or even make potentially dangerous decisions, so it is crucial to ensure their stability in security-sensitive scenarios. In this paper, we first introduce the basic framework of DRL and analyze the main security challenges faced in complex and changing environments. In addition, this paper proposes an adversarial attack classification framework based on perturbation type and attack target and reviews the mainstream adversarial attack methods against DRL in detail, including various attack methods such as perturbation state space, action space, reward function and model space. To effectively counter the attacks, this paper systematically summarizes various current robustness training strategies, including adversarial training, competitive training, robust learning, adversarial detection, defense distillation and other related defense techniques, we also discuss the advantages and shortcomings of these methods in improving the robustness of DRL. Finally, this paper looks into the future research direction of DRL in adversarial environments, emphasizing the research needs in terms of improving generalization, reducing computational complexity, and enhancing scalability and explainability, aiming to provide valuable references and directions for researchers.

**arXiv ID:** 2510.20314
</details>

<details>
<summary><strong>GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning</strong> - Jinchang Luo, Mingquan Cheng, Fan Wan, Ni Li, Xiaoling Xia, Shuangshuang Tian, Tingcheng Bian, Haiwei Wang, Haohuan Fu, Yan Tao - [[pdf]](https://arxiv.org/pdf/2510.20548)</summary>

**Abstract:** Reinforcement learning has recently shown promise in improving retrieval-augmented generation (RAG). Despite these advances, its effectiveness in multi-hop question answering (QA) remains limited by two fundamental limitations: (i) global planning absence to structure multi-step reasoning, and (ii) unfaithful execution, which hinders effective query formulation and consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement learning framework designed to enhance global reasoning in multi-hop QA. GlobalRAG decomposes questions into subgoals, coordinates retrieval with reasoning, and refines evidence iteratively. To guide this process, we introduce Planning Quality Reward and SubGoal Completion Reward, which encourage coherent planning and reliable subgoal execution. In addition, a progressive weight annealing strategy balances process-oriented and outcome-based objectives. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms strong baselines while using only 8k training data (42% of the training data used by strong baselines), achieving average improvements of 14.2% in both EM and F1.

**arXiv ID:** 2510.20548
</details>

<details>
<summary><strong>AdaDoS: Adaptive DoS Attack via Deep Adversarial Reinforcement Learning in SDN</strong> - Wei Shao, Yuhao Wang, Rongguang He, Muhammad Ejaz Ahmed, Seyit Camtepe - [[pdf]](https://arxiv.org/pdf/2510.20566)</summary>

**Abstract:** Existing defence mechanisms have demonstrated significant effectiveness in mitigating rule-based Denial-of-Service (DoS) attacks, leveraging predefined signatures and static heuristics to identify and block malicious traffic. However, the emergence of AI-driven techniques presents new challenges to SDN security, potentially compromising the efficacy of existing defence mechanisms. In this paper, we introduce~AdaDoS, an adaptive attack model that disrupt network operations while evading detection by existing DoS-based detectors through adversarial reinforcement learning (RL). Specifically, AdaDoS models the problem as a competitive game between an attacker, whose goal is to obstruct network traffic without being detected, and a detector, which aims to identify malicious traffic. AdaDoS can solve this game by dynamically adjusting its attack strategy based on feedback from the SDN and the detector. Additionally, recognising that attackers typically have less information than defenders, AdaDoS formulates the DoS-like attack as a partially observed Markov decision process (POMDP), with the attacker having access only to delay information between attacker and victim nodes. We address this challenge with a novel reciprocal learning module, where the student agent, with limited observations, enhances its performance by learning from the teacher agent, who has full observational capabilities in the SDN environment. AdaDoS represents the first application of RL to develop DoS-like attack sequences, capable of adaptively evading both machine learning-based and rule-based DoS-like attack detectors.

**arXiv ID:** 2510.20566
</details>

<details>
<summary><strong>Real-Time Gait Adaptation for Quadrupeds using Model Predictive Control and Reinforcement Learning</strong> - Ganga Nair B, Prakrut Kotecha, Shishir Kolathaya - [[pdf]](https://arxiv.org/pdf/2510.20706)</summary>

**Abstract:** Model-free reinforcement learning (RL) has enabled adaptable and agile quadruped locomotion; however, policies often converge to a single gait, leading to suboptimal performance. Traditionally, Model Predictive Control (MPC) has been extensively used to obtain task-specific optimal policies but lacks the ability to adapt to varying environments. To address these limitations, we propose an optimization framework for real-time gait adaptation in a continuous gait space, combining the Model Predictive Path Integral (MPPI) algorithm with a Dreamer module to produce adaptive and optimal policies for quadruped locomotion. At each time step, MPPI jointly optimizes the actions and gait variables using a learned Dreamer reward that promotes velocity tracking, energy efficiency, stability, and smooth transitions, while penalizing abrupt gait changes. A learned value function is incorporated as terminal reward, extending the formulation to an infinite-horizon planner. We evaluate our framework in simulation on the Unitree Go1, demonstrating an average reduction of up to 36.48\% in energy consumption across varying target speeds, while maintaining accurate tracking and adaptive, task-appropriate gaits.

**arXiv ID:** 2510.20706
</details>

<details>
<summary><strong>Reinforcement Learning and Consumption-Savings Behavior</strong> - Brandon Kaplowitz - [[pdf]](https://arxiv.org/pdf/2510.20748)</summary>

**Abstract:** This paper demonstrates how reinforcement learning can explain two puzzling empirical patterns in household consumption behavior during economic downturns. I develop a model where agents use Q-learning with neural network approximation to make consumption-savings decisions under income uncertainty, departing from standard rational expectations assumptions. The model replicates two key findings from recent literature: (1) unemployed households with previously low liquid assets exhibit substantially higher marginal propensities to consume (MPCs) out of stimulus transfers compared to high-asset households (0.50 vs 0.34), even when neither group faces borrowing constraints, consistent with Ganong et al. (2024); and (2) households with more past unemployment experiences maintain persistently lower consumption levels after controlling for current economic conditions, a "scarring" effect documented by Malmendier and Shen (2024). Unlike existing explanations based on belief updating about income risk or ex-ante heterogeneity, the reinforcement learning mechanism generates both higher MPCs and lower consumption levels simultaneously through value function approximation errors that evolve with experience. Simulation results closely match the empirical estimates, suggesting that adaptive learning through reinforcement learning provides a unifying framework for understanding how past experiences shape current consumption behavior beyond what current economic conditions would predict.

**arXiv ID:** 2510.20748
</details>

<details>
<summary><strong>Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?</strong> - Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang - [[pdf]](https://arxiv.org/pdf/2504.13837)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.

**arXiv ID:** 2504.13837
</details>

<details>
<summary><strong>DAIL: Beyond Task Ambiguity for Language-Conditioned Reinforcement Learning</strong> - Runpeng Xie, Quanwei Wang, Hao Hu, Zherui Zhou, Ni Mu, Xiyun Li, Yiqin Yang, Shuang Xu, Qianchuan Zhao, Bo XU - [[pdf]](https://arxiv.org/pdf/2510.19562)</summary>

**Abstract:** Comprehending natural language and following human instructions are critical capabilities for intelligent agents. However, the flexibility of linguistic instructions induces substantial ambiguity across language-conditioned tasks, severely degrading algorithmic performance. To address these limitations, we present a novel method named DAIL (Distributional Aligned Learning), featuring two key components: distributional policy and semantic alignment. Specifically, we provide theoretical results that the value distribution estimation mechanism enhances task differentiability. Meanwhile, the semantic alignment module captures the correspondence between trajectories and linguistic instructions. Extensive experimental results on both structured and visual observation benchmarks demonstrate that DAIL effectively resolves instruction ambiguities, achieving superior performance to baseline methods. Our implementation is available at this https URL.

**arXiv ID:** 2510.19562
</details>

<details>
<summary><strong>LLM-Explorer: A Plug-in Reinforcement Learning Policy Exploration Enhancement Driven by Large Language Models</strong> - Qianyue Hao, Yiwen Song, Qingmin Liao, Jian Yuan, Yong Li - [[pdf]](https://arxiv.org/pdf/2505.15293)</summary>

**Abstract:** Policy exploration is critical in reinforcement learning (RL), where existing approaches include greedy, Gaussian process, etc. However, these approaches utilize preset stochastic processes and are indiscriminately applied in all kinds of RL tasks without considering task-specific features that influence policy exploration. Moreover, during RL training, the evolution of such stochastic processes is rigid, which typically only incorporates a decay in the variance, failing to adjust flexibly according to the agent's real-time learning status. Inspired by the analyzing and reasoning capability of large language models (LLMs), we design LLM-Explorer to adaptively generate task-specific exploration strategies with LLMs, enhancing the policy exploration in RL. In our design, we sample the learning trajectory of the agent during the RL training in a given task and prompt the LLM to analyze the agent's current policy learning status and then generate a probability distribution for future policy exploration. Updating the probability distribution periodically, we derive a stochastic process specialized for the particular task and dynamically adjusted to adapt to the learning process. Our design is a plug-in module compatible with various widely applied RL algorithms, including the DQN series, DDPG, TD3, and any possible variants developed based on them. Through extensive experiments on the Atari and MuJoCo benchmarks, we demonstrate LLM-Explorer's capability to enhance RL policy exploration, achieving an average performance improvement up to 37.27%. Our code is open-source at this https URL for reproducibility.

**arXiv ID:** 2505.15293
</details>

<details>
<summary><strong>How Ensembles of Distilled Policies Improve Generalisation in Reinforcement Learning</strong> - Max Weltevrede, Moritz A. Zanger, Matthijs T.J. Spaan, Wendelin Böhmer - [[pdf]](https://arxiv.org/pdf/2505.16581)</summary>

**Abstract:** In the zero-shot policy transfer setting in reinforcement learning, the goal is to train an agent on a fixed set of training environments so that it can generalise to similar, but unseen, testing environments. Previous work has shown that policy distillation after training can sometimes produce a policy that outperforms the original in the testing environments. However, it is not yet entirely clear why that is, or what data should be used to distil the policy. In this paper, we prove, under certain assumptions, a generalisation bound for policy distillation after training. The theory provides two practical insights: for improved generalisation, you should 1) train an ensemble of distilled policies, and 2) distil it on as much data from the training environments as possible. We empirically verify that these insights hold in more general settings, when the assumptions required for the theory no longer hold. Finally, we demonstrate that an ensemble of policies distilled on a diverse dataset can generalise significantly better than the original agent.

**arXiv ID:** 2505.16581
</details>

<details>
<summary><strong>Leveraging Analytic Gradients in Provably Safe Reinforcement Learning</strong> - Tim Walter, Hannah Markgraf, Jonathan Külz, Matthias Althoff - [[pdf]](https://arxiv.org/pdf/2506.01665)</summary>

**Abstract:** The deployment of autonomous robots in safety-critical applications requires safety guarantees. Provably safe reinforcement learning is an active field of research that aims to provide such guarantees using safeguards. These safeguards should be integrated during training to reduce the sim-to-real gap. While there are several approaches for safeguarding sampling-based reinforcement learning, analytic gradient-based reinforcement learning often achieves superior performance from fewer environment interactions. However, there is no safeguarding approach for this learning paradigm yet. Our work addresses this gap by developing the first effective safeguard for analytic gradient-based reinforcement learning. We analyse existing, differentiable safeguards, adapt them through modified mappings and gradient formulations, and integrate them into a state-of-the-art learning algorithm and a differentiable simulation. Using numerical experiments on three control tasks, we evaluate how different safeguards affect learning. The results demonstrate safeguarded training without compromising performance. Additional visuals are provided at \href{this https URL}{this http URL}.

**arXiv ID:** 2506.01665
</details>

<details>
<summary><strong>Multi-Modal Decentralized Reinforcement Learning for Modular Reconfigurable Lunar Robots</strong> - Ashutosh Mishra, Shreya Santra, Elian Neppel, Edoardo M. Rossi Lombardi, Shamistan Karimov, Kentaro Uno, Kazuya Yoshida - [[pdf]](https://arxiv.org/pdf/2510.20347)</summary>

**Abstract:** Modular reconfigurable robots suit task-specific space operations, but the combinatorial growth of morphologies hinders unified control. We propose a decentralized reinforcement learning (Dec-RL) scheme where each module learns its own policy: wheel modules use Soft Actor-Critic (SAC) for locomotion and 7-DoF limbs use Proximal Policy Optimization (PPO) for steering and manipulation, enabling zero-shot generalization to unseen configurations. In simulation, the steering policy achieved a mean absolute error of 3.63° between desired and induced angles; the manipulation policy plateaued at 84.6 % success on a target-offset criterion; and the wheel policy cut average motor torque by 95.4 % relative to baseline while maintaining 99.6 % success. Lunar-analogue field tests validated zero-shot integration for autonomous locomotion, steering, and preliminary alignment for reconfiguration. The system transitioned smoothly among synchronous, parallel, and sequential modes for Policy Execution, without idle states or control conflicts, indicating a scalable, reusable, and robust approach for modular lunar robots.

**arXiv ID:** 2510.20347
</details>

<details>
<summary><strong>Every Question Has Its Own Value: Reinforcement Learning with Explicit Human Values</strong> - Dian Yu, Yulai Zhao, Kishan Panaganti, Linfeng Song, Haitao Mi, Dong Yu - [[pdf]](https://arxiv.org/pdf/2510.20187)</summary>

**Abstract:** We propose Reinforcement Learning with Explicit Human Values (RLEV), a method that aligns Large Language Model (LLM) optimization directly with quantifiable human value signals. While Reinforcement Learning with Verifiable Rewards (RLVR) effectively trains models in objective domains using binary correctness rewards, it overlooks that not all tasks are equally significant. RLEV extends this framework by incorporating human-defined value signals directly into the reward function. Using exam-style data with explicit ground-truth value labels, RLEV consistently outperforms correctness-only baselines across multiple RL algorithms and model scales. Crucially, RLEV policies not only improve value-weighted accuracy but also learn a value-sensitive termination policy: concise for low-value prompts, thorough for high-value ones. We demonstrate this behavior stems from value-weighted gradient amplification on end-of-sequence tokens. Ablation studies confirm the gain is causally linked to value alignment. RLEV remains robust under noisy value signals, such as difficulty-based labels, demonstrating that optimizing for an explicit utility function offers a practical path to aligning LLMs with human priorities.

**arXiv ID:** 2510.20187
</details>

<details>
<summary><strong>Dialogue Is Not Enough to Make a Communicative BabyLM (But Neither Is Developmentally Inspired Reinforcement Learning)</strong> - Francesca Padovani, Bastian Bunzeck, Manar Ali, Omar Momen, Arianna Bisazza, Hendrik Buschmeier, Sina Zarrieß - [[pdf]](https://arxiv.org/pdf/2510.20358)</summary>

**Abstract:** We investigate whether pre-training exclusively on dialogue data results in formally and functionally apt small language models. Based on this pre-trained llamalogue model, we employ a variety of fine-tuning strategies to enforce "more communicative" text generations by our models. Although our models underperform on most standard BabyLM benchmarks, they excel at dialogue continuation prediction in a minimal pair setting. While PPO fine-tuning has mixed to adversarial effects on our models, DPO fine-tuning further improves their performance on our custom dialogue benchmark.

**arXiv ID:** 2510.20358
</details>

<details>
<summary><strong>Hybrid Latent Reasoning via Reinforcement Learning</strong> - Zhenrui Yue, Bowen Jin, Huimin Zeng, Honglei Zhuang, Zhen Qin, Jinsung Yoon, Lanyu Shang, Jiawei Han, Dong Wang - [[pdf]](https://arxiv.org/pdf/2505.18454)</summary>

**Abstract:** Recent advances in large language models (LLMs) have introduced latent reasoning as a promising alternative to autoregressive reasoning. By performing internal computation with hidden states from previous steps, latent reasoning benefit from more informative features rather than sampling a discrete chain-of-thought (CoT) path. Yet latent reasoning approaches are often incompatible with LLMs, as their continuous paradigm conflicts with the discrete nature of autoregressive generation. Moreover, these methods rely on CoT traces for training and thus fail to exploit the inherent reasoning patterns of LLMs. In this work, we explore latent reasoning by leveraging the intrinsic capabilities of LLMs via reinforcement learning (RL). To this end, we introduce hybrid reasoning policy optimization (HRPO), an RL-based hybrid latent reasoning approach that (1) integrates prior hidden states into sampled tokens with a learnable gating mechanism, and (2) initializes training with predominantly token embeddings while progressively incorporating more hidden features. This design maintains LLMs' generative capabilities and incentivizes hybrid reasoning using both discrete and continuous representations. In addition, the hybrid HRPO introduces stochasticity into latent reasoning via token sampling, thereby enabling RL-based optimization without requiring CoT trajectories. Extensive evaluations across diverse benchmarks show that HRPO outperforms prior methods in both knowledge- and reasoning-intensive tasks. Furthermore, HRPO-trained LLMs remain interpretable and exhibit intriguing behaviors like cross-lingual patterns and shorter completion lengths, highlighting the potential of our RL-based approach and offer insights for future work in latent reasoning.

**arXiv ID:** 2505.18454
</details>

<details>
<summary><strong>FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning</strong> - Shiqi Dai, Wei Dai, Jiaee Cheong, Paul Pu Liang - [[pdf]](https://arxiv.org/pdf/2510.19893)</summary>

**Abstract:** Medical artificial intelligence systems have achieved remarkable diagnostic capabilities, yet they consistently exhibit performance disparities across demographic groups, causing real-world harm to underrepresented populations. While recent multimodal reasoning foundation models have advanced clinical diagnosis through integrated analysis of diverse medical data, reasoning trainings via reinforcement learning inherit and often amplify biases present in training datasets dominated by majority populations. We introduce Fairness-aware Group Relative Policy Optimization (FairGRPO), a hierarchical reinforcement learning approach that promotes equitable learning across heterogeneous clinical populations. FairGRPO employs adaptive importance weighting of advantages based on representation, task difficulty, and data source. To address the common issue of missing demographic labels in the clinical domain, we further employ unsupervised clustering, which automatically discovers latent demographic groups when labels are unavailable. Through comprehensive experiments across 7 clinical diagnostic datasets spanning 5 clinical modalities across X-ray, CT scan, dermoscropy, mammography and ultrasound, we demonstrate that FairGRPO reduces predictive parity by 27.2% against all vanilla and bias mitigated RL baselines, while improving F1 score by 12.49%. Furthermore, training dynamics analysis reveals that FairGRPO progressively improves fairness throughout optimization, while baseline RL methods exhibit deteriorating fairness as training progresses. Based on FairGRPO, we release FairMedGemma-4B, a fairness-aware clinical VLLM that achieves state-of-the-art performance while demonstrating significantly reduced disparities across demographic groups.

**arXiv ID:** 2510.19893
</details>

<details>
<summary><strong>SALT: Step-level Advantage Assignment for Long-horizon Agents via Trajectory Graph</strong> - Jiazheng Li, Yawei Wang, David Yan, Yijun Tian, Zhichao Xu, Huan Song, Panpan Xu, Lin Lee Cheong - [[pdf]](https://arxiv.org/pdf/2510.20022)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable capabilities, enabling language agents to excel at single-turn tasks. However, their application to complex, multi-step, and long-horizon tasks remains challenging. While reinforcement learning (RL) offers a promising avenue for addressing these challenges, mainstream approaches typically rely solely on sparse, outcome-based rewards, a limitation that becomes especially problematic for group-based RL algorithms lacking critic models, such as Group Relative Policy Optimization (GRPO). In such methods, uniformly rewarding or penalizing all actions within a trajectory can lead to training instability and suboptimal policies, because beneficial and detrimental actions are often entangled across multi-step interactions. To address this challenge, we propose SALT, a novel and lightweight framework that provides a finer-grained advantage assignment, derived solely from outcome rewards. We achieve this by constructing a graph from trajectories of the same prompt, which allows us to quantify the quality of each step and assign advantages accordingly. Crucially, SALT is designed as a plug-and-play module that seamlessly integrates with existing group-based RL algorithms, requiring no modifications to the rollout procedure and introducing negligible computational overhead. Extensive experiments on the WebShop, ALFWorld, and AppWorld benchmarks with various model sizes demonstrate that SALT consistently improves performance. We also conduct a thorough analysis to validate the design choices behind SALT and offer actionable insights.

**arXiv ID:** 2510.20022
</details>

<details>
<summary><strong>Learning Personalized Ad Impact via Contextual Reinforcement Learning under Delayed Rewards</strong> - Yuwei Cheng, Zifeng Zhao, Haifeng Xu - [[pdf]](https://arxiv.org/pdf/2510.20055)</summary>

**Abstract:** Online advertising platforms use automated auctions to connect advertisers with potential customers, requiring effective bidding strategies to maximize profits. Accurate ad impact estimation requires considering three key factors: delayed and long-term effects, cumulative ad impacts such as reinforcement or fatigue, and customer heterogeneity. However, these effects are often not jointly addressed in previous studies. To capture these factors, we model ad bidding as a Contextual Markov Decision Process (CMDP) with delayed Poisson rewards. For efficient estimation, we propose a two-stage maximum likelihood estimator combined with data-splitting strategies, ensuring controlled estimation error based on the first-stage estimator's (in)accuracy. Building on this, we design a reinforcement learning algorithm to derive efficient personalized bidding strategies. This approach achieves a near-optimal regret bound of $\tilde{O}{(dH^2\sqrt{T})}$, where $d$ is the contextual dimension, $H$ is the number of rounds, and $T$ is the number of customers. Our theoretical findings are validated by simulation experiments.

**arXiv ID:** 2510.20055
</details>

<details>
<summary><strong>Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents</strong> - Jane H. Lee, Baturay Saglam, Spyridon Pougkakiotis, Amin Karbasi, Dionysis Kalogerias - [[pdf]](https://arxiv.org/pdf/2510.20199)</summary>

**Abstract:** Constrained optimization provides a common framework for dealing with conflicting objectives in reinforcement learning (RL). In most of these settings, the objectives (and constraints) are expressed though the expected accumulated reward. However, this formulation neglects risky or even possibly catastrophic events at the tails of the reward distribution, and is often insufficient for high-stakes applications in which the risk involved in outliers is critical. In this work, we propose a framework for risk-aware constrained RL, which exhibits per-stage robustness properties jointly in reward values and time using optimized certainty equivalents (OCEs). Our framework ensures an exact equivalent to the original constrained problem within a parameterized strong Lagrangian duality framework under appropriate constraint qualifications, and yields a simple algorithmic recipe which can be wrapped around standard RL solvers, such as PPO. Lastly, we establish the convergence of the proposed algorithm under common assumptions, and verify the risk-aware properties of our approach through several numerical experiments.

**arXiv ID:** 2510.20199
</details>

<details>
<summary><strong>A Unified Framework for Zero-Shot Reinforcement Learning</strong> - Jacopo Di Ventura, Jan Felix Kleuker, Aske Plaat, Thomas Moerland - [[pdf]](https://arxiv.org/pdf/2510.20542)</summary>

**Abstract:** Zero-shot reinforcement learning (RL) has emerged as a setting for developing general agents in an unsupervised manner, capable of solving downstream tasks without additional training or planning at test-time. Unlike conventional RL, which optimizes policies for a fixed reward, zero-shot RL requires agents to encode representations rich enough to support immediate adaptation to any objective, drawing parallels to vision and language foundation models. Despite growing interest, the field lacks a common analytical lens.
We present the first unified framework for zero-shot RL. Our formulation introduces a consistent notation and taxonomy that organizes existing approaches and allows direct comparison between them. Central to our framework is the classification of algorithms into two families: direct representations, which learn end-to-end mappings from rewards to policies, and compositional representations, which decompose the representation leveraging the substructure of the value function. Within this framework, we highlight shared principles and key differences across methods, and we derive an extended bound for successor-feature methods, offering a new perspective on their performance in the zero-shot regime. By consolidating existing work under a common lens, our framework provides a principled foundation for future research in zero-shot RL and outlines a clear path toward developing more general agents.

**arXiv ID:** 2510.20542
</details>

<details>
<summary><strong>KL-Regularized Reinforcement Learning is Designed to Mode Collapse</strong> - Anthony GX-Chen, Jatin Prakash, Jeff Guo, Rob Fergus, Rajesh Ranganath - [[pdf]](https://arxiv.org/pdf/2510.20817)</summary>

**Abstract:** It is commonly believed that optimizing the reverse KL divergence results in "mode seeking", while optimizing forward KL results in "mass covering", with the latter being preferred if the goal is to sample from multiple diverse modes. We show -- mathematically and empirically -- that this intuition does not necessarily transfer well to doing reinforcement learning with reverse/forward KL regularization (e.g. as commonly used with language models). Instead, the choice of reverse/forward KL determines the family of optimal target distributions, parameterized by the regularization coefficient. Mode coverage depends primarily on other factors, such as regularization strength, and relative scales between rewards and reference probabilities. Further, we show commonly used settings such as low regularization strength and equal verifiable rewards tend to specify unimodal target distributions, meaning the optimization objective is, by construction, non-diverse. We leverage these insights to construct a simple, scalable, and theoretically justified algorithm. It makes minimal changes to reward magnitudes, yet optimizes for a target distribution which puts high probability over all high-quality sampling modes. In experiments, this simple modification works to post-train both Large Language Models and Chemical Language Models to have higher solution quality and diversity, without any external signals of diversity, and works with both forward and reverse KL when using either naively fails.

**arXiv ID:** 2510.20817
</details>

<details>
<summary><strong>Multi Task Inverse Reinforcement Learning for Common Sense Reward</strong> - Neta Glazer, Aviv Navon, Aviv Shamsian, Ethan Fetaya - [[pdf]](https://arxiv.org/pdf/2402.11367)</summary>

**Abstract:** One of the challenges in applying reinforcement learning in a complex real-world environment lies in providing the agent with a sufficiently detailed reward function. Any misalignment between the reward and the desired behavior can result in unwanted outcomes. This may lead to issues like "reward hacking" where the agent maximizes rewards by unintended behavior. In this work, we propose to disentangle the reward into two distinct parts. A simple task-specific reward, outlining the particulars of the task at hand, and an unknown common-sense reward, indicating the expected behavior of the agent within the environment. We then explore how this common-sense reward can be learned from expert demonstrations. We first show that inverse reinforcement learning, even when it succeeds in training an agent, does not learn a useful reward function. That is, training a new agent with the learned reward does not impair the desired behaviors. We then demonstrate that this problem can be solved by training simultaneously on multiple tasks. That is, multi-task inverse reinforcement learning can be applied to learn a useful reward function.

**arXiv ID:** 2402.11367
</details>

<details>
<summary><strong>Reinforcement Learning-based Robust Wall Climbing Locomotion Controller in Ferromagnetic Environment</strong> - Yong Um, Young-Ha Shin, Joon-Ha Kim, Soonpyo Kwon, Hae-Won Park - [[pdf]](https://arxiv.org/pdf/2510.20174)</summary>

**Abstract:** We present a reinforcement learning framework for quadrupedal wall-climbing locomotion that explicitly addresses uncertainty in magnetic foot adhesion. A physics-based adhesion model of a quadrupedal magnetic climbing robot is incorporated into simulation to capture partial contact, air-gap sensitivity, and probabilistic attachment failures. To stabilize learning and enable reliable transfer, we design a three-phase curriculum: (1) acquire a crawl gait on flat ground without adhesion, (2) gradually rotate the gravity vector to vertical while activating the adhesion model, and (3) inject stochastic adhesion failures to encourage slip recovery. The learned policy achieves a high success rate, strong adhesion retention, and rapid recovery from detachment in simulation under degraded adhesion. Compared with a model predictive control (MPC) baseline that assumes perfect adhesion, our controller maintains locomotion when attachment is intermittently lost. Hardware experiments with the untethered robot further confirm robust vertical crawling on steel surfaces, maintaining stability despite transient misalignment and incomplete attachment. These results show that combining curriculum learning with realistic adhesion modeling provides a resilient sim-to-real framework for magnetic climbing robots in complex environments.

**arXiv ID:** 2510.20174
</details>

<details>
<summary><strong>Towards Robust Zero-Shot Reinforcement Learning</strong> - Kexin Zheng, Lauriane Teyssier, Yinan Zheng, Yu Luo, Xianyuan Zhan - [[pdf]](https://arxiv.org/pdf/2510.15382)</summary>

**Abstract:** The recent development of zero-shot reinforcement learning (RL) has opened a new avenue for learning pre-trained generalist policies that can adapt to arbitrary new tasks in a zero-shot manner. While the popular Forward-Backward representations (FB) and related methods have shown promise in zero-shot RL, we empirically found that their modeling lacks expressivity and that extrapolation errors caused by out-of-distribution (OOD) actions during offline learning sometimes lead to biased representations, ultimately resulting in suboptimal performance. To address these issues, we propose Behavior-REgularizEd Zero-shot RL with Expressivity enhancement (BREEZE), an upgraded FB-based framework that simultaneously enhances learning stability, policy extraction capability, and representation learning quality. BREEZE introduces behavioral regularization in zero-shot RL policy learning, transforming policy optimization into a stable in-sample learning paradigm. Additionally, BREEZE extracts the policy using a task-conditioned diffusion model, enabling the generation of high-quality and multimodal action distributions in zero-shot RL settings. Moreover, BREEZE employs expressive attention-based architectures for representation modeling to capture the complex relationships between environmental dynamics. Extensive experiments on ExORL and D4RL Kitchen demonstrate that BREEZE achieves the best or near-the-best performance while exhibiting superior robustness compared to prior offline zero-shot RL methods. The official implementation is available at: this https URL.

**arXiv ID:** 2510.15382
</details>

<details>
<summary><strong>EmbodiedBrain: Expanding Performance Boundaries of Task Planning for Embodied Intelligence</strong> - Ding Zou, Feifan Wang, Mengyu Ge, Siyuan Fan, Zongbing Zhang, Wei Chen, Lingfeng Wang, Zhongyou Hu, Wenrui Yan, Zhengwei Gao, Hao Wang, Weizhao Jin, Yu Zhang, Hainan Zhao, Mingliang Zhang, Xianxian Xi, Yaru Zhang, Wenyuan Li, Zhengguang Gao, Yurui Zhu - [[pdf]](https://arxiv.org/pdf/2510.20578)</summary>

**Abstract:** The realization of Artificial General Intelligence (AGI) necessitates Embodied AI agents capable of robust spatial perception, effective task planning, and adaptive execution in physical environments. However, current large language models (LLMs) and multimodal LLMs (MLLMs) for embodied tasks suffer from key limitations, including a significant gap between model design and agent requirements, an unavoidable trade-off between real-time latency and performance, and the use of unauthentic, offline evaluation metrics. To address these challenges, we propose EmbodiedBrain, a novel vision-language foundation model available in both 7B and 32B parameter sizes. Our framework features an agent-aligned data structure and employs a powerful training methodology that integrates large-scale Supervised Fine-Tuning (SFT) with Step-Augumented Group Relative Policy Optimization (Step-GRPO), which boosts long-horizon task success by integrating preceding steps as Guided Precursors. Furthermore, we incorporate a comprehensive reward system, including a Generative Reward Model (GRM) accelerated at the infrastructure level, to improve training efficiency. For enable thorough validation, we establish a three-part evaluation system encompassing General, Planning, and End-to-End Simulation Benchmarks, highlighted by the proposal and open-sourcing of a novel, challenging simulation environment. Experimental results demonstrate that EmbodiedBrain achieves superior performance across all metrics, establishing a new state-of-the-art for embodied foundation models. Towards paving the way for the next generation of generalist embodied agents, we open-source all of our data, model weight, and evaluating methods, which are available at this https URL.

**arXiv ID:** 2510.20578
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
