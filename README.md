# Agent arXiv Daily

**Last Updated:** 2025-12-24 02:22:14

**Total Papers:** 89

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
<summary><strong>$γ(3,4)$ `Attention' in Cognitive Agents: Ontology-Free Knowledge Representations With Promise Theoretic Semantics</strong> - Mark Burgess - [[pdf]](https://arxiv.org/pdf/2512.19084)</summary>

**Abstract:** The semantics and dynamics of `attention' are closely related to promise theoretic notions developed for autonomous agents and can thus easily be written down in promise framework. In this way one may establish a bridge between vectorized Machine Learning and Knowledge Graph representations without relying on language models implicitly. Our expectations for knowledge presume a degree of statistical stability, i.e. average invariance under repeated observation, or `trust' in the data. Both learning networks and knowledge graph representations can meaningfully coexist to preserve different aspects of data. While vectorized data are useful for probabilistic estimation, graphs preserve the intentionality of the source even under data fractionation. Using a Semantic Spacetime $\gamma(3,4)$ graph, one avoids complex ontologies in favour of classification of features by their roles in semantic processes. The latter favours an approach to reasoning under conditions of uncertainty. Appropriate attention to causal boundary conditions may lead to orders of magnitude compression of data required for such context determination, as required in the contexts of autonomous robotics, defence deployments, and ad hoc emergency services.

**arXiv ID:** 2512.19084
</details>

<details>
<summary><strong>LLM-based Few-Shot Early Rumor Detection with Imitation Agent</strong> - Fengzhu Zeng, Qian Shao, Ling Cheng, Wei Gao, Shih-Fen Cheng, Jing Ma, Cheng Niu - [[pdf]](https://arxiv.org/pdf/2512.18352)</summary>

**Abstract:** Early Rumor Detection (EARD) aims to identify the earliest point at which a claim can be accurately classified based on a sequence of social media posts. This is especially challenging in data-scarce settings. While Large Language Models (LLMs) perform well in few-shot NLP tasks, they are not well-suited for time-series data and are computationally expensive for both training and inference. In this work, we propose a novel EARD framework that combines an autonomous agent and an LLM-based detection model, where the agent acts as a reliable decision-maker for \textit{early time point determination}, while the LLM serves as a powerful \textit{rumor detector}. This approach offers the first solution for few-shot EARD, necessitating only the training of a lightweight agent and allowing the LLM to remain training-free. Extensive experiments on four real-world datasets show our approach boosts performance across LLMs and surpasses existing EARD methods in accuracy and earliness.

**arXiv ID:** 2512.18352
</details>

<details>
<summary><strong>Dagstuhl Perspectives Workshop 24352 -- Conversational Agents: A Framework for Evaluation (CAFE): Manifesto</strong> - Christine Bauer, Li Chen, Nicola Ferro, Norbert Fuhr, Avishek Anand, Timo Breuer, Guglielmo Faggioli, Ophir Frieder, Hideo Joho, Jussi Karlgren, Johannes Kiesel, Bart P. Knijnenburg, Aldo Lipani, Lien Michiels, Andrea Papenmeier, Maria Soledad Pera, Mark Sanderson, Scott Sanner, Benno Stein, Johanne R. Trippas, Karin Verspoor, Martijn C Willemsen - [[pdf]](https://arxiv.org/pdf/2506.11112)</summary>

**Abstract:** During the workshop, we deeply discussed what CONversational Information ACcess (CONIAC) is and its unique features, proposing a world model abstracting it, and defined the Conversational Agents Framework for Evaluation (CAFE) for the evaluation of CONIAC systems, consisting of six major components: 1) goals of the system's stakeholders, 2) user tasks to be studied in the evaluation, 3) aspects of the users carrying out the tasks, 4) evaluation criteria to be considered, 5) evaluation methodology to be applied, and 6) measures for the quantitative criteria chosen.

**arXiv ID:** 2506.11112
</details>

<details>
<summary><strong>Affective Multimodal Agents with Proactive Knowledge Grounding for Emotionally Aligned Marketing Dialogue</strong> - Lin Yu, Xiaofei Han, Yifei Kang, Chiung-Yi Tseng, Danyang Zhang, Ziqian Bi, Zhimo Han - [[pdf]](https://arxiv.org/pdf/2511.21728)</summary>

**Abstract:** Recent advances in large language models (LLMs) have enabled fluent dialogue systems, but most remain reactive and struggle in emotionally rich, goal-oriented settings such as marketing conversations. To address this limitation, we propose AffectMind, a multimodal affective dialogue agent that performs proactive reasoning and dynamic knowledge grounding to sustain emotionally aligned and persuasive interactions. AffectMind combines three components: a Proactive Knowledge Grounding Network (PKGN) that continuously updates factual and affective context from text, vision, and prosody; an Emotion--Intent Alignment Model (EIAM) that jointly models user emotion and purchase intent to adapt persuasion strategies; and a Reinforced Discourse Loop (RDL) that optimizes emotional coherence and engagement via reinforcement signals from user responses. Experiments on two newly curated marketing dialogue datasets, MM-ConvMarket and AffectPromo, show that AffectMind outperforms strong LLM-based baselines in emotional consistency (+26\%), persuasive success rate (+19\%), and long-term user engagement (+23\%), highlighting emotion-grounded proactivity as a key capability for commercial multimodal agents.

**arXiv ID:** 2511.21728
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (16 papers)</h2></summary>

<details>
<summary><strong>DeliveryBench: Can Agents Earn Profit in Real World?</strong> - Lingjun Mao, Jiawei Ren, Kun Zhou, Jixuan Chen, Ziqiao Ma, Lianhui Qin - [[pdf]](https://arxiv.org/pdf/2512.19234)</summary>

**Abstract:** LLMs and VLMs are increasingly deployed as embodied agents, yet existing benchmarks largely revolve around simple short-term tasks and struggle to capture rich realistic constraints that shape real-world decision making. To close this gap, we propose DeliveryBench, a city-scale embodied benchmark grounded in the real-world profession of food delivery. Food couriers naturally operate under long-horizon objectives (maximizing net profit over hours) while managing diverse constraints, e.g., delivery deadline, transportation expense, vehicle battery, and necessary interactions with other couriers and customers. DeliveryBench instantiates this setting in procedurally generated 3D cities with diverse road networks, buildings, functional locations, transportation modes, and realistic resource dynamics, enabling systematic evaluation of constraint-aware, long-horizon planning. We benchmark a range of VLM-based agents across nine cities and compare them with human players. Our results reveal a substantial performance gap to humans, and find that these agents are short-sighted and frequently break basic commonsense constraints. Additionally, we observe distinct personalities across models (e.g., adventurous GPT-5 vs. conservative Claude), highlighting both the brittleness and the diversity of current VLM-based embodied agents in realistic, constraint-dense environments. Our code, data, and benchmark are available at this https URL.

**arXiv ID:** 2512.19234
</details>

<details>
<summary><strong>EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration</strong> - Runze Li, Yuwen Zhai, Bo Xu, LiWu Xu, Nian Shi, Wei Zhang, Ran Lin, Liang Wang - [[pdf]](https://arxiv.org/pdf/2512.19396)</summary>

**Abstract:** Contemporary GUI agents, while increasingly capable due to advances in Large Vision-Language Models (VLMs), often operate with a critical limitation: they treat each task in isolation, lacking a mechanism to systematically learn from past successes. This digital ''amnesia'' results in sub-optimal performance, repeated errors, and poor generalization to novel challenges. To bridge this gap, we introduce EchoTrail-GUI, a novel framework designed to mimic human-like experiential learning by equipping agents with a dynamic, accessible memory. Our framework operates in three distinct stages. First, during Experience Exploration, an agent autonomously interacts with GUI environments to build a curated database of successful task trajectories, validated by a reward model. Crucially, the entire knowledge base construction is thus fully automated, requiring no human supervision. Second, in the Memory Injection stage, upon receiving a new task, our system efficiently retrieves the most relevant past trajectories to serve as actionable ''memories''. Finally, during GUI Task Inference, these memories are injected as in-context guidance to inform the agent's reasoning and decision-making process. We demonstrate the efficacy of our approach on benchmarks including Android World and AndroidLab. The results show that EchoTrail-GUI significantly improves the task success rate and operational efficiency of baseline agents, validating the power of structured memory in creating more robust and intelligent GUI automation.

**arXiv ID:** 2512.19396
</details>

<details>
<summary><strong>An Agentic Framework for Autonomous Materials Computation</strong> - Zeyu Xia, Jinzhe Ma, Congjie Zheng, Shufei Zhang, Yuqiang Li, Hang Su, P. Hu, Changshui Zhang, Xingao Gong, Wanli Ouyang, Lei Bai, Dongzhan Zhou, Mao Su - [[pdf]](https://arxiv.org/pdf/2512.19458)</summary>

**Abstract:** Large Language Models (LLMs) have emerged as powerful tools for accelerating scientific discovery, yet their static knowledge and hallucination issues hinder autonomous research applications. Recent advances integrate LLMs into agentic frameworks, enabling retrieval, reasoning, and tool use for complex scientific workflows. Here, we present a domain-specialized agent designed for reliable automation of first-principles materials computations. By embedding domain expertise, the agent ensures physically coherent multi-step workflows and consistently selects convergent, well-posed parameters, thereby enabling reliable end-to-end computational execution. A new benchmark of diverse computational tasks demonstrates that our system significantly outperforms standalone LLMs in both accuracy and robustness. This work establishes a verifiable foundation for autonomous computational experimentation and represents a key step toward fully automated scientific discovery.

**arXiv ID:** 2512.19458
</details>

<details>
<summary><strong>Towards Closed-Loop Embodied Empathy Evolution: Probing LLM-Centric Lifelong Empathic Motion Generation in Unseen Scenarios</strong> - Jiawen Wang, Jingjing Wang Tianyang Chen, Min Zhang, Guodong Zhou - [[pdf]](https://arxiv.org/pdf/2512.19551)</summary>

**Abstract:** In the literature, existing human-centric emotional motion generation methods primarily focus on boosting performance within a single scale-fixed dataset, largely neglecting the flexible and scale-increasing motion scenarios (e.g., sports, dance), whereas effectively learning these newly emerging scenarios can significantly enhance the model's real-world generalization ability. Inspired by this, this paper proposes a new LLM-Centric Lifelong Empathic Motion Generation (L^2-EMG) task, which aims to equip LLMs with the capability to continually acquire emotional motion generation knowledge across different unseen scenarios, potentially contributing to building a closed-loop and self-evolving embodied agent equipped with both empathy and intelligence. Further, this paper poses two key challenges in the L^2-EMG task, i.e., the emotion decoupling challenge and the scenario adapting challenge. To this end, this paper proposes an Emotion-Transferable and Scenario-Adapted Mixture of Experts (ES-MoE) approach which designs a causal-guided emotion decoupling block and a scenario-adapted expert constructing block to address the two challenges, respectively. Especially, this paper constructs multiple L^2-EMG datasets to validate the effectiveness of the ES-MoE approach. Extensive evaluations show that ES-MoE outperforms advanced baselines.

**arXiv ID:** 2512.19551
</details>

<details>
<summary><strong>VeruSAGE: A Study of Agent-Based Verification for Rust Systems</strong> - Chenyuan Yang, Natalie Neamtu, Chris Hawblitzel, Jacob R. Lorch, Shan Lu - [[pdf]](https://arxiv.org/pdf/2512.18436)</summary>

**Abstract:** Large language models (LLMs) have shown impressive capability to understand and develop code. However, their capability to rigorously reason about and prove code correctness remains in question. This paper offers a comprehensive study of LLMs' capability to develop correctness proofs for system software written in Rust. We curate a new system-verification benchmark suite, VeruSAGE-Bench, which consists of 849 proof tasks extracted from eight open-source Verus-verified Rust systems. Furthermore, we design different agent systems to match the strengths and weaknesses of different LLMs (o4-mini, GPT-5, Sonnet 4, and Sonnet 4.5). Our study shows that different tools and agent settings are needed to stimulate the system-verification capability of different types of LLMs. The best LLM-agent combination in our study completes over 80% of system-verification tasks in VeruSAGE-Bench. It also completes over 90% of a set of system proof tasks not part of VeruSAGE-Bench because they had not yet been finished by human experts. This result shows the great potential for LLM-assisted development of verified system software.

**arXiv ID:** 2512.18436
</details>

<details>
<summary><strong>SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios</strong> - Minh V. T. Thai, Tue Le, Dung Nguyen Manh, Huy Phan Nhat, Nghi D. Q. Bui - [[pdf]](https://arxiv.org/pdf/2512.18470)</summary>

**Abstract:** Existing benchmarks for AI coding agents focus on isolated, single-issue tasks such as fixing a bug or implementing a small feature. However, real-world software engineering is fundamentally a long-horizon endeavor: developers must interpret high-level requirements, plan coordinated changes across many files, and evolve codebases over multiple iterations while preserving existing functionality. We introduce SWE-EVO, a benchmark that evaluates agents on this long-horizon software evolution challenge. Constructed from release notes and version histories of seven mature open-source Python projects, Tool comprises 48 evolution tasks that require agents to implement multi-step modifications spanning an average of 21 files, validated against comprehensive test suites averaging 874 tests per instance. Experiments with state-of-the-art models reveal a striking capability gap: even GPT-5 with OpenHands achieves only a 21 percent resolution rate on Tool, compared to 65 percent on the single-issue SWE-Bench Verified. This demonstrates that current agents struggle with sustained, multi-file reasoning. We also propose Fix Rate, a fine-grained metric that captures partial progress toward solving these complex, long-horizon tasks.

**arXiv ID:** 2512.18470
</details>

<details>
<summary><strong>Results of the 2024 CommonRoad Motion Planning Competition for Autonomous Vehicles</strong> - Yanliang Huang, Xia Yan, Peiran Yin, Zhenduo Zhang, Zeyan Shao, Youran Wang, Haoliang Huang, Matthias Althoff - [[pdf]](https://arxiv.org/pdf/2512.19564)</summary>

**Abstract:** Over the past decade, a wide range of motion planning approaches for autonomous vehicles has been developed to handle increasingly complex traffic scenarios. However, these approaches are rarely compared on standardized benchmarks, limiting the assessment of relative strengths and weaknesses. To address this gap, we present the setup and results of the 4th CommonRoad Motion Planning Competition held in 2024, conducted using the CommonRoad benchmark suite. This annual competition provides an open-source and reproducible framework for benchmarking motion planning algorithms. The benchmark scenarios span highway and urban environments with diverse traffic participants, including passenger cars, buses, and bicycles. Planner performance is evaluated along four dimensions: efficiency, safety, comfort, and compliance with selected traffic rules. This report introduces the competition format and provides a comparison of representative high-performing planners from the 2023 and 2024 editions.

**arXiv ID:** 2512.19564
</details>

<details>
<summary><strong>MemEvolve: Meta-Evolution of Agent Memory Systems</strong> - Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, Shuicheng Yan - [[pdf]](https://arxiv.org/pdf/2512.18746)</summary>

**Abstract:** Self-evolving memory systems are unprecedentedly reshaping the evolutionary paradigm of large language model (LLM)-based agents. Prior work has predominantly relied on manually engineered memory architectures to store trajectories, distill experience, and synthesize reusable tools, enabling agents to evolve on the fly within environment interactions. However, this paradigm is fundamentally constrained by the staticity of the memory system itself: while memory facilitates agent-level evolving, the underlying memory architecture cannot be meta-adapted to diverse task contexts. To address this gap, we propose MemEvolve, a meta-evolutionary framework that jointly evolves agents' experiential knowledge and their memory architecture, allowing agent systems not only to accumulate experience but also to progressively refine how they learn from it. To ground MemEvolve in prior research and foster openness in future self-evolving systems, we introduce EvolveLab, a unified self-evolving memory codebase that distills twelve representative memory systems into a modular design space (encode, store, retrieve, manage), providing both a standardized implementation substrate and a fair experimental arena. Extensive evaluations on four challenging agentic benchmarks demonstrate that MemEvolve achieves (I) substantial performance gains, improving frameworks such as SmolAgent and Flash-Searcher by up to $17.06\%$; and (II) strong cross-task and cross-LLM generalization, designing memory architectures that transfer effectively across diverse benchmarks and backbone models.

**arXiv ID:** 2512.18746
</details>

<details>
<summary><strong>Towards Efficient Agents: A Co-Design of Inference Architecture and System</strong> - Weizhe Lin, Hui-Ling Zhen, Shuai Yang, Xian Wang, Renxi Liu, Hanting Chen, Wangze Zhang, Chuansai Zhou, Yiming Li, Chen Chen, Xing Li, Zhiyuan Yang, Xiaosong Li, Xianzhi Yu, Zhenhua Dong, Mingxuan Yuan, Yunhe Wang - [[pdf]](https://arxiv.org/pdf/2512.18337)</summary>

**Abstract:** The rapid development of large language model (LLM)-based agents has unlocked new possibilities for autonomous multi-turn reasoning and tool-augmented decision-making. However, their real-world deployment is hindered by severe inefficiencies that arise not from isolated model inference, but from the systemic latency accumulated across reasoning loops, context growth, and heterogeneous tool interactions. This paper presents AgentInfer, a unified framework for end-to-end agent acceleration that bridges inference optimization and architectural design. We decompose the problem into four synergistic components: AgentCollab, a hierarchical dual-model reasoning framework that balances large- and small-model usage through dynamic role assignment; AgentSched, a cache-aware hybrid scheduler that minimizes latency under heterogeneous request patterns; AgentSAM, a suffix-automaton-based speculative decoding method that reuses multi-session semantic memory to achieve low-overhead inference acceleration; and AgentCompress, a semantic compression mechanism that asynchronously distills and reorganizes agent memory without disrupting ongoing reasoning. Together, these modules form a Self-Evolution Engine capable of sustaining efficiency and cognitive stability throughout long-horizon reasoning tasks. Experiments on the BrowseComp-zh and DeepDiver benchmarks demonstrate that through the synergistic collaboration of these methods, AgentInfer reduces ineffective token consumption by over 50%, achieving an overall 1.8-2.5 times speedup with preserved accuracy. These results underscore that optimizing for agentic task completion-rather than merely per-token throughput-is the key to building scalable, efficient, and self-improving intelligent systems.

**arXiv ID:** 2512.18337
</details>

<details>
<summary><strong>Confucius Code Agent: Scalable Agent Scaffolding for Real-World Codebases</strong> - Sherman Wong, Zhenting Qi, Zhaodong Wang, Nathan Hu, Samuel Lin, Jun Ge, Erwin Gao, Wenlin Chen, Yilun Du, Minlan Yu, Ying Zhang - [[pdf]](https://arxiv.org/pdf/2512.10398)</summary>

**Abstract:** Real-world software engineering tasks require coding agents that can operate over massive repositories, sustain long-horizon sessions, and reliably coordinate complex toolchains at test time. Existing research-grade coding agents offer transparency but struggle when scaled to heavier, production-level workloads, while production-grade systems achieve strong practical performance but provide limited extensibility, interpretability, and controllability. We introduce the Confucius Code Agent (CCA), a software engineering agent that can operate at large-scale codebases. CCA is built on top of the Confucius SDK, an agent development platform structured around three complementary perspectives: Agent Experience (AX), User Experience (UX), and Developer Experience (DX). The SDK integrates a unified orchestrator with hierarchical working memory for long-context reasoning, a persistent note-taking system for cross-session continual learning, and a modular extension system for reliable tool use. In addition, we introduce a meta-agent that automates the synthesis, evaluation, and refinement of agent configurations through a build-test-improve loop, enabling rapid adaptation to new tasks, environments, and tool stacks. Instantiated with these mechanisms, CCA demonstrates strong performance on real-world software engineering tasks. On SWE-Bench-Pro, CCA reaches a Resolve@1 of 54.3%, exceeding prior research baselines and comparing favorably to commercial results, under identical repositories, model backends, and tool access.

**arXiv ID:** 2512.10398
</details>

<details>
<summary><strong>Look Twice before You Leap: A Rational Agent Framework for Localized Adversarial Anonymization</strong> - Donghang Duan, Xu Zheng, Yuefeng He, Chong Mu, Leyi Cai, Lizong Zhang - [[pdf]](https://arxiv.org/pdf/2512.06713)</summary>

**Abstract:** Current LLM-based text anonymization frameworks usually rely on remote API services from powerful LLMs, which creates an inherent privacy paradox: users must disclose data to untrusted third parties for guaranteed privacy preservation. Moreover, directly migrating current solutions to local small-scale models (LSMs) offers a suboptimal solution with severe utility collapse. Our work argues that this failure stems not merely from the capability deficits of LSMs, but significantly from the inherent irrationality of the greedy adversarial strategies employed by current state-of-the-art (SOTA) methods. To address this, we propose Rational Localized Adversarial Anonymization (RLAA), a fully localized and training-free framework featuring an Attacker-Arbitrator-Anonymizer architecture. We model the anonymization process as a trade-off between Marginal Privacy Gain (MPG) and Marginal Utility Cost (MUC), and demonstrate that greedy strategies tend to drift into an irrational state. Instead, RLAA introduces an arbitrator that acts as a rationality gatekeeper, validating the attacker's inference to filter out feedback providing negligible privacy benefits. This mechanism promotes a rational early-stopping criterion, and structurally prevents utility collapse. Extensive experiments on different benchmarks demonstrate that RLAA achieves a superior privacy-utility trade-off compared to strong baselines.

**arXiv ID:** 2512.06713
</details>

<details>
<summary><strong>Embodied4C: Measuring What Matters for Embodied Vision-Language Navigation</strong> - Tin Stribor Sohn, Maximilian Dillitzer, Jason J. Corso, Eric Sax - [[pdf]](https://arxiv.org/pdf/2512.18028)</summary>

**Abstract:** Vision-language navigation requires agents to reason and act under constraints of embodiment. While vision-language models (VLMs) demonstrate strong generalization, current benchmarks provide limited understanding of how embodiment -- i.e., the choice of physical platform, sensor configuration, and modality alignment -- influences perception, reasoning, and control. We introduce Embodied4C, a closed-loop benchmark designed as a Turing test for embodied reasoning. The benchmark evaluates the core embodied capabilities of VLMs across three heterogeneous embodiments -- autonomous vehicles, aerial drones, and robotic manipulators -- through approximately 1.1K one-shot reasoning questions and 58 goal-directed navigation tasks. These tasks jointly assess four foundational dimensions: semantic, spatial, temporal, and physical reasoning. Each embodiment presents dynamic sensor configurations and environment variations to probe generalization beyond platform-specific adaptation. To prevent embodiment overfitting, Embodied4C integrates domain-far queries targeting abstract and cross-context reasoning. Comprehensive evaluation across ten state-of-the-art VLMs and four embodied control baselines shows that cross-modal alignment and instruction tuning matter more than scale, while spatial and temporal reasoning remains the primary bottleneck for reliable embodied competence.

**arXiv ID:** 2512.18028
</details>

<details>
<summary><strong>Are All Data Necessary? Efficient Data Pruning for Large-scale Autonomous Driving Dataset via Trajectory Entropy Maximization</strong> - Zhaoyang Liu, Weitao Zhou, Junze Wen, Cheng Jing, Qian Cheng, Kun Jiang, Diange Yang - [[pdf]](https://arxiv.org/pdf/2512.19270)</summary>

**Abstract:** Collecting large-scale naturalistic driving data is essential for training robust autonomous driving planners. However, real-world datasets often contain a substantial amount of repetitive and low-value samples, which lead to excessive storage costs and bring limited benefits to policy learning. To address this issue, we propose an information-theoretic data pruning method that effectively reduces the training data volume without compromising model performance. Our approach evaluates the trajectory distribution information entropy of driving data and iteratively selects high-value samples that preserve the statistical characteristics of the original dataset in a model-agnostic manner. From a theoretical perspective, we show that maximizing trajectory entropy effectively constrains the Kullback-Leibler divergence between the pruned subset and the original data distribution, thereby maintaining generalization ability. Comprehensive experiments on the NuPlan benchmark with a large-scale imitation learning framework demonstrate that the proposed method can reduce the dataset size by up to 40% while maintaining closed-loop performance. This work provides a lightweight and theoretically grounded approach for scalable data management and efficient policy learning in autonomous driving systems.

**arXiv ID:** 2512.19270
</details>

<details>
<summary><strong>VERDI: VLM-Embedded Reasoning for Autonomous Driving</strong> - Bowen Feng, Zhiting Mei, Baiang Li, Julian Ost, Filippo Ghilotti, Roger Girgis, Anirudha Majumdar, Felix Heide - [[pdf]](https://arxiv.org/pdf/2505.15925)</summary>

**Abstract:** While autonomous driving (AD) stacks struggle with decision making under partial observability and real-world complexity, human drivers are capable of commonsense reasoning to make near-optimal decisions with limited information. Recent work has attempted to leverage finetuned Vision-Language Models (VLMs) for trajectory planning at inference time to emulate human behavior. Despite their success in benchmark evaluations, these methods are often impractical to deploy (a 70B parameter VLM inference at merely 8 tokens per second requires more than 160G of memory), and their monolithic network structure prohibits safety decomposition. To bridge this gap, we propose VLM-Embedded Reasoning for autonomous Driving (VERDI), a training-time framework that distills the reasoning process and commonsense knowledge of VLMs into the AD stack. VERDI augments modular differentiable end-to-end (e2e) AD models by aligning intermediate module outputs at the perception, prediction, and planning stages with text features explaining the driving reasoning process produced by VLMs. By encouraging alignment in latent space, VERDI enables the modular AD stack to internalize structured reasoning, without incurring the inference-time costs of large VLMs. We validate VERDI in both open-loop (NuScenes and Bench2Drive benchmarks) and closed-loop (HugSim Simulator) settings. We find that VERDI outperforms existing e2e methods that do not embed reasoning by up to 11% in $\ell_{2}$ distance and 11% in driving performance, while maintaining real-time inference speed.

**arXiv ID:** 2505.15925
</details>

<details>
<summary><strong>Towards a Multi-Embodied Grasping Agent</strong> - Roman Freiberg, Alexander Qualmann, Ngo Anh Vien, Gerhard Neumann - [[pdf]](https://arxiv.org/pdf/2510.27420)</summary>

**Abstract:** Multi-embodiment grasping focuses on developing approaches that exhibit generalist behavior across diverse gripper designs. Existing methods often learn the kinematic structure of the robot implicitly and face challenges due to the difficulty of sourcing the required large-scale data. In this work, we present a data-efficient, flow-based, equivariant grasp synthesis architecture that can handle different gripper types with variable degrees of freedom and successfully exploit the underlying kinematic model, deducing all necessary information solely from the gripper and scene geometry. Unlike previous equivariant grasping methods, we translated all modules from the ground up to JAX and provide a model with batching capabilities over scenes, grippers, and grasps, resulting in smoother learning, improved performance and faster inference time. Our dataset encompasses grippers ranging from humanoid hands to parallel yaw grippers and includes 25,000 scenes and 20 million grasps.

**arXiv ID:** 2510.27420
</details>

<details>
<summary><strong>TakeAD: Preference-based Post-optimization for End-to-end Autonomous Driving with Expert Takeover Data</strong> - Deqing Liu, Yinfeng Gao, Deheng Qian, Qichao Zhang, Xiaoqing Ye, Junyu Han, Yupeng Zheng, Xueyi Liu, Zhongpu Xia, Dawei Ding, Yifeng Pan, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2512.17370)</summary>

**Abstract:** Existing end-to-end autonomous driving methods typically rely on imitation learning (IL) but face a key challenge: the misalignment between open-loop training and closed-loop deployment. This misalignment often triggers driver-initiated takeovers and system disengagements during closed-loop execution. How to leverage those expert takeover data from disengagement scenarios and effectively expand the IL policy's capability presents a valuable yet unexplored challenge. In this paper, we propose TakeAD, a novel preference-based post-optimization framework that fine-tunes the pre-trained IL policy with this disengagement data to enhance the closed-loop driving performance. First, we design an efficient expert takeover data collection pipeline inspired by human takeover mechanisms in real-world autonomous driving systems. Then, this post optimization framework integrates iterative Dataset Aggregation (DAgger) for imitation learning with Direct Preference Optimization (DPO) for preference alignment. The DAgger stage equips the policy with fundamental capabilities to handle disengagement states through direct imitation of expert interventions. Subsequently, the DPO stage refines the policy's behavior to better align with expert preferences in disengagement scenarios. Through multiple iterations, the policy progressively learns recovery strategies for disengagement states, thereby mitigating the open-loop gap. Experiments on the closed-loop Bench2Drive benchmark demonstrate our method's effectiveness compared with pure IL methods, with comprehensive ablations confirming the contribution of each component.

**arXiv ID:** 2512.17370
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>Intelligent Human-Machine Partnership for Manufacturing: Enhancing Warehouse Planning through Simulation-Driven Knowledge Graphs and LLM Collaboration</strong> - Himabindu Thogaru, Saisubramaniam Gopalakrishnan, Zishan Ahmad, Anirudh Deodhar - [[pdf]](https://arxiv.org/pdf/2512.18265)</summary>

**Abstract:** Manufacturing planners face complex operational challenges that require seamless collaboration between human expertise and intelligent systems to achieve optimal performance in modern production environments. Traditional approaches to analyzing simulation-based manufacturing data often create barriers between human decision-makers and critical operational insights, limiting effective partnership in manufacturing planning. Our framework establishes a collaborative intelligence system integrating Knowledge Graphs and Large Language Model-based agents to bridge this gap, empowering manufacturing professionals through natural language interfaces for complex operational analysis. The system transforms simulation data into semantically rich representations, enabling planners to interact naturally with operational insights without specialized expertise. A collaborative LLM agent works alongside human decision-makers, employing iterative reasoning that mirrors human analytical thinking while generating precise queries for knowledge extraction and providing transparent validation. This partnership approach to manufacturing bottleneck identification, validated through operational scenarios, demonstrates enhanced performance while maintaining human oversight and decision authority. For operational inquiries, the system achieves near-perfect accuracy through natural language interaction. For investigative scenarios requiring collaborative analysis, we demonstrate the framework's effectiveness in supporting human experts to uncover interconnected operational issues that enhance understanding and decision-making. This work advances collaborative manufacturing by creating intuitive methods for actionable insights, reducing cognitive load while amplifying human analytical capabilities in evolving manufacturing ecosystems.

**arXiv ID:** 2512.18265
</details>

<details>
<summary><strong>ESearch-R1: Learning Cost-Aware MLLM Agents for Interactive Embodied Search via Reinforcement Learning</strong> - Weijie Zhou, Xuangtang Xiong, Ye Tian, Lijun Yue, Xinyu Wu, Wei Li, Chaoyang Zhao, Honghui Dong, Ming Tang, Jinqiao Wang, Zhengyou Zhang - [[pdf]](https://arxiv.org/pdf/2512.18571)</summary>

**Abstract:** Multimodal Large Language Models (MLLMs) have empowered embodied agents with remarkable capabilities in planning and reasoning. However, when facing ambiguous natural language instructions (e.g., "fetch the tool" in a cluttered room), current agents often fail to balance the high cost of physical exploration against the cognitive cost of human interaction. They typically treat disambiguation as a passive perception problem, lacking the strategic reasoning to minimize total task execution costs. To bridge this gap, we propose ESearch-R1, a cost-aware embodied reasoning framework that unifies interactive dialogue (Ask), episodic memory retrieval (GetMemory), and physical navigation (Navigate) into a single decision process. We introduce HC-GRPO (Heterogeneous Cost-Aware Group Relative Policy Optimization). Unlike traditional PPO which relies on a separate value critic, HC-GRPO optimizes the MLLM by sampling groups of reasoning trajectories and reinforcing those that achieve the optimal trade-off between information gain and heterogeneous costs (e.g., navigate time, and human attention). Extensive experiments in AI2-THOR demonstrate that ESearch-R1 significantly outperforms standard ReAct-based agents. It improves task success rates while reducing total operational costs by approximately 50\%, validating the effectiveness of GRPO in aligning MLLM agents with physical world constraints.

**arXiv ID:** 2512.18571
</details>

<details>
<summary><strong>LLM Agents Implement an NLG System from Scratch: Building Interpretable Rule-Based RDF-to-Text Generators</strong> - Mateusz Lango, Ondřej Dušek - [[pdf]](https://arxiv.org/pdf/2512.18360)</summary>

**Abstract:** We present a novel neurosymbolic framework for RDF-to-text generation, in which the model is "trained" through collaborative interactions among multiple LLM agents rather than traditional backpropagation. The LLM agents produce rule-based Python code for a generator for the given domain, based on RDF triples only, with no in-domain human reference texts. The resulting system is fully interpretable, requires no supervised training data, and generates text nearly instantaneously using only a single CPU. Our experiments on the WebNLG and OpenDialKG data show that outputs produced by our approach reduce hallucination, with only slight fluency penalties compared to finetuned or prompted language models

**arXiv ID:** 2512.18360
</details>

<details>
<summary><strong>Toward Training Superintelligent Software Agents through Self-Play SWE-RL</strong> - Yuxiang Wei, Zhiqing Sun, Emily McMilin, Jonas Gehring, David Zhang, Gabriel Synnaeve, Daniel Fried, Lingming Zhang, Sida Wang - [[pdf]](https://arxiv.org/pdf/2512.18552)</summary>

**Abstract:** While current software agents powered by large language models (LLMs) and agentic reinforcement learning (RL) can boost programmer productivity, their training data (e.g., GitHub issues and pull requests) and environments (e.g., pass-to-pass and fail-to-pass tests) heavily depend on human knowledge or curation, posing a fundamental barrier to superintelligence. In this paper, we present Self-play SWE-RL (SSR), a first step toward training paradigms for superintelligent software agents. Our approach takes minimal data assumptions, only requiring access to sandboxed repositories with source code and installed dependencies, with no need for human-labeled issues or tests. Grounded in these real-world codebases, a single LLM agent is trained via reinforcement learning in a self-play setting to iteratively inject and repair software bugs of increasing complexity, with each bug formally specified by a test patch rather than a natural language issue description. On the SWE-bench Verified and SWE-Bench Pro benchmarks, SSR achieves notable self-improvement (+10.4 and +7.8 points, respectively) and consistently outperforms the human-data baseline over the entire training trajectory, despite being evaluated on natural language issues absent from self-play. Our results, albeit early, suggest a path where agents autonomously gather extensive learning experiences from real-world software repositories, ultimately enabling superintelligent systems that exceed human capabilities in understanding how systems are constructed, solving novel challenges, and autonomously creating new software from scratch.

**arXiv ID:** 2512.18552
</details>

<details>
<summary><strong>Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement</strong> - Saman Forouzandeh, Wei Peng, Parham Moradi, Xinghuo Yu, Mahdi Jalili - [[pdf]](https://arxiv.org/pdf/2512.18950)</summary>

**Abstract:** We present MACLA, a framework that decouples reasoning from learning by maintaining a frozen large language model while performing all adaptation in an external hierarchical procedural memory. MACLA extracts reusable procedures from trajectories, tracks reliability via Bayesian posteriors, selects actions through expected-utility scoring, and refines procedures by contrasting successes and failures. Across four benchmarks (ALFWorld, WebShop, TravelPlanner, InterCodeSQL), MACLA achieves 78.1 percent average performance, outperforming all baselines. On ALFWorld unseen tasks, MACLA reaches 90.3 percent with 3.1 percent positive generalization. The system constructs memory in 56 seconds, 2800 times faster than the state-of-the-art LLM parameter-training baseline, compressing 2851 trajectories into 187 procedures. Experimental results demonstrate that structured external memory with Bayesian selection and contrastive refinement enables sample-efficient, interpretable, and continually improving agents without LLM parameter updates.

**arXiv ID:** 2512.18950
</details>

<details>
<summary><strong>WebATLAS: An LLM Agent with Experience-Driven Memory and Action Simulation</strong> - Jiali Cheng, Anjishnu Kumar, Roshan Lal, Rishi Rajasekaran, Hani Ramezani, Omar Zia Khan, Oleg Rokhlenko, Sunny Chiu-Webster, Gang Hua, Hadi Amiri - [[pdf]](https://arxiv.org/pdf/2510.22732)</summary>

**Abstract:** Large Language Model (LLM) web agents often struggle with long-horizon web navigation and web task completion in new websites, producing inefficient action sequences unless fine-tuned on environment-specific data. We show that experience-driven memory, combined with look-ahead action simulation, is sufficient for LLM agents to adapt to unseen web environments by remembering past failures and predicting the consequences of future actions. We introduce WebATLAS (Actor-Critic Task-completion with Look-ahead Action Simulation), a memory-augmented LLM web agent that learns a lightweight internal model of the environment from interaction experience and performs hypothetical action rollouts before acting in the real world. WebATLAS builds a persistent cognitive map via curiosity-driven exploration, stores interaction outcomes as experience-based memory, and evaluates candidate actions in cognitive space using a planner--simulator--critic loop. This enables the agent to reuse past experience, avoid previously unsuccessful behaviors, and generate more efficient plans. We evaluate WebATLAS on the WebArena-Lite benchmark for autonomous web navigation and demonstrate a success rate of 63%, outperforming the previous state-of-the-art at 53.9%. Unlike previous systems, our modular architecture requires no website-specific LLM fine-tuning. Ablation studies confirm that experience-driven memory, look-ahead action simulation, and hierarchical replanning play complementary roles in enabling robust, training-free web agents.

**arXiv ID:** 2510.22732
</details>

<details>
<summary><strong>GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators</strong> - Jiacheng Guo, Ling Yang, Peter Chen, Qixin Xiao, Yinjie Wang, Xinzhe Juan, Jiahao Qiu, Ke Shen, Mengdi Wang - [[pdf]](https://arxiv.org/pdf/2512.19682)</summary>

**Abstract:** Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutionary game between an agent and a scalable, generative environment simulator. Unlike traditional methods that evolve models on static datasets, GenEnv instantiates a dataevolving: the simulator acts as a dynamic curriculum policy, continuously generating tasks specifically tailored to the agent's ``zone of proximal development''. This process is guided by a simple but effective $\alpha$-Curriculum Reward, which aligns task difficulty with the agent's current capabilities. We evaluate GenEnv on five benchmarks, including API-Bank, ALFWorld, BFCL, Bamboogle, and TravelPlanner. Across these tasks, GenEnv improves agent performance by up to \textbf{+40.3\%} over 7B baselines and matches or exceeds the average performance of larger models. Compared to Gemini 2.5 Pro-based offline data augmentation, GenEnv achieves better performance while using 3.3$\times$ less data. By shifting from static supervision to adaptive simulation, GenEnv provides a data-efficient pathway for scaling agent capabilities.

**arXiv ID:** 2512.19682
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>Rethinking Multi-Agent Intelligence Through the Lens of Small-World Networks</strong> - Boxuan Wang, Zhuoyun Li, Xiaowei Huang, Yi Dong - [[pdf]](https://arxiv.org/pdf/2512.18094)</summary>

**Abstract:** Large language models (LLMs) have enabled multi-agent systems (MAS) in which multiple agents argue, critique, and coordinate to solve complex tasks, making communication topology a first-class design choice. Yet most existing LLM-based MAS either adopt fully connected graphs, simple sparse rings, or ad-hoc dynamic selection, with little structural guidance. In this work, we revisit classic theory on small-world (SW) networks and ask: what changes if we treat SW connectivity as a design prior for MAS? We first bridge insights from neuroscience and complex networks to MAS, highlighting how SW structures balance local clustering and long-range integration. Using multi-agent debate (MAD) as a controlled testbed, experiment results show that SW connectivity yields nearly the same accuracy and token cost, while substantially stabilizing consensus trajectories. Building on this, we introduce an uncertainty-guided rewiring scheme for scaling MAS, where long-range shortcuts are added between epistemically divergent agents using LLM-oriented uncertainty signals (e.g., semantic entropy). This yields controllable SW structures that adapt to task difficulty and agent heterogeneity. Finally, we discuss broader implications of SW priors for MAS design, framing them as stabilizers of reasoning, enhancers of robustness, scalable coordinators, and inductive biases for emergent cognitive roles.

**arXiv ID:** 2512.18094
</details>

<details>
<summary><strong>IntelliCode: A Multi-Agent LLM Tutoring System with Centralized Learner Modeling</strong> - Jones David, Shreya Ghosh - [[pdf]](https://arxiv.org/pdf/2512.18669)</summary>

**Abstract:** LLM-based tutors are typically single-turn assistants that lack persistent representations of learner knowledge, making it difficult to provide principled, transparent, and long-term pedagogical support. We introduce IntelliCode, a multi-agent LLM tutoring system built around a centralized, versioned learner state that integrates mastery estimates, misconceptions, review schedules, and engagement signals. A StateGraph Orchestrator coordinates six specialized agents: skill assessment, learner profiling, graduated hinting, curriculum selection, spaced repetition, and engagement monitoring, each operating as a pure transformation over the shared state under a single-writer policy. This architecture enables auditable mastery updates, proficiency-aware hints, dependency-aware curriculum adaptation, and safety-aligned prompting.
The demo showcases an end-to-end tutoring workflow: a learner attempts a DSA problem, receives a conceptual hint when stuck, submits a corrected solution, and immediately sees mastery updates and a personalized review interval. We report validation results with simulated learners, showing stable state updates, improved task success with graduated hints, and diverse curriculum coverage. IntelliCode demonstrates how persistent learner modeling, orchestrated multi-agent reasoning, and principled instructional design can be combined to produce transparent and reliable LLM-driven tutoring.

**arXiv ID:** 2512.18669
</details>

<details>
<summary><strong>Byzantine Fault-Tolerant Multi-Agent System for Healthcare: A Gossip Protocol Approach to Secure Medical Message Propagation</strong> - Nihir Chadderwala - [[pdf]](https://arxiv.org/pdf/2512.17913)</summary>

**Abstract:** Recent advances in generative AI have enabled sophisticated multi-agent architectures for healthcare, where large language models power collaborative clinical decision-making. However, these distributed systems face critical challenges in ensuring message integrity and fault tolerance when operating in adversarial or untrusted this http URL paper presents a novel Byzantine fault-tolerant multi-agent system specifically designed for healthcare applications, integrating gossip-based message propagation with cryptographic validation mechanisms. Our system employs specialized AI agents for diagnosis, treatment planning, emergency response, and data analysis, coordinated through a Byzantine consensus protocol that tolerates up to f faulty nodes among n = 3f + 1 total nodes. We implement a gossip protocol for decentralized message dissemination, achieving consensus with 2f + 1 votes while maintaining system operation even under Byzantine failures. Experimental results demonstrate that our approach successfully validates medical messages with cryptographic signatures, prevents replay attacks through timestamp validation, and maintains consensus accuracy of 100% with up to 33% Byzantine nodes. The system provides real-time visualization of consensus rounds, vote tallies, and network topology, enabling transparent monitoring of fault-tolerant operations. This work contributes a practical framework for building secure, resilient healthcare multi-agent systems capable of collaborative medical decision-making in untrusted environments.

**arXiv ID:** 2512.17913
</details>

<details>
<summary><strong>A Multi-agent Text2SQL Framework using Small Language Models and Execution Feedback</strong> - Thanh Dat Hoang, Thanh Trung Huynh, Matthias Weidlich, Thanh Tam Nguyen, Tong Chen, Hongzhi Yin, Quoc Viet Hung Nguyen - [[pdf]](https://arxiv.org/pdf/2512.18622)</summary>

**Abstract:** Text2SQL, the task of generating SQL queries from natural language text, is a critical challenge in data engineering. Recently, Large Language Models (LLMs) have demonstrated superior performance for this task due to their advanced comprehension and generation capabilities. However, privacy and cost considerations prevent companies from using Text2SQL solutions based on external LLMs offered as a service. Rather, small LLMs (SLMs) that are openly available and can hosted in-house are adopted. These SLMs, in turn, lack the generalization capabilities of larger LLMs, which impairs their effectiveness for complex tasks such as Text2SQL. To address these limitations, we propose MATS, a novel Text2SQL framework designed specifically for SLMs. MATS uses a multi-agent mechanism that assigns specialized roles to auxiliary agents, reducing individual workloads and fostering interaction. A training scheme based on reinforcement learning aligns these agents using feedback obtained during execution, thereby maintaining competitive performance despite a limited LLM size. Evaluation results using on benchmark datasets show that MATS, deployed on a single- GPU server, yields accuracy that are on-par with large-scale LLMs when using significantly fewer parameters. Our source code and data are available at this https URL.

**arXiv ID:** 2512.18622
</details>

<details>
<summary><strong>Explainable and Fine-Grained Safeguarding of LLM Multi-Agent Systems via Bi-Level Graph Anomaly Detection</strong> - Junjun Pan, Yixin Liu, Rui Miao, Kaize Ding, Yu Zheng, Quoc Viet Hung Nguyen, Alan Wee-Chung Liew, Shirui Pan - [[pdf]](https://arxiv.org/pdf/2512.18733)</summary>

**Abstract:** Large language model (LLM)-based multi-agent systems (MAS) have shown strong capabilities in solving complex tasks. As MAS become increasingly autonomous in various safety-critical tasks, detecting malicious agents has become a critical security concern. Although existing graph anomaly detection (GAD)-based defenses can identify anomalous agents, they mainly rely on coarse sentence-level information and overlook fine-grained lexical cues, leading to suboptimal performance. Moreover, the lack of interpretability in these methods limits their reliability and real-world applicability. To address these limitations, we propose XG-Guard, an explainable and fine-grained safeguarding framework for detecting malicious agents in MAS. To incorporate both coarse and fine-grained textual information for anomalous agent identification, we utilize a bi-level agent encoder to jointly model the sentence- and token-level representations of each agent. A theme-based anomaly detector further captures the evolving discussion focus in MAS dialogues, while a bi-level score fusion mechanism quantifies token-level contributions for explanation. Extensive experiments across diverse MAS topologies and attack scenarios demonstrate robust detection performance and strong interpretability of XG-Guard.

**arXiv ID:** 2512.18733
</details>

<details>
<summary><strong>Q-KVComm: Efficient Multi-Agent Communication Via Adaptive KV Cache Compression</strong> - Boris Kriuk, Logic Ng - [[pdf]](https://arxiv.org/pdf/2512.17914)</summary>

**Abstract:** Multi-agent Large Language Model (LLM) systems face a critical bottleneck: redundant transmission of contextual information between agents consumes excessive bandwidth and computational resources. Traditional approaches discard internal semantic representations and transmit raw text, forcing receiving agents to recompute similar representations from scratch. We introduce Q-KVComm, a new protocol that enables direct transmission of compressed key-value (KV) cache representations between LLM agents. Q-KVComm combines three key innovations: (1) adaptive layer-wise quantization that allocates variable bit-widths based on sensitivity profiling, (2) hybrid information extraction that preserves critical facts across content domains, and (3) heterogeneous model calibration establishing cross-architecture communication. Extensive experiments across three diverse question-answering datasets demonstrate that Q-KVComm achieves 5-6x compression ratios while maintaining semantic fidelity, with coherence quality scores above 0.77 across all scenarios. The protocol exhibits robust performance across model sizes (1.1B-1.5B parameters) and adapts to real-world applications including conversational QA and multi-hop reasoning. Our work establishes a new paradigm for LLM agent communication, shifting from text-based to representation-based information exchange.

**arXiv ID:** 2512.17914
</details>

<details>
<summary><strong>Networked Communication for Decentralised Cooperative Agents in Mean-Field Control</strong> - Patrick Benjamin, Alessandro Abate - [[pdf]](https://arxiv.org/pdf/2503.09400)</summary>

**Abstract:** The mean-field framework has been used to find approximate solutions to problems involving very large populations of symmetric, anonymous agents, which may be intractable by other methods. The cooperative mean-field control (MFC) problem has received less attention than the non-cooperative mean-field game (MFG), despite the former potentially being more useful as a tool for engineering large-scale collective behaviours. Decentralised communication algorithms have recently been introduced to MFGs, giving benefits to learning speed and robustness. Inspired by this, we introduce networked communication to MFC - where populations arguably have broader incentive to communicate - and in particular to the setting where decentralised agents learn online from a single, non-episodic run of the empirical system. We adapt recent MFG algorithms to this new setting, as well as contributing a novel sub-routine allowing networked agents to estimate the global average reward from their local neighbourhood. Previous theoretical analysis of decentralised communication in MFGs does not extend trivially to MFC. We therefore contribute new theory proving that in MFC the networked communication scheme allows agents to increase social welfare faster than under both of the two typical alternative architectures, namely independent and centralised learning. We provide experiments that support this new result across different classes of cooperative game, and also give numerous ablation studies and additional experiments concerning numbers of communication round and robustness to communication failures.

**arXiv ID:** 2503.09400
</details>

<details>
<summary><strong>SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication</strong> - Ruijia Zhang, Xinyan Zhao, Ruixiang Wang, Sigen Chen, Guibin Zhang, An Zhang, Kun Wang, Qingsong Wen - [[pdf]](https://arxiv.org/pdf/2508.11733)</summary>

**Abstract:** LLM-based multi-agent systems exhibit strong collaborative capabilities but often suffer from redundant communication and excessive token overhead. Existing methods typically enhance efficiency through pretrained GNNs or greedy algorithms, but often isolate pre- and post-task optimization, lacking a unified strategy. To this end, we present SafeSieve, a progressive and adaptive multi-agent pruning algorithm that dynamically refines the inter-agent communication through a novel dual-mechanism. SafeSieve integrates initial LLM-based semantic evaluation with accumulated performance feedback, enabling a smooth transition from heuristic initialization to experience-driven refinement. Unlike existing greedy Top-k pruning methods, SafeSieve employs 0-extension clustering to preserve structurally coherent agent groups while eliminating ineffective links. Experiments across benchmarks (SVAMP, HumanEval, etc.) showcase that SafeSieve achieves 94.01% average accuracy while reducing token usage by 12.4%-27.8%. Results further demonstrate robustness under prompt injection attacks (1.23% average accuracy drop). In heterogeneous settings, SafeSieve reduces deployment costs by 13.3% while maintaining performance. These results establish SafeSieve as an efficient, GPU-free, and scalable framework for practical multi-agent systems. Our code can be found here: this https URL

**arXiv ID:** 2508.11733
</details>

<details>
<summary><strong>RadAgents: Multimodal Agentic Reasoning for Chest X-ray Interpretation with Radiologist-like Workflows</strong> - Kai Zhang, Corey D Barrett, Jangwon Kim, Lichao Sun, Tara Taghavi, Krishnaram Kenthapadi - [[pdf]](https://arxiv.org/pdf/2509.20490)</summary>

**Abstract:** Agentic systems offer a potential path to solve complex clinical tasks through collaboration among specialized agents, augmented by tool use and external knowledge bases. Nevertheless, for chest X-ray (CXR) interpretation, prevailing methods remain limited: (i) reasoning is frequently neither clinically interpretable nor aligned with guidelines, reflecting mere aggregation of tool outputs; (ii) multimodal evidence is insufficiently fused, yielding text-only rationales that are not visually grounded; and (iii) systems rarely detect or resolve cross-tool inconsistencies and provide no principled verification mechanisms. To bridge the above gaps, we present RadAgents, a multi-agent framework that couples clinical priors with task-aware multimodal reasoning and encodes a radiologist-style workflow into a modular, auditable pipeline. In addition, we integrate grounding and multimodal retrieval-augmentation to verify and resolve context conflicts, resulting in outputs that are more reliable, transparent, and consistent with clinical practice.

**arXiv ID:** 2509.20490
</details>

<details>
<summary><strong>HyperAgent: Leveraging Hypergraphs for Topology Optimization in Multi-Agent Communication</strong> - Heng Zhang, Yuling Shi, Xiaodong Gu, Zijian Zhang, Haochen You, Lubin Gan, Yilei Yuan, Jin Huang - [[pdf]](https://arxiv.org/pdf/2510.10611)</summary>

**Abstract:** Recent advances in large language model-powered multi-agent systems have demonstrated remarkable collective intelligence through effective communication. However, existing approaches face two primary challenges: (i) \textit{Ineffective group collaboration modeling}, as they rely on pairwise edge representations in graph structures, limiting their ability to capture relationships among multiple agents; and (ii) \textit{Limited task-adaptiveness in communication topology design}, leading to excessive communication cost for simple tasks and insufficient coordination for complex scenarios. These issues restrict the scalability and practical deployment of adaptive collaboration frameworks. To address these challenges, we propose \textbf{HyperAgent}, a hypergraph-based framework that optimizes communication topologies and effectively captures group collaboration patterns using direct hyperedge representations. Unlike edge-based approaches, HyperAgent uses hyperedges to link multiple agents within the same subtask and employs hypergraph convolutional layers to achieve one-step information aggregation in collaboration groups. Additionally, it incorporates a variational autoencoder framework with sparsity regularization to dynamically adjust hypergraph topologies based on task complexity. Experiments highlight the superiority of HyperAgent in both performance and efficiency. For instance, on GSM8K, HyperAgent achieves 95.07\% accuracy while reducing token consumption by 25.33\%, demonstrating the potential of hypergraph-based optimization for multi-agent communication.

**arXiv ID:** 2510.10611
</details>

<details>
<summary><strong>MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems</strong> - Barak Or - [[pdf]](https://arxiv.org/pdf/2511.20663)</summary>

**Abstract:** Reliability in multi-agent systems (MAS) built on large language models is increasingly limited by cognitive failures rather than infrastructure faults. Existing observability tools describe failures but do not quantify how quickly distributed reasoning recovers once coherence is lost. We introduce MTTR-A (Mean Time-to-Recovery for Agentic Systems), a runtime reliability metric that measures cognitive recovery latency in MAS. MTTR-A adapts classical dependability theory to agentic orchestration, capturing the time required to detect reasoning drift and restore coherent operation. We further define complementary metrics, including MTBF and a normalized recovery ratio (NRR), and establish theoretical bounds linking recovery latency to long-run cognitive uptime. Using a LangGraph-based benchmark with simulated drift and reflex recovery, we empirically demonstrate measurable recovery behavior across multiple reflex strategies. This work establishes a quantitative foundation for runtime cognitive dependability in distributed agentic systems.

**arXiv ID:** 2511.20663
</details>

<details>
<summary><strong>MAPPO-LCR: Multi-Agent Proximal Policy Optimization with Local Cooperation Reward in Spatial Public Goods Games</strong> - Zhaoqilin Yang, Axin Xiang, Kedi Yang, Tianjun Liu, Youliang Tian - [[pdf]](https://arxiv.org/pdf/2512.17187)</summary>

**Abstract:** Spatial public goods games model collective dilemmas where individual payoffs depend on population-level strategy configurations. Most existing studies rely on evolutionary update rules or value-based reinforcement learning methods. These approaches struggle to represent payoff coupling and non-stationarity in large interacting populations. This work introduces Multi-Agent Proximal Policy Optimization (MAPPO) into spatial public goods games for the first time. In these games, individual returns are intrinsically coupled through overlapping group interactions. Proximal Policy Optimization (PPO) treats agents as independent learners and ignores this coupling during value estimation. MAPPO addresses this limitation through a centralized critic that evaluates joint strategy configurations. To study neighborhood-level cooperation signals under this framework, we propose MAPPO with Local Cooperation Reward, termed MAPPO-LCR. The local cooperation reward aligns policy updates with surrounding cooperative density without altering the original game structure. MAPPO-LCR preserves decentralized execution while enabling population-level value estimation during training. Extensive simulations demonstrate stable cooperation emergence and reliable convergence across enhancement factors. Statistical analyses further confirm the learning advantage of MAPPO over PPO in spatial public goods games.

**arXiv ID:** 2512.17187
</details>

<details>
<summary><strong>JustAct+: A Framework for Auditable Multi-Agent Systems Regulated by Inter-Organisational Policies</strong> - Christopher A. Esterhuyse, Tim Müller, L. Thomas van Binsbergen - [[pdf]](https://arxiv.org/pdf/2502.00138)</summary>

**Abstract:** In open multi-agent agent systems that cross organisational boundaries, agent actions must be regulated by complex policies. Consider medical data processing systems, which must observe generic laws (e.g., EU data protection regulations) and also specific participants' resource conditions (e.g., Bob consents to sharing his X-Rays with EU hospitals). Presently, we address the implementation of these systems as distributed software. Solutions to key sub-problems are available: existing policy languages capture the necessary normative concepts and formalise the computational representation and reasoning about policies, and existing distributed algorithms and protocols coordinate agents' changing actions and policies. But which policies and protocols are useful in application? With the JustAct framework, we characterise a class of multi-agent systems where actors justify their actions with sufficient policy information collected from dynamic policy statements and agreements. We prove key properties of these systems, e.g., any decision that an action is permitted now cannot be refuted later, regardless of any added statements or updated agreements. We study a particular instance of the framework by specifying (in Rocq) and implementing (in Rust) a particular policy language and runtime system for mediating agent communications. We demonstrate and assess JustAct via a case study of this implementation: we reproduce the usage scenarios of Brane, an existing policy-regulated, inter-domain, medical data processing system.

**arXiv ID:** 2502.00138
</details>

<details>
<summary><strong>Scaling up Stability: Reinforcement Learning for Distributed Control of Networked Systems in the Space of Stabilizing Policies</strong> - John Cao, Luca Furieri - [[pdf]](https://arxiv.org/pdf/2512.18540)</summary>

**Abstract:** We study distributed control of networked systems through reinforcement learning, where neural policies must be simultaneously scalable, expressive and stabilizing. We introduce a policy parameterization that embeds Graph Neural Networks (GNNs) into a Youla-like magnitude-direction parameterization, yielding distributed stochastic controllers that guarantee network-level closed-loop stability by design. The magnitude is implemented as a stable operator consisting of a GNN acting on disturbance feedback, while the direction is a GNN acting on local observations. We prove robustness of the closed loop to perturbations in both the graph topology and model parameters, and show how to integrate our parameterization with Proximal Policy Optimization. Experiments on a multi-agent navigation task show that policies trained on small networks transfer directly to larger ones and unseen network topologies, achieve higher returns and lower variance than a state-of-the-art MARL baseline while preserving stability.

**arXiv ID:** 2512.18540
</details>

</details>

<details open>
<summary><h2>Other Agent Research (12 papers)</h2></summary>

<details>
<summary><strong>Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap</strong> - Zijun Wang, Yijiahao Qi, Hanqiu Chen, Zishen Wan, Gongjin Sun, Dongyang Li, Shuyi Pei, Cong Hao - [[pdf]](https://arxiv.org/pdf/2512.18126)</summary>

**Abstract:** Mixture-of-Agents (MoA) inference can suffer from dense inter-agent communication and low hardware utilization, which jointly inflate serving latency. We present a serving design that targets these bottlenecks through an algorithm-system co-design. First, we replace dense agent interaction graphs with a hierarchical tree topology that induces structured sparsity in inter-agent communication. Second, we introduce a runtime adaptive mechanism that selectively terminates or skips downstream agent invocations using semantic agreement and confidence signals from intermediate outputs. Third, we pipeline agent execution by overlapping incremental prefilling with decoding across dependency-related agents, improving utilization and reducing inference latency. Across representative tasks, this approach substantially reduces end-to-end latency (up to 90%) while maintaining comparable accuracy (within $\pm$1%) relative to dense-connectivity MoA baselines, and can improve accuracy in certain settings.

**arXiv ID:** 2512.18126
</details>

<details>
<summary><strong>Sophia: A Persistent Agent Framework of Artificial Life</strong> - Mingyang Sun, Feng Hong, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2512.18202)</summary>

**Abstract:** The development of LLMs has elevated AI agents from task-specific tools to long-lived, decision-making entities. Yet, most architectures remain static and reactive, tethered to manually defined, narrow scenarios. These systems excel at perception (System 1) and deliberation (System 2) but lack a persistent meta-layer to maintain identity, verify reasoning, and align short-term actions with long-term survival. We first propose a third stratum, System 3, that presides over the agent's narrative identity and long-horizon adaptation. The framework maps selected psychological constructs to concrete computational modules, thereby translating abstract notions of artificial life into implementable design requirements. The ideas coalesce in Sophia, a "Persistent Agent" wrapper that grafts a continuous self-improvement loop onto any LLM-centric System 1/2 stack. Sophia is driven by four synergistic mechanisms: process-supervised thought search, narrative memory, user and self modeling, and a hybrid reward system. Together, they transform repetitive reasoning into a self-driven, autobiographical process, enabling identity continuity and transparent behavioral explanations. Although the paper is primarily conceptual, we provide a compact engineering prototype to anchor the discussion. Quantitatively, Sophia independently initiates and executes various intrinsic tasks while achieving an 80% reduction in reasoning steps for recurring operations. Notably, meta-cognitive persistence yielded a 40% gain in success for high-complexity tasks, effectively bridging the performance gap between simple and sophisticated goals. Qualitatively, System 3 exhibited a coherent narrative identity and an innate capacity for task organization. By fusing psychological insight with a lightweight reinforcement-learning core, the persistent agent architecture advances a possible practical pathway toward artificial life.

**arXiv ID:** 2512.18202
</details>

<details>
<summary><strong>Securing Agentic AI Systems -- A Multilayer Security Framework</strong> - Sunil Arora, John Hastings - [[pdf]](https://arxiv.org/pdf/2512.18043)</summary>

**Abstract:** Securing Agentic Artificial Intelligence (AI) systems requires addressing the complex cyber risks introduced by autonomous, decision-making, and adaptive behaviors. Agentic AI systems are increasingly deployed across industries, organizations, and critical sectors such as cybersecurity, finance, and healthcare. However, their autonomy introduces unique security challenges, including unauthorized actions, adversarial manipulation, and dynamic environmental interactions. Existing AI security frameworks do not adequately address these challenges or the unique nuances of agentic AI. This research develops a lifecycle-aware security framework specifically designed for agentic AI systems using the Design Science Research (DSR) methodology. The paper introduces MAAIS, an agentic security framework, and the agentic AI CIAA (Confidentiality, Integrity, Availability, and Accountability) concept. MAAIS integrates multiple defense layers to maintain CIAA across the AI lifecycle. Framework validation is conducted by mapping with the established MITRE ATLAS (Adversarial Threat Landscape for Artificial-Intelligence Systems) AI tactics. The study contributes a structured, standardized, and framework-based approach for the secure deployment and governance of agentic AI in enterprise environments. This framework is intended for enterprise CISOs, security, AI platform, and engineering teams and offers a detailed step-by-step approach to securing agentic AI workloads.

**arXiv ID:** 2512.18043
</details>

<details>
<summary><strong>Quantifying the Lifelong Impact of Resilience Interventions via Agent-Based LLM Simulation</strong> - Vivienne L'Ecuyer Ming - [[pdf]](https://arxiv.org/pdf/2512.18803)</summary>

**Abstract:** Establishing the long-term, causal impact of psychological interventions on life outcomes is a grand challenge for the social sciences, caught between the limitations of correlational longitudinal studies and short-term randomized controlled trials (RCTs). This paper introduces Large-Scale Agent-based Longitudinal Simulation (LALS), a framework that resolves this impasse by simulating multi-decade, counterfactual life trajectories. The methodology employs a "digital clone" design where 2,500 unique LLM-based agent personas (grounded in a curated corpus of 3,917 empirical research articles) are each cloned across a 2x2 factorial experiment. Specifically, the simulation models the efficacy of extended psychological resilience training (Intervention vs. Control) either in childhood or as a young adult (age 6 vs. age 18). Comparing digital clones enables exceptionally precise causal inference. The simulation provides a quantitative, causal estimate of a resilience intervention's lifelong effects, revealing significant reductions in mortality, a lower incidence of dementia, and a substantial increase in accumulated wealth. Crucially, the results uncover a crucial developmental window: the intervention administered at age 6 produced more than double the positive impact on lifetime wealth compared to the same intervention at age 18. These benefits were most pronounced for agents from low-socioeconomic backgrounds, highlighting a powerful buffering effect. The LALS framework serves as a "computational wind tunnel" for social science, offering a new paradigm for generating and testing causal hypotheses about the complex, lifelong dynamics that shape human capital and well-being.

**arXiv ID:** 2512.18803
</details>

<details>
<summary><strong>Stop saying LLM: Large Discourse Models (LDM) and Artificial Discursive Agent (ADA)?</strong> - Amar Lakel - [[pdf]](https://arxiv.org/pdf/2512.19117)</summary>

**Abstract:** This paper proposes an epistemological shift in the analysis of large generative models, replacing the category ''Large Language Models'' (LLM) with that of ''Large Discourse Models'' (LDM), and then with that of Artificial Discursive Agent (ADA). The theoretical framework is based on an ontological triad distinguishing three regulatory instances: the apprehension of the phenomenal regularities of the referential world, the structuring of embodied cognition, and the structural-linguistic sedimentation of the utterance within a socio-historical context. LDMs, operating on the product of these three instances (the document), model the discursive projection of a portion of human experience reified by the learning corpus. The proposed program aims to replace the ''fascination/fear'' dichotomy with public trials and procedures that make the place, uses, and limits of artificial discursive agents in contemporary social space decipherable, situating this approach within a perspective of governance and co-regulation involving the State, industry, civil society, and academia.

**arXiv ID:** 2512.19117
</details>

<details>
<summary><strong>Affordance RAG: Hierarchical Multimodal Retrieval with Affordance-Aware Embodied Memory for Mobile Manipulation</strong> - Ryosuke Korekata, Quanting Xie, Yonatan Bisk, Komei Sugiura - [[pdf]](https://arxiv.org/pdf/2512.18987)</summary>

**Abstract:** In this study, we address the problem of open-vocabulary mobile manipulation, where a robot is required to carry a wide range of objects to receptacles based on free-form natural language instructions. This task is challenging, as it involves understanding visual semantics and the affordance of manipulation actions. To tackle these challenges, we propose Affordance RAG, a zero-shot hierarchical multimodal retrieval framework that constructs Affordance-Aware Embodied Memory from pre-explored images. The model retrieves candidate targets based on regional and visual semantics and reranks them with affordance scores, allowing the robot to identify manipulation options that are likely to be executable in real-world environments. Our method outperformed existing approaches in retrieval performance for mobile manipulation instruction in large-scale indoor environments. Furthermore, in real-world experiments where the robot performed mobile manipulation in indoor environments based on free-form instructions, the proposed method achieved a task success rate of 85%, outperforming existing methods in both retrieval performance and overall task success.

**arXiv ID:** 2512.18987
</details>

<details>
<summary><strong>A Survey on Agentic Security: Applications, Threats and Defenses</strong> - Asif Shahriar, Md Nafiu Rahman, Sadif Ahmed, Farig Sadeque, Md Rizwan Parvez - [[pdf]](https://arxiv.org/pdf/2510.06445)</summary>

**Abstract:** In this work we present the first holistic survey of the agentic security landscape, structuring the field around three fundamental pillars: Applications, Threats, and Defenses. We provide a comprehensive taxonomy of over 160 papers, explaining how agents are used in downstream cybersecurity applications, inherent threats to agentic systems, and countermeasures designed to protect them. A detailed cross-cutting analysis shows emerging trends in agent architecture while revealing critical research gaps in model and modality coverage. A complete and continuously updated list of all surveyed papers is publicly available at this https URL.

**arXiv ID:** 2510.06445
</details>

<details>
<summary><strong>LeJOT: An Intelligent Job Cost Orchestration Solution for Databricks Platform</strong> - Lizhi Ma, Yi-Xiang Hu, Yuke Wang, Yifang Zhao, Yihui Ren, Jian-Xiang Liao, Feng Wu, Xiang-Yang Li - [[pdf]](https://arxiv.org/pdf/2512.18266)</summary>

**Abstract:** With the rapid advancements in big data technologies, the Databricks platform has become a cornerstone for enterprises and research institutions, offering high computational efficiency and a robust ecosystem. However, managing the escalating operational costs associated with job execution remains a critical challenge. Existing solutions rely on static configurations or reactive adjustments, which fail to adapt to the dynamic nature of workloads. To address this, we introduce LeJOT, an intelligent job cost orchestration framework that leverages machine learning for execution time prediction and a solver-based optimization model for real-time resource allocation. Unlike conventional scheduling techniques, LeJOT proactively predicts workload demands, dynamically allocates computing resources, and minimizes costs while ensuring performance requirements are met. Experimental results on real-world Databricks workloads demonstrate that LeJOT achieves an average 20% reduction in cloud computing costs within a minute-level scheduling timeframe, outperforming traditional static allocation strategies. Our approach provides a scalable and adaptive solution for cost-efficient job scheduling in Data Lakehouse environments.

**arXiv ID:** 2512.18266
</details>

<details>
<summary><strong>Elevating Intrusion Detection and Security Fortification in Intelligent Networks through Cutting-Edge Machine Learning Paradigms</strong> - Md Minhazul Islam Munna, Md Mahbubur Rahman, Jaroslav Frnda, Muhammad Shahid Anwar, Alpamis Kutlimuratov - [[pdf]](https://arxiv.org/pdf/2512.19037)</summary>

**Abstract:** The proliferation of IoT devices and their reliance on Wi-Fi networks have introduced significant security vulnerabilities, particularly the KRACK and Kr00k attacks, which exploit weaknesses in WPA2 encryption to intercept and manipulate sensitive data. Traditional IDS using classifiers face challenges such as model overfitting, incomplete feature extraction, and high false positive rates, limiting their effectiveness in real-world deployments. To address these challenges, this study proposes a robust multiclass machine learning based intrusion detection framework. The methodology integrates advanced feature selection techniques to identify critical attributes, mitigating redundancy and enhancing detection accuracy. Two distinct ML architectures are implemented: a baseline classifier pipeline and a stacked ensemble model combining noise injection, Principal Component Analysis (PCA), and meta learning to improve generalization and reduce false positives. Evaluated on the AWID3 data set, the proposed ensemble architecture achieves superior performance, with an accuracy of 98%, precision of 98%, recall of 98%, and a false positive rate of just 2%, outperforming existing state-of-the-art methods. This work demonstrates the efficacy of combining preprocessing strategies with ensemble learning to fortify network security against sophisticated Wi-Fi attacks, offering a scalable and reliable solution for IoT environments. Future directions include real-time deployment and adversarial resilience testing to further enhance the model's adaptability.

**arXiv ID:** 2512.19037
</details>

<details>
<summary><strong>Multimodal Classification Network Guided Trajectory Planning for Four-Wheel Independent Steering Autonomous Parking Considering Obstacle Attributes</strong> - Jingjia Teng, Yang Li, Jianqiang Wang, Yingbai Hu, Songyuan Tang, Manjiang Hu - [[pdf]](https://arxiv.org/pdf/2512.18836)</summary>

**Abstract:** Four-wheel Independent Steering (4WIS) vehicles have attracted increasing attention for their superior maneuverability. Human drivers typically choose to cross or drive over the low-profile obstacles (e.g., plastic bags) to efficiently navigate through narrow spaces, while existing planners neglect obstacle attributes, causing inefficiency or path-finding failures. To address this, we propose a trajectory planning framework integrating the 4WIS hybrid A* and Optimal Control Problem (OCP), in which the hybrid A* provides an initial path to enhance the OCP solution. Specifically, a multimodal classification network is introduced to assess scene complexity (hard/easy task) by fusing image and vehicle state data. For hard tasks, guided points are set to decompose complex tasks into local subtasks, improving the search efficiency of 4WIS hybrid A*. The multiple steering modes of 4WIS vehicles (Ackermann, diagonal, and zero-turn) are also incorporated into node expansion and heuristic designs. Moreover, a hierarchical obstacle handling strategy is designed to guide the node expansion considering obstacle attributes, i.e., 'non-traversable', 'crossable', and 'drive-over' obstacles. It allows crossing or driving over obstacles instead of the 'avoid-only' strategy, greatly enhancing success rates of pathfinding. We also design a logical constraint for the 'drive-over' obstacle by limiting its velocity to ensure safety. Furthermore, to address dynamic obstacles with motion uncertainty, we introduce a probabilistic risk field model, constructing risk-aware driving corridors that serve as linear collision constraints in OCP. Experimental results demonstrate the proposed framework's effectiveness in generating safe, efficient, and smooth trajectories for 4WIS vehicles, especially in constrained environments.

**arXiv ID:** 2512.18836
</details>

<details>
<summary><strong>DTCCL: Disengagement-Triggered Contrastive Continual Learning for Autonomous Bus Planners</strong> - Yanding Yang, Weitao Zhou, Jinhai Wang, Xiaomin Guo, Junze Wen, Xiaolong Liu, Lang Ding, Zheng Fu, Jinyu Miao, Kun Jiang, Diange Yang - [[pdf]](https://arxiv.org/pdf/2512.18988)</summary>

**Abstract:** Autonomous buses run on fixed routes but must operate in open, dynamic urban environments. Disengagement events on these routes are often geographically concentrated and typically arise from planner failures in highly interactive regions. Such policy-level failures are difficult to correct using conventional imitation learning, which easily overfits to sparse disengagement data. To address this issue, this paper presents a Disengagement-Triggered Contrastive Continual Learning (DTCCL) framework that enables autonomous buses to improve planning policies through real-world operation. Each disengagement triggers cloud-based data augmentation that generates positive and negative samples by perturbing surrounding agents while preserving route context. Contrastive learning refines policy representations to better distinguish safe and unsafe behaviors, and continual updates are applied in a cloud-edge loop without human supervision. Experiments on urban bus routes demonstrate that DTCCL improves overall planning performance by 48.6 percent compared with direct retraining, validating its effectiveness for scalable, closed-loop policy improvement in autonomous public transport.

**arXiv ID:** 2512.18988
</details>

<details>
<summary><strong>Vidar: Embodied Video Diffusion Model for Generalist Manipulation</strong> - Yao Feng, Hengkai Tan, Xinyi Mao, Chendong Xiang, Guodong Liu, Shuhe Huang, Hang Su, Jun Zhu - [[pdf]](https://arxiv.org/pdf/2507.12898)</summary>

**Abstract:** Scaling general-purpose manipulation to new robot embodiments remains challenging: each platform typically needs large, homogeneous demonstrations, and end-to-end pixel-to-action pipelines may degenerate under background and viewpoint shifts. Based on previous advances in video-based robot control, we present Vidar, consisting of an embodied video diffusion model as the generalizable prior and a masked inverse dynamics model (MIDM) as the adapter. We leverage a video diffusion model pre-trained at Internet scale, and further continuously pre-train it for the embodied domain using 750K multi-view trajectories collected from three real-world robot platforms. For this embodied pre-training, we introduce a unified observation space that jointly encodes robot, camera, task, and scene contexts. The MIDM module learns action-relevant pixel masks without dense labels, grounding the prior into the target embodiment's action space while suppressing distractors. With only 20 minutes of human demonstrations on an unseen robot (1% of typical data), Vidar outperforms state-of-the-art baselines and generalizes to unseen tasks, backgrounds, and camera layouts. Our results suggest a scalable recipe for "one prior, many embodiments": strong, inexpensive video priors together with minimal on-robot alignment.

**arXiv ID:** 2507.12898
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (36 papers)</h2></summary>

<details>
<summary><strong>Unifying Causal Reinforcement Learning: Survey, Taxonomy, Algorithms and Applications</strong> - Cristiano da Costa Cunha, Wei Liu, Tim French, Ajmal Mian - [[pdf]](https://arxiv.org/pdf/2512.18135)</summary>

**Abstract:** Integrating causal inference (CI) with reinforcement learning (RL) has emerged as a powerful paradigm to address critical limitations in classical RL, including low explainability, lack of robustness and generalization failures. Traditional RL techniques, which typically rely on correlation-driven decision-making, struggle when faced with distribution shifts, confounding variables, and dynamic environments. Causal reinforcement learning (CRL), leveraging the foundational principles of causal inference, offers promising solutions to these challenges by explicitly modeling cause-and-effect relationships. In this survey, we systematically review recent advancements at the intersection of causal inference and RL. We categorize existing approaches into causal representation learning, counterfactual policy optimization, offline causal RL, causal transfer learning, and causal explainability. Through this structured analysis, we identify prevailing challenges, highlight empirical successes in practical applications, and discuss open problems. Finally, we provide future research directions, underscoring the potential of CRL for developing robust, generalizable, and interpretable artificial intelligence systems.

**arXiv ID:** 2512.18135
</details>

<details>
<summary><strong>Agent-Based Output Drift Detection for Breast Cancer Response Prediction in a Multisite Clinical Decision Support System</strong> - Xavier Rafael-Palou, Jose Munuera, Ana Jimenez-Pastor, Richard Osuala, Karim Lekadir, Oliver Diaz - [[pdf]](https://arxiv.org/pdf/2512.18450)</summary>

**Abstract:** Modern clinical decision support systems can concurrently serve multiple, independent medical imaging institutions, but their predictive performance may degrade across sites due to variations in patient populations, imaging hardware, and acquisition protocols. Continuous surveillance of predictive model outputs offers a safe and reliable approach for identifying such distributional shifts without ground truth labels. However, most existing methods rely on centralized monitoring of aggregated predictions, overlooking site-specific drift dynamics. We propose an agent-based framework for detecting drift and assessing its severity in multisite clinical AI systems. To evaluate its effectiveness, we simulate a multi-center environment for output-based drift detection, assigning each site a drift monitoring agent that performs batch-wise comparisons of model outputs against a reference distribution. We analyse several multi-center monitoring schemes, that differ in how the reference is obtained (site-specific, global, production-only and adaptive), alongside a centralized baseline. Results on real-world breast cancer imaging data using a pathological complete response prediction model shows that all multi-center schemes outperform centralized monitoring, with F1-score improvements up to 10.3% in drift detection. In the absence of site-specific references, the adaptive scheme performs best, with F1-scores of 74.3% for drift detection and 83.7% for drift severity classification. These findings suggest that adaptive, site-aware agent-based drift monitoring can enhance reliability of multisite clinical decision support systems.

**arXiv ID:** 2512.18450
</details>

<details>
<summary><strong>Multimodal Bayesian Network for Robust Assessment of Casualties in Autonomous Triage</strong> - Szymon Rusiecki, Cecilia G. Morales, Kimberly Elenberg, Leonard Weiss, Artur Dubrawski - [[pdf]](https://arxiv.org/pdf/2512.18908)</summary>

**Abstract:** Mass Casualty Incidents can overwhelm emergency medical systems and resulting delays or errors in the assessment of casualties can lead to preventable deaths. We present a decision support framework that fuses outputs from multiple computer vision models, estimating signs of severe hemorrhage, respiratory distress, physical alertness, or visible trauma, into a Bayesian network constructed entirely from expert-defined rules. Unlike traditional data-driven models, our approach does not require training data, supports inference with incomplete information, and is robust to noisy or uncertain observations. We report performance for two missions involving 11 and 9 casualties, respectively, where our Bayesian network model substantially outperformed vision-only baselines during evaluation of our system in the DARPA Triage Challenge (DTC) field scenarios. The accuracy of physiological assessment improved from 15% to 42% in the first scenario and from 19% to 46% in the second, representing nearly threefold increase in performance. More importantly, overall triage accuracy increased from 14% to 53% in all patients, while the diagnostic coverage of the system expanded from 31% to 95% of the cases requiring assessment. These results demonstrate that expert-knowledge-guided probabilistic reasoning can significantly enhance automated triage systems, offering a promising approach to supporting emergency responders in MCIs. This approach enabled Team Chiron to achieve 4th place out of 11 teams during the 1st physical round of the DTC.

**arXiv ID:** 2512.18908
</details>

<details>
<summary><strong>SafeMed-R1: Adversarial Reinforcement Learning for Generalizable and Robust Medical Reasoning in Vision-Language Models</strong> - A.A. Gde Yogi Pramana, Jason Ray, Anthony Jaya, Michael Wijaya - [[pdf]](https://arxiv.org/pdf/2512.19317)</summary>

**Abstract:** Vision--Language Models (VLMs) show significant promise for Medical Visual Question Answering (VQA), yet their deployment in clinical settings is hindered by severe vulnerability to adversarial attacks. Standard adversarial training, while effective for simpler tasks, often degrades both generalization performance and the quality of generated clinical reasoning. We introduce SafeMed-R1, a hybrid defense framework that ensures robust performance while preserving high-quality, interpretable medical reasoning. SafeMed-R1 employs a two-stage approach: at training time, we integrate Adversarial Training with Group Relative Policy Optimization (AT-GRPO) to explicitly robustify the reasoning process against worst-case perturbations; at inference time, we augment the model with Randomized Smoothing to provide certified $L_2$-norm robustness guarantees. We evaluate SafeMed-R1 on the OmniMedVQA benchmark across eight medical imaging modalities comprising over 88,000 samples. Our experiments reveal that standard fine-tuned VLMs, despite achieving 95\% accuracy on clean inputs, collapse to approximately 25\% under PGD attacks. In contrast, SafeMed-R1 maintains 84.45\% accuracy under the same adversarial conditions, representing a 59 percentage point improvement in robustness. Furthermore, we demonstrate that models trained with explicit chain-of-thought reasoning exhibit superior adversarial robustness compared to instruction-only variants, suggesting a synergy between interpretability and security in medical AI systems.

**arXiv ID:** 2512.19317
</details>

<details>
<summary><strong>Graph-O1 : Monte Carlo Tree Search with Reinforcement Learning for Text-Attributed Graph Reasoning</strong> - Lihui Liu - [[pdf]](https://arxiv.org/pdf/2512.17912)</summary>

**Abstract:** ChatGPT said: Text-attributed graphs, where nodes and edges contain rich textual information, are widely used across diverse domains. A central challenge in this setting is question answering, which requires jointly leveraging unstructured text and the structured relational signals within the graph. Although Large Language Models (LLMs) have made significant advances in natural language understanding, their direct use for reasoning over text-attributed graphs remains limited. Retrieval-augmented generation methods that operate purely on text often treat passages as isolated units, ignoring the interconnected structure of the graph. Conversely, graph-based RAG methods that serialize large subgraphs into long textual sequences quickly become infeasible due to LLM context-length constraints, resulting in fragmented reasoning and degraded accuracy. To overcome these limitations, we introduce Graph-O1, an agentic GraphRAG framework that enables LLMs to conduct stepwise, interactive reasoning over graphs. Our approach integrates Monte Carlo Tree Search (MCTS) with end-to-end reinforcement learning, allowing the model to selectively explore and retrieve only the most informative subgraph components. The reasoning procedure is framed as a multi-turn interaction between the agent and the graph environment, and the agent is trained through a unified reward mechanism. Extensive experiments across multiple LLM backbones demonstrate that Graph-O1 consistently surpasses state-of-the-art baselines, producing answers that are more accurate, reliable, and interpretable.

**arXiv ID:** 2512.17912
</details>

<details>
<summary><strong>Reinforcement Learning for Monetary Policy Under Macroeconomic Uncertainty: Analyzing Tabular and Function Approximation Methods</strong> - Sheryl Chen, Tony Wang, Kyle Feinstein - [[pdf]](https://arxiv.org/pdf/2512.17929)</summary>

**Abstract:** We study how a central bank should dynamically set short-term nominal interest rates to stabilize inflation and unemployment when macroeconomic relationships are uncertain and time-varying. We model monetary policy as a sequential decision-making problem where the central bank observes macroeconomic conditions quarterly and chooses interest rate adjustments. Using publically accessible historical Federal Reserve Economic Data (FRED), we construct a linear-Gaussian transition model and implement a discrete-action Markov Decision Process with a quadratic loss reward function. We chose to compare nine different reinforcement learning style approaches against Taylor Rule and naive baselines, including tabular Q-learning variants, SARSA, Actor-Critic, Deep Q-Networks, Bayesian Q-learning with uncertainty quantification, and POMDP formulations with partial observability. Surprisingly, standard tabular Q-learning achieved the best performance (-615.13 +- 309.58 mean return), outperforming both enhanced RL methods and traditional policy rules. Our results suggest that while sophisticated RL techniques show promise for monetary policy applications, simpler approaches may be more robust in this domain, highlighting important challenges in applying modern RL to macroeconomic policy.

**arXiv ID:** 2512.17929
</details>

<details>
<summary><strong>Adaptive Agents in Spatial Double-Auction Markets: Modeling the Emergence of Industrial Symbiosis</strong> - Matthieu Mastio, Paul Saves, Benoit Gaudou, Nicolas Verstaevel - [[pdf]](https://arxiv.org/pdf/2512.17979)</summary>

**Abstract:** Industrial symbiosis fosters circularity by enabling firms to repurpose residual resources, yet its emergence is constrained by socio-spatial frictions that shape costs, matching opportunities, and market efficiency. Existing models often overlook the interaction between spatial structure, market design, and adaptive firm behavior, limiting our understanding of where and how symbiosis arises. We develop an agent-based model where heterogeneous firms trade byproducts through a spatially embedded double-auction market, with prices and quantities emerging endogenously from local interactions. Leveraging reinforcement learning, firms adapt their bidding strategies to maximize profit while accounting for transport costs, disposal penalties, and resource scarcity. Simulation experiments reveal the economic and spatial conditions under which decentralized exchanges converge toward stable and efficient outcomes. Counterfactual regret analysis shows that sellers' strategies approach a near Nash equilibrium, while sensitivity analysis highlights how spatial structures and market parameters jointly govern circularity. Our model provides a basis for exploring policy interventions that seek to align firm incentives with sustainability goals, and more broadly demonstrates how decentralized coordination can emerge from adaptive agents in spatially constrained markets.

**arXiv ID:** 2512.17979
</details>

<details>
<summary><strong>From Prompt to Product: A Human-Centered Benchmark of Agentic App Generation Systems</strong> - Marcos Ortiz, Justin Hill, Collin Overbay, Ingrida Semenec, Frederic Sauve-Hoover, Jim Schwoebel, Joel Shor - [[pdf]](https://arxiv.org/pdf/2512.18080)</summary>

**Abstract:** Agentic AI systems capable of generating full-stack web applications from natural language prompts ("prompt- to-app") represent a significant shift in software development. However, evaluating these systems remains challenging, as visual polish, functional correctness, and user trust are often misaligned. As a result, it is unclear how existing prompt-to-app tools compare under realistic, human-centered evaluation criteria. In this paper, we introduce a human-centered benchmark for evaluating prompt-to-app systems and conduct a large-scale comparative study of three widely used platforms: Replit, Bolt, and Firebase Studio. Using a diverse set of 96 prompts spanning common web application tasks, we generate 288 unique application artifacts. We evaluate these systems through a large-scale human-rater study involving 205 participants and 1,071 quality-filtered pairwise comparisons, assessing task-based ease of use, visual appeal, perceived completeness, and user trust. Our results show that these systems are not interchangeable: Firebase Studio consistently outperforms competing platforms across all human-evaluated dimensions, achieving the highest win rates for ease of use, trust, visual appeal, and visual appropriateness. Bolt performs competitively on visual appeal but trails Firebase on usability and trust, while Replit underperforms relative to both across most metrics. These findings highlight a persistent gap between visual polish and functional reliability in prompt-to-app systems and demonstrate the necessity of interactive, task-based evaluation. We release our benchmark framework, prompt set, and generated artifacts to support reproducible evaluation and future research in agentic application generation.

**arXiv ID:** 2512.18080
</details>

<details>
<summary><strong>Trustworthy and Explainable Deep Reinforcement Learning for Safe and Energy-Efficient Process Control: A Use Case in Industrial Compressed Air Systems</strong> - Vincent Bezold, Patrick Wagner, Jakob Hofmann, Marco Huber, Alexander Sauer - [[pdf]](https://arxiv.org/pdf/2512.18317)</summary>

**Abstract:** This paper presents a trustworthy reinforcement learning approach for the control of industrial compressed air systems. We develop a framework that enables safe and energy-efficient operation under realistic boundary conditions and introduce a multi-level explainability pipeline combining input perturbation tests, gradient-based sensitivity analysis, and SHAP (SHapley Additive exPlanations) feature attribution. An empirical evaluation across multiple compressor configurations shows that the learned policy is physically plausible, anticipates future demand, and consistently respects system boundaries. Compared to the installed industrial controller, the proposed approach reduces unnecessary overpressure and achieves energy savings of approximately 4\,\% without relying on explicit physics models. The results further indicate that system pressure and forecast information dominate policy decisions, while compressor-level inputs play a secondary role. Overall, the combination of efficiency gains, predictive behavior, and transparent validation supports the trustworthy deployment of reinforcement learning in industrial energy systems.

**arXiv ID:** 2512.18317
</details>

<details>
<summary><strong>Reinforcement Learning Position Control of a Quadrotor Using Soft Actor-Critic (SAC)</strong> - Youssef Mahran, Zeyad Gamal, Ayman El-Badawy - [[pdf]](https://arxiv.org/pdf/2512.18333)</summary>

**Abstract:** This paper proposes a new Reinforcement Learning (RL) based control architecture for quadrotors. With the literature focusing on controlling the four rotors' RPMs directly, this paper aims to control the quadrotor's thrust vector. The RL agent computes the percentage of overall thrust along the quadrotor's z-axis along with the desired Roll ($\phi$) and Pitch ($\theta$) angles. The agent then sends the calculated control signals along with the current quadrotor's Yaw angle ($\psi$) to an attitude PID controller. The PID controller then maps the control signals to motor RPMs. The Soft Actor-Critic algorithm, a model-free off-policy stochastic RL algorithm, was used to train the RL agents. Training results show the faster training time of the proposed thrust vector controller in comparison to the conventional RPM controllers. Simulation results show smoother and more accurate path-following for the proposed thrust vector controller.

**arXiv ID:** 2512.18333
</details>

<details>
<summary><strong>Dynamic Entropy Tuning in Reinforcement Learning Low-Level Quadcopter Control: Stochasticity vs Determinism</strong> - Youssef Mahran, Zeyad Gamal, Ayman El-Badawy - [[pdf]](https://arxiv.org/pdf/2512.18336)</summary>

**Abstract:** This paper explores the impact of dynamic entropy tuning in Reinforcement Learning (RL) algorithms that train a stochastic policy. Its performance is compared against algorithms that train a deterministic one. Stochastic policies optimize a probability distribution over actions to maximize rewards, while deterministic policies select a single deterministic action per state. The effect of training a stochastic policy with both static entropy and dynamic entropy and then executing deterministic actions to control the quadcopter is explored. It is then compared against training a deterministic policy and executing deterministic actions. For the purpose of this research, the Soft Actor-Critic (SAC) algorithm was chosen for the stochastic algorithm while the Twin Delayed Deep Deterministic Policy Gradient (TD3) was chosen for the deterministic algorithm. The training and simulation results show the positive effect the dynamic entropy tuning has on controlling the quadcopter by preventing catastrophic forgetting and improving exploration efficiency.

**arXiv ID:** 2512.18336
</details>

<details>
<summary><strong>Federated Learning Based Decentralized Adaptive Intelligent Transmission Protocol for Privacy Preserving 6G Networks</strong> - Ansar Ahmed - [[pdf]](https://arxiv.org/pdf/2512.18432)</summary>

**Abstract:** The move to 6th Generation (6G) wireless networks creates new issues with privacy, scalability, and adaptability. The data-intensive nature of 6G is not handled well by older, centralized network models. A shift toward more secure and decentralized systems is therefore required. A new framework called the Federated Learning-based Decentralized Adaptive Intelligent Transmission Protocol (AITP) is proposed to meet these challenges. The AITP uses the distributed learning of Federated Learning (FL) within a decentralized system. Transmission parameters can be adjusted intelligently in real time. User privacy is maintained by keeping raw data on local edge devices. The protocol's performance was evaluated with mathematical modeling and detailed simulations. It was shown to be superior to traditional non-adaptive and centralized AI methods across several key metrics. These included latency, network throughput, energy efficiency, and robustness. The AITP is presented as a foundational technology for future 6G networks that supports a user-centric, privacy-first design. This study is a step forward for privacy-preserving research in 6G.

**arXiv ID:** 2512.18432
</details>

<details>
<summary><strong>An Agentic AI Framework for Training General Practitioner Student Skills</strong> - Victor De Marez, Jens Van Nooten, Luna De Bruyne, Walter Daelemans - [[pdf]](https://arxiv.org/pdf/2512.18440)</summary>

**Abstract:** Advancements in large language models offer strong potential for enhancing virtual simulated patients (VSPs) in medical education by providing scalable alternatives to resource-intensive traditional methods. However, current VSPs often struggle with medical accuracy, consistent roleplaying, scenario generation for VSP use, and educationally structured feedback. We introduce an agentic framework for training general practitioner student skills that unifies (i) configurable, evidence-based vignette generation, (ii) controlled persona-driven patient dialogue with optional retrieval grounding, and (iii) standards-based assessment and feedback for both communication and clinical reasoning. We instantiate the framework in an interactive spoken consultation setting and evaluate it with medical students ($\mathbf{N{=}14}$). Participants reported realistic and vignette-faithful dialogue, appropriate difficulty calibration, a stable personality signal, and highly useful example-rich feedback, alongside excellent overall usability. These results support agentic separation of scenario control, interaction control, and standards-based assessment as a practical pattern for building dependable and pedagogically valuable VSP training tools.

**arXiv ID:** 2512.18440
</details>

<details>
<summary><strong>CauTraj: A Causal-Knowledge-Guided Framework for Lane-Changing Trajectory Planning of Autonomous Vehicles</strong> - Cailin Lei, Haiyang Wu, Yuxiong Ji, Xiaoyu Cai, Yuchuan Du - [[pdf]](https://arxiv.org/pdf/2512.18703)</summary>

**Abstract:** Enhancing the performance of trajectory planners for lane - changing vehicles is one of the key challenges in autonomous driving within human - machine mixed traffic. Most existing studies have not incorporated human drivers' prior knowledge when designing trajectory planning models. To address this issue, this study proposes a novel trajectory planning framework that integrates causal prior knowledge into the control process. Both longitudinal and lateral microscopic behaviors of vehicles are modeled to quantify interaction risk, and a staged causal graph is constructed to capture causal dependencies in lane-changing scenarios. Causal effects between the lane-changing vehicle and surrounding vehicles are then estimated using causal inference, including average causal effects (ATE) and conditional average treatment effects (CATE). These causal priors are embedded into a model predictive control (MPC) framework to enhance trajectory planning. The proposed approach is validated on naturalistic vehicle trajectory datasets. Experimental results show that: (1) causal inference provides interpretable and stable quantification of vehicle interactions; (2) individual causal effects reveal driver heterogeneity; and (3) compared with the baseline MPC, the proposed method achieves a closer alignment with human driving behaviors, reducing maximum trajectory deviation from 1.2 m to 0.2 m, lateral velocity fluctuation by 60%, and yaw angle variability by 50%. These findings provide methodological support for human-like trajectory planning and practical value for improving safety, stability, and realism in autonomous vehicle testing and traffic simulation platforms.

**arXiv ID:** 2512.18703
</details>

<details>
<summary><strong>Structural Reinforcement Learning for Heterogeneous Agent Macroeconomics</strong> - Yucheng Yang, Chiyuan Wang, Andreas Schaab, Benjamin Moll - [[pdf]](https://arxiv.org/pdf/2512.18892)</summary>

**Abstract:** We present a new approach to formulating and solving heterogeneous agent models with aggregate risk. We replace the cross-sectional distribution with low-dimensional prices as state variables and let agents learn equilibrium price dynamics directly from simulated paths. To do so, we introduce a structural reinforcement learning (SRL) method which treats prices via simulation while exploiting agents' structural knowledge of their own individual dynamics. Our SRL method yields a general and highly efficient global solution method for heterogeneous agent models that sidesteps the Master equation and handles problems traditional methods struggle with, in particular nontrivial market-clearing conditions. We illustrate the approach in the Krusell-Smith model, the Huggett model with aggregate shocks, and a HANK model with a forward-looking Phillips curve, all of which we solve globally within minutes.

**arXiv ID:** 2512.18892
</details>

<details>
<summary><strong>LacaDM: A Latent Causal Diffusion Model for Multiobjective Reinforcement Learning</strong> - Xueming Yan, Bo Yin, Yaochu Jin - [[pdf]](https://arxiv.org/pdf/2512.19516)</summary>

**Abstract:** Multiobjective reinforcement learning (MORL) poses significant challenges due to the inherent conflicts between objectives and the difficulty of adapting to dynamic environments. Traditional methods often struggle to generalize effectively, particularly in large and complex state-action spaces. To address these limitations, we introduce the Latent Causal Diffusion Model (LacaDM), a novel approach designed to enhance the adaptability of MORL in discrete and continuous environments. Unlike existing methods that primarily address conflicts between objectives, LacaDM learns latent temporal causal relationships between environmental states and policies, enabling efficient knowledge transfer across diverse MORL scenarios. By embedding these causal structures within a diffusion model-based framework, LacaDM achieves a balance between conflicting objectives while maintaining strong generalization capabilities in previously unseen environments. Empirical evaluations on various tasks from the MOGymnasium framework demonstrate that LacaDM consistently outperforms the state-of-art baselines in terms of hypervolume, sparsity, and expected utility maximization, showcasing its effectiveness in complex multiobjective tasks.

**arXiv ID:** 2512.19516
</details>

<details>
<summary><strong>Does It Tie Out? Towards Autonomous Legal Agents in Venture Capital</strong> - Pierre Colombo, Malik Boudiaf, Allyn Sweet, Michael Desa, Hongxi Wang, Kevin Candra, Syméon del Marmol - [[pdf]](https://arxiv.org/pdf/2512.18658)</summary>

**Abstract:** Before closing venture capital financing rounds, lawyers conduct diligence that includes tying out the capitalization table: verifying that every security (for example, shares, options, warrants) and issuance term (for example, vesting schedules, acceleration triggers, transfer restrictions) is supported by large sets of underlying legal documentation. While LLMs continue to improve on legal benchmarks, specialized legal workflows, such as capitalization tie-out, remain out of reach even for strong agentic systems. The task requires multi-document reasoning, strict evidence traceability, and deterministic outputs that current approaches fail to reliably deliver. We characterize capitalization tie-out as an instance of a real-world benchmark for legal AI, analyze and compare the performance of existing agentic systems, and propose a world model architecture toward tie-out automation-and more broadly as a foundation for applied legal intelligence.

**arXiv ID:** 2512.18658
</details>

<details>
<summary><strong>MobileWorld: Benchmarking Autonomous Mobile Agents in Agent-User Interactive, and MCP-Augmented Environments</strong> - Quyu Kong, Xu Zhang, Zhenyu Yang, Nolan Gao, Chen Liu, Panrong Tong, Chenglin Cai, Hanzhang Zhou, Jianan Zhang, Liangyu Chen, Zhidan Liu, Steven Hoi, Yue Wang - [[pdf]](https://arxiv.org/pdf/2512.19432)</summary>

**Abstract:** Among existing online mobile-use benchmarks, AndroidWorld has emerged as the dominant benchmark due to its reproducible environment and deterministic evaluation; however, recent agents achieving over 90% success rates indicate its saturation and motivate the need for a more challenging benchmark. In addition, its environment lacks key application categories, such as e-commerce and enterprise communication, and does not reflect realistic mobile-use scenarios characterized by vague user instructions and hybrid tool usage. To bridge this gap, we introduce MobileWorld, a substantially more challenging benchmark designed to better reflect real-world mobile usage, comprising 201 tasks across 20 applications, while maintaining the same level of reproducible evaluation as AndroidWorld. The difficulty of MobileWorld is twofold. First, it emphasizes long-horizon tasks with cross-application interactions: MobileWorld requires nearly twice as many task-completion steps on average (27.8 vs. 14.3) and includes far more multi-application tasks (62.2% vs. 9.5%) compared to AndroidWorld. Second, MobileWorld extends beyond standard GUI manipulation by introducing novel task categories, including agent-user interaction and MCP-augmented tasks. To ensure robust evaluation, we provide snapshot-based container environment and precise functional verifications, including backend database inspection and task callback APIs. We further develop a planner-executor agentic framework with extended action spaces to support user interactions and MCP calls. Our results reveal a sharp performance drop compared to AndroidWorld, with the best agentic framework and end-to-end model achieving 51.7% and 20.9% success rates, respectively. Our analysis shows that current models struggle significantly with user interaction and MCP calls, offering a strategic roadmap toward more robust, next-generation mobile intelligence.

**arXiv ID:** 2512.19432
</details>

<details>
<summary><strong>Survey and Experiments on Mental Disorder Detection via Social Media: From Large Language Models and RAG to Agents</strong> - Zhuohan Ge, Darian Li, Yubo Wang, Nicole Hu, Xinyi Zhu, Haoyang Li, Xin Zhang, Mingtao Zhang, Shihao Qi, Yuming Xu, Han Shi, Chen Jason Zhang, Qing Li - [[pdf]](https://arxiv.org/pdf/2504.02800)</summary>

**Abstract:** Mental disorders represent a critical global health challenge, and social media is increasingly viewed as a vital resource for real-time digital phenotyping and intervention. To leverage this data, large language models (LLMs) have been introduced, offering stronger semantic understanding and reasoning than traditional deep learning, thereby enhancing the explainability of detection results. Despite the growing prominence of LLMs in this field, there is a scarcity of scholarly works that systematically synthesize how advanced enhancement techniques, specifically Retrieval-Augmented Generation (RAG) and Agentic systems, can be utilized to address these reliability and reasoning limitations. Here, we systematically survey the evolving landscape of LLM-based methods for social media mental disorder analysis, spanning standard pre-trained language models, RAG to mitigate hallucinations and contextual gaps, and agentic systems for autonomous reasoning and multi-step intervention. We organize existing work by technical paradigm and clinical target, extending beyond common internalizing disorders to include psychotic disorders and externalizing behaviors. Additionally, the paper comprehensively evaluates the performance of LLMs, including the impact of RAG, across various tasks. This work establishes a unified benchmark for the field, paving the way for the development of trustworthy, autonomous AI systems that can deliver precise and explainable mental health support.

**arXiv ID:** 2504.02800
</details>

<details>
<summary><strong>SPELL: Self-Play Reinforcement Learning for evolving Long-Context Language Models</strong> - Ziyi Yang, Weizhou Shen, Chenliang Li, Ruijun Chen, Fanqi Wan, Ming Yan, Xiaojun Quan, Fei Huang - [[pdf]](https://arxiv.org/pdf/2509.23863)</summary>

**Abstract:** Progress in long-context reasoning for large language models (LLMs) has lagged behind other recent advances. This gap arises not only from the intrinsic difficulty of processing long texts, but also from the scarcity of reliable human annotations and programmatically verifiable reward signals. In this paper, we propose SPELL, a multi-role self-play reinforcement learning framework that enables scalable, label-free optimization for long-context reasoning. SPELL integrates three cyclical roles-questioner, responder, and verifier-within a single model to enable continual self-improvement. The questioner generates questions from raw documents paired with reference answers; the responder learns to solve these questions based on the documents; and the verifier evaluates semantic equivalence between the responder's output and the questioner's reference answer, producing reward signals to guide continual training. To stabilize training, we introduce an automated curriculum that gradually increases document length and a reward function that adapts question difficulty to the model's evolving capabilities. Extensive experiments on six long-context benchmarks show that SPELL consistently improves performance across diverse LLMs and outperforms equally sized models fine-tuned on large-scale annotated data. Notably, SPELL achieves an average 7.6-point gain in pass@8 on the strong reasoning model Qwen3-30B-A3B-Thinking, raising its performance ceiling and showing promise for scaling to even more capable models.

**arXiv ID:** 2509.23863
</details>

<details>
<summary><strong>Adaptation of Agentic AI</strong> - Pengcheng Jiang, Jiacheng Lin, Zhiyi Shi, Zifeng Wang, Luxi He, Yichen Wu, Ming Zhong, Peiyang Song, Qizheng Zhang, Heng Wang, Xueqiang Xu, Hanwen Xu, Pengrui Han, Dylan Zhang, Jiashuo Sun, Chaoqi Yang, Kun Qian, Tian Wang, Changran Hu, Manling Li, Quanzheng Li, Hao Peng, Sheng Wang, Jingbo Shang, Chao Zhang, Jiaxuan You, Liyuan Liu, Pan Lu, Yu Zhang, Heng Ji, Yejin Choi, Dawn Song, Jimeng Sun, Jiawei Han - [[pdf]](https://arxiv.org/pdf/2512.16301)</summary>

**Abstract:** Cutting-edge agentic AI systems are built on foundation models that can be adapted to plan, reason, and interact with external tools to perform increasingly complex and specialized tasks. As these systems grow in capability and scope, adaptation becomes a central mechanism for improving performance, reliability, and generalization. In this paper, we unify the rapidly expanding research landscape into a systematic framework that spans both agent adaptations and tool adaptations. We further decompose these into tool-execution-signaled and agent-output-signaled forms of agent adaptation, as well as agent-agnostic and agent-supervised forms of tool adaptation. We demonstrate that this framework helps clarify the design space of adaptation strategies in agentic AI, makes their trade-offs explicit, and provides practical guidance for selecting or switching among strategies during system design. We then review the representative approaches in each category, analyze their strengths and limitations, and highlight key open challenges and future opportunities. Overall, this paper aims to offer a conceptual foundation and practical roadmap for researchers and practitioners seeking to build more capable, efficient, and reliable agentic AI systems.

**arXiv ID:** 2512.16301
</details>

<details>
<summary><strong>Demonstration-Guided Continual Reinforcement Learning in Dynamic Environments</strong> - Xue Yang, Michael Schukat, Junlin Lu, Patrick Mannion, Karl Mason, Enda Howley - [[pdf]](https://arxiv.org/pdf/2512.18670)</summary>

**Abstract:** Reinforcement learning (RL) excels in various applications but struggles in dynamic environments where the underlying Markov decision process evolves. Continual reinforcement learning (CRL) enables RL agents to continually learn and adapt to new tasks, but balancing stability (preserving prior knowledge) and plasticity (acquiring new knowledge) remains challenging. Existing methods primarily address the stability-plasticity dilemma through mechanisms where past knowledge influences optimization but rarely affects the agent's behavior directly, which may hinder effective knowledge reuse and efficient learning. In contrast, we propose demonstration-guided continual reinforcement learning (DGCRL), which stores prior knowledge in an external, self-evolving demonstration repository that directly guides RL exploration and adaptation. For each task, the agent dynamically selects the most relevant demonstration and follows a curriculum-based strategy to accelerate learning, gradually shifting from demonstration-guided exploration to fully self-exploration. Extensive experiments on 2D navigation and MuJoCo locomotion tasks demonstrate its superior average performance, enhanced knowledge transfer, mitigation of forgetting, and training efficiency. The additional sensitivity analysis and ablation study further validate its effectiveness.

**arXiv ID:** 2512.18670
</details>

<details>
<summary><strong>Gaussian-Mixture-Model Q-Functions for Policy Iteration in Reinforcement Learning</strong> - Minh Vu, Konstantinos Slavakis - [[pdf]](https://arxiv.org/pdf/2512.18763)</summary>

**Abstract:** Unlike their conventional use as estimators of probability density functions in reinforcement learning (RL), this paper introduces a novel function-approximation role for Gaussian mixture models (GMMs) as direct surrogates for Q-function losses. These parametric models, termed GMM-QFs, possess substantial representational capacity, as they are shown to be universal approximators over a broad class of functions. They are further embedded within Bellman residuals, where their learnable parameters -- a fixed number of mixing weights, together with Gaussian mean vectors and covariance matrices -- are inferred from data via optimization on a Riemannian manifold. This geometric perspective on the parameter space naturally incorporates Riemannian optimization into the policy-evaluation step of standard policy-iteration frameworks. Rigorous theoretical results are established, and supporting numerical tests show that, even without access to experience data, GMM-QFs deliver competitive performance and, in some cases, outperform state-of-the-art approaches across a range of benchmark RL tasks, all while maintaining a significantly smaller computational footprint than deep-learning methods that rely on experience data.

**arXiv ID:** 2512.18763
</details>

<details>
<summary><strong>Scaling Online Distributionally Robust Reinforcement Learning: Sample-Efficient Guarantees with General Function Approximation</strong> - Debamita Ghosh, George K. Atia, Yue Wang - [[pdf]](https://arxiv.org/pdf/2512.18957)</summary>

**Abstract:** The deployment of reinforcement learning (RL) agents in real-world applications is often hindered by performance degradation caused by mismatches between training and deployment environments. Distributionally robust RL (DR-RL) addresses this issue by optimizing worst-case performance over an uncertainty set of transition dynamics. However, existing work typically relies on substantial prior knowledge-such as access to a generative model or a large offline dataset-and largely focuses on tabular methods that do not scale to complex domains. We overcome these limitations by proposing an online DR-RL algorithm with general function approximation that learns an optimal robust policy purely through interaction with the environment, without requiring prior models or offline data, enabling deployment in high-dimensional tasks. We further provide a theoretical analysis establishing a near-optimal sublinear regret bound under a total variation uncertainty set, demonstrating the sample efficiency and effectiveness of our method.

**arXiv ID:** 2512.18957
</details>

<details>
<summary><strong>Trajectory-Aware Eligibility Traces for Off-Policy Reinforcement Learning</strong> - Brett Daley, Martha White, Christopher Amato, Marlos C. Machado - [[pdf]](https://arxiv.org/pdf/2301.11321)</summary>

**Abstract:** Off-policy learning from multistep returns is crucial for sample-efficient reinforcement learning, but counteracting off-policy bias without exacerbating variance is challenging. Classically, off-policy bias is corrected in a per-decision manner: past temporal-difference errors are re-weighted by the instantaneous Importance Sampling (IS) ratio after each action via eligibility traces. Many off-policy algorithms rely on this mechanism, along with differing protocols for cutting the IS ratios to combat the variance of the IS estimator. Unfortunately, once a trace has been fully cut, the effect cannot be reversed. This has led to the development of credit-assignment strategies that account for multiple past experiences at a time. These trajectory-aware methods have not been extensively analyzed, and their theoretical justification remains uncertain. In this paper, we propose a multistep operator that can express both per-decision and trajectory-aware methods. We prove convergence conditions for our operator in the tabular setting, establishing the first guarantees for several existing methods as well as many new ones. Finally, we introduce Recency-Bounded Importance Sampling (RBIS), which leverages trajectory awareness to perform robustly across $\lambda$-values in an off-policy control task.

**arXiv ID:** 2301.11321
</details>

<details>
<summary><strong>Averaging $n$-step Returns Reduces Variance in Reinforcement Learning</strong> - Brett Daley, Martha White, Marlos C. Machado - [[pdf]](https://arxiv.org/pdf/2402.03903)</summary>

**Abstract:** Multistep returns, such as $n$-step returns and $\lambda$-returns, are commonly used to improve the sample efficiency of reinforcement learning (RL) methods. The variance of the multistep returns becomes the limiting factor in their length; looking too far into the future increases variance and reverses the benefits of multistep learning. In our work, we demonstrate the ability of compound returns -- weighted averages of $n$-step returns -- to reduce variance. We prove for the first time that any compound return with the same contraction modulus as a given $n$-step return has strictly lower variance. We additionally prove that this variance-reduction property improves the finite-sample complexity of temporal-difference learning under linear function approximation. Because general compound returns can be expensive to implement, we introduce two-bootstrap returns which reduce variance while remaining efficient, even when using minibatched experience replay. We conduct experiments showing that compound returns often increase the sample efficiency of $n$-step deep RL agents like DQN and PPO.

**arXiv ID:** 2402.03903
</details>

<details>
<summary><strong>A Reinforcement Learning Environment for Automatic Code Optimization in the MLIR Compiler</strong> - Mohammed Tirichine, Nassim Ameur, Nazim Bendib, Iheb Nassim Aouadj, Bouchama Djad, Rafik Bouloudene, Riyadh Baghdadi - [[pdf]](https://arxiv.org/pdf/2409.11068)</summary>

**Abstract:** Code optimization is a crucial task that aims to enhance code performance. However, this process is often tedious and complex, highlighting the necessity for automatic code optimization techniques. Reinforcement Learning (RL) has emerged as a promising approach for tackling such complex optimization problems. In this project, we introduce MLIR RL, an RL environment for the MLIR compiler, dedicated to facilitating MLIR compiler research and enabling automatic code optimization. We propose a multi-discrete formulation of the action space where the action space is the Cartesian product of simpler action subspaces. We also propose a new method, called level pointers, to reduce the size of the action space related to the loop interchange transformation. This enables more efficient and effective learning of the policy. To demonstrate the effectiveness of MLIR RL, we train an RL agent to optimize MLIR Linalg code, targeting CPU. The code is generated from two domain-specific frameworks: deep-learning models generated from PyTorch, and LQCD (Lattice Quantum Chromodynamics) code generated from an LQCD compiler. The result of this work is a research environment that allows the community to experiment with novel ideas in RL-driven loop-nest optimization.

**arXiv ID:** 2409.11068
</details>

<details>
<summary><strong>Towards Autonomous Navigation in Endovascular Interventions</strong> - Tudor Jianu, Anh Nguyen, Sebastiano Fichera, Pierre Berthet-Rayne - [[pdf]](https://arxiv.org/pdf/2512.18081)</summary>

**Abstract:** Cardiovascular diseases remain the leading cause of global mortality, with minimally invasive treatment options offered through endovascular interventions. However, the precision and adaptability of current robotic systems for endovascular navigation are limited by heuristic control, low autonomy, and the absence of haptic feedback. This thesis presents an integrated AI-driven framework for autonomous guidewire navigation in complex vascular environments, addressing key challenges in data availability, simulation fidelity, and navigational accuracy.
A high-fidelity, real-time simulation platform, CathSim, is introduced for reinforcement learning based catheter navigation, featuring anatomically accurate vascular models and contact dynamics. Building on CathSim, the Expert Navigation Network is developed, a policy that fuses visual, kinematic, and force feedback for autonomous tool control. To mitigate data scarcity, the open-source, bi-planar fluoroscopic dataset Guide3D is proposed, comprising more than 8,700 annotated images for 3D guidewire reconstruction. Finally, SplineFormer, a transformer-based model, is introduced to directly predict guidewire geometry as continuous B-spline parameters, enabling interpretable, real-time navigation.
The findings show that combining high-fidelity simulation, multimodal sensory fusion, and geometric modelling substantially improves autonomous endovascular navigation and supports safer, more precise minimally invasive procedures.

**arXiv ID:** 2512.18081
</details>

<details>
<summary><strong>Offline Reinforcement Learning for End-to-End Autonomous Driving</strong> - Chihiro Noguchi, Takaki Yamamoto - [[pdf]](https://arxiv.org/pdf/2512.18662)</summary>

**Abstract:** End-to-end (E2E) autonomous driving models that take only camera images as input and directly predict a future trajectory are appealing for their computational efficiency and potential for improved generalization via unified optimization; however, persistent failure modes remain due to reliance on imitation learning (IL). While online reinforcement learning (RL) could mitigate IL-induced issues, the computational burden of neural rendering-based simulation and large E2E networks renders iterative reward and hyperparameter tuning costly. We introduce a camera-only E2E offline RL framework that performs no additional exploration and trains solely on a fixed simulator dataset. Offline RL offers strong data efficiency and rapid experimental iteration, yet is susceptible to instability from overestimation on out-of-distribution (OOD) actions. To address this, we construct pseudo ground-truth trajectories from expert driving logs and use them as a behavior regularization signal, suppressing imitation of unsafe or suboptimal behavior while stabilizing value learning. Training and closed-loop evaluation are conducted in a neural rendering environment learned from the public nuScenes dataset. Empirically, the proposed method achieves substantial improvements in collision rate and route completion compared with IL baselines. Our code will be available at [URL].

**arXiv ID:** 2512.18662
</details>

<details>
<summary><strong>InDRiVE: Reward-Free World-Model Pretraining for Autonomous Driving via Latent Disagreement</strong> - Feeza Khan Khanzada, Jaerock Kwon - [[pdf]](https://arxiv.org/pdf/2512.18850)</summary>

**Abstract:** Model-based reinforcement learning (MBRL) can reduce interaction cost for autonomous driving by learning a predictive world model, but it typically still depends on task-specific rewards that are difficult to design and often brittle under distribution shift. This paper presents InDRiVE, a DreamerV3-style MBRL agent that performs reward-free pretraining in CARLA using only intrinsic motivation derived from latent ensemble disagreement. Disagreement acts as a proxy for epistemic uncertainty and drives the agent toward under-explored driving situations, while an imagination-based actor-critic learns a planner-free exploration policy directly from the learned world model. After intrinsic pretraining, we evaluate zero-shot transfer by freezing all parameters and deploying the pretrained exploration policy in unseen towns and routes. We then study few-shot adaptation by training a task policy with limited extrinsic feedback for downstream objectives (lane following and collision avoidance). Experiments in CARLA across towns, routes, and traffic densities show that disagreement-based pretraining yields stronger zero-shot robustness and robust few-shot collision avoidance under town shift and matched interaction budgets, supporting the use of intrinsic disagreement as a practical reward-free pretraining signal for reusable driving world models.

**arXiv ID:** 2512.18850
</details>

<details>
<summary><strong>CoDrone: Autonomous Drone Navigation Assisted by Edge and Cloud Foundation Models</strong> - Pengyu Chen, Tao Ouyang, Ke Luo, Weijie Hong, Xu Chen - [[pdf]](https://arxiv.org/pdf/2512.19083)</summary>

**Abstract:** Autonomous navigation for Unmanned Aerial Vehicles faces key challenges from limited onboard computational resources, which restrict deployed deep neural networks to shallow architectures incapable of handling complex environments. Offloading tasks to remote edge servers introduces high latency, creating an inherent trade-off in system design. To address these limitations, we propose CoDrone - the first cloud-edge-end collaborative computing framework integrating foundation models into autonomous UAV cruising scenarios - effectively leveraging foundation models to enhance performance of resource-constrained unmanned aerial vehicle platforms. To reduce onboard computation and data transmission overhead, CoDrone employs grayscale imagery for the navigation model. When enhanced environmental perception is required, CoDrone leverages the edge-assisted foundation model Depth Anything V2 for depth estimation and introduces a novel one-dimensional occupancy grid-based navigation method - enabling fine-grained scene understanding while advancing efficiency and representational simplicity of autonomous navigation. A key component of CoDrone is a Deep Reinforcement Learning-based neural scheduler that seamlessly integrates depth estimation with autonomous navigation decisions, enabling real-time adaptation to dynamic environments. Furthermore, the framework introduces a UAV-specific vision language interaction module incorporating domain-tailored low-level flight primitives to enable effective interaction between the cloud foundation model and the UAV. The introduction of VLM enhances open-set reasoning capabilities in complex unseen scenarios. Experimental results show CoDrone outperforms baseline methods under varying flight speeds and network conditions, achieving a 40% increase in average flight distance and a 5% improvement in average Quality of Navigation.

**arXiv ID:** 2512.19083
</details>

<details>
<summary><strong>WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving</strong> - Pengxuan Yang, Ben Lu, Zhongpu Xia, Chao Han, Yinfeng Gao, Teng Zhang, Kun Zhan, XianPeng Lang, Yupeng Zheng, Qichao Zhang - [[pdf]](https://arxiv.org/pdf/2512.19133)</summary>

**Abstract:** Latent World Models enhance scene representation through temporal self-supervised learning, presenting a perception annotation-free paradigm for end-to-end autonomous driving. However, the reconstruction-oriented representation learning tangles perception with planning tasks, leading to suboptimal optimization for planning. To address this challenge, we propose WorldRFT, a planning-oriented latent world model framework that aligns scene representation learning with planning via a hierarchical planning decomposition and local-aware interactive refinement mechanism, augmented by reinforcement learning fine-tuning (RFT) to enhance safety-critical policy performance. Specifically, WorldRFT integrates a vision-geometry foundation model to improve 3D spatial awareness, employs hierarchical planning task decomposition to guide representation optimization, and utilizes local-aware iterative refinement to derive a planning-oriented driving policy. Furthermore, we introduce Group Relative Policy Optimization (GRPO), which applies trajectory Gaussianization and collision-aware rewards to fine-tune the driving policy, yielding systematic improvements in safety. WorldRFT achieves state-of-the-art (SOTA) performance on both open-loop nuScenes and closed-loop NavSim benchmarks. On nuScenes, it reduces collision rates by 83% (0.30% -> 0.05%). On NavSim, using camera-only sensors input, it attains competitive performance with the LiDAR-based SOTA method DiffusionDrive (87.8 vs. 88.1 PDMS).

**arXiv ID:** 2512.19133
</details>

<details>
<summary><strong>MaP-AVR: A Meta-Action Planner for Agents Leveraging Vision Language Models and Retrieval-Augmented Generation</strong> - Zhenglong Guo, Yiming Zhao, Feng Jiang, Heng Jin, Zongbao Feng, Jianbin Zhou, Siyuan Xu - [[pdf]](https://arxiv.org/pdf/2512.19453)</summary>

**Abstract:** Embodied robotic AI systems designed to manage complex daily tasks rely on a task planner to understand and decompose high-level tasks. While most research focuses on enhancing the task-understanding abilities of LLMs/VLMs through fine-tuning or chain-of-thought prompting, this paper argues that defining the planned skill set is equally crucial. To handle the complexity of daily environments, the skill set should possess a high degree of generalization ability. Empirically, more abstract expressions tend to be more generalizable. Therefore, we propose to abstract the planned result as a set of meta-actions. Each meta-action comprises three components: {move/rotate, end-effector status change, relationship with the environment}. This abstraction replaces human-centric concepts, such as grasping or pushing, with the robot's intrinsic functionalities. As a result, the planned outcomes align seamlessly with the complete range of actions that the robot is capable of performing. Furthermore, to ensure that the LLM/VLM accurately produces the desired meta-action format, we employ the Retrieval-Augmented Generation (RAG) technique, which leverages a database of human-annotated planning demonstrations to facilitate in-context learning. As the system successfully completes more tasks, the database will self-augment to continue supporting diversity. The meta-action set and its integration with RAG are two novel contributions of our planner, denoted as MaP-AVR, the meta-action planner for agents composed of VLM and RAG. To validate its efficacy, we design experiments using GPT-4o as the pre-trained LLM/VLM model and OmniGibson as our robotic platform. Our approach demonstrates promising performance compared to the current state-of-the-art method. Project page: this https URL.

**arXiv ID:** 2512.19453
</details>

<details>
<summary><strong>VLNVerse: A Benchmark for Vision-Language Navigation with Versatile, Embodied, Realistic Simulation and Evaluation</strong> - Sihao Lin, Zerui Li, Xunyi Zhao, Gengze Zhou, Liuyi Wang, Rong Wei, Rui Tang, Juncheng Li, Hanqing Wang, Jiangmiao Pang, Anton van den Hengel, Jiajun Liu, Qi Wu - [[pdf]](https://arxiv.org/pdf/2512.19021)</summary>

**Abstract:** Despite remarkable progress in Vision-Language Navigation (VLN), existing benchmarks remain confined to fixed, small-scale datasets with naive physical simulation. These shortcomings limit the insight that the benchmarks provide into sim-to-real generalization, and create a significant research gap. Furthermore, task fragmentation prevents unified/shared progress in the area, while limited data scales fail to meet the demands of modern LLM-based pretraining. To overcome these limitations, we introduce VLNVerse: a new large-scale, extensible benchmark designed for Versatile, Embodied, Realistic Simulation, and Evaluation. VLNVerse redefines VLN as a scalable, full-stack embodied AI problem. Its Versatile nature unifies previously fragmented tasks into a single framework and provides an extensible toolkit for researchers. Its Embodied design moves beyond intangible and teleporting "ghost" agents that support full-kinematics in a Realistic Simulation powered by a robust physics engine. We leverage the scale and diversity of VLNVerse to conduct a comprehensive Evaluation of existing methods, from classic models to MLLM-based agents. We also propose a novel unified multi-task model capable of addressing all tasks within the benchmark. VLNVerse aims to narrow the gap between simulated navigation and real-world generalization, providing the community with a vital tool to boost research towards scalable, general-purpose embodied locomotion agents.

**arXiv ID:** 2512.19021
</details>

<details>
<summary><strong>GUIDEd Agents: Enhancing Navigation Policies through Task-Specific Uncertainty Abstraction in Localization-Limited Environments</strong> - Gokul Puthumanaillam, Paulo Padrao, Jose Fuentes, Leonardo Bobadilla, Melkior Ornik - [[pdf]](https://arxiv.org/pdf/2410.15178)</summary>

**Abstract:** Autonomous vehicles performing navigation tasks in complex environments face significant challenges due to uncertainty in state estimation. In many scenarios, such as stealth operations or resource-constrained settings, accessing high-precision localization comes at a significant cost, forcing robots to rely primarily on less precise state estimates. Our key observation is that different tasks require varying levels of precision in different regions: a robot navigating a crowded space might need precise localization near obstacles but can operate effectively with less precision elsewhere. In this paper, we present a planning method for integrating task-specific uncertainty requirements directly into navigation policies. We introduce Task-Specific Uncertainty Maps (TSUMs), which abstract the acceptable levels of state estimation uncertainty across different regions. TSUMs align task requirements and environmental features using a shared representation space, generated via a domain-adapted encoder. Using TSUMs, we propose Generalized Uncertainty Integration for Decision-Making and Execution (GUIDE), a policy conditioning framework that incorporates these uncertainty requirements into robot decision-making. We find that TSUMs provide an effective way to abstract task-specific uncertainty requirements, and conditioning policies on TSUMs enables the robot to reason about the context-dependent value of certainty and adapt its behavior accordingly. We show how integrating GUIDE into reinforcement learning frameworks allows the agent to learn navigation policies that effectively balance task completion and uncertainty management without explicit reward engineering. We evaluate GUIDE on various real-world robotic navigation tasks and find that it demonstrates significant improvement in task completion rates compared to baseline methods that do not explicitly consider task-specific uncertainty.

**arXiv ID:** 2410.15178
</details>

<details>
<summary><strong>FreeAskWorld: An Interactive and Closed-Loop Simulator for Human-Centric Embodied AI</strong> - Yuhang Peng, Yizhou Pan, Xinning He, Jihaoyu Yang, Xinyu Yin, Han Wang, Xiaoji Zheng, Chao Gao, Jiangtao Gong - [[pdf]](https://arxiv.org/pdf/2511.13524)</summary>

**Abstract:** As embodied intelligence emerges as a core frontier in artificial intelligence research, simulation platforms must evolve beyond low-level physical interactions to capture complex, human-centered social behaviors. We introduce FreeAskWorld, an interactive simulation framework that integrates large language models (LLMs) for high-level behavior planning and semantically grounded interaction, informed by theories of intention and social cognition. Our framework supports scalable, realistic human-agent simulations and includes a modular data generation pipeline tailored for diverse embodied tasks. To validate the framework, we extend the classic Vision-and-Language Navigation (VLN) task into a interaction enriched Direction Inquiry setting, wherein agents can actively seek and interpret navigational guidance. We present and publicly release FreeAskWorld, a large-scale benchmark dataset comprising reconstructed environments, six diverse task types, 16 core object categories, 63,429 annotated sample frames, and more than 17 hours of interaction data to support training and evaluation of embodied AI systems. We benchmark VLN models, and human participants under both open-loop and closed-loop settings. Experimental results demonstrate that models fine-tuned on FreeAskWorld outperform their original counterparts, achieving enhanced semantic understanding and interaction competency. These findings underscore the efficacy of socially grounded simulation frameworks in advancing embodied AI systems toward sophisticated high-level planning and more naturalistic human-agent interaction. Importantly, our work underscores that interaction itself serves as an additional information modality.

**arXiv ID:** 2511.13524
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
