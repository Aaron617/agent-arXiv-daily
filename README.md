# Agent arXiv Daily

**Last Updated:** 2025-09-10 02:34:51

**Total Papers:** 41

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (1 papers)</h2></summary>

<details>
<summary><strong>Robust Docking Maneuvers for Autonomous Trolley Collection: An Optimization-Based Visual Servoing Scheme</strong> - Yuhan Pang, Bingyi Xia, Zhe Zhang, Zhirui Sun, Peijia Xie, Bike Zhu, Wenjun Xu, Jiankun Wang - [[pdf]](https://arxiv.org/pdf/2509.07413)</summary>

**Abstract:** Service robots have demonstrated significant potential for autonomous trolley collection and redistribution in public spaces like airports or warehouses to improve efficiency and reduce cost. Usually, a fully autonomous system for the collection and transportation of multiple trolleys is based on a Leader-Follower formation of mobile manipulators, where reliable docking maneuvers of the mobile base are essential to align trolleys into organized queues. However, developing a vision-based robotic docking system faces significant challenges: high precision requirements, environmental disturbances, and inherent robot constraints. To address these challenges, we propose an optimization-based Visual Servoing scheme that incorporates active infrared markers for robust feature extraction across diverse lighting conditions. This framework explicitly models nonholonomic kinematics and visibility constraints within the Hybrid Visual Servoing problem, augmented with an observer for disturbance rejection to ensure precise and stable docking. Experimental results across diverse environments demonstrate the robustness of this system, with quantitative evaluations confirming high docking accuracy.

**arXiv ID:** 2509.07413
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (6 papers)</h2></summary>

<details>
<summary><strong>Towards Generalized Routing: Model and Agent Orchestration for Adaptive and Efficient Inference</strong> - Xiyu Guo, Shan Wang, Chunfang Ji, Xuefeng Zhao, Wenhao Xi, Yaoyao Liu, Qinglan Li, Chao Deng, Junlan Feng - [[pdf]](https://arxiv.org/pdf/2509.07571)</summary>

**Abstract:** The rapid advancement of large language models (LLMs) and domain-specific AI agents has greatly expanded the ecosystem of AI-powered services. User queries, however, are highly diverse and often span multiple domains and task types, resulting in a complex and heterogeneous landscape. This diversity presents a fundamental routing challenge: how to accurately direct each query to an appropriate execution unit while optimizing both performance and efficiency. To address this, we propose MoMA (Mixture of Models and Agents), a generalized routing framework that integrates both LLM and agent-based routing. Built upon a deep understanding of model and agent capabilities, MoMA effectively handles diverse queries through precise intent recognition and adaptive routing strategies, achieving an optimal balance between efficiency and cost. Specifically, we construct a detailed training dataset to profile the capabilities of various LLMs under different routing model structures, identifying the most suitable tasks for each LLM. During inference, queries are dynamically routed to the LLM with the best cost-performance efficiency. We also introduce an efficient agent selection strategy based on a context-aware state machine and dynamic masking. Experimental results demonstrate that the MoMA router offers superior cost-efficiency and scalability compared to existing approaches.

**arXiv ID:** 2509.07571
</details>

<details>
<summary><strong>EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control</strong> - Delin Qu, Haoming Song, Qizhi Chen, Zhaoqing Chen, Xianqiang Gao, Xinyi Ye, Qi Lv, Modi Shi, Guanghui Ren, Cheng Ruan, Maoqing Yao, Haoran Yang, Jiacheng Bao, Bin Zhao, Dong Wang - [[pdf]](https://arxiv.org/pdf/2508.21112)</summary>

**Abstract:** The human ability to seamlessly perform multimodal reasoning and physical interaction in the open world is a core goal for general-purpose embodied intelligent systems. Recent vision-language-action (VLA) models, which are co-trained on large-scale robot and visual-text data, have demonstrated notable progress in general robot control. However, they still fail to achieve human-level flexibility in interleaved reasoning and interaction. In this work, introduce EO-Robotics, consists of EO-1 model and EO-Data1.5M dataset. EO-1 is a unified embodied foundation model that achieves superior performance in multimodal embodied reasoning and robot control through interleaved vision-text-action pre-training. The development of EO-1 is based on two key pillars: (i) a unified architecture that processes multimodal inputs indiscriminately (image, text, video, and action), and (ii) a massive, high-quality multimodal embodied reasoning dataset, EO-Data1.5M, which contains over 1.5 million samples with emphasis on interleaved vision-text-action comprehension. EO-1 is trained through synergies between auto-regressive decoding and flow matching denoising on EO-Data1.5M, enabling seamless robot action generation and multimodal embodied reasoning. Extensive experiments demonstrate the effectiveness of interleaved vision-text-action learning for open-world understanding and generalization, validated through a variety of long-horizon, dexterous manipulation tasks across multiple embodiments. This paper details the architecture of EO-1, the data construction strategy of EO-Data1.5M, and the training methodology, offering valuable insights for developing advanced embodied foundation models.

**arXiv ID:** 2508.21112
</details>

<details>
<summary><strong>Adaptive Evolutionary Framework for Safe, Efficient, and Cooperative Autonomous Vehicle Interactions</strong> - Zhen Tian, Zhihao Lin - [[pdf]](https://arxiv.org/pdf/2509.07411)</summary>

**Abstract:** Modern transportation systems face significant challenges in ensuring road safety, given serious injuries caused by road accidents. The rapid growth of autonomous vehicles (AVs) has prompted new traffic designs that aim to optimize interactions among AVs. However, effective interactions between AVs remains challenging due to the absence of centralized control. Besides, there is a need for balancing multiple factors, including passenger demands and overall traffic efficiency. Traditional rule-based, optimization-based, and game-theoretic approaches each have limitations in addressing these challenges. Rule-based methods struggle with adaptability and generalization in complex scenarios, while optimization-based methods often require high computational resources. Game-theoretic approaches, such as Stackelberg and Nash games, suffer from limited adaptability and potential inefficiencies in cooperative settings. This paper proposes an Evolutionary Game Theory (EGT)-based framework for AV interactions that overcomes these limitations by utilizing a decentralized and adaptive strategy evolution mechanism. A causal evaluation module (CEGT) is introduced to optimize the evolutionary rate, balancing mutation and evolution by learning from historical interactions. Simulation results demonstrate the proposed CEGT outperforms EGT and popular benchmark games in terms of lower collision rates, improved safety distances, higher speeds, and overall better performance compared to Nash and Stackelberg games across diverse scenarios and parameter settings.

**arXiv ID:** 2509.07411
</details>

<details>
<summary><strong>VeriOS: Query-Driven Proactive Human-Agent-GUI Interaction for Trustworthy OS Agents</strong> - Zheng Wu, Heyuan Huang, Xingyu Lou, Xiangmou Qu, Pengzhou Cheng, Zongru Wu, Weiwen Liu, Weinan Zhang, Jun Wang, Zhaoxiang Wang, Zhuosheng Zhang - [[pdf]](https://arxiv.org/pdf/2509.07553)</summary>

**Abstract:** With the rapid progress of multimodal large language models, operating system (OS) agents become increasingly capable of automating tasks through on-device graphical user interfaces (GUIs). However, most existing OS agents are designed for idealized settings, whereas real-world environments often present untrustworthy conditions. To mitigate risks of over-execution in such scenarios, we propose a query-driven human-agent-GUI interaction framework that enables OS agents to decide when to query humans for more reliable task completion. Built upon this framework, we introduce VeriOS-Agent, a trustworthy OS agent trained with a two-stage learning paradigm that falicitate the decoupling and utilization of meta-knowledge. Concretely, VeriOS-Agent autonomously executes actions in normal conditions while proactively querying humans in untrustworthy scenarios. Experiments show that VeriOS-Agent improves the average step-wise success rate by 20.64\% in untrustworthy scenarios over the state-of-the-art, without compromising normal performance. Analysis highlights VeriOS-Agent's rationality, generalizability, and scalability. The codes, datasets and models are available at this https URL.

**arXiv ID:** 2509.07553
</details>

<details>
<summary><strong>Semi-SMD: Semi-Supervised Metric Depth Estimation via Surrounding Cameras for Autonomous Driving</strong> - Yusen Xie, Zhengmin Huang, Shaojie Shen, Jun Ma - [[pdf]](https://arxiv.org/pdf/2503.19713)</summary>

**Abstract:** In this paper, we introduce Semi-SMD, a novel metric depth estimation framework tailored for surrounding cameras equipment in autonomous driving. In this work, the input data consists of adjacent surrounding frames and camera parameters. We propose a unified spatial-temporal-semantic fusion module to construct the visual fused features. Cross-attention components for surrounding cameras and adjacent frames are utilized to focus on metric scale information refinement and temporal feature matching. Building on this, we propose a pose estimation framework using surrounding cameras, their corresponding estimated depths, and extrinsic parameters, which effectively address the scale ambiguity in multi-camera setups. Moreover, semantic world model and monocular depth estimation world model are integrated to supervised the depth estimation, which improve the quality of depth estimation. We evaluate our algorithm on DDAD and nuScenes datasets, and the results demonstrate that our method achieves state-of-the-art performance in terms of surrounding camera based depth estimation quality. The source code will be available on this https URL.

**arXiv ID:** 2503.19713
</details>

<details>
<summary><strong>T-araVLN: Translator for Agricultural Robotic Agents on Vision-and-Language Navigation</strong> - Xiaobei Zhao, Xingqi Lyu, Xiang Li - [[pdf]](https://arxiv.org/pdf/2509.06644)</summary>

**Abstract:** Agricultural robotic agents have been becoming powerful helpers in a wide range of agricultural tasks, nevertheless, still heavily rely on manual operation or untransportable railway for movement. The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling agents navigate to the target position following the natural language instructions. AgriVLN effectively understands the simple instructions, however, often misunderstands the complicated instructions. To bridge this gap, we propose the method of Translator for Agricultural Robotic Agents on Vision-and-Language Navigation (T-araVLN), in which the Instruction Translator module translates the original instruction to be both refined and precise. Being evaluated on the A2A benchmark, our T-araVLN effectively improves SR from 0.47 to 0.63 and reduces NE from 2.91m to 2.28m, demonstrating the state-of-the-art performance in the agricultural domain. Code: this https URL.

**arXiv ID:** 2509.06644
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>Autonomous Code Evolution Meets NP-Completeness</strong> - Cunxi Yu, Rongjian Liang, Chia-Tung Ho, Haoxing Ren - [[pdf]](https://arxiv.org/pdf/2509.07367)</summary>

**Abstract:** Large language models (LLMs) have recently shown strong coding abilities, enabling not only static code generation but also iterative code self-evolving through agentic frameworks. Recently, AlphaEvolve \cite{novikov2025alphaevolve} demonstrated that LLM-based coding agents can autonomously improve algorithms and surpass human experts, with scopes limited to isolated kernels spanning hundreds of lines of code. Inspired by AlphaEvolve, we present SATLUTION, the first framework to extend LLM-based code evolution to the full repository scale, encompassing hundreds of files and tens of thousands of lines of C/C++ code. Targeting Boolean Satisfiability (SAT), the canonical NP-complete problem and a cornerstone of both theory and applications. SATLUTION orchestrates LLM agents to directly evolve solver repositories under strict correctness guarantees and distributed runtime feedback, while simultaneously self-evolving its own evolution policies and rules. Starting from SAT Competition 2024 codebases and benchmark, SATLUTION evolved solvers that decisively outperformed the human-designed winners of the SAT Competition 2025, and also surpassed both 2024 and 2025 champions on the 2024 benchmarks.

**arXiv ID:** 2509.07367
</details>

<details>
<summary><strong>Talking with Oompa Loompas: A novel framework for evaluating linguistic acquisition of LLM agents</strong> - Sankalp Tattwadarshi Swain, Anshika Krishnatray, Dhruv Kumar, Jagat Sesh Challa - [[pdf]](https://arxiv.org/pdf/2509.07389)</summary>

**Abstract:** Existing evaluation studies on linguistic competence of large language models (LLM agents) have focused primarily on vocabulary learning, morphological rule induction, syntactic generalization, pragmatic inference, and cross-linguistic transfer. However, none assess whether LLM agents can acquire a language through pattern recognition and interactive feedback, a central feature of human language acquisition. We propose a novel experimental framework in which an LLM agent is evaluated on its ability to acquire and use a newly constructed language (Tinkatongue) in conversation with a bot that understands only Tinkatongue. Our findings show that LLM agents fail to establish a conversation within 100 responses, yet they adopt distinct strategies that mirror human approaches to language learning. The results suggest a new direction for evaluation benchmarks and open pathways to model designs that learn more effectively from interactive feedback.

**arXiv ID:** 2509.07389
</details>

<details>
<summary><strong>EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation</strong> - Yunbo Long, Liming Xu, Lukas Beckenbauer, Yuhan Liu, Alexandra Brintrup - [[pdf]](https://arxiv.org/pdf/2509.04310)</summary>

**Abstract:** Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in \textit{complex}, \textit{multi-turn} negotiations, opening new avenues for agentic AI. However, existing LLM agents largely overlook the functional role of emotions in such negotiations, instead generating passive, preference-driven emotional responses that make them vulnerable to manipulation and strategic exploitation by adversarial counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines -- vanilla strategies and fixed-emotion strategies -- for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation.

**arXiv ID:** 2509.04310
</details>

<details>
<summary><strong>Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection</strong> - Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu - [[pdf]](https://arxiv.org/pdf/2503.15552)</summary>

**Abstract:** The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the SE attack mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.

**arXiv ID:** 2503.15552
</details>

<details>
<summary><strong>CAViAR: Critic-Augmented Video Agentic Reasoning</strong> - Sachit Menon, Ahmet Iscen, Arsha Nagrani, Tobias Weyand, Carl Vondrick, Cordelia Schmid - [[pdf]](https://arxiv.org/pdf/2509.07680)</summary>

**Abstract:** Video understanding has seen significant progress in recent years, with models' performance on perception from short clips continuing to rise. Yet, multiple recent benchmarks, such as LVBench, Neptune, and ActivityNet-RTL, show performance wanes for tasks requiring complex reasoning on videos as queries grow more complex and videos grow longer. In this work, we ask: can existing perception capabilities be leveraged to successfully perform more complex video reasoning? In particular, we develop a large language model agent given access to video modules as subagents or tools. Rather than following a fixed procedure to solve queries as in previous work such as Visual Programming, ViperGPT, and MoReVQA, the agent uses the results of each call to a module to determine subsequent steps. Inspired by work in the textual reasoning domain, we introduce a critic to distinguish between instances of successful and unsuccessful sequences from the agent. We show that the combination of our agent and critic achieve strong performance on the previously-mentioned datasets.

**arXiv ID:** 2509.07680
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (6 papers)</h2></summary>

<details>
<summary><strong>Astra: A Multi-Agent System for GPU Kernel Performance Optimization</strong> - Anjiang Wei, Tianran Sun, Yogesh Seenichamy, Hang Song, Anne Ouyang, Azalia Mirhoseini, Ke Wang, Alex Aiken - [[pdf]](https://arxiv.org/pdf/2509.07506)</summary>

**Abstract:** GPU kernel optimization has long been a central challenge at the intersection of high-performance computing and machine learning. Efficient kernels are crucial for accelerating large language model (LLM) training and serving, yet attaining high performance typically requires extensive manual tuning. Compiler-based systems reduce some of this burden, but still demand substantial manual design and engineering effort. Recently, researchers have explored using LLMs for GPU kernel generation, though prior work has largely focused on translating high-level PyTorch modules into CUDA code. In this work, we introduce Astra, the first LLM-based multi-agent system for GPU kernel optimization. Unlike previous approaches, Astra starts from existing CUDA implementations extracted from SGLang, a widely deployed framework for serving LLMs, rather than treating PyTorch modules as the specification. Within Astra, specialized LLM agents collaborate through iterative code generation, testing, profiling, and planning to produce kernels that are both correct and high-performance. On kernels from SGLang, Astra achieves an average speedup of 1.32x using zero-shot prompting with OpenAI o4-mini. A detailed case study further demonstrates that LLMs can autonomously apply loop transformations, optimize memory access patterns, exploit CUDA intrinsics, and leverage fast math operations to yield substantial performance gains. Our work highlights multi-agent LLM systems as a promising new paradigm for GPU kernel optimization.

**arXiv ID:** 2509.07506
</details>

<details>
<summary><strong>COMMA: A Communicative Multimodal Multi-Agent Benchmark</strong> - Timothy Ossowski, Jixuan Chen, Danyal Maqbool, Zefan Cai, Tyler Bradshaw, Junjie Hu - [[pdf]](https://arxiv.org/pdf/2410.07553)</summary>

**Abstract:** The rapid advances of multimodal agents built on large foundation models have largely overlooked their potential for language-based communication between agents in collaborative tasks. This oversight presents a critical gap in understanding their effectiveness in real-world deployments, particularly when communicating with humans. Existing agentic benchmarks fail to address key aspects of inter-agent communication and collaboration, particularly in scenarios where agents have unequal access to information and must work together to achieve tasks beyond the scope of individual capabilities. To fill this gap, we introduce COMMA: a novel puzzle benchmark designed to evaluate the collaborative performance of multimodal multi-agent systems through language communication. Our benchmark features a variety of multimodal puzzles, providing a comprehensive evaluation across four key categories of agentic capability in a communicative collaboration setting. Our findings reveal surprising weaknesses in state-of-the-art models, including strong proprietary models like GPT-4o and reasoning models like o4-mini. Many chain of thought reasoning models such as R1-Onevision and LLaVA-CoT struggle to outperform even a random baseline in agent-agent collaboration, indicating a potential growth area in their communication abilities.

**arXiv ID:** 2410.07553
</details>

<details>
<summary><strong>SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents</strong> - Xuan-Phi Nguyen, Shrey Pandit, Revanth Gangi Reddy, Austin Xu, Silvio Savarese, Caiming Xiong, Shafiq Joty - [[pdf]](https://arxiv.org/pdf/2509.06283)</summary>

**Abstract:** Equipping large language models (LLMs) with complex, interleaved reasoning and tool-use capabilities has become a key focus in agentic AI research, especially with recent advances in reasoning-oriented (``thinking'') models. Such capabilities are key to unlocking a number of important applications. One such application is Deep Research (DR), which requires extensive search and reasoning over many sources. Our work in this paper focuses on the development of native Autonomous Single-Agent models for DR featuring minimal web crawling and Python tool integration. Unlike multi-agent systems, where agents take up pre-defined roles and are told what to do at each step in a static workflow, an autonomous single-agent determines its next action dynamically based on context, without manual directive. While prior work has proposed training recipes for base or instruction-tuned LLMs, we focus on continual reinforcement learning (RL) of reasoning-optimized models to further enhance agentic skills while preserving reasoning ability. Towards this end, we propose a simple RL recipe with entirely synthetic data, which we apply to various open-source LLMs. Our best variant SFR-DR-20B achieves up to 28.7% on Humanity's Last Exam benchmark. In addition, we conduct key analysis experiments to provide more insights into our methodologies.

**arXiv ID:** 2509.06283
</details>

<details>
<summary><strong>Grid-Agent: An LLM-Powered Multi-Agent System for Power Grid Control</strong> - Yan Zhang, Ahmad Mohammad Saber, Amr Youssef, Deepa Kundur - [[pdf]](https://arxiv.org/pdf/2508.05702)</summary>

**Abstract:** Modern power grids face unprecedented complexity from Distributed Energy Resources (DERs), Electric Vehicles (EVs), and extreme weather, while also being increasingly exposed to cyberattacks that can trigger grid violations. This paper introduces Grid-Agent, an autonomous AI-driven framework that leverages Large Language Models (LLMs) within a multi-agent system to detect and remediate violations. Grid-Agent integrates semantic reasoning with numerical precision through modular agents: a planning agent generates coordinated action sequences using power flow solvers, while a validation agent ensures stability and safety through sandboxed execution with rollback mechanisms. To enhance scalability, the framework employs an adaptive multi-scale network representation that dynamically adjusts encoding schemes based on system size and complexity. Violation resolution is achieved through optimizing switch configurations, battery deployment, and load curtailment. Our experiments on IEEE and CIGRE benchmark networks, including the IEEE 69-bus, CIGRE MV, IEEE 30-bus test systems, demonstrate superior mitigation performance, highlighting Grid-Agent's suitability for modern smart grids requiring rapid, adaptive response.

**arXiv ID:** 2508.05702
</details>

<details>
<summary><strong>Efficient Multi-Agent Coordination via Dynamic Joint-State Graph Construction</strong> - Yanlin Zhou, Manshi Limbu, Xuesu Xiao - [[pdf]](https://arxiv.org/pdf/2509.07234)</summary>

**Abstract:** Multi-agent pathfinding (MAPF) traditionally focuses on collision avoidance, but many real-world applications require active coordination between agents to improve team performance. This paper introduces Team Coordination on Graphs with Risky Edges (TCGRE), where agents collaborate to reduce traversal costs on high-risk edges via support from teammates. We reformulate TCGRE as a 3D matching problem-mapping robot pairs, support pairs, and time steps-and rigorously prove its NP-hardness via reduction from Minimum 3D Matching. To address this complexity, (in the conference version) we proposed efficient decomposition methods, reducing the problem to tractable subproblems: Joint-State Graph (JSG): Encodes coordination as a single-agent shortest-path problem. Coordination-Exhaustive Search (CES): Optimizes support assignments via exhaustive pairing. Receding-Horizon Optimistic Cooperative A* (RHOCA*): Balances optimality and scalability via horizon-limited planning. Further in this extension, we introduce a dynamic graph construction method (Dynamic-HJSG), leveraging agent homogeneity to prune redundant states and reduce computational overhead by constructing the joint-state graph dynamically. Theoretical analysis shows Dynamic-HJSG preserves optimality while lowering complexity from exponential to polynomial in key cases. Empirical results validate scalability for large teams and graphs, with HJSG outperforming baselines greatly in runtime in different sizes and types of graphs. This work bridges combinatorial optimization and multi-agent planning, offering a principled framework for collaborative pathfinding with provable guarantees, and the key idea of the solution can be widely extended to many other collaborative optimization problems, such as MAPF.

**arXiv ID:** 2509.07234
</details>

<details>
<summary><strong>A data-driven discretized CS:GO simulation environment to facilitate strategic multi-agent planning research</strong> - Yunzhe Wang, Volkan Ustun, Chris McGroarty - [[pdf]](https://arxiv.org/pdf/2509.06355)</summary>

**Abstract:** Modern simulation environments for complex multi-agent interactions must balance high-fidelity detail with computational efficiency. We present DECOY, a novel multi-agent simulator that abstracts strategic, long-horizon planning in 3D terrains into high-level discretized simulation while preserving low-level environmental fidelity. Using Counter-Strike: Global Offensive (CS:GO) as a testbed, our framework accurately simulates gameplay using only movement decisions as tactical positioning -- without explicitly modeling low-level mechanics such as aiming and shooting. Central to our approach is a waypoint system that simplifies and discretizes continuous states and actions, paired with neural predictive and generative models trained on real CS:GO tournament data to reconstruct event outcomes. Extensive evaluations show that replays generated from human data in DECOY closely match those observed in the original game. Our publicly available simulation environment provides a valuable tool for advancing research in strategic multi-agent planning and behavior generation.

**arXiv ID:** 2509.06355
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>Instruction Agent: Enhancing Agent with Expert Demonstration</strong> - Yinheng Li, Hailey Hultquist, Justin Wagle, Kazuhito Koishida - [[pdf]](https://arxiv.org/pdf/2509.07098)</summary>

**Abstract:** Graphical user interface (GUI) agents have advanced rapidly but still struggle with complex tasks involving novel UI elements, long-horizon actions, and personalized trajectories. In this work, we introduce Instruction Agent, a GUI agent that leverages expert demonstrations to solve such tasks, enabling completion of otherwise difficult workflows. Given a single demonstration, the agent extracts step-by-step instructions and executes them by strictly following the trajectory intended by the user, which avoids making mistakes during execution. The agent leverages the verifier and backtracker modules further to improve robustness. Both modules are critical to understand the current outcome from each action and handle unexpected interruptions(such as pop-up windows) during execution. Our experiments show that Instruction Agent achieves a 60% success rate on a set of tasks in OSWorld that all top-ranked agents failed to complete. The Instruction Agent offers a practical and extensible framework, bridging the gap between current GUI agents and reliable real-world GUI task automation.

**arXiv ID:** 2509.07098
</details>

<details>
<summary><strong>A Mixed User-Centered Approach to Enable Augmented Intelligence in Intelligent Tutoring Systems: The Case of MathAIde app</strong> - Guilherme Guerino, Luiz Rodrigues, Luana Bianchini, Mariana Alves, Marcelo Marinho, Thomaz Veloso, Valmir Macario, Diego Dermeval, Thales Vieira, Ig Bittencourt, Seiji Isotani - [[pdf]](https://arxiv.org/pdf/2508.00103)</summary>

**Abstract:** This study explores the integration of Augmented Intelligence (AuI) in Intelligent Tutoring Systems (ITS) to address challenges in Artificial Intelligence in Education (AIED), including teacher involvement, AI reliability, and resource accessibility. We present MathAIde, an ITS that uses computer vision and AI to correct mathematics exercises from student work photos and provide feedback. The system was designed through a collaborative process involving brainstorming with teachers, high-fidelity prototyping, A/B testing, and a real-world case study. Findings emphasize the importance of a teacher-centered, user-driven approach, where AI suggests remediation alternatives while teachers retain decision-making. Results highlight efficiency, usability, and adoption potential in classroom contexts, particularly in resource-limited environments. The study contributes practical insights into designing ITSs that balance user needs and technological feasibility, while advancing AIED research by demonstrating the effectiveness of a mixed-methods, user-centered approach to implementing AuI in educational technologies.

**arXiv ID:** 2508.00103
</details>

<details>
<summary><strong>Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models</strong> - Qiming Guo, Jinwen Tang, Xingran Huang - [[pdf]](https://arxiv.org/pdf/2508.17674)</summary>

**Abstract:** We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.

**arXiv ID:** 2508.17674
</details>

<details>
<summary><strong>Machine Generalize Learning in Agent-Based Models: Going Beyond Surrogate Models for Calibration in ABMs</strong> - Sima Najafzadehkhoei, George Vega Yon, Bernardo Modenesi, Derek S.Meyer - [[pdf]](https://arxiv.org/pdf/2509.07013)</summary>

**Abstract:** Calibrating agent-based epidemic models is computationally demanding. We present a supervised machine learning calibrator that learns the inverse mapping from epidemic time series to SIR parameters. A three-layer bidirectional LSTM ingests 60-day incidence together with population size and recovery rate, and outputs transmission probability, contact rate, and R0. Training uses a composite loss with an epidemiology-motivated consistency penalty that encourages R0 \* recovery rate to equal transmission probability \* contact rate.
In a 1000-scenario simulation study, we compare the calibrator with Approximate Bayesian Computation (likelihood-free MCMC). The method achieves lower error across all targets (MAE: R0 0.0616 vs 0.275; transmission 0.0715 vs 0.128; contact 1.02 vs 4.24), produces tighter predictive intervals with near nominal coverage, and reduces wall clock time from 77.4 s to 2.35 s per calibration. Although contact rate and transmission probability are partially nonidentifiable, the approach reproduces epidemic curves more faithfully than ABC, enabling fast and practical calibration. We evaluate it on SIR agent based epidemics generated with epiworldR and provide an implementation in R.

**arXiv ID:** 2509.07013
</details>

<details>
<summary><strong>Safe and Non-Conservative Contingency Planning for Autonomous Vehicles via Online Learning-Based Reachable Set Barriers</strong> - Rui Yang, Lei Zheng, Shuzhi Sam Ge, Jun Ma - [[pdf]](https://arxiv.org/pdf/2509.07464)</summary>

**Abstract:** Autonomous vehicles must navigate dynamically uncertain environments while balancing the safety and driving efficiency. This challenge is exacerbated by the unpredictable nature of surrounding human-driven vehicles (HVs) and perception inaccuracies, which require planners to adapt to evolving uncertainties while maintaining safe trajectories. Overly conservative planners degrade driving efficiency, while deterministic approaches may encounter serious issues and risks of failure when faced with sudden and unexpected maneuvers. To address these issues, we propose a real-time contingency trajectory optimization framework in this paper. By employing event-triggered online learning of HV control-intent sets, our method dynamically quantifies multi-modal HV uncertainties and refines the forward reachable set (FRS) incrementally. Crucially, we enforce invariant safety through FRS-based barrier constraints that ensure safety without reliance on accurate trajectory prediction of HVs. These constraints are embedded in contingency trajectory optimization and solved efficiently through consensus alternative direction method of multipliers (ADMM). The system continuously adapts to the uncertainties in HV behaviors, preserving feasibility and safety without resorting to excessive conservatism. High-fidelity simulations on highway and urban scenarios, as well as a series of real-world experiments demonstrate significant improvements in driving efficiency and passenger comfort while maintaining safety under uncertainty. The project page is available at this https URL.

**arXiv ID:** 2509.07464
</details>

<details>
<summary><strong>Prototype: A Keyword Spotting-Based Intelligent Audio SoC for IoT</strong> - Huihong Liang, Dongxuan Jia, Youquan Wang, Longtao Huang, Shida Zhong, Luping Xiang, Lei Huang, Tao Yuan - [[pdf]](https://arxiv.org/pdf/2509.06964)</summary>

**Abstract:** In this demo, we present a compact intelligent audio system-on-chip (SoC) integrated with a keyword spotting accelerator, enabling ultra-low latency, low-power, and low-cost voice interaction in Internet of Things (IoT) devices. Through algorithm-hardware co-design, the system's energy efficiency is maximized. We demonstrate the system's capabilities through a live FPGA-based prototype, showcasing stable performance and real-time voice interaction for edge intelligence applications.

**arXiv ID:** 2509.06964
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (17 papers)</h2></summary>

<details>
<summary><strong>PaVeRL-SQL: Text-to-SQL via Partial-Match Rewards and Verbal Reinforcement Learning</strong> - Heng Hao, Wenjun Hu, Oxana Verkholyak, Davoud Ataee Tarzanagh, Baruch Gutow, Sima Didari, Masoud Faraki, Hankyu Moon, Seungjai Min - [[pdf]](https://arxiv.org/pdf/2509.07159)</summary>

**Abstract:** Text-to-SQL models allow users to interact with a database more easily by generating executable SQL statements from natural-language questions. Despite recent successes on simpler databases and questions, current Text-to-SQL methods still suffer from low execution accuracy on industry-scale databases and complex questions involving domain-specific business logic. We present \emph{PaVeRL-SQL}, a framework that combines \emph{Partial-Match Rewards} and \emph{Verbal Reinforcement Learning} to drive self-improvement in reasoning language models (RLMs) for Text-to-SQL. To handle practical use cases, we adopt two pipelines: (1) a newly designed in-context learning framework with group self-evaluation (verbal-RL), using capable open- and closed-source large language models (LLMs) as backbones; and (2) a chain-of-thought (CoT) RL pipeline with a small backbone model (OmniSQL-7B) trained with a specially designed reward function and two-stage RL. These pipelines achieve state-of-the-art (SOTA) results on popular Text-to-SQL benchmarks -- Spider, Spider 2.0, and BIRD. For the industrial-level Spider2.0-SQLite benchmark, the verbal-RL pipeline achieves an execution accuracy 7.4\% higher than SOTA, and the CoT pipeline is 1.4\% higher. RL training with mixed SQL dialects yields strong, threefold gains, particularly for dialects with limited training data. Overall, \emph{PaVeRL-SQL} delivers reliable, SOTA Text-to-SQL under realistic industrial constraints. The code is available at this https URL.

**arXiv ID:** 2509.07159
</details>

<details>
<summary><strong>RLFactory: A Plug-and-Play Reinforcement Learning Post-Training Framework for LLM Multi-Turn Tool-Use</strong> - Jiajun Chai, Guojun Yin, Zekun Xu, Chuhuai Yue, Yi Jia, Siyu Xia, Xiaohan Wang, Jiwen Jiang, Xiaoguang Li, Chengqi Dong, Hang He, Wei Lin - [[pdf]](https://arxiv.org/pdf/2509.06980)</summary>

**Abstract:** Large language models excel at basic reasoning but struggle with tasks that require interaction with external tools. We present RLFactory, a plug-and-play reinforcement learning post-training framework for multi-round tool use. RLFactory tackles (i) tool-call stability and adaptability amid tool heterogeneity and interface issues via an asyncio-based asynchronous caller and a decoupled tool/training architecture, and (ii) diverse evaluation needs via a reward layer supporting rule-based, model-judgment, and tool-verification signals. It reconstructs the MDP by introducing observation markers from tool feedback, closing the loop among model, tools, and environment, and implements a generate-parse-invoke-update workflow for dynamic policy optimization. On Search-R1 with Qwen3-4B, RLFactory achieves a 0.486 test score on the Natural Questions (NQ) dataset, surpassing larger models trained with similar techniques (e.g., Qwen2.5-7B-Instruct-GRPO at 0.473), and increases training throughput by 6.8x. RLFactory provides a low-barrier, highly adaptable framework for strengthening multi-round tool use of LLMs in real-world scenarios. Code: this https URL.

**arXiv ID:** 2509.06980
</details>

<details>
<summary><strong>An efficient deep reinforcement learning environment for flexible job-shop scheduling</strong> - Xinquan Wu, Xuefeng Yan, Mingqiang Wei, Donghai Guan - [[pdf]](https://arxiv.org/pdf/2509.07019)</summary>

**Abstract:** The Flexible Job-shop Scheduling Problem (FJSP) is a classical combinatorial optimization problem that has a wide-range of applications in the real world. In order to generate fast and accurate scheduling solutions for FJSP, various deep reinforcement learning (DRL) scheduling methods have been developed. However, these methods are mainly focused on the design of DRL scheduling Agent, overlooking the modeling of DRL environment. This paper presents a simple chronological DRL environment for FJSP based on discrete event simulation and an end-to-end DRL scheduling model is proposed based on the proximal policy optimization (PPO). Furthermore, a short novel state representation of FJSP is proposed based on two state variables in the scheduling environment and a novel comprehensible reward function is designed based on the scheduling area of machines. Experimental results on public benchmark instances show that the performance of simple priority dispatching rules (PDR) is improved in our scheduling environment and our DRL scheduling model obtains competing performance compared with OR-Tools, meta-heuristic, DRL and PDR scheduling methods.

**arXiv ID:** 2509.07019
</details>

<details>
<summary><strong>SoK: Security and Privacy of AI Agents for Blockchain</strong> - Nicolò Romandini, Carlo Mazzocca, Kai Otsuki, Rebecca Montanari - [[pdf]](https://arxiv.org/pdf/2509.07131)</summary>

**Abstract:** Blockchain and smart contracts have garnered significant interest in recent years as the foundation of a decentralized, trustless digital ecosystem, thereby eliminating the need for traditional centralized authorities. Despite their central role in powering Web3, their complexity still presents significant barriers for non-expert users. To bridge this gap, Artificial Intelligence (AI)-based agents have emerged as valuable tools for interacting with blockchain environments, supporting a range of tasks, from analyzing on-chain data and optimizing transaction strategies to detecting vulnerabilities within smart contracts. While interest in applying AI to blockchain is growing, the literature still lacks a comprehensive survey that focuses specifically on the intersection with AI agents. Most of the related work only provides general considerations, without focusing on any specific domain. This paper addresses this gap by presenting the first Systematization of Knowledge dedicated to AI-driven systems for blockchain, with a special focus on their security and privacy dimensions, shedding light on their applications, limitations, and future research directions.

**arXiv ID:** 2509.07131
</details>

<details>
<summary><strong>The Choice of Divergence: A Neglected Key to Mitigating Diversity Collapse in Reinforcement Learning with Verifiable Reward</strong> - Long Li, Jiaran Hao, Jason Klein Liu, Zhijian Zhou, Xiaoyu Tan, Wei Chu, Zhe Wang, Shirui Pan, Chao Qu, Yuan Qi - [[pdf]](https://arxiv.org/pdf/2509.07430)</summary>

**Abstract:** A central paradox in fine-tuning Large Language Models (LLMs) with Reinforcement Learning with Verifiable Reward (RLVR) is the frequent degradation of multi-attempt performance (Pass@k) despite improvements in single-attempt accuracy (Pass@1). This is often accompanied by catastrophic forgetting, where models lose previously acquired skills. While various methods have been proposed, the choice and function of the divergence term have been surprisingly unexamined as a proactive solution. We argue that standard RLVR objectives -- both those using the mode-seeking reverse KL-divergence and those forgoing a divergence term entirely -- lack a crucial mechanism for knowledge retention. The reverse-KL actively accelerates this decay by narrowing the policy, while its absence provides no safeguard against the model drifting from its diverse knowledge base. We propose a fundamental shift in perspective: using the divergence term itself as the solution. Our framework, Diversity-Preserving Hybrid RL (DPH-RL), leverages mass-covering f-divergences (like forward-KL and JS-divergence) to function as a rehearsal mechanism. By continuously referencing the initial policy, this approach forces the model to maintain broad solution coverage. Extensive experiments on math and SQL generation demonstrate that DPH-RL not only resolves the Pass@k degradation but improves both Pass@1 and Pass@k in- and out-of-domain. Additionally, DPH-RL is more training-efficient because it computes f-divergence using generator functions, requiring only sampling from the initial policy and no online reference model. Our work highlights a crucial, overlooked axis for improving RLVR, demonstrating that the proper selection of a divergence measure is a powerful tool for building more general and diverse reasoning models.

**arXiv ID:** 2509.07430
</details>

<details>
<summary><strong>Can SSD-Mamba2 Unlock Reinforcement Learning for End-to-End Motion Control?</strong> - Gavin Tao, Yinuo Wang, Jinzhao Zhou - [[pdf]](https://arxiv.org/pdf/2509.07593)</summary>

**Abstract:** End-to-end reinforcement learning for motion control promises unified perception-action policies that scale across embodiments and tasks, yet most deployed controllers are either blind (proprioception-only) or rely on fusion backbones with unfavorable compute-memory trade-offs. Recurrent controllers struggle with long-horizon credit assignment, and Transformer-based fusion incurs quadratic cost in token length, limiting temporal and spatial context. We present a vision-driven cross-modal RL framework built on SSD-Mamba2, a selective state-space backbone that applies state-space duality (SSD) to enable both recurrent and convolutional scanning with hardware-aware streaming and near-linear scaling. Proprioceptive states and exteroceptive observations (e.g., depth tokens) are encoded into compact tokens and fused by stacked SSD-Mamba2 layers. The selective state-space updates retain long-range dependencies with markedly lower latency and memory use than quadratic self-attention, enabling longer look-ahead, higher token resolution, and stable training under limited compute. Policies are trained end-to-end under curricula that randomize terrain and appearance and progressively increase scene complexity. A compact, state-centric reward balances task progress, energy efficiency, and safety. Across diverse motion-control scenarios, our approach consistently surpasses strong state-of-the-art baselines in return, safety (collisions and falls), and sample efficiency, while converging faster at the same compute budget. These results suggest that SSD-Mamba2 provides a practical fusion backbone for scalable, foresightful, and efficient end-to-end motion control.

**arXiv ID:** 2509.07593
</details>

<details>
<summary><strong>AI-SearchPlanner: Modular Agentic Search via Pareto-Optimal Multi-Objective Reinforcement Learning</strong> - Lang Mei, Zhihan Yang, Chong Chen - [[pdf]](https://arxiv.org/pdf/2508.20368)</summary>

**Abstract:** Recent studies have explored integrating Large Language Models (LLMs) with search engines to leverage both the LLMs' internal pre-trained knowledge and external information. Specially, reinforcement learning (RL) has emerged as a promising paradigm for enhancing LLM reasoning through multi-turn interactions with search engines. However, existing RL-based search agents rely on a single LLM to handle both search planning and question-answering (QA) tasks in an end-to-end manner, which limits their ability to optimize both capabilities simultaneously. In practice, sophisticated AI search systems often employ a large, frozen LLM (e.g., GPT-4, DeepSeek-R1) to ensure high-quality QA. Thus, a more effective and efficient approach is to utilize a small, trainable LLM dedicated to search planning. In this paper, we propose \textbf{AI-SearchPlanner}, a novel reinforcement learning framework designed to enhance the performance of frozen QA models by focusing on search planning. Specifically, our approach introduces three key innovations: 1) Decoupling the Architecture of the Search Planner and Generator, 2) Dual-Reward Alignment for Search Planning, and 3) Pareto Optimization of Planning Utility and Cost, to achieve the objectives. Extensive experiments on real-world datasets demonstrate that AI SearchPlanner outperforms existing RL-based search agents in both effectiveness and efficiency, while exhibiting strong generalization capabilities across diverse frozen QA models and data domains.

**arXiv ID:** 2508.20368
</details>

<details>
<summary><strong>Solving Truly Massive Budgeted Monotonic POMDPs with Oracle-Guided Meta-Reinforcement Learning</strong> - Manav Vora, Jonas Liang, Melkior Ornik - [[pdf]](https://arxiv.org/pdf/2408.07192)</summary>

**Abstract:** Monotonic Partially Observable Markov Decision Processes (POMDPs), where the system state progressively decreases until a restorative action is performed, can be used to model sequential repair problems effectively. This paper considers the problem of solving budget-constrained multi-component monotonic POMDPs, where a finite budget limits the maximal number of restorative actions. For a large number of components, solving such a POMDP using current methods is computationally intractable due to the exponential growth in the state space with an increasing number of components. To address this challenge, we propose a two-step approach. Since the individual components of a budget-constrained multi-component monotonic POMDP are only connected via the shared budget, we first approximate the optimal budget allocation among these components using an approximation of each component POMDP's optimal value function which is obtained through a random forest model. Subsequently, we introduce an oracle-guided meta-trained Proximal Policy Optimization (PPO) algorithm to solve each of the independent budget-constrained single-component monotonic POMDPs. The oracle policy is obtained by performing value iteration on the corresponding monotonic Markov Decision Process (MDP). This two-step method provides scalability in solving truly massive multi-component monotonic POMDPs. To demonstrate the efficacy of our approach, we consider a real-world maintenance scenario that involves inspection and repair of an administrative building by a team of agents within a maintenance budget. Finally, we perform a computational complexity analysis for a varying number of components to show the scalability of the proposed approach.

**arXiv ID:** 2408.07192
</details>

<details>
<summary><strong>Understanding Behavioral Metric Learning: A Large-Scale Study on Distracting Reinforcement Learning Environments</strong> - Ziyan Luo, Tianwei Ni, Pierre-Luc Bacon, Doina Precup, Xujie Si - [[pdf]](https://arxiv.org/pdf/2506.00563)</summary>

**Abstract:** A key approach to state abstraction is approximating behavioral metrics (notably, bisimulation metrics) in the observation space and embedding these learned distances in the representation space. While promising for robustness to task-irrelevant noise, as shown in prior work, accurately estimating these metrics remains challenging, requiring various design choices that create gaps between theory and practice. Prior evaluations focus mainly on final returns, leaving the quality of learned metrics and the source of performance gains unclear. To systematically assess how metric learning works in deep reinforcement learning (RL), we evaluate five recent approaches, unified conceptually as isometric embeddings with varying design choices. We benchmark them with baselines across 20 state-based and 14 pixel-based tasks, spanning 370 task configurations with diverse noise settings. Beyond final returns, we introduce the evaluation of a denoising factor to quantify the encoder's ability to filter distractions. To further isolate the effect of metric learning, we propose and evaluate an isolated metric estimation setting, in which the encoder is influenced solely by the metric loss. Finally, we release an open-source, modular codebase to improve reproducibility and support future research on metric learning in deep RL.

**arXiv ID:** 2506.00563
</details>

<details>
<summary><strong>Toward a Metrology for Artificial Intelligence: Hidden-Rule Environments and Reinforcement Learning</strong> - Christo Mathew, Wentian Wang, Jacob Feldman, Lazaros K. Gallos, Paul B. Kantor, Vladimir Menkov, Hao Wang - [[pdf]](https://arxiv.org/pdf/2509.06213)</summary>

**Abstract:** We investigate reinforcement learning in the Game Of Hidden Rules (GOHR) environment, a complex puzzle in which an agent must infer and execute hidden rules to clear a 6$\times$6 board by placing game pieces into buckets. We explore two state representation strategies, namely Feature-Centric (FC) and Object-Centric (OC), and employ a Transformer-based Advantage Actor-Critic (A2C) algorithm for training. The agent has access only to partial observations and must simultaneously infer the governing rule and learn the optimal policy through experience. We evaluate our models across multiple rule-based and trial-list-based experimental setups, analyzing transfer effects and the impact of representation on learning efficiency.

**arXiv ID:** 2509.06213
</details>

<details>
<summary><strong>Parallel-R1: Towards Parallel Thinking via Reinforcement Learning</strong> - Tong Zheng, Hongming Zhang, Wenhao Yu, Xiaoyang Wang, Xinyu Yang, Runpeng Dai, Rui Liu, Huiwen Bao, Chengsong Huang, Heng Huang, Dong Yu - [[pdf]](https://arxiv.org/pdf/2509.07980)</summary>

**Abstract:** Parallel thinking has emerged as a novel approach for enhancing the reasoning capabilities of large language models (LLMs) by exploring multiple reasoning paths concurrently. However, activating such capabilities through training remains challenging, as existing methods predominantly rely on supervised fine-tuning (SFT) over synthetic data, which encourages teacher-forced imitation rather than exploration and generalization. Different from them, we propose \textbf{Parallel-R1}, the first reinforcement learning (RL) framework that enables parallel thinking behaviors for complex real-world reasoning tasks. Our framework employs a progressive curriculum that explicitly addresses the cold-start problem in training parallel thinking with RL. We first use SFT on prompt-generated trajectories from easier tasks to instill the parallel thinking ability, then transition to RL to explore and generalize this skill on harder problems. Experiments on various math benchmarks, including MATH, AMC23, and AIME, show that Parallel-R1 successfully instills parallel thinking, leading to 8.4% accuracy improvements over the sequential thinking model trained directly on challenging tasks with RL. Further analysis reveals a clear shift in the model's thinking behavior: at an early stage, it uses parallel thinking as an exploration strategy, while in a later stage, it uses the same capability for multi-perspective verification. Most significantly, we validate parallel thinking as a \textbf{mid-training exploration scaffold}, where this temporary exploratory phase unlocks a higher performance ceiling after RL, yielding a 42.9% improvement over the baseline on AIME25. Our model, data, and code will be open-source at this https URL.

**arXiv ID:** 2509.07980
</details>

<details>
<summary><strong>Advancing SLM Tool-Use Capability using Reinforcement Learning</strong> - Dhruvi Paprunia, Vansh Kharidia, Pankti Doshi - [[pdf]](https://arxiv.org/pdf/2509.04518)</summary>

**Abstract:** In an era where tool-augmented AI agents are becoming increasingly vital, our findings highlight the ability of Group Relative Policy Optimization (GRPO) to empower SLMs, which are traditionally constrained in tool use. The ability to use tools effectively has become a defining feature of Large Language Models (LLMs), allowing them to access external data and internal resources. As AI agents grow more sophisticated, tool-use capabilities have become indispensable. While LLMs have made significant progress in this area, Small Language Models (SLMs) still face challenges in accurately integrating tool use, especially in resource-constrained settings.
This study investigates how Reinforcement Learning, specifically Group Relative Policy Optimization (GRPO), can enhance the tool-use accuracy of SLMs. By designing a well-defined reward system that reinforces structured JSON output, correct tool selection, and precise parameter usage, we demonstrate that GRPO enables SLMs to achieve significant improvements in tool-use capabilities (function calling/JSON output). Our approach provides a computationally efficient training method that enhances SLMs practical deployment in real-world AI applications.

**arXiv ID:** 2509.04518
</details>

<details>
<summary><strong>Reinforcement learning for online hyperparameter tuning in convex quadratic programming</strong> - Jeremy Bertoncini, Alberto De Marchi, Matthias Gerdts, Simon Gottschalk - [[pdf]](https://arxiv.org/pdf/2509.07404)</summary>

**Abstract:** Quadratic programming is a workhorse of modern nonlinear optimization, control, and data science. Although regularized methods offer convergence guarantees under minimal assumptions on the problem data, they can exhibit the slow tail-convergence typical of first-order schemes, thus requiring many iterations to achieve high-accuracy solutions. Moreover, hyperparameter tuning significantly impacts on the solver performance but how to find an appropriate parameter configuration remains an elusive research question. To address these issues, we explore how data-driven approaches can accelerate the solution process. Aiming at high-accuracy solutions, we focus on a stabilized interior-point solver and carefully handle its two-loop flow and control parameters. We will show that reinforcement learning can make a significant contribution to facilitating the solver tuning and to speeding up the optimization process. Numerical experiments demonstrate that, after a lightweight training, the learned policy generalizes well to different problem classes with varying dimensions and to various solver configurations.

**arXiv ID:** 2509.07404
</details>

<details>
<summary><strong>GCN-Driven Reinforcement Learning for Probabilistic Real-Time Guarantees in Industrial URLLC</strong> - Eman Alqudah, Ashfaq Khokhar - [[pdf]](https://arxiv.org/pdf/2506.15011)</summary>

**Abstract:** Ensuring packet-level communication quality is vital for ultra-reliable, low-latency communications (URLLC) in large-scale industrial wireless networks. We enhance the Local Deadline Partition (LDP) algorithm by introducing a Graph Convolutional Network (GCN) integrated with a Deep Q-Network (DQN) reinforcement learning framework for improved interference coordination in multi-cell, multi-channel networks. Unlike LDP's static priorities, our approach dynamically learns link priorities based on real-time traffic demand, network topology, remaining transmission opportunities, and interference patterns. The GCN captures spatial dependencies, while the DQN enables adaptive scheduling decisions through reward-guided exploration. Simulation results show that our GCN-DQN model achieves mean SINR improvements of 179.6\%, 197.4\%, and 175.2\% over LDP across three network configurations. Additionally, the GCN-DQN model demonstrates mean SINR improvements of 31.5\%, 53.0\%, and 84.7\% over our previous CNN-based approach across the same configurations. These results underscore the effectiveness of our GCN-DQN model in addressing complex URLLC requirements with minimal overhead and superior network performance.

**arXiv ID:** 2506.15011
</details>

<details>
<summary><strong>Attention and Risk-Aware Decision Framework for Safe Autonomous Driving</strong> - Zhen Tian, Fujiang Yuan, Yangfan He, Qinghao Li, Changlin Chen, Huilin Chen, Tianxiang Xu, Jianyu Duan, Yanhong Peng, Zhihao Lin - [[pdf]](https://arxiv.org/pdf/2509.07412)</summary>

**Abstract:** Autonomous driving has attracted great interest due to its potential capability in full-unsupervised driving. Model-based and learning-based methods are widely used in autonomous driving. Model-based methods rely on pre-defined models of the environment and may struggle with unforeseen events. Proximal policy optimization (PPO), an advanced learning-based method, can adapt to the above limits by learning from interactions with the environment. However, existing PPO faces challenges with poor training results, and low training efficiency in long sequences. Moreover, the poor training results are equivalent to collisions in driving tasks. To solve these issues, this paper develops an improved PPO by introducing the risk-aware mechanism, a risk-attention decision network, a balanced reward function, and a safety-assisted mechanism. The risk-aware mechanism focuses on highlighting areas with potential collisions, facilitating safe-driving learning of the PPO. The balanced reward function adjusts rewards based on the number of surrounding vehicles, promoting efficient exploration of the control strategy during training. Additionally, the risk-attention network enhances the PPO to hold channel and spatial attention for the high-risk areas of input images. Moreover, the safety-assisted mechanism supervises and prevents the actions with risks of collisions during the lane keeping and lane changing. Simulation results on a physical engine demonstrate that the proposed algorithm outperforms benchmark algorithms in collision avoidance, achieving higher peak reward with less training time, and shorter driving time remaining on the risky areas among multiple testing traffic flow scenarios.

**arXiv ID:** 2509.07412
</details>

<details>
<summary><strong>Fault Tolerant Control of a Quadcopter using Reinforcement Learning</strong> - Muzaffar Habib, Adnan Maqsood, Adnan Fayyaz ud Din - [[pdf]](https://arxiv.org/pdf/2509.07707)</summary>

**Abstract:** This study presents a novel reinforcement learning (RL)-based control framework aimed at enhancing the safety and robustness of the quadcopter, with a specific focus on resilience to in-flight one propeller failure. Addressing the critical need of a robust control strategy for maintaining a desired altitude for the quadcopter to safe the hardware and the payload in physical applications. The proposed framework investigates two RL methodologies Dynamic Programming (DP) and Deep Deterministic Policy Gradient (DDPG), to overcome the challenges posed by the rotor failure mechanism of the quadcopter. DP, a model-based approach, is leveraged for its convergence guarantees, despite high computational demands, whereas DDPG, a model-free technique, facilitates rapid computation but with constraints on solution duration. The research challenge arises from training RL algorithms on large dimensions and action domains. With modifications to the existing DP and DDPG algorithms, the controllers were trained not only to cater for large continuous state and action domain and also achieve a desired state after an inflight propeller failure. To verify the robustness of the proposed control framework, extensive simulations were conducted in a MATLAB environment across various initial conditions and underscoring its viability for mission-critical quadcopter applications. A comparative analysis was performed between both RL algorithms and their potential for applications in faulty aerial systems.

**arXiv ID:** 2509.07707
</details>

<details>
<summary><strong>Interactive Shaping of Granular Media Using Reinforcement Learning</strong> - Benedikt Kreis, Malte Mosbach, Anny Ripke, Muhammad Ehsan Ullah, Sven Behnke, Maren Bennewitz - [[pdf]](https://arxiv.org/pdf/2509.06469)</summary>

**Abstract:** Autonomous manipulation of granular media, such as sand, is crucial for applications in construction, excavation, and additive manufacturing. However, shaping granular materials presents unique challenges due to their high-dimensional configuration space and complex dynamics, where traditional rule-based approaches struggle without extensive engineering efforts. Reinforcement learning (RL) offers a promising alternative by enabling agents to learn adaptive manipulation strategies through trial and error. In this work, we present an RL framework that enables a robotic arm with a cubic end-effector and a stereo camera to shape granular media into desired target structures. We show the importance of compact observations and concise reward formulations for the large configuration space, validating our design choices with an ablation study. Our results demonstrate the effectiveness of the proposed approach for the training of visual policies that manipulate granular media including their real-world deployment, significantly outperforming two baseline approaches in terms of target shape accuracy.

**arXiv ID:** 2509.06469
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
