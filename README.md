# Agent arXiv Daily

**Last Updated:** 2025-10-20 02:17:15

**Total Papers:** 68

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
<summary><strong>HugAgent: Evaluating LLMs in Simulating Human-Like Individual Reasoning on Open-Ended Tasks</strong> - Chance Jiajie Li, Zhenze Mo, Yuhan Tang, Ao Qu, Jiayi Wu, Kaiya Ivy Zhao, Yulu Gan, Jie Fan, Jiangbo Yu, Hang Jiang, Paul Pu Liang, Jinhua Zhao, Luis Alberto Alonso Pastor, Kent Larson - [[pdf]](https://arxiv.org/pdf/2510.15144)</summary>

**Abstract:** Simulating human reasoning in open-ended tasks has been a long-standing aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), a benchmark for average-to-individual reasoning adaptation. The task is to predict how a specific person would reason and update their beliefs in novel scenarios, given partial evidence of their past views. HugAgent adopts a dual-track design: a synthetic track for scale and systematic stress tests, and a human track for ecologically valid, "out-loud" reasoning data. This design enables scalable, reproducible evaluation of intra-agent fidelity: whether models can capture not just what people believe, but how their reasoning evolves. Experiments with state-of-the-art LLMs reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. Our benchmark and chatbot are open-sourced as HugAgent (this https URL) and TraceYourThinking (this https URL).

**arXiv ID:** 2510.15144
</details>

<details>
<summary><strong>AURA: An Agent Autonomy Risk Assessment Framework</strong> - Lorenzo Satta Chiris, Ayush Mishra - [[pdf]](https://arxiv.org/pdf/2510.15739)</summary>

**Abstract:** As autonomous agentic AI systems see increasing adoption across organisations, persistent challenges in alignment, governance, and risk management threaten to impede deployment at scale. We present AURA (Agent aUtonomy Risk Assessment), a unified framework designed to detect, quantify, and mitigate risks arising from agentic AI. Building on recent research and practical deployments, AURA introduces a gamma-based risk scoring methodology that balances risk assessment accuracy with computational efficiency and practical considerations. AURA provides an interactive process to score, evaluate and mitigate the risks of running one or multiple AI Agents, synchronously or asynchronously (autonomously). The framework is engineered for Human-in-the-Loop (HITL) oversight and presents Agent-to-Human (A2H) communication mechanisms, allowing for seamless integration with agentic systems for autonomous self-assessment, rendering it interoperable with established protocols (MCP and A2A) and tools. AURA supports a responsible and transparent adoption of agentic AI and provides robust risk detection and mitigation while balancing computational resources, positioning it as a critical enabler for large-scale, governable agentic AI in enterprise environments.

**arXiv ID:** 2510.15739
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (7 papers)</h2></summary>

<details>
<summary><strong>AUGUSTUS: An LLM-Driven Multimodal Agent System with Contextualized User Memory</strong> - Jitesh Jain, Shubham Maheshwari, Ning Yu, Wen-mei Hwu, Humphrey Shi - [[pdf]](https://arxiv.org/pdf/2510.15261)</summary>

**Abstract:** Riding on the success of LLMs with retrieval-augmented generation (RAG), there has been a growing interest in augmenting agent systems with external memory databases. However, the existing systems focus on storing text information in their memory, ignoring the importance of multimodal signals. Motivated by the multimodal nature of human memory, we present AUGUSTUS, a multimodal agent system aligned with the ideas of human memory in cognitive science. Technically, our system consists of 4 stages connected in a loop: (i) encode: understanding the inputs; (ii) store in memory: saving important information; (iii) retrieve: searching for relevant context from memory; and (iv) act: perform the task. Unlike existing systems that use vector databases, we propose conceptualizing information into semantic tags and associating the tags with their context to store them in a graph-structured multimodal contextual memory for efficient concept-driven retrieval. Our system outperforms the traditional multimodal RAG approach while being 3.5 times faster for ImageNet classification and outperforming MemGPT on the MSC benchmark.

**arXiv ID:** 2510.15261
</details>

<details>
<summary><strong>Beyond Final Code: A Process-Oriented Error Analysis of Software Development Agents in Real-World GitHub Scenarios</strong> - Zhi Chen, Wei Ma, Lingxiao Jiang - [[pdf]](https://arxiv.org/pdf/2503.12374)</summary>

**Abstract:** AI-driven software development has rapidly advanced with the emergence of software development agents that leverage large language models (LLMs) to tackle complex, repository-level software engineering tasks. These agents go beyond just generation of final code; they engage in multi-step reasoning, utilize various tools for code modification and debugging, and interact with execution environments to diagnose and iteratively resolve issues. However, most existing evaluations focus primarily on static analyses of final code outputs, yielding limited insights into the agents' dynamic problem-solving processes. To fill this gap, we conduct an in-depth empirical study on 3,977 solving-phase trajectories and 3,931 testing-phase logs from 8 top-ranked agents evaluated on 500 GitHub issues in the SWE-Bench benchmark.
Our exploratory analysis shows that Python execution errors during the issue resolution phase correlate with lower resolution rates and increased reasoning overheads. We have identified the most prevalent errors -- such as ModuleNotFoundError and TypeError -- and highlighted particularly challenging errors like OSError and database-related issues (e.g., IntegrityError) that demand significantly more debugging effort. Furthermore, we have discovered 3 bugs in the SWE-Bench platform that affect benchmark fairness and accuracy; these issues have been reported to and confirmed by the maintainers. To promote transparency and foster future research, we publicly share our datasets and analysis scripts.

**arXiv ID:** 2503.12374
</details>

<details>
<summary><strong>WebInject: Prompt Injection Attack to Web Agents</strong> - Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong - [[pdf]](https://arxiv.org/pdf/2505.11717)</summary>

**Abstract:** Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.

**arXiv ID:** 2505.11717
</details>

<details>
<summary><strong>TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving</strong> - Yiming Yang, Yueru Luo, Bingkun He, Hongbin Lin, Suzhong Fu, Chao Zheng, Zhipeng Cao, Erlong Li, Chao Yan, Shuguang Cui, Zhen Li - [[pdf]](https://arxiv.org/pdf/2507.00709)</summary>

**Abstract:** Lane segment topology reasoning constructs a comprehensive road network by capturing the topological relationships between lane segments and their semantic types. This enables end-to-end autonomous driving systems to perform road-dependent maneuvers such as turning and lane changing. However, the limitations in consistent positional embedding and temporal multiple attribute learning in existing methods hinder accurate roadnet reconstruction. To address these issues, we propose TopoStreamer, an end-to-end temporal perception model for lane segment topology reasoning. Specifically, TopoStreamer introduces three key improvements: streaming attribute constraints, dynamic lane boundary positional encoding, and lane segment denoising. The streaming attribute constraints enforce temporal consistency in both centerline and boundary coordinates, along with their classifications. Meanwhile, dynamic lane boundary positional encoding enhances the learning of up-to-date positional information within queries, while lane segment denoising helps capture diverse lane segment patterns, ultimately improving model performance. Additionally, we assess the accuracy of existing models using a lane boundary classification metric, which serves as a crucial measure for lane-changing scenarios in autonomous driving. On the OpenLane-V2 dataset, TopoStreamer demonstrates significant improvements over state-of-the-art methods, achieving substantial performance gains of +3.0% mAP in lane segment perception and +1.7% OLS in centerline perception tasks.

**arXiv ID:** 2507.00709
</details>

<details>
<summary><strong>Perfect Prediction or Plenty of Proposals? What Matters Most in Planning for Autonomous Driving</strong> - Aron Distelzweig, Faris Janjoš, Oliver Scheel, Sirish Reddy Varra, Raghu Rajan, Joschka Boedecker - [[pdf]](https://arxiv.org/pdf/2510.15505)</summary>

**Abstract:** Traditionally, prediction and planning in autonomous driving (AD) have been treated as separate, sequential modules. Recently, there has been a growing shift towards tighter integration of these components, known as Integrated Prediction and Planning (IPP), with the aim of enabling more informed and adaptive decision-making. However, it remains unclear to what extent this integration actually improves planning performance. In this work, we investigate the role of prediction in IPP approaches, drawing on the widely adopted Val14 benchmark, which encompasses more common driving scenarios with relatively low interaction complexity, and the interPlan benchmark, which includes highly interactive and out-of-distribution driving situations. Our analysis reveals that even access to perfect future predictions does not lead to better planning outcomes, indicating that current IPP methods often fail to fully exploit future behavior information. Instead, we focus on high-quality proposal generation, while using predictions primarily for collision checks. We find that many imitation learning-based planners struggle to generate realistic and plausible proposals, performing worse than PDM - a simple lane-following approach. Motivated by this observation, we build on PDM with an enhanced proposal generation method, shifting the emphasis towards producing diverse but realistic and high-quality proposals. This proposal-centric approach significantly outperforms existing methods, especially in out-of-distribution and highly interactive settings, where it sets new state-of-the-art results.

**arXiv ID:** 2510.15505
</details>

<details>
<summary><strong>TAS: A Transit-Aware Strategy for Embodied Navigation with Non-Stationary Targets</strong> - Vishnu Sashank Dorbala, Bhrij Patel, Amrit Singh Bedi, Dinesh Manocha - [[pdf]](https://arxiv.org/pdf/2403.09905)</summary>

**Abstract:** Embodied navigation methods commonly operate in static environments with stationary targets. In this work, we present a new algorithm for navigation in dynamic scenarios with non-stationary targets. Our novel Transit-Aware Strategy (TAS) enriches embodied navigation policies with object path information. TAS improves performance in non-stationary environments by rewarding agents for synchronizing their routes with target routes. To evaluate TAS, we further introduce Dynamic Object Maps (DOMs), a dynamic variant of node-attributed topological graphs with structured object transitions. DOMs are inspired by human habits to simulate realistic object routes on a graph. Our experiments show that on average, TAS improves agent Success Rate (SR) by 21.1 in non-stationary environments, while also generalizing better from static environments by 44.5% when measured by Relative Change in Success (RCS). We qualitatively investigate TAS-agent performance on DOMs and draw various inferences to help better model generalist navigation policies. To the best of our knowledge, ours is the first work that quantifies the adaptability of embodied navigation methods in non-stationary environments. Code and data for our benchmark will be made publicly available.

**arXiv ID:** 2403.09905
</details>

<details>
<summary><strong>Self-supervised Multi-future Occupancy Forecasting for Autonomous Driving</strong> - Bernard Lange, Masha Itkina, Jiachen Li, Mykel J. Kochenderfer - [[pdf]](https://arxiv.org/pdf/2407.21126)</summary>

**Abstract:** Environment prediction frameworks are critical for the safe navigation of autonomous vehicles (AVs) in dynamic settings. LiDAR-generated occupancy grid maps (L-OGMs) offer a robust bird's-eye view for the scene representation, enabling self-supervised joint scene predictions while exhibiting resilience to partial observability and perception detection failures. Prior approaches have focused on deterministic L-OGM prediction architectures within the grid cell space. While these methods have seen some success, they frequently produce unrealistic predictions and fail to capture the stochastic nature of the environment. Additionally, they do not effectively integrate additional sensor modalities present in AVs. Our proposed framework, Latent Occupancy Prediction (LOPR), performs stochastic L-OGM prediction in the latent space of a generative architecture and allows for conditioning on RGB cameras, maps, and planned trajectories. We decode predictions using either a single-step decoder, which provides high-quality predictions in real-time, or a diffusion-based batch decoder, which can further refine the decoded frames to address temporal consistency issues and reduce compression losses. Our experiments on the nuScenes and Waymo Open datasets show that all variants of our approach qualitatively and quantitatively outperform prior approaches.

**arXiv ID:** 2407.21126
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>Multi-dimensional Data Analysis and Applications Basing on LLM Agents and Knowledge Graph Interactions</strong> - Xi Wang, Xianyao Ling, Kun Li, Gang Yin, Liang Zhang, Jiang Wu, Jun Xu, Fu Zhang, Wenbo Lei, Annie Wang, Peng Gong - [[pdf]](https://arxiv.org/pdf/2510.15258)</summary>

**Abstract:** In the current era of big data, extracting deep insights from massive, heterogeneous, and complexly associated multi-dimensional data has become a significant challenge. Large Language Models (LLMs) perform well in natural language understanding and generation, but still suffer from "hallucination" issues when processing structured knowledge and are difficult to update in real-time. Although Knowledge Graphs (KGs) can explicitly store structured knowledge, their static nature limits dynamic interaction and analytical capabilities. Therefore, this paper proposes a multi-dimensional data analysis method based on the interactions between LLM agents and KGs, constructing a dynamic, collaborative analytical ecosystem. This method utilizes LLM agents to automatically extract product data from unstructured data, constructs and visualizes the KG in real-time, and supports users in deep exploration and analysis of graph nodes through an interactive platform. Experimental results show that this method has significant advantages in product ecosystem analysis, relationship mining, and user-driven exploratory analysis, providing new ideas and tools for multi-dimensional data analysis.

**arXiv ID:** 2510.15258
</details>

<details>
<summary><strong>Exemplar-Guided Planing: Enhanced LLM Agent for KGQA</strong> - Jingao Xu, Shuoyoucheng Ma, Xin Song, Rong Jiang, Hongkui Tu, Bin Zhou - [[pdf]](https://arxiv.org/pdf/2510.15283)</summary>

**Abstract:** Large Language Models (LLMs) as interactive agents show significant promise in Knowledge Graph Question Answering (KGQA) but often struggle with the semantic gap between natural language queries and structured knowledge graph (KG) representations. This leads to suboptimal planning and inefficient exploration on KG, while training-free approaches often underutilize valuable reasoning patterns in training data. To address these limitations, we propose a novel framework, Exemplar-Guided Planning (EGP), which enhances the planning capabilities of LLM agents for KGQA. EGP first preprocesses the training set questions via entity templating to normalize semantic variations. It then retrieves highly similar exemplary questions and their successful reasoning paths from this preprocessed set using semantic embeddings and an efficient FAISS index. These retrieved exemplars dynamically guide the LLM's planning process in two key phases: (1) Task Decomposition, by aligning generated sub-objectives with proven reasoning steps, and (2) Relation Exploration, by providing high-quality auxiliary information to improve relation pruning accuracy. Additionally, we introduce a Smart Lookahead mechanism during relation exploration to improve efficiency by preemptively exploring promising paths and potentially terminating exploration earlier. We apply EGP to the Plan-on-Graph (PoG) framework, termed PoG-EGP. Extensive experiments on two real-world KGQA datasets, WebQSP and CWQ, demonstrate that PoG-EGP significantly improves over the baseline PoG system and other compared methods.

**arXiv ID:** 2510.15283
</details>

<details>
<summary><strong>ACON: Optimizing Context Compression for Long-horizon LLM Agents</strong> - Minki Kang, Wei-Ning Chen, Dongge Han, Huseyin A. Inan, Lukas Wutschitz, Yanzhi Chen, Robert Sim, Saravan Rajmohan - [[pdf]](https://arxiv.org/pdf/2510.00615)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations. This expansion raises costs and reduces efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations. ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly. Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module. Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement. Our code is available at this https URL.

**arXiv ID:** 2510.00615
</details>

<details>
<summary><strong>Where to Search: Measure the Prior-Structured Search Space of LLM Agents</strong> - Zhuo-Yang Song - [[pdf]](https://arxiv.org/pdf/2510.14846)</summary>

**Abstract:** The generate-filter-refine (iterative) paradigm based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs.

**arXiv ID:** 2510.14846
</details>

<details>
<summary><strong>VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications</strong> - Wei He, Yueqing Sun, Hongyan Hao, Xueyuan Hao, Zhikang Xia, Qi Gu, Chengcheng Han, Dengchang Zhao, Hui Su, Kefeng Zhang, Man Gao, Xi Su, Xiaodong Cai, Xunliang Cai, Yu Yang, Yunke Zhao - [[pdf]](https://arxiv.org/pdf/2509.26490)</summary>

**Abstract:** As LLM-based agents are increasingly deployed in real-life scenarios, existing benchmarks fail to capture their inherent complexity of handling extensive information, leveraging diverse resources, and managing dynamic user interactions. To address this gap, we introduce VitaBench, a challenging benchmark that evaluates agents on versatile interactive tasks grounded in real-world settings. Drawing from daily applications in food delivery, in-store consumption, and online travel services, VitaBench presents agents with the most complex life-serving simulation environment to date, comprising 66 tools. Through a framework that eliminates domain-specific policies, we enable flexible composition of these scenarios and tools, yielding 100 cross-scenario tasks (main results) and 300 single-scenario tasks. Each task is derived from multiple real user requests and requires agents to reason across temporal and spatial dimensions, utilize complex tool sets, proactively clarify ambiguous instructions, and track shifting user intent throughout multi-turn conversations. Moreover, we propose a rubric-based sliding window evaluator, enabling robust assessment of diverse solution pathways in complex environments and stochastic interactions. Our comprehensive evaluation reveals that even the most advanced models achieve only 30% success rate on cross-scenario tasks, and less than 50% success rate on others. Overall, we believe VitaBench will serve as a valuable resource for advancing the development of AI agents in practical real-world applications. The code, dataset, and leaderboard are available at this https URL

**arXiv ID:** 2509.26490
</details>

<details>
<summary><strong>Internalizing World Models via Self-Play Finetuning for Agentic RL</strong> - Shiqi Chen, Tongyao Zhu, Zian Wang, Jinghan Zhang, Kangrui Wang, Siyang Gao, Teng Xiao, Yee Whye Teh, Junxian He, Manling Li - [[pdf]](https://arxiv.org/pdf/2510.15047)</summary>

**Abstract:** Large Language Models (LLMs) as agents often struggle in out-of-distribution (OOD) scenarios. Real-world environments are complex and dynamic, governed by task-specific rules and stochasticity, which makes it difficult for LLMs to ground their internal knowledge in those dynamics. Under such OOD conditions, vanilla RL training often fails to scale; we observe Pass@k--the probability that at least one of (k) sampled trajectories succeeds--drops markedly across training steps, indicating brittle exploration and limited generalization. Inspired by model-based reinforcement learning, we hypothesize that equipping LLM agents with an internal world model can better align reasoning with environmental dynamics and improve decision-making. We show how to encode this world model by decomposing it into two components: state representation and transition modeling. Building on this, we introduce SPA, a simple reinforcement learning framework that cold-starts the policy via a Self-Play supervised finetuning (SFT) stage to learn the world model by interacting with the environment, then uses it to simulate future states prior to policy optimization. This simple initialization outperforms the online world-modeling baseline and greatly boosts the RL-based agent training performance. Experiments across diverse environments like Sokoban, FrozenLake, and Sudoku show that our approach significantly improves performance. For example, SPA boosts the Sokoban success rate from 25.6% to 59.8% and raises the FrozenLake score from 22.1% to 70.9% for the Qwen2.5-1.5B-Instruct model.

**arXiv ID:** 2510.15047
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (15 papers)</h2></summary>

<details>
<summary><strong>MARS: Reinforcing Multi-Agent Reasoning of LLMs through Self-Play in Strategic Games</strong> - Huining Yuan, Zelai Xu, Zheyue Tan, Xiangmin Yi, Mo Guang, Kaiwen Long, Haojia Hui, Boxun Li, Xinlei Chen, Bo Zhao, Xiao-Ping Zhang, Chao Yu, Yu Wang - [[pdf]](https://arxiv.org/pdf/2510.15414)</summary>

**Abstract:** Developing Large Language Models (LLMs) to cooperate and compete effectively within multi-agent systems is a critical step towards more advanced intelligence. While reinforcement learning (RL) has proven effective for enhancing reasoning in single-agent tasks, its extension to multi-turn, multi-agent scenarios remains underexplored due to the challenges of long-horizon credit assignment and agent-specific advantage estimation. To address these challenges, we introduce MARS, an end-to-end RL framework that incentivizes Multi-Agent Reasoning of LLMs through Self-play in both cooperative and competitive games. MARS features a turn-level advantage estimator that aligns learning signals with each interaction for credit assignment, and an agent-specific advantage normalization to stabilize multi-agent training. By learning with self-play across cooperative and competitive games, the MARS agent trained from Qwen3-4B develops strong strategic abilities that generalize to held-out games with up to 28.7% performance improvements. More importantly, the capability acquired through self-play generalizes beyond games, yielding consistent performance gains of multi-agent systems in reasoning benchmarks. When integrated into leading multi-agent systems, our MARS agent achieves significant performance gains of 10.0% on AIME and 12.5% on GPQA-Diamond. These results establish end-to-end RL training with self-play in strategic games as a powerful approach for developing generalizable multi-agent reasoning capabilities in LLMs. Our code and models are publicly available at this https URL.

**arXiv ID:** 2510.15414
</details>

<details>
<summary><strong>Adaptive Minds: Empowering Agents with LoRA-as-Tools</strong> - Pavan C Shekar, Ashwanth Krishnan - [[pdf]](https://arxiv.org/pdf/2510.15416)</summary>

**Abstract:** We present Adaptive Minds, an agentic system that treats LoRA adapters as domain-specific tools. Instead of relying on a single fine-tuned model or rigid rule-based routing, our approach empowers the base LLM itself to act as a semantic router analyzing each query and dynamically selecting the most relevant LoRA tool. This enables the agent to seamlessly switch between different domain experts on demand. By combining the flexibility of multi-agent orchestration with the efficiency of parameter-efficient fine-tuning, Adaptive Minds delivers accurate, specialized responses while preserving conversational ability. The system is built with LangGraph for workflow management, supports both API and web interfaces, and is fully open source, providing a scalable and extensible foundation for domain-adaptive AI assistance.

**arXiv ID:** 2510.15416
</details>

<details>
<summary><strong>The Spark Effect: On Engineering Creative Diversity in Multi-Agent AI Systems</strong> - Alexander Doudkin, Anton Voelker, Friedrich von Borries - [[pdf]](https://arxiv.org/pdf/2510.15568)</summary>

**Abstract:** Creative services teams increasingly rely on large language models (LLMs) to accelerate ideation, yet production systems often converge on homogeneous outputs that fail to meet brand or artistic expectations. Art of X developed persona-conditioned LLM agents -- internally branded as "Sparks" and instantiated through a library of role-inspired system prompts -- to intentionally diversify agent behaviour within a multi-agent workflow. This white paper documents the problem framing, experimental design, and quantitative evidence behind the Spark agent programme. Using an LLM-as-a-judge protocol calibrated against human gold standards, we observe a mean diversity gain of +4.1 points (on a 1-10 scale) when persona-conditioned Spark agents replace a uniform system prompt, narrowing the gap to human experts to 1.0 point. We also surface evaluator bias and procedural considerations for future deployments.

**arXiv ID:** 2510.15568
</details>

<details>
<summary><strong>Where Common Knowledge Cannot Be Formed, Common Belief Can -- Planning with Multi-Agent Belief Using Group Justified Perspectives</strong> - Guang Hu, Tim Miller, Nir Lipovetzky - [[pdf]](https://arxiv.org/pdf/2412.07981)</summary>

**Abstract:** Epistemic planning is the sub-field of AI planning that focuses on changing knowledge and belief. It is important in both multi-agent domains where agents need to have knowledge/belief regarding the environment, but also the beliefs of other agents, including nested beliefs. When modeling knowledge in multi-agent settings, many models face an exponential growth challenge in terms of nested depth. A contemporary method, known as Planning with Perspectives (PWP), addresses these challenges through the use of perspectives and set operations for knowledge. The JP model defines that an agent's belief is justified if and only if the agent has seen evidence that this belief was true in the past and has not seen evidence to suggest that this has changed. The current paper extends the JP model to handle \emph{group belief}, including distributed belief and common belief. We call this the Group Justified Perspective (GJP) model. Using experimental problems crafted by adapting well-known benchmarks to a group setting, we show the efficiency and expressiveness of our GJP model at handling planning problems that cannot be handled by other epistemic planning tools.

**arXiv ID:** 2412.07981
</details>

<details>
<summary><strong>AppCopilot: Toward General, Accurate, Long-Horizon, and Efficient Mobile Agent</strong> - Jingru Fan, Yufan Dang, Jingyao Wu, Huatao Li, Runde Yang, Xiyuan Yang, Yuheng Wang, Chen Qian - [[pdf]](https://arxiv.org/pdf/2509.02444)</summary>

**Abstract:** With the raid evolution of large language models and multimodal models, the mobile-agent landscape has proliferated without converging on the fundamental challenges. This paper identifies four core problems that should be solved for mobile agents to deliver practical, scalable impact: (1) generalization across tasks, APPs, and devices; (2) accuracy, specifically precise on-screen interaction and click targeting; (3) long-horizon capability for sustained, multi-step goals; and (4) efficiency, specifically high-performance runtime on resource-constrained devices. We present AppCopilot, a multimodal, multi-agent, general-purpose mobile agent that operates across applications. AppCopilot operationalizes this position through an end-to-end pipeline spanning data collection, training, finetuning, efficient inference, and PC/mobile application. At the model layer, it integrates multimodal foundation models with robust Chinese-English support. At the reasoning and control layer, it combines chain-of-thought reasoning, hierarchical task planning and decomposition, and multi-agent collaboration. At the execution layer, it enables experiential adaptation, voice interaction, function calling, cross-APP and cross-device orchestration, and comprehensive mobile APP support. The system design incorporates profiling-driven optimization for latency and memory across heterogeneous hardware. Empirically, AppCopilot achieves significant improvements on four dimensions: stronger generalization, higher precision of on screen actions, more reliable long horizon task completion, and faster, more resource efficient runtime. By articulating a cohesive position and a reference architecture that closes the loop from data collection, training to finetuning and efficient inference, this paper offers a concrete roadmap for general purpose mobile agent and provides actionable guidance.

**arXiv ID:** 2509.02444
</details>

<details>
<summary><strong>Where Did It All Go Wrong? A Hierarchical Look into Multi-Agent Error Attribution</strong> - Adi Banerjee, Anirudh Nair, Tarik Borogovac - [[pdf]](https://arxiv.org/pdf/2510.04886)</summary>

**Abstract:** Error attribution in Large Language Model (LLM) multi-agent systems presents a significant challenge in debugging and improving collaborative AI systems. Current approaches to pinpointing agent and step level failures in interaction traces - whether using all-at-once evaluation, step-by-step analysis, or binary search - fall short when analyzing complex patterns, struggling with both accuracy and consistency. We present ECHO (Error attribution through Contextual Hierarchy and Objective consensus analysis), a novel algorithm that combines hierarchical context representation, objective analysis-based evaluation, and consensus voting to improve error attribution accuracy. Our approach leverages a positional-based leveling of contextual understanding while maintaining objective evaluation criteria, ultimately reaching conclusions through a consensus mechanism. Experimental results demonstrate that ECHO outperforms existing methods across various multi-agent interaction scenarios, showing particular strength in cases involving subtle reasoning errors and complex interdependencies. Our findings suggest that leveraging these concepts of structured, hierarchical context representation combined with consensus-based objective decision-making, provides a more robust framework for error attribution in multi-agent systems.

**arXiv ID:** 2510.04886
</details>

<details>
<summary><strong>Scaling Multi Agent Reinforcement Learning for Underwater Acoustic Tracking via Autonomous Vehicles</strong> - Matteo Gallici, Ivan Masmitja, Mario Martín - [[pdf]](https://arxiv.org/pdf/2505.08222)</summary>

**Abstract:** Autonomous vehicles (AV) offer a cost-effective solution for scientific missions such as underwater tracking. Recently, reinforcement learning (RL) has emerged as a powerful method for controlling AVs in complex marine environments. However, scaling these techniques to a fleet--essential for multi-target tracking or targets with rapid, unpredictable motion--presents significant computational challenges. Multi-Agent Reinforcement Learning (MARL) is notoriously sample-inefficient, and while high-fidelity simulators like Gazebo's LRAUV provide 100x faster-than-real-time single-robot simulations, they offer no significant speedup for multi-vehicle scenarios, making MARL training impractical. To address these limitations, we propose an iterative distillation method that transfers high-fidelity simulations into a simplified, GPU-accelerated environment while preserving high-level dynamics. This approach achieves up to a 30,000x speedup over Gazebo through parallelization, enabling efficient training via end-to-end GPU acceleration. Additionally, we introduce a novel Transformer-based architecture (TransfMAPPO) that learns multi-agent policies invariant to the number of agents and targets, significantly improving sample efficiency. Following large-scale curriculum learning conducted entirely on GPU, we perform extensive evaluations in Gazebo, demonstrating that our method maintains tracking errors below 5 meters over extended durations, even in the presence of multiple fast-moving targets. This work bridges the gap between large-scale MARL training and high-fidelity deployment, providing a scalable framework for autonomous fleet control in real-world sea missions.

**arXiv ID:** 2505.08222
</details>

<details>
<summary><strong>Topological Structure Learning Should Be A Research Priority for LLM-Based Multi-Agent Systems</strong> - Jiaxi Yang, Mengqi Zhang, Yiqiao Jin, Hao Chen, Qingsong Wen, Lu Lin, Yi He, Srijan Kumar, Weijie Xu, James Evans, Jindong Wang - [[pdf]](https://arxiv.org/pdf/2505.22467)</summary>

**Abstract:** Large Language Model-based Multi-Agent Systems (MASs) have emerged as a powerful paradigm for tackling complex tasks through collaborative intelligence. However, the topology of these systems--how agents in MASs should be configured, connected, and coordinated--remains largely unexplored. In this position paper, we call for a paradigm shift toward \emph{topology-aware MASs} that explicitly model and dynamically optimize the structure of inter-agent interactions. We identify three fundamental components--agents, communication links, and overall topology--that collectively determine the system's adaptability, efficiency, robustness, and fairness. To operationalize this vision, we introduce a systematic three-stage framework: 1) agent selection, 2) structure profiling, and 3) topology synthesis. This framework not only provides a principled foundation for designing MASs but also opens new research frontiers across language modeling, reinforcement learning, graph learning, and generative modeling to ultimately unleash their full potential in complex real-world applications. We conclude by outlining key challenges and opportunities in MASs evaluation. We hope our framework and perspectives offer critical new insights in the era of agentic AI.

**arXiv ID:** 2505.22467
</details>

<details>
<summary><strong>Decentralizing Multi-Agent Reinforcement Learning with Temporal Causal Information</strong> - Jan Corazza, Hadi Partovi Aria, Hyohun Kim, Daniel Neider, Zhe Xu - [[pdf]](https://arxiv.org/pdf/2506.07829)</summary>

**Abstract:** Reinforcement learning (RL) algorithms can find an optimal policy for a single agent to accomplish a particular task. However, many real-world problems require multiple agents to collaborate in order to achieve a common goal. For example, a robot executing a task in a warehouse may require the assistance of a drone to retrieve items from high shelves. In Decentralized Multi-Agent RL (DMARL), agents learn independently and then combine their policies at execution time, but often must satisfy constraints on compatibility of local policies to ensure that they can achieve the global task when combined. In this paper, we study how providing high-level symbolic knowledge to agents can help address unique challenges of this setting, such as privacy constraints, communication limitations, and performance concerns. In particular, we extend the formal tools used to check the compatibility of local policies with the team task, making decentralized training with theoretical guarantees usable in more scenarios. Furthermore, we empirically demonstrate that symbolic knowledge about the temporal evolution of events in the environment can significantly expedite the learning process in DMARL.

**arXiv ID:** 2506.07829
</details>

<details>
<summary><strong>Bayesian Ego-graph inference for Networked Multi-Agent Reinforcement Learning</strong> - Wei Duan, Jie Lu, Junyu Xuan - [[pdf]](https://arxiv.org/pdf/2509.16606)</summary>

**Abstract:** In networked multi-agent reinforcement learning (Networked-MARL), decentralized agents must act under local observability and constrained communication over fixed physical graphs. Existing methods often assume static neighborhoods, limiting adaptability to dynamic or heterogeneous environments. While centralized frameworks can learn dynamic graphs, their reliance on global state access and centralized infrastructure is impractical in real-world decentralized systems. We propose a stochastic graph-based policy for Networked-MARL, where each agent conditions its decision on a sampled subgraph over its local physical neighborhood. Building on this formulation, we introduce BayesG, a decentralized actor-framework that learns sparse, context-aware interaction structures via Bayesian variational inference. Each agent operates over an ego-graph and samples a latent communication mask to guide message passing and policy computation. The variational distribution is trained end-to-end alongside the policy using an evidence lower bound (ELBO) objective, enabling agents to jointly learn both interaction topology and decision-making strategies. BayesG outperforms strong MARL baselines on large-scale traffic control tasks with up to 167 agents, demonstrating superior scalability, efficiency, and performance.

**arXiv ID:** 2509.16606
</details>

<details>
<summary><strong>Cooperative Bargaining Games Without Utilities: Mediated Solutions from Direction Oracles</strong> - Kushagra Gupta, Surya Murthy, Mustafa O. Karabag, Ufuk Topcu, David Fridovich-Keil - [[pdf]](https://arxiv.org/pdf/2505.14817)</summary>

**Abstract:** Cooperative bargaining games are widely used to model resource allocation and conflict resolution. Traditional solutions assume the mediator can access agents utility function values and gradients. However, there is an increasing number of settings, such as human AI interactions, where utility values may be inaccessible or incomparable due to unknown, nonaffine transformations. To model such settings, we consider that the mediator has access only to agents most preferred directions, i.e., normalized utility gradients in the decision space. To this end, we propose a cooperative bargaining algorithm where a mediator has access to only the direction oracle of each agent. We prove that unlike popular approaches such as the Nash and Kalai Smorodinsky bargaining solutions, our approach is invariant to monotonic nonaffine transformations, and that under strong convexity and smoothness assumptions, this approach enjoys global asymptotic convergence to Pareto stationary solutions. Moreover, we show that the bargaining solutions found by our algorithm also satisfy the axioms of symmetry and (under slightly stronger conditions) independence of irrelevant alternatives, which are popular in the literature. Finally, we conduct experiments in two domains, multi agent formation assignment and mediated stock portfolio allocation, which validate these theoretic results. All code for our experiments can be found at this https URL.

**arXiv ID:** 2505.14817
</details>

<details>
<summary><strong>MAGPIE: A benchmark for Multi-AGent contextual PrIvacy Evaluation</strong> - Gurusha Juneja, Jayanth Naga Sai Pasupulati, Alon Albalak, Wenyue Hua, William Yang Wang - [[pdf]](https://arxiv.org/pdf/2510.15186)</summary>

**Abstract:** A core challenge for autonomous LLM agents in collaborative settings is balancing robust privacy understanding and preservation alongside task efficacy. Existing privacy benchmarks only focus on simplistic, single-turn interactions where private information can be trivially omitted without affecting task outcomes. In this paper, we introduce MAGPIE (Multi-AGent contextual PrIvacy Evaluation), a novel benchmark of 200 high-stakes tasks designed to evaluate privacy understanding and preservation in multi-agent collaborative, non-adversarial scenarios. MAGPIE integrates private information as essential for task resolution, forcing agents to balance effective collaboration with strategic information control. Our evaluation reveals that state-of-the-art agents, including GPT-5 and Gemini 2.5-Pro, exhibit significant privacy leakage, with Gemini 2.5-Pro leaking up to 50.7% and GPT-5 up to 35.1% of the sensitive information even when explicitly instructed not to. Moreover, these agents struggle to achieve consensus or task completion and often resort to undesirable behaviors such as manipulation and power-seeking (e.g., Gemini 2.5-Pro demonstrating manipulation in 38.2% of the cases). These findings underscore that current LLM agents lack robust privacy understanding and are not yet adequately aligned to simultaneously preserve privacy and maintain effective collaboration in complex environments.

**arXiv ID:** 2510.15186
</details>

<details>
<summary><strong>SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation</strong> - Ines Besrour, Jingbo He, Tobias Schreieder, Michael Färber - [[pdf]](https://arxiv.org/pdf/2510.15682)</summary>

**Abstract:** We present SQuAI (this https URL), a scalable and trustworthy multi-agent retrieval-augmented generation (RAG) framework for scientific question answering (QA) with large language models (LLMs). SQuAI addresses key limitations of existing RAG systems in the scholarly domain, where complex, open-domain questions demand accurate answers, explicit claims with citations, and retrieval across millions of scientific documents. Built on over 2.3 million full-text papers from arXiv.org, SQuAI employs four collaborative agents to decompose complex questions into sub-questions, retrieve targeted evidence via hybrid sparse-dense retrieval, and adaptively filter documents to improve contextual relevance. To ensure faithfulness and traceability, SQuAI integrates in-line citations for each generated claim and provides supporting sentences from the source documents. Our system improves faithfulness, answer relevance, and contextual relevance by up to +0.088 (12%) over a strong RAG baseline. We further release a benchmark of 1,000 scientific question-answer-evidence triplets to support reproducibility. With transparent reasoning, verifiable citations, and domain-wide scalability, SQuAI demonstrates how multi-agent RAG enables more trustworthy scientific QA with LLMs.

**arXiv ID:** 2510.15682
</details>

<details>
<summary><strong>Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition</strong> - Zheng Hui, Xiaokai Wei, Yexi Jiang, Kevin Gao, Chen Wang, Frank Ong, Se-eun Yoon, Rachit Pareek, Michelle Gong - [[pdf]](https://arxiv.org/pdf/2504.20094)</summary>

**Abstract:** Conversational recommender systems (CRS) have advanced with large language models, showing strong results in domains like movies. These domains typically involve fixed content and passive consumption, where user preferences can be matched by genre or theme. In contrast, games present distinct challenges: fast-evolving catalogs, interaction-driven preferences (e.g., skill level, mechanics, hardware), and increased risk of unsafe responses in open-ended conversation. We propose MATCHA, a multi-agent framework for CRS that assigns specialized agents for intent parsing, tool-augmented retrieval, multi-LLM ranking with reflection, explanation, and risk control which enabling finer personalization, long-tail coverage, and stronger safety. Evaluated on real user request dataset, MATCHA outperforms six baselines across eight metrics, improving Hit@5 by 20%, reducing popularity bias by 24%, and achieving 97.9% adversarial defense. Human and virtual-judge evaluations confirm improved explanation quality and user alignment.

**arXiv ID:** 2504.20094
</details>

<details>
<summary><strong>Onboard Mission Replanning for Adaptive Cooperative Multi-Robot Systems</strong> - Elim Kwan, Rehman Qureshi, Liam Fletcher, Colin Laganier, Victoria Nockles, Richard Walters - [[pdf]](https://arxiv.org/pdf/2506.06094)</summary>

**Abstract:** Cooperative autonomous robotic systems have significant potential for executing complex multi-task missions across space, air, ground, and maritime domains. But they commonly operate in remote, dynamic and hazardous environments, requiring rapid in-mission adaptation without reliance on fragile or slow communication links to centralised compute. Fast, on-board replanning algorithms are therefore needed to enhance resilience. Reinforcement Learning shows strong promise for efficiently solving mission planning tasks when formulated as Travelling Salesperson Problems (TSPs), but existing methods: 1) are unsuitable for replanning, where agents do not start at a single location; 2) do not allow cooperation between agents; 3) are unable to model tasks with variable durations; or 4) lack practical considerations for on-board deployment. Here we define the Cooperative Mission Replanning Problem as a novel variant of multiple TSP with adaptations to overcome these issues, and develop a new encoder/decoder-based model using Graph Attention Networks and Attention Models to solve it effectively and efficiently. Using a simple example of cooperative drones, we show our replanner consistently (90% of the time) maintains performance within 10% of the state-of-the-art LKH3 heuristic solver, whilst running 85-370 times faster on a Raspberry Pi. This work paves the way for increased resilience in autonomous multi-agent systems.

**arXiv ID:** 2506.06094
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>Build Your Personalized Research Group: A Multiagent Framework for Continual and Interactive Science Automation</strong> - Ed Li, Junyu Ren, Xintian Pan, Cat Yan, Chuanhao Li, Dirk Bergemann, Zhuoran Yang - [[pdf]](https://arxiv.org/pdf/2510.15624)</summary>

**Abstract:** The automation of scientific discovery represents a critical milestone in Artificial Intelligence (AI) research. However, existing agentic systems for science suffer from two fundamental limitations: rigid, pre-programmed workflows that cannot adapt to intermediate findings, and inadequate context management that hinders long-horizon research. We present \texttt{freephdlabor}, an open-source multiagent framework featuring \textit{fully dynamic workflows} determined by real-time agent reasoning and a \coloremph{\textit{modular architecture}} enabling seamless customization -- users can modify, add, or remove agents to address domain-specific requirements. The framework provides comprehensive infrastructure including \textit{automatic context compaction}, \textit{workspace-based communication} to prevent information degradation, \textit{memory persistence} across sessions, and \textit{non-blocking human intervention} mechanisms. These features collectively transform automated research from isolated, single-run attempts into \textit{continual research programs} that build systematically on prior explorations and incorporate human feedback. By providing both the architectural principles and practical implementation for building customizable co-scientist systems, this work aims to facilitate broader adoption of automated research across scientific domains, enabling practitioners to deploy interactive multiagent systems that autonomously conduct end-to-end research -- from ideation through experimentation to publication-ready manuscripts.

**arXiv ID:** 2510.15624
</details>

<details>
<summary><strong>PowerChain: A Verifiable Agentic AI System for Automating Distribution Grid Analyses</strong> - Emmanuel O. Badmus, Peng Sang, Dimitrios Stamoulis, Amritanshu Pandey - [[pdf]](https://arxiv.org/pdf/2508.17094)</summary>

**Abstract:** Rapid electrification and decarbonization are increasing the complexity of distribution grid (DG) operation and planning, necessitating advanced computational analyses to ensure reliability and resilience. These analyses depend on disparate workflows comprising complex models, function calls, and data pipelines that require substantial expert knowledge and remain difficult to automate. Workforce and budget constraints further limit utilities' ability to apply such analyses at scale. To address this gap, we build an agentic system PowerChain, which is capable of autonomously performing complex grid analyses. Existing agentic AI systems are typically developed in a bottom-up manner with customized context for predefined analysis tasks; therefore, they do not generalize to tasks that the agent has never seen. In comparison, to generalize to unseen DG analysis tasks, PowerChain dynamically generates structured context by leveraging supervisory signals from self-contained power systems tools (e.g., GridLAB-D) and an optimized set of expert-annotated and verified reasoning trajectories. For complex DG tasks defined in natural language, empirical results on real utility data demonstrate that PowerChain achieves up to a 144/% improvement in performance over baselines.

**arXiv ID:** 2508.17094
</details>

<details>
<summary><strong>CORE: Reducing UI Exposure in Mobile Agents via Collaboration Between Cloud and Local LLMs</strong> - Gucongcong Fan, Chaoyue Niu, Chengfei Lyu, Fan Wu, Guihai Chen - [[pdf]](https://arxiv.org/pdf/2510.15455)</summary>

**Abstract:** Mobile agents rely on Large Language Models (LLMs) to plan and execute tasks on smartphone user interfaces (UIs). While cloud-based LLMs achieve high task accuracy, they require uploading the full UI state at every step, exposing unnecessary and often irrelevant information. In contrast, local LLMs avoid UI uploads but suffer from limited capacity, resulting in lower task success rates. We propose $\textbf{CORE}$, a $\textbf{CO}$llaborative framework that combines the strengths of cloud and local LLMs to $\textbf{R}$educe UI $\textbf{E}$xposure, while maintaining task accuracy for mobile agents. CORE comprises three key components: (1) $\textbf{Layout-aware block partitioning}$, which groups semantically related UI elements based on the XML screen hierarchy; (2) $\textbf{Co-planning}$, where local and cloud LLMs collaboratively identify the current sub-task; and (3) $\textbf{Co-decision-making}$, where the local LLM ranks relevant UI blocks, and the cloud LLM selects specific UI elements within the top-ranked block. CORE further introduces a multi-round accumulation mechanism to mitigate local misjudgment or limited context. Experiments across diverse mobile apps and tasks show that CORE reduces UI exposure by up to 55.6% while maintaining task success rates slightly below cloud-only agents, effectively mitigating unnecessary privacy exposure to the cloud. The code is available at this https URL.

**arXiv ID:** 2510.15455
</details>

<details>
<summary><strong>Nauplius Optimisation for Autonomous Hydrodynamics</strong> - Shyalan Ramesh, Scott Mann, Alex Stumpf - [[pdf]](https://arxiv.org/pdf/2510.15350)</summary>

**Abstract:** Autonomous Underwater vehicles must operate in strong currents, limited acoustic bandwidth, and persistent sensing requirements where conventional swarm optimisation methods are unreliable. This paper presents NOAH, a novel nature-inspired swarm optimisation algorithm that combines current-aware drift, irreversible settlement in persistent sensing nodes, and colony-based communication. Drawing inspiration from the behaviour of barnacle nauplii, NOAH addresses the critical limitations of existing swarm algorithms by providing hydrodynamic awareness, irreversible anchoring mechanisms, and colony-based communication capabilities essential for underwater exploration missions. The algorithm establishes a comprehensive foundation for scalable and energy-efficient underwater swarm robotics with validated performance analysis. Validation studies demonstrate an 86% success rate for permanent anchoring scenarios, providing a unified formulation for hydrodynamic constraints and irreversible settlement behaviours with an empirical study under flow.

**arXiv ID:** 2510.15350
</details>

<details>
<summary><strong>Free Lunch for User Experience: Crowdsourcing Agents for Scalable User Studies</strong> - Siyang Liu, Sahand Sabour, Xiaoyang Wang, Rada Mihalcea - [[pdf]](https://arxiv.org/pdf/2505.22981)</summary>

**Abstract:** User studies are central to user experience research, yet recruiting participant is expensive, slow, and limited in diversity. Recent work has explored using Large Language Models as simulated users, but doubts about fidelity have hindered practical adoption. We deepen this line of research by asking whether scale itself can enable useful simulation, even if not perfectly accurate. We introduce Crowdsourcing Simulated User Agents, a method that recruits generative agents from billion-scale profile assets to act as study participants. Unlike handcrafted simulations, agents are treated as recruitable, screenable, and engageable across UX research stages. To ground this method, we demonstrate a game prototyping study with hundreds of simulated players, comparing their insights against a 10-participant local user study and a 20-participant crowdsourcing study with humans. We find a clear scaling effect: as the number of simulated user agents increases, coverage of human findings rises smoothly and plateaus around 90\%. 12.8 simulated agents are as useful as one locally recruited human, and 3.2 agents are as useful as one crowdsourced human. Results show that while individual agents are imperfect, aggregated simulations produce representative and actionable insights comparable to real users. Professional designers further rated these insights as balancing fidelity, cost, time efficiency, and usefulness. Finally, we release an agent crowdsourcing toolkit with a modular open-source pipeline and a curated pool of profiles synced from ongoing simulation research, to lower the barrier for researchers to adopt simulated participants. Together, this work contributes a validated method and reusable toolkit that expand the options for conducting scalable and practical UX studies.

**arXiv ID:** 2505.22981
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (33 papers)</h2></summary>

<details>
<summary><strong>Procedural Game Level Design with Deep Reinforcement Learning</strong> - Miraç Buğra Özkan - [[pdf]](https://arxiv.org/pdf/2510.15120)</summary>

**Abstract:** Procedural content generation (PCG) has become an increasingly popular technique in game development, allowing developers to generate dynamic, replayable, and scalable environments with reduced manual effort. In this study, a novel method for procedural level design using Deep Reinforcement Learning (DRL) within a Unity-based 3D environment is proposed. The system comprises two agents: a hummingbird agent, acting as a solver, and a floating island agent, responsible for generating and placing collectible objects (flowers) on the terrain in a realistic and context-aware manner. The hummingbird is trained using the Proximal Policy Optimization (PPO) algorithm from the Unity ML-Agents toolkit. It learns to navigate through the terrain efficiently, locate flowers, and collect them while adapting to the ever-changing procedural layout of the island. The island agent is also trained using the Proximal Policy Optimization (PPO) algorithm. It learns to generate flower layouts based on observed obstacle positions, the hummingbird's initial state, and performance feedback from previous episodes. The interaction between these agents leads to emergent behavior and robust generalization across various environmental configurations. The results demonstrate that the approach not only produces effective and efficient agent behavior but also opens up new opportunities for autonomous game level design driven by machine learning. This work highlights the potential of DRL in enabling intelligent agents to both generate and solve content in virtual environments, pushing the boundaries of what AI can contribute to creative game development processes.

**arXiv ID:** 2510.15120
</details>

<details>
<summary><strong>Experience-Driven Exploration for Efficient API-Free AI Agents</strong> - Chenwei Tang, Jingyu Xing, Xinyu Liu, Zizhou Wang, Jiawei Du, Liangli Zhen, Jiancheng Lv - [[pdf]](https://arxiv.org/pdf/2510.15259)</summary>

**Abstract:** Most existing software lacks accessible Application Programming Interfaces (APIs), requiring agents to operate solely through pixel-based Graphical User Interfaces (GUIs). In this API-free setting, large language model (LLM)-based agents face severe efficiency bottlenecks: limited to local visual experiences, they make myopic decisions and rely on inefficient trial-and-error, hindering both skill acquisition and long-term planning. To address these challenges, we propose KG-Agent, an experience-driven learning framework that structures an agent's raw pixel-level interactions into a persistent State-Action Knowledge Graph (SA-KG). KG-Agent overcomes inefficient exploration by linking functionally similar but visually distinct GUI states, forming a rich neighborhood of experience that enables the agent to generalize from a diverse set of historical strategies. To support long-horizon reasoning, we design a hybrid intrinsic reward mechanism based on the graph topology, combining a state value reward for exploiting known high-value pathways with a novelty reward that encourages targeted exploration. This approach decouples strategic planning from pure discovery, allowing the agent to effectively value setup actions with delayed gratification. We evaluate KG-Agent in two complex, open-ended GUI-based decision-making environments (Civilization V and Slay the Spire), demonstrating significant improvements in exploration efficiency and strategic depth over the state-of-the-art methods.

**arXiv ID:** 2510.15259
</details>

<details>
<summary><strong>Taming the Judge: Deconflicting AI Feedback for Stable Reinforcement Learning</strong> - Boyin Liu, Zhuo Zhang, Sen Huang, Lipeng Xie, Qingxu Fu, Haoran Chen, LI YU, Tianyi Hu, Zhaoyang Liu, Bolin Ding, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2510.15514)</summary>

**Abstract:** However, this method often faces judgment inconsistencies that can destabilize reinforcement learning. While prior research has focused on the accuracy of judgments, the critical issue of logical coherence especially issues such as preference cycles hasn't been fully addressed. To fill this gap, we introduce a comprehensive framework designed to systematically detect and resolve these inconsistencies during the reinforcement learning training process. Our framework includes two main contributions: first, the Conflict Detection Rate (CDR), a new metric that quantifies judgment conflicts, and second, Deconflicted Graph Rewards (DGR), a framework that purifies signals by removing cycles before policy optimization. DGR constructs preference graphs from the initial judgments, transforms them into conflict-free Directed Acyclic Graphs (DAGs), and generates a logically coherent reward signal that is compatible with any policy optimizer. Experimental results show that our framework significantly enhances training stability and model performance compared to strong baselines, establishing logical consistency as a crucial and now manageable dimension of AI feedback.

**arXiv ID:** 2510.15514
</details>

<details>
<summary><strong>PokeeResearch: Effective Deep Research via Reinforcement Learning from AI Feedback and Robust Reasoning Scaffold</strong> - Yi Wan, Jiuqi Wang, Liam Li, Jinsong Liu, Ruihao Zhu, Zheqing Zhu - [[pdf]](https://arxiv.org/pdf/2510.15862)</summary>

**Abstract:** Tool-augmented large language models (LLMs) are emerging as deep research agents, systems that decompose complex queries, retrieve external evidence, and synthesize grounded responses. Yet current agents remain limited by shallow retrieval, weak alignment metrics, and brittle tool-use behavior. We introduce PokeeResearch-7B, a 7B-parameter deep research agent built under a unified reinforcement learning framework for robustness, alignment, and scalability. PokeeResearch-7B is trained by an annotation-free Reinforcement Learning from AI Feedback (RLAIF) framework to optimize policies using LLM-based reward signals that capture factual accuracy, citation faithfulness, and instruction adherence. A chain-of-thought-driven multi-call reasoning scaffold further enhances robustness through self-verification and adaptive recovery from tool failures. Among 10 popular deep research benchmarks, PokeeResearch-7B achieves state-of-the-art performance among 7B-scale deep research agents. This highlights that careful reinforcement learning and reasoning design can produce efficient, resilient, and research-grade AI agents. The model and inference code is open-sourced under MIT license at this https URL.

**arXiv ID:** 2510.15862
</details>

<details>
<summary><strong>Reinforcement Learning with Stochastic Reward Machines</strong> - Jan Corazza, Ivan Gavran, Daniel Neider - [[pdf]](https://arxiv.org/pdf/2510.14837)</summary>

**Abstract:** Reward machines are an established tool for dealing with reinforcement learning problems in which rewards are sparse and depend on complex sequences of actions. However, existing algorithms for learning reward machines assume an overly idealized setting where rewards have to be free of noise. To overcome this practical limitation, we introduce a novel type of reward machines, called stochastic reward machines, and an algorithm for learning them. Our algorithm, based on constraint solving, learns minimal stochastic reward machines from the explorations of a reinforcement learning agent. This algorithm can easily be paired with existing reinforcement learning algorithms for reward machines and guarantees to converge to an optimal policy in the limit. We demonstrate the effectiveness of our algorithm in two case studies and show that it outperforms both existing methods and a naive approach for handling noisy reward functions.

**arXiv ID:** 2510.14837
</details>

<details>
<summary><strong>DLER: Doing Length pEnalty Right - Incentivizing More Intelligence per Token via Reinforcement Learning</strong> - Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, Pavlo Molchanov - [[pdf]](https://arxiv.org/pdf/2510.15110)</summary>

**Abstract:** Reasoning language models such as OpenAI-o1, DeepSeek-R1, and Qwen achieve strong performance via extended chains of thought but often generate unnecessarily long outputs. Maximizing intelligence per token--accuracy relative to response length--remains an open problem. We revisit reinforcement learning (RL) with the simplest length penalty--truncation--and show that accuracy degradation arises not from the lack of sophisticated penalties but from inadequate RL optimization. We identify three key challenges: (i) large bias in advantage estimation, (ii) entropy collapse, and (iii) sparse reward signal. We address them with Doing Length pEnalty Right (DLER), a training recipe combining batch-wise reward normalization, higher clipping, dynamic sampling, and a simple truncation length penalty. DLER achieves state-of-the-art accuracy--efficiency trade-offs, cutting output length by over 70 percent while surpassing all previous baseline accuracy. It also improves test-time scaling: compared to DeepSeek-R1-7B, DLER-7B generates multiple concise responses in parallel with 28 percent higher accuracy and lower latency. We further introduce Difficulty-Aware DLER, which adaptively tightens truncation on easier questions for additional efficiency gains. We also propose an update-selective merging method that preserves baseline accuracy while retaining the concise reasoning ability of the DLER model, which is useful for scenarios where RL training data is scarce.

**arXiv ID:** 2510.15110
</details>

<details>
<summary><strong>Structure-R1: Dynamically Leveraging Structural Knowledge in LLM Reasoning through Reinforcement Learning</strong> - Junlin Wu, Xianrui Zhong, Jiashuo Sun, Bolian Li, Bowen Jin, Jiawei Han, Qingkai Zeng - [[pdf]](https://arxiv.org/pdf/2510.15191)</summary>

**Abstract:** Large language models (LLMs) have demonstrated remarkable advances in reasoning capabilities. However, their performance remains constrained by limited access to explicit and structured domain knowledge. Retrieval-Augmented Generation (RAG) addresses this by incorporating external information as context to augment reasoning. Nevertheless, traditional RAG systems typically operate over unstructured and fragmented text, resulting in low information density and suboptimal reasoning. To overcome these limitations, we propose \textsc{Structure-R1}, a novel framework that transforms retrieved content into structured representations optimized for reasoning. Leveraging reinforcement learning, \textsc{Structure-R1} learns a content representation policy that dynamically generates and adapts structural formats based on the demands of multi-step reasoning. Unlike prior methods that rely on fixed schemas, our approach adopts a generative paradigm capable of producing task-specific structures tailored to individual queries. To ensure the quality and reliability of these representations, we introduce a self-reward structural verification mechanism that checks whether the generated structures are both correct and self-contained. Extensive experiments on seven knowledge-intensive benchmarks show that \textsc{Structure-R1} consistently achieves competitive performance with a 7B-scale backbone model and matches the performance of much larger models. Additionally, our theoretical analysis demonstrates how structured representations enhance reasoning by improving information density and contextual clarity. Our code and data are available at: this https URL.

**arXiv ID:** 2510.15191
</details>

<details>
<summary><strong>Towards Robust Zero-Shot Reinforcement Learning</strong> - Kexin Zheng, Lauriane Teyssier, Yinan Zheng, Yu Luo, Xiayuan Zhan - [[pdf]](https://arxiv.org/pdf/2510.15382)</summary>

**Abstract:** The recent development of zero-shot reinforcement learning (RL) has opened a new avenue for learning pre-trained generalist policies that can adapt to arbitrary new tasks in a zero-shot manner. While the popular Forward-Backward representations (FB) and related methods have shown promise in zero-shot RL, we empirically found that their modeling lacks expressivity and that extrapolation errors caused by out-of-distribution (OOD) actions during offline learning sometimes lead to biased representations, ultimately resulting in suboptimal performance. To address these issues, we propose Behavior-REgularizEd Zero-shot RL with Expressivity enhancement (BREEZE), an upgraded FB-based framework that simultaneously enhances learning stability, policy extraction capability, and representation learning quality. BREEZE introduces behavioral regularization in zero-shot RL policy learning, transforming policy optimization into a stable in-sample learning paradigm. Additionally, BREEZE extracts the policy using a task-conditioned diffusion model, enabling the generation of high-quality and multimodal action distributions in zero-shot RL settings. Moreover, BREEZE employs expressive attention-based architectures for representation modeling to capture the complex relationships between environmental dynamics. Extensive experiments on ExORL and D4RL Kitchen demonstrate that BREEZE achieves the best or near-the-best performance while exhibiting superior robustness compared to prior offline zero-shot RL methods. The official implementation is available at: this https URL.

**arXiv ID:** 2510.15382
</details>

<details>
<summary><strong>Expediting Reinforcement Learning by Incorporating Knowledge About Temporal Causality in the Environment</strong> - Jan Corazza, Hadi Partovi Aria, Daniel Neider, Zhe Xu - [[pdf]](https://arxiv.org/pdf/2510.15456)</summary>

**Abstract:** Reinforcement learning (RL) algorithms struggle with learning optimal policies for tasks where reward feedback is sparse and depends on a complex sequence of events in the environment. Probabilistic reward machines (PRMs) are finite-state formalisms that can capture temporal dependencies in the reward signal, along with nondeterministic task outcomes. While special RL algorithms can exploit this finite-state structure to expedite learning, PRMs remain difficult to modify and design by hand. This hinders the already difficult tasks of utilizing high-level causal knowledge about the environment, and transferring the reward formalism into a new domain with a different causal structure. This paper proposes a novel method to incorporate causal information in the form of Temporal Logic-based Causal Diagrams into the reward formalism, thereby expediting policy learning and aiding the transfer of task specifications to new environments. Furthermore, we provide a theoretical result about convergence to optimal policy for our method, and demonstrate its strengths empirically.

**arXiv ID:** 2510.15456
</details>

<details>
<summary><strong>OffSim: Offline Simulator for Model-based Offline Inverse Reinforcement Learning</strong> - Woo-Jin Ahn, Sang-Ryul Baek, Yong-Jun Lee, Hyun-Duck Choi, Myo-Taeg Lim - [[pdf]](https://arxiv.org/pdf/2510.15495)</summary>

**Abstract:** Reinforcement learning algorithms typically utilize an interactive simulator (i.e., environment) with a predefined reward function for policy training. Developing such simulators and manually defining reward functions, however, is often time-consuming and labor-intensive. To address this, we propose an Offline Simulator (OffSim), a novel model-based offline inverse reinforcement learning (IRL) framework, to emulate environmental dynamics and reward structure directly from expert-generated state-action trajectories. OffSim jointly optimizes a high-entropy transition model and an IRL-based reward function to enhance exploration and improve the generalizability of the learned reward. Leveraging these learned components, OffSim can subsequently train a policy offline without further interaction with the real environment. Additionally, we introduce OffSim$^+$, an extension that incorporates a marginal reward for multi-dataset settings to enhance exploration. Extensive MuJoCo experiments demonstrate that OffSim achieves substantial performance gains over existing offline IRL methods, confirming its efficacy and robustness.

**arXiv ID:** 2510.15495
</details>

<details>
<summary><strong>ProSh: Probabilistic Shielding for Model-free Reinforcement Learning</strong> - Edwin Hamel-De le Court, Gaspard Ohlmann, Francesco Belardinelli - [[pdf]](https://arxiv.org/pdf/2510.15720)</summary>

**Abstract:** Safety is a major concern in reinforcement learning (RL): we aim at developing RL systems that not only perform optimally, but are also safe to deploy by providing formal guarantees about their safety. To this end, we introduce Probabilistic Shielding via Risk Augmentation (ProSh), a model-free algorithm for safe reinforcement learning under cost constraints. ProSh augments the Constrained MDP state space with a risk budget and enforces safety by applying a shield to the agent's policy distribution using a learned cost critic. The shield ensures that all sampled actions remain safe in expectation. We also show that optimality is preserved when the environment is deterministic. Since ProSh is model-free, safety during training depends on the knowledge we have acquired about the environment. We provide a tight upper-bound on the cost in expectation, depending only on the backup-critic accuracy, that is always satisfied during training. Under mild, practically achievable assumptions, ProSh guarantees safety even at training time, as shown in the experiments.

**arXiv ID:** 2510.15720
</details>

<details>
<summary><strong>RLAF: Reinforcement Learning from Automaton Feedback</strong> - Mahyar Alinejad, Alvaro Velasquez, Yue Wang, George Atia - [[pdf]](https://arxiv.org/pdf/2510.15728)</summary>

**Abstract:** Reinforcement Learning (RL) in environments with complex, history-dependent reward structures poses significant challenges for traditional methods. In this work, we introduce a novel approach that leverages automaton-based feedback to guide the learning process, replacing explicit reward functions with preferences derived from a deterministic finite automaton (DFA). Unlike conventional approaches that use automata for direct reward specification, our method employs the structure of the DFA to generate preferences over trajectories that are used to learn a reward function, eliminating the need for manual reward engineering. Our framework introduces a static approach that uses the learned reward function directly for policy optimization and a dynamic approach that involves continuous refining of the reward function and policy through iterative updates until convergence.
Our experiments in both discrete and continuous environments demonstrate that our approach enables the RL agent to learn effective policies for tasks with temporal dependencies, outperforming traditional reward engineering and automaton-based baselines such as reward machines and LTL-guided methods. Our results highlight the advantages of automaton-based preferences in handling non-Markovian rewards, offering a scalable, efficient, and human-independent alternative to traditional reward modeling. We also provide a convergence guarantee showing that under standard assumptions our automaton-guided preference-based framework learns a policy that is near-optimal with respect to the true non-Markovian objective.

**arXiv ID:** 2510.15728
</details>

<details>
<summary><strong>VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use</strong> - Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai Zou, Chao Du, Tianyu Pang, Wenhu Chen - [[pdf]](https://arxiv.org/pdf/2509.01055)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated success in enhancing LLM reasoning capabilities, but remains limited to single-turn interactions without tool integration. While recent Agentic Reinforcement Learning with Tool use (ARLT) approaches have emerged to address multi-turn tool interactions, existing works develop task-specific codebases that suffer from fragmentation, synchronous execution bottlenecks, and limited extensibility across domains. These inefficiencies hinder broader community adoption and algorithmic innovation. We introduce VerlTool, a unified and modular framework that addresses these limitations through systematic design principles. VerlTool provides four key contributions: (1) upstream alignment with VeRL ensuring compatibility and simplified maintenance, (2) unified tool management via standardized APIs supporting diverse modalities including code execution, search, SQL databases, and vision processing, (3) asynchronous rollout execution achieving near 2$\times$ speedup by eliminating synchronization bottlenecks, and (4) comprehensive evaluation demonstrating competitive performance across 6 ARLT domains. Our framework formalizes ARLT as multi-turn trajectories with multi-modal observation tokens (text/image/video), extending beyond single-turn RLVR paradigms. We train and evaluate models on mathematical reasoning, knowledge QA, SQL generation, visual reasoning, web search, and software engineering tasks, achieving results comparable to specialized systems while providing unified training infrastructure. The modular plugin architecture enables rapid tool integration requiring only lightweight Python definitions, significantly reducing development overhead and providing a scalable foundation for tool-augmented RL research. Our code is open-sourced at this https URL.

**arXiv ID:** 2509.01055
</details>

<details>
<summary><strong>Towards smart and adaptive agents for active sensing on edge devices</strong> - Devendra Vyas, Nikola Pižurica, Nikola Milović, Igor Jovančević, Miguel de Prado, Tim Verbelen - [[pdf]](https://arxiv.org/pdf/2501.06262)</summary>

**Abstract:** TinyML has made deploying deep learning models on low-power edge devices feasible, creating new opportunities for real-time perception in constrained environments. However, the adaptability of such deep learning methods remains limited to data drift adaptation, lacking broader capabilities that account for the environment's underlying dynamics and inherent uncertainty. Deep learning's scaling laws, which counterbalance this limitation by massively up-scaling data and model size, cannot be applied when deploying on the Edge, where deep learning limitations are further amplified as models are scaled down for deployment on resource-constrained devices.
This paper presents an innovative agentic system capable of performing on-device perception and planning, enabling active sensing on the edge. By incorporating active inference into our solution, our approach extends beyond deep learning capabilities, allowing the system to plan in dynamic environments while operating in real-time with a compact memory footprint of as little as 300 MB. We showcase our proposed system by creating and deploying a saccade agent connected to an IoT camera with pan and tilt capabilities on an NVIDIA Jetson embedded device. The saccade agent controls the camera's field of view following optimal policies derived from the active inference principles, simulating human-like saccadic motion for surveillance and robotics applications.

**arXiv ID:** 2501.06262
</details>

<details>
<summary><strong>MOBODY: Model Based Off-Dynamics Offline Reinforcement Learning</strong> - Yihong Guo, Yu Yang, Pan Xu, Anqi Liu - [[pdf]](https://arxiv.org/pdf/2506.08460)</summary>

**Abstract:** We study off-dynamics offline reinforcement learning, where the goal is to learn a policy from offline source and limited target datasets with mismatched dynamics. Existing methods either penalize the reward or discard source transitions occurring in parts of the transition space with high dynamics shift. As a result, they optimize the policy using data from low-shift regions, limiting exploration of high-reward states in the target domain that do not fall within these regions. Consequently, such methods often fail when the dynamics shift is significant or the optimal trajectories lie outside the low-shift regions. To overcome this limitation, we propose MOBODY, a Model-Based Off-Dynamics Offline RL algorithm that optimizes a policy using learned target dynamics transitions to explore the target domain, rather than only being trained with the low dynamics-shift transitions. For the dynamics learning, built on the observation that achieving the same next state requires taking different actions in different domains, MOBODY employs separate action encoders for each domain to encode different actions to the shared latent space while sharing a unified representation of states and a common transition function. We further introduce a target Q-weighted behavior cloning loss in policy optimization to avoid out-of-distribution actions, which push the policy toward actions with high target-domain Q-values, rather than high source domain Q-values or uniformly imitating all actions in the offline dataset. We evaluate MOBODY on a wide range of MuJoCo and Adroit benchmarks, demonstrating that it outperforms state-of-the-art off-dynamics RL baselines as well as policy learning methods based on different dynamics learning baselines, with especially pronounced improvements in challenging scenarios where existing methods struggle.

**arXiv ID:** 2506.08460
</details>

<details>
<summary><strong>The Choice of Divergence: A Neglected Key to Mitigating Diversity Collapse in Reinforcement Learning with Verifiable Reward</strong> - Long Li, Jiaran Hao, Jason Klein Liu, Zhijian Zhou, Yanting Miao, Wei Pang, Xiaoyu Tan, Wei Chu, Zhe Wang, Shirui Pan, Chao Qu, Yuan Qi - [[pdf]](https://arxiv.org/pdf/2509.07430)</summary>

**Abstract:** A central paradox in fine-tuning Large Language Models (LLMs) with Reinforcement Learning with Verifiable Reward (RLVR) is the frequent degradation of multi-attempt performance (Pass@k) despite improvements in single-attempt accuracy (Pass@1). This is often accompanied by catastrophic forgetting, where models lose previously acquired skills. While various methods have been proposed, the choice and function of the divergence term have been surprisingly unexamined as a proactive solution. We argue that standard RLVR objectives -- both those using the mode-seeking reverse KL-divergence and those forgoing a divergence term entirely -- lack a crucial mechanism for knowledge retention. The reverse-KL actively accelerates this decay by narrowing the policy, while its absence provides no safeguard against the model drifting from its diverse knowledge base. We propose a fundamental shift in perspective: using the divergence term itself as the solution. Our framework, Diversity-Preserving Hybrid RL (DPH-RL), leverages mass-covering f-divergences (like forward-KL and JS-divergence) to function as a rehearsal mechanism. By continuously referencing the initial policy, this approach forces the model to maintain broad solution coverage. Extensive experiments on math and SQL generation demonstrate that DPH-RL not only resolves the Pass@k degradation but improves both Pass@1 and Pass@k in- and out-of-domain. Additionally, DPH-RL is more training-efficient because it computes f-divergence using generator functions, requiring only sampling from the initial policy and no online reference model. Our work highlights a crucial, overlooked axis for improving RLVR, demonstrating that the proper selection of a divergence measure is a powerful tool for building more general and diverse reasoning models.

**arXiv ID:** 2509.07430
</details>

<details>
<summary><strong>Perception Before Reasoning: Two-Stage Reinforcement Learning for Visual Reasoning in Vision-Language Models</strong> - Yan Chen, Long Li, Teng Xi, Long Zeng, Jingdong Wang - [[pdf]](https://arxiv.org/pdf/2509.13031)</summary>

**Abstract:** Reinforcement learning (RL) has proven highly effective in eliciting the reasoning capabilities of large language models (LLMs). Inspired by this success, recent studies have explored applying similar techniques to vision-language models (VLMs), aiming to enhance their reasoning performance. However, directly transplanting RL methods from LLMs to VLMs is suboptimal, as the tasks faced by VLMs are inherently more complex. Specifically, VLMs must first accurately perceive and understand visual inputs before reasoning can be effectively performed. To address this challenge, we propose a two-stage reinforcement learning framework designed to jointly enhance both the perceptual and reasoning capabilities of VLMs. To mitigate the vanishing advantage issue commonly observed in RL training, we first perform dataset-level sampling to selectively strengthen specific capabilities using distinct data sources. During training, the first stage focuses on improving the model's visual perception through coarse- and fine-grained visual understanding, while the second stage targets the enhancement of reasoning abilities. After the proposed two-stage reinforcement learning process, we obtain PeBR-R1, a vision-language model with significantly enhanced perceptual and reasoning capabilities. Experimental results on seven benchmark datasets demonstrate the effectiveness of our approach and validate the superior performance of PeBR-R1 across diverse visual reasoning tasks.

**arXiv ID:** 2509.13031
</details>

<details>
<summary><strong>Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers</strong> - Xin-Qiang Cai, Wei Wang, Feng Liu, Tongliang Liu, Gang Niu, Masashi Sugiyama - [[pdf]](https://arxiv.org/pdf/2510.00915)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) trains policies against automated verifiers to avoid costly human labeling. To reduce vulnerability to verifier hacking, many RLVR systems collapse rewards to binary $\{0,1\}$ during training. This choice carries a cost: it introduces \textit{false negatives} (rejecting correct answers, FNs) and \textit{false positives} (accepting incorrect ones, FPs). For instance, a rule-based checker may mark the correct fraction $\frac{12}{36}$ as wrong when compared against the canonical $\frac{1}{3}$ due to brittle parsing/equivalence rules (FN), while a large language model (LLM) judges can be gamed by superficial cues or even a single adversarial token, yielding inflated correctness for wrong solutions (FP). We formalize verifier unreliability by modeling the verifier as a stochastic reward channel with asymmetric noise rates. From this abstraction, we derive two correction algorithms for verifier errors. The first is a \textit{backward} correction that de-biases the observed binary reward to recover an \textit{unbiased} estimator of the clean policy gradient. The second is a \textit{forward} correction that reweights score-function terms so that the expected update direction aligns with the \textit{clean gradient}; notably, it requires only the FN rate. We implement both as lightweight hooks in a group relative policy optimization (GRPO)-based RLVR pipeline and evaluate them on math-reasoning models and benchmarks. Across models and datasets, both corrections improve over uncorrected training; the forward variant converges faster and remains stable under heavier noise. Finally, we show a practical appeal mechanism in which a lightweight LLM verifier estimates the FN rate online by rechecking rule-based negatives, obtaining outperformance compared with other state-of-the-art contenders.

**arXiv ID:** 2510.00915
</details>

<details>
<summary><strong>Grassroots Logic Programs: A Secure, Multiagent, Concurrent, Logic Programming Language</strong> - Ehud Shapiro - [[pdf]](https://arxiv.org/pdf/2510.15747)</summary>

**Abstract:** Grassroots platforms are distributed applications run by\linebreak cryptographically-identified people on their networked personal devices, where multiple disjoint platform instances emerge independently and coalesce when they interoperate. Their foundation is the grassroots social graph, upon which grassroots social networks, grassroots cryptocurrencies, and grassroots democratic federations can be built.
Grassroots platforms have yet to be implemented, the key challenge being faulty and malicious participants: without secure programming support, correct participants cannot reliably identify each other, establish secure communication, or verify each other's code integrity.
We present Grassroots Logic Programs (GLP), a secure, multiagent, concurrent, logic programming language for implementing grassroots platforms. GLP extends logic programs with paired single-reader/single-writer (SRSW) logic variables, providing secure communication channels among cryptographically-identified people through encrypted, signed and attested messages, which enable identity and code integrity verification. We present GLP progressively: logic programs, concurrent GLP, multiagent GLP, augmenting it with cryptographic security, and providing smartphone implementation-ready specifications. We prove safety properties including that GLP computations are deductions, SRSW preservation, acyclicity, and monotonicity. We prove multiagent GLP is grassroots and that GLP streams achieve blockchain security properties. We present a grassroots social graph protocol establishing authenticated peer-to-peer connections and demonstrate secure grassroots social networking applications.

**arXiv ID:** 2510.15747
</details>

<details>
<summary><strong>AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction</strong> - Hong Ting Tsang, Jiaxin Bai, Haoyu Huang, Qiao Xiao, Tianshi Zheng, Baixuan Xu, Shujie Liu, Yangqiu Song - [[pdf]](https://arxiv.org/pdf/2510.15339)</summary>

**Abstract:** Building effective knowledge graphs (KGs) for Retrieval-Augmented Generation (RAG) is pivotal for advancing question answering (QA) systems. However, its effectiveness is hindered by a fundamental disconnect: the knowledge graph (KG) construction process is decoupled from its downstream application, yielding suboptimal graph structures. To bridge this gap, we introduce AutoGraph-R1, the first framework to directly optimize KG construction for task performance using Reinforcement Learning (RL). AutoGraph-R1 trains an LLM constructor by framing graph generation as a policy learning problem, where the reward is derived from the graph's functional utility in a RAG pipeline. We design two novel, task-aware reward functions, one for graphs as knowledge carriers and another as knowledge indices. Across multiple QA benchmarks, AutoGraph-R1 consistently enables graph RAG methods to achieve significant performance gains over using task-agnostic baseline graphs. Our work shows it is possible to close the loop between construction and application, shifting the paradigm from building intrinsically ``good'' graphs to building demonstrably ``useful'' ones.

**arXiv ID:** 2510.15339
</details>

<details>
<summary><strong>Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing</strong> - Baode Wang, Biao Wu, Weizhen Li, Meng Fang, Zuming Huang, Jun Huang, Haozhe Wang, Yanjie Liang, Ling Chen, Wei Chu, Yuan Qi - [[pdf]](https://arxiv.org/pdf/2510.15349)</summary>

**Abstract:** Document parsing from scanned images into structured formats remains a significant challenge due to its complexly intertwined elements such as text paragraphs, figures, formulas, and tables. Existing supervised fine-tuning methods often struggle to generalize across diverse document types, leading to poor performance, particularly on out-of-distribution data. This issue is further exacerbated by the limited availability of high-quality training data for layout-aware parsing tasks. To address these challenges, we introduce LayoutRL, a reinforcement learning framework that optimizes layout understanding through composite rewards integrating normalized edit distance, paragraph count accuracy, and reading order preservation. To support this training, we construct the Infinity-Doc-400K dataset, which we use to train Infinity-Parser, a vision-language model demonstrating robust generalization across various domains. Extensive evaluations on benchmarks including OmniDocBench, olmOCR-Bench, PubTabNet, and FinTabNet show that Infinity-Parser consistently achieves state-of-the-art performance across a broad range of document types, languages, and structural complexities, substantially outperforming both specialized document parsing systems and general-purpose vision-language models. We will release our code, dataset, and model to facilitate reproducible research in document parsing.

**arXiv ID:** 2510.15349
</details>

<details>
<summary><strong>ES-C51: Expected Sarsa Based C51 Distributional Reinforcement Learning Algorithm</strong> - Rijul Tandon, Peter Vamplew, Cameron Foale - [[pdf]](https://arxiv.org/pdf/2510.15006)</summary>

**Abstract:** In most value-based reinforcement learning (RL) algorithms, the agent estimates only the expected reward for each action and selects the action with the highest reward. In contrast, Distributional Reinforcement Learning (DRL) estimates the entire probability distribution of possible rewards, providing richer information about uncertainty and variability. C51 is a popular DRL algorithm for discrete action spaces. It uses a Q-learning approach, where the distribution is learned using a greedy Bellman update. However, this can cause problems if multiple actions at a state have similar expected reward but with different distributions, as the algorithm may not learn a stable distribution. This study presents a modified version of C51 (ES-C51) that replaces the greedy Q-learning update with an Expected Sarsa update, which uses a softmax calculation to combine information from all possible actions at a state rather than relying on a single best action. This reduces instability when actions have similar expected rewards and allows the agent to learn higher-performing policies. This approach is evaluated on classic control environments from Gym, and Atari-10 games. For a fair comparison, we modify the standard C51's exploration strategy from e-greedy to softmax, which we refer to as QL-C51 (Q- Learning based C51). The results demonstrate that ES-C51 outperforms QL-C51 across many environments.

**arXiv ID:** 2510.15006
</details>

<details>
<summary><strong>Learn to Change the World: Multi-level Reinforcement Learning with Model-Changing Actions</strong> - Ziqing Lu, Babak Hassibi, Lifeng Lai, Weiyu Xu - [[pdf]](https://arxiv.org/pdf/2510.15056)</summary>

**Abstract:** Reinforcement learning usually assumes a given or sometimes even fixed environment in which an agent seeks an optimal policy to maximize its long-term discounted reward. In contrast, we consider agents that are not limited to passive adaptations: they instead have model-changing actions that actively modify the RL model of world dynamics itself. Reconfiguring the underlying transition processes can potentially increase the agents' rewards. Motivated by this setting, we introduce the multi-layer configurable time-varying Markov decision process (MCTVMDP). In an MCTVMDP, the lower-level MDP has a non-stationary transition function that is configurable through upper-level model-changing actions. The agent's objective consists of two parts: Optimize the configuration policies in the upper-level MDP and optimize the primitive action policies in the lower-level MDP to jointly improve its expected long-term reward.

**arXiv ID:** 2510.15056
</details>

<details>
<summary><strong>Dual-Weighted Reinforcement Learning for Generative Preference Modeling</strong> - Shengyu Feng, Yun He, Shuang Ma, Beibin Li, Yuanhao Xiong, Vincent Li, Karishma Mandyam, Julian Katz-Samuels, Shengjie Bi, Licheng Yu, Hejia Zhang, Karthik Abinav Sankararaman, Han Fang, Riham Mansour, Yiming Yang, Manaal Faruqui - [[pdf]](https://arxiv.org/pdf/2510.15242)</summary>

**Abstract:** Reinforcement learning (RL) has recently proven effective at scaling chain-of-thought (CoT) reasoning in large language models on tasks with verifiable answers. However, extending RL to more general non-verifiable tasks, typically in the format of human preference pairs, remains both challenging and underexplored. In this work, we propose Dual-Weighted Reinforcement Learning (DWRL), a new framework for preference modeling that integrates CoT reasoning with the Bradley-Terry (BT) model via a dual-weighted RL objective that preserves preference-modeling inductive bias. DWRL approximates the maximum-likelihood objective of the BT model with two complementary weights: an instance-wise misalignment weight, which emphasizes under-trained pairs misaligned with human preference, and a group-wise (self-normalized) conditional preference score, which promotes promising thoughts. In this paper, we apply DWRL to preference modeling by training generative preference models (GPMs) to first generate a thought and then predict the human preference score. Across multiple benchmarks and model scales (Llama3 and Qwen2.5), DWRL consistently outperforms both GPM baselines and scalar models, while producing coherent, interpretable thoughts. In summary, our results position DWRL as a general framework for reasoning-enhanced preference learning beyond verifiable tasks.

**arXiv ID:** 2510.15242
</details>

<details>
<summary><strong>Iterative Refinement of Flow Policies in Probability Space for Online Reinforcement Learning</strong> - Mingyang Sun, Pengxiang Ding, Weinan Zhang, Donglin Wang - [[pdf]](https://arxiv.org/pdf/2510.15388)</summary>

**Abstract:** While behavior cloning with flow/diffusion policies excels at learning complex skills from demonstrations, it remains vulnerable to distributional shift, and standard RL methods struggle to fine-tune these models due to their iterative inference process and the limitations of existing workarounds. In this work, we introduce the Stepwise Flow Policy (SWFP) framework, founded on the key insight that discretizing the flow matching inference process via a fixed-step Euler scheme inherently aligns it with the variational Jordan-Kinderlehrer-Otto (JKO) principle from optimal transport. SWFP decomposes the global flow into a sequence of small, incremental transformations between proximate distributions. Each step corresponds to a JKO update, regularizing policy changes to stay near the previous iterate and ensuring stable online adaptation with entropic regularization. This decomposition yields an efficient algorithm that fine-tunes pre-trained flows via a cascade of small flow blocks, offering significant advantages: simpler/faster training of sub-models, reduced computational/memory costs, and provable stability grounded in Wasserstein trust regions. Comprehensive experiments demonstrate SWFP's enhanced stability, efficiency, and superior adaptation performance across diverse robotic control benchmarks.

**arXiv ID:** 2510.15388
</details>

<details>
<summary><strong>Safe, Efficient, and Robust Reinforcement Learning for Ranking and Diffusion Models</strong> - Shashank Gupta - [[pdf]](https://arxiv.org/pdf/2510.15429)</summary>

**Abstract:** This dissertation investigates how reinforcement learning (RL) methods can be designed to be safe, sample-efficient, and robust. Framed through the unifying perspective of contextual-bandit RL, the work addresses two major application domains - ranking and recommendation, and text-to-image diffusion models. The first part of the thesis develops theory and algorithms for safe deployment in ranking systems. An exposure-based generalisation bound is derived, leading to a counterfactual risk-minimisation objective whose solution is guaranteed not to underperform the logging policy, even with sparse feedback. This guarantee is extended to doubly robust estimators, enabling safety even under adversarial or misspecified user models and offering practitioners explicit control over permissible utility loss. The second part turns to single-action bandits, where various off-policy estimators are unified within a baseline-correction framework. A closed-form optimal baseline is proposed and shown to minimise both evaluation and policy-gradient variance, thereby improving off-policy learning reliability. The final part examines the trade-offs between efficiency and effectiveness in generative RL. A systematic study of PPO and REINFORCE motivates the Leave-One-Out PPO (LOOP) algorithm, which combines multiple diffusion trajectories with a REINFORCE-style baseline inside PPO's clipped objective. LOOP achieves PPO-level sample efficiency while producing generations that align more faithfully with textual attributes.

**arXiv ID:** 2510.15429
</details>

<details>
<summary><strong>FIDDLE: Reinforcement Learning for Quantum Fidelity Enhancement</strong> - Hoang M. Ngo, Tamer Kahveci, My T. Thai - [[pdf]](https://arxiv.org/pdf/2510.15833)</summary>

**Abstract:** Quantum computing has the potential to revolutionize fields like quantum optimization and quantum machine learning. However, current quantum devices are hindered by noise, reducing their reliability. A key challenge in gate-based quantum computing is improving the reliability of quantum circuits, measured by process fidelity, during the transpilation process, particularly in the routing stage. In this paper, we address the Fidelity Maximization in Routing Stage (FMRS) problem by introducing FIDDLE, a novel learning framework comprising two modules: a Gaussian Process-based surrogate model to estimate process fidelity with limited training samples and a reinforcement learning module to optimize routing. Our approach is the first to directly maximize process fidelity, outperforming traditional methods that rely on indirect metrics such as circuit depth or gate count. We rigorously evaluate FIDDLE by comparing it with state-of-the-art fidelity estimation techniques and routing optimization methods. The results demonstrate that our proposed surrogate model is able to provide a better estimation on the process fidelity compared to existing learning techniques, and our end-to-end framework significantly improves the process fidelity of quantum circuits across various noise models.

**arXiv ID:** 2510.15833
</details>

<details>
<summary><strong>All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning</strong> - Gokul Swamy, Sanjiban Choudhury, Wen Sun, Zhiwei Steven Wu, J. Andrew Bagnell - [[pdf]](https://arxiv.org/pdf/2503.01067)</summary>

**Abstract:** From a first-principles perspective, it may seem odd that the strongest results in foundation model fine-tuning (FT) are achieved via a relatively complex, two-stage training procedure. Specifically, one first trains a reward model (RM) on some dataset (e.g., human preferences) before using it to provide online feedback as part of a downstream reinforcement learning (RL) procedure, rather than directly optimizing the policy parameters on said dataset via offline maximum likelihood estimation. In fact, from an information-theoretic perspective, we can only lose information via passing through a reward model and cannot create any new information via on-policy sampling. To explain this discrepancy, we scrutinize several hypotheses on the value of RL in FT through both theoretical and empirical lenses. Of the hypotheses considered, we find the most support for the explanation that on problems with a generation-verification gap, (1) it is relatively easy to learn the relatively simple RM (verifier) from the preference data. Then, (2) the downstream RL procedure only returns policies (generators) that are optimal for such relatively simple verifiers. Thus, end-to-end, two-stage online FT only has to search over a reduced subset of the full space of policies, requiring less data than offline FT.

**arXiv ID:** 2503.01067
</details>

<details>
<summary><strong>Autonomous Reactive Masonry Construction using Collaborative Heterogeneous Aerial Robots with Experimental Demonstration</strong> - Marios-Nektarios Stamatopoulos, Elias Small, Shridhar Velhal, Avijit Banerjee, George Nikolakopoulos - [[pdf]](https://arxiv.org/pdf/2510.15114)</summary>

**Abstract:** This article presents a fully autonomous aerial masonry construction framework using heterogeneous unmanned aerial vehicles (UAVs), supported by experimental validation. Two specialized UAVs were developed for the task: (i) a brick-carrier UAV equipped with a ball-joint actuation mechanism for precise brick manipulation, and (ii) an adhesion UAV integrating a servo-controlled valve and extruder nozzle for accurate adhesion application. The proposed framework employs a reactive mission planning unit that combines a dependency graph of the construction layout with a conflict graph to manage simultaneous task execution, while hierarchical state machines ensure robust operation and safe transitions during task execution. Dynamic task allocation allows real-time adaptation to environmental feedback, while minimum-jerk trajectory generation ensures smooth and precise UAV motion during brick pickup and placement. Additionally, the brick-carrier UAV employs an onboard vision system that estimates brick poses in real time using ArUco markers and a least-squares optimization filter, enabling accurate alignment during construction. To the best of the authors' knowledge, this work represents the first experimental demonstration of fully autonomous aerial masonry construction using heterogeneous UAVs, where one UAV precisely places the bricks while another autonomously applies adhesion material between them. The experimental results supported by the video showcase the effectiveness of the proposed framework and demonstrate its potential to serve as a foundation for future developments in autonomous aerial robotic construction.

**arXiv ID:** 2510.15114
</details>

<details>
<summary><strong>RM-RL: Role-Model Reinforcement Learning for Precise Robot Manipulation</strong> - Xiangyu Chen, Chuhao Zhou, Yuxi Liu, Jianfei Yang - [[pdf]](https://arxiv.org/pdf/2510.15189)</summary>

**Abstract:** Precise robot manipulation is critical for fine-grained applications such as chemical and biological experiments, where even small errors (e.g., reagent spillage) can invalidate an entire task. Existing approaches often rely on pre-collected expert demonstrations and train policies via imitation learning (IL) or offline reinforcement learning (RL). However, obtaining high-quality demonstrations for precision tasks is difficult and time-consuming, while offline RL commonly suffers from distribution shifts and low data efficiency. We introduce a Role-Model Reinforcement Learning (RM-RL) framework that unifies online and offline training in real-world environments. The key idea is a role-model strategy that automatically generates labels for online training data using approximately optimal actions, eliminating the need for human demonstrations. RM-RL reformulates policy learning as supervised training, reducing instability from distribution mismatch and improving efficiency. A hybrid training scheme further leverages online role-model data for offline reuse, enhancing data efficiency through repeated sampling. Extensive experiments show that RM-RL converges faster and more stably than existing RL methods, yielding significant gains in real-world manipulation: 53% improvement in translation accuracy and 20% in rotation accuracy. Finally, we demonstrate the successful execution of a challenging task, precisely placing a cell plate onto a shelf, highlighting the framework's effectiveness where prior methods fail.

**arXiv ID:** 2510.15189
</details>

<details>
<summary><strong>VDRive: Leveraging Reinforced VLA and Diffusion Policy for End-to-end Autonomous Driving</strong> - Ziang Guo, Zufeng Zhang - [[pdf]](https://arxiv.org/pdf/2510.15446)</summary>

**Abstract:** In autonomous driving, dynamic environment and corner cases pose significant challenges to the robustness of ego vehicle's state understanding and decision making. We introduce VDRive, a novel pipeline for end-to-end autonomous driving that explicitly models state-action mapping to address these challenges, enabling interpretable and robust decision making. By leveraging the advancement of the state understanding of the Vision Language Action Model (VLA) with generative diffusion policy-based action head, our VDRive guides the driving contextually and geometrically. Contextually, VLA predicts future observations through token generation pre-training, where the observations are represented as discrete codes by a Conditional Vector Quantized Variational Autoencoder (CVQ-VAE). Geometrically, we perform reinforcement learning fine-tuning of the VLA to predict future trajectories and actions based on current driving conditions. VLA supplies the current state tokens and predicted state tokens for the action policy head to generate hierarchical actions and trajectories. During policy training, a learned critic evaluates the actions generated by the policy and provides gradient-based feedback, forming an actor-critic framework that enables a reinforcement-based policy learning pipeline. Experiments show that our VDRive achieves state-of-the-art performance in the Bench2Drive closed-loop benchmark and nuScenes open-loop planning.

**arXiv ID:** 2510.15446
</details>

<details>
<summary><strong>HEADER: Hierarchical Robot Exploration via Attention-Based Deep Reinforcement Learning with Expert-Guided Reward</strong> - Yuhong Cao, Yizhuo Wang, Jingsong Liang, Shuhao Liao, Yifeng Zhang, Peizhuo Li, Guillaume Sartoretti - [[pdf]](https://arxiv.org/pdf/2510.15679)</summary>

**Abstract:** This work pushes the boundaries of learning-based methods in autonomous robot exploration in terms of environmental scale and exploration efficiency. We present HEADER, an attention-based reinforcement learning approach with hierarchical graphs for efficient exploration in large-scale environments. HEADER follows existing conventional methods to construct hierarchical representations for the robot belief/map, but further designs a novel community-based algorithm to construct and update a global graph, which remains fully incremental, shape-adaptive, and operates with linear complexity. Building upon attention-based networks, our planner finely reasons about the nearby belief within the local range while coarsely leveraging distant information at the global scale, enabling next-best-viewpoint decisions that consider multi-scale spatial dependencies. Beyond novel map representation, we introduce a parameter-free privileged reward that significantly improves model performance and produces near-optimal exploration behaviors, by avoiding training objective bias caused by handcrafted reward shaping. In simulated challenging, large-scale exploration scenarios, HEADER demonstrates better scalability than most existing learning and non-learning methods, while achieving a significant improvement in exploration efficiency (up to 20%) over state-of-the-art baselines. We also deploy HEADER on hardware and validate it in complex, large-scale real-life scenarios, including a 300m*230m campus environment.

**arXiv ID:** 2510.15679
</details>

<details>
<summary><strong>Learning to Capture Rocks using an Excavator: A Reinforcement Learning Approach with Guiding Reward Formulation</strong> - Amirmasoud Molaei, Mohammad Heravi, Reza Ghabcheloo - [[pdf]](https://arxiv.org/pdf/2510.04168)</summary>

**Abstract:** Rock capturing with standard excavator buckets is a challenging task typically requiring the expertise of skilled operators. Unlike soil digging, it involves manipulating large, irregular rocks in unstructured environments where complex contact interactions with granular material make model-based control impractical. Existing autonomous excavation methods focus mainly on continuous media or rely on specialized grippers, limiting their applicability to real-world construction sites. This paper introduces a fully data-driven control framework for rock capturing that eliminates the need for explicit modeling of rock or soil properties. A model-free reinforcement learning agent is trained in the AGX Dynamics simulator using the Proximal Policy Optimization (PPO) algorithm and a guiding reward formulation. The learned policy outputs joint velocity commands directly to the boom, arm, and bucket of a CAT365 excavator model. Robustness is enhanced through extensive domain randomization of rock geometry, density, and mass, as well as the initial configurations of the bucket, rock, and goal position. To the best of our knowledge, this is the first study to develop and evaluate an RL-based controller for the rock capturing task. Experimental results show that the policy generalizes well to unseen rocks and varying soil conditions, achieving high success rates comparable to those of human participants while maintaining machine stability. These findings demonstrate the feasibility of learning-based excavation strategies for discrete object manipulation without requiring specialized hardware or detailed material models.

**arXiv ID:** 2510.04168
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
