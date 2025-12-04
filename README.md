# Agent arXiv Daily

**Last Updated:** 2025-12-04 02:55:36

**Total Papers:** 70

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (5 papers)</h2></summary>

<details>
<summary><strong>Autonomous Agents and Policy Compliance: A Framework for Reasoning About Penalties</strong> - Vineel Tummala, Daniela Inclezan - [[pdf]](https://arxiv.org/pdf/2512.03931)</summary>

**Abstract:** This paper presents a logic programming-based framework for policy-aware autonomous agents that can reason about potential penalties for non-compliance and act accordingly. While prior work has primarily focused on ensuring compliance, our approach considers scenarios where deviating from policies may be necessary to achieve high-stakes goals. Additionally, modeling non-compliant behavior can assist policymakers by simulating realistic human decision-making. Our framework extends Gelfond and Lobo's Authorization and Obligation Policy Language (AOPL) to incorporate penalties and integrates Answer Set Programming (ASP) for reasoning. Compared to previous approaches, our method ensures well-formed policies, accounts for policy priorities, and enhances explainability by explicitly identifying rule violations and their consequences. Building on the work of Harders and Inclezan, we introduce penalty-based reasoning to distinguish between non-compliant plans, prioritizing those with minimal repercussions. To support this, we develop an automated translation from the extended AOPL into ASP and refine ASP-based planning algorithms to account for incurred penalties. Experiments in two domains demonstrate that our framework generates higher-quality plans that avoid harmful actions while, in some cases, also improving computational efficiency. These findings underscore its potential for enhancing autonomous decision-making and informing policy refinement. Under consideration in Theory and Practice of Logic Programming (TPLP).

**arXiv ID:** 2512.03931
</details>

<details>
<summary><strong>Young children's anthropomorphism of an AI chatbot: Brain activation and the role of parent co-presence</strong> - Pilyoung Kim, Jenna H. Chin, Yun Xie, Nolan Brady, Tom Yeh, Sujin Yang - [[pdf]](https://arxiv.org/pdf/2512.02179)</summary>

**Abstract:** Artificial Intelligence (AI) chatbots powered by a large language model (LLM) are entering young children's learning and play, yet little is known about how young children construe these agents or how such construals relate to engagement. We examined anthropomorphism of a social AI chatbot during collaborative storytelling and asked how children's attributions related to their behavior and prefrontal activation. Children at ages 5-6 (N = 23) completed three storytelling sessions: interacting with (1) an AI chatbot only, (2) a parent only, and (3) the AI and a parent together. After the sessions, children completed an interview assessing anthropomorphism toward both the AI chatbot and the parent. Behavioral engagement was indexed by the conversational turn count (CTC) ratio, and concurrent fNIRS measured oxygenated hemoglobin in bilateral vmPFC and dmPFC regions. Children reported higher anthropomorphism for parents than for the AI chatbot overall, although AI ratings were relatively high for perceptive abilities and epistemic states. Anthropomorphism was not associated with CTC. In the right dmPFC, higher perceptive scores were associated with greater activation during the AI-only condition and with lower activation during the AI+Parent condition. Exploratory analyses indicated that higher dmPFC activation during the AI-only condition correlated with higher end-of-session "scared" mood ratings. Findings suggest that stronger perceptive anthropomorphism can be associated with greater brain activation related to interpreting the AI's mental states, whereas parent co-presence may help some children interpret and regulate novel AI interactions. These results may have design implications for encouraging parent-AI co-use in early childhood.

**arXiv ID:** 2512.02179
</details>

<details>
<summary><strong>Are you a robot? Detecting Autonomous Vehicles from Behavior Analysis</strong> - Fabio Maresca, Filippo Grazioli, Antonio Albanese, Vincenzo Sciancalepore, Gianpiero Negri, Xavier Costa-Perez - [[pdf]](https://arxiv.org/pdf/2403.09571)</summary>

**Abstract:** The tremendous hype around autonomous driving is eagerly calling for emerging and novel technologies to support advanced mobility use cases. As car manufactures keep developing SAE level 3+ systems to improve the safety and comfort of passengers, traffic authorities need to establish new procedures to manage the transition from human-driven to fully-autonomous vehicles while providing a feedback-loop mechanism to fine-tune envisioned autonomous systems. Thus, a way to automatically profile autonomous vehicles and differentiate those from human-driven ones is a must. In this paper, we present a fully-fledged framework that monitors active vehicles using camera images and state information in order to determine whether vehicles are autonomous, without requiring any active notification from the vehicles themselves. Essentially, it builds on the cooperation among vehicles, which share their data acquired on the road feeding a machine learning model to identify autonomous cars. We extensively tested our solution and created the NexusStreet dataset, by means of the CARLA simulator, employing an autonomous driving control agent and a steering wheel maneuvered by licensed drivers. Experiments show it is possible to discriminate the two behaviors by analyzing video clips with an accuracy of 80%, which improves up to 93% when the target state information is available. Lastly, we deliberately degraded the state to observe how the framework performs under non-ideal data collection conditions.

**arXiv ID:** 2403.09571
</details>

<details>
<summary><strong>A Modular Architecture Design for Autonomous Driving Racing in Controlled Environments</strong> - Brais Fontan-Costas, M. Diaz-Cacho, Ruben Fernandez-Boullon, Manuel Alonso-Carracedo, Javier Perez-Robles - [[pdf]](https://arxiv.org/pdf/2512.03886)</summary>

**Abstract:** This paper presents an Autonomous System (AS) architecture for vehicles in a closed circuit. The AS performs precision tasks including computer vision for environment perception, positioning and mapping for accurate localization, path planning for optimal trajectory generation, and control for precise vehicle actuation. Each subsystem operates independently while connecting data through a cohesive pipeline architecture. The system implements a modular design that combines state-of-the-art technologies for real-time autonomous navigation in controlled environments.

**arXiv ID:** 2512.03886
</details>

<details>
<summary><strong>Left shifting analysis of Human-Autonomous Team interactions to analyse risks of autonomy in high-stakes AI systems</strong> - Ben Larwood, Oliver J. Sutton, Callum Cockburn - [[pdf]](https://arxiv.org/pdf/2512.03519)</summary>

**Abstract:** Developing high-stakes autonomous systems that include Artificial Intelligence (AI) components is complex; the consequences of errors can be catastrophic, yet it is challenging to plan for all operational cases. In stressful scenarios for the human operator, such as short decision-making timescales, the risk of failures is exacerbated. A lack of understanding of AI failure modes obstructs this and so blocks the robust implementation of applications of AI in smart systems. This prevents early risk identification, leading to increased time, risk and cost of projects.
A key tenet of Systems Engineering and acquisition engineering is centred around a "left-shift" in test and evaluation activities to earlier in the system lifecycle, to allow for "accelerated delivery of [systems] that work". We argue it is therefore essential that this shift includes the analysis of AI failure cases as part of the design stages of the system life cycle. Our proposed framework enables the early characterisation of risks emerging from human-autonomy teaming (HAT) in operational contexts. The cornerstone of this is a new analysis of AI failure modes, built on the seminal modelling of human-autonomy teams laid out by LaMonica et al., 2022. Using the analysis of the interactions between human and autonomous systems and exploring the failure modes within each aspect, our approach provides a way to systematically identify human-AI interactions risks across the operational domain of the system of interest. The understanding of the emergent behaviour enables increased robustness of the system, for which the analysis should be undertaken over the whole scope of its operational design domain. This approach is illustrated through an example use case for an AI assistant supporting a Command & Control (C2) System.

**arXiv ID:** 2512.03519
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>A Hierarchical Tree-based approach for creating Configurable and Static Deep Research Agent (Static-DRA)</strong> - Saurav Prateek - [[pdf]](https://arxiv.org/pdf/2512.03887)</summary>

**Abstract:** The advancement in Large Language Models has driven the creation of complex agentic systems, such as Deep Research Agents (DRAs), to overcome the limitations of static Retrieval Augmented Generation (RAG) pipelines in handling complex, multi-turn research tasks. This paper introduces the Static Deep Research Agent (Static-DRA), a novel solution built upon a configurable and hierarchical Tree-based static workflow.
The core contribution is the integration of two user-tunable parameters, Depth and Breadth, which provide granular control over the research intensity. This design allows end-users to consciously balance the desired quality and comprehensiveness of the research report against the associated computational cost of Large Language Model (LLM) interactions. The agent's architecture, comprising Supervisor, Independent, and Worker agents, facilitates effective multi-hop information retrieval and parallel sub-topic investigation.
We evaluate the Static-DRA against the established DeepResearch Bench using the RACE (Reference-based Adaptive Criteria-driven Evaluation) framework. Configured with a depth of 2 and a breadth of 5, and powered by the gemini-2.5-pro model, the agent achieved an overall score of 34.72. Our experiments validate that increasing the configured Depth and Breadth parameters results in a more in-depth research process and a correspondingly higher evaluation score. The Static-DRA offers a pragmatic and resource-aware solution, empowering users with transparent control over the deep research process. The entire source code, outputs and benchmark results are open-sourced at this https URL

**arXiv ID:** 2512.03887
</details>

<details>
<summary><strong>Password-Activated Shutdown Protocols for Misaligned Frontier Agents</strong> - Kai Williams, Rohan Subramani, Francis Rhys Ward - [[pdf]](https://arxiv.org/pdf/2512.03089)</summary>

**Abstract:** Frontier AI developers may fail to align or control highly-capable AI agents. In many cases, it could be useful to have emergency shutdown mechanisms which effectively prevent misaligned agents from carrying out harmful actions in the world. We introduce password-activated shutdown protocols (PAS protocols) -- methods for designing frontier agents to implement a safe shutdown protocol when given a password. We motivate PAS protocols by describing intuitive use-cases in which they mitigate risks from misaligned systems that subvert other control efforts, for instance, by disabling automated monitors or self-exfiltrating to external data centres. PAS protocols supplement other safety efforts, such as alignment fine-tuning or monitoring, contributing to defence-in-depth against AI risk. We provide a concrete demonstration in SHADE-Arena, a benchmark for AI monitoring and subversion capabilities, in which PAS protocols supplement monitoring to increase safety with little cost to performance. Next, PAS protocols should be robust to malicious actors who want to bypass shutdown. Therefore, we conduct a red-team blue-team game between the developers (blue-team), who must implement a robust PAS protocol, and a red-team trying to subvert the protocol. We conduct experiments in a code-generation setting, finding that there are effective strategies for the red-team, such as using another model to filter inputs, or fine-tuning the model to prevent shutdown behaviour. We then outline key challenges to implementing PAS protocols in real-life systems, including: security considerations of the password and decisions regarding when, and in which systems, to use them. PAS protocols are an intuitive mechanism for increasing the safety of frontier AI. We encourage developers to consider implementing PAS protocols prior to internal deployment of particularly dangerous systems to reduce loss-of-control risks.

**arXiv ID:** 2512.03089
</details>

<details>
<summary><strong>E-valuator: Reliable Agent Verifiers with Sequential Hypothesis Testing</strong> - Shuvom Sadhuka, Drew Prinster, Clara Fannjiang, Gabriele Scalia, Aviv Regev, Hanchen Wang - [[pdf]](https://arxiv.org/pdf/2512.03109)</summary>

**Abstract:** Agentic AI systems execute a sequence of actions, such as reasoning steps or tool calls, in response to a user prompt. To evaluate the success of their trajectories, researchers have developed verifiers, such as LLM judges and process-reward models, to score the quality of each action in an agent's trajectory. Although these heuristic scores can be informative, there are no guarantees of correctness when used to decide whether an agent will yield a successful output. Here, we introduce e-valuator, a method to convert any black-box verifier score into a decision rule with provable control of false alarm rates. We frame the problem of distinguishing successful trajectories (that is, a sequence of actions that will lead to a correct response to the user's prompt) and unsuccessful trajectories as a sequential hypothesis testing problem. E-valuator builds on tools from e-processes to develop a sequential hypothesis test that remains statistically valid at every step of an agent's trajectory, enabling online monitoring of agents over arbitrarily long sequences of actions. Empirically, we demonstrate that e-valuator provides greater statistical power and better false alarm rate control than other strategies across six datasets and three agents. We additionally show that e-valuator can be used for to quickly terminate problematic trajectories and save tokens. Together, e-valuator provides a lightweight, model-agnostic framework that converts verifier heuristics into decisions rules with statistical guarantees, enabling the deployment of more reliable agentic systems.

**arXiv ID:** 2512.03109
</details>

<details>
<summary><strong>AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning</strong> - Jiayi Zhang, Yiran Peng, Fanqi Kong, Cheng Yang, Yifan Wu, Zhaoyang Yu, Jinyu Xiang, Jianhao Ruan, Jinlin Wang, Maojia Song, HongZhang Liu, Xiangru Tang, Bang Liu, Chenglin Wu, Yuyu Luo - [[pdf]](https://arxiv.org/pdf/2511.19304)</summary>

**Abstract:** Humans naturally adapt to diverse environments by learning underlying rules across worlds with different dynamics, observations, and reward structures. In contrast, existing agents typically demonstrate improvements via self-evolving within a single domain, implicitly assuming a fixed environment distribution. Cross-environment learning has remained largely unmeasured: there is no standard collection of controllable, heterogeneous environments, nor a unified way to represent how agents learn. We address these gaps in two steps. First, we propose AutoEnv, an automated framework that treats environments as factorizable distributions over transitions, observations, and rewards, enabling low-cost (4.12 USD on average) generation of heterogeneous worlds. Using AutoEnv, we construct AutoEnv-36, a dataset of 36 environments with 358 validated levels, on which seven language models achieve 12-49% normalized reward, demonstrating the challenge of AutoEnv-36. Second, we formalize agent learning as a component-centric process driven by three stages of Selection, Optimization, and Evaluation applied to an improvable agent component. Using this formulation, we design eight learning methods and evaluate them on AutoEnv-36. Empirically, the gain of any single learning method quickly decrease as the number of environments increases, revealing that fixed learning methods do not scale across heterogeneous environments. Environment-adaptive selection of learning methods substantially improves performance but exhibits diminishing returns as the method space expands. These results highlight both the necessity and the current limitations of agent learning for scalable cross-environment generalization, and position AutoEnv and AutoEnv-36 as a testbed for studying cross-environment agent learning. The code is avaiable at this https URL.

**arXiv ID:** 2511.19304
</details>

<details>
<summary><strong>VICoT-Agent: A Vision-Interleaved Chain-of-Thought Framework for Interpretable Multimodal Reasoning and Scalable Remote Sensing Analysis</strong> - Chujie Wang, Zhiyuan Luo, Ruiqi Liu, Can Ran, Shenghua Fan, Xi Chen, Chu He - [[pdf]](https://arxiv.org/pdf/2511.20085)</summary>

**Abstract:** The current remote sensing image analysis task is increasingly evolving from traditional object recognition to complex intelligence reasoning, which places higher requirements on the model's reasoning ability and the flexibility of tool invocation. To this end, we propose a new multimodal agent framework, Vision-Interleaved Chain-of-Thought Framework (VICoT), which implements explicit multi-round reasoning by dynamically incorporating visual tools into the chain of thought. Through a stack-based reasoning structure and a modular MCP-compatible tool suite, VICoT enables LLMs to efficiently perform multi-round, interleaved vision-language reasoning tasks with strong generalization and this http URL also propose the Reasoning Stack distillation method to migrate complex Agent behaviors to small, lightweight models, which ensures the reasoning capability while significantly reducing complexity. Experiments on multiple remote sensing benchmarks demonstrate that VICoT significantly outperforms existing SOTA frameworks in reasoning transparency, execution efficiency, and generation quality.

**arXiv ID:** 2511.20085
</details>

<details>
<summary><strong>Real-Time Procedural Learning From Experience for AI Agents</strong> - Dasheng Bi, Yubin Hu, Mohammed N. Nasir - [[pdf]](https://arxiv.org/pdf/2511.22074)</summary>

**Abstract:** Learning how to do things from trial and error in real time is a hallmark of biological intelligence, yet most LLM-based agents lack mechanisms to acquire procedural knowledge after deployment. We propose Procedural Recall for Agents with eXperiences Indexed by State (PRAXIS), a lightweight post-training learning mechanism that stores the consequences of actions and retrieves them by jointly matching environmental and internal states of past episodes to the current state. PRAXIS augments agentic action selection with retrieved state-action-result exemplars that are generated in real time. When evaluated on the REAL web browsing benchmark, PRAXIS improves task completion accuracy, reliability, and cost efficiency across different foundation model backbones, and shows preliminary generalization to unseen tasks in similar environments. These results demonstrate that PRAXIS enables the practical adoption of AI agents in fast-evolving stateful environments by helping them learn new procedures effectively.

**arXiv ID:** 2511.22074
</details>

<details>
<summary><strong>PerfBench: Can Agents Resolve Real-World Performance Bugs?</strong> - Spandan Garg, Roshanak Zilouchian Moghaddam, Neel Sundaresan - [[pdf]](https://arxiv.org/pdf/2509.24091)</summary>

**Abstract:** Performance bugs are inefficiencies in software that waste computational resources without causing functional failures, making them particularly challenging to detect and fix. While recent advances in Software Engineering agents have shown promise in automated bug fixing, existing benchmarks primarily focus on functional correctness and fail to evaluate agents' abilities to identify and resolve non-functional issues like performance bugs. We introduce PerfBench, a benchmark comprising 81 real-world performance bug-fixing tasks from popular .NET repositories on GitHub. Unlike existing benchmarks that rely on pre-existing test suites, PerfBench features a novel evaluation harness that allows agents to generate their own performance benchmarks and validates fixes by comparing execution metrics collected for developer fix and agent fix. Each task in PerfBench is derived from actual developer fixes linked to performance-related issues, which are then verified by human experts, ensuring real-world relevance. Our evaluation reveals that current state-of-the-art coding agents struggle with performance optimization tasks, with baseline OpenHands agent achieving only a ~3% success rate on our benchmark. We develop OpenHands-Perf-Agent, which incorporates performance-aware tooling and instructions and achieves a ~20% success rate on the benchmark. We show that by ensuring the agent has proper instructions to benchmark its changes and tooling for benchmark output processing, we can improve the agent performance significantly, but room for improvement still remains. PerfBench provides a challenging test set for furthering the capabilities of agents in fixing performance issues.

**arXiv ID:** 2509.24091
</details>

<details>
<summary><strong>AGENTSAFE: A Unified Framework for Ethical Assurance and Governance in Agentic AI</strong> - Rafflesia Khan, Declan Joyce, Mansura Habiba - [[pdf]](https://arxiv.org/pdf/2512.03180)</summary>

**Abstract:** The rapid deployment of large language model (LLM)-based agents introduces a new class of risks, driven by their capacity for autonomous planning, multi-step tool integration, and emergent interactions. It raises some risk factors for existing governance approaches as they remain fragmented: Existing frameworks are either static taxonomies driven; however, they lack an integrated end-to-end pipeline from risk identification to operational assurance, especially for an agentic platform. We propose AGENTSAFE, a practical governance framework for LLM-based agentic systems. The framework operationalises the AI Risk Repository into design, runtime, and audit controls, offering a governance framework for risk identification and assurance. The proposed framework, AGENTSAFE, profiles agentic loops (plan -> act -> observe -> reflect) and toolchains, and maps risks onto structured taxonomies extended with agent-specific vulnerabilities. It introduces safeguards that constrain risky behaviours, escalates high-impact actions to human oversight, and evaluates systems through pre-deployment scenario banks spanning security, privacy, fairness, and systemic safety. During deployment, AGENTSAFE ensures continuous governance through semantic telemetry, dynamic authorization, anomaly detection, and interruptibility mechanisms. Provenance and accountability are reinforced through cryptographic tracing and organizational controls, enabling measurable, auditable assurance across the lifecycle of agentic AI systems. The key contributions of this paper are: (1) a unified governance framework that translates risk taxonomies into actionable design, runtime, and audit controls; (2) an Agent Safety Evaluation methodology that provides measurable pre-deployment assurance; and (3) a set of runtime governance and accountability mechanisms that institutionalise trust in agentic AI ecosystems.

**arXiv ID:** 2512.03180
</details>

<details>
<summary><strong>Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks</strong> - Songwen Zhao, Danqing Wang, Kexun Zhang, Jiaxuan Luo, Zhuo Li, Lei Li - [[pdf]](https://arxiv.org/pdf/2512.03262)</summary>

**Abstract:** Vibe coding is a new programming paradigm in which human engineers instruct large language model (LLM) agents to complete complex coding tasks with little supervision. Although it is increasingly adopted, are vibe coding outputs really safe to deploy in production? To answer this question, we propose SU S VI B E S, a benchmark consisting of 200 feature-request software engineering tasks from real-world open-source projects, which, when given to human programmers, led to vulnerable implementations. We evaluate multiple widely used coding agents with frontier models on this benchmark. Disturbingly, all agents perform poorly in terms of software security. Although 61% of the solutions from SWE-Agent with Claude 4 Sonnet are functionally correct, only 10.5% are secure. Further experiments demonstrate that preliminary security strategies, such as augmenting the feature request with vulnerability hints, cannot mitigate these security issues. Our findings raise serious concerns about the widespread adoption of vibe-coding, particularly in security-sensitive applications.

**arXiv ID:** 2512.03262
</details>

<details>
<summary><strong>SpeechRole: A Large-Scale Dataset and Benchmark for Evaluating Speech Role-Playing Agents</strong> - Changhao Jiang, Jiajun Sun, Yifei Cao, Jiabao Zhuang, Hui Li, Baoyu Fan, Tao Ji, Tao Gui, Qi Zhang - [[pdf]](https://arxiv.org/pdf/2508.02013)</summary>

**Abstract:** Recently, role-playing agents have emerged as a promising paradigm for achieving personalized interaction and emotional resonance. Existing research primarily focuses on the textual modality, neglecting the critical dimension of speech in realistic interactive scenarios. In particular, there is a lack of systematic evaluation for Speech Role-Playing Agents (SRPAs). To address this gap, we construct SpeechRole-Data, a large-scale, high-quality dataset that comprises 98 diverse roles and 112k speech-based single-turn and multi-turn conversations. Each role demonstrates distinct vocal characteristics, including timbre and prosody, thereby enabling more sophisticated speech role-playing. Furthermore, we propose SpeechRole-Eval, a multidimensional evaluation benchmark that systematically assesses SRPAs performance in key aspects such as fundamental interaction ability, speech expressiveness, and role-playing fidelity. Experimental results reveal the advantages and challenges of both cascaded and end-to-end speech role-playing agents in maintaining vocal style consistency and role coherence. We release all data, code, and baseline models to provide a solid foundation for speech-driven multimodal role-playing research and to foster further developments in this field.

**arXiv ID:** 2508.02013
</details>

<details>
<summary><strong>LargeAD: Large-Scale Cross-Sensor Data Pretraining for Autonomous Driving</strong> - Lingdong Kong, Xiang Xu, Youquan Liu, Jun Cen, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu - [[pdf]](https://arxiv.org/pdf/2501.04005)</summary>

**Abstract:** Recent advancements in vision foundation models (VFMs) have revolutionized visual perception in 2D, yet their potential for 3D scene understanding, particularly in autonomous driving applications, remains underexplored. In this paper, we introduce LargeAD, a versatile and scalable framework designed for large-scale 3D pretraining across diverse real-world driving datasets. Our framework leverages VFMs to extract semantically rich superpixels from 2D images, which are aligned with LiDAR point clouds to generate high-quality contrastive samples. This alignment facilitates cross-modal representation learning, enhancing the semantic consistency between 2D and 3D data. We introduce several key innovations: (i) VFM-driven superpixel generation for detailed semantic representation, (ii) a VFM-assisted contrastive learning strategy to align multimodal features, (iii) superpoint temporal consistency to maintain stable representations across time, and (iv) multi-source data pretraining to generalize across various LiDAR configurations. Our approach achieves substantial gains over state-of-the-art methods in linear probing and fine-tuning for LiDAR-based segmentation and object detection. Extensive experiments on 11 large-scale multi-sensor datasets highlight our superior performance, demonstrating adaptability, efficiency, and robustness in real-world autonomous driving scenarios.

**arXiv ID:** 2501.04005
</details>

<details>
<summary><strong>CSMapping: Scalable Crowdsourced Semantic Mapping and Topology Inference for Autonomous Driving</strong> - Zhijian Qiao, Zehuan Yu, Tong Li, Chih-Chung Chou, Wenchao Ding, Shaojie Shen - [[pdf]](https://arxiv.org/pdf/2512.03510)</summary>

**Abstract:** Crowdsourcing enables scalable autonomous driving map construction, but low-cost sensor noise hinders quality from improving with data volume. We propose CSMapping, a system that produces accurate semantic maps and topological road centerlines whose quality consistently increases with more crowdsourced data. For semantic mapping, we train a latent diffusion model on HD maps (optionally conditioned on SD maps) to learn a generative prior of real-world map structure, without requiring paired crowdsourced/HD-map supervision. This prior is incorporated via constrained MAP optimization in latent space, ensuring robustness to severe noise and plausible completion in unobserved areas. Initialization uses a robust vectorized mapping module followed by diffusion inversion; optimization employs efficient Gaussian-basis reparameterization, projected gradient descent zobracket multi-start, and latent-space factor-graph for global consistency. For topological mapping, we apply confidence-weighted k-medoids clustering and kinematic refinement to trajectories, yielding smooth, human-like centerlines robust to trajectory variation. Experiments on nuScenes, Argoverse 2, and a large proprietary dataset achieve state-of-the-art semantic and topological mapping performance, with thorough ablation and scalability studies.

**arXiv ID:** 2512.03510
</details>

<details>
<summary><strong>Magnetic Tactile-Driven Soft Actuator for Intelligent Grasping and Firmness Evaluation</strong> - Chengjin Du, Federico Bernabei, Zhengyin Du, Sergio Decherchi, Matteo Lo Preti, Lucia Beccai - [[pdf]](https://arxiv.org/pdf/2512.00907)</summary>

**Abstract:** Soft robots are powerful tools for manipulating delicate objects, yet their adoption is hindered by two gaps: the lack of integrated tactile sensing and sensor signal distortion caused by actuator deformations. This paper addresses these challenges by introducing the SoftMag actuator: a magnetic tactile-sensorized soft actuator. Unlike systems relying on attached sensors or treating sensing and actuation separately, SoftMag unifies them through a shared architecture while confronting the mechanical parasitic effect, where deformations corrupt tactile signals. A multiphysics simulation framework models this coupling, and a neural-network-based decoupling strategy removes the parasitic component, restoring sensing fidelity. Experiments including indentation, quasi-static and step actuation, and fatigue tests validate the actuator's performance and decoupling effectiveness. Building upon this foundation, the system is extended into a two-finger SoftMag gripper, where a multi-task neural network enables real-time prediction of tri-axial contact forces and position. Furthermore, a probing-based strategy estimates object firmness during grasping. Validation on apricots shows a strong correlation (Pearson r over 0.8) between gripper-estimated firmness and reference measurements, confirming the system's capability for non-destructive quality assessment. Results demonstrate that combining integrated magnetic sensing, learning-based correction, and real-time inference enables a soft robotic platform that adapts its grasp and quantifies material properties. The framework offers an approach for advancing sensorized soft actuators toward intelligent, material-aware robotics.

**arXiv ID:** 2512.00907
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>Benchmark for Planning and Control with Large Language Model Agents: Blocksworld with Model Context Protocol</strong> - Niklas Jobs, Luis Miguel Vieira da Silva, Jayanth Somashekaraiah, Maximilian Weigand, David Kube, Felix Gehlhoff - [[pdf]](https://arxiv.org/pdf/2512.03955)</summary>

**Abstract:** Industrial automation increasingly requires flexible control strategies that can adapt to changing tasks and environments. Agents based on Large Language Models (LLMs) offer potential for such adaptive planning and execution but lack standardized benchmarks for systematic comparison. We introduce a benchmark with an executable simulation environment representing the Blocksworld problem providing five complexity categories. By integrating the Model Context Protocol (MCP) as a standardized tool interface, diverse agent architectures can be connected to and evaluated against the benchmark without implementation-specific modifications. A single-agent implementation demonstrates the benchmark's applicability, establishing quantitative metrics for comparison of LLM-based planning and execution approaches.

**arXiv ID:** 2512.03955
</details>

<details>
<summary><strong>CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency</strong> - Jiacheng Guo, Suozhi Huang, Zixin Yao, Yifan Zhang, Yifu Lu, Jiashuo Liu, Zihao Li, Nicholas Deng, Qixin Xiao, Jia Tian, Kanghong Zhan, Tianyi Li, Xiaochen Liu, Jason Ge, Chaoyang He, Kaixuan Huang, Lin Yang, Wenhao Huang, Mengdi Wang - [[pdf]](https://arxiv.org/pdf/2512.00417)</summary>

**Abstract:** This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills.
Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.

**arXiv ID:** 2512.00417
</details>

<details>
<summary><strong>Grounded Test-Time Adaptation for LLM Agents</strong> - Arthur Chen, Zuxin Liu, Jianguo Zhang, Akshara Prabhakar, Zhiwei Liu, Shelby Heinecke, Silvio Savarese, Victor Zhong, Caiming Xiong - [[pdf]](https://arxiv.org/pdf/2511.04847)</summary>

**Abstract:** Large language model (LLM)-based agents struggle to generalize to novel and complex environments, such as unseen websites or new sets of functions, due to a fundamental mismatch between their pre-training and test-time conditions. This challenge stems from two distinct failure modes: a syntactic misunderstanding of environment-specific components like observation formats, and a semantic misunderstanding of state-transition dynamics, which are only revealed at test time. To address these issues, we propose two distinct and complementary strategies for adapting LLM agents by leveraging environment-specific information available during deployment. First, an online distributional adaptation method parameterizes environmental nuances by learning a lightweight adaptation vector that biases the model's output distribution, enabling rapid alignment with an environment response format. Second, a deployment-time dynamics grounding method employs a persona-driven exploration phase to systematically probe and learn the environment's causal dynamics before task execution, equipping the agent with a nonparametric world model. We evaluate these strategies across diverse agentic benchmarks, including function calling and web navigation. Our empirical results show the effectiveness of both strategies across all benchmarks with minimal computational cost. We find that dynamics grounding is particularly effective in complex environments where unpredictable dynamics pose a major obstacle, demonstrating a robust path toward more generalizable and capable LLM-based agents. For example, on the WebArena multi-site split, this method increases the agent's success rate from 2% to 23%.

**arXiv ID:** 2511.04847
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (19 papers)</h2></summary>

<details>
<summary><strong>Evaluating Generalization Capabilities of LLM-Based Agents in Mixed-Motive Scenarios Using Concordia</strong> - Chandler Smith, Marwa Abdulhai, Manfred Diaz, Marko Tesic, Rakshit S. Trivedi, Alexander Sasha Vezhnevets, Lewis Hammond, Jesse Clifton, Minsuk Chang, Edgar A. Duéñez-Guzmán, John P. Agapiou, Jayd Matyas, Danny Karmon, Akash Kundu, Aliaksei Korshuk, Ananya Ananya, Arrasy Rahman, Avinaash Anand Kulandaivel, Bain McHale, Beining Zhang, Buyantuev Alexander, Carlos Saith Rodriguez Rojas, Caroline Wang, Chetan Talele, Chenao Liu, Chichen Lin, Diana Riazi, Di Yang Shi, Emanuel Tewolde, Elizaveta Tennant, Fangwei Zhong, Fuyang Cui, Gang Zhao, Gema Parreño Piqueras, Hyeonggeun Yun, Ilya Makarov, Jiaxun Cui, Jebish Purbey, Jim Dilkes, Jord Nguyen, Lingyun Xiao, Luis Felipe Giraldo, Manuela Chacon-Chamorro, Manuel Sebastian Rios Beltran, Marta Emili García Segura, Mengmeng Wang, Mogtaba Alim, Nicanor Quijano, Nico Schiavone, Olivia Macmillan-Scott, Oswaldo Peña, Peter Stone, Ram Mohan Rao Kadiyala, Rolando Fernandez, Ruben Manrique, Sunjia Lu, Sheila A. McIlraith, Shamika Dhuri, Shuqing Shi, Siddhant Gupta, Sneheel Sarangi, Sriram Ganapathi Subramanian, Taehun Cha, Toryn Q. Klassen, Wenming Tu, Weijian Fan, Wu Ruiyang, Xue Feng, Yali Du, Yang Liu, Yiding Wang, Yipeng Kang, Yoonchang Sung, Yuxuan Chen, Zhaowei Zhang, Zhihan Wang, Zhiqiang Wu, Ziang Chen, Zilong Zheng, Zixia Jia, Ziyan Wang, Dylan Hadfield-Menell, Natasha Jaques, Tim Baarslag, Jose Hernandez-Orallo, Joel Z. Leibo - [[pdf]](https://arxiv.org/pdf/2512.03318)</summary>

**Abstract:** Large Language Model (LLM) agents have demonstrated impressive capabilities for social interaction and are increasingly being deployed in situations where they might engage with both human and artificial agents. These interactions represent a critical frontier for LLM-based agents, yet existing evaluation methods fail to measure how well these capabilities generalize to novel social situations. In this paper, we introduce a method for evaluating the ability of LLM-based agents to cooperate in zero-shot, mixed-motive environments using Concordia, a natural language multi-agent simulation environment. Our method measures general cooperative intelligence by testing an agent's ability to identify and exploit opportunities for mutual gain across diverse partners and contexts. We present empirical results from the NeurIPS 2024 Concordia Contest, where agents were evaluated on their ability to achieve mutual gains across a suite of diverse scenarios ranging from negotiation to collective action problems. Our findings reveal significant gaps between current agent capabilities and the robust generalization required for reliable cooperation, particularly in scenarios demanding persuasion and norm enforcement.

**arXiv ID:** 2512.03318
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning with Communication-Constrained Priors</strong> - Guang Yang, Tianpei Yang, Jingwen Qiao, Yanqing Wu, Jing Huo, Xingguo Chen, Yang Gao - [[pdf]](https://arxiv.org/pdf/2512.03528)</summary>

**Abstract:** Communication is one of the effective means to improve the learning of cooperative policy in multi-agent systems. However, in most real-world scenarios, lossy communication is a prevalent issue. Existing multi-agent reinforcement learning with communication, due to their limited scalability and robustness, struggles to apply to complex and dynamic real-world environments. To address these challenges, we propose a generalized communication-constrained model to uniformly characterize communication conditions across different scenarios. Based on this, we utilize it as a learning prior to distinguish between lossy and lossless messages for specific scenarios. Additionally, we decouple the impact of lossy and lossless messages on distributed decision-making, drawing on a dual mutual information estimatior, and introduce a communication-constrained multi-agent reinforcement learning framework, quantifying the impact of communication messages into the global reward. Finally, we validate the effectiveness of our approach across several communication-constrained benchmarks.

**arXiv ID:** 2512.03528
</details>

<details>
<summary><strong>PARC: An Autonomous Self-Reflective Coding Agent for Robust Execution of Long-Horizon Tasks</strong> - Yuki Orimo, Iori Kurata, Hodaka Mori, Ryuhei Okuno, Ryohto Sawada, Daisuke Okanohara - [[pdf]](https://arxiv.org/pdf/2512.03549)</summary>

**Abstract:** We introduce PARC, a coding agent for the autonomous and robust execution of long-horizon computational tasks. PARC is built on a hierarchical multi-agent architecture incorporating task planning, execution, and a mechanism that evaluates its own actions and their outcomes from an independent context and provides feedback, namely self-assessment and self-feedback. This design enables PARC to detect and correct high-level strategic errors and sustain progress without human intervention. We evaluate PARC across computational science and data science tasks. In materials science, it autonomously reproduces key results from studies on lithium-ion conduction and alloy segregation. In particular, it coordinates dozens of parallel simulation tasks, each requiring roughly 43 hours of computation, managing orchestration, monitoring, and error correction end-to-end. In Kaggle-based experiments, starting from minimal natural-language instructions, PARC conducts data analysis and implements search strategies, producing solutions competitive with human-engineered baselines. These results highlight the potential of integrating a hierarchical multi-agent system with self-assessment and self-feedback to enable AI systems capable of independent, large-scale scientific and analytical work.

**arXiv ID:** 2512.03549
</details>

<details>
<summary><strong>Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases</strong> - Michael Theologitis, Dan Suciu - [[pdf]](https://arxiv.org/pdf/2512.03278)</summary>

**Abstract:** In today's age, it is becoming increasingly difficult to decipher truth from lies. Every day, politicians, media outlets, and public figures make conflicting claims$\unicode{x2014}$often about topics that can, in principle, be verified against structured data. For instance, statements about crime rates, economic growth or healthcare can all be verified against official public records and structured datasets. Building a system that can automatically do that would have sounded like science fiction just a few years ago. Yet, with the extraordinary progress in LLMs and agentic AI, this is now within reach. Still, there remains a striking gap between what is technically possible and what is being demonstrated by recent work. Most existing verification systems operate only on small, single-table databases$\unicode{x2014}$typically a few hundred rows$\unicode{x2014}$that conveniently fit within an LLM's context window.
In this paper we report our progress on Thucy, the first cross-database, cross-table multi-agent claim verification system that also provides concrete evidence for each verification verdict. Thucy remains completely agnostic to the underlying data sources before deployment and must therefore autonomously discover, inspect, and reason over all available relational databases to verify claims. Importantly, Thucy also reports the exact SQL queries that support its verdict (whether the claim is accurate or not) offering full transparency to expert users familiar with SQL. When evaluated on the TabFact dataset$\unicode{x2014}$the standard benchmark for fact verification over structured data$\unicode{x2014}$Thucy surpasses the previous state of the art by 5.6 percentage points in accuracy (94.3% vs. 88.7%).

**arXiv ID:** 2512.03278
</details>

<details>
<summary><strong>Multi-Aspect Knowledge-Enhanced Medical Vision-Language Pretraining with Multi-Agent Data Generation</strong> - Xieji Li, Siyuan Yan, Yingsheng Liu, H. Peter Soyer, Monika Janda, Victoria Mar, Zongyuan Ge - [[pdf]](https://arxiv.org/pdf/2512.03445)</summary>

**Abstract:** Vision-language pretraining (VLP) has emerged as a powerful paradigm in medical image analysis, enabling representation learning from large-scale image-text pairs without relying on expensive manual annotations. However, existing methods often struggle with the noise inherent in web-collected data and the complexity of unstructured long medical texts. To address these challenges, we propose a novel VLP framework integrating a Multi-Agent data GENeration (MAGEN) system and Ontology-based Multi-Aspect Knowledge-Enhanced (O-MAKE) pretraining. First, MAGEN enhances data quality by synthesizing knowledge-enriched descriptions via a foundation model-assisted captioning and retrieval-based verification pipeline. Second, O-MAKE addresses the difficulty of learning from long, unstructured texts by decomposing them into distinct knowledge aspects. This facilitates fine-grained alignment at both global and patch levels, while explicitly modeling medical concept relationships through ontology-guided mechanisms. We validate our framework in the field of dermatology, where comprehensive experiments demonstrate the effectiveness of each component. Our approach achieves state-of-the-art zero-shot performance on disease classification and cross-modal retrieval tasks across eight datasets. Our code and the augmented dataset Derm1M-AgentAug, comprising over 400k skin-image-text pairs, will be released at this https URL.

**arXiv ID:** 2512.03445
</details>

<details>
<summary><strong>Think Before You Drive: World Model-Inspired Multimodal Grounding for Autonomous Vehicles</strong> - Haicheng Liao, Huanming Shen, Bonan Wang, Yongkang Li, Yihong Tang, Chengyue Wang, Dingyi Zhuang, Kehua Chen, Hai Yang, Chengzhong Xu, Zhenning Li - [[pdf]](https://arxiv.org/pdf/2512.03454)</summary>

**Abstract:** Interpreting natural-language commands to localize target objects is critical for autonomous driving (AD). Existing visual grounding (VG) methods for autonomous vehicles (AVs) typically struggle with ambiguous, context-dependent instructions, as they lack reasoning over 3D spatial relations and anticipated scene evolution. Grounded in the principles of world models, we propose ThinkDeeper, a framework that reasons about future spatial states before making grounding decisions. At its core is a Spatial-Aware World Model (SA-WM) that learns to reason ahead by distilling the current scene into a command-aware latent state and rolling out a sequence of future latent states, providing forward-looking cues for disambiguation. Complementing this, a hypergraph-guided decoder then hierarchically fuses these states with the multimodal input, capturing higher-order spatial dependencies for robust localization. In addition, we present DrivePilot, a multi-source VG dataset in AD, featuring semantic annotations generated by a Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT)-prompted LLM pipeline. Extensive evaluations on six benchmarks, ThinkDeeper ranks #1 on the Talk2Car leaderboard and surpasses state-of-the-art baselines on DrivePilot, MoCAD, and RefCOCO/+/g benchmarks. Notably, it shows strong robustness and efficiency in challenging scenes (long-text, multi-agent, ambiguity) and retains superior performance even when trained on 50% of the data.

**arXiv ID:** 2512.03454
</details>

<details>
<summary><strong>AsymPuzl: An Asymmetric Puzzle for multi-agent cooperation</strong> - Xavier Cadet, Edward Koh, Peter Chin - [[pdf]](https://arxiv.org/pdf/2512.03466)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly studied in multi-turn, multi-agent scenarios, yet most existing setups emphasize open-ended role-play rather than controlled evaluation. We introduce AsymPuzl, a minimal but expressive two-agent puzzle environment designed to isolate communication under information asymmetry. Each agent observes complementary but incomplete views of a symbolic puzzle and must exchange messages to solve it cooperatively. Using a diverse set of current-generation and open-source LLMs, we show that (i) strong models such as GPT-5 and Claude-4.0 reliably converge across puzzle sizes on the solution by sharing complete information in two turns, (ii) weaker models often ignore partner messages or over-correct their hypotheses, and (iii) feedback design is non-trivial: simple self-feedback improves success rates, while detailed joint feedback can hurt performance. These findings show that even in simple cooperative tasks, LLM communication strategies diverge and depend on the granularity of feedback signals. AsymPuzl thus provides a testbed for probing the limits of multi-turn cooperation and opens avenues for studying coordination mechanisms.

**arXiv ID:** 2512.03466
</details>

<details>
<summary><strong>TeamMedAgents: Enhancing Medical Decision-Making of LLMs Through Structured Teamwork</strong> - Pranav Pushkar Mishra, Mohammad Arvan, Mohan Zalake - [[pdf]](https://arxiv.org/pdf/2508.08115)</summary>

**Abstract:** We present TeamMedAgents, a modular multi-agent framework that systematically translates evidence-based teamwork principles from organizational psychology into large language model collaboration for medical decision-making. Building upon Salas et al.'s "Big Five" teamwork model, we operationalize five core components as independently configurable mechanisms: shared mental models, team leadership, team orientation, trust networks, and mutual monitoring. Our architecture dynamically recruits 2-4 specialist agents and employs structured four-phase deliberation with adaptive component selection. Evaluation across eight medical benchmarks encompassing 11,545 questions demonstrates TeamMedAgents achieves 77.63% overall accuracy (text-based: 81.30%, vision-language: 66.60%). Systematic ablation studies comparing three single-agent baselines (Zero-Shot, Few-Shot, CoT) against individual teamwork components reveal task-specific optimization patterns: shared mental models excel on knowledge tasks, trust mechanisms improve differential diagnosis, while comprehensive integration degrades performance. Adaptive component selection yields 2-10 percentage point improvements over strongest baselines, with 96.2% agent convergence validating structured coordination effectiveness.
TeamMedAgents establishes principled methodology for translating human teamwork theory into multi-agent systems, demonstrating that evidence-based collaboration patterns enhance AI performance in safety-critical domains through modular component design and selective activation strategies.

**arXiv ID:** 2508.08115
</details>

<details>
<summary><strong>Mobile-Agent-RAG: Driving Smart Multi-Agent Coordination with Contextual Knowledge Empowerment for Long-Horizon Mobile Automation</strong> - Yuxiang Zhou, Jichang Li, Yanhao Zhang, Haonan Lu, Guanbin Li - [[pdf]](https://arxiv.org/pdf/2511.12254)</summary>

**Abstract:** Mobile agents show immense potential, yet current state-of-the-art (SoTA) agents exhibit inadequate success rates on real-world, long-horizon, cross-application tasks. We attribute this bottleneck to the agents' excessive reliance on static, internal knowledge within MLLMs, which leads to two critical failure points: 1) strategic hallucinations in high-level planning and 2) operational errors during low-level execution on user interfaces (UI). The core insight of this paper is that high-level planning and low-level UI operations require fundamentally distinct types of knowledge. Planning demands high-level, strategy-oriented experiences, whereas operations necessitate low-level, precise instructions closely tied to specific app UIs. Motivated by these insights, we propose Mobile-Agent-RAG, a novel hierarchical multi-agent framework that innovatively integrates dual-level retrieval augmentation. At the planning stage, we introduce Manager-RAG to reduce strategic hallucinations by retrieving human-validated comprehensive task plans that provide high-level guidance. At the execution stage, we develop Operator-RAG to improve execution accuracy by retrieving the most precise low-level guidance for accurate atomic actions, aligned with the current app and subtask. To accurately deliver these knowledge types, we construct two specialized retrieval-oriented knowledge bases. Furthermore, we introduce Mobile-Eval-RAG, a challenging benchmark for evaluating such agents on realistic multi-app, long-horizon tasks. Extensive experiments demonstrate that Mobile-Agent-RAG significantly outperforms SoTA baselines, improving task completion rate by 11.0% and step efficiency by 10.2%, establishing a robust paradigm for context-aware, reliable multi-agent mobile automation.

**arXiv ID:** 2511.12254
</details>

<details>
<summary><strong>Astra: A Multi-Agent System for GPU Kernel Performance Optimization</strong> - Anjiang Wei, Tianran Sun, Yogesh Seenichamy, Hang Song, Anne Ouyang, Azalia Mirhoseini, Ke Wang, Alex Aiken - [[pdf]](https://arxiv.org/pdf/2509.07506)</summary>

**Abstract:** GPU kernel optimization has long been a central challenge at the intersection of high-performance computing and machine learning. Efficient kernels are crucial for accelerating large language model (LLM) training and serving, yet attaining high performance typically requires extensive manual tuning. Compiler-based systems reduce some of this burden, but still demand substantial manual design and engineering effort. Recently, researchers have explored using LLMs for GPU kernel generation, though prior work has largely focused on translating high-level PyTorch modules into CUDA code. In this work, we introduce Astra, the first LLM-based multi-agent system for GPU kernel optimization. Unlike previous approaches, Astra starts from existing CUDA implementations extracted from SGLang, a widely deployed framework for serving LLMs, rather than treating PyTorch modules as the specification. Within Astra, specialized LLM agents collaborate through iterative code generation, testing, profiling, and planning to produce kernels that are both correct and high-performance. On kernels from SGLang, Astra achieves an average speedup of 1.32x using zero-shot prompting with OpenAI o4-mini. A detailed case study further demonstrates that LLMs can autonomously apply loop transformations, optimize memory access patterns, exploit CUDA intrinsics, and leverage fast math operations to yield substantial performance gains. Our work highlights multi-agent LLM systems as a promising new paradigm for GPU kernel optimization. Our code is publicly available at this https URL.

**arXiv ID:** 2509.07506
</details>

<details>
<summary><strong>A Gossip-Enhanced Communication Substrate for Agentic AI: Toward Decentralized Coordination in Large-Scale Multi-Agent Systems</strong> - Nafiul I. Khan, Mansura Habiba, Rafflesia Khan - [[pdf]](https://arxiv.org/pdf/2512.03285)</summary>

**Abstract:** As agentic platforms scale, agents are moving beyond fixed roles and predefined toolchains, creating an urgent need for flexible and decentralized coordination. Current structured communication protocols such as direct agent-to-agent messaging or MCP-style tool calls offer reliability, but they struggle to support the emergent and swarm-like intelligence required in large adaptive systems. Distributed agents must learn continuously, share context fluidly, and coordinate without depending solely on central planners.
This paper revisits gossip protocols as a complementary substrate for agentic communication. Gossip mechanisms, long valued in distributed systems for their decentralized and fault-tolerant properties, provide scalable and adaptive diffusion of knowledge and fill gaps that structured protocols alone cannot efficiently address. However, gossip also introduces challenges, including semantic relevance, temporal staleness, and limited guarantees on action consistency in rapidly changing environments.
We examine how gossip can support context-rich state propagation, resilient coordination under uncertainty, and emergent global awareness. We also outline open problems around semantic filtering, trust, and knowledge decay. Rather than proposing a complete framework, this paper presents a research agenda for integrating gossip into multi-agent communication stacks and argues that gossip is essential for future agentic ecosystems that must remain robust, adaptive, and self-organizing as their scale and autonomy increase.

**arXiv ID:** 2512.03285
</details>

<details>
<summary><strong>SRPG: Semantically Reconstructed Privacy Guard for Zero-Trust Privacy in Educational Multi-Agent Systems</strong> - Shuang Guo, Zihui Li - [[pdf]](https://arxiv.org/pdf/2512.03694)</summary>

**Abstract:** Multi-Agent Systems (MAS) with large language models (LLMs) enable personalized education but risk leaking minors personally identifiable information (PII) via unstructured dialogue. Existing privacy methods struggle to balance security and utility: role-based access control fails on unstructured text, while naive masking destroys pedagogical context. We propose SRPG, a privacy guard for educational MAS, using a Dual-Stream Reconstruction Mechanism: a strict sanitization stream ensures zero PII leakage, and a context reconstruction stream (LLM driven) recovers mathematical logic. This decouples instructional content from private data, preserving teaching efficacy. Tests on MathDial show SRPG works across models; with GPT-4o, it achieves 0.0000 Attack Success Rate (ASR) (zero leakage) and 0.8267 Exact Match, far outperforming the zero trust Pure LLM baseline (0.2138). SRPG effectively protects minors privacy without sacrificing mathematical instructional quality.

**arXiv ID:** 2512.03694
</details>

<details>
<summary><strong>Many-to-One Adversarial Consensus: Exposing Multi-Agent Collusion Risks in AI-Based Healthcare</strong> - Adeela Bashir, Anh han, Zia Ush Shamszaman - [[pdf]](https://arxiv.org/pdf/2512.03097)</summary>

**Abstract:** The integration of large language models (LLMs) into healthcare IoT systems promises faster decisions and improved medical support. LLMs are also deployed as multi-agent teams to assist AI doctors by debating, voting, or advising on decisions. However, when multiple assistant agents interact, coordinated adversaries can collude to create false consensus, pushing an AI doctor toward harmful prescriptions. We develop an experimental framework with scripted and unscripted doctor agents, adversarial assistants, and a verifier agent that checks decisions against clinical guidelines. Using 50 representative clinical questions, we find that collusion drives the Attack Success Rate (ASR) and Harmful Recommendation Rates (HRR) up to 100% in unprotected systems. In contrast, the verifier agent restores 100% accuracy by blocking adversarial consensus. This work provides the first systematic evidence of collusion risk in AI healthcare and demonstrates a practical, lightweight defence that ensures guideline fidelity.

**arXiv ID:** 2512.03097
</details>

<details>
<summary><strong>GRAND: Guidance, Rebalancing, and Assignment for Networked Dispatch in Multi-Agent Path Finding</strong> - Johannes Gaber, Meshal Alharbi, Daniele Gammelli, Gioele Zardini - [[pdf]](https://arxiv.org/pdf/2512.03194)</summary>

**Abstract:** Large robot fleets are now common in warehouses and other logistics settings, where small control gains translate into large operational impacts. In this article, we address task scheduling for lifelong Multi-Agent Pickup-and-Delivery (MAPD) and propose a hybrid method that couples learning-based global guidance with lightweight optimization. A graph neural network policy trained via reinforcement learning outputs a desired distribution of free agents over an aggregated warehouse graph. This signal is converted into region-to-region rebalancing through a minimum-cost flow, and finalized by small, local assignment problems, preserving accuracy while keeping per-step latency within a 1 s compute budget. On congested warehouse benchmarks from the League of Robot Runners (LRR) with up to 500 agents, our approach improves throughput by up to 10% over the 2024 winning scheduler while maintaining real-time execution. The results indicate that coupling graph-structured learned guidance with tractable solvers reduces congestion and yields a practical, scalable blueprint for high-throughput scheduling in large fleets.

**arXiv ID:** 2512.03194
</details>

<details>
<summary><strong>Welfare and Cost Aggregation for Multi-Agent Control: When to Choose Which Social Cost Function, and Why?</strong> - Ilia Shilov, Ezzat Elokda, Sophie Hall, Heinrich H. Nax, Saverio Bolognani - [[pdf]](https://arxiv.org/pdf/2503.20772)</summary>

**Abstract:** Many multi-agent socio-technical systems rely on aggregating heterogeneous agents' costs into a social cost function (SCF) to coordinate resource allocation in domains like energy grids, water allocation, or traffic management. The choice of SCF often entails implicit assumptions and may lead to undesirable outcomes if not rigorously justified. In this paper, we demonstrate that what determines which SCF ought to be used is the degree to which individual costs can be compared across agents and which axioms the aggregation shall fulfill. Drawing on the results from social choice theory, we provide guidance on how this process can be used in control applications. We demonstrate which assumptions about interpersonal utility comparability - ranging from ordinal level comparability to full cardinal comparability - together with a choice of desirable axioms, inform the selection of a correct SCF, be it the classical utilitarian sum, the Nash SCF, or maximin. We then demonstrate how the proposed framework can be applied for principled allocations of water and transportation resources.

**arXiv ID:** 2503.20772
</details>

<details>
<summary><strong>MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management</strong> - Jiayi Chen, Jing Li, Guiling Wang - [[pdf]](https://arxiv.org/pdf/2508.01173)</summary>

**Abstract:** Reinforcement Learning (RL) has shown significant promise in automated portfolio management; however, effectively balancing risk and return remains a central challenge, as many models fail to adapt to dynamically changing market conditions. We propose Meta-controlled Agents for a Risk-aware System (MARS), a novel framework addressing this through a multi-agent, risk-aware approach. MARS replaces monolithic models with a Heterogeneous Agent Ensemble, where each agent's unique risk profile is enforced by a Safety-Critic network to span behaviors from capital preservation to aggressive growth. A high-level Meta-Adaptive Controller (MAC) dynamically orchestrates this ensemble, shifting reliance between conservative and aggressive agents to minimize drawdown during downturns while seizing opportunities in bull markets. This two-tiered structure leverages behavioral diversity rather than explicit feature engineering to ensure a disciplined portfolio robust across market regimes. Experiments on major international indexes confirm that our framework significantly reduces maximum drawdown and volatility while maintaining competitive returns.

**arXiv ID:** 2508.01173
</details>

<details>
<summary><strong>A Multi-Agent, Policy-Gradient approach to Network Routing</strong> - Nigel Tao, Jonathan Baxter, Lex Weaver - [[pdf]](https://arxiv.org/pdf/2512.03211)</summary>

**Abstract:** Network routing is a distributed decision problem which naturally admits numerical performance measures, such as the average time for a packet to travel from source to destination. OLPOMDP, a policy-gradient reinforcement learning algorithm, was successfully applied to simulated network routing under a number of network models. Multiple distributed agents (routers) learned co-operative behavior without explicit inter-agent communication, and they avoided behavior which was individually desirable, but detrimental to the group's overall performance. Furthermore, shaping the reward signal by explicitly penalizing certain patterns of sub-optimal behavior was found to dramatically improve the convergence rate.

**arXiv ID:** 2512.03211
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning and Real-Time Decision-Making in Robotic Soccer for Virtual Environments</strong> - Aya Taourirte, Md Sohag Mia - [[pdf]](https://arxiv.org/pdf/2512.03166)</summary>

**Abstract:** The deployment of multi-agent systems in dynamic, adversarial environments like robotic soccer necessitates real-time decision-making, sophisticated cooperation, and scalable algorithms to avoid the curse of dimensionality. While Reinforcement Learning (RL) offers a promising framework, existing methods often struggle with the multi-granularity of tasks (long-term strategy vs. instant actions) and the complexity of large-scale agent interactions. This paper presents a unified Multi-Agent Reinforcement Learning (MARL) framework that addresses these challenges. First, we establish a baseline using Proximal Policy Optimization (PPO) within a client-server architecture for real-time action scheduling, with PPO demonstrating superior performance (4.32 avg. goals, 82.9% ball control). Second, we introduce a Hierarchical RL (HRL) structure based on the options framework to decompose the problem into a high-level trajectory planning layer (modeled as a Semi-Markov Decision Process) and a low-level action execution layer, improving global strategy (avg. goals increased to 5.26). Finally, to ensure scalability, we integrate mean-field theory into the HRL framework, simplifying many-agent interactions into a single agent vs. the population average. Our mean-field actor-critic method achieves a significant performance boost (5.93 avg. goals, 89.1% ball control, 92.3% passing accuracy) and enhanced training stability. Extensive simulations of 4v4 matches in the Webots environment validate our approach, demonstrating its potential for robust, scalable, and cooperative behavior in complex multi-agent domains.

**arXiv ID:** 2512.03166
</details>

<details>
<summary><strong>Context-Triggered Contingency Games for Strategic Multi-Agent Interaction</strong> - Kilian Schweppe, Anne-Kathrin Schmuck - [[pdf]](https://arxiv.org/pdf/2512.03639)</summary>

**Abstract:** We address the challenge of reliable and efficient interaction in autonomous multi-agent systems, where agents must balance long-term strategic objectives with short-term dynamic adaptation. We propose context-triggered contingency games, a novel integration of strategic games derived from temporal logic specifications with dynamic contingency games solved in real time. Our two-layered architecture leverages strategy templates to guarantee satisfaction of high-level objectives, while a new factor-graph-based solver enables scalable, real-time model predictive control of dynamic interactions. The resulting framework ensures both safety and progress in uncertain, interactive environments. We validate our approach through simulations and hardware experiments in autonomous driving and robotic navigation, demonstrating efficient, reliable, and adaptive multi-agent interaction.

**arXiv ID:** 2512.03639
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths</strong> - Zhening Li, Armando Solar-Lezama, Yisong Yue, Stephan Zheng - [[pdf]](https://arxiv.org/pdf/2512.03571)</summary>

**Abstract:** We introduce a new approach to agent programming, the development of LLM-based agents. Current approaches to agent programming often entangle two aspects of agent design: the core workflow logic and the inference-time strategy (e.g., tree search). We introduce "probabilistic angelic nondeterminism" ("PAN"), a programming model that disentangles these two concerns, allowing the programmer to describe the agent workflow and independently experiment with different inference-time strategies by simply changing a few inputs. We provide an implementation of PAN in Python as the EnCompass framework, which uses a Python decorator to compile agent workflow programs into a search space. We present three case studies that demonstrate how the framework lets the programmer quickly improve the reliability of an agent and easily switch between different inference-time strategies, all with little additional coding.

**arXiv ID:** 2512.03571
</details>

<details>
<summary><strong>ATHENA: Agentic Team for Hierarchical Evolutionary Numerical Algorithms</strong> - Juan Diego Toscano, Daniel T. Chen, George Em Karniadakis - [[pdf]](https://arxiv.org/pdf/2512.03476)</summary>

**Abstract:** Bridging the gap between theoretical conceptualization and computational implementation is a major bottleneck in Scientific Computing (SciC) and Scientific Machine Learning (SciML). We introduce ATHENA (Agentic Team for Hierarchical Evolutionary Numerical Algorithms), an agentic framework designed as an Autonomous Lab to manage the end-to-end computational research lifecycle. Its core is the HENA loop, a knowledge-driven diagnostic process framed as a Contextual Bandit problem. Acting as an online learner, the system analyzes prior trials to select structural `actions' ($A_n$) from combinatorial spaces guided by expert blueprints (e.g., Universal Approximation, Physics-Informed constraints). These actions are translated into executable code ($S_n$) to generate scientific rewards ($R_n$). ATHENA transcends standard automation: in SciC, it autonomously identifies mathematical symmetries for exact analytical solutions or derives stable numerical solvers where foundation models fail. In SciML, it performs deep diagnosis to tackle ill-posed formulations and combines hybrid symbolic-numeric workflows (e.g., coupling PINNs with FEM) to resolve multiphysics problems. The framework achieves super-human performance, reaching validation errors of $10^{-14}$. Furthermore, collaborative ``human-in-the-loop" intervention allows the system to bridge stability gaps, improving results by an order of magnitude. This paradigm shift focuses from implementation mechanics to methodological innovation, accelerating scientific discovery.

**arXiv ID:** 2512.03476
</details>

<details>
<summary><strong>Large Language Model-Based Agents for Software Engineering: A Survey</strong> - Junwei Liu, Kaixin Wang, Yixuan Chen, Xin Peng, Zhenpeng Chen, Lingming Zhang, Yiling Lou - [[pdf]](https://arxiv.org/pdf/2409.02977)</summary>

**Abstract:** The recent advance in Large Language Models (LLMs) has shaped a new paradigm of AI agents, i.e., LLM-based agents. Compared to standalone LLMs, LLM-based agents substantially extend the versatility and expertise of LLMs by enhancing LLMs with the capabilities of perceiving and utilizing external resources and tools. To date, LLM-based agents have been applied and shown remarkable effectiveness in Software Engineering (SE). The synergy between multiple agents and human interaction brings further promise in tackling complex real-world SE problems. In this work, we present a comprehensive and systematic survey on LLM-based agents for SE. We collect 124 papers and categorize them from two perspectives, i.e., the SE and agent perspectives. In addition, we discuss open challenges and future directions in this critical domain. The repository of this survey is at this https URL.

**arXiv ID:** 2409.02977
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (27 papers)</h2></summary>

<details>
<summary><strong>Prior preferences in active inference agents: soft, hard, and goal shaping</strong> - Filippo Torresan, Ryota Kanai, Manuel Baltieri - [[pdf]](https://arxiv.org/pdf/2512.03293)</summary>

**Abstract:** Active inference proposes expected free energy as an objective for planning and decision-making to adequately balance exploitative and explorative drives in learning agents. The exploitative drive, or what an agent wants to achieve, is formalised as the Kullback-Leibler divergence between a variational probability distribution, updated at each inference step, and a preference probability distribution that indicates what states or observations are more likely for the agent, hence determining the agent's goal in a certain environment. In the literature, the questions of how the preference distribution should be specified and of how a certain specification impacts inference and learning in an active inference agent have been given hardly any attention. In this work, we consider four possible ways of defining the preference distribution, either providing the agents with hard or soft goals and either involving or not goal shaping (i.e., intermediate goals). We compare the performances of four agents, each given one of the possible preference distributions, in a grid world navigation task. Our results show that goal shaping enables the best performance overall (i.e., it promotes exploitation) while sacrificing learning about the environment's transition dynamics (i.e., it hampers exploration).

**arXiv ID:** 2512.03293
</details>

<details>
<summary><strong>Multimodal Reinforcement Learning with Agentic Verifier for AI Agents</strong> - Reuben Tan, Baolin Peng, Zhengyuan Yang, Hao Cheng, Oier Mees, Theodore Zhao, Andrea Tupini, Isar Meijier, Qianhui Wu, Yuncong Yang, Lars Liden, Yu Gu, Sheng Zhang, Xiaodong Liu, Lijuan Wang, Marc Pollefeys, Yong Jae Lee, Jianfeng Gao - [[pdf]](https://arxiv.org/pdf/2512.03438)</summary>

**Abstract:** Agentic reasoning models trained with multimodal reinforcement learning (MMRL) have become increasingly capable, yet they are almost universally optimized using sparse, outcome-based rewards computed based on the final answers. Richer rewards computed from the reasoning tokens can improve learning significantly by providing more fine-grained guidance. However, it is challenging to compute more informative rewards in MMRL beyond those based on outcomes since different samples may require different scoring functions and teacher models may provide noisy reward signals too. In this paper, we introduce the Argos (Agentic Reward for Grounded & Objective Scoring), a principled reward agent to train multimodal reasoning models for agentic tasks. For each sample, Argos selects from a pool of teacher-model derived and rule-based scoring functions to simultaneously evaluate: (i) final response accuracy, (ii) spatiotemporal localization of referred entities and actions, and (iii) the quality of the reasoning process. We find that by leveraging our agentic verifier across both SFT data curation and RL training, our model achieves state-of-the-art results across multiple agentic tasks such as spatial reasoning, visual hallucination as well as robotics and embodied AI benchmarks. Critically, we demonstrate that just relying on SFT post-training on highly curated reasoning data is insufficient, as agents invariably collapse to ungrounded solutions during RL without our online verification. We also show that our agentic verifier can help to reduce reward-hacking in MMRL. Finally, we also provide a theoretical justification for the effectiveness of Argos through the concept of pareto-optimality.

**arXiv ID:** 2512.03438
</details>

<details>
<summary><strong>MemVerse: Multimodal Memory for Lifelong Learning Agents</strong> - Junming Liu, Yifei Sun, Weihua Cheng, Haodong Lei, Yirong Chen, Licheng Wen, Xuemeng Yang, Daocheng Fu, Pinlong Cai, Nianchen Deng, Yi Yu, Shuyue Hu, Botian Shi, Ding Wang - [[pdf]](https://arxiv.org/pdf/2512.03627)</summary>

**Abstract:** Despite rapid progress in large-scale language and vision models, AI agents still suffer from a fundamental limitation: they cannot remember. Without reliable memory, agents catastrophically forget past experiences, struggle with long-horizon reasoning, and fail to operate coherently in multimodal or interactive environments. We introduce MemVerse, a model-agnostic, plug-and-play memory framework that bridges fast parametric recall with hierarchical retrieval-based memory, enabling scalable and adaptive multimodal intelligence. MemVerse maintains short-term memory for recent context while transforming raw multimodal experiences into structured long-term memories organized as hierarchical knowledge graphs. This design supports continual consolidation, adaptive forgetting, and bounded memory growth. To handle real-time demands, MemVerse introduces a periodic distillation mechanism that compresses essential knowledge from long-term memory into the parametric model, allowing fast, differentiable recall while preserving interpretability. Extensive experiments demonstrate that MemVerse significantly improves multimodal reasoning and continual learning efficiency, empowering agents to remember, adapt, and reason coherently across extended interactions.

**arXiv ID:** 2512.03627
</details>

<details>
<summary><strong>Omni-AutoThink: Adaptive Multimodal Reasoning via Reinforcement Learning</strong> - Dongchao Yang, Songxiang Liu, Disong Wang, Yuanyuan Wang, Guanglu Wan, Helen Meng - [[pdf]](https://arxiv.org/pdf/2512.03783)</summary>

**Abstract:** Recent advances in Omni models have enabled unified multimodal perception and generation. However, most existing systems still exhibit rigid reasoning behaviors, either overthinking simple problems or failing to reason when necessary. To address this limitation, we propose Omni-AutoThink, a novel adaptive reasoning framework that dynamically adjusts the model's reasoning depth according to task difficulty. Our framework comprises two stages: (1) an Adaptive Supervised Fine-Tuning (Adaptive SFT) stage, which endows the Omni model with fundamental reasoning capability using large-scale reasoning-augmented data, and (2) an Adaptive Reinforcement Learning (Adaptive GRPO) stage, which optimizes reasoning behaviors based on task complexity and reward feedback. We further construct a comprehensive adaptive reasoning benchmark that spans text-only, text-audio, text-visual, and text-audio-visual modalities, providing both training and evaluation splits for multimodal reasoning assessment. Experimental results demonstrate that our proposed framework significantly improves adaptive reasoning performance compared to previous baselines. All benchmark data and code will be publicly released.

**arXiv ID:** 2512.03783
</details>

<details>
<summary><strong>Optimizing Life Sciences Agents in Real-Time using Reinforcement Learning</strong> - Nihir Chadderwala - [[pdf]](https://arxiv.org/pdf/2512.03065)</summary>

**Abstract:** Generative AI agents in life sciences face a critical challenge: determining the optimal approach for diverse queries ranging from simple factoid questions to complex mechanistic reasoning. Traditional methods rely on fixed rules or expensive labeled training data, neither of which adapts to changing conditions or user preferences. We present a novel framework that combines AWS Strands Agents with Thompson Sampling contextual bandits to enable AI agents to learn optimal decision-making strategies from user feedback alone. Our system optimizes three key dimensions: generation strategy selection (direct vs. chain-of-thought), tool selection (literature search, drug databases, etc.), and domain routing (pharmacology, molecular biology, clinical specialists). Through empirical evaluation on life science queries, we demonstrate 15-30\% improvement in user satisfaction compared to random baselines, with clear learning patterns emerging after 20-30 queries. Our approach requires no ground truth labels, adapts continuously to user preferences, and provides a principled solution to the exploration-exploitation dilemma in agentic AI systems.

**arXiv ID:** 2512.03065
</details>

<details>
<summary><strong>SPARK: Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning</strong> - Salman Rahman, Sruthi Gorantla, Arpit Gupta, Swastik Roy, Nanyun Peng, Yang Liu - [[pdf]](https://arxiv.org/pdf/2512.03244)</summary>

**Abstract:** Process reward models (PRMs) that provide dense, step-level feedback have shown promise for reinforcement learning, yet their adoption remains limited by the need for expensive step-level annotations or ground truth references. We propose SPARK: a three-stage framework where in the first stage a generator model produces diverse solutions and a verifier model evaluates them using parallel scaling (self-consistency) and sequential scaling (meta-critique). In the second stage, we use these verification outputs as synthetic training data to fine-tune generative process reward models, which subsequently serve as reward signals during training. We show that aggregating multiple independent verifications at the step level produces training data for process reward models that surpass ground-truth outcome supervision, achieving 67.5 F1 on ProcessBench (a benchmark for identifying erroneous steps in mathematical reasoning) compared to 66.4 for reference-guided training and 61.9 for GPT-4o. In the final stage, we apply our generative PRM with chain-of-thought verification (PRM-CoT) as the reward model in RL experiments on mathematical reasoning, and introduce format constraints to prevent reward hacking. Using Qwen2.5-Math-7B, we achieve 47.4% average accuracy across six mathematical reasoning benchmarks, outperforming ground-truth-based RLVR (43.9%). Our work enables reference-free RL training that exceeds ground-truth methods, opening new possibilities for domains lacking verifiable answers or accessible ground truth.

**arXiv ID:** 2512.03244
</details>

<details>
<summary><strong>World Models for Autonomous Navigation of Terrestrial Robots from LIDAR Observations</strong> - Raul Steinmetz, Fabio Demo Rosa, Victor Augusto Kich, Jair Augusto Bottega, Ricardo Bedin Grando, Daniel Fernando Tello Gamarra - [[pdf]](https://arxiv.org/pdf/2512.03429)</summary>

**Abstract:** Autonomous navigation of terrestrial robots using Reinforcement Learning (RL) from LIDAR observations remains challenging due to the high dimensionality of sensor data and the sample inefficiency of model-free approaches. Conventional policy networks struggle to process full-resolution LIDAR inputs, forcing prior works to rely on simplified observations that reduce spatial awareness and navigation robustness. This paper presents a novel model-based RL framework built on top of the DreamerV3 algorithm, integrating a Multi-Layer Perceptron Variational Autoencoder (MLP-VAE) within a world model to encode high-dimensional LIDAR readings into compact latent representations. These latent features, combined with a learned dynamics predictor, enable efficient imagination-based policy optimization. Experiments on simulated TurtleBot3 navigation tasks demonstrate that the proposed architecture achieves faster convergence and higher success rate compared to model-free baselines such as SAC, DDPG, and TD3. It is worth emphasizing that the DreamerV3-based agent attains a 100% success rate across all evaluated environments when using the full dataset of the Turtlebot3 LIDAR (360 readings), while model-free methods plateaued below 85%. These findings demonstrate that integrating predictive world models with learned latent representations enables more efficient and robust navigation from high-dimensional sensory data.

**arXiv ID:** 2512.03429
</details>

<details>
<summary><strong>MPCFormer: A physics-informed data-driven approach for explainable socially-aware autonomous driving</strong> - Jia Hu, Zhexi Lian, Xuerun Yan, Ruiang Bi, Dou Shen, Yu Ruan, Haoran Wang - [[pdf]](https://arxiv.org/pdf/2512.03795)</summary>

**Abstract:** Autonomous Driving (AD) vehicles still struggle to exhibit human-like behavior in highly dynamic and interactive traffic scenarios. The key challenge lies in AD's limited ability to interact with surrounding vehicles, largely due to a lack of understanding the underlying mechanisms of social interaction. To address this issue, we introduce MPCFormer, an explainable socially-aware autonomous driving approach with physics-informed and data-driven coupled social interaction dynamics. In this model, the dynamics are formulated into a discrete space-state representation, which embeds physics priors to enhance modeling explainability. The dynamics coefficients are learned from naturalistic driving data via a Transformer-based encoder-decoder architecture. To the best of our knowledge, MPCFormer is the first approach to explicitly model the dynamics of multi-vehicle social interactions. The learned social interaction dynamics enable the planner to generate manifold, human-like behaviors when interacting with surrounding traffic. By leveraging the MPC framework, the approach mitigates the potential safety risks typically associated with purely learning-based methods. Open-looped evaluation on NGSIM dataset demonstrates that MPCFormer achieves superior social interaction awareness, yielding the lowest trajectory prediction errors compared with other state-of-the-art approach. The prediction achieves an ADE as low as 0.86 m over a long prediction horizon of 5 seconds. Close-looped experiments in highly intense interaction scenarios, where consecutive lane changes are required to exit an off-ramp, further validate the effectiveness of MPCFormer. Results show that MPCFormer achieves the highest planning success rate of 94.67%, improves driving efficiency by 15.75%, and reduces the collision rate from 21.25% to 0.5%, outperforming a frontier Reinforcement Learning (RL) based planner.

**arXiv ID:** 2512.03795
</details>

<details>
<summary><strong>Autonomous Reinforcement Learning Robot Control with Intel's Loihi 2 Neuromorphic Hardware</strong> - Kenneth Stewart, Roxana Leontie, Samantha Chapin, Joe Hays, Sumit Bam Shrestha, Carl Glen Henshaw - [[pdf]](https://arxiv.org/pdf/2512.03911)</summary>

**Abstract:** We present an end-to-end pipeline for deploying reinforcement learning (RL) trained Artificial Neural Networks (ANNs) on neuromorphic hardware by converting them into spiking Sigma-Delta Neural Networks (SDNNs). We demonstrate that an ANN policy trained entirely in simulation can be transformed into an SDNN compatible with Intel's Loihi 2 architecture, enabling low-latency and energy-efficient inference. As a test case, we use an RL policy for controlling the Astrobee free-flying robot, similar to a previously hardware in space-validated controller. The policy, trained with Rectified Linear Units (ReLUs), is converted to an SDNN and deployed on Intel's Loihi 2, then evaluated in NVIDIA's Omniverse Isaac Lab simulation environment for closed-loop control of Astrobee's motion. We compare execution performance between GPU and Loihi 2. The results highlight the feasibility of using neuromorphic platforms for robotic control and establish a pathway toward energy-efficient, real-time neuromorphic computation in future space and terrestrial robotics applications.

**arXiv ID:** 2512.03911
</details>

<details>
<summary><strong>Guided Flow Policy: Learning from High-Value Actions in Offline Reinforcement Learning</strong> - Franki Nguimatsia Tiofack, Théotime Le Hellard, Fabian Schramm, Nicolas Perrin-Gilbert, Justin Carpentier - [[pdf]](https://arxiv.org/pdf/2512.03973)</summary>

**Abstract:** Offline reinforcement learning often relies on behavior regularization that enforces policies to remain close to the dataset distribution. However, such approaches fail to distinguish between high-value and low-value actions in their regularization components. We introduce Guided Flow Policy (GFP), which couples a multi-step flow-matching policy with a distilled one-step actor. The actor directs the flow policy through weighted behavior cloning to focus on cloning high-value actions from the dataset rather than indiscriminately imitating all state-action pairs. In turn, the flow policy constrains the actor to remain aligned with the dataset's best transitions while maximizing the critic. This mutual guidance enables GFP to achieve state-of-the-art performance across 144 state and pixel-based tasks from the OGBench, Minari, and D4RL benchmarks, with substantial gains on suboptimal datasets and challenging tasks. Webpage: this https URL

**arXiv ID:** 2512.03973
</details>

<details>
<summary><strong>CuES: A Curiosity-driven and Environment-grounded Synthesis Framework for Agentic RL</strong> - Shinji Mai, Yunpeng Zhai, Ziqian Chen, Cheng Chen, Anni Zou, Shuchang Tao, Zhaoyang Liu, Bolin Ding - [[pdf]](https://arxiv.org/pdf/2512.01311)</summary>

**Abstract:** Large language model based agents are increasingly deployed in complex, tool augmented environments. While reinforcement learning provides a principled mechanism for such agents to improve through interaction, its effectiveness critically depends on the availability of structured training tasks. In many realistic settings, however, no such tasks exist a challenge we term task scarcity, which has become a key bottleneck for scaling agentic RL. Existing approaches typically assume predefined task collections, an assumption that fails in novel environments where tool semantics and affordances are initially unknown. To address this limitation, we formalize the problem of Task Generation for Agentic RL, where an agent must learn within a given environment that lacks predefined tasks. We propose CuES, a Curiosity driven and Environment grounded Synthesis framework that autonomously generates diverse, executable, and meaningful tasks directly from the environment structure and affordances, without relying on handcrafted seeds or external corpora. CuES drives exploration through intrinsic curiosity, abstracts interaction patterns into reusable task schemas, and refines them through lightweight top down guidance and memory based quality control. Across three representative environments, AppWorld, BFCL, and WebShop, CuES produces task distributions that match or surpass manually curated datasets in both diversity and executability, yielding substantial downstream policy improvements. These results demonstrate that curiosity driven, environment grounded task generation provides a scalable foundation for agents that not only learn how to act, but also learn what to learn. The code is available at this https URL.

**arXiv ID:** 2512.01311
</details>

<details>
<summary><strong>Scaling Multimodal Search and Recommendation with Small Language Models via Upside-Down Reinforcement Learning</strong> - Yu-Chen Lin, Sanat Sharma, Hari Manikandan, Jayant Kumar, Tracy Holloway King, Jing Zheng - [[pdf]](https://arxiv.org/pdf/2502.09854)</summary>

**Abstract:** In this work, we investigate how small language models (SLMs) can be scaled to support multimodal search and recommendation use cases while remaining efficient enough for real-time, resource-constrained deployments. We present a framework that combines upside-down reinforcement learning with synthetic data distillation from a large language model (Llama-3) to train a 100M-parameter GPT-2 model for multitask prompt generation. Despite being up to 80 times smaller than state-of-the-art large language models (LLMs), our SLM achieves relevance and diversity scores within 6% of competitive baselines such as Llama-3 8B, Qwen3 8B, and Ministral 8B. These results demonstrate that SLMs can effectively handle multimodal search and recommendation tasks, while dramatically reducing inference latency and memory overhead. Our study highlights the potential of lightweight models as practical engines for scalable multimodal discovery, bridging the gap between cutting-edge research and real-world multimodal applications such as media recommendations and creative content generation.

**arXiv ID:** 2502.09854
</details>

<details>
<summary><strong>Ergodic Risk Measures: Towards a Risk-Aware Foundation for Continual Reinforcement Learning</strong> - Juan Sebastian Rojas, Chi-Guhn Lee - [[pdf]](https://arxiv.org/pdf/2510.02945)</summary>

**Abstract:** Continual reinforcement learning (continual RL) seeks to formalize the notions of lifelong learning and endless adaptation in RL. In particular, the aim of continual RL is to develop RL agents that can maintain a careful balance between retaining useful information and adapting to new situations. To date, continual RL has been explored almost exclusively through the lens of risk-neutral decision-making, in which the agent aims to optimize the expected long-run performance. In this work, we present the first formal theoretical treatment of continual RL through the lens of risk-aware decision-making, in which the behaviour of the agent is directed towards optimizing a measure of long-run performance beyond the mean. In particular, we show that the classical theory of risk measures, widely used as a theoretical foundation in non-continual risk-aware RL, is, in its current form, incompatible with continual learning. Then, building on this insight, we extend risk measure theory into the continual setting by introducing a new class of ergodic risk measures that are compatible with continual learning. Finally, we provide a case study of risk-aware continual learning, along with empirical results, which show the intuitive appeal of ergodic risk measures in continual settings.

**arXiv ID:** 2510.02945
</details>

<details>
<summary><strong>Stabilizing Reinforcement Learning with LLMs: Formulation and Practices</strong> - Chujie Zheng, Kai Dang, Bowen Yu, Mingze Li, Huiqiang Jiang, Junrong Lin, Yuqiong Liu, Hao Lin, Chencan Wu, Feng Hu, An Yang, Jingren Zhou, Junyang Lin - [[pdf]](https://arxiv.org/pdf/2512.01374)</summary>

**Abstract:** This paper proposes a novel formulation for reinforcement learning (RL) with large language models, explaining why and under what conditions the true sequence-level reward can be optimized via a surrogate token-level objective in policy gradient methods such as REINFORCE. Specifically, through a first-order approximation, we show that this surrogate becomes increasingly valid only when both the training-inference discrepancy and policy staleness are minimized. This insight provides a principled explanation for the crucial role of several widely adopted techniques in stabilizing RL training, including importance sampling correction, clipping, and particularly Routing Replay for Mixture-of-Experts (MoE) models. Through extensive experiments with a 30B MoE model totaling hundreds of thousands of GPU hours, we show that for on-policy training, the basic policy gradient algorithm with importance sampling correction achieves the highest training stability. When off-policy updates are introduced to accelerate convergence, combining clipping and Routing Replay becomes essential to mitigate the instability caused by policy staleness. Notably, once training is stabilized, prolonged optimization consistently yields comparable final performance regardless of cold-start initialization. We hope that the shared insights and the developed recipes for stable RL training will facilitate future research.

**arXiv ID:** 2512.01374
</details>

<details>
<summary><strong>A Diffusion Model Framework for Maximum Entropy Reinforcement Learning</strong> - Sebastian Sanokowski, Kaustubh Patil, Alois Knoll - [[pdf]](https://arxiv.org/pdf/2512.02019)</summary>

**Abstract:** Diffusion models have achieved remarkable success in data-driven learning and in sampling from complex, unnormalized target distributions. Building on this progress, we reinterpret Maximum Entropy Reinforcement Learning (MaxEntRL) as a diffusion model-based sampling problem. We tackle this problem by minimizing the reverse Kullback-Leibler (KL) divergence between the diffusion policy and the optimal policy distribution using a tractable upper bound. By applying the policy gradient theorem to this objective, we derive a modified surrogate objective for MaxEntRL that incorporates diffusion dynamics in a principled way. This leads to simple diffusion-based variants of Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO) and Wasserstein Policy Optimization (WPO), termed DiffSAC, DiffPPO and DiffWPO. All of these methods require only minor implementation changes to their base algorithm. We find that on standard continuous control benchmarks, DiffSAC, DiffPPO and DiffWPO achieve better returns and higher sample efficiency than SAC and PPO.

**arXiv ID:** 2512.02019
</details>

<details>
<summary><strong>RECAP: Transparent Inference-Time Emotion Alignment for Medical Dialogue Systems</strong> - Adarsh Srinivasan, Jacob Dineen, Muhammad Umar Afzal, Muhammad Uzair Sarfraz, Irbaz B. Riaz, Ben Zhou - [[pdf]](https://arxiv.org/pdf/2509.10746)</summary>

**Abstract:** Large language models in healthcare often miss critical emotional cues, delivering medically sound but emotionally flat advice. Such responses are insufficient in clinical encounters, where distressed or vulnerable patients rely on empathic communication to support safety, adherence, and trust. We present RECAP (Reflect-Extract-Calibrate-Align-Produce), an inference-time framework that guides models through structured emotional reasoning without retraining. RECAP decomposes patient input into appraisal-theoretic stages, identifies psychological factors, and assigns Likert-based emotion likelihoods that clinicians can inspect or override, producing nuanced and auditable responses. Across EmoBench, SECEU, and EQ-Bench, RECAP improves emotional reasoning by 22-28% on 8B models and 10-13% on larger models over zero-shot baselines. In blinded evaluations, oncology clinicians rated RECAP's responses as more empathetic, supportive, and context-appropriate than prompting baselines. These findings demonstrate that modular, principled prompting can enhance emotional intelligence in medical AI while maintaining transparency and accountability for clinical deployment.

**arXiv ID:** 2509.10746
</details>

<details>
<summary><strong>From Code Foundation Models to Agents and Applications: A Comprehensive Survey and Practical Guide to Code Intelligence</strong> - Jian Yang, Xianglong Liu, Weifeng Lv, Ken Deng, Shawn Guo, Lin Jing, Yizhi Li, Shark Liu, Xianzhen Luo, Yuyu Luo, Changzai Pan, Ensheng Shi, Yingshui Tan, Renshuai Tao, Jiajun Wu, Xianjie Wu, Zhenhe Wu, Daoguang Zan, Chenchen Zhang, Wei Zhang, He Zhu, Terry Yue Zhuo, Kerui Cao, Xianfu Cheng, Jun Dong, Shengjie Fang, Zhiwei Fei, Xiangyuan Guan, Qipeng Guo, Zhiguang Han, Joseph James, Tianqi Luo, Renyuan Li, Yuhang Li, Yiming Liang, Congnan Liu, Jiaheng Liu, Qian Liu, Ruitong Liu, Tyler Loakman, Xiangxin Meng, Chuang Peng, Tianhao Peng, Jiajun Shi, Mingjie Tang, Boyang Wang, Haowen Wang, Yunli Wang, Fanglin Xu, Zihan Xu, Fei Yuan, Ge Zhang, Jiayi Zhang, Xinhao Zhang, Wangchunshu Zhou, Hualei Zhu, King Zhu, Bryan Dai, Aishan Liu, Zhoujun Li, Chenghua Lin, Tianyu Liu, Chao Peng, Kai Shen, Libo Qin, Shuangyong Song, Zizheng Zhan, Jiajun Zhang, Jie Zhang, Zhaoxiang Zhang, Bo Zheng - [[pdf]](https://arxiv.org/pdf/2511.18538)</summary>

**Abstract:** Large language models (LLMs) have fundamentally transformed automated software development by enabling direct translation of natural language descriptions into functional code, driving commercial adoption through tools like Github Copilot (Microsoft), Cursor (Anysphere), Trae (ByteDance), and Claude Code (Anthropic). While the field has evolved dramatically from rule-based systems to Transformer-based architectures, achieving performance improvements from single-digit to over 95\% success rates on benchmarks like HumanEval. In this work, we provide a comprehensive synthesis and practical guide (a series of analytic and probing experiments) about code LLMs, systematically examining the complete model life cycle from data curation to post-training through advanced prompting paradigms, code pre-training, supervised fine-tuning, reinforcement learning, and autonomous coding agents. We analyze the code capability of the general LLMs (GPT-4, Claude, LLaMA) and code-specialized LLMs (StarCoder, Code LLaMA, DeepSeek-Coder, and QwenCoder), critically examining the techniques, design decisions, and trade-offs. Further, we articulate the research-practice gap between academic research (e.g., benchmarks and tasks) and real-world deployment (e.g., software-related code tasks), including code correctness, security, contextual awareness of large codebases, and integration with development workflows, and map promising research directions to practical needs. Last, we conduct a series of experiments to provide a comprehensive analysis of code pre-training, supervised fine-tuning, and reinforcement learning, covering scaling law, framework selection, hyperparameter sensitivity, model architectures, and dataset comparisons.

**arXiv ID:** 2511.18538
</details>

<details>
<summary><strong>Adaptive sampling using variational autoencoder and reinforcement learning</strong> - Adil Rasheed, Mikael Aleksander Jansen Shahly, Muhammad Faisal Aftab - [[pdf]](https://arxiv.org/pdf/2512.03525)</summary>

**Abstract:** Compressed sensing enables sparse sampling but relies on generic bases and random measurements, limiting efficiency and reconstruction quality. Optimal sensor placement uses historcal data to design tailored sampling patterns, yet its fixed, linear bases cannot adapt to nonlinear or sample-specific variations. Generative model-based compressed sensing improves reconstruction using deep generative priors but still employs suboptimal random sampling. We propose an adaptive sparse sensing framework that couples a variational autoencoder prior with reinforcement learning to select measurements sequentially. Experiments show that this approach outperforms CS, OSP, and Generative model-based reconstruction from sparse measurements.

**arXiv ID:** 2512.03525
</details>

<details>
<summary><strong>Deep Reinforcement Learning for Dynamic Algorithm Configuration: A Case Study on Optimizing OneMax with the (1+($λ$,$λ$))-GA</strong> - Tai Nguyen, Phong Le, André Biedenkapp, Carola Doerr, Nguyen Dang - [[pdf]](https://arxiv.org/pdf/2512.03805)</summary>

**Abstract:** 

**arXiv ID:** 2512.03805
</details>

<details>
<summary><strong>Autonomous Planning In-space Assembly Reinforcement-learning free-flYer (APIARY) International Space Station Astrobee Testing</strong> - Samantha Chapin, Kenneth Stewart, Roxana Leontie, Carl Glen Henshaw - [[pdf]](https://arxiv.org/pdf/2512.03729)</summary>

**Abstract:** The US Naval Research Laboratory's (NRL's) Autonomous Planning In-space Assembly Reinforcement-learning free-flYer (APIARY) experiment pioneers the use of reinforcement learning (RL) for control of free-flying robots in the zero-gravity (zero-G) environment of space. On Tuesday, May 27th 2025 the APIARY team conducted the first ever, to our knowledge, RL control of a free-flyer in space using the NASA Astrobee robot on-board the International Space Station (ISS). A robust 6-degrees of freedom (DOF) control policy was trained using an actor-critic Proximal Policy Optimization (PPO) network within the NVIDIA Isaac Lab simulation environment, randomizing over goal poses and mass distributions to enhance robustness. This paper details the simulation testing, ground testing, and flight validation of this experiment. This on-orbit demonstration validates the transformative potential of RL for improving robotic autonomy, enabling rapid development and deployment (in minutes to hours) of tailored behaviors for space exploration, logistics, and real-time mission needs.

**arXiv ID:** 2512.03729
</details>

<details>
<summary><strong>Crossing the Sim2Real Gap Between Simulation and Ground Testing to Space Deployment of Autonomous Free-flyer Control</strong> - Kenneth Stewart, Samantha Chapin, Roxana Leontie, Carl Glen Henshaw - [[pdf]](https://arxiv.org/pdf/2512.03736)</summary>

**Abstract:** Reinforcement learning (RL) offers transformative potential for robotic control in space. We present the first on-orbit demonstration of RL-based autonomous control of a free-flying robot, the NASA Astrobee, aboard the International Space Station (ISS). Using NVIDIA's Omniverse physics simulator and curriculum learning, we trained a deep neural network to replace Astrobee's standard attitude and translation control, enabling it to navigate in microgravity. Our results validate a novel training pipeline that bridges the simulation-to-reality (Sim2Real) gap, utilizing a GPU-accelerated, scientific-grade simulation environment for efficient Monte Carlo RL training. This successful deployment demonstrates the feasibility of training RL policies terrestrially and transferring them to space-based applications. This paves the way for future work in In-Space Servicing, Assembly, and Manufacturing (ISAM), enabling rapid on-orbit adaptation to dynamic mission requirements.

**arXiv ID:** 2512.03736
</details>

<details>
<summary><strong>Cross-embodied Co-design for Dexterous Hands</strong> - Kehlani Fay, Darin Anthony Djapri, Anya Zorin, James Clinton, Ali El Lahib, Hao Su, Michael T. Tolley, Sha Yi, Xiaolong Wang - [[pdf]](https://arxiv.org/pdf/2512.03743)</summary>

**Abstract:** Dexterous manipulation is limited by both control and design, without consensus as to what makes manipulators best for performing dexterous tasks. This raises a fundamental challenge: how should we design and control robot manipulators that are optimized for dexterity? We present a co-design framework that learns task-specific hand morphology and complementary dexterous control policies. The framework supports 1) an expansive morphology search space including joint, finger, and palm generation, 2) scalable evaluation across the wide design space via morphology-conditioned cross-embodied control, and 3) real-world fabrication with accessible components. We evaluate the approach across multiple dexterous tasks, including in-hand rotation with simulation and real deployment. Our framework enables an end-to-end pipeline that can design, train, fabricate, and deploy a new robotic hand in under 24 hours. The full framework will be open-sourced and available on our website.

**arXiv ID:** 2512.03743
</details>

<details>
<summary><strong>Digital Twin-based Control Co-Design of Full Vehicle Active Suspensions via Deep Reinforcement Learning</strong> - Ying-Kuan Tsai, Yi-Ping Chen, Vispi Karkaria, Wei Chen - [[pdf]](https://arxiv.org/pdf/2512.03891)</summary>

**Abstract:** Active suspension systems are critical for enhancing vehicle comfort, safety, and stability, yet their performance is often limited by fixed hardware designs and control strategies that cannot adapt to uncertain and dynamic operating conditions. Recent advances in digital twins (DTs) and deep reinforcement learning (DRL) offer new opportunities for real-time, data-driven optimization across a vehicle's lifecycle. However, integrating these technologies into a unified framework remains an open challenge. This work presents a DT-based control co-design (CCD) framework for full-vehicle active suspensions using multi-generation design concepts. By integrating automatic differentiation into DRL, we jointly optimize physical suspension components and control policies under varying driver behaviors and environmental uncertainties. DRL also addresses the challenge of partial observability, where only limited states can be sensed and fed back to the controller, by learning optimal control actions directly from available sensor information. The framework incorporates model updating with quantile learning to capture data uncertainty, enabling real-time decision-making and adaptive learning from digital-physical interactions. The approach demonstrates personalized optimization of suspension systems under two distinct driving settings (mild and aggressive). Results show that the optimized systems achieve smoother trajectories and reduce control efforts by approximately 43% and 52% for mild and aggressive, respectively, while maintaining ride comfort and stability. Contributions include: developing a DT-enabled CCD framework integrating DRL and uncertainty-aware model updating for full-vehicle active suspensions, introducing a multi-generation design strategy for self-improving systems, and demonstrating personalized optimization of active suspension systems for distinct driver types.

**arXiv ID:** 2512.03891
</details>

<details>
<summary><strong>Instance-Dependent Continuous-Time Reinforcement Learning via Maximum Likelihood Estimation</strong> - Runze Zhao, Yue Yu, Ruhan Wang, Chunfeng Huang, Dongruo Zhou - [[pdf]](https://arxiv.org/pdf/2508.02103)</summary>

**Abstract:** Continuous-time reinforcement learning (CTRL) provides a natural framework for sequential decision-making in dynamic environments where interactions evolve continuously over time. While CTRL has shown growing empirical success, its ability to adapt to varying levels of problem difficulty remains poorly understood. In this work, we investigate the instance-dependent behavior of CTRL and introduce a simple, model-based algorithm built on maximum likelihood estimation (MLE) with a general function approximator. Unlike existing approaches that estimate system dynamics directly, our method estimates the state marginal density to guide learning. We establish instance-dependent performance guarantees by deriving a regret bound that scales with the total reward variance and measurement resolution. Notably, the regret becomes independent of the specific measurement strategy when the observation frequency adapts appropriately to the problem's complexity. To further improve performance, our algorithm incorporates a randomized measurement schedule that enhances sample efficiency without increasing measurement cost. These results highlight a new direction for designing CTRL algorithms that automatically adjust their learning behavior based on the underlying difficulty of the environment.

**arXiv ID:** 2508.02103
</details>

<details>
<summary><strong>ContactRL: Safe Reinforcement Learning based Motion Planning for Contact based Human Robot Collaboration</strong> - Sundas Rafat Mulkana, Ronyu Yu, Tanaya Guha, Emma Li - [[pdf]](https://arxiv.org/pdf/2512.03707)</summary>

**Abstract:** In collaborative human-robot tasks, safety requires not only avoiding collisions but also ensuring safe, intentional physical contact. We present ContactRL, a reinforcement learning (RL) based framework that directly incorporates contact safety into the reward function through force feedback. This enables a robot to learn adaptive motion profiles that minimize human-robot contact forces while maintaining task efficiency. In simulation, ContactRL achieves a low safety violation rate of 0.2\% with a high task success rate of 87.7\%, outperforming state-of-the-art constrained RL baselines. In order to guarantee deployment safety, we augment the learned policy with a kinetic energy based Control Barrier Function (eCBF) shield. Real-world experiments on an UR3e robotic platform performing small object handovers from a human hand across 360 trials confirm safe contact, with measured normal forces consistently below 10N. These results demonstrate that ContactRL enables safe and efficient physical collaboration, thereby advancing the deployment of collaborative robots in contact-rich tasks.

**arXiv ID:** 2512.03707
</details>

<details>
<summary><strong>Safety Reinforced Model Predictive Control (SRMPC): Improving MPC with Reinforcement Learning for Motion Planning in Autonomous Driving</strong> - Johannes Fischer, Marlon Steiner, Ömer Sahin Tas, Christoph Stiller - [[pdf]](https://arxiv.org/pdf/2512.03774)</summary>

**Abstract:** Model predictive control (MPC) is widely used for motion planning, particularly in autonomous driving. Real-time capability of the planner requires utilizing convex approximation of optimal control problems (OCPs) for the planner. However, such approximations confine the solution to a subspace, which might not contain the global optimum. To address this, we propose using safe reinforcement learning (SRL) to obtain a new and safe reference trajectory within MPC. By employing a learning-based approach, the MPC can explore solutions beyond the close neighborhood of the previous one, potentially finding global optima. We incorporate constrained reinforcement learning (CRL) to ensure safety in automated driving, using a handcrafted energy function-based safety index as the constraint objective to model safe and unsafe regions. Our approach utilizes a state-dependent Lagrangian multiplier, learned concurrently with the safe policy, to solve the CRL problem. Through experimentation in a highway scenario, we demonstrate the superiority of our approach over both MPC and SRL in terms of safety and performance measures.

**arXiv ID:** 2512.03774
</details>

<details>
<summary><strong>HybridWorldSim: A Scalable and Controllable High-fidelity Simulator for Autonomous Driving</strong> - Qiang Li, Yingwenqi Jiang, Tuoxi Li, Duyu Chen, Xiang Feng, Yucheng Ao, Shangyue Liu, Xingchen Yu, Youcheng Cai, Yumeng Liu, Yuexin Ma, Xin Hu, Li Liu, Yu Zhang, Linkun Xu, Bingtao Gao, Xueyuan Wang, Shuchang Zhou, Xianming Liu, Ligang Liu - [[pdf]](https://arxiv.org/pdf/2511.22187)</summary>

**Abstract:** Realistic and controllable simulation is critical for advancing end-to-end autonomous driving, yet existing approaches often struggle to support novel view synthesis under large viewpoint changes or to ensure geometric consistency. We introduce HybridWorldSim, a hybrid simulation framework that integrates multi-traversal neural reconstruction for static backgrounds with generative modeling for dynamic agents. This unified design addresses key limitations of previous methods, enabling the creation of diverse and high-fidelity driving scenarios with reliable visual and spatial consistency. To facilitate robust benchmarking, we further release a new multi-traversal dataset MIRROR that captures a wide range of routes and environmental conditions across different cities. Extensive experiments demonstrate that HybridWorldSim surpasses previous state-of-the-art methods, providing a practical and scalable solution for high-fidelity simulation and a valuable resource for research and development in autonomous driving.

**arXiv ID:** 2511.22187
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
