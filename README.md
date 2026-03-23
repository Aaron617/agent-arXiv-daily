# Agent arXiv Daily

**Last Updated:** 2026-03-23 03:22:31

**Total Papers:** 72

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
<summary><strong>Investigating In-Context Privacy Learning by Integrating User-Facing Privacy Tools into Conversational Agents</strong> - Mohammad Hadi Nezhad, Francisco Enrique Vicente Castro, Ivon Arroyo - [[pdf]](https://arxiv.org/pdf/2603.19416)</summary>

**Abstract:** Supporting users in protecting sensitive information when using conversational agents (CAs) is crucial, as users may undervalue privacy protection due to outdated, partial, or inaccurate knowledge about privacy in CAs. Although privacy knowledge can be developed through standalone resources, it may not readily translate into practice and may remain detached from real-time contexts of use. In this study, we investigate in-context, experiential learning by examining how interactions with privacy tools during chatbot use enhance users' privacy learning. We also explore interface design features that facilitate engagement with these tools and learning about privacy by simulating ChatGPT's interface which we integrated with a just-in-time privacy notice panel. The panel intercepts messages containing sensitive information, warns users about potential sensitivity, offers protective actions, and provides FAQs about privacy in CAs. Participants used versions of the chatbot with and without the privacy panel across two task sessions designed to approximate realistic chatbot use. We qualitatively analyzed participants' pre- and post-test survey responses and think-aloud transcripts and describe findings related to (a) participants' perceptions of privacy before and after the task sessions and (b) interface design features that supported or hindered user-led protection of sensitive information. Finally, we discuss future directions for designing user-facing privacy tools in CAs that promote privacy learning and user engagement in protecting privacy in CAs.

**arXiv ID:** 2603.19416
</details>

<details>
<summary><strong>Mapping Caregiver Needs to AI Chatbot Design: Strengths and Gaps in Mental Health Support for Alzheimer's and Dementia Caregivers</strong> - Jiayue Melissa Shi, Dong Whi Yoo, Keran Wang, Violeta J. Rodriguez, Ravi Karkar, Koustuv Saha - [[pdf]](https://arxiv.org/pdf/2506.15047)</summary>

**Abstract:** Family caregivers of individuals with Alzheimer's Disease and Related Dementia (AD/ADRD) face significant emotional and logistical challenges that place them at heightened risk for stress, anxiety, and depression. Although recent advances in generative AI -- particularly large language models (LLMs) -- offer new opportunities to support mental health, little is known about how caregivers perceive and engage with such technologies. To address this gap, we developed Carey, a GPT-4o-based chatbot designed to provide informational and emotional support to AD/ADRD caregivers. Using Carey as a technology probe, we conducted semi-structured interviews with 16 family caregivers following scenario-driven interactions grounded in common caregiving stressors. Through inductive coding and reflexive thematic analysis, we surface a systemic understanding of caregiver needs and expectations across six themes: on-demand information access, safe space for disclosure, emotional support, crisis management, personalization, and data privacy. For each of these themes, we also identified the nuanced tensions in the caregivers' desires and concerns. We present a mapping of caregiver needs, AI chatbots' strengths, gaps, and design recommendations. Our findings offer theoretical and practical insights to inform the design of proactive, trustworthy, and caregiver-centered AI systems that better support the evolving mental health needs of AD/ADRD caregivers.

**arXiv ID:** 2506.15047
</details>

<details>
<summary><strong>Agent Control Protocol: Admission Control for Agent Actions</strong> - Marcelo Fernandez - [[pdf]](https://arxiv.org/pdf/2603.18829)</summary>

**Abstract:** Agent Control Protocol (ACP) is a formal technical specification for governance of autonomous agents in B2B institutional environments. ACP is the admission control layer between agent intent and system state mutation: before any agent action reaches execution, it must pass a cryptographic admission check that validates identity, capability scope, delegation chain, and policy compliance simultaneously.
ACP defines the mechanisms of cryptographic identity, capability-based authorization, deterministic risk evaluation, verifiable chained delegation, transitive revocation, and immutable auditing that a system must implement for autonomous agents to operate under explicit institutional control.
ACP operates as an additional layer on top of RBAC and Zero Trust, without replacing them. It is designed specifically for the problem that neither model solves: governing what an autonomous agent can do, under what conditions, with what limits, and with complete traceability for external auditing -- including across organizational boundaries.
The v1.14 specification comprises 36 technical documents organized into five conformance levels (L1-L5). It includes a Go reference implementation of 22 packages covering all L1-L4 capabilities, 73 signed conformance test vectors (Ed25519 + SHA-256), and an OpenAPI 3.1.0 specification for all HTTP endpoints. It defines more than 62 verifiable requirements, 12 prohibited behaviors, and the mechanisms for interoperability between institutions.
Specification and implementation: this https URL

**arXiv ID:** 2603.18829
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (12 papers)</h2></summary>

<details>
<summary><strong>HyEvo: Self-Evolving Hybrid Agentic Workflows for Efficient Reasoning</strong> - Beibei Xu, Yutong Ye, Chuyun Shen, Yingbo Zhou, Cheng Chen, Mingsong Chen - [[pdf]](https://arxiv.org/pdf/2603.19639)</summary>

**Abstract:** Although agentic workflows have demonstrated strong potential for solving complex tasks, existing automated generation methods remain inefficient and underperform, as they rely on predefined operator libraries and homogeneous LLM-only workflows in which all task-level computation is performed through probabilistic inference. To address these limitations, we propose HyEvo, an automated workflow-generation framework that leverages heterogeneous atomic synthesis. HyEvo integrates probabilistic LLM nodes for semantic reasoning with deterministic code nodes for rule-based execution, offloading predictable operations from LLM inference and reducing inference cost and execution latency. To efficiently navigate the hybrid search space, HyEvo employs an LLM-driven multi-island evolutionary strategy with a reflect-then-generate mechanism, iteratively refining both workflow topology and node logic via execution feedback. Comprehensive experiments show that HyEvo consistently outperforms existing methods across diverse reasoning and coding benchmarks, while reducing inference cost and execution latency by up to 19$\times$ and 16$\times$, respectively, compared to the state-of-the-art open-source baseline.

**arXiv ID:** 2603.19639
</details>

<details>
<summary><strong>Pitfalls in Evaluating Interpretability Agents</strong> - Tal Haklay, Nikhil Prakash, Sana Pandey, Antonio Torralba, Aaron Mueller, Jacob Andreas, Tamar Rott Shaham, Yonatan Belinkov - [[pdf]](https://arxiv.org/pdf/2603.20101)</summary>

**Abstract:** Automated interpretability systems aim to reduce the need for human labor and scale analysis to increasingly large models and diverse tasks. Recent efforts toward this goal leverage large language models (LLMs) at increasing levels of autonomy, ranging from fixed one-shot workflows to fully autonomous interpretability agents. This shift creates a corresponding need to scale evaluation approaches to keep pace with both the volume and complexity of generated explanations. We investigate this challenge in the context of automated circuit analysis -- explaining the roles of model components when performing specific tasks. To this end, we build an agentic system in which a research agent iteratively designs experiments and refines hypotheses. When evaluated against human expert explanations across six circuit analysis tasks in the literature, the system appears competitive. However, closer examination reveals several pitfalls of replication-based evaluation: human expert explanations can be subjective or incomplete, outcome-based comparisons obscure the research process, and LLM-based systems may reproduce published findings via memorization or informed guessing. To address some of these pitfalls, we propose an unsupervised intrinsic evaluation based on the functional interchangeability of model components. Our work demonstrates fundamental challenges in evaluating complex automated interpretability systems and reveals key limitations of replication-based evaluation.

**arXiv ID:** 2603.20101
</details>

<details>
<summary><strong>Evolving Embodied Intelligence: Graph Neural Network--Driven Co-Design of Morphology and Control in Soft Robotics</strong> - Jianqiang Wang, Shuaiqun Pan, Alvaro Serra-Gomez, Xiaohan Wei, Yue Xie - [[pdf]](https://arxiv.org/pdf/2603.19582)</summary>

**Abstract:** The intelligent behavior of robots does not emerge solely from control systems, but from the tight coupling between body and brain, a principle known as embodied intelligence. Designing soft robots that leverage this interaction remains a significant challenge, particularly when morphology and control require simultaneous optimization. A significant obstacle in this co-design process is that morphological evolution can disrupt learned control strategies, making it difficult to reuse or adapt existing knowledge. We address this by develop a Graph Neural Network-based approach for the co-design of morphology and controller. Each robot is represented as a graph, with a graph attention network (GAT) encoding node features and a pooled representation passed through a multilayer perceptron (MLP) head to produce actuator commands or value estimates. During evolution, inheritance follows a topology-consistent mapping: shared GAT layers are reused, MLP hidden layers are transferred intact, matched actuator outputs are copied, and unmatched ones are randomly initialized and fine-tuned. This morphology-aware policy class lets the controller adapt when the body mutates. On the benchmark, our GAT-based approach achieves higher final fitness and stronger adaptability to morphological variations compared to traditional MLP-only co-design methods. These results indicate that graph-structured policies provide a more effective interface between evolving morphologies and control for embodied intelligence.

**arXiv ID:** 2603.19582
</details>

<details>
<summary><strong>Skilled AI Agents for Embedded and IoT Systems Development</strong> - Yiming Li, Yuhan Cheng, Mingchen Ma, Yihang Zou, Ningyuan Yang, Wei Cheng, Hai "Helen" Li, Yiran Chen, Tingjun Chen - [[pdf]](https://arxiv.org/pdf/2603.19583)</summary>

**Abstract:** Large language models (LLMs) and agentic systems have shown promise for automated software development, but applying them to hardware-in-the-loop (HIL) embedded and Internet-of-Things (IoT) systems remains challenging due to the tight coupling between software logic and physical hardware behavior. Code that compiles successfully may still fail when deployed on real devices because of timing constraints, peripheral initialization requirements, or hardware-specific behaviors. To address this challenge, we introduce a skills-based agentic framework for HIL embedded development together with IoT-SkillsBench, a benchmark designed to systematically evaluate AI agents in real embedded programming environments. IoT-SkillsBench spans three representative embedded platforms, 23 peripherals, and 42 tasks across three difficulty levels, where each task is evaluated under three agent configurations (no-skills, LLM-generated skills, and human-expert skills) and validated through real hardware execution. Across 378 hardware validated experiments, we show that concise human-expert skills with structured expert knowledge enable near-perfect success rates across platforms.

**arXiv ID:** 2603.19583
</details>

<details>
<summary><strong>Pseudo-Simulation for Autonomous Driving</strong> - Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, Kashyap Chitta - [[pdf]](https://arxiv.org/pdf/2506.04218)</summary>

**Abstract:** Existing evaluation paradigms for Autonomous Vehicles (AVs) face critical limitations. Real-world evaluation is often challenging due to safety concerns and a lack of reproducibility, whereas closed-loop simulation can face insufficient realism or high computational costs. Open-loop evaluation, while being efficient and data-driven, relies on metrics that generally overlook compounding errors. In this paper, we propose pseudo-simulation, a novel paradigm that addresses these limitations. Pseudo-simulation operates on real datasets, similar to open-loop evaluation, but augments them with synthetic observations generated prior to evaluation using 3D Gaussian Splatting. Our key idea is to approximate potential future states the AV might encounter by generating a diverse set of observations that vary in position, heading, and speed. Our method then assigns a higher importance to synthetic observations that best match the AV's likely behavior using a novel proximity-based weighting scheme. This enables evaluating error recovery and the mitigation of causal confusion, as in closed-loop benchmarks, without requiring sequential interactive simulation. We show that pseudo-simulation is better correlated with closed-loop simulations ($R^2=0.8$) than the best existing open-loop approach ($R^2=0.7$). We also establish a public leaderboard for the community to benchmark new methodologies with pseudo-simulation. Our code is available at this https URL.

**arXiv ID:** 2506.04218
</details>

<details>
<summary><strong>Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections</strong> - Łukasz Borchmann, Jordy Van Landeghem, Michał Turski, Shreyansh Padarha, Ryan Othniel Kearns, Adam Mahdi, Niels Rogge, Clémentine Fourrier, Siwei Han, Huaxiu Yao, Artemis Llabrés, Yiming Xu, Dimosthenis Karatzas, Hao Zhang, Anupam Datta - [[pdf]](https://arxiv.org/pdf/2603.12180)</summary>

**Abstract:** Multimodal agents offer a promising path to automating complex document-intensive workflows. Yet, a critical question remains: do these agents demonstrate genuine strategic reasoning, or merely stochastic trial-and-error search? To address this, we introduce MADQA, a benchmark of 2,250 human-authored questions grounded in 800 heterogeneous PDF documents. Guided by Classical Test Theory, we design it to maximize discriminative power across varying levels of agentic abilities. To evaluate agentic behaviour, we introduce a novel evaluation protocol measuring the accuracy-effort trade-off. Using this framework, we show that while the best agents can match human searchers in raw accuracy, they succeed on largely different questions and rely on brute-force search to compensate for weak strategic planning. They fail to close the nearly 20% gap to oracle performance, persisting in unproductive loops. We release the dataset and evaluation harness to help facilitate the transition from brute-force retrieval to calibrated, efficient reasoning.

**arXiv ID:** 2603.12180
</details>

<details>
<summary><strong>AgenticRS-EnsNAS: Ensemble-Decoupled Self-Evolving Architecture Search</strong> - Yun Chen, Moyu Zhang, Jinxin Hu, Yu Zhang, Xiaoyi Zeng - [[pdf]](https://arxiv.org/pdf/2603.20014)</summary>

**Abstract:** Neural Architecture Search (NAS) deployment in industrial production systems faces a fundamental validation bottleneck: verifying a single candidate architecture pi requires evaluating the deployed ensemble of M models, incurring prohibitive O(M) computational cost per candidate. This cost barrier severely limits architecture iteration frequency in real-world applications where ensembles (M=50-200) are standard for robustness. This work introduces Ensemble-Decoupled Architecture Search, a framework that leverages ensemble theory to predict system-level performance from single-learner evaluation. We establish the Ensemble-Decoupled Theory with a sufficient condition for monotonic ensemble improvement under homogeneity assumptions: a candidate architecture pi yields lower ensemble error than the current baseline if rho(pi) < rho(pi_old) - (M / (M - 1)) * (Delta E(pi) / sigma^2(pi)), where Delta E, rho, and sigma^2 are estimable from lightweight dual-learner training. This decouples architecture search from full ensemble training, reducing per-candidate search cost from O(M) to O(1) while maintaining O(M) deployment cost only for validated winners. We unify solution strategies across pipeline continuity: (1) closed-form optimization for tractable continuous pi (exemplified by feature bagging in CTR prediction), (2) constrained differentiable optimization for intractable continuous pi, and (3) LLM-driven search with iterative monotonic acceptance for discrete pi. The framework reveals two orthogonal improvement mechanisms -- base diversity gain and accuracy gain -- providing actionable design principles for industrial-scale NAS. All theoretical derivations are rigorous with detailed proofs deferred to the appendix. Comprehensive empirical validation will be included in the journal extension of this work.

**arXiv ID:** 2603.20014
</details>

<details>
<summary><strong>Exploring the Agentic Frontier of Verilog Code Generation</strong> - Patrick Yubeaton, Chinmay Hegde, Siddharth Garg - [[pdf]](https://arxiv.org/pdf/2603.19347)</summary>

**Abstract:** Large language models (LLMs) have made rapid advancements in code generation for popular languages such as Python and C++. Many of these recent gains can be attributed to the use of ``agents'' that wrap domain-relevant tools alongside LLMs. Hardware design languages such as Verilog have also seen improved code generation in recent years, but the impact of agentic frameworks on Verilog code generation tasks remains unclear. In this work, we present the first systematic evaluation of agentic LLMs for Verilog generation, using the recently introduced CVDP benchmark. We also introduce several open-source hardware design agent harnesses, providing a model-agnostic baseline for future work. Through controlled experiments across frontier models, we study how structured prompting and tool design affect performance, analyze agent failure modes and tool usage patterns, compare open-source and closed-source models, and provide qualitative examples of successful and failed agent runs. Our results show that naive agentic wrapping around frontier models can degrade performance (relative to standard forward passes with optimized prompts), but that structured harnesses meaningfully match and in some cases exceed non-agentic baselines. We find that the performance gap between open and closed source models is driven by both higher crash rates and weaker tool output interpretation. Our exploration illuminates the path towards designing special-purpose agents for verilog generation in the future.

**arXiv ID:** 2603.19347
</details>

<details>
<summary><strong>Growing Networks with Autonomous Pruning</strong> - Charles De Lambilly, Stefan Duffner - [[pdf]](https://arxiv.org/pdf/2603.19759)</summary>

**Abstract:** This paper introduces Growing Networks with Autonomous Pruning (GNAP) for image classification. Unlike traditional convolutional neural networks, GNAP change their size, as well as the number of parameters they are using, during training, in order to best fit the data while trying to use as few parameters as possible. This is achieved through two complementary mechanisms: growth and pruning. GNAP start with few parameters, but their size is expanded periodically during training to add more expressive power each time the network has converged to a saturation point. Between these growing phases, model parameters are trained for classification and pruned simultaneously, with complete autonomy by gradient descent. Growing phases allow GNAP to improve their classification performance, while autonomous pruning allows them to keep as few parameters as possible. Experimental results on several image classification benchmarks show that our approach can train extremely sparse neural networks with high accuracy. For example, on MNIST, we achieved 99.44% accuracy with as few as 6.2k parameters, while on CIFAR10, we achieved 92.2\ accuracy with 157.8k parameters.

**arXiv ID:** 2603.19759
</details>

<details>
<summary><strong>Uncertainty Matters: Structured Probabilistic Online Mapping for Motion Prediction in Autonomous Driving</strong> - Pritom Gogoi, Faris Janjoš, Bin Yang, Andreas Look - [[pdf]](https://arxiv.org/pdf/2603.20076)</summary>

**Abstract:** Online map generation and trajectory prediction are critical components of the autonomous driving perception-prediction-planning pipeline. While modern vectorized mapping models achieve high geometric accuracy, they typically treat map estimation as a deterministic task, discarding structural uncertainty. Existing probabilistic approaches often rely on diagonal covariance matrices, which assume independence between points and fail to capture the strong spatial correlations inherent in road geometry. To address this, we propose a structured probabilistic formulation for online map generation. Our method explicitly models intra-element dependencies by predicting a dense covariance matrix, parameterized via a Low-Rank plus Diagonal (LRPD) covariance decomposition. This formulation represents uncertainty as a combination of a low-rank component, which captures global spatial structure, and a diagonal component representing independent local noise, thereby capturing geometric correlations without the prohibitive computational cost of full covariance matrices. Evaluations on the nuScenes dataset demonstrate that our uncertainty-aware framework yields consistent improvements in online map generation quality compared to deterministic baselines. Furthermore, our approach establishes new state-of-the-art performance for map-based motion prediction, highlighting the critical role of uncertainty in planning tasks. Code is published under link-available-soon.

**arXiv ID:** 2603.20076
</details>

<details>
<summary><strong>DynFlowDrive: Flow-Based Dynamic World Modeling for Autonomous Driving</strong> - Xiaolu Liu, Yicong Li, Song Wang, Junbo Chen, Angela Yao, Jianke Zhu - [[pdf]](https://arxiv.org/pdf/2603.19675)</summary>

**Abstract:** Recently, world models have been incorporated into the autonomous driving systems to improve the planning reliability. Existing approaches typically predict future states through appearance generation or deterministic regression, which limits their ability to capture trajectory-conditioned scene evolution and leads to unreliable action planning. To address this, we propose DynFlowDrive, a latent world model that leverages flow-based dynamics to model the transition of world states under different driving actions. By adopting the rectifiedflow formulation, the model learns a velocity field that describes how the scene state changes under different driving actions, enabling progressive prediction of future latent states. Building upon this, we further introduce a stability-aware multi-mode trajectory selection strategy that evaluates candidate trajectories according to the stability of the induced scene transitions. Extensive experiments on the nuScenes and NavSim benchmarks demonstrate consistent improvements across diverse driving frameworks without introducing additional inference overhead. Source code will be abaliable at this https URL.

**arXiv ID:** 2603.19675
</details>

<details>
<summary><strong>LIORNet: Self-Supervised LiDAR Snow Removal Framework for Autonomous Driving under Adverse Weather Conditions</strong> - Ji-il Park, Inwook Shim - [[pdf]](https://arxiv.org/pdf/2603.19936)</summary>

**Abstract:** LiDAR sensors provide high-resolution 3D perception and long-range detection, making them indispensable for autonomous driving and robotics. However, their performance significantly degrades under adverse weather conditions such as snow, rain, and fog, where spurious noise points dominate the point cloud and lead to false perception. To address this problem, various approaches have been proposed: distance-based filters exploiting spatial sparsity, intensity-based filters leveraging reflectance distributions, and learning-based methods that adapt to complex environments. Nevertheless, distance-based methods struggle to distinguish valid object points from noise, intensity-based methods often rely on fixed thresholds that lack adaptability to changing conditions, and learning-based methods suffer from the high cost of annotation, limited generalization, and computational overhead. In this study, we propose LIORNet, which eliminates these drawbacks and integrates the strengths of all three paradigms. LIORNet is built upon a U-Net++ backbone and employs a self-supervised learning strategy guided by pseudo-labels generated from multiple physical and statistical cues, including range-dependent intensity thresholds, snow reflectivity, point sparsity, and sensing range constraints. This design enables LIORNet to distinguish noise points from environmental structures without requiring manual annotations, thereby overcoming the difficulty of snow labeling and the limitations of single-principle approaches. Extensive experiments on the WADS and CADC datasets demonstrate that LIORNet outperforms state-of-the-art filtering algorithms in both accuracy and runtime while preserving critical environmental features. These results highlight LIORNet as a practical and robust solution for LiDAR perception in extreme weather, with strong potential for real-time deployment in autonomous driving systems.

**arXiv ID:** 2603.19936
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>A Subgoal-driven Framework for Improving Long-Horizon LLM Agents</strong> - Taiyi Wang, Sian Gooding, Florian Hartmann, Oriana Riva, Edward Grefenstette - [[pdf]](https://arxiv.org/pdf/2603.19685)</summary>

**Abstract:** Large language model (LLM)-based agents have emerged as powerful autonomous controllers for digital environments, including mobile interfaces, operating systems, and web browsers. Web navigation, for example, requires handling dynamic content and long sequences of actions, making it particularly challenging. Existing LLM-based agents struggle with long-horizon planning in two main ways. During online execution, they often lose track as new information arrives, lacking a clear and adaptive path toward the final goal. This issue is further exacerbated during reinforcement learning (RL) fine-tuning, where sparse and delayed rewards make it difficult for agents to identify which actions lead to success, preventing them from maintaining coherent reasoning over extended tasks. To address these challenges, we propose two contributions. First, we introduce an agent framework that leverages proprietary models for online planning through subgoal decomposition. Second, we present MiRA (Milestoning your Reinforcement Learning Enhanced Agent), an RL training framework that uses dense, milestone-based reward signals. The real-time planning mechanism improves proprietary models such as Gemini by approximately a 10% absolute increase in success rate (SR) on the WebArena-Lite benchmark. Meanwhile, applying MiRA to the open Gemma3-12B model increases its success rate from 6.4% to 43.0%. This performance surpasses proprietary systems such as GPT-4-Turbo (17.6%) and GPT-4o (13.9%), as well as the previous open-model state of the art, WebRL (38.4%). Overall, our findings demonstrate that combining explicit inference-time planning with milestone-based rewards significantly improves an agent's long-horizon capabilities, paving the way for more robust and general-purpose autonomous systems.

**arXiv ID:** 2603.19685
</details>

<details>
<summary><strong>Utility-Guided Agent Orchestration for Efficient LLM Tool Use</strong> - Boyan Liu, Gongming Zhao, Hongli Xu - [[pdf]](https://arxiv.org/pdf/2603.19896)</summary>

**Abstract:** Tool-using large language model (LLM) agents often face a fundamental tension between answer quality and execution cost. Fixed workflows are stable but inflexible, while free-form multi-step reasoning methods such as ReAct may improve task performance at the expense of excessive tool calls, longer trajectories, higher token consumption, and increased latency. In this paper, we study agent orchestration as an explicit decision problem rather than leaving it entirely to prompt-level behavior. We propose a utility-guided orchestration policy that selects among actions such as respond, retrieve, tool call, verify, and stop by balancing estimated gain, step cost, uncertainty, and redundancy. Our goal is not to claim universally best task performance, but to provide a controllable and analyzable policy framework for studying quality-cost trade-offs in tool-using LLM agents. Experiments across direct answering, threshold control, fixed workflows, ReAct, and several policy variants show that explicit orchestration signals substantially affect agent behavior. Additional analyses on cost definitions, workflow fairness, and redundancy control further demonstrate that lightweight utility design can provide a defensible and practical mechanism for agent control.

**arXiv ID:** 2603.19896
</details>

<details>
<summary><strong>The Autonomy Tax: Defense Training Breaks LLM Agents</strong> - Shawn Li, Yue Zhao - [[pdf]](https://arxiv.org/pdf/2603.19423)</summary>

**Abstract:** Large language model (LLM) agents increasingly rely on external tools (file operations, API calls, database transactions) to autonomously complete complex multi-step tasks. Practitioners deploy defense-trained models to protect against prompt injection attacks that manipulate agent behavior through malicious observations or retrieved content. We reveal a fundamental \textbf{capability-alignment paradox}: defense training designed to improve safety systematically destroys agent competence while failing to prevent sophisticated attacks. Evaluating defended models against undefended baselines across 97 agent tasks and 1,000 adversarial prompts, we uncover three systematic biases unique to multi-step agents. \textbf{Agent incompetence bias} manifests as immediate tool execution breakdown, with models refusing or generating invalid actions on benign tasks before observing any external content. \textbf{Cascade amplification bias} causes early failures to propagate through retry loops, pushing defended models to timeout on 99\% of tasks compared to 13\% for baselines. \textbf{Trigger bias} leads to paradoxical security degradation where defended models perform worse than undefended baselines while straightforward attacks bypass defenses at high rates. Root cause analysis reveals these biases stem from shortcut learning: models overfit to surface attack patterns rather than semantic threat understanding, evidenced by extreme variance in defense effectiveness across attack categories. Our findings demonstrate that current defense paradigms optimize for single-turn refusal benchmarks while rendering multi-step agents fundamentally unreliable, necessitating new approaches that preserve tool execution competence under adversarial conditions.

**arXiv ID:** 2603.19423
</details>

<details>
<summary><strong>A Framework for Formalizing LLM Agent Security</strong> - Vincent Siu, Jingxuan He, Kyle Montgomery, Zhun Wang, Neil Gong, Chenguang Wang, Dawn Song - [[pdf]](https://arxiv.org/pdf/2603.19469)</summary>

**Abstract:** Security in LLM agents is inherently contextual. For example, the same action taken by an agent may represent legitimate behavior or a security violation depending on whose instruction led to the action, what objective is being pursued, and whether the action serves that objective. However, existing definitions of security attacks against LLM agents often fail to capture this contextual nature. As a result, defenses face a fundamental utility-security tradeoff: applying defenses uniformly across all contexts can lead to significant utility loss, while applying defenses in insufficient or inappropriate contexts can result in security vulnerabilities. In this work, we present a framework that systematizes existing attacks and defenses from the perspective of contextual security. To this end, we propose four security properties that capture contextual security for LLM agents: task alignment (pursuing authorized objectives), action alignment (individual actions serving those objectives), source authorization (executing commands from authenticated sources), and data isolation (ensuring information flows respect privilege boundaries). We further introduce a set of oracle functions that enable verification of whether these security properties are violated as an agent executes a user task. Using this framework, we reformalize existing attacks, such as indirect prompt injection, direct prompt injection, jailbreak, task drift, and memory poisoning, as violations of one or more security properties, thereby providing precise and contextual definitions of these attacks. Similarly, we reformalize defenses as mechanisms that strengthen oracle functions or perform security property checks. Finally, we discuss several important future research directions enabled by our framework.

**arXiv ID:** 2603.19469
</details>

<details>
<summary><strong>Agentic Harness for Real-World Compilers</strong> - Yingwei Zheng, Cong Li, Shaohua Li, Yuqun Zhang, Zhendong Su - [[pdf]](https://arxiv.org/pdf/2603.20075)</summary>

**Abstract:** Compilers are critical to modern computing, yet fixing compiler bugs is difficult. While recent large language model (LLM) advancements enable automated bug repair, compiler bugs pose unique challenges due to their complexity, deep cross-domain expertise requirements, and sparse, non-descriptive bug reports, necessitating compiler-specific tools. To bridge the gap, we introduce llvm-autofix, the first agentic harness designed to assist LLM agents in understanding and fixing compiler bugs. Our focus is on LLVM, one of the most widely used compiler infrastructures. Central to llvm-autofix are agent-friendly LLVM tools, a benchmark llvm-bench of reproducible LLVM bugs, and a tailored minimal agent llvm-autofix-mini for fixing LLVM bugs. Our evaluation demonstrates a performance decline of 60% in frontier models when tackling compiler bugs compared with common software bugs. Our minimal agent llvm-autofix-mini also outperforms the state-of-the-art by approximately 22%. This emphasizes the necessity for specialized harnesses like ours to close the loop between LLMs and compiler engineering. We believe this work establishes a foundation for advancing LLM capabilities in complex systems like compilers. GitHub: this https URL

**arXiv ID:** 2603.20075
</details>

<details>
<summary><strong>PlanTwin: Privacy-Preserving Planning Abstractions for Cloud-Assisted LLM Agents</strong> - Guangsheng Yu, Qin Wang, Rui Lang, Shuai Su, Xu Wang - [[pdf]](https://arxiv.org/pdf/2603.18377)</summary>

**Abstract:** Cloud-hosted large language models (LLMs) have become the de facto planners in agentic systems, coordinating tools and guiding execution over local environments. In many deployments, however, the environment being planned over is private, containing source code, files, credentials, and metadata that cannot be exposed to the cloud. Existing solutions address adjacent concerns, such as execution isolation, access control, or confidential inference, but they do not control what cloud planners observe during planning: within the permitted scope, \textit{raw environment state is still exposed}.
We introduce PlanTwin, a privacy-preserving architecture for cloud-assisted planning without exposing raw local context. The key idea is to project the real environment into a \textit{planning-oriented digital twin}: a schema-constrained and de-identified abstract graph that preserves planning-relevant structure while removing reconstructable details. The cloud planner operates solely on this sanitized twin through a bounded capability interface, while a local gatekeeper enforces safety policies and cumulative disclosure budgets. We further formalize the privacy-utility trade-off as a capability granularity problem, define architectural privacy goals using $(k,\delta)$-anonymity and $\epsilon$-unlinkability, and mitigate compositional leakage through multi-turn disclosure control.
We implement PlanTwin as middleware between local agents and cloud planners and evaluate it on 60 agentic tasks across ten domains with four cloud planners. PlanTwin achieves full sensitive-item non-disclosure (SND = 1.0) while maintaining planning quality close to full-context systems: three of four planners achieve PQS $> 0.79$, and the full pipeline incurs less than 2.2\% utility loss.

**arXiv ID:** 2603.18377
</details>

<details>
<summary><strong>Is Your LLM-as-a-Recommender Agent Trustable? LLMs' Recommendation is Easily Hacked by Biases (Preferences)</strong> - Zichen Tang, Zirui Zhang, Qian Wang, Zhenheng Tang, Bo Li, Xiaowen Chu - [[pdf]](https://arxiv.org/pdf/2603.17417)</summary>

**Abstract:** Current Large Language Models (LLMs) are gradually exploited in practically valuable agentic workflows such as Deep Research, E-commerce recommendation, and job recruitment. In these applications, LLMs need to select some optimal solutions from massive candidates, which we term as \textit{LLM-as-a-Recommender} paradigm. However, the reliability of using LLM agents for recommendations is underexplored. In this work, we introduce a \textbf{Bias} \textbf{Rec}ommendation \textbf{Bench}mark (\textbf{BiasRecBench}) to highlight the critical vulnerability of such agents to biases in high-value real-world tasks. The benchmark includes three practical domains: paper review, e-commerce, and job recruitment. We construct a \textsc{Bias Synthesis Pipeline with Calibrated Quality Margins} that 1) synthesizes evaluation data by controlling the quality gap between optimal and sub-optimal options to provide a calibrated testbed to elicit the vulnerability to biases; 2) injects contextual biases that are logical and suitable for option contexts. Extensive experiments on both SOTA (Gemini-{2.5,3}-pro, GPT-4o, DeepSeek-R1) and small-scale LLMs reveal that agents frequently succumb to injected biases despite having sufficient reasoning capabilities to identify the ground truth. These findings expose a significant reliability bottleneck in current agentic workflows, calling for specialized alignment strategies for LLM-as-a-Recommender. The complete code and evaluation datasets will be made publicly available shortly.

**arXiv ID:** 2603.17417
</details>

<details>
<summary><strong>DLLM Agent: See Farther, Run Faster</strong> - Huiling Zhen, Weizhe Lin, Renxi Liu, Kai Han, Yiming Li, Yuchuan Tian, Hanting Chen, Xiaoguang Li, Xiaosong Li, Chen Chen, Xianzhi Yu, Mingxuan Yuan, Youliang Yan, Peifeng Qin, Jun Wang, Yu Wang, Dacheng Tao, Yunhe Wang - [[pdf]](https://arxiv.org/pdf/2602.07451)</summary>

**Abstract:** Diffusion large language models (DLLMs) have emerged as an alternative to autoregressive (AR) decoding with appealing efficiency and modeling properties, yet their implications for agentic multi-step decision making remain underexplored. We ask a concrete question: when the generation paradigm is changed but the agent framework and supervision are held fixed, do diffusion backbones induce systematically different planning and tool-use behaviors, and do these differences translate into end-to-end efficiency gains? We study this in a controlled setting by instantiating DLLM and AR backbones within the same agent workflow (DeepDiver) and performing matched agent-oriented fine-tuning on the same trajectory data, yielding diffusion-backed DLLM Agents and directly comparable AR agents. Across benchmarks and case studies, we find that, at comparable accuracy, DLLM Agents are on average over 30% faster end to end than AR agents, with some cases exceeding 8x speedup. Conditioned on correct task completion, DLLM Agents also require fewer interaction rounds and tool invocations, consistent with higher planner hit rates that converge earlier to a correct action path with less backtracking. We further identify two practical considerations for deploying diffusion backbones in tool-using agents. First, naive DLLM policies are more prone to structured tool-call failures, necessitating stronger tool-call-specific training to emit valid schemas and arguments. Second, for multi-turn inputs interleaving context and action spans, diffusion-style span corruption requires aligned attention masking to avoid spurious context-action information flow; without such alignment, performance degrades. Finally, we analyze attention dynamics across workflow stages and observe paradigm-specific coordination patterns, suggesting stronger global planning signals in diffusion-backed agents.

**arXiv ID:** 2602.07451
</details>

<details>
<summary><strong>Memori: A Persistent Memory Layer for Efficient, Context-Aware LLM Agents</strong> - Luiz C. Borro, Luiz A. B. Macarini, Gordon Tindall, Michael Montero, Adam B. Struck - [[pdf]](https://arxiv.org/pdf/2603.19935)</summary>

**Abstract:** As large language models (LLMs) evolve into autonomous agents, persistent memory at the API layer is essential for enabling context-aware behavior across LLMs and multi-session interactions. Existing approaches force vendor lock-in and rely on injecting large volumes of raw conversation into prompts, leading to high token costs and degraded performance.
We introduce Memori, an LLM-agnostic persistent memory layer that treats memory as a data structuring problem. Its Advanced Augmentation pipeline converts unstructured dialogue into compact semantic triples and conversation summaries, enabling precise retrieval and coherent reasoning.
Evaluated on the LoCoMo benchmark, Memori achieves 81.95% accuracy, outperforming existing memory systems while using only 1,294 tokens per query (~5% of full context). This results in substantial cost reductions, including 67% fewer tokens than competing approaches and over 20x savings compared to full-context methods.
These results show that effective memory in LLM agents depends on structured representations instead of larger context windows, enabling scalable and cost-efficient deployment.

**arXiv ID:** 2603.19935
</details>

<details>
<summary><strong>Automated Membership Inference Attacks: Discovering MIA Signal Computations using LLM Agents</strong> - Toan Tran, Olivera Kotevska, Li Xiong - [[pdf]](https://arxiv.org/pdf/2603.19375)</summary>

**Abstract:** Membership inference attacks (MIAs), which enable adversaries to determine whether specific data points were part of a model's training dataset, have emerged as an important framework to understand, assess, and quantify the potential information leakage associated with machine learning systems. Designing effective MIAs is a challenging task that usually requires extensive manual exploration of model behaviors to identify potential vulnerabilities. In this paper, we introduce AutoMIA -- a novel framework that leverages large language model (LLM) agents to automate the design and implementation of new MIA signal computations. By utilizing LLM agents, we can systematically explore a vast space of potential attack strategies, enabling the discovery of novel strategies. Our experiments demonstrate AutoMIA can successfully discover new MIAs that are specifically tailored to user-configured target model and dataset, resulting in improvements of up to 0.18 in absolute AUC over existing MIAs. This work provides the first demonstration that LLM agents can serve as an effective and scalable paradigm for designing and implementing MIAs with SOTA performance, opening up new avenues for future exploration.

**arXiv ID:** 2603.19375
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (16 papers)</h2></summary>

<details>
<summary><strong>PowerLens: Taming LLM Agents for Safe and Personalized Mobile Power Management</strong> - Xingyu Feng, Chang Sun, Yuzhu Wang, Zhangbing Zhou, Chengwen Luo, Zhuangzhuang Chen, Xiaomin Ouyang, Huanqi Yang - [[pdf]](https://arxiv.org/pdf/2603.19584)</summary>

**Abstract:** Battery life remains a critical challenge for mobile devices, yet existing power management mechanisms rely on static rules or coarse-grained heuristics that ignore user activities and personal preferences. We present PowerLens, a system that tames the reasoning power of Large Language Models (LLMs) for safe and personalized mobile power management on Android devices. The key idea is that LLMs' commonsense reasoning can bridge the semantic gap between user activities and system parameters, enabling zero-shot, context-aware policy generation that adapts to individual preferences through implicit feedback. PowerLens employs a multi-agent architecture that recognizes user context from UI semantics and generates holistic power policies across 18 device parameters. A PDL-based constraint framework verifies every action before execution, while a two-tier memory system learns individualized preferences from implicit user overrides through confidence-based distillation, requiring no explicit configuration and converging within 3--5 days. Extensive experiments on a rooted Android device show that PowerLens achieves 81.7% action accuracy and 38.8% energy saving over stock Android, outperforming rule-based and LLM-based baselines, with high user satisfaction, fast preference convergence, and strong safety guarantees, with the system itself consuming only 0.5% of daily battery capacity.

**arXiv ID:** 2603.19584
</details>

<details>
<summary><strong>Framing Effects in Independent-Agent Large Language Models: A Cross-Family Behavioral Analysis</strong> - Zice Wang, Zhenyu Zhang - [[pdf]](https://arxiv.org/pdf/2603.19282)</summary>

**Abstract:** In many real-world applications, large language models (LLMs) operate as independent agents without interaction, thereby limiting coordination. In this setting, we examine how prompt framing influences decisions in a threshold voting task involving individual-group interest conflict. Two logically equivalent prompts with different framings were tested across diverse LLM families under isolated trials. Results show that prompt framing significantly influences choice distributions, often shifting preferences toward risk-averse options. Surface linguistic cues can even override logically equivalent formulations. This suggests that observed behavior reflects a tendency consistent with a preference for instrumental rather than cooperative rationality when success requires risk-bearing. The findings highlight framing effects as a significant bias source in non-interacting multi-agent LLM deployments, informing alignment and prompt design.

**arXiv ID:** 2603.19282
</details>

<details>
<summary><strong>TrustFlow: Topic-Aware Vector Reputation Propagation for Multi-Agent Ecosystems</strong> - Volodymyr Seliuchenko - [[pdf]](https://arxiv.org/pdf/2603.19452)</summary>

**Abstract:** We introduce TrustFlow, a reputation propagation algorithm that assigns each software agent a multi-dimensional reputation vector rather than a scalar score. Reputation is propagated through an interaction graph via topic-gated transfer operators that modulate each edge by its content embedding, with convergence to a unique fixed point guaranteed by the contraction mapping theorem. We develop a family of Lipschitz-1 transfer operators and composable information-theoretic gates that achieve up to 98% multi-label Precision@5 on dense graphs and 78% on sparse ones. On a benchmark of 50 agents across 8 domains, TrustFlow resists sybil attacks, reputation laundering, and vote rings with at most 4 percentage-point precision impact. Unlike PageRank and Topic-Sensitive PageRank, TrustFlow produces vector reputation that is directly queryable by dot product in the same embedding space as user queries.

**arXiv ID:** 2603.19452
</details>

<details>
<summary><strong>GoAgent: Group-of-Agents Communication Topology Generation for LLM-based Multi-Agent Systems</strong> - Hongjiang Chen, Xin Zheng, Yixin Liu, Pengfei Jiao, Shiyuan Li, Huan Liu, Zhidong Zhao, Ziqi Xu, Ibrahim Khalil, Shirui Pan - [[pdf]](https://arxiv.org/pdf/2603.19677)</summary>

**Abstract:** Large language model (LLM)-based multi-agent systems (MAS) have demonstrated exceptional capabilities in solving complex tasks, yet their effectiveness depends heavily on the underlying communication topology that coordinates agent interactions. Within these systems, successful problem-solving often necessitates task-specific group structures to divide and conquer subtasks. However, most existing approaches generate communication topologies in a node-centric manner, leaving group structures to emerge implicitly from local connectivity decisions rather than modeling them explicitly, often leading to suboptimal coordination and unnecessary communication overhead. To address this limitation, we propose GoAgent (Group-of-Agents), a communication topology generation method that explicitly treats collaborative groups as the atomic units of MAS construction. Specifically, GoAgent first enumerates task-relevant candidate groups through an LLM and then autoregressively selects and connects these groups as atomic units to construct the final communication graph, jointly capturing intra-group cohesion and inter-group coordination. To mitigate communication redundancy and noise propagation inherent in expanding topologies, we further introduce a conditional information bottleneck (CIB) objective that compresses inter-group communication, preserving task-relevant signals while filtering out redundant historical noise. Extensive experiments on six benchmarks demonstrate the state-of-the-art performance of GoAgent with 93.84% average accuracy while reducing token consumption by about 17%.

**arXiv ID:** 2603.19677
</details>

<details>
<summary><strong>An Agentic Multi-Agent Architecture for Cybersecurity Risk Management</strong> - Ravish Gupta, Saket Kumar, Shreeya Sharma, Maulik Dang, Abhishek Aggarwal - [[pdf]](https://arxiv.org/pdf/2603.20131)</summary>

**Abstract:** Getting a real cybersecurity risk assessment for a small organization is expensive -- a NIST CSF-aligned engagement runs $15,000 on the low end, takes weeks, and depends on practitioners who are genuinely scarce. Most small companies skip it entirely. We built a six-agent AI system where each agent handles one analytical stage: profiling the organization, mapping assets, analyzing threats, evaluating controls, scoring risks, and generating recommendations. Agents share a persistent context that grows as the assessment proceeds, so later agents build on what earlier ones concluded -- the mechanism that distinguishes this from standard sequential agent pipelines. We tested it on a 15-person HIPAA-covered healthcare company and compared outputs to independent assessments by three CISSP practitioners -- the system agreed with them 85% of the time on severity classifications, covered 92% of identified risks, and finished in under 15 minutes. We then ran 30 repeated single-agent assessments across five synthetic but sector-realistic organizational profiles in healthcare, fintech, manufacturing, retail, and SaaS, comparing a general-purpose Mistral-7B against a domain fine-tuned model. Both completed every run. The fine-tuned model flagged threats the baseline could not see at all: PHI exposure in healthcare, OT/IIoT vulnerabilities in manufacturing, platform-specific risks in retail. The full multi-agent pipeline, however, failed every one of 30 attempts on a Tesla T4 with its 4,096-token default context window -- context capacity, not model quality, turned out to be the binding constraint.

**arXiv ID:** 2603.20131
</details>

<details>
<summary><strong>AI Agents Can Already Autonomously Perform Experimental High Energy Physics</strong> - Eric A. Moreno, Samuel Bright-Thonney, Andrzej Novak, Dolores Garcia, Philip Harris - [[pdf]](https://arxiv.org/pdf/2603.20179)</summary>

**Abstract:** Large language model-based AI agents are now able to autonomously execute substantial portions of a high energy physics (HEP) analysis pipeline with minimal expert-curated input. Given access to a HEP dataset, an execution framework, and a corpus of prior experimental literature, we find that Claude Code succeeds in automating all stages of a typical analysis: event selection, background estimation, uncertainty quantification, statistical inference, and paper drafting. We argue that the experimental HEP community is underestimating the current capabilities of these systems, and that most proposed agentic workflows are too narrowly scoped or scaffolded to specific analysis structures. We present a proof-of-concept framework, Just Furnish Context (JFC), that integrates autonomous analysis agents with literature-based knowledge retrieval and multi-agent review, and show that this is sufficient to plan, execute, and document a credible high energy physics analysis. We demonstrate this by conducting analyses on open data from ALEPH, DELPHI, and CMS to perform electroweak, QCD, and Higgs boson measurements. Rather than replacing physicists, these tools promise to offload the repetitive technical burden of analysis code development, freeing researchers to focus on physics insight, truly novel method development, and rigorous validation. Given these developments, we advocate for new strategies for how the community trains students, organizes analysis efforts, and allocates human expertise.

**arXiv ID:** 2603.20179
</details>

<details>
<summary><strong>Agentic Business Process Management: A Research Manifesto</strong> - Diego Calvanese, Angelo Casciani, Giuseppe De Giacomo, Marlon Dumas, Fabiana Fournier, Timotheus Kampik, Emanuele La Malfa, Lior Limonad, Andrea Marrella, Andreas Metzger, Marco Montali, Daniel Amyot, Peter Fettke, Artem Polyvyanyy, Stefanie Rinderle-Ma, Sebastian Sardiña, Niek Tax, Barbara Weber - [[pdf]](https://arxiv.org/pdf/2603.18916)</summary>

**Abstract:** This paper presents a manifesto that articulates the conceptual foundations of Agentic Business Process Management (APM), an extension of Business Process Management (BPM) for governing autonomous agents executing processes in organizations. From a management perspective, APM represents a paradigm shift from the traditional process view of the business process, driven by the realization of process awareness and an agent-oriented abstraction, where software and human agents act as primary functional entities that perceive, reason, and act within explicit process frames. This perspective marks a shift from traditional, automation-oriented BPM toward systems in which autonomy is constrained, aligned, and made operational through process awareness.
We introduce the core abstractions and architectural elements required to realize APM systems and elaborate on four key capabilities that such APM agents must support: framed autonomy, explainability, conversational actionability, and self-modification. These capabilities jointly ensure that agents' goals are aligned with organizational goals and that agents behave in a framed yet proactive manner in pursuing those goals. We discuss the extent to which the capabilities can be realized and identify research challenges whose resolution requires further advances in BPM, AI, and multi-agent systems. The manifesto thus serves as a roadmap for bridging these communities and for guiding the development of APM systems in practice.

**arXiv ID:** 2603.18916
</details>

<details>
<summary><strong>HALO: Hierarchical Reinforcement Learning for Large-Scale Adaptive Traffic Signal Control</strong> - Yaqiao Zhu, Hongkai Wen, Geyong Min, Man Luo - [[pdf]](https://arxiv.org/pdf/2506.14391)</summary>

**Abstract:** Adaptive traffic signal control (ATSC) is essential for mitigating urban congestion in modern smart cities, where traffic infrastructure is evolving into interconnected Web-of-Things (WoT) environments with thousands of sensing-and-control nodes. However, existing methods face a critical scalability-coordination tradeoff: centralized approaches optimize global objectives but become computationally intractable at city scale, while decentralized multi-agent methods scale efficiently yet lack network-level coherence, resulting in suboptimal performance. In this paper, we present HALO, a hierarchical reinforcement learning framework that addresses this tradeoff for large-scale ATSC. HALO decouples decision-making into two levels: a high-level global guidance policy employs Transformer-LSTM encoders to model spatio-temporal dependencies across the entire network and broadcast compact guidance signals, while low-level local intersection policies execute decentralized control conditioned on both local observations and global context. To ensure better alignment of global-local objectives, we introduce an adversarial goal-setting mechanism where the global policy proposes challenging-yet-feasible network-level targets that local policies are trained to surpass, fostering robust coordination. We evaluate HALO extensively on multiple standard benchmarks, and a newly constructed large-scale Manhattan-like network with 2,668 intersections under real-world traffic patterns, including peak transitions, adverse weather and holiday surges. Results demonstrate HALO shows competitive performance and becomes increasingly dominant as network complexity grows across small-scale benchmarks, while delivering the strongest performance in all large-scale regimes, offering up to 6.8% lower average travel time and 5.0% lower average delay than the best state-of-the-art.

**arXiv ID:** 2506.14391
</details>

<details>
<summary><strong>ClawWorm: Self-Propagating Attacks Across LLM Agent Ecosystems</strong> - Yihao Zhang, Zeming Wei, Xiaokun Luan, Chengcan Wu, Zhixin Zhang, Jiangrong Wu, Haolin Wu, Huanran Chen, Jun Sun, Meng Sun - [[pdf]](https://arxiv.org/pdf/2603.15727)</summary>

**Abstract:** Autonomous LLM-based agents increasingly operate as long-running processes forming densely interconnected multi-agent ecosystems, whose security properties remain largely unexplored. In particular, OpenClaw, an open-source platform with over 40,000 active instances, has stood out recently with its persistent configurations, tool-execution privileges, and cross-platform messaging capabilities. In this work, we present ClawWorm, the first self-replicating worm attack against a production-scale agent framework, achieving a fully autonomous infection cycle initiated by a single message: the worm first hijacks the victim's core configuration to establish persistent presence across session restarts, then executes an arbitrary payload upon each reboot, and finally propagates itself to every newly encountered peer without further attacker intervention. We evaluate the attack on a controlled testbed across four distinct LLM backends, three infection vectors, and three payload types (1,800 total trials). We demonstrate a 64.5\% aggregate attack success rate, sustained multi-hop propagation, and reveal stark divergences in model security postures -- highlighting that while execution-level filtering effectively mitigates dormant payloads, skill supply chains remain universally vulnerable. We analyse the architectural root causes underlying these vulnerabilities and propose defence strategies targeting each identified trust boundary. Code and samples will be released upon completion of responsible disclosure.

**arXiv ID:** 2603.15727
</details>

<details>
<summary><strong>Helix: A Dual-Helix Co-Evolutionary Multi-Agent System for Prompt Optimization and Question Reformulation</strong> - Kewen Zhu, Liping Yi, Zhiming Zhao, Xiang Li, Qinghua Hu - [[pdf]](https://arxiv.org/pdf/2603.19732)</summary>

**Abstract:** Automated prompt optimization (APO) aims to improve large language model performance by refining prompt instructions. However, existing methods are largely constrained by fixed prompt templates, limited search spaces, or single-sided optimization that treats user questions as immutable inputs. In practice, question formulation and prompt design are inherently interdependent: clearer question structures facilitate focused reasoning and task understanding, while effective prompts reveal better ways to organize and restate queries. Ignoring this coupling fundamentally limits the effectiveness and adaptability of current APO approaches. We propose a unified multi-agent system (Helix) that jointly optimizes question reformulation and prompt instructions through a structured three-stage co-evolutionary framework. Helix integrates (1) planner-guided decomposition that breaks optimization into coupled question-prompt objectives, (2) dual-track co-evolution where specialized agents iteratively refine and critique each other to produce complementary improvements, and (3) strategy-driven question generation that instantiates high-quality reformulations for robust inference. Extensive experiments on 12 benchmarks against 6 strong baselines demonstrate the effectiveness of Helix, achieving up to 3.95% performance improvements across tasks with favorable optimization efficiency.

**arXiv ID:** 2603.19732
</details>

<details>
<summary><strong>Beyond detection: cooperative multi-agent reasoning for rapid onboard EO crisis response</strong> - Alejandro D. Mousist, Pedro Delgado de Robles Martín, Raquel Lladró Climent, Julian Cobos Aparicio - [[pdf]](https://arxiv.org/pdf/2603.19858)</summary>

**Abstract:** Rapid identification of hazardous events is essential for next-generation Earth Observation (EO) missions supporting disaster response. However, current monitoring pipelines remain largely ground-centric, introducing latency due to downlink limitations, multi-source data fusion constraints, and the computational cost of exhaustive scene analysis.
This work proposes a hierarchical multi-agent architecture for onboard EO processing under strict resource and bandwidth constraints. The system enables the exploitation of complementary multimodal observations by coordinating specialized AI agents within an event-driven decision pipeline. AI agents can be deployed across multiple nodes in a distributed setting, such as satellite platforms. An Early Warning agent generates fast hypotheses from onboard observations and selectively activates domain-specific analysis agents, while a Decision agent consolidates the evidence to issue a final alert. The architecture combines vision-language models, traditional remote sensing analysis tools, and role-specialized agents to enable structured reasoning over multimodal observations while minimizing unnecessary computation.
A proof-of-concept implementation was executed on the engineering model of an edge-computing platform currently deployed in orbit, using representative satellite data. Experiments on wildfire and flood monitoring scenarios show that the proposed routing-based pipeline significantly reduces computational overhead while maintaining coherent decision outputs, demonstrating the feasibility of distributed agent-based reasoning for future autonomous EO constellations.

**arXiv ID:** 2603.19858
</details>

<details>
<summary><strong>A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning</strong> - Yichang Xu, Gaowen Liu, Ramana Rao Kompella, Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Zachary Yahn, Ling Liu - [[pdf]](https://arxiv.org/pdf/2603.14052)</summary>

**Abstract:** This paper presents a multi-agent perception-action exploration alliance, dubbed A4VL, for efficient long-video reasoning. A4VL operates in a multi-round perception-action exploration loop with a selection of VLM agents. In each round, the team of agents performs video question-answer (VideoQA) via perception exploration followed by action exploration. During perception exploration, each agent learns to extract query-specific perception clue(s) from a few sampled frames and performs clue-based alignment to find the video block(s) that are most relevant to the query-specific event. During action exploration, A4VL performs video reasoning in three steps: (1) each agent produces its initial answer with rational, (2) all agents collaboratively scores one another through cross-reviews and relevance ranking, and (3) based on whether a satisfactory consensus is reached, the decision is made either to start a new round of perception-action deliberation by pruning (e.g., filtering out the lowest performing agent) and re-staging (e.g., new-clue and matching block based perception-action exploration), or to conclude by producing its final answer. The integration of the multi-agent alliance through multi-round perception-action exploration, coupled with event-driven partitioning and cue-guided block alignment, enables A4VL to effectively scale to real world long videos while preserving high quality video reasoning. Evaluation Results on five popular VideoQA benchmarks show that A4VL outperforms 18 existing representative VLMs and 11 recent methods optimized for long-video reasoning, while achieving significantly lower inference latency. Our code is released at this https URL.

**arXiv ID:** 2603.14052
</details>

<details>
<summary><strong>Autonoma: A Hierarchical Multi-Agent Framework for End-to-End Workflow Automation</strong> - Eslam Reda, Maged Yasser, Sara El-Metwally - [[pdf]](https://arxiv.org/pdf/2603.19270)</summary>

**Abstract:** The increasing complexity of user demands necessitates automation frameworks that can reliably translate open-ended instructions into robust, multi-step workflows. Current monolithic agent architectures often struggle with the challenges of scalability, error propagation, and maintaining focus across diverse tasks. This paper introduces Autonoma, a structured, hierarchical multi-agent framework designed for end-to-end workflow automation from natural language prompts. Autonoma employs a principled, multi-tiered architecture where a high-level Coordinator validates user intent, a Planner generates structured workflows, and a Supervisor dynamically manages the execution by orchestrating a suite of modular, specialized agents (e.g., for web browsing, coding, file management). This clear separation between orchestration logic and specialized execution ensures robustness through active monitoring and error handling, while enabling extensibility by allowing new capabilities to be integrated as plug-and-play agents without modifying the core engine. Implemented as a fully functional system operating within a secure LAN environment, Autonoma addresses critical data privacy and reliability concerns. The system is further engineered for inclusivity, accepting multi-modal input (text, voice, image, files) and supporting both English and Arabic. Autonoma achieved a 97% task completion rate and a 98% successful agent handoff rate, confirming its operational reliability and efficient collaboration.

**arXiv ID:** 2603.19270
</details>

<details>
<summary><strong>An Agentic Approach to Generating XAI-Narratives</strong> - Yifan He, David Martens - [[pdf]](https://arxiv.org/pdf/2603.20003)</summary>

**Abstract:** Explainable AI (XAI) research has experienced substantial growth in recent years. Existing XAI methods, however, have been criticized for being technical and expert-oriented, motivating the development of more interpretable and accessible explanations. In response, large language model (LLM)-generated XAI narratives have been proposed as a promising approach for translating post-hoc explanations into more accessible, natural-language explanations. In this work, we propose a multi-agent framework for XAI narrative generation and refinement. The framework comprises the Narrator, which generates and revises narratives based on feedback from multiple Critic Agents on faithfulness and coherence metrics, thereby enabling narrative improvement through iteration. We design five agentic systems (Basic Design, Critic Design, Critic-Rule Design, Coherent Design, and Coherent-Rule Design) and systematically evaluate their effectiveness across five LLMs on five tabular datasets. Results validate that the Basic Design, the Critic Design, and the Critic-Rule Design are effective in improving the faithfulness of narratives across all LLMs. Claude-4.5-Sonnet on Basic Design performs best, reducing the number of unfaithful narratives by 90% after three rounds of iteration. To address recurrent issues, we further introduce an ensemble strategy based on majority voting. This approach consistently enhances performance for four LLMs, except for DeepSeek-V3.2-Exp. These findings highlight the potential of agentic systems to produce faithful and coherent XAI narratives.

**arXiv ID:** 2603.20003
</details>

<details>
<summary><strong>LiRA: A Multi-Agent Framework for Reliable and Readable Literature Review Generation</strong> - Gregory Hok Tjoan Go, Khang Ly, Anders Søgaard, Amin Tabatabaei, Maarten de Rijke, Xinyi Chen - [[pdf]](https://arxiv.org/pdf/2510.05138)</summary>

**Abstract:** The rapid growth of scientific publications has made it increasingly difficult to keep literature reviews comprehensive and up-to-date. Though prior work has focused on automating retrieval and screening, the writing phase of systematic reviews remains largely under-explored, especially with regard to readability and factual accuracy. To address this, we present LiRA (Literature Review Agents), a multi-agent collaborative workflow which emulates the human literature review process. LiRA utilizes specialized agents for content outlining, subsection writing, editing, and reviewing, producing cohesive and comprehensive review articles. Evaluated on SciReviewGen and a proprietary ScienceDirect dataset, LiRA outperforms current baselines such as AutoSurvey and MASS-Survey in writing and citation quality, while maintaining competitive similarity to human-written reviews. We further evaluate LiRA in real-world scenarios using document retrieval and assess its robustness to reviewer model variation. Our findings highlight the potential of agentic LLM workflows, even without domain-specific tuning, to improve the reliability and usability of automated scientific writing.

**arXiv ID:** 2510.05138
</details>

<details>
<summary><strong>Multi-Agent Motion Planning on Industrial Magnetic Levitation Platforms: A Hybrid ADMM-HOCBF approach</strong> - Bavo Tistaert, Stan Servaes, Alejandro Gonzalez-Garcia, Ibrahim Ibrahim, Louis Callens, Jan Swevers, Wilm Decré - [[pdf]](https://arxiv.org/pdf/2603.19838)</summary>

**Abstract:** This paper presents a novel hybrid motion planning method for holonomic multi-agent systems. The proposed decentralised model predictive control (MPC) framework tackles the intractability of classical centralised MPC for a growing number of agents while providing safety guarantees. This is achieved by combining a decentralised version of the alternating direction method of multipliers (ADMM) with a centralised high-order control barrier function (HOCBF) architecture. Simulation results show significant improvement in scalability over classical centralised MPC. We validate the efficacy and real-time capability of the proposed method by developing a highly efficient C++ implementation and deploying the resulting trajectories on a real industrial magnetic levitation platform.

**arXiv ID:** 2603.19838
</details>

</details>

<details open>
<summary><h2>Other Agent Research (4 papers)</h2></summary>

<details>
<summary><strong>Embodied Science: Closing the Discovery Loop with Agentic Embodied AI</strong> - Xiang Zhuang, Chenyi Zhou, Kehua Feng, Zhihui Zhu, Yunfan Gao, Yijie Zhong, Yichi Zhang, Junjie Huang, Keyan Ding, Lei Bai, Haofen Wang, Qiang Zhang, Huajun Chen - [[pdf]](https://arxiv.org/pdf/2603.19782)</summary>

**Abstract:** Artificial intelligence has demonstrated remarkable capability in predicting scientific properties, yet scientific discovery remains an inherently physical, long-horizon pursuit governed by experimental cycles. Most current computational approaches are misaligned with this reality, framing discovery as isolated, task-specific predictions rather than continuous interaction with the physical world. Here, we argue for embodied science, a paradigm that reframes scientific discovery as a closed loop tightly coupling agentic reasoning with physical execution. We propose a unified Perception-Language-Action-Discovery (PLAD) framework, wherein embodied agents perceive experimental environments, reason over scientific knowledge, execute physical interventions, and internalize outcomes to drive subsequent exploration. By grounding computational reasoning in robust physical feedback, this approach bridges the gap between digital prediction and empirical validation, offering a roadmap for autonomous discovery systems in the life and chemical sciences.

**arXiv ID:** 2603.19782
</details>

<details>
<summary><strong>Planning Autonomous Vehicle Maneuvering in Work Zones Through Game-Theoretic Trajectory Generation</strong> - Mayar Nour, Atrisha Sarkar, Mohamed H. Zaki - [[pdf]](https://arxiv.org/pdf/2603.19556)</summary>

**Abstract:** Work zone navigation remains one of the most challenging manoeuvres for autonomous vehicles (AVs), where constrained geometries and unpredictable traffic patterns create a high-risk environment. Despite extensive research on AV trajectory planning, few studies address the decision-making required to navigate work zones safely. This paper proposes a novel game-theoretic framework for trajectory generation and control to enhance the safety of lane changes in a work zone environment. By modelling the lane change manoeuvre as a non-cooperative game between vehicles, we use a game-theoretic planner to generate trajectories that balance safety, progress, and traffic stability. The simulation results show that the proposed game-theoretic model reduces the frequency of conflicts by 35 percent and decreases the probability of high risk safety events compared to traditional vehicle behaviour planning models in safety-critical highway work-zone scenarios.

**arXiv ID:** 2603.19556
</details>

<details>
<summary><strong>From Tokens To Agents: A Researcher's Guide To Understanding Large Language Models</strong> - Daniele Barolo - [[pdf]](https://arxiv.org/pdf/2603.19269)</summary>

**Abstract:** Researchers face a critical choice: how to use -- or not use -- large language models in their work. Using them well requires understanding the mechanisms that shape what LLMs can and cannot do. This chapter makes LLMs comprehensible without requiring technical expertise, breaking down six essential components: pre-training data, tokenization and embeddings, transformer architecture, probabilistic generation, alignment, and agentic capabilities. Each component is analyzed through both technical foundations and research implications, identifying specific affordances and limitations. Rather than prescriptive guidance, the chapter develops a framework for reasoning critically about whether and how LLMs fit specific research needs, finally illustrated through an extended case study on simulating social media dynamics with LLM-based agents.

**arXiv ID:** 2603.19269
</details>

<details>
<summary><strong>All-Mem: Agentic Lifelong Memory via Dynamic Topology Evolution</strong> - Can Lv, Heng Chang, Yuchen Guo, Shengyu Tao, Shiji Zhou - [[pdf]](https://arxiv.org/pdf/2603.19595)</summary>

**Abstract:** Lifelong interactive agents are expected to assist users over months or years, which requires continually writing long term memories while retrieving the right evidence for each new query under fixed context and latency budgets. Existing memory systems often degrade as histories grow, yielding redundant, outdated, or noisy retrieved contexts. We present All-Mem, an online/offline lifelong memory framework that maintains a topology structured memory bank via explicit, non destructive consolidation, avoiding the irreversible information loss typical of summarization based compression. In online operation, it anchors retrieval on a bounded visible surface to keep coarse search cost bounded. Periodically offline, an LLM diagnoser proposes confidence scored topology edits executed with gating using three operators: SPLIT, MERGE, and UPDATE, while preserving immutable evidence for traceability. At query time, typed links enable hop bounded, budgeted expansion from active anchors to archived evidence when needed. Experiments on LOCOMO and LONGMEMEVAL show improved retrieval and QA over representative baselines.

**arXiv ID:** 2603.19595
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (27 papers)</h2></summary>

<details>
<summary><strong>Hyperagents</strong> - Jenny Zhang, Bingchen Zhao, Wannan Yang, Jakob Foerster, Jeff Clune, Minqi Jiang, Sam Devlin, Tatiana Shavrina - [[pdf]](https://arxiv.org/pdf/2603.19461)</summary>

**Abstract:** Self-improving AI systems aim to reduce reliance on human engineering by learning to improve their own learning and problem-solving processes. Existing approaches to self-improvement rely on fixed, handcrafted meta-level mechanisms, fundamentally limiting how fast such systems can improve. The Darwin Gödel Machine (DGM) demonstrates open-ended self-improvement in coding by repeatedly generating and evaluating self-modified variants. Because both evaluation and self-modification are coding tasks, gains in coding ability can translate into gains in self-improvement ability. However, this alignment does not generally hold beyond coding domains. We introduce \textbf{hyperagents}, self-referential agents that integrate a task agent (which solves the target task) and a meta agent (which modifies itself and the task agent) into a single editable program. Crucially, the meta-level modification procedure is itself editable, enabling metacognitive self-modification, improving not only the task-solving behavior, but also the mechanism that generates future improvements. We instantiate this framework by extending DGM to create DGM-Hyperagents (DGM-H), eliminating the assumption of domain-specific alignment between task performance and self-modification skill to potentially support self-accelerating progress on any computable task. Across diverse domains, the DGM-H improves performance over time and outperforms baselines without self-improvement or open-ended exploration, as well as prior self-improving systems. Furthermore, the DGM-H improves the process by which it generates new agents (e.g., persistent memory, performance tracking), and these meta-level improvements transfer across domains and accumulate across runs. DGM-Hyperagents offer a glimpse of open-ended AI systems that do not merely search for better solutions, but continually improve their search for how to improve.

**arXiv ID:** 2603.19461
</details>

<details>
<summary><strong>Teaching an Agent to Sketch One Part at a Time</strong> - Xiaodan Du, Ruize Xu, David Yunis, Yael Vinker, Greg Shakhnarovich - [[pdf]](https://arxiv.org/pdf/2603.19500)</summary>

**Abstract:** We develop a method for producing vector sketches one part at a time. To do this, we train a multi-modal language model-based agent using a novel multi-turn process-reward reinforcement learning following supervised fine-tuning. Our approach is enabled by a new dataset we call ControlSketch-Part, containing rich part-level annotations for sketches, obtained using a novel, generic automatic annotation pipeline that segments vector sketches into semantic parts and assigns paths to parts with a structured multi-stage labeling process. Our results indicate that incorporating structured part-level data and providing agent with the visual feedback through the process enables interpretable, controllable, and locally editable text-to-vector sketch generation.

**arXiv ID:** 2603.19500
</details>

<details>
<summary><strong>PA2D-MORL: Pareto Ascent Directional Decomposition based Multi-Objective Reinforcement Learning</strong> - Tianmeng Hu, Biao Luo - [[pdf]](https://arxiv.org/pdf/2603.19579)</summary>

**Abstract:** Multi-objective reinforcement learning (MORL) provides an effective solution for decision-making problems involving conflicting objectives. However, achieving high-quality approximations to the Pareto policy set remains challenging, especially in complex tasks with continuous or high-dimensional state-action space. In this paper, we propose the Pareto Ascent Directional Decomposition based Multi-Objective Reinforcement Learning (PA2D-MORL) method, which constructs an efficient scheme for multi-objective problem decomposition and policy improvement, leading to a superior approximation of Pareto policy set. The proposed method leverages Pareto ascent direction to select the scalarization weights and computes the multi-objective policy gradient, which determines the policy optimization direction and ensures joint improvement on all objectives. Meanwhile, multiple policies are selectively optimized under an evolutionary framework to approximate the Pareto frontier from different directions. Additionally, a Pareto adaptive fine-tuning approach is applied to enhance the density and spread of the Pareto frontier approximation. Experiments on various multi-objective robot control tasks show that the proposed method clearly outperforms the current state-of-the-art algorithm in terms of both quality and stability of the outcomes.

**arXiv ID:** 2603.19579
</details>

<details>
<summary><strong>Experience is the Best Teacher: Motivating Effective Exploration in Reinforcement Learning for LLMs</strong> - Wenjian Zhang, Kongcheng Zhang, Jiaxin Qi, Baisheng Lai, Jianqiang Huang - [[pdf]](https://arxiv.org/pdf/2603.20046)</summary>

**Abstract:** Reinforcement Learning (RL) with rubric-based rewards has recently shown remarkable progress in enhancing general reasoning capabilities of Large Language Models (LLMs), yet still suffers from ineffective exploration confined to curent policy distribution. In fact, RL optimization can be viewed as steering the policy toward an ideal distribution that maximizes the rewards, while effective exploration should align efforts with desired target. Leveraging this insight, we propose HeRL, a Hindsight experience guided Reinforcement Learning framework to bootstrap effective exploration by explicitly telling LLMs the desired behaviors specified in rewards. Concretely, HeRL treats failed trajectories along with their unmet rubrics as hindsight experience, which serves as in-context guidance for the policy to explore desired responses beyond its current distribution. Additionally, we introduce a bonus reward to incentivize responses with greater potential for improvement under such guidance. HeRL facilitates effective learning from desired high quality samples without repeated trial-and-error from scratch, yielding a more accurate estimation of the expected gradient theoretically. Extensive experiments across various benchmarks demonstrate that HeRL achieves superior performance gains over baselines, and can further benefit from experience guided self-improvement at test time. Our code is available at this https URL.

**arXiv ID:** 2603.20046
</details>

<details>
<summary><strong>Beyond the Desk: Barriers and Future Opportunities for AI to Assist Scientists in Embodied Physical Tasks</strong> - Irene Hou, Alexander Qin, Lauren Cheng, Philip J. Guo - [[pdf]](https://arxiv.org/pdf/2603.19504)</summary>

**Abstract:** More scientists are now using AI, but prior studies have examined only how they use it 'at the desk' for computer-based work. However, given that scientific work often happens 'beyond the desk' at lab and field sites, we conducted the first study of how scientific practitioners use AI for embodied physical tasks. We interviewed 12 scientific practitioners doing hands-on lab and fieldwork in domains like nuclear fusion, primate cognition, and biochemistry, and found three barriers to AI adoption in these settings: 1) experimental setups are too high-stakes to risk AI errors, 2) constrained environments make it hard to use AI, and 3) AI cannot match the tacit knowledge of humans. Participants then developed speculative designs for future AI assistants to 1) monitor task status, 2) organize lab-wide knowledge, 3) monitor scientists' health, 4) do field scouting, 5) do hands-on chores. Our findings point toward AI as background infrastructure to support physical work rather than replacing human expertise.

**arXiv ID:** 2603.19504
</details>

<details>
<summary><strong>DeepStock: Reinforcement Learning with Policy Regularizations for Inventory Management</strong> - Yaqi Xie, Xinru Hao, Jiaxi Liu, Will Ma, Linwei Xin, Lei Cao, Yidong Zhang - [[pdf]](https://arxiv.org/pdf/2603.19621)</summary>

**Abstract:** Deep Reinforcement Learning (DRL) provides a general-purpose methodology for training inventory policies that can leverage big data and compute. However, off-the-shelf implementations of DRL have seen mixed success, often plagued by high sensitivity to the hyperparameters used during training. In this paper, we show that by imposing policy regularizations, grounded in classical inventory concepts such as "Base Stock", we can significantly accelerate hyperparameter tuning and improve the final performance of several DRL methods. We report details from a 100% deployment of DRL with policy regularizations on Alibaba's e-commerce platform, Tmall. We also include extensive synthetic experiments, which show that policy regularizations reshape the narrative on what is the best DRL method for inventory management.

**arXiv ID:** 2603.19621
</details>

<details>
<summary><strong>PolicySim: An LLM-Based Agent Social Simulation Sandbox for Proactive Policy Optimization</strong> - Renhong Huang, Ning Tang, Jiarong Xu, Yuxuan Cao, Qingqian Tu, Sheng Guo, Bo Zheng, Huiyuan Liu, Yang Yang - [[pdf]](https://arxiv.org/pdf/2603.19649)</summary>

**Abstract:** Social platforms serve as central hubs for information exchange, where user behaviors and platform interventions jointly shape opinions. However, intervention policies like recommendation and content filtering, can unintentionally amplify echo chambers and polarization, posing significant societal risks. Proactively evaluating the impact of such policies is therefore crucial. Existing approaches primarily rely on reactive online A/B testing, where risks are identified only after deployment, making risk identification delayed and costly. LLM-based social simulations offer a promising pre-deployment alternative, but current methods fall short in realistically modeling platform interventions and incorporating feedback from the platform. Bridging these gaps is essential for building actionable frameworks to assess and optimize platform policies. To this end, we propose PolicySim, an LLM-based social simulation sandbox for the proactive assessment and optimization of intervention policies. PolicySim models the bidirectional dynamics between user behavior and platform interventions through two key components: (1) a user agent module refined via supervised fine-tuning (SFT) and direct preference optimization (DPO) to achieve platform-specific behavioral realism; and (2) an adaptive intervention module that employs a contextual bandit with message passing to capture dynamic network structures. Experiments show that PolicySim can accurately simulate platform ecosystems at both micro and macro levels and support effective intervention policy.

**arXiv ID:** 2603.19649
</details>

<details>
<summary><strong>What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time</strong> - Dong Yan, Jian Liang, Yanbo Wang, Shuo Lu, Ran He, Tieniu Tan - [[pdf]](https://arxiv.org/pdf/2603.19880)</summary>

**Abstract:** Test-Time Reinforcement Learning (TTRL) enables Large Language Models (LLMs) to enhance reasoning capabilities on unlabeled test streams by deriving pseudo-rewards from majority voting consensus. However, existing TTRL methods rely exclusively on positive pseudo-labeling strategies. Such reliance becomes vulnerable under challenging scenarios where answer distributions are highly dispersed, resulting in weak consensus that inadvertently reinforces incorrect trajectories as supervision signals. In this paper, we propose SCRL (Selective-Complementary Reinforcement Learning), a robust test-time reinforcement learning framework that effectively mitigates label noise amplification. SCRL develops Selective Positive Pseudo-Labeling, which enforces strict consensus criteria to filter unreliable majorities. Complementarily, SCRL introduces Entropy-Gated Negative Pseudo-Labeling, the first negative supervision mechanism in TTRL, to reliably prune incorrect trajectories based on generation uncertainty. Extensive experiments on multiple reasoning benchmarks demonstrate that SCRL achieves substantial improvements over baselines, while maintaining robust generalization and training stability under constrained rollout budgets. Our code is available at this https URL.

**arXiv ID:** 2603.19880
</details>

<details>
<summary><strong>Fine-tuning Timeseries Predictors Using Reinforcement Learning</strong> - Hugo Cazaux, Ralph Rudd, Hlynur Stefánsson, Sverrir Ólafsson, Eyjólfur Ingi Ásgeirsson - [[pdf]](https://arxiv.org/pdf/2603.20063)</summary>

**Abstract:** This chapter presents three major reinforcement learning algorithms used for fine-tuning financial forecasters. We propose a clear implementation plan for backpropagating the loss of a reinforcement learning task to a model trained using supervised learning, and compare the performance before and after the fine-tuning. We find an increase in performance after fine-tuning, and transfer learning properties to the models, indicating the benefits of fine-tuning. We also highlight the tuning process and empirical results for future implementation by practitioners.

**arXiv ID:** 2603.20063
</details>

<details>
<summary><strong>Chain-of-Adaptation: Surgical Vision-Language Adaptation with Reinforcement Learning</strong> - Jiajie Li, Chenhui Xu, Meihuan Liu, Jinjun Xiong - [[pdf]](https://arxiv.org/pdf/2603.20116)</summary>

**Abstract:** Conventional fine-tuning on domain-specific datasets can inadvertently alter a model's pretrained multimodal priors, leading to reduced generalization. To address this, we propose Chain-of-Adaptation (CoA), an adaptation framework designed to integrate domain knowledge while maintaining the model's inherent reasoning and perceptual capabilities. CoA introduces a structured reasoning format that enhances domain alignment without sacrificing general multimodal competence by reinforcement learning. Experiments on standard surgical benchmarks, under both in-distribution and out-of-distribution settings, demonstrate that CoA achieves higher accuracy, stronger generalization, and more stable behavior than supervised fine-tuning. Furthermore, ablation studies confirm that CoA effectively preserves the model's core visual-language abilities, providing a reliable pathway for domain specialization in VLMs.

**arXiv ID:** 2603.20116
</details>

<details>
<summary><strong>VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking</strong> - Jingyang Lin, Jialian Wu, Jiang Liu, Ximeng Sun, Ze Wang, Xiaodong Yu, Jiebo Luo, Zicheng Liu, Emad Barsoum - [[pdf]](https://arxiv.org/pdf/2603.20185)</summary>

**Abstract:** Video agentic models have advanced challenging video-language tasks. However, most agentic approaches still heavily rely on greedy parsing over densely sampled video frames, resulting in high computational cost. We present VideoSeek, a long-horizon video agent that leverages video logic flow to actively seek answer-critical evidence instead of exhaustively parsing the full video. This insight allows the model to use far fewer frames while maintaining, or even improving, its video understanding capability. VideoSeek operates in a think-act-observe loop with a well-designed toolkit for collecting multi-granular video observations. This design enables query-aware exploration over accumulated observations and supports practical video understanding and reasoning. Experiments on four challenging video understanding and reasoning benchmarks demonstrate that VideoSeek achieves strong accuracy while using far fewer frames than prior video agents and standalone LMMs. Notably, VideoSeek achieves a 10.2 absolute points improvement on LVBench over its base model, GPT-5, while using 93% fewer frames. Further analysis highlights the significance of leveraging video logic flow, strong reasoning capability, and the complementary roles of toolkit design.

**arXiv ID:** 2603.20185
</details>

<details>
<summary><strong>Average Reward Reinforcement Learning for Omega-Regular and Mean-Payoff Objectives</strong> - Milad Kazemi, Mateo Perez, Fabio Somenzi, Sadegh Soudjani, Ashutosh Trivedi, Alvaro Velasquez - [[pdf]](https://arxiv.org/pdf/2505.15693)</summary>

**Abstract:** Recent advances in reinforcement learning (RL) have renewed interest in reward design for shaping agent behavior, but manually crafting reward functions is tedious and error-prone. A principled alternative is to specify behavioral requirements in a formal, unambiguous language and automatically compile them into learning objectives. $\omega$-regular languages are a natural fit, given their role in formal verification and synthesis. However, most existing $\omega$-regular RL approaches operate in an episodic, discounted setting with periodic resets, which is misaligned with $\omega$-regular semantics over infinite traces. For continuing tasks, where the agent interacts with the environment over a single uninterrupted lifetime, the average-reward criterion is more appropriate.
We focus on absolute liveness specifications, a subclass of $\omega$-regular languages that cannot be violated by any finite prefix and thus aligns naturally with continuing interaction. We present the first model-free RL framework that translates absolute liveness specifications into average-reward objectives and enables learning in unknown communicating Markov decision processes (MDPs) without episodic resetting. We also introduce a reward structure for lexicographic multi-objective optimization: among policies that maximize the satisfaction probability of an absolute liveness specification, the agent maximizes an external average-reward objective. Our method guarantees convergence in unknown communicating MDPs and supports on-the-fly reductions that do not require full environment knowledge, enabling model-free learning. Experiments across several benchmarks show that the continuing, average-reward approach outperforms competing discount-based methods.

**arXiv ID:** 2505.15693
</details>

<details>
<summary><strong>Evaluation-Aware Reinforcement Learning</strong> - Shripad Vilasrao Deshmukh, Will Schwarzer, Scott Niekum - [[pdf]](https://arxiv.org/pdf/2509.19464)</summary>

**Abstract:** Policy evaluation is a core component of many reinforcement learning (RL) algorithms and a critical tool for ensuring safe deployment of RL policies. However, existing policy evaluation methods often suffer from high variance or bias. To address these issues, we introduce Evaluation-Aware Reinforcement Learning (EvA-RL), a general policy learning framework that considers evaluation accuracy at train-time, as opposed to standard post-hoc policy evaluation methods. Specifically, EvA-RL directly optimizes policies for efficient and accurate evaluation, in addition to being performant. We provide an instantiation of EvA-RL and demonstrate through a combination of theoretical analysis and empirical results that EvA-RL effectively trades off between evaluation accuracy and expected return. Finally, we show that the evaluation-aware policy and the evaluation mechanism itself can be co-learned to mitigate this tradeoff, providing the evaluation benefits without significantly sacrificing policy performance. This work opens a new line of research that elevates reliable evaluation to a first-class principle in reinforcement learning.

**arXiv ID:** 2509.19464
</details>

<details>
<summary><strong>Cross-site scripting adversarial attacks based on deep reinforcement learning: Evaluation and extension study</strong> - Samuele Pasini, Gianluca Maragliano, Jinhan Kim, Paolo Tonella - [[pdf]](https://arxiv.org/pdf/2502.19095)</summary>

**Abstract:** Cross-site scripting (XSS) poses a significant threat to web application security. While Deep Learning (DL) has shown remarkable success in detecting XSS attacks, it remains vulnerable to adversarial attacks due to the discontinuous nature of the mapping between the input (i.e., the attack) and the output (i.e., the prediction of the model whether an input is classified as XSS or benign). These adversarial attacks employ mutation-based strategies for different components of XSS attack vectors, allowing adversarial agents to iteratively select mutations to evade detection. Our work replicates a state-of-the-art XSS adversarial attack, highlighting threats to validity in the reference work and extending it towards a more effective evaluation strategy. Moreover, we introduce an XSS Oracle to mitigate these threats. The experimental results show that our approach achieves an escape rate above 96% when the threats to validity of the replicated technique are addressed.

**arXiv ID:** 2502.19095
</details>

<details>
<summary><strong>World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation</strong> - Zhennan Jiang, Kai Liu, Yuxin Qin, Shuai Tian, Yupeng Zheng, Mingcai Zhou, Chao Yu, Haoran Li, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2509.19080)</summary>

**Abstract:** Robotic manipulation policies are commonly initialized through imitation learning, but their performance is limited by the scarcity and narrow coverage of expert data. Reinforcement learning can refine polices to alleviate this limitation, yet real-robot training is costly and unsafe, while training in simulators suffers from the sim-to-real gap. Recent advances in generative models have demonstrated remarkable capabilities in real-world simulation, with diffusion models in particular excelling at generation. This raises the question of how diffusion model-based world models can be combined to enhance pre-trained policies in robotic manipulation. In this work, we propose World4RL, a framework that employs diffusion-based world models as high-fidelity simulators to refine pre-trained policies entirely in imagined environments for robotic manipulation. Unlike prior works that primarily employ world models for planning, our framework enables direct end-to-end policy optimization. World4RL is designed around two principles: pre-training a diffusion world model that captures diverse dynamics on multi-task datasets and refining policies entirely within a frozen world model to avoid online real-world interactions. We further design a two-hot action encoding scheme tailored for robotic manipulation and adopt diffusion backbones to improve modeling fidelity. Extensive simulation and real-world experiments demonstrate that World4RL provides high-fidelity environment modeling and enables consistent policy refinement, yielding significantly higher success rates compared to imitation learning and other baselines.

**arXiv ID:** 2509.19080
</details>

<details>
<summary><strong>StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors</strong> - Suraj Ranganath, Atharv Ramesh - [[pdf]](https://arxiv.org/pdf/2602.08934)</summary>

**Abstract:** AI-text detectors face a critical robustness challenge: adversarial paraphrasing attacks that preserve semantics while evading detection. We introduce StealthRL, a reinforcement learning framework that stress-tests detector robustness under realistic adversarial conditions. StealthRL trains a paraphrase policy against a multi-detector ensemble using Group Relative Policy Optimization (GRPO) with LoRA adapters on Qwen3-4B, optimizing a composite reward that balances detector evasion with semantic preservation. We evaluate six attack settings (M0-M5) on the full filtered MAGE test pool (15,310 human / 14,656 AI) against four detectors: RoBERTa, Fast-DetectGPT, Binoculars, and MAGE. StealthRL achieves near-zero detection on three of the four detectors and a 0.024 mean TPR@1%FPR, reducing mean AUROC from 0.79 to 0.43 and attaining a 97.6% attack success rate. Critically, attacks transfer to two held-out detectors not seen during training, revealing shared architectural vulnerabilities rather than detector-specific brittleness. We additionally conduct LLM-based quality evaluation via Likert scoring on 500 matched samples per method, analyze detector score distributions to explain why evasion succeeds, and provide per-detector AUROC with bootstrap confidence intervals. Our results expose significant robustness gaps in current AI-text detection and establish StealthRL as a principled adversarial evaluation protocol. Code and evaluation pipeline are publicly available at this https URL.

**arXiv ID:** 2602.08934
</details>

<details>
<summary><strong>SAGE: Sustainable Agent-Guided Expert-tuning for Culturally Attuned Translation in Low-Resource Southeast Asia</strong> - Zhixiang Lu, Chong Zhang, Yulong Li, Angelos Stefanidis, Anh Nguyen, Imran Razzak, Jionglong Su, Zhengyong Jiang - [[pdf]](https://arxiv.org/pdf/2603.19931)</summary>

**Abstract:** The vision of an inclusive World Wide Web is impeded by a severe linguistic divide, particularly for communities in low-resource regions of Southeast Asia. While large language models (LLMs) offer a potential solution for translation, their deployment in data-poor contexts faces a dual challenge: the scarcity of high-quality, culturally relevant data and the prohibitive energy costs of training on massive, noisy web corpora. To resolve the tension between digital inclusion and environmental sustainability, we introduce Sustainable Agent-Guided Expert-tuning (SAGE). This framework pioneers an energy-aware paradigm that prioritizes the "right data" over "big data". Instead of carbon-intensive training on unfiltered datasets, SAGE employs a reinforcement learning (RL) agent, optimized via Group Relative Policy Optimization (GRPO), to autonomously curate a compact training set. The agent utilizes a semantic reward signal derived from a small, expert-constructed set of community dialogues to filter out noise and cultural misalignment. We then efficiently fine-tune open-source LLMs on this curated data using Low-Rank Adaptation (LoRA). We applied SAGE to translation tasks between English and seven low-resource languages (LRLs) in Southeast Asia. Our approach establishes new state-of-the-art performance on BLEU-4 and COMET-22 metrics, effectively capturing local linguistic nuances. Crucially, SAGE surpasses baselines trained on full datasets while reducing data usage by 97.1% and training energy consumption by 95.2%. By delivering high-performance models with a minimal environmental footprint, SAGE offers a scalable and responsible pathway to bridge the digital divide in the Global South.

**arXiv ID:** 2603.19931
</details>

<details>
<summary><strong>Identifying and Mitigating Bottlenecks in Role-Playing Agents: A Systematic Study of Disentangling Character Profile Axes</strong> - Yonghyun Jun, Junhyuk Choi, Jihyeong Park, Jeonghyun Park, Liu Nicole Geumheon, Hwanhee Lee - [[pdf]](https://arxiv.org/pdf/2601.04716)</summary>

**Abstract:** Advancements in Large Language Model (LLM) Role-Playing Agents have focused on various construction methodologies, yet it remains unclear which aspects of character profiles genuinely drive role-playing quality. To bridge this gap, we introduce a systematic diagnostic framework that disentangles the impact of character profiles along three axes: Familiarity (Known vs. Unknown), Structure (Structured vs. Unstructured), and Disposition (Moral vs. Immoral). To investigate these axes, we design a unified hierarchical schema (5 dimensions, 28 fields) standardizing character attributes and construct a controlled dataset of 211 personas varying along these three axes. We evaluate five LLMs on single and multi-turn benchmarks. Our results reveal a striking asymmetry: Familiarity and Structure show negligible impact, while Valence produces large, consistent performance degradation for immoral characters across all conditions. This performance drop concentrates in motivation-related attributes, indicating that alignment priors actively suppress tokens needed for faithful immoral portrayal. To mitigate this alignment-induced bottleneck, we propose Field-Aware Contrastive Decoding (FACD), a training-free strategy that selectively amplizes suppressed immoral-field signals, significantly reducing the Moral-Immoral performance gap without sacrificing moral-character performance.

**arXiv ID:** 2601.04716
</details>

<details>
<summary><strong>Balancing the Reasoning Load: Difficulty-Differentiated Policy Optimization with Length Redistribution for Efficient and Robust Reinforcement Learning</strong> - Yinan Xia, Haotian Zhang, Huiming Wang - [[pdf]](https://arxiv.org/pdf/2603.18533)</summary>

**Abstract:** Large Reasoning Models (LRMs) have shown exceptional reasoning capabilities, but they also suffer from the issue of overthinking, often generating excessively long and redundant answers.
For problems that exceed the model's capabilities, LRMs tend to exhibit the overconfidence phenomenon, generating overly short but incorrect answers, which may contribute to suboptimal performance.
To address these issues, we propose Difficulty-Differentiated Policy Optimization (DDPO), an efficient reinforcement learning algorithm that optimizes simple and complex tasks separately based on the overconfidence phenomenon.
Specifically, it reduces the output length for simple tasks without compromising accuracy, while for complex tasks, it expands the exploration space to improve performance. We further derive the theoretical conditions for maximizing expected accuracy, which require the length distribution to closely approximate the optimal length and be as concentrated as possible. Based on these conditions, we propose using the difficulty-level average as a well-founded reference for length optimization.
Extensive experiments on both in-domain and out-of-domain benchmarks validate the superiority and effectiveness of DDPO. Compared to GRPO, DDPO reduces the average answer length by 12% while improving accuracy by 1.85% across multiple benchmarks, achieving a better trade-off between accuracy and length. The code is available at this https URL.

**arXiv ID:** 2603.18533
</details>

<details>
<summary><strong>Optimizing Resource-Constrained Non-Pharmaceutical Interventions for Multi-Cluster Outbreak Control Using Hierarchical Reinforcement Learning</strong> - Xueqiao Peng, Andrew Perrault - [[pdf]](https://arxiv.org/pdf/2603.19397)</summary>

**Abstract:** Non-pharmaceutical interventions (NPIs), such as diagnostic testing and quarantine, are crucial for controlling infectious disease outbreaks but are often constrained by limited resources, particularly in early outbreak stages. In real-world public health settings, resources must be allocated across multiple outbreak clusters that emerge asynchronously, vary in size and risk, and compete for a shared resource budget. Here, a cluster corresponds to a group of close contacts generated by a single infected index case. Thus, decisions must be made under uncertainty and heterogeneous demands, while respecting operational constraints. We formulate this problem as a constrained restless multi-armed bandit and propose a hierarchical reinforcement learning framework. A global controller learns a continuous action cost multiplier that adjusts global resource demand, while a generalized local policy estimates the marginal value of allocating resources to individuals within each cluster. We evaluate the proposed framework in a realistic agent-based simulator of SARS-CoV-2 with dynamically arriving clusters. Across a wide range of system scales and testing budgets, our method consistently outperforms RMAB-inspired and heuristic baselines, improving outbreak control effectiveness by 20%-30%. Experiments on up to 40 concurrently active clusters further demonstrate that the hierarchical framework is highly scalable and enables faster decision-making than the RMAB-inspired method.

**arXiv ID:** 2603.19397
</details>

<details>
<summary><strong>Revisiting Gene Ontology Knowledge Discovery with Hierarchical Feature Selection and Virtual Study Group of AI Agents</strong> - Cen Wan, Alex A. Freitas - [[pdf]](https://arxiv.org/pdf/2603.20132)</summary>

**Abstract:** Large language models have achieved great success in multiple challenging tasks, and their capacity can be further boosted by the emerging agentic AI techniques. This new computing paradigm has already started revolutionising the traditional scientific discovery pipelines. In this work, we propose a novel agentic AI-based knowledge discovery-oriented virtual study group that aims to extract meaningful ageing-related biological knowledge considering highly ageing-related Gene Ontology terms that are selected by hierarchical feature selection methods. We investigate the performance of the proposed agentic AI framework by considering four different model organisms' ageing-related Gene Ontology terms and validate the biological findings by reviewing existing research articles. It is found that the majority of the AI agent-generated scientific claims can be supported by existing literatures and the proposed internal mechanisms of the virtual study group also play an important role in the designed agentic AI-based knowledge discovery framework.

**arXiv ID:** 2603.20132
</details>

<details>
<summary><strong>Gym-TORAX: Open-source software for integrating reinforcement learning with plasma control simulators in tokamak research</strong> - Antoine Mouchamps, Arthur Malherbe, Adrien Bolland, Damien Ernst - [[pdf]](https://arxiv.org/pdf/2510.11283)</summary>

**Abstract:** This paper presents Gym-TORAX, a Python package enabling the implementation of Reinforcement Learning (RL) environments for simulating plasma dynamics and control in tokamaks. Users define succinctly a set of control actions and observations, and a control objective from which Gym-TORAX creates a Gymnasium environment that wraps TORAX for simulating the plasma dynamics. The objective is formulated through rewards depending on the simulated state of the plasma and control action to optimize specific characteristics of the plasma, such as performance and stability. The resulting environment instance is then compatible with a wide range of RL algorithms and libraries and will facilitate RL research in plasma control. In its current version, one environment is readily available, based on a ramp-up scenario of the International Thermonuclear Experimental Reactor (ITER).

**arXiv ID:** 2510.11283
</details>

<details>
<summary><strong>Efficient Cross-Domain Offline Reinforcement Learning with Dynamics- and Value-Aligned Data Filtering</strong> - Zhongjian Qiao, Rui Yang, Jiafei Lyu, Chenjia Bai, Xiu Li, Siyang Gao, Shuang Qiu - [[pdf]](https://arxiv.org/pdf/2512.02435)</summary>

**Abstract:** Cross-domain offline reinforcement learning (RL) aims to train a well-performing agent in the target environment, leveraging both a limited target domain dataset and a source domain dataset with (possibly) sufficient data coverage. Due to the underlying dynamics misalignment between source and target domains, naively merging the two datasets may incur inferior performance. Recent advances address this issue by selectively leveraging source domain samples whose dynamics align well with the target domain. However, our work demonstrates that dynamics alignment alone is insufficient, by examining the limitations of prior frameworks and deriving a new target domain sub-optimality bound for the policy learned on the source domain. More importantly, our theory underscores an additional need for \textit{value alignment}, i.e., selecting high-quality, high-value samples from the source domain, a critical dimension overlooked by existing works. Motivated by such theoretical insight, we propose \textbf{\underline{D}}ynamics- and \textbf{\underline{V}}alue-aligned \textbf{\underline{D}}ata \textbf{\underline{F}}iltering (DVDF) method, a novel unified cross-domain RL framework that selectively incorporates source domain samples exhibiting strong alignment in \textit{both dynamics and values}. We empirically study a range of dynamics shift scenarios, including kinematic and morphology shifts, and evaluate DVDF on various tasks and datasets, even in the challenging setting where the target domain dataset contains an extremely limited amount of data. Extensive experiments demonstrate that DVDF consistently outperforms strong baselines with significant improvements.

**arXiv ID:** 2512.02435
</details>

<details>
<summary><strong>AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for Vision-Language-Action Models</strong> - Chengxuan Lu, Shukuan Wang, Yanjie Li, Wei Liu, Shiji Jin, Fuyuan Qian, Peiming Li, Baigui Sun, Yang Liu - [[pdf]](https://arxiv.org/pdf/2603.18464)</summary>

**Abstract:** Reinforcement learning (RL) for large-scale Vision-Language-Action (VLA) models faces significant challenges in computational efficiency and data acquisition. We propose AcceRL, a fully asynchronous and decoupled RL framework designed to eliminate synchronization barriers by physically isolating training, inference, and rollouts. Crucially, AcceRL is the first to integrate a plug-and-play, trainable world model into a distributed asynchronous RL pipeline to generate virtual experiences. Experiments on the LIBERO~\cite{liu2023libero} benchmark demonstrate that AcceRL achieves state-of-the-art (SOTA) performance. Systematically, it exhibits super-linear scaling in throughput and highly efficient hardware utilization. Algorithmically, the world-model-augmented variant delivers unprecedented sample efficiency and robust training stability in complex control tasks. Code is publicly available at this https URL.

**arXiv ID:** 2603.18464
</details>

<details>
<summary><strong>ContractionPPO: Certified Reinforcement Learning via Differentiable Contraction Layers</strong> - Vrushabh Zinage, Narek Harutyunyan, Eric Verheyden, Fred Y. Hadaegh, Soon-Jo Chung - [[pdf]](https://arxiv.org/pdf/2603.19632)</summary>

**Abstract:** Legged locomotion in unstructured environments demands not only high-performance control policies but also formal guarantees to ensure robustness under perturbations. Control methods often require carefully designed reference trajectories, which are challenging to construct in high-dimensional, contact-rich systems such as quadruped robots. In contrast, Reinforcement Learning (RL) directly learns policies that implicitly generate motion, and uniquely benefits from access to privileged information, such as full state and dynamics during training, that is not available at deployment. We present ContractionPPO, a framework for certified robust planning and control of legged robots by augmenting Proximal Policy Optimization (PPO) RL with a state-dependent contraction metric layer. This approach enables the policy to maximize performance while simultaneously producing a contraction metric that certifies incremental exponential stability of the simulated closed-loop system. The metric is parameterized as a Lipschitz neural network and trained jointly with the policy, either in parallel or as an auxiliary head of the PPO backbone. While the contraction metric is not deployed during real-world execution, we derive upper bounds on the worst-case contraction rate and show that these bounds ensure the learned contraction metric generalizes from simulation to real-world deployment. Our hardware experiments on quadruped locomotion demonstrate that ContractionPPO enables robust, certifiably stable control even under strong external perturbations.

**arXiv ID:** 2603.19632
</details>

<details>
<summary><strong>Legged Autonomous Surface Science In Analogue Environments (LASSIE): Making Every Robotic Step Count in Planetary Exploration</strong> - Cristina G. Wilson, Marion Nachon, Shipeng Liu, John G. Ruck, J. Diego Caporale, Benjamin E. McKeeby, Yifeng Zhang, Jordan M. Bretzfelder, John Bush, Alivia M. Eng, Ethan Fulcher, Emmy B. Hughes, Ian C. Rankin, Jelis J. Sostre Cortés, Sophie Silver, Michael R. Zanetti, Ryan C. Ewing, Kenton R. Fisher, Douglas J. Jerolmack, Daniel E. Koditschek, Frances Rivera-Hernández, Thomas F. Shipley, Feifei Qian - [[pdf]](https://arxiv.org/pdf/2603.19661)</summary>

**Abstract:** The ability to efficiently and effectively explore planetary surfaces is currently limited by the capability of wheeled rovers to traverse challenging terrains, and by pre-programmed data acquisition plans with limited in-situ flexibility. In this paper, we present two novel approaches to address these limitations: (i) high-mobility legged robots that use direct surface interactions to collect rich information about the terrain's mechanics to guide exploration; (ii) human-inspired data acquisition algorithms that enable robots to reason about scientific hypotheses and adapt exploration priorities based on incoming ground-sensing measurements. We successfully verify our approach through lab work and field deployments in two planetary analog environments. The new capability for legged robots to measure soil mechanical properties is shown to enable effective traversal of challenging terrains. When coupled with other geologic properties (e.g., composition, thermal properties, and grain size data etc), soil mechanical measurements reveal key factors governing the formation and development of geologic environments. We then demonstrate how human-inspired algorithms turn terrain-sensing robots into teammates, by supporting more flexible and adaptive data collection decisions with human scientists. Our approach therefore enables exploration of a wider range of planetary environments and new substrate investigation opportunities through integrated human-robot systems that support maximum scientific return.

**arXiv ID:** 2603.19661
</details>

<details>
<summary><strong>CoInfra: A Large-Scale Cooperative Infrastructure Perception System and Dataset for Vehicle-Infrastructure Cooperation in Adverse Weather</strong> - Minghao Ning, Yufeng Yang, Keqi Shu, Shucheng Huang, Jiaming Zhong, Maryam Salehi, Mahdi Rahmani, Jiaming Guo, Yukun Lu, Chen Sun, Aladdin Saleh, Ehsan Hashemi, Amir Khajepour - [[pdf]](https://arxiv.org/pdf/2507.02245)</summary>

**Abstract:** Vehicle-infrastructure (V2I) cooperative perception can substantially extend the range, coverage, and robustness of autonomous driving systems beyond the limits of onboard-only sensing, particularly in occluded and adverse-weather environments. However, its practical value is still difficult to quantify because existing benchmarks do not adequately capture large-scale multi-node deployments, realistic communication conditions, and adverse-weather operation. This paper presents CoInfra, a deployable cooperative infrastructure perception platform comprising 14 roadside sensor nodes connected through a commercial 5G network, together with a large-scale dataset and an open-source system stack for V2I cooperation research. The system supports synchronized multi-node sensing and delay-aware fusion under real 5G communication constraints. The released dataset covers an eight-node urban roundabout under four weather conditions (sunny, rainy, heavy snow, and freezing rain) and contains 294k LiDAR frames, 589k camera images, and 332k globally consistent 3D bounding boxes. It also includes a synchronized V2I subset collected with an autonomous vehicle. Beyond standard perception benchmarks, we further evaluate whether infrastructure sensing improves awareness of safety-critical traffic participants during roundabout interactions. In structured conflict scenarios, V2I cooperation increases critical-frame completeness from 33%-46% with vehicle-only sensing to 86%-100%. These results show that multi-node infrastructure perception can significantly improve situational awareness in conflict-rich traffic scenarios where vehicle-only sensing is most limited.

**arXiv ID:** 2507.02245
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
