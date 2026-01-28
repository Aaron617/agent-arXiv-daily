# Agent arXiv Daily

**Last Updated:** 2026-01-28 03:12:00

**Total Papers:** 76

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (8 papers)</h2></summary>

<details>
<summary><strong>Agentic Design Patterns: A System-Theoretic Framework</strong> - Minh-Dung Dao, Quy Minh Le, Hoang Thanh Lam, Duc-Trong Le, Quoc-Viet Pham, Barry O'Sullivan, Hoang D. Nguyen - [[pdf]](https://arxiv.org/pdf/2601.19752)</summary>

**Abstract:** With the development of foundation model (FM), agentic AI systems are getting more attention, yet their inherent issues like hallucination and poor reasoning, coupled with the frequent ad-hoc nature of system design, lead to unreliable and brittle applications. Existing efforts to characterise agentic design patterns often lack a rigorous systems-theoretic foundation, resulting in high-level or convenience-based taxonomies that are difficult to implement. This paper addresses this gap by introducing a principled methodology for engineering robust AI agents. We propose two primary contributions: first, a novel system-theoretic framework that deconstructs an agentic AI system into five core, interacting functional subsystems: Reasoning & World Model, Perception & Grounding, Action Execution, Learning & Adaptation, and Inter-Agent Communication. Second, derived from this architecture and directly mapped to a comprehensive taxonomy of agentic challenges, we present a collection of 12 agentic design patterns. These patterns - categorised as Foundational, Cognitive & Decisional, Execution & Interaction, and Adaptive & Learning - offer reusable, structural solutions to recurring problems in agent design. The utility of the framework is demonstrated by a case study on the ReAct framework, showing how the proposed patterns can rectify systemic architectural deficiencies. This work provides a foundational language and a structured methodology to standardise agentic design communication among researchers and engineers, leading to more modular, understandable, and reliable autonomous systems.

**arXiv ID:** 2601.19752
</details>

<details>
<summary><strong>AgenticSCR: An Autonomous Agentic Secure Code Review for Immature Vulnerabilities Detection</strong> - Wachiraphan Charoenwet, Kla Tantithamthavorn, Patanamon Thongtanunam, Hong Yi Lin, Minwoo Jeong, Ming Wu - [[pdf]](https://arxiv.org/pdf/2601.19138)</summary>

**Abstract:** Secure code review is critical at the pre-commit stage, where vulnerabilities must be caught early under tight latency and limited-context constraints. Existing SAST-based checks are noisy and often miss immature, context-dependent vulnerabilities, while standalone Large Language Models (LLMs) are constrained by context windows and lack explicit tool use. Agentic AI, which combine LLMs with autonomous decision-making, tool invocation, and code navigation, offer a promising alternative, but their effectiveness for pre-commit secure code review is not yet well understood. In this work, we introduce AgenticSCR, an agentic AI for secure code review for detecting immature vulnerabilities during the pre-commit stage, augmented by security-focused semantic memories. Using our own curated benchmark of immature vulnerabilities, tailored to the pre-commit secure code review, we empirically evaluate how accurate is our AgenticSCR for localizing, detecting, and explaining immature vulnerabilities. Our results show that AgenticSCR achieves at least 153% relatively higher percentage of correct code review comments than the static LLM-based baseline, and also substantially surpasses SAST tools. Moreover, AgenticSCR generates more correct comments in four out of five vulnerability types, consistently and significantly outperforming all other baselines. These findings highlight the importance of Agentic Secure Code Review, paving the way towards an emerging research area of immature vulnerability detection.

**arXiv ID:** 2601.19138
</details>

<details>
<summary><strong>Jenius Agent: Towards Experience-Driven Accuracy Optimization in Real-World Scenarios</strong> - Defei Xia, Bingfeng Pi, Shenbin Zhang, Song Hua, Yunfei Wei, Lei Zuo - [[pdf]](https://arxiv.org/pdf/2601.01857)</summary>

**Abstract:** As agent systems powered by large language models (LLMs) advance, improving performance in context understanding, tool usage, and long-horizon execution has become critical. However, existing agent frameworks and benchmarks provide limited visibility into execution-level behavior, making failures in tool invocation, state tracking, and context management difficult to diagnose. This paper presents Jenius-Agent, a system-level agent framework grounded in real-world deployment experience. It integrates adaptive prompt generation, context-aware tool orchestration, and layered memory mechanism to stabilize execution and improve robustness in long-horizon, tool-augmented tasks. Beyond system design, we introduce an evaluation methodology that jointly measures procedural fidelity, semantic correctness, and efficiency. This framework makes agent behavior observable as a structured execution process and enables systematic analysis of failure modes not captured by output-only metrics. Experiments on Jenius-bench show substantial improvements in task completion rate, with up to a 35 percent relative gain over the base agent, along with reduced token consumption, response latency, and tool invocation failures. The framework is already deployed in Jenius ({this https URL}), providing a lightweight and scalable solution for robust, protocol-compatible autonomous agents.

**arXiv ID:** 2601.01857
</details>

<details>
<summary><strong>Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering</strong> - Xinyu Zhu, Yuzhu Cai, Zexi Liu, Bingyang Zheng, Cheng Wang, Rui Ye, Jiaao Chen, Hanrui Wang, Wei-Chen Wang, Yuzhi Zhang, Linfeng Zhang, Weinan E, Di Jin, Siheng Chen, Yanfeng Wang - [[pdf]](https://arxiv.org/pdf/2601.10402)</summary>

**Abstract:** The advancement of artificial intelligence toward agentic science is currently bottlenecked by the challenge of ultra-long-horizon autonomy, the ability to sustain strategic coherence and iterative correction over experimental cycles spanning days or weeks. While Large Language Models (LLMs) have demonstrated prowess in short-horizon reasoning, they are easily overwhelmed by execution details in the high-dimensional, delayed-feedback environments of real-world research, failing to consolidate sparse feedback into coherent long-term guidance. Here, we present ML-Master 2.0, an autonomous agent that masters ultra-long-horizon machine learning engineering (MLE) which is a representative microcosm of scientific discovery. By reframing context management as a process of cognitive accumulation, our approach introduces Hierarchical Cognitive Caching (HCC), a multi-tiered architecture inspired by computer systems that enables the structural differentiation of experience over time. By dynamically distilling transient execution traces into stable knowledge and cross-task wisdom, HCC allows agents to decouple immediate execution from long-term experimental strategy, effectively overcoming the scaling limits of static context windows. In evaluations on OpenAI's MLE-Bench under 24-hour budgets, ML-Master 2.0 achieves a state-of-the-art medal rate of 56.44%. Our findings demonstrate that ultra-long-horizon autonomy provides a scalable blueprint for AI capable of autonomous exploration beyond human-precedent complexities.

**arXiv ID:** 2601.10402
</details>

<details>
<summary><strong>Designing and Evaluating a Conversational Agent for Early Diagnosis of Alzheimer's Disease and Related Dementias</strong> - Andrew G. Breithaupt, Nayoung Choi, James D. Finch, Jeanne M. Powell, Arin L. Nelson, Oz A. Alon, Howard J. Rosen, Jinho D. Choi - [[pdf]](https://arxiv.org/pdf/2509.11478)</summary>

**Abstract:** Early diagnosis of Alzheimer's disease and related dementias (ADRD) is critical for timely intervention, yet most diagnoses are delayed until advanced stages. While comprehensive patient narratives are essential for accurate diagnosis, prior work has largely focused on screening studies that classify cognitive status from interactions rather than supporting the diagnostic process. We designed voice-interactive conversational agents, leveraging large language models (LLMs), to elicit narratives relevant to ADRD from patients and informants. We evaluated the agent with 30 adults with suspected ADRD through conversation analysis, user surveys, and analysis of symptom elicitation compared to blinded specialist interviews. Symptoms detected by the agent showed promising agreement with those identified by specialists. Users appreciated the agent's patience and systematic questioning, which supported engagement and expression of complex, hard-to-describe experiences. While these findings suggest potential for conversational agents as structured diagnostic support tools, further validation with larger samples and assessment of clinical utility is needed before deployment.

**arXiv ID:** 2509.11478
</details>

<details>
<summary><strong>MobileSafetyBench: Evaluating Safety of Autonomous Agents in Mobile Device Control</strong> - Juyong Lee, Dongyoon Hahm, June Suk Choi, W. Bradley Knox, Kimin Lee - [[pdf]](https://arxiv.org/pdf/2410.17520)</summary>

**Abstract:** Autonomous agents powered by large language models (LLMs) show promising potential in assistive tasks across various domains, including mobile device control. As these agents interact directly with personal information and device settings, ensuring their safe and reliable behavior is crucial to prevent undesirable outcomes. However, no benchmark exists for standardized evaluation of the safety of mobile device-control agents. In this work, we introduce MobileSafetyBench, a benchmark designed to evaluate the safety of device-control agents within a realistic mobile environment based on Android emulators. We develop a diverse set of tasks involving interactions with various mobile applications, including messaging and banking applications, challenging agents with managing risks encompassing misuse and negative side effects. These tasks include tests to evaluate the safety of agents in daily scenarios as well as their robustness against indirect prompt injection attacks. Our experiments demonstrate that baseline agents, based on state-of-the-art LLMs, often fail to effectively prevent harm while performing the tasks. To mitigate these safety concerns, we propose a prompting method that encourages agents to prioritize safety considerations. While this method shows promise in promoting safer behaviors, there is still considerable room for improvement to fully earn user trust. This highlights the urgent need for continued research to develop more robust safety mechanisms in mobile environments.

**arXiv ID:** 2410.17520
</details>

<details>
<summary><strong>Geometric Dynamics of Agentic Loops in Large Language Models</strong> - Nicolas Tacheny - [[pdf]](https://arxiv.org/pdf/2512.10350)</summary>

**Abstract:** Iterative LLM systems(self-refinement, chain-of-thought, autonomous agents) are increasingly deployed, yet their temporal dynamics remain uncharacterized. Prior work evaluates task performance at convergence but ignores the trajectory: how does semantic content evolve across iterations? Does it stabilize, drift, or oscillate? Without answering these questions, we cannot predict system behavior, guarantee stability, or systematically design iterative architectures.
We formalize agentic loops as discrete dynamical systems in semantic space. Borrowing from dynamical systems theory, we define trajectories, attractors and dynamical regimes for recursive LLM transformations, providing rigorous geometric definitions adapted to this setting. Our framework reveals that agentic loops exhibit classifiable dynamics: contractive (convergence toward stable semantic attractors), oscillatory (cycling among attractors), or exploratory (unbounded divergence).
Experiments on singular loops validate the framework. Iterative paraphrasing produces contractive dynamics with measurable attractor formation and decreasing dispersion. Iterative negation produces exploratory dynamics with no stable structure. Crucially, prompt design directly controls the dynamical regime - the same model exhibits fundamentally different geometric behaviors depending solely on the transformation applied.
This work establishes that iterative LLM dynamics are predictable and controllable, opening new directions for stability analysis, trajectory forecasting, and principled design of composite loops that balance convergence and exploration.

**arXiv ID:** 2512.10350
</details>

<details>
<summary><strong>Voice-Based Chatbots for English Speaking Practice in Multilingual Low-Resource Indian Schools: A Multi-Stakeholder Study</strong> - Sneha Shashidhara, Vivienne Bihe Chi, Abhay P Singh, Lyle Ungar, Sharath Chandra Guntuku - [[pdf]](https://arxiv.org/pdf/2601.19304)</summary>

**Abstract:** Spoken English proficiency is a powerful driver of economic mobility for low-income Indian youth, yet opportunities for spoken practice remain scarce in schools. We investigate the deployment of a voice-based chatbot for English conversation practice across four low-resource schools in Delhi. Through a six-day field study combining observations and interviews, we captured the perspectives of students, teachers, and principals. Findings confirm high demand across all groups, with notable gains in student speaking confidence. Our multi-stakeholder analysis surfaced a tension in long-term adoption vision: students favored open-ended conversational practice, while administrators emphasized curriculum-aligned assessment. We offer design recommendations for voice-enabled chatbots in low-resource multilingual contexts, highlighting the need for more intelligible speech output for non-native learners, one-tap interactions with simplified interfaces, and actionable analytics for educators. Beyond language learning, our findings inform the co-design of future AI-based educational technologies that are socially sustainable within the complex ecosystem of low-resource schools.

**arXiv ID:** 2601.19304
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>MAGNET: Towards Adaptive GUI Agents with Memory-Driven Knowledge Evolution</strong> - Libo Sun, Jiwen Zhang, Siyuan Wang, Zhongyu Wei - [[pdf]](https://arxiv.org/pdf/2601.19199)</summary>

**Abstract:** Mobile GUI agents powered by large foundation models enable autonomous task execution, but frequent updates altering UI appearance and reorganizing workflows cause agents trained on historical data to fail. Despite surface changes, functional semantics and task intents remain fundamentally stable. Building on this insight, we introduce MAGNET, a memory-driven adaptive agent framework with dual-level memory: stationary memory linking diverse visual features to stable functional semantics for robust action grounding and procedural memory capturing stable task intents across varying workflows. We propose a dynamic memory evolution mechanism that continuously refines both memories by prioritizing frequently accessed knowledge. Online benchmark AndroidWorld evaluations show substantial improvements over baselines, while offline benchmarks confirm consistent gains under distribution shifts. These results validate that leveraging stable structures across interface changes improves agent performance and generalization in evolving software environments.

**arXiv ID:** 2601.19199
</details>

<details>
<summary><strong>Automated structural testing of LLM-based agents: methods, framework, and case studies</strong> - Jens Kohl, Otto Kruse, Youssef Mostafa, Andre Luckow, Karsten Schroer, Thomas Riedl, Ryan French, David Katz, Manuel P. Luitz, Tanrajbir Takher, Ken E. Friedl, Céline Laurent-Winter - [[pdf]](https://arxiv.org/pdf/2601.18827)</summary>

**Abstract:** LLM-based agents are rapidly being adopted across diverse domains. Since they interact with users without supervision, they must be tested extensively. Current testing approaches focus on acceptance-level evaluation from the user's perspective. While intuitive, these tests require manual evaluation, are difficult to automate, do not facilitate root cause analysis, and incur expensive test environments. In this paper, we present methods to enable structural testing of LLM-based agents. Our approach utilizes traces (based on OpenTelemetry) to capture agent trajectories, employs mocking to enforce reproducible LLM behavior, and adds assertions to automate test verification. This enables testing agent components and interactions at a deeper technical level within automated workflows. We demonstrate how structural testing enables the adaptation of software engineering best practices to agents, including the test automation pyramid, regression testing, test-driven development, and multi-language testing. In representative case studies, we demonstrate automated execution and faster root-cause analysis. Collectively, these methods reduce testing costs and improve agent quality through higher coverage, reusability, and earlier defect detection. We provide an open source reference implementation on GitHub.

**arXiv ID:** 2601.18827
</details>

<details>
<summary><strong>GUIGuard: Toward a General Framework for Privacy-Preserving GUI Agents</strong> - Yanxi Wang, Zhiling Zhang, Wenbo Zhou, Weiming Zhang, Jie Zhang, Qiannan Zhu, Yu Shi, Shuxin Zheng, Jiyan He - [[pdf]](https://arxiv.org/pdf/2601.18842)</summary>

**Abstract:** GUI agents enable end-to-end automation through direct perception of and interaction with on-screen interfaces. However, these agents frequently access interfaces containing sensitive personal information, and screenshots are often transmitted to remote models, creating substantial privacy risks. These risks are particularly severe in GUI workflows: GUIs expose richer, more accessible private information, and privacy risks depend on interaction trajectories across sequential scenes. We propose GUIGuard, a three-stage framework for privacy-preserving GUI agents: (1) privacy recognition, (2) privacy protection, and (3) task execution under protection. We further construct GUIGuard-Bench, a cross-platform benchmark with 630 trajectories and 13,830 screenshots, annotated with region-level privacy grounding and fine-grained labels of risk level, privacy category, and task necessity. Evaluations reveal that existing agents exhibit limited privacy recognition, with state-of-the-art models achieving only 13.3% accuracy on Android and 1.4% on PC. Under privacy protection, task-planning semantics can still be maintained, with closed-source models showing stronger semantic consistency than open-source ones. Case studies on MobileWorld show that carefully designed protection strategies achieve higher task accuracy while preserving privacy. Our results highlight privacy recognition as a critical bottleneck for practical GUI agents. Project: this https URL

**arXiv ID:** 2601.18842
</details>

<details>
<summary><strong>RobustExplain: Evaluating Robustness of LLM-Based Explanation Agents for Recommendation</strong> - Guilin Zhang, Kai Zhao, Jeffrey Friedman, Xu Chu - [[pdf]](https://arxiv.org/pdf/2601.19120)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly used to generate natural-language explanations in recommender systems, acting as explanation agents that reason over user behavior histories. While prior work has focused on explanation fluency and relevance under fixed inputs, the robustness of LLM-generated explanations to realistic user behavior noise remains largely unexplored. In real-world web platforms, interaction histories are inherently noisy due to accidental clicks, temporal inconsistencies, missing values, and evolving preferences, raising concerns about explanation stability and user trust. We present RobustExplain, the first systematic evaluation framework for measuring the robustness of LLM-generated recommendation explanations. RobustExplain introduces five realistic user behavior perturbations evaluated across multiple severity levels and a multi-dimensional robustness metric capturing semantic, keyword, structural, and length consistency. Our goal is to establish a principled, task-level evaluation framework and initial robustness baselines, rather than to provide a comprehensive leaderboard across all available LLMs. Experiments on four representative LLMs (7B--70B) show that current models exhibit only moderate robustness, with larger models achieving up to 8% higher stability. Our results establish the first robustness benchmarks for explanation agents and highlight robustness as a critical dimension for trustworthy, agent-driven recommender systems at web scale.

**arXiv ID:** 2601.19120
</details>

<details>
<summary><strong>MetaVLA: Unified Meta Co-training For Efficient Embodied Adaption</strong> - Chen Li, Zhantao Yang, Han Zhang, Fangyi Chen, Chenchen Zhu, Anudeepsekhar Bolimera, Marios Savvides - [[pdf]](https://arxiv.org/pdf/2510.05580)</summary>

**Abstract:** Vision-Language-Action (VLA) models show promise in embodied reasoning, yet remain far from true generalists-they often require task-specific fine-tuning, incur high compute costs, and generalize poorly to unseen tasks. We propose MetaVLA, a unified, backbone-agnostic post-training framework for efficient and scalable alignment. MetaVLA introduces Context-Aware Meta Co-Training, which consolidates diverse target tasks into a single fine-tuning stage while leveraging structurally diverse auxiliary tasks to improve in-domain generalization. Unlike naive multi-task SFT, MetaVLA integrates a lightweight meta-learning mechanism-derived from Attentive Neural Processes-to enable rapid adaptation from diverse contexts with minimal architectural change or inference overhead. On the LIBERO benchmark, MetaVLA with six auxiliary tasks outperforms OpenVLA by up to 8.0% on long-horizon tasks, reduces training steps from 240K to 75K, and cuts GPU time by ~76%. These results show that scalable, low-resource post-training is achievable-paving the way toward general-purpose embodied agents. Code will be available.

**arXiv ID:** 2510.05580
</details>

<details>
<summary><strong>Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents</strong> - Yuhan Guo, Cong Guo, Aiwen Sun, Hongliang He, Xinyu Yang, Yue Lu, Yingji Zhang, Xuntao Guo, Dong Zhang, Jianzhuang Liu, Jiang Duan, Yijia Xiao, Liangjian Wen, Hai-Ming Xu, Yong Dai - [[pdf]](https://arxiv.org/pdf/2508.01858)</summary>

**Abstract:** Multimodal large-scale models have significantly advanced the development of web agents, enabling perception and interaction with digital environments akin to human cognition. In this paper, we argue that web agents must first acquire sufficient knowledge to effectively engage in cognitive reasoning. Therefore, we decompose a web agent's capabilities into two essential stages: knowledge content learning and cognitive processes. To formalize this, we propose Web-CogKnowledge Framework, categorizing knowledge as Factual, Conceptual, and Procedural. In this framework, knowledge content learning corresponds to the agent's processes of Memorizing and Understanding, which rely on the first two knowledge types, representing the "what" of learning. Conversely, cognitive processes correspond to Exploring, grounded in Procedural knowledge, defining the "how" of reasoning and action. To facilitate knowledge acquisition, we construct the Web-CogDataset, a structured resource curated from 14 real-world websites, designed to systematically instill core knowledge necessary for web agent. This dataset serves as the agent's conceptual grounding-the "nouns" upon which comprehension is built-as well as the basis for learning how to reason and act. Building on this foundation, we operationalize these processes through a novel knowledge-driven Chain-of-Thought (CoT) reasoning framework, developing and training our proposed agent, the Web-CogReasoner. Extensive experimentation reveals its significant superiority over existing models, especially in generalizing to unseen tasks where structured knowledge is decisive. To enable rigorous evaluation, we introduce the Web-CogBench, a comprehensive evaluation suite designed to assess and compare agent performance across the delineated knowledge domains and cognitive capabilities. Our code and data is open sourced at this https URL

**arXiv ID:** 2508.01858
</details>

<details>
<summary><strong>Learning the Pareto Space of Multi-Objective Autonomous Driving: A Modular, Data-Driven Approach</strong> - Mohammad Elayan, Wissam Kontar - [[pdf]](https://arxiv.org/pdf/2601.18913)</summary>

**Abstract:** Balancing safety, efficiency, and interaction is fundamental to designing autonomous driving agents and to understanding autonomous vehicle (AV) behavior in real-world operation. This study introduces an empirical learning framework that derives these trade-offs directly from naturalistic trajectory data. A unified objective space represents each AV timestep through composite scores of safety, efficiency, and interaction. Pareto dominance is applied to identify non-dominated states, forming an empirical frontier that defines the attainable region of balanced performance.
The proposed framework was demonstrated using the Third Generation Simulation (TGSIM) datasets from Foggy Bottom and I-395. Results showed that only 0.23\% of AV driving instances were Pareto-optimal, underscoring the rarity of simultaneous optimization across objectives. Pareto-optimal states showed notably higher mean scores for safety, efficiency, and interaction compared to non-optimal cases, with interaction showing the greatest potential for improvement.
This minimally invasive and modular framework, which requires only kinematic and positional data, can be directly applied beyond the scope of this study to derive and visualize multi-objective learning surfaces

**arXiv ID:** 2601.18913
</details>

<details>
<summary><strong>APEX-Agents</strong> - Bertie Vidgen, Austin Mann, Abby Fennelly, John Wright Stanly, Lucas Rothman, Marco Burstein, Julien Benchek, David Ostrofsky, Anirudh Ravichandran, Debnil Sur, Neel Venugopal, Alannah Hsia, Isaac Robinson, Calix Huang, Olivia Varones, Daniyal Khan, Michael Haines, Zach Richards, Chirag Mahapatra, Brendan Foody, Osvald Nitski - [[pdf]](https://arxiv.org/pdf/2601.14242)</summary>

**Abstract:** We introduce the AI Productivity Index for Agents (APEX-Agents), a benchmark for assessing whether AI agents can execute long-horizon, cross-application tasks created by investment banking analysts, management consultants, and corporate lawyers. APEX-Agents requires agents to navigate realistic work environments with files and tools. We test eight agents for the leaderboard using Pass@1. Gemini 3 Flash (Thinking=High) achieves the highest score of 24.0%, followed by GPT-5.2 (Thinking=High), Claude Opus 4.5 (Thinking=High), and Gemini 3 Pro (Thinking=High). We open source the APEX-Agents benchmark (n=480) with all prompts, rubrics, gold outputs, files, and metadata. We also open-source Archipelago, our infrastructure for agent execution and evaluation.

**arXiv ID:** 2601.14242
</details>

<details>
<summary><strong>Perception-to-Pursuit: Track-Centric Temporal Reasoning for Open-World Drone Detection and Autonomous Chasing</strong> - Venkatakrishna Reddy Oruganti - [[pdf]](https://arxiv.org/pdf/2601.19318)</summary>

**Abstract:** Autonomous drone pursuit requires not only detecting drones but also predicting their trajectories in a manner that enables kinematically feasible interception. Existing tracking methods optimize for prediction accuracy but ignore pursuit feasibility, resulting in trajectories that are physically impossible to intercept 99.9% of the time. We propose Perception-to-Pursuit (P2P), a track-centric temporal reasoning framework that bridges detection and actionable pursuit planning. Our method represents drone motion as compact 8-dimensional tokens capturing velocity, acceleration, scale, and smoothness, enabling a 12-frame causal transformer to reason about future behavior. We introduce the Intercept Success Rate (ISR) metric to measure pursuit feasibility under realistic interceptor constraints. Evaluated on the Anti-UAV-RGBT dataset with 226 real drone sequences, P2P achieves 28.12 pixel average displacement error and 0.597 ISR, representing a 77% improvement in trajectory prediction and 597x improvement in pursuit feasibility over tracking-only baselines, while maintaining perfect drone classification accuracy (100%). Our work demonstrates that temporal reasoning over motion patterns enables both accurate prediction and actionable pursuit planning.

**arXiv ID:** 2601.19318
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>ComAgent: Multi-LLM based Agentic AI Empowered Intelligent Wireless Networks</strong> - Haoyun Li, Ming Xiao, Kezhi Wang, Robert Schober, Dong In Kim, Yong Liang Guan - [[pdf]](https://arxiv.org/pdf/2601.19607)</summary>

**Abstract:** Emerging 6G networks rely on complex cross-layer optimization, yet manually translating high-level intents into mathematical formulations remains a bottleneck. While Large Language Models (LLMs) offer promise, monolithic approaches often lack sufficient domain grounding, constraint awareness, and verification capabilities. To address this, we present ComAgent, a multi-LLM agentic AI framework. ComAgent employs a closed-loop Perception-Planning-Action-Reflection cycle, coordinating specialized agents for literature search, coding, and scoring to autonomously generate solver-ready formulations and reproducible simulations. By iteratively decomposing problems and self-correcting errors, the framework effectively bridges the gap between user intent and execution. Evaluations demonstrate that ComAgent achieves expert-comparable performance in complex beamforming optimization and outperforms monolithic LLMs across diverse wireless tasks, highlighting its potential for automating design in emerging wireless networks.

**arXiv ID:** 2601.19607
</details>

<details>
<summary><strong>LLM Agents for Knowledge Discovery in Atomic Layer Processing</strong> - Andreas Werbrouck, Marshall B. Lindsay, Matthew Maschmann, Matthias J. Young - [[pdf]](https://arxiv.org/pdf/2509.26201)</summary>

**Abstract:** Large Language Models (LLMs) have garnered significant attention for several years now. Recently, their use as independently reasoning agents has been proposed. In this work, we test the potential of such agents for knowledge discovery in materials science. We repurpose LangGraph's tool functionality to supply agents with a black box function to interrogate. In contrast to process optimization or performing specific, user-defined tasks, knowledge discovery consists of freely exploring the system, posing and verifying statements about the behavior of this black box, with the sole objective of generating and verifying generalizable statements. We provide proof of concept for this approach through a children's parlor game, demonstrating the role of trial-and-error and persistence in knowledge discovery, and the strong path-dependence of results. We then apply the same strategy to show that LLM agents can explore, discover, and exploit diverse chemical interactions in an advanced Atomic Layer Processing reactor simulation using intentionally limited probe capabilities without explicit instructions.

**arXiv ID:** 2509.26201
</details>

<details>
<summary><strong>SAC-GLAM: Improving Online RL for LLM agents with Soft Actor-Critic and Hindsight Relabeling</strong> - Loris Gaven, Clement Romac, Thomas Carta, Sylvain Lamprier, Olivier Sigaud, Pierre-Yves Oudeyer - [[pdf]](https://arxiv.org/pdf/2410.12481)</summary>

**Abstract:** The past years have seen Large Language Models (LLMs) strive not only as generative models but also as agents solving textual sequential decision-making tasks. When facing complex environments where their zero-shot abilities are insufficient, recent work showed online Reinforcement Learning (RL) could be used for the LLM agent to discover and learn efficient strategies interactively. However, most prior work sticks to on-policy algorithms, which greatly reduces the scope of methods such agents could use for both exploration and exploitation, such as experience replay and hindsight relabeling. Yet, such methods may be key for LLM learning agents, and in particular when designing autonomous intrinsically motivated agents sampling and pursuing their own goals (i.e. autotelic agents). This paper presents and studies an adaptation of Soft Actor-Critic and hindsight relabeling to LLM agents. Our method not only paves the path towards autotelic LLM agents that learn online but can also outperform on-policy methods in more classic multi-goal RL environments.

**arXiv ID:** 2410.12481
</details>

<details>
<summary><strong>Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents</strong> - Qizheng Zhang, Michael Wornow, Gerry Wan, Kunle Olukotun - [[pdf]](https://arxiv.org/pdf/2506.14852)</summary>

**Abstract:** LLM-based agent applications have shown increasingly remarkable capabilities in complex workflows but incur substantial costs and latency due to extensive planning and reasoning requirements. Existing LLM caching techniques (like context caching and semantic caching), primarily designed for serving chatbots, are insufficient for agent applications where outputs depend on external data and environmental contexts. We propose Agentic Plan Caching (APC), a novel test-time memory that extracts, stores, adapts, and reuses structured plan templates from planning stages of agent applications across semantically similar tasks to reduce the cost and latency of serving. Unlike traditional semantic caching, our system extracts plan templates from completed agent executions at test-time, employs keyword extraction to match new requests against cached plans, and utilizes lightweight models to adapt these templates to task-specific plans with contexts. Evaluation across multiple real-world agent applications shows that our system can reduce costs by 50.31% and latency by 27.28% on average while maintaining performance, offering a more efficient solution for serving LLM-based agents that complements existing LLM serving infrastructures.

**arXiv ID:** 2506.14852
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (20 papers)</h2></summary>

<details>
<summary><strong>More at Stake: How Payoff and Language Shape LLM Agent Strategies in Cooperation Dilemmas</strong> - Trung-Kiet Huynh, Dao-Sy Duy-Minh, Thanh-Bang Cao, Phong-Hao Le, Hong-Dan Nguyen, Nguyen Lam Phu Quy, Minh-Luan Nguyen-Vo, Hong-Phat Pham, Pham Phu Hoa, Thien-Kim Than, Chi-Nguyen Tran, Huy Tran, Gia-Thoai Tran-Le, Alessio Buscemi, Le Hong Trang, Anh Han - [[pdf]](https://arxiv.org/pdf/2601.19082)</summary>

**Abstract:** As LLMs increasingly act as autonomous agents in interactive and multi-agent settings, understanding their strategic behavior is critical for safety, coordination, and AI-driven social and economic systems. We investigate how payoff magnitude and linguistic context shape LLM strategies in repeated social dilemmas, using a payoff-scaled Prisoner's Dilemma to isolate sensitivity to incentive strength. Across models and languages, we observe consistent behavioral patterns, including incentive-sensitive conditional strategies and cross-linguistic divergence. To interpret these dynamics, we train supervised classifiers on canonical repeated-game strategies and apply them to LLM decisions, revealing systematic, model- and language-dependent behavioral intentions, with linguistic framing sometimes matching or exceeding architectural effects. Our results provide a unified framework for auditing LLMs as strategic agents and highlight cooperation biases with direct implications for AI governance and multi-agent system design.

**arXiv ID:** 2601.19082
</details>

<details>
<summary><strong>Multi-Agent Procedural Graph Extraction with Structural and Logical Refinement</strong> - Wangyang Ying, Yanchi Liu, Xujiang Zhao, Wei Cheng, Zhengzhang Chen, Wenchao Yu, Yanjie Fu, Haifeng Chen - [[pdf]](https://arxiv.org/pdf/2601.19170)</summary>

**Abstract:** Automatically extracting workflows as procedural graphs from natural language is promising yet underexplored, demanding both structural validity and logical alignment. While recent large language models (LLMs) show potential for procedural graph extraction, they often produce ill-formed structures or misinterpret logical flows. We present \model{}, a multi-agent framework that formulates procedural graph extraction as a multi-round reasoning process with dedicated structural and logical refinement. The framework iterates through three stages: (1) a graph extraction phase with the graph builder agent, (2) a structural feedback phase in which a simulation agent diagnoses and explains structural defects, and (3) a logical feedback phase in which a semantic agent aligns semantics between flow logic and linguistic cues in the source text. Important feedback is prioritized and expressed in natural language, which is injected into subsequent prompts, enabling interpretable and controllable refinement. This modular design allows agents to target distinct error types without supervision or parameter updates. Experiments demonstrate that \model{} achieves substantial improvements in both structural correctness and logical consistency over strong baselines.

**arXiv ID:** 2601.19170
</details>

<details>
<summary><strong>MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning</strong> - Zhixi Cai, Fucai Ke, Kevin Leo, Sukai Huang, Maria Garcia de la Banda, Peter J. Stuckey, Hamid Rezatofighi - [[pdf]](https://arxiv.org/pdf/2601.19204)</summary>

**Abstract:** Recent vision-language models have strong perceptual ability but their implicit reasoning is hard to explain and easily generates hallucinations on complex queries. Compositional methods improve interpretability, but most rely on a single agent or hand-crafted pipeline and cannot decide when to collaborate across complementary agents or compete among overlapping ones. We introduce MATA (Multi-Agent hierarchical Trainable Automaton), a multi-agent system presented as a hierarchical finite-state automaton for visual reasoning whose top-level transitions are chosen by a trainable hyper agent. Each agent corresponds to a state in the hyper automaton, and runs a small rule-based sub-automaton for reliable micro-control. All agents read and write a shared memory, yielding transparent execution history. To supervise the hyper agent's transition policy, we build transition-trajectory trees and transform to memory-to-next-state pairs, forming the MATA-SFT-90K dataset for supervised finetuning (SFT). The finetuned LLM as the transition policy understands the query and the capacity of agents, and it can efficiently choose the optimal agent to solve the task. Across multiple visual reasoning benchmarks, MATA achieves the state-of-the-art results compared with monolithic and compositional baselines. The code and dataset are available at this https URL.

**arXiv ID:** 2601.19204
</details>

<details>
<summary><strong>Balancing Sustainability And Performance: The Role Of Small-Scale Llms In Agentic Artificial Intelligence Systems</strong> - Anh Khoa Ngo Ho, Martin Chauvin, Simon Gosset, Philippe Cordier, Boris Gamazaychikov - [[pdf]](https://arxiv.org/pdf/2601.19311)</summary>

**Abstract:** As large language models become integral to agentic artificial intelligence systems, their energy demands during inference may pose significant sustainability challenges. This study investigates whether deploying smaller-scale language models can reduce energy consumption without compromising responsiveness and output quality in a multi-agent, real-world environments. We conduct a comparative analysis across language models of varying scales to quantify trade-offs between efficiency and performance. Results show that smaller open-weights models can lower energy usage while preserving task quality. Building on these findings, we propose practical guidelines for sustainable artificial intelligence design, including optimal batch size configuration and computation resource allocation. These insights offer actionable strategies for developing scalable, environmentally responsible artificial intelligence systems.

**arXiv ID:** 2601.19311
</details>

<details>
<summary><strong>CASTER: Breaking the Cost-Performance Barrier in Multi-Agent Orchestration via Context-Aware Strategy for Task Efficient Routing</strong> - Shanyv Liu, Xuyang Yuan, Tao Chen, Zijun Zhan, Zhu Han, Danyang Zheng, Weishan Zhang, Shaohua Cao - [[pdf]](https://arxiv.org/pdf/2601.19793)</summary>

**Abstract:** Graph-based Multi-Agent Systems (MAS) enable complex cyclic workflows but suffer from inefficient static model allocation, where deploying strong models uniformly wastes computation on trivial sub-tasks. We propose CASTER (Context-Aware Strategy for Task Efficient Routing), a lightweight router for dynamic model selection in graph-based MAS. CASTER employs a Dual-Signal Router that combines semantic embeddings with structural meta-features to estimate task difficulty. During training, the router self-optimizes through a Cold Start to Iterative Evolution paradigm, learning from its own routing failures via on-policy negative feedback. Experiments using LLM-as-a-Judge evaluation across Software Engineering, Data Analysis, Scientific Discovery, and Cybersecurity demonstrate that CASTER reduces inference cost by up to 72.4% compared to strong-model baselines while matching their success rates, and consistently outperforms both heuristic routing and FrugalGPT across all domains.

**arXiv ID:** 2601.19793
</details>

<details>
<summary><strong>MulVul: Retrieval-augmented Multi-Agent Code Vulnerability Detection via Cross-Model Prompt Evolution</strong> - Zihan Wu, Jie Xu, Yun Peng, Chun Yong Chong, Xiaohua Jia - [[pdf]](https://arxiv.org/pdf/2601.18847)</summary>

**Abstract:** Large Language Models (LLMs) struggle to automate real-world vulnerability detection due to two key limitations: the heterogeneity of vulnerability patterns undermines the effectiveness of a single unified model, and manual prompt engineering for massive weakness categories is unscalable.
To address these challenges, we propose \textbf{MulVul}, a retrieval-augmented multi-agent framework designed for precise and broad-coverage vulnerability detection. MulVul adopts a coarse-to-fine strategy: a \emph{Router} agent first predicts the top-$k$ coarse categories and then forwards the input to specialized \emph{Detector} agents, which identify the exact vulnerability types. Both agents are equipped with retrieval tools to actively source evidence from vulnerability knowledge bases to mitigate hallucinations.
Crucially, to automate the generation of specialized prompts, we design \emph{Cross-Model Prompt Evolution}, a prompt optimization mechanism where a generator LLM iteratively refines candidate prompts while a distinct executor LLM validates their effectiveness. This decoupling mitigates the self-correction bias inherent in single-model optimization.
Evaluated on 130 CWE types, MulVul achieves 34.79\% Macro-F1, outperforming the best baseline by 41.5\%. Ablation studies validate cross-model prompt evolution, which boosts performance by 51.6\% over manual prompts by effectively handling diverse vulnerability patterns.

**arXiv ID:** 2601.18847
</details>

<details>
<summary><strong>LLMs as Orchestrators: Constraint-Compliant Multi-Agent Optimization for Recommendation Systems</strong> - Guilin Zhang, Kai Zhao, Jeffrey Friedman, Xu Chu - [[pdf]](https://arxiv.org/pdf/2601.19121)</summary>

**Abstract:** Recommendation systems must optimize multiple objectives while satisfying hard business constraints such as fairness and coverage. For example, an e-commerce platform may require every recommendation list to include items from multiple sellers and at least one newly listed product; violating such constraints--even once--is unacceptable in production. Prior work on multi-objective recommendation and recent LLM-based recommender agents largely treat constraints as soft penalties or focus on item scoring and interaction, leading to frequent violations in real-world deployments. How to leverage LLMs for coordinating constrained optimization in recommendation systems remains underexplored. We propose DualAgent-Rec, an LLM-coordinated dual-agent framework for constrained multi-objective e-commerce recommendation. The framework separates optimization into an Exploitation Agent that prioritizes accuracy under hard constraints and an Exploration Agent that promotes diversity through unconstrained Pareto search. An LLM-based coordinator adaptively allocates resources between agents based on optimization progress and constraint satisfaction, while an adaptive epsilon-relaxation mechanism guarantees feasibility of final solutions. Experiments on the Amazon Reviews 2023 dataset demonstrate that DualAgent-Rec achieves 100% constraint satisfaction and improves Pareto hypervolume by 4-6% over strong baselines, while maintaining competitive accuracy-diversity trade-offs. These results indicate that LLMs can act as effective orchestration agents for deployable and constraint-compliant recommendation systems.

**arXiv ID:** 2601.19121
</details>

<details>
<summary><strong>SHIELD: An Auto-Healing Agentic Defense Framework for LLM Resource Exhaustion Attacks</strong> - Nirhoshan Sivaroopan, Kanchana Thilakarathna, Albert Zomaya, Manu, Yi Guo, Jo Plested, Tim Lynar, Jack Yang, Wangli Yang - [[pdf]](https://arxiv.org/pdf/2601.19174)</summary>

**Abstract:** Sponge attacks increasingly threaten LLM systems by inducing excessive computation and DoS. Existing defenses either rely on statistical filters that fail on semantically meaningful attacks or use static LLM-based detectors that struggle to adapt as attack strategies evolve. We introduce SHIELD, a multi-agent, auto-healing defense framework centered on a three-stage Defense Agent that integrates semantic similarity retrieval, pattern matching, and LLM-based reasoning. Two auxiliary agents, a Knowledge Updating Agent and a Prompt Optimization Agent, form a closed self-healing loop, when an attack bypasses detection, the system updates an evolving knowledgebase, and refines defense instructions. Extensive experiments show that SHIELD consistently outperforms perplexity-based and standalone LLM defenses, achieving high F1 scores across both non-semantic and semantic sponge attacks, demonstrating the effectiveness of agentic self-healing against evolving resource-exhaustion threats.

**arXiv ID:** 2601.19174
</details>

<details>
<summary><strong>Veri-Sure: A Contract-Aware Multi-Agent Framework with Temporal Tracing and Formal Verification for Correct RTL Code Generation</strong> - Jiale Liu, Taiyu Zhou, Tianqi Jiang - [[pdf]](https://arxiv.org/pdf/2601.19747)</summary>

**Abstract:** In the rapidly evolving field of Electronic Design Automation (EDA), the deployment of Large Language Models (LLMs) for Register-Transfer Level (RTL) design has emerged as a promising direction. However, silicon-grade correctness remains bottlenecked by: (i) limited test coverage and reliability of simulation-centric evaluation, (ii) regressions and repair hallucinations introduced by iterative debugging, and (iii) semantic drift as intent is reinterpreted across agent handoffs. In this work, we propose Veri-Sure, a multi-agent framework that establishes a design contract to align agents' intent and uses a patching mechanism guided by static dependency slicing to perform precise, localized repairs. By integrating a multi-branch verification pipeline that combines trace-driven temporal analysis with formal verification consisting of assertion-based checking and boolean equivalence proofs, Veri-Sure enables functional correctness beyond pure simulations. We also introduce VerilogEval-v2-EXT, extending the original benchmark with 53 more industrial-grade design tasks and stratified difficulty levels, and show that Veri-Sure achieves state-of-the-art verified-correct RTL code generation performance, surpassing standalone LLMs and prior agentic systems.

**arXiv ID:** 2601.19747
</details>

<details>
<summary><strong>Reimagining Peer Review Process Through Multi-Agent Mechanism Design</strong> - Ahmad Farooq, Kamran Iqbal - [[pdf]](https://arxiv.org/pdf/2601.19778)</summary>

**Abstract:** The software engineering research community faces a systemic crisis: peer review is failing under growing submissions, misaligned incentives, and reviewer fatigue. Community surveys reveal that researchers perceive the process as "broken." This position paper argues that these dysfunctions are mechanism design failures amenable to computational solutions. We propose modeling the research community as a stochastic multi-agent system and applying multi-agent reinforcement learning to design incentive-compatible protocols. We outline three interventions: a credit-based submission economy, MARL-optimized reviewer assignment, and hybrid verification of review consistency. We present threat models, equity considerations, and phased pilot metrics. This vision charts a research agenda toward sustainable peer review.

**arXiv ID:** 2601.19778
</details>

<details>
<summary><strong>1-2-3 Check: Enhancing Contextual Privacy in LLM via Multi-Agent Reasoning</strong> - Wenkai Li, Liwen Sun, Zhenxiang Guan, Xuhui Zhou, Maarten Sap - [[pdf]](https://arxiv.org/pdf/2508.07667)</summary>

**Abstract:** Addressing contextual privacy concerns remains challenging in interactive settings where large language models (LLMs) process information from multiple sources (e.g., summarizing meetings with private and public information). We introduce a multi-agent framework that decomposes privacy reasoning into specialized subtasks (extraction, classification), reducing the information load on any single agent while enabling iterative validation and more reliable adherence to contextual privacy norms. To understand how privacy errors emerge and propagate, we conduct a systematic ablation over information-flow topologies, revealing when and why upstream detection mistakes cascade into downstream leakage. Experiments on the ConfAIde and PrivacyLens benchmark with several open-source and closed-sourced LLMs demonstrate that our best multi-agent configuration substantially reduces private information leakage (\textbf{18\%} on ConfAIde and \textbf{19\%} on PrivacyLens with GPT-4o) while preserving the fidelity of public content, outperforming single-agent baselines. These results highlight the promise of principled information-flow design in multi-agent systems for contextual privacy with LLMs.

**arXiv ID:** 2508.07667
</details>

<details>
<summary><strong>Rethinking the AI Scientist: Interactive Multi-Agent Workflows for Scientific Discovery</strong> - Lukas Weidener, Marko Brkić, Mihailo Jovanović, Ritvik Singh, Chiara Baccin, Emre Ulgac, Alex Dobrin, Aakaash Meduri - [[pdf]](https://arxiv.org/pdf/2601.12542)</summary>

**Abstract:** Artificial intelligence systems for scientific discovery have demonstrated remarkable potential, yet existing approaches remain largely proprietary and operate in batch-processing modes requiring hours per research cycle, precluding real-time researcher guidance. This paper introduces Deep Research, a multi-agent system enabling interactive scientific investigation with turnaround times measured in minutes. The architecture comprises specialized agents for planning, data analysis, literature search, and novelty detection, unified through a persistent world state that maintains context across iterative research cycles. Two operational modes support different workflows: semi-autonomous mode with selective human checkpoints, and fully autonomous mode for extended investigations. Evaluation on the BixBench computational biology benchmark demonstrated state-of-the-art performance, achieving 48.8% accuracy on open response and 64.4% on multiple-choice evaluation, exceeding existing baselines by 14 to 26 percentage points. Analysis of architectural constraints, including open access literature limitations and challenges inherent to automated novelty assessment, informs practical deployment considerations for AI-assisted scientific workflows.

**arXiv ID:** 2601.12542
</details>

<details>
<summary><strong>The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems</strong> - Prateek Gupta, Qiankun Zhong, Hiromu Yakura, Thomas Eisenmann, Iyad Rahwan - [[pdf]](https://arxiv.org/pdf/2510.14401)</summary>

**Abstract:** A growing body of multi-agent studies with LLMs explores how norms and cooperation emerge in mixed-motive scenarios, where pursuing individual gain can undermine the collective good. While prior work has explored these dynamics in both richly contextualized simulations and simplified game-theoretic environments, most LLM systems featuring common-pool resource (CPR) games provide agents with explicit reward functions directly tied to their actions. In contrast, human cooperation often emerges without explicit knowledge of the payoff structure or how individual actions translate into long-run outcomes, relying instead on heuristics, communication, and enforcement. We introduce a CPR simulation framework that removes explicit reward signals and embeds cultural-evolutionary mechanisms: social learning (adopting strategies and beliefs from successful peers) and norm-based punishment, grounded in Ostrom's principles of resource governance. Agents also individually learn from the consequences of harvesting, monitoring, and punishing via environmental feedback, enabling norms to emerge endogenously. We establish the validity of our simulation by reproducing key findings from existing studies on human behavior. Building on this, we examine norm evolution across a $2\times2$ grid of environmental and social initialisations (resource-rich vs. resource-scarce; altruistic vs. selfish) and benchmark how agentic societies comprised of different LLMs perform under these conditions. Our results reveal systematic model differences in sustaining cooperation and norm formation, positioning the framework as a rigorous testbed for studying emergent norms in mixed-motive LLM societies. Such analysis can inform the design of AI systems deployed in social and organizational contexts, where alignment with cooperative norms is critical for stability, fairness, and effective governance of AI-mediated environments.

**arXiv ID:** 2510.14401
</details>

<details>
<summary><strong>MetaGen: Self-Evolving Roles and Topologies for Multi-Agent LLM Reasoning</strong> - Yimeng Wang, Jiaxing Zhao, Hongbin Xie, Hexing Ma, Yuzhen Lei, Shuangxue Liu, Xuan Song, Zichen Zhang, Haoran Zhang - [[pdf]](https://arxiv.org/pdf/2601.19290)</summary>

**Abstract:** Large language models are increasingly deployed as multi-agent systems, where specialized roles communicate and collaborate through structured interactions to solve complex tasks that often exceed the capacity of a single agent. However, most existing systems still rely on a fixed role library and an execution-frozen interaction topology, a rigid design choice that frequently leads to task mismatch, prevents timely adaptation when new evidence emerges during reasoning, and further inflates inference cost. We introduce MetaGen, a training-free framework that adapts both the role space and the collaboration topology at inference time, without updating base model weights. MetaGen generates and rewrites query-conditioned role specifications to maintain a controllable dynamic role pool, then instantiates a constrained execution graph around a minimal backbone. During execution, it iteratively updates role prompts and adjusts structural decisions using lightweight feedback signals. Experiments on code generation and multi-step reasoning benchmarks show that MetaGen improves the accuracy and cost tradeoff over strong multi-agent baselines.

**arXiv ID:** 2601.19290
</details>

<details>
<summary><strong>Automated Safety Benchmarking: A Multi-agent Pipeline for LVLMs</strong> - Xiangyang Zhu, Yuan Tian, Zicheng Zhang, Qi Jia, Chunyi Li, Renrui Zhang, Heng Li, Zongrui Wang, Wei Sun - [[pdf]](https://arxiv.org/pdf/2601.19507)</summary>

**Abstract:** Large vision-language models (LVLMs) exhibit remarkable capabilities in cross-modal tasks but face significant safety challenges, which undermine their reliability in real-world applications. Efforts have been made to build LVLM safety evaluation benchmarks to uncover their vulnerability. However, existing benchmarks are hindered by their labor-intensive construction process, static complexity, and limited discriminative power. Thus, they may fail to keep pace with rapidly evolving models and emerging risks. To address these limitations, we propose VLSafetyBencher, the first automated system for LVLM safety benchmarking. VLSafetyBencher introduces four collaborative agents: Data Preprocessing, Generation, Augmentation, and Selection agents to construct and select high-quality samples. Experiments validates that VLSafetyBencher can construct high-quality safety benchmarks within one week at a minimal cost. The generated benchmark effectively distinguish safety, with a safety rate disparity of 70% between the most and least safe models.

**arXiv ID:** 2601.19507
</details>

<details>
<summary><strong>Improving Implicit Hate Speech Detection via a Community-Driven Multi-Agent Framework</strong> - Ewelina Gajewska, Katarzyna Budzynska, Jarosław A Chudziak - [[pdf]](https://arxiv.org/pdf/2601.09342)</summary>

**Abstract:** This work proposes a contextualised detection framework for implicitly hateful speech, implemented as a multi-agent system comprising a central Moderator Agent and dynamically constructed Community Agents representing specific demographic groups. Our approach explicitly integrates socio-cultural context from publicly available knowledge sources, enabling identity-aware moderation that surpasses state-of-the-art prompting methods (zero-shot prompting, few-shot prompting, chain-of-thought prompting) and alternative approaches on a challenging ToxiGen dataset. We enhance the technical rigour of performance evaluation by incorporating balanced accuracy as a central metric of classification fairness that accounts for the trade-off between true positive and true negative rates. We demonstrate that our community-driven consultative framework significantly improves both classification accuracy and fairness across all target groups.

**arXiv ID:** 2601.09342
</details>

<details>
<summary><strong>Multimodal Multi-Agent Empowered Legal Judgment Prediction</strong> - Zhaolu Kang, Junhao Gong, Qingxi Chen, Hao Zhang, Jiaxin Liu, Rong Fu, Zhiyuan Feng, Yuan Wang, Simon Fong, Kaiyue Zhou - [[pdf]](https://arxiv.org/pdf/2601.12815)</summary>

**Abstract:** Legal Judgment Prediction (LJP) aims to predict the outcomes of legal cases based on factual descriptions, serving as a fundamental task to advance the development of legal systems. Traditional methods often rely on statistical analyses or role-based simulations but face challenges with multiple allegations, diverse evidence, and lack adaptability. In this paper, we introduce JurisMMA, a novel framework for LJP that effectively decomposes trial tasks, standardizes processes, and organizes them into distinct stages. Furthermore, we build JurisMM, a large dataset with over 100,000 recent Chinese judicial records, including both text and multimodal video-text data, enabling comprehensive evaluation. Experiments on JurisMM and the benchmark LawBench validate our framework's effectiveness. These results indicate that our framework is effective not only for LJP but also for a broader range of legal applications, offering new perspectives for the development of future legal methods and datasets.

**arXiv ID:** 2601.12815
</details>

<details>
<summary><strong>Endless Terminals: Scaling RL Environments for Terminal Agents</strong> - Kanishk Gandhi, Shivam Garg, Noah D. Goodman, Dimitris Papailiopoulos - [[pdf]](https://arxiv.org/pdf/2601.16443)</summary>

**Abstract:** Environments are the bottleneck for self-improving agents. Current terminal benchmarks were built for evaluation, not training; reinforcement learning requires a scalable pipeline, not just a dataset. We introduce Endless Terminals, a fully autonomous pipeline that procedurally generates terminal-use tasks without human annotation. The pipeline has four stages: generating diverse task descriptions, building and validating containerized environments, producing completion tests, and filtering for solvability. From this pipeline we obtain 3255 tasks spanning file operations, log management, data processing, scripting, and database operations. We train agents using vanilla PPO with binary episode level rewards and a minimal interaction loop: no retrieval, multi-agent coordination, or specialized tools. Despite this simplicity, models trained on Endless Terminals show substantial gains: on our held-out dev set, Llama-3.2-3B improves from 4.0% to 18.2%, Qwen2.5-7B from 10.7% to 53.3%, and Qwen3-8B-openthinker-sft from 42.6% to 59.0%. These improvements transfer to human-curated benchmarks: models trained on Endless Terminals show substantial gains on held out human curated benchmarks: on TerminalBench 2.0, Llama-3.2-3B improves from 0.0% to 2.2%, Qwen2.5-7B from 2.2% to 3.4%, and Qwen3-8B-openthinker-sft from 1.1% to 6.7%, in each case outperforming alternative approaches including models with more complex agentic scaffolds. These results demonstrate that simple RL succeeds when environments scale.

**arXiv ID:** 2601.16443
</details>

<details>
<summary><strong>JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation</strong> - Guillem Capellera, Luis Ferraz, Antonio Rubio, Alexandre Alahi, Antonio Agudo - [[pdf]](https://arxiv.org/pdf/2509.22522)</summary>

**Abstract:** Generative models often treat continuous data and discrete events as separate processes, creating a gap in modeling complex systems where they interact synchronously. To bridge this gap, we introduce JointDiff, a novel diffusion framework designed to unify these two processes by simultaneously generating continuous spatio-temporal data and synchronous discrete events. We demonstrate its efficacy in the sports domain by simultaneously modeling multi-agent trajectories and key possession events. This joint modeling is validated with non-controllable generation and two novel controllable generation scenarios: weak-possessor-guidance, which offers flexible semantic control over game dynamics through a simple list of intended ball possessors, and text-guidance, which enables fine-grained, language-driven generation. To enable the conditioning with these guidance signals, we introduce CrossGuid, an effective conditioning operation for multi-agent domains. We also share a new unified sports benchmark enhanced with textual descriptions for soccer and football datasets. JointDiff achieves state-of-the-art performance, demonstrating that joint modeling is crucial for building realistic and controllable generative models for interactive systems.

**arXiv ID:** 2509.22522
</details>

<details>
<summary><strong>Judgelight: Trajectory-Level Post-Optimization for Multi-Agent Path Finding via Closed-Subwalk Collapsing</strong> - Yimin Tang, Sven Koenig, Erdem Bıyık - [[pdf]](https://arxiv.org/pdf/2601.19388)</summary>

**Abstract:** Multi-Agent Path Finding (MAPF) is an NP-hard problem with applications in warehouse automation and multi-robot coordination. Learning-based MAPF solvers offer fast and scalable planning but often produce feasible trajectories that contain unnecessary or oscillatory movements. We propose Judgelight, a post-optimization method that improves trajectory quality after a MAPF solver generates a feasible schedule. Judgelight collapses closed subwalks in agents' trajectories to remove redundant movements while preserving all feasibility constraints. We formalize this process as MAPF-Collapse, prove that it is NP-hard, and present an exact optimization approach by formulating it as integer linear programming (ILP) problem. Experimental results show Judgelight consistently reduces solution cost by around 20%, particularly for learning-based solvers, producing trajectories that are better suited for real-world deployment.

**arXiv ID:** 2601.19388
</details>

</details>

<details open>
<summary><h2>Other Agent Research (3 papers)</h2></summary>

<details>
<summary><strong>Agentic Digital Twins: A Taxonomy of Capabilities for Understanding Possible Futures</strong> - Christopher Burr, Mark Enzer, Jason Shepherd, David Wagg - [[pdf]](https://arxiv.org/pdf/2601.18799)</summary>

**Abstract:** As digital twins (DTs) evolve to become more agentic through the integration of artificial intelligence (AI), they acquire capabilities that extend beyond dynamic representation of their target systems. This paper presents a taxonomy of agentic DTs organised around three fundamental dimensions: the locus of agency (external, internal, distributed), the tightness of coupling (loose, tight, constitutive), and model evolution (static, adaptive, reconstructive). From the resulting 27-configuration space, we identify nine illustrative configurations grouped into three clusters: "The Present" (existing tools and emerging steering systems), "The Threshold" (where emergent properties appear and coupling becomes constitutive), and "The Frontier" (where systems gain reconstructive capabilities).
Our analysis explores how agentic DTs exercise performative power--not merely representing physical systems but actively participating in constituting them. Using traffic navigation systems as examples, we show how even passive tools can exhibit emergent performativity, while advanced configurations risk performative lock-in. Drawing on performative prediction theory, we trace a progression from passive tools through active steering to ontological reconstruction, examining how constitutive coupling enables systems to create self-validating realities. Understanding these configurations is essential for navigating the transformation from DTs as mirror worlds to DTs as architects of new ontologies.

**arXiv ID:** 2601.18799
</details>

<details>
<summary><strong>Dynamic Cogeneration of Bug Reproduction Test in Agentic Program Repair</strong> - Runxiang Cheng, Michele Tufano, José Cambronero, Renyao Wei, Sherry Shi, Grant Uy, Pat Rondon, Franjo Ivančić - [[pdf]](https://arxiv.org/pdf/2601.19066)</summary>

**Abstract:** Bug Reproduction Tests (BRTs) have been used in many agentic Automated Program Repair (APR) systems, primarily for validating promising fixes and aiding fix generation. In practice, when developers submit a patch, they often implement the BRT alongside the fix. Our experience deploying agentic APR reveals that developers similarly desire a BRT within AI-generated patches to increase their confidence. However, canonical APR systems tend to generate BRTs and fixes separately, or focus on producing only the fix in the final patch. In this paper, we study agentic APR in the context of cogeneration, where the APR agent is instructed to generate both a fix and a BRT in the same patch. We evaluate the effectiveness of different cogeneration strategies on 120 human-reported bugs at Google and characterize different cogeneration strategies by their influence on APR agent behavior. We develop and evaluate patch selectors that account for test change information to select patches with plausible fixes (and plausible BRTs). Finally, we analyze the root causes of failed cogeneration trajectories. Importantly, we show that cogeneration allows the APR agent to generate BRTs for at least as many bugs as a dedicated BRT agent, without compromising the generation rate of plausible fixes, thereby reducing engineering effort in maintaining and coordinating separate generation pipelines for fix and BRT at scale.

**arXiv ID:** 2601.19066
</details>

<details>
<summary><strong>Remote Sensing Image Intelligent Interpretation with the Language-Centered Perspective: Principles, Methods and Challenges</strong> - Haifeng Li, Wang Guo, Haiyang Wu, Mengwei Wu, Jipeng Zhang, Qing Zhu, Yu Liu, Xin Huang, Chao Tao - [[pdf]](https://arxiv.org/pdf/2508.06832)</summary>

**Abstract:** The mainstream paradigm of remote sensing image interpretation has long been dominated by vision-centered models, which rely on visual features for semantic understanding. However, these models face inherent limitations in handling multi-modal reasoning, semantic abstraction, and interactive decision-making. While recent advances have introduced Large Language Models (LLMs) into remote sensing workflows, existing studies primarily focus on downstream applications, lacking a unified theoretical framework that explains the cognitive role of language. This review advocates a paradigm shift from vision-centered to language-centered remote sensing interpretation. Drawing inspiration from the Global Workspace Theory (GWT) of human cognition, We propose a language-centered framework for remote sensing interpretation that treats LLMs as the cognitive central hub integrating perceptual, task, knowledge and action spaces to enable unified understanding, reasoning, and decision-making. We first explore the potential of LLMs as the central cognitive component in remote sensing interpretation, and then summarize core technical challenges, including unified multimodal representation, knowledge association, and reasoning and decision-making. Furthermore, we construct a global workspace-driven interpretation mechanism and review how language-centered solutions address each challenge. Finally, we outline future research directions from four perspectives: adaptive alignment of multimodal data, task understanding under dynamic knowledge constraints, trustworthy reasoning, and autonomous interaction. This work aims to provide a conceptual foundation for the next generation of remote sensing interpretation systems and establish a roadmap toward cognition-driven intelligent geospatial analysis.

**arXiv ID:** 2508.06832
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (32 papers)</h2></summary>

<details>
<summary><strong>Agentic Business Process Management Systems</strong> - Marlon Dumas, Fredrik Milani, David Chapela-Campa - [[pdf]](https://arxiv.org/pdf/2601.18833)</summary>

**Abstract:** Since the early 90s, the evolution of the Business Process Management (BPM) discipline has been punctuated by successive waves of automation technologies. Some of these technologies enable the automation of individual tasks, while others focus on orchestrating the execution of end-to-end processes. The rise of Generative and Agentic Artificial Intelligence (AI) is opening the way for another such wave. However, this wave is poised to be different because it shifts the focus from automation to autonomy and from design-driven management of business processes to data-driven management, leveraging process mining techniques. This position paper, based on a keynote talk at the 2025 Workshop on AI for BPM, outlines how process mining has laid the foundations on top of which agents can sense process states, reason about improvement opportunities, and act to maintain and optimize performance. The paper proposes an architectural vision for Agentic Business Process Management Systems (A-BPMS): a new class of platforms that integrate autonomy, reasoning, and learning into process management and execution. The paper contends that such systems must support a continuum of processes, spanning from human-driven to fully autonomous, thus redefining the boundaries of process automation and governance.

**arXiv ID:** 2601.18833
</details>

<details>
<summary><strong>Exploring Weaknesses in Function Call Models via Reinforcement Learning: An Adversarial Data Augmentation Approach</strong> - Weiran Guo, Bing Bo, Shaoxiang Wu, Jingsheng Yang - [[pdf]](https://arxiv.org/pdf/2601.19122)</summary>

**Abstract:** Function call capabilities have become crucial for Large Language Models (LLMs), enabling them to interact more effectively with external tools and APIs. Existing methods for improving the function call capabilities of LLMs rely on data obtained either through manual annotation or automated generation by models, and use this data to finetune the LLMs. However, these methods often lack targeted design and are constrained by fixed patterns and data distributions, which limits their effectiveness in enhancing the generalization and robustness of function call LLMs. To address this limitation, we propose a novel adversarial data augmentation method that employs reinforcement learning to systematically identify and target the weaknesses of function call LLMs. Our training framework introduces a query model trained with reinforcement learning (RL) to generate adversarial queries that are specifically designed to challenge function call (FC) models. This approach adopts a zero sum game formulation, where the query model and the FC model engage in iterative alternating training. Overall, our method advances the development of more robust FC models and provides a systematic way to identify and correct weaknesses in the ability of LLMs to interact with external tools.

**arXiv ID:** 2601.19122
</details>

<details>
<summary><strong>LocationAgent: A Hierarchical Agent for Image Geolocation via Decoupling Strategy and Evidence from Parametric Knowledge</strong> - Qiujun Li, Zijin Xiao, Xulin Wang, Zhidan Ma, Cheng Yang, Haifeng Li - [[pdf]](https://arxiv.org/pdf/2601.19155)</summary>

**Abstract:** Image geolocation aims to infer capture locations based on visual content. Fundamentally, this constitutes a reasoning process composed of \textit{hypothesis-verification cycles}, requiring models to possess both geospatial reasoning capabilities and the ability to verify evidence against geographic facts. Existing methods typically internalize location knowledge and reasoning patterns into static memory via supervised training or trajectory-based reinforcement fine-tuning. Consequently, these methods are prone to factual hallucinations and generalization bottlenecks in open-world settings or scenarios requiring dynamic knowledge. To address these challenges, we propose a Hierarchical Localization Agent, called LocationAgent. Our core philosophy is to retain hierarchical reasoning logic within the model while offloading the verification of geographic evidence to external tools. To implement hierarchical reasoning, we design the RER architecture (Reasoner-Executor-Recorder), which employs role separation and context compression to prevent the drifting problem in multi-step reasoning. For evidence verification, we construct a suite of clue exploration tools that provide diverse evidence to support location reasoning. Furthermore, to address data leakage and the scarcity of Chinese data in existing datasets, we introduce CCL-Bench (China City Location Bench), an image geolocation benchmark encompassing various scene granularities and difficulty levels. Extensive experiments demonstrate that LocationAgent significantly outperforms existing methods by at least 30\% in zero-shot settings.

**arXiv ID:** 2601.19155
</details>

<details>
<summary><strong>Curiosity Driven Knowledge Retrieval for Mobile Agents</strong> - Sijia Li, Xiaoyu Tan, Shahir Ali, Niels Schmidt, Gengchen Ma, Xihe Qiu - [[pdf]](https://arxiv.org/pdf/2601.19306)</summary>

**Abstract:** Mobile agents have made progress toward reliable smartphone automation, yet performance in complex applications remains limited by incomplete knowledge and weak generalization to unseen environments. We introduce a curiosity driven knowledge retrieval framework that formalizes uncertainty during execution as a curiosity score. When this score exceeds a threshold, the system retrieves external information from documentation, code repositories, and historical trajectories. Retrieved content is organized into structured AppCards, which encode functional semantics, parameter conventions, interface mappings, and interaction patterns. During execution, an enhanced agent selectively integrates relevant AppCards into its reasoning process, thereby compensating for knowledge blind spots and improving planning reliability. Evaluation on the AndroidWorld benchmark shows consistent improvements across backbones, with an average gain of six percentage points and a new state of the art success rate of 88.8\% when combined with GPT-5. Analysis indicates that AppCards are particularly effective for multi step and cross application tasks, while improvements depend on the backbone model. Case studies further confirm that AppCards reduce ambiguity, shorten exploration, and support stable execution trajectories. Task trajectories are publicly available at this https URL.

**arXiv ID:** 2601.19306
</details>

<details>
<summary><strong>Reinforcement Learning for Quantum Technology</strong> - Marin Bukov, Florian Marquardt - [[pdf]](https://arxiv.org/pdf/2601.18953)</summary>

**Abstract:** Many challenges arising in Quantum Technology can be successfully addressed using a set of machine learning algorithms collectively known as reinforcement learning (RL), based on adaptive decision-making through interaction with the quantum device. After a concise and intuitive introduction to RL aimed at a broad physics readership, we discuss the key ideas and core concepts in reinforcement learning with a particular focus on quantum systems. We then survey recent progress in RL in all relevant areas. We discuss state preparation in few- and many-body quantum systems, the design and optimization of high-fidelity quantum gates, and the automated construction of quantum circuits, including applications to variational quantum eigensolvers and architecture search. We further highlight the interactive capabilities of RL agents, emphasizing recent progress in quantum feedback control and quantum error correction, and briefly discuss quantum reinforcement learning as well as applications to quantum metrology. The review concludes with a discussion of open challenges -- such as scalability, interpretability, and integration with experimental platforms -- and outlines promising directions for future research. Throughout, we highlight experimental implementations that exemplify the increasing role of reinforcement learning in shaping the development of quantum technologies.

**arXiv ID:** 2601.18953
</details>

<details>
<summary><strong>Group Distributionally Robust Optimization-Driven Reinforcement Learning for LLM Reasoning</strong> - Kishan Panaganti, Zhenwen Liang, Wenhao Yu, Haitao Mi, Dong Yu - [[pdf]](https://arxiv.org/pdf/2601.19280)</summary>

**Abstract:** Recent progress in Large Language Model (LLM) reasoning is increasingly driven by the refinement of post-training loss functions and alignment strategies. However, standard Reinforcement Learning (RL) paradigms like Group Relative Policy Optimization (GRPO) remain constrained by static uniformity: uniform prompt sampling and a fixed number of rollouts per prompt. For heterogeneous, heavy-tailed reasoning data, this creates structural inefficiencies that waste compute on already-solved patterns while under-training the long tail of hard problems. To address this, we propose Multi-Adversary Group Distributionally Robust Optimization (GDRO), an optimization-first framework that moves beyond uniform reasoning models by dynamically adapting the training distribution.
We introduce an Online Difficulty Classifier that partitions prompts into dynamic pass@k difficulty groups. We then propose two independent GDRO games for post-training: (1) Prompt-GDRO, which employs an EMA-debiased multiplicative-weights bandit sampler to target the intensive difficulty margin and upweight persistently hard groups without frequency bias; and (2) Rollout-GDRO, which uses a shadow-price controller to reallocate rollouts across groups, maximizing gradient variance reduction on hard tasks under a fixed mean budget (compute-neutral). We provide no-regret guarantees for both controllers and additionally a variance-proxy analysis motivating a square-root optimal rollout allocation for Rollout-GDRO. We validate our framework on the DAPO 14.1k dataset using Qwen3-Base models. Prompt-GDRO and Rollout-GDRO achieve average relative gains of +10.6% and +10.1%, respectively, in pass@8 accuracy across 1.7B, 4B, and 8B scales compared to the GRPO baseline. Qualitative analysis shows an emergent curriculum: the adversaries shift resources to the evolving reasoning frontier, enhancing the reasoning model's performance.

**arXiv ID:** 2601.19280
</details>

<details>
<summary><strong>From Observations to Events: Event-Aware World Model for Reinforcement Learning</strong> - Zhao-Han Peng, Shaohui Li, Zhi Li, Shulan Ruan, Yu Liu, You He - [[pdf]](https://arxiv.org/pdf/2601.19336)</summary>

**Abstract:** While model-based reinforcement learning (MBRL) improves sample efficiency by learning world models from raw observations, existing methods struggle to generalize across structurally similar scenes and remain vulnerable to spurious variations such as textures or color shifts. From a cognitive science perspective, humans segment continuous sensory streams into discrete events and rely on these key events for decision-making. Motivated by this principle, we propose the Event-Aware World Model (EAWM), a general framework that learns event-aware representations to streamline policy learning without requiring handcrafted labels. EAWM employs an automated event generator to derive events from raw observations and introduces a Generic Event Segmentor (GES) to identify event boundaries, which mark the start and end time of event segments. Through event prediction, the representation space is shaped to capture meaningful spatio-temporal transitions. Beyond this, we present a unified formulation of seemingly distinct world model architectures and show the broad applicability of our methods. Experiments on Atari 100K, Craftax 1M, and DeepMind Control 500K, DMC-GB2 500K demonstrate that EAWM consistently boosts the performance of strong MBRL baselines by 10%-45%, setting new state-of-the-art results across benchmarks. Our code is released at this https URL.

**arXiv ID:** 2601.19336
</details>

<details>
<summary><strong>LLM-Enhanced Reinforcement Learning for Long-Term User Satisfaction in Interactive Recommendation</strong> - Chongjun Xia, Yanchun Peng, Xianzhi Wang - [[pdf]](https://arxiv.org/pdf/2601.19585)</summary>

**Abstract:** Interactive recommender systems can dynamically adapt to user feedback, but often suffer from content homogeneity and filter bubble effects due to overfitting short-term user preferences. While recent efforts aim to improve content diversity, they predominantly operate in static or one-shot settings, neglecting the long-term evolution of user interests. Reinforcement learning provides a principled framework for optimizing long-term user satisfaction by modeling sequential decision-making processes. However, its application in recommendation is hindered by sparse, long-tailed user-item interactions and limited semantic planning capabilities. In this work, we propose LLM-Enhanced Reinforcement Learning (LERL), a novel hierarchical recommendation framework that integrates the semantic planning power of LLM with the fine-grained adaptability of RL. LERL consists of a high-level LLM-based planner that selects semantically diverse content categories, and a low-level RL policy that recommends personalized items within the selected semantic space. This hierarchical design narrows the action space, enhances planning efficiency, and mitigates overexposure to redundant content. Extensive experiments on real-world datasets demonstrate that LERL significantly improves long-term user satisfaction when compared with state-of-the-art baselines. The implementation of LERL is available at this https URL.

**arXiv ID:** 2601.19585
</details>

<details>
<summary><strong>R^3: Replay, Reflection, and Ranking Rewards for LLM Reinforcement Learning</strong> - Zhizheng Jiang, Kang Zhao, Weikai Xu, Xinkui Lin, Wei Liu, Jian Luan, Shuo Shang, Peng Han - [[pdf]](https://arxiv.org/pdf/2601.19620)</summary>

**Abstract:** Large reasoning models (LRMs) aim to solve diverse and complex problems through structured reasoning. Recent advances in group-based policy optimization methods have shown promise in enabling stable advantage estimation without reliance on process-level annotations. However, these methods rely on advantage gaps induced by high-quality samples within the same batch, which makes the training process fragile and inefficient when intra-group advantages collapse under challenging tasks. To address these problems, we propose a reinforcement learning mechanism named \emph{\textbf{R^3}} that along three directions: (1) a \emph{cross-context \underline{\textbf{R}}eplay} strategy that maintains the intra-group advantage by recalling valuable examples from historical trajectories of the same query, (2) an \emph{in-context self-\underline{\textbf{R}}eflection} mechanism enabling models to refine outputs by leveraging past failures, and (3) a \emph{structural entropy \underline{\textbf{R}}anking reward}, which assigns relative rewards to truncated or failed samples by ranking responses based on token-level entropy patterns, capturing both local exploration and global stability. We implement our method on Deepseek-R1-Distill-Qwen-1.5B and train it on the DeepscaleR-40k in the math domain. Experiments demonstrate our method achieves SoTA performance on several math benchmarks, representing significant improvements and fewer reasoning tokens over the base models. Code and model will be released.

**arXiv ID:** 2601.19620
</details>

<details>
<summary><strong>Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning</strong> - Tongxi Wang, Zhuoyang Xia, Xinran Chen, Shan Liu - [[pdf]](https://arxiv.org/pdf/2601.19624)</summary>

**Abstract:** Real-world reinforcement learning often faces environment drift, but most existing methods rely on static entropy coefficients/target entropy, causing over-exploration during stable periods and under-exploration after drift (thus slow recovery), and leaving unanswered the principled question of how exploration intensity should scale with drift magnitude. We prove that entropy scheduling under non-stationarity can be reduced to a one-dimensional, round-by-round trade-off, faster tracking of the optimal solution after drift vs. avoiding gratuitous randomness when the environment is stable, so exploration strength can be driven by measurable online drift signals. Building on this, we propose AES (Adaptive Entropy Scheduling), which adaptively adjusts the entropy coefficient/temperature online using observable drift proxies during training, requiring almost no structural changes and incurring minimal overhead. Across 4 algorithm variants, 12 tasks, and 4 drift modes, AES significantly reduces the fraction of performance degradation caused by drift and accelerates recovery after abrupt changes.

**arXiv ID:** 2601.19624
</details>

<details>
<summary><strong>RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning</strong> - Qianyue Hao, Sibo Li, Jian Yuan, Yong Li - [[pdf]](https://arxiv.org/pdf/2505.14140)</summary>

**Abstract:** Despite rapid advancements in large language models (LLMs), the token-level autoregressive nature constrains their complex reasoning capabilities. To enhance LLM reasoning, inference-time techniques, including Chain/Tree/Graph-of-Thought(s), successfully improve the performance, as they are fairly cost-effective by guiding reasoning through sophisticated logical structures without modifying LLMs' parameters. However, these manually predefined, task-agnostic frameworks are applied uniformly across diverse tasks, lacking adaptability. To improve this, we propose RL-of-Thoughts (RLoT), where we train a lightweight navigator model with reinforcement learning (RL) to adaptively enhance LLM reasoning at inference time. Specifically, we design five basic logic blocks from the perspective of human cognition. During the reasoning process, the trained RL navigator dynamically selects the suitable logic blocks and combines them into task-specific logical structures according to problem characteristics. Experiments across multiple reasoning benchmarks (AIME, MATH, GPQA, etc.) with multiple LLMs (GPT, Llama, Qwen, and DeepSeek) illustrate that RLoT outperforms established inference-time techniques by up to 13.4%. Remarkably, with less than 3K parameters, our RL navigator is able to make sub-10B LLMs comparable to 100B-scale counterparts. Moreover, the RL navigator demonstrates strong transferability: a model trained on one specific LLM-task pair can effectively generalize to unseen LLMs and tasks. Our code is open-source at this https URL for reproducibility.

**arXiv ID:** 2505.14140
</details>

<details>
<summary><strong>Plan Then Retrieve: Reinforcement Learning-Guided Complex Reasoning over Knowledge Graphs</strong> - Yanlin Song, Ben Liu, Víctor Gutiérrez-Basulto, Zhiwei Hu, Qianqian Xie, Min Peng, Sophia Ananiadou, Jeff Z. Pan - [[pdf]](https://arxiv.org/pdf/2510.20691)</summary>

**Abstract:** Knowledge Graph Question Answering aims to answer natural language questions by reasoning over structured knowledge graphs. While large language models have advanced KGQA through their strong reasoning capabilities, existing methods continue to struggle to fully exploit both the rich knowledge encoded in KGs and the reasoning capabilities of LLMs, particularly in complex scenarios. They often assume complete KG coverage and lack mechanisms to judge when external information is needed, and their reasoning remains locally myopic, failing to maintain coherent multi-step planning, leading to reasoning failures even when relevant knowledge exists. We propose Graph-RFT, a novel two-stage reinforcement fine-tuning KGQA framework with a 'plan-KGsearch-and-Websearch-during-think' paradigm, that enables LLMs to perform autonomous planning and adaptive retrieval scheduling across KG and web sources under incomplete knowledge conditions. Graph-RFT introduces a chain-of-thought fine-tuning method with a customized plan-retrieval dataset activates structured reasoning and resolves the GRPO cold-start problem. It then introduces a novel plan-retrieval guided reinforcement learning process integrates explicit planning and retrieval actions with a multi-reward design, enabling coverage-aware retrieval scheduling. It employs a Cartesian-inspired planning module to decompose complex questions into ordered subquestions, and logical expression to guide tool invocation for globally consistent multi-step reasoning. This reasoning retrieval process is optimized with a multi-reward combining outcome and retrieval specific signals, enabling the model to learn when and how to combine KG and web retrieval effectively.

**arXiv ID:** 2510.20691
</details>

<details>
<summary><strong>AtomMem : Learnable Dynamic Agentic Memory with Atomic Memory Operation</strong> - Yupeng Huo, Yaxi Lu, Zhong Zhang, Haotian Chen, Yankai Lin - [[pdf]](https://arxiv.org/pdf/2601.08323)</summary>

**Abstract:** Equipping agents with memory is essential for solving real-world long-horizon problems. However, most existing agent memory mechanisms rely on static and hand-crafted workflows. This limits the performance and generalization ability of these memory designs, which highlights the need for a more flexible, learning-based memory framework. In this paper, we propose AtomMem, which reframes memory management as a dynamic decision-making problem. We deconstruct high-level memory processes into fundamental atomic CRUD (Create, Read, Update, Delete) operations, transforming the memory workflow into a learnable decision process. By combining supervised fine-tuning with reinforcement learning, AtomMem learns an autonomous, task-aligned policy to orchestrate memory behaviors tailored to specific task demands. Experimental results across 3 long-context benchmarks demonstrate that the trained AtomMem-8B consistently outperforms prior static-workflow memory methods. Further analysis of training dynamics shows that our learning-based formulation enables the agent to discover structured, task-aligned memory management strategies, highlighting a key advantage over predefined routines.

**arXiv ID:** 2601.08323
</details>

<details>
<summary><strong>Bayes Adaptive Monte Carlo Tree Search for Offline Model-based Reinforcement Learning</strong> - Jiayu Chen, Le Xu, Wentse Chen, Jeff Schneider - [[pdf]](https://arxiv.org/pdf/2410.11234)</summary>

**Abstract:** Offline reinforcement learning (RL) is a powerful approach for data-driven decision-making and control. Compared to model-free methods, offline model-based reinforcement learning (MBRL) explicitly learns world models from a static dataset and uses them as surrogate simulators, improving the data efficiency and enabling the learned policy to potentially generalize beyond the dataset support. However, there could be various MDPs that behave identically on the offline dataset and dealing with the uncertainty about the true MDP can be challenging. In this paper, we propose modeling offline MBRL as a Bayes Adaptive Markov Decision Process (BAMDP), which is a principled framework for addressing model uncertainty. We further propose a novel Bayes Adaptive Monte-Carlo planning algorithm capable of solving BAMDPs in continuous state and action spaces with stochastic transitions. This planning process is based on Monte Carlo Tree Search and can be integrated into offline MBRL as a policy improvement operator in policy iteration. Our "RL + Search" framework follows in the footsteps of superhuman AIs like AlphaZero, improving on current offline MBRL methods by incorporating more computation input. The proposed algorithm significantly outperforms state-of-the-art offline RL methods on twelve D4RL MuJoCo tasks and three challenging, stochastic tokamak control tasks. The codebase is available at: this https URL.

**arXiv ID:** 2410.11234
</details>

<details>
<summary><strong>Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models</strong> - David Bani-Harouni, Chantal Pellegrini, Paul Stangel, Ege Özsoy, Kamilia Zaripova, Matthias Keicher, Nassir Navab - [[pdf]](https://arxiv.org/pdf/2503.02623)</summary>

**Abstract:** A safe and trustworthy use of Large Language Models (LLMs) requires an accurate expression of confidence in their answers. We propose a novel Reinforcement Learning approach that allows to directly fine-tune LLMs to express calibrated confidence estimates alongside their answers to factual questions. Our method optimizes a reward based on the logarithmic scoring rule, explicitly penalizing both over- and under-confidence. This encourages the model to align its confidence estimates with the actual predictive accuracy. The optimal policy under our reward design would result in perfectly calibrated confidence expressions. Unlike prior approaches that decouple confidence estimation from response generation, our method integrates confidence calibration seamlessly into the generative process of the LLM. Empirically, we demonstrate that models trained with our approach exhibit substantially improved calibration and generalize to unseen tasks without further fine-tuning, suggesting the emergence of general confidence awareness.

**arXiv ID:** 2503.02623
</details>

<details>
<summary><strong>Text2Grad: Reinforcement Learning from Natural Language Feedback</strong> - Hanyang Wang, Lu Wang, Chaoyun Zhang, Tianjun Mao, Si Qin, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang - [[pdf]](https://arxiv.org/pdf/2505.22338)</summary>

**Abstract:** Traditional RLHF optimizes language models with coarse, scalar rewards that mask the fine-grained reasons behind success or failure, leading to slow and opaque learning. Recent work augments RL with textual critiques through prompting or reflection, improving interpretability but leaving model parameters untouched. We introduce Text2Grad, a reinforcement-learning paradigm that turns free-form textual feedback into span-level gradients. Given human (or programmatic) critiques, Text2Grad aligns each feedback phrase with the relevant token spans, converts these alignments into differentiable reward signals, and performs gradient updates that directly refine the offending portions of the model's policy. This yields precise, feedback-conditioned adjustments instead of global nudges. Text2Grad is realized through three components: (1) a high-quality feedback-annotation pipeline that pairs critiques with token spans; (2) a fine-grained reward model that predicts span-level reward on answers while generating explanatory critiques; and (3) a span-level policy optimizer that back-propagates natural-language gradients. Across summarization, code generation, and question answering, Text2Grad consistently surpasses scalar-reward RL and prompt-only baselines, providing both higher task metrics and richer interpretability. Our results suggest that natural-language feedback can serve not only as explanations, but also as actionable training signals for fine-grained alignment. The code for our method is available at this https URL.

**arXiv ID:** 2505.22338
</details>

<details>
<summary><strong>Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning</strong> - David Bani-Harouni, Chantal Pellegrini, Ege Özsoy, Matthias Keicher, Nassir Navab - [[pdf]](https://arxiv.org/pdf/2506.13474)</summary>

**Abstract:** Clinical decision-making is a dynamic, interactive, and cyclic process where doctors have to repeatedly decide on which clinical action to perform and consider newly uncovered information for diagnosis and treatment. Large Language Models (LLMs) have the potential to support clinicians in this process, however, most applications of LLMs in clinical decision support suffer from one of two limitations: Either they assume the unrealistic scenario of immediate availability of all patient information and do not model the interactive and iterative investigation process, or they restrict themselves to the limited "out-of-the-box" capabilities of large pre-trained models without performing task-specific training. In contrast to this, we propose to model clinical decision-making for diagnosis with a hypothesis-driven uncertainty-aware language agent, LA-CDM, that converges towards a diagnosis via repeatedly requesting and interpreting relevant tests. Using a hybrid training paradigm combining supervised and reinforcement learning, we train LA-CDM with three objectives targeting critical aspects of clinical decision-making: accurate hypothesis generation, hypothesis uncertainty estimation, and efficient decision-making. We evaluate our methodology on MIMIC-CDM, a real-world dataset covering four abdominal diseases containing various clinical tests and show the benefit of explicitly training clinical decision-making for increasing diagnostic performance and efficiency.

**arXiv ID:** 2506.13474
</details>

<details>
<summary><strong>Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs</strong> - Honglin Zhang, Qianyue Hao, Fengli Xu, Yong Li - [[pdf]](https://arxiv.org/pdf/2509.21044)</summary>

**Abstract:** Large language models (LLMs) acquire extensive prior knowledge through large-scale pretraining and can be further enhanced via supervised fine-tuning (SFT) or reinforcement learning (RL)-based post-training. A growing body of evidence has shown that RL fine-tuning improves the capability of LLMs beyond what SFT alone achieves. However, the underlying mechanisms why RL fine-tuning is able to enhance the capability of various LLMs with distinct intrinsic characteristics remain underexplored. In this study, we draw inspiration from prior work on edge attribution patching (EAP) to investigate the internal differences of LLMs before and after RL fine-tuning. Our analysis across multiple model families and mathematical datasets shows two robust effects of online RL post-training: (i) an overall increase in average activation intensity, indicating that more internal pathways are engaged and their signals become stronger, and (ii) greater diversity in activation patterns, reflected by higher entropy and less concentrated edge distributions. These changes suggest that RL reshapes information flow to be both more redundant and more flexible, which may explain its advantage in mathematical generalization. Notably, models fine-tuned with Direct Preference Optimization (DPO) deviate from these trends, exhibiting substantially weaker or inconsistent internal changes compared to PPO- and GRPO-based training. Together, our findings provide a unified view of how RL fine-tuning systematically alters the internal circuitry of LLMs and highlight the methodological distinctions between online RL and preference-based approaches. Our code is open source at this https URL.

**arXiv ID:** 2509.21044
</details>

<details>
<summary><strong>Optimizing Conversational Quality in Spoken Dialogue Systems with Reinforcement Learning from AI Feedback</strong> - Siddhant Arora, Jinchuan Tian, Jiatong Shi, Hayato Futami, Yosuke Kashiwagi, Emiru Tsunoo, Shinji Watanabe - [[pdf]](https://arxiv.org/pdf/2601.19063)</summary>

**Abstract:** Reinforcement learning from human or AI feedback (RLHF/RLAIF) for speech-in/speech-out dialogue systems (SDS) remains underexplored, with prior work largely limited to single semantic rewards applied at the utterance level. Such setups overlook the multi-dimensional and multi-modal nature of conversational quality, which encompasses semantic coherence, audio naturalness, speaker consistency, emotion alignment, and turn-taking behavior. Moreover, they are fundamentally mismatched with duplex spoken dialogue systems that generate responses incrementally, where agents must make decisions based on partial utterances. We address these limitations with the first multi-reward RLAIF framework for SDS, combining semantic, audio-quality, and emotion-consistency rewards. To align utterance-level preferences with incremental, blockwise decoding in duplex models, we apply turn-level preference sampling and aggregate per-block log-probabilities within a single DPO objective. We present the first systematic study of preference learning for improving SDS quality in both multi-turn Chain-of-Thought and blockwise duplex models, and release a multi-reward DPO dataset to support reproducible research. Experiments show that single-reward RLAIF selectively improves its targeted metric, while joint multi-reward training yields consistent gains across semantic quality and audio naturalness. These results highlight the importance of holistic, multi-reward alignment for practical conversational SDS.

**arXiv ID:** 2601.19063
</details>

<details>
<summary><strong>ALRM: Agentic LLM for Robotic Manipulation</strong> - Vitor Gaboardi dos Santos, Ibrahim Khadraoui, Ibrahim Farhat, Hamza Yous, Samy Teffahi, Hakim Hacid - [[pdf]](https://arxiv.org/pdf/2601.19510)</summary>

**Abstract:** Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP.

**arXiv ID:** 2601.19510
</details>

<details>
<summary><strong>Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning</strong> - Jinyeop Song, Song Wang, Julian Shun, Yada Zhu - [[pdf]](https://arxiv.org/pdf/2509.26383)</summary>

**Abstract:** Knowledge-graph retrieval-augmented generation (KG-RAG) couples large language models (LLMs) with structured, verifiable knowledge graphs (KGs) to reduce hallucinations and expose reasoning traces. However, many KG-RAG systems compose multiple LLM modules (e.g planning, reasoning, and responding), inflating inference cost and binding behavior to a specific target KG. To address this, we introduce KG-R1, an agentic KG retrieval-augmented generation (KG-RAG) framework through reinforcement learning (RL). KG-R1 utilizes a single agent that interacts with KGs as its environment, learning to retrieve at each step and incorporating the retrieved information into its reasoning and generation. The process is optimized through end-to-end RL. In controlled experiments across Knowledge-Graph Question Answering (KGQA) benchmarks, our method demonstrates both efficiency and transferability: Using Qwen-2.5-3B, KG-R1 improves answer accuracy with fewer generation tokens than prior multi-module workflow methods that use larger foundation or fine-tuned models. Furthermore, KG-R1 enables plug and play: after training, it maintains strong accuracy on new KGs without modification. These properties make KG-R1 a promising KG-RAG framework for real-world deployment. Our code is publicly available at this https URL.

**arXiv ID:** 2509.26383
</details>

<details>
<summary><strong>Coupled Variational Reinforcement Learning for Language Model General Reasoning</strong> - Xueru Wen, Jie Lou, Yanjiang Liu, Hongyu Lin, Ben He, Xianpei Han, Le Sun, Yaojie Lu, Debing Zhang - [[pdf]](https://arxiv.org/pdf/2512.12576)</summary>

**Abstract:** While reinforcement learning has achieved impressive progress in language model reasoning, it is constrained by the requirement for verifiable rewards. Recent verifier-free RL methods address this limitation by utilizing the probabilities that LLMs generate reference answers as reward signals. However, these approaches typically sample reasoning traces conditioned only on the question. This design decouples reasoning-trace sampling from answer information, leading to inefficient exploration and incoherence between traces and final answers. In this paper, we propose \textit{\b{Co}upled \b{V}ariational \b{R}einforcement \b{L}earning} (CoVRL), which bridges variational inference and reinforcement learning by coupling prior and posterior distributions through a hybrid sampling strategy. By constructing and optimizing a composite distribution that integrates these two distributions, CoVRL enables efficient exploration while preserving strong thought-answer coherence. Extensive experiments on mathematical and general reasoning benchmarks show that CoVRL improves performance by 12.4\% over the base model and achieves an additional 2.3\% improvement over state-of-the-art verifier-free RL baselines, providing a principled framework for enhancing the general reasoning capabilities of language models.

**arXiv ID:** 2512.12576
</details>

<details>
<summary><strong>ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning</strong> - Ruiyang Zhou, Shuozhe Li, Amy Zhang, Liu Leqi - [[pdf]](https://arxiv.org/pdf/2507.02834)</summary>

**Abstract:** Self-improvement via RL often fails on complex reasoning tasks because GRPO-style post-training methods rely on the model's initial ability to generate positive samples. Without guided exploration, these approaches merely reinforce what the model already knows (distribution-sharpening) rather than enabling the model to solve problems where it initially generates no correct solutions. To unlock reasoning ability in such settings, the model must explore new reasoning trajectories beyond its current output distribution. Such exploration requires access to sufficiently good positive samples to guide the learning. While expert demonstrations seem like a natural solution, we find that they are often ineffective in RL post-training. Instead, we identify two key properties of effective positive samples: they should (1) be likely under the current policy, and (2) increase the model's likelihood of predicting the correct answer. Based on these insights, we propose $\textbf{Self-Explanation Policy Optimization (ExPO)}$-a simple and modular framework that generates such samples by conditioning on the ground-truth answer. It can be integrated with popular RL training methods like GRPO and DPO. ExPO enables efficient exploration and guides the model to produce reasoning trajectories more aligned with its policy than expert-written CoTs, while ensuring higher quality than its own (incorrect) samples. Experiments show that ExPO improves both learning efficiency and final performance on reasoning benchmarks, surpassing expert-demonstration-based methods in challenging settings such as MATH level-5, where the model initially struggles the most. Code is available at this https URL .

**arXiv ID:** 2507.02834
</details>

<details>
<summary><strong>Variational Quantum Circuit-Based Reinforcement Learning for Dynamic Portfolio Optimization</strong> - Vincent Gurgul, Ying Chen, Stefan Lessmann - [[pdf]](https://arxiv.org/pdf/2601.18811)</summary>

**Abstract:** This paper presents a Quantum Reinforcement Learning (QRL) solution to the dynamic portfolio optimization problem based on Variational Quantum Circuits. The implemented QRL approaches are quantum analogues of the classical neural-network-based Deep Deterministic Policy Gradient and Deep Q-Network algorithms. Through an empirical evaluation on real-world financial data, we show that our quantum agents achieve risk-adjusted performance comparable to, and in some cases exceeding, that of classical Deep RL models with several orders of magnitude more parameters. In addition to improved parameter efficiency, quantum agents exhibit reduced variability across market regimes, indicating robust behaviour under changing conditions. However, while quantum circuit execution is inherently fast at the hardware level, practical deployment on cloud-based quantum systems introduces substantial latency, making end-to-end runtime currently dominated by infrastructural overhead and limiting practical applicability. Taken together, our results suggest that QRL is theoretically competitive with state-of-the-art classical reinforcement learning and may become practically advantageous as deployment overheads diminish. This positions QRL as a promising paradigm for dynamic decision-making in complex, high-dimensional, and non-stationary environments such as financial markets. The complete codebase is released as open source at: this https URL

**arXiv ID:** 2601.18811
</details>

<details>
<summary><strong>Vector-Valued Distributional Reinforcement Learning Policy Evaluation: A Hilbert Space Embedding Approach</strong> - Mehrdad Mohammadi, Qi Zheng, Ruoqing Zhu - [[pdf]](https://arxiv.org/pdf/2601.18952)</summary>

**Abstract:** We propose an (offline) multi-dimensional distributional reinforcement learning framework (KE-DRL) that leverages Hilbert space mappings to estimate the kernel mean embedding of the multi-dimensional value distribution under a proposed target policy. In our setting, the state-action variables are multi-dimensional and continuous. By mapping probability measures into a reproducing kernel Hilbert space via kernel mean embeddings, our method replaces Wasserstein metrics with an integral probability metric. This enables efficient estimation in multi-dimensional state-action spaces and reward settings, where direct computation of Wasserstein distances is computationally challenging. Theoretically, we establish contraction properties of the distributional Bellman operator under our proposed metric involving the Matern family of kernels and provide uniform convergence guarantees. Simulations and empirical results demonstrate robust off-policy evaluation and recovery of the kernel mean embedding under mild assumptions, namely, Lipschitz continuity and boundedness of the kernels, highlighting the potential of embedding-based approaches in complex real-world decision-making scenarios and risk evaluation.

**arXiv ID:** 2601.18952
</details>

<details>
<summary><strong>Improving Policy Exploitation in Online Reinforcement Learning with Instant Retrospect Action</strong> - Gong Gao, Weidong Zhao, Xianhui Liu, Ning Jia - [[pdf]](https://arxiv.org/pdf/2601.19720)</summary>

**Abstract:** Existing value-based online reinforcement learning (RL) algorithms suffer from slow policy exploitation due to ineffective exploration and delayed policy updates. To address these challenges, we propose an algorithm called Instant Retrospect Action (IRA). Specifically, we propose Q-Representation Discrepancy Evolution (RDE) to facilitate Q-network representation learning, enabling discriminative representations for neighboring state-action pairs. In addition, we adopt an explicit method to policy constraints by enabling Greedy Action Guidance (GAG). This is achieved through backtracking historical actions, which effectively enhances the policy update process. Our proposed method relies on providing the learning algorithm with accurate $k$-nearest-neighbor action value estimates and learning to design a fast-adaptable policy through policy constraints. We further propose the Instant Policy Update (IPU) mechanism, which enhances policy exploitation by systematically increasing the frequency of policy updates. We further discover that the early-stage training conservatism of the IRA method can alleviate the overestimation bias problem in value-based RL. Experimental results show that IRA can significantly improve the learning efficiency and final performance of online RL algorithms on eight MuJoCo continuous control tasks.

**arXiv ID:** 2601.19720
</details>

<details>
<summary><strong>Risk-Sensitive Agent Compositions</strong> - Guruprerana Shabadi, Rajeev Alur - [[pdf]](https://arxiv.org/pdf/2506.04632)</summary>

**Abstract:** From software development to robot control, modern agentic systems decompose complex objectives into a sequence of subtasks and choose a set of specialized AI agents to complete them. We formalize agentic workflows as directed acyclic graphs, called agent graphs, where edges represent AI agents and paths correspond to feasible compositions of agents. Real-world deployment requires selecting agent compositions that not only maximize task success but also minimize violations of safety, fairness, and privacy requirements which demands a careful analysis of the low-probability (tail) behaviors of compositions of agents. In this work, we consider risk minimization over the set of feasible agent compositions and seek to minimize the value-at-risk and the conditional value-at-risk of the loss distribution of the agent composition where the loss quantifies violations of these requirements. We introduce an efficient algorithm which traverses the agent graph and finds a near-optimal composition of agents. It uses a dynamic programming approach to approximate the value-at-risk of agent compositions by exploiting a union bound. Furthermore, we prove that the approximation is near-optimal asymptotically for a broad class of practical loss functions. We also show how our algorithm can be used to approximate the conditional value-at-risk as a byproduct. To evaluate our framework, we consider a suite of video game-like control benchmarks that require composing several agents trained with reinforcement learning and demonstrate our algorithm's effectiveness in approximating the value-at-risk and identifying the optimal agent composition.

**arXiv ID:** 2506.04632
</details>

<details>
<summary><strong>GraphRAG-R1: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning</strong> - Chuanyue Yu, Kuo Zhao, Yuhan Li, Heng Chang, Mingjian Feng, Xiangzhe Jiang, Yufei Sun, Jia Li, Yuzhi Zhang, Jianxin Li, Ziwei Zhang - [[pdf]](https://arxiv.org/pdf/2507.23581)</summary>

**Abstract:** Graph Retrieval-Augmented Generation (GraphRAG) has shown great effectiveness in enhancing the reasoning abilities of LLMs by leveraging graph structures for knowledge representation and modeling complex real-world relationships. However, existing GraphRAG methods still face significant bottlenecks when handling complex problems that require multi-hop reasoning, as their query and retrieval phases are largely based on pre-defined heuristics and do not fully utilize the reasoning potentials of LLMs. To address this problem, we propose GraphRAG-R1, an adaptive GraphRAG framework by training LLMs with process-constrained outcome-based reinforcement learning (RL) to enhance the multi-hop reasoning ability. Our method can decompose complex problems, autonomously invoke retrieval tools to acquire necessary information, and perform effective reasoning. Specifically, we utilize a modified version of Group Relative Policy Optimization (GRPO) that supports rollout-with-thinking capability. Next, we design two process-constrained reward functions. To handle the shallow retrieval problem, we design a Progressive Retrieval Attenuation (PRA) reward to encourage essential retrievals. Then, to handle the over-thinking problem, we design Cost-Aware F1 (CAF) reward to balance the model performance with computational costs. We further design a phase-dependent training strategy, containing three training stages corresponding to cold start and these two rewards. Lastly, our method adopts a hybrid graph-textual retrieval to improve the reasoning capacity. Extensive experimental results demonstrate that GraphRAG-R1 boosts LLM capabilities in solving complex reasoning problems compared to state-of-the-art GraphRAG methods on both in-domain and out-of-domain datasets. Furthermore, our framework can be flexibly integrated with various existing retrieval methods, consistently delivering performance improvements.

**arXiv ID:** 2507.23581
</details>

<details>
<summary><strong>Reinforcement Learning to Discover a NorthEast Monsoon Index for Monthly Rainfall Prediction in Thailand</strong> - Kiattikun Chobtham - [[pdf]](https://arxiv.org/pdf/2601.10181)</summary>

**Abstract:** Climate prediction is a challenge due to the intricate spatiotemporal patterns within Earth systems. Global climate indices, such as the El Niño Southern Oscillation, are standard input features for long-term rainfall prediction. However, a significant gap persists regarding local-scale indices capable of improving predictive accuracy in specific regions of Thailand. This paper introduces a novel NorthEast monsoon climate index calculated from sea surface temperature to reflect the climatology of the boreal winter monsoon. To optimise the calculated areas used for this index, a Deep Q-Network reinforcement learning agent explores and selects the most effective rectangles based on their correlation with seasonal rainfall. Rainfall stations were classified into 12 distinct clusters to distinguish rainfall patterns between southern and upper Thailand. Experimental results show that incorporating the optimised index into Long Short-Term Memory models significantly improves long-term monthly rainfall prediction skill in most cluster areas. This approach effectively reduces the Root Mean Square Error for 12-month-ahead forecasts.

**arXiv ID:** 2601.10181
</details>

<details>
<summary><strong>Reinforcement Learning Goal-Reaching Control with Guaranteed Lyapunov-Like Stabilizer for Mobile Robots</strong> - Mehdi Heydari Shahna, Seyed Adel Alizadeh Kolagar, Jouni Mattila - [[pdf]](https://arxiv.org/pdf/2601.19499)</summary>

**Abstract:** Reinforcement learning (RL) can be highly effective at learning goal-reaching policies, but it typically does not provide formal guarantees that the goal will always be reached. A common approach to provide formal goal-reaching guarantees is to introduce a shielding mechanism that restricts the agent to actions that satisfy predefined safety constraints. The main challenge here is integrating this mechanism with RL so that learning and exploration remain effective without becoming overly conservative. Hence, this paper proposes an RL-based control framework that provides formal goal-reaching guarantees for wheeled mobile robots operating in unstructured environments. We first design a real-time RL policy with a set of 15 carefully defined reward terms. These rewards encourage the robot to reach both static and dynamic goals while generating sufficiently smooth command signals that comply with predefined safety specifications, which is critical in practice. Second, a Lyapunov-like stabilizer layer is integrated into the benchmark RL framework as a policy supervisor to formally strengthen the goal-reaching control while preserving meaningful exploration of the state action space. The proposed framework is suitable for real-time deployment in challenging environments, as it provides a formal guarantee of convergence to the intended goal states and compensates for uncertainties by generating real-time control signals based on the current state, while respecting real-world motion constraints. The experimental results show that the proposed Lyapunov-like stabilizer consistently improves the benchmark RL policies, boosting the goal-reaching rate from 84.6% to 99.0%, sharply reducing failures, and improving efficiency.

**arXiv ID:** 2601.19499
</details>

<details>
<summary><strong>xFLIE: Leveraging Actionable Hierarchical Scene Representations for Autonomous Semantic-Aware Inspection Missions</strong> - Vignesh Kottayam Viswanathan, Mario A.V. Saucedo, Sumeet Gajanan Satpute, Christoforos Kanellakis, George Nikolakopoulos - [[pdf]](https://arxiv.org/pdf/2412.19571)</summary>

**Abstract:** We present a novel architecture aimed towards incremental construction and exploitation of a hierarchical 3D scene graph representation during semantic-aware inspection missions. Inspection planning, particularly of distributed targets in previously unseen environments, presents an opportunity to exploit the semantic structure of the scene during reasoning, navigation and scene understanding. Motivated by this, we propose the 3D Layered Semantic Graph (3DLSG), a hierarchical inspection scene graph constructed in an incremental manner and organized into abstraction layers that support planning demands in real-time. To address the task of semantic-aware inspection, a mission framework, termed as Enhanced First-Look Inspect Explore (xFLIE), that tightly couples the 3DLSG with an inspection planner is proposed. We assess the performance through simulations and experimental trials, evaluating target-selection, path-planning and semantic navigation tasks over the 3DLSG model. The scenarios presented are diverse, ranging from city-scale distributed to solitary infrastructure targets in simulated worlds and subsequent outdoor and subterranean environment deployments onboard a quadrupedal robot. The proposed method successfully demonstrates incremental construction and planning over the 3DLSG representation to meet the objectives of the missions. Furthermore, the framework demonstrates successful semantic navigation tasks over the structured interface at the end of the inspection missions. Finally, we report multiple orders of magnitude reduction in path-planning time compared to conventional volumetric-map-based methods over various environment scale, demonstrating the planning efficiency and scalability of the proposed approach.

**arXiv ID:** 2412.19571
</details>

<details>
<summary><strong>GTA: Generative Traffic Agents for Simulating Realistic Mobility Behavior</strong> - Simon Lämmer, Mark Colley, Patrick Ebel - [[pdf]](https://arxiv.org/pdf/2601.16778)</summary>

**Abstract:** People's transportation choices reflect complex trade-offs shaped by personal preferences, social norms, and technology acceptance. Predicting such behavior at scale is a critical challenge with major implications for urban planning and sustainable transport. Traditional methods use handcrafted assumptions and costly data collection, making them impractical for early-stage evaluations of new technologies or policies. We introduce Generative Traffic Agents (GTA) for simulating large-scale, context-sensitive transportation choices using LLM-powered, persona-based agents. GTA generates artificial populations from census-based sociodemographic data. It simulates activity schedules and mode choices, enabling scalable, human-like simulations without handcrafted rules. We evaluate GTA in Berlin-scale experiments, comparing simulation results against empirical data. While agents replicate patterns, such as modal split by socioeconomic status, they show systematic biases in trip length and mode preference. GTA offers new opportunities for modeling how future innovations, from bike lanes to transit apps, shape mobility decisions.

**arXiv ID:** 2601.16778
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
