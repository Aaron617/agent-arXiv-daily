# Agent arXiv Daily

**Last Updated:** 2025-12-17 02:56:59

**Total Papers:** 53

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
<summary><strong>Enhancing Long-term RAG Chatbots with Psychological Models of Memory Importance and Forgetting</strong> - Ryuichi Sumida, Koji Inoue, Tatsuya Kawahara - [[pdf]](https://arxiv.org/pdf/2409.12524)</summary>

**Abstract:** While Retrieval-Augmented Generation (RAG) has shown promise in enhancing long-term conversations, the increasing memory load as conversations progress degrades retrieval accuracy. Drawing on psychological insights, we propose LUFY, a simple yet effective method that focuses on emotionally arousing memories and retains less than 10% of the conversation. In the user experiment, participants interacted with three types of RAG chatbots, each for 2 hours over 4 sessions, marking the most extensive assessment of a chatbot's long-term capabilities to date -- more than four times longer than any existing benchmark. The results demonstrate that prioritizing arousing memories while forgetting the majority of the conversation significantly enhances user experience. This study pushes the frontier of long-term conversations and highlights the importance of forgetting unimportant parts of conversations. Code and Dataset: this https URL

**arXiv ID:** 2409.12524
</details>

<details>
<summary><strong>A Comprehensive Safety Metric to Evaluate Perception in Autonomous Systems</strong> - Georg Volk, Jörg Gamerdinger, Alexander von Bernuth, Oliver Bringmann - [[pdf]](https://arxiv.org/pdf/2512.14367)</summary>

**Abstract:** Complete perception of the environment and its correct interpretation is crucial for autonomous vehicles. Object perception is the main component of automotive surround sensing. Various metrics already exist for the evaluation of object perception. However, objects can be of different importance depending on their velocity, orientation, distance, size, or the potential damage that could be caused by a collision due to a missed detection. Thus, these additional parameters have to be considered for safety evaluation. We propose a new safety metric that incorporates all these parameters and returns a single easily interpretable safety assessment score for object perception. This new metric is evaluated with both real world and virtual data sets and compared to state of the art metrics.

**arXiv ID:** 2512.14367
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (6 papers)</h2></summary>

<details>
<summary><strong>MobileWorldBench: Towards Semantic World Modeling For Mobile Agents</strong> - Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Aditya Grover - [[pdf]](https://arxiv.org/pdf/2512.14014)</summary>

**Abstract:** World models have shown great utility in improving the task performance of embodied agents. While prior work largely focuses on pixel-space world models, these approaches face practical limitations in GUI settings, where predicting complex visual elements in future states is often difficult. In this work, we explore an alternative formulation of world modeling for GUI agents, where state transitions are described in natural language rather than predicting raw pixels. First, we introduce MobileWorldBench, a benchmark that evaluates the ability of vision-language models (VLMs) to function as world models for mobile GUI agents. Second, we release MobileWorld, a large-scale dataset consisting of 1.4M samples, that significantly improves the world modeling capabilities of VLMs. Finally, we propose a novel framework that integrates VLM world models into the planning framework of mobile agents, demonstrating that semantic world models can directly benefit mobile agents by improving task success rates. The code and dataset is available at this https URL

**arXiv ID:** 2512.14014
</details>

<details>
<summary><strong>MCTS-EP: Empowering Embodied Planning with Online Preference Optimization</strong> - Hang Xu, Zang Yu, Yehui Tang, Pengbo Hu, Yuhao Tang, Hao Dong - [[pdf]](https://arxiv.org/pdf/2509.17116)</summary>

**Abstract:** This paper introduces MCTS-EP, an online learning framework that combines large language models (LLM) with Monte Carlo Tree Search (MCTS) for training embodied agents. MCTS-EP integrates three key components: MCTS-guided exploration for preference data collection, efficient multi-modal reasoning mechanism, and iterative training pipeline based on preference optimization. We theoretically prove that MCTS-EP achieves better performance bounds than conventional on-policy algorithms when the loss function is strongly convex, and demonstrate that it can be formulated as a search-enhanced variant of GAIL. MCTS-EP achieves state-of-the-art performace across serval benchmarks. In ALFWorld, it achieves 92% and 87% success rates for textual and visual tasks. In WebShop, it reaches an average reward of 0.81. MTCS-EP also reduces average interaction steps from from 18.7/19.5 to 10.2/9.9 steps in visual this http URL available at: this https URL

**arXiv ID:** 2509.17116
</details>

<details>
<summary><strong>Echo-CoPilot: A Multi-View, Multi-Task Agent for Echocardiography Interpretation and Reporting</strong> - Moein Heidari, Mohammad Amin Roohi, Ilker Hacihaliloglu - [[pdf]](https://arxiv.org/pdf/2512.09944)</summary>

**Abstract:** Echocardiography is central to contemporary cardiovascular care, but full-study interpretation remains a cognitively demanding, multi-view task that is still performed manually. While recent foundation models for echocardiography can achieve strong performance on individual perceptual subtasks such as view classification, segmentation, or disease prediction, they typically operate in isolation and do not provide a unified, clinically coherent assessment. In this work, we introduce Echo-CoPilot, a multi-view, multi-task agent that uses a large language model to orchestrate a suite of specialized echocardiography tools. Within a ReAct-style loop, the agent decomposes clinician queries, invokes tools for view recognition, cardiac structure segmentation, measurement and disease prediction, and report synthesis, and integrates their outputs into guideline-aware answers and narrative summaries. We evaluate Echo-CoPilot on the public MIMIC-EchoQA benchmark, where it achieves an accuracy of 50.8\%, outperforming both general-purpose and biomedical video vision-language models. Qualitative analyses further show that the agent leverages quantitative measurements and physiologic context to resolve challenging cases near clinical decision thresholds, such as borderline left ventricular hypertrophy or pericardial effusion severity. The code will be released upon acceptance of the paper.

**arXiv ID:** 2512.09944
</details>

<details>
<summary><strong>Confucius Code Agent: Scalable Agent Scaffolding for Real-World Codebases</strong> - Zhaodong Wang, Zhenting Qi, Sherman Wong, Nathan Hu, Samuel Lin, Jun Ge, Erwin Gao, Wenlin Chen, Yilun Du, Minlan Yu, Ying Zhang - [[pdf]](https://arxiv.org/pdf/2512.10398)</summary>

**Abstract:** Real-world software engineering tasks require coding agents that can operate over massive repositories, sustain long-horizon sessions, and reliably coordinate complex toolchains at test time. Existing research-grade agents offer transparency but struggle when scaled to real-world workloads, while proprietary systems achieve strong practical performance but provide limited extensibility, interpretability, and controllability. We introduce the Confucius Code Agent (CCA), a scalable software engineering agent that can operate at enterprise-level codebases. CCA is built on top of the Confucius SDK, an agent development platform structured around three complementary perspectives: Agent Experience (AX), User Experience (UX), and Developer Experience (DX). The SDK integrates a unified orchestrator with hierarchical working memory for long-context operation, a persistent note-taking mechanism for cross-session continual learning, and a modular extension system for reliable tool use. In addition, we introduce a meta-agent that automates the synthesis, evaluation, and refinement of agent configurations through a build-test-improve loop, enabling rapid adaptation to new tasks, environments, and tool stacks. Instantiated with these mechanisms, CCA demonstrates strong performance on real-world software engineering tasks. On SWE-Bench-Pro, CCA achieves a Resolve@1 of 54.3%, surpassing both research-grade and proprietary coding agents under comparable model conditions. Together, the Confucius SDK and CCA form a general, extensible, and production-grade foundation for building robust coding agents, bridging the gap between research prototypes and practical large-scale deployment.

**arXiv ID:** 2512.10398
</details>

<details>
<summary><strong>Autonomous Construction-Site Safety Inspection Using Mobile Robots: A Multilayer VLM-LLM Pipeline</strong> - Hossein Naderi, Alireza Shojaei, Philip Agee, Kereshmeh Afsari, Abiola Akanmu - [[pdf]](https://arxiv.org/pdf/2512.13974)</summary>

**Abstract:** Construction safety inspection remains mostly manual, and automated approaches still rely on task-specific datasets that are hard to maintain in fast-changing construction environments due to frequent retraining. Meanwhile, field inspection with robots still depends on human teleoperation and manual reporting, which are labor-intensive. This paper aims to connect what a robot sees during autonomous navigation to the safety rules that are common in construction sites, automatically generating a safety inspection report. To this end, we proposed a multi-layer framework with two main modules: robotics and AI. On the robotics side, SLAM and autonomous navigation provide repeatable coverage and targeted revisits via waypoints. On AI side, a Vision Language Model (VLM)-based layer produces scene descriptions; a retrieval component powered grounds those descriptions in OSHA and site policies; Another VLM-based layer assesses the safety situation based on rules; and finally Large Language Model (LLM) layer generates safety reports based on previous outputs. The framework is validated with a proof-of-concept implementation and evaluated in a lab environment that simulates common hazards across three scenarios. Results show high recall with competitive precision compared to state-of-the-art closed-source models. This paper contributes a transparent, generalizable pipeline that moves beyond black-box models by exposing intermediate artifacts from each layer and keeping the human in the loop. This work provides a foundation for future extensions to additional tasks and settings within and beyond construction context.

**arXiv ID:** 2512.13974
</details>

<details>
<summary><strong>A4-Agent: An Agentic Framework for Zero-Shot Affordance Reasoning</strong> - Zixin Zhang, Kanghao Chen, Hanqing Wang, Hongfei Zhang, Harold Haodong Chen, Chenfei Liao, Litao Guo, Ying-Cong Chen - [[pdf]](https://arxiv.org/pdf/2512.14442)</summary>

**Abstract:** Affordance prediction, which identifies interaction regions on objects based on language instructions, is critical for embodied AI. Prevailing end-to-end models couple high-level reasoning and low-level grounding into a single monolithic pipeline and rely on training over annotated datasets, which leads to poor generalization on novel objects and unseen environments. In this paper, we move beyond this paradigm by proposing A4-Agent, a training-free agentic framework that decouples affordance prediction into a three-stage pipeline. Our framework coordinates specialized foundation models at test time: (1) a $\textbf{Dreamer}$ that employs generative models to visualize $\textit{how}$ an interaction would look; (2) a $\textbf{Thinker}$ that utilizes large vision-language models to decide $\textit{what}$ object part to interact with; and (3) a $\textbf{Spotter}$ that orchestrates vision foundation models to precisely locate $\textit{where}$ the interaction area is. By leveraging the complementary strengths of pre-trained models without any task-specific fine-tuning, our zero-shot framework significantly outperforms state-of-the-art supervised methods across multiple benchmarks and demonstrates robust generalization to real-world settings.

**arXiv ID:** 2512.14442
</details>

</details>

<details open>
<summary><h2>LLM Agents (2 papers)</h2></summary>

<details>
<summary><strong>Model-First Reasoning LLM Agents: Reducing Hallucinations through Explicit Problem Modeling</strong> - Annu Rana, Gaurav Kumar - [[pdf]](https://arxiv.org/pdf/2512.14474)</summary>

**Abstract:** Large Language Models (LLMs) often struggle with complex multi-step planning tasks, showing high rates of constraint violations and inconsistent solutions. Existing strategies such as Chain-of-Thought and ReAct rely on implicit state tracking and lack an explicit problem representation. Inspired by classical AI planning, we propose Model-First Reasoning (MFR), a two-phase paradigm in which the LLM first constructs an explicit model of the problem, defining entities, state variables, actions, and constraints, before generating a solution plan. Across multiple planning domains, including medical scheduling, route planning, resource allocation, logic puzzles, and procedural synthesis, MFR reduces constraint violations and improves solution quality compared to Chain-of-Thought and ReAct. Ablation studies show that the explicit modeling phase is critical for these gains. Our results suggest that many LLM planning failures stem from representational deficiencies rather than reasoning limitations, highlighting explicit modeling as a key component for robust and interpretable AI agents. All prompts, evaluation procedures, and task datasets are documented to facilitate reproducibility.

**arXiv ID:** 2512.14474
</details>

<details>
<summary><strong>Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space</strong> - Xingfu Zhou, Pengfei Wang - [[pdf]](https://arxiv.org/pdf/2512.14448)</summary>

**Abstract:** Large Language Model (LLM) agents relying on external retrieval are increasingly deployed in high-stakes environments. While existing adversarial attacks primarily focus on content falsification or instruction injection, we identify a novel, process-oriented attack surface: the agent's reasoning style. We propose Reasoning-Style Poisoning (RSP), a paradigm that manipulates how agents process information rather than what they process. We introduce Generative Style Injection (GSI), an attack method that rewrites retrieved documents into pathological tones--specifically "analysis paralysis" or "cognitive haste"--without altering underlying facts or using explicit triggers. To quantify these shifts, we develop the Reasoning Style Vector (RSV), a metric tracking Verification depth, Self-confidence, and Attention focus. Experiments on HotpotQA and FEVER using ReAct, Reflection, and Tree of Thoughts (ToT) architectures reveal that GSI significantly degrades performance. It increases reasoning steps by up to 4.4 times or induces premature errors, successfully bypassing state-of-the-art content filters. Finally, we propose RSP-M, a lightweight runtime monitor that calculates RSV metrics in real-time and triggers alerts when values exceed safety thresholds. Our work demonstrates that reasoning style is a distinct, exploitable vulnerability, necessitating process-level defenses beyond static content analysis.

**arXiv ID:** 2512.14448
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>Adjudicator: Correcting Noisy Labels with a KG-Informed Council of LLM Agents</strong> - Doohee You, Sundeep Paul - [[pdf]](https://arxiv.org/pdf/2512.13704)</summary>

**Abstract:** The performance of production machine learning systems is fundamentally limited by the quality of their training data. In high-stakes industrial applications, noisy labels can degrade performance and erode user trust. This paper presents Adjudicator, a system that addresses the critical data mining challenge of automatically identifying and correcting label noise and has been validated for production deployment. Adjudicator models this as a neuro-symbolic task, first constructing a dynamic Knowledge Graph (KG) to unify item context. This KG then informs a "Council of Agents," a novel multi-agent Large Language Model architecture where specialized agents debate and vote on a label's validity. We validate our system on a 1,000-item balanced subset of the AlleNoise benchmark. Our KG-informed model achieves a 0.99 F1-score, significantly outperforming a single-LLM baseline (0.48 F1) and a non-KG council (0.59 F1). Our analysis reveals this is due to a Precision, achieved by a novel override logic that uses the KG to perfectly identify complex, structural errors (complete Recall) -- a class of errors that baselines fail to find. This result demonstrates a robust and explainable system for automated, high-precision data verification, serving as a vital proof-of-concept for generating golden datasets in strictly governed industrial environments.

**arXiv ID:** 2512.13704
</details>

<details>
<summary><strong>Grammar Search for Multi-Agent Systems</strong> - Mayank Singh, Vikas Yadav, Shiva Krishna Reddy Malay, Shravan Nayak, Sai Rajeswar, Sathwik Tejaswi Madhusudhan, Eduardo Blanco - [[pdf]](https://arxiv.org/pdf/2512.14079)</summary>

**Abstract:** Automatic search for Multi-Agent Systems has recently emerged as a key focus in agentic AI research. Several prior approaches have relied on LLM-based free-form search over the code space. In this work, we propose a more structured framework that explores the same space through a fixed set of simple, composable components. We show that, despite lacking the generative flexibility of LLMs during the candidate generation stage, our method outperforms prior approaches on four out of five benchmarks across two domains: mathematics and question answering. Furthermore, our method offers additional advantages, including a more cost-efficient search process and the generation of modular, interpretable multi-agent systems with simpler logic.

**arXiv ID:** 2512.14079
</details>

<details>
<summary><strong>Hierarchical Multi-agent Large Language Model Reasoning for Autonomous Functional Materials Discovery</strong> - Samuel Rothfarb, Megan C. Davis, Ivana Matanovic, Baikun Li, Edward F. Holby, Wilton J.M. Kort-Kamp - [[pdf]](https://arxiv.org/pdf/2512.13930)</summary>

**Abstract:** Artificial intelligence is reshaping scientific exploration, but most methods automate procedural tasks without engaging in scientific reasoning, limiting autonomy in discovery. We introduce Materials Agents for Simulation and Theory in Electronic-structure Reasoning (MASTER), an active learning framework where large language models autonomously design, execute, and interpret atomistic simulations. In MASTER, a multimodal system translates natural language into density functional theory workflows, while higher-level reasoning agents guide discovery through a hierarchy of strategies, including a single agent baseline and three multi-agent approaches: peer review, triage-ranking, and triage-forms. Across two chemical applications, CO adsorption on Cu-surface transition metal (M) adatoms and on M-N-C catalysts, reasoning-driven exploration reduces required atomistic simulations by up to 90% relative to trial-and-error selection. Reasoning trajectories reveal chemically grounded decisions that cannot be explained by stochastic sampling or semantic bias. Altogether, multi-agent collaboration accelerates materials discovery and marks a new paradigm for autonomous scientific exploration.

**arXiv ID:** 2512.13930
</details>

<details>
<summary><strong>Multi-Agent Collaborative Framework for Intelligent IT Operations: An AOI System with Context-Aware Compression and Dynamic Task Scheduling</strong> - Zishan Bai, Enze Ge, Junfeng Hao - [[pdf]](https://arxiv.org/pdf/2512.13956)</summary>

**Abstract:** The proliferation of cloud-native architectures, characterized by microservices and dynamic orchestration, has rendered modern IT infrastructures exceedingly complex and volatile. This complexity generates overwhelming volumes of operational data, leading to critical bottlenecks in conventional systems: inefficient information processing, poor task coordination, and loss of contextual continuity during fault diagnosis and remediation. To address these challenges, we propose AOI (AI-Oriented Operations), a novel multi-agent collaborative framework that integrates three specialized agents with an LLM-based Context Compressor. Its core innovations include: (1) a dynamic task scheduling strategy that adaptively prioritizes operations based on real-time system states, and (2) a three-layer memory architecture comprising Working, Episodic, and Semantic layers that optimizes context retention and retrieval. Extensive experiments on both synthetic and real-world benchmarks demonstrate that AOI effectively mitigates information overload, achieving a 72.4% context compression ratio while preserving 92.8% of critical information and significantly enhances operational efficiency, attaining a 94.2% task success rate and reducing the Mean Time to Repair (MTTR) by 34.4% compared to the best baseline. This work presents a paradigm shift towards scalable, adaptive, and context-aware autonomous operations, enabling robust management of next-generation IT infrastructures with minimal human intervention.

**arXiv ID:** 2512.13956
</details>

<details>
<summary><strong>COMMA: A Communicative Multimodal Multi-Agent Benchmark</strong> - Timothy Ossowski, Danyal Maqbool, Jixuan Chen, Zefan Cai, Tyler Bradshaw, Junjie Hu - [[pdf]](https://arxiv.org/pdf/2410.07553)</summary>

**Abstract:** The rapid advances of multimodal agents built on large foundation models have largely overlooked their potential for language-based communication between agents in collaborative tasks. This oversight presents a critical gap in understanding their effectiveness in real-world deployments, particularly when communicating with humans. Existing agentic benchmarks fail to address key aspects of inter-agent communication and collaboration, particularly in scenarios where agents have unequal access to information and must work together to achieve tasks beyond the scope of individual capabilities. To fill this gap, we introduce COMMA: a novel puzzle benchmark designed to evaluate the collaborative performance of multimodal multi-agent systems through language communication. Our benchmark features a variety of multimodal puzzles, providing a comprehensive evaluation across four key categories of agentic capability in a communicative collaboration setting. Our findings reveal surprising weaknesses in state-of-the-art models, including strong proprietary models like GPT-4o and reasoning models like o4-mini. Many chain of thought reasoning models such as R1-Onevision and LLaVA-CoT struggle to outperform even a random baseline in agent-agent collaboration, indicating a potential growth area in their communication abilities.

**arXiv ID:** 2410.07553
</details>

<details>
<summary><strong>STEMS: Spatial-Temporal Enhanced Safe Multi-Agent Coordination for Building Energy Management</strong> - Huiliang Zhang, Di Wu, Arnaud Zinflou, Benoit Boulet - [[pdf]](https://arxiv.org/pdf/2510.14112)</summary>

**Abstract:** Building energy management is essential for achieving carbon reduction goals, improving occupant comfort, and reducing energy costs. Coordinated building energy management faces critical challenges in exploiting spatial-temporal dependencies while ensuring operational safety across multi-building systems. Current multi-building energy systems face three key challenges: insufficient spatial-temporal information exploitation, lack of rigorous safety guarantees, and system complexity. This paper proposes Spatial-Temporal Enhanced Safe Multi-Agent Coordination (STEMS), a novel safety-constrained multi-agent reinforcement learning framework for coordinated building energy management. STEMS integrates two core components: (1) a spatial-temporal graph representation learning framework using a GCN-Transformer fusion architecture to capture inter-building relationships and temporal patterns, and (2) a safety-constrained multi-agent RL algorithm incorporating Control Barrier Functions to provide mathematical safety guarantees. Extensive experiments on real-world building datasets demonstrate STEMS's superior performance over existing methods, showing that STEMS achieves 21% cost reduction, 18% emission reduction, and dramatically reduces safety violations from 35.1% to 5.6% while maintaining optimal comfort with only 0.13 discomfort proportion. The framework also demonstrates strong robustness during extreme weather conditions and maintains effectiveness across different building types.

**arXiv ID:** 2510.14112
</details>

<details>
<summary><strong>Single-Agent Scaling Fails Multi-Agent Intelligence: Towards Foundation Models with Native Multi-Agent Intelligence</strong> - Shuyue Hu, Haoyang Yan, Yiqun Zhang, Yang Chen, Dongzhan Zhou, Lei Bai - [[pdf]](https://arxiv.org/pdf/2512.08743)</summary>

**Abstract:** Foundation models (FMs) are increasingly assuming the role of the ''brain'' of AI agents. While recent efforts have begun to equip FMs with native single-agent abilities -- such as GUI interaction or integrated tool use -- we argue that the next frontier is endowing FMs with native multi-agent intelligence. We identify four core capabilities of FMs in multi-agent contexts: understanding, planning, efficient communication, and adaptation. Contrary to assumptions about the spontaneous emergence of such abilities, we provide extensive empirical evidence, across 41 large language models and 7 challenging benchmarks, showing that scaling single-agent performance alone does not automatically yield robust multi-agent intelligence. To address this gap, we outline key research directions -- spanning dataset construction, evaluation, training paradigms, and safety considerations -- for building FMs with native multi-agent intelligence.

**arXiv ID:** 2512.08743
</details>

<details>
<summary><strong>Convergence dynamics of Agent-to-Agent Interactions with Misaligned objectives</strong> - Romain Cosentino, Sarath Shekkizhar, Adam Earle - [[pdf]](https://arxiv.org/pdf/2511.08710)</summary>

**Abstract:** We develop and analyze a theoretical framework for agent-to-agent interactions in a simplified in-context linear regression setting. In our model, each agent is instantiated as a single-layer transformer with linear self-attention (LSA) trained to implement gradient-descent-like updates on a quadratic regression objective from in-context examples. We then study the coupled dynamics when two such LSA agents alternately update from each other's outputs under potentially misaligned fixed objectives. Within this framework, we characterize the generation dynamics and show that misalignment leads to a biased equilibrium where neither agent reaches its target, with residual errors predictable from the objective gap and the prompt-induced geometry. We further contrast this fixed objective regime with an adaptive multi-agent setting, wherein a helper agent updates a turn-based objective to implement a Newton-like step for the main agent, eliminating the plateau and accelerating its convergence. Experiments with trained LSA agents, as well as black-box GPT-5-mini runs on in-context linear regression tasks, are consistent with our theoretical predictions within this simplified setting. We view our framework as a mechanistic framework that links prompt geometry and objective misalignment to stability, bias, and robustness, and as a stepping stone toward analyzing more realistic multi-agent LLM systems.

**arXiv ID:** 2511.08710
</details>

<details>
<summary><strong>Rethinking the Reliability of Multi-agent System: A Perspective from Byzantine Fault Tolerance</strong> - Lifan Zheng, Jiawei Chen, Qinghong Yin, Jingyuan Zhang, Xinyi Zeng, Yu Tian - [[pdf]](https://arxiv.org/pdf/2511.10400)</summary>

**Abstract:** Ensuring the reliability of agent architectures and effectively identifying problematic agents when failures occur are crucial challenges in multi-agent systems (MAS). Advances in large language models (LLMs) have established LLM-based agents as a major branch of MAS, enabling major breakthroughs in complex problem solving and world modeling. However, the reliability implications of this shift remain largely unexplored. i.e., whether substituting traditional agents with LLM-based agents can effectively enhance the reliability of MAS. In this work, we investigate and quantify the reliability of LLM-based agents from the perspective of Byzantine fault tolerance. We observe that LLM-based agents demonstrate stronger skepticism when processing erroneous message flows, a characteristic that enables them to outperform traditional agents across different topological structures. Motivated by the results of the pilot experiment, we design CP-WBFT, a confidence probe-based weighted Byzantine Fault Tolerant consensus mechanism to enhance the stability of MAS with different topologies. It capitalizes on the intrinsic reflective and discriminative capabilities of LLMs by employing a probe-based, weighted information flow transmission method to improve the reliability of LLM-based agents. Extensive experiments demonstrate that CP-WBFT achieves superior performance across diverse network topologies under extreme Byzantine conditions (85.7\% fault rate). Notably, our approach surpasses traditional methods by attaining remarkable accuracy on various topologies and maintaining strong reliability in both mathematical reasoning and safety assessment tasks.

**arXiv ID:** 2511.10400
</details>

<details>
<summary><strong>Training Multi-Image Vision Agents via End2End Reinforcement Learning</strong> - Chengqi Dong, Chuhuai Yue, Hang He, Rongge Mao, Fenghe Tang, S Kevin Zhou, Zekun Xu, Xiaohan Wang, Jiajun Chai, Wei Lin, Guojun Yin - [[pdf]](https://arxiv.org/pdf/2512.08980)</summary>

**Abstract:** Recent VLM-based agents aim to replicate OpenAI O3's ``thinking with images" via tool use, but most open-source methods limit input to a single image, falling short on real-world multi-image QA tasks. To address this, we propose IMAgent, an open-source vision agent trained via end-to-end reinforcement learning dedicated for complex multi-image tasks. By leveraging a multi-agent system, we generate challenging and visually-rich multi-image QA pairs to fully activate the tool-use potential of the base VLM. Through manual verification, we obtain MIFG-QA, comprising 10k samples for training and evaluation. With deeper reasoning steps, VLMs may increasingly ignore visual inputs. We therefore develop two specialized tools for visual reflection and confirmation, allowing the model to proactively reallocate its attention to image content during inference. Benefiting from our well-designed action-trajectory two-level mask strategy, IMAgent achieves stable tool use behavior via pure RL training without requiring costly supervised fine-tuning data. Extensive experiments demonstrate that IMAgent maintains strong performance on existing single-image benchmarks while achieving substantial improvements on our proposed multi-image dataset, with our analysis providing actionable insights for the research community. Codes and data will be released soon.

**arXiv ID:** 2512.08980
</details>

<details>
<summary><strong>Beyond Task Completion: An Assessment Framework for Evaluating Agentic AI Systems</strong> - Sreemaee Akshathala, Bassam Adnan, Mahisha Ramesh, Karthik Vaidhyanathan, Basil Muhammed, Kannan Parthasarathy - [[pdf]](https://arxiv.org/pdf/2512.12791)</summary>

**Abstract:** Recent advances in agentic AI have shifted the focus from standalone Large Language Models (LLMs) to integrated systems that combine LLMs with tools, memory, and other agents to perform complex tasks. These multi-agent architectures enable coordinated reasoning, planning, and execution across diverse domains, allowing agents to collaboratively automate complex workflows. Despite these advances, evaluation and assessment of LLM agents and the multi-agent systems they constitute remain a fundamental challenge. Although various approaches have been proposed in the software engineering literature for evaluating conventional software components, existing methods for AI-based systems often overlook the non-deterministic nature of models. This non-determinism introduces behavioral uncertainty during execution, yet existing evaluations rely on binary task completion metrics that fail to capture it. Evaluating agentic systems therefore requires examining additional dimensions, including the agent ability to invoke tools, ingest and retrieve memory, collaborate with other agents, and interact effectively with its environment. These challenges emerged during our ongoing industry collaboration with MontyCloud Inc., when we deployed an agentic system in production. These limitations surfaced during deployment, highlighting practical gaps in the current evaluation methods and the need for a systematic assessment of agent behavior beyond task outcomes. Informed by these observations and established definitions of agentic systems, we propose an end-to-end Agent Assessment Framework with four evaluation pillars encompassing LLMs, Memory, Tools, and Environment. We validate the framework on a representative Autonomous CloudOps use case, where experiments reveal behavioral deviations overlooked by conventional metrics, demonstrating its effectiveness in capturing runtime uncertainties.

**arXiv ID:** 2512.12791
</details>

<details>
<summary><strong>Multi-Agent Medical Decision Consensus Matrix System: An Intelligent Collaborative Framework for Oncology MDT Consultations</strong> - Xudong Han, Xianglun Gao, Xiaoyi Qu, Zhenyu Yu - [[pdf]](https://arxiv.org/pdf/2512.14321)</summary>

**Abstract:** Multidisciplinary team (MDT) consultations are the gold standard for cancer care decision-making, yet current practice lacks structured mechanisms for quantifying consensus and ensuring decision traceability. We introduce a Multi-Agent Medical Decision Consensus Matrix System that deploys seven specialized large language model agents, including an oncologist, a radiologist, a nurse, a psychologist, a patient advocate, a nutritionist and a rehabilitation therapist, to simulate realistic MDT workflows. The framework incorporates a mathematically grounded consensus matrix that uses Kendall's coefficient of concordance to objectively assess agreement. To further enhance treatment recommendation quality and consensus efficiency, the system integrates reinforcement learning methods, including Q-Learning, PPO and DQN. Evaluation across five medical benchmarks (MedQA, PubMedQA, DDXPlus, MedBullets and SymCat) shows substantial gains over existing approaches, achieving an average accuracy of 87.5% compared with 83.8% for the strongest baseline, a consensus achievement rate of 89.3% and a mean Kendall's W of 0.823. Expert reviewers rated the clinical appropriateness of system outputs at 8.9/10. The system guarantees full evidence traceability through mandatory citations of clinical guidelines and peer-reviewed literature, following GRADE principles. This work advances medical AI by providing structured consensus measurement, role-specialized multi-agent collaboration and evidence-based explainability to improve the quality and efficiency of clinical decision-making.

**arXiv ID:** 2512.14321
</details>

<details>
<summary><strong>Bayesian Ego-graph Inference for Networked Multi-Agent Reinforcement Learning</strong> - Wei Duan, Jie Lu, Junyu Xuan - [[pdf]](https://arxiv.org/pdf/2509.16606)</summary>

**Abstract:** In networked multi-agent reinforcement learning (Networked-MARL), decentralized agents must act under local observability and constrained communication over fixed physical graphs. Existing methods often assume static neighborhoods, limiting adaptability to dynamic or heterogeneous environments. While centralized frameworks can learn dynamic graphs, their reliance on global state access and centralized infrastructure is impractical in real-world decentralized systems. We propose a stochastic graph-based policy for Networked-MARL, where each agent conditions its decision on a sampled subgraph over its local physical neighborhood. Building on this formulation, we introduce BayesG, a decentralized actor-framework that learns sparse, context-aware interaction structures via Bayesian variational inference. Each agent operates over an ego-graph and samples a latent communication mask to guide message passing and policy computation. The variational distribution is trained end-to-end alongside the policy using an evidence lower bound (ELBO) objective, enabling agents to jointly learn both interaction topology and decision-making strategies. BayesG outperforms strong MARL baselines on large-scale traffic control tasks with up to 167 agents, demonstrating superior scalability, efficiency, and performance.

**arXiv ID:** 2509.16606
</details>

<details>
<summary><strong>Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review</strong> - Céline Finet, Stephane Da Silva Martins, Jean-Bernard Hayet, Ioannis Karamouzas, Javad Amirian, Sylvie Le Hégarat-Mascle, Julien Pettré, Emanuel Aldea - [[pdf]](https://arxiv.org/pdf/2506.14831)</summary>

**Abstract:** With the emergence of powerful data-driven methods in human trajectory prediction (HTP), gaining a finer understanding of multi-agent interactions lies within hand's reach, with important implications in areas such as social robot navigation, autonomous navigation, and crowd modeling. This survey reviews some of the most recent advancements in deep learning-based multi-agent trajectory prediction, focusing on studies published between 2020 and 2025. We categorize the existing methods based on their architectural design, their input representations, and their overall prediction strategies, placing a particular emphasis on models evaluated using the ETH/UCY benchmark. Furthermore, we highlight key challenges and future research directions in the field of multi-agent HTP.

**arXiv ID:** 2506.14831
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>PortAgent: LLM-driven Vehicle Dispatching Agent for Port Terminals</strong> - Jia Hu, Junqi Li, Weimeng Lin, Peng Jia, Yuxiong Ji, Jintao Lai - [[pdf]](https://arxiv.org/pdf/2512.14417)</summary>

**Abstract:** Vehicle Dispatching Systems (VDSs) are critical to the operational efficiency of Automated Container Terminals (ACTs). However, their widespread commercialization is hindered due to their low transferability across diverse terminals. This transferability challenge stems from three limitations: high reliance on port operational specialists, a high demand for terminal-specific data, and time-consuming manual deployment processes. Leveraging the emergence of Large Language Models (LLMs), this paper proposes PortAgent, an LLM-driven vehicle dispatching agent that fully automates the VDS transferring workflow. It bears three features: (1) no need for port operations specialists; (2) low need of data; and (3) fast deployment. Specifically, specialist dependency is eliminated by the Virtual Expert Team (VET). The VET collaborates with four virtual experts, including a Knowledge Retriever, Modeler, Coder, and Debugger, to emulate a human expert team for the VDS transferring workflow. These experts specialize in the domain of terminal VDS via a few-shot example learning approach. Through this approach, the experts are able to learn VDS-domain knowledge from a few VDS examples. These examples are retrieved via a Retrieval-Augmented Generation (RAG) mechanism, mitigating the high demand for terminal-specific data. Furthermore, an automatic VDS design workflow is established among these experts to avoid extra manual interventions. In this workflow, a self-correction loop inspired by the LLM Reflexion framework is created

**arXiv ID:** 2512.14417
</details>

<details>
<summary><strong>Intelligent matter consisting of active particles</strong> - Julian Jeggle, Raphael Wittkowski - [[pdf]](https://arxiv.org/pdf/2512.13912)</summary>

**Abstract:** In this book chapter, we review how systems of simple motile agents can be used as a pathway to intelligent systems. It is a well known result from nature that large groups of entities following simple rules, such as swarms of animals, can give rise to much more complex collective behavior in a display of emergence. This begs the question whether we can emulate this behavior in synthetic matter and drive it to a point where the collective behavior reaches the complexity level of intelligent systems. Here, we will use a formalized notion of "intelligent matter" and compare it to recent results in the field of active matter. First, we will explore the approach of emergent computing in which specialized active matter systems are designed to directly solve a given task through emergent behavior. This we will then contrast with the approach of physical reservoir computing powered by the dynamics of active particle systems. In this context, we will also describe a novel reservoir computing scheme for active particles driven ultrasonically or via light refraction.

**arXiv ID:** 2512.13912
</details>

<details>
<summary><strong>Criminal Liability in AI-Enabled Autonomous Vehicles: A Comparative Study</strong> - Sahibpreet Singh, Manjit Singh - [[pdf]](https://arxiv.org/pdf/2512.14330)</summary>

**Abstract:** AI revolutionizes transportation through autonomous vehicles (AVs) but introduces complex criminal liability issues regarding infractions. This study employs a comparative legal analysis of primary statutes, real-world liability claims, and academic literature across the US, Germany, UK, China, and India; jurisdictions selected for their technological advancement and contrasting regulatory approaches. The research examines the attribution of human error, AI moral agency, and the identification of primary offenders in AV incidents. Findings reveal fragmented regulatory landscapes: India and the US rely on loose networks of state laws, whereas the UK enacted the pioneering Automated and Electric Vehicles Act 2018. Germany enforces strict safety standards, distinguishing liability based on the vehicle's operating mode, while China similarly aims for a stringent liability regime. The study concludes that globally harmonized legal standards are essential to foster technological innovation while ensuring minimum risk and clear liability attribution.

**arXiv ID:** 2512.14330
</details>

<details>
<summary><strong>CADDesigner: Conceptual Design of CAD Models Based on General-Purpose Agent</strong> - Fengxiao Fan, Jingzhe Ni, Xiaolong Yin, Sirui Wang, Xingyu Lu, Qiang Zou, Ruofeng Tong, Min Tang, Peng Du - [[pdf]](https://arxiv.org/pdf/2508.01031)</summary>

**Abstract:** Computer Aided Design (CAD) plays a pivotal role in industrial manufacturing but typically requires a high level of expertise from designers. To lower the entry barrier and improve design efficiency, we present an agent for CAD conceptual design powered by large language models (LLMs). The agent accepts both textual descriptions and sketches as input, engaging in interactive dialogue with users to refine and clarify design requirements through comprehensive requirement analysis. Built upon a novel Explicit Context Imperative Paradigm (ECIP), the agent generates high-quality CAD modeling code. During the generation process, the agent incorporates iterative visual feedback to improve model quality. Generated design cases are stored in a structured knowledge base, enabling continuous improvement of the agent's code generation capabilities. Experimental results demonstrate that our method achieves state-of-the-art performance in CAD code generation.

**arXiv ID:** 2508.01031
</details>

<details>
<summary><strong>Exploring Network-Knowledge Graph Duality: A Case Study in Agentic Supply Chain Risk Analysis</strong> - Evan Heus, Rick Bookstaber, Dhruv Sharma - [[pdf]](https://arxiv.org/pdf/2510.01115)</summary>

**Abstract:** Large Language Models (LLMs) struggle with the complex, multi-modal, and network-native data underlying financial risk. Standard Retrieval-Augmented Generation (RAG) oversimplifies relationships, while specialist models are costly and static. We address this gap with an LLM-centric agent framework for supply chain risk analysis. Our core contribution is to exploit the inherent duality between networks and knowledge graphs (KG). We treat the supply chain network as a KG, allowing us to use structural network science principles for retrieval. A graph traverser, guided by network centrality scores, efficiently extracts the most economically salient risk paths. An agentic architecture orchestrates this graph retrieval alongside data from numerical factor tables and news streams. Crucially, it employs novel ``context shells'' -- descriptive templates that embed raw figures in natural language -- to make quantitative data fully intelligible to the LLM. This lightweight approach enables the model to generate concise, explainable, and context-rich risk narratives in real-time without costly fine-tuning or a dedicated graph database.

**arXiv ID:** 2510.01115
</details>

<details>
<summary><strong>Astraea: A State-Aware Scheduling Engine for LLM-Powered Agents</strong> - Hongqiu Ni, Jiabao Zhang, Guopeng Li, Zilong Wang, Ruiqi Wu, Chi Zhang, Haisheng Tan - [[pdf]](https://arxiv.org/pdf/2512.14142)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly being deployed as intelligent agents. Their multi-stage workflows, which alternate between local computation and calls to external network services like Web APIs, introduce a mismatch in their execution pattern and the scheduling granularity of existing inference systems such as vLLM. Existing systems typically focus on per-segment optimization which prevents them from minimizing the end-to-end latency of the complete agentic workflow, i.e., the global Job Completion Time (JCT) over the entire request lifecycle. To address this limitation, we propose Astraea, a service engine designed to shift the optimization from local segments to the global request lifecycle. Astraea employs a state-aware, hierarchical scheduling algorithm that integrates a request's historical state with future predictions. It dynamically classifies requests by their I/O and compute intensive nature and uses an enhanced HRRN policy to balance efficiency and fairness. Astraea also implements an adaptive KV cache manager that intelligently handles the agent state during I/O waits based on the system memory pressure. Extensive experiments show that Astraea reduces average JCT by up to 25.5\% compared to baseline methods. Moreover, our approach demonstrates strong robustness and stability under high load across various model scales.

**arXiv ID:** 2512.14142
</details>

<details>
<summary><strong>MAPS$^2$: Multi-Robot Autonomous Motion Planning under Signal Temporal Logic Specifications</strong> - Mayank Sewlia, Christos K. Verginis, Dimos V. Dimarogonas - [[pdf]](https://arxiv.org/pdf/2309.05632)</summary>

**Abstract:** This article presents MAPS$^2$ : a distributed algorithm that allows multi-robot systems to deliver coupled tasks expressed as Signal Temporal Logic (STL) constraints. Classical control theoretical tools addressing STL constraints either adopt a limited fragment of the STL formula or require approximations of min/max operators, whereas works maximising robustness through optimisation-based methods often suffer from local minima, relaxing any completeness arguments due to the NP-hard nature of the problem. Endowed with probabilistic guarantees, MAPS$^2$ provides an anytime algorithm that iteratively improves the robots' trajectories. The algorithm selectively imposes spatial constraints by taking advantage of the temporal properties of the STL. The algorithm is distributed, in the sense that each robot calculates its trajectory by communicating only with its immediate neighbours as defined via a communication graph. We illustrate the efficiency of MAPS$^2$ by conducting extensive simulation and experimental studies, verifying the generation of STL satisfying trajectories.

**arXiv ID:** 2309.05632
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (22 papers)</h2></summary>

<details>
<summary><strong>Meta Hierarchical Reinforcement Learning for Scalable Resource Management in O-RAN</strong> - Fatemeh Lotfi, Fatemeh Afghah - [[pdf]](https://arxiv.org/pdf/2512.13715)</summary>

**Abstract:** The increasing complexity of modern applications demands wireless networks capable of real time adaptability and efficient resource management. The Open Radio Access Network (O-RAN) architecture, with its RAN Intelligent Controller (RIC) modules, has emerged as a pivotal solution for dynamic resource management and network slicing. While artificial intelligence (AI) driven methods have shown promise, most approaches struggle to maintain performance under unpredictable and highly dynamic conditions. This paper proposes an adaptive Meta Hierarchical Reinforcement Learning (Meta-HRL) framework, inspired by Model Agnostic Meta Learning (MAML), to jointly optimize resource allocation and network slicing in O-RAN. The framework integrates hierarchical control with meta learning to enable both global and local adaptation: the high-level controller allocates resources across slices, while low level agents perform intra slice scheduling. The adaptive meta-update mechanism weights tasks by temporal difference error variance, improving stability and prioritizing complex network scenarios. Theoretical analysis establishes sublinear convergence and regret guarantees for the two-level learning process. Simulation results demonstrate a 19.8% improvement in network management efficiency compared with baseline RL and meta-RL approaches, along with faster adaptation and higher QoS satisfaction across eMBB, URLLC, and mMTC slices. Additional ablation and scalability studies confirm the method's robustness, achieving up to 40% faster adaptation and consistent fairness, latency, and throughput performance as network scale increases.

**arXiv ID:** 2512.13715
</details>

<details>
<summary><strong>Evaluating Small Language Models for Agentic On-Farm Decision Support Systems</strong> - Enhong Liu, Haiyu Yang, Miel Hostens - [[pdf]](https://arxiv.org/pdf/2512.14043)</summary>

**Abstract:** 

**arXiv ID:** 2512.14043
</details>

<details>
<summary><strong>Seismology modeling agent: A smart assistant for geophysical researchers</strong> - Yukun Ren, Siwei Yu, Kai Chen, Jianwei Ma - [[pdf]](https://arxiv.org/pdf/2512.14429)</summary>

**Abstract:** To address the steep learning curve and reliance on complex manual file editing and command-line operations in the traditional workflow of the mainstream open-source seismic wave simulation software SPECFEM, this paper proposes an intelligent, interactive workflow powered by Large Language Models (LLMs). We introduce the first Model Context Protocol (MCP) server suite for SPECFEM (supporting 2D, 3D Cartesian, and 3D Globe versions), which decomposes the entire simulation process into discrete, agent-executable tools spanning from parameter generation and mesh partitioning to solver execution and visualization. This approach enables a paradigm shift from file-driven to intent-driven conversational interactions. The framework supports both fully automated execution and human-in-the-loop collaboration, allowing researchers to guide simulation strategies in real time and retain scientific decision-making authority while significantly reducing tedious low-level operations. Validated through multiple case studies, the workflow operates seamlessly in both autonomous and interactive modes, yielding high-fidelity results consistent with standard baselines. As the first application of MCP technology to computational seismology, this study significantly lowers the entry barrier, enhances reproducibility, and offers a promising avenue for advancing computational geophysics toward AI-assisted and automated scientific research. The complete source code is available at this https URL.

**arXiv ID:** 2512.14429
</details>

<details>
<summary><strong>Context-Picker: Dynamic context selection using multi-stage reinforcement learning</strong> - Siyuan Zhu, Chengdong Xu, Kaiqiang Ke, Chao Yu - [[pdf]](https://arxiv.org/pdf/2512.14465)</summary>

**Abstract:** In long-context question answering (LCQA), determining the optimal amount of context for a given query is a significant challenge. Including too few passages may omit critical information, while including too many can introduce noise and reduce the quality of the answer. Traditional approaches, such as fixed Top-$K$ retrieval and single-stage reranking, face the dilemma of selecting the right number of passages. This problem is particularly pronounced for factoid questions, which often require only a few specific pieces of evidence. To address this issue, we introduce \emph{Context-Picker}, a reasoning-aware framework that shifts the paradigm from similarity-based ranking to minimal sufficient subset selection. Context-Picker treats context selection as a decision-making process optimized via a human-inspired, two-stage reinforcement learning schedule: a \emph{recall-oriented} stage that prioritizes the coverage of reasoning chains, followed by a \emph{precision-oriented} stage that aggressively prunes redundancy to distill a compact evidence set. To resolve reward sparsity, we propose an offline evidence distillation pipeline that mines "minimal sufficient sets" via a Leave-One-Out (LOO) procedure, providing dense, task-aligned supervision. Experiments on five long-context and multi-hop QA benchmarks demonstrate that Context-Picker significantly outperforms strong RAG baselines, achieving superior answer accuracy with comparable or reduced context lengths. Ablation studies indicate that the coarse-to-fine optimization schedule, the redundancy-aware reward shaping, and the rationale-guided format all contribute substantially to these gains.

**arXiv ID:** 2512.14465
</details>

<details>
<summary><strong>Time-Constrained Recommendations: Reinforcement Learning Strategies for E-Commerce</strong> - Sayak Chakrabarty, Souradip Pal - [[pdf]](https://arxiv.org/pdf/2512.13726)</summary>

**Abstract:** Unlike traditional recommendation tasks, finite user time budgets introduce a critical resource constraint, requiring the recommender system to balance item relevance and evaluation cost. For example, in a mobile shopping interface, users interact with recommendations by scrolling, where each scroll triggers a list of items called slate. Users incur an evaluation cost - time spent assessing item features before deciding to click. Highly relevant items having higher evaluation costs may not fit within the user's time budget, affecting engagement. In this position paper, our objective is to evaluate reinforcement learning algorithms that learn patterns in user preferences and time budgets simultaneously, crafting recommendations with higher engagement potential under resource constraints. Our experiments explore the use of reinforcement learning to recommend items for users using Alibaba's Personalized Re-ranking dataset supporting slate optimization in e-commerce contexts. Our contributions include (i) a unified formulation of time-constrained slate recommendation modeled as Markov Decision Processes (MDPs) with budget-aware utilities; (ii) a simulation framework to study policy behavior on re-ranking data; and (iii) empirical evidence that on-policy and off-policy control can improve performance under tight time budgets than traditional contextual bandit-based methods.

**arXiv ID:** 2512.13726
</details>

<details>
<summary><strong>Professional Software Developers Don't Vibe, They Control: AI Agent Use for Coding in 2025</strong> - Ruanqianqian Huang, Avery Reyna, Sorin Lerner, Haijun Xia, Brian Hempel - [[pdf]](https://arxiv.org/pdf/2512.14012)</summary>

**Abstract:** The rise of AI agents is transforming how software can be built. The promise of agents is that developers might write code quicker, delegate multiple tasks to different agents, and even write a full piece of software purely out of natural language. In reality, what roles agents play in professional software development remains in question. This paper investigates how experienced developers use agents in building software, including their motivations, strategies, task suitability, and sentiments. Through field observations (N=13) and qualitative surveys (N=99), we find that while experienced developers value agents as a productivity boost, they retain their agency in software design and implementation out of insistence on fundamental software quality attributes, employing strategies for controlling agent behavior leveraging their expertise. In addition, experienced developers feel overall positive about incorporating agents into software development given their confidence in complementing the agents' limitations. Our results shed light on the value of software development best practices in effective use of agents, suggest the kinds of tasks for which agents may be suitable, and point towards future opportunities for better agentic interfaces and agentic use guidelines.

**arXiv ID:** 2512.14012
</details>

<details>
<summary><strong>Sample-Efficient Robot Skill Learning for Construction Tasks: Benchmarking Hierarchical Reinforcement Learning and Vision-Language-Action VLA Model</strong> - Zhaofeng Hu, Hongrui Yu, Vaidhyanathan Chandramouli, Ci-Jyun Liang - [[pdf]](https://arxiv.org/pdf/2512.14031)</summary>

**Abstract:** This study evaluates two leading approaches for teaching construction robots new skills to understand their applicability for construction automation: a Vision-Language-Action (VLA) model and Reinforcement Learning (RL) methods. The goal is to understand both task performance and the practical effort needed to deploy each approach on real jobs. The authors developed two teleoperation interfaces to control the robots and collect the demonstrations needed, both of which proved effective for training robots for long-horizon and dexterous tasks. In addition, the authors conduct a three-stage evaluation. First, the authors compare a Multi-Layer Perceptron (MLP) policy with a Deep Q-network (DQN) imitation model to identify the stronger RL baseline, focusing on model performance, generalization, and a pick-up experiment. Second, three different VLA models are trained in two different scenarios and compared with each other. Third, the authors benchmark the selected RL baseline against the VLA model using computational and sample-efficiency measures and then a robot experiment on a multi-stage panel installation task that includes transport and installation. The VLA model demonstrates strong generalization and few-shot capability, achieving 60% and 100% success in the pickup phase. In comparison, DQN can be made robust but needs additional noise during tuning, which increases the workload. Overall, the findings indicate that VLA offers practical advantages for changing tasks by reducing programming effort and enabling useful performance with minimal data, while DQN provides a viable baseline when sufficient tuning effort is acceptable.

**arXiv ID:** 2512.14031
</details>

<details>
<summary><strong>OmniDrive-R1: Reinforcement-driven Interleaved Multi-modal Chain-of-Thought for Trustworthy Vision-Language Autonomous Driving</strong> - Zhenguo Zhang, Haohan Zhen, Yishen Wang, Le Xu, Tianchen Deng, Xuefeng Chen, Qu Chen, Bo Zhang, Wuxiong Huang - [[pdf]](https://arxiv.org/pdf/2512.14044)</summary>

**Abstract:** The deployment of Vision-Language Models (VLMs) in safety-critical domains like autonomous driving (AD) is critically hindered by reliability failures, most notably object hallucination. This failure stems from their reliance on ungrounded, text-based Chain-of-Thought (CoT) this http URL existing multi-modal CoT approaches attempt mitigation, they suffer from two fundamental flaws: (1) decoupled perception and reasoning stages that prevent end-to-end joint optimization, and (2) reliance on expensive, dense localization this http URL we introduce OmniDrive-R1, an end-to-end VLM framework designed for autonomous driving, which unifies perception and reasoning through an interleaved Multi-modal Chain-of-Thought (iMCoT) mechanism. Our core innovation is an Reinforcement-driven visual grounding capability, enabling the model to autonomously direct its attention and "zoom in" on critical regions for fine-grained analysis. This capability is enabled by our pure two-stage reinforcement learning training pipeline and Clip-GRPO algorithm. Crucially, Clip-GRPO introduces an annotation-free, process-based grounding reward. This reward not only eliminates the need for dense labels but also circumvents the instability of external tool calls by enforcing real-time cross-modal consistency between the visual focus and the textual reasoning. Extensive experiments on DriveLMM-o1 demonstrate our model's significant improvements. Compared to the baseline Qwen2.5VL-7B, OmniDrive-R1 improves the overall reasoning score from 51.77% to 80.35%, and the final answer accuracy from 37.81% to 73.62%.

**arXiv ID:** 2512.14044
</details>

<details>
<summary><strong>Understanding and Improving Hyperbolic Deep Reinforcement Learning</strong> - Timo Klein, Thomas Lang, Andrii Shkabrii, Alexander Sturm, Kevin Sidak, Lukas Miklautz, Claudia Plant, Yllka Velaj, Sebastian Tschiatschek - [[pdf]](https://arxiv.org/pdf/2512.14202)</summary>

**Abstract:** The performance of reinforcement learning (RL) agents depends critically on the quality of the underlying feature representations. Hyperbolic feature spaces are well-suited for this purpose, as they naturally capture hierarchical and relational structure often present in complex RL environments. However, leveraging these spaces commonly faces optimization challenges due to the nonstationarity of RL. In this work, we identify key factors that determine the success and failure of training hyperbolic deep RL agents. By analyzing the gradients of core operations in the Poincaré Ball and Hyperboloid models of hyperbolic geometry, we show that large-norm embeddings destabilize gradient-based training, leading to trust-region violations in proximal policy optimization (PPO). Based on these insights, we introduce Hyper++, a new hyperbolic PPO agent that consists of three components: (i) stable critic training through a categorical value loss instead of regression; (ii) feature regularization guaranteeing bounded norms while avoiding the curse of dimensionality from clipping; and (iii) using a more optimization-friendly formulation of hyperbolic network layers. In experiments on ProcGen, we show that Hyper++ guarantees stable learning, outperforms prior hyperbolic agents, and reduces wall-clock time by approximately 30%. On Atari-5 with Double DQN, Hyper++ strongly outperforms Euclidean and hyperbolic baselines. We release our code at this https URL .

**arXiv ID:** 2512.14202
</details>

<details>
<summary><strong>Model-Based Reinforcement Learning in Discrete-Action Non-Markovian Reward Decision Processes</strong> - Alessandro Trapasso, Luca Iocchi, Fabio Patrizi - [[pdf]](https://arxiv.org/pdf/2512.14617)</summary>

**Abstract:** Many practical decision-making problems involve tasks whose success depends on the entire system history, rather than on achieving a state with desired properties. Markovian Reinforcement Learning (RL) approaches are not suitable for such tasks, while RL with non-Markovian reward decision processes (NMRDPs) enables agents to tackle temporal-dependency tasks. This approach has long been known to lack formal guarantees on both (near-)optimality and sample efficiency. We contribute to solving both issues with QR-MAX, a novel model-based algorithm for discrete NMRDPs that factorizes Markovian transition learning from non-Markovian reward handling via reward machines. To the best of our knowledge, this is the first model-based RL algorithm for discrete-action NMRDPs that exploits this factorization to obtain PAC convergence to $\varepsilon$-optimal policies with polynomial sample complexity. We then extend QR-MAX to continuous state spaces with Bucket-QR-MAX, a SimHash-based discretiser that preserves the same factorized structure and achieves fast and stable learning without manual gridding or function approximation. We experimentally compare our method with modern state-of-the-art model-based RL approaches on environments of increasing complexity, showing a significant improvement in sample efficiency and increased robustness in finding optimal policies.

**arXiv ID:** 2512.14617
</details>

<details>
<summary><strong>Meta-Reinforcement Learning for Building Energy Management System</strong> - Huiliang Zhang, Di Wu, Arnaud Zinflou, Benoit Boulet - [[pdf]](https://arxiv.org/pdf/2210.12590)</summary>

**Abstract:** The building sector is one of the largest contributors to global energy consumption. Improving its energy efficiency is essential for reducing operational costs and greenhouse gas emissions. Energy management systems (EMS) play a key role in monitoring and controlling building appliances efficiently and reliably. With the increasing integration of renewable energy, intelligent EMS solutions have received growing attention. Reinforcement learning (RL) has recently been explored for this purpose and shows strong potential. However, most RL-based EMS methods require a large number of training steps to learn effective control policies, especially when adapting to unseen buildings, which limits their practical deployment. This paper introduces MetaEMS, a meta-reinforcement learning framework for EMS. MetaEMS improves learning efficiency by transferring knowledge from previously solved tasks to new ones through group-level and building-level adaptation, enabling fast adaptation and effective control across diverse building environments. Experimental results demonstrate that MetaEMS adapts more rapidly to unseen buildings and consistently outperforms baseline methods across various scenarios.

**arXiv ID:** 2210.12590
</details>

<details>
<summary><strong>Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning</strong> - Hongye Cao, Zhixin Bai, Ziyue Peng, Boyan Wang, Tianpei Yang, Jing Huo, Yuyao Zhang, Yang Gao - [[pdf]](https://arxiv.org/pdf/2512.04359)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has demonstrated superior performance in enhancing the reasoning capability of large language models (LLMs). However, this accuracy-oriented learning paradigm often suffers from entropy collapse, which reduces policy exploration and limits reasoning capabilities. To address this challenge, we propose an efficient reinforcement learning framework that leverages entropy signals at both the semantic and token levels to improve reasoning. From the data perspective, we introduce semantic entropy-guided curriculum learning, organizing training data from low to high semantic entropy to guide progressive optimization from easier to more challenging tasks. For the algorithmic design, we adopt non-uniform token treatment by imposing KL regularization on low-entropy tokens that critically impact policy exploration and applying stronger constraints on high-covariance portions within these tokens. By jointly optimizing data organization and algorithmic design, our method effectively mitigates entropy collapse and enhances LLM reasoning. Experimental results across 6 benchmarks with 3 different parameter-scale base models demonstrate that our method outperforms other entropy-based approaches in improving reasoning.

**arXiv ID:** 2512.04359
</details>

<details>
<summary><strong>Optimizing Highway Traffic Flow in Mixed Autonomy: A Multiagent Truncated Rollout Approach</strong> - Lu Liu, Chi Xie, Xi Xiong - [[pdf]](https://arxiv.org/pdf/2508.19203)</summary>

**Abstract:** The development of connected and autonomous vehicles (CAVs) offers substantial opportunities to enhance traffic efficiency. However, in mixed autonomy environments where CAVs coexist with human-driven vehicles (HDVs), achieving efficient coordination among CAVs remains challenging due to heterogeneous driving behaviors. To address this, this paper proposes a multiagent truncated rollout approach that enhances CAV speed coordination to improve highway throughput while reducing computational overhead. In this approach, a traffic density evolution equation is formulated that comprehensively accounts for the presence or absence of CAVs, and a distributed coordination control framework is established accordingly. By incorporating kinematic information from neighbor agents and employing an agent-by-agent sequential solution mechanism, our method enables explicit cooperation among CAVs. Furthermore, we introduce a truncated rollout scheme that adaptively shortens the optimization horizon based on the evaluation of control sequences. This significantly reduces the time complexity, thereby improving real-time performance and scalability. Theoretical analysis provides rigorous guarantees on the stability and performance improvement of the system. Simulations conducted on real-world bottleneck scenarios demonstrate that, in large-scale mixed traffic flows, the proposed method outperforms conventional model predictive control methods by reducing both the average travel time in the bottleneck area and overall computational time, highlighting its strong potential for practical deployment.

**arXiv ID:** 2508.19203
</details>

<details>
<summary><strong>RAST-MoE-RL: A Regime-Aware Spatio-Temporal MoE Framework for Deep Reinforcement Learning in Ride-Hailing</strong> - Yuhan Tang, Kangxin Cui, Jung Ho Park, Yibo Zhao, Xuan Jiang, Haoze He, Dingyi Zhuang, Shenhao Wang, Jiangbo Yu, Haris Koutsopoulos, Jinhua Zhao - [[pdf]](https://arxiv.org/pdf/2512.13727)</summary>

**Abstract:** Ride-hailing platforms face the challenge of balancing passenger waiting times with overall system efficiency under highly uncertain supply-demand conditions. Adaptive delayed matching creates a trade-off between matching and pickup delays by deciding whether to assign drivers immediately or batch requests. Since outcomes accumulate over long horizons with stochastic dynamics, reinforcement learning (RL) is a suitable framework. However, existing approaches often oversimplify traffic dynamics or use shallow encoders that miss complex spatiotemporal patterns.
We introduce the Regime-Aware Spatio-Temporal Mixture-of-Experts (RAST-MoE), which formalizes adaptive delayed matching as a regime-aware MDP equipped with a self-attention MoE encoder. Unlike monolithic networks, our experts specialize automatically, improving representation capacity while maintaining computational efficiency. A physics-informed congestion surrogate preserves realistic density-speed feedback, enabling millions of efficient rollouts, while an adaptive reward scheme guards against pathological strategies.
With only 12M parameters, our framework outperforms strong baselines. On real-world Uber trajectory data (San Francisco), it improves total reward by over 13%, reducing average matching and pickup delays by 10% and 15% respectively. It demonstrates robustness across unseen demand regimes and stable training. These findings highlight the potential of MoE-enhanced RL for large-scale decision-making with complex spatiotemporal dynamics.

**arXiv ID:** 2512.13727
</details>

<details>
<summary><strong>Explainable reinforcement learning from human feedback to improve alignment</strong> - Shicheng Liu, Siyuan Xu, Wenjie Qiu, Hangfan Zhang, Minghui Zhu - [[pdf]](https://arxiv.org/pdf/2512.13837)</summary>

**Abstract:** A common and effective strategy for humans to improve an unsatisfactory outcome in daily life is to find a cause of this outcome and correct the cause. In this paper, we investigate whether this human improvement strategy can be applied to improving reinforcement learning from human feedback (RLHF) for alignment of language models (LMs). In particular, it is observed in the literature that LMs tuned by RLHF can still output unsatisfactory responses. This paper proposes a method to improve the unsatisfactory responses by correcting their causes. Our method has two parts. The first part proposes a post-hoc explanation method to explain why an unsatisfactory response is generated to a prompt by identifying the training data that lead to this response. We formulate this problem as a constrained combinatorial optimization problem where the objective is to find a set of training data closest to this prompt-response pair in a feature representation space, and the constraint is that the prompt-response pair can be decomposed as a convex combination of this set of training data in the feature space. We propose an efficient iterative data selection algorithm to solve this problem. The second part proposes an unlearning method that improves unsatisfactory responses to some prompts by unlearning the training data that lead to these unsatisfactory responses and, meanwhile, does not significantly degrade satisfactory responses to other prompts. Experimental results demonstrate that our algorithm can improve RLHF.

**arXiv ID:** 2512.13837
</details>

<details>
<summary><strong>Sim2Real Reinforcement Learning for Soccer skills</strong> - Jonathan Spraggett - [[pdf]](https://arxiv.org/pdf/2512.12437)</summary>

**Abstract:** This thesis work presents a more efficient and effective approach to training control-related tasks for humanoid robots using Reinforcement Learning (RL). The traditional RL methods are limited in adapting to real-world environments, complexity, and natural motions, but the proposed approach overcomes these limitations by using curriculum training and Adversarial Motion Priors (AMP) technique. The results show that the developed RL policies for kicking, walking, and jumping are more dynamic, and adaptive, and outperformed previous methods. However, the transfer of the learned policy from simulation to the real world was unsuccessful, highlighting the limitations of current RL methods in fully adapting to real-world scenarios.

**arXiv ID:** 2512.12437
</details>

<details>
<summary><strong>Group-Theoretic Reinforcement Learning of Dynamical Decoupling Sequences</strong> - Charles Marrder, Shuo Sun, Murray J. Holland - [[pdf]](https://arxiv.org/pdf/2512.13890)</summary>

**Abstract:** Dynamical decoupling seeks to mitigate phase decoherence in qubits by applying a carefully designed sequence of effectively instantaneous electromagnetic pulses. Although analytic solutions exist for pulse timings that are optimal under specific noise regimes, identifying the optimal timings for a realistic noise spectrum remains challenging. We propose a reinforcement learning (RL)-based method for designing pulse sequences on qubits. Our novel action set enables the RL agent to efficiently navigate this inherently non-convex optimization landscape. The action set, derived from Thompson's group $F$, is applicable to a broad class of sequential decision problems whose states can be represented as bounded sequences. We demonstrate that our RL agent can learn pulse sequences that minimize dephasing without requiring explicit knowledge of the underlying noise spectrum. This work opens the possibility for real-time learning of optimal dynamical decoupling sequences on qubits which are dephasing-limited. The model-free nature of our algorithm suggests that the agent may ultimately learn optimal pulse sequences even in the presence of unmodeled physical effects, such as pulse errors or non-Gaussian noise.

**arXiv ID:** 2512.13890
</details>

<details>
<summary><strong>ChartAgent: A Chart Understanding Framework with Tool Integrated Reasoning</strong> - Boran Wang, Xinming Wang, Yi Chen, Xiang Li, Jian Xu, Jing Yuan, Chenglin Liu - [[pdf]](https://arxiv.org/pdf/2512.14040)</summary>

**Abstract:** With their high information density and intuitive readability, charts have become the de facto medium for data analysis and communication across disciplines. Recent multimodal large language models (MLLMs) have made notable progress in automated chart understanding, yet they remain heavily dependent on explicit textual annotations and the performance degrades markedly when key numerals are absent. To address this limitation, we introduce ChartAgent, a chart understanding framework grounded in Tool-Integrated Reasoning (TIR). Inspired by human cognition, ChartAgent decomposes complex chart analysis into a sequence of observable, replayable steps. Supporting this architecture is an extensible, modular tool library comprising more than a dozen core tools, such as keyelement detection, instance segmentation, and optical character recognition (OCR), which the agent dynamically orchestrates to achieve systematic visual parsing across diverse chart types. Leveraging TIRs transparency and verifiability, ChartAgent moves beyond the black box paradigm by standardizing and consolidating intermediate outputs into a structured Evidence Package, providing traceable and reproducible support for final conclusions. Experiments show that ChartAgent substantially improves robustness under sparse annotation settings, offering a practical path toward trustworthy and extensible systems for chart understanding.

**arXiv ID:** 2512.14040
</details>

<details>
<summary><strong>CiRL: Open-Source Environments for Reinforcement Learning in Circular Economy and Net Zero</strong> - Federico Zocco, Andrea Corti, Monica Malvezzi - [[pdf]](https://arxiv.org/pdf/2505.21536)</summary>

**Abstract:** The demand of finite raw materials will keep increasing as they fuel modern society. Simultaneously, solutions for stopping carbon emissions in the short term are not available, thus making the net zero target extremely challenging to achieve at scale. The circular economy (CE) paradigm is gaining attention as a solution to address climate change and the uncertainties of supplies of critical materials. Hence, in this paper, we introduce CiRL, a deep reinforcement learning (DRL) library of environments focused on the circularity control of both solid and fluid materials. The integration of DRL into the design of material circularity is possible thanks to the formalism of thermodynamical material networks, which is underpinned by compartmental dynamical thermodynamics. Along with the focus on circularity, this library has three more features: the new CE-oriented environments are in the state-space form, which is typically used in dynamical systems analysis and control design; it is based on a state-of-the-art Python library of DRL algorithms, namely, Stable-Baselines3; and it is developed in Google Colaboratory to be accessible to researchers from different disciplines and backgrounds as is often the case for circular economy researchers and engineers. CiRL is intended to be a tool to generate AI-driven actions for optimizing the circularity of supply-recovery chains and to be combined with human-driven decisions derived from material flow analysis (MFA) studies. CiRL is publicly available.

**arXiv ID:** 2505.21536
</details>

<details>
<summary><strong>PrivORL: Differentially Private Synthetic Dataset for Offline Reinforcement Learning</strong> - Chen Gong, Zheng Liu, Kecen Li, Tianhao Wang - [[pdf]](https://arxiv.org/pdf/2512.07342)</summary>

**Abstract:** Recently, offline reinforcement learning (RL) has become a popular RL paradigm. In offline RL, data providers share pre-collected datasets -- either as individual transitions or sequences of transitions forming trajectories -- to enable the training of RL models (also called agents) without direct interaction with the environments. Offline RL saves interactions with environments compared to traditional RL, and has been effective in critical areas, such as navigation tasks. Meanwhile, concerns about privacy leakage from offline RL datasets have emerged.
To safeguard private information in offline RL datasets, we propose the first differential privacy (DP) offline dataset synthesis method, PrivORL, which leverages a diffusion model and diffusion transformer to synthesize transitions and trajectories, respectively, under DP. The synthetic dataset can then be securely released for downstream analysis and research. PrivORL adopts the popular approach of pre-training a synthesizer on public datasets, and then fine-tuning on sensitive datasets using DP Stochastic Gradient Descent (DP-SGD). Additionally, PrivORL introduces curiosity-driven pre-training, which uses feedback from the curiosity module to diversify the synthetic dataset and thus can generate diverse synthetic transitions and trajectories that closely resemble the sensitive dataset. Extensive experiments on five sensitive offline RL datasets show that our method achieves better utility and fidelity in both DP transition and trajectory synthesis compared to baselines. The replication package is available at the GitHub repository.

**arXiv ID:** 2512.07342
</details>

<details>
<summary><strong>Context Representation via Action-Free Transformer encoder-decoder for Meta Reinforcement Learning</strong> - Amir M. Soufi Enayati, Homayoun Honari, Homayoun Najjaran - [[pdf]](https://arxiv.org/pdf/2512.14057)</summary>

**Abstract:** Reinforcement learning (RL) enables robots to operate in uncertain environments, but standard approaches often struggle with poor generalization to unseen tasks. Context-adaptive meta reinforcement learning addresses these limitations by conditioning on the task representation, yet they mostly rely on complete action information in the experience making task inference tightly coupled to a specific policy. This paper introduces Context Representation via Action Free Transformer encoder decoder (CRAFT), a belief model that infers task representations solely from sequences of states and rewards. By removing the dependence on actions, CRAFT decouples task inference from policy optimization, supports modular training, and leverages amortized variational inference for scalable belief updates. Built on a transformer encoder decoder with rotary positional embeddings, the model captures long range temporal dependencies and robustly encodes both parametric and non-parametric task variations. Experiments on the MetaWorld ML-10 robotic manipulation benchmark show that CRAFT achieves faster adaptation, improved generalization, and more effective exploration compared to context adaptive meta--RL baselines. These findings highlight the potential of action-free inference as a foundation for scalable RL in robotic control.

**arXiv ID:** 2512.14057
</details>

<details>
<summary><strong>MindDrive: A Vision-Language-Action Model for Autonomous Driving via Online Reinforcement Learning</strong> - Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Hongwei Xie, Bing Wang, Guang Chen, Dingkang Liang, Xiang Bai - [[pdf]](https://arxiv.org/pdf/2512.13636)</summary>

**Abstract:** Current Vision-Language-Action (VLA) paradigms in autonomous driving primarily rely on Imitation Learning (IL), which introduces inherent challenges such as distribution shift and causal confusion. Online Reinforcement Learning offers a promising pathway to address these issues through trial-and-error learning. However, applying online reinforcement learning to VLA models in autonomous driving is hindered by inefficient exploration in continuous action spaces. To overcome this limitation, we propose MindDrive, a VLA framework comprising a large language model (LLM) with two distinct sets of LoRA parameters. The one LLM serves as a Decision Expert for scenario reasoning and driving decision-making, while the other acts as an Action Expert that dynamically maps linguistic decisions into feasible trajectories. By feeding trajectory-level rewards back into the reasoning space, MindDrive enables trial-and-error learning over a finite set of discrete linguistic driving decisions, instead of operating directly in a continuous action space. This approach effectively balances optimal decision-making in complex scenarios, human-like driving behavior, and efficient exploration in online reinforcement learning. Using the lightweight Qwen-0.5B LLM, MindDrive achieves Driving Score (DS) of 78.04 and Success Rate (SR) of 55.09% on the challenging Bench2Drive benchmark. To the best of our knowledge, this is the first work to demonstrate the effectiveness of online reinforcement learning for the VLA model in autonomous driving.

**arXiv ID:** 2512.13636
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
