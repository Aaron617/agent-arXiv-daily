# Agent arXiv Daily

**Last Updated:** 2025-12-15 03:05:52

**Total Papers:** 49

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (2 papers)</h2></summary>

<details>
<summary><strong>Driving Through Uncertainty: Risk-Averse Control with LLM Commonsense for Autonomous Driving under Perception Deficits</strong> - Yuting Hu, Chenhui Xu, Ruiyang Qin, Dancheng Liu, Amir Nassereldine, Yiyu Shi, Jinjun Xiong - [[pdf]](https://arxiv.org/pdf/2503.07020)</summary>

**Abstract:** Partial perception deficits can compromise autonomous vehicle safety by disrupting environmental understanding. Existing protocols typically default to entirely risk-avoidant actions such as immediate stops, which are detrimental to navigation goals and lack flexibility for rare driving scenarios. Yet, in cases of minor risk, halting the vehicle may be unnecessary, and more adaptive responses are preferable. In this paper, we propose LLM-RCO, a risk-averse framework leveraging large language models (LLMs) to integrate human-like driving commonsense into autonomous systems facing perception deficits. LLM-RCO features four key modules interacting with the dynamic driving environment: hazard inference, short-term motion planner, action condition verifier, and safety constraint generator, enabling proactive and context-aware actions in such challenging conditions. To enhance the driving decision-making of LLMs, we construct DriveLM-Deficit, a dataset of 53,895 video clips featuring deficits of safety-critical objects, annotated for LLM fine-tuning in hazard detection and motion planning. Extensive experiments in adverse driving conditions with the CARLA simulator demonstrate that LLM-RCO promotes proactive maneuvers over purely risk-averse actions in perception deficit scenarios, underscoring its value for boosting autonomous driving resilience against perception loss challenges.

**arXiv ID:** 2503.07020
</details>

<details>
<summary><strong>When Actions Teach You to Think: Reasoning-Action Synergy via Reinforcement Learning in Conversational Agents</strong> - Mrinal Rawat, Arkajyoti Chakraborty, Neha Gupta, Roberto Pieraccini - [[pdf]](https://arxiv.org/pdf/2512.11277)</summary>

**Abstract:** Supervised fine-tuning (SFT) has emerged as one of the most effective ways to improve the performance of large language models (LLMs) in downstream tasks. However, SFT can have difficulty generalizing when the underlying data distribution changes, even when the new data does not fall completely outside the training domain. Recent reasoning-focused models such as o1 and R1 have demonstrated consistent gains over their non-reasoning counterparts, highlighting the importance of reasoning for improved generalization and reliability. However, collecting high-quality reasoning traces for SFT remains challenging -- annotations are costly, subjective, and difficult to scale. To address this limitation, we leverage Reinforcement Learning (RL) to enable models to learn reasoning strategies directly from task outcomes. We propose a pipeline in which LLMs generate reasoning steps that guide both the invocation of tools (e.g., function calls) and the final answer generation for conversational agents. Our method employs Group Relative Policy Optimization (GRPO) with rewards designed around tool accuracy and answer correctness, allowing the model to iteratively refine its reasoning and actions. Experimental results demonstrate that our approach improves both the quality of reasoning and the precision of tool invocations, achieving a 1.5% relative improvement over the SFT model (trained without explicit thinking) and a 40% gain compared to the base of the vanilla Qwen3-1.7B model. These findings demonstrate the promise of unifying reasoning and action learning through RL to build more capable and generalizable conversational agents.

**arXiv ID:** 2512.11277
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (8 papers)</h2></summary>

<details>
<summary><strong>MedAI: Evaluating TxAgent's Therapeutic Agentic Reasoning in the NeurIPS CURE-Bench Competition</strong> - Tim Cofala, Christian Kalfar, Jingge Xiao, Johanna Schrader, Michelle Tang, Wolfgang Nejdl - [[pdf]](https://arxiv.org/pdf/2512.11682)</summary>

**Abstract:** Therapeutic decision-making in clinical medicine constitutes a high-stakes domain in which AI guidance interacts with complex interactions among patient characteristics, disease processes, and pharmacological agents. Tasks such as drug recommendation, treatment planning, and adverse-effect prediction demand robust, multi-step reasoning grounded in reliable biomedical knowledge. Agentic AI methods, exemplified by TxAgent, address these challenges through iterative retrieval-augmented generation (RAG). TxAgent employs a fine-tuned Llama-3.1-8B model that dynamically generates and executes function calls to a unified biomedical tool suite (ToolUniverse), integrating FDA Drug API, OpenTargets, and Monarch resources to ensure access to current therapeutic information. In contrast to general-purpose RAG systems, medical applications impose stringent safety constraints, rendering the accuracy of both the reasoning trace and the sequence of tool invocations critical. These considerations motivate evaluation protocols treating token-level reasoning and tool-usage behaviors as explicit supervision signals. This work presents insights derived from our participation in the CURE-Bench NeurIPS 2025 Challenge, which benchmarks therapeutic-reasoning systems using metrics that assess correctness, tool utilization, and reasoning quality. We analyze how retrieval quality for function (tool) calls influences overall model performance and demonstrate performance gains achieved through improved tool-retrieval strategies. Our work was awarded the Excellence Award in Open Science. Complete information can be found at this https URL.

**arXiv ID:** 2512.11682
</details>

<details>
<summary><strong>Scalable Data Synthesis for Computer Use Agents with Step-Level Filtering</strong> - Yifei He, Pranit Chawla, Yaser Souri, Subhojit Som, Xia Song - [[pdf]](https://arxiv.org/pdf/2512.10962)</summary>

**Abstract:** Computer use agents (CUAs) can operate real-world digital interfaces but remain difficult to train due to the high cost of graphical user interface (GUI) interaction and the scarcity of high-quality trajectory data. Existing datasets rely on human demonstrations, limiting scalability. A natural alternative is to synthesize data from strong CUAs, yet their rollouts are highly noisy, with incorrect or suboptimal actions consisting a large proportion of the steps, making naive imitation ineffective. To tackle this challenge, we introduce a scalable data synthesis pipeline that transforms noisy rollouts into reliable supervision without human annotation. The core idea is step-level filtering, which evaluates actions individually to retain only correct steps, complemented by reasoning augmentation for improved planning. Using this pipeline, we construct WebSTAR, a dataset of 13.3K trajectories and 100K graded, reasoning-rich steps synthesized from OpenAI's computer-use-preview model. We train Qwen-2.5-VL-Instruct models (7B and 32B) on WebSTAR. On WebVoyager, our 7B model surpasses SoTA open-source CUA model UI-TARS-1.5-7B by more than 15% with only supervised finetuning. Building on step-level grading, we further create WebSCORE, a dataset of graded step-level actions, and train StepRM, a 7B multimodal reward model distilled from o4-mini, which matches its grading quality while being far more efficient to deploy at scale. Our results establish step-level filtering as a key principle for scalable CUA training and construct two new datasets (WebSTAR, WebSCORE) and a lightweight reward model (StepRM) as practical tools to advance robust and efficient CUAs.

**arXiv ID:** 2512.10962
</details>

<details>
<summary><strong>MiniScope: A Least Privilege Framework for Authorizing Tool Calling Agents</strong> - Jinhao Zhu, Kevin Tseng, Gil Vernik, Xiao Huang, Shishir G. Patil, Vivian Fang, Raluca Ada Popa - [[pdf]](https://arxiv.org/pdf/2512.11147)</summary>

**Abstract:** Tool calling agents are an emerging paradigm in LLM deployment, with major platforms such as ChatGPT, Claude, and Gemini adding connectors and autonomous capabilities. However, the inherent unreliability of LLMs introduces fundamental security risks when these agents operate over sensitive user services. Prior approaches either rely on manually written policies that require security expertise, or place LLMs in the confinement loop, which lacks rigorous security guarantees. We present MiniScope, a framework that enables tool calling agents to operate on user accounts while confining potential damage from unreliable LLMs. MiniScope introduces a novel way to automatically and rigorously enforce least privilege principles by reconstructing permission hierarchies that reflect relationships among tool calls and combining them with a mobile-style permission model to balance security and ease of use. To evaluate MiniScope, we create a synthetic dataset derived from ten popular real-world applications, capturing the complexity of realistic agentic tasks beyond existing simplified benchmarks. Our evaluation shows that MiniScope incurs only 1-6% latency overhead compared to vanilla tool calling agents, while significantly outperforming the LLM based baseline in minimizing permissions as well as computational and operational costs.

**arXiv ID:** 2512.11147
</details>

<details>
<summary><strong>Atomic Action Slicing: Planner-Aligned Options for Generalist VLA Agents</strong> - Stefan Tabakov, Asen Popov, Dimitar Dimitrov, S. Ensiye Kiyamousavi, Vladimir Hristov, Boris Kraychev - [[pdf]](https://arxiv.org/pdf/2512.11584)</summary>

**Abstract:** Current vision-language-action (VLA) models generalize poorly, particularly when tasks require new compositions of skills or objects. We introduce Atomic Action Slicing (AAS), a planner-aligned approach that decomposes long-horizon demonstrations into short, typed atomic actions that are easier for planners to use and policies to learn. Using LIBERO demonstrations, AAS produces a validated dataset of 2,124 atomic segments labeled with action type, temporal span, and confidence. A stronger segmenter (Gemini 2.5 Pro) closely matches planner-defined plans and remains robust under keyframe jitter, while smaller models perform worse on multi-object tasks. Fine-tuning CLIP-RT+ on our atomic dataset improves task success from 94.2% to 95.3% on LIBERO-Goal and 83.8% to 88.8% on LIBERO-Long. We publicly release the GATE-VLAP dataset on HuggingFace(this https URL)

**arXiv ID:** 2512.11584
</details>

<details>
<summary><strong>Octopus: Agentic Multimodal Reasoning with Six-Capability Orchestration</strong> - Yifu Guo, Zishan Xu, Zhiyuan Yao, Yuquan Lu, Jiaye Lin, Sen Hu, Zhenheng Tang, Huacan Wang, Ronghao Chen - [[pdf]](https://arxiv.org/pdf/2511.15351)</summary>

**Abstract:** Existing multimodal reasoning models and frameworks suffer from fundamental architectural limitations: most lack the human-like ability to autonomously explore diverse reasoning pathways-whether in direct inference, tool-driven visual exploration, programmatic visual manipulation, or intrinsic visual imagination. Consequently, they struggle to adapt to dynamically changing capability requirements in real-world tasks. Meanwhile, humans exhibit a complementary set of thinking abilities when addressing such tasks, whereas existing methods typically cover only a subset of these dimensions. Inspired by this, we propose Octopus: Agentic Multimodal Reasoning with Six-Capability Orchestration, a new paradigm for multimodal agentic reasoning. We define six core capabilities essential for multimodal reasoning and organize a comprehensive evaluation benchmark, Octopus-Bench, accordingly. Octopus is capable of autonomously exploring during reasoning and dynamically selecting the most appropriate capability based on the current state. Experimental results show that Octopus achieves the best performance on the vast majority of tasks in Octopus-Bench, highlighting the crucial role of capability coordination in agentic multimodal reasoning.

**arXiv ID:** 2511.15351
</details>

<details>
<summary><strong>RECAP: REwriting Conversations for Intent Understanding in Agentic Planning</strong> - Kushan Mitra, Dan Zhang, Hannah Kim, Estevam Hruschka - [[pdf]](https://arxiv.org/pdf/2509.04472)</summary>

**Abstract:** Understanding user intent is essential for effective planning in conversational assistants, particularly those powered by large language models (LLMs) coordinating multiple agents. However, real-world dialogues are often ambiguous, underspecified, or dynamic, making intent detection a persistent challenge. Traditional classification-based approaches struggle to generalize in open-ended settings, leading to brittle interpretations and poor downstream planning. We propose RECAP (REwriting Conversations for Agent Planning), a new benchmark designed to evaluate and advance intent rewriting, reframing user-agent dialogues into concise representations of user goals. RECAP captures diverse challenges such as ambiguity, intent drift, vagueness, and mixed-goal conversations. Alongside the dataset, we introduce an LLM-based evaluator that assesses planning utility given the rewritten intent. Using RECAP, we develop a prompt-based rewriting approach that outperforms baselines, in terms of plan preference. We further demonstrate that fine-tuning two DPO-based rewriters yields additional utility gains. Our results highlight intent rewriting as a critical and tractable component for improving agentic planning in open-domain dialogue systems.

**arXiv ID:** 2509.04472
</details>

<details>
<summary><strong>Osprey: Production-Ready Agentic AI for Safety-Critical Control Systems</strong> - Thorsten Hellert, João Montenegro, Antonin Sulc - [[pdf]](https://arxiv.org/pdf/2508.15066)</summary>

**Abstract:** Operating large-scale scientific facilities requires coordinating diverse subsystems, translating operator intent into precise hardware actions, and maintaining strict safety oversight. Language model-driven agents offer a natural interface for these tasks, but most existing approaches are not yet reliable or safe enough for production use. In this paper, we introduce Osprey, a framework for using agentic AI in large, safety-critical facility operations. Osprey is built around the needs of control rooms and addresses these challenges in four ways. First, it uses a plan-first orchestrator that generates complete execution plans, including all dependencies, for human review before any hardware is touched. Second, a coordination layer manages complex data flows, keeps data types consistent, and automatically downsamples large datasets when needed. Third, a classifier dynamically selects only the tools required for a given task, keeping prompts compact as facilities add capabilities. Fourth, connector abstractions and deployment patterns work across different control systems and are ready for day-to-day use. We demonstrate the framework through two case studies: a control-assistant tutorial showing semantic channel mapping and historical data integration, and a production deployment at the Advanced Light Source, where Osprey manages real-time operations across hundreds of thousands of control channels. These results establish Osprey as a production-ready framework for deploying agentic AI in complex, safety-critical environments.

**arXiv ID:** 2508.15066
</details>

<details>
<summary><strong>Two-dimensional Decompositions of High-dimensional Configurations for Efficient Multi-vehicle Coordination at Intelligent Intersections</strong> - Amirreza Akbari, Johan Thunberg - [[pdf]](https://arxiv.org/pdf/2512.11713)</summary>

**Abstract:** For multi-vehicle complex traffic scenarios in shared spaces such as intelligent intersections, safe coordination and trajectory planning is challenging due to computational complexity. To meet this challenge, we introduce a computationally efficient method for generating collision-free trajectories along predefined vehicle paths. We reformulate a constrained minimum-time trajectory planning problem as a problem in a high-dimensional configuration space, where conflict zones are modeled by high-dimensional polyhedra constructed from two-dimensional rectangles. Still, in such a formulation, as the number of vehicles involved increases, the computational complexity increases significantly. To address this, we propose two algorithms for near-optimal local optimization that significantly reduce the computational complexity by decomposing the high-dimensional problem into a sequence of 2D graph search problems. The resulting trajectories are then incorporated into a Nonlinear Model Predictive Control (NMPC) framework to ensure safe and smooth vehicle motion. We furthermore show in numerical evaluation that this approach significantly outperforms existing MILP-based time-scheduling; both in terms of objective-value and computational time.

**arXiv ID:** 2512.11713
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Towards Trustworthy Multi-Turn LLM Agents via Behavioral Guidance</strong> - Gonca Gürsun - [[pdf]](https://arxiv.org/pdf/2512.11421)</summary>

**Abstract:** Large Language Models demonstrate strong reasoning and generation abilities, yet their behavior in multi-turn tasks often lacks reliability and verifiability. We present a task completion framework that enables LLM-based agents to act under explicit behavioral guidance in environments described by reinforcement learning formalisms with defined observation, action, and reward signals.
The framework integrates three components: a lightweight task profiler that selects reasoning and generation strategies, a reasoning module that learns verifiable observation - action mappings, and a generation module that enforces constraint-compliant outputs through validation or deterministic synthesis. We show that as the agent interacts with the environment, these components co-evolve, yielding trustworthy behavior.

**arXiv ID:** 2512.11421
</details>

<details>
<summary><strong>Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation</strong> - Lim Chien Her, Ming Yan, Yunshu Bai, Ruihao Li, Hao Zhang - [[pdf]](https://arxiv.org/pdf/2512.10501)</summary>

**Abstract:** Procedural Content Generation (PCG) offers scalable methods for algorithmically creating complex, customizable worlds. However, controlling these pipelines requires the precise configuration of opaque technical parameters. We propose a training-free architecture that utilizes LLM agents for zero-shot PCG parameter configuration. While Large Language Models (LLMs) promise a natural language interface for PCG tools, off-the-shelf models often fail to bridge the semantic gap between abstract user instructions and strict parameter specifications. Our system pairs an Actor agent with a Critic agent, enabling an iterative workflow where the system autonomously reasons over tool parameters and refines configurations to progressively align with human design preferences. We validate this approach on the generation of various 3D maps, establishing a new benchmark for instruction-following in PCG. Experiments demonstrate that our approach outperforms single-agent baselines, producing diverse and structurally valid environments from natural language descriptions. These results demonstrate that off-the-shelf LLMs can be effectively repurposed as generalized agents for arbitrary PCG tools. By shifting the burden from model training to architectural reasoning, our method offers a scalable framework for mastering complex software without task-specific fine-tuning.

**arXiv ID:** 2512.10501
</details>

<details>
<summary><strong>Achieving Olympia-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning</strong> - Haiteng Zhao, Junhao Shen, Yiming Zhang, Songyang Gao, Kuikun Liu, Tianyou Ma, Fan Zheng, Dahua Lin, Wenwei Zhang, Kai Chen - [[pdf]](https://arxiv.org/pdf/2512.10534)</summary>

**Abstract:** Large language model (LLM) agents exhibit strong mathematical problem-solving abilities and can even solve International Mathematical Olympiad (IMO) level problems with the assistance of formal proof systems. However, due to weak heuristics for auxiliary constructions, AI for geometry problem solving remains dominated by expert models such as AlphaGeometry 2, which rely heavily on large-scale data synthesis and search for both training and evaluation. In this work, we make the first attempt to build a medalist-level LLM agent for geometry and present InternGeometry. InternGeometry overcomes the heuristic limitations in geometry by iteratively proposing propositions and auxiliary constructions, verifying them with a symbolic engine, and reflecting on the engine's feedback to guide subsequent proposals. A dynamic memory mechanism enables InternGeometry to conduct more than two hundred interactions with the symbolic engine per problem. To further accelerate learning, we introduce Complexity-Boosting Reinforcement Learning (CBRL), which gradually increases the complexity of synthesized problems across training stages. Built on InternThinker-32B, InternGeometry solves 44 of 50 IMO geometry problems (2000-2024), exceeding the average gold medalist score (40.9), using only 13K training examples, just 0.004% of the data used by AlphaGeometry 2, demonstrating the potential of LLM agents on expert-level geometry tasks. InternGeometry can also propose novel auxiliary constructions for IMO problems that do not appear in human solutions. We will release the model, data, and symbolic engine to support future research.

**arXiv ID:** 2512.10534
</details>

<details>
<summary><strong>Large Language Model Agent for Modular Task Execution in Drug Discovery</strong> - Janghoon Ock, Radheesh Sharma Meda, Srivathsan Badrinarayanan, Neha S. Aluru, Achuth Chandrasekhar, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2507.02925)</summary>

**Abstract:** We present a modular framework powered by large language models (LLMs) that automates and streamlines key tasks across the early-stage computational drug discovery pipeline. By combining LLM reasoning with domain-specific tools, the framework performs biomedical data retrieval, literature-grounded question answering via retrieval-augmented generation, molecular generation, multi-property prediction, property-aware molecular refinement, and 3D protein-ligand structure generation. The agent autonomously retrieved relevant biomolecular information, including FASTA sequences, SMILES representations, and literature, and answered mechanistic questions with improved contextual accuracy compared to standard LLMs. It then generated chemically diverse seed molecules and predicted 75 properties, including ADMET-related and general physicochemical descriptors, which guided iterative molecular refinement. Across two refinement rounds, the number of molecules with QED > 0.6 increased from 34 to 55. The number of molecules satisfying empirical drug-likeness filters also rose; for example, compliance with the Ghose filter increased from 32 to 55 within a pool of 100 molecules. The framework also employed Boltz-2 to generate 3D protein-ligand complexes and provide rapid binding affinity estimates for candidate compounds. These results demonstrate that the approach effectively supports molecular screening, prioritization, and structure evaluation. Its modular design enables flexible integration of evolving tools and models, providing a scalable foundation for AI-assisted therapeutic discovery.

**arXiv ID:** 2507.02925
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (17 papers)</h2></summary>

<details>
<summary><strong>FutureWeaver: Planning Test-Time Compute for Multi-Agent Systems with Modularized Collaboration</strong> - Dongwon Jung, Peng Shi, Yi Zhang - [[pdf]](https://arxiv.org/pdf/2512.11213)</summary>

**Abstract:** Scaling test-time computation improves large language model performance without additional training. Recent work demonstrates that techniques such as repeated sampling, self-verification, and self-reflection can significantly enhance task success by allocating more inference-time compute. However, applying these techniques across multiple agents in a multi-agent system is difficult: there does not exist principled mechanisms to allocate compute to foster collaboration among agents, to extend test-time scaling to collaborative interactions, or to distribute compute across agents under explicit budget constraints. To address this gap, we propose FutureWeaver, a framework for planning and optimizing test-time compute allocation in multi-agent systems under fixed budgets. FutureWeaver introduces modularized collaboration, formalized as callable functions that encapsulate reusable multi-agent workflows. These modules are automatically derived through self-play reflection by abstracting recurring interaction patterns from past trajectories. Building on these modules, FutureWeaver employs a dual-level planning architecture that optimizes compute allocation by reasoning over the current task state while also speculating on future steps. Experiments on complex agent benchmarks demonstrate that FutureWeaver consistently outperforms baselines across diverse budget settings, validating its effectiveness for multi-agent collaboration in inference-time optimization.

**arXiv ID:** 2512.11213
</details>

<details>
<summary><strong>TriFlow: A Progressive Multi-Agent Framework for Intelligent Trip Planning</strong> - Yuxing Chen, Basem Suleiman, Qifan Chen - [[pdf]](https://arxiv.org/pdf/2512.11271)</summary>

**Abstract:** Real-world trip planning requires transforming open-ended user requests into executable itineraries under strict spatial, temporal, and budgetary constraints while aligning with user preferences. Existing LLM-based agents struggle with constraint satisfaction, tool coordination, and efficiency, often producing infeasible or costly plans. To address these limitations, we present TriFlow, a progressive multi-agent framework that unifies structured reasoning and language-based flexibility through a three-stage pipeline of retrieval, planning, and governance. By this design, TriFlow progressively narrows the search space, assembles constraint-consistent itineraries via rule-LLM collaboration, and performs bounded iterative refinement to ensure global feasibility and personalisation. Evaluations on TravelPlanner and TripTailor benchmarks demonstrated state-of-the-art results, achieving 91.1% and 97% final pass rates, respectively, with over 10x runtime efficiency improvement compared to current SOTA.

**arXiv ID:** 2512.11271
</details>

<details>
<summary><strong>AgentBalance: Backbone-then-Topology Design for Cost-Effective Multi-Agent Systems under Budget Constraints</strong> - Shuowei Cai, Yansong Ning, Hao Liu - [[pdf]](https://arxiv.org/pdf/2512.11426)</summary>

**Abstract:** Large Language Model (LLM)-based multi-agent systems (MAS) are becoming indispensable building blocks for web-scale applications such as web search, social network analytics, and online customer support, where cost-effectiveness is increasingly the primary constraint for large-scale deployment. While recent work improves MAS cost-effectiveness by shaping inter-agent communication topologies and selecting agent backbones, it rarely models and optimizes under explicit token-cost and latency budgets that reflect deployment constraints. This often leads to topology-first designs and suboptimal cost-effectiveness when budgets are binding. We present AgentBalance, a framework for constructing cost-effective MAS under explicit token-cost and latency budgets via a backbone-then-topology design. AgentBalance first performs backbone-oriented agent generation, constructing agents with heterogeneous backbones through LLM pool construction, pool selection, and role-backbone matching. It then performs adaptive MAS topology generation, guiding inter-agent communication via agent representation learning, gating, and latency-aware topology synthesis. Experiments on benchmarks with 14 candidate LLM backbones show that AgentBalance achieves up to 10% and 22% performance gains under matched token-cost and latency budgets, respectively, and yields strong AUC on performance-versus-budget curves across benchmarks. AgentBalance also functions as a plug-in for existing MAS, improving performance under the same token-cost and latency constraints, and it generalizes well to unseen LLMs for practical, budget-aware deployment. Code: this https URL

**arXiv ID:** 2512.11426
</details>

<details>
<summary><strong>Agent-Based Modular Learning for Multimodal Emotion Recognition in Human-Agent Systems</strong> - Matvey Nepomnyaschiy, Oleg Pereziabov, Anvar Tliamov, Stanislav Mikhailov, Ilya Afanasyev - [[pdf]](https://arxiv.org/pdf/2512.10975)</summary>

**Abstract:** Effective human-agent interaction (HAI) relies on accurate and adaptive perception of human emotional states. While multimodal deep learning models - leveraging facial expressions, speech, and textual cues - offer high accuracy in emotion recognition, their training and maintenance are often computationally intensive and inflexible to modality changes. In this work, we propose a novel multi-agent framework for training multimodal emotion recognition systems, where each modality encoder and the fusion classifier operate as autonomous agents coordinated by a central supervisor. This architecture enables modular integration of new modalities (e.g., audio features via emotion2vec), seamless replacement of outdated components, and reduced computational overhead during training. We demonstrate the feasibility of our approach through a proof-of-concept implementation supporting vision, audio, and text modalities, with the classifier serving as a shared decision-making agent. Our framework not only improves training efficiency but also contributes to the design of more flexible, scalable, and maintainable perception modules for embodied and virtual agents in HAI scenarios.

**arXiv ID:** 2512.10975
</details>

<details>
<summary><strong>Agile Flight Emerges from Multi-Agent Competitive Racing</strong> - Vineet Pasumarti, Lorenzo Bianchi, Antonio Loquercio - [[pdf]](https://arxiv.org/pdf/2512.11781)</summary>

**Abstract:** Through multi-agent competition and the sparse high-level objective of winning a race, we find that both agile flight (e.g., high-speed motion pushing the platform to its physical limits) and strategy (e.g., overtaking or blocking) emerge from agents trained with reinforcement learning. We provide evidence in both simulation and the real world that this approach outperforms the common paradigm of training agents in isolation with rewards that prescribe behavior, e.g., progress on the raceline, in particular when the complexity of the environment increases, e.g., in the presence of obstacles. Moreover, we find that multi-agent competition yields policies that transfer more reliably to the real world than policies trained with a single-agent progress-based reward, despite the two methods using the same simulation environment, randomization strategy, and hardware. In addition to improved sim-to-real transfer, the multi-agent policies also exhibit some degree of generalization to opponents unseen at training time. Overall, our work, following in the tradition of multi-agent competitive game-play in digital domains, shows that sparse task-level rewards are sufficient for training agents capable of advanced low-level control in the physical world.
Code: this https URL

**arXiv ID:** 2512.11781
</details>

<details>
<summary><strong>From Bits to Boardrooms: A Cutting-Edge Multi-Agent LLM Framework for Business Excellence</strong> - Zihao Wang, Junming Zhang - [[pdf]](https://arxiv.org/pdf/2508.15447)</summary>

**Abstract:** Large Language Models (LLMs) have shown promising potential in business applications, particularly in enterprise decision support and strategic planning, yet current approaches often struggle to reconcile intricate operational analyses with overarching strategic goals across diverse market environments, leading to fragmented workflows and reduced collaboration across organizational levels. This paper introduces BusiAgent, a novel multi-agent framework leveraging LLMs for advanced decision-making in complex corporate environments. BusiAgent integrates three core innovations: an extended Continuous Time Markov Decision Process (CTMDP) for dynamic agent modeling, a generalized entropy measure to optimize collaborative efficiency, and a multi-level Stackelberg game to handle hierarchical decision processes. Additionally, contextual Thompson sampling is employed for prompt optimization, supported by a comprehensive quality assurance system to mitigate errors. Extensive empirical evaluations across diverse business scenarios validate BusiAgent's efficacy, demonstrating its capacity to generate coherent, client-focused solutions that smoothly integrate granular insights with high-level strategy, significantly outperforming established approaches in both solution quality and user satisfaction. By fusing cutting-edge AI technologies with deep business insights, BusiAgent marks a substantial step forward in AI-driven enterprise decision-making, empowering organizations to navigate complex business landscapes more effectively.

**arXiv ID:** 2508.15447
</details>

<details>
<summary><strong>SDialog: A Python Toolkit for End-to-End Agent Building, User Simulation, Dialog Generation, and Evaluation</strong> - Sergio Burdisso, Séverin Baroudi, Yanis Labrak, David Grunert, Pawel Cyrta, Yiyang Chen, Srikanth Madikeri, Esaú Villatoro-Tello, Thomas Schaaf, Ricard Marxer, Petr Motlicek - [[pdf]](https://arxiv.org/pdf/2512.09142)</summary>

**Abstract:** We present SDialog, an MIT-licensed open-source Python toolkit that unifies dialog generation, evaluation and mechanistic interpretability into a single end-to-end framework for building and analyzing LLM-based conversational agents. Built around a standardized \texttt{Dialog} representation, SDialog provides: (1) persona-driven multi-agent simulation with composable orchestration for controlled, synthetic dialog generation, (2) comprehensive evaluation combining linguistic metrics, LLM-as-a-judge and functional correctness validators, (3) mechanistic interpretability tools for activation inspection and steering via feature ablation and induction, and (4) audio generation with full acoustic simulation including 3D room modeling and microphone effects. The toolkit integrates with all major LLM backends, enabling mixed-backend experiments under a unified API. By coupling generation, evaluation, and interpretability in a dialog-centric architecture, SDialog enables researchers to build, benchmark and understand conversational systems more systematically.

**arXiv ID:** 2512.09142
</details>

<details>
<summary><strong>EpiPlanAgent: Agentic Automated Epidemic Response Planning</strong> - Kangkun Mao, Fang Xu, Jinru Ding, Yidong Jiang, Yujun Yao, Yirong Chen, Junming Liu, Xiaoqin Wu, Qian Wu, Xiaoyan Huang, Jie Xu - [[pdf]](https://arxiv.org/pdf/2512.10313)</summary>

**Abstract:** Epidemic response planning is essential yet traditionally reliant on labor-intensive manual methods. This study aimed to design and evaluate EpiPlanAgent, an agent-based system using large language models (LLMs) to automate the generation and validation of digital emergency response plans. The multi-agent framework integrated task decomposition, knowledge grounding, and simulation modules. Public health professionals tested the system using real-world outbreak scenarios in a controlled evaluation. Results demonstrated that EpiPlanAgent significantly improved the completeness and guideline alignment of plans while drastically reducing development time compared to manual workflows. Expert evaluation confirmed high consistency between AI-generated and human-authored content. User feedback indicated strong perceived utility. In conclusion, EpiPlanAgent provides an effective, scalable solution for intelligent epidemic response planning, demonstrating the potential of agentic AI to transform public health preparedness.

**arXiv ID:** 2512.10313
</details>

<details>
<summary><strong>SDialog: A Python Toolkit for End-to-End Agent Building, User Simulation, Dialog Generation, and Evaluation</strong> - Sergio Burdisso, Séverin Baroudi, Yanis Labrak, David Grunert, Pawel Cyrta, Yiyang Chen, Srikanth Madikeri, Esaú Villatoro-Tello, Thomas Schaaf, Ricard Marxer, Petr Motlicek - [[pdf]](https://arxiv.org/pdf/2506.10622)</summary>

**Abstract:** We present SDialog, an MIT-licensed open-source Python toolkit that unifies dialog generation, evaluation and mechanistic interpretability into a single end-to-end framework for building and analyzing LLM-based conversational agents. Built around a standardized \texttt{Dialog} representation, SDialog provides: (1) persona-driven multi-agent simulation with composable orchestration for controlled, synthetic dialog generation, (2) comprehensive evaluation combining linguistic metrics, LLM-as-a-judge and functional correctness validators, (3) mechanistic interpretability tools for activation inspection and steering via feature ablation and induction, and (4) audio generation with full acoustic simulation including 3D room modeling and microphone effects. The toolkit integrates with all major LLM backends, enabling mixed-backend experiments under a unified API. By coupling generation, evaluation, and interpretability in a dialog-centric architecture, SDialog enables researchers to build, benchmark and understand conversational systems more systematically.

**arXiv ID:** 2506.10622
</details>

<details>
<summary><strong>CREW-WILDFIRE: Benchmarking Agentic Multi-Agent Collaborations at Scale</strong> - Jonathan Hyun, Nicholas R Waytowich, Boyuan Chen - [[pdf]](https://arxiv.org/pdf/2507.05178)</summary>

**Abstract:** Despite rapid progress in large language model (LLM)-based multi-agent systems, current benchmarks fall short in evaluating their scalability, robustness, and coordination capabilities in complex, dynamic, real-world tasks. Existing environments typically focus on small-scale, fully observable, or low-complexity domains, limiting their utility for developing and assessing next-generation multi-agent Agentic AI frameworks. We introduce CREW-Wildfire, an open-source benchmark designed to close this gap. Built atop the human-AI teaming CREW simulation platform, CREW-Wildfire offers procedurally generated wildfire response scenarios featuring large maps, heterogeneous agents, partial observability, stochastic dynamics, and long-horizon planning objectives. The environment supports both low-level control and high-level natural language interactions through modular Perception and Execution modules. We implement and evaluate several state-of-the-art LLM-based multi-agent Agentic AI frameworks, uncovering significant performance gaps that highlight the unsolved challenges in large-scale coordination, communication, spatial reasoning, and long-horizon planning under uncertainty. By providing more realistic complexity, scalable architecture, and behavioral evaluation metrics, CREW-Wildfire establishes a critical foundation for advancing research in scalable multi-agent Agentic intelligence. All code, environments, data, and baselines will be released to support future research in this emerging domain.

**arXiv ID:** 2507.05178
</details>

<details>
<summary><strong>MTTR-A: Measuring Cognitive Recovery Latency in Multi-Agent Systems</strong> - Barak Or - [[pdf]](https://arxiv.org/pdf/2511.20663)</summary>

**Abstract:** Ensuring cognitive stability in autonomous multi-agent systems (MAS) is a central challenge for large-scale, distributed AI. While existing observability tools monitor system outputs, they cannot quantify how rapidly agentic workflows recover once reasoning coherence has been lost. We adapt classical reliability metrics-Mean Time-to-Recovery (MTTR), Mean Time Between Failures (MTBF), and related ratios-into the cognitive domain, defining MTTR-A (Mean Time-to-Recovery for Agentic Systems) as a runtime measure of cognitive recovery latency. MTTR-A quantifies the time required for a MAS to detect reasoning drift and restore consistent operation, capturing the recovery of reasoning coherence rather than infrastructural repair.
A benchmark simulation using the AG~News corpus and the LangGraph orchestration framework was conducted, modeling recovery latencies across multiple reflex modes. Automated reflexes restored stability within approximately 6s on average, while human-approval interventions required about 12s. Across 200 runs, the median simulated MTTR-A was 6.21+-2.14s, MTBF=6.7+-2.14s, and NRR=0.08, demonstrating measurable runtime resilience across reflex strategies.
By formalizing recovery latency as a quantifiable property of distributed reasoning-and deriving reliability bounds linking recovery time and cognitive uptime-this work establishes a foundation for runtime dependability in agentic cognition, transforming cognitive recovery from an ad-hoc process into a standardized, interpretable performance

**arXiv ID:** 2511.20663
</details>

<details>
<summary><strong>Understanding LLM Agent Behaviours via Game Theory: Strategy Recognition, Biases and Multi-Agent Dynamics</strong> - Trung-Kiet Huynh, Duy-Minh Dao-Sy, Thanh-Bang Cao, Phong-Hao Le, Hong-Dan Nguyen, Phu-Quy Nguyen-Lam, Minh-Luan Nguyen-Vo, Hong-Phat Pham, Phu-Hoa Pham, Thien-Kim Than, Chi-Nguyen Tran, Huy Tran, Gia-Thoai Tran-Le, Alessio Buscemi, Le Hong Trang, Anh Han - [[pdf]](https://arxiv.org/pdf/2512.07462)</summary>

**Abstract:** As Large Language Models (LLMs) increasingly operate as autonomous decision-makers in interactive and multi-agent systems and human societies, understanding their strategic behaviour has profound implications for safety, coordination, and the design of AI-driven social and economic infrastructures. Assessing such behaviour requires methods that capture not only what LLMs output, but the underlying intentions that guide their decisions. In this work, we extend the FAIRGAME framework to systematically evaluate LLM behaviour in repeated social dilemmas through two complementary advances: a payoff-scaled Prisoners Dilemma isolating sensitivity to incentive magnitude, and an integrated multi-agent Public Goods Game with dynamic payoffs and multi-agent histories. These environments reveal consistent behavioural signatures across models and languages, including incentive-sensitive cooperation, cross-linguistic divergence and end-game alignment toward defection. To interpret these patterns, we train traditional supervised classification models on canonical repeated-game strategies and apply them to FAIRGAME trajectories, showing that LLMs exhibit systematic, model- and language-dependent behavioural intentions, with linguistic framing at times exerting effects as strong as architectural differences. Together, these findings provide a unified methodological foundation for auditing LLMs as strategic agents and reveal systematic cooperation biases with direct implications for AI governance, collective decision-making, and the design of safe multi-agent systems.

**arXiv ID:** 2512.07462
</details>

<details>
<summary><strong>Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving</strong> - Songyang Gao, Yuzhe Gu, Zijian Wu, Lingkai Kong, Wenwei Zhang, Zhongrui Cai, Fan Zheng, Tianyou Ma, Junhao Shen, Haiteng Zhao, Duanyang Zhang, Huilun Zhang, Kuikun Liu, Chengqi Lyu, Yanhui Duan, Chiyu Chen, Ningsheng Ma, Jianfei Gao, Han Lyu, Dahua Lin, Kai Chen - [[pdf]](https://arxiv.org/pdf/2512.10739)</summary>

**Abstract:** Large Reasoning Models (LRMs) have expanded the mathematical reasoning frontier through Chain-of-Thought (CoT) techniques and Reinforcement Learning with Verifiable Rewards (RLVR), capable of solving AIME-level problems. However, the performance of LRMs is heavily dependent on the extended reasoning context length. For solving ultra-hard problems like those in the International Mathematical Olympiad (IMO), the required reasoning complexity surpasses the space that an LRM can explore in a single round. Previous works attempt to extend the reasoning context of LRMs but remain prompt-based and built upon proprietary models, lacking systematic structures and training pipelines. Therefore, this paper introduces Intern-S1-MO, a long-horizon math agent that conducts multi-round hierarchical reasoning, composed of an LRM-based multi-agent system including reasoning, summary, and verification. By maintaining a compact memory in the form of lemmas, Intern-S1-MO can more freely explore the lemma-rich reasoning spaces in multiple reasoning stages, thereby breaking through the context constraints for IMO-level math problems. Furthermore, we propose OREAL-H, an RL framework for training the LRM using the online explored trajectories to simultaneously bootstrap the reasoning ability of LRM and elevate the overall performance of Intern-S1-MO. Experiments show that Intern-S1-MO can obtain 26 out of 35 points on the non-geometry problems of IMO2025, matching the performance of silver medalists. It also surpasses the current advanced LRMs on inference benchmarks such as HMMT2025, AIME2025, and CNMO2025. In addition, our agent officially participates in CMO2025 and achieves a score of 102/126 under the judgment of human experts, reaching the gold medal level.

**arXiv ID:** 2512.10739
</details>

<details>
<summary><strong>Evaluating Cooperative Resilience in Multiagent Systems: A Comparison Between Humans and LLMs</strong> - Manuela Chacon-Chamorro, Juan Sebastián Pinzón, Rubén Manrique, Luis Felipe Giraldo, Nicanor Quijano - [[pdf]](https://arxiv.org/pdf/2512.11689)</summary>

**Abstract:** This paper presents a comparative analysis of cooperative resilience in multi-agent systems, defined as the ability to anticipate, resist, recover from, and transform to disruptive events that affect collective well-being. We focus on mixed-motive social dilemmas instantiated as a \textit{Tragedy of the Commons} environment from the Melting Pot suite, where we systematically compare human groups and Large Language Model (LLM)-based agents, each evaluated with and without explicit communication. Cooperative resilience is assessed under a continuously disruptive condition induced by a persistent unsustainable consumption bot, together with intermittent environmental shocks implemented as stochastic removal of shared resources across scenarios. This experimental design establishes a benchmark for cooperative resilience across agent architectures and interaction modalities, constituting a key step toward systematically comparing humans and LLM-based agents. Using this framework, we find that human groups with communication achieve the highest cooperative resilience compared to all other groups. Communication also improves the resilience of LLM agents, but their performance remains below human levels. Motivated by the performance of humans, we further examine a long-horizon setting with harsher environmental conditions, where humans sustain the shared resource and maintain high resilience in diverse disruption scenarios. Together, these results suggest that human decision-making under adverse social conditions can inform the design of artificial agents that promote prosocial and resilient behaviors.

**arXiv ID:** 2512.11689
</details>

<details>
<summary><strong>Query Optimization Beyond Data Systems: The Case for Multi-Agent Systems</strong> - Zoi Kaoudi, Ioana Giurgiu - [[pdf]](https://arxiv.org/pdf/2512.11001)</summary>

**Abstract:** The proliferation of large language models (LLMs) has accelerated the adoption of agent-based workflows, where multiple autonomous agents reason, invoke functions, and collaborate to compose complex data pipelines. However, current approaches to building such agentic architectures remain largely ad hoc, lacking generality, scalability, and systematic optimization. Existing systems often rely on fixed models and single execution engines and are unable to efficiently optimize multiple agents operating over heterogeneous data sources and query engines. This paper presents a vision for a next-generation query optimization framework tailored to multi-agent workflows. We argue that optimizing these workflows can benefit from redesigning query optimization principles to account for new challenges: orchestration of diverse agents, cost efficiency under expensive LLM calls and across heterogeneous engines, and redundancy across tasks. Led by a real-world example and building on an analysis of multi-agent workflows, we outline our envisioned architecture and the main research challenges of building a multi-agent query optimization framework, which aims at enabling automated model selection, workflow composition, and execution across heterogeneous engines. This vision establishes the groundwork for query optimization in emerging multi-agent architectures and opens up a set of future research directions.

**arXiv ID:** 2512.11001
</details>

<details>
<summary><strong>Bandwidth-constrained Variational Message Encoding for Cooperative Multi-agent Reinforcement Learning</strong> - Wei Duan, Jie Lu, En Yu, Junyu Xuan - [[pdf]](https://arxiv.org/pdf/2512.11179)</summary>

**Abstract:** Graph-based multi-agent reinforcement learning (MARL) enables coordinated behavior under partial observability by modeling agents as nodes and communication links as edges. While recent methods excel at learning sparse coordination graphs-determining who communicates with whom-they do not address what information should be transmitted under hard bandwidth constraints. We study this bandwidth-limited regime and show that naive dimensionality reduction consistently degrades coordination performance. Hard bandwidth constraints force selective encoding, but deterministic projections lack mechanisms to control how compression occurs. We introduce Bandwidth-constrained Variational Message Encoding (BVME), a lightweight module that treats messages as samples from learned Gaussian posteriors regularized via KL divergence to an uninformative prior. BVME's variational framework provides principled, tunable control over compression strength through interpretable hyperparameters, directly constraining the representations used for decision-making. Across SMACv1, SMACv2, and MPE benchmarks, BVME achieves comparable or superior performance while using 67--83% fewer message dimensions, with gains most pronounced on sparse graphs where message quality critically impacts coordination. Ablations reveal U-shaped sensitivity to bandwidth, with BVME excelling at extreme ratios while adding minimal overhead.

**arXiv ID:** 2512.11179
</details>

<details>
<summary><strong>AutoFSM: A Multi-agent Framework for FSM Code Generation with IR and SystemC-Based Testing</strong> - Qiuming Luo, Yanming Lei, Kunzhong Wu, Yixuan Cao, Chengjian Liu - [[pdf]](https://arxiv.org/pdf/2512.11398)</summary>

**Abstract:** With the rapid advancement of large language models (LLMs) in code generation, their applications in hardware design are receiving growing attention. However, existing LLMs face several challenges when generating Verilog code for finite state machine (FSM) control logic, including frequent syntax errors, low debugging efficiency, and heavy reliance on test benchmarks. To address these challenges, this paper proposes AutoFSM, a multi-agent collaborative framework designed for FSM code generation tasks. AutoFSM introduces a structurally clear intermediate representation (IR) to reduce syntax error rate during code generation and provides a supporting toolchain to enable automatic translation from IR to Verilog. Furthermore, AutoFSM is the first to integrate SystemC-based modeling with automatic testbench generation, thereby improving debugging efficiency and feedback quality. To systematically evaluate the framework's performance, we construct SKT-FSM, the first hierarchical FSM benchmark in the field, comprising 67 FSM samples across different complexity levels. Experimental results show that, under the same base LLM, AutoFSM consistently outperforms the open-source framework MAGE on the SKT-FSM benchmark, achieving up to an 11.94% improvement in pass rate and up to a 17.62% reduction in syntax error rate. These results demonstrate the potential of combining LLMs with structured IR and automated testing to improve the reliability and scalability of register-transfer level (RTL) code generation.

**arXiv ID:** 2512.11398
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (18 papers)</h2></summary>

<details>
<summary><strong>CORL: Reinforcement Learning of MILP Policies Solved via Branch and Bound</strong> - Akhil S Anand, Elias Aarekol, Martin Mziray Dalseg, Magnus Stalhane, Sebastien Gros - [[pdf]](https://arxiv.org/pdf/2512.11169)</summary>

**Abstract:** Combinatorial sequential decision making problems are typically modeled as mixed integer linear programs (MILPs) and solved via branch and bound (B&B) algorithms. The inherent difficulty of modeling MILPs that accurately represent stochastic real world problems leads to suboptimal performance in the real world. Recently, machine learning methods have been applied to build MILP models for decision quality rather than how accurately they model the real world problem. However, these approaches typically rely on supervised learning, assume access to true optimal decisions, and use surrogates for the MILP gradients. In this work, we introduce a proof of concept CORL framework that end to end fine tunes an MILP scheme using reinforcement learning (RL) on real world data to maximize its operational performance. We enable this by casting an MILP solved by B&B as a differentiable stochastic policy compatible with RL. We validate the CORL method in a simple illustrative combinatorial sequential decision making example.

**arXiv ID:** 2512.11169
</details>

<details>
<summary><strong>A-LAMP: Agentic LLM-Based Framework for Automated MDP Modeling and Policy Generation</strong> - Hong Je-Gal, Chan-Bin Yi, Hyun-Suk Lee - [[pdf]](https://arxiv.org/pdf/2512.11270)</summary>

**Abstract:** Applying reinforcement learning (RL) to real-world tasks requires converting informal descriptions into a formal Markov decision process (MDP), implementing an executable environment, and training a policy agent. Automating this process is challenging due to modeling errors, fragile code, and misaligned objectives, which often impede policy training. We introduce an agentic large language model (LLM)-based framework for automated MDP modeling and policy generation (A-LAMP), that automatically translates free-form natural language task descriptions into an MDP formulation and trained policy. The framework decomposes modeling, coding, and training into verifiable stages, ensuring semantic alignment throughout the pipeline. Across both classic control and custom RL domains, A-LAMP consistently achieves higher policy generation capability than a single state-of-the-art LLM model. Notably, even its lightweight variant, which is built on smaller language models, approaches the performance of much larger models. Failure analysis reveals why these improvements occur. In addition, a case study also demonstrates that A-LAMP generates environments and policies that preserve the task's optimality, confirming its correctness and reliability.

**arXiv ID:** 2512.11270
</details>

<details>
<summary><strong>Words to Describe What I'm Feeling: Exploring the Potential of AI Agents for High Subjectivity Decisions in Advance Care Planning</strong> - Kellie Yu Hui Sim, Pin Sym Foong, Chenyu Zhao, Melanie Yi Ning Quek, Swarangi Subodh Mehta, Kenny Tsu Wei Choo - [[pdf]](https://arxiv.org/pdf/2512.11276)</summary>

**Abstract:** Serious illness can deprive patients of the capacity to speak for themselves. As populations age and caregiver networks shrink, the need for reliable support in Advance Care Planning (ACP) grows. To probe this fraught design space of using proxy agents for high-risk, high-subjectivity decisions, we built an experience prototype (\acpagent{}) and asked 15 participants in 4 workshops to train it to be their personal proxy in ACP decisions. We analysed their coping strategies and feature requests and mapped the results onto axes of agent autonomy and human control. Our findings argue for a potential new role of AI in ACP where agents act as personal advocates for individuals, building mutual intelligibility over time. We conclude with design recommendations to balance the risks and benefits of such an agent.

**arXiv ID:** 2512.11276
</details>

<details>
<summary><strong>UpBench: A Dynamically Evolving Real-World Labor-Market Agentic Benchmark Framework Built for Human-Centric AI</strong> - Darvin Yi, Teng Liu, Mattie Terzolo, Lance Hasson, Ayan Sinha, Pablo Mendes, Andrew Rabinovich - [[pdf]](https://arxiv.org/pdf/2511.12306)</summary>

**Abstract:** As large language model (LLM) agents increasingly undertake digital work, reliable frameworks are needed to evaluate their real-world competence, adaptability, and capacity for human collaboration. Existing benchmarks remain largely static, synthetic, or domain-limited, providing limited insight into how agents perform in dynamic, economically meaningful environments. We introduce UpBench, a dynamically evolving benchmark grounded in real jobs drawn from the global Upwork labor marketplace. Each task corresponds to a verified client transaction, anchoring evaluation in genuine work activity and financial outcomes. UpBench employs a rubric-based evaluation framework, in which expert freelancers decompose each job into detailed, verifiable acceptance criteria and assess AI submissions with per-criterion feedback. This structure enables fine-grained analysis of model strengths, weaknesses, and instruction-following fidelity beyond binary pass/fail metrics. Human expertise is integrated throughout the data pipeline (from job curation and rubric construction to evaluation) ensuring fidelity to real professional standards and supporting research on human-AI collaboration. By regularly refreshing tasks to reflect the evolving nature of online work, UpBench provides a scalable, human-centered foundation for evaluating agentic systems in authentic labor-market contexts, offering a path toward a collaborative framework, where AI amplifies human capability through partnership rather than replacement.

**arXiv ID:** 2511.12306
</details>

<details>
<summary><strong>ICPO: Intrinsic Confidence-Driven Group Relative Preference Optimization for Efficient Reinforcement Learning</strong> - Jinpeng Wang, Chao Li, Ting Ye, Mengyuan Zhang, Wei Liu, Jian Luan - [[pdf]](https://arxiv.org/pdf/2511.21005)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) demonstrates significant potential in enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing RLVR methods are often constrained by issues such as coarse-grained rewards, reward noise, and inefficient exploration, which lead to unstable training and entropy collapse. To address this challenge, we propose the Intrinsic Confidence-Driven Group Relative Preference Optimization method (ICPO). The intuition behind it lies in the fact that the probabilities of an LLM generating different responses can inherently and directly reflect its self-assessment of the reasoning process. Inspired by the idea of preference modeling, ICPO calculates a preference advantage score for each response by comparing the relative generation probabilities of multiple responses under the same input prompt, and integrates this score with verifiable rewards to guide the exploration process. We have discovered that the preference advantage score not only alleviates the issues of coarse-grained rewards and reward noise but also effectively curbs overconfident errors, enhances the relative superiority of undervalued high-quality responses, and prevents the model from overfitting to specific strategies. Comprehensive experiments across four general-domain benchmarks and three mathematical benchmarks demonstrate that ICPO steadily boosts reasoning compared to GRPO.

**arXiv ID:** 2511.21005
</details>

<details>
<summary><strong>SATURN: SAT-based Reinforcement Learning to Unleash LLMs Reasoning</strong> - Huanyu Liu, Ge Li, Jia Li, Hao Zhu, Kechi Zhang, Yihong Dong - [[pdf]](https://arxiv.org/pdf/2505.16368)</summary>

**Abstract:** How to design reinforcement learning (RL) tasks that effectively unleash the reasoning capability of large language models (LLMs) remains an open question. Existing RL tasks (e.g., math, programming, and constructing reasoning tasks) suffer from three key limitations: (1) Scalability. They rely heavily on human annotation or expensive LLM synthesis to generate sufficient training data. (2) Verifiability. LLMs' outputs are hard to verify automatically and reliably. (3) Controllable Difficulty. Most tasks lack fine-grained difficulty control, making it hard to train LLMs to develop reasoning ability from easy to hard.
To address these limitations, we propose Saturn, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLMs reasoning. Saturn enables scalable task construction, rule-based verification, and precise difficulty control. Saturn designs a curriculum learning pipeline that continuously improves LLMs' reasoning capability by constructing SAT tasks of increasing difficulty and training LLMs from easy to hard. To ensure stable training, we design a principled mechanism to control difficulty transitions.
We introduce Saturn-2.6k, a dataset of 2,660 SAT problems with varying difficulty. It supports the evaluation of how LLM reasoning changes with problem difficulty. We apply Saturn to DeepSeek-R1-Distill-Qwen and obtain Saturn-1.5B and Saturn-7B. We achieve several notable results: (1) On SAT problems, Saturn-1.5B and Saturn-7B achieve average pass@3 improvements of +14.0 and +28.1, respectively. (2) On math and programming tasks, Saturn-1.5B and Saturn-7B improve average scores by +4.9 and +1.8 on benchmarks (e.g., AIME, LiveCodeBench). (3) Compared to the state-of-the-art (SOTA) approach in constructing RL tasks, Saturn achieves further improvements of +8.8%. We release the source code, data, and models to support future research.

**arXiv ID:** 2505.16368
</details>

<details>
<summary><strong>Aligning Humans and Robots via Reinforcement Learning from Implicit Human Feedback</strong> - Suzie Kim, Hye-Bin Shin, Seong-Whan Lee - [[pdf]](https://arxiv.org/pdf/2507.13171)</summary>

**Abstract:** Conventional reinforcement learning (RL) ap proaches often struggle to learn effective policies under sparse reward conditions, necessitating the manual design of complex, task-specific reward functions. To address this limitation, rein forcement learning from human feedback (RLHF) has emerged as a promising strategy that complements hand-crafted rewards with human-derived evaluation signals. However, most existing RLHF methods depend on explicit feedback mechanisms such as button presses or preference labels, which disrupt the natural interaction process and impose a substantial cognitive load on the user. We propose a novel reinforcement learning from implicit human feedback (RLIHF) framework that utilizes non-invasive electroencephalography (EEG) signals, specifically error-related potentials (ErrPs), to provide continuous, implicit feedback without requiring explicit user intervention. The proposed method adopts a pre-trained decoder to transform raw EEG signals into probabilistic reward components, en abling effective policy learning even in the presence of sparse external rewards. We evaluate our approach in a simulation environment built on the MuJoCo physics engine, using a Kinova Gen2 robotic arm to perform a complex pick-and-place task that requires avoiding obstacles while manipulating target objects. The results show that agents trained with decoded EEG feedback achieve performance comparable to those trained with dense, manually designed rewards. These findings validate the potential of using implicit neural feedback for scalable and human-aligned reinforcement learning in interactive robotics.

**arXiv ID:** 2507.13171
</details>

<details>
<summary><strong>Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning</strong> - Alexander W. Goodall, Edwin Hamel-De le Court, Francesco Belardinelli - [[pdf]](https://arxiv.org/pdf/2511.10843)</summary>

**Abstract:** Many reinforcement learning algorithms, particularly those that rely on return estimates for policy improvement, can suffer from poor sample efficiency and training instability due to high-variance return estimates. In this paper we leverage new results from off-policy evaluation; it has recently been shown that well-designed behaviour policies can be used to collect off-policy data for provably lower variance return estimates. This result is surprising as it means collecting data on-policy is not variance optimal. We extend this key insight to the online reinforcement learning setting, where both policy evaluation and improvement are interleaved to learn optimal policies. Off-policy RL has been well studied (e.g., IMPALA), with correct and truncated importance weighted samples for de-biasing and managing variance appropriately. Generally these approaches are concerned with reconciling data collected from multiple workers in parallel, while the policy is updated asynchronously, mismatch between the workers and policy is corrected in a mathematically sound way. Here we consider only one worker - the behaviour policy, which is used to collect data for policy improvement, with provably lower variance return estimates. In our experiments we extend two policy-gradient methods with this regime, demonstrating better sample efficiency and performance over a diverse set of environments.

**arXiv ID:** 2511.10843
</details>

<details>
<summary><strong>CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning</strong> - Songqiao Su, Xiaofei Sun, Xiaoya Li, Albert Wang, Jiwei Li, Chris Shum - [[pdf]](https://arxiv.org/pdf/2512.02551)</summary>

**Abstract:** In this paper, we propose CUDA-L2, a system that combines large language models (LLMs) and reinforcement learning (RL) to automatically optimize Half-precision General Matrix Multiply (HGEMM) CUDA kernels. Using CUDA execution speed as the RL reward, CUDA-L2 automatically optimizes HGEMM kernels across 1,000 configurations. CUDA-L2 systematically outperforms major matmul baselines to date, from the widely-used this http URL to state-of-the-art Nvidia's closed-source libraries, i.e., cuBLAS, cuBLASLt. In offline mode, where kernels are executed consecutively without time intervals, CUDA-L2 yields +22.0% over this http URL on average; +19.2% over cuBLAS using the optimal layout configuration (normal-normal NN and transposed-normal TN); +16.8% over cuBLASLt-heuristic, which queries cuBLASLt library and selects the algorithm based on the heuristic's suggestion; and +11.4% over the most competitive cuBLASLt-AutoTuning model, which selects the fastest algorithm from up to 100 candidates from cuBLASLt's suggestions. In server mode, where kernels are executed at random intervals simulating real-time inference, the speedups further increase to +28.7%, +26.0%, +22.4%, and +15.9% for this http URL, cuBLAS, cuBLASLt-heuristic, and cuBLASLt-AutoTuning respectively. CUDA-L2 shows that even the most performance-critical, heavily-optimized kernels like HGEMM can be improved through LLM-guided RL automation by systematically exploring configuration spaces at scales impractical for humans. Project and code can be found at this http URL

**arXiv ID:** 2512.02551
</details>

<details>
<summary><strong>Confucius Code Agent: An Open-sourced AI Software Engineer at Industrial Scale</strong> - Zhaodong Wang, Zhenting Qi, Sherman Wong, Nathan Hu, Samuel Lin, Jun Ge, Erwin Gao, Yining Yang, Ben Maurer, Wenlin Chen, David Recordon, Yilun Du, Minlan Yu, Ying Zhang - [[pdf]](https://arxiv.org/pdf/2512.10398)</summary>

**Abstract:** Real-world AI software engineering demands coding agents that can reason over massive repositories, maintain durable memory across and within long sessions, and robustly coordinate complex toolchains at test time. Existing open-source coding agents provide transparency but frequently fall short when pushed to these industrial-scale workloads, while proprietary coding agents offer strong practical performance but limited extensibility, interpretability, and controllability. We present the Confucius Code Agent (CCA), an open-sourced AI software engineer that can operate at an industrial scale. CCA is built atop the Confucius SDK, an open-sourced agent development platform designed around three complementary perspectives: Agent Experience (AX), User Experience (UX), and Developer Experience (DX). The SDK introduces a unified orchestrator with hierarchical working memory for long-context reasoning, a persistent note-taking system for cross-session continual learning, and a modular extension module for robust tool use. Moreover, a meta-agent automates the synthesis, evaluation, and refinement of agent configurations through a build-test-improve loop, enabling rapid agent development on new tasks, environments, and tool stacks. Instantiated on Confucius SDK with these mechanisms, CCA delivers strong performance on real-world software engineering tasks. On SWE-Bench-Pro, CCA achieves a state-of-the-art Resolve@1 performance of 54.3%, substantially improving over prior coding agents. Together, the Confucius SDK and CCA provide a transparent, extensible, and reproducible foundation for AI agents, bridge gaps between research prototypes and production-grade systems, and support agent development and deployment at industrial scale.

**arXiv ID:** 2512.10398
</details>

<details>
<summary><strong>Multi-Objective Reinforcement Learning for Large-Scale Mixed Traffic Control</strong> - Iftekharul Islam, Weizi Li - [[pdf]](https://arxiv.org/pdf/2512.11247)</summary>

**Abstract:** Effective mixed traffic control requires balancing efficiency, fairness, and safety. Existing approaches excel at optimizing efficiency and enforcing safety constraints but lack mechanisms to ensure equitable service, resulting in systematic starvation of vehicles on low-demand approaches. We propose a hierarchical framework combining multi-objective reinforcement learning for local intersection control with strategic routing for network-level coordination. Our approach introduces a Conflict Threat Vector that provides agents with explicit risk signals for proactive conflict avoidance, and a queue parity penalty that ensures equitable service across all traffic streams. Extensive experiments on a real-world network across different robot vehicle (RV) penetration rates demonstrate substantial improvements: up to 53% reductions in average wait time, up to 86% reductions in maximum starvation, and up to 86\% reduction in conflict rate compared to baselines, while maintaining fuel efficiency. Our analysis reveals that strategic routing effectiveness scales with RV penetration, becoming increasingly valuable at higher autonomy levels. The results demonstrate that multi-objective optimization through well-curated reward functions paired with strategic RV routing yields significant benefits in fairness and safety metrics critical for equitable mixed-autonomy deployment.

**arXiv ID:** 2512.11247
</details>

<details>
<summary><strong>Annotation-Free Reinforcement Learning Query Rewriting via Verifiable Search Reward</strong> - Sungguk Cha, DongWook Kim, Taeseung Hahn, Mintae Kim, Youngsub Han, Byoung-Ki Jeon - [[pdf]](https://arxiv.org/pdf/2507.23242)</summary>

**Abstract:** Optimizing queries for Retrieval-Augmented Generation (RAG) systems poses a significant challenge, particularly across diverse modal indices. We introduce RL-QR, a novel annotation-free reinforcement learning framework for query rewriting that eliminates the need for costly human-annotated data. By leveraging verifiable search rewards derived from index-aligned synthetic queries, RL-QR overcomes human-annotation dependencies, extending its applicability to various modalities and index domains. Experimental results demonstrate the framework's robustness, achieving substantial retrieval performance gains of up to 3.9$\times$ on lexical retrievers and 3.5$\times$ on semantic retrievers on the MTEB VIDORE V2 benchmark for unstructured visual documents, along with consistent 5\% to 10\% improvements on MS MARCO v2.1 and internal industrial datasets.

**arXiv ID:** 2507.23242
</details>

<details>
<summary><strong>TECM*: A Data-Driven Assessment to Reinforcement Learning Methods and Application to Heparin Treatment Strategy for Surgical Sepsis</strong> - Jiang Liu, Yujie Li, Chan Zhou, Yihao Xie, Qilong Sun, Xin Shu, Peiwei Li, Chunyong Yang, Yiziting Zhu, Jiaqi Zhu, Yuwen Chen, Bo An, Hao Wu, Bin Yi - [[pdf]](https://arxiv.org/pdf/2512.10973)</summary>

**Abstract:** Objective: Sepsis is a life-threatening condition caused by severe infection leading to acute organ dysfunction. This study proposes a data-driven metric and a continuous reward function to optimize personalized heparin therapy in surgical sepsis patients. Methods: Data from the MIMIC-IV v1.0 and eICU v2.0 databases were used for model development and evaluation. The training cohort consisted of abdominal surgery patients receiving unfractionated heparin (UFH) after postoperative sepsis onset. We introduce a new RL-based framework: converting the discrete SOFA score to a continuous cxSOFA for more nuanced state and reward functions; Second, defining "good" or "bad" strategies based on cxSOFA by a stepwise manner; Third, proposing a Treatment Effect Comparison Matrix (TECM), analogous to a confusion matrix for classification tasks, to evaluate the treatment strategies. We applied different RL algorithms, Q-Learning, DQN, DDQN, BCQ and CQL to optimize the treatment and comprehensively evaluated the framework. Results: Among the AI-derived strategies, the cxSOFA-CQL model achieved the best performance, reducing mortality from 1.83% to 0.74% with the average hospital stay from 11.11 to 9.42 days. TECM demonstrated consistent outcomes across models, highlighting robustness. Conclusion: The proposed RL framework enables interpretable and robust optimization of heparin therapy in surgical sepsis. Continuous cxSOFA scoring and TECM-based evaluation provide nuanced treatment assessment, showing promise for improving clinical outcomes and decision-support reliability.

**arXiv ID:** 2512.10973
</details>

<details>
<summary><strong>DAPO: Design Structure-Aware Pass Ordering in High-Level Synthesis with Graph Contrastive and Reinforcement Learning</strong> - Jinming Ge, Linfeng Du, Likith Anaparty, Shangkun Li, Tingyuan Liang, Afzal Ahmad, Vivek Chaturvedi, Sharad Sinha, Zhiyao Xie, Jiang Xu, Wei Zhang - [[pdf]](https://arxiv.org/pdf/2512.11342)</summary>

**Abstract:** High-Level Synthesis (HLS) tools are widely adopted in FPGA-based domain-specific accelerator design. However, existing tools rely on fixed optimization strategies inherited from software compilations, limiting their effectiveness. Tailoring optimization strategies to specific designs requires deep semantic understanding, accurate hardware metric estimation, and advanced search algorithms -- capabilities that current approaches lack.
We propose DAPO, a design structure-aware pass ordering framework that extracts program semantics from control and data flow graphs, employs contrastive learning to generate rich embeddings, and leverages an analytical model for accurate hardware metric estimation. These components jointly guide a reinforcement learning agent to discover design-specific optimization strategies. Evaluations on classic HLS designs demonstrate that our end-to-end flow delivers a 2.36 speedup over Vitis HLS on average.

**arXiv ID:** 2512.11342
</details>

<details>
<summary><strong>Equilibrium Policy Generalization: A Reinforcement Learning Framework for Cross-Graph Zero-Shot Generalization in Pursuit-Evasion Games</strong> - Runyu Lu, Peng Zhang, Ruochuan Shi, Yuanheng Zhu, Dongbin Zhao, Yang Liu, Dong Wang, Cesare Alippi - [[pdf]](https://arxiv.org/pdf/2511.00811)</summary>

**Abstract:** Equilibrium learning in adversarial games is an important topic widely examined in the fields of game theory and reinforcement learning (RL). Pursuit-evasion game (PEG), as an important class of real-world games from the fields of robotics and security, requires exponential time to be accurately solved. When the underlying graph structure varies, even the state-of-the-art RL methods require recomputation or at least fine-tuning, which can be time-consuming and impair real-time applicability. This paper proposes an Equilibrium Policy Generalization (EPG) framework to effectively learn a generalized policy with robust cross-graph zero-shot performance. In the context of PEGs, our framework is generally applicable to both pursuer and evader sides in both no-exit and multi-exit scenarios. These two generalizability properties, to our knowledge, are the first to appear in this domain. The core idea of the EPG framework is to train an RL policy across different graph structures against the equilibrium policy for each single graph. To construct an equilibrium oracle for single-graph policies, we present a dynamic programming (DP) algorithm that provably generates pure-strategy Nash equilibrium with near-optimal time complexity. To guarantee scalability with respect to pursuer number, we further extend DP and RL by designing a grouping mechanism and a sequence model for joint policy decomposition, respectively. Experimental results show that, using equilibrium guidance and a distance feature proposed for cross-graph PEG training, the EPG framework guarantees desirable zero-shot performance in various unseen real-world graphs. Besides, when trained under an equilibrium heuristic proposed for the graphs with exits, our generalized pursuer policy can even match the performance of the fine-tuned policies from the state-of-the-art PEG methods.

**arXiv ID:** 2511.00811
</details>

<details>
<summary><strong>Architecting Large Action Models for Human-in-the-Loop Intelligent Robots</strong> - Kanisorn Sangchai, Methasit Boonpun, Withawin Kraipetchara, Paulo Garcia - [[pdf]](https://arxiv.org/pdf/2512.11620)</summary>

**Abstract:** The realization of intelligent robots, operating autonomously and interacting with other intelligent agents, human or artificial, requires the integration of environment perception, reasoning, and action. Classic Artificial Intelligence techniques for this purpose, focusing on symbolic approaches, have long-ago hit the scalability wall on compute and memory costs. Advances in Large Language Models in the past decade (neural approaches) have resulted in unprecedented displays of capability, at the cost of control, explainability, and interpretability. Large Action Models aim at extending Large Language Models to encompass the full perception, reasoning, and action cycle; however, they typically require substantially more comprehensive training and suffer from the same deficiencies in reliability. Here, we show it is possible to build competent Large Action Models by composing off-the-shelf foundation models, and that their control, interpretability, and explainability can be effected by incorporating symbolic wrappers and associated verification on their outputs, achieving verifiable neuro-symbolic solutions for intelligent robots. Our experiments on a multi-modal robot demonstrate that Large Action Model intelligence does not require massive end-to-end training, but can be achieved by integrating efficient perception models with a logic-driven core. We find that driving action execution through the generation of Planning Domain Definition Language (PDDL) code enables a human-in-the-loop verification stage that effectively mitigates action hallucinations. These results can support practitioners in the design and development of robotic Large Action Models across novel industries, and shed light on the ongoing challenges that must be addressed to ensure safety in the field.

**arXiv ID:** 2512.11620
</details>

<details>
<summary><strong>From "Thumbs Up" to "10 out of 10": Reconsidering Scalar Feedback in Interactive Reinforcement Learning</strong> - Hang Yu, Reuben M. Aronson, Katherine H. Allen, Elaine Schaertl Short - [[pdf]](https://arxiv.org/pdf/2311.10284)</summary>

**Abstract:** Learning from human feedback is an effective way to improve robotic learning in exploration-heavy tasks. Compared to the wide application of binary human feedback, scalar human feedback has been used less because it is believed to be noisy and unstable. In this paper, we compare scalar and binary feedback, and demonstrate that scalar feedback benefits learning when properly handled. We collected binary or scalar feedback respectively from two groups of crowdworkers on a robot task. We found that when considering how consistently a participant labeled the same data, scalar feedback led to less consistency than binary feedback; however, the difference vanishes if small mismatches are allowed. Additionally, scalar and binary feedback show no significant differences in their correlations with key Reinforcement Learning targets. We then introduce Stabilizing TEacher Assessment DYnamics (STEADY) to improve learning from scalar feedback. Based on the idea that scalar feedback is muti-distributional, STEADY re-constructs underlying positive and negative feedback distributions and re-scales scalar feedback based on feedback statistics. We show that models trained with \textit{scalar feedback + STEADY } outperform baselines, including binary feedback and raw scalar feedback, in a robot reaching task with non-expert human feedback. Our results show that both binary feedback and scalar feedback are dynamic, and scalar feedback is a promising signal for use in interactive Reinforcement Learning.

**arXiv ID:** 2311.10284
</details>

<details>
<summary><strong>Model-Based Lookahead Reinforcement Learning for in-hand manipulation</strong> - Alexandre Lopes, Catarina Barata, Plinio Moreno - [[pdf]](https://arxiv.org/pdf/2510.08884)</summary>

**Abstract:** In-Hand Manipulation, as many other dexterous tasks, remains a difficult challenge in robotics by combining complex dynamic systems with the capability to control and manoeuvre various objects using its actuators. This work presents the application of a previously developed hybrid Reinforcement Learning (RL) Framework to In-Hand Manipulation task, verifying that it is capable of improving the performance of the task. The model combines concepts of both Model-Free and Model-Based Reinforcement Learning, by guiding a trained policy with the help of a dynamic model and value-function through trajectory evaluation, as done in Model Predictive Control. This work evaluates the performance of the model by comparing it with the policy that will be guided. To fully explore this, various tests are performed using both fully-actuated and under-actuated simulated robotic hands to manipulate different objects for a given task. The performance of the model will also be tested for generalization tests, by changing the properties of the objects in which both the policy and dynamic model were trained, such as density and size, and additionally by guiding a trained policy in a certain object to perform the same task in a different one. The results of this work show that, given a policy with high average reward and an accurate dynamic model, the hybrid framework improves the performance of in-hand manipulation tasks for most test cases, even when the object properties are changed. However, this improvement comes at the expense of increasing the computational cost, due to the complexity of trajectory evaluation.

**arXiv ID:** 2510.08884
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
