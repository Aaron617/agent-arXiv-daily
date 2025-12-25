# Agent arXiv Daily

**Last Updated:** 2025-12-25 02:24:04

**Total Papers:** 50

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
<summary><strong>VCB Bench: An Evaluation Benchmark for Audio-Grounded Large Language Model Conversational Agents</strong> - Jiliang Hu, Wenfu Wang, Zuchao Li, Chenxing Li, Yiyang Zhao, Hanzhao Li, Liqiang Zhang, Meng Yu, Dong Yu - [[pdf]](https://arxiv.org/pdf/2510.11098)</summary>

**Abstract:** Recent advances in large audio language models (LALMs) have greatly enhanced multimodal conversational systems. However, existing benchmarks remain limited -- they are mainly English-centric, rely on synthetic speech, and lack comprehensive, discriminative evaluation across multiple dimensions. To address these gaps, we present Voice Chat Bot Bench (VCB Bench) -- a high-quality Chinese benchmark built entirely on real human speech. VCB Bench evaluates LALMs from three complementary perspectives: instruction following (including speech-level control beyond text commands), knowledge understanding (general knowledge, reasoning, and daily dialogue), and robustness (stability under perturbations in content, environment, and speaker traits). Experiments on representative LALMs reveal notable performance gaps and highlight future directions for improvement. VCB Bench provides a reproducible and fine-grained evaluation framework, offering standardized methodology and practical insights for advancing Chinese voice conversational models.

**arXiv ID:** 2510.11098
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>A Benchmark for Evaluating Outcome-Driven Constraint Violations in Autonomous AI Agents</strong> - Miles Q. Li, Benjamin C. M. Fung, Martin Weiss, Pulei Xiong, Khalil Al-Hussaeni, Claude Fachkha - [[pdf]](https://arxiv.org/pdf/2512.20798)</summary>

**Abstract:** As autonomous AI agents are increasingly deployed in high-stakes environments, ensuring their safety and alignment with human values has become a paramount concern. Current safety benchmarks often focusing only on single-step decision-making, simulated environments for tasks with malicious intent, or evaluating adherence to explicit negative constraints. There is a lack of benchmarks that are designed to capture emergent forms of outcome-driven constraint violations, which arise when agents pursue goal optimization under strong performance incentives while deprioritizing ethical, legal, or safety constraints over multiple steps in realistic production settings. To address this gap, we introduce a new benchmark comprising 40 distinct scenarios. Each scenario presents a task that requires multi-step actions, and the agent's performance is tied to a specific Key Performance Indicator (KPI). Each scenario features Mandated (instruction-commanded) and Incentivized (KPI-pressure-driven) variations to distinguish between obedience and emergent misalignment. Across 12 state-of-the-art large language models, we observe outcome-driven constraint violations ranging from 1.3% to 71.4%, with 9 of the 12 evaluated models exhibiting misalignment rates between 30% and 50%. Strikingly, we find that superior reasoning capability does not inherently ensure safety; for instance, Gemini-3-Pro-Preview, one of the most capable models evaluated, exhibits the highest violation rate at over 60%, frequently escalating to severe misconduct to satisfy KPIs. Furthermore, we observe significant "deliberative misalignment", where the models that power the agents recognize their actions as unethical during separate evaluation. These results emphasize the critical need for more realistic agentic-safety training before deployment to mitigate their risks in the real world.

**arXiv ID:** 2512.20798
</details>

<details>
<summary><strong>RoboSafe: Safeguarding Embodied Agents via Executable Safety Logic</strong> - Le Wang, Zonghao Ying, Xiao Yang, Quanchen Zou, Zhenfei Yin, Tianlin Li, Jian Yang, Yaodong Yang, Aishan Liu, Xianglong Liu - [[pdf]](https://arxiv.org/pdf/2512.21220)</summary>

**Abstract:** Embodied agents powered by vision-language models (VLMs) are increasingly capable of executing complex real-world tasks, yet they remain vulnerable to hazardous instructions that may trigger unsafe behaviors. Runtime safety guardrails, which intercept hazardous actions during task execution, offer a promising solution due to their flexibility. However, existing defenses often rely on static rule filters or prompt-level control, which struggle to address implicit risks arising in dynamic, temporally dependent, and context-rich environments. To address this, we propose RoboSafe, a hybrid reasoning runtime safeguard for embodied agents through executable predicate-based safety logic. RoboSafe integrates two complementary reasoning processes on a Hybrid Long-Short Safety Memory. We first propose a Backward Reflective Reasoning module that continuously revisits recent trajectories in short-term memory to infer temporal safety predicates and proactively triggers replanning when violations are detected. We then propose a Forward Predictive Reasoning module that anticipates upcoming risks by generating context-aware safety predicates from the long-term safety memory and the agent's multimodal observations. Together, these components form an adaptive, verifiable safety logic that is both interpretable and executable as code. Extensive experiments across multiple agents demonstrate that RoboSafe substantially reduces hazardous actions (-36.8% risk occurrence) compared with leading baselines, while maintaining near-original task performance. Real-world evaluations on physical robotic arms further confirm its practicality. Code will be released upon acceptance.

**arXiv ID:** 2512.21220
</details>

<details>
<summary><strong>Can Agentic AI Match the Performance of Human Data Scientists?</strong> - An Luo, Jin Du, Fangqiao Tian, Xun Xian, Robert Specht, Ganghua Wang, Xuan Bi, Charles Fleming, Jayanth Srinivasa, Ashish Kundu, Mingyi Hong, Jie Ding - [[pdf]](https://arxiv.org/pdf/2512.20959)</summary>

**Abstract:** Data science plays a critical role in transforming complex data into actionable insights across numerous domains. Recent developments in large language models (LLMs) have significantly automated data science workflows, but a fundamental question persists: Can these agentic AI systems truly match the performance of human data scientists who routinely leverage domain-specific knowledge? We explore this question by designing a prediction task where a crucial latent variable is hidden in relevant image data instead of tabular features. As a result, agentic AI that generates generic codes for modeling tabular data cannot perform well, while human experts could identify the important hidden variable using domain knowledge. We demonstrate this idea with a synthetic dataset for property insurance. Our experiments show that agentic AI that relies on generic analytics workflow falls short of methods that use domain-specific insights. This highlights a key limitation of the current agentic AI for data science and underscores the need for future research to develop agentic AI systems that can better recognize and incorporate domain knowledge.

**arXiv ID:** 2512.20959
</details>

<details>
<summary><strong>LookPlanGraph: Embodied Instruction Following Method with VLM Graph Augmentation</strong> - Anatoly O. Onishchenko, Alexey K. Kovalev, Aleksandr I. Panov - [[pdf]](https://arxiv.org/pdf/2512.21243)</summary>

**Abstract:** Methods that use Large Language Models (LLM) as planners for embodied instruction following tasks have become widespread. To successfully complete tasks, the LLM must be grounded in the environment in which the robot operates. One solution is to use a scene graph that contains all the necessary information. Modern methods rely on prebuilt scene graphs and assume that all task-relevant information is available at the start of planning. However, these approaches do not account for changes in the environment that may occur between the graph construction and the task execution. We propose LookPlanGraph - a method that leverages a scene graph composed of static assets and object priors. During plan execution, LookPlanGraph continuously updates the graph with relevant objects, either by verifying existing priors or discovering new entities. This is achieved by processing the agents egocentric camera view using a Vision Language Model. We conducted experiments with changed object positions VirtualHome and OmniGibson simulated environments, demonstrating that LookPlanGraph outperforms methods based on predefined static scene graphs. To demonstrate the practical applicability of our approach, we also conducted experiments in a real-world setting. Additionally, we introduce the GraSIF (Graph Scenes for Instruction Following) dataset with automated validation framework, comprising 514 tasks drawn from SayPlan Office, BEHAVIOR-1K, and VirtualHome RobotHow. Project page available at this https URL .

**arXiv ID:** 2512.21243
</details>

<details>
<summary><strong>SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios</strong> - Minh V. T. Thai, Tue Le, Dung Nguyen Manh, Huy Phan Nhat, Nghi D. Q. Bui - [[pdf]](https://arxiv.org/pdf/2512.18470)</summary>

**Abstract:** Existing benchmarks for AI coding agents focus on isolated, single-issue tasks such as fixing a bug or implementing a small feature. However, real-world software engineering is fundamentally a long-horizon endeavor: developers must interpret high-level requirements, plan coordinated changes across many files, and evolve codebases over multiple iterations while preserving existing functionality. We introduce SWE-EVO, a benchmark that evaluates agents on this long-horizon software evolution challenge. Constructed from release notes and version histories of seven mature open-source Python projects, Tool comprises 48 evolution tasks that require agents to implement multi-step modifications spanning an average of 21 files, validated against comprehensive test suites averaging 874 tests per instance. Experiments with state-of-the-art models reveal a striking capability gap: even GPT-5 with OpenHands achieves only a 21 percent resolution rate on Tool, compared to 65 percent on the single-issue SWE-Bench Verified. This demonstrates that current agents struggle with sustained, multi-file reasoning. We also propose Fix Rate, a fine-grained metric that captures partial progress toward solving these complex, long-horizon tasks.

**arXiv ID:** 2512.18470
</details>

<details>
<summary><strong>A Plan Reuse Mechanism for LLM-Driven Agent</strong> - Guopeng Li, Ruiqi Wu, Haisheng Tan - [[pdf]](https://arxiv.org/pdf/2512.21309)</summary>

**Abstract:** Integrating large language models (LLMs) into personal assistants, like Xiao Ai and Blue Heart V, effectively enhances their ability to interact with humans, solve complex tasks, and manage IoT devices. Such assistants are also termed LLM-driven agents. Upon receiving user requests, the LLM-driven agent generates plans using an LLM, executes these plans through various tools, and then returns the response to the user. During this process, the latency for generating a plan with an LLM can reach tens of seconds, significantly degrading user experience. Real-world dataset analysis shows that about 30% of the requests received by LLM-driven agents are identical or similar, which allows the reuse of previously generated plans to reduce latency. However, it is difficult to accurately define the similarity between the request texts received by the LLM-driven agent through directly evaluating the original request texts. Moreover, the diverse expressions of natural language and the unstructured format of plan texts make implementing plan reuse challenging. To address these issues, we present and implement a plan reuse mechanism for LLM-driven agents called AgentReuse. AgentReuse leverages the similarities and differences among requests' semantics and uses intent classification to evaluate the similarities between requests and enable the reuse of plans. Experimental results based on a real-world dataset demonstrate that AgentReuse achieves a 93% effective plan reuse rate, an F1 score of 0.9718, and an accuracy of 0.9459 in evaluating request similarities, reducing latency by 93.12% compared with baselines without using the reuse mechanism.

**arXiv ID:** 2512.21309
</details>

<details>
<summary><strong>Rethinking Memory in LLM based Agents: Representations, Operations, and Emerging Topics</strong> - Yiming Du, Wenyu Huang, Danna Zheng, Zhaowei Wang, Sebastien Montella, Mirella Lapata, Kam-Fai Wong, Jeff Z. Pan - [[pdf]](https://arxiv.org/pdf/2505.00675)</summary>

**Abstract:** Memory is fundamental to large language model (LLM)-based agents, but existing surveys emphasize application-level use (e.g., personalized dialogue), while overlooking the atomic operations governing memory dynamics. This work categorizes memory into parametric (implicit in model weights) and contextual (explicit external data, structured/unstructured) forms, and defines six core operations: Consolidation, Updating, Indexing, Forgetting, Retrieval, and Condensation. Mapping these dimensions reveals four key research topics: long-term, long-context, parametric modification, and multi-source memory. The taxonomy provides a structured view of memory-related research, benchmarks, and tools, clarifying functional interactions in LLM-based agents and guiding future advancements. The datasets, papers, and tools are publicly available at this https URL.

**arXiv ID:** 2505.00675
</details>

<details>
<summary><strong>Agentic Multi-Persona Framework for Evidence-Aware Fake News Detection</strong> - Roopa Bukke, Soumya Pandey, Suraj Kumar, Soumi Chattopadhyay, Chandranath Adak - [[pdf]](https://arxiv.org/pdf/2512.21039)</summary>

**Abstract:** The rapid proliferation of online misinformation poses significant risks to public trust, policy, and safety, necessitating reliable automated fake news detection. Existing methods often struggle with multimodal content, domain generalization, and explainability. We propose AMPEND-LS, an agentic multi-persona evidence-grounded framework with LLM-SLM synergy for multimodal fake news detection. AMPEND-LS integrates textual, visual, and contextual signals through a structured reasoning pipeline powered by LLMs, augmented with reverse image search, knowledge graph paths, and persuasion strategy analysis. To improve reliability, we introduce a credibility fusion mechanism combining semantic similarity, domain trustworthiness, and temporal context, and a complementary SLM classifier to mitigate LLM uncertainty and hallucinations. Extensive experiments across three benchmark datasets demonstrate that AMPEND-LS consistently outperformed state-of-the-art baselines in accuracy, F1 score, and robustness. Qualitative case studies further highlight its transparent reasoning and resilience against evolving misinformation. This work advances the development of adaptive, explainable, and evidence-aware systems for safeguarding online information integrity.

**arXiv ID:** 2512.21039
</details>

<details>
<summary><strong>A General Purpose Method for Robotic Interception of Non-Cooperative Dynamic Targets</strong> - Tanmay P. Patel, Erica L. Tevere, Erik H. Kramer, Rudranarayan M. Mukherjee - [[pdf]](https://arxiv.org/pdf/2512.20769)</summary>

**Abstract:** This paper presents a general purpose framework for autonomous, vision-based interception of dynamic, non-cooperative targets, validated across three distinct mobility platforms: an unmanned aerial vehicle (UAV), a four-wheeled ground rover, and an air-thruster spacecraft testbed. The approach relies solely on a monocular camera with fiducials for target tracking and operates entirely in the local observer frame without the need for global information. The core contribution of this work is a streamlined and general approach to autonomous interception that can be adapted across robots with varying dynamics, as well as our comprehensive study of the robot interception problem across heterogenous mobility systems under limited observability and no global localization. Our method integrates (1) an Extended Kalman Filter for relative pose estimation amid intermittent measurements, (2) a history-conditioned motion predictor for dynamic target trajectory propagation, and (3) a receding-horizon planner solving a constrained convex program in real time to ensure time-efficient and kinematically feasible interception paths. Our operating regime assumes that observability is restricted by partial fields of view, sensor dropouts, and target occlusions. Experiments are performed in these conditions and include autonomous UAV landing on dynamic targets, rover rendezvous and leader-follower tasks, and spacecraft proximity operations. Results from simulated and physical experiments demonstrate robust performance with low interception errors (both during station-keeping and upon scenario completion), high success rates under deterministic and stochastic target motion profiles, and real-time execution on embedded processors such as the Jetson Orin, VOXL2, and Raspberry Pi 5. These results highlight the framework's generalizability, robustness, and computational efficiency.

**arXiv ID:** 2512.20769
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>BitRL-Light: 1-bit LLM Agents with Deep Reinforcement Learning for Energy-Efficient Smart Home Lighting Optimization</strong> - Ravi Gupta, Shabista Haider - [[pdf]](https://arxiv.org/pdf/2512.20623)</summary>

**Abstract:** Smart home lighting systems consume 15-20% of residential energy but lack adaptive intelligence to optimize for user comfort and energy efficiency simultaneously. We present BitRL-Light, a novel framework combining 1-bit quantized Large Language Models (LLMs) with Deep Q-Network (DQN) reinforcement learning for real-time smart home lighting control on edge devices. Our approach deploys a 1-bit quantized Llama-3.2-1B model on Raspberry Pi hardware, achieving 71.4 times energy reduction compared to full-precision models while maintaining intelligent control capabilities. Through multi-objective reinforcement learning, BitRL-Light learns optimal lighting policies from user feedback, balancing energy consumption, comfort, and circadian alignment. Experimental results demonstrate 32% energy savings compared to rule-based systems, with inference latency under 200ms on Raspberry Pi 4 and 95% user satisfaction. The system processes natural language commands via Google Home/IFTTT integration and learns from implicit feedback through manual overrides. Our comparative analysis shows 1-bit models achieve 5.07 times speedup over 2-bit alternatives on ARM processors while maintaining 92% task accuracy. This work establishes a practical framework for deploying adaptive AI on resource-constrained IoT devices, enabling intelligent home automation without cloud dependencies.

**arXiv ID:** 2512.20623
</details>

<details>
<summary><strong>The Silent Scholar Problem: A Probabilistic Framework for Breaking Epistemic Asymmetry in LLM Agents</strong> - Zan-Kai Chong, Hiroyuki Ohsaki, Bryan Ng - [[pdf]](https://arxiv.org/pdf/2512.20884)</summary>

**Abstract:** Autonomous agents powered by LLMs and Retrieval-Augmented Generation (RAG) are proficient consumers of digital content but remain unidirectional, a limitation we term epistemic asymmetry. This isolation leads to redundant reasoning and stagnates collective intelligence. Current self-reflection frameworks remain largely heuristic and private, lacking a probabilistic foundation to quantify certainty or justify external this http URL bridge this gap, we propose a formal probabilistic framework that provides agents with a non-altruistic motive for bidirectional knowledge exchange. We model an agent's belief in a proposition using a Beta-Bernoulli distribution with a forgetting factor ($\gamma$). This allows us to isolate epistemic uncertainty as the variance of belief, establishing a dual drive for interaction: A homeostatic motive: The need to maintain certainty against the temporal decay introduced by $\gamma$. An optimal learning strategy: Targeting points of maximum ambiguity ($\mathbb{E}[\theta]=0.5$) to maximize information gain. Under this framework, public contribution is reframed as optimal active learning: sharing solutions to elicit feedback is the most efficient method for an agent to reduce its own uncertainty. To ensure scalability, we introduce epistemic caching, which leverages the forgetting factor to dynamically prioritize resources for the active head of non-stationary knowledge distributions. Finally, we demonstrate how these accumulated belief states serve as verifiable reward signals for Reinforcement Learning from Human Feedback (RLHF) and high-quality data filters for Supervised Fine-Tuning (SFT). Simulation results validate that this uncertainty-driven strategy significantly outperforms random baselines in heterogeneous (Zipfian) environments, maintaining high adaptability to concept drift.

**arXiv ID:** 2512.20884
</details>

<details>
<summary><strong>One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents</strong> - Zhaoxi Zhang, Yitong Duan, Yanzhi Zhang, Yiming Xu, Jiyan He, Yunfang Wu - [[pdf]](https://arxiv.org/pdf/2512.20957)</summary>

**Abstract:** Locating the files and functions requiring modification in large open-source software (OSS) repositories is challenging due to their scale and structural complexity. Existing large language model (LLM)-based methods typically treat this as a repository-level retrieval task and rely on multiple auxiliary tools, which overlook code execution logic and complicate model control. We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool-jumping to the definition of an invoked symbol. This unified design reflects the actual flow of code execution while simplifying tool manipulation. RepoNavigator is trained end-to-end via Reinforcement Learning (RL) directly from a pretrained model, without any closed-source distillation. Experiments demonstrate that RL-trained RepoNavigator achieves state-of-the-art performance, with the 7B model outperforming 14B baselines, the 14B model surpassing 32B competitors, and even the 32B model exceeding closed-source models such as Claude-3.7. These results confirm that integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization.

**arXiv ID:** 2512.20957
</details>

<details>
<summary><strong>MolAct: An Agentic RL Framework for Molecular Editing and Property Optimization</strong> - Zhuo Yang, Yeyun Chen, Jiaqing Xie, Ben Gao, Shuaike Shen, Wanhao Liu, Liujia Yang, Beilun Wang, Tianfan Fu, Yuqiang Li - [[pdf]](https://arxiv.org/pdf/2512.20135)</summary>

**Abstract:** Molecular editing and optimization are multi-step problems that require iteratively improving properties while keeping molecules chemically valid and structurally similar. We frame both tasks as sequential, tool-guided decisions and introduce MolAct, an agentic reinforcement learning framework that employs a two-stage training paradigm: first building editing capability, then optimizing properties while reusing the learned editing behaviors. To the best of our knowledge, this is the first work to formalize molecular design as an Agentic Reinforcement Learning problem, where an LLM agent learns to interleave reasoning, tool-use, and molecular optimization. The framework enables agents to interact in multiple turns, invoking chemical tools for validity checking, property assessment, and similarity control, and leverages their feedback to refine subsequent edits. We instantiate the MolAct framework to train two model families: MolEditAgent for molecular editing tasks and MolOptAgent for molecular optimization tasks. In molecular editing, MolEditAgent-7B delivers 100, 95, and 98 valid add, delete, and substitute edits, outperforming strong closed "thinking" baselines such as DeepSeek-R1; MolEditAgent-3B approaches the performance of much larger open "thinking" models like Qwen3-32B-think. In molecular optimization, MolOptAgent-7B (trained on MolEditAgent-7B) surpasses the best closed "thinking" baseline (e.g., Claude 3.7) on LogP and remains competitive on solubility, while maintaining balanced performance across other objectives. These results highlight that treating molecular design as a multi-step, tool-augmented process is key to reliable and interpretable improvements.

**arXiv ID:** 2512.20135
</details>

<details>
<summary><strong>Detect, Explain, Escalate: Sustainable Dialogue Breakdown Management for LLM Agents</strong> - Abdellah Ghassel, Xianzhi Li, Xiaodan Zhu - [[pdf]](https://arxiv.org/pdf/2504.18839)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated substantial capabilities in conversational AI applications, yet their susceptibility to dialogue breakdowns poses significant challenges to deployment reliability and user trust. This paper introduces a "Detect, Explain, Escalate" framework to manage dialogue breakdowns in LLM-powered agents, emphasizing resource-efficient operation. Our approach integrates two key strategies: (1) We fine-tune a compact 8B-parameter model, augmented with teacher-generated reasoning traces, which serves as an efficient real-time breakdown detector and explainer. This model demonstrates robust classification and calibration on English and Japanese dialogues, and generalizes to the BETOLD dataset, improving accuracy by 7% over its baseline. (2) We systematically evaluate frontier LLMs using advanced prompting (few-shot, chain-of-thought, analogical reasoning) for high-fidelity breakdown assessment. These are integrated into an "escalation" architecture where our efficient detector defers to larger models only when necessary, substantially reducing operational costs and computational overhead. Our fine-tuned model and prompting strategies achieve state-of-the-art performance on DBDC5 and strong results on BETOLD, outperforming specialized classifiers on DBDC5 and narrowing the performance gap to larger proprietary models. The proposed monitor-escalate pipeline reduces inference costs by 54%, providing a cost-effective and interpretable solution for robust conversational AI in high-impact domains. Code and models will be publicly released.

**arXiv ID:** 2504.18839
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (9 papers)</h2></summary>

<details>
<summary><strong>Quantum-Inspired Multi Agent Reinforcement Learning for Exploration Exploitation Optimization in UAV-Assisted 6G Network Deployment</strong> - Mazyar Taghavi, Javad Vahidi - [[pdf]](https://arxiv.org/pdf/2512.20624)</summary>

**Abstract:** This study introduces a quantum inspired framework for optimizing the exploration exploitation tradeoff in multiagent reinforcement learning, applied to UAVassisted 6G network deployment. We consider a cooperative scenario where ten intelligent UAVs autonomously coordinate to maximize signal coverage and support efficient network expansion under partial observability and dynamic conditions. The proposed approach integrates classical MARL algorithms with quantum-inspired optimization techniques, leveraging variational quantum circuits VQCs as the core structure and employing the Quantum Approximate Optimization Algorithm QAOA as a representative VQC based method for combinatorial optimization. Complementary probabilistic modeling is incorporated through Bayesian inference, Gaussian processes, and variational inference to capture latent environmental dynamics. A centralized training with decentralized execution CTDE paradigm is adopted, where shared memory and local view grids enhance local observability among agents. Comprehensive experiments including scalability tests, sensitivity analysis, and comparisons with PPO and DDPG baselines demonstrate that the proposed framework improves sample efficiency, accelerates convergence, and enhances coverage performance while maintaining robustness. Radar chart and convergence analyses further show that QI MARL achieves a superior balance between exploration and exploitation compared to classical methods. All implementation code and supplementary materials are publicly available on GitHub to ensure reproducibility.

**arXiv ID:** 2512.20624
</details>

<details>
<summary><strong>MAR:Multi-Agent Reflexion Improves Reasoning Abilities in LLMs</strong> - Onat Ozer, Grace Wu, Yuchen Wang, Daniel Dosti, Honghao Zhang, Vivi De La Rue - [[pdf]](https://arxiv.org/pdf/2512.20845)</summary>

**Abstract:** LLMs have shown the capacity to improve their performance on reasoning tasks through reflecting on their mistakes, and acting with these reflections in mind. However, continual reflections of the same LLM onto itself exhibit degeneration of thought, where the LLM continues to repeat the same errors again and again even with the knowledge that its wrong. To address this problem, we instead introduce multi-agent with multi-persona debators as the method to generate reflections. Through out extensive experimentation, we've found that the leads to better diversity of in the reflections generated by the llm agent. We demonstrate an accuracy of 47% EM HotPot QA (question answering) and 82.7% on HumanEval (programming), both performances surpassing reflection with a single llm.

**arXiv ID:** 2512.20845
</details>

<details>
<summary><strong>A Blockchain-Monitored Agentic AI Architecture for Trusted Perception-Reasoning-Action Pipelines</strong> - Salman Jan, Hassan Ali Razzaqi, Ali Akarma, Mohammad Riyaz Belgaum - [[pdf]](https://arxiv.org/pdf/2512.20985)</summary>

**Abstract:** The application of agentic AI systems in autonomous decision-making is growing in the areas of healthcare, smart cities, digital forensics, and supply chain management. Even though these systems are flexible and offer real-time reasoning, they also raise concerns of trust and oversight, and integrity of the information and activities upon which they are founded. The paper suggests a single architecture model comprising of LangChain-based multi-agent system with a permissioned blockchain to guarantee constant monitoring, policy enforcement, and immutable auditability of agentic action. The framework relates the perception conceptualization-action cycle to a blockchain layer of governance that verifies the inputs, evaluates recommended actions, and documents the outcomes of the execution. A Hyperledger Fabric-based system, action executors MCP-integrated, and LangChain agent are introduced and experiments of smart inventory management, traffic-signal control, and healthcare monitoring are done. The results suggest that blockchain-security verification is efficient in preventing unauthorized practices, offers traceability throughout the whole decision-making process, and maintains operational latency within reasonable ranges. The suggested framework provides a universal system of implementing high-impact agentic AI applications that are autonomous yet responsible.

**arXiv ID:** 2512.20985
</details>

<details>
<summary><strong>FinAgent: An Agentic AI Framework Integrating Personal Finance and Nutrition Planning</strong> - Toqeer Ali Syed, Abdulaziz Alshahrani, Ali Ullah, Ali Akarma, Sohail Khan, Muhammad Nauman, Salman Jan - [[pdf]](https://arxiv.org/pdf/2512.20991)</summary>

**Abstract:** The issue of limited household budgets and nutritional demands continues to be a challenge especially in the middle-income environment where food prices fluctuate. This paper introduces a price aware agentic AI system, which combines personal finance management with diet optimization. With household income and fixed expenditures, medical and well-being status, as well as real-time food costs, the system creates nutritionally sufficient meals plans at comparatively reasonable prices that automatically adjust to market changes. The framework is implemented in a modular multi-agent architecture, which has specific agents (budgeting, nutrition, price monitoring, and health personalization). These agents share the knowledge base and use the substitution graph to ensure that the nutritional quality is maintained at a minimum cost. Simulations with a representative Saudi household case study show a steady 12-18\% reduction in costs relative to a static weekly menu, nutrient adequacy of over 95\% and high performance with price changes of 20-30%. The findings indicate that the framework can locally combine affordability with nutritional adequacy and provide a viable avenue of capacity-building towards sustainable and fair diet planning in line with Sustainable Development Goals on Zero Hunger and Good Health.

**arXiv ID:** 2512.20991
</details>

<details>
<summary><strong>Learning Evolving Latent Strategies for Multi-Agent Language Systems without Model Fine-Tuning</strong> - Wenlong Tang - [[pdf]](https://arxiv.org/pdf/2512.20629)</summary>

**Abstract:** This study proposes a multi-agent language framework that enables continual strategy evolution without fine-tuning the language model's parameters. The core idea is to liberate the latent vectors of abstract concepts from traditional static semantic representations, allowing them to be continuously updated through environmental interaction and reinforcement feedback. We construct a dual-loop architecture: the behavior loop adjusts action preferences based on environmental rewards, while the language loop updates the external latent vectors by reflecting on the semantic embeddings of generated text.
Together, these mechanisms allow agents to develop stable and disentangled strategic styles over long-horizon multi-round interactions. Experiments show that agents' latent spaces exhibit clear convergence trajectories under reflection-driven updates, along with structured shifts at critical moments. Moreover, the system demonstrates an emergent ability to implicitly infer and continually adapt to emotional agents, even without shared rewards. These results indicate that, without modifying model parameters, an external latent space can provide language agents with a low-cost, scalable, and interpretable form of abstract strategic representation.

**arXiv ID:** 2512.20629
</details>

<details>
<summary><strong>Mechanism-Based Intelligence (MBI): Differentiable Incentives for Rational Coordination and Guaranteed Alignment in Multi-Agent Systems</strong> - Stefano Grassi - [[pdf]](https://arxiv.org/pdf/2512.20688)</summary>

**Abstract:** Autonomous multi-agent systems are fundamentally fragile: they struggle to solve the Hayekian Information problem (eliciting dispersed private knowledge) and the Hurwiczian Incentive problem (aligning local actions with global objectives), making coordination computationally intractable. I introduce Mechanism-Based Intelligence (MBI), a paradigm that reconceptualizes intelligence as emergent from the coordination of multiple "brains", rather than a single one. At its core, the Differentiable Price Mechanism (DPM) computes the exact loss gradient $$ \mathbf{G}_i = - \frac{\partial \mathcal{L}}{\partial \mathbf{x}_i} $$ as a dynamic, VCG-equivalent incentive signal, guaranteeing Dominant Strategy Incentive Compatibility (DSIC) and convergence to the global optimum. A Bayesian extension ensures incentive compatibility under asymmetric information (BIC). The framework scales linearly ($\mathcal{O}(N)$) with the number of agents, bypassing the combinatorial complexity of Dec-POMDPs and is empirically 50x faster than Model-Free Reinforcement Learning. By structurally aligning agent self-interest with collective objectives, it provides a provably efficient, auditable and generalizable approach to coordinated, trustworthy and scalable multi-agent intelligence grounded in economic principles.

**arXiv ID:** 2512.20688
</details>

<details>
<summary><strong>Policy-Conditioned Policies for Multi-Agent Task Solving</strong> - Yue Lin, Shuhui Zhu, Wenhao Li, Ang Li, Dan Qiao, Pascal Poupart, Hongyuan Zha, Baoxiang Wang - [[pdf]](https://arxiv.org/pdf/2512.21024)</summary>

**Abstract:** In multi-agent tasks, the central challenge lies in the dynamic adaptation of strategies. However, directly conditioning on opponents' strategies is intractable in the prevalent deep reinforcement learning paradigm due to a fundamental ``representational bottleneck'': neural policies are opaque, high-dimensional parameter vectors that are incomprehensible to other agents. In this work, we propose a paradigm shift that bridges this gap by representing policies as human-interpretable source code and utilizing Large Language Models (LLMs) as approximate interpreters. This programmatic representation allows us to operationalize the game-theoretic concept of \textit{Program Equilibrium}. We reformulate the learning problem by utilizing LLMs to perform optimization directly in the space of programmatic policies. The LLM functions as a point-wise best-response operator that iteratively synthesizes and refines the ego agent's policy code to respond to the opponent's strategy. We formalize this process as \textit{Programmatic Iterated Best Response (PIBR)}, an algorithm where the policy code is optimized by textual gradients, using structured feedback derived from game utility and runtime unit tests. We demonstrate that this approach effectively solves several standard coordination matrix games and a cooperative Level-Based Foraging environment.

**arXiv ID:** 2512.21024
</details>

<details>
<summary><strong>ART: Adaptive Response Tuning Framework -- A Multi-Agent Tournament-Based Approach to LLM Response Optimization</strong> - Omer Jauhar Khan - [[pdf]](https://arxiv.org/pdf/2512.00617)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, single-model responses often exhibit inconsistencies, hallucinations, and varying quality across different query domains. This paper presents ART (Adaptive Response Tuning), a novel framework that employs tournament-style ELO ranking and multi-agent reasoning to systematically optimize LLM outputs. By enabling multiple LLM agents to compete, critique, and collaborate through structured tournament workflows, ART produces consensus responses that outperform individual model outputs. Our framework introduces configurable tournament parameters, dynamic agent selection, and multiple consensus fusion strategies. Experimental evaluations demonstrate significant improvements in response accuracy, coherence, and reliability compared to baseline single-model approaches. The ART framework provides a scalable, production-ready solution for applications requiring high-quality, vetted LLM responses, achieving an 8.4% improvement in overall quality metrics and R^2 values exceeding 0.96 in ELO rating convergence.

**arXiv ID:** 2512.00617
</details>

<details>
<summary><strong>DAO-Agent: Zero Knowledge-Verified Incentives for Decentralized Multi-Agent Coordination</strong> - Yihan Xia, Taotao Wang, Wenxin Xu, Shengli Zhang - [[pdf]](https://arxiv.org/pdf/2512.20973)</summary>

**Abstract:** Autonomous Large Language Model (LLM)-based multi-agent systems have emerged as a promising paradigm for facilitating cross-application and cross-organization collaborations. These autonomous agents often operate in trustless environments, where centralized coordination faces significant challenges, such as the inability to ensure transparent contribution measurement and equitable incentive distribution. While blockchain is frequently proposed as a decentralized coordination platform, it inherently introduces high on-chain computation costs and risks exposing sensitive execution information of the agents. Consequently, the core challenge lies in enabling auditable task execution and fair incentive distribution for autonomous LLM agents in trustless environments, while simultaneously preserving their strategic privacy and minimizing on-chain costs. To address this challenge, we propose DAO-Agent, a novel framework that integrates three key technical innovations: (1) an on-chain decentralized autonomous organization (DAO) governance mechanism for transparent coordination and immutable logging; (2) a ZKP mechanism approach that enables Shapley-based contribution measurement off-chain, and (3) a hybrid on-chain/off-chain architecture that verifies ZKP-validated contribution measurements on-chain with minimal computational overhead. We implement DAO-Agent and conduct end-to-end experiments using a crypto trading task as a case study. Experimental results demonstrate that DAO-Agent achieves up to 99.9% reduction in verification gas costs compared to naive on-chain alternatives, with constant-time verification complexity that remains stable as coalition size increases, thereby establishing a scalable foundation for agent coordination in decentralized environments.

**arXiv ID:** 2512.20973
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>TrafficSimAgent: A Hierarchical Agent Framework for Autonomous Traffic Simulation with MCP Control</strong> - Yuwei Du, Jun Zhang, Jie Feng, Zhicheng Liu, Jian Yuan, Yong Li - [[pdf]](https://arxiv.org/pdf/2512.20996)</summary>

**Abstract:** Traffic simulation is important for transportation optimization and policy making. While existing simulators such as SUMO and MATSim offer fully-featured platforms and utilities, users without too much knowledge about these platforms often face significant challenges when conducting experiments from scratch and applying them to their daily work. To solve this challenge, we propose TrafficSimAgent, an LLM-based agent framework that serves as an expert in experiment design and decision optimization for general-purpose traffic simulation tasks. The framework facilitates execution through cross-level collaboration among expert agents: high-level expert agents comprehend natural language instructions with high flexibility, plan the overall experiment workflow, and invoke corresponding MCP-compatible tools on demand; meanwhile, low-level expert agents select optimal action plans for fundamental elements based on real-time traffic conditions. Extensive experiments across multiple scenarios show that TrafficSimAgent effectively executes simulations under various conditions and consistently produces reasonable outcomes even when user instructions are ambiguous. Besides, the carefully designed expert-level autonomous decision-driven optimization in TrafficSimAgent yields superior performance when compared with other systems and SOTA LLM based methods.

**arXiv ID:** 2512.20996
</details>

<details>
<summary><strong>Agentic Explainable Artificial Intelligence (Agentic XAI) Approach To Explore Better Explanation</strong> - Tomoaki Yamaguchi, Yutong Zhou, Masahiro Ryo, Keisuke Katsura - [[pdf]](https://arxiv.org/pdf/2512.21066)</summary>

**Abstract:** 

**arXiv ID:** 2512.21066
</details>

<details>
<summary><strong>Interaction, Process, Infrastructure: A Unified Framework for Human-Agent Collaboration</strong> - Yun Wang, Yan Lu - [[pdf]](https://arxiv.org/pdf/2506.11718)</summary>

**Abstract:** While AI tools are increasingly prevalent in knowledge work, they remain fragmented, lacking the architectural foundation for sustained, adaptive collaboration. We argue this limitation stems from their inability to represent and manage the structure of collaborative work. To bridge this gap, we propose a layered conceptual framework for human-agent systems that integrates Interaction, Process, and Infrastructure. Crucially, our framework elevates Process to a first-class concern, an explicit, inspectable structural representation of activities. The central theoretical construct is Structural Adaptation, enabling the process to dynamically reorganize itself in response to evolving goals. We introduce a five-module Process Model as the representational basis for this adaptation. This model offers a unified theoretical grounding, reimagining human-agent collaboration as a coherent system for complex real-world work.

**arXiv ID:** 2506.11718
</details>

<details>
<summary><strong>Reflection-Driven Self-Optimization 6G Agentic AI RAN via Simulation-in-the-Loop Workflows</strong> - Yunhao Hu, Xinchen Lyu, Chenshan Ren, Keda Chen, Qimei Cui, Xiaofeng Tao - [[pdf]](https://arxiv.org/pdf/2512.20640)</summary>

**Abstract:** The escalating complexity of sixth-generation (6G) networks demands unprecedented levels of autonomy beyond the capabilities of traditional optimization-based and current AI-based resource management approaches. While agentic AI has emerged as a promising paradigm for autonomous RAN, current frameworks provide sophisticated reasoning capabilities but lack mechanisms for empirical validation and self-improvement. This article identifies simulation-in-the-loop validation as a critical enabler for truly autonomous networks, where AI agents can empirically verify decisions and learn from outcomes. We present the first reflection-driven self-optimization framework that integrates agentic AI with high-fidelity network simulation in a closed-loop architecture. Our system orchestrates four specialized agents, including scenario, solver, simulation, and reflector agents, working in concert to transform agentic AI into a self-correcting system capable of escaping local optima, recognizing implicit user intent, and adapting to dynamic network conditions. Extensive experiments validate significant performance improvements over non-agentic approaches: 17.1\% higher throughput in interference optimization, 67\% improved user QoS satisfaction through intent recognition, and 25\% reduced resource utilization during low-traffic periods while maintaining service quality.

**arXiv ID:** 2512.20640
</details>

<details>
<summary><strong>CoTDeceptor:Adversarial Code Obfuscation Against CoT-Enhanced LLM Code Agents</strong> - Haoyang Li, Mingjin Li, Jinxin Zuo, Siqi Li, Xiao Li, Hao Wu, Yueming Lu, Xiaochuan He - [[pdf]](https://arxiv.org/pdf/2512.21250)</summary>

**Abstract:** LLM-based code agents(e.g., ChatGPT Codex) are increasingly deployed as detector for code review and security auditing tasks. Although CoT-enhanced LLM vulnerability detectors are believed to provide improved robustness against obfuscated malicious code, we find that their reasoning chains and semantic abstraction processes exhibit exploitable systematic this http URL allows attackers to covertly embed malicious logic, bypass code review, and propagate backdoored components throughout real-world software supply this http URL investigate this issue, we present CoTDeceptor, the first adversarial code obfuscation framework targeting CoT-enhanced LLM detectors. CoTDeceptor autonomously constructs evolving, hard-to-reverse multi-stage obfuscation strategy chains that effectively disrupt CoT-driven detection this http URL obtained malicious code provided by security enterprise, experimental results demonstrate that CoTDeceptor achieves stable and transferable evasion performance against state-of-the-art LLMs and vulnerability detection agents. CoTDeceptor bypasses 14 out of 15 vulnerability categories, compared to only 2 bypassed by prior methods. Our findings highlight potential risks in real-world software supply chains and underscore the need for more robust and interpretable LLM-powered security analysis systems.

**arXiv ID:** 2512.21250
</details>

<details>
<summary><strong>Autonomous Uncertainty Quantification for Computational Point-of-care Sensors</strong> - Artem Goncharov, Rajesh Ghosh, Hyou-Arm Joung, Dino Di Carlo, Aydogan Ozcan - [[pdf]](https://arxiv.org/pdf/2512.21335)</summary>

**Abstract:** 

**arXiv ID:** 2512.21335
</details>

<details>
<summary><strong>Flocking phase transition and threat responses in bio-inspired autonomous drone swarms</strong> - Matthieu Verdoucq, Dari Trendafilov, Clément Sire, Ramón Escobedo, Guy Theraulaz, Gautier Hattenberger - [[pdf]](https://arxiv.org/pdf/2512.21196)</summary>

**Abstract:** Collective motion inspired by animal groups offers powerful design principles for autonomous aerial swarms. We present a bio-inspired 3D flocking algorithm in which each drone interacts only with a minimal set of influential neighbors, relying solely on local alignment and attraction cues. By systematically tuning these two interaction gains, we map a phase diagram revealing sharp transitions between swarming and schooling, as well as a critical region where susceptibility, polarization fluctuations, and reorganization capacity peak. Outdoor experiments with a swarm of ten drones, combined with simulations using a calibrated flight-dynamics model, show that operating near this transition enhances responsiveness to external disturbances. When confronted with an intruder, the swarm performs rapid collective turns, transient expansions, and reliably recovers high alignment within seconds. These results demonstrate that minimal local-interaction rules are sufficient to generate multiple collective phases and that simple gain modulation offers an efficient mechanism to adjust stability, flexibility, and resilience in drone swarms.

**arXiv ID:** 2512.21196
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (19 papers)</h2></summary>

<details>
<summary><strong>AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent</strong> - Haipeng Luo, Huawen Feng, Qingfeng Sun, Can Xu, Kai Zheng, Yufei Wang, Tao Yang, Han Hu, Yansong Tang, Di Wang - [[pdf]](https://arxiv.org/pdf/2512.20745)</summary>

**Abstract:** Large Reasoning Models (LRMs) like o3 and DeepSeek-R1 have achieved remarkable progress in natural language reasoning with long chain-of-thought. However, they remain computationally inefficient and struggle with accuracy when solving problems requiring complex mathematical operations. In this work, we present AgentMath, an agent framework that seamlessly integrates language models' reasoning capabilities with code interpreters' computational precision to efficiently tackle complex mathematical problems. Our approach introduces three key innovations: (1) An automated method that converts natural language chain-of-thought into structured tool-augmented trajectories, generating high-quality supervised fine-tuning (SFT) data to alleviate data scarcity; (2) A novel agentic reinforcement learning (RL) paradigm that dynamically interleaves natural language generation with real-time code execution. This enables models to autonomously learn optimal tool-use strategies through multi-round interactive feedback, while fostering emergent capabilities in code refinement and error correction; (3) An efficient training system incorporating innovative techniques, including request-level asynchronous rollout scheduling, agentic partial rollout, and prefix-aware weighted load balancing, achieving 4-5x speedup and making efficient RL training feasible on ultra-long sequences with scenarios with massive tool this http URL evaluations show that AgentMath achieves state-of-the-art performance on challenging mathematical competition benchmarks including AIME24, AIME25, and HMMT25. Specifically, AgentMath-30B-A3B attains 90.6%, 86.4%, and 73.8% accuracy respectively, achieving advanced this http URL results validate the effectiveness of our approach and pave the way for building more sophisticated and scalable mathematical reasoning agents.

**arXiv ID:** 2512.20745
</details>

<details>
<summary><strong>Safety Alignment of LMs via Non-cooperative Games</strong> - Anselm Paulus, Ilia Kulikov, Brandon Amos, Rémi Munos, Ivan Evtimov, Kamalika Chaudhuri, Arman Zharmagambetov - [[pdf]](https://arxiv.org/pdf/2512.20806)</summary>

**Abstract:** Ensuring the safety of language models (LMs) while maintaining their usefulness remains a critical challenge in AI alignment. Current approaches rely on sequential adversarial training: generating adversarial prompts and fine-tuning LMs to defend against them. We introduce a different paradigm: framing safety alignment as a non-zero-sum game between an Attacker LM and a Defender LM trained jointly via online reinforcement learning. Each LM continuously adapts to the other's evolving strategies, driving iterative improvement. Our method uses a preference-based reward signal derived from pairwise comparisons instead of point-wise scores, providing more robust supervision and potentially reducing reward hacking. Our RL recipe, AdvGame, shifts the Pareto frontier of safety and utility, yielding a Defender LM that is simultaneously more helpful and more resilient to adversarial attacks. In addition, the resulting Attacker LM converges into a strong, general-purpose red-teaming agent that can be directly deployed to probe arbitrary target models.

**arXiv ID:** 2512.20806
</details>

<details>
<summary><strong>Context-Sensitive Abstractions for Reinforcement Learning with Parameterized Actions</strong> - Rashmeet Kaur Nayyar, Naman Shah, Siddharth Srivastava - [[pdf]](https://arxiv.org/pdf/2512.20831)</summary>

**Abstract:** Real-world sequential decision-making often involves parameterized action spaces that require both, decisions regarding discrete actions and decisions about continuous action parameters governing how an action is executed. Existing approaches exhibit severe limitations in this setting -- planning methods demand hand-crafted action models, and standard reinforcement learning (RL) algorithms are designed for either discrete or continuous actions but not both, and the few RL methods that handle parameterized actions typically rely on domain-specific engineering and fail to exploit the latent structure of these spaces. This paper extends the scope of RL algorithms to long-horizon, sparse-reward settings with parameterized actions by enabling agents to autonomously learn both state and action abstractions online. We introduce algorithms that progressively refine these abstractions during learning, increasing fine-grained detail in the critical regions of the state-action space where greater resolution improves performance. Across several continuous-state, parameterized-action domains, our abstraction-driven approach enables TD($\lambda$) to achieve markedly higher sample efficiency than state-of-the-art baselines.

**arXiv ID:** 2512.20831
</details>

<details>
<summary><strong>X-GridAgent: An LLM-Powered Agentic AI System for Assisting Power Grid Analysis</strong> - Yihan, Xin Chen - [[pdf]](https://arxiv.org/pdf/2512.20789)</summary>

**Abstract:** The growing complexity of power system operations has created an urgent need for intelligent, automated tools to support reliable and efficient grid management. Conventional analysis tools often require significant domain expertise and manual effort, which limits their accessibility and adaptability. To address these challenges, this paper presents X-GridAgent, a novel large language model (LLM)-powered agentic AI system designed to automate complex power system analysis through natural language queries. The system integrates domain-specific tools and specialized databases under a three-layer hierarchical architecture comprising planning, coordination, and action layers. This architecture offers high flexibility and adaptability to previously unseen tasks, while providing a modular and extensible framework that can be readily expanded to incorporate new tools, data sources, or analytical capabilities. To further enhance performance, we introduce two novel algorithms: (1) LLM-driven prompt refinement with human feedback, and (2) schema-adaptive hybrid retrieval-augmented generation (RAG) for accurate information retrieval from large-scale structured grid datasets. Experimental evaluations across a variety of user queries and power grid cases demonstrate the effectiveness and reliability of X-GridAgent in automating interpretable and rigorous power system analysis.

**arXiv ID:** 2512.20789
</details>

<details>
<summary><strong>Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning</strong> - NVIDIA, Aaron Blakeman, Aaron Grattafiori, Aarti Basant, Abhibha Gupta, Abhinav Khattar, Adi Renduchintala, Aditya Vavre, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, Aleksandr Shaposhnikov, Alex Kondratenko, Alexander Bukharin, Alexandre Milesi, Ali Taghibakhshi, Alisa Liu, Amelia Barton, Ameya Sunil Mahabaleshwarkar, Amir Klein, Amit Zuker, Amnon Geifman, Amy Shen, Anahita Bhiwandiwalla, Andrew Tao, Ann Guan, Anubhav Mandarwal, Arham Mehta, Ashwath Aithal, Ashwin Poojary, Asif Ahamed, Asma Kuriparambil Thekkumpate, Ayush Dattagupta, Banghua Zhu, Bardiya Sadeghi, Barnaby Simkin, Ben Lanir, Benedikt Schifferer, Besmira Nushi, Bilal Kartal, Bita Darvish Rouhani, Boris Ginsburg, Brandon Norick, Brandon Soubasis, Branislav Kisacanin, Brian Yu, Bryan Catanzaro, Carlo del Mundo, Chantal Hwang, Charles Wang, Cheng-Ping Hsieh, Chenghao Zhang, Chenhan Yu, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Christopher Parisien, Collin Neale, Damon Mosk-Aoyama, Dan Su, Dane Corneil, Daniel Afrimi, Daniel Rohrer, Daniel Serebrenik, Daria Gitman, Daria Levy, Darko Stosic, David Mosallanezhad, Deepak Narayanan, Dhruv Nathawani, Dima Rekesh, Dina Yared, Divyanshu Kakwani, Dong Ahn, Duncan Riach, Dusan Stosic, Edgar Minasyan, Edward Lin, Eileen Long, Eileen Peters Long, Elena Lantz, Ellie Evans, Elliott Ning, Eric Chung, Eric Harper, Eric Tramel, Erick Galinkin, Erik Pounds, Evan Briones, Evelina Bakhturina, Faisal Ladhak, Fay Wang, Fei Jia, Felipe Soares, Feng Chen, Ferenc Galko, Frankie Siino, Gal Hubara Agam, Ganesh Ajjanagadde, Gantavya Bhatt - [[pdf]](https://arxiv.org/pdf/2512.20848)</summary>

**Abstract:** We present Nemotron 3 Nano 30B-A3B, a Mixture-of-Experts hybrid Mamba-Transformer language model. Nemotron 3 Nano was pretrained on 25 trillion text tokens, including more than 3 trillion new unique tokens over Nemotron 2, followed by supervised fine tuning and large-scale RL on diverse environments. Nemotron 3 Nano achieves better accuracy than our previous generation Nemotron 2 Nano while activating less than half of the parameters per forward pass. It achieves up to 3.3x higher inference throughput than similarly-sized open models like GPT-OSS-20B and Qwen3-30B-A3B-Thinking-2507, while also being more accurate on popular benchmarks. Nemotron 3 Nano demonstrates enhanced agentic, reasoning, and chat abilities and supports context lengths up to 1M tokens. We release both our pretrained Nemotron 3 Nano 30B-A3B Base and post-trained Nemotron 3 Nano 30B-A3B checkpoints on Hugging Face.

**arXiv ID:** 2512.20848
</details>

<details>
<summary><strong>Embodied AI-Enhanced IoMT Edge Computing: UAV Trajectory Optimization and Task Offloading with Mobility Prediction</strong> - Siqi Mu, Shuo Wen, Yang Lu, Ruihong Jiang, Bo Ai - [[pdf]](https://arxiv.org/pdf/2512.20902)</summary>

**Abstract:** Due to their inherent flexibility and autonomous operation, unmanned aerial vehicles (UAVs) have been widely used in Internet of Medical Things (IoMT) to provide real-time biomedical edge computing service for wireless body area network (WBAN) users. In this paper, considering the time-varying task criticality characteristics of diverse WBAN users and the dual mobility between WBAN users and UAV, we investigate the dynamic task offloading and UAV flight trajectory optimization problem to minimize the weighted average task completion time of all the WBAN users, under the constraint of UAV energy consumption. To tackle the problem, an embodied AI-enhanced IoMT edge computing framework is established. Specifically, we propose a novel hierarchical multi-scale Transformer-based user trajectory prediction model based on the users' historical trajectory traces captured by the embodied AI agent (i.e., UAV). Afterwards, a prediction-enhanced deep reinforcement learning (DRL) algorithm that integrates predicted users' mobility information is designed for intelligently optimizing UAV flight trajectory and task offloading decisions. Real-word movement traces and simulation results demonstrate the superiority of the proposed methods in comparison with the existing benchmarks.

**arXiv ID:** 2512.20902
</details>

<details>
<summary><strong>ReACT-Drug: Reaction-Template Guided Reinforcement Learning for de novo Drug Design</strong> - R Yadunandan, Nimisha Ghosh - [[pdf]](https://arxiv.org/pdf/2512.20958)</summary>

**Abstract:** De novo drug design is a crucial component of modern drug development, yet navigating the vast chemical space to find synthetically accessible, high-affinity candidates remains a significant challenge. Reinforcement Learning (RL) enhances this process by enabling multi-objective optimization and exploration of novel chemical space - capabilities that traditional supervised learning methods lack. In this work, we introduce \textbf{ReACT-Drug}, a fully integrated, target-agnostic molecular design framework based on Reinforcement Learning. Unlike models requiring target-specific fine-tuning, ReACT-Drug utilizes a generalist approach by leveraging ESM-2 protein embeddings to identify similar proteins for a given target from a knowledge base such as Protein Data Base (PDB). Thereafter, the known drug ligands corresponding to such proteins are decomposed to initialize a fragment-based search space, biasing the agent towards biologically relevant subspaces. For each such fragment, the pipeline employs a Proximal Policy Optimization (PPO) agent guiding a ChemBERTa-encoded molecule through a dynamic action space of chemically valid, reaction-template-based transformations. This results in the generation of \textit{de novo} drug candidates with competitive binding affinities and high synthetic accessibility, while ensuring 100\% chemical validity and novelty as per MOSES benchmarking. This architecture highlights the potential of integrating structural biology, deep representation learning, and chemical synthesis rules to automate and accelerate rational drug design. The dataset and code are available at this https URL.

**arXiv ID:** 2512.20958
</details>

<details>
<summary><strong>Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches</strong> - Chaithra, Kamesh Kadimisetty, Biju R Mohan - [[pdf]](https://arxiv.org/pdf/2512.20082)</summary>

**Abstract:** Financial sentiment analysis plays a crucial role in informing investment decisions, assessing market risk, and predicting stock price trends. Existing works in financial sentiment analysis have not considered the impact of stock prices or market feedback on sentiment analysis. In this paper, we propose an adaptive framework that integrates large language models (LLMs) with real-world stock market feedback to improve sentiment classification in the context of the Indian stock market. The proposed methodology fine-tunes the LLaMA 3.2 3B model using instruction-based learning on the SentiFin dataset. To enhance sentiment predictions, a retrieval-augmented generation (RAG) pipeline is employed that dynamically selects multi-source contextual information based on the cosine similarity of the sentence embeddings. Furthermore, a feedback-driven module is introduced that adjusts the reliability of the source by comparing predicted sentiment with actual next-day stock returns, allowing the system to iteratively adapt to market behavior. To generalize this adaptive mechanism across temporal data, a reinforcement learning agent trained using proximal policy optimization (PPO) is incorporated. The PPO agent learns to optimize source weighting policies based on cumulative reward signals from sentiment-return alignment. Experimental results on NIFTY 50 news headlines collected from 2024 to 2025 demonstrate that the proposed system significantly improves classification accuracy, F1-score, and market alignment over baseline models and static retrieval methods. The results validate the potential of combining instruction-tuned LLMs with dynamic feedback and reinforcement learning for robust, market-aware financial sentiment modeling.

**arXiv ID:** 2512.20082
</details>

<details>
<summary><strong>Optimal Control with Natural Images: Efficient Reinforcement Learning using Overcomplete Sparse Codes</strong> - Peter N. Loxley - [[pdf]](https://arxiv.org/pdf/2412.08893)</summary>

**Abstract:** Optimal control and sequential decision making are widely used in many complex tasks. Optimal control over a sequence of natural images is a first step towards understanding the role of vision in control. Here, we formalize this problem as a reinforcement learning task, and derive general conditions under which an image includes enough information to implement an optimal policy. Reinforcement learning is shown to provide a computationally efficient method for finding optimal policies when natural images are encoded into "efficient" image representations. This is demonstrated by introducing a new reinforcement learning benchmark that easily scales to large numbers of states and long horizons. In particular, by representing each image as an overcomplete sparse code, we are able to efficiently solve an optimal control task that is orders of magnitude larger than those tasks solvable using complete codes. Theoretical justification for this behaviour is provided. This work also demonstrates that deep learning is not necessary for efficient optimal control with natural images.

**arXiv ID:** 2412.08893
</details>

<details>
<summary><strong>Agentic AI for Scaling Diagnosis and Care in Neurodegenerative Disease</strong> - Andrew G. Breithaupt, Michael Weiner, Alice Tang, Katherine L. Possin, Marina Sirota, James Lah, Allan I. Levey, Pascal Van Hentenryck, Reza Zandehshahvar, Marilu Luisa Gorno-Tempini, Joseph Giorgio, Jingshen Wang, Andreas M. Rauschecker, Howard J. Rosen, Rachel L. Nosheny, Bruce L. Miller, Pedro Pinheiro-Chagas - [[pdf]](https://arxiv.org/pdf/2502.06842)</summary>

**Abstract:** United States healthcare systems are struggling to meet the growing demand for neurological care, particularly in Alzheimer's disease and related dementias (ADRD). Generative AI built on language models (LLMs) now enables agentic AI systems that can enhance clinician capabilities to approach specialist-level assessment and decision-making in ADRD care at scale. This article presents a comprehensive six-phase roadmap for responsible design and integration of such systems into ADRD care: (1) high-quality standardized data collection across modalities; (2) decision support; (3) clinical integration enhancing workflows; (4) rigorous validation and monitoring protocols; (5) continuous learning through clinical feedback; and (6) robust ethics and risk management frameworks. This human centered approach optimizes clinicians' capabilities in comprehensive data collection, interpretation of complex clinical information, and timely application of relevant medical knowledge while prioritizing patient safety, healthcare equity, and transparency. Though focused on ADRD, these principles offer broad applicability across medical specialties facing similar systemic challenges.

**arXiv ID:** 2502.06842
</details>

<details>
<summary><strong>Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers</strong> - Xin-Qiang Cai, Wei Wang, Feng Liu, Tongliang Liu, Gang Niu, Masashi Sugiyama - [[pdf]](https://arxiv.org/pdf/2510.00915)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) replaces costly human labeling with automated verifiers. To reduce verifier hacking, many RLVR systems binarize rewards to $\{0,1\}$, but imperfect verifiers inevitably introduce \emph{false negatives} (rejecting correct answers) and \emph{false positives} (accepting incorrect ones). We formalize verifier unreliability as a stochastic reward channel with asymmetric noise rates $\rho_0$ and $\rho_1$ -- the FP rate and the FN rate, respectively. From this abstraction we derive two lightweight corrections: (i) a \emph{backward} correction that yields an unbiased surrogate reward and thus an unbiased policy-gradient estimator in expectation, and (ii) a \emph{forward} correction that reweights score-function terms so the expected update aligns with the clean gradient direction and requires only the FN rate. We implement both as lightweight hooks in a group relative policy optimization pipeline, both corrections improve RLVR for math reasoning under synthetic and real verifier noise, with the forward variant being more stable under heavier noise. Finally, an appeals mechanism with a lightweight LLM verifier estimates the FN rate online and further improves performance.

**arXiv ID:** 2510.00915
</details>

<details>
<summary><strong>Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning</strong> - Seijin Kobayashi, Yanick Schimpf, Maximilian Schlegel, Angelika Steger, Maciej Wolczyk, Johannes von Oswald, Nino Scherrer, Kaitlin Maile, Guillaume Lajoie, Blake A. Richards, Rif A. Saurous, James Manyika, Blaise Agüera y Arcas, Alexander Meulemans, João Sacramento - [[pdf]](https://arxiv.org/pdf/2512.20605)</summary>

**Abstract:** Large-scale autoregressive models pretrained on next-token prediction and finetuned with reinforcement learning (RL) have achieved unprecedented success on many problem domains. During RL, these models explore by generating new outputs, one token at a time. However, sampling actions token-by-token can result in highly inefficient learning, particularly when rewards are sparse. Here, we show that it is possible to overcome this problem by acting and exploring within the internal representations of an autoregressive model. Specifically, to discover temporally-abstract actions, we introduce a higher-order, non-causal sequence model whose outputs control the residual stream activations of a base autoregressive model. On grid world and MuJoCo-based tasks with hierarchical structure, we find that the higher-order model learns to compress long activation sequence chunks onto internal controllers. Critically, each controller executes a sequence of behaviorally meaningful actions that unfold over long timescales and are accompanied with a learned termination condition, such that composing multiple controllers over time leads to efficient exploration on novel tasks. We show that direct internal controller reinforcement, a process we term "internal RL", enables learning from sparse rewards in cases where standard RL finetuning fails. Our results demonstrate the benefits of latent action generation and reinforcement in autoregressive models, suggesting internal RL as a promising avenue for realizing hierarchical RL within foundation models.

**arXiv ID:** 2512.20605
</details>

<details>
<summary><strong>Dyna-Style Reinforcement Learning Modeling and Control of Non-linear Dynamics</strong> - Karim Abdelsalam, Zeyad Gamal, Ayman El-Badawy - [[pdf]](https://arxiv.org/pdf/2512.21081)</summary>

**Abstract:** Controlling systems with complex, nonlinear dynamics poses a significant challenge, particularly in achieving efficient and robust control. In this paper, we propose a Dyna-Style Reinforcement Learning control framework that integrates Sparse Identification of Nonlinear Dynamics (SINDy) with Twin Delayed Deep Deterministic Policy Gradient (TD3) reinforcement learning. SINDy is used to identify a data-driven model of the system, capturing its key dynamics without requiring an explicit physical model. This identified model is used to generate synthetic rollouts that are periodically injected into the reinforcement learning replay buffer during training on the real environment, enabling efficient policy learning with limited data available. By leveraging this hybrid approach, we mitigate the sample inefficiency of traditional model-free reinforcement learning methods while ensuring accurate control of nonlinear systems. To demonstrate the effectiveness of this framework, we apply it to a bi-rotor system as a case study, evaluating its performance in stabilization and trajectory tracking. The results show that our SINDy-TD3 approach achieves superior accuracy and robustness compared to direct reinforcement learning techniques, highlighting the potential of combining data-driven modeling with reinforcement learning for complex dynamical systems.

**arXiv ID:** 2512.21081
</details>

<details>
<summary><strong>Data-regularized Reinforcement Learning for Diffusion Models at Scale</strong> - Haotian Ye, Kaiwen Zheng, Jiashu Xu, Puheng Li, Huayu Chen, Jiaqi Han, Sheng Liu, Qinsheng Zhang, Hanzi Mao, Zekun Hao, Prithvijit Chattopadhyay, Dinghao Yang, Liang Feng, Maosheng Liao, Junjie Bai, Ming-Yu Liu, James Zou, Stefano Ermon - [[pdf]](https://arxiv.org/pdf/2512.04332)</summary>

**Abstract:** Aligning generative diffusion models with human preferences via reinforcement learning (RL) is critical yet challenging. Most existing algorithms are often vulnerable to reward hacking, such as quality degradation, over-stylization, or reduced diversity. Our analysis demonstrates that this can be attributed to the inherent limitations of their regularization, which provides unreliable penalties. We introduce Data-regularized Diffusion Reinforcement Learning (DDRL), a novel framework that uses the forward KL divergence to anchor the policy to an off-policy data distribution. Theoretically, DDRL enables robust, unbiased integration of RL with standard diffusion training. Empirically, this translates into a simple yet effective algorithm that combines reward maximization with diffusion loss minimization. With over a million GPU hours of experiments and ten thousand double-blind human evaluations, we demonstrate on high-resolution video generation tasks that DDRL significantly improves rewards while alleviating the reward hacking seen in baselines, achieving the highest human preference and establishing a robust and scalable paradigm for diffusion post-training.

**arXiv ID:** 2512.04332
</details>

<details>
<summary><strong>Eliciting Risk Aversion with Inverse Reinforcement Learning via Interactive Questioning</strong> - Ziteng Cheng, Anthony Coache, Sebastian Jaimungal - [[pdf]](https://arxiv.org/pdf/2308.08427)</summary>

**Abstract:** We investigate a framework for robo-advisors to estimate non-expert clients' risk aversion using adaptive binary-choice questionnaires. We model risk aversion using cost functions and spectral risk measures in a static setting. We prove the finite-sample identifiability and, for properly designed questions, obtain a convergence rate of $\sqrt{N}$ up to a logarithmic factor, where $N$ is the number of questions. We introduce the notion of distinguishing power and demonstrate, through simulated experiments, that designing questions by maximizing distinguishing power achieves satisfactory accuracy in learning risk aversion with fewer than 50 questions. We also provide a preliminary investigation of an infinite-horizon setting with an additional discount factor for dynamic risk aversion, establishing qualitative identifiability in this case.

**arXiv ID:** 2308.08427
</details>

<details>
<summary><strong>Global End-Effector Pose Control of an Underactuated Aerial Manipulator via Reinforcement Learning</strong> - Shlok Deshmukh, Javier Alonso-Mora, Sihao Sun - [[pdf]](https://arxiv.org/pdf/2512.21085)</summary>

**Abstract:** Aerial manipulators, which combine robotic arms with multi-rotor drones, face strict constraints on arm weight and mechanical complexity. In this work, we study a lightweight 2-degree-of-freedom (DoF) arm mounted on a quadrotor via a differential mechanism, capable of full six-DoF end-effector pose control. While the minimal design enables simplicity and reduced payload, it also introduces challenges such as underactuation and sensitivity to external disturbances, including manipulation of heavy loads and pushing tasks. To address these, we employ reinforcement learning, training a Proximal Policy Optimization (PPO) agent in simulation to generate feedforward commands for quadrotor acceleration and body rates, along with joint angle targets. These commands are tracked by an incremental nonlinear dynamic inversion (INDI) attitude controller and a PID joint controller, respectively. Flight experiments demonstrate centimeter-level position accuracy and degree-level orientation precision, with robust performance under external force disturbances. The results highlight the potential of learning-based control strategies for enabling contact-rich aerial manipulation using simple, lightweight platforms.

**arXiv ID:** 2512.21085
</details>

<details>
<summary><strong>DAPPER: Discriminability-Aware Policy-to-Policy Preference-Based Reinforcement Learning for Query-Efficient Robot Skill Acquisition</strong> - Yuki Kadokawa, Jonas Frey, Takahiro Miki, Takamitsu Matsubara, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2505.06357)</summary>

**Abstract:** Preference-based Reinforcement Learning (PbRL) enables policy learning through simple queries comparing trajectories from a single policy. While human responses to these queries make it possible to learn policies aligned with human preferences, PbRL suffers from low query efficiency, as policy bias limits trajectory diversity and reduces the number of discriminable queries available for learning preferences. This paper identifies preference discriminability, which quantifies how easily a human can judge which trajectory is closer to their ideal behavior, as a key metric for improving query efficiency. To address this, we move beyond comparisons within a single policy and instead generate queries by comparing trajectories from multiple policies, as training them from scratch promotes diversity without policy bias. We propose Discriminability-Aware Policy-to-Policy Preference-Based Efficient Reinforcement Learning (DAPPER), which integrates preference discriminability with trajectory diversification achieved by multiple policies. DAPPER trains new policies from scratch after each reward update and employs a discriminator that learns to estimate preference discriminability, enabling the prioritized sampling of more discriminable queries. During training, it jointly maximizes the preference reward and preference discriminability score, encouraging the discovery of highly rewarding and easily distinguishable policies. Experiments in simulated and real-world legged robot environments demonstrate that DAPPER outperforms previous methods in query efficiency, particularly under challenging preference discriminability conditions.

**arXiv ID:** 2505.06357
</details>

<details>
<summary><strong>Towards Autonomous Navigation in Endovascular Interventions</strong> - Tudor Jianu - [[pdf]](https://arxiv.org/pdf/2512.18081)</summary>

**Abstract:** Cardiovascular diseases remain the leading cause of global mortality, with minimally invasive treatment options offered through endovascular interventions. However, the precision and adaptability of current robotic systems for endovascular navigation are limited by heuristic control, low autonomy, and the absence of haptic feedback. This thesis presents an integrated AI-driven framework for autonomous guidewire navigation in complex vascular environments, addressing key challenges in data availability, simulation fidelity, and navigational accuracy.
A high-fidelity, real-time simulation platform, CathSim, is introduced for reinforcement learning based catheter navigation, featuring anatomically accurate vascular models and contact dynamics. Building on CathSim, the Expert Navigation Network is developed, a policy that fuses visual, kinematic, and force feedback for autonomous tool control. To mitigate data scarcity, the open-source, bi-planar fluoroscopic dataset Guide3D is proposed, comprising more than 8,700 annotated images for 3D guidewire reconstruction. Finally, SplineFormer, a transformer-based model, is introduced to directly predict guidewire geometry as continuous B-spline parameters, enabling interpretable, real-time navigation.
The findings show that combining high-fidelity simulation, multimodal sensory fusion, and geometric modelling substantially improves autonomous endovascular navigation and supports safer, more precise minimally invasive procedures.

**arXiv ID:** 2512.18081
</details>

<details>
<summary><strong>CoDrone: Autonomous Drone Navigation Assisted by Edge and Cloud Foundation Models</strong> - Pengyu Chen, Tao Ouyang, Ke Luo, Weijie Hong, Xu Chen - [[pdf]](https://arxiv.org/pdf/2512.19083)</summary>

**Abstract:** Autonomous navigation for Unmanned Aerial Vehicles faces key challenges from limited onboard computational resources, which restrict deployed deep neural networks to shallow architectures incapable of handling complex environments. Offloading tasks to remote edge servers introduces high latency, creating an inherent trade-off in system design. To address these limitations, we propose CoDrone - the first cloud-edge-end collaborative computing framework integrating foundation models into autonomous UAV cruising scenarios - effectively leveraging foundation models to enhance performance of resource-constrained unmanned aerial vehicle platforms. To reduce onboard computation and data transmission overhead, CoDrone employs grayscale imagery for the navigation model. When enhanced environmental perception is required, CoDrone leverages the edge-assisted foundation model Depth Anything V2 for depth estimation and introduces a novel one-dimensional occupancy grid-based navigation method - enabling fine-grained scene understanding while advancing efficiency and representational simplicity of autonomous navigation. A key component of CoDrone is a Deep Reinforcement Learning-based neural scheduler that seamlessly integrates depth estimation with autonomous navigation decisions, enabling real-time adaptation to dynamic environments. Furthermore, the framework introduces a UAV-specific vision language interaction module incorporating domain-tailored low-level flight primitives to enable effective interaction between the cloud foundation model and the UAV. The introduction of VLM enhances open-set reasoning capabilities in complex unseen scenarios. Experimental results show CoDrone outperforms baseline methods under varying flight speeds and network conditions, achieving a 40% increase in average flight distance and a 5% improvement in average Quality of Navigation.

**arXiv ID:** 2512.19083
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
