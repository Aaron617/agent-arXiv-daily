# Agent arXiv Daily

**Last Updated:** 2025-09-11 02:38:46

**Total Papers:** 38

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
<summary><strong>EnvX: Agentize Everything with Agentic AI</strong> - Linyao Chen, Zimian Peng, Yingxuan Yang, Yikun Wang, Wenzheng Tom Tang, Hiroki H. Kobayashi, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2509.08088)</summary>

**Abstract:** The widespread availability of open-source repositories has led to a vast collection of reusable software components, yet their utilization remains manual, error-prone, and disconnected. Developers must navigate documentation, understand APIs, and write integration code, creating significant barriers to efficient software reuse. To address this, we present EnvX, a framework that leverages Agentic AI to agentize GitHub repositories, transforming them into intelligent, autonomous agents capable of natural language interaction and inter-agent collaboration. Unlike existing approaches that treat repositories as static code resources, EnvX reimagines them as active agents through a three-phase process: (1) TODO-guided environment initialization, which sets up the necessary dependencies, data, and validation datasets; (2) human-aligned agentic automation, allowing repository-specific agents to autonomously perform real-world tasks; and (3) Agent-to-Agent (A2A) protocol, enabling multiple agents to collaborate. By combining large language model capabilities with structured tool integration, EnvX automates not just code generation, but the entire process of understanding, initializing, and operationalizing repository functionality. We evaluate EnvX on the GitTaskBench benchmark, using 18 repositories across domains such as image processing, speech recognition, document analysis, and video manipulation. Our results show that EnvX achieves a 74.07% execution completion rate and 51.85% task pass rate, outperforming existing frameworks. Case studies further demonstrate EnvX's ability to enable multi-repository collaboration via the A2A protocol. This work marks a shift from treating repositories as passive code resources to intelligent, interactive agents, fostering greater accessibility and collaboration within the open-source ecosystem.

**arXiv ID:** 2509.08088
</details>

<details>
<summary><strong>Co-Investigator AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives</strong> - Prathamesh Vasudeo Naik, Naresh Kumar Dintakurthi, Zhanghao Hu, Yue Wang, Robby Qiu - [[pdf]](https://arxiv.org/pdf/2509.08380)</summary>

**Abstract:** Generating regulatorily compliant Suspicious Activity Report (SAR) remains a high-cost, low-scalability bottleneck in Anti-Money Laundering (AML) workflows. While large language models (LLMs) offer promising fluency, they suffer from factual hallucination, limited crime typology alignment, and poor explainability -- posing unacceptable risks in compliance-critical domains. This paper introduces Co-Investigator AI, an agentic framework optimized to produce Suspicious Activity Reports (SARs) significantly faster and with greater accuracy than traditional methods. Drawing inspiration from recent advances in autonomous agent architectures, such as the AI Co-Scientist, our approach integrates specialized agents for planning, crime type detection, external intelligence gathering, and compliance validation. The system features dynamic memory management, an AI-Privacy Guard layer for sensitive data handling, and a real-time validation agent employing the Agent-as-a-Judge paradigm to ensure continuous narrative quality assurance. Human investigators remain firmly in the loop, empowered to review and refine drafts in a collaborative workflow that blends AI efficiency with domain expertise. We demonstrate the versatility of Co-Investigator AI across a range of complex financial crime scenarios, highlighting its ability to streamline SAR drafting, align narratives with regulatory expectations, and enable compliance teams to focus on higher-order analytical work. This approach marks the beginning of a new era in compliance reporting -- bringing the transformative benefits of AI agents to the core of regulatory processes and paving the way for scalable, reliable, and transparent SAR generation.

**arXiv ID:** 2509.08380
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (7 papers)</h2></summary>

<details>
<summary><strong>Trust Semantics Distillation for Collaborator Selection via Memory-Augmented Agentic AI</strong> - Botao Zhu, Jeslyn Wang, Dusit Niyato, Xianbin Wang - [[pdf]](https://arxiv.org/pdf/2509.08151)</summary>

**Abstract:** Accurate trustworthiness evaluation of potential collaborating devices is essential for the effective execution of complex computing tasks. This evaluation process involves collecting diverse trust-related data from potential collaborators, including historical performance and available resources, for collaborator selection. However, when each task owner independently assesses all collaborators' trustworthiness, frequent data exchange, complex reasoning, and dynamic situation changes can result in significant overhead and deteriorated trust evaluation. To overcome these challenges, we propose a task-specific trust semantics distillation (2TSD) model based on a large AI model (LAM)-driven teacher-student agent architecture. The teacher agent is deployed on a server with powerful computational capabilities and an augmented memory module dedicated to multidimensional trust-related data collection, task-specific trust semantics extraction, and task-collaborator matching analysis. Upon receiving task-specific requests from device-side student agents, the teacher agent transfers the trust semantics of potential collaborators to the student agents, enabling rapid and accurate collaborator selection. Experimental results demonstrate that the proposed 2TSD model can reduce collaborator evaluation time, decrease device resource consumption, and improve the accuracy of collaborator selection.

**arXiv ID:** 2509.08151
</details>

<details>
<summary><strong>Exploratory Retrieval-Augmented Planning For Continual Embodied Instruction Following</strong> - Minjong Yoo, Jinwoo Jang, Wei-jin Park, Honguk Woo - [[pdf]](https://arxiv.org/pdf/2509.08222)</summary>

**Abstract:** This study presents an Exploratory Retrieval-Augmented Planning (ExRAP) framework, designed to tackle continual instruction following tasks of embodied agents in dynamic, non-stationary environments. The framework enhances Large Language Models' (LLMs) embodied reasoning capabilities by efficiently exploring the physical environment and establishing the environmental context memory, thereby effectively grounding the task planning process in time-varying environment contexts. In ExRAP, given multiple continual instruction following tasks, each instruction is decomposed into queries on the environmental context memory and task executions conditioned on the query results. To efficiently handle these multiple tasks that are performed continuously and simultaneously, we implement an exploration-integrated task planning scheme by incorporating the {information-based exploration} into the LLM-based planning process. Combined with memory-augmented query evaluation, this integrated scheme not only allows for a better balance between the validity of the environmental context memory and the load of environment exploration, but also improves overall task performance. Furthermore, we devise a {temporal consistency refinement} scheme for query evaluation to address the inherent decay of knowledge in the memory. Through experiments with VirtualHome, ALFRED, and CARLA, our approach demonstrates robustness against a variety of embodied instruction following scenarios involving different instruction scales and types, and non-stationarity degrees, and it consistently outperforms other state-of-the-art LLM-based task planning approaches in terms of both goal success rate and execution efficiency.

**arXiv ID:** 2509.08222
</details>

<details>
<summary><strong>Agents of Discovery</strong> - Sascha Diefenbacher, Anna Hallin, Gregor Kasieczka, Michael Krämer, Anne Lauscher, Tim Lukas - [[pdf]](https://arxiv.org/pdf/2509.08535)</summary>

**Abstract:** The substantial data volumes encountered in modern particle physics and other domains of fundamental physics research allow (and require) the use of increasingly complex data analysis tools and workflows. While the use of machine learning (ML) tools for data analysis has recently proliferated, these tools are typically special-purpose algorithms that rely, for example, on encoded physics knowledge to reach optimal performance. In this work, we investigate a new and orthogonal direction: Using recent progress in large language models (LLMs) to create a team of agents -- instances of LLMs with specific subtasks -- that jointly solve data analysis-based research problems in a way similar to how a human researcher might: by creating code to operate standard tools and libraries (including ML systems) and by building on results of previous iterations. If successful, such agent-based systems could be deployed to automate routine analysis components to counteract the increasing complexity of modern tool chains. To investigate the capabilities of current-generation commercial LLMs, we consider the task of anomaly detection via the publicly available and highly-studied LHC Olympics dataset. Several current models by OpenAI (GPT-4o, o4-mini, GPT-4.1, and GPT-5) are investigated and their stability tested. Overall, we observe the capacity of the agent-based system to solve this data analysis problem. The best agent-created solutions mirror the performance of human state-of-the-art results.

**arXiv ID:** 2509.08535
</details>

<details>
<summary><strong>Evaluating LLMs Without Oracle Feedback: Agentic Annotation Evaluation Through Unsupervised Consistency Signals</strong> - Cheng Chen, Haiyan Yin, Ivor Tsang - [[pdf]](https://arxiv.org/pdf/2509.08809)</summary>

**Abstract:** Large Language Models (LLMs), when paired with prompt-based tasks, have significantly reduced data annotation costs and reliance on human annotators. However, evaluating the quality of their annotations remains challenging in dynamic, unsupervised environments where oracle feedback is scarce and conventional methods fail. To address this challenge, we propose a novel agentic annotation paradigm, where a student model collaborates with a noisy teacher (the LLM) to assess and refine annotation quality without relying on oracle feedback. The student model, acting as an unsupervised feedback mechanism, employs a user preference-based majority voting strategy to evaluate the consistency of the LLM outputs. To systematically measure the reliability of LLM-generated annotations, we introduce the Consistent and Inconsistent (CAI) Ratio, a novel unsupervised evaluation metric. The CAI Ratio not only quantifies the annotation quality of the noisy teacher under limited user preferences but also plays a critical role in model selection, enabling the identification of robust LLMs in dynamic, unsupervised environments. Applied to ten open-domain NLP datasets across four LLMs, the CAI Ratio demonstrates a strong positive correlation with LLM accuracy, establishing it as an essential tool for unsupervised evaluation and model selection in real-world settings.

**arXiv ID:** 2509.08809
</details>

<details>
<summary><strong>Foundation Models for Autonomous Driving Perception: A Survey Through Core Capabilities</strong> - Rajendramayavan Sathyam, Yueqi Li - [[pdf]](https://arxiv.org/pdf/2509.08302)</summary>

**Abstract:** Foundation models are revolutionizing autonomous driving perception, transitioning the field from narrow, task-specific deep learning models to versatile, general-purpose architectures trained on vast, diverse datasets. This survey examines how these models address critical challenges in autonomous perception, including limitations in generalization, scalability, and robustness to distributional shifts. The survey introduces a novel taxonomy structured around four essential capabilities for robust performance in dynamic driving environments: generalized knowledge, spatial understanding, multi-sensor robustness, and temporal reasoning. For each capability, the survey elucidates its significance and comprehensively reviews cutting-edge approaches. Diverging from traditional method-centric surveys, our unique framework prioritizes conceptual design principles, providing a capability-driven guide for model development and clearer insights into foundational aspects. We conclude by discussing key challenges, particularly those associated with the integration of these capabilities into real-time, scalable systems, and broader deployment challenges related to computational demands and ensuring model reliability against issues like hallucinations and out-of-distribution failures. The survey also outlines crucial future research directions to enable the safe and effective deployment of foundation models in autonomous driving systems.

**arXiv ID:** 2509.08302
</details>

<details>
<summary><strong>AutoODD: Agentic Audits via Bayesian Red Teaming in Black-Box Models</strong> - Rebecca Martin, Jay Patrikar, Sebastian Scherer - [[pdf]](https://arxiv.org/pdf/2509.08638)</summary>

**Abstract:** Specialized machine learning models, regardless of architecture and training, are susceptible to failures in deployment. With their increasing use in high risk situations, the ability to audit these models by determining their operational design domain (ODD) is crucial in ensuring safety and compliance. However, given the high-dimensional input spaces, this process often requires significant human resources and domain expertise. To alleviate this, we introduce \coolname, an LLM-Agent centric framework for automated generation of semantically relevant test cases to search for failure modes in specialized black-box models. By leveraging LLM-Agents as tool orchestrators, we aim to fit a uncertainty-aware failure distribution model on a learned text-embedding manifold by projecting the high-dimension input space to low-dimension text-embedding latent space. The LLM-Agent is tasked with iteratively building the failure landscape by leveraging tools for generating test-cases to probe the model-under-test (MUT) and recording the response. The agent also guides the search using tools to probe uncertainty estimate on the low dimensional manifold. We demonstrate this process in a simple case using models trained with missing digits on the MNIST dataset and in the real world setting of vision-based intruder detection for aerial vehicles.

**arXiv ID:** 2509.08638
</details>

<details>
<summary><strong>T-araVLN: Translator for Agricultural Robotic Agents on Vision-and-Language Navigation</strong> - Xiaobei Zhao, Xingqi Lyu, Xiang Li - [[pdf]](https://arxiv.org/pdf/2509.06644)</summary>

**Abstract:** Agricultural robotic agents have been becoming powerful helpers in a wide range of agricultural tasks, nevertheless, still heavily rely on manual operation or untransportable railway for movement. The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling agents navigate to the target position following the natural language instructions. AgriVLN effectively understands the simple instructions, however, often misunderstands the complicated instructions. To bridge this gap, we propose the method of Translator for Agricultural Robotic Agents on Vision-and-Language Navigation (T-araVLN), in which the Instruction Translator module translates the original instruction to be both refined and precise. Being evaluated on the A2A benchmark, our T-araVLN effectively improves Success Rate from 0.47 to 0.63 and reduces Navigation Error from 2.91m to 2.28m, demonstrating the state-of-the-art performance in the agricultural domain. Code: this https URL.

**arXiv ID:** 2509.06644
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>Architecting Resilient LLM Agents: A Guide to Secure Plan-then-Execute Implementations</strong> - Ron F. Del Rosario, Klaudia Krawiecka, Christian Schroeder de Witt - [[pdf]](https://arxiv.org/pdf/2509.08646)</summary>

**Abstract:** As Large Language Model (LLM) agents become increasingly capable of automating complex, multi-step tasks, the need for robust, secure, and predictable architectural patterns is paramount. This paper provides a comprehensive guide to the ``Plan-then-Execute'' (P-t-E) pattern, an agentic design that separates strategic planning from tactical execution. We explore the foundational principles of P-t-E, detailing its core components - the Planner and the Executor - and its architectural advantages in predictability, cost-efficiency, and reasoning quality over reactive patterns like ReAct (Reason + Act). A central focus is placed on the security implications of this design, particularly its inherent resilience to indirect prompt injection attacks by establishing control-flow integrity. We argue that while P-t-E provides a strong foundation, a defense-in-depth strategy is necessary, and we detail essential complementary controls such as the Principle of Least Privilege, task-scoped tool access, and sandboxed code execution. To make these principles actionable, this guide provides detailed implementation blueprints and working code references for three leading agentic frameworks: LangChain (via LangGraph), CrewAI, and AutoGen. Each framework's approach to implementing the P-t-E pattern is analyzed, highlighting unique features like LangGraph's stateful graphs for re-planning, CrewAI's declarative tool scoping for security, and AutoGen's built-in Docker sandboxing. Finally, we discuss advanced patterns, including dynamic re-planning loops, parallel execution with Directed Acyclic Graphs (DAGs), and the critical role of Human-in-the-Loop (HITL) verification, to offer a complete strategic blueprint for architects, developers, and security engineers aiming to build production-grade, resilient, and trustworthy LLM agents.

**arXiv ID:** 2509.08646
</details>

<details>
<summary><strong>AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning</strong> - Zhiheng Xi, Jixuan Huang, Chenyang Liao, Baodai Huang, Honglin Guo, Jiaqi Liu, Rui Zheng, Junjie Ye, Jiazheng Zhang, Wenxiang Chen, Wei He, Yiwen Ding, Guanyu Li, Zehui Chen, Zhengyin Du, Xuesong Yao, Yufei Xu, Jiecao Chen, Tao Gui, Zuxuan Wu, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang - [[pdf]](https://arxiv.org/pdf/2509.08755)</summary>

**Abstract:** Developing autonomous LLM agents capable of making a series of intelligent decisions to solve complex, real-world tasks is a fast-evolving frontier. Like human cognitive development, agents are expected to acquire knowledge and skills through exploration and interaction with the environment. Despite advances, the community still lacks a unified, interactive reinforcement learning (RL) framework that can effectively train such agents from scratch -- without relying on supervised fine-tuning (SFT) -- across diverse and realistic environments. To bridge this gap, we introduce AgentGym-RL, a new framework to train LLM agents for multi-turn interactive decision-making through RL. The framework features a modular and decoupled architecture, ensuring high flexibility and extensibility. It encompasses a wide variety of real-world scenarios, and supports mainstream RL algorithms. Furthermore, we propose ScalingInter-RL, a training approach designed for exploration-exploitation balance and stable RL optimization. In early stages, it emphasizes exploitation by restricting the number of interactions, and gradually shifts towards exploration with larger horizons to encourage diverse problem-solving strategies. In this way, the agent develops more diverse behaviors and is less prone to collapse under long horizons. We perform extensive experiments to validate the stability and effectiveness of both the AgentGym-RL framework and the ScalingInter-RL approach. Our agents match or surpass commercial models on 27 tasks across diverse environments. We offer key insights and will open-source the complete AgentGym-RL framework -- including code and datasets -- to empower the research community in developing the next generation of intelligent agents.

**arXiv ID:** 2509.08755
</details>

<details>
<summary><strong>MPO: Boosting LLM Agents with Meta Plan Optimization</strong> - Weimin Xiong, Yifan Song, Qingxiu Dong, Bingchan Zhao, Feifan Song, Xun Wang, Sujian Li - [[pdf]](https://arxiv.org/pdf/2503.02682)</summary>

**Abstract:** Recent advancements in large language models (LLMs) have enabled LLM-based agents to successfully tackle interactive planning tasks. However, despite their successes, existing approaches often suffer from planning hallucinations and require retraining for each new agent. To address these challenges, we propose the Meta Plan Optimization (MPO) framework, , which enhances agent planning capabilities by directly incorporating explicit guidance. Unlike previous methods that rely on complex knowledge, which either require significant human effort or lack quality assurance, MPO leverages high-level general guidance through meta plans to assist agent planning and enables continuous optimization of the meta plans based on feedback from the agent's task execution. Our experiments conducted on two representative tasks demonstrate that MPO significantly outperforms existing baselines. Moreover, our analysis indicates that MPO provides a plug-and-play solution that enhances both task completion efficiency and generalization capabilities in previous unseen scenarios.

**arXiv ID:** 2503.02682
</details>

<details>
<summary><strong>CyberRAG: An Agentic RAG cyber attack classification and reporting tool</strong> - Francesco Blefari, Cristian Cosentino, Francesco Aurelio Pironti, Angelo Furfaro, Fabrizio Marozzo - [[pdf]](https://arxiv.org/pdf/2507.02424)</summary>

**Abstract:** Intrusion Detection and Prevention Systems (IDS/IPS) in large enterprises can generate hundreds of thousands of alerts per hour, overwhelming analysts with logs requiring rapidly evolving expertise. Conventional machine-learning detectors reduce alert volume but still yield many false positives, while standard Retrieval-Augmented Generation (RAG) pipelines often retrieve irrelevant context and fail to justify predictions. We present CyberRAG, a modular agent-based RAG framework that delivers real-time classification, explanation, and structured reporting for cyber-attacks. A central LLM agent orchestrates: (i) fine-tuned classifiers specialized by attack family; (ii) tool adapters for enrichment and alerting; and (iii) an iterative retrieval-and-reason loop that queries a domain-specific knowledge base until evidence is relevant and self-consistent. Unlike traditional RAG, CyberRAG adopts an agentic design that enables dynamic control flow and adaptive reasoning. This architecture autonomously refines threat labels and natural-language justifications, reducing false positives and enhancing interpretability. It is also extensible: new attack types can be supported by adding classifiers without retraining the core agent. CyberRAG was evaluated on SQL Injection, XSS, and SSTI, achieving over 94\% accuracy per class and a final classification accuracy of 94.92\% through semantic orchestration. Generated explanations reached 0.94 in BERTScore and 4.9/5 in GPT-4-based expert evaluation, with robustness preserved against adversarial and unseen payloads. These results show that agentic, specialist-oriented RAG can combine high detection accuracy with trustworthy, SOC-ready prose, offering a flexible path toward partially automated cyber-defense workflows.

**arXiv ID:** 2507.02424
</details>

<details>
<summary><strong>Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL</strong> - Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, Yi Wu - [[pdf]](https://arxiv.org/pdf/2508.07976)</summary>

**Abstract:** Recent advancements in LLM-based agents have demonstrated remarkable capabilities in handling complex, knowledge-intensive tasks by integrating external tools. Among diverse choices of tools, search tools play a pivotal role in accessing vast external knowledge. However, open-source agents still fall short of achieving expert-level Search Intelligence, the ability to resolve ambiguous queries, generate precise searches, analyze results, and conduct thorough exploration. Existing approaches fall short in scalability, efficiency, and data quality. For example, small turn limits in existing online RL methods, e.g. <=10, restrict complex strategy learning. This paper introduces ASearcher, an open-source project for large-scale RL training of search agents. Our key contributions include: (1) Scalable fully asynchronous RL training that enables long-horizon search while maintaining high training efficiency. (2) A prompt-based LLM agent that autonomously synthesizes high-quality and challenging QAs, creating a large-scale QA dataset. Through RL training, our prompt-based QwQ-32B agent achieves substantial improvements, with 46.7% and 20.8% Avg@4 gains on xBench and GAIA, respectively. Notably, our agent exhibits extreme long-horizon search, with tool calls exceeding 40 turns and output tokens exceeding 150k during training time. With a simple agent design and no external LLMs, ASearcher-Web-QwQ achieves Avg@4 scores of 42.1 on xBench and 52.8 on GAIA, surpassing existing open-source 32B agents. We open-source our models, training data, and codes in this https URL.

**arXiv ID:** 2508.07976
</details>

<details>
<summary><strong>TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks</strong> - Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, Shuyan Zhou, Graham Neubig - [[pdf]](https://arxiv.org/pdf/2412.14161)</summary>

**Abstract:** We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at accelerating or even autonomously performing work-related tasks? The answer to this question has important implications both for industry looking to adopt AI into their workflows and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents' performance on performing real-world professional tasks, in this paper we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that the most competitive agent can complete 30% of tasks autonomously. This paints a nuanced picture on task automation with LM agents--in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems. We release code, data, environment, and experiments on this https URL.

**arXiv ID:** 2412.14161
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (10 papers)</h2></summary>

<details>
<summary><strong>Automatic Failure Attribution and Critical Step Prediction Method for Multi-Agent Systems Based on Causal Inference</strong> - Guoqing Ma, Jia Zhu, Hanghui Guo, Weijie Shi, Jiawei Shen, Jingjiang Liu, Yidan Liang - [[pdf]](https://arxiv.org/pdf/2509.08682)</summary>

**Abstract:** Multi-agent systems (MAS) are critical for automating complex tasks, yet their practical deployment is severely hampered by the challenge of failure attribution. Current diagnostic tools, which rely on statistical correlations, are fundamentally inadequate; on challenging benchmarks like Who\&When, state-of-the-art methods achieve less than 15\% accuracy in locating the root-cause step of a failure. To address this critical gap, we introduce the first failure attribution framework for MAS grounded in multi-granularity causal inference. Our approach makes two key technical contributions: (1) a performance causal inversion principle, which correctly models performance dependencies by reversing the data flow in execution logs, combined with Shapley values to accurately assign agent-level blame; (2) a novel causal discovery algorithm, CDC-MAS, that robustly identifies critical failure steps by tackling the non-stationary nature of MAS interaction data. The framework's attribution results directly fuel an automated optimization loop, generating targeted suggestions whose efficacy is validated via counterfactual simulations. Evaluations on the Who\&When and TRAIL benchmarks demonstrate a significant leap in performance. Our method achieves up to 36.2\% step-level accuracy. Crucially, the generated optimizations boost overall task success rates by an average of 22.4\%. This work provides a principled and effective solution for debugging complex agent interactions, paving the way for more reliable and interpretable multi-agent systems.

**arXiv ID:** 2509.08682
</details>

<details>
<summary><strong>Risk-Bounded Multi-Agent Visual Navigation via Dynamic Budget Allocation</strong> - Viraj Parimi, Brian C. Williams - [[pdf]](https://arxiv.org/pdf/2509.08157)</summary>

**Abstract:** Safe navigation is essential for autonomous systems operating in hazardous environments, especially when multiple agents must coordinate using just visual inputs over extended time horizons. Traditional planning methods excel at solving long-horizon tasks but rely on predefined distance metrics, while safe Reinforcement Learning (RL) can learn complex behaviors using high-dimensional inputs yet struggles with multi-agent, goal-conditioned scenarios. Recent work combined these paradigms by leveraging goal-conditioned RL (GCRL) to build an intermediate graph from replay buffer states, pruning unsafe edges, and using Conflict-Based Search (CBS) for multi-agent path planning. Although effective, this graph-pruning approach can be overly conservative, limiting mission efficiency by precluding missions that must traverse high-risk regions. To address this limitation, we propose RB-CBS, a novel extension to CBS that dynamically allocates and adjusts user-specified risk bound ($\Delta$) across agents to flexibly trade off safety and speed. Our improved planner ensures that each agent receives a local risk budget ($\delta$) enabling more efficient navigation while still respecting overall safety constraints. Experimental results demonstrate that this iterative risk-allocation framework yields superior performance in complex environments, allowing multiple agents to find collision-free paths within the user-specified $\Delta$.

**arXiv ID:** 2509.08157
</details>

<details>
<summary><strong>Symmetry-Guided Multi-Agent Inverse Reinforcement Learnin</strong> - Yongkai Tian, Yirong Qi, Xin Yu, Wenjun Wu, Jie Luo - [[pdf]](https://arxiv.org/pdf/2509.08257)</summary>

**Abstract:** In robotic systems, the performance of reinforcement learning depends on the rationality of predefined reward functions. However, manually designed reward functions often lead to policy failures due to inaccuracies. Inverse Reinforcement Learning (IRL) addresses this problem by inferring implicit reward functions from expert demonstrations. Nevertheless, existing methods rely heavily on large amounts of expert demonstrations to accurately recover the reward function. The high cost of collecting expert demonstrations in robotic applications, particularly in multi-robot systems, severely hinders the practical deployment of IRL. Consequently, improving sample efficiency has emerged as a critical challenge in multi-agent inverse reinforcement learning (MIRL). Inspired by the symmetry inherent in multi-agent systems, this work theoretically demonstrates that leveraging symmetry enables the recovery of more accurate reward functions. Building upon this insight, we propose a universal framework that integrates symmetry into existing multi-agent adversarial IRL algorithms, thereby significantly enhancing sample efficiency. Experimental results from multiple challenging tasks have demonstrated the effectiveness of this framework. Further validation in physical multi-robot systems has shown the practicality of our method.

**arXiv ID:** 2509.08257
</details>

<details>
<summary><strong>Game-Theoretic Resilience Framework for Cyber-Physical Microgrids using Multi-Agent Reinforcement Learning</strong> - S Krishna Niketh, Sagar Babu Mitikiri, V Vignesh, Vedantham Lakshmi Srinivas, Mayukha Pal - [[pdf]](https://arxiv.org/pdf/2509.08310)</summary>

**Abstract:** The increasing reliance on cyber physical infrastructure in modern power systems has amplified the risk of targeted cyber attacks, necessitating robust and adaptive resilience strategies. This paper presents a mathematically rigorous game theoretic framework to evaluate and enhance microgrid resilience using a combination of quantitative resilience metrics Load Served Ratio LSR, Critical Load Resilience CLR, Topological Survivability Score TSS, and DER Resilience Score DRS. These are integrated into a unified payoff matrix using the Analytic Hierarchy Process AHP to assess attack defense interactions. The framework is formalized as a finite horizon Markov Decision Process MDP with formal convergence guarantees and computational complexity bounds. Three case studies are developed 1. static attacks analyzed via Nash equilibrium, 2. severe attacks incorporating high impact strategies, and 3. adaptive attacks using Stackelberg games, regret matching, softmax heuristics, and Multi Agent Q Learning. Rigorous theoretical analysis provides convergence proofs with explicit rates , PAC learning sample complexity bounds, and computational complexity analysis. The framework is tested on an enhanced IEEE 33bus distribution system with DERs and control switches, demonstrating the effectiveness of adaptive and strategic defenses in improving cyber physical resilience with statistically significant improvements of 18.7% 2.1% over static approaches.

**arXiv ID:** 2509.08310
</details>

<details>
<summary><strong>Adaptive Monitoring and Real-World Evaluation of Agentic AI Systems</strong> - Manish Shukla - [[pdf]](https://arxiv.org/pdf/2509.00115)</summary>

**Abstract:** Agentic artificial intelligence (AI) -- multi-agent systems that combine large language models with external tools and autonomous planning -- are rapidly transitioning from research laboratories into high-stakes domains. Our earlier "Basic" paper introduced a five-axis framework and proposed preliminary metrics such as goal drift and harm reduction but did not provide an algorithmic instantiation or empirical evidence. This "Advanced" sequel fills that gap. First, we revisit recent benchmarks and industrial deployments to show that technical metrics still dominate evaluations: a systematic review of 84 papers from 2023--2025 found that 83% report capability metrics while only 30% consider human-centred or economic axes [2]. Second, we formalise an Adaptive Multi-Dimensional Monitoring (AMDM) algorithm that normalises heterogeneous metrics, applies per-axis exponentially weighted moving-average thresholds and performs joint anomaly detection via the Mahalanobis distance. Third, we conduct simulations and real-world experiments. AMDM cuts anomaly-detection latency from 12.3 s to 5.6 s on simulated goal drift and reduces false-positive rates from 4.5% to 0.9% compared with static thresholds. We present a comparison table and ROC/PR curves, and we reanalyse case studies to surface missing metrics. Code, data and a reproducibility checklist accompany this paper to facilitate replication. The code supporting this work is available at this https URL.

**arXiv ID:** 2509.00115
</details>

<details>
<summary><strong>From Static to Adaptive Defense: Federated Multi-Agent Deep Reinforcement Learning-Driven Moving Target Defense Against DoS Attacks in UAV Swarm Networks</strong> - Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Tian Qin, Yuyu Zhao - [[pdf]](https://arxiv.org/pdf/2506.07392)</summary>

**Abstract:** The proliferation of UAVs has enabled a wide range of mission-critical applications and is becoming a cornerstone of low-altitude networks, supporting smart cities, emergency response, and more. However, the open wireless environment, dynamic topology, and resource constraints of UAVs expose low-altitude networks to severe DoS threats. Traditional defense approaches, which rely on fixed configurations or centralized decision-making, cannot effectively respond to the rapidly changing conditions in UAV swarm environments. To address these challenges, we propose a novel federated multi-agent deep reinforcement learning (FMADRL)-driven moving target defense (MTD) framework for proactive DoS mitigation in low-altitude networks. Specifically, we design lightweight and coordinated MTD mechanisms, including leader switching, route mutation, and frequency hopping, to disrupt attacker efforts and enhance network resilience. The defense problem is formulated as a multi-agent partially observable Markov decision process, capturing the uncertain nature of UAV swarms under attack. Each UAV is equipped with a policy agent that autonomously selects MTD actions based on partial observations and local experiences. By employing a policy gradient-based algorithm, UAVs collaboratively optimize their policies via reward-weighted aggregation. Extensive simulations demonstrate that our approach significantly outperforms state-of-the-art baselines, achieving up to a 34.6% improvement in attack mitigation rate, a reduction in average recovery time of up to 94.6%, and decreases in energy consumption and defense cost by as much as 29.3% and 98.3%, respectively, under various DoS attack strategies. These results highlight the potential of intelligent, distributed defense mechanisms to protect low-altitude networks, paving the way for reliable and scalable low-altitude economy.

**arXiv ID:** 2506.07392
</details>

<details>
<summary><strong>ChemBOMAS: Accelerated BO in Chemistry with LLM-Enhanced Multi-Agent System</strong> - Dong Han, Zhehong Ai, Pengxiang Cai, Shuzhou Sun, Shanya Lu, Jianpeng Chen, Ben Gao, Lingli Ge, Weida Wang, Xiangxin Zhou, Xihui Liu, Mao Su, Wanli Ouyang, Lei Bai, Dongzhan Zhou, Tao XU, Yuqiang Li, Shufei Zhang - [[pdf]](https://arxiv.org/pdf/2509.08736)</summary>

**Abstract:** The efficiency of Bayesian optimization (BO) in chemistry is often hindered by sparse experimental data and complex reaction mechanisms. To overcome these limitations, we introduce ChemBOMAS, a new framework named LLM-Enhanced Multi-Agent System for accelerating BO in chemistry. ChemBOMAS's optimization process is enhanced by LLMs and synergistically employs two strategies: knowledge-driven coarse-grained optimization and data-driven fine-grained optimization. First, in the knowledge-driven coarse-grained optimization stage, LLMs intelligently decompose the vast search space by reasoning over existing chemical knowledge to identify promising candidate regions. Subsequently, in the data-driven fine-grained optimization stage, LLMs enhance the BO process within these candidate regions by generating pseudo-data points, thereby improving data utilization efficiency and accelerating convergence. Benchmark evaluations** further confirm that ChemBOMAS significantly enhances optimization effectiveness and efficiency compared to various BO algorithms. Importantly, the practical utility of ChemBOMAS was validated through wet-lab experiments conducted under pharmaceutical industry protocols, targeting conditional optimization for a previously unreported and challenging chemical reaction. In the wet experiment, ChemBOMAS achieved an optimal objective value of 96%. This was substantially higher than the 15% achieved by domain experts. This real-world success, together with strong performance on benchmark evaluations, highlights ChemBOMAS as a powerful tool to accelerate chemical discovery.

**arXiv ID:** 2509.08736
</details>

<details>
<summary><strong>A Comprehensive Review of Reinforcement Learning for Autonomous Driving in the CARLA Simulator</strong> - Elahe Delavari, Feeza Khan Khanzada, Jaerock Kwon - [[pdf]](https://arxiv.org/pdf/2509.08221)</summary>

**Abstract:** Autonomous-driving research has recently embraced deep Reinforcement Learning (RL) as a promising framework for data-driven decision making, yet a clear picture of how these algorithms are currently employed, benchmarked and evaluated is still missing. This survey fills that gap by systematically analysing around 100 peer-reviewed papers that train, test or validate RL policies inside the open-source CARLA simulator. We first categorize the literature by algorithmic family model-free, model-based, hierarchical, and hybrid and quantify their prevalence, highlighting that more than 80% of existing studies still rely on model-free methods such as DQN, PPO and SAC. Next, we explain the diverse state, action and reward formulations adopted across works, illustrating how choices of sensor modality (RGB, LiDAR, BEV, semantic maps, and carla kinematics states), control abstraction (discrete vs. continuous) and reward shaping are used across various literature. We also consolidate the evaluation landscape by listing the most common metrics (success rate, collision rate, lane deviation, driving score) and the towns, scenarios and traffic configurations used in CARLA benchmarks. Persistent challenges including sparse rewards, sim-to-real transfer, safety guarantees and limited behaviour diversity are distilled into a set of open research questions, and promising directions such as model-based RL, meta-learning and richer multi-agent simulations are outlined. By providing a unified taxonomy, quantitative statistics and a critical discussion of limitations, this review aims to serve both as a reference for newcomers and as a roadmap for advancing RL-based autonomous driving toward real-world deployment.

**arXiv ID:** 2509.08221
</details>

<details>
<summary><strong>Behaviorally Heterogeneous Multi-Agent Exploration Using Distributed Task Allocation</strong> - Nirabhra Mandal, Aamodh Suresh, Carlos Nieto-Granda, Sonia Martínez - [[pdf]](https://arxiv.org/pdf/2509.08242)</summary>

**Abstract:** We study a problem of multi-agent exploration with behaviorally heterogeneous robots. Each robot maps its surroundings using SLAM and identifies a set of areas of interest (AoIs) or frontiers that are the most informative to explore next. The robots assess the utility of going to a frontier using Behavioral Entropy (BE) and then determine which frontier to go to via a distributed task assignment scheme. We convert the task assignment problem into a non-cooperative game and use a distributed algorithm (d-PBRAG) to converge to the Nash equilibrium (which we show is the optimal task allocation solution). For unknown utility cases, we provide robust bounds using approximate rewards. We test our algorithm (which has less communication cost and fast convergence) in simulation, where we explore the effect of sensing radii, sensing accuracy, and heterogeneity among robotic teams with respect to the time taken to complete exploration and path traveled. We observe that having a team of agents with heterogeneous behaviors is beneficial.

**arXiv ID:** 2509.08242
</details>

<details>
<summary><strong>A Survey of World Models for Autonomous Driving</strong> - Tuo Feng, Wenguan Wang, Yi Yang - [[pdf]](https://arxiv.org/pdf/2501.11260)</summary>

**Abstract:** Recent breakthroughs in autonomous driving have been propelled by advances in robust world modeling, fundamentally transforming how vehicles interpret dynamic scenes and execute safe decision-making. World models have emerged as a linchpin technology, offering high-fidelity representations of the driving environment that integrate multi-sensor data, semantic cues, and temporal dynamics. This paper systematically reviews recent advances in world models for autonomous driving, proposing a three-tiered taxonomy: (i) Generation of Future Physical World, covering Image-, BEV-, OG-, and PC-based generation methods that enhance scene evolution modeling through diffusion models and 4D occupancy forecasting; (ii) Behavior Planning for Intelligent Agents, combining rule-driven and learning-based paradigms with cost map optimization and reinforcement learning for trajectory generation in complex traffic conditions; (ii) Interaction between Prediction and Planning, achieving multi-agent collaborative decision-making through latent space diffusion and memory-augmented architectures. The study further analyzes training paradigms, including self-supervised learning, multimodal pretraining, and generative data augmentation, while evaluating world models' performance in scene understanding and motion prediction tasks. Future research must address key challenges in self-supervised representation learning, multimodal fusion, and advanced simulation to advance the practical deployment of world models in complex urban environments. Overall, the comprehensive analysis provides a technical roadmap for harnessing the transformative potential of world models in advancing safe and reliable autonomous driving solutions.

**arXiv ID:** 2501.11260
</details>

</details>

<details open>
<summary><h2>Other Agent Research (2 papers)</h2></summary>

<details>
<summary><strong>Leveraging AI Agents for Autonomous Networks: A Reference Architecture and Empirical Studies</strong> - Binghan Wu, Shoufeng Wang, Yunxin Liu, Ya-Qin Zhang, Joseph Sifakis, Ye Ouyang - [[pdf]](https://arxiv.org/pdf/2509.08312)</summary>

**Abstract:** The evolution toward Level 4 (L4) Autonomous Networks (AN) represents a strategic inflection point in telecommunications, where networks must transcend reactive automation to achieve genuine cognitive capabilities--fulfilling TM Forum's vision of self-configuring, self-healing, and self-optimizing systems that deliver zero-wait, zero-touch, and zero-fault services. This work bridges the gap between architectural theory and operational reality by implementing Joseph Sifakis's AN Agent reference architecture in a functional cognitive system, deploying coordinated proactive-reactive runtimes driven by hybrid knowledge representation. Through an empirical case study of a Radio Access Network (RAN) Link Adaptation (LA) Agent, we validate this framework's transformative potential: demonstrating sub-10 ms real-time control in 5G NR sub-6 GHz while achieving 6% higher downlink throughput than Outer Loop Link Adaptation (OLLA) algorithms and 67% Block Error Rate (BLER) reduction for ultra-reliable services through dynamic Modulation and Coding Scheme (MCS) optimization. These improvements confirm the architecture's viability in overcoming traditional autonomy barriers and advancing critical L4-enabling capabilities toward next-generation objectives.

**arXiv ID:** 2509.08312
</details>

<details>
<summary><strong>Zero-Shot Metric Depth Estimation via Monocular Visual-Inertial Rescaling for Autonomous Aerial Navigation</strong> - Steven Yang, Xiaoyu Tian, Kshitij Goel, Wennie Tabib - [[pdf]](https://arxiv.org/pdf/2509.08159)</summary>

**Abstract:** This paper presents a methodology to predict metric depth from monocular RGB images and an inertial measurement unit (IMU). To enable collision avoidance during autonomous flight, prior works either leverage heavy sensors (e.g., LiDARs or stereo cameras) or data-intensive and domain-specific fine-tuning of monocular metric depth estimation methods. In contrast, we propose several lightweight zero-shot rescaling strategies to obtain metric depth from relative depth estimates via the sparse 3D feature map created using a visual-inertial navigation system. These strategies are compared for their accuracy in diverse simulation environments. The best performing approach, which leverages monotonic spline fitting, is deployed in the real-world on a compute-constrained quadrotor. We obtain on-board metric depth estimates at 15 Hz and demonstrate successful collision avoidance after integrating the proposed method with a motion primitives-based planner.

**arXiv ID:** 2509.08159
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (11 papers)</h2></summary>

<details>
<summary><strong>TCPO: Thought-Centric Preference Optimization for Effective Embodied Decision-making</strong> - Kechen Jiao, Zhirui Fang, Jiahao Liu, Bei Li, Qifan Wang, Xinyu Liu, Junhao Ruan, Zhongjian Qiao, Yifan Zhu, Yaxin Xu, Jingang Wang, Xiu Li - [[pdf]](https://arxiv.org/pdf/2509.08500)</summary>

**Abstract:** Using effective generalization capabilities of vision language models (VLMs) in context-specific dynamic tasks for embodied artificial intelligence remains a significant challenge. Although supervised fine-tuned models can better align with the real physical world, they still exhibit sluggish responses and hallucination issues in dynamically changing environments, necessitating further alignment. Existing post-SFT methods, reliant on reinforcement learning and chain-of-thought (CoT) approaches, are constrained by sparse rewards and action-only optimization, resulting in low sample efficiency, poor consistency, and model degradation. To address these issues, this paper proposes Thought-Centric Preference Optimization (TCPO) for effective embodied decision-making. Specifically, TCPO introduces a stepwise preference-based optimization approach, transforming sparse reward signals into richer step sample pairs. It emphasizes the alignment of the model's intermediate reasoning process, mitigating the problem of model degradation. Moreover, by incorporating Action Policy Consistency Constraint (APC), it further imposes consistency constraints on the model output. Experiments in the ALFWorld environment demonstrate an average success rate of 26.67%, achieving a 6% improvement over RL4VLM and validating the effectiveness of our approach in mitigating model degradation after fine-tuning. These results highlight the potential of integrating preference-based learning techniques with CoT processes to enhance the decision-making capabilities of vision-language models in embodied agents.

**arXiv ID:** 2509.08500
</details>

<details>
<summary><strong>Narrative-Guided Reinforcement Learning: A Platform for Studying Language Model Influence on Decision Making</strong> - Anup Tuladhar, Araz Minhas, Adam Kirton, Eli Kinney-Lang - [[pdf]](https://arxiv.org/pdf/2509.08785)</summary>

**Abstract:** We present a preliminary experimental platform that explores how narrative elements might shape AI decision-making by combining reinforcement learning (RL) with language model reasoning. While AI systems can now both make decisions and engage in narrative reasoning, these capabilities have mostly been studied separately. Our platform attempts to bridge this gap using a dual-system architecture to examine how narrative frameworks could influence reward-based learning. The system comprises a reinforcement learning policy that suggests actions based on past experience, and a language model that processes these suggestions through different narrative frameworks to guide decisions. This setup enables initial experimentation with narrative elements while maintaining consistent environment and reward structures. We implement this architecture in a configurable gridworld environment, where agents receive both policy suggestions and information about their surroundings. The platform's modular design facilitates controlled testing of environmental complexity, narrative parameters, and the interaction between reinforcement learning and narrative-based decisions. Our logging system captures basic decision metrics, from RL policy values to language model reasoning to action selection patterns. While preliminary, this implementation provides a foundation for studying how different narrative frameworks might affect reward-based decisions and exploring potential interactions between optimization-based learning and symbolic reasoning in AI systems.

**arXiv ID:** 2509.08785
</details>

<details>
<summary><strong>Quadrotor Navigation using Reinforcement Learning with Privileged Information</strong> - Jonathan Lee, Abhishek Rathod, Kshitij Goel, John Stecklein, Wennie Tabib - [[pdf]](https://arxiv.org/pdf/2509.08177)</summary>

**Abstract:** This paper presents a reinforcement learning-based quadrotor navigation method that leverages efficient differentiable simulation, novel loss functions, and privileged information to navigate around large obstacles. Prior learning-based methods perform well in scenes that exhibit narrow obstacles, but struggle when the goal location is blocked by large walls or terrain. In contrast, the proposed method utilizes time-of-arrival (ToA) maps as privileged information and a yaw alignment loss to guide the robot around large obstacles. The policy is evaluated in photo-realistic simulation environments containing large obstacles, sharp corners, and dead-ends. Our approach achieves an 86% success rate and outperforms baseline strategies by 34%. We deploy the policy onboard a custom quadrotor in outdoor cluttered environments both during the day and night. The policy is validated across 20 flights, covering 589 meters without collisions at speeds up to 4 m/s.

**arXiv ID:** 2509.08177
</details>

<details>
<summary><strong>Accelerating Reinforcement Learning Algorithms Convergence using Pre-trained Large Language Models as Tutors With Advice Reusing</strong> - Lukas Toral, Teddy Lazebnik - [[pdf]](https://arxiv.org/pdf/2509.08329)</summary>

**Abstract:** Reinforcement Learning (RL) algorithms often require long training to become useful, especially in complex environments with sparse rewards. While techniques like reward shaping and curriculum learning exist to accelerate training, these are often extremely specific and require the developer's professionalism and dedicated expertise in the problem's domain. Tackling this challenge, in this study, we explore the effectiveness of pre-trained Large Language Models (LLMs) as tutors in a student-teacher architecture with RL algorithms, hypothesizing that LLM-generated guidance allows for faster convergence. In particular, we explore the effectiveness of reusing the LLM's advice on the RL's convergence dynamics. Through an extensive empirical examination, which included 54 configurations, varying the RL algorithm (DQN, PPO, A2C), LLM tutor (Llama, Vicuna, DeepSeek), and environment (Blackjack, Snake, Connect Four), our results demonstrate that LLM tutoring significantly accelerates RL convergence while maintaining comparable optimal performance. Furthermore, the advice reuse mechanism shows a further improvement in training duration but also results in less stable convergence dynamics. Our findings suggest that LLM tutoring generally improves convergence, and its effectiveness is sensitive to the specific task, RL algorithm, and LLM model combination.

**arXiv ID:** 2509.08329
</details>

<details>
<summary><strong>A Survey of Reinforcement Learning for Large Reasoning Models</strong> - Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, Yu Fu, Xingtai Lv, Yuchen Zhang, Sihang Zeng, Shang Qu, Haozhan Li, Shijie Wang, Yuru Wang, Xinwei Long, Fangfu Liu, Xiang Xu, Jiaze Ma, Xuekai Zhu, Ermo Hua, Yihao Liu, Zonglin Li, Huayu Chen, Xiaoye Qu, Yafu Li, Weize Chen, Zhenzhao Yuan, Junqi Gao, Dong Li, Zhiyuan Ma, Ganqu Cui, Zhiyuan Liu, Biqing Qi, Ning Ding, Bowen Zhou - [[pdf]](https://arxiv.org/pdf/2509.08827)</summary>

**Abstract:** In this paper, we survey recent advances in Reinforcement Learning (RL) for reasoning with Large Language Models (LLMs). RL has achieved remarkable success in advancing the frontier of LLM capabilities, particularly in addressing complex logical tasks such as mathematics and coding. As a result, RL has emerged as a foundational methodology for transforming LLMs into LRMs. With the rapid progress of the field, further scaling of RL for LRMs now faces foundational challenges not only in computational resources but also in algorithm design, training data, and infrastructure. To this end, it is timely to revisit the development of this domain, reassess its trajectory, and explore strategies to enhance the scalability of RL toward Artificial SuperIntelligence (ASI). In particular, we examine research applying RL to LLMs and LRMs for reasoning abilities, especially since the release of DeepSeek-R1, including foundational components, core problems, training resources, and downstream applications, to identify future opportunities and directions for this rapidly evolving area. We hope this review will promote future research on RL for broader reasoning models. Github: this https URL

**arXiv ID:** 2509.08827
</details>

<details>
<summary><strong>VIDEE: Visual and Interactive Decomposition, Execution, and Evaluation of Text Analytics with Intelligent Agents</strong> - Sam Yu-Te Lee, Chenyang Ji, Shicheng Wen, Lifu Huang, Dongyu Liu, Kwan-Liu Ma - [[pdf]](https://arxiv.org/pdf/2506.21582)</summary>

**Abstract:** Text analytics has traditionally required specialized knowledge in Natural Language Processing (NLP) or text analysis, which presents a barrier for entry-level analysts. Recent advances in large language models (LLMs) have changed the landscape of NLP by enabling more accessible and automated text analysis (e.g., topic detection, summarization, information extraction, etc.). We introduce VIDEE, a system that supports entry-level data analysts to conduct advanced text analytics with intelligent agents. VIDEE instantiates a human-agent collaroration workflow consisting of three stages: (1) Decomposition, which incorporates a human-in-the-loop Monte-Carlo Tree Search algorithm to support generative reasoning with human feedback, (2) Execution, which generates an executable text analytics pipeline, and (3) Evaluation, which integrates LLM-based evaluation and visualizations to support user validation of execution results. We conduct two quantitative experiments to evaluate VIDEE's effectiveness and analyze common agent errors. A user study involving participants with varying levels of NLP and text analytics experience -- from none to expert -- demonstrates the system's usability and reveals distinct user behavior patterns. The findings identify design implications for human-agent collaboration, validate the practical utility of VIDEE for non-expert users, and inform future improvements to intelligent text analytics systems.

**arXiv ID:** 2506.21582
</details>

<details>
<summary><strong>Multi-Timescale Hierarchical Reinforcement Learning for Unified Behavior and Control of Autonomous Driving</strong> - Guizhe Jin, Zhuoren Li, Bo Leng, Ran Yu, Lu Xiong, Chen Sun - [[pdf]](https://arxiv.org/pdf/2506.23771)</summary>

**Abstract:** Reinforcement Learning (RL) is increasingly used in autonomous driving (AD) and shows clear advantages. However, most RL-based AD methods overlook policy structure design. An RL policy that only outputs short-timescale vehicle control commands results in fluctuating driving behavior due to fluctuations in network outputs, while one that only outputs long-timescale driving goals cannot achieve unified optimality of driving behavior and control. Therefore, we propose a multi-timescale hierarchical reinforcement learning approach. Our approach adopts a hierarchical policy structure, where high- and low-level RL policies are unified-trained to produce long-timescale motion guidance and short-timescale control commands, respectively. Therein, motion guidance is explicitly represented by hybrid actions to capture multimodal driving behaviors on structured road and support incremental low-level extend-state updates. Additionally, a hierarchical safety mechanism is designed to ensure multi-timescale safety. Evaluation in simulator-based and HighD dataset-based highway multi-lane scenarios demonstrates that our approach significantly improves AD performance, effectively increasing driving efficiency, action consistency and safety.

**arXiv ID:** 2506.23771
</details>

<details>
<summary><strong>How Should We Meta-Learn Reinforcement Learning Algorithms?</strong> - Alexander David Goldie, Zilin Wang, Jaron Cohen, Jakob Nicolaus Foerster, Shimon Whiteson - [[pdf]](https://arxiv.org/pdf/2507.17668)</summary>

**Abstract:** The process of meta-learning algorithms from data, instead of relying on manual design, is growing in popularity as a paradigm for improving the performance of machine learning systems. Meta-learning shows particular promise for reinforcement learning (RL), where algorithms are often adapted from supervised or unsupervised learning despite their suboptimality for RL. However, until now there has been a severe lack of comparison between different meta-learning algorithms, such as using evolution to optimise over black-box functions or LLMs to propose code. In this paper, we carry out this empirical comparison of the different approaches when applied to a range of meta-learned algorithms which target different parts of the RL pipeline. In addition to meta-train and meta-test performance, we also investigate factors including the interpretability, sample cost and train time for each meta-learning algorithm. Based on these findings, we propose several guidelines for meta-learning new RL algorithms which will help ensure that future learned algorithms are as performant as possible.

**arXiv ID:** 2507.17668
</details>

<details>
<summary><strong>ACE-RL: Adaptive Constraint-Enhanced Reward for Long-form Generation Reinforcement Learning</strong> - Jianghao Chen, Wei Sun, Qixiang Yin, Lingxing Kong, Zhixing Tan, Jiajun Zhang - [[pdf]](https://arxiv.org/pdf/2509.04903)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable progress in long-context understanding, yet they face significant challenges in high-quality long-form generation. Existing studies primarily suffer from two limitations: (1) A heavy reliance on scarce, high-quality long-form response data for supervised fine-tuning (SFT) or for pairwise preference reward in reinforcement learning (RL). (2) Focus on coarse-grained quality optimization dimensions, such as relevance, coherence, and helpfulness, overlooking the fine-grained specifics inherent to diverse long-form generation scenarios. To address this issue, we propose a framework using Adaptive Constraint-Enhanced reward for long-form generation Reinforcement Learning (ACE-RL). ACE-RL first automatically deconstructs each instruction into a set of fine-grained, adaptive constraint criteria by identifying its underlying intents and demands. Subsequently, we design a reward mechanism that quantifies the quality of long-form responses based on their satisfaction over corresponding constraints, converting subjective quality evaluation into constraint verification. Finally, we utilize reinforcement learning to guide models toward superior long-form generation capabilities. Experimental results demonstrate that our ACE-RL framework significantly outperforms existing SFT and RL baselines by 20.70% and 7.32% on WritingBench, and our top-performing model even surpasses proprietary systems like GPT-4o by 7.10%, providing a more effective training paradigm for LLMs to generate high-quality content across diverse long-form generation scenarios.

**arXiv ID:** 2509.04903
</details>

<details>
<summary><strong>Replicable Reinforcement Learning with Linear Function Approximation</strong> - Eric Eaton, Marcel Hussing, Michael Kearns, Aaron Roth, Sikata Bela Sengupta, Jessica Sorrell - [[pdf]](https://arxiv.org/pdf/2509.08660)</summary>

**Abstract:** Replication of experimental results has been a challenge faced by many scientific disciplines, including the field of machine learning. Recent work on the theory of machine learning has formalized replicability as the demand that an algorithm produce identical outcomes when executed twice on different samples from the same distribution. Provably replicable algorithms are especially interesting for reinforcement learning (RL), where algorithms are known to be unstable in practice. While replicable algorithms exist for tabular RL settings, extending these guarantees to more practical function approximation settings has remained an open problem. In this work, we make progress by developing replicable methods for linear function approximation in RL. We first introduce two efficient algorithms for replicable random design regression and uncentered covariance estimation, each of independent interest. We then leverage these tools to provide the first provably efficient replicable RL algorithms for linear Markov decision processes in both the generative model and episodic settings. Finally, we evaluate our algorithms experimentally and show how they can inspire more consistent neural policies.

**arXiv ID:** 2509.08660
</details>

<details>
<summary><strong>Reward function compression facilitates goal-dependent reinforcement learning</strong> - Gaia Molinaro, Anne G. E. Collins - [[pdf]](https://arxiv.org/pdf/2509.06810)</summary>

**Abstract:** Reinforcement learning agents learn from rewards, but humans can uniquely assign value to novel, abstract outcomes in a goal-dependent manner. However, this flexibility is cognitively costly, making learning less efficient. Here, we propose that goal-dependent learning is initially supported by a capacity-limited working memory system. With consistent experience, learners create a "compressed" reward function (a simplified rule defining the goal) which is then transferred to long-term memory and applied automatically upon receiving feedback. This process frees up working memory resources, boosting learning efficiency. We test this theory across six experiments. Consistent with our predictions, our findings demonstrate that learning is parametrically impaired by the size of the goal space, but improves when the goal space structure allows for compression. We also find faster reward processing to correlate with better learning performance, supporting the idea that as goal valuation becomes more automatic, more resources are available for learning. We leverage computational modeling to support this interpretation. Our work suggests that efficient goal-directed learning relies on compressing complex goal information into a stable reward function, shedding light on the cognitive mechanisms of human motivation. These findings generate new insights into the neuroscience of intrinsic motivation and could help improve behavioral techniques that support people in achieving their goals.

**arXiv ID:** 2509.06810
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
