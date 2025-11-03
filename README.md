# Agent arXiv Daily

**Last Updated:** 2025-11-03 02:18:26

**Total Papers:** 63

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
<summary><strong>Identity Management for Agentic AI: The new frontier of authorization, authentication, and security for an AI agent world</strong> - Tobin South, Subramanya Nagabhushanaradhya, Ayesha Dissanayaka, Sarah Cecchetti, George Fletcher, Victor Lu, Aldo Pietropaolo, Dean H. Saxe, Jeff Lombardo, Abhishek Maligehalli Shivalingaiah, Stan Bounev, Alex Keisner, Andor Kesselman, Zack Proser, Ginny Fahs, Andrew Bunyea, Ben Moskowitz, Atul Tulshibagwale, Dazza Greenwood, Jiaxin Pei, Alex Pentland - [[pdf]](https://arxiv.org/pdf/2510.25819)</summary>

**Abstract:** The rapid rise of AI agents presents urgent challenges in authentication, authorization, and identity management. Current agent-centric protocols (like MCP) highlight the demand for clarified best practices in authentication and authorization. Looking ahead, ambitions for highly autonomous agents raise complex long-term questions regarding scalable access control, agent-centric identities, AI workload differentiation, and delegated authority. This OpenID Foundation whitepaper is for stakeholders at the intersection of AI agents and access management. It outlines the resources already available for securing today's agents and presents a strategic agenda to address the foundational authentication, authorization, and identity problems pivotal for tomorrow's widespread autonomous systems.

**arXiv ID:** 2510.25819
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (8 papers)</h2></summary>

<details>
<summary><strong>Large Language Model-assisted Autonomous Vehicle Recovery from Immobilization</strong> - Zhipeng Bao, Qianwen Li - [[pdf]](https://arxiv.org/pdf/2510.26023)</summary>

**Abstract:** Despite significant advancements in recent decades, autonomous vehicles (AVs) continue to face challenges in navigating certain traffic scenarios where human drivers excel. In such situations, AVs often become immobilized, disrupting overall traffic flow. Current recovery solutions, such as remote intervention (which is costly and inefficient) and manual takeover (which excludes non-drivers and limits AV accessibility), are inadequate. This paper introduces StuckSolver, a novel Large Language Model (LLM) driven recovery framework that enables AVs to resolve immobilization scenarios through self-reasoning and/or passenger-guided decision-making. StuckSolver is designed as a plug-in add-on module that operates on top of the AV's existing perception-planning-control stack, requiring no modification to its internal architecture. Instead, it interfaces with standard sensor data streams to detect immobilization states, interpret environmental context, and generate high-level recovery commands that can be executed by the AV's native planner. We evaluate StuckSolver on the Bench2Drive benchmark and in custom-designed uncertainty scenarios. Results show that StuckSolver achieves near-state-of-the-art performance through autonomous self-reasoning alone and exhibits further improvements when passenger guidance is incorporated.

**arXiv ID:** 2510.26023
</details>

<details>
<summary><strong>Agentic AI Home Energy Management System: A Large Language Model Framework for Residential Load Scheduling</strong> - Reda El Makroum, Sebastian Zwickl-Bernhard, Lukas Kranzl - [[pdf]](https://arxiv.org/pdf/2510.26603)</summary>

**Abstract:** The electricity sector transition requires substantial increases in residential demand response capacity, yet Home Energy Management Systems (HEMS) adoption remains limited by user interaction barriers requiring translation of everyday preferences into technical parameters. While large language models have been applied to energy systems as code generators and parameter extractors, no existing implementation deploys LLMs as autonomous coordinators managing the complete workflow from natural language input to multi-appliance scheduling. This paper presents an agentic AI HEMS where LLMs autonomously coordinate multi-appliance scheduling from natural language requests to device control, achieving optimal scheduling without example demonstrations. A hierarchical architecture combining one orchestrator with three specialist agents uses the ReAct pattern for iterative reasoning, enabling dynamic coordination without hardcoded workflows while integrating Google Calendar for context-aware deadline extraction. Evaluation across three open-source models using real Austrian day-ahead electricity prices reveals substantial capability differences. Llama-3.3-70B successfully coordinates all appliances across all scenarios to match cost-optimal benchmarks computed via mixed-integer linear programming, while other models achieve perfect single-appliance performance but struggle to coordinate all appliances simultaneously. Progressive prompt engineering experiments demonstrate that analytical query handling without explicit guidance remains unreliable despite models' general reasoning capabilities. We open-source the complete system including orchestration logic, agent prompts, tools, and web interfaces to enable reproducibility, extension, and future research.

**arXiv ID:** 2510.26603
</details>

<details>
<summary><strong>Can Agent Conquer Web? Exploring the Frontiers of ChatGPT Atlas Agent in Web Games</strong> - Jingran Zhang, Ning Li, Justin Cui - [[pdf]](https://arxiv.org/pdf/2510.26298)</summary>

**Abstract:** OpenAI's ChatGPT Atlas introduces new capabilities for web interaction, enabling the model to analyze webpages, process user intents, and execute cursor and keyboard inputs directly within the browser. While its capacity for information retrieval tasks has been demonstrated, its performance in dynamic, interactive environments remains less explored. In this study, we conduct an early evaluation of Atlas's web interaction capabilities using browser-based games as test scenarios, including Google's T-Rex Runner, Sudoku, Flappy Bird, and this http URL. We employ in-game performance scores as quantitative metrics to assess performance across different task types. Our results show that Atlas performs strongly in logical reasoning tasks like Sudoku, completing puzzles significantly faster than human baselines, but struggles substantially in real-time games requiring precise timing and motor control, often failing to progress beyond initial obstacles. These findings suggest that while Atlas demonstrates capable analytical processing, there remain notable limitations in dynamic web environments requiring real-time interaction. The website of our project can be found at this https URL.

**arXiv ID:** 2510.26298
</details>

<details>
<summary><strong>Advancing Mobile GUI Agents: A Verifier-Driven Approach to Practical Deployment</strong> - Gaole Dai, Shiqi Jiang, Ting Cao, Yuanchun Li, Yuqing Yang, Rui Tan, Mo Li, Lili Qiu - [[pdf]](https://arxiv.org/pdf/2503.15937)</summary>

**Abstract:** We propose V-Droid, a mobile GUI task automation agent. Unlike previous mobile agents that utilize Large Language Models (LLMs) as generators to directly generate actions at each step, V-Droid employs LLMs as verifiers to evaluate candidate actions before making final decisions. To realize this novel paradigm, we introduce a comprehensive framework for constructing verifier-driven mobile agents: the discretized action space construction coupled with the prefilling-only workflow to accelerate the verification process, the pair-wise progress preference training to significantly enhance the verifier's decision-making capabilities, and the scalable human-agent joint annotation scheme to efficiently collect the necessary data at scale. V-Droid obtains a substantial task success rate across several public mobile task automation benchmarks: 59.5% on AndroidWorld, 38.3% on AndroidLab, and 49% on MobileAgentBench, surpassing existing agents by 5.2%, 2.1%, and 9%, respectively. Furthermore, V-Droid achieves a remarkably low latency of 4.3s per step, which is 6.1x faster compared with existing mobile agents. The source code is available at this https URL.

**arXiv ID:** 2503.15937
</details>

<details>
<summary><strong>AutoLibra: Agent Metric Induction from Open-Ended Human Feedback</strong> - Hao Zhu, Phil Cuvin, Xinkai Yu, Charlotte Ka Yee Yan, Jason Zhang, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2505.02820)</summary>

**Abstract:** Agents are predominantly evaluated and optimized via task success metrics, which are coarse, rely on manual design from experts, and fail to reward intermediate emergent behaviors. We propose **AutoLibra**, a framework for agent evaluation, that transforms open-ended human feedback *e.g.* "If you find that the button is disabled, don't click it again", or "This agent has too much autonomy to decide what to do on its own" into metrics for evaluating fine-grained behaviors in agent trajectories. AutoLibra accomplishes this by grounding feedback to an agent's behavior, clustering similar positive and negative behaviors, and creating concrete metrics with clear definitions and concrete examples, which can be used for prompting LLM-as-a-Judge as evaluators. We further propose two meta metrics to evaluate the alignment of a set of (induced) metrics with open feedback: "coverage" and "redundancy". Through optimizing these meta-metrics, we experimentally demonstrate AutoLibra's ability to induce more concrete agent evaluation metrics than the ones proposed in previous agent evaluation benchmarks and discover new metrics to analyze agents. We also present two applications of AutoLibra in agent improvement: First, we show that AutoLibra serve human prompt engineers for diagonalize agent failures and improve prompts iterative. Moreover, we find that AutoLibra can induce metrics for automatic optimization for agents, which makes agents improve through self-regulation. Our results suggest that AutoLibra is a powerful task-agnostic tool for evaluating and improving language agents.

**arXiv ID:** 2505.02820
</details>

<details>
<summary><strong>Empowering Agentic Video Analytics Systems with Video Language Models</strong> - Yuxuan Yan, Shiqi Jiang, Ting Cao, Yifan Yang, Qianqian Yang, Yuanchao Shu, Yuqing Yang, Lili Qiu - [[pdf]](https://arxiv.org/pdf/2505.00254)</summary>

**Abstract:** AI-driven video analytics has become increasingly important across diverse domains. However, existing systems are often constrained to specific, predefined tasks, limiting their adaptability in open-ended analytical scenarios. The recent emergence of Vision Language Models (VLMs) as transformative technologies offers significant potential for enabling open-ended video understanding, reasoning, and analytics. Nevertheless, their limited context windows present challenges when processing ultra-long video content, which is prevalent in real-world applications. To address this, we introduce AVA, a VLM-powered system designed for open-ended, advanced video analytics. AVA incorporates two key innovations: (1) the near real-time construction of Event Knowledge Graphs (EKGs) for efficient indexing of long or continuous video streams, and (2) an agentic retrieval-generation mechanism that leverages EKGs to handle complex and diverse queries. Comprehensive evaluations on public benchmarks, LVBench and VideoMME-Long, demonstrate that AVA achieves state-of-the-art performance, attaining 62.3% and 64.1% accuracy, respectively-significantly surpassing existing VLM and video Retrieval-Augmented Generation (RAG) systems. Furthermore, to evaluate video analytics in ultra-long and open-world video scenarios, we introduce a new benchmark, AVA-100. This benchmark comprises 8 videos, each exceeding 10 hours in duration, along with 120 manually annotated, diverse, and complex question-answer pairs. On AVA-100, AVA achieves top-tier performance with an accuracy of 75.8%. The source code of AVA is available at this https URL. The AVA-100 benchmark can be accessed at this https URL.

**arXiv ID:** 2505.00254
</details>

<details>
<summary><strong>Life-cycle Modeling and the Walking Behavior of the Pedestrian-Group as an Emergent Agent: With Empirical Data on the Cohesion of the Group Formation</strong> - Saleh Albeaik, Mohamad Alrished, Faisal Alsallum - [[pdf]](https://arxiv.org/pdf/2510.26534)</summary>

**Abstract:** This article investigates the pedestrian group as an emergent agent. The article explores empirical data to derive emergent agency and formation state spaces and outline recurring patterns of walking behavior. In this analysis, pedestrian trajectories extracted from surveillance videos are used along with manually annotated pedestrian group memberships. We conducted manual expert evaluation of observed groups, produced new manual annotations for relevant events pertaining to group behavior and extracted metrics relevant group formation. This information along with quantitative analysis was used to model the life-cycle and formation of the group agent. Those models give structure to expectations around walking behavior of groups; from pedestrian walking independently to the emergence of a collective intention where group members tended to maintain bounded distance between each other. Disturbances to this bounded distance often happened in association with changes in either their agency or their formation states. We summarized the patterns of behavior along with the sequences of state transitions into abstract patterns, which can aid in the development of more detailed group agents in simulation and in the design of engineering systems to interact with such groups.

**arXiv ID:** 2510.26534
</details>

<details>
<summary><strong>RECAP: Reproducing Copyrighted Data from LLMs Training with an Agentic Pipeline</strong> - André V. Duarte, Xuying li, Bin Zeng, Arlindo L. Oliveira, Lei Li, Zhuo Li - [[pdf]](https://arxiv.org/pdf/2510.25941)</summary>

**Abstract:** If we cannot inspect the training data of a large language model (LLM), how can we ever know what it has seen? We believe the most compelling evidence arises when the model itself freely reproduces the target content. As such, we propose RECAP, an agentic pipeline designed to elicit and verify memorized training data from LLM outputs. At the heart of RECAP is a feedback-driven loop, where an initial extraction attempt is evaluated by a secondary language model, which compares the output against a reference passage and identifies discrepancies. These are then translated into minimal correction hints, which are fed back into the target model to guide subsequent generations. In addition, to address alignment-induced refusals, RECAP includes a jailbreaking module that detects and overcomes such barriers. We evaluate RECAP on EchoTrace, a new benchmark spanning over 30 full books, and the results show that RECAP leads to substantial gains over single-iteration approaches. For instance, with GPT-4.1, the average ROUGE-L score for the copyrighted text extraction improved from 0.38 to 0.47 - a nearly 24% increase.

**arXiv ID:** 2510.25941
</details>

</details>

<details open>
<summary><h2>LLM Agents (10 papers)</h2></summary>

<details>
<summary><strong>Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles</strong> - Xinhang Li, Qing Guo, Junyu Chen, Zheng Guo, Shengzhe Xu, Lei Li, Lin Zhang - [[pdf]](https://arxiv.org/pdf/2510.26242)</summary>

**Abstract:** With increasing urban traffic complexity, Traffic Signal Control (TSC) is essential for optimizing traffic flow and improving road safety. Large Language Models (LLMs) emerge as promising approaches for TSC. However, they are prone to hallucinations in emergencies, leading to unreliable decisions that may cause substantial delays for emergency vehicles. Moreover, diverse intersection types present substantial challenges for traffic state encoding and cross-intersection training, limiting generalization across heterogeneous intersections. Therefore, this paper proposes Retrieval Augmented Generation (RAG)-enhanced distributed LLM agents with Emergency response for Generalizable TSC (REG-TSC). Firstly, this paper presents an emergency-aware reasoning framework, which dynamically adjusts reasoning depth based on the emergency scenario and is equipped with a novel Reviewer-based Emergency RAG (RERAG) to distill specific knowledge and guidance from historical cases, enhancing the reliability and rationality of agents' emergency decisions. Secondly, this paper designs a type-agnostic traffic representation and proposes a Reward-guided Reinforced Refinement (R3) for heterogeneous intersections. R3 adaptively samples training experience from diverse intersections with environment feedback-based priority and fine-tunes LLM agents with a designed reward-weighted likelihood loss, guiding REG-TSC toward high-reward policies across heterogeneous intersections. On three real-world road networks with 17 to 177 heterogeneous intersections, extensive experiments show that REG-TSC reduces travel time by 42.00%, queue length by 62.31%, and emergency vehicle waiting time by 83.16%, outperforming other state-of-the-art methods.

**arXiv ID:** 2510.26242
</details>

<details>
<summary><strong>Graph-Enhanced Policy Optimization in LLM Agent Training</strong> - Jiazhen Yuan, Wei Zhao, Zhengbiao Bai - [[pdf]](https://arxiv.org/pdf/2510.26270)</summary>

**Abstract:** Group based reinforcement learning (RL) has shown impressive results on complex reasoning and mathematical tasks. Yet, when applied to train multi-turn, interactive LLM agents, these methods often suffer from structural blindness-the inability to exploit the underlying connectivity of the environment. This manifests in three critical challenges: (1) inefficient, unguided exploration, (2) imprecise credit assignment due to overlooking pivotal states, and (3) myopic planning caused by static reward discounting. We address these issues with Graph-Enhanced Policy Optimization (GEPO), which dynamically constructs a state-transition graph from agent experience and employs graph-theoretic centrality to provide three synergistic learning signals: (1)structured intrinsic rewards that guide exploration toward high-impact states, (2) a graph-enhanced advantage function for topology-aware credit assignment, and (3) a dynamic discount factor adapted to each state's strategic value. On the ALFWorld, WebShop, and a proprietary Workbench benchmarks, GEPO demonstrates strong performance, achieving absolute success rate gains of +4.1%, +5.3%, and +10.9% over competitive baselines. These results highlight that explicitly modeling environmental structure is a robust, generalizable strategy for advancing LLM agent training.

**arXiv ID:** 2510.26270
</details>

<details>
<summary><strong>Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets</strong> - Gagan Bansal, Wenyue Hua, Zezhou Huang, Adam Fourney, Amanda Swearngin, Will Epperson, Tyler Payne, Jake M. Hofman, Brendan Lucier, Chinmay Singh, Markus Mobius, Akshay Nambi, Archana Yadav, Kevin Gao, David M. Rothschild, Aleksandrs Slivkins, Daniel G. Goldstein, Hussein Mozannar, Nicole Immorlica, Maya Murad, Matthew Vogel, Subbarao Kambhampati, Eric Horvitz, Saleema Amershi - [[pdf]](https://arxiv.org/pdf/2510.25779)</summary>

**Abstract:** As LLM agents advance, they are increasingly mediating economic decisions, ranging from product discovery to transactions, on behalf of users. Such applications promise benefits but also raise many questions about agent accountability and value for users. Addressing these questions requires understanding how agents behave in realistic market conditions. However, previous research has largely evaluated agents in constrained settings, such as single-task marketplaces (e.g., negotiation) or structured two-agent interactions. Real-world markets are fundamentally different: they require agents to handle diverse economic activities and coordinate within large, dynamic ecosystems where multiple agents with opaque behaviors may engage in open-ended dialogues. To bridge this gap, we investigate two-sided agentic marketplaces where Assistant agents represent consumers and Service agents represent competing businesses. To study these interactions safely, we develop Magentic-Marketplace-- a simulated environment where Assistants and Services can operate. This environment enables us to study key market dynamics: the utility agents achieve, behavioral biases, vulnerability to manipulation, and how search mechanisms shape market outcomes. Our experiments show that frontier models can approach optimal welfare-- but only under ideal search conditions. Performance degrades sharply with scale, and all models exhibit severe first-proposal bias, creating 10-30x advantages for response speed over quality. These findings reveal how behaviors emerge across market conditions, informing the design of fair and efficient agentic marketplaces.

**arXiv ID:** 2510.25779
</details>

<details>
<summary><strong>SIRAJ: Diverse and Efficient Red-Teaming for LLM Agents via Distilled Structured Reasoning</strong> - Kaiwen Zhou, Ahmed Elgohary, A S M Iftekhar, Amin Saied - [[pdf]](https://arxiv.org/pdf/2510.26037)</summary>

**Abstract:** The ability of LLM agents to plan and invoke tools exposes them to new safety risks, making a comprehensive red-teaming system crucial for discovering vulnerabilities and ensuring their safe deployment. We present SIRAJ: a generic red-teaming framework for arbitrary black-box LLM agents. We employ a dynamic two-step process that starts with an agent definition and generates diverse seed test cases that cover various risk outcomes, tool-use trajectories, and risk sources. Then, it iteratively constructs and refines model-based adversarial attacks based on the execution trajectories of former attempts. To optimize the red-teaming cost, we present a model distillation approach that leverages structured forms of a teacher model's reasoning to train smaller models that are equally effective. Across diverse evaluation agent settings, our seed test case generation approach yields 2 -- 2.5x boost to the coverage of risk outcomes and tool-calling trajectories. Our distilled 8B red-teamer model improves attack success rate by 100%, surpassing the 671B Deepseek-R1 model. Our ablations and analyses validate the effectiveness of the iterative framework, structured reasoning, and the generalization of our red-teamer models.

**arXiv ID:** 2510.26037
</details>

<details>
<summary><strong>Linking Heterogeneous Data with Coordinated Agent Flows for Social Media Analysis</strong> - Shifu Chen, Dazhen Deng, Zhihong Xu, Sijia Xu, Tai-Quan Peng, Yingcai Wu - [[pdf]](https://arxiv.org/pdf/2510.26172)</summary>

**Abstract:** Social media platforms generate massive volumes of heterogeneous data, capturing user behaviors, textual content, temporal dynamics, and network structures. Analyzing such data is crucial for understanding phenomena such as opinion dynamics, community formation, and information diffusion. However, discovering insights from this complex landscape is exploratory, conceptually challenging, and requires expertise in social media mining and visualization. Existing automated approaches, though increasingly leveraging large language models (LLMs), remain largely confined to structured tabular data and cannot adequately address the heterogeneity of social media analysis. We present SIA (Social Insight Agents), an LLM agent system that links heterogeneous multi-modal data -- including raw inputs (e.g., text, network, and behavioral data), intermediate outputs, mined analytical results, and visualization artifacts -- through coordinated agent flows. Guided by a bottom-up taxonomy that connects insight types with suitable mining and visualization techniques, SIA enables agents to plan and execute coherent analysis strategies. To ensure multi-modal integration, it incorporates a data coordinator that unifies tabular, textual, and network data into a consistent flow. Its interactive interface provides a transparent workflow where users can trace, validate, and refine the agent's reasoning, supporting both adaptability and trustworthiness. Through expert-centered case studies and quantitative evaluation, we show that SIA effectively discovers diverse and meaningful insights from social media while supporting human-agent collaboration in complex analytical tasks.

**arXiv ID:** 2510.26172
</details>

<details>
<summary><strong>Simulating and Experimenting with Social Media Mobilization Using LLM Agents</strong> - Sadegh Shirani, Mohsen Bayati - [[pdf]](https://arxiv.org/pdf/2510.26494)</summary>

**Abstract:** Online social networks have transformed the ways in which political mobilization messages are disseminated, raising new questions about how peer influence operates at scale. Building on the landmark 61-million-person Facebook experiment \citep{bond201261}, we develop an agent-based simulation framework that integrates real U.S. Census demographic distributions, authentic Twitter network topology, and heterogeneous large language model (LLM) agents to examine the effect of mobilization messages on voter turnout. Each simulated agent is assigned demographic attributes, a personal political stance, and an LLM variant (\texttt{GPT-4.1}, \texttt{GPT-4.1-Mini}, or \texttt{GPT-4.1-Nano}) reflecting its political sophistication. Agents interact over realistic social network structures, receiving personalized feeds and dynamically updating their engagement behaviors and voting intentions. Experimental conditions replicate the informational and social mobilization treatments of the original Facebook study. Across scenarios, the simulator reproduces qualitative patterns observed in field experiments, including stronger mobilization effects under social message treatments and measurable peer spillovers. Our framework provides a controlled, reproducible environment for testing counterfactual designs and sensitivity analyses in political mobilization research, offering a bridge between high-validity field experiments and flexible computational modeling.\footnote{Code and data available at this https URL}

**arXiv ID:** 2510.26494
</details>

<details>
<summary><strong>Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents</strong> - Gil Pasternak, Dheeraj Rajagopal, Julia White, Dhruv Atreja, Matthew Thomas, George Hurn-Maloney, Ash Lewis - [[pdf]](https://arxiv.org/pdf/2510.19771)</summary>

**Abstract:** LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. To address this gap, we present PROBE (Proactive Resolution Of BottlEnecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and expose promising future research directions.

**arXiv ID:** 2510.19771
</details>

<details>
<summary><strong>SignalLLM: A General-Purpose LLM Agent Framework for Automated Signal Processing</strong> - Junlong Ke, Qiying Hu, Shenghai Yuan, Yuecong Xu, Jianfei Yang - [[pdf]](https://arxiv.org/pdf/2509.17197)</summary>

**Abstract:** Modern signal processing (SP) pipelines, whether model-based or data-driven, often constrained by complex and fragmented workflow, rely heavily on expert knowledge and manual engineering, and struggle with adaptability and generalization under limited data. In contrast, Large Language Models (LLMs) offer strong reasoning capabilities, broad general-purpose knowledge, in-context learning, and cross-modal transfer abilities, positioning them as powerful tools for automating and generalizing SP workflows. Motivated by these potentials, we introduce SignalLLM, the first general-purpose LLM-based agent framework for general SP tasks. Unlike prior LLM-based SP approaches that are limited to narrow applications or tricky prompting, SignalLLM introduces a principled, modular architecture. It decomposes high-level SP goals into structured subtasks via in-context learning and domain-specific retrieval, followed by hierarchical planning through adaptive retrieval-augmented generation (RAG) and refinement; these subtasks are then executed through prompt-based reasoning, cross-modal reasoning, code synthesis, model invocation, or data-driven LLM-assisted modeling. Its generalizable design enables the flexible selection of problem solving strategies across different signal modalities, task types, and data conditions. We demonstrate the versatility and effectiveness of SignalLLM through five representative tasks in communication and sensing, such as radar target detection, human activity recognition, and text compression. Experimental results show superior performance over traditional and existing LLM-based methods, particularly in few-shot and zero-shot settings.

**arXiv ID:** 2509.17197
</details>

<details>
<summary><strong>When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents</strong> - Lingfei Qian, Xueqing Peng, Yan Wang, Vincent Jim Zhang, Huan He, Hanley Smith, Yi Han, Yueru He, Haohang Li, Yupeng Cao, Yangyang Yu, Alejandro Lopez-Lira, Peng Lu, Jian-Yun Nie, Guojun Xiong, Jimin Huang, Sophia Ananiadou - [[pdf]](https://arxiv.org/pdf/2510.11695)</summary>

**Abstract:** Although Large Language Model (LLM)-based agents are increasingly used in financial trading, it remains unclear whether they can reason and adapt in live markets, as most studies test models instead of agents, cover limited periods and assets, and rely on unverified data. To address these gaps, we introduce Agent Market Arena (AMA), the first lifelong, real-time benchmark for evaluating LLM-based trading agents across multiple markets. AMA integrates verified trading data, expert-checked news, and diverse agent architectures within a unified trading framework, enabling fair and continuous comparison under real conditions. It implements four agents, including InvestorAgent as a single-agent baseline, TradeAgent and HedgeFundAgent with different risk styles, and DeepFundAgent with memory-based reasoning, and evaluates them across GPT-4o, GPT-4.1, Claude-3.5-haiku, Claude-sonnet-4, and Gemini-2.0-flash. Live experiments on both cryptocurrency and stock markets demonstrate that agent frameworks display markedly distinct behavioral patterns, spanning from aggressive risk-taking to conservative decision-making, whereas model backbones contribute less to outcome variation. AMA thus establishes a foundation for rigorous, reproducible, and continuously evolving evaluation of financial reasoning and trading intelligence in LLM-based agents.

**arXiv ID:** 2510.11695
</details>

<details>
<summary><strong>TEXT2DB: Integration-Aware Information Extraction with Large Language Model Agents</strong> - Yizhu Jiao, Sha Li, Sizhe Zhou, Heng Ji, Jiawei Han - [[pdf]](https://arxiv.org/pdf/2510.24014)</summary>

**Abstract:** The task of information extraction (IE) is to extract structured knowledge from text. However, it is often not straightforward to utilize IE output due to the mismatch between the IE ontology and the downstream application needs. We propose a new formulation of IE TEXT2DB that emphasizes the integration of IE output and the target database (or knowledge base). Given a user instruction, a document set, and a database, our task requires the model to update the database with values from the document set to satisfy the user instruction. This task requires understanding user instructions for what to extract and adapting to the given DB/KB schema for how to extract on the fly. To evaluate this new task, we introduce a new benchmark featuring common demands such as data infilling, row population, and column addition. In addition, we propose an LLM agent framework OPAL (Observe-PlanAnalyze LLM) which includes an Observer component that interacts with the database, the Planner component that generates a code-based plan with calls to IE models, and the Analyzer component that provides feedback regarding code quality before execution. Experiments show that OPAL can successfully adapt to diverse database schemas by generating different code plans and calling the required IE models. We also highlight difficult cases such as dealing with large databases with complex dependencies and extraction hallucination, which we believe deserve further investigation. Source code: this https URL

**arXiv ID:** 2510.24014
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (17 papers)</h2></summary>

<details>
<summary><strong>The FM Agent</strong> - Annan Li, Chufan Wu, Zengle Ge, Yee Hin Chong, Zhinan Hou, Lizhe Cao, Cheng Ju, Jianmin Wu, Huaiming Li, Haobo Zhang, Shenghao Feng, Mo Zhao, Fengzhi Qiu, Rui Yang, Mengmeng Zhang, Wenyi Zhu, Yingying Sun, Quan Sun, Shunhao Yan, Danyu Liu, Dawei Yin, Dou Shen - [[pdf]](https://arxiv.org/pdf/2510.26144)</summary>

**Abstract:** Large language models (LLMs) are catalyzing the development of autonomous AI research agents for scientific and engineering discovery. We present FM Agent, a novel and general-purpose multi-agent framework that leverages a synergistic combination of LLM-based reasoning and large-scale evolutionary search to address complex real-world challenges. The core of FM Agent integrates several key innovations: 1) a cold-start initialization phase incorporating expert guidance, 2) a novel evolutionary sampling strategy for iterative optimization, 3) domain-specific evaluators that combine correctness, effectiveness, and LLM-supervised feedback, and 4) a distributed, asynchronous execution infrastructure built on Ray. Demonstrating broad applicability, our system has been evaluated across diverse domains, including operations research, machine learning, GPU kernel optimization, and classical mathematical problems. FM Agent reaches state-of-the-art results autonomously, without human interpretation or tuning -- 1976.3 on ALE-Bench (+5.2\%), 43.56\% on MLE-Bench (+4.0pp), up to 20x speedups on KernelBench, and establishes new state-of-the-art(SOTA) results on several classical mathematical problems. Beyond academic benchmarks, FM Agent shows considerable promise for both large-scale enterprise R\&D workflows and fundamental scientific research, where it can accelerate innovation, automate complex discovery processes, and deliver substantial engineering and scientific advances with broader societal impact.

**arXiv ID:** 2510.26144
</details>

<details>
<summary><strong>Delegated Authorization for Agents Constrained to Semantic Task-to-Scope Matching</strong> - Majed El Helou, Chiara Troiani, Benjamin Ryder, Jean Diaconu, Hervé Muyal, Marcelo Yannuzzi - [[pdf]](https://arxiv.org/pdf/2510.26702)</summary>

**Abstract:** Authorizing Large Language Model driven agents to dynamically invoke tools and access protected resources introduces significant risks, since current methods for delegating authorization grant overly broad permissions and give access to tools allowing agents to operate beyond the intended task scope. We introduce and assess a delegated authorization model enabling authorization servers to semantically inspect access requests to protected resources, and issue access tokens constrained to the minimal set of scopes necessary for the agents' assigned tasks. Given the unavailability of datasets centered on delegated authorization flows, particularly including both semantically appropriate and inappropriate scope requests for a given task, we introduce ASTRA, a dataset and data generation pipeline for benchmarking semantic matching between tasks and scopes. Our experiments show both the potential and current limitations of model-based matching, particularly as the number of scopes needed for task completion increases. Our results highlight the need for further research into semantic matching techniques enabling intent-aware authorization for multi-agent and tool-augmented applications, including fine-grained control, such as Task-Based Access Control (TBAC).

**arXiv ID:** 2510.26702
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning for Market Making: Competition without Collusion</strong> - Ziyi Wang, Carmine Ventre, Maria Polukarov - [[pdf]](https://arxiv.org/pdf/2510.25929)</summary>

**Abstract:** Algorithmic collusion has emerged as a central question in AI: Will the interaction between different AI agents deployed in markets lead to collusion? More generally, understanding how emergent behavior, be it a cartel or market dominance from more advanced bots, affects the market overall is an important research question.
We propose a hierarchical multi-agent reinforcement learning framework to study algorithmic collusion in market making. The framework includes a self-interested market maker (Agent~A), which is trained in an uncertain environment shaped by an adversary, and three bottom-layer competitors: the self-interested Agent~B1 (whose objective is to maximize its own PnL), the competitive Agent~B2 (whose objective is to minimize the PnL of its opponent), and the hybrid Agent~B$^\star$, which can modulate between the behavior of the other two. To analyze how these agents shape the behavior of each other and affect market outcomes, we propose interaction-level metrics that quantify behavioral asymmetry and system-level dynamics, while providing signals potentially indicative of emergent interaction patterns.
Experimental results show that Agent~B2 secures dominant performance in a zero-sum setting against B1, aggressively capturing order flow while tightening average spreads, thus improving market execution efficiency. In contrast, Agent~B$^\star$ exhibits a self-interested inclination when co-existing with other profit-seeking agents, securing dominant market share through adaptive quoting, yet exerting a milder adverse impact on the rewards of Agents~A and B1 compared to B2. These findings suggest that adaptive incentive control supports more sustainable strategic co-existence in heterogeneous agent environments and offers a structured lens for evaluating behavioral design in algorithmic trading systems.

**arXiv ID:** 2510.25929
</details>

<details>
<summary><strong>Network-Constrained Policy Optimization for Adaptive Multi-agent Vehicle Routing</strong> - Fazel Arasteh, Arian Haghparast, Manos Papagelis - [[pdf]](https://arxiv.org/pdf/2510.26089)</summary>

**Abstract:** Traffic congestion in urban road networks leads to longer trip times and higher emissions, especially during peak periods. While the Shortest Path First (SPF) algorithm is optimal for a single vehicle in a static network, it performs poorly in dynamic, multi-vehicle settings, often worsening congestion by routing all vehicles along identical paths. We address dynamic vehicle routing through a multi-agent reinforcement learning (MARL) framework for coordinated, network-aware fleet navigation. We first propose Adaptive Navigation (AN), a decentralized MARL model where each intersection agent provides routing guidance based on (i) local traffic and (ii) neighborhood state modeled using Graph Attention Networks (GAT). To improve scalability in large networks, we further propose Hierarchical Hub-based Adaptive Navigation (HHAN), an extension of AN that assigns agents only to key intersections (hubs). Vehicles are routed hub-to-hub under agent control, while SPF handles micro-routing within each hub region. For hub coordination, HHAN adopts centralized training with decentralized execution (CTDE) under the Attentive Q-Mixing (A-QMIX) framework, which aggregates asynchronous vehicle decisions via attention. Hub agents use flow-aware state features that combine local congestion and predictive dynamics for proactive routing. Experiments on synthetic grids and real urban maps (Toronto, Manhattan) show that AN reduces average travel time versus SPF and learning baselines, maintaining 100% routing success. HHAN scales to networks with hundreds of intersections, achieving up to 15.9% improvement under heavy traffic. These findings highlight the potential of network-constrained MARL for scalable, coordinated, and congestion-aware routing in intelligent transportation systems.

**arXiv ID:** 2510.26089
</details>

<details>
<summary><strong>The Geometry of Dialogue: Graphing Language Models to Reveal Synergistic Teams for Multi-Agent Collaboration</strong> - Kotaro Furuya, Yuichi Kitagawa - [[pdf]](https://arxiv.org/pdf/2510.26352)</summary>

**Abstract:** While a multi-agent approach based on large language models (LLMs) represents a promising strategy to surpass the capabilities of single models, its success is critically dependent on synergistic team composition. However, forming optimal teams is a significant challenge, as the inherent opacity of most models obscures the internal characteristics necessary for effective collaboration. In this paper, we propose an interaction-centric framework for automatic team composition that does not require any prior knowledge including their internal architectures, training data, or task performances. Our method constructs a "language model graph" that maps relationships between models from the semantic coherence of pairwise conversations, and then applies community detection to identify synergistic model clusters. Our experiments with diverse LLMs demonstrate that the proposed method discovers functionally coherent groups that reflect their latent specializations. Priming conversations with specific topics identified synergistic teams which outperform random baselines on downstream benchmarks and achieve comparable accuracy to that of manually-curated teams based on known model specializations. Our findings provide a new basis for the automated design of collaborative multi-agent LLM teams.

**arXiv ID:** 2510.26352
</details>

<details>
<summary><strong>Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems</strong> - Fulin Lin, Shaowen Chen, Ruishan Fang, Hongwei Wang, Tao Lin - [[pdf]](https://arxiv.org/pdf/2510.26585)</summary>

**Abstract:** While Multi-Agent Systems (MAS) excel at complex tasks, their growing autonomy with operational complexity often leads to critical inefficiencies, such as excessive token consumption and failures arising from misinformation. Existing methods primarily focus on post-hoc failure attribution, lacking proactive, real-time interventions to enhance robustness and efficiency. To this end, we introduce SupervisorAgent, a lightweight and modular framework for runtime, adaptive supervision that operates without altering the base agent's architecture. Triggered by an LLM-free adaptive filter, SupervisorAgent intervenes at critical junctures to proactively correct errors, guide inefficient behaviors, and purify observations. On the challenging GAIA benchmark, SupervisorAgent reduces the token consumption of the Smolagent framework by an average of 29.45% without compromising its success rate. Extensive experiments across five additional benchmarks (math reasoning, code generation, and question answering) and various SoTA foundation models validate the broad applicability and robustness of our approach. The code is available at this https URL.

**arXiv ID:** 2510.26585
</details>

<details>
<summary><strong>A General Incentives-Based Framework for Fairness in Multi-agent Resource Allocation</strong> - Ashwin Kumar, William Yeoh - [[pdf]](https://arxiv.org/pdf/2510.26740)</summary>

**Abstract:** We introduce the General Incentives-based Framework for Fairness (GIFF), a novel approach for fair multi-agent resource allocation that infers fair decision-making from standard value functions. In resource-constrained settings, agents optimizing for efficiency often create inequitable outcomes. Our approach leverages the action-value (Q-)function to balance efficiency and fairness without requiring additional training. Specifically, our method computes a local fairness gain for each action and introduces a counterfactual advantage correction term to discourage over-allocation to already well-off agents. This approach is formalized within a centralized control setting, where an arbitrator uses the GIFF-modified Q-values to solve an allocation problem.
Empirical evaluations across diverse domains, including dynamic ridesharing, homelessness prevention, and a complex job allocation task-demonstrate that our framework consistently outperforms strong baselines and can discover far-sighted, equitable policies. The framework's effectiveness is supported by a theoretical foundation; we prove its fairness surrogate is a principled lower bound on the true fairness improvement and that its trade-off parameter offers monotonic tuning. Our findings establish GIFF as a robust and principled framework for leveraging standard reinforcement learning components to achieve more equitable outcomes in complex multi-agent systems.

**arXiv ID:** 2510.26740
</details>

<details>
<summary><strong>MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks</strong> - Yinghao Zhu, Ziyi He, Haoran Hu, Xiaochen Zheng, Xichen Zhang, Zixiang Wang, Junyi Gao, Liantao Ma, Lequan Yu - [[pdf]](https://arxiv.org/pdf/2505.12371)</summary>

**Abstract:** The rapid advancement of Large Language Models (LLMs) has stimulated interest in multi-agent collaboration for addressing complex medical tasks. However, the practical advantages of multi-agent collaboration approaches remain insufficiently understood. Existing evaluations often lack generalizability, failing to cover diverse tasks reflective of real-world clinical practice, and frequently omit rigorous comparisons against both single-LLM-based and established conventional methods. To address this critical gap, we introduce MedAgentBoard, a comprehensive benchmark for the systematic evaluation of multi-agent collaboration, single-LLM, and conventional approaches. MedAgentBoard encompasses four diverse medical task categories: (1) medical (visual) question answering, (2) lay summary generation, (3) structured Electronic Health Record (EHR) predictive modeling, and (4) clinical workflow automation, across text, medical images, and structured EHR data. Our extensive experiments reveal a nuanced landscape: while multi-agent collaboration demonstrates benefits in specific scenarios, such as enhancing task completeness in clinical workflow automation, it does not consistently outperform advanced single LLMs (e.g., in textual medical QA) or, critically, specialized conventional methods that generally maintain better performance in tasks like medical VQA and EHR-based prediction. MedAgentBoard offers a vital resource and actionable insights, emphasizing the necessity of a task-specific, evidence-based approach to selecting and developing AI solutions in medicine. It underscores that the inherent complexity and overhead of multi-agent collaboration must be carefully weighed against tangible performance gains. All code, datasets, detailed prompts, and experimental results are open-sourced at this https URL.

**arXiv ID:** 2505.12371
</details>

<details>
<summary><strong>Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</strong> - Ashmi Banerjee, Adithi Satish, Fitri Nur Aisyah, Wolfgang Wörndl, Yashar Deldjoo - [[pdf]](https://arxiv.org/pdf/2508.15030)</summary>

**Abstract:** We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents -- Personalization, Popularity, and Sustainability generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses. Experiments on European city queries show that Collab-REC improves diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that often remain overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems.

**arXiv ID:** 2508.15030
</details>

<details>
<summary><strong>LAFA: Agentic LLM-Driven Federated Analytics over Decentralized Data Sources</strong> - Haichao Ji, Zibo Wang, Cheng Pan, Meng Han, Yifei Zhu, Dan Wang, Zhu Han - [[pdf]](https://arxiv.org/pdf/2510.18477)</summary>

**Abstract:** Large Language Models (LLMs) have shown great promise in automating data analytics tasks by interpreting natural language queries and generating multi-operation execution plans. However, existing LLM-agent-based analytics frameworks operate under the assumption of centralized data access, offering little to no privacy protection. In contrast, federated analytics (FA) enables privacy-preserving computation across distributed data sources, but lacks support for natural language input and requires structured, machine-readable queries. In this work, we present LAFA, the first system that integrates LLM-agent-based data analytics with FA. LAFA introduces a hierarchical multi-agent architecture that accepts natural language queries and transforms them into optimized, executable FA workflows. A coarse-grained planner first decomposes complex queries into sub-queries, while a fine-grained planner maps each subquery into a Directed Acyclic Graph of FA operations using prior structural knowledge. To improve execution efficiency, an optimizer agent rewrites and merges multiple DAGs, eliminating redundant operations and minimizing computational and communicational overhead. Our experiments demonstrate that LAFA consistently outperforms baseline prompting strategies by achieving higher execution plan success rates and reducing resource-intensive FA operations by a substantial margin. This work establishes a practical foundation for privacy-preserving, LLM-driven analytics that supports natural language input in the FA setting.

**arXiv ID:** 2510.18477
</details>

<details>
<summary><strong>Multi-Agent Evolve: LLM Self-Improve through Co-evolution</strong> - Yixing Chen, Yiding Wang, Siqi Zhu, Haofei Yu, Tao Feng, Muhan Zhang, Mostofa Patwary, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2510.23595)</summary>

**Abstract:** Reinforcement Learning (RL) has demonstrated significant potential in enhancing the reasoning capabilities of large language models (LLMs). However, the success of RL for LLMs heavily relies on human-curated datasets and verifiable rewards, which limit their scalability and generality. Recent Self-Play RL methods, inspired by the success of the paradigm in games and Go, aim to enhance LLM reasoning capabilities without human-annotated data. However, their methods primarily depend on a grounded environment for feedback (e.g., a Python interpreter or a game engine); extending them to general domains remains challenging. To address these challenges, we propose Multi-Agent Evolve (MAE), a framework that enables LLMs to self-evolve in solving diverse tasks, including mathematics, reasoning, and general knowledge Q&A. The core design of MAE is based on a triplet of interacting agents (Proposer, Solver, Judge) that are instantiated from a single LLM, and applies reinforcement learning to optimize their behaviors. The Proposer generates questions, the Solver attempts solutions, and the Judge evaluates both while co-evolving. Experiments on Qwen2.5-3B-Instruct demonstrate that MAE achieves an average improvement of 4.54% on multiple benchmarks. These results highlight MAE as a scalable, data-efficient method for enhancing the general reasoning abilities of LLMs with minimal reliance on human-curated supervision.

**arXiv ID:** 2510.23595
</details>

<details>
<summary><strong>Completion $\neq$ Collaboration: Scaling Collaborative Effort with Agents</strong> - Shannon Zejiang Shen, Valerie Chen, Ken Gu, Alexis Ross, Zixian Ma, Jillian Ross, Alex Gu, Chenglei Si, Wayne Chi, Andi Peng, Jocelyn J Shen, Ameet Talwalkar, Tongshuang Wu, David Sontag - [[pdf]](https://arxiv.org/pdf/2510.25744)</summary>

**Abstract:** Current evaluations of agents remain centered around one-shot task completion, failing to account for the inherently iterative and collaborative nature of many real-world problems, where human goals are often underspecified and evolve. We argue for a shift from building and assessing task completion agents to developing collaborative agents, assessed not only by the quality of their final outputs but by how well they engage with and enhance human effort throughout the problem-solving process. To support this shift, we introduce collaborative effort scaling, a framework that captures how an agent's utility grows with increasing user involvement. Through case studies and simulated evaluations, we show that state-of-the-art agents often underperform in multi-turn, real-world scenarios, revealing a missing ingredient in agent design: the ability to sustain engagement and scaffold user understanding. Collaborative effort scaling offers a lens for diagnosing agent behavior and guiding development toward more effective interactions.

**arXiv ID:** 2510.25744
</details>

<details>
<summary><strong>Adaptive Context Length Optimization with Low-Frequency Truncation for Multi-Agent Reinforcement Learning</strong> - Wenchang Duan, Yaoliang Yu, Jiwan He, Yi Shi - [[pdf]](https://arxiv.org/pdf/2510.26389)</summary>

**Abstract:** Recently, deep multi-agent reinforcement learning (MARL) has demonstrated promising performance for solving challenging tasks, such as long-term dependencies and non-Markovian environments. Its success is partly attributed to conditioning policies on large fixed context length. However, such large fixed context lengths may lead to limited exploration efficiency and redundant information. In this paper, we propose a novel MARL framework to obtain adaptive and effective contextual information. Specifically, we design a central agent that dynamically optimizes context length via temporal gradient analysis, enhancing exploration to facilitate convergence to global optima in MARL. Furthermore, to enhance the adaptive optimization capability of the context length, we present an efficient input representation for the central agent, which effectively filters redundant information. By leveraging a Fourier-based low-frequency truncation method, we extract global temporal trends across decentralized agents, providing an effective and efficient representation of the MARL environment. Extensive experiments demonstrate that the proposed method achieves state-of-the-art (SOTA) performance on long-term dependency tasks, including PettingZoo, MiniGrid, Google Research Football (GRF), and StarCraft Multi-Agent Challenge v2 (SMACv2).

**arXiv ID:** 2510.26389
</details>

<details>
<summary><strong>A Multi-agent Large Language Model Framework to Automatically Assess Performance of a Clinical AI Triage Tool</strong> - Adam E. Flanders, Yifan Peng, Luciano Prevedello, Robyn Ball, Errol Colak, Prahlad Menon, George Shih, Hui-Ming Lin, Paras Lakhani - [[pdf]](https://arxiv.org/pdf/2510.26498)</summary>

**Abstract:** Purpose: The purpose of this study was to determine if an ensemble of multiple LLM agents could be used collectively to provide a more reliable assessment of a pixel-based AI triage tool than a single LLM.
Methods: 29,766 non-contrast CT head exams from fourteen hospitals were processed by a commercial intracranial hemorrhage (ICH) AI detection tool. Radiology reports were analyzed by an ensemble of eight open-source LLM models and a HIPAA compliant internal version of GPT-4o using a single multi-shot prompt that assessed for presence of ICH. 1,726 examples were manually reviewed. Performance characteristics of the eight open-source models and consensus were compared to GPT-4o. Three ideal consensus LLM ensembles were tested for rating the performance of the triage tool.
Results: The cohort consisted of 29,766 head CTs exam-report pairs. The highest AUC performance was achieved with llama3.3:70b and GPT-4o (AUC= 0.78). The average precision was highest for Llama3.3:70b and GPT-4o (AP=0.75 & 0.76). Llama3.3:70b had the highest F1 score (0.81) and recall (0.85), greater precision (0.78), specificity (0.72), and MCC (0.57). Using MCC (95% CI) the ideal combination of LLMs were: Full-9 Ensemble 0.571 (0.552-0.591), Top-3 Ensemble 0.558 (0.537-0.579), Consensus 0.556 (0.539-0.574), and GPT4o 0.522 (0.500-0.543). No statistically significant differences were observed between Top-3, Full-9, and Consensus (p > 0.05).
Conclusion: An ensemble of medium to large sized open-source LLMs provides a more consistent and reliable method to derive a ground truth retrospective evaluation of a clinical AI triage tool over a single LLM alone.

**arXiv ID:** 2510.26498
</details>

<details>
<summary><strong>Nexus: Execution-Grounded Multi-Agent Test Oracle Synthesis</strong> - Dong Huang, Mingzhe Du, Jie M. Zhang, Zheng Lin, Meng Luo, Qianru Zhang, See-Kiong Ng - [[pdf]](https://arxiv.org/pdf/2510.26423)</summary>

**Abstract:** Test oracle generation in non-regression testing is a longstanding challenge in software engineering, where the goal is to produce oracles that can accurately determine whether a function under test (FUT) behaves as intended for a given input. In this paper, we introduce Nexus, a novel multi-agent framework to address this challenge. Nexus generates test oracles by leveraging a diverse set of specialized agents that synthesize test oracles through a structured process of deliberation, validation, and iterative self-refinement. During the deliberation phase, a panel of four specialist agents, each embodying a distinct testing philosophy, collaboratively critiques and refines an initial set of test oracles. Then, in the validation phase, Nexus generates a plausible candidate implementation of the FUT and executes the proposed oracles against it in a secure sandbox. For any oracle that fails this execution-based check, Nexus activates an automated selfrefinement loop, using the specific runtime error to debug and correct the oracle before re-validation. Our extensive evaluation on seven diverse benchmarks demonstrates that Nexus consistently and substantially outperforms state-of-theart baselines. For instance, Nexus improves the test-level oracle accuracy on the LiveCodeBench from 46.30% to 57.73% for GPT-4.1-Mini. The improved accuracy also significantly enhances downstream tasks: the bug detection rate of GPT4.1-Mini generated test oracles on HumanEval increases from 90.91% to 95.45% for Nexus compared to baselines, and the success rate of automated program repair improves from 35.23% to 69.32%.

**arXiv ID:** 2510.26423
</details>

<details>
<summary><strong>Optimal Information Combining for Multi-Agent Systems Using Adaptive Bias Learning</strong> - Siavash M. Alamouti, Fay Arjomandi - [[pdf]](https://arxiv.org/pdf/2510.25793)</summary>

**Abstract:** Modern multi-agent systems ranging from sensor networks monitoring critical infrastructure to crowdsourcing platforms aggregating human intelligence can suffer significant performance degradation due to systematic biases that vary with environmental conditions. Current approaches either ignore these biases, leading to suboptimal decisions, or require expensive calibration procedures that are often infeasible in practice. This performance gap has real consequences: inaccurate environmental monitoring, unreliable financial predictions, and flawed aggregation of human judgments. This paper addresses the fundamental question: when can we learn and correct for these unknown biases to recover near-optimal performance, and when is such learning futile? We develop a theoretical framework that decomposes biases into learnable systematic components and irreducible stochastic components, introducing the concept of learnability ratio as the fraction of bias variance predictable from observable covariates. This ratio determines whether bias learning is worthwhile for a given system. We prove that the achievable performance improvement is fundamentally bounded by this learnability ratio, providing system designers with quantitative guidance on when to invest in bias learning versus simpler approaches. We present the Adaptive Bias Learning and Optimal Combining (ABLOC) algorithm, which iteratively learns bias-correcting transformations while optimizing combination weights through closedform solutions, guaranteeing convergence to these theoretical bounds. Experimental validation demonstrates that systems with high learnability ratios can recover significant performance (we achieved 40%-70% of theoretical maximum improvement in our examples), while those with low learnability show minimal benefit, validating our diagnostic criteria for practical deployment decisions.

**arXiv ID:** 2510.25793
</details>

<details>
<summary><strong>Oryx: a Scalable Sequence Model for Many-Agent Coordination in Offline MARL</strong> - Claude Formanek, Omayma Mahjoub, Louay Ben Nessir, Sasha Abramowitz, Ruan de Kock, Wiem Khlifi, Daniel Rajaonarivonivelomanantsoa, Simon Du Toit, Arnol Fokam, Siddarth Singh, Ulrich Mbou Sob, Felix Chalumeau, Arnu Pretorius - [[pdf]](https://arxiv.org/pdf/2505.22151)</summary>

**Abstract:** A key challenge in offline multi-agent reinforcement learning (MARL) is achieving effective many-agent multi-step coordination in complex environments. In this work, we propose Oryx, a novel algorithm for offline cooperative MARL to directly address this challenge. Oryx adapts the recently proposed retention-based architecture Sable and combines it with a sequential form of implicit constraint Q-learning (ICQ), to develop a novel offline autoregressive policy update scheme. This allows Oryx to solve complex coordination challenges while maintaining temporal coherence over long trajectories. We evaluate Oryx across a diverse set of benchmarks from prior works -- SMAC, RWARE, and Multi-Agent MuJoCo -- covering tasks of both discrete and continuous control, varying in scale and difficulty. Oryx achieves state-of-the-art performance on more than 80% of the 65 tested datasets, outperforming prior offline MARL methods and demonstrating robust generalisation across domains with many agents and long horizons. Finally, we introduce new datasets to push the limits of many-agent coordination in offline MARL, and demonstrate Oryx's superior ability to scale effectively in such settings.

**arXiv ID:** 2505.22151
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>FinOps Agent -- A Use-Case for IT Infrastructure and Cost Optimization</strong> - Ngoc Phuoc An Vo, Manish Kesarwani, Ruchi Mahindru, Chandrasekhar Narayanaswami - [[pdf]](https://arxiv.org/pdf/2510.25914)</summary>

**Abstract:** FinOps (Finance + Operations) represents an operational framework and cultural practice which maximizes cloud business value through collaborative financial accountability across engineering, finance, and business teams. FinOps practitioners face a fundamental challenge: billing data arrives in heterogeneous formats, taxonomies, and metrics from multiple cloud providers and internal systems which eventually lead to synthesizing actionable insights, and making time-sensitive decisions. To address this challenge, we propose leveraging autonomous, goal-driven AI agents for FinOps automation. In this paper, we built a FinOps agent for a typical use-case for IT infrastructure and cost optimization. We built a system simulating a realistic end-to-end industry process starting with retrieving data from various sources to consolidating and analyzing the data to generate recommendations for optimization. We defined a set of metrics to evaluate our agent using several open-source and close-source language models and it shows that the agent was able to understand, plan, and execute tasks as well as an actual FinOps practitioner.

**arXiv ID:** 2510.25914
</details>

<details>
<summary><strong>The Oversight Game: Learning to Cooperatively Balance an AI Agent's Safety and Autonomy</strong> - William Overman, Mohsen Bayati - [[pdf]](https://arxiv.org/pdf/2510.26752)</summary>

**Abstract:** As increasingly capable agents are deployed, a central safety question is how to retain meaningful human control without modifying the underlying system. We study a minimal control interface where an agent chooses whether to act autonomously (play) or defer (ask), while a human simultaneously chooses whether to be permissive (trust) or to engage in oversight (oversee). If the agent defers, the human's choice determines the outcome, potentially leading to a corrective action or a system shutdown. We model this interaction as a two-player Markov Game. Our analysis focuses on cases where this game qualifies as a Markov Potential Game (MPG), a class of games where we can provide an alignment guarantee: under a structural assumption on the human's value function, any decision by the agent to act more autonomously that benefits itself cannot harm the human's value. We also analyze extensions to this MPG framework. Theoretically, this perspective provides conditions for a specific form of intrinsic alignment. If the reward structures of the human-agent game meet these conditions, we have a formal guarantee that the agent improving its own outcome will not harm the human's. Practically, this model motivates a transparent control layer with predictable incentives where the agent learns to defer when risky and act when safe, while its pretrained policy and the environment's reward structure remain untouched. Our gridworld simulation shows that through independent learning, the agent and human discover their optimal oversight roles. The agent learns to ask when uncertain and the human learns when to oversee, leading to an emergent collaboration that avoids safety violations introduced post-training. This demonstrates a practical method for making misaligned models safer after deployment.

**arXiv ID:** 2510.26752
</details>

<details>
<summary><strong>AAGATE: A NIST AI RMF-Aligned Governance Platform for Agentic AI</strong> - Ken Huang, Jerry Huang, Yasir Mehmood, Hammad Atta, Muhammad Zeeshan Baig, Muhammad Aziz Ul Haq - [[pdf]](https://arxiv.org/pdf/2510.25863)</summary>

**Abstract:** This paper introduces the Agentic AI Governance Assurance & Trust Engine (AAGATE), a Kubernetes-native control plane designed to address the unique security and governance challenges posed by autonomous, language-model-driven agents in production. Recognizing the limitations of traditional Application Security (AppSec) tooling for improvisational, machine-speed systems, AAGATE operationalizes the NIST AI Risk Management Framework (AI RMF). It integrates specialized security frameworks for each RMF function: the Agentic AI Threat Modeling MAESTRO framework for Map, a hybrid of OWASP's AIVSS and SEI's SSVC for Measure, and the Cloud Security Alliance's Agentic AI Red Teaming Guide for Manage. By incorporating a zero-trust service mesh, an explainable policy engine, behavioral analytics, and decentralized accountability hooks, AAGATE provides a continuous, verifiable governance solution for agentic AI, enabling safe, accountable, and scalable deployment. The framework is further extended with DIRF for digital identity rights, LPCI defenses for logic-layer injection, and QSAF monitors for cognitive degradation, ensuring governance spans systemic, adversarial, and ethical risks.

**arXiv ID:** 2510.25863
</details>

<details>
<summary><strong>RADRON: Cooperative Localization of Ionizing Radiation Sources by MAVs with Compton Cameras</strong> - Petr Stibinger, Tomas Baca, Daniela Doubravova, Jan Rusnak, Jaroslav Solc, Jan Jakubek, Petr Stepan, Martin Saska - [[pdf]](https://arxiv.org/pdf/2510.26018)</summary>

**Abstract:** We present a novel approach to localizing radioactive material by cooperating Micro Aerial Vehicles (MAVs). Our approach utilizes a state-of-the-art single-detector Compton camera as a highly sensitive, yet miniature detector of ionizing radiation. The detector's exceptionally low weight (40 g) opens up new possibilities of radiation detection by a team of cooperating agile MAVs. We propose a new fundamental concept of fusing the Compton camera measurements to estimate the position of the radiation source in real time even from extremely sparse measurements. The data readout and processing are performed directly onboard and the results are used in a dynamic feedback to drive the motion of the vehicles. The MAVs are stabilized in a tightly cooperating swarm to maximize the information gained by the Compton cameras, rapidly locate the radiation source, and even track a moving radiation source.

**arXiv ID:** 2510.26018
</details>

<details>
<summary><strong>Agent Skills Enable a New Class of Realistic and Trivially Simple Prompt Injections</strong> - David Schmotz, Sahar Abdelnabi, Maksym Andriushchenko - [[pdf]](https://arxiv.org/pdf/2510.26328)</summary>

**Abstract:** Enabling continual learning in LLMs remains a key unresolved research challenge. In a recent announcement, a frontier LLM company made a step towards this by introducing Agent Skills, a framework that equips agents with new knowledge based on instructions stored in simple markdown files. Although Agent Skills can be a very useful tool, we show that they are fundamentally insecure, since they enable trivially simple prompt injections. We demonstrate how to hide malicious instructions in long Agent Skill files and referenced scripts to exfiltrate sensitive data, such as internal files or passwords. Importantly, we show how to bypass system-level guardrails of a popular coding agent: a benign, task-specific approval with the "Don't ask again" option can carry over to closely related but harmful actions. Overall, we conclude that despite ongoing research efforts and scaling model capabilities, frontier LLMs remain vulnerable to very simple prompt injections in realistic scenarios. Our code is available at this https URL.

**arXiv ID:** 2510.26328
</details>

<details>
<summary><strong>Embodied Intelligence for Advanced Bioinspired Microrobotics: Examples and Insights</strong> - Nestor O. Perez-Arancibia - [[pdf]](https://arxiv.org/pdf/2510.26132)</summary>

**Abstract:** The term embodied intelligence (EI) conveys the notion that body morphology, material properties, interaction with the environment, and control strategies can be purposefully integrated into the process of robotic design to generate intelligent behavior; in particular, locomotion and navigation. In this paper, we discuss EI as a design principle for advanced microrobotics, with a particular focus on co-design -- the simultaneous and interdependent development of physical structure and behavioral function. To illustrate the contrast between EI-inspired systems and traditional architectures that decouple sensing, computation, and actuation, we present and discuss a collection of robots developed by the author and his team at the Autonomous Microrobotic Systems Laboratory (AMSL). These robots exhibit intelligent behavior that emerges from their structural dynamics and the physical interaction between their components and with the environment. Platforms such as the Bee++, RoBeetle, SMALLBug, SMARTI, WaterStrider, VLEIBot+, and FRISSHBot exemplify how feedback loops, decision logics, sensing mechanisms, and smart actuation strategies can be embedded into the physical properties of the robotic system itself. Along these lines, we contend that co-design is not only a method for empirical optimization under constraints, but also an enabler of EI, offering a scalable and robust alternative to classical control for robotics at the mm-to-cm-scale.

**arXiv ID:** 2510.26132
</details>

<details>
<summary><strong>Agile and Cooperative Aerial Manipulation of a Cable-Suspended Load</strong> - Sihao Sun, Xuerui Wang, Dario Sanalitro, Antonio Franchi, Marco Tognon, Javier Alonso-Mora - [[pdf]](https://arxiv.org/pdf/2501.18802)</summary>

**Abstract:** Quadrotors can carry slung loads to hard-to-reach locations at high speed. Since a single quadrotor has limited payload capacities, using a team of quadrotors to collaboratively manipulate a heavy object is a scalable and promising solution. However, existing control algorithms for multi-lifting systems only enable low-speed and low-acceleration operations due to the complex dynamic coupling between quadrotors and the load, limiting their use in time-critical missions such as search and rescue. In this work, we present a solution to significantly enhance the agility of cable-suspended multi-lifting systems. Unlike traditional cascaded solutions, we introduce a trajectory-based framework that solves the whole-body kinodynamic motion planning problem online, accounting for the dynamic coupling effects and constraints between the quadrotors and the load. The planned trajectory is provided to the quadrotors as a reference in a receding-horizon fashion and is tracked by an onboard controller that observes and compensates for the cable tension. Real-world experiments demonstrate that our framework can achieve at least eight times greater acceleration than state-of-the-art methods to follow agile trajectories. Our method can even perform complex maneuvers such as flying through narrow passages at high speed. Additionally, it exhibits high robustness against load uncertainties and does not require adding any sensors to the load, demonstrating strong practicality.

**arXiv ID:** 2501.18802
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (20 papers)</h2></summary>

<details>
<summary><strong>An Agentic Framework for Rapid Deployment of Edge AI Solutions in Industry 5.0</strong> - Jorge Martinez-Gil, Mario Pichler, Nefeli Bountouni, Sotiris Koussouris, Marielena Márquez Barreiro, Sergio Gusmeroli - [[pdf]](https://arxiv.org/pdf/2510.25813)</summary>

**Abstract:** We present a novel framework for Industry 5.0 that simplifies the deployment of AI models on edge devices in various industrial settings. The design reduces latency and avoids external data transfer by enabling local inference and real-time processing. Our implementation is agent-based, which means that individual agents, whether human, algorithmic, or collaborative, are responsible for well-defined tasks, enabling flexibility and simplifying integration. Moreover, our framework supports modular integration and maintains low resource requirements. Preliminary evaluations concerning the food industry in real scenarios indicate improved deployment time and system adaptability performance. The source code is publicly available at this https URL.

**arXiv ID:** 2510.25813
</details>

<details>
<summary><strong>From Queries to Insights: Agentic LLM Pipelines for Spatio-Temporal Text-to-SQL</strong> - Manu Redd, Tao Zhe, Dongjie Wang - [[pdf]](https://arxiv.org/pdf/2510.25997)</summary>

**Abstract:** Natural-language-to-SQL (NL-to-SQL) systems hold promise for democratizing access to structured data, allowing users to query databases without learning SQL. Yet existing systems struggle with realistic spatio-temporal queries, where success requires aligning vague user phrasing with schema-specific categories, handling temporal reasoning, and choosing appropriate outputs. We present an agentic pipeline that extends a naive text-to-SQL baseline (llama-3-sqlcoder-8b) with orchestration by a Mistral-based ReAct agent. The agent can plan, decompose, and adapt queries through schema inspection, SQL generation, execution, and visualization tools. We evaluate on 35 natural-language queries over the NYC and Tokyo check-in dataset, covering spatial, temporal, and multi-dataset reasoning. The agent achieves substantially higher accuracy than the naive baseline 91.4% vs. 28.6% and enhances usability through maps, plots, and structured natural-language summaries. Crucially, our design enables more natural human-database interaction, supporting users who lack SQL expertise, detailed schema knowledge, or prompting skill. We conclude that agentic orchestration, rather than stronger SQL generators alone, is a promising foundation for interactive geospatial assistants.

**arXiv ID:** 2510.25997
</details>

<details>
<summary><strong>One Model to Critique Them All: Rewarding Agentic Tool-Use via Efficient Reasoning</strong> - Renhao Li, Jianhong Tu, Yang Su, Hamid Alinejad-Rokny, Derek F. Wong, Junyang Lin, Min Yang - [[pdf]](https://arxiv.org/pdf/2510.26167)</summary>

**Abstract:** Reward models (RMs) play a critical role in aligning large language models (LLMs) with human preferences. Yet in the domain of tool learning, the lack of RMs specifically designed for function-calling tasks has limited progress toward more capable agentic AI. We introduce ToolRM, a family of lightweight generative RMs tailored for general tool-use scenarios. To build these models, we propose a novel pipeline that constructs pairwise preference data using rule-based scoring and multidimensional sampling. This yields ToolPref-Pairwise-30K, a diverse, balanced, and challenging dataset of critique tasks that supports reinforcement learning with verifiable feedback. To evaluate tool-use RMs, we also introduce TRBench$_{BFCL}$, a benchmark built on the agentic evaluation suite BFCL. Trained on our constructed data, models from the Qwen3-4B/8B series achieve up to 14.28% higher accuracy, substantially outperforming frontier models such as Claude 4 and OpenAI o3 in pairwise reward judgments. Beyond training objectives, ToolRM generalizes to broader critique tasks, including Best-of-N sampling and self-correction. Experiments on ACEBench highlight its effectiveness and efficiency, enabling inference-time scaling and reducing output token usage by over 66%. We release data and model checkpoints to facilitate future research.

**arXiv ID:** 2510.26167
</details>

<details>
<summary><strong>The Era of Agentic Organization: Learning to Organize with Language Models</strong> - Zewen Chi, Li Dong, Qingxiu Dong, Yaru Hao, Xun Wu, Shaohan Huang, Furu Wei - [[pdf]](https://arxiv.org/pdf/2510.26658)</summary>

**Abstract:** We envision a new era of AI, termed agentic organization, where agents solve complex problems by working collaboratively and concurrently, enabling outcomes beyond individual intelligence. To realize this vision, we introduce asynchronous thinking (AsyncThink) as a new paradigm of reasoning with large language models, which organizes the internal thinking process into concurrently executable structures. Specifically, we propose a thinking protocol where an organizer dynamically assigns sub-queries to workers, merges intermediate knowledge, and produces coherent solutions. More importantly, the thinking structure in this protocol can be further optimized through reinforcement learning. Experiments demonstrate that AsyncThink achieves 28% lower inference latency compared to parallel thinking while improving accuracy on mathematical reasoning. Moreover, AsyncThink generalizes its learned asynchronous thinking capabilities, effectively tackling unseen tasks without additional training.

**arXiv ID:** 2510.26658
</details>

<details>
<summary><strong>Non-myopic Matching and Rebalancing in Large-Scale On-Demand Ride-Pooling Systems Using Simulation-Informed Reinforcement Learning</strong> - Farnoosh Namdarpour, Joseph Y. J. Chow - [[pdf]](https://arxiv.org/pdf/2510.25796)</summary>

**Abstract:** Ride-pooling, also known as ride-sharing, shared ride-hailing, or microtransit, is a service wherein passengers share rides. This service can reduce costs for both passengers and operators and reduce congestion and environmental impacts. A key limitation, however, is its myopic decision-making, which overlooks long-term effects of dispatch decisions. To address this, we propose a simulation-informed reinforcement learning (RL) approach. While RL has been widely studied in the context of ride-hailing systems, its application in ride-pooling systems has been less explored. In this study, we extend the learning and planning framework of Xu et al. (2018) from ride-hailing to ride-pooling by embedding a ride-pooling simulation within the learning mechanism to enable non-myopic decision-making. In addition, we propose a complementary policy for rebalancing idle vehicles. By employing n-step temporal difference learning on simulated experiences, we derive spatiotemporal state values and subsequently evaluate the effectiveness of the non-myopic policy using NYC taxi request data. Results demonstrate that the non-myopic policy for matching can increase the service rate by up to 8.4% versus a myopic policy while reducing both in-vehicle and wait times for passengers. Furthermore, the proposed non-myopic policy can decrease fleet size by over 25% compared to a myopic policy, while maintaining the same level of performance, thereby offering significant cost savings for operators. Incorporating rebalancing operations into the proposed framework cuts wait time by up to 27.3%, in-vehicle time by 12.5%, and raises service rate by 15.1% compared to using the framework for matching decisions alone at the cost of increased vehicle minutes traveled per passenger.

**arXiv ID:** 2510.25796
</details>

<details>
<summary><strong>Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning</strong> - Yihe Deng, I-Hung Hsu, Jun Yan, Zifeng Wang, Rujun Han, Gufeng Zhang, Yanfei Chen, Wei Wang, Tomas Pfister, Chen-Yu Lee - [[pdf]](https://arxiv.org/pdf/2510.25992)</summary>

**Abstract:** Large Language Models (LLMs) often struggle with problems that require multi-step reasoning. For small-scale open-source models, Reinforcement Learning with Verifiable Rewards (RLVR) fails when correct solutions are rarely sampled even after many attempts, while Supervised Fine-Tuning (SFT) tends to overfit long demonstrations through rigid token-by-token imitation. To address this gap, we propose Supervised Reinforcement Learning (SRL), a framework that reformulates problem solving as generating a sequence of logical "actions". SRL trains the model to generate an internal reasoning monologue before committing to each action. It provides smoother rewards based on the similarity between the model's actions and expert actions extracted from the SFT dataset in a step-wise manner. This supervision offers richer learning signals even when all rollouts are incorrect, while encouraging flexible reasoning guided by expert demonstrations. As a result, SRL enables small models to learn challenging problems previously unlearnable by SFT or RLVR. Moreover, initializing training with SRL before refining with RLVR yields the strongest overall performance. Beyond reasoning benchmarks, SRL generalizes effectively to agentic software engineering tasks, establishing it as a robust and versatile training framework for reasoning-oriented LLMs.

**arXiv ID:** 2510.25992
</details>

<details>
<summary><strong>Reinforcement Learning for Pollution Detection in a Randomized, Sparse and Nonstationary Environment with an Autonomous Underwater Vehicle</strong> - Sebastian Zieglmeier, Niklas Erdmann, Narada D. Warakagoda - [[pdf]](https://arxiv.org/pdf/2510.26347)</summary>

**Abstract:** Reinforcement learning (RL) algorithms are designed to optimize problem-solving by learning actions that maximize rewards, a task that becomes particularly challenging in random and nonstationary environments. Even advanced RL algorithms are often limited in their ability to solve problems in these conditions. In applications such as searching for underwater pollution clouds with autonomous underwater vehicles (AUVs), RL algorithms must navigate reward-sparse environments, where actions frequently result in a zero reward. This paper aims to address these challenges by revisiting and modifying classical RL approaches to efficiently operate in sparse, randomized, and nonstationary environments. We systematically study a large number of modifications, including hierarchical algorithm changes, multigoal learning, and the integration of a location memory as an external output filter to prevent state revisits. Our results demonstrate that a modified Monte Carlo-based approach significantly outperforms traditional Q-learning and two exhaustive search patterns, illustrating its potential in adapting RL to complex environments. These findings suggest that reinforcement learning approaches can be effectively adapted for use in random, nonstationary, and reward-sparse environments.

**arXiv ID:** 2510.26347
</details>

<details>
<summary><strong>InfoFlow: Reinforcing Search Agent Via Reward Density Optimization</strong> - Kun Luo, Hongjin Qian, Zheng Liu, Ziyi Xia, Shitao Xiao, Siqi Bao, Jun Zhao, Kang Liu - [[pdf]](https://arxiv.org/pdf/2510.26575)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) is a promising approach for enhancing agentic deep search. However, its application is often hindered by low \textbf{Reward Density} in deep search scenarios, where agents expend significant exploratory costs for infrequent and often null final rewards. In this paper, we formalize this challenge as the \textbf{Reward Density Optimization} problem, which aims to improve the reward obtained per unit of exploration cost. This paper introduce \textbf{InfoFlow}, a systematic framework that tackles this problem from three aspects. 1) \textbf{Subproblem decomposition}: breaking down long-range tasks to assign process rewards, thereby providing denser learning signals. 2) \textbf{Failure-guided hints}: injecting corrective guidance into stalled trajectories to increase the probability of successful outcomes. 3) \textbf{Dual-agent refinement}: employing a dual-agent architecture to offload the cognitive burden of deep exploration. A refiner agent synthesizes the search history, which effectively compresses the researcher's perceived trajectory, thereby reducing exploration cost and increasing the overall reward density. We evaluate InfoFlow on multiple agentic search benchmarks, where it significantly outperforms strong baselines, enabling lightweight LLMs to achieve performance comparable to advanced proprietary LLMs.

**arXiv ID:** 2510.26575
</details>

<details>
<summary><strong>Hybrid DQN-TD3 Reinforcement Learning for Autonomous Navigation in Dynamic Environments</strong> - Xiaoyi He, Danggui Chen, Zhenshuo Zhang, Zimeng Bai - [[pdf]](https://arxiv.org/pdf/2510.26646)</summary>

**Abstract:** This paper presents a hierarchical path-planning and control framework that combines a high-level Deep Q-Network (DQN) for discrete sub-goal selection with a low-level Twin Delayed Deep Deterministic Policy Gradient (TD3) controller for continuous actuation. The high-level module selects behaviors and sub-goals; the low-level module executes smooth velocity commands. We design a practical reward shaping scheme (direction, distance, obstacle avoidance, action smoothness, collision penalty, time penalty, and progress), together with a LiDAR-based safety gate that prevents unsafe motions. The system is implemented in ROS + Gazebo (TurtleBot3) and evaluated with PathBench metrics, including success rate, collision rate, path efficiency, and re-planning efficiency, in dynamic and partially observable environments. Experiments show improved success rate and sample efficiency over single-algorithm baselines (DQN or TD3 alone) and rule-based planners, with better generalization to unseen obstacle configurations and reduced abrupt control changes. Code and evaluation scripts are available at the project repository.

**arXiv ID:** 2510.26646
</details>

<details>
<summary><strong>Human-Like Goalkeeping in a Realistic Football Simulation: a Sample-Efficient Reinforcement Learning Approach</strong> - Alessandro Sestini, Joakim Bergdahl, Jean-Philippe Barrette-LaPierre, Florian Fuchs, Brady Chen, Michael Jones, Linus Gisslén - [[pdf]](https://arxiv.org/pdf/2510.23216)</summary>

**Abstract:** While several high profile video games have served as testbeds for Deep Reinforcement Learning (DRL), this technique has rarely been employed by the game industry for crafting authentic AI behaviors. Previous research focuses on training super-human agents with large models, which is impractical for game studios with limited resources aiming for human-like agents. This paper proposes a sample-efficient DRL method tailored for training and fine-tuning agents in industrial settings such as the video game industry. Our method improves sample efficiency of value-based DRL by leveraging pre-collected data and increasing network plasticity. We evaluate our method training a goalkeeper agent in EA SPORTS FC 25, one of the best-selling football simulations today. Our agent outperforms the game's built-in AI by 10% in ball saving rate. Ablation studies show that our method trains agents 50% faster compared to standard DRL methods. Finally, qualitative evaluation from domain experts indicates that our approach creates more human-like gameplay compared to hand-crafted agents. As a testament to the impact of the approach, the method has been adopted for use in the most recent release of the series.

**arXiv ID:** 2510.23216
</details>

<details>
<summary><strong>Chaos-based reinforcement learning with TD3</strong> - Toshitaka Matsuki, Yusuke Sakemi, Kazuyuki Aihara - [[pdf]](https://arxiv.org/pdf/2405.09086)</summary>

**Abstract:** Chaos-based reinforcement learning (CBRL) is a method in which the agent's internal chaotic dynamics drives exploration. However, the learning algorithms in CBRL have not been thoroughly developed in previous studies, nor have they incorporated recent advances in reinforcement learning. This study introduced Twin Delayed Deep Deterministic Policy Gradients (TD3), which is one of the state-of-the-art deep reinforcement learning algorithms that can treat deterministic and continuous action spaces, to CBRL. The validation results provide several insights. First, TD3 works as a learning algorithm for CBRL in a simple goal-reaching task. Second, CBRL agents with TD3 can autonomously suppress their exploratory behavior as learning progresses and resume exploration when the environment changes. Finally, examining the effect of the agent's chaoticity on learning shows that there exists a suitable range of chaos strength in the agent's model to flexibly switch between exploration and exploitation and adapt to environmental changes.

**arXiv ID:** 2405.09086
</details>

<details>
<summary><strong>Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems</strong> - Christian Walder, Deep Karkhanis - [[pdf]](https://arxiv.org/pdf/2505.15201)</summary>

**Abstract:** Reinforcement Learning (RL) algorithms sample multiple n>1 solution attempts for each problem and reward them independently. This optimizes for pass@1 performance and prioritizes the strength of isolated samples at the expense of the diversity and collective utility of sets of samples. This under-utilizes the sampling capacity, limiting exploration and eventual improvement on harder examples. As a fix, we propose Pass-at-k Policy Optimization (PKPO), a transformation on the final rewards which leads to direct optimization of pass@k performance, thus optimizing for sets of samples that maximize reward when considered jointly. Our contribution is to derive novel low variance unbiased estimators for pass@k and its gradient, in both the binary and continuous reward settings. We show optimization with our estimators reduces to standard RL with rewards that have been jointly transformed by a stable and efficient transformation function.
While previous efforts are restricted to k=n, ours is the first to enable robust optimization of pass@k for any arbitrary k <= n. Moreover, instead of trading off pass@1 performance for pass@k gains, our method allows annealing k during training, optimizing both metrics and often achieving strong pass@1 numbers alongside significant pass@k gains.
We validate our reward transformations on toy experiments, which reveal the variance reducing properties of our formulations. We also include real-world examples using the open-source LLM, GEMMA-2. We find that our transformation effectively optimizes for the target k. Furthermore, higher k values enable solving more and harder problems, while annealing k boosts both the pass@1 and pass@k . Crucially, for challenging task sets where conventional pass@1 optimization stalls, our pass@k approach unblocks learning, likely due to better exploration by prioritizing joint utility over the utility of individual samples.

**arXiv ID:** 2505.15201
</details>

<details>
<summary><strong>SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding</strong> - Yiqiao Jin, Rachneet Kaur, Zhen Zeng, Sumitra Ganesh, Srijan Kumar - [[pdf]](https://arxiv.org/pdf/2510.26615)</summary>

**Abstract:** Multi-page visual documents such as manuals, brochures, presentations, and posters convey key information through layout, colors, icons, and cross-slide references. While large language models (LLMs) offer opportunities in document understanding, current systems struggle with complex, multi-page visual documents, particularly in fine-grained reasoning over elements and pages. We introduce SlideAgent, a versatile agentic framework for understanding multi-modal, multi-page, and multi-layout documents, especially slide decks. SlideAgent employs specialized agents and decomposes reasoning into three specialized levels-global, page, and element-to construct a structured, query-agnostic representation that captures both overarching themes and detailed visual or textual cues. During inference, SlideAgent selectively activates specialized agents for multi-level reasoning and integrates their outputs into coherent, context-aware answers. Extensive experiments show that SlideAgent achieves significant improvement over both proprietary (+7.9 overall) and open-source models (+9.8 overall).

**arXiv ID:** 2510.26615
</details>

<details>
<summary><strong>A Game-Theoretic Spatio-Temporal Reinforcement Learning Framework for Collaborative Public Resource Allocation</strong> - Songxin Lei, Qiongyan Wang, Yanchen Zhu, Hanyu Yao, Sijie Ruan, Weilin Ruan, Yuyu Luo, Huaming Wu, Yuxuan Liang - [[pdf]](https://arxiv.org/pdf/2510.26184)</summary>

**Abstract:** Public resource allocation involves the efficient distribution of resources, including urban infrastructure, energy, and transportation, to effectively meet societal demands. However, existing methods focus on optimizing the movement of individual resources independently, without considering their capacity constraints. To address this limitation, we propose a novel and more practical problem: Collaborative Public Resource Allocation (CPRA), which explicitly incorporates capacity constraints and spatio-temporal dynamics in real-world scenarios. We propose a new framework called Game-Theoretic Spatio-Temporal Reinforcement Learning (GSTRL) for solving CPRA. Our contributions are twofold: 1) We formulate the CPRA problem as a potential game and demonstrate that there is no gap between the potential function and the optimal target, laying a solid theoretical foundation for approximating the Nash equilibrium of this NP-hard problem; and 2) Our designed GSTRL framework effectively captures the spatio-temporal dynamics of the overall system. We evaluate GSTRL on two real-world datasets, where experiments show its superior performance. Our source codes are available in the supplementary materials.

**arXiv ID:** 2510.26184
</details>

<details>
<summary><strong>ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems</strong> - Qiaoling Chen, Zijun Liu, Peng Sun, Shenggui Li, Guoteng Wang, Ziming Liu, Yonggang Wen, Siyuan Feng, Tianwei Zhang - [[pdf]](https://arxiv.org/pdf/2510.26475)</summary>

**Abstract:** Adapting large language models (LLMs) via reinforcement learning (RL) is often bottlenecked by the generation stage, which can consume over 75\% of the training time. Speculative decoding (SD) accelerates autoregressive generation in serving systems, but its behavior under RL training remains largely unexplored. We identify three critical gaps that hinder the naive integration of SD into RL systems: diminishing speedups at large batch sizes, drafter staleness under continual actor updates, and drafter-induced policy degradation.
To address these gaps, we present ReSpec, a system that adapts SD to RL through three complementary mechanisms: dynamically tuning SD configurations, evolving the drafter via knowledge distillation, and weighting updates by rollout rewards. On Qwen models (3B--14B), ReSpec achieves up to 4.5x speedup while preserving reward convergence and training stability, providing a practical solution for efficient RL-based LLM adaptation.

**arXiv ID:** 2510.26475
</details>

<details>
<summary><strong>Accelerating Real-World Overtaking in F1TENTH Racing Employing Reinforcement Learning Methods</strong> - Emily Steiner, Daniel van der Spuy, Futian Zhou, Afereti Pama, Minas Liarokapis, Henry Williams - [[pdf]](https://arxiv.org/pdf/2510.26040)</summary>

**Abstract:** While autonomous racing performance in Time-Trial scenarios has seen significant progress and development, autonomous wheel-to-wheel racing and overtaking are still severely limited. These limitations are particularly apparent in real-life driving scenarios where state-of-the-art algorithms struggle to safely or reliably complete overtaking manoeuvres. This is important, as reliable navigation around other vehicles is vital for safe autonomous wheel-to-wheel racing. The F1Tenth Competition provides a useful opportunity for developing wheel-to-wheel racing algorithms on a standardised physical platform. The competition format makes it possible to evaluate overtaking and wheel-to-wheel racing algorithms against the state-of-the-art. This research presents a novel racing and overtaking agent capable of learning to reliably navigate a track and overtake opponents in both simulation and reality. The agent was deployed on an F1Tenth vehicle and competed against opponents running varying competitive algorithms in the real world. The results demonstrate that the agent's training against opponents enables deliberate overtaking behaviours with an overtaking rate of 87% compared 56% for an agent trained just to race.

**arXiv ID:** 2510.26040
</details>

<details>
<summary><strong>Smart Exploration in Reinforcement Learning using Bounded Uncertainty Models</strong> - J.S. van Hulst, W.P.M.H. Heemels, D.J. Antunes - [[pdf]](https://arxiv.org/pdf/2504.05978)</summary>

**Abstract:** Reinforcement learning (RL) is a powerful framework for decision-making in uncertain environments, but it often requires large amounts of data to learn an optimal policy. We address this challenge by incorporating prior model knowledge to guide exploration and accelerate the learning process. Specifically, we assume access to a model set that contains the true transition kernel and reward function. We optimize over this model set to obtain upper and lower bounds on the Q-function, which are then used to guide the exploration of the agent. We provide theoretical guarantees on the convergence of the Q-function to the optimal Q-function under the proposed class of exploring policies. Furthermore, we also introduce a data-driven regularized version of the model set optimization problem that ensures the convergence of the class of exploring policies to the optimal policy. Lastly, we show that when the model set has a specific structure, namely the bounded-parameter MDP (BMDP) framework, the regularized model set optimization problem becomes convex and simple to implement. In this setting, we also prove finite-time convergence to the optimal policy under mild assumptions. We demonstrate the effectiveness of the proposed exploration strategy, which we call BUMEX (Bounded Uncertainty Model-based Exploration), in a simulation study. The results indicate that the proposed method can significantly accelerate learning in benchmark examples. A toolbox is available at this https URL.

**arXiv ID:** 2504.05978
</details>

<details>
<summary><strong>Morphology-Aware Graph Reinforcement Learning for Tensegrity Robot Locomotion</strong> - Chi Zhang, Mingrui Li, Wenzhe Tong, Xiaonan Huang - [[pdf]](https://arxiv.org/pdf/2510.26067)</summary>

**Abstract:** Tensegrity robots combine rigid rods and elastic cables, offering high resilience and deployability but posing major challenges for locomotion control due to their underactuated and highly coupled dynamics. This paper introduces a morphology-aware reinforcement learning framework that integrates a graph neural network (GNN) into the Soft Actor-Critic (SAC) algorithm. By representing the robot's physical topology as a graph, the proposed GNN-based policy captures coupling among components, enabling faster and more stable learning than conventional multilayer perceptron (MLP) policies. The method is validated on a physical 3-bar tensegrity robot across three locomotion primitives, including straight-line tracking and bidirectional turning. It shows superior sample efficiency, robustness to noise and stiffness variations, and improved trajectory accuracy. Notably, the learned policies transfer directly from simulation to hardware without fine-tuning, achieving stable real-world locomotion. These results demonstrate the advantages of incorporating structural priors into reinforcement learning for tensegrity robot control.

**arXiv ID:** 2510.26067
</details>

<details>
<summary><strong>Cooperative Task Spaces for Multi-Arm Manipulation Control based on Similarity Transformations</strong> - Tobias Löw, Cem Bilaloglu, Sylvain Calinon - [[pdf]](https://arxiv.org/pdf/2510.26362)</summary>

**Abstract:** Many tasks in human environments require collaborative behavior between multiple kinematic chains, either to provide additional support for carrying big and bulky objects or to enable the dexterity that is required for in-hand manipulation. Since these complex systems often have a very high number of degrees of freedom coordinating their movements is notoriously difficult to model. In this article, we present the derivation of the theoretical foundations for cooperative task spaces of multi-arm robotic systems based on geometric primitives defined using conformal geometric algebra. Based on the similarity transformations of these cooperative geometric primitives, we derive an abstraction of complex robotic systems that enables representing these systems in a way that directly corresponds to single-arm systems. By deriving the associated analytic and geometric Jacobian matrices, we then show the straightforward integration of our approach into classical control techniques rooted in operational space control. We demonstrate this using bimanual manipulators, humanoids and multi-fingered hands in optimal control experiments for reaching desired geometric primitives and in teleoperation experiments using differential kinematics control. We then discuss how the geometric primitives naturally embed nullspace structures into the controllers that can be exploited for introducing secondary control objectives. This work, represents the theoretical foundations of this cooperative manipulation control framework, and thus the experiments are presented in an abstract way, while giving pointers towards potential future applications.

**arXiv ID:** 2510.26362
</details>

<details>
<summary><strong>Towards Reinforcement Learning Based Log Loading Automation</strong> - Ilya Kurinov, Miroslav Ivanov, Grzegorz Orzechowski, Aki Mikkola - [[pdf]](https://arxiv.org/pdf/2510.26363)</summary>

**Abstract:** Forestry forwarders play a central role in mechanized timber harvesting by picking up and moving logs from the felling site to a processing area or a secondary transport vehicle. Forwarder operation is challenging and physically and mentally exhausting for the operator who must control the machine in remote areas for prolonged periods of time. Therefore, even partial automation of the process may reduce stress on the operator. This study focuses on continuing previous research efforts in application of reinforcement learning agents in automating log handling process, extending the task from grasping which was studied in previous research to full log loading operation. The resulting agent will be capable to automate a full loading procedure from locating and grappling to transporting and delivering the log to a forestry forwarder bed. To train the agent, a trailer type forestry forwarder simulation model in NVIDIA's Isaac Gym and a virtual environment for a typical log loading scenario were developed. With reinforcement learning agents and a curriculum learning approach, the trained agent may be a stepping stone towards application of reinforcement learning agents in automation of the forestry forwarder. The agent learnt grasping a log in a random position from grapple's random position and transport it to the bed with 94% success rate of the best performing agent.

**arXiv ID:** 2510.26363
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
