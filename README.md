# Agent arXiv Daily

**Last Updated:** 2025-09-09 02:40:54

**Total Papers:** 93

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (6 papers)</h2></summary>

<details>
<summary><strong>Towards Log Analysis with AI Agents: Cowrie Case Study</strong> - Enis Karaarslan, Esin Güler, Efe Emir Yüce, Cagatay Coban - [[pdf]](https://arxiv.org/pdf/2509.05306)</summary>

**Abstract:** The scarcity of real-world attack data significantly hinders progress in cybersecurity research and education. Although honeypots like Cowrie effectively collect live threat intelligence, they generate overwhelming volumes of unstructured and heterogeneous logs, rendering manual analysis impractical. As a first step in our project on secure and efficient AI automation, this study explores the use of AI agents for automated log analysis. We present a lightweight and automated approach to process Cowrie honeypot logs. Our approach leverages AI agents to intelligently parse, summarize, and extract insights from raw data, while also considering the security implications of deploying such an autonomous system. Preliminary results demonstrate the pipeline's effectiveness in reducing manual effort and identifying attack patterns, paving the way for more advanced autonomous cybersecurity analysis in future work.

**arXiv ID:** 2509.05306
</details>

<details>
<summary><strong>Learning Tool-Aware Adaptive Compliant Control for Autonomous Regolith Excavation</strong> - Andrej Orsula, Matthieu Geist, Miguel Olivares-Mendez, Carol Martinez - [[pdf]](https://arxiv.org/pdf/2509.05475)</summary>

**Abstract:** Autonomous regolith excavation is a cornerstone of in-situ resource utilization for a sustained human presence beyond Earth. However, this task is fundamentally hindered by the complex interaction dynamics of granular media and the operational need for robots to use diverse tools. To address these challenges, this work introduces a framework where a model-based reinforcement learning agent learns within a parallelized simulation. This environment leverages high-fidelity particle physics and procedural generation to create a vast distribution of both lunar terrains and excavation tool geometries. To master this diversity, the agent learns an adaptive interaction strategy by dynamically modulating its own stiffness and damping at each control step through operational space control. Our experiments demonstrate that training with a procedural distribution of tools is critical for generalization and enables the development of sophisticated tool-aware behavior. Furthermore, we show that augmenting the agent with visual feedback significantly improves task success. These results represent a validated methodology for developing the robust and versatile autonomous systems required for the foundational tasks of future space missions.

**arXiv ID:** 2509.05475
</details>

<details>
<summary><strong>Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System</strong> - Yu Liu, Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She - [[pdf]](https://arxiv.org/pdf/2509.05755)</summary>

**Abstract:** LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.

**arXiv ID:** 2509.05755
</details>

<details>
<summary><strong>AxelSMOTE: An Agent-Based Oversampling Algorithm for Imbalanced Classification</strong> - Sukumar Kishanthan, Asela Hevapathige - [[pdf]](https://arxiv.org/pdf/2509.06875)</summary>

**Abstract:** Class imbalance in machine learning poses a significant challenge, as skewed datasets often hinder performance on minority classes. Traditional oversampling techniques, which are commonly used to alleviate class imbalance, have several drawbacks: they treat features independently, lack similarity-based controls, limit sample diversity, and fail to manage synthetic variety effectively. To overcome these issues, we introduce AxelSMOTE, an innovative agent-based approach that views data instances as autonomous agents engaging in complex interactions. Based on Axelrod's cultural dissemination model, AxelSMOTE implements four key innovations: (1) trait-based feature grouping to preserve correlations; (2) a similarity-based probabilistic exchange mechanism for meaningful interactions; (3) Beta distribution blending for realistic interpolation; and (4) controlled diversity injection to avoid overfitting. Experiments on eight imbalanced datasets demonstrate that AxelSMOTE outperforms state-of-the-art sampling methods while maintaining computational efficiency.

**arXiv ID:** 2509.06875
</details>

<details>
<summary><strong>AARK: An Open Toolkit for Autonomous Racing Research</strong> - James Bockman, Matthew Howe, Adrian Orenstein, Feras Dayoub - [[pdf]](https://arxiv.org/pdf/2410.00358)</summary>

**Abstract:** Autonomous racing demands safe control of vehicles at their physical limits for extended periods of time, providing insights into advanced vehicle safety systems which increasingly rely on intervention provided by vehicle autonomy. Participation in this field carries with it a high barrier to entry. Physical platforms and their associated sensor suites require large capital outlays before any demonstrable progress can be made. Simulators allow researches to develop soft autonomous systems without purchasing a platform. However, currently available simulators lack visual and dynamic fidelity, can still be expensive to buy, lack customisation, and are difficult to use. AARK provides three packages, ACI, ACDG, and ACMPC. These packages enable research into autonomous control systems in the demanding environment of racing to bring more people into the field and improve reproducibility: ACI provides researchers with a computer vision-friendly interface to Assetto Corsa for convenient comparison and evaluation of autonomous control solutions; ACDG enables generation of depth, normal and semantic segmentation data for training computer vision models to use in perception systems; and ACMPC gives newcomers to the field a modular full-stack autonomous control solution, capable of controlling vehicles to build from. AARK aims to unify and democratise research into a field critical to providing safer roads and trusted autonomous systems.

**arXiv ID:** 2410.00358
</details>

<details>
<summary><strong>Talking to an AI Mirror: Designing Self-Clone Chatbots for Enhanced Engagement in Digital Mental Health Support</strong> - Mehrnoosh Sadat Shirvani, Jackie Liu, Thomas Chao, Suky Martinez, Laura Brandt, Ig-Jae Kim, Dongwook Yoon - [[pdf]](https://arxiv.org/pdf/2509.06393)</summary>

**Abstract:** Mental health conversational agents have the potential to deliver valuable therapeutic impact, but low user engagement remains a critical barrier hindering their efficacy. Existing therapeutic approaches have leveraged clients' internal dialogues (e.g., journaling, talking to an empty chair) to enhance engagement through accountable, self-sourced support. Inspired by these, we designed novel AI-driven self-clone chatbots that replicate users' support strategies and conversational patterns to improve therapeutic engagement through externalized meaningful self-conversation. Validated through a semi-controlled experiment (N=180), significantly higher emotional and cognitive engagement was demonstrated with self-clone chatbots than a chatbot with a generic counselor persona. Our findings highlight self-clone believability as a mediator and emphasize the balance required in maintaining convincing self-representation while creating positive interactions. This study contributes to AI-based mental health interventions by introducing and evaluating self-clones as a promising approach to increasing user engagement, while exploring implications for their application in mental health care.

**arXiv ID:** 2509.06393
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (14 papers)</h2></summary>

<details>
<summary><strong>Evaluating Multi-Turn Bargain Skills in LLM-Based Seller Agent</strong> - Issue Yishu Wang, Kakam Chong, Xiaofeng Wang, Xu Yan, DeXin Kong, Chen Ju, Ming Chen, Shuai Xiao, Shuguang Han, jufeng chen - [[pdf]](https://arxiv.org/pdf/2509.06341)</summary>

**Abstract:** In online second-hand marketplaces, multi-turn bargaining is a crucial part of seller-buyer interactions. Large Language Models (LLMs) can act as seller agents, negotiating with buyers on behalf of sellers under given business constraints. A critical ability for such agents is to track and accurately interpret cumulative buyer intents across long negotiations, which directly impacts bargaining effectiveness. We introduce a multi-turn evaluation framework for measuring the bargaining ability of seller agents in e-commerce dialogues. The framework tests whether an agent can extract and track buyer intents. Our contributions are: (1) a large-scale e-commerce bargaining benchmark spanning 622 categories, 9,892 products, and 3,014 tasks; (2) a turn-level evaluation framework grounded in Theory of Mind (ToM) with annotated buyer intents, moving beyond outcome-only metrics; and (3) an automated pipeline that extracts reliable intent from massive dialogue data.

**arXiv ID:** 2509.06341
</details>

<details>
<summary><strong>MAS-Bench: A Unified Benchmark for Shortcut-Augmented Hybrid Mobile GUI Agents</strong> - Pengxiang Zhao, Guangyi Liu, Yaozhen Liang, Weiqing He, Zhengxi Lu, Yuehao Huang, Yaxuan Guo, Kexin Zhang, Hao Wang, Liang Liu, Yong Liu - [[pdf]](https://arxiv.org/pdf/2509.06477)</summary>

**Abstract:** To enhance the efficiency of GUI agents on various platforms like smartphones and computers, a hybrid paradigm that combines flexible GUI operations with efficient shortcuts (e.g., API, deep links) is emerging as a promising direction. However, a framework for systematically benchmarking these hybrid agents is still underexplored. To take the first step in bridging this gap, we introduce MAS-Bench, a benchmark that pioneers the evaluation of GUI-shortcut hybrid agents with a specific focus on the mobile domain. Beyond merely using predefined shortcuts, MAS-Bench assesses an agent's capability to autonomously generate shortcuts by discovering and creating reusable, low-cost workflows. It features 139 complex tasks across 11 real-world applications, a knowledge base of 88 predefined shortcuts (APIs, deep-links, RPA scripts), and 7 evaluation metrics. The tasks are designed to be solvable via GUI-only operations, but can be significantly accelerated by intelligently embedding shortcuts. Experiments show that hybrid agents achieve significantly higher success rates and efficiency than their GUI-only counterparts. This result also demonstrates the effectiveness of our method for evaluating an agent's shortcut generation capabilities. MAS-Bench fills a critical evaluation gap, providing a foundational platform for future advancements in creating more efficient and robust intelligent agents.

**arXiv ID:** 2509.06477
</details>

<details>
<summary><strong>VehicleWorld: A Highly Integrated Multi-Device Environment for Intelligent Vehicle Interaction</strong> - Jie Yang, Jiajun Chen, Zhangyue Yin, Shuo Chen, Yuxin Wang, Yiran Guo, Yuan Li, Yining Zheng, Xuanjing Huang, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2509.06736)</summary>

**Abstract:** Intelligent vehicle cockpits present unique challenges for API Agents, requiring coordination across tightly-coupled subsystems that exceed typical task environments' complexity. Traditional Function Calling (FC) approaches operate statelessly, requiring multiple exploratory calls to build environmental awareness before execution, leading to inefficiency and limited error recovery. We introduce VehicleWorld, the first comprehensive environment for the automotive domain, featuring 30 modules, 250 APIs, and 680 properties with fully executable implementations that provide real-time state information during agent execution. This environment enables precise evaluation of vehicle agent behaviors across diverse, challenging scenarios. Through systematic analysis, we discovered that direct state prediction outperforms function calling for environmental control. Building on this insight, we propose State-based Function Call (SFC), a novel approach that maintains explicit system state awareness and implements direct state transitions to achieve target conditions. Experimental results demonstrate that SFC significantly outperforms traditional FC approaches, achieving superior execution accuracy and reduced latency. We have made all implementation code publicly available on Github this https URL.

**arXiv ID:** 2509.06736
</details>

<details>
<summary><strong>Evaluation of Large Language Models for Anomaly Detection in Autonomous Vehicles</strong> - Petros Loukas, David Bassir, Savvas Chatzichristofis, Angelos Amanatiadis - [[pdf]](https://arxiv.org/pdf/2509.05315)</summary>

**Abstract:** The rapid evolution of large language models (LLMs) has pushed their boundaries to many applications in various domains. Recently, the research community has started to evaluate their potential adoption in autonomous vehicles and especially as complementary modules in the perception and planning software stacks. However, their evaluation is limited in synthetic datasets or manually driving datasets without the ground truth knowledge and more precisely, how the current perception and planning algorithms would perform in the cases under evaluation. For this reason, this work evaluates LLMs on real-world edge cases where current autonomous vehicles have been proven to fail. The proposed architecture consists of an open vocabulary object detector coupled with prompt engineering and large language model contextual reasoning. We evaluate several state-of-the-art models against real edge cases and provide qualitative comparison results along with a discussion on the findings for the potential application of LLMs as anomaly detectors in autonomous vehicles.

**arXiv ID:** 2509.05315
</details>

<details>
<summary><strong>ChinaTravel: An Open-Ended Benchmark for Language Agents in Chinese Travel Planning</strong> - Jie-Jing Shao, Bo-Wen Zhang, Xiao-Wen Yang, Baizhi Chen, Si-Yu Han, Wen-Da Wei, Guohao Cai, Zhenhua Dong, Lan-Zhe Guo, Yu-Feng Li - [[pdf]](https://arxiv.org/pdf/2412.13682)</summary>

**Abstract:** Recent advances in LLMs, particularly in language reasoning and tool integration, have rapidly sparked the \emph{Language Agents} for real-world development. Among these, travel planning represents a prominent domain, combining complex multi-objective planning challenges with practical deployment demands. However, existing benchmarks often oversimplify real-world requirements by focusing on synthetic queries and limited constraints. We address the gap of evaluating language agents in multi-day, multi-POI travel planning scenarios with diverse and open human needs. Specifically, we introduce \emph{ChinaTravel}, the first open-ended benchmark grounded in authentic Chinese travel requirements collected from 1,154 human participants. We design a compositionally generalizable domain-specific language (DSL) for scalable evaluation, covering feasibility, constraint satisfaction, and preference comparison. Empirical studies reveal the potential of neuro-symbolic agents in travel planning, achieving a 37.0\% constraint satisfaction rate on human queries, a 10\times improvement over purely neural models. These findings highlight ChinaTravel as a pivotal milestone for advancing language agents in complex, real-world planning scenarios.

**arXiv ID:** 2412.13682
</details>

<details>
<summary><strong>MedFactEval and MedAgentBrief: A Framework and Workflow for Generating and Evaluating Factual Clinical Summaries</strong> - François Grolleau, Emily Alsentzer, Timothy Keyes, Philip Chung, Akshay Swaminathan, Asad Aali, Jason Hom, Tridu Huynh, Thomas Lew, April S. Liang, Weihan Chu, Natasha Z. Steele, Christina F. Lin, Jingkun Yang, Kameron C. Black, Stephen P. Ma, Fateme N. Haredasht, Nigam H. Shah, Kevin Schulman, Jonathan H. Chen - [[pdf]](https://arxiv.org/pdf/2509.05878)</summary>

**Abstract:** Evaluating factual accuracy in Large Language Model (LLM)-generated clinical text is a critical barrier to adoption, as expert review is unscalable for the continuous quality assurance these systems require. We address this challenge with two complementary contributions. First, we introduce MedFactEval, a framework for scalable, fact-grounded evaluation where clinicians define high-salience key facts and an "LLM Jury"--a multi-LLM majority vote--assesses their inclusion in generated summaries. Second, we present MedAgentBrief, a model-agnostic, multi-step workflow designed to generate high-quality, factual discharge summaries. To validate our evaluation framework, we established a gold-standard reference using a seven-physician majority vote on clinician-defined key facts from inpatient cases. The MedFactEval LLM Jury achieved almost perfect agreement with this panel (Cohen's kappa=81%), a performance statistically non-inferior to that of a single human expert (kappa=67%, P < 0.001). Our work provides both a robust evaluation framework (MedFactEval) and a high-performing generation workflow (MedAgentBrief), offering a comprehensive approach to advance the responsible deployment of generative AI in clinical workflows.

**arXiv ID:** 2509.05878
</details>

<details>
<summary><strong>Conversational Code Generation: a Case Study of Designing a Dialogue System for Generating Driving Scenarios for Testing Autonomous Vehicles</strong> - Rimvydas Rubavicius, Antonio Valerio Miceli-Barone, Alex Lascarides, Subramanian Ramamoorthy - [[pdf]](https://arxiv.org/pdf/2410.09829)</summary>

**Abstract:** Cyber-physical systems like autonomous vehicles are tested in simulation before deployment, using domain-specific programs for scenario specification. To aid the testing of autonomous vehicles in simulation, we design a natural language interface, using an instruction-following large language model, to assist a non-coding domain expert in synthesising the desired scenarios and vehicle behaviours. We show that using it to convert utterances to the symbolic program is feasible, despite the very small training dataset. Human experiments show that dialogue is critical to successful simulation generation, leading to a 4.5 times higher success rate than a generation without engaging in extended conversation.

**arXiv ID:** 2410.09829
</details>

<details>
<summary><strong>Building Self-Evolving Agents via Experience-Driven Lifelong Learning: A Framework and Benchmark</strong> - Yuxuan Cai, Yipeng Hao, Jie Zhou, Hang Yan, Zhikai Lei, Rui Zhen, Zhenhua Han, Yutao Yang, Junsong Li, Qianjun Pan, Tianyu Huai, Qin Chen, Xin Li, Kai Chen, Bo Zhang, Xipeng Qiu, Liang He - [[pdf]](https://arxiv.org/pdf/2508.19005)</summary>

**Abstract:** As AI advances toward general intelligence, the focus is shifting from systems optimized for static tasks to creating open-ended agents that learn continuously. In this paper, we introduce Experience-driven Lifelong Learning (ELL), a framework for building self-evolving agents capable of continuous growth through real-world interaction. The framework is built on four core principles: (1) Experience Exploration: Agents learn through continuous, self-motivated interaction with dynamic environments, navigating interdependent tasks and generating rich experiential trajectories. (2) Long-term Memory: Agents preserve and structure historical knowledge, including personal experiences, domain expertise, and commonsense reasoning, into a persistent memory system. (3) Skill Learning: Agents autonomously improve by abstracting recurring patterns from experience into reusable skills, which are actively refined and validated for application in new tasks. (4) Knowledge Internalization: Agents internalize explicit and discrete experiences into implicit and intuitive capabilities as "second nature".
We also introduce StuLife, a benchmark dataset for ELL that simulates a student's holistic college journey, from enrollment to academic and personal development, across three core phases and ten detailed sub-scenarios. StuLife is designed around three key paradigm

**arXiv ID:** 2508.19005
</details>

<details>
<summary><strong>IPR: Intelligent Prompt Routing with User-Controlled Quality-Cost Trade-offs</strong> - Aosong Feng, Zhichao Xu, Xian Wu, Kang Zhou, Sheng Guan, Yueyan Chen, Ninad Kulkarni, Yun Zhou, Balasubramaniam Srinivasan, Haibo Ding, Lin Lee Cheong - [[pdf]](https://arxiv.org/pdf/2509.06274)</summary>

**Abstract:** Routing incoming queries to the most cost-effective LLM while maintaining response quality poses a fundamental challenge in optimizing performance-cost trade-offs for large-scale commercial systems. We present IPR\, a quality-constrained Intelligent Prompt Routing framework that dynamically selects optimal models based on predicted response quality and user-specified tolerance levels. IPR introduces three key innovations: (1) a modular architecture with lightweight quality estimators trained on 1.5M prompts annotated with calibrated quality scores, enabling fine-grained quality prediction across model families; (2) a user-controlled routing mechanism with tolerance parameter $\tau \in [0,1]$ that provides explicit control over quality-cost trade-offs; and (3) an extensible design using frozen encoders with model-specific adapters, reducing new model integration from days to hours. To rigorously train and evaluate IPR, we curate an industrial-level dataset IPRBench\footnote{IPRBench will be released upon legal approval.}, a comprehensive benchmark containing 1.5 million examples with response quality annotations across 11 LLM candidates. Deployed on a major cloud platform, IPR achieves 43.9\% cost reduction while maintaining quality parity with the strongest model in the Claude family and processes requests with sub-150ms latency.

**arXiv ID:** 2509.06274
</details>

<details>
<summary><strong>Scenario-based Decision-making Using Game Theory for Interactive Autonomous Driving: A Survey</strong> - Zhihao Lin, Zhen Tian - [[pdf]](https://arxiv.org/pdf/2509.05777)</summary>

**Abstract:** Game-based interactive driving simulations have emerged as versatile platforms for advancing decision-making algorithms in road transport mobility. While these environments offer safe, scalable, and engaging settings for testing driving strategies, ensuring both realism and robust performance amid dynamic and diverse scenarios remains a significant challenge. Recently, the integration of game-based techniques with advanced learning frameworks has enabled the development of adaptive decision-making models that effectively manage the complexities inherent in varied driving conditions. These models outperform traditional simulation methods, especially when addressing scenario-specific challenges, ranging from obstacle avoidance on highways and precise maneuvering during on-ramp merging to navigation in roundabouts, unsignalized intersections, and even the high-speed demands of autonomous racing. Despite numerous innovations in game-based interactive driving, a systematic review comparing these approaches across different scenarios is still missing. This survey provides a comprehensive evaluation of game-based interactive driving methods by summarizing recent advancements and inherent roadway features in each scenario. Furthermore, the reviewed algorithms are critically assessed based on their adaptation of the standard game model and an analysis of their specific mechanisms to understand their impact on decision-making performance. Finally, the survey discusses the limitations of current approaches and outlines promising directions for future research.

**arXiv ID:** 2509.05777
</details>

<details>
<summary><strong>T-araVLN: Translator for Agricultural Robotic Agents on Vision-and-Language Navigation</strong> - Xiaobei Zhao, Xingqi Lyu, Xiang Li - [[pdf]](https://arxiv.org/pdf/2509.06644)</summary>

**Abstract:** Agricultural robotic agents have been becoming powerful helpers in a wide range of agricultural tasks, nevertheless, still heavily rely on manual operation or untransportable railway for movement. The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling agents navigate to the target position following the natural language instructions. AgriVLN effectively understands the simple instructions, however, often misunderstands the complicated instructions. To bridge this gap, we propose the method of Translator for Agricultural Robotic Agents on Vision-and-Language Navigation (T-araVLN), in which the Instruction Translator module translates the original instruction to be both refined and precise. Being evaluated on the A2A benchmark, our T-araVLN effectively improves SR from 0.47 to 0.63 and reduces NE from 2.91m to 2.28m, demonstrating the state-of-the-art performance in the agricultural domain. Code: this https URL.

**arXiv ID:** 2509.06644
</details>

<details>
<summary><strong>Sense4FL: Vehicular Crowdsensing Enhanced Federated Learning for Object Detection in Autonomous Driving</strong> - Yanan Ma, Senkang Hu, Zhengru Fang, Yun Ji, Yiqin Deng, Yuguang Fang - [[pdf]](https://arxiv.org/pdf/2503.17697)</summary>

**Abstract:** To accommodate constantly changing road conditions, real-time vision model training is essential for autonomous driving (AD). Federated learning (FL) serves as a promising paradigm to enable autonomous vehicles to train models collaboratively with their onboard computing resources. However, existing vehicle selection schemes for FL all assume predetermined and location-independent vehicles' datasets, neglecting the fact that vehicles collect training data along their routes, thereby resulting in suboptimal vehicle selection. In this paper, we focus on the fundamental perception problem and propose Sense4FL, a vehicular crowdsensing-enhanced FL framework featuring \textit{trajectory-dependent} vehicular \textit{training data collection} to \rev{improve the object detection quality} in AD for a region. To this end, we first derive the convergence bound of FL by considering the impact of both vehicles' uncertain trajectories and uploading probabilities, from which we discover that minimizing the training loss is equivalent to minimizing a weighted sum of local and global earth mover's distance (EMD) between vehicles' collected data distribution and global data distribution. Based on this observation, we formulate the trajectory-dependent vehicle selection and data collection problem for FL in AD. Given that the problem is NP-hard, we develop an efficient algorithm to find the solution with an approximation guarantee. Extensive simulation results have demonstrated the effectiveness of our approach in improving object detection performance compared with existing benchmarks.

**arXiv ID:** 2503.17697
</details>

<details>
<summary><strong>Semi-SMD: Semi-Supervised Metric Depth Estimation via Surrounding Cameras for Autonomous Driving</strong> - Yusen Xie, Zhengmin Huang, Shaojie Shen, Jun Ma - [[pdf]](https://arxiv.org/pdf/2503.19713)</summary>

**Abstract:** In this paper, we introduce Semi-SD, a novel metric depth estimation framework tailored for surrounding cameras equipment in autonomous driving. In this work, the input data consists of adjacent surrounding frames and camera parameters. We propose a unified spatial-temporal-semantic fusion module to construct the visual fused features. Cross-attention components for surrounding cameras and adjacent frames are utilized to focus on metric scale information refinement and temporal feature matching. Building on this, we propose a pose estimation framework using surrounding cameras, their corresponding estimated depths, and extrinsic parameters, which effectively address the scale ambiguity in multi-camera setups. Moreover, semantic world model and monocular depth estimation world model are integrated to supervised the depth estimation, which improve the quality of depth estimation. We evaluate our algorithm on DDAD and nuScenes datasets, and the results demonstrate that our method achieves state-of-the-art performance in terms of surrounding camera based depth estimation quality. The source code will be available on this https URL.

**arXiv ID:** 2503.19713
</details>

<details>
<summary><strong>Multi-Modal Multi-Task (M3T) Federated Foundation Models for Embodied AI: Potentials and Challenges for Edge Integration</strong> - Kasra Borazjani, Payam Abdisarabshali, Fardis Nadimi, Naji Khosravan, Minghui Liwang, Xianbin Wang, Yiguang Hong, Seyyedali Hosseinalipour - [[pdf]](https://arxiv.org/pdf/2505.11191)</summary>

**Abstract:** As embodied AI systems become increasingly multi-modal, personalized, and interactive, they must learn effectively from diverse sensory inputs, adapt continually to user preferences, and operate safely under resource and privacy constraints. These challenges expose a pressing need for machine learning models capable of swift, context-aware adaptation while balancing model generalization and personalization. Here, two methods emerge as suitable candidates, each offering parts of these capabilities: multi-modal multi-task foundation models (M3T-FMs) provide a pathway toward generalization across tasks and modalities, whereas federated learning (FL) offers the infrastructure for distributed, privacy-preserving model updates and user-level model personalization. However, when used in isolation, each of these approaches falls short of meeting the complex and diverse capability requirements of real-world embodied AI environments. In this vision paper, we introduce multi-modal multi-task federated foundation models (M3T-FFMs) for embodied AI, a new paradigm that unifies the strengths of M3T-FMs with the privacy-preserving distributed training nature of FL, enabling intelligent systems at the wireless edge. We collect critical deployment dimensions of M3T-FFMs in embodied AI ecosystems under a unified framework, which we name "EMBODY": Embodiment heterogeneity, Modality richness and imbalance, Bandwidth and compute constraints, On-device continual learning, Distributed control and autonomy, and Yielding safety, privacy, and personalization. For each, we identify concrete challenges and envision actionable research directions. We also present an evaluation framework for deploying M3T-FFMs in embodied AI systems, along with the associated trade-offs. Finally, we present a prototype implementation of M3T-FFMs and evaluate their energy and latency performance.

**arXiv ID:** 2505.11191
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>REMI: A Novel Causal Schema Memory Architecture for Personalized Lifestyle Recommendation Agents</strong> - Vishal Raman, Vijai Aravindh R, Abhijith Ragav - [[pdf]](https://arxiv.org/pdf/2509.06269)</summary>

**Abstract:** Personalized AI assistants often struggle to incorporate complex personal data and causal knowledge, leading to generic advice that lacks explanatory power. We propose REMI, a Causal Schema Memory architecture for a multimodal lifestyle agent that integrates a personal causal knowledge graph, a causal reasoning engine, and a schema based planning module. The idea is to deliver explainable, personalized recommendations in domains like fashion, personal wellness, and lifestyle planning. Our architecture uses a personal causal graph of the user's life events and habits, performs goal directed causal traversals enriched with external knowledge and hypothetical reasoning, and retrieves adaptable plan schemas to generate tailored action plans. A Large Language Model orchestrates these components, producing answers with transparent causal explanations. We outline the CSM system design and introduce new evaluation metrics for personalization and explainability, including Personalization Salience Score and Causal Reasoning Accuracy, to rigorously assess its performance. Results indicate that CSM based agents can provide more context aware, user aligned recommendations compared to baseline LLM agents. This work demonstrates a novel approach to memory augmented, causal reasoning in personalized agents, advancing the development of transparent and trustworthy AI lifestyle assistants.

**arXiv ID:** 2509.06269
</details>

<details>
<summary><strong>Transforming Wearable Data into Personal Health Insights using Large Language Model Agents</strong> - Mike A. Merrill, Akshay Paruchuri, Naghmeh Rezaei, Geza Kovacs, Javier Perez, Yun Liu, Erik Schenck, Nova Hammerquist, Jake Sunshine, Shyam Tailor, Kumar Ayush, Hao-Wei Su, Qian He, Cory Y. McLean, Mark Malhotra, Shwetak Patel, Jiening Zhan, Tim Althoff, Daniel McDuff, Xin Liu - [[pdf]](https://arxiv.org/pdf/2406.06464)</summary>

**Abstract:** Deriving personalized insights from popular wearable trackers requires complex numerical reasoning that challenges standard LLMs, necessitating tool-based approaches like code generation. Large language model (LLM) agents present a promising yet largely untapped solution for this analysis at scale. We introduce the Personal Health Insights Agent (PHIA), a system leveraging multistep reasoning with code generation and information retrieval to analyze and interpret behavioral health data. To test its capabilities, we create and share two benchmark datasets with over 4000 health insights questions. A 650-hour human expert evaluation shows that PHIA significantly outperforms a strong code generation baseline, achieving 84% accuracy on objective, numerical questions and, for open-ended ones, earning 83% favorable ratings while being twice as likely to achieve the highest quality rating. This work can advance behavioral health by empowering individuals to understand their data, enabling a new era of accessible, personalized, and data-driven wellness for the wider population.

**arXiv ID:** 2406.06464
</details>

<details>
<summary><strong>Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem</strong> - Ryosuke Takata, Atsushi Masumori, Takashi Ikegami - [[pdf]](https://arxiv.org/pdf/2509.04537)</summary>

**Abstract:** We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents.

**arXiv ID:** 2509.04537
</details>

<details>
<summary><strong>Are LLM Agents Behaviorally Coherent? Latent Profiles for Social Simulation</strong> - James Mooney, Josef Woldense, Zheng Robert Jia, Shirley Anugrah Hayati, My Ha Nguyen, Vipul Raheja, Dongyeop Kang - [[pdf]](https://arxiv.org/pdf/2509.03736)</summary>

**Abstract:** The impressive capabilities of Large Language Models (LLMs) have fueled the notion that synthetic agents can serve as substitutes for real participants in human-subject research. In an effort to evaluate the merits of this claim, social science researchers have largely focused on whether LLM-generated survey data corresponds to that of a human counterpart whom the LLM is prompted to represent. In contrast, we address a more fundamental question: Do agents maintain internal consistency, retaining similar behaviors when examined under different experimental settings? To this end, we develop a study designed to (a) reveal the agent's internal state and (b) examine agent behavior in a basic dialogue setting. This design enables us to explore a set of behavioral hypotheses to assess whether an agent's conversation behavior is consistent with what we would expect from their revealed internal state. Our findings on these hypotheses show significant internal inconsistencies in LLMs across model families and at differing model sizes. Most importantly, we find that, although agents may generate responses matching those of their human counterparts, they fail to be internally consistent, representing a critical gap in their capabilities to accurately substitute for real participants in human-subject research. Our simulation code and data are publicly accessible.

**arXiv ID:** 2509.03736
</details>

<details>
<summary><strong>FinAgentBench: A Benchmark Dataset for Agentic Retrieval in Financial Question Answering</strong> - Chanyeol Choi, Jihoon Kwon, Alejandro Lopez-Lira, Chaewoon Kim, Minjae Kim, Juneha Hwang, Jaeseon Ha, Hojun Choi, Suyeol Yun, Yongjin Kim, Yongjae Lee - [[pdf]](https://arxiv.org/pdf/2508.14052)</summary>

**Abstract:** Accurate information retrieval (IR) is critical in the financial domain, where investors must identify relevant information from large collections of documents. Traditional IR methods-whether sparse or dense-often fall short in retrieval accuracy, as it requires not only capturing semantic similarity but also performing fine-grained reasoning over document structure and domain-specific knowledge. Recent advances in large language models (LLMs) have opened up new opportunities for retrieval with multi-step reasoning, where the model ranks passages through iterative reasoning about which information is most relevant to a given query. However, there exists no benchmark to evaluate such capabilities in the financial domain. To address this gap, we introduce FinAgentBench, the first large-scale benchmark for evaluating retrieval with multi-step reasoning in finance -- a setting we term agentic retrieval. The benchmark consists of 3,429 expert-annotated examples on S&P-100 listed firms and assesses whether LLM agents can (1) identify the most relevant document type among candidates, and (2) pinpoint the key passage within the selected document. Our evaluation framework explicitly separates these two reasoning steps to address context limitations. This design enables to provide a quantitative basis for understanding retrieval-centric LLM behavior in finance. We evaluate a suite of state-of-the-art models and further demonstrated how targeted fine-tuning can significantly improve agentic retrieval performance. Our benchmark provides a foundation for studying retrieval-centric LLM behavior in complex, domain-specific tasks for finance.

**arXiv ID:** 2508.14052
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (27 papers)</h2></summary>

<details>
<summary><strong>SasAgent: Multi-Agent AI System for Small-Angle Scattering Data Analysis</strong> - Lijie Ding, Changwoo Do - [[pdf]](https://arxiv.org/pdf/2509.05363)</summary>

**Abstract:** We introduce SasAgent, a multi-agent AI system powered by large language models (LLMs) that automates small-angle scattering (SAS) data analysis by leveraging tools from the SasView software and enables user interaction via text input. SasAgent features a coordinator agent that interprets user prompts and delegates tasks to three specialized agents for scattering length density (SLD) calculation, synthetic data generation, and experimental data fitting. These agents utilize LLM-friendly tools to execute tasks efficiently. These tools, including the model data tool, Retrieval-Augmented Generation (RAG) documentation tool, bump fitting tool, and SLD calculator tool, are derived from the SasView Python library. A user-friendly Gradio-based interface enhances user accessibility. Through diverse examples, we demonstrate SasAgent's ability to interpret complex prompts, calculate SLDs, generate accurate scattering data, and fit experimental datasets with high precision. This work showcases the potential of LLM-driven AI systems to streamline scientific workflows and enhance automation in SAS research.

**arXiv ID:** 2509.05363
</details>

<details>
<summary><strong>Code Like Humans: A Multi-Agent Solution for Medical Coding</strong> - Andreas Motzfeldt, Joakim Edin, Casper L. Christensen, Christian Hardmeier, Lars Maaløe, Anna Rogers - [[pdf]](https://arxiv.org/pdf/2509.05378)</summary>

**Abstract:** In medical coding, experts map unstructured clinical notes to alphanumeric codes for diagnoses and procedures. We introduce Code Like Humans: a new agentic framework for medical coding with large language models. It implements official coding guidelines for human experts, and it is the first solution that can support the full ICD-10 coding system (+70K labels). It achieves the best performance to date on rare diagnosis codes (fine-tuned discriminative classifiers retain an advantage for high-frequency codes, to which they are limited). Towards future work, we also contribute an analysis of system performance and identify its `blind spots' (codes that are systematically undercoded).

**arXiv ID:** 2509.05378
</details>

<details>
<summary><strong>From Image Generation to Infrastructure Design: a Multi-agent Pipeline for Street Design Generation</strong> - Chenguang Wang, Xiang Yan, Yilong Dai, Ziyi Wang, Susu Xu - [[pdf]](https://arxiv.org/pdf/2509.05469)</summary>

**Abstract:** Realistic visual renderings of street-design scenarios are essential for public engagement in active transportation planning. Traditional approaches are labor-intensive, hindering collective deliberation and collaborative decision-making. While AI-assisted generative design shows transformative potential by enabling rapid creation of design scenarios, existing generative approaches typically require large amounts of domain-specific training data and struggle to enable precise spatial variations of design/configuration in complex street-view scenes. We introduce a multi-agent system that edits and redesigns bicycle facilities directly on real-world street-view imagery. The framework integrates lane localization, prompt optimization, design generation, and automated evaluation to synthesize realistic, contextually appropriate designs. Experiments across diverse urban scenarios demonstrate that the system can adapt to varying road geometries and environmental conditions, consistently yielding visually coherent and instruction-compliant results. This work establishes a foundation for applying multi-agent pipelines to transportation infrastructure planning and facility design.

**arXiv ID:** 2509.05469
</details>

<details>
<summary><strong>Chatbot To Help Patients Understand Their Health</strong> - Won Seok Jang, Hieu Tran, Manav Mistry, SaiKiran Gandluri, Yifan Zhang, Sharmin Sultana, Sunjae Kown, Yuan Zhang, Zonghai Yao, Hong Yu - [[pdf]](https://arxiv.org/pdf/2509.05818)</summary>

**Abstract:** Patients must possess the knowledge necessary to actively participate in their care. We present NoteAid-Chatbot, a conversational AI that promotes patient understanding via a novel 'learning as conversation' framework, built on a multi-agent large language model (LLM) and reinforcement learning (RL) setup without human-labeled data. NoteAid-Chatbot was built on a lightweight LLaMA 3.2 3B model trained in two stages: initial supervised fine-tuning on conversational data synthetically generated using medical conversation strategies, followed by RL with rewards derived from patient understanding assessments in simulated hospital discharge scenarios. Our evaluation, which includes comprehensive human-aligned assessments and case studies, demonstrates that NoteAid-Chatbot exhibits key emergent behaviors critical for patient education, such as clarity, relevance, and structured dialogue, even though it received no explicit supervision for these attributes. Our results show that even simple Proximal Policy Optimization (PPO)-based reward modeling can successfully train lightweight, domain-specific chatbots to handle multi-turn interactions, incorporate diverse educational strategies, and meet nuanced communication objectives. Our Turing test demonstrates that NoteAid-Chatbot surpasses non-expert human. Although our current focus is on healthcare, the framework we present illustrates the feasibility and promise of applying low-cost, PPO-based RL to realistic, open-ended conversational domains, broadening the applicability of RL-based alignment methods.

**arXiv ID:** 2509.05818
</details>

<details>
<summary><strong>MapAgent: A Hierarchical Agent for Geospatial Reasoning with Dynamic Map Tool Integration</strong> - Md Hasebul Hasan, Mahir Labib Dihan, Mohammed Eunus Ali, Md Rizwan Parvez - [[pdf]](https://arxiv.org/pdf/2509.05933)</summary>

**Abstract:** Agentic AI has significantly extended the capabilities of large language models (LLMs) by enabling complex reasoning and tool use. However, most existing frameworks are tailored to domains such as mathematics, coding, or web automation, and fall short on geospatial tasks that require spatial reasoning, multi-hop planning, and real-time map interaction. To address these challenges, we introduce MapAgent, a hierarchical multi-agent plug-and-play framework with customized toolsets and agentic scaffolds for map-integrated geospatial reasoning. Unlike existing flat agent-based approaches that treat tools uniformly-often overwhelming the LLM when handling similar but subtly different geospatial APIs-MapAgent decouples planning from execution. A high-level planner decomposes complex queries into subgoals, which are routed to specialized modules. For tool-heavy modules-such as map-based services-we then design a dedicated map-tool agent that efficiently orchestrates related APIs adaptively in parallel to effectively fetch geospatial data relevant for the query, while simpler modules (e.g., solution generation or answer extraction) operate without additional agent overhead. This hierarchical design reduces cognitive load, improves tool selection accuracy, and enables precise coordination across similar APIs. We evaluate MapAgent on four diverse geospatial benchmarks-MapEval-Textual, MapEval-API, MapEval-Visual, and MapQA-and demonstrate substantial gains over state-of-the-art tool-augmented and agentic baselines. We open-source our framwork at this https URL.

**arXiv ID:** 2509.05933
</details>

<details>
<summary><strong>PillagerBench: Benchmarking LLM-Based Agents in Competitive Minecraft Team Environments</strong> - Olivier Schipper, Yudi Zhang, Yali Du, Mykola Pechenizkiy, Meng Fang - [[pdf]](https://arxiv.org/pdf/2509.06235)</summary>

**Abstract:** LLM-based agents have shown promise in various cooperative and strategic reasoning tasks, but their effectiveness in competitive multi-agent environments remains underexplored. To address this gap, we introduce PillagerBench, a novel framework for evaluating multi-agent systems in real-time competitive team-vs-team scenarios in Minecraft. It provides an extensible API, multi-round testing, and rule-based built-in opponents for fair, reproducible comparisons. We also propose TactiCrafter, an LLM-based multi-agent system that facilitates teamwork through human-readable tactics, learns causal dependencies, and adapts to opponent strategies. Our evaluation demonstrates that TactiCrafter outperforms baseline approaches and showcases adaptive learning through self-play. Additionally, we analyze its learning process and strategic evolution over multiple game episodes. To encourage further research, we have open-sourced PillagerBench, fostering advancements in multi-agent AI for competitive environments.

**arXiv ID:** 2509.06235
</details>

<details>
<summary><strong>SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents</strong> - Xuan-Phi Nguyen, Shrey Pandit, Revanth Gangi Reddy, Austin Xu, Silvio Savarese, Caiming Xiong, Shafiq Joty - [[pdf]](https://arxiv.org/pdf/2509.06283)</summary>

**Abstract:** Equipping large language models (LLMs) with complex, interleaved reasoning and tool-use capabilities has become a key focus in agentic AI research, especially with recent advances in reasoning-oriented (``thinking'') models. Such capabilities are key to unlocking a number of important applications. One such application is Deep Research (DR), which requires extensive search and reasoning over many sources. Our work in this paper focuses on the development of native Autonomous Single-Agent models for DR featuring minimal web crawling and Python tool integration. Unlike multi-agent systems, where agents take up pre-defined roles and are told what to do at each step in a static workflow, an autonomous single-agent determines its next action dynamically based on context, without manual directive. While prior work has proposed training recipes for base or instruction-tuned LLMs, we focus on continual reinforcement learning (RL) of reasoning-optimized models to further enhance agentic skills while preserving reasoning ability. Towards this end, we propose a simple RL recipe with entirely synthetic data, which we apply to various open-source LLMs. Our best variant SFR-DR-20B achieves up to 28.7% on Humanity's Last Exam benchmark. In addition, we conduct key analysis experiments to provide more insights into our methodologies.

**arXiv ID:** 2509.06283
</details>

<details>
<summary><strong>A data-driven discretized CS:GO simulation environment to facilitate strategic multi-agent planning research</strong> - Yunzhe Wang, Volkan Ustun, Chris McGroarty - [[pdf]](https://arxiv.org/pdf/2509.06355)</summary>

**Abstract:** Modern simulation environments for complex multi-agent interactions must balance high-fidelity detail with computational efficiency. We present DECOY, a novel multi-agent simulator that abstracts strategic, long-horizon planning in 3D terrains into high-level discretized simulation while preserving low-level environmental fidelity. Using Counter-Strike: Global Offensive (CS:GO) as a testbed, our framework accurately simulates gameplay using only movement decisions as tactical positioning -- without explicitly modeling low-level mechanics such as aiming and shooting. Central to our approach is a waypoint system that simplifies and discretizes continuous states and actions, paired with neural predictive and generative models trained on real CS:GO tournament data to reconstruct event outcomes. Extensive evaluations show that replays generated from human data in DECOY closely match those observed in the original game. Our publicly available simulation environment provides a valuable tool for advancing research in strategic multi-agent planning and behavior generation.

**arXiv ID:** 2509.06355
</details>

<details>
<summary><strong>Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning</strong> - Song Yu, Xiaofei Xu, Ke Deng, Li Li, Lin Tian - [[pdf]](https://arxiv.org/pdf/2509.06436)</summary>

**Abstract:** Large language models (LLMs) face persistent challenges when handling long-context tasks, most notably the lost in the middle issue, where information located in the middle of a long input tends to be underutilized. Some existing methods that reduce input have the risk of discarding key information, while others that extend context windows often lead to attention dispersion. To address these limitations, we propose Tree of Agents (TOA), a multi-agent reasoning framework that segments the input into chunks processed by independent agents. Each agent generates its local cognition, then agents dynamically exchange information for collaborative reasoning along tree-structured paths. TOA enables agents to probe different reasoning orders for multi-perspective understanding, effectively mitigating position bias and reducing hallucinations. To improve processing efficiency, we incorporate prefix-hash caching and adaptive pruning strategies, achieving significant performance improvements with comparable API overhead. Experiments show that TOA, powered by compact LLaMA3.1-8B, significantly outperforms multiple baselines and demonstrates comparable performance to the latest and much larger commercial models, such as Gemini1.5-pro, on various long-context tasks. Code is available at this https URL.

**arXiv ID:** 2509.06436
</details>

<details>
<summary><strong>Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers</strong> - Ran Xin, Zeyu Zheng, Yanchen Nie, Kun Yuan, Xia Xiao - [[pdf]](https://arxiv.org/pdf/2509.06493)</summary>

**Abstract:** The integration of Large Language Models (LLMs) into automated theorem proving has shown immense promise, yet is fundamentally constrained by challenges in scaling up both training-time reinforcement learning (RL) and inference-time compute. This paper introduces \texttt{BFS-Prover-V2}, a system designed to address this dual scaling problem. We present two primary innovations. The first is a novel multi-turn off-policy RL framework for continually improving the performance of LLM step-prover at training time. This framework, inspired by the principles of AlphaZero, utilizes a multi-stage expert iteration pipeline featuring adaptive tactic-level data filtering and periodic retraining to surmount the performance plateaus that typically curtail long-term RL in LLM-based agents. The second innovation is a planner-enhanced multi-agent search architecture that scales reasoning capabilities at inference time. This architecture employs a general reasoning model as a high-level planner to iteratively decompose complex theorems into a sequence of simpler subgoals. This hierarchical approach substantially reduces the search space, enabling a team of parallel prover agents to collaborate efficiently by leveraging a shared proof cache. We demonstrate that this dual approach to scaling yields state-of-the-art results on established formal mathematics benchmarks. \texttt{BFS-Prover-V2} achieves 95.08\% and 41.4\% on the MiniF2F and ProofNet test sets respectively. While demonstrated in the domain of formal mathematics, the RL and inference techniques presented in this work are of broader interest and may be applied to other domains requiring long-horizon multi-turn reasoning and complex search.

**arXiv ID:** 2509.06493
</details>

<details>
<summary><strong>Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate</strong> - Andrea Wynn, Harsh Satija, Gillian Hadfield - [[pdf]](https://arxiv.org/pdf/2509.05396)</summary>

**Abstract:** While multi-agent debate has been proposed as a promising strategy for improving AI reasoning ability, we find that debate can sometimes be harmful rather than helpful. The prior work has exclusively focused on debates within homogeneous groups of agents, whereas we explore how diversity in model capabilities influences the dynamics and outcomes of multi-agent interactions. Through a series of experiments, we demonstrate that debate can lead to a decrease in accuracy over time -- even in settings where stronger (i.e., more capable) models outnumber their weaker counterparts. Our analysis reveals that models frequently shift from correct to incorrect answers in response to peer reasoning, favoring agreement over challenging flawed reasoning. These results highlight important failure modes in the exchange of reasons during multi-agent debate, suggesting that naive applications of debate may cause performance degradation when agents are neither incentivized nor adequately equipped to resist persuasive but incorrect reasoning.

**arXiv ID:** 2509.05396
</details>

<details>
<summary><strong>Orchestrator: Active Inference for Multi-Agent Systems in Long-Horizon Tasks</strong> - Lukas Beckenbauer, Johannes-Lucas Loewe, Ge Zheng, Alexandra Brintrup - [[pdf]](https://arxiv.org/pdf/2509.05651)</summary>

**Abstract:** Complex, non-linear tasks challenge LLM-enhanced multi-agent systems (MAS) due to partial observability and suboptimal coordination. We propose Orchestrator, a novel MAS framework that leverages attention-inspired self-emergent coordination and reflective benchmarking to optimize global task performance. Orchestrator introduces a monitoring mechanism to track agent-environment dynamics, using active inference benchmarks to optimize system behavior. By tracking agent-to-agent and agent-to-environment interaction, Orchestrator mitigates the effects of partial observability and enables agents to approximate global task solutions more efficiently. We evaluate the framework on a series of maze puzzles of increasing complexity, demonstrating its effectiveness in enhancing coordination and performance in dynamic, non-linear environments with long-horizon objectives.

**arXiv ID:** 2509.05651
</details>

<details>
<summary><strong>HECATE: An ECS-based Framework for Teaching and Developing Multi-Agent Systems</strong> - Arthur Casals, Anarosa A. F. Brandão - [[pdf]](https://arxiv.org/pdf/2509.06431)</summary>

**Abstract:** This paper introduces HECATE, a novel framework based on the Entity-Component-System (ECS) architectural pattern that bridges the gap between distributed systems engineering and MAS development. HECATE is built using the Entity-Component-System architectural pattern, leveraging data-oriented design to implement multiagent systems. This approach involves engineering multiagent systems (MAS) from a distributed systems (DS) perspective, integrating agent concepts directly into the DS domain. This approach simplifies MAS development by (i) reducing the need for specialized agent knowledge and (ii) leveraging familiar DS patterns and standards to minimize the agent-specific knowledge required for engineering MAS. We present the framework's architecture, core components, and implementation approach, demonstrating how it supports different agent models.

**arXiv ID:** 2509.06431
</details>

<details>
<summary><strong>Demo: Healthcare Agent Orchestrator (HAO) for Patient Summarization in Molecular Tumor Boards</strong> - Noel Codella, Sam Preston, Hao Qiu, Leonardo Schettini, Wen-wai Yim, Mert Öz, Shrey Jain, Matthew P. Lungren, Thomas Osborne - [[pdf]](https://arxiv.org/pdf/2509.06602)</summary>

**Abstract:** Molecular Tumor Boards (MTBs) are multidisciplinary forums where oncology specialists collaboratively assess complex patient cases to determine optimal treatment strategies. A central element of this process is the patient summary, typically compiled by a medical oncologist, radiation oncologist, or surgeon, or their trained medical assistant, who distills heterogeneous medical records into a concise narrative to facilitate discussion. This manual approach is often labor-intensive, subjective, and prone to omissions of critical information. To address these limitations, we introduce the Healthcare Agent Orchestrator (HAO), a Large Language Model (LLM)-driven AI agent that coordinates a multi-agent clinical workflow to generate accurate and comprehensive patient summaries for MTBs. Evaluating predicted patient summaries against ground truth presents additional challenges due to stylistic variation, ordering, synonym usage, and phrasing differences, which complicate the measurement of both succinctness and completeness. To overcome these evaluation hurdles, we propose TBFact, a ``model-as-a-judge'' framework designed to assess the comprehensiveness and succinctness of generated summaries. Using a benchmark dataset derived from de-identified tumor board discussions, we applied TBFact to evaluate our Patient History agent. Results show that the agent captured 94% of high-importance information (including partial entailments) and achieved a TBFact recall of 0.84 under strict entailment criteria. We further demonstrate that TBFact enables a data-free evaluation framework that institutions can deploy locally without sharing sensitive clinical data. Together, HAO and TBFact establish a robust foundation for delivering reliable and scalable support to MTBs.

**arXiv ID:** 2509.06602
</details>

<details>
<summary><strong>GameGPT: Multi-agent Collaborative Framework for Game Development</strong> - Dake Chen, Haoyang Zhang, Hanbin Wang, Yunhao Huo, Yuzhao Li, Junjie Wang - [[pdf]](https://arxiv.org/pdf/2310.08067)</summary>

**Abstract:** The large language model (LLM) based agents have demonstrated their capacity to automate and expedite software development processes. In this paper, we focus on game development and propose a multi-agent collaborative framework, dubbed GameGPT, to automate game development. While many studies have pinpointed hallucination as a primary roadblock for deploying LLMs in production, we identify another concern: redundancy. Our framework presents a series of methods to mitigate both concerns. These methods include dual collaboration and layered approaches with several in-house lexicons, to mitigate the hallucination and redundancy in the planning, task identification, and implementation phases. Furthermore, a decoupling approach is also introduced to achieve code generation with better precision.

**arXiv ID:** 2310.08067
</details>

<details>
<summary><strong>KABB: Knowledge-Aware Bayesian Bandits for Dynamic Expert Coordination in Multi-Agent Systems</strong> - Jusheng Zhang, Zimeng Huang, Yijia Fan, Ningyuan Liu, Mingyan Li, Zhuojie Yang, Jiawei Yao, Jian Wang, Keze Wang - [[pdf]](https://arxiv.org/pdf/2502.07350)</summary>

**Abstract:** As scaling large language models faces prohibitive costs, multi-agent systems emerge as a promising alternative, though challenged by static knowledge assumptions and coordination inefficiencies. We introduces Knowledge-Aware Bayesian Bandits (KABB), a novel framework that enhances multi-agent system coordination through semantic understanding and dynamic adaptation. The framework features three key innovations: a three-dimensional knowledge distance model for deep semantic understanding, a dual-adaptation mechanism for continuous expert optimization, and a knowledge-aware Thompson Sampling strategy for efficient expert selection. Extensive evaluation demonstrates KABB achieves an optimal cost-performance balance, maintaining high performance while keeping computational demands relatively low in multi-agent coordination.

**arXiv ID:** 2502.07350
</details>

<details>
<summary><strong>MAPF-HD: Multi-Agent Path Finding in High-Density Environments</strong> - Hiroya Makino, Seigo Ito - [[pdf]](https://arxiv.org/pdf/2509.06374)</summary>

**Abstract:** Multi-agent path finding (MAPF) involves planning efficient paths for multiple agents to move simultaneously while avoiding collisions. In typical warehouse environments, agents are often sparsely distributed along aisles. However, increasing the agent density can improve space efficiency. When the agent density is high, we must optimize the paths not only for goal-assigned agents but also for those obstructing them. This study proposes a novel MAPF framework for high-density environments (MAPF-HD). Several studies have explored MAPF in similar settings using integer linear programming (ILP). However, ILP-based methods require substantial computation time to optimize all agent paths simultaneously. Even in small grid-based environments with fewer than $100$ cells, these computations can incur tens to hundreds of seconds. These high computational costs render these methods impractical for large-scale applications such as automated warehouses and valet parking. To address these limitations, we introduce the phased null-agent swapping (PHANS) method. PHANS employs a heuristic approach to incrementally swap positions between agents and empty vertices. This method solves the MAPF-HD problem within seconds to tens of seconds, even in large environments containing more than $700$ cells. The proposed method can potentially improve efficiency in various real-world applications such as warehouse logistics, traffic management, or crowd control. Code is available at this https URL.

**arXiv ID:** 2509.06374
</details>

<details>
<summary><strong>Code2MCP: A Multi-Agent Framework for Automated Transformation of Code Repositories into Model Context Protocol Services</strong> - Chaoqian Ouyang, Ling Yue, Shimin Di, Libin Zheng, Shaowu Pan, Min-Ling Zhang - [[pdf]](https://arxiv.org/pdf/2509.05941)</summary>

**Abstract:** The proliferation of Large Language Models (LLMs) has created a significant integration challenge in the AI agent ecosystem, often called the "$N \times M$ problem," where N models require custom integrations for M tools. This fragmentation stifles innovation and creates substantial development overhead. While the Model Context Protocol (MCP) has emerged as a standard to resolve this, its adoption is hindered by the manual effort required to convert the vast universe of existing software into MCP-compliant services. This is especially true for the millions of open-source repositories on GitHub, the world's largest collection of functional code. This paper introduces Code2MCP, a highly automated, agentic framework designed to transform any GitHub repository into a functional MCP service with minimal human intervention. Our system employs a multi-stage workflow that automates the entire process, from code analysis and environment configuration to service generation and deployment. A key innovation of our framework is an LLM-driven, closed-loop "Run--Review--Fix" cycle, which enables the system to autonomously debug and repair the code it generates. Code2MCP produces not only deployable services but also comprehensive technical documentation, acting as a catalyst to accelerate the MCP ecosystem by systematically unlocking the world's largest open-source code repository and automating the critical last mile of tool integration. The code is open-sourced at this https URL.

**arXiv ID:** 2509.05941
</details>

<details>
<summary><strong>Game Theory and Multi-Agent Reinforcement Learning for Zonal Ancillary Markets</strong> - Francesco Morri, Hélène Le Cadre, Pierre Gruet, Luce Brotcorne - [[pdf]](https://arxiv.org/pdf/2505.03288)</summary>

**Abstract:** We characterize zonal ancillary market coupling relying on noncooperative game theory. To that purpose, we formulate the ancillary market as a multi-leader single follower bilevel problem, that we subsequently cast as a generalized Nash game with side constraints and nonconvex feasibility sets. We determine conditions for equilibrium existence and show that the game has a generalized potential game structure. To compute market equilibrium, we rely on two exact approaches: an integrated optimization approach and Gauss-Seidel best-response, that we compare against multi-agent deep reinforcement learning. On real data from Germany and Austria, simulations indicate that multi-agent deep reinforcement learning achieves the smallest convergence rate but requires pretraining, while best-response is the slowest. On the economics side, multi-agent deep reinforcement learning results in smaller market costs compared to the exact methods, but at the cost of higher variability in the profit allocation among stakeholders. Further, stronger coupling between zones tends to reduce costs for larger zones.

**arXiv ID:** 2505.03288
</details>

<details>
<summary><strong>MAPF-World: Action World Model for Multi-Agent Path Finding</strong> - Zhanjiang Yang, Yang Shen, Yueming Li, Meng Li, Lijun Sun - [[pdf]](https://arxiv.org/pdf/2508.12087)</summary>

**Abstract:** Multi-agent path finding (MAPF) is the problem of planning conflict-free paths from the designated start locations to goal positions for multiple agents. It underlies a variety of real-world tasks, including multi-robot coordination, robot-assisted logistics, and social navigation. Recent decentralized learnable solvers have shown great promise for large-scale MAPF, especially when leveraging foundation models and large datasets. However, these agents are reactive policy models and exhibit limited modeling of environmental temporal dynamics and inter-agent dependencies, resulting in performance degradation in complex, long-term planning scenarios. To address these limitations, we propose MAPF-World, an autoregressive action world model for MAPF that unifies situation understanding and action generation, guiding decisions beyond immediate local observations. It improves situational awareness by explicitly modeling environmental dynamics, including spatial features and temporal dependencies, through future state and actions prediction. By incorporating these predicted futures, MAPF-World enables more informed, coordinated, and far-sighted decision-making, especially in complex multi-agent settings. Furthermore, we augment MAPF benchmarks by introducing an automatic map generator grounded in real-world scenarios, capturing practical map layouts for training and evaluating MAPF solvers. Extensive experiments demonstrate that MAPF-World outperforms state-of-the-art learnable solvers, showcasing superior zero-shot generalization to out-of-distribution cases. Notably, MAPF-World is trained with a 96.5% smaller model size and 92% reduced data.

**arXiv ID:** 2508.12087
</details>

<details>
<summary><strong>Beyond Keywords: Driving Generative Search Engine Optimization with Content-Centric Agents</strong> - Qiyuan Chen, Jiahe Chen, Hongsen Huang, Qian Shao, Jintai Chen, Renjie Hua, Hongxia Xu, Ruijia Wu, Ren Chuan, Jian Wu - [[pdf]](https://arxiv.org/pdf/2509.05607)</summary>

**Abstract:** The paradigm shift from traditional ranked-based search to Generative Search Engines has rendered conventional SEO metrics obsolete, creating an urgent need to understand, measure, and optimize for content influence on synthesized answers. This paper introduces a comprehensive, end-to-end framework for Generative Search Engine Optimization (GSEO) to address this challenge. We make two primary contributions. First, we construct CC-GSEO-Bench, a large-scale, content-centric benchmark, and propose a multi-dimensional evaluation framework that systematically quantifies influence, moving beyond surface-level attribution to assess substantive semantic impact. Second, we design a novel multi-agent system that operationalizes this framework, automating the strategic refinement of content through a collaborative analyze-revise-evaluate workflow. Our empirical analysis using this framework reveals novel insights into the dynamics of content influence, offering actionable strategies for creators and establishing a principled foundation for future GSEO research.

**arXiv ID:** 2509.05607
</details>

<details>
<summary><strong>ChatCFD: An LLM-Driven Agent for End-to-End CFD Automation with Domain-Specific Structured Reasoning</strong> - E Fan, Kang Hu, Zhuowen Wu, Jiangyang Ge, Jiawei Miao, Yuzhi Zhang, He Sun, Weizong Wang, Tianhan Zhang - [[pdf]](https://arxiv.org/pdf/2506.02019)</summary>

**Abstract:** Computational Fluid Dynamics (CFD) is essential for advancing scientific and engineering fields but is hindered by operational complexity, high expertise requirements, and limited accessibility. This paper introduces ChatCFD, an automated agent system for OpenFOAM simulations that processes multi-modal inputs (e.g., research papers, meshes) via an interactive interface, leveraging DeepSeek-R1 and DeepSeek-V3 large language models, a multi-agent architecture, and OpenFOAM knowledge. Its four-stage pipeline (Knowledge Base Construction, User Input Processing, Case File Generation, and Execution and Error Reflection) enables iterative trial-reflection-refinement for intricate setups, supporting diverse physical models and external meshes. Validation on 205 benchmark tutorial cases, 110 perturbed variants, and 2 literature-derived cases shows ChatCFD's 82.1 percent operational success rate on basic cases, outperforming MetaOpenFOAM (6.2 percent) and Foam-Agent (42.3 percent), and 60-80 percent on literature-derived complex cases. Turbulence model studies show a 40 percent success rate for common models versus 10 percent for rare ones like RNG k-epsilon. Physics coupling analyses reveal higher resource demands for multi-physics-coupled cases, while LLM bias toward simpler setups introduces persistent errors, such as dimensional inconsistency. Ablation studies highlight the efficacy of RAG-based modules and reflection mechanisms. By automating hypothesis testing and parameter exploration, ChatCFD accelerates scientific discovery in fluid mechanics and engineering, addressing LLM limitations through structured design and showing strong potential as a modular component in MCP-based agent networks for collaborative multi-agent systems, paving the way for scalable AI-driven CFD innovation. The code for ChatCFD is available at this https URL.

**arXiv ID:** 2506.02019
</details>

<details>
<summary><strong>Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting</strong> - Ana Rita Ortigoso, Gabriel Vieira, Daniel Fuentes, Luis Frazão, Nuno Costa, António Pereira - [[pdf]](https://arxiv.org/pdf/2505.20521)</summary>

**Abstract:** This paper presents Project Riley, a novel multimodal and multi-model conversational AI architecture oriented towards the simulation of reasoning influenced by emotional states. Drawing inspiration from Pixar's Inside Out, the system comprises five distinct emotional agents - Joy, Sadness, Fear, Anger, and Disgust - that engage in structured multi-round dialogues to generate, criticise, and iteratively refine responses. A final reasoning mechanism synthesises the contributions of these agents into a coherent output that either reflects the dominant emotion or integrates multiple perspectives. The architecture incorporates both textual and visual large language models (LLMs), alongside advanced reasoning and self-refinement processes. A functional prototype was deployed locally in an offline environment, optimised for emotional expressiveness and computational efficiency. From this initial prototype, another one emerged, called Armando, which was developed for use in emergency contexts, delivering emotionally calibrated and factually accurate information through the integration of Retrieval-Augmented Generation (RAG) and cumulative context tracking. The Project Riley prototype was evaluated through user testing, in which participants interacted with the chatbot and completed a structured questionnaire assessing three dimensions: Emotional Appropriateness, Clarity and Utility, and Naturalness and Human-likeness. The results indicate strong performance in structured scenarios, particularly with respect to emotional alignment and communicative clarity.

**arXiv ID:** 2505.20521
</details>

<details>
<summary><strong>ProfilingAgent: Profiling-Guided Agentic Reasoning for Adaptive Model Optimization</strong> - Sadegh Jafari, Aishwarya Sarkar, Mohiuddin Bilwal, Ali Jannesari - [[pdf]](https://arxiv.org/pdf/2509.05584)</summary>

**Abstract:** Foundation models face growing compute and memory bottlenecks, hindering deployment on resource-limited platforms. While compression techniques such as pruning and quantization are widely used, most rely on uniform heuristics that ignore architectural and runtime heterogeneity. Profiling tools expose per-layer latency, memory, and compute cost, yet are rarely integrated into automated pipelines. We propose ProfilingAgent, a profiling-guided, agentic approach that uses large language models (LLMs) to automate compression via structured pruning and post-training dynamic quantization. Our modular multi-agent system reasons over static metrics (MACs, parameter counts) and dynamic signals (latency, memory) to design architecture-specific strategies. Unlike heuristic baselines, ProfilingAgent tailors layer-wise decisions to bottlenecks. Experiments on ImageNet-1K, CIFAR-10, and CIFAR-100 with ResNet-101, ViT-B/16, Swin-B, and DeiT-B/16 show pruning maintains competitive or improved accuracy (about 1% drop on ImageNet-1K, +2% gains for ViT-B/16 on smaller datasets), while quantization achieves up to 74% memory savings with <0.5% accuracy loss. Our quantization also yields consistent inference speedups of up to 1.74 times faster. Comparative studies with GPT-4o and GPT-4-Turbo highlight the importance of LLM reasoning quality for iterative pruning. These results establish agentic systems as scalable solutions for profiling-guided model optimization.

**arXiv ID:** 2509.05584
</details>

<details>
<summary><strong>A Composable Agentic System for Automated Visual Data Reporting</strong> - Péter Ferenc Gyarmati, Dominik Moritz, Torsten Möller, Laura Koesten - [[pdf]](https://arxiv.org/pdf/2509.05721)</summary>

**Abstract:** To address the brittleness of monolithic AI agents, our prototype for automated visual data reporting explores a Human-AI Partnership model. Its hybrid, multi-agent architecture strategically externalizes logic from LLMs to deterministic modules, leveraging the rule-based system Draco for principled visualization design. The system delivers a dual-output: an interactive Observable report with Mosaic for reader exploration, and executable Marimo notebooks for deep, analyst-facing traceability. This granular architecture yields a fully automatic yet auditable and steerable system, charting a path toward a more synergistic partnership between human experts and AI. For reproducibility, our implementation and examples are available at this https URL.

**arXiv ID:** 2509.05721
</details>

<details>
<summary><strong>Role-Playing LLM-Based Multi-Agent Support Framework for Detecting and Addressing Family Communication Bias</strong> - Rushia Harada, Yuken Kimura, Keito Inoshita - [[pdf]](https://arxiv.org/pdf/2507.11210)</summary>

**Abstract:** Well-being in family settings involves subtle psychological dynamics that conventional metrics often overlook. In particular, unconscious parental expectations, termed ideal parent bias, can suppress children's emotional expression and autonomy. This suppression, referred to as suppressed emotion, often stems from well-meaning but value-driven communication, which is difficult to detect or address from outside the family. Focusing on these latent dynamics, this study explores Large Language Model (LLM)-based support for psychologically safe family communication. We constructed a Japanese parent-child dialogue corpus of 30 scenarios, each annotated with metadata on ideal parent bias and suppressed emotion. Based on this corpus, we developed a Role-Playing LLM-based multi-agent dialogue support framework that analyzes dialogue and generates feedback. Specialized agents detect suppressed emotion, describe implicit ideal parent bias in parental speech, and infer contextual attributes such as the child's age and background. A meta-agent compiles these outputs into a structured report, which is then passed to five selected expert agents. These agents collaboratively generate empathetic and actionable feedback through a structured four-step discussion process. Experiments show that the system can detect categories of suppressed emotion with moderate accuracy and produce feedback rated highly in empathy and practicality. Moreover, simulated follow-up dialogues incorporating this feedback exhibited signs of improved emotional expression and mutual understanding, suggesting the framework's potential in supporting positive transformation in family interactions.

**arXiv ID:** 2507.11210
</details>

<details>
<summary><strong>TalkToAgent: A Human-centric Explanation of Reinforcement Learning Agents with Large Language Models</strong> - Haechang Kim, Hao Chen, Can Li, Jong Min Lee - [[pdf]](https://arxiv.org/pdf/2509.04809)</summary>

**Abstract:** Explainable Reinforcement Learning (XRL) has emerged as a promising approach in improving the transparency of Reinforcement Learning (RL) agents. However, there remains a gap between complex RL policies and domain experts, due to the limited comprehensibility of XRL results and isolated coverage of current XRL approaches that leave users uncertain about which tools to employ. To address these challenges, we introduce TalkToAgent, a multi-agent Large Language Models (LLM) framework that delivers interactive, natural language explanations for RL policies. The architecture with five specialized LLM agents (Coordinator, Explainer, Coder, Evaluator, and Debugger) enables TalkToAgent to automatically map user queries to relevant XRL tools and clarify an agent's actions in terms of either key state variables, expected outcomes, or counterfactual explanations. Moreover, our approach extends previous counterfactual explanations by deriving alternative scenarios from qualitative behavioral descriptions, or even new rule-based policies. We validated TalkToAgent on quadruple-tank process control problem, a well-known nonlinear control benchmark. Results demonstrated that TalkToAgent successfully mapped user queries into XRL tasks with high accuracy, and coder-debugger interactions minimized failures in counterfactual generation. Furthermore, qualitative evaluation confirmed that TalkToAgent effectively interpreted agent's actions and contextualized their meaning within the problem domain.

**arXiv ID:** 2509.04809
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>DRF: LLM-AGENT Dynamic Reputation Filtering Framework</strong> - Yuwei Lou, Hao Hu, Shaocong Ma, Zongfei Zhang, Liang Wang, Jidong Ge, Xianping Tao - [[pdf]](https://arxiv.org/pdf/2509.05764)</summary>

**Abstract:** With the evolution of generative AI, multi - agent systems leveraging large - language models(LLMs) have emerged as a powerful tool for complex tasks. However, these systems face challenges in quantifying agent performance and lack mechanisms to assess agent credibility. To address these issues, we introduce DRF, a dynamic reputation filtering framework. DRF constructs an interactive rating network to quantify agent performance, designs a reputation scoring mechanism to measure agent honesty and capability, and integrates an Upper Confidence Bound - based strategy to enhance agent selection efficiency. Experiments show that DRF significantly improves task completion quality and collaboration efficiency in logical reasoning and code - generation tasks, offering a new approach for multi - agent systems to handle large - scale tasks.

**arXiv ID:** 2509.05764
</details>

<details>
<summary><strong>Paper2Agent: Reimagining Research Papers As Interactive and Reliable AI Agents</strong> - Jiacheng Miao, Joe R. Davis, Jonathan K. Pritchard, James Zou - [[pdf]](https://arxiv.org/pdf/2509.06917)</summary>

**Abstract:** We introduce Paper2Agent, an automated framework that converts research papers into AI agents. Paper2Agent transforms research output from passive artifacts into active systems that can accelerate downstream use, adoption, and discovery. Conventional research papers require readers to invest substantial effort to understand and adapt a paper's code, data, and methods to their own work, creating barriers to dissemination and reuse. Paper2Agent addresses this challenge by automatically converting a paper into an AI agent that acts as a knowledgeable research assistant. It systematically analyzes the paper and the associated codebase using multiple agents to construct a Model Context Protocol (MCP) server, then iteratively generates and runs tests to refine and robustify the resulting MCP. These paper MCPs can then be flexibly connected to a chat agent (e.g. Claude Code) to carry out complex scientific queries through natural language while invoking tools and workflows from the original paper. We demonstrate Paper2Agent's effectiveness in creating reliable and capable paper agents through in-depth case studies. Paper2Agent created an agent that leverages AlphaGenome to interpret genomic variants and agents based on ScanPy and TISSUE to carry out single-cell and spatial transcriptomics analyses. We validate that these paper agents can reproduce the original paper's results and can correctly carry out novel user queries. By turning static papers into dynamic, interactive AI agents, Paper2Agent introduces a new paradigm for knowledge dissemination and a foundation for the collaborative ecosystem of AI co-scientists.

**arXiv ID:** 2509.06917
</details>

<details>
<summary><strong>Plantbot: Integrating Plant and Robot through LLM Modular Agent Networks</strong> - Atsushi Masumori, Norihiro Maruyama, Itsuki Doi, johnsmith, Hiroki Sato, Takashi Ikegami - [[pdf]](https://arxiv.org/pdf/2509.05338)</summary>

**Abstract:** We introduce Plantbot, a hybrid lifeform that connects a living plant with a mobile robot through a network of large language model (LLM) modules. Each module - responsible for sensing, vision, dialogue, or action - operates asynchronously and communicates via natural language, enabling seamless interaction across biological and artificial domains. This architecture leverages the capacity of LLMs to serve as hybrid interfaces, where natural language functions as a universal protocol, translating multimodal data (soil moisture, temperature, visual context) into linguistic messages that coordinate system behaviors. The integrated network transforms plant states into robotic actions, installing normativity essential for agency within the sensor-motor loop. By combining biological and robotic elements through LLM-mediated communication, Plantbot behaves as an embodied, adaptive agent capable of responding autonomously to environmental conditions. This approach suggests possibilities for a new model of artificial life, where decentralized, LLM modules coordination enable novel interactions between biological and artificial systems.

**arXiv ID:** 2509.05338
</details>

<details>
<summary><strong>ThreatGPT: An Agentic AI Framework for Enhancing Public Safety through Threat Modeling</strong> - Sharif Noor Zisad, Ragib Hasan - [[pdf]](https://arxiv.org/pdf/2509.05379)</summary>

**Abstract:** As our cities and communities become smarter, the systems that keep us safe, such as traffic control centers, emergency response networks, and public transportation, also become more complex. With this complexity comes a greater risk of security threats that can affect not just machines but real people's lives. To address this challenge, we present ThreatGPT, an agentic Artificial Intelligence (AI) assistant built to help people whether they are engineers, safety officers, or policy makers to understand and analyze threats in public safety systems. Instead of requiring deep cybersecurity expertise, it allows users to simply describe the components of a system they are concerned about, such as login systems, data storage, or communication networks. Then, with the click of a button, users can choose how they want the system to be analyzed by using popular frameworks such as STRIDE, MITRE ATT&CK, CVE reports, NIST, or CISA. ThreatGPT is unique because it does not just provide threat information, but rather it acts like a knowledgeable partner. Using few-shot learning, the AI learns from examples and generates relevant smart threat models. It can highlight what might go wrong, how attackers could take advantage, and what can be done to prevent harm. Whether securing a city's infrastructure or a local health service, this tool adapts to users' needs. In simple terms, ThreatGPT brings together AI and human judgment to make our public systems safer. It is designed not just to analyze threats, but to empower people to understand and act on them, faster, smarter, and with more confidence.

**arXiv ID:** 2509.05379
</details>

<details>
<summary><strong>Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks</strong> - Su Hyeong Lee, Risi Kondor, Richard Ngo - [[pdf]](https://arxiv.org/pdf/2509.06701)</summary>

**Abstract:** We develop a theory of intelligent agency grounded in probabilistic modeling for neural models. Agents are represented as outcome distributions with epistemic utility given by log score, and compositions are defined through weighted logarithmic pooling that strictly improves every member's welfare. We prove that strict unanimity is impossible under linear pooling or in binary outcome spaces, but possible with three or more outcomes. Our framework admits recursive structure via cloning invariance, continuity, and openness, while tilt-based analysis rules out trivial duplication. Finally, we formalize an agentic alignment phenomenon in LLMs using our theory: eliciting a benevolent persona ("Luigi'") induces an antagonistic counterpart ("Waluigi"), while a manifest-then-suppress Waluigi strategy yields strictly larger first-order misalignment reduction than pure Luigi reinforcement alone. These results clarify how developing a principled mathematical framework for how subagents can coalesce into coherent higher-level entities provides novel implications for alignment in agentic AI systems.

**arXiv ID:** 2509.06701
</details>

<details>
<summary><strong>Adaptive Evolution Factor Risk Ellipse Framework for Reliable and Safe Autonomous Driving</strong> - Fujiang Yuan, Zhen Tian, Yangfan He, Guojian Zou, Chunhong Yuan, Yanhong Peng, Zhihao Lin - [[pdf]](https://arxiv.org/pdf/2509.06375)</summary>

**Abstract:** In recent years, ensuring safety, efficiency, and comfort in interactive autonomous driving has become a critical challenge. Traditional model-based techniques, such as game-theoretic methods and robust control, are often overly conservative or computationally intensive. Conversely, learning-based approaches typically require extensive training data and frequently exhibit limited interpretability and generalizability. Simpler strategies, such as Risk Potential Fields (RPF), provide lightweight alternatives with minimal data demands but are inherently static and struggle to adapt effectively to dynamic traffic conditions. To overcome these limitations, we propose the Evolutionary Risk Potential Field (ERPF), a novel approach that dynamically updates risk assessments in dynamical scenarios based on historical obstacle proximity data. We introduce a Risk-Ellipse construct that combines longitudinal reach and lateral uncertainty into a unified spatial temporal collision envelope. Additionally, we define an adaptive Evolution Factor metric, computed through sigmoid normalization of Time to Collision (TTC) and Time-Window-of-Hazard (TWH), which dynamically adjusts the dimensions of the ellipse axes in real time. This adaptive risk metric is integrated seamlessly into a Model Predictive Control (MPC) framework, enabling autonomous vehicles to proactively address complex interactive driving scenarios in terms of uncertain driving of surrounding vehicles. Comprehensive comparative experiments demonstrate that our ERPF-MPC approach consistently achieves smoother trajectories, higher average speeds, and collision-free navigation, offering a robust and adaptive solution suitable for complex interactive driving environments.

**arXiv ID:** 2509.06375
</details>

<details>
<summary><strong>An Adaptive Coverage Control Approach for Multiple Autonomous Off-road Vehicles in Dynamic Agricultural Fields</strong> - Sajad Ahmadi, Mohammadreza Davoodi, Javad Mohammadpour Velni - [[pdf]](https://arxiv.org/pdf/2509.06682)</summary>

**Abstract:** This paper presents an adaptive coverage control method for a fleet of off-road and Unmanned Ground Vehicles (UGVs) operating in dynamic (time-varying) agricultural environments. Traditional coverage control approaches often assume static conditions, making them unsuitable for real-world farming scenarios where obstacles, such as moving machinery and uneven terrains, create continuous challenges. To address this, we propose a real-time path planning framework that integrates Unmanned Aerial Vehicles (UAVs) for obstacle detection and terrain assessment, allowing UGVs to dynamically adjust their coverage paths. The environment is modeled as a weighted directed graph, where the edge weights are continuously updated based on the UAV observations to reflect obstacle motion and terrain variations. The proposed approach incorporates Voronoi-based partitioning, adaptive edge weight assignment, and cost-based path optimization to enhance navigation efficiency. Simulation results demonstrate the effectiveness of the proposed method in improving path planning, reducing traversal costs, and maintaining robust coverage in the presence of dynamic obstacles and muddy terrains.

**arXiv ID:** 2509.06682
</details>

<details>
<summary><strong>Embodied Hazard Mitigation using Vision-Language Models for Autonomous Mobile Robots</strong> - Oluwadamilola Sotomi, Devika Kodi, Kiruthiga Chandra Shekar, Aliasghar Arab - [[pdf]](https://arxiv.org/pdf/2509.06768)</summary>

**Abstract:** Autonomous robots operating in dynamic environments should identify and report anomalies. Embodying proactive mitigation improves safety and operational continuity. This paper presents a multimodal anomaly detection and mitigation system that integrates vision-language models and large language models to identify and report hazardous situations and conflicts in real-time. The proposed system enables robots to perceive, interpret, report, and if possible respond to urban and environmental anomalies through proactive detection mechanisms and automated mitigation actions. A key contribution in this paper is the integration of Hazardous and Conflict states into the robot's decision-making framework, where each anomaly type can trigger specific mitigation strategies. User studies (n = 30) demonstrated the effectiveness of the system in anomaly detection with 91.2% prediction accuracy and relatively low latency response times using edge-ai architecture.

**arXiv ID:** 2509.06768
</details>

<details>
<summary><strong>Dynamic Modeling and Efficient Data-Driven Optimal Control for Micro Autonomous Surface Vehicles</strong> - Zhiheng Chen, Wei Wang - [[pdf]](https://arxiv.org/pdf/2509.06882)</summary>

**Abstract:** Micro Autonomous Surface Vehicles (MicroASVs) offer significant potential for operations in confined or shallow waters and swarm robotics applications. However, achieving precise and robust control at such small scales remains highly challenging, mainly due to the complexity of modeling nonlinear hydrodynamic forces and the increased sensitivity to self-motion effects and environmental disturbances, including waves and boundary effects in confined spaces. This paper presents a physics-driven dynamics model for an over-actuated MicroASV and introduces a data-driven optimal control framework that leverages a weak formulation-based online model learning method. Our approach continuously refines the physics-driven model in real time, enabling adaptive control that adjusts to changing system parameters. Simulation results demonstrate that the proposed method substantially enhances trajectory tracking accuracy and robustness, even under unknown payloads and external disturbances. These findings highlight the potential of data-driven online learning-based optimal control to improve MicroASV performance, paving the way for more reliable and precise autonomous surface vehicle operations.

**arXiv ID:** 2509.06882
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (32 papers)</h2></summary>

<details>
<summary><strong>Proof2Silicon: Prompt Repair for Verified Code and Hardware Generation via Reinforcement Learning</strong> - Manvi Jha, Jiaxin Wan, Deming Chen - [[pdf]](https://arxiv.org/pdf/2509.06239)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated impressive capabilities in automated code generation but frequently produce code that fails formal verification, an essential requirement for hardware and safety-critical domains. To overcome this fundamental limitation, we previously proposed PREFACE, a model-agnostic framework based on reinforcement learning (RL) that iteratively repairs the prompts provided to frozen LLMs, systematically steering them toward generating formally verifiable Dafny code without costly fine-tuning. This work presents Proof2Silicon, a novel end-to-end synthesis framework that embeds the previously proposed PREFACE flow to enable the generation of correctness-by-construction hardware directly from natural language specifications. Proof2Silicon operates by: (1) leveraging PREFACE's verifier-driven RL agent to optimize prompt generation iteratively, ensuring Dafny code correctness; (2) automatically translating verified Dafny programs into synthesizable high-level C using Dafny's Python backend and PyLog; and (3) employing Vivado HLS to produce RTL implementations. Evaluated rigorously on a challenging 100-task benchmark, PREFACE's RL-guided prompt optimization consistently improved Dafny verification success rates across diverse LLMs by up to 21%. Crucially, Proof2Silicon achieved an end-to-end hardware synthesis success rate of up to 72%, generating RTL designs through Vivado HLS synthesis flows. These results demonstrate a robust, scalable, and automated pipeline for LLM-driven, formally verified hardware synthesis, bridging natural-language specification and silicon realization.

**arXiv ID:** 2509.06239
</details>

<details>
<summary><strong>TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning</strong> - Chuang Jiang, Mingyue Cheng, Xiaoyu Tao, Qingyang Mao, Jie Ouyang, Qi Liu - [[pdf]](https://arxiv.org/pdf/2509.06278)</summary>

**Abstract:** Table reasoning is crucial for leveraging structured data in domains such as finance, healthcare, and scientific research. While large language models (LLMs) show promise in multi-step reasoning, purely text-based methods often struggle with the complex numerical computations and fine-grained operations inherently required in this task. Tool-integrated reasoning improves computational accuracy via explicit code execution, yet existing systems frequently rely on rigid patterns, supervised imitation, and lack true autonomous adaptability. In this paper, we present TableMind, an LLM-driven table reasoning agent that (i) autonomously performs multi-turn tool invocation, (ii) writes and executes data-analyzing code in a secure sandbox environment for data analysis and precise numerical reasoning, and (iii) exhibits high-level capabilities such as planning and self-reflection to adapt strategies. To realize these capabilities, we adopt a two-stage fine-tuning paradigm built on top of a powerful pre-trained language model: supervised fine-tuning on high-quality reasoning trajectories to establish effective tool usage patterns, followed by reinforcement fine-tuning to optimize multi-objective strategies. In particular, we propose Rank-Aware Policy Optimization (RAPO), which increases the update weight of high-quality trajectories when their output probabilities are lower than those of low-quality ones, thereby guiding the model more consistently toward better and more accurate answers. Extensive experiments on several mainstream benchmarks demonstrate that TableMind achieves superior performance compared to competitive baselines, yielding substantial gains in both reasoning accuracy and computational precision.

**arXiv ID:** 2509.06278
</details>

<details>
<summary><strong>MORSE: Multi-Objective Reinforcement Learning via Strategy Evolution for Supply Chain Optimization</strong> - Niki Kotecha, Ehecatl Antonio del Rio Chanona - [[pdf]](https://arxiv.org/pdf/2509.06490)</summary>

**Abstract:** In supply chain management, decision-making often involves balancing multiple conflicting objectives, such as cost reduction, service level improvement, and environmental sustainability. Traditional multi-objective optimization methods, such as linear programming and evolutionary algorithms, struggle to adapt in real-time to the dynamic nature of supply chains. In this paper, we propose an approach that combines Reinforcement Learning (RL) and Multi-Objective Evolutionary Algorithms (MOEAs) to address these challenges for dynamic multi-objective optimization under uncertainty. Our method leverages MOEAs to search the parameter space of policy neural networks, generating a Pareto front of policies. This provides decision-makers with a diverse population of policies that can be dynamically switched based on the current system objectives, ensuring flexibility and adaptability in real-time decision-making. We also introduce Conditional Value-at-Risk (CVaR) to incorporate risk-sensitive decision-making, enhancing resilience in uncertain environments. We demonstrate the effectiveness of our approach through case studies, showcasing its ability to respond to supply chain dynamics and outperforming state-of-the-art methods in an inventory management case study. The proposed strategy not only improves decision-making efficiency but also offers a more robust framework for managing uncertainty and optimizing performance in supply chains.

**arXiv ID:** 2509.06490
</details>

<details>
<summary><strong>Reinforcement Learning Foundations for Deep Research Systems: A Survey</strong> - Wenjun Li, Zhi Chen, Jingru Lin, Hannan Cao, Wei Han, Sheng Liang, Zhi Zhang, Kuicai Dong, Dexun Li, Chen Zhang, Yong Liu - [[pdf]](https://arxiv.org/pdf/2509.06733)</summary>

**Abstract:** Deep research systems, agentic AI that solve complex, multi-step tasks by coordinating reasoning, search across the open web and user files, and tool use, are moving toward hierarchical deployments with a Planner, Coordinator, and Executors. In practice, training entire stacks end-to-end remains impractical, so most work trains a single planner connected to core tools such as search, browsing, and code. While SFT imparts protocol fidelity, it suffers from imitation and exposure biases and underuses environment feedback. Preference alignment methods such as DPO are schema and proxy-dependent, off-policy, and weak for long-horizon credit assignment and multi-objective trade-offs. A further limitation of SFT and DPO is their reliance on human defined decision points and subskills through schema design and labeled comparisons. Reinforcement learning aligns with closed-loop, tool-interaction research by optimizing trajectory-level policies, enabling exploration, recovery behaviors, and principled credit assignment, and it reduces dependence on such human priors and rater biases.
This survey is, to our knowledge, the first dedicated to the RL foundations of deep research systems. It systematizes work after DeepSeek-R1 along three axes: (i) data synthesis and curation; (ii) RL methods for agentic research covering stability, sample efficiency, long context handling, reward and credit design, multi-objective optimization, and multimodal integration; and (iii) agentic RL training systems and frameworks. We also cover agent architecture and coordination, as well as evaluation and benchmarks, including recent QA, VQA, long-form synthesis, and domain-grounded, tool-interaction tasks. We distill recurring patterns, surface infrastructure bottlenecks, and offer practical guidance for training robust, transparent deep research agents with RL.

**arXiv ID:** 2509.06733
</details>

<details>
<summary><strong>LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba</strong> - Yinuo Wang, Gavin Tao - [[pdf]](https://arxiv.org/pdf/2508.11849)</summary>

**Abstract:** We introduce LocoMamba, a vision-driven cross-modal DRL framework built on selective state-space models, specifically leveraging Mamba, that achieves near-linear-time sequence modeling, effectively captures long-range dependencies, and enables efficient training with longer sequences. First, we embed proprioceptive states with a multilayer perceptron and patchify depth images with a lightweight convolutional neural network, producing compact tokens that improve state representation. Second, stacked Mamba layers fuse these tokens via near-linear-time selective scanning, reducing latency and memory footprint, remaining robust to token length and image resolution, and providing an inductive bias that mitigates overfitting. Third, we train the policy end-to-end with Proximal Policy Optimization under terrain and appearance randomization and an obstacle-density curriculum, using a compact state-centric reward that balances progress, smoothness, and safety. We evaluate our method in challenging simulated environments with static and moving obstacles as well as uneven terrain. Compared with state-of-the-art baselines, our method achieves higher returns and success rates with fewer collisions, exhibits stronger generalization to unseen terrains and obstacle densities, and improves training efficiency by converging in fewer updates under the same compute budget.

**arXiv ID:** 2508.11849
</details>

<details>
<summary><strong>Livia: An Emotion-Aware AR Companion Powered by Modular AI Agents and Progressive Memory Compression</strong> - Rui Xi, Xianghan Wang - [[pdf]](https://arxiv.org/pdf/2509.05298)</summary>

**Abstract:** Loneliness and social isolation pose significant emotional and health challenges, prompting the development of technology-based solutions for companionship and emotional support. This paper introduces Livia, an emotion-aware augmented reality (AR) companion app designed to provide personalized emotional support by combining modular artificial intelligence (AI) agents, multimodal affective computing, progressive memory compression, and AR driven embodied interaction. Livia employs a modular AI architecture with specialized agents responsible for emotion analysis, dialogue generation, memory management, and behavioral orchestration, ensuring robust and adaptive interactions. Two novel algorithms-Temporal Binary Compression (TBC) and Dynamic Importance Memory Filter (DIMF)-effectively manage and prioritize long-term memory, significantly reducing storage requirements while retaining critical context. Our multimodal emotion detection approach achieves high accuracy, enhancing proactive and empathetic engagement. User evaluations demonstrated increased emotional bonds, improved satisfaction, and statistically significant reductions in loneliness. Users particularly valued Livia's adaptive personality evolution and realistic AR embodiment. Future research directions include expanding gesture and tactile interactions, supporting multi-user experiences, and exploring customized hardware implementations.

**arXiv ID:** 2509.05298
</details>

<details>
<summary><strong>Large Language Model Integration with Reinforcement Learning to Augment Decision-Making in Autonomous Cyber Operations</strong> - Konur Tholl, François Rivest, Mariam El Mezouar, Ranwa Al Mallah - [[pdf]](https://arxiv.org/pdf/2509.05311)</summary>

**Abstract:** Reinforcement Learning (RL) has shown great potential for autonomous decision-making in the cybersecurity domain, enabling agents to learn through direct environment interaction. However, RL agents in Autonomous Cyber Operations (ACO) typically learn from scratch, requiring them to execute undesirable actions to learn their consequences. In this study, we integrate external knowledge in the form of a Large Language Model (LLM) pretrained on cybersecurity data that our RL agent can directly leverage to make informed decisions. By guiding initial training with an LLM, we improve baseline performance and reduce the need for exploratory actions with obviously negative outcomes. We evaluate our LLM-integrated approach in a simulated cybersecurity environment, and demonstrate that our guided agent achieves over 2x higher rewards during early training and converges to a favorable policy approximately 4,500 episodes faster than the baseline.

**arXiv ID:** 2509.05311
</details>

<details>
<summary><strong>Integrated Simulation Framework for Adversarial Attacks on Autonomous Vehicles</strong> - Christos Anagnostopoulos, Ioulia Kapsali, Alexandros Gkillas, Nikos Piperigkos, Aris S. Lalos - [[pdf]](https://arxiv.org/pdf/2509.05332)</summary>

**Abstract:** Autonomous vehicles (AVs) rely on complex perception and communication systems, making them vulnerable to adversarial attacks that can compromise safety. While simulation offers a scalable and safe environment for robustness testing, existing frameworks typically lack comprehensive supportfor modeling multi-domain adversarial scenarios. This paper introduces a novel, open-source integrated simulation framework designed to generate adversarial attacks targeting both perception and communication layers of AVs. The framework provides high-fidelity modeling of physical environments, traffic dynamics, and V2X networking, orchestrating these components through a unified core that synchronizes multiple simulators based on a single configuration file. Our implementation supports diverse perception-level attacks on LiDAR sensor data, along with communication-level threats such as V2X message manipulation and GPS spoofing. Furthermore, ROS 2 integration ensures seamless compatibility with third-party AV software stacks. We demonstrate the framework's effectiveness by evaluating the impact of generated adversarial scenarios on a state-of-the-art 3D object detector, revealing significant performance degradation under realistic conditions.

**arXiv ID:** 2509.05332
</details>

<details>
<summary><strong>Learning to Construct Knowledge through Sparse Reference Selection with Reinforcement Learning</strong> - Shao-An Yin - [[pdf]](https://arxiv.org/pdf/2509.05874)</summary>

**Abstract:** The rapid expansion of scientific literature makes it increasingly difficult to acquire new knowledge, particularly in specialized domains where reasoning is complex, full-text access is restricted, and target references are sparse among a large set of candidates. We present a Deep Reinforcement Learning framework for sparse reference selection that emulates human knowledge construction, prioritizing which papers to read under limited time and cost. Evaluated on drug--gene relation discovery with access restricted to titles and abstracts, our approach demonstrates that both humans and machines can construct knowledge effectively from partial information.

**arXiv ID:** 2509.05874
</details>

<details>
<summary><strong>Teaching Precommitted Agents: Model-Free Policy Evaluation and Control in Quasi-Hyperbolic Discounted MDPs</strong> - S.R. Eshwar - [[pdf]](https://arxiv.org/pdf/2509.06094)</summary>

**Abstract:** Time-inconsistent preferences, where agents favor smaller-sooner over larger-later rewards, are a key feature of human and animal decision-making. Quasi-Hyperbolic (QH) discounting provides a simple yet powerful model for this behavior, but its integration into the reinforcement learning (RL) framework has been limited. This paper addresses key theoretical and algorithmic gaps for precommitted agents with QH preferences. We make two primary contributions: (i) we formally characterize the structure of the optimal policy, proving for the first time that it reduces to a simple one-step non-stationary form; and (ii) we design the first practical, model-free algorithms for both policy evaluation and Q-learning in this setting, both with provable convergence guarantees. Our results provide foundational insights for incorporating QH preferences in RL.

**arXiv ID:** 2509.06094
</details>

<details>
<summary><strong>Toward a Metrology for Artificial Intelligence: Hidden-Rule Environments and Reinforcement Learning</strong> - Christo Mathew, Wentian Wang, Lazaros Gallos, Paul Kantor, Vladimir Menkov, Hao Wang - [[pdf]](https://arxiv.org/pdf/2509.06213)</summary>

**Abstract:** We investigate reinforcement learning in the Game Of Hidden Rules (GOHR) environment, a complex puzzle in which an agent must infer and execute hidden rules to clear a 6$\times$6 board by placing game pieces into buckets. We explore two state representation strategies, namely Feature-Centric (FC) and Object-Centric (OC), and employ a Transformer-based Advantage Actor-Critic (A2C) algorithm for training. The agent has access only to partial observations and must simultaneously infer the governing rule and learn the optimal policy through experience. We evaluate our models across multiple rule-based and trial-list-based experimental setups, analyzing transfer effects and the impact of representation on learning efficiency.

**arXiv ID:** 2509.06213
</details>

<details>
<summary><strong>Agentic Software Engineering: Foundational Pillars and a Research Roadmap</strong> - Ahmed E. Hassan, Hao Li, Dayi Lin, Bram Adams, Tse-Hsun Chen, Yutaro Kashiwa, Dong Qiu - [[pdf]](https://arxiv.org/pdf/2509.06216)</summary>

**Abstract:** Agentic Software Engineering (SE 3.0) represents a new era where intelligent agents are tasked not with simple code generation, but with achieving complex, goal-oriented SE objectives. To harness these new capabilities while ensuring trustworthiness, we must recognize a fundamental duality within the SE field in the Agentic SE era, comprising two symbiotic modalities: SE for Humans and SE for Agents. This duality demands a radical reimagining of the foundational pillars of SE (actors, processes, tools, and artifacts) which manifest differently across each modality. We propose two purpose-built workbenches to support this vision. The Agent Command Environment (ACE) serves as a command center where humans orchestrate and mentor agent teams, handling outputs such as Merge-Readiness Packs (MRPs) and Consultation Request Packs (CRPs). The Agent Execution Environment (AEE) is a digital workspace where agents perform tasks while invoking human expertise when facing ambiguity or complex trade-offs. This bi-directional partnership, which supports agent-initiated human callbacks and handovers, gives rise to new, structured engineering activities (i.e., processes) that redefine human-AI collaboration, elevating the practice from agentic coding to true agentic software engineering. This paper presents the Structured Agentic Software Engineering (SASE) vision, outlining several of the foundational pillars for the future of SE. The paper culminates in a research roadmap that identifies a few key challenges and opportunities while briefly discussing the resulting impact of this future on SE education. Our goal is not to offer a definitive solution, but to provide a conceptual scaffold with structured vocabulary to catalyze a community-wide dialogue, pushing the SE community to think beyond its classic, human-centric tenets toward a disciplined, scalable, and trustworthy agentic future.

**arXiv ID:** 2509.06216
</details>

<details>
<summary><strong>Aligning Large Vision-Language Models by Deep Reinforcement Learning and Direct Preference Optimization</strong> - Thanh Thi Nguyen, Campbell Wilson, Janis Dalins - [[pdf]](https://arxiv.org/pdf/2509.06759)</summary>

**Abstract:** Large Vision-Language Models (LVLMs) or multimodal large language models represent a significant advancement in artificial intelligence, enabling systems to understand and generate content across both visual and textual modalities. While large-scale pretraining has driven substantial progress, fine-tuning these models for aligning with human values or engaging in specific tasks or behaviors remains a critical challenge. Deep Reinforcement Learning (DRL) and Direct Preference Optimization (DPO) offer promising frameworks for this aligning process. While DRL enables models to optimize actions using reward signals instead of relying solely on supervised preference data, DPO directly aligns the policy with preferences, eliminating the need for an explicit reward model. This overview explores paradigms for fine-tuning LVLMs, highlighting how DRL and DPO techniques can be used to align models with human preferences and values, improve task performance, and enable adaptive multimodal interaction. We categorize key approaches, examine sources of preference data, reward signals, and discuss open challenges such as scalability, sample efficiency, continual learning, generalization, and safety. The goal is to provide a clear understanding of how DRL and DPO contribute to the evolution of robust and human-aligned LVLMs.

**arXiv ID:** 2509.06759
</details>

<details>
<summary><strong>Reinforcement learning meets bioprocess control through behaviour cloning: Real-world deployment in an industrial photobioreactor</strong> - Juan D. Gil, Ehecatl Antonio Del Rio Chanona, José L. Guzmán, Manuel Berenguel - [[pdf]](https://arxiv.org/pdf/2509.06853)</summary>

**Abstract:** The inherent complexity of living cells as production units creates major challenges for maintaining stable and optimal bioprocess conditions, especially in open Photobioreactors (PBRs) exposed to fluctuating environments. To address this, we propose a Reinforcement Learning (RL) control approach, combined with Behavior Cloning (BC), for pH regulation in open PBR systems. This represents, to the best of our knowledge, the first application of an RL-based control strategy to such a nonlinear and disturbance-prone bioprocess. Our method begins with an offline training stage in which the RL agent learns from trajectories generated by a nominal Proportional-Integral-Derivative (PID) controller, without direct interaction with the real system. This is followed by a daily online fine-tuning phase, enabling adaptation to evolving process dynamics and stronger rejection of fast, transient disturbances. This hybrid offline-online strategy allows deployment of an adaptive control policy capable of handling the inherent nonlinearities and external perturbations in open PBRs. Simulation studies highlight the advantages of our method: the Integral of Absolute Error (IAE) was reduced by 8% compared to PID control and by 5% relative to standard off-policy RL. Moreover, control effort decreased substantially-by 54% compared to PID and 7% compared to standard RL-an important factor for minimizing operational costs. Finally, an 8-day experimental validation under varying environmental conditions confirmed the robustness and reliability of the proposed approach. Overall, this work demonstrates the potential of RL-based methods for bioprocess control and paves the way for their broader application to other nonlinear, disturbance-prone systems.

**arXiv ID:** 2509.06853
</details>

<details>
<summary><strong>Categorical semantics of compositional reinforcement learning</strong> - Georgios Bakirtzis, Michail Savvas, Ufuk Topcu - [[pdf]](https://arxiv.org/pdf/2208.13687)</summary>

**Abstract:** Compositional knowledge representations in reinforcement learning (RL) facilitate modular, interpretable, and safe task specifications. However, generating compositional models requires the characterization of minimal assumptions for the robustness of the compositionality feature, especially in the case of functional decompositions. Using a categorical point of view, we develop a knowledge representation framework for a compositional theory of RL. Our approach relies on the theoretical study of the category MDP, whose objects are Markov decision processes (MDPs) acting as models of tasks. The categorical semantics models the compositionality of tasks through the application of pushout operations akin to combining puzzle pieces. As a practical application of these pushout operations, we introduce zig-zag diagrams that rely on the compositional guarantees engendered by the category MDP. We further prove that properties of the category MDP unify concepts, such as enforcing safety requirements and exploiting symmetries, generalizing previous abstraction theories for RL.

**arXiv ID:** 2208.13687
</details>

<details>
<summary><strong>ResearchArena: Benchmarking Large Language Models' Ability to Collect and Organize Information as Research Agents</strong> - Hao Kang, Chenyan Xiong - [[pdf]](https://arxiv.org/pdf/2406.10291)</summary>

**Abstract:** Large language models (LLMs) excel across many natural language processing tasks but face challenges in domain-specific, analytical tasks such as conducting research surveys. This study introduces ResearchArena, a benchmark designed to evaluate LLMs' capabilities in conducting academic surveys -- a foundational step in academic research. ResearchArena models the process in three stages: (1) information discovery, identifying relevant literature; (2) information selection, evaluating papers' relevance and impact; and (3) information organization, structuring knowledge into hierarchical frameworks such as mind-maps. Notably, mind-map construction is treated as a bonus task, reflecting its supplementary role in survey-writing. To support these evaluations, we construct an offline environment of 12M full-text academic papers and 7.9K survey papers. To ensure ethical compliance, we do not redistribute copyrighted materials; instead, we provide code to construct the environment from the Semantic Scholar Open Research Corpus (S2ORC). Preliminary evaluations reveal that LLM-based approaches underperform compared to simpler keyword-based retrieval methods, though recent reasoning models such as DeepSeek-R1 show slightly better zero-shot performance. These results underscore significant opportunities for advancing LLMs in autonomous research. We open-source the code to construct the ResearchArena benchmark at this https URL.

**arXiv ID:** 2406.10291
</details>

<details>
<summary><strong>WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents</strong> - Junteng Liu, Yunji Li, Chi Zhang, Jingyang Li, Aili Chen, Ke Ji, Weiyu Cheng, Zijia Wu, Chengyu Du, Qidi Xu, Jiayuan Song, Zhengmao Zhu, Wenhu Chen, Pengyu Zhao, Junxian He - [[pdf]](https://arxiv.org/pdf/2509.06501)</summary>

**Abstract:** The paradigm of Large Language Models (LLMs) has increasingly shifted toward agentic applications, where web browsing capabilities are fundamental for retrieving information from diverse online sources. However, existing open-source web agents either demonstrate limited information-seeking abilities on complex tasks or lack transparent implementations. In this work, we identify that the key challenge lies in the scarcity of challenging data for information seeking. To address this limitation, we introduce WebExplorer: a systematic data generation approach using model-based exploration and iterative, long-to-short query evolution. This method creates challenging query-answer pairs that require multi-step reasoning and complex web navigation. By leveraging our curated high-quality dataset, we successfully develop advanced web agent WebExplorer-8B through supervised fine-tuning followed by reinforcement learning. Our model supports 128K context length and up to 100 tool calling turns, enabling long-horizon problem solving. Across diverse information-seeking benchmarks, WebExplorer-8B achieves the state-of-the-art performance at its scale. Notably, as an 8B-sized model, WebExplorer-8B is able to effectively search over an average of 16 turns after RL training, achieving higher accuracy than WebSailor-72B on BrowseComp-en/zh and attaining the best performance among models up to 100B parameters on WebWalkerQA and FRAMES. Beyond these information-seeking tasks, our model also achieves strong generalization on the HLE benchmark even though it is only trained on knowledge-intensive QA data. These results highlight our approach as a practical path toward long-horizon web agents.

**arXiv ID:** 2509.06501
</details>

<details>
<summary><strong>Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning</strong> - Liang Chen, Xueting Han, Li Shen, Jing Bai, Kam-Fai Wong - [[pdf]](https://arxiv.org/pdf/2509.06948)</summary>

**Abstract:** Reinforcement learning (RL) has proven effective in incentivizing the reasoning abilities of large language models (LLMs), but suffers from severe efficiency challenges due to its trial-and-error nature. While the common practice employs supervised fine-tuning (SFT) as a warm-up stage for RL, this decoupled two-stage approach limits interaction between SFT and RL, thereby constraining overall effectiveness. This study introduces a novel method for learning reasoning models that employs bilevel optimization to facilitate better cooperation between these training paradigms. By conditioning the SFT objective on the optimal RL policy, our approach enables SFT to meta-learn how to guide RL's optimization process. During training, the lower level performs RL updates while simultaneously receiving SFT supervision, and the upper level explicitly maximizes the cooperative gain-the performance advantage of joint SFT-RL training over RL alone. Empirical evaluations on five reasoning benchmarks demonstrate that our method consistently outperforms baselines and achieves a better balance between effectiveness and efficiency.

**arXiv ID:** 2509.06948
</details>

<details>
<summary><strong>Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models</strong> - Yinjie Wang, Ling Yang, Bowen Li, Ye Tian, Ke Shen, Mengdi Wang - [[pdf]](https://arxiv.org/pdf/2509.06949)</summary>

**Abstract:** We propose TraceRL, a trajectory-aware reinforcement learning framework for diffusion language models (DLMs) that incorporates preferred inference trajectory into post-training, and is applicable across different architectures. Equipped with a diffusion-based value model that enhances training stability, we demonstrate improved reasoning performance on complex math and coding tasks. Besides, it can also be applied to adapt block-specific models to larger blocks, which improves sampling flexibility. Employing TraceRL, we derive a series of state-of-the-art diffusion language models, namely TraDo. Although smaller than 7B-scale AR models, TraDo-4B-Instruct still consistently outperforms them across complex math reasoning tasks. TraDo-8B-Instruct achieves relative accuracy improvements of 6.1% over Qwen2.5-7B-Instruct and 51.3% over Llama3.1-8B-Instruct on mathematical reasoning benchmarks. Through curriculum learning, we also derive the first long-CoT DLM, outperforming Qwen2.5-7B-Instruct on MATH500 with an 18.1% relative accuracy gain. To facilitate reproducible research and practical applications, we release a comprehensive open-source framework for building, training, and deploying diffusion LLMs across diverse architectures. The framework integrates accelerated KV-cache techniques and inference engines for both inference and reinforcement learning, and includes implementations of various supervised fine-tuning and RL methods for mathematics, coding, and general tasks. Code and Models: this https URL

**arXiv ID:** 2509.06949
</details>

<details>
<summary><strong>HierTOD: A Task-Oriented Dialogue System Driven by Hierarchical Goals</strong> - Lingbo Mo, Shun Jiang, Akash Maharaj, Bernard Hishamunda, Yunyao Li - [[pdf]](https://arxiv.org/pdf/2411.07152)</summary>

**Abstract:** Task-Oriented Dialogue (TOD) systems assist users in completing tasks through natural language interactions, often relying on a single-layered workflow structure for slot-filling in public tasks, such as hotel bookings. However, in enterprise environments, which involve rich domain-specific knowledge, TOD systems face challenges due to task complexity and the lack of standardized documentation. In this work, we introduce HierTOD, an enterprise TOD system driven by hierarchical goals that can support composite workflows. By focusing on goal-driven interactions, our system serves a more proactive role, facilitating mixed-initiative dialogue and improving task completion. Equipped with components for natural language understanding, composite goal retriever, dialogue management, and response generation, backed by a well-organized data service with domain knowledge base and retrieval engine, HierTOD delivers efficient task assistance as judged by human evaluators. Furthermore, our system implementation unifies two TOD paradigms: slot-filling for information collection and step-by-step guidance for task execution. Our user study demonstrates the effectiveness and helpfulness of HierTOD in performing both paradigms.

**arXiv ID:** 2411.07152
</details>

<details>
<summary><strong>Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning</strong> - Haozhe Wang, Qixin Xu, Che Liu, Junhong Wu, Fangzhen Lin, Wenhu Chen - [[pdf]](https://arxiv.org/pdf/2509.03646)</summary>

**Abstract:** Reinforcement Learning (RL) has proven highly effective at enhancing the complex reasoning abilities of Large Language Models (LLMs), yet underlying mechanisms driving this success remain largely opaque. Our analysis reveals that puzzling phenomena like ``aha moments", ``length-scaling'' and entropy dynamics are not disparate occurrences but hallmarks of an emergent reasoning hierarchy, akin to the separation of high-level strategic planning from low-level procedural execution in human cognition. We uncover a compelling two-phase dynamic: initially, a model is constrained by procedural correctness and must improve its low-level skills. The learning bottleneck then decisively shifts, with performance gains being driven by the exploration and mastery of high-level strategic planning. This insight exposes a core inefficiency in prevailing RL algorithms like GRPO, which apply optimization pressure agnostically and dilute the learning signal across all tokens. To address this, we propose HIerarchy-Aware Credit Assignment (HICRA), an algorithm that concentrates optimization efforts on high-impact planning tokens. HICRA significantly outperforms strong baselines, demonstrating that focusing on this strategic bottleneck is key to unlocking advanced reasoning. Furthermore, we validate semantic entropy as a superior compass for measuring strategic exploration over misleading metrics such as token-level entropy.

**arXiv ID:** 2509.03646
</details>

<details>
<summary><strong>Reinforcement Learning with Anticipation: A Hierarchical Approach for Long-Horizon Tasks</strong> - Yang Yu - [[pdf]](https://arxiv.org/pdf/2509.05545)</summary>

**Abstract:** Solving long-horizon goal-conditioned tasks remains a significant challenge in reinforcement learning (RL). Hierarchical reinforcement learning (HRL) addresses this by decomposing tasks into more manageable sub-tasks, but the automatic discovery of the hierarchy and the joint training of multi-level policies often suffer from instability and can lack theoretical guarantees. In this paper, we introduce Reinforcement Learning with Anticipation (RLA), a principled and potentially scalable framework designed to address these limitations. The RLA agent learns two synergistic models: a low-level, goal-conditioned policy that learns to reach specified subgoals, and a high-level anticipation model that functions as a planner, proposing intermediate subgoals on the optimal path to a final goal. The key feature of RLA is the training of the anticipation model, which is guided by a principle of value geometric consistency, regularized to prevent degenerate solutions. We present proofs that RLA approaches the globally optimal policy under various conditions, establishing a principled and convergent method for hierarchical planning and execution in long-horizon goal-conditioned tasks.

**arXiv ID:** 2509.05545
</details>

<details>
<summary><strong>Physics-informed Value Learner for Offline Goal-Conditioned Reinforcement Learning</strong> - Vittorio Giammarino, Ruiqi Ni, Ahmed H. Qureshi - [[pdf]](https://arxiv.org/pdf/2509.06782)</summary>

**Abstract:** Offline Goal-Conditioned Reinforcement Learning (GCRL) holds great promise for domains such as autonomous navigation and locomotion, where collecting interactive data is costly and unsafe. However, it remains challenging in practice due to the need to learn from datasets with limited coverage of the state-action space and to generalize across long-horizon tasks. To improve on these challenges, we propose a Physics-informed (Pi) regularized loss for value learning, derived from the Eikonal Partial Differential Equation (PDE) and which induces a geometric inductive bias in the learned value function. Unlike generic gradient penalties that are primarily used to stabilize training, our formulation is grounded in continuous-time optimal control and encourages value functions to align with cost-to-go structures. The proposed regularizer is broadly compatible with temporal-difference-based value learning and can be integrated into existing Offline GCRL algorithms. When combined with Hierarchical Implicit Q-Learning (HIQL), the resulting method, Physics-informed HIQL (Pi-HIQL), yields significant improvements in both performance and generalization, with pronounced gains in stitching regimes and large-scale navigation tasks.

**arXiv ID:** 2509.06782
</details>

<details>
<summary><strong>RoboBallet: Planning for Multi-Robot Reaching with Graph Neural Networks and Reinforcement Learning</strong> - Matthew Lai, Keegan Go, Zhibin Li, Torsten Kroger, Stefan Schaal, Kelsey Allen, Jonathan Scholz - [[pdf]](https://arxiv.org/pdf/2509.05397)</summary>

**Abstract:** Modern robotic manufacturing requires collision-free coordination of multiple robots to complete numerous tasks in shared, obstacle-rich workspaces. Although individual tasks may be simple in isolation, automated joint task allocation, scheduling, and motion planning under spatio-temporal constraints remain computationally intractable for classical methods at real-world scales. Existing multi-arm systems deployed in the industry rely on human intuition and experience to design feasible trajectories manually in a labor-intensive process. To address this challenge, we propose a reinforcement learning (RL) framework to achieve automated task and motion planning, tested in an obstacle-rich environment with eight robots performing 40 reaching tasks in a shared workspace, where any robot can perform any task in any order. Our approach builds on a graph neural network (GNN) policy trained via RL on procedurally-generated environments with diverse obstacle layouts, robot configurations, and task distributions. It employs a graph representation of scenes and a graph policy neural network trained through reinforcement learning to generate trajectories of multiple robots, jointly solving the sub-problems of task allocation, scheduling, and motion planning. Trained on large randomly generated task sets in simulation, our policy generalizes zero-shot to unseen settings with varying robot placements, obstacle geometries, and task poses. We further demonstrate that the high-speed capability of our solution enables its use in workcell layout optimization, improving solution times. The speed and scalability of our planner also open the door to new capabilities such as fault-tolerant planning and online perception-based re-planning, where rapid adaptation to dynamic task sets is required.

**arXiv ID:** 2509.05397
</details>

<details>
<summary><strong>Using Reinforcement Learning to Optimize the Global and Local Crossing Number</strong> - Timo Brand, Henry Förster, Stephen Kobourov, Robin Schukrafft, Markus Wallinger, Johannes Zink - [[pdf]](https://arxiv.org/pdf/2509.06108)</summary>

**Abstract:** We present a novel approach to graph drawing based on reinforcement learning for minimizing the global and the local crossing number, that is, the total number of edge crossings and the maximum number of crossings on any edge, respectively. In our framework, an agent learns how to move a vertex based on a given observation vector in order to optimize its position. The agent receives feedback in the form of local reward signals tied to crossing reduction. To generate an initial layout, we use a stress-based graph-drawing algorithm. We compare our method against force- and stress-based (baseline) algorithms as well as three established algorithms for global crossing minimization on a suite of benchmark graphs. The experiments show mixed results: our current algorithm is mainly competitive for the local crossing number. We see a potential for further development of the approach in the future.

**arXiv ID:** 2509.06108
</details>

<details>
<summary><strong>Reward function compression facilitates goal-dependent reinforcement learning</strong> - Gaia Molinaro, Anne G. E. Collins - [[pdf]](https://arxiv.org/pdf/2509.06810)</summary>

**Abstract:** Reinforcement learning agents learn from rewards, but humans can uniquely assign value to novel, abstract outcomes in a goal-dependent manner. However, this flexibility is cognitively costly, making learning less efficient. Here, we propose that goal-dependent learning is initially supported by a capacity-limited working memory system. With consistent experience, learners create a "compressed" reward function (a simplified rule defining the goal) which is then transferred to long-term memory and applied automatically upon receiving feedback. This process frees up working memory resources, boosting learning efficiency. We test this theory across six experiments. Consistent with our predictions, our findings demonstrate that learning is parametrically impaired by the size of the goal space, but improves when the goal space structure allows for compression. We also find faster reward processing to correlate with better learning performance, supporting the idea that as goal valuation becomes more automatic, more resources are available for learning. We leverage computational modeling to support this interpretation. Our work suggests that efficient goal-directed learning relies on compressing complex goal information into a stable reward function, shedding light on the cognitive mechanisms of human motivation. These findings generate new insights into the neuroscience of intrinsic motivation and could help improve behavioral techniques that support people in achieving their goals.

**arXiv ID:** 2509.06810
</details>

<details>
<summary><strong>The Curious Price of Distributional Robustness in Reinforcement Learning with a Generative Model</strong> - Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, Matthieu Geist, Yuejie Chi - [[pdf]](https://arxiv.org/pdf/2305.16589)</summary>

**Abstract:** This paper investigates model robustness in reinforcement learning (RL) to reduce the sim-to-real gap in practice. We adopt the framework of distributionally robust Markov decision processes (RMDPs), aimed at learning a policy that optimizes the worst-case performance when the deployed environment falls within a prescribed uncertainty set around the nominal MDP. Despite recent efforts, the sample complexity of RMDPs remained mostly unsettled regardless of the uncertainty set in use. It was unclear if distributional robustness bears any statistical consequences when benchmarked against standard RL. Assuming access to a generative model that draws samples based on the nominal MDP, we provide a near-optimal characterization of the sample complexity of RMDPs when the uncertainty set is specified via either the total variation (TV) distance or chi-squared divergence. The algorithm studied here is a model-based method called distributionally robust value iteration, which is shown to be near-optimal for the full range of uncertainty levels. Somewhat surprisingly, our results uncover that RMDPs are not necessarily easier or harder to learn than standard MDPs. The statistical consequence incurred by the robustness requirement depends heavily on the size and shape of the uncertainty set: in the case w.r.t.~the TV distance, the minimax sample complexity of RMDPs is always smaller than that of standard MDPs; in the case w.r.t.~the chi-squared divergence, the sample complexity of RMDPs far exceeds the standard MDP counterpart.

**arXiv ID:** 2305.16589
</details>

<details>
<summary><strong>CAREL: Instruction-guided reinforcement learning with cross-modal auxiliary objectives</strong> - Armin Saghafian, Amirmohammad Izadi, Negin Hashemi Dijujin, Mahdieh Soleymani Baghshah - [[pdf]](https://arxiv.org/pdf/2411.19787)</summary>

**Abstract:** Grounding the instruction in the environment is a key step in solving language-guided goal-reaching reinforcement learning problems. In automated reinforcement learning, a key concern is to enhance the model's ability to generalize across various tasks and environments. In goal-reaching scenarios, the agent must comprehend the different parts of the instructions within the environmental context in order to complete the overall task successfully. In this work, we propose CAREL (Cross-modal Auxiliary REinforcement Learning) as a new framework to solve this problem using auxiliary loss functions inspired by video-text retrieval literature and a novel method called instruction tracking, which automatically keeps track of progress in an environment. The results of our experiments suggest superior sample efficiency and systematic generalization for this framework in multi-modal reinforcement learning problems. Our code base is available here.

**arXiv ID:** 2411.19787
</details>

<details>
<summary><strong>Interactive Shaping of Granular Media Using Reinforcement Learning</strong> - Benedikt Kreis, Malte Mosbach, Anny Ripke, Muhammad Ehsan Ullah, Sven Behnke, Maren Bennewitz - [[pdf]](https://arxiv.org/pdf/2509.06469)</summary>

**Abstract:** Autonomous manipulation of granular media, such as sand, is crucial for applications in construction, excavation, and additive manufacturing. However, shaping granular materials presents unique challenges due to their high-dimensional configuration space and complex dynamics, where traditional rule-based approaches struggle without extensive engineering efforts. Reinforcement learning (RL) offers a promising alternative by enabling agents to learn adaptive manipulation strategies through trial and error. In this work, we present an RL framework that enables a robotic arm with a cubic end-effector and a stereo camera to shape granular media into desired target structures. We show the importance of compact observations and concise reward formulations for the large configuration space, validating our design choices with an ablation study. Our results demonstrate the effectiveness of the proposed approach for the training of visual policies that manipulate granular media including their real-world deployment, outperforming two baseline approaches.

**arXiv ID:** 2509.06469
</details>

<details>
<summary><strong>Ask1: Development and Reinforcement Learning-Based Control of a Custom Quadruped Robot</strong> - Yang Zhang, Yuxing Lu, Guiyang Xin, Yufei Xue, Chenkun Qi, Kairong Qin, Yan Zhuang - [[pdf]](https://arxiv.org/pdf/2412.08019)</summary>

**Abstract:** In this work, we present the design, development, and experimental validation of a custom-built quadruped robot, Ask1. The Ask1 robot shares similar morphology with the Unitree Go1, but features custom hardware components and a different control architecture. We transfer and extend previous reinforcement learning (RL)-based control methods to the Ask1 robot, demonstrating the applicability of our approach in real-world scenarios. By eliminating the need for Adversarial Motion Priors (AMP) and reference trajectories, we introduce a novel reward function to guide the robot's motion style. We demonstrate the generalization capability of the proposed RL algorithm by training it on both the Go1 and Ask1 robots. Simulation and real-world experiments validate the effectiveness of this method, showing that Ask1, like the Go1, is capable of navigating various rugged terrains.

**arXiv ID:** 2412.08019
</details>

<details>
<summary><strong>QuadKAN: KAN-Enhanced Quadruped Motion Control via End-to-End Reinforcement Learning</strong> - Yinuo Wang, Gavin Tao - [[pdf]](https://arxiv.org/pdf/2508.19153)</summary>

**Abstract:** We address vision-guided quadruped motion control with reinforcement learning (RL) and highlight the necessity of combining proprioception with vision for robust control. We propose QuadKAN, a spline-parameterized cross-modal policy instantiated with Kolmogorov-Arnold Networks (KANs). The framework incorporates a spline encoder for proprioception and a spline fusion head for proprioception-vision inputs. This structured function class aligns the state-to-action mapping with the piecewise-smooth nature of gait, improving sample efficiency, reducing action jitter and energy consumption, and providing interpretable posture-action sensitivities. We adopt Multi-Modal Delay Randomization (MMDR) and perform end-to-end training with Proximal Policy Optimization (PPO). Evaluations across diverse terrains, including both even and uneven surfaces and scenarios with static or dynamic obstacles, demonstrate that QuadKAN achieves consistently higher returns, greater distances, and fewer collisions than state-of-the-art (SOTA) baselines. These results show that spline-parameterized policies offer a simple, effective, and interpretable alternative for robust vision-guided locomotion. A repository will be made available upon acceptance.

**arXiv ID:** 2508.19153
</details>

<details>
<summary><strong>GestoBrush: Facilitating Graffiti Artists' Digital Creation Experiences through Embodied AR Interactions</strong> - Ruiqi Chen, Qingyang He, Hanxi Bao, Jung Choi, Xin Tong - [[pdf]](https://arxiv.org/pdf/2509.05619)</summary>

**Abstract:** Graffiti has long documented the socio-cultural landscapes of urban spaces, yet increasing global regulations have constrained artists' creative freedom, prompting exploration of digital alternatives. Augmented Reality (AR) offers opportunities to extend graffiti into digital environments while retaining spatial and cultural significance, but prior research has largely centered on audience engagement rather than the embodied creative processes of graffiti artists. To address this, we developed GestoBrush, a mobile AR prototype that turns smartphones into virtual spray cans, enabling graffiti creation through embodied gestures. A co-design workshop underscored the role of embodiment-physical engagement with surroundings and body-driven creative processes-in digital workflows. We evaluated GestoBrush with six graffiti artists and findings suggested that embodied AR interactions supporting artists bypass real-world constraints and explore new artistic possibilities, whose AR artworks created enhanced senses of intuitiveness, immersion, and expressiveness. This work highlight how embodied AR tools can bridge the gap between physical graffiti practice and digital expression, suggesting pathways for designing immersive creative systems that respect the cultural ethos of street art while expanding its possibilities in virtual spaces.

**arXiv ID:** 2509.05619
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
