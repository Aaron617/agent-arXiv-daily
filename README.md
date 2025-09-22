# Agent arXiv Daily

**Last Updated:** 2025-09-22 02:46:26

**Total Papers:** 66

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (4 papers)</h2></summary>

<details>
<summary><strong>Collective Voice: Recovered-Peer Support Mediated by An LLM-Based Chatbot for Eating Disorder Recovery</strong> - Ryuhaerang Choi, Taehan Kim, Subin Park, Seohyeon Yoo, Jennifer G. Kim, Sung-Ju Lee - [[pdf]](https://arxiv.org/pdf/2509.15289)</summary>

**Abstract:** Peer recovery narratives provide unique benefits beyond professional or lay mentoring by fostering hope and sustained recovery in eating disorder (ED) contexts. Yet, such support is limited by the scarcity of peer-involved programs and potential drawbacks on recovered peers, including relapse risk. To address this, we designed RecoveryTeller, a chatbot adopting a recovered-peer persona that portrays itself as someone recovered from an ED. We examined whether such a persona can reproduce the support affordances of peer recovery narratives. We compared RecoveryTeller with a lay-mentor persona chatbot offering similar guidance but without a recovery background. We conducted a 20-day cross-over deployment study with 26 ED participants, each using both chatbots for 10 days. RecoveryTeller elicited stronger emotional resonance than a lay-mentor chatbot, yet tensions between emotional and epistemic trust led participants to view the two personas as complementary rather than substitutes. We provide design implications for mental health chatbot persona design.

**arXiv ID:** 2509.15289
</details>

<details>
<summary><strong>On the Security of Tool-Invocation Prompts for LLM-Based Agentic Systems: An Empirical Risk Assessment</strong> - Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Yu Liu, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She - [[pdf]](https://arxiv.org/pdf/2509.05755)</summary>

**Abstract:** LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.

**arXiv ID:** 2509.05755
</details>

<details>
<summary><strong>Designing with Culture: How Social Norms Shape Trust and Preference in Health Chatbots</strong> - Arpita Wadhwa, Aditya Vashistha, Mohit Jain - [[pdf]](https://arxiv.org/pdf/2509.15575)</summary>

**Abstract:** AI-driven chatbots are increasingly used to support community health workers (CHWs) in developing regions, yet little is known about how cultural framings in chatbot design shape trust in collectivist contexts where decisions are rarely made in isolation. This paper examines how CHWs in rural India responded to chatbots that delivered identical health content but varied in one specific cultural lever -- social norms. Through a mixed-methods study with 61 ASHAs who compared four normative framings -- neutral, descriptive, narrative identity, and injunctive authority -- we (1) analyze how framings influence preferences and trust, and (2) compare effects across low- and high-ambiguity scenarios. Results show that narrative framings were most preferred but encouraged uncritical overreliance, while authority framings were least preferred yet supported calibrated trust. We conclude with design recommendations for dynamic framing strategies that adapt to context and argue for calibrated trust -- following correct advice and resisting incorrect advice -- as a critical evaluation metric for safe, culturally-grounded AI.

**arXiv ID:** 2509.15575
</details>

<details>
<summary><strong>Campus AI vs. Commercial AI: How Customizations Shape Trust and Usage of LLM as-a-Service Chatbots</strong> - Leon Hannig, Annika Bush, Meltem Aksoy, Steffen Becker, Greta Ontrup - [[pdf]](https://arxiv.org/pdf/2509.15826)</summary>

**Abstract:** As the use of LLM chatbots by students and researchers becomes more prevalent, universities are pressed to develop AI strategies. One strategy that many universities pursue is to customize pre-trained LLM as-a-service (LLMaaS). While most studies on LLMaaS chatbots prioritize technical adaptations, we focus on psychological effects of user-salient customizations, such as interface changes. We assume that such customizations influence users' perception of the system and are therefore important in guiding safe and appropriate use. In a field study, we examine how students and employees (N = 526) at a German university perceive and use their institution's customized LLMaaS chatbot compared to ChatGPT. Participants using both systems (n = 116) reported greater trust, higher perceived privacy and less experienced hallucinations with their university's customized LLMaaS chatbot in contrast to ChatGPT. We discuss theoretical implications for research on calibrated trust, and offer guidance on the design and deployment of LLMaaS chatbots.

**arXiv ID:** 2509.15826
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (5 papers)</h2></summary>

<details>
<summary><strong>DebFlow: Automating Agent Creation via Agent Debate</strong> - Jinwei Su, Yinghui Xia, Yiqun Duan, Jun Du, Jianuo Huang, Tianyu Shi, Lewei He - [[pdf]](https://arxiv.org/pdf/2503.23781)</summary>

**Abstract:** Large language models (LLMs) have demonstrated strong potential and impressive performance in automating the generation and optimization of workflows. However, existing approaches are marked by limited reasoning capabilities, high computational demands, and significant resource requirements. To address these issues, we propose DebFlow, a framework that employs a debate mechanism to optimize workflows and integrates reflexion to improve based on previous experiences. We evaluated our method across six benchmark datasets, including HotpotQA, MATH, and ALFWorld. Our approach achieved a 3\% average performance improvement over the latest baselines, demonstrating its effectiveness in diverse problem domains. In particular, during training, our framework reduces resource consumption by 37\% compared to the state-of-the-art baselines. Additionally, we performed ablation studies. Removing the Debate component resulted in a 4\% performance drop across two benchmark datasets, significantly greater than the 2\% drop observed when the Reflection component was removed. These findings strongly demonstrate the critical role of Debate in enhancing framework performance, while also highlighting the auxiliary contribution of reflexion to overall optimization.

**arXiv ID:** 2503.23781
</details>

<details>
<summary><strong>CoPAD : Multi-source Trajectory Fusion and Cooperative Trajectory Prediction with Anchor-oriented Decoder in V2X Scenarios</strong> - Kangyu Wu, Jiaqi Qiao, Ya Zhang - [[pdf]](https://arxiv.org/pdf/2509.15984)</summary>

**Abstract:** Recently, data-driven trajectory prediction methods have achieved remarkable results, significantly advancing the development of autonomous driving. However, the instability of single-vehicle perception introduces certain limitations to trajectory prediction. In this paper, a novel lightweight framework for cooperative trajectory prediction, CoPAD, is proposed. This framework incorporates a fusion module based on the Hungarian algorithm and Kalman filtering, along with the Past Time Attention (PTA) module, mode attention module and anchor-oriented decoder (AoD). It effectively performs early fusion on multi-source trajectory data from vehicles and road infrastructure, enabling the trajectories with high completeness and accuracy. The PTA module can efficiently capture potential interaction information among historical trajectories, and the mode attention module is proposed to enrich the diversity of predictions. Additionally, the decoder based on sparse anchors is designed to generate the final complete trajectories. Extensive experiments show that CoPAD achieves the state-of-the-art performance on the DAIR-V2X-Seq dataset, validating the effectiveness of the model in cooperative trajectory prediction in V2X scenarios.

**arXiv ID:** 2509.15984
</details>

<details>
<summary><strong>Online Slip Detection and Friction Coefficient Estimation for Autonomous Racing</strong> - Christopher Oeltjen, Carson Sobolewski, Saleh Faghfoorian, Lorant Domokos, Giancarlo Vidal, Ivan Ruchkin - [[pdf]](https://arxiv.org/pdf/2509.15423)</summary>

**Abstract:** Accurate knowledge of the tire-road friction coefficient (TRFC) is essential for vehicle safety, stability, and performance, especially in autonomous racing, where vehicles often operate at the friction limit. However, TRFC cannot be directly measured with standard sensors, and existing estimation methods either depend on vehicle or tire models with uncertain parameters or require large training datasets. In this paper, we present a lightweight approach for online slip detection and TRFC estimation. Our approach relies solely on IMU and LiDAR measurements and the control actions, without special dynamical or tire models, parameter identification, or training data. Slip events are detected in real time by comparing commanded and measured motions, and the TRFC is then estimated directly from observed accelerations under no-slip conditions. Experiments with a 1:10-scale autonomous racing car across different friction levels demonstrate that the proposed approach achieves accurate and consistent slip detections and friction coefficients, with results closely matching ground-truth measurements. These findings highlight the potential of our simple, deployable, and computationally efficient approach for real-time slip monitoring and friction coefficient estimation in autonomous driving.

**arXiv ID:** 2509.15423
</details>

<details>
<summary><strong>CoReVLA: A Dual-Stage End-to-End Autonomous Driving Framework for Long-Tail Scenarios via Collect-and-Refine</strong> - Shiyu Fang, Yiming Cui, Haoyang Liang, Chen Lv, Peng Hang, Jian Sun - [[pdf]](https://arxiv.org/pdf/2509.15968)</summary>

**Abstract:** Autonomous Driving (AD) systems have made notable progress, but their performance in long-tail, safety-critical scenarios remains limited. These rare cases contribute a disproportionate number of accidents. Vision-Language Action (VLA) models have strong reasoning abilities and offer a potential solution, but their effectiveness is limited by the lack of high-quality data and inefficient learning in such conditions. To address these challenges, we propose CoReVLA, a continual learning end-to-end autonomous driving framework that improves the performance in long-tail scenarios through a dual-stage process of data Collection and behavior Refinement. First, the model is jointly fine-tuned on a mixture of open-source driving QA datasets, allowing it to acquire a foundational understanding of driving scenarios. Next, CoReVLA is deployed within the Cave Automatic Virtual Environment (CAVE) simulation platform, where driver takeover data is collected from real-time interactions. Each takeover indicates a long-tail scenario that CoReVLA fails to handle reliably. Finally, the model is refined via Direct Preference Optimization (DPO), allowing it to learn directly from human preferences and thereby avoid reward hacking caused by manually designed rewards. Extensive open-loop and closed-loop experiments demonstrate that the proposed CoReVLA model can accurately perceive driving scenarios and make appropriate decisions. On the Bench2Drive benchmark, CoReVLA achieves a Driving Score (DS) of 72.18 and a Success Rate (SR) of 50%, outperforming state-of-the-art methods by 7.96 DS and 15% SR under long-tail, safety-critical scenarios. Furthermore, case studies demonstrate the model's ability to continually improve its performance in similar failure-prone scenarios by leveraging past takeover experiences. All codea and preprocessed datasets are available at: this https URL

**arXiv ID:** 2509.15968
</details>

<details>
<summary><strong>FlightDiffusion: Revolutionising Autonomous Drone Training with Diffusion Models Generating FPV Video</strong> - Valerii Serpiva, Artem Lykov, Faryal Batool, Vladislav Kozlovskiy, Miguel Altamirano Cabrera, Dzmitry Tsetserukou - [[pdf]](https://arxiv.org/pdf/2509.14082)</summary>

**Abstract:** We present FlightDiffusion, a diffusion-model-based framework for training autonomous drones from first-person view (FPV) video. Our model generates realistic video sequences from a single frame, enriched with corresponding action spaces to enable reasoning-driven navigation in dynamic environments. Beyond direct policy learning, FlightDiffusion leverages its generative capabilities to synthesize diverse FPV trajectories and state-action pairs, facilitating the creation of large-scale training datasets without the high cost of real-world data collection. Our evaluation demonstrates that the generated trajectories are physically plausible and executable, with a mean position error of 0.25 m (RMSE 0.28 m) and a mean orientation error of 0.19 rad (RMSE 0.24 rad). This approach enables improved policy learning and dataset scalability, leading to superior performance in downstream navigation tasks. Results in simulated environments highlight enhanced robustness, smoother trajectory planning, and adaptability to unseen conditions. An ANOVA revealed no statistically significant difference between performance in simulation and reality (F(1, 16) = 0.394, p = 0.541), with success rates of M = 0.628 (SD = 0.162) and M = 0.617 (SD = 0.177), respectively, indicating strong sim-to-real transfer. The generated datasets provide a valuable resource for future UAV research. This work introduces diffusion-based reasoning as a promising paradigm for unifying navigation, action generation, and data synthesis in aerial robotics.

**arXiv ID:** 2509.14082
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>MicroRCA-Agent: Microservice Root Cause Analysis Method Based on Large Language Model Agents</strong> - Pan Tang, Shixiang Tang, Huanqi Pu, Zhiqing Miao, Zhixing Wang - [[pdf]](https://arxiv.org/pdf/2509.15635)</summary>

**Abstract:** This paper presents MicroRCA-Agent, an innovative solution for microservice root cause analysis based on large language model agents, which constructs an intelligent fault root cause localization system with multimodal data fusion. The technical innovations are embodied in three key aspects: First, we combine the pre-trained Drain log parsing algorithm with multi-level data filtering mechanism to efficiently compress massive logs into high-quality fault features. Second, we employ a dual anomaly detection approach that integrates Isolation Forest unsupervised learning algorithms with status code validation to achieve comprehensive trace anomaly identification. Third, we design a statistical symmetry ratio filtering mechanism coupled with a two-stage LLM analysis strategy to enable full-stack phenomenon summarization across node-service-pod hierarchies. The multimodal root cause analysis module leverages carefully designed cross-modal prompts to deeply integrate multimodal anomaly information, fully exploiting the cross-modal understanding and logical reasoning capabilities of large language models to generate structured analysis results encompassing fault components, root cause descriptions, and reasoning trace. Comprehensive ablation studies validate the complementary value of each modal data and the effectiveness of the system architecture. The proposed solution demonstrates superior performance in complex microservice fault scenarios, achieving a final score of 50.71. The code has been released at: this https URL.

**arXiv ID:** 2509.15635
</details>

<details>
<summary><strong>Watson: A Cognitive Observability Framework for the Reasoning of LLM-Powered Agents</strong> - Benjamin Rombaut, Sogol Masoumzadeh, Kirill Vasilevski, Dayi Lin, Ahmed E. Hassan - [[pdf]](https://arxiv.org/pdf/2411.03455)</summary>

**Abstract:** Large language models (LLMs) are increasingly integrated into autonomous systems, giving rise to a new class of software known as Agentware, where LLM-powered agents perform complex, open-ended tasks in domains such as software engineering, customer service, and data analysis. However, their high autonomy and opaque reasoning processes pose significant challenges for traditional software observability methods. To address this, we introduce the concept of cognitive observability - the ability to recover and inspect the implicit reasoning behind agent decisions. We present Watson, a general-purpose framework for observing the reasoning processes of fast-thinking LLM agents without altering their behavior. Watson retroactively infers reasoning traces using prompt attribution techniques. We evaluate Watson in both manual debugging and automated correction scenarios across the MMLU benchmark and the AutoCodeRover and OpenHands agents on the SWE-bench-lite dataset. In both static and dynamic settings, Watson surfaces actionable reasoning insights and supports targeted interventions, demonstrating its practical utility for improving transparency and reliability in Agentware systems.

**arXiv ID:** 2411.03455
</details>

<details>
<summary><strong>World Modelling Improves Language Model Agents</strong> - Shangmin Guo, Omar Darwiche Domingues, Raphaël Avalos, Aaron Courville, Florian Strub - [[pdf]](https://arxiv.org/pdf/2506.02918)</summary>

**Abstract:** Tool use in stateful environments presents unique challenges for large language models (LLMs), where existing test-time compute strategies relying on repeated trials in the environment are impractical. We propose dynamics modelling (DyMo), a method that augments LLMs with a state prediction capability alongside function calling during post-training. This enables LLMs to predict the future states of their actions through an internal environment model. On the Berkeley Function Calling Leaderboard V2, DyMo improves success rates and significantly reduces hallucinations. We further integrate the internal environment model into self-verification sampling (SVS), and show that this substantially improves pass^k over number of trials k, and allows the model to refuse unreliable outputs. Together, DyMo and SVS greatly enhance the effectiveness and reliability of LLMs for tool use. We believe this work charts a path towards scalable planning RL methods for LLM inference without repeatedly querying the oracle environment.

**arXiv ID:** 2506.02918
</details>

<details>
<summary><strong>Adaptive Self-improvement LLM Agentic System for ML Library Development</strong> - Genghan Zhang, Weixin Liang, Olivia Hsu, Kunle Olukotun - [[pdf]](https://arxiv.org/pdf/2502.02534)</summary>

**Abstract:** ML libraries, often written in architecture-specific programming languages (ASPLs) that target domain-specific architectures, are key to efficient ML systems. However, writing these high-performance ML libraries is challenging because it requires expert knowledge of ML algorithms and the ASPL. Large language models (LLMs), on the other hand, have shown general coding capabilities. However, challenges remain when using LLMs for generating ML libraries using ASPLs because 1) this task is complicated even for experienced human programmers and 2) there are limited code examples because of the esoteric and evolving nature of ASPLs. Therefore, LLMs need complex reasoning with limited data in order to complete this task. To address these challenges, we introduce an adaptive self-improvement agentic system. In order to evaluate the effectiveness of our system, we construct a benchmark of a typical ML library and generate ASPL code with both open and closed-source LLMs on this benchmark. Our results show improvements of up to $3.9\times$ over a baseline single LLM.

**arXiv ID:** 2502.02534
</details>

<details>
<summary><strong>UXAgent: A System for Simulating Usability Testing of Web Design with LLM Agents</strong> - Yuxuan Lu, Bingsheng Yao, Hansu Gu, Jing Huang, Jessie Wang, Yang Li, Jiri Gesi, Qi He, Toby Jia-Jun Li, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2504.09407)</summary>

**Abstract:** Usability testing is a fundamental research method that user experience (UX) researchers use to evaluate and iterate their new designs. But what about evaluating and iterating the usability testing study design itself? Recent advances in Large Language Model-simulated Agent (LLM Agent) research inspired us to design UXAgent to support UX researchers in evaluating and iterating their study design before they conduct the real human-subject study. Our system features a Persona Generator module, an LLM Agent module, and a Universal Browser Connector module to automatically generate thousands of simulated users and to interactively test the target website. The system also provides a Result Viewer Interface so that the UX researchers can easily review and analyze the generated qualitative (e.g., agents' post-study surveys) and quantitative data (e.g., agents' interaction logs), or even interview agents directly. Through a heuristic evaluation with 16 UX researchers, participants praised the innovation of our system but also expressed concerns about the future of LLM Agent usage in UX studies.

**arXiv ID:** 2504.09407
</details>

<details>
<summary><strong>AgentA/B: Automated and Scalable Web A/BTesting with Interactive LLM Agents</strong> - Dakuo Wang, Ting-Yao Hsu, Yuxuan Lu, Hansu Gu, Limeng Cui, Yaochen Xie, William Headean, Bingsheng Yao, Akash Veeragouni, Jiapeng Liu, Sreyashi Nag, Jessie Wang - [[pdf]](https://arxiv.org/pdf/2504.09723)</summary>

**Abstract:** A/B testing experiment is a widely adopted method for evaluating UI/UX design decisions in modern web applications. Yet, traditional A/B testing remains constrained by its dependence on the large-scale and live traffic of human participants, and the long time of waiting for the testing result. Through formative interviews with six experienced industry practitioners, we identified critical bottlenecks in current A/B testing workflows. In response, we present AgentA/B, a novel system that leverages Large Language Model-based autonomous agents (LLM Agents) to automatically simulate user interaction behaviors with real webpages. AgentA/B enables scalable deployment of LLM agents with diverse personas, each capable of navigating the dynamic webpage and interactively executing multi-step interactions like search, clicking, filtering, and purchasing. In a demonstrative controlled experiment, we employ AgentA/B to simulate a between-subject A/B testing with 1,000 LLM agents this http URL, and compare agent behaviors with real human shopping behaviors at a scale. Our findings suggest AgentA/B can emulate human-like behavior patterns.

**arXiv ID:** 2504.09723
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>MICA: Multi-Agent Industrial Coordination Assistant</strong> - Di Wen, Kunyu Peng, Junwei Zheng, Yufan Chen, Yitain Shi, Jiale Wei, Ruiping Liu, Kailun Yang, Rainer Stiefelhagen - [[pdf]](https://arxiv.org/pdf/2509.15237)</summary>

**Abstract:** Industrial workflows demand adaptive and trustworthy assistance that can operate under limited computing, connectivity, and strict privacy constraints. In this work, we present MICA (Multi-Agent Industrial Coordination Assistant), a perception-grounded and speech-interactive system that delivers real-time guidance for assembly, troubleshooting, part queries, and maintenance. MICA coordinates five role-specialized language agents, audited by a safety checker, to ensure accurate and compliant support. To achieve robust step understanding, we introduce Adaptive Step Fusion (ASF), which dynamically blends expert reasoning with online adaptation from natural speech feedback. Furthermore, we establish a new multi-agent coordination benchmark across representative task categories and propose evaluation metrics tailored to industrial assistance, enabling systematic comparison of different coordination topologies. Our experiments demonstrate that MICA consistently improves task success, reliability, and responsiveness over baseline structures, while remaining deployable on practical offline hardware. Together, these contributions highlight MICA as a step toward deployable, privacy-preserving multi-agent assistants for dynamic factory environments. The source code will be made publicly available at this https URL.

**arXiv ID:** 2509.15237
</details>

<details>
<summary><strong>Diagnostics of cognitive failures in multi-agent expert systems using dynamic evaluation protocols and subsequent mutation of the processing context</strong> - Andrejs Sorstkins, Josh Bailey, Dr Alistair Baron - [[pdf]](https://arxiv.org/pdf/2509.15366)</summary>

**Abstract:** The rapid evolution of neural architectures - from multilayer perceptrons to large-scale Transformer-based models - has enabled language models (LLMs) to exhibit emergent agentic behaviours when equipped with memory, planning, and external tool use. However, their inherent stochasticity and multi-step decision processes render classical evaluation methods inadequate for diagnosing agentic performance. This work introduces a diagnostic framework for expert systems that not only evaluates but also facilitates the transfer of expert behaviour into LLM-powered agents. The framework integrates (i) curated golden datasets of expert annotations, (ii) silver datasets generated through controlled behavioural mutation, and (iii) an LLM-based Agent Judge that scores and prescribes targeted improvements. These prescriptions are embedded into a vectorized recommendation map, allowing expert interventions to propagate as reusable improvement trajectories across multiple system instances. We demonstrate the framework on a multi-agent recruiter-assistant system, showing that it uncovers latent cognitive failures - such as biased phrasing, extraction drift, and tool misrouting - while simultaneously steering agents toward expert-level reasoning and style. The results establish a foundation for standardized, reproducible expert behaviour transfer in stochastic, tool-augmented LLM agents, moving beyond static evaluation to active expert system refinement.

**arXiv ID:** 2509.15366
</details>

<details>
<summary><strong>Building Data-Driven Occupation Taxonomies: A Bottom-Up Multi-Stage Approach via Semantic Clustering and Multi-Agent Collaboration</strong> - Nan Li, Bo Kang, Tijl De Bie - [[pdf]](https://arxiv.org/pdf/2509.15786)</summary>

**Abstract:** Creating robust occupation taxonomies, vital for applications ranging from job recommendation to labor market intelligence, is challenging. Manual curation is slow, while existing automated methods are either not adaptive to dynamic regional markets (top-down) or struggle to build coherent hierarchies from noisy data (bottom-up). We introduce CLIMB (CLusterIng-based Multi-agent taxonomy Builder), a framework that fully automates the creation of high-quality, data-driven taxonomies from raw job postings. CLIMB uses global semantic clustering to distill core occupations, then employs a reflection-based multi-agent system to iteratively build a coherent hierarchy. On three diverse, real-world datasets, we show that CLIMB produces taxonomies that are more coherent and scalable than existing methods and successfully capture unique regional characteristics. We release our code and datasets at this https URL.

**arXiv ID:** 2509.15786
</details>

<details>
<summary><strong>Explainable AI-Enhanced Supervisory Control for Robust Multi-Agent Robotic Systems</strong> - Reza Pirayeshshirazinezhad, Nima Fathi - [[pdf]](https://arxiv.org/pdf/2509.15491)</summary>

**Abstract:** We present an explainable AI-enhanced supervisory control framework for multi-agent robotics that combines (i) a timed-automata supervisor for safe, auditable mode switching, (ii) robust continuous control (Lyapunov-based controller for large-angle maneuver; sliding-mode controller (SMC) with boundary layers for precision and disturbance rejection), and (iii) an explainable predictor that maps mission context to gains and expected performance (energy, error). Monte Carlo-driven optimization provides the training data, enabling transparent real-time trade-offs.
We validated the approach in two contrasting domains, spacecraft formation flying and autonomous underwater vehicles (AUVs). Despite different environments (gravity/actuator bias vs. hydrodynamic drag/currents), both share uncertain six degrees of freedom (6-DOF) rigid-body dynamics, relative motion, and tight tracking needs, making them representative of general robotic systems. In the space mission, the supervisory logic selects parameters that meet mission criteria. In AUV leader-follower tests, the same SMC structure maintains a fixed offset under stochastic currents with bounded steady error. In spacecraft validation, the SMC controller achieved submillimeter alignment with 21.7% lower tracking error and 81.4% lower energy consumption compared to Proportional-Derivative PD controller baselines. At the same time, in AUV tests, SMC maintained bounded errors under stochastic currents. These results highlight both the portability and the interpretability of the approach for safety-critical, resource-constrained multi-agent robotics.

**arXiv ID:** 2509.15491
</details>

<details>
<summary><strong>Hierarchical Reinforcement Learning with Low-Level MPC for Multi-Agent Control</strong> - Max Studt, Georg Schildbach - [[pdf]](https://arxiv.org/pdf/2509.15799)</summary>

**Abstract:** Achieving safe and coordinated behavior in dynamic, constraint-rich environments remains a major challenge for learning-based control. Pure end-to-end learning often suffers from poor sample efficiency and limited reliability, while model-based methods depend on predefined references and struggle to generalize. We propose a hierarchical framework that combines tactical decision-making via reinforcement learning (RL) with low-level execution through Model Predictive Control (MPC). For the case of multi-agent systems this means that high-level policies select abstract targets from structured regions of interest (ROIs), while MPC ensures dynamically feasible and safe motion. Tested on a predator-prey benchmark, our approach outperforms end-to-end and shielding-based RL baselines in terms of reward, safety, and consistency, underscoring the benefits of combining structured learning with model-based control.

**arXiv ID:** 2509.15799
</details>

<details>
<summary><strong>The Anatomy of a Personal Health Agent</strong> - A. Ali Heydari, Ken Gu, Vidya Srinivas, Hong Yu, Zhihan Zhang, Yuwei Zhang, Akshay Paruchuri, Qian He, Hamid Palangi, Nova Hammerquist, Ahmed A. Metwally, Brent Winslow, Yubin Kim, Kumar Ayush, Yuzhe Yang, Girish Narayanswamy, Maxwell A. Xu, Jake Garrison, Amy Armento Lee, Jenny Vafeiadou, Ben Graef, Isaac R. Galatzer-Levy, Erik Schenck, Andrew Barakat, Javier Perez, Jacqueline Shreibati, John Hernandez, Anthony Z. Faranesh, Javier L. Prieto, Connor Heneghan, Yun Liu, Jiening Zhan, Mark Malhotra, Shwetak Patel, Tim Althoff, Xin Liu, Daniel McDuff, Xuhai "Orson" Xu - [[pdf]](https://arxiv.org/pdf/2508.20148)</summary>

**Abstract:** Health is a fundamental pillar of human wellness, and the rapid advancements in large language models (LLMs) have driven the development of a new generation of health agents. However, the application of health agents to fulfill the diverse needs of individuals in daily non-clinical settings is underexplored. In this work, we aim to build a comprehensive personal health agent that is able to reason about multimodal data from everyday consumer wellness devices and common personal health records, and provide personalized health recommendations. To understand end-users' needs when interacting with such an assistant, we conducted an in-depth analysis of web search and health forum queries, alongside qualitative insights from users and health experts gathered through a user-centered design process. Based on these findings, we identified three major categories of consumer health needs, each of which is supported by a specialist sub-agent: (1) a data science agent that analyzes personal time-series wearable and health record data, (2) a health domain expert agent that integrates users' health and contextual data to generate accurate, personalized insights, and (3) a health coach agent that synthesizes data insights, guiding users using a specified psychological strategy and tracking users' progress. Furthermore, we propose and develop the Personal Health Agent (PHA), a multi-agent framework that enables dynamic, personalized interactions to address individual health needs. To evaluate each sub-agent and the multi-agent system, we conducted automated and human evaluations across 10 benchmark tasks, involving more than 7,000 annotations and 1,100 hours of effort from health experts and end-users. Our work represents the most comprehensive evaluation of a health agent to date and establishes a strong foundation towards the futuristic vision of a personal health agent accessible to everyone.

**arXiv ID:** 2508.20148
</details>

<details>
<summary><strong>Dynamic Agent Grouping ECBS: Scaling Windowed Multi-Agent Path Finding with Completeness Guarantees</strong> - Tiannan Zhang, Rishi Veerapaneni, Shao-Hung Chan, Jiaoyang Li, Maxim Likhachev - [[pdf]](https://arxiv.org/pdf/2509.15381)</summary>

**Abstract:** Multi-Agent Path Finding (MAPF) is the problem of finding a set of collision-free paths for a team of agents. Although several MAPF methods which solve full-horizon MAPF have completeness guarantees, very few MAPF methods that plan partial paths have completeness guarantees. Recent work introduced the Windowed Complete MAPF (WinC-MAPF) framework, which shows how windowed optimal MAPF solvers (e.g., SS-CBS) can use heuristic updates and disjoint agent groups to maintain completeness even when planning partial paths (Veerapaneni et al. 2024). A core limitation of WinC-MAPF is that they required optimal MAPF solvers. Our main contribution is to extend WinC-MAPF by showing how we can use a bounded suboptimal solver while maintaining completeness. In particular, we design Dynamic Agent Grouping ECBS (DAG-ECBS) which dynamically creates and plans agent groups while maintaining that each agent group solution is bounded suboptimal. We prove how DAG-ECBS can maintain completeness in the WinC-MAPF framework. DAG-ECBS shows improved scalability compared to SS-CBS and can outperform windowed ECBS without completeness guarantees. More broadly, our work serves as a blueprint for designing more MAPF methods that can use the WinC-MAPF framework.

**arXiv ID:** 2509.15381
</details>

<details>
<summary><strong>Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning</strong> - Simin Li, Zheng Yuwei, Zihao Mao, Linhao Wang, Ruixiao Xu, Chengdong Ma, Xin Yu, Yuqing Ma, Qi Dou, Xin Wang, Jie Luo, Bo An, Yaodong Yang, Weifeng Lv, Xianglong Liu - [[pdf]](https://arxiv.org/pdf/2509.15103)</summary>

**Abstract:** Partial agent failure becomes inevitable when systems scale up, making it crucial to identify the subset of agents whose compromise would most severely degrade overall performance. In this paper, we study this Vulnerable Agent Identification (VAI) problem in large-scale multi-agent reinforcement learning (MARL). We frame VAI as a Hierarchical Adversarial Decentralized Mean Field Control (HAD-MFC), where the upper level involves an NP-hard combinatorial task of selecting the most vulnerable agents, and the lower level learns worst-case adversarial policies for these agents using mean-field MARL. The two problems are coupled together, making HAD-MFC difficult to solve. To solve this, we first decouple the hierarchical process by Fenchel-Rockafellar transform, resulting a regularized mean-field Bellman operator for upper level that enables independent learning at each level, thus reducing computational complexity. We then reformulate the upper-level combinatorial problem as a MDP with dense rewards from our regularized mean-field Bellman operator, enabling us to sequentially identify the most vulnerable agents by greedy and RL algorithms. This decomposition provably preserves the optimal solution of the original HAD-MFC. Experiments show our method effectively identifies more vulnerable agents in large-scale MARL and the rule-based system, fooling system into worse failures, and learns a value function that reveals the vulnerability of each agent.

**arXiv ID:** 2509.15103
</details>

<details>
<summary><strong>LLM Agents at the Roundtable: A Multi-Perspective and Dialectical Reasoning Framework for Essay Scoring</strong> - Jinhee Jang, Ayoung Moon, Minkyoung Jung, YoungBin Kim, Seung Jin Lee - [[pdf]](https://arxiv.org/pdf/2509.14834)</summary>

**Abstract:** The emergence of large language models (LLMs) has brought a new paradigm to automated essay scoring (AES), a long-standing and practical application of natural language processing in education. However, achieving human-level multi-perspective understanding and judgment remains a challenge. In this work, we propose Roundtable Essay Scoring (RES), a multi-agent evaluation framework designed to perform precise and human-aligned scoring under a zero-shot setting. RES constructs evaluator agents based on LLMs, each tailored to a specific prompt and topic context. Each agent independently generates a trait-based rubric and conducts a multi-perspective evaluation. Then, by simulating a roundtable-style discussion, RES consolidates individual evaluations through a dialectical reasoning process to produce a final holistic score that more closely aligns with human evaluation. By enabling collaboration and consensus among agents with diverse evaluation perspectives, RES outperforms prior zero-shot AES approaches. Experiments on the ASAP dataset using ChatGPT and Claude show that RES achieves up to a 34.86% improvement in average QWK over straightforward prompting (Vanilla) methods.

**arXiv ID:** 2509.14834
</details>

<details>
<summary><strong>MountainLion: A Multi-Modal LLM-Based Agent System for Interpretable and Adaptive Financial Trading</strong> - Siyi Wu, Junqiao Wang, Zhaoyang Guan, Leyi Zhao, Xinyuan Song, Xinyu Ying, Dexu Yu, Jinhao Wang, Hanlin Zhang, Michele Pak, Yangfan He, Yi Xin, Jianhui Wang, Tianyu Shi - [[pdf]](https://arxiv.org/pdf/2507.20474)</summary>

**Abstract:** Cryptocurrency trading is a challenging task requiring the integration of heterogeneous data from multiple modalities. Traditional deep learning and reinforcement learning approaches typically demand large training datasets and encode diverse inputs into numerical representations, often at the cost of interpretability. Recent progress in large language model (LLM)-based agents has demonstrated the capacity to process multi-modal data and support complex investment decision-making. Building on these advances, we present \textbf{MountainLion}, a multi-modal, multi-agent system for financial trading that coordinates specialized LLM-based agents to interpret financial data and generate investment strategies. MountainLion processes textual news, candlestick charts, and trading signal charts to produce high-quality financial reports, while also enabling modification of reports and investment recommendations through data-driven user interaction and question answering. A central reflection module analyzes historical trading signals and outcomes to continuously refine decision processes, and the system is capable of real-time report analysis, summarization, and dynamic adjustment of investment strategies. Empirical results confirm that MountainLion systematically enriches technical price triggers with contextual macroeconomic and capital flow signals, providing a more interpretable, robust, and actionable investment framework that improves returns and strengthens investor confidence.

**arXiv ID:** 2507.20474
</details>

<details>
<summary><strong>Fully Decentralized Cooperative Multi-Agent Reinforcement Learning is A Context Modeling Problem</strong> - Chao Li, Bingkun Bao, Yang Gao - [[pdf]](https://arxiv.org/pdf/2509.15519)</summary>

**Abstract:** This paper studies fully decentralized cooperative multi-agent reinforcement learning, where each agent solely observes the states, its local actions, and the shared rewards. The inability to access other agents' actions often leads to non-stationarity during value function updates and relative overgeneralization during value function estimation, hindering effective cooperative policy learning. However, existing works fail to address both issues simultaneously, due to their inability to model the joint policy of other agents in a fully decentralized setting. To overcome this limitation, we propose a novel method named Dynamics-Aware Context (DAC), which formalizes the task, as locally perceived by each agent, as an Contextual Markov Decision Process, and further addresses both non-stationarity and relative overgeneralization through dynamics-aware context modeling. Specifically, DAC attributes the non-stationary local task dynamics of each agent to switches between unobserved contexts, each corresponding to a distinct joint policy. Then, DAC models the step-wise dynamics distribution using latent variables and refers to them as contexts. For each agent, DAC introduces a context-based value function to address the non-stationarity issue during value function update. For value function estimation, an optimistic marginal value is derived to promote the selection of cooperative actions, thereby addressing the relative overgeneralization issue. Experimentally, we evaluate DAC on various cooperative tasks (including matrix game, predator and prey, and SMAC), and its superior performance against multiple baselines validates its effectiveness.

**arXiv ID:** 2509.15519
</details>

<details>
<summary><strong>Automated Cyber Defense with Generalizable Graph-based Reinforcement Learning Agents</strong> - Isaiah J. King, Benjamin Bowman, H. Howie Huang - [[pdf]](https://arxiv.org/pdf/2509.16151)</summary>

**Abstract:** Deep reinforcement learning (RL) is emerging as a viable strategy for automated cyber defense (ACD). The traditional RL approach represents networks as a list of computers in various states of safety or threat. Unfortunately, these models are forced to overfit to specific network topologies, rendering them ineffective when faced with even small environmental perturbations. In this work, we frame ACD as a two-player context-based partially observable Markov decision problem with observations represented as attributed graphs. This approach allows our agents to reason through the lens of relational inductive bias. Agents learn how to reason about hosts interacting with other system entities in a more general manner, and their actions are understood as edits to the graph representing the environment. By introducing this bias, we will show that our agents can better reason about the states of networks and zero-shot adapt to new ones. We show that this approach outperforms the state-of-the-art by a wide margin, and makes our agents capable of defending never-before-seen networks against a wide range of adversaries in a variety of complex, and multi-agent environments.

**arXiv ID:** 2509.16151
</details>

<details>
<summary><strong>MatchFixAgent: Language-Agnostic Autonomous Repository-Level Code Translation Validation and Repair</strong> - Ali Reza Ibrahimzada, Brandon Paulsen, Reyhaneh Jabbarvand, Joey Dodds, Daniel Kroening - [[pdf]](https://arxiv.org/pdf/2509.16187)</summary>

**Abstract:** Code translation transforms source code from one programming language (PL) to another. Validating the functional equivalence of translation and repairing, if necessary, are critical steps in code translation. Existing automated validation and repair approaches struggle to generalize to many PLs due to high engineering overhead, and they rely on existing and often inadequate test suites, which results in false claims of equivalence and ineffective translation repair. We develop MatchFixAgent, a large language model (LLM)-based, PL-agnostic framework for equivalence validation and repair of translations. MatchFixAgent features a multi-agent architecture that divides equivalence validation into several sub-tasks to ensure thorough and consistent semantic analysis of the translation. Then it feeds this analysis to test agent to write and execute tests. Upon observing a test failure, the repair agent attempts to fix the translation bug. The final (in)equivalence decision is made by the verdict agent, considering semantic analyses and test execution results.
We compare MatchFixAgent's validation and repair results with four repository-level code translation techniques. We use 2,219 translation pairs from their artifacts, which cover 6 PL pairs, and are collected from 24 GitHub projects totaling over 900K lines of code. Our results demonstrate that MatchFixAgent produces (in)equivalence verdicts for 99.2% of translation pairs, with the same equivalence validation result as prior work on 72.8% of them. When MatchFixAgent's result disagrees with prior work, we find that 60.7% of the time MatchFixAgent's result is actually correct. In addition, we show that MatchFixAgent can repair 50.6% of inequivalent translation, compared to prior work's 18.5%. This demonstrates that MatchFixAgent is far more adaptable to many PL pairs than prior work, while producing highly accurate validation results.

**arXiv ID:** 2509.16187
</details>

<details>
<summary><strong>SMART: Scalable Multi-Agent Reasoning and Trajectory Planning in Dense Environments</strong> - Heye Huang, Yibin Yang, Wang Chen, Tiantian Chen, Xiaopeng Li, Sikai Chen - [[pdf]](https://arxiv.org/pdf/2509.15737)</summary>

**Abstract:** Multi-vehicle trajectory planning is a non-convex problem that becomes increasingly difficult in dense environments due to the rapid growth of collision constraints. Efficient exploration of feasible behaviors and resolution of tight interactions are essential for real-time, large-scale coordination. This paper introduces SMART, Scalable Multi-Agent Reasoning and Trajectory Planning, a hierarchical framework that combines priority-based search with distributed optimization to achieve efficient and feasible multi-vehicle planning. The upper layer explores diverse interaction modes using reinforcement learning-based priority estimation and large-step hybrid A* search, while the lower layer refines solutions via parallelizable convex optimization. By partitioning space among neighboring vehicles and constructing robust feasible corridors, the method decouples the joint non-convex problem into convex subproblems solved efficiently in parallel. This design alleviates the step-size trade-off while ensuring kinematic feasibility and collision avoidance. Experiments show that SMART consistently outperforms baselines. On 50 m x 50 m maps, it sustains over 90% success within 1 s up to 25 vehicles, while baselines often drop below 50%. On 100 m x 100 m maps, SMART achieves above 95% success up to 50 vehicles and remains feasible up to 90 vehicles, with runtimes more than an order of magnitude faster than optimization-only approaches. Built on vehicle-to-everything communication, SMART incorporates vehicle-infrastructure cooperation through roadside sensing and agent coordination, improving scalability and safety. Real-world experiments further validate this design, achieving planning times as low as 0.014 s while preserving cooperative behaviors.

**arXiv ID:** 2509.15737
</details>

</details>

<details open>
<summary><h2>Other Agent Research (5 papers)</h2></summary>

<details>
<summary><strong>Generating Plans for Belief-Desire-Intention (BDI) Agents Using Alternating-Time Temporal Logic (ATL)</strong> - Dylan Léveillé - [[pdf]](https://arxiv.org/pdf/2509.15238)</summary>

**Abstract:** Belief-Desire-Intention (BDI) is a framework for modelling agents based on their beliefs, desires, and intentions. Plans are a central component of BDI agents, and define sequences of actions that an agent must undertake to achieve a certain goal. Existing approaches to plan generation often require significant manual effort, and are mainly focused on single-agent systems. As a result, in this work, we have developed a tool that automatically generates BDI plans using Alternating-Time Temporal Logic (ATL). By using ATL, the plans generated accommodate for possible competition or cooperation between the agents in the system. We demonstrate the effectiveness of the tool by generating plans for an illustrative game that requires agent collaboration to achieve a shared goal. We show that the generated plans allow the agents to successfully attain this goal.

**arXiv ID:** 2509.15238
</details>

<details>
<summary><strong>Towards Interactive and Learnable Cooperative Driving Automation: a Large Language Model-Driven Decision-Making Framework</strong> - Shiyu Fang, Jiaqi Liu, Mingyu Ding, Yiming Cui, Chen Lv, Peng Hang, Jian Sun - [[pdf]](https://arxiv.org/pdf/2409.12812)</summary>

**Abstract:** At present, Connected Autonomous Vehicles (CAVs) have begun to open road testing around the world, but their safety and efficiency performance in complex scenarios is still not satisfactory. Cooperative driving leverages the connectivity ability of CAVs to achieve synergies greater than the sum of their parts, making it a promising approach to improving CAV performance in complex scenarios. However, the lack of interaction and continuous learning ability limits current cooperative driving to single-scenario applications and specific Cooperative Driving Automation (CDA). To address these challenges, this paper proposes CoDrivingLLM, an interactive and learnable LLM-driven cooperative driving framework, to achieve all-scenario and all-CDA. First, since Large Language Models(LLMs) are not adept at handling mathematical calculations, an environment module is introduced to update vehicle positions based on semantic decisions, thus avoiding potential errors from direct LLM control of vehicle positions. Second, based on the four levels of CDA defined by the SAE J3216 standard, we propose a Chain-of-Thought (COT) based reasoning module that includes state perception, intent sharing, negotiation, and decision-making, enhancing the stability of LLMs in multi-step reasoning tasks. Centralized conflict resolution is then managed through a conflict coordinator in the reasoning process. Finally, by introducing a memory module and employing retrieval-augmented generation, CAVs are endowed with the ability to learn from their past experiences. We validate the proposed CoDrivingLLM through ablation experiments on the negotiation module, reasoning with different shots experience, and comparison with other cooperative driving methods.

**arXiv ID:** 2409.12812
</details>

<details>
<summary><strong>Trust-Aware Embodied Bayesian Persuasion for Mixed-Autonomy</strong> - Shaoting Peng, Katherine Driggs-Campbell, Roy Dong - [[pdf]](https://arxiv.org/pdf/2509.15404)</summary>

**Abstract:** Safe and efficient interaction between autonomous vehicles (AVs) and human-driven vehicles (HVs) is a critical challenge for future transportation systems. While game-theoretic models capture how AVs influence HVs, they often suffer from a long-term decay of influence and can be perceived as manipulative, eroding the human's trust. This can paradoxically lead to riskier human driving behavior over repeated interactions. In this paper, we address this challenge by proposing the Trust-Aware Embodied Bayesian Persuasion (TA-EBP) framework. Our work makes three key contributions: First, we apply Bayesian persuasion to model communication at traffic intersections, offering a transparent alternative to traditional game-theoretic models. Second, we introduce a trust parameter to the persuasion framework, deriving a theorem for the minimum trust level required for influence. Finally, we ground the abstract signals of Bayesian persuasion theory into a continuous, physically meaningful action space, deriving a second theorem for the optimal signal magnitude, realized as an AV's forward nudge. Additionally, we validate our framework in a mixed-autonomy traffic simulation, demonstrating that TA-EBP successfully persuades HVs to drive more cautiously, eliminating collisions and improving traffic flow compared to baselines that either ignore trust or lack communication. Our work provides a transparent and non-strategic framework for influence in human-robot interaction, enhancing both safety and efficiency.

**arXiv ID:** 2509.15404
</details>

<details>
<summary><strong>Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories</strong> - Yifan Lin, Sophie Ziyu Liu, Ran Qi, George Z. Xue, Xinping Song, Chao Qin, Hugh H.-T. Liu - [[pdf]](https://arxiv.org/pdf/2509.16176)</summary>

**Abstract:** We present Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories (ACDC), an autonomous drone cinematography system driven by natural language communication between human directors and drones. The main limitation of previous drone cinematography workflows is that they require manual selection of waypoints and view angles based on predefined human intent, which is labor-intensive and yields inconsistent performance. In this paper, we propose employing large language models (LLMs) and vision foundation models (VFMs) to convert free-form natural language prompts directly into executable indoor UAV video tours. Specifically, our method comprises a vision-language retrieval pipeline for initial waypoint selection, a preference-based Bayesian optimization framework that refines poses using aesthetic feedback, and a motion planner that generates safe quadrotor trajectories. We validate ACDC through both simulation and hardware-in-the-loop experiments, demonstrating that it robustly produces professional-quality footage across diverse indoor scenes without requiring expertise in robotics or cinematography. These results highlight the potential of embodied AI agents to close the loop from open-vocabulary dialogue to real-world autonomous aerial cinematography.

**arXiv ID:** 2509.16176
</details>

<details>
<summary><strong>Multi-Quadruped Cooperative Object Transport: Learning Decentralized Pinch-Lift-Move</strong> - Bikram Pandit, Aayam Kumar Shrestha, Alan Fern - [[pdf]](https://arxiv.org/pdf/2509.14342)</summary>

**Abstract:** We study decentralized cooperative transport using teams of N-quadruped robots with arm that must pinch, lift, and move ungraspable objects through physical contact alone. Unlike prior work that relies on rigid mechanical coupling between robots and objects, we address the more challenging setting where mechanically independent robots must coordinate through contact forces alone without any communication or centralized control. To this end, we employ a hierarchical policy architecture that separates base locomotion from arm control, and propose a constellation reward formulation that unifies position and orientation tracking to enforce rigid contact behavior. The key insight is encouraging robots to behave as if rigidly connected to the object through careful reward design and training curriculum rather than explicit mechanical constraints. Our approach enables coordination through shared policy parameters and implicit synchronization cues - scaling to arbitrary team sizes without retraining. We show extensive simulation experiments to demonstrate robust transport across 2-10 robots on diverse object geometries and masses, along with sim2real transfer results on lightweight objects.

**arXiv ID:** 2509.14342
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (32 papers)</h2></summary>

<details>
<summary><strong>The Distribution Shift Problem in Transportation Networks using Reinforcement Learning and AI</strong> - Federico Taschin, Abderrahmane Lazaraq, Ozan K. Tonguz, Inci Ozgunes - [[pdf]](https://arxiv.org/pdf/2509.15291)</summary>

**Abstract:** The use of Machine Learning (ML) and Artificial Intelligence (AI) in smart transportation networks has increased significantly in the last few years. Among these ML and AI approaches, Reinforcement Learning (RL) has been shown to be a very promising approach by several authors. However, a problem with using Reinforcement Learning in Traffic Signal Control is the reliability of the trained RL agents due to the dynamically changing distribution of the input data with respect to the distribution of the data used for training. This presents a major challenge and a reliability problem for the trained network of AI agents and could have very undesirable and even detrimental consequences if a suitable solution is not found. Several researchers have tried to address this problem using different approaches. In particular, Meta Reinforcement Learning (Meta RL) promises to be an effective solution. In this paper, we evaluate and analyze a state-of-the-art Meta RL approach called MetaLight and show that, while under certain conditions MetaLight can indeed lead to reasonably good results, under some other conditions it might not perform well (with errors of up to 22%), suggesting that Meta RL schemes are often not robust enough and can even pose major reliability problems.

**arXiv ID:** 2509.15291
</details>

<details>
<summary><strong>CCrepairBench: A High-Fidelity Benchmark and Reinforcement Learning Framework for C++ Compilation Repair</strong> - Weixuan Sun, Jucai Zhai, Dengfeng Liu, Xin Zhang, Xiaojun Wu, Qiaobo Hao, AIMgroup, Yang Fang, Jiuyang Tang - [[pdf]](https://arxiv.org/pdf/2509.15690)</summary>

**Abstract:** The automated repair of C++ compilation errors presents a significant challenge, the resolution of which is critical for developer productivity. Progress in this domain is constrained by two primary factors: the scarcity of large-scale, high-fidelity datasets and the limitations of conventional supervised methods, which often fail to generate semantically correct this http URL paper addresses these gaps by introducing a comprehensive framework with three core contributions. First, we present CCrepair, a novel, large-scale C++ compilation error dataset constructed through a sophisticated generate-and-verify pipeline. Second, we propose a Reinforcement Learning (RL) paradigm guided by a hybrid reward signal, shifting the focus from mere compilability to the semantic quality of the fix. Finally, we establish the robust, two-stage evaluation system providing this signal, centered on an LLM-as-a-Judge whose reliability has been rigorously validated against the collective judgments of a panel of human experts. This integrated approach aligns the training objective with generating high-quality, non-trivial patches that are both syntactically and semantically correct. The effectiveness of our approach was demonstrated experimentally. Our RL-trained Qwen2.5-1.5B-Instruct model achieved performance comparable to a Qwen2.5-14B-Instruct model, validating the efficiency of our training paradigm. Our work provides the research community with a valuable new dataset and a more effective paradigm for training and evaluating robust compilation repair models, paving the way for more practical and reliable automated programming assistants.

**arXiv ID:** 2509.15690
</details>

<details>
<summary><strong>A Nascent Taxonomy of Machine Learning in Intelligent Robotic Process Automation</strong> - Lukas Laakmann, Seyyid A. Ciftci, Christian Janiesch - [[pdf]](https://arxiv.org/pdf/2509.15730)</summary>

**Abstract:** Robotic process automation (RPA) is a lightweight approach to automating business processes using software robots that emulate user actions at the graphical user interface level. While RPA has gained popularity for its cost-effective and timely automation of rule-based, well-structured tasks, its symbolic nature has inherent limitations when approaching more complex tasks currently performed by human agents. Machine learning concepts enabling intelligent RPA provide an opportunity to broaden the range of automatable tasks. In this paper, we conduct a literature review to explore the connections between RPA and machine learning and organize the joint concept intelligent RPA into a taxonomy. Our taxonomy comprises the two meta-characteristics RPA-ML integration and RPA-ML interaction. Together, they comprise eight dimensions: architecture and ecosystem, capabilities, data basis, intelligence level, and technical depth of integration as well as deployment environment, lifecycle phase, and user-robot relation.

**arXiv ID:** 2509.15730
</details>

<details>
<summary><strong>ORCA: Agentic Reasoning For Hallucination and Adversarial Robustness in Vision-Language Models</strong> - Chung-En Johnny Yu, Hsuan-Chih, Chen, Brian Jalaian, Nathaniel D. Bastian - [[pdf]](https://arxiv.org/pdf/2509.15435)</summary>

**Abstract:** Large Vision-Language Models (LVLMs) exhibit strong multimodal capabilities but remain vulnerable to hallucinations from intrinsic errors and adversarial attacks from external exploitations, limiting their reliability in real-world applications. We present ORCA, an agentic reasoning framework that improves the factual accuracy and adversarial robustness of pretrained LVLMs through test-time structured inference reasoning with a suite of small vision models (less than 3B parameters). ORCA operates via an Observe--Reason--Critique--Act loop, querying multiple visual tools with evidential questions, validating cross-model inconsistencies, and refining predictions iteratively without access to model internals or retraining. ORCA also stores intermediate reasoning traces, which supports auditable decision-making. Though designed primarily to mitigate object-level hallucinations, ORCA also exhibits emergent adversarial robustness without requiring adversarial training or defense mechanisms. We evaluate ORCA across three settings: (1) clean images on hallucination benchmarks, (2) adversarially perturbed images without defense, and (3) adversarially perturbed images with defense applied. On the POPE hallucination benchmark, ORCA improves standalone LVLM performance by +3.64\% to +40.67\% across different subsets. Under adversarial perturbations on POPE, ORCA achieves an average accuracy gain of +20.11\% across LVLMs. When combined with defense techniques on adversarially perturbed AMBER images, ORCA further improves standalone LVLM performance, with gains ranging from +1.20\% to +48.00\% across evaluation metrics. These results demonstrate that ORCA offers a promising path toward building more reliable and robust multimodal systems.

**arXiv ID:** 2509.15435
</details>

<details>
<summary><strong>GUI-ARP: Enhancing Grounding with Adaptive Region Perception for GUI Agents</strong> - Xianhang Ye, Yiqing Li, Wei Dai, Miancan Liu, Ziyuan Chen, Zhangye Han, Hongbo Min, Jinkui Ren, Xiantao Zhang, Wen Yang, Zhi Jin - [[pdf]](https://arxiv.org/pdf/2509.15532)</summary>

**Abstract:** Existing GUI grounding methods often struggle with fine-grained localization in high-resolution screenshots. To address this, we propose GUI-ARP, a novel framework that enables adaptive multi-stage inference. Equipped with the proposed Adaptive Region Perception (ARP) and Adaptive Stage Controlling (ASC), GUI-ARP dynamically exploits visual attention for cropping task-relevant regions and adapts its inference strategy, performing a single-stage inference for simple cases and a multi-stage analysis for more complex scenarios. This is achieved through a two-phase training pipeline that integrates supervised fine-tuning with reinforcement fine-tuning based on Group Relative Policy Optimization (GRPO). Extensive experiments demonstrate that the proposed GUI-ARP achieves state-of-the-art performance on challenging GUI grounding benchmarks, with a 7B model reaching 60.8% accuracy on ScreenSpot-Pro and 30.9% on UI-Vision benchmark. Notably, GUI-ARP-7B demonstrates strong competitiveness against open-source 72B models (UI-TARS-72B at 38.1%) and proprietary models.

**arXiv ID:** 2509.15532
</details>

<details>
<summary><strong>BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent</strong> - Shaojie Zhang, Ruoceng Zhang, Pei Fu, Shaokang Wang, Jiahui Yang, Xin Du, Shiqi Cui, Bin Qin, Ying Huang, Zhenbo Luo, Jian Luan - [[pdf]](https://arxiv.org/pdf/2509.15566)</summary>

**Abstract:** In the field of AI-driven human-GUI interaction automation, while rapid advances in multimodal large language models and reinforcement fine-tuning techniques have yielded remarkable progress, a fundamental challenge persists: their interaction logic significantly deviates from natural human-GUI communication patterns. To fill this gap, we propose "Blink-Think-Link" (BTL), a brain-inspired framework for human-GUI interaction that mimics the human cognitive process between users and graphical interfaces. The system decomposes interactions into three biologically plausible phases: (1) Blink - rapid detection and attention to relevant screen areas, analogous to saccadic eye movements; (2) Think - higher-level reasoning and decision-making, mirroring cognitive planning; and (3) Link - generation of executable commands for precise motor control, emulating human action selection mechanisms. Additionally, we introduce two key technical innovations for the BTL framework: (1) Blink Data Generation - an automated annotation pipeline specifically optimized for blink data, and (2) BTL Reward -- the first rule-based reward mechanism that enables reinforcement learning driven by both process and outcome. Building upon this framework, we develop a GUI agent model named BTL-UI, which demonstrates consistent state-of-the-art performance across both static GUI understanding and dynamic interaction tasks in comprehensive benchmarks. These results provide conclusive empirical validation of the framework's efficacy in developing advanced GUI Agents.

**arXiv ID:** 2509.15566
</details>

<details>
<summary><strong>ChronoForge-RL: Chronological Forging through Reinforcement Learning for Enhanced Video Understanding</strong> - Kehua Chen - [[pdf]](https://arxiv.org/pdf/2509.15800)</summary>

**Abstract:** Current state-of-the-art video understanding methods typically struggle with two critical challenges: (1) the computational infeasibility of processing every frame in dense video content and (2) the difficulty in identifying semantically significant frames through naive uniform sampling strategies. In this paper, we propose a novel video understanding framework, called ChronoForge-RL, which combines Temporal Apex Distillation (TAD) and KeyFrame-aware Group Relative Policy Optimization (KF-GRPO) to tackle these issues. Concretely, we introduce a differentiable keyframe selection mechanism that systematically identifies semantic inflection points through a three-stage process to enhance computational efficiency while preserving temporal information. Then, two particular modules are proposed to enable effective temporal reasoning: Firstly, TAD leverages variation scoring, inflection detection, and prioritized distillation to select the most informative frames. Secondly, we introduce KF-GRPO which implements a contrastive learning paradigm with a saliency-enhanced reward mechanism that explicitly incentivizes models to leverage both frame content and temporal relationships. Finally, our proposed ChronoForge-RL achieves 69.1% on VideoMME and 52.7% on LVBench compared to baseline methods, clearly surpassing previous approaches while enabling our 7B parameter model to achieve performance comparable to 72B parameter alternatives.

**arXiv ID:** 2509.15800
</details>

<details>
<summary><strong>A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning</strong> - Shaopeng Zhai, Qi Zhang, Tianyi Zhang, Fuxian Huang, Haoran Zhang, Ming Zhou, Shengzhe Zhang, Litao Liu, Sixu Lin, Jiangmiao Pang - [[pdf]](https://arxiv.org/pdf/2509.15937)</summary>

**Abstract:** Robotic real-world reinforcement learning (RL) with vision-language-action (VLA) models is bottlenecked by sparse, handcrafted rewards and inefficient exploration. We introduce VLAC, a general process reward model built upon InternVL and trained on large scale heterogeneous datasets. Given pairwise observations and a language goal, it outputs dense progress delta and done signal, eliminating task-specific reward engineering, and supports one-shot in-context transfer to unseen tasks and environments. VLAC is trained on vision-language datasets to strengthen perception, dialogic and reasoning capabilities, together with robot and human trajectories data that ground action generation and progress estimation, and additionally strengthened to reject irrelevant prompts as well as detect regression or stagnation by constructing large numbers of negative and semantically mismatched samples. With prompt control, a single VLAC model alternately generating reward and action tokens, unifying critic and policy. Deployed inside an asynchronous real-world RL loop, we layer a graded human-in-the-loop protocol (offline demonstration replay, return and explore, human guided explore) that accelerates exploration and stabilizes early learning. Across four distinct real-world manipulation tasks, VLAC lifts success rates from about 30\% to about 90\% within 200 real-world interaction episodes; incorporating human-in-the-loop interventions yields a further 50% improvement in sample efficiency and achieves up to 100% final success.

**arXiv ID:** 2509.15937
</details>

<details>
<summary><strong>Explainable AI for Maritime Autonomous Surface Ships (MASS): Adaptive Interfaces and Trustworthy Human-AI Collaboration</strong> - Zhuoyue Zhang, Haitong Xu - [[pdf]](https://arxiv.org/pdf/2509.15959)</summary>

**Abstract:** Autonomous navigation in maritime domains is accelerating alongside advances in artificial intelligence, sensing, and connectivity. Opaque decision-making and poorly calibrated human-automation interaction remain key barriers to safe adoption. This article synthesizes 100 studies on automation transparency for Maritime Autonomous Surface Ships (MASS) spanning situation awareness (SA), human factors, interface design, and regulation. We (i) map the Guidance-Navigation-Control stack to shore-based operational modes -- remote supervision (RSM) and remote control (RCM) -- and identify where human unsafe control actions (Human-UCAs) concentrate in handover and emergency loops; (ii) summarize evidence that transparency features (decision rationales, alternatives, confidence/uncertainty, and rule-compliance indicators) improve understanding and support trust calibration, though reliability and predictability often dominate trust; (iii) distill design strategies for transparency at three layers: sensor/SA acquisition and fusion, HMI/eHMI presentation (textual/graphical overlays, color coding, conversational and immersive UIs), and engineer-facing processes (resilient interaction design, validation, and standardization). We integrate methods for Human-UCA identification (STPA-Cog + IDAC), quantitative trust/SA assessment, and operator workload monitoring, and outline regulatory and rule-based implications including COLREGs formalization and route exchange. We conclude with an adaptive transparency framework that couples operator state estimation with explainable decision support to reduce cognitive overload and improve takeover timeliness. The review highlights actionable figure-of-merit displays (e.g., CPA/TCPA risk bars, robustness heatmaps), transparent model outputs (rule traceability, confidence), and training pipelines (HIL/MIL, simulation) as near-term levers for safer MASS operations.

**arXiv ID:** 2509.15959
</details>

<details>
<summary><strong>RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation</strong> - Chao Yu, Yuanqing Wang, Zhen Guo, Hao Lin, Si Xu, Hongzhi Zang, Quanlu Zhang, Yongji Wu, Chunyang Zhu, Junhao Hu, Zixiao Huang, Mingjie Wei, Yuqing Xie, Ke Yang, Bo Dai, Zhexuan Xu, Xiangyuan Wang, Xu Fu, Zhihao Liu, Kang Chen, Weilin Liu, Gang Liu, Boxun Li, Jianlei Yang, Zhi Yang, Guohao Dai, Yu Wang - [[pdf]](https://arxiv.org/pdf/2509.15965)</summary>

**Abstract:** Reinforcement learning (RL) has demonstrated immense potential in advancing artificial general intelligence, agentic intelligence, and embodied intelligence. However, the inherent heterogeneity and dynamicity of RL workflows often lead to low hardware utilization and slow training on existing systems. In this paper, we present RLinf, a high-performance RL training system based on our key observation that the major roadblock to efficient RL training lies in system flexibility. To maximize flexibility and efficiency, RLinf is built atop a novel RL system design paradigm called macro-to-micro flow transformation (M2Flow), which automatically breaks down high-level, easy-to-compose RL workflows at both the temporal and spatial dimensions, and recomposes them into optimized execution flows. Supported by RLinf worker's adaptive communication capability, we devise context switching and elastic pipelining to realize M2Flow transformation, and a profiling-guided scheduling policy to generate optimal execution plans. Extensive evaluations on both reasoning RL and embodied RL tasks demonstrate that RLinf consistently outperforms state-of-the-art systems, achieving 1.1x-2.13x speedup in end-to-end training throughput.

**arXiv ID:** 2509.15965
</details>

<details>
<summary><strong>Uncertainty-Based Smooth Policy Regularisation for Reinforcement Learning with Few Demonstrations</strong> - Yujie Zhu, Charles A. Hepburn, Matthew Thorpe, Giovanni Montana - [[pdf]](https://arxiv.org/pdf/2509.15981)</summary>

**Abstract:** In reinforcement learning with sparse rewards, demonstrations can accelerate learning, but determining when to imitate them remains challenging. We propose Smooth Policy Regularisation from Demonstrations (SPReD), a framework that addresses the fundamental question: when should an agent imitate a demonstration versus follow its own policy? SPReD uses ensemble methods to explicitly model Q-value distributions for both demonstration and policy actions, quantifying uncertainty for comparisons. We develop two complementary uncertainty-aware methods: a probabilistic approach estimating the likelihood of demonstration superiority, and an advantage-based approach scaling imitation by statistical significance. Unlike prevailing methods (e.g. Q-filter) that make binary imitation decisions, SPReD applies continuous, uncertainty-proportional regularisation weights, reducing gradient variance during training. Despite its computational simplicity, SPReD achieves remarkable gains in experiments across eight robotics tasks, outperforming existing approaches by up to a factor of 14 in complex tasks while maintaining robustness to demonstration quality and quantity. Our code is available at this https URL.

**arXiv ID:** 2509.15981
</details>

<details>
<summary><strong>Accelerating Atomic Fine Structure Determination with Graph Reinforcement Learning</strong> - M. Ding, V.-A. Darvariu, A. N. Ryabtsev, N. Hawes, J. C. Pickering - [[pdf]](https://arxiv.org/pdf/2509.16184)</summary>

**Abstract:** Atomic data determined by analysis of observed atomic spectra are essential for plasma diagnostics. For each low-ionisation open d- and f-subshell atomic species, around $10^3$ fine structure level energies can be determined through years of analysis of $10^4$ observable spectral lines. We propose the automation of this task by casting the analysis procedure as a Markov decision process and solving it by graph reinforcement learning using reward functions learned on historical human decisions. In our evaluations on existing spectral line lists and theoretical calculations for Co II and Nd II-III, hundreds of level energies were computed within hours, agreeing with published values in 95% of cases for Co II and 54-87% for Nd II-III. As the current efficiency in atomic fine structure determination struggles to meet growing atomic data demands from astronomy and fusion science, our new artificial intelligence approach sets the stage for closing this gap.

**arXiv ID:** 2509.16184
</details>

<details>
<summary><strong>Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering</strong> - Liang Zhang, Justin Lieffers, Adarsh Pyarelal - [[pdf]](https://arxiv.org/pdf/2409.17411)</summary>

**Abstract:** In this paper, we explore semantic clustering properties of deep reinforcement learning (DRL) to improve its interpretability and deepen our understanding of its internal semantic organization. In this context, semantic clustering refers to the ability of neural networks to cluster inputs based on their semantic similarity in the internal space. We propose a DRL architecture that incorporates a novel semantic clustering module that combines feature dimensionality reduction with online clustering. This module integrates seamlessly into the DRL training pipeline, addressing the instability of t-SNE and eliminating the need for extensive manual annotation inherent to prior semantic analysis methods. We experimentally validate the effectiveness of the proposed module and demonstrate its ability to reveal semantic clustering properties within DRL. Furthermore, we introduce new analytical methods based on these properties to provide insights into the hierarchical structure of policies and semantic organization within the feature space.

**arXiv ID:** 2409.17411
</details>

<details>
<summary><strong>Dynamic Neural Curiosity Enhances Learning Flexibility for Autonomous Goal Discovery</strong> - Quentin Houbre, Roel Pieters - [[pdf]](https://arxiv.org/pdf/2412.00152)</summary>

**Abstract:** The autonomous learning of new goals in robotics remains a complex issue to address. Here, we propose a model where curiosity influence learning flexibility. To do so, this paper proposes to root curiosity and attention together by taking inspiration from the Locus Coeruleus-Norepinephrine system along with various cognitive processes such as cognitive persistence and visual habituation. We apply our approach by experimenting with a simulated robotic arm on a set of objects with varying difficulty. The robot first discovers new goals via bottom-up attention through motor babbling with an inhibition of return mechanism, then engage to the learning of goals due to neural activity arising within the curiosity mechanism. The architecture is modelled with dynamic neural fields and the learning of goals such as pushing the objects in diverse directions is supported by the use of forward and inverse models implemented by multi-layer perceptrons. The adoption of dynamic neural fields to model curiosity, habituation and persistence allows the robot to demonstrate various learning trajectories depending on the object. In addition, the approach exhibits interesting properties regarding the learning of similar goals as well as the continuous switch between exploration and exploitation.

**arXiv ID:** 2412.00152
</details>

<details>
<summary><strong>Deep Reinforcement Learning with Gradient Eligibility Traces</strong> - Esraa Elelimy, Brett Daley, Andrew Patterson, Marlos C. Machado, Adam White, Martha White - [[pdf]](https://arxiv.org/pdf/2507.09087)</summary>

**Abstract:** Achieving fast and stable off-policy learning in deep reinforcement learning (RL) is challenging. Most existing methods rely on semi-gradient temporal-difference (TD) methods for their simplicity and efficiency, but are consequently susceptible to divergence. While more principled approaches like Gradient TD (GTD) methods have strong convergence guarantees, they have rarely been used in deep RL. Recent work introduced the generalized Projected Bellman Error ($\overline{\text{PBE}}$), enabling GTD methods to work efficiently with nonlinear function approximation. However, this work is limited to one-step methods, which are slow at credit assignment and require a large number of samples. In this paper, we extend the generalized $\overline{\text{PBE}}$ objective to support multistep credit assignment based on the $\lambda$-return and derive three gradient-based methods that optimize this new objective. We provide both a forward-view formulation compatible with experience replay and a backward-view formulation compatible with streaming algorithms. Finally, we evaluate the proposed algorithms and show that they outperform both PPO and StreamQ in MuJoCo and MinAtar environments, respectively. Code available at this https URL\_algos

**arXiv ID:** 2507.09087
</details>

<details>
<summary><strong>CORE-RAG: Lossless Compression for Retrieval-Augmented LLMs via Reinforcement Learning</strong> - Ziqiang Cui, Yunpeng Weng, Xing Tang, Peiyang Liu, Shiwei Li, Bowei He, Jiamin Chen, Yansen Zhang, Xiuqiang He, Chen Ma - [[pdf]](https://arxiv.org/pdf/2508.19282)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) has emerged as a promising approach to enhance the timeliness of knowledge and the factual accuracy of responses in Large Language Models (LLMs). However, the inclusion of excessive retrieved documents substantially increases the input length, leading to higher computational costs. Previous studies have attempted to compress retrieved documents into shorter texts before in-context integration, but such methods often compromise end-task performance. The lack of well-defined compression targets forces many approaches to rely on fixed heuristics, which cannot guarantee that the compressed content will effectively support the end task. To address these limitations, we propose CORE, a novel method designed to achieve lossless context compression for RAG. CORE employs reinforcement learning to optimize the compression process without relying on predefined compression labels, which enables the compressor to generate summaries that maximize the accuracy of answers generated by the LLM. Extensive experiments on four datasets demonstrate the superiority of our approach. With a high compression ratio of 3\%, our method not only avoids performance degradation compared to prepending full documents across all datasets but also improves the average Exact Match (EM) score by 3.3 points. The code will be released soon.

**arXiv ID:** 2508.19282
</details>

<details>
<summary><strong>PVPO: Pre-Estimated Value-Based Policy Optimization for Agentic Reasoning</strong> - Wenfeng Feng, Penghong Zhao, Guochao Jiang, Chuzhan Hao, Yuewei Zhang, Guohua Liu, Hao Wang - [[pdf]](https://arxiv.org/pdf/2508.21104)</summary>

**Abstract:** Critic-free reinforcement learning methods, particularly group policies, have attracted considerable attention for their efficiency in complex tasks. However, these methods rely heavily on multiple sampling and comparisons within the policy to estimate advantage, which may cause the policy to fall into local optimum and increase computational cost. To address these issues, we propose PVPO, an efficient reinforcement learning method enhanced by an advantage reference anchor and data pre-sampling. Specifically, we use the reference model to rollout in advance and employ the calculated reward score as a reference anchor. Our approach effectively corrects the cumulative bias introduced by intra-group comparisons and significantly reduces reliance on the number of rollouts during training. Meanwhile, the reference model can assess sample difficulty during data pre-sampling, enabling effective selection of high-gain data to improve training efficiency. Moreover, PVPO is orthogonal to other advanced critic-free RL algorithms, making it compatible with and complementary to these methods. Experiments conducted on nine datasets across two domains demonstrate that PVPO achieves State-Of-The-Art (SOTA) performance. Our approach not only demonstrates robust generalization across multiple tasks, but also exhibits scalable performance across models of varying scales.

**arXiv ID:** 2508.21104
</details>

<details>
<summary><strong>SWE-Effi: Re-Evaluating Software AI Agent System Effectiveness Under Resource Constraints</strong> - Zhiyu Fan, Kirill Vasilevski, Dayi Lin, Boyuan Chen, Yihao Chen, Zhiqing Zhong, Jie M. Zhang, Pinjia He, Ahmed E. Hassan - [[pdf]](https://arxiv.org/pdf/2509.09853)</summary>

**Abstract:** The advancement of large language models (LLMs) and code agents has demonstrated significant potential to assist software engineering (SWE) tasks, such as autonomous issue resolution and feature addition. Existing AI for software engineering leaderboards (e.g., SWE-bench) focus solely on solution accuracy, ignoring the crucial factor of effectiveness in a resource-constrained world. This is a universal problem that also exists beyond software engineering tasks: any AI system should be more than correct - it must also be cost-effective. To address this gap, we introduce SWE-Effi, a set of new metrics to re-evaluate AI systems in terms of holistic effectiveness scores. We define effectiveness as the balance between the accuracy of outcome (e.g., issue resolve rate) and the resources consumed (e.g., token and time). In this paper, we specifically focus on the software engineering scenario by re-ranking popular AI systems for issue resolution on a subset of the SWE-bench benchmark using our new multi-dimensional metrics. We found that AI system's effectiveness depends not just on the scaffold itself, but on how well it integrates with the base model, which is key to achieving strong performance in a resource-efficient manner. We also identified systematic challenges such as the "token snowball" effect and, more significantly, a pattern of "expensive failures". In these cases, agents consume excessive resources while stuck on unsolvable tasks - an issue that not only limits practical deployment but also drives up the cost of failed rollouts during RL training. Lastly, we observed a clear trade-off between effectiveness under the token budget and effectiveness under the time budget, which plays a crucial role in managing project budgets and enabling scalable reinforcement learning, where fast responses are essential.

**arXiv ID:** 2509.09853
</details>

<details>
<summary><strong>TGPO: Tree-Guided Preference Optimization for Robust Web Agent Reinforcement Learning</strong> - Ziyuan Chen, Zhenghui Zhao, Zhangye Han, Miancan Liu, Xianhang Ye, Yiqing Li, Hongbo Min, Jinkui Ren, Xiantao Zhang, Guitao Cao - [[pdf]](https://arxiv.org/pdf/2509.14172)</summary>

**Abstract:** With the rapid advancement of large language models and vision-language models, employing large models as Web Agents has become essential for automated web interaction. However, training Web Agents with reinforcement learning faces critical challenges including credit assignment misallocation, prohibitively high annotation costs, and reward sparsity. To address these issues, we propose Tree-Guided Preference Optimization (TGPO), an offline reinforcement learning framework that proposes a tree-structured trajectory representation merging semantically identical states across trajectories to eliminate label conflicts. Our framework incorporates a Process Reward Model that automatically generates fine-grained rewards through subgoal progress, redundancy detection, and action verification. Additionally, a dynamic weighting mechanism prioritizes high-impact decision points during training. Experiments on Online-Mind2Web and our self-constructed C-WebShop datasets demonstrate that TGPO significantly outperforms existing methods, achieving higher success rates with fewer redundant steps.

**arXiv ID:** 2509.14172
</details>

<details>
<summary><strong>Empathy-R1: A Chain-of-Empathy and Reinforcement Learning Framework for Long-Form Mental Health Support</strong> - Xianrong Yao, Dong She, Chenxu Zhang, Yimeng Zhang, Yueru Sun, Noman Ahmed, Yang Gao, Zhanpeng Jin - [[pdf]](https://arxiv.org/pdf/2509.14851)</summary>

**Abstract:** Empathy is critical for effective mental health support, especially when addressing Long Counseling Texts (LCTs). However, existing Large Language Models (LLMs) often generate replies that are semantically fluent but lack the structured reasoning necessary for genuine psychological support, particularly in a Chinese context. To bridge this gap, we introduce Empathy-R1, a novel framework that integrates a Chain-of-Empathy (CoE) reasoning process with Reinforcement Learning (RL) to enhance response quality for LCTs. Inspired by cognitive-behavioral therapy, our CoE paradigm guides the model to sequentially reason about a help-seeker's emotions, causes, and intentions, making its thinking process both transparent and interpretable. Our framework is empowered by a new large-scale Chinese dataset, Empathy-QA, and a two-stage training process. First, Supervised Fine-Tuning instills the CoE's reasoning structure. Subsequently, RL, guided by a dedicated reward model, refines the therapeutic relevance and contextual appropriateness of the final responses. Experiments show that Empathy-R1 achieves strong performance on key automatic metrics. More importantly, human evaluations confirm its superiority, showing a clear preference over strong baselines and achieving a Win@1 rate of 44.30% on our new benchmark. By enabling interpretable and contextually nuanced responses, Empathy-R1 represents a significant advancement in developing responsible and genuinely beneficial AI for mental health support.

**arXiv ID:** 2509.14851
</details>

<details>
<summary><strong>Video2Roleplay: A Multimodal Dataset and Framework for Video-Guided Role-playing Agents</strong> - Xueqiao Zhang, Chao Zhang, Jingtao Xu, Yifan Zhu, Xin Shi, Yi Yang, Yawei Luo - [[pdf]](https://arxiv.org/pdf/2509.15233)</summary>

**Abstract:** Role-playing agents (RPAs) have attracted growing interest for their ability to simulate immersive and interactive characters. However, existing approaches primarily focus on static role profiles, overlooking the dynamic perceptual abilities inherent to humans. To bridge this gap, we introduce the concept of dynamic role profiles by incorporating video modality into RPAs. To support this, we construct Role-playing-Video60k, a large-scale, high-quality dataset comprising 60k videos and 700k corresponding dialogues. Based on this dataset, we develop a comprehensive RPA framework that combines adaptive temporal sampling with both dynamic and static role profile representations. Specifically, the dynamic profile is created by adaptively sampling video frames and feeding them to the LLM in temporal order, while the static profile consists of (1) character dialogues from training videos during fine-tuning, and (2) a summary context from the input video during inference. This joint integration enables RPAs to generate greater responses. Furthermore, we propose a robust evaluation method covering eight metrics. Experimental results demonstrate the effectiveness of our framework, highlighting the importance of dynamic role profiles in developing RPAs.

**arXiv ID:** 2509.15233
</details>

<details>
<summary><strong>Fleming-R1: Toward Expert-Level Medical Reasoning via Reinforcement Learning</strong> - Chi Liu, Derek Li, Yan Shu, Robin Chen, Derek Duan, Teng Fang, Bryan Dai - [[pdf]](https://arxiv.org/pdf/2509.15279)</summary>

**Abstract:** While large language models show promise in medical applications, achieving expert-level clinical reasoning remains challenging due to the need for both accurate answers and transparent reasoning processes. To address this challenge, we introduce Fleming-R1, a model designed for verifiable medical reasoning through three complementary innovations. First, our Reasoning-Oriented Data Strategy (RODS) combines curated medical QA datasets with knowledge-graph-guided synthesis to improve coverage of underrepresented diseases, drugs, and multi-hop reasoning chains. Second, we employ Chain-of-Thought (CoT) cold start to distill high-quality reasoning trajectories from teacher models, establishing robust inference priors. Third, we implement a two-stage Reinforcement Learning from Verifiable Rewards (RLVR) framework using Group Relative Policy Optimization, which consolidates core reasoning skills while targeting persistent failure modes through adaptive hard-sample mining. Across diverse medical benchmarks, Fleming-R1 delivers substantial parameter-efficient improvements: the 7B variant surpasses much larger baselines, while the 32B model achieves near-parity with GPT-4o and consistently outperforms strong open-source alternatives. These results demonstrate that structured data design, reasoning-oriented initialization, and verifiable reinforcement learning can advance clinical reasoning beyond simple accuracy optimization. We release Fleming-R1 publicly to promote transparent, reproducible, and auditable progress in medical AI, enabling safer deployment in high-stakes clinical environments.

**arXiv ID:** 2509.15279
</details>

<details>
<summary><strong>Nonconvex Regularization for Feature Selection in Reinforcement Learning</strong> - Kyohei Suzuki, Konstantinos Slavakis - [[pdf]](https://arxiv.org/pdf/2509.15652)</summary>

**Abstract:** This work proposes an efficient batch algorithm for feature selection in reinforcement learning (RL) with theoretical convergence guarantees. To mitigate the estimation bias inherent in conventional regularization schemes, the first contribution extends policy evaluation within the classical least-squares temporal-difference (LSTD) framework by formulating a Bellman-residual objective regularized with the sparsity-inducing, nonconvex projected minimax concave (PMC) penalty. Owing to the weak convexity of the PMC penalty, this formulation can be interpreted as a special instance of a general nonmonotone-inclusion problem. The second contribution establishes novel convergence conditions for the forward-reflected-backward splitting (FRBS) algorithm to solve this class of problems. Numerical experiments on benchmark datasets demonstrate that the proposed approach substantially outperforms state-of-the-art feature-selection methods, particularly in scenarios with many noisy features.

**arXiv ID:** 2509.15652
</details>

<details>
<summary><strong>GUI-ReWalk: Massive Data Generation for GUI Agent via Stochastic Exploration and Intent-Aware Reasoning</strong> - Musen Lin, Minghao Liu, Taoran Lu, Lichen Yuan, Yiwei Liu, Haonan Xu, Yu Miao, Yuhao Chao, Zhaojian Li - [[pdf]](https://arxiv.org/pdf/2509.15738)</summary>

**Abstract:** Graphical User Interface (GUI) Agents, powered by large language and vision-language models, hold promise for enabling end-to-end automation in digital environments. However, their progress is fundamentally constrained by the scarcity of scalable, high-quality trajectory data. Existing data collection strategies either rely on costly and inconsistent manual annotations or on synthetic generation methods that trade off between diversity and meaningful task coverage. To bridge this gap, we present GUI-ReWalk: a reasoning-enhanced, multi-stage framework for synthesizing realistic and diverse GUI trajectories. GUI-ReWalk begins with a stochastic exploration phase that emulates human trial-and-error behaviors, and progressively transitions into a reasoning-guided phase where inferred goals drive coherent and purposeful interactions. Moreover, it supports multi-stride task generation, enabling the construction of long-horizon workflows across multiple applications. By combining randomness for diversity with goal-aware reasoning for structure, GUI-ReWalk produces data that better reflects the intent-aware, adaptive nature of human-computer interaction. We further train Qwen2.5-VL-7B on the GUI-ReWalk dataset and evaluate it across multiple benchmarks, including Screenspot-Pro, OSWorld-G, UI-Vision, AndroidControl, and GUI-Odyssey. Results demonstrate that GUI-ReWalk enables superior coverage of diverse interaction flows, higher trajectory entropy, and more realistic user intent. These findings establish GUI-ReWalk as a scalable and data-efficient framework for advancing GUI agent research and enabling robust real-world automation.

**arXiv ID:** 2509.15738
</details>

<details>
<summary><strong>Quantum Reinforcement Learning with Dynamic-Circuit Qubit Reuse and Grover-Based Trajectory Optimization</strong> - Thet Htar Su, Shaswot Shresthamali, Masaaki Kondo - [[pdf]](https://arxiv.org/pdf/2509.16002)</summary>

**Abstract:** A fully quantum reinforcement learning framework is developed that integrates a quantum Markov decision process, dynamic circuit-based qubit reuse, and Grover's algorithm for trajectory optimization. The framework encodes states, actions, rewards, and transitions entirely within the quantum domain, enabling parallel exploration of state-action sequences through superposition and eliminating classical subroutines. Dynamic circuit operations, including mid-circuit measurement and reset, allow reuse of the same physical qubits across multiple agent-environment interactions, reducing qubit requirements from 7*T to 7 for T time steps while preserving logical continuity. Quantum arithmetic computes trajectory returns, and Grover's search is applied to the superposition of these evaluated trajectories to amplify the probability of measuring those with the highest return, thereby accelerating the identification of the optimal policy. Simulations demonstrate that the dynamic-circuit-based implementation preserves trajectory fidelity while reducing qubit usage by 66 percent relative to the static design. Experimental deployment on IBM Heron-class quantum hardware confirms that the framework operates within the constraints of current quantum processors and validates the feasibility of fully quantum multi-step reinforcement learning under noisy intermediate-scale quantum conditions. This framework advances the scalability and practical application of quantum reinforcement learning for large-scale sequential decision-making tasks.

**arXiv ID:** 2509.16002
</details>

<details>
<summary><strong>Revealing Human Internal Attention Patterns from Gameplay Analysis for Reinforcement Learning</strong> - Henrik Krauss, Takehisa Yairi - [[pdf]](https://arxiv.org/pdf/2504.11118)</summary>

**Abstract:** This study introduces a novel method for revealing human internal attention patterns from gameplay data alone, leveraging offline attention techniques from reinforcement learning (RL). We propose contextualized, task-relevant (CTR) attention networks, which generate attention maps from both human and RL agent gameplay in Atari environments. To evaluate whether the human CTR maps reveal internal attention, we validate our model by quantitative and qualitative comparison to the agent maps as well as to a temporally integrated overt attention (TIOA) model based on human eye-tracking data. Our results show that human CTR maps are more sparse than the agent ones and align better with the TIOA maps. Following a qualitative visual comparison we conclude that they likely capture patterns of internal attention. As a further application, we use these maps to guide RL agents, finding that human internal attention-guided agents achieve slightly improved and more stable learning compared to baselines. This work advances the understanding of human-agent attention differences and provides a new approach for extracting and validating internal attention from behavioral data.

**arXiv ID:** 2504.11118
</details>

<details>
<summary><strong>PAC Apprenticeship Learning with Bayesian Active Inverse Reinforcement Learning</strong> - Ondrej Bajgar, Dewi S.W. Gould, Jonathon Liu, Alessandro Abate, Konstantinos Gatsis, Michael A. Osborne - [[pdf]](https://arxiv.org/pdf/2508.03693)</summary>

**Abstract:** As AI systems become increasingly autonomous, reliably aligning their decision-making with human preferences is essential. Inverse reinforcement learning (IRL) offers a promising approach to infer preferences from demonstrations. These preferences can then be used to produce an apprentice policy that performs well on the demonstrated task. However, in domains like autonomous driving or robotics, where errors can have serious consequences, we need not just good average performance but reliable policies with formal guarantees -- yet obtaining sufficient human demonstrations for reliability guarantees can be costly. Active IRL addresses this challenge by strategically selecting the most informative scenarios for human demonstration. We introduce PAC-EIG, an information-theoretic acquisition function that directly targets probably-approximately-correct (PAC) guarantees for the learned policy -- providing the first such theoretical guarantee for active IRL with noisy expert demonstrations. Our method maximises information gain about the regret of the apprentice policy, efficiently identifying states requiring further demonstration. We also present Reward-EIG as an alternative when learning the reward itself is the primary objective. Focusing on finite state-action spaces, we prove convergence bounds, illustrate failure modes of prior heuristic methods, and demonstrate our method's advantages experimentally.

**arXiv ID:** 2508.03693
</details>

<details>
<summary><strong>cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning</strong> - Maksim Kolodiazhnyi, Denis Tarasov, Dmitrii Zhemchuzhnikov, Alexander Nikulin, Ilya Zisman, Anna Vorontsova, Anton Konushin, Vladislav Kurenkov, Danila Rukhovich - [[pdf]](https://arxiv.org/pdf/2505.22914)</summary>

**Abstract:** Computer-Aided Design (CAD) plays a central role in engineering and manufacturing, making it possible to create precise and editable 3D models. Using a variety of sensor or user-provided data as inputs for CAD reconstruction can democratize access to design applications. However, existing methods typically focus on a single input modality, such as point clouds, images, or text, which limits their generalizability and robustness. Leveraging recent advances in vision-language models (VLM), we propose a multi-modal CAD reconstruction model that simultaneously processes all three input modalities. Inspired by large language model (LLM) training paradigms, we adopt a two-stage pipeline: supervised fine-tuning (SFT) on large-scale procedurally generated data, followed by reinforcement learning (RL) fine-tuning using online feedback, obtained programatically. Furthermore, we are the first to explore RL fine-tuning of LLMs for CAD tasks demonstrating that online RL algorithms such as Group Relative Preference Optimization (GRPO) outperform offline alternatives. In the DeepCAD benchmark, our SFT model outperforms existing single-modal approaches in all three input modalities simultaneously. More importantly, after RL fine-tuning, cadrille sets new state-of-the-art on three challenging datasets, including a real-world one.

**arXiv ID:** 2505.22914
</details>

<details>
<summary><strong>Embodied Arena: A Comprehensive, Unified, and Evolving Evaluation Platform for Embodied AI</strong> - Fei Ni, Min Zhang, Pengyi Li, Yifu Yuan, Lingfeng Zhang, Yuecheng Liu, Peilong Han, Longxin Kou, Shaojin Ma, Jinbin Qiao, David Gamaliel Arcos Bravo, Yuening Wang, Xiao Hu, Zhanguang Zhang, Xianze Yao, Yutong Li, Zhao Zhang, Ying Wen, Ying-Cong Chen, Xiaodan Liang, Liang Lin, Bin He, Haitham Bou-Ammar, He Wang, Huazhe Xu, Jiankang Deng, Shan Luo, Shuqiang Jiang, Wei Pan, Yang Gao, Stefanos Zafeiriou, Jan Peters, Yuzheng Zhuang, Yingxue Zhang, Yan Zheng, Hongyao Tang, Jianye Hao - [[pdf]](https://arxiv.org/pdf/2509.15273)</summary>

**Abstract:** Embodied AI development significantly lags behind large foundation models due to three critical challenges: (1) lack of systematic understanding of core capabilities needed for Embodied AI, making research lack clear objectives; (2) absence of unified and standardized evaluation systems, rendering cross-benchmark evaluation infeasible; and (3) underdeveloped automated and scalable acquisition methods for embodied data, creating critical bottlenecks for model scaling. To address these obstacles, we present Embodied Arena, a comprehensive, unified, and evolving evaluation platform for Embodied AI. Our platform establishes a systematic embodied capability taxonomy spanning three levels (perception, reasoning, task execution), seven core capabilities, and 25 fine-grained dimensions, enabling unified evaluation with systematic research objectives. We introduce a standardized evaluation system built upon unified infrastructure supporting flexible integration of 22 diverse benchmarks across three domains (2D/3D Embodied Q&A, Navigation, Task Planning) and 30+ advanced models from 20+ worldwide institutes. Additionally, we develop a novel LLM-driven automated generation pipeline ensuring scalable embodied evaluation data with continuous evolution for diversity and comprehensiveness. Embodied Arena publishes three real-time leaderboards (Embodied Q&A, Navigation, Task Planning) with dual perspectives (benchmark view and capability view), providing comprehensive overviews of advanced model capabilities. Especially, we present nine findings summarized from the evaluation results on the leaderboards of Embodied Arena. This helps to establish clear research veins and pinpoint critical research problems, thereby driving forward progress in the field of Embodied AI.

**arXiv ID:** 2509.15273
</details>

<details>
<summary><strong>PRIMT: Preference-based Reinforcement Learning with Multimodal Feedback and Trajectory Synthesis from Foundation Models</strong> - Ruiqi Wang, Dezhong Zhao, Ziqin Yuan, Tianyu Shao, Guohua Chen, Dominic Kao, Sungeun Hong, Byung-Cheol Min - [[pdf]](https://arxiv.org/pdf/2509.15607)</summary>

**Abstract:** Preference-based reinforcement learning (PbRL) has emerged as a promising paradigm for teaching robots complex behaviors without reward engineering. However, its effectiveness is often limited by two critical challenges: the reliance on extensive human input and the inherent difficulties in resolving query ambiguity and credit assignment during reward learning. In this paper, we introduce PRIMT, a PbRL framework designed to overcome these challenges by leveraging foundation models (FMs) for multimodal synthetic feedback and trajectory synthesis. Unlike prior approaches that rely on single-modality FM evaluations, PRIMT employs a hierarchical neuro-symbolic fusion strategy, integrating the complementary strengths of large language models and vision-language models in evaluating robot behaviors for more reliable and comprehensive feedback. PRIMT also incorporates foresight trajectory generation, which reduces early-stage query ambiguity by warm-starting the trajectory buffer with bootstrapped samples, and hindsight trajectory augmentation, which enables counterfactual reasoning with a causal auxiliary loss to improve credit assignment. We evaluate PRIMT on 2 locomotion and 6 manipulation tasks on various benchmarks, demonstrating superior performance over FM-based and scripted baselines.

**arXiv ID:** 2509.15607
</details>

<details>
<summary><strong>Reward Evolution with Graph-of-Thoughts: A Bi-Level Language Model Framework for Reinforcement Learning</strong> - Changwei Yao, Xinzi Liu, Chen Li, Marios Savvides - [[pdf]](https://arxiv.org/pdf/2509.16136)</summary>

**Abstract:** Designing effective reward functions remains a major challenge in reinforcement learning (RL), often requiring considerable human expertise and iterative refinement. Recent advances leverage Large Language Models (LLMs) for automated reward design, but these approaches are limited by hallucinations, reliance on human feedback, and challenges with handling complex, multi-step tasks. In this work, we introduce Reward Evolution with Graph-of-Thoughts (RE-GoT), a novel bi-level framework that enhances LLMs with structured graph-based reasoning and integrates Visual Language Models (VLMs) for automated rollout evaluation. RE-GoT first decomposes tasks into text-attributed graphs, enabling comprehensive analysis and reward function generation, and then iteratively refines rewards using visual feedback from VLMs without human intervention. Extensive experiments on 10 RoboGen and 4 ManiSkill2 tasks demonstrate that RE-GoT consistently outperforms existing LLM-based baselines. On RoboGen, our method improves average task success rates by 32.25%, with notable gains on complex multi-step tasks. On ManiSkill2, RE-GoT achieves an average success rate of 93.73% across four diverse manipulation tasks, significantly surpassing prior LLM-based approaches and even exceeding expert-designed rewards. Our results indicate that combining LLMs and VLMs with graph-of-thoughts reasoning provides a scalable and effective solution for autonomous reward evolution in RL.

**arXiv ID:** 2509.16136
</details>

<details>
<summary><strong>How Good are Foundation Models in Step-by-Step Embodied Reasoning?</strong> - Dinura Dissanayake, Ahmed Heakl, Omkar Thawakar, Noor Ahsan, Ritesh Thawkar, Ketan More, Jean Lahoud, Rao Anwer, Hisham Cholakkal, Ivan Laptev, Fahad Shahbaz Khan, Salman Khan - [[pdf]](https://arxiv.org/pdf/2509.15293)</summary>

**Abstract:** Embodied agents operating in the physical world must make decisions that are not only effective but also safe, spatially coherent, and grounded in context. While recent advances in large multimodal models (LMMs) have shown promising capabilities in visual understanding and language generation, their ability to perform structured reasoning for real-world embodied tasks remains underexplored. In this work, we aim to understand how well foundation models can perform step-by-step reasoning in embodied environments. To this end, we propose the Foundation Model Embodied Reasoning (FoMER) benchmark, designed to evaluate the reasoning capabilities of LMMs in complex embodied decision-making scenarios. Our benchmark spans a diverse set of tasks that require agents to interpret multimodal observations, reason about physical constraints and safety, and generate valid next actions in natural language. We present (i) a large-scale, curated suite of embodied reasoning tasks, (ii) a novel evaluation framework that disentangles perceptual grounding from action reasoning, and (iii) empirical analysis of several leading LMMs under this setting. Our benchmark includes over 1.1k samples with detailed step-by-step reasoning across 10 tasks and 8 embodiments, covering three different robot types. Our results highlight both the potential and current limitations of LMMs in embodied reasoning, pointing towards key challenges and opportunities for future research in robot intelligence. Our data and code will be made publicly available.

**arXiv ID:** 2509.15293
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
