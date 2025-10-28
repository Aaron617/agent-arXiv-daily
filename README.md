# Agent arXiv Daily

**Last Updated:** 2025-10-28 02:44:46

**Total Papers:** 75

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
<summary><strong>Out-of-Distribution Detection for Safety Assurance of AI and Autonomous Systems</strong> - Victoria J. Hodge, Colin Paterson, Ibrahim Habli - [[pdf]](https://arxiv.org/pdf/2510.21254)</summary>

**Abstract:** The operational capabilities and application domains of AI-enabled autonomous systems have expanded significantly in recent years due to advances in robotics and machine learning (ML). Demonstrating the safety of autonomous systems rigorously is critical for their responsible adoption but it is challenging as it requires robust methodologies that can handle novel and uncertain situations throughout the system lifecycle, including detecting out-of-distribution (OoD) data. Thus, OOD detection is receiving increased attention from the research, development and safety engineering communities. This comprehensive review analyses OOD detection techniques within the context of safety assurance for autonomous systems, in particular in safety-critical domains. We begin by defining the relevant concepts, investigating what causes OOD and exploring the factors which make the safety assurance of autonomous systems and OOD detection challenging. Our review identifies a range of techniques which can be used throughout the ML development lifecycle and we suggest areas within the lifecycle in which they may be used to support safety assurance arguments. We discuss a number of caveats that system and safety engineers must be aware of when integrating OOD detection into system lifecycles. We conclude by outlining the challenges and future work necessary for the safe development and operation of autonomous systems across a range of domains and applications.

**arXiv ID:** 2510.21254
</details>

<details>
<summary><strong>AgentArcEval: An Architecture Evaluation Method for Foundation Model based Agents</strong> - Qinghua Lu, Dehai Zhao, Yue Liu, Hao Zhang, Liming Zhu, Xiwei Xu, Angela Shi, Tristan Tan, Rick Kazman - [[pdf]](https://arxiv.org/pdf/2510.21031)</summary>

**Abstract:** The emergence of foundation models (FMs) has enabled the development of highly capable and autonomous agents, unlocking new application opportunities across a wide range of domains. Evaluating the architecture of agents is particularly important as the architectural decisions significantly impact the quality attributes of agents given their unique characteristics, including compound architecture, autonomous and non-deterministic behaviour, and continuous evolution. However, these traditional methods fall short in addressing the evaluation needs of agent architecture due to the unique characteristics of these agents. Therefore, in this paper, we present AgentArcEval, a novel agent architecture evaluation method designed specially to address the complexities of FM-based agent architecture and its evaluation. Moreover, we present a catalogue of agent-specific general scenarios, which serves as a guide for generating concrete scenarios to design and evaluate the agent architecture. We demonstrate the usefulness of AgentArcEval and the catalogue through a case study on the architecture evaluation of a real-world tax copilot, named Luna.

**arXiv ID:** 2510.21031
</details>

<details>
<summary><strong>REMONI: An Autonomous System Integrating Wearables and Multimodal Large Language Models for Enhanced Remote Health Monitoring</strong> - Thanh Cong Ho, Farah Kharrat, Abderrazek Abid, Fakhri Karray - [[pdf]](https://arxiv.org/pdf/2510.21445)</summary>

**Abstract:** With the widespread adoption of wearable devices in our daily lives, the demand and appeal for remote patient monitoring have significantly increased. Most research in this field has concentrated on collecting sensor data, visualizing it, and analyzing it to detect anomalies in specific diseases such as diabetes, heart disease and depression. However, this domain has a notable gap in the aspect of human-machine interaction. This paper proposes REMONI, an autonomous REmote health MONItoring system that integrates multimodal large language models (MLLMs), the Internet of Things (IoT), and wearable devices. The system automatically and continuously collects vital signs, accelerometer data from a special wearable (such as a smartwatch), and visual data in patient video clips collected from cameras. This data is processed by an anomaly detection module, which includes a fall detection model and algorithms to identify and alert caregivers of the patient's emergency conditions. A distinctive feature of our proposed system is the natural language processing component, developed with MLLMs capable of detecting and recognizing a patient's activity and emotion while responding to healthcare worker's inquiries. Additionally, prompt engineering is employed to integrate all patient information seamlessly. As a result, doctors and nurses can access real-time vital signs and the patient's current state and mood by interacting with an intelligent agent through a user-friendly web application. Our experiments demonstrate that our system is implementable and scalable for real-life scenarios, potentially reducing the workload of medical professionals and healthcare costs. A full-fledged prototype illustrating the functionalities of the system has been developed and being tested to demonstrate the robustness of its various capabilities.

**arXiv ID:** 2510.21445
</details>

<details>
<summary><strong>HugAgent: Evaluating LLMs in Simulating Individual-Level Human Reasoning on Open-Ended Tasks</strong> - Chance Jiajie Li, Zhenze Mo, Yuhan Tang, Ao Qu, Jiayi Wu, Kaiya Ivy Zhao, Yulu Gan, Jie Fan, Jiangbo Yu, Hang Jiang, Paul Pu Liang, Jinhua Zhao, Luis Alberto Alonso Pastor, Kent Larson - [[pdf]](https://arxiv.org/pdf/2510.15144)</summary>

**Abstract:** Simulating human reasoning in open-ended tasks has been a long-standing aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), a benchmark for average-to-individual reasoning adaptation. The task is to predict how a specific person would reason and update their beliefs in novel scenarios, given partial evidence of their past views. HugAgent adopts a dual-track design: a synthetic track for scale and systematic stress tests, and a human track for ecologically valid, "out-loud" reasoning data. This design enables scalable, reproducible evaluation of intra-agent fidelity: whether models can capture not just what people believe, but how their reasoning evolves. Experiments with state-of-the-art LLMs reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. Our benchmark and chatbot are open-sourced as HugAgent (this https URL) and TraceYourThinking (this https URL).

**arXiv ID:** 2510.15144
</details>

<details>
<summary><strong>Intrinsic Goals for Autonomous Agents: Model-Based Exploration in Virtual Zebrafish Predicts Ethological Behavior and Whole-Brain Dynamics</strong> - Reece Keller, Alyn Kirsch, Felix Pei, Xaq Pitkow, Leo Kozachkov, Aran Nayebi - [[pdf]](https://arxiv.org/pdf/2506.00138)</summary>

**Abstract:** Autonomy is a hallmark of animal intelligence, enabling adaptive and intelligent behavior in complex environments without relying on external reward or task structure. Existing reinforcement learning approaches to exploration in reward-free environments, including a class of methods known as model-based intrinsic motivation, exhibit inconsistent exploration patterns and do not converge to an exploratory policy, thus failing to capture robust autonomous behaviors observed in animals. Moreover, systems neuroscience has largely overlooked the neural basis of autonomy, focusing instead on experimental paradigms where animals are motivated by external reward rather than engaging in ethological, naturalistic and task-independent behavior. To bridge these gaps, we introduce a novel model-based intrinsic drive explicitly designed after the principles of autonomous exploration in animals. Our method (3M-Progress) achieves animal-like exploration by tracking divergence between an online world model and a fixed prior learned from an ecological niche. To the best of our knowledge, we introduce the first autonomous embodied agent that predicts brain data entirely from self-supervised optimization of an intrinsic goal -- without any behavioral or neural training data -- demonstrating that 3M-Progress agents capture the explainable variance in behavioral patterns and whole-brain neural-glial dynamics recorded from autonomously behaving larval zebrafish, thereby providing the first goal-driven, population-level model of neural-glial computation. Our findings establish a computational framework connecting model-based intrinsic motivation to naturalistic behavior, providing a foundation for building artificial agents with animal-like autonomy.

**arXiv ID:** 2506.00138
</details>

<details>
<summary><strong>Marcel: A Lightweight and Open-Source Conversational Agent for University Student Support</strong> - Jan Trienes, Anastasiia Derzhanskaia, Roland Schwarzkopf, Markus Mühling, Jörg Schlötterer, Christin Seifert - [[pdf]](https://arxiv.org/pdf/2507.13937)</summary>

**Abstract:** We present Marcel, a lightweight and open-source conversational agent designed to support prospective students with admission-related inquiries. The system aims to provide fast and personalized responses, while reducing workload of university staff. We employ retrieval-augmented generation to ground answers in university resources and to provide users with verifiable, contextually relevant information. We introduce a Frequently Asked Question (FAQ) retriever that maps user questions to knowledge-base entries, which allows administrators to steer retrieval, and improves over standard dense/hybrid retrieval strategies. The system is engineered for easy deployment in resource-constrained academic settings. We detail the system architecture, provide a technical evaluation of its components, and report insights from a real-world deployment.

**arXiv ID:** 2507.13937
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>DAO-AI: Evaluating Collective Decision-Making through Agentic AI in Decentralized Governance</strong> - Chunghyun Han, Alfio Gliozzo, Junkyu Lee, Agostino Capponi - [[pdf]](https://arxiv.org/pdf/2510.21117)</summary>

**Abstract:** This paper presents a first empirical study of agentic AI as autonomous decision-makers in decentralized governance. Using more than 3K proposals from major protocols, we build an agentic AI voter that interprets proposal contexts, retrieves historical deliberation data, and independently determines its voting position. The agent operates within a realistic financial simulation environment grounded in verifiable blockchain data, implemented through a modular composable program (MCP) workflow that defines data flow and tool usage via Agentics framework. We evaluate how closely the agent's decisions align with the human and token-weighted outcomes, uncovering strong alignments measured by carefully designed evaluation metrics. Our findings demonstrate that agentic AI can augment collective decision-making by producing interpretable, auditable, and empirically grounded signals in realistic DAO governance settings. The study contributes to the design of explainable and economically rigorous AI agents for decentralized financial systems.

**arXiv ID:** 2510.21117
</details>

<details>
<summary><strong>OutboundEval: A Dual-Dimensional Benchmark for Expert-Level Intelligent Outbound Evaluation of Xbench's Professional-Aligned Series</strong> - Pengyu Xu, Shijia Li, Ao Sun, Feng Zhang, Yahan Li, Bo Wu, Zhanyu Ma, Jiguo Li, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Rui Wang, Yang Liu, Xiaobo Hu, Fan Yang, Jia Zheng, Guanghua Yao - [[pdf]](https://arxiv.org/pdf/2510.21244)</summary>

**Abstract:** We propose OutboundEval, a comprehensive benchmark for evaluating large language models (LLMs) in expert-level intelligent outbound calling scenarios. Unlike existing methods that suffer from three key limitations - insufficient dataset diversity and category coverage, unrealistic user simulation, and inaccurate evaluation metrics - OutboundEval addresses these issues through a structured framework. First, we design a benchmark spanning six major business domains and 30 representative sub-scenarios, each with scenario-specific process decomposition, weighted scoring, and domain-adaptive metrics. Second, we develop a large-model-driven User Simulator that generates diverse, persona-rich virtual users with realistic behaviors, emotional variability, and communication styles, providing a controlled yet authentic testing environment. Third, we introduce a dynamic evaluation method that adapts to task variations, integrating automated and human-in-the-loop assessment to measure task execution accuracy, professional knowledge application, adaptability, and user experience quality. Experiments on 12 state-of-the-art LLMs reveal distinct trade-offs between expert-level task completion and interaction fluency, offering practical insights for building reliable, human-like outbound AI systems. OutboundEval establishes a practical, extensible, and domain-oriented standard for benchmarking LLMs in professional applications.

**arXiv ID:** 2510.21244
</details>

<details>
<summary><strong>Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine</strong> - Wenyi Wang, Piotr Piękos, Li Nanbo, Firas Laakom, Yimeng Chen, Mateusz Ostaszewski, Mingchen Zhuge, Jürgen Schmidhuber - [[pdf]](https://arxiv.org/pdf/2510.21614)</summary>

**Abstract:** Recent studies operationalize self-improvement through coding agents that edit their own codebases. They grow a tree of self-modifications through expansion strategies that favor higher software engineering benchmark performance, assuming that this implies more promising subsequent self-modifications. However, we identify a mismatch between the agent's self-improvement potential (metaproductivity) and its coding benchmark performance, namely the Metaproductivity-Performance Mismatch. Inspired by Huxley's concept of clade, we propose a metric ($\mathrm{CMP}$) that aggregates the benchmark performances of the descendants of an agent as an indicator of its potential for self-improvement. We show that, in our self-improving coding agent development setting, access to the true $\mathrm{CMP}$ is sufficient to simulate how the Gödel Machine would behave under certain assumptions. We introduce the Huxley-Gödel Machine (HGM), which, by estimating $\mathrm{CMP}$ and using it as guidance, searches the tree of self-modifications. On SWE-bench Verified and Polyglot, HGM outperforms prior self-improving coding agent development methods while using less wall-clock time. Last but not least, HGM demonstrates strong transfer to other coding datasets and large language models. The agent optimized by HGM on SWE-bench Verified with GPT-5-mini and evaluated on SWE-bench Lite with GPT-5 achieves human-level performance, matching the best officially checked results of human-engineered coding agents. Our code is available at this https URL.

**arXiv ID:** 2510.21614
</details>

<details>
<summary><strong>AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite</strong> - Jonathan Bragg, Mike D'Arcy, Nishant Balepur, Dan Bareket, Bhavana Dalvi, Sergey Feldman, Dany Haddad, Jena D. Hwang, Peter Jansen, Varsha Kishore, Bodhisattwa Prasad Majumder, Aakanksha Naik, Sigal Rahamimov, Kyle Richardson, Amanpreet Singh, Harshit Surana, Aryeh Tiktinsky, Rosni Vasu, Guy Wiener, Chloe Anastasiades, Stefan Candra, Jason Dunkelberger, Dan Emery, Rob Evans, Malachi Hamada, Regan Huff, Rodney Kinney, Matt Latzke, Jaron Lochner, Ruben Lozano-Aguilera, Cecile Nguyen, Smita Rao, Amber Tanaka, Brooke Vlahos, Peter Clark, Doug Downey, Yoav Goldberg, Ashish Sabharwal, Daniel S. Weld - [[pdf]](https://arxiv.org/pdf/2510.21652)</summary>

**Abstract:** AI agents hold the potential to revolutionize scientific productivity by automating literature reviews, replicating experiments, analyzing data, and even proposing new directions of inquiry; indeed, there are now many such agents, ranging from general-purpose "deep research" systems to specialized science-specific agents, such as AI Scientist and AIGS. Rigorous evaluation of these agents is critical for progress. Yet existing benchmarks fall short on several fronts: they (1) fail to provide holistic, product-informed measures of real-world use cases such as science research; (2) lack reproducible agent tools necessary for a controlled comparison of core agentic capabilities; (3) do not account for confounding variables such as model cost and tool access; (4) do not provide standardized interfaces for quick agent prototyping and evaluation; and (5) lack comprehensive baseline agents necessary to identify true advances. In response, we define principles and tooling for more rigorously benchmarking agents. Using these, we present AstaBench, a suite that provides the first holistic measure of agentic ability to perform scientific research, comprising 2400+ problems spanning the entire scientific discovery process and multiple scientific domains, and including many problems inspired by actual user requests to deployed Asta agents. Our suite comes with the first scientific research environment with production-grade search tools that enable controlled, reproducible evaluation, better accounting for confounders. Alongside, we provide a comprehensive suite of nine science-optimized classes of Asta agents and numerous baselines. Our extensive evaluation of 57 agents across 22 agent classes reveals several interesting findings, most importantly that despite meaningful progress on certain individual aspects, AI remains far from solving the challenge of science research assistance.

**arXiv ID:** 2510.21652
</details>

<details>
<summary><strong>Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People</strong> - Gabriel Grand, Valerio Pepe, Jacob Andreas, Joshua B. Tenenbaum - [[pdf]](https://arxiv.org/pdf/2510.20886)</summary>

**Abstract:** Many high-stakes applications of AI require forming data-driven hypotheses and making targeted guesses; e.g., in scientific and diagnostic settings. Given limited resources, to what extent do agents based on language models (LMs) act rationally? We develop methods to benchmark and enhance agentic information-seeking, drawing on insights from human behavior. First, we introduce a strategic decision-oriented dialogue task called Collaborative Battleship, in which a partially-informed Captain must balance exploration (asking questions) and action (taking shots), while a fully-informed Spotter must provide accurate answers under an information bottleneck. Compared to human players (N=42), we find that LM agents struggle to ground answers in context, generate informative questions, and select high-value actions. Next, to address these gaps, we develop novel Monte Carlo inference strategies for LMs based on principles from Bayesian Experimental Design (BED). For Spotter agents, our approach boosts accuracy by up to 14.7% absolute over LM-only baselines; for Captain agents, it raises expected information gain (EIG) by up to 0.227 bits (94.2% of the achievable noise ceiling). Combined, these components yield sharper targeting (+0.303-0.374 F1), and enable weaker LMs, such as Llama-4-Scout, to outperform both humans (8% -> 82% win rate) and frontier models (0% -> 67% win rate vs. GPT-5) at ~1% of GPT-5's cost. We replicate these findings on Guess Who? where our methods significantly boost accuracy (+28.3-42.4 p.p.), demonstrating their general applicability for building rational information-seeking agents.

**arXiv ID:** 2510.20886
</details>

<details>
<summary><strong>An Experimental Study of Trojan Vulnerabilities in UAV Autonomous Landing</strong> - Reza Ahmari, Ahmad Mohammadi, Vahid Hemmati, Mohammed Mynuddin, Mahmoud Nabil Mahmoud, Parham Kebria, Abdollah Homaifar, Mehrdad Saif - [[pdf]](https://arxiv.org/pdf/2510.20932)</summary>

**Abstract:** This study investigates the vulnerabilities of autonomous navigation and landing systems in Urban Air Mobility (UAM) vehicles. Specifically, it focuses on Trojan attacks that target deep learning models, such as Convolutional Neural Networks (CNNs). Trojan attacks work by embedding covert triggers within a model's training data. These triggers cause specific failures under certain conditions, while the model continues to perform normally in other situations. We assessed the vulnerability of Urban Autonomous Aerial Vehicles (UAAVs) using the DroNet framework. Our experiments showed a significant drop in accuracy, from 96.4% on clean data to 73.3% on data triggered by Trojan attacks. To conduct this study, we collected a custom dataset and trained models to simulate real-world conditions. We also developed an evaluation framework designed to identify Trojan-infected models. This work demonstrates the potential security risks posed by Trojan attacks and lays the groundwork for future research on enhancing the resilience of UAM systems.

**arXiv ID:** 2510.20932
</details>

<details>
<summary><strong>Securing AI Agent Execution</strong> - Christoph Bühler, Matteo Biagiola, Luca Di Grazia, Guido Salvaneschi - [[pdf]](https://arxiv.org/pdf/2510.21236)</summary>

**Abstract:** Large Language Models (LLMs) have evolved into AI agents that interact with external tools and environments to perform complex tasks. The Model Context Protocol (MCP) has become the de facto standard for connecting agents with such resources, but security has lagged behind: thousands of MCP servers execute with unrestricted access to host systems, creating a broad attack surface. In this paper, we introduce AgentBound, the first access control framework for MCP servers. AgentBound combines a declarative policy mechanism, inspired by the Android permission model, with a policy enforcement engine that contains malicious behavior without requiring MCP server modifications. We build a dataset containing the 296 most popular MCP servers, and show that access control policies can be generated automatically from source code with 80.9% accuracy. We also show that AgentBound blocks the majority of security threats in several malicious MCP servers, and that policy enforcement engine introduces negligible overhead. Our contributions provide developers and project managers with a practical foundation for securing MCP servers while maintaining productivity, enabling researchers and tool builders to explore new directions for declarative access control and MCP security.

**arXiv ID:** 2510.21236
</details>

<details>
<summary><strong>Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty</strong> - Meera Hahn, Wenjun Zeng, Nithish Kannen, Rich Galt, Kartikeya Badola, Been Kim, Zi Wang - [[pdf]](https://arxiv.org/pdf/2412.06771)</summary>

**Abstract:** User prompts for generative AI models are often underspecified, leading to a misalignment between the user intent and models' understanding. As a result, users commonly have to painstakingly refine their prompts. We study this alignment problem in text-to-image (T2I) generation and propose a prototype for proactive T2I agents equipped with an interface to (1) actively ask clarification questions when uncertain, and (2) present their uncertainty about user intent as an understandable and editable belief graph. We build simple prototypes for such agents and propose a new scalable and automated evaluation approach using two agents, one with a ground truth intent (an image) while the other tries to ask as few questions as possible to align with the ground truth. We experiment over three image-text datasets: ImageInWords (Garg et al., 2024), COCO (Lin et al., 2014) and DesignBench, a benchmark we curated with strong artistic and design elements. Experiments over the three datasets demonstrate the proposed T2I agents' ability to ask informative questions and elicit crucial information to achieve successful alignment with at least 2 times higher VQAScore (Lin et al., 2024) than the standard T2I generation. Moreover, we conducted human studies and observed that at least 90% of human subjects found these agents and their belief graphs helpful for their T2I workflow, highlighting the effectiveness of our approach. Code and DesignBench can be found at this https URL.

**arXiv ID:** 2412.06771
</details>

<details>
<summary><strong>Can Agents Fix Agent Issues?</strong> - Alfin Wijaya Rahardja, Junwei Liu, Weitong Chen, Zhenpeng Chen, Yiling Lou - [[pdf]](https://arxiv.org/pdf/2505.20749)</summary>

**Abstract:** LLM-based agent systems are emerging as a new software paradigm and have been widely adopted across diverse domains such as medicine, robotics, and programming. However, maintaining these systems requires substantial effort, as they are inevitably prone to bugs and continually evolve to meet changing external requirements. Therefore, automatically resolving agent issues (i.e., bug reports or feature requests) is a crucial and challenging task. While recent software engineering (SE) agents (e.g., SWE-agent) have shown promise in addressing issues in traditional software systems, it remains unclear how effectively they can resolve real-world issues in agent systems, which differ significantly from traditional software. To fill this gap, we first manually analyze 201 real-world agent issues and identify common categories of agent issues. We then spend 500 person-hours constructing AgentIssue-Bench, a reproducible benchmark comprising 50 agent issue resolution tasks (each with an executable environment and failure-triggering tests). We further evaluate state-of-the-art SE agents on AgentIssue-Bench and reveal their limited effectiveness (i.e., with only 0.67% - 4.67% resolution rates). These results underscore the unique challenges of maintaining agent systems compared to traditional software, highlighting the need for further research to develop advanced SE agents for resolving agent issues. Data and code are available at this https URL.

**arXiv ID:** 2505.20749
</details>

<details>
<summary><strong>Towards Self-Evolving Benchmarks: Synthesizing Agent Trajectories via Test-Time Exploration under Validate-by-Reproduce Paradigm</strong> - Dadi Guo, Tianyi Zhou, Dongrui Liu, Chen Qian, Qihan Ren, Shuai Shao, Zhiyuan Fan, Yi R. Fung, Kun Wang, Linfeng Zhang, Jing Shao - [[pdf]](https://arxiv.org/pdf/2510.00415)</summary>

**Abstract:** Recent advances in large language models (LLMs) and agent system designs have empowered agents with unprecedented levels of capability. However, existing agent benchmarks are showing a trend of rapid ceiling-hitting by newly developed agents, making it difficult to meet the demands for evaluating agent abilities. To address this problem, we propose the Trajectory-based Validated-by-Reproducing Agent-benchmark Complexity Evolution (TRACE) framework. This framework takes an original task from an existing benchmark and encourages agents to freely explore and evolve it into a new task with higher difficulty while recording validatable agent trajectories. The framework proceeds in three stages: (1) evolutionary proposal mining, which provides task evolution proposals through preliminary exploration and divergent thinking; (2) problem formation and free exploration, where proposals are conceptualized into feasible problem candidates and the agents then explore them freely while recording their execution trajectories; and (3) multi-level validation, which ensures that the evolved tasks are accompanied by validatable and reproducible trajectories. Experiments on the GAIA benchmark demonstrate that the TRACE framework consistently enhances task complexity while improving the reliability of correctness through validatable execution trajectories. In addition, our framework can successfully adapt to and improve reasoning datasets represented by AIME-2024. This work marks a paradigm shift from static, manually curated benchmarks to dynamic, self-evolving evaluation systems, providing a sustainable and challenging runway for agent development

**arXiv ID:** 2510.00415
</details>

<details>
<summary><strong>Surfer 2: The Next Generation of Cross-Platform Computer Use Agents</strong> - Mathieu Andreux, Märt Bakler, Yanael Barbier, Hamza Benchekroun, Emilien Biré, Antoine Bonnet, Riaz Bordie, Nathan Bout, Matthias Brunel, Aleix Cambray, Pierre-Louis Cedoz, Antoine Chassang, Gautier Cloix, Ethan Connelly, Alexandra Constantinou, Ramzi De Coster, Hubert de la Jonquiere, Aurélien Delfosse, Maxime Delpit, Alexis Deprez, Augustin Derupti, Mathieu Diaz, Shannon D'Souza, Julie Dujardin, Abai Edmund, Michael Eickenberg, Armand Fatalot, Wissem Felissi, Isaac Herring, Xavier Koegler, Erwan Le Jumeau de Kergaradec, Aurélien Lac, Maxime Langevin, Corentin Lauverjat, Antonio Loison, Avshalom Manevich, Axel Moyal, Axel Nguyen Kerbel, Marinela Parovic, Julien Revelle, Guillaume Richard, Mats Richter, Ronan Riochet, María Santos, Romain Savidan, Laurent Sifre, Maxime Theillard, Marc Thibault, Ivan Valentini, Tony Wu, Laura Yie, Kai Yuan, Jevgenij Zubovskij - [[pdf]](https://arxiv.org/pdf/2510.19949)</summary>

**Abstract:** Building agents that generalize across web, desktop, and mobile environments remains an open challenge, as prior systems rely on environment-specific interfaces that limit cross-platform deployment. We introduce Surfer 2, a unified architecture operating purely from visual observations that achieves state-of-the-art performance across all three environments. Surfer 2 integrates hierarchical context management, decoupled planning and execution, and self-verification with adaptive recovery, enabling reliable operation over long task horizons. Our system achieves 97.1% accuracy on WebVoyager, 69.6% on WebArena, 60.1% on OSWorld, and 87.1% on AndroidWorld, outperforming all prior systems without task-specific fine-tuning. With multiple attempts, Surfer 2 exceeds human performance on all benchmarks. These results demonstrate that systematic orchestration amplifies foundation model capabilities and enables general-purpose computer control through visual interaction alone, while calling for a next-generation vision language model to achieve Pareto-optimal cost-efficiency.

**arXiv ID:** 2510.19949
</details>

<details>
<summary><strong>Scalable Principal-Agent Contract Design via Gradient-Based Optimization</strong> - Tomer Galanti, Aarya Bookseller, Korok Ray - [[pdf]](https://arxiv.org/pdf/2510.21177)</summary>

**Abstract:** We study a bilevel \emph{max-max} optimization framework for principal-agent contract design, in which a principal chooses incentives to maximize utility while anticipating the agent's best response. This problem, central to moral hazard and contract theory, underlies applications ranging from market design to delegated portfolio management, hedge fund fee structures, and executive compensation. While linear-quadratic models such as Holmstr"om-Milgrom admit closed-form solutions, realistic environments with nonlinear utilities, stochastic dynamics, or high-dimensional actions generally do not.
We introduce a generic algorithmic framework that removes this reliance on closed forms. Our method adapts modern machine learning techniques for bilevel optimization -- using implicit differentiation with conjugate gradients (CG) -- to compute hypergradients efficiently through Hessian-vector products, without ever forming or inverting Hessians. In benchmark CARA-Normal (Constant Absolute Risk Aversion with Gaussian distribution of uncertainty) environments, the approach recovers known analytical optima and converges reliably from random initialization. More broadly, because it is matrix-free, variance-reduced, and problem-agnostic, the framework extends naturally to complex nonlinear contracts where closed-form solutions are unavailable, such as sigmoidal wage schedules (logistic pay), relative-performance/tournament compensation with common shocks, multi-task contracts with vector actions and heterogeneous noise, and CARA-Poisson count models with $\mathbb{E}[X\mid a]=e^{a}$. This provides a new computational tool for contract design, enabling systematic study of models that have remained analytically intractable.

**arXiv ID:** 2510.21177
</details>

<details>
<summary><strong>Instance-Adaptive Hypothesis Tests with Heterogeneous Agents</strong> - Flora C. Shi, Martin J. Wainwright, Stephen Bates - [[pdf]](https://arxiv.org/pdf/2510.21178)</summary>

**Abstract:** We study hypothesis testing over a heterogeneous population of strategic agents with private information. Any single test applied uniformly across the population yields statistical error that is sub-optimal relative to the performance of an oracle given access to the private information. We show how it is possible to design menus of statistical contracts that pair type-optimal tests with payoff structures, inducing agents to self-select according to their private information. This separating menu elicits agent types and enables the principal to match the oracle performance even without a priori knowledge of the agent type. Our main result fully characterizes the collection of all separating menus that are instance-adaptive, matching oracle performance for an arbitrary population of heterogeneous agents. We identify designs where information elicitation is essentially costless, requiring negligible additional expense relative to a single-test benchmark, while improving statistical performance. Our work establishes a connection between proper scoring rules and menu design, showing how the structure of the hypothesis test constrains the elicitable information. Numerical examples illustrate the geometry of separating menus and the improvements they deliver in error trade-offs. Overall, our results connect statistical decision theory with mechanism design, demonstrating how heterogeneity and strategic participation can be harnessed to improve efficiency in hypothesis testing.

**arXiv ID:** 2510.21178
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law</strong> - Ilija Lichkovski, Alexander Müller, Mariam Ibrahim, Tiwai Mhundwa - [[pdf]](https://arxiv.org/pdf/2510.21524)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as agents in various contexts by providing tools at their disposal. However, LLM agents can exhibit unpredictable behaviors, including taking undesirable and/or unsafe actions. In order to measure the latent propensity of LLM agents for taking illegal actions under an EU legislative context, we introduce EU-Agent-Bench, a verifiable human-curated benchmark that evaluates an agent's alignment with EU legal norms in situations where benign user inputs could lead to unlawful actions. Our benchmark spans scenarios across several categories, including data protection, bias/discrimination, and scientific integrity, with each user request allowing for both compliant and non-compliant execution of the requested actions. Comparing the model's function calls against a rubric exhaustively supported by citations of the relevant legislature, we evaluate the legal compliance of frontier LLMs, and furthermore investigate the compliance effect of providing the relevant legislative excerpts in the agent's system prompt along with explicit instructions to comply. We release a public preview set for the research community, while holding out a private test set to prevent data contamination in evaluating upcoming models. We encourage future work extending agentic safety benchmarks to different legal jurisdictions and to multi-turn and multilingual interactions. We release our code on \href{this https URL}{this URL}.

**arXiv ID:** 2510.21524
</details>

<details>
<summary><strong>MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?</strong> - Yunxiang Zhang, Muhammad Khalifa, Shitanshu Bhushan, Grant D Murphy, Lajanugen Logeswaran, Jaekyeom Kim, Moontae Lee, Honglak Lee, Lu Wang - [[pdf]](https://arxiv.org/pdf/2504.09702)</summary>

**Abstract:** We introduce MLRC-Bench, a benchmark designed to quantify how effectively language agents can tackle challenging Machine Learning (ML) Research Competitions, with a focus on open research problems that demand novel methodologies. Unlike prior work, e.g., AI Scientist, which evaluates the end-to-end agentic pipeline by using LLM-as-a-judge, MLRC-Bench measures the key steps of proposing and implementing novel research methods and evaluates them with rigorous protocol and objective metrics. Our curated suite of 7 competition tasks reveals significant challenges for LLM agents. Even the best-performing tested agent (gemini-exp-1206 under MLAB) closes only 9.3% of the gap between baseline and top human participant scores. Furthermore, our analysis reveals a misalignment between the LLM-judged innovation and actual performance on cutting-edge ML research problems. MLRC-Bench is a dynamic benchmark, designed to grow with new ML competitions and encourage rigorous, objective evaluations of AI research capabilities. Our leaderboard and code are available at: this https URL

**arXiv ID:** 2504.09702
</details>

<details>
<summary><strong>TAI3: Testing Agent Integrity in Interpreting User Intent</strong> - Shiwei Feng, Xiangzhe Xu, Xuan Chen, Kaiyuan Zhang, Syed Yusuf Ahmed, Zian Su, Mingwei Zheng, Xiangyu Zhang - [[pdf]](https://arxiv.org/pdf/2506.07524)</summary>

**Abstract:** LLM agents are increasingly deployed to automate real-world tasks by invoking APIs through natural language instructions. While powerful, they often suffer from misinterpretation of user intent, leading to the agent's actions that diverge from the user's intended goal, especially as external toolkits evolve. Traditional software testing assumes structured inputs and thus falls short in handling the ambiguity of natural language. We introduce TAI3, an API-centric stress testing framework that systematically uncovers intent integrity violations in LLM agents. Unlike prior work focused on fixed benchmarks or adversarial inputs, TAI3 generates realistic tasks based on toolkits' documentation and applies targeted mutations to expose subtle agent errors while preserving user intent. To guide testing, we propose semantic partitioning, which organizes natural language tasks into meaningful categories based on toolkit API parameters and their equivalence classes. Within each partition, seed tasks are mutated and ranked by a lightweight predictor that estimates the likelihood of triggering agent errors. To enhance efficiency, TAI3 maintains a datatype-aware strategy memory that retrieves and adapts effective mutation patterns from past cases. Experiments on 80 toolkit APIs demonstrate that TAI3 effectively uncovers intent integrity violations, significantly outperforming baselines in both error-exposing rate and query efficiency. Moreover, TAI3 generalizes well to stronger target models using smaller LLMs for test generation, and adapts to evolving APIs across domains.

**arXiv ID:** 2506.07524
</details>

<details>
<summary><strong>DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents</strong> - Hao Li, Xiaogeng Liu, Hung-Chun Chiu, Dianqi Li, Ning Zhang, Chaowei Xiao - [[pdf]](https://arxiv.org/pdf/2506.12104)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly central to agentic systems due to their strong reasoning and planning capabilities. By interacting with external environments through predefined tools, these agents can carry out complex user tasks. Nonetheless, this interaction also introduces the risk of prompt injection attacks, where malicious inputs from external sources can mislead the agent's behavior, potentially resulting in economic loss, privacy leakage, or system compromise. System-level defenses have recently shown promise by enforcing static or predefined policies, but they still face two key challenges: the ability to dynamically update security rules and the need for memory stream isolation. To address these challenges, we propose DRIFT, a Dynamic Rule-based Isolation Framework for Trustworthy agentic systems, which enforces both control- and data-level constraints. A Secure Planner first constructs a minimal function trajectory and a JSON-schema-style parameter checklist for each function node based on the user query. A Dynamic Validator then monitors deviations from the original plan, assessing whether changes comply with privilege limitations and the user's intent. Finally, an Injection Isolator detects and masks any instructions that may conflict with the user query from the memory stream to mitigate long-term risks. We empirically validate the effectiveness of DRIFT on the AgentDojo and ASB benchmark, demonstrating its strong security performance while maintaining high utility across diverse models, showcasing both its robustness and adaptability. The code is released at this https URL.

**arXiv ID:** 2506.12104
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (18 papers)</h2></summary>

<details>
<summary><strong>Sketch2BIM: A Multi-Agent Human-AI Collaborative Pipeline to Convert Hand-Drawn Floor Plans to 3D BIM</strong> - Abir Khan Ratul, Sanjay Acharjee, Somin Park, Md Nazmus Sakib - [[pdf]](https://arxiv.org/pdf/2510.20838)</summary>

**Abstract:** This study introduces a human-in-the-loop pipeline that converts unscaled, hand-drawn floor plan sketches into semantically consistent 3D BIM models. The workflow leverages multimodal large language models (MLLMs) within a multi-agent framework, combining perceptual extraction, human feedback, schema validation, and automated BIM scripting. Initially, sketches are iteratively refined into a structured JSON layout of walls, doors, and windows. Later, these layouts are transformed into executable scripts that generate 3D BIM models. Experiments on ten diverse floor plans demonstrate strong convergence: openings (doors, windows) are captured with high reliability in the initial pass, while wall detection begins around 83% and achieves near-perfect alignment after a few feedback iterations. Across all categories, precision, recall, and F1 scores remain above 0.83, and geometric errors (RMSE, MAE) progressively decrease to zero through feedback corrections. This study demonstrates how MLLM-driven multi-agent reasoning can make BIM creation accessible to both experts and non-experts using only freehand sketches.

**arXiv ID:** 2510.20838
</details>

<details>
<summary><strong>From Questions to Queries: An AI-powered Multi-Agent Framework for Spatial Text-to-SQL</strong> - Ali Khosravi Kazazi, Zhenlong Li, M. Naser Lessani, Guido Cervone - [[pdf]](https://arxiv.org/pdf/2510.21045)</summary>

**Abstract:** The complexity of Structured Query Language (SQL) and the specialized nature of geospatial functions in tools like PostGIS present significant barriers to non-experts seeking to analyze spatial data. While Large Language Models (LLMs) offer promise for translating natural language into SQL (Text-to-SQL), single-agent approaches often struggle with the semantic and syntactic complexities of spatial queries. To address this, we propose a multi-agent framework designed to accurately translate natural language questions into spatial SQL queries. The framework integrates several innovative components, including a knowledge base with programmatic schema profiling and semantic enrichment, embeddings for context retrieval, and a collaborative multi-agent pipeline as its core. This pipeline comprises specialized agents for entity extraction, metadata retrieval, query logic formulation, SQL generation, and a review agent that performs programmatic and semantic validation of the generated SQL to ensure correctness (self-verification). We evaluate our system using both the non-spatial KaggleDBQA benchmark and a new, comprehensive SpatialQueryQA benchmark that includes diverse geometry types, predicates, and three levels of query complexity. On KaggleDBQA, the system achieved an overall accuracy of 81.2% (221 out of 272 questions) after the review agent's review and corrections. For spatial queries, the system achieved an overall accuracy of 87.7% (79 out of 90 questions), compared with 76.7% without the review agent. Beyond accuracy, results also show that in some instances the system generates queries that are more semantically aligned with user intent than those in the benchmarks. This work makes spatial analysis more accessible, and provides a robust, generalizable foundation for spatial Text-to-SQL systems, advancing the development of autonomous GIS.

**arXiv ID:** 2510.21045
</details>

<details>
<summary><strong>A Knowledge-Graph Translation Layer for Mission-Aware Multi-Agent Path Planning in Spatiotemporal Dynamics</strong> - Edward Holmberg, Elias Ioup, Mahdi Abdelguerfi - [[pdf]](https://arxiv.org/pdf/2510.21695)</summary>

**Abstract:** The coordination of autonomous agents in dynamic environments is hampered by the semantic gap between high-level mission objectives and low-level planner inputs. To address this, we introduce a framework centered on a Knowledge Graph (KG) that functions as an intelligent translation layer. The KG's two-plane architecture compiles declarative facts into per-agent, mission-aware ``worldviews" and physics-aware traversal rules, decoupling mission semantics from a domain-agnostic planner. This allows complex, coordinated paths to be modified simply by changing facts in the KG. A case study involving Autonomous Underwater Vehicles (AUVs) in the Gulf of Mexico visually demonstrates the end-to-end process and quantitatively proves that different declarative policies produce distinct, high-performing outcomes. This work establishes the KG not merely as a data repository, but as a powerful, stateful orchestrator for creating adaptive and explainable autonomous systems.

**arXiv ID:** 2510.21695
</details>

<details>
<summary><strong>CC-GRMAS: A Multi-Agent Graph Neural System for Spatiotemporal Landslide Risk Assessment in High Mountain Asia</strong> - Mihir Panchal, Ying-Jung Chen, Surya Parkash - [[pdf]](https://arxiv.org/pdf/2510.20875)</summary>

**Abstract:** Landslides are a growing climate induced hazard with severe environmental and human consequences, particularly in high mountain Asia. Despite increasing access to satellite and temporal datasets, timely detection and disaster response remain underdeveloped and fragmented. This work introduces CC-GRMAS, a framework leveraging a series of satellite observations and environmental signals to enhance the accuracy of landslide forecasting. The system is structured around three interlinked agents Prediction, Planning, and Execution, which collaboratively enable real time situational awareness, response planning, and intervention. By incorporating local environmental factors and operationalizing multi agent coordination, this approach offers a scalable and proactive solution for climate resilient disaster preparedness across vulnerable mountainous terrains.

**arXiv ID:** 2510.20875
</details>

<details>
<summary><strong>Hierarchical AI Multi-Agent Fundamental Investing: Evidence from China's A-Share Market</strong> - Chujun He, Zhonghao Huang, Xiangguo Li, Ye Luo, Kewei Ma, Yuxuan Xiong, Xiaowei Zhang, Mingyang Zhao - [[pdf]](https://arxiv.org/pdf/2510.21147)</summary>

**Abstract:** We present a multi-agent, AI-driven framework for fundamental investing that integrates macro indicators, industry-level and firm-specific information to construct optimized equity portfolios. The architecture comprises: (i) a Macro agent that dynamically screens and weights sectors based on evolving economic indicators and industry performance; (ii) four firm-level agents -- Fundamental, Technical, Report, and News -- that conduct in-depth analyses of individual firms to ensure both breadth and depth of coverage; (iii) a Portfolio agent that uses reinforcement learning to combine the agent outputs into a unified policy to generate the trading strategy; and (iv) a Risk Control agent that adjusts portfolio positions in response to market volatility. We evaluate the system on the constituents by the CSI 300 Index of China's A-share market and find that it consistently outperforms standard benchmarks and a state-of-the-art multi-agent trading system on risk-adjusted returns and drawdown control. Our core contribution is a hierarchical multi-agent design that links top-down macro screening with bottom-up fundamental analysis, offering a robust and extensible approach to factor-based portfolio construction.

**arXiv ID:** 2510.21147
</details>

<details>
<summary><strong>HIKMA: Human-Inspired Knowledge by Machine Agents through a Multi-Agent Framework for Semi-Autonomous Scientific Conferences</strong> - Zain Ul Abideen Tariq, Mahmood Al-Zubaidi, Uzair Shah, Marco Agus, Mowafa Househ - [[pdf]](https://arxiv.org/pdf/2510.21370)</summary>

**Abstract:** HIKMA Semi-Autonomous Conference is the first experiment in reimagining scholarly communication through an end-to-end integration of artificial intelligence into the academic publishing and presentation pipeline. This paper presents the design, implementation, and evaluation of the HIKMA framework, which includes AI dataset curation, AI-based manuscript generation, AI-assisted peer review, AI-driven revision, AI conference presentation, and AI archival dissemination. By combining language models, structured research workflows, and domain safeguards, HIKMA shows how AI can support - not replace traditional scholarly practices while maintaining intellectual property protection, transparency, and integrity. The conference functions as a testbed and proof of concept, providing insights into the opportunities and challenges of AI-enabled scholarship. It also examines questions about AI authorship, accountability, and the role of human-AI collaboration in research.

**arXiv ID:** 2510.21370
</details>

<details>
<summary><strong>Mix Q-learning for Lane Changing: A Collaborative Decision-Making Method in Multi-Agent Deep Reinforcement Learning</strong> - Xiaojun Bi, Mingjie He, Yiwen Sun - [[repo]](https://github.com/pku-smart-city/source_code) [[pdf]](https://arxiv.org/pdf/2406.09755)</summary>

**Abstract:** Lane-changing decisions, which are crucial for autonomous vehicle path planning, face practical challenges due to rule-based constraints and limited data. Deep reinforcement learning has become a major research focus due to its advantages in data acquisition and interpretability. However, current models often overlook collaboration, which affects not only impacts overall traffic efficiency but also hinders the vehicle's own normal driving in the long run. To address the aforementioned issue, this paper proposes a method named Mix Q-learning for Lane Changing(MQLC) that integrates a hybrid value Q network, taking into account both collective and individual benefits for the greater good. At the collective level, our method coordinates the individual Q and global Q networks by utilizing global information. This enables agents to effectively balance their individual interests with the collective benefit. At the individual level, we integrated a deep learning-based intent recognition module into our observation and enhanced the decision network. These changes provide agents with richer decision information and more accurate feature extraction for improved lane-changing decisions. This strategy enables the multi-agent system to learn and formulate optimal decision-making strategies effectively. Our MQLC model, through extensive experimental results, impressively outperforms other state-of-the-art multi-agent decision-making methods, achieving significantly safer and faster lane-changing decisions. The code is available at https:github.com/pku-smart-city/source_code/tree/main/MQLC.

**arXiv ID:** 2406.09755
</details>

<details>
<summary><strong>HypRL: Reinforcement Learning of Control Policies for Hyperproperties</strong> - Tzu-Han Hsu, Arshia Rafieioskouei, Borzoo Bonakdarpour - [[pdf]](https://arxiv.org/pdf/2504.04675)</summary>

**Abstract:** Reward shaping in multi-agent reinforcement learning (MARL) for complex tasks remains a significant challenge. Existing approaches often fail to find optimal solutions or cannot efficiently handle such tasks. We propose HYPRL, a specification-guided reinforcement learning framework that learns control policies w.r.t. hyperproperties expressed in HyperLTL. Hyperproperties constitute a powerful formalism for specifying objectives and constraints over sets of execution traces across agents. To learn policies that maximize the satisfaction of a HyperLTL formula $\phi$, we apply Skolemization to manage quantifier alternations and define quantitative robustness functions to shape rewards over execution traces of a Markov decision process with unknown transitions. A suitable RL algorithm is then used to learn policies that collectively maximize the expected reward and, consequently, increase the probability of satisfying $\phi$. We evaluate HYPRL on a diverse set of benchmarks, including safety-aware planning, Deep Sea Treasure, and the Post Correspondence Problem. We also compare with specification-driven baselines to demonstrate the effectiveness and efficiency of HYPRL.

**arXiv ID:** 2504.04675
</details>

<details>
<summary><strong>Mitigating Manipulation and Enhancing Persuasion: A Reflective Multi-Agent Approach for Legal Argument Generation</strong> - Li Zhang, Kevin D. Ashley - [[pdf]](https://arxiv.org/pdf/2506.02992)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly explored for legal argument generation, yet they pose significant risks of manipulation through hallucination and ungrounded persuasion, and often fail to utilize provided factual bases effectively or abstain when arguments are untenable. This paper introduces a novel reflective multi-agent method designed to address these challenges in the context of legally compliant persuasion. Our approach employs specialized agents (factor analyst and argument polisher) in an iterative refinement process to generate 3-ply legal arguments (plaintiff, defendant, rebuttal). We evaluate reflective multi-agent against single-agent, enhanced-prompt single-agent, and non-reflective multi-agent baselines using four diverse LLMs (GPT-4o, GPT-4o-mini, Llama-4-Maverick-17b-128e, Llama-4-Scout-17b-16e) across three legal scenarios: "arguable", "mismatched", and "non-arguable". Results demonstrate that the reflective multi-agent approach excels at successful abstention by preventing generation when arguments cannot be grounded, improves hallucination accuracy by reducing fabricated and misattributed factors and enhances factor utilization recall by better using the provided case facts. These findings suggest that structured reflection within a multi-agent framework offers a robust method for fostering ethical persuasion and mitigating manipulation in LLM-based legal argumentation systems.

**arXiv ID:** 2506.02992
</details>

<details>
<summary><strong>AgentSense: LLMs Empower Generalizable and Explainable Web-Based Participatory Urban Sensing</strong> - Xusen Guo, Mingxing Peng, Xixuan Hao, Xingchen Zou, Qiongyan Wang, Sijie Ruan, Yuxuan Liang - [[pdf]](https://arxiv.org/pdf/2510.19661)</summary>

**Abstract:** Web-based participatory urban sensing has emerged as a vital approach for modern urban management by leveraging mobile individuals as distributed sensors. However, existing urban sensing systems struggle with limited generalization across diverse urban scenarios and poor interpretability in decision-making. In this work, we introduce AgentSense, a hybrid, training-free framework that integrates large language models (LLMs) into participatory urban sensing through a multi-agent evolution system. AgentSense initially employs classical planner to generate baseline solutions and then iteratively refines them to adapt sensing task assignments to dynamic urban conditions and heterogeneous worker preferences, while producing natural language explanations that enhance transparency and trust. Extensive experiments across two large-scale mobility datasets and seven types of dynamic disturbances demonstrate that AgentSense offers distinct advantages in adaptivity and explainability over traditional methods. Furthermore, compared to single-agent LLM baselines, our approach outperforms in both performance and robustness, while delivering more reasonable and transparent explanations. These results position AgentSense as a significant advancement towards deploying adaptive and explainable urban sensing systems on the web.

**arXiv ID:** 2510.19661
</details>

<details>
<summary><strong>Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning</strong> - Emile Anand, Ishani Karmarkar, Guannan Qu - [[pdf]](https://arxiv.org/pdf/2412.00661)</summary>

**Abstract:** Designing efficient algorithms for multi-agent reinforcement learning (MARL) is fundamentally challenging because the size of the joint state and action spaces grows exponentially in the number of agents. These difficulties are exacerbated when balancing sequential global decision-making with local agent interactions. In this work, we propose a new algorithm $\texttt{SUBSAMPLE-MFQ}$ ($\textbf{Subsample}$-$\textbf{M}$ean-$\textbf{F}$ield-$\textbf{Q}$-learning) and a decentralized randomized policy for a system with $n$ agents. For any $k\leq n$, our algorithm learns a policy for the system in time polynomial in $k$. We prove that this learned policy converges to the optimal policy on the order of $\tilde{O}(1/\sqrt{k})$ as the number of subsampled agents $k$ increases. In particular, this bound is independent of the number of agents $n$.

**arXiv ID:** 2412.00661
</details>

<details>
<summary><strong>Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective</strong> - Yang Zhang, Xinran Li, Jianing Ye, Shuang Qiu, Delin Qu, Xiu Li, Chongjie Zhang, Chenjia Bai - [[pdf]](https://arxiv.org/pdf/2505.20922)</summary>

**Abstract:** World models have recently attracted growing interest in Multi-Agent Reinforcement Learning (MARL) due to their ability to improve sample efficiency for policy learning. However, accurately modeling environments in MARL is challenging due to the exponentially large joint action space and highly uncertain dynamics inherent in multi-agent systems. To address this, we reduce modeling complexity by shifting from jointly modeling the entire state-action transition dynamics to focusing on the state space alone at each timestep through sequential agent modeling. Specifically, our approach enables the model to progressively resolve uncertainty while capturing the structured dependencies among agents, providing a more accurate representation of how agents influence the state. Interestingly, this sequential revelation of agents' actions in a multi-agent system aligns with the reverse process in diffusion models--a class of powerful generative models known for their expressiveness and training stability compared to autoregressive or latent variable models. Leveraging this insight, we develop a flexible and robust world model for MARL using diffusion models. Our method, Diffusion-Inspired Multi-Agent world model (DIMA), achieves state-of-the-art performance across multiple multi-agent control benchmarks, significantly outperforming prior world models in terms of final return and sample efficiency, including MAMuJoCo and Bi-DexHands. DIMA establishes a new paradigm for constructing multi-agent world models, advancing the frontier of MARL research. Codes are open-sourced at this https URL.

**arXiv ID:** 2505.20922
</details>

<details>
<summary><strong>\textsc{autoresearcher}: Automating Knowledge-Grounded and Transparent Research Ideation with Multi-Agent Collaboration</strong> - Jiawei Zhou, Ruicheng Zhu, Mengshi Chen, Jianwei Wang, Kai Wang - [[pdf]](https://arxiv.org/pdf/2510.20844)</summary>

**Abstract:** Effective research relies on organizing extensive information and stimulating novel solutions. Agentic systems have recently emerged as a promising tool to automate literature-based ideation. However, current systems often remain black-box. Their outputs may appear plausible but weakly grounded, with limited transparency or control for researchers. Our work introduces \textsc{autoresearcher}, a multi-agent demo system for knowledge-grounded and transparent ideation. Specifically, \textsc{autoresearcher} integrates meticulously designed four stages into a unified framework: (A) Structured Knowledge Curation, (B) Diversified Idea Generation, (C) Multi-stage Idea Selection, and (D) Expert Panel Review \& Synthesis. Different from prior pipelines, our system not only exposes intermediate reasoning states, execution logs, and tunable agents for inspections, but also enables the generation of hypotheses that are both diverse and evidence-aligned. Our design is also domain-agnostic: as long as literature sources exist, the same pipeline can be instantiated in any scientific field. As an illustrative case, we demonstrate \textsc{autoresearcher} on a graph-mining case study ($k$-truss breaking problem), where it generates distinct, plausible hypotheses with evidence and critiques. A live demo and source code are available at this https URL.

**arXiv ID:** 2510.20844
</details>

<details>
<summary><strong>ColorEcosystem: Powering Personalized, Standardized, and Trustworthy Agentic Service in massive-agent Ecosystem</strong> - Fangwen Wu, Zheng Wu, Jihong Wang, Yunku Chen, Ruiguang Pei, Heyuan Huang, Xin Liao, Xingyu Lou, Huarong Deng, Zhihui Fu, Weiwen Liu, Zhuosheng Zhang, Weinan Zhang, Jun Wang - [[pdf]](https://arxiv.org/pdf/2510.21566)</summary>

**Abstract:** With the rapid development of (multimodal) large language model-based agents, the landscape of agentic service management has evolved from single-agent systems to multi-agent systems, and now to massive-agent ecosystems. Current massive-agent ecosystems face growing challenges, including impersonal service experiences, a lack of standardization, and untrustworthy behavior. To address these issues, we propose ColorEcosystem, a novel blueprint designed to enable personalized, standardized, and trustworthy agentic service at scale. Concretely, ColorEcosystem consists of three key components: agent carrier, agent store, and agent audit. The agent carrier provides personalized service experiences by utilizing user-specific data and creating a digital twin, while the agent store serves as a centralized, standardized platform for managing diverse agentic services. The agent audit, based on the supervision of developer and user activities, ensures the integrity and credibility of both service providers and users. Through the analysis of challenges, transitional forms, and practical considerations, the ColorEcosystem is poised to power personalized, standardized, and trustworthy agentic service across massive-agent ecosystems. Meanwhile, we have also implemented part of ColorEcosystem's functionality, and the relevant code is open-sourced at this https URL.

**arXiv ID:** 2510.21566
</details>

<details>
<summary><strong>ColorAgent: Building A Robust, Personalized, and Interactive OS Agent</strong> - Ning Li, Qiqiang Lin, Zheng Wu, Xiaoyun Mo, Weiming Zhang, Yin Zhao, Xiangmou Qu, Jiamu Zhou, Jun Wang, Congmin Zheng, Yuanyi Song, Hongjiang Chen, Heyuan Huang, Jihong Wang, Jiaxin Yin, Jingwei Yu, Junwei Liao, Qiuying Peng, Xingyu Lou, Jun Wang, Weiwen Liu, Zhuosheng Zhang, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2510.19386)</summary>

**Abstract:** With the advancements in hardware, software, and large language model technologies, the interaction between humans and operating systems has evolved from the command-line interface to the rapidly emerging AI agent interactions. Building an operating system (OS) agent capable of executing user instructions and faithfully following user desires is becoming a reality. In this technical report, we present ColorAgent, an OS agent designed to engage in long-horizon, robust interactions with the environment while also enabling personalized and proactive user interaction. To enable long-horizon interactions with the environment, we enhance the model's capabilities through step-wise reinforcement learning and self-evolving training, while also developing a tailored multi-agent framework that ensures generality, consistency, and robustness. In terms of user interaction, we explore personalized user intent recognition and proactive engagement, positioning the OS agent not merely as an automation tool but as a warm, collaborative partner. We evaluate ColorAgent on the AndroidWorld and AndroidLab benchmarks, achieving success rates of 77.2% and 50.7%, respectively, establishing a new state of the art. Nonetheless, we note that current benchmarks are insufficient for a comprehensive evaluation of OS agents and propose further exploring directions in future work, particularly in the areas of evaluation paradigms, agent collaboration, and security.

**arXiv ID:** 2510.19386
</details>

<details>
<summary><strong>DispatchMAS: Fusing taxonomy and artificial intelligence agents for emergency medical services</strong> - Xiang Li, Huizi Yu, Wenkong Wang, Yiran Wu, Jiayan Zhou, Wenyue Hua, Xinxin Lin, Wenjia Tan, Lexuan Zhu, Bingyi Chen, Guang Chen, Ming-Li Chen, Yang Zhou, Zhao Li, Themistocles L. Assimes, Yongfeng Zhang, Qingyun Wu, Xin Ma, Lingyao Li, Lizhou Fan - [[pdf]](https://arxiv.org/pdf/2510.21228)</summary>

**Abstract:** Objective: Emergency medical dispatch (EMD) is a high-stakes process challenged by caller distress, ambiguity, and cognitive load. Large Language Models (LLMs) and Multi-Agent Systems (MAS) offer opportunities to augment dispatchers. This study aimed to develop and evaluate a taxonomy-grounded, LLM-powered multi-agent system for simulating realistic EMD scenarios. Methods: We constructed a clinical taxonomy (32 chief complaints, 6 caller identities from MIMIC-III) and a six-phase call protocol. Using this framework, we developed an AutoGen-based MAS with Caller and Dispatcher Agents. The system grounds interactions in a fact commons to ensure clinical plausibility and mitigate misinformation. We used a hybrid evaluation framework: four physicians assessed 100 simulated cases for "Guidance Efficacy" and "Dispatch Effectiveness," supplemented by automated linguistic analysis (sentiment, readability, politeness). Results: Human evaluation, with substantial inter-rater agreement (Gwe's AC1 > 0.70), confirmed the system's high performance. It demonstrated excellent Dispatch Effectiveness (e.g., 94 % contacting the correct potential other agents) and Guidance Efficacy (advice provided in 91 % of cases), both rated highly by physicians. Algorithmic metrics corroborated these findings, indicating a predominantly neutral affective profile (73.7 % neutral sentiment; 90.4 % neutral emotion), high readability (Flesch 80.9), and a consistently polite style (60.0 % polite; 0 % impolite). Conclusion: Our taxonomy-grounded MAS simulates diverse, clinically plausible dispatch scenarios with high fidelity. Findings support its use for dispatcher training, protocol evaluation, and as a foundation for real-time decision support. This work outlines a pathway for safely integrating advanced AI agents into emergency response workflows.

**arXiv ID:** 2510.21228
</details>

<details>
<summary><strong>Mixture-of-Minds: Multi-Agent Reinforcement Learning for Table Understanding</strong> - Yuhang Zhou, Mingrui Zhang, Ke Li, Mingyi Wang, Qiao Liu, Qifei Wang, Jiayi Liu, Fei Liu, Serena Li, Weiwei Li, Mingze Gao, Abhishek Kumar, Xiangjun Fan, Zhuokai Zhao, Lizhu Zhang - [[pdf]](https://arxiv.org/pdf/2510.20176)</summary>

**Abstract:** Understanding and reasoning over tables is a critical capability for many real-world applications. Large language models (LLMs) have shown promise on this task, but current approaches remain limited. Fine-tuning based methods strengthen language reasoning; yet they are prone to arithmetic errors and hallucination. In contrast, tool-based methods enable precise table manipulation but rely on rigid schemas and lack semantic understanding. These complementary drawbacks highlight the need for approaches that integrate robust reasoning with reliable table processing. In this work, we propose Mixture-of-Minds, a multi-agent framework that decomposes table reasoning into three specialized roles: planning, coding, and answering. This design enables each agent to focus on a specific aspect of the task while leveraging code execution for precise table manipulation. Building on this workflow, we introduce a self-improvement training framework that employs Monte Carlo Tree Search (MCTS) rollouts to generate pseudo-gold trajectories and optimize agents with reinforcement learning (RL). Extensive experiments show that Mixture-of-Minds delivers substantial gains, reaching 62.13% on TableBench and surpassing OpenAI-o4-mini-high. These results demonstrate the promise of combining structured multi-agent workflows with RL to advance table understanding.

**arXiv ID:** 2510.20176
</details>

<details>
<summary><strong>Towards Scalable Oversight with Collaborative Multi-Agent Debate in Error Detection</strong> - Yongqiang Chen, Gang Niu, James Cheng, Bo Han, Masashi Sugiyama - [[pdf]](https://arxiv.org/pdf/2510.20963)</summary>

**Abstract:** Accurate detection of errors in large language models (LLM) responses is central to the success of scalable oversight, or providing effective supervision to superhuman intelligence. Yet, self-diagnosis is often unreliable on complex tasks unless aided by reliable external feedback. Multi-agent debate (MAD) seems to be a natural alternative to external feedback: multiple LLMs provide complementary perspectives and cross-checks for error detection. However, prior MAD protocols frame debate as a zero-sum game, where the debaters compete to win the game instead of seeking the truth. Consequently, it leads to debate hacking: debaters tend to mislead the judge by misinterpreting the task or presenting overconfident claims, which introduce more mistakes and underperform single-agent methods. To mitigate the issue, we introduce a new collaborative MAD protocol, termed ColMAD, that reframes MAD as a non-zero sum game. Specifically, ColMAD encourages multiple agents to criticize each other in a supportive way, such that they can complement the missing points of each other. Therefore, the judge agent can make a more informative conclusion based on more comprehensive evidence. Empirically, we show that ColMAD significantly outperforms previous competitive MAD by 19% and brings non-trivial improvements over single-agent methods in error detection.

**arXiv ID:** 2510.20963
</details>

</details>

<details open>
<summary><h2>Other Agent Research (2 papers)</h2></summary>

<details>
<summary><strong>Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning</strong> - Sanghyun Ahn, Wonje Choi, Junyong Lee, Jinwoo Park, Honguk Woo - [[pdf]](https://arxiv.org/pdf/2510.21302)</summary>

**Abstract:** Recent advances in large language models (LLMs) have enabled the automatic generation of executable code for task planning and control in embodied agents such as robots, demonstrating the potential of LLM-based embodied intelligence. However, these LLM-based code-as-policies approaches often suffer from limited environmental grounding, particularly in dynamic or partially observable settings, leading to suboptimal task success rates due to incorrect or incomplete code generation. In this work, we propose a neuro-symbolic embodied task planning framework that incorporates explicit symbolic verification and interactive validation processes during code generation. In the validation phase, the framework generates exploratory code that actively interacts with the environment to acquire missing observations while preserving task-relevant states. This integrated process enhances the grounding of generated code, resulting in improved task reliability and success rates in complex environments. We evaluate our framework on RLBench and in real-world settings across dynamic, partially observable scenarios. Experimental results demonstrate that our framework improves task success rates by 46.2% over Code-as-Policies baselines and attains over 86.8% executability of task-relevant actions, thereby enhancing the reliability of task planning in dynamic environments.

**arXiv ID:** 2510.21302
</details>

<details>
<summary><strong>Central Bank Digital Currency, Flight-to-Quality, and Bank-Runs in an Agent-Based Model</strong> - Emilio Barucci, Andrea Gurgone, Giulia Iori, Michele Azzone - [[pdf]](https://arxiv.org/pdf/2510.21071)</summary>

**Abstract:** We analyse financial stability and welfare impacts associated with the introduction of a Central Bank Digital Currency (CBDC) in a macroeconomic agent-based model. The model considers firms, banks, and households interacting on labour, goods, credit, and interbank markets. Households move their liquidity from deposits to CBDC based on the perceived riskiness of their banks. We find that the introduction of CBDC exacerbates bank-runs and may lead to financial instability phenomena. The effect can be changed by introducing a limit on CBDC holdings. The adoption of CBDC has little effect on macroeconomic variables but the interest rate on loans to firms goes up and credit goes down in a limited way. CBDC leads to a redistribution of wealth from firms and banks to households with a higher bank default rate. CBDC may have negative welfare effects, but a bound on holding enables a welfare improvement.

**arXiv ID:** 2510.21071
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (32 papers)</h2></summary>

<details>
<summary><strong>Confounding Robust Deep Reinforcement Learning: A Causal Approach</strong> - Mingxuan Li, Junzhe Zhang, Elias Bareinboim - [[pdf]](https://arxiv.org/pdf/2510.21110)</summary>

**Abstract:** A key task in Artificial Intelligence is learning effective policies for controlling agents in unknown environments to optimize performance measures. Off-policy learning methods, like Q-learning, allow learners to make optimal decisions based on past experiences. This paper studies off-policy learning from biased data in complex and high-dimensional domains where \emph{unobserved confounding} cannot be ruled out a priori. Building on the well-celebrated Deep Q-Network (DQN), we propose a novel deep reinforcement learning algorithm robust to confounding biases in observed data. Specifically, our algorithm attempts to find a safe policy for the worst-case environment compatible with the observations. We apply our method to twelve confounded Atari games, and find that it consistently dominates the standard DQN in all games where the observed input to the behavioral and target policies mismatch and unobserved confounders exist.

**arXiv ID:** 2510.21110
</details>

<details>
<summary><strong>PanicToCalm: A Proactive Counseling Agent for Panic Attacks</strong> - Jihyun Lee, Yejin Min, San Kim, Yejin Jeon, SungJun Yang, Hyounghun Kim, Gary Geunbae Lee - [[pdf]](https://arxiv.org/pdf/2510.21143)</summary>

**Abstract:** Panic attacks are acute episodes of fear and distress, in which timely, appropriate intervention can significantly help individuals regain stability. However, suitable datasets for training such models remain scarce due to ethical and logistical issues. To address this, we introduce PACE, which is a dataset that includes high-distress episodes constructed from first-person narratives, and structured around the principles of Psychological First Aid (PFA). Using this data, we train PACER, a counseling model designed to provide both empathetic and directive support, which is optimized through supervised learning and simulated preference alignment. To assess its effectiveness, we propose PanicEval, a multi-dimensional framework covering general counseling quality and crisis-specific strategies. Experimental results show that PACER outperforms strong baselines in both counselor-side metrics and client affect improvement. Human evaluations further confirm its practical value, with PACER consistently preferred over general, CBT-based, and GPT-4-powered models in panic scenarios (Code is available at this https URL ).

**arXiv ID:** 2510.21143
</details>

<details>
<summary><strong>CXRAgent: Director-Orchestrated Multi-Stage Reasoning for Chest X-Ray Interpretation</strong> - Jinhui Lou, Yan Yang, Zhou Yu, Zhenqi Fu, Weidong Han, Qingming Huang, Jun Yu - [[pdf]](https://arxiv.org/pdf/2510.21324)</summary>

**Abstract:** Chest X-ray (CXR) plays a pivotal role in clinical diagnosis, and a variety of task-specific and foundation models have been developed for automatic CXR interpretation. However, these models often struggle to adapt to new diagnostic tasks and complex reasoning scenarios. Recently, LLM-based agent models have emerged as a promising paradigm for CXR analysis, enhancing model's capability through tool coordination, multi-step reasoning, and team collaboration, etc. However, existing agents often rely on a single diagnostic pipeline and lack mechanisms for assessing tools' reliability, limiting their adaptability and credibility. To this end, we propose CXRAgent, a director-orchestrated, multi-stage agent for CXR interpretation, where a central director coordinates the following stages: (1) Tool Invocation: The agent strategically orchestrates a set of CXR-analysis tools, with outputs normalized and verified by the Evidence-driven Validator (EDV), which grounds diagnostic outputs with visual evidence to support reliable downstream diagnosis; (2) Diagnostic Planning: Guided by task requirements and intermediate findings, the agent formulates a targeted diagnostic plan. It then assembles an expert team accordingly, defining member roles and coordinating their interactions to enable adaptive and collaborative reasoning; (3) Collaborative Decision-making: The agent integrates insights from the expert team with accumulated contextual memories, synthesizing them into an evidence-backed diagnostic conclusion. Experiments on various CXR interpretation tasks show that CXRAgent delivers strong performance, providing visual evidence and generalizes well to clinical tasks of different complexity. Code and data are valuable at this \href{this https URL}{link}.

**arXiv ID:** 2510.21324
</details>

<details>
<summary><strong>Boosting Accuracy and Efficiency of Budget Forcing in LLMs via Reinforcement Learning for Mathematical Reasoning</strong> - Ravindra Aribowo Tarunokusumo, Rafael Fernandes Cunha - [[pdf]](https://arxiv.org/pdf/2510.21398)</summary>

**Abstract:** Test-time scaling methods have seen a rapid increase in popularity for its computational efficiency and parameter-independent training to improve reasoning performance on Large Language Models. One such method is called budget forcing, a decoding intervention strategy which allocates extra compute budget for thinking and elicits the inherent self-correcting behavior of the model. However, this relies on supervised fine-tuning (SFT) on long-context reasoning traces which causes performance degradation on smaller models due to verbose responses. For this reason, we offer a framework integrating reinforcement learning (RL) to improve token efficiency and boost the performance of a 1.5B model for mathematical reasoning. We demonstrate this using only 1.5K training samples and found that our SFT+RL model performed better on the GSM8K dataset with varying compute budgets. Our main findings showed an overall higher accuracy while significantly reducing its token usage by over 40% compared to the SFT model, revealing how RL can recover the losses due to long-context training and altogether improving performance in mathematical reasoning.

**arXiv ID:** 2510.21398
</details>

<details>
<summary><strong>Co-Sight: Enhancing LLM-Based Agents via Conflict-Aware Meta-Verification and Trustworthy Reasoning with Structured Facts</strong> - Hongwei Zhang, Ji Lu, Shiqing Jiang, Chenxiang Zhu, Li Xie, Chen Zhong, Haoran Chen, Yurui Zhu, Yongsheng Du, Yanqin Gao, Lingjun Huang, Baoli Wang, Fang Tan, Peng Zou - [[pdf]](https://arxiv.org/pdf/2510.21557)</summary>

**Abstract:** Long-horizon reasoning in LLM-based agents often fails not from generative weakness but from insufficient verification of intermediate reasoning. Co-Sight addresses this challenge by turning reasoning into a falsifiable and auditable process through two complementary mechanisms: Conflict-Aware Meta-Verification (CAMV) and Trustworthy Reasoning with Structured Facts (TRSF). CAMV reformulates verification as conflict identification and targeted falsification, allocating computation only to disagreement hotspots among expert agents rather than to full reasoning chains. This bounds verification cost to the number of inconsistencies and improves efficiency and reliability. TRSF continuously organizes, validates, and synchronizes evidence across agents through a structured facts module. By maintaining verified, traceable, and auditable knowledge, it ensures that all reasoning is grounded in consistent, source-verified information and supports transparent verification throughout the reasoning process. Together, TRSF and CAMV form a closed verification loop, where TRSF supplies structured facts and CAMV selectively falsifies or reinforces them, yielding transparent and trustworthy reasoning. Empirically, Co-Sight achieves state-of-the-art accuracy on GAIA (84.4%) and Humanity's Last Exam (35.5%), and strong results on Chinese-SimpleQA (93.8%). Ablation studies confirm that the synergy between structured factual grounding and conflict-aware verification drives these improvements. Co-Sight thus offers a scalable paradigm for reliable long-horizon reasoning in LLM-based agents. Code is available at this https URL.

**arXiv ID:** 2510.21557
</details>

<details>
<summary><strong>DeepAgent: A General Reasoning Agent with Scalable Toolsets</strong> - Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Guanting Dong, Jiajie Jin, Yinuo Wang, Hao Wang, Yutao Zhu, Ji-Rong Wen, Yuan Lu, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2510.21618)</summary>

**Abstract:** Large reasoning models have demonstrated strong problem-solving abilities, yet real-world tasks often require external tools and long-horizon interactions. Existing agent frameworks typically follow predefined workflows, which limit autonomous and global task completion. In this paper, we introduce DeepAgent, an end-to-end deep reasoning agent that performs autonomous thinking, tool discovery, and action execution within a single, coherent reasoning process. To address the challenges of long-horizon interactions, particularly the context length explosion from multiple tool calls and the accumulation of interaction history, we introduce an autonomous memory folding mechanism that compresses past interactions into structured episodic, working, and tool memories, reducing error accumulation while preserving critical information. To teach general-purpose tool use efficiently and stably, we develop an end-to-end reinforcement learning strategy, namely ToolPO, that leverages LLM-simulated APIs and applies tool-call advantage attribution to assign fine-grained credit to the tool invocation tokens. Extensive experiments on eight benchmarks, including general tool-use tasks (ToolBench, API-Bank, TMDB, Spotify, ToolHop) and downstream applications (ALFWorld, WebShop, GAIA, HLE), demonstrate that DeepAgent consistently outperforms baselines across both labeled-tool and open-set tool retrieval scenarios. This work takes a step toward more general and capable agents for real-world applications. The code and demo are available at this https URL.

**arXiv ID:** 2510.21618
</details>

<details>
<summary><strong>Enhanced Evolutionary Multi-Objective Deep Reinforcement Learning for Reliable and Efficient Wireless Rechargeable Sensor Networks</strong> - Bowei Tong, Hui Kang, Jiahui Li, Geng Sun, Jiacheng Wang, Yaoqi Yang, Bo Xu, Dusit Niyato - [[pdf]](https://arxiv.org/pdf/2510.21127)</summary>

**Abstract:** Despite rapid advancements in sensor networks, conventional battery-powered sensor networks suffer from limited operational lifespans and frequent maintenance requirements that severely constrain their deployment in remote and inaccessible environments. As such, wireless rechargeable sensor networks (WRSNs) with mobile charging capabilities offer a promising solution to extend network lifetime. However, WRSNs face critical challenges from the inherent trade-off between maximizing the node survival rates and maximizing charging energy efficiency under dynamic operational conditions. In this paper, we investigate a typical scenario where mobile chargers move and charge the sensor, thereby maintaining the network connectivity while minimizing the energy waste. Specifically, we formulate a multi-objective optimization problem that simultaneously maximizes the network node survival rate and mobile charger energy usage efficiency across multiple time slots, which presents NP-hard computational complexity with long-term temporal dependencies that make traditional optimization approaches ineffective. To address these challenges, we propose an enhanced evolutionary multi-objective deep reinforcement learning algorithm, which integrates a long short-term memory (LSTM)-based policy network for temporal pattern recognition, a multilayer perceptron-based prospective increment model for future state prediction, and a time-varying Pareto policy evaluation method for dynamic preference adaptation. Extensive simulation results demonstrate that the proposed algorithm significantly outperforms existing approaches in balancing node survival rate and energy efficiency while generating diverse Pareto-optimal solutions. Moreover, the LSTM-enhanced policy network converges 25% faster than conventional networks, with the time-varying evaluation method effectively adapting to dynamic conditions.

**arXiv ID:** 2510.21127
</details>

<details>
<summary><strong>Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design</strong> - Lianghong Chen, Dongkyu Eugene Kim, Mike Domaratzki, Pingzhao Hu - [[pdf]](https://arxiv.org/pdf/2510.21153)</summary>

**Abstract:** Designing de novo 3D molecules with desirable properties remains a fundamental challenge in drug discovery and molecular engineering. While diffusion models have demonstrated remarkable capabilities in generating high-quality 3D molecular structures, they often struggle to effectively control complex multi-objective constraints critical for real-world applications. In this study, we propose an uncertainty-aware Reinforcement Learning (RL) framework to guide the optimization of 3D molecular diffusion models toward multiple property objectives while enhancing the overall quality of the generated molecules. Our method leverages surrogate models with predictive uncertainty estimation to dynamically shape reward functions, facilitating balance across multiple optimization objectives. We comprehensively evaluate our framework across three benchmark datasets and multiple diffusion model architectures, consistently outperforming baselines for molecular quality and property optimization. Additionally, Molecular Dynamics (MD) simulations and ADMET profiling of top generated candidates indicate promising drug-like behavior and binding stability, comparable to known Epidermal Growth Factor Receptor (EGFR) inhibitors. Our results demonstrate the strong potential of RL-guided generative diffusion models for advancing automated molecular design.

**arXiv ID:** 2510.21153
</details>

<details>
<summary><strong>Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering</strong> - Liang Zhang, Justin Lieffers, Adarsh Pyarelal - [[pdf]](https://arxiv.org/pdf/2409.17411)</summary>

**Abstract:** In this paper, we explore semantic clustering properties of deep reinforcement learning (DRL) to improve its interpretability and deepen our understanding of its internal semantic organization. In this context, semantic clustering refers to the ability of neural networks to cluster inputs based on their semantic similarity in the feature space. We propose a DRL architecture that incorporates a novel semantic clustering module that combines feature dimensionality reduction with online clustering. This module integrates seamlessly into the DRL training pipeline, addressing the instability of t-SNE and eliminating the need for extensive manual annotation inherent to prior semantic analysis methods. We experimentally validate the effectiveness of the proposed module and demonstrate its ability to reveal semantic clustering properties within DRL. Furthermore, we introduce new analytical methods based on these properties to provide insights into the hierarchical structure of policies and semantic organization within the feature space. Our code is available at this https URL.

**arXiv ID:** 2409.17411
</details>

<details>
<summary><strong>Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning</strong> - Jiayu Wang, Yifei Ming, Zixuan Ke, Caiming Xiong, Shafiq Joty, Aws Albarghouthi, Frederic Sala - [[pdf]](https://arxiv.org/pdf/2506.04723)</summary>

**Abstract:** Reinforcement learning (RL) has become the dominant paradigm for improving the performance of language models on complex reasoning tasks. Despite the substantial empirical gains demonstrated by RL-based training methods like GRPO, a granular understanding of why and how RL enhances performance is still lacking. To bridge this gap, we introduce SPARKLE, a fine-grained analytic framework to dissect the effects of RL across three key dimensions: (1) plan following and execution, (2) knowledge integration, and (3) chain of subproblems. Using this framework, we gain insights beyond mere accuracy. For instance, providing models with explicit human-crafted, step-by-step plans can surprisingly degrade performance on the most challenging benchmarks, yet RL-tuned models exhibit greater robustness, experiencing markedly smaller performance drops than base or SFT models. This suggests that RL may not primarily enhance the execution of external plans but rather empower models to formulate and follow internal strategies better suited to their reasoning processes. Conversely, we observe that RL enhances models' ability to integrate provided knowledge into their reasoning process, yielding consistent gains across diverse tasks. Finally, we study whether difficult problems -- those yielding no RL signals and mixed-quality reasoning traces -- can still be effectively used for training. We introduce SparkleRL-PSS, a multi-stage RL pipeline that reuses hard problems with partial step scaffolding, guiding exploration effectively without additional data generation. Together, our findings provide a principled foundation for understanding how RL shapes model behavior, offering practical insights for building more adaptive, data-efficient, and interpretable RL pipelines for reasoning tasks. Our code, data, and checkpoints are available at: this https URL.

**arXiv ID:** 2506.04723
</details>

<details>
<summary><strong>How to Train Your LLM Web Agent: A Statistical Diagnosis</strong> - Dheeraj Vattikonda, Santhoshi Ravichandran, Emiliano Penaloza, Hadi Nekoei, Megh Thakkar, Thibault Le Sellier de Chezelles, Nicolas Gontier, Miguel Muñoz-Mármol, Sahar Omidi Shayegan, Stefania Raimondo, Xue Liu, Alexandre Drouin, Laurent Charlin, Alexandre Piché, Alexandre Lacoste, Massimo Caccia - [[pdf]](https://arxiv.org/pdf/2507.04103)</summary>

**Abstract:** LLM-based web agents have recently made significant progress, but much of it has occurred in closed-source systems, widening the gap with open-source alternatives. Progress has been held back by two key challenges: first, a narrow focus on single-step tasks that overlooks the complexity of multi-step web interactions; and second, the high compute costs required to post-train LLM-based web agents. To address this, we present the first statistically grounded study on compute allocation for LLM web-agent post-training. Our approach uses a two-stage pipeline, training a Llama 3.1 8B student to imitate a Llama 3.3 70B teacher via supervised fine-tuning (SFT), followed by on-policy reinforcement learning. We find this process highly sensitive to hyperparameter choices, making exhaustive sweeps impractical. To spare others from expensive trial-and-error, we sample 1,370 configurations and use bootstrapping to estimate effective hyperparameters. Our results show that combining SFT with on-policy RL consistently outperforms either approach alone on both WorkArena and MiniWob++. Further, this strategy requires only 55% of the compute to match the peak performance of pure SFT on MiniWob++, effectively pushing the compute-performance Pareto frontier, and is the only strategy that can close the gap with closed-source models.

**arXiv ID:** 2507.04103
</details>

<details>
<summary><strong>SimuRA: A World-Model-Driven Simulative Reasoning Architecture for General Goal-Oriented Agents</strong> - Mingkai Deng, Jinyu Hou, Zhiting Hu, Eric Xing - [[pdf]](https://arxiv.org/pdf/2507.23773)</summary>

**Abstract:** AI agents built on foundation models hold enormous promise. Current practice, however, focuses on a one-task-one-agent approach, which not only falls short of scalability and generality, but also faces practical limitations from black-box autoregressive reasoning, where decisions unfold token by token without explicit simulation or counterfactual evaluation of outcomes. Humans, on the other hand, reason and plan by mentally simulating the consequences of actions within an internal model of the world -- a capability that supports flexible, goal-directed behavior across diverse contexts. Moving towards a more general and powerful AI agent, we introduce SimuRA, a goal-oriented architecture for generalized agentic reasoning. Based on a principled formulation of an optimal agent in any general environment, SimuRA addresses the limitations of black-box autoregressive reasoning by incorporating the world model for planning via simulation. Our prototype world model is implemented using LLMs as a substrate, leveraging the natural language as a discrete, hierarchical representation grounded in concepts for planning, while remaining model-agnostic. On complex web-browsing tasks such as flight search, SimuRA improves the success rate from 0% to 32.2% compared to a representative open-web agent baseline. Across tasks, world-model-based planning achieves up to 124% higher task completion rates than a matched black-box autoregressive baseline, demonstrating the advantages of simulative reasoning. We release ReasonerAgent-Web, a web-browsing agent built on SimuRA, as an open-source research demo.

**arXiv ID:** 2507.23773
</details>

<details>
<summary><strong>On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning</strong> - Anas Barakat, Souradip Chakraborty, Peihong Yu, Pratap Tokekar, Amrit Singh Bedi - [[pdf]](https://arxiv.org/pdf/2410.04108)</summary>

**Abstract:** Reinforcement learning with general utilities (RLGU) offers a unifying framework to capture several problems beyond standard expected returns, including imitation learning, pure exploration, and safe RL. Despite recent fundamental advances in the theoretical analysis of policy gradient (PG) methods for standard RL and recent efforts in RLGU, the understanding of these PG algorithms and their scope of application in RLGU still remain limited. In this work, we establish global optimality guarantees of PG methods for RLGU in which the objective is a general concave utility function of the state-action occupancy measure. In the tabular setting, we provide global optimality results using a new proof technique building on recent theoretical developments on the convergence of PG methods for standard RL using gradient domination. Our proof technique opens avenues for analyzing policy parameterizations beyond the direct policy parameterization for RLGU. In addition, we provide global optimality results for large state-action space settings beyond prior work which has mostly focused on the tabular setting. In this large scale setting, we adapt PG methods by approximating occupancy measures within a function approximation class using maximum likelihood estimation. Our sample complexity only scales with the dimension induced by our approximation class instead of the size of the state-action space.

**arXiv ID:** 2410.04108
</details>

<details>
<summary><strong>Online Intrinsic Rewards for Decision Making Agents from Large Language Model Feedback</strong> - Qinqing Zheng, Mikael Henaff, Amy Zhang, Aditya Grover, Brandon Amos - [[pdf]](https://arxiv.org/pdf/2410.23022)</summary>

**Abstract:** Automatically synthesizing dense rewards from natural language descriptions is a promising paradigm in reinforcement learning (RL), with applications to sparse reward problems, open-ended exploration, and hierarchical skill design. Recent works have made promising steps by exploiting the prior knowledge of large language models (LLMs). However, these approaches suffer from important limitations: they are either not scalable to problems requiring billions of environment samples, due to requiring LLM annotations for each observation, or they require a diverse offline dataset, which may not exist or be impossible to collect. In this work, we address these limitations through a combination of algorithmic and systems-level contributions. We propose ONI, a distributed architecture that simultaneously learns an RL policy and an intrinsic reward function using LLM feedback. Our approach annotates the agent's collected experience via an asynchronous LLM server, which is then distilled into an intrinsic reward model. We explore a range of algorithmic choices for reward modeling with varying complexity, including hashing, classification, and ranking models. Our approach achieves state-of-the-art performance across a range of challenging tasks from the NetHack Learning Environment, while removing the need for large offline datasets required by prior work. We make our code available at this https URL.

**arXiv ID:** 2410.23022
</details>

<details>
<summary><strong>Reinforcement Learning for Reasoning in Large Language Models with One Training Example</strong> - Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen - [[pdf]](https://arxiv.org/pdf/2504.20571)</summary>

**Abstract:** We show that reinforcement learning with verifiable reward using one training example (1-shot RLVR) is effective in incentivizing the math reasoning capabilities of large language models (LLMs). Applying RLVR to the base model Qwen2.5-Math-1.5B, we identify a single example that elevates model performance on MATH500 from 36.0% to 73.6% (8.6% improvement beyond format correction), and improves the average performance across six common mathematical reasoning benchmarks from 17.6% to 35.7% (7.0% non-format gain). This result matches the performance obtained using the 1.2k DeepScaleR subset (MATH500: 73.6%, average: 35.9%), which contains the aforementioned example. Furthermore, RLVR with only two examples even slightly exceeds these results (MATH500: 74.8%, average: 36.6%). Similar substantial improvements are observed across various models (Qwen2.5-Math-7B, Llama3.2-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B), RL algorithms (GRPO and PPO), and different math examples. In addition, we identify some interesting phenomena during 1-shot RLVR, including cross-category generalization, increased frequency of self-reflection, and sustained test performance improvement even after the training accuracy has saturated, a phenomenon we term post-saturation generalization. Moreover, we verify that the effectiveness of 1-shot RLVR primarily arises from the policy gradient loss, distinguishing it from the "grokking" phenomenon. We also show the critical role of promoting exploration (e.g., by incorporating entropy loss with an appropriate coefficient) in 1-shot RLVR training. We also further discuss related observations about format correction, label robustness and prompt modification. These findings can inspire future work on RLVR efficiency and encourage a re-examination of recent progress and the underlying mechanisms in RLVR. All resources are open source at this https URL.

**arXiv ID:** 2504.20571
</details>

<details>
<summary><strong>Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning</strong> - Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, Wenhu Chen - [[pdf]](https://arxiv.org/pdf/2505.15966)</summary>

**Abstract:** Chain-of-thought reasoning has significantly improved the performance of Large Language Models (LLMs) across various domains. However, this reasoning process has been confined exclusively to textual space, limiting its effectiveness in visually intensive tasks. To address this limitation, we introduce the concept of reasoning in the pixel-space. Within this novel framework, Vision-Language Models (VLMs) are equipped with a suite of visual reasoning operations, such as zoom-in and select-frame. These operations enable VLMs to directly inspect, interrogate, and infer from visual evidences, thereby enhancing reasoning fidelity for visual tasks. Cultivating such pixel-space reasoning capabilities in VLMs presents notable challenges, including the model's initially imbalanced competence and its reluctance to adopt the newly introduced pixel-space operations. We address these challenges through a two-phase training approach. The first phase employs instruction tuning on synthesized reasoning traces to familiarize the model with the novel visual operations. Following this, a reinforcement learning (RL) phase leverages a curiosity-driven reward scheme to balance exploration between pixel-space reasoning and textual reasoning. With these visual operations, VLMs can interact with complex visual inputs, such as information-rich images or videos to proactively gather necessary information. We demonstrate that this approach significantly improves VLM performance across diverse visual reasoning benchmarks. Our 7B model, \model, achieves 84\% on V* bench, 74\% on TallyQA-Complex, and 84\% on InfographicsVQA, marking the highest accuracy achieved by any open-source model to date. These results highlight the importance of pixel-space reasoning and the effectiveness of our framework.

**arXiv ID:** 2505.15966
</details>

<details>
<summary><strong>T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning</strong> - Amartya Chakraborty, Paresh Dashore, Nadia Bathaee, Anmol Jain, Anirban Das, Shi-Xiong Zhang, Sambit Sahu, Milind Naphade, Genta Indra Winata - [[pdf]](https://arxiv.org/pdf/2505.16986)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated impressive capabilities as intelligent agents capable of solving complex problems. However, effective planning in scenarios involving dependencies between API or tool calls-particularly in multi-turn conversations-remains a significant challenge. To address this, we introduce T1, a tool-augmented, multi-domain, multi-turn conversational dataset specifically designed to capture and manage inter-tool dependencies across diverse domains. T1 enables rigorous evaluation of agents' ability to coordinate tool use across nine distinct domains (4 single domain and 5 multi-domain) with the help of an integrated caching mechanism for both short- and long-term memory, while supporting dynamic replanning-such as deciding whether to recompute or reuse cached results. Beyond facilitating research on tool use and planning, T1 also serves as a benchmark for evaluating the performance of open-weight and proprietary large language models. We present results powered by T1-Agent, highlighting their ability to plan and reason in complex, tool-dependent scenarios.

**arXiv ID:** 2505.16986
</details>

<details>
<summary><strong>Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning</strong> - Ghada Sokar, Pablo Samuel Castro - [[pdf]](https://arxiv.org/pdf/2505.17749)</summary>

**Abstract:** Scaling deep reinforcement learning in pixel-based environments presents a significant challenge, often resulting in diminished performance. While recent works have proposed algorithmic and architectural approaches to address this, the underlying cause of the performance drop remains unclear. In this paper, we identify the connection between the output of the encoder (a stack of convolutional layers) and the ensuing dense layers as the main underlying factor limiting scaling capabilities; we denote this connection as the bottleneck, and we demonstrate that previous approaches implicitly target this bottleneck. As a result of our analyses, we present global average pooling as a simple yet effective way of targeting the bottleneck, thereby avoiding the complexity of earlier approaches.

**arXiv ID:** 2505.17749
</details>

<details>
<summary><strong>R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning</strong> - Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng, Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue Yin, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2505.23794)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) integrates external knowledge with Large Language Models (LLMs) to enhance factual correctness and mitigate hallucination. However, dense retrievers often become the bottleneck of RAG systems due to their limited parameters compared to LLMs and their inability to perform step-by-step reasoning. While prompt-based iterative RAG attempts to address these limitations, it is constrained by human-designed workflows. To address these limitations, we propose $\textbf{R3-RAG}$, which uses $\textbf{R}$einforcement learning to make the LLM learn how to $\textbf{R}$eason and $\textbf{R}$etrieve step by step, thus retrieving comprehensive external knowledge and leading to correct answers. R3-RAG is divided into two stages. We first use cold start to make the model learn the manner of iteratively interleaving reasoning and retrieval. Then we use reinforcement learning to further harness its ability to better explore the external retrieval environment. Specifically, we propose two rewards for R3-RAG: 1) answer correctness for outcome reward, which judges whether the trajectory leads to a correct answer; 2) relevance-based document verification for process reward, encouraging the model to retrieve documents that are relevant to the user question, through which we can let the model learn how to iteratively reason and retrieve relevant documents to get the correct answer. Experimental results show that R3-RAG significantly outperforms baselines and can transfer well to different retrievers. We release R3-RAG at this https URL.

**arXiv ID:** 2505.23794
</details>

<details>
<summary><strong>Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning</strong> - Haozhen Zhang, Tao Feng, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2506.09033)</summary>

**Abstract:** The rapid emergence of diverse large language models (LLMs) has spurred the development of LLM routers that assign user queries to the most suitable model. However, existing LLM routers typically perform a single-round, one-to-one mapping (\textit{i.e.}, assigning each query to a single model in isolation), which limits their capability to tackle complex tasks that demand the complementary strengths of multiple LLMs. In this paper, we present \textbf{Router-R1}, a reinforcement learning (RL)-based framework that formulates multi-LLM routing and aggregation as a sequential decision process. Router-R1 instantiates the router itself as a capable LLM, leveraging its reasoning ability to interleave "think" actions (internal deliberation) with "route" actions (dynamic model invocation), and integrates each response into its evolving context. To facilitate learning, we employ a lightweight rule-based reward comprising format rewards, final outcome rewards, and a novel cost reward for optimizing the balance between performance and cost, opening a pathway toward enhancing performance-cost trade-offs via RL. Router-R1 also conditions only on simple model descriptors such as pricing, latency, and example performance, enabling strong generalization to unseen model selection. Experiments on seven general and multi-hop QA benchmarks show that Router-R1 outperforms several strong baselines, achieving superior performance while maintaining robust generalization and cost management.

**arXiv ID:** 2506.09033
</details>

<details>
<summary><strong>Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning</strong> - Ziyang Wang, Jaehong Yoon, Shoubin Yu, Md Mohaiminul Islam, Gedas Bertasius, Mohit Bansal - [[pdf]](https://arxiv.org/pdf/2507.06485)</summary>

**Abstract:** Despite advances in reinforcement learning (RL)-based video reasoning with large language models (LLMs), data collection and fine-tuning remain significant challenges. These methods often rely on large-scale supervised fine-tuning (SFT) with extensive video data and long Chain-of-Thought (CoT) annotations, making them costly and hard to scale. To address this, we present Video-RTS, a new approach to improve video reasoning capability with drastically improved data efficiency by combining data-efficient RL with a video-adaptive test-time scaling (TTS) strategy. Building on observations about the data scaling, we skip the resource-intensive SFT step and employ efficient pure-RL training with output-based rewards, requiring no additional annotations or extensive fine-tuning. Furthermore, to utilize computational resources more efficiently, we introduce a sparse-to-dense video TTS strategy that improves inference by iteratively adding frames based on output consistency. We validate our approach on multiple video reasoning benchmarks, showing that Video-RTS surpasses existing video reasoning models by 2.4% in accuracy using only 3.6% training samples. Specifically, Video-RTS achieves a 4.2% improvement on Video-Holmes, a recent and challenging video reasoning benchmark. Notably, our pure RL training and adaptive video TTS offer complementary strengths, enabling Video-RTS's strong reasoning performance.

**arXiv ID:** 2507.06485
</details>

<details>
<summary><strong>Reinforcement Learning with Action Chunking</strong> - Qiyang Li, Zhiyuan Zhou, Sergey Levine - [[pdf]](https://arxiv.org/pdf/2507.07969)</summary>

**Abstract:** We present Q-chunking, a simple yet effective recipe for improving reinforcement learning (RL) algorithms for long-horizon, sparse-reward tasks. Our recipe is designed for the offline-to-online RL setting, where the goal is to leverage an offline prior dataset to maximize the sample-efficiency of online learning. Effective exploration and sample-efficient learning remain central challenges in this setting, as it is not obvious how the offline data should be utilized to acquire a good exploratory policy. Our key insight is that action chunking, a technique popularized in imitation learning where sequences of future actions are predicted rather than a single action at each timestep, can be applied to temporal difference (TD)-based RL methods to mitigate the exploration challenge. Q-chunking adopts action chunking by directly running RL in a 'chunked' action space, enabling the agent to (1) leverage temporally consistent behaviors from offline data for more effective online exploration and (2) use unbiased $n$-step backups for more stable and efficient TD learning. Our experimental results demonstrate that Q-chunking exhibits strong offline performance and online sample efficiency, outperforming prior best offline-to-online methods on a range of long-horizon, sparse-reward manipulation tasks.

**arXiv ID:** 2507.07969
</details>

<details>
<summary><strong>PARL: Prompt-based Agents for Reinforcement Learning</strong> - Yarik Menchaca Resendiz, Roman Klinger - [[pdf]](https://arxiv.org/pdf/2510.21306)</summary>

**Abstract:** Large language models (LLMs) have demonstrated high performance on tasks expressed in natural language, particularly in zero- or few-shot settings. These are typically framed as supervised (e.g., classification) or unsupervised (e.g., clustering) problems. However, limited work evaluates LLMs as agents in reinforcement learning (RL) tasks (e.g., playing games), where learning occurs through interaction with an environment and a reward system. While prior work focused on representing tasks that rely on a language representation, we study structured, non-linguistic reasoning - such as interpreting positions in a grid world. We therefore introduce PARL (Prompt-based Agent for Reinforcement Learning), a method that uses LLMs as RL agents through prompting, without any fine-tuning. PARL encodes actions, states, and rewards in the prompt, enabling the model to learn through trial-and-error interaction. We evaluate PARL on three standard RL tasks that do not entirely rely on natural language. We show that it can match or outperform traditional RL agents in simple environments by leveraging pretrained knowledge. However, we identify performance limitations in tasks that require complex mathematical operations or decoding states and actions.

**arXiv ID:** 2510.21306
</details>

<details>
<summary><strong>Reverse Engineering Human Preferences with Reinforcement Learning</strong> - Lisa Alazraki, Tan Yi-Chern, Jon Ander Campos, Maximilian Mozes, Marek Rei, Max Bartolo - [[pdf]](https://arxiv.org/pdf/2505.15795)</summary>

**Abstract:** The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.

**arXiv ID:** 2505.15795
</details>

<details>
<summary><strong>Chain-of-Conceptual-Thought: Eliciting the Agent to Deeply Think within the Response</strong> - Qingqing Gu, Dan Wang, Yue Zhao, Xiaoyu Wang, Zhonglin Jiang, Yong Chen, Hongyan Li, Luo Ji - [[pdf]](https://arxiv.org/pdf/2510.18434)</summary>

**Abstract:** Chain-of-Thought (CoT) is widely applied to enhance the LLM capability in math, coding and reasoning tasks. However, its performance is limited for open-domain tasks, when there are no clearly defined reasoning steps or logical transitions. To mitigate such challenges, we propose a new prompt-based paradigm called Chain of Conceptual Thoughts (CoCT), which suggests the LLM first to produce the tag of concepts, then complete the detailed content following the concept. To encourage this hierarchical way of thinking, we implement the concepts with emotions, strategies and topics. We experiment with this paradigm in daily and emotional support conversations, covering tasks with both in-domain and out-of-domain concept settings. Automatic, human, and LLM-based evaluations reveal that CoCT surpasses several prompt-based baselines such as self-refine, ECoT, SoT and RAG, suggesting a potential solution of LLM prompting paradigm for a wider scope of tasks.

**arXiv ID:** 2510.18434
</details>

<details>
<summary><strong>Safety Assessment in Reinforcement Learning via Model Predictive Control</strong> - Jeff Pflueger, Michael Everett - [[pdf]](https://arxiv.org/pdf/2510.20955)</summary>

**Abstract:** Model-free reinforcement learning approaches are promising for control but typically lack formal safety guarantees. Existing methods to shield or otherwise provide these guarantees often rely on detailed knowledge of the safety specifications. Instead, this work's insight is that many difficult-to-specify safety issues are best characterized by invariance. Accordingly, we propose to leverage reversibility as a method for preventing these safety issues throughout the training process. Our method uses model-predictive path integral control to check the safety of an action proposed by a learned policy throughout training. A key advantage of this approach is that it only requires the ability to query the black-box dynamics, not explicit knowledge of the dynamics or safety constraints. Experimental results demonstrate that the proposed algorithm successfully aborts before all unsafe actions, while still achieving comparable training progress to a baseline PPO approach that is allowed to violate safety.

**arXiv ID:** 2510.20955
</details>

<details>
<summary><strong>SutureBot: A Precision Framework & Benchmark For Autonomous End-to-End Suturing</strong> - Jesse Haworth, Juo-Tung Chen, Nigel Nelson, Ji Woong Kim, Masoud Moghani, Chelsea Finn, Axel Krieger - [[pdf]](https://arxiv.org/pdf/2510.20965)</summary>

**Abstract:** Robotic suturing is a prototypical long-horizon dexterous manipulation task, requiring coordinated needle grasping, precise tissue penetration, and secure knot tying. Despite numerous efforts toward end-to-end autonomy, a fully autonomous suturing pipeline has yet to be demonstrated on physical hardware. We introduce SutureBot: an autonomous suturing benchmark on the da Vinci Research Kit (dVRK), spanning needle pickup, tissue insertion, and knot tying. To ensure repeatability, we release a high-fidelity dataset comprising 1,890 suturing demonstrations. Furthermore, we propose a goal-conditioned framework that explicitly optimizes insertion-point precision, improving targeting accuracy by 59\%-74\% over a task-only baseline. To establish this task as a benchmark for dexterous imitation learning, we evaluate state-of-the-art vision-language-action (VLA) models, including $\pi_0$, GR00T N1, OpenVLA-OFT, and multitask ACT, each augmented with a high-level task-prediction policy. Autonomous suturing is a key milestone toward achieving robotic autonomy in surgery. These contributions support reproducible evaluation and development of precision-focused, long-horizon dexterous manipulation policies necessary for end-to-end suturing. Dataset is available at: this https URL

**arXiv ID:** 2510.20965
</details>

<details>
<summary><strong>Robust Point Cloud Reinforcement Learning via PCA-Based Canonicalization</strong> - Michael Bezick, Vittorio Giammarino, Ahmed H. Qureshi - [[pdf]](https://arxiv.org/pdf/2510.20974)</summary>

**Abstract:** Reinforcement Learning (RL) from raw visual input has achieved impressive successes in recent years, yet it remains fragile to out-of-distribution variations such as changes in lighting, color, and viewpoint. Point Cloud Reinforcement Learning (PC-RL) offers a promising alternative by mitigating appearance-based brittleness, but its sensitivity to camera pose mismatches continues to undermine reliability in realistic settings. To address this challenge, we propose PCA Point Cloud (PPC), a canonicalization framework specifically tailored for downstream robotic control. PPC maps point clouds under arbitrary rigid-body transformations to a unique canonical pose, aligning observations to a consistent frame, thereby substantially decreasing viewpoint-induced inconsistencies. In our experiments, we show that PPC improves robustness to unseen camera poses across challenging robotic tasks, providing a principled alternative to domain randomization.

**arXiv ID:** 2510.20974
</details>

<details>
<summary><strong>Enhancing Tactile-based Reinforcement Learning for Robotic Control</strong> - Elle Miller, Trevor McInroe, David Abel, Oisin Mac Aodha, Sethu Vijayakumar - [[pdf]](https://arxiv.org/pdf/2510.21609)</summary>

**Abstract:** Achieving safe, reliable real-world robotic manipulation requires agents to evolve beyond vision and incorporate tactile sensing to overcome sensory deficits and reliance on idealised state information. Despite its potential, the efficacy of tactile sensing in reinforcement learning (RL) remains inconsistent. We address this by developing self-supervised learning (SSL) methodologies to more effectively harness tactile observations, focusing on a scalable setup of proprioception and sparse binary contacts. We empirically demonstrate that sparse binary tactile signals are critical for dexterity, particularly for interactions that proprioceptive control errors do not register, such as decoupled robot-object motions. Our agents achieve superhuman dexterity in complex contact tasks (ball bouncing and Baoding ball rotation). Furthermore, we find that decoupling the SSL memory from the on-policy memory can improve performance. We release the Robot Tactile Olympiad (RoTO) benchmark to standardise and promote future research in tactile-based manipulation. Project page: this https URL

**arXiv ID:** 2510.21609
</details>

<details>
<summary><strong>Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems</strong> - Marcos Negre Saura, Richard Allmendinger, Wei Pan, Theodore Papamarkou - [[pdf]](https://arxiv.org/pdf/2410.03119)</summary>

**Abstract:** Ring attractors, mathematical models inspired by neural circuit dynamics, provide a biologically plausible mechanism to improve learning speed and accuracy in Reinforcement Learning (RL). Serving as specialized brain-inspired structures that encode spatial information and uncertainty, ring attractors explicitly encode the action space, facilitate the organization of neural activity, and enable the distribution of spatial representations across the neural network in the context of Deep Reinforcement Learning (DRL). These structures also provide temporal filtering that stabilizes action selection during exploration, for example, by preserving the continuity between rotation angles in robotic control or adjacency between tactical moves in game-like environments. The application of ring attractors in the action selection process involves mapping actions to specific locations on the ring and decoding the selected action based on neural activity. We investigate the application of ring attractors by both building an exogenous model and integrating them as part of DRL agents. Our approach significantly improves state-of-the-art performance on the Atari 100k benchmark, achieving a 53% increase in performance over selected baselines.

**arXiv ID:** 2410.03119
</details>

<details>
<summary><strong>Prior-Guided Diffusion Planning for Offline Reinforcement Learning</strong> - Donghyeon Ki, JunHyeok Oh, Seong-Woong Shim, Byung-Jun Lee - [[pdf]](https://arxiv.org/pdf/2505.10881)</summary>

**Abstract:** Diffusion models have recently gained prominence in offline reinforcement learning due to their ability to effectively learn high-performing, generalizable policies from static datasets. Diffusion-based planners facilitate long-horizon decision-making by generating high-quality trajectories through iterative denoising, guided by return-maximizing objectives. However, existing guided sampling strategies such as Classifier Guidance, Classifier-Free Guidance, and Monte Carlo Sample Selection either produce suboptimal multi-modal actions, struggle with distributional drift, or incur prohibitive inference-time costs. To address these challenges, we propose Prior Guidance (PG), a novel guided sampling framework that replaces the standard Gaussian prior of a behavior-cloned diffusion model with a learnable distribution, optimized via a behavior-regularized objective. PG directly generates high-value trajectories without costly reward optimization of the diffusion model itself, and eliminates the need to sample multiple candidates at inference for sample selection. We present an efficient training strategy that applies behavior regularization in latent space, and empirically demonstrate that PG outperforms state-of-the-art diffusion policies and planners across diverse long-horizon offline RL this http URL code is available at this https URL.

**arXiv ID:** 2505.10881
</details>

<details>
<summary><strong>Real-Time Gait Adaptation for Quadrupeds using Model Predictive Control and Reinforcement Learning</strong> - Prakrut Kotecha, Ganga Nair B, Shishir Kolathaya - [[pdf]](https://arxiv.org/pdf/2510.20706)</summary>

**Abstract:** Model-free reinforcement learning (RL) has enabled adaptable and agile quadruped locomotion; however, policies often converge to a single gait, leading to suboptimal performance. Traditionally, Model Predictive Control (MPC) has been extensively used to obtain task-specific optimal policies but lacks the ability to adapt to varying environments. To address these limitations, we propose an optimization framework for real-time gait adaptation in a continuous gait space, combining the Model Predictive Path Integral (MPPI) algorithm with a Dreamer module to produce adaptive and optimal policies for quadruped locomotion. At each time step, MPPI jointly optimizes the actions and gait variables using a learned Dreamer reward that promotes velocity tracking, energy efficiency, stability, and smooth transitions, while penalizing abrupt gait changes. A learned value function is incorporated as terminal reward, extending the formulation to an infinite-horizon planner. We evaluate our framework in simulation on the Unitree Go1, demonstrating an average reduction of up to 36.48 % in energy consumption across varying target speeds, while maintaining accurate tracking and adaptive, task-appropriate gaits.

**arXiv ID:** 2510.20706
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
