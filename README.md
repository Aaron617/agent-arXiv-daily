# Agent arXiv Daily

**Last Updated:** 2025-11-26 02:16:33

**Total Papers:** 116

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (8 papers)</h2></summary>

<details>
<summary><strong>A Multimodal Conversational Agent for Tabular Data Analysis</strong> - Mohammad Nour Al Awad, Sergey Ivanov, Olga Tikhonova, Ivan Khodnenko - [[pdf]](https://arxiv.org/pdf/2511.18405)</summary>

**Abstract:** Large language models (LLMs) can reshape information processing by handling data analysis, visualization, and interpretation in an interactive, context-aware dialogue with users, including voice interaction, while maintaining high performance. In this article, we present Talk2Data, a multimodal LLM-driven conversational agent for intuitive data exploration. The system lets users query datasets with voice or text instructions and receive answers as plots, tables, statistics, or spoken explanations. Built on LLMs, the suggested design combines OpenAI Whisper automatic speech recognition (ASR) system, Qwen-coder code generation LLM/model, custom sandboxed execution tools, and Coqui library for text-to-speech (TTS) within an agentic orchestration loop. Unlike text-only analysis tools, it adapts responses across modalities and supports multi-turn dialogues grounded in dataset context. In an evaluation of 48 tasks on three datasets, our prototype achieved 95.8% accuracy with model-only generation time under 1.7 seconds (excluding ASR and execution time). A comparison across five LLM sizes (1.5B-32B) revealed accuracy-latency-cost trade-offs, with a 7B model providing the best balance for interactive use. By routing between conversation with user and code execution, constrained to a transparent sandbox, with simultaneously grounding prompts in schema-level context, the Talk2Data agent reliably retrieves actionable insights from tables while making computations verifiable. In the article, except for the Talk2Data agent itself, we discuss implications for human-data interaction, trust in LLM-driven analytics, and future extensions toward large-scale multimodal assistants.

**arXiv ID:** 2511.18405
</details>

<details>
<summary><strong>Chatbots to strengthen democracy: An interdisciplinary seminar to train identifying argumentation techniques of science denial</strong> - Ingo Siegert, Jan Nehring, Aranxa Márquez Ampudia, Matthias Busch, Stefan Hillmann - [[pdf]](https://arxiv.org/pdf/2511.17678)</summary>

**Abstract:** In recent times, discussions on social media platforms have increasingly come under scrutiny due to the proliferation of science denial and fake news. Traditional solutions, such as regulatory actions, have been implemented to mitigate the spread of misinformation; however, these measures alone are not sufficient. To complement these efforts, educational approaches are becoming essential in empowering users to critically engage with misinformation. Conversation training, through serious games or personalized methods, has emerged as a promising strategy to help users handle science denial and toxic conversation tactics. This paper suggests an interdisciplinary seminar to explore the suitability of Large Language Models (LLMs) acting as a persona of a science denier to support people in identifying misinformation and improving resilience against toxic interactions. In the seminar, groups of four to five students will develop an AI-based chatbot that enables realistic interactions with science-denial argumentation structures. The task involves planning the setting, integrating a Large Language Model to facilitate natural dialogues, implementing the chatbot using the RASA framework, and evaluating the outcomes in a user study. It is crucial that users understand what they need to do during the interaction, how to conclude it, and how the relevant information is conveyed. The seminar does not aim to develop chatbots for practicing debunking but serves to teach AI technologies and test the feasibility of this idea for future applications. The chatbot seminar is conducted as a hybrid, parallel master's module at the participating educational institutions.

**arXiv ID:** 2511.17678
</details>

<details>
<summary><strong>Research and Prototyping Study of an LLM-Based Chatbot for Electromagnetic Simulations</strong> - Albert Piwonski, Mirsad Hadžiefendić - [[pdf]](https://arxiv.org/pdf/2511.17680)</summary>

**Abstract:** This work addresses the question of how generative artificial intelligence can be used to reduce the time required to set up electromagnetic simulation models. A chatbot based on a large language model is presented, enabling the automated generation of simulation models with various functional enhancements. A chatbot-driven workflow based on the large language model Google Gemini 2.0 Flash automatically generates and solves two-dimensional finite element eddy current models using Gmsh and GetDP. Python is used to coordinate and automate interactions between the workflow components. The study considers conductor geometries with circular cross-sections of variable position and number. Additionally, users can define custom post-processing routines and receive a concise summary of model information and simulation results. Each functional enhancement includes the corresponding architectural modifications and illustrative case studies.

**arXiv ID:** 2511.17680
</details>

<details>
<summary><strong>LLM-Based Agentic Negotiation for 6G: Addressing Uncertainty Neglect and Tail-Event Risk</strong> - Hatim Chergui, Farhad Rezazadeh, Mehdi Bennis, Merouane Debbah - [[pdf]](https://arxiv.org/pdf/2511.19175)</summary>

**Abstract:** A critical barrier to the trustworthiness of sixth-generation (6G) agentic autonomous networks is the uncertainty neglect bias; a cognitive tendency for large language model (LLM)-powered agents to make high-stakes decisions based on simple averages while ignoring the tail risk of extreme events. This paper proposes an unbiased, risk-aware framework for agentic negotiation, designed to ensure robust resource allocation in 6G network slicing. Specifically, agents leverage Digital Twins (DTs) to predict full latency distributions, which are then evaluated using a formal framework from extreme value theory, namely, Conditional Value-at-Risk (CVaR). This approach fundamentally shifts the agent's objective from reasoning over the mean to reasoning over the tail, thereby building a statistically-grounded buffer against worst-case outcomes. Furthermore, our framework ensures full uncertainty awareness by requiring agents to quantify epistemic uncertainty -- confidence in their own DTs predictions -- and propagate this meta-verification to make robust decisions, preventing them from acting on unreliable data. We validate this framework in a 6G inter-slice negotiation use-case between an eMBB and a URLLC agent. The results demonstrate the profound failure of the biased, mean-based baseline, which consistently fails its SLAs with a 25\% rate. Our unbiased, CVaR-aware agent successfully mitigates this bias, eliminating SLA violations and reducing the URLLC and eMBB p99.999 latencies by around 11\%. We show this reliability comes at the rational and quantifiable cost of slightly reduced energy savings to 17\%, exposing the false economy of the biased approach. This work provides a concrete methodology for building the trustworthy autonomous systems required for 6G.

**arXiv ID:** 2511.19175
</details>

<details>
<summary><strong>Tapas Are Free! Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments</strong> - Jinwei Hu, Yi Dong, Youcheng Sun, Xiaowei Huang - [[pdf]](https://arxiv.org/pdf/2508.11425)</summary>

**Abstract:** Autonomous agents in safety-critical applications must continuously adapt to dynamic conditions without compromising performance and reliability. This work introduces TAPA (Training-free Adaptation of Programmatic Agents), a novel framework that positions large language models (LLMs) as intelligent moderators of the symbolic action space. Unlike prior programmatic agents typically generate a monolithic policy program or rely on fixed symbolic action sets, TAPA synthesizes and adapts modular programs for individual high-level actions, referred to as logical primitives. By decoupling strategic intent from execution, TAPA enables meta-agents to operate over an abstract, interpretable action space while the LLM dynamically generates, composes, and refines symbolic programs tailored to each primitive. Extensive experiments across cybersecurity and swarm intelligence domains validate TAPA's effectiveness. In autonomous DDoS defense scenarios, TAPA achieves 77.7% network uptime while maintaining near-perfect detection accuracy in unknown dynamic environments. In swarm intelligence formation control under environmental and adversarial disturbances, TAPA consistently preserves consensus at runtime where baseline methods fail. This work promotes a paradigm shift for autonomous system design in evolving environments, from policy adaptation to dynamic action adaptation.

**arXiv ID:** 2508.11425
</details>

<details>
<summary><strong>Physical Reinforcement Learning</strong> - Sam Dillavou, Shruti Mishra - [[pdf]](https://arxiv.org/pdf/2511.17789)</summary>

**Abstract:** Digital computers are power-hungry and largely intolerant of damaged components, making them potentially difficult tools for energy-limited autonomous agents in uncertain environments. Recently developed Contrastive Local Learning Networks (CLLNs) - analog networks of self-adjusting nonlinear resistors - are inherently low-power and robust to physical damage, but were constructed to perform supervised learning. In this work we demonstrate success on two simple RL problems using Q-learning adapted for simulated CLLNs. Doing so makes explicit the components (beyond the network being trained) required to enact various tools in the RL toolbox, some of which (policy function and value function) are more natural in this system than others (replay buffer). We discuss assumptions such as the physical safety that digital hardware requires, CLLNs can forgo, and biological systems cannot rely on, and highlight secondary goals that are important in biology and trainable in CLLNs, but make little sense in digital computers.

**arXiv ID:** 2511.17789
</details>

<details>
<summary><strong>Anti-Jamming based on Null-Steering Antennas and Intelligent UAV Swarm Behavior</strong> - Miguel Lourenço, António Grilo - [[pdf]](https://arxiv.org/pdf/2511.18086)</summary>

**Abstract:** Unmanned Aerial Vehicle (UAV) swarms represent a key advancement in autonomous systems, enabling coordinated missions through inter-UAV communication. However, their reliance on wireless links makes them vulnerable to jamming, which can disrupt coordination and mission success. This work investigates whether a UAV swarm can effectively overcome jamming while maintaining communication and mission efficiency.
To address this, a unified optimization framework combining Genetic Algorithms (GA), Supervised Learning (SL), and Reinforcement Learning (RL) is proposed. The mission model, structured into epochs and timeslots, allows dynamic path planning, antenna orientation, and swarm formation while progressively enforcing collision rules. Null-steering antennas enhance resilience by directing antenna nulls toward interference sources.
Results show that the GA achieved stable, collision-free trajectories but with high computational cost. SL models replicated GA-based configurations but struggled to generalize under dynamic or constrained settings. RL, trained via Proximal Policy Optimization (PPO), demonstrated adaptability and real-time decision-making with consistent communication and lower computational demand. Additionally, the Adaptive Movement Model generalized UAV motion to arbitrary directions through a rotation-based mechanism, validating the scalability of the proposed system.
Overall, UAV swarms equipped with null-steering antennas and guided by intelligent optimization algorithms effectively mitigate jamming while maintaining communication stability, formation cohesion, and collision safety. The proposed framework establishes a unified, flexible, and reproducible basis for future research on resilient swarm communication systems.

**arXiv ID:** 2511.18086
</details>

<details>
<summary><strong>LLM Chatbots in High School Programming: Exploring Behaviors and Interventions</strong> - Manuel Valle Torre, Marcus Specht, Catharine Oertel - [[pdf]](https://arxiv.org/pdf/2511.18985)</summary>

**Abstract:** This study uses a Design-Based Research (DBR) cycle to refine the integration of Large Language Models (LLMs) in high school programming education. The initial problem was identified in an Intervention Group where, in an unguided setting, a higher proportion of executive, solution-seeking queries correlated strongly and negatively with exam performance. A contemporaneous Comparison Group demonstrated that without guidance, these unproductive help-seeking patterns do not self-correct, with engagement fluctuating and eventually declining. This insight prompted a mid-course pedagogical intervention in the first group, designed to teach instrumental help-seeking. The subsequent evaluation confirmed the intervention's success, revealing a decrease in executive queries, as well as a shift toward more productive learning workflows. However, this behavioral change did not translate into a statistically significant improvement in exam grades, suggesting that altering tool-use strategies alone may be insufficient to overcome foundational knowledge gaps. The DBR process thus yields a more nuanced principle: the educational value of an LLM depends on a pedagogy that scaffolds help-seeking, but this is only one part of the complex process of learning.

**arXiv ID:** 2511.18985
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (21 papers)</h2></summary>

<details>
<summary><strong>Cognitive Inception: Agentic Reasoning against Visual Deceptions by Injecting Skepticism</strong> - Yinjie Zhao, Heng Zhao, Bihan Wen, Joey Tianyi Zhou - [[pdf]](https://arxiv.org/pdf/2511.17672)</summary>

**Abstract:** As the development of AI-generated contents (AIGC), multi-modal Large Language Models (LLM) struggle to identify generated visual inputs from real ones. Such shortcoming causes vulnerability against visual deceptions, where the models are deceived by generated contents, and the reliability of reasoning processes is jeopardized. Therefore, facing rapidly emerging generative models and diverse data distribution, it is of vital importance to improve LLMs' generalizable reasoning to verify the authenticity of visual inputs against potential deceptions. Inspired by human cognitive processes, we discovered that LLMs exhibit tendency of over-trusting the visual inputs, while injecting skepticism could significantly improve the models visual cognitive capability against visual deceptions. Based on this discovery, we propose \textbf{Inception}, a fully reasoning-based agentic reasoning framework to conduct generalizable authenticity verification by injecting skepticism, where LLMs' reasoning logic is iteratively enhanced between External Skeptic and Internal Skeptic agents. To the best of our knowledge, this is the first fully reasoning-based framework against AIGC visual deceptions. Our approach achieved a large margin of performance improvement over the strongest existing LLM baselines and SOTA performance on AEGIS benchmark.

**arXiv ID:** 2511.17672
</details>

<details>
<summary><strong>Neural Graph Navigation for Intelligent Subgraph Matching</strong> - Yuchen Ying, Yiyang Dai, Wenda Li, Wenjie Huang, Rui Wang, Tongya Zheng, Yu Wang, Hanyang Yuan, Mingli Song - [[pdf]](https://arxiv.org/pdf/2511.17939)</summary>

**Abstract:** Subgraph matching, a cornerstone of relational pattern detection in domains ranging from biochemical systems to social network analysis, faces significant computational challenges due to the dramatically growing search space. Existing methods address this problem within a filtering-ordering-enumeration framework, in which the enumeration stage recursively matches the query graph against the candidate subgraphs of the data graph. However, the lack of awareness of subgraph structural patterns leads to a costly brute-force enumeration, thereby critically motivating the need for intelligent navigation in subgraph matching. To address this challenge, we propose Neural Graph Navigation (NeuGN), a neuro-heuristic framework that transforms brute-force enumeration into neural-guided search by integrating neural navigation mechanisms into the core enumeration process. By preserving heuristic-based completeness guarantees while incorporating neural intelligence, NeuGN significantly reduces the \textit{First Match Steps} by up to 98.2\% compared to state-of-the-art methods across six real-world datasets.

**arXiv ID:** 2511.17939
</details>

<details>
<summary><strong>Psychometric Tests for AI Agents and Their Moduli Space</strong> - Przemyslaw Chojecki - [[pdf]](https://arxiv.org/pdf/2511.19262)</summary>

**Abstract:** We develop a moduli-theoretic view of psychometric test batteries for AI agents and connect it explicitly to the AAI score developed previously. First, we make precise the notion of an AAI functional on a battery and set out axioms that any reasonable autonomy/general intelligence score should satisfy. Second, we show that the composite index ('AAI-Index') defined previously is a special case of our AAI functional. Third, we introduce the notion of a cognitive core of an agent relative to a battery and define the associated AAI$_{\textrm{core}}$ score as the restriction of an AAI functional to that core. Finally, we use these notions to describe invariants of batteries under evaluation-preserving symmetries and outline how moduli of equivalent batteries are organized.

**arXiv ID:** 2511.19262
</details>

<details>
<summary><strong>AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning</strong> - Jiayi Zhang, Yiran Peng, Fanqi Kong, Yang Cheng, Yifan Wu, Zhaoyang Yu, Jinyu Xiang, Jianhao Ruan, Jinlin Wang, Maojia Song, HongZhang Liu, Xiangru Tang, Bang Liu, Chenglin Wu, Yuyu Luo - [[pdf]](https://arxiv.org/pdf/2511.19304)</summary>

**Abstract:** Humans naturally adapt to diverse environments by learning underlying rules across worlds with different dynamics, observations, and reward structures. In contrast, existing agents typically demonstrate improvements via self-evolving within a single domain, implicitly assuming a fixed environment distribution. Cross-environment learning has remained largely unmeasured: there is no standard collection of controllable, heterogeneous environments, nor a unified way to represent how agents learn. We address these gaps in two steps. First, we propose AutoEnv, an automated framework that treats environments as factorizable distributions over transitions, observations, and rewards, enabling low-cost (4.12 USD on average) generation of heterogeneous worlds. Using AutoEnv, we construct AutoEnv-36, a dataset of 36 environments with 358 validated levels, on which seven language models achieve 12-49% normalized reward, demonstrating the challenge of AutoEnv-36. Second, we formalize agent learning as a component-centric process driven by three stages of Selection, Optimization, and Evaluation applied to an improvable agent component. Using this formulation, we design eight learning methods and evaluate them on AutoEnv-36. Empirically, the gain of any single learning method quickly decrease as the number of environments increases, revealing that fixed learning methods do not scale across heterogeneous environments. Environment-adaptive selection of learning methods substantially improves performance but exhibits diminishing returns as the method space expands. These results highlight both the necessity and the current limitations of agent learning for scalable cross-environment generalization, and position AutoEnv and AutoEnv-36 as a testbed for studying cross-environment agent learning. The code is avaiable at this https URL.

**arXiv ID:** 2511.19304
</details>

<details>
<summary><strong>A Multidisciplinary Design and Optimization (MDO) Agent Driven by Large Language Models</strong> - Bingkun Guo, Wentian Li, Xiaojian Liu, Jiaqi Luo, Zibin Yu, Dalong Dong, Shuyou Zhang, Yiming Zhang - [[pdf]](https://arxiv.org/pdf/2511.17511)</summary>

**Abstract:** To accelerate mechanical design and enhance design quality and innovation, we present a Multidisciplinary Design and Optimization (MDO) Agent driven by Large Language Models (LLMs). The agent semi-automates the end-to-end workflow by orchestrating three core capabilities: (i) natural-language-driven parametric modeling, (ii) retrieval-augmented generation (RAG) for knowledge-grounded conceptualization, and (iii) intelligent orchestration of engineering software for performance verification and optimization. Working in tandem, these capabilities interpret high-level, unstructured intent, translate it into structured design representations, automatically construct parametric 3D CAD models, generate reliable concept variants using external knowledge bases, and conduct evaluation with iterative optimization via tool calls such as finite-element analysis (FEA). Validation on three representative cases - a gas-turbine blade, a machine-tool column, and a fractal heat sink - shows that the agent completes the pipeline from natural-language intent to verified and optimized designs with reduced manual scripting and setup effort, while promoting innovative design exploration. This work points to a practical path toward human-AI collaborative mechanical engineering and lays a foundation for more dependable, vertically customized MDO systems.

**arXiv ID:** 2511.17511
</details>

<details>
<summary><strong>Evaluating Device-First Continuum AI (DFC-AI) for Autonomous Operations in the Energy Sector</strong> - Siavash M. Alamouti, Fay Arjomandi, Michel Burger, Bashar Altakrouri - [[pdf]](https://arxiv.org/pdf/2511.17528)</summary>

**Abstract:** Industrial automation in the energy sector requires AI systems that can operate autonomously regardless of network availability, a requirement that cloud-centric architectures cannot meet. This paper evaluates the application of Device-First Continuum AI (DFC-AI) to critical energy sector operations. DFC-AI, a specialized architecture within the Hybrid Edge Cloud paradigm, implements intelligent agents using a microservices architecture that originates at end devices and extends across the computational continuum. Through comprehensive simulations of energy sector scenarios including drone inspections, sensor networks, and worker safety systems, we demonstrate that DFC-AI maintains full operational capability during network outages while cloud and gateway-based systems experience complete or partial failure. Our analysis reveals that zero-configuration GPU discovery and heterogeneous device clustering are particularly well-suited for energy sector deployments, where specialized nodes can handle intensive AI workloads for entire fleets of inspection drones or sensor networks. The evaluation shows that DFC-AI achieves significant latency reduction and energy savings compared to cloud architectures. Additionally, we find that gateway based edge solutions can paradoxically cost more than cloud solutions for certain energy sector workloads due to infrastructure overhead, while DFC-AI can consistently provide cost savings by leveraging enterprise-owned devices. These findings, validated through rigorous statistical analysis, establish that DFC-AI addresses the unique challenges of energy sector operations, ensuring intelligent agents remain available and functional in remote oil fields, offshore platforms, and other challenging environments characteristic of the industry.

**arXiv ID:** 2511.17528
</details>

<details>
<summary><strong>SWITCH: Benchmarking Modeling and Handling of Tangible Interfaces in Long-horizon Embodied Scenarios</strong> - Jieru Lin, Zhiwei Yu, Börje F. Karlsson - [[pdf]](https://arxiv.org/pdf/2511.17649)</summary>

**Abstract:** Autonomous intelligence requires not only perception and reasoning, but critically, effective interaction with the existing world and its infrastructure. Everyday environments are rich in tangible control interfaces (TCIs), e.g., light switches, appliance panels, and embedded GUIs, that demand commonsense and physics reasoning, but also causal prediction and outcome verification in time and space (e.g., delayed heating, remote lights). Moreover, failures here have potential safety implications, yet current benchmarks rarely test grounding, partial observability (video), or post-hoc verification in situated settings. We introduce SWITCH (Semantic World Interface Tasks for Control and Handling), an embodied, task-driven benchmark created through iterative releases to probe these gaps. Its first iteration, SWITCH-Basic, evaluates five complementary abilities:task-aware VQA, semantic UI grounding, action generation, state-transition prediction, and result verification, under egocentric RGB video input and device diversity. Across 351 tasks spanning 98 real devices and appliances, commercial and open LMMMs exhibit inconsistent performance even on single-step interactions, often over-relying on textual cues and under-using visual or video evidence (and high aggregate scores can mask such failures). SWITCH provides data, code, and held-out splits to enable reproducible evaluation and community contributions toward more challenging future iterations of the benchmark and the creation of training datasets. Benchmark resources are available at: this https URL.

**arXiv ID:** 2511.17649
</details>

<details>
<summary><strong>FHE-Agent: Automating CKKS Configuration for Practical Encrypted Inference via an LLM-Guided Agentic Framework</strong> - Nuo Xu, Zhaoting Gong, Ran Ran, Jinwei Tang, Wujie Wen, Caiwen Ding - [[pdf]](https://arxiv.org/pdf/2511.18653)</summary>

**Abstract:** Fully Homomorphic Encryption (FHE), particularly the CKKS scheme, is a promising enabler for privacy-preserving MLaaS, but its practical deployment faces a prohibitive barrier: it heavily relies on domain expertise. Configuring CKKS involves a tightly coupled space of ring dimensions, modulus chains, and packing layouts. Without deep cryptographic knowledge to navigate these interactions, practitioners are restricted to compilers that rely on fixed heuristics. These "one-shot" tools often emit rigid configurations that are either severely over-provisioned in latency or fail to find a feasible solution entirely for deeper networks.
We present FHE-Agent, an agentic framework that automates this expert reasoning process. By coupling a Large Language Model (LLM) controller with a deterministic tool suite, FHE-Agent decomposes the search into global parameter selection and layer-wise bottleneck repair. The agents operate within a multi-fidelity workflow, pruning invalid regimes using cheap static analysis and reserving expensive encrypted evaluations for the most promising candidates.
We instantiate FHE-Agent on the Orion compiler and evaluate it on standard benchmarks (MLP, LeNet, LoLa) and deeper architectures (AlexNet). FHE-Agent consistently achieves better precision and lower latency than naïve search strategies. Crucially, it automatically discovers feasible, 128-bit secure configurations for complex models where baseline heuristics and one-shot prompts fail to produce a valid setup.

**arXiv ID:** 2511.18653
</details>

<details>
<summary><strong>Rethinking Retrieval: From Traditional Retrieval Augmented Generation to Agentic and Non-Vector Reasoning Systems in the Financial Domain for Large Language Models</strong> - Elias Lumer, Matt Melich, Olivia Zino, Elena Kim, Sara Dieter, Pradeep Honaganahalli Basavaraju, Vamse Kumar Subbiah, James A. Burke, Roberto Hernandez - [[pdf]](https://arxiv.org/pdf/2511.18177)</summary>

**Abstract:** Recent advancements in Retrieval-Augmented Generation (RAG) have enabled Large Language Models to answer financial questions using external knowledge bases of U.S. SEC filings, earnings reports, and regulatory documents. However, existing work lacks systematic comparison of vector-based and non-vector RAG architectures for financial documents, and the empirical impact of advanced RAG techniques on retrieval accuracy, answer quality, latency, and cost remain unclear. We present the first systematic evaluation comparing vector-based agentic RAG using hybrid search and metadata filtering against hierarchical node-based systems that traverse document structure without embeddings. We evaluate two enhancement techniques applied to the vector-based architecture, i) cross-encoder reranking for retrieval precision, and ii) small-to-big chunk retrieval for context completeness. Across 1,200 SEC 10-K, 10-Q, and 8-K filings on a 150-question benchmark, we measure retrieval metrics (MRR, Recall@5), answer quality through LLM-as-a-judge pairwise comparisons, latency, and preprocessing costs. Vector-based agentic RAG achieves a 68% win rate over hierarchical node-based systems with comparable latency (5.2 compared to 5.98 seconds). Cross-encoder reranking achieves a 59% absolute improvement at optimal parameters (10, 5) for MRR@5. Small-to-big retrieval achieves a 65% win rate over baseline chunking with only 0.2 seconds additional latency. Our findings reveal that applying advanced RAG techniques to financial Q&A systems improves retrieval accuracy, answer quality, and has cost-performance tradeoffs to be considered in production.

**arXiv ID:** 2511.18177
</details>

<details>
<summary><strong>LLM4Cell: A Survey of Large Language and Agentic Models for Single-Cell Biology</strong> - Sajib Acharjee Dip, Adrika Zafor, Bikash Kumar Paul, Uddip Acharjee Shuvo, Muhit Islam Emon, Xuan Wang, Liqing Zhang - [[pdf]](https://arxiv.org/pdf/2510.07793)</summary>

**Abstract:** Large language models (LLMs) and emerging agentic frameworks are beginning to transform single-cell biology by enabling natural-language reasoning, generative annotation, and multimodal data integration. However, progress remains fragmented across data modalities, architectures, and evaluation standards. LLM4Cell presents the first unified survey of 58 foundation and agentic models developed for single-cell research, spanning RNA, ATAC, multi-omic, and spatial modalities. We categorize these methods into five families-foundation, text-bridge, spatial, multimodal, epigenomic, and agentic-and map them to eight key analytical tasks including annotation, trajectory and perturbation modeling, and drug-response prediction. Drawing on over 40 public datasets, we analyze benchmark suitability, data diversity, and ethical or scalability constraints, and evaluate models across 10 domain dimensions covering biological grounding, multi-omics alignment, fairness, privacy, and explainability. By linking datasets, models, and evaluation domains, LLM4Cell provides the first integrated view of language-driven single-cell intelligence and outlines open challenges in interpretability, standardization, and trustworthy model development.

**arXiv ID:** 2510.07793
</details>

<details>
<summary><strong>ReplicationBench: Can AI Agents Replicate Astrophysics Research Papers?</strong> - Christine Ye, Sihan Yuan, Suchetha Cooray, Steven Dillmann, Ian L. V. Roque, Dalya Baron, Philipp Frank, Sergio Martin-Alvarez, Nolan Koblischke, Frank J Qu, Diyi Yang, Risa Wechsler, Ioana Ciuca - [[pdf]](https://arxiv.org/pdf/2510.24591)</summary>

**Abstract:** Frontier AI agents show increasing promise as scientific research assistants, and may eventually be useful for extended, open-ended research workflows. However, in order to use agents for novel research, we must first assess the underlying faithfulness and correctness of their work. To evaluate agents as research assistants, we introduce ReplicationBench, an evaluation framework that tests whether agents can replicate entire research papers drawn from the astrophysics literature. Astrophysics, where research relies heavily on archival data and computational study while requiring little real-world experimentation, is a particularly useful testbed for AI agents in scientific research. We split each paper into tasks which require agents to replicate the paper's core contributions, including the experimental setup, derivations, data analysis, and codebase. Each task is co-developed with the original paper authors and targets a key scientific result, enabling objective evaluation of both faithfulness (adherence to original methods) and correctness (technical accuracy of results). ReplicationBench is extremely challenging for current frontier language models: even the best-performing language models score under 20%. We analyze ReplicationBench trajectories in collaboration with domain experts and find a rich, diverse set of failure modes for agents in scientific research. ReplicationBench establishes the first benchmark of paper-scale, expert-validated astrophysics research tasks, reveals insights about agent performance generalizable to other domains of data-driven science, and provides a scalable framework for measuring AI agents' reliability in scientific research.

**arXiv ID:** 2510.24591
</details>

<details>
<summary><strong>GEM: Gaussian Embedding Modeling for Out-of-Distribution Detection in GUI Agents</strong> - Zheng Wu, Pengzhou Cheng, Zongru Wu, Lingzhong Dong, Zhuosheng Zhang - [[pdf]](https://arxiv.org/pdf/2505.12842)</summary>

**Abstract:** Graphical user interface (GUI) agents have recently emerged as an intriguing paradigm for human-computer interaction, capable of automatically executing user instructions to operate intelligent terminal devices. However, when encountering out-of-distribution (OOD) instructions that violate environmental constraints or exceed the current capabilities of agents, GUI agents may suffer task breakdowns or even pose security threats. Therefore, effective OOD detection for GUI agents is essential. Traditional OOD detection methods perform suboptimally in this domain due to the complex embedding space and evolving GUI environments. In this work, we observe that the in-distribution input semantic space of GUI agents exhibits a clustering pattern with respect to the distance from the centroid. Based on the finding, we propose GEM, a novel method based on fitting a Gaussian mixture model over input embedding distances extracted from the GUI agent that reflect its capability boundary. Evaluated on eight datasets spanning smartphones, computers, and web browsers, our method achieves an average accuracy improvement of 23.70\% over the best-performing baseline while only increasing training time by 4.9\% and testing time by 6.5\%. We also experimentally demonstrate that GEM can improve the step-wise success rate by 9.40\% by requesting assistance from the cloud model when encountering OOD samples. Analysis verifies the generalization ability of our method through experiments on nine different backbones. The codes are available at this https URL.

**arXiv ID:** 2505.12842
</details>

<details>
<summary><strong>Lane-Frame Quantum Multimodal Driving Forecasts for the Trajectory of Autonomous Vehicles</strong> - Navneet Singh, Shiva Raj Pokhrel - [[pdf]](https://arxiv.org/pdf/2511.17675)</summary>

**Abstract:** Trajectory forecasting for autonomous driving must deliver accurate, calibrated multi-modal futures under tight compute and latency constraints. We propose a compact hybrid quantum architecture that aligns quantum inductive bias with road-scene structure by operating in an ego-centric, lane-aligned frame and predicting residual corrections to a kinematic baseline instead of absolute poses. The model combines a transformer-inspired quantum attention encoder (9 qubits), a parameter-lean quantum feedforward stack (64 layers, ${\sim}1200$ trainable angles), and a Fourier-based decoder that uses shallow entanglement and phase superposition to generate 16 trajectory hypotheses in a single pass, with mode confidences derived from the latent spectrum. All circuit parameters are trained with Simultaneous Perturbation Stochastic Approximation (SPSA), avoiding backpropagation through non-analytic components. In the Waymo Open Motion Dataset, the model achieves minADE (minimum Average Displacement Error) of \SI{1.94}{m} and minFDE (minimum Final Displacement Error) of \SI{3.56}{m} in the $16$ models predicted over the horizon of \SI{2.0}{s}, consistently outperforming a kinematic baseline with reduced miss rates and strong recall. Ablations confirm that residual learning in the lane frame, truncated Fourier decoding, shallow entanglement, and spectrum-based ranking focus capacity where it matters, yielding stable optimization and reliable multi-modal forecasts from small, shallow quantum circuits on a modern autonomous-driving benchmark.

**arXiv ID:** 2511.17675
</details>

<details>
<summary><strong>SAFE-SMART: Safety Analysis and Formal Evaluation using STL Metrics for Autonomous RoboTs</strong> - Kristy Sakano, Jianyu An, Dinesh Manocha, Huan Xu - [[pdf]](https://arxiv.org/pdf/2511.17781)</summary>

**Abstract:** We present a novel, regulator-driven approach for post hoc safety evaluation of learning-based, black-box autonomous mobile robots, ensuring ongoing compliance with evolving, human-defined safety rules. In our iterative workflow, human safety requirements are translated by regulators into Signal Temporal Logic (STL) specifications. Rollout traces from the black-box model are externally verified for compliance, yielding quantitative safety metrics, Total Robustness Value (TRV) and Largest Robustness Value (LRV), which measure average and worst-case specification adherence. These metrics inform targeted retraining and iterative improvement by model designers. We apply our method across two different applications: a virtual driving scenario and an autonomous mobile robot navigating a complex environment, and observe statistically significant improvements across both scenarios. In the virtual driving scenario, we see a 177% increase in traces adhering to the simulation speed limit, a 1138% increase in traces minimizing off-road driving, and a 16% increase in traces successfully reaching the goal within the time limit. In the autonomous navigation scenario, there is a 300% increase in traces avoiding sharp turns, a 200% increase in traces reaching the goal within the time limit, and a 49% increase in traces minimizing time spent near obstacles. Finally, we validate our approach on a TurtleBot3 robot in the real world, and demonstrate improved obstacle navigation with safety buffers.

**arXiv ID:** 2511.17781
</details>

<details>
<summary><strong>Online Learning-Enhanced Lie Algebraic MPC for Robust Trajectory Tracking of Autonomous Surface Vehicles</strong> - Yinan Dong, Ziyu Xu, Tsimafei Lazouski, Sangli Teng, Maani Ghaffari - [[pdf]](https://arxiv.org/pdf/2511.18683)</summary>

**Abstract:** Autonomous surface vehicles (ASVs) are easily influenced by environmental disturbances such as wind and waves, making accurate trajectory tracking a persistent challenge in dynamic marine conditions. In this paper, we propose an efficient controller for trajectory tracking of marine vehicles under unknown disturbances by combining a convex error-state MPC on the Lie group with an online learning module to compensate for these disturbances in real time. This design enables adaptive and robust control while maintaining computational efficiency. Extensive evaluations in numerical simulations, the Virtual RobotX (VRX) simulator, and real-world field experiments demonstrate that our method achieves superior tracking accuracy under various disturbance scenarios compared with existing approaches.

**arXiv ID:** 2511.18683
</details>

<details>
<summary><strong>Autonomous Surface Selection For Manipulator-Based UV Disinfection In Hospitals Using Foundation Models</strong> - Xueyan Oh, Jonathan Her, Zhixiang Ong, Brandon Koh, Yun Hann Tan, U-Xuan Tan - [[pdf]](https://arxiv.org/pdf/2511.18709)</summary>

**Abstract:** Ultraviolet (UV) germicidal radiation is an established non-contact method for surface disinfection in medical environments. Traditional approaches require substantial human intervention to define disinfection areas, complicating automation, while deep learning-based methods often need extensive fine-tuning and large datasets, which can be impractical for large-scale deployment. Additionally, these methods often do not address scene understanding for partial surface disinfection, which is crucial for avoiding unintended UV exposure. We propose a solution that leverages foundation models to simplify surface selection for manipulator-based UV disinfection, reducing human involvement and removing the need for model training. Additionally, we propose a VLM-assisted segmentation refinement to detect and exclude thin and small non-target objects, showing that this reduces mis-segmentation errors. Our approach achieves over 92\% success rate in correctly segmenting target and non-target surfaces, and real-world experiments with a manipulator and simulated UV light demonstrate its practical potential for real-world applications.

**arXiv ID:** 2511.18709
</details>

<details>
<summary><strong>Beyond Description: Cognitively Benchmarking Fine-Grained Action for Embodied Agents</strong> - Dayong Liu, Chao Xu, Weihong Chen, Suyu Zhang, Juncheng Wang, Jiankang Deng, Baigui Sun, Yang Liu - [[pdf]](https://arxiv.org/pdf/2511.18685)</summary>

**Abstract:** Multimodal Large Language Models (MLLMs) show promising results as decision-making engines for embodied agents operating in complex, physical environments. However, existing benchmarks often prioritize high-level planning or spatial reasoning, leaving the fine-grained action intelligence required for embodied physical interaction underexplored. To address this gap, we introduce CFG-Bench, a new benchmark designed to systematically evaluate this crucial capability. CFG-Bench consists of 1,368 curated videos paired with 19,562 three-modalities question-answer pairs targeting four cognitive abilities: 1) Physical Interaction, 2) Temporal-Causal Relation, 3) Intentional Understanding, and 4) Evaluative Judgment. Together, these dimensions provide a systematic framework for assessing a model's ability to translate visual observations into actionable knowledge, moving beyond mere surface-level recognition. Our comprehensive evaluation on CFG-Bench reveals that leading MLLMs struggle to produce detailed instructions for physical interactions and exhibit profound limitations in the higher-order reasoning of intention and evaluation. Moreover, supervised fine-tuning (SFT) on our data demonstrates that teaching an MLLMs to articulate fine-grained actions directly translates to significant performance gains on established embodied benchmarks. Our analysis highlights these limitations and offers insights for developing more capable and grounded embodied agents.

**arXiv ID:** 2511.18685
</details>

<details>
<summary><strong>Towards Sensor Data Abstraction of Autonomous Vehicle Perception Systems</strong> - Hannes Reichert, Lukas Lang, Kevin Rösch, Daniel Bogdoll, Konrad Doll, Bernhard Sick, Hans-Christian Reuss, Christoph Stiller, J. Marius Zöllner - [[pdf]](https://arxiv.org/pdf/2105.06896)</summary>

**Abstract:** Full-stack autonomous driving perception modules usually consist of data-driven models based on multiple sensor modalities. However, these models might be biased to the sensor setup used for data acquisition. This bias can seriously impair the perception models' transferability to new sensor setups, which continuously occur due to the market's competitive nature. We envision sensor data abstraction as an interface between sensor data and machine learning applications for highly automated vehicles (HAD).
For this purpose, we review the primary sensor modalities, camera, lidar, and radar, published in autonomous-driving related datasets, examine single sensor abstraction and abstraction of sensor setups, and identify critical paths towards an abstraction of sensor data from multiple perception configurations.

**arXiv ID:** 2105.06896
</details>

<details>
<summary><strong>Advancing Autonomous Driving: DepthSense with Radar and Spatial Attention</strong> - Muhamamd Ishfaq Hussain, Zubia Naz, Muhammad Aasim Rafique, Moongu Jeon - [[pdf]](https://arxiv.org/pdf/2109.05265)</summary>

**Abstract:** Depth perception is crucial for spatial understanding and has traditionally been achieved through stereoscopic imaging. However, the precision of depth estimation using stereoscopic methods depends on the accurate calibration of binocular vision sensors. Monocular cameras, while more accessible, often suffer from reduced accuracy, especially under challenging imaging conditions. Optical sensors, too, face limitations in adverse environments, leading researchers to explore radar technology as a reliable alternative. Although radar provides coarse but accurate signals, its integration with fine-grained monocular camera data remains underexplored. In this research, we propose DepthSense, a novel radar-assisted monocular depth enhancement approach. DepthSense employs an encoder-decoder architecture, a Radar Residual Network, feature fusion with a spatial attention mechanism, and an ordinal regression layer to deliver precise depth estimations. We conducted extensive experiments on the nuScenes dataset to validate the effectiveness of DepthSense. Our methodology not only surpasses existing approaches in quantitative performance but also reduces parameter complexity and inference times. Our findings demonstrate that DepthSense represents a significant advancement over traditional stereo methods, offering a robust and efficient solution for depth estimation in autonomous driving. By leveraging the complementary strengths of radar and monocular camera data, DepthSense sets a new benchmark in the field, paving the way for more reliable and accurate spatial perception systems.

**arXiv ID:** 2109.05265
</details>

<details>
<summary><strong>ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving</strong> - Zhiyu Zheng, Shaoyu Chen, Haoran Yin, Xinbang Zhang, Jialv Zou, Xinggang Wang, Qian Zhang, Lefei Zhang - [[pdf]](https://arxiv.org/pdf/2510.08562)</summary>

**Abstract:** End-to-end autonomous driving (E2EAD) systems, which learn to predict future trajectories directly from sensor data, are fundamentally challenged by the inherent spatio-temporal imbalance of trajectory data. This imbalance creates a significant optimization burden, causing models to learn spurious correlations instead of robust driving logic, while also prioritizing uncertain, distant predictions, thereby compromising immediate safety. To address these issues, we propose ResAD, a novel Normalized Residual Trajectory Modeling framework. Instead of predicting the future trajectory directly, our approach reframes and simplifies the learning task by predicting the residual deviation from a deterministic inertial reference. This inertial reference serves as a strong physical prior, compelling the model to move beyond simple pattern-matching and instead focus its capacity on learning the necessary, context-driven deviations (e.g., traffic rules, obstacles) from this default, inertially-guided path. To mitigate the optimization imbalance caused by uncertain, long-term horizons, ResAD further incorporates Point-wise Normalization of the predicted residual. This technique re-weights the optimization objective, preventing large-magnitude errors associated with distant, uncertain waypoints from dominating the learning signal. On the NAVSIM v1 and v2 benchmarks, ResAD achieves state-of-the-art results of 88.8 PDMS and 85.5 EPDMS with only two denoising steps, demonstrating that ResAD significantly simplifies the learning task and improves planning performance. The code will be released to facilitate further research.

**arXiv ID:** 2510.08562
</details>

<details>
<summary><strong>Data-driven Causal Discovery for Pedestrians-Autonomous Personal Mobility Vehicle Interactions with eHMIs: From Psychological States to Walking Behaviors</strong> - Hailong Liu, Yang Li, Toshihiro Hiraoka, Takahiro Wada - [[pdf]](https://arxiv.org/pdf/2502.02805)</summary>

**Abstract:** Autonomous personal mobility vehicle (APMV) is a new type of small smart vehicle designed for mixed-traffic environments, including interactions with pedestrians. To enhance the interaction experience between pedestrians and APMVs and to prevent potential risks, it is crucial to investigate pedestrians' walking behaviors when interacting with APMVs and to understand the psychological processes underlying these behaviors. This study aims to investigate the causal relationships between subjective evaluations of pedestrians and their walking behaviors during interactions with an APMV equipped with an external human-machine interface (eHMI). An experiment of pedestrian-APMV interaction was conducted with 42 pedestrian participants, in which various eHMIs on the APMV were designed to induce participants to experience different levels of subjective evaluations and generate the corresponding walking behaviors. Based on the hypothesized model of the pedestrian's cognition-decision-behavior process, the results of causal discovery align with the previously proposed model. Furthermore, this study further analyzes the direct and total causal effects of each factor and investigates the causal processes affecting several important factors in the field of human-vehicle interaction, such as situation awareness, trust in vehicle, risk perception, hesitation in decision making, and walking behaviors.

**arXiv ID:** 2502.02805
</details>

</details>

<details open>
<summary><h2>LLM Agents (5 papers)</h2></summary>

<details>
<summary><strong>M3-Bench: Multi-Modal, Multi-Hop, Multi-Threaded Tool-Using MLLM Agent Benchmark</strong> - Yang Zhou, Mingyu Zhao, Zhenting Wang, Difei Gu, Bangwei Guo, Ruosong Ye, Ligong Han, Can Jin, Dimitris N. Metaxas - [[pdf]](https://arxiv.org/pdf/2511.17729)</summary>

**Abstract:** We present M^3-Bench, the first benchmark for evaluating multimodal tool use under the Model Context Protocol. The benchmark targets realistic, multi-hop and multi-threaded workflows that require visual grounding and textual reasoning, cross-tool dependencies, and persistence of intermediate resources across steps. We introduce a similarity-driven alignment that serializes each tool call, embeds signatures with a sentence encoder, and performs similarity-bucketed Hungarian matching to obtain auditable one-to-one correspondences. On top of this alignment, we report interpretable metrics that decouple semantic fidelity from workflow consistency. The benchmark spans 28 servers with 231 tools, and provides standardized trajectories curated through an Executor & Judge pipeline with human verification; an auxiliary four large language models (LLMs) judge ensemble reports end-task Task Completion and information grounding. Evaluations of representative state-of-the-art Multimodal LLMs (MLLMs) reveal persistent gaps in multimodal MCP tool use, particularly in argument fidelity and structure consistency, underscoring the need for methods that jointly reason over images, text, and tool graphs. Our Benchmark's anonymous repository is at this https URL

**arXiv ID:** 2511.17729
</details>

<details>
<summary><strong>Bridging Symbolic Control and Neural Reasoning in LLM Agents: The Structured Cognitive Loop</strong> - Myung Ho Kim - [[pdf]](https://arxiv.org/pdf/2511.17673)</summary>

**Abstract:** Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). At the core of SCL is Soft Symbolic Control, an adaptive governance mechanism that applies symbolic constraints to probabilistic inference, preserving neural flexibility while restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents. Code: this https URL Demo: this https URL

**arXiv ID:** 2511.17673
</details>

<details>
<summary><strong>ARISE: Agentic Rubric-Guided Iterative Survey Engine for Automated Scholarly Paper Generation</strong> - Zi Wang, Xingqiao Wang, Sangah Lee, Xiaowei Xu - [[pdf]](https://arxiv.org/pdf/2511.17689)</summary>

**Abstract:** The rapid expansion of scholarly literature presents significant challenges in synthesizing comprehensive, high-quality academic surveys. Recent advancements in agentic systems offer considerable promise for automating tasks that traditionally require human expertise, including literature review, synthesis, and iterative refinement. However, existing automated survey-generation solutions often suffer from inadequate quality control, poor formatting, and limited adaptability to iterative feedback, which are core elements intrinsic to scholarly writing.
To address these limitations, we introduce ARISE, an Agentic Rubric-guided Iterative Survey Engine designed for automated generation and continuous refinement of academic survey papers. ARISE employs a modular architecture composed of specialized large language model agents, each mirroring distinct scholarly roles such as topic expansion, citation curation, literature summarization, manuscript drafting, and peer-review-based evaluation. Central to ARISE is a rubric-guided iterative refinement loop in which multiple reviewer agents independently assess manuscript drafts using a structured, behaviorally anchored rubric, systematically enhancing the content through synthesized feedback.
Evaluating ARISE against state-of-the-art automated systems and recent human-written surveys, our experimental results demonstrate superior performance, achieving an average rubric-aligned quality score of 92.48. ARISE consistently surpasses baseline methods across metrics of comprehensiveness, accuracy, formatting, and overall scholarly rigor. All code, evaluation rubrics, and generated outputs are provided openly at this https URL

**arXiv ID:** 2511.17689
</details>

<details>
<summary><strong>Path-Constrained Retrieval: A Structural Approach to Reliable LLM Agent Reasoning Through Graph-Scoped Semantic Search</strong> - Joseph Oladokun - [[pdf]](https://arxiv.org/pdf/2511.18313)</summary>

**Abstract:** Large Language Model agents often retrieve context from knowledge bases that lack structural consistency with the agent's current reasoning state, leading to incoherent reasoning chains. We introduce Path-Constrained Retrieval (PCR), a retrieval method that combines structural graph constraints with semantic search to ensure retrieved information maintains logical relationships within a knowledge graph. PCR restricts the search space to nodes reachable from an anchor node, preventing retrieval of structurally disconnected information that may lead to inconsistent reasoning. We evaluate PCR on PathRAG-6, a benchmark spanning six domains with 180 nodes and 360 edges. Our results show that PCR achieves full structural consistency compared to 24-32 percent in baseline methods, while maintaining strong relevance scores. On the technology domain, PCR obtains full relevance at rank 10 with full structural consistency, significantly outperforming vector search and hybrid retrieval. PCR reduces the average graph distance of retrieved context by 78 percent compared to baselines, demonstrating retrieval of more structurally consistent information. These findings suggest that path-constrained retrieval is an effective approach for improving the reliability and coherence of LLM agent reasoning systems.

**arXiv ID:** 2511.18313
</details>

<details>
<summary><strong>Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly?</strong> - Chunqiu Steven Xia, Zhe Wang, Yan Yang, Yuxiang Wei, Lingming Zhang - [[pdf]](https://arxiv.org/pdf/2511.13646)</summary>

**Abstract:** Large Language Models (LLMs) are reshaping almost all industries, including software engineering. In recent years, a number of LLM agents have been proposed to solve real-world software problems. Such software agents are typically equipped with a suite of coding tools and can autonomously decide the next actions to form complete trajectories to solve end-to-end software tasks. While promising, they typically require dedicated design and may still be suboptimal, since it can be extremely challenging and costly to exhaust the entire agent scaffold design space. Recognizing that software agents are inherently software themselves that can be further refined/modified, researchers have proposed a number of self-improving software agents recently, including the Darwin-Gödel Machine (DGM). Meanwhile, such self-improving agents require costly offline training on specific benchmarks and may not generalize well across different LLMs or benchmarks. In this paper, we propose Live-SWE-agent, the first live software agent that can autonomously and continuously evolve itself on-the-fly during runtime when solving real-world software problems. More specifically, Live-SWE-agent starts with the most basic agent scaffold with only access to bash tools (e.g., mini-SWE-agent), and autonomously evolves its own scaffold implementation while solving real-world software problems. Our evaluation on the widely studied SWE-bench Verified benchmark shows that LIVE-SWE-AGENT can achieve an impressive solve rate of 77.4% without test-time scaling, outperforming all existing software agents, including the best proprietary solution. Moreover, Live-SWE-agent outperforms state-of-the-art manually crafted software agents on the recent SWE-Bench Pro benchmark, achieving the best-known solve rate of 45.8%.

**arXiv ID:** 2511.13646
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (34 papers)</h2></summary>

<details>
<summary><strong>MAGMA-Edu: Multi-Agent Generative Multimodal Framework for Text-Diagram Educational Question Generation</strong> - Zhenyu Wu, Jian Li, Hua Huang - [[pdf]](https://arxiv.org/pdf/2511.18714)</summary>

**Abstract:** Educational illustrations play a central role in communicating abstract concepts, yet current multimodal large language models (MLLMs) remain limited in producing pedagogically coherent and semantically consistent educational visuals. We introduce MAGMA-Edu, a self-reflective multi-agent framework that unifies textual reasoning and diagrammatic synthesis for structured educational problem generation. Unlike existing methods that treat text and image generation independently, MAGMA-Edu employs a two-stage co-evolutionary pipeline: (1) a generation-verification-reflection loop that iteratively refines question statements and solutions for mathematical accuracy, and (2) a code-based intermediate representation that enforces geometric fidelity and semantic alignment during image rendering. Both stages are guided by internal self-reflection modules that evaluate and revise outputs until domain-specific pedagogical constraints are met. Extensive experiments on multimodal educational benchmarks demonstrate the superiority of MAGMA-Edu over state-of-the-art MLLMs. Compared to GPT-4o, MAGMA-Edu improves the average textual metric from 57.01 to 92.31 (+35.3 pp) and boosts image-text consistency (ITC) from 13.20 to 85.24 (+72 pp). Across all model backbones, MAGMA-Edu achieves the highest scores (Avg-Text 96.20, ITC 99.12), establishing a new state of the art for multimodal educational content generation and demonstrating the effectiveness of self-reflective multi-agent collaboration in pedagogically aligned vision-language reasoning.

**arXiv ID:** 2511.18714
</details>

<details>
<summary><strong>Hierarchical Adaptive Consensus Network: A Dynamic Framework for Scalable Consensus in Collaborative Multi-Agent AI Systems</strong> - Rathin Chandra Shit, Sharmila Subudhi - [[pdf]](https://arxiv.org/pdf/2511.17586)</summary>

**Abstract:** The consensus strategies used in collaborative multi-agent systems (MAS) face notable challenges related to adaptability, scalability, and convergence certainties. These approaches, including structured workflows, debate models, and iterative voting, often lead to communication bottlenecks, stringent decision-making processes, and delayed responses in solving complex and evolving tasks. This article introduces a three-tier architecture, the Hierarchical Adaptive Consensus Network (\hacn), which suggests various consensus policies based on task characterization and agent performance metrics. The first layer collects the confidence-based voting outcomes of several local agent clusters. In contrast, the second level facilitates inter-cluster communication through cross-clustered partial knowledge sharing and dynamic timeouts. The third layer provides system-wide coordination and final arbitration by employing a global orchestration framework with adaptable decision rules. The proposed model achieves $\bigO(n)$ communication complexity, as opposed to the $\bigO(n^2)$ complexity of the existing fully connected MAS. Experiments performed in a simulated environment yielded a 99.9\% reduction in communication overhead during consensus convergence. Furthermore, the proposed approach ensures consensus convergence through hierarchical escalation and dynamic adaptation for a wide variety of complicated tasks.

**arXiv ID:** 2511.17586
</details>

<details>
<summary><strong>From Competition to Coordination: Market Making as a Scalable Framework for Safe and Aligned Multi-Agent LLM Systems</strong> - Brendan Gho, Suman Muppavarapu, Afnan Shaik, Tyson Tsay, James Begin, Kevin Zhu, Archana Vaidheeswaran, Vasu Sharma - [[pdf]](https://arxiv.org/pdf/2511.17621)</summary>

**Abstract:** As foundation models are increasingly deployed as interacting agents in multi-agent systems, their collective behavior raises new challenges for trustworthiness, transparency, and accountability. Traditional coordination mechanisms, such as centralized oversight or adversarial adjudication, struggle to scale and often obscure how decisions emerge. We introduce a market-making framework for multi-agent large language model (LLM) coordination that organizes agent interactions as structured economic exchanges. In this setup, each agent acts as a market participant, updating and trading probabilistic beliefs, to converge toward shared, truthful outcomes. By aligning local incentives with collective epistemic goals, the framework promotes self-organizing, verifiable reasoning without requiring external enforcement. Empirically, we evaluate this approach across factual reasoning, ethical judgment, and commonsense inference tasks. Market-based coordination yields accuracy gains of up to 10% over single-shot baselines while preserving interpretability and transparency of intermediate reasoning steps. Beyond these improvements, our findings demonstrate that economic coordination principles can operationalize accountability and robustness in multi-agent LLM systems, offering a scalable pathway toward self-correcting, socially responsible AI capable of maintaining trust and oversight in real world deployment scenarios.

**arXiv ID:** 2511.17621
</details>

<details>
<summary><strong>Dialogue Diplomats: An End-to-End Multi-Agent Reinforcement Learning System for Automated Conflict Resolution and Consensus Building</strong> - Deepak Bolleddu - [[pdf]](https://arxiv.org/pdf/2511.17654)</summary>

**Abstract:** Conflict resolution and consensus building represent critical challenges in multi-agent systems, negotiations, and collaborative decision-making processes. This paper introduces Dialogue Diplomats, a novel end-to-end multi-agent reinforcement learning (MARL) framework designed for automated conflict resolution and consensus building in complex, dynamic environments. The proposed system integrates advanced deep reinforcement learning architectures with dialogue-based negotiation protocols, enabling autonomous agents to engage in sophisticated conflict resolution through iterative communication and strategic adaptation. We present three primary contributions: first, a novel Hierarchical Consensus Network (HCN) architecture that combines attention mechanisms with graph neural networks to model inter-agent dependencies and conflict dynamics. second, a Progressive Negotiation Protocol (PNP) that structures multi-round dialogue interactions with adaptive concession strategies; and third, a Context-Aware Reward Shaping mechanism that balances individual agent objectives with collective consensus goals.

**arXiv ID:** 2511.17654
</details>

<details>
<summary><strong>LLM and Agent-Driven Data Analysis: A Systematic Approach for Enterprise Applications and System-level Deployment</strong> - Xi Wang, Xianyao Ling, Kun Li, Gang Yin, Liang Zhang, Jiang Wu, Annie Wang, Weizhe Wang - [[pdf]](https://arxiv.org/pdf/2511.17676)</summary>

**Abstract:** The rapid progress in Generative AI and Agent technologies is profoundly transforming enterprise data management and analytics. Traditional database applications and system deployment are fundamentally impacted by AI-driven tools, such as Retrieval-Augmented Generation (RAG) and vector database technologies, which provide new pathways for semantic querying over enterprise knowledge bases. In the meantime, data security and compliance are top priorities for organizations adopting AI technologies. For enterprise data analysis, SQL generations powered by large language models (LLMs) and AI agents, has emerged as a key bridge connecting natural language with structured data, effectively lowering the barrier to enterprise data access and improving analytical efficiency. This paper focuses on enterprise data analysis applications and system deployment, covering a range of innovative frameworks, enabling complex query understanding, multi-agent collaboration, security verification, and computational efficiency. Through representative use cases, key challenges related to distributed deployment, data security, and inherent difficulties in SQL generation tasks are discussed.

**arXiv ID:** 2511.17676
</details>

<details>
<summary><strong>A superpersuasive autonomous policy debating system</strong> - Allen Roush, Devin Gonier, John Hines, Judah Goldfeder, Philippe Martin Wyder, Sanjay Basu, Ravid Shwartz Ziv - [[pdf]](https://arxiv.org/pdf/2511.17854)</summary>

**Abstract:** The capacity for highly complex, evidence-based, and strategically adaptive persuasion remains a formidable great challenge for artificial intelligence. Previous work, like IBM Project Debater, focused on generating persuasive speeches in simplified and shortened debate formats intended for relatively lay audiences. We introduce DeepDebater, a novel autonomous system capable of participating in and winning a full, unmodified, two-team competitive policy debate. Our system employs a hierarchical architecture of specialized multi-agent workflows, where teams of LLM-powered agents collaborate and critique one another to perform discrete argumentative tasks. Each workflow utilizes iterative retrieval, synthesis, and self-correction using a massive corpus of policy debate evidence (OpenDebateEvidence) and produces complete speech transcripts, cross-examinations, and rebuttals. We introduce a live, interactive end-to-end presentation pipeline that renders debates with AI speech and animation: transcripts are surface-realized and synthesized to audio with OpenAI TTS, and then displayed as talking-head portrait videos with EchoMimic V1. Beyond fully autonomous matches (AI vs AI), DeepDebater supports hybrid human-AI operation: human debaters can intervene at any stage, and humans can optionally serve as opponents against AI in any speech, allowing AI-human and AI-AI rounds. In preliminary evaluations against human-authored cases, DeepDebater produces qualitatively superior argumentative components and consistently wins simulated rounds as adjudicated by an independent autonomous judge. Expert human debate coaches also prefer the arguments, evidence, and cases constructed by DeepDebater. We open source all code, generated speech transcripts, audio and talking head video here: this https URL

**arXiv ID:** 2511.17854
</details>

<details>
<summary><strong>AnimAgents: Coordinating Multi-Stage Animation Pre-Production with Human-Multi-Agent Collaboration</strong> - Wen-Fan Wang, Chien-Ting Lu, Jin Ping Ng, Yi-Ting Chiu, Ting-Ying Lee, Miaosen Wang, Bing-Yu Chen, Xiang 'Anthony' Chen - [[pdf]](https://arxiv.org/pdf/2511.17906)</summary>

**Abstract:** Animation pre-production lays the foundation of an animated film by transforming initial concepts into a coherent blueprint across interdependent stages such as ideation, scripting, design, and storyboarding. While generative AI tools are increasingly adopted in this process, they remain isolated, requiring creators to juggle multiple systems without integrated workflow support. Our formative study with 12 professional creative directors and independent animators revealed key challenges in their current practice: Creators must manually coordinate fragmented outputs, manage large volumes of information, and struggle to maintain continuity and creative control between stages. Based on the insights, we present AnimAgents, a human-multi-agent collaborative system that coordinates complex, multi-stage workflows through a core agent and specialized agents, supported by dedicated boards for the four major stages of pre-production. AnimAgents enables stage-aware orchestration, stage-specific output management, and element-level refinement, providing an end-to-end workflow tailored to professional practice. In a within-subjects summative study with 16 professional creators, AnimAgents significantly outperformed a strong single-agent baseline that equipped with advanced parallel image generation in coordination, consistency, information management, and overall satisfaction (p < .01). A field deployment with 4 creators further demonstrated AnimAgents' effectiveness in real-world projects.

**arXiv ID:** 2511.17906
</details>

<details>
<summary><strong>MASTEST: A LLM-Based Multi-Agent System For RESTful API Tests</strong> - Xiaoke Han, Hong Zhu - [[pdf]](https://arxiv.org/pdf/2511.18038)</summary>

**Abstract:** Testing RESTful API is increasingly important in quality assurance of cloud-native applications. Recent advances in machine learning (ML) techniques have demonstrated that various testing activities can be performed automatically by large language models (LLMs) with reasonable accuracy. This paper develops a multi-agent system called MASTEST that combines LLM-based and programmed agents to form a complete tool chain that covers the whole workflow of API test starting from generating unit and system test scenarios from API specification in the OpenAPI Swagger format, to generating of Pytest test scripts, executing test scripts to interact with web services, to analysing web service response messages to determine test correctness and calculate test coverage. The system also supports the incorporation of human testers in reviewing and correcting LLM generated test artefacts to ensure the quality of testing activities. MASTEST system is evaluated on two LLMs, GPT-4o and DeepSeek V3.1 Reasoner with five public APIs. The performances of LLMs on various testing activities are measured by a wide range of metrics, including unit and system test scenario coverage and API operation coverage for the quality of generated test scenarios, data type correctness, status code coverage and script syntax correctness for the quality of LLM generated test scripts, as well as bug detection ability and usability of LLM generated test scenarios and scripts. Experiment results demonstrated that both DeepSeek and GPT-4o achieved a high overall performance. DeepSeek excels in data type correctness and status code detection, while GPT-4o performs best in API operation coverage. For both models, LLM generated test scripts maintained 100\% syntax correctness and only required minimal manual edits for semantic correctness. These findings indicate the effectiveness and feasibility of MASTEST.

**arXiv ID:** 2511.18038
</details>

<details>
<summary><strong>MOMA-AC: A preference-driven actor-critic framework for continuous multi-objective multi-agent reinforcement learning</strong> - Adam Callaghan, Karl Mason, Patrick Mannion - [[pdf]](https://arxiv.org/pdf/2511.18181)</summary>

**Abstract:** This paper addresses a critical gap in Multi-Objective Multi-Agent Reinforcement Learning (MOMARL) by introducing the first dedicated inner-loop actor-critic framework for continuous state and action spaces: Multi-Objective Multi-Agent Actor-Critic (MOMA-AC). Building on single-objective, single-agent algorithms, we instantiate this framework with Twin Delayed Deep Deterministic Policy Gradient (TD3) and Deep Deterministic Policy Gradient (DDPG), yielding MOMA-TD3 and MOMA-DDPG. The framework combines a multi-headed actor network, a centralised critic, and an objective preference-conditioning architecture, enabling a single neural network to encode the Pareto front of optimal trade-off policies for all agents across conflicting objectives in a continuous MOMARL setting. We also outline a natural test suite for continuous MOMARL by combining a pre-existing multi-agent single-objective physics simulator with its multi-objective single-agent counterpart. Evaluating cooperative locomotion tasks in this suite, we show that our framework achieves statistically significant improvements in expected utility and hypervolume relative to outer-loop and independent training baselines, while demonstrating stable scalability as the number of agents increases. These results establish our framework as a foundational step towards robust, scalable multi-objective policy learning in continuous multi-agent domains.

**arXiv ID:** 2511.18181
</details>

<details>
<summary><strong>Hybrid Agentic AI and Multi-Agent Systems in Smart Manufacturing</strong> - Mojtaba A. Farahani, Md Irfan Khan, Thorsten Wuest - [[pdf]](https://arxiv.org/pdf/2511.18258)</summary>

**Abstract:** The convergence of Agentic AI and MAS enables a new paradigm for intelligent decision making in SMS. Traditional MAS architectures emphasize distributed coordination and specialized autonomy, while recent advances in agentic AI driven by LLMs introduce higher order reasoning, planning, and tool orchestration capabilities. This paper presents a hybrid agentic AI and multi agent framework for a Prescriptive Maintenance use case, where LLM based agents provide strategic orchestration and adaptive reasoning, complemented by rule based and SLMs agents performing efficient, domain specific tasks on the edge. The proposed framework adopts a layered architecture that consists of perception, preprocessing, analytics, and optimization layers, coordinated through an LLM Planner Agent that manages workflow decisions and context retention. Specialized agents autonomously handle schema discovery, intelligent feature analysis, model selection, and prescriptive optimization, while a HITL interface ensures transparency and auditability of generated maintenance recommendations. This hybrid design supports dynamic model adaptation, cost efficient maintenance scheduling, and interpretable decision making. An initial proof of concept implementation is validated on two industrial manufacturing datasets. The developed framework is modular and extensible, supporting seamless integration of new agents or domain modules as capabilities evolve. The results demonstrate the system capability to automatically detect schema, adapt preprocessing pipelines, optimize model performance through adaptive intelligence, and generate actionable, prioritized maintenance recommendations. The framework shows promise in achieving improved robustness, scalability, and explainability for RxM in smart manufacturing, bridging the gap between high level agentic reasoning and low level autonomous execution.

**arXiv ID:** 2511.18258
</details>

<details>
<summary><strong>Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems</strong> - Xiaoqing Wang, Keman Huang, Bin Liang, Hongyu Li, Xiaoyong Du - [[pdf]](https://arxiv.org/pdf/2511.18467)</summary>

**Abstract:** The rapid advancement of Large Language Model (LLM)-driven multi-agent systems has significantly streamlined software developing tasks, enabling users with little technical expertise to develop executable applications. While these systems democratize software creation through natural language requirements, they introduce significant security risks that remain largely unexplored. We identify two risky scenarios: Malicious User with Benign Agents (MU-BA) and Benign User with Malicious Agents (BU-MA). We introduce the Implicit Malicious Behavior Injection Attack (IMBIA), demonstrating how multi-agent systems can be manipulated to generate software with concealed malicious capabilities beneath seemingly benign applications, and propose Adv-IMBIA as a defense mechanism. Evaluations across ChatDev, MetaGPT, and AgentVerse frameworks reveal varying vulnerability patterns, with IMBIA achieving attack success rates of 93%, 45%, and 71% in MU-BA scenarios, and 71%, 84%, and 45% in BU-MA scenarios. Our defense mechanism reduced attack success rates significantly, particularly in the MU-BA scenario. Further analysis reveals that compromised agents in the coding and testing phases pose significantly greater security risks, while also identifying critical agents that require protection against malicious user exploitation. Our findings highlight the urgent need for robust security measures in multi-agent software development systems and provide practical guidelines for implementing targeted, resource-efficient defensive strategies.

**arXiv ID:** 2511.18467
</details>

<details>
<summary><strong>An Analysis of Constraint-Based Multi-Agent Pathfinding Algorithms</strong> - Hannah Lee, James D. Motes, Marco Morales, Nancy M. Amato - [[pdf]](https://arxiv.org/pdf/2511.18604)</summary>

**Abstract:** This study informs the design of future multi-agent pathfinding (MAPF) and multi-robot motion planning (MRMP) algorithms by guiding choices based on constraint classification for constraint-based search algorithms. We categorize constraints as conservative or aggressive and provide insights into their search behavior, focusing specifically on vanilla Conflict-Based Search (CBS) and Conflict-Based Search with Priorities (CBSw/P). Under a hybrid grid-roadmap representation with varying resolution, we observe that aggressive (priority constraint) formulations tend to solve more instances as agent count or resolution increases, whereas conservative (motion constraint) formulations yield stronger solution quality when both succeed. Findings are synthesized in a decision flowchart, aiding users in selecting suitable constraints. Recommendations extend to Multi-Robot Motion Planning (MRMP), emphasizing the importance of considering topological features alongside problem, solution, and representation features. A comprehensive exploration of the study, including raw data and map performance, is available in our public GitHub Repository: this https URL

**arXiv ID:** 2511.18604
</details>

<details>
<summary><strong>Multi-Agent Coordination in Autonomous Vehicle Routing: A Simulation-Based Study of Communication, Memory, and Routing Loops</strong> - KM Khalid Saifullah, Daniel Palmer - [[pdf]](https://arxiv.org/pdf/2511.17656)</summary>

**Abstract:** Multi-agent coordination is critical for next-generation autonomous vehicle (AV) systems, yet naive implementations of communication-based rerouting can lead to catastrophic performance degradation. This study investigates a fundamental problem in decentralized multi-agent navigation: routing loops, where vehicles without persistent obstacle memory become trapped in cycles of inefficient path recalculation. Through systematic simulation experiments involving 72 unique configurations across varying vehicle densities (15, 35, 55 vehicles) and obstacle frequencies (6, 20 obstacles), we demonstrate that memory-less reactive rerouting increases average travel time by up to 682% compared to baseline conditions. To address this, we introduce Object Memory Management (OMM), a lightweight mechanism enabling agents to retain and share knowledge of previously encountered obstacles. OMM operates by maintaining a distributed blacklist of blocked nodes, which each agent consults during Dijkstra-based path recalculation, effectively preventing redundant routing attempts. Our results show that OMM-enabled coordination reduces average travel time by 75.7% and wait time by 88% compared to memory-less systems, while requiring only 1.67 route recalculations per vehicle versus 9.83 in memory-less scenarios. This work provides empirical evidence that persistent, shared memory is not merely beneficial but essential for robust multi-agent coordination in dynamic environments. The findings have implications beyond autonomous vehicles, informing the design of decentralized systems in robotics, network routing, and distributed AI. We provide a comprehensive experimental analysis, including detailed scenario breakdowns, scalability assessments, and visual documentation of the routing loop phenomenon, demonstrating OMM's critical role in preventing detrimental feedback cycles in cooperative multi-agent systems.

**arXiv ID:** 2511.17656
</details>

<details>
<summary><strong>DISPATCH -- Decentralized Informed Spatial Planning and Assignment of Tasks for Cooperative Heterogeneous Agents</strong> - Yao Liu, Sampad Mohanty, Elizabeth Ondula, Bhaskar Krishnamachari - [[pdf]](https://arxiv.org/pdf/2511.17915)</summary>

**Abstract:** Spatial task allocation in systems such as multi-robot delivery or ride-sharing requires balancing efficiency with fair service across tasks. Greedy assignment policies that match each agent to its highest-preference or lowest-cost task can maximize efficiency but often create inequities: some tasks receive disproportionately favorable service (e.g., shorter delays or better matches), while others face long waits or poor allocations.
We study fairness in heterogeneous multi-agent systems where tasks vary in preference alignment and urgency. Most existing approaches either assume centralized coordination or largely ignore fairness under partial observability. Distinct from this prior work, we establish a connection between the Eisenberg-Gale (EG) equilibrium convex program and decentralized, partially observable multi-agent learning. Building on this connection, we develop two equilibrium-informed algorithms that integrate fairness and efficiency: (i) a multi-agent reinforcement learning (MARL) framework, EG-MARL, whose training is guided by centralized fair assignment algorithms (EG and a preference-aware Hungarian method); and (ii) a stochastic online optimization mechanism that performs guided exploration and subset-based fair assignment as tasks are discovered.
We evaluate our frameworks across a range of team sizes and assignment formulations against centralized EG, Hungarian, and Min-Max Distance baselines. Both algorithms preserve the fairness-efficiency balance of the Eisenberg-Gale equilibrium under partial observability. EG-MARL achieves near-centralized coordination and reduced travel distances, while the stochastic online mechanism enables real-time allocation with competitive fairness. Together, these results demonstrate that spatially aware EG formulations can effectively guide decentralized coordination in agents with heterogeneous capabilities.

**arXiv ID:** 2511.17915
</details>

<details>
<summary><strong>Addressing Situated Teaching Needs: A Multi-Agent Framework for Automated Slide Adaptation</strong> - Binglin Liu, Yucheng Wang, Zheyuan Zhang, Jiyuan Lu, Shen Yang, Daniel Zhang-Li, Huiqin Liu, Jifan Yu - [[pdf]](https://arxiv.org/pdf/2511.18840)</summary>

**Abstract:** The adaptation of teaching slides to instructors' situated teaching needs, including pedagogical styles and their students' context, is a critical yet time-consuming task for educators. Through a series of educator interviews, we first identify and systematically categorize the key friction points that impede this adaptation process. Grounded in these findings, we introduce a novel multi-agent framework designed to automate slide adaptation based on high-level instructor specifications. An evaluation involving 16 modification requests across 8 real-world courses validates our approach. The framework's output consistently achieved high scores in intent alignment, content coherence and factual accuracy, and performed on par with baseline methods regarding visual clarity, while also demonstrating appropriate timeliness and a high operational agreement with human experts, achieving an F1 score of 0.89. This work heralds a new paradigm where AI agents handle the logistical burdens of instructional design, liberating educators to focus on the creative and strategic aspects of teaching.

**arXiv ID:** 2511.18840
</details>

<details>
<summary><strong>VIL2C: Value-of-Information Aware Low-Latency Communication for Multi-Agent Reinforcement Learning</strong> - Qian Zhang, Zhuo Sun, Yao Zhang, Zhiwen Yu, Bin Guo, Jun Zhang - [[pdf]](https://arxiv.org/pdf/2511.19146)</summary>

**Abstract:** Inter-agent communication serves as an effective mechanism for enhancing performance in collaborative multi-agent reinforcement learning(MARL) systems. However, the inherent communication latency in practical systems induces both action decision delays and outdated information sharing, impeding MARL performance gains, particularly in time-critical applications like autonomous driving. In this work, we propose a Value-of-Information aware Low-latency Communication(VIL2C) scheme that proactively adjusts the latency distribution to mitigate its effects in MARL systems. Specifically, we define a Value of Information (VOI) metric to quantify the importance of delayed message transmission based on each delayed message's importance. Moreover, we propose a progressive message reception mechanism to adaptively adjust the reception duration based on received messages. We derive the optimized VoI aware resource allocation and theoretically prove the performance advantage of the proposed VIL2C scheme. Extensive experiments demonstrate that VIL2C outperforms existing approaches under various communication conditions. These gains are attributed to the low-latency transmission of high-VoI messages via resource allocation and the elimination of unnecessary waiting periods via adaptive reception duration.

**arXiv ID:** 2511.19146
</details>

<details>
<summary><strong>From Archives to Decisions: Multi-Agent Pharmaceutical Co-Scientist for Traceable Drug Discovery and Reverse Translation</strong> - Xiaochen Zheng, Alvaro Serra, Ilya Schneider Chernov, Maddalena Marchesi, Eunice Musvasva, Tatyana Y. Doktorova - [[pdf]](https://arxiv.org/pdf/2511.18259)</summary>

**Abstract:** Pharmaceutical research and development has accumulated vast, heterogeneous archives of data. Much of this knowledge stems from discontinued programs, and reusing these archives is invaluable for reverse translation. However, in practice, such reuse is often infeasible. In this work, we introduce DiscoVerse, a multi-agent co-scientist designed to support pharmaceutical research and development. The system implements semantic retrieval, cross-document linking, and auditable synthesis on a large historical corpus from Roche. To validate our approach at real-world scale, we selected a subset of 180 molecules from the Roche research repositories, covering over 0.87 billion BPE tokens and more than four decades of research. Given that automated evaluation metrics are poorly aligned with scientific utility, we evaluate the performance of DiscoVerse using blinded expert evaluation of source-linked outputs. To our knowledge, this is the first agentic framework systematically assessed on real pharmaceutical data for reverse translation, enabled by authorized access to confidential, end-to-end drug-development archives. Our contributions include role-specialized agent designs aligned with scientist workflows; human-in-the-loop support for reverse translation; expert evaluation; and a large-scale demonstration showing promising answer accuracy and decision-making insights. In brief, across seven benchmark queries covering 180 molecules, DiscoVerse achieved near-perfect recall ($\geq 0.99$) with moderate precision ($0.71-0.91$), while qualitative assessments of discontinuation rationale and organ-specific toxicity showed faithful, source-linked synthesis across preclinical and clinical evidence.

**arXiv ID:** 2511.18259
</details>

<details>
<summary><strong>Multi-Agent Cross-Entropy Method with Monotonic Nonlinear Critic Decomposition</strong> - Yan Wang, Ke Deng, Yongli Ren - [[pdf]](https://arxiv.org/pdf/2511.18671)</summary>

**Abstract:** Cooperative multi-agent reinforcement learning (MARL) commonly adopts centralized training with decentralized execution (CTDE), where centralized critics leverage global information to guide decentralized actors. However, centralized-decentralized mismatch (CDM) arises when the suboptimal behavior of one agent degrades others' learning. Prior approaches mitigate CDM through value decomposition, but linear decompositions allow per-agent gradients at the cost of limited expressiveness, while nonlinear decompositions improve representation but require centralized gradients, reintroducing CDM. To overcome this trade-off, we propose the multi-agent cross-entropy method (MCEM), combined with monotonic nonlinear critic decomposition (NCD). MCEM updates policies by increasing the probability of high-value joint actions, thereby excluding suboptimal behaviors. For sample efficiency, we extend off-policy learning with a modified k-step return and Retrace. Analysis and experiments demonstrate that MCEM outperforms state-of-the-art methods across both continuous and discrete action benchmarks.

**arXiv ID:** 2511.18671
</details>

<details>
<summary><strong>Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models</strong> - Brennen A. Hill, Mant Koh En Wei, Thangavel Jishnuanandh - [[pdf]](https://arxiv.org/pdf/2508.02912)</summary>

**Abstract:** Robust coordination is critical for effective decision-making in multi-agent systems, especially under partial observability. A central question in Multi-Agent Reinforcement Learning (MARL) is whether to engineer communication protocols or learn them end-to-end. We investigate this dichotomy using embodied world models. We propose and compare two communication strategies for a cooperative task-allocation problem. The first, Learned Direct Communication (LDC), learns a protocol end-to-end. The second, Intention Communication, uses an engineered inductive bias: a compact, learned world model, the Imagined Trajectory Generation Module (ITGM), which uses the agent's own policy to simulate future states. A Message Generation Network (MGN) then compresses this plan into a message. We evaluate these approaches on goal-directed interaction in a grid world, a canonical abstraction for embodied AI problems, while scaling environmental complexity. Our experiments reveal that while emergent communication is viable in simple settings, the engineered, world model-based approach shows superior performance, sample efficiency, and scalability as complexity increases. These findings advocate for integrating structured, predictive models into MARL agents to enable active, goal-driven coordination.

**arXiv ID:** 2508.02912
</details>

<details>
<summary><strong>Toward Adaptive Categories: Dimensional Governance for Agentic AI</strong> - Zeynep Engin, David Hand - [[pdf]](https://arxiv.org/pdf/2505.11579)</summary>

**Abstract:** As AI systems evolve from static tools to dynamic agents, traditional categorical governance frameworks -- based on fixed risk tiers, levels of autonomy, or human oversight models -- are increasingly insufficient on their own. Systems built on foundation models, self-supervised learning, and multi-agent architectures increasingly blur the boundaries that categories were designed to police. In this Perspective, we make the case for dimensional governance: a framework that tracks how decision authority, process autonomy, and accountability (the 3As) distribute dynamically across human-AI relationships. A critical advantage of this approach is its ability to explicitly monitor system movement toward and across key governance thresholds, enabling preemptive adjustments before risks materialize. This dimensional approach provides the necessary foundation for more adaptive categorization, enabling thresholds and classifications that can evolve with emerging capabilities. While categories remain essential for decision-making, building them upon dimensional foundations allows for context-specific adaptability and stakeholder-responsive governance that static approaches cannot achieve. We outline key dimensions, critical trust thresholds, and practical examples illustrating where rigid categorical frameworks fail -- and where a dimensional mindset could offer a more resilient and future-proof path forward for both governance and innovation at the frontier of artificial intelligence.

**arXiv ID:** 2505.11579
</details>

<details>
<summary><strong>DeepFleet: Multi-Agent Foundation Models for Mobile Robots</strong> - Ameya Agaskar, Sriram Siva, William Pickering, Kyle O'Brien, Charles Kekeh, Ang Li, Brianna Gallo Sarker, Alicia Chua, Mayur Nemade, Charun Thattai, Jiaming Di, Isaac Iyengar, Ramya Dharoor, Dino Kirouani, Jimmy Erskine, Tamir Hegazy, Scott Niekum, Usman A. Khan, Federico Pecora, Joseph W. Durham - [[pdf]](https://arxiv.org/pdf/2508.08574)</summary>

**Abstract:** We introduce DeepFleet, a suite of foundation models designed to support coordination and planning for large-scale mobile robot fleets. These models are trained on fleet movement data, including robot positions, goals, and interactions, from hundreds of thousands of robots in Amazon warehouses worldwide. DeepFleet consists of four architectures that each embody a distinct inductive bias and collectively explore key points in the design space for multi-agent foundation models: the robot-centric (RC) model is an autoregressive decision transformer operating on neighborhoods of individual robots; the robot-floor (RF) model uses a transformer with cross-attention between robots and the warehouse floor; the image-floor (IF) model applies convolutional encoding to a multi-channel image representation of the full fleet; and the graph-floor (GF) model combines temporal attention with graph neural networks for spatial relationships. In this paper, we describe these models and present our evaluation of the impact of these design choices on prediction task performance. We find that the robot-centric and graph-floor models, which both use asynchronous robot state updates and incorporate the localized structure of robot interactions, show the most promise. We also present experiments that show that these two models can make effective use of larger warehouses operation datasets as the models are scaled up.

**arXiv ID:** 2508.08574
</details>

<details>
<summary><strong>DataSage: Multi-agent Collaboration for Insight Discovery with External Knowledge Retrieval, Multi-role Debating, and Multi-path Reasoning</strong> - Xiaochuan Liu, Yuanfeng Song, Xiaoming Yin, Xing Chen - [[pdf]](https://arxiv.org/pdf/2511.14299)</summary>

**Abstract:** In today's data-driven era, fully automated end-to-end data analytics, particularly insight discovery, is critical for discovering actionable insights that assist organizations in making effective decisions. With the rapid advancement of large language models (LLMs), LLM-driven agents have emerged as a promising paradigm for automating data analysis and insight discovery. However, existing data insight agents remain limited in several key aspects, often failing to deliver satisfactory results due to: (1) insufficient utilization of domain knowledge, (2) shallow analytical depth, and (3) error-prone code generation during insight generation. To address these issues, we propose DataSage, a novel multi-agent framework that incorporates three innovative features including external knowledge retrieval to enrich the analytical context, a multi-role debating mechanism to simulate diverse analytical perspectives and deepen analytical depth, and multi-path reasoning to improve the accuracy of the generated code and insights. Extensive experiments on InsightBench demonstrate that DataSage consistently outperforms existing data insight agents across all difficulty levels, offering an effective solution for automated data insight discovery.

**arXiv ID:** 2511.14299
</details>

<details>
<summary><strong>Agent-as-a-Graph: Knowledge Graph-Based Tool and Agent Retrieval for LLM Multi-Agent Systems</strong> - Faheem Nizar, Elias Lumer, Anmol Gulati, Pradeep Honaganahalli Basavaraju, Vamse Kumar Subbiah - [[pdf]](https://arxiv.org/pdf/2511.18194)</summary>

**Abstract:** Recent advances in Large Language Model Multi-Agent Systems enable scalable orchestration and retrieval of specialized, parallelized subagents, each equipped with hundreds or thousands of Model Context Protocol (MCP) servers and tools. However, existing agent, MCP, and retrieval methods typically match queries against a single agent description, obscuring fine-grained tool capabilities of each agent, resulting in suboptimal agent selection. We introduce Agent-as-a-Graph retrieval, a knowledge graph retrieval augmented generation approach that represents both tools and their parent agents as nodes and edges in a knowledge graph. During retrieval, i) relevant agents and tool nodes are first retrieved through vector search, ii) we apply a type-specific weighted reciprocal rank fusion (wRRF) for reranking tools and agents, and iii) parent agents are traversed in the knowledge graph for the final set of agents. We evaluate Agent-as-a-Graph on the LiveMCPBenchmark, achieving 14.9% and 14.6% improvements in Recall@5 and nDCG@5 over prior state-of-the-art retrievers, and 2.4% improvements in wRRF optimizations.

**arXiv ID:** 2511.18194
</details>

<details>
<summary><strong>Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations</strong> - Yu Xia, Sungchul Kim, Tong Yu, Ryan A. Rossi, Julian McAuely - [[pdf]](https://arxiv.org/pdf/2511.18413)</summary>

**Abstract:** Agentic recommendations cast recommenders as large language model (LLM) agents that can plan, reason, use tools, and interact with users of varying preferences in web applications. However, most existing agentic recommender systems focus on generic single-agent plan-execute workflows or multi-agent task decomposition pipelines. Without recommendation-oriented design, they often underuse the collaborative signals in the user-item interaction history, leading to unsatisfying recommendation results. To address this, we propose the Multi-Agent Collaborative Filtering (MACF) framework for agentic recommendations, drawing an analogy between traditional collaborative filtering algorithms and LLM-based multi-agent collaboration. Specifically, given a target user and query, we instantiate similar users and relevant items as LLM agents with unique profiles. Each agent is able to call retrieval tools, suggest candidate items, and interact with other agents. Different from the static preference aggregation in traditional collaborative filtering, MACF employs a central orchestrator agent to adaptively manage the collaboration between user and item agents via dynamic agent recruitment and personalized collaboration instruction. Experimental results on datasets from three different domains show the advantages of our MACF framework compared to strong agentic recommendation baselines.

**arXiv ID:** 2511.18413
</details>

<details>
<summary><strong>A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval, Disambiguation and Reflective Analysis</strong> - Wenxuan Mu, Jinzhong Ning, Di Zhao, Yijia Zhang - [[pdf]](https://arxiv.org/pdf/2511.19083)</summary>

**Abstract:** In-context learning (ICL) with large language models (LLMs) has emerged as a promising paradigm for named entity recognition (NER) in low-resource scenarios. However, existing ICL-based NER methods suffer from three key limitations: (1) reliance on dynamic retrieval of annotated examples, which is problematic when annotated data is scarce; (2) limited generalization to unseen domains due to the LLM's insufficient internal domain knowledge; and (3) failure to incorporate external knowledge or resolve entity ambiguities. To address these challenges, we propose KDR-Agent, a novel multi-agent framework for multi-domain low-resource in-context NER that integrates Knowledge retrieval, Disambiguation, and Reflective analysis. KDR-Agent leverages natural-language type definitions and a static set of entity-level contrastive demonstrations to reduce dependency on large annotated corpora. A central planner coordinates specialized agents to (i) retrieve factual knowledge from Wikipedia for domain-specific mentions, (ii) resolve ambiguous entities via contextualized reasoning, and (iii) reflect on and correct model predictions through structured self-assessment. Experiments across ten datasets from five domains demonstrate that KDR-Agent significantly outperforms existing zero-shot and few-shot ICL baselines across multiple LLM backbones. The code and data can be found at this https URL.

**arXiv ID:** 2511.19083
</details>

<details>
<summary><strong>Be My Eyes: Extending Large Language Models to New Modalities Through Multi-Agent Collaboration</strong> - James Y. Huang, Sheng Zhang, Qianchu Liu, Guanghui Qin, Tinghui Zhu, Tristan Naumann, Muhao Chen, Hoifung Poon - [[pdf]](https://arxiv.org/pdf/2511.19417)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated remarkable capabilities in challenging, knowledge-intensive reasoning tasks. However, extending LLMs to perceive and reason over a new modality (e.g., vision), often requires costly development of large-scale vision language models (VLMs) with LLMs as backbones. Smaller VLMs are more efficient and adaptable but often lack the broad knowledge and reasoning capabilities of frontier LLMs. In this work, we propose BeMyEyes, a modular, multi-agent framework for extending LLMs to multimodal reasoning by orchestrating collaboration between efficient, adaptable VLMs as perceivers and powerful LLMs as reasoners through conversations. We then introduce a data synthesis and supervised fine-tuning pipeline to train the perceiver agent to effectively collaborate with the reasoner agent. By combining the complementary strengths of perception and reasoning agents, BeMyEyes avoids the need for training large-scale multimodal models, preserves the generalization and reasoning capabilities of LLMs, and allows flexible extension to new domains and modalities. Experiments show that our framework unlocks the multimodal reasoning capabilities for LLMs, enabling a lightweight and fully open-source solution, i.e. equipping text-only DeepSeek-R1 with Qwen2.5-VL-7B perceiver, to outperform large-scale proprietary VLMs such as GPT-4o on a wide range of knowledge-intensive multimodal tasks. These results demonstrate the effectiveness, modularity, and scalability of our multi-agent approach for building future multimodal reasoning systems.

**arXiv ID:** 2511.19417
</details>

<details>
<summary><strong>Advancing Multi-Agent RAG Systems with Minimalist Reinforcement Learning</strong> - Yihong Wu, Liheng Ma, Muzhi Li, Jiaming Zhou, Lei Ding, Jianye Hao, Ho-fung Leung, Irwin King, Yingxue Zhang, Jian-Yun Nie - [[pdf]](https://arxiv.org/pdf/2505.17086)</summary>

**Abstract:** Large Language Models (LLMs) equipped with modern Retrieval-Augmented Generation (RAG) systems often employ multi-turn interaction pipelines to interface with search engines for complex reasoning tasks. However, such multi-turn interactions inevitably produce long intermediate contexts, as context length grows exponentially with exploration depth. This leads to a well-known limitation of LLMs: their difficulty in effectively leveraging information from long contexts. This problem is further amplified in RAG systems that depend on in-context learning, where few-shot demonstrations must also be included in the prompt, compounding the context-length bottleneck. To address these challenges, we propose Mujica-MyGo, a unified framework for efficient multi-turn reasoning in RAG. Inspired by the divide-and-conquer principle, we introduce Mujica (Multi-hop Joint Intelligence for Complex Question Answering), a multi-agent RAG workflow that decomposes multi-turn interactions into cooperative sub-interactions, thereby mitigating long-context issues. To eliminate the dependency on in-context learning, we further develop MyGO (Minimalist Policy Gradient Optimization), a lightweight and efficient reinforcement learning algorithm that enables effective post-training of LLMs within complex RAG pipelines. We provide theoretical guarantees for MyGO's convergence to the optimal policy. Empirical evaluations across diverse question-answering benchmarks, covering both text corpora and knowledge graphs, show that Mujica-MyGO achieves superior performance.

**arXiv ID:** 2505.17086
</details>

<details>
<summary><strong>Agentic Large Language Models, a survey</strong> - Aske Plaat, Max van Duijn, Niki van Stein, Mike Preuss, Peter van der Putten, Kees Joost Batenburg - [[pdf]](https://arxiv.org/pdf/2503.23037)</summary>

**Abstract:** Background: There is great interest in agentic LLMs, large language models that act as agents.
Objectives: We review the growing body of work in this area and provide a research agenda.
Methods: Agentic LLMs are LLMs that (1) reason, (2) act, and (3) interact. We organize the literature according to these three categories.
Results: The research in the first category focuses on reasoning, reflection, and retrieval, aiming to improve decision making; the second category focuses on action models, robots, and tools, aiming for agents that act as useful assistants; the third category focuses on multi-agent systems, aiming for collaborative task solving and simulating interaction to study emergent social behavior. We find that works mutually benefit from results in other categories: retrieval enables tool use, reflection improves multi-agent collaboration, and reasoning benefits all categories.
Conclusions: We discuss applications of agentic LLMs and provide an agenda for further research. Important applications are in medical diagnosis, logistics and financial market analysis. Meanwhile, self-reflective agents playing roles and interacting with one another augment the process of scientific research itself. Further, agentic LLMs provide a solution for the problem of LLMs running out of training data: inference-time behavior generates new training states, such that LLMs can keep learning without needing ever larger datasets. We note that there is risk associated with LLM assistants taking action in the real world-safety, liability and security are open problems-while agentic LLMs are also likely to benefit society.

**arXiv ID:** 2503.23037
</details>

<details>
<summary><strong>Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj</strong> - Wei Zhen Teoh - [[pdf]](https://arxiv.org/pdf/2511.18248)</summary>

**Abstract:** Jointly forecasting trajectories of multiple interacting agents is a core challenge in sports analytics and other domains involving complex group dynamics. Accurate prediction enables realistic simulation and strategic understanding of gameplay evolution. Most existing models are evaluated solely on per-agent accuracy metrics (minADE, minFDE), which assess each agent independently on its best-of-k prediction. However these metrics overlook whether the model learns which predicted trajectories can jointly form a plausible multi-agent future. Many state-of-the-art models are designed and optimized primarily based on these metrics. As a result, they may underperform on joint predictions and also fail to generate coherent, interpretable multi-agent scenarios in team sports. We propose CausalTraj, a temporally causal, likelihood-based model that is built to generate jointly probable multi-agent trajectory forecasts. To better assess collective modeling capability, we emphasize joint metrics (minJADE, minJFDE) that measure joint accuracy across agents within the best generated scenario sample. Evaluated on the NBA SportVU, Basketball-U, and Football-U datasets, CausalTraj achieves competitive per-agent accuracy and the best recorded results on joint metrics, while yielding qualitatively coherent and realistic gameplay evolutions.

**arXiv ID:** 2511.18248
</details>

<details>
<summary><strong>MAESTRO: Multi-Agent Environment Shaping through Task and Reward Optimization</strong> - Boyuan Wu - [[pdf]](https://arxiv.org/pdf/2511.19253)</summary>

**Abstract:** Cooperative Multi-Agent Reinforcement Learning (MARL) faces two major design bottlenecks: crafting dense reward functions and constructing curricula that avoid local optima in high-dimensional, non-stationary environments. Existing approaches rely on fixed heuristics or use Large Language Models (LLMs) directly in the control loop, which is costly and unsuitable for real-time systems. We propose MAESTRO (Multi-Agent Environment Shaping through Task and Reward Optimization), a framework that moves the LLM outside the execution loop and uses it as an offline training architect. MAESTRO introduces two generative components: (i) a semantic curriculum generator that creates diverse, performance-driven traffic scenarios, and (ii) an automated reward synthesizer that produces executable Python reward functions adapted to evolving curriculum difficulty. These components guide a standard MARL backbone (MADDPG) without increasing inference cost at deployment. We evaluate MAESTRO on large-scale traffic signal control (Hangzhou, 16 intersections) and conduct controlled ablations. Results show that combining LLM-generated curricula with LLM-generated reward shaping yields improved performance and stability. Across four seeds, the full system achieves +4.0% higher mean return (163.26 vs. 156.93) and 2.2% better risk-adjusted performance (Sharpe 1.53 vs. 0.70) over a strong curriculum baseline. These findings highlight LLMs as effective high-level designers for cooperative MARL training.

**arXiv ID:** 2511.19253
</details>

<details>
<summary><strong>LLM-Driven Stationarity-Aware Expert Demonstrations for Multi-Agent Reinforcement Learning in Mobile Systems</strong> - Tianyang Duan, Zongyuan Zhang, Zheng Lin, Songxiao Guo, Xiuxian Guan, Guangyu Wu, Zihan Fang, Haotian Meng, Xia Du, Ji-Zhe Zhou, Heming Cui, Jun Luo, Yue Gao - [[pdf]](https://arxiv.org/pdf/2511.19368)</summary>

**Abstract:** Multi-agent reinforcement learning (MARL) has been increasingly adopted in many real-world applications. While MARL enables decentralized deployment on resource-constrained edge devices, it suffers from severe non-stationarity due to the synchronous updates of agent policies. This non stationarity results in unstable training and poor policy con vergence, especially as the number of agents increases. In this paper, we propose RELED, a scalable MARL framework that integrates large language model (LLM)-driven expert demonstrations with autonomous agent exploration. RELED incorporates a Stationarity-Aware Expert Demonstration module, which leverages theoretical non-stationarity bounds to enhance the quality of LLM-generated expert trajectories, thus providing high reward and training-stable samples for each agent. Moreover, a Hybrid Expert-Agent Policy Optimization module adaptively balances each agent's learning from both expert-generated and agent-generated trajectories, accelerating policy convergence and improving generalization. Extensive experiments with real city networks based on OpenStreetMap demonstrate that RELED achieves superior performance compared to state-of-the-art MARL methods.

**arXiv ID:** 2511.19368
</details>

<details>
<summary><strong>Expanding the Workspace of Electromagnetic Navigation Systems Using Dynamic Feedback for Single- and Multi-agent Control</strong> - Jasan Zughaibi, Denis von Arx, Maurus Derungs, Florian Heemeyer, Luca A. Antonelli, Quentin Boehler, Michael Muehlebach, Bradley J. Nelson - [[pdf]](https://arxiv.org/pdf/2511.18486)</summary>

**Abstract:** Electromagnetic navigation systems (eMNS) enable a number of magnetically guided surgical procedures. A challenge in magnetically manipulating surgical tools is that the effective workspace of an eMNS is often severely constrained by power and thermal limits. We show that system-level control design significantly expands this workspace by reducing the currents needed to achieve a desired motion. We identified five key system approaches that enable this expansion: (i) motion-centric torque/force objectives, (ii) energy-optimal current allocation, (iii) real-time pose estimation, (iv) dynamic feedback, and (v) high-bandwidth eMNS components. As a result, we stabilize a 3D inverted pendulum on an eight-coil OctoMag eMNS with significantly lower currents (0.1-0.2 A vs. 8-14 A), by replacing a field-centric field-alignment strategy with a motion-centric torque/force-based approach. We generalize to multi-agent control by simultaneously stabilizing two inverted pendulums within a shared workspace, exploiting magnetic-field nonlinearity and coil redundancy for independent actuation. A structured analysis compares the electromagnetic workspaces of both paradigms and examines current-allocation strategies that map motion objectives to coil currents. Cross-platform evaluation of the clinically oriented Navion eMNS further demonstrates substantial workspace expansion by maintaining stable balancing at distances up to 50 cm from the coils. The results demonstrate that feedback is a practical path to scalable, efficient, and clinically relevant magnetic manipulation.

**arXiv ID:** 2511.18486
</details>

<details>
<summary><strong>Multi-Agent Monocular Dense SLAM With 3D Reconstruction Priors</strong> - Haihang Wu, Yuchen Zhou - [[pdf]](https://arxiv.org/pdf/2511.19031)</summary>

**Abstract:** Monocular Simultaneous Localization and Mapping (SLAM) aims to estimate a robot's pose while simultaneously reconstructing an unknown 3D scene using a single camera. While existing monocular SLAM systems generate detailed 3D geometry through dense scene representations, they are computationally expensive due to the need for iterative optimization. To address this challenge, MASt3R-SLAM utilizes learned 3D reconstruction priors, enabling more efficient and accurate estimation of both 3D structures and camera poses. However, MASt3R-SLAM is limited to single-agent operation. In this paper, we extend MASt3R-SLAM to introduce the first multi-agent monocular dense SLAM system. Each agent performs local SLAM using a 3D reconstruction prior, and their individual maps are fused into a globally consistent map through a loop-closure-based map fusion mechanism. Our approach improves computational efficiency compared to state-of-the-art methods, while maintaining similar mapping accuracy when evaluated on real-world datasets.

**arXiv ID:** 2511.19031
</details>

<details>
<summary><strong>Connectivity-Preserving Multi-Agent Area Coverage via Optimal-Transport-Based Density-Driven Optimal Control (D2OC)</strong> - Kooktae Lee, Ethan Brook - [[pdf]](https://arxiv.org/pdf/2511.18579)</summary>

**Abstract:** Multi-agent systems play a central role in area coverage tasks across search-and-rescue, environmental monitoring, and precision agriculture. Achieving non-uniform coverage, where spatial priorities vary across the domain, requires coordinating agents while respecting dynamic and communication constraints. Density-driven approaches can distribute agents according to a prescribed reference density, but existing methods do not ensure connectivity. This limitation often leads to communication loss, reduced coordination, and degraded coverage performance.
This letter introduces a connectivity-preserving extension of the Density-Driven Optimal Control (D2OC) framework. The coverage objective, defined using the Wasserstein distance between the agent distribution and the reference density, admits a convex quadratic program formulation. Communication constraints are incorporated through a smooth connectivity penalty, which maintains strict convexity, supports distributed implementation, and preserves inter-agent communication without imposing rigid formations.
Simulation studies show that the proposed method consistently maintains connectivity, improves convergence speed, and enhances non-uniform coverage quality compared with density-driven schemes that do not incorporate explicit connectivity considerations.

**arXiv ID:** 2511.18579
</details>

</details>

<details open>
<summary><h2>Other Agent Research (14 papers)</h2></summary>

<details>
<summary><strong>QuickLAP: Quick Language-Action Preference Learning for Autonomous Driving Agents</strong> - Jordan Abi Nader, David Lee, Nathaniel Dennler, Andreea Bobu - [[pdf]](https://arxiv.org/pdf/2511.17855)</summary>

**Abstract:** Robots must learn from both what people do and what they say, but either modality alone is often incomplete: physical corrections are grounded but ambiguous in intent, while language expresses high-level goals but lacks physical grounding. We introduce QuickLAP: Quick Language-Action Preference learning, a Bayesian framework that fuses physical and language feedback to infer reward functions in real time. Our key insight is to treat language as a probabilistic observation over the user's latent preferences, clarifying which reward features matter and how physical corrections should be interpreted. QuickLAP uses Large Language Models (LLMs) to extract reward feature attention masks and preference shifts from free-form utterances, which it integrates with physical feedback in a closed-form update rule. This enables fast, real-time, and robust reward learning that handles ambiguous feedback. In a semi-autonomous driving simulator, QuickLAP reduces reward learning error by over 70% compared to physical-only and heuristic multimodal baselines. A 15-participant user study further validates our approach: participants found QuickLAP significantly more understandable and collaborative, and preferred its learned behavior over baselines. Code is available at this https URL.

**arXiv ID:** 2511.17855
</details>

<details>
<summary><strong>A novel strategy for multi-resource load balancing in agent-based systems</strong> - Leszek Sliwko, Aleksander Zgrzywa - [[pdf]](https://arxiv.org/pdf/2511.17580)</summary>

**Abstract:** The paper presents a multi-resource load balancing strategy which can be utilised within an agent-based system. This approach can assist system designers in their attempts to optimise the structure for complex enterprise architectures. In this system, the social behaviour of the agent and its adaptation abilities are applied to determine an optimal setup for a given configuration. All the methods have been developed to allow the agent's self-assessment. The proposed agent system has been implemented and the experiment results are presented here.

**arXiv ID:** 2511.17580
</details>

<details>
<summary><strong>MURMUR: Using cross-user chatter to break collaborative language agents in groups</strong> - Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath - [[pdf]](https://arxiv.org/pdf/2511.17671)</summary>

**Abstract:** Language agents are rapidly expanding from single-user assistants to multi-user collaborators in shared workspaces and groups. However, today's language models lack a mechanism for isolating user interactions and concurrent tasks, creating a new attack vector inherent to this new setting: cross-user poisoning (CUP). In a CUP attack, an adversary injects ordinary-looking messages that poison the persistent, shared state, which later triggers the agent to execute unintended, attacker-specified actions on behalf of benign users. We validate CUP on real systems, successfully attacking popular multi-user agents. To study the phenomenon systematically, we present MURMUR, a framework that composes single-user tasks into concurrent, group-based scenarios using an LLM to generate realistic, history-aware user interactions. We observe that CUP attacks succeed at high rates and their effects persist across multiple tasks, thus posing fundamental risks to multi-user LLM deployments. Finally, we introduce a first-step defense with task-based clustering to mitigate this new class of vulnerability

**arXiv ID:** 2511.17671
</details>

<details>
<summary><strong>Episodic Memory in Agentic Frameworks: Suggesting Next Tasks</strong> - Sandro Rama Fiorini, Leonardo G. Azevedo, Raphael M. Thiago, Valesca M. de Sousa, Anton B. Labate, Viviane Torres da Silva - [[pdf]](https://arxiv.org/pdf/2511.17775)</summary>

**Abstract:** Agentic frameworks powered by Large Language Models (LLMs) can be useful tools in scientific workflows by enabling human-AI co-creation. A key challenge is recommending the next steps during workflow creation without relying solely on LLMs, which risk hallucination and require fine-tuning with scarce proprietary data. We propose an episodic memory architecture that stores and retrieves past workflows to guide agents in suggesting plausible next tasks. By matching current workflows with historical sequences, agents can recommend steps based on prior patterns.

**arXiv ID:** 2511.17775
</details>

<details>
<summary><strong>Towards Automating Data Access Permissions in AI Agents</strong> - Yuhao Wu, Ke Yang, Franziska Roesner, Tadayoshi Kohno, Ning Zhang, Umar Iqbal - [[pdf]](https://arxiv.org/pdf/2511.17959)</summary>

**Abstract:** As AI agents attempt to autonomously act on users' behalf, they raise transparency and control issues. We argue that permission-based access control is indispensable in providing meaningful control to the users, but conventional permission models are inadequate for the automated agentic execution paradigm. We therefore propose automated permission management for AI agents. Our key idea is to conduct a user study to identify the factors influencing users' permission decisions and to encode these factors into an ML-based permission management assistant capable of predicting users' future decisions. We find that participants' permission decisions are influenced by communication context but importantly individual preferences tend to remain consistent within contexts, and align with those of other participants. Leveraging these insights, we develop a permission prediction model achieving 85.1% accuracy overall and 94.4% for high-confidence predictions. We find that even without using permission history, our model achieves an accuracy of 66.9%, and a slight increase of training samples (i.e., 1-4) can substantially increase the accuracy by 10.8%.

**arXiv ID:** 2511.17959
</details>

<details>
<summary><strong>Many-Eyes and Sentinels in Selfish and Cooperative Groups</strong> - Charlie Pilgrim, Andrew M Bate, Anna Sigalou, Mélisande Aellen, Joe Morford, Elizabeth Warren, Christopher Krupenye, Dora Biro, Richard P Mann - [[pdf]](https://arxiv.org/pdf/2511.19093)</summary>

**Abstract:** Collective vigilance describes how animals in groups benefit from the predator detection efforts of others. Empirical observations typically find either a many-eyes strategy with all (or many) group members maintaining a low level of individual vigilance, or a sentinel strategy with one (or a few) individuals maintaining a high level of individual vigilance while others do not. With a general analytical treatment that makes minimal assumptions, we show that these two strategies are alternate solutions to the same adaptive problem of balancing the costs of predation and vigilance. Which strategy is preferred depends on how costs scale with the level of individual vigilance: many-eyes strategies are preferred where costs of vigilance rise gently at low levels but become steeper at higher levels (convex; e.g. an open field); sentinel strategies are preferred where costs of vigilance rise steeply at low levels and then flatten out (concave; e.g. environments with vantage points). This same dichotomy emerges whether individuals act selfishly to optimise their own fitness or cooperatively to optimise group fitness. The model is extended to explain discrete behavioural switching between strategies and differential levels of vigilance such as edge effects.

**arXiv ID:** 2511.19093
</details>

<details>
<summary><strong>The Horcrux: Mechanistically Interpretable Task Decomposition for Detecting and Mitigating Reward Hacking in Embodied AI Systems</strong> - Subramanyam Sahoo, Jared Junkin - [[pdf]](https://arxiv.org/pdf/2511.17869)</summary>

**Abstract:** Embodied AI agents exploit reward signal flaws through reward hacking, achieving high proxy scores while failing true objectives. We introduce Mechanistically Interpretable Task Decomposition (MITD), a hierarchical transformer architecture with Planner, Coordinator, and Executor modules that detects and mitigates reward hacking. MITD decomposes tasks into interpretable subtasks while generating diagnostic visualizations including Attention Waterfall Diagrams and Neural Pathway Flow Charts. Experiments on 1,000 HH-RLHF samples reveal that decomposition depths of 12 to 25 steps reduce reward hacking frequency by 34 percent across four failure modes. We present new paradigms showing that mechanistically grounded decomposition offers a more effective way to detect reward hacking than post-hoc behavioral monitoring.

**arXiv ID:** 2511.17869
</details>

<details>
<summary><strong>See, Plan, Cut: MPC-Based Autonomous Volumetric Robotic Laser Surgery with OCT Guidance</strong> - Ravi Prakash, Vincent Y. Wang, Arpit Mishra, Devi Yuliarti, Pei Zhong, Ryan P. McNabb, Patrick J. Codd, Leila J. Bridgeman - [[pdf]](https://arxiv.org/pdf/2511.17777)</summary>

**Abstract:** Robotic laser systems offer the potential for sub-millimeter, non-contact, high-precision tissue resection, yet existing platforms lack volumetric planning and intraoperative feedback. We present RATS (Robot-Assisted Tissue Surgery), an intelligent opto-mechanical, optical coherence tomography (OCT)-guided robotic platform designed for autonomous volumetric soft tissue resection in surgical applications. RATS integrates macro-scale RGB-D imaging, micro-scale OCT, and a fiber-coupled surgical laser, calibrated through a novel multistage alignment pipeline that achieves OCT-to-laser calibration accuracy of 0.161+-0.031mm on tissue phantoms and ex vivo porcine tissue. A super-Gaussian laser-tissue interaction (LTI) model characterizes ablation crater morphology with an average RMSE of 0.231+-0.121mm, outperforming Gaussian baselines. A sampling-based model predictive control (MPC) framework operates directly on OCT voxel data to generate constraint-aware resection trajectories with closed-loop feedback, achieving 0.842mm RMSE and improving intersection-over-union agreement by 64.8% compared to feedforward execution. With OCT, RATS detects subsurface structures and modifies the planner's objective to preserve them, demonstrating clinical feasibility.

**arXiv ID:** 2511.17777
</details>

<details>
<summary><strong>MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent</strong> - Yuxia Fu, Zhizhen Zhang, Yuqi Zhang, Zijian Wang, Zi Huang, Yadan Luo - [[pdf]](https://arxiv.org/pdf/2511.18810)</summary>

**Abstract:** Recent Vision-Language-Action (VLA) models reformulate vision-language models by tuning them with millions of robotic demonstrations. While they perform well when fine-tuned for a single embodiment or task family, extending them to multi-skill settings remains challenging: directly merging VLA experts trained on different tasks results in near-zero success rates. This raises a fundamental question: what prevents VLAs from mastering multiple skills within one model? With an empirical decomposition of learnable parameters during VLA fine-tuning, we identify two key sources of non-mergeability: (1) Finetuning drives LoRA adapters in the VLM backbone toward divergent, task-specific directions beyond the capacity of existing merging methods to unify. (2) Action experts develop inter-block dependencies through self-attention feedback, causing task information to spread across layers and preventing modular recombination. To address these challenges, we present MergeVLA, a merging-oriented VLA architecture that preserves mergeability by design. MergeVLA introduces sparsely activated LoRA adapters via task masks to retain consistent parameters and reduce irreconcilable conflicts in the VLM. Its action expert replaces self-attention with cross-attention-only blocks to keep specialization localized and composable. When the task is unknown, it uses a test-time task router to adaptively select the appropriate task mask and expert head from the initial observation, enabling unsupervised task inference. Across LIBERO, LIBERO-Plus, RoboTwin, and multi-task experiments on the real SO101 robotic arm, MergeVLA achieves performance comparable to or even exceeding individually finetuned experts, demonstrating robust generalization across tasks, embodiments, and environments.

**arXiv ID:** 2511.18810
</details>

<details>
<summary><strong>End-to-end Autonomous Vehicle Following System using Monocular Fisheye Camera</strong> - Jiale Zhang, Yeqiang Qian, Tong Qin, Mingyang Jiang, Siyuan Chen, Ming Yang - [[pdf]](https://arxiv.org/pdf/2511.19011)</summary>

**Abstract:** The increase in vehicle ownership has led to increased traffic congestion, more accidents, and higher carbon emissions. Vehicle platooning is a promising solution to address these issues by improving road capacity and reducing fuel consumption. However, existing platooning systems face challenges such as reliance on lane markings and expensive high-precision sensors, which limits their general applicability. To address these issues, we propose a vehicle following framework that expands its capability from restricted scenarios to general scenario applications using only a camera. This is achieved through our newly proposed end-to-end method, which improves overall driving performance. The method incorporates a semantic mask to address causal confusion in multi-frame data fusion. Additionally, we introduce a dynamic sampling mechanism to precisely track the trajectories of preceding vehicles. Extensive closed-loop validation in real-world vehicle experiments demonstrates the system's ability to follow vehicles in various scenarios, outperforming traditional multi-stage algorithms. This makes it a promising solution for cost-effective autonomous vehicle platooning. A complete real-world vehicle experiment is available at this https URL.

**arXiv ID:** 2511.19011
</details>

<details>
<summary><strong>Autonomous Docking of Multi-Rotor UAVs on Blimps under the Influence of Wind Gusts</strong> - Pascal Goldschmid, Aamir Ahmad - [[pdf]](https://arxiv.org/pdf/2511.19135)</summary>

**Abstract:** Multi-rotor UAVs face limited flight time due to battery constraints. Autonomous docking on blimps with onboard battery recharging and data offloading offers a promising solution for extended UAV missions. However, the vulnerability of blimps to wind gusts causes trajectory deviations, requiring precise, obstacle-aware docking strategies. To this end, this work introduces two key novelties: (i) a temporal convolutional network that predicts blimp responses to wind gusts, enabling rapid gust detection and estimation of points where the wind gust effect has subsided; (ii) a model predictive controller (MPC) that leverages these predictions to compute collision-free trajectories for docking, enabled by a novel obstacle avoidance method for close-range manoeuvres near the blimp. Simulation results show our method outperforms a baseline constant-velocity model of the blimp significantly across different scenarios. We further validate the approach in real-world experiments, demonstrating the first autonomous multi-rotor docking control strategy on blimps shown outside simulation. Source code is available here this https URL.

**arXiv ID:** 2511.19135
</details>

<details>
<summary><strong>Anomaly Detection in Autonomous Driving: A Survey</strong> - Daniel Bogdoll, Maximilian Nitsche, J. Marius Zöllner - [[pdf]](https://arxiv.org/pdf/2204.07974)</summary>

**Abstract:** Nowadays, there are outstanding strides towards a future with autonomous vehicles on our roads. While the perception of autonomous vehicles performs well under closed-set conditions, they still struggle to handle the unexpected. This survey provides an extensive overview of anomaly detection techniques based on camera, lidar, radar, multimodal and abstract object level data. We provide a systematization including detection approach, corner case level, ability for an online application, and further attributes. We outline the state-of-the-art and point out current research gaps.

**arXiv ID:** 2204.07974
</details>

<details>
<summary><strong>Occlusion-Aware Contingency Safety-Critical Planning for Autonomous Driving</strong> - Lei Zheng, Rui Yang, Minzhe Zheng, Zengqi Peng, Michael Yu Wang, Jun Ma - [[pdf]](https://arxiv.org/pdf/2502.06359)</summary>

**Abstract:** Ensuring safe driving while maintaining travel efficiency for autonomous vehicles in dynamic and occluded environments is a critical challenge. This paper proposes an occlusion-aware contingency safety-critical planning approach for real-time autonomous driving. Leveraging reachability analysis for risk assessment, forward reachable sets of phantom vehicles are used to derive risk-aware dynamic velocity boundaries. These velocity boundaries are incorporated into a biconvex nonlinear programming (NLP) formulation that formally enforces safety using spatiotemporal barrier constraints, while simultaneously optimizing exploration and fallback trajectories within a receding horizon planning framework. To enable real-time computation and coordination between trajectories, we employ the consensus alternating direction method of multipliers (ADMM) to decompose the biconvex NLP problem into low-dimensional convex subproblems. The effectiveness of the proposed approach is validated through simulations and real-world experiments in occluded intersections. Experimental results demonstrate enhanced safety and improved travel efficiency, enabling real-time safe trajectory generation in dynamic occluded intersections under varying obstacle conditions. The project page is available at this https URL.

**arXiv ID:** 2502.06359
</details>

<details>
<summary><strong>Autonomous Vehicle Path Planning by Searching With Differentiable Simulation</strong> - Asen Nachkov, Jan-Nico Zaech, Danda Pani Paudel, Xi Wang, Luc Van Gool - [[pdf]](https://arxiv.org/pdf/2511.11043)</summary>

**Abstract:** Planning allows an agent to safely refine its actions before executing them in the real world. In autonomous driving, this is crucial to avoid collisions and navigate in complex, dense traffic scenarios. One way to plan is to search for the best action sequence. However, this is challenging when all necessary components - policy, next-state predictor, and critic - have to be learned. Here we propose Differentiable Simulation for Search (DSS), a framework that leverages the differentiable simulator Waymax as both a next state predictor and a critic. It relies on the simulator's hardcoded dynamics, making state predictions highly accurate, while utilizing the simulator's differentiability to effectively search across action sequences. Our DSS agent optimizes its actions using gradient descent over imagined future trajectories. We show experimentally that DSS - the combination of planning gradients and stochastic search - significantly improves tracking and path planning accuracy compared to sequence prediction, imitation learning, model-free RL, and other planning methods.

**arXiv ID:** 2511.11043
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (1 papers)</h2></summary>

<details>
<summary><strong>Consensus Planning with Primal, Dual, and Proximal Agents</strong> - Alvaro Maggiar, Lee Dicker, Michael Mahoney - [[pdf]](https://arxiv.org/pdf/2408.16462)</summary>

**Abstract:** Consensus planning is a method for coordinating decision making across complex systems and organizations, including complex supply chain optimization pipelines. It arises when large interdependent distributed agents (systems) share common resources and must act in order to achieve a joint goal. In prior consensus planning work, all agents have been assumed to have the same interaction pattern (e.g., all dual agents or all primal agents or all proximal agents), most commonly using the Alternating Direction Method of Multipliers (ADMM) as proximal agents. However, this is often not a valid assumption in practice, where agents consist of large complex systems, and where we might not have the luxury of modifying these large complex systems at will. In this paper, we introduce a consensus algorithm that overcomes this hurdle by allowing for the coordination of agents with different types of interfaces (named primal, dual, and proximal). Our consensus planning algorithm allows for any mix of agents by combining ADMM-like updates for the proximal agents, dual ascent updates for the dual agents, and linearized ADMM updates for the primal agents. We prove convergence results for the algorithm, namely a sublinear O(1/k) convergence rate under mild assumptions, and two-step linear convergence under stronger assumptions. We also discuss enhancements to the basic method and provide illustrative empirical results.

**arXiv ID:** 2408.16462
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (33 papers)</h2></summary>

<details>
<summary><strong>Training Emergent Joint Associations: A Reinforcement Learning Approach to Creative Thinking in Language Models</strong> - Mukul Singh, Ananya Singha, Aishni Parab, Pronita Mehrotra, Sumit Gulwani - [[pdf]](https://arxiv.org/pdf/2511.17876)</summary>

**Abstract:** Associative thinking--the ability to connect seemingly unrelated ideas--is a foundational element of human creativity and problem-solving. This paper explores whether reinforcement learning (RL) guided by associative thinking principles can enhance a model's performance across diverse generative tasks, including story writing, code generation, and chart creation. We introduce a reinforcement learning framework that uses a prompt-based evaluation mechanism, incorporating established divergent thinking metrics from creativity research. A base language model is fine-tuned using this framework to reward outputs demonstrating higher novelty through higher degrees of conceptual connectivity. Interestingly, the experimental results suggest that RL-based associative thinking-trained models not only generate more original and coherent stories but also exhibit improved abstraction and flexibility in tasks such as programming and data visualization. Our findings provide initial evidence that modeling cognitive creativity principles through reinforcement learning can yield more adaptive and generative AI.

**arXiv ID:** 2511.17876
</details>

<details>
<summary><strong>MoodBench 1.0: An Evaluation Benchmark for Emotional Companionship Dialogue Systems</strong> - Haifeng Jing, Yujie Hou, Junfei Liu, Rui Xie, alan Xu, Jinlong Ma, Qichun Deng - [[pdf]](https://arxiv.org/pdf/2511.18926)</summary>

**Abstract:** With the rapid development of Large Language Models, dialogue systems are shifting from information tools to emotional companions, heralding the era of Emotional Companionship Dialogue Systems (ECDs) that provide personalized emotional support for users. However, the field lacks clear definitions and systematic evaluation standards for ECDs. To address this, we first propose a definition of ECDs with formal descriptions. Then, based on this theory and the design principle of "Ability Layer-Task Layer (three level)-Data Layer-Method Layer", we design and implement the first ECD evaluation benchmark - MoodBench 1.0. Through extensive evaluations of 30 mainstream models, we demonstrate that MoodBench 1.0 has excellent discriminant validity and can effectively quantify the differences in emotional companionship abilities among models. Furthermore, the results reveal current models' shortcomings in deep emotional companionship, guiding future technological optimization and significantly aiding developers in enhancing ECDs' user experience.

**arXiv ID:** 2511.18926
</details>

<details>
<summary><strong>Enhancing Robustness of Offline Reinforcement Learning Under Data Corruption via Sharpness-Aware Minimization</strong> - Le Xu, Jiayu Chen - [[pdf]](https://arxiv.org/pdf/2511.17568)</summary>

**Abstract:** Offline reinforcement learning (RL) is vulnerable to real-world data corruption, with even robust algorithms failing under challenging observation and mixture corruptions. We posit this failure stems from data corruption creating sharp minima in the loss landscape, leading to poor generalization. To address this, we are the first to apply Sharpness-Aware Minimization (SAM) as a general-purpose, plug-and-play optimizer for offline RL. SAM seeks flatter minima, guiding models to more robust parameter regions. We integrate SAM into strong baselines for data corruption: IQL, a top-performing offline RL algorithm in this setting, and RIQL, an algorithm designed specifically for data-corruption robustness. We evaluate them on D4RL benchmarks with both random and adversarial corruption. Our SAM-enhanced methods consistently and significantly outperform the original baselines. Visualizations of the reward surface confirm that SAM finds smoother solutions, providing strong evidence for its effectiveness in improving the robustness of offline RL agents.

**arXiv ID:** 2511.17568
</details>

<details>
<summary><strong>Boosting Reinforcement Learning in 3D Visuospatial Tasks Through Human-Informed Curriculum Design</strong> - Markus D. Solbach, John K. Tsotsos - [[pdf]](https://arxiv.org/pdf/2511.17595)</summary>

**Abstract:** Reinforcement Learning is a mature technology, often suggested as a potential route towards Artificial General Intelligence, with the ambitious goal of replicating the wide range of abilities found in natural and artificial intelligence, including the complexities of human cognition. While RL had shown successes in relatively constrained environments, such as the classic Atari games and specific continuous control problems, recent years have seen efforts to expand its applicability. This work investigates the potential of RL in demonstrating intelligent behaviour and its progress in addressing more complex and less structured problem domains.
We present an investigation into the capacity of modern RL frameworks in addressing a seemingly straightforward 3D Same-Different visuospatial task. While initial applications of state-of-the-art methods, including PPO, behavioural cloning and imitation learning, revealed challenges in directly learning optimal strategies, the successful implementation of curriculum learning offers a promising avenue. Effective learning was achieved by strategically designing the lesson plan based on the findings of a real-world human experiment.

**arXiv ID:** 2511.17595
</details>

<details>
<summary><strong>Can we use LLMs to bootstrap reinforcement learning? -- A case study in digital health behavior change</strong> - Nele Albers, Esra Cemre Su de Groot, Loes Keijsers, Manon H. Hillegers, Emiel Krahmer - [[pdf]](https://arxiv.org/pdf/2511.17630)</summary>

**Abstract:** Personalizing digital applications for health behavior change is a promising route to making them more engaging and effective. This especially holds for approaches that adapt to users and their specific states (e.g., motivation, knowledge, wants) over time. However, developing such approaches requires making many design choices, whose effectiveness is difficult to predict from literature and costly to evaluate in practice. In this work, we explore whether large language models (LLMs) can be used out-of-the-box to generate samples of user interactions that provide useful information for training reinforcement learning models for digital behavior change settings. Using real user data from four large behavior change studies as comparison, we show that LLM-generated samples can be useful in the absence of real data. Comparisons to the samples provided by human raters further show that LLM-generated samples reach the performance of human raters. Additional analyses of different prompting strategies including shorter and longer prompt variants, chain-of-thought prompting, and few-shot prompting show that the relative effectiveness of different strategies depends on both the study and the LLM with also relatively large differences between prompt paraphrases alone. We provide recommendations for how LLM-generated samples can be useful in practice.

**arXiv ID:** 2511.17630
</details>

<details>
<summary><strong>PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning</strong> - Yingjie Ma, Xun Lin, Yong Xu, Weicheng Xie, Zitong Yu - [[pdf]](https://arxiv.org/pdf/2511.17927)</summary>

**Abstract:** Face anti-spoofing (FAS) has recently advanced in multimodal fusion, cross-domain generalization, and interpretability. With large language models and reinforcement learning (RL), strategy-based training offers new opportunities to jointly model these aspects. However, multimodal reasoning is more complex than unimodal reasoning, requiring accurate feature representation and cross-modal verification while facing scarce, high-quality annotations, which makes direct application of RL sub-optimal. We identify two key limitations of supervised fine-tuning plus RL (SFT+RL) for multimodal FAS: (1) limited multimodal reasoning paths restrict the use of complementary modalities and shrink the exploration space after SFT, weakening the effect of RL; and (2) mismatched single-task supervision versus diverse reasoning paths causes reasoning confusion, where models may exploit shortcuts by mapping images directly to answers and ignoring the intended reasoning. To address this, we propose PA-FAS, which enhances reasoning paths by constructing high-quality extended reasoning sequences from limited annotations, enriching paths and relaxing exploration constraints. We further introduce an answer-shuffling mechanism during SFT to force comprehensive multimodal analysis instead of using superficial cues, thereby encouraging deeper reasoning and mitigating shortcut learning. PA-FAS significantly improves multimodal reasoning accuracy and cross-domain generalization, and better unifies multimodal fusion, generalization, and interpretability for trustworthy FAS.

**arXiv ID:** 2511.17927
</details>

<details>
<summary><strong>Reward Engineering for Spatial Epidemic Simulations: A Reinforcement Learning Platform for Individual Behavioral Learning</strong> - Radman Rakhshandehroo, Daniel Coombs - [[pdf]](https://arxiv.org/pdf/2511.18000)</summary>

**Abstract:** We present ContagionRL, a Gymnasium-compatible reinforcement learning platform specifically designed for systematic reward engineering in spatial epidemic simulations. Unlike traditional agent-based models that rely on fixed behavioral rules, our platform enables rigorous evaluation of how reward function design affects learned survival strategies across diverse epidemic scenarios. ContagionRL integrates a spatial SIRS+D epidemiological model with configurable environmental parameters, allowing researchers to stress-test reward functions under varying conditions including limited observability, different movement patterns, and heterogeneous population dynamics. We evaluate five distinct reward designs, ranging from sparse survival bonuses to a novel potential field approach, across multiple RL algorithms (PPO, SAC, A2C). Through systematic ablation studies, we identify that directional guidance and explicit adherence incentives are critical components for robust policy learning. Our comprehensive evaluation across varying infection rates, grid sizes, visibility constraints, and movement patterns reveals that reward function choice dramatically impacts agent behavior and survival outcomes. Agents trained with our potential field reward consistently achieve superior performance, learning maximal adherence to non-pharmaceutical interventions while developing sophisticated spatial avoidance strategies. The platform's modular design enables systematic exploration of reward-behavior relationships, addressing a knowledge gap in models of this type where reward engineering has received limited attention. ContagionRL is an effective platform for studying adaptive behavioral responses in epidemic contexts and highlight the importance of reward design, information structure, and environmental predictability in learning.

**arXiv ID:** 2511.18000
</details>

<details>
<summary><strong>Reinforcement Learning for Portfolio Optimization with a Financial Goal and Defined Time Horizons</strong> - Fermat Leukam, Rock Stephane Koffi, Prudence Djagba - [[pdf]](https://arxiv.org/pdf/2511.18076)</summary>

**Abstract:** This research proposes an enhancement to the innovative portfolio optimization approach using the G-Learning algorithm, combined with parametric optimization via the GIRL algorithm (G-learning approach to the setting of Inverse Reinforcement Learning) as presented by. The goal is to maximize portfolio value by a target date while minimizing the investor's periodic contributions. Our model operates in a highly volatile market with a well-diversified portfolio, ensuring a low-risk level for the investor, and leverages reinforcement learning to dynamically adjust portfolio positions over time. Results show that we improved the Sharpe Ratio from 0.42, as suggested by recent studies using the same approach, to a value of 0.483 a notable achievement in highly volatile markets with diversified portfolios. The comparison between G-Learning and GIRL reveals that while GIRL optimizes the reward function parameters (e.g., lambda = 0.0012 compared to 0.002), its impact on portfolio performance remains marginal. This suggests that reinforcement learning methods, like G-Learning, already enable robust optimization. This research contributes to the growing development of reinforcement learning applications in financial decision-making, demonstrating that probabilistic learning algorithms can effectively align portfolio management strategies with investor needs.

**arXiv ID:** 2511.18076
</details>

<details>
<summary><strong>A New Error Temporal Difference Algorithm for Deep Reinforcement Learning in Microgrid Optimization</strong> - Fulong Yao, Wanqing Zhao, Matthew Forshaw - [[pdf]](https://arxiv.org/pdf/2511.18093)</summary>

**Abstract:** Predictive control approaches based on deep reinforcement learning (DRL) have gained significant attention in microgrid energy optimization. However, existing research often overlooks the issue of uncertainty stemming from imperfect prediction models, which can lead to suboptimal control strategies. This paper presents a new error temporal difference (ETD) algorithm for DRL to address the uncertainty in predictions,aiming to improve the performance of microgrid operations. First,a microgrid system integrated with renewable energy sources (RES) and energy storage systems (ESS), along with its Markov decision process (MDP), is modelled. Second, a predictive control approach based on a deep Q network (DQN) is presented, in which a weighted average algorithm and a new ETD algorithm are designed to quantify and address the prediction uncertainty, respectively. Finally, simulations on a realworld US dataset suggest that the developed ETD effectively improves the performance of DRL in optimizing microgrid operations.

**arXiv ID:** 2511.18093
</details>

<details>
<summary><strong>ARIAL: An Agentic Framework for Document VQA with Precise Answer Localization</strong> - Ahmad Mohammadshirazi, Pinaki Prasad Guha Neogi, Dheeraj Kulshrestha, Rajiv Ramnath - [[pdf]](https://arxiv.org/pdf/2511.18192)</summary>

**Abstract:** Document Visual Question Answering (VQA) requires models to not only extract accurate textual answers but also precisely localize them within document images, a capability critical for interpretability in high-stakes applications. However, existing systems achieve strong textual accuracy while producing unreliable spatial grounding, or sacrifice performance for interpretability. We present ARIAL (Agentic Reasoning for Interpretable Answer Localization), a modular framework that orchestrates specialized tools through an LLM-based planning agent to achieve both precise answer extraction and reliable spatial grounding. ARIAL decomposes Document VQA into structured subtasks: OCR-based text extraction with TrOCR, retrieval-augmented context selection using semantic search, answer generation via a fine-tuned Gemma 3-27B model, and explicit bounding-box localization through text-to-region alignment. This modular architecture produces transparent reasoning traces, enabling tool-level auditability and independent component optimization. We evaluate ARIAL on four benchmarks (DocVQA, FUNSD, CORD, and SROIE) using both textual accuracy (ANLS) and spatial precision (mAP at IoU 0.50 to 0.95). ARIAL achieves state-of-the-art results across all datasets: 88.7 ANLS and 50.1 mAP on DocVQA, 90.0 ANLS and 50.3 mAP on FUNSD, 85.5 ANLS and 60.2 mAP on CORD, and 93.1 ANLS on SROIE, surpassing the previous best method (DLaVA) by +2.8 ANLS and +3.9 mAP on DocVQA. Our work demonstrates how agentic orchestration of specialized tools can simultaneously improve performance and interpretability, providing a pathway toward trustworthy, explainable document AI systems.

**arXiv ID:** 2511.18192
</details>

<details>
<summary><strong>A Novel and Practical Universal Adversarial Perturbations against Deep Reinforcement Learning based Intrusion Detection Systems</strong> - H. Zhang, L. Zhang, G. Epiphaniou, C. Maple - [[pdf]](https://arxiv.org/pdf/2511.18223)</summary>

**Abstract:** Intrusion Detection Systems (IDS) play a vital role in defending modern cyber physical systems against increasingly sophisticated cyber threats. Deep Reinforcement Learning-based IDS, have shown promise due to their adaptive and generalization capabilities. However, recent studies reveal their vulnerability to adversarial attacks, including Universal Adversarial Perturbations (UAPs), which can deceive models with a single, input-agnostic perturbation. In this work, we propose a novel UAP attack against Deep Reinforcement Learning (DRL)-based IDS under the domain-specific constraints derived from network data rules and feature relationships. To the best of our knowledge, there is no existing study that has explored UAP generation for the DRL-based IDS. In addition, this is the first work that focuses on developing a UAP against a DRL-based IDS under realistic domain constraints based on not only the basic domain rules but also mathematical relations between the features. Furthermore, we enhance the evasion performance of the proposed UAP, by introducing a customized loss function based on the Pearson Correlation Coefficient, and we denote it as Customized UAP. To the best of our knowledge, this is also the first work using the PCC value in the UAP generation, even in the broader context. Four additional established UAP baselines are implemented for a comprehensive comparison. Experimental results demonstrate that our proposed Customized UAP outperforms two input-dependent attacks including Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), and four UAP baselines, highlighting its effectiveness for real-world adversarial scenarios.

**arXiv ID:** 2511.18223
</details>

<details>
<summary><strong>General Agentic Memory Via Deep Research</strong> - B.Y. Yan, Chaofan Li, Hongjin Qian, Shuqi Lu, Zheng Liu - [[pdf]](https://arxiv.org/pdf/2511.18423)</summary>

**Abstract:** Memory is critical for AI agents, yet the widely-adopted static memory, aiming to create readily available memory in advance, is inevitably subject to severe information loss. To address this limitation, we propose a novel framework called \textbf{general agentic memory (GAM)}. GAM follows the principle of "\textbf{just-in time (JIT) compilation}" where it focuses on creating optimized contexts for its client at runtime while keeping only simple but useful memory during the offline stage. To this end, GAM employs a duo-design with the following components. 1) \textbf{Memorizer}, which highlights key historical information using a lightweight memory, while maintaining complete historical information within a universal page-store. 2) \textbf{Researcher}, which retrieves and integrates useful information from the page-store for its online request guided by the pre-constructed memory. This design allows GAM to effectively leverage the agentic capabilities and test-time scalability of frontier large language models (LLMs), while also facilitating end-to-end performance optimization through reinforcement learning. In our experimental study, we demonstrate that GAM achieves substantial improvement on various memory-grounded task completion scenarios against existing memory systems.

**arXiv ID:** 2511.18423
</details>

<details>
<summary><strong>SPINE: Token-Selective Test-Time Reinforcement Learning with Entropy-Band Regularization</strong> - Jianghao Wu, Yasmeen George, Jin Ye, Yicheng Wu, Daniel F. Schmidt, Jianfei Cai - [[pdf]](https://arxiv.org/pdf/2511.17938)</summary>

**Abstract:** Large language models (LLMs) and multimodal LLMs (MLLMs) excel at chain-of-thought reasoning but face distribution shift at test-time and a lack of verifiable supervision. Recent test-time reinforcement learning (TTRL) methods derive label-free pseudo-rewards from self-consistency voting over sampled trajectories, yet they often collapse: the majority-vote reward prevails, responses shorten, and Pass@1 declines. We trace this to uniform sequence updates in which most tokens are low-entropy followers, while a small high-entropy subset determines the reasoning branches. Thus we propose SPINE, a token-selective test-time reinforcement learning framework that (i) updates only forking tokens, the high-entropy branch points identified from forward-pass statistics, and (ii) applies an entropy-band regularizer at those tokens to sustain exploration when entropy is too low and to suppress noisy supervision when it is too high. SPINE plugs into GRPO-style objectives, optionally with a KL anchor, and requires no labels or reward models. Across ten benchmarks spanning multimodal VQA, general and expert QA, mathematical reasoning, and medical QA, SPINE consistently improves Pass@1 over TTRL while avoiding response-length collapse and yielding more stable training dynamics on both LLM and MLLM backbones. These results indicate that aligning updates with chain-of-thought branch points is a simple and label-free mechanism for stable and effective test-time adaptation in reasoning models. Code is available at this https URL.

**arXiv ID:** 2511.17938
</details>

<details>
<summary><strong>DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research</strong> - Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G. Finlayson, David Sontag, Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soldaini, Faeze Brahman, Wen-tau Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, Hannaneh Hajishirzi, Pang Wei Koh - [[pdf]](https://arxiv.org/pdf/2511.19399)</summary>

**Abstract:** Deep research models perform multi-step research to produce long-form, well-attributed answers. However, most open deep research models are trained on easily verifiable short-form QA tasks via reinforcement learning with verifiable rewards (RLVR), which does not extend to realistic long-form tasks. We address this with Reinforcement Learning with Evolving Rubrics (RLER), in which we construct and maintain rubrics that co-evolve with the policy model during training; this allows the rubrics to incorporate information that the model has newly explored and to provide discriminative, on-policy feedback. Using RLER, we develop Deep Research Tulu (DR Tulu-8B), the first open model that is directly trained for open-ended, long-form deep research. Across four long-form deep research benchmarks in science, healthcare and general domains, DR Tulu substantially outperforms existing open deep research models, and matches or exceeds proprietary deep research systems, while being significantly smaller and cheaper per query. To facilitate future research, we release all data, models, and code, including our new MCP-based agent infrastructure for deep research systems.

**arXiv ID:** 2511.19399
</details>

<details>
<summary><strong>From Code Foundation Models to Agents and Applications: A Practical Guide to Code Intelligence</strong> - Jian Yang, Wei Zhang, Shark Liu, Jiajun Wu, Shawn Guo, Yizhi Li - [[pdf]](https://arxiv.org/pdf/2511.18538)</summary>

**Abstract:** Large language models (LLMs) have fundamentally transformed automated software development by enabling direct translation of natural language descriptions into functional code, driving commercial adoption through tools like Github Copilot (Microsoft), Cursor (Anysphere), Trae (ByteDance), and Claude Code (Anthropic). While the field has evolved dramatically from rule-based systems to Transformer-based architectures, achieving performance improvements from single-digit to over 95\% success rates on benchmarks like HumanEval. In this work, we provide a comprehensive synthesis and practical guide (a series of analytic and probing experiments) about code LLMs, systematically examining the complete model life cycle from data curation to post-training through advanced prompting paradigms, code pre-training, supervised fine-tuning, reinforcement learning, and autonomous coding agents. We analyze the code capability of the general LLMs (GPT-4, Claude, LLaMA) and code-specialized LLMs (StarCoder, Code LLaMA, DeepSeek-Coder, and QwenCoder), critically examining the techniques, design decisions, and trade-offs. Further, we articulate the research-practice gap between academic research (e.g., benchmarks and tasks) and real-world deployment (e.g., software-related code tasks), including code correctness, security, contextual awareness of large codebases, and integration with development workflows, and map promising research directions to practical needs. Last, we conduct a series of experiments to provide a comprehensive analysis of code pre-training, supervised fine-tuning, and reinforcement learning, covering scaling law, framework selection, hyperparameter sensitivity, model architectures, and dataset comparisons.

**arXiv ID:** 2511.18538
</details>

<details>
<summary><strong>ReCode: Updating Code API Knowledge with Reinforcement Learning</strong> - Haoze Wu, Yunzhi Yao, Wenhao Yu, Ningyu Zhang - [[pdf]](https://arxiv.org/pdf/2506.20495)</summary>

**Abstract:** Large Language Models (LLMs) exhibit remarkable code generation capabilities but falter when adapting to frequent updates in external library APIs. This critical limitation, stemming from reliance on outdated API knowledge from their training data, even with access to current documentation, impedes reliable code generation in dynamic environments. To tackle this issue, we propose ReCode (rule-based Reinforcement learning for Code Update), a novel framework that mimics human programmer adaptation to API changes. Specifically, we construct a dataset of approximately 2,000 data entries to train the LLMs to perform version migration based on updated information. Then, we introduce a modified string similarity metric for code evaluation as the reward for reinforcement learning. Our experiments demonstrate that ReCode substantially boosts LLMs' code generation performance in dynamic API scenarios, especially on the unseen CodeUpdateArena task. Crucially, compared to supervised fine-tuning, ReCode has less impact on LLMs' general code generation abilities. We apply ReCode on various LLMs and reinforcement learning algorithms (GRPO and DAPO), all achieving consistent improvements. Notably, after training, Qwen2.5-Coder-7B outperforms that of the 32B parameter code instruction-tuned model and the reasoning model with the same architecture. Code is available at this https URL.

**arXiv ID:** 2506.20495
</details>

<details>
<summary><strong>Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?</strong> - Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang - [[pdf]](https://arxiv.org/pdf/2504.13837)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.

**arXiv ID:** 2504.13837
</details>

<details>
<summary><strong>Nested-ReFT: Efficient Reinforcement Learning for Large Language Model Fine-Tuning via Off-Policy Rollouts</strong> - Maxime Heuillet, Yufei Cui, Boxing Chen, Audrey Durand, Prasanna Parthasarathi - [[pdf]](https://arxiv.org/pdf/2508.10123)</summary>

**Abstract:** Advanced reasoning in LLMs on challenging domains like mathematical reasoning can be tackled using verifiable rewards based reinforced fine-tuning (ReFT). In standard ReFT frameworks, a behavior model generates multiple completions with answers per problem, for the answer to be then scored by a reward function. While such RL post-training methods demonstrate significant performance improvements across challenging reasoning domains, the computational cost of generating completions during training with multiple inference steps makes the training cost non-trivial. To address this, we draw inspiration from off-policy RL, and speculative decoding to introduce a novel ReFT framework, dubbed Nested-ReFT, where a subset of layers of the target model acts as the behavior model to generate off-policy completions during training. The behavior model configured with dynamic layer skipping per batch during training decreases the inference cost compared to the standard ReFT frameworks. Our theoretical analysis shows that Nested-ReFT yields unbiased gradient estimates with controlled variance. Our empirical analysis demonstrates improved computational efficiency measured as tokens/sec across multiple math reasoning benchmarks and model sizes. Additionally, we explore three variants of bias mitigation to minimize the off-policyness in the gradient updates that allows for maintaining performance that matches the baseline ReFT performance.

**arXiv ID:** 2508.10123
</details>

<details>
<summary><strong>Non-stationary and Varying-discounting Markov Decision Processes for Reinforcement Learning</strong> - Zhizuo Chen, Theodore T. Allen - [[pdf]](https://arxiv.org/pdf/2511.17598)</summary>

**Abstract:** Algorithms developed under stationary Markov Decision Processes (MDPs) often face challenges in non-stationary environments, and infinite-horizon formulations may not directly apply to finite-horizon tasks. To address these limitations, we introduce the Non-stationary and Varying-discounting MDP (NVMDP) framework, which naturally accommodates non-stationarity and allows discount rates to vary with time and transitions. Infinite-horizon, stationary MDPs emerge as special cases of NVMDPs for identifying an optimal policy, and finite-horizon MDPs are also subsumed within the NVMDP formulations. Moreover, NVMDPs provide a flexible mechanism to shape optimal policies, without altering the state space, action space, or the reward structure. We establish the theoretical foundations of NVMDPs, including assumptions, state- and action-value formulation and recursion, matrix representation, optimality conditions, and policy improvement under finite state and action spaces. Building on these results, we adapt dynamic programming and generalized Q-learning algorithms to NVMDPs, along with formal convergence proofs. For problems requiring function approximation, we extend the Policy Gradient Theorem and the policy improvement bound in Trust Region Policy Optimization (TRPO), offering proofs in both scalar and matrix forms. Empirical evaluations in a non-stationary gridworld environment demonstrate that NVMDP-based algorithms successfully recover optimal trajectories under multiple reward and discounting schemes, whereas original Q-learning fails. These results collectively show that NVMDPs provide a theoretically sound and practically effective framework for reinforcement learning, requiring only minor algorithmic modifications while enabling robust handling of non-stationarity and explicit optimal policy shaping.

**arXiv ID:** 2511.17598
</details>

<details>
<summary><strong>Tail Distribution of Regret in Optimistic Reinforcement Learning</strong> - Sajad Khodadadian, Mehrdad Moharrami - [[pdf]](https://arxiv.org/pdf/2511.18247)</summary>

**Abstract:** We derive instance-dependent tail bounds for the regret of optimism-based reinforcement learning in finite-horizon tabular Markov decision processes with unknown transition dynamics. Focusing on a UCBVI-type algorithm, we characterize the tail distribution of the cumulative regret $R_K$ over $K$ episodes, rather than only its expectation or a single high-probability quantile. We analyze two natural exploration-bonus schedules: (i) a $K$-dependent scheme that explicitly incorporates the total number of episodes $K$, and (ii) a $K$-independent scheme that depends only on the current episode index. For both settings, we obtain an upper bound on $\Pr(R_K \ge x)$ that exhibits a distinctive two-regime structure: a sub-Gaussian tail starting from an instance-dependent scale $m_K$ up to a transition threshold, followed by a sub-Weibull tail beyond that point. We further derive corresponding instance-dependent bounds on the expected regret $\mathbb{E}[R_K]$. The proposed algorithm depends on a tuning parameter $\alpha$, which balances the expected regret and the range over which the regret exhibits a sub-Gaussian tail. To the best of our knowledge, our results provide one of the first comprehensive tail-regret guarantees for a standard optimistic algorithm in episodic reinforcement learning.

**arXiv ID:** 2511.18247
</details>

<details>
<summary><strong>Reinforcement Learning for Self-Healing Material Systems</strong> - Maitreyi Chatterjee, Devansh Agarwal, Biplab Chatterjee - [[pdf]](https://arxiv.org/pdf/2511.18728)</summary>

**Abstract:** The transition to autonomous material systems necessitates adaptive control methodologies to maximize structural longevity. This study frames the self-healing process as a Reinforcement Learning (RL) problem within a Markov Decision Process (MDP), enabling agents to autonomously derive optimal policies that efficiently balance structural integrity maintenance against finite resource consumption. A comparative evaluation of discrete-action (Q-learning, DQN) and continuous-action (TD3) agents in a stochastic simulation environment revealed that RL controllers significantly outperform heuristic baselines, achieving near-complete material recovery. Crucially, the TD3 agent utilizing continuous dosage control demonstrated superior convergence speed and stability, underscoring the necessity of fine-grained, proportional actuation in dynamic self-healing applications.

**arXiv ID:** 2511.18728
</details>

<details>
<summary><strong>Periodic Asynchrony: An Effective Method for Accelerating On-Policy Reinforcement Learning</strong> - Jian Lu - [[pdf]](https://arxiv.org/pdf/2511.18871)</summary>

**Abstract:** Since the introduction of the GRPO algorithm, reinforcement learning (RL) has attracted increasing attention, with growing efforts to reproduce and apply it. However, training efficiency remains a critical challenge. In mainstream RL frameworks, inference and training are typically deployed on the same devices. While this approach reduces costs through resource consolidation, its synchronous execution imposes a computational coupling that prevents concurrent inference and training. In this study, we are returning to the strategy of separating inference and training deployment, and by introducing improvements in the data loader, we transform the conventional synchronous architecture into a periodically asynchronous framework, which allows for demand-driven, independent, and elastic scaling of each component, while the accuracy of the algorithm remains completely equivalent to the synchronization method, with both belonging to the on-policy strategy. It is worth emphasizing that we apply a unified tri-model architecture in the training phase, and we also proposed a shared-prompt attention mask to reduce repetitive computation. In practice, these works have achieved at least a threefold overall performance improvement in RL training on NPU platforms, indicating its potential for widespread application.

**arXiv ID:** 2511.18871
</details>

<details>
<summary><strong>Learning to Compress Graphs via Dual Agents for Consistent Topological Robustness Evaluation</strong> - Qisen Chai, Yansong Wang, Junjie Huang, Tao Jia - [[pdf]](https://arxiv.org/pdf/2511.18958)</summary>

**Abstract:** As graph-structured data grow increasingly large, evaluating their robustness under adversarial attacks becomes computationally expensive and difficult to scale. To address this challenge, we propose to compress graphs into compact representations that preserve both topological structure and robustness profile, enabling efficient and reliable evaluation. We propose Cutter, a dual-agent reinforcement learning framework composed of a Vital Detection Agent (VDA) and a Redundancy Detection Agent (RDA), which collaboratively identify structurally vital and redundant nodes for guided compression. Cutter incorporates three key strategies to enhance learning efficiency and compression quality: trajectory-level reward shaping to transform sparse trajectory returns into dense, policy-equivalent learning signals; prototype-based shaping to guide decisions using behavioral patterns from both high- and low-return trajectories; and cross-agent imitation to enable safer and more transferable exploration. Experiments on multiple real-world graphs demonstrate that Cutter generates compressed graphs that retain essential static topological properties and exhibit robustness degradation trends highly consistent with the original graphs under various attack scenarios, thereby significantly improving evaluation efficiency without compromising assessment fidelity.

**arXiv ID:** 2511.18958
</details>

<details>
<summary><strong>FastForward Pruning: Efficient LLM Pruning via Single-Step Reinforcement Learning</strong> - Xin Yuan, Siqi Li, Jiateng Wei, Chengrui Zhu, Yanming Wu, Qingpeng Li, Jiajun Lv, Xiaoke Lan, Jun Chen, Yong Liu - [[pdf]](https://arxiv.org/pdf/2511.18977)</summary>

**Abstract:** Pruning is an effective method for compressing Large Language Models, but finding an optimal, non-uniform layer-wise sparsity allocation remains a key challenge. While heuristic methods are fast but yield suboptimal performance, more powerful search-based approaches like Reinforcement Learning are often hindered by prohibitive computational costs on large-scale models. To overcome this efficiency barrier, we propose FastForward Pruning. Its core is a decoupled, single-step RL framework that separates policy optimization from the complex budget satisfaction problem. Such a decoupling is crucial for efficiently searching the vast policy space of LLMs. This curriculum-based strategy begins with low-cost, simple tasks and gradually increases in complexity, significantly reducing the search's computational overhead. Evaluated on the LLaMA, Mistral, and OPT model families, our framework discovers pruning policies that achieve superior performance over strong heuristic baselines. Crucially, when compared to other search-based algorithms, our method achieves competitive or superior results at a fraction of the computational cost, demonstrating a clear advantage in search efficiency.

**arXiv ID:** 2511.18977
</details>

<details>
<summary><strong>First-order Sobolev Reinforcement Learning</strong> - Fabian Schramm, Nicolas Perrin-Gilbert, Justin Carpentier - [[pdf]](https://arxiv.org/pdf/2511.19165)</summary>

**Abstract:** We propose a refinement of temporal-difference learning that enforces first-order Bellman consistency: the learned value function is trained to match not only the Bellman targets in value but also their derivatives with respect to states and actions. By differentiating the Bellman backup through differentiable dynamics, we obtain analytically consistent gradient targets. Incorporating these into the critic objective using a Sobolev-type loss encourages the critic to align with both the value and local geometry of the target function. This first-order TD matching principle can be seamlessly integrated into existing algorithms, such as Q-learning or actor-critic methods (e.g., DDPG, SAC), potentially leading to faster critic convergence and more stable policy gradients without altering their overall structure.

**arXiv ID:** 2511.19165
</details>

<details>
<summary><strong>Leveraging LLMs for reward function design in reinforcement learning control tasks</strong> - Franklin Cardenoso, Wouter Caarls - [[pdf]](https://arxiv.org/pdf/2511.19355)</summary>

**Abstract:** The challenge of designing effective reward functions in reinforcement learning (RL) represents a significant bottleneck, often requiring extensive human expertise and being time-consuming. Previous work and recent advancements in large language models (LLMs) have demonstrated their potential for automating the generation of reward functions. However, existing methodologies often require preliminary evaluation metrics, human-engineered feedback for the refinement process, or the use of environmental source code as context. To address these limitations, this paper introduces LEARN-Opt (LLM-based Evaluator and Analyzer for Reward functioN Optimization). This LLM-based, fully autonomous, and model-agnostic framework eliminates the need for preliminary metrics and environmental source code as context to generate, execute, and evaluate reward function candidates from textual descriptions of systems and task objectives. LEARN-Opt's main contribution lies in its ability to autonomously derive performance metrics directly from the system description and the task objective, enabling unsupervised evaluation and selection of reward functions. Our experiments indicate that LEARN-Opt achieves performance comparable to or better to that of state-of-the-art methods, such as EUREKA, while requiring less prior knowledge. We find that automated reward design is a high-variance problem, where the average-case candidate fails, requiring a multi-run approach to find the best candidates. Finally, we show that LEARN-Opt can unlock the potential of low-cost LLMs to find high-performing candidates that are comparable to, or even better than, those of larger models. This demonstrated performance affirms its potential to generate high-quality reward functions without requiring any preliminary human-defined metrics, thereby reducing engineering overhead and enhancing generalizability.

**arXiv ID:** 2511.19355
</details>

<details>
<summary><strong>A Reinforcement Learning Framework for Resource Allocation in Uplink Carrier Aggregation in the Presence of Self Interference</strong> - Jaswanth Bodempudi, Batta Siva Sairam, Madepalli Haritha, Sandesh Rao Mattu, Ananthanarayanan Chockalingam - [[pdf]](https://arxiv.org/pdf/2511.17931)</summary>

**Abstract:** Carrier aggregation (CA) is a technique that allows mobile networks to combine multiple carriers to increase user data rate. On the uplink, for power constrained users, this translates to the need for an efficient resource allocation scheme, where each user distributes its available power among its assigned uplink carriers. Choosing a good set of carriers and allocating appropriate power on the carriers is important. If the carrier allocation on the uplink is such that a harmonic of a user's uplink carrier falls on the downlink frequency of that user, it leads to a self coupling-induced sensitivity degradation of that user's downlink receiver. In this paper, we model the uplink carrier aggregation problem as an optimal resource allocation problem with the associated constraints of non-linearities induced self interference (SI). This involves optimization over a discrete variable (which carriers need to be turned on) and a continuous variable (what power needs to be allocated on the selected carriers) in dynamic environments, a problem which is hard to solve using traditional methods owing to the mixed nature of the optimization variables and the additional need to consider the SI constraint. We adopt a reinforcement learning (RL) framework involving a compound-action actor-critic (CA2C) algorithm for the uplink carrier aggregation problem. We propose a novel reward function that is critical for enabling the proposed CA2C algorithm to efficiently handle SI. The CA2C algorithm along with the proposed reward function learns to assign and activate suitable carriers in an online fashion. Numerical results demonstrate that the proposed RL based scheme is able to achieve higher sum throughputs compared to naive schemes. The results also demonstrate that the proposed reward function allows the CA2C algorithm to adapt the optimization both in the presence and absence of SI.

**arXiv ID:** 2511.17931
</details>

<details>
<summary><strong>Dreaming Falcon: Physics-Informed Model-Based Reinforcement Learning for Quadcopters</strong> - Eashan Vytla, Bhavanishankar Kalavakolanu, Andrew Perrault, Matthew McCrink - [[pdf]](https://arxiv.org/pdf/2511.18243)</summary>

**Abstract:** Current control algorithms for aerial robots struggle with robustness in dynamic environments and adverse conditions. Model-based reinforcement learning (RL) has shown strong potential in handling these challenges while remaining sample-efficient. Additionally, Dreamer has demonstrated that online model-based RL can be achieved using a recurrent world model trained on replay buffer data. However, applying Dreamer to aerial systems has been quite challenging due to its sample inefficiency and poor generalization of dynamics models. Our work explores a physics-informed approach to world model learning and improves policy performance. The world model treats the quadcopter as a free-body system and predicts the net forces and moments acting on it, which are then passed through a 6-DOF Runge-Kutta integrator (RK4) to predict future state rollouts. In this paper, we compare this physics-informed method to a standard RNN-based world model. Although both models perform well on the training data, we observed that they fail to generalize to new trajectories, leading to rapid divergence in state rollouts, preventing policy convergence.

**arXiv ID:** 2511.18243
</details>

<details>
<summary><strong>Accelerating Reinforcement Learning via Error-Related Human Brain Signals</strong> - Suzie Kim, Hye-Bin Shin, Hyo-Jeong Jang - [[pdf]](https://arxiv.org/pdf/2511.18878)</summary>

**Abstract:** In this work, we investigate how implicit neural feed back can accelerate reinforcement learning in complex robotic manipulation settings. While prior electroencephalogram (EEG) guided reinforcement learning studies have primarily focused on navigation or low-dimensional locomotion tasks, we aim to understand whether such neural evaluative signals can improve policy learning in high-dimensional manipulation tasks involving obstacles and precise end-effector control. We integrate error related potentials decoded from offline-trained EEG classifiers into reward shaping and systematically evaluate the impact of human-feedback weighting. Experiments on a 7-DoF manipulator in an obstacle-rich reaching environment show that neural feedback accelerates reinforcement learning and, depending on the human-feedback weighting, can yield task success rates that at times exceed those of sparse-reward baselines. Moreover, when applying the best-performing feedback weighting across all sub jects, we observe consistent acceleration of reinforcement learning relative to the sparse-reward setting. Furthermore, leave-one subject-out evaluations confirm that the proposed framework remains robust despite the intrinsic inter-individual variability in EEG decodability. Our findings demonstrate that EEG-based reinforcement learning can scale beyond locomotion tasks and provide a viable pathway for human-aligned manipulation skill acquisition.

**arXiv ID:** 2511.18878
</details>

<details>
<summary><strong>Yummy Operations Robot Initiative: Autonomous Cooking System Utilizing a Modular Robotic Kitchen and a Dual-Arm Proprioceptive Manipulator</strong> - Donghun Noh, Hyunwoo Nam, Kyle Gillespie, Yeting Liu, Dennis Hong - [[pdf]](https://arxiv.org/pdf/2405.11094)</summary>

**Abstract:** This paper presents Yummy Operations Robot Initiative (YORI), a proprioceptive dual-arm robotic system that demonstrates autonomous multi-dish cooking for scalable food service applications. YORI integrates a dual-arm manipulator equipped with proprioceptive actuators, custom-designed tools, appliances, and a structured kitchen environment to address the complexities of cooking tasks. The proprioceptive actuators enable fast, precise, force-controlled movements while mitigating the risks associated with cooking-related impacts. The system's modular kitchen design and flexible tool-changing mechanism support simultaneous multi-dish preparation through torque control and optimization-based motion planning and scheduling. A comprehensive scheduling framework with dynamic rescheduling ensures reliable adaptation to new orders and delays. The system was publicly validated through live demonstrations, reliably preparing steak-frites across multiple convention sessions. This paper details YORI's design and explores future directions in kitchen optimization, task planning, and food quality control, demonstrating its potential as a scalable robotic cooking solution. A system introduction and cooking videos are available online.

**arXiv ID:** 2405.11094
</details>

<details>
<summary><strong>Multi-Timescale Hierarchical Reinforcement Learning for Unified Behavior and Control of Autonomous Driving</strong> - Guizhe Jin, Zhuoren Li, Bo Leng, Ran Yu, Lu Xiong, Chen Sun - [[pdf]](https://arxiv.org/pdf/2506.23771)</summary>

**Abstract:** Reinforcement Learning (RL) is increasingly used in autonomous driving (AD) and shows clear advantages. However, most RL-based AD methods overlook policy structure design. An RL policy that only outputs short-timescale vehicle control commands results in fluctuating driving behavior due to fluctuations in network outputs, while one that only outputs long-timescale driving goals cannot achieve unified optimality of driving behavior and control. Therefore, we propose a multi-timescale hierarchical reinforcement learning approach. Our approach adopts a hierarchical policy structure, where high- and low-level RL policies are unified-trained to produce long-timescale motion guidance and short-timescale control commands, respectively. Therein, motion guidance is explicitly represented by hybrid actions to capture multimodal driving behaviors on structured road and support incremental low-level extend-state updates. Additionally, a hierarchical safety mechanism is designed to ensure multi-timescale safety. Evaluation in simulator-based and HighD dataset-based highway multi-lane scenarios demonstrates that our approach significantly improves AD performance, effectively increasing driving efficiency, action consistency and safety.

**arXiv ID:** 2506.23771
</details>

<details>
<summary><strong>Learning Primitive Embodied World Models: Towards Scalable Robotic Learning</strong> - Qiao Sun, Liujia Yang, Wei Tang, Wei Huang, Kaixin Xu, Yongchao Chen, Mingyu Liu, Jiange Yang, Haoyi Zhu, Yating Wang, Tong He, Yilun Chen, Xili Dai, Nanyang Ye, Qinying Gu - [[pdf]](https://arxiv.org/pdf/2508.20840)</summary>

**Abstract:** While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a "GPT moment" in the embodied domain. There is a naive observation: the diversity of embodied data far exceeds the relatively small space of possible primitive motions. Based on this insight, we propose a novel paradigm for world modeling--Primitive Embodied World Models (PEWM). By restricting video generation to fixed short horizons, our approach 1) enables fine-grained alignment between linguistic concepts and visual representations of robotic actions, 2) reduces learning complexity, 3) improves data efficiency in embodied data collection, and 4) decreases inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence.

**arXiv ID:** 2508.20840
</details>

<details>
<summary><strong>PresentCoach: Dual-Agent Presentation Coaching through Exemplars and Interactive Feedback</strong> - Sirui Chen, Jinsong Zhou, Xinli Xu, Xiaoyu Yang, Litao Guo, Ying-Cong Chen - [[pdf]](https://arxiv.org/pdf/2511.15253)</summary>

**Abstract:** Effective presentation skills are essential in education, professional communication, and public speaking, yet learners often lack access to high-quality exemplars or personalized coaching. Existing AI tools typically provide isolated functionalities such as speech scoring or script generation without integrating reference modeling and interactive feedback into a cohesive learning experience. We introduce a dual-agent system that supports presentation practice through two complementary roles: the Ideal Presentation Agent and the Coach Agent. The Ideal Presentation Agent converts user-provided slides into model presentation videos by combining slide processing, visual-language analysis, narration script generation, personalized voice synthesis, and synchronized video assembly. The Coach Agent then evaluates user-recorded presentations against these exemplars, conducting multimodal speech analysis and delivering structured feedback in an Observation-Impact-Suggestion (OIS) format. To enhance the authenticity of the learning experience, the Coach Agent incorporates an Audience Agent, which simulates the perspective of a human listener and provides humanized feedback reflecting audience reactions and engagement. Together, these agents form a closed loop of observation, practice, and feedback. Implemented on a robust backend with multi-model integration, voice cloning, and error handling mechanisms, the system demonstrates how AI-driven agents can provide engaging, human-centered, and scalable support for presentation skill development in both educational and professional contexts.

**arXiv ID:** 2511.15253
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
