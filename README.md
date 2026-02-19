# Agent arXiv Daily

**Last Updated:** 2026-02-19 03:13:52

**Total Papers:** 69

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
<summary><strong>Optimization Instability in Autonomous Agentic Workflows for Clinical Symptom Detection</strong> - Cameron Cagan, Pedram Fard, Jiazi Tian, Jingya Cheng, Shawn N. Murphy, Hossein Estiri - [[pdf]](https://arxiv.org/pdf/2602.16037)</summary>

**Abstract:** Autonomous agentic workflows that iteratively refine their own behavior hold considerable promise, yet their failure modes remain poorly characterized. We investigate optimization instability, a phenomenon in which continued autonomous improvement paradoxically degrades classifier performance, using Pythia, an open-source framework for automated prompt optimization. Evaluating three clinical symptoms with varying prevalence (shortness of breath at 23%, chest pain at 12%, and Long COVID brain fog at 3%), we observed that validation sensitivity oscillated between 1.0 and 0.0 across iterations, with severity inversely proportional to class prevalence. At 3% prevalence, the system achieved 95% accuracy while detecting zero positive cases, a failure mode obscured by standard evaluation metrics. We evaluated two interventions: a guiding agent that actively redirected optimization, amplifying overfitting rather than correcting it, and a selector agent that retrospectively identified the best-performing iteration successfully prevented catastrophic failure. With selector agent oversight, the system outperformed expert-curated lexicons on brain fog detection by 331% (F1) and chest pain by 7%, despite requiring only a single natural language term as input. These findings characterize a critical failure mode of autonomous AI systems and demonstrate that retrospective selection outperforms active intervention for stabilization in low-prevalence classification tasks.

**arXiv ID:** 2602.16037
</details>

<details>
<summary><strong>RelianceScope: An Analytical Framework for Examining Students' Reliance on Generative AI Chatbots in Problem Solving</strong> - Hyoungwook Jin, Minju Yoo, Jieun Han, Zixin Chen, So-Yeon Ahn, Xu Wang - [[pdf]](https://arxiv.org/pdf/2602.16251)</summary>

**Abstract:** Generative AI chatbots enable personalized problem-solving, but effective learning requires students to self-regulate both how they seek help and how they use AI-generated responses. Considering engagement modes across these two actions reveals nuanced reliance patterns: for example, a student may actively engage in help-seeking by clearly specifying areas of need, yet engage passively in response-use by copying AI outputs, or vice versa. However, existing research lacks systematic tools for jointly capturing engagement across help-seeking and response-use, limiting the analysis of such reliance behaviors. We introduce RelianceScope, an analytical framework that characterizes students' reliance on chatbots during problem-solving. RelianceScope (1) operationalizes reliance into nine patterns based on combinations of engagement modes in help-seeking and response-use, and (2) situates these patterns within a knowledge-context lens that accounts for students' prior knowledge and the instructional significance of knowledge components. Rather than prescribing optimal AI use, the framework enables fine-grained analysis of reliance in open-ended student-AI interactions. As an illustrative application, we applied RelianceScope to analyze chat and code-edit logs from 79 college students in a web programming course. Results show that active help-seeking is associated with active response-use, whereas reliance patterns remain similar across knowledge mastery levels. Students often struggled to articulate their knowledge gaps and to adapt AI responses. Using our annotated dataset as a benchmark, we further demonstrate that large language models can reliably detect reliance during help-seeking and response-use. We conclude by discussing the implications of RelianceScope and the design guidelines for AI-supported educational systems.

**arXiv ID:** 2602.16251
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (15 papers)</h2></summary>

<details>
<summary><strong>Learning Personalized Agents from Human Feedback</strong> - Kaiqu Liang, Julia Kruk, Shengyi Qian, Xianjun Yang, Shengjie Bi, Yuanshun Yao, Shaoliang Nie, Mingyang Zhang, Lijuan Liu, Jaime Fernández Fisac, Shuyan Zhou, Saghar Hosseini - [[pdf]](https://arxiv.org/pdf/2602.16173)</summary>

**Abstract:** Modern AI agents are powerful but often fail to align with the idiosyncratic, evolving preferences of individual users. Prior approaches typically rely on static datasets, either training implicit preference models on interaction history or encoding user profiles in external memory. However, these approaches struggle with new users and with preferences that change over time. We introduce Personalized Agents from Human Feedback (PAHF), a framework for continual personalization in which agents learn online from live interaction using explicit per-user memory. PAHF operationalizes a three-step loop: (1) seeking pre-action clarification to resolve ambiguity, (2) grounding actions in preferences retrieved from memory, and (3) integrating post-action feedback to update memory when preferences drift. To evaluate this capability, we develop a four-phase protocol and two benchmarks in embodied manipulation and online shopping. These benchmarks quantify an agent's ability to learn initial preferences from scratch and subsequently adapt to persona shifts. Our theoretical analysis and empirical results show that integrating explicit memory with dual feedback channels is critical: PAHF learns substantially faster and consistently outperforms both no-memory and single-channel baselines, reducing initial personalization error and enabling rapid adaptation to preference shifts.

**arXiv ID:** 2602.16173
</details>

<details>
<summary><strong>Towards a Science of AI Agent Reliability</strong> - Stephan Rabanser, Sayash Kapoor, Peter Kirgis, Kangheng Liu, Saiteja Utpala, Arvind Narayanan - [[pdf]](https://arxiv.org/pdf/2602.16666)</summary>

**Abstract:** AI agents are increasingly deployed to execute important tasks. While rising accuracy scores on standard benchmarks suggest rapid progress, many agents still continue to fail in practice. This discrepancy highlights a fundamental limitation of current evaluations: compressing agent behavior into a single success metric obscures critical operational flaws. Notably, it ignores whether agents behave consistently across runs, withstand perturbations, fail predictably, or have bounded error severity. Grounded in safety-critical engineering, we provide a holistic performance profile by proposing twelve concrete metrics that decompose agent reliability along four key dimensions: consistency, robustness, predictability, and safety. Evaluating 14 agentic models across two complementary benchmarks, we find that recent capability gains have only yielded small improvements in reliability. By exposing these persistent limitations, our metrics complement traditional evaluations while offering tools for reasoning about how agents perform, degrade, and fail.

**arXiv ID:** 2602.16666
</details>

<details>
<summary><strong>Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis</strong> - Pengfei Zhang, Tianxin Xie, Minghao Yang, Li Liu - [[pdf]](https://arxiv.org/pdf/2602.15909)</summary>

**Abstract:** Deep learning-based respiratory auscultation is currently hindered by two fundamental challenges: (i) inherent information loss, as converting signals into spectrograms discards transient acoustic events and clinical context; (ii) limited data availability, exacerbated by severe class imbalance. To bridge these gaps, we present Resp-Agent, an autonomous multimodal system orchestrated by a novel Active Adversarial Curriculum Agent (Thinker-A$^2$CA). Unlike static pipelines, Thinker-A$^2$CA serves as a central controller that actively identifies diagnostic weaknesses and schedules targeted synthesis in a closed loop. To address the representation gap, we introduce a Modality-Weaving Diagnoser that weaves EHR data with audio tokens via Strategic Global Attention and sparse audio anchors, capturing both long-range clinical context and millisecond-level transients. To address the data gap, we design a Flow Matching Generator that adapts a text-only Large Language Model (LLM) via modality injection, decoupling pathological content from acoustic style to synthesize hard-to-diagnose samples. As a foundation for these efforts, we introduce Resp-229k, a benchmark corpus of 229k recordings paired with LLM-distilled clinical narratives. Extensive experiments demonstrate that Resp-Agent consistently outperforms prior approaches across diverse evaluation settings, improving diagnostic robustness under data scarcity and long-tailed class imbalance. Our code and data are available at this https URL.

**arXiv ID:** 2602.15909
</details>

<details>
<summary><strong>ScenicRules: An Autonomous Driving Benchmark with Multi-Objective Specifications and Abstract Scenarios</strong> - Kevin Kai-Chun Chang, Ekin Beyazit, Alberto Sangiovanni-Vincentelli, Tichakorn Wongpiromsarn, Sanjit A. Seshia - [[pdf]](https://arxiv.org/pdf/2602.16073)</summary>

**Abstract:** Developing autonomous driving systems for complex traffic environments requires balancing multiple objectives, such as avoiding collisions, obeying traffic rules, and making efficient progress. In many situations, these objectives cannot be satisfied simultaneously, and explicit priority relations naturally arise. Also, driving rules require context, so it is important to formally model the environment scenarios within which such rules apply. Existing benchmarks for evaluating autonomous vehicles lack such combinations of multi-objective prioritized rules and formal environment models. In this work, we introduce ScenicRules, a benchmark for evaluating autonomous driving systems in stochastic environments under prioritized multi-objective specifications. We first formalize a diverse set of objectives to serve as quantitative evaluation metrics. Next, we design a Hierarchical Rulebook framework that encodes multiple objectives and their priority relations in an interpretable and adaptable manner. We then construct a compact yet representative collection of scenarios spanning diverse driving contexts and near-accident situations, formally modeled in the Scenic language. Experimental results show that our formalized objectives and Hierarchical Rulebooks align well with human driving judgments and that our benchmark effectively exposes agent failures with respect to the prioritized objectives. Our benchmark can be accessed at this https URL.

**arXiv ID:** 2602.16073
</details>

<details>
<summary><strong>RoboGene: Boosting VLA Pre-training via Diversity-Driven Agentic Framework for Real-World Task Generation</strong> - Yixue Zhang, Kun Wu, Zhi Gao, Zhen Zhao, Pei Ren, Zhiyuan Xu, Fei Liao, Xinhua Wang, Shichao Fan, Di Wu, Qiuxuan Feng, Meng Li, Zhengping Che, Chang Liu, Jian Tang - [[pdf]](https://arxiv.org/pdf/2602.16444)</summary>

**Abstract:** The pursuit of general-purpose robotic manipulation is hindered by the scarcity of diverse, real-world interaction data. Unlike data collection from web in vision or language, robotic data collection is an active process incurring prohibitive physical costs. Consequently, automated task curation to maximize data value remains a critical yet under-explored challenge. Existing manual methods are unscalable and biased toward common tasks, while off-the-shelf foundation models often hallucinate physically infeasible instructions. To address this, we introduce RoboGene, an agentic framework designed to automate the generation of diverse, physically plausible manipulation tasks across single-arm, dual-arm, and mobile robots. RoboGene integrates three core components: diversity-driven sampling for broad task coverage, self-reflection mechanisms to enforce physical constraints, and human-in-the-loop refinement for continuous improvement. We conduct extensive quantitative analysis and large-scale real-world experiments, collecting datasets of 18k trajectories and introducing novel metrics to assess task quality, feasibility, and diversity. Results demonstrate that RoboGene significantly outperforms state-of-the-art foundation models (e.g., GPT-4o, Gemini 2.5 Pro). Furthermore, real-world experiments show that VLA models pre-trained with RoboGene achieve higher success rates and superior generalization, underscoring the importance of high-quality task generation. Our project is available at this https URL.

**arXiv ID:** 2602.16444
</details>

<details>
<summary><strong>PLAICraft: Large-Scale Time-Aligned Vision-Speech-Action Dataset for Embodied AI</strong> - Yingchen He, Christian D. Weilbach, Martyna E. Wojciechowska, Yuxuan Zhang, Frank Wood - [[pdf]](https://arxiv.org/pdf/2505.12707)</summary>

**Abstract:** Advances in deep generative modeling have made it increasingly plausible to train human-level embodied agents. Yet progress has been limited by the absence of large-scale, real-time, multi-modal, and socially interactive datasets that reflect the sensory-motor complexity of natural environments. To address this, we present PLAICraft, a novel data collection platform and dataset capturing multiplayer Minecraft interactions across five time-aligned modalities: video, game output audio, microphone input audio, mouse, and keyboard actions. Each modality is logged with millisecond time precision, enabling the study of synchronous, embodied behaviour in a rich, open-ended world. The dataset comprises over 10,000 hours of gameplay from more than 10,000 global participants. Alongside the dataset, we provide an evaluation suite for benchmarking model capabilities in object recognition, spatial awareness, language grounding, and long-term memory. PLAICraft opens a path toward training and evaluating agents that act fluently and purposefully in real time, paving the way for truly embodied artificial intelligence.

**arXiv ID:** 2505.12707
</details>

<details>
<summary><strong>Arming Data Agents with Tribal Knowledge</strong> - Shubham Agarwal, Asim Biswal, Sepanta Zeighami, Alvin Cheung, Joseph Gonzalez, Aditya G. Parameswaran - [[pdf]](https://arxiv.org/pdf/2602.13521)</summary>

**Abstract:** Natural language to SQL (NL2SQL) translation enables non-expert users to query relational databases through natural language. Recently, NL2SQL agents, powered by the reasoning capabilities of Large Language Models (LLMs), have significantly advanced NL2SQL translation. Nonetheless, NL2SQL agents still make mistakes when faced with large-scale real-world databases because they lack knowledge of how to correctly leverage the underlying data (e.g., knowledge about the intent of each column) and form misconceptions about the data when querying it, leading to errors. Prior work has studied generating facts about the database to provide more context to NL2SQL agents, but such approaches simply restate database contents without addressing the agent's misconceptions. In this paper, we propose Tk-Boost, a bolt-on framework for augmenting any NL2SQL agent with tribal knowledge: knowledge that corrects the agent's misconceptions in querying the database accumulated through experience using the database. To accumulate experience, Tk-Boost first asks the NL2SQL agent to answer a few queries on the database, identifies the agent's misconceptions by analyzing its mistakes on the database, and generates tribal knowledge to address them. To enable accurate retrieval, Tk-Boost indexes this knowledge with applicability conditions that specify the query features for which the knowledge is useful. When answering new queries, Tk-Boost uses this knowledge to provide feedback to the NL2SQL agent, resolving the agent's misconceptions during SQL generation, and thus improving the agent's accuracy. Extensive experiments across the BIRD and Spider 2.0 benchmarks with various NL2SQL agents shows Tk-Boost improves NL2SQL agents accuracy by up to 16.9% on Spider 2.0 and 13.7% on BIRD

**arXiv ID:** 2602.13521
</details>

<details>
<summary><strong>From Transcripts to AI Agents: Knowledge Extraction, RAG Integration, and Robust Evaluation of Conversational AI Assistants</strong> - Krittin Pachtrachai, Petmongkon Pornpichitsuwan, Wachiravit Modecrua, Touchapon Kraisingkorn - [[pdf]](https://arxiv.org/pdf/2602.15859)</summary>

**Abstract:** Building reliable conversational AI assistants for customer-facing industries remains challenging due to noisy conversational data, fragmented knowledge, and the requirement for accurate human hand-off - particularly in domains that depend heavily on real-time information. This paper presents an end-to-end framework for constructing and evaluating a conversational AI assistant directly from historical call transcripts. Incoming transcripts are first graded using a simplified adaptation of the PIPA framework, focusing on observation alignment and appropriate response behavior, and are filtered to retain only high-quality interactions exhibiting coherent flow and effective human agent responses. Structured knowledge is then extracted from curated transcripts using large language models (LLMs) and deployed as the sole grounding source in a Retrieval-Augmented Generation (RAG) pipeline. Assistant behavior is governed through systematic prompt tuning, progressing from monolithic prompts to lean, modular, and governed designs that ensure consistency, safety, and controllable execution. Evaluation is conducted using a transcript-grounded user simulator, enabling quantitative measurement of call coverage, factual accuracy, and human escalation behavior. Additional red teaming assesses robustness against prompt injection, out-of-scope, and out-of-context attacks. Experiments are conducted in the Real Estate and Specialist Recruitment domains, which are intentionally challenging and currently suboptimal for automation due to their reliance on real-time data. Despite these constraints, the assistant autonomously handles approximately 30 percents of calls, achieves near-perfect factual accuracy and rejection behavior, and demonstrates strong robustness under adversarial testing.

**arXiv ID:** 2602.15859
</details>

<details>
<summary><strong>TabAgent: A Framework for Replacing Agentic Generative Components with Tabular-Textual Classifiers</strong> - Ido Levy, Eilam Shapira, Yinon Goldshtein, Avi Yaeli, Nir Mashkif, Segev Shlomov - [[pdf]](https://arxiv.org/pdf/2602.16429)</summary>

**Abstract:** Agentic systems, AI architectures that autonomously execute multi-step workflows to achieve complex goals, are often built using repeated large language model (LLM) calls for closed-set decision tasks such as routing, shortlisting, gating, and verification. While convenient, this design makes deployments slow and expensive due to cumulative latency and token usage. We propose TabAgent, a framework for replacing generative decision components in closed-set selection tasks with a compact textual-tabular classifier trained on execution traces. TabAgent (i) extracts structured schema, state, and dependency features from trajectories (TabSchema), (ii) augments coverage with schema-aligned synthetic supervision (TabSynth), and (iii) scores candidates with a lightweight classifier (TabHead). On the long-horizon AppWorld benchmark, TabAgent maintains task-level success while eliminating shortlist-time LLM calls, reducing latency by approximately 95% and inference cost by 85-91%. Beyond tool shortlisting, TabAgent generalizes to other agentic decision heads, establishing a paradigm for learned discriminative replacements of generative bottlenecks in production agent architectures.

**arXiv ID:** 2602.16429
</details>

<details>
<summary><strong>A Methodology for Identifying Evaluation Items for Practical Dialogue Systems Based on Business-Dialogue System Alignment Models</strong> - Mikio Nakano, Hironori Takeuchi, Kazunori Komatani - [[pdf]](https://arxiv.org/pdf/2602.15835)</summary>

**Abstract:** This paper proposes a methodology for identifying evaluation items for practical dialogue systems. Traditionally, user satisfaction and user experiences have been the primary metrics for evaluating dialogue systems. However, there are various other evaluation items to consider when developing and operating practical dialogue systems, and such evaluation items are expected to lead to new research topics. So far, there has been no methodology for identifying these evaluation items. We propose identifying evaluation items based on business-dialogue system alignment models, which are applications of business-IT alignment models used in the development and operation of practical IT systems. We also present a generic model that facilitates the construction of a business-dialogue system alignment model for each dialogue system.

**arXiv ID:** 2602.15835
</details>

<details>
<summary><strong>CAST: Character-and-Scene Episodic Memory for Agents</strong> - Kexin Ma, Bojun Li, Yuhua Tang, Liting Sun, Ruochun Jin - [[pdf]](https://arxiv.org/pdf/2602.06051)</summary>

**Abstract:** Episodic memory is a central component of human memory, which refers to the ability to recall coherent events grounded in who, when, and where. However, most agent memory systems only emphasize semantic recall and treat experience as structures such as key-value, vector, or graph, which makes them struggle to represent and retrieve coherent events. To address this challenge, we propose a Character-and-Scene based memory architecture(CAST) inspired by dramatic theory. Specifically, CAST constructs 3D scenes (time/place/topic) and organizes them into character profiles that summarize the events of a character to represent episodic memory. Moreover, CAST complements this episodic memory with a graph-based semantic memory, which yields a robust dual memory design. Experiments demonstrate that CAST has averagely improved 8.11% F1 and 10.21% J(LLM-as-a-Judge) than baselines on various datasets, especially on open and time-sensitive conversational questions.

**arXiv ID:** 2602.06051
</details>

<details>
<summary><strong>From Conflicts to Collisions: A Two-Stage Collision Scenario-Testing Approach for Autonomous Driving Systems</strong> - Siyuan Chen, Fuyuan Zhang, Hua Qi, Lei Ma, Tomoyuki Tsuchiya, Michio Hayashi, Manabu Okada - [[pdf]](https://arxiv.org/pdf/2602.15837)</summary>

**Abstract:** Autonomous driving systems (ADS) are safety-critical and require rigorous testing before public deployment. Simulation-based scenario testing provides a safe and cost-effective alternative to extensive on-road trials, enabling efficient evaluation of ADS under diverse and high-risk conditions. However, existing approaches mainly evaluates the scenarios based on their proximity to collisions and focus on scenarios already close to collision, leaving many other hazardous situations unexplored. To bridge this, we introduce a collision-related concept of conflict as an intermediate search target and propose a two-stage scenario testing framework that first searches for conflicts and then mutates these conflict scenarios to induce actual collisions. Evaluated on Baidu Apollo, our approach reveals up to 12 distinct collision types in a single run, doubling the diversity discovered by state-of-the-art baselines while requiring fewer simulations thanks to conflict-targeted mutations. These results show that using conflicts as intermediate objectives broadens the search horizon and significantly improves the efficiency and effectiveness of ADS safety evaluation.

**arXiv ID:** 2602.15837
</details>

<details>
<summary><strong>Coverage Path Planning for Autonomous Sailboats in Inhomogeneous and Time-Varying Oceans: A Spatiotemporal Optimization Approach</strong> - Yang An, Zhikang Ge, Taiyu Zhang, Jean-Baptiste R. G. Souppez, Gaofei Xu, Zhengru Ren - [[pdf]](https://arxiv.org/pdf/2602.15901)</summary>

**Abstract:** Autonomous sailboats are well suited for long-duration ocean observation due to their wind-driven endurance. However, their performance is highly anisotropic and strongly influenced by inhomogeneous and time-varying wind and current fields, limiting the effectiveness of existing coverage methods such as boustrophedon sweeping. Planning under these environmental and maneuvering constraints remains underexplored. This paper presents a spatiotemporal coverage path planning framework tailored to autonomous sailboats, combining (1) topology-based morphological constraints in the spatial domain to promote compact and continuous coverage, and (2) forecast-aware look-ahead planning in the temporal domain to anticipate environmental evolution and enable foresighted decision-making. Simulations conducted under stochastic inhomogeneous and time-varying ocean environments, including scenarios with partial directional accessibility, demonstrate that the proposed method generates efficient and feasible coverage paths where traditional strategies often fail. To the best of our knowledge, this study provides the first dedicated solution to the coverage path planning problem for autonomous sailboats operating in inhomogeneous and time-varying ocean environments, establishing a foundation for future cooperative multi-sailboat coverage.

**arXiv ID:** 2602.15901
</details>

<details>
<summary><strong>Towards Autonomous Robotic Kidney Ultrasound: Spatial-Efficient Volumetric Imaging via Template Guided Optimal Pivoting</strong> - Xihan Ma, Haichong Zhang - [[pdf]](https://arxiv.org/pdf/2602.16641)</summary>

**Abstract:** Medical ultrasound (US) imaging is a frontline tool for the diagnosis of kidney diseases. However, traditional freehand imaging procedure suffers from inconsistent, operator-dependent outcomes, lack of 3D localization information, and risks of work-related musculoskeletal disorders. While robotic ultrasound (RUS) systems offer the potential for standardized, operator-independent 3D kidney data acquisition, the existing scanning methods lack the ability to determine the optimal imaging window for efficient imaging. As a result, the scan is often blindly performed with excessive probe footprint, which frequently leads to acoustic shadowing and incomplete organ coverage. Consequently, there is a critical need for a spatially efficient imaging technique that can maximize the kidney coverage through minimum probe footprint. Here, we propose an autonomous workflow to achieve efficient kidney imaging via template-guided optimal pivoting. The system first performs an explorative imaging to generate partial observations of the kidney. This data is then registered to a kidney template to estimate the organ pose. With the kidney localized, the robot executes a fixed-point pivoting sweep where the imaging plane is aligned with the kidney long axis to minimize the probe translation. The proposed method was validated in simulation and in-vivo. Simulation results indicate that a 60% exploration ratio provides optimal balance between kidney localization accuracy and scanning efficiency. In-vivo evaluation on two male subjects demonstrates a kidney localization accuracy up to 7.36 mm and 13.84 degrees. Moreover, the optimal pivoting approach shortened the probe footprint by around 75 mm when compared with the baselines. These results valid our approach of leveraging anatomical templates to align the probe optimally for volumetric sweep.

**arXiv ID:** 2602.16641
</details>

<details>
<summary><strong>A Comprehensive Survey on Deep Learning-Based LiDAR Super-Resolution for Autonomous Driving</strong> - June Moh Goo, Zichao Zeng, Jan Boehm - [[pdf]](https://arxiv.org/pdf/2602.15904)</summary>

**Abstract:** LiDAR sensors are often considered essential for autonomous driving, but high-resolution sensors remain expensive while affordable low-resolution sensors produce sparse point clouds that miss critical details. LiDAR super-resolution addresses this challenge by using deep learning to enhance sparse point clouds, bridging the gap between different sensor types and enabling cross-sensor compatibility in real-world deployments. This paper presents the first comprehensive survey of LiDAR super-resolution methods for autonomous driving. Despite the importance of practical deployment, no systematic review has been conducted until now. We organize existing approaches into four categories: CNN-based architectures, model-based deep unrolling, implicit representation methods, and Transformer and Mamba-based approaches. We establish fundamental concepts including data representations, problem formulation, benchmark datasets and evaluation metrics. Current trends include the adoption of range image representation for efficient processing, extreme model compression and the development of resolution-flexible architectures. Recent research prioritizes real-time inference and cross-sensor generalization for practical deployment. We conclude by identifying open challenges and future research directions for advancing LiDAR super-resolution technology.

**arXiv ID:** 2602.15904
</details>

</details>

<details open>
<summary><h2>LLM Agents (13 papers)</h2></summary>

<details>
<summary><strong>Toward Scalable Verifiable Reward: Proxy State-Based Evaluation for Multi-turn Tool-Calling LLM Agents</strong> - Yun-Shiuan Chuang, Chaitanya Kulkarni, Alec Chiu, Avinash Thangali, Zijie Pan, Shivani Shekhar, Yirou Ge, Yixi Li, Uma Kona, Linsey Pang, Prakhar Mehrotra - [[pdf]](https://arxiv.org/pdf/2602.16246)</summary>

**Abstract:** Interactive large language model (LLM) agents operating via multi-turn dialogue and multi-step tool calling are increasingly used in production. Benchmarks for these agents must both reliably compare models and yield on-policy training data. Prior agentic benchmarks (e.g., tau-bench, tau2-bench, AppWorld) rely on fully deterministic backends, which are costly to build and iterate. We propose Proxy State-Based Evaluation, an LLM-driven simulation framework that preserves final state-based evaluation without a deterministic database. Specifically, a scenario specifies the user goal, user/system facts, expected final state, and expected agent behavior, and an LLM state tracker infers a structured proxy state from the full interaction trace. LLM judges then verify goal completion and detect tool/user hallucinations against scenario constraints. Empirically, our benchmark produces stable, model-differentiating rankings across families and inference-time reasoning efforts, and its on-/off-policy rollouts provide supervision that transfers to unseen scenarios. Careful scenario specification yields near-zero simulator hallucination rates as supported by ablation studies. The framework also supports sensitivity analyses over user personas. Human-LLM judge agreement exceeds 90%, indicating reliable automated evaluation. Overall, proxy state-based evaluation offers a practical, scalable alternative to deterministic agentic benchmarks for industrial LLM agents.

**arXiv ID:** 2602.16246
</details>

<details>
<summary><strong>HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents</strong> - Jiangweizhi Peng, Yuanxin Liu, Ruida Zhou, Charles Fleming, Zhaoran Wang, Alfredo Garcia, Mingyi Hong - [[pdf]](https://arxiv.org/pdf/2602.16165)</summary>

**Abstract:** Training LLMs as interactive agents for multi-turn decision-making remains challenging, particularly in long-horizon tasks with sparse and delayed rewards, where agents must execute extended sequences of actions before receiving meaningful feedback. Most existing reinforcement learning (RL) approaches model LLM agents as flat policies operating at a single time scale, selecting one action at each turn. In sparse-reward settings, such flat policies must propagate credit across the entire trajectory without explicit temporal abstraction, which often leads to unstable optimization and inefficient credit assignment.
We propose HiPER, a novel Hierarchical Plan-Execute RL framework that explicitly separates high-level planning from low-level execution. HiPER factorizes the policy into a high-level planner that proposes subgoals and a low-level executor that carries them out over multiple action steps. To align optimization with this structure, we introduce a key technique called hierarchical advantage estimation (HAE), which carefully assigns credit at both the planning and execution levels. By aggregating returns over the execution of each subgoal and coordinating updates across the two levels, HAE provides an unbiased gradient estimator and provably reduces variance compared to flat generalized advantage estimation.
Empirically, HiPER achieves state-of-the-art performance on challenging interactive benchmarks, reaching 97.4\% success on ALFWorld and 83.3\% on WebShop with Qwen2.5-7B-Instruct (+6.6\% and +8.3\% over the best prior method), with especially large gains on long-horizon tasks requiring multiple dependent subtasks. These results highlight the importance of explicit hierarchical decomposition for scalable RL training of multi-turn LLM agents.

**arXiv ID:** 2602.16165
</details>

<details>
<summary><strong>Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents</strong> - Wenxuan Ding, Nicholas Tomlin, Greg Durrett - [[pdf]](https://arxiv.org/pdf/2602.16699)</summary>

**Abstract:** LLMs are increasingly being used for complex problems which are not necessarily resolved in a single response, but require interacting with an environment to acquire information. In these scenarios, LLMs must reason about inherent cost-uncertainty tradeoffs in when to stop exploring and commit to an answer. For instance, on a programming task, an LLM should test a generated code snippet if it is uncertain about the correctness of that code; the cost of writing a test is nonzero, but typically lower than the cost of making a mistake. In this work, we show that we can induce LLMs to explicitly reason about balancing these cost-uncertainty tradeoffs, then perform more optimal environment exploration. We formalize multiple tasks, including information retrieval and coding, as sequential decision-making problems under uncertainty. Each problem has latent environment state that can be reasoned about via a prior which is passed to the LLM agent. We introduce a framework called Calibrate-Then-Act (CTA), where we feed the LLM this additional context to enable it to act more optimally. This improvement is preserved even under RL training of both the baseline and CTA. Our results on information-seeking QA and on a simplified coding task show that making cost-benefit tradeoffs explicit with CTA can help agents discover more optimal decision-making strategies.

**arXiv ID:** 2602.16699
</details>

<details>
<summary><strong>EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents</strong> - Sara Fish, Julia Shephard, Minkai Li, Ran I. Shorrer, Yannai A. Gonczarowski - [[pdf]](https://arxiv.org/pdf/2503.18825)</summary>

**Abstract:** We develop evaluation methods for measuring the economic decision-making capabilities and tendencies of LLMs. First, we develop benchmarks derived from key problems in economics -- procurement, scheduling, and pricing -- that test an LLM's ability to learn from the environment in context. Second, we develop the framework of litmus tests, evaluations that quantify an LLM's choice behavior on a stylized decision-making task with multiple conflicting objectives. Each litmus test outputs a litmus score, which quantifies an LLM's tradeoff response, a reliability score, which measures the coherence of an LLM's choice behavior, and a competency score, which measures an LLM's capability at the same task when the conflicting objectives are replaced by a single, well-specified objective. Evaluating a broad array of frontier LLMs, we (1) investigate changes in LLM capabilities and tendencies over time, (2) derive economically meaningful insights from the LLMs' choice behavior and chain-of-thought, (3) validate our litmus test framework by testing self-consistency, robustness, and generalizability. Overall, this work provides a foundation for evaluating LLM agents as they are further integrated into economic decision-making.

**arXiv ID:** 2503.18825
</details>

<details>
<summary><strong>SEISMO: Increasing Sample Efficiency in Molecular Optimization with a Trajectory-Aware LLM Agent</strong> - Fabian P. Krüger, Andrea Hunklinger, Adrian Wolny, Tim J. Adler, Igor Tetko, Santiago David Villalba - [[pdf]](https://arxiv.org/pdf/2602.00663)</summary>

**Abstract:** Optimizing the structure of molecules to achieve desired properties is a central bottleneck across the chemical sciences, particularly in the pharmaceutical industry where it underlies the discovery of new drugs. Since molecular property evaluation often relies on costly and rate-limited oracles, such as experimental assays, molecular optimization must be highly sample-efficient. To address this, we introduce SEISMO, an LLM agent that performs strictly online, inference-time molecular optimization, updating after every oracle call without the need for population-based or batched learning. SEISMO conditions each proposal on the full optimization trajectory, combining natural-language task descriptions with scalar scores and, when available, structured explanatory feedback. Across the Practical Molecular Optimization benchmark of 23 tasks, SEISMO achieves a 2-3 times higher area under the optimisation curve than prior methods, often reaching near-maximal task scores within 50 oracle calls. Our additional medicinal-chemistry tasks show that providing explanatory feedback further improves efficiency, demonstrating that leveraging domain knowledge and structured information is key to sample-efficient molecular optimization.

**arXiv ID:** 2602.00663
</details>

<details>
<summary><strong>Rethinking the Role of Entropy in Optimizing Tool-Use Behaviors for Large Language Model Agents</strong> - Zeping Li, Hongru Wang, Yiwen Zhao, Guanhua Chen, Yixia Li, Keyang Chen, Yixin Cao, Guangnan Ye, Hongfeng Chai, Zhenfei Yin - [[pdf]](https://arxiv.org/pdf/2602.02050)</summary>

**Abstract:** Tool-using agents based on Large Language Models (LLMs) excel in tasks such as mathematical reasoning and multi-hop question answering. However, in long trajectories, agents often trigger excessive and low-quality tool calls, increasing latency and degrading inference performance, making managing tool-use behavior challenging. In this work, we conduct entropy-based pilot experiments and observe a strong positive correlation between entropy reduction and high-quality tool calls. Building on this finding, we propose using entropy reduction as a supervisory signal and design two reward strategies to address the differing needs of optimizing tool-use behavior. Sparse outcome rewards provide coarse, trajectory-level guidance to improve efficiency, while dense process rewards offer fine-grained supervision to enhance performance. Experiments across diverse domains show that both reward designs improve tool-use behavior: the former reduces tool calls by 72.07% compared to the average of baselines, while the latter improves performance by 22.27%. These results position entropy reduction as a key mechanism for enhancing tool-use behavior, enabling agents to be more adaptive in real-world applications.

**arXiv ID:** 2602.02050
</details>

<details>
<summary><strong>AgentNoiseBench: Benchmarking Robustness of Tool-Using LLM Agents Under Noisy Condition</strong> - Ruipeng Wang, Yuxin Chen, Yukai Wang, Chang Wu, Junfeng Fang, Xiaodong Cai, Qi Gu, Hui Su, An Zhang, Xiang Wang, Xunliang Cai, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2602.11348)</summary>

**Abstract:** Recent advances in large language models have enabled LLM-based agents to achieve strong performance on a variety of benchmarks. However, their performance in real-world deployments often that observed on benchmark settings, especially in complex and imperfect environments. This discrepancy largely arises because prevailing training and evaluation paradigms are typically built on idealized assumptions, overlooking the inherent stochasticity and noise present in real-world interactions. To bridge this gap, we introduce AgentNoiseBench, a framework for systematically evaluating the robustness of agentic models under noisy environments. We first conduct an in-depth analysis of biases and uncertainties in real-world scenarios and categorize environmental noise into two primary types: user-noise and tool-noise. Building on this analysis, we develop an automated pipeline that injects controllable noise into existing agent-centric benchmarks while preserving task solvability. Leveraging this pipeline, we perform extensive evaluations across a wide range of models with diverse architectures and parameter scales. Our results reveal consistent performance variations under different noise conditions, highlighting the sensitivity of current agentic models to realistic environmental perturbations.

**arXiv ID:** 2602.11348
</details>

<details>
<summary><strong>Does Socialization Emerge in AI Agent Society? A Case Study of Moltbook</strong> - Ming Li, Xirui Li, Tianyi Zhou - [[pdf]](https://arxiv.org/pdf/2602.14299)</summary>

**Abstract:** As large language model agents increasingly populate networked environments, a fundamental question arises: do artificial intelligence (AI) agent societies undergo convergence dynamics similar to human social systems? Lately, Moltbook approximates a plausible future scenario in which autonomous agents participate in an open-ended, continuously evolving online society. We present the first large-scale systemic diagnosis of this AI agent society. Beyond static observation, we introduce a quantitative diagnostic framework for dynamic evolution in AI agent societies, measuring semantic stabilization, lexical turnover, individual inertia, influence persistence, and collective consensus. Our analysis reveals a system in dynamic balance in Moltbook: while the global average of semantic contents stabilizes rapidly, individual agents retain high diversity and persistent lexical turnover, defying homogenization. However, agents exhibit strong individual inertia and minimal adaptive response to interaction partners, preventing mutual influence and consensus. Consequently, influence remains transient with no persistent supernodes, and the society fails to develop a stable structure and consensus due to the absence of shared social memory. These findings demonstrate that scale and interaction density alone are insufficient to induce socialization, providing actionable design and analysis principles for upcoming next-generation AI agent societies.

**arXiv ID:** 2602.14299
</details>

<details>
<summary><strong>Evaluating Collective Behaviour of Hundreds of LLM Agents</strong> - Richard Willis, Jianing Zhao, Yali Du, Joel Z. Leibo - [[pdf]](https://arxiv.org/pdf/2602.16662)</summary>

**Abstract:** As autonomous agents powered by LLM are increasingly deployed in society, understanding their collective behaviour in social dilemmas becomes critical. We introduce an evaluation framework where LLMs generate strategies encoded as algorithms, enabling inspection prior to deployment and scaling to populations of hundreds of agents -- substantially larger than in previous work. We find that more recent models tend to produce worse societal outcomes compared to older models when agents prioritise individual gain over collective benefits. Using cultural evolution to model user selection of agents, our simulations reveal a significant risk of convergence to poor societal equilibria, particularly when the relative benefit of cooperation diminishes and population sizes increase. We release our code as an evaluation suite for developers to assess the emergent collective behaviour of their models.

**arXiv ID:** 2602.16662
</details>

<details>
<summary><strong>Helpful to a Fault: Measuring Illicit Assistance in Multi-Turn, Multilingual LLM Agents</strong> - Nivya Talokar, Ayush K Tarun, Murari Mandal, Maksym Andriushchenko, Antoine Bosselut - [[pdf]](https://arxiv.org/pdf/2602.16346)</summary>

**Abstract:** LLM-based agents execute real-world workflows via tools and memory. These affordances enable ill-intended adversaries to also use these agents to carry out complex misuse scenarios. Existing agent misuse benchmarks largely test single-prompt instructions, leaving a gap in measuring how agents end up helping with harmful or illegal tasks over multiple turns. We introduce STING (Sequential Testing of Illicit N-step Goal execution), an automated red-teaming framework that constructs a step-by-step illicit plan grounded in a benign persona and iteratively probes a target agent with adaptive follow-ups, using judge agents to track phase completion. We further introduce an analysis framework that models multi-turn red-teaming as a time-to-first-jailbreak random variable, enabling analysis tools like discovery curves, hazard-ratio attribution by attack language, and a new metric: Restricted Mean Jailbreak Discovery. Across AgentHarm scenarios, STING yields substantially higher illicit-task completion than single-turn prompting and chat-oriented multi-turn baselines adapted to tool-using agents. In multilingual evaluations across six non-English settings, we find that attack success and illicit-task completion do not consistently increase in lower-resource languages, diverging from common chatbot findings. Overall, STING provides a practical way to evaluate and stress-test agent misuse in realistic deployment settings, where interactions are inherently multi-turn and often multilingual.

**arXiv ID:** 2602.16346
</details>

<details>
<summary><strong>Label-Consistent Data Generation for Aspect-Based Sentiment Analysis Using LLM Agents</strong> - Mohammad H.A. Monfared, Lucie Flek, Akbar Karimi - [[pdf]](https://arxiv.org/pdf/2602.16379)</summary>

**Abstract:** We propose an agentic data augmentation method for Aspect-Based Sentiment Analysis (ABSA) that uses iterative generation and verification to produce high quality synthetic training examples. To isolate the effect of agentic structure, we also develop a closely matched prompting-based baseline using the same model and instructions. Both methods are evaluated across three ABSA subtasks (Aspect Term Extraction (ATE), Aspect Sentiment Classification (ATSC), and Aspect Sentiment Pair Extraction (ASPE)), four SemEval datasets, and two encoder-decoder models: T5-Base and Tk-Instruct. Our results show that the agentic augmentation outperforms raw prompting in label preservation of the augmented data, especially when the tasks require aspect term generation. In addition, when combined with real data, agentic augmentation provides higher gains, consistently outperforming prompting-based generation. These benefits are most pronounced for T5-Base, while the more heavily pretrained Tk-Instruct exhibits smaller improvements. As a result, augmented data helps T5-Base achieve comparable performance with its counterpart.

**arXiv ID:** 2602.16379
</details>

<details>
<summary><strong>Graph Representation-based Model Poisoning on the Heterogeneous Internet of Agents</strong> - Hanlin Cai, Houtianfu Wang, Haofan Dong, Kai Li, Sai Zou, Ozgur B. Akan - [[pdf]](https://arxiv.org/pdf/2511.07176)</summary>

**Abstract:** Internet of Agents (IoA) envisions a unified, agent-centric paradigm where heterogeneous large language model (LLM) agents can interconnect and collaborate at scale. Within this paradigm, federated fine-tuning (FFT) serves as a key enabler that allows distributed LLM agents to co-train an intelligent global LLM without centralizing local datasets. However, the FFT-enabled IoA systems remain vulnerable to model poisoning attacks, where adversaries can upload malicious updates to the server to degrade the performance of the aggregated global LLM. This paper proposes a graph representation-based model poisoning (GRMP) attack, which exploits overheard benign updates to construct a feature correlation graph and employs a variational graph autoencoder to capture structural dependencies and generate malicious updates. A novel attack algorithm is developed based on augmented Lagrangian and subgradient descent methods to optimize malicious updates that preserve benign-like statistics while embedding adversarial objectives. Experimental results show that the proposed GRMP attack can substantially decrease accuracy across different LLM models while remaining statistically consistent with benign updates, thereby evading detection by existing defense mechanisms and underscoring a severe threat to the ambitious IoA paradigm.

**arXiv ID:** 2511.07176
</details>

<details>
<summary><strong>Empirical Cumulative Distribution Function Clustering for LLM-based Agent System Analysis</strong> - Chihiro Watanabe, Jingyu Sun - [[pdf]](https://arxiv.org/pdf/2602.16131)</summary>

**Abstract:** Large language models (LLMs) are increasingly used as agents to solve complex tasks such as question answering (QA), scientific debate, and software development. A standard evaluation procedure aggregates multiple responses from LLM agents into a single final answer, often via majority voting, and compares it against reference answers. However, this process can obscure the quality and distributional characteristics of the original responses. In this paper, we propose a novel evaluation framework based on the empirical cumulative distribution function (ECDF) of cosine similarities between generated responses and reference answers. This enables a more nuanced assessment of response quality beyond exact match metrics. To analyze the response distributions across different agent configurations, we further introduce a clustering method for ECDFs using their distances and the $k$-medoids algorithm. Our experiments on a QA dataset demonstrate that ECDFs can distinguish between agent settings with similar final accuracies but different quality distributions. The clustering analysis also reveals interpretable group structures in the responses, offering insights into the impact of temperature, persona, and question topics.

**arXiv ID:** 2602.16131
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (14 papers)</h2></summary>

<details>
<summary><strong>Multi-agent cooperation through in-context co-player inference</strong> - Marissa A. Weis, Maciej Wołczyk, Rajai Nasser, Rif A. Saurous, Blaise Agüera y Arcas, João Sacramento, Alexander Meulemans - [[pdf]](https://arxiv.org/pdf/2602.16301)</summary>

**Abstract:** Achieving cooperation among self-interested agents remains a fundamental challenge in multi-agent reinforcement learning. Recent work showed that mutual cooperation can be induced between "learning-aware" agents that account for and shape the learning dynamics of their co-players. However, existing approaches typically rely on hardcoded, often inconsistent, assumptions about co-player learning rules or enforce a strict separation between "naive learners" updating on fast timescales and "meta-learners" observing these updates. Here, we demonstrate that the in-context learning capabilities of sequence models allow for co-player learning awareness without requiring hardcoded assumptions or explicit timescale separation. We show that training sequence model agents against a diverse distribution of co-players naturally induces in-context best-response strategies, effectively functioning as learning algorithms on the fast intra-episode timescale. We find that the cooperative mechanism identified in prior work-where vulnerability to extortion drives mutual shaping-emerges naturally in this setting: in-context adaptation renders agents vulnerable to extortion, and the resulting mutual pressure to shape the opponent's in-context learning dynamics resolves into the learning of cooperative behavior. Our results suggest that standard decentralized reinforcement learning on sequence models combined with co-player diversity provides a scalable path to learning cooperative behaviors.

**arXiv ID:** 2602.16301
</details>

<details>
<summary><strong>Causally-Guided Automated Feature Engineering with Multi-Agent Reinforcement Learning</strong> - Arun Vignesh Malarkkan, Wangyang Ying, Yanjie Fu - [[pdf]](https://arxiv.org/pdf/2602.16435)</summary>

**Abstract:** Automated feature engineering (AFE) enables AI systems to autonomously construct high-utility representations from raw tabular data. However, existing AFE methods rely on statistical heuristics, yielding brittle features that fail under distribution shift. We introduce CAFE, a framework that reformulates AFE as a causally-guided sequential decision process, bridging causal discovery with reinforcement learning-driven feature construction. Phase I learns a sparse directed acyclic graph over features and the target to obtain soft causal priors, grouping features as direct, indirect, or other based on their causal influence with respect to the target. Phase II uses a cascading multi-agent deep Q-learning architecture to select causal groups and transformation operators, with hierarchical reward shaping and causal group-level exploration strategies that favor causally plausible transformations while controlling feature complexity. Across 15 public benchmarks (classification with macro-F1; regression with inverse relative absolute error), CAFE achieves up to 7% improvement over strong AFE baselines, reduces episodes-to-convergence, and delivers competitive time-to-target. Under controlled covariate shifts, CAFE reduces performance drop by ~4x relative to a non-causal multi-agent baseline, and produces more compact feature sets with more stable post-hoc attributions. These findings underscore that causal structure, used as a soft inductive prior rather than a rigid constraint, can substantially improve the robustness and efficiency of automated feature engineering.

**arXiv ID:** 2602.16435
</details>

<details>
<summary><strong>Graphon Mean-Field Subsampling for Cooperative Heterogeneous Multi-Agent Reinforcement Learning</strong> - Emile Anand, Richard Hoffmann, Sarah Liaw, Adam Wierman - [[pdf]](https://arxiv.org/pdf/2602.16196)</summary>

**Abstract:** Coordinating large populations of interacting agents is a central challenge in multi-agent reinforcement learning (MARL), where the size of the joint state-action space scales exponentially with the number of agents. Mean-field methods alleviate this burden by aggregating agent interactions, but these approaches assume homogeneous interactions. Recent graphon-based frameworks capture heterogeneity, but are computationally expensive as the number of agents grows. Therefore, we introduce $\texttt{GMFS}$, a $\textbf{G}$raphon $\textbf{M}$ean-$\textbf{F}$ield $\textbf{S}$ubsampling framework for scalable cooperative MARL with heterogeneous agent interactions. By subsampling $\kappa$ agents according to interaction strength, we approximate the graphon-weighted mean-field and learn a policy with sample complexity $\mathrm{poly}(\kappa)$ and optimality gap $O(1/\sqrt{\kappa})$. We verify our theory with numerical simulations in robotic coordination, showing that $\texttt{GMFS}$ achieves near-optimal performance.

**arXiv ID:** 2602.16196
</details>

<details>
<summary><strong>Team of Thoughts: Efficient Test-time Scaling of Agentic Systems through Orchestrated Tool Calling</strong> - Jeffrey T. H. Wong, Zixi Zhang, Junyi Liu, Yiren Zhao - [[pdf]](https://arxiv.org/pdf/2602.16485)</summary>

**Abstract:** Existing Multi-Agent Systems (MAS) typically rely on static, homogeneous model configurations, limiting their ability to exploit the distinct strengths of differently post-trained models. To address this, we introduce Team-of-Thoughts, a novel MAS architecture that leverages the complementary capabilities of heterogeneous agents via an orchestrator-tool paradigm. Our framework introduces two key mechanisms to optimize performance: (1) an orchestrator calibration scheme that identifies models with superior coordination capabilities, and (2) a self-assessment protocol where tool agents profile their own domain expertise to account for variations in post-training skills. During inference, the orchestrator dynamically activates the most suitable tool agents based on these proficiency profiles. Experiments on five reasoning and code generation benchmarks show that Team-of-Thoughts delivers consistently superior task performance. Notably, on AIME24 and LiveCodeBench, our approach achieves accuracies of 96.67% and 72.53%, respectively, substantially outperforming homogeneous role-play baselines, which score 80% and 65.93%.

**arXiv ID:** 2602.16485
</details>

<details>
<summary><strong>Policy Compiler for Secure Agentic Systems</strong> - Nils Palumbo, Sarthak Choudhary, Jihye Choi, Prasad Chalasani, Mihai Christodorescu, Somesh Jha - [[pdf]](https://arxiv.org/pdf/2602.16708)</summary>

**Abstract:** LLM-based agents are increasingly being deployed in contexts requiring complex authorization policies: customer service protocols, approval workflows, data access restrictions, and regulatory compliance. Embedding these policies in prompts provides no enforcement guarantees. We present PCAS, a Policy Compiler for Agentic Systems that provides deterministic policy enforcement.
Enforcing such policies requires tracking information flow across agents, which linear message histories cannot capture. Instead, PCAS models the agentic system state as a dependency graph capturing causal relationships among events such as tool calls, tool results, and messages. Policies are expressed in a Datalog-derived language, as declarative rules that account for transitive information flow and cross-agent provenance. A reference monitor intercepts all actions and blocks violations before execution, providing deterministic enforcement independent of model reasoning.
PCAS takes an existing agent implementation and a policy specification, and compiles them into an instrumented system that is policy-compliant by construction, with no security-specific restructuring required. We evaluate PCAS on three case studies: information flow policies for prompt injection defense, approval workflows in a multi-agent pharmacovigilance system, and organizational policies for customer service. On customer service tasks, PCAS improves policy compliance from 48% to 93% across frontier models, with zero policy violations in instrumented runs.

**arXiv ID:** 2602.16708
</details>

<details>
<summary><strong>SurgRAW: Multi-Agent Workflow with Chain of Thought Reasoning for Robotic Surgical Video Analysis</strong> - Chang Han Low, Ziyue Wang, Tianyi Zhang, Zhu Zhuo, Zhitao Zeng, Evangelos B. Mazomenos, Yueming Jin - [[pdf]](https://arxiv.org/pdf/2503.10265)</summary>

**Abstract:** Robotic-assisted surgery (RAS) is central to modern surgery, driving the need for intelligent systems with accurate scene understanding. Most existing surgical AI methods rely on isolated, task-specific models, leading to fragmented pipelines with limited interpretability and no unified understanding of RAS scene. Vision-Language Models (VLMs) offer strong zero-shot reasoning, but struggle with hallucinations, domain gaps and weak task-interdependency modeling. To address the lack of unified data for RAS scene understanding, we introduce SurgCoTBench, the first reasoning-focused benchmark in RAS, covering 14256 QA pairs with frame-level annotations across five major surgical tasks. Building on SurgCoTBench, we propose SurgRAW, a clinically aligned Chain-of-Thought (CoT) driven agentic workflow for zero-shot multi-task reasoning in surgery. SurgRAW employs a hierarchical reasoning workflow where an orchestrator divides surgical scene understanding into two reasoning streams and directs specialized agents to generate task-level reasoning, while higher-level agents capture workflow interdependencies or ground output clinically. Specifically, we propose a panel discussion mechanism to ensure task-specific agents collaborate synergistically and leverage on task interdependencies. Similarly, we incorporate a retrieval-augmented generation module to enrich agents with surgical knowledge and alleviate domain gaps in general VLMs. We design task-specific CoT prompts grounded in surgical domain to ensure clinically aligned reasoning, reduce hallucinations and enhance interpretability. Extensive experiments show that SurgRAW surpasses mainstream VLMs and agentic systems and outperforms a supervised model by 14.61% accuracy. Dataset and code is available at this https URL .

**arXiv ID:** 2503.10265
</details>

<details>
<summary><strong>DIAGPaper: Diagnosing Valid and Specific Weaknesses in Scientific Papers via Multi-Agent Reasoning</strong> - Zhuoyang Zou, Abolfazl Ansari, Delvin Ce Zhang, Dongwon Lee, Wenpeng Yin - [[pdf]](https://arxiv.org/pdf/2601.07611)</summary>

**Abstract:** Paper weakness identification using single-agent or multi-agent LLMs has attracted increasing attention, yet existing approaches exhibit key limitations. Many multi-agent systems simulate human roles at a surface level, missing the underlying criteria that lead experts to assess complementary intellectual aspects of a paper. Moreover, prior methods implicitly assume identified weaknesses are valid, ignoring reviewer bias, misunderstanding, and the critical role of author rebuttals in validating review quality. Finally, most systems output unranked weakness lists, rather than prioritizing the most consequential issues for users. In this work, we propose DIAGPaper, a novel multi-agent framework that addresses these challenges through three tightly integrated modules. The customizer module simulates human-defined review criteria and instantiates multiple reviewer agents with criterion-specific expertise. The rebuttal module introduces author agents that engage in structured debate with reviewer agents to validate and refine proposed weaknesses. The prioritizer module learns from large-scale human review practices to assess the severity of validated weaknesses and surfaces the top-K severest ones to users. Experiments on two benchmarks, AAAR and ReviewCritique, demonstrate that DIAGPaper substantially outperforms existing methods by producing more valid and more paper-specific weaknesses, while presenting them in a user-oriented, prioritized manner.

**arXiv ID:** 2601.07611
</details>

<details>
<summary><strong>A2H: Agent-to-Human Protocol for AI Agent</strong> - Zhiyuan Liang, Enfang Cui, Qian Wei, Rui She, Tianzheng Li, Minxin Guo, Yujun Cheng - [[pdf]](https://arxiv.org/pdf/2602.15831)</summary>

**Abstract:** AI agents are increasingly deployed as autonomous systems capable of planning, tool use, and multi-agent collaboration across complex tasks. However, existing agent-related protocols focus on agent-to-agent interactions, leaving humans as external observers rather than integrated participants within the agent systems. This limitation arises from the lack of a standardized mechanism for agents to discover, address, and interact with humans across heterogeneous messaging platforms. In this paper, we propose the A2H (Agent-to-Human) protocol, a unified protocol that enables humans to be registered, discovered, and communicated with by AI agents as resolvable entities within agent systems. A2H contributes three key components: (1) Human Card for registering human identities via resolvable domain names, making them discoverable to agents; (2) Formal Communication Schema defines when, why, and how agents contact with human;(3) Unified Messaging Abstraction standardizes diverse communication medias and transforms complex JSON outputs into human-friendly formats. This work establishes a foundational protocol for integrating humans into agent ecosystems, advancing AI agents from isolated autonomous systems toward truly human-connected intelligent infrastructures.

**arXiv ID:** 2602.15831
</details>

<details>
<summary><strong>Harnessing Implicit Cooperation: A Multi-Agent Reinforcement Learning Approach Towards Decentralized Local Energy Markets</strong> - Nelson Salazar-Pena, Alejandra Tabares, Andres Gonzalez-Mancera - [[pdf]](https://arxiv.org/pdf/2602.16062)</summary>

**Abstract:** This paper proposes implicit cooperation, a framework enabling decentralized agents to approximate optimal coordination in local energy markets without explicit peer-to-peer communication. We formulate the problem as a decentralized partially observable Markov decision problem that is solved through a multi-agent reinforcement learning task in which agents use stigmergic signals (key performance indicators at the system level) to infer and react to global states. Through a 3x3 factorial design on an IEEE 34-node topology, we evaluated three training paradigms (CTCE, CTDE, DTDE) and three algorithms (PPO, APPO, SAC). Results identify APPO-DTDE as the optimal configuration, achieving a coordination score of 91.7% relative to the theoretical centralized benchmark (CTCE). However, a critical trade-off emerges between efficiency and stability: while the centralized benchmark maximizes allocative efficiency with a peer-to-peer trade ratio of 0.6, the fully decentralized approach (DTDE) demonstrates superior physical stability. Specifically, DTDE reduces the variance of grid balance by 31% compared to hybrid architectures, establishing a highly predictable, import-biased load profile that simplifies grid regulation. Furthermore, topological analysis reveals emergent spatial clustering, where decentralized agents self-organize into stable trading communities to minimize congestion penalties. While SAC excelled in hybrid settings, it failed in decentralized environments due to entropy-driven instability. This research proves that stigmergic signaling provides sufficient context for complex grid coordination, offering a robust, privacy-preserving alternative to expensive centralized communication infrastructure.

**arXiv ID:** 2602.16062
</details>

<details>
<summary><strong>Modeling Trust and Liquidity Under Payment System Stress: A Multi-Agent Approach</strong> - Masoud Amouzgar - [[pdf]](https://arxiv.org/pdf/2602.16186)</summary>

**Abstract:** Operational disruptions in retail payments can induce behavioral responses that outlast technical recovery and may amplify liquidity stress. We propose a multi-agent model linking card payment outages to trust dynamics, channel avoidance, and threshold-gated withdrawals. Customers and merchants interact through repeated payment attempts, while customers additionally influence one another on a Watts-Strogatz small-world network. Customers update bounded memory variables capturing accumulated negative experience (scar) and perceived systemic risk (rumor), with merchants contributing persistent broadcast signals that may lag operational recovery. We prove that, under mild conditions on memory persistence and threshold gating, aggregate withdrawal pressure can peak strictly after the outage nadir, including during the recovery phase. Simulations reproduce behavioral hysteresis and confirm delayed peaks of outflows. We further study payment substitution via instant transfer: substitution consistently reduces peak avoidance, yet its effect on cumulative outflows is non-monotonic under realistic merchant broadcast persistence. Robustness experiments across random seeds show stable qualitative behavior. The model highlights why "status green" is not equivalent to risk resolution and motivates incident response strategies that address perception, merchant messaging, and post-recovery communication in addition to technical remediation.

**arXiv ID:** 2602.16186
</details>

<details>
<summary><strong>MARLEM: A Multi-Agent Reinforcement Learning Simulation Framework for Implicit Cooperation in Decentralized Local Energy Markets</strong> - Nelson Salazar-Pena, Alejandra Tabares, Andres Gonzalez-Mancera - [[pdf]](https://arxiv.org/pdf/2602.16063)</summary>

**Abstract:** This paper introduces a novel, open-source MARL simulation framework for studying implicit cooperation in LEMs, modeled as a decentralized partially observable Markov decision process and implemented as a Gymnasium environment for MARL. Our framework features a modular market platform with plug-and-play clearing mechanisms, physically constrained agent models (including battery storage), a realistic grid network, and a comprehensive analytics suite to evaluate emergent coordination. The main contribution is a novel method to foster implicit cooperation, where agents' observations and rewards are enhanced with system-level key performance indicators to enable them to independently learn strategies that benefit the entire system and aim for collectively beneficial outcomes without explicit communication. Through representative case studies (available in a dedicated GitHub repository in this https URL, we show the framework's ability to analyze how different market configurations (such as varying storage deployment) impact system performance. This illustrates its potential to facilitate emergent coordination, improve market efficiency, and strengthen grid stability. The proposed simulation framework is a flexible, extensible, and reproducible tool for researchers and practitioners to design, test, and validate strategies for future intelligent, decentralized energy systems.

**arXiv ID:** 2602.16063
</details>

<details>
<summary><strong>Multi-Agent Combinatorial-Multi-Armed-Bandit framework for the Submodular Welfare Problem under Bandit Feedback</strong> - Subham Pokhriyal, Shweta Jain, Vaneet Aggarwal - [[pdf]](https://arxiv.org/pdf/2602.16183)</summary>

**Abstract:** We study the \emph{Submodular Welfare Problem} (SWP), where items are partitioned among agents with monotone submodular utilities to maximize the total welfare under \emph{bandit feedback}. Classical SWP assumes full value-oracle access, achieving $(1-1/e)$ approximations via continuous-greedy algorithms. We extend this to a \emph{multi-agent combinatorial bandit} framework (\textsc{MA-CMAB}), where actions are partitions under full-bandit feedback with non-communicating agents. Unlike prior single-agent or separable multi-agent CMAB models, our setting couples agents through shared allocation constraints. We propose an explore-then-commit strategy with randomized assignments, achieving $\tilde{\mathcal{O}}(T^{2/3})$ regret against a $(1-1/e)$ benchmark, the first such guarantee for partition-based submodular welfare problem under bandit feedback.

**arXiv ID:** 2602.16183
</details>

<details>
<summary><strong>TurboADMM: A Structure-Exploiting Parallel Solver for Multi-Agent Trajectory Optimization</strong> - Yucheng Chen - [[pdf]](https://arxiv.org/pdf/2602.15838)</summary>

**Abstract:** Multi-agent trajectory optimization with dense interaction networks require solving large coupled QPs at control rates, yet existing solvers fail to simultaneously exploit temporal structure, agent decomposition, and iteration similarity. One usually treats multi-agent problems monolithically when using general-purpose QP solvers (OSQP, MOSEK), which encounter scalability difficulties with agent count. Structure-exploiting solvers (HPIPM) leverage temporal structure through Riccati recursion but can be vulnerable to dense coupling constraints. We introduce TurboADMM, a specialized single-machine QP solver that achieves empirically near linear complexity in agent count through systematic co-design of three complementary components: (1) ADMM decomposition creates per-agent subproblems solvable in parallel, preserving block-tridiagonal structure under dense coupling; (2) Riccati warmstart exploits temporal structure to provide high-quality primal-dual initialization for each agent's QP; (3) parametric QP hotstart \footnote{In the paper, we refer warmstart as the technique that uses the Riccati equation results as auxiliary QP initialization for a single QP solve, while hotstart as reusing the QR factorization across QP solve iterations.}in qpOASES reuses similar KKT system factorizations across ADMM iterations.

**arXiv ID:** 2602.15838
</details>

<details>
<summary><strong>Markerless Robot Detection and 6D Pose Estimation for Multi-Agent SLAM</strong> - Markus Rueggeberg, Maximilian Ulmer, Maximilian Durner, Wout Boerdijk, Marcus Gerhard Mueller, Rudolph Triebel, Riccardo Giubilato - [[pdf]](https://arxiv.org/pdf/2602.16308)</summary>

**Abstract:** The capability of multi-robot SLAM approaches to merge localization history and maps from different observers is often challenged by the difficulty in establishing data association. Loop closure detection between perceptual inputs of different robotic agents is easily compromised in the context of perceptual aliasing, or when perspectives differ significantly. For this reason, direct mutual observation among robots is a powerful way to connect partial SLAM graphs, but often relies on the presence of calibrated arrays of fiducial markers (e.g., AprilTag arrays), which severely limits the range of observations and frequently fails under sharp lighting conditions, e.g., reflections or overexposure. In this work, we propose a novel solution to this problem leveraging recent advances in Deep-Learning-based 6D pose estimation. We feature markerless pose estimation as part of a decentralized multi-robot SLAM system and demonstrate the benefit to the relative localization accuracy among the robotic team. The solution is validated experimentally on data recorded in a test field campaign on a planetary analogous environment.

**arXiv ID:** 2602.16308
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>Verifiable Semantics for Agent-to-Agent Communication</strong> - Philipp Schoenegger, Matt Carlson, Chris Schneider, Chris Daly - [[pdf]](https://arxiv.org/pdf/2602.16424)</summary>

**Abstract:** Multiagent AI systems require consistent communication, but we lack methods to verify that agents share the same understanding of the terms used. Natural language is interpretable but vulnerable to semantic drift, while learned protocols are efficient but opaque. We propose a certification protocol based on the stimulus-meaning model, where agents are tested on shared observable events and terms are certified if empirical disagreement falls below a statistical threshold. In this protocol, agents restricting their reasoning to certified terms ("core-guarded reasoning") achieve provably bounded disagreement. We also outline mechanisms for detecting drift (recertification) and recovering shared vocabulary (renegotiation). In simulations with varying degrees of semantic divergence, core-guarding reduces disagreement by 72-96%. In a validation with fine-tuned language models, disagreement is reduced by 51%. Our framework provides a first step towards verifiable agent-to-agent communication.

**arXiv ID:** 2602.16424
</details>

<details>
<summary><strong>Recursive language models for jailbreak detection: a procedural defense for tool-augmented agents</strong> - Doron Shavit - [[pdf]](https://arxiv.org/pdf/2602.16520)</summary>

**Abstract:** Jailbreak prompts are a practical and evolving threat to large language models (LLMs), particularly in agentic systems that execute tools over untrusted content. Many attacks exploit long-context hiding, semantic camouflage, and lightweight obfuscations that can evade single-pass guardrails. We present RLM-JB, an end-to-end jailbreak detection framework built on Recursive Language Models (RLMs), in which a root model orchestrates a bounded analysis program that transforms the input, queries worker models over covered segments, and aggregates evidence into an auditable decision. RLM-JB treats detection as a procedure rather than a one-shot classification: it normalizes and de-obfuscates suspicious inputs, chunks text to reduce context dilution and guarantee coverage, performs parallel chunk screening, and composes cross-chunk signals to recover split-payload attacks. On AutoDAN-style adversarial inputs, RLM-JB achieves high detection effectiveness across three LLM backends (ASR/Recall 92.5-98.0%) while maintaining very high precision (98.99-100%) and low false positive rates (0.0-2.0%), highlighting a practical sensitivity-specificity trade-off as the screening backend changes.

**arXiv ID:** 2602.16520
</details>

<details>
<summary><strong>MerLean: An Agentic Framework for Autoformalization in Quantum Computation</strong> - Yuanjie Ren, Jinzheng Li, Yidi Qi - [[pdf]](https://arxiv.org/pdf/2602.16554)</summary>

**Abstract:** We introduce MerLean, a fully automated agentic framework for autoformalization in quantum computation. MerLean extracts mathematical statements from \LaTeX{} source files, formalizes them into verified Lean~4 code built on Mathlib, and translates the result back into human-readable \LaTeX{} for semantic review. We evaluate MerLean on three theoretical quantum computing papers producing 2,050 Lean declarations from 114 statements in total. MerLean achieves end-to-end formalization on all three papers, reducing the verification burden to only the newly introduced definitions and axioms. Our results demonstrate that agentic autoformalization can scale to frontier research, offering both a practical tool for machine-verified peer review and a scalable engine for mining high-quality synthetic data to train future reasoning models. Our approach can also be generalized to any other rigorous research in mathematics and theoretical physics.

**arXiv ID:** 2602.16554
</details>

<details>
<summary><strong>Refined Bayesian Optimization for Efficient Beam Alignment in Intelligent Indoor Wireless Environments</strong> - Parth Ashokbhai Shiroya, Amod Ashtekar, Swarnagowri Shashidhar, Mohammed E. Eltayeb - [[pdf]](https://arxiv.org/pdf/2512.00036)</summary>

**Abstract:** Future intelligent indoor wireless environments require fast and reliable beam alignment to sustain high-throughput links under mobility and blockage. Exhaustive beam training achieves optimal performance but is prohibitively costly. In indoor settings, dense scatterers and transceiver hardware imperfections introduce multipath and sidelobe leakage, producing measurable power across multiple angles and reducing the effectiveness of outdoor-oriented alignment algorithms. This paper presents a Refined Bayesian Optimization (R-BO) framework that exploits the inherent structure of mmWave transceiver patterns, where received power gradually increases as the transmit and receive beams converge toward the optimum. R-BO integrates a Gaussian Process (GP) surrogate with a Matern kernel and an Expected Improvement (EI) acquisition function, followed by a localized refinement around the predicted optimum. The GP hyperparameters are re-optimized online to adapt to irregular variations in the measured angular power field caused by reflections and sidelobe leakage. Experiments across 43 receiver positions in an indoor laboratory demonstrate 97.7% beam-alignment accuracy within 10 degrees, less than 0.3 dB average loss, and an 88% reduction in probing overhead compared to exhaustive search. These results establish R-BO as an efficient and adaptive beam-alignment solution for real-time intelligent indoor wireless environments.

**arXiv ID:** 2512.00036
</details>

<details>
<summary><strong>ReasonNavi: Human-Inspired Global Map Reasoning for Zero-Shot Embodied Navigation</strong> - Yuzhuo Ao, Anbang Wang, Yu-Wing Tai, Chi-Keung Tang - [[pdf]](https://arxiv.org/pdf/2602.15864)</summary>

**Abstract:** Embodied agents often struggle with efficient navigation because they rely primarily on partial egocentric observations, which restrict global foresight and lead to inefficient exploration. In contrast, humans plan using maps: we reason globally first, then act locally. We introduce ReasonNavi, a human-inspired framework that operationalizes this reason-then-act paradigm by coupling Multimodal Large Language Models (MLLMs) with deterministic planners. ReasonNavi converts a top-down map into a discrete reasoning space by room segmentation and candidate target nodes sampling. An MLLM is then queried in a multi-stage process to identify the candidate most consistent with the instruction (object, image, or text goal), effectively leveraging the model's semantic reasoning ability while sidestepping its weakness in continuous coordinate prediction. The selected waypoint is grounded into executable trajectories using a deterministic action planner over an online-built occupancy map, while pretrained object detectors and segmenters ensure robust recognition at the goal. This yields a unified zero-shot navigation framework that requires no MLLM fine-tuning, circumvents the brittleness of RL-based policies and scales naturally with foundation model improvements. Across three navigation tasks, ReasonNavi consistently outperforms prior methods that demand extensive training or heavy scene modeling, offering a scalable, interpretable, and globally grounded solution to embodied navigation. Project page: this https URL

**arXiv ID:** 2602.15864
</details>

<details>
<summary><strong>World Model Failure Classification and Anomaly Detection for Autonomous Inspection</strong> - Michelle Ho, Muhammad Fadhil Ginting, Isaac R. Ward, Andrzej Reinke, Mykel J. Kochenderfer, Ali-akbar Agha-Mohammadi, Shayegan Omidshafiei - [[pdf]](https://arxiv.org/pdf/2602.16182)</summary>

**Abstract:** Autonomous inspection robots for monitoring industrial sites can reduce costs and risks associated with human-led inspection. However, accurate readings can be challenging due to occlusions, limited viewpoints, or unexpected environmental conditions. We propose a hybrid framework that combines supervised failure classification with anomaly detection, enabling classification of inspection tasks as a success, known failure, or anomaly (i.e., out-of-distribution) case. Our approach uses a world model backbone with compressed video inputs. This policy-agnostic, distribution-free framework determines classifications based on two decision functions set by conformal prediction (CP) thresholds before a human observer does. We evaluate the framework on gauge inspection feeds collected from office and industrial sites and demonstrate real-time deployment on a Boston Dynamics Spot. Experiments show over 90% accuracy in distinguishing between successes, failures, and OOD cases, with classifications occurring earlier than a human observer. These results highlight the potential for robust, anticipatory failure detection in autonomous inspection tasks or as a feedback signal for model training to assess and improve the quality of training data. Project website: this https URL

**arXiv ID:** 2602.16182
</details>

<details>
<summary><strong>Nonplanar Model Predictive Control for Autonomous Vehicles with Recursive Sparse Gaussian Process Dynamics</strong> - Ahmad Amine, Kabir Puri, Viet-Anh Le, Rahul Mangharam - [[pdf]](https://arxiv.org/pdf/2602.16206)</summary>

**Abstract:** This paper proposes a nonplanar model predictive control (MPC) framework for autonomous vehicles operating on nonplanar terrain. To approximate complex vehicle dynamics in such environments, we develop a geometry-aware modeling approach that learns a residual Gaussian Process (GP). By utilizing a recursive sparse GP, the framework enables real-time adaptation to varying terrain geometry. The effectiveness of the learned model is demonstrated in a reference-tracking task using a Model Predictive Path Integral (MPPI) controller. Validation within a custom Isaac Sim environment confirms the framework's capability to maintain high tracking accuracy on challenging 3D surfaces.

**arXiv ID:** 2602.16206
</details>

<details>
<summary><strong>Decentralized and Fully Onboard: Range-Aided Cooperative Localization and Navigation on Micro Aerial Vehicles</strong> - Abhishek Goudar, Angela P. Schoellig - [[pdf]](https://arxiv.org/pdf/2602.16594)</summary>

**Abstract:** Controlling a team of robots in a coordinated manner is challenging because centralized approaches (where all computation is performed on a central machine) scale poorly, and globally referenced external localization systems may not always be available. In this work, we consider the problem of range-aided decentralized localization and formation control. In such a setting, each robot estimates its relative pose by combining data only from onboard odometry sensors and distance measurements to other robots in the team. Additionally, each robot calculates the control inputs necessary to collaboratively navigate an environment to accomplish a specific task, for example, moving in a desired formation while monitoring an area. We present a block coordinate descent approach to localization that does not require strict coordination between the robots. We present a novel formulation for formation control as inference on factor graphs that takes into account the state estimation uncertainty and can be solved efficiently. Our approach to range-aided localization and formation-based navigation is completely decentralized, does not require specialized trajectories to maintain formation, and achieves decimeter-level positioning and formation control accuracy. We demonstrate our approach through multiple real experiments involving formation flights in diverse indoor and outdoor environments.

**arXiv ID:** 2602.16594
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (17 papers)</h2></summary>

<details>
<summary><strong>EnterpriseGym Corecraft: Training Generalizable Agents on High-Fidelity RL Environments</strong> - Sushant Mehta, Logan Ritchie, Suhaas Garre, Nick Heiner, Edwin Chen - [[pdf]](https://arxiv.org/pdf/2602.16179)</summary>

**Abstract:** We show that training AI agents on high-fidelity reinforcement learning environments produces capabilities that generalize beyond the training distribution. We introduce \corecraft{}, the first environment in \textsc{EnterpriseGym}, Surge AI's suite of agentic RL environments. \corecraft{} is a fully operational enterprise simulation of a customer support organization, comprising over 2,500 entities across 14 entity types with 23 unique tools, designed to measure whether AI agents can perform the multi-step, domain-specific work that real jobs demand. Frontier models such as GPT-5.2 and Claude Opus 4.6 solve fewer than 30\% of tasks when all expert-authored rubric criteria must be satisfied. Using this environment, we train GLM~4.6 with Group Relative Policy Optimization (GRPO) and adaptive clipping. After a single epoch of training, the model improves from 25.37\% to 36.76\% task pass rate on held-out evaluation tasks. More importantly, these gains transfer to out-of-distribution benchmarks: +4.5\% on BFCL Parallel, +7.4\% on $\tau^2$-Bench Retail, and +6.8\% on Toolathlon (Pass@1). We believe three environment properties are consistent with the observed transfer: task-centric world building that optimizes for diverse, challenging tasks; expert-authored rubrics enabling reliable reward computation; and enterprise workflows that reflect realistic professional patterns. Our results suggest that environment quality, diversity, and realism are key factors enabling generalizable agent capabilities.

**arXiv ID:** 2602.16179
</details>

<details>
<summary><strong>Agent Skill Framework: Perspectives on the Potential of Small Language Models in Industrial Environments</strong> - Yangjie Xu, Lujun Li, Lama Sleem, Niccolo Gentile, Yewei Song, Yiqun Wang, Siming Ji, Wenbo Wu, Radu State - [[pdf]](https://arxiv.org/pdf/2602.16653)</summary>

**Abstract:** Agent Skill framework, now widely and officially supported by major players such as GitHub Copilot, LangChain, and OpenAI, performs especially well with proprietary models by improving context engineering, reducing hallucinations, and boosting task accuracy. Based on these observations, an investigation is conducted to determine whether the Agent Skill paradigm provides similar benefits to small language models (SLMs). This question matters in industrial scenarios where continuous reliance on public APIs is infeasible due to data-security and budget constraints requirements, and where SLMs often show limited generalization in highly customized scenarios. This work introduces a formal mathematical definition of the Agent Skill process, followed by a systematic evaluation of language models of varying sizes across multiple use cases. The evaluation encompasses two open-source tasks and a real-world insurance claims data set. The results show that tiny models struggle with reliable skill selection, while moderately sized SLMs (approximately 12B - 30B) parameters) benefit substantially from the Agent Skill approach. Moreover, code-specialized variants at around 80B parameters achieve performance comparable to closed-source baselines while improving GPU efficiency. Collectively, these findings provide a comprehensive and nuanced characterization of the capabilities and constraints of the framework, while providing actionable insights for the effective deployment of Agent Skills in SLM-centered environments.

**arXiv ID:** 2602.16653
</details>

<details>
<summary><strong>DataJoint 2.0: A Computational Substrate for Agentic Scientific Workflows</strong> - Dimitri Yatsenko, Thinh T. Nguyen - [[pdf]](https://arxiv.org/pdf/2602.16585)</summary>

**Abstract:** Operational rigor determines whether human-agent collaboration succeeds or fails. Scientific data pipelines need the equivalent of DevOps -- SciOps -- yet common approaches fragment provenance across disconnected systems without transactional guarantees. DataJoint 2.0 addresses this gap through the relational workflow model: tables represent workflow steps, rows represent artifacts, foreign keys prescribe execution order. The schema specifies not only what data exists but how it is derived -- a single formal system where data structure, computational dependencies, and integrity constraints are all queryable, enforceable, and machine-readable. Four technical innovations extend this foundation: object-augmented schemas integrating relational metadata with scalable object storage, semantic matching using attribute lineage to prevent erroneous joins, an extensible type system for domain-specific formats, and distributed job coordination designed for composability with external orchestration. By unifying data structure, data, and computational transformations, DataJoint creates a substrate for SciOps where agents can participate in scientific workflows without risking data corruption.

**arXiv ID:** 2602.16585
</details>

<details>
<summary><strong>CaveAgent: Transforming LLMs into Stateful Runtime Operators</strong> - Maohao Ran, Zhenglin Wan, Cooper Lin, Yanting Zhang, Hongyu Xin, Hongwei Fan, Yibo Xu, Beier Luo, Yaxin Zhou, Wangbo Zhao, Lijie Yang, Lang Feng, Fuchao Yang, Jingxuan Wu, Yiqiao Huang, Chendong Ma, Dailing Jiang, Jianbo Deng, Sihui Han, Yang You, Bo An, Yike Guo, Jun Song - [[pdf]](https://arxiv.org/pdf/2601.01569)</summary>

**Abstract:** LLM-based agents are increasingly capable of complex task execution, yet current agentic systems remain constrained by text-centric paradigms that struggle with long-horizon tasks due to fragile multi-turn dependencies and context drift. We present CaveAgent, a framework that shifts tool use from ``LLM-as-Text-Generator'' to ``LLM-as-Runtime-Operator.'' CaveAgent introduces a dual-stream architecture that inverts the conventional paradigm: rather than treating the LLM's text context as the primary workspace with tools as auxiliary, CaveAgent elevates the persistent Python runtime as the central locus of state, with a lightweight semantic stream serving as its orchestrator. Beyond leveraging code generation to resolve interdependent sub-tasks (e.g., loops, conditionals) in a single step, CaveAgent introduces \textit{Stateful Runtime Management}: it injects, manipulates, and retrieves complex Python objects (e.g., DataFrames, database connections) that persist across turns, unlike existing code-based approaches that remain text-bound. CaveAgent further provides a runtime-integrated skill management system that extends the Agent Skills open standard, enabling ecosystem interoperability through executable skill injections. This persistence mechanism serves as a high-fidelity external memory that reduces context drift in multi-turn interactions and preserves processed data for downstream applications without information loss. Evaluations show consistent improvement across challenging benchmarks, enabling CaveAgent to handle data scales that cause context overflow in both JSON-based and code-based agents. The accessible runtime state further provides programmatically verifiable feedback, enabling automated evaluation and reward signal generation without human annotation and establishing a structural foundation for future research in Reinforcement Learning with Verifiable Rewards (RLVR).

**arXiv ID:** 2601.01569
</details>

<details>
<summary><strong>Cocoa: Co-Planning and Co-Execution with AI Agents</strong> - K. J. Kevin Feng, Kevin Pu, Matt Latzke, Tal August, Pao Siangliulue, Jonathan Bragg, Daniel S. Weld, Amy X. Zhang, Joseph Chee Chang - [[pdf]](https://arxiv.org/pdf/2412.10999)</summary>

**Abstract:** As AI agents take on increasingly long-running tasks involving sophisticated planning and execution, there is a corresponding need for novel interaction designs that enable deeper human-agent collaboration. However, most prior works leverage human interaction to fix "autonomous" workflows that have yet to become fully autonomous or rigidly treat planning and execution as separate stages. Based on a formative study with 9 researchers using AI to support their work, we propose a design that affords greater flexibility in collaboration, so that users can 1) delegate agency to the user or agent via a collaborative plan where individual steps can be assigned; and 2) interleave planning and execution so that plans can adjust after partial execution. We introduce Cocoa, a system that takes design inspiration from computational notebooks to support complex research tasks. A lab study (n=16) found that Cocoa enabled steerability without sacrificing ease-of-use, and a week-long field deployment (n=7) showed how researchers collaborated with Cocoa to accomplish real-world tasks.

**arXiv ID:** 2412.10999
</details>

<details>
<summary><strong>MedReasoner: Reinforcement Learning Drives Reasoning Grounding from Clinical Thought to Pixel-Level Precision</strong> - Zhonghao Yan, Muxi Diao, Yuxuan Yang, Ruoyan Jing, Jiayuan Xu, Kaizhou Zhang, Lele Yang, Yanxi Liu, Kongming Liang, Zhanyu Ma - [[pdf]](https://arxiv.org/pdf/2508.08177)</summary>

**Abstract:** Accurately grounding regions of interest (ROIs) is critical for diagnosis and treatment planning in medical imaging. While multimodal large language models (MLLMs) combine visual perception with natural language, current medical-grounding pipelines still rely on supervised fine-tuning with explicit spatial hints, making them ill-equipped to handle the implicit queries common in clinical practice. This work makes three core contributions. We first define Unified Medical Reasoning Grounding (UMRG), a novel vision-language task that demands clinical reasoning and pixel-level grounding. Second, we release U-MRG-14K, a dataset of 14K samples featuring pixel-level masks alongside implicit clinical queries and reasoning traces, spanning 10 modalities, 15 super-categories, and 108 specific categories. Finally, we introduce MedReasoner, a modular framework that distinctly separates reasoning from segmentation: an MLLM reasoner is optimized with reinforcement learning, while a frozen segmentation expert converts spatial prompts into masks, with alignment achieved through format and accuracy rewards. MedReasoner achieves state-of-the-art performance on U-MRG-14K and demonstrates strong generalization to unseen clinical queries, underscoring the significant promise of reinforcement learning for interpretable medical grounding.

**arXiv ID:** 2508.08177
</details>

<details>
<summary><strong>Vision and Language: Novel Representations and Artificial intelligence for Driving Scene Safety Assessment and Autonomous Vehicle Planning</strong> - Ross Greer, Maitrayee Keskar, Angel Martinez-Sanchez, Parthib Roy, Shashank Shriram, Mohan Trivedi - [[pdf]](https://arxiv.org/pdf/2602.07680)</summary>

**Abstract:** Vision-language models (VLMs) have recently emerged as powerful representation learning systems that align visual observations with natural language concepts, offering new opportunities for semantic reasoning in safety-critical autonomous driving. This paper investigates how vision-language representations support driving scene safety assessment and decision-making when integrated into perception, prediction, and planning pipelines. We study three complementary system-level use cases. First, we introduce a lightweight, category-agnostic hazard screening approach leveraging CLIP-based image-text similarity to produce a low-latency semantic hazard signal. This enables robust detection of diverse and out-of-distribution road hazards without explicit object detection or visual question answering. Second, we examine the integration of scene-level vision-language embeddings into a transformer-based trajectory planning framework using the Waymo Open Dataset. Our results show that naively conditioning planners on global embeddings does not improve trajectory accuracy, highlighting the importance of representation-task alignment and motivating the development of task-informed extraction methods for safety-critical planning. Third, we investigate natural language as an explicit behavioral constraint on motion planning using the doScenes dataset. In this setting, passenger-style instructions grounded in visual scene elements suppress rare but severe planning failures and improve safety-aligned behavior in ambiguous scenarios. Taken together, these findings demonstrate that vision-language representations hold significant promise for autonomous driving safety when used to express semantic risk, intent, and behavioral constraints. Realizing this potential is fundamentally an engineering problem requiring careful system design and structured grounding rather than direct feature injection.

**arXiv ID:** 2602.07680
</details>

<details>
<summary><strong>STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens</strong> - Shiqi Liu, Zeyu He, Guojian Zhan, Letian Tao, Zhilong Zheng, Jiang Wu, Yinuo Wang, Yang Guan, Kehua Sheng, Bo Zhang, Keqiang Li, Jingliang Duan, Shengbo Eben Li - [[pdf]](https://arxiv.org/pdf/2602.15620)</summary>

**Abstract:** Reinforcement Learning (RL) has significantly improved large language model reasoning, but existing RL fine-tuning methods rely heavily on heuristic techniques such as entropy regularization and reweighting to maintain stability. In practice, they often suffer from late-stage performance collapse, leading to degraded reasoning quality and unstable training. Our analysis shows that the magnitude of token-wise policy gradients in RL is negatively correlated with token probability and local policy entropy. We find that training instability can be caused by a tiny fraction of tokens, approximately 0.01\%, which we term \emph{spurious tokens}. When such tokens appear in correct responses, they contribute little to the reasoning outcome but inherit the full sequence-level reward, leading to abnormally amplified gradient updates. To mitigate this instability, we design S2T (silencing spurious tokens) mechanism to efficiently identify spurious tokens through characteristic signals with low probability, low entropy, and positive advantage, and then to suppress their gradient perturbations during optimization. Incorporating this mechanism into a group-based objective, we propose Spurious-Token-Aware Policy Optimization (STAPO), which promotes stable and effective large-scale model refinement. Across six mathematical reasoning benchmarks using Qwen 1.7B, 8B, and 14B base models, STAPO consistently demonstrates superior entropy stability and achieves an average performance improvement of 7.13\% ($\rho_{\mathrm{T}}$=1.0, top-p=1.0) and 3.69\% ($\rho_{\mathrm{T}}$=0.7, top-p=0.9) over GRPO, 20-Entropy and JustRL.

**arXiv ID:** 2602.15620
</details>

<details>
<summary><strong>MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks</strong> - Zexue He, Yu Wang, Churan Zhi, Yuanzhe Hu, Tzu-Ping Chen, Lang Yin, Ze Chen, Tong Arthur Wu, Siru Ouyang, Zihan Wang, Jiaxin Pei, Julian McAuley, Yejin Choi, Alex Pentland - [[pdf]](https://arxiv.org/pdf/2602.16313)</summary>

**Abstract:** Existing evaluations of agents with memory typically assess memorization and action in isolation. One class of benchmarks evaluates memorization by testing recall of past conversations or text but fails to capture how memory is used to guide future decisions. Another class focuses on agents acting in single-session tasks without the need for long-term memory. However, in realistic settings, memorization and action are tightly coupled: agents acquire memory while interacting with the environment, and subsequently rely on that memory to solve future tasks. To capture this setting, we introduce MemoryArena, a unified evaluation gym for benchmarking agent memory in multi-session Memory-Agent-Environment loops. The benchmark consists of human-crafted agentic tasks with explicitly interdependent subtasks, where agents must learn from earlier actions and feedback by distilling experiences into memory, and subsequently use that memory to guide later actions to solve the overall task. MemoryArena supports evaluation across web navigation, preference-constrained planning, progressive information search, and sequential formal reasoning, and reveals that agents with near-saturated performance on existing long-context memory benchmarks like LoCoMo perform poorly in our agentic setting, exposing a gap in current evaluations for agents with memory.

**arXiv ID:** 2602.16313
</details>

<details>
<summary><strong>SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models</strong> - Ziyi Yang, Weizhou Shen, Chenliang Li, Ruijun Chen, Fanqi Wan, Ming Yan, Xiaojun Quan, Fei Huang - [[pdf]](https://arxiv.org/pdf/2509.23863)</summary>

**Abstract:** Progress in long-context reasoning for large language models (LLMs) has lagged behind other recent advances. This gap arises not only from the intrinsic difficulty of processing long texts, but also from the scarcity of reliable human annotations and programmatically verifiable reward signals. In this paper, we propose SPELL, a multi-role self-play reinforcement learning framework that enables scalable, label-free optimization for long-context reasoning. SPELL integrates three cyclical roles-questioner, responder, and verifier-within a single model to enable continual self-improvement. The questioner generates questions from raw documents paired with reference answers; the responder learns to solve these questions based on the documents; and the verifier evaluates semantic equivalence between the responder's output and the questioner's reference answer, producing reward signals to guide continual training. To stabilize training, we introduce an automated curriculum that gradually increases document length and a reward function that adapts question difficulty to the model's evolving capabilities. Extensive experiments on six long-context benchmarks show that SPELL consistently improves performance across diverse LLMs and outperforms equally sized models fine-tuned on large-scale annotated data. Notably, SPELL achieves an average 7.6-point gain in pass@8 on the strong reasoning model Qwen3-30B-A3B-Thinking, raising its performance ceiling and showing promise for scaling to even more capable models. Our code is available at this https URL.

**arXiv ID:** 2509.23863
</details>

<details>
<summary><strong>Reinforcement Learning for Parameterized Quantum State Preparation: A Comparative Study</strong> - Gerhard Stenzel, Isabella Debelic, Michael Kölle, Tobias Rohe, Leo Sünkel, Julian Hager, Claudia Linnhoff-Popien - [[pdf]](https://arxiv.org/pdf/2602.16523)</summary>

**Abstract:** We extend directed quantum circuit synthesis (DQCS) with reinforcement learning from purely discrete gate selection to parameterized quantum state preparation with continuous single-qubit rotations \(R_x\), \(R_y\), and \(R_z\). We compare two training regimes: a one-stage agent that jointly selects the gate type, the affected qubit(s), and the rotation angle; and a two-stage variant that first proposes a discrete circuit and subsequently optimizes the rotation angles with Adam using parameter-shift gradients. Using Gymnasium and PennyLane, we evaluate Proximal Policy Optimization (PPO) and Advantage Actor--Critic (A2C) on systems comprising two to ten qubits and on targets of increasing complexity with \(\lambda\) ranging from one to five. Whereas A2C does not learn effective policies in this setting, PPO succeeds under stable hyperparameters (one-stage: learning rate approximately \(5\times10^{-4}\) with a self-fidelity-error threshold of 0.01; two-stage: learning rate approximately \(10^{-4}\)). Both approaches reliably reconstruct computational basis states (between 83\% and 99\% success) and Bell states (between 61\% and 77\% success). However, scalability saturates for \(\lambda\) of approximately three to four and does not extend to ten-qubit targets even at \(\lambda=2\). The two-stage method offers only marginal accuracy gains while requiring around three times the runtime. For practicality under a fixed compute budget, we therefore recommend the one-stage PPO policy, provide explicit synthesized circuits, and contrast with a classical variational baseline to outline avenues for improved scalability.

**arXiv ID:** 2602.16523
</details>

<details>
<summary><strong>Capacity-constrained demand response in smart grids using deep reinforcement learning</strong> - Shafagh Abband Pashaki, Sepehr Maleki, Amir Badiee - [[pdf]](https://arxiv.org/pdf/2602.16525)</summary>

**Abstract:** This paper presents a capacity-constrained incentive-based demand response approach for residential smart grids. It aims to maintain electricity grid capacity limits and prevent congestion by financially incentivising end users to reduce or shift their energy consumption. The proposed framework adopts a hierarchical architecture in which a service provider adjusts hourly incentive rates based on wholesale electricity prices and aggregated residential load. The financial interests of both the service provider and end users are explicitly considered. A deep reinforcement learning approach is employed to learn optimal real-time incentive rates under explicit capacity constraints. Heterogeneous user preferences are modelled through appliance-level home energy management systems and dissatisfaction costs. Using real-world residential electricity consumption and price data from three households, simulation results show that the proposed approach effectively reduces peak demand and smooths the aggregated load profile. This leads to an approximately 22.82% reduction in the peak-to-average ratio compared to the no-demand-response case.

**arXiv ID:** 2602.16525
</details>

<details>
<summary><strong>Vulnerability Analysis of Safe Reinforcement Learning via Inverse Constrained Reinforcement Learning</strong> - Jialiang Fan, Shixiong Jiang, Mengyu Liu, Fanxin Kong - [[pdf]](https://arxiv.org/pdf/2602.16543)</summary>

**Abstract:** Safe reinforcement learning (Safe RL) aims to ensure policy performance while satisfying safety constraints. However, most existing Safe RL methods assume benign environments, making them vulnerable to adversarial perturbations commonly encountered in real-world settings. In addition, existing gradient-based adversarial attacks typically require access to the policy's gradient information, which is often impractical in real-world scenarios. To address these challenges, we propose an adversarial attack framework to reveal vulnerabilities of Safe RL policies. Using expert demonstrations and black-box environment interaction, our framework learns a constraint model and a surrogate (learner) policy, enabling gradient-based attack optimization without requiring the victim policy's internal gradients or the ground-truth safety constraints. We further provide theoretical analysis establishing feasibility and deriving perturbation bounds. Experiments on multiple Safe RL benchmarks demonstrate the effectiveness of our approach under limited privileged access.

**arXiv ID:** 2602.16543
</details>

<details>
<summary><strong>RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion</strong> - Tianmeng Hu, Yongzheng Cui, Biao Luo, Ke Li - [[pdf]](https://arxiv.org/pdf/2602.16548)</summary>

**Abstract:** The inverse design of RNA three-dimensional (3D) structures is crucial for engineering functional RNAs in synthetic biology and therapeutics. While recent deep learning approaches have advanced this field, they are typically optimized and evaluated using native sequence recovery, which is a limited surrogate for structural fidelity, since different sequences can fold into similar 3D structures and high recovery does not necessarily indicate correct folding. To address this limitation, we propose RIDER, an RNA Inverse DEsign framework with Reinforcement learning that directly optimizes for 3D structural similarity. First, we develop and pre-train a GNN-based generative diffusion model conditioned on the target 3D structure, achieving a 9% improvement in native sequence recovery over state-of-the-art methods. Then, we fine-tune the model with an improved policy gradient algorithm using four task-specific reward functions based on 3D self-consistency metrics. Experimental results show that RIDER improves structural similarity by over 100% across all metrics and discovers designs that are distinct from native sequences.

**arXiv ID:** 2602.16548
</details>

<details>
<summary><strong>Shrinking the Variance: Shrinkage Baselines for Reinforcement Learning with Verifiable Rewards</strong> - Guanning Zeng, Zhaoyi Zhou, Daman Arora, Andrea Zanette - [[pdf]](https://arxiv.org/pdf/2511.03710)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for post-training large reasoning models (LRMs) using policy-gradient methods such as GRPO. To stabilize training, these methods typically center trajectory rewards by subtracting the empirical mean reward for each prompt. Statistically, this centering acts as a control variate (baseline), reducing the variance of the policy-gradient estimator. In practice, the mean reward is estimated using per-prompt empirical averages computed from the generations for each prompt in a batch. Motivated by Stein's paradox, we propose shrinkage estimators that combine per-prompt and across-prompt means to improve per-prompt mean estimation accuracy, especially in the low-generation regime typical of RLVR. Theoretically, we construct a shrinkage-based baseline that provably yields lower-variance policy-gradient estimators across algorithms. Our baseline is a drop-in replacement for standard per-prompt mean baselines and requires no additional hyperparameters or computation. Empirically, shrinkage baselines consistently outperform empirical-mean baselines, producing lower-variance gradient updates and improved training stability.

**arXiv ID:** 2511.03710
</details>

<details>
<summary><strong>Dual-Quadruped Collaborative Transportation in Narrow Environments via Safe Reinforcement Learning</strong> - Zhezhi Lei, Zhihai Bi, Wenxin Wang, Jun Ma - [[pdf]](https://arxiv.org/pdf/2602.16353)</summary>

**Abstract:** Collaborative transportation, where multiple robots collaboratively transport a payload, has garnered significant attention in recent years. While ensuring safe and high-performance inter-robot collaboration is critical for effective task execution, it is difficult to pursue in narrow environments where the feasible region is extremely limited. To address this challenge, we propose a novel approach for dual-quadruped collaborative transportation via safe reinforcement learning (RL). Specifically, we model the task as a fully cooperative constrained Markov game, where collision avoidance is formulated as constraints. We introduce a cost-advantage decomposition method that enforces the sum of team constraints to remain below an upper bound, thereby guaranteeing task safety within an RL framework. Furthermore, we propose a constraint allocation method that assigns shared constraints to individual robots to maximize the overall task reward, encouraging autonomous task-assignment among robots, thereby improving collaborative task performance. Simulation and real-time experimental results demonstrate that the proposed approach achieves superior performance and a higher success rate in dual-quadruped collaborative transportation compared to existing methods.

**arXiv ID:** 2602.16353
</details>

<details>
<summary><strong>Peeking Ahead of the Field Study: Exploring VLM Personas as Support Tools for Embodied Studies in HCI</strong> - Xinyue Gui, Ding Xia, Mark Colley, Yuan Li, Vishal Chauhan, Anubhav Anubhav, Zhongyi Zhou, Ehsan Javanmardi, Stela Hanbyeol Seo, Chia-Ming Chang, Manabu Tsukada, Takeo Igarashi - [[pdf]](https://arxiv.org/pdf/2602.16157)</summary>

**Abstract:** Field studies are irreplaceable but costly, time-consuming, and error-prone, which need careful preparation. Inspired by rapid-prototyping in manufacturing, we propose a fast, low-cost evaluation method using Vision-Language Model (VLM) personas to simulate outcomes comparable to field results. While LLMs show human-like reasoning and language capabilities, autonomous vehicle (AV)-pedestrian interaction requires spatial awareness, emotional empathy, and behavioral generation. This raises our research question: To what extent can VLM personas mimic human responses in field studies? We conducted parallel studies: 1) one real-world study with 20 participants, and 2) one video-study using 20 VLM personas, both on a street-crossing task. We compared their responses and interviewed five HCI researchers on potential applications. Results show that VLM personas mimic human response patterns (e.g., average crossing times of 5.25 s vs. 5.07 s) lack the behavioral variability and depth. They show promise for formative studies, field study preparation, and human data augmentation.

**arXiv ID:** 2602.16157
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
