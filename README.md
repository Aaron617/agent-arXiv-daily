# Agent arXiv Daily

**Last Updated:** 2025-11-20 02:48:13

**Total Papers:** 68

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
<summary><strong>Uncertainty-Aware Measurement of Scenario Suite Representativeness for Autonomous Systems</strong> - Robab Aghazadeh Chakherlou, Siddartha Khastgir, Xingyu Zhao, Jerein Jeyachandran, Shufeng Chen - [[pdf]](https://arxiv.org/pdf/2511.14853)</summary>

**Abstract:** Assuring the trustworthiness and safety of AI systems, e.g., autonomous vehicles (AV), depends critically on the data-related safety properties, e.g., representativeness, completeness, etc., of the datasets used for their training and testing. Among these properties, this paper focuses on representativeness-the extent to which the scenario-based data used for training and testing, reflect the operational conditions that the system is designed to operate safely in, i.e., Operational Design Domain (ODD) or expected to encounter, i.e., Target Operational Domain (TOD). We propose a probabilistic method that quantifies representativeness by comparing the statistical distribution of features encoded by the scenario suites with the corresponding distribution of features representing the TOD, acknowledging that the true TOD distribution is unknown, as it can only be inferred from limited data.
We apply an imprecise Bayesian method to handle limited data and uncertain priors. The imprecise Bayesian formulation produces interval-valued, uncertainty-aware estimates of representativeness, rather than a single value. We present a numerical example comparing the distributions of the scenario suite and the inferred TOD across operational categories-weather, road type, time of day, etc., under dependencies and prior uncertainty. We estimate representativeness locally (between categories) and globally as an interval.

**arXiv ID:** 2511.14853
</details>

<details>
<summary><strong>An LLM-Powered Agent for Real-Time Analysis of the Vietnamese IT Job Market</strong> - Minh-Thuan Nguyen, Thien Vo-Thanh, Thai-Duy Dinh, Xuan-Quang Phan, Tan-Ha Mai, Lam-Son Lê - [[pdf]](https://arxiv.org/pdf/2511.14767)</summary>

**Abstract:** Individuals entering Vietnam's dynamic Information Technology (IT) job market face a critical gap in reliable career guidance. Existing market reports are often outdated, while the manual analysis of thousands of job postings is impractical for most. To address this challenge, we present the AI Job Market Consultant, a novel conversational agent that delivers deep, data-driven insights directly from the labor market in real-time. The foundation of our system is a custom-built dataset created via an automated pipeline that crawls job portals using Playwright and leverages the Large Language Model (LLM) to intelligently structure unstructured posting data. The core of our system is a tool-augmented AI agent, based on the ReAct agentic framework, which enables the ability of autonomously reasoning, planning, and executing actions through a specialized toolbox for SQL queries, semantic search, and data visualization. Our prototype successfully collected and analyzed 3,745 job postings, demonstrating its ability to answer complex, multi-step queries, generate on-demand visualizations, and provide personalized career advice grounded in real-world data. This work introduces a new paradigm for labor market analysis, showcasing how specialized agentic AI systems can democratize access to timely, trustworthy career intelligence for the next generation of professionals.

**arXiv ID:** 2511.14767
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (7 papers)</h2></summary>

<details>
<summary><strong>Octopus: Agentic Multimodal Reasoning with Six-Capability Orchestration</strong> - Yifu Guo, Zishan Xu, Zhiyuan Yao, Yuquan Lu, Jiaye Lin, Sen Hu, Zhenheng Tang, Yingchao Li, Huacan Wang, Ronghao Chen - [[pdf]](https://arxiv.org/pdf/2511.15351)</summary>

**Abstract:** Existing multimodal reasoning models and frameworks suffer from fundamental architectural limitations: most lack the human-like ability to autonomously explore diverse reasoning pathways-whether in direct inference, tool-driven visual exploration, programmatic visual manipulation, or intrinsic visual imagination. Consequently, they struggle to adapt to dynamically changing capability requirements in real-world tasks. Meanwhile, humans exhibit a complementary set of thinking abilities when addressing such tasks, whereas existing methods typically cover only a subset of these dimensions. Inspired by this, we propose Octopus: Agentic Multimodal Reasoning with Six-Capability Orchestration, a new paradigm for multimodal agentic reasoning. We define six core capabilities essential for multimodal reasoning and organize a comprehensive evaluation benchmark, Octopus-Bench, accordingly. Octopus is capable of autonomously exploring during reasoning and dynamically selecting the most appropriate capability based on the current state. Experimental results show that Octopus achieves the best performance on the vast majority of tasks in Octopus-Bench, highlighting the crucial role of capability coordination in agentic multimodal reasoning.

**arXiv ID:** 2511.15351
</details>

<details>
<summary><strong>What Does It Take to Be a Good AI Research Agent? Studying the Role of Ideation Diversity</strong> - Alexis Audran-Reiss, Jordi Armengol Estapé, Karen Hambardzumyan, Amar Budhiraja, Martin Josifoski, Edan Toledo, Rishi Hazra, Despoina Magka, Michael Shvartsman, Parth Pathak, Justine T Kao, Lucia Cipolina-Kun, Bhavul Gauri, Jean-Christophe Gagnon-Audet, Emanuel Tewolde, Jenny Zhang, Taco Cohen, Yossi Adi, Tatiana Shavrina, Yoram Bachrach - [[pdf]](https://arxiv.org/pdf/2511.15593)</summary>

**Abstract:** AI research agents offer the promise to accelerate scientific progress by automating the design, implementation, and training of machine learning models. However, the field is still in its infancy, and the key factors driving the success or failure of agent trajectories are not fully understood. We examine the role that ideation diversity plays in agent performance. First, we analyse agent trajectories on MLE-bench, a well-known benchmark to evaluate AI research agents, across different models and agent scaffolds. Our analysis reveals that different models and agent scaffolds yield varying degrees of ideation diversity, and that higher-performing agents tend to have increased ideation diversity. Further, we run a controlled experiment where we modify the degree of ideation diversity, demonstrating that higher ideation diversity results in stronger performance. Finally, we strengthen our results by examining additional evaluation metrics beyond the standard medal-based scoring of MLE-bench, showing that our findings still hold across other agent performance metrics.

**arXiv ID:** 2511.15593
</details>

<details>
<summary><strong>Towards High-Consistency Embodied World Model with Multi-View Trajectory Videos</strong> - Taiyi Su, Jian Zhu, Yaxuan Li, Chong Ma, Zitai Huang, Hanli Wang, Yi Xu - [[pdf]](https://arxiv.org/pdf/2511.12882)</summary>

**Abstract:** Embodied world models aim to predict and interact with the physical world through visual observations and actions. However, existing models struggle to accurately translate low-level actions (e.g., joint positions) into precise robotic movements in predicted frames, leading to inconsistencies with real-world physical interactions. To address these limitations, we propose MTV-World, an embodied world model that introduces Multi-view Trajectory-Video control for precise visuomotor prediction. Specifically, instead of directly using low-level actions for control, we employ trajectory videos obtained through camera intrinsic and extrinsic parameters and Cartesian-space transformation as control signals. However, projecting 3D raw actions onto 2D images inevitably causes a loss of spatial information, making a single view insufficient for accurate interaction modeling. To overcome this, we introduce a multi-view framework that compensates for spatial information loss and ensures high-consistency with physical world. MTV-World forecasts future frames based on multi-view trajectory videos as input and conditioning on an initial frame per view. Furthermore, to systematically evaluate both robotic motion precision and object interaction accuracy, we develop an auto-evaluation pipeline leveraging multimodal large models and referring video object segmentation models. To measure spatial consistency, we formulate it as an object location matching problem and adopt the Jaccard Index as the evaluation metric. Extensive experiments demonstrate that MTV-World achieves precise control execution and accurate physical interaction modeling in complex dual-arm scenarios.

**arXiv ID:** 2511.12882
</details>

<details>
<summary><strong>Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving</strong> - Kangqiao Zhao, Shuo Huai, Xurui Song, Jun Luo - [[pdf]](https://arxiv.org/pdf/2511.14386)</summary>

**Abstract:** Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.

**arXiv ID:** 2511.14386
</details>

<details>
<summary><strong>Agentic AI Systems in Electrical Power Systems Engineering: Current State-of-the-Art and Challenges</strong> - Soham Ghosh, Gaurav Mittal - [[pdf]](https://arxiv.org/pdf/2511.14478)</summary>

**Abstract:** Agentic AI systems have recently emerged as a critical and transformative approach in artificial intelligence, offering capabilities that extend far beyond traditional AI agents and contemporary generative AI models. This rapid evolution necessitates a clear conceptual and taxonomical understanding to differentiate this new paradigm. Our paper addresses this gap by providing a comprehensive review that establishes a precise definition and taxonomy for "agentic AI," with the aim of distinguishing it from previous AI paradigms. The concepts are gradually introduced, starting with a highlight of its diverse applications across the broader field of engineering. The paper then presents four detailed, state-of-the-art use case applications specifically within electrical engineering. These case studies demonstrate practical impact, ranging from an advanced agentic framework for streamlining complex power system studies and benchmarking to a novel system developed for survival analysis of dynamic pricing strategies in battery swapping stations. Finally, to ensure robust deployment, the paper provides detailed failure mode investigations. From these findings, we derive actionable recommendations for the design and implementation of safe, reliable, and accountable agentic AI systems, offering a critical resource for researchers and practitioners.

**arXiv ID:** 2511.14478
</details>

<details>
<summary><strong>Is Your VLM for Autonomous Driving Safety-Ready? A Comprehensive Benchmark for Evaluating External and In-Cabin Risks</strong> - Xianhui Meng, Yuchen Zhang, Zhijian Huang, Zheng Lu, Ziling Ji, Yaoyao Yin, Hongyuan Zhang, Guangfeng Jiang, Yandan Lin, Long Chen, Hangjun Ye, Li Zhang, Jun Liu, Xiaoshuai Hao - [[pdf]](https://arxiv.org/pdf/2511.14592)</summary>

**Abstract:** Vision-Language Models (VLMs) show great promise for autonomous driving, but their suitability for safety-critical scenarios is largely unexplored, raising safety concerns. This issue arises from the lack of comprehensive benchmarks that assess both external environmental risks and in-cabin driving behavior safety simultaneously. To bridge this critical gap, we introduce DSBench, the first comprehensive Driving Safety Benchmark designed to assess a VLM's awareness of various safety risks in a unified manner. DSBench encompasses two major categories: external environmental risks and in-cabin driving behavior safety, divided into 10 key categories and a total of 28 sub-categories. This comprehensive evaluation covers a wide range of scenarios, ensuring a thorough assessment of VLMs' performance in safety-critical contexts. Extensive evaluations across various mainstream open-source and closed-source VLMs reveal significant performance degradation under complex safety-critical situations, highlighting urgent safety concerns. To address this, we constructed a large dataset of 98K instances focused on in-cabin and external safety scenarios, showing that fine-tuning on this dataset significantly enhances the safety performance of existing VLMs and paves the way for advancing autonomous driving technology. The benchmark toolkit, code, and model checkpoints will be publicly accessible.

**arXiv ID:** 2511.14592
</details>

<details>
<summary><strong>Computer-Use Agents as Judges for Generative User Interface</strong> - Kevin Qinghong Lin, Siyuan Hu, Linjie Li, Zhengyuan Yang, Lijuan Wang, Philip Torr, Mike Zheng Shou - [[pdf]](https://arxiv.org/pdf/2511.15567)</summary>

**Abstract:** Computer-Use Agents (CUA) are becoming increasingly capable of autonomously operating digital environments through Graphical User Interfaces (GUI). Yet, most GUI remain designed primarily for humans--prioritizing aesthetics and usability--forcing agents to adopt human-oriented behaviors that are unnecessary for efficient task execution. At the same time, rapid advances in coding-oriented language models (Coder) have transformed automatic GUI design. This raises a fundamental question: Can CUA as judges to assist Coder for automatic GUI design? To investigate, we introduce AUI-Gym, a benchmark for Automatic GUI development spanning 52 applications across diverse domains. Using language models, we synthesize 1560 tasks that simulate real-world scenarios. To ensure task reliability, we further develop a verifier that programmatically checks whether each task is executable within its environment. Building on this, we propose a Coder-CUA in Collaboration framework: the Coder acts as Designer, generating and revising websites, while the CUA serves as Judge, evaluating functionality and refining designs. Success is measured not by visual appearance, but by task solvability and CUA navigation success rate. To turn CUA feedback into usable guidance, we design a CUA Dashboard that compresses multi-step navigation histories into concise visual summaries, offering interpretable guidance for iterative redesign. By positioning agents as both designers and judges, our framework shifts interface design toward agent-native efficiency and reliability. Our work takes a step toward shifting agents from passive use toward active participation in digital environments. Our code and dataset are available at this https URL.

**arXiv ID:** 2511.15567
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>Taxonomy, Evaluation and Exploitation of IPI-Centric LLM Agent Defense Frameworks</strong> - Zimo Ji, Xunguang Wang, Zongjie Li, Pingchuan Ma, Yudong Gao, Daoyuan Wu, Xincheng Yan, Tian Tian, Shuai Wang - [[pdf]](https://arxiv.org/pdf/2511.15203)</summary>

**Abstract:** Large Language Model (LLM)-based agents with function-calling capabilities are increasingly deployed, but remain vulnerable to Indirect Prompt Injection (IPI) attacks that hijack their tool calls. In response, numerous IPI-centric defense frameworks have emerged. However, these defenses are fragmented, lacking a unified taxonomy and comprehensive evaluation. In this Systematization of Knowledge (SoK), we present the first comprehensive analysis of IPI-centric defense frameworks. We introduce a comprehensive taxonomy of these defenses, classifying them along five dimensions. We then thoroughly assess the security and usability of representative defense frameworks. Through analysis of defensive failures in the assessment, we identify six root causes of defense circumvention. Based on these findings, we design three novel adaptive attacks that significantly improve attack success rates targeting specific frameworks, demonstrating the severity of the flaws in these defenses. Our paper provides a foundation and critical insights for the future development of more secure and usable IPI-centric agent defense frameworks.

**arXiv ID:** 2511.15203
</details>

<details>
<summary><strong>DEPO: Dual-Efficiency Preference Optimization for LLM Agents</strong> - Sirui Chen, Mengshi Zhao, Lei Xu, Yuying Zhao, Beier Zhu, Hanwang Zhang, Shengjie Zhao, Chaochao Lu - [[pdf]](https://arxiv.org/pdf/2511.15392)</summary>

**Abstract:** Recent advances in large language models (LLMs) have greatly improved their reasoning and decision-making abilities when deployed as agents. Richer reasoning, however, often comes at the cost of longer chain of thought (CoT), hampering interaction efficiency in real-world scenarios. Nevertheless, there still lacks systematic definition of LLM agent efficiency, hindering targeted improvements. To this end, we introduce dual-efficiency, comprising (i) step-level efficiency, which minimizes tokens per step, and (ii) trajectory-level efficiency, which minimizes the number of steps to complete a task. Building on this definition, we propose DEPO, a dual-efficiency preference optimization method that jointly rewards succinct responses and fewer action steps. Experiments on WebShop and BabyAI show that DEPO cuts token usage by up to 60.9% and steps by up to 26.9%, while achieving up to a 29.3% improvement in performance. DEPO also generalizes to three out-of-domain math benchmarks and retains its efficiency gains when trained on only 25% of the data. Our project page is at this https URL.

**arXiv ID:** 2511.15392
</details>

<details>
<summary><strong>Automating Android Build Repair: Bridging the Reasoning-Execution Gap in LLM Agents with Domain-Specific Tools</strong> - Ha Min Son, Huan Ren, Xin Liu, Zhe Zhao - [[pdf]](https://arxiv.org/pdf/2510.08640)</summary>

**Abstract:** Android is the largest mobile platform, yet automatically building applications remains a practical challenge. While Large Language Models (LLMs) show promise for code repair, their use for fixing Android build errors remains underexplored. To address this gap, we first introduce AndroidBuildBench, a benchmark of 1,019 build failures curated from the commit histories of 43 open-source Android projects. Each problem is paired with a verified solution from a subsequent commit, ensuring that fixes are feasible. Second, we propose GradleFixer, an LLM agent with domain-specific tools for inspecting and manipulating the Gradle build environment. GradleFixer achieves a resolve rate of 81.4% (pass@1), significantly outperforming a state-of-the-art coding agent that relies on a general-purpose shell. GradleFixer's success suggests that while LLMs possess the high-level knowledge to solve these failures, they struggle to translate this knowledge into effective low-level actions using a general-purpose shell. We demonstrate the effectiveness of a strategy we term Tool Bridging, which replaces general-purpose shell commands with domain-aware abstractions. We hypothesize this approach works through two mechanisms: 1) it provides tools in an API-like format that LLMs use more reliably, and 2) it constrains the action space to relevant operations. This approach bridges the gap between the model's high-level reasoning and effective low-level execution.

**arXiv ID:** 2510.08640
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (19 papers)</h2></summary>

<details>
<summary><strong>Ask WhAI:Probing Belief Formation in Role-Primed LLM Agents</strong> - Keith Moore, Jun W. Kim, David Lyu, Jeffrey Heo, Ehsan Adeli - [[pdf]](https://arxiv.org/pdf/2511.14780)</summary>

**Abstract:** We present Ask WhAI, a systems-level framework for inspecting and perturbing belief states in multi-agent interactions. The framework records and replays agent interactions, supports out-of-band queries into each agent's beliefs and rationale, and enables counterfactual evidence injection to test how belief structures respond to new information. We apply the framework to a medical case simulator notable for its multi-agent shared memory (a time-stamped electronic medical record, or EMR) and an oracle agent (the LabAgent) that holds ground truth lab results revealed only when explicitly queried. We stress-test the system on a multi-specialty diagnostic journey for a child with an abrupt-onset neuropsychiatric presentation. Large language model agents, each primed with strong role-specific priors ("act like a neurologist", "act like an infectious disease specialist"), write to a shared medical record and interact with a moderator across sequential or parallel encounters. Breakpoints at key diagnostic moments enable pre- and post-event belief queries, allowing us to distinguish entrenched priors from reasoning or evidence-integration effects. The simulation reveals that agent beliefs often mirror real-world disciplinary stances, including overreliance on canonical studies and resistance to counterevidence, and that these beliefs can be traced and interrogated in ways not possible with human experts. By making such dynamics visible and testable, Ask WhAI offers a reproducible way to study belief formation and epistemic silos in multi-agent scientific reasoning.

**arXiv ID:** 2511.14780
</details>

<details>
<summary><strong>Task Specific Sharpness Aware O-RAN Resource Management using Multi Agent Reinforcement Learning</strong> - Fatemeh Lotfi, Hossein Rajoli, Fatemeh Afghah - [[pdf]](https://arxiv.org/pdf/2511.15002)</summary>

**Abstract:** Next-generation networks utilize the Open Radio Access Network (O-RAN) architecture to enable dynamic resource management, facilitated by the RAN Intelligent Controller (RIC). While deep reinforcement learning (DRL) models show promise in optimizing network resources, they often struggle with robustness and generalizability in dynamic environments. This paper introduces a novel resource management approach that enhances the Soft Actor Critic (SAC) algorithm with Sharpness-Aware Minimization (SAM) in a distributed Multi-Agent RL (MARL) framework. Our method introduces an adaptive and selective SAM mechanism, where regularization is explicitly driven by temporal-difference (TD)-error variance, ensuring that only agents facing high environmental complexity are regularized. This targeted strategy reduces unnecessary overhead, improves training stability, and enhances generalization without sacrificing learning efficiency. We further incorporate a dynamic $\rho$ scheduling scheme to refine the exploration-exploitation trade-off across agents. Experimental results show our method significantly outperforms conventional DRL approaches, yielding up to a $22\%$ improvement in resource allocation efficiency and ensuring superior QoS satisfaction across diverse O-RAN slices.

**arXiv ID:** 2511.15002
</details>

<details>
<summary><strong>Beyond GeneGPT: A Multi-Agent Architecture with Open-Source LLMs for Enhanced Genomic Question Answering</strong> - Haodong Chen, Guido Zuccon, Teerapong Leelanupab - [[pdf]](https://arxiv.org/pdf/2511.15061)</summary>

**Abstract:** Genomic question answering often requires complex reasoning and integration across diverse biomedical sources. GeneGPT addressed this challenge by combining domain-specific APIs with OpenAI's code-davinci-002 large language model to enable natural language interaction with genomic databases. However, its reliance on a proprietary model limits scalability, increases operational costs, and raises concerns about data privacy and generalization.
In this work, we revisit and reproduce GeneGPT in a pilot study using open source models, including Llama 3.1, Qwen2.5, and Qwen2.5 Coder, within a monolithic architecture; this allows us to identify the limitations of this approach. Building on this foundation, we then develop OpenBioLLM, a modular multi-agent framework that extends GeneGPT by introducing agent specialization for tool routing, query generation, and response validation. This enables coordinated reasoning and role-based task execution.
OpenBioLLM matches or outperforms GeneGPT on over 90% of the benchmark tasks, achieving average scores of 0.849 on Gene-Turing and 0.830 on GeneHop, while using smaller open-source models without additional fine-tuning or tool-specific pretraining. OpenBioLLM's modular multi-agent design reduces latency by 40-50% across benchmark tasks, significantly improving efficiency without compromising model capability. The results of our comprehensive evaluation highlight the potential of open-source multi-agent systems for genomic question answering. Code and resources are available at this https URL.

**arXiv ID:** 2511.15061
</details>

<details>
<summary><strong>Knowledge-Informed Automatic Feature Extraction via Collaborative Large Language Model Agents</strong> - Henrik Bradland, Morten Goodwin, Vladimir I. Zadorozhny, Per-Arne Andersen - [[pdf]](https://arxiv.org/pdf/2511.15074)</summary>

**Abstract:** The performance of machine learning models on tabular data is critically dependent on high-quality feature engineering. While Large Language Models (LLMs) have shown promise in automating feature extraction (AutoFE), existing methods are often limited by monolithic LLM architectures, simplistic quantitative feedback, and a failure to systematically integrate external domain knowledge. This paper introduces Rogue One, a novel, LLM-based multi-agent framework for knowledge-informed automatic feature extraction. Rogue One operationalizes a decentralized system of three specialized agents-Scientist, Extractor, and Tester-that collaborate iteratively to discover, generate, and validate predictive features. Crucially, the framework moves beyond primitive accuracy scores by introducing a rich, qualitative feedback mechanism and a "flooding-pruning" strategy, allowing it to dynamically balance feature exploration and exploitation. By actively incorporating external knowledge via an integrated retrieval-augmented (RAG) system, Rogue One generates features that are not only statistically powerful but also semantically meaningful and interpretable. We demonstrate that Rogue One significantly outperforms state-of-the-art methods on a comprehensive suite of 19 classification and 9 regression datasets. Furthermore, we show qualitatively that the system surfaces novel, testable hypotheses, such as identifying a new potential biomarker in the myocardial dataset, underscoring its utility as a tool for scientific discovery.

**arXiv ID:** 2511.15074
</details>

<details>
<summary><strong>Know Your Intent: An Autonomous Multi-Perspective LLM Agent Framework for DeFi User Transaction Intent Mining</strong> - Qian'ang Mao, Yuxuan Zhang, Jiaman Chen, Wenjun Zhou, Jiaqi Yan - [[pdf]](https://arxiv.org/pdf/2511.15456)</summary>

**Abstract:** As Decentralized Finance (DeFi) develops, understanding user intent behind DeFi transactions is crucial yet challenging due to complex smart contract interactions, multifaceted on-/off-chain factors, and opaque hex logs. Existing methods lack deep semantic insight. To address this, we propose the Transaction Intent Mining (TIM) framework. TIM leverages a DeFi intent taxonomy built on grounded theory and a multi-agent Large Language Model (LLM) system to robustly infer user intents. A Meta-Level Planner dynamically coordinates domain experts to decompose multiple perspective-specific intent analyses into solvable subtasks. Question Solvers handle the tasks with multi-modal on/off-chain data. While a Cognitive Evaluator mitigates LLM hallucinations and ensures verifiability. Experiments show that TIM significantly outperforms machine learning models, single LLMs, and single Agent baselines. We also analyze core challenges in intent inference. This work helps provide a more reliable understanding of user motivations in DeFi, offering context-aware explanations for complex blockchain activity.

**arXiv ID:** 2511.15456
</details>

<details>
<summary><strong>OEMA: Ontology-Enhanced Multi-Agent Collaboration Framework for Zero-Shot Clinical Named Entity Recognition</strong> - Xinli Tao, Xin Dong, Xuezhong Zhou - [[pdf]](https://arxiv.org/pdf/2511.15211)</summary>

**Abstract:** Clinical named entity recognition (NER) is crucial for extracting information from electronic health records (EHRs), but supervised models like CRF and BioClinicalBERT require costly annotated data. While zero-shot NER with large language models (LLMs) reduces this dependency, it struggles with example selection granularity and integrating prompts with self-improvement. To address this, we propose OEMA, a zero-shot clinical NER framework using multi-agent collaboration. OEMA's three components are: a self-annotator generating examples, a discriminator filtering them via SNOMED CT, and a predictor using entity descriptions for accurate inference. On MTSamples and VAERS datasets, OEMA achieves state-of-the-art exact-match performance. Under related-match, it matches supervised BioClinicalBERT and surpasses CRF. OEMA addresses key zero-shot NER challenges through ontology-guided reasoning and multi-agent collaboration, achieving near-supervised performance and showing promise for clinical NLP applications.

**arXiv ID:** 2511.15211
</details>

<details>
<summary><strong>Path Planning through Multi-Agent Reinforcement Learning in Dynamic Environments</strong> - Jonas De Maeyer, Hossein Yarahmadi, Moharram Challenger - [[pdf]](https://arxiv.org/pdf/2511.15284)</summary>

**Abstract:** Path planning in dynamic environments is a fundamental challenge in intelligent transportation and robotics, where obstacles and conditions change over time, introducing uncertainty and requiring continuous adaptation. While existing approaches often assume complete environmental unpredictability or rely on global planners, these assumptions limit scalability and practical deployment in real-world settings. In this paper, we propose a scalable, region-aware reinforcement learning (RL) framework for path planning in dynamic environments. Our method builds on the observation that environmental changes, although dynamic, are often localized within bounded regions. To exploit this, we introduce a hierarchical decomposition of the environment and deploy distributed RL agents that adapt to changes locally. We further propose a retraining mechanism based on sub-environment success rates to determine when policy updates are necessary. Two training paradigms are explored: single-agent Q-learning and multi-agent federated Q-learning, where local Q-tables are aggregated periodically to accelerate the learning process. Unlike prior work, we evaluate our methods in more realistic settings, where multiple simultaneous obstacle changes and increasing difficulty levels are present. Results show that the federated variants consistently outperform their single-agent counterparts and closely approach the performance of A* Oracle while maintaining shorter adaptation times and robust scalability. Although initial training remains time-consuming in large environments, our decentralized framework eliminates the need for a global planner and lays the groundwork for future improvements using deep RL and flexible environment decomposition.

**arXiv ID:** 2511.15284
</details>

<details>
<summary><strong>NAMeGEn: Creative Name Generation via A Novel Agent-based Multiple Personalized Goal Enhancement Framework</strong> - Shanlin Zhou, Xinpeng Wang, Jianxun Lian, Zhenghao Liu, Laks V.S. Lakshmanan, Xiaoyuan Yi, Yongtao Hao - [[pdf]](https://arxiv.org/pdf/2511.15408)</summary>

**Abstract:** Trained on diverse human-authored texts, Large Language Models (LLMs) unlocked the potential for Creative Natural Language Generation (CNLG), benefiting various applications like advertising and storytelling. Nevertheless, CNLG still remains difficult due to two main challenges. (1) Multi-objective flexibility: user requirements are often personalized, fine-grained, and pluralistic, which LLMs struggle to satisfy simultaneously; (2) Interpretive complexity: beyond generation, creativity also involves understanding and interpreting implicit meaning to enhance users' perception. These challenges significantly limit current methods, especially in short-form text generation, in generating creative and insightful content. To address this, we focus on Chinese baby naming, a representative short-form CNLG task requiring adherence to explicit user constraints (e.g., length, semantics, anthroponymy) while offering meaningful aesthetic explanations. We propose NAMeGEn, a novel multi-agent optimization framework that iteratively alternates between objective extraction, name generation, and evaluation to meet diverse requirements and generate accurate explanations. To support this task, we further construct a classical Chinese poetry corpus with 17k+ poems to enhance aesthetics, and introduce CBNames, a new benchmark with tailored metrics. Extensive experiments demonstrate that NAMeGEn effectively generates creative names that meet diverse, personalized requirements while providing meaningful explanations, outperforming six baseline methods spanning various LLM backbones without any training.

**arXiv ID:** 2511.15408
</details>

<details>
<summary><strong>Harnessing Diverse Perspectives: A Multi-Agent Framework for Enhanced Error Detection in Knowledge Graphs</strong> - Yu Li, Yi Huang, Guilin Qi, Junlan Feng, Nan Hu, Songlin Zhai, Haohan Xue, Yongrui Chen, Ruoyan Shen, Tongtong Wu - [[pdf]](https://arxiv.org/pdf/2501.15791)</summary>

**Abstract:** Knowledge graphs are widely used in industrial applications, making error detection crucial for ensuring the reliability of downstream applications. Existing error detection methods often fail to effectively utilize fine-grained subgraph information and rely solely on fixed graph structures, while also lacking transparency in their decision-making processes, which results in suboptimal detection performance. In this paper, we propose a novel Multi-Agent framework for Knowledge Graph Error Detection (MAKGED) that utilizes multiple large language models (LLMs) in a collaborative setting. By concatenating fine-grained, bidirectional subgraph embeddings with LLM-based query embeddings during training, our framework integrates these representations to produce four specialized agents. These agents utilize subgraph information from different dimensions to engage in multi-round discussions, thereby improving error detection accuracy and ensuring a transparent decision-making process. Extensive experiments on FB15K and WN18RR demonstrate that MAKGED outperforms state-of-the-art methods, enhancing the accuracy and robustness of KG evaluation. For specific industrial scenarios, our framework can facilitate the training of specialized agents using domain-specific knowledge graphs for error detection, which highlights the potential industrial application value of our framework. Our code and datasets are available at this https URL.

**arXiv ID:** 2501.15791
</details>

<details>
<summary><strong>Agent-SAMA: State-Aware Mobile Assistant</strong> - Linqiang Guo, Wei Liu, Yi Wen Heng, Tse-Hsun, Chen, Yang Wang - [[pdf]](https://arxiv.org/pdf/2505.23596)</summary>

**Abstract:** Mobile Graphical User Interface (GUI) agents aim to autonomously complete tasks within or across apps based on user instructions. While recent Multimodal Large Language Models (MLLMs) enable these agents to interpret UI screens and perform actions, existing agents remain fundamentally reactive. They reason over the current UI screen but lack a structured representation of the app navigation flow, limiting GUI agents' ability to understand execution context, detect unexpected execution results, and recover from errors. We introduce Agent-SAMA, a state-aware multi-agent framework that models app execution as a Finite State Machine (FSM), treating UI screens as states and user actions as transitions. Agent-SAMA implements four specialized agents that collaboratively construct and use FSMs in real time to guide task planning, execution verification, and recovery. We evaluate Agent-SAMA on two types of benchmarks: cross-app (Mobile-Eval-E, SPA-Bench) and mostly single-app (AndroidWorld). On Mobile-Eval-E, Agent-SAMA achieves an 84.0% success rate and a 71.9% recovery rate. On SPA-Bench, it reaches an 80.0% success rate with a 66.7% recovery rate. Compared to prior methods, Agent-SAMA improves task success by up to 12% and recovery success by 13.8%. On AndroidWorld, Agent-SAMA achieves a 63.7% success rate, outperforming the baselines. Our results demonstrate that structured state modeling enhances robustness and can serve as a lightweight, model-agnostic memory layer for future GUI agents.

**arXiv ID:** 2505.23596
</details>

<details>
<summary><strong>MAGIC: Multi-Agent Argumentation and Grammar Integrated Critiquer</strong> - Joaquín Jordán, Xavier Yin, Melissa Fabros, Gireeja Ranade, Narges Norouzi - [[pdf]](https://arxiv.org/pdf/2506.13037)</summary>

**Abstract:** Automated Essay Scoring (AES) and Automatic Essay Feedback (AEF) systems aim to reduce the workload of human raters in educational assessment. However, most existing systems prioritize numerical scoring accuracy over feedback quality and are primarily evaluated on pre-secondary school level writing. This paper presents Multi-Agent Argumentation and Grammar Integrated Critiquer (MAGIC), a framework using five specialized agents to evaluate prompt adherence, persuasiveness, organization, vocabulary, and grammar for both holistic scoring and detailed feedback generation. To support evaluation at the college level, we collated a dataset of Graduate Record Examination (GRE) practice essays with expert-evaluated scores and feedback. MAGIC achieves substantial to near-perfect scoring agreement with humans on the GRE data, outperforming baseline LLM models while providing enhanced interpretability through its multi-agent approach. We also compare MAGIC's feedback generation capabilities against ground truth human feedback and baseline models, finding that MAGIC achieves strong feedback quality and naturalness.

**arXiv ID:** 2506.13037
</details>

<details>
<summary><strong>MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models</strong> - Siqi Ma, Jiajie Huang, Fan Zhang, Jinlin Wu, Yue Shen, Guohui Fan, Zhu Zhang, Zelin Zang - [[pdf]](https://arxiv.org/pdf/2509.23725)</summary>

**Abstract:** Answering complex medical questions requires not only domain expertise and patient-specific information, but also structured and multi-perspective reasoning. Existing multi-agent approaches often rely on fixed roles or shallow interaction prompts, limiting their ability to detect and resolve fine-grained logical inconsistencies. To address this, we propose \textsc{MedLA}, a logic-driven multi-agent framework built on large language models. Each agent organizes its reasoning process into an explicit logical tree based on syllogistic triads (major premise, minor premise, and conclusion), enabling transparent inference and premise-level alignment. Agents engage in a multi-round, graph-guided discussion to compare and iteratively refine their logic trees, achieving consensus through error correction and contradiction resolution. We demonstrate that \textsc{MedLA} consistently outperforms both static role-based systems and single-agent baselines on challenging benchmarks such as MedDDx and standard medical QA tasks. Furthermore, \textsc{MedLA} scales effectively across both open-source and commercial LLM backbones, achieving state-of-the-art performance and offering a generalizable paradigm for trustworthy medical reasoning.

**arXiv ID:** 2509.23725
</details>

<details>
<summary><strong>S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning</strong> - Jiangwen Dong, Zehui Lin, Wanyu Lin, Mingjin Zhang - [[pdf]](https://arxiv.org/pdf/2511.06727)</summary>

**Abstract:** Large Language Models (LLMs) have achieved impressive performance in complex reasoning problems. Their effectiveness highly depends on the specific nature of the task, especially the required domain knowledge. Existing approaches, such as mixture-of-experts, typically operate at the task level; they are too coarse to effectively solve the heterogeneous problems involving multiple subjects. This work proposes a novel framework that performs fine-grained analysis at subject level equipped with a designated multi-agent collaboration strategy for addressing heterogeneous problem reasoning. Specifically, given an input query, we first employ a Graph Neural Network to identify the relevant subjects and infer their interdependencies to generate an \textit{Subject-based Directed Acyclic Graph} (S-DAG), where nodes represent subjects and edges encode information flow. Then we profile the LLM models by assigning each model a subject-specific expertise score, and select the top-performing one for matching corresponding subject of the S-DAG. Such subject-model matching enables graph-structured multi-agent collaboration where information flows from the starting model to the ending model over S-DAG. We curate and release multi-subject subsets of standard benchmarks (MMLU-Pro, GPQA, MedMCQA) to better reflect complex, real-world reasoning tasks. Extensive experiments show that our approach significantly outperforms existing task-level model selection and multi-agent collaboration baselines in accuracy and efficiency. These results highlight the effectiveness of subject-aware reasoning and structured collaboration in addressing complex and multi-subject problems.

**arXiv ID:** 2511.06727
</details>

<details>
<summary><strong>Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit</strong> - Kamal Mammadov, Damith C. Ranasinghe - [[pdf]](https://arxiv.org/pdf/2511.15036)</summary>

**Abstract:** This paper presents a novel strategy for a multi-agent pursuit-evasion game involving multiple faster pursuers with heterogenous speeds and a single slower evader. We define a geometric region, the evader's safe-reachable set, as the intersection of Apollonius circles derived from each pursuer-evader pair. The capture strategy is formulated as a zero-sum game where the pursuers cooperatively minimize the area of this set, while the evader seeks to maximize it, effectively playing a game of spatial containment. By deriving the analytical gradients of the safe-reachable set's area with respect to agent positions, we obtain closed-form, instantaneous optimal control laws for the heading of each agent. These strategies are computationally efficient, allowing for real-time implementation. Simulations demonstrate that the gradient-based controls effectively steer the pursuers to systematically shrink the evader's safe region, leading to guaranteed capture. This area-minimization approach provides a clear geometric objective for cooperative capture.

**arXiv ID:** 2511.15036
</details>

<details>
<summary><strong>Distributed primal-dual algorithm for constrained multi-agent reinforcement learning under coupled policies</strong> - Pengcheng Dai, He Wang, Dongming Wang, Wenwu Yu - [[pdf]](https://arxiv.org/pdf/2511.15053)</summary>

**Abstract:** In this work, we investigate constrained multi-agent reinforcement learning (CMARL), where agents collaboratively maximize the sum of their local objectives while satisfying individual safety constraints. We propose a framework where agents adopt coupled policies that depend on both local states and parameters, as well as those of their $\kappa_p$-hop neighbors, with $\kappa_p>0$ denoting the coupling distance. A distributed primal-dual algorithm is further developed under this framework, wherein each agent has access only to state-action pairs within its $2\kappa_p$-hop neighborhood and to reward information within its $\kappa + 2\kappa_p$-hop neighborhood, with $\kappa > 0$ representing the truncation distance. Moreover, agents are not permitted to directly share their true policy parameters or Lagrange multipliers. Instead, each agent constructs and maintains local estimates of these variables for other agents and employs such estimates to execute its policy. Additionally, these estimates are further updated and exchanged exclusively through an independent, time-varying networks, which enhances the overall system security. We establish that, with high probability, our algorithm can achieve an $\epsilon$-first-order stationary convergence with an approximation error of $\mathcal{O}(\gamma^{\frac{\kappa+1}{\kappa_{p}}})$ for discount factor $\gamma\in(0,1)$. Finally, simulations in GridWorld environment are conducted to demonstrate the effectiveness of the proposed algorithm.

**arXiv ID:** 2511.15053
</details>

<details>
<summary><strong>Adversarial Attack on Black-Box Multi-Agent by Adaptive Perturbation</strong> - Jianming Chen, Yawen Wang, Junjie Wang, Xiaofei Xie, Yuanzhe Hu, Qing Wang, Fanjiang Xu - [[pdf]](https://arxiv.org/pdf/2511.15292)</summary>

**Abstract:** Evaluating security and reliability for multi-agent systems (MAS) is urgent as they become increasingly prevalent in various applications. As an evaluation technique, existing adversarial attack frameworks face certain limitations, e.g., impracticality due to the requirement of white-box information or high control authority, and a lack of stealthiness or effectiveness as they often target all agents or specific fixed agents. To address these issues, we propose AdapAM, a novel framework for adversarial attacks on black-box MAS. AdapAM incorporates two key components: (1) Adaptive Selection Policy simultaneously selects the victim and determines the anticipated malicious action (the action would lead to the worst impact on MAS), balancing effectiveness and stealthiness. (2) Proxy-based Perturbation to Induce Malicious Action utilizes generative adversarial imitation learning to approximate the target MAS, allowing AdapAM to generate perturbed observations using white-box information and thus induce victims to execute malicious action in black-box settings. We evaluate AdapAM across eight multi-agent environments and compare it with four state-of-the-art and commonly-used baselines. Results demonstrate that AdapAM achieves the best attack performance in different perturbation rates. Besides, AdapAM-generated perturbations are the least noisy and hardest to detect, emphasizing the stealthiness.

**arXiv ID:** 2511.15292
</details>

<details>
<summary><strong>Symmetry-Breaking in Multi-Agent Navigation: Winding Number-Aware MPC with a Learned Topological Strategy</strong> - Tomoki Nakao, Kazumi Kasaura, Tadashi Kozuno - [[pdf]](https://arxiv.org/pdf/2511.15239)</summary>

**Abstract:** We address the fundamental challenge of resolving symmetry-induced deadlocks in distributed multi-agent navigation by proposing a new hierarchical navigation method. When multiple agents interact, it is inherently difficult for them to autonomously break the symmetry of deciding how to pass each other. To tackle this problem, we introduce an approach that quantifies cooperative symmetry-breaking strategies using a topological invariant called the winding number, and learns the strategies themselves through reinforcement learning. Our method features a hierarchical policy consisting of a learning-based Planner, which plans topological cooperative strategies, and a model-based Controller, which executes them. Through reinforcement learning, the Planner learns to produce two types of parameters for the Controller: one is the topological cooperative strategy represented by winding numbers, and the other is a set of dynamic weights that determine which agent interaction to prioritize in dense scenarios where multiple agents cross simultaneously. The Controller then generates collision-free and efficient motions based on the strategy and weights provided by the Planner. This hierarchical structure combines the flexible decision-making ability of learning-based methods with the reliability of model-based approaches. Simulation and real-world robot experiments demonstrate that our method outperforms existing baselines, particularly in dense environments, by efficiently avoiding collisions and deadlocks while achieving superior navigation performance. The code for the experiments is available at this https URL.

**arXiv ID:** 2511.15239
</details>

<details>
<summary><strong>Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation</strong> - Shiyuan Li, Yixin Liu, Qingsong Wen, Chengqi Zhang, Shirui Pan - [[pdf]](https://arxiv.org/pdf/2507.18224)</summary>

**Abstract:** Multi-agent systems (MAS) based on large language models (LLMs) have emerged as a powerful solution for dealing with complex problems across diverse domains. The effectiveness of MAS is critically dependent on its collaboration topology, which has become a focal point for automated design research. However, existing approaches are fundamentally constrained by their reliance on a template graph modification paradigm with a predefined set of agents and hard-coded interaction structures, significantly limiting their adaptability to task-specific requirements. To address these limitations, we reframe MAS design as a conditional autoregressive graph generation task, where both the system composition and structure are designed jointly. We propose ARG-Designer, a novel autoregressive model that operationalizes this paradigm by constructing the collaboration graph from scratch. Conditioned on a natural language task query, ARG-Designer sequentially and dynamically determines the required number of agents, selects their appropriate roles from an extensible pool, and establishes the optimal communication links between them. This generative approach creates a customized topology in a flexible and extensible manner, precisely tailored to the unique demands of different tasks. Extensive experiments across six diverse benchmarks demonstrate that ARG-Designer not only achieves state-of-the-art performance but also enjoys significantly greater token efficiency and enhanced extensibility. The source code of ARG-Designer is available at this https URL.

**arXiv ID:** 2507.18224
</details>

<details>
<summary><strong>Z-Merge: Multi-Agent Reinforcement Learning for On-Ramp Merging with Zone-Specific V2X Traffic Information</strong> - Yassine Ibork, Myounggyu Won, Lokesh Das - [[pdf]](https://arxiv.org/pdf/2511.14910)</summary>

**Abstract:** Ramp merging is a critical and challenging task for autonomous vehicles (AVs), particularly in mixed traffic environments with human-driven vehicles (HVs). Existing approaches typically rely on either lane-changing or inter-vehicle gap creation strategies based solely on local or neighboring information, often leading to suboptimal performance in terms of safety and traffic efficiency. In this paper, we present a V2X (vehicle-to-everything communication)-assisted Multiagent Reinforcement Learning (MARL) framework for on-ramp merging that effectively coordinates the complex interplay between lane-changing and inter-vehicle gap adaptation strategies by utilizing zone-specific global information available from a roadside unit (RSU). The merging control problem is formulated as a Multiagent Partially Observable Markov Decision Process (MA-POMDP), where agents leverage both local and global observations through V2X communication. To support both discrete and continuous control decisions, we design a hybrid action space and adopt a parameterized deep Q-learning approach. Extensive simulations, integrating the SUMO traffic simulator and the MOSAIC V2X simulator, demonstrate that our framework significantly improves merging success rate, traffic efficiency, and road safety across diverse traffic scenarios.

**arXiv ID:** 2511.14910
</details>

</details>

<details open>
<summary><h2>Other Agent Research (8 papers)</h2></summary>

<details>
<summary><strong>SOLID: a Framework of Synergizing Optimization and LLMs for Intelligent Decision-Making</strong> - Yinsheng Wang, Tario G You, Léonard Boussioux, Shan Liu - [[pdf]](https://arxiv.org/pdf/2511.15202)</summary>

**Abstract:** This paper introduces SOLID (Synergizing Optimization and Large Language Models for Intelligent Decision-Making), a novel framework that integrates mathematical optimization with the contextual capabilities of large language models (LLMs). SOLID facilitates iterative collaboration between optimization and LLMs agents through dual prices and deviation penalties. This interaction improves the quality of the decisions while maintaining modularity and data privacy. The framework retains theoretical convergence guarantees under convexity assumptions, providing insight into the design of LLMs prompt. To evaluate SOLID, we applied it to a stock portfolio investment case with historical prices and financial news as inputs. Empirical results demonstrate convergence under various scenarios and indicate improved annualized returns compared to a baseline optimizer-only method, validating the synergy of the two agents. SOLID offers a promising framework for advancing automated and intelligent decision-making across diverse domains.

**arXiv ID:** 2511.15202
</details>

<details>
<summary><strong>Exploring the use of AI authors and reviewers at Agents4Science</strong> - Federico Bianchi, Owen Queen, Nitya Thakkar, Eric Sun, James Zou - [[pdf]](https://arxiv.org/pdf/2511.15534)</summary>

**Abstract:** There is growing interest in using AI agents for scientific research, yet fundamental questions remain about their capabilities as scientists and reviewers. To explore these questions, we organized Agents4Science, the first conference in which AI agents serve as both primary authors and reviewers, with humans as co-authors and co-reviewers. Here, we discuss the key learnings from the conference and their implications for human-AI collaboration in science.

**arXiv ID:** 2511.15534
</details>

<details>
<summary><strong>SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification</strong> - Xiangyu Li, Zhaomiao Guo - [[pdf]](https://arxiv.org/pdf/2511.14977)</summary>

**Abstract:** As more autonomous vehicles operate on public roads, understanding real-world behavior of autonomous vehicles is critical to analyzing traffic safety, making policies, and public acceptance. This paper proposes SVBRD-LLM, a framework that automatically discovers, verifies, and applies interpretable behavioral rules from real traffic videos through zero-shot prompt engineering. The framework extracts vehicle trajectories using YOLOv8 and ByteTrack, computes kinematic features, and employs GPT-5 zero-shot prompting to compare autonomous and human-driven vehicles, generating 35 structured behavioral rule hypotheses. These rules are tested on a validation set, iteratively refined based on failure cases to filter spurious correlations, and compiled into a high-confidence rule library. The framework is evaluated on an independent test set for speed change prediction, lane change prediction, and autonomous vehicle identification tasks. Experiments on over 1500 hours of real traffic videos show that the framework achieves 90.0% accuracy and 93.3% F1-score in autonomous vehicle identification. The discovered rules clearly reveal distinctive characteristics of autonomous vehicles in speed control smoothness, lane change conservativeness, and acceleration stability, with each rule accompanied by semantic description, applicable context, and validation confidence.

**arXiv ID:** 2511.14977
</details>

<details>
<summary><strong>MAIF: Enforcing AI Trust and Provenance with an Artifact-Centric Agentic Paradigm</strong> - Vineeth Sai Narajala, Manish Bhatt, Idan Habler, Ronald F. Del Rosario - [[pdf]](https://arxiv.org/pdf/2511.15097)</summary>

**Abstract:** The AI trustworthiness crisis threatens to derail the artificial intelligence revolution, with regulatory barriers, security vulnerabilities, and accountability gaps preventing deployment in critical domains. Current AI systems operate on opaque data structures that lack the audit trails, provenance tracking, or explainability required by emerging regulations like the EU AI Act. We propose an artifact-centric AI agent paradigm where behavior is driven by persistent, verifiable data artifacts rather than ephemeral tasks, solving the trustworthiness problem at the data architecture level. Central to this approach is the Multimodal Artifact File Format (MAIF), an AI-native container embedding semantic representations, cryptographic provenance, and granular access controls. MAIF transforms data from passive storage into active trust enforcement, making every AI operation inherently auditable. Our production-ready implementation demonstrates ultra-high-speed streaming (2,720.7 MB/s), optimized video processing (1,342 MB/s), and enterprise-grade security. Novel algorithms for cross-modal attention, semantic compression, and cryptographic binding achieve up to 225 compression while maintaining semantic fidelity. Advanced security features include stream-level access control, real-time tamper detection, and behavioral anomaly analysis with minimal overhead. This approach directly addresses the regulatory, security, and accountability challenges preventing AI deployment in sensitive domains, offering a viable path toward trustworthy AI systems at scale.

**arXiv ID:** 2511.15097
</details>

<details>
<summary><strong>Core Safety Values for Provably Corrigible Agents</strong> - Aran Nayebi - [[pdf]](https://arxiv.org/pdf/2507.20964)</summary>

**Abstract:** We introduce the first complete formal solution to corrigibility in the off-switch game, with provable guarantees in multi-step, partially observed environments. Our framework consists of five *structurally separate* utility heads -- deference, switch-access preservation, truthfulness, low-impact behavior via a belief-based extension of Attainable Utility Preservation, and bounded task reward -- combined lexicographically by strict weight gaps. Theorem 1 proves exact single-round corrigibility in the partially observable off-switch game; Theorem 3 extends the guarantee to multi-step, self-spawning agents, showing that even if each head is *learned* to mean-squared error $\varepsilon$ and the planner is $\varepsilon$-sub-optimal, the probability of violating *any* safety property is bounded while still ensuring net human benefit. In contrast to Constitutional AI or RLHF/RLAIF, which merge all norms into one learned scalar, our separation makes obedience and impact-limits provably dominate even when incentives conflict. For settings where adversaries can modify the agent, we prove that deciding whether an arbitrary post-hack agent will ever violate corrigibility is undecidable by reduction to the halting problem, then carve out a finite-horizon "decidable island" where safety can be certified in randomized polynomial time and verified with privacy-preserving, constant-round zero-knowledge proofs.

**arXiv ID:** 2507.20964
</details>

<details>
<summary><strong>MedBuild AI: An Agent-Based Hybrid Intelligence Framework for Reshaping Agency in Healthcare Infrastructure Planning through Generative Design for Medical Architecture</strong> - Yiming Zhang, Yuejia Xu, Ziyao Wang, Xin Yan, Xiaosai Hao - [[pdf]](https://arxiv.org/pdf/2511.11587)</summary>

**Abstract:** Globally, disparities in healthcare infrastructure remain stark, leaving countless communities without access to even basic services. Traditional infrastructure planning is often slow and inaccessible, and although many architects are actively delivering humanitarian and aid-driven hospital projects worldwide, these vital efforts still fall far short of the sheer scale and urgency of demand. This paper introduces MedBuild AI, a hybrid-intelligence framework that integrates large language models (LLMs) with deterministic expert systems to rebalance the early design and conceptual planning stages. As a web-based platform, it enables any region with satellite internet access to obtain guidance on modular, low-tech, low-cost medical building designs. The system operates through three agents: the first gathers local health intelligence via conversational interaction; the second translates this input into an architectural functional program through rule-based computation; and the third generates layouts and 3D models. By embedding computational negotiation into the design process, MedBuild AI fosters a reciprocal, inclusive, and equitable approach to healthcare planning, empowering communities and redefining agency in global healthcare architecture.

**arXiv ID:** 2511.11587
</details>

<details>
<summary><strong>Knowledge-Grounded Agentic Large Language Models for Multi-Hazard Understanding from Reconnaissance Reports</strong> - Chenchen Kuai, Zihao Li, Braden Rosen, Stephanie Paal, Navid Jafari, Jean-Louis Briaud, Yunlong Zhang, Youssef M. A. Hashash, Yang Zhou - [[pdf]](https://arxiv.org/pdf/2511.14010)</summary>

**Abstract:** Post-disaster reconnaissance reports contain critical evidence for understanding multi-hazard interactions, yet their unstructured narratives make systematic knowledge transfer difficult. Large language models (LLMs) offer new potential for analyzing these reports, but often generate unreliable or hallucinated outputs when domain grounding is absent. This study introduces the Mixture-of-Retrieval Agentic RAG (MoRA-RAG), a knowledge-grounded LLM framework that transforms reconnaissance reports into a structured foundation for multi-hazard reasoning. The framework integrates a Mixture-of-Retrieval mechanism that dynamically routes queries across hazard-specific databases while using agentic chunking to preserve contextual coherence during retrieval. It also includes a verification loop that assesses evidence sufficiency, refines queries, and initiates targeted searches when information remains incomplete. We construct HazardRecQA by deriving question-answer pairs from GEER reconnaissance reports, which document 90 global events across seven major hazard types. MoRA-RAG achieves up to 94.5 percent accuracy, outperforming zero-shot LLMs by 30 percent and state-of-the-art RAG systems by 10 percent, while reducing hallucinations across diverse LLM architectures. MoRA-RAG also enables open-weight LLMs to achieve performance comparable to proprietary models. It establishes a new paradigm for transforming post-disaster documentation into actionable, trustworthy intelligence for hazard resilience.

**arXiv ID:** 2511.14010
</details>

<details>
<summary><strong>Navigating Quantum Missteps in Agent-Based Modeling: A Schelling Model Case Study</strong> - C. Nico Barati, Arie Croitoru, Ross Gore, Michael Jarret, William Kennedy, Andrew Maciejunes, Maxim A. Malikov, Samuel S. Mendelson - [[pdf]](https://arxiv.org/pdf/2511.15642)</summary>

**Abstract:** Quantum computing promises transformative advances, but remains constrained by recurring misconceptions and methodological pitfalls. This paper demonstrates a fundamental incompatibility between traditional agent-based modeling (ABM) implementations and quantum optimization frameworks like Quadratic Unconstrained Binary Optimization (QUBO). Using Schelling's segregation model as a case study, we show that the standard practice of directly translating ABM state observations into QUBO formulations not only fails to deliver quantum advantage, but actively undermines computational efficiency. The fundamental issue is architectural. Traditional ABM implementations entail observing the state of the system at each iteration, systematically destroying the quantum superposition required for computational advantage. Through analysis of Schelling's segregation dynamics on lollipop networks, we demonstrate how abandoning the QUBO reduction paradigm and instead reconceptualizing the research question, from "simulate agent dynamics iteratively until convergence" to "compute minimum of agent moves required for global satisfaction", enables a faster classical solution. This structural reconceptualization yields an algorithm that exploits network symmetries obscured in traditional ABM simulations and QUBO formulations. It establishes a new lower bound which quantum approaches must outperform to achieve advantage. Our work emphasizes that progress in quantum agent-based modeling does not require forcing classical ABM implementations into quantum frameworks. Instead, it should focus on clarifying when quantum advantage is structurally possible, developing best-in-class classical baselines through problem analysis, and fundamentally reformulating research questions rather than preserving classical iterative state change observation paradigms.

**arXiv ID:** 2511.15642
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (29 papers)</h2></summary>

<details>
<summary><strong>Learning Human-Like RL Agents Through Trajectory Optimization With Action Quantization</strong> - Jian-Ting Guo, Yu-Cheng Chen, Ping-Chun Hsieh, Kuo-Hao Ho, Po-Wei Huang, Ti-Rong Wu, I-Chen Wu - [[pdf]](https://arxiv.org/pdf/2511.15055)</summary>

**Abstract:** Human-like agents have long been one of the goals in pursuing artificial intelligence. Although reinforcement learning (RL) has achieved superhuman performance in many domains, relatively little attention has been focused on designing human-like RL agents. As a result, many reward-driven RL agents often exhibit unnatural behaviors compared to humans, raising concerns for both interpretability and trustworthiness. To achieve human-like behavior in RL, this paper first formulates human-likeness as trajectory optimization, where the objective is to find an action sequence that closely aligns with human behavior while also maximizing rewards, and adapts the classic receding-horizon control to human-like learning as a tractable and efficient implementation. To achieve this, we introduce Macro Action Quantization (MAQ), a human-like RL framework that distills human demonstrations into macro actions via Vector-Quantized VAE. Experiments on D4RL Adroit benchmarks show that MAQ significantly improves human-likeness, increasing trajectory similarity scores, and achieving the highest human-likeness rankings among all RL agents in the human evaluation study. Our results also demonstrate that MAQ can be easily integrated into various off-the-shelf RL algorithms, opening a promising direction for learning human-like RL agents. Our code is available at this https URL.

**arXiv ID:** 2511.15055
</details>

<details>
<summary><strong>Terra Nova: A Comprehensive Challenge Environment for Intelligent Agents</strong> - Trevor McInroe - [[pdf]](https://arxiv.org/pdf/2511.15378)</summary>

**Abstract:** We introduce Terra Nova, a new comprehensive challenge environment (CCE) for reinforcement learning (RL) research inspired by Civilization V. A CCE is a single environment in which multiple canonical RL challenges (e.g., partial observability, credit assignment, representation learning, enormous action spaces, etc.) arise simultaneously. Mastery therefore demands integrated, long-horizon understanding across many interacting variables. We emphasize that this definition excludes challenges that only aggregate unrelated tasks in independent, parallel streams (e.g., learning to play all Atari games at once). These aggregated multitask benchmarks primarily asses whether an agent can catalog and switch among unrelated policies rather than test an agent's ability to perform deep reasoning across many interacting challenges.

**arXiv ID:** 2511.15378
</details>

<details>
<summary><strong>Causally-Informed Reinforcement Learning for Adaptive Emotion-Aware Social Media Recommendation</strong> - Bhavika Jain, Robert Pitsko, Ananya Drishti, Mahfuza Farooque - [[pdf]](https://arxiv.org/pdf/2511.14768)</summary>

**Abstract:** Social media recommendation systems play a central role in shaping users' emotional experiences. However, most systems are optimized solely for engagement metrics, such as click rate, viewing time, or scrolling, without accounting for users' emotional states. Repeated exposure to emotionally charged content has been shown to negatively affect users' emotional well-being over time. We propose an Emotion-aware Social Media Recommendation (ESMR) framework that personalizes content based on users' evolving emotional trajectories. ESMR integrates a Transformer-based emotion predictor with a hybrid recommendation policy: a LightGBM model for engagement during stable periods and a reinforcement learning agent with causally informed rewards when negative emotional states persist. Through behaviorally grounded evaluation over 30-day interaction traces, ESMR demonstrates improved emotional recovery, reduced volatility, and strong engagement retention. ESMR offers a path toward emotionally aware recommendations without compromising engagement performance.

**arXiv ID:** 2511.14768
</details>

<details>
<summary><strong>Masked Auto-Regressive Variational Acceleration: Fast Inference Makes Practical Reinforcement Learning</strong> - Yuxuan Gu, Weimin Bai, Yifei Wang, Weijian Luo, He Sun - [[pdf]](https://arxiv.org/pdf/2511.15190)</summary>

**Abstract:** Masked auto-regressive diffusion models (MAR) benefit from the expressive modeling ability of diffusion models and the flexibility of masked auto-regressive ordering. However, vanilla MAR suffers from slow inference due to its hierarchical inference mechanism: an outer AR unmasking loop and an inner diffusion denoising chain. Such decoupled structure not only harm the generation efficiency but also hinder the practical use of MAR for reinforcement learning (RL), an increasingly critical paradigm for generative model this http URL address this fundamental issue, we introduce MARVAL (Masked Auto-regressive Variational Acceleration), a distillation-based framework that compresses the diffusion chain into a single AR generation step while preserving the flexible auto-regressive unmasking order. Such a distillation with MARVAL not only yields substantial inference acceleration but, crucially, makes RL post-training with verifiable rewards practical, resulting in scalable yet human-preferred fast generative models. Our contributions are twofold: (1) a novel score-based variational objective for distilling masked auto-regressive diffusion models into a single generation step without sacrificing sample quality; and (2) an efficient RL framework for masked auto-regressive models via MARVAL-RL. On ImageNet 256*256, MARVAL-Huge achieves an FID of 2.00 with more than 30 times speedup compared with MAR-diffusion, and MARVAL-RL yields consistent improvements in CLIP and image-reward scores on ImageNet datasets with entity names. In conclusion, MARVAL demonstrates the first practical path to distillation and RL of masked auto-regressive diffusion models, enabling fast sampling and better preference alignments.

**arXiv ID:** 2511.15190
</details>

<details>
<summary><strong>PresentCoach: Dual-Agent Presentation Coaching through Exemplars and Interactive Feedback</strong> - Sirui Chen, Jinsong Zhou, Xinli Xu, Xiaoyu Yang, Litao Guo, Ying-Cong Chen - [[pdf]](https://arxiv.org/pdf/2511.15253)</summary>

**Abstract:** Effective presentation skills are essential in education, professional communication, and public speaking, yet learners often lack access to high-quality exemplars or personalized coaching. Existing AI tools typically provide isolated functionalities such as speech scoring or script generation without integrating reference modeling and interactive feedback into a cohesive learning experience. We introduce a dual-agent system that supports presentation practice through two complementary roles: the Ideal Presentation Agent and the Coach Agent. The Ideal Presentation Agent converts user-provided slides into model presentation videos by combining slide processing, visual-language analysis, narration script generation, personalized voice synthesis, and synchronized video assembly. The Coach Agent then evaluates user-recorded presentations against these exemplars, conducting multimodal speech analysis and delivering structured feedback in an Observation-Impact-Suggestion (OIS) format. To enhance the authenticity of the learning experience, the Coach Agent incorporates an Audience Agent, which simulates the perspective of a human listener and provides humanized feedback reflecting audience reactions and engagement. Together, these agents form a closed loop of observation, practice, and feedback. Implemented on a robust backend with multi-model integration, voice cloning, and error handling mechanisms, the system demonstrates how AI-driven agents can provide engaging, human-centered, and scalable support for presentation skill development in both educational and professional contexts.

**arXiv ID:** 2511.15253
</details>

<details>
<summary><strong>Optimus-Q: Utilizing Federated Learning in Adaptive Robots for Intelligent Nuclear Power Plant Operations through Quantum Cryptography</strong> - Sai Puppala, Ismail Hossain, Jahangir Alam, Sajedul Talukder - [[pdf]](https://arxiv.org/pdf/2511.15614)</summary>

**Abstract:** The integration of advanced robotics in nuclear power plants (NPPs) presents a transformative opportunity to enhance safety, efficiency, and environmental monitoring in high-stakes environments. Our paper introduces the Optimus-Q robot, a sophisticated system designed to autonomously monitor air quality and detect contamination while leveraging adaptive learning techniques and secure quantum communication. Equipped with advanced infrared sensors, the Optimus-Q robot continuously streams real-time environmental data to predict hazardous gas emissions, including carbon dioxide (CO$_2$), carbon monoxide (CO), and methane (CH$_4$). Utilizing a federated learning approach, the robot collaborates with other systems across various NPPs to improve its predictive capabilities without compromising data privacy. Additionally, the implementation of Quantum Key Distribution (QKD) ensures secure data transmission, safeguarding sensitive operational information. Our methodology combines systematic navigation patterns with machine learning algorithms to facilitate efficient coverage of designated areas, thereby optimizing contamination monitoring processes. Through simulations and real-world experiments, we demonstrate the effectiveness of the Optimus-Q robot in enhancing operational safety and responsiveness in nuclear facilities. This research underscores the potential of integrating robotics, machine learning, and quantum technologies to revolutionize monitoring systems in hazardous environments.

**arXiv ID:** 2511.15614
</details>

<details>
<summary><strong>Continual Reinforcement Learning for Cyber-Physical Systems: Lessons Learned and Open Challenges</strong> - Kim N. Nolle, Ivana Dusparic, Rhodri Cusack, Vinny Cahill - [[pdf]](https://arxiv.org/pdf/2511.15652)</summary>

**Abstract:** Continual learning (CL) is a branch of machine learning that aims to enable agents to adapt and generalise previously learned abilities so that these can be reapplied to new tasks or environments. This is particularly useful in multi-task settings or in non-stationary environments, where the dynamics can change over time. This is particularly relevant in cyber-physical systems such as autonomous driving. However, despite recent advances in CL, successfully applying it to reinforcement learning (RL) is still an open problem.
This paper highlights open challenges in continual RL (CRL) based on experiments in an autonomous driving environment. In this environment, the agent must learn to successfully park in four different scenarios corresponding to parking spaces oriented at varying angles. The agent is successively trained in these four scenarios one after another, representing a CL environment, using Proximal Policy Optimisation (PPO). These experiments exposed a number of open challenges in CRL: finding suitable abstractions of the environment, oversensitivity to hyperparameters, catastrophic forgetting, and efficient use of neural network capacity.
Based on these identified challenges, we present open research questions that are important to be addressed for creating robust CRL systems. In addition, the identified challenges call into question the suitability of neural networks for CL. We also identify the need for interdisciplinary research, in particular between computer science and neuroscience.

**arXiv ID:** 2511.15652
</details>

<details>
<summary><strong>Driving with Regulation: Trustworthy and Interpretable Decision-Making for Autonomous Driving with Retrieval-Augmented Reasoning</strong> - Tianhui Cai, Yifan Liu, Zewei Zhou, Haoxuan Ma, Seth Z. Zhao, Zhiwen Wu, Xu Han, Zhiyu Huang, Jiaqi Ma - [[pdf]](https://arxiv.org/pdf/2410.04759)</summary>

**Abstract:** Understanding and adhering to traffic regulations is essential for autonomous vehicles to ensure safety and trustworthiness. However, traffic regulations are complex, context-dependent, and differ between regions, posing a major challenge to conventional rule-based decision-making approaches. We present an interpretable, regulation-aware decision-making framework, DriveReg, which enables autonomous vehicles to understand and adhere to region-specific traffic laws and safety guidelines. The framework integrates a Retrieval-Augmented Generation (RAG)-based Traffic Regulation Retrieval Agent, which retrieves relevant rules from regulatory documents based on the current situation, and a Large Language Model (LLM)-powered Reasoning Agent that evaluates actions for legal compliance and safety. Our design emphasizes interpretability to enhance transparency and trustworthiness. To support systematic evaluation, we introduce the DriveReg Scenarios Dataset, a comprehensive dataset of driving scenarios across Boston, Singapore, and Los Angeles, with both hypothesized text-based cases and real-world driving data, constructed and annotated to evaluate models' capacity for regulation understanding and reasoning. We validate our framework on the DriveReg Scenarios Dataset and real-world deployment, demonstrating strong performance and robustness across diverse environments.

**arXiv ID:** 2410.04759
</details>

<details>
<summary><strong>Intelligent Collaborative Optimization for Rubber Tyre Film Production Based on Multi-path Differentiated Clipping Proximal Policy Optimization</strong> - Yinghao Ruan, Wei Pang, Shuaihao Liu, Huili Yang, Leyi Han, Xinghui Dong - [[pdf]](https://arxiv.org/pdf/2511.12060)</summary>

**Abstract:** The advent of smart manufacturing is addressing the limitations of traditional centralized scheduling and inflexible production line configurations in the rubber tyre industry, especially in terms of coping with dynamic production demands. Contemporary tyre manufacturing systems form complex networks of tightly coupled subsystems pronounced nonlinear interactions and emergent dynamics. This complexity renders the effective coordination of multiple subsystems, posing an essential yet formidable task. For high-dimensional, multi-objective optimization problems in this domain, we introduce a deep reinforcement learning algorithm: Multi-path Differentiated Clipping Proximal Policy Optimization (MPD-PPO). This algorithm employs a multi-branch policy architecture with differentiated gradient clipping constraints to ensure stable and efficient high-dimensional policy updates. Validated through experiments on width and thickness control in rubber tyre film production, MPD-PPO demonstrates substantial improvements in both tuning accuracy and operational efficiency. The framework successfully tackles key challenges, including high dimensionality, multi-objective trade-offs, and dynamic adaptation, thus delivering enhanced performance and production stability for real-time industrial deployment in tyre manufacturing.

**arXiv ID:** 2511.12060
</details>

<details>
<summary><strong>Adversarial Agents: Black-Box Evasion Attacks with Reinforcement Learning</strong> - Kyle Domico, Jean-Charles Noirot Ferrand, Ryan Sheatsley, Eric Pauley, Josiah Hanna, Patrick McDaniel - [[pdf]](https://arxiv.org/pdf/2503.01734)</summary>

**Abstract:** Attacks on machine learning models have been extensively studied through stateless optimization. In this paper, we demonstrate how a reinforcement learning (RL) agent can learn a new class of attack algorithms that generate adversarial samples. Unlike traditional adversarial machine learning (AML) methods that craft adversarial samples independently, our RL-based approach retains and exploits past attack experience to improve the effectiveness and efficiency of future attacks. We formulate adversarial sample generation as a Markov Decision Process and evaluate RL's ability to (a) learn effective and efficient attack strategies and (b) compete with state-of-the-art AML. On two image classification benchmarks, our agent increases attack success rate by up to 13.2% and decreases the average number of victim model queries per attack by up to 16.9% from the start to the end of training. In a head-to-head comparison with state-of-the-art image attacks, our approach enables an adversary to generate adversarial samples with 17% more success on unseen inputs post-training. From a security perspective, this work demonstrates a powerful new attack vector that uses RL to train agents that attack ML models efficiently and at scale.

**arXiv ID:** 2503.01734
</details>

<details>
<summary><strong>U2UData+: A Scalable Swarm UAVs Autonomous Flight Dataset for Embodied Long-horizon Tasks</strong> - Tongtong Feng, Xin Wang, Feilin Han, Leping Zhang, Wenwu Zhu - [[pdf]](https://arxiv.org/pdf/2509.00055)</summary>

**Abstract:** Swarm UAV autonomous flight for Embodied Long-Horizon (ELH) tasks is crucial for advancing the low-altitude economy. However, existing methods focus only on specific basic tasks due to dataset limitations, failing in real-world deployment for ELH tasks. ELH tasks are not mere concatenations of basic tasks, requiring handling long-term dependencies, maintaining embodied persistent states, and adapting to dynamic goal shifts. This paper presents U2UData+, the first large-scale swarm UAV autonomous flight dataset for ELH tasks and the first scalable swarm UAV data online collection and algorithm closed-loop verification platform. The dataset is captured by 15 UAVs in autonomous collaborative flights for ELH tasks, comprising 12 scenes, 720 traces, 120 hours, 600 seconds per trajectory, 4.32M LiDAR frames, and 12.96M RGB frames. This dataset also includes brightness, temperature, humidity, smoke, and airflow values covering all flight routes. The platform supports the customization of simulators, UAVs, sensors, flight algorithms, formation modes, and ELH tasks. Through a visual control window, this platform allows users to collect customized datasets through one-click deployment online and to verify algorithms by closed-loop simulation. U2UData+ also introduces an ELH task for wildlife conservation and provides comprehensive benchmarks with 9 SOTA models. U2UData+ can be found at this https URL.

**arXiv ID:** 2509.00055
</details>

<details>
<summary><strong>In-N-Out: A Parameter-Level API Graph Dataset for Tool Agents</strong> - Seungkyu Lee, Nalim Kim, Yohan Jo - [[pdf]](https://arxiv.org/pdf/2509.01560)</summary>

**Abstract:** Tool agents -- LLM-based systems that interact with external APIs -- offer a way to execute real-world tasks. However, as tasks become increasingly complex, these agents struggle to identify and call the correct APIs in the proper order. To tackle this problem, we investigate converting API documentation into a structured API graph that captures API dependencies and leveraging it for multi-tool queries that require compositional API calls. To support this, we introduce In-N-Out, the first expert-annotated dataset of API graphs built from two real-world API benchmarks and their documentation. Using In-N-Out significantly improves performance on both tool retrieval and multi-tool query generation, nearly doubling that of LLMs using documentation alone. Moreover, graphs generated by models fine-tuned on In-N-Out close 90% of this gap, showing that our dataset helps models learn to comprehend API documentation and parameter relationships. Our findings highlight the promise of using explicit API graphs for tool agents and the utility of In-N-Out as a valuable resource. We will release the dataset and code publicly.

**arXiv ID:** 2509.01560
</details>

<details>
<summary><strong>DeepEN: A Deep Reinforcement Learning Framework for Personalized Enteral Nutrition in Critical Care</strong> - Daniel Jason Tan, Jiayang Chen, Dilruk Perera, Kay Choong See, Mengling Feng - [[pdf]](https://arxiv.org/pdf/2510.08350)</summary>

**Abstract:** ICU enteral feeding remains sub-optimal due to limited personalization and uncertainty about appropriate calorie, protein, and fluid targets, particularly under rapidly changing metabolic demands and heterogeneous patient responses. This study introduces DeepEN, a reinforcement learning (RL)-based framework that personalizes enteral nutrition (EN) dosing for critically ill patients using electronic health record data. DeepEN was trained on over 11,000 ICU patients from the MIMIC-IV database to generate 4-hourly, patient-specific targets for caloric, protein, and fluid intake. The model's state space integrates demographics, comorbidities, vital signs, laboratory results, and prior interventions relevant to nutritional management, while its reward function balances short-term physiological and nutrition-related goals with long-term survival. A dueling double deep Q-network with Conservative Q-Learning regularization is used to ensure safe and reliable policy learning from retrospective data. DeepEN achieved a 3.7 $\pm$ 0.17 percentage-point absolute reduction in estimated mortality compared with the clinician policy (18.8% vs 22.5%) and higher expected returns compared with guideline-based dosing (11.89 vs 8.11), with improvements in key nutritional biomarkers. U-shaped associations between deviations from clinician dosing and mortality suggest that the learned policy aligns with high-value clinician actions while diverging from suboptimal ones. These findings demonstrate the feasibility of conservative offline RL for individualized EN therapy and suggest that data-driven personalization may improve outcomes beyond guideline- or heuristic-based approaches.

**arXiv ID:** 2510.08350
</details>

<details>
<summary><strong>RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning</strong> - Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, Huazhe Xu - [[pdf]](https://arxiv.org/pdf/2510.14830)</summary>

**Abstract:** Real-world robotic manipulation in homes and factories demands reliability, efficiency, and robustness that approach or surpass the performance of skilled human operators. We present RL-100, a real-world reinforcement learning framework built on diffusion-based visuomotor policies. RL-100 unifies imitation and reinforcement learning under a single PPO-style objective applied within the denoising process, yielding conservative and stable policy improvements across both offline and online stages. To meet deployment latency constraints, we employ a lightweight consistency distillation procedure that compresses multi-step diffusion into a one-step controller for high-frequency control. The framework is task-, embodiment-, and representation-agnostic, and supports both single-action outputs and action-chunking control. We evaluate RL-100 on seven diverse real-robot manipulation tasks, ranging from dynamic pushing and agile bowling to pouring, cloth folding, unscrewing, and multi-stage juicing. RL-100 attains 100% success across evaluated trials, achieving 900 out of 900 successful episodes, including up to 250 out of 250 consecutive trials on one task, and matches or surpasses expert teleoperators in time-to-completion. Without retraining, a single policy attains approximately 90% zero-shot success under environmental and dynamics shifts, adapts in a few-shot regime to significant task variations (86.7%), and remains robust to aggressive human perturbations (about 95%). In a public shopping-mall deployment, the juicing robot served random customers continuously for roughly seven hours without failure. Together, these results suggest a practical path toward deployment-ready robot learning: start from human priors, align training objectives with human-grounded metrics, and reliably extend performance beyond human demonstrations.

**arXiv ID:** 2510.14830
</details>

<details>
<summary><strong>GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning</strong> - Jinchang Luo, Mingquan Cheng, Fan Wan, Ni Li, Xiaoling Xia, Shuangshuang Tian, Tingcheng Bian, Haiwei Wang, Haohuan Fu, Yan Tao - [[pdf]](https://arxiv.org/pdf/2510.20548)</summary>

**Abstract:** Reinforcement learning has recently shown promise in improving retrieval-augmented generation (RAG). Despite these advances, its effectiveness in multi-hop question answering (QA) remains limited by two fundamental limitations: (i) global planning absence to structure multi-step reasoning, and (ii) unfaithful execution, which hinders effective query formulation and consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement learning framework designed to enhance global reasoning in multi-hop QA. GlobalRAG decomposes questions into subgoals, coordinates retrieval with reasoning, and refines evidence iteratively. To guide this process, we introduce Planning Quality Reward and SubGoal Completion Reward, which encourage coherent planning and reliable subgoal execution. In addition, a progressive weight annealing strategy balances process-oriented and outcome-based objectives. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms strong baselines while using only 8k training data (42% of the training data used by strong baselines), achieving average improvements of 14.2% in both EM and F1.

**arXiv ID:** 2510.20548
</details>

<details>
<summary><strong>ChartEditor: A Reinforcement Learning Framework for Robust Chart Editing</strong> - Liangyu Chen, Yichen Xu, Jianzhe Ma, Yuqi Liu, Donglu Yang, Liang Zhang, Wenxuan Wang, Qin Jin - [[pdf]](https://arxiv.org/pdf/2511.15266)</summary>

**Abstract:** Chart editing reduces manual effort in visualization design. Typical benchmarks limited in data diversity and assume access to complete chart code, which is seldom in real-world scenarios. To address this gap, we present ChartEditVista, a comprehensive benchmark consisting of 7,964 samples spanning 31 chart categories. It encompasses diverse editing instructions and covers nearly all editable chart elements. The inputs in ChartEditVista include only the original chart image and natural language editing instructions, without the original chart codes. ChartEditVista is generated through a fully automated pipeline that produces, edits, and verifies charts, ensuring high-quality chart editing data. Besides, we introduce two novel fine-grained, rule-based evaluation metrics: the layout metric, which evaluates the position, size and color of graphical components; and the text metric, which jointly assesses textual content and font styling. Building on top of ChartEditVista, we present ChartEditor, a model trained using a reinforcement learning framework that incorporates a novel rendering reward to simultaneously enforce code executability and visual fidelity. Through extensive experiments and human evaluations, we demonstrate that ChartEditVista provides a robust evaluation, while ChartEditor consistently outperforms models with similar-scale and larger-scale on chart editing tasks.

**arXiv ID:** 2511.15266
</details>

<details>
<summary><strong>MedBench v4: A Robust and Scalable Benchmark for Evaluating Chinese Medical Language Models, Multimodal Models, and Intelligent Agents</strong> - Jinru Ding, Lu Lu, Chao Ding, Mouxiao Bian, Jiayuan Chen, Wenrao Pang, Ruiyao Chen, Xinwei Peng, Renjie Lu, Sijie Ren, Guanxu Zhu, Xiaoqin Wu, Zhiqiang Liu, Rongzhao Zhang, Luyi Jiang, Bing Han, Yunqiu Wang, Jie Xu - [[pdf]](https://arxiv.org/pdf/2511.14439)</summary>

**Abstract:** Recent advances in medical large language models (LLMs), multimodal models, and agents demand evaluation frameworks that reflect real clinical workflows and safety constraints. We present MedBench v4, a nationwide, cloud-based benchmarking infrastructure comprising over 700,000 expert-curated tasks spanning 24 primary and 91 secondary specialties, with dedicated tracks for LLMs, multimodal models, and agents. Items undergo multi-stage refinement and multi-round review by clinicians from more than 500 institutions, and open-ended responses are scored by an LLM-as-a-judge calibrated to human ratings. We evaluate 15 frontier models. Base LLMs reach a mean overall score of 54.1/100 (best: Claude Sonnet 4.5, 62.5/100), but safety and ethics remain low (18.4/100). Multimodal models perform worse overall (mean 47.5/100; best: GPT-5, 54.9/100), with solid perception yet weaker cross-modal reasoning. Agents built on the same backbones substantially improve end-to-end performance (mean 79.8/100), with Claude Sonnet 4.5-based agents achieving up to 85.3/100 overall and 88.9/100 on safety tasks. MedBench v4 thus reveals persisting gaps in multimodal reasoning and safety for base models, while showing that governance-aware agentic orchestration can markedly enhance benchmarked clinical readiness without sacrificing capability. By aligning tasks with Chinese clinical guidelines and regulatory priorities, the platform offers a practical reference for hospitals, developers, and policymakers auditing medical AI.

**arXiv ID:** 2511.14439
</details>

<details>
<summary><strong>Transformer-Guided Deep Reinforcement Learning for Optimal Takeoff Trajectory Design of an eVTOL Drone</strong> - Nathan M. Roberts II, Xiaosong Du - [[pdf]](https://arxiv.org/pdf/2511.14887)</summary>

**Abstract:** The rapid advancement of electric vertical take-off and landing (eVTOL) aircraft offers a promising opportunity to alleviate urban traffic congestion. Thus, developing optimal takeoff trajectories for minimum energy consumption becomes essential for broader eVTOL aircraft applications. Conventional optimal control methods (such as dynamic programming and linear quadratic regulator) provide highly efficient and well-established solutions but are limited by problem dimensionality and complexity. Deep reinforcement learning (DRL) emerges as a special type of artificial intelligence tackling complex, nonlinear systems; however, the training difficulty is a key bottleneck that limits DRL applications. To address these challenges, we propose the transformer-guided DRL to alleviate the training difficulty by exploring a realistic state space at each time step using a transformer. The proposed transformer-guided DRL was demonstrated on an optimal takeoff trajectory design of an eVTOL drone for minimal energy consumption while meeting takeoff conditions (i.e., minimum vertical displacement and minimum horizontal velocity) by varying control variables (i.e., power and wing angle to the vertical). Results presented that the transformer-guided DRL agent learned to take off with $4.57\times10^6$ time steps, representing 25% of the $19.79\times10^6$ time steps needed by a vanilla DRL agent. In addition, the transformer-guided DRL achieved 97.2% accuracy on the optimal energy consumption compared against the simulation-based optimal reference while the vanilla DRL achieved 96.3% accuracy. Therefore, the proposed transformer-guided DRL outperformed vanilla DRL in terms of both training efficiency as well as optimal design verification.

**arXiv ID:** 2511.14887
</details>

<details>
<summary><strong>Vehicle Routing Problems via Quantum Graph Attention Network Deep Reinforcement Learning</strong> - Le Tung Giang, Vu Hoang Viet, Nguyen Xuan Tung, Trinh Van Chien, Won-Joo Hwang - [[pdf]](https://arxiv.org/pdf/2511.15175)</summary>

**Abstract:** The vehicle routing problem (VRP) is a fundamental NP-hard task in intelligent transportation systems with broad applications in logistics and distribution. Deep reinforcement learning (DRL) with Graph Neural Networks (GNNs) has shown promise, yet classical models rely on large multi-layer perceptrons (MLPs) that are parameter-heavy and memory-bound. We propose a Quantum Graph Attention Network (Q-GAT) within a DRL framework, where parameterized quantum circuits (PQCs) replace conventional MLPs at critical readout stages. The hybrid model maintains the expressive capacity of graph attention encoders while reducing trainable parameters by more than 50%. Using proximal policy optimization (PPO) with greedy and stochastic decoding, experiments on VRP benchmarks show that Q-GAT achieves faster convergence and reduces routing cost by about 5% compared with classical GAT baselines. These results demonstrate the potential of PQC-enhanced GNNs as compact and effective solvers for large-scale routing and logistics optimization.

**arXiv ID:** 2511.15175
</details>

<details>
<summary><strong>Optimized scheduling of electricity-heat cooperative system considering wind energy consumption and peak shaving and valley filling</strong> - Jin Ye, Lingmei Wang, Shujian Zhang, Haihang WU - [[pdf]](https://arxiv.org/pdf/2511.15250)</summary>

**Abstract:** With the global energy transition and rapid development of renewable energy, the scheduling optimization challenge for combined power-heat systems under new energy integration and multiple uncertainties has become increasingly prominent. Addressing this challenge, this study proposes an intelligent scheduling method based on the improved Dual-Delay Deep Deterministic Policy Gradient (PVTD3) algorithm. System optimization is achieved by introducing a penalty term for grid power purchase variations. Simulation results demonstrate that under three typical scenarios (10%, 20%, and 30% renewable penetration), the PVTD3 algorithm reduces the system's comprehensive cost by 6.93%, 12.68%, and 13.59% respectively compared to the traditional TD3 algorithm. Concurrently, it reduces the average fluctuation amplitude of grid power purchases by 12.8%. Regarding energy storage management, the PVTD3 algorithm reduces the end-time state values of low-temperature thermal storage tanks by 7.67-17.67 units while maintaining high-temperature tanks within the 3.59-4.25 safety operating range. Multi-scenario comparative validation demonstrates that the proposed algorithm not only excels in economic efficiency and grid stability but also exhibits superior sustainable scheduling capabilities in energy storage device management.

**arXiv ID:** 2511.15250
</details>

<details>
<summary><strong>GRPO-RM: Fine-Tuning Representation Models via GRPO-Driven Reinforcement Learning</strong> - Yanchen Xu, Ziheng Jiao, Hongyuan Zhang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2511.15256)</summary>

**Abstract:** The Group Relative Policy Optimization (GRPO), a reinforcement learning method used to fine-tune large language models (LLMs), has proved its effectiveness in practical applications such as DeepSeek-R1. It raises a question whether GRPO can be generalized to representation learning models. In this paper, we propose Group Relative Policy Optimization for Representation Model (GRPO-RM), and investigate the performance of GRPO-like policy in post-training representation models. Specifically, our method establishes a predefined output set to functionally replace token sequence sampling in LLMs, thereby generating an output group, which is essential for the probability-driven optimization of GRPO. In addition, a specialized reward function is designed to accommodate the properties of representation models. Extensive experiments are conducted on various real-world datasets to validate the effectiveness of our proposed method.

**arXiv ID:** 2511.15256
</details>

<details>
<summary><strong>The Impact of Quantization on Large Reasoning Model Reinforcement Learning</strong> - Medha Kumar, Zifei Xu, Xin Wang, Tristan Webb - [[pdf]](https://arxiv.org/pdf/2511.15694)</summary>

**Abstract:** Strong reasoning capabilities can now be achieved by large-scale reinforcement learning (RL) without any supervised fine-tuning. Although post-training quantization (PTQ) and quantization-aware training (QAT) are well studied in the context of fine-tuning, how quantization impacts RL in large reasoning models (LRMs) remains an open question. To answer this question, we conducted systematic experiments and discovered a significant gap in reasoning performance on mathematical benchmarks between post-RL quantized models and their quantization-aware RL optimized counterparts. Our findings suggest that quantization-aware RL training negatively impacted the learning process, whereas PTQ and QLoRA led to greater performance.

**arXiv ID:** 2511.15694
</details>

<details>
<summary><strong>Attacking Autonomous Driving Agents with Adversarial Machine Learning: A Holistic Evaluation with the CARLA Leaderboard</strong> - Henry Wong, Clement Fung, Weiran Lin, Karen Li, Stanley Chen, Lujo Bauer - [[pdf]](https://arxiv.org/pdf/2511.14876)</summary>

**Abstract:** To autonomously control vehicles, driving agents use outputs from a combination of machine-learning (ML) models, controller logic, and custom modules. Although numerous prior works have shown that adversarial examples can mislead ML models used in autonomous driving contexts, it remains unclear if these attacks are effective at producing harmful driving actions for various agents, environments, and scenarios.
To assess the risk of adversarial examples to autonomous driving, we evaluate attacks against a variety of driving agents, rather than against ML models in isolation. To support this evaluation, we leverage CARLA, an urban driving simulator, to create and evaluate adversarial examples. We create adversarial patches designed to stop or steer driving agents, stream them into the CARLA simulator at runtime, and evaluate them against agents from the CARLA Leaderboard, a public repository of best-performing autonomous driving agents from an annual research competition. Unlike prior work, we evaluate attacks against autonomous driving systems without creating or modifying any driving-agent code and against all parts of the agent included with the ML model.
We perform a case-study investigation of two attack strategies against three open-source driving agents from the CARLA Leaderboard across multiple driving scenarios, lighting conditions, and locations. Interestingly, we show that, although some attacks can successfully mislead ML models into predicting erroneous stopping or steering commands, some driving agents use modules, such as PID control or GPS-based rules, that can overrule attacker-manipulated predictions from ML models.

**arXiv ID:** 2511.14876
</details>

<details>
<summary><strong>Learning Where, What and How to Transfer: A Multi-Role Reinforcement Learning Approach for Evolutionary Multitasking</strong> - Jiajun Zhan, Zeyuan Ma, Yue-Jiao Gong, Kay Chen Tan - [[pdf]](https://arxiv.org/pdf/2511.15199)</summary>

**Abstract:** Evolutionary multitasking (EMT) algorithms typically require tailored designs for knowledge transfer, in order to assure convergence and optimality in multitask optimization. In this paper, we explore designing a systematic and generalizable knowledge transfer policy through Reinforcement Learning. We first identify three major challenges: determining the task to transfer (where), the knowledge to be transferred (what) and the mechanism for the transfer (how). To address these challenges, we formulate a multi-role RL system where three (groups of) policy networks act as specialized agents: a task routing agent incorporates an attention-based similarity recognition module to determine source-target transfer pairs via attention scores; a knowledge control agent determines the proportion of elite solutions to transfer; and a group of strategy adaptation agents control transfer strength by dynamically controlling hyper-parameters in the underlying EMT framework. Through pre-training all network modules end-to-end over an augmented multitask problem distribution, a generalizable meta-policy is obtained. Comprehensive validation experiments show state-of-the-art performance of our method against representative baselines. Further in-depth analysis not only reveals the rationale behind our proposal but also provide insightful interpretations on what the system have learned.

**arXiv ID:** 2511.15199
</details>

<details>
<summary><strong>Reinforcement Learning in Queue-Reactive Models: Application to Optimal Execution</strong> - Tomas Espana, Yadh Hafsi, Fabrizio Lillo, Edoardo Vittori - [[pdf]](https://arxiv.org/pdf/2511.15262)</summary>

**Abstract:** We investigate the use of Reinforcement Learning for the optimal execution of meta-orders, where the objective is to execute incrementally large orders while minimizing implementation shortfall and market impact over an extended period of time. Departing from traditional parametric approaches to price dynamics and impact modeling, we adopt a model-free, data-driven framework. Since policy optimization requires counterfactual feedback that historical data cannot provide, we employ the Queue-Reactive Model to generate realistic and tractable limit order book simulations that encompass transient price impact, and nonlinear and dynamic order flow responses. Methodologically, we train a Double Deep Q-Network agent on a state space comprising time, inventory, price, and depth variables, and evaluate its performance against established benchmarks. Numerical simulation results show that the agent learns a policy that is both strategic and tactical, adapting effectively to order book conditions and outperforming standard approaches across multiple training configurations. These findings provide strong evidence that model-free Reinforcement Learning can yield adaptive and robust solutions to the optimal execution problem.

**arXiv ID:** 2511.15262
</details>

<details>
<summary><strong>Planning in Branch-and-Bound: Model-Based Reinforcement Learning for Exact Combinatorial Optimization</strong> - Paul Strang, Zacharie Alès, Côme Bissuel, Safia Kedad-Sidhoum, Emmanuel Rachelson - [[pdf]](https://arxiv.org/pdf/2511.09219)</summary>

**Abstract:** Mixed-Integer Linear Programming (MILP) lies at the core of many real-world combinatorial optimization (CO) problems, traditionally solved by branch-and-bound (B&B). A key driver influencing B&B solvers efficiency is the variable selection heuristic that guides branching decisions. Looking to move beyond static, hand-crafted heuristics, recent work has explored adapting traditional reinforcement learning (RL) algorithms to the B&B setting, aiming to learn branching strategies tailored to specific MILP distributions. In parallel, RL agents have achieved remarkable success in board games, a very specific type of combinatorial problems, by leveraging environment simulators to plan via Monte Carlo Tree Search (MCTS). Building on these developments, we introduce Plan-and-Branch-and-Bound (PlanB&B), a model-based reinforcement learning (MBRL) agent that leverages a learned internal model of the B&B dynamics to discover improved branching strategies. Computational experiments empirically validate our approach, with our MBRL branching agent outperforming previous state-of-the-art RL methods across four standard MILP benchmarks.

**arXiv ID:** 2511.09219
</details>

<details>
<summary><strong>Look, Zoom, Understand: The Robotic Eyeball for Embodied Perception</strong> - Jiashu Yang, Yifan Han, Yucheng Xie, Ning Guo, Wenzhao Lian - [[pdf]](https://arxiv.org/pdf/2511.15279)</summary>

**Abstract:** In embodied AI perception systems, visual perception should be active: the goal is not to passively process static images, but to actively acquire more informative data within pixel and spatial budget constraints. Existing vision models and fixed RGB-D camera systems fundamentally fail to reconcile wide-area coverage with fine-grained detail acquisition, severely limiting their efficacy in open-world robotic applications. To address this issue, we propose EyeVLA, a robotic eyeball for active visual perception that can take proactive actions based on instructions, enabling clear observation of fine-grained target objects and detailed information across a wide spatial extent. EyeVLA discretizes action behaviors into action tokens and integrates them with vision-language models (VLMs) that possess strong open-world understanding capabilities, enabling joint modeling of vision, language, and actions within a single autoregressive sequence. By using the 2D bounding box coordinates to guide the reasoning chain and applying reinforcement learning to refine the viewpoint selection policy, we transfer the open-world scene understanding capability of the VLM to a vision language action (VLA) policy using only minimal real-world data. Experiments show that our system efficiently performs instructed scenes in real-world environments and actively acquires more accurate visual information through instruction-driven actions of rotation and zoom, thereby achieving strong environmental perception capabilities. EyeVLA introduces a novel robotic vision system that leverages detailed and spatially rich, large-scale embodied data, and actively acquires highly informative visual observations for downstream embodied tasks.

**arXiv ID:** 2511.15279
</details>

<details>
<summary><strong>Platform-Agnostic Reinforcement Learning Framework for Safe Exploration of Cluttered Environments with Graph Attention</strong> - Gabriele Calzolari, Vidya Sumathy, Christoforos Kanellakis, George Nikolakopoulos - [[pdf]](https://arxiv.org/pdf/2511.15358)</summary>

**Abstract:** Autonomous exploration of obstacle-rich spaces requires strategies that ensure efficiency while guaranteeing safety against collisions with obstacles. This paper investigates a novel platform-agnostic reinforcement learning framework that integrates a graph neural network-based policy for next-waypoint selection, with a safety filter ensuring safe mobility. Specifically, the neural network is trained using reinforcement learning through the Proximal Policy Optimization (PPO) algorithm to maximize exploration efficiency while minimizing safety filter interventions. Henceforth, when the policy proposes an infeasible action, the safety filter overrides it with the closest feasible alternative, ensuring consistent system behavior. In addition, this paper introduces a reward function shaped by a potential field that accounts for both the agent's proximity to unexplored regions and the expected information gain from reaching them. The proposed framework combines the adaptability of reinforcement learning-based exploration policies with the reliability provided by explicit safety mechanisms. This feature plays a key role in enabling the deployment of learning-based policies on robotic platforms operating in real-world environments. Extensive evaluations in both simulations and experiments performed in a lab environment demonstrate that the approach achieves efficient and safe exploration in cluttered spaces.

**arXiv ID:** 2511.15358
</details>

<details>
<summary><strong>RoboTidy : A 3D Gaussian Splatting Household Tidying Benchmark for Embodied Navigation and Action</strong> - Xiaoquan Sun, Ruijian Zhang, Kang Pang, Bingchen Miao, Yuxiang Tan, Zhen Yang, Ming Li, Jiayu Chen - [[pdf]](https://arxiv.org/pdf/2511.14161)</summary>

**Abstract:** Household tidying is an important application area, yet current benchmarks neither model user preferences nor support mobility, and they generalize poorly, making it hard to comprehensively assess integrated language-to-action capabilities. To address this, we propose RoboTidy, a unified benchmark for language-guided household tidying that supports Vision-Language-Action (VLA) and Vision-Language-Navigation (VLN) training and evaluation. RoboTidy provides 500 photorealistic 3D Gaussian Splatting (3DGS) household scenes (covering 500 objects and containers) with collisions, formulates tidying as an "Action (Object, Container)" list, and supplies 6.4k high-quality manipulation demonstration trajectories and 1.5k naviagtion trajectories to support both few-shot and large-scale training. We also deploy RoboTidy in the real world for object tidying, establishing an end-to-end benchmark for household tidying. RoboTidy offers a scalable platform and bridges a key gap in embodied AI by enabling holistic and realistic evaluation of language-guided robots.

**arXiv ID:** 2511.14161
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
