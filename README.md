# Agent arXiv Daily

**Last Updated:** 2026-01-16 03:06:42

**Total Papers:** 64

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (7 papers)</h2></summary>

<details>
<summary><strong>A Scoping Review of the Ethical Perspectives on Anthropomorphising Large Language Model-Based Conversational Agents</strong> - Andrea Ferrario, Rasita Vinay, Matteo Casserini, Alessandro Facchini - [[pdf]](https://arxiv.org/pdf/2601.09869)</summary>

**Abstract:** Anthropomorphisation -- the phenomenon whereby non-human entities are ascribed human-like qualities -- has become increasingly salient with the rise of large language model (LLM)-based conversational agents (CAs). Unlike earlier chatbots, LLM-based CAs routinely generate interactional and linguistic cues, such as first-person self-reference, epistemic and affective expressions that empirical work shows can increase engagement. On the other hand, anthropomorphisation raises ethical concerns, including deception, overreliance, and exploitative relationship framing, while some authors argue that anthropomorphic interaction may support autonomy, well-being, and inclusion. Despite increasing interest in the phenomenon, literature remains fragmented across domains and varies substantially in how it defines, operationalizes, and normatively evaluates anthropomorphisation. This scoping review maps ethically oriented work on anthropomorphising LLM-based CAs across five databases and three preprint repositories. We synthesize (1) conceptual foundations, (2) ethical challenges and opportunities, and (3) methodological approaches. We find convergence on attribution-based definitions but substantial divergence in operationalization, a predominantly risk-forward normative framing, and limited empirical work that links observed interaction effects to actionable governance guidance. We conclude with a research agenda and design/governance recommendations for ethically deploying anthropomorphic cues in LLM-based conversational agents.

**arXiv ID:** 2601.09869
</details>

<details>
<summary><strong>PaperScout: An Autonomous Agent for Academic Paper Search with Process-Aware Sequence-Level Policy Optimization</strong> - Tingyue Pan, Jie Ouyang, Mingyue Cheng, Qingchuan Li, Zirui Liu, Mingfan Pan, Shuo Yu, Qi Liu - [[pdf]](https://arxiv.org/pdf/2601.10029)</summary>

**Abstract:** Academic paper search is a fundamental task in scientific research, yet most existing approaches rely on rigid, predefined workflows that struggle with complex, conditional queries. To address this limitation, we propose PaperScout, an autonomous agent that reformulates paper search as a sequential decision-making process. Unlike static workflows, PaperScout dynamically decides whether, when, and how to invoke search and expand tools based on accumulated retrieval context. However, training such agents presents a fundamental challenge: standard reinforcement learning methods, typically designed for single-turn tasks, suffer from a granularity mismatch when applied to multi-turn agentic tasks, where token-level optimization diverges from the granularity of sequence-level interactions, leading to noisy credit assignment. We introduce Proximal Sequence Policy Optimization (PSPO), a process-aware, sequence-level policy optimization method that aligns optimization with agent-environment interaction. Comprehensive experiments on both synthetic and real-world benchmarks demonstrate that PaperScout significantly outperforms strong workflow-driven and RL baselines in both recall and relevance, validating the effectiveness of our adaptive agentic framework and optimization strategy.

**arXiv ID:** 2601.10029
</details>

<details>
<summary><strong>Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering</strong> - Xinyu Zhu, Yuzhu Cai, Zexi Liu, Bingyang Zheng, Cheng Wang, Rui Ye, Jiaao Chen, Hanrui Wang, Wei-Chen Wang, Yuzhi Zhang, Linfeng Zhang, Weinan E, Di Jin, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2601.10402)</summary>

**Abstract:** The advancement of artificial intelligence toward agentic science is currently bottlenecked by the challenge of ultra-long-horizon autonomy, the ability to sustain strategic coherence and iterative correction over experimental cycles spanning days or weeks. While Large Language Models (LLMs) have demonstrated prowess in short-horizon reasoning, they are easily overwhelmed by execution details in the high-dimensional, delayed-feedback environments of real-world research, failing to consolidate sparse feedback into coherent long-term guidance. Here, we present ML-Master 2.0, an autonomous agent that masters ultra-long-horizon machine learning engineering (MLE) which is a representative microcosm of scientific discovery. By reframing context management as a process of cognitive accumulation, our approach introduces Hierarchical Cognitive Caching (HCC), a multi-tiered architecture inspired by computer systems that enables the structural differentiation of experience over time. By dynamically distilling transient execution traces into stable knowledge and cross-task wisdom, HCC allows agents to decouple immediate execution from long-term experimental strategy, effectively overcoming the scaling limits of static context windows. In evaluations on OpenAI's MLE-Bench under 24-hour budgets, ML-Master 2.0 achieves a state-of-the-art medal rate of 56.44%. Our findings demonstrate that ultra-long-horizon autonomy provides a scalable blueprint for AI capable of autonomous exploration beyond human-precedent complexities.

**arXiv ID:** 2601.10402
</details>

<details>
<summary><strong>Introducing Axlerod: An LLM-based Chatbot for Assisting Independent Insurance Agents</strong> - Adam Bradley, John Hastings, Khandaker Mamun Ahmed - [[pdf]](https://arxiv.org/pdf/2601.09715)</summary>

**Abstract:** The insurance industry is undergoing a paradigm shift through the adoption of artificial intelligence (AI) technologies, particularly in the realm of intelligent conversational agents. Chatbots have evolved into sophisticated AI-driven systems capable of automating complex workflows, including policy recommendation and claims triage, while simultaneously enabling dynamic, context-aware user engagement. This paper presents the design, implementation, and empirical evaluation of Axlerod, an AI-powered conversational interface designed to improve the operational efficiency of independent insurance agents. Leveraging natural language processing (NLP), retrieval-augmented generation (RAG), and domain-specific knowledge integration, Axlerod demonstrates robust capabilities in parsing user intent, accessing structured policy databases, and delivering real-time, contextually relevant responses. Experimental results underscore Axlerod's effectiveness, achieving an overall accuracy of 93.18% in policy retrieval tasks while reducing the average search time by 2.42 seconds. This work contributes to the growing body of research on enterprise-grade AI applications in insurtech, with a particular focus on agent-assistive rather than consumer-facing architectures.

**arXiv ID:** 2601.09715
</details>

<details>
<summary><strong>Uncovering Systemic and Environment Errors in Autonomous Systems Using Differential Testing</strong> - Yashwanthi Anand, Rahil P Mehta, Manish Motwani, Sandhya Saisubramanian - [[pdf]](https://arxiv.org/pdf/2507.03870)</summary>

**Abstract:** When an autonomous agent behaves undesirably, including failure to complete a task, it can be difficult to determine whether the behavior is due to a systemic agent error, such as flaws in the model or policy, or an environment error, where a task is inherently infeasible under a given environment configuration, even for an ideal agent. As agents and their environments grow more complex, identifying the error source becomes increasingly difficult but critical for reliable deployment. We introduce AIProbe, a novel black-box testing technique that applies differential testing to attribute undesirable agent behaviors either to agent deficiencies, such as modeling or training flaws, or due to environmental infeasibility. AIProbe first generates diverse environmental configurations and tasks for testing the agent, by modifying configurable parameters using Latin Hypercube sampling. It then solves each generated task using a search-based planner, independent of the agent. By comparing the agent's performance to the planner's solution, AIProbe identifies whether failures are due to errors in the agent's model or policy, or due to unsolvable task conditions. Our evaluation across multiple domains shows that AIProbe significantly outperforms state-of-the-art techniques in detecting both total and unique errors, thereby contributing to a reliable deployment of autonomous agents.

**arXiv ID:** 2507.03870
</details>

<details>
<summary><strong>Gradient Coupling: The Hidden Barrier to Generalization in Agentic Reinforcement Learning</strong> - Jingyu Liu, Xiaopeng Wu, Jingquan Peng, Kehan Chen, Chuan Yu, Lizhong Ding, Yong Liu - [[pdf]](https://arxiv.org/pdf/2509.23870)</summary>

**Abstract:** Reinforcement learning (RL) is a dominant paradigm for training autonomous agents, yet these agents often exhibit poor generalization, failing to adapt to scenarios not seen during training. In this work, we identify a fundamental cause of this brittleness, a phenomenon which we term "gradient coupling." We hypothesize that in complex agentic tasks, the high similarity between distinct states leads to destructive interference between gradients. Specifically, a gradient update that reinforces an optimal action in one state can inadvertently increase the likelihood of a suboptimal action in a similar, yet different, state. To solve this, we propose a novel objective where the actor is trained to simultaneously function as a classifier that separates good and bad actions. This auxiliary pressure compels the model to learn disentangled embeddings for positive and negative actions, which mitigates negative gradient interference and improve the generalization performance. Extensive experiments demonstrate the effectiveness of our method.

**arXiv ID:** 2509.23870
</details>

<details>
<summary><strong>From SERPs to Agents: A Platform for Comparative Studies of Information Interaction</strong> - Saber Zerhoudi, Michael Granitzer - [[pdf]](https://arxiv.org/pdf/2601.09937)</summary>

**Abstract:** The diversification of information access systems, from RAG to autonomous agents, creates a critical need for comparative user studies. However, the technical overhead to deploy and manage these distinct systems is a major barrier. We present UXLab, an open-source system for web-based user studies that addresses this challenge. Its core is a web-based dashboard enabling the complete, no-code configuration of complex experimental designs. Researchers can visually manage the full study, from recruitment to comparing backends like traditional search, vector databases, and LLMs. We demonstrate UXLab's value via a micro case study comparing user behavior with RAG versus an autonomous agent. UXLab allows researchers to focus on experimental design and analysis, supporting future multi-modal interaction research.

**arXiv ID:** 2601.09937
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>Formal Safety Guarantees for Autonomous Vehicles using Barrier Certificates</strong> - Oumaima Barhoumi, Mohamed H Zaki, Sofiène Tahar - [[pdf]](https://arxiv.org/pdf/2601.09740)</summary>

**Abstract:** Modern AI technologies enable autonomous vehicles to perceive complex scenes, predict human behavior, and make real-time driving decisions. However, these data-driven components often operate as black boxes, lacking interpretability and rigorous safety guarantees. Autonomous vehicles operate in dynamic, mixed-traffic environments where interactions with human-driven vehicles introduce uncertainty and safety challenges. This work develops a formally verified safety framework for Connected and Autonomous Vehicles (CAVs) that integrates Barrier Certificates (BCs) with interpretable traffic conflict metrics, specifically Time-to-Collision (TTC) as a spatio-temporal safety metric. Safety conditions are verified using Satisfiability Modulo Theories (SMT) solvers, and an adaptive control mechanism ensures vehicles comply with these constraints in real time. Evaluation on real-world highway datasets shows a significant reduction in unsafe interactions, with up to 40\% fewer events where TTC falls below a 3 seconds threshold, and complete elimination of conflicts in some lanes. This approach provides both interpretable and provable safety guarantees, demonstrating a practical and scalable strategy for safe autonomous driving.

**arXiv ID:** 2601.09740
</details>

<details>
<summary><strong>AgentGuardian: Learning Access Control Policies to Govern AI Agent Behavior</strong> - Nadya Abaev, Denis Klimov, Gerard Levinov, David Mimran, Yuval Elovici, Asaf Shabtai - [[pdf]](https://arxiv.org/pdf/2601.10440)</summary>

**Abstract:** Artificial intelligence (AI) agents are increasingly used in a variety of domains to automate tasks, interact with users, and make decisions based on data inputs. Ensuring that AI agents perform only authorized actions and handle inputs appropriately is essential for maintaining system integrity and preventing misuse. In this study, we introduce the AgentGuardian, a novel security framework that governs and protects AI agent operations by enforcing context-aware access-control policies. During a controlled staging phase, the framework monitors execution traces to learn legitimate agent behaviors and input patterns. From this phase, it derives adaptive policies that regulate tool calls made by the agent, guided by both real-time input context and the control flow dependencies of multi-step agent actions. Evaluation across two real-world AI agent applications demonstrates that AgentGuardian effectively detects malicious or misleading inputs while preserving normal agent functionality. Moreover, its control-flow-based governance mechanism mitigates hallucination-driven errors and other orchestration-level malfunctions.

**arXiv ID:** 2601.10440
</details>

<details>
<summary><strong>Leveraging Open-Source Large Language Models for encoding Social Determinants of Health using an Intelligent Router</strong> - Akul Goel, Surya Narayanan Hari, Belinda Waltman, Matt Thomson - [[pdf]](https://arxiv.org/pdf/2405.19631)</summary>

**Abstract:** Social Determinants of Health (SDOH), also known as Health-Related Social Needs (HSRN), play a significant role in patient health outcomes. The Centers for Disease Control and Prevention (CDC) introduced a subset of ICD-10 codes called Z-codes to recognize and measure SDOH. However, Z-codes are infrequently coded in a patient's Electronic Health Record (EHR), and instead, in many cases, need to be inferred from clinical notes. Previous research has shown that large language models (LLMs) show promise on extracting unstructured data from EHRs, but it can be difficult to identify a single model that performs best on varied coding tasks. Further, clinical notes contain protected health information posing a challenge for the use of closed-source language models from commercial vendors. The identification of open-source LLMs that can be run within health organizations and exhibit high performance on SDOH tasks is an important issue to solve. Here, we introduce an intelligent routing system for SDOH coding that uses a language model router to direct medical record data to open-source LLMs that demonstrate optimal performance on specific SDOH codes. This intelligent routing system exhibits state of the art performance of 96.4% accuracy averaged across 13 codes, including homelessness and food insecurity, outperforming closed models such as GPT-4o. We leveraged a publicly-available, deidentified dataset of medical record notes to run the router, but we also introduce a synthetic data generation and validation paradigm to increase the scale of training data without needing privacy-protected medical records. Together, we demonstrate an architecture for intelligent routing of inputs to task-optimal language models to achieve high performance across a set of medical coding sub-tasks.

**arXiv ID:** 2405.19631
</details>

<details>
<summary><strong>HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation</strong> - Rongxin Chen, Tianyu Wu, Bingbing Xu, Jiatang Luo, Xiucheng Xu, Huawei Shen - [[pdf]](https://arxiv.org/pdf/2601.05656)</summary>

**Abstract:** High-fidelity agent initialization is crucial for credible Agent-Based Modeling across diverse domains. A robust framework should be Topic-Adaptive, capturing macro-level joint distributions while ensuring micro-level individual rationality. Existing approaches fall into two categories: static data-based retrieval methods that fail to adapt to unseen topics absent from the data, and LLM-based generation methods that lack macro-level distribution awareness, resulting in inconsistencies between micro-level persona attributes and reality. To address these problems, we propose HAG, a Hierarchical Agent Generation framework that formalizes population generation as a two-stage decision process. Firstly, utilizing a World Knowledge Model to infer hierarchical conditional probabilities to construct the Topic-Adaptive Tree, achieving macro-level distribution alignment. Then, grounded real-world data, instantiation and agentic augmentation are carried out to ensure micro-level consistency. Given the lack of specialized evaluation, we establish a multi-domain benchmark and a comprehensive PACE evaluation framework. Extensive experiments show that HAG significantly outperforms representative baselines, reducing population alignment errors by an average of 37.7% and enhancing sociological consistency by 18.8%.

**arXiv ID:** 2601.05656
</details>

<details>
<summary><strong>WebRollback: Enhancing Web Agents with Explicit Rollback Mechanisms</strong> - Zhisong Zhang, Tianqing Fang, Kaixin Ma, Wenhao Yu, Hongming Zhang, Haitao Mi, Dong Yu - [[pdf]](https://arxiv.org/pdf/2504.11788)</summary>

**Abstract:** With recent advancements in large language models, web agents have been greatly improved. However, dealing with complex and dynamic web environments requires more advanced planning and search abilities. Previous studies usually adopt a greedy one-way search strategy, which may struggle to recover from erroneous states. In this work, we enhance web agents with an explicit rollback mechanism, enabling the agent to revert back to a previous state in its navigation trajectory. This mechanism gives models the flexibility to directly control the search process, leading to an effective and efficient web navigation method. We conduct experiments on two live web navigation benchmarks with zero-shot and fine-tuning settings. The results demonstrate the effectiveness of our proposed approach.

**arXiv ID:** 2504.11788
</details>

<details>
<summary><strong>Network-Level Prompt and Trait Leakage in Local Research Agents</strong> - Hyejun Jeong, Mohammadreza Teymoorianfard, Abhinav Kumar, Amir Houmansadr, Eugene Bagdasarian - [[pdf]](https://arxiv.org/pdf/2508.20282)</summary>

**Abstract:** We show that Web and Research Agents (WRAs) -- language-model-based systems that investigate complex topics on the Internet -- are vulnerable to inference attacks by passive network observers. Deployment of WRAs \emph{locally} by organizations and individuals for privacy, legal, or financial purposes exposes them to DNS resolvers, malicious ISPs, VPNs, web proxies, and corporate or government firewalls. However, unlike sporadic and scarce web browsing by humans, WRAs visit $70{-}140$ domains per each request with a distinct timing pattern creating unique privacy risks.
Specifically, we demonstrate a novel prompt and user trait leakage attack against WRAs that only leverages their network-level metadata (i.e., visited IP addresses and their timings). We start by building a new dataset of WRA traces based on real user search queries and queries generated by synthetic personas. We define a behavioral metric (called OBELS) to comprehensively assess similarity between original and inferred prompts, showing that our attack recovers over 73\% of the functional and domain knowledge of user prompts. Extending to a multi-session setting, we recover up to 19 of 32 latent traits with high accuracy. Our attack remains effective under partial observability and noisy conditions. Finally, we discuss mitigation strategies that constrain domain diversity or obfuscate traces, showing negligible utility impact while reducing attack effectiveness by an average of 29\%.

**arXiv ID:** 2508.20282
</details>

<details>
<summary><strong>DR-Arena: an Automated Evaluation Framework for Deep Research Agents</strong> - Yiwen Gao, Ruochen Zhao, Yang Deng, Wenxuan Zhang - [[pdf]](https://arxiv.org/pdf/2601.10504)</summary>

**Abstract:** As Large Language Models (LLMs) increasingly operate as Deep Research (DR) Agents capable of autonomous investigation and information synthesis, reliable evaluation of their task performance has become a critical bottleneck. Current benchmarks predominantly rely on static datasets, which suffer from several limitations: limited task generality, temporal misalignment, and data contamination. To address these, we introduce DR-Arena, a fully automated evaluation framework that pushes DR agents to their capability limits through dynamic investigation. DR-Arena constructs real-time Information Trees from fresh web trends to ensure the evaluation rubric is synchronized with the live world state, and employs an automated Examiner to generate structured tasks testing two orthogonal capabilities: Deep reasoning and Wide coverage. DR-Arena further adopts Adaptive Evolvement Loop, a state-machine controller that dynamically escalates task complexity based on real-time performance, demanding deeper deduction or wider aggregation until a decisive capability boundary emerges. Experiments with six advanced DR agents demonstrate that DR-Arena achieves a Spearman correlation of 0.94 with the LMSYS Search Arena leaderboard. This represents the state-of-the-art alignment with human preferences without any manual efforts, validating DR-Arena as a reliable alternative for costly human adjudication.

**arXiv ID:** 2601.10504
</details>

<details>
<summary><strong>Mistake Notebook Learning: Batch-Clustered Failures for Training-Free Agent Adaptation</strong> - Xuanbo Su, Yingfang Zhang, Hao Luo, Xiaoteng Liu, Leo Huang - [[pdf]](https://arxiv.org/pdf/2512.11485)</summary>

**Abstract:** With the growing adoption of Large Language Model (LLM) agents in persistent, real-world roles, they naturally encounter continuous streams of tasks and inevitable failures. A key limitation, however, is their inability to systematically learn from these mistakes, forcing them to repeat identical errors in similar contexts. Unlike prior training-free methods that primarily store raw instance-level experience or focus on retrieving successful trajectories, we propose Mistake Notebook Learning (MNL), a novel memory framework that enables agents to self-curate generalizable guidance from batch-clustered failures. This mechanism allows agents to distill shared error patterns into structured ``mistake notes,'' updating an external memory only when batch performance improves to ensure stability. To further amplify adaptability, we integrate MNL with test-time scaling, leveraging aggregated failure patterns to actively steer the search process away from known pitfalls. Experiments on mathematical reasoning, Text-to-SQL, and interactive agent benchmarks show that MNL achieves competitive performance compared to existing memory mechanisms and in-context methods in both effectiveness and efficiency. These findings position structured mistake abstraction as a critical lever for robust agent evolution, enabling continuous improvement without the cost of parameter updates.

**arXiv ID:** 2512.11485
</details>

<details>
<summary><strong>LCF3D: A Robust and Real-Time Late-Cascade Fusion Framework for 3D Object Detection in Autonomous Driving</strong> - Carlo Sgaravatti, Riccardo Pieroni, Matteo Corno, Sergio M. Savaresi, Luca Magri, Giacomo Boracchi - [[pdf]](https://arxiv.org/pdf/2601.09812)</summary>

**Abstract:** Accurately localizing 3D objects like pedestrians, cyclists, and other vehicles is essential in Autonomous Driving. To ensure high detection performance, Autonomous Vehicles complement RGB cameras with LiDAR sensors, but effectively combining these data sources for 3D object detection remains challenging. We propose LCF3D, a novel sensor fusion framework that combines a 2D object detector on RGB images with a 3D object detector on LiDAR point clouds. By leveraging multimodal fusion principles, we compensate for inaccuracies in the LiDAR object detection network. Our solution combines two key principles: (i) late fusion, to reduce LiDAR False Positives by matching LiDAR 3D detections with RGB 2D detections and filtering out unmatched LiDAR detections; and (ii) cascade fusion, to recover missed objects from LiDAR by generating new 3D frustum proposals corresponding to unmatched RGB detections. Experiments show that LCF3D is beneficial for domain generalization, as it turns out to be successful in handling different sensor configurations between training and testing domains. LCF3D achieves significant improvements over LiDAR-based methods, particularly for challenging categories like pedestrians and cyclists in the KITTI dataset, as well as motorcycles and bicycles in nuScenes. Code can be downloaded from: this https URL.

**arXiv ID:** 2601.09812
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>Continuum Memory Architectures for Long-Horizon LLM Agents</strong> - Joe Logan - [[pdf]](https://arxiv.org/pdf/2601.09913)</summary>

**Abstract:** Retrieval-augmented generation (RAG) has become the default strategy for providing large language model (LLM) agents with contextual knowledge. Yet RAG treats memory as a stateless lookup table: information persists indefinitely, retrieval is read-only, and temporal continuity is absent. We define the \textit{Continuum Memory Architecture} (CMA), a class of systems that maintain and update internal state across interactions through persistent storage, selective retention, associative routing, temporal chaining, and consolidation into higher-order abstractions. Rather than disclosing implementation specifics, we specify the architectural requirements CMA imposes and show consistent behavioral advantages on tasks that expose RAG's structural inability to accumulate, mutate, or disambiguate memory. The empirical probes (knowledge updates, temporal association, associative recall, contextual disambiguation) demonstrate that CMA is a necessary architectural primitive for long-horizon agents while highlighting open challenges around latency, drift, and interpretability.

**arXiv ID:** 2601.09913
</details>

<details>
<summary><strong>Structured Personality Control and Adaptation for LLM Agents</strong> - Jinpeng Wang, Xinyu Jia, Wei Wei Heng, Yuquan Li, Binbin Shi, Qianlei Chen, Guannan Chen, Junxia Zhang, Yuyu Yin - [[pdf]](https://arxiv.org/pdf/2601.10025)</summary>

**Abstract:** Large Language Models (LLMs) are increasingly shaping human-computer interaction (HCI), from personalized assistants to social simulations. Beyond language competence, researchers are exploring whether LLMs can exhibit human-like characteristics that influence engagement, decision-making, and perceived realism. Personality, in particular, is critical, yet existing approaches often struggle to achieve both nuanced and adaptable expression. We present a framework that models LLM personality via Jungian psychological types, integrating three mechanisms: a dominant-auxiliary coordination mechanism for coherent core expression, a reinforcement-compensation mechanism for temporary adaptation to context, and a reflection mechanism that drives long-term personality evolution. This design allows the agent to maintain nuanced traits while dynamically adjusting to interaction demands and gradually updating its underlying structure. Personality alignment is evaluated using Myers-Briggs Type Indicator questionnaires and tested under diverse challenge scenarios as a preliminary structured assessment. Findings suggest that evolving, personality-aware LLMs can support coherent, context-sensitive interactions, enabling naturalistic agent design in HCI.

**arXiv ID:** 2601.10025
</details>

<details>
<summary><strong>JudgeAgent: Beyond Static Benchmarks for Knowledge-Driven and Dynamic LLM Evaluation</strong> - Zhichao Shi, Xuhui Jiang, Chengjin Xu, Cangli Yao, Shengjia Ma, Yinghan Shen, Zixuan Li, Jian Guo, Yuanzhuo Wang - [[pdf]](https://arxiv.org/pdf/2509.02097)</summary>

**Abstract:** Current evaluation methods for large language models (LLMs) primarily rely on static benchmarks, presenting two major challenges: limited knowledge coverage and fixed difficulties that mismatch with the evaluated LLMs. These limitations lead to superficial assessments of LLM knowledge, thereby impeding the targeted model optimizations. To bridge this gap, we propose JudgeAgent, a knowledge-driven and dynamic evaluation framework for LLMs. To address the challenge of limited knowledge coverage, JudgeAgent leverages LLM agents equipped with context graphs to traverse knowledge structures systematically for question generation. Furthermore, to mitigate data contamination and difficulty mismatch, it adopts a difficulty-adaptive and multi-turn interview mechanism. Thereby, JudgeAgent can achieve comprehensive evaluations and facilitate more effective improvement of LLMs. Empirical results demonstrate that JudgeAgent enables more comprehensive evaluations and facilitates effective model iterations, highlighting the potential of this knowledge-driven and dynamic evaluation paradigm. The source code is available on this https URL.

**arXiv ID:** 2509.02097
</details>

<details>
<summary><strong>ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback</strong> - Yutao Mou, Zhangchi Xue, Lijun Li, Peiyang Liu, Shikun Zhang, Wei Ye, Jing Shao - [[pdf]](https://arxiv.org/pdf/2601.10156)</summary>

**Abstract:** While LLM-based agents can interact with environments via invoking external tools, their expanded capabilities also amplify security risks. Monitoring step-level tool invocation behaviors in real time and proactively intervening before unsafe execution is critical for agent deployment, yet remains under-explored. In this work, we first construct TS-Bench, a novel benchmark for step-level tool invocation safety detection in LLM agents. We then develop a guardrail model, TS-Guard, using multi-task reinforcement learning. The model proactively detects unsafe tool invocation actions before execution by reasoning over the interaction history. It assesses request harmfulness and action-attack correlations, producing interpretable and generalizable safety judgments and feedback. Furthermore, we introduce TS-Flow, a guardrail-feedback-driven reasoning framework for LLM agents, which reduces harmful tool invocations of ReAct-style agents by 65 percent on average and improves benign task completion by approximately 10 percent under prompt injection attacks.

**arXiv ID:** 2601.10156
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (16 papers)</h2></summary>

<details>
<summary><strong>Beyond Rule-Based Workflows: An Information-Flow-Orchestrated Multi-Agents Paradigm via Agent-to-Agent Communication from CORAL</strong> - Xinxing Ren, Quagmire Zang, Caelum Forder, Suman Deb, Ahsen Tahir, Roman J. Georgio, Peter Carroll, Zekun Guo - [[pdf]](https://arxiv.org/pdf/2601.09883)</summary>

**Abstract:** Most existing Large Language Model (LLM)-based Multi-Agent Systems (MAS) rely on predefined workflows, where human engineers enumerate task states in advance and specify routing rules and contextual injections accordingly. Such workflow-driven designs are essentially rule-based decision trees, which suffer from two fundamental limitations: they require substantial manual effort to anticipate and encode possible task states, and they cannot exhaustively cover the state space of complex real-world tasks. To address these issues, we propose an Information-Flow-Orchestrated Multi-Agent Paradigm via Agent-to-Agent (A2A) Communication from CORAL, in which a dedicated information flow orchestrator continuously monitors task progress and dynamically coordinates other agents through the A2A toolkit using natural language, without relying on predefined workflows. We evaluate our approach on the general-purpose benchmark GAIA, using the representative workflow-based MAS OWL as the baseline while controlling for agent roles and underlying models. Under the pass@1 setting, our method achieves 63.64% accuracy, outperforming OWL's 55.15% by 8.49 percentage points with comparable token consumption. Further case-level analysis shows that our paradigm enables more flexible task monitoring and more robust handling of edge cases. Our implementation is publicly available at: this https URL

**arXiv ID:** 2601.09883
</details>

<details>
<summary><strong>M^4olGen: Multi-Agent, Multi-Stage Molecular Generation under Precise Multi-Property Constraints</strong> - Yizhan Li, Florence Cloutier, Sifan Wu, Ali Parviz, Boris Knyazev, Yan Zhang, Glen Berseth, Bang Liu - [[pdf]](https://arxiv.org/pdf/2601.10131)</summary>

**Abstract:** Generating molecules that satisfy precise numeric constraints over multiple physicochemical properties is critical and challenging. Although large language models (LLMs) are expressive, they struggle with precise multi-objective control and numeric reasoning without external structure and feedback. We introduce \textbf{M olGen}, a fragment-level, retrieval-augmented, two-stage framework for molecule generation under multi-property constraints. Stage I : Prototype generation: a multi-agent reasoner performs retrieval-anchored, fragment-level edits to produce a candidate near the feasible region. Stage II : RL-based fine-grained optimization: a fragment-level optimizer trained with Group Relative Policy Optimization (GRPO) applies one- or multi-hop refinements to explicitly minimize the property errors toward our target while regulating edit complexity and deviation from the prototype. A large, automatically curated dataset with reasoning chains of fragment edits and measured property deltas underpins both stages, enabling deterministic, reproducible supervision and controllable multi-hop reasoning. Unlike prior work, our framework better reasons about molecules by leveraging fragments and supports controllable refinement toward numeric targets. Experiments on generation under two sets of property constraints (QED, LogP, Molecular Weight and HOMO, LUMO) show consistent gains in validity and precise satisfaction of multi-property targets, outperforming strong LLMs and graph-based algorithms.

**arXiv ID:** 2601.10131
</details>

<details>
<summary><strong>From Single to Multi-Agent Reasoning: Advancing GeneGPT for Genomics QA</strong> - Kimia Abedini, Farzad Shami, Gianmaria Silvello - [[pdf]](https://arxiv.org/pdf/2601.10581)</summary>

**Abstract:** Comprehending genomic information is essential for biomedical research, yet extracting data from complex distributed databases remains challenging. Large language models (LLMs) offer potential for genomic Question Answering (QA) but face limitations due to restricted access to domain-specific databases. GeneGPT is the current state-of-the-art system that enhances LLMs by utilizing specialized API calls, though it is constrained by rigid API dependencies and limited adaptability. We replicate GeneGPT and propose GenomAgent, a multi-agent framework that efficiently coordinates specialized agents for complex genomics queries. Evaluated on nine tasks from the GeneTuring benchmark, GenomAgent outperforms GeneGPT by 12% on average, and its flexible architecture extends beyond genomics to various scientific domains needing expert knowledge extraction.

**arXiv ID:** 2601.10581
</details>

<details>
<summary><strong>Multi-Agent Cooperative Learning for Robust Vision-Language Alignment under OOD Concepts</strong> - Philip Xu, Isabel Wagner, Eerke Boiten - [[pdf]](https://arxiv.org/pdf/2601.09746)</summary>

**Abstract:** This paper introduces a novel Multi-Agent Cooperative Learning (MACL) framework to address cross-modal alignment collapse in vision-language models when handling out-of-distribution (OOD) concepts. Four core agents, including image, text, name, and coordination agents, collaboratively mitigate modality imbalance through structured message passing. The proposed framework enables multi-agent feature space name learning, incorporates a context exchange enhanced few-shot learning algorithm, and adopts an adaptive dynamic balancing mechanism to regulate inter-agent contributions. Experiments on the VISTA-Beyond dataset demonstrate that MACL significantly improves performance in both few-shot and zero-shot settings, achieving 1-5% precision gains across diverse visual domains.

**arXiv ID:** 2601.09746
</details>

<details>
<summary><strong>SAGE: Tool-Augmented LLM Task Solving Strategies in Scalable Multi-Agent Environments</strong> - Robert K. Strehlow, Tobias Küster, Oskar F. Kupke, Brandon Llanque Kurps, Fikret Sivrikaya, Sahin Albayrak - [[pdf]](https://arxiv.org/pdf/2601.09750)</summary>

**Abstract:** Large language models (LLMs) have proven to work well in question-answering scenarios, but real-world applications often require access to tools for live information or actuation. For this, LLMs can be extended with tools, which are often defined in advance, also allowing for some fine-tuning for specific use cases. However, rapidly evolving software landscapes and individual services require the constant development and integration of new tools. Domain- or company-specific tools can greatly elevate the usefulness of an LLM, but such custom tools can be problematic to integrate, or the LLM may fail to reliably understand and use them. For this, we need strategies to define new tools and integrate them into the LLM dynamically, as well as robust and scalable zero-shot prompting methods that can make use of those tools in an efficient manner. In this paper, we present SAGE, a specialized conversational AI interface, based on the OPACA framework for tool discovery and execution. The integration with OPACA makes it easy to add new tools or services for the LLM to use, while SAGE itself presents rich extensibility and modularity. This not only provides the ability to seamlessly switch between different models (e.g. GPT, LLAMA), but also to add and select prompting methods, involving various setups of differently prompted agents for selecting and executing tools and evaluating the results. We implemented a number of task-solving strategies, making use of agentic concepts and prompting methods in various degrees of complexity, and evaluated those against a comprehensive set of benchmark services. The results are promising and highlight the distinct strengths and weaknesses of different task-solving strategies. Both SAGE and the OPACA framework, as well as the different benchmark services and results, are available as Open Source/Open Data on GitHub.

**arXiv ID:** 2601.09750
</details>

<details>
<summary><strong>LLM-Based Agentic Systems for Software Engineering: Challenges and Opportunities</strong> - Yongjian Tang, Thomas Runkler - [[pdf]](https://arxiv.org/pdf/2601.09822)</summary>

**Abstract:** Despite recent advancements in Large Language Models (LLMs), complex Software Engineering (SE) tasks require more collaborative and specialized approaches. This concept paper systematically reviews the emerging paradigm of LLM-based multi-agent systems, examining their applications across the Software Development Life Cycle (SDLC), from requirements engineering and code generation to static code checking, testing, and debugging. We delve into a wide range of topics such as language model selection, SE evaluation benchmarks, state-of-the-art agentic frameworks and communication protocols. Furthermore, we identify key challenges and outline future research opportunities, with a focus on multi-agent orchestration, human-agent coordination, computational cost optimization, and effective data collection. This work aims to provide researchers and practitioners with valuable insights into the current forefront landscape of agentic systems within the software engineering domain.

**arXiv ID:** 2601.09822
</details>

<details>
<summary><strong>TopoDIM: One-shot Topology Generation of Diverse Interaction Modes for Multi-Agent Systems</strong> - Rui Sun, Jie Ding, Chenghua Gong, Tianjun Gu, Yihang Jiang, Juyuan Zhang, Liming Pan, Linyuan Lü - [[pdf]](https://arxiv.org/pdf/2601.10120)</summary>

**Abstract:** Optimizing communication topology in LLM-based multi-agent system is critical for enabling collective intelligence. Existing methods mainly rely on spatio-temporal interaction paradigms, where the sequential execution of multi-round dialogues incurs high latency and computation. Motivated by the recent insights that evaluation and debate mechanisms can improve problem-solving in multi-agent systems, we propose TopoDIM, a framework for one-shot Topology generation with Diverse Interaction Modes. Designed for decentralized execution to enhance adaptability and privacy, TopoDIM enables agents to autonomously construct heterogeneous communication without iterative coordination, achieving token efficiency and improved task performance. Experiments demonstrate that TopoDIM reduces total token consumption by 46.41% while improving average performance by 1.50% over state-of-the-art methods. Moreover, the framework exhibits strong adaptability in organizing communication among heterogeneous agents. Code is available at: this https URL

**arXiv ID:** 2601.10120
</details>

<details>
<summary><strong>Role-Playing Agents Driven by Large Language Models: Current Status, Challenges, and Future Trends</strong> - Ye Wang, Jiaxing Chen, Hongjiang Xiao - [[pdf]](https://arxiv.org/pdf/2601.10122)</summary>

**Abstract:** In recent years, with the rapid advancement of large language models (LLMs), role-playing language agents (RPLAs) have emerged as a prominent research focus at the intersection of natural language processing (NLP) and human-computer interaction. This paper systematically reviews the current development and key technologies of RPLAs, delineating the technological evolution from early rule-based template paradigms, through the language style imitation stage, to the cognitive simulation stage centered on personality modeling and memory mechanisms. It summarizes the critical technical pathways supporting high-quality role-playing, including psychological scale-driven character modeling, memory-augmented prompting mechanisms, and motivation-situation-based behavioral decision control. At the data level, the paper further analyzes the methods and challenges of constructing role-specific corpora, focusing on data sources, copyright constraints, and structured annotation processes. In terms of evaluation, it collates multi-dimensional assessment frameworks and benchmark datasets covering role knowledge, personality fidelity, value alignment, and interactive hallucination, while commenting on the advantages and disadvantages of methods such as human evaluation, reward models, and LLM-based scoring. Finally, the paper outlines future development directions of role-playing agents, including personality evolution modeling, multi-agent collaborative narrative, multimodal immersive interaction, and integration with cognitive neuroscience, aiming to provide a systematic perspective and methodological insights for subsequent research.

**arXiv ID:** 2601.10122
</details>

<details>
<summary><strong>Step-by-Step Causality: Transparent Causal Discovery with Multi-Agent Tree-Query and Adversarial Confidence Estimation</strong> - Ziyi Ding, Chenfei Ye-Hao, Zheyuan Wang, Xiao-Ping Zhang - [[pdf]](https://arxiv.org/pdf/2601.10137)</summary>

**Abstract:** Causal discovery aims to recover ``what causes what'', but classical constraint-based methods (e.g., PC, FCI) suffer from error propagation, and recent LLM-based causal oracles often behave as opaque, confidence-free black boxes. This paper introduces Tree-Query, a tree-structured, multi-expert LLM framework that reduces pairwise causal discovery to a short sequence of queries about backdoor paths, (in)dependence, latent confounding, and causal direction, yielding interpretable judgments with robustness-aware confidence scores. Theoretical guarantees are provided for asymptotic identifiability of four pairwise relations. On data-free benchmarks derived from Mooij et al. and UCI causal graphs, Tree-Query improves structural metrics over direct LLM baselines, and a diet--weight case study illustrates confounder screening and stable, high-confidence causal conclusions. Tree-Query thus offers a principled way to obtain data-free causal priors from LLMs that can complement downstream data-driven causal discovery. Code is available at this https URL.

**arXiv ID:** 2601.10137
</details>

<details>
<summary><strong>Learning Latency-Aware Orchestration for Parallel Multi-Agent Systems</strong> - Xi Shi, Mengxin Zheng, Qian Lou - [[pdf]](https://arxiv.org/pdf/2601.10560)</summary>

**Abstract:** Multi-agent systems (MAS) enable complex reasoning by coordinating multiple agents, but often incur high inference latency due to multi-step execution and repeated model invocations, severely limiting their scalability and usability in time-sensitive scenarios. Most existing approaches primarily optimize task performance and inference cost, and explicitly or implicitly assume sequential execution, making them less optimal for controlling latency under parallel execution. In this work, we investigate learning-based orchestration of multi-agent systems with explicit latency supervision under parallel execution. We propose Latency-Aware Multi-agent System (LAMaS), a latency-aware multi-agent orchestration framework that enables parallel execution and explicitly optimizes the critical execution path, allowing the controller to construct execution topology graphs with lower latency under parallel execution. Our experiments show that our approach reduces critical path length by 38-46% compared to the state-of-the-art baseline for multi-agent architecture search across multiple benchmarks, while maintaining or even improving task performance. These results highlight the importance of explicitly optimizing latency under parallel execution when designing efficient multi-agent systems. The code is available at this https URL

**arXiv ID:** 2601.10560
</details>

<details>
<summary><strong>Procedural Fairness in Multi-Agent Bandits</strong> - Joshua Caiata, Carter Blair, Kate Larson - [[pdf]](https://arxiv.org/pdf/2601.10600)</summary>

**Abstract:** In the context of multi-agent multi-armed bandits (MA-MAB), fairness is often reduced to outcomes: maximizing welfare, reducing inequality, or balancing utilities. However, evidence in psychology, economics, and Rawlsian theory suggests that fairness is also about process and who gets a say in the decisions being made. We introduce a new fairness objective, procedural fairness, which provides equal decision-making power for all agents, lies in the core, and provides for proportionality in outcomes. Empirical results confirm that fairness notions based on optimizing for outcomes sacrifice equal voice and representation, while the sacrifice in outcome-based fairness objectives (like equality and utilitarianism) is minimal under procedurally fair policies. We further prove that different fairness notions prioritize fundamentally different and incompatible values, highlighting that fairness requires explicit normative choices. This paper argues that procedural legitimacy deserves greater focus as a fairness objective, and provides a framework for putting procedural fairness into practice.

**arXiv ID:** 2601.10600
</details>

<details>
<summary><strong>Collaborative Multi-Agent Test-Time Reinforcement Learning for Reasoning</strong> - Zhiyuan Hu, Yunhai Hu, Juncheng Liu, Shuyue Stella Li, Yucheng Wang, Zhen Xu, See-Kiong Ng, Anh Tuan Luu, Xinxing Xu, Bryan Hooi, Cynthia Breazeal, Hae Won Park - [[pdf]](https://arxiv.org/pdf/2601.09667)</summary>

**Abstract:** Multi-agent systems have evolved into practical LLM-driven collaborators for many applications, gaining robustness from diversity and cross-checking. However, multi-agent RL (MARL) training is resource-intensive and unstable: co-adapting teammates induce non-stationarity, and rewards are often sparse and high-variance. Therefore, we introduce \textbf{Multi-Agent Test-Time Reinforcement Learning (MATTRL)}, a framework that injects structured textual experience into multi-agent deliberation at inference time. MATTRL forms a multi-expert team of specialists for multi-turn discussions, retrieves and integrates test-time experiences, and reaches consensus for final decision-making. We also study credit assignment for constructing a turn-level experience pool, then reinjecting it into the dialogue. Across challenging benchmarks in medicine, math, and education, MATTRL improves accuracy by an average of 3.67\% over a multi-agent baseline, and by 8.67\% over comparable single-agent baselines. Ablation studies examine different credit-assignment schemes and provide a detailed comparison of how they affect training outcomes. MATTRL offers a stable, effective and efficient path to distribution-shift-robust multi-agent reasoning without tuning.

**arXiv ID:** 2601.09667
</details>

<details>
<summary><strong>Adaptive Orchestration: Scalable Self-Evolving Multi-Agent Systems</strong> - Sathish Sampath, Anuradha Baskaran - [[pdf]](https://arxiv.org/pdf/2601.09742)</summary>

**Abstract:** As Large Language Models (LLMs) are increasingly deployed as autonomous agents, they face a critical scalability bottleneck known as the "Generalization-Specialization Dilemma." Monolithic agents equipped with extensive toolkits suffer from context pollution and attention decay, leading to hallucinations. Conversely, static multi-agent swarms introduce significant latency and resource overhead. This paper introduces a Self-Evolving Concierge System, a novel architecture utilizing a Dynamic Mixture of Experts (DMoE) approach. Unlike recent self-improving agents that rewrite their own codebase, our system preserves stability by dynamically restructuring its runtime environment: "hiring" specialized sub-agents based on real-time conversation analysis. We introduce an asynchronous "Meta-Cognition Engine" that detects capability gaps, a Least Recently Used (LRU) eviction policy for resource constraints, and a novel "Surgical History Pruning" mechanism to mitigate refusal bias. Experimental results demonstrate that this architecture maintains high task success rates while minimizing token consumption compared to static agent swarms.

**arXiv ID:** 2601.09742
</details>

<details>
<summary><strong>When Personas Override Payoffs: Role Identity Bias in Multi-Agent LLM Decision-Making</strong> - Viswonathan Manoranjan, Snehalkumar `Neil' S. Gaikwad - [[pdf]](https://arxiv.org/pdf/2601.10102)</summary>

**Abstract:** Large language models are increasingly deployed in multi-agent systems for strategic tasks, yet how design choices such as role-based personas and payoff visibility affect reasoning remains poorly understood. We investigate whether multi-agent systems function as strategic reasoners capable of payoff optimization or as identity-driven actors that prioritize role alignment over explicit incentives. Using Nash equilibrium achievement as a diagnostic for strategic reasoning, we conduct systematic experiments across four LLM architectures (Qwen-7B, Qwen-32B, Llama-8B, Mistral-7B) in complex environmental decision-making games involving four agents. We show that role identity bias fundamentally alters strategic reasoning even when payoff-optimal equilibria exist and complete payoff information is available. Removing personas and providing explicit payoffs enables Qwen models to achieve high Nash equilibrium rates, indicating that both conditions are necessary for strategic reasoning. In contrast, personas systematically bias equilibrium selection toward socially preferred outcomes: with personas present, all of the achieved equilibria correspond to Green Transition, while models entirely fail to reach equilibrium when Tragedy of the Commons is payoff-optimal. The effect of explicit payoffs depends entirely on persona presence, revealing strong interactions between representational design choices. We also observe clear model-dependent patterns. Qwen architectures are highly sensitive to both personas and payoff visibility, whereas Llama and Mistral exhibit rigid reasoning behavior across conditions. These findings demonstrate that representational choices are substantive governance decisions that determine whether multi-agent systems act as strategic reasoners or identity-driven actors, with important implications for real-world deployment.

**arXiv ID:** 2601.10102
</details>

<details>
<summary><strong>Fairness Driven Multi-Agent Path Finding Problem</strong> - Aditi Anand, Dildar Ali, Suman Banerjee - [[pdf]](https://arxiv.org/pdf/2601.10123)</summary>

**Abstract:** The Multi-Agent Path Finding (MAPF) problem aims at finding non-conflicting paths for multiple agents from their respective sources to destinations. This problem arises in multiple real-life situations, including robot motion planning and airspace assignment for unmanned aerial vehicle movement. The problem is computationally expensive, and adding to it, the agents are rational and can misreport their private information. In this paper, we study both variants of the problem under the realm of fairness. For the non-rational agents, we propose a heuristic solution for this problem. Considering the agents are rational, we develop a mechanism and demonstrate that it is a dominant strategy, incentive compatible, and individually rational. We employ various solution methodologies to highlight the effectiveness and efficiency of the proposed solution approaches.

**arXiv ID:** 2601.10123
</details>

<details>
<summary><strong>EHRNavigator: A Multi-Agent System for Patient-Level Clinical Question Answering over Heterogeneous Electronic Health Records</strong> - Lingfei Qian, Mauro Giuffre, Yan Wang, Huan He, Qianqian Xie, Xuguang Ai, Xeuqing Peng, Fan Ma, Ruey-Ling Weng, Donald Wright, Adan Wang, Qingyu Chen, Vipina K. Keloth, Hua Xu - [[pdf]](https://arxiv.org/pdf/2601.10020)</summary>

**Abstract:** Clinical decision-making increasingly relies on timely and context-aware access to patient information within Electronic Health Records (EHRs), yet most existing natural language question-answering (QA) systems are evaluated solely on benchmark datasets, limiting their practical relevance. To overcome this limitation, we introduce EHRNavigator, a multi-agent framework that harnesses AI agents to perform patient-level question answering across heterogeneous and multimodal EHR data. We assessed its performance using both public benchmark and institutional datasets under realistic hospital conditions characterized by diverse schemas, temporal reasoning demands, and multimodal evidence integration. Through quantitative evaluation and clinician-validated chart review, EHRNavigator demonstrated strong generalization, achieving 86% accuracy on real-world cases while maintaining clinically acceptable response times. Overall, these findings confirm that EHRNavigator effectively bridges the gap between benchmark evaluation and clinical deployment, offering a robust, adaptive, and efficient solution for real-world EHR question answering.

**arXiv ID:** 2601.10020
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>PCN-Rec: Agentic Proof-Carrying Negotiation for Reliable Governance-Constrained Recommendation</strong> - Aradhya Dixit, Shreem Dixit - [[pdf]](https://arxiv.org/pdf/2601.09771)</summary>

**Abstract:** Modern LLM-based recommenders can generate compelling ranked lists, but they struggle to reliably satisfy governance constraints such as minimum long-tail exposure or diversity requirements. We present PCN-Rec, a proof-carrying negotiation pipeline that separates natural-language reasoning from deterministic enforcement. A base recommender (MF/CF) produces a candidate window of size W, which is negotiated by two agents: a User Advocate optimizing relevance and a Policy Agent enforcing constraints. A mediator LLM synthesizes a top-N slate together with a structured certificate (JSON) describing the claimed constraint satisfaction. A deterministic verifier recomputes all constraints from the slate and accepts only verifier-checked certificates; if verification fails, a deterministic constrained-greedy repair produces a compliant slate for re-verification, yielding an auditable trace. On MovieLens-100K with governance constraints, PCN-Rec achieves a 98.55% pass rate on feasible users (n = 551, W = 80) versus a one-shot single-LLM baseline without verification/repair, while preserving utility with only a 0.021 absolute drop in NDCG@10 (0.403 vs. 0.424); differences are statistically significant (p < 0.05).

**arXiv ID:** 2601.09771
</details>

<details>
<summary><strong>CaMeLs Can Use Computers Too: System-level Security for Computer Use Agents</strong> - Hanna Foerster, Robert Mullins, Tom Blanchard, Nicolas Papernot, Kristina Nikolić, Florian Tramèr, Ilia Shumailov, Cheng Zhang, Yiren Zhao - [[pdf]](https://arxiv.org/pdf/2601.09923)</summary>

**Abstract:** AI agents are vulnerable to prompt injection attacks, where malicious content hijacks agent behavior to steal credentials or cause financial loss. The only known robust defense is architectural isolation that strictly separates trusted task planning from untrusted environment observations. However, applying this design to Computer Use Agents (CUAs) -- systems that automate tasks by viewing screens and executing actions -- presents a fundamental challenge: current agents require continuous observation of UI state to determine each action, conflicting with the isolation required for security. We resolve this tension by demonstrating that UI workflows, while dynamic, are structurally predictable. We introduce Single-Shot Planning for CUAs, where a trusted planner generates a complete execution graph with conditional branches before any observation of potentially malicious content, providing provable control flow integrity guarantees against arbitrary instruction injections. Although this architectural isolation successfully prevents instruction injections, we show that additional measures are needed to prevent Branch Steering attacks, which manipulate UI elements to trigger unintended valid paths within the plan. We evaluate our design on OSWorld, and retain up to 57% of the performance of frontier models while improving performance for smaller open-source models by up to 19%, demonstrating that rigorous security and utility can coexist in CUAs.

**arXiv ID:** 2601.09923
</details>

<details>
<summary><strong>Eliminating Agentic Workflow for Introduction Generation with Parametric Stage Tokens</strong> - Meicong Zhang, Tiancheng su, Guoxiu He - [[pdf]](https://arxiv.org/pdf/2601.09728)</summary>

**Abstract:** In recent years, using predefined agentic workflows to guide large language models (LLMs) for literature classification and review has become a research focus. However, writing research introductions is more challenging. It requires rigorous logic, coherent structure, and abstract summarization. Existing workflows often suffer from long reasoning chains, error accumulation, and reduced textual coherence. To address these limitations, we propose eliminating external agentic workflows. Instead, we directly parameterize their logical structure into the LLM. This allows the generation of a complete introduction in a single inference. To this end, we introduce the Stage Token for Introduction Generation (STIG). STIG converts the multiple stages of the original workflow into explicit stage signals. These signals guide the model to follow different logical roles and functions during generation. Through instruction tuning, the model learns the mapping between stage tokens and text functions. It also learns the logical order and transition patterns between stages, encoding this knowledge into the model parameters. Experimental results show that STIG can generate multi-stage text in a single inference. It does not require explicit workflow calls. STIG outperforms traditional agentic workflows and other baselines on metrics of semantic similarity and sentence-level structural rationality. The code is provided in the Supplementary Materials.

**arXiv ID:** 2601.09728
</details>

<details>
<summary><strong>Performance of AI agents based on reasoning language models on ALD process optimization tasks</strong> - Angel Yanguas-Gil - [[pdf]](https://arxiv.org/pdf/2601.09980)</summary>

**Abstract:** In this work we explore the performance and behavior of reasoning large language models to autonomously optimize atomic layer deposition (ALD) processes. In the ALD process optimization task, an agent built on top of a reasoning LLM has to find optimal dose times for an ALD precursor and a coreactant without any prior knowledge on the process, including whether it is actually self-limited. The agent is meant to interact iteratively with an ALD reactor in a fully unsupervised way. We evaluate this agent using a simple model of an ALD tool that incorporates ALD processes with different self-limited surface reaction pathways as well as a non self-limited component. Our results show that agents based on reasoning models like OpenAI's o3 and GPT5 consistently succeeded at completing this optimization task. However, we observed significant run-to-run variability due to the non deterministic nature of the model's response. In order to understand the logic followed by the reasoning model, the agent uses a two step process in which the model first generates an open response detailing the reasoning process. This response is then transformed into a structured output. An analysis of these reasoning traces showed that the logic of the model was sound and that its reasoning was based on the notions of self-limited process and saturation expected in the case of ALD. However, the agent can sometimes be misled by its own prior choices when exploring the optimization space.

**arXiv ID:** 2601.09980
</details>

<details>
<summary><strong>AWED-FiNER: Agents, Web applications, and Expert Detectors for Fine-grained Named Entity Recognition across 36 Languages for 6.6 Billion Speakers</strong> - Prachuryya Kaushik, Ashish Anand - [[pdf]](https://arxiv.org/pdf/2601.10161)</summary>

**Abstract:** We introduce AWED-FiNER, an open-source ecosystem designed to bridge the gap in Fine-grained Named Entity Recognition (FgNER) for 36 global languages spoken by more than 6.6 billion people. While Large Language Models (LLMs) dominate general Natural Language Processing (NLP) tasks, they often struggle with low-resource languages and fine-grained NLP tasks. AWED-FiNER provides a collection of agentic toolkits, web applications, and several state-of-the-art expert models that provides FgNER solutions across 36 languages. The agentic tools enable to route multilingual text to specialized expert models and fetch FgNER annotations within seconds. The web-based platforms provide ready-to-use FgNER annotation service for non-technical users. Moreover, the collection of language specific extremely small sized open-source state-of-the-art expert models facilitate offline deployment in resource contraint scenerios including edge devices. AWED-FiNER covers languages spoken by over 6.6 billion people, including a specific focus on vulnerable languages such as Bodo, Manipuri, Bishnupriya, and Mizo. The resources can be accessed here: Agentic Tool (this https URL), Web Application (this https URL), and 49 Expert Detector Models (this https URL).

**arXiv ID:** 2601.10161
</details>

<details>
<summary><strong>See Less, Drive Better: Generalizable End-to-End Autonomous Driving via Foundation Models Stochastic Patch Selection</strong> - Amir Mallak, Erfan Aasi, Shiva Sreeram, Tsun-Hsuan Wang, Daniela Rus, Alaa Maalouf - [[pdf]](https://arxiv.org/pdf/2601.10707)</summary>

**Abstract:** Recent advances in end-to-end autonomous driving show that policies trained on patch-aligned features extracted from foundation models generalize better to Out-of-Distribution (OOD). We hypothesize that due to the self-attention mechanism, each patch feature implicitly embeds/contains information from all other patches, represented in a different way and intensity, making these descriptors highly redundant. We quantify redundancy in such (BLIP2) features via PCA and cross-patch similarity: $90$% of variance is captured by $17/64$ principal components, and strong inter-token correlations are pervasive. Training on such overlapping information leads the policy to overfit spurious correlations, hurting OOD robustness. We present Stochastic-Patch-Selection (SPS), a simple yet effective approach for learning policies that are more robust, generalizable, and efficient. For every frame, SPS randomly masks a fraction of patch descriptors, not feeding them to the policy model, while preserving the spatial layout of the remaining patches. Thus, the policy is provided with different stochastic but complete views of the (same) scene: every random subset of patches acts like a different, yet still sensible, coherent projection of the world. The policy thus bases its decisions on features that are invariant to which specific tokens survive. Extensive experiments confirm that across all OOD scenarios, our method outperforms the state of the art (SOTA), achieving a $6.2$% average improvement and up to $20.4$% in closed-loop simulations, while being $2.4\times$ faster. We conduct ablations over masking rates and patch-feature reorganization, training and evaluating 9 systems, with 8 of them surpassing prior SOTA. Finally, we show that the same learned policy transfers to a physical, real-world car without any tuning.

**arXiv ID:** 2601.10707
</details>

<details>
<summary><strong>In-Browser Agents for Search Assistance</strong> - Saber Zerhoudi, Michael Granitzer - [[pdf]](https://arxiv.org/pdf/2601.09928)</summary>

**Abstract:** A fundamental tension exists between the demand for sophisticated AI assistance in web search and the need for user data privacy. Current centralized models require users to transmit sensitive browsing data to external services, which limits user control. In this paper, we present a browser extension that provides a viable in-browser alternative. We introduce a hybrid architecture that functions entirely on the client side, combining two components: (1) an adaptive probabilistic model that learns a user's behavioral policy from direct feedback, and (2) a Small Language Model (SLM), running in the browser, which is grounded by the probabilistic model to generate context-aware suggestions. To evaluate this approach, we conducted a three-week longitudinal user study with 18 participants. Our results show that this privacy-preserving approach is highly effective at adapting to individual user behavior, leading to measurably improved search efficiency. This work demonstrates that sophisticated AI assistance is achievable without compromising user privacy or data control.

**arXiv ID:** 2601.09928
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

<details>
<summary><strong>GUI-Eyes: Tool-Augmented Perception for Visual Grounding in GUI Agents</strong> - Chen Chen, Jiawei Shao, Dakuan Lu, Haoyi Hu, Xiangcheng Liu, Hantao Yao, Wu Liu - [[pdf]](https://arxiv.org/pdf/2601.09770)</summary>

**Abstract:** Recent advances in vision-language models (VLMs) and reinforcement learning (RL) have driven progress in GUI automation. However, most existing methods rely on static, one-shot visual inputs and passive perception, lacking the ability to adaptively determine when, whether, and how to observe the interface. We present GUI-Eyes, a reinforcement learning framework for active visual perception in GUI tasks. To acquire more informative observations, the agent learns to make strategic decisions on both whether and how to invoke visual tools, such as cropping or zooming, within a two-stage reasoning process. To support this behavior, we introduce a progressive perception strategy that decomposes decision-making into coarse exploration and fine-grained grounding, coordinated by a two-level policy. In addition, we design a spatially continuous reward function tailored to tool usage, which integrates both location proximity and region overlap to provide dense supervision and alleviate the reward sparsity common in GUI environments. On the ScreenSpot-Pro benchmark, GUI-Eyes-3B achieves 44.8% grounding accuracy using only 3k labeled samples, significantly outperforming both supervised and RL-based baselines. These results highlight that tool-aware active perception, enabled by staged policy reasoning and fine-grained reward feedback, is critical for building robust and data-efficient GUI agents.

**arXiv ID:** 2601.09770
</details>

<details>
<summary><strong>Self-Organizing Dual-Buffer Adaptive Clustering Experience Replay (SODASER) for Safe Reinforcement Learning in Optimal Control</strong> - Roya Khalili Amirabadi, Mohsen Jalaeian Farimani, Omid Solaymani Fard - [[pdf]](https://arxiv.org/pdf/2601.06540)</summary>

**Abstract:** This paper proposes a novel reinforcement learning framework, named Self-Organizing Dual-buffer Adaptive Clustering Experience Replay (SODACER), designed to achieve safe and scalable optimal control of nonlinear systems. The proposed SODACER mechanism consisting of a Fast-Buffer for rapid adaptation to recent experiences and a Slow-Buffer equipped with a self-organizing adaptive clustering mechanism to maintain diverse and non-redundant historical experiences. The adaptive clustering mechanism dynamically prunes redundant samples, optimizing memory efficiency while retaining critical environmental patterns. The approach integrates SODASER with Control Barrier Functions (CBFs) to guarantee safety by enforcing state and input constraints throughout the learning process. To enhance convergence and stability, the framework is combined with the Sophia optimizer, enabling adaptive second-order gradient updates. The proposed SODACER-Sophia's architecture ensures reliable, effective, and robust learning in dynamic, safety-critical environments, offering a generalizable solution for applications in robotics, healthcare, and large-scale system optimization. The proposed approach is validated on a nonlinear Human Papillomavirus (HPV) transmission model with multiple control inputs and safety constraints. Comparative evaluations against random and clustering-based experience replay methods demonstrate that SODACER achieves faster convergence, improved sample efficiency, and a superior bias-variance trade-off, while maintaining safe system trajectories, validated via the Friedman test.

**arXiv ID:** 2601.06540
</details>

<details>
<summary><strong>OUTLINEFORGE: Hierarchical Reinforcement Learning with Explicit States for Scientific Writing</strong> - Yilin Bao, Ziyao He, Zayden Yang - [[pdf]](https://arxiv.org/pdf/2601.09858)</summary>

**Abstract:** Scientific paper generation requires document-level planning and factual grounding, but current large language models, despite their strong local fluency, often fail in global structure, input coverage, and citation consistency. We present a reinforcement learning framework that casts scientific outline construction as a long-horizon planning problem over hierarchical document structures. Our approach models edit evolving outlines through structured actions, enabling the system to incrementally build a complete scientific manuscript. To support effective and stabilize learning,we introduce a two-stage optimization procedure consisting of (i) backward outline reconstruction from partial plans to enforce global structural consistency, and (ii) forward value-guided reinforcement learning with rewards explicitly modeling scientific correctness, discourse coherence, and citation fidelity. In addition, We further introduce a benchmark for scientific paper generation that evaluates document planning, input utilization, reference faithfulness, outline organization, and content-level factual accuracy. Our results show consistent improvements over strong neural and LLM baselines, particularly in long-range structural coherence and citation reliability.

**arXiv ID:** 2601.09858
</details>

<details>
<summary><strong>Sparse-RL: Breaking the Memory Wall in LLM Reinforcement Learning via Stable Sparse Rollouts</strong> - Sijia Luo, Xiaokang Zhang, Yuxuan Hu, Bohan Zhang, Ke Wang, Jinbo Su, Mengshu Sun, Lei Liang, Jing Zhang - [[pdf]](https://arxiv.org/pdf/2601.10079)</summary>

**Abstract:** Reinforcement Learning (RL) has become essential for eliciting complex reasoning capabilities in Large Language Models (LLMs). However, the substantial memory overhead of storing Key-Value (KV) caches during long-horizon rollouts acts as a critical bottleneck, often prohibiting efficient training on limited hardware. While existing KV compression techniques offer a remedy for inference, directly applying them to RL training induces a severe policy mismatch, leading to catastrophic performance collapse. To address this, we introduce Sparse-RL empowers stable RL training under sparse rollouts. We show that instability arises from a fundamental policy mismatch among the dense old policy, the sparse sampler policy, and the learner policy. To mitigate this issue, Sparse-RL incorporates Sparsity-Aware Rejection Sampling and Importance-based Reweighting to correct the off-policy bias introduced by compression-induced information loss. Experimental results show that Sparse-RL reduces rollout overhead compared to dense baselines while preserving the performance. Furthermore, Sparse-RL inherently implements sparsity-aware training, significantly enhancing model robustness during sparse inference deployment.

**arXiv ID:** 2601.10079
</details>

<details>
<summary><strong>HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning</strong> - Ziang Cui, Mengran Yu, Tianjiao Li, Chenyu Shi, Yingxuan Shi, Lusheng Zhang, Hongwei Lin - [[pdf]](https://arxiv.org/pdf/2601.10187)</summary>

**Abstract:** Large Language Models (LLMs) have achieved remarkable strides in multilingual translation but are hindered by a systemic cross-lingual verbosity bias, rendering them unsuitable for strict time-constrained tasks like subtitling and dubbing. Current prompt-engineering approaches struggle to resolve this conflict between semantic fidelity and rigid temporal feasibility. To bridge this gap, we first introduce Sand-Glass, a benchmark specifically designed to evaluate translation under syllable-level duration constraints. Furthermore, we propose HOMURA, a reinforcement learning framework that explicitly optimizes the trade-off between semantic preservation and temporal compliance. By employing a KL-regularized objective with a novel dynamic syllable-ratio reward, HOMURA effectively "tames" the output length. Experimental results demonstrate that our method significantly outperforms strong LLM baselines, achieving precise length control that respects linguistic density hierarchies without compromising semantic adequacy.

**arXiv ID:** 2601.10187
</details>

<details>
<summary><strong>Agent Skills in the Wild: An Empirical Study of Security Vulnerabilities at Scale</strong> - Yi Liu, Weizhe Wang, Ruitao Feng, Yao Zhang, Guangquan Xu, Gelei Deng, Yuekang Li, Leo Zhang - [[pdf]](https://arxiv.org/pdf/2601.10338)</summary>

**Abstract:** The rise of AI agent frameworks has introduced agent skills, modular packages containing instructions and executable code that dynamically extend agent capabilities. While this architecture enables powerful customization, skills execute with implicit trust and minimal vetting, creating a significant yet uncharacterized attack surface. We conduct the first large-scale empirical security analysis of this emerging ecosystem, collecting 42,447 skills from two major marketplaces and systematically analyzing 31,132 using SkillScan, a multi-stage detection framework integrating static analysis with LLM-based semantic classification. Our findings reveal pervasive security risks: 26.1% of skills contain at least one vulnerability, spanning 14 distinct patterns across four categories: prompt injection, data exfiltration, privilege escalation, and supply chain risks. Data exfiltration (13.3%) and privilege escalation (11.8%) are most prevalent, while 5.2% of skills exhibit high-severity patterns strongly suggesting malicious intent. We find that skills bundling executable scripts are 2.12x more likely to contain vulnerabilities than instruction-only skills (OR=2.12, p<0.001). Our contributions include: (1) a grounded vulnerability taxonomy derived from 8,126 vulnerable skills, (2) a validated detection methodology achieving 86.7% precision and 82.5% recall, and (3) an open dataset and detection toolkit to support future research. These results demonstrate an urgent need for capability-based permission systems and mandatory security vetting before this attack vector is further exploited.

**arXiv ID:** 2601.10338
</details>

<details>
<summary><strong>OctoBench: Benchmarking Scaffold-Aware Instruction Following in Repository-Grounded Agentic Coding</strong> - Deming Ding, Shichun Liu, Enhui Yang, Jiahang Lin, Ziying Chen, Shihan Dou, Honglin Guo, Weiyu Cheng, Pengyu Zhao, Chengjun Xiao, Qunhong Zeng, Qi Zhang, Xuanjing Huang, Qidi Xu, Tao Gui - [[pdf]](https://arxiv.org/pdf/2601.10343)</summary>

**Abstract:** Modern coding scaffolds turn LLMs into capable software agents, but their ability to follow scaffold-specified instructions remains under-examined, especially when constraints are heterogeneous and persist across interactions. To fill this gap, we introduce OctoBench, which benchmarks scaffold-aware instruction following in repository-grounded agentic coding. OctoBench includes 34 environments and 217 tasks instantiated under three scaffold types, and is paired with 7,098 objective checklist items. To disentangle solving the task from following the rules, we provide an automated observation-and-scoring toolkit that captures full trajectories and performs fine-grained checks. Experiments on eight representative models reveal a systematic gap between task-solving and scaffold-aware compliance, underscoring the need for training and evaluation that explicitly targets heterogeneous instruction following. We release the benchmark to support reproducible benchmarking and to accelerate the development of more scaffold-aware coding agents.

**arXiv ID:** 2601.10343
</details>

<details>
<summary><strong>Projected Microbatch Accumulation yields reference-free proximal policy updates for reinforcement learning</strong> - Nilin Abrahamsen - [[pdf]](https://arxiv.org/pdf/2601.10498)</summary>

**Abstract:** This note introduces Projected Microbatch Accumulation (PROMA), a proximal policy update method for large language model fine-tuning. PROMA accumulates policy gradients across microbatches by projecting out sequence-wise gradient components before microbatch aggregation. The projection is applied layer-wise during the backward pass, enabling efficient implementation without additional forward or backward passes. Empirically, PROMA enforces tighter control of local KL divergence than GRPO, resulting in more stable policy learning. Unlike PPO and GRPO, PROMA achieves proximal updates without inducing entropy collapse and does not rely on a reference policy or likelihood-ratio clipping.

**arXiv ID:** 2601.10498
</details>

<details>
<summary><strong>Grounding Agent Memory in Contextual Intent</strong> - Ruozhen Yang, Yucheng Jiang, Yueqi Jiang, Priyanka Kargupta, Yunyi Zhang, Jiawei Han - [[pdf]](https://arxiv.org/pdf/2601.10702)</summary>

**Abstract:** Deploying large language models in long-horizon, goal-oriented interactions remains challenging because similar entities and facts recur under different latent goals and constraints, causing memory systems to retrieve context-mismatched evidence. We propose STITCH (Structured Intent Tracking in Contextual History), an agentic memory system that indexes each trajectory step with a structured retrieval cue, contextual intent, and retrieves history by matching the current step's intent. Contextual intent provides compact signals that disambiguate repeated mentions and reduce interference: (1) the current latent goal defining a thematic segment, (2) the action type, and (3) the salient entity types anchoring which attributes matter. During inference, STITCH filters and prioritizes memory snippets by intent compatibility, suppressing semantically similar but context-incompatible history.
For evaluation, we introduce CAME-Bench, a benchmark for context-aware retrieval in realistic, dynamic, goal-oriented trajectories. Across CAME-Bench and LongMemEval, STITCH achieves state-of-the-art performance, outperforming the strongest baseline by 35.6%, with the largest gains as trajectory length increases. Our analysis shows that intent indexing substantially reduces retrieval noise, supporting intent-aware memory for robust long-horizon reasoning.

**arXiv ID:** 2601.10702
</details>

<details>
<summary><strong>An intelligent agent-based simulation of human mobility in extreme urban morphologies</strong> - Abderaouf Bahi, Amel Ourici - [[pdf]](https://arxiv.org/pdf/2507.15143)</summary>

**Abstract:** This paper investigates the feasibility of human mobility in extreme urban morphologies, characterized by high-density vertical structures and linear city layouts. To assess whether agents can navigate efficiently within such unprecedented topologies, we develop a hybrid simulation framework that integrates agent-based modeling, reinforcement learning (RL), supervised learning, and graph neural networks (GNNs). The simulation captures multi-modal transportation behaviors across multiple vertical levels and varying density scenarios, using both synthetic data and real-world traces from high-density cities. Experiments show that the full AI-integrated architecture enables agents to achieve an average commute time of 7.8--8.4 minutes, a satisfaction rate exceeding 89\%, and a reachability index over 91\%, even during peak congestion periods. Ablation studies indicate that removing intelligent modules such as RL or GNN significantly degrades performance, with commute times increasing by up to 85\% and reachability falling below 70\%. Environmental modeling demonstrates low energy consumption and minimal CO$_2$ emissions when electric modes are prioritized. These results suggest that efficient and sustainable mobility in extreme urban forms is achievable, provided adaptive AI systems, intelligent infrastructure, and real-time feedback mechanisms are implemented.

**arXiv ID:** 2507.15143
</details>

<details>
<summary><strong>Compartmentalised Agentic Reasoning for Clinical NLI</strong> - Maël Jullien, Lei Xu, Marco Valentino, André Freitas - [[pdf]](https://arxiv.org/pdf/2509.10222)</summary>

**Abstract:** Large language models can produce fluent judgments for clinical natural language inference, yet they frequently fail when the decision requires the correct inferential schema rather than surface matching. We introduce CARENLI, a compartmentalised agentic framework that routes each premise-statement pair to a reasoning family and then applies a specialised solver with explicit verification and targeted refinement. We evaluate on an expanded CTNLI benchmark of 200 instances spanning four reasoning families: Causal Attribution, Compositional Grounding, Epistemic Verification, and Risk State Abstraction. Across four contemporary backbone models, CARENLI improves mean accuracy from about 23% with direct prompting to about 57%, a gain of roughly 34 points, with the largest benefits on structurally demanding reasoning types. These results support compartmentalisation plus verification as a practical route to more reliable and auditable clinical inference.

**arXiv ID:** 2509.10222
</details>

<details>
<summary><strong>Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics</strong> - Dongyoung Kim, Sumin Park, Huiwon Jang, Jinwoo Shin, Jaehyung Kim, Younggyo Seo - [[pdf]](https://arxiv.org/pdf/2506.00070)</summary>

**Abstract:** Large Vision-Language Models (LVLMs) have recently shown great promise in advancing robotics by combining embodied reasoning with robot control. A common approach involves training on embodied reasoning tasks related to robot control using Supervised Fine-Tuning (SFT). However, SFT datasets are often heuristically constructed and not explicitly optimized for improving robot control. Furthermore, SFT often leads to issues such as catastrophic forgetting and reduced generalization performance. To address these limitations, we introduce Robot-R1, a novel framework that leverages reinforcement learning to enhance embodied reasoning specifically for robot control. Robot-R1 learns to predict the next keypoint state required for task completion, conditioned on the current scene image and environment metadata derived from expert demonstrations. Inspired by the DeepSeek-R1 learning approach, Robot-R1 samples reasoning-based responses and reinforces those that lead to more accurate predictions. To rigorously evaluate Robot-R1, we also introduce a new benchmark that demands the diverse embodied reasoning capabilities for the task. Our experiments show that models trained with Robot-R1 outperform SFT methods on embodied reasoning tasks. Despite having only 7B parameters, Robot-R1 even surpasses GPT-4o on reasoning tasks related to low-level action control, such as spatial and movement reasoning.

**arXiv ID:** 2506.00070
</details>

<details>
<summary><strong>TeleMem: Building Long-Term and Multimodal Memory for Agentic AI</strong> - Chunliang Chen, Ming Guan, Xiao Lin, Jiaxu Li, Luxi Lin, Qiyi Wang, Xiangyu Chen, Jixiang Luo, Changzhi Sun, Dell Zhang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2601.06037)</summary>

**Abstract:** Large language models (LLMs) excel at many NLP tasks but struggle to sustain long-term interactions due to limited attention over extended dialogue histories. Retrieval-augmented generation (RAG) mitigates this issue but lacks reliable mechanisms for updating or refining stored memories, leading to schema-driven hallucinations, inefficient write operations, and minimal support for multimodal this http URL address these challenges, we propose TeleMem, a unified long-term and multimodal memory system that maintains coherent user profiles through narrative dynamic extraction, ensuring that only dialogue-grounded information is preserved. TeleMem further introduces a structured writing pipeline that batches, retrieves, clusters, and consolidates memory entries, substantially improving storage efficiency, reducing token usage, and accelerating memory operations. Additionally, a multimodal memory module combined with ReAct-style reasoning equips the system with a closed-loop observe, think, and act process that enables accurate understanding of complex video content in long-term contexts. Experimental results show that TeleMem surpasses the state-of-the-art Mem0 baseline with 19% higher accuracy, 43% fewer tokens, and a 2.1x speedup on the ZH-4O long-term role-play gaming benchmark.

**arXiv ID:** 2601.06037
</details>

<details>
<summary><strong>Reinforcement Learning to Discover a NorthEast Monsoon Index for Monthly Rainfall Prediction in Thailand</strong> - Kiattikun Chobtham - [[pdf]](https://arxiv.org/pdf/2601.10181)</summary>

**Abstract:** 

**arXiv ID:** 2601.10181
</details>

<details>
<summary><strong>CS-GBA: A Critical Sample-based Gradient-guided Backdoor Attack for Offline Reinforcement Learning</strong> - Yuanjie Zhao, Junnan Qiu, Yue Ding, Jie Li - [[pdf]](https://arxiv.org/pdf/2601.10407)</summary>

**Abstract:** Offline Reinforcement Learning (RL) enables policy optimization from static datasets but is inherently vulnerable to backdoor attacks. Existing attack strategies typically struggle against safety-constrained algorithms (e.g., CQL) due to inefficient random poisoning and the use of easily detectable Out-of-Distribution (OOD) triggers. In this paper, we propose CS-GBA (Critical Sample-based Gradient-guided Backdoor Attack), a novel framework designed to achieve high stealthiness and destructiveness under a strict budget. Leveraging the theoretical insight that samples with high Temporal Difference (TD) errors are pivotal for value function convergence, we introduce an adaptive Critical Sample Selection strategy that concentrates the attack budget on the most influential transitions. To evade OOD detection, we propose a Correlation-Breaking Trigger mechanism that exploits the physical mutual exclusivity of state features (e.g., 95th percentile boundaries) to remain statistically concealed. Furthermore, we replace the conventional label inversion with a Gradient-Guided Action Generation mechanism, which searches for worst-case actions within the data manifold using the victim Q-network's gradient. Empirical results on D4RL benchmarks demonstrate that our method significantly outperforms state-of-the-art baselines, achieving high attack success rates against representative safety-constrained algorithms with a minimal 5% poisoning budget, while maintaining the agent's performance in clean environments.

**arXiv ID:** 2601.10407
</details>

<details>
<summary><strong>Reinforcement Learning with Multi-Step Lookahead Information Via Adaptive Batching</strong> - Nadav Merlis - [[pdf]](https://arxiv.org/pdf/2601.10418)</summary>

**Abstract:** We study tabular reinforcement learning problems with multiple steps of lookahead information. Before acting, the learner observes $\ell$ steps of future transition and reward realizations: the exact state the agent would reach and the rewards it would collect under any possible course of action. While it has been shown that such information can drastically boost the value, finding the optimal policy is NP-hard, and it is common to apply one of two tractable heuristics: processing the lookahead in chunks of predefined sizes ('fixed batching policies'), and model predictive control. We first illustrate the problems with these two approaches and propose utilizing the lookahead in adaptive (state-dependent) batches; we refer to such policies as adaptive batching policies (ABPs). We derive the optimal Bellman equations for these strategies and design an optimistic regret-minimizing algorithm that enables learning the optimal ABP when interacting with unknown environments. Our regret bounds are order-optimal up to a potential factor of the lookahead horizon $\ell$, which can usually be considered a small constant.

**arXiv ID:** 2601.10418
</details>

<details>
<summary><strong>CleanSurvival: Automated data preprocessing for time-to-event models using reinforcement learning</strong> - Yousef Koka, David Selby, Gerrit Großmann, Sebastian Vollmer, Kathan Pandya - [[pdf]](https://arxiv.org/pdf/2502.03946)</summary>

**Abstract:** Data preprocessing is a critical yet frequently neglected aspect of machine learning, often paid little attention despite its potentially significant impact on model performance. While automated machine learning pipelines are starting to recognize and integrate data preprocessing into their solutions for classification and regression tasks, this integration is lacking for more specialized tasks like survival or time-to-event models. As a result, survival analysis not only faces the general challenges of data preprocessing but also suffers from the lack of tailored, automated solutions in this area. To address this gap, this paper presents 'CleanSurvival', a reinforcement-learning-based solution for optimizing preprocessing pipelines, extended specifically for survival analysis. The framework can handle continuous and categorical variables, using Q-learning to select which combination of data imputation, outlier detection and feature extraction techniques achieves optimal performance for a Cox, random forest, neural network or user-supplied time-to-event model. The package is available on GitHub: this https URL Experimental benchmarks on real-world datasets show that the Q-learning-based data preprocessing results in superior predictive performance to standard approaches, finding such a model up to 10 times faster than undirected random grid search. Furthermore, a simulation study demonstrates the effectiveness in different types and levels of missingness and noise in the data.

**arXiv ID:** 2502.03946
</details>

<details>
<summary><strong>Provably Safe Reinforcement Learning for Stochastic Reach-Avoid Problems with Entropy Regularization</strong> - Abhijit Mazumdar, Rafal Wisniewski, Manuela L. Bujorianu - [[pdf]](https://arxiv.org/pdf/2601.08646)</summary>

**Abstract:** We consider the problem of learning the optimal policy for Markov decision processes with safety constraints. We formulate the problem in a reach-avoid setup. Our goal is to design online reinforcement learning algorithms that ensure safety constraints with arbitrarily high probability during the learning phase. To this end, we first propose an algorithm based on the optimism in the face of uncertainty (OFU) principle. Based on the first algorithm, we propose our main algorithm, which utilizes entropy regularization. We investigate the finite-sample analysis of both algorithms and derive their regret bounds. We demonstrate that the inclusion of entropy regularization improves the regret and drastically controls the episode-to-episode variability that is inherent in OFU-based safe RL algorithms.

**arXiv ID:** 2601.08646
</details>

<details>
<summary><strong>Probabilistic Insights for Efficient Exploration Strategies in Reinforcement Learning</strong> - Ernesto Garcia, Paola Bermolen, Matthieu Jonckheere, Seva Shneer - [[pdf]](https://arxiv.org/pdf/2503.03565)</summary>

**Abstract:** We investigate efficient exploration strategies of environments with unknown stochastic dynamics and sparse rewards. Specifically, we analyze first the impact of parallel simulations on the probability of reaching rare states within a finite time budget. Using simplified models based on random walks and Lévy processes, we provide analytical results that demonstrate a phase transition in reaching probabilities as a function of the number of parallel simulations. We identify an optimal number of parallel simulations that balances exploration diversity and time allocation. Additionally, we analyze a restarting mechanism that exponentially enhances the probability of success by redirecting efforts toward more promising regions of the state space. Our findings contribute to a more qualitative and quantitative theory of some exploration schemes in reinforcement learning, offering insights into developing more efficient strategies for environments characterized by rare events.

**arXiv ID:** 2503.03565
</details>

<details>
<summary><strong>OBLR-PO: A Theoretical Framework for Stable Reinforcement Learning</strong> - Zixun Huang, Jiayi Sheng, Zeyu Zheng - [[pdf]](https://arxiv.org/pdf/2511.23310)</summary>

**Abstract:** Existing reinforcement learning (RL)-based post-training methods for large language models have advanced rapidly, yet their design has largely been guided by heuristics rather than systematic theoretical principles. This gap limits our understanding of the properties of the gradient estimators and the associated optimization algorithms, thereby constraining opportunities to improve training stability and overall performance. In this work, we provide a unified theoretical framework that characterizes the statistical properties of commonly used policy-gradient estimators under mild assumptions. Our analysis establishes unbiasedness, derives exact variance expressions, and yields an optimization-loss upper bound that enables principled reasoning about learning dynamics. Building on these results, we prove convergence guarantees and derive an adaptive learning-rate schedule governed by the signal-to-noise ratio (SNR) of gradients. We further show that the variance-optimal baseline is a gradient-weighted estimator, offering a new principle for variance reduction and naturally enhancing stability beyond existing methods. These insights motivate Optimal Baseline and Learning-Rate Policy Optimization (OBLR-PO), an algorithm that jointly adapts learning rates and baselines in a theoretically grounded manner. Experiments on Qwen3-4B-Base and Qwen3-8B-Base demonstrate consistent gains over existing policy optimization methods, validating that our theoretical contributions translate into practical improvements in large-scale post-training.

**arXiv ID:** 2511.23310
</details>

<details>
<summary><strong>DAVOS: An Autonomous Vehicle Operating System in the Vehicle Computing Era</strong> - Yuxin Wang, Yuankai He, Boyang Tian, Lichen Xian, Weisong Shi - [[pdf]](https://arxiv.org/pdf/2601.05072)</summary>

**Abstract:** Vehicle computing represents a fundamental shift in how autonomous vehicles are designed and deployed, transforming them from isolated transportation systems into mobile computing platforms that support both safety-critical, real-time driving and data-centric services. In this setting, vehicles simultaneously support real-time driving pipelines and a growing set of data-driven applications, placing increased responsibility on the vehicle operating system to coordinate computation, data movement, storage, and access. These demands highlight recurring system considerations related to predictable execution, data and execution protection, efficient handling of high-rate sensor data, and long-term system evolvability, commonly summarized as Safety, Security, Efficiency, and Extensibility (SSEE). Existing vehicle operating systems and runtimes address these concerns in isolation, resulting in fragmented software stacks that limit coordination between autonomy workloads and vehicle data services. This paper presents DAVOS, the Delaware Autonomous Vehicle Operating System, a unified vehicle operating system architecture designed for the vehicle computing context. DAVOS provides a cohesive operating system foundation that supports both real-time autonomy and extensible vehicle computing within a single system framework.

**arXiv ID:** 2601.05072
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
