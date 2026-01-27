# Agent arXiv Daily

**Last Updated:** 2026-01-27 03:16:34

**Total Papers:** 53

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (3 papers)</h2></summary>

<details>
<summary><strong>AgentDrive: An Open Benchmark Dataset for Agentic AI Reasoning with LLM-Generated Scenarios in Autonomous Systems</strong> - Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah - [[pdf]](https://arxiv.org/pdf/2601.16964)</summary>

**Abstract:** The rapid advancement of large language models (LLMs) has sparked growing interest in their integration into autonomous systems for reasoning-driven perception, planning, and decision-making. However, evaluating and training such agentic AI models remains challenging due to the lack of large-scale, structured, and safety-critical benchmarks. This paper introduces AgentDrive, an open benchmark dataset containing 300,000 LLM-generated driving scenarios designed for training, fine-tuning, and evaluating autonomous agents under diverse conditions. AgentDrive formalizes a factorized scenario space across seven orthogonal axes: scenario type, driver behavior, environment, road layout, objective, difficulty, and traffic density. An LLM-driven prompt-to-JSON pipeline generates semantically rich, simulation-ready specifications that are validated against physical and schema constraints. Each scenario undergoes simulation rollouts, surrogate safety metric computation, and rule-based outcome labeling. To complement simulation-based evaluation, we introduce AgentDrive-MCQ, a 100,000-question multiple-choice benchmark spanning five reasoning dimensions: physics, policy, hybrid, scenario, and comparative reasoning. We conduct a large-scale evaluation of fifty leading LLMs on AgentDrive-MCQ. Results show that while proprietary frontier models perform best in contextual and policy reasoning, advanced open models are rapidly closing the gap in structured and physics-grounded reasoning. We release the AgentDrive dataset, AgentDrive-MCQ benchmark, evaluation code, and related materials at this https URL

**arXiv ID:** 2601.16964
</details>

<details>
<summary><strong>Who You Explain To Matters: Learning by Explaining to Conversational Agents with Different Pedagogical Roles</strong> - Zhengtao Xu, Junti Zhang, Anthony Tang, Yi-Chieh Lee - [[pdf]](https://arxiv.org/pdf/2601.16583)</summary>

**Abstract:** Conversational agents are increasingly used in education for learning support. An application is "learning by explaining", where learners explain their understanding to an agent. However, existing research focuses on single roles, leaving it unclear how different pedagogical roles influence learners' interaction patterns, learning outcomes and experiences. We conducted a between-subjects study (N=96) comparing agents with three pedagogical roles (Tutee, Peer, Challenger) and a control condition while learning an economics concept. We found that different pedagogical roles shaped learning dynamics, including interaction patterns and experiences. Specifically, the Tutee agent elicited the most cognitive investment but led to high pressure. The Peer agent fostered high absorption and interest through collaborative dialogue. The Challenger agent promoted cognitive and metacognitive acts, enhancing critical thinking with moderate pressure. The findings highlight how agent roles shape different learning dynamics, guiding the design of educational agents tailored to specific pedagogical goals and learning phases.

**arXiv ID:** 2601.16583
</details>

<details>
<summary><strong>Watching AI Think: User Perceptions of Visible Thinking in Chatbots</strong> - Samuel Rhys Cox, Jade Martin-Lise, Simo Hosio, Niels van Berkel - [[pdf]](https://arxiv.org/pdf/2601.16720)</summary>

**Abstract:** People increasingly turn to conversational agents such as ChatGPT to seek guidance for their personal problems. As these systems grow in capability, many now display elements of "thinking": short reflective statements that reveal a model's intentions or values before responding. While initially introduced to promote transparency, such visible thinking can also anthropomorphise the agent and shape user expectations. Yet little is known about how these displays affect user perceptions in help-seeking contexts. We conducted a 3 x 2 mixed design experiment examining the impact of 'Thinking Content' (None, Emotionally-Supportive, Expertise-Supportive) and 'Conversation Context' (Habit-related vs. Feelings-related problems) on users' perceptions of empathy, warmth, competence, and engagement. Participants interacted with a chatbot that either showed no visible thinking or presented value-oriented reflections prior to its response. Our findings contribute to understanding how thinking transparency influences user experience in supportive dialogues, and offer implications for designing conversational agents that communicate intentions in sensitive, help-seeking scenarios.

**arXiv ID:** 2601.16720
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (14 papers)</h2></summary>

<details>
<summary><strong>SemanticALLI: Caching Reasoning, Not Just Responses, in Agentic Systems</strong> - Varun Chillara, Dylan Kline, Christopher Alvares, Evan Wooten, Huan Yang, Shlok Khetan, Cade Bauer, Tré Guillory, Tanishka Shah, Yashodhara Dhariwal, Volodymyr Pavlov, George Popstefanov - [[pdf]](https://arxiv.org/pdf/2601.16286)</summary>

**Abstract:** Agentic AI pipelines suffer from a hidden inefficiency: they frequently reconstruct identical intermediate logic, such as metric normalization or chart scaffolding, even when the user's natural language phrasing is entirely novel. Conventional boundary caching fails to capture this inefficiency because it treats inference as a monolithic black box.
We introduce SemanticALLI, a pipeline-aware architecture within Alli (PMG's marketing intelligence platform), designed to operationalize redundant reasoning. By decomposing generation into Analytic Intent Resolution (AIR) and Visualization Synthesis (VS), SemanticALLI elevates structured intermediate representations (IRs) to first-class, cacheable artifacts.
The impact of caching within the agentic loop is substantial. In our evaluation, baseline monolithic caching caps at a 38.7% hit rate due to linguistic variance. In contrast, our structured approach allows for an additional stage, the Visualization Synthesis stage, to achieve an 83.10% hit rate, bypassing 4,023 LLM calls with a median latency of just 2.66 ms. This internal reuse reduces total token consumption, offering a practical lesson for AI system design: even when users rarely repeat themselves, the pipeline often does, at stable, structured checkpoints where caching is most reliable.

**arXiv ID:** 2601.16286
</details>

<details>
<summary><strong>DSGym: A Holistic Framework for Evaluating and Training Data Science Agents</strong> - Fan Nie, Junlin Wang, Harper Hua, Federico Bianchi, Yongchan Kwon, Zhenting Qi, Owen Queen, Shang Zhu, James Zou - [[pdf]](https://arxiv.org/pdf/2601.16344)</summary>

**Abstract:** Data science agents promise to accelerate discovery and insight-generation by turning data into executable analyses and findings. Yet existing data science benchmarks fall short due to fragmented evaluation interfaces that make cross-benchmark comparison difficult, narrow task coverage and a lack of rigorous data grounding. In particular, we show that a substantial portion of tasks in current benchmarks can be solved without using the actual data. To address these limitations, we introduce DSGym, a standardized framework for evaluating and training data science agents in self-contained execution environments. Unlike static benchmarks, DSGym provides a modular architecture that makes it easy to add tasks, agent scaffolds, and tools, positioning it as a live, extensible testbed. We curate DSGym-Tasks, a holistic task suite that standardizes and refines existing benchmarks via quality and shortcut solvability filtering. We further expand coverage with (1) DSBio: expert-derived bioinformatics tasks grounded in literature and (2) DSPredict: challenging prediction tasks spanning domains such as computer vision, molecular prediction, and single-cell perturbation. Beyond evaluation, DSGym enables agent training via execution-verified data synthesis pipeline. As a case study, we build a 2,000-example training set and trained a 4B model in DSGym that outperforms GPT-4o on standardized analysis benchmarks. Overall, DSGym enables rigorous end-to-end measurement of whether agents can plan, implement, and validate data analyses in realistic scientific context.

**arXiv ID:** 2601.16344
</details>

<details>
<summary><strong>LUMINA: Long-horizon Understanding for Multi-turn Interactive Agents</strong> - Amin Rakhsha, Thomas Hehn, Pietro Mazzaglia, Fabio Valerio Massoli, Arash Behboodi, Tribhuvanesh Orekondy - [[pdf]](https://arxiv.org/pdf/2601.16649)</summary>

**Abstract:** Large language models can perform well on many isolated tasks, yet they continue to struggle on multi-turn, long-horizon agentic problems that require skills such as planning, state tracking, and long context processing. In this work, we aim to better understand the relative importance of advancing these underlying capabilities for success on such tasks. We develop an oracle counterfactual framework for multi-turn problems that asks: how would an agent perform if it could leverage an oracle to perfectly perform a specific task? The change in the agent's performance due to this oracle assistance allows us to measure the criticality of such oracle skill in the future advancement of AI agents. We introduce a suite of procedurally generated, game-like tasks with tunable complexity. These controlled environments allow us to provide precise oracle interventions, such as perfect planning or flawless state tracking, and make it possible to isolate the contribution of each oracle without confounding effects present in real-world benchmarks. Our results show that while some interventions (e.g., planning) consistently improve performance across settings, the usefulness of other skills is dependent on the properties of the environment and language model. Our work sheds light on the challenges of multi-turn agentic environments to guide the future efforts in the development of AI agents and language models.

**arXiv ID:** 2601.16649
</details>

<details>
<summary><strong>Spatial-Agent: Agentic Geo-spatial Reasoning with Scientific Core Concepts</strong> - Riyang Bao, Cheng Yang, Dazhou Yu, Zhexiang Tang, Gengchen Mai, Liang Zhao - [[pdf]](https://arxiv.org/pdf/2601.16965)</summary>

**Abstract:** Geospatial reasoning is essential for real-world applications such as urban analytics, transportation planning, and disaster response. However, existing LLM-based agents often fail at genuine geospatial computation, relying instead on web search or pattern matching while hallucinating spatial relationships. We present Spatial-Agent, an AI agent grounded in foundational theories of spatial information science. Our approach formalizes geo-analytical question answering as a concept transformation problem, where natural-language questions are parsed into executable workflows represented as GeoFlow Graphs -- directed acyclic graphs with nodes corresponding to spatial concepts and edges representing transformations. Drawing on spatial information theory, Spatial-Agent extracts spatial concepts, assigns functional roles with principled ordering constraints, and composes transformation sequences through template-based generation. Extensive experiments on MapEval-API and MapQA benchmarks demonstrate that Spatial-Agent significantly outperforms existing baselines including ReAct and Reflexion, while producing interpretable and executable geospatial workflows.

**arXiv ID:** 2601.16965
</details>

<details>
<summary><strong>Attention-MoA: Enhancing Mixture-of-Agents via Inter-Agent Semantic Attention and Deep Residual Synthesis</strong> - Jianyu Wen, Yang Wei, Xiongxi Yu, Changxuan Xiao, Ke Zeng - [[pdf]](https://arxiv.org/pdf/2601.16596)</summary>

**Abstract:** As the development of Large Language Models (LLMs) shifts from parameter scaling to inference-time collaboration, the Mixture-of-Agents (MoA) framework has emerged as a general paradigm to harness collective intelligence by layering diverse models. While recent MoA variants have introduced dynamic routing and residual connections to improve efficiency, these methods often fail to facilitate deep semantic interaction between agents, limiting the system's ability to actively correct hallucinations and refine logic. In this paper, we introduce Attention-MoA, a novel MoA-based framework that redefines collaboration through Inter-agent Semantic Attention. Complemented by an Inter-layer Residual Module with Adaptive Early Stopping Mechanism, our architecture mitigates information degradation in deep layers while improving computational efficiency. Extensive evaluations across AlpacaEval 2.0, MT-Bench, and FLASK demonstrate that Attention-MoA significantly outperforms state-of-the-art baselines, achieving a 91.15% Length-Controlled Win Rate on AlpacaEval 2.0 and dominating in 10 out of 12 capabilities on FLASK. Notably, Attention-MoA enables an ensemble of small open-source models to outperform massive proprietary models like Claude-4.5-Sonnet and GPT-4.1, achieving an MT-Bench score of 8.83 and an AlpacaEval 2.0 LC Win Rate of 77.36%.

**arXiv ID:** 2601.16596
</details>

<details>
<summary><strong>Cognitive Control Architecture (CCA): A Lifecycle Supervision Framework for Robustly Aligned AI Agents</strong> - Zhibo Liang, Tianze Hu, Zaiye Chen, Mingjie Tang - [[pdf]](https://arxiv.org/pdf/2512.06716)</summary>

**Abstract:** Autonomous Large Language Model (LLM) agents exhibit significant vulnerability to Indirect Prompt Injection (IPI) attacks. These attacks hijack agent behavior by polluting external information sources, exploiting fundamental trade-offs between security and functionality in existing defense mechanisms. This leads to malicious and unauthorized tool invocations, diverting agents from their original objectives. The success of complex IPIs reveals a deeper systemic fragility: while current defenses demonstrate some effectiveness, most defense architectures are inherently fragmented. Consequently, they fail to provide full integrity assurance across the entire task execution pipeline, forcing unacceptable multi-dimensional compromises among security, functionality, and efficiency. Our method is predicated on a core insight: no matter how subtle an IPI attack, its pursuit of a malicious objective will ultimately manifest as a detectable deviation in the action trajectory, distinct from the expected legitimate plan. Based on this, we propose the Cognitive Control Architecture (CCA), a holistic framework achieving full-lifecycle cognitive supervision. CCA constructs an efficient, dual-layered defense system through two synergistic pillars: (i) proactive and preemptive control-flow and data-flow integrity enforcement via a pre-generated "Intent Graph"; and (ii) an innovative "Tiered Adjudicator" that, upon deviation detection, initiates deep reasoning based on multi-dimensional scoring, specifically designed to counter complex conditional attacks. Experiments on the AgentDojo benchmark substantiate that CCA not only effectively withstands sophisticated attacks that challenge other advanced defense methods but also achieves uncompromised security with notable efficiency and robustness, thereby reconciling the aforementioned multi-dimensional trade-off.

**arXiv ID:** 2512.06716
</details>

<details>
<summary><strong>Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing</strong> - Yinqiu Huang, Shuli Wang, Min Gao, Xue Wei, Changhao Li, Chuan Luo, Yinhua Zhu, Xiong Xiao, Yi Luo - [[pdf]](https://arxiv.org/pdf/2402.03379)</summary>

**Abstract:** Uplift modeling, vital in online marketing, seeks to accurately measure the impact of various strategies, such as coupons or discounts, on different users by predicting the Individual Treatment Effect (ITE). In an e-commerce setting, user behavior follows a defined sequential chain, including impression, click, and conversion. Marketing strategies exert varied uplift effects at each stage within this chain, impacting metrics like click-through and conversion rate. Despite its utility, existing research has neglected to consider the inter-task across all stages impacts within a specific treatment and has insufficiently utilized the treatment information, potentially introducing substantial bias into subsequent marketing decisions. We identify these two issues as the chain-bias problem and the treatment-unadaptive problem. This paper introduces the Entire Chain UPlift method with context-enhanced learning (ECUP), devised to tackle these issues. ECUP consists of two primary components: 1) the Entire Chain-Enhanced Network, which utilizes user behavior patterns to estimate ITE throughout the entire chain space, models the various impacts of treatments on each task, and integrates task prior information to enhance context awareness across all stages, capturing the impact of treatment on different tasks, and 2) the Treatment-Enhanced Network, which facilitates fine-grained treatment modeling through bit-level feature interactions, thereby enabling adaptive feature adjustment. Extensive experiments on public and industrial datasets validate ECUPs effectiveness. Moreover, ECUP has been deployed on the Meituan food delivery platform, serving millions of daily active users, with the related dataset released for future research.

**arXiv ID:** 2402.03379
</details>

<details>
<summary><strong>EmbedAgent: Benchmarking Large Language Models in Embedded System Development</strong> - Ruiyang Xu, Jialun Cao, Mingyuan Wu, Wenliang Zhong, Yaojie Lu, Ben He, Xianpei Han, Shing-Chi Cheung, Le Sun - [[pdf]](https://arxiv.org/pdf/2506.11003)</summary>

**Abstract:** Large Language Models (LLMs) have shown promise in various tasks, yet few benchmarks assess their capabilities in embedded system development. In this paper, we introduce EmbedAgent, a paradigm designed to simulate real-world roles in embedded system development, such as Embedded System Programmer, Architect, and Integrator. This paradigm enables LLMs to be tested in tasks that bridge the gap between digital and physical systems, allowing for a more comprehensive assessment of their capabilities. To evaluate LLMs on these tasks, we propose Embedbench, the first comprehensive benchmark for embedded system programming, circuit design, and cross-platform migration. Embedbench consists of 126 cases, covering 9 electronic components across 3 hardware platforms. Through extensive experiments on 10 mainstream LLMs, we uncover several key findings. Surprisingly, despite the simplicity of the cases, DeepSeek-R1 achieves only a 55.6% pass@1 rate when provided with schematic information, and 50.0% when tasked with generating the schematics itself. In the cross-platform migration tasks, LLMs show relatively strong performance with MicroPython on the Raspberry Pi Pico (with the top model achieving 73.8% pass@1), but perform poorly on ESP-IDF, where the best model reaches only 29.4% pass@1. Interestingly, we observe that general-purpose chat LLMs like DeepSeek-V3 often fail to utilize relevant pre-trained knowledge in this domain, while reasoning LLMs tend to overthink and overlook efficient knowledge during pretraining. Based on these insights, we propose two strategies: retrieval augmented generation and compiler feedback-to enhance LLM performance. These strategies result in significant improvements, with Deepseek-R1 reaching a 65.1% pass@1 with correct schematics, and 53.1% without. Additionally, the accuracy of the Arduino to ESP32 migration task improves from 21.4% to 27.8%.

**arXiv ID:** 2506.11003
</details>

<details>
<summary><strong>Intelligent Systems in Neuroimaging: Pioneering AI Techniques for Brain Tumor Detection</strong> - Md. Mohaiminul Islam, Md. Mofazzal Hossen, Maher Ali Rusho, Nahiyan Nazah Ridita, Zarin Tasnia Shanta, Md. Simanto Haider, Ahmed Faizul Haque Dhrubo, Md. Khurshid Jahan, Mohammad Abdul Qayum - [[pdf]](https://arxiv.org/pdf/2511.17655)</summary>

**Abstract:** This study deliberates on the application of advanced AI techniques for brain tumor classification through MRI, wherein the training includes the present best deep learning models to enhance diagnosis accuracy and the potential of usability in clinical practice. By combining custom convolutional models with pre-trained neural network architectures, our approach exposes the utmost performance in the classification of four classes: glioma, meningioma, pituitary tumors, and no-tumor cases. Assessing the models on a large dataset of over 7,000 MRI images focused on detection accuracy, computational efficiency, and generalization to unseen data. The results indicate that the Xception architecture surpasses all other were tested, obtaining a testing accuracy of 98.71% with the least validation loss. While presenting this case with findings that demonstrate AI as a probable scorer in brain tumor diagnosis, we demonstrate further motivation by reducing computational complexity toward real-world clinical deployment. These aspirations offer an abundant future for progress in automated neuroimaging diagnostics.

**arXiv ID:** 2511.17655
</details>

<details>
<summary><strong>AMBER: A Columnar Architecture for High-Performance Agent-Based Modeling in Python</strong> - Anh-Duy Pham - [[pdf]](https://arxiv.org/pdf/2601.16292)</summary>

**Abstract:** Agent-based modeling (ABM) has emerged as an indispensable methodology for studying complex adaptive systems across the natural and social sciences. However, Python-based ABM frameworks face a fundamental tension between the accessibility that has made Python dominant in scientific computing and the performance requirements of large-scale simulations. This paper introduces AMBER, a framework that resolves this tension through a novel architectural approach: replacing the conventional object-per-agent representation with columnar state management using the Polars DataFrame library. We analyze the computational characteristics of both paradigms, present the architectural design of AMBER including its core abstractions, spatial environments, experiment management, and optimization capabilities. Empirical evaluation on three canonical benchmarks demonstrates that AMBER achieves speedups of 1.2x to 93x depending on workload characteristics, with the greatest advantages for models dominated by population-wide attribute operations. Memory profiling reveals 30-50% reduction in peak usage compared to object-oriented frameworks. Our results establish columnar state management as a viable architectural foundation for high-performance ABM in interpreted languages.

**arXiv ID:** 2601.16292
</details>

<details>
<summary><strong>Curate-Train-Refine: A Closed-Loop Agentic Framework for Zero Shot Classification</strong> - Gaurav Maheshwari, Kevin El Haddad - [[pdf]](https://arxiv.org/pdf/2601.16530)</summary>

**Abstract:** Large language models (LLMs) and high-capacity encoders have advanced zero and few-shot classification, but their inference cost and latency limit practical deployment. We propose training lightweight text classifiers using dynamically generated supervision from an LLM. Our method employs an iterative, agentic loop in which the LLM curates training data, analyzes model successes and failures, and synthesizes targeted examples to address observed errors. This closed-loop generation and evaluation process progressively improves data quality and adapts it to the downstream classifier and task. Across four widely used benchmarks, our approach consistently outperforms standard zero and few-shot baselines. These results indicate that LLMs can serve effectively as data curators, enabling accurate and efficient classification without the operational cost of large-model deployment.

**arXiv ID:** 2601.16530
</details>

<details>
<summary><strong>EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents</strong> - Xinze Li, Ziyue Zhu, Siyuan Liu, Yubo Ma, Yuhang Zang, Yixin Cao, Aixin Sun - [[pdf]](https://arxiv.org/pdf/2601.16690)</summary>

**Abstract:** We introduce EMemBench, a programmatic benchmark for evaluating long-term memory of agents through interactive games. Rather than using a fixed set of questions, EMemBench generates questions from each agent's own trajectory, covering both text and visual game environments. Each template computes verifiable ground truth from underlying game signals, with controlled answerability and balanced coverage over memory skills: single/multi-hop recall, induction, temporal, spatial, logical, and adversarial. We evaluate memory agents with strong LMs/VLMs as backbones, using in-context prompting as baselines. Across 15 text games and multiple visual seeds, results are far from saturated: induction and spatial reasoning are persistent bottlenecks, especially in visual setting. Persistent memory yields clear gains for open backbones on text games, but improvements are less consistent for VLM agents, suggesting that visually grounded episodic memory remains an open challenge. A human study further confirms the difficulty of EMemBench.

**arXiv ID:** 2601.16690
</details>

<details>
<summary><strong>I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search</strong> - Zujie Liang, Feng Wei, Wujiang Xu, Lin Chen, Yuxi Qian, Xinhui Wu - [[pdf]](https://arxiv.org/pdf/2502.14693)</summary>

**Abstract:** Recent advancements in large language models (LLMs) have shown remarkable potential in automating machine learning tasks. However, existing LLM-based agents often struggle with low-diversity and suboptimal code generation. While recent work has introduced Monte Carlo Tree Search (MCTS) to address these issues, limitations persist in the quality and diversity of thoughts generated, as well as in the scalar value feedback mechanisms used for node selection. In this study, we introduce Introspective Monte Carlo Tree Search (I-MCTS), a novel approach that iteratively expands tree nodes through an introspective process that meticulously analyzes solutions and results from parent and sibling nodes. This facilitates a continuous refinement of the node in the search tree, thereby enhancing the overall decision-making process. Furthermore, we integrate a Large Language Model (LLM)-based value model to facilitate direct evaluation of each node's solution prior to conducting comprehensive computational rollouts. A hybrid rewarding mechanism is implemented to seamlessly transition the Q-value from LLM-estimated scores to actual performance scores. This allows higher-quality nodes to be traversed earlier. Applied to the various ML tasks, our approach demonstrates a 4% absolute improvement in performance compared to the strong open-source AutoML agents, showcasing its effectiveness in enhancing agentic AutoML systems. Resource available at this https URL

**arXiv ID:** 2502.14693
</details>

<details>
<summary><strong>The Behavioral Fabric of LLM-Powered GUI Agents: Human Values and Interaction Outcomes</strong> - Simret Araya Gebreegziabher, Yukun Yang, Charles Chiang, Hojun Yoo, Chaoran Chen, Hyo Jin Do, Zahra Ashktorab, Werner Geyer, Diego Gómez-Zará, Toby Jia-Jun Li - [[pdf]](https://arxiv.org/pdf/2601.16356)</summary>

**Abstract:** Large Language Model (LLM)-powered web GUI agents are increasingly automating everyday online tasks. Despite their popularity, little is known about how users' preferences and values impact agents' reasoning and behavior. In this work, we investigate how both explicit and implicit user preferences, as well as the underlying user values, influence agent decision-making and action trajectories. We built a controlled testbed of 14 common interactive web tasks, spanning shopping, travel, dining, and housing, each replicated from real websites and integrated with a low-fidelity LLM-based recommender system. We injected 12 human preferences and values as personas into four state-of-the-art agents and systematically analyzed their task behaviors. Our results show that preference and value-infused prompts consistently guided agents toward outcomes that reflected these preferences and values. While the absence of user preference or value guidance led agents to exhibit a strong efficiency bias and employ shortest-path strategies, their presence steered agents' behavior trajectories through the greater use of corresponding filters and interactive web features. Despite their influence, dominant interface cues, such as discounts and advertisements, frequently overrode these effects, shortening the agents' action trajectories and inducing rationalizations that masked rather than reflected value-consistent reasoning. The contributions of this paper are twofold: (1) an open-source testbed for studying the influence of values in agent behaviors, and (2) an empirical investigation of how user preferences and values shape web agent behaviors.

**arXiv ID:** 2601.16356
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>Ready Jurist One: Benchmarking Language Agents for Legal Intelligence in Dynamic Environments</strong> - Zheng Jia, Shengbin Yue, Wei Chen, Siyuan Wang, Yidong Liu, Zejun Li, Yun Song, Zhongyu Wei - [[pdf]](https://arxiv.org/pdf/2507.04037)</summary>

**Abstract:** The gap between static benchmarks and the dynamic nature of real-world legal practice poses a key barrier to advancing legal intelligence. To this end, we introduce J1-ENVS, the first interactive and dynamic legal environment tailored for LLM-based agents. Guided by legal experts, it comprises six representative scenarios from Chinese legal practices across three levels of environmental complexity. We further introduce J1-EVAL, a fine-grained evaluation framework, designed to assess both task performance and procedural compliance across varying levels of legal proficiency. Extensive experiments on 17 LLM agents reveal that, while many models demonstrate solid legal knowledge, they struggle with procedural execution in dynamic settings. Even the SOTA model, GPT-4o, falls short of 60% overall performance. These findings highlight persistent challenges in achieving dynamic legal intelligence and offer valuable insights to guide future research.

**arXiv ID:** 2507.04037
</details>

<details>
<summary><strong>PolyAgent: Large Language Model Agent for Polymer Design</strong> - Vani Nigam, Achuth Chandrasekhar, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2601.16376)</summary>

**Abstract:** On-demand Polymer discovery is essential for various industries, ranging from biomedical to reinforcement materials. Experiments with polymers have a long trial-and-error process, leading to long procedures and extensive resources. For these processes, machine learning has accelerated scientific discovery at the property prediction and latent space search fronts. However, laboratory researchers cannot readily access codes and these models to extract individual structures and properties due to infrastructure limitations. We present a closed-loop polymer structure-property predictor integrated in a terminal for early-stage polymer discovery. The framework is powered by LLM reasoning to provide users with property prediction, property-guided polymer structure generation, and structure modification capabilities. The SMILES sequences are guided by the synthetic accessibility score and the synthetic complexity score (SC Score) to ensure that polymer generation is as close as possible to synthetically accessible monomer-level structures. This framework addresses the challenge of generating novel polymer structures for laboratory researchers, thereby providing computational insights into polymer research.

**arXiv ID:** 2601.16376
</details>

<details>
<summary><strong>SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents</strong> - Yuhang Wang, Yuling Shi, Mo Yang, Rongrui Zhang, Shilin He, Heng Lian, Yuting Chen, Siyu Ye, Kai Cai, Xiaodong Gu - [[pdf]](https://arxiv.org/pdf/2601.16746)</summary>

**Abstract:** LLM agents have demonstrated remarkable capabilities in software development, but their performance is hampered by long interaction contexts, which incur high API costs and latency. While various context compression approaches such as LongLLMLingua have emerged to tackle this challenge, they typically rely on fixed metrics such as PPL, ignoring the task-specific nature of code understanding. As a result, they frequently disrupt syntactic and logical structure and fail to retain critical implementation details. In this paper, we propose SWE-Pruner, a self-adaptive context pruning framework tailored for coding agents. Drawing inspiration from how human programmers "selectively skim" source code during development and debugging, SWE-Pruner performs task-aware adaptive pruning for long contexts. Given the current task, the agent formulates an explicit goal (e.g., "focus on error handling") as a hint to guide the pruning targets. A lightweight neural skimmer (0.6B parameters) is trained to dynamically select relevant lines from the surrounding context given the goal. Evaluations across four benchmarks and multiple models validate SWE-Pruner's effectiveness in various scenarios, achieving 23-54% token reduction on agent tasks like SWE-Bench Verified and up to 14.84x compression on single-turn tasks like LongCodeQA with minimal performance impact.

**arXiv ID:** 2601.16746
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (11 papers)</h2></summary>

<details>
<summary><strong>When Agents Fail to Act: A Diagnostic Framework for Tool Invocation Reliability in Multi-Agent LLM Systems</strong> - Donghao Huang, Gauri Malwe, Zhaoxia Wang - [[pdf]](https://arxiv.org/pdf/2601.16280)</summary>

**Abstract:** Multi-agent systems powered by large language models (LLMs) are transforming enterprise automation, yet systematic evaluation methodologies for assessing tool-use reliability remain underdeveloped. We introduce a comprehensive diagnostic framework that leverages big data analytics to evaluate procedural reliability in intelligent agent systems, addressing critical needs for SME-centric deployment in privacy-sensitive environments. Our approach features a 12-category error taxonomy capturing failure modes across tool initialization, parameter handling, execution, and result interpretation. Through systematic evaluation of 1,980 deterministic test instances spanning both open-weight models (Qwen2.5 series, Functionary) and proprietary alternatives (GPT-4, Claude 3.5/3.7) across diverse edge hardware configurations, we identify actionable reliability thresholds for production deployment. Our analysis reveals that procedural reliability, particularly tool initialization failures, constitutes the primary bottleneck for smaller models, while qwen2.5:32b achieves flawless performance matching GPT-4.1. The framework demonstrates that mid-sized models (qwen2.5:14b) offer practical accuracy-efficiency trade-offs on commodity hardware (96.6\% success rate, 7.3 s latency), enabling cost-effective intelligent agent deployment for resource-constrained organizations. This work establishes foundational infrastructure for systematic reliability evaluation of tool-augmented multi-agent AI systems.

**arXiv ID:** 2601.16280
</details>

<details>
<summary><strong>AgentsEval: Clinically Faithful Evaluation of Medical Imaging Reports via Multi-Agent Reasoning</strong> - Suzhong Fu, Jingqi Dong, Xuan Ding, Rui Sun, Yiming Yang, Shuguang Cui, Zhen Li - [[pdf]](https://arxiv.org/pdf/2601.16685)</summary>

**Abstract:** Evaluating the clinical correctness and reasoning fidelity of automatically generated medical imaging reports remains a critical yet unresolved challenge. Existing evaluation methods often fail to capture the structured diagnostic logic that underlies radiological interpretation, resulting in unreliable judgments and limited clinical relevance. We introduce AgentsEval, a multi-agent stream reasoning framework that emulates the collaborative diagnostic workflow of radiologists. By dividing the evaluation process into interpretable steps including criteria definition, evidence extraction, alignment, and consistency scoring, AgentsEval provides explicit reasoning traces and structured clinical feedback. We also construct a multi-domain perturbation-based benchmark covering five medical report datasets with diverse imaging modalities and controlled semantic variations. Experimental results demonstrate that AgentsEval delivers clinically aligned, semantically faithful, and interpretable evaluations that remain robust under paraphrastic, semantic, and stylistic perturbations. This framework represents a step toward transparent and clinically grounded assessment of medical report generation systems, fostering trustworthy integration of large language models into clinical practice.

**arXiv ID:** 2601.16685
</details>

<details>
<summary><strong>MAGE-KT: Multi-Agent Graph-Enhanced Knowledge Tracing with Subgraph Retrieval and Asymmetric Fusion</strong> - Chi Yu, Hongyu Yuan, Zhiyi Duan - [[pdf]](https://arxiv.org/pdf/2601.16886)</summary>

**Abstract:** Knowledge Tracing (KT) aims to model a student's learning trajectory and predict performance on the next question. A key challenge is how to better represent the relationships among students, questions, and knowledge concepts (KCs). Recently, graph-based KT paradigms have shown promise for this problem. However, existing methods have not sufficiently explored inter-concept relations, often inferred solely from interaction sequences. In addition, the scale and heterogeneity of KT graphs make full-graph encoding both computationally both costly and noise-prone, causing attention to bleed into student-irrelevant regions and degrading the fidelity of inter-KC relations. To address these issues, we propose a novel framework: Multi-Agent Graph-Enhanced Knowledge Tracing (MAGE-KT). It constructs a multi-view heterogeneous graph by combining a multi-agent KC relation extractor and a student-question interaction graph, capturing complementary semantic and behavioral signals. Conditioned on the target student's history, it retrieves compact, high-value subgraphs and integrates them using an Asymmetric Cross-attention Fusion Module to enhance prediction while avoiding attention diffusion and irrelevant computation. Experiments on three widely used KT datasets show substantial improvements in KC-relation accuracy and clear gains in next-question prediction over existing methods.

**arXiv ID:** 2601.16886
</details>

<details>
<summary><strong>EvoConfig: Self-Evolving Multi-Agent Systems for Efficient Autonomous Environment Configuration</strong> - Xinshuai Guo, Jiayi Kuang, Linyue Pan, Yinghui Li, Yangning Li, Hai-Tao Zheng, Ying Shen, Di Yin, Xing Sun - [[pdf]](https://arxiv.org/pdf/2601.16489)</summary>

**Abstract:** A reliable executable environment is the foundation for ensuring that large language models solve software engineering tasks. Due to the complex and tedious construction process, large-scale configuration is relatively inefficient. However, most methods always overlook fine-grained analysis of the actions performed by the agent, making it difficult to handle complex errors and resulting in configuration failures. To address this bottleneck, we propose EvoConfig, an efficient environment configuration framework that optimizes multi-agent collaboration to build correct runtime environments. EvoConfig features an expert diagnosis module for fine-grained post-execution analysis, and a self-evolving mechanism that lets expert agents self-feedback and dynamically adjust error-fixing priorities in real time. Empirically, EvoConfig matches the previous state-of-the-art Repo2Run on Repo2Run's 420 repositories, while delivering clear gains on harder cases: on the more challenging Envbench, EvoConfig achieves a 78.1% success rate, outperforming Repo2Run by 7.1%. Beyond end-to-end success, EvoConfig also demonstrates stronger debugging competence, achieving higher accuracy in error identification and producing more effective repair recommendations than existing methods.

**arXiv ID:** 2601.16489
</details>

<details>
<summary><strong>Towards Open-World Retrieval-Augmented Generation on Knowledge Graph: A Multi-Agent Collaboration Framework</strong> - Jiasheng Xu, Mingda Li, Yongqiang Tang, Peijie Wang, Wensheng Zhang - [[pdf]](https://arxiv.org/pdf/2509.01238)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated strong capabilities in web search and reasoning. However, their dependence on static training corpora makes them prone to factual errors and knowledge gaps. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge sources, especially structured Knowledge Graphs (KGs), which provide explicit semantics and efficient retrieval. Existing KG-based RAG approaches, however, generally assume that anchor entities are accessible to initiate graph traversal, which limits their robustness in open-world settings where accurate linking between the user query and the KG entity is unreliable. To overcome this limitation, we propose AnchorRAG, a novel multi-agent collaboration framework for open-world RAG without the predefined anchor entities. Specifically, a predictor agent dynamically identifies candidate anchor entities by aligning user query terms with KG nodes and initializes independent retriever agents to conduct parallel multi-hop explorations from each candidate. Then a supervisor agent formulates the iterative retrieval strategy for these retriever agents and synthesizes the resulting knowledge paths to generate the final answer. This multi-agent collaboration framework improves retrieval robustness and mitigates the impact of ambiguous or erroneous anchors. Extensive experiments on four public benchmarks demonstrate that AnchorRAG significantly outperforms existing baselines and establishes new state-of-the-art results on the real-world reasoning tasks.

**arXiv ID:** 2509.01238
</details>

<details>
<summary><strong>SimWorld: An Open-ended Realistic Simulator for Autonomous Agents in Physical and Social Worlds</strong> - Jiawei Ren, Yan Zhuang, Xiaokang Ye, Lingjun Mao, Xuhong He, Jianzhi Shen, Mrinaal Dogra, Yiming Liang, Ruixuan Zhang, Tianai Yue, Yiqing Yang, Eric Liu, Ryan Wu, Kevin Benavente, Rajiv Mandya Nagaraju, Muhammad Faayez, Xiyan Zhang, Dhruv Vivek Sharma, Xianrui Zhong, Ziqiao Ma, Tianmin Shu, Zhiting Hu, Lianhui Qin - [[pdf]](https://arxiv.org/pdf/2512.01078)</summary>

**Abstract:** While LLM/VLM-powered AI agents have advanced rapidly in math, coding, and computer use, their applications in complex physical and social environments remain challenging. Building agents that can survive and thrive in the real world (for example, by autonomously earning income or running a business) requires massive-scale interaction, reasoning, training, and evaluation across diverse embodied scenarios. However, existing world simulators for such development fall short: they often rely on limited hand-crafted environments, simulate simplified game-like physics and social rules, and lack native support for LLM/VLM agents. We introduce SimWorld, a new simulator built on Unreal Engine 5, designed for developing and evaluating LLM/VLM agents in rich, real-world-like settings. SimWorld offers three core capabilities: (1) realistic, open-ended world simulation, including accurate physical and social dynamics and language-driven procedural environment generation; (2) a rich interface for LLM/VLM agents, with multimodal world inputs and open-vocabulary actions at varying levels of abstraction; and (3) diverse and extensible physical and social reasoning scenarios that are easily customizable by users. We demonstrate SimWorld by deploying frontier LLM agents (e.g., GPT-4o, Gemini-2.5-Flash, Claude-3.5, and DeepSeek-Prover-V2) on long-horizon multi-agent delivery tasks involving strategic cooperation and competition. The results reveal distinct reasoning patterns and limitations across models. We open-source SimWorld and hope it becomes a foundational platform for advancing real-world agent intelligence across disciplines: this https URL.

**arXiv ID:** 2512.01078
</details>

<details>
<summary><strong>Emergent Coordination in Multi-Agent Systems via Pressure Fields and Temporal Decay</strong> - Roland Rodriguez - [[pdf]](https://arxiv.org/pdf/2601.08129)</summary>

**Abstract:** Current multi-agent LLM frameworks rely on explicit orchestration patterns borrowed from human organizational structures: planners delegate to executors, managers coordinate workers, and hierarchical control flow governs agent interactions. These approaches suffer from coordination overhead that scales poorly with agent count and task complexity. We propose a fundamentally different paradigm inspired by natural coordination mechanisms: agents operate locally on a shared artifact, guided only by pressure gradients derived from measurable quality signals, with temporal decay preventing premature convergence. We formalize this as optimization over a pressure landscape and prove convergence guarantees under mild conditions. Empirically, on meeting room scheduling across 1,350 trials, pressure-field coordination outperforms all baselines: 48.5% aggregate solve rate versus 12.6% for conversation-based coordination, 1.5% for hierarchical control, and 0.4% for sequential and random baselines (all pairwise comparisons p < 0.001). Temporal decay is essential: disabling it reduces solve rate by 10 percentage points. On easy problems, pressure-field achieves 86.7% solve rate. The approach maintains consistent performance from 1 to 4 agents. Implicit coordination through shared pressure gradients outperforms explicit hierarchical control, suggesting that constraint-driven emergence offers a simpler and more effective foundation for multi-agent AI.

**arXiv ID:** 2601.08129
</details>

<details>
<summary><strong>MACTAS: Self-Attention-Based Inter-Agent Communication in Multi-Agent Reinforcement Learning with Action-Value Function Decomposition</strong> - Maciej Wojtala, Bogusz Stefańczyk, Dominik Bogucki, Łukasz Lepak, Jakub Strykowski, Paweł Wawrzyński - [[pdf]](https://arxiv.org/pdf/2508.13661)</summary>

**Abstract:** Communication is essential for the collective execution of complex tasks by human agents, motivating interest in communication mechanisms for multi-agent reinforcement learning (MARL). However, existing communication protocols in MARL are often complex and non-differentiable. In this work, we introduce a self-attention-based communication method that exchanges information between the agents in MARL. Our proposed approach is fully differentiable, allowing agents to learn to generate messages in a reward-driven manner. The method can be seamlessly integrated with any action-value function decomposition algorithm and can be viewed as an orthogonal extension of such decompositions. Notably, it includes a fixed number of trainable parameters, independent of the number of agents, which makes it scalable to large systems. Experimental results on the SMACv2 benchmark demonstrate the effectiveness of our approach, which achieves state-of-the-art performance on a number of maps. makes it scalable to large systems. Experimental results on the SMACv2 benchmark demonstrate the effectiveness of our approach, which achieves state-of-the-art performance on a number of maps.

**arXiv ID:** 2508.13661
</details>

<details>
<summary><strong>Endless Terminals: Scaling RL Environments for Terminal Agents</strong> - Kanishk Gandhi, Shivam Garg, Noah D. Goodman, Dimitris Papailiopoulos - [[pdf]](https://arxiv.org/pdf/2601.16443)</summary>

**Abstract:** Environments are the bottleneck for self-improving agents. Current terminal benchmarks were built for evaluation, not training; reinforcement learning requires a scalable pipeline, not just a dataset. We introduce Endless Terminals, a fully autonomous pipeline that procedurally generates terminal-use tasks without human annotation. The pipeline has four stages: generating diverse task descriptions, building and validating containerized environments, producing completion tests, and filtering for solvability. From this pipeline we obtain 3255 tasks spanning file operations, log management, data processing, scripting, and database operations. We train agents using vanilla PPO with binary episode level rewards and a minimal interaction loop: no retrieval, multi-agent coordination, or specialized tools. Despite this simplicity, models trained on Endless Terminals show substantial gains: on our held-out dev set, Llama-3.2-3B improves from 4.0% to 18.2%, Qwen2.5-7B from 10.7% to 53.3%, and Qwen3-8B-openthinker-sft from 42.6% to 59.0%. These improvements transfer to human-curated benchmarks: models trained on Endless Terminals show substantial gains on held out human curated benchmarks: on TerminalBench 2.0, Llama-3.2-3B improves from 0.0% to 2.2%, Qwen2.5-7B from 2.2% to 3.4%, and Qwen3-8B-openthinker-sft from 1.1% to 6.7%, in each case outperforming alternative approaches including models with more complex agentic scaffolds. These results demonstrate that simple RL succeeds when environments scale.

**arXiv ID:** 2601.16443
</details>

<details>
<summary><strong>The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check</strong> - Qingyu Lu, Liang Ding, Kanjian Zhang, Jinxia Zhang, Dacheng Tao - [[pdf]](https://arxiv.org/pdf/2601.12979)</summary>

**Abstract:** The pursuit of real-time agentic interaction has driven interest in Diffusion-based Large Language Models (dLLMs) as alternatives to auto-regressive backbones, promising to break the sequential latency bottleneck. However, does such efficiency gains translate into effective agentic behavior? In this work, we present a comprehensive evaluation of dLLMs (e.g., LLaDA, Dream) across two distinct agentic paradigms: Embodied Agents (requiring long-horizon planning) and Tool-Calling Agents (requiring precise formatting). Contrary to the efficiency hype, our results on Agentboard and BFCL reveal a "bitter lesson": current dLLMs fail to serve as reliable agentic backbones, frequently leading to systematically failure. (1) In Embodied settings, dLLMs suffer repeated attempts, failing to branch under temporal feedback. (2) In Tool-Calling settings, dLLMs fail to maintain symbolic precision (e.g. strict JSON schemas) under diffusion noise. To assess the potential of dLLMs in agentic workflows, we introduce DiffuAgent, a multi-agent evaluation framework that integrates dLLMs as plug-and-play cognitive cores. Our analysis shows that dLLMs are effective in non-causal roles (e.g., memory summarization and tool selection) but require the incorporation of causal, precise, and logically grounded reasoning mechanisms into the denoising process to be viable for agentic tasks.

**arXiv ID:** 2601.12979
</details>

<details>
<summary><strong>VALISENS: A Validated Innovative Multi-Sensor System for Cooperative Automated Driving</strong> - Lei Wan, Prabesh Gupta, Andreas Eich, Marcel Kettelgerdes, Hannan Ejaz Keen, Michael Klöppel-Gersdorf, Alexey Vinel - [[pdf]](https://arxiv.org/pdf/2505.06980)</summary>

**Abstract:** Reliable perception remains a key challenge for Connected Automated Vehicles (CAVs) in complex real-world environments, where varying lighting conditions and adverse weather degrade sensing performance. While existing multi-sensor solutions improve local robustness, they remain constrained by limited sensing range, line-of-sight occlusions, and sensor failures on individual vehicles. This paper introduces VALISENS, a validated cooperative perception system that extends multi-sensor fusion beyond a single vehicle through Vehicle-to-Everything (V2X)-enabled collaboration between Connected Automated Vehicles (CAVs) and intelligent infrastructure. VALISENS integrates onboard and roadside LiDARs, radars, RGB cameras, and thermal cameras within a unified multi-agent perception framework. Thermal cameras enhances the detection of Vulnerable Road Users (VRUs) under challenging lighting conditions, while roadside sensors reduce occlusions and expand the effective perception range. In addition, an integrated sensor monitoring module continuously assesses sensor health and detects anomalies before system degradation occurs. The proposed system is implemented and evaluated in a dedicated real-world testbed. Experimental results show that VALISENS improves pedestrian situational awareness by up to 18% compared with vehicle-only sensing, while the sensor monitoring module achieves over 97% accuracy, demonstrating its effectiveness and its potential to support future Cooperative Intelligent Transport Systems (C-ITS) applications.

**arXiv ID:** 2505.06980
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (22 papers)</h2></summary>

<details>
<summary><strong>Mixture-of-Models: Unifying Heterogeneous Agents via N-Way Self-Evaluating Deliberation</strong> - Tims Pecerskis, Aivars Smirnovs - [[pdf]](https://arxiv.org/pdf/2601.16863)</summary>

**Abstract:** This paper introduces the N-Way Self-Evaluating Deliberation (NSED) protocol, a Runtime Mixture-of-Models (MoM) architecture that constructs emergent composite models from a plurality of distinct expert agents. Unlike traditional Mixture-of-Experts (MoE) which rely on static gating networks, NSED employs a Dynamic Expertise Broker - a runtime optimization engine that treats model selection as a variation of the Knapsack Problem, binding heterogeneous checkpoints to functional roles based on live telemetry and cost constraints. At the execution layer, we formalize deliberation as a Macro-Scale Recurrent Neural Network (RNN), where the consensus state loops back through a semantic forget gate to enable iterative refinement without proportional VRAM scaling. Key components include an orchestration fabric for trustless N-to-N peer review, a Quadratic Voting activation function for non-linear consensus, and a feedback-driven state update. Empirical validation on challenging benchmarks (AIME 2025, LiveCodeBench) demonstrates that this topology allows ensembles of small (less than 20B) consumer-grade models to match or exceed the performance of state-of-the-art 100B+ parameter models, establishing a new hardware arbitrage efficiency frontier. Furthermore, testing on the DarkBench safety suite reveals intrinsic alignment properties, with peer-mediated correction reducing sycophancy scores below that of any individual agent.

**arXiv ID:** 2601.16863
</details>

<details>
<summary><strong>DMV-AVP: Distributed Multi-Vehicle Autonomous Valet Parking using Autoware</strong> - Zubair Islam, Mohamed El-Darieby - [[pdf]](https://arxiv.org/pdf/2601.16327)</summary>

**Abstract:** This paper presents the DMV-AVP System, a distributed simulation of Multi-Vehicle Autonomous Valet Parking (AVP). The system was implemented as an application of the Distributed Multi-Vehicle Architecture (DMAVA) for synchronized multi-host execution. Most existing simulation approaches rely on centralized or non-distributed designs that constrain scalability and limit fully autonomous control. This work introduces two modules built on top of the DMAVA: 1) a Multi-Vehicle AVP Node that performs state-based coordination, queuing, and reservation management across multiple vehicles, and 2) a Unity-Integrated YOLOv5 Parking Spot Detection Module that provides real-time, vision-based perception within AWSIM Labs. Both modules integrate seamlessly with the DMAVA and extend it specifically for multi-vehicle AVP operation, supported by a Zenoh-based communication layer that ensures low-latency topic synchronization and coordinated behavior across hosts. Experiments conducted on two- and three-host configurations demonstrate deterministic coordination, conflict-free parking behavior, and scalable performance across distributed Autoware instances. The results confirm that the proposed Distributed Multi-Vehicle AVP System supports cooperative AVP simulation and establishes a foundation for future real-world and hardware-in-the-loop validation. Demo videos and source code are available at this https URL

**arXiv ID:** 2601.16327
</details>

<details>
<summary><strong>VibeTensor: System Software for Deep Learning, Fully Generated by AI Agents</strong> - Bing Xu, Terry Chen, Fengzhe Zhou, Tianqi Chen, Yangqing Jia, Vinod Grover, Haicheng Wu, Wei Liu, Craig Wittenbrink, Wen-mei Hwu, Roger Bringmann, Ming-Yu Liu, Luis Ceze, Michael Lightstone, Humphrey Shi - [[pdf]](https://arxiv.org/pdf/2601.16238)</summary>

**Abstract:** VIBETENSOR is an open-source research system software stack for deep learning, generated by LLM-powered coding agents under high-level human guidance. In this paper, "fully generated" refers to code provenance: implementation changes were produced and applied as agent-proposed diffs; validation relied on agent-run builds, tests, and differential checks, without per-change manual diff review. It implements a PyTorch-style eager tensor library with a C++20 core (CPU+CUDA), a torch-like Python overlay via nanobind, and an experimental this http URL interface. Unlike thin bindings, VIBETENSOR includes its own tensor/storage system, schema-lite dispatcher, reverse-mode autograd, CUDA runtime (streams/events/graphs), a stream-ordered caching allocator with diagnostics, and a stable C ABI for dynamically loaded operator plugins. We view this release as a milestone for AI-assisted software engineering: it shows coding agents can generate a coherent deep learning runtime spanning language bindings down to CUDA memory management, validated primarily by builds and tests. We describe the architecture, summarize the workflow used to produce and validate the system, and evaluate the artifact. We report repository scale and test-suite composition, and summarize reproducible microbenchmarks from an accompanying AI-generated kernel suite, including fused attention versus PyTorch SDPA/FlashAttention. We also report end-to-end training sanity checks on 3 small workloads (sequence reversal, ViT, miniGPT) on NVIDIA H100 (Hopper, SM90) and Blackwell-class GPUs; multi-GPU results are Blackwell-only and use an optional CUTLASS-based ring-allreduce plugin gated on CUDA 13+ and sm103a toolchain support. Finally, we discuss failure modes in generated system software, including a "Frankenstein" composition effect where locally correct subsystems interact to yield globally suboptimal performance.

**arXiv ID:** 2601.16238
</details>

<details>
<summary><strong>DMAVA: Distributed Multi-Autonomous Vehicle Architecture Using Autoware</strong> - Zubair Islam, Mohamed El-Darieby - [[pdf]](https://arxiv.org/pdf/2601.16336)</summary>

**Abstract:** Simulating and validating coordination among multiple autonomous vehicles (AVs) is a challenging task as most existing simulation architectures are limited to single-vehicle operation or rely on centralized control. This paper presents a Distributed Multi-AV Architecture (DMAVA) that enables synchronized, real-time autonomous driving simulation across multiple physical hosts. Each vehicle runs its own complete AV stack and operates independently from other AVs. The vehicles in the simulation maintain synchronized coordination through a low-latency data-centric communication layer. The proposed system integrates ROS 2 Humble, Autoware Universe, AWSIM Labs, and Zenoh to support concurrent execution of multiple Autoware stacks within a shared Unity-based environment. Experiments conducted on multiple-host configurations demonstrate stable localization, reliable inter-host communication, and fully synchronized closed-loop control. The DMAVA also serves as a foundation for Multi-Vehicle Autonomous Valet Parking, demonstrating its extensibility toward higher-level cooperative autonomy. Demo videos and source code are available at: this https URL.

**arXiv ID:** 2601.16336
</details>

<details>
<summary><strong>ResAgent: Entropy-based Prior Point Discovery and Visual Reasoning for Referring Expression Segmentation</strong> - Yihao Wang, Jusheng Zhang, Ziyi Tang, Keze Wang, Meng Yang - [[pdf]](https://arxiv.org/pdf/2601.16394)</summary>

**Abstract:** Referring Expression Segmentation (RES) is a core vision-language segmentation task that enables pixel-level understanding of targets via free-form linguistic expressions, supporting critical applications such as human-robot interaction and augmented reality. Despite the progress of Multimodal Large Language Model (MLLM)-based approaches, existing RES methods still suffer from two key limitations: first, the coarse bounding boxes from MLLMs lead to redundant or non-discriminative point prompts; second, the prevalent reliance on textual coordinate reasoning is unreliable, as it fails to distinguish targets from visually similar distractors. To address these issues, we propose \textbf{\model}, a novel RES framework integrating \textbf{E}ntropy-\textbf{B}ased Point \textbf{D}iscovery (\textbf{EBD}) and \textbf{V}ision-\textbf{B}ased \textbf{R}easoning (\textbf{VBR}). Specifically, EBD identifies high-information candidate points by modeling spatial uncertainty within coarse bounding boxes, treating point selection as an information maximization process. VBR verifies point correctness through joint visual-semantic alignment, abandoning text-only coordinate inference for more robust validation. Built on these components, \model implements a coarse-to-fine workflow: bounding box initialization, entropy-guided point discovery, vision-based validation, and mask decoding. Extensive evaluations on four benchmark datasets (RefCOCO, RefCOCO+, RefCOCOg, and ReasonSeg) demonstrate that \model achieves new state-of-the-art performance across all four benchmarks, highlighting its effectiveness in generating accurate and semantically grounded segmentation masks with minimal prompts.

**arXiv ID:** 2601.16394
</details>

<details>
<summary><strong>DeepEra: A Deep Evidence Reranking Agent for Scientific Retrieval-Augmented Generated Question Answering</strong> - Haotian Chen, Qingqing Long, Siyu Pu, Xiao Luo, Wei Ju, Meng Xiao, Yuanchun Zhou, Jianghua Zhao, Xuezhi Wang - [[pdf]](https://arxiv.org/pdf/2601.16478)</summary>

**Abstract:** With the rapid growth of scientific literature, scientific question answering (SciQA) has become increasingly critical for exploring and utilizing scientific knowledge. Retrieval-Augmented Generation (RAG) enhances LLMs by incorporating knowledge from external sources, thereby providing credible evidence for scientific question answering. But existing retrieval and reranking methods remain vulnerable to passages that are semantically similar but logically irrelevant, often reducing factual reliability and amplifying this http URL address this challenge, we propose a Deep Evidence Reranking Agent (DeepEra) that integrates step-by-step reasoning, enabling more precise evaluation of candidate passages beyond surface-level semantics. To support systematic evaluation, we construct SciRAG-SSLI (Scientific RAG - Semantically Similar but Logically Irrelevant), a large-scale dataset comprising about 300K SciQA instances across 10 subjects, constructed from 10M scientific corpus. The dataset combines naturally retrieved contexts with systematically generated distractors to test logical robustness and factual grounding. Comprehensive evaluations confirm that our approach achieves superior retrieval performance compared to leading rerankers. To our knowledge, this work is the first to comprehensively study and empirically validate innegligible SSLI issues in two-stage RAG frameworks.

**arXiv ID:** 2601.16478
</details>

<details>
<summary><strong>Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic</strong> - Yichuan Ma, Linyang Li, Yongkang chen, Peiji Li, Xiaozhe Li, Qipeng Guo, Dahua Lin, Kai Chen - [[pdf]](https://arxiv.org/pdf/2601.16486)</summary>

**Abstract:** As large language models (LLMs) increasingly tackle complex reasoning tasks, test-time scaling has become critical for enhancing capabilities. However, in agentic scenarios with frequent tool calls, the traditional generation-length-based definition breaks down: tool latency decouples inference time from generation length. We propose Timely Machine, redefining test-time as wall-clock time, where models dynamically adjust strategies based on time budgets. We introduce Timely-Eval, a benchmark spanning high-frequency tool calls, low-frequency tool calls, and time-constrained reasoning. By varying tool latency, we find smaller models excel with fast feedback through more interactions, while larger models dominate high-latency settings via superior interaction quality. Moreover, existing models fail to adapt reasoning to time budgets. We propose Timely-RL to address this gap. After cold-start supervised fine-tuning, we use reinforcement learning to enhance temporal planning. Timely-RL improves time budget awareness and consistently boosts performance across Timely-Eval. We hope our work offers a new perspective on test-time scaling for the agentic era.

**arXiv ID:** 2601.16486
</details>

<details>
<summary><strong>GTA: Generative Traffic Agents for Simulating Realistic Mobility Behavior</strong> - Simon Lämmer, Mark Colley, Patrick Ebel - [[pdf]](https://arxiv.org/pdf/2601.16778)</summary>

**Abstract:** People's transportation choices reflect complex trade-offs shaped by personal preferences, social norms, and technology acceptance. Predicting such behavior at scale is a critical challenge with major implications for urban planning and sustainable transport. Traditional methods use handcrafted assumptions and costly data collection, making them impractical for early-stage evaluations of new technologies or policies. We introduce Generative Traffic Agents (GTA) for simulating large-scale, context-sensitive transportation choices using LLM-powered, persona-based agents. GTA generates artificial populations from census-based sociodemographic data. It simulates activity schedules and mode choices, enabling scalable, human-like simulations without handcrafted rules. We evaluate GTA in Berlin-scale experiments, comparing simulation results against empirical data. While agents replicate patterns, such as modal split by socioeconomic status, they show systematic biases in trip length and mode preference. GTA offers new opportunities for modeling how future innovations, from bike lanes to transit apps, shape mobility decisions.

**arXiv ID:** 2601.16778
</details>

<details>
<summary><strong>Boosting Deep Reinforcement Learning with Semantic Knowledge for Robotic Manipulators</strong> - Lucía Güitta-López, Vincenzo Suriani, Jaime Boal, Álvaro J. López-López, Daniele Nardi - [[pdf]](https://arxiv.org/pdf/2601.16866)</summary>

**Abstract:** Deep Reinforcement Learning (DRL) is a powerful framework for solving complex sequential decision-making problems, particularly in robotic control. However, its practical deployment is often hindered by the substantial amount of experience required for learning, which results in high computational and time costs. In this work, we propose a novel integration of DRL with semantic knowledge in the form of Knowledge Graph Embeddings (KGEs), aiming to enhance learning efficiency by providing contextual information to the agent. Our architecture combines KGEs with visual observations, enabling the agent to exploit environmental knowledge during training. Experimental validation with robotic manipulators in environments featuring both fixed and randomized target attributes demonstrates that our method achieves up to {60}{\%} reduction in learning time and improves task accuracy by approximately 15 percentage points, without increasing training time or computational complexity. These results highlight the potential of semantic knowledge to reduce sample complexity and improve the effectiveness of DRL in robotic applications.

**arXiv ID:** 2601.16866
</details>

<details>
<summary><strong>Enhancing Study-Level Inference from Clinical Trial Papers via Reinforcement Learning-Based Numeric Reasoning</strong> - Massimiliano Pronesti, Michela Lorandi, Paul Flanagan, Oisin Redmond, Anya Belz, Yufang Hou - [[pdf]](https://arxiv.org/pdf/2505.22928)</summary>

**Abstract:** Systematic reviews in medicine play a critical role in evidence-based decision-making by aggregating findings from multiple studies. A central bottleneck in automating this process is extracting numeric evidence and determining study-level conclusions for specific outcomes and comparisons. Prior work has framed this problem as a textual inference task by retrieving relevant content fragments and inferring conclusions from them. However, such approaches often rely on shallow textual cues and fail to capture the underlying numeric reasoning behind expert assessments.
In this work, we conceptualise the problem as one of quantitative reasoning. Rather than inferring conclusions from surface text, we extract structured numerical evidence (e.g., event counts or standard deviations) and apply domain knowledge informed logic to derive outcome-specific conclusions. We develop a numeric reasoning system composed of a numeric data extraction model and an effect estimate component, enabling more accurate and interpretable inference aligned with the domain expert principles. We train the numeric data extraction model using different strategies, including supervised fine-tuning (SFT) and reinforcement learning (RL) with a new value reward model.
When evaluated on the CochraneForest benchmark, our best-performing approach -- using RL to train a small-scale number extraction model -- yields up to a 21% absolute improvement in F1 score over retrieval-based systems and outperforms general-purpose LLMs of over 400B parameters by up to 9% on the RCTs benchmark. Our results demonstrate the promise of reasoning-driven approaches for automating systematic evidence synthesis.

**arXiv ID:** 2505.22928
</details>

<details>
<summary><strong>Proof-of-Use: Mitigating Tool-Call Hacking in Deep Research Agents</strong> - SHengjie Ma, Chenlong Deng, Jiaxin Mao, Jiadeng Huang, Teng Wang, Junjie Wu, Changwang Zhang, Jun wang - [[pdf]](https://arxiv.org/pdf/2510.10931)</summary>

**Abstract:** While reinforcement learning (RL) enhances their ability to plan and reason across retrieval steps, we identify a critical failure mode in this setting: Tool-Call Hacking. Unlike execution-based tools (e.g., code or math), whose effects are directly observable, the weak observability of causal dependencies between retrieved evidence and reasoning under format- and outcome-level supervision enables agents to maximize surface-level reward signals without genuinely grounding their reasoning in the returned evidence. This leads to distinctive pathologies, including mode collapse via tool overuse and hallucinated tool usage where tool calls are largely decorative.
To address this issue, we propose Proof-of-Use (PoU), an evidence grounded RL framework that explicitly optimizes the causal dependency from retrieval to reasoning and final answers. PoU re-fomulate a fine-grained stepwise interaction protocol in which agents must auditably cite normalized evidence identifiers. We operationalize this via a multi-objective reward design consisting of: (1) two progressive process rewards that constrain citation validity at intermediate steps; (2) a global Answer--Support Alignment reward that enforces consistency between final answers and retrieved evidence; and (3) a curriculum-style adaptive reward mixing mechanism that smoothly transitions agents from dense process supervision to sparse outcome-based objectives. Extensive experiments show the strong performance of PoU and demonstrate the effectiveness in mitigating tool-call hacking. Beyond this, PoU exhibits a notable emergent property: adaptive and robust tool-usage patterns naturally arise under domain and tool shifts, even though PoU does not explicitly optimize for tool adaptation.

**arXiv ID:** 2510.10931
</details>

<details>
<summary><strong>EvoCUA: Evolving Computer Use Agents via Learning from Scalable Synthetic Experience</strong> - Taofeng Xue, Chong Peng, Mianqiu Huang, Linsen Guo, Tiancheng Han, Haozhe Wang, Jianing Wang, Xiaocheng Zhang, Xin Yang, Dengchang Zhao, Jinrui Ding, Xiandi Ma, Yuchen Xie, Peng Pei, Xunliang Cai, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2601.15876)</summary>

**Abstract:** The development of native computer-use agents (CUA) represents a significant leap in multimodal AI. However, their potential is currently bottlenecked by the constraints of static data scaling. Existing paradigms relying primarily on passive imitation of static datasets struggle to capture the intricate causal dynamics inherent in long-horizon computer tasks. In this work, we introduce EvoCUA, a native computer use agentic model. Unlike static imitation, EvoCUA integrates data generation and policy optimization into a self-sustaining evolutionary cycle. To mitigate data scarcity, we develop a verifiable synthesis engine that autonomously generates diverse tasks coupled with executable validators. To enable large-scale experience acquisition, we design a scalable infrastructure orchestrating tens of thousands of asynchronous sandbox rollouts. Building on these massive trajectories, we propose an iterative evolving learning strategy to efficiently internalize this experience. This mechanism dynamically regulates policy updates by identifying capability boundaries -- reinforcing successful routines while transforming failure trajectories into rich supervision through error analysis and self-correction. Empirical evaluations on the OSWorld benchmark demonstrate that EvoCUA achieves a success rate of 56.7%, establishing a new open-source state-of-the-art. Notably, EvoCUA significantly outperforms the previous best open-source model, OpenCUA-72B (45.0%), and surpasses leading closed-weights models such as UI-TARS-2 (53.1%). Crucially, our results underscore the generalizability of this approach: the evolving paradigm driven by learning from experience yields consistent performance gains across foundation models of varying scales, establishing a robust and scalable path for advancing native agent capabilities.

**arXiv ID:** 2601.15876
</details>

<details>
<summary><strong>Towards Fast Safe Online Reinforcement Learning via Policy Finetuning</strong> - Keru Chen, Honghao Wei, Zhigang Deng, Sen Lin - [[pdf]](https://arxiv.org/pdf/2412.04426)</summary>

**Abstract:** The high costs and risks involved in extensive environment interactions hinder the practical application of current online safe reinforcement learning (RL) methods. While offline safe RL addresses this by learning policies from static datasets, the performance therein is usually limited due to reliance on data quality and challenges with out-of-distribution (OOD) actions. Inspired by recent successes in offline-to-online (O2O) RL, it is crucial to explore whether offline safe RL can be leveraged to facilitate faster and safer online policy learning, a direction that has yet to be fully investigated. To fill this gap, we first demonstrate that naively applying existing O2O algorithms from standard RL would not work well in the safe RL setting due to two unique challenges: \emph{erroneous Q-estimations}, resulted from offline-online objective mismatch and offline cost sparsity, and \emph{Lagrangian mismatch}, resulted from difficulties in aligning Lagrange multipliers between offline and online policies. To address these challenges, we introduce \textbf{Marvel}, a novel framework for O2O safe RL, comprising two key components that work in concert: \emph{Value Pre-Alignment} to align the Q-functions with the underlying truth before online learning, and \emph{Adaptive PID Control} to effectively adjust the Lagrange multipliers during online finetuning. Extensive experiments demonstrate that Marvel significantly outperforms existing baselines in both reward maximization and safety constraint satisfaction. By introducing the first policy-finetuning based framework for O2O safe RL, which is compatible with many offline and online safe RL methods, our work has the great potential to advance the field towards more efficient and practical safe RL solutions.

**arXiv ID:** 2412.04426
</details>

<details>
<summary><strong>Reinforcement Learning for Charging Optimization of Inhomogeneous Dicke Quantum Batteries</strong> - Xiaobin Song, Siyuan Bai, Da-Wei Wang, Hanxiao Tao, Xizhe Wang, Rebing Wu, Benben Jiang - [[pdf]](https://arxiv.org/pdf/2511.12176)</summary>

**Abstract:** Charging optimization is a key challenge to the implementation of quantum batteries, particularly under inhomogeneity and partial observability. This paper employs reinforcement learning to optimize piecewise-constant charging policies for an inhomogeneous Dicke battery. We systematically compare policies across four observability regimes, from full-state access to experimentally accessible observables (energies of individual two-level systems (TLSs), first-order averages, and second-order correlations). Simulation results demonstrate that full observability yields near-optimal ergotropy with low variability, while under partial observability, access to only single-TLS energies or energies plus first-order averages lags behind the fully observed baseline. However, augmenting partial observations with second-order correlations recovers most of the gap, reaching 94%-98% of the full-state baseline. The learned schedules are nonmyopic, trading temporary plateaus or declines for superior terminal outcomes. These findings highlight a practical route to effective fast-charging protocols under realistic information constraints.

**arXiv ID:** 2511.12176
</details>

<details>
<summary><strong>UACER: An Uncertainty-Adaptive Critic Ensemble Framework for Robust Adversarial Reinforcement Learning</strong> - Jiaxi Wu, Tiantian Zhang, Yuxing Wang, Yongzhe Chang, Xueqian Wang - [[pdf]](https://arxiv.org/pdf/2512.10492)</summary>

**Abstract:** Robust adversarial reinforcement learning has emerged as an effective paradigm for training agents to handle uncertain disturbance in real environments, with critical applications in sequential decision-making domains such as autonomous driving and robotic control. Within this paradigm, agent training is typically formulated as a zero-sum Markov game between a protagonist and an adversary to enhance policy robustness. However, the trainable nature of the adversary inevitably induces non-stationarity in the learning dynamics, leading to exacerbated training instability and convergence difficulties, particularly in high-dimensional complex environments. In this paper, we propose a novel approach, Uncertainty-Adaptive Critic Ensemble for robust adversarial Reinforcement learning (UACER), which consists of two components: 1) Diversified critic ensemble: A diverse set of K critic networks is employed in parallel to stabilize Q-value estimation in robust adversarial reinforcement learning, reducing variance and enhancing robustness compared to conventional single-critic designs. 2) Time-varying Decay Uncertainty (TDU) mechanism: Moving beyond simple linear combinations, we propose a variance-derived Q-value aggregation strategy that explicitly incorporates epistemic uncertainty to adaptively regulate the exploration-exploitation trade-off while stabilizing the training process. Comprehensive experiments across several challenging MuJoCo control problems validate the superior effectiveness of UACER, outperforming state-of-the-art methods in terms of overall performance, stability, and efficiency.

**arXiv ID:** 2512.10492
</details>

<details>
<summary><strong>IBISAgent: Reinforcing Pixel-Level Visual Reasoning in MLLMs for Universal Biomedical Object Referring and Segmentation</strong> - Yankai Jiang, Qiaoru Li, Binlu Xu, Haoran Sun, Chao Ding, Junting Dong, Yuxiang Cai, Xuhong Zhang, Jianwei Yin - [[pdf]](https://arxiv.org/pdf/2601.03054)</summary>

**Abstract:** Recent research on medical MLLMs has gradually shifted its focus from image-level understanding to fine-grained, pixel-level comprehension. Although segmentation serves as the foundation for pixel-level understanding, existing approaches face two major challenges. First, they introduce implicit segmentation tokens and require simultaneous fine-tuning of both the MLLM and external pixel decoders, which increases the risk of catastrophic forgetting and limits generalization to out-of-domain scenarios. Second, most methods rely on single-pass reasoning and lack the capability to iteratively refine segmentation results, leading to suboptimal performance. To overcome these limitations, we propose a novel agentic MLLM, named IBISAgent, that reformulates segmentation as a vision-centric, multi-step decision-making process. IBISAgent enables MLLMs to generate interleaved reasoning and text-based click actions, invoke segmentation tools, and produce high-quality masks without architectural modifications. By iteratively performing multi-step visual reasoning on masked image features, IBISAgent naturally supports mask refinement and promotes the development of pixel-level visual reasoning capabilities. We further design a two-stage training framework consisting of cold-start supervised fine-tuning and agentic reinforcement learning with tailored, fine-grained rewards, enhancing the model's robustness in complex medical referring and reasoning segmentation tasks. Extensive experiments demonstrate that IBISAgent consistently outperforms both closed-source and open-source SOTA methods. All datasets, code, and trained models will be released publicly.

**arXiv ID:** 2601.03054
</details>

<details>
<summary><strong>Clarify or Answer: Reinforcement Learning for Agentic VQA with Context Under-specification</strong> - Zongwan Cao, Bingbing Wen, Lucy Lu Wang - [[pdf]](https://arxiv.org/pdf/2601.16400)</summary>

**Abstract:** Real-world visual question answering (VQA) is often context-dependent: an image-question pair may be under-specified, such that the correct answer depends on external information that is not observable in the image. In such cases, directly answering can lead to confident but incorrect predictions. We propose CoA(Clarify-or-Answer), an ask-or-answer agent that separately models the decision to ask or answer, and what to ask if needed. CoA first determines whether clarification is necessary; if so, it asks a single focused question and then incorporates the response to produce the final answer. We introduce CONTEXTCLARIFY with a set of ambiguous VQA questions and the contrast set that is non-ambiguous. We further introduce GRPO-CR (Clarification Reasoning), a reinforcement learning approach that optimizes clarification question generation with multiple reward signals encouraging well-formed, focused, non-trivial questions that resolve ambiguity. Across three VLLMs and three datasets, CoA achieves consistent improvements at both the module and system levels, improving end-to-end VQA accuracy by an average of +15.3 points (83%) over prompting-based baselines

**arXiv ID:** 2601.16400
</details>

<details>
<summary><strong>A Regularized Actor-Critic Algorithm for Bi-Level Reinforcement Learning</strong> - Sihan Zeng, Sujay Bhatt, Sumitra Ganesh, Alec Koppel - [[pdf]](https://arxiv.org/pdf/2601.16399)</summary>

**Abstract:** We study a structured bi-level optimization problem where the upper-level objective is a smooth function and the lower-level problem is policy optimization in a Markov decision process (MDP). The upper-level decision variable parameterizes the reward of the lower-level MDP, and the upper-level objective depends on the optimal induced policy. Existing methods for bi-level optimization and RL often require second-order information, impose strong regularization at the lower level, or inefficiently use samples through nested-loop procedures. In this work, we propose a single-loop, first-order actor-critic algorithm that optimizes the bi-level objective via a penalty-based reformulation. We introduce into the lower-level RL objective an attenuating entropy regularization, which enables asymptotically unbiased upper-level hyper-gradient estimation without solving the unregularized RL problem exactly. We establish the finite-time and finite-sample convergence of the proposed algorithm to a stationary point of the original, unregularized bi-level optimization problem through a novel lower-level residual analysis under a special type of Polyak-Lojasiewicz condition. We validate the performance of our method through experiments on a GridWorld goal position problem and on happy tweet generation through reinforcement learning from human feedback (RLHF).

**arXiv ID:** 2601.16399
</details>

<details>
<summary><strong>Reinforcement Learning-Based Energy-Aware Coverage Path Planning for Precision Agriculture</strong> - Beining Wu, Zihao Ding, Leo Ostigaard, Jun Huang - [[pdf]](https://arxiv.org/pdf/2601.16405)</summary>

**Abstract:** Coverage Path Planning (CPP) is a fundamental capability for agricultural robots; however, existing solutions often overlook energy constraints, resulting in incomplete operations in large-scale or resource-limited environments. This paper proposes an energy-aware CPP framework grounded in Soft Actor-Critic (SAC) reinforcement learning, designed for grid-based environments with obstacles and charging stations. To enable robust and adaptive decision-making under energy limitations, the framework integrates Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal dynamics. A dedicated reward function is designed to jointly optimize coverage efficiency, energy consumption, and return-to-base constraints. Experimental results demonstrate that the proposed approach consistently achieves over 90% coverage while ensuring energy safety, outperforming traditional heuristic algorithms such as Rapidly-exploring Random Tree (RRT), Particle Swarm Optimization (PSO), and Ant Colony Optimization (ACO) baselines by 13.4-19.5% in coverage and reducing constraint violations by 59.9-88.3%. These findings validate the proposed SAC-based framework as an effective and scalable solution for energy-constrained CPP in agricultural robotics.

**arXiv ID:** 2601.16405
</details>

<details>
<summary><strong>Task Aware Dreamer for Task Generalization in Reinforcement Learning</strong> - Chengyang Ying, Xinning Zhou, Zhongkai Hao, Hang Su, Songming Liu, Dong Yan, Jun Zhu - [[pdf]](https://arxiv.org/pdf/2303.05092)</summary>

**Abstract:** A long-standing goal of reinforcement learning is to acquire agents that can learn on training tasks and generalize well on unseen tasks that may share a similar dynamic but with different reward functions. The ability to generalize across tasks is important as it determines an agent's adaptability to real-world scenarios where reward mechanisms might vary. In this work, we first show that training a general world model can utilize similar structures in these tasks and help train more generalizable agents. Extending world models into the task generalization setting, we introduce a novel method named Task Aware Dreamer (TAD), which integrates reward-informed features to identify consistent latent characteristics across tasks. Within TAD, we compute the variational lower bound of sample data log-likelihood, which introduces a new term designed to differentiate tasks using their states, as the optimization objective of our reward-informed world models. To demonstrate the advantages of the reward-informed policy in TAD, we introduce a new metric called Task Distribution Relevance (TDR) which quantitatively measures the relevance of different tasks. For tasks exhibiting a high TDR, i.e., the tasks differ significantly, we illustrate that Markovian policies struggle to distinguish them, thus it is necessary to utilize reward-informed policies in TAD. Extensive experiments in both image-based and state-based tasks show that TAD can significantly improve the performance of handling different tasks simultaneously, especially for those with high TDR, and display a strong generalization ability to unseen tasks.

**arXiv ID:** 2303.05092
</details>

<details>
<summary><strong>Adaptive Reinforcement and Model Predictive Control Switching for Safe Human-Robot Cooperative Navigation</strong> - Ning Liu, Sen Shen, Zheng Li, Matthew D'Souza, Jen Jen Chung, Thomas Braunl - [[pdf]](https://arxiv.org/pdf/2601.16686)</summary>

**Abstract:** This paper addresses the challenge of human-guided navigation for mobile collaborative robots under simultaneous proximity regulation and safety constraints. We introduce Adaptive Reinforcement and Model Predictive Control Switching (ARMS), a hybrid learning-control framework that integrates a reinforcement learning follower trained with Proximal Policy Optimization (PPO) and an analytical one-step Model Predictive Control (MPC) formulated as a quadratic program safety filter. To enable robust perception under partial observability and non-stationary human motion, ARMS employs a decoupled sensing architecture with a Long Short-Term Memory (LSTM) temporal encoder for the human-robot relative state and a spatial encoder for 360-degree LiDAR scans. The core contribution is a learned adaptive neural switcher that performs context-aware soft action fusion between the two controllers, favoring conservative, constraint-aware QP-based control in low-risk regions while progressively shifting control authority to the learned follower in highly cluttered or constrained scenarios where maneuverability is critical, and reverting to the follower action when the QP becomes infeasible. Extensive evaluations against Pure Pursuit, Dynamic Window Approach (DWA), and an RL-only baseline demonstrate that ARMS achieves an 82.5 percent success rate in highly cluttered environments, outperforming DWA and RL-only approaches by 7.1 percent and 3.1 percent, respectively, while reducing average computational latency by 33 percent to 5.2 milliseconds compared to a multi-step MPC baseline. Additional simulation transfer in Gazebo and initial real-world deployment results further indicate the practicality and robustness of ARMS for safe and efficient human-robot collaboration. Source code and a demonstration video are available at this https URL.

**arXiv ID:** 2601.16686
</details>

<details>
<summary><strong>DAVOS: An Autonomous Vehicle Operating System in the Vehicle Computing Era</strong> - Yuxin Wang, Yuankai He, Boyang Tian, Lichen Xian, Weisong Shi - [[pdf]](https://arxiv.org/pdf/2601.05072)</summary>

**Abstract:** Vehicle computing represents a fundamental shift in how autonomous vehicles are designed and deployed, transforming them from isolated transportation systems into mobile computing platforms that support both safety-critical, real-time driving and data-centric services. In this setting, vehicles simultaneously support real-time driving pipelines and a growing set of data-driven applications, placing increased responsibility on the vehicle operating system to coordinate computation, data movement, storage, and access. These demands highlight recurring system considerations related to predictable execution, data and execution protection, efficient handling of high-rate sensor data, and long-term system evolvability, commonly summarized as Safety, Security, Efficiency, and Extensibility (SSEE). Existing vehicle operating systems and runtimes address these concerns in isolation, resulting in fragmented software stacks that limit coordination between autonomy workloads and vehicle data services. This paper presents DAVOS, the Dependable Autonomous Vehicle Operating System, a unified vehicle operating system architecture designed for the vehicle computing context. DAVOS provides a cohesive operating system foundation that supports both real-time autonomy and extensible vehicle computing within a single system framework.

**arXiv ID:** 2601.05072
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
