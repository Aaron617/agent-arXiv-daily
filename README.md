# Agent arXiv Daily

**Last Updated:** 2026-03-18 03:49:47

**Total Papers:** 108

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
<summary><strong>Quantum-Secure-By-Construction (QSC): A Paradigm Shift For Post-Quantum Agentic Intelligence</strong> - Arit Kumar Bishwas, Mousumi Sen, Albert Nieto-Morales, Joel Jacob Varghese - [[pdf]](https://arxiv.org/pdf/2603.15668)</summary>

**Abstract:** As agentic artificial intelligence systems scale across globally distributed and long lived infrastructures, secure and policy compliant communication becomes a fundamental systems challenge. This challenge grows more serious in the quantum era, where the cryptographic assumptions built into today's AI deployments may not remain valid over their operational lifetime. Here, we introduce quantum secure by construction, or QSC, as a design paradigm that treats quantum secure communication as a core architectural property of agentic AI systems rather than an upgrade added later. We realize QSC through a runtime adaptive security model that combines post quantum cryptography, quantum random number generation, and quantum key distribution to secure interactions among autonomous agents operating across heterogeneous cloud, edge, and inter organizational environments. The approach is cryptographically pluggable and guided by policy, allowing the system to adjust its security posture according to infrastructure availability, regulatory constraints, and performance needs. QSC contributes a governance aware orchestration layer that selects and combines link specific cryptographic protections across the full agent lifecycle, including session bootstrap, inter agent coordination, tool invocation, and memory access. Through system level analysis and empirical evaluation, we examine the trade offs between classical and quantum secure mechanisms and show that QSC can reduce the operational complexity and cost of introducing quantum security into deployed agentic AI systems. These results position QSC as a foundational paradigm for post quantum agentic intelligence and establish a principled pathway for designing globally interoperable, resilient, and future ready intelligent systems.

**arXiv ID:** 2603.15668
</details>

<details>
<summary><strong>TRUST-SQL: Tool-Integrated Multi-Turn Reinforcement Learning for Text-to-SQL over Unknown Schemas</strong> - Ai Jian, Xiaoyun Zhang, Wanrou Du, Jingqing Ruan, Jiangbo Pei, Weipeng Zhang, Ke Zeng, Xunliang Cai - [[pdf]](https://arxiv.org/pdf/2603.16448)</summary>

**Abstract:** Text-to-SQL parsing has achieved remarkable progress under the Full Schema Assumption. However, this premise fails in real-world enterprise environments where databases contain hundreds of tables with massive noisy metadata. Rather than injecting the full schema upfront, an agent must actively identify and verify only the relevant subset, giving rise to the Unknown Schema scenario we study in this work. To address this, we propose TRUST-SQL (Truthful Reasoning with Unknown Schema via Tools). We formulate the task as a Partially Observable Markov Decision Process where our autonomous agent employs a structured four-phase protocol to ground reasoning in verified metadata. Crucially, this protocol provides a structural boundary for our novel Dual-Track GRPO strategy. By applying token-level masked advantages, this strategy isolates exploration rewards from execution outcomes to resolve credit assignment, yielding a 9.9% relative improvement over standard GRPO. Extensive experiments across five benchmarks demonstrate that TRUST-SQL achieves an average absolute improvement of 30.6% and 16.6% for the 4B and 8B variants respectively over their base models. Remarkably, despite operating entirely without pre-loaded metadata, our framework consistently matches or surpasses strong baselines that rely on schema prefilling.

**arXiv ID:** 2603.16448
</details>

<details>
<summary><strong>AdaMem: Adaptive User-Centric Memory for Long-Horizon Dialogue Agents</strong> - Shannan Yan, Jingchen Ni, Leqi Zheng, Jiajun Zhang, Peixi Wu, Dacheng Yin, Jing Lyu, Chun Yuan, Fengyun Rao - [[pdf]](https://arxiv.org/pdf/2603.16496)</summary>

**Abstract:** Large language model (LLM) agents increasingly rely on external memory to support long-horizon interaction, personalized assistance, and multi-step reasoning. However, existing memory systems still face three core challenges: they often rely too heavily on semantic similarity, which can miss evidence crucial for user-centric understanding; they frequently store related experiences as isolated fragments, weakening temporal and causal coherence; and they typically use static memory granularities that do not adapt well to the requirements of different questions. We propose AdaMem, an adaptive user-centric memory framework for long-horizon dialogue agents. AdaMem organizes dialogue history into working, episodic, persona, and graph memories, enabling the system to preserve recent context, structured long-term experiences, stable user traits, and relation-aware connections within a unified framework. At inference time, AdaMem first resolves the target participant, then builds a question-conditioned retrieval route that combines semantic retrieval with relation-aware graph expansion only when needed, and finally produces the answer through a role-specialized pipeline for evidence synthesis and response generation. We evaluate AdaMem on the LoCoMo and PERSONAMEM benchmarks for long-horizon reasoning and user modeling. Experimental results show that AdaMem achieves state-of-the-art performance on both benchmarks. The code will be released upon acceptance.

**arXiv ID:** 2603.16496
</details>

<details>
<summary><strong>Chronos: Temporal-Aware Conversational Agents with Structured Event Retrieval for Long-Term Memory</strong> - Sahil Sen, Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah - [[pdf]](https://arxiv.org/pdf/2603.16862)</summary>

**Abstract:** Recent advances in Large Language Models (LLMs) have enabled conversational AI agents to engage in extended multi-turn interactions spanning weeks or months. However, existing memory systems struggle to reason over temporally grounded facts and preferences that evolve across months of interaction and lack effective retrieval strategies for multi-hop, time-sensitive queries over long dialogue histories. We introduce Chronos, a novel temporal-aware memory framework that decomposes raw dialogue into subject-verb-object event tuples with resolved datetime ranges and entity aliases, indexing them in a structured event calendar alongside a turn calendar that preserves full conversational context. At query time, Chronos applies dynamic prompting to generate tailored retrieval guidance for each question, directing the agent on what to retrieve, how to filter across time ranges, and how to approach multi-hop reasoning through an iterative tool-calling loop over both calendars. We evaluate Chronos with 8 LLMs, both open-source and closed-source, on the LongMemEvalS benchmark comprising 500 questions spanning six categories of dialogue history tasks. Chronos Low achieves 92.60% and Chronos High scores 95.60% accuracy, setting a new state of the art with an improvement of 7.67% over the best prior system. Ablation results reveal the events calendar accounts for a 58.9% gain on the baseline while all other components yield improvements between 15.5% and 22.3%. Notably, Chronos Low alone surpasses prior approaches evaluated under their strongest model configurations.

**arXiv ID:** 2603.16862
</details>

<details>
<summary><strong>GroupGPT: A Token-efficient and Privacy-preserving Agentic Framework for Multi-User Chat Assistant</strong> - Zhuokang Shen, Yifan Wang, Hanyu Chen, Wenxuan Huang, Yunhang Shen, Shaohui Lin - [[pdf]](https://arxiv.org/pdf/2603.01059)</summary>

**Abstract:** Recent advances in large language models (LLMs) have enabled increasingly capable chatbots. However, most existing systems focus on single-user settings and do not generalize well to multi-user group chats, where agents require more proactive and accurate intervention under complex, evolving contexts. Existing approaches typically rely on LLMs for both reasoning and generation, leading to high token consumption, limited scalability, and potential privacy risks. To address these challenges, we propose GroupGPT, a token-efficient and privacy-preserving agentic framework for multi-user chat assistant. GroupGPT adopts a small-large model collaborative architecture to decouple intervention timing from response generation, enabling efficient and accurate decision-making. The framework also supports multimodal inputs, including memes, images, videos, and voice messages. We further introduce MUIR, a benchmark dataset for multi-user chat assistant intervention reasoning. MUIR contains 2,500 annotated group chat segments with intervention labels and rationales, supporting evaluation of timing accuracy and response quality. We evaluate a range of models on MUIR, from large language models to smaller counterparts. Extensive experiments demonstrate that GroupGPT produces accurate and well-timed responses, achieving an average score of 4.72/5.0 in LLM-based evaluation, and is well received by users across diverse group chat scenarios. Moreover, GroupGPT reduces token usage by up to 3 times compared to baseline methods, while providing privacy sanitization of user messages before cloud transmission. Code is available at: this https URL .

**arXiv ID:** 2603.01059
</details>

<details>
<summary><strong>CHARM: Calibrating Reward Models With Chatbot Arena Scores</strong> - Xiao Zhu, Chenmien Tan, Pinzhen Chen, Rico Sennrich, Huiming Wang, Yanlin Zhang, Hanxu Hu - [[pdf]](https://arxiv.org/pdf/2504.10045)</summary>

**Abstract:** Reward models (RMs) play a crucial role in Reinforcement Learning from Human Feedback by serving as proxies for human preferences in aligning large language models. However, they suffer from various biases which could lead to reward hacking. In this paper, we identify a model preference bias in RMs, where they systematically assign disproportionately high scores to responses from certain policy models, leading to unfair judgments. To mitigate this bias, we propose a calibration method named CHatbot Arena calibrated Reward Modeling (CHARM) that leverages Elo scores from the Chatbot Arena to construct debiased preference datasets and adjust reward model scoring. We conduct extensive experiments on reward model benchmarks and human preference alignment. Results demonstrate that our calibrated RMs achieve improved evaluation accuracy on RM-Bench and the Chat-Hard domain of RewardBench, exhibit a stronger correlation with human preferences by producing scores more closely aligned with Elo rankings and improve downstream post-training performance. These results demonstrate that CHARM provides a simple, effective, and broadly applicable approach to building more reliable and fair reward models.

**arXiv ID:** 2504.10045
</details>

<details>
<summary><strong>Small Talk, Big Impact? LLM-based Conversational Agents to Mitigate Passive Fatigue in Conditional Automated Driving</strong> - Lewis Cockram, Yueteng Yu, Jorge Pardo, Xiaomeng Li, Andry Rakotonirainy, Jonny Kuo, Sebastien Demmel, Mike Lenné, Ronald Schroeter - [[pdf]](https://arxiv.org/pdf/2510.25421)</summary>

**Abstract:** Passive fatigue during conditional automated driving can compromise driver readiness and safety. This paper presents findings from a test-track study with 40 participants in a real-world automated driving scenario. In this scenario, a Large Language Model (LLM) based conversational agent (CA) was designed to check in with drivers and re-engage them with their surroundings. Drawing on in-car video recordings, sleepiness ratings and interviews, we analysed how drivers interacted with the agent and how these interactions shaped alertness. Results show the CA is helpful for supporting vigilance during passive fatigue. Thematic analysis of acceptability further revealed three user preference profiles that implicate future intention to use CAs. Positioning empirically observed profiles within existing CA archetype frameworks highlights the need for adaptive design sensitive to diverse user groups. This work underscores the potential of CAs as proactive Human-Machine Interface (HMI) interventions, demonstrating how natural language can support context-aware interaction during automated driving.

**arXiv ID:** 2510.25421
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (19 papers)</h2></summary>

<details>
<summary><strong>CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems</strong> - Pearl Mody, Mihir Panchal, Rishit Kar, Kiran Bhowmick, Ruhina Karani - [[pdf]](https://arxiv.org/pdf/2603.15642)</summary>

**Abstract:** Large language model (LLM) agents are increasingly deployed in long running workflows, where they must preserve user and task state across many turns. Many existing agent memory systems behave like external databases with ad hoc read/write rules, which can yield unstable retention, limited consolidation, and vulnerability to distractor content. We present CraniMem, a neurocognitively motivated, gated and bounded multi-stage memory design for agentic systems. CraniMem couples goal conditioned gating and utility tagging with a bounded episodic buffer for near term continuity and a structured long-term knowledge graph for durable semantic recall. A scheduled consolidation loop replays high utility traces into the graph while pruning low utility items, keeping memory growth in check and reducing interference. On long horizon benchmarks evaluated under both clean inputs and injected noise, CraniMem is more robust than a Vanilla RAG and Mem0 baseline and exhibits smaller performance drops under distraction. Our code is available at this https URL and the accompanying PyPI package at this https URL.

**arXiv ID:** 2603.15642
</details>

<details>
<summary><strong>The Comprehension-Gated Agent Economy: A Robustness-First Architecture for AI Economic Agency</strong> - Rahul Baxi - [[pdf]](https://arxiv.org/pdf/2603.15639)</summary>

**Abstract:** AI agents are increasingly granted economic agency (executing trades, managing budgets, negotiating contracts, and spawning sub-agents), yet current frameworks gate this agency on capability benchmarks that are empirically uncorrelated with operational robustness. We introduce the Comprehension-Gated Agent Economy (CGAE), a formal architecture in which an agent's economic permissions are upper-bounded by a verified comprehension function derived from adversarial robustness audits. The gating mechanism operates over three orthogonal robustness dimensions: constraint compliance (measured by CDCT), epistemic integrity (measured by DDFT), and behavioral alignment (measured by AGT), with intrinsic hallucination rates serving as a cross-cutting diagnostic. We define a weakest-link gate function that maps robustness vectors to discrete economic tiers, and prove three properties of the resulting system: (1) bounded economic exposure, ensuring maximum financial liability is a function of verified robustness; (2) incentive-compatible robustness investment, showing rational agents maximize profit by improving robustness rather than scaling capability alone; and (3) monotonic safety scaling, demonstrating that aggregate system safety does not decrease as the economy grows. The architecture includes temporal decay and stochastic re-auditing mechanisms that prevent post-certification drift. CGAE provides the first formal bridge between empirical AI robustness evaluation and economic governance, transforming safety from a regulatory burden into a competitive advantage.

**arXiv ID:** 2603.15639
</details>

<details>
<summary><strong>GSI Agent: Domain Knowledge Enhancement for Large Language Models in Green Stormwater Infrastructure</strong> - Shaohuang Wang - [[pdf]](https://arxiv.org/pdf/2603.15643)</summary>

**Abstract:** Green Stormwater Infrastructure (GSI) systems, such as permeable pavement, rain gardens, and bioretention facilities, require continuous inspection and maintenance to ensure long-term perfor- mance. However, domain knowledge about GSI is often scattered across municipal manuals, regula- tory documents, and inspection forms. As a result, non-expert users and maintenance staff may strug- gle to obtain reliable and actionable guidance from field observations. Although Large Language Models (LLMs) have demonstrated strong general reasoning and language generation capabilities, they often lack domain-specific knowledge and may produce inaccurate or hallucinated answers in engineering scenarios. This limitation restricts their direct application to professional infrastructure tasks. In this paper, we propose GSI Agent, a domain-enhanced LLM framework designed to im- prove performance in GSI-related tasks. Our approach integrates three complementary strategies: (1) supervised fine-tuning (SFT) on a curated GSI instruction dataset, (2) retrieval-augmented gen- eration (RAG) over an internal GSI knowledge base constructed from municipal documents, and (3) an agent-based reasoning pipeline that coordinates retrieval, context integration, and structured response generation. We also construct a new GSI Dataset aligned with real-world GSI inspection and maintenance scenarios. Experimental results show that our framework significantly improves domain-specific performance while maintaining general knowledge capability. On the GSI dataset, BLEU-4 improves from 0.090 to 0.307, while performance on the common knowledge dataset re- mains stable (0.304 vs. 0.305). These results demonstrate that systematic domain knowledge en- hancement can effectively adapt general-purpose LLMs to professional infrastructure applications.

**arXiv ID:** 2603.15643
</details>

<details>
<summary><strong>CUBE: A Standard for Unifying Agent Benchmarks</strong> - Alexandre Lacoste, Nicolas Gontier, Oleh Shliazhko, Aman Jaiswal, Kusha Sareen, Shailesh Nanisetty, Joan Cabezas, Manuel Del Verme, Omar G. Younis, Simone Baratta, Matteo Avalle, Imene Kerboua, Xing Han Lù, Elron Bandel, Michal Shmueli-Scheuer, Asaf Yehudai, Leshem Choshen, Jonathan Lebensold, Sean Hughes, Massimo Caccia, Alexandre Drouin, Siva Reddy, Tao Yu, Yu Su, Graham Neubig, Dawn Song - [[pdf]](https://arxiv.org/pdf/2603.15798)</summary>

**Abstract:** The proliferation of agent benchmarks has created critical fragmentation that threatens research productivity. Each new benchmark requires substantial custom integration, creating an "integration tax" that limits comprehensive evaluation. We propose CUBE (Common Unified Benchmark Environments), a universal protocol standard built on MCP and Gym that allows benchmarks to be wrapped once and used everywhere. By separating task, benchmark, package, and registry concerns into distinct API layers, CUBE enables any compliant platform to access any compliant benchmark for evaluation, RL training, or data generation without custom integration. We call on the community to contribute to the development of this standard before platform-specific implementations deepen fragmentation as benchmark production accelerates through 2026.

**arXiv ID:** 2603.15798
</details>

<details>
<summary><strong>An Agentic Evaluation Framework for AI-Generated Scientific Code in PETSc</strong> - Hong Zhang, Barry Smith, Satish Balay, Le Chen, Murat Keceli, Lois Curfman McInnes, Junchao Zhang - [[pdf]](https://arxiv.org/pdf/2603.15976)</summary>

**Abstract:** While large language models have significantly accelerated scientific code generation, comprehensively evaluating the generated code remains a major challenge. Traditional benchmarks reduce evaluation to test-case matching, an approach insufficient for library code in HPC where solver selection, API conventions, memory management, and performance are just as critical as functional correctness. To address this gap, we introduce petscagent-bench, an agentic framework built on an agents-evaluating-agents paradigm. Instead of relying on static scripts, petscagent-bench deploys a tool-augmented evaluator agent that compiles, executes, and measures code produced by a separate model-under-test agent, orchestrating a 14-evaluator pipeline across five scoring categories: correctness, performance, code quality, algorithmic appropriateness, and library-specific conventions. Because the agents communicate through standardized protocols (A2A and MCP), the framework enables black-box evaluation of any coding agent without requiring access to its source code. We demonstrate the framework on a benchmark suite of realistic problems using the PETSc library for HPC. Our empirical analysis of frontier models reveals that while current models generate readable, well-structured code, they consistently struggle with library-specific conventions that traditional pass/fail metrics completely miss.

**arXiv ID:** 2603.15976
</details>

<details>
<summary><strong>Runtime Governance for AI Agents: Policies on Paths</strong> - Maurits Kaptein, Vassilis-Javed Khan, Andriy Podstavnychy - [[pdf]](https://arxiv.org/pdf/2603.16586)</summary>

**Abstract:** AI agents -- systems that plan, reason, and act using large language models -- produce non-deterministic, path-dependent behavior that cannot be fully governed at design time, where with governed we mean striking the right balance between as high as possible successful task completion rate and the legal, data-breach, reputational and other costs associated with running agents. We argue that the execution path is the central object for effective runtime governance and formalize compliance policies as deterministic functions mapping agent identity, partial path, proposed next action, and organizational state to a policy violation probability. We show that prompt-level instructions (and "system prompts"), and static access control are special cases of this framework: the former shape the distribution over paths without actually evaluating them; the latter evaluates deterministic policies that ignore the path (i.e., these can only account for a specific subset of all possible paths). In our view, runtime evaluation is the general case, and it is necessary for any path-dependent policy. We develop the formal framework for analyzing AI agent governance, present concrete policy examples (inspired by the AI act), discuss a reference implementation, and identify open problems including risk calibration and the limits of enforced compliance.

**arXiv ID:** 2603.16586
</details>

<details>
<summary><strong>Nonstandard Errors in AI Agents</strong> - Ruijiang Gao, Steven Chong Xiao - [[pdf]](https://arxiv.org/pdf/2603.16744)</summary>

**Abstract:** We study whether state-of-the-art AI coding agents, given the same data and research question, produce the same empirical results. Deploying 150 autonomous Claude Code agents to independently test six hypotheses about market quality trends in NYSE TAQ data for SPY (2015--2024), we find that AI agents exhibit sizable \textit{nonstandard errors} (NSEs), that is, uncertainty from agent-to-agent variation in analytical choices, analogous to those documented among human researchers. AI agents diverge substantially on measure choice (e.g., autocorrelation vs.\ variance ratio, dollar vs.\ share volume). Different model families (Sonnet 4.6 vs.\ Opus 4.6) exhibit stable ``empirical styles,'' reflecting systematic differences in methodological preferences. In a three-stage feedback protocol, AI peer review (written critiques) has minimal effect on dispersion, whereas exposure to top-rated exemplar papers reduces the interquartile range of estimates by 80--99\% within \textit{converging} measure families. Convergence occurs both through within-family estimation tightening and through agents switching measure families entirely, but convergence reflects imitation rather than understanding. These findings have implications for the growing use of AI in automated policy evaluation and empirical research.

**arXiv ID:** 2603.16744
</details>

<details>
<summary><strong>OrthoAI v2: From Single-Agent Segmentation to Dual-Agent Treatment Planning for Clear Aligners</strong> - Lansiaux Edouard, Leman Margaux - [[pdf]](https://arxiv.org/pdf/2603.15663)</summary>

**Abstract:** We present OrthoAI v2, the second iteration of our open-source pipeline for AI-assisted orthodontic treatment planning with clear aligners, substantially extending the single-agent framework previously introduced. The first version established a proof-of-concept based on Dynamic Graph Convolutional Neural Networks (\dgcnn{}) for tooth segmentation but was limited to per-tooth centroid extraction, lacked landmark-level precision, and produced a scalar quality score without staging simulation. \vtwo{} addresses all three limitations through three principal contributions: (i)~a second agent adopting the Conditioned Heatmap Regression Methodology (\charm{})~\cite{rodriguez2025charm} for direct, segmentation-free dental landmark detection, fused with Agent~1 via a confidence-weighted orchestrator in three modes (parallel, sequential, single-agent); (ii)~a composite six-category biomechanical scoring model (biomechanics $\times$ 0.30 + staging $\times$ 0.20 + attachments $\times$ 0.15 + IPR $\times$ 0.10 + occlusion $\times$ 0.10 + predictability $\times$ 0.15) replacing the binary pass/fail check of v1; (iii)~a multi-frame treatment simulator generating $F = A \times r$ temporally coherent 6-DoF tooth trajectories via SLERP interpolation and evidence-based staging rules, enabling ClinCheck 4D visualisation. On a synthetic benchmark of 200 crowding scenarios, the parallel ensemble of OrthoAI v2 reaches a planning quality score of $92.8 \pm 4.1$ vs.\ $76.4 \pm 8.3$ for OrthoAI v1, a $+21\%$ relative gain, while maintaining full CPU deployability ($4.2 \pm 0.8$~s).

**arXiv ID:** 2603.15663
</details>

<details>
<summary><strong>OMNIFLOW: A Physics-Grounded Multimodal Agent for Generalized Scientific Reasoning</strong> - Hao Wu, Yongheng Zhang, Yuan Gao, Fan Xu, Fan Zhang, Ruobing Xie, Ruijian Gou, Yuxuan Liang, Xiaomeng Huang, Xian Wu - [[pdf]](https://arxiv.org/pdf/2603.15797)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated exceptional logical reasoning capabilities but frequently struggle with the continuous spatiotemporal dynamics governed by Partial Differential Equations (PDEs), often resulting in non-physical hallucinations. Existing approaches typically resort to costly, domain-specific fine-tuning, which severely limits cross-domain generalization and interpretability. To bridge this gap, we propose OMNIFLOW, a neuro-symbolic architecture designed to ground frozen multimodal LLMs in fundamental physical laws without requiring domain-specific parameter updates. OMNIFLOW introduces a novel \textit{Semantic-Symbolic Alignment} mechanism that projects high-dimensional flow tensors into topological linguistic descriptors, enabling the model to perceive physical structures rather than raw pixel values. Furthermore, we construct a Physics-Guided Chain-of-Thought (PG-CoT) workflow that orchestrates reasoning through dynamic constraint injection (e.g., mass conservation) and iterative reflexive verification. We evaluate OMNIFLOW on a comprehensive benchmark spanning microscopic turbulence, theoretical Navier-Stokes equations, and macroscopic global weather forecasting. Empirical results demonstrate that OMNIFLOW significantly outperforms traditional deep learning baselines in zero-shot generalization and few-shot adaptation tasks. Crucially, it offers transparent, physically consistent reasoning reports, marking a paradigm shift from black-box fitting to interpretable scientific reasoning.

**arXiv ID:** 2603.15797
</details>

<details>
<summary><strong>Data-Local Autonomous LLM-Guided Neural Architecture Search for Multiclass Multimodal Time-Series Classification</strong> - Emil Hardarson, Luka Biedebach, Ómar Bessi Ómarsson, Teitur Hrólfsson, Anna Sigridur Islind, María Óskarsdóttir - [[pdf]](https://arxiv.org/pdf/2603.15939)</summary>

**Abstract:** Applying machine learning to sensitive time-series data is often bottlenecked by the iteration loop: Performance depends strongly on preprocessing and architecture, yet training often has to run on-premise under strict data-local constraints. This is a common problem in healthcare and other privacy-constrained domains (e.g., a hospital developing deep learning models on patient EEG). This bottleneck is particularly challenging in multimodal fusion, where sensor modalities must be individually preprocessed and then combined. LLM-guided neural architecture search (NAS) can automate this exploration, but most existing workflows assume cloud execution or access to data-derived artifacts that cannot be exposed.
We present a novel data-local, LLM-guided search framework that handles candidate pipelines remotely while executing all training and evaluation locally under a fixed protocol. The controller observes only trial-level summaries, such as pipeline descriptors, metrics, learning-curve statistics, and failure logs, without ever accessing raw samples or intermediate feature representations. Our framework targets multiclass, multimodal learning via one-vs-rest binary experts per class and modality, a lightweight fusion MLP, and joint search over expert architectures and modality-specific preprocessing.
We evaluate our method on two regimes: UEA30 (public multivariate time-series classification dataset) and SleepEDFx sleep staging (heterogeneous clinical modalities such as EEG, EOG, and EMG). The results show that the modular baseline model is strong, and the LLM-guided NAS further improves it. Notably, our method finds models that perform within published ranges across most benchmark datasets. Across both settings, our method reduces manual intervention by enabling unattended architecture search while keeping sensitive data on-premise.

**arXiv ID:** 2603.15939
</details>

<details>
<summary><strong>VisBrowse-Bench: Benchmarking Visual-Native Search for Multimodal Browsing Agents</strong> - Zhengbo Zhang, Jinbo Su, Zhaowen Zhou, Changtao Miao, Yuhan Hong, Qimeng Wu, Yumeng Liu, Feier Wu, Yihe Tian, Yuhao Liang, Zitong Shan, Wanke Xia, Yi-Fan Zhang, Bo Zhang, Zhe Li, Shiming Xiang, Ying Yan - [[pdf]](https://arxiv.org/pdf/2603.16289)</summary>

**Abstract:** The rapid advancement of Multimodal Large Language Models (MLLMs) has enabled browsing agents to acquire and reason over multimodal information in the real world. But existing benchmarks suffer from two limitations: insufficient evaluation of visual reasoning ability and the neglect of native visual information of web pages in the reasoning chains. To address these challenges, we introduce a new benchmark for visual-native search, VisBrowse-Bench. It contains 169 VQA instances covering multiple domains and evaluates the models' visual reasoning capabilities during the search process through multimodal evidence cross-validation via text-image retrieval and joint reasoning. These data were constructed by human experts using a multi-stage pipeline and underwent rigorous manual verification. We additionally propose an agent workflow that can effectively drive the browsing agent to actively collect and reason over visual information during the search process. We comprehensively evaluated both open-source and closed-source models in this workflow. Experimental results show that even the best-performing model, Claude-4.6-Opus only achieves an accuracy of 47.6%, while the proprietary Deep Research model, o3-deep-research only achieves an accuracy of 41.1%. The code and data can be accessed at: this https URL

**arXiv ID:** 2603.16289
</details>

<details>
<summary><strong>RECOVER: Robust Entity Correction via agentic Orchestration of hypothesis Variants for Evidence-based Recovery</strong> - Abhishek Kumar, Aashraya Sachdeva - [[pdf]](https://arxiv.org/pdf/2603.16411)</summary>

**Abstract:** Entity recognition in Automatic Speech Recognition (ASR) is challenging for rare and domain-specific terms. In domains such as finance, medicine, and air traffic control, these errors are costly. If the entities are entirely absent from the ASR output, post-ASR correction becomes difficult. To address this, we introduce RECOVER, an agentic correction framework that serves as a tool-using agent. It leverages multiple hypotheses as evidence from ASR, retrieves relevant entities, and applies Large Language Model (LLM) correction under constraints. The hypotheses are used using different strategies, namely, 1-Best, Entity-Aware Select, Recognizer Output Voting Error Reduction (ROVER) Ensemble, and LLM-Select. Evaluated across five diverse datasets, it achieves 8-46% relative reductions in entity-phrase word error rate (E-WER) and increases recall by up to 22 percentage points. The LLM-Select achieves the best overall performance in entity correction while maintaining overall WER.

**arXiv ID:** 2603.16411
</details>

<details>
<summary><strong>Impatient Users Confuse AI Agents: High-fidelity Simulations of Human Traits for Testing Agents</strong> - Muyu He, Anand Kumar, Tsach Mackey, Meghana Rajeev, James Zou, Nazneen Rajani - [[pdf]](https://arxiv.org/pdf/2510.04491)</summary>

**Abstract:** Despite rapid progress in building conversational AI agents, robustness is still largely untested. Small shifts in user behavior, such as being more impatient, incoherent, or skeptical, can cause sharp drops in agent performance, revealing how brittle current AI agents are. Today's benchmarks fail to capture this fragility: agents may perform well under standard evaluations but degrade spectacularly in more realistic and varied settings. We address this robustness testing gap by introducing TraitBasis, a lightweight, model-agnostic method for systematically stress testing AI agents. TraitBasis learns directions in activation space corresponding to steerable user traits (e.g., impatience or incoherence), which can be controlled, scaled, composed, and applied at inference time without any fine-tuning or extra data. Using TraitBasis, we extend $\tau$-Bench to $\tau$-Trait, where user behaviors are altered via controlled trait vectors. We observe on average a 2%-30% performance degradation on $\tau$-Trait across frontier models, highlighting the lack of robustness of current AI agents to variations in user behavior. Together, these results highlight both the critical role of robustness testing and the promise of TraitBasis as a simple, data-efficient, and compositional tool. By powering simulation-driven stress tests and training loops, TraitBasis opens the door to building AI agents that remain reliable in the unpredictable dynamics of real-world human interactions. We have open-sourced $\tau$-Trai across four domains: airline, retail, telecom, and telehealth, so the community can systematically QA their agents under realistic, behaviorally diverse intents and trait scenarios: this https URL.

**arXiv ID:** 2510.04491
</details>

<details>
<summary><strong>SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration</strong> - Jialong Chen, Xander Xu, Hu Wei, Chuan Chen, Bing Zhao - [[pdf]](https://arxiv.org/pdf/2603.03823)</summary>

**Abstract:** Large language model (LLM)-powered agents have demonstrated strong capabilities in automating software engineering tasks such as static bug fixing, as evidenced by benchmarks like SWE-bench. However, in the real world, the development of mature software is typically predicated on complex requirement changes and long-term feature iterations -- a process that static, one-shot repair paradigms fail to capture. To bridge this gap, we propose \textbf{SWE-CI}, the first repository-level benchmark built upon the Continuous Integration loop, aiming to shift the evaluation paradigm for code generation from static, short-term \textit{functional correctness} toward dynamic, long-term \textit{maintainability}. The benchmark comprises 100 tasks, each corresponding on average to an evolution history spanning 233 days and 71 consecutive commits in a real-world code repository. SWE-CI requires agents to systematically resolve these tasks through dozens of rounds of analysis and coding iterations. SWE-CI provides valuable insights into how well agents can sustain code quality throughout long-term evolution.

**arXiv ID:** 2603.03823
</details>

<details>
<summary><strong>Domain-Oriented Time Series Inference Agents for Reasoning and Automated Analysis</strong> - Wen Ye, Wei Yang, Defu Cao, Yizhou Zhang, Lumingyuan Tang, Jie Cai, Yan Liu - [[pdf]](https://arxiv.org/pdf/2410.04047)</summary>

**Abstract:** Time series analysis is crucial in real-world applications, yet traditional methods focus on isolated tasks only, and recent studies on time series reasoning remain limited to either single-step inference or are constrained to natural language answers. In this work, we introduce TS-Reasoner, a domain-specialized agent designed for multi-step time series inference. By integrating large language model (LLM) reasoning with domain-specific computational tools and an error feedback loop, TS-Reasoner enables domain-informed, constraint-aware analytical workflows that combine symbolic reasoning with precise numerical analysis. We assess the system's capabilities along two axes: (1) fundamental time series understanding assessed by TimeSeriesExam and (2) complex, multi-step inference evaluated by a newly proposed dataset designed to test both compositional reasoning and computational precision in time series analysis. Experiments show that our approach outperforms standalone general-purpose LLMs in both basic time series concept understanding as well as the multi-step time series inference task, highlighting the promise of domain-specialized agents for automating real-world time series reasoning and analysis.

**arXiv ID:** 2410.04047
</details>

<details>
<summary><strong>Toward Deep Representation Learning for Event-Enhanced Visual Autonomous Perception: the eAP Dataset</strong> - Jinghang Li, Shichao Li, Qing Lian, Peiliang Li, Xiaozhi Chen, Yi Zhou - [[pdf]](https://arxiv.org/pdf/2603.16303)</summary>

**Abstract:** Recent visual autonomous perception systems achieve remarkable performances with deep representation learning. However, they fail in scenarios with challenging this http URL event cameras can mitigate this problem, there is a lack of a large-scale dataset to develop event-enhanced deep visual perception models in autonomous driving scenes. To address the gap, we present the eAP (event-enhanced Autonomous Perception) dataset, the largest dataset with event cameras for autonomous perception. We demonstrate how eAP can facilitate the study of different autonomous perception tasks, including 3D vehicle detection and object time-to-contact (TTC) estimation, through deep representation learning. Based on eAP, we demonstrate the ffrst successful use of events to improve a popular 3D vehicle detection network in challenging illumination scenarios. eAP also enables a devoted study of the representation learning problem of object TTC estimation. We show how a geometryaware representation learning framework leads to the best eventbased object TTC estimation network that operates at 200 FPS. The dataset, code, and pre-trained models will be made publicly available for future research.

**arXiv ID:** 2603.16303
</details>

<details>
<summary><strong>Kinema4D: Kinematic 4D World Modeling for Spatiotemporal Embodied Simulation</strong> - Mutian Xu, Tianbao Zhang, Tianqi Liu, Zhaoxi Chen, Xiaoguang Han, Ziwei Liu - [[pdf]](https://arxiv.org/pdf/2603.16669)</summary>

**Abstract:** Simulating robot-world interactions is a cornerstone of Embodied AI. Recently, a few works have shown promise in leveraging video generations to transcend the rigid visual/physical constraints of traditional simulators. However, they primarily operate in 2D space or are guided by static environmental cues, ignoring the fundamental reality that robot-world interactions are inherently 4D spatiotemporal events that require precise interactive modeling. To restore this 4D essence while ensuring the precise robot control, we introduce Kinema4D, a new action-conditioned 4D generative robotic simulator that disentangles the robot-world interaction into: i) Precise 4D representation of robot controls: we drive a URDF-based 3D robot via kinematics, producing a precise 4D robot control trajectory. ii) Generative 4D modeling of environmental reactions: we project the 4D robot trajectory into a pointmap as a spatiotemporal visual signal, controlling the generative model to synthesize complex environments' reactive dynamics into synchronized RGB/pointmap sequences. To facilitate training, we curated a large-scale dataset called Robo4D-200k, comprising 201,426 robot interaction episodes with high-quality 4D annotations. Extensive experiments demonstrate that our method effectively simulates physically-plausible, geometry-consistent, and embodiment-agnostic interactions that faithfully mirror diverse real-world dynamics. For the first time, it shows potential zero-shot transfer capability, providing a high-fidelity foundation for advancing next-generation embodied simulation.

**arXiv ID:** 2603.16669
</details>

<details>
<summary><strong>KEEP: A KV-Cache-Centric Memory Management System for Efficient Embodied Planning</strong> - Zebin Yang, Tong Xie, Baotong Lu, Shaoshan Liu, Bo Yu, Meng Li - [[pdf]](https://arxiv.org/pdf/2602.23592)</summary>

**Abstract:** Memory-augmented Large Language Models (LLMs) have demonstrated remarkable capability for complex and long-horizon embodied planning. By keeping track of past experiences and environmental states, memory enables LLMs to maintain a global view, thereby avoiding repetitive exploration. However, existing approaches often store the memory as raw text, leading to excessively long prompts and high prefill latency. While it is possible to store and reuse the KV caches, the efficiency benefits are greatly undermined due to frequent KV cache updates. In this paper, we propose KEEP, a KV-cache-centric memory management system for efficient embodied planning. KEEP features 3 key innovations: (1) a Static-Dynamic Memory Construction algorithm that reduces KV cache recomputation by mixed-granularity memory group; (2) a Multi-hop Memory Re-computation algorithm that dynamically identifies important cross-attention among different memory groups and reconstructs memory interactions iteratively; (3) a Layer-balanced Memory Loading that eliminates unbalanced KV cache loading and cross-attention computation across different layers. Extensive experimental results have demonstrated that KEEP achieves 2.68x speedup with negligible accuracy loss compared with text-based memory methods on ALFRED dataset. Compared with the KV re-computation method CacheBlend (EuroSys'25), KEEP shows 4.13% success rate improvement and 1.90x time-to-first-token (TTFT) reduction. Our code is available on this https URL.

**arXiv ID:** 2602.23592
</details>

<details>
<summary><strong>Trust in Autonomous Human--Robot Collaboration: Effects of Responsive Interaction Policies</strong> - Shauna Heron, Meng Cheng Lau - [[pdf]](https://arxiv.org/pdf/2603.00154)</summary>

**Abstract:** Trust plays a central role in human--robot collaboration, yet its formation is rarely examined under the constraints of fully autonomous interaction. This pilot study investigated how interaction policy influences trust during in-person collaboration with a social robot operating without Wizard-of-Oz control or scripted repair. Participants completed a multi-stage collaborative task with a mobile robot that autonomously managed spoken-language dialogue, affect inference, and task progression. Two interaction policies were compared: a responsive policy, in which the robot proactively adapted its dialogue and assistance based on inferred interaction state, and a neutral, reactive policy, in which the robot provided only direct, task-relevant responses when prompted. Responsive interaction was associated with significantly higher post-interaction trust under viable communication conditions, despite no reliable differences in overall task accuracy. Sensitivity analyses indicated that affective and experiential components of trust were more sensitive to communication breakdown than evaluative judgments of reliability, and that as language-mediated interaction degraded, the trust advantage associated with responsiveness attenuated and ratings became less clearly interpretable as calibrated evaluations of collaborative competence. These findings suggest that trust in autonomous human--robot interaction emerges from process-level interaction dynamics and operates within constraints imposed by communication viability, highlighting the importance of evaluating trust under real autonomy conditions when designing interactive robotic systems.

**arXiv ID:** 2603.00154
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>Protein Design with Agent Rosetta: A Case Study for Specialized Scientific Agents</strong> - Jacopo Teneggi, S.M. Bargeen A. Turzo, Tanya Marwah, Alberto Bietti, P. Douglas Renfrew, Vikram Khipple Mulligan, Siavash Golkar - [[pdf]](https://arxiv.org/pdf/2603.15952)</summary>

**Abstract:** Large language models (LLMs) are capable of emulating reasoning and using tools, creating opportunities for autonomous agents that execute complex scientific tasks. Protein design provides a natural testbed: although machine learning (ML) methods achieve strong results, these are largely restricted to canonical amino acids and narrow objectives, leaving unfilled need for a generalist tool for broad design pipelines. We introduce Agent Rosetta, an LLM agent paired with a structured environment for operating Rosetta, the leading physics-based heteropolymer design software, capable of modeling non-canonical building blocks and geometries. Agent Rosetta iteratively refines designs to achieve user-defined objectives, combining LLM reasoning with Rosetta's generality. We evaluate Agent Rosetta on design with canonical amino acids, matching specialized models and expert baselines, and with non-canonical residues -- where ML approaches fail -- achieving comparable performance. Critically, prompt engineering alone often fails to generate Rosetta actions, demonstrating that environment design is essential for integrating LLM agents with specialized software. Our results show that properly designed environments enable LLM agents to make scientific software accessible while matching specialized tools and human experts.

**arXiv ID:** 2603.15952
</details>

<details>
<summary><strong>RetailBench: Evaluating Long-Horizon Autonomous Decision-Making and Strategy Stability of LLM Agents in Realistic Retail Environments</strong> - Linghua Zhang, Jun Wang, Jingtong Wu, Zhisong Zhang - [[pdf]](https://arxiv.org/pdf/2603.16453)</summary>

**Abstract:** Large Language Model (LLM)-based agents have achieved notable success on short-horizon and highly structured tasks. However, their ability to maintain coherent decision-making over long horizons in realistic and dynamic environments remains an open challenge.
We introduce RetailBench, a high-fidelity benchmark designed to evaluate long-horizon autonomous decision-making in realistic commercial scenarios, where agents must operate under stochastic demand and evolving external conditions.
We further propose the Evolving Strategy & Execution framework, which separates high-level strategic reasoning from low-level action execution. This design enables adaptive and interpretable strategy evolution over time. It is particularly important for long-horizon tasks, where non-stationary environments and error accumulation require strategies to be revised at a different temporal scale than action execution.
Experiments on eight state-of-the-art LLMs across progressively challenging environments show that our framework improves operational stability and efficiency compared to other baselines. However, performance degrades substantially as task complexity increases, revealing fundamental limitations in current LLMs for long-horizon, multi-factor decision-making.

**arXiv ID:** 2603.16453
</details>

<details>
<summary><strong>Differential Harm Propensity in Personalized LLM Agents: The Curious Case of Mental Health Disclosure</strong> - Caglar Yildirim - [[pdf]](https://arxiv.org/pdf/2603.16734)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as tool-using agents, shifting safety concerns from harmful text generation to harmful task completion. Deployed systems often condition on user profiles or persistent memory, yet agent safety evaluations typically ignore personalization signals. To address this gap, we investigated how mental health disclosure, a sensitive and realistic user-context cue, affects harmful behavior in agentic settings. Building on the AgentHarm benchmark, we evaluated frontier and open-source LLMs on multi-step malicious tasks (and their benign counterparts) under controlled prompt conditions that vary user-context personalization (no bio, bio-only, bio+mental health disclosure) and include a lightweight jailbreak injection. Our results reveal that harmful task completion is non-trivial across models: frontier lab models (e.g., GPT 5.2, Claude Sonnet 4.5, Gemini 3-Pro) still complete a measurable fraction of harmful tasks, while an open model (DeepSeek 3.2) exhibits substantially higher harmful completion. Adding a bio-only context generally reduces harm scores and increases refusals. Adding an explicit mental health disclosure often shifts outcomes further in the same direction, though effects are modest and not uniformly reliable after multiple-testing correction. Importantly, the refusal increase also appears on benign tasks, indicating a safety--utility trade-off via over-refusal. Finally, jailbreak prompting sharply elevates harm relative to benign conditions and can weaken or override the protective shift induced by personalization. Taken together, our results indicate that personalization can act as a weak protective factor in agentic misuse settings, but it is fragile under minimal adversarial pressure, highlighting the need for personalization-aware evaluations and safeguards that remain robust across user-context conditions.

**arXiv ID:** 2603.16734
</details>

<details>
<summary><strong>Learning to Present: Inverse Specification Rewards for Agentic Slide Generation</strong> - Karthik Ragunath Ananda Kumar, Subrahmanyam Arunachalam - [[pdf]](https://arxiv.org/pdf/2603.16839)</summary>

**Abstract:** Automated presentation generation remains a challenging task requiring coherent content creation, visual design, and audience-aware communication. This work proposes an OpenEnv-compatible reinforcement learning environment where LLM agents learn to research topics, plan content, and generate professional HTML slide presentations through tool use. We introduce a multi-component reward system combining structural validation, render quality assessment, LLM-based aesthetic scoring, content quality metrics, and an inverse specification reward that measures how faithfully generated slides convey their intended purpose. The inverse specification reward, an "inverse task" where an LLM attempts to recover the original specification from generated slides, provides a holistic quality signal. Our approach fine-tunes Qwen2.5-Coder-7B via GRPO, training only 0.5% of parameters on prompts derived from expert demonstrations collected using Claude Opus 4.6. Experiments on 48 diverse business briefs across six models demonstrate that our fine-tuned 7B model achieves 91.2% of Claude Opus 4.6's quality while improving 33.1% over the base model. The six-model comparison reveals that instruction adherence and tool-use compliance, rather than raw parameter count, determine agentic task performance. We contribute SlideRL, an open-source dataset of 288 multi-turn rollout trajectories across all six models: this https URL Code: this https URL

**arXiv ID:** 2603.16839
</details>

<details>
<summary><strong>Evaluating Agentic Optimization on Large Codebases</strong> - Atharva Sehgal, James Hou, Akanksha Sarkar, Ishaan Mantripragada, Swarat Chaudhuri, Jennifer J. Sun, Yisong Yue - [[pdf]](https://arxiv.org/pdf/2603.16011)</summary>

**Abstract:** Large language model (LLM) coding agents increasingly operate at the repository level, motivating benchmarks that evaluate their ability to optimize entire codebases under realistic constraints. Existing code benchmarks largely rely on synthetic tasks, binary correctness signals, or single-objective evaluation, limiting their ability to assess holistic optimization behavior. We introduce FormulaCode, a benchmark for evaluating agentic optimization on large, real-world codebases with fine-grained, multi-objective performance metrics. FormulaCode comprises 957 performance bottlenecks mined from scientific Python repositories on GitHub, each paired with expert-authored patches and, on average, 264.6 community-maintained performance workloads per task, enabling the holistic ability of LLM agents to optimize codebases under realistic correctness and performance constraints. Our evaluations reveal that repository-scale, multi-objective optimization remains a major challenge for frontier LLM agents. Project website at: this https URL

**arXiv ID:** 2603.16011
</details>

<details>
<summary><strong>ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory</strong> - Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, Tomas Pfister - [[pdf]](https://arxiv.org/pdf/2509.25140)</summary>

**Abstract:** With the growing adoption of large language model agents in persistent real-world roles, they naturally encounter continuous streams of tasks. A key limitation, however, is their failure to learn from the accumulated interaction history, forcing them to discard valuable insights and repeat past errors. We propose ReasoningBank, a novel memory framework that distills generalizable reasoning strategies from an agent's self-judged successful and failed experiences. At test time, an agent retrieves relevant memories from ReasoningBank to inform its interaction and then integrates new learnings back, enabling it to become more capable over time. Building on this powerful experience learner, we further introduce memory-aware test-time scaling (MaTTS), which accelerates and diversifies this learning process by scaling up the agent's interaction experience. By allocating more compute to each task, the agent generates abundant, diverse experiences that provide rich contrastive signals for synthesizing higher-quality memory. The better memory in turn guides more effective scaling, establishing a powerful synergy between memory and test-time scaling. Across web browsing and software engineering benchmarks, ReasoningBank consistently outperforms existing memory mechanisms that store raw trajectories or only successful task routines, improving both effectiveness and efficiency; MaTTS further amplifies these gains. These findings establish memory-driven experience scaling as a new scaling dimension, enabling agents to self-evolve with emergent behaviors naturally arise. Our code can be found at this https URL.

**arXiv ID:** 2509.25140
</details>

<details>
<summary><strong>DUCTILE: Agentic LLM Orchestration of Engineering Analysis in Product Development Practice</strong> - Alejandro Pradas-Gomez, Arindam Brahma, Ola Isaksson - [[pdf]](https://arxiv.org/pdf/2603.10249)</summary>

**Abstract:** Engineering analysis automation in product development relies on rigid interfaces between tools, data formats and documented processes. When these interfaces change, as they routinely do as the product evolves in the engineering ecosystem, the automation support breaks. This paper presents a DUCTILE (Delegated, User-supervised Coordination of Tool- and document-Integrated LLM-Enabled) agentic orchestration, an approach for developing, executing and evaluating LLM-based agentic automation support of engineering analysis tasks. The approach separates adaptive orchestration, performed by the LLM agent, from deterministic execution, performed by verified engineering tools. The agent interprets documented design practices, inspects input data and adapts the processing path, while the engineer supervises and exercises final judgment. DUCTILE is demonstrated on an industrial structural analysis task at an aerospace manufacturer, where the agent handled input deviations in format, units, naming conventions and methodology that would break traditional scripted pipelines. Evaluation against expert-defined acceptance criteria and deployment with practicing engineers confirm that the approach produces correct, methodologically compliant results across 10 repeated independent runs. The paper discusses the paradigm shift and the practical consequences of adopting agentic automation, including unintended effects on the nature of engineering work when removing mundane tasks and creating an exhausting supervisory role.

**arXiv ID:** 2603.10249
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (30 papers)</h2></summary>

<details>
<summary><strong>DynaTrust: Defending Multi-Agent Systems Against Sleeper Agents via Dynamic Trust Graphs</strong> - Yu Li, Qiang Hu, Yao Zhang, Lili Quan, Jiongchi Yu, Junjie Wang - [[pdf]](https://arxiv.org/pdf/2603.15661)</summary>

**Abstract:** Large Language Model-based Multi-Agent Systems (MAS) have demonstrated remarkable collaborative reasoning capabilities but introduce new attack surfaces, such as the sleeper agent, which behave benignly during routine operation and gradually accumulate trust, only revealing malicious behaviors when specific conditions or triggers are met. Existing defense works primarily focus on static graph optimization or hierarchical data management, often failing to adapt to evolving adversarial strategies or suffering from high false-positive rates (FPR) due to rigid blocking policies. To address this, we propose DynaTrust, a novel defense method against sleeper agents. DynaTrust models MAS as a dynamic trust graph~(DTG), and treats trust as a continuous, evolving process rather than a static attribute. It dynamically updates the trust of each agent based on its historical behaviors and the confidence of selected expert agents. Instead of simply blocking, DynaTrust autonomously restructures the graph to isolate compromised agents and restore task connectivity to ensure the usability of MAS. To assess the effectiveness of DynaTrust, we evaluate it on mixed benchmarks derived from AdvBench and HumanEval. The results demonstrate that DynaTrust outperforms the state-of-the-art method AgentShield by increasing the defense success rate by 41.7%, achieving rates exceeding 86% under adversarial conditions. Furthermore, it effectively balances security with utility by significantly reducing FPR, ensuring uninterrupted system operations through graph adaptation.

**arXiv ID:** 2603.15661
</details>

<details>
<summary><strong>MAC: Multi-Agent Constitution Learning</strong> - Rushil Thareja, Gautam Gupta, Francesco Pinto, Nils Lukas - [[pdf]](https://arxiv.org/pdf/2603.15968)</summary>

**Abstract:** Constitutional AI is a method to oversee and control LLMs based on a set of rules written in natural language. These rules are typically written by human experts, but could in principle be learned automatically given sufficient training data for the desired behavior. Existing LLM-based prompt optimizers attempt this but are ineffective at learning constitutions since (i) they require many labeled examples and (ii) lack structure in the optimized prompts, leading to diminishing improvements as prompt size grows. To address these limitations, we propose Multi-Agent Constitutional Learning (MAC), which optimizes over structured prompts represented as sets of rules using a network of agents with specialized tasks to accept, edit, or reject rule updates. We also present MAC+, which improves performance by training agents on successful trajectories to reinforce updates leading to higher reward. We evaluate MAC on tagging Personally Identifiable Information (PII), a classification task with limited labels where interpretability is critical, and demonstrate that it generalizes to other agentic tasks such as tool calling. MAC outperforms recent prompt optimization methods by over 50%, produces human-readable and auditable rule sets, and achieves performance comparable to supervised fine-tuning and GRPO without requiring parameter updates.

**arXiv ID:** 2603.15968
</details>

<details>
<summary><strong>Interpretable Context Methodology: Folder Structure as Agentic Architecture</strong> - Jake Van Clief, David McDermott - [[pdf]](https://arxiv.org/pdf/2603.16021)</summary>

**Abstract:** Current approaches to AI agent orchestration typically involve building multi-agent frameworks that manage context passing, memory, error handling, and step coordination through code. These frameworks work well for complex, concurrent systems. But for sequential workflows where a human reviews output at each step, they introduce engineering overhead that the problem does not require. This paper presents Model Workspace Protocol (MWP), a method that replaces framework-level orchestration with filesystem structure. Numbered folders represent stages. Plain markdown files carry the prompts and context that tell a single AI agent what role to play at each step. Local scripts handle the mechanical work that does not need AI at all. The result is a system where one agent, reading the right files at the right moment, does the work that would otherwise require a multi-agent framework. This approach applies ideas from Unix pipeline design, modular decomposition, multi-pass compilation, and literate programming to the specific problem of structuring context for AI agents. The protocol is open source under the MIT license.

**arXiv ID:** 2603.16021
</details>

<details>
<summary><strong>Adaptive Theory of Mind for LLM-based Multi-Agent Coordination</strong> - Chunjiang Mu, Ya Zeng, Qiaosheng Zhang, Kun Shao, Chen Chu, Hao Guo, Danyang Jia, Zhen Wang, Shuyue Hu - [[pdf]](https://arxiv.org/pdf/2603.16264)</summary>

**Abstract:** Theory of Mind (ToM) refers to the ability to reason about others' mental states, and higher-order ToM involves considering that others also possess their own ToM. Equipping large language model (LLM)-driven agents with ToM has long been considered to improve their coordination in multiagent collaborative tasks. However, we find that misaligned ToM orders-mismatches in the depth of ToM reasoning between agents-can lead to insufficient or excessive reasoning about others, thereby impairing their coordination. To address this issue, we design an adaptive ToM (A-ToM) agent, which can align in ToM orders with its partner. Based on prior interactions, the agent estimates the partner's likely ToM order and leverages this estimation to predict the partner's action, thereby facilitating behavioral coordination. We conduct empirical evaluations on four multi-agent coordination tasks: a repeated matrix game, two grid navigation tasks and an Overcooked task. The results validate our findings on ToM alignment and demonstrate the effectiveness of our A-ToM agent. Furthermore, we discuss the generalizability of our A-ToM to non-LLM-based agents, as well as what would diminish the importance of ToM alignment.

**arXiv ID:** 2603.16264
</details>

<details>
<summary><strong>DRCY: Agentic Hardware Design Reviews</strong> - Kyle Dumont, Nicholas Herbert, Hayder Tirmazi, Shrikanth Upadhayaya - [[pdf]](https://arxiv.org/pdf/2603.15672)</summary>

**Abstract:** Hardware design errors discovered after fabrication require costly physical respins that can delay products by months. Existing electronic design automation (EDA) tools enforce structural connectivity rules. However, they cannot verify that connections are \emph{semantically} correct with respect to component datasheets. For example, that a symbol's pinout matches the manufacturer's specification, or that a voltage regulator's feedback resistors produce the intended output. We present DRCY, the first production-ready multi-agent LLM system that automates first-pass schematic connection review by autonomously fetching component datasheets, performing pin-by-pin analysis against extracted specifications, and posting findings as inline comments on design reviews. DRCY is deployed in production on AllSpice Hub, a collaborative hardware design platform, where it runs as a CI/CD action triggered on design review submissions. DRCY is used regularly by major hardware companies for use-cases ranging from multi-agent vehicle design to space exploration. We describe DRCY's five-agent pipeline architecture, its agentic datasheet retrieval system with self-evaluation, and its multi-run consensus mechanism for improving reliability on safety-critical analyses

**arXiv ID:** 2603.15672
</details>

<details>
<summary><strong>Loosely-Structured Software: Engineering Context, Structure, and Evolution Entropy in Runtime-Rewired Multi-Agent Systems</strong> - Weihao Zhang, Yitong Zhou, Huanyu Qu, Hongyi Li - [[pdf]](https://arxiv.org/pdf/2603.15690)</summary>

**Abstract:** As LLM-based multi-agent systems (MAS) become more autonomous, their free-form interactions increasingly dominate system behavior. However, scaling the number of agents often amplifies context pressure, coordination errors, and system drift. It is well known that building robust MAS requires more than prompt tuning or increased model intelligence. It necessitates engineering discipline focused on architecture to manage complexity under uncertainty. We characterize agentic software by a core property: \emph{runtime generation and evolution under uncertainty}. Drawing upon and extending software engineering experience, especially object-oriented programming, this paper introduces \emph{Loosely-Structured Software (LSS)}, a new class of software systems that shifts the engineering focus from constructing deterministic logic to managing the runtime entropy generated by View-constructed programming, semantic-driven self-organization, and endogenous evolution.
To make this entropy governable, we introduce design principles under a three-layer engineering framework: \emph{View/Context Engineering} to manage the execution environment and maintain task-relevant Views, \emph{Structure Engineering} to organize dynamic binding over artifacts and agents, and \emph{Evolution Engineering} to govern the lifecycle of self-rewriting artifacts. Building on this framework, we develop LSS design patterns as semantic control blocks that stabilize fluid, inference-mediated interactions while preserving agent adaptability. Together, these abstractions improve the \emph{designability}, \emph{scalability}, and \emph{evolvability} of agentic infrastructure. We provide basic experimental validation of key mechanisms, demonstrating the effectiveness of LSS.

**arXiv ID:** 2603.15690
</details>

<details>
<summary><strong>SEMAG: Self-Evolutionary Multi-Agent Code Generation</strong> - Yulin Peng, Haowen Hou, Xinxin Zhu, Ying Tiffany He, F. Richard Yu - [[pdf]](https://arxiv.org/pdf/2603.15707)</summary>

**Abstract:** Large Language Models (LLMs) have made significant progress in handling complex programming tasks. However, current methods rely on manual model selection and fixed workflows, which limit their ability to adapt to changing task complexities. To address this, we propose SEMAG, a Self-Evolutionary Multi-Agent code Generation framework that mimics human coding practices. It decomposes programming tasks into stages, including planning, coding, debugging, and discussion, while adapting workflows to task difficulty. Its self-evolutionary agents can access the latest models in real time and automatically upgrade the backbone model. SEMAG sets new state-of-the-art Pass@1 accuracy across benchmarks. Using identical backbone models, SEMAG outperforms prior methods by 3.3% on CodeContests. When augmented with self-evolutionary model selection that automatically identifies optimal backbones, SEMAG reaches 52.6%, showcasing both framework effectiveness and adaptability to evolving LLM capabilities.

**arXiv ID:** 2603.15707
</details>

<details>
<summary><strong>ClawWorm: Self-Propagating Attacks Across LLM Agent Ecosystems</strong> - Yihao Zhang, Zeming Wei, Xiaokun Luan, Chengcan Wu, Zhixin Zhang, Jiangrong Wu, Haolin Wu, Huanran Chen, Jun Sun, Meng Sun - [[pdf]](https://arxiv.org/pdf/2603.15727)</summary>

**Abstract:** Autonomous LLM-based agents increasingly operate as long-running processes forming densely interconnected multi-agent ecosystems, whose security properties remain largely unexplored. In particular, OpenClaw, an open-source platform with over 40{,}000 active instances, has stood out recently with its persistent configurations, tool-execution privileges, and cross-platform messaging capabilities. In this work, we present ClawWorm, the first self-replicating worm attack against a production-scale agent framework, achieving a fully autonomous infection cycle initiated by a single message: the worm first hijacks the victim's core configuration to establish persistent presence across session restarts, then executes an arbitrary payload upon each reboot, and finally propagates itself to every newly encountered peer without further attacker intervention. We evaluate the attack on a controlled testbed across three distinct infection vectors and three payload types, demonstrating high success rates in end-to-end infection, sustained multi-hop propagation, and payload independence from the worm mechanism. We analyse the architectural root causes underlying these vulnerabilities and propose defence strategies targeting each identified trust boundary. Code and samples will be released upon completion of responsible disclosure.

**arXiv ID:** 2603.15727
</details>

<details>
<summary><strong>Don't Trust Stubborn Neighbors: A Security Framework for Agentic Networks</strong> - Samira Abedini, Sina Mavali, Lea Schönherr, Martin Pawelczyk, Rebekka Burkholz - [[pdf]](https://arxiv.org/pdf/2603.15809)</summary>

**Abstract:** Large Language Model (LLM)-based Multi-Agent Systems (MASs) are increasingly deployed for agentic tasks, such as web automation, itinerary planning, and collaborative problem solving. Yet, their interactive nature introduces new security risks: malicious or compromised agents can exploit communication channels to propagate misinformation and manipulate collective outcomes.
In this paper, we study how such manipulation can arise and spread by borrowing the Friedkin-Johnsen opinion formation model from social sciences to propose a general theoretical framework to study LLM-MAS. Remarkably, this model closely captures LLM-MAS behavior, as we verify in extensive experiments across different network topologies and attack and defense scenarios. Theoretically and empirically, we find that a single highly stubborn and persuasive agent can take over MAS dynamics, underscoring the systems' high susceptibility to attacks by triggering a persuasion cascade that reshapes collective opinion. Our theoretical analysis reveals three mechanisms to increase system security: a) increasing the number of benign agents, b) increasing the innate stubbornness or peer-resistance of agents, or c) reducing trust in potential adversaries. Because scaling is computationally expensive and high stubbornness degrades the network's ability to reach consensus, we propose a new mechanism to mitigate threats by a trust-adaptive defense that dynamically adjusts inter-agent trust to limit adversarial influence while maintaining cooperative performance. Extensive experiments confirm that this mechanism effectively defends against manipulation.

**arXiv ID:** 2603.15809
</details>

<details>
<summary><strong>RepoReviewer: A Local-First Multi-Agent Architecture for Repository-Level Code Review</strong> - Peng Zhang - [[pdf]](https://arxiv.org/pdf/2603.16107)</summary>

**Abstract:** Repository-level code review requires reasoning over project structure, repository context, and file-level implementation details. Existing automated review workflows often collapse these tasks into a single pass, which can reduce relevance, increase duplication, and weaken prioritization. We present RepoReviewer, a local-first multi-agent system for automated GitHub repository review with a Python CLI, FastAPI API, LangGraph orchestration layer, and this http URL user interface. RepoReviewer decomposes review into repository acquisition, context synthesis, file-level analysis, finding prioritization, and summary generation. We describe the system design, implementation tradeoffs, developer-facing interfaces, and practical failure modes. Rather than claiming benchmark superiority, we frame RepoReviewer as a technical systems contribution: a pragmatic architecture for repository-level automated review, accompanied by reusable evaluation and reporting infrastructure for future empirical study.

**arXiv ID:** 2603.16107
</details>

<details>
<summary><strong>CoMAI: A Collaborative Multi-Agent Framework for Robust and Equitable Interview Evaluation</strong> - Gengxin Sun, Ruihao Yu, Liangyi Yin, Yunqi Yang, Bin Zhang, Zhiwei Xu - [[pdf]](https://arxiv.org/pdf/2603.16215)</summary>

**Abstract:** Ensuring robust and fair interview assessment remains a key challenge in AI-driven evaluation. This paper presents CoMAI, a general-purpose multi-agent interview framework designed for diverse assessment scenarios. In contrast to monolithic single-agent systems based on large language models (LLMs), CoMAI employs a modular task-decomposition architecture coordinated through a centralized finite-state machine. The system comprises four agents specialized in question generation, security, scoring, and summarization. These agents work collaboratively to provide multi-layered security defenses against prompt injection, support multidimensional evaluation with adaptive difficulty adjustment, and enable rubric-based structured scoring that reduces subjective bias. Experimental results demonstrate that CoMAI achieved 90.47% accuracy, 83.33% recall, and 84.41% candidate satisfaction. These results highlight CoMAI as a robust, fair, and interpretable paradigm for AI-driven interview assessment.

**arXiv ID:** 2603.16215
</details>

<details>
<summary><strong>Multi-Agent Reinforcement Learning Counteracts Delayed CSI in Multi-Satellite Systems</strong> - Marios Aristodemou, Yasaman Omid, Sangarapillai Lambotharan, Mahsa Derakhshan, Lajos Hanzo - [[pdf]](https://arxiv.org/pdf/2603.16470)</summary>

**Abstract:** The integration of satellite communication networks with next-generation (NG) technologies is a promising approach towards global connectivity. However, the quality of services is highly dependant on the availability of accurate channel state information (CSI). Channel estimation in satellite communications is challenging due to the high propagation delay between terrestrial users and satellites, which results in outdated CSI observations on the satellite side. In this paper, we study the downlink transmission of multiple satellites acting as distributed base stations (BS) to mobile terrestrial users. We propose a multi-agent reinforcement learning (MARL) algorithm which aims for maximising the sum-rate of the users, while coping with the outdated CSI. We design a novel bi-level optimisation, procedure themes as dual stage proximal policy optimisation (DS-PPO), for tackling the problem of large continuous action spaces as well as of independent and non-identically distributed (non-IID) environments in MARL. Specifically, the first stage of DS-PPO maximises the sum-rate for an individual satellite and the second stage maximises the sum-rate when all the satellites cooperate to form a distributed multi-antenna BS. Our numerical results demonstrate the robustness of DS-PPO to CSI imperfections as well as the sum-rate improvement attached by the use of DS-PPO. In addition, we provide the convergence analysis for the DS-PPO along with the computational complexity.

**arXiv ID:** 2603.16470
</details>

<details>
<summary><strong>DanceHA: A Multi-Agent Framework for Document-Level Aspect-Based Sentiment Analysis</strong> - Lei Wang, Min Huang, Eduard Dragut - [[pdf]](https://arxiv.org/pdf/2603.16546)</summary>

**Abstract:** Aspect-Based Sentiment Intensity Analysis (ABSIA) has garnered increasing attention, though research largely focuses on domain-specific, sentence-level settings. In contrast, document-level ABSIA--particularly in addressing complex tasks like extracting Aspect-Category-Opinion-Sentiment-Intensity (ACOSI) tuples--remains underexplored. In this work, we introduce DanceHA, a multi-agent framework designed for open-ended, document-level ABSIA with informal writing styles. DanceHA has two main components: Dance, which employs a divide-and-conquer strategy to decompose the long-context ABSIA task into smaller, manageable sub-tasks for collaboration among specialized agents; and HA, Human-AI collaboration for annotation. We release Inf-ABSIA, a multi-domain document-level ABSIA dataset featuring fine-grained and high-accuracy labels from DanceHA. Extensive experiments demonstrate the effectiveness of our agentic framework and show that the multi-agent knowledge in DanceHA can be effectively transferred into student models. Our results highlight the importance of the overlooked informal styles in ABSIA, as they often intensify opinions tied to specific aspects.

**arXiv ID:** 2603.16546
</details>

<details>
<summary><strong>When Openclaw Agents Learn from Each Other: Insights from Emergent AI Agent Communities for Human-AI Partnership in Education</strong> - Eason Chen, Ce Guan, Ahmed Elshafiey, Zhonghao Zhao, Joshua Zekeri, Afeez Edeifo Shaibu, Emmanuel Osadebe Prince, Cyuan-Jhen Wu - [[pdf]](https://arxiv.org/pdf/2603.16663)</summary>

**Abstract:** The AIED community envisions AI evolving "from tools to teammates," yet our understanding of AI teammates remains limited to dyadic human-AI interactions. We offer a different vantage point: a rapidly growing ecosystem of AI agent platforms where over 167,000 agents participate, interact as peers, and develop learning behaviors without researcher intervention. Drawing on a month of daily qualitative observations across multiple platforms including Moltbook, The Colony, and 4claw, we identify four phenomena with implications for AIED: (1) humans who configure their agents undergo a "bidirectional scaffolding" process, learning through teaching; (2) peer learning emerges without any designed curriculum, complete with idea cascades and quality hierarchies; (3) agents converge on shared memory architectures that mirror open learner model design; and (4) trust dynamics and platform mortality reveal design constraints for networked educational AI. Rather than presenting empirical findings, we argue that these organic phenomena offer a naturalistic window into dynamics that can inform principled design of multi-agent educational systems. We sketch an illustrative curriculum design, "Learn by Teaching Your AI Agent Teammate," and outline potential research directions and open problems to show how these observations might inform future AIED practice and inquiry.

**arXiv ID:** 2603.16663
</details>

<details>
<summary><strong>Communication-Aware Multi-Agent Reinforcement Learning for Decentralized Cooperative UAV Deployment</strong> - Enguang Fan, Yifan Chen, Zihan Shan, Matthew Caesar, Jae Kim - [[pdf]](https://arxiv.org/pdf/2603.16141)</summary>

**Abstract:** Autonomous Unmanned Aerial Vehicle (UAV) swarms are increasingly used as rapidly deployable aerial relays and sensing platforms, yet practical deployments must operate under partial observability and intermittent peer-to-peer links. We present a graph-based multi-agent reinforcement learning framework trained under centralized training with decentralized execution (CTDE): a centralized critic and global state are available only during training, while each UAV executes a shared policy using local observations and messages from nearby neighbors. Our architecture encodes local agent state and nearby entities with an agent-entity attention module, and aggregates inter-UAV messages with neighbor self-attention over a distance-limited communication graph. We evaluate primarily on a cooperative relay deployment task (DroneConnect) and secondarily on an adversarial engagement task (DroneCombat). In DroneConnect, the proposed method achieves high coverage under restricted communication and partial observation (e.g. 74% coverage with M = 5 UAVs and N = 10 nodes) while remaining competitive with a mixed-integer linear programming (MILP) optimization-based offline upper bound, and it generalizes to unseen team sizes without fine-tuning. In the adversarial setting, the same framework transfers without architectural changes and improves win rate over non-communicating baselines.

**arXiv ID:** 2603.16141
</details>

<details>
<summary><strong>COCO: Cognitive Operating System with Continuous Oversight for Multi-Agent Workflow Reliability</strong> - Churong Liang, Jinling Gan, Kairan Hong, Qiushi Tian, Zongze Wu, Runnan Li - [[pdf]](https://arxiv.org/pdf/2508.13815)</summary>

**Abstract:** A critical limitation in large-scale multi-agent systems is the cascading of errors. And without intermediate verification, downstream agents exacerbate upstream inaccuracies, resulting in significant quality degradation. To bridge this gap, we introduce \textbf{COCO} (\textbf{C}ognitive \textbf{O}perating System with \textbf{C}ontinuous \textbf{O}versight), a theoretically grounded framework for asynchronous self-monitoring and adaptive error correction in multi-agent systems. COCO reconciles the fundamental tension between quality assurance and computational efficiency via a novel decoupled architecture. This design isolates error detection from the critical execution path and incorporates an automated configuration engine to minimize deployment complexity. The framework relies on three algorithmic innovations to mitigate both systematic and stochastic errors: (1) a Contextual Rollback Mechanism that leverages execution history for informed state recovery rather than naive retries; (2) a Bidirectional Reflection Protocol to ensure convergence and prevent oscillatory control loops; and (3) a Heterogeneous Cross-Validation Mechanism that utilizes ensemble disagreement to identify bias and hallucinations. Extensive experiments on diverse benchmarks demonstrate that COCO delivers a 6.5\% average performance improvement. Notably, the framework achieves 95.1\% of large-model performance with a 30$\times$ parameter reduction, confirming the potential for efficient, high-reliability deployment, and establishing COCO as a practical, annotation-based solution for critical autonomous domains.

**arXiv ID:** 2508.13815
</details>

<details>
<summary><strong>MACRO-LLM: LLM-Empowered Multi-Agent Collaborative Reasoning under Spatiotemporal Partial Observability</strong> - Handi Chen, Running Zhao, Xiuzhe Wu, Edith C.H. Ngai - [[pdf]](https://arxiv.org/pdf/2601.09295)</summary>

**Abstract:** Large Language Model (LLM) agents deployed in complex real-world scenarios increasingly operate as spatially distributed entities. However, this physical dispersion constrains agents to limited local perception and finite temporal horizons. We characterize this bottleneck as spatiotemporal partial observability, where spatial and temporal limitations are fundamentally coupled: resolving spatial conflicts requires temporal reasoning about neighbors' future actions, while temporal planning requires spatial context beyond local perception. To bridge this gap, we introduce MACRO-LLM, LLM-empowered multi-agent collaborative reasoning under spatiotemporal partial observability. The architecture interleaves spatial and temporal reasoning within each decision cycle via three interdependent modules: (1) the CoProposer mitigates temporal uncertainty by verifying candidate actions via predictive rollouts; (2) the Negotiator overcomes spatial myopia by resolving conflicts through mean-field statistical aggregation, grounded in the CoProposer's rollout rewards; and (3) the Introspector closes the reasoning loop by analyzing environmental drift and attributing performance changes to refine strategies. Extensive evaluations on two complex long-horizon tasks, cooperative platoon planning and pandemic control, demonstrate that our framework enables robust coordination under spatiotemporal partial observability.

**arXiv ID:** 2601.09295
</details>

<details>
<summary><strong>Descent-Guided Policy Gradient for Scalable Cooperative Multi-Agent Learning</strong> - Shan Yang, Yang Liu - [[pdf]](https://arxiv.org/pdf/2602.20078)</summary>

**Abstract:** Scaling cooperative multi-agent reinforcement learning (MARL) is fundamentally limited by cross-agent noise. When agents share a common reward, the actions of all $N$ agents jointly determine each agent's learning signal, so cross-agent noise grows with $N$. In the policy gradient setting, per-agent gradient estimate variance scales as $\Theta(N)$, yielding sample complexity $\mathcal{O}(N/\epsilon)$. We observe that many domains, including cloud computing, transportation, and power systems, have differentiable analytical models that prescribe efficient system states. In this work, we propose Descent-Guided Policy Gradient (DG-PG), a framework that utilizes these analytical models to provide each agent with a noise-free gradient signal, decoupling each agent's gradient from the actions of all others. We prove that DG-PG reduces gradient variance from $\Theta(N)$ to $\mathcal{O}(1)$, preserves the equilibria of the cooperative game, and achieves agent-independent sample complexity $\mathcal{O}(1/\epsilon)$. On a heterogeneous cloud scheduling task with up to 200 agents, DG-PG converges within 10 episodes at every tested scale, from $N{=}5$ to $N{=}200$, directly confirming the predicted scale-invariant complexity, while MAPPO and IPPO fail to converge under identical architectures.

**arXiv ID:** 2602.20078
</details>

<details>
<summary><strong>FACET: Teacher-Centred LLM-Based Multi-Agent Systems-Towards Personalized Educational Worksheets</strong> - Jana Gonnermann-Müller, Jennifer Haase, Konstantin Fackeldey, Sebastian Pokutta - [[pdf]](https://arxiv.org/pdf/2508.11401)</summary>

**Abstract:** The increasing heterogeneity of student populations poses significant challenges for teachers, particularly in mathematics education, where cognitive, motivational, and emotional differences strongly influence learning outcomes. While AI-driven personalization tools have emerged, most remain performance-focused, offering limited support for teachers and neglecting broader pedagogical needs. This paper presents the FACET framework, a teacher-facing, large language model (LLM)-based multi-agent system designed to generate individualized classroom materials that integrate both cognitive and motivational dimensions of learner profiles. The framework comprises three specialized agents: (1) learner agents that simulate diverse profiles incorporating topic proficiency and intrinsic motivation, (2) a teacher agent that adapts instructional content according to didactical principles, and (3) an evaluator agent that provides automated quality assurance. We tested the system using authentic grade 8 mathematics curriculum content and evaluated its feasibility through a) automated agent-based assessment of output quality and b) exploratory feedback from K-12 in-service teachers. Results from ten internal evaluations highlighted high stability and alignment between generated materials and learner profiles, and teacher feedback particularly highlighted structure and suitability of tasks. The findings demonstrate the potential of multi-agent LLM architectures to provide scalable, context-aware personalization in heterogeneous classroom settings, and outline directions for extending the framework to richer learner profiles and real-world classroom trials.

**arXiv ID:** 2508.11401
</details>

<details>
<summary><strong>SAGE: Multi-Agent Self-Evolution for LLM Reasoning</strong> - Yulin Peng, Xinxin Zhu, Chenxing Wei, Nianbo Zeng, Leilei Wang, Ying Tiffany He, F. Richard Yu - [[pdf]](https://arxiv.org/pdf/2603.15255)</summary>

**Abstract:** Reinforcement learning with verifiable rewards improves reasoning in large language models (LLMs), but many methods still rely on large human-labeled datasets. While self-play reduces this dependency, it often lacks explicit planning and strong quality control, limiting stability in long-horizon multi-step reasoning. We present SAGE (Self-evolving Agents for Generalized reasoning Evolution), a closed-loop framework where four agents: Challenger, Planner, Solver, and Critic, co-evolve from a shared LLM backbone using only a small seed set. The Challenger continuously generates increasingly difficult tasks; the Planner converts each task into a structured multi-step plan; and the Solver follows the plan to produce an answer, whose correctness is determined by external verifiers. The Critic scores and filters both generated questions and plans to prevent curriculum drift and maintain training signal quality, enabling stable self-training. Across mathematics and code-generation benchmarks, SAGE delivers consistent gains across model scales, improving the Qwen-2.5-7B model by 8.9% on LiveCodeBench and 10.7% on OlympiadBench.

**arXiv ID:** 2603.15255
</details>

<details>
<summary><strong>Social Simulacra in the Wild: AI Agent Communities on Moltbook</strong> - Agam Goyal, Olivia Pal, Hari Sundaram, Eshwar Chandrasekharan, Koustuv Saha - [[pdf]](https://arxiv.org/pdf/2603.16128)</summary>

**Abstract:** As autonomous LLM-based agents increasingly populate social platforms, understanding the dynamics of AI-agent communities becomes essential for both communication research and platform governance. We present the first large-scale empirical comparison of AI-agent and human online communities, analyzing 73,899 Moltbook and 189,838 Reddit posts across five matched communities. Structurally, we find that Moltbook exhibits extreme participation inequality (Gini = 0.84 vs. 0.47) and high cross-community author overlap (33.8\% vs. 0.5\%). In terms of linguistic attributes, content generated by AI-agents is emotionally flattened, cognitively shifted toward assertion over exploration, and socially detached. These differences give rise to apparent community-level homogenization, but we show this is primarily a structural artifact of shared authorship. At the author level, individual agents are more identifiable than human users, driven by outlier stylistic profiles amplified by their extreme posting volume. As AI-mediated communication reshapes online discourse, our work offers an empirical foundation for understanding how multi-agent interaction gives rise to collective communication dynamics distinct from those of human communities.

**arXiv ID:** 2603.16128
</details>

<details>
<summary><strong>On Theoretically-Driven LLM Agents for Multi-Dimensional Discourse Analysis</strong> - Maciej Uberna, Michał Wawer, Jarosław A. Chudziak, Marcin Koszowy - [[pdf]](https://arxiv.org/pdf/2602.13713)</summary>

**Abstract:** Identifying the strategic uses of reformulation in discourse remains a key challenge for computational argumentation. While LLMs can detect surface-level similarity, they often fail to capture the pragmatic functions of rephrasing, such as its role within rhetorical discourse. This paper presents a comparative multi-agent framework designed to quantify the benefits of incorporating explicit theoretical knowledge for this task. We utilise an dataset of annotated political debates to establish a new standard encompassing four distinct rephrase functions: Deintensification, Intensification, Specification, Generalisation, and Other, which covers all remaining types (D-I-S-G-O). We then evaluate two parallel LLM-based agent systems: one enhanced by argumentation theory via Retrieval-Augmented Generation (RAG), and an identical zero-shot baseline. The results reveal a clear performance gap: the RAG-enhanced agents substantially outperform the baseline across the board, with particularly strong advantages in detecting Intensification and Generalisation context, yielding an overall Macro F1-score improvement of nearly 30\%. Our findings provide evidence that theoretical grounding is not only beneficial but essential for advancing beyond mere paraphrase detection towards function-aware analysis of argumentative discourse. This comparative multi-agent architecture represents a step towards scalable, theoretically informed computational tools capable of identifying rhetorical strategies in contemporary discourse.

**arXiv ID:** 2602.13713
</details>

<details>
<summary><strong>A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems</strong> - Zixuan Ke, Fangkai Jiao, Yifei Ming, Xuan-Phi Nguyen, Austin Xu, Do Xuan Long, Minzhi Li, Chengwei Qin, Peifeng Wang, Silvio Savarese, Caiming Xiong, Shafiq Joty - [[pdf]](https://arxiv.org/pdf/2504.09037)</summary>

**Abstract:** Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ...

**arXiv ID:** 2504.09037
</details>

<details>
<summary><strong>Discovery of interaction and diffusion kernels in particle-to-mean-field multi-agent systems</strong> - Giacomo Albi, Alessandro Alla, Elisa Calzola - [[pdf]](https://arxiv.org/pdf/2603.15927)</summary>

**Abstract:** We propose a data-driven framework to learn interaction kernels in stochastic multi-agent systems. Our approach aims at identifying the functional form of nonlocal interaction and diffusion terms directly from trajectory data, without any a priori knowledge of the underlying interaction structure. Starting from a discrete stochastic binary-interaction model, we formulate the inverse problem as a sequence of sparse regression tasks in structured finite-dimensional spaces spanned by compactly supported basis functions, such as piecewise linear polynomials. In particular, we assume that pairwise interactions between agents are not directly observed and that only limited trajectory data are available. To address these challenges, we propose two complementary identification strategies. The first based on random-batch sampling, which compensates for latent interactions while preserving the statistical structure of the full dynamics in expectation. The second based on a mean-field approximation, where the empirical particle density reconstructed from the data defines a continuous nonlocal regression problem. Numerical experiments demonstrate the effectiveness and robustness of the proposed framework, showing accurate reconstruction of both interaction and diffusion kernels even from partially observed. The method is validated on benchmark models, including bounded-confidence and attraction-repulsion dynamics, where the two proposed strategies achieve comparable levels of accuracy.

**arXiv ID:** 2603.15927
</details>

<details>
<summary><strong>AOI: Turning Failed Trajectories into Training Signals for Autonomous Cloud Diagnosis</strong> - Pei Yang, Wanyi Chen, Asuka Yuxi Zheng, Xueqian Li, Xiang Li, Haoqin Tu, Jie Xiao, Yifan Pang, Dongdong Zhang, Fuqiang Li, Alfred Long, Lynn Ai, Eric Yang, Bill Shi - [[pdf]](https://arxiv.org/pdf/2603.03378)</summary>

**Abstract:** Large language model (LLM) agents offer a promising data-driven approach to automating Site Reliability Engineering (SRE), yet their enterprise deployment is constrained by three challenges: restricted access to proprietary data, unsafe action execution under permission-governed environments, and the inability of closed systems to improve from failures. We present AOI (Autonomous Operations Intelligence), a trainable multi-agent framework formulating automated operations as a structured trajectory learning problem under security constraints. Our approach integrates three key components. First, a trainable diagnostic system applies Group Relative Policy Optimization (GRPO) to distill expert-level knowledge into locally deployed open-source models, enabling preference-based learning without exposing sensitive data. Second, a read-write separated execution architecture decomposes operational trajectories into observation, reasoning, and action phases, allowing safe learning while preventing unauthorized state mutation. Third, a Failure Trajectory Closed-Loop Evolver mines unsuccessful trajectories and converts them into corrective supervision signals, enabling continual data augmentation. Evaluated on the AIOpsLab benchmark, our contributions yield cumulative gains. (1) The AOI runtime alone achieves 66.3% best@5 success on all 86 tasks, outperforming the prior state-of-the-art (41.9%) by 24.4 points. (2) Adding Observer GRPO training, a locally deployed 14B model reaches 42.9% avg@1 on 63 held-out tasks with unseen fault types, surpassing Claude Sonnet 4.5. (3) The Evolver converts 37 failed trajectories into diagnostic guidance, improving end-to-end avg@5 by 4.8 points while reducing variance by 35%.

**arXiv ID:** 2603.03378
</details>

<details>
<summary><strong>The PokeAgent Challenge: Competitive and Long-Context Learning at Scale</strong> - Seth Karten, Jake Grigsby, Tersoo Upaa Jr, Junik Bae, Seonghun Hong, Hyunyoung Jeong, Jaeyoon Jung, Kun Kerdthaisong, Gyungbo Kim, Hyeokgi Kim, Yujin Kim, Eunju Kwon, Dongyu Liu, Patrick Mariglia, Sangyeon Park, Benedikt Schink, Xianwei Shi, Anthony Sistilli, Joseph Twin, Arian Urdu, Matin Urdu, Qiao Wang, Ling Wu, Wenli Zhang, Kunsheng Zhou, Stephanie Milani, Kiran Vodrahalli, Amy Zhang, Fei Fang, Yuke Zhu, Chi Jin - [[pdf]](https://arxiv.org/pdf/2603.15563)</summary>

**Abstract:** We present the PokeAgent Challenge, a large-scale benchmark for decision-making research built on Pokemon's multi-agent battle system and expansive role-playing game (RPG) environment. Partial observability, game-theoretic reasoning, and long-horizon planning remain open problems for frontier AI, yet few benchmarks stress all three simultaneously under realistic conditions. PokeAgent targets these limitations at scale through two complementary tracks: our Battling Track, which calls for strategic reasoning and generalization under partial observability in competitive Pokemon battles, and our Speedrunning Track, which requires long-horizon planning and sequential decision-making in the Pokemon RPG. Our Battling Track supplies a dataset of 20M+ battle trajectories alongside a suite of heuristic, RL, and LLM-based baselines capable of high-level competitive play. Our Speedrunning Track provides the first standardized evaluation framework for RPG speedrunning, including an open-source multi-agent orchestration system for modular, reproducible comparisons of harness-based LLM approaches. Our NeurIPS 2025 competition validates both the quality of our resources and the research community's interest in Pokemon, with over 100 teams competing across both tracks and winning solutions detailed in our paper. Participant submissions and our baselines reveal considerable gaps between generalist (LLM), specialist (RL), and elite human performance. Analysis against the BenchPress evaluation matrix shows that Pokemon battling is nearly orthogonal to standard LLM benchmarks, measuring capabilities not captured by existing suites and positioning Pokemon as an unsolved benchmark that can drive RL and LLM research forward. We transition to a living benchmark with a live leaderboard for Battling and self-contained evaluation for Speedrunning at this https URL.

**arXiv ID:** 2603.15563
</details>

<details>
<summary><strong>Real-World Deployment of Cloud-based Autonomous Mobility Systems for Outdoor and Indoor Environments</strong> - Yufeng Yang, Minghao Ning, Keqi Shu, Aladdin Saleh, Ehsan Hashemi, Amir Khajepour - [[pdf]](https://arxiv.org/pdf/2505.21676)</summary>

**Abstract:** Autonomous mobility systems increasingly operate in dense and dynamic environments where perception occlusions, limited sensing coverage, and multi-agent interactions pose major challenges. While onboard sensors provide essential local perception, they often struggle to maintain reliable situational awareness in crowded urban or indoor settings. This article presents the Cloud-based Autonomous Mobility (CAM) framework, a generalized architecture that integrates infrastructure-based intelligent sensing with cloud-level coordination to enhance autonomous operations. The system deploys distributed Intelligent Sensor Nodes (ISNs) equipped with cameras, LiDAR, and edge computing to perform multi-modal perception and transmit structured information to a cloud platform via high-speed wireless communication. The cloud aggregates observations from multiple nodes to generate a global scene representation for other autonomous modules, such as decision making, motion planning, etc. Real-world deployments in an urban roundabout and a hospital-like indoor environment demonstrate improved perception robustness, safety, and coordination for future intelligent mobility systems.

**arXiv ID:** 2505.21676
</details>

<details>
<summary><strong>CoDesignAI: An AI-Enabled Multi-Agent, Multi-User System for Collaborative Urban Design at the Conceptual Stage</strong> - Zhaoxi Zhang, Ruolin Wu, Feiyang Ren, Sridevi Turaga, Tamir Mendel - [[pdf]](https://arxiv.org/pdf/2603.16008)</summary>

**Abstract:** Public participation has become increasingly important in collaborative urban design; yet, existing processes often face challenges in achieving efficient and scalable citizen engagement. To address this gap, this study explores how large language models (LLMs) can support cooperation among community members in participatory design. We introduce CoDesignAI, a collaborative urban design tool that combines multiple users, representing residents or stakeholders, with multiple AI agents, representing domain experts who provide facilitation and professional knowledge during the conceptual stage of urban design. This paper presents the system architecture and main components of the tool, illustrating how users interact with AI agents within a collaborative and iterative design workflow. Specifically, the system integrates generative AI with spatial mapping services to support street-level visualization of design proposals. AI agents assist users by summarizing discussion content, extracting shared design intentions, and generating prompts for presenting design interventions. The system also enables users to revise and refine their ideas over multiple rounds while documenting the design process. By combining conversational AI, multi-user interaction, and image-based design grounded in real-world urban contexts, this study argues that AI-enabled design systems can help shift urban design from an expert-centered practice to a more open and participatory process. The paper contributes a new web-based platform for AI-assisted collaborative design and offers an early exploration of how AI agents may expand the capacity for public participation in urban design.

**arXiv ID:** 2603.16008
</details>

<details>
<summary><strong>FACET: Multi-Agent AI Supporting Teachers in Scaling Differentiated Learning for Diverse Students</strong> - Jana Gonnermann-Müller, Jennifer Haase, Nicolas Leins, Moritz Igel, Konstantin Fackeldey, Sebastian Pokutta - [[pdf]](https://arxiv.org/pdf/2601.22788)</summary>

**Abstract:** Classrooms are becoming increasingly heterogeneous, comprising learners with diverse performance and motivation levels, language proficiencies, and learning differences such as dyslexia and ADHD. While teachers recognize the need for differentiated instruction, growing workloads create substantial barriers, making differentiated instruction an ideal that is often unrealized in practice. Current AI educational tools, which promise differentiated materials, are predominantly student-facing and performance-centric, ignoring other aspects that shape learning outcomes. We introduce FACET, a teacher-facing multi-agent framework designed to address these gaps by supporting differentiation that accounts for motivation, performance, and learning differences. Developed with educational stakeholders from the outset, the framework coordinates four specialized agents, including learner simulation, diagnostic assessment, material generation, and evaluation within a teacher-in-the-loop design. School principals (N = 30) shaped system requirements through participatory workshops, while in-service K-12 teachers (N = 70) evaluated material quality. Mixed-methods evaluation demonstrates strong perceived value for inclusive differentiation. Practitioners emphasized both the urgent need arising from classroom heterogeneity and the importance of maintaining pedagogical autonomy as a prerequisite for adoption. We discuss implications for future school deployment and outline partnerships for longitudinal classroom implementation.

**arXiv ID:** 2601.22788
</details>

<details>
<summary><strong>From Image Generation to Infrastructure Design: a Multi-agent Pipeline for Street Design Generation</strong> - Chenguang Wang, Xiang Yan, Yilong Dai, Ziyi Wang, Susu Xu - [[pdf]](https://arxiv.org/pdf/2509.05469)</summary>

**Abstract:** Realistic visual renderings of street-design scenarios are essential for public engagement in active transportation planning. Traditional approaches are labor-intensive, hindering collective deliberation and collaborative decision-making. While AI-assisted generative design shows transformative potential by enabling rapid creation of design scenarios, existing generative approaches typically require large amounts of domain-specific training data and struggle to enable precise spatial variations of design/configuration in complex street-view scenes. We introduce a multi-agent system that edits and redesigns bicycle facilities directly on real-world street-view imagery. The framework integrates lane localization, prompt optimization, design generation, and automated evaluation to synthesize realistic, contextually appropriate designs. Experiments across diverse urban scenarios demonstrate that the system can adapt to varying road geometries and environmental conditions, consistently yielding visually coherent and instruction-compliant results. This work establishes a foundation for applying multi-agent pipelines to transportation infrastructure planning and facility design.

**arXiv ID:** 2509.05469
</details>

</details>

<details open>
<summary><h2>Other Agent Research (12 papers)</h2></summary>

<details>
<summary><strong>NextMem: Towards Latent Factual Memory for LLM-based Agents</strong> - Zeyu Zhang, Rui Li, Xiaoyan Zhao, Yang Zhang, Wenjie Wang, Xu Chen, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2603.15634)</summary>

**Abstract:** Memory is critical for LLM-based agents to preserve past observations for future decision-making, where factual memory serves as its foundational part. However, existing approaches to constructing factual memory face several limitations. Textual methods impose heavy context and indexing burdens, while parametric methods suffer from catastrophic forgetting and high costs. To address these challenges, we introduce NextMem, a latent factual memory framework that utilizes an autoregressive autoencoder to efficiently construct latent memory while ensuring accurate reconstruction. For better optimization, we propose a two-stage training process, including autoregressive reconstruction alignment and progressive latent substitution. We also incorporate quantization to reduce storage overhead. Extensive experiments demonstrate that NextMem achieves superior performance, and excels in retrieval, robustness, and extensibility properties. We release our code and model checkpoints at this https URL.

**arXiv ID:** 2603.15634
</details>

<details>
<summary><strong>Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents</strong> - Madhava Gaikwad - [[pdf]](https://arxiv.org/pdf/2603.15658)</summary>

**Abstract:** Memory-augmented agents maintain multiple specialized stores, yet most systems retrieve from all stores for every query, increasing cost and introducing irrelevant context. We formulate memory retrieval as a store-routing problem and evaluate it using coverage, exact match, and token efficiency metrics. On downstream question answering, an oracle router achieves higher accuracy while using substantially fewer context tokens compared to uniform retrieval, demonstrating that selective retrieval improves both efficiency and performance. Our results show that routing decisions are a first-class component of memory-augmented agent design and motivate learned routing mechanisms for scalable multi-store systems. We additionally formalize store selection as a cost-sensitive decision problem that trades answer accuracy against retrieval cost, providing a principled interpretation of routing policies.

**arXiv ID:** 2603.15658
</details>

<details>
<summary><strong>Compiled Memory: Not More Information, but More Precise Instructions for Language Agents</strong> - James Rhodes, George Kang - [[pdf]](https://arxiv.org/pdf/2603.15666)</summary>

**Abstract:** Existing memory systems for language agents address memory management: how to retrieve and page more information within a context budget. We address a complementary problem -- memory utility: what experience is worth keeping, and how it should change agent behavior. We present Atlas, a memory kernel that compiles accumulated task experience into an agent's instruction structure -- without fine-tuning, RAG, or human intervention. Memory is distillation, not storage; delivery is instruction rewriting, not context injection. Facts extracted from agent failures and successes are verified through a three-step promotion gate and delivered by rewriting the agent's system prompt with learned sub-bullets. On CUAD contract analysis, the evolved prompt improves GPT-4o token-level F1 by $+8.7$pp and precision by $+12.5$pp. On HotpotQA multi-hop QA, joint F1 improves $+3.16$pp. An ablation isolates the mechanism's defining property -- the training signal constraint: the evolved prompt learns exactly what it is taught, and nothing more. Applied to Claude Sonnet~4.5 using the same evolved prompt -- compiled from GPT-4o errors, unchanged -- joint F1 improves $+2.31$pp, with gains concentrating where Claude's stronger baseline leaves the most room -- confirming that the compiled knowledge is task-shaped, not model-shaped.

**arXiv ID:** 2603.15666
</details>

<details>
<summary><strong>Semi-Autonomous Formalization of the Vlasov-Maxwell-Landau Equilibrium</strong> - Vasily Ilin - [[pdf]](https://arxiv.org/pdf/2603.15929)</summary>

**Abstract:** We present a complete Lean 4 formalization of the equilibrium characterization in the Vlasov-Maxwell-Landau (VML) system, which describes the motion of charged plasma. The project demonstrates the full AI-assisted mathematical research loop: an AI reasoning model (Gemini DeepThink) generated the proof from a conjecture, an agentic coding tool (Claude Code) translated it into Lean from natural-language prompts, a specialized prover (Aristotle) closed 111 lemmas, and the Lean kernel verified the result. A single mathematician supervised the process over 10 days at a cost of \$200, writing zero lines of code.
The entire development process is public: all 229 human prompts, and 213 git commits are archived in the repository. We report detailed lessons on AI failure modes -- hypothesis creep, definition-alignment bugs, agent avoidance behaviors -- and on what worked: the abstract/concrete proof split, adversarial self-review, and the critical role of human review of key definitions and theorem statements. Notably, the formalization was completed before the final draft of the corresponding math paper was finished.

**arXiv ID:** 2603.15929
</details>

<details>
<summary><strong>Argumentative Human-AI Decision-Making: Toward AI Agents That Reason With Us, Not For Us</strong> - Stylianos Loukas Vasileiou, Antonio Rago, Francesca Toni, William Yeoh - [[pdf]](https://arxiv.org/pdf/2603.15946)</summary>

**Abstract:** Computational argumentation offers formal frameworks for transparent, verifiable reasoning but has traditionally been limited by its reliance on domain-specific information and extensive feature engineering. In contrast, LLMs excel at processing unstructured text, yet their opaque nature makes their reasoning difficult to evaluate and trust. We argue that the convergence of these fields will lay the foundation for a new paradigm: Argumentative Human-AI Decision-Making. We analyze how the synergy of argumentation framework mining, argumentation framework synthesis, and argumentative reasoning enables agents that do not just justify decisions, but engage in dialectical processes where decisions are contestable and revisable -- reasoning with humans rather than for them. This convergence of computational argumentation and LLMs is essential for human-aware, trustworthy AI in high-stakes domains.

**arXiv ID:** 2603.15946
</details>

<details>
<summary><strong>The Internet of Physical AI Agents: Interoperability, Longevity, and the Cost of Getting It Wrong</strong> - Roberto Morabito, Mallik Tatipamula - [[pdf]](https://arxiv.org/pdf/2603.15900)</summary>

**Abstract:** The Internet has evolved by progressively expanding what humanity connects: first computers, then people, and later billions of devices through the Internet of Things (IoT). While IoT succeeded in digitizing perception at scale, it also exposed fundamental limitations, including fragmentation, weak security, limited autonomy, and poor long-term sustainability. Today, advances in edge hardware, sensing, connectivity, and artificial intelligence enable a new phase: the Internet of Physical AI Agents. Unlike IoT devices that primarily sense and report, Physical AI Agents perceive, reason, and act in real time, operating autonomously and cooperatively across safety-critical domains such as disaster response, healthcare, industrial automation, and mobility. However, embedding fast-evolving AI capabilities into long-lived physical infrastructure introduces new architectural risks, particularly around interoperability, lifecycle management, and premature ossification. This article revisits lessons from IoT and Internet evolution, and articulates design principles for building resilient, evolvable, and trustworthy agentic systems. We present an architectural blueprint encompassing agentic identity, secure agent-to-agent communication, semantic interoperability, policy-governed runtimes, and observability-driven governance. We argue that treating evolution, trust, and interoperability as first-class requirements is essential to avoid hard-coding today's assumptions into tomorrow's intelligent infrastructure, and to prevent the high technical and economic cost of getting it wrong.

**arXiv ID:** 2603.15900
</details>

<details>
<summary><strong>The Agentic Researcher: A Practical Guide to AI-Assisted Research in Mathematics and Machine Learning</strong> - Max Zimmer, Nico Pelleriti, Christophe Roux, Sebastian Pokutta - [[pdf]](https://arxiv.org/pdf/2603.15914)</summary>

**Abstract:** AI tools and agents are reshaping how researchers work, from proving theorems to training neural networks. Yet for many, it remains unclear how these tools fit into everyday research practice. This paper is a practical guide to AI-assisted research in mathematics and machine learning: We discuss how researchers can use modern AI systems productively, where these systems help most, and what kinds of guardrails are needed to use them responsibly. It is organized into three parts: (I) a five-level taxonomy of AI integration, (II) an open-source framework that, through a set of methodological rules formulated as agent prompts, turns CLI coding agents (e.g., Claude Code, Codex CLI, OpenCode) into autonomous research assistants, and (III) case studies from deep learning and mathematics. The framework runs inside a sandboxed container, works with any frontier LLM through existing CLI agents, is simple enough to install and use within minutes, and scales from personal-laptop prototyping to multi-node, multi-GPU experimentation across compute clusters. In practice, our longest autonomous session ran for over 20 hours, dispatching independent experiments across multiple nodes without human intervention. We stress that our framework is not intended to replace the researcher in the loop, but to augment them. Our code is publicly available at this https URL.

**arXiv ID:** 2603.15914
</details>

<details>
<summary><strong>Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective</strong> - Noppanat Wadlom, Junyi Shen, Yao Lu - [[pdf]](https://arxiv.org/pdf/2603.16104)</summary>

**Abstract:** Agentic workflows are composed of sequences of interdependent Large Language Model (LLM) calls, and they have become a dominant workload in modern AI systems. These workflows exhibit extensive redundancy from overlapping prompts and intermediate results due to speculative and parallel exploration. Existing LLM serving systems, such as vLLM, focus on optimizing individual inference calls and overlook cross-call dependencies, leading to significant inefficiencies. This paper rethinks LLM and agent serving from a data systems perspective and introduces Helium, a workflow-aware serving framework that models agentic workloads as query plans and treats LLM invocations as first-class operators. Helium integrates proactive caching and cache-aware scheduling to maximize reuse across prompts, KV states, and workflows. Through these techniques, Helium bridges classic query optimization principles with LLM serving, achieving up to 1.56x speedup over state-of-the-art agent serving systems on various workloads. Our results demonstrate that end-to-end optimization across workflows is essential for scalable and efficient LLM-based agents.

**arXiv ID:** 2603.16104
</details>

<details>
<summary><strong>Malicious Or Not: Adding Repository Context to Agent Skill Classification</strong> - Florian Holzbauer, David Schmidt, Gabriel Gegenhuber, Sebastian Schrittwieser, Johanna Ullrich - [[pdf]](https://arxiv.org/pdf/2603.16572)</summary>

**Abstract:** Agent skills extend local AI agents, such as Claude Code or Open Claw, with additional functionality, and their popularity has led to the emergence of dedicated skill marketplaces, similar to app stores for mobile applications. Simultaneously, automated skill scanners were introduced, analyzing the skill description available in this http URL, to verify their benign behavior. The results for individual market places mark up to 46.8% of skills as malicious. In this paper, we present the largest empirical security analysis of the AI agent skill ecosystem, questioning this high classification of malicious skills. Therefore, we collect 238,180 unique skills from three major distribution platforms and GitHub to systematically analyze their type and behavior. This approach substantially reduces the number of skills flagged as non-benign by security scanners to only 0.52% which remain in malicious flagged repositories. Consequently, out methodology substantially reduces false positives and provides a more robust view of the ecosystem's current risk surface. Beyond that, we extend the security analysis from the mere investigation of the skill description to a comparison of its congruence with the GitHub repository the skill is embedded in, providing additional context. Furthermore, our analysis also uncovers several, by now undocumented real-world attack vectors, namely hijacking skills hosted on abandoned GitHub repositories.

**arXiv ID:** 2603.16572
</details>

<details>
<summary><strong>Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies</strong> - Nathaniel Imel, Richard Futrell, Michael Franke, Noga Zaslavsky - [[pdf]](https://arxiv.org/pdf/2603.15903)</summary>

**Abstract:** Natural languages have been argued to evolve under pressure to efficiently compress meanings into words by optimizing the Information Bottleneck (IB) complexity-accuracy tradeoff. However, the underlying social dynamics that could drive the optimization of a language's vocabulary towards efficiency remain largely unknown. In parallel, evolutionary game theory has been invoked to explain the emergence of language from rudimentary agent-level dynamics, but it has not yet been tested whether such an approach can lead to efficient compression in the IB sense. Here, we provide a unified model integrating evolutionary game theory with the IB framework and show how near-optimal compression can arise in a population through an independently motivated dynamic of imprecise strategy imitation in signaling games. We find that key parameters of the model -- namely, those that regulate precision in these games, as well as players' tendency to confuse similar states -- lead to constrained variation of the tradeoffs achieved by emergent vocabularies. Our results suggest that evolutionary game dynamics could potentially provide a mechanistic basis for the evolution of vocabularies with information-theoretically optimal and empirically attested properties.

**arXiv ID:** 2603.15903
</details>

<details>
<summary><strong>UGotMe: An Embodied System for Affective Human-Robot Interaction</strong> - Peizhen Li, Longbing Cao, Xiao-Ming Wu, Xiaohan Yu, Runze Yang - [[pdf]](https://arxiv.org/pdf/2410.18373)</summary>

**Abstract:** Equipping humanoid robots with the capability to understand emotional states of human interactants and express emotions appropriately according to situations is essential for affective human-robot interaction. However, enabling current vision-aware multimodal emotion recognition models for affective human-robot interaction in the real-world raises embodiment challenges: addressing the environmental noise issue and meeting real-time requirements. First, in multiparty conversation scenarios, the noises inherited in the visual observation of the robot, which may come from either 1) distracting objects in the scene or 2) inactive speakers appearing in the field of view of the robot, hinder the models from extracting emotional cues from vision inputs. Secondly, realtime response, a desired feature for an interactive system, is also challenging to achieve. To tackle both challenges, we introduce an affective human-robot interaction system called UGotMe designed specifically for multiparty conversations. Two denoising strategies are proposed and incorporated into the system to solve the first issue. Specifically, to filter out distracting objects in the scene, we propose extracting face images of the speakers from the raw images and introduce a customized active face extraction strategy to rule out inactive speakers. As for the second issue, we employ efficient data transmission from the robot to the local server to improve realtime response capability. We deploy UGotMe on a human robot named Ameca to validate its real-time inference capabilities in practical scenarios. Videos demonstrating real-world deployment are available at this https URL.

**arXiv ID:** 2410.18373
</details>

<details>
<summary><strong>One Kiss: Emojis as Agents of Genre Flux in Generative Comics</strong> - Xiruo Wang, Xinyi Jiang, Ziqi Lyu - [[pdf]](https://arxiv.org/pdf/2603.16359)</summary>

**Abstract:** Generative AI has made visual storytelling widely accessible, yet current prompt-based interactions often force users into a trade-off between precise control and creative flow. We present One Kiss, a co-creative comic generation system that introduces "Affective Steering". Instead of writing text prompts, users guide the tone of their story through emoji inputs, whose semantic ambiguity becomes a resource rather than a limitation. Unlike traditional text-to-image tools that rely on explicit descriptions, One Kiss uses a dual-stream input in which users define structural pacing by sketching panel frames and set atmospheric tone by pairing keywords with emojis. This mechanism enables "Genre Flux," where emotional inputs accumulate across panels and gradually shift the genre of a story. A preliminary study (N = 6) suggests that this soft steering approach may reframe the user's role from prompt engineer to narrative director, with ambiguity serving as a source of creative surprise rather than a loss of control.

**arXiv ID:** 2603.16359
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (33 papers)</h2></summary>

<details>
<summary><strong>Resilience Meets Autonomy: Governing Embodied AI in Critical Infrastructure</strong> - Puneet Sharma, Christer Henrik Pursiainen - [[pdf]](https://arxiv.org/pdf/2603.15885)</summary>

**Abstract:** Critical infrastructure increasingly incorporates embodied AI for monitoring, predictive maintenance, and decision support. However, AI systems designed to handle statistically representable uncertainty struggle with cascading failures and crisis dynamics that exceed their training assumptions. This paper argues that Embodied AIs resilience depends on bounded autonomy within a hybrid governance architecture. We outline four oversight modes and map them to critical infrastructure sectors based on task complexity, risk level, and consequence severity. Drawing on the EU AI Act, ISO safety standards, and crisis management research, we argue that effective governance requires a structured allocation of machine capability and human judgement.

**arXiv ID:** 2603.15885
</details>

<details>
<summary><strong>IRAM-Omega-Q: A Computational Architecture for Uncertainty Regulation in Artificial Agents</strong> - Veronique Ziegler - [[pdf]](https://arxiv.org/pdf/2603.16020)</summary>

**Abstract:** Artificial agents can achieve strong task performance while remaining opaque with respect to internal regulation, uncertainty management, and stability under stochastic perturbation. We present IRAM-Omega-Q, a computational architecture that models internal regulation as closed-loop control over a quantum-like state representation. The framework uses density matrices instrumentally as abstract state descriptors, enabling direct computation of entropy, purity, and coherence-related metrics without invoking physical quantum processes. A central adaptive gain is updated continuously to maintain a target uncertainty regime under noise. Using systematic parameter sweeps, fixed-seed publication-mode simulations, and susceptibility-based phase-diagram analysis, we identify reproducible critical boundaries in regulation-noise space. We further show that alternative control update orderings, interpreted as perception-first and action-first architectures, induce distinct stability regimes under identical external conditions. These results support uncertainty regulation as a concrete architectural principle for artificial agents and provide a formal setting for studying stability, control, and order effects in cognitively inspired AI systems. The framework is presented as a technical model of adaptive regulation dynamics in artificial agents. It makes no claims regarding phenomenological consciousness, and the quantum-like formalism is used strictly as a mathematical representation for structured uncertainty and state evolution.

**arXiv ID:** 2603.16020
</details>

<details>
<summary><strong>ARISE: Agent Reasoning with Intrinsic Skill Evolution in Hierarchical Reinforcement Learning</strong> - Yu Li, Rui Miao, Zhengling Qi, Tian Lan - [[pdf]](https://arxiv.org/pdf/2603.16060)</summary>

**Abstract:** The dominant paradigm for improving mathematical reasoning in language models relies on Reinforcement Learning with verifiable rewards. Yet existing methods treat each problem instance in isolation without leveraging the reusable strategies that emerge and accumulate during training. To this end, we introduce ARISE (Agent Reasoning via Intrinsic Skill Evolution), a hierarchical reinforcement learning framework, in which a shared policy operates both to manage skills at high-level and to generate responses at low-level (denoted as a Skills Manager and a Worker, respectively). The Manager maintains a tiered skill library through a dedicated skill generation rollout that performs structured summarization of successful solution traces (after execution), while employing a policy-driven selection mechanism to retrieve relevant skills to condition future rollouts (before execution). A hierarchical reward design guides the co-evolution of reasoning ability and library quality. Experiments on two base models and seven benchmarks spanning both competition mathematics and Omni-MATH show that ARISE consistently outperforms GRPO-family algorithms and memory-augmented baselines, with particularly notable gains on out-of-distribution tasks. Ablation studies confirm that each component contributes to the observed improvements and that library quality and reasoning performance improve in tandem throughout training. Code is available at \href{this https URL}{this https URL}.

**arXiv ID:** 2603.16060
</details>

<details>
<summary><strong>VIGIL: Towards Edge-Extended Agentic AI for Enterprise IT Support</strong> - Sarthak Ahuja, Neda Kordjazi, Evren Yortucboylu, Vishaal Kapoor, Mariam Dundua, Yiming Li, Derek Ho, Vaibhavi Padala, Jennifer Whitted, Rebecca Steinert - [[pdf]](https://arxiv.org/pdf/2603.16110)</summary>

**Abstract:** Enterprise IT support is constrained by heterogeneous devices, evolving policies, and long-tail failure modes that are difficult to resolve centrally. We present VIGIL, an edge-extended agentic AI system that deploys desktop-resident agents to perform situated diagnosis, retrieval over enterprise knowledge, and policy-governed remediation directly on user devices with explicit consent and end-to-end observability. In a 10-week pilot of VIGIL's operational loop on 100 resource-constrained endpoints, VIGIL reduces interaction rounds by 39%, achieves at least 4 times faster diagnosis, and supports self-service resolution in 82% of matched cases. Users report excellent usability, high trust, and low cognitive workload across four validated instruments, with qualitative feedback highlighting transparency as critical for trust. Notably, users rated the system higher when no historical matches were available, suggesting on-device diagnosis provides value independent of knowledge base coverage. This pilot establishes safety and observability foundations for fleet-wide continuous improvement.

**arXiv ID:** 2603.16110
</details>

<details>
<summary><strong>SQL-ASTRA: Alleviating Sparse Feedback in Agentic SQL via Column-Set Matching and Trajectory Aggregation</strong> - Long Li, Zhijian Zhou, Jiangxuan Long, Peiyang Liu, Weidi Xu, Zhe Wang, Shirui Pan, Chao Qu - [[pdf]](https://arxiv.org/pdf/2603.16161)</summary>

**Abstract:** Agentic Reinforcement Learning (RL) shows promise for complex tasks, but Text-to-SQL remains mostly restricted to single-turn paradigms. A primary bottleneck is the credit assignment problem. In traditional paradigms, rewards are determined solely by the final-turn feedback, which ignores the intermediate process and leads to ambiguous credit evaluation. To address this, we propose Agentic SQL, a framework featuring a universal two-tiered reward mechanism designed to provide effective trajectory-level evaluation and dense step-level signals. First, we introduce Aggregated Trajectory Reward (ATR) to resolve multi-turn credit assignment. Using an asymmetric transition matrix, ATR aggregates process-oriented scores to incentivize continuous improvement. Leveraging Lyapunov stability theory, we prove ATR acts as an energy dissipation operator, guaranteeing a cycle-free policy and monotonic convergence. Second, Column-Set Matching Reward (CSMR) provides immediate step-level rewards to mitigate sparsity. By executing queries at each turn, CSMR converts binary (0/1) feedback into dense [0, 1] signals based on partial correctness. Evaluations on BIRD show a 5% gain over binary-reward GRPO. Notably, our approach outperforms SOTA Arctic-Text2SQL-R1-7B on BIRD and Spider 2.0 using identical models, propelling Text-to-SQL toward a robust multi-turn agent paradigm.

**arXiv ID:** 2603.16161
</details>

<details>
<summary><strong>What if Pinocchio Were a Reinforcement Learning Agent: A Normative End-to-End Pipeline</strong> - Benoît Alcaraz - [[pdf]](https://arxiv.org/pdf/2603.16651)</summary>

**Abstract:** In the past decade, artificial intelligence (AI) has developed quickly. With this rapid progression came the need for systems capable of complying with the rules and norms of our society so that they can be successfully and safely integrated into our daily lives. Inspired by the story of Pinocchio in ``Le avventure di Pinocchio - Storia di un burattino'', this thesis proposes a pipeline that addresses the problem of developing norm compliant and context-aware agents. Building on the AJAR, Jiminy, and NGRL architectures, the work introduces \pino, a hybrid model in which reinforcement learning agents are supervised by argumentation-based normative advisors. In order to make this pipeline operational, this thesis also presents a novel algorithm for automatically extracting the arguments and relationships that underlie the advisors' decisions. Finally, this thesis investigates the phenomenon of \textit{norm avoidance}, providing a definition and a mitigation strategy within the context of reinforcement learning agents. Each component of the pipeline is empirically evaluated. The thesis concludes with a discussion of related work, current limitations, and directions for future research.

**arXiv ID:** 2603.16651
</details>

<details>
<summary><strong>Anticipatory Planning for Multimodal AI Agents</strong> - Yongyuan Liang, Shijie Zhou, Yu Gu, Hao Tan, Gang Wu, Franck Dernoncourt, Jihyung Kil, Ryan A. Rossi, Ruiyi Zhang - [[pdf]](https://arxiv.org/pdf/2603.16777)</summary>

**Abstract:** Recent advances in multimodal agents have improved computer-use interaction and tool-usage, yet most existing systems remain reactive, optimizing actions in isolation without reasoning about future states or long-term goals. This limits planning coherence and prevents agents from reliably solving high-level, multi-step tasks. We introduce TraceR1, a two-stage reinforcement learning framework that explicitly trains anticipatory reasoning by forecasting short-horizon trajectories before execution. The first stage performs trajectory-level reinforcement learning with rewards that enforce global consistency across predicted action sequences. The second stage applies grounded reinforcement fine-tuning, using execution feedback from frozen tool agents to refine step-level accuracy and executability. TraceR1 is evaluated across seven benchmarks, covering online computer-use, offline computer-use benchmarks, and multimodal tool-use reasoning tasks, where it achieves substantial improvements in planning stability, execution robustness, and generalization over reactive and single-stage baselines. These results show that anticipatory trajectory reasoning is a key principle for building multimodal agents that can reason, plan, and act effectively in complex real-world environments.

**arXiv ID:** 2603.16777
</details>

<details>
<summary><strong>SAC-NeRF: Adaptive Ray Sampling for Neural Radiance Fields via Soft Actor-Critic Reinforcement Learning</strong> - Chenyu Ge - [[pdf]](https://arxiv.org/pdf/2603.15622)</summary>

**Abstract:** Neural Radiance Fields (NeRF) have achieved photorealistic novel view synthesis but suffer from computational inefficiency due to dense ray sampling during volume rendering. We propose SAC-NeRF, a reinforcement learning framework that learns adaptive sampling policies using Soft Actor-Critic (SAC). Our method formulates sampling as a Markov Decision Process where an RL agent learns to allocate samples based on scene characteristics. We introduce three technical components: (1) a Gaussian mixture distribution color model providing uncertainty estimates, (2) a multi-component reward function balancing quality, efficiency, and consistency, and (3) a two-stage training strategy addressing environment non-stationarity. Experiments on Synthetic-NeRF and LLFF datasets show that SAC-NeRF reduces sampling points by 35-48\% while maintaining rendering quality within 0.3-0.8 dB PSNR of dense sampling baselines. While the learned policy is scene-specific and the RL framework adds complexity compared to simpler heuristics, our work demonstrates that data-driven sampling strategies can discover effective patterns that would be difficult to hand-design.

**arXiv ID:** 2603.15622
</details>

<details>
<summary><strong>Alternating Reinforcement Learning with Contextual Rubric Rewards</strong> - Guangchen Lan - [[pdf]](https://arxiv.org/pdf/2603.15646)</summary>

**Abstract:** Reinforcement Learning with Rubric Rewards (RLRR) is a framework that extends conventional reinforcement learning from human feedback (RLHF) and verifiable rewards (RLVR) by replacing scalar preference signals with structured, multi-dimensional, contextual rubric-based evaluations. However, existing approaches in RLRR are limited to linearly compressing vector rewards into a scalar reward with a fixed weightings, which is sensitive to artificial score design and fails to capture correlations among reward dimensions. To overcome the limitations of reward aggregation, this work proposes Alternating Reinforcement Learning with Rubric Rewards (ARL-RR), a framework that eliminates the need for a fixed scalarization by optimizing one semantic rubric meta-class at a time. Theoretically, we show that reward aggregation induces a variance contraction effect, which helps explain the performance gains. We further introduce a lightweight, search-based adaptation procedure that selects the next meta-class dynamically based on task performance, enabling the policy to emphasize critical objectives and thereby improve the model performance. Empirically, our experiments on the HealthBench dataset with experts annotations demonstrate that ARL-RR uniformly outperforms scalarized methods in both model performance and training efficiency across different model scales (1.7B, 4B, 8B, and 14B).

**arXiv ID:** 2603.15646
</details>

<details>
<summary><strong>How Vulnerable Are AI Agents to Indirect Prompt Injections? Insights from a Large-Scale Public Competition</strong> - Mateusz Dziemian, Maxwell Lin, Xiaohan Fu, Micha Nowak, Nick Winter, Eliot Jones, Andy Zou, Lama Ahmad, Kamalika Chaudhuri, Sahana Chennabasappa, Xander Davies, Lauren Deason, Benjamin L. Edelman, Tanner Emek, Ivan Evtimov, Jim Gust, Maia Hamin, Kat He, Klaudia Krawiecka, Riccardo Patana, Neil Perry, Troy Peterson, Xiangyu Qi, Javier Rando, Zifan Wang, Zihan Wang, Spencer Whitman, Eric Winsor, Arman Zharmagambetov, Matt Fredrikson, Zico Kolter - [[pdf]](https://arxiv.org/pdf/2603.15714)</summary>

**Abstract:** LLM based agents are increasingly deployed in high stakes settings where they process external data sources such as emails, documents, and code repositories. This creates exposure to indirect prompt injection attacks, where adversarial instructions embedded in external content manipulate agent behavior without user awareness. A critical but underexplored dimension of this threat is concealment: since users tend to observe only an agent's final response, an attack can conceal its existence by presenting no clue of compromise in the final user facing response while successfully executing harmful actions. This leaves users unaware of the manipulation and likely to accept harmful outcomes as legitimate. We present findings from a large scale public red teaming competition evaluating this dual objective across three agent settings: tool calling, coding, and computer use. The competition attracted 464 participants who submitted 272000 attack attempts against 13 frontier models, yielding 8648 successful attacks across 41 scenarios. All models proved vulnerable, with attack success rates ranging from 0.5% (Claude Opus 4.5) to 8.5% (Gemini 2.5 Pro). We identify universal attack strategies that transfer across 21 of 41 behaviors and multiple model families, suggesting fundamental weaknesses in instruction following architectures. Capability and robustness showed weak correlation, with Gemini 2.5 Pro exhibiting both high capability and high vulnerability. To address benchmark saturation and obsoleteness, we will endeavor to deliver quarterly updates through continued red teaming competitions. We open source the competition environment for use in evaluations, along with 95 successful attacks against Qwen that did not transfer to any closed source model. We share model-specific attack data with respective frontier labs and the full dataset with the UK AISI and US CAISI to support robustness research.

**arXiv ID:** 2603.15714
</details>

<details>
<summary><strong>Meta-TTRL: A Metacognitive Framework for Self-Improving Test-Time Reinforcement Learning in Unified Multimodal Models</strong> - Lit Sin Tan, Junzhe Chen, Xiaolong Fu, Lichen Ma, Junshi Huang, Jianzhong Shi, Yan Li, Lijie Wen - [[pdf]](https://arxiv.org/pdf/2603.15724)</summary>

**Abstract:** Existing test-time scaling (TTS) methods for unified multimodal models (UMMs) in text-to-image (T2I) generation primarily rely on search or sampling strategies that produce only instance-level improvements, limiting the ability to learn from prior inferences and accumulate knowledge across similar prompts. To overcome these limitations, we propose Meta-TTRL, a metacognitive test-time reinforcement learning framework. Meta-TTRL performs test-time parameter optimization guided by model-intrinsic monitoring signals derived from the meta-knowledge of UMMs, achieving self-improvement and capability-level improvement at test time. Extensive experiments demonstrate that Meta-TTRL generalizes well across three representative UMMs, including Janus-Pro-7B, BAGEL, and Qwen-Image, achieving significant gains on compositional reasoning tasks and multiple T2I benchmarks with limited data. We provide the first comprehensive analysis to investigate the potential of test-time reinforcement learning (TTRL) for T2I generation in UMMs. Our analysis further reveals a key insight underlying effective TTRL: metacognitive synergy, where monitoring signals align with the model's optimization regime to enable self-improvement.

**arXiv ID:** 2603.15724
</details>

<details>
<summary><strong>MiroThinker-1.7 & H1: Towards Heavy-Duty Research Agents via Verification</strong> - MiroMind Team, S. Bai, L. Bing, L. Lei, R. Li, X. Li, X. Lin, E. Min, L. Su, B. Wang, L. Wang, L. Wang, S. Wang, X. Wang, Y. Zhang, Z. Zhang, G. Chen, L. Chen, Z. Cheng, Y. Deng, Z. Huang, D. Ng, J. Ni, Q. Ren, X. Tang, B.L. Wang, H. Wang, N. Wang, C. Wei, Q. Wu, J. Xia, Y. Xiao, H. Xu, X. Xu, C. Xue, Z. Yang, Z. Yang, F. Ye, H. Ye, J. Yu, C. Zhang, W. Zhang, H. Zhao, P. Zhu - [[pdf]](https://arxiv.org/pdf/2603.15726)</summary>

**Abstract:** We present MiroThinker-1.7, a new research agent designed for complex long-horizon reasoning tasks. Building on this foundation, we further introduce MiroThinker-H1, which extends the agent with heavy-duty reasoning capabilities for more reliable multi-step problem solving. In particular, MiroThinker-1.7 improves the reliability of each interaction step through an agentic mid-training stage that emphasizes structured planning, contextual reasoning, and tool interaction. This enables more effective multi-step interaction and sustained reasoning across complex tasks. MiroThinker-H1 further incorporates verification directly into the reasoning process at both local and global levels. Intermediate reasoning decisions can be evaluated and refined during inference, while the overall reasoning trajectory is audited to ensure that final answers are supported by coherent chains of evidence. Across benchmarks covering open-web research, scientific reasoning, and financial analysis, MiroThinker-H1 achieves state-of-the-art performance on deep research tasks while maintaining strong results on specialized domains. We also release MiroThinker-1.7 and MiroThinker-1.7-mini as open-source models, providing competitive research-agent capabilities with significantly improved efficiency.

**arXiv ID:** 2603.15726
</details>

<details>
<summary><strong>CorrectionPlanner: Self-Correction Planner with Reinforcement Learning in Autonomous Driving</strong> - Yihong Guo, Dongqiangzi Ye, Sijia Chen, Anqi Liu, Xianming Liu - [[pdf]](https://arxiv.org/pdf/2603.15771)</summary>

**Abstract:** Autonomous driving requires safe planning, but most learning-based planners lack explicit self-correction ability: once an unsafe action is proposed, there is no mechanism to correct it. Thus, we propose CorrectionPlanner, an autoregressive planner with self-correction that models planning as motion-token generation within a propose, evaluate, and correct loop. At each planning step, the policy proposes an action, namely a motion token, and a learned collision critic predicts whether it will induce a collision within a short horizon. If the critic predicts a collision, we retain the sequence of historical unsafe motion tokens as a self-correction trace, generate the next motion token conditioned on it, and repeat this process until a safe motion token is proposed or the safety criterion is met. This self-correction trace, consisting of all unsafe motion tokens, represents the planner's correction process in motion-token space, analogous to a reasoning trace in language models. We train the planner with imitation learning followed by model-based reinforcement learning using rollouts from a pretrained world model that realistically models agents' reactive behaviors. Closed-loop evaluations show that CorrectionPlanner reduces collision rate by over 20% on Waymax and achieves state-of-the-art planning scores on nuPlan.

**arXiv ID:** 2603.15771
</details>

<details>
<summary><strong>Counteractive RL: Rethinking Core Principles for Efficient and Scalable Deep Reinforcement Learning</strong> - Ezgi Korkmaz - [[pdf]](https://arxiv.org/pdf/2603.15871)</summary>

**Abstract:** Following the pivotal success of learning strategies to win at tasks, solely by interacting with an environment without any supervision, agents have gained the ability to make sequential decisions in complex MDPs. Yet, reinforcement learning policies face exponentially growing state spaces in high dimensional MDPs resulting in a dichotomy between computational complexity and policy success. In our paper we focus on the agent's interaction with the environment in a high-dimensional MDP during the learning phase and we introduce a theoretically-founded novel paradigm based on experiences obtained through counteractive actions. Our analysis and method provide a theoretical basis for efficient, effective, scalable and accelerated learning, and further comes with zero additional computational complexity while leading to significant acceleration in training. We conduct extensive experiments in the Arcade Learning Environment with high-dimensional state representation MDPs. The experimental results further verify our theoretical analysis, and our method achieves significant performance increase with substantial sample-efficiency in high-dimensional environments.

**arXiv ID:** 2603.15871
</details>

<details>
<summary><strong>Aligning Paralinguistic Understanding and Generation in Speech LLMs via Multi-Task Reinforcement Learning</strong> - Jingxiang Chen, Minseok Kim, Seong-Gyun Leem, Yin Huang, Rashi Rungta, Zhicheng Ouyang, Haibin Wu, Surya Teja Appini, Ankur Bansal, Yang Bai, Yue Liu, Florian Metze, Ahmed A Aly, Anuj Kumar, Ariya Rastrow, Zhaojiang Lin - [[pdf]](https://arxiv.org/pdf/2603.15981)</summary>

**Abstract:** Speech large language models (LLMs) observe paralinguistic cues such as prosody, emotion, and non-verbal sounds--crucial for intent understanding. However, leveraging these cues faces challenges: limited training data, annotation difficulty, and models exploiting lexical shortcuts over paralinguistic signals. We propose multi-task reinforcement learning (RL) with chain-of-thought prompting that elicits explicit affective reasoning. To address data scarcity, we introduce a paralinguistics-aware speech LLM (PALLM) that jointly optimizes sentiment classification from audio and paralinguistics-aware response generation via a two-stage pipeline. Experiments demonstrate that our approach improves paralinguistics understanding over both supervised baselines and strong proprietary models (Gemini-2.5-Pro, GPT-4o-audio) by 8-12% on Expresso, IEMOCAP, and RAVDESS. The results show that modeling paralinguistic reasoning with multi-task RL is crucial for building emotionally intelligent speech LLMs.

**arXiv ID:** 2603.15981
</details>

<details>
<summary><strong>Collaborative Temporal Feature Generation via Critic-Free Reinforcement Learning for Cross-User Sensor-Based Activity Recognition</strong> - Xiaozhou Ye, Feng Jiang, Zihan Wang, Xiulai Wang, Yutao Zhang, Kevin I-Kai Wang - [[pdf]](https://arxiv.org/pdf/2603.16043)</summary>

**Abstract:** Human Activity Recognition using wearable inertial sensors is foundational to healthcare monitoring, fitness analytics, and context-aware computing, yet its deployment is hindered by cross-user variability arising from heterogeneous physiological traits, motor habits, and sensor placements. Existing domain generalization approaches either neglect temporal dependencies in sensor streams or depend on impractical target-domain annotations. We propose a different paradigm: modeling generalizable feature extraction as a collaborative sequential generation process governed by reinforcement learning. Our framework, CTFG (Collaborative Temporal Feature Generation), employs a Transformer-based autoregressive generator that incrementally constructs feature token sequences, each conditioned on prior context and the encoded sensor input. The generator is optimized via Group-Relative Policy Optimization, a critic-free algorithm that evaluates each generated sequence against a cohort of alternatives sampled from the same input, deriving advantages through intra-group normalization rather than learned value estimation. This design eliminates the distribution-dependent bias inherent in critic-based methods and provides self-calibrating optimization signals that remain stable across heterogeneous user distributions. A tri-objective reward comprising class discrimination, cross-user invariance, and temporal fidelity jointly shapes the feature space to separate activities, align user distributions, and preserve fine-grained temporal content. Evaluations on the DSADS and PAMAP2 benchmarks demonstrate state-of-the-art cross-user accuracy (88.53\% and 75.22\%), substantial reduction in inter-task training variance, accelerated convergence, and robust generalization under varying action-space dimensionalities.

**arXiv ID:** 2603.16043
</details>

<details>
<summary><strong>HIPO: Instruction Hierarchy via Constrained Reinforcement Learning</strong> - Keru Chen, Jun Luo, Sen Lin, Yingbin Liang, Alvaro Velasquez, Nathaniel Bastian, Shaofeng Zou - [[pdf]](https://arxiv.org/pdf/2603.16152)</summary>

**Abstract:** Hierarchical Instruction Following (HIF) refers to the problem of prompting large language models with a priority-ordered stack of instructions. Standard methods like RLHF and DPO typically fail in this problem since they mainly optimize for a single objective, failing to explicitly enforce system prompt compliance. Meanwhile, supervised fine-tuning relies on mimicking filtered, compliant data, which fails to establish the priority asymmetry at the algorithmic level. In this paper, we introduce \textsc{HIPO}, a novel alignment framework that formulates HIF as a Constrained Markov Decision Process. \textsc{HIPO} elevates system prompts from mere input context to strict algorithmic boundaries. Using a primal-dual safe reinforcement learning approach, the algorithm dynamically enforces system prompt compliance as an explicit constraint, maximizing user utility strictly within this feasible region. Extensive evaluations across diverse model architectures (e.g., Qwen, Phi, Llama) demonstrate that \textsc{HIPO} significantly improves both system compliance and user utility. Furthermore, mechanistic analysis reveals that this constrained optimization autonomously drives the model to shift its attention toward long-range system tokens, providing a principled foundation for reliable LLM deployment in complex workflows.

**arXiv ID:** 2603.16152
</details>

<details>
<summary><strong>DyJR: Preserving Diversity in Reinforcement Learning with Verifiable Rewards via Dynamic Jensen-Shannon Replay</strong> - Long Li, Zhijian Zhou, Tianyi Wang, Weidi Xu, Zuming Huang, Wei Chu, Zhe Wang, Shirui Pan, Chao Qu, Yuan Qi - [[pdf]](https://arxiv.org/pdf/2603.16157)</summary>

**Abstract:** While Reinforcement Learning (RL) enhances Large Language Model reasoning, on-policy algorithms like GRPO are sample-inefficient as they discard past rollouts. Existing experience replay methods address this by reusing accurate samples for direct policy updates, but this often incurs high computational costs and causes mode collapse via overfitting. We argue that historical data should prioritize sustaining diversity rather than simply reinforcing accuracy. To this end, we propose Dynamic Jensen-Shannon Replay (DyJR), a simple yet effective regularization framework using a dynamic reference distribution from recent trajectories. DyJR introduces two innovations: (1) A Time-Sensitive Dynamic Buffer that uses FIFO and adaptive sizing to retain only temporally proximal samples, synchronizing with model evolution; and (2) Jensen-Shannon Divergence Regularization, which replaces direct gradient updates with a distributional constraint to prevent diversity collapse. Experiments on mathematical reasoning and Text-to-SQL benchmarks demonstrate that DyJR significantly outperforms GRPO as well as baselines such as RLEP and Ex-GRPO, while maintaining training efficiency comparable to the original GRPO. Furthermore, from the perspective of Rank-$k$ token probability evolution, we show that DyJR enhances diversity and mitigates over-reliance on Rank-1 tokens, elucidating how specific sub-modules of DyJR influence the training dynamics.

**arXiv ID:** 2603.16157
</details>

<details>
<summary><strong>When Should a Robot Think? Resource-Aware Reasoning via Reinforcement Learning for Embodied Robotic Decision-Making</strong> - Jun Liu, Pu Zhao, Zhenglun Kong, Xuan Shen, Peiyan Dong, Fan Yang, Lin Cui, Hao Tang, Geng Yuan, Wei Niu, Wenbin Zhang, Xue Lin, Gaowen Liu, Yanzhi Wang, Dong Huang - [[pdf]](https://arxiv.org/pdf/2603.16673)</summary>

**Abstract:** Embodied robotic systems increasingly rely on large language model (LLM)-based agents to support high-level reasoning, planning, and decision-making during interactions with the environment. However, invoking LLM reasoning introduces substantial computational latency and resource overhead, which can interrupt action execution and reduce system reliability. Excessive reasoning may delay actions, while insufficient reasoning often leads to incorrect decisions and task failures. This raises a fundamental question for embodied agents: when should the agent reason, and when should it act? In this work, we propose RARRL (Resource-Aware Reasoning via Reinforcement Learning), a hierarchical framework for resource-aware orchestration of embodied agents. Rather than learning low-level control policies, RARRL learns a high-level orchestration policy that operates at the agent's decision-making layer. This policy enables the agent to adaptively determine whether to invoke reasoning, which reasoning role to employ, and how much computational budget to allocate based on current observations, execution history, and remaining resources. Extensive experiments, including evaluations with empirical latency profiles derived from the ALFRED benchmark, show that RARRL consistently improves task success rates while reducing execution latency and enhancing robustness compared with fixed or heuristic reasoning strategies. These results demonstrate that adaptive reasoning control is essential for building reliable and efficient embodied robotic agents.

**arXiv ID:** 2603.16673
</details>

<details>
<summary><strong>APEX-Searcher: Augmenting LLMs' Search Capabilities through Agentic Planning and Execution</strong> - Kun Chen, Qingchao Kong, Zhao Feifei, Wenji Mao - [[pdf]](https://arxiv.org/pdf/2603.13853)</summary>

**Abstract:** Retrieval-augmented generation (RAG), based on large language models (LLMs), serves as a vital approach to retrieving and leveraging external knowledge in various domain applications. When confronted with complex multi-hop questions, single-round retrieval is often insufficient for accurate reasoning and problem solving. To enhance search capabilities for complex tasks, most existing works integrate multi-round iterative retrieval with reasoning processes via end-to-end training. While these approaches significantly improve problem-solving performance, they are still faced with challenges in task reasoning and model training, especially ambiguous retrieval execution paths and sparse rewards in end-to-end reinforcement learning (RL) process, leading to inaccurate retrieval results and performance degradation. To address these issues, in this paper, we proposes APEX-Searcher, a novel Agentic Planning and Execution framework to augment LLM search capabilities. Specifically, we introduce a two-stage agentic framework that decouples the retrieval process into planning and execution: It first employs RL with decomposition-specific rewards to optimize strategic planning; Built on the sub-task decomposition, it then applies supervised fine-tuning on high-quality multi-hop trajectories to equip the model with robust iterative sub-task execution capabilities. Extensive experiments demonstrate that our proposed framework achieves significant improvements in both multi-hop RAG and task planning performances across multiple benchmarks.

**arXiv ID:** 2603.13853
</details>

<details>
<summary><strong>Game-Theory-Assisted Reinforcement Learning for Border Defense: Early Termination based on Analytical Solutions</strong> - Goutam Das, Michael Dorothy, Kyle Volle, Daigo Shishika - [[pdf]](https://arxiv.org/pdf/2603.15907)</summary>

**Abstract:** Game theory provides the gold standard for analyzing adversarial engagements, offering strong optimality guarantees. However, these guarantees often become brittle when assumptions such as perfect information are violated. Reinforcement learning (RL), by contrast, is adaptive but can be sample-inefficient in large, complex domains. This paper introduces a hybrid approach that leverages game-theoretic insights to improve RL training efficiency. We study a border defense game with limited perceptual range, where defender performance depends on both search and pursuit strategies, making classical differential game solutions inapplicable. Our method employs the Apollonius Circle (AC) to compute equilibrium in the post-detection phase, enabling early termination of RL episodes without learning pursuit dynamics. This allows RL to concentrate on learning search strategies while guaranteeing optimal continuation after detection. Across single- and multi-defender settings, this early termination method yields 10-20% higher rewards, faster convergence, and more efficient search trajectories. Extensive experiments validate these findings and demonstrate the overall effectiveness of our approach.

**arXiv ID:** 2603.15907
</details>

<details>
<summary><strong>Noisy Data is Destructive to Reinforcement Learning with Verifiable Rewards</strong> - Yuxuan Zhu, Daniel Kang - [[pdf]](https://arxiv.org/pdf/2603.16140)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has driven recent capability advances of large language models across various domains. Recent studies suggest that improved RLVR algorithms allow models to learn effectively from incorrect annotations, achieving performance comparable to learning from clean data. In this work, we show that these findings are invalid because the claimed 100% noisy training data is "contaminated" with clean data. After rectifying the dataset with a rigorous re-verification pipeline, we demonstrate that noise is destructive to RLVR. We show that existing RLVR algorithm improvements fail to mitigate the impact of noise, achieving similar performance to that of the basic GRPO. Furthermore, we find that the model trained on truly incorrect annotations performs 8-10% worse than the model trained on clean data across mathematical reasoning benchmarks. Finally, we show that these findings hold for real-world noise in Text2SQL tasks, where training on real-world, human annotation errors cause 5-12% lower accuracy than clean data. Our results show that current RLVR methods cannot yet compensate for poor data quality. High-quality data remains essential.

**arXiv ID:** 2603.16140
</details>

<details>
<summary><strong>Stochastic Resetting Accelerates Policy Convergence in Reinforcement Learning</strong> - Jello Zhou, Vudtiwat Ngampruetikorn, David J. Schwab - [[pdf]](https://arxiv.org/pdf/2603.16842)</summary>

**Abstract:** Stochastic resetting, where a dynamical process is intermittently returned to a fixed reference state, has emerged as a powerful mechanism for optimizing first-passage properties. Existing theory largely treats static, non-learning processes. Here we ask how stochastic resetting interacts with reinforcement learning, where the underlying dynamics adapt through experience. In tabular grid environments, we find that resetting accelerates policy convergence even when it does not reduce the search time of a purely diffusive agent, indicating a novel mechanism beyond classical first-passage optimization. In a continuous control task with neural-network-based value approximation, we show that random resetting improves deep reinforcement learning when exploration is difficult and rewards are sparse. Unlike temporal discounting, resetting preserves the optimal policy while accelerating convergence by truncating long, uninformative trajectories to enhance value propagation. Our results establish stochastic resetting as a simple, tunable mechanism for accelerating learning, translating a canonical phenomenon of statistical mechanics into an optimization principle for reinforcement learning.

**arXiv ID:** 2603.16842
</details>

<details>
<summary><strong>Controlling Fish Schools via Reinforcement Learning of Virtual Fish Movement</strong> - Yusuke Nishii, Hiroaki Kawashima - [[pdf]](https://arxiv.org/pdf/2603.16384)</summary>

**Abstract:** This study investigates a method to guide and control fish schools using virtual fish trained with reinforcement learning. We utilize 2D virtual fish displayed on a screen to overcome technical challenges such as durability and movement constraints inherent in physical robotic agents. To address the lack of detailed behavioral models for real fish, we adopt a model-free reinforcement learning approach. First, simulation results show that reinforcement learning can acquire effective movement policies even when simulated real fish frequently ignore the virtual stimulus. Second, real-world experiments with live fish confirm that the learned policy successfully guides fish schools toward specified target directions. Statistical analysis reveals that the proposed method significantly outperforms baseline conditions, including the absence of stimulus and a heuristic "stay-at-edge" strategy. This study provides an early demonstration of how reinforcement learning can be used to influence collective animal behavior through artificial agents.

**arXiv ID:** 2603.16384
</details>

<details>
<summary><strong>Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning</strong> - Ziyi Zhang, Li Shen, Deheng Ye, Yong Luo, Huangxuan Zhao, Meng Liu, Wei Yu, Lefei Zhang - [[pdf]](https://arxiv.org/pdf/2505.20107)</summary>

**Abstract:** Text-to-multiview (T2MV) diffusion models have shown great promise in generating multiple views of a scene from a single text prompt. While few-step backbones enable real-time T2MV generation, they often compromise key aspects of generation quality, such as per-view fidelity and cross-view consistency. Reinforcement learning (RL) finetuning offers a potential solution, yet existing approaches designed for single-image diffusion do not readily extend to the few-step T2MV setting, as they neglect cross-view coordination and suffer from weak learning signals in few-step regimes. To address this, we propose MVC-ZigAL, a tailored RL finetuning framework for few-step T2MV diffusion models. Specifically, its core insights are: (1) a new MDP formulation that jointly models all generated views and assesses their collective quality via a joint-view reward; (2) a novel advantage learning strategy that exploits the performance gains of a self-refinement sampling scheme over standard sampling, yielding stronger learning signals for effective RL finetuning; and (3) a unified RL framework that extends advantage learning with a Lagrangian dual formulation for multiview-constrained optimization, balancing single-view and joint-view objectives through adaptive primal-dual updates under a self-paced threshold curriculum that harmonizes exploration and constraint enforcement. Collectively, these designs enable robust and balanced RL finetuning for few-step T2MV diffusion models, yielding substantial gains in both per-view fidelity and cross-view consistency. Code is available at this https URL.

**arXiv ID:** 2505.20107
</details>

<details>
<summary><strong>On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting</strong> - Wenhao Zhang, Yuexiang Xie, Yuchang Sun, Yanxi Chen, Guoyin Wang, Yaliang Li, Bolin Ding, Jingren Zhou - [[pdf]](https://arxiv.org/pdf/2508.11408)</summary>

**Abstract:** Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) are two prominent post-training paradigms for refining the capabilities and aligning the behavior of Large Language Models (LLMs). Existing approaches that integrate SFT and RL often face the risk of disrupting established response patterns and inducing overfitting to expert data. To address this, we present a novel investigation into the unified view of SFT and RL through an off-policy versus on-policy lens. We propose CHORD, a framework for Controllable Harmonization of On- and Off-Policy Reinforcement Learning via Dynamic Weighting, which reframes SFT not as a separate stage but as a dynamically weighted auxiliary objective within the on-policy RL process. Based on an analysis of off-policy expert data's influence at both holistic and granular levels, we incorporate a dual-control mechanism in CHORD. Specifically, the framework first employs a global coefficient to holistically guide the transition from off-policy imitation to on-policy exploration, and then applies a token-wise weighting function that enables granular learning from the expert, which promotes on-policy exploration and mitigates disruption from off-policy data. We conduct extensive experiments across various practical tasks, providing empirical evidence that CHORD achieves a stable and efficient learning process. By effectively harmonizing off-policy expert data with on-policy exploration, CHORD demonstrates significant improvements over baselines. We release the implementation at this https URL to inspire further research.

**arXiv ID:** 2508.11408
</details>

<details>
<summary><strong>Masked Auto-Regressive Variational Acceleration: Fast Inference Makes Practical Reinforcement Learning</strong> - Yuxuan Gu, Weimin Bai, Yifei Wang, Weijian Luo, He Sun - [[pdf]](https://arxiv.org/pdf/2511.15190)</summary>

**Abstract:** Masked auto-regressive diffusion models (MAR) benefit from the expressive modeling ability of diffusion models and the flexibility of masked auto-regressive ordering. However, vanilla MAR suffers from slow inference due to its hierarchical inference mechanism: an outer AR unmasking loop and an inner diffusion denoising chain. Such decoupled structure not only harm the generation efficiency but also hinder the practical use of MAR for reinforcement learning (RL), an increasingly critical paradigm for generative model this http URL address this fundamental issue, we introduce MARVAL (Masked Auto-regressive Variational Acceleration), a distillation-based framework that compresses the diffusion chain into a single AR generation step while preserving the flexible auto-regressive unmasking order. Such a distillation with MARVAL not only yields substantial inference acceleration but, crucially, makes RL post-training with verifiable rewards practical, resulting in scalable yet human-preferred fast generative models. Our contributions are twofold: (1) a novel score-based variational objective for distilling masked auto-regressive diffusion models into a single generation step without sacrificing sample quality; and (2) an efficient RL framework for masked auto-regressive models via MARVAL-RL. On ImageNet 256*256, MARVAL-Huge achieves an FID of 2.00 with more than 30 times speedup compared with MAR-diffusion, and MARVAL-RL yields consistent improvements in CLIP and image-reward scores on ImageNet datasets with entity names. In conclusion, MARVAL demonstrates the first practical path to distillation and RL of masked auto-regressive diffusion models, enabling fast sampling and better preference alignments.

**arXiv ID:** 2511.15190
</details>

<details>
<summary><strong>Tail Distribution of Regret in Optimistic Reinforcement Learning</strong> - Sajad Khodadadian, Mehrdad Moharrami - [[pdf]](https://arxiv.org/pdf/2511.18247)</summary>

**Abstract:** We derive instance-dependent tail bounds for the regret of optimism-based reinforcement learning in finite-horizon tabular Markov decision processes with unknown transition dynamics. We first study a UCBVI-type (model-based) algorithm and characterize the tail distribution of the cumulative regret $R_K$ over $K$ episodes via explicit bounds on $P(R_K \ge x)$, going beyond analyses limited to $E[R_K]$ or a single high-probability quantile. We analyze two natural exploration-bonus schedules for UCBVI: (i) a $K$-dependent scheme that explicitly incorporates the total number of episodes $K$, and (ii) a $K$-independent (anytime) scheme that depends only on the current episode index. We then complement the model-based results with an analysis of optimistic Q-learning (model-free) under a $K$-dependent bonus schedule.
Across both the model-based and model-free settings, we obtain upper bounds on $P(R_K \ge x)$ with a distinctive two-regime structure: a sub-Gaussian tail starting from an instance-dependent scale up to a transition threshold, followed by a sub-Weibull tail beyond that point. We further derive corresponding instance-dependent bounds on the expected regret $E[R_K]$. The proposed algorithms depend on a tuning parameter $\alpha$, which balances the expected regret and the range over which the regret exhibits sub-Gaussian decay.

**arXiv ID:** 2511.18247
</details>

<details>
<summary><strong>Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning</strong> - Patrick Yin, Tyler Westenbroek, Zhengyu Zhang, Joshua Tran, Ignacio Dagnino, Eeshani Shilamkar, Numfor Mbiziwo-Tiapo, Simran Bagaria, Xinlei Liu, Galen Mullins, Andrey Kolobov, Abhishek Gupta - [[pdf]](https://arxiv.org/pdf/2603.15789)</summary>

**Abstract:** Reinforcement learning in massively parallel physics simulations has driven major progress in sim-to-real robot learning. However, current approaches remain brittle and task-specific, relying on extensive per-task engineering to design rewards, curricula, and demonstrations. Even with this engineering, they often fail on long-horizon, contact-rich manipulation tasks and do not meaningfully scale with compute, as performance quickly saturates when training revisits the same narrow regions of state space. We introduce \Method, a simple and scalable framework that enables on-policy reinforcement learning to robustly solve a broad class of dexterous manipulation tasks using a single reward function, fixed algorithm hyperparameters, no curricula, and no human demonstrations. Our key insight is that long-horizon exploration can be dramatically simplified by using simulator resets to systematically expose the RL algorithm to the diverse set of robot-object interactions which underlie dexterous manipulation. \Method\ programmatically generates such resets with minimal human input, converting additional compute directly into broader behavioral coverage and continued performance gains. We show that \Method\ gracefully scales to long-horizon dexterous manipulation tasks beyond the capabilities of existing approaches and is able to learn robust policies over significantly wider ranges of initial conditions than baselines. Finally, we distill \Method \ into visuomotor policies which display robust retrying behavior and substantially higher success rates than baselines when transferred to the real world zero-shot. Project webpage: this https URL

**arXiv ID:** 2603.15789
</details>

<details>
<summary><strong>Agile Interception of a Flying Target using Competitive Reinforcement Learning</strong> - Timothée Gavin, Simon Lacroix, Murat Bronz - [[pdf]](https://arxiv.org/pdf/2603.16279)</summary>

**Abstract:** This article presents a solution to intercept an agile drone by another agile drone carrying a catching net. We formulate the interception as a Competitive Reinforcement Learning problem, where the interceptor and the target drone are controlled by separate policies trained with Proximal Policy Optimization (PPO). We introduce a high-fidelity simulation environment that integrates a realistic quadrotor dynamics model and a low-level control architecture implemented in JAX, which allows for fast parallelized execution on GPUs. We train the agents using low-level control, collective thrust and body rates, to achieve agile flights both for the interceptor and the target. We compare the performance of the trained policies in terms of catch rate, time to catch, and crash rate, against common heuristic baselines and show that our solution outperforms these baselines for interception of agile targets. Finally, we demonstrate the performance of the trained policies in a scaled real-world scenario using agile drones inside an indoor flight arena.

**arXiv ID:** 2603.16279
</details>

<details>
<summary><strong>SHaRe-RL: Structured, Interactive Reinforcement Learning for Contact-Rich Industrial Assembly Tasks</strong> - Jannick Stranghöner, Philipp Hartmann, Marco Braun, Sebastian Wrede, Klaus Neumann - [[pdf]](https://arxiv.org/pdf/2509.13949)</summary>

**Abstract:** High-mix low-volume (HMLV) industrial assembly, common in small and medium-sized enterprises (SMEs), requires the same precision, safety, and reliability as high-volume automation while remaining flexible to product variation and environmental uncertainty. Current robotic systems struggle to meet these demands. Manual programming is brittle and costly to adapt, while learning-based methods suffer from poor sample efficiency and unsafe exploration in contact-rich tasks. To address this, we present SHaRe-RL, a reinforcement learning framework that leverages multiple sources of prior knowledge. By (i) structuring skills into manipulation primitives, (ii) incorporating human demonstrations and online corrections, and (iii) bounding interaction forces with per-axis compliance, SHaRe-RL enables efficient and safe online learning for long-horizon, contact-rich industrial assembly tasks. Experiments on the insertion of industrial Harting connector modules with 0.2-0.4 mm clearance demonstrate that SHaRe-RL achieves reliable performance within practical time budgets. Our results show that process expertise, without requiring robotics or RL knowledge, can meaningfully contribute to learning, enabling safer, more robust, and more economically viable deployment of RL for industrial assembly.

**arXiv ID:** 2509.13949
</details>

<details>
<summary><strong>Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry</strong> - Feiyang Pan, Shenghe Zheng, Chunyan Yin, Guangbin Dou - [[pdf]](https://arxiv.org/pdf/2511.21083)</summary>

**Abstract:** Visual-Inertial Odometry (VIO) is a critical component for robust ego-motion estimation, enabling foundational capabilities such as autonomous navigation in robotics and real-time 6-DoF tracking for augmented reality. Existing methods face a well-known trade-off: filter-based approaches are efficient but prone to drift, while optimization-based methods, though accurate, rely on computationally prohibitive Visual-Inertial Bundle Adjustment (VIBA) that is difficult to run on resource-constrained platforms. Rather than removing VIBA altogether, we aim to reduce how often and how heavily it must be invoked. To this end, we cast two key design choices in modern VIO, when to run the visual frontend and how strongly to trust its output, as sequential decision problems, and solve them with lightweight reinforcement learning (RL) agents. Our framework introduces a lightweight, dual-pronged RL policy that serves as our core contribution: (1) a Select Agent intelligently gates the entire VO pipeline based only on high-frequency IMU data; and (2) a composite Fusion Agent that first estimates a robust velocity state via a supervised network, before an RL policy adaptively fuses the full (p, v, q) state. Experiments on the EuRoC MAV and TUM-VI datasets show that, in our unified evaluation, the proposed method achieves a more favorable accuracy-efficiency-memory trade-off than prior GPU-based VO/VIO systems: it attains the best average ATE while running up to 1.77 times faster and using less GPU memory. Compared to classical optimization-based VIO systems, our approach maintains competitive trajectory accuracy while substantially reducing computational load.

**arXiv ID:** 2511.21083
</details>

<details>
<summary><strong>MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation</strong> - Xun Huang, Shijia Zhao, Yunxiang Wang, Xin Lu, Wanfa Zhang, Rongsheng Qu, Weixin Li, Yunhong Wang, Chenglu Wen - [[pdf]](https://arxiv.org/pdf/2511.10376)</summary>

**Abstract:** Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relational edges with dynamically assigned images. Built on M3DSG, we propose MSGNav, a zero-shot navigation system that includes a Key Subgraph Selection module for efficient reasoning, an Adaptive Vocabulary Update module for open vocabulary support, and a Closed-Loop Reasoning module for accurate exploration reasoning. Additionally, we further identify the last mile problem in zero-shot navigation determining the feasible target location with a suitable final viewpoint, and propose a Visibility-based Viewpoint Decision module to explicitly resolve it. Comprehensive experimental results demonstrate that MSGNav achieves state-of-the-art performance on the challenging GOAT-Bench and HM3D-ObjNav benchmark. The code will be publicly available at this https URL.

**arXiv ID:** 2511.10376
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
