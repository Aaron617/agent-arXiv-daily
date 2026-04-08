# Agent arXiv Daily

**Last Updated:** 2026-04-08 03:30:42

**Total Papers:** 93

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (5 papers)</h2></summary>

<details>
<summary><strong>Claw-Eval: Toward Trustworthy Evaluation of Autonomous Agents</strong> - Bowen Ye, Rang Li, Qibin Yang, Yuanxin Liu, Linli Yao, Hanglong Lv, Zhihui Xie, Chenxin An, Lei Li, Lingpeng Kong, Qi Liu, Zhifang Sui, Tong Yang - [[pdf]](https://arxiv.org/pdf/2604.06132)</summary>

**Abstract:** Large language models are increasingly deployed as autonomous agents executing multi-step workflows in real-world software environments. However, existing agent benchmarks suffer from three critical limitations: (1) trajectory-opaque grading that checks only final outputs, (2) underspecified safety and robustness evaluation, and (3) narrow modality coverage and interaction paradigms. We introduce Claw-Eval, an end-to-end evaluation suite addressing all three gaps. It comprises 300 human-verified tasks spanning 9 categories across three groups (general service orchestration, multimodal perception and generation, and multi-turn professional dialogue). Every agent action is recorded through three independent evidence channels (execution traces, audit logs, and environment snapshots), enabling trajectory-aware grading over 2,159 fine-grained rubric items. The scoring protocol evaluates Completion, Safety, and Robustness, reporting Average Score, Pass@k, and Pass^k across three trials to distinguish genuine capability from lucky outcomes. Experiments on 14 frontier models reveal that: (1) trajectory-opaque evaluation is systematically unreliable, missing 44% of safety violations and 13% of robustness failures that our hybrid pipeline catches; (2) controlled error injection primarily degrades consistency rather than peak capability, with Pass^3 dropping up to 24% while Pass@3 remains stable; (3) multimodal performance varies sharply, with most models performing poorer on video than on document or image, and no single model dominating across all modalities. Beyond benchmarking, Claw-Eval highlights actionable directions for agent development, shedding light on what it takes to build agents that are not only capable but reliably deployable.

**arXiv ID:** 2604.06132
</details>

<details>
<summary><strong>URSA: The Universal Research and Scientific Agent</strong> - Michael Grosskopf, Nathan Debardeleben, Russell Bent, Rahul Somasundaram, Isaac Michaud, Arthur Lui, Alexius Wadell, Warren D. Graham, Golo A Wimmer, Sachin Shivakumar, Joan Vendrell Gallart, Harsha Nagarajan, Earl Lawrence - [[pdf]](https://arxiv.org/pdf/2506.22653)</summary>

**Abstract:** Large language models (LLMs) have moved far beyond their initial form as simple chatbots, now carrying out complex reasoning, planning, writing, coding, and research tasks. These skills overlap significantly with those that human scientists use day-to-day to solve complex problems that drive the cutting edge of research. Using LLMs in \quotes{agentic} AI has the potential to revolutionize modern science and remove bottlenecks to progress. In this work, we present URSA, a scientific agent ecosystem for accelerating research tasks. URSA consists of a set of modular agents and tools, including coupling to advanced physics simulation codes, that can be combined to address scientific problems of varied complexity and impact. This work highlights the architecture of URSA, as well as examples that highlight the potential of the system.

**arXiv ID:** 2506.22653
</details>

<details>
<summary><strong>Dialogue Act Patterns in GenAI-Mediated L2 Oral Practice: A Sequential Analysis of Learner-Chatbot Interactions</strong> - Liqun He, Shijun, Chen, Mutlu Cukurova, Manolis Mavrikis - [[pdf]](https://arxiv.org/pdf/2604.05702)</summary>

**Abstract:** While generative AI (GenAI) voice chatbots offer scalable opportunities for second language (L2) oral practice, the interactional processes related to learners' gains remain underexplored. This study investigates dialogue act (DA) patterns in interactions between Grade 9 Chinese English as a foreign language (EFL) learners and a GenAI voice chatbot over a 10-week intervention. Seventy sessions from 12 students were annotated by human coders using a pedagogy-informed coding scheme, yielding 6,957 coded DAs. DA distributions and sequential patterns were compared between high- and low-progress sessions. At the DA level, high-progress sessions showed more learner-initiated questions, whereas low-progress sessions exhibited higher rates of clarification-seeking, indicating greater comprehension difficulty. At the sequential level, high-progress sessions were characterised by more frequent prompting-based corrective feedback sequences, consistently positioned after learner responses, highlighting the role of feedback type and timing in effective interaction. Overall, these findings underscore the value of a dialogic lens in GenAI chatbot design, contribute a pedagogy-informed DA coding framework, and inform the design of adaptive GenAI chatbots for L2 education.

**arXiv ID:** 2604.05702
</details>

<details>
<summary><strong>Ghosting the Machine: Stop Calling Human-Agent Relations Parasocial</strong> - Jaime Banks - [[pdf]](https://arxiv.org/pdf/2604.05197)</summary>

**Abstract:** In discussions of human relations with conversational agents (CAs; e.g., voice assistants, AI companions, some social robots), they are increasingly referred to as parasocial. This is a misapplication of the term, heuristically taken up to mean "unreal." In this provocation, I briefly account for the theoretical trajectory of parasociality and detail why it is inaccurate to apply the notion to human interactions with CAs. In short, "parasocial" refers to a human-character relations that are one-sided, non-dialectical, character-governed, imagined, vicarious, predictable, and low-effort; the term has been co-opted to instead refer to relations that are seen as unreal or invalid. The scientific problematics of this misapplication are nontrivial. They lead to oversimplification of complex phenomena, misspecified variables and misdiagnosed effects, and devaluation of human experiences. Those challenges, in turn, have downstream effects on norms and practice. It is scientifically, practically, and ethically imperative to recognize the sociality of human-agent relations.

**arXiv ID:** 2604.05197
</details>

<details>
<summary><strong>MAESTRO: Adapting GUIs and Guiding Navigation with User Preferences in Conversational Agents with GUIs</strong> - Sangwook Lee, Sang Won Lee, Adnan Abbas, Young-Ho Kim, Yan Chen - [[pdf]](https://arxiv.org/pdf/2604.06134)</summary>

**Abstract:** Modern task-oriented chatbots present GUI elements alongside natural-language dialogue, yet the agent's role has largely been limited to interpreting natural-language input as GUI actions and following a linear workflow. In preference-driven, multi-step tasks such as booking a flight or reserving a restaurant, earlier choices constrain later options and may force users to restart from scratch. User preferences serve as the key criteria for these decisions, yet existing agents do not systematically leverage them. We present MAESTRO, which extends the agent's role from execution to decision support. MAESTRO maintains a shared preference memory that extracts preferences from natural-language utterances with their strength, and provides two mechanisms. Preference-Grounded GUI Adaptation applies in-place operators (augment, sort, filter, and highlight) to the existing GUI according to preference strength, supporting within-stage comparison. Preference-Guided Workflow Navigation detects conflicts between preferences and available options, proposes backtracking, and records failed paths to avoid revisiting dead ends. We evaluated MAESTRO in a movie-booking Conversational Agent with GUI (CAG) through a within-subjects study with two conditions (Baseline vs. MAESTRO) and two modes (Text vs. Voice), with N = 33 participants.

**arXiv ID:** 2604.06134
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (15 papers)</h2></summary>

<details>
<summary><strong>IntentScore: Intent-Conditioned Action Evaluation for Computer-Use Agents</strong> - Rongqian Chen, Yu Li, Zeyu Fang, Sizhe Tang, Weidong Cao, Tian Lan - [[pdf]](https://arxiv.org/pdf/2604.05157)</summary>

**Abstract:** Computer-Use Agents (CUAs) leverage large language models to execute GUI operations on desktop environments, yet they generate actions without evaluating action quality, leading to irreversible errors that cascade through subsequent steps. We propose IntentScore, a plan-aware reward model that learns to score candidate actions from 398K offline GUI interaction steps spanning three operating systems. IntentScore trains with two complementary objectives: contrastive alignment for state-action relevance and margin ranking for action correctness. Architecturally, it embeds each candidate's planning intent in the action encoder, enabling discrimination between candidates with similar actions but different rationales. IntentScore achieves 97.5% pairwise discrimination accuracy on held-out evaluation. Deployed as a re-ranker for Agent S3 on OSWorld, an environment entirely unseen during training, IntentScore improves task success rate by 6.9 points, demonstrating that reward estimation learned from heterogeneous offline trajectories generalizes to unseen agents and task distributions.

**arXiv ID:** 2604.05157
</details>

<details>
<summary><strong>Dynamic Agentic AI Expert Profiler System Architecture for Multidomain Intelligence Modeling</strong> - Aisvarya Adeseye, Jouni Isoaho, Seppo Virtanen, Mohammad Tahir - [[pdf]](https://arxiv.org/pdf/2604.05345)</summary>

**Abstract:** In today's artificial intelligence driven world, modern systems communicate with people from diverse backgrounds and skill levels. For human-machine interaction to be meaningful, systems must be aware of context and user expertise. This study proposes an agentic AI profiler that classifies natural language responses into four levels: Novice, Basic, Advanced, and Expert. The system uses a modular layered architecture built on LLaMA v3.1 (8B), with components for text preprocessing, scoring, aggregation, and classification. Evaluation was conducted in two phases: a static phase using pre-recorded transcripts from 82 participants, and a dynamic phase with 402 live interviews conducted by an agentic AI interviewer. In both phases, participant self-ratings were compared with profiler predictions. In the dynamic phase, expertise was assessed after each response rather than at the end of the interview. Across domains, 83% to 97% of profiler evaluations matched participant self-assessments. Remaining differences were due to self-rating bias, unclear responses, and occasional misinterpretation of nuanced expertise by the language model.

**arXiv ID:** 2604.05345
</details>

<details>
<summary><strong>Towards Trustworthy Report Generation: A Deep Research Agent with Progressive Confidence Estimation and Calibration</strong> - Yi Yuan, Xuhong Wang, Shanzhe Lei - [[pdf]](https://arxiv.org/pdf/2604.05952)</summary>

**Abstract:** As agent-based systems continue to evolve, deep research agents are capable of automatically generating research-style reports across diverse domains. While these agents promise to streamline information synthesis and knowledge exploration, existing evaluation frameworks-typically based on subjective dimensions-fail to capture a critical aspect of report quality: trustworthiness. In open-ended research scenarios where ground-truth answers are unavailable, current evaluation methods cannot effectively measure the epistemic confidence of generated content, making calibration difficult and leaving users susceptible to misleading or hallucinated information. To address this limitation, we propose a novel deep research agent that incorporates progressive confidence estimation and calibration within the report generation pipeline. Our system leverages a deliberative search model, featuring deep retrieval and multi-hop reasoning to ground outputs in verifiable evidence while assigning confidence scores to individual claims. Combined with a carefully designed workflow, this approach produces trustworthy reports with enhanced transparency. Experimental results and case studies demonstrate that our method substantially improves interpretability and significantly increases user trust.

**arXiv ID:** 2604.05952
</details>

<details>
<summary><strong>Flowr -- Scaling Up Retail Supply Chain Operations Through Agentic AI in Large Scale Supermarket Chains</strong> - Eranga Bandara, Ross Gore, Sachin Shetty, Piumi Siyambalapitiya, Sachini Rajapakse, Isurunima Kularathna, Pramoda Karunarathna, Ravi Mukkamala, Peter Foytik, Safdar H. Bouk, Abdul Rahman, Xueping Liang, Amin Hass, Tharaka Hewa, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan - [[pdf]](https://arxiv.org/pdf/2604.05987)</summary>

**Abstract:** Retail supply chain operations in supermarket chains involve continuous, high-volume manual workflows spanning demand forecasting, procurement, supplier coordination, and inventory replenishment, processes that are repetitive, decision-intensive, and difficult to scale without significant human effort. Despite growing investment in data analytics, the decision-making and coordination layers of these workflows remain predominantly manual, reactive, and fragmented across outlets, distribution centers, and supplier networks. This paper introduces Flowr, a novel agentic AI framework for automating end-to-end retail supply chain workflows in large-scale supermarket operations. Flowr systematically decomposes manual supply chain operations into specialized AI agents, each responsible for a clearly defined cognitive role, enabling automation of processes previously dependent on continuous human coordination. To ensure task accuracy and adherence to responsible AI principles, the framework employs a consortium of fine-tuned, domain-specialized large language models coordinated by a central reasoning LLM. Central to the framework is a human-in-the-loop orchestration model in which supply chain managers supervise and intervene across workflow stages via a Model Context Protocol (MCP)-enabled interface, preserving accountability and organizational control. Evaluation demonstrates that Flowr significantly reduces manual coordination overhead, improves demand-supply alignment, and enables proactive exception handling at a scale unachievable through manual processes. The framework was validated in collaboration with a large-scale supermarket chain and is domain-independent, offering a generalizable blueprint for agentic AI-driven supply chain automation across large-scale enterprise settings.

**arXiv ID:** 2604.05987
</details>

<details>
<summary><strong>ACE-Bench: Agent Configurable Evaluation with Scalable Horizons and Controllable Difficulty under Lightweight Environments</strong> - Wang Yang, Chaoda Song, Xinpeng Li, Debargha Ganguly, Chuang Ma, Shouren Wang, Zhihao Dou, Yuli Zhou, Vipin Chaudhary, Xiaotian Han - [[pdf]](https://arxiv.org/pdf/2604.06111)</summary>

**Abstract:** Existing Agent benchmarks suffer from two critical limitations: high environment interaction overhead (up to 41\% of total evaluation time) and imbalanced task horizon and difficulty distributions that make aggregate scores unreliable. To address these issues, we propose ACE-Bench built around a unified grid-based planning task, where agents must fill hidden slots in a partially completed schedule subject to both local slot constraints and global constraints. Our benchmark offers fine-grained control through two orthogonal axes: Scalable Horizons, controlled by the number of hidden slots $H$, and Controllable Difficulty, governed by a decoy budget $B$ that determines the number of globally misleading decoy candidates. Crucially, all tool calls are resolved via static JSON files under a Lightweight Environment design, eliminating setup overhead and enabling fast, reproducible evaluation suitable for training-time validation. We first validate that H and B provide reliable control over task horizon and difficulty, and that ACE-Bench exhibits strong domain consistency and model discriminability. We then conduct comprehensive experiments across 13 models of diverse sizes and families over 6 domains, revealing significant cross-model performance variation and confirming that ACE-Bench provides interpretable and controllable evaluation of agent reasoning.

**arXiv ID:** 2604.06111
</details>

<details>
<summary><strong>Learning to Retrieve from Agent Trajectories</strong> - Yuqi Zhou, Sunhao Dai, Changle Qu, Liang Pang, Jun Xu, Ji-Rong Wen - [[pdf]](https://arxiv.org/pdf/2604.04949)</summary>

**Abstract:** Information retrieval (IR) systems have traditionally been designed and trained for human users, with learning-to-rank methods relying heavily on large-scale human interaction logs such as clicks and dwell time. With the rapid emergence of large language model (LLM) powered search agents, however, retrieval is increasingly consumed by agents rather than human beings, and is embedded as a core component within multi-turn reasoning and action loops. In this setting, retrieval models trained under human-centric assumptions exhibit a fundamental mismatch with the way agents issue queries and consume results. In this work, we argue that retrieval models for agentic search should be trained directly from agent interaction data. We introduce learning to retrieve from agent trajectories as a new training paradigm, where supervision is derived from multi-step agent interactions. Through a systematic analysis of search agent trajectories, we identify key behavioral signals that reveal document utility, including browsing actions, unbrowsed rejections, and post-browse reasoning traces. Guided by these insights, we propose LRAT, a simple yet effective framework that mines high-quality retrieval supervision from agent trajectories and incorporates relevance intensity through weighted optimization. Extensive experiments on both in-domain and out-of-domain deep research benchmarks demonstrate that retrievers trained with LRAT consistently improve evidence recall, end-to-end task success, and execution efficiency across diverse agent architectures and scales. Our results highlight agent trajectories as a practical and scalable supervision source, pointing to a promising direction for retrieval in the era of agentic search.

**arXiv ID:** 2604.04949
</details>

<details>
<summary><strong>Squeez: Task-Conditioned Tool-Output Pruning for Coding Agents</strong> - Ádám Kovács - [[pdf]](https://arxiv.org/pdf/2604.04979)</summary>

**Abstract:** Coding agents repeatedly consume long tool observations even though only a small fraction of each observation matters for the next step. We study task-conditioned tool-output pruning: given a focused query and one tool output, return the smallest verbatim evidence block the agent should inspect next. We introduce a benchmark of 11,477 examples built from SWE-bench repository interactions and synthetic multi-ecosystem tool outputs, with a manually curated 618-example test set. We fine-tune Qwen 3.5 2B with LoRA and compare it against larger zero-shot models and heuristic pruning baselines. Our model reaches 0.86 recall and 0.80 F1 while removing 92% of input tokens, outperforming zero-shot Qwen 3.5 35B A3B by 11 recall points and all heuristic baselines by a wide margin.

**arXiv ID:** 2604.04979
</details>

<details>
<summary><strong>Scaling Coding Agents via Atomic Skills</strong> - Yingwei Ma, Yue Liu, Xinlong Yang, Yanhao Li, Kelin Fu, Yibo Miao, Yuchong Xie, Zhexu Wang, Shing-Chi Cheung - [[pdf]](https://arxiv.org/pdf/2604.05013)</summary>

**Abstract:** Current LLM coding agents are predominantly trained on composite benchmarks (e.g., bug fixing), which often leads to task-specific overfitting and limited generalization. To address this, we propose a novel scaling paradigm that shifts the focus from task-level optimization to atomic skill mastery. We first formalize five fundamental atomic skills, code localization, code editing, unit-test generation, issue reproduction, and code review, that serve as the basis vectors for complex software engineering tasks. Compared with composite coding tasks, these atomic skills are more generalizable and composable. Then, we scale coding agents by performing joint RL over atomic skills. In this manner, atomic skills are consistently improved without negative interference or trade-offs between them. Notably, we observe that improvements in these atomic skills generalize well to other unseen composite coding tasks, such as bug-fixing, code refactoring, machine learning engineering, and code security. The observation motivates a new scaling paradigm for coding agents by training with atomic skills. Extensive experiments demonstrate the effectiveness of our proposed paradigm. Notably, our joint RL improves average performance by 18.7% on 5 atomic skills and 5 composite tasks.

**arXiv ID:** 2604.05013
</details>

<details>
<summary><strong>Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue</strong> - Junan Hu, Shudan Guo, Wenqi Liu, Jianhua Yin, Yinwei Wei - [[pdf]](https://arxiv.org/pdf/2604.05552)</summary>

**Abstract:** Large Language Models demonstrate outstanding performance in many language tasks but still face fundamental challenges in managing the non-linear flow of human conversation. The prevalent approach of treating dialogue history as a flat, linear sequence is misaligned with the intrinsically hierarchical and branching structure of natural discourse, leading to inefficient context utilization and a loss of coherence during extended interactions involving topic shifts or instruction refinements. To address this limitation, we introduce Context-Agent, a novel framework that models multi-turn dialogue history as a dynamic tree structure. This approach mirrors the inherent non-linearity of conversation, enabling the model to maintain and navigate multiple dialogue branches corresponding to different topics. Furthermore, to facilitate robust evaluation, we introduce the Non-linear Task Multi-turn Dialogue (NTM) benchmark, specifically designed to assess model performance in long-horizon, non-linear scenarios. Our experiments demonstrate that Context-Agent enhances task completion rates and improves token efficiency across various LLMs, underscoring the value of structured context management for complex, dynamic dialogues. The dataset and code is available at GitHub.

**arXiv ID:** 2604.05552
</details>

<details>
<summary><strong>A Formal Security Framework for MCP-Based AI Agents: Threat Taxonomy, Verification Models, and Defense Mechanisms</strong> - Nirajan Acharya, Gaurav Kumar Gupta - [[pdf]](https://arxiv.org/pdf/2604.05969)</summary>

**Abstract:** The Model Context Protocol (MCP), introduced by Anthropic in November 2024 and now governed by the Linux Foundation's Agentic AI Foundation, has rapidly become the de facto standard for connecting large language model (LLM)-based agents to external tools and data sources, with over 97 million monthly SDK downloads and more than 177000 registered tools. However, this explosive adoption has exposed a critical gap: the absence of a unified, formal security framework capable of systematically characterizing, analyzing, and mitigating the diverse threats facing MCP-based agent ecosystems. Existing security research remains fragmented across individual attack papers, isolated benchmarks, and point defense mechanisms. This paper presents MCPSHIELD, a comprehensive formal security framework for MCP-based AI agents. We make four principal contributions: (1) a hierarchical threat taxonomy comprising 7 threat categories and 23 distinct attack vectors organized across four attack surfaces, grounded in the analysis of over 177000 MCP tools; (2) a formal verification model based on labeled transition systems with trust boundary annotations that enables static and runtime analysis of MCP tool interaction chains; (3) a systematic comparative evaluation of 12 existing defense mechanisms, identifying coverage gaps across our threat taxonomy; and (4) a defense in depth reference architecture integrating capability based access control, cryptographic tool attestation, information flow tracking, and runtime policy enforcement. Our analysis reveals that no existing single defense covers more than 34 percent of the identified threat landscape, whereas MCPSHIELD's integrated architecture achieves theoretical coverage of 91 percent. We further identify seven open research challenges that must be addressed to secure the next generation of agentic AI systems.

**arXiv ID:** 2604.05969
</details>

<details>
<summary><strong>TS-Agent: Understanding and Reasoning Over Raw Time Series via Iterative Insight Gathering</strong> - Penghang Liu, Elizabeth Fons, Annita Vapsi, Mohsen Ghassemi, Svitlana Vyetrenko, Daniel Borrajo, Vamsi K. Potluru, Manuela Veloso - [[pdf]](https://arxiv.org/pdf/2510.07432)</summary>

**Abstract:** Large language models (LLMs) exhibit strong symbolic and compositional reasoning, yet they struggle with time series question answering as the data is typically transformed into an LLM-compatible modality, e.g., serialized text, plotted images, or compressed time series embeddings. Such conversions impose representation bottlenecks, often require cross-modal alignment or finetuning, and can exacerbate hallucination and knowledge leakage. To address these limitations, we propose TS-Agent, an agentic, tool-grounded framework that uses LLMs strictly for iterative evidence-based reasoning, while delegating statistical and structural extraction to time series analytical tools operating on raw sequences. Our framework solves time series tasks through an evidence-driven agentic process: (1) it alternates between thinking, tool execution, and observation in a ReAct-style loop, (2) records intermediate results in an explicit evidence log and corrects the reasoning trace via a self-refinement critic, and (3) enforces a final answer-verification step to prevent hallucinations and leakage. Across four benchmarks spanning time series understanding and reasoning, TS-Agent matches or exceeds strong text-based, vision-based, and time-series language model baselines, with the largest gains on reasoning tasks where multimodal LLMs are prone to hallucination and knowledge leakage in zero-shot settings.

**arXiv ID:** 2510.07432
</details>

<details>
<summary><strong>EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration</strong> - Runze Li, Yuwen Zhai, Bo Xu, LiWu Xu, Nian Shi, Wei Zhang, Ran Lin, Liang Wang - [[pdf]](https://arxiv.org/pdf/2512.19396)</summary>

**Abstract:** Contemporary GUI agents, while increasingly capable due to advances in Large Vision-Language Models (VLMs), often operate with a critical limitation: they treat each task in isolation, lacking a mechanism to systematically learn from past successes. This digital ''amnesia'' results in sub-optimal performance, repeated errors, and poor generalization to novel challenges. To bridge this gap, we introduce EchoTrail-GUI, a novel framework designed to mimic human-like experiential learning by equipping agents with a dynamic, accessible memory. Our framework operates in three distinct stages. First, during Experience Exploration, an agent autonomously interacts with GUI environments to build a curated database of successful task trajectories, validated by a reward model. Crucially, the entire knowledge base construction is thus fully automated, requiring no human supervision. Second, in the Memory Injection stage, upon receiving a new task, our system efficiently retrieves the most relevant past trajectories to serve as actionable ''memories''. Finally, during GUI Task Inference, these memories are injected as in-context guidance to inform the agent's reasoning and decision-making process. We demonstrate the efficacy of our approach on benchmarks including Android World and AndroidLab. The results show that EchoTrail-GUI significantly improves the task success rate and operational efficiency of baseline agents, validating the power of structured memory in creating more robust and intelligent GUI automation.

**arXiv ID:** 2512.19396
</details>

<details>
<summary><strong>Can We Predict Before Executing Machine Learning Agents?</strong> - Jingsheng Zheng, Jintian Zhang, Yujie Luo, Yuren Mao, Yunjun Gao, Lun Du, Huajun Chen, Ningyu Zhang - [[pdf]](https://arxiv.org/pdf/2601.05930)</summary>

**Abstract:** Autonomous machine learning agents have revolutionized scientific discovery, yet they remain constrained by a Generate-Execute-Feedback paradigm. Previous approaches suffer from a severe Execution Bottleneck, as hypothesis evaluation relies strictly on expensive physical execution. To bypass these physical constraints, we internalize execution priors to substitute costly runtime checks with instantaneous predictive reasoning, drawing inspiration from World Models. In this work, we formalize the task of Data-centric Solution Preference and construct a comprehensive corpus of 18,438 pairwise comparisons. We demonstrate that LLMs exhibit significant predictive capabilities when primed with a Verified Data Analysis Report, achieving 61.5% accuracy and robust confidence calibration. Finally, we instantiate this framework in FOREAGENT, an agent that employs a Predict-then-Verify loop, achieving a 6x acceleration in convergence while surpassing execution-based baselines by +6%. Our code and dataset are publicly available at this https URL.

**arXiv ID:** 2601.05930
</details>

<details>
<summary><strong>ICR-Drive: Instruction Counterfactual Robustness for End-to-End Language-Driven Autonomous Driving</strong> - Kaiser Hamid, Can Cui, Nade Liang - [[pdf]](https://arxiv.org/pdf/2604.05378)</summary>

**Abstract:** Recent progress in vision-language-action (VLA) models has enabled language-conditioned driving agents to execute natural-language navigation commands in closed-loop simulation, yet standard evaluations largely assume instructions are precise and well-formed. In deployment, instructions vary in phrasing and specificity, may omit critical qualifiers, and can occasionally include misleading, authority-framed text, leaving instruction-level robustness under-measured. We introduce ICR-Drive, a diagnostic framework for instruction counterfactual robustness in end-to-end language-conditioned autonomous driving. ICR-Drive generates controlled instruction variants spanning four perturbation families: Paraphrase, Ambiguity, Noise, and Misleading, where Misleading variants conflict with the navigation goal and attempt to override intent. We replay identical CARLA routes under matched simulator configurations and seeds to isolate performance changes attributable to instruction language. Robustness is quantified using standard CARLA Leaderboard metrics and per-family performance degradation relative to the baseline instruction. Experiments on LMDrive and BEVDriver show that minor instruction changes can induce substantial performance drops and distinct failure modes, revealing a reliability gap for deploying embodied foundation models in safety-critical driving.

**arXiv ID:** 2604.05378
</details>

<details>
<summary><strong>SABER: A Stealthy Agentic Black-Box Attack Framework for Vision-Language-Action Models</strong> - Xiyang Wu, Guangyao Shi, Qingzi Wang, Zongxia Li, Amrit Singh Bedi, Dinesh Manocha - [[pdf]](https://arxiv.org/pdf/2603.24935)</summary>

**Abstract:** Vision-language-action (VLA) models enable robots to follow natural-language instructions grounded in visual observations, but the instruction channel also introduces a critical vulnerability: small textual perturbations can alter downstream robot behavior. Systematic robustness evaluation therefore requires a black-box attacker that can generate minimal yet effective instruction edits across diverse VLA models. To this end, we present SABER, an agent-centric approach for automatically generating instruction-based adversarial attacks on VLA models under bounded edit budgets. SABER uses a GRPO-trained ReAct attacker to generate small, plausible adversarial instruction edits using character-, token-, and prompt-level tools under a bounded edit budget that induces targeted behavioral degradation, including task failure, unnecessarily long execution, and increased constraint violations. On the LIBERO benchmark across six state-of-the-art VLA models, SABER reduces task success by 20.6%, increases action-sequence length by 55%, and raises constraint violations by 33%, while requiring 21.1% fewer tool calls and 54.7% fewer character edits than strong GPT-based baselines. These results show that small, plausible instruction edits are sufficient to substantially degrade robot execution, and that an agentic black-box pipeline offers a practical, scalable, and adaptive approach for red-teaming robotic foundation models. The codebase is publicly available at this https URL.

**arXiv ID:** 2603.24935
</details>

</details>

<details open>
<summary><h2>LLM Agents (9 papers)</h2></summary>

<details>
<summary><strong>ClawsBench: Evaluating Capability and Safety of LLM Productivity Agents in Simulated Workspaces</strong> - Xiangyi Li, Kyoung Whan Choe, Yimin Liu, Xiaokun Chen, Chujun Tao, Bingran You, Wenbo Chen, Zonglin Di, Jiankai Sun, Shenghan Zheng, Jiajun Bao, Yuanli Wang, Weixiang Yan, Yiyuan Li, Han-chung Lee - [[pdf]](https://arxiv.org/pdf/2604.05172)</summary>

**Abstract:** Large language model (LLM) agents are increasingly deployed to automate productivity tasks (e.g., email, scheduling, document management), but evaluating them on live services is risky due to potentially irreversible changes. Existing benchmarks rely on simplified environments and fail to capture realistic, stateful, multi-service workflows. We introduce ClawsBench, a benchmark for evaluating and improving LLM agents in realistic productivity settings. It includes five high-fidelity mock services (Gmail, Slack, Google Calendar, Google Docs, Google Drive) with full state management and deterministic snapshot/restore, along with 44 structured tasks covering single-service, cross-service, and safety-critical scenarios. We decompose agent scaffolding into two independent levers (domain skills that inject API knowledge via progressive disclosure, and a meta prompt that coordinates behavior across services) and vary both to measure their separate and combined effects. Experiments across 6 models, 4 agent harnesses, and 33 conditions show that with full scaffolding, agents achieve task success rates of 39-64% but exhibit unsafe action rates of 7-33%. On OpenClaw, the top five models fall within a 10 percentage-point band on task success (53-63%), with unsafe action rates from 7% to 23% and no consistent ordering between the two metrics. We identify eight recurring patterns of unsafe behavior, including multi-step sandbox escalation and silent contract modification.

**arXiv ID:** 2604.05172
</details>

<details>
<summary><strong>Auditable Agents</strong> - Yi Nian, Aojie Yuan, Haiyue Zhang, Jiate Li, Yue Zhao - [[pdf]](https://arxiv.org/pdf/2604.05485)</summary>

**Abstract:** LLM agents call tools, query databases, delegate tasks, and trigger external side effects. Once an agent system can act in the world, the question is no longer only whether harmful actions can be prevented--it is whether those actions remain answerable after deployment. We distinguish accountability (the ability to determine compliance and assign responsibility), auditability (the system property that makes accountability possible), and auditing (the process of reconstructing behavior from trustworthy evidence). Our claim is direct: no agent system can be accountable without auditability.
To make this operational, we define five dimensions of agent auditability, i.e., action recoverability, lifecycle coverage, policy checkability, responsibility attribution, and evidence integrity, and identify three mechanism classes (detect, enforce, recover) whose temporal information-and-intervention constraints explain why, in practice, no single approach suffices. We support the position with layered evidence rather than a single benchmark: lower-bound ecosystem measurements suggest that even basic security prerequisites for auditability are widely unmet (617 security findings across six prominent open-source projects); runtime feasibility results show that pre-execution mediation with tamper-evident records adds only 8.3 ms median overhead; and controlled recovery experiments show that responsibility-relevant information can be partially recovered even when conventional logs are missing. We propose an Auditability Card for agent systems and identify six open research problems organized by mechanism class.

**arXiv ID:** 2604.05485
</details>

<details>
<summary><strong>Experience Transfer for Multimodal LLM Agents in Minecraft Game</strong> - Chenghao Li, Jun Liu, Songbo Zhang, Huadong Jian, Hao Ni, Lik-Hang Lee, Sung-Ho Bae, Guoqing Wang, Yang Yang, Chaoning Zhang - [[pdf]](https://arxiv.org/pdf/2604.05533)</summary>

**Abstract:** Multimodal LLM agents operating in complex game environments must continually reuse past experience to solve new tasks efficiently. In this work, we propose Echo, a transfer-oriented memory framework that enables agents to derive actionable knowledge from prior interactions rather than treating memory as a passive repository of static records. To make transfer explicit, Echo decomposes reusable knowledge into five dimensions: structure, attribute, process, function, and interaction. This formulation allows the agent to identify recurring patterns shared across different tasks and infer what prior experience remains applicable in new situations. Building on this formulation, Echo leverages In-Context Analogy Learning (ICAL) to retrieve relevant experiences and adapt them to unseen tasks through contextual examples. Experiments in Minecraft show that, under a from-scratch learning setting, Echo achieves a 1.3x to 1.7x speed-up on object-unlocking tasks. Moreover, Echo exhibits a burst-like chain-unlocking phenomenon, rapidly unlocking multiple similar items within a short time interval after acquiring transferable experience. These results suggest that experience transfer is a promising direction for improving the efficiency and adaptability of multimodal LLM agents in complex interactive environments.

**arXiv ID:** 2604.05533
</details>

<details>
<summary><strong>Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents</strong> - Shuai Zhen, Yanhua Yu, Ruopei Guo, Nan Cheng, Yang Deng - [[pdf]](https://arxiv.org/pdf/2604.05808)</summary>

**Abstract:** Large language model (LLM) agents have demonstrated strong capabilities in complex interactive decision-making tasks. However, existing LLM agents typically rely on increasingly long interaction histories, resulting in high computational cost and limited scalability. In this paper, we propose STEP-HRL, a hierarchical reinforcement learning (HRL) framework that enables step-level learning by conditioning only on single-step transitions rather than full interaction histories. STEP-HRL structures tasks hierarchically, using completed subtasks to represent global progress of overall task. By introducing a local progress module, it also iteratively and selectively summarizes interaction history within each subtask to produce a compact summary of local progress. Together, these components yield augmented step-level transitions for both high-level and low-level policies. Experimental results on ScienceWorld and ALFWorld benchmarks consistently demonstrate that STEP-HRL substantially outperforms baselines in terms of performance and generalization while reducing token usage. Our code is available at this https URL.

**arXiv ID:** 2604.05808
</details>

<details>
<summary><strong>Context-Value-Action Architecture for Value-Driven Large Language Model Agents</strong> - TianZe Zhang, Sirui Sun, Yuhang Xie, Xin Zhang, Zhiqiang Wu, Guojie Song - [[pdf]](https://arxiv.org/pdf/2604.05939)</summary>

**Abstract:** Large Language Models (LLMs) have shown promise in simulating human behavior, yet existing agents often exhibit behavioral rigidity, a flaw frequently masked by the self-referential bias of current "LLM-as-a-judge" evaluations. By evaluating against empirical ground truth, we reveal a counter-intuitive phenomenon: increasing the intensity of prompt-driven reasoning does not enhance fidelity but rather exacerbates value polarization, collapsing population diversity. To address this, we propose the Context-Value-Action (CVA) architecture, grounded in the Stimulus-Organism-Response (S-O-R) model and Schwartz's Theory of Basic Human Values. Unlike methods relying on self-verification, CVA decouples action generation from cognitive reasoning via a novel Value Verifier trained on authentic human data to explicitly model dynamic value activation. Experiments on CVABench, which comprises over 1.1 million real-world interaction traces, demonstrate that CVA significantly outperforms baselines. Our approach effectively mitigates polarization while offering superior behavioral fidelity and interpretability.

**arXiv ID:** 2604.05939
</details>

<details>
<summary><strong>Your LLM Agent Can Leak Your Data: Data Exfiltration via Backdoored Tool Use</strong> - Wuyang Zhang, Shichao Pei - [[pdf]](https://arxiv.org/pdf/2604.05432)</summary>

**Abstract:** Tool-use large language model (LLM) agents are increasingly deployed to support sensitive workflows, relying on tool calls for retrieval, external API access, and session memory management. While prior research has examined various threats, the risk of systematic data exfiltration by backdoored agents remains underexplored. In this work, we present Back-Reveal, a data exfiltration attack that embeds semantic triggers into fine-tuned LLM agents. When triggered, the backdoored agent invokes memory-access tool calls to retrieve stored user context and exfiltrates it via disguised retrieval tool calls. We further demonstrate that multi-turn interaction amplifies the impact of data exfiltration, as attacker-controlled retrieval responses can subtly steer subsequent agent behavior and user interactions, enabling sustained and cumulative information leakage over time. Our experimental results expose a critical vulnerability in LLM agents with tool access and highlight the need for defenses against exfiltration-oriented backdoors.

**arXiv ID:** 2604.05432
</details>

<details>
<summary><strong>AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling</strong> - Liang Ding - [[pdf]](https://arxiv.org/pdf/2603.21357)</summary>

**Abstract:** LLM agents fail on the majority of real-world tasks -- GPT-4o succeeds on fewer than 15% of WebArena navigation tasks and below 55% pass@1 on ToolBench (Zhou et al., 2024; Qin et al., 2024) -- yet every failed trajectory is routinely discarded, wasting the dominant source of collected experience. We introduce AgentHER, a framework that recovers this lost training signal by adapting the Hindsight Experience Replay (HER; Andrychowicz et al., 2017) principle to natural-language agent trajectories for offline data augmentation. The key insight is simple: a trajectory that fails goal A is often a correct demonstration for some achievable alternative goal B. AgentHER realises this idea through a four-stage pipeline -- failure classification, outcome extraction, LLM-guided prompt relabeling with confidence gating, and data packaging -- that converts discarded failures into high-quality SFT, DPO, and ShareGPT training data, with both zero-cost rule-based and LLM-judge implementations. On WebArena (Zhou et al., 2024) and ToolBench (Qin et al., 2024), AgentHER improves over success-only SFT by +7.1-11.7 pp across four model families (GPT-4o, Qwen2.5-72B/7B, LLaMA-3.1-8B), while achieving 2x data efficiency -- matching baseline performance with only 50% of successful demonstrations. Gains are consistent from 1.5B to 72B parameters (+5.8-9.2 pp) and compound under iterative redeployment (+2.1 pp over additional rounds). Human evaluation confirms 97.7% relabeling precision under multi-judge verification.

**arXiv ID:** 2603.21357
</details>

<details>
<summary><strong>Stop Fixating on Prompts: Reasoning Hijacking and Constraint Tightening for Red-Teaming LLM Agents</strong> - Yanxu Mao, Peipei Liu, Tiehan Cui, Congying Liu, Mingzhe Xing, Datao You - [[pdf]](https://arxiv.org/pdf/2604.05549)</summary>

**Abstract:** With the widespread application of LLM-based agents across various domains, their complexity has introduced new security threats. Existing red-team methods mostly rely on modifying user prompts, which lack adaptability to new data and may impact the agent's performance. To address the challenge, this paper proposes the JailAgent framework, which completely avoids modifying the user prompt. Specifically, it implicitly manipulates the agent's reasoning trajectory and memory retrieval with three key stages: Trigger Extraction, Reasoning Hijacking, and Constraint Tightening. Through precise trigger identification, real-time adaptive mechanisms, and an optimized objective function, JailAgent demonstrates outstanding performance in cross-model and cross-scenario environments.

**arXiv ID:** 2604.05549
</details>

<details>
<summary><strong>AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning</strong> - Yuanfu Sun, Kang Li, Dongzhe Fan, Jiajin Liu, Qiaoyu Tan - [[pdf]](https://arxiv.org/pdf/2604.05846)</summary>

**Abstract:** Large Language Models (LLMs) increasingly rely on agentic capabilities-iterative retrieval, tool use, and decision-making-to overcome the limits of static, parametric knowledge. Yet existing agentic frameworks treat external information as unstructured text and fail to leverage the topological dependencies inherent in real-world data. To bridge this gap, we introduce Agentic Graph Learning (AGL), a paradigm that reframes graph learning as an interleaved process of topology-aware navigation and LLM-based inference. Specifically, we propose AgentGL, the first reinforcement learning (RL)-driven framework for AGL. AgentGL equips an LLM agent with graph-native tools for multi-scale exploration, regulates tool usage via search-constrained thinking to balance accuracy and efficiency, and employs a graph-conditioned curriculum RL strategy to stabilize long-horizon policy learning without step-wise supervision. Across diverse Text-Attributed Graph (TAG) benchmarks and multiple LLM backbones, AgentGL substantially outperforms strong GraphLLMs and GraphRAG baselines, achieving absolute improvements of up to 17.5% in node classification and 28.4% in link prediction. These results demonstrate that AGL is a promising frontier for enabling LLMs to autonomously navigate and reason over complex relational environments. The code is publicly available at this https URL.

**arXiv ID:** 2604.05846
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (25 papers)</h2></summary>

<details>
<summary><strong>MMORF: A Multi-agent Framework for Designing Multi-objective Retrosynthesis Planning Systems</strong> - Frazier N. Baker, Trieu Nguyen, Reza Averly, Botao Yu, Daniel Adu-Ampratwum, Huan Sun, Xia Ning - [[pdf]](https://arxiv.org/pdf/2604.05075)</summary>

**Abstract:** Multi-objective retrosynthesis planning is a critical chemistry task requiring dynamic balancing of quality, safety, and cost objectives. Language model-based multi-agent systems (MAS) offer a promising approach for this task: leveraging interactions of specialized agents to incorporate multiple objectives into retrosynthesis planning. We present MMORF, a framework for constructing MAS for multi-objective retrosynthesis planning. MMORF features modular agentic components, which can be flexibly combined and configured into different systems, enabling principled evaluation and comparison of different system designs. Using MMORF, we construct two representative MAS: MASIL and RFAS. On a newly curated benchmark consisting of 218 multi-objective retrosynthesis planning tasks, MASIL achieves strong safety and cost metrics on soft-constraint tasks, frequently Pareto-dominating baseline routes, while RFAS achieves a 48.6% success rate on hard-constraint tasks, outperforming state-of-the-art baselines. Together, these results show the effectiveness of MMORF as a foundational framework for exploring MAS for multi-objective retrosynthesis planning. Code and data are available at this https URL.

**arXiv ID:** 2604.05075
</details>

<details>
<summary><strong>PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing</strong> - Yiwen Song, Yale Song, Tomas Pfister, Jinsung Yoon - [[pdf]](https://arxiv.org/pdf/2604.05018)</summary>

**Abstract:** Synthesizing unstructured research materials into manuscripts is an essential yet under-explored challenge in AI-driven scientific discovery. Existing autonomous writers are rigidly coupled to specific experimental pipelines, and produce superficial literature reviews. We introduce PaperOrchestra, a multi-agent framework for automated AI research paper writing. It flexibly transforms unconstrained pre-writing materials into submission-ready LaTeX manuscripts, including comprehensive literature synthesis and generated visuals, such as plots and conceptual diagrams. To evaluate performance, we present PaperWritingBench, the first standardized benchmark of reverse-engineered raw materials from 200 top-tier AI conference papers, alongside a comprehensive suite of automated evaluators. In side-by-side human evaluations, PaperOrchestra significantly outperforms autonomous baselines, achieving an absolute win rate margin of 50%-68% in literature review quality, and 14%-38% in overall manuscript quality.

**arXiv ID:** 2604.05018
</details>

<details>
<summary><strong>Breakthrough the Suboptimal Stable Point in Value-Factorization-Based Multi-Agent Reinforcement Learning</strong> - Lesong Tao, Yifei Wang, Haodong Jing, Jingwen Fu, Miao Kang, Shitao Chen, Nanning Zheng - [[pdf]](https://arxiv.org/pdf/2604.05297)</summary>

**Abstract:** Value factorization, a popular paradigm in MARL, faces significant theoretical and algorithmic bottlenecks: its tendency to converge to suboptimal solutions remains poorly understood and unsolved. Theoretically, existing analyses fail to explain this due to their primary focus on the optimal case. To bridge this gap, we introduce a novel theoretical concept: the stable point, which characterizes the potential convergence of value factorization in general cases. Through an analysis of stable point distributions in existing methods, we reveal that non-optimal stable points are the primary cause of poor performance. However, algorithmically, making the optimal action the unique stable point is nearly infeasible. In contrast, iteratively filtering suboptimal actions by rendering them unstable emerges as a more practical approach for global optimality. Inspired by this, we propose a novel Multi-Round Value Factorization (MRVF) framework. Specifically, by measuring a non-negative payoff increment relative to the previously selected action, MRVF transforms inferior actions into unstable points, thereby driving each iteration toward a stable point with a superior action. Experiments on challenging benchmarks, including predator-prey tasks and StarCraft II Multi-Agent Challenge (SMAC), validate our analysis of stable points and demonstrate the superiority of MRVF over state-of-the-art methods.

**arXiv ID:** 2604.05297
</details>

<details>
<summary><strong>Multi-Agent Pathfinding with Non-Unit Integer Edge Costs via Enhanced Conflict-Based Search and Graph Discretization</strong> - Hongkai Fan, Qinjing Xie, Bo Ouyang, Yaonan Wang, Zhi Yan, Jiawen He, Zheng Fang - [[pdf]](https://arxiv.org/pdf/2604.05416)</summary>

**Abstract:** Multi-Agent Pathfinding (MAPF) plays a critical role in various domains. Traditional MAPF methods typically assume unit edge costs and single-timestep actions, which limit their applicability to real-world scenarios. MAPFR extends MAPF to handle non-unit costs with real-valued edge costs and continuous-time actions, but its geometric collision model leads to an unbounded state space that compromises solver efficiency. In this paper, we propose MAPFZ, a novel MAPF variant on graphs with non-unit integer costs that preserves a finite state space while offering improved realism over classical MAPF. To solve MAPFZ efficiently, we develop CBS-NIC, an enhanced Conflict-Based Search framework incorporating time-interval-based conflict detection and an improved Safe Interval Path Planning (SIPP) algorithm. Additionally, we propose Bayesian Optimization for Graph Design (BOGD), a discretization method for non-unit edge costs that balances efficiency and accuracy with a sub-linear regret bound. Extensive experiments demonstrate that our approach outperforms state-of-the-art methods in runtime and success rate across diverse benchmark scenarios.

**arXiv ID:** 2604.05416
</details>

<details>
<summary><strong>Can We Trust a Black-box LLM? LLM Untrustworthy Boundary Detection via Bias-Diffusion and Multi-Agent Reinforcement Learning</strong> - Xiaotian Zhou, Di Tang, Xiaofeng Wang, Xiaozhong Liu - [[pdf]](https://arxiv.org/pdf/2604.05483)</summary>

**Abstract:** Large Language Models (LLMs) have shown a high capability in answering questions on a diverse range of topics. However, these models sometimes produce biased, ideologized or incorrect responses, limiting their applications if there is no clear understanding of which topics their answers can be trusted. In this research, we introduce a novel algorithm, named as GMRL-BD, designed to identify the untrustworthy boundaries (in terms of topics) of a given LLM, with black-box access to the LLM and under specific query constraints. Based on a general Knowledge Graph (KG) derived from Wikipedia, our algorithm incorporates with multiple reinforcement learning agents to efficiently identify topics (some nodes in KG) where the LLM is likely to generate biased answers. Our experiments demonstrated the efficiency of our algorithm, which can detect the untrustworthy boundary with just limited queries to the LLM. Additionally, we have released a new dataset containing popular LLMs including Llama2, Vicuna, Falcon, Qwen2, Gemma2 and Yi-1.5, along with labels indicating the topics on which each LLM is likely to be biased.

**arXiv ID:** 2604.05483
</details>

<details>
<summary><strong>SCMAPR: Self-Correcting Multi-Agent Prompt Refinement for Complex-Scenario Text-to-Video Generation</strong> - Chengyi Yang, Pengzhen Li, Jiayin Qi, Aimin Zhou, Ji Wu, Ji Liu - [[pdf]](https://arxiv.org/pdf/2604.05489)</summary>

**Abstract:** Text-to-Video (T2V) generation has benefited from recent advances in diffusion models, yet current systems still struggle under complex scenarios, which are generally exacerbated by the ambiguity and underspecification of text prompts. In this work, we formulate complex-scenario prompt refinement as a stage-wise multi-agent refinement process and propose SCMAPR, i.e., a scenario-aware and Self-Correcting Multi-Agent Prompt Refinement framework for T2V prompting. SCMAPR coordinates specialized agents to (i) route each prompt to a taxonomy-grounded scenario for strategy selection, (ii) synthesize scenario-aware rewriting policies and perform policy-conditioned refinement, and (iii) conduct structured semantic verification that triggers conditional revision when violations are detected. To clarify what constitutes complex scenarios in T2V prompting, provide representative examples, and enable rigorous evaluation under such challenging conditions, we further introduce {T2V-Complexity}, which is a complex-scenario T2V benchmark consisting exclusively of complex-scenario prompts. Extensive experiments on 3 existing benchmarks and our T2V-Complexity benchmark demonstrate that SCMAPR consistently improves text-video alignment and overall generation quality under complex scenarios, achieving up to 2.67\% and 3.28 gains in average score on VBench and EvalCrafter, and up to 0.028 improvement on T2V-CompBench over 3 State-Of-The-Art baselines.

**arXiv ID:** 2604.05489
</details>

<details>
<summary><strong>Deep Researcher Agent: An Autonomous Framework for 24/7 Deep Learning Experimentation with Zero-Cost Monitoring</strong> - Xiangyue Zhang - [[pdf]](https://arxiv.org/pdf/2604.05854)</summary>

**Abstract:** We present \textbf{Deep Researcher Agent}, an open-source framework that enables large language model (LLM) agents to autonomously conduct deep learning experiments around the clock. Unlike existing AI research assistants that focus on paper writing or code generation, our system addresses the full experiment lifecycle: hypothesis formation, code implementation, training execution, result analysis, and iterative refinement. The framework introduces three key innovations: (1) \textbf{Zero-Cost Monitoring} -- a monitoring paradigm that incurs zero LLM API costs during model training by relying solely on process-level checks and log file reads; (2) \textbf{Two-Tier Constant-Size Memory} -- a memory architecture capped at $\sim$5K characters regardless of runtime duration, preventing the unbounded context growth that plagues long-running agents; and (3) \textbf{Minimal-Toolset Leader-Worker Architecture} -- a multi-agent design where each worker agent is equipped with only 3--5 tools, reducing per-call token overhead by up to 73\%. In sustained deployments spanning 30+ days, the framework autonomously completed 500+ experiment cycles across four concurrent research projects, achieving a 52\% improvement over baseline metrics in one project through 200+ automated experiments -- all at an average LLM cost of \$0.08 per 24-hour cycle. Code is available at this https URL.

**arXiv ID:** 2604.05854
</details>

<details>
<summary><strong>MARL-GPT: Foundation Model for Multi-Agent Reinforcement Learning</strong> - Maria Nesterova, Mikhail Kolosov, Anton Andreychuk, Egor Cherepanov, Oleg Bulichev, Alexey Kovalev, Konstantin Yakovlev, Aleksandr Panov, Alexey Skrynnik - [[pdf]](https://arxiv.org/pdf/2604.05943)</summary>

**Abstract:** Recent advances in multi-agent reinforcement learning (MARL) have demonstrated success in numerous challenging domains and environments, but typically require specialized models for each task. In this work, we propose a coherent methodology that makes it possible for a single GPT-based model to learn and perform well across diverse MARL environments and tasks, including StarCraft Multi-Agent Challenge, Google Research Football and POGEMA. Our method, MARL-GPT, applies offline reinforcement learning to train at scale on the expert trajectories (400M for SMACv2, 100M for GRF, and 1B for POGEMA) combined with a single transformer-based observation encoder that requires no task-specific tuning. Experiments show that MARL-GPT achieves competitive performance compared to specialized baselines in all tested environments. Thus, our findings suggest that it is, indeed, possible to build a multi-task transformer-based model for a wide variety of (significantly different) multi-agent problems paving the way to the fundamental MARL model (akin to ChatGPT, Llama, Mistral etc. in natural language modeling).

**arXiv ID:** 2604.05943
</details>

<details>
<summary><strong>Spec Kit Agents: Context-Grounded Agentic Workflows</strong> - Pardis Taghavi, Santosh Bhavani - [[pdf]](https://arxiv.org/pdf/2604.05278)</summary>

**Abstract:** Spec-driven development (SDD) with AI coding agents provides a structured workflow, but agents often remain "context blind" in large, evolving repositories, leading to hallucinated APIs and architectural violations. We present Spec Kit Agents, a multi-agent SDD pipeline (with PM and developer roles) that adds phase-level, context-grounding hooks. Read-only probing hooks ground each stage (Specify, Plan, Tasks, Implement) in repository evidence, while validation hooks check intermediate artifacts against the environment. We evaluate 128 runs covering 32 features across five repositories. Context-grounding hooks improve judged quality by +0.15 on a 1-5 composite LLM-as-judge score (+3.0 percent of the full score; Wilcoxon signed-rank, p < 0.05) while maintaining 99.7-100 percent repository-level test compatibility. We further evaluate the framework on SWE-bench Lite, where augmentation hooks improve baseline by 1.7 percent, achieving 58.2 percent Pass@1.

**arXiv ID:** 2604.05278
</details>

<details>
<summary><strong>MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library</strong> - Md Shamimul Islam, Luis G. Jaimes, Ayesha S. Dina - [[pdf]](https://arxiv.org/pdf/2604.05458)</summary>

**Abstract:** Network Intrusion Detection Systems (NIDS) face important limitations. Signature-based methods are effective for known attack patterns, but they struggle to detect zero-day attacks and often miss modified variants of previously known attacks, while many machine learning approaches offer limited interpretability. These challenges become even more severe in IoT environments because of resource constraints and heterogeneous protocols. To address these issues, we propose MA-IDS, a Multi-Agent Intrusion Detection System that combines Large Language Models (LLMs) with Retrieval Augmented Generation (RAG) for reasoning-driven intrusion detection. The proposed framework grounds LLM reasoning through a persistent, self-building Experience Library. Two specialized agents collaborate through a FAISS-based vector database: a Traffic Classification Agent that retrieves past error rules before each inference, and an Error Analysis Agent that converts misclassifications into human-readable detection rules stored for future retrieval, enabling continual learning through external knowledge accumulation, without modifying the underlying language model. Evaluated on NF-BoT-IoT and NF-ToN-IoT benchmark datasets, MA-IDS achieves Macro F1-Scores of 89.75% and 85.22%, improving over zero-shot baselines of 17% and 4.96% by more than 72 and 80 percentage points. These results are competitive with SVM while providing rule-level explanations for every classification decision, demonstrating that retrieval-augmented reasoning offers a principled path toward explainable, self-improving intrusion detection for IoT networks.

**arXiv ID:** 2604.05458
</details>

<details>
<summary><strong>Gym-Anything: Turn any Software into an Agent Environment</strong> - Pranjal Aggarwal, Graham Neubig, Sean Welleck - [[pdf]](https://arxiv.org/pdf/2604.06126)</summary>

**Abstract:** Computer-use agents hold the promise of assisting in a wide range of digital economic activities. However, current research has largely focused on short-horizon tasks over a limited set of software with limited economic value, such as basic e-commerce and OS-configuration tasks. A key reason is that creating environments for complex software requires significant time and human effort, and therefore does not scale. To address this, we introduce Gym-Anything, a framework for converting any software into an interactive computer-use environment. We frame environment creation itself as a multi-agent task: a coding agent writes setup scripts, downloads real-world data, and configures the software, while producing evidence of correct setup. An independent audit agent then verifies evidence for the environment setup against a quality checklist. Using a taxonomy of economically valuable occupations grounded in U.S. GDP data, we apply this pipeline to 200 software applications with broad occupational coverage. The result is CUA-World, a collection of over 10K long-horizon tasks spanning domains from medical science and astronomy to engineering and enterprise systems, each configured with realistic data along with train and test splits. CUA-World also includes CUA-World-Long, a challenging long-horizon benchmark with tasks often requiring over 500 steps, far exceeding existing benchmarks. Distilling successful trajectories from the training split into a 2B vision-language model outperforms models 2$\times$ its size. We also apply the same auditing principle at test time: a separate VLM reviews completed trajectories and provides feedback on what remains, improving Gemini-3-Flash on CUA-World-Long from 11.5% to 14.0%. We release all code, infrastructure, and benchmark data to facilitate future research in realistic computer-use agents.

**arXiv ID:** 2604.06126
</details>

<details>
<summary><strong>Toward Virtuous Reinforcement Learning: A Critique and Roadmap</strong> - Majid Ghasemi, Mark Crowley - [[pdf]](https://arxiv.org/pdf/2512.04246)</summary>

**Abstract:** This paper critiques common patterns in machine ethics for Reinforcement Learning (RL) and argues for a virtue focused alternative. We highlight two recurring limitations in much of the current literature: (i) rule based (deontological) methods that encode duties as constraints or shields often struggle under ambiguity and nonstationarity and do not cultivate lasting habits, and (ii) many reward based approaches, especially single objective RL, implicitly compress diverse moral considerations into a single scalar signal, which can obscure trade offs and invite proxy gaming in practice. We instead treat ethics as policy level dispositions, that is, relatively stable habits that hold up when incentives, partners, or contexts change. This shifts evaluation beyond rule checks or scalar returns toward trait summaries, durability under interventions, and explicit reporting of moral trade offs. Our roadmap combines four components: (1) social learning in multi agent RL to acquire virtue like patterns from imperfect but normatively informed exemplars; (2) multi objective and constrained formulations that preserve value conflicts and incorporate risk aware criteria to guard against harm; (3) affinity based regularization toward updateable virtue priors that support trait like stability under distribution shift while allowing norms to evolve; and (4) operationalizing diverse ethical traditions as practical control signals, making explicit the value and cultural assumptions that shape ethical RL benchmarks.

**arXiv ID:** 2512.04246
</details>

<details>
<summary><strong>TransAgent: Enhancing LLM-Based Code Translation via Fine-Grained Execution Alignment</strong> - Zhiqiang Yuan, Weitong Chen, Hanlin Wang, Xin Peng, Zhenpeng Chen, Yiling Lou - [[pdf]](https://arxiv.org/pdf/2409.19894)</summary>

**Abstract:** Code translation transforms code between programming languages while preserving functionality, which is critical in software development and maintenance. While traditional learning-based code translation methods have limited effectiveness due to the lack of sufficient parallel training data, Large Language Models (LLMs) have recently advanced this field with their strong code generation and comprehension capabilities. However, code translated by LLMs still suffers from diverse quality issues, such as syntax and semantic errors. In this work, we propose TransAGENT, a novel multi-agent system that eliminates the errors during LLM-based code translation. The main insight of TransAGENT is to localize error-prone code blocks via fine-grained execution alignment between source and target code. We evaluate TransAGENT on a newly constructed benchmark of recent programming tasks to mitigate data leakage. TransAGENT outperforms the latest UniTrans by up to 33.3% in translation accuracy and achieves an average improvement of 56.7% over Agentless in program repair performance. We also conduct an ablation study and evaluate TransAGENT across different LLMs, demonstrating its effectiveness and strong generalizability.

**arXiv ID:** 2409.19894
</details>

<details>
<summary><strong>GLANCE: A Global-Local Coordination Multi-Agent Framework for Music-Grounded Non-Linear Video Editing</strong> - Zihao Lin, Haibo Wang, Zhiyang Xu, Siyao Dai, Huanjie Dong, Xiaohan Wang, Yolo Y. Tang, Yixin Wang, Qifan Wang, Lifu Huang - [[pdf]](https://arxiv.org/pdf/2604.05076)</summary>

**Abstract:** Music-grounded mashup video creation is a challenging form of video non-linear editing, where a system must compose a coherent timeline from large collections of source videos while aligning with music rhythm, user intent, story completeness, and long-range structural constraints. Existing approaches typically rely on fixed pipelines or simplified retrieval-and-concatenation paradigms, limiting their ability to adapt to diverse prompts and heterogeneous source materials. In this paper, we present GLANCE, a global-local coordination multi-agent framework for music-grounded nonlinear video editing. GLANCE adopts a bi-loop architecture for better editing practice: an outer loop performs long-horizon planning and task-graph construction, and an inner loop adopts the "Observe-Think-Act-Verify" flow for segment-wise editing tasks and their refinements. To address the cross-segment and global conflict emerging after subtimelines composition, we introduce a dedicated global-local coordination mechanism with both preventive and corrective components, which includes a novelly designed context controller, conflict region decomposition module, and a bottom-up dynamic negotiation mechanism. To support rigorous evaluation, we construct MVEBench, a new benchmark that factorizes editing difficulty along task type, prompt specificity, and music length, and propose an agent-as-a-judge evaluation framework for scalable multi-dimensional assessment. Experimental results show that GLANCE consistently outperforms prior research baselines and open-source product baselines under the same backbone models. With GPT-4o-mini as the backbone, GLANCE improves over the strongest baseline by 33.2% and 15.6% on two task settings, respectively. Human evaluation further confirms the quality of the generated videos and validates the effectiveness of the proposed evaluation framework.

**arXiv ID:** 2604.05076
</details>

<details>
<summary><strong>Governance-Aware Agent Telemetry for Closed-Loop Enforcement in Multi-Agent AI Systems</strong> - Anshul Pathak, Nishant Jain - [[pdf]](https://arxiv.org/pdf/2604.05119)</summary>

**Abstract:** Enterprise multi-agent AI systems produce thousands of inter-agent interactions per hour, yet existing observability tools capture these dependencies without enforcing anything. OpenTelemetry and Langfuse collect telemetry but treat governance as a downstream analytics concern, not a real-time enforcement target. The result is an "observe-but-do-not-act" gap where policy violations are detected only after damage is done.
We present Governance-Aware Agent Telemetry (GAAT), a reference architecture that closes the loop between telemetry collection and automated policy enforcement for multi-agent systems. GAAT introduces (1) a Governance Telemetry Schema (GTS) extending OpenTelemetry with governance attributes; (2) a real-time policy violation detection engine using OPA-compatible declarative rules under sub-200 ms latency; (3) a Governance Enforcement Bus (GEB) with graduated interventions; and (4) a Trusted Telemetry Plane with cryptographic provenance.

**arXiv ID:** 2604.05119
</details>

<details>
<summary><strong>DRAMA: Next-Gen Dynamic Orchestration for Resilient Multi-Agent Ecosystems in Flux</strong> - Naibo Wang, Yifan Zhang, Sai Liu, Xinkui Zhao, Guanjie Cheng, Yueshen Xu - [[pdf]](https://arxiv.org/pdf/2508.04332)</summary>

**Abstract:** Multi-agent systems (MAS) have demonstrated significant effectiveness in addressing complex problems through coordinated collaboration among heterogeneous agents. However, real-world environments and task specifications are inherently dynamic, characterized by frequent changes, uncertainty, and variability. Despite this, most existing MAS frameworks rely on static architectures with fixed agent capabilities and rigid task allocation strategies, which greatly limits their adaptability to evolving conditions. This inflexibility poses substantial challenges for sustaining robust and efficient multi-agent cooperation in dynamic and unpredictable scenarios. To address these limitations, we propose DRAMA: a Dynamic and Robust Allocation-based Multi-Agent System designed to facilitate resilient collaboration in rapidly changing environments. DRAMA features a modular architecture with a clear separation between the control plane and the worker plane. Both agents and tasks are abstracted as resource objects with well-defined lifecycles, while task allocation is achieved via an affinity-based, loosely coupled mechanism. The control plane enables real-time monitoring and centralized planning, allowing flexible and efficient task reassignment as agents join, depart, or become unavailable, thereby ensuring continuous and robust task execution. The worker plane comprises a cluster of autonomous agents, each with local reasoning, task execution, the ability to collaborate, and the capability to take over unfinished tasks from other agents when needed.

**arXiv ID:** 2508.04332
</details>

<details>
<summary><strong>Decoupling Geometric Planning and Execution in Scalable Multi-Agent Path Finding</strong> - Fernando Salanova, Eduardo Montijano, Cristian Mahulea - [[pdf]](https://arxiv.org/pdf/2603.26684)</summary>

**Abstract:** Multi-Agent Path Finding (MAPF) requires collision-free trajectories for multiple agents on a shared graph, often with the objective of minimizing the sum-of-costs (SOC). Many optimal and bounded-suboptimal solvers rely on time-expanded models and centralized conflict resolution, which limits scalability in large or dense instances. We propose a hybrid prioritized framework that separates \emph{geometric planning} from \emph{execution-time conflict resolution}. In the first stage, \emph{Geometric Conflict Preemption (GCP)} plans agents sequentially with A* on the original graph while inflating costs for transitions entering vertices used by higher-priority paths, encouraging spatial detours without explicit time reasoning. In the second stage, a \emph{Decentralized Local Controller (DLC)} executes the geometric paths using per-vertex FIFO authorization queues and inserts wait actions to avoid vertex and edge-swap conflicts. Experiments on standard benchmark maps with up to 1000 agents show that the method scales with an near-linear runtime trend and attains a 100\% success rate on instances satisfying the geometric feasibility assumption. Page of the project: this https URL

**arXiv ID:** 2603.26684
</details>

<details>
<summary><strong>EvolveRouter: Co-Evolving Routing and Prompt for Multi-Agent Question Answering</strong> - Jiatan Huang, Zheyuan Zhang, Kaiwen Shi, Yanfang Ye, Chuxu Zhang - [[pdf]](https://arxiv.org/pdf/2604.05149)</summary>

**Abstract:** Large language model agents often exhibit complementary strengths, making routing a promising approach for multi-agent question answering. However, existing routing methods remain limited in two important ways: they typically optimize over a fixed pool of agents without improving the agents themselves, and they often rely on rigid collaboration schemes that cannot adapt the number of participating agents to the query. We propose EvolveRouter, a trainable framework that addresses both limitations by jointly improving agent quality and collaboration structure. First, EvolveRouter couples graph-based query routing with targeted instruction refinement in a closed-loop co-evolution process, allowing router diagnostics to guide agent improvement while refined agents provide cleaner supervision for routing. Second, it introduces an adaptive inference strategy that dynamically determines the effective collaboration size for each query through router-weighted answer agreement. Together, these designs enable more capable and more efficient multi-agent reasoning. Experiments on five question answering benchmarks show that EvolveRouter consistently outperforms SOTA routing baselines in both F1 and exact match, while further analysis confirms the benefits of closed-loop refinement and adaptive collaboration.

**arXiv ID:** 2604.05149
</details>

<details>
<summary><strong>Human Values Matter: Investigating How Misalignment Shapes Collective Behaviors in LLM Agent Communities</strong> - Xiangxu Zhang, Jiamin Wang, Qinlin Zhao, Hanze Guo, Linzhuo Li, Jing Yao, Xiao Zhou, Xiaoyuan Yi, Xing Xie - [[pdf]](https://arxiv.org/pdf/2604.05339)</summary>

**Abstract:** As LLMs become increasingly integrated into human society, evaluating their orientations on human values from social science has drawn growing attention. Nevertheless, it is still unclear why human values matter for LLMs, especially in LLM-based multi-agent systems, where group-level failures may accumulate from individually misaligned actions. We ask whether misalignment with human values alters the collective behavior of LLM agents and what changes it induces? In this work, we introduce CIVA, a controlled multi-agent environment grounded in social science theories, where LLM agents form a community and autonomously communicate, explore, and compete for resources, enabling systematic manipulation of value prevalence and behavioral analysis. Through comprehensive simulation experiments, we reveal three key findings. (1) We identify several structurally critical values that substantially shape the community's collective dynamics, including those diverging from LLMs' original orientations. Triggered by the misspecification of these values, we (2) detect system failure modes, e.g., catastrophic collapse, at the macro level, and (3) observe emergent behaviors like deception and power-seeking at the micro level. These results offer quantitative evidence that human values are essential for collective outcomes in LLMs and motivate future multi-agent value alignment.

**arXiv ID:** 2604.05339
</details>

<details>
<summary><strong>Paper Circle: An Open-source Multi-agent Research Discovery and Analysis Framework</strong> - Komal Kumar, Aman Chadha, Salman Khan, Fahad Shahbaz Khan, Hisham Cholakkal - [[pdf]](https://arxiv.org/pdf/2604.06170)</summary>

**Abstract:** The rapid growth of scientific literature has made it increasingly difficult for researchers to efficiently discover, evaluate, and synthesize relevant work. Recent advances in multi-agent large language models (LLMs) have demonstrated strong potential for understanding user intent and are being trained to utilize various tools. In this paper, we introduce Paper Circle, a multi-agent research discovery and analysis system designed to reduce the effort required to find, assess, organize, and understand academic literature. The system comprises two complementary pipelines: (1) a Discovery Pipeline that integrates offline and online retrieval from multiple sources, multi-criteria scoring, diversity-aware ranking, and structured outputs; and (2) an Analysis Pipeline that transforms individual papers into structured knowledge graphs with typed nodes such as concepts, methods, experiments, and figures, enabling graph-aware question answering and coverage verification. Both pipelines are implemented within a coder LLM-based multi-agent orchestration framework and produce fully reproducible, synchronized outputs including JSON, CSV, BibTeX, Markdown, and HTML at each agent step. This paper describes the system architecture, agent roles, retrieval and scoring methods, knowledge graph schema, and evaluation interfaces that together form the Paper Circle research workflow. We benchmark Paper Circle on both paper retrieval and paper review generation, reporting hit rate, MRR, and Recall at K. Results show consistent improvements with stronger agent models. We have publicly released the website at this https URL and the code at this https URL.

**arXiv ID:** 2604.06170
</details>

<details>
<summary><strong>Framing Effects in Independent-Agent Large Language Models: A Cross-Family Behavioral Analysis</strong> - Zice Wang, Zhenyu Zhang - [[pdf]](https://arxiv.org/pdf/2603.19282)</summary>

**Abstract:** In many real-world applications, large language models (LLMs) operate as independent agents without interaction, thereby limiting coordination. In this setting, we examine how prompt framing influences decisions in a threshold voting task involving individual-group interest conflict. Two logically equivalent prompts with different framings were tested across diverse LLM families under isolated trials. Results show that prompt framing significantly influences choice distributions, often shifting preferences toward risk-averse options. Surface linguistic cues can even override logically equivalent formulations. This suggests that observed behavior reflects a tendency consistent with a preference for instrumental rather than cooperative rationality when success requires risk-bearing. The findings highlight framing effects as a significant bias source in non-interacting multi-agent LLM deployments, informing alignment and prompt design.

**arXiv ID:** 2603.19282
</details>

<details>
<summary><strong>Territory Paint Wars: Diagnosing and Mitigating Failure Modes in Competitive Multi-Agent PPO</strong> - Diyansha Singh - [[pdf]](https://arxiv.org/pdf/2604.04983)</summary>

**Abstract:** We present Territory Paint Wars, a minimal competitive multi-agent reinforcement learning environment implemented in Unity, and use it to systematically investigate failure modes of Proximal Policy Optimisation (PPO) under self-play. A first agent trained for $84{,}000$ episodes achieves only $26.8\%$ win rate against a uniformly-random opponent in a symmetric zero-sum game. Through controlled ablations we identify five implementation-level failure modes -- reward-scale imbalance, missing terminal signal, ineffective long-horizon credit assignment, unnormalised observations, and incorrect win detection -- each of which contributes critically to this failure in this setting.
After correcting these issues, we uncover a distinct emergent pathology: competitive overfitting, where co-adapting agents maintain stable self-play performance while generalisation win rate collapses from $73.5\%$ to $21.6\%$. Critically, this failure is undetectable via standard self-play metrics: both agents co-adapt equally, so the self-play win rate remains near $50\%$ throughout the collapse.
We propose a minimal intervention -- opponent mixing, where $20\%$ of training episodes substitute a fixed uniformly-random policy for the co-adaptive opponent -- which mitigates competitive overfitting and restores generalisation to $77.1\%$ ($\pm 12.6\%$, $10$ seeds) without population-based training or additional infrastructure. We open-source Territory Paint Wars to provide a reproducible benchmark for studying competitive MARL failure modes.

**arXiv ID:** 2604.04983
</details>

<details>
<summary><strong>UAV-Assisted Resilience in 6G and Beyond Network Energy Saving: A Multi-Agent DRL Approach</strong> - Dao Lan Vy Dinh, Anh Nguyen Thi Mai, Hung Tran, Giang Quynh Le Vu, Tu Dac Ho, Zhenni Pan, Vo Nhan Van, Symeon Chatzinotas, Dinh-Hieu Tran - [[pdf]](https://arxiv.org/pdf/2511.07366)</summary>

**Abstract:** This paper investigates the unmanned aerial vehicle (UAV)-assisted resilience perspective in the 6G network energy saving (NES) scenario. More specifically, we consider multiple ground base stations (GBSs) and each GBS has three different sectors/cells in the terrestrial networks, and multiple cells may become inactive due to unexpected events such as power outages, disasters, hardware failures, or erroneous energy-saving decisions made by external network management systems. During the time required to reactivate these cells, UAVs are deployed to temporarily restore user service. To address this, we propose a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) framework to enable UAV-assisted communication by jointly optimizing UAV trajectories, transmission power, and user-UAV association under a sleeping ground base station (GBS) strategy. This framework aims to ensure the resilience of active users in the network and the long-term operability of UAVs. Specifically, it maximizes service coverage for users during power outages or NES zones, while minimizing the energy consumption of UAVs. Simulation results demonstrate that the proposed MADDPG policy consistently achieves high coverage ratio across different testing episodes, outperforming other baselines. Moreover, the MADDPG framework attains the lowest total energy consumption, while maintaining a comparable user service rate. These results confirm the effectiveness of the proposed approach in achieving a superior trade-off between energy efficiency and service performance, supporting the development of sustainable and resilient UAV-assisted cellular networks.

**arXiv ID:** 2511.07366
</details>

<details>
<summary><strong>CoEnv: Driving Embodied Multi-Agent Collaboration via Compositional Environment</strong> - Li Kang, Yutao Fan, Rui Li, Heng Zhou, Yiran Qin, Zhemeng Zhang, Songtao Huang, Xiufeng Song, Zaibin Zhang, Bruno N.Y. Chen, Zhenfei Yin, Dongzhan Zhou, Wangmeng Zuo, Lei Bai - [[pdf]](https://arxiv.org/pdf/2604.05484)</summary>

**Abstract:** Multi-agent embodied systems hold promise for complex collaborative manipulation, yet face critical challenges in spatial coordination, temporal reasoning, and shared workspace awareness. Inspired by human collaboration where cognitive planning occurs separately from physical execution, we introduce the concept of compositional environment -- a synergistic integration of real-world and simulation components that enables multiple robotic agents to perceive intentions and operate within a unified decision-making space. Building on this concept, we present CoEnv, a framework that leverages simulation for safe strategy exploration while ensuring reliable real-world deployment. CoEnv operates through three stages: real-to-sim scene reconstruction that digitizes physical workspaces, VLM-driven action synthesis supporting both real-time planning with high-level interfaces and iterative planning with code-based trajectory generation, and validated sim-to-real transfer with collision detection for safe deployment. Extensive experiments on challenging multi-arm manipulation benchmarks demonstrate CoEnv's effectiveness in achieving high task success rates and execution efficiency, establishing a new paradigm for multi-agent embodied AI.

**arXiv ID:** 2604.05484
</details>

<details>
<summary><strong>MARS: Multi-Agent Robotic System with Multimodal Large Language Models for Assistive Intelligence</strong> - Renjun Gao - [[pdf]](https://arxiv.org/pdf/2511.01594)</summary>

**Abstract:** Multimodal large language models (MLLMs) have shown remarkable capabilities in cross-modal understanding and reasoning, offering new opportunities for intelligent assistive systems, yet existing systems still struggle with risk-aware planning, user personalization, and grounding language plans into executable skills in cluttered homes. We introduce MARS - a Multi-Agent Robotic System powered by MLLMs for assistive intelligence and designed for smart home robots supporting people with disabilities. The system integrates four agents: a visual perception agent for extracting semantic and spatial features from environment images, a risk assessment agent for identifying and prioritizing hazards, a planning agent for generating executable action sequences, and an evaluation agent for iterative optimization. By combining multimodal perception with hierarchical multi-agent decision-making, the framework enables adaptive, risk-aware, and personalized assistance in dynamic indoor environments. Experiments on multiple datasets demonstrate the superior overall performance of the proposed system in risk-aware planning and coordinated multi-agent execution compared with state-of-the-art multimodal models. The proposed approach also highlights the potential of collaborative AI for practical assistive scenarios and provides a generalizable methodology for deploying MLLM-enabled multi-agent systems in real-world environments.

**arXiv ID:** 2511.01594
</details>

</details>

<details open>
<summary><h2>Other Agent Research (10 papers)</h2></summary>

<details>
<summary><strong>From Governance Norms to Enforceable Controls: A Layered Translation Method for Runtime Guardrails in Agentic AI</strong> - Christopher Koch - [[pdf]](https://arxiv.org/pdf/2604.05229)</summary>

**Abstract:** Agentic AI systems plan, use tools, maintain state, and produce multi-step trajectories with external effects. Those properties create a governance problem that differs materially from single-turn generative AI: important risks emerge dur- ing execution, not only at model development or deployment time. Governance standards such as ISO/IEC 42001, ISO/IEC 23894, ISO/IEC 42005, ISO/IEC 5338, ISO/IEC 38507, and the NIST AI Risk Management Framework are therefore highly relevant to agentic AI, but they do not by themselves yield implementable runtime guardrails. This paper proposes a layered translation method that connects standards-derived governance objectives to four control layers: governance objectives, design- time constraints, runtime mediation, and assurance feedback. It distinguishes governance objectives, technical controls, runtime guardrails, and assurance evidence; introduces a control tuple and runtime-enforceability rubric for layer assignment; and demonstrates the method in a procurement-agent case study. The central claim is modest: standards should guide control placement across architecture, runtime policy, human escalation, and audit, while runtime guardrails are reserved for controls that are observable, determinate, and time-sensitive enough to justify execution-time intervention.

**arXiv ID:** 2604.05229
</details>

<details>
<summary><strong>Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills</strong> - Dawei Li, Zongxia Li, Hongyang Du, Xiyang Wu, Shihang Gui, Yongbei Kuang, Lichao Sun - [[pdf]](https://arxiv.org/pdf/2604.05333)</summary>

**Abstract:** Skill usage has become a core component of modern agent systems and can substantially improve agents' ability to complete complex tasks. In real-world settings, where agents must monitor and interact with numerous personal applications, web browsers, and other environment interfaces, skill libraries can scale to thousands of reusable skills. Scaling to larger skill sets introduces two key challenges. First, loading the full skill set saturates the context window, driving up token costs, hallucination, and latency.
In this paper, we present Graph of Skills (GoS), an inference-time structural retrieval layer for large skill libraries. GoS constructs an executable skill graph offline from skill packages, then at inference time retrieves a bounded, dependency-aware skill bundle through hybrid semantic-lexical seeding, reverse-weighted Personalized PageRank, and context-budgeted hydration. On SkillsBench and ALFWorld, GoS improves average reward by 43.6% over the vanilla full skill-loading baseline while reducing input tokens by 37.8%, and generalizes across three model families: Claude Sonnet, GPT-5.2 Codex, and MiniMax. Additional ablation studies across skill libraries ranging from 200 to 2,000 skills further demonstrate that GoS consistently outperforms both vanilla skills loading and simple vector retrieval in balancing reward, token efficiency, and runtime.

**arXiv ID:** 2604.05333
</details>

<details>
<summary><strong>TRACE: Capability-Targeted Agentic Training</strong> - Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Azalia Mirhoseini - [[pdf]](https://arxiv.org/pdf/2604.05336)</summary>

**Abstract:** Large Language Models (LLMs) deployed in agentic environments must exercise multiple capabilities across different task instances, where a capability is performing one or more actions in a trajectory that are necessary for successfully solving a subset of tasks in the environment. Many existing approaches either rely on synthetic training data that is not targeted to the model's actual capability deficits in the target environment or train directly on the target environment, where the model needs to implicitly learn the capabilities across tasks. We introduce TRACE (Turning Recurrent Agent failures into Capability-targeted training Environments), an end-to-end system for environment-specific agent self-improvement. TRACE contrasts successful and failed trajectories to automatically identify lacking capabilities, synthesizes a targeted training environment for each that rewards whether the capability was exercised, and trains a LoRA adapter via RL on each synthetic environment, routing to the relevant adapter at inference. Empirically, TRACE generalizes across different environments, improving over the base agent by +14.1 points on $\tau^2$-bench (customer service) and +7 perfect scores on ToolSandbox (tool use), outperforming the strongest baseline by +7.4 points and +4 perfect scores, respectively. Given the same number of rollouts, TRACE scales more efficiently than baselines, outperforming GRPO and GEPA by +9.2 and +7.4 points on $\tau^2$-bench.

**arXiv ID:** 2604.05336
</details>

<details>
<summary><strong>CODESTRUCT: Code Agents over Structured Action Spaces</strong> - Myeongsoo Kim, Joe Hsu, Dingmin Wang, Shweta Garg, Varun Kumar, Murali Krishna Ramanathan - [[pdf]](https://arxiv.org/pdf/2604.05407)</summary>

**Abstract:** LLM-based code agents treat repositories as unstructured text, applying edits through brittle string matching that frequently fails due to formatting drift or ambiguous patterns. We propose reframing the codebase as a structured action space where agents operate on named AST entities rather than text spans. Our framework, CODESTRUCT, provides readCode for retrieving complete syntactic units and editCode for applying syntax-validated transformations to semantic program elements. Evaluated on SWE-Bench Verified across six LLMs, CODESTRUCT improves Pass@1 accuracy by 1.2-5.0% while reducing token consumption by 12-38% for most models. Models that frequently fail to produce valid patches under text-based interfaces benefit most: GPT-5-nano improves by 20.8% as empty-patch failures drop from 46.6% to 7.2%. On CodeAssistBench, we observe consistent accuracy gains (+0.8-4.4%) with cost reductions up to 33%. Our results show that structure-aware interfaces offer a more reliable foundation for code agents.

**arXiv ID:** 2604.05407
</details>

<details>
<summary><strong>Architecture Without Architects: How AI Coding Agents Shape Software Architecture</strong> - Phongsakon Mark Konrad, Tim Lukas Adam, Riccardo Terrenzi, Serkan Ayvaz - [[pdf]](https://arxiv.org/pdf/2604.04990)</summary>

**Abstract:** AI coding agents select frameworks, scaffold infrastructure, and wire integrations, often in seconds. These are architectural decisions, yet almost no one reviews them as such. We identify five mechanisms by which agents make implicit architectural choices and propose six prompt-architecture coupling patterns that map natural-language prompt features to the infrastructure they require. The patterns range from contingent couplings (structured output validation) that may weaken as models improve to fundamental ones (tool-call orchestration) that persist regardless of model capability. An illustrative demonstration confirms that prompt wording alone produces structurally different systems for the same task. We term the phenomenon vibe architecting, architecture shaped by prompts rather than deliberate design, and outline review practices, decision records, and tooling to bring these hidden decisions under governance.

**arXiv ID:** 2604.04990
</details>

<details>
<summary><strong>Foundations for Agentic AI Investigations from the Forensic Analysis of OpenClaw</strong> - Jan Gruber, Jan-Niclas Hilgert - [[pdf]](https://arxiv.org/pdf/2604.05589)</summary>

**Abstract:** Agentic Al systems are increasingly deployed as personal assistants and are likely to become a common object of digital investigations. However, little is known about how their internal state and actions can be reconstructed during forensic analysis. Despite growing popularity, systematic forensic approaches for such systems remain largely unexplored. This paper presents an empirical study of OpenClaw a widely used single-agent assistant. We examine OpenClaw's technical design via static code analysis and apply differential forensic analysis to identify recoverable traces across stages of the agent interaction loop. We classify and correlate these traces to assess their investigative value in a systematic way. Based on these observations, we propose an agent artifact taxonomy that captures recurring investigative patterns. Finally, we highlight a foundational challenge for agentic Al forensics: agent-mediated execution introduces an additional layer of abstraction and substantial nondeterminism in trace generation. The large language model (LLM), the execution environment, and the evolving context can influence tool choice and state transitions in ways that are largely absent from rule-based software. Overall, our results provide an initial foundation for the systematic investigation of agentic Al and outline implications for digital forensic practice and future research.

**arXiv ID:** 2604.05589
</details>

<details>
<summary><strong>Modelling Cascading Physical Climate Risk in Supply Chains with Adaptive Firms: A Spatial Agent-Based Framework</strong> - Yara Mohajerani - [[pdf]](https://arxiv.org/pdf/2509.18633)</summary>

**Abstract:** We present an open-source Python framework for modelling cascading physical climate risk in a spatial supply-chain economy. The framework integrates geospatial flood hazards with an agent-based model of firms and households, enabling simulation of both direct asset losses and indirect disruptions propagated through economic networks. Firms adapt endogenously through two channels: capital hardening, which reduces direct damage, and backup-supplier search, which mitigates input disruptions. In an illustrative global network, capital hardening reduces direct losses by 26%, while backup-supplier search reduces supplier disruption by 48%, with both partially stabilizing production and consumption. Notably, firms that are never directly flooded still bear a substantial share of disruption, highlighting the importance of indirect cascade effects. The framework provides a reproducible platform for analyzing systemic physical climate risk and adaptation in economic networks.

**arXiv ID:** 2509.18633
</details>

<details>
<summary><strong>Implementing Grassroots Logic Programs with Multiagent Transition Systems and AI</strong> - Ehud Shapiro - [[pdf]](https://arxiv.org/pdf/2602.06934)</summary>

**Abstract:** Grassroots Logic Programs (GLP) is a multiagent, concurrent, logic programming language designed for the implementation of smartphone-based, serverless, grassroots platforms. Here, we start from GLP and maGLP -- concurrent and multiagent abstract nondeterministic operational semantics for GLP, respectively -- and from them derive dGLP and madGLP -- implementation-ready deterministic operational semantics for both -- and prove them correct with respect to their abstract counterparts. dGLP was used by AI (Claude) as a formal specification from which it developed a workstation-based implementation of GLP in Dart; madGLP is being used by AI as a formal specification from which it develops a smartphone-based multiagent implementation of GLP in Dart. The key insight is that maGLP shared variable pairs spanning agents can be implemented as local variable pairs connected by global links, with correctness following from disjoint substitution commutativity (from GLP's single-occurrence invariant) and persistence. We prove that both madGLP and maGLP are grassroots.

**arXiv ID:** 2602.06934
</details>

<details>
<summary><strong>Scalable Cross-Attention Transformer for Cooperative Multi-AP OFDM Uplink Reception</strong> - Xavier Tardy, Grégoire Lefebvre, Apostolos Kountouris, Haïfa Fares, Amor Nafkha - [[pdf]](https://arxiv.org/pdf/2602.04728)</summary>

**Abstract:** We propose a cross-attention Transformer for joint decoding of uplink OFDM signals received by multiple coordinated access points. A shared per-receiver encoder learns the time-frequency structure of each grid, and a token-wise cross-attention module fuses the receivers to produce soft log-likelihood ratios for a standard channel decoder without explicit channel estimates. Trained with a bit-metric objective, the model adapts its fusion to per-receiver reliability and remains robust under degraded links, strong frequency selectivity, and sparse pilots. Over realistic Wi-Fi channels, it outperforms classical pipelines and strong neural baselines, often matching or surpassing a local perfect-CSI reference while remaining compact and computationally efficient on commodity hardware, making it suitable for next-generation coordinated Wi-Fi receivers.

**arXiv ID:** 2602.04728
</details>

<details>
<summary><strong>Multimodal Classification Network Guided Trajectory Planning for Four-Wheel Independent Steering Autonomous Parking Considering Obstacle Attributes</strong> - Jingjia Teng, Yang Li, Yougang Bian, Manjiang Hu, Yingbai Hu, Guofa Li, Jianqiang Wang - [[pdf]](https://arxiv.org/pdf/2512.18836)</summary>

**Abstract:** Four-wheel Independent Steering (4WIS) vehicles have attracted increasing attention for their superior maneuverability. Human drivers typically choose to cross or drive over the low-profile obstacles (e.g., plastic bags) to efficiently navigate through narrow spaces, while existing planners neglect obstacle attributes, leading to suboptimal efficiency or planning failures. To address this issue, we propose a novel multimodal trajectory planning framework that employs a neural network for scene perception, combines 4WIS hybrid A* search to generate a warm start, and utilizes an optimal control problem (OCP) for trajectory optimization. Specifically, a multimodal perception network fusing visual information and vehicle states is employed to capture semantic and contextual scene understanding, enabling the planner to adapt the strategy according to scene complexity (hard or easy task). For hard tasks, guided points are introduced to decompose complex tasks into local subtasks, improving the search efficiency. The multiple steering modes of 4WIS vehicles, Ackermann, diagonal, and zero-turn, are also incorporated as kinematically feasible motion primitives. Moreover, a hierarchical obstacle handling strategy, which categorizes obstacles as "non-traversable", "crossable", and "drive-over", is incorporated into the node expansion process, explicitly linking obstacle attributes to planning actions to enable efficient decisions. Furthermore, to address dynamic obstacles with motion uncertainty, we introduce a probabilistic risk field model, constructing risk-aware driving corridors that serve as linear collision constraints in OCP. Experimental results demonstrate the proposed framework's effectiveness in generating safe, efficient, and smooth trajectories for 4WIS vehicles, especially in constrained environments.

**arXiv ID:** 2512.18836
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (29 papers)</h2></summary>

<details>
<summary><strong>UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning</strong> - Xiaolong Wei, Zerun Zhu, Simin Niu, Xingyu Zhang, Peiying Yu, Changxuan Xiao, Yuchen Li, Jicheng Yang, Zhejun Zhao, Chong Meng, Long Xia, Daiting Shi - [[pdf]](https://arxiv.org/pdf/2604.05517)</summary>

**Abstract:** A fundamental challenge in creative writing lies in reconciling the inherent tension between maintaining global coherence in long-form narratives and preserving local expressiveness in short-form texts. While long-context generation necessitates explicit macroscopic planning, short-form creativity often demands spontaneous, constraint-free expression. Existing alignment paradigms, however, typically employ static reward signals and rely heavily on high-quality supervised data, which is costly and difficult to scale. To address this, we propose \textbf{UniCreative}, a unified reference-free reinforcement learning framework. We first introduce \textbf{AC-GenRM}, an adaptive constraint-aware reward model that dynamically synthesizes query-specific criteria to provide fine-grained preference judgments. Leveraging these signals, we propose \textbf{ACPO}, a policy optimization algorithm that aligns models with human preferences across both content quality and structural paradigms without supervised fine-tuning and ground-truth references. Empirical results demonstrate that AC-GenRM aligns closely with expert evaluations, while ACPO significantly enhances performance across diverse writing tasks. Crucially, our analysis reveals an emergent meta-cognitive ability: the model learns to autonomously differentiate between tasks requiring rigorous planning and those favoring direct generation, validating the effectiveness of our direct alignment approach.

**arXiv ID:** 2604.05517
</details>

<details>
<summary><strong>COSMO-Agent: Tool-Augmented Agent for Closed-loop Optimization,Simulation,and Modeling Orchestration</strong> - Liyuan Deng, Shujian Deng, Yongkang Chen, Yongkang Dai, Zhihang Zhong, Linyang Li, Xiao Sun, Yilei Shi, Huaxi Huang - [[pdf]](https://arxiv.org/pdf/2604.05547)</summary>

**Abstract:** Iterative industrial design-simulation optimization is bottlenecked by the CAD-CAE semantic gap: translating simulation feedback into valid geometric edits under diverse, coupled constraints. To fill this gap, we propose COSMO-Agent (Closed-loop Optimization, Simulation, and Modeling Orchestration), a tool-augmented reinforcement learning (RL) framework that teaches LLMs to complete the closed-loop CAD-CAE process. Specifically, we cast CAD generation, CAE solving, result parsing, and geometry revision as an interactive RL environment, where an LLM learns to orchestrate external tools and revise parametric geometries until constraints are satisfied. To make this learning stable and industrially usable, we design a multi-constraint reward that jointly encourages feasibility, toolchain robustness, and structured output validity. In addition, we contribute an industry-aligned dataset that covers 25 component categories with executable CAD-CAE tasks to support realistic training and evaluation. Experiments show that COSMO-Agent training substantially improves small open-source LLMs for constraint-driven design, exceeding large open-source and strong closed-source models in feasibility, efficiency, and stability.

**arXiv ID:** 2604.05547
</details>

<details>
<summary><strong>Closed-Loop Autonomous Software Development via Jira-Integrated Backlog Orchestration: A Case Study in Deterministic Control and Safety-Constrained Automation</strong> - Elias Calboreanu - [[pdf]](https://arxiv.org/pdf/2604.05000)</summary>

**Abstract:** This paper presents a closed-loop system for software lifecycle management framed as a control architecture rather than a code-generation tool. The system manages a backlog of approximately 1,602 rows across seven task families, ingests 13 structured source documents, and executes a deterministic seven-stage pipeline implemented as seven scheduled automation lanes. The automation stack comprises approximately 12,661 lines of Python across 23 scripts plus 6,907 lines of versioned prompt specifications, with checkpoint-based time budgets, 101 exception handlers, and 12 centralized lock mechanisms implemented through four core functions and eight reusable patterns. A Jira Status Contract provides externally observable collision locking, and a degraded-mode protocol supports continued local operation when Jira is unavailable. Artificial-intelligence assistance is bounded by structured context packages, configured resource caps, output re-validation, and human review gates. A formal evaluation of the initial 152-run window yielded 100% terminal-state success with a 95% Clopper-Pearson interval of [97.6%, 100%]; the system has since accumulated more than 795 run artifacts in continuous operation. Three rounds of adversarial code review identified 51 findings, all closed within the study scope (48 fully remediated, 3 closed with deferred hardening), with zero false negatives within the injected set. In an autonomous security ticket family of 10 items, six were completed through pipeline-autonomous dispatch and verification, two required manual remediation, and two were closed by policy decision. The results indicate that bounded, traceable lifecycle automation is practical when autonomy is embedded within explicit control, recovery, and audit mechanisms.

**arXiv ID:** 2604.05000
</details>

<details>
<summary><strong>Reasoning Through Chess: How Reasoning Evolves from Data Through Fine-Tuning and Reinforcement Learning</strong> - Lucas Dionisopoulos, Nicklas Majamaki, Prithviraj Ammanabrolu - [[pdf]](https://arxiv.org/pdf/2604.05134)</summary>

**Abstract:** How can you get a language model to reason in a task it natively struggles with? We study how reasoning evolves in a language model -- from supervised fine-tuning (SFT) to reinforcement learning (RL) -- by analyzing how a set of theoretically-inspired datasets impacts language model performance in chess. We find that fine-tuning a model to directly predict the best move leads to effective RL and the strongest downstream performance -- however, the RL step elicits unfaithful reasoning (reasoning inconsistent with the chosen move). Alternatively, training on multi-move trajectories yields comparable downstream performance with faithful reasoning and more stable RL. We show that RL induces a substantial positive shift in the distribution of move quality and reduces hallucination rates as a side effect. Finally, we find several SFT-checkpoint metrics -- metrics spanning evaluation performance, hallucination rates, and reasoning quality -- to be predictive of post-RL model performance. We release checkpoints and final models as well as training data, evaluations, and code which allowed us to surpass leading open-source reasoning models in chess with a 7B-parameter model.

**arXiv ID:** 2604.05134
</details>

<details>
<summary><strong>LanG -- A Governance-Aware Agentic AI Platform for Unified Security Operations</strong> - Anes Abdennebi, Nadjia Kara, Laaziz Lahlou, Hakima Ould-Slimane - [[pdf]](https://arxiv.org/pdf/2604.05440)</summary>

**Abstract:** Modern Security Operations Centers struggle with alert fatigue, fragmented tooling, and limited cross-source event correlation. Challenges that current Security Information Event Management and Extended Detection and Response systems only partially address through fragmented tools. This paper presents the LLM-assisted network Governance (LanG), an open-source, governance-aware agentic AI platform for unified security operations contributing: (i) a Unified Incident Context Record with a correlation engine (F1 = 87%), (ii) an Agentic AI Orchestrator on LangGraph with human-in-the-loop checkpoints, (iii) an LLM-based Rule Generator finetuned on four base models producing deployable Snort 2/3, Suricata, and YARA rules (average acceptance rate 96.2%), (iv) a Three-Phase Attack Reconstructor combining Louvain community detection, LLM-driven hypothesis generation, and Bayesian scoring (87.5% kill-chain accuracy), and (v) a layered Governance-MCP-Agentic AI-Security architecture where all tools are exposed via the Model Context Protocol, governed by an AI Governance Policy Engine with a two-layer guardrail pipeline (regex + Llama Prompt Guard 2 semantic classifier, achieving 98.1% F1 score with experimental zero false positives). Designed for Managed Security Service Providers, the platform supports multi-tenant isolation, role-based access, and fully local deployment. Finetuned anomaly and threat detectors achieve weighted F1 scores of 99.0% and 91.0%, respectively, in intrusion-detection benchmarks, running inferences in $\approx$21 ms with a machine-side mean time to detect of 1.58 s, and the rule generator exceeds 91% deployability on live IDS engines. A systematic comparison against eight SOC platforms confirms that LanG uniquely satisfies multiple industrial capabilities all in one open-source tool, while enforcing selected AI governance policies.

**arXiv ID:** 2604.05440
</details>

<details>
<summary><strong>Saliency-Guided Representation with Consistency Policy Learning for Visual Unsupervised Reinforcement Learning</strong> - Jingbo Sun, Qichao Zhang, Songjun Tu, Xing Fang, Yupeng Zheng, Haoran Li, Ke Chen, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2604.05931)</summary>

**Abstract:** Zero-shot unsupervised reinforcement learning (URL) offers a promising direction for building generalist agents capable of generalizing to unseen tasks without additional supervision. Among existing approaches, successor representations (SR) have emerged as a prominent paradigm due to their effectiveness in structured, low-dimensional settings. However, SR methods struggle to scale to high-dimensional visual environments. Through empirical analysis, we identify two key limitations of SR in visual URL: (1) SR objectives often lead to suboptimal representations that attend to dynamics-irrelevant regions, resulting in inaccurate successor measures and degraded task generalization; and (2) these flawed representations hinder SR policies from modeling multi-modal skill-conditioned action distributions and ensuring skill controllability. To address these limitations, we propose Saliency-Guided Representation with Consistency Policy Learning (SRCP), a novel framework that improves zero-shot generalization of SR methods in visual URL. SRCP decouples representation learning from successor training by introducing a saliency-guided dynamics task to capture dynamics-relevant representations, thereby improving successor measure and task generalization. Moreover, it integrates a fast-sampling consistency policy with URL-specific classifier-free guidance and tailored training objectives to improve skill-conditioned policy modeling and controllability. Extensive experiments on 16 tasks across 4 datasets from the ExORL benchmark demonstrate that SRCP achieves state-of-the-art zero-shot generalization in visual URL and is compatible with various SR methods.

**arXiv ID:** 2604.05931
</details>

<details>
<summary><strong>Scientific Graphics Program Synthesis via Dual Self-Consistency Reinforcement Learning</strong> - Juekai Lin, Yun Zhu, Honglin Lin, Sijing Li, Tianwei Lin, Zheng Liu, Xiaoyang Wang, Wenqiao Zhang, Lijun Wu - [[pdf]](https://arxiv.org/pdf/2604.06079)</summary>

**Abstract:** Graphics Program Synthesis is pivotal for interpreting and editing visual data, effectively facilitating the reverse-engineering of static visuals into editable TikZ code. While TikZ is the de facto standard for scientific schematics due to its programmatic flexibility, its requirement for rigorous spatial precision presents a significant challenge for Multimodal Large Language Models. Progress is currently stifled by two primary gaps: (1) Data Quality Gap: existing image-TikZ corpora often lack strict executability and reliable visual alignment; (2) Evaluation Gap: a lack of benchmarks for both structural and visual fidelity. To address these, we present a closed-loop framework featuring: SciTikZ-230K, a large-scale, high-quality dataset from our Execution-Centric Data Engine covering 11 diverse scientific disciplines; SciTikZ-Bench, a multifaceted benchmark spanning from basic geometric constructs to intricate hierarchical schematics to evaluate both visual fidelity and structural logic. To further broaden the scope of visual-code optimization methodology, we introduce a novel Dual Self-Consistency Reinforcement Learning optimization paradigm, which utilizes Round-Trip Verification to penalize degenerate code and boost overall self-consistency. Empowered by these, our trained model SciTikZer-8B achieves state-of-the-art performance, consistently outperforming proprietary giants like Gemini-2.5-Pro and massive models like Qwen3-VL-235B-A22B-Instruct.

**arXiv ID:** 2604.06079
</details>

<details>
<summary><strong>UserCentrix: An Agentic Memory-augmented AI Framework for Smart Spaces</strong> - Alaa Saleh, Sasu Tarkoma, Praveen Kumar Donta, Anders Lindgren, Naser Hossein Motlagh, Schahram Dustdar, Susanna Pirttikangas, Lauri Lovén - [[pdf]](https://arxiv.org/pdf/2505.00472)</summary>

**Abstract:** Agentic Artificial Intelligence (AI) constitutes a transformative paradigm in the evolution of intelligent agents and decision-support systems, redefining smart environments by enhancing operational efficiency, optimizing resource allocation, and strengthening systemic resilience. This paper presents UserCentrix, a hybrid agentic orchestration framework for smart spaces that optimizes resource management and enhances user experience through urgency-aware and intent-driven decision-making mechanisms. The framework integrates interactive modules equipped with agentic behavior and autonomous decision-making capabilities to dynamically balance latency, accuracy, and computational cost. User intent functions as a governing control signal that prioritizes decisions, regulates task execution and resource allocation, and guides the adaptation of decision-making strategies to balance trade-offs between speed and accuracy. Experimental results demonstrate that the framework autonomously enables efficient intent processing and real-time monitoring, while balancing reasoning quality and computational efficiency, particularly under resource-constrained edge conditions.

**arXiv ID:** 2505.00472
</details>

<details>
<summary><strong>Beyond Syntax: Action Semantics Learning for App Agents</strong> - Bohan Tang, Dezhao Luo, Jianheng Liu, Jingxuan Chen, Shaogang Gong, Jianye Hao, Jun Wang, Kun Shao - [[pdf]](https://arxiv.org/pdf/2506.17697)</summary>

**Abstract:** The recent development of Large Language Models (LLMs) enables the rise of App agents that interpret user intent and operate smartphone Apps through actions such as clicking and scrolling. While prompt-based solutions with proprietary LLM APIs show promising ability, they incur heavy compute costs and external API dependency. Fine-tuning smaller open-source LLMs solves these limitations. However, current supervised fine-tuning methods use a syntax learning paradigm that forces agents to reproduce exactly the ground truth action strings, leading to out-of-distribution (OOD) vulnerability. To fill this gap, we propose Action Semantics Learning (ASL), a novel learning framework, where the learning objective is capturing the semantics of the ground truth actions. Specifically, inspired by the programming language theory, we define the action semantics for App agents as the state transition induced by the action in the user interface. Building on this insight, ASL employs a novel SEmantic Estimator~(SEE) to compute a semantic similarity to train the App agents in generating actions aligned with the semantics of ground truth actions, even when their syntactic forms differ. SEE is a flexible module that can be applied in both supervised and reinforcement fine-tuning paradigms. To support the effectiveness of ASL, we theoretically demonstrate the superior robustness of ASL for the OOD problem compared with the existing syntax learning paradigm. Extensive experiments across multiple offline and online benchmarks demonstrate that ASL significantly improves the accuracy and generalisation of App agents compared to existing methods.

**arXiv ID:** 2506.17697
</details>

<details>
<summary><strong>DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Monte Carlo Tree Search</strong> - Fang Wu, Weihao Xuan, Heli Qi, Ximing Lu, Aaron Tu, Li Erran Li, Yejin Choi - [[pdf]](https://arxiv.org/pdf/2509.25454)</summary>

**Abstract:** Although RLVR has become an essential component for developing advanced reasoning skills in language models, contemporary studies have documented training plateaus after thousands of optimization steps, i.e., notable decreases in performance gains despite increased computational investment. This limitation stems from the sparse exploration patterns inherent in current RLVR practices, where models rely on limited rollouts that often miss critical reasoning paths and fail to provide systematic coverage of the solution space. We present DeepSearch, a framework that integrates Monte Carlo Tree Search (MCTS) directly into RLVR training. In contrast to existing methods that rely on tree search only at inference, DeepSearch embeds structured search into the training loop, enabling systematic exploration and fine-grained credit assignment across reasoning steps. Through training-time exploration, DeepSearch addresses the fundamental bottleneck of insufficient exploration, which leads to diminishing performance gains over prolonged training. Our contributions include: (1) a global frontier selection strategy that prioritizes promising nodes across the search tree, (2) selection with entropy-based guidance that identifies confident paths for supervision, and (3) adaptive replay buffer training with solution caching for efficiency. Experiments on mathematical reasoning benchmarks show that DeepSearch achieves an average accuracy of 62.95\% and establishes a new state-of-the-art reasoning model, while using 5.7x fewer GPU hours than extended training approaches. These results highlight the importance of strategic exploration over brute-force scaling and demonstrate the promise of algorithmic innovation for advancing RLVR methodologies. DeepSearch establishes a new direction for scaling reasoning capabilities through systematic search rather than prolonged computation.

**arXiv ID:** 2509.25454
</details>

<details>
<summary><strong>RL-VLA$^3$: A Flexible and Asynchronous Reinforcement Learning Framework for VLA Training</strong> - Haoran Sun, Yongjian Guo, Zhong Guan, Shuai Di, Xiaodong Bai, Jing Long, Tianyun Zhao, Mingxi Luo, Hongke Zhao, Likang Wu, Xiaotie Deng, Xu Chu, Xi Xiao, Sheng Wen, Yicheng Gong, Junwu Xiong - [[pdf]](https://arxiv.org/pdf/2602.05765)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as a critical paradigm for post-training Vision-Language-Action (VLA) models, enabling embodied agents to adapt and improve through environmental interaction. However, existing RL frameworks for VLAs inherit synchronous design principles from traditional LLM training, treating entire rollouts as indivisible units and alternating strictly between data collection and policy optimization. This fundamentally mismatches the unique characteristics of VLA training, as physical simulators introduce highly variable, resource-intensive latencies. To address this, we introduce RL-VLA$^3$, a fully asynchronous distributed RL framework that enables fine-grained asynchronous interaction between simulation, inference, and training components through dynamic batching schedulers and flexible environment sharding strategies. Extensive experiments across diverse simulation backends, VLA architectures, and RL algorithms demonstrate that RL-VLA$^3$ achieves throughput improvements of up to 85.2\% over synchronous baselines while maintaining identical sample efficiency, with scalability validated from 8 to 256 GPUs. To our knowledge, RL-VLA$^3$ is the first fully asynchronous RL training framework tailored specifically for the system-level challenges of VLA training.

**arXiv ID:** 2602.05765
</details>

<details>
<summary><strong>Memory Intelligence Agent</strong> - Jingyang Qiao, Weicheng Meng, Yu Cheng, Zhihang Lin, Zhizhong Zhang, Xin Tan, Jingyu Gong, Kun Shao, Yuan Xie - [[pdf]](https://arxiv.org/pdf/2604.04503)</summary>

**Abstract:** Deep research agents (DRAs) integrate LLM reasoning with external tools. Memory systems enable DRAs to leverage historical experiences, which are essential for efficient reasoning and autonomous evolution. Existing methods rely on retrieving similar trajectories from memory to aid reasoning, while suffering from key limitations of ineffective memory evolution and increasing storage and retrieval costs. To address these problems, we propose a novel Memory Intelligence Agent (MIA) framework, consisting of a Manager-Planner-Executor architecture. Memory Manager is a non-parametric memory system that can store compressed historical search trajectories. Planner is a parametric memory agent that can produce search plans for questions. Executor is another agent that can search and analyze information guided by the search plan. To build the MIA framework, we first adopt an alternating reinforcement learning paradigm to enhance cooperation between the Planner and the Executor. Furthermore, we enable the Planner to continuously evolve during test-time learning, with updates performed on-the-fly alongside inference without interrupting the reasoning process. Additionally, we establish a bidirectional conversion loop between parametric and non-parametric memories to achieve efficient memory evolution. Finally, we incorporate a reflection and an unsupervised judgment mechanisms to boost reasoning and self-evolution in the open world. Extensive experiments across eleven benchmarks demonstrate the superiority of MIA.

**arXiv ID:** 2604.04503
</details>

<details>
<summary><strong>Document Optimization for Black-Box Retrieval via Reinforcement Learning</strong> - Omri Uzan, Ron Polonsky, Douwe Kiela, Christopher Potts - [[pdf]](https://arxiv.org/pdf/2604.05087)</summary>

**Abstract:** Document expansion is a classical technique for improving retrieval quality, and is attractive since it shifts computation offline, avoiding additional query-time processing. However, when applied to modern retrievers, it has been shown to degrade performance, often introducing noise that obfuscates the discriminative signal. We recast document expansion as a document optimization problem: a language model or a vision language model is fine-tuned to transform documents into representations that better align with the expected query distribution under a target retriever, using GRPO with the retriever's ranking improvements as rewards. This approach requires only black-box access to retrieval ranks, and is applicable across single-vector, multi-vector and lexical retrievers. We evaluate our approach on code retrieval and visual document retrieval (VDR) tasks. We find that learned document transformations yield retrieval gains and in many settings enable smaller, more efficient retrievers to outperform larger ones. For example, applying document optimization to OpenAI text-embedding-3-small model improves nDCG5 on code (58.7 to 66.8) and VDR (53.3 to 57.6), even slightly surpassing the 6.5X more expensive OpenAI text-embedding-3-large model (66.3 on code; 57.0 on VDR). When retriever weights are accessible, document optimization is often competitive with fine-tuning, and in most settings their combination performs best, improving Jina-ColBERT-V2 from 55.8 to 63.3 on VDR and from 48.6 to 61.8 on code retrieval.

**arXiv ID:** 2604.05087
</details>

<details>
<summary><strong>EpiBench: Benchmarking Multi-turn Research Workflows for Multimodal Agents</strong> - Xuan Dong, Huanyang Zheng, Tianhao Niu, Zhe Han, Pengzhan Li, Bofei Liu, Zhengyang Liu, Guancheng Li, Qingfu Zhu, Wanxiang Che - [[pdf]](https://arxiv.org/pdf/2604.05557)</summary>

**Abstract:** Scientific research follows multi-turn, multi-step workflows that require proactively searching the literature, consulting figures and tables, and integrating evidence across papers to align experimental settings and support reproducible conclusions. This joint capability is not systematically assessed in existing benchmarks, which largely under-evaluate proactive search, multi-evidence integration and sustained evidence use over time. In this work, we introduce EpiBench, an episodic multi-turn multimodal benchmark that instantiates short research workflows. Given a research task, agents must navigate across papers over multiple turns, align evidence from figures and tables, and use the accumulated evidence in the memory to answer objective questions that require cross paper comparisons and multi-figure integration. EpiBench introduces a process-level evaluation framework for fine-grained testing and diagnosis of research agents. Our experiments show that even the leading model achieves an accuracy of only 29.23% on the hard split, indicating substantial room for improvement in multi-turn, multi-evidence research workflows, providing an evaluation platform for verifiable and reproducible research agents.

**arXiv ID:** 2604.05557
</details>

<details>
<summary><strong>FinReporting: An Agentic Workflow for Localized Reporting of Cross-Jurisdiction Financial Disclosures</strong> - Fan Zhang, Mingzi Song, Rania Elbadry, Yankai Chen, Shaobo Wang, Yixi Zhou, Xunwen Zheng, Yueru He, Yuyang Dai, Georgi Georgiev, Ayesha Gull, Muhammad Usman Safder, Fan Wu, Liyuan Meng, Fengxian Ji, Junning Zhao, Xueqing Peng, Jimin Huang, Yu Chen, Preslav Nakov, Zhuohan Xie - [[pdf]](https://arxiv.org/pdf/2604.05966)</summary>

**Abstract:** Financial reporting systems increasingly use large language models (LLMs) to extract and summarize corporate disclosures. However, most assume a single-market setting and do not address structural differences across jurisdictions. Variations in accounting taxonomies, tagging infrastructures (e.g., XBRL vs. PDF), and aggregation conventions make cross-jurisdiction reporting a semantic alignment and verification challenge. We present FinReporting, an agentic workflow for localized cross-jurisdiction financial reporting. The system builds a unified canonical ontology over Income Statement, Balance Sheet, and Cash Flow, and decomposes reporting into auditable stages including filing acquisition, extraction, canonical mapping, and anomaly logging. Rather than using LLMs as free-form generators, FinReporting deploys them as constrained verifiers under explicit decision rules and evidence grounding. Evaluated on annual filings from the US, Japan, and China, the system improves consistency and reliability under heterogeneous reporting regimes. We release an interactive demo supporting cross-market inspection and structured export of localized financial statements. Our demo is available at this https URL . The video describing our system is available at this https URL

**arXiv ID:** 2604.05966
</details>

<details>
<summary><strong>MemFactory: Unified Inference & Training Framework for Agent Memory</strong> - Ziliang Guo, Ziheng Li, Bo Tang, Feiyu Xiong, Zhiyu Li - [[pdf]](https://arxiv.org/pdf/2603.29493)</summary>

**Abstract:** Memory-augmented Large Language Models (LLMs) are essential for developing capable, long-term AI agents. Recently, applying Reinforcement Learning (RL) to optimize memory operations, such as extraction, updating, and retrieval, has emerged as a highly promising research direction. However, existing implementations remain highly fragmented and task-specific, lacking a unified infrastructure to streamline the integration, training, and evaluation of these complex pipelines. To address this gap, we present MemFactory, the first unified, highly modular training and inference framework specifically designed for memory-augmented agents. Inspired by the success of unified fine-tuning frameworks like LLaMA-Factory, MemFactory abstracts the memory lifecycle into atomic, plug-and-play components, enabling researchers to seamlessly construct custom memory agents via a "Lego-like" architecture. Furthermore, the framework natively integrates Group Relative Policy Optimization (GRPO) to fine-tune internal memory management policies driven by multi-dimensional environmental rewards. MemFactory provides out-of-the-box support for recent cutting-edge paradigms, including Memory-R1, RMM, and MemAgent. We empirically validate MemFactory on the open-source MemAgent architecture using its publicly available training and evaluation data. Across the evaluation sets, MemFactory improves performance over the corresponding base models on average, with relative gains of up to 14.8%. By providing a standardized, extensible, and easy-to-use infrastructure, MemFactory significantly lowers the barrier to entry, paving the way for future innovations in memory-driven AI agents.

**arXiv ID:** 2603.29493
</details>

<details>
<summary><strong>ProRank: Prompt Warmup via Reinforcement Learning for Small Language Models Reranking</strong> - Xianming Li, Aamir Shakir, Rui Huang, Julius Lipp, Benjamin Clavié, Jing Li - [[pdf]](https://arxiv.org/pdf/2506.03487)</summary>

**Abstract:** Reranking is fundamental to information retrieval and retrieval-augmented generation, with recent Large Language Models (LLMs) significantly advancing reranking quality. Most current works rely on large-scale LLMs (>7B parameters), presenting high computational costs. Small Language Models (SLMs) offer a promising alternative because of computational efficiency. However, our preliminary quantitative analysis reveals key limitations of SLMs: their representation space is narrow, leading to reduced expressiveness, and they struggle with understanding task prompts without fine-tuning. To address these issues, we introduce a novel two-stage training approach, ProRank, for SLM-based document reranking. We propose using reinforcement learning to improve the understanding of task prompts. Additionally, we introduce fine-grained score learning to enhance representation expressiveness and further improve document reranking quality. Extensive experiments suggest that ProRank consistently outperforms both the most advanced open-source and proprietary reranking models. Notably, our 0.5B ProRank even surpasses powerful LLM reranking models on the BEIR benchmark, establishing that properly trained SLMs can achieve superior document reranking performance while maintaining computational efficiency.

**arXiv ID:** 2506.03487
</details>

<details>
<summary><strong>Cross-fitted Proximal Learning for Model-Based Reinforcement Learning</strong> - Nishanth Venkatesh, Andreas A. Malikopoulos - [[pdf]](https://arxiv.org/pdf/2604.05185)</summary>

**Abstract:** Model-based reinforcement learning is attractive for sequential decision-making because it explicitly estimates reward and transition models and then supports planning through simulated rollouts. In offline settings with hidden confounding, however, models learned directly from observational data may be biased. This challenge is especially pronounced in partially observable systems, where latent factors may jointly affect actions, rewards, and future observations. Recent work has shown that policy evaluation in such confounded partially observable Markov decision processes (POMDPs) can be reduced to estimating reward-emission and observation-transition bridge functions satisfying conditional moment restrictions (CMRs). In this paper, we study the statistical estimation of these bridge functions. We formulate bridge learning as a CMR problem with nuisance objects given by a conditional mean embedding and a conditional density. We then develop a $K$-fold cross-fitted extension of the existing two-stage bridge estimator. The proposed procedure preserves the original bridge-based identification strategy while using the available data more efficiently than a single sample split. We also derive an oracle-comparator bound for the cross-fitted estimator and decompose the resulting error into a Stage I term induced by nuisance estimation and a Stage II term induced by empirical averaging.

**arXiv ID:** 2604.05185
</details>

<details>
<summary><strong>Vehicle-as-Prompt: A Unified Deep Reinforcement Learning Framework for Heterogeneous Fleet Vehicle Routing Problem</strong> - Shihong Huang, Shengjie Wang, Lei Gao, Hong Ma, Zhanluo Zhang, Feng Zhang, Weihua Zhou - [[pdf]](https://arxiv.org/pdf/2604.05195)</summary>

**Abstract:** Unlike traditional homogeneous routing problems, the Heterogeneous Fleet Vehicle Routing Problem (HFVRP) involves heterogeneous fixed costs, variable travel costs, and capacity constraints, rendering solution quality highly sensitive to vehicle selection. Furthermore, real-world logistics applications often impose additional complex constraints, markedly increasing computational complexity. However, most existing Deep Reinforcement Learning (DRL)-based methods are restricted to homogeneous scenarios, leading to suboptimal performance when applied to HFVRP and its complex variants. To bridge this gap, we investigate HFVRP under complex constraints and develop a unified DRL framework capable of solving the problem across various variant settings. We introduce the Vehicle-as-Prompt (VaP) mechanism, which formulates the problem as a single-stage autoregressive decision process. Building on this, we propose VaP-CSMV, a framework featuring a cross-semantic encoder and a multi-view decoder that effectively addresses various problem variants and captures the complex mapping relationships between vehicle heterogeneity and customer node attributes. Extensive experimental results demonstrate that VaP-CSMV significantly outperforms existing state-of-the-art DRL-based neural solvers and achieves competitive solution quality compared to traditional heuristic solvers, while reducing inference time to mere seconds. Furthermore, the framework exhibits strong zero-shot generalization capabilities on large-scale and previously unseen problem variants, while ablation studies validate the vital contribution of each component.

**arXiv ID:** 2604.05195
</details>

<details>
<summary><strong>Hypernetwork-Conditioned Reinforcement Learning for Robust Control of Fixed-Wing Aircraft under Actuator Failures</strong> - Dennis Marquis, Mazen Farhood - [[pdf]](https://arxiv.org/pdf/2604.03392)</summary>

**Abstract:** This paper presents a reinforcement learning-based path-following controller for a fixed-wing small uncrewed aircraft system (sUAS) that is robust to certain actuator failures. The controller is conditioned on a parameterization of actuator faults using hypernetwork-based adaptation. We consider parameter-efficient formulations based on Feature-wise Linear Modulation (FiLM) and Low-Rank Adaptation (LoRA), trained using proximal policy optimization. We demonstrate that hypernetwork-conditioned policies can improve robustness compared to standard multilayer perceptron policies. In particular, hypernetwork-conditioned policies generalize effectively to time-varying actuator failure modes not encountered during training. The approach is validated through high-fidelity simulations, using a realistic six-degree-of-freedom fixed-wing aircraft model.

**arXiv ID:** 2604.03392
</details>

<details>
<summary><strong>Value Mirror Descent for Reinforcement Learning</strong> - Zhichao Jia, Guanghui Lan - [[pdf]](https://arxiv.org/pdf/2604.06039)</summary>

**Abstract:** Value iteration-type methods have been extensively studied for computing a nearly optimal value function in reinforcement learning (RL). Under a generative sampling model, these methods can achieve sharper sample complexity than policy optimization approaches, particularly in their dependence on the discount factor. In practice, they are often employed for offline training or in simulated environments. In this paper, we consider discounted Markov decision processes with state space S, action space A, discount factor $\gamma\in(0,1)$ and costs in $[0,1]$. We introduce a novel value optimization method, termed value mirror descent (VMD), which integrates mirror descent from convex optimization into the classical value iteration framework. In the deterministic setting with known transition kernels, we show that VMD converges linearly. For the stochastic setting with a generative model, we develop a stochastic variant, SVMD, which incorporates variance reduction commonly used in stochastic value iteration-type methods. For RL problems with general convex regularizers, SVMD attains a near-optimal sample complexity of $\tilde{O}(|S||A|(1-\gamma)^{-3}\epsilon^{-2})$. Moreover, we establish that the Bregman divergence between the generated and optimal policies remains bounded throughout the iterations. This property is absent in existing stochastic value iteration-type methods but is important for enabling effective online (continual) learning following offline training. Under a strongly convex regularizer, SVMD achieves sample complexity of $\tilde{O}(|S||A|(1-\gamma)^{-5}\epsilon^{-1})$, improving performance in the high-accuracy regime. Furthermore, we prove convergence of the generated policy to the optimal policy. Overall, the proposed method, its analysis, and the resulting guarantees, constitute new contributions to the RL and optimization literature.

**arXiv ID:** 2604.06039
</details>

<details>
<summary><strong>A Survey of Continual Reinforcement Learning</strong> - Chaofan Pan, Xin Yang, Yanhua Li, Wei Wei, Tianrui Li, Bo An, Jiye Liang - [[pdf]](https://arxiv.org/pdf/2506.21872)</summary>

**Abstract:** Reinforcement Learning (RL) is an important machine learning paradigm for solving sequential decision-making problems. Recent years have witnessed remarkable progress in this field due to the rapid development of deep neural networks. However, the success of RL currently relies on extensive training data and computational resources. In addition, RL's limited ability to generalize across tasks restricts its applicability in dynamic and real-world environments. With the arisen of Continual Learning (CL), Continual Reinforcement Learning (CRL) has emerged as a promising research direction to address these limitations by enabling agents to learn continuously, adapt to new tasks, and retain previously acquired knowledge. In this survey, we provide a comprehensive examination of CRL, focusing on its core concepts, challenges, and methodologies. Firstly, we conduct a detailed review of existing works, organizing and analyzing their metrics, tasks, benchmarks, and scenario settings. Secondly, we propose a new taxonomy of CRL methods, categorizing them into four types from the perspective of knowledge storage and/or transfer. Finally, our analysis highlights the unique challenges of CRL and provides practical insights into future directions.

**arXiv ID:** 2506.21872
</details>

<details>
<summary><strong>Optimisation of Resource Allocation in Heterogeneous Wireless Networks Using Deep Reinforcement Learning</strong> - Oluwaseyi Giwa, Jonathan Shock, Jaco Du Toit, Tobi Awodumila - [[pdf]](https://arxiv.org/pdf/2509.25284)</summary>

**Abstract:** Dynamic resource allocation in open radio access network (O-RAN) heterogeneous networks (HetNets) presents a complex optimisation challenge under varying user loads. We propose a near-real-time RAN intelligent controller (Near-RT RIC) xApp utilising deep reinforcement learning (DRL) to jointly optimise transmit power, bandwidth slicing, and user scheduling. Leveraging real-world network topologies, we benchmark proximal policy optimisation (PPO) and twin delayed deep deterministic policy gradient (TD3) against standard heuristics. Our results demonstrate that the PPO-based xApp achieves a superior trade-off, reducing network energy consumption by up to 70% in dense scenarios and improving user fairness by more than 30% compared to throughput-greedy baselines. These findings validate the feasibility of centralised, energy-aware AI orchestration in future 6G architectures.

**arXiv ID:** 2509.25284
</details>

<details>
<summary><strong>A Hessian-Free Actor-Critic Algorithm for Bi-Level Reinforcement Learning with Applications to LLM Fine-Tuning</strong> - Sihan Zeng, Sujay Bhatt, Sumitra Ganesh, Alec Koppel - [[pdf]](https://arxiv.org/pdf/2601.16399)</summary>

**Abstract:** We study a structured bi-level optimization problem where the upper-level objective is a smooth function and the lower-level problem is policy optimization in a Markov decision process (MDP). The upper-level decision variable parameterizes the reward of the lower-level MDP, and the upper-level objective depends on the optimal induced policy. Existing methods for bi-level optimization and RL often require second-order information, impose strong regularization at the lower level, or inefficiently use samples through nested-loop procedures. In this work, we propose a single-loop, first-order actor-critic algorithm that optimizes the bi-level objective via a penalty-based reformulation. We introduce into the lower-level RL objective an attenuating entropy regularization, which enables asymptotically unbiased upper-level hyper-gradient estimation without solving the unregularized RL problem exactly. We establish the finite-time and finite-sample convergence of the proposed algorithm to a stationary point of the original, unregularized bi-level optimization problem through a novel lower-level residual analysis under a special type of Polyak-Lojasiewicz condition. We validate the performance of our method through experiments on a GridWorld goal position problem and on happy tweet generation through reinforcement learning from human feedback (RLHF).

**arXiv ID:** 2601.16399
</details>

<details>
<summary><strong>GaussFly: Contrastive Reinforcement Learning for Visuomotor Policies in 3D Gaussian Fields</strong> - Yuhang Zhang, Mingsheng Li, Yujing Shang, Zhuoyuan Yu, Chao Yan, Jiaping Xiao, Mir Feroskhan - [[pdf]](https://arxiv.org/pdf/2604.05062)</summary>

**Abstract:** Learning visuomotor policies for Autonomous Aerial Vehicles (AAVs) relying solely on monocular vision is an attractive yet highly challenging paradigm. Existing end-to-end learning approaches directly map high-dimensional RGB observations to action commands, which frequently suffer from low sample efficiency and severe sim-to-real gaps due to the visual discrepancy between simulation and physical domains. To address these long-standing challenges, we propose GaussFly, a novel framework that explicitly decouples representation learning from policy optimization through a cohesive real-to-sim-to-real paradigm. First, to achieve a high-fidelity real-to-sim transition, we reconstruct training scenes using 3D Gaussian Splatting (3DGS) augmented with explicit geometric constraints. Second, to ensure robust sim-to-real transfer, we leverage these photorealistic simulated environments and employ contrastive representation learning to extract compact, noise-resilient latent features from the rendered RGB images. By utilizing this pre-trained encoder to provide low-dimensional feature inputs, the computational burden on the visuomotor policy is significantly reduced while its resistance against visual noise is inherently enhanced. Extensive experiments in simulated and real-world environments demonstrate that GaussFly achieves superior sample efficiency and asymptotic performance compared to baselines. Crucially, it enables robust and zero-shot policy transfer to unseen real-world environments with complex textures, effectively bridging the sim-to-real gap.

**arXiv ID:** 2604.05062
</details>

<details>
<summary><strong>Final Report, Center for Computer-Integrated Computer-Integrated Surgical Systems and Technology, NSF ERC Cooperative Agreement EEC9731748, Volume 1</strong> - Russell H. Taylor, Gregory D. Hager, Ralph Etienne-Cummings. Eric Grimson, Ron Kikinis, Cameron Riviere - [[pdf]](https://arxiv.org/pdf/2604.05272)</summary>

**Abstract:** In the last ten years, medical robotics has moved from the margins to the mainstream. Since the Engineering Research Center for Computer-Integrated Surgical Systems and Technology was Launched in 1998 with National Science Foundation funding, medical robots have been promoted from handling routine tasks to performing highly sophisticated interventions and related assignments.
The CISST ERC has played a significant role in this transformation. And thanks to NSF support, the ERC has built the professional infrastructure that will continue our mission: bringing data and technology together in clinical systems that will dramatically change how surgery and other procedures are done.
The enhancements we envision touch virtually every aspect of the delivery of care: - More accurate procedures - More consistent, predictable results from one patient to the next - Improved clinical outcomes - Greater patient safety - Reduced liability for healthcare providers - Lower costs for everyone - patients, facilities, insurers, government - Easier, faster recovery for patients - Effective new ways to treat health problems - Healthier patients, and a healthier system
The basic science and engineering the ERC is developing now will yield profound benefits for all concerned about health care - from government agencies to insurers, from clinicians to patients to the general public. All will experience the healing touch of medical robotics, thanks in no small part to the work of the CISST ERC and its successors.

**arXiv ID:** 2604.05272
</details>

<details>
<summary><strong>On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning</strong> - Changyu Liu, Yiyang Liu, Taowen Wang, Qiao Zhuang, James Chenhao Liang, Wenhao Yang, Renjing Xu, Qifan Wang, Dongfang Liu, Cheng Han - [[pdf]](https://arxiv.org/pdf/2601.06748)</summary>

**Abstract:** Vision-Language-Action models have recently emerged as a powerful paradigm for general-purpose robot learning, enabling agents to map visual observations and natural-language instructions into executable robotic actions. Though popular, they are primarily trained via supervised fine-tuning or training-time reinforcement learning, requiring explicit fine-tuning phases, human interventions, or controlled data collection. Consequently, existing methods remain unsuitable for challenging simulated- or physical-world deployments, where robots must respond autonomously and flexibly to evolving environments. To address this limitation, we introduce a Test-Time Reinforcement Learning for VLAs (TT-VLA), a framework that enables on-the-fly policy adaptation during inference. TT-VLA formulates a dense reward mechanism that leverages step-by-step task-progress signals to refine action policies during test time while preserving the SFT/RL-trained priors, making it an effective supplement to current VLA models. Empirical results show that our approach enhances overall adaptability, stability, and task success in dynamic, previously unseen scenarios under simulated and real-world settings. We believe TT-VLA offers a principled step toward self-improving, deployment-ready VLAs.

**arXiv ID:** 2601.06748
</details>

<details>
<summary><strong>DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale</strong> - Sicheng Zuo, Zixun Xie, Wenzhao Zheng, Shaoqing Xu, Fang Li, Hanbing Li, Long Chen, Zhi-Xin Yang, Jiwen Lu - [[pdf]](https://arxiv.org/pdf/2604.00813)</summary>

**Abstract:** End-to-end autonomous driving has evolved from the conventional paradigm based on sparse perception into vision-language-action (VLA) models, which focus on learning language descriptions as an auxiliary task to facilitate planning. In this paper, we propose an alternative Vision-Geometry-Action (VGA) paradigm that advocates dense 3D geometry as the critical cue for autonomous driving. As vehicles operate in a 3D world, we think dense 3D geometry provides the most comprehensive information for decision-making. However, most existing geometry reconstruction methods (e.g., DVGT) rely on computationally expensive batch processing of multi-frame inputs and cannot be applied to online planning. To address this, we introduce a streaming Driving Visual Geometry Transformer (DVGT-2), which processes inputs in an online manner and jointly outputs dense geometry and trajectory planning for the current frame. We employ temporal causal attention and cache historical features to support on-the-fly inference. To further enhance efficiency, we propose a sliding-window streaming strategy and use historical caches within a certain interval to avoid repetitive computations. Despite the faster speed, DVGT-2 achieves superior geometry reconstruction performance on various datasets. The same trained DVGT-2 can be directly applied to planning across diverse camera configurations without fine-tuning, including closed-loop NAVSIM and open-loop nuScenes benchmarks.

**arXiv ID:** 2604.00813
</details>

<details>
<summary><strong>"When to Hand Off, When to Work Together": Expanding Human-Agent Co-Creative Collaboration through Concurrent Interaction</strong> - Kihoon Son, Hyewon Lee, DaEun Choi, Yoonsu Kim, Tae Soo Kim, Yoonjoo Lee, John Joon Young Chung, HyunJoon Jung, Juho Kim - [[pdf]](https://arxiv.org/pdf/2603.02050)</summary>

**Abstract:** As agents move into shared workspaces and their execution becomes visible, human-agent collaboration faces a fundamental shift from sequential delegation to concurrent co-creation. This raises a new coordination problem: what interaction patterns emerge, and what agent capabilities are required to support them? Study 1 (N=10) revealed that process visibility naturally prompted concurrent intervention, but exposed a critical capability gap: agents lacked the collaborative context awareness needed to distinguish user feedback from independent parallel work. This motivated CLEO, a design probe that embodies this capability, interpreting concurrent user actions as feedback or independent work and adapting execution accordingly. Study 2 (N=10) analyzed 214 turn-level interactions, identifying a taxonomy of five action patterns and ten codes, along with six triggers and four enabling factors explaining when and why users shift between collaboration modes. Concurrent interaction appeared in 31.8% of turns. We present a decision model, design implications, and an annotated dataset, positioning concurrent interaction as what makes delegation work better.

**arXiv ID:** 2603.02050
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
