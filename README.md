# Agent arXiv Daily

**Last Updated:** 2026-04-15 04:16:01

**Total Papers:** 52

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (3 papers)</h2></summary>

<details>
<summary><strong>Agentic LLM Reasoning in a Self-Driving Laboratory for Air-Sensitive Lithium Halide Spinel Conductors</strong> - Yuxing Fei, Bernardus Rendy, Xiaochen Yang, Junhee Woo, Xu Huang, Chang Li, Shilong Wang, David Milsted, Yan Zeng, Gerbrand Ceder - [[pdf]](https://arxiv.org/pdf/2604.11957)</summary>

**Abstract:** Self-driving laboratories promise to accelerate materials discovery. Yet current automated solid-state synthesis platforms are limited to ambient conditions, thereby precluding their use for air-sensitive materials. Here, we present A-Lab for Glovebox Powder Solid-state Synthesis (A-Lab GPSS), a robotic platform capable of synthesizing and characterizing air-sensitive inorganic materials under strict air-free conditions. By integrating an agentic AI framework into the A-Lab GPSS platform, we structure autonomous experimental design through abductive and inductive reasoning. We deploy this platform to explore the vast compositional space of lithium halide spinel solid-state ionic conductors. Across a synthesis campaign comprising 352 samples with diverse compositions, the system explores a broad chemical space, experimentally realizing 72% of the 171 possible pairwise combinations among the 19 metals considered in this study. Over the course of the campaign, the fraction of compositions exhibiting both good ionic conductivity (> 0.05 mS/cm) and high halide spinel phase purity increases from 1.33% in the first 75 agent-proposed samples to 5.33% in the final 75. Furthermore, by inspecting the AI's reasoning processes, we reveal distinct yet complementary discovery strategies: abductive reasoning interrogates abnormal observations within already explored regions, whereas inductive reasoning expands the search into broader, previously unvisited chemical space. This work establishes a scalable platform for the autonomous discovery of complex, air-sensitive solid-state materials.

**arXiv ID:** 2604.11957
</details>

<details>
<summary><strong>Mutual Information Surprise: Rethinking Unexpectedness in Autonomous Systems</strong> - Yinsong Wang, Quan Zeng, Xiao Liu, Yu Ding - [[pdf]](https://arxiv.org/pdf/2508.17403)</summary>

**Abstract:** A community of researchers appears to think that a machine can be surprised and have introduced various surprise measures, principally the Shannon Surprise and the Bayesian Surprise. The questions of what constitutes a surprise and how to react to one still elicit debates. In this work, we introduce Mutual Information Surprise (MIS), a new framework that redefines surprise not as anomaly measure, but as a signal of epistemic growth. Furthermore, we develop a statistical test sequence that could trigger a surprise reaction and propose a MIS-based reaction policy that dynamically governs system behavior through sampling adjustment and process forking. Empirical evaluations -- on both synthetic domains and a dynamic pollution map estimation task -- show that a system governed by the MIS-based reaction policy significantly outperforms those under classical surprise-based approaches in stability, responsiveness, and predictive accuracy. The important implication of our new proposal is that MIS quantifies the impact of new observations on mutual information, shifts surprise from reactive to reflective, enables reflection on learning progression, and thus offers a path toward self-aware and adaptive autonomous systems. We expect the new surprise measure to play a critical role in further advancing autonomous systems on their ability to learn and adapt in a complex and dynamic environment.

**arXiv ID:** 2508.17403
</details>

<details>
<summary><strong>Dialogue Agents that Share Family Information to Strengthen Grandparent-Grandchild Relationships</strong> - Seiya Mitsuno, Midori Ban, Hiroshi Ishiguro, Yuichiro Yoshikawa - [[pdf]](https://arxiv.org/pdf/2604.12310)</summary>

**Abstract:** Social isolation among older adults has become a critical concern, as reduced opportunities for conversation and weakened family relationships negatively affect mental health. This study proposes a dialogue agent that supports older adults by fostering both a relationship with the agent and a relationship with their grandchild through sharing everyday information. The agent operates on a chatbot platform and engages in daily conversations with older adults and their grandchildren, exchanging information gathered from each party to enhance conversational engagement and social connection. We conducted a ten-day empirical experiment with 108 grandparent-grandchild pairs. The results suggest that older adults became more willing to interact with the proposed agent, which shared information about their grandchildren, and that the psychological connection between grandparents and grandchildren was strengthened. Furthermore, daily interactions with the agent were associated with reduced anxiety in both older adults and their grandchildren. These findings indicate that a dialogue agent that shares personal information can be an effective approach to supporting older adults by simultaneously offering conversational opportunities and promoting family connectedness. Overall, this study provides valuable insights into the design of dialogue agents that effectively address social isolation among older adults.

**arXiv ID:** 2604.12310
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (13 papers)</h2></summary>

<details>
<summary><strong>AlphaEval: Evaluating Agents in Production</strong> - Pengrui Lu, Bingyu Xu, Wenjun Zhang, Shengjia Hua, Xuanjian Gao, Ranxiang Ge, Lyumanshan Ye, Linxuan Wu, Yiran Li, Junfei Fish Yu, Yibo Zhang, Ruixin Li, Manxiang Li, Xiao Han, Xiaocong Zhou, Guangyao Chi, Zisheng Chen, Kaishen Chen, Kun Wang, Qihua Xu, Fengyue Meng, Yuchen Ni, Jiajun Li, Jinxiu Liu, Danfeng Zhang, Jingru Zhao, Pengfei Liu - [[pdf]](https://arxiv.org/pdf/2604.12162)</summary>

**Abstract:** The rapid deployment of AI agents in commercial settings has outpaced the development of evaluation methodologies that reflect production realities. Existing benchmarks measure agent capabilities through retrospectively curated tasks with well-specified requirements and deterministic metrics -- conditions that diverge fundamentally from production environments where requirements contain implicit constraints, inputs are heterogeneous multi-modal documents with information fragmented across sources, tasks demand undeclared domain expertise, outputs are long-horizon professional deliverables, and success is judged by domain experts whose standards evolve over time. We present AlphaEval, a production-grounded benchmark of 94 tasks sourced from seven companies deploying AI agents in their core business, spanning six O*NET (Occupational Information Network) domains. Unlike model-centric benchmarks, AlphaEval evaluates complete agent products -- Claude Code, Codex, etc. -- as commercial systems, capturing performance variations invisible to model-level evaluation. Our evaluation framework covers multiple paradigms (LLM-as-a-Judge, reference-driven metrics, formal verification, rubric-based assessment, automated UI testing, etc.), with individual domains composing multiple paradigms. Beyond the benchmark itself, we contribute a requirement-to-benchmark construction framework -- a systematic methodology that transforms authentic production requirements into executable evaluation tasks in minimal time. This framework standardizes the entire pipeline from requirement to evaluation, providing a reproducible, modular process that any organization can adopt to construct production-grounded benchmarks for their own domains.

**arXiv ID:** 2604.12162
</details>

<details>
<summary><strong>Thought-Retriever: Don't Just Retrieve Raw Data, Retrieve Thoughts for Memory-Augmented Agentic Systems</strong> - Tao Feng, Pengrui Han, Guanyu Lin, Ge Liu, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2604.12231)</summary>

**Abstract:** Large language models (LLMs) have transformed AI research thanks to their powerful internal capabilities and knowledge. However, existing LLMs still fail to effectively incorporate the massive external knowledge when interacting with the world. Although retrieval-augmented LLMs are proposed to mitigate the issue, they are still fundamentally constrained by the context length of LLMs, as they can only retrieve top-K raw data chunks from the external knowledge base which often consists of millions of data chunks. Here we propose Thought-Retriever, a novel model-agnostic algorithm that helps LLMs generate output conditioned on arbitrarily long external data, without being constrained by the context length or number of retrieved data chunks. Our key insight is to let an LLM fully leverage its intermediate responses generated when solving past user queries (thoughts), filtering meaningless and redundant thoughts, organizing them in thought memory, and retrieving the relevant thoughts when addressing new queries. This effectively equips LLM-based agents with a self-evolving long-term memory that grows more capable through continuous interaction. Besides algorithmic innovation, we further meticulously prepare a novel benchmark, AcademicEval, which requires an LLM to faithfully leverage ultra-long context to answer queries based on real-world academic papers. Extensive experiments on AcademicEval and two other public datasets validate that Thought-Retriever remarkably outperforms state-of-the-art baselines, achieving an average increase of at least 7.6% in F1 score and 16% in win rate across various tasks. More importantly, we further demonstrate two exciting findings: (1) Thought-Retriever can indeed help LLM self-evolve after solving more user queries; (2) Thought-Retriever learns to leverage deeper thoughts to answer more abstract user queries.

**arXiv ID:** 2604.12231
</details>

<details>
<summary><strong>CompliBench: Benchmarking LLM Judges for Compliance Violation Detection in Dialogue Systems</strong> - Jingbo Yang, Guanyu Yao, Bairu Hou, Xinghan Yang, Nikolai Glushnev, Iwona Bialynicka-Birula, Duo Ding, Shiyu Chang - [[pdf]](https://arxiv.org/pdf/2604.12312)</summary>

**Abstract:** As Large Language Models (LLMs) are increasingly deployed as task-oriented agents in enterprise environments, ensuring their strict adherence to complex, domain-specific operational guidelines is critical. While utilizing an LLM-as-a-Judge is a promising solution for scalable evaluation, the reliability of these judges in detecting specific policy violations remains largely unexplored. This gap is primarily due to the lack of a systematic data generation method, which has been hindered by the extensive cost of fine-grained human annotation and the difficulty of synthesizing realistic agent violations. In this paper, we introduce CompliBench, a novel benchmark designed to evaluate the ability of LLM judges to detect and localize guideline violations in multi-turn dialogues. To overcome data scarcity, we develop a scalable, automated data generation pipeline that simulates user-agent interactions. Our controllable flaw injection process automatically yields precise ground-truth labels for the violated guideline and the exact conversation turn, while an adversarial search method ensures these introduced perturbations are highly challenging. Our comprehensive evaluation reveals that current state-of-the-art proprietary LLMs struggle significantly with this task. In addition, we demonstrate that a small-scale judge model fine-tuned on our synthesized data outperforms leading LLMs and generalizes well to unseen business domains, highlighting our pipeline as an effective foundation for training robust generative reward models.

**arXiv ID:** 2604.12312
</details>

<details>
<summary><strong>Cooperative Memory Paging with Keyword Bookmarks for Long-Horizon LLM Conversations</strong> - Ziyang Liu - [[pdf]](https://arxiv.org/pdf/2604.12376)</summary>

**Abstract:** When LLM conversations grow beyond the context window, old content must be evicted -- but how does the model recover it when needed? We propose cooperative paging: evicted segments are replaced with minimal keyword bookmarks ([pN:keywords], ~8-24 tokens each), and the model is given a recall() tool to retrieve full content on demand. On the LoCoMo benchmark (10 real multi-session conversations, 300+ turns), cooperative paging achieves the highest answer quality among six methods -- outperforming truncation, BM25, word-overlap retrieval, a search-tool baseline, and full context -- on four models (GPT-4o-mini, DeepSeek-v3.2, Claude Haiku, GLM-5), confirmed by four independent LLM judges ($p=0.017$, paired bootstrap). We then study the paging design space with a 5x4 ablation over boundary strategies and eviction policies (3,176 synthetic probes, 1,600 LoCoMo probes). Key findings: (1) coarse fixed-size pages (fixed_20) reach 96.7% while content-aware topic_shift collapses to 56.7%; (2) eviction policy choice is data-dependent (FIFO best on synthetic, LFU on LoCoMo); (3) two bookmark generation strategies improve over the heuristic baseline (+4.4 and +8.7 E2E points); (4) the remaining bottleneck is bookmark discrimination -- the model triggers recall() 96% of the time but selects the correct page only 57% when bookmarks are insufficiently distinctive. Keyword specificity alone accounts for a 25 percentage point accuracy difference.

**arXiv ID:** 2604.12376
</details>

<details>
<summary><strong>Toward Autonomous Long-Horizon Engineering for ML Research</strong> - Guoxin Chen, Jie Chen, Lei Chen, Jiale Zhao, Fanzhe Meng, Wayne Xin Zhao, Ruihua Song, Cheng Chen, Ji-Rong Wen, Kai Jia - [[pdf]](https://arxiv.org/pdf/2604.13018)</summary>

**Abstract:** Autonomous AI research has advanced rapidly, but long-horizon ML research engineering remains difficult: agents must sustain coherent progress across task comprehension, environment setup, implementation, experimentation, and debugging over hours or days. We introduce AiScientist, a system for autonomous long-horizon engineering for ML research built on a simple principle: strong long-horizon performance requires both structured orchestration and durable state continuity. To this end, AiScientist combines hierarchical orchestration with a permission-scoped File-as-Bus workspace: a top-level Orchestrator maintains stage-level control through concise summaries and a workspace map, while specialized agents repeatedly re-ground on durable artifacts such as analyses, plans, code, and experimental evidence rather than relying primarily on conversational handoffs, yielding thin control over thick state. Across two complementary benchmarks, AiScientist improves PaperBench score by 10.54 points on average over the best matched baseline and achieves 81.82 Any Medal% on MLE-Bench Lite. Ablation studies further show that File-as-Bus protocol is a key driver of performance, reducing PaperBench by 6.41 points and MLE-Bench Lite by 31.82 points when removed. These results suggest that long-horizon ML research engineering is a systems problem of coordinating specialized work over durable project state, rather than a purely local reasoning problem.

**arXiv ID:** 2604.13018
</details>

<details>
<summary><strong>From Plan to Action: How Well Do Agents Follow the Plan?</strong> - Shuyang Liu, Saman Dehghan, Jatin Ganhotra, Martin Hirzel, Reyhaneh Jabbarvand - [[pdf]](https://arxiv.org/pdf/2604.12147)</summary>

**Abstract:** Agents aspire to eliminate the need for task-specific prompt crafting through autonomous reason-act-observe loops. Still, they are commonly instructed to follow a task-specific plan for guidance, e.g., to resolve software issues following phases for navigation, reproduction, patch, and validation. Unfortunately, it is unknown to what extent agents actually follow such instructed plans. Without such an analysis, determining the extent agents comply with a given plan, it is impossible to assess whether a solution was reached through correct strategic reasoning or through other means, e.g., data contamination or overfitting to a benchmark. This paper presents the first extensive, systematic analysis of plan compliance in programming agents, examining 16,991 trajectories from SWE-agent across four LLMs on SWE-bench Verified and SWE-bench Pro under eight plan variations. Without an explicit plan, agents fall back on workflows internalized during training, which are often incomplete, overfit, or inconsistently applied. Providing the standard plan improves issue resolution, and we observe that periodic plan reminders can mitigate plan violations and improve task success. A subpar plan hurts performance even more than no plan at all. Surprisingly, augmenting a plan with additional task-relevant phases in the early stage can degrade performance, particularly when these phases do not align with the model's internal problem-solving strategy. These findings highlight a research gap: fine-tuning paradigms that teach models to follow instructed plans, rather than encoding task-specific plans in them. This requires teaching models to reason and act adaptively, rather than memorizing workflows.

**arXiv ID:** 2604.12147
</details>

<details>
<summary><strong>Policy-Invisible Violations in LLM-Based Agents</strong> - Jie Wu, Ming Gong - [[pdf]](https://arxiv.org/pdf/2604.12177)</summary>

**Abstract:** LLM-based agents can execute actions that are syntactically valid, user-sanctioned, and semantically appropriate, yet still violate organizational policy because the facts needed for correct policy judgment are hidden at decision time. We call this failure mode policy-invisible violations: cases in which compliance depends on entity attributes, contextual state, or session history absent from the agent's visible context. We present PhantomPolicy, a benchmark spanning eight violation categories with balanced violation and safe-control cases, in which all tool responses contain clean business data without policy metadata. We manually review all 600 model traces produced by five frontier models and evaluate them using human-reviewed trace labels. Manual review changes 32 labels (5.3%) relative to the original case-level annotations, confirming the need for trace-level human review. To demonstrate what world-state-grounded enforcement can achieve under favorable conditions, we introduce Sentinel, an enforcement framework based on counterfactual graph simulation. Sentinel treats every agent action as a proposed mutation to an organizational knowledge graph, performs speculative execution to materialize the post-action world state, and verifies graph-structural invariants to decide Allow/Block/Clarify. Against human-reviewed trace labels, Sentinel substantially outperforms a content-only DLP baseline (68.8% vs. 93.0% accuracy) while maintaining high precision, though it still leaves room for improvement on certain violation categories. These results demonstrate what becomes achievable once policy-relevant world state is made available to the enforcement layer.

**arXiv ID:** 2604.12177
</details>

<details>
<summary><strong>Enhancing Agentic Textual Graph Retrieval with Synthetic Stepwise Supervision</strong> - Ge Chang, Jinbo Su, Jiacheng Liu, Pengfei Yang, Yuhao Shang, Huiwen Zheng, Hongli Ma, Yan Liang, Yuanchun Li, Yunxin Liu - [[pdf]](https://arxiv.org/pdf/2510.03323)</summary>

**Abstract:** Integrating textual graphs into Large Language Models (LLMs) is promising for complex graph-based QA. However, a key bottleneck is retrieving informative yet compact subgraphs that fit the LLM context. Existing retrievers often struggle, relying either on shallow embedding similarity or costly interactive policies that require excessive supervision. To address these challenges, we introduce an agentic textual graph reasoning framework featuring an LLM-based retriever trained with synthetic stepwise supervision. Rather than relying on final answer rewards which often yield sparse and unstable signals, we optimize the retriever by evaluating each step against offline-extracted golden subgraphs. Our approach distills golden subgraphs via a specialized data synthesis pipeline to formulate dense rewards, facilitating a two-stage training scheme that effectively learns the interactive graph exploration policy. Based on extensive experiments on three common datasets in comparison with seven strong baselines, our approach achieves an average improvement of 8.1% in accuracy and 9.7% in F1 score. The advantage is even higher in more complicated multi-hop reasoning tasks. Our code will be open-sourced.

**arXiv ID:** 2510.03323
</details>

<details>
<summary><strong>Interactive ASR: Towards Human-Like Interaction and Semantic Coherence Evaluation for Agentic Speech Recognition</strong> - Peng Wang, Yanqiao Zhu, Zixuan Jiang, Qinyuan Chen, Xingjian Zhao, Xipeng Qiu, Wupeng Wang, Zhifu Gao, Xiangang Li, Kai Yu, Xie Chen - [[pdf]](https://arxiv.org/pdf/2604.09121)</summary>

**Abstract:** Recent years have witnessed remarkable progress in automatic speech recognition (ASR), driven by advances in model architectures and large-scale training data. However, two important aspects remain underexplored. First, Word Error Rate (WER), the dominant evaluation metric for decades, treats all words equally and often fails to reflect the semantic correctness of an utterance at the sentence level. Second, interactive correction-an essential component of human communication-has rarely been systematically studied in ASR research. In this paper, we integrate these two perspectives under an agentic framework for interactive ASR. We propose leveraging LLM-as-a-Judge as a semantic-aware evaluation metric to assess recognition quality beyond token-level accuracy. Furthermore, we design an LLM-driven agent framework to simulate human-like multi-turn interaction, enabling iterative refinement of recognition outputs through semantic feedback. Extensive experiments are conducted on standard benchmarks, including GigaSpeech (English), WenetSpeech (Chinese), the ASRU 2019 code-switching test set. Both objective and subjective evaluations demonstrate the effectiveness of the proposed framework in improving semantic fidelity and interactive correction capability. We will release the code to facilitate future research in interactive and agentic ASR.

**arXiv ID:** 2604.09121
</details>

<details>
<summary><strong>Spatial Atlas: Compute-Grounded Reasoning for Spatial-Aware Research Agent Benchmarks</strong> - Arun Sharma - [[pdf]](https://arxiv.org/pdf/2604.12102)</summary>

**Abstract:** We introduce compute-grounded reasoning (CGR), a design paradigm for spatial-aware research agents in which every answerable sub-problem is resolved by deterministic computation before a language model is asked to generate. Spatial Atlas instantiates CGR as a single Agent-to-Agent (A2A) server that handles two challenging benchmarks: FieldWorkArena, a multimodal spatial question-answering benchmark spanning factory, warehouse, and retail environments, and MLE-Bench, a suite of 75 Kaggle machine learning competitions requiring end-to-end ML engineering. A structured spatial scene graph engine extracts entities and relations from vision descriptions, computes distances and safety violations deterministically, then feeds computed facts to large language models, thereby avoiding hallucinated spatial reasoning. Entropy-guided action selection maximizes information gain per step and routes queries across a three-tier frontier model stack (OpenAI + Anthropic). A self-healing ML pipeline with strategy-aware code generation, a score-driven iterative refinement loop, and a prompt-based leak audit registry round out the system. We evaluate across both benchmarks and show that CGR yields competitive accuracy while maintaining interpretability through structured intermediate representations and deterministic spatial computations.

**arXiv ID:** 2604.12102
</details>

<details>
<summary><strong>FeaXDrive: Feasibility-aware Trajectory-Centric Diffusion Planning for End-to-End Autonomous Driving</strong> - Baoyun Wang, Zhuoren Li, Ming Liu, Xinrui Zhang, Bo Leng, Lu Xiong - [[pdf]](https://arxiv.org/pdf/2604.12656)</summary>

**Abstract:** End-to-end diffusion planning has shown strong potential for autonomous driving, but the physical feasibility of generated trajectories remains insufficiently addressed. In particular, generated trajectories may exhibit local geometric irregularities, violate trajectory-level kinematic constraints, or deviate from the drivable area, indicating that the commonly used noise-centric formulation in diffusion planning is not yet well aligned with the trajectory space where feasibility is more naturally characterized. To address this issue, we propose FeaXDrive, a feasibility-aware trajectory-centric diffusion planning method for end-to-end autonomous driving. The core idea is to treat the clean trajectory as the unified object for feasibility-aware modeling throughout the diffusion process. Built on this trajectory-centric formulation, FeaXDrive integrates adaptive curvature-constrained training to improve intrinsic geometric and kinematic feasibility, drivable-area guidance within reverse diffusion sampling to enhance consistency with the drivable area, and feasibility-aware GRPO post-training to further improve planning performance while balancing trajectory-space feasibility. Experiments on the NAVSIM benchmark show that FeaXDrive achieves strong closed-loop planning performance while substantially improving trajectory-space feasibility. These findings highlight the importance of explicitly modeling trajectory-space feasibility in end-to-end diffusion planning and provide a step toward more reliable and physically grounded autonomous driving planners.

**arXiv ID:** 2604.12656
</details>

<details>
<summary><strong>Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain</strong> - Léo Boisvert, Abhay Puri, Chandra Kiran Reddy Evuru, Nazanin Sepahvand, Nicolas Chapados, Quentin Cappart, Alexandre Lacoste, Krishnamurthy Dj Dvijotham, Alexandre Drouin - [[pdf]](https://arxiv.org/pdf/2510.05159)</summary>

**Abstract:** While finetuning AI agents on interaction data -- such as web browsing or tool use -- improves their capabilities, it also introduces critical security vulnerabilities within the agentic AI supply chain. We show that adversaries can effectively poison the data collection pipeline at multiple stages to embed hard-to-detect backdoors that, when triggered, cause unsafe or malicious behavior. We formalize three realistic threat models across distinct layers of the supply chain: direct poisoning of finetuning data, pre-backdoored base models, and environment poisoning, a novel attack vector that exploits vulnerabilities specific to agentic training pipelines. Evaluated on two widely adopted agentic benchmarks, all three threat models prove effective: poisoning only a small number of demonstrations is sufficient to embed a backdoor that causes an agent to leak confidential user information with over 80\% success.

**arXiv ID:** 2510.05159
</details>

<details>
<summary><strong>TriDeliver: Cooperative Air-Ground Instant Delivery with UAVs, Couriers, and Crowdsourced Ground Vehicles</strong> - Junhui Gao, Yan Pan, Qianru Wang, Wenzhe Hou, Yiqin Deng, Liangliang Jiang, Yuguang Fang - [[pdf]](https://arxiv.org/pdf/2604.09049)</summary>

**Abstract:** Instant delivery, shipping items before critical deadlines, is essential in daily life. While multiple delivery agents, such as couriers, Unmanned Aerial Vehicles (UAVs), and crowdsourced agents, have been widely employed, each of them faces inherent limitations (e.g., low efficiency/labor shortages, flight control, and dynamic capabilities, respectively), preventing them from meeting the surging demands alone. This paper proposes TriDeliver, the first hierarchical cooperative framework, integrating human couriers, UAVs, and crowdsourced ground vehicles (GVs) for efficient instant delivery. To obtain the initial scheduling knowledge for GVs and UAVs as well as improve the cooperative delivery performance, we design a Transfer Learning (TL)-based algorithm to extract delivery knowledge from couriers' behavioral history and transfer their knowledge to UAVs and GVs with fine-tunings, which is then used to dispatch parcels for efficient delivery. Evaluated on one-month real-world trajectory and delivery datasets, it has been demonstrated that 1) by integrating couriers, UAVs, and crowdsourced GVs, TriDeliver reduces the delivery cost by $65.8\%$ versus state-of-the-art cooperative delivery by UAVs and couriers; 2) TriDeliver achieves further improvements in terms of delivery time ($-17.7\%$), delivery cost ($-9.8\%$), and impacts on original tasks of crowdsourced GVs ($-43.6\%$), even with the representation of the transferred knowledge by simple neural networks, respectively.

**arXiv ID:** 2604.09049
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>AgenticAI-DialogGen: Topic-Guided Conversation Generation for Fine-Tuning and Evaluating Short- and Long-Term Memories of LLMs</strong> - Manoj Madushanka Perera, Adnan Mahmood, Kasun Eranda Wijethilake, Quan Z. Sheng - [[pdf]](https://arxiv.org/pdf/2604.12179)</summary>

**Abstract:** Recent advancements in Large Language Models (LLMs) have improved their ability to process extended conversational contexts, yet fine-tuning and evaluating short- and long-term memories remain difficult due to the absence of datasets that encode both short- and long-term conversational history. Existing conversational datasets lack memory grounding, overlook topic continuity, or rely on costly human annotation. To address these gaps, we introduce AgenticAI-DialogGen, a modular agent-based framework that generates persona-grounded and topic-guided conversations without human supervision. The framework uses LLM agents to extract knowledge graphs, identify topics, build speaker personas, and simulate topic-guided conversations from unstructured conversations. A QA module generates memory-grounded Question Answer (QA) pairs drawn from short- and long-term conversational histories. We also generated a new dataset entitled, TopicGuidedChat (TGC), where long-term memory is encoded as speaker-specific knowledge graphs and short-term memory as newly generated topic-guided conversations. Evaluations depict that AgenticAI-DialogGen yields higher conversational quality and LLMs fine-tuned on TGC dataset achieve improved performance on memory-grounded QA tasks.

**arXiv ID:** 2604.12179
</details>

<details>
<summary><strong>Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization</strong> - Yizhe Chi, Deyao Hong, Dapeng Jiang, Tianwei Luo, Kaisen Yang, Boshi Zhang, Zhe Cao, Xiaoyan Fan, Bingxiang He, Han Hao, Weiyang Jin, Dianqiao Lei, Qingle Liu, Houde Qian, Bowen Wang, Situ Wang, Youjie Zheng, Yifan Zhou, Calvin Xiao, Eren Cai, Qinhuai Na - [[pdf]](https://arxiv.org/pdf/2604.12290)</summary>

**Abstract:** Current LLM agent benchmarks, which predominantly focus on binary pass/fail tasks such as code generation or search-based question answering, often neglect the value of real-world engineering that is often captured through the iterative optimization of feasible designs. To this end, we introduce Frontier-Eng, a human-verified benchmark for generative optimization -- an iterative propose-execute-evaluate loop in which an agent generates candidate artifacts, receives executable verifier feedback, and revises them under a fixed interaction budget -- spanning $47$ tasks across five broad engineering categories. Unlike previous suites, Frontier-Eng tasks are grounded in industrial-grade simulators and verifiers that provide continuous reward signals and enforce hard feasibility constraints under constrained budgets. We evaluate eight frontier language models using representative search frameworks, finding that while Claude 4.6 Opus achieves the most robust performance, the benchmark remains challenging for all models. Our analysis suggests a dual power-law decay in improvement frequency ($\sim$ 1/iteration) and magnitude ($\sim$ 1/improvement count). We further show that although width improves parallelism and diversity, depth remains crucial for hard-won improvements under a fixed budget. Frontier-Eng establishes a new standard for assessing the capacity of AI agents to integrate domain knowledge with executable feedback to solve complex, open-ended engineering problems.

**arXiv ID:** 2604.12290
</details>

<details>
<summary><strong>Many-Tier Instruction Hierarchy in LLM Agents</strong> - Jingyu Zhang, Tianjian Li, William Jurayj, Hongyuan Zhan, Benjamin Van Durme, Daniel Khashabi - [[pdf]](https://arxiv.org/pdf/2604.09443)</summary>

**Abstract:** Large language model agents receive instructions from many sources-system messages, user prompts, tool outputs, other agents, and more-each carrying different levels of trust and authority. When these instructions conflict, agents must reliably follow the highest-privilege instruction to remain safe and effective. The dominant paradigm, instruction hierarchy (IH), assumes a fixed, small set of privilege levels (typically fewer than five) defined by rigid role labels (e.g., system > user). This is inadequate for real-world agentic settings, where conflicts can arise across far more sources and contexts. In this work, we propose Many-Tier Instruction Hierarchy (ManyIH), a paradigm for resolving instruction conflicts among instructions with arbitrarily many privilege levels. We introduce ManyIH-Bench, the first benchmark for ManyIH. ManyIH-Bench requires models to navigate up to 12 levels of conflicting instructions with varying privileges, comprising 853 agentic tasks (427 coding and 426 instruction-following). ManyIH-Bench composes constraints developed by LLMs and verified by humans to create realistic and difficult test cases spanning 46 real-world agents. Our experiments show that even the current frontier models perform poorly (~40% accuracy) when instruction conflict scales. This work underscores the urgent need for methods that explicitly target fine-grained, scalable instruction conflict resolution in agentic settings.

**arXiv ID:** 2604.09443
</details>

<details>
<summary><strong>CocoaBench: Evaluating Unified Digital Agents in the Wild</strong> - CocoaBench Team, Shibo Hao, Zhining Zhang, Zhiqi Liang, Tianyang Liu, Yuheng Zha, Qiyue Gao, Jixuan Chen, Zilong Wang, Zhoujun Cheng, Haoxiang Zhang, Junli Wang, Hexi Jin, Boyuan Zheng, Kun Zhou, Yu Wang, Feng Yao, Licheng Liu, Yijiang Li, Zhifei Li, Zhengtao Han, Pracha Promthaw, Tommaso Cerruti, Xiaohan Fu, Ziqiao Ma, Jingbo Shang, Lianhui Qin, Julian McAuley, Eric P. Xing, Zhengzhong Liu, Rupesh Kumar Srivastava, Zhiting Hu - [[pdf]](https://arxiv.org/pdf/2604.11201)</summary>

**Abstract:** LLM agents now perform strongly in software engineering, deep research, GUI automation, and various other applications, while recent agent scaffolds and models are increasingly integrating these capabilities into unified systems. Yet, most evaluations still test these capabilities in isolation, which leaves a gap for more diverse use cases that require agents to combine different capabilities. We introduce CocoaBench, a benchmark for unified digital agents built from human-designed, long-horizon tasks that require flexible composition of vision, search, and coding. Tasks are specified only by an instruction and an automatic evaluation function over the final output, enabling reliable and scalable evaluation across diverse agent infrastructures. We also present CocoaAgent, a lightweight shared scaffold for controlled comparison across model backbones. Experiments show that current agents remain far from reliable on CocoaBench, with the best evaluated system achieving only 45.1% success rate. Our analysis further points to substantial room for improvement in reasoning and planning, tool use and execution, and visual grounding.

**arXiv ID:** 2604.11201
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (11 papers)</h2></summary>

<details>
<summary><strong>CascadeDebate: Multi-Agent Deliberation for Cost-Aware LLM Cascades</strong> - Raeyoung Chang, Dongwook Kwon, Jisoo Lee, Nikhil Verma - [[pdf]](https://arxiv.org/pdf/2604.12262)</summary>

**Abstract:** Cascaded LLM systems coordinate models of varying sizes with human experts to balance accuracy, cost, and abstention under uncertainty. However, single-model tiers at each stage often struggle with ambiguous queries, triggering premature escalations to costlier models or experts due to under-confidence and inefficient compute scaling. CascadeDebate addresses this gap by inserting multi-agent deliberation directly at each tier's escalation boundary. Confidence-based routers activate lightweight agent ensembles only for uncertain cases, enabling consensus-driven resolution of ambiguities internally without invoking higher-cost upgrades. Our unified architecture alternates single-model inference with selective multi-agent deliberation across model scales, culminating in human experts as the final fallback. This design scales test-time compute dynamically according to query difficulty. Across five benchmarks spanning science, medicine, and general knowledge, CascadeDebate outperforms strong single-model cascades and standalone multi-agent systems by up to 26.75 percent. An online threshold optimizer proves essential, boosting accuracy by 20.98 to 52.33 percent relative improvement over fixed policies and enabling elastic adaptation to real-world distributions.

**arXiv ID:** 2604.12262
</details>

<details>
<summary><strong>Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Reasoning</strong> - Houxing Ren, Mingjie Zhan, Zimu Lu, Ke Wang, Yunqiao Yang, Haotian Hou, Hongsheng Li - [[pdf]](https://arxiv.org/pdf/2604.12282)</summary>

**Abstract:** Spreadsheets are central to real-world applications such as enterprise reporting, auditing, and scientific data management. Despite their ubiquity, existing large language model based approaches typically treat tables as plain text, overlooking critical layout cues and visual semantics. Moreover, real-world spreadsheets are often massive in scale, exceeding the input length that LLMs can efficiently process. To address these challenges, we propose SpreadsheetAgent, a two-stage multi-agent framework for spreadsheet understanding that adopts a step-by-step reading and reasoning paradigm. Instead of loading the entire spreadsheet at once, SpreadsheetAgent incrementally interprets localized regions through multiple modalities, including code execution results, images, and LaTeX tables. The method first constructs a structural sketch and row/column summaries, and then performs task-driven reasoning over this intermediate representation in the Solving Stage. To further enhance reliability, we design a verification module that validates extracted structures via targeted inspections, reducing error propagation and ensuring trustworthy inputs for downstream reasoning. Extensive experiments on two spreadsheet datasets demonstrate the effectiveness of our approach. With GPT-OSS-120B, SpreadsheetAgent achieves 38.16% on Spreadsheet Bench, outperforming the ChatGPT Agent baseline (35.27%) by 2.89 absolute points. These results highlight the potential of SpreadsheetAgent to advance robust and scalable spreadsheet understanding in real-world applications. Code is available at this https URL.

**arXiv ID:** 2604.12282
</details>

<details>
<summary><strong>EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution</strong> - Shiyu He, Minchi Kuang, Mengxian Wang, Bin Hu, Tingxiang Gu - [[pdf]](https://arxiv.org/pdf/2604.12776)</summary>

**Abstract:** Realizing endogenous narrative evolution in LLM-based multi-agent systems is hindered by the inherent stochasticity of generative emergence. In particular, long-horizon simulations suffer from social memory stacking, where conflicting relational states accumulate without resolution, and narrative-spatial dissonance, where spatial logic detaches from the evolving plot. To bridge this gap, we propose EvoSpark, a framework specifically designed to sustain logically coherent long-horizon narratives within Endogenous Interactive Agent Societies. To ensure consistency, the Stratified Narrative Memory employs a Role Socio-Evolutionary Base as living cognition, dynamically metabolizing experiences to resolve historical conflicts. Complementarily, Generative Mise-en-Scène mechanism enforces Role-Location-Plot alignment, synchronizing character presence with the narrative flow. Underpinning these is the Unified Narrative Operation Engine, which integrates an Emergent Character Grounding Protocol to transform stochastic sparking into persistent characters. This engine establishes a substrate that expands a minimal premise into an open-ended, evolving story world. Experiments demonstrate that EvoSpark significantly outperforms baselines across diverse paradigms, enabling the sustained generation of expressive and coherent narrative experiences.

**arXiv ID:** 2604.12776
</details>

<details>
<summary><strong>How memory can affect collective and cooperative behaviors in an LLM-Based Social Particle Swarm</strong> - Taisei Hishiki, Takaya Arita, Reiji Suzuki - [[pdf]](https://arxiv.org/pdf/2604.12250)</summary>

**Abstract:** This study examines how model-specific characteristics of Large Language Model (LLM) agents, including internal alignment, shape the effect of memory on their collective and cooperative dynamics in a multi-agent system. To this end, we extend the Social Particle Swarm (SPS) model, in which agents move in a two-dimensional space and play the Prisoner's Dilemma with neighboring agents, by replacing its rule-based agents with LLM agents endowed with Big Five personality scores and varying memory lengths. Using Gemini-2.0-Flash, we find that memory length is a critical parameter governing collective behavior: even a minimal memory drastically suppressed cooperation, transitioning the system from stable cooperative clusters through cyclical formation and collapse of clusters to a state of scattered defection as memory length increased. Big Five personality traits correlated with agent behaviors in partial agreement with findings from experiments with human participants, supporting the validity of the model. Comparative experiments using Gemma~3:4b revealed the opposite trend: longer memory promoted cooperation, accompanied by the formation of dense cooperative clusters. Sentiment analysis of agents' reasoning texts showed that Gemini interprets memory increasingly negatively as its length grows, while Gemma interprets it less negatively, and that this difference persists in the early phase of experiments before the macro-level dynamics converge. These results suggest that model-specific characteristics of LLMs, potentially including alignment, play a fundamental role in determining emergent social behavior in Generative Agent-Based Modeling, and provide a micro-level cognitive account of the contradictions found in prior work on memory and cooperation.

**arXiv ID:** 2604.12250
</details>

<details>
<summary><strong>Advancing Multi-Agent RAG Systems with Minimalist Reinforcement Learning</strong> - Yihong Wu, Liheng Ma, Muzhi Li, Jiaming Zhou, Lei Ding, Jianye Hao, Ho-fung Leung, Irwin King, Yingxue Zhang, Jian-Yun Nie - [[pdf]](https://arxiv.org/pdf/2505.17086)</summary>

**Abstract:** Large Language Models (LLMs) equipped with modern Retrieval-Augmented Generation (RAG) systems often employ multi-turn interaction pipelines to interface with search engines for complex reasoning tasks. However, such multi-turn interactions inevitably produce long intermediate contexts, as context length grows exponentially with exploration depth. This leads to a well-known limitation of LLMs: their difficulty in effectively leveraging information from long contexts. This problem is further amplified in RAG systems that depend on in-context learning, where few-shot demonstrations must also be included in the prompt, compounding the context-length bottleneck. To address these challenges, we propose Mujica-MyGo, a unified framework for efficient multi-turn reasoning in RAG. Inspired by the divide-and-conquer principle, we introduce Mujica (Multi-hop Joint Intelligence for Complex Question Answering), a multi-agent RAG workflow that decomposes multi-turn interactions into cooperative sub-interactions, thereby mitigating long-context issues. To eliminate the dependency on in-context learning, we further develop MyGO (Minimalist Policy Gradient Optimization), a lightweight and efficient reinforcement learning algorithm that enables effective post-training of LLMs within complex RAG pipelines. We provide theoretical guarantees for MyGO's convergence to the optimal policy. Empirical evaluations across diverse question-answering benchmarks, covering both text corpora and knowledge graphs, show that Mujica-MyGO achieves superior performance.

**arXiv ID:** 2505.17086
</details>

<details>
<summary><strong>Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention</strong> - Manh Nguyen, Anh Nguyen, Dung Nguyen, Svetha Venkatesh, Hung Le - [[pdf]](https://arxiv.org/pdf/2603.20640)</summary>

**Abstract:** Multi-Agent Debate has emerged as a promising framework for improving the reasoning quality of large language models through iterative inter-agent communication. However, broadcasting all agent messages at every round introduces noise and redundancy that can degrade debate quality and waste computational resources. Current approaches rely on uncertainty estimation to filter low-confidence responses before broadcasting, but this approach is unreliable due to miscalibrated confidence scores and sensitivity to threshold selection. To address this, we propose Diversity-Aware Retention (DAR), a lightweight debate framework that, at each debate round, selects the subset of agent responses that maximally disagree with each other and with the majority vote before broadcasting. Through an explicit index-based retention mechanism, DAR preserves the original messages without modification, ensuring that retained disagreements remain authentic. Experiments on diverse reasoning and question answering benchmarks demonstrate that our selective message propagation consistently improves debate performance, particularly as the number of agents scales, where noise accumulation is most severe. Our results highlight that what agents hear is as important as what agents say in multi-agent reasoning systems. Code is publicly available at this https URL.

**arXiv ID:** 2603.20640
</details>

<details>
<summary><strong>OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning</strong> - Pan Lu, Bowen Chen, Sheng Liu, Rahul Thapa, Joseph Boen, James Zou - [[pdf]](https://arxiv.org/pdf/2502.11271)</summary>

**Abstract:** Solving complex reasoning tasks may involve visual understanding, domain knowledge retrieval, numerical calculation, and multi-step reasoning. Existing methods augment large language models (LLMs) with external tools but are restricted to specialized domains, limited tool types, or require additional training data. In this paper, we introduce OctoTools, a training-free, user-friendly, and easily extensible multi-agent framework designed to tackle complex reasoning across diverse domains. OctoTools introduces standardized tool cards to encapsulate tool functionality, a planner for both high-level and low-level planning, and an executor to carry out tool usage. We validate OctoTools' generality across 16 diverse tasks (including MathVista, MMLU-Pro, MedQA, and GAIA-Text), achieving substantial average accuracy gains of 9.3% over GPT-4o. Furthermore, OctoTools also outperforms AutoGen, GPT-Functions, and LangChain by up to 10.6% when given the same set of tools. Through comprehensive analysi, ablations, and robustness tests with compact backbones and noisy tool environments, OctoTools demonstrates advantages in task planning, effective tool usage, and multi-step problem solving. Code, demos, and visualization are publicly available at this https URL.

**arXiv ID:** 2502.11271
</details>

<details>
<summary><strong>SEW: Self-Evolving Agentic Workflows for Automated Code Generation</strong> - Siwei Liu, Jinyuan Fang, Han Zhou, Yingxu Wang, Zaiqiao Meng - [[pdf]](https://arxiv.org/pdf/2505.18646)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated effectiveness in code generation tasks. To enable LLMs to address more complex coding challenges, existing research has focused on crafting multi-agent systems with agentic workflows, where complex coding tasks are decomposed into sub-tasks, assigned to specialized agents. Despite their effectiveness, current approaches heavily rely on hand-crafted agentic workflows, with both agent topologies and prompts manually designed, which limits their ability to automatically adapt to different types of coding problems. To address these limitations and enable automated workflow design, we propose \textbf{S}elf-\textbf{E}volving \textbf{W}orkflow (\textbf{SEW}), a novel self-evolving framework that automatically generates and optimises multi-agent workflows. Extensive experiments on three coding benchmark datasets, including the challenging LiveCodeBench, demonstrate that our SEW can automatically design agentic workflows and optimise them through self-evolution, bringing up to 12\% improvement on LiveCodeBench compared to using the backbone LLM only. Furthermore, by investigating different representation schemes of workflow, we provide insights into the optimal way to encode workflow information with text.

**arXiv ID:** 2505.18646
</details>

<details>
<summary><strong>When Reasoning Models Hurt Behavioral Simulation: A Solver-Sampler Mismatch in Multi-Agent LLM Negotiation</strong> - Sandro Andric - [[pdf]](https://arxiv.org/pdf/2604.11840)</summary>

**Abstract:** Large language models are increasingly used as agents in social, economic, and policy simulations. A common assumption is that stronger reasoning should improve simulation fidelity. We argue that this assumption can fail when the objective is not to solve a strategic problem, but to sample plausible boundedly rational behavior. In such settings, reasoning-enhanced models can become better solvers and worse simulators: they can over-optimize for strategically dominant actions, collapse compromise-oriented terminal behavior, and sometimes exhibit a diversity-without-fidelity pattern in which local variation survives without outcome-level fidelity. We study this solver-sampler mismatch in three multi-agent negotiation environments adapted from earlier simulation work: an ambiguous fragmented-authority trading-limits scenario, an ambiguous unified-opposition trading-limits scenario, and a new-domain grid-curtailment case in emergency electricity management. We compare three reflection conditions, no reflection, bounded reflection, and native reasoning, across two primary model families and then extend the same protocol to direct OpenAI runs with GPT-4.1 and GPT-5.2. Across all three experiments, bounded reflection produces substantially more diverse and compromise-oriented trajectories than either no reflection or native reasoning. In the direct OpenAI extension, GPT-5.2 native ends in authority decisions in 45 of 45 runs across the three experiments, while GPT-5.2 bounded recovers compromise outcomes in every environment. The contribution is not a claim that reasoning is generally harmful. It is a methodological warning: model capability and simulation fidelity are different objectives, and behavioral simulation should qualify models as samplers, not only as solvers.

**arXiv ID:** 2604.11840
</details>

<details>
<summary><strong>AutoSurrogate: An LLM-Driven Multi-Agent Framework for Autonomous Construction of Deep Learning Surrogate Models in Subsurface Flow</strong> - Jiale Liu, Nanzhe Wang - [[pdf]](https://arxiv.org/pdf/2604.11945)</summary>

**Abstract:** High-fidelity numerical simulation of subsurface flow is computationally intensive, especially for many-query tasks such as uncertainty quantification and data assimilation. Deep learning (DL) surrogates can significantly accelerate forward simulations, yet constructing them requires substantial machine learning (ML) expertise - from architecture design to hyperparameter tuning - that most domain scientists do not possess. Furthermore, the process is predominantly manual and relies heavily on heuristic choices. This expertise gap remains a key barrier to the broader adoption of DL surrogate techniques. For this reason, we present AutoSurrogate, a large-language-model-driven multi-agent framework that enables practitioners without ML expertise to build high-quality surrogates for subsurface flow problems through natural-language instructions. Given simulation data and optional preferences, four specialized agents collaboratively execute data profiling, architecture selection from a model zoo, Bayesian hyperparameter optimization, model training, and quality assessment against user-specified thresholds. The system also handles common failure modes autonomously, including restarting training with adjusted configurations when numerical instabilities occur and switching to alternative architectures when predictive accuracy falls short of targets. In our setting, a single natural-language sentence can be sufficient to produce a deployment-ready surrogate model, with minimum human intervention required at any intermediate stage. We demonstrate the utility of AutoSurrogate on a 3D geological carbon storage modeling task, mapping permeability fields to pressure and CO$_2$ saturation fields over 31 timesteps. Without any manual tuning, AutoSurrogate is able to outperform expert-designed baselines and domain-agnostic AutoML methods, demonstrating strong potential for practical deployment.

**arXiv ID:** 2604.11945
</details>

<details>
<summary><strong>Towards Autonomous Mechanistic Reasoning in Virtual Cells</strong> - Yunhui Jang, Lu Zhu, Jake Fawkes, Alisandra Kaye Denton, Dominique Beaini, Emmanuel Noutahi - [[pdf]](https://arxiv.org/pdf/2604.11661)</summary>

**Abstract:** Large language models (LLMs) have recently gained significant attention as a promising approach to accelerate scientific discovery. However, their application in open-ended scientific domains such as biology remains limited, primarily due to the lack of factually grounded and actionable explanations. To address this, we introduce a structured explanation formalism for virtual cells that represents biological reasoning as mechanistic action graphs, enabling systematic verification and falsification. Building upon this, we propose VCR-Agent, a multi-agent framework that integrates biologically grounded knowledge retrieval with a verifier-based filtering approach to generate and validate mechanistic reasoning autonomously. Using this framework, we release VC-TRACES dataset, which consists of verified mechanistic explanations derived from the Tahoe-100M atlas. Empirically, we demonstrate that training with these explanations improves factual precision and provides a more effective supervision signal for downstream gene expression prediction. These results underscore the importance of reliable mechanistic reasoning for virtual cells, achieved through the synergy of multi-agent and rigorous verification.

**arXiv ID:** 2604.11661
</details>

</details>

<details open>
<summary><h2>Other Agent Research (2 papers)</h2></summary>

<details>
<summary><strong>Thermodynamic Liquid Manifold Networks: Physics-Bounded Deep Learning for Solar Forecasting in Autonomous Off-Grid Microgrids</strong> - Mohammed Ezzaldin Babiker Abdullah - [[pdf]](https://arxiv.org/pdf/2604.11909)</summary>

**Abstract:** The stable operation of autonomous off-grid photovoltaic systems requires solar forecasting algorithms that respect atmospheric thermodynamics. Contemporary deep learning models consistently exhibit critical anomalies, primarily severe temporal phase lags during cloud transients and physically impossible nocturnal power generation. To resolve this divergence between data-driven modeling and deterministic celestial mechanics, this research introduces the Thermodynamic Liquid Manifold Network. The methodology projects 22 meteorological and geometric variables into a Koopman-linearized Riemannian manifold to systematically map complex climatic dynamics. The architecture integrates a Spectral Calibration unit and a multiplicative Thermodynamic Alpha-Gate. This system synthesizes real-time atmospheric opacity with theoretical clear-sky boundary models, structurally enforcing strict celestial geometry compliance. This completely neutralizes phantom nocturnal generation while maintaining zero-lag synchronization during rapid weather shifts. Validated against a rigorous five-year testing horizon in a severe semi-arid climate, the framework achieves an RMSE of 18.31 Wh/m2 and a Pearson correlation of 0.988. The model strictly maintains a zero-magnitude nocturnal error across all 1826 testing days and exhibits a sub-30-minute phase response during high-frequency optical transients. Comprising exactly 63,458 trainable parameters, this ultra-lightweight design establishes a robust, thermodynamically consistent standard for edge-deployable microgrid controllers.

**arXiv ID:** 2604.11909
</details>

<details>
<summary><strong>Identity as Attractor: Geometric Evidence for Persistent Agent Architecture in LLM Activation Space</strong> - Vladimir Vasilenko - [[pdf]](https://arxiv.org/pdf/2604.12016)</summary>

**Abstract:** Large language models map semantically related prompts to similar internal representations -- a phenomenon interpretable as attractor-like dynamics. We ask whether the identity document of a persistent cognitive agent (its cognitive_core) exhibits analogous attractor-like behavior. We present a controlled experiment on Llama 3.1 8B Instruct, comparing hidden states of an original cognitive_core (Condition A), seven paraphrases (Condition B), and seven structurally matched controls (Condition C). Mean-pooled states at layers 8, 16, and 24 show that paraphrases converge to a tighter cluster than controls (Cohen's d > 1.88, p < 10^{-27}, Bonferroni-corrected). Replication on Gemma 2 9B confirms cross-architecture generalizability. Ablations suggest the effect is primarily semantic rather than structural, and that structural completeness appears necessary to reach the attractor region. An exploratory experiment shows that reading a scientific description of the agent shifts internal state toward the attractor -- closer than a sham preprint -- distinguishing knowing about an identity from operating as that identity. These results provide representational evidence that agent identity documents induce attractor-like geometry in LLM activation space.

**arXiv ID:** 2604.12016
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (19 papers)</h2></summary>

<details>
<summary><strong>Coding-Free and Privacy-Preserving MCP Framework for Clinical Agentic Research Intelligence System</strong> - Taehun Kim, Hyeryun Park, Hyeonhoon Lee, Yushin Lee, Kyungsang Kim, Hyung-Chul Lee - [[pdf]](https://arxiv.org/pdf/2604.12258)</summary>

**Abstract:** Clinical research involves labor-intensive processes such as study design, cohort construction, model development, and documentation, requiring domain expertise, programming skills, and access to sensitive patient data. These demands create barriers for clinicians and external researchers conducting data-driven studies. To overcome these limitations, we developed a Clinical Agentic Research Intelligence System (CARIS) that automates the clinical research workflow while preserving data privacy, enabling comprehensive studies without direct access to raw data. CARIS integrates Large Language Models (LLMs) with modular tools via the Model Context Protocol (MCP), enabling natural language-driven orchestration of appropriate tools. Databases remain securely within the MCP server, and users access only the outputs and final research reports. Based on user intent, CARIS automatically executes the full pipeline: research planning, literature search, cohort construction, Institutional Review Board (IRB) documentation, Vibe Machine Learning (ML), and report generation, with iterative human-in-the-loop refinement. We evaluated CARIS on three heterogeneous datasets with distinct clinical tasks. Research plans and IRB documents were finalized within three to four iterations, using evidence from literature and data. The system supported Vibe ML by exploring feature-model combinations, ranking the top ten models, and generating performance visualizations. Final reports showed high completeness based on a checklist derived from the TRIPOD+AI framework, achieving 96% coverage in LLM evaluation and 82% in human evaluation. CARIS demonstrates that agentic AI can transform clinical hypotheses into executable research workflows across heterogeneous datasets. By eliminating the need for coding and direct data access, the system lowers barriers and bridges public and private clinical data environments.

**arXiv ID:** 2604.12258
</details>

<details>
<summary><strong>Agentic Insight Generation in VSM Simulations</strong> - Micha Selak, Dirk Krechel, Adrian Ulges, Sven Spieckermann, Niklas Stoehr, Andreas Loehr - [[pdf]](https://arxiv.org/pdf/2604.12421)</summary>

**Abstract:** Extracting actionable insights from complex value stream map simulations can be challenging, time-consuming, and error-prone. Recent advances in large language models offer new avenues to support users with this task. While existing approaches excel at processing raw data to gain information, they are structurally unfit to pick up on subtle situational differences needed to distinguish similar data sources in this domain. To address this issue, we propose a decoupled, two-step agentic architecture. By separating orchestration from data analysis, the system leverages progressive data discovery infused with domain expert knowledge. This architecture allows the orchestration to intelligently select data sources and perform multi-hop reasoning across data structures while maintaining a slim internal context. Results from multiple state-of-the-art large language models demonstrate the framework's viability: with top-tier models achieving accuracies of up to 86% and demonstrating high robustness across evaluation runs.

**arXiv ID:** 2604.12421
</details>

<details>
<summary><strong>Teaching LLMs Human-Like Editing of Inappropriate Argumentation via Reinforcement Learning</strong> - Timon Ziegenbein, Maja Stahl, Henning Wachsmuth - [[pdf]](https://arxiv.org/pdf/2604.12770)</summary>

**Abstract:** Editing human-written text has become a standard use case of large language models (LLMs), for example, to make one's arguments more appropriate for a discussion. Comparing human to LLM-generated edits, however, we observe a mismatch in editing strategies: While LLMs often perform multiple scattered edits and tend to change meaning notably, humans rather encapsulate dependent changes in self-contained, meaning-preserving edits. In this paper, we present a reinforcement learning approach that teaches LLMs human-like editing to improve the appropriateness of arguments. Our approach produces self-contained sentence-level edit suggestions that can be accepted or rejected independently. We train the approach using group relative policy optimization with a multi-component reward function that jointly optimizes edit-level semantic similarity, fluency, and pattern conformity as well as argument-level appropriateness. In automatic and human evaluation, it outperforms competitive baselines and the state of the art in human-like editing, with multi-round editing achieving appropriateness close to full rewriting.

**arXiv ID:** 2604.12770
</details>

<details>
<summary><strong>MolMem: Memory-Augmented Agentic Reinforcement Learning for Sample-Efficient Molecular Optimization</strong> - Ziqing Wang, Yibo Wen, Abhishek Pandy, Han Liu, Kaize Ding - [[pdf]](https://arxiv.org/pdf/2604.12237)</summary>

**Abstract:** In drug discovery, molecular optimization aims to iteratively refine a lead compound to improve molecular properties while preserving structural similarity to the original molecule. However, each oracle evaluation is expensive, making sample efficiency a key challenge for existing methods under a limited oracle budget. Trial-and-error approaches require many oracle calls, while methods that leverage external knowledge tend to reuse familiar templates and struggle on challenging objectives. A key missing piece is long-term memory that can ground decisions and provide reusable insights for future optimizations. To address this, we present MolMem (\textbf{Mol}ecular optimization with \textbf{Mem}ory), a multi-turn agentic reinforcement learning (RL) framework with a dual-memory system. Specifically, MolMem uses Static Exemplar Memory to retrieve relevant exemplars for cold-start grounding, and Evolving Skill Memory to distill successful trajectories into reusable strategies. Built on this memory-augmented formulation, we train the policy with dense step-wise rewards, turning costly rollouts into long-term knowledge that improves future optimization. Extensive experiments show that MolMem achieves 90\% success on single-property tasks (1.5$\times$ over the best baseline) and 52\% on multi-property tasks using only 500 oracle calls. Our code is available at this https URL.

**arXiv ID:** 2604.12237
</details>

<details>
<summary><strong>Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning</strong> - NVIDIA, Aakshita Chandiramani, Aaron Blakeman, Abdullahi Olaoye, Abhibha Gupta, Abhilash Somasamudramath, Abhinav Khattar, Adeola Adesoba, Adi Renduchintala, Adil Asif, Aditya Agrawal, Aditya Vavre, Ahmad Kiswani, Aishwarya Padmakumar, Ajay Hotchandani, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, Aleksandr Shaposhnikov, Alex Gronskiy, Alex Kondratenko, Alex Neefus, Alex Steiner, Alex Yang, Alexander Bukharin, Alexander Young, Ali Hatamizadeh, Ali Taghibakhshi, Alina Galiautdinova, Alisa Liu, Alok Kumar, Ameya Sunil Mahabaleshwarkar, Amir Klein, Amit Zuker, Amnon Geifman, Anahita Bhiwandiwalla, Ananth Subramaniam, Andrew Tao, Anjaney Shrivastava, Anjulie Agrusa, Ankur Srivastava, Ankur Verma, Ann Guan, Anna Shors, Annamalai Chockalingam, Anubhav Mandarwal, Aparnaa Ramani, Arham Mehta, Arti Jain, Arun Venkatesan, Asha Anoosheh, Ashwath Aithal, Ashwin Poojary, Asif Ahamed, Asit Mishra, Asli Sabanci Demiroz, Asma Kuriparambil Thekkumpate, Atefeh Sohrabizadeh, Avinash Kaur, Ayush Dattagupta, Barath Subramaniam Anandan, Bardiya Sadeghi, Barnaby Simkin, Ben Lanir, Benedikt Schifferer, Benjamin Chislett, Besmira Nushi, Bilal Kartal, Bill Thiede, Bita Darvish Rouhani, Bobby Chen, Boris Ginsburg, Brandon Norick, Branislav Kisacanin, Brian Yu, Bryan Catanzaro, Buvaneswari Mani, Carlo del Mundo, Chankyu Lee, Chanran Kim, Chantal Hwang, Chao Ni, Charles Wang, Charlie Truong, Cheng-Ping Hsieh, Chenhan Yu, Chenjie Luo, Cherie Wang, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Chris Holguin, Chris Wing, Christian Munley, Christopher Parisien, Chuck Desai, Chunyang Sheng, Collin Neale, Cyril Meurillon, Dakshi Kumar - [[pdf]](https://arxiv.org/pdf/2604.12374)</summary>

**Abstract:** We describe the pre-training, post-training, and quantization of Nemotron 3 Super, a 120 billion (active 12 billion) parameter hybrid Mamba-Attention Mixture-of-Experts model. Nemotron 3 Super is the first model in the Nemotron 3 family to 1) be pre-trained in NVFP4, 2) leverage LatentMoE, a new Mixture-of-Experts architecture that optimizes for both accuracy per FLOP and accuracy per parameter, and 3) include MTP layers for inference acceleration through native speculative decoding. We pre-trained Nemotron 3 Super on 25 trillion tokens followed by post-training using supervised fine tuning (SFT) and reinforcement learning (RL). The final model supports up to 1M context length and achieves comparable accuracy on common benchmarks, while also achieving up to 2.2x and 7.5x higher inference throughput compared to GPT-OSS-120B and Qwen3.5-122B, respectively. Nemotron 3 Super datasets, along with the base, post-trained, and quantized checkpoints, are open-sourced on HuggingFace.

**arXiv ID:** 2604.12374
</details>

<details>
<summary><strong>Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following</strong> - Qingyu Ren, Qianyu He, Powei Chang, Jie Zeng, Zeye Sun, Fei Yu, Jiaqing Liang, Yanghua Xiao - [[pdf]](https://arxiv.org/pdf/2510.14420)</summary>

**Abstract:** Language models often struggle to follow multi-constraint instructions that are crucial for real-world applications. Existing reinforcement learning (RL) approaches suffer from dependency on external supervision and sparse reward signals from multi-constraint tasks. We propose a label-free self-supervised RL framework that eliminates dependency on external supervision by deriving reward signals directly from instructions and generating pseudo-labels for reward model training. Our approach introduces constraint decomposition strategies and efficient constraint-wise binary classification to address sparse reward challenges while maintaining computational efficiency. Experiments show that our approach generalizes well, achieving strong improvements across 3 in-domain and 5 out-of-domain datasets, including challenging agentic and multi-turn instruction following. The data and code are publicly available at this https URL

**arXiv ID:** 2510.14420
</details>

<details>
<summary><strong>KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning</strong> - Shuai Wang, Yinan Yu - [[pdf]](https://arxiv.org/pdf/2603.21440)</summary>

**Abstract:** Large Language Models (LLMs) demonstrate impressive natural language capabilities but often struggle with knowledge-intensive reasoning tasks. Knowledge Base Question Answering (KBQA), which leverages structured Knowledge Graphs (KGs) exemplifies this challenge due to the need for accurate multi-hop reasoning. Existing approaches typically perform sequential reasoning steps guided by predefined pipelines, restricting flexibility and causing error cascades due to isolated reasoning at each step. To address these limitations, we propose KG-Hopper, a novel Reinforcement Learning (RL) framework that empowers compact open LLMs with the ability to perform integrated multi-hop KG reasoning within a single inference round. Rather than reasoning step-by-step, we train a Reasoning LLM that embeds the entire KG traversal and decision process into a unified ``thinking'' stage, enabling global reasoning over cross-step dependencies and dynamic path exploration with backtracking. Experimental results on eight KG reasoning benchmarks show that KG-Hopper, based on a 7B-parameter LLM, consistently outperforms larger multi-step systems (up to 70B) and achieves competitive performance with proprietary models such as GPT-3.5-Turbo and GPT-4o-mini, while remaining compact, open, and data-efficient. The code is publicly available at: this https URL.

**arXiv ID:** 2603.21440
</details>

<details>
<summary><strong>Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale</strong> - Liujie Zhang, Benzhe Ning, Rui Yang, Xiaoyan Yu, Jiaxing Li, Lumeng Wu, Jia Liu, Minghao Li, Weihang Chen, Weiqi Hu, Lei Zhang - [[pdf]](https://arxiv.org/pdf/2604.11554)</summary>

**Abstract:** Reinforcement learning (RL) post-training has proven effective at unlocking reasoning, self-reflection, and tool-use capabilities in large language models. As models extend to omni-modal inputs and agentic multi-turn workflows, RL training systems face three interdependent challenges: heterogeneous data flows, operational robustness at scale, and the staleness -- throughput tradeoff. We present \textbf{Relax} (Reinforcement Engine Leveraging Agentic X-modality), an open-source RL training engine that addresses these challenges through three co-designed architectural layers. First, an \emph{omni-native architecture} builds multimodal support into the full stack -- from data preprocessing and modality-aware parallelism to inference generation -- rather than retrofitting it onto a text-centric pipeline. Second, each RL role runs as an independent, fault-isolated service that can be scaled, recovered, and upgraded without global coordination. Third, service-level decoupling enables asynchronous training via the TransferQueue data bus, where a single staleness parameter smoothly interpolates among on-policy, near-on-policy, and fully asynchronous execution. Relax achieves a 1.20$\times$ end-to-end speedup over veRL on Qwen3-4B on-policy training. Its fully async mode delivers a 1.76$\times$ speedup over colocate on Qwen3-4B and a 2.00$\times$ speedup on Qwen3-Omni-30B, while all modes converge to the same reward level. Relax supports R3 (Rollout Routing Replay)~\cite{ma2025r3} for MoE models with only 1.9\% overhead, compared to 32\% degradation in veRL under the same configuration. It further demonstrates stable omni-modal RL convergence on Qwen3-Omni across image, text, and audio, sustaining over 2{,}000 steps on video without degradation. Relax is available at this https URL.

**arXiv ID:** 2604.11554
</details>

<details>
<summary><strong>Offline-Online Reinforcement Learning for Linear Mixture MDPs</strong> - Zhongjun Zhang, Sean R. Sinclair - [[pdf]](https://arxiv.org/pdf/2604.11994)</summary>

**Abstract:** We study offline-online reinforcement learning in linear mixture Markov decision processes (MDPs) under environment shift. In the offline phase, data are collected by an unknown behavior policy and may come from a mismatched environment, while in the online phase the learner interacts with the target environment. We propose an algorithm that adaptively leverages offline data. When the offline data are informative, either due to sufficient coverage or small environment shift, the algorithm provably improves over purely online learning. When the offline data are uninformative, it safely ignores them and matches the online-only performance. We establish regret upper bounds that explicitly characterize when offline data are beneficial, together with nearly matching lower bounds. Numerical experiments further corroborate our theoretical findings.

**arXiv ID:** 2604.11994
</details>

<details>
<summary><strong>Labeled TrustSet Guided: Batch Active Learning with Reinforcement Learning</strong> - Guofeng Cui, Yang Liu, Pichao Wang, Hankai Hsu, Xiaohang Sun, Xiang Hao, Zhu Liu - [[pdf]](https://arxiv.org/pdf/2604.12303)</summary>

**Abstract:** Batch active learning (BAL) is a crucial technique for reducing labeling costs and improving data efficiency in training large-scale deep learning models. Traditional BAL methods often rely on metrics like Mahalanobis Distance to balance uncertainty and diversity when selecting data for annotation. However, these methods predominantly focus on the distribution of unlabeled data and fail to leverage feedback from labeled data or the model's performance. To address these limitations, we introduce TrustSet, a novel approach that selects the most informative data from the labeled dataset, ensuring a balanced class distribution to mitigate the long-tail problem. Unlike CoreSet, which focuses on maintaining the overall data distribution, TrustSet optimizes the model's performance by pruning redundant data and using label information to refine the selection process. To extend the benefits of TrustSet to the unlabeled pool, we propose a reinforcement learning (RL)-based sampling policy that approximates the selection of high-quality TrustSet candidates from the unlabeled data. Combining TrustSet and RL, we introduce the Batch Reinforcement Active Learning with TrustSet (BRAL-T) framework. BRAL-T achieves state-of-the-art results across 10 image classification benchmarks and 2 active fine-tuning tasks, demonstrating its effectiveness and efficiency in various domains.

**arXiv ID:** 2604.12303
</details>

<details>
<summary><strong>GCA Framework: A Gulf-Grounded Dataset and Agentic Pipeline for Climate Decision Support</strong> - Muhammad Umer Sheikh, Khawar Shehzad, Salman Khan, Fahad Shahbaz Khan, Muhammad Haris Khan - [[pdf]](https://arxiv.org/pdf/2604.12306)</summary>

**Abstract:** Climate decision-making in the Gulf increasingly demands systems that can translate heterogeneous scientific and policy evidence into actionable guidance, yet general-purpose large language models (LLMs) remain weak both in region-specific climate knowledge and grounded interaction with geospatial and forecasting tools. We present the GCA framework, which unifies (i) GCA-DS, a curated Gulf-focused multimodal dataset, and (ii) Gulf Climate Agent (GCA), a tool-augmented agent for climate analysis. GCA-DS comprises ~200k question-answer pairs spanning governmental policies and adaptation plans, NGO and international frameworks, academic literature, and event-driven reporting on heatwaves, dust storms, and floods, complemented with remote-sensing inputs that couple imagery with textual evidence. Building on this foundation, the GCA agent orchestrates a modular tool pipeline grounded in real-time and historical signals and geospatial processing that produces derived indices and interpretable visualizations. Finally, we benchmark open and proprietary LLMs on Gulf climate tasks and show that domain fine-tuning and tool integration substantially improve reliability over general-purpose baselines.

**arXiv ID:** 2604.12306
</details>

<details>
<summary><strong>Agentic Control in Variational Language Models</strong> - Yves Ruffenach - [[pdf]](https://arxiv.org/pdf/2604.12513)</summary>

**Abstract:** We study whether a variational language model can support a minimal and measurable form of agentic control grounded in its own internal evidence. Our model combines local variational hidden computation (EVE), a homeostatic latent regulator, structurally aware checkpoint retention and a calibrated uncertainty-aware controller operating on top of the retained model. Rather than treating uncertainty as a passive diagnostic measured after prediction, we treat it as an operational signal that can regulate training, support checkpoint retention and guide inference-time intervention. The resulting framework is deliberately focused. It studies a closed-loop form of internal control in which structural and predictive signals become actionable. Empirically, the variational backbone improves over a matched deterministic reference on the language-modeling task while also exhibiting a richer and more usable uncertainty profile. On top of this backbone, the calibrated controller remains active, uses multiple actions under a full agentic evaluation and yields a positive quality-cost trade-off. These results support a precise claim: internal uncertainty can serve not only as a descriptive property of a variational language model, but also as a practical control interface for regulation, checkpoint retention and minimal agentic routing.

**arXiv ID:** 2604.12513
</details>

<details>
<summary><strong>Free Random Projection for In-Context Reinforcement Learning</strong> - Tomohiro Hayase, Benoît Collins, Nakamasa Inoue - [[pdf]](https://arxiv.org/pdf/2504.06983)</summary>

**Abstract:** Hierarchical inductive biases are hypothesized to promote generalizable policies in reinforcement learning, as demonstrated by explicit hyperbolic latent representations and architectures. Therefore, a more flexible approach is to have these biases emerge naturally from the algorithm. We introduce Free Random Projection, an input mapping grounded in free probability theory that constructs random orthogonal matrices where hierarchical structure arises inherently. The free random projection integrates seamlessly into existing in-context reinforcement learning frameworks by encoding hierarchical organization within the input space without requiring explicit architectural modifications. Empirical results on multi-environment benchmarks show that free random projection consistently outperforms the standard random projection, leading to improvements in generalization. Furthermore, analyses within linearly solvable Markov decision processes and investigations of the spectrum of kernel random matrices reveal the theoretical underpinnings of free random projection's enhanced performance, highlighting its capacity for effective adaptation in hierarchically structured state spaces.

**arXiv ID:** 2504.06983
</details>

<details>
<summary><strong>Replicable Reinforcement Learning with Linear Function Approximation</strong> - Eric Eaton, Marcel Hussing, Michael Kearns, Aaron Roth, Sikata Bela Sengupta, Jessica Sorrell - [[pdf]](https://arxiv.org/pdf/2509.08660)</summary>

**Abstract:** Replication of experimental results has been a challenge faced by many scientific disciplines, including the field of machine learning. Recent work on the theory of machine learning has formalized replicability as the demand that an algorithm produce identical outcomes when executed twice on different samples from the same distribution. Provably replicable algorithms are especially interesting for reinforcement learning (RL), where algorithms are known to be unstable in practice. While replicable algorithms exist for tabular RL settings, extending these guarantees to more practical function approximation settings has remained an open problem. In this work, we make progress by developing replicable methods for linear function approximation in RL. We first introduce two efficient algorithms for replicable random design regression and uncentered covariance estimation, each of independent interest. We then leverage these tools to provide the first provably efficient replicable RL algorithms for linear Markov decision processes in both the generative model and episodic settings. Finally, we evaluate our algorithms experimentally and show how they can inspire more consistent neural policies.

**arXiv ID:** 2509.08660
</details>

<details>
<summary><strong>Hybrid-AIRL: Enhancing Inverse Reinforcement Learning with Supervised Expert Guidance</strong> - Bram Silue, Santiago Amaya-Corredor, Patrick Mannion, Lander Willem, Pieter Libin - [[pdf]](https://arxiv.org/pdf/2511.21356)</summary>

**Abstract:** Adversarial Inverse Reinforcement Learning (AIRL) has shown promise in addressing the sparse reward problem in reinforcement learning (RL) by inferring dense reward functions from expert demonstrations. However, its performance in highly complex, imperfect-information settings remains largely unexplored. To explore this gap, we evaluate AIRL in the context of Heads-Up Limit Hold'em (HULHE) poker, a domain characterized by sparse, delayed rewards and significant uncertainty. In this setting, we find that AIRL struggles to infer a sufficiently informative reward function. To overcome this limitation, we contribute Hybrid-AIRL (H-AIRL), an extension that enhances reward inference and policy learning by incorporating a supervised loss derived from expert data and a stochastic regularization mechanism. We evaluate H-AIRL on a carefully selected set of Gymnasium benchmarks and the HULHE poker setting. Additionally, we analyze the learned reward function through visualization to gain deeper insights into the learning process. Our experimental results show that H-AIRL achieves higher sample efficiency and more stable learning compared to AIRL. This highlights the benefits of incorporating supervised signals into inverse RL and establishes H-AIRL as a promising framework for tackling challenging, real-world settings.

**arXiv ID:** 2511.21356
</details>

<details>
<summary><strong>DRPG (Decompose, Retrieve, Plan, Generate): An Agentic Framework for Academic Rebuttal</strong> - Peixuan Han, Yingjie Yu, Jingjun Xu, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2601.18081)</summary>

**Abstract:** Despite the growing adoption of large language models (LLMs) in scientific research workflows, automated support for academic rebuttal, a crucial step in academic communication and peer review, remains largely underexplored. Existing approaches typically rely on off-the-shelf LLMs or simple pipelines, which struggle with long-context understanding and often fail to produce targeted and persuasive responses. In this paper, we propose DRPG, an agentic framework for automatic academic rebuttal generation that operates through four steps: Decompose reviews into atomic concerns, Retrieve relevant evidence from the paper, Plan rebuttal strategies, and Generate responses accordingly. Notably, the Planner in DRPG reaches over 98% accuracy in identifying the most feasible rebuttal direction. Experiments on data from top-tier conferences demonstrate that DRPG significantly outperforms existing rebuttal pipelines and achieves performance beyond the average human level using only an 8B model. Our analysis further demonstrates the effectiveness of the planner design and its value in providing multi-perspective and explainable suggestions. We also showed that DRPG works well in a more complex multi-round setting. These results highlight the effectiveness of DRPG and its potential to provide high-quality rebuttal content and support the scaling of academic discussions. Codes for this work are available at this https URL.

**arXiv ID:** 2601.18081
</details>

<details>
<summary><strong>RLGT: A reinforcement learning framework for extremal graph theory</strong> - Ivan Damnjanović, Uroš Milivojević, Irena Đorđević, Dragan Stevanović - [[pdf]](https://arxiv.org/pdf/2602.17276)</summary>

**Abstract:** Reinforcement learning (RL) is a subfield of machine learning that focuses on developing models that can autonomously learn optimal decision-making strategies over time. In a recent pioneering paper, Wagner demonstrated how the Deep Cross-Entropy RL method can be applied to tackle various problems from extremal graph theory by reformulating them as combinatorial optimization problems. Subsequently, many researchers became interested in refining and extending the framework introduced by Wagner, thereby creating various RL environments specialized for graph theory. Moreover, a number of problems from extremal graph theory were solved through the use of RL. In particular, several inequalities concerning the Laplacian spectral radius of graphs were refuted, new lower bounds were obtained for certain Ramsey numbers, and contributions were made to the Turán-type extremal problem in which the forbidden structures are cycles of length three and four. Here, we present Reinforcement Learning for Graph Theory (RLGT), a novel RL framework that systematizes the previous work and provides support for both undirected and directed graphs, with or without loops, and with an arbitrary number of edge colors. The framework efficiently represents graphs and aims to facilitate future RL-based research in extremal graph theory through optimized computational performance and a clean and modular design.

**arXiv ID:** 2602.17276
</details>

<details>
<summary><strong>A Two-Timescale Primal-Dual Framework for Reinforcement Learning via Online Dual Variable Guidance</strong> - Axel Friedrich Wolter, Tobias Sutter - [[pdf]](https://arxiv.org/pdf/2505.04494)</summary>

**Abstract:** We study reinforcement learning by combining recent advances in regularized linear programming formulations with the classical theory of stochastic approximation. Motivated by the challenge of designing algorithms that leverage off-policy data while maintaining on-policy exploration, we propose PGDA-RL, a novel primal-dual Projected Gradient Descent-Ascent algorithm for solving regularized Markov Decision Processes (MDPs). PGDA-RL integrates experience replay-based gradient estimation with a two-timescale decomposition of the underlying nested optimization problem. The algorithm operates asynchronously, interacts with the environment through a single trajectory of correlated data, and updates its policy online in response to the dual variable associated with the occupancy measure of the underlying MDP. We prove that PGDA-RL converges almost surely to the optimal value function and policy of the regularized MDP. Our convergence analysis relies on tools from stochastic approximation theory and holds under weaker assumptions than those required by existing primal-dual RL approaches, notably removing the need for a simulator or a fixed behavioral policy. Under a strengthened ergodicity assumption on the underlying Markov chain, we establish a last-iterate finite-time guarantee with $\tilde{O} (k^{-2/3})$ mean-square convergence, aligning with the best-known rates for two-timescale stochastic approximation methods under Markovian sampling and biased gradient estimates.

**arXiv ID:** 2505.04494
</details>

<details>
<summary><strong>A longitudinal health agent framework</strong> - Georgianna, Rencong Jiang, Noémie Elhadad, Xuhai "Orson" Xu - [[pdf]](https://arxiv.org/pdf/2604.12019)</summary>

**Abstract:** Although artificial intelligence (AI) agents are increasingly proposed to support potentially longitudinal health tasks, such as symptom management, behavior change, and patient support, most current implementations fall short of facilitating user intent and fostering accountability. This contrasts with prior work on supporting longitudinal needs, where follow-up, coherent reasoning, and sustained alignment with individuals' goals are critical for both effectiveness and safety. In this paper, we draw on established clinical and personal health informatics frameworks to define what it would mean to orchestrate longitudinal health interactions with AI agents. We propose a multi-layer framework and corresponding agent architecture that operationalizes adaptation, coherence, continuity, and agency across repeated interactions. Through representative use cases, we demonstrate how longitudinal agents can maintain meaningful engagement, adapt to evolving goals, and support safe, personalized decision-making over time. Our findings underscore both the promise and the complexity of designing systems capable of supporting health trajectories beyond isolated interactions, and we offer guidance for future research and development in multi-session, user-centered health AI.

**arXiv ID:** 2604.12019
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
