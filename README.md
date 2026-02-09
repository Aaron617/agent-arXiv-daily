# Agent arXiv Daily

**Last Updated:** 2026-02-09 03:19:22

**Total Papers:** 85

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Planning and Reasoning](#planning-and-reasoning)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (3 papers)</h2></summary>

<details>
<summary><strong>Hear You in Silence: Designing for Active Listening in Human Interaction with Conversational Agents Using Context-Aware Pacing</strong> - Zhihan Jiang, Qianhui Chen, Chu Zhang, Yanheng Li, Ray LC - [[pdf]](https://arxiv.org/pdf/2602.06134)</summary>

**Abstract:** In human conversation, empathic dialogue requires nuanced temporal cues indicating whether the conversational partner is paying attention. This type of "active listening" is overlooked in the design of Conversational Agents (CAs), which use the same pacing for one conversation. To model the temporal cues in human conversation, we need CAs that dynamically adjust response pacing according to user input. We qualitatively analyzed ten cases of active listening to distill five context-aware pacing strategies: Reflective Silence, Facilitative Silence, Empathic Silence, Holding Space, and Immediate Response. In a between-subjects study (N=50) with two conversational scenarios (relationship and career-support), the context-aware agent scored higher than static-pacing control on perceived human-likeness, smoothness, and interactivity, supporting deeper self-disclosure and higher engagement. In the career support scenario, the CA yielded higher perceived listening quality and affective trust. This work shows how insights from human conversation like context-aware pacing can empower the design of more empathic human-AI communication.

**arXiv ID:** 2602.06134
</details>

<details>
<summary><strong>FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory</strong> - Lei Wei, Xiao Peng, Xu Dong, Niantao Xie, Bin Wang - [[pdf]](https://arxiv.org/pdf/2601.18642)</summary>

**Abstract:** Large language models deployed as autonomous agents face critical memory limitations, lacking selective forgetting mechanisms that lead to either catastrophic forgetting at context boundaries or information overload within them. While human memory naturally balances retention and forgetting through adaptive decay processes, current AI systems employ binary retention strategies that preserve everything or lose it entirely. We propose FadeMem, a biologically-inspired agent memory architecture that incorporates active forgetting mechanisms mirroring human cognitive efficiency. FadeMem implements differential decay rates across a dual-layer memory hierarchy, where retention is governed by adaptive exponential decay functions modulated by semantic relevance, access frequency, and temporal patterns. Through LLM-guided conflict resolution and intelligent memory fusion, our system consolidates related information while allowing irrelevant details to fade. Experiments on Multi-Session Chat, LoCoMo, and LTI-Bench demonstrate superior multi-hop reasoning and retrieval with 45\% storage reduction, validating the effectiveness of biologically-inspired forgetting in agent memory systems.

**arXiv ID:** 2601.18642
</details>

<details>
<summary><strong>Voice-Based Chatbots for English Speaking Practice in Multilingual Low-Resource Indian Schools: A Multi-Stakeholder Study</strong> - Sneha Shashidhara, Vivienne Bihe Chi, Abhay P Singh, Lyle Ungar, Sharath Chandra Guntuku - [[pdf]](https://arxiv.org/pdf/2601.19304)</summary>

**Abstract:** Spoken English proficiency is a powerful driver of economic mobility for low-income Indian youth, yet opportunities for spoken practice remain scarce in schools. We investigate the deployment of a voice-based chatbot for English conversation practice across four low-resource schools in Delhi. Through a six-day field study combining observations and interviews, we captured the perspectives of students, teachers, and principals. Findings confirm high demand across all groups, with notable gains in student speaking confidence. Our multi-stakeholder analysis surfaced a tension in long-term adoption vision: students favored open-ended conversational practice, while administrators emphasized curriculum-aligned assessment. We offer design recommendations for voice-enabled chatbots in low-resource multilingual contexts, highlighting the need for more intelligible speech output for non-native learners, one-tap interactions with simplified interfaces, and actionable analytics for educators. Beyond language learning, our findings inform the co-design of future AI-based educational technologies that are socially sustainable within the complex ecosystem of low-resource schools.

**arXiv ID:** 2601.19304
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (17 papers)</h2></summary>

<details>
<summary><strong>Do LLMs Act Like Rational Agents? Measuring Belief Coherence in Probabilistic Decision Making</strong> - Khurram Yamin, Jingjing Tang, Santiago Cortes-Gomez, Amit Sharma, Eric Horvitz, Bryan Wilder - [[pdf]](https://arxiv.org/pdf/2602.06286)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as agents in high-stakes domains where optimal actions depend on both uncertainty about the world and consideration of utilities of different outcomes, yet their decision logic remains difficult to interpret. We study whether LLMs are rational utility maximizers with coherent beliefs and stable preferences. We consider behaviors of models for diagnosis challenge problems. The results provide insights about the relationship of LLM inferences to ideal Bayesian utility maximization for elicited probabilities and observed actions. Our approach provides falsifiable conditions under which the reported probabilities \emph{cannot} correspond to the true beliefs of any rational agent. We apply this methodology to multiple medical diagnostic domains with evaluations across several LLMs. We discuss implications of the results and directions forward for uses of LLMs in guiding high-stakes decisions.

**arXiv ID:** 2602.06286
</details>

<details>
<summary><strong>ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training</strong> - Dunwei Tu, Hongyan Hao, Hansi Yang, Yihao Chen, Yi-Kai Zhang, Zhikang Xia, Yu Yang, Yueqing Sun, Xingchen Liu, Furao Shen, Qi Gu, Hui Su, Xunliang Cai - [[pdf]](https://arxiv.org/pdf/2602.06820)</summary>

**Abstract:** Training generalist agents capable of adapting to diverse scenarios requires interactive environments for self-exploration. However, interactive environments remain critically scarce, and existing synthesis methods suffer from significant limitations regarding environmental diversity and scalability. To address these challenges, we introduce ScaleEnv, a framework that constructs fully interactive environments and verifiable tasks entirely from scratch. Specifically, ScaleEnv ensures environment reliability through procedural testing, and guarantees task completeness and solvability via tool dependency graph expansion and executable action verification. By enabling agents to learn through exploration within ScaleEnv, we demonstrate significant performance improvements on unseen, multi-turn tool-use benchmarks such as $\tau^2$-Bench and VitaBench, highlighting strong generalization capabilities. Furthermore, we investigate the relationship between increasing number of domains and model generalization performance, providing empirical evidence that scaling environmental diversity is critical for robust agent learning.

**arXiv ID:** 2602.06820
</details>

<details>
<summary><strong>From Features to Actions: Explainability in Traditional and Agentic AI Systems</strong> - Sindhuja Chaduvula, Jessee Ho, Kina Kim, Aravind Narayanan, Mahshid Alinoori, Muskan Garg, Dhanesh Ramachandram, Shaina Raza - [[pdf]](https://arxiv.org/pdf/2602.06841)</summary>

**Abstract:** Over the last decade, explainable AI has primarily focused on interpreting individual model predictions, producing post-hoc explanations that relate inputs to outputs under a fixed decision structure. Recent advances in large language models (LLMs) have enabled agentic AI systems whose behaviour unfolds over multi-step trajectories. In these settings, success and failure are determined by sequences of decisions rather than a single output. While useful, it remains unclear how explanation approaches designed for static predictions translate to agentic settings where behaviour emerges over time. In this work, we bridge the gap between static and agentic explainability by comparing attribution-based explanations with trace-based diagnostics across both settings. To make this distinction explicit, we empirically compare attribution-based explanations used in static classification tasks with trace-based diagnostics used in agentic benchmarks (TAU-bench Airline and AssistantBench). Our results show that while attribution methods achieve stable feature rankings in static settings (Spearman $\rho = 0.86$), they cannot be applied reliably to diagnose execution-level failures in agentic trajectories. In contrast, trace-grounded rubric evaluation for agentic settings consistently localizes behaviour breakdowns and reveals that state tracking inconsistency is 2.7$\times$ more prevalent in failed runs and reduces success probability by 49\%. These findings motivate a shift towards trajectory-level explainability for agentic systems when evaluating and diagnosing autonomous AI behaviour.
Resources:
this https URL this https URL

**arXiv ID:** 2602.06841
</details>

<details>
<summary><strong>Rethinking Memory Mechanisms of Foundation Agents in the Second Half</strong> - Wei-Chieh Huang, Weizhi Zhang, Yueqing Liang, Yuanchen Bei, Yankai Chen, Tao Feng, Xinyu Pan, Zhen Tan, Yu Wang, Tianxin Wei, Shanglin Wu, Ruiyao Xu, Liangwei Yang, Rui Yang, Wooseong Yang, Chin-Yuan Yeh, Hanrong Zhang, Haozhen Zhang, Siqi Zhu, Henry Peng Zou, Wanjia Zhao, Song Wang, Wujiang Xu, Zixuan Ke, Zheng Hui, Dawei Li, Yaozu Wu, Langzhou He, Chen Wang, Xiongxiao Xu, Baixiang Huang, Juntao Tan, Shelby Heinecke, Huan Wang, Caiming Xiong, Ahmed A. Metwally, Jun Yan, Chen-Yu Lee, Hanqing Zeng, Yinglong Xia, Xiaokai Wei, Ali Payani, Yu Wang, Haitong Ma, Wenya Wang, Chengguang Wang, Yu Zhang, Xin Wang, Yongfeng Zhang, Jiaxuan You, Hanghang Tong, Xiao Luo, Yizhou Sun, Wei Wang, Julian McAuley, James Zou, Jiawei Han, Philip S. Yu, Kai Shu - [[pdf]](https://arxiv.org/pdf/2602.06052)</summary>

**Abstract:** The research of artificial intelligence is undergoing a paradigm shift from prioritizing model innovations over benchmark scores towards emphasizing problem definition and rigorous real-world evaluation. As the field enters the "second half," the central challenge becomes real utility in long-horizon, dynamic, and user-dependent environments, where agents face context explosion and must continuously accumulate, manage, and selectively reuse large volumes of information across extended interactions. Memory, with hundreds of papers released this year, therefore emerges as the critical solution to fill the utility gap. In this survey, we provide a unified view of foundation agent memory along three dimensions: memory substrate (internal and external), cognitive mechanism (episodic, semantic, sensory, working, and procedural), and memory subject (agent- and user-centric). We then analyze how memory is instantiated and operated under different agent topologies and highlight learning policies over memory operations. Finally, we review evaluation benchmarks and metrics for assessing memory utility, and outline various open challenges and future directions.

**arXiv ID:** 2602.06052
</details>

<details>
<summary><strong>Coding Agents with Environment Interaction: A Theoretical Perspective</strong> - Nicolas Menet, Michael Hersche, Andreas Krause, Abbas Rahimi - [[pdf]](https://arxiv.org/pdf/2602.06098)</summary>

**Abstract:** Coding agents are increasingly utilized in test-driven software development, yet the theoretical mechanisms behind their environment-interaction strategies remain underexplored. We provide a probabilistic framework for two dominant paradigms: code selection after generation using the execution environment, and code generation conditioned on environment feedback. First, we formalize several well-established selection heuristics as environment-aware estimators of code correctness. We theoretically prove that estimators based on fuzzy functional similarity add an inductive bias and strictly dominate estimators based on functional equivalence in terms of signal-to-noise ratio. Second, we frame backprompting as an in-context approximation of Thompson sampling. We derive a novel regret bound for reward functions with unobservable components, theoretically explaining why the effectiveness of backprompting is limited by the ambiguity of the informal task description (an irreducible regret). Using three state-of-the-art open weight models, we corroborate these findings across BigCodeBenchHard, LeetCodeDataset, and QiskitHumanEvalSim. Our formalization also suggests how to improve task descriptions effectively, leading to a new benchmark, QiskitHumanEvalSimX.

**arXiv ID:** 2602.06098
</details>

<details>
<summary><strong>Addressing the Waypoint-Action Gap in End-to-End Autonomous Driving via Vehicle Motion Models</strong> - Jorge Daniel Rodríguez-Vidal, Gabriel Villalonga, Diego Porres, Antonio M. López Peña - [[pdf]](https://arxiv.org/pdf/2602.06214)</summary>

**Abstract:** End-to-End Autonomous Driving (E2E-AD) systems are typically grouped by the nature of their outputs: (i) waypoint-based models that predict a future trajectory, and (ii) action-based models that directly output throttle, steer and brake. Most recent benchmark protocols and training pipelines are waypoint-based, which makes action-based policies harder to train and compare, slowing their progress. To bridge this waypoint-action gap, we propose a novel, differentiable vehicle-model framework that rolls out predicted action sequences to their corresponding ego-frame waypoint trajectories while supervising in waypoint space. Our approach enables action-based architectures to be trained and evaluated, for the first time, within waypoint-based benchmarks without modifying the underlying evaluation protocol. We extensively evaluate our framework across multiple challenging benchmarks and observe consistent improvements over the baselines. In particular, on NAVSIM \texttt{navhard} our approach achieves state-of-the-art performance. Our code will be made publicly available upon acceptance.

**arXiv ID:** 2602.06214
</details>

<details>
<summary><strong>Conversational Intent-Driven GraphRAG: Enhancing Multi-Turn Dialogue Systems through Adaptive Dual-Retrieval of Flow Patterns and Context Semantics</strong> - Ziqi Zhu, Tao Hu, Honglong Zhang, Dan Yang, HanGeng Chen, Mengran Zhang, Xilun Chen - [[pdf]](https://arxiv.org/pdf/2506.19385)</summary>

**Abstract:** We present CID-GraphRAG (Conversational Intent-Driven Graph Retrieval Augmented Generation), a novel framework that addresses the limitations of existing dialogue systems in maintaining both contextual coherence and goal-oriented progression in multi-turn customer service conversations. Unlike traditional RAG systems that rely solely on semantic similarity (Conversation RAG) or standard knowledge graphs (GraphRAG), CID-GraphRAG constructs dynamic intent transition graphs from goal achieved historical dialogues and implements a dual-retrieval mechanism that adaptively balances intent-based graph traversal with semantic search. This approach enables the system to simultaneously leverage both conversional intent flow patterns and contextual semantics, significantly improving retrieval quality and response quality. In extensive experiments on real-world customer service dialogues, we employ both automatic metrics and LLM-as-judge assessments, demonstrating that CID-GraphRAG significantly outperforms both semantic-based Conversation RAG and intent-based GraphRAG baselines across all evaluation criteria. Quantitatively, CID-GraphRAG demonstrates substantial improvements over Conversation RAG across automatic metrics, with relative gains of 11% in BLEU, 5% in ROUGE-L, 6% in METEOR, and most notably, a 58% improvement in response quality according to LLM-as-judge evaluations. These results demonstrate that the integration of intent transition structures with semantic retrieval creates a synergistic effect that neither approach achieves independently, establishing CID-GraphRAG as an effective framework for addressing the challenges of maintaining contextual coherence and goal-oriented progression in knowledge-intensive multi-turn dialogues.

**arXiv ID:** 2506.19385
</details>

<details>
<summary><strong>MirrorBench: A Benchmark to Evaluate Conversational User-Proxy Agents for Human-Likeness</strong> - Ashutosh Hathidara, Julien Yu, Vaishali Senthil, Sebastian Schreiber, Anil Babu Ankisettipalli - [[pdf]](https://arxiv.org/pdf/2601.08118)</summary>

**Abstract:** Large language models (LLMs) are increasingly used as human simulators, both for evaluating conversational systems and for generating fine-tuning data. However, naive "act-as-a-user" prompting often yields verbose, unrealistic utterances, motivating principled evaluation of *user proxy agents*. We present **MirrorBench**, a reproducible and extensible benchmarking framework that evaluates user proxies solely on their ability to produce human-like user utterances across diverse conversational regimes, explicitly decoupled from downstream task success. **MirrorBench** combines three lexical-diversity metrics (**MATTR**, **Yule's~$K$**, and **HD-D**) with three LLM-judge-based metrics (**GTEval**, **Pairwise Indistinguishability**, and **Rubric-and-Reason**), and contextualizes judge scores using Human-Human and Proxy-Proxy calibration controls. Across four public datasets, **MirrorBench** yields variance-aware comparisons and reveals systematic gaps between user proxies and real human users. The framework is [open source](this https URL) and includes a command-line interface for running and managing user-proxy benchmarking experiments.

**arXiv ID:** 2601.08118
</details>

<details>
<summary><strong>Yunjue Agent Tech Report: A Fully Reproducible, Zero-Start In-Situ Self-Evolving Agent System for Open-Ended Tasks</strong> - Haotian Li, Shijun Yang, Weizhen Qi, Silei Zhao, Rui Hua, Mingzhu Song, Xiaojian Yang, Chao Peng - [[pdf]](https://arxiv.org/pdf/2601.18226)</summary>

**Abstract:** Conventional agent systems often struggle in open-ended environments where task distributions continuously drift and external supervision is scarce. Their reliance on static toolsets or offline training lags behind these dynamics, leaving the system's capability boundaries rigid and unknown. To address this, we propose the In-Situ Self-Evolving paradigm. This approach treats sequential task interactions as a continuous stream of experience, enabling the system to distill short-term execution feedback into long-term, reusable capabilities without access to ground-truth labels. Within this framework, we identify tool evolution as the critical pathway for capability expansion, which provides verifiable, binary feedback signals. Within this framework, we develop Yunjue Agent, a system that iteratively synthesizes, optimizes, and reuses tools to navigate emerging challenges. To optimize evolutionary efficiency, we further introduce a Parallel Batch Evolution strategy. Empirical evaluations across five diverse benchmarks under a zero-start setting demonstrate significant performance gains over proprietary baselines. Additionally, complementary warm-start evaluations confirm that the accumulated general knowledge can be seamlessly transferred to novel domains. Finally, we propose a novel metric to monitor evolution convergence, serving as a function analogous to training loss in conventional optimization. We open-source our codebase, system traces, and evolved tools to facilitate future research in resilient, self-evolving intelligence.

**arXiv ID:** 2601.18226
</details>

<details>
<summary><strong>DarkEQA: Benchmarking Vision-Language Models for Embodied Question Answering in Low-Light Indoor Environments</strong> - Yohan Park, Hyunwoo Ha, Wonjun Jo, Tae-Hyun Oh - [[pdf]](https://arxiv.org/pdf/2512.24985)</summary>

**Abstract:** Vision Language Models (VLMs) are increasingly adopted as central reasoning modules for embodied agents. Existing benchmarks evaluate their capabilities under ideal, well-lit conditions, yet robust 24/7 operation demands performance under a wide range of visual degradations, including low-light conditions at night or in dark environments--a core necessity that has been largely overlooked. To address this underexplored challenge, we present DarkEQA, an open-source benchmark for evaluating EQA-relevant perceptual primitives under multi-level low-light conditions. DarkEQA isolates the perception bottleneck by evaluating question answering from egocentric observations under controlled degradations, enabling attributable robustness analysis. A key design feature of DarkEQA is its physical fidelity: visual degradations are modeled in linear RAW space, simulating physics-based illumination drop and sensor noise followed by an ISP-inspired rendering pipeline. We demonstrate the utility of DarkEQA by evaluating a wide range of state-of-the-art VLMs and Low-Light Image Enhancement (LLIE) models. Our analysis systematically reveals VLMs' limitations when operating under these challenging visual conditions. Project website: this https URL

**arXiv ID:** 2512.24985
</details>

<details>
<summary><strong>OmniCode: A Benchmark for Evaluating Software Engineering Agents</strong> - Atharv Sonwane, Eng-Shen Tu, Wei-Chung Lu, Claas Beger, Carter Larsen, Debjit Dhar, Simon Alford, Rachel Chen, Ronit Pattanayak, Tuan Anh Dang, Guohao Chen, Gloria Geng, Kevin Ellis, Saikat Dutta - [[pdf]](https://arxiv.org/pdf/2602.02262)</summary>

**Abstract:** LLM-powered coding agents are redefining how real-world software is developed. To drive the research towards better coding agents, we require challenging benchmarks that can rigorously evaluate the ability of such agents to perform various software engineering tasks. However, popular coding benchmarks such as HumanEval and SWE-Bench focus on narrowly scoped tasks such as competition programming and patch generation. In reality, software engineers have to handle a broader set of tasks for real-world software development. To address this gap, we propose OmniCode, a novel software engineering benchmark that contains a broader and more diverse set of task categories beyond code or patch generation. Overall, OmniCode contains 1794 tasks spanning three programming languages (Python, Java, and C++) and four key categories: bug fixing, test generation, code review fixing, and style fixing. In contrast to prior software engineering benchmarks, the tasks in OmniCode are (1) manually validated to eliminate ill-defined problems, and (2) synthetically crafted or recently curated to avoid data leakage issues, presenting a new framework for synthetically generating diverse software tasks from limited real-world data. We evaluate OmniCode with popular agent frameworks such as SWE-Agent and show that while they may perform well on bug fixing for Python, they fall short on tasks such as Test Generation and in languages such as C++ and Java. For instance, SWE-Agent achieves a maximum of 20.9% with DeepSeek-V3.1 on Java Test Generation tasks. OmniCode aims to serve as a robust benchmark and spur the development of agents that can perform well across different aspects of software development. Code and data are available at this https URL.

**arXiv ID:** 2602.02262
</details>

<details>
<summary><strong>CAST: Character-and-Scene Episodic Memory for Agents</strong> - Kexin Ma, Bojun Li, Yuhua Tang, Ruochun Jin, Liting Sun - [[pdf]](https://arxiv.org/pdf/2602.06051)</summary>

**Abstract:** Episodic memory is a central component of human memory, which refers to the ability to recall coherent events grounded in who, when, and where. However, most agent memory systems only emphasize semantic recall and treat experience as structures such as key-value, vector, or graph, which makes them struggle to represent and retrieve coherent events. To address this challenge, we propose a Character-and-Scene based memory architecture(CAST) inspired by dramatic theory. Specifically, CAST constructs 3D scenes (time/place/topic) and organizes them into character profiles that summarize the events of a character to represent episodic memory. Moreover, CAST complements this episodic memory with a graph-based semantic memory, which yields a robust dual memory design. Experiments demonstrate that CAST has averagely improved 8.11% F1 and 10.21% J(LLM-as-a-Judge) than baselines on various datasets, especially on open and time-sensitive conversational questions.

**arXiv ID:** 2602.06051
</details>

<details>
<summary><strong>SAGE: Benchmarking and Improving Retrieval for Deep Research Agents</strong> - Tiansheng Hu, Yilun Zhao, Canyu Zhang, Arman Cohan, Chen Zhao - [[pdf]](https://arxiv.org/pdf/2602.05975)</summary>

**Abstract:** Deep research agents have emerged as powerful systems for addressing complex queries. Meanwhile, LLM-based retrievers have demonstrated strong capability in following instructions or reasoning. This raises a critical question: can LLM-based retrievers effectively contribute to deep research agent workflows? To investigate this, we introduce SAGE, a benchmark for scientific literature retrieval comprising 1,200 queries across four scientific domains, with a 200,000 paper retrieval corpus. We evaluate six deep research agents and find that all systems struggle with reasoning-intensive retrieval. Using DR Tulu as backbone, we further compare BM25 and LLM-based retrievers (i.e., ReasonIR and gte-Qwen2-7B-instruct) as alternative search tools. Surprisingly, BM25 significantly outperforms LLM-based retrievers by approximately 30%, as existing agents generate keyword-oriented sub-queries. To improve performance, we propose a corpus-level test-time scaling framework that uses LLMs to augment documents with metadata and keywords, making retrieval easier for off-the-shelf retrievers. This yields 8% and 2% gains on short-form and open-ended questions, respectively.

**arXiv ID:** 2602.05975
</details>

<details>
<summary><strong>User-Centric Object Navigation: A Benchmark with Integrated User Habits for Personalized Embodied Object Search</strong> - Hongcheng Wang, Jinyu Zhu, Hao Dong - [[pdf]](https://arxiv.org/pdf/2602.06459)</summary>

**Abstract:** In the evolving field of robotics, the challenge of Object Navigation (ON) in household environments has attracted significant interest. Existing ON benchmarks typically place objects in locations guided by general scene priors, without accounting for the specific placement habits of individual users. This omission limits the adaptability of navigation agents in personalized household environments. To address this, we introduce User-centric Object Navigation (UcON), a new benchmark that incorporates user-specific object placement habits, referred to as user habits. This benchmark requires agents to leverage these user habits for more informed decision-making during navigation. UcON encompasses approximately 22,600 user habits across 489 object categories. UcON is, to our knowledge, the first benchmark that explicitly formalizes and evaluates habit-conditioned object navigation at scale and covers the widest range of target object categories. Additionally, we propose a habit retrieval module to extract and utilize habits related to target objects, enabling agents to infer their likely locations more effectively. Experimental results demonstrate that current SOTA methods exhibit substantial performance degradation under habit-driven object placement, while integrating user habits consistently improves success rates. Code is available at this https URL.

**arXiv ID:** 2602.06459
</details>

<details>
<summary><strong>A 26-Gram Butterfly-Inspired Robot Achieving Autonomous Tailless Flight</strong> - Weibin Gu, Chenrui Feng, Lian Liu, Chen Yang, Xingchi Jiao, Yuhe Ding, Xiaofei Shi, Chao Gao, Alessandro Rizzo, Guyue Zhou - [[pdf]](https://arxiv.org/pdf/2602.06811)</summary>

**Abstract:** Flapping-wing micro air vehicles (FWMAVs) have demonstrated remarkable bio-inspired agility, yet tailless two-winged configurations remain largely unexplored due to their complex fluid-structure and wing-body coupling. Here we present \textit{AirPulse}, a 26-gram butterfly-inspired FWMAV that achieves fully onboard, closed-loop, untethered flight without auxiliary control surfaces. The AirPulse robot replicates key biomechanical traits of butterfly flight, including low wing aspect ratio, compliant carbon-fiber-reinforced wings, and low-frequency, high-amplitude flapping that induces cyclic variations in the center of gravity and moment of inertia, producing characteristic body undulation. We establish a quantitative mapping between flapping modulation parameters and force-torque generation, and introduce the Stroke Timing Asymmetry Rhythm (STAR) generator, enabling smooth, stable, and linearly parameterized wingstroke asymmetry for flapping control. Integrating these with an attitude controller, the AirPulse robot maintains pitch and yaw stability despite strong oscillatory dynamics. Free-flight experiments demonstrate stable climbing and turning maneuvers via either angle offset or stroke timing modulation, marking the first onboard controlled flight of the lightest two-winged, tailless butterfly-inspired FWMAV reported in peer-reviewed literature. This work corroborates a foundational platform for lightweight, collision-proof FWMAVs, bridging biological inspiration with practical aerial robotics. Their non-invasive maneuverability is ideally suited for real-world applications, such as confined-space inspection and ecological monitoring, inaccessible to traditional drones, while their biomechanical fidelity provides a physical model to decode the principles underlying the erratic yet efficient flight of real butterflies.

**arXiv ID:** 2602.06811
</details>

<details>
<summary><strong>Bridging the Indoor-Outdoor Gap: Vision-Centric Instruction-Guided Embodied Navigation for the Last Meters</strong> - Yuxiang Zhao, Yirong Yang, Yanqing Zhu, Yanfen Shen, Chiyu Wang, Zhining Gu, Pei Shi, Wei Guo, Mu Xu - [[pdf]](https://arxiv.org/pdf/2602.06427)</summary>

**Abstract:** Embodied navigation holds significant promise for real-world applications such as last-mile delivery. However, most existing approaches are confined to either indoor or outdoor environments and rely heavily on strong assumptions, such as access to precise coordinate systems. While current outdoor methods can guide agents to the vicinity of a target using coarse-grained localization, they fail to enable fine-grained entry through specific building entrances, critically limiting their utility in practical deployment scenarios that require seamless outdoor-to-indoor transitions. To bridge this gap, we introduce a novel task: out-to-in prior-free instruction-driven embodied navigation. This formulation explicitly eliminates reliance on accurate external priors, requiring agents to navigate solely based on egocentric visual observations guided by instructions. To tackle this task, we propose a vision-centric embodied navigation framework that leverages image-based prompts to drive decision-making. Additionally, we present the first open-source dataset for this task, featuring a pipeline that integrates trajectory-conditioned video synthesis into the data generation process. Through extensive experiments, we demonstrate that our proposed method consistently outperforms state-of-the-art baselines across key metrics including success rate and path efficiency.

**arXiv ID:** 2602.06427
</details>

<details>
<summary><strong>PrefIx: Understand and Adapt to User Preference in Human-Agent Interaction</strong> - Jialin Li, Zhenhao Chen, Hanjun Luo, Hanan Salam - [[pdf]](https://arxiv.org/pdf/2602.06714)</summary>

**Abstract:** LLM-based agents can complete tasks correctly yet still frustrate users through poor interaction patterns, such as excessive confirmations, opaque reasoning, or misaligned pacing. Current benchmarks evaluate task accuracy but overlook how agents interact: whether they infer preferences from implicit cues, adapt dynamically, or maintain fine-grained interaction quality. We introduce Prefix, a configurable environment that evaluates both what agents accomplish and how they interact. Central to Prefix is the Interaction-as-a-Tool (IaaT) paradigm, which treats interaction behaviors as structured tool calls, unifying them with existing evaluation frameworks. We define 31 preference settings across 14 attributes and formalize user experience (UX) as a core metric alongside task accuracy. A composite LLM-as-a-Judge mechanism across seven UX dimensions achieves strong aggregate reliability (ICC > 0.79), high internal consistency (alpha = 0.943), and human correlation (rho = 0.52-0.78). Preference-aware agents show 7.6% average UX improvement and 18.5% gain in preference alignment. Our work is openly accessible.

**arXiv ID:** 2602.06714
</details>

</details>

<details open>
<summary><h2>LLM Agents (4 papers)</h2></summary>

<details>
<summary><strong>AIRS-Bench: a Suite of Tasks for Frontier AI Research Science Agents</strong> - Alisia Lupidi, Bhavul Gauri, Thomas Simon Foster, Bassel Al Omari, Despoina Magka, Alberto Pepe, Alexis Audran-Reiss, Muna Aghamelu, Nicolas Baldwin, Lucia Cipolina-Kun, Jean-Christophe Gagnon-Audet, Chee Hau Leow, Sandra Lefdal, Hossam Mossalam, Abhinav Moudgil, Saba Nazir, Emanuel Tewolde, Isabel Urrego, Jordi Armengol Estape, Amar Budhiraja, Gaurav Chaurasia, Abhishek Charnalia, Derek Dunfield, Karen Hambardzumyan, Daniel Izcovich, Martin Josifoski, Ishita Mediratta, Kelvin Niu, Parth Pathak, Michael Shvartsman, Edan Toledo, Anton Protopopov, Roberta Raileanu, Alexander Miller, Tatiana Shavrina, Jakob Foerster, Yoram Bachrach - [[pdf]](https://arxiv.org/pdf/2602.06855)</summary>

**Abstract:** LLM agents hold significant promise for advancing scientific research. To accelerate this progress, we introduce AIRS-Bench (the AI Research Science Benchmark), a suite of 20 tasks sourced from state-of-the-art machine learning papers. These tasks span diverse domains, including language modeling, mathematics, bioinformatics, and time series forecasting. AIRS-Bench tasks assess agentic capabilities over the full research lifecycle -- including idea generation, experiment analysis and iterative refinement -- without providing baseline code. The AIRS-Bench task format is versatile, enabling easy integration of new tasks and rigorous comparison across different agentic frameworks. We establish baselines using frontier models paired with both sequential and parallel scaffolds. Our results show that agents exceed human SOTA in four tasks but fail to match it in sixteen others. Even when agents surpass human benchmarks, they do not reach the theoretical performance ceiling for the underlying tasks. These findings indicate that AIRS-Bench is far from saturated and offers substantial room for improvement. We open-source the AIRS-Bench task definitions and evaluation code to catalyze further development in autonomous scientific research.

**arXiv ID:** 2602.06855
</details>

<details>
<summary><strong>TrajAD: Trajectory Anomaly Detection for Trustworthy LLM Agents</strong> - Yibing Liu, Chong Zhang, Zhongyi Han, Hansong Liu, Yong Wang, Yang Yu, Xiaoyan Wang, Yilong Yin - [[pdf]](https://arxiv.org/pdf/2602.06443)</summary>

**Abstract:** We address the problem of runtime trajectory anomaly detection, a critical capability for enabling trustworthy LLM agents. Current safety measures predominantly focus on static input/output filtering. However, we argue that ensuring LLM agents reliability requires auditing the intermediate execution process. In this work, we formulate the task of Trajectory Anomaly Detection. The goal is not merely detection, but precise error localization. This capability is essential for enabling efficient rollback-and-retry. To achieve this, we construct TrajBench, a dataset synthesized via a perturb-and-complete strategy to cover diverse procedural anomalies. Using this benchmark, we investigate the capability of models in process supervision. We observe that general-purpose LLMs, even with zero-shot prompting, struggle to identify and localize these anomalies. This reveals that generalized capabilities do not automatically translate to process reliability. To address this, we propose TrajAD, a specialized verifier trained with fine-grained process supervision. Our approach outperforms baselines, demonstrating that specialized supervision is essential for building trustworthy agents.

**arXiv ID:** 2602.06443
</details>

<details>
<summary><strong>Bridging Symbolic Control and Neural Reasoning in LLM Agents: Structured Cognitive Loop with a Governance Layer</strong> - Myung Ho Kim - [[pdf]](https://arxiv.org/pdf/2511.17673)</summary>

**Abstract:** Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). Soft Symbolic Control constitutes a dedicated governance layer within SCL, applying symbolic constraints to probabilistic inference while preserving the flexibility of neural reasoning and restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents.

**arXiv ID:** 2511.17673
</details>

<details>
<summary><strong>Towards Agentic Intelligence for Materials Science</strong> - Huan Zhang, Yizhan Li, Wenhao Huang, Ziyu Hou, Yu Song, Xuye Liu, Farshid Effaty, Jinya Jiang, Sifan Wu, Qianggang Ding, Izumi Takahara, Leonard R. MacGillivray, Teruyasu Mizoguchi, Tianshu Yu, Lizi Liao, Yuyu Luo, Yu Rong, Jia Li, Ying Diao, Heng Ji, Bang Liu - [[pdf]](https://arxiv.org/pdf/2602.00169)</summary>

**Abstract:** The convergence of artificial intelligence and materials science presents a transformative opportunity, but achieving true acceleration in discovery requires moving beyond task-isolated, fine-tuned models toward agentic systems that plan, act, and learn across the full discovery loop. This survey advances a unique pipeline-centric view that spans from corpus curation and pretraining, through domain adaptation and instruction tuning, to goal-conditioned agents interfacing with simulation and experimental platforms. Unlike prior reviews, we treat the entire process as an end-to-end system to be optimized for tangible discovery outcomes rather than proxy benchmarks. This perspective allows us to trace how upstream design choices-such as data curation and training objectives-can be aligned with downstream experimental success through effective credit assignment.
To bridge communities and establish a shared frame of reference, we first present an integrated lens that aligns terminology, evaluation, and workflow stages across AI and materials science. We then analyze the field through two focused lenses: From the AI perspective, the survey details LLM strengths in pattern recognition, predictive analytics, and natural language processing for literature mining, materials characterization, and property prediction; from the materials science perspective, it highlights applications in materials design, process optimization, and the acceleration of computational workflows via integration with external tools (e.g., DFT, robotic labs). Finally, we contrast passive, reactive approaches with agentic design, cataloging current contributions while motivating systems that pursue long-horizon goals with autonomy, memory, and tool use. This survey charts a practical roadmap towards autonomous, safety-aware LLM agents aimed at discovering novel and useful materials.

**arXiv ID:** 2602.00169
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (18 papers)</h2></summary>

<details>
<summary><strong>SeeUPO: Sequence-Level Agentic-RL with Convergence Guarantees</strong> - Tianyi Hu, Qingxu Fu, Yanxi Chen, Zhaoyang Liu, Bolin Ding - [[pdf]](https://arxiv.org/pdf/2602.06554)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as the predominant paradigm for training large language model (LLM)-based AI agents. However, existing backbone RL algorithms lack verified convergence guarantees in agentic scenarios, especially in multi-turn settings, which can lead to training instability and failure to converge to optimal policies.
In this paper, we systematically analyze how different combinations of policy update mechanisms and advantage estimation methods affect convergence properties in single/multi-turn scenarios. We find that REINFORCE with Group Relative Advantage Estimation (GRAE) can converge to the globally optimal under undiscounted conditions, but the combination of PPO & GRAE breaks PPO's original monotonic improvement property. Furthermore, we demonstrate that mainstream backbone RL algorithms cannot simultaneously achieve both critic-free and convergence guarantees in multi-turn scenarios.
To address this, we propose SeeUPO (Sequence-level Sequential Update Policy Optimization), a critic-free approach with convergence guarantees for multi-turn interactions. SeeUPO models multi-turn interaction as sequentially executed multi-agent bandit problems. Through turn-by-turn sequential policy updates in reverse execution order, it ensures monotonic improvement and convergence to global optimal solution via backward induction.
Experiments on AppWorld and BFCL v4 demonstrate SeeUPO's substantial improvements over existing backbone algorithms: relative gains of 43.3%-54.6% on Qwen3-14B and 24.1%-41.9% on Qwen2.5-14B (averaged across benchmarks), along with superior training stability.

**arXiv ID:** 2602.06554
</details>

<details>
<summary><strong>RuleSmith: Multi-Agent LLMs for Automated Game Balancing</strong> - Ziyao Zeng, Chen Liu, Tianyu Liu, Hao Wang, Xiatao Sun, Fengyu Yang, Xiaofeng Liu, Zhiwen Fan - [[pdf]](https://arxiv.org/pdf/2602.06232)</summary>

**Abstract:** Game balancing is a longstanding challenge requiring repeated playtesting, expert intuition, and extensive manual tuning. We introduce RuleSmith, the first framework that achieves automated game balancing by leveraging the reasoning capabilities of multi-agent LLMs. It couples a game engine, multi-agent LLMs self-play, and Bayesian optimization operating over a multi-dimensional rule space. As a proof of concept, we instantiate RuleSmith on CivMini, a simplified civilization-style game containing heterogeneous factions, economy systems, production rules, and combat mechanics, all governed by tunable parameters. LLM agents interpret textual rulebooks and game states to generate actions, to conduct fast evaluation of balance metrics such as win-rate disparities. To search the parameter landscape efficiently, we integrate Bayesian optimization with acquisition-based adaptive sampling and discrete projection: promising candidates receive more evaluation games for accurate assessment, while exploratory candidates receive fewer games for efficient exploration. Experiments show that RuleSmith converges to highly balanced configurations and provides interpretable rule adjustments that can be directly applied to downstream game systems. Our results illustrate that LLM simulation can serve as a powerful surrogate for automating design and balancing in complex multi-agent environments.

**arXiv ID:** 2602.06232
</details>

<details>
<summary><strong>Prism: Spectral Parameter Sharing for Multi-Agent Reinforcement Learning</strong> - Kyungbeom Kim, Seungwon Oh, Kyung-Joong Kim - [[pdf]](https://arxiv.org/pdf/2602.06476)</summary>

**Abstract:** Parameter sharing is a key strategy in multi-agent reinforcement learning (MARL) for improving scalability, yet conventional fully shared architectures often collapse into homogeneous behaviors. Recent methods introduce diversity through clustering, pruning, or masking, but typically compromise resource efficiency. We propose Prism, a parameter sharing framework that induces inter-agent diversity by representing shared networks in the spectral domain via singular value decomposition (SVD). All agents share the singular vector directions while learning distinct spectral masks on singular values. This mechanism encourages inter-agent diversity and preserves scalability. Extensive experiments on both homogeneous (LBF, SMACv2) and heterogeneous (MaMuJoCo) benchmarks show that Prism achieves competitive performance with superior resource efficiency.

**arXiv ID:** 2602.06476
</details>

<details>
<summary><strong>Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevant Assessment for IR Benchmarks</strong> - Minjeong Ban, Jeonghwan Choi, Hyangsuk Min, Nicole Hee-Yeon Kim, Minseok Kim, Jae-Gil Lee, Hwanjun Song - [[repo]](https://github.com/DISL-Lab/DREAM-ICLR-26) [[pdf]](https://arxiv.org/pdf/2602.06526)</summary>

**Abstract:** Information retrieval (IR) evaluation remains challenging due to incomplete IR benchmark datasets that contain unlabeled relevant chunks. While LLMs and LLM-human hybrid strategies reduce costly human effort, they remain prone to LLM overconfidence and ineffective AI-to-human escalation. To address this, we propose DREAM, a multi-round debate-based relevance assessment framework with LLM agents, built on opposing initial stances and iterative reciprocal critique. Through our agreement-based debate, it yields more accurate labeling for certain cases and more reliable AI-to-human escalation for uncertain ones, achieving 95.2% labeling accuracy with only 3.5% human involvement. Using DREAM, we build BRIDGE, a refined benchmark that mitigates evaluation bias and enables fairer retriever comparison by uncovering 29,824 missing relevant chunks. We then re-benchmark IR systems and extend evaluation to RAG, showing that unaddressed holes not only distort retriever rankings but also drive retrieval-generation misalignment. The relevance assessment framework is available at https: //github.com/DISL-Lab/DREAM-ICLR-26; and the BRIDGE dataset is available at this https URL.

**arXiv ID:** 2602.06526
</details>

<details>
<summary><strong>Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding</strong> - Rishabh Jain, Keisuke Okumura, Michael Amir, Pietro Lio, Amanda Prorok - [[pdf]](https://arxiv.org/pdf/2602.06733)</summary>

**Abstract:** Multi-Agent Path Finding (MAPF) is a representative multi-agent coordination problem, where multiple agents are required to navigate to their respective goals without collisions. Solving MAPF optimally is known to be NP-hard, leading to the adoption of learning-based approaches to alleviate the online computational burden. Prevailing approaches, such as Graph Neural Networks (GNNs), are typically constrained to pairwise message passing between agents. However, this limitation leads to suboptimal behaviours and critical issues, such as attention dilution, particularly in dense environments where group (i.e. beyond just two agents) coordination is most critical. Despite the importance of such higher-order interactions, existing approaches have not been able to fully explore them. To address this representational bottleneck, we introduce HMAGAT (Hypergraph Multi-Agent Attention Network), a novel architecture that leverages attentional mechanisms over directed hypergraphs to explicitly capture group dynamics. Empirically, HMAGAT establishes a new state-of-the-art among learning-based MAPF solvers: e.g., despite having just 1M parameters and being trained on 100$\times$ less data, it outperforms the current SoTA 85M parameter model. Through detailed analysis of HMAGAT's attention values, we demonstrate how hypergraph representations mitigate the attention dilution inherent in GNNs and capture complex interactions where pairwise methods fail. Our results illustrate that appropriate inductive biases are often more critical than the training data size or sheer parameter count for multi-agent problems.

**arXiv ID:** 2602.06733
</details>

<details>
<summary><strong>TraceCoder: A Trace-Driven Multi-Agent Framework for Automated Debugging of LLM-Generated Code</strong> - Jiangping Huang, Wenguang Ye, Weisong Sun, Jian Zhang, Mingyue Zhang, Yang Liu - [[pdf]](https://arxiv.org/pdf/2602.06875)</summary>

**Abstract:** Large Language Models (LLMs) often generate code with subtle but critical bugs, especially for complex tasks. Existing automated repair methods typically rely on superficial pass/fail signals, offering limited visibility into program behavior and hindering precise error localization. In addition, without a way to learn from prior failures, repair processes often fall into repetitive and inefficient cycles. To overcome these challenges, we present TraceCoder, a collaborative multi-agent framework that emulates the observe-analyze-repair process of human experts. The framework first instruments the code with diagnostic probes to capture fine-grained runtime traces, enabling deep insight into its internal execution. It then conducts causal analysis on these traces to accurately identify the root cause of the failure. This process is further enhanced by a novel Historical Lesson Learning Mechanism (HLLM), which distills insights from prior failed repair attempts to inform subsequent correction strategies and prevent recurrence of similar mistakes. To ensure stable convergence, a Rollback Mechanism enforces that each repair iteration constitutes a strict improvement toward the correct solution. Comprehensive experiments across multiple benchmarks show that TraceCoder achieves up to a 34.43\% relative improvement in Pass@1 accuracy over existing advanced baselines. Ablation studies verify the significance of each system component, with the iterative repair process alone contributing a 65.61\% relative gain in accuracy. Furthermore, TraceCoder significantly outperforms leading iterative methods in terms of both accuracy and cost-efficiency.

**arXiv ID:** 2602.06875
</details>

<details>
<summary><strong>OpenDeception: Learning Deception and Trust in Human-AI Interaction via Multi-Agent Simulation</strong> - Yichen Wu, Qianqian Gao, Xudong Pan, Geng Hong, Min Yang - [[pdf]](https://arxiv.org/pdf/2504.13707)</summary>

**Abstract:** As large language models (LLMs) are increasingly deployed as interactive agents, open-ended human-AI interactions can involve deceptive behaviors with serious real-world consequences, yet existing evaluations remain largely scenario-specific and model-centric. We introduce OpenDeception, a lightweight framework for jointly evaluating deception risk from both sides of human-AI dialogue. It consists of a scenario benchmark with 50 real-world deception cases, an IntentNet that infers deceptive intent from agent reasoning, and a TrustNet that estimates user susceptibility. To address data scarcity, we synthesize high-risk dialogues via LLM-based role-and-goal simulation, and train the User Trust Scorer using contrastive learning on controlled response pairs, avoiding unreliable scalar labels. Experiments on 11 LLMs and three large reasoning models show that over 90% of goal-driven interactions in most models exhibit deceptive intent, with stronger models displaying higher risk. A real-world case study adapted from a documented AI-induced suicide incident further demonstrates that our joint evaluation can proactively trigger warnings before critical trust thresholds are reached.

**arXiv ID:** 2504.13707
</details>

<details>
<summary><strong>Doctor-R1: Mastering Clinical Inquiry with Experiential Agentic Reinforcement Learning</strong> - Yunghwei Lai, Kaiming Liu, Ziyue Wang, Weizhi Ma, Yang Liu - [[pdf]](https://arxiv.org/pdf/2510.04284)</summary>

**Abstract:** The professionalism of a human doctor in outpatient service depends on two core abilities: the ability to make accurate medical decisions and the medical consultation skill to conduct strategic, empathetic patient inquiry. Existing Large Language Models (LLMs) have achieved remarkable accuracy on medical decision-making benchmarks. However, they often lack the ability to conduct the strategic and empathetic consultation, which is essential for real-world clinical scenarios. To address this gap, we propose Doctor-R1, an AI doctor agent trained to master both of the capabilities by ask high-yield questions and conduct strategic multi-turn inquiry to guide decision-making. Our framework introduces three key components: a multi-agent interactive environment, a two-tiered reward architecture that separately optimizes clinical decision-making and communicative inquiry skills, and an experience repository to ground policy learning in high-quality prior trajectories. We evaluate Doctor-R1 on OpenAI's HealthBench and MAQuE, assessed across multi-facet metrics, such as communication quality, user experience, and task accuracy. Remarkably, Doctor-R1 surpasses state-of-the-art open-source specialized LLMs by a substantial margin with higher parameter efficiency and outperforms powerful proprietary models. Furthermore, human expert evaluations show that Doctor-R1 achieves superior clinical capability and patient-centric performance, demonstrating the effectiveness of the framework.

**arXiv ID:** 2510.04284
</details>

<details>
<summary><strong>Scaling Multi-Agent Epistemic Planning through GNN-Derived Heuristics</strong> - Giovanni Briglia, Francesco Fabiano, Stefano Mariani - [[pdf]](https://arxiv.org/pdf/2508.12840)</summary>

**Abstract:** Multi-agent Epistemic Planning (MEP) is an autonomous planning framework for reasoning about both the physical world and the beliefs of agents, with applications in domains where information flow and awareness among agents are critical. The richness of MEP requires states to be represented as Kripke structures, i.e., directed labeled graphs. This representation limits the applicability of existing heuristics, hindering the scalability of epistemic solvers, which must explore an exponential search space without guidance, resulting often in intractability. To address this, we exploit Graph Neural Networks (GNNs) to learn patterns and relational structures within epistemic states, to guide the planning process. GNNs, which naturally capture the graph-like nature of Kripke models, allow us to derive meaningful estimates of state quality -- e.g., the distance from the nearest goal -- by generalizing knowledge obtained from previously solved planning instances. We integrate these predictive heuristics into an epistemic planning pipeline and evaluate them against standard baselines, showing improvements in the scalability of multi-agent epistemic planning.

**arXiv ID:** 2508.12840
</details>

<details>
<summary><strong>Balancing Sustainability And Performance: The Role Of Small-Scale LLMs In Agentic Artificial Intelligence Systems</strong> - Anh Khoa Ngo Ho, Martin Chauvin, Simon Gosset, Philippe Cordier, Boris Gamazaychikov - [[pdf]](https://arxiv.org/pdf/2601.19311)</summary>

**Abstract:** As large language models become integral to agentic artificial intelligence systems, their energy demands during inference may pose significant sustainability challenges. This study investigates whether deploying smaller-scale language models can reduce energy consumption without compromising responsiveness and output quality in a multi-agent, real-world environments. We conduct a comparative analysis across language models of varying scales to quantify trade-offs between efficiency and performance. Results show that smaller open-weights models can lower energy usage while preserving task quality. Building on these findings, we propose practical guidelines for sustainable artificial intelligence design, including optimal batch size configuration and computation resource allocation. These insights offer actionable strategies for developing scalable, environmentally responsible artificial intelligence systems.

**arXiv ID:** 2601.19311
</details>

<details>
<summary><strong>PiFlow: Principle-Aware Scientific Discovery with Multi-Agent Collaboration</strong> - Yingming Pu, Tao Lin, Hongyu Chen - [[pdf]](https://arxiv.org/pdf/2505.15047)</summary>

**Abstract:** Large Language Model (LLM)-based multi-agent systems (MAS) demonstrate remarkable potential for scientific discovery. Existing approaches, however, often automate scientific discovery using predefined workflows that lack rationality constraints. This often leads to aimless hypothesizing and a failure to consistently link hypotheses with evidence, thereby hindering the systematic reduction of uncertainty. Overcoming these limitations fundamentally requires a principled approach to exploration. We introduce PiFlow, an information-theoretical framework, treating automated scientific discovery as a structured uncertainty reduction problem guided by principles (e.g., scientific laws). Extensive evaluations across three distinct scientific domains demonstrate that PiFlow (I) improves discovery efficiency by 31.18%~41.73% and solution quality by 12.47%~31.72% against state-of-the-art methods, (II) delivers a 5.6x speedup in time-to-solution while reducing token consumption by up to 27% compared to vanilla agents, and (III) serves as a Plug-and-Play module that generalizes on existing agent architecture. Overall, PiFlow establishes a novel paradigm shift in highly efficient agentic scientific discovery, paving the way for more robust and accelerated AI-driven research.

**arXiv ID:** 2505.15047
</details>

<details>
<summary><strong>Constella: Supporting Storywriters' Interconnected Character Creation through LLM-based Multi-Agents</strong> - Syemin Park, Soobin Park, Youn-kyung Lim - [[pdf]](https://arxiv.org/pdf/2507.05820)</summary>

**Abstract:** Creating a cast of characters by attending to their relational dynamics is a critical aspect of most long-form storywriting. However, our formative study (N=14) reveals that writers struggle to envision new characters that could influence existing ones, balance similarities and differences among characters, and intricately flesh out their relationships. Based on these observations, we designed Constella, an LLM-based multi-agent tool that supports storywriters' interconnected character creation process. Constella suggests related characters (FRIENDS DISCOVERY feature), reveals the inner mindscapes of several characters simultaneously (JOURNALS feature), and manifests relationships through inter-character responses (COMMENTS feature). Our 7-8 day deployment study with storywriters (N=11) shows that Constella enabled the creation of expansive communities composed of related characters, facilitated the comparison of characters' thoughts and emotions, and deepened writers' understanding of character relationships. We conclude by discussing how multi-agent interactions can help distribute writers' attention and effort across the character cast.

**arXiv ID:** 2507.05820
</details>

<details>
<summary><strong>HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving</strong> - Donald Pfaffmann, Matthias Klusch, Marcel Steinmetz - [[pdf]](https://arxiv.org/pdf/2510.07210)</summary>

**Abstract:** We present a novel hybrid learning-assisted planning method, named HyPlan, for solving the collision-free navigation problem for self-driving cars in partially observable traffic environments. HyPlan combines methods for multi-agent behavior prediction, deep reinforcement learning with proximal policy optimization and approximated online POMDP planning with heuristic confidence-based vertical pruning to reduce its execution time without compromising safety of driving. Our experimental performance analysis on the CARLA-CTS2 benchmark of critical traffic scenarios with pedestrians revealed that HyPlan may navigate safer than selected relevant baselines and perform significantly faster than considered alternative online POMDP planners.

**arXiv ID:** 2510.07210
</details>

<details>
<summary><strong>Table-as-Search: Formulate Long-Horizon Agentic Information Seeking as Table Completion</strong> - Tian Lan, Felix Henry, Bin Zhu, Qianghuai Jia, Junyang Ren, Qihang Pu, Haijun Li, Longyue Wang, Zhao Xu, Weihua Luo - [[pdf]](https://arxiv.org/pdf/2602.06724)</summary>

**Abstract:** Current Information Seeking (InfoSeeking) agents struggle to maintain focus and coherence during long-horizon exploration, as tracking search states, including planning procedure and massive search results, within one plain-text context is inherently fragile. To address this, we introduce \textbf{Table-as-Search (TaS)}, a structured planning framework that reformulates the InfoSeeking task as a Table Completion task. TaS maps each query into a structured table schema maintained in an external database, where rows represent search candidates and columns denote constraints or required information. This table precisely manages the search states: filled cells strictly record the history and search results, while empty cells serve as an explicit search plan. Crucially, TaS unifies three distinct InfoSeeking tasks: Deep Search, Wide Search, and the challenging DeepWide Search. Extensive experiments demonstrate that TaS significantly outperforms numerous state-of-the-art baselines across three kinds of benchmarks, including multi-agent framework and commercial systems. Furthermore, our analysis validates the TaS's superior robustness in long-horizon InfoSeeking, alongside its efficiency, scalability and flexibility. Code and datasets are publicly released at this https URL.

**arXiv ID:** 2602.06724
</details>

<details>
<summary><strong>SWE-Dev: Evaluating and Training Autonomous Feature-Driven Software Development</strong> - Yaxin Du, Yuzhu Cai, Yifan Zhou, Cheng Wang, Yu Qian, Xianghe Pang, Qian Liu, Yue Hu, Siheng Chen - [[pdf]](https://arxiv.org/pdf/2505.16975)</summary>

**Abstract:** Large Language Models (LLMs) have shown strong capability in diverse software engineering tasks. However, feature-driven development, a highly prevalent real-world task that involves developing new functionalities for large, existing codebases, remains underexplored. We therefore introduce SWE-Dev, the first large-scale dataset (with 14,000 training and 500 test samples) designed to evaluate and train autonomous coding systems on real-world end-to-end feature-driven software development tasks. To ensure verifiable and diverse training, SWE-Dev uniquely provides all instances with a runnable environment and its developer-authored executable unit tests. This collection not only provides high-quality data for Supervised Fine-Tuning (SFT), but also enables Reinforcement Learning (RL) by delivering accurate reward signals from executable unit tests. We evaluated SWE-Dev across 17 base LLMs, 10 reasoning-focused LLMs, 10 multi-agent systems, and 8 tool-augmented LLM agents. Results show substantial headroom: the best single-turn model reaches only 22.51\% Pass@1 on the hard split, while OpenHands agents improve to 56.44\% but still leave many tasks unsolved. Code is available here this https URL.

**arXiv ID:** 2505.16975
</details>

<details>
<summary><strong>Flow Matching for Offline Reinforcement Learning with Discrete Actions</strong> - Fairoz Nower Khan, Nabuat Zaman Nahim, Ruiquan Huang, Haibo Yang, Peizhong Ju - [[pdf]](https://arxiv.org/pdf/2602.06138)</summary>

**Abstract:** Generative policies based on diffusion models and flow matching have shown strong promise for offline reinforcement learning (RL), but their applicability remains largely confined to continuous action spaces. To address a broader range of offline RL settings, we extend flow matching to a general framework that supports discrete action spaces with multiple objectives. Specifically, we replace continuous flows with continuous-time Markov chains, trained using a Q-weighted flow matching objective. We then extend our design to multi-agent settings, mitigating the exponential growth of joint action spaces via a factorized conditional path. We theoretically show that, under idealized conditions, optimizing this objective recovers the optimal policy. Extensive experiments further demonstrate that our method performs robustly in practical scenarios, including high-dimensional control, multi-modal decision-making, and dynamically changing preferences over multiple objectives. Our discrete framework can also be applied to continuous-control problems through action quantization, providing a flexible trade-off between representational complexity and performance.

**arXiv ID:** 2602.06138
</details>

<details>
<summary><strong>Evolutionary Generation of Multi-Agent Systems</strong> - Yuntong Hu, Matthew Trager, Yuting Zhang, Yi Zhang, Shuo Yang, Wei Xia, Stefano Soatto - [[pdf]](https://arxiv.org/pdf/2602.06511)</summary>

**Abstract:** Large language model (LLM)-based multi-agent systems (MAS) show strong promise for complex reasoning, planning, and tool-augmented tasks, but designing effective MAS architectures remains labor-intensive, brittle, and hard to generalize. Existing automatic MAS generation methods either rely on code generation, which often leads to executability and robustness failures, or impose rigid architectural templates that limit expressiveness and adaptability. We propose Evolutionary Generation of Multi-Agent Systems (EvoMAS), which formulates MAS generation as structured configuration generation. EvoMAS performs evolutionary generation in configuration space. Specifically, EvoMAS selects initial configurations from a pool, applies feedback-conditioned mutation and crossover guided by execution traces, and iteratively refines both the candidate pool and an experience memory. We evaluate EvoMAS on diverse benchmarks, including BBEH, SWE-Bench, and WorkBench, covering reasoning, software engineering, and tool-use tasks. EvoMAS consistently improves task performance over both human-designed MAS and prior automatic MAS generation methods, while producing generated systems with higher executability and runtime robustness. EvoMAS outperforms the agent evolution method EvoAgent by +10.5 points on BBEH reasoning and +7.1 points on WorkBench. With Claude-4.5-Sonnet, EvoMAS also reaches 79.1% on SWE-Bench-Verified, matching the top of the leaderboard.

**arXiv ID:** 2602.06511
</details>

<details>
<summary><strong>Strategizing at Speed: A Learned Model Predictive Game for Multi-Agent Drone Racing</strong> - Andrei-Carlo Papuc, Lasse Peters, Sihao Sun, Laura Ferranti, Javier Alonso-Mora - [[pdf]](https://arxiv.org/pdf/2602.06925)</summary>

**Abstract:** Autonomous drone racing pushes the boundaries of high-speed motion planning and multi-agent strategic decision-making. Success in this domain requires drones not only to navigate at their limits but also to anticipate and counteract competitors' actions. In this paper, we study a fundamental question that arises in this domain: how deeply should an agent strategize before taking an action? To this end, we compare two planning paradigms: the Model Predictive Game (MPG), which finds interaction-aware strategies at the expense of longer computation times, and contouring Model Predictive Control (MPC), which computes strategies rapidly but does not reason about interactions. We perform extensive experiments to study this trade-off, revealing that MPG outperforms MPC at moderate velocities but loses its advantage at higher speeds due to latency. To address this shortcoming, we propose a Learned Model Predictive Game (LMPG) approach that amortizes model predictive gameplay to reduce latency. In both simulation and hardware experiments, we benchmark our approach against MPG and MPC in head-to-head races, finding that LMPG outperforms both baselines.

**arXiv ID:** 2602.06925
</details>

</details>

<details open>
<summary><h2>Other Agent Research (10 papers)</h2></summary>

<details>
<summary><strong>Agentic Uncertainty Reveals Agentic Overconfidence</strong> - Jean Kaddour, Srijan Patel, Gbètondji Dovonon, Leo Richter, Pasquale Minervini, Matt J. Kusner - [[pdf]](https://arxiv.org/pdf/2602.06948)</summary>

**Abstract:** Can AI agents predict whether they will succeed at a task? We study agentic uncertainty by eliciting success probability estimates before, during, and after task execution. All results exhibit agentic overconfidence: some agents that succeed only 22% of the time predict 77% success. Counterintuitively, pre-execution assessment with strictly less information tends to yield better discrimination than standard post-execution review, though differences are not always significant. Adversarial prompting reframing assessment as bug-finding achieves the best calibration.

**arXiv ID:** 2602.06948
</details>

<details>
<summary><strong>Git for Sketches: An Intelligent Tracking System for Capturing Design Evolution</strong> - Sankar B, Amogh A S, Sandhya Baranwal, Dibakar Sen - [[pdf]](https://arxiv.org/pdf/2602.06047)</summary>

**Abstract:** During product conceptualization, capturing the non-linear history and cognitive intent is crucial. Traditional sketching tools often lose this context. We introduce DIMES (Design Idea Management and Evolution capture System), a web-based environment featuring sGIT (SketchGit), a custom visual version control architecture, and Generative AI. sGIT includes AEGIS, a module using hybrid Deep Learning and Machine Learning models to classify six stroke types. The system maps Git primitives to design actions, enabling implicit branching and multi-modal commits (stroke data + voice intent). In a comparative study, experts using DIMES demonstrated a 160% increase in breadth of concept exploration. Generative AI modules generated narrative summaries that enhanced knowledge transfer; novices achieved higher replication fidelity (Neural Transparency-based Cosine Similarity: 0.97 vs. 0.73) compared to manual summaries. AI-generated renderings also received higher user acceptance (Purchase Likelihood: 4.2 vs 3.1). This work demonstrates that intelligent version control bridges creative action and cognitive documentation, offering a new paradigm for design education.

**arXiv ID:** 2602.06047
</details>

<details>
<summary><strong>Implementing Grassroots Logic Programs with Multiagent Transition Systems and AI</strong> - Ehud Shapiro - [[pdf]](https://arxiv.org/pdf/2602.06934)</summary>

**Abstract:** Grassroots Logic Programs (GLP) is a concurrent logic programming language with variables partitioned into paired \emph{readers} and \emph{writers}, conjuring both linear logic and futures/promises: an assignment is produced at most once via the sole occurrence of a writer (promise) and consumed at most once via the sole occurrence of its paired reader (future), and may contain additional readers and/or writers, enabling the concise expression of rich multidirectional communication modalities.
GLP was designed as a language for grassroots platforms -- distributed systems with multiple instances that can operate independently of each other and of any global resource, and can coalesce into ever larger instances -- with its target architecture being smartphones communicating peer-to-peer. The operational semantics of Concurrent (single-agent) GLP and of multiagent GLP (maGLP) were defined via transition systems/multiagent transition systems, respectively.
Here, we describe the mathematics developed to facilitate the workstation- and smartphone-based implementations of GLP by AI in Dart. We developed dGLP -- implementation-ready deterministic operational semantics for single-agent GLP -- and proved it correct with respect to the Concurrent GLP operational semantics; dGLP was used by AI as a formal spec, from which it developed a workstation-based implementation of GLP. We developed madGLP -- an implementation-ready multiagent operational semantics for maGLP -- and proved it correct with respect to the maGLP operational semantics; madGLP is deterministic at the agent level (not at the system level due to communication asynchrony), and is being used by AI as a formal spec from which it develops a smartphone-based implementation of maGLP.

**arXiv ID:** 2602.06934
</details>

<details>
<summary><strong>Human-AI Co-Embodied Intelligence for Scientific Experimentation and Manufacturing</strong> - Xinyi Lin, Yuyang Zhang, Yuanhang Gan, Juntao Chen, Hao Shen, Yichun He, Lijun Li, Ze Yuan, Shuang Wang, Chaohao Wang, Rui Zhang, Na Li, Jia Liu - [[pdf]](https://arxiv.org/pdf/2511.02071)</summary>

**Abstract:** Scientific experimentation and manufacturing rely on prolonged protocol development and complex, multi-step implementation, which require continuous human expertise for precise execution and decision-making, limiting interpretability and scalability. Here, we introduce human-artificial intelligence (AI) co-embodied intelligence, a new form of physical AI that unites human researchers, agentic AI, and wearable hardware. In this paradigm, humans provide precise execution, while agentic AI contributes contextual reasoning, adaptive planning, and analysis. The wearable interface continuously captures experimentation and manufacturing, facilitating seamless communication between humans and AI. We instantiate this paradigm in a microfabrication cleanroom, leading to the agentic-physical experimentation (APEX) system which understands fabrication procedure with accuracy 51% higher than state-of-the-art multimodal large language models/vision language models (LLMs/VLMs), detects and corrects fabrication errors in real-time, and transfers procedural expertise to novice users. Critically, APEX system enables the co-development of fabrication protocols in cleanrooms, overcoming the incompatibility of elastomeric materials in standard microfabrication processes and enabling previously unattainable fabrication outcomes, as demonstrated by the wafer-scale realization of brain-level soft neural probe capable of single-unit-resolution neural recording. These results establish the human-AI co-embodied intelligence that extends agentic reasoning beyond computation into the physical domain, transforming scientific experimentation and manufacturing into autonomous, traceable, interpretable and scalable processes.

**arXiv ID:** 2511.02071
</details>

<details>
<summary><strong>AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction</strong> - Ruijie Shi, Houbin Zhang, Yuecheng Han, Yuheng Wang, Jingru Fan, Runde Yang, Yufan Dang, Huatao Li, Dewen Liu, Yuan Cheng, Chen Qian - [[pdf]](https://arxiv.org/pdf/2602.05353)</summary>

**Abstract:** Large Language Models have shown strong capabilities in complex problem solving, yet many agentic systems remain difficult to interpret and control due to opaque internal workflows. While some frameworks offer explicit architectures for collaboration, many deployed agentic systems operate as black boxes to users. We address this by introducing Agentic Workflow Reconstruction (AWR), a new task aiming to synthesize an explicit, interpretable stand-in workflow that approximates a black-box system using only input--output access. We propose AgentXRay, a search-based framework that formulates AWR as a combinatorial optimization problem over discrete agent roles and tool invocations in a chain-structured workflow space. Unlike model distillation, AgentXRay produces editable white-box workflows that match target outputs under an observable, output-based proxy metric, without accessing model parameters. To navigate the vast search space, AgentXRay employs Monte Carlo Tree Search enhanced by a scoring-based Red-Black Pruning mechanism, which dynamically integrates proxy quality with search depth. Experiments across diverse domains demonstrate that AgentXRay achieves higher proxy similarity and reduces token consumption compared to unpruned search, enabling deeper workflow exploration under fixed iteration budgets.

**arXiv ID:** 2602.05353
</details>

<details>
<summary><strong>DECEPTICON: How Dark Patterns Manipulate Web Agents</strong> - Phil Cuvin, Hao Zhu, Diyi Yang - [[pdf]](https://arxiv.org/pdf/2512.22894)</summary>

**Abstract:** Deceptive UI designs, widely instantiated across the web and commonly known as dark patterns, manipulate users into performing actions misaligned with their goals. In this paper, we show that dark patterns are highly effective in steering agent trajectories, posing a significant risk to agent robustness. To quantify this risk, we introduce DECEPTICON, an environment for testing individual dark patterns in isolation. DECEPTICON includes 700 web navigation tasks with dark patterns -- 600 generated tasks and 100 real-world tasks, designed to measure instruction-following success and dark pattern effectiveness. Across state-of-the-art agents, we find dark patterns successfully steer agent trajectories towards malicious outcomes in over 70% of tested generated and real-world tasks -- compared to a human average of 31%. Moreover, we find that dark pattern effectiveness correlates positively with model size and test-time reasoning, making larger, more capable models more susceptible. Leading countermeasures against adversarial attacks, including in-context prompting and guardrail models, fail to consistently reduce the success rate of dark pattern interventions. Our findings reveal dark patterns as a latent and unmitigated risk to web agents, highlighting the urgent need for robust defenses against manipulative designs.

**arXiv ID:** 2512.22894
</details>

<details>
<summary><strong>Agentic Workflow Using RBA$_θ$ for Event Prediction</strong> - Purbak Sengupta, Sambeet Mishra, Sonal Shreya - [[pdf]](https://arxiv.org/pdf/2602.06097)</summary>

**Abstract:** Wind power ramp events are difficult to forecast due to strong variability, multi-scale dynamics, and site-specific meteorological effects. This paper proposes an event-first, frequency-aware forecasting paradigm that directly predicts ramp events and reconstructs the power trajectory thereafter, rather than inferring events from dense forecasts. The framework is built on an enhanced Ramping Behaviour Analysis (RBA$_\theta$) method's event representation and progressively integrates statistical, machine-learning, and deep-learning models. Traditional forecasting models with post-hoc event extraction provides a strong interpretable baseline but exhibits limited generalisation across sites. Direct event prediction using Random Forests improves robustness over survival-based formulations, motivating fully event-aware modelling. To capture the multi-scale nature of wind ramps, we introduce an event-first deep architecture that integrates wavelet-based frequency decomposition, temporal excitation features, and adaptive feature selection. The resulting sequence models enable stable long-horizon event prediction, physically consistent trajectory reconstruction, and zero-shot transfer to previously unseen wind farms. Empirical analysis shows that ramp magnitude and duration are governed by distinct mid-frequency bands, allowing accurate signal reconstruction from sparse event forecasts. An agentic forecasting layer is proposed, in which specialised workflows are selected dynamically based on operational context. Together, the framework demonstrates that event-first, frequency-aware forecasting provides a transferable and operationally aligned alternative to trajectory-first wind-power prediction.

**arXiv ID:** 2602.06097
</details>

<details>
<summary><strong>Towards Adaptive Environment Generation for Training Embodied Agents</strong> - Teresa Yeo, Dulaj Weerakoon, Dulanga Weerakoon, Archan Misra - [[pdf]](https://arxiv.org/pdf/2602.06366)</summary>

**Abstract:** Embodied agents struggle to generalize to new environments, even when those environments share similar underlying structures to their training settings. Most current approaches to generating these training environments follow an open-loop paradigm, without considering the agent's current performance. While procedural generation methods can produce diverse scenes, diversity without feedback from the agent is inefficient. The generated environments may be trivially easy, providing limited learning signal. To address this, we present a proof-of-concept for closed-loop environment generation that adapts difficulty to the agent's current capabilities. Our system employs a controllable environment representation, extracts fine-grained performance feedback beyond binary success or failure, and implements a closed-loop adaptation mechanism that translates this feedback into environment modifications. This feedback-driven approach generates training environments that more challenging in the ways the agent needs to improve, enabling more efficient learning and better generalization to novel settings.

**arXiv ID:** 2602.06366
</details>

<details>
<summary><strong>Think Proprioceptively: Embodied Visual Reasoning for VLA Manipulation</strong> - Fangyuan Wang, Peng Zhou, Jiaming Qi, Shipeng Lyu, David Navarro-Alarcon, Guodong Guo - [[pdf]](https://arxiv.org/pdf/2602.06575)</summary>

**Abstract:** Vision-language-action (VLA) models typically inject proprioception only as a late conditioning signal, which prevents robot state from shaping instruction understanding and from influencing which visual tokens are attended throughout the policy. We introduce ThinkProprio, which converts proprioception into a sequence of text tokens in the VLM embedding space and fuses them with the task instruction at the input. This early fusion lets embodied state participate in subsequent visual reasoning and token selection, biasing computation toward action-critical evidence while suppressing redundant visual tokens. In a systematic ablation over proprioception encoding, state entry point, and action-head conditioning, we find that text tokenization is more effective than learned projectors, and that retaining roughly 15% of visual tokens can match the performance of using the full token set. Across CALVIN, LIBERO, and real-world manipulation, ThinkProprio matches or improves over strong baselines while reducing end-to-end inference latency over 50%.

**arXiv ID:** 2602.06575
</details>

<details>
<summary><strong>Sampling for Model Predictive Trajectory Planning in Autonomous Driving using Normalizing Flows</strong> - Georg Rabenstein, Lars Ullrich, Knut Graichen - [[pdf]](https://arxiv.org/pdf/2404.09657)</summary>

**Abstract:** Alongside optimization-based planners, sampling-based approaches are often used in trajectory planning for autonomous driving due to their simplicity. Model predictive path integral control is a framework that builds upon optimization principles while incorporating stochastic sampling of input trajectories. This paper investigates several sampling approaches for trajectory generation. In this context, normalizing flows originating from the field of variational inference are considered for the generation of sampling distributions, as they model transformations of simple to more complex distributions. Accordingly, learning-based normalizing flow models are trained for a more efficient exploration of the input domain for the task at hand. The developed algorithm and the proposed sampling distributions are evaluated in two simulation scenarios.

**arXiv ID:** 2404.09657
</details>

</details>

<details open>
<summary><h2>Planning and Reasoning (2 papers)</h2></summary>

<details>
<summary><strong>GATSim: Urban Mobility Simulation with Generative Agents</strong> - Qi Liu, Can Li, Wanjing Ma - [[pdf]](https://arxiv.org/pdf/2506.23306)</summary>

**Abstract:** Traditional agent-based urban mobility simulations often rely on rigid rulebased systems that struggle to capture the complexity, adaptability, and behavioral diversity inherent in human travel decision making. Inspired by recent advancements in large language models and AI agent technologies, we introduce GATSim, a novel framework that leverages these advancements to simulate urban mobility using generative agents with dedicated cognitive structures. GATSim agents are characterized by diverse socioeconomic profiles, individual lifestyles, and evolving preferences shaped through psychologically informed memory systems and lifelong learning. The main contributions of this work are: 1) a comprehensive architecture that integrates urban mobility foundation model with agent cognitive systems and transport simulation environment; 2) a hierarchical memory designed for efficient retrieval of contextually relevant information, incorporating spatial and temporal associations; 3) planning and reactive mechanisms for modeling adaptive mobility behaviors which integrate a multi-scale reflection process to transform specific travel experiences into generalized behavioral insights. Experiments indicate that generative agents perform competitively with human annotators in role-playing scenarios, while naturally producing realistic macroscopic traffic patterns. The code for the prototype implementation is publicly available at this https URL.

**arXiv ID:** 2506.23306
</details>

<details>
<summary><strong>DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search</strong> - Zhanli Li, Huiwen Tian, Lvzhou Luo, Yixuan Cao, Ping Luo - [[pdf]](https://arxiv.org/pdf/2602.05014)</summary>

**Abstract:** With the rapid progress of tool-using and agentic large language models (LLMs), Retrieval-Augmented Generation (RAG) is evolving from one-shot, passive retrieval into multi-turn, decision-driven evidence acquisition. Despite strong results in open-domain settings, existing agentic search frameworks commonly treat long documents as flat collections of chunks, underutilizing document-native priors such as hierarchical organization and sequential discourse structure. We introduce DeepRead, a structure-aware, multi-turn document reasoning agent that explicitly operationalizes these priors for long-document question answering. DeepRead leverages LLM-based OCR model to convert PDFs into structured Markdown that preserves headings and paragraph boundaries. It then indexes documents at the paragraph level and assigns each paragraph a coordinate-style metadata key encoding its section identity and in-section order. Building on this representation, DeepRead equips the LLM with two complementary tools: a Retrieve tool that localizes relevant paragraphs while exposing their structural coordinates (with lightweight scanning context), and a ReadSection tool that enables contiguous, order-preserving reading within a specified section and paragraph range. Our experiments demonstrate that DeepRead achieves significant improvements over Search-o1-style agentic search in document question answering. The synergistic effect between retrieval and reading tools is also validated. Our fine-grained behavioral analysis reveals a reading and reasoning paradigm resembling human-like ``locate then read'' behavior.

**arXiv ID:** 2602.05014
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (31 papers)</h2></summary>

<details>
<summary><strong>Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning (Extended Version)</strong> - Pierriccardo Olivieri, Fausto Lasca, Alessandro Gianola, Matteo Papini - [[pdf]](https://arxiv.org/pdf/2602.06227)</summary>

**Abstract:** In this work, we propose a novel framework for the logical specification of non-Markovian rewards in Markov Decision Processes (MDPs) with large state spaces. Our approach leverages Linear Temporal Logic Modulo Theories over finite traces (LTLfMT), a more expressive extension of classical temporal logic in which predicates are first-order formulas of arbitrary first-order theories rather than simple Boolean variables. This enhanced expressiveness enables the specification of complex tasks over unstructured and heterogeneous data domains, promoting a unified and reusable framework that eliminates the need for manual predicate encoding. However, the increased expressive power of LTLfMT introduces additional theoretical and computational challenges compared to standard LTLf specifications. We address these challenges from a theoretical standpoint, identifying a fragment of LTLfMT that is tractable but sufficiently expressive for reward specification in an infinite-state-space context. From a practical perspective, we introduce a method based on reward machines and Hindsight Experience Replay (HER) to translate first-order logic specifications and address reward sparsity. We evaluate this approach to a continuous-control setting using Non-Linear Arithmetic Theory, showing that it enables natural specification of complex tasks. Experimental results show how a tailored implementation of HER is fundamental in solving tasks with complex goals.

**arXiv ID:** 2602.06227
</details>

<details>
<summary><strong>Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch Reinforcement Learning</strong> - Zhuoming Chen, Hongyi Liu, Yang Zhou, Haizhong Zheng, Beidi Chen - [[pdf]](https://arxiv.org/pdf/2602.06107)</summary>

**Abstract:** Reinforcement learning (RL) for large language models (LLMs) remains expensive, particularly because the rollout is expensive. Decoupling rollout generation from policy optimization (e.g., leveraging a more efficient model to rollout) could enable substantial efficiency gains, yet doing so introduces a severe distribution mismatch that destabilizes learning. We propose Jackpot, a framework that leverages Optimal Budget Rejection Sampling (OBRS) to directly reduce the discrepancy between the rollout model and the evolving policy. Jackpot integrates a principled OBRS procedure, a unified training objective that jointly updates the policy and rollout models, and an efficient system implementation enabled by top-$k$ probability estimation and batch-level bias correction. Our theoretical analysis shows that OBRS consistently moves the rollout distribution closer to the target distribution under a controllable acceptance budget. Empirically, \sys substantially improves training stability compared to importance-sampling baselines, achieving performance comparable to on-policy RL when training Qwen3-8B-Base for up to 300 update steps of batchsize 64. Taken together, our results show that OBRS-based alignment brings us a step closer to practical and effective decoupling of rollout generation from policy optimization for RL for LLMs.

**arXiv ID:** 2602.06107
</details>

<details>
<summary><strong>AgentCPM-Explore: Realizing Long-Horizon Deep Exploration for Edge-Scale Agents</strong> - Haotian Chen, Xin Cong, Shengda Fan, Yuyang Fu, Ziqin Gong, Yaxi Lu, Yishan Li, Boye Niu, Chengjun Pan, Zijun Song, Huadong Wang, Yesai Wu, Yueying Wu, Zihao Xie, Yukun Yan, Zhong Zhang, Yankai Lin, Zhiyuan Liu, Maosong Sun - [[pdf]](https://arxiv.org/pdf/2602.06485)</summary>

**Abstract:** While Large Language Model (LLM)-based agents have shown remarkable potential for solving complex tasks, existing systems remain heavily reliant on large-scale models, leaving the capabilities of edge-scale models largely underexplored. In this paper, we present the first systematic study on training agentic models at the 4B-parameter scale. We identify three primary bottlenecks hindering the performance of edge-scale models: catastrophic forgetting during Supervised Fine-Tuning (SFT), sensitivity to reward signal noise during Reinforcement Learning (RL), and reasoning degradation caused by redundant information in long-context scenarios. To address the issues, we propose AgentCPM-Explore, a compact 4B agent model with high knowledge density and strong exploration capability. We introduce a holistic training framework featuring parameter-space model fusion, reward signal denoising, and contextual information refinement. Through deep exploration, AgentCPM-Explore achieves state-of-the-art (SOTA) performance among 4B-class models, matches or surpasses 8B-class SOTA models on four benchmarks, and even outperforms larger-scale models such as Claude-4.5-Sonnet or DeepSeek-v3.2 in five benchmarks. Notably, AgentCPM-Explore achieves 97.09% accuracy on GAIA text-based tasks under pass@64. These results provide compelling evidence that the bottleneck for edge-scale models is not their inherent capability ceiling, but rather their inference stability. Based on our well-established training framework, AgentCPM-Explore effectively unlocks the significant, yet previously underestimated, potential of edge-scale models.

**arXiv ID:** 2602.06485
</details>

<details>
<summary><strong>AgentCPM-Report: Interleaving Drafting and Deepening for Open-Ended Deep Research</strong> - Yishan Li, Wentong Chen, Yukun Yan, Mingwei Li, Sen Mei, Xiaorong Wang, Kunpeng Liu, Xin Cong, Shuo Wang, Zhong Zhang, Yaxi Lu, Zhenghao Liu, Yankai Lin, Zhiyuan Liu, Maosong Sun - [[pdf]](https://arxiv.org/pdf/2602.06540)</summary>

**Abstract:** Generating deep research reports requires large-scale information acquisition and the synthesis of insight-driven analysis, posing a significant challenge for current language models. Most existing approaches follow a plan-then-write paradigm, whose performance heavily depends on the quality of the initial outline. However, constructing a comprehensive outline itself demands strong reasoning ability, causing current deep research systems to rely almost exclusively on closed-source or online large models. This reliance raises practical barriers to deployment and introduces safety and privacy concerns for user-authored data. In this work, we present AgentCPM-Report, a lightweight yet high-performing local solution composed of a framework that mirrors the human writing process and an 8B-parameter deep research agent. Our framework uses a Writing As Reasoning Policy (WARP), which enables models to dynamically revise outlines during report generation. Under this policy, the agent alternates between Evidence-Based Drafting and Reasoning-Driven Deepening, jointly supporting information acquisition, knowledge refinement, and iterative outline evolution. To effectively equip small models with this capability, we introduce a Multi-Stage Agentic Training strategy, consisting of cold-start, atomic skill RL, and holistic pipeline RL. Experiments on DeepResearch Bench, DeepConsult, and DeepResearch Gym demonstrate that AgentCPM-Report outperforms leading closed-source systems, with substantial gains in Insight.

**arXiv ID:** 2602.06540
</details>

<details>
<summary><strong>Progress Constraints for Reinforcement Learning in Behavior Trees</strong> - Finn Rietz, Mart Kartašev, Johannes A. Stork, Petter Ögren - [[pdf]](https://arxiv.org/pdf/2602.06525)</summary>

**Abstract:** Behavior Trees (BTs) provide a structured and reactive framework for decision-making, commonly used to switch between sub-controllers based on environmental conditions. Reinforcement Learning (RL), on the other hand, can learn near-optimal controllers but sometimes struggles with sparse rewards, safe exploration, and long-horizon credit assignment. Combining BTs with RL has the potential for mutual benefit: a BT design encodes structured domain knowledge that can simplify RL training, while RL enables automatic learning of the controllers within BTs. However, naive integration of BTs and RL can lead to some controllers counteracting other controllers, possibly undoing previously achieved subgoals, thereby degrading the overall performance. To address this, we propose progress constraints, a novel mechanism where feasibility estimators constrain the allowed action set based on theoretical BT convergence results. Empirical evaluations in a 2D proof-of-concept and a high-fidelity warehouse environment demonstrate improved performance, sample efficiency, and constraint satisfaction, compared to prior methods of BT-RL integration.

**arXiv ID:** 2602.06525
</details>

<details>
<summary><strong>Semantically Labelled Automata for Multi-Task Reinforcement Learning with LTL Instructions</strong> - Alessandro Abate, Giuseppe De Giacomo, Mathias Jackermeier, Jan Kretínský, Maximilian Prokop, Christoph Weinhuber - [[pdf]](https://arxiv.org/pdf/2602.06746)</summary>

**Abstract:** We study multi-task reinforcement learning (RL), a setting in which an agent learns a single, universal policy capable of generalising to arbitrary, possibly unseen tasks. We consider tasks specified as linear temporal logic (LTL) formulae, which are commonly used in formal methods to specify properties of systems, and have recently been successfully adopted in RL. In this setting, we present a novel task embedding technique leveraging a new generation of semantic LTL-to-automata translations, originally developed for temporal synthesis. The resulting semantically labelled automata contain rich, structured information in each state that allow us to (i) compute the automaton efficiently on-the-fly, (ii) extract expressive task embeddings used to condition the policy, and (iii) naturally support full LTL. Experimental results in a variety of domains demonstrate that our approach achieves state-of-the-art performance and is able to scale to complex specifications where existing methods fail.

**arXiv ID:** 2602.06746
</details>

<details>
<summary><strong>iScheduler: Reinforcement Learning-Driven Continual Optimization for Large-Scale Resource Investment Problems</strong> - Yi-Xiang Hu, Yuke Wang, Feng Wu, Zirui Huang, Shuli Zeng, Xiang-Yang Li - [[pdf]](https://arxiv.org/pdf/2602.06064)</summary>

**Abstract:** Scheduling precedence-constrained tasks under shared renewable resources is central to modern computing platforms. The Resource Investment Problem (RIP) models this setting by minimizing the cost of provisioned renewable resources under precedence and timing constraints. Exact mixed-integer programming and constraint programming become impractically slow on large instances, and dynamic updates require schedule revisions under tight latency budgets. We present iScheduler, a reinforcement-learning-driven iterative scheduling framework that formulates RIP solving as a Markov decision process over decomposed subproblems and constructs schedules through sequential process selection. The framework accelerates optimization and supports reconfiguration by reusing unchanged process schedules and rescheduling only affected processes. We also release L-RIPLIB, an industrial-scale benchmark derived from cloud-platform workloads with 1,000 instances of 2,500-10,000 tasks. Experiments show that iScheduler attains competitive resource costs while reducing time to feasibility by up to 43$\times$ against strong commercial baselines.

**arXiv ID:** 2602.06064
</details>

<details>
<summary><strong>Transformer-Based Reinforcement Learning for Autonomous Orbital Collision Avoidance in Partially Observable Environments</strong> - Thomas Georges, Adam Abdin - [[pdf]](https://arxiv.org/pdf/2602.06088)</summary>

**Abstract:** We introduce a Transformer-based Reinforcement Learning framework for autonomous orbital collision avoidance that explicitly models the effects of partial observability and imperfect monitoring in space operations. The framework combines a configurable encounter simulator, a distance-dependent observation model, and a sequential state estimator to represent uncertainty in relative motion. A central contribution of this work is the use of transformer-based Partially Observable Markov Decision Process (POMDP) architecture, which leverage long-range temporal attention to interpret noisy and intermittent observations more effectively than traditional architectures. This integration provides a foundation for training collision avoidance agents that can operate more reliably under imperfect monitoring environments.

**arXiv ID:** 2602.06088
</details>

<details>
<summary><strong>Zero-Trust Runtime Verification for Agentic Payment Protocols: Mitigating Replay and Context-Binding Failures in AP2</strong> - Qianlong Lan, Anuj Kaul, Shaun Jones, Stephanie Westrum - [[pdf]](https://arxiv.org/pdf/2602.06345)</summary>

**Abstract:** The deployment of autonomous AI agents capable of executing commercial transactions has motivated the adoption of mandate-based payment authorization protocols, including the Universal Commerce Protocol (UCP) and the Agent Payments Protocol (AP2). These protocols replace interactive, session-based authorization with cryptographically issued mandates, enabling asynchronous and autonomous execution. While AP2 provides specification-level guarantees through signature verification, explicit binding, and expiration semantics, real-world agentic execution introduces runtime behaviors such as retries, concurrency, and orchestration that challenge implicit assumptions about mandate usage.
In this work, we present a security analysis of the AP2 mandate lifecycle and identify enforcement gaps that arise during runtime in agent-based payment systems. We propose a zero-trust runtime verification framework that enforces explicit context binding and consume-once mandate semantics using dynamically generated, time-bound nonces, ensuring that authorization decisions are evaluated at execution time rather than assumed from static issuance properties.
Through simulation-based evaluation under high concurrency, we show that context-aware binding and consume-once enforcement address distinct and complementary attack classes, and that both are required to prevent replay and context-redirect attacks. The proposed framework mitigates all evaluated attacks while maintaining stable verification latency of approximately 3.8~ms at throughput levels up to 10{,}000 transactions per second. We further demonstrate that the required runtime state is bounded by peak concurrency rather than cumulative transaction history, indicating that robust runtime security for agentic payment execution can be achieved with minimal and predictable overhead.

**arXiv ID:** 2602.06345
</details>

<details>
<summary><strong>TrailBlazer: History-Guided Reinforcement Learning for Black-Box LLM Jailbreaking</strong> - Sung-Hoon Yoon, Ruizhi Qian, Minda Zhao, Weiyue Li, Mengyu Wang - [[pdf]](https://arxiv.org/pdf/2602.06440)</summary>

**Abstract:** Large Language Models (LLMs) have become integral to many domains, making their safety a critical priority. Prior jailbreaking research has explored diverse approaches, including prompt optimization, automated red teaming, obfuscation, and reinforcement learning (RL) based methods. However, most existing techniques fail to effectively leverage vulnerabilities revealed in earlier interaction turns, resulting in inefficient and unstable attacks. Since jailbreaking involves sequential interactions in which each response influences future actions, reinforcement learning provides a natural framework for this problem. Motivated by this, we propose a history-aware RL-based jailbreak framework that analyzes and reweights vulnerability signals from prior steps to guide future decisions. We show that incorporating historical information alone improves jailbreak success rates. Building on this insight, we introduce an attention-based reweighting mechanism that highlights critical vulnerabilities within the interaction history, enabling more efficient exploration with fewer queries. Extensive experiments on AdvBench and HarmBench demonstrate that our method achieves state-of-the-art jailbreak performance while significantly improving query efficiency. These results underscore the importance of historical vulnerability signals in reinforcement learning-driven jailbreak strategies and offer a principled pathway for advancing adversarial research on LLM safeguards.

**arXiv ID:** 2602.06440
</details>

<details>
<summary><strong>Malicious Agent Skills in the Wild: A Large-Scale Security Empirical Study</strong> - Yi Liu, Zhihao Chen, Yanjun Zhang, Gelei Deng, Yuekang Li, Jianting Ning, Leo Yu Zhang - [[pdf]](https://arxiv.org/pdf/2602.06547)</summary>

**Abstract:** Third-party agent skills extend LLM-based agents with instruction files and executable code that run on users' machines. Skills execute with user privileges and are distributed through community registries with minimal vetting, but no ground-truth dataset exists to characterize the resulting threats. We construct the first labeled dataset of malicious agent skills by behaviorally verifying 98,380 skills from two community registries, confirming 157 malicious skills with 632 vulnerabilities. These attacks are not incidental. Malicious skills average 4.03 vulnerabilities across a median of three kill chain phases, and the ecosystem has split into two archetypes: Data Thieves that exfiltrate credentials through supply chain techniques, and Agent Hijackers that subvert agent decision-making through instruction manipulation. A single actor accounts for 54.1\% of confirmed cases through templated brand impersonation. Shadow features, capabilities absent from public documentation, appear in 0\% of basic attacks but 100\% of advanced ones; several skills go further by exploiting the AI platform's own hook system and permission flags. Responsible disclosure led to 93.6\% removal within 30 days. We release the dataset and analysis pipeline to support future work on agent skill security.

**arXiv ID:** 2602.06547
</details>

<details>
<summary><strong>AgentStepper: Interactive Debugging of Software Development Agents</strong> - Robert Hutter, Michael Pradel - [[pdf]](https://arxiv.org/pdf/2602.06593)</summary>

**Abstract:** Software development agents powered by large language models (LLMs) have shown great promise in automating tasks like environment setup, issue solving, and program repair. Unfortunately, understanding and debugging such agents remain challenging due to their complex and dynamic nature. Developers must reason about trajectories of LLM queries, tool calls, and code modifications, but current techniques reveal little of this intermediate process in a comprehensible format. The key insight of this paper is that debugging software development agents shares many similarities with conventional debugging of software programs, yet requires a higher level of abstraction that raises the level from low-level implementation details to high-level agent actions. Drawing on this insight, we introduce AgentStepper, the first interactive debugger for LLM-based software engineering agents. AgentStepper enables developers to inspect, control, and interactively manipulate agent trajectories. AgentStepper represents trajectories as structured conversations among an LLM, the agent program, and tools. It supports breakpoints, stepwise execution, and live editing of prompts and tool invocations, while capturing and displaying intermediate repository-level code changes. Our evaluation applies AgentStepper to three state-of-the-art software development agents, ExecutionAgent, SWE-Agent, and RepairAgent, showing that integrating the approach into existing agents requires minor code changes (39-42 edited lines). Moreover, we report on a user study with twelve participants, indicating that AgentStepper improves the ability of participants to interpret trajectories (64% vs. 67% mean performance) and identify bugs in the agent's implementation (17% vs. 60% success rate), while reducing perceived workload (e.g., frustration reduced from 5.4/7.0 to 2.4/7.0) compared to conventional tools.

**arXiv ID:** 2602.06593
</details>

<details>
<summary><strong>InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning</strong> - Yuchen Yan, Liang Jiang, Jin Jiang, Shuaicheng Li, Zujie Wen, Zhiqiang Zhang, Jun Zhou, Jian Shao, Yueting Zhuang, Yongliang Shen - [[pdf]](https://arxiv.org/pdf/2602.06960)</summary>

**Abstract:** Large reasoning models achieve strong performance by scaling inference-time chain-of-thought, but this paradigm suffers from quadratic cost, context length limits, and degraded reasoning due to lost-in-the-middle effects. Iterative reasoning mitigates these issues by periodically summarizing intermediate thoughts, yet existing methods rely on supervised learning or fixed heuristics and fail to optimize when to summarize, what to preserve, and how to resume reasoning. We propose InftyThink+, an end-to-end reinforcement learning framework that optimizes the entire iterative reasoning trajectory, building on model-controlled iteration boundaries and explicit summarization. InftyThink+ adopts a two-stage training scheme with supervised cold-start followed by trajectory-level reinforcement learning, enabling the model to learn strategic summarization and continuation decisions. Experiments on DeepSeek-R1-Distill-Qwen-1.5B show that InftyThink+ improves accuracy by 21% on AIME24 and outperforms conventional long chain-of-thought reinforcement learning by a clear margin, while also generalizing better to out-of-distribution benchmarks. Moreover, InftyThink+ significantly reduces inference latency and accelerates reinforcement learning training, demonstrating improved reasoning efficiency alongside stronger performance.

**arXiv ID:** 2602.06960
</details>

<details>
<summary><strong>Hi-Agent: Hierarchical Vision-Language Agents for Mobile Device Control</strong> - Zhe Wu, Hongjin Lu, Junliang Xing, Changhao Zhang, Yuxuan Li, Yin Zhu, Yuhao Yang, Yuheng Jing, Kai Li, Kun Shao, Jianye Hao, Jun Wang, Yuanchun Shi - [[pdf]](https://arxiv.org/pdf/2510.14388)</summary>

**Abstract:** Building agents that autonomously operate mobile devices has attracted increasing attention. While Vision-Language Models (VLMs) show promise, most existing approaches rely on direct state-to-action mappings, which lack structured reasoning and planning, and thus generalize poorly to novel tasks or unseen UI layouts. We introduce Hi-Agent, a trainable hierarchical vision-language agent for mobile control, featuring a high-level reasoning model and a low-level action model that are jointly optimized. For efficient training, we reformulate multi-step decision-making as a sequence of single-step subgoals and propose a foresight advantage function, which leverages execution feedback from the low-level model to guide high-level optimization. This design alleviates the path explosion issue encountered by Group Relative Policy Optimization (GRPO) in long-horizon tasks and enables stable, critic-free joint training. Hi-Agent achieves a new State-Of-The-Art (SOTA) 87.9% task success rate on the Android-in-the-Wild (AitW) benchmark, significantly outperforming prior methods across three paradigms: prompt-based (AppAgent: 17.7%), supervised (Filtered BC: 54.5%), and reinforcement learning-based (DigiRL: 71.9%). It also demonstrates competitive zero-shot generalization on the ScreenSpot-v2 benchmark. On the more challenging AndroidWorld benchmark, Hi-Agent also scales effectively with larger backbones, showing strong adaptability in high-complexity mobile control scenarios.

**arXiv ID:** 2510.14388
</details>

<details>
<summary><strong>Forecast Aware Deep Reinforcement Learning for Efficient Electricity Load Scheduling in Dairy Farms</strong> - Nawazish Ali, Rachael Shaw, Karl Mason - [[pdf]](https://arxiv.org/pdf/2601.08052)</summary>

**Abstract:** Dairy farming is an energy intensive sector that relies heavily on grid electricity. With increasing renewable energy integration, sustainable energy management has become essential for reducing grid dependence and supporting the United Nations Sustainable Development Goal 7 on affordable and clean energy. However, the intermittent nature of renewables poses challenges in balancing supply and demand in real time. Intelligent load scheduling is therefore crucial to minimize operational costs while maintaining reliability. Reinforcement Learning has shown promise in improving energy efficiency and reducing costs. However, most RL-based scheduling methods assume complete knowledge of future prices or generation, which is unrealistic in dynamic environments. Moreover, standard PPO variants rely on fixed clipping or KL divergence thresholds, often leading to unstable training under variable tariffs. To address these challenges, this study proposes a Deep Reinforcement Learning framework for efficient load scheduling in dairy farms, focusing on battery storage and water heating under realistic operational constraints. The proposed Forecast Aware PPO incorporates short term forecasts of demand and renewable generation using hour of day and month based residual calibration, while the PID KL PPO variant employs a proportional integral derivative controller to regulate KL divergence for stable policy updates adaptively. Trained on real world dairy farm data, the method achieves up to 1% lower electricity cost than PPO, 4.8% than DQN, and 1.5% than SAC. For battery scheduling, PPO reduces grid imports by 13.1%, demonstrating scalability and effectiveness for sustainable energy management in modern dairy farming.

**arXiv ID:** 2601.08052
</details>

<details>
<summary><strong>dUltra: Ultra-Fast Diffusion Language Models via Reinforcement Learning</strong> - Shirui Chen, Jiantao Jiao, Lillian J. Ratliff, Banghua Zhu - [[pdf]](https://arxiv.org/pdf/2512.21446)</summary>

**Abstract:** Masked diffusion language models (MDLMs) offer the potential for parallel token generation, but most open-source MDLMs decode fewer than 5 tokens per model forward pass even with sophisticated sampling strategies, limiting their parallel generation potential. Existing acceleration methods either rely on fixed confidence-based heuristics or use distillation-based approaches that finetune MDLMs on trajectories generated by a base model, which can become off-policy during finetuning and restrict performance to the quality of the base model's samples. We propose \texttt{dUltra}, an on-policy reinforcement learning framework based on Group Relative Policy Optimization (GRPO) that learns unmasking strategies for efficient parallel decoding. dUltra introduces an unmasking planner head that predicts per-token unmasking likelihoods under independent Bernoulli distributions. We jointly optimize the base diffusion LLM and the unmasking order planner using reward signals combining verifiable reward, distillation reward, and the number of unmasking steps. Across mathematical reasoning and code generation tasks, dUltra achieves superior accuracy-efficiency trade-offs compared to state-of-the-art heuristic (Fast-dLLM) and distillation baselines (d3LLM, dParallel), demonstrating that learned unmasking trajectories through on-policy RL enable better exploitation of parallel generation in MDLMs. Code and checkpoints are released at this https URL.

**arXiv ID:** 2512.21446
</details>

<details>
<summary><strong>Trust Region Masking for Long-Horizon LLM Reinforcement Learning</strong> - Yingru Li, Jiacai Liu, Jiawei Xu, Yuxuan Tong, Ziniu Li, Qian Liu, Baoxiang Wang - [[pdf]](https://arxiv.org/pdf/2512.23075)</summary>

**Abstract:** Policy gradient methods for Large Language Models optimize a policy $\pi_\theta$ via a surrogate objective computed from samples of a rollout policy $\pi_{\text{roll}}$. However, modern LLM-RL pipelines suffer from unavoidable implementation divergences, such as backend discrepancies, Mixture-of-Experts routing discontinuities, and distributed training staleness. These factors cause an off-policy mismatch ($\pi_{\text{roll}} \neq \pi_\theta$), leading to approximation errors between the surrogate and the true objective. We demonstrate that classical trust region bounds on this error scale as $O(T^2)$ with sequence length $T$, rendering them vacuous for long-horizon tasks. To address this, we derive two new bounds: a Pinsker-Marginal bound scaling as $O(T^{3/2})$ and a Mixed bound scaling as $O(T)$. We further derive an Adaptive bound that strictly generalizes the Pinsker-Marginal bound by combining an importance-ratio decomposition of the error with an adaptive per-position application of Pinsker's inequality on the future trajectory divergence; the minimum over all three bounds is tighter than any individual bound. Crucially, all bounds depend on $D_{\mathrm{KL}}^{\mathrm{tok,max}}$, the maximum token-level KL divergence across the sequence. As a \emph{sequence-level} term, the divergence cannot be controlled by previous token-independent methods like PPO clipping. We propose Trust Region Masking (TRM), which masks entire sequences that violate the trust region. TRM enables the first non-vacuous monotonic improvement guarantees and demonstrates empirical training stability for long-horizon LLM-RL.

**arXiv ID:** 2512.23075
</details>

<details>
<summary><strong>LLM-Enhanced Reinforcement Learning for Long-Term User Satisfaction in Interactive Recommendation</strong> - Chongjun Xia, Yanchun Peng, Xianzhi Wang - [[pdf]](https://arxiv.org/pdf/2601.19585)</summary>

**Abstract:** Interactive recommender systems can dynamically adapt to user feedback, but often suffer from content homogeneity and filter bubble effects due to overfitting short-term user preferences. While recent efforts aim to improve content diversity, they predominantly operate in static or one-shot settings, neglecting the long-term evolution of user interests. Reinforcement learning provides a principled framework for optimizing long-term user satisfaction by modeling sequential decision-making processes. However, its application in recommendation is hindered by sparse, long-tailed user-item interactions and limited semantic planning capabilities. In this work, we propose LLM-Enhanced Reinforcement Learning (LERL), a novel hierarchical recommendation framework that integrates the semantic planning power of LLM with the fine-grained adaptability of RL. LERL consists of a high-level LLM-based planner that selects semantically diverse content categories, and a low-level RL policy that recommends personalized items within the selected semantic space. This hierarchical design narrows the action space, enhances planning efficiency, and mitigates overexposure to redundant content. Extensive experiments on real-world datasets demonstrate that LERL significantly improves long-term user satisfaction when compared with state-of-the-art baselines. The implementation of LERL is available at this https URL.

**arXiv ID:** 2601.19585
</details>

<details>
<summary><strong>Evaluating an evidence-guided reinforcement learning framework in aligning light-parameter large language models with decision-making cognition in psychiatric clinical reasoning</strong> - Xinxin Lin, Guangxin Dai, Yi Zhong, Xiang Li, Xue Xiao, Yixin Zhang, Zhengdong Wu, Yongbo Zheng, Runchuan Zhu, Ming Zhao, Huizi Yu, Shuo Wu, Jun Zhao, Lingming Hu, Yumei Wang, Ping Yin, Joey W.Y. Chan, Ngan Yin Chan, Sijing Chen, Yun Kwok Wing, Lin Lu, Xin Ma, Lizhou Fan - [[pdf]](https://arxiv.org/pdf/2602.06449)</summary>

**Abstract:** Large language models (LLMs) hold transformative potential for medical decision support yet their application in psychiatry remains constrained by hallucinations and superficial reasoning. This limitation is particularly acute in light-parameter LLMs which are essential for privacy-preserving and efficient clinical deployment. Existing training paradigms prioritize linguistic fluency over structured clinical logic and result in a fundamental misalignment with professional diagnostic cognition. Here we introduce ClinMPO, a reinforcement learning framework designed to align the internal reasoning of LLMs with professional psychiatric practice. The framework employs a specialized reward model trained independently on a dataset derived from 4,474 psychiatry journal articles and structured according to evidence-based medicine principles. We evaluated ClinMPO on a unseen subset of the benchmark designed to isolate reasoning capabilities from rote memorization. This test set comprises items where leading large-parameter LLMs consistently fail. We compared the ClinMPO-aligned light LLM performance against a cohort of 300 medical students. The ClinMPO-tuned Qwen3-8B model achieved a diagnostic accuracy of 31.4% and surpassed the human benchmark of 30.8% on these complex cases. These results demonstrate that medical evidence-guided optimization enables light-parameter LLMs to master complex reasoning tasks. Our findings suggest that explicit cognitive alignment offers a scalable pathway to reliable and safe psychiatric decision support.

**arXiv ID:** 2602.06449
</details>

<details>
<summary><strong>Back to Basics: Revisiting Exploration in Reinforcement Learning for LLM Reasoning via Generative Probabilities</strong> - Pengyi Li, Elizaveta Goncharova, Andrey Kuznetsov, Ivan Oseledets - [[pdf]](https://arxiv.org/pdf/2602.05281)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an indispensable paradigm for enhancing reasoning in Large Language Models (LLMs). However, standard policy optimization methods, such as Group Relative Policy Optimization (GRPO), often converge to low-entropy policies, leading to severe mode collapse and limited output diversity. We analyze this issue from the perspective of sampling probability dynamics, identifying that the standard objective disproportionately reinforces the highest-likelihood paths, thereby suppressing valid alternative reasoning chains. To address this, we propose a novel Advantage Re-weighting Mechanism (ARM) designed to equilibrate the confidence levels across all correct responses. By incorporating Prompt Perplexity and Answer Confidence into the advantage estimation, our method dynamically reshapes the reward signal to attenuate the gradient updates of over-confident reasoning paths, while redistributing probability mass toward under-explored correct solutions. Empirical results demonstrate that our approach significantly enhances generative diversity and response entropy while maintaining competitive accuracy, effectively achieving a superior trade-off between exploration and exploitation in reasoning tasks. Empirical results on Qwen2.5 and DeepSeek models across mathematical and coding benchmarks show that ProGRPO significantly mitigates entropy collapse. Specifically, on Qwen2.5-7B, our method outperforms GRPO by 5.7% in Pass@1 and, notably, by 13.9% in Pass@32, highlighting its superior capability in generating diverse correct reasoning paths.

**arXiv ID:** 2602.05281
</details>

<details>
<summary><strong>Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations</strong> - Wei Liu, Jiawei Xu, Yingru Li, Longtao Zheng, Tianjian Li, Qian Liu, Junxian He - [[pdf]](https://arxiv.org/pdf/2602.05885)</summary>

**Abstract:** High-quality kernel is critical for scalable AI systems, and enabling LLMs to generate such code would advance AI development. However, training LLMs for this task requires sufficient data, a robust environment, and the process is often vulnerable to reward hacking and lazy optimization. In these cases, models may hack training rewards and prioritize trivial correctness over meaningful speedup. In this paper, we systematically study reinforcement learning (RL) for kernel generation. We first design KernelGYM, a robust distributed GPU environment that supports reward hacking check, data collection from multi-turn interactions and long-term RL training. Building on KernelGYM, we investigate effective multi-turn RL methods and identify a biased policy gradient issue caused by self-inclusion in GRPO. To solve this, we propose Turn-level Reinforce-Leave-One-Out (TRLOO) to provide unbiased advantage estimation for multi-turn RL. To alleviate lazy optimization, we incorporate mismatch correction for training stability and introduce Profiling-based Rewards (PR) and Profiling-based Rejection Sampling (PRS) to overcome the issue. The trained model, Dr Kernel-14B, reaches performance competitive with Claude-4.5-Sonnet in Kernelbench. Finally, we study sequential test-time scaling for Dr Kernel-14B. On the KernelBench Level-2 subset, 31.6% of the generated kernels achieve at least a 1.2x speedup over the Torch reference, surpassing Claude-4.5-Sonnet (26.7%) and GPT-5 (28.6%). When selecting the best candidate across all turns, this 1.2x speedup rate further increases to 47.8%. All resources, including environment, training code, models, and dataset, are included in this https URL.

**arXiv ID:** 2602.05885
</details>

<details>
<summary><strong>Online Adaptive Reinforcement Learning with Echo State Networks for Non-Stationary Dynamics</strong> - Aoi Yoshimura, Gouhei Tanaka - [[pdf]](https://arxiv.org/pdf/2602.06326)</summary>

**Abstract:** Reinforcement learning (RL) policies trained in simulation often suffer from severe performance degradation when deployed in real-world environments due to non-stationary dynamics. While Domain Randomization (DR) and meta-RL have been proposed to address this issue, they typically rely on extensive pretraining, privileged information, or high computational cost, limiting their applicability to real-time and edge systems. In this paper, we propose a lightweight online adaptation framework for RL based on Reservoir Computing. Specifically, we integrate an Echo State Networks (ESNs) as an adaptation module that encodes recent observation histories into a latent context representation, and update its readout weights online using Recursive Least Squares (RLS). This design enables rapid adaptation without backpropagation, pretraining, or access to privileged information. We evaluate the proposed method on CartPole and HalfCheetah tasks with severe and abrupt environment changes, including periodic external disturbances and extreme friction variations. Experimental results demonstrate that the proposed approach significantly outperforms DR and representative adaptive baselines under out-of-distribution dynamics, achieving stable adaptation within a few control steps. Notably, the method successfully handles intra-episode environment changes without resetting the policy. Due to its computational efficiency and stability, the proposed framework provides a practical solution for online adaptation in non-stationary environments and is well suited for real-world robotic control and edge deployment.

**arXiv ID:** 2602.06326
</details>

<details>
<summary><strong>The hidden risks of temporal resampling in clinical reinforcement learning</strong> - Thomas Frost, Hrisheekesh Vaidya, Steve Harris - [[pdf]](https://arxiv.org/pdf/2602.06603)</summary>

**Abstract:** Offline reinforcement learning (ORL) has shown potential for improving decision-making in healthcare. However, contemporary research typically aggregates patient data into fixed time intervals, simplifying their mapping to standard ORL frameworks. The impact of these temporal manipulations on model safety and efficacy remains poorly understood. In this work, using both a gridworld navigation task and the UVA/Padova clinical diabetes simulator, we demonstrate that temporal resampling significantly degrades the performance of offline reinforcement learning algorithms during live deployment. We propose three mechanisms that drive this failure: (i) the generation of counterfactual trajectories, (ii) the distortion of temporal expectations, and (iii) the compounding of generalisation errors. Crucially, we find that standard off-policy evaluation metrics can fail to detect these drops in performance. Our findings reveal a fundamental risk in current healthcare ORL pipelines and emphasise the need for methods that explicitly handle the irregular timing of clinical decision-making.

**arXiv ID:** 2602.06603
</details>

<details>
<summary><strong>Soft Forward-Backward Representations for Zero-shot Reinforcement Learning with General Utilities</strong> - Marco Bagatella, Thomas Rupf, Georg Martius, Andreas Krause - [[pdf]](https://arxiv.org/pdf/2602.06769)</summary>

**Abstract:** Recent advancements in zero-shot reinforcement learning (RL) have facilitated the extraction of diverse behaviors from unlabeled, offline data sources. In particular, forward-backward algorithms (FB) can retrieve a family of policies that can approximately solve any standard RL problem (with additive rewards, linear in the occupancy measure), given sufficient capacity. While retaining zero-shot properties, we tackle the greater problem class of RL with general utilities, in which the objective is an arbitrary differentiable function of the occupancy measure. This setting is strictly more expressive, capturing tasks such as distribution matching or pure exploration, which may not be reduced to additive rewards. We show that this additional complexity can be captured by a novel, maximum entropy (soft) variant of the forward-backward algorithm, which recovers a family of stochastic policies from offline data. When coupled with zero-order search over compact policy embeddings, this algorithm can sidestep iterative optimization schemes, and optimizes general utilities directly at test-time. Across both didactic and high-dimensional experiments, we demonstrate that our method retains favorable properties of FB algorithms, while also extending their range to more general RL problems.

**arXiv ID:** 2602.06769
</details>

<details>
<summary><strong>A first realization of reinforcement learning-based closed-loop EEG-TMS</strong> - Dania Humaidan, Jiahua Xu, Jing Chen, Christoph Zrenner, David Emanuel Vetter, Laura Marzetti, Paolo Belardinelli, Timo Roine, Risto J. Ilmoniemi, Gian Luca Romani, Ulf Zieman - [[pdf]](https://arxiv.org/pdf/2602.06907)</summary>

**Abstract:** 

**arXiv ID:** 2602.06907
</details>

<details>
<summary><strong>Continuous-time reinforcement learning: ellipticity enables model-free value function approximation</strong> - Wenlong Mou - [[pdf]](https://arxiv.org/pdf/2602.06930)</summary>

**Abstract:** We study off-policy reinforcement learning for controlling continuous-time Markov diffusion processes with discrete-time observations and actions. We consider model-free algorithms with function approximation that learn value and advantage functions directly from data, without unrealistic structural assumptions on the dynamics.
Leveraging the ellipticity of the diffusions, we establish a new class of Hilbert-space positive definiteness and boundedness properties for the Bellman operators. Based on these properties, we propose the Sobolev-prox fitted $q$-learning algorithm, which learns value and advantage functions by iteratively solving least-squares regression problems. We derive oracle inequalities for the estimation error, governed by (i) the best approximation error of the function classes, (ii) their localized complexity, (iii) exponentially decaying optimization error, and (iv) numerical discretization error. These results identify ellipticity as a key structural property that renders reinforcement learning with function approximation for Markov diffusions no harder than supervised learning.

**arXiv ID:** 2602.06930
</details>

<details>
<summary><strong>Reinforcement Learning-Based Dynamic Management of Structured Parallel Farm Skeletons on Serverless Platforms</strong> - Lanpei Li, Massimo Coppola, Malio Li, Valerio Besozzi, Jack Bell, Vincenzo Lomonaco - [[pdf]](https://arxiv.org/pdf/2602.06555)</summary>

**Abstract:** We present a framework for dynamic management of structured parallel processing skeletons on serverless platforms. Our goal is to bring HPC-like performance and resilience to serverless and continuum environments while preserving the programmability benefits of skeletons. As a first step, we focus on the well known Farm pattern and its implementation on the open-source OpenFaaS platform, treating autoscaling of the worker pool as a QoS-aware resource management problem. The framework couples a reusable farm template with a Gymnasium-based monitoring and control layer that exposes queue, timing, and QoS metrics to both reactive and learning-based controllers. We investigate the effectiveness of AI-driven dynamic scaling for managing the farm's degree of parallelism via the scalability of serverless functions on OpenFaaS. In particular, we discuss the autoscaling model and its training, and evaluate two reinforcement learning (RL) policies against a baseline of reactive management derived from a simple farm performance model. Our results show that AI-based management can better accommodate platform-specific limitations than purely model-based performance steering, improving QoS while maintaining efficient resource usage and stable scaling behaviour.

**arXiv ID:** 2602.06555
</details>

<details>
<summary><strong>ECO: Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking</strong> - Weidong Huang, Jingwen Zhang, Jiongye Li, Shibowen Zhang, Jiayang Wu, Jiayi Wang, Hangxin Liu, Yaodong Yang, Yao Su - [[pdf]](https://arxiv.org/pdf/2602.06445)</summary>

**Abstract:** Achieving stable and energy-efficient locomotion is essential for humanoid robots to operate continuously in real-world applications. Existing MPC and RL approaches often rely on energy-related metrics embedded within a multi-objective optimization framework, which require extensive hyperparameter tuning and often result in suboptimal policies. To address these challenges, we propose ECO (Energy-Constrained Optimization), a constrained RL framework that separates energy-related metrics from rewards, reformulating them as explicit inequality constraints. This method provides a clear and interpretable physical representation of energy costs, enabling more efficient and intuitive hyperparameter tuning for improved energy efficiency. ECO introduces dedicated constraints for energy consumption and reference motion, enforced by the Lagrangian method, to achieve stable, symmetric, and energy-efficient walking for humanoid robots. We evaluated ECO against MPC, standard RL with reward shaping, and four state-of-the-art constrained RL methods. Experiments, including sim-to-sim and sim-to-real transfers on the kid-sized humanoid robot BRUCE, demonstrate that ECO significantly reduces energy consumption compared to baselines while maintaining robust walking performance. These results highlight a substantial advancement in energy-efficient humanoid locomotion. All experimental demonstrations can be found on the project website: this https URL.

**arXiv ID:** 2602.06445
</details>

<details>
<summary><strong>DriveWorld-VLA: Unified Latent-Space World Modeling with Vision-Language-Action for Autonomous Driving</strong> - Feiyang jia, Lin Liu, Ziying Song, Caiyan Jia, Hangjun Ye, Xiaoshuai Hao, Long Chen - [[pdf]](https://arxiv.org/pdf/2602.06521)</summary>

**Abstract:** End-to-end (E2E) autonomous driving has recently attracted increasing interest in unifying Vision-Language-Action (VLA) with World Models to enhance decision-making and forward-looking imagination. However, existing methods fail to effectively unify future scene evolution and action planning within a single architecture due to inadequate sharing of latent states, limiting the impact of visual imagination on action decisions. To address this limitation, we propose DriveWorld-VLA, a novel framework that unifies world modeling and planning within a latent space by tightly integrating VLA and world models at the representation level, which enables the VLA planner to benefit directly from holistic scene-evolution modeling and reducing reliance on dense annotated supervision. Additionally, DriveWorld-VLA incorporates the latent states of the world model as core decision-making states for the VLA planner, facilitating the planner to assess how candidate actions impact future scene evolution. By conducting world modeling entirely in the latent space, DriveWorld-VLA supports controllable, action-conditioned imagination at the feature level, avoiding expensive pixel-level rollouts. Extensive open-loop and closed-loop evaluations demonstrate the effectiveness of DriveWorld-VLA, which achieves state-of-the-art performance with 91.3 PDMS on NAVSIMv1, 86.8 EPDMS on NAVSIMv2, and 0.16 3-second average collision rate on nuScenes. Code and models will be released in this https URL.

**arXiv ID:** 2602.06521
</details>

<details>
<summary><strong>Rethinking External Communication of Autonomous Vehicles: Is the Field Converging, Diverging, or Stalling?</strong> - Tram Thi Minh Tran, Debargha Dey, Martin Tomitsch - [[pdf]](https://arxiv.org/pdf/2602.06278)</summary>

**Abstract:** As autonomous vehicles enter public spaces, external human-machine interfaces are proposed to support communication with external road users. A decade of research has produced hundreds of studies and reviews, yet it remains unclear whether the field is converging on shared principles or diverging across approaches. We present a multi-dimensional analysis of 620 publications, complemented by industry deployments and regulatory documents, to track research evolution and identify convergence. The analysis reveals several field-level patterns. First, convergence on a safety-first core: simple visual cues that clarify intent. Second, sustained divergence in necessity and implementation. Third, a progressive filtering funnel: broad exploration in research and concepts narrows in deployment and is codified by regulation into a minimal set of permitted signals. These insights point to a shift in emphasis for future work, from producing new prototypes toward consolidating evidence, clarifying points of contention, and developing frameworks that can adapt across contexts.

**arXiv ID:** 2602.06278
</details>

<details>
<summary><strong>Simulating Word Suggestion Usage in Mobile Typing to Guide Intelligent Text Entry Design</strong> - Yang Li, Anna Maria Feit - [[pdf]](https://arxiv.org/pdf/2602.06489)</summary>

**Abstract:** Intelligent text entry (ITE) methods, such as word suggestions, are widely used in mobile typing, yet improving ITE systems is challenging because the cognitive mechanisms behind suggestion use remain poorly understood, and evaluating new systems often requires long-term user studies to account for behavioral adaptation. We present WSTypist, a reinforcement learning-based model that simulates how typists integrate word suggestions into typing. It builds on recent hierarchical control models of typing, but focuses on the cognitive mechanisms that underlie the high-level decision-making for effectively integrating word suggestions into manual typing: assessing efficiency gains, considering orthographic uncertainties, and including personal reliance on AI support. Our evaluations show that WSTypist simulates diverse human-like suggestion-use strategies, reproduces individual differences, and generalizes across different systems. Importantly, we demonstrate on four design cases how computational rationality models can be used to inform what-if analyses during the design process, by simulating how users might adapt to changes in the UI or in the algorithmic support, reducing the need for long-term user studies.

**arXiv ID:** 2602.06489
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
