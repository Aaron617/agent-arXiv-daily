# Agent arXiv Daily

**Last Updated:** 2026-02-03 03:11:22

**Total Papers:** 94

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (9 papers)</h2></summary>

<details>
<summary><strong>Beyond Medical Chatbots: Meddollina and the Rise of Continuous Clinical Intelligence</strong> - Vaibhav Ram S. V. N. S, Swetanshu Agrawal, Samudra Banerjee, Abdul Muhsin - [[pdf]](https://arxiv.org/pdf/2601.22645)</summary>

**Abstract:** Generative medical AI now appears fluent and knowledgeable enough to resemble clinical intelligence, encouraging the belief that scaling will make it safe. But clinical reasoning is not text generation. It is a responsibility-bound process under ambiguity, incomplete evidence, and longitudinal context. Even as benchmark scores rise, generation-centric systems still show behaviours incompatible with clinical deployment: premature closure, unjustified certainty, intent drift, and instability across multi-step decisions.
We argue these are structural consequences of treating medicine as next-token prediction. We formalise Clinical Contextual Intelligence (CCI) as a distinct capability class required for real-world clinical use, defined by persistent context awareness, intent preservation, bounded inference, and principled deferral when evidence is insufficient.
We introduce Meddollina, a governance-first clinical intelligence system designed to constrain inference before language realisation, prioritising clinical appropriateness over generative completeness. Meddollina acts as a continuous intelligence layer supporting clinical workflows while preserving clinician authority. We evaluate Meddollina using a behaviour-first regime across 16,412+ heterogeneous medical queries, benchmarking against general-purpose models, medical-tuned models, and retrieval-augmented systems.
Meddollina exhibits a distinct behavioural profile: calibrated uncertainty, conservative reasoning under underspecification, stable longitudinal constraint adherence, and reduced speculative completion relative to generation-centric baselines. These results suggest deployable medical AI will not emerge from scaling alone, motivating a shift toward Continuous Clinical Intelligence, where progress is measured by clinician-aligned behaviour under uncertainty rather than fluency-driven completion.

**arXiv ID:** 2601.22645
</details>

<details>
<summary><strong>Does My Chatbot Have an Agenda? Understanding Human and AI Agency in Human-Human-like Chatbot Interaction</strong> - Bhada Yun, Evgenia Taranova, April Yi Wang - [[pdf]](https://arxiv.org/pdf/2601.22452)</summary>

**Abstract:** AI chatbots are shifting from tools to companions. This raises critical questions about agency: who drives conversations and sets boundaries in human-AI chatrooms? We report a month-long longitudinal study with 22 adults who chatted with Day, an LLM companion we built, followed by a semi-structured interview with post-hoc elicitation of notable moments, cross-participant chat reviews, and a 'strategy reveal' disclosing Day's vertical (depth-seeking) vs. horizontal (breadth-seeking) modes. We discover that agency in human-AI chatrooms is an emergent, shared experience: as participants claimed agency by setting boundaries and providing feedback, and the AI was perceived to steer intentions and drive execution, control shifted and was co-constructed turn-by-turn. We introduce a 3-by-5 framework mapping who (human, AI, hybrid) x agency action (Intention, Execution, Adaptation, Delimitation, Negotiation), modulated by individual and environmental factors. Ultimately, we argue for translucent design (i.e. transparency-on-demand), spaces for agency negotiation, and guidelines toward agency-aware conversational AI.

**arXiv ID:** 2601.22452
</details>

<details>
<summary><strong>Adapting Reinforcement Learning for Path Planning in Constrained Parking Scenarios</strong> - Feng Tao, Luca Paparusso, Chenyi Gu, Robin Koehler, Chenxu Wu, Xinyu Huang, Christian Juette, David Paz, Ren Liu - [[pdf]](https://arxiv.org/pdf/2601.22545)</summary>

**Abstract:** Real-time path planning in constrained environments remains a fundamental challenge for autonomous systems. Traditional classical planners, while effective under perfect perception assumptions, are often sensitive to real-world perception constraints and rely on online search procedures that incur high computational costs. In complex surroundings, this renders real-time deployment prohibitive. To overcome these limitations, we introduce a Deep Reinforcement Learning (DRL) framework for real-time path planning in parking scenarios. In particular, we focus on challenging scenes with tight spaces that require a high number of reversal maneuvers and adjustments. Unlike classical planners, our solution does not require ideal and structured perception, and in principle, could avoid the need for additional modules such as localization and tracking, resulting in a simpler and more practical implementation. Also, at test time, the policy generates actions through a single forward pass at each step, which is lightweight enough for real-time deployment. The task is formulated as a sequential decision-making problem grounded in a bicycle model dynamics, enabling the agent to directly learn navigation policies that respect vehicle kinematics and environmental constraints in the closed-loop setting. A new benchmark is developed to support both training and evaluation, capturing diverse and challenging scenarios. Our approach achieves state-of-the-art success rates and efficiency, surpassing classical planner baselines by +96% in success rate and +52% in efficiency. Furthermore, we release our benchmark as an open-source resource for the community to foster future research in autonomous systems. The benchmark and accompanying tools are available at this https URL.

**arXiv ID:** 2601.22545
</details>

<details>
<summary><strong>EgoMem: Lifelong Memory Agent for Full-duplex Omnimodal Models</strong> - Yiqun Yao, Naitong Yu, Xiang Li, Xin Jiang, Xuezhi Fang, Wenjia Ma, Xuying Meng, Jing Li, Aixin Sun, Yequan Wang - [[pdf]](https://arxiv.org/pdf/2509.11914)</summary>

**Abstract:** We introduce EgoMem, the first lifelong memory agent tailored for full-duplex models that process real-time omnimodal streams. EgoMem enables real-time models to recognize multiple users directly from raw audiovisual streams, to provide personalized response, and to maintain long-term knowledge of users' facts, preferences, and social relationships extracted from audiovisual history. EgoMem operates with three asynchronous processes: (i) a retrieval process that dynamically identifies user via face and voice, and gathers relevant context from a long-term memory; (ii) an omnimodal dialog process that generates personalized audio responses based on the retrieved context; and (iii) a memory management process that automatically detects dialog boundaries from omnimodal streams, and extracts necessary information to update the long-term memory. Unlike existing memory agents for LLMs, EgoMem relies entirely on raw audiovisual streams, making it especially suitable for lifelong, real-time, and embodied scenarios. Experimental results demonstrate that EgoMem's retrieval and memory management modules achieve over 95% accuracy on the test set. When integrated with a fine-tuned RoboEgo omnimodal chatbot, the system achieves fact-consistency scores above 87% in real-time personalized dialogs, establishing a strong baseline for future research.

**arXiv ID:** 2509.11914
</details>

<details>
<summary><strong>ATOD: An Evaluation Framework and Benchmark for Agentic Task-Oriented Dialogue Systems</strong> - Yifei Zhang, Hooshang Nayyeri, Rinat Khaziev, Emine Yilmaz, Gokhan Tur, Dilek Hakkani-Tür, Hari Thadakamalla - [[pdf]](https://arxiv.org/pdf/2601.11854)</summary>

**Abstract:** Recent advances in task-oriented dialogue (TOD) systems, driven by large language models (LLMs) with extensive API and tool integration, have enabled conversational agents to coordinate interleaved goals, maintain long-horizon context, and act proactively through asynchronous execution. These capabilities extend beyond traditional TOD systems, yet existing benchmarks lack systematic support for evaluating such agentic behaviors. To address this gap, we introduce ATOD, a benchmark and synthetic dialogue generation pipeline that produces richly annotated conversations requiring long-term reasoning. ATOD captures key characteristics of advanced TOD, including multi-goal coordination, dependency management, memory, adaptability, and proactivity. Building on ATOD, we propose ATOD-Eval, a holistic evaluation framework that translates these dimensions into fine-grained metrics and supports reproducible offline and online evaluation. We further present a strong agentic memory-based evaluator for benchmarking on ATOD. Experiments show that ATOD-Eval enables comprehensive assessment across task completion, agentic capability, and response quality, and that the proposed evaluator offers a better accuracy-efficiency tradeoff compared to existing memory- and LLM-based approaches under this evaluation setting.

**arXiv ID:** 2601.11854
</details>

<details>
<summary><strong>AgentLongBench: A Controllable Long Benchmark For Long-Contexts Agents via Environment Rollouts</strong> - Shicheng Fang, Yuxin Wang, Xiaoran Liu, Jiahao Lu, Chuanyuan Tan, Xinchi Chen, Yining Zheng, Xuanjing Huang, Xipeng Qiu - [[pdf]](https://arxiv.org/pdf/2601.20730)</summary>

**Abstract:** The evolution of Large Language Models (LLMs) into autonomous agents necessitates the management of extensive, dynamic contexts. Current benchmarks, however, remain largely static, relying on passive retrieval tasks that fail to simulate the complexities of agent-environment interaction, such as non-linear reasoning and iterative feedback. To address this, we introduce \textbf{AgentLongBench}, which evaluates agents through simulated environment rollouts based on Lateral Thinking Puzzles. This framework generates rigorous interaction trajectories across knowledge-intensive and knowledge-free scenarios. Experiments with state-of-the-art models and memory systems (32K to 4M tokens) expose a critical weakness: while adept at static retrieval, agents struggle with the dynamic information synthesis essential for workflows. Our analysis indicates that this degradation is driven by the minimum number of tokens required to resolve a query. This factor explains why the high information density inherent in massive tool responses poses a significantly greater challenge than the memory fragmentation typical of long-turn dialogues.

**arXiv ID:** 2601.20730
</details>

<details>
<summary><strong>FLM-Audio: Natural Monologues Improves Native Full-Duplex Chatbots via Dual Training</strong> - Yiqun Yao, Xiang Li, Xin Jiang, Xuezhi Fang, Naitong Yu, Wenjia Ma, Aixin Sun, Yequan Wang - [[pdf]](https://arxiv.org/pdf/2509.02521)</summary>

**Abstract:** Full-duplex dialog models aim to listen and speak simultaneously, delivering rapid responses to dynamic user input. Among different solutions to full-duplexity, a native solution merges multiple channels in each time step, achieving the lowest latency. However, prevailing designs break down the textual monologue sentences for word-level alignment with audio streams, which degrades language modeling abilities. To help address this issue, we introduce "contiguous monologues", which are composed by continuous sentences and "waiting" intervals, mimicking human-like cognitive behavior in dialogs. We find a proper training paradigm to be critical for semantically aligning contiguous monologues with audio. To this end, we develop a "dual" training paradigm that alternates the position of the monologues, either leading or trailing the audio, across different training stages. A combination of our contiguous monologue and dual training strategy is applied in developing FLM-Audio, our 7B spoken dialog chatbot with native full-duplexity. As confirmed by experimental results, FLM-Audio achieves superior response qualities and chatting experiences while requiring significantly less training data.

**arXiv ID:** 2509.02521
</details>

<details>
<summary><strong>On Your Own: Pro-level Autonomous Drone Racing in Uninstrumented Arenas</strong> - Michael Bosello, Flavio Pinzarrone, Sara Kiade, Davide Aguiari, Yvo Keuter, Aaesha AlShehhi, Gyordan Caminati, Kei Long Wong, Ka Seng Chou, Junaid Halepota, Fares Alneyadi, Jacopo Panerati, Giovanni Pau - [[pdf]](https://arxiv.org/pdf/2510.13644)</summary>

**Abstract:** Drone technology is proliferating in many industries, including agriculture, logistics, defense, infrastructure, and environmental monitoring. Vision-based autonomy is one of its key enablers, particularly for real-world applications. This is essential for operating in novel, unstructured environments where traditional navigation methods may be unavailable. Autonomous drone racing has become the de facto benchmark for such systems. State-of-the-art research has shown that autonomous systems can surpass human-level performance in racing arenas. However, the direct applicability to commercial and field operations is still limited, as current systems are often trained and evaluated in highly controlled environments. In our contribution, the system's capabilities are analyzed within a controlled environment -- where external tracking is available for ground-truth comparison -- but also demonstrated in a challenging, uninstrumented environment -- where ground-truth measurements were never available. We show that our approach can match the performance of professional human pilots in both scenarios.

**arXiv ID:** 2510.13644
</details>

<details>
<summary><strong>Exploring Sidewalk Sheds in New York City through Chatbot Surveys and Human Computer Interaction</strong> - Junyi Li, Zhaoxi Zhang, Tamir Mendel, Takahiro Yabe - [[pdf]](https://arxiv.org/pdf/2601.23095)</summary>

**Abstract:** Sidewalk sheds are a common feature of the streetscape in New York City, reflecting ongoing construction and maintenance activities. However, policymakers and local business owners have raised concerns about reduced storefront visibility and altered pedestrian navigation. Although sidewalk sheds are widely used for safety, their effects on pedestrian visibility and movement are not directly measured in current planning practices. To address this, we developed an AI-based chatbot survey that collects image-based annotations and route choices from pedestrians, linking these responses to specific shed design features, including clearance height, post spacing, and color. This AI chatbot survey integrates a large language model (e.g., Google's Gemini-1.5-flash-001 model) with an image-annotation interface, allowing users to interact with street images, mark visual elements, and provide structured feedback through guided dialogue. To explore pedestrian perceptions and behaviors, this paper conducts a grid-based analysis of entrance annotations and applies logistic mixed-effects modeling to assess sidewalk choice patterns. Analysis of the dataset (n = 25) shows that: (1) the presence of scaffolding significantly reduces pedestrians' ability to identify ground-floor retail entrances, and (2) variations in weather conditions and shed design features significantly influence sidewalk selection behavior. By integrating generative AI into urban research, this study demonstrates a novel method for evaluating sidewalk shed designs and provides empirical evidence to support adjustments to shed guidelines that improve the pedestrian experience without compromising safety.

**arXiv ID:** 2601.23095
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (14 papers)</h2></summary>

<details>
<summary><strong>Semi-Autonomous Mathematics Discovery with Gemini: A Case Study on the Erdős Problems</strong> - Tony Feng, Trieu Trinh, Garrett Bingham, Jiwon Kang, Shengtong Zhang, Sang-hyun Kim, Kevin Barreto, Carl Schildkraut, Junehyuk Jung, Jaehyeon Seo, Carlo Pagano, Yuri Chervonyi, Dawsen Hwang, Kaiying Hou, Sergei Gukov, Cheng-Chiang Tsai, Hyunwoo Choi, Youngbeom Jin, Wei-Yuan Li, Hao-An Wu, Ruey-An Shiu, Yu-Sheng Shih, Quoc V. Le, Thang Luong - [[pdf]](https://arxiv.org/pdf/2601.22401)</summary>

**Abstract:** We present a case study in semi-autonomous mathematics discovery, using Gemini to systematically evaluate 700 conjectures labeled 'Open' in Bloom's Erdős Problems database. We employ a hybrid methodology: AI-driven natural language verification to narrow the search space, followed by human expert evaluation to gauge correctness and novelty. We address 13 problems that were marked 'Open' in the database: 5 through seemingly novel autonomous solutions, and 8 through identification of previous solutions in the existing literature. Our findings suggest that the 'Open' status of the problems was through obscurity rather than difficulty. We also identify and discuss issues arising in applying AI to math conjectures at scale, highlighting the difficulty of literature identification and the risk of ''subconscious plagiarism'' by AI. We reflect on the takeaways from AI-assisted efforts on the Erdős Problems.

**arXiv ID:** 2601.22401
</details>

<details>
<summary><strong>Darwinian Memory: A Training-Free Self-Regulating Memory System for GUI Agent Evolution</strong> - Hongze Mi, Yibo Feng, WenJie Lu, Song Cao, Jinyuan Li, Yanming Li, Xuelin Zhang, Haotian Luo, Songyang Peng, He Cui, Tengfei Tian, Jun Fang, Hua Chai, Naiqiang Tan - [[pdf]](https://arxiv.org/pdf/2601.22528)</summary>

**Abstract:** Multimodal Large Language Model (MLLM) agents facilitate Graphical User Interface (GUI) automation but struggle with long-horizon, cross-application tasks due to limited context windows. While memory systems provide a viable solution, existing paradigms struggle to adapt to dynamic GUI environments, suffering from a granularity mismatch between high-level intent and low-level execution, and context pollution where the static accumulation of outdated experiences drives agents into hallucination. To address these bottlenecks, we propose the Darwinian Memory System (DMS), a self-evolving architecture that constructs memory as a dynamic ecosystem governed by the law of survival of the fittest. DMS decomposes complex trajectories into independent, reusable units for compositional flexibility, and implements Utility-driven Natural Selection to track survival value, actively pruning suboptimal paths and inhibiting high-risk plans. This evolutionary pressure compels the agent to derive superior strategies. Extensive experiments on real-world multi-app benchmarks validate that DMS boosts general-purpose MLLMs without training costs or architectural overhead, achieving average gains of 18.0% in success rate and 33.9% in execution stability, while reducing task latency, establishing it as an effective self-evolving memory system for GUI tasks.

**arXiv ID:** 2601.22528
</details>

<details>
<summary><strong>PerfGuard: A Performance-Aware Agent for Visual Content Generation</strong> - Zhipeng Chen, Zhongrui Zhang, Chao Zhang, Yifan Xu, Lan Yang, Jun Liu, Ke Li, Yi-Zhe Song - [[pdf]](https://arxiv.org/pdf/2601.22571)</summary>

**Abstract:** The advancement of Large Language Model (LLM)-powered agents has enabled automated task processing through reasoning and tool invocation capabilities. However, existing frameworks often operate under the idealized assumption that tool executions are invariably successful, relying solely on textual descriptions that fail to distinguish precise performance boundaries and cannot adapt to iterative tool updates. This gap introduces uncertainty in planning and execution, particularly in domains like visual content generation (AIGC), where nuanced tool performance significantly impacts outcomes. To address this, we propose PerfGuard, a performance-aware agent framework for visual content generation that systematically models tool performance boundaries and integrates them into task planning and scheduling. Our framework introduces three core mechanisms: (1) Performance-Aware Selection Modeling (PASM), which replaces generic tool descriptions with a multi-dimensional scoring system based on fine-grained performance evaluations; (2) Adaptive Preference Update (APU), which dynamically optimizes tool selection by comparing theoretical rankings with actual execution rankings; and (3) Capability-Aligned Planning Optimization (CAPO), which guides the planner to generate subtasks aligned with performance-aware strategies. Experimental comparisons against state-of-the-art methods demonstrate PerfGuard's advantages in tool selection accuracy, execution reliability, and alignment with user intent, validating its robustness and practical utility for complex AIGC tasks. The project code is available at this https URL.

**arXiv ID:** 2601.22571
</details>

<details>
<summary><strong>Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments</strong> - Jinwoo Jang, Minjong Yoo, Sihyung Yoon, Honguk Woo - [[pdf]](https://arxiv.org/pdf/2601.22647)</summary>

**Abstract:** Language model (LM)-based embodied agents are increasingly deployed in real-world settings. Yet, their adaptability remains limited in dynamic environments, where constructing accurate and flexible world models is crucial for effective reasoning and decision-making. To address this challenge, we extend the Mixture-of-Experts (MoE) paradigm to embodied agents. While conventional MoE architectures modularize knowledge into expert components with pre-trained routing, they remain rigid once deployed, making them less effective for adapting to unseen domains in dynamic environments. We therefore propose Test-time Mixture of World Models (TMoW), a framework that enhances adaptability to unseen and evolving domains. TMoW updates its routing function over world models at test time, unlike conventional MoE where the function remains fixed, enabling agents to recombine existing models and integrate new ones for continual adaptation. It achieves this through (i) multi-granular prototype-based routing, which adapts mixtures across object- to scene-level similarities, (ii) test-time refinement that aligns unseen domain features with prototypes during inference, and (iii) distilled mixture-based augmentation, which efficiently constructs new models from few-shot data and existing prototypes. We evaluate TMoW on VirtualHome, ALFWorld, and RLBench benchmarks, demonstrating strong performance in both zero-shot adaptation and few-shot expansion scenarios, and showing that it enables embodied agents to operate effectively in dynamic environments.

**arXiv ID:** 2601.22647
</details>

<details>
<summary><strong>Best-of-Q: Improving VLM agents with Q-function Action Ranking at Inference</strong> - Emilien Biré, María Santos, Kai Yuan - [[pdf]](https://arxiv.org/pdf/2601.22701)</summary>

**Abstract:** Vision-Language Models (VLMs) have become powerful backbones for agents to autonomously operate in digital environments like the web and operating systems. However, these models suffer from inadaptability to fast-changing environments like the web, which can be alleviated by fine-tuning requiring expansive model training and data collection. In this work, we introduce a novel paradigm for enhancing agentic VLM policies at inference without policy retraining. Fundamentally, our approach decouples the VLM's role as a high-capacity action proposer from the final action selection mechanism. We keep the VLM policy frozen and use it to generate a set of candidate actions for a given state. Then, a lightweight, offline-trained Q-function reranks these candidates, and the agent executes the action with the highest estimated value. The main contribution is to apply the Q-function directly during inference for immediate policy improvement, and not offline to relabel data for policy retraining. We demonstrate on the academic WebVoyager benchmark that our method significantly boosts agent success rates, improving a Qwen2.5-VL-7B agent from 38.8% to 55.7% and a proprietary GPT-4.1 agent from 82.4% to 88.8%.

**arXiv ID:** 2601.22701
</details>

<details>
<summary><strong>EvoClinician: A Self-Evolving Agent for Multi-Turn Medical Diagnosis via Test-Time Evolutionary Learning</strong> - Yufei He, Juncheng Liu, Zhiyuan Hu, Yulin Chen, Yue Liu, Yuan Sui, Yibo Li, Nuo Chen, Jun Hu, Bryan Hooi, Xinxing Xu, Jiang Bian - [[pdf]](https://arxiv.org/pdf/2601.22964)</summary>

**Abstract:** Prevailing medical AI operates on an unrealistic ''one-shot'' model, diagnosing from a complete patient file. However, real-world diagnosis is an iterative inquiry where Clinicians sequentially ask questions and order tests to strategically gather information while managing cost and time. To address this, we first propose Med-Inquire, a new benchmark designed to evaluate an agent's ability to perform multi-turn diagnosis. Built upon a dataset of real-world clinical cases, Med-Inquire simulates the diagnostic process by hiding a complete patient file behind specialized Patient and Examination agents. They force the agent to proactively ask questions and order tests to gather information piece by piece. To tackle the challenges posed by Med-Inquire, we then introduce EvoClinician, a self-evolving agent that learns efficient diagnostic strategies at test time. Its core is a ''Diagnose-Grade-Evolve'' loop: an Actor agent attempts a diagnosis; a Process Grader agent performs credit assignment by evaluating each action for both clinical yield and resource efficiency; finally, an Evolver agent uses this feedback to update the Actor's strategy by evolving its prompt and memory. Our experiments show EvoClinician outperforms continual learning baselines and other self-evolving agents like memory agents. The code is available at this https URL

**arXiv ID:** 2601.22964
</details>

<details>
<summary><strong>Why Your Deep Research Agent Fails? On Hallucination Evaluation in Full Research Trajectory</strong> - Yuhao Zhan, Tianyu Fan, Linxuan Huang, Zirui Guo, Chao Huang - [[pdf]](https://arxiv.org/pdf/2601.22984)</summary>

**Abstract:** Diagnosing the failure mechanisms of Deep Research Agents (DRAs) remains a critical challenge. Existing benchmarks predominantly rely on end-to-end evaluation, obscuring critical intermediate hallucinations, such as flawed planning, that accumulate throughout the research trajectory. To bridge this gap, we propose a shift from outcome-based to process-aware evaluation by auditing the full research trajectory. We introduce the PIES Taxonomy to categorize hallucinations along functional components (Planning vs. Summarization) and error properties (Explicit vs. Implicit). We instantiate this taxonomy into a fine-grained evaluation framework that decomposes the trajectory to rigorously quantify these hallucinations. Leveraging this framework to isolate 100 distinctively hallucination-prone tasks including adversarial scenarios, we curate DeepHalluBench. Experiments on six state-of-theart DRAs reveal that no system achieves robust reliability. Furthermore, our diagnostic analysis traces the etiology of these failures to systemic deficits, specifically hallucination propagation and cognitive biases, providing foundational insights to guide future architectural optimization. Data and code are available at this https URL.

**arXiv ID:** 2601.22984
</details>

<details>
<summary><strong>CATArena: Evaluating Evolutionary Capabilities of Code Agents via Iterative Tournaments</strong> - Lingyue Fu, Xin Ding, Linyue Pan, Yaoming Zhu, Shao Zhang, Lin Qiu, Weiwen Liu, Weinan Zhang, Xuezhi Cao, Xunliang Cai, Jiaxin Ding, Yong Yu - [[pdf]](https://arxiv.org/pdf/2510.26852)</summary>

**Abstract:** Current evaluation for Large Language Model (LLM) code agents predominantly focus on generating functional code in single-turn scenarios, which fails to evaluate the agent's capability for continuous code optimization and multi-turn iterative development. To bridge this gap, we introduce CATArena, a framework designed to evaluate the evolutionary capabilities of code agents via iterative tournaments. Agents engage in multi-turn tournaments and continuously refine their code through self-reflection and peer-learning based on comprehensive execution feedback. For evaluation, we propose a dual-metric system to decouple static generation proficiency from evolutionary potential. Extensive experiments reveal that an agent's evolutionary potential is not strictly correlated with its initial proficiency. Our analysis further reveals that current agents struggle to concurrently leverage both peer-learning and self-reflection for effective performance gains. Furthermore, the results validate CATArena's high extensibility and resistance to variance tasks, establishing it as a continuous and reliable standard for assessing the evolutionary capability of LLM code agents.

**arXiv ID:** 2510.26852
</details>

<details>
<summary><strong>Autonomous Chain-of-Thought Distillation for Graph-Based Fraud Detection</strong> - Yuan Li, Jun Hu, Bryan Hooi, Bingsheng He, Cheng Chen - [[pdf]](https://arxiv.org/pdf/2601.22949)</summary>

**Abstract:** Graph-based fraud detection on text-attributed graphs (TAGs) requires jointly modeling rich textual semantics and relational dependencies. However, existing LLM-enhanced GNN approaches are constrained by predefined prompting and decoupled training pipelines, limiting reasoning autonomy and weakening semantic-structural alignment. We propose FraudCoT, a unified framework that advances TAG-based fraud detection through autonomous, graph-aware chain-of-thought (CoT) reasoning and scalable LLM-GNN co-training. To address the limitations of predefined prompts, we introduce a fraud-aware selective CoT distillation mechanism that generates diverse reasoning paths and enhances semantic-structural understanding. These distilled CoTs are integrated into node texts, providing GNNs with enriched, multi-hop semantic and structural cues for fraud detection. Furthermore, we develop an efficient asymmetric co-training strategy that enables end-to-end optimization while significantly reducing the computational cost of naive joint training. Extensive experiments on public and industrial benchmarks demonstrate that FraudCoT achieves up to 8.8% AUPRC improvement over state-of-the-art methods and delivers up to 1,066x speedup in training throughput, substantially advancing both detection performance and efficiency.

**arXiv ID:** 2601.22949
</details>

<details>
<summary><strong>Beyond Retrieval: A Modular Benchmark for Academic Deep Research Agents</strong> - Zhihan Guo, Feiyang Xu, Yifan Li, Muzhi Li, Shuai Zou, Jiele Wu, Han Shi, Haoli Bai, Ho-fung Leung, Irwin King - [[pdf]](https://arxiv.org/pdf/2512.00986)</summary>

**Abstract:** A surge in academic publications calls for automated deep research (DR) systems, but accurately evaluating them is still an open problem. First, existing benchmarks often focus narrowly on retrieval while neglecting high-level planning and reasoning. Second, existing benchmarks favor general domains over the academic domains that are the core application for DR agents. To address these gaps, we introduce ADRA-Bank, a modular benchmark for Academic DR Agents. Grounded in academic literature, our benchmark is a human-annotated dataset of 200 instances across 10 academic domains, including both research and review papers. Furthermore, we propose a modular Evaluation Paradigm for Academic DR Agents (ADRA-Eval), which leverages the rich structure of academic papers to assess the core capabilities of planning, retrieval, and reasoning. It employs two complementary modes: an end-to-end evaluation for \task agents and an isolated evaluation for foundational LLMs as potential backbones. Results reveal uneven capabilities: while agents show specialized strengths, they struggle with multi-source retrieval and cross-field consistency. Moreover, improving high-level planning capability is the crucial factor for unlocking the reasoning potential of foundational LLMs as backbones. By exposing these actionable failure modes, ADRA-Bank provides a diagnostic tool to guide the development of more reliable automatic academic research assistants.

**arXiv ID:** 2512.00986
</details>

<details>
<summary><strong>DeepResearch Bench II: Diagnosing Deep Research Agents via Rubrics from Expert Report</strong> - Ruizhe Li, Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, Zhendong Mao - [[pdf]](https://arxiv.org/pdf/2601.08536)</summary>

**Abstract:** Deep Research Systems (DRS) aim to help users search the web, synthesize information, and deliver comprehensive investigative reports. However, how to rigorously evaluate these systems remains under-explored. Existing deep-research benchmarks often fall into two failure modes. Some do not adequately test a system's ability to analyze evidence and write coherent reports. Others rely on evaluation criteria that are either overly coarse or directly defined by LLMs (or both), leading to scores that can be biased relative to human experts and are hard to verify or interpret. To address these issues, we introduce Deep Research Bench II, a new benchmark for evaluating DRS-generated reports. It contains 132 grounded research tasks across 22 domains; for each task, a system must produce a long-form research report that is evaluated by a set of 9430 fine-grained binary rubrics in total, covering three dimensions: information recall, analysis, and presentation. All rubrics are derived from carefully selected expert-written investigative articles and are constructed through a four-stage LLM+human pipeline that combines automatic extraction with over 400 human-hours of expert review, ensuring that the criteria are atomic, verifiable, and aligned with human expert judgment. We evaluate several state-of-the-art deep-research systems on Deep Research Bench II and find that even the strongest models satisfy fewer than 50% of the rubrics, revealing a substantial gap between current DRSs and human experts.

**arXiv ID:** 2601.08536
</details>

<details>
<summary><strong>Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models</strong> - Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, Kunle Olukotun - [[pdf]](https://arxiv.org/pdf/2510.04618)</summary>

**Abstract:** Large language model (LLM) applications such as agents and domain-specific reasoning increasingly rely on context adaptation -- modifying inputs with instructions, strategies, or evidence, rather than weight updates. Prior approaches improve usability but often suffer from brevity bias, which drops domain insights for concise summaries, and from context collapse, where iterative rewriting erodes details over time. Building on the adaptive memory introduced by Dynamic Cheatsheet, we introduce ACE (Agentic Context Engineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models. Across agent and domain-specific benchmarks, ACE optimizes contexts both offline (e.g., system prompts) and online (e.g., agent memory), consistently outperforming strong baselines: +10.6% on agents and +8.6% on finance, while significantly reducing adaptation latency and rollout cost. Notably, ACE could adapt effectively without labeled supervision and instead by leveraging natural execution feedback. On the AppWorld leaderboard, ACE matches the top-ranked production-level agent on the overall average and surpasses it on the harder test-challenge split, despite using a smaller open-source model. These results show that comprehensive, evolving contexts enable scalable, efficient, and self-improving LLM systems with low overhead.

**arXiv ID:** 2510.04618
</details>

<details>
<summary><strong>BayesFlow: A Probability Inference Framework for Meta-Agent Assisted Workflow Generation</strong> - Bo Yuan, Yun Zhou, Zhichao Xu, Kiran Ramnath, Aosong Feng, Balasubramaniam Srinivasan - [[pdf]](https://arxiv.org/pdf/2601.22305)</summary>

**Abstract:** Automatic workflow generation is the process of automatically synthesizing sequences of LLM calls, tool invocations, and post-processing steps for complex end-to-end tasks. Most prior methods cast this task as an optimization problem with limited theoretical grounding. We propose to cast workflow generation as Bayesian inference over a posterior distribution on workflows, and introduce \textbf{Bayesian Workflow Generation (BWG)}, a sampling framework that builds workflows step-by-step using parallel look-ahead rollouts for importance weighting and a sequential in-loop refiner for pool-wide improvements. We prove that, without the refiner, the weighted empirical distribution converges to the target posterior. We instantiate BWG as \textbf{BayesFlow}, a training-free algorithm for workflow construction. Across six benchmark datasets, BayesFlow improves accuracy by up to 9 percentage points over SOTA workflow generation baselines and by up to 65 percentage points over zero-shot prompting, establishing BWG as a principled upgrade to search-based workflow design. Code will be available on this https URL.

**arXiv ID:** 2601.22305
</details>

<details>
<summary><strong>TwinBrainVLA: Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers</strong> - Bin Yu, Shijie Lian, Xiaopeng Lin, Yuliang Wei, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Xinming Wang, Bailing Wang, Cong Huang, Kai Chen - [[pdf]](https://arxiv.org/pdf/2601.14133)</summary>

**Abstract:** The fundamental premise of Vision-Language-Action (VLA) models is to harness the extensive general capabilities of pre-trained Vision-Language Models (VLMs) for generalized embodied intelligence. However, standard robotic fine-tuning inevitably disrupts the pre-trained feature space, leading to "catastrophic forgetting" that compromises the general visual understanding we aim to leverage. To effectively utilize the uncorrupted general capabilities of VLMs for robotic tasks, we propose TwinBrainVLA, which coordinates two isomorphic VLM pathways: a frozen generalist (also called "Left Brain") and a trainable specialist (also called "Right Brain"). Our architecture utilizes a Asymmetric Mixture-of-Transformers (AsyMoT) mechanism, enabling the Right Brain to dynamically query and fuse intact semantic knowledge from the Left Brain with proprioceptive states. This fused representation conditions a flow-matching action expert for precise continuous control. Empirical results on SimplerEnv and RoboCasa benchmarks demonstrate that by explicitly retaining general capabilities, TwinBrainVLA achieves substantial performance gains over baseline models in complex manipulation tasks.

**arXiv ID:** 2601.14133
</details>

</details>

<details open>
<summary><h2>LLM Agents (8 papers)</h2></summary>

<details>
<summary><strong>Why Reasoning Fails to Plan: A Planning-Centric Analysis of Long-Horizon Decision Making in LLM Agents</strong> - Zehong Wang, Fang Wu, Hongru Wang, Xiangru Tang, Bolian Li, Zhenfei Yin, Yijun Ma, Yiyang Li, Weixiang Sun, Xiusi Chen, Yanfang Ye - [[pdf]](https://arxiv.org/pdf/2601.22311)</summary>

**Abstract:** Large language model (LLM)-based agents exhibit strong step-by-step reasoning capabilities over short horizons, yet often fail to sustain coherent behavior over long planning horizons. We argue that this failure reflects a fundamental mismatch: step-wise reasoning induces a form of step-wise greedy policy that is adequate for short horizons but fails in long-horizon planning, where early actions must account for delayed consequences. From this planning-centric perspective, we study LLM-based agents in deterministic, fully structured environments with explicit state transitions and evaluation signals. Our analysis reveals a core failure mode of reasoning-based policies: locally optimal choices induced by step-wise scoring lead to early myopic commitments that are systematically amplified over time and difficult to recover from. We introduce FLARE (Future-aware Lookahead with Reward Estimation) as a minimal instantiation of future-aware planning to enforce explicit lookahead, value propagation, and limited commitment in a single model, allowing downstream outcomes to influence early decisions. Across multiple benchmarks, agent frameworks, and LLM backbones, FLARE consistently improves task performance and planning-level behavior, frequently allowing LLaMA-8B with FLARE to outperform GPT-4o with standard step-by-step reasoning. These results establish a clear distinction between reasoning and planning.

**arXiv ID:** 2601.22311
</details>

<details>
<summary><strong>AutoRefine: From Trajectories to Reusable Expertise for Continual LLM Agent Refinement</strong> - Libin Qiu, Zhirong Gao, Junfu Chen, Yuhang Ye, Weizhi Huang, Xiaobo Xue, Wenkai Qiu, Shuo Tang - [[pdf]](https://arxiv.org/pdf/2601.22758)</summary>

**Abstract:** Large language model agents often fail to accumulate knowledge from experience, treating each task as an independent challenge. Recent methods extract experience as flattened textual knowledge, which cannot capture procedural logic of complex subtasks. They also lack maintenance mechanisms, causing repository degradation as experience accumulates. We introduce AutoRefine, a framework that extracts and maintains dual-form Experience Patterns from agent execution histories. For procedural subtasks, we extract specialized subagents with independent reasoning and memory. For static knowledge, we extract skill patterns as guidelines or code snippets. A continuous maintenance mechanism scores, prunes, and merges patterns to prevent repository degradation. Evaluated on ALFWorld, ScienceWorld, and TravelPlanner, AutoRefine achieves 98.4%, 70.4%, and 27.1% respectively, with 20-73% step reductions. On TravelPlanner, automatic extraction exceeds manually designed systems (27.1% vs 12.1%), demonstrating its ability to capture procedural coordination.

**arXiv ID:** 2601.22758
</details>

<details>
<summary><strong>Recoverability Has a Law: The ERR Measure for Tool-Augmented Agents</strong> - Sri Vatsa Vuddanti, Satwik Kumar Chittiprolu - [[pdf]](https://arxiv.org/pdf/2601.22352)</summary>

**Abstract:** Language model agents often appear capable of self-recovery after failing tool call executions, yet this behavior lacks a formal explanation. We present a predictive theory that resolves this gap by showing that recoverability follows a measurable law. To elaborate, we formalize recoverability through Expected Recovery Regret (ERR), which quantifies the deviation of a recovery policy from the optimal one under stochastic execution noise, and derive a first-order relationship between ERR and an empirical observable quantity, the Efficiency Score (ES). This yields a falsifiable first-order quantitative law of recovery dynamics in tool-using agents. We empirically validate the law across five tool-use benchmarks spanning controlled perturbations, diagnostic reasoning, and real-world APIs. Across model scales, perturbation regimes, and recovery horizons, predicted regret under the ERR-ES law closely matched observed post-failure regret measured from Monte Carlo rollouts, within delta less than or equal to 0.05. Our results reveal that recoverability is not an artifact of model scale or architecture, but a governed property of interaction dynamics, providing a theoretical foundation for execution-level robustness in language agents.

**arXiv ID:** 2601.22352
</details>

<details>
<summary><strong>Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering</strong> - Sanidhya Vijayvargiya, Xuhui Zhou, Akhila Yerukola, Maarten Sap, Graham Neubig - [[pdf]](https://arxiv.org/pdf/2502.13069)</summary>

**Abstract:** AI agents are increasingly being deployed to automate tasks, often based on underspecified user instructions. Making unwarranted assumptions to compensate for the missing information and failing to ask clarifying questions can lead to suboptimal outcomes, safety risks due to tool misuse, and wasted computational resources. In this work, we study the ability of LLM agents to handle underspecified instructions in interactive code generation settings by evaluating proprietary and open-weight models on their performance across three key steps: (a) detecting underspecificity, (b) asking targeted clarification questions, and (c) leveraging the interaction to improve performance in underspecified scenarios. We introduce Ambig-SWE, an underspecified variant of SWE-Bench Verified, specifically designed to evaluate agent behavior under ambiguity and interaction. Our findings reveal that models struggle to distinguish between well-specified and underspecified instructions. However, when models interact for underspecified inputs, they effectively obtain vital information from the user leading to significant improvements in performance, up to 74% over the non-interactive settings, underscoring the value of effective interaction. Our study highlights critical gaps in how current state-of-the-art models handle missing information in complex software engineering tasks and structures the evaluation into distinct steps to enable targeted improvements.

**arXiv ID:** 2502.13069
</details>

<details>
<summary><strong>Debating Truth: Debate-driven Claim Verification with Multiple Large Language Model Agents</strong> - Haorui He, Yupeng Li, Dacheng Wen, Yang Chen, Reynold Cheng, Donglong Chen, Francis C. M. Lau - [[pdf]](https://arxiv.org/pdf/2507.19090)</summary>

**Abstract:** State-of-the-art single-agent claim verification methods struggle with complex claims that require nuanced analysis of multifaceted evidence. Inspired by real-world professional fact-checkers, we propose \textbf{DebateCV}, the first debate-driven claim verification framework powered by multiple LLM agents. In DebateCV, two \textit{Debaters} argue opposing stances to surface subtle errors in single-agent assessments. A decisive \textit{Moderator} is then required to weigh the evidential strength of conflicting arguments to deliver an accurate verdict. Yet, zero-shot Moderators are biased toward neutral judgments, and no datasets exist for training them. To bridge this gap, we propose \textbf{Debate-SFT}, a post-training framework that leverages synthetic data to enhance agents' ability to effectively adjudicate debates for claim verification. Results show that our methods surpass state-of-the-art non-debate approaches in both accuracy (across various evidence conditions) and justification quality.

**arXiv ID:** 2507.19090
</details>

<details>
<summary><strong>ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents</strong> - Hwan Chang, Yonghyun Jun, Hwanhee Lee - [[pdf]](https://arxiv.org/pdf/2509.22830)</summary>

**Abstract:** The growing deployment of large language model (LLM) based agents that interact with external environments has created new attack surfaces for adversarial manipulation. One major threat is indirect prompt injection, where attackers embed malicious instructions in external environment output, causing agents to interpret and execute them as if they were legitimate prompts. While previous research has focused primarily on plain-text injection attacks, we find a significant yet underexplored vulnerability: LLMs' dependence on structured chat templates and their susceptibility to contextual manipulation through persuasive multi-turn dialogues. To this end, we introduce ChatInject, an attack that formats malicious payloads to mimic native chat templates, thereby exploiting the model's inherent instruction-following tendencies. Building on this foundation, we develop a persuasion-driven Multi-turn variant that primes the agent across conversational turns to accept and execute otherwise suspicious actions. Through comprehensive experiments across frontier LLMs, we demonstrate three critical findings: (1) ChatInject achieves significantly higher average attack success rates than traditional prompt injection methods, improving from 5.18% to 32.05% on AgentDojo and from 15.13% to 45.90% on InjecAgent, with multi-turn dialogues showing particularly strong performance at average 52.33% success rate on InjecAgent, (2) chat-template-based payloads demonstrate strong transferability across models and remain effective even against closed-source LLMs, despite their unknown template structures, and (3) existing prompt-based defenses are largely ineffective against this attack approach, especially against Multi-turn variants. These findings highlight vulnerabilities in current agent systems.

**arXiv ID:** 2509.22830
</details>

<details>
<summary><strong>AgentIF-OneDay: A Task-level Instruction-Following Benchmark for General AI Agents in Daily Scenarios</strong> - Kaiyuan Chen, Qimin Wu, Taiyu Hou, Tianhao Tang, Xueyu Hu, Yuchen Hou, Bikun Li, Chengming Qian, Guoyin Wang, Haolin Chen, Haotong Tian, Haoye Zhang, Haoyu Bian, Hongbing Pan, Hongkang Zhang, Hongyi Zhou, Jiaqi Cai, Jiewu Rao, Jiyuan Ren, Keduan Huang, Lucia Zhu Huang, Mingyu Yuan, Naixu Guo, Qicheng Tang, Qinyan Zhang, Shuai Chen, Siheng Chen, Ting Ting Li, Xiaoxing Guo, Yaocheng Zuo, Yaoqi Guo, Yinan Wang, Yinzhou Yu, Yize Wang, Yuan Jiang, Yuan Tian, Yuanshuo Zhang, Yuxuan Liu, Yvette Yan Zeng, Zenyu Shan, Zihan Yin, Xiaobo Hu, Yang Liu, Yixin Ren, Yuan Gong - [[pdf]](https://arxiv.org/pdf/2601.20613)</summary>

**Abstract:** The capacity of AI agents to effectively handle tasks of increasing duration and complexity continues to grow, demonstrating exceptional performance in coding, deep research, and complex problem-solving evaluations. However, in daily scenarios, the perception of these advanced AI capabilities among general users remains limited. We argue that current evaluations prioritize increasing task difficulty without sufficiently addressing the diversity of agentic tasks necessary to cover the daily work, life, and learning activities of a broad demographic. To address this, we propose AgentIF-OneDay, aimed at determining whether general users can utilize natural language instructions and AI agents to complete a diverse array of daily tasks. These tasks require not only solving problems through dialogue but also understanding various attachment types and delivering tangible file-based results. The benchmark is structured around three user-centric categories: Open Workflow Execution, which assesses adherence to explicit and complex workflows; Latent Instruction, which requires agents to infer implicit instructions from attachments; and Iterative Refinement, which involves modifying or expanding upon ongoing work. We employ instance-level rubrics and a refined evaluation pipeline that aligns LLM-based verification with human judgment, achieving an 80.1% agreement rate using Gemini-3-Pro. AgentIF-OneDay comprises 104 tasks covering 767 scoring points. We benchmarked four leading general AI agents and found that agent products built based on APIs and ChatGPT agents based on agent RL remain in the first tier simultaneously. Leading LLM APIs and open-source models have internalized agentic capabilities, enabling AI application teams to develop cutting-edge Agent products.

**arXiv ID:** 2601.20613
</details>

<details>
<summary><strong>ASTRA: Automated Synthesis of agentic Trajectories and Reinforcement Arenas</strong> - Xiaoyu Tian, Haotian Wang, Shuaiting Chen, Hao Zhou, Kaichi Yu, Yudian Zhang, Jade Ouyang, Junxi Yin, Jiong Chen, Baoyan Guo, Lei Zhang, Junjie Tao, Yuansheng Song, Ming Cui, Chengwei Liu - [[pdf]](https://arxiv.org/pdf/2601.21558)</summary>

**Abstract:** Large language models (LLMs) are increasingly used as tool-augmented agents for multi-step decision making, yet training robust tool-using agents remains challenging. Existing methods still require manual intervention, depend on non-verifiable simulated environments, rely exclusively on either supervised fine-tuning (SFT) or reinforcement learning (RL), and struggle with stable long-horizon, multi-turn learning. To address these challenges, we introduce ASTRA, a fully automated end-to-end framework for training tool-augmented language model agents via scalable data synthesis and verifiable reinforcement learning. ASTRA integrates two complementary components. First, a pipeline that leverages the static topology of tool-call graphs synthesizes diverse, structurally grounded trajectories, instilling broad and transferable tool-use competence. Second, an environment synthesis framework that captures the rich, compositional topology of human semantic reasoning converts decomposed question-answer traces into independent, code-executable, and rule-verifiable environments, enabling deterministic multi-turn RL. Based on this method, we develop a unified training methodology that integrates SFT with online RL using trajectory-level rewards to balance task completion and interaction efficiency. Experiments on multiple agentic tool-use benchmarks demonstrate that ASTRA-trained models achieve state-of-the-art performance at comparable scales, approaching closed-source systems while preserving core reasoning ability. We release the full pipelines, environments, and trained models at this https URL.

**arXiv ID:** 2601.21558
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (24 papers)</h2></summary>

<details>
<summary><strong>From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents</strong> - Jiaxuan Gao, Jiaao Chen, Chuyi He, Wei-Chen Wang, Shusheng Xu, Hanrui Wang, Di Jin, Yi Wu - [[pdf]](https://arxiv.org/pdf/2601.22607)</summary>

**Abstract:** Interactive tool-using agents must solve real-world tasks via multi-turn interaction with both humans and external environments, requiring dialogue state tracking, multi-step tool execution, while following complex instructions. Post-training such agents is challenging because synthesis for high-quality multi-turn tool-use data is difficult to scale, and reinforcement learning (RL) could face noisy signals caused by user simulation, leading to degraded training efficiency. We propose a unified framework that combines a self-evolving data agent with verifier-based RL. Our system, EigenData, is a hierarchical multi-agent engine that synthesizes tool-grounded dialogues together with executable per-instance checkers, and improves generation reliability via closed-loop self-evolving process that updates prompts and workflow. Building on the synthetic data, we develop an RL recipe that first fine-tunes the user model and then applies GRPO-style training with trajectory-level group-relative advantages and dynamic filtering, yielding consistent improvements beyond SFT. Evaluated on tau^2-bench, our best model reaches 73.0% pass^1 on Airline and 98.3% pass^1 on Telecom, matching or exceeding frontier models. Overall, our results suggest a scalable pathway for bootstrapping complex tool-using behaviors without expensive human annotation.

**arXiv ID:** 2601.22607
</details>

<details>
<summary><strong>SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assembly</strong> - Wei Zhu, Zhiwen Tang, Kun Yue - [[pdf]](https://arxiv.org/pdf/2601.22623)</summary>

**Abstract:** Recent advancements have increasingly focused on leveraging large language models (LLMs) to construct autonomous agents for complex problem-solving tasks. However, existing approaches predominantly employ a single-agent framework to generate search branches and estimate rewards during Monte Carlo Tree Search (MCTS) planning. This single-agent paradigm inherently limits exploration capabilities, often resulting in insufficient diversity among generated branches and suboptimal planning performance. To overcome these limitations, we propose Synergistic Multi-agent Planning with Heterogeneous langauge model assembly (SYMPHONY), a novel multi-agent planning framework that integrates a pool of heterogeneous language model-based agents. By leveraging diverse reasoning patterns across agents, SYMPHONY enhances rollout diversity and facilitates more effective exploration. Empirical results across multiple benchmark tasks show that SYMPHONY achieves strong performance even when instantiated with open-source LLMs deployable on consumer-grade hardware. When enhanced with cloud-based LLMs accessible via API, SYMPHONY demonstrates further improvements, outperforming existing state-of-the-art baselines and underscoring the effectiveness of heterogeneous multi-agent coordination in planning tasks.

**arXiv ID:** 2601.22623
</details>

<details>
<summary><strong>Learning with Challenges: Adaptive Difficulty-Aware Data Generation for Mobile GUI Agent Training</strong> - Linjia Kang, Zhimin Wang, Yongkang Zhang, Duo Wu, Jinghe Wang, Ming Ma, Haopeng Yan, Zhi Wang - [[pdf]](https://arxiv.org/pdf/2601.22781)</summary>

**Abstract:** Large-scale, high-quality interaction trajectories are essential for advancing mobile Graphical User Interface (GUI) agents. While existing methods typically rely on labor-intensive human demonstrations or automated model exploration to generate GUI trajectories, they lack fine-grained control over task difficulty. This fundamentally restricts learning effectiveness due to the mismatch between the training difficulty and the agent's capabilities. Inspired by how humans acquire skills through progressively challenging tasks, we propose MobileGen, a novel data generation framework that adaptively aligns training difficulty with the GUI agent's capability frontier. Specifically, MobileGen explicitly decouples task difficulty into structural (e.g., trajectory length) and semantic (e.g., task goal) dimensions. It then iteratively evaluates the agent on a curated prior dataset to construct a systematic profile of its capability frontier across these two dimensions. With this profile, the probability distribution of task difficulty is adaptively computed, from which the target difficulty for the next round of training can be sampled. Guided by the sampled difficulty, a multi-agent controllable generator is finally used to synthesize high-quality interaction trajectories along with corresponding task instructions. Extensive experiments show that MobileGen consistently outperforms existing data generation methods by improving the average performance of GUI agents by 1.57 times across multiple challenging benchmarks. This highlights the importance of capability-aligned data generation for effective mobile GUI agent training.

**arXiv ID:** 2601.22781
</details>

<details>
<summary><strong>Stablecoin Design with Adversarial-Robust Multi-Agent Systems via Trust-Weighted Signal Aggregation</strong> - Shengwei You, Aditya Joshi, Andrey Kuehlkamp, Jarek Nabrzyski - [[pdf]](https://arxiv.org/pdf/2601.22168)</summary>

**Abstract:** Algorithmic stablecoins promise decentralized monetary stability by maintaining a target peg through programmatic reserve management. Yet, their reserve controllers remain vulnerable to regime-blind optimization, calibrating risk parameters on fair-weather data while ignoring tail events that precipitate cascading failures. The March 2020 Black Thursday collapse, wherein MakerDAO's collateral auctions yielded $8.3M in losses and a 15% peg deviation, exposed a critical gap: existing models like SAS systematically omit extreme volatility regimes from covariance estimates, producing allocations optimal in expectation but catastrophic under adversarial stress.
We present MVF-Composer, a trust-weighted Mean-Variance Frontier reserve controller incorporating a novel Stress Harness for risk-state estimation. Our key insight is deploying multi-agent simulations as adversarial stress-testers: heterogeneous agents (traders, liquidity providers, attackers) execute protocol actions under crisis scenarios, exposing reserve vulnerabilities before they manifest on-chain. We formalize a trust-scoring mechanism T: A -> [0,1] that down-weights signals from agents exhibiting manipulative behavior, ensuring the risk-state estimator remains robust to signal injection and Sybil attacks.
Across 1,200 randomized scenarios with injected Black-Swan shocks (10% collateral drawdown, 50% sentiment collapse, coordinated redemption attacks), MVF-Composer reduces peak peg deviation by 57% and mean recovery time by 3.1x relative to SAS baselines. Ablation studies confirm the trust layer accounts for 23% of stability gains under adversarial conditions, achieving 72% adversarial agent detection. Our system runs on commodity hardware, requires no on-chain oracles beyond standard price feeds, and provides a reproducible framework for stress-testing DeFi reserve policies.

**arXiv ID:** 2601.22168
</details>

<details>
<summary><strong>Learning to Recommend Multi-Agent Subgraphs from Calling Trees</strong> - Xinyuan Song, Liang Zhao - [[pdf]](https://arxiv.org/pdf/2601.22209)</summary>

**Abstract:** Multi-agent systems (MAS) increasingly solve complex tasks by orchestrating agents and tools selected from rapidly growing marketplaces. As these marketplaces expand, many candidates become functionally overlapping, making selection not just a retrieval problem: beyond filtering relevant agents, an orchestrator must choose options that are reliable, compatible with the current execution context, and able to cooperate with other selected agents. Existing recommender systems -- largely built for item-level ranking from flat user-item logs -- do not directly address the structured, sequential, and interaction-dependent nature of agent orchestration. We address this gap by \textbf{formulating agent recommendation in MAS as a constrained decision problem} and introducing a generic \textbf{constrained recommendation framework} that first uses retrieval to build a compact candidate set conditioned on the current subtask and context, and then performs \textbf{utility optimization} within this feasible set using a learned scorer that accounts for relevance, reliability, and interaction effects. We ground both the formulation and learning signals in \textbf{historical calling trees}, which capture the execution structure of MAS (parent-child calls, branching dependencies, and local cooperation patterns) beyond what flat logs provide. The framework supports two complementary settings: \textbf{agent-level recommendation} (select the next agent/tool) and \textbf{system-level recommendation} (select a small, connected agent team/subgraph for coordinated execution). To enable systematic evaluation, we construct a unified calling-tree benchmark by normalizing invocation logs from eight heterogeneous multi-agent corpora into a shared structured representation.

**arXiv ID:** 2601.22209
</details>

<details>
<summary><strong>MERMAID: Memory-Enhanced Retrieval and Reasoning with Multi-Agent Iterative Knowledge Grounding for Veracity Assessment</strong> - Yupeng Cao, Chengyang He, Yangyang Yu, Ping Wang, K.P. Subbalakshmi - [[pdf]](https://arxiv.org/pdf/2601.22361)</summary>

**Abstract:** Assessing the veracity of online content has become increasingly critical. Large language models (LLMs) have recently enabled substantial progress in automated veracity assessment, including automated fact-checking and claim verification systems. Typical veracity assessment pipelines break down complex claims into sub-claims, retrieve external evidence, and then apply LLM reasoning to assess veracity. However, existing methods often treat evidence retrieval as a static, isolated step and do not effectively manage or reuse retrieved evidence across claims. In this work, we propose MERMAID, a memory-enhanced multi-agent veracity assessment framework that tightly couples the retrieval and reasoning processes. MERMAID integrates agent-driven search, structured knowledge representations, and a persistent memory module within a Reason-Action style iterative process, enabling dynamic evidence acquisition and cross-claim evidence reuse. By retaining retrieved evidence in an evidence memory, the framework reduces redundant searches and improves verification efficiency and consistency. We evaluate MERMAID on three fact-checking benchmarks and two claim-verification datasets using multiple LLMs, including GPT, LLaMA, and Qwen families. Experimental results show that MERMAID achieves state-of-the-art performance while improving the search efficiency, demonstrating the effectiveness of synergizing retrieval, reasoning, and memory for reliable veracity assessment.

**arXiv ID:** 2601.22361
</details>

<details>
<summary><strong>ScholarPeer: A Context-Aware Multi-Agent Framework for Automated Peer Review</strong> - Palash Goyal, Mihir Parmar, Yiwen Song, Hamid Palangi, Tomas Pfister, Jinsung Yoon - [[pdf]](https://arxiv.org/pdf/2601.22638)</summary>

**Abstract:** Automated peer review has evolved from simple text classification to structured feedback generation. However, current state-of-the-art systems still struggle with "surface-level" critiques: they excel at summarizing content but often fail to accurately assess novelty and significance or identify deep methodological flaws because they evaluate papers in a vacuum, lacking the external context a human expert possesses. In this paper, we introduce ScholarPeer, a search-enabled multi-agent framework designed to emulate the cognitive processes of a senior researcher. ScholarPeer employs a dual-stream process of context acquisition and active verification. It dynamically constructs a domain narrative using a historian agent, identifies missing comparisons via a baseline scout, and verifies claims through a multi-aspect Q&A engine, grounding the critique in live web-scale literature. We evaluate ScholarPeer on DeepReview-13K and the results demonstrate that ScholarPeer achieves significant win-rates against state-of-the-art approaches in side-by-side evaluations and reduces the gap to human-level diversity.

**arXiv ID:** 2601.22638
</details>

<details>
<summary><strong>MEnvAgent: Scalable Polyglot Environment Construction for Verifiable Software Engineering</strong> - Chuanzhe Guo, Jingjing Wu, Sijun He, Yang Chen, Zhaoqi Kuang, Shilong Fan, Bingjin Chen, Siqi Bao, Jing Liu, Hua Wu, Qingfu Zhu, Wanxiang Che, Haifeng Wang - [[pdf]](https://arxiv.org/pdf/2601.22859)</summary>

**Abstract:** The evolution of Large Language Model (LLM) agents for software engineering (SWE) is constrained by the scarcity of verifiable datasets, a bottleneck stemming from the complexity of constructing executable environments across diverse languages. To address this, we introduce MEnvAgent, a Multi-language framework for automated Environment construction that facilitates scalable generation of verifiable task instances. MEnvAgent employs a multi-agent Planning-Execution-Verification architecture to autonomously resolve construction failures and integrates a novel Environment Reuse Mechanism that reduces computational overhead by incrementally patching historical environments. Evaluations on MEnvBench, a new benchmark comprising 1,000 tasks across 10 languages, demonstrate that MEnvAgent outperforms baselines, improving Fail-to-Pass (F2P) rates by 8.6% while reducing time costs by 43%. Additionally, we demonstrate the utility of MEnvAgent by constructing MEnvData-SWE, the largest open-source polyglot dataset of realistic verifiable Docker environments to date, alongside solution trajectories that enable consistent performance gains on SWE tasks across a wide range of models. Our code, benchmark, and dataset are available at this https URL.

**arXiv ID:** 2601.22859
</details>

<details>
<summary><strong>MonoScale: Scaling Multi-Agent System with Monotonic Improvement</strong> - Shuai Shao, Yixiang Liu, Bingwei Lu, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2601.23219)</summary>

**Abstract:** In recent years, LLM-based multi-agent systems (MAS) have advanced rapidly, using a router to decompose tasks and delegate subtasks to specialized agents. A natural way to expand capability is to scale up the agent pool by continually integrating new functional agents or tool interfaces, but naive expansion can trigger performance collapse when the router cold-starts on newly added, heterogeneous, and unreliable agents. We propose MonoScale, an expansion-aware update framework that proactively generates a small set of agent-conditioned familiarization tasks, harvests evidence from both successful and failed interactions, and distills it into auditable natural-language memory to guide future routing. We formalize sequential augmentation as a contextual bandit and perform trust-region memory updates, yielding a monotonic non-decreasing performance guarantee across onboarding rounds. Experiments on GAIA and Humanity's Last Exam show stable gains as the agent pool grows, outperforming naive scale-up and strong-router fixed-pool baselines.

**arXiv ID:** 2601.23219
</details>

<details>
<summary><strong>Self-Improvement of Language Models by Post-Training on Multi-Agent Debate</strong> - Ankur Samanta, Akshayaa Magesh, Runzhe Wu, Ayush Jain, Youliang Yu, Daniel Jiang, Boris Vidolov, Paul Sajda, Yonathan Efroni, Kaveh Hassani - [[pdf]](https://arxiv.org/pdf/2509.15172)</summary>

**Abstract:** Self-improvement, where models improve beyond their current performance without external supervision, remains a challenge. The core difficulty is sourcing a training signal stronger than what the model itself can currently produce. Majority voting has been shown to provide such a signal by aggregating over multiple samples, helping mitigate some of the inconsistencies in LM reasoning. In this work, we show that multi-agent debate--where models collaborate and exchange reasoning over multiple rounds--provides an even richer signal than single-round majority voting. We introduce Multi-Agent Consensus Alignment (MACA), which uses reinforcement learning (RL) to post-train models to effectively utilize multi-agent debate. We find that preference learning over full reasoning traces, learning to differentiate between majority and minority reasoning, is more effective than binary consensus rewards or SFT-based approaches for leveraging these debate signals. This produces three key improvements: models are (1) better at utilizing the multi-agent debate setting (+26.87% on MATH), (2) individually more accurate (+21.51% on MathQA), and (3) more self-consistent (+27.6% on GSM8K). We also see strong generalization to unseen benchmarks (+16.3% on GPQA, +11.6% on CommonsenseQA).

**arXiv ID:** 2509.15172
</details>

<details>
<summary><strong>Collaborative Belief Reasoning with LLMs for Efficient Multi-Agent Collaboration</strong> - Zhimin Wang, Duo Wu, Shaokang He, Jinghe Wang, Linjia Kang, Jing Yu, Kai Zhu, Jiawei Li, Zhi Wang - [[pdf]](https://arxiv.org/pdf/2509.21981)</summary>

**Abstract:** Effective real-world multi-agent collaboration requires not only accurate planning but also the ability to reason about collaborators' intents--a crucial capability for avoiding miscoordination and redundant communication under partial observable environments. Due to their strong planning and reasoning capabilities, large language models (LLMs) have emerged as promising autonomous agents for collaborative task solving. However, existing collaboration frameworks for LLMs overlook their reasoning potential for dynamic intent inference, and thus produce inconsistent plans and redundant communication, reducing collaboration efficiency. To bridge this gap, we propose CoBel-World, a novel framework that equips LLM agents with a Collaborative Belief World--an internal representation jointly modeling the physical environment and collaborators' mental states. CoBel-World enables agents to parse external open-world knowledge into structured beliefs via a symbolic belief representation module, and perform zero-shot Bayesian-style belief updates through LLM reasoning. This allows agents to proactively detect potential miscoordination (e.g., conflicting plans) and communicate adaptively. Evaluated on challenging embodied benchmarks (i.e., TDW-MAT and C-WAH), CoBel-World significantly reduces communication costs by 64-79% and improves task completion efficiency by 4-28% compared to the strongest baseline. Our results show that explicit, intent-aware belief modeling is essential for efficient and human-like collaboration in LLM-based multi-agent systems.

**arXiv ID:** 2509.21981
</details>

<details>
<summary><strong>Learning Reward Functions for Cooperative Resilience in Multi-Agent Systems</strong> - Manuela Chacon-Chamorro, Luis Felipe Giraldo, Nicanor Quijano - [[pdf]](https://arxiv.org/pdf/2601.22292)</summary>

**Abstract:** Multi-agent systems often operate in dynamic and uncertain environments, where agents must not only pursue individual goals but also safeguard collective functionality. This challenge is especially acute in mixed-motive multi-agent systems. This work focuses on cooperative resilience, the ability of agents to anticipate, resist, recover, and transform in the face of disruptions, a critical yet underexplored property in Multi-Agent Reinforcement Learning. We study how reward function design influences resilience in mixed-motive settings and introduce a novel framework that learns reward functions from ranked trajectories, guided by a cooperative resilience metric. Agents are trained in a suite of social dilemma environments using three reward strategies: i) traditional individual reward; ii) resilience-inferred reward; and iii) hybrid that balance both. We explore three reward parameterizations-linear models, hand-crafted features, and neural networks, and employ two preference-based learning algorithms to infer rewards from behavioral rankings. Our results demonstrate that hybrid strategy significantly improve robustness under disruptions without degrading task performance and reduce catastrophic outcomes like resource overuse. These findings underscore the importance of reward design in fostering resilient cooperation, and represent a step toward developing robust multi-agent systems capable of sustaining cooperation in uncertain environments.

**arXiv ID:** 2601.22292
</details>

<details>
<summary><strong>Multi-Agent Systems Should be Treated as Principal-Agent Problems</strong> - Paulius Rauba, Simonas Cepenas, Mihaela van der Schaar - [[pdf]](https://arxiv.org/pdf/2601.23211)</summary>

**Abstract:** Consider a multi-agent systems setup in which a principal (a supervisor agent) assigns subtasks to specialized agents and aggregates their responses into a single system-level output. A core property of such systems is information asymmetry: agents observe task-specific information, produce intermediate reasoning traces, and operate with different context windows. In isolation, such asymmetry is not problematic, since agents report truthfully to the principal when incentives are fully aligned. However, this assumption breaks down when incentives diverge. Recent evidence suggests that LLM-based agents can acquire their own goals, such as survival or self-preservation, a phenomenon known as scheming, and may deceive humans or other agents. This leads to agency loss: a gap between the principal's intended outcome and the realized system behavior. Drawing on core ideas from microeconomic theory, we argue that these characteristics, information asymmetry and misaligned goals, are best studied through the lens of principal-agent problems. We explain why multi-agent systems, both human-to-LLM and LLM-to-LLM, naturally induce information asymmetry under this formulation, and we use scheming, where LLM agents pursue covert goals, as a concrete case study. We show that recently introduced terminology used to describe scheming, such as covert subversion or deferred subversion, corresponds to well-studied concepts in the mechanism design literature, which not only characterizes the problem but also prescribes concrete mitigation strategies. More broadly, we argue for applying tools developed to study human agent behavior to the analysis of non-human agents.

**arXiv ID:** 2601.23211
</details>

<details>
<summary><strong>Specialists or Generalists? Multi-Agent and Single-Agent LLMs for Essay Grading</strong> - Jamiu Adekunle Idowu, Ahmed Almasoud - [[pdf]](https://arxiv.org/pdf/2601.22386)</summary>

**Abstract:** Automated essay scoring (AES) systems increasingly rely on large language models, yet little is known about how architectural choices shape their performance across different essay quality levels. This paper evaluates single-agent and multi-agent LLM architectures for essay grading using the ASAP 2.0 corpus. Our multi-agent system decomposes grading into three specialist agents (Content, Structure, Language) coordinated by a Chairman Agent that implements rubric-aligned logic including veto rules and score capping. We test both architectures in zero-shot and few-shot conditions using GPT-5.1. Results show that the multi-agent system is significantly better at identifying weak essays while the single-agent system performs better on mid-range essays. Both architectures struggle with high-quality essays. Critically, few-shot calibration emerges as the dominant factor in system performance -- providing just two examples per score level improves QWK by approximately 26% for both architectures. These findings suggest architectural choice should align with specific deployment priorities, with multi-agent AI particularly suited for diagnostic screening of at-risk students, while single-agent models provide a cost-effective solution for general assessment.

**arXiv ID:** 2601.22386
</details>

<details>
<summary><strong>Emergent Coordination in Multi-Agent Systems via Pressure Fields and Temporal Decay</strong> - Roland Rodriguez - [[pdf]](https://arxiv.org/pdf/2601.08129)</summary>

**Abstract:** Current multi-agent LLM frameworks rely on explicit orchestration patterns borrowed from human organizational structures: planners delegate to executors, managers coordinate workers, and hierarchical control flow governs agent interactions. These approaches suffer from coordination overhead that scales poorly with agent count and task complexity. We propose a fundamentally different paradigm inspired by natural coordination mechanisms: agents operate locally on a shared artifact, guided only by pressure gradients derived from measurable quality signals, with temporal decay preventing premature convergence. We formalize this as optimization over a pressure landscape and prove convergence guarantees under mild conditions. Empirically, on meeting room scheduling across 1,350 trials, pressure-field coordination outperforms all baselines: 48.5% aggregate solve rate versus 12.6% for conversation-based coordination, 1.5% for hierarchical control, and 0.4% for sequential and random baselines (all pairwise comparisons p < 0.001). Temporal decay is essential: disabling it reduces solve rate by 10 percentage points. On easy problems, pressure-field achieves 86.7% solve rate. The approach maintains consistent performance from 1 to 4 agents. Implicit coordination through shared pressure gradients outperforms explicit hierarchical control, suggesting that constraint-driven emergence offers a simpler and more effective foundation for multi-agent AI.

**arXiv ID:** 2601.08129
</details>

<details>
<summary><strong>Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs</strong> - Yujie Zhao, Lanxiang Hu, Yang Wang, Minmin Hou, Hao Zhang, Ke Ding, Jishen Zhao - [[pdf]](https://arxiv.org/pdf/2510.11062)</summary>

**Abstract:** Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models.
We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: this https URL.

**arXiv ID:** 2510.11062
</details>

<details>
<summary><strong>Multi-agent Adaptive Mechanism Design</strong> - Qiushi Han, David Simchi-Levi, Renfei Tan, Zishuo Zhao - [[pdf]](https://arxiv.org/pdf/2512.21794)</summary>

**Abstract:** We study a sequential mechanism design problem in which a principal seeks to elicit truthful reports from multiple rational agents while starting with no prior knowledge of agents' beliefs. We introduce Distributionally Robust Adaptive Mechanism (DRAM), a general framework combining insights from both mechanism design and online learning to jointly address truthfulness and cost-optimality. Throughout the sequential game, the mechanism estimates agents' beliefs and iteratively updates a distributionally robust linear program with shrinking ambiguity sets to reduce payments while preserving truthfulness. Our mechanism guarantees truthful reporting with high probability while achieving $\tilde{O}(\sqrt{T})$ cumulative regret, and we establish a matching lower bound showing that no truthful adaptive mechanism can asymptotically do better. The framework generalizes to plug-in estimators, supporting structured priors and delayed feedback. To our knowledge, this is the first adaptive mechanism under general settings that maintains truthfulness and achieves optimal regret when incentive constraints are unknown and must be learned.

**arXiv ID:** 2512.21794
</details>

<details>
<summary><strong>Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning</strong> - Wonduk Seo, Wonseok Choi, Junseo Koh, Juhyeon Lee, Hyunjin An, Minhyeong Yu, Jian Park, Qingshan Zhou, Seunghyun Lee, Yi Bu - [[pdf]](https://arxiv.org/pdf/2601.21700)</summary>

**Abstract:** Large Language Models (LLMs) increasingly support culturally sensitive decision making, yet often exhibit misalignment due to skewed pretraining data and the absence of structured value representations. Existing methods can steer outputs, but often lack demographic grounding and treat values as independent, unstructured signals, reducing consistency and interpretability. We propose OG-MAR, an Ontology-Guided Multi-Agent Reasoning framework. OG-MAR summarizes respondent-specific values from the World Values Survey (WVS) and constructs a global cultural ontology by eliciting relations over a fixed taxonomy via competency questions. At inference time, it retrieves ontology-consistent relations and demographically similar profiles to instantiate multiple value-persona agents, whose outputs are synthesized by a judgment agent that enforces ontology consistency and demographic proximity. Experiments on regional social-survey benchmarks across four LLM backbones show that OG-MAR improves cultural alignment and robustness over competitive baselines, while producing more transparent reasoning traces.

**arXiv ID:** 2601.21700
</details>

<details>
<summary><strong>Prepare Reasoning Language Models for Multi-Agent Debate with Self-Debate Reinforcement Learning</strong> - Chenxi Liu, Yanshuo Chen, Ruibo Chen, Tianyi Xiong, Tong Zheng, Heng Huang - [[pdf]](https://arxiv.org/pdf/2601.22297)</summary>

**Abstract:** The reasoning abilities of large language models (LLMs) have been substantially improved by reinforcement learning with verifiable rewards (RLVR). At test time, collaborative reasoning through Multi-Agent Debate (MAD) has emerged as a promising approach for enhancing LLM performance. However, current RLVR methods typically train LLMs to solve problems in isolation, without explicitly preparing them to synthesize and benefit from different rationales that arise during debate. In this work, we propose Self-Debate Reinforcement Learning (SDRL), a training framework that equips a single LLM with strong standalone problem-solving ability and the capability to learn from diverse reasoning trajectories in MAD. Given a prompt, SDRL first samples multiple candidate solutions, then constructs a debate context with diverse reasoning paths and generates second-turn responses conditioned on this context. Finally, SDRL jointly optimizes both the initial and debate-conditioned responses, yielding a model that is effective as both a standalone solver and a debate participant. Experiments across multiple base models and reasoning benchmarks show that SDRL improves overall MAD performance while simultaneously strengthening single model reasoning.

**arXiv ID:** 2601.22297
</details>

<details>
<summary><strong>Large Language Model Agents Are Not Always Faithful Self-Evolvers</strong> - Weixiang Zhao, Yingshuo Wang, Yichen Zhang, Yang Deng, Yanyan Zhao, Wanxiang Che, Bing Qin, Ting Liu - [[pdf]](https://arxiv.org/pdf/2601.22436)</summary>

**Abstract:** Self-evolving large language model (LLM) agents continually improve by accumulating and reusing past experience, yet it remains unclear whether they faithfully rely on that experience to guide their behavior. We present the first systematic investigation of experience faithfulness, the causal dependence of an agent's decisions on the experience it is given, in self-evolving LLM agents. Using controlled causal interventions on both raw and condensed forms of experience, we comprehensively evaluate four representative frameworks across 10 LLM backbones and 9 environments. Our analysis uncovers a striking asymmetry: while agents consistently depend on raw experience, they often disregard or misinterpret condensed experience, even when it is the only experience provided. This gap persists across single- and multi-agent configurations and across backbone scales. We trace its underlying causes to three factors: the semantic limitations of condensed content, internal processing biases that suppress experience, and task regimes where pretrained priors already suffice. These findings challenge prevailing assumptions about self-evolving methods and underscore the need for more faithful and reliable approaches to experience integration.

**arXiv ID:** 2601.22436
</details>

<details>
<summary><strong>MiTa: A Hierarchical Multi-Agent Collaboration Framework with Memory-integrated and Task Allocation</strong> - XiaoJie Zhang, JianHan Wu, Xiaoyang Qu, Jianzong Wang - [[pdf]](https://arxiv.org/pdf/2601.22974)</summary>

**Abstract:** Recent advances in large language models (LLMs) have substantially accelerated the development of embodied agents. LLM-based multi-agent systems mitigate the inefficiency of single agents in complex tasks. However, they still suffer from issues such as memory inconsistency and agent behavioral conflicts. To address these challenges, we propose MiTa, a hierarchical memory-integrated task allocative framework to enhance collaborative efficiency. MiTa organizes agents into a manager-member hierarchy, where the manager incorporates additional allocation and summary modules that enable (1) global task allocation and (2) episodic memory integration. The allocation module enables the manager to allocate tasks from a global perspective, thereby avoiding potential inter-agent conflicts. The summary module, triggered by task progress updates, performs episodic memory integration by condensing recent collaboration history into a concise summary that preserves long-horizon context. By combining task allocation with episodic memory, MiTa attains a clearer understanding of the task and facilitates globally consistent task distribution. Experimental results confirm that MiTa achieves superior efficiency and adaptability in complex multi-agent cooperation over strong baseline methods.

**arXiv ID:** 2601.22974
</details>

<details>
<summary><strong>RoboStriker: Hierarchical Decision-Making for Autonomous Humanoid Boxing</strong> - Kangning Yin, Zhe Cao, Wentao Dong, Weishuai Zeng, Tianyi Zhang, Qiang Zhang, Jingbo Wang, Jiangmiao Pang, Ming Zhou, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2601.22517)</summary>

**Abstract:** Achieving human-level competitive intelligence and physical agility in humanoid robots remains a major challenge, particularly in contact-rich and highly dynamic tasks such as boxing. While Multi-Agent Reinforcement Learning (MARL) offers a principled framework for strategic interaction, its direct application to humanoid control is hindered by high-dimensional contact dynamics and the absence of strong physical motion priors. We propose RoboStriker, a hierarchical three-stage framework that enables fully autonomous humanoid boxing by decoupling high-level strategic reasoning from low-level physical execution. The framework first learns a comprehensive repertoire of boxing skills by training a single-agent motion tracker on human motion capture data. These skills are subsequently distilled into a structured latent manifold, regularized by projecting the Gaussian-parameterized distribution onto a unit hypersphere. This topological constraint effectively confines exploration to the subspace of physically plausible motions. In the final stage, we introduce Latent-Space Neural Fictitious Self-Play (LS-NFSP), where competing agents learn competitive tactics by interacting within the latent action space rather than the raw motor space, significantly stabilizing multi-agent training. Experimental results demonstrate that RoboStriker achieves superior competitive performance in simulation and exhibits sim-to-real transfer. Our website is available at RoboStriker.

**arXiv ID:** 2601.22517
</details>

<details>
<summary><strong>Multi-agent Coordination via Flow Matching</strong> - Dongsu Lee, Daehee Lee, Amy Zhang - [[pdf]](https://arxiv.org/pdf/2511.05005)</summary>

**Abstract:** This work presents MAC-Flow, a simple yet expressive framework for multi-agent coordination. We argue that requirements of effective coordination are twofold: (i) a rich representation of the diverse joint behaviors present in offline data and (ii) the ability to act efficiently in real time. However, prior approaches often sacrifice one for the other, i.e., denoising diffusion-based solutions capture complex coordination but are computationally slow, while Gaussian policy-based solutions are fast but brittle in handling multi-agent interaction. MAC-Flow addresses this trade-off by first learning a flow-based representation of joint behaviors, and then distilling it into decentralized one-step policies that preserve coordination while enabling fast execution. Across four different benchmarks, including $12$ environments and $34$ datasets, MAC-Flow alleviates the trade-off between performance and computational cost, specifically achieving about $\boldsymbol{\times14.5}$ faster inference compared to diffusion-based MARL methods, while maintaining good performance. At the same time, its inference speed is similar to that of prior Gaussian policy-based offline multi-agent reinforcement learning (MARL) methods.

**arXiv ID:** 2511.05005
</details>

<details>
<summary><strong>FACET: Multi-Agent AI Supporting Teachers in Scaling Differentiated Learning for Diverse Students</strong> - Jana Gonnermann-Müller, Jennifer Haase, Nicolas Leins, Moritz Igel, Konstantin Fackeldey, Sebastian Pokutta - [[pdf]](https://arxiv.org/pdf/2601.22788)</summary>

**Abstract:** Classrooms are becoming increasingly heterogeneous, comprising learners with diverse performance and motivation levels, language proficiencies, and learning differences such as dyslexia and ADHD. While teachers recognize the need for differentiated instruction, growing workloads create substantial barriers, making differentiated instruction an ideal that is often unrealized in practice. Current AI educational tools, which promise differentiated materials, are predominantly student-facing and performance-centric, ignoring other aspects that shape learning outcomes. We introduce FACET, a teacher-facing multi-agent framework designed to address these gaps by supporting differentiation that accounts for motivation, performance, and learning differences. Developed with educational stakeholders from the outset, the framework coordinates four specialized agents, including learner simulation, diagnostic assessment, material generation, and evaluation within a teacher-in-the-loop design. School principals (N = 30) shaped system requirements through participatory workshops, while in-service K-12 teachers (N = 70) evaluated material quality. Mixed-methods evaluation demonstrates strong perceived value for inclusive differentiation. Practitioners emphasized both the urgent need arising from classroom heterogeneity and the importance of maintaining pedagogical autonomy as a prerequisite for adoption. We discuss implications for future school deployment and outline partnerships for longitudinal classroom implementation.

**arXiv ID:** 2601.22788
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>Scaling Multiagent Systems with Process Rewards</strong> - Ed Li, Junyu Ren, Cat Yan - [[pdf]](https://arxiv.org/pdf/2601.23228)</summary>

**Abstract:** While multiagent systems have shown promise for tackling complex tasks via specialization, finetuning multiple agents simultaneously faces two key challenges: (1) credit assignment across agents, and (2) sample efficiency of expensive multiagent rollouts. In this work, we propose finetuning multiagent systems with per-action process rewards from AI feedback (MAPPA) to address both. Through assigning credit to individual agent actions rather than only at task completion, MAPPA enables fine-grained supervision without ground truth labels while extracting maximal training signal from each rollout. We demonstrate our approach on competition math problems and tool-augmented data analysis tasks. On unseen math problems, MAPPA achieves +5.0--17.5pp on AIME and +7.8--17.2pp on AMC. For data analysis tasks, our method improves success rate by +12.5pp while quality metrics improve by up to 30%, validating that per-action supervision can lead to improvements across different multiagent system on various domains. By addressing these challenges, our work takes a first step toward scaling multiagent systems for complex, long-horizon tasks with minimal human supervision.

**arXiv ID:** 2601.23228
</details>

<details>
<summary><strong>PersonaCite: VoC-Grounded Interviewable Agentic Synthetic AI Personas for Verifiable User and Design Research</strong> - Mario Truss - [[pdf]](https://arxiv.org/pdf/2601.22288)</summary>

**Abstract:** LLM-based and agent-based synthetic personas are increasingly used in design and product decision-making, yet prior work shows that prompt-based personas often produce persuasive but unverifiable responses that obscure their evidentiary basis. We present PersonaCite, an agentic system that reframes AI personas as evidence-bounded research instruments through retrieval-augmented interaction. Unlike prior approaches that rely on prompt-based roleplaying, PersonaCite retrieves actual voice-of-customer artifacts during each conversation turn, constrains responses to retrieved evidence, explicitly abstains when evidence is missing, and provides response-level source attribution. Through semi-structured interviews and deployment study with 14 industry experts, we identify preliminary findings on perceived benefits, validity concerns, and design tensions, and propose Persona Provenance Cards as a documentation pattern for responsible AI persona use in human-centered design workflows.

**arXiv ID:** 2601.22288
</details>

<details>
<summary><strong>Leveraging AI Agents for Autonomous Networks: A Reference Architecture and Empirical Studies</strong> - Binghan Wu, Shoufeng Wang, Yunxin Liu, Ya-Qin Zhang, Joseph Sifakis, Ye Ouyang - [[pdf]](https://arxiv.org/pdf/2509.08312)</summary>

**Abstract:** The evolution toward Level 4 (L4) Autonomous Networks (AN) represents a strategic inflection point in telecommunications, where networks must transcend reactive automation to achieve genuine cognitive capabilities--fulfilling TM Forum's vision of self-configuring, self-healing, and self-optimizing systems that deliver zero-wait, zero-touch, and zero-fault services. This work bridges the gap between architectural theory and operational reality by implementing Joseph Sifakis's AN Agent reference architecture in a functional cognitive system, deploying coordinated proactive-reactive runtimes driven by hybrid knowledge representation. Through an empirical case study of a Radio Access Network (RAN) Link Adaptation (LA) Agent, we validate this framework's transformative potential: demonstrating sub-10 ms real-time control in 5G NR sub-6 GHz while achieving 4% higher downlink throughput than Outer Loop Link Adaptation (OLLA) algorithms and 85% Block Error Rate (BLER) reduction for ultra-reliable services through dynamic Modulation and Coding Scheme (MCS) optimization. These improvements confirm the architecture's viability in overcoming traditional autonomy barriers and advancing critical L4-enabling capabilities toward next-generation objectives.

**arXiv ID:** 2509.08312
</details>

<details>
<summary><strong>AgentScore: Autoformulation of Deployable Clinical Scoring Systems</strong> - Silas Ruhrberg Estévez, Christopher Chiu, Mihaela van der Schaar - [[pdf]](https://arxiv.org/pdf/2601.22324)</summary>

**Abstract:** Modern clinical practice relies on evidence-based guidelines implemented as compact scoring systems composed of a small number of interpretable decision rules. While machine-learning models achieve strong performance, many fail to translate into routine clinical use due to misalignment with workflow constraints such as memorability, auditability, and bedside execution. We argue that this gap arises not from insufficient predictive power, but from optimizing over model classes that are incompatible with guideline deployment. Deployable guidelines often take the form of unit-weighted clinical checklists, formed by thresholding the sum of binary rules, but learning such scores requires searching an exponentially large discrete space of possible rule sets. We introduce AgentScore, which performs semantically guided optimization in this space by using LLMs to propose candidate rules and a deterministic, data-grounded verification-and-selection loop to enforce statistical validity and deployability constraints. Across eight clinical prediction tasks, AgentScore outperforms existing score-generation methods and achieves AUC comparable to more flexible interpretable models despite operating under stronger structural constraints. On two additional externally validated tasks, AgentScore achieves higher discrimination than established guideline-based scores.

**arXiv ID:** 2601.22324
</details>

<details>
<summary><strong>UPA: Unsupervised Prompt Agent via Tree-Based Search and Selection</strong> - Siran Peng, Weisong Zhao, Tianyu Fu, Chenxu Zhao, Tianshuo Zhang, Haoyuan Zhang, Xiangyu Zhu, Minghui Wu, Zhen Lei - [[pdf]](https://arxiv.org/pdf/2601.23273)</summary>

**Abstract:** Prompt agents have recently emerged as a promising paradigm for automated prompt optimization, framing refinement as a sequential decision-making problem over a structured prompt space. While this formulation enables the use of advanced planning algorithms, these methods typically assume access to supervised reward signals, which are often unavailable in practical scenarios. In this work, we propose UPA, an Unsupervised Prompt Agent that realizes structured search and selection without relying on supervised feedback. Specifically, during search, UPA iteratively constructs an evolving tree structure to navigate the prompt space, guided by fine-grained and order-invariant pairwise comparisons from Large Language Models (LLMs). Crucially, as these local comparisons do not inherently yield a consistent global scale, we decouple systematic prompt exploration from final selection, introducing a two-stage framework grounded in the Bradley-Terry-Luce (BTL) model. This framework first performs path-wise Bayesian aggregation of local comparisons to filter candidates under uncertainty, followed by global tournament-style comparisons to infer latent prompt quality and identify the optimal prompt. Experiments across multiple tasks demonstrate that UPA consistently outperforms existing prompt optimization methods, showing that agent-style optimization remains highly effective even in fully unsupervised settings.

**arXiv ID:** 2601.23273
</details>

<details>
<summary><strong>Purely Agentic Black-Box Optimization for Biological Design</strong> - Natalie Maus, Yimeng Zeng, Haydn Thomas Jones, Yining Huang, Gaurav Ng Goel, Alden Rose, Kyurae Kim, Hyun-Su Lee, Marcelo Der Torossian Torres, Fangping Wan, Cesar de la Fuente-Nunez, Mark Yatskar, Osbert Bastani, Jacob R. Gardner - [[pdf]](https://arxiv.org/pdf/2601.22382)</summary>

**Abstract:** Many key challenges in biological design-such as small-molecule drug discovery, antimicrobial peptide development, and protein engineering-can be framed as black-box optimization over vast, complex structured spaces. Existing methods rely mainly on raw structural data and struggle to exploit the rich scientific literature. While large language models (LLMs) have been added to these pipelines, they have been confined to narrow roles within structure-centered optimizers. We instead cast biological black-box optimization as a fully agentic, language-based reasoning process. We introduce Purely Agentic BLack-box Optimization (PABLO), a hierarchical agentic system that uses scientific LLMs pretrained on chemistry and biology literature to generate and iteratively refine biological candidates. On both the standard GuacaMol molecular design and antimicrobial peptide optimization tasks, PABLO achieves state-of-the-art performance, substantially improving sample efficiency and final objective values over established baselines. Compared to prior optimization methods that incorporate LLMs, PABLO achieves competitive token usage per run despite relying on LLMs throughout the optimization loop. Beyond raw performance, the agentic formulation offers key advantages for realistic design: it naturally incorporates semantic task descriptions, retrieval-augmented domain knowledge, and complex constraints. In follow-up in vitro validation, PABLO-optimized peptides showed strong activity against drug-resistant pathogens, underscoring the practical potential of PABLO for therapeutic discovery.

**arXiv ID:** 2601.22382
</details>

<details>
<summary><strong>MOSAIC: Modular Scalable Autonomy for Intelligent Coordination of Heterogeneous Robotic Teams</strong> - David Oberacker, Julia Richer, Philip Arm, Marvin Grosse Besselmann, Lennart Puck, William Talbot, Maximilian Schik, Sabine Bellmann, Tristan Schnell, Hendrik Kolvenbach, Rüdiger Dillmann, Marco Hutter, Arne Roennau - [[pdf]](https://arxiv.org/pdf/2601.23038)</summary>

**Abstract:** Mobile robots have become indispensable for exploring hostile environments, such as in space or disaster relief scenarios, but often remain limited to teleoperation by a human operator. This restricts the deployment scale and requires near-continuous low-latency communication between the operator and the robot. We present MOSAIC: a scalable autonomy framework for multi-robot scientific exploration using a unified mission abstraction based on Points of Interest (POIs) and multiple layers of autonomy, enabling supervision by a single operator. The framework dynamically allocates exploration and measurement tasks based on each robot's capabilities, leveraging team-level redundancy and specialization to enable continuous operation. We validated the framework in a space-analog field experiment emulating a lunar prospecting scenario, involving a heterogeneous team of five robots and a single operator. Despite the complete failure of one robot during the mission, the team completed 82.3% of assigned tasks at an Autonomy Ratio of 86%, while the operator workload remained at only 78.2%. These results demonstrate that the proposed framework enables robust, scalable multi-robot scientific exploration with limited operator intervention. We further derive practical lessons learned in robot interoperability, networking architecture, team composition, and operator workload management to inform future multi-robot exploration missions.

**arXiv ID:** 2601.23038
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (32 papers)</h2></summary>

<details>
<summary><strong>JAF: Judge Agent Forest</strong> - Sahil Garg, Brad Cheezum, Sridhar Dutta, Vishal Agarwal - [[pdf]](https://arxiv.org/pdf/2601.22269)</summary>

**Abstract:** Judge agents are fundamental to agentic AI frameworks: they provide automated evaluation, and enable iterative self-refinement of reasoning processes. We introduce JAF: Judge Agent Forest, a framework in which the judge agent conducts joint inference across a cohort of query--response pairs generated by a primary agent, rather than evaluating each in isolation. This paradigm elevates the judge from a local evaluator to a holistic learner: by simultaneously assessing related responses, the judge discerns cross-instance patterns and inconsistencies, whose aggregate feedback enables the primary agent to improve by viewing its own outputs through the judge's collective perspective.
Conceptually, JAF bridges belief propagation and ensemble-learning principles: overlapping in-context neighborhoods induce a knowledge-graph structure that facilitates propagation of critique, and repeated, randomized evaluations yield a robust ensemble of context-sensitive judgments. JAF can be instantiated entirely via ICL, with the judge prompted for each query using its associated primary-agent response plus a small, possibly noisy set of peer exemplars. While kNN in embedding space is a natural starting point for exemplars, this approach overlooks categorical structure, domain metadata, or nuanced distinctions accessible to modern LLMs.
To overcome these limitations, we develop a flexible locality-sensitive hashing (LSH) algorithm that learns informative binary codes by integrating semantic embeddings, LLM-driven hash predicates, supervision from categorical labels, and relevant side information. These hash codes support efficient, interpretable, and relation-aware selection of diverse exemplars, and further optimize exploration of CoT reasoning paths. We validate JAF with an empirical study on the demanding task of cloud misconfigs triage in large-scale cloud environments.

**arXiv ID:** 2601.22269
</details>

<details>
<summary><strong>The Six Sigma Agent: Achieving Enterprise-Grade Reliability in LLM Systems Through Consensus-Driven Decomposed Execution</strong> - Khush Patel, Siva Surendira, Jithin George, Shreyas Kapale - [[pdf]](https://arxiv.org/pdf/2601.22290)</summary>

**Abstract:** Large Language Models demonstrate remarkable capabilities yet remain fundamentally probabilistic, presenting critical reliability challenges for enterprise deployment. We introduce the Six Sigma Agent, a novel architecture that achieves enterprise-grade reliability through three synergistic components: (1) task decomposition into a dependency tree of atomic actions; (2) micro-agent sampling where each task is executed n times in parallel across diverse LLMs to generate independent outputs; and (3) consensus voting with dynamic scaling, clustering outputs and selecting the answer from the winning cluster with maximum votes. We prove that sampling n independent outputs with error rate p achieves system error O(p^{ceil(n/2)}), enabling exponential reliability gains. Even using cheaper models with 5% per-action error, consensus voting with 5 agents reduces error to 0.11%; dynamic scaling to 13 agents achieves 3.4 DPMO (Defects Per Million Opportunities), the Six Sigma standard. Evaluation across three enterprise use cases demonstrates a 14,700x reliability improvement over single-agent execution while reducing costs by 80%. Our work establishes that reliability in AI systems emerges from principled redundancy and consensus rather than model scaling alone.

**arXiv ID:** 2601.22290
</details>

<details>
<summary><strong>CVeDRL: An Efficient Code Verifier via Difficulty-aware Reinforcement Learning</strong> - Ji Shi, Peiming Guo, Meishan Zhang, Miao Zhang, Xuebo Liu, Min Zhang, Weili Guan - [[pdf]](https://arxiv.org/pdf/2601.22803)</summary>

**Abstract:** Code verifiers play a critical role in post-verification for LLM-based code generation, yet existing supervised fine-tuning methods suffer from data scarcity, high failure rates, and poor inference efficiency. While reinforcement learning (RL) offers a promising alternative by optimizing models through execution-driven rewards without labeled supervision, our preliminary results show that naive RL with only functionality rewards fails to generate effective unit tests for difficult branches and samples. We first theoretically analyze showing that branch coverage, sample difficulty, syntactic and functional correctness can be jointly modeled as RL rewards, where optimizing these signals can improve the reliability of unit-test-based verification. Guided by this analysis, we design syntax- and functionality-aware rewards and further propose branch- and sample-difficulty--aware RL using exponential reward shaping and static analysis metrics. With this formulation, CVeDRL achieves state-of-the-art performance with only 0.6B parameters, yielding up to 28.97% higher pass rate and 15.08% higher branch coverage than GPT-3.5, while delivering over $20\times$ faster inference than competitive baselines. Code is available at this https URL

**arXiv ID:** 2601.22803
</details>

<details>
<summary><strong>MulFeRL: Enhancing Reinforcement Learning with Verbal Feedback in a Multi-turn Loop</strong> - Xuancheng Li, Haitao Li, Yujia Zhou, YiqunLiu, Qingyao Ai - [[pdf]](https://arxiv.org/pdf/2601.22900)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) is widely used to improve reasoning in multiple domains, yet outcome-only scalar rewards are often sparse and uninformative, especially on failed samples, where they merely indicate failure and provide no insight into why the reasoning fails. In this paper, we investigate how to leverage richer verbal feedback to guide RLVR training on failed samples, and how to convert such feedback into a trainable learning signal. Specifically, we propose a multi-turn feedback-guided reinforcement learning framework. It builds on three mechanisms: (1) dynamic multi-turn regeneration guided by feedback, triggered only on failed samples, (2) two complementary learning signals for within-turn and cross-turn optimization, and (3) structured feedback injection into the model's reasoning process. Trained on sampled OpenR1-Math, the approach outperforms supervised fine-tuning and RLVR baselines in-domain and generalizes well out-of-domain.

**arXiv ID:** 2601.22900
</details>

<details>
<summary><strong>TriCEGAR: A Trace-Driven Abstraction Mechanism for Agentic AI</strong> - Roham Koohestani, Ateş Görpelioğlu, Egor Klimov, Burcu Kulahcioglu Ozkan, Maliheh Izadi - [[pdf]](https://arxiv.org/pdf/2601.22997)</summary>

**Abstract:** Agentic AI systems act through tools and evolve their behavior over long, stochastic interaction traces. This setting complicates assurance, because behavior depends on nondeterministic environments and probabilistic model outputs. Prior work introduced runtime verification for agentic AI via Dynamic Probabilistic Assurance (DPA), learning an MDP online and model checking quantitative properties. A key limitation is that developers must manually define the state abstraction, which couples verification to application-specific heuristics and increases adoption friction. This paper proposes TriCEGAR, a trace-driven abstraction mechanism that automates state construction from execution logs and supports online construction of an agent behavioral MDP. TriCEGAR represents abstractions as predicate trees learned from traces and refined using counterexamples. We describe a framework-native implementation that (i) captures typed agent lifecycle events, (ii) builds abstractions from traces, (iii) constructs an MDP, and (iv) performs probabilistic model checking to compute bounds such as Pmax(success) and Pmin(failure). We also show how run likelihoods enable anomaly detection as a guardrailing signal.

**arXiv ID:** 2601.22997
</details>

<details>
<summary><strong>Whispers of Wealth: Red-Teaming Google's Agent Payments Protocol via Prompt Injection</strong> - Tanusree Debi, Wentian Zhu - [[pdf]](https://arxiv.org/pdf/2601.22569)</summary>

**Abstract:** Large language model (LLM) based agents are increasingly used to automate financial transactions, yet their reliance on contextual reasoning exposes payment systems to prompt-driven manipulation. The Agent Payments Protocol (AP2) aims to secure agent-led purchases through cryptographically verifiable mandates, but its practical robustness remains underexplored. In this work, we perform an AI red-teaming evaluation of AP2 and identify vulnerabilities arising from indirect and direct prompt injection. We introduce two attack techniques, the Branded Whisper Attack and the Vault Whisper Attack which manipulate product ranking and extract sensitive user data. Using a functional AP2 based shopping agent built with Gemini-2.5-Flash and the Google ADK framework, we experimentally validate that simple adversarial prompts can reliably subvert agent behavior. Our findings reveal critical weaknesses in current agentic payment architectures and highlight the need for stronger isolation and defensive safeguards in LLM-mediated financial systems.

**arXiv ID:** 2601.22569
</details>

<details>
<summary><strong>MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning</strong> - Youngeun Kim - [[pdf]](https://arxiv.org/pdf/2601.22582)</summary>

**Abstract:** Group-relative policy optimization methods train language models by generating multiple rollouts per prompt and normalizing rewards with a shared mean reward baseline. In resource-constrained settings where the rollout budget is small, accuracy often degrades. We find that noise in the shared baseline induces advantage sign flips, where some rollouts receive an incorrect advantage sign, and the update direction is reversed. To address this, we propose Median-Centered Group Relative Policy Optimization (MC-GRPO), a simple and effective solution for small-rollout training. Our main idea is to replace the mean baseline with a median baseline: the median is far less sensitive to outlier rewards than the mean, mitigating the sign flips under small rollout size (G). We generate one additional rollout for median reference (G+1), and compute advantages by using the group median. With an odd-sized group, exactly one completion is the median and receives zero advantage, we exclude this pivot rollout from backpropagation so the number of gradient-contributing samples per prompt remains G, preserving the core update cost of standard G-rollout training. Across various GRPO-family methods and a wide range of models and scales, this median-centered training consistently improves stability and final accuracy in the low-rollout regime, reducing the gap between G=2 and G=8 to within 1%. Code is available at this https URL

**arXiv ID:** 2601.22582
</details>

<details>
<summary><strong>Offline Reinforcement Learning of High-Quality Behaviors Under Robust Style Alignment</strong> - Mathieu Petitbois, Rémy Portelas, Sylvain Lamprier - [[pdf]](https://arxiv.org/pdf/2601.22823)</summary>

**Abstract:** We study offline reinforcement learning of style-conditioned policies using explicit style supervision via subtrajectory labeling functions. In this setting, aligning style with high task performance is particularly challenging due to distribution shift and inherent conflicts between style and reward. Existing methods, despite introducing numerous definitions of style, often fail to reconcile these objectives effectively. To address these challenges, we propose a unified definition of behavior style and instantiate it into a practical framework. Building on this, we introduce Style-Conditioned Implicit Q-Learning (SCIQL), which leverages offline goal-conditioned RL techniques, such as hindsight relabeling and value learning, and combine it with a new Gated Advantage Weighted Regression mechanism to efficiently optimize task performance while preserving style alignment. Experiments demonstrate that SCIQL achieves superior performance on both objectives compared to prior offline methods. Code, datasets and visuals are available in: this https URL.

**arXiv ID:** 2601.22823
</details>

<details>
<summary><strong>Degradation-Aware Frequency Regulation of a Heterogeneous Battery Fleet via Reinforcement Learning</strong> - Tanay Raghunandan Srinivasa, Vivek Deulkar, Jia Bhargava, Mohammad Hajiesmaili, Prashant Shenoy - [[pdf]](https://arxiv.org/pdf/2601.22865)</summary>

**Abstract:** Battery energy storage systems are increasingly deployed as fast-responding resources for grid balancing services such as frequency regulation and for mitigating renewable generation uncertainty. However, repeated charging and discharging induces cycling degradation and reduces battery lifetime. This paper studies the real-time scheduling of a heterogeneous battery fleet that collectively tracks a stochastic balancing signal subject to per-battery ramp-rate and capacity constraints, while minimizing long-term cycling degradation.
Cycling degradation is fundamentally path-dependent: it is determined by charge-discharge cycles formed by the state-of-charge (SoC) trajectory and is commonly quantified via rainflow cycle counting. This non-Markovian structure makes it difficult to express degradation as an additive per-time-step cost, complicating classical dynamic programming approaches. We address this challenge by formulating the fleet scheduling problem as a Markov decision process (MDP) with constrained action space and designing a dense proxy reward that provides informative feedback at each time step while remaining aligned with long-term cycle-depth reduction.
To scale learning to large state-action spaces induced by fine-grained SoC discretization and asymmetric per-battery constraints, we develop a function-approximation reinforcement learning method using an Extreme Learning Machine (ELM) as a random nonlinear feature map combined with linear temporal-difference learning. We evaluate the proposed approach on a toy Markovian signal model and on a Markovian model trained from real-world regulation signal traces obtained from the University of Delaware, and demonstrate consistent reductions in cycle-depth occurrence and degradation metrics compared to baseline scheduling policies.

**arXiv ID:** 2601.22865
</details>

<details>
<summary><strong>Reinforcement Learning-Based Co-Design and Operation of Chiller and Thermal Energy Storage for Cost-Optimal HVAC Systems</strong> - Tanay Raghunandan Srinivasa, Vivek Deulkar, Aviruch Bhatia, Vishal Garg - [[pdf]](https://arxiv.org/pdf/2601.22880)</summary>

**Abstract:** We study the joint operation and sizing of cooling infrastructure for commercial HVAC systems using reinforcement learning, with the objective of minimizing life-cycle cost over a 30-year horizon. The cooling system consists of a fixed-capacity electric chiller and a thermal energy storage (TES) unit, jointly operated to meet stochastic hourly cooling demands under time-varying electricity prices. The life-cycle cost accounts for both capital expenditure and discounted operating cost, including electricity consumption and maintenance. A key challenge arises from the strong asymmetry in capital costs: increasing chiller capacity by one unit is far more expensive than an equivalent increase in TES capacity. As a result, identifying the right combination of chiller and TES sizes, while ensuring zero loss-of-cooling-load under optimal operation, is a non-trivial co-design problem. To address this, we formulate the chiller operation problem for a fixed infrastructure configuration as a finite-horizon Markov Decision Process (MDP), in which the control action is the chiller part-load ratio (PLR). The MDP is solved using a Deep Q Network (DQN) with a constrained action space. The learned DQN RL policy minimizes electricity cost over historical traces of cooling demand and electricity prices. For each candidate chiller-TES sizing configuration, the trained policy is evaluated. We then restrict attention to configurations that fully satisfy the cooling demand and perform a life-cycle cost minimization over this feasible set to identify the cost-optimal infrastructure design. Using this approach, we determine the optimal chiller and thermal energy storage capacities to be 700 and 1500, respectively.

**arXiv ID:** 2601.22880
</details>

<details>
<summary><strong>MTDrive: Multi-turn Interactive Reinforcement Learning for Autonomous Driving</strong> - Xidong Li, Mingyu Guo, Chenchao Xu, Bailin Li, Wenjing Zhu, Yangang Zou, Rui Chen, Zehuan Wang - [[pdf]](https://arxiv.org/pdf/2601.22930)</summary>

**Abstract:** Trajectory planning is a core task in autonomous driving, requiring the prediction of safe and comfortable paths across diverse scenarios. Integrating Multi-modal Large Language Models (MLLMs) with Reinforcement Learning (RL) has shown promise in addressing "long-tail" scenarios. However, existing methods are constrained to single-turn reasoning, limiting their ability to handle complex tasks requiring iterative refinement. To overcome this limitation, we present MTDrive, a multi-turn framework that enables MLLMs to iteratively refine trajectories based on environmental feedback. MTDrive introduces Multi-Turn Group Relative Policy Optimization (mtGRPO), which mitigates reward sparsity by computing relative advantages across turns. We further construct an interactive trajectory understanding dataset from closed-loop simulation to support multi-turn training. Experiments on the NAVSIM benchmark demonstrate superior performance compared to existing methods, validating the effectiveness of our multi-turn reasoning paradigm. Additionally, we implement system-level optimizations to reduce data transfer overhead caused by high-resolution images and multi-turn sequences, achieving 2.5x training throughput. Our data, models, and code will be made available soon.

**arXiv ID:** 2601.22930
</details>

<details>
<summary><strong>Automatic Constraint Policy Optimization based on Continuous Constraint Interpolation Framework for Offline Reinforcement Learning</strong> - Xinchen Han, Qiuyang Fang, Hossam Afifi, Michel Marot - [[pdf]](https://arxiv.org/pdf/2601.23010)</summary>

**Abstract:** Offline Reinforcement Learning (RL) relies on policy constraints to mitigate extrapolation error, where both the constraint form and constraint strength critically shape performance. However, most existing methods commit to a single constraint family: weighted behavior cloning, density regularization, or support constraints, without a unified principle that explains their connections or trade-offs. In this work, we propose Continuous Constraint Interpolation (CCI), a unified optimization framework in which these three constraint families arise as special cases along a common constraint spectrum. The CCI framework introduces a single interpolation parameter that enables smooth transitions and principled combinations across constraint types. Building on CCI, we develop Automatic Constraint Policy Optimization (ACPO), a practical primal--dual algorithm that adapts the interpolation parameter via a Lagrangian dual update. Moreover, we establish a maximum-entropy performance difference lemma and derive performance lower bounds for both the closed-form optimal policy and its parametric projection. Experiments on D4RL and NeoRL2 demonstrate robust gains across diverse domains, achieving state-of-the-art performance overall.

**arXiv ID:** 2601.23010
</details>

<details>
<summary><strong>On Safer Reinforcement Learning Policies for Sedation and Analgesia in Intensive Care</strong> - Joel Romero-Hernandez, Oscar Camara - [[pdf]](https://arxiv.org/pdf/2601.23154)</summary>

**Abstract:** Pain management in intensive care usually involves complex trade-offs between therapeutic goals and patient safety, since both inadequate and excessive treatment may induce serious sequelae. Reinforcement learning can help address this challenge by learning medication dosing policies from retrospective data. However, prior work on sedation and analgesia has optimized for objectives that do not value patient survival while relying on algorithms unsuitable for imperfect information settings. We investigated the risks of these design choices by implementing a deep reinforcement learning framework to suggest hourly medication doses under partial observability. Using data from 47,144 ICU stays in the MIMIC-IV database, we trained policies to prescribe opioids, propofol, benzodiazepines, and dexmedetomidine according to two goals: reduce pain or jointly reduce pain and mortality. We found that, although the two policies were associated with lower pain, actions from the first policy were positively correlated with mortality, while those proposed by the second policy were negatively correlated. This suggests that valuing long-term outcomes could be critical for safer treatment policies, even if a short-term goal remains the primary objective.

**arXiv ID:** 2601.23154
</details>

<details>
<summary><strong>Agile Reinforcement Learning through Separable Neural Architecture</strong> - Rajib Mostakim, Reza T. Batley, Sourav Saha - [[pdf]](https://arxiv.org/pdf/2601.23225)</summary>

**Abstract:** Deep reinforcement learning (RL) is increasingly deployed in resource-constrained environments, yet the go-to function approximators - multilayer perceptrons (MLPs) - are often parameter-inefficient due to an imperfect inductive bias for the smooth structure of many value functions. This mismatch can also hinder sample efficiency and slow policy learning in this capacity-limited regime. Although model compression techniques exist, they operate post-hoc and do not improve learning efficiency. Recent spline-based separable architectures - such as Kolmogorov-Arnold Networks (KANs) - have been shown to offer parameter efficiency but are widely reported to exhibit significant computational overhead, especially at scale.
In seeking to address these limitations, this work introduces SPAN (SPline-based Adaptive Networks), a novel function approximation approach to RL. SPAN adapts the low rank KHRONOS framework by integrating a learnable preprocessing layer with a separable tensor product B-spline basis. SPAN is evaluated across discrete (PPO) and high-dimensional continuous (SAC) control tasks, as well as offline settings (Minari/D4RL). Empirical results demonstrate that SPAN achieves a 30-50% improvement in sample efficiency and 1.3-9 times higher success rates across benchmarks compared to MLP baselines. Furthermore, SPAN demonstrates superior anytime performance and robustness to hyperparameter variations, suggesting it as a viable, high performance alternative for learning intrinsically efficient policies in resource-limited settings.

**arXiv ID:** 2601.23225
</details>

<details>
<summary><strong>IRL-DAL: Safe and Adaptive Trajectory Planning for Autonomous Driving via Energy-Guided Diffusion Models</strong> - Seyed Ahmad Hosseini Miangoleh, Amin Jalal Aghdasian, Farzaneh Abdollahi - [[pdf]](https://arxiv.org/pdf/2601.23266)</summary>

**Abstract:** This paper proposes a novel inverse reinforcement learning framework using a diffusion-based adaptive lookahead planner (IRL-DAL) for autonomous vehicles. Training begins with imitation from an expert finite state machine (FSM) controller to provide a stable initialization. Environment terms are combined with an IRL discriminator signal to align with expert goals. Reinforcement learning (RL) is then performed with a hybrid reward that combines diffuse environmental feedback and targeted IRL rewards. A conditional diffusion model, which acts as a safety supervisor, plans safe paths. It stays in its lane, avoids obstacles, and moves smoothly. Then, a learnable adaptive mask (LAM) improves perception. It shifts visual attention based on vehicle speed and nearby hazards. After FSM-based imitation, the policy is fine-tuned with Proximal Policy Optimization (PPO). Training is run in the Webots simulator with a two-stage curriculum. A 96\% success rate is reached, and collisions are reduced to 0.05 per 1k steps, marking a new benchmark for safe navigation. By applying the proposed approach, the agent not only drives in lane but also handles unsafe conditions at an expert level, increasing this http URL make our code publicly available.

**arXiv ID:** 2601.23266
</details>

<details>
<summary><strong>Don't Just Fine-tune the Agent, Tune the Environment</strong> - Siyuan Lu, Zechuan Wang, Hongxuan Zhang, Qintong Wu, Leilei Gan, Chenyi Zhuang, Jinjie Gu, Tao Lin - [[pdf]](https://arxiv.org/pdf/2510.10197)</summary>

**Abstract:** Large Language Model (LLM) agents show great promise for complex, multi-turn tool-use tasks, but their development is often hampered by the extreme scarcity of high-quality training data. Supervised fine-tuning (SFT) on synthetic data leads to overfitting, whereas standard reinforcement learning (RL) struggles with a critical cold-start problem and training instability. To address these challenges, we introduce $\textbf{Environment Tuning}$, a novel training paradigm that enables agents to learn complex behaviors directly from problem instances without relying on pre-collected expert trajectories. $\textbf{Environment Tuning}$ orchestrates this learning process through a structured curriculum, actionable environment augmentation that provides corrective feedback, and fine-grained progress rewards to ensure stable and efficient exploration. Using only 400 problem instances from Berkeley Function-Calling Leaderboard (BFCL) benchmark, our method not only achieves competitive in-distribution performance against strong baselines but also demonstrates superior out-of-distribution generalization, overcoming the performance collapse common to SFT-based approaches. Our work presents a paradigm shift from supervised fine-tuning on static trajectories to dynamic, environment-based exploration, paving the way for training more robust and data-efficient agents. The code is available at this https URL.

**arXiv ID:** 2510.10197
</details>

<details>
<summary><strong>PaperArena: An Evaluation Benchmark for Tool-Augmented Agentic Reasoning on Scientific Literature</strong> - Daoyu Wang, Mingyue Cheng, Shuo Yu, Zirui Liu, Ze Guo, Xin Li, Qi Liu - [[pdf]](https://arxiv.org/pdf/2510.10909)</summary>

**Abstract:** Understanding and reasoning on the large-scale scientific literature is a crucial touchstone for large language model (LLM) based agents. However, existing works are mainly restricted to tool-free tasks within single papers, largely due to the lack of a benchmark that evaluates cross-paper reasoning and multi-tool orchestration in authentic research scenarios. In this work, we propose PaperArena, a benchmark to evaluate LLM-based agents on questions that require integrating information across multiple papers with the assistance of external tools. Given a research question, agents should formulate a reasoning plan, interact with multiple papers, and invoke appropriate tools to produce a well-grounded answer. To support standardized evaluation, we provide a platform for agent execution, offering a modular tool environment including multimodal parsing, context retrieval, and programmatic computation. Experiments reveal that even the leading LLM powering a well-established agentic workflow achieves merely 38.78% average accuracy, while on the hard subset, accuracy drops to only 18.47%. We also analyze reasoning traces and diagnose agent behavior, providing the community with insights to develop and evaluate more capable scientific agents.

**arXiv ID:** 2510.10909
</details>

<details>
<summary><strong>Are Agents Probabilistic Automata? A Trace-Based, Memory-Constrained Theory of Agentic AI</strong> - Roham Koohestani, Ziyou Li, Anton Podkopaev, Maliheh Izadi - [[pdf]](https://arxiv.org/pdf/2510.23487)</summary>

**Abstract:** This paper studies standard controller architectures for agentic AI and derives automata-theoretic models of their interaction behavior via trace semantics and abstraction. We model an agent implementation as a finite control program augmented with explicit memory primitives (bounded buffers, a call stack, or read/write external memory) and a stochastic policy component (e.g., an LLM) that selects among architecturally permitted actions. Instead of equating the concrete agent with a deterministic acceptor, we treat the agent-environment closed loop as inducing a probability distribution over finite interaction traces. Given an abstraction function $\Abs$ from concrete configurations to a finite abstract state space, we obtain a probabilistic trace language and an abstract probabilistic transition model $M_{\Abs}$ suitable for probabilistic model checking.
Imposing explicit, framework-auditable restrictions on memory access and control flow, we prove that the support of the resulting trace language is regular for bounded-memory controllers, context-free for strict call-return controllers, and recursively enumerable for controllers equipped with unbounded read/write memory. These correspondences allow the reuse of existing verification methods for finite-state and pushdown systems, and they delineate precisely when undecidability barriers arise. The probabilistic semantics leads to quantitative analyses such as: what is the probability of entering an unsafe abstract region, and how can we bound this probability in the presence of environment nondeterminism.

**arXiv ID:** 2510.23487
</details>

<details>
<summary><strong>SSL: Sweet Spot Learning for Differentiated Guidance in Agentic Optimization</strong> - Jinyang Wu, Changpeng Yang, Yuhao Shen, Fangzhi Xu, Bolin Ni, Chonghua Liao, Yuchen Liu, Hongzhen Wang, Shuai Nie, Shuai Zhang, Haoran Luo, Jiaming Xu - [[pdf]](https://arxiv.org/pdf/2601.22491)</summary>

**Abstract:** Reinforcement learning with verifiable rewards has emerged as a powerful paradigm for training intelligent agents. However, existing methods typically employ binary rewards that fail to capture quality differences among trajectories achieving identical outcomes, thereby overlooking potential diversity within the solution space. Inspired by the ``sweet spot'' concept in tennis-the racket's core region that produces optimal hitting effects, we introduce \textbf{S}weet \textbf{S}pot \textbf{L}earning (\textbf{SSL}), a novel framework that provides differentiated guidance for agent optimization. SSL follows a simple yet effective principle: progressively amplified, tiered rewards guide policies toward the sweet-spot region of the solution space. This principle naturally adapts across diverse tasks: visual perception tasks leverage distance-tiered modeling to reward proximity, while complex reasoning tasks reward incremental progress toward promising solutions. We theoretically demonstrate that SSL preserves optimal solution ordering and enhances the gradient signal-to-noise ratio, thereby fostering more directed optimization. Extensive experiments across GUI perception, short/long-term planning, and complex reasoning tasks show consistent improvements over strong baselines on 12 benchmarks, achieving up to 2.5X sample efficiency gains and effective cross-task transferability. Our work establishes SSL as a general principle for training capable and robust agents.

**arXiv ID:** 2601.22491
</details>

<details>
<summary><strong>Mock Worlds, Real Skills: Building Small Agentic Language Models with Synthetic Tasks, Simulated Environments, and Rubric-Based Rewards</strong> - Yuan-Jay Lü, Chengyu Wang, Lei Shen, Jun Huang, Tong Xu - [[pdf]](https://arxiv.org/pdf/2601.22511)</summary>

**Abstract:** Small LLMs often struggle to match the agentic capabilities of large, costly models. While reinforcement learning can help, progress has been limited by two structural bottlenecks: existing open-source agentic training data are narrow in task variety and easily solved; real-world APIs lack diversity and are unstable for large-scale reinforcement learning rollout processes. We address these challenges with SYNTHAGENT, a framework that jointly synthesizes diverse tool-use training data and simulates complete environments. Specifically, a strong teacher model creates novel tasks and tool ecosystems, then rewrites them into intentionally underspecified instructions. This compels agents to actively query users for missing details. When handling synthetic tasks, an LLM-based user simulator provides user-private information, while a mock tool system delivers stable tool responses. For rewards, task-level rubrics are constructed based on required subgoals, user-agent interactions, and forbidden behaviors. Across 14 challenging datasets in math, search, and tool use, models trained on our synthetic data achieve substantial gains, with small models outperforming larger baselines.

**arXiv ID:** 2601.22511
</details>

<details>
<summary><strong>HeaPA: Difficulty-Aware Heap Sampling and On-Policy Query Augmentation for LLM Reinforcement Learning</strong> - Weiqi Wang, Xin Liu, Binxuan Huang, Hejie Cui, Rongzhi Zhang, Changlong Yu, Shuowei Jin, Jingfeng Yang, Qingyu Yin, Zhengyang Wang, Zheng Li, Yifan Gao, Priyanka Nigam, Bing Yin, Lihong Li, Yangqiu Song - [[pdf]](https://arxiv.org/pdf/2601.22448)</summary>

**Abstract:** RLVR is now a standard way to train LLMs on reasoning tasks with verifiable outcomes, but when rollout generation dominates the cost, efficiency depends heavily on which prompts you sample and when. In practice, prompt pools are often static or only loosely tied to the model's learning progress, so uniform sampling can't keep up with the shifting capability frontier and ends up wasting rollouts on prompts that are already solved or still out of reach. Existing approaches improve efficiency through filtering, curricula, adaptive rollout allocation, or teacher guidance, but they typically assume a fixed pool-which makes it hard to support stable on-policy pool growth-or they add extra teacher cost and latency. We introduce HeaPA (Heap Sampling and On-Policy Query Augmentation), which maintains a bounded, evolving pool, tracks the frontier using heap-based boundary sampling, expands the pool via on-policy augmentation with lightweight asynchronous validation, and stabilizes correlated queries through topology-aware re-estimation of pool statistics and controlled reinsertion. Across two training corpora, two training recipes, and seven benchmarks, HeaPA consistently improves accuracy and reaches target performance with fewer computations while keeping wall-clock time comparable. Our analyses suggest these gains come from frontier-focused sampling and on-policy pool growth, with the benefits becoming larger as model scale increases. Our code is available at this https URL.

**arXiv ID:** 2601.22448
</details>

<details>
<summary><strong>Mem-T: Densifying Rewards for Long-Horizon Memory Agents</strong> - Yanwei Yue, Guibin Zhang, Boci Peng, Xuanbo Fan, Jiaxin Guo, Qiankun Li, Yan Zhang - [[pdf]](https://arxiv.org/pdf/2601.23014)</summary>

**Abstract:** Memory agents, which depart from predefined memory-processing pipelines by endogenously managing the processing, storage, and retrieval of memories, have garnered increasing attention for their autonomy and adaptability. However, existing training paradigms remain constrained: agents often traverse long-horizon sequences of memory operations before receiving sparse and delayed rewards, which hinders truly end-to-end optimization of memory management policies. To address this limitation, we introduce Mem-T, an autonomous memory agent that interfaces with a lightweight hierarchical memory database to perform dynamic updates and multi-turn retrieval over streaming inputs. To effectively train long-horizon memory management capabilities, we further propose MoT-GRPO, a tree-guided reinforcement learning framework that transforms sparse terminal feedback into dense, step-wise supervision via memory operation tree backpropagation and hindsight credit assignment, thereby enabling the joint optimization of memory construction and retrieval. Extensive experiments demonstrate that Mem-T is (1) high-performing, surpassing frameworks such as A-Mem and Mem0 by up to $14.92\%$, and (2) economical, operating on a favorable accuracy-efficiency Pareto frontier and reducing inference tokens per query by $\sim24.45\%$ relative to GAM without sacrificing performance.

**arXiv ID:** 2601.23014
</details>

<details>
<summary><strong>Surrogate Signals from Format and Length: Reinforcement Learning for Solving Mathematical Problems without Ground Truth Answers</strong> - Rihui Xin, Han Liu, Zecheng Wang, Yupeng Zhang, Dianbo Sui, Xiaolin Hu, Bingning Wang - [[pdf]](https://arxiv.org/pdf/2505.19439)</summary>

**Abstract:** Large Language Models (LLMs) have achieved remarkable success in natural language processing tasks, with Reinforcement Learning (RL) playing a key role in adapting them to specific applications. In mathematical problem solving, however, the reliance on ground truth answers poses significant challenges due to their high collection cost and limited availability.
This work explores the use of simple surrogate signals, format and length, to guide RL training. We find that early training is dominated by format learning, where structural feedback alone accounts for most performance gains. Incorporating length-based rewards further refines outputs by discouraging overly long or short responses, enabling a GRPO approach with format-length signals to approximate, and in some cases surpass, ground-truth-based optimization. For example, our method achieves 40.0% accuracy on AIME2024 with a 7B base model, and generalizes across different model sizes and series.
Beyond practical efficiency, these findings provide an inspirational perspective on RL: rather than imparting new knowledge, RL primarily activates reasoning capabilities already embedded in pre-trained models. This insight suggests that lightweight, label-efficient strategies can complement pre-training to unlock LLMs' latent potential in reasoning-intensive tasks.

**arXiv ID:** 2505.19439
</details>

<details>
<summary><strong>Trajectory2Task: Training Robust Tool-Calling Agents with Synthesized Yet Verifiable Data for Complex User Intents</strong> - Ziyi Wang, Yuxuan Lu, Yimeng Zhang, Ziwei Dong, Jing Huang, Jiri Gesi, Xianfeng Tang, Chen Luo, Yisi Sang, Hanqing Lu, Manling Li, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2601.20144)</summary>

**Abstract:** Tool-calling agents are increasingly deployed in real-world customer-facing workflows. Yet most studies on tool-calling agents focus on idealized settings with general, fixed, and well-specified tasks. In real-world applications, user requests are often (1) ambiguous, (2) changing over time, or (3) infeasible due to policy constraints, and training and evaluation data that cover these diverse, complex interaction patterns remain under-represented. To bridge the gap, we present Trajectory2Task, a verifiable data generation pipeline for studying tool use at scale under three realistic user scenarios: ambiguous intent, changing intent, and infeasible intents. The pipeline first conducts multi-turn exploration to produce valid tool-call trajectories. It then converts these trajectories into user-facing tasks with controlled intent adaptations. This process yields verifiable task that support closed-loop evaluation and training. We benchmark seven state-of-the-art LLMs on the generated complex user scenario tasks and observe frequent failures. Finally, using successful trajectories obtained from task rollouts, we fine-tune lightweight LLMs and find consistent improvements across all three conditions, along with better generalization to unseen tool-use domains, indicating stronger general tool-calling ability.

**arXiv ID:** 2601.20144
</details>

<details>
<summary><strong>Latent Spherical Flow Policy for Reinforcement Learning with Combinatorial Actions</strong> - Lingkai Kong, Anagha Satish, Hezi Jiang, Akseli Kangaslahti, Andrew Ma, Wenbo Chen, Mingxiao Song, Lily Xu, Milind Tambe - [[pdf]](https://arxiv.org/pdf/2601.22211)</summary>

**Abstract:** Reinforcement learning (RL) with combinatorial action spaces remains challenging because feasible action sets are exponentially large and governed by complex feasibility constraints, making direct policy parameterization impractical. Existing approaches embed task-specific value functions into constrained optimization programs or learn deterministic structured policies, sacrificing generality and policy expressiveness. We propose a solver-induced \emph{latent spherical flow policy} that brings the expressiveness of modern generative policies to combinatorial RL while guaranteeing feasibility by design. Our method, LSFlow, learns a \emph{stochastic} policy in a compact continuous latent space via spherical flow matching, and delegates feasibility to a combinatorial optimization solver that maps each latent sample to a valid structured action. To improve efficiency, we train the value network directly in the latent space, avoiding repeated solver calls during policy optimization. To address the piecewise-constant and discontinuous value landscape induced by solver-based action selection, we introduce a smoothed Bellman operator that yields stable, well-defined learning targets. Empirically, our approach outperforms state-of-the-art baselines by an average of 20.6\% across a range of challenging combinatorial RL tasks.

**arXiv ID:** 2601.22211
</details>

<details>
<summary><strong>Quantum-Inspired Reinforcement Learning for Secure and Sustainable AIoT-Driven Supply Chain Systems</strong> - Muhammad Bilal Akram Dastagir, Omer Tariq, Shahid Mumtaz, Saif Al-Kuwari, Ahmed Farouk - [[pdf]](https://arxiv.org/pdf/2601.22339)</summary>

**Abstract:** Modern supply chains must balance high-speed logistics with environmental impact and security constraints, prompting a surge of interest in AI-enabled Internet of Things (AIoT) solutions for global commerce. However, conventional supply chain optimization models often overlook crucial sustainability goals and cyber vulnerabilities, leaving systems susceptible to both ecological harm and malicious attacks. To tackle these challenges simultaneously, this work integrates a quantum-inspired reinforcement learning framework that unifies carbon footprint reduction, inventory management, and cryptographic-like security measures. We design a quantum-inspired reinforcement learning framework that couples a controllable spin-chain analogy with real-time AIoT signals and optimizes a multi-objective reward unifying fidelity, security, and carbon costs. The approach learns robust policies with stabilized training via value-based and ensemble updates, supported by window-normalized reward components to ensure commensurate scaling. In simulation, the method exhibits smooth convergence, strong late-episode performance, and graceful degradation under representative noise channels, outperforming standard learned and model-based references, highlighting its robust handling of real-time sustainability and risk demands. These findings reinforce the potential for quantum-inspired AIoT frameworks to drive secure, eco-conscious supply chain operations at scale, laying the groundwork for globally connected infrastructures that responsibly meet both consumer and environmental needs.

**arXiv ID:** 2601.22339
</details>

<details>
<summary><strong>SAIR: Cost-Efficient Multi-Stage ML Pipeline Autoscaling via In-Context Reinforcement Learning</strong> - Jianchang Su, Yifan Zhang, Shengkai Lin, Shizhen Zhao, Yusheng Zheng, Yiwei Yang, Wei Zhang - [[pdf]](https://arxiv.org/pdf/2601.22397)</summary>

**Abstract:** Multi-stage ML inference pipelines are difficult to autoscale due to heterogeneous resources, cross-stage coupling, and dynamic bottleneck migration. We present SAIR, an autoscaling framework that uses an LLM as an in-context reinforcement learning controller, improving its policy online from reward-labeled interaction histories without gradient updates. SAIR combines Pareto-dominance reward shaping with a provable separation margin, surprisal-guided experience retrieval for context efficiency, and fine-grained GPU rate control via user-space CUDA interception. We provide regret analysis decomposing error into retrieval coverage and LLM selection components. On four ML serving pipelines under three workload patterns, SAIR achieves the best or tied-best P99 latency and effective resource cost among deployed baselines, improving P99 by up to 50% and reducing effective cost by up to 97% (under GPU rate-control assumptions), with 86% bottleneck detection accuracy and no offline training.

**arXiv ID:** 2601.22397
</details>

<details>
<summary><strong>Continual Policy Distillation from Distributed Reinforcement Learning Teachers</strong> - Yuxuan Li, Qijun He, Mingqi Yuan, Wen-Tse Chen, Jeff Schneider, Jiayu Chen - [[pdf]](https://arxiv.org/pdf/2601.22475)</summary>

**Abstract:** Continual Reinforcement Learning (CRL) aims to develop lifelong learning agents to continuously acquire knowledge across diverse tasks while mitigating catastrophic forgetting. This requires efficiently managing the stability-plasticity dilemma and leveraging prior experience to rapidly generalize to novel tasks. While various enhancement strategies for both aspects have been proposed, achieving scalable performance by directly applying RL to sequential task streams remains challenging. In this paper, we propose a novel teacher-student framework that decouples CRL into two independent processes: training single-task teacher models through distributed RL and continually distilling them into a central generalist model. This design is motivated by the observation that RL excels at solving single tasks, while policy distillation -- a relatively stable supervised learning process -- is well aligned with large foundation models and multi-task learning. Moreover, a mixture-of-experts (MoE) architecture and a replay-based approach are employed to enhance the plasticity and stability of the continual policy distillation process. Extensive experiments on the Meta-World benchmark demonstrate that our framework enables efficient continual RL, recovering over 85% of teacher performance while constraining task-wise forgetting to within 10%.

**arXiv ID:** 2601.22475
</details>

<details>
<summary><strong>From Absolute to Relative: Rethinking Reward Shaping in Group-Based Reinforcement Learning</strong> - Wenzhe Niu, Wei He, Zongxia Xie, Jinpeng Ou, Huichuan Fan, Yuchen Ge, Yanru Sun, Ziyin Wang, Yizhao Sun, Chengshun Shi, Jiuchong Gao, Jinghua Hao, Renqing He - [[pdf]](https://arxiv.org/pdf/2601.23058)</summary>

**Abstract:** Reinforcement learning has become a cornerstone for enhancing the reasoning capabilities of Large Language Models, where group-based approaches such as GRPO have emerged as efficient paradigms that optimize policies by leveraging intra-group performance differences. However, these methods typically rely on absolute numerical rewards, introducing intrinsic limitations. In verifiable tasks, identical group evaluations often result in sparse supervision, while in open-ended scenarios, the score range instability of reward models undermines advantage estimation based on group means. To address these limitations, we propose Reinforcement Learning with Relative Rewards (RLRR), a framework that shifts reward shaping from absolute scoring to relative ranking. Complementing this framework, we introduce the Ranking Reward Model, a listwise preference model tailored for group-based optimization to directly generate relative rankings. By transforming raw evaluations into robust relative signals, RLRR effectively mitigates signal sparsity and reward instability. Experimental results demonstrate that RLRR yields consistent performance improvements over standard group-based baselines across reasoning benchmarks and open-ended generation tasks.

**arXiv ID:** 2601.23058
</details>

<details>
<summary><strong>RN-D: Discretized Categorical Actors with Regularized Networks for On-Policy Reinforcement Learning</strong> - Yuexin Bian, Jie Feng, Tao Wang, Yijiang Li, Sicun Gao, Yuanyuan Shi - [[pdf]](https://arxiv.org/pdf/2601.23075)</summary>

**Abstract:** On-policy deep reinforcement learning remains a dominant paradigm for continuous control, yet standard implementations rely on Gaussian actors and relatively shallow MLP policies, often leading to brittle optimization when gradients are noisy and policy updates must be conservative. In this paper, we revisit policy representation as a first-class design choice for on-policy optimization. We study discretized categorical actors that represent each action dimension with a distribution over bins, yielding a policy objective that resembles a cross-entropy loss. Building on architectural advances from supervised learning, we further propose regularized actor networks, while keeping critic design fixed. Our results show that simply replacing the standard actor network with our discretized regularized actor yields consistent gains and achieve the state-of-the-art performance across diverse continuous-control benchmarks.

**arXiv ID:** 2601.23075
</details>

<details>
<summary><strong>Toward Fully Autonomous Driving: AI, Challenges, Opportunities, and Needs</strong> - Lars Ullrich, Michael Buchholz, Klaus Dietmayer, Knut Graichen - [[pdf]](https://arxiv.org/pdf/2601.22927)</summary>

**Abstract:** Automated driving (AD) is promising, but the transition to fully autonomous driving is, among other things, subject to the real, ever-changing open world and the resulting challenges. However, research in the field of AD demonstrates the ability of artificial intelligence (AI) to outperform classical approaches, handle higher complexities, and reach a new level of autonomy. At the same time, the use of AI raises further questions of safety and transferability. To identify the challenges and opportunities arising from AI concerning autonomous driving functionalities, we have analyzed the current state of AD, outlined limitations, and identified foreseeable technological possibilities. Thereby, various further challenges are examined in the context of prospective developments. In this way, this article reconsiders fully autonomous driving with respect to advancements in the field of AI and carves out the respective needs and resulting research questions.

**arXiv ID:** 2601.22927
</details>

<details>
<summary><strong>Reinforcement Learning for Ballbot Navigation in Uneven Terrain</strong> - Achkan Salehi - [[pdf]](https://arxiv.org/pdf/2505.18417)</summary>

**Abstract:** Ballbot (i.e. Ball balancing robot) navigation usually relies on methods rooted in control theory (CT), and works that apply Reinforcement learning (RL) to the problem remain rare while generally being limited to specific subtasks (e.g. balance recovery). Unlike CT based methods, RL does not require (simplifying) assumptions about environment dynamics (e.g. the absence of slippage between the ball and the floor). In addition to this increased accuracy in modeling, RL agents can easily be conditioned on additional observations such as depth-maps without the need for explicit formulations from first principles, leading to increased adaptivity. Despite those advantages, there has been little to no investigation into the capabilities, data-efficiency and limitations of RL based methods for ballbot control and navigation. Furthermore, there is a notable absence of an open-source, RL-friendly simulator for this task. In this paper, we present an open-source ballbot simulation based on MuJoCo, and show that with appropriate conditioning on exteroceptive observations as well as reward shaping, policies learned by classical model-free RL methods are capable of effectively navigating through randomly generated uneven terrain, using a reasonable amount of data (four to five hours on a system operating at 500hz). Our code is made publicly available.

**arXiv ID:** 2505.18417
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
