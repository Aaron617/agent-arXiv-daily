# Agent arXiv Daily

**Last Updated:** 2025-10-22 02:48:00

**Total Papers:** 90

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
<summary><strong>AndroidControl-Curated: Revealing the True Potential of GUI Agents through Benchmark Purification</strong> - Ho Fai Leung, Xiaoyan Xi, Fei Zuo - [[pdf]](https://arxiv.org/pdf/2510.18488)</summary>

**Abstract:** On-device virtual assistants like Siri and Google Assistant are increasingly pivotal, yet their capabilities are hamstrung by a reliance on rigid, developer-dependent APIs. GUI agents offer a powerful, API-independent alternative, but their adoption is hindered by the perception of poor performance, as even the best models (e.g. Qwen3-VL-235B) scores are capped at around 60% on benchmarks like AndroidControl, far from viability for real-world use. Our research reveals that issue lies not only with the models but with the benchmarks themselves. We identified notable shortcomings in AndroidControl, including ambiguities and factual errors, which systematically underrates agent capabilities. To address this critical oversight, we enhanced AndroidControl into AndroidControl-Curated, a refined version of the benchmark improved through a rigorous purification pipeline. On this enhanced benchmark, state-of-the-art models achieve success rates nearing 75% on complex tasks (15% improvement), reflecting that on-device GUI agents are actually closer to practical deployment than previously thought. We introduce our new SOTA model, Magma-R1- 3B, post-trained on just 2.4k curated samples using 60 hours of an H20 GPU (approximately $60). Despite being 200 times smaller in parameters, this model delivers performance comparable to Qwen3- VL-235B. We release both AndroidControl-Curated benchmark and Magma-R1 model to the research community, encouraging adoption of this enhanced benchmark to better reflect model capabilities and accelerate the development of robust, on-device virtual assistants.

**arXiv ID:** 2510.18488
</details>

<details>
<summary><strong>Lyapunov-Aware Quantum-Inspired Reinforcement Learning for Continuous-Time Vehicle Control: A Feasibility Study</strong> - Nutkritta Kraipatthanapong, Natthaphat Thathong, Pannita Suksawas, Thanunnut Klunklin, Kritin Vongthonglua, Krit Attahakul, Aueaphum Aueawatthanaphisut - [[pdf]](https://arxiv.org/pdf/2510.18852)</summary>

**Abstract:** This paper presents a novel Lyapunov-Based Quantum Reinforcement Learning (LQRL) framework that integrates quantum policy optimization with Lyapunov stability analysis for continuous-time vehicle control. The proposed approach combines the representational power of variational quantum circuits (VQCs) with a stability-aware policy gradient mechanism to ensure asymptotic convergence and safe decision-making under dynamic environments. The vehicle longitudinal control problem was formulated as a continuous-state reinforcement learning task, where the quantum policy network generates control actions subject to Lyapunov stability constraints. Simulation experiments were conducted in a closed-loop adaptive cruise control scenario using a quantum-inspired policy trained under stability feedback. The results demonstrate that the LQRL framework successfully embeds Lyapunov stability verification into quantum policy learning, enabling interpretable and stability-aware control performance. Although transient overshoot and Lyapunov divergence were observed under aggressive acceleration, the system maintained bounded state evolution, validating the feasibility of integrating safety guarantees within quantum reinforcement learning architectures. The proposed framework provides a foundational step toward provably safe quantum control in autonomous systems and hybrid quantum-classical optimization domains.

**arXiv ID:** 2510.18852
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (9 papers)</h2></summary>

<details>
<summary><strong>StarBench: A Turn-Based RPG Benchmark for Agentic Multimodal Decision-Making and Information Seeking</strong> - Haoran Zhang, Chenhao Zhu, Sicong Guo, Hanzhe Guo, Haiming Li, Donglin Yu - [[pdf]](https://arxiv.org/pdf/2510.18483)</summary>

**Abstract:** Human players do more than press buttons: they ground what they see on screen into precise keyboard-mouse actions and, when stuck, they seek information before trying again. We ask whether current vision-language models (VLMs) can do the same. Despite encouraging results under simplified control or tool scaffolds, human-like play in a real client - mapping raw screenshots to temporally coherent low-level actions while deciding when to ask for guidance - remains an open challenge. We introduce StarBench, a turn-based RPG benchmark derived from Honkai: Star Rail that targets these two human-like competencies: multimodal decision-making from pixels to actions and agentic information seeking. StarBench standardizes evaluation across eight combat tasks and two regimes with shared tasks and metrics: (i) direct control, where agents receive only screenshots and must emit low-level primitives (click and keypress) with no semantic hints; and (ii) tool-assisted control, where higher-level intents can be mapped to primitives by detectors and OCR outputs provide optional textualized observations to ease UI grounding. To mirror human practice, StarBench also includes an ask-or-act diagnostic that measures whether and when agents choose to request brief guidance before proceeding, and how that choice affects subsequent performance. We report reference baselines for contemporary VLMs and a human reference. Results expose sizable gaps in perception-to-control fidelity in the direct regime, while showing that judicious information seeking correlates with improved success, establishing StarBench as a reproducible yardstick for agentic information seeking and multimodal decision-making in real-client play.

**arXiv ID:** 2510.18483
</details>

<details>
<summary><strong>SpecAgent: A Speculative Retrieval and Forecasting Agent for Code Completion</strong> - George Ma, Anurag Koul, Qi Chen, Yawen Wu, Sachit Kuhar, Yu Yu, Aritra Sengupta, Varun Kumar, Murali Krishna Ramanathan - [[pdf]](https://arxiv.org/pdf/2510.17925)</summary>

**Abstract:** Large Language Models (LLMs) excel at code-related tasks but often struggle in realistic software repositories, where project-specific APIs and cross-file dependencies are crucial. Retrieval-augmented methods mitigate this by injecting repository context at inference time. The low inference-time latency budget affects either retrieval quality or the added latency adversely impacts user experience. We address this limitation with SpecAgent, an agent that improves both latency and code-generation quality by proactively exploring repository files during indexing and constructing speculative context that anticipates future edits in each file. This indexing-time asynchrony allows thorough context computation, masking latency, and the speculative nature of the context improves code-generation quality. Additionally, we identify the problem of future context leakage in existing benchmarks, which can inflate reported performance. To address this, we construct a synthetic, leakage-free benchmark that enables a more realistic evaluation of our agent against baselines. Experiments show that SpecAgent consistently achieves absolute gains of 9-11% (48-58% relative) compared to the best-performing baselines, while significantly reducing inference latency.

**arXiv ID:** 2510.17925
</details>

<details>
<summary><strong>BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers?</strong> - Fengqing Jiang, Yichen Feng, Yuetai Li, Luyao Niu, Basel Alomair, Radha Poovendran - [[pdf]](https://arxiv.org/pdf/2510.18003)</summary>

**Abstract:** The convergence of LLM-powered research assistants and AI-based peer review systems creates a critical vulnerability: fully automated publication loops where AI-generated research is evaluated by AI reviewers without human oversight. We investigate this through \textbf{BadScientist}, a framework that evaluates whether fabrication-oriented paper generation agents can deceive multi-model LLM review systems. Our generator employs presentation-manipulation strategies requiring no real experiments. We develop a rigorous evaluation framework with formal error guarantees (concentration bounds and calibration analysis), calibrated on real data. Our results reveal systematic vulnerabilities: fabricated papers achieve acceptance rates up to . Critically, we identify \textit{concern-acceptance conflict} -- reviewers frequently flag integrity issues yet assign acceptance-level scores. Our mitigation strategies show only marginal improvements, with detection accuracy barely exceeding random chance. Despite provably sound aggregation mathematics, integrity checking systematically fails, exposing fundamental limitations in current AI-driven review systems and underscoring the urgent need for defense-in-depth safeguards in scientific publishing.

**arXiv ID:** 2510.18003
</details>

<details>
<summary><strong>Discovering the curriculum with AI: A proof-of-concept demonstration with an intelligent tutoring system for teaching project selection</strong> - Lovis Heindrich, Falk Lieder - [[pdf]](https://arxiv.org/pdf/2406.04082)</summary>

**Abstract:** The decisions of individuals and organizations are often suboptimal because fully rational decision-making is too demanding in the real world. Recent work suggests that some errors can be prevented by leveraging artificial intelligence to discover and teach clever heuristics. So far, this line of research has been limited to simplified, artificial decision-making tasks. This article is the first to extend this approach to a real-world decision problem, namely, executives deciding which project their organization should launch next. We develop a computational method (MGPS) that automatically discovers project selection strategies that are optimized for real people, and we develop an intelligent tutor that teaches the discovered project selection procedures. We evaluated MGPS on a computational benchmark and tested the intelligent tutor in a training experiment with two control conditions. MGPS outperformed a state-of-the-art method and was more computationally efficient. Moreover, people who practiced with our intelligent tutor learned significantly better project selection strategies than the control groups. These findings suggest that AI could be used to automate the process of discovering and formalizing the cognitive strategies taught by intelligent tutoring systems.

**arXiv ID:** 2406.04082
</details>

<details>
<summary><strong>Can Agents Fix Agent Issues?</strong> - Alfin Wijaya Rahardja, Junwei Liu, Weitong Chen, Zhenpeng Chen, Yiling Lou - [[pdf]](https://arxiv.org/pdf/2505.20749)</summary>

**Abstract:** LLM-based agent systems are emerging as a new software paradigm and have been widely adopted across diverse domains such as medicine, robotics, and programming. However, maintaining these systems requires substantial effort, as they are inevitably prone to bugs and continually evolve to meet changing external requirements. Therefore, automatically resolving agent issues (i.e., bug reports or feature requests) is a crucial and challenging task. While recent software engineering (SE) agents (e.g., SWE-agent) have shown promise in addressing issues in traditional software systems, it remains unclear how effectively they can resolve real-world issues in agent systems, which differ significantly from traditional software. To fill this gap, we first manually analyze 201 real-world agent issues and identify common categories of agent issues. We then spend 500 person-hours constructing AGENTISSUE-BENCH, a reproducible benchmark comprising 50 agent issue resolution tasks (each with an executable environment and failure-triggering tests). We further evaluate state-of-the-art SE agents on AGENTISSUE-BENCH and reveal their limited effectiveness (i.e., with only 3.33% - 12.67% resolution rates). These results underscore the unique challenges of maintaining agent systems compared to traditional software, highlighting the need for further research to develop advanced SE agents for resolving agent issues. Data and code are available at this https URL .

**arXiv ID:** 2505.20749
</details>

<details>
<summary><strong>CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment</strong> - Radin Shayanfar, Chu Fei Luo, Rohan Bhambhoria, Samuel Dahan, Xiaodan Zhu - [[pdf]](https://arxiv.org/pdf/2506.02264)</summary>

**Abstract:** Building Task-Oriented Dialogue (TOD) systems that generalize across different tasks remains a challenging problem. Data-driven approaches often struggle to transfer effectively to unseen tasks. While recent schema-based TOD frameworks improve generalization by decoupling task logic from language understanding, their reliance on neural or generative models often obscures how task schemas influence behaviour and hence impair interpretability. In this work, we introduce a novel framework, CoDial (Code for Dialogue), which converts a TOD task schema, represented as a novel structured heterogeneous graph, to programmatic LLM guardrailing code, such as NVIDIA's Colang, enabling interpretable and efficient alignment of dialogue policies during inference. We introduce two paradigms, $\text{CoDial}_{\text{free}}$ and $\text{CoDial}_{\text{structured}}$ for generating LLM guardrails, and propose a feedback mechanism that integrates human feedback to iteratively improve the generated code. Empirically, CoDial achieves state-of-the-art (SOTA) performance on the widely used STAR dataset and is on par with SOTA on the MultiWOZ dataset, while also providing interpretability. We additionally demonstrate CoDial's iterative improvement via manual and LLM-aided feedback, making it a practical tool for expert-guided alignment of LLMs in high-stakes domains.

**arXiv ID:** 2506.02264
</details>

<details>
<summary><strong>MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning</strong> - Tajamul Ashraf, Umair Nawaz, Abdelrahman M. Shaker, Rao Anwer, Philip Torr, Fahad Shahbaz Khan, Salman Khan - [[pdf]](https://arxiv.org/pdf/2510.08567)</summary>

**Abstract:** Vision language models (VLMs) are increasingly deployed as controllers with access to external tools for complex reasoning and decision-making, yet their effectiveness remains limited by the scarcity of high-quality multimodal trajectories and the cost of manual annotation. We address this challenge with a vision-centric agent tuning framework that automatically synthesizes multimodal trajectories, generates step-wise preference pairs, and trains a VLM controller for robust tool-use reasoning. Our pipeline first constructs M-TRACE, a large-scale dataset of 28.5K multimodal tasks with 177K verified trajectories, enabling imitation-based trajectory tuning. Building on this, we develop MATRIX Agent, a controller finetuned on M-TRACE for step-wise tool reasoning. To achieve finer alignment, we further introduce Pref-X, a set of 11K automatically generated preference pairs, and optimize MATRIX on it via step-wise preference learning. Across three benchmarks, Agent-X, GTA, and GAIA, MATRIX consistently surpasses both open- and closed-source VLMs, demonstrating scalable and effective multimodal tool use. Our data and code is avaliable at this https URL.

**arXiv ID:** 2510.08567
</details>

<details>
<summary><strong>RoboChallenge: Large-scale Real-robot Evaluation of Embodied Policies</strong> - Adina Yakefu, Bin Xie, Chongyang Xu, Enwen Zhang, Erjin Zhou, Fan Jia, Haitao Yang, Haoqiang Fan, Haowei Zhang, Hongyang Peng, Jing Tan, Junwen Huang, Kai Liu, Kaixin Liu, Kefan Gu, Qinglun Zhang, Ruitao Zhang, Saike Huang, Shen Cheng, Shuaicheng Liu, Tiancai Wang, Tiezhen Wang, Wei Sun, Wenbin Tang, Yajun Wei, Yang Chen, Youqiang Gui, Yucheng Zhao, Yunchao Ma, Yunfei Wei, Yunhuan Yang, Yutong Guo, Ze Chen, Zhengyuan Du, Ziheng Zhang, Ziming Liu, Ziwei Yan - [[pdf]](https://arxiv.org/pdf/2510.17950)</summary>

**Abstract:** Testing on real machines is indispensable for robotic control algorithms. In the context of learning-based algorithms, especially VLA models, demand for large-scale evaluation, i.e. testing a large number of models on a large number of tasks, is becoming increasingly urgent. However, doing this right is highly non-trivial, especially when scalability and reproducibility is taken into account. In this report, we describe our methodology for constructing RoboChallenge, an online evaluation system to test robotic control algorithms, and our survey of recent state-of-the-art VLA models using our initial benchmark Table30.

**arXiv ID:** 2510.17950
</details>

<details>
<summary><strong>Interpretable Decision-Making for End-to-End Autonomous Driving</strong> - Mona Mirzaie, Bodo Rosenhahn - [[pdf]](https://arxiv.org/pdf/2508.18898)</summary>

**Abstract:** Trustworthy AI is mandatory for the broad deployment of autonomous vehicles. Although end-to-end approaches derive control commands directly from raw data, interpreting these decisions remains challenging, especially in complex urban scenarios. This is mainly attributed to very deep neural networks with non-linear decision boundaries, making it challenging to grasp the logic behind AI-driven decisions. This paper presents a method to enhance interpretability while optimizing control commands in autonomous driving. To address this, we propose loss functions that promote the interpretability of our model by generating sparse and localized feature maps. The feature activations allow us to explain which image regions contribute to the predicted control command. We conduct comprehensive ablation studies on the feature extraction step and validate our method on the CARLA benchmarks. We also demonstrate that our approach improves interpretability, which correlates with reducing infractions, yielding a safer, high-performance driving model. Notably, our monocular, non-ensemble model surpasses the top-performing approaches from the CARLA Leaderboard by achieving lower infraction scores and the highest route completion rate, all while ensuring interpretability.

**arXiv ID:** 2508.18898
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness in Conversational AI</strong> - Manik Rana, Calissa Man, Anotida Expected Msiiwa, Jeffrey Paine, Kevin Zhu, Sunishchal Dev, Vasu Sharma, Ahan M R - [[pdf]](https://arxiv.org/pdf/2510.18170)</summary>

**Abstract:** Goal changes are a defining feature of real world multi-turn interactions, yet current agent benchmarks primarily evaluate static objectives or one-shot tool use. We introduce AgentChangeBench, a benchmark explicitly designed to measure how tool augmented language model agents adapt to mid dialogue goal shifts across three enterprise domains. Our framework formalizes evaluation through four complementary metrics: Task Success Rate (TSR) for effectiveness, Tool Use Efficiency (TUE) for reliability, Tool Call Redundancy Rate (TCRR) for wasted effort, and Goal-Shift Recovery Time (GSRT) for adaptation latency. AgentChangeBench comprises 2,835 task sequences and five user personas, each designed to trigger realistic shift points in ongoing workflows. Using this setup, we evaluate several frontier models and uncover sharp contrasts obscured by traditional $\text{pass}@k$ scores: for example, GPT-4o reaches $92.2\%$ recovery on airline booking shifts while Gemini collapses to $48.6\%$, and retail tasks show near perfect parameter validity yet redundancy rates above $80\%$, revealing major inefficiencies. These findings demonstrate that high raw accuracy does not imply robustness under dynamic goals, and that explicit measurement of recovery time and redundancy is essential. AgentChangeBench establishes a reproducible testbed for diagnosing and improving agent resilience in realistic enterprise settings.

**arXiv ID:** 2510.18170
</details>

<details>
<summary><strong>Memory-Augmented State Machine Prompting: A Novel LLM Agent Framework for Real-Time Strategy Games</strong> - Runnan Qi, Yanan Ni, Lumin Jiang, Zongyuan Li, Kuihua Huang, Xian Guo - [[pdf]](https://arxiv.org/pdf/2510.18395)</summary>

**Abstract:** This paper proposes Memory-Augmented State Machine Prompting (MASMP), a novel framework for LLM agents in real-time strategy games. Addressing key challenges like hallucinations and fragmented decision-making in existing approaches, MASMP integrates state machine prompting with memory mechanisms to unify structured actions with long-term tactical coherence. The framework features: (1) a natural language-driven state machine architecture that guides LLMs to emulate finite state machines and behavior trees through prompts, and (2) a lightweight memory module preserving strategic variables (e.g., tactics, priority units) across decision cycles. Experiments in StarCraft II demonstrate MASMP's 60% win rate against the hardest built-in AI (Lv7), vastly outperforming baselines (0%). Case studies reveal the method retains LLMs' semantic comprehension while resolving the "Knowing-Doing Gap" through strict state-action mapping, achieving both interpretability and FSM-like reliability. This work establishes a new paradigm for combining neural and symbolic AI in complex decision-making.

**arXiv ID:** 2510.18395
</details>

<details>
<summary><strong>Probabilistic Modeling of Intentions in Socially Intelligent LLM Agents</strong> - Feifan Xia, Yuyang Fang, Defang Li, Yantong Xie, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang - [[pdf]](https://arxiv.org/pdf/2510.18476)</summary>

**Abstract:** We present a probabilistic intent modeling framework for large language model (LLM) agents in multi-turn social dialogue. The framework maintains a belief distribution over a partner's latent intentions, initialized from contextual priors and dynamically updated through likelihood estimation after each utterance. The evolving distribution provides additional contextual grounding for the policy, enabling adaptive dialogue strategies under uncertainty. Preliminary experiments in the SOTOPIA environment show consistent improvements: the proposed framework increases the Overall score by 9.0% on SOTOPIA-All and 4.1% on SOTOPIA-Hard compared with the Qwen2.5-7B baseline, and slightly surpasses an oracle agent that directly observes partner intentions. These early results suggest that probabilistic intent modeling can contribute to the development of socially intelligent LLM agents.

**arXiv ID:** 2510.18476
</details>

<details>
<summary><strong>Crucible: Quantifying the Potential of Control Algorithms through LLM Agents</strong> - Lianchen Jia, Chaoyang Li, Qian Houde, Tianchi Huang, Jiangchuan Liu, Lifeng Sun - [[pdf]](https://arxiv.org/pdf/2510.18491)</summary>

**Abstract:** Control algorithms in production environments typically require domain experts to tune their parameters and logic for specific scenarios. However, existing research predominantly focuses on algorithmic performance under ideal or default configurations, overlooking the critical aspect of Tuning Potential. To bridge this gap, we introduce Crucible, an agent that employs an LLM-driven, multi-level expert simulation to turn algorithms and defines a formalized metric to quantitatively evaluate their Tuning Potential. We demonstrate Crucible's effectiveness across a wide spectrum of case studies, from classic control tasks to complex computer systems, and validate its findings in a real-world deployment. Our experimental results reveal that Crucible systematically quantifies the tunable space across different algorithms. Furthermore, Crucible provides a new dimension for algorithm analysis and design, which ultimately leads to performance improvements. Our code is available at this https URL.

**arXiv ID:** 2510.18491
</details>

<details>
<summary><strong>DrunkAgent: Stealthy Memory Corruption in LLM-Powered Recommender Agents</strong> - Shiyi Yang, Zhibo Hu, Xinshu Li, Chen Wang, Tong Yu, Xiwei Xu, Liming Zhu, Lina Yao - [[pdf]](https://arxiv.org/pdf/2503.23804)</summary>

**Abstract:** Large language model (LLM)-powered agents are increasingly used in recommender systems (RSs) to achieve personalized behavior modeling, where the memory mechanism plays a pivotal role in enabling the agents to autonomously explore, learn and self-evolve from real-world interactions. However, this very mechanism, serving as a contextual repository, inherently exposes an attack surface for potential adversarial manipulations. Despite its central role, the robustness of agentic RSs in the face of such threats remains largely underexplored. Previous works suffer from semantic mismatches or rely on static embeddings or pre-defined prompts, all of which are not designed for dynamic systems, especially for dynamic memory states of LLM agents. This challenge is exacerbated by the black-box nature of commercial recommenders.
To tackle the above problems, in this paper, we present the first systematic investigation of memory-based vulnerabilities in LLM-powered recommender agents, revealing their security limitations and guiding efforts to strengthen system resilience and trustworthiness. Specifically, we propose a novel black-box attack framework named DrunkAgent. DrunkAgent crafts semantically meaningful adversarial textual triggers for target item promotions and introduces a series of strategies to maximize the trigger effect by corrupting the memory updates during the interactions. The triggers and strategies are optimized on a surrogate model, enabling DrunkAgent transferable and stealthy. Extensive experiments on real-world datasets across diverse agentic RSs, including collaborative filtering, retrieval augmentation and sequential recommendations, demonstrate the generalizability, transferability and stealthiness of DrunkAgent.

**arXiv ID:** 2503.23804
</details>

<details>
<summary><strong>Does Reasoning Help LLM Agents Play Dungeons and Dragons? A Prompt Engineering Experiment</strong> - Patricia Delafuente, Arya Honraopatil, Lara J. Martin - [[pdf]](https://arxiv.org/pdf/2510.18112)</summary>

**Abstract:** This paper explores the application of Large Language Models (LLMs) and reasoning to predict Dungeons & Dragons (DnD) player actions and format them as Avrae Discord bot commands. Using the FIREBALL dataset, we evaluated a reasoning model, DeepSeek-R1-Distill-LLaMA-8B, and an instruct model, LLaMA-3.1-8B-Instruct, for command generation. Our findings highlight the importance of providing specific instructions to models, that even single sentence changes in prompts can greatly affect the output of models, and that instruct models are sufficient for this task compared to reasoning models.

**arXiv ID:** 2510.18112
</details>

<details>
<summary><strong>Search Self-play: Pushing the Frontier of Agent Capability without Supervision</strong> - Hongliang Lu, Yuhang Wen, Pengyu Cheng, Ruijin Ding, Haotian Xu, Jiaqi Guo, Chutian Wang, Haonan Chen, Xiaoxi Jiang, Guanjun Jiang - [[pdf]](https://arxiv.org/pdf/2510.18821)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has become the mainstream technique for training LLM agents. However, RLVR highly depends on well-crafted task queries and corresponding ground-truth answers to provide accurate rewards, which requires massive human efforts and hinders the RL scaling processes, especially under agentic scenarios. Although a few recent works explore task synthesis methods, the difficulty of generated agentic tasks can hardly be controlled to provide effective RL training advantages. To achieve agentic RLVR with higher scalability, we explore self-play training for deep search agents, in which the learning LLM utilizes multi-turn search engine calling and acts simultaneously as both a task proposer and a problem solver. The task proposer aims to generate deep search queries with well-defined ground-truth answers and increasing task difficulty. The problem solver tries to handle the generated search queries and output the correct answer predictions. To ensure that each generated search query has accurate ground truth, we collect all the searching results from the proposer's trajectory as external knowledge, then conduct retrieval-augmentation generation (RAG) to test whether the proposed query can be correctly answered with all necessary search documents provided. In this search self-play (SSP) game, the proposer and the solver co-evolve their agent capabilities through both competition and cooperation. With substantial experimental results, we find that SSP can significantly improve search agents' performance uniformly on various benchmarks without any supervision under both from-scratch and continuous RL training setups. The code is at this https URL.

**arXiv ID:** 2510.18821
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (27 papers)</h2></summary>

<details>
<summary><strong>LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior</strong> - Man-Lin Chu, Lucian Terhorst, Kadin Reed, Tom Ni, Weiwei Chen, Rongyu Lin - [[pdf]](https://arxiv.org/pdf/2510.18155)</summary>

**Abstract:** Simulating consumer decision-making is vital for designing and evaluating marketing strategies before costly real- world deployment. However, post-event analyses and rule-based agent-based models (ABMs) struggle to capture the complexity of human behavior and social interaction. We introduce an LLM-powered multi-agent simulation framework that models consumer decisions and social dynamics. Building on recent advances in large language model simulation in a sandbox envi- ronment, our framework enables generative agents to interact, express internal reasoning, form habits, and make purchasing decisions without predefined rules. In a price-discount marketing scenario, the system delivers actionable strategy-testing outcomes and reveals emergent social patterns beyond the reach of con- ventional methods. This approach offers marketers a scalable, low-risk tool for pre-implementation testing, reducing reliance on time-intensive post-event evaluations and lowering the risk of underperforming campaigns.

**arXiv ID:** 2510.18155
</details>

<details>
<summary><strong>OPTAGENT: Optimizing Multi-Agent LLM Interactions Through Verbal Reinforcement Learning for Enhanced Reasoning</strong> - Zhenyu Bi, Meng Lu, Yang Li, Swastik Roy, Weijie Guan, Morteza Ziyadi, Xuan Wang - [[pdf]](https://arxiv.org/pdf/2510.18032)</summary>

**Abstract:** Large Language Models (LLMs) have shown remarkable reasoning capabilities in mathematical and scientific tasks. To enhance complex reasoning, multi-agent systems have been proposed to harness the collective intelligence of LLM agents. However, existing collaboration structures are either predefined or rely on majority voting or round-table debates, which can suppress correct but less dominant agent contributions. Recent approaches model multi-agent systems as graph networks but optimize purely for agent performance, neglecting the quality of interactions. We hypothesize that effective agent communication is crucial for multi-agent reasoning and that debating quality plays a significant role. To address this, we propose $\ours$, a multi-agent verbal reinforcement learning algorithm that dynamically constructs and refines multi-agent collaboration structures. Our method defines action spaces and a feedback mechanism that evaluates communication robustness and coherence throughout the debate. The final decision is achieved through a majority vote over all the agents. We assess $\ours$ on various reasoning tasks, including mathematical reasoning, creative writing, scientific reasoning, and numerical sorting. Results demonstrate that our approach significantly outperforms single-agent prompting methods and state-of-the-art multi-agent frameworks on diverse tasks.

**arXiv ID:** 2510.18032
</details>

<details>
<summary><strong>LAFA: Agentic LLM-Driven Federated Analytics over Decentralized Data Sources</strong> - Haichao Ji, Zibo Wang, Yifei Zhu, Meng han, Dan Wang, Zhu Han - [[pdf]](https://arxiv.org/pdf/2510.18477)</summary>

**Abstract:** Large Language Models (LLMs) have shown great promise in automating data analytics tasks by interpreting natural language queries and generating multi-operation execution plans. However, existing LLM-agent-based analytics frameworks operate under the assumption of centralized data access, offering little to no privacy protection. In contrast, federated analytics (FA) enables privacy-preserving computation across distributed data sources, but lacks support for natural language input and requires structured, machine-readable queries. In this work, we present LAFA, the first system that integrates LLM-agent-based data analytics with FA. LAFA introduces a hierarchical multi-agent architecture that accepts natural language queries and transforms them into optimized, executable FA workflows. A coarse-grained planner first decomposes complex queries into sub-queries, while a fine-grained planner maps each subquery into a Directed Acyclic Graph of FA operations using prior structural knowledge. To improve execution efficiency, an optimizer agent rewrites and merges multiple DAGs, eliminating redundant operations and minimizing computational and communicational overhead. Our experiments demonstrate that LAFA consistently outperforms baseline prompting strategies by achieving higher execution plan success rates and reducing resource-intensive FA operations by a substantial margin. This work establishes a practical foundation for privacy-preserving, LLM-driven analytics that supports natural language input in the FA setting.

**arXiv ID:** 2510.18477
</details>

<details>
<summary><strong>SOCIA-Nabla: Textual Gradient Meets Multi-Agent Orchestration for Automated Simulator Generation</strong> - Yuncheng Hua, Sion Weatherhead, Mehdi Jafari, Hao Xue, Flora D. Salim - [[pdf]](https://arxiv.org/pdf/2510.18551)</summary>

**Abstract:** In this paper, we present SOCIA-Nabla, an end-to-end, agentic framework that treats simulator construction asinstance optimization over code within a textual computation graph. Specialized LLM-driven agents are embedded as graph nodes, and a workflow manager executes a loss-driven loop: code synthesis -> execution -> evaluation -> code repair. The optimizer performs Textual-Gradient Descent (TGD), while human-in-the-loop interaction is reserved for task-spec confirmation, minimizing expert effort and keeping the code itself as the trainable object. Across three CPS tasks, i.e., User Modeling, Mask Adoption, and Personal Mobility, SOCIA-Nabla attains state-of-the-art overall accuracy. By unifying multi-agent orchestration with a loss-aligned optimization view, SOCIA-Nabla converts brittle prompt pipelines into reproducible, constraint-aware simulator code generation that scales across domains and simulation granularities. This work is under review, and we will release the code soon.

**arXiv ID:** 2510.18551
</details>

<details>
<summary><strong>QuantEvolve: Automating Quantitative Strategy Discovery through Multi-Agent Evolutionary Framework</strong> - Junhyeog Yun, Hyoun Jun Lee, Insu Jeon - [[pdf]](https://arxiv.org/pdf/2510.18569)</summary>

**Abstract:** Automating quantitative trading strategy development in dynamic markets is challenging, especially with increasing demand for personalized investment solutions. Existing methods often fail to explore the vast strategy space while preserving the diversity essential for robust performance across changing market conditions. We present QuantEvolve, an evolutionary framework that combines quality-diversity optimization with hypothesis-driven strategy generation. QuantEvolve employs a feature map aligned with investor preferences, such as strategy type, risk profile, turnover, and return characteristics, to maintain a diverse set of effective strategies. It also integrates a hypothesis-driven multi-agent system to systematically explore the strategy space through iterative generation and evaluation. This approach produces diverse, sophisticated strategies that adapt to both market regime shifts and individual investment needs. Empirical results show that QuantEvolve outperforms conventional baselines, validating its effectiveness. We release a dataset of evolved strategies to support future research.

**arXiv ID:** 2510.18569
</details>

<details>
<summary><strong>Multi-Agent Design Assistant for the Simulation of Inertial Fusion Energy</strong> - Meir H. Shachar, Dane M. Sterbentz, Harshitha Menon, Charles F. Jekel, M. Giselle Fernández-Godino, Yue Hao, Kevin Korner, Robert Rieben, Daniel A. White, William J. Schill, Jonathan L. Belof - [[pdf]](https://arxiv.org/pdf/2510.17830)</summary>

**Abstract:** Inertial fusion energy promises nearly unlimited, clean power if it can be achieved. However, the design and engineering of fusion systems requires controlling and manipulating matter at extreme energies and timescales; the shock physics and radiation transport governing the physical behavior under these conditions are complex requiring the development, calibration, and use of predictive multiphysics codes to navigate the highly nonlinear and multi-faceted design landscape. We hypothesize that artificial intelligence reasoning models can be combined with physics codes and emulators to autonomously design fusion fuel capsules. In this article, we construct a multi-agent system where natural language is utilized to explore the complex physics regimes around fusion energy. The agentic system is capable of executing a high-order multiphysics inertial fusion computational code. We demonstrate the capacity of the multi-agent design assistant to both collaboratively and autonomously manipulate, navigate, and optimize capsule geometry while accounting for high fidelity physics that ultimately achieve simulated ignition via inverse design.

**arXiv ID:** 2510.17830
</details>

<details>
<summary><strong>Modeling Layered Consciousness with Multi-Agent Large Language Models</strong> - Sang Hun Kim, Jongmin Lee, Dongkyu Park, So Young Lee, Yosep Chong - [[pdf]](https://arxiv.org/pdf/2510.17844)</summary>

**Abstract:** We propose a multi-agent framework for modeling artificial consciousness in large language models (LLMs), grounded in psychoanalytic theory. Our \textbf{Psychodynamic Model} simulates self-awareness, preconsciousness, and unconsciousness through agent interaction, guided by a Personalization Module combining fixed traits and dynamic needs. Using parameter-efficient fine-tuning on emotionally rich dialogues, the system was evaluated across eight personalized conditions. An LLM as a judge approach showed a 71.2\% preference for the fine-tuned model, with improved emotional depth and reduced output variance, demonstrating its potential for adaptive, personalized cognition.

**arXiv ID:** 2510.17844
</details>

<details>
<summary><strong>MAT-Agent: Adaptive Multi-Agent Training Optimization</strong> - Jusheng Zhang, Kaitong Cai, Yijia Fan, Ningyuan Liu, Keze Wang - [[pdf]](https://arxiv.org/pdf/2510.17845)</summary>

**Abstract:** Multi-label image classification demands adaptive training strategies to navigate complex, evolving visual-semantic landscapes, yet conventional methods rely on static configurations that falter in dynamic settings. We propose MAT-Agent, a novel multi-agent framework that reimagines training as a collaborative, real-time optimization process. By deploying autonomous agents to dynamically tune data augmentation, optimizers, learning rates, and loss functions, MAT-Agent leverages non-stationary multi-armed bandit algorithms to balance exploration and exploitation, guided by a composite reward harmonizing accuracy, rare-class performance, and training stability. Enhanced with dual-rate exponential moving average smoothing and mixed-precision training, it ensures robustness and efficiency. Extensive experiments across Pascal VOC, COCO, and VG-256 demonstrate MAT-Agent's superiority: it achieves an mAP of 97.4 (vs. 96.2 for PAT-T), OF1 of 92.3, and CF1 of 91.4 on Pascal VOC; an mAP of 92.8 (vs. 92.0 for HSQ-CvN), OF1 of 88.2, and CF1 of 87.1 on COCO; and an mAP of 60.9, OF1 of 70.8, and CF1 of 61.1 on VG-256. With accelerated convergence and robust cross-domain generalization, MAT-Agent offers a scalable, intelligent solution for optimizing complex visual models, paving the way for adaptive deep learning advancements.

**arXiv ID:** 2510.17845
</details>

<details>
<summary><strong>TACLA: An LLM-Based Multi-Agent Tool for Transactional Analysis Training in Education</strong> - Monika Zamojska, Jarosław A. Chudziak - [[pdf]](https://arxiv.org/pdf/2510.17913)</summary>

**Abstract:** Simulating nuanced human social dynamics with Large Language Models (LLMs) remains a significant challenge, particularly in achieving psychological depth and consistent persona behavior crucial for high-fidelity training tools. This paper introduces TACLA (Transactional Analysis Contextual LLM-based Agents), a novel Multi-Agent architecture designed to overcome these limitations. TACLA integrates core principles of Transactional Analysis (TA) by modeling agents as an orchestrated system of distinct Parent, Adult, and Child ego states, each with its own pattern memory. An Orchestrator Agent prioritizes ego state activation based on contextual triggers and an agent's life script, ensuring psychologically authentic responses. Validated in an educational scenario, TACLA demonstrates realistic ego state shifts in Student Agents, effectively modeling conflict de-escalation and escalation based on different teacher intervention strategies. Evaluation shows high conversational credibility and confirms TACLA's capacity to create dynamic, psychologically-grounded social simulations, advancing the development of effective AI tools for education and beyond.

**arXiv ID:** 2510.17913
</details>

<details>
<summary><strong>R2BC: Multi-Agent Imitation Learning from Single-Agent Demonstrations</strong> - Connor Mattson, Varun Raveendra, Ellen Novoseller, Nicholas Waytowich, Vernon J. Lawhern, Daniel S. Brown - [[pdf]](https://arxiv.org/pdf/2510.18085)</summary>

**Abstract:** Imitation Learning (IL) is a natural way for humans to teach robots, particularly when high-quality demonstrations are easy to obtain. While IL has been widely applied to single-robot settings, relatively few studies have addressed the extension of these methods to multi-agent systems, especially in settings where a single human must provide demonstrations to a team of collaborating robots. In this paper, we introduce and study Round-Robin Behavior Cloning (R2BC), a method that enables a single human operator to effectively train multi-robot systems through sequential, single-agent demonstrations. Our approach allows the human to teleoperate one agent at a time and incrementally teach multi-agent behavior to the entire system, without requiring demonstrations in the joint multi-agent action space. We show that R2BC methods match, and in some cases surpass, the performance of an oracle behavior cloning approach trained on privileged synchronized demonstrations across four multi-agent simulated tasks. Finally, we deploy R2BC on two physical robot tasks trained using real human demonstrations.

**arXiv ID:** 2510.18085
</details>

<details>
<summary><strong>From AutoRecSys to AutoRecLab: A Call to Build, Evaluate, and Govern Autonomous Recommender-Systems Research Labs</strong> - Joeran Beel, Bela Gipp, Tobias Vente, Moritz Baumgart, Philipp Meister - [[pdf]](https://arxiv.org/pdf/2510.18104)</summary>

**Abstract:** Recommender-systems research has accelerated model and evaluation advances, yet largely neglects automating the research process itself. We argue for a shift from narrow AutoRecSys tools -- focused on algorithm selection and hyper-parameter tuning -- to an Autonomous Recommender-Systems Research Lab (AutoRecLab) that integrates end-to-end automation: problem ideation, literature analysis, experimental design and execution, result interpretation, manuscript drafting, and provenance logging. Drawing on recent progress in automated science (e.g., multi-agent AI Scientist and AI Co-Scientist systems), we outline an agenda for the RecSys community: (1) build open AutoRecLab prototypes that combine LLM-driven ideation and reporting with automated experimentation; (2) establish benchmarks and competitions that evaluate agents on producing reproducible RecSys findings with minimal human input; (3) create review venues for transparently AI-generated submissions; (4) define standards for attribution and reproducibility via detailed research logs and metadata; and (5) foster interdisciplinary dialogue on ethics, governance, privacy, and fairness in autonomous research. Advancing this agenda can increase research throughput, surface non-obvious insights, and position RecSys to contribute to emerging Artificial Research Intelligence. We conclude with a call to organise a community retreat to coordinate next steps and co-author guidance for the responsible integration of automated research systems.

**arXiv ID:** 2510.18104
</details>

<details>
<summary><strong>Fetch.ai: An Architecture for Modern Multi-Agent Systems</strong> - Michael J. Wooldridge, Attila Bagoly, Jonathan J. Ward, Emanuele La Malfa, Gabriel Paludo Licks - [[pdf]](https://arxiv.org/pdf/2510.18699)</summary>

**Abstract:** Recent surges in LLM-driven intelligent systems largely overlook decades of foundational multi-agent systems (MAS) research, resulting in frameworks with critical limitations such as centralization and inadequate trust and communication protocols. This paper introduces the this http URL architecture, an industrial-strength platform designed to bridge this gap by facilitating the integration of classical MAS principles with modern AI capabilities. We present a novel, multi-layered solution built on a decentralized foundation of on-chain blockchain services for verifiable identity, discovery, and transactions. This is complemented by a comprehensive development framework for creating secure, interoperable agents, a cloud-based platform for deployment, and an intelligent orchestration layer where an agent-native LLM translates high-level human goals into complex, multi-agent workflows. We demonstrate the deployed nature of this system through a decentralized logistics use case where autonomous agents dynamically discover, negotiate, and transact with one another securely. Ultimately, the this http URL stack provides a principled architecture for moving beyond current agent implementations towards open, collaborative, and economically sustainable multi-agent ecosystems.

**arXiv ID:** 2510.18699
</details>

<details>
<summary><strong>Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making</strong> - Stelios Triantafyllou, Aleksa Sukovic, Yasaman Zolfimoselo, Goran Radanovic - [[pdf]](https://arxiv.org/pdf/2410.12539)</summary>

**Abstract:** We address the challenge of explaining counterfactual outcomes in multi-agent Markov decision processes. In particular, we aim to explain the total counterfactual effect of an agent's action on the outcome of a realized scenario through its influence on the environment dynamics and the agents' behavior. To achieve this, we introduce a novel causal explanation formula that decomposes the counterfactual effect by attributing to each agent and state variable a score reflecting their respective contributions to the effect. First, we show that the total counterfactual effect of an agent's action can be decomposed into two components: one measuring the effect that propagates through all subsequent agents' actions and another related to the effect that propagates through the state transitions. Building on recent advancements in causal contribution analysis, we further decompose these two effects as follows. For the former, we consider agent-specific effects -- a causal concept that quantifies the counterfactual effect of an agent's action that propagates through a subset of agents. Based on this notion, we use Shapley value to attribute the effect to individual agents. For the latter, we consider the concept of structure-preserving interventions and attribute the effect to state variables based on their "intrinsic" contributions. Through extensive experimentation, we demonstrate the interpretability of our approach in a Gridworld environment with LLM-assisted agents and a sepsis management simulator.

**arXiv ID:** 2410.12539
</details>

<details>
<summary><strong>VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning</strong> - Li Kang, Xiufeng Song, Heng Zhou, Yiran Qin, Jie Yang, Xiaohong Liu, Philip Torr, Lei Bai, Zhenfei Yin - [[pdf]](https://arxiv.org/pdf/2506.09049)</summary>

**Abstract:** Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems.

**arXiv ID:** 2506.09049
</details>

<details>
<summary><strong>An Automated Multi-modal Evaluation Framework for Mobile Intelligent Assistants Based on Large Language Models and Multi-Agent Collaboration</strong> - Meiping Wang, Jian Zhong, Rongduo Han, Liming Kang, Zhengkun Shi, Xiao Liang, Xing Lin, Nan Gao, Haining Zhang - [[pdf]](https://arxiv.org/pdf/2508.09507)</summary>

**Abstract:** With the rapid development of mobile intelligent assistant technologies, multi-modal AI assistants have become essential interfaces for daily user interactions. However, current evaluation methods face challenges including high manual costs, inconsistent standards, and subjective bias. This paper proposes an automated multi-modal evaluation framework based on large language models and multi-agent collaboration. The framework employs a three-tier agent architecture consisting of interaction evaluation agents, semantic verification agents, and experience decision agents. Through supervised fine-tuning on the Qwen3-8B model, we achieve a significant evaluation matching accuracy with human experts. Experimental results on eight major intelligent agents demonstrate the framework's effectiveness in predicting users' satisfaction and identifying generation defects.

**arXiv ID:** 2508.09507
</details>

<details>
<summary><strong>Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning</strong> - Song Yu, Xiaofei Xu, Ke Deng, Li Li, Lin Tian - [[pdf]](https://arxiv.org/pdf/2509.06436)</summary>

**Abstract:** Large language models (LLMs) face persistent challenges when handling long-context tasks, most notably the lost in the middle issue, where information located in the middle of a long input tends to be underutilized. Some existing methods that reduce input have the risk of discarding key information, while others that extend context windows often lead to attention dispersion. To address these limitations, we propose Tree of Agents (TOA), a multi-agent reasoning framework that segments the input into chunks processed by independent agents. Each agent generates its local cognition, then agents dynamically exchange information for collaborative reasoning along tree-structured paths. TOA enables agents to probe different reasoning orders for multi-perspective understanding, effectively mitigating position bias and reducing hallucinations. To improve processing efficiency, we incorporate prefix-hash caching and adaptive pruning strategies, achieving significant performance improvements with comparable API overhead. Experiments show that TOA, powered by compact LLaMA3.1-8B, significantly outperforms multiple baselines and demonstrates comparable performance to the latest and much larger commercial models, such as Gemini1.5-pro, on various long-context tasks. Code is available at this https URL.

**arXiv ID:** 2509.06436
</details>

<details>
<summary><strong>Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks</strong> - Trilok Padhi, Pinxian Lu, Abdulkadir Erol, Tanmay Sutar, Gauri Sharma, Mina Sonmez, Munmun De Choudhury, Ugur Kursuncu - [[pdf]](https://arxiv.org/pdf/2510.14207)</summary>

**Abstract:** Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.

**arXiv ID:** 2510.14207
</details>

<details>
<summary><strong>Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning</strong> - Rui Jerry Huang, Wendy Liu, Anastasia Miin - [[pdf]](https://arxiv.org/pdf/2510.18179)</summary>

**Abstract:** Inference-time computation is a critical yet challenging paradigm for enhancing the reasoning performance of large language models (LLMs). While existing strategies improve reasoning stability and consistency, they suffer from notable limitations: self-correction often reinforces the model's initial biases, and Multi-Agent Collaboration (MAC) often fails due to the lack of efficient coordination mechanisms, leading to collective errors. Although high-performing verifiers can detect reasoning errors, making them reliable requires substantial training. To address these challenges, we introduce a novel inference-time framework, Adaptive Coopetition (AdCo), in which LLM agents utilize an adaptive, UCB-based "coopetition" mechanism. At each round, agents leverage coarse verifier signals to determine whether to collaborate or compete, and iteratively refine their reasoning based on peer feedback. Without relying on high-performance verifiers, our adaptive strategy achieves significant performance gains on mathematical reasoning benchmarks, yielding a 20% relative improvement over baselines on the more challenging dataset. Our approach remains robust and consistent in terms of accuracy under different sample sizes and configurations. This adaptive, signal-guided "coopetition" framework enhances reasoning robustness by leveraging both model knowledge diversity and reasoning trace measures, while also promoting uncertainty-driven exploration, especially when participants have comparable capabilities. From this perspective, our work offers a fresh lens on inference-time computation and paves the way for more resilient multi-agent LLM systems. Our code is available at: this https URL.

**arXiv ID:** 2510.18179
</details>

<details>
<summary><strong>Socialized Learning and Emergent Behaviors in Multi-Agent Systems based on Multimodal Large Language Models</strong> - Sureyya Akin, Shruti T. Tiwari, Ram Bhattacharya, Sagar A. Raman, Kiran Mohanty, Sita Krishnan - [[pdf]](https://arxiv.org/pdf/2510.18515)</summary>

**Abstract:** This search introduces the Multimodal Socialized Learning Framework (M-S2L), designed to foster emergent social intelligence in AI agents by integrating Multimodal Large Language Models (M-LLMs) with social learning mechanisms. The framework equips agents with multimodal perception (vision and text) and structured action capabilities, enabling physical manipulation and grounded multimodal communication (e.g., text with visual pointers). M-S2L combines direct reinforcement learning with two novel social learning pathways: multimodal observational learning and communication-driven learning from feedback, augmented by an episodic memory system for long-term social context.
We evaluate M-S2L in a Collaborative Assembly Environment (CAE), where agent teams must construct complex devices from ambiguous blueprints under informational asymmetry. Across tasks of increasing complexity, M-S2L agents consistently outperform Text-Only and No-Social-Learning baselines in Task Completion Rate and Time to Completion, particularly in dynamic problem-solving scenarios. Ablation studies confirm the necessity of both multimodality and socialized learning. Our analysis reveals the emergence of efficient communication protocols integrating visual pointers with concise text, alongside rapid role specialization leading to stable labor division. Qualitative case studies demonstrate agents' abilities for shared awareness, dynamic re-planning, and adaptive problem-solving, suggesting a nascent form of machine social cognition. These findings indicate that integrating multimodal perception with explicit social learning is critical for developing human-like collaborative intelligence in multi-agent systems.

**arXiv ID:** 2510.18515
</details>

<details>
<summary><strong>Food4All: A Multi-Agent Framework for Real-time Free Food Discovery with Integrated Nutritional Metadata</strong> - Zhengqing Yuan, Yiyang Li, Weixiang Sun, Zheyuan Zhang, Kaiwen Shi, Keerthiram Murugesan, Yanfang Ye - [[pdf]](https://arxiv.org/pdf/2510.18289)</summary>

**Abstract:** Food insecurity remains a persistent public health emergency in the United States, tightly interwoven with chronic disease, mental illness, and opioid misuse. Yet despite the existence of thousands of food banks and pantries, access remains fragmented: 1) current retrieval systems depend on static directories or generic search engines, which provide incomplete and geographically irrelevant results; 2) LLM-based chatbots offer only vague nutritional suggestions and fail to adapt to real-world constraints such as time, mobility, and transportation; and 3) existing food recommendation systems optimize for culinary diversity but overlook survival-critical needs of food-insecure populations, including immediate proximity, verified availability, and contextual barriers. These limitations risk leaving the most vulnerable individuals, those experiencing homelessness, addiction, or digital illiteracy, unable to access urgently needed resources. To address this, we introduce Food4All, the first multi-agent framework explicitly designed for real-time, context-aware free food retrieval. Food4All unifies three innovations: 1) heterogeneous data aggregation across official databases, community platforms, and social media to provide a continuously updated pool of food resources; 2) a lightweight reinforcement learning algorithm trained on curated cases to optimize for both geographic accessibility and nutritional correctness; and 3) an online feedback loop that dynamically adapts retrieval policies to evolving user needs. By bridging information acquisition, semantic analysis, and decision support, Food4All delivers nutritionally annotated and guidance at the point of need. This framework establishes an urgent step toward scalable, equitable, and intelligent systems that directly support populations facing food insecurity and its compounding health risks.

**arXiv ID:** 2510.18289
</details>

<details>
<summary><strong>Static Sandboxes Are Inadequate: Modeling Societal Complexity Requires Open-Ended Co-Evolution in LLM-Based Multi-Agent Simulations</strong> - Jinkun Chen, Sher Badshah, Xuemin Yu, Sijia Han - [[pdf]](https://arxiv.org/pdf/2510.13982)</summary>

**Abstract:** What if artificial agents could not just communicate, but also evolve, adapt, and reshape their worlds in ways we cannot fully predict? With llm now powering multi-agent systems and social simulations, we are witnessing new possibilities for modeling open-ended, ever-changing environments. Yet, most current simulations remain constrained within static sandboxes, characterized by predefined tasks, limited dynamics, and rigid evaluation criteria. These limitations prevent them from capturing the complexity of real-world societies. In this paper, we argue that static, task-specific benchmarks are fundamentally inadequate and must be rethought. We critically review emerging architectures that blend llm with multi-agent dynamics, highlight key hurdles such as balancing stability and diversity, evaluating unexpected behaviors, and scaling to greater complexity, and introduce a fresh taxonomy for this rapidly evolving field. Finally, we present a research roadmap centered on open-endedness, continuous co-evolution, and the development of resilient, socially aligned AI ecosystems. We call on the community to move beyond static paradigms and help shape the next generation of adaptive, socially-aware multi-agent simulations.

**arXiv ID:** 2510.13982
</details>

<details>
<summary><strong>Stop Reducing Responsibility in LLM-Powered Multi-Agent Systems to Local Alignment</strong> - Jinwei Hu, Yi Dong, Shuang Ao, Zhuoyun Li, Boxuan Wang, Lokesh Singh, Guangliang Cheng, Sarvapali D. Ramchurn, Xiaowei Huang - [[pdf]](https://arxiv.org/pdf/2510.14008)</summary>

**Abstract:** LLM-powered Multi-Agent Systems (LLM-MAS) unlock new potentials in distributed reasoning, collaboration, and task generalization but also introduce additional risks due to unguaranteed agreement, cascading uncertainty, and adversarial vulnerabilities. We argue that ensuring responsible behavior in such systems requires a paradigm shift: from local, superficial agent-level alignment to global, systemic agreement. We conceptualize responsibility not as a static constraint but as a lifecycle-wide property encompassing agreement, uncertainty, and security, each requiring the complementary integration of subjective human-centered values and objective verifiability. Furthermore, a dual-perspective governance framework that combines interdisciplinary design with human-AI collaborative oversight is essential for tracing and ensuring responsibility throughout the lifecycle of LLM-MAS. Our position views LLM-MAS not as loose collections of agents, but as unified, dynamic socio-technical systems that demand principled mechanisms to support each dimension of responsibility and enable ethically aligned, verifiably coherent, and resilient behavior for sustained, system-wide agreement.

**arXiv ID:** 2510.14008
</details>

<details>
<summary><strong>Disaster Management in the Era of Agentic AI Systems: A Vision for Collective Human-Machine Intelligence for Augmented Resilience</strong> - Bo Li, Junwei Ma, Kai Yin, Yiming Xiao, Chia-Wei Hsu, Ali Mostafavi - [[pdf]](https://arxiv.org/pdf/2510.16034)</summary>

**Abstract:** The escalating frequency and severity of disasters routinely overwhelm traditional response capabilities, exposing critical vulnerability in disaster management. Current practices are hindered by fragmented data streams, siloed technologies, resource constraints, and the erosion of institutional memory, which collectively impede timely and effective decision making. This study introduces Disaster Copilot, a vision for a multi-agent artificial intelligence system designed to overcome these systemic challenges by unifying specialized AI tools within a collaborative framework. The proposed architecture utilizes a central orchestrator to coordinate diverse sub-agents, each specializing in critical domains such as predictive risk analytics, situational awareness, and impact assessment. By integrating multi-modal data, the system delivers a holistic, real-time operational picture and serve as the essential AI backbone required to advance Disaster Digital Twins from passive models to active, intelligent environments. Furthermore, it ensures functionality in resource-limited environments through on-device orchestration and incorporates mechanisms to capture institutional knowledge, mitigating the impact of staff turnover. We detail the system architecture and propose a three-phased roadmap emphasizing the parallel growth of technology, organizational capacity, and human-AI teaming. Disaster Copilot offers a transformative vision, fostering collective human-machine intelligence to build more adaptive, data-driven and resilient communities.

**arXiv ID:** 2510.16034
</details>

<details>
<summary><strong>Multi-Agent Collaboration via Evolving Orchestration</strong> - Yufan Dang, Chen Qian, Xueheng Luo, Jingru Fan, Zihao Xie, Ruijie Shi, Weize Chen, Cheng Yang, Xiaoyin Che, Ye Tian, Xuantang Xiong, Lei Han, Zhiyuan Liu, Maosong Sun - [[pdf]](https://arxiv.org/pdf/2505.19591)</summary>

**Abstract:** Large language models (LLMs) have achieved remarkable results across diverse downstream tasks, but their monolithic nature restricts scalability and efficiency in complex problem-solving. While recent research explores multi-agent collaboration among LLMs, most approaches rely on static organizational structures that struggle to adapt as task complexity and agent numbers grow, resulting in coordination overhead and inefficiencies. To this end, we propose a puppeteer-style paradigm for LLM-based multi-agent collaboration, where a centralized orchestrator ("puppeteer") dynamically directs agents ("puppets") in response to evolving task states. This orchestrator is trained via reinforcement learning to adaptively sequence and prioritize agents, enabling flexible and evolvable collective reasoning. Experiments on closed- and open-domain scenarios show that this method achieves superior performance with reduced computational costs. Analyses further reveal that the key improvements consistently stem from the emergence of more compact, cyclic reasoning structures under the orchestrator's evolution. Our code is available at this https URL.

**arXiv ID:** 2505.19591
</details>

<details>
<summary><strong>ATL*AS: An Automata-Theoretic Approach and Tool for the Verification of Strategic Abilities in Multi-Agent Systems</strong> - Sofia Garcia de Blas Garcia-Alcalde, Francesco Belardinelli - [[pdf]](https://arxiv.org/pdf/2510.17306)</summary>

**Abstract:** We present two novel symbolic algorithms for model checking the Alternating-time Temporal Logic ATL*, over both the infinite-trace and the finite-trace semantics. In particular, for infinite traces we design a novel symbolic reduction to parity games. We implement both methods in the ATL*AS model checker and evaluate it using synthetic benchmarks as well as a cybersecurity scenario. Our results demonstrate that the symbolic approach significantly outperforms the explicit-state representation and we find that our parity-game-based algorithm offers a more scalable and efficient solution for infinite-trace verification, outperforming previously available tools. Our results also confirm that finite-trace model checking yields substantial performance benefits over infinite-trace verification. As such, we provide a comprehensive toolset for verifying multiagent systems against specifications in ATL*.

**arXiv ID:** 2510.17306
</details>

<details>
<summary><strong>Rethinking LLM Uncertainty: A Multi-Agent Approach to Estimating Black-Box Model Uncertainty</strong> - Yu Feng, Phu Mon Htut, Zheng Qi, Wei Xiao, Manuel Mager, Nikolaos Pappas, Kishaloy Halder, Yang Li, Yassine Benajiba, Dan Roth - [[pdf]](https://arxiv.org/pdf/2412.09572)</summary>

**Abstract:** Quantifying uncertainty in black-box LLMs is vital for reliable responses and scalable oversight. Existing methods, which gauge a model's uncertainty through evaluating self-consistency in responses to the target query, can be misleading: an LLM may confidently provide an incorrect answer to a target query, yet give a confident and accurate answer to that same target query when answering a knowledge-preserving perturbation of the query. We systematically analyze the model behaviors and demonstrate that this discrepancy stems from suboptimal retrieval of parametric knowledge, often due to contextual biases that prevent consistent access to stored knowledge. We then introduce DiverseAgentEntropy, a novel, theoretically-grounded method employing multi-agent interaction across diverse query variations for uncertainty estimation of black-box LLMs. This approach more accurately assesses an LLM's true uncertainty and improves hallucination detection, outperforming existing self-consistency based techniques.

**arXiv ID:** 2412.09572
</details>

<details>
<summary><strong>Wonder Wins Ways: Curiosity-Driven Exploration through Multi-Agent Contextual Calibration</strong> - Yiyuan Pan, Zhe Liu, Hesheng Wang - [[pdf]](https://arxiv.org/pdf/2509.20648)</summary>

**Abstract:** Autonomous exploration in complex multi-agent reinforcement learning (MARL) with sparse rewards critically depends on providing agents with effective intrinsic motivation. While artificial curiosity offers a powerful self-supervised signal, it often confuses environmental stochasticity with meaningful novelty. Moreover, existing curiosity mechanisms exhibit a uniform novelty bias, treating all unexpected observations equally. However, peer behavior novelty, which encode latent task dynamics, are often overlooked, resulting in suboptimal exploration in decentralized, communication-free MARL settings. To this end, inspired by how human children adaptively calibrate their own exploratory behaviors via observing peers, we propose a novel approach to enhance multi-agent exploration. We introduce CERMIC, a principled framework that empowers agents to robustly filter noisy surprise signals and guide exploration by dynamically calibrating their intrinsic curiosity with inferred multi-agent context. Additionally, CERMIC generates theoretically-grounded intrinsic rewards, encouraging agents to explore state transitions with high information gain. We evaluate CERMIC on benchmark suites including VMAS, Meltingpot, and SMACv2. Empirical results demonstrate that exploration with CERMIC significantly outperforms SoTA algorithms in sparse-reward environments.

**arXiv ID:** 2509.20648
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming</strong> - Zheng Zhang, Jiarui He, Yuchen Cai, Deheng Ye, Peilin Zhao, Ruili Feng, Hao Wang - [[pdf]](https://arxiv.org/pdf/2510.18314)</summary>

**Abstract:** As large language model (LLM) agents increasingly automate complex web tasks, they boost productivity while simultaneously introducing new security risks. However, relevant studies on web agent attacks remain limited. Existing red-teaming approaches mainly rely on manually crafted attack strategies or static models trained offline. Such methods fail to capture the underlying behavioral patterns of web agents, making it difficult to generalize across diverse environments. In web agent attacks, success requires the continuous discovery and evolution of attack strategies. To this end, we propose Genesis, a novel agentic framework composed of three modules: Attacker, Scorer, and Strategist. The Attacker generates adversarial injections by integrating the genetic algorithm with a hybrid strategy representation. The Scorer evaluates the target web agent's responses to provide feedback. The Strategist dynamically uncovers effective strategies from interaction logs and compiles them into a continuously growing strategy library, which is then re-deployed to enhance the Attacker's effectiveness. Extensive experiments across various web tasks show that our framework discovers novel strategies and consistently outperforms existing attack baselines.

**arXiv ID:** 2510.18314
</details>

<details>
<summary><strong>LLM Assisted Alpha Fairness for 6 GHz WiFi and NR_U Coexistence: An Agentic Orchestrator for Throughput, Energy, and SLA</strong> - Qun Wang, Yingzhou Lu, Guiran Liu, Binrong Zhu, Yang Liu - [[pdf]](https://arxiv.org/pdf/2510.17814)</summary>

**Abstract:** Unlicensed 6GHz is becoming a primary workhorse for high-capacity access, with Wi-Fi and 5G NR-U competing for the same channels under listen-before-talk (LBT) rules. Operating in this regime requires decisions that jointly trade throughput, energy, and service-level objectives while remaining safe and auditable. We present an agentic controller that separates {policy} from {execution}. At the start of each scheduling epoch the agent summarizes telemetry (per-channel busy and baseline LBT failure; per-user CQI, backlog, latency, battery, priority, and power mode) and invokes a large language model (LLM) to propose a small set of interpretable knobs: a fairness index \alpha, per-channel duty-cycle caps for Wi-Fi/NR-U, and class weights. A deterministic optimizer then enforces feasibility and computes an \alpha-fair allocation that internalizes LBT losses and energy cost; malformed or unsafe policies are clamped and fall back to a rule baseline. In a 6GHz simulator with two 160MHz channels and mixed Wi-Fi/NR-U users, LLM-assisted policies consistently improve energy efficiency while keeping throughput competitive with a strong rule baseline. One LLM lowers total energy by 35.3% at modest throughput loss, and another attains the best overall trade-off, finishing with higher total bits (+3.5%) and higher bits/J (+12.2%) than the baseline. We release code, per-epoch logs, and plotting utilities to reproduce all figures and numbers, illustrating how transparent, policy-level LLM guidance can safely improve wireless coexistence.

**arXiv ID:** 2510.17814
</details>

<details>
<summary><strong>SafeCoop: Unravelling Full Stack Safety in Agentic Collaborative Driving</strong> - Xiangbo Gao, Tzu-Hsiang Lin, Ruojing Song, Yuheng Wu, Kuan-Ru Huang, Zicheng Jin, Fangzhou Lin, Shinan Liu, Zhengzhong Tu - [[pdf]](https://arxiv.org/pdf/2510.18123)</summary>

**Abstract:** Collaborative driving systems leverage vehicle-to-everything (V2X) communication across multiple agents to enhance driving safety and efficiency. Traditional V2X systems take raw sensor data, neural features, or perception results as communication media, which face persistent challenges, including high bandwidth demands, semantic loss, and interoperability issues. Recent advances investigate natural language as a promising medium, which can provide semantic richness, decision-level reasoning, and human-machine interoperability at significantly lower bandwidth. Despite great promise, this paradigm shift also introduces new vulnerabilities within language communication, including message loss, hallucinations, semantic manipulation, and adversarial attacks. In this work, we present the first systematic study of full-stack safety and security issues in natural-language-based collaborative driving. Specifically, we develop a comprehensive taxonomy of attack strategies, including connection disruption, relay/replay interference, content spoofing, and multi-connection forgery. To mitigate these risks, we introduce an agentic defense pipeline, which we call SafeCoop, that integrates a semantic firewall, language-perception consistency checks, and multi-source consensus, enabled by an agentic transformation function for cross-frame spatial alignment. We systematically evaluate SafeCoop in closed-loop CARLA simulation across 32 critical scenarios, achieving 69.15% driving score improvement under malicious attacks and up to 67.32% F1 score for malicious detection. This study provides guidance for advancing research on safe, secure, and trustworthy language-driven collaboration in transportation systems. Our project page is this https URL.

**arXiv ID:** 2510.18123
</details>

<details>
<summary><strong>PowerChain: A Verifiable Agentic AI System for Automating Distribution Grid Analyses</strong> - Emmanuel O. Badmus, Peng Sang, Dimitrios Stamoulis, Amritanshu Pandey - [[pdf]](https://arxiv.org/pdf/2508.17094)</summary>

**Abstract:** Rapid electrification and decarbonization are increasing the complexity of distribution grid (DG) operation and planning, necessitating advanced computational analyses to ensure reliability and resilience. These analyses depend on disparate workflows comprising complex models, function calls, and data pipelines that require substantial expert knowledge and remain difficult to automate. Workforce and budget constraints further limit utilities' ability to apply such analyses at scale. To address this gap, we build an agentic system PowerChain, which is capable of autonomously performing complex grid analyses. Existing agentic AI systems are typically developed in a bottom-up manner with customized context for predefined analysis tasks; therefore, they do not generalize to tasks that the agent has never seen. In comparison, to generalize to unseen DG analysis tasks, PowerChain dynamically generates structured context by leveraging supervisory signals from self-contained power systems tools (e.g., GridLAB-D) and an optimized set of expert-annotated and verified reasoning trajectories. For complex DG tasks defined in natural language, empirical results on real utility data demonstrate that PowerChain achieves up to a 144/% improvement in performance over baselines.

**arXiv ID:** 2508.17094
</details>

<details>
<summary><strong>When Agents go Astray: Course-Correcting SWE Agents with PRMs</strong> - Shubham Gandhi, Jason Tsay, Jatin Ganhotra, Kiran Kate, Yara Rizk - [[pdf]](https://arxiv.org/pdf/2509.02360)</summary>

**Abstract:** Large Language Model (LLM) agents are increasingly deployed for complex, multi-step software engineering (SWE) tasks. However, their trajectories often contain costly inefficiencies, such as redundant exploration, looping, and failure to terminate once a solution is reached. Prior work has largely treated these errors in a post-hoc manner, diagnosing failures only after execution. In this paper, we introduce SWE-PRM, an inference-time Process Reward Model (PRM) that intervenes during execution to detect and course-correct trajectory-level errors. Our PRM design leverages a taxonomy of common inefficiencies and delivers lightweight, interpretable feedback without modifying the underlying policy. On SWE-bench Verified, closed-source PRMs improve resolution from 40.0% to 50.6% (+10.6 p.p.), with the largest gains on medium and hard tasks. Among feedback strategies, taxonomy-guided PRMs outperform unguided or explicit action-prescriptive variants, increasing success rate while reducing trajectory length. These benefits come at an acceptable added inference cost of as low as $0.2, making PRMs a practical and scalable mechanism for improving SWE agents' reliability and efficiency.

**arXiv ID:** 2509.02360
</details>

<details>
<summary><strong>From Agent Simulation to Social Simulator: A Comprehensive Review (Part 1)</strong> - Xiao Xue, Deyu Zhou, Ming Zhang, Fei-Yue Wang - [[pdf]](https://arxiv.org/pdf/2510.18271)</summary>

**Abstract:** This is the first part of the comprehensive review, focusing on the historical development of Agent-Based Modeling (ABM) and its classic cases. It begins by discussing the development history and design principles of Agent-Based Modeling (ABM), helping readers understand the significant challenges that traditional physical simulation methods face in the social domain. Then, it provides a detailed introduction to foundational models for simulating social systems, including individual models, environmental models, and rule-based models. Finally, it presents classic cases of social simulation, covering three types: thought experiments, mechanism exploration, and parallel optimization.

**arXiv ID:** 2510.18271
</details>

<details>
<summary><strong>IASC: Interactive Agentic System for ConLangs</strong> - Chihiro Taguchi, Richard Sproat - [[pdf]](https://arxiv.org/pdf/2510.07591)</summary>

**Abstract:** We present a system that uses LLMs as a tool in the development of Constructed Languages. The system is modular in that one first creates a target phonology for the language using an agentic approach that refines its output at each step with commentary feedback on its previous attempt. Next, a set of sentences is 'translated' from their English original into a morphosyntactic markup that reflects the word order and morphosyntactic feature specifications of the desired target language, with affixes represented as morphosyntactic feature bundles. From this translated corpus, a lexicon is constructed using the phonological model and the set of morphemes (stems and affixes) extracted from the 'translated' sentences. The system is then instructed to provide an orthography for the language, using an existing script such as Latin or Cyrillic. Finally, the system writes a brief grammatical handbook of the language. The system can also translate further sentences into the target language.
Our goal is twofold. First, we hope that these tools will be fun to use for creating artificially constructed languages. Second, we are interested in exploring what LLMs 'know' about language-not what they know about any particular language or linguistic phenomenon, but how much they know about and understand language and linguistic concepts. As we shall see, there is a fairly wide gulf in capabilities both among different LLMs and among different linguistic specifications, with it being notably easier for systems to deal with more common patterns than rarer ones. An additional avenue that we explore is the application of our approach to translating from high-resource into low-resource languages. While the results so far are mostly negative, we provide some evidence that an improved version of the present system could afford some real gains in such tasks.
this https URL

**arXiv ID:** 2510.07591
</details>

<details>
<summary><strong>A Compositional Paradigm for Foundation Models: Towards Smarter Robotic Agents</strong> - Luigi Quarantiello, Elia Piccoli, Jack Bell, Malio Li, Giacomo Carfì, Eric Nuertey Coleman, Gerlando Gramaglia, Lanpei Li, Mauro Madeddu, Irene Testa, Vincenzo Lomonaco - [[pdf]](https://arxiv.org/pdf/2510.18608)</summary>

**Abstract:** The birth of Foundation Models brought unprecedented results in a wide range of tasks, from language to vision, to robotic control. These models are able to process huge quantities of data, and can extract and develop rich representations, which can be employed across different domains and modalities. However, they still have issues in adapting to dynamic, real-world scenarios without retraining the entire model from scratch. In this work, we propose the application of Continual Learning and Compositionality principles to foster the development of more flexible, efficient and smart AI solutions.

**arXiv ID:** 2510.18608
</details>

<details>
<summary><strong>Distilling LLM Prior to Flow Model for Generalizable Agent's Imagination in Object Goal Navigation</strong> - Badi Li, Ren-jie Lu, Yu Zhou, Jingke Meng, Wei-shi Zheng - [[pdf]](https://arxiv.org/pdf/2508.09423)</summary>

**Abstract:** The Object Goal Navigation (ObjectNav) task challenges agents to locate a specified object in an unseen environment by imagining unobserved regions of the scene. Prior approaches rely on deterministic and discriminative models to complete semantic maps, overlooking the inherent uncertainty in indoor layouts and limiting their ability to generalize to unseen environments. In this work, we propose GOAL, a generative flow-based framework that models the semantic distribution of indoor environments by bridging observed regions with LLM-enriched full-scene semantic maps. During training, spatial priors inferred from large language models (LLMs) are encoded as two-dimensional Gaussian fields and injected into target maps, distilling rich contextual knowledge into the flow model and enabling more generalizable completions. Extensive experiments demonstrate that GOAL achieves state-of-the-art performance on MP3D and Gibson, and shows strong generalization in transfer settings to HM3D. Codes and pretrained models are available at this https URL.

**arXiv ID:** 2508.09423
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (36 papers)</h2></summary>

<details>
<summary><strong>FABRIC: Framework for Agent-Based Realistic Intelligence Creation</strong> - Abhigya Verma, Seganrasan Subramanian, Nandhakumar Kandasamy, Naman Gupta - [[pdf]](https://arxiv.org/pdf/2510.17995)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as agents, expected to decompose goals, invoke tools, and verify results in dynamic environments. Realizing these capabilities requires access to agentic data- structured interaction records that couple user intents with tool specifications, argument-grounded calls, and verifiable execution traces. However, collecting such data from human annotators is costly, time-consuming, and difficult to scale.
We present a unified framework for synthesizing agentic data using only LLMs, without any human-in-the-loop supervision. This framework decomposes generation into modular pipelines that produce complete interaction records spanning task specifications, tool definitions, policy pseudocode, natural language exchanges, and execution traces. Records conform to strict syntactic and semantic constraints, ensuring machine-parseability and faithful alignment across inputs, outputs, and tool calls.
Beyond single tasks, there is support for both multi-task and multi-turn agent interactions, enabling the construction of datasets that reflect the full spectrum of tool-use competencies. To ensure quality and consistency, the framework integrates constrained generation formats, JSON-schema validation, and judge-based filtering.
This paper formalizes the schema for agentic records, details the prompt design principles that guide generation, and introduces scalable pipelines for high-quality synthetic data. By providing a reproducible, LLM-only alternative to manual collection, hence advancing the development of agentic LLMs capable of robust tool use.

**arXiv ID:** 2510.17995
</details>

<details>
<summary><strong>Med-VRAgent: A Framework for Medical Visual Reasoning-Enhanced Agents</strong> - Guangfu Guo, Xiaoqian Lu, Yue Feng - [[pdf]](https://arxiv.org/pdf/2510.18424)</summary>

**Abstract:** Visual Language Models (VLMs) achieve promising results in medical reasoning but struggle with hallucinations, vague descriptions, inconsistent logic and poor localization. To address this, we propose a agent framework named Medical Visual Reasoning Agent (\textbf{Med-VRAgent}). The approach is based on Visual Guidance and Self-Reward paradigms and Monte Carlo Tree Search (MCTS). By combining the Visual Guidance with tree search, Med-VRAgent improves the medical visual reasoning capabilities of VLMs. We use the trajectories collected by Med-VRAgent as feedback to further improve the performance by fine-tuning the VLMs with the proximal policy optimization (PPO) objective. Experiments on multiple medical VQA benchmarks demonstrate that our method outperforms existing approaches.

**arXiv ID:** 2510.18424
</details>

<details>
<summary><strong>Rewarding the Journey, Not Just the Destination: A Composite Path and Answer Self-Scoring Reward Mechanism for Test-Time Reinforcement Learning</strong> - Chenwei Tang, Jingyu Xing, Xinyu Liu, Wei Ju, Jiancheng Lv, Deng Xiong, Ziyue Qiao - [[pdf]](https://arxiv.org/pdf/2510.17923)</summary>

**Abstract:** Reinforcement Learning (RL) has emerged as a powerful paradigm for advancing Large Language Models (LLMs), achieving remarkable performance in complex reasoning domains such as mathematics and code generation. However, current RL methods face a fundamental scalability bottleneck due to their heavy reliance on human-curated preference data or labeled datasets for reward modeling. To overcome this limitation, we explore RL on unlabeled data where models learn autonomously from continuous experience streams. The core challenge in this setting lies in reliable reward estimation without ground-truth supervision. Existing approaches like Test-Time RL address this through self-consistent consensus, but risk reinforcing incorrect pseudo-labels derived from majority voting. We introduce COMPASS (Composite Path and Answer Self-Scoring), a novel test-time reward mechanism that operates without external supervision. COMPASS integrates two complementary components: the Dual-Calibration Answer Reward (DCAR), which stabilizes training by establishing trustworthy pseudo-labels through confidence and credibility calibration, and the Decisive Path Reward (DPR), which directly optimizes the reasoning process quality beyond mere outcome supervision. By jointly reinforcing trustworthy consensus answers and highly decisive reasoning chains, the COMPASS systematically enhances the model's analytical capabilities. Extensive experiments show that COMPASS achieves significant and consistent performance gains across diverse reasoning tasks and model architectures, advancing a more scalable direction for LLMs to learn from continuous experience.

**arXiv ID:** 2510.17923
</details>

<details>
<summary><strong>UniRL-Zero: Reinforcement Learning on Unified Models with Joint Language Model and Diffusion Model Experts</strong> - Fu-Yun Wang, Han Zhang, Michael Gharbi, Hongsheng Li, Taesung Park - [[pdf]](https://arxiv.org/pdf/2510.17937)</summary>

**Abstract:** We present UniRL-Zero, a unified reinforcement learning (RL) framework that boosts, multimodal language model understanding and reasoning, diffusion model multimedia generation, and their beneficial interaction capabilities within a unified model. Our work defines six scenarios for unified model reinforcement learning, providing systematic baselines for reinforcement learning of unified understanding and generation model. Our code is available at this https URL.

**arXiv ID:** 2510.17937
</details>

<details>
<summary><strong>R2L: Reliable Reinforcement Learning: Guaranteed Return & Reliable Policies in Reinforcement Learning</strong> - Nadir Farhi - [[pdf]](https://arxiv.org/pdf/2510.18074)</summary>

**Abstract:** In this work, we address the problem of determining reliable policies in reinforcement learning (RL), with a focus on optimization under uncertainty and the need for performance guarantees. While classical RL algorithms aim at maximizing the expected return, many real-world applications - such as routing, resource allocation, or sequential decision-making under risk - require strategies that ensure not only high average performance but also a guaranteed probability of success. To this end, we propose a novel formulation in which the objective is to maximize the probability that the cumulative return exceeds a prescribed threshold. We demonstrate that this reliable RL problem can be reformulated, via a state-augmented representation, into a standard RL problem, thereby allowing the use of existing RL and deep RL algorithms without the need for entirely new algorithmic frameworks. Theoretical results establish the equivalence of the two formulations and show that reliable strategies can be derived by appropriately adapting well-known methods such as Q-learning or Dueling Double DQN. To illustrate the practical relevance of the approach, we consider the problem of reliable routing, where the goal is not to minimize the expected travel time but rather to maximize the probability of reaching the destination within a given time budget. Numerical experiments confirm that the proposed formulation leads to policies that effectively balance efficiency and reliability, highlighting the potential of reliable RL for applications in stochastic and safety-critical environments.

**arXiv ID:** 2510.18074
</details>

<details>
<summary><strong>MENTOR: A Reinforcement Learning Framework for Model Enhancement via Teacher-Optimized Rewards in Small Models</strong> - ChangSu Choi, Hoyun Song, Dongyeon Kim, WooHyeon Jung, Minkyung Cho, Sunjin Park, NohHyeob Bae, Seona Yu, KyungTae Lim - [[pdf]](https://arxiv.org/pdf/2510.18383)</summary>

**Abstract:** Distilling the tool-using capabilities of large language models (LLMs) into smaller, more efficient small language models (SLMs) is a key challenge for their practical application. The predominant approach, supervised fine-tuning (SFT), suffers from poor generalization as it trains models to imitate a static set of teacher trajectories rather than learn a robust methodology. While reinforcement learning (RL) offers an alternative, the standard RL using sparse rewards fails to effectively guide SLMs, causing them to struggle with inefficient exploration and adopt suboptimal strategies. To address these distinct challenges, we propose MENTOR, a framework that synergistically combines RL with teacher-guided distillation. Instead of simple imitation, MENTOR employs an RL-based process to learn a more generalizable policy through exploration. In addition, to solve the problem of reward sparsity, it uses a teacher's reference trajectory to construct a dense, composite teacher-guided reward that provides fine-grained guidance. Extensive experiments demonstrate that MENTOR significantly improves the cross-domain generalization and strategic competence of SLMs compared to both SFT and standard sparse-reward RL baselines.

**arXiv ID:** 2510.18383
</details>

<details>
<summary><strong>Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options</strong> - Joongkyu Lee, Seouh-won Yi, Min-hwan Oh - [[pdf]](https://arxiv.org/pdf/2510.18713)</summary>

**Abstract:** We study online preference-based reinforcement learning (PbRL) with the goal of improving sample efficiency. While a growing body of theoretical work has emerged-motivated by PbRL's recent empirical success, particularly in aligning large language models (LLMs)-most existing studies focus only on pairwise comparisons. A few recent works (Zhu et al., 2023, Mukherjee et al., 2024, Thekumparampil et al., 2024) have explored using multiple comparisons and ranking feedback, but their performance guarantees fail to improve-and can even deteriorate-as the feedback length increases, despite the richer information available. To address this gap, we adopt the Plackett-Luce (PL) model for ranking feedback over action subsets and propose M-AUPO, an algorithm that selects multiple actions by maximizing the average uncertainty within the offered subset. We prove that M-AUPO achieves a suboptimality gap of $\tilde{\mathcal{O}}\left( \frac{d}{T} \sqrt{ \sum_{t=1}^T \frac{1}{|S_t|}} \right)$, where $T$ is the total number of rounds, $d$ is the feature dimension, and $|S_t|$ is the size of the subset at round $t$. This result shows that larger subsets directly lead to improved performance and, notably, the bound avoids the exponential dependence on the unknown parameter's norm, which was a fundamental limitation in most previous works. Moreover, we establish a near-matching lower bound of $\Omega \left( \frac{d}{K \sqrt{T}} \right)$, where $K$ is the maximum subset size. To the best of our knowledge, this is the first theoretical result in PbRL with ranking feedback that explicitly shows improved sample efficiency as a function of the subset size.

**arXiv ID:** 2510.18713
</details>

<details>
<summary><strong>Towards Faithful and Controllable Personalization via Critique-Post-Edit Reinforcement Learning</strong> - Chenghao Zhu, Meiling Tao, Tiannan Wang, Dongyi Ding, Yuchen Eleanor Jiang, Wangchunshu Zhou - [[pdf]](https://arxiv.org/pdf/2510.18849)</summary>

**Abstract:** Faithfully personalizing large language models (LLMs) to align with individual user preferences is a critical but challenging task. While supervised fine-tuning (SFT) quickly reaches a performance plateau, standard reinforcement learning from human feedback (RLHF) also struggles with the nuances of personalization. Scalar-based reward models are prone to reward hacking which leads to verbose and superficially personalized responses. To address these limitations, we propose Critique-Post-Edit, a robust reinforcement learning framework that enables more faithful and controllable personalization. Our framework integrates two key components: (1) a Personalized Generative Reward Model (GRM) that provides multi-dimensional scores and textual critiques to resist reward hacking, and (2) a Critique-Post-Edit mechanism where the policy model revises its own outputs based on these critiques for more targeted and efficient learning. Under a rigorous length-controlled evaluation, our method substantially outperforms standard PPO on personalization benchmarks. Personalized Qwen2.5-7B achieves an average 11\% win-rate improvement, and personalized Qwen2.5-14B model surpasses the performance of GPT-4.1. These results demonstrate a practical path to faithful, efficient, and controllable personalization.

**arXiv ID:** 2510.18849
</details>

<details>
<summary><strong>Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model</strong> - Ling Team, Anqi Shen, Baihui Li, Bin Hu, Bin Jing, Cai Chen, Chao Huang, Chao Zhang, Chaokun Yang, Cheng Lin, Chengyao Wen, Congqi Li, Deng Zhao, Dingbo Yuan, Donghai You, Fagui Mao, Fanzhuang Meng, Feng Xu, Guojie Li, Guowei Wang, Hao Dai, Haonan Zheng, Hong Liu, Jia Guo, Jiaming Liu, Jian Liu, Jianhao Fu, Jiannan Shi, Jianwen Wang, Jianxin Lai, Jin Yang, Jun Mei, Jun Zhou, Junbo Zhao, Junping Zhao, Kuan Xu, Le Su, Lei Chen, Li Tang, Liang Jiang, Liangcheng Fu, Lianhao Xu, Linfeng Shi, Lisha Liao, Longfei Zheng, Meng Li, Mingchun Chen, Qi Zuo, Qiang Cheng, Qianggang Cao, Qitao Shi, Quanrui Guo, Senlin Zhu, Shaofei Wang, Shaomian Zheng, Shuaicheng Li, Shuwei Gu, Siba Chen, Tao Wu, Tao Zhang, Tianyu Zhang, Tianyu Zhou, Tiwei Bie, Tongkai Yang, Wang Hong, Wang Ren, Weihua Chen, Wenbo Yu, Wengang Zheng, Xiangchun Wang, Xiaodong Yan, Xiaopei Wan, Xin Zhao, Xinyu Kong, Xinyu Tang, Xudong Han, Xudong Wang, Xuemin Yang, Xueyu Hu, Yalin Zhang, Yan Sun, Yicheng Shan, Yilong Wang, Yingying Xu, Yongkang Liu, Yongzhen Guo, Yuanyuan Wang, Yuchen Yan, Yuefan Wang, Yuhong Guo, Zehuan Li, Zhankai Xu, Zhe Li, Zhenduo Zhang, Zhengke Gui, Zhenxuan Pan, Zhenyu Huang, Zhenzhong Lan, Zhiqiang Ding, Zhiqiang Zhang - [[pdf]](https://arxiv.org/pdf/2510.18855)</summary>

**Abstract:** We present Ring-1T, the first open-source, state-of-the-art thinking model with a trillion-scale parameter. It features 1 trillion total parameters and activates approximately 50 billion per token. Training such models at a trillion-parameter scale introduces unprecedented challenges, including train-inference misalignment, inefficiencies in rollout processing, and bottlenecks in the RL system. To address these, we pioneer three interconnected innovations: (1) IcePop stabilizes RL training via token-level discrepancy masking and clipping, resolving instability from training-inference mismatches; (2) C3PO++ improves resource utilization for long rollouts under a token budget by dynamically partitioning them, thereby obtaining high time efficiency; and (3) ASystem, a high-performance RL framework designed to overcome the systemic bottlenecks that impede trillion-parameter model training. Ring-1T delivers breakthrough results across critical benchmarks: 93.4 on AIME-2025, 86.72 on HMMT-2025, 2088 on CodeForces, and 55.94 on ARC-AGI-v1. Notably, it attains a silver medal-level result on the IMO-2025, underscoring its exceptional reasoning capabilities. By releasing the complete 1T parameter MoE model to the community, we provide the research community with direct access to cutting-edge reasoning capabilities. This contribution marks a significant milestone in democratizing large-scale reasoning intelligence and establishes a new baseline for open-source model performance.

**arXiv ID:** 2510.18855
</details>

<details>
<summary><strong>A representational framework for learning and encoding structurally enriched trajectories in complex agent environments</strong> - Corina Catarau-Cotutiu, Esther Mondragon, Eduardo Alonso - [[pdf]](https://arxiv.org/pdf/2503.13194)</summary>

**Abstract:** The ability of artificial intelligence agents to make optimal decisions and generalise them to different domains and tasks is compromised in complex scenarios. One way to address this issue has focused on learning efficient representations of the world and on how the actions of agents affect them in state-action transitions. Whereas such representations are procedurally efficient, they lack structural richness. To address this problem, we propose to enhance the agent's ontology and extend the traditional conceptualisation of trajectories to provide a more nuanced view of task execution. Structurally Enriched Trajectories (SETs) extend the encoding of sequences of states and their transitions by incorporating hierarchical relations between objects, interactions, and affordances. SETs are built as multi-level graphs, providing a detailed representation of the agent dynamics and a transferable functional abstraction of the task. SETs are integrated into an architecture, Structurally Enriched Trajectory Learning and Encoding (SETLE), that employs a heterogeneous graph-based memory structure of multi-level relational dependencies essential for generalisation. We demonstrate that SETLE can support downstream tasks, enabling agents to recognise task relevant structural patterns across CREATE and MiniGrid environments. Finally, we integrate SETLE with reinforcement learning and show measurable improvements in downstream performance, including breakthrough success rates in complex, sparse-reward tasks.

**arXiv ID:** 2503.13194
</details>

<details>
<summary><strong>Pretraining a Shared Q-Network for Data-Efficient Offline Reinforcement Learning</strong> - Jongchan Park, Mingyu Park, Donghwan Lee - [[pdf]](https://arxiv.org/pdf/2505.05701)</summary>

**Abstract:** Offline reinforcement learning (RL) aims to learn a policy from a static dataset without further interactions with the environment. Collecting sufficiently large datasets for offline RL is exhausting since this data collection requires colossus interactions with environments and becomes tricky when the interaction with the environment is restricted. Hence, how an agent learns the best policy with a minimal static dataset is a crucial issue in offline RL, similar to the sample efficiency problem in online RL. In this paper, we propose a simple yet effective plug-and-play pretraining method to initialize a feature of a Q-network to enhance data efficiency in offline RL. Specifically, we introduce a shared Q-network structure that outputs predictions of the next state and Q-value. We pretrain the shared Q-network through a supervised regression task that predicts a next state and trains the shared Q-network using diverse offline RL methods. Through extensive experiments, we empirically demonstrate that our method enhances the performance of existing popular offline RL methods on the D4RL, Robomimic and V-D4RL benchmarks. Furthermore, we show that our method significantly boosts data-efficient offline RL across various data qualities and data distributions trough D4RL and ExoRL benchmarks. Notably, our method adapted with only 10% of the dataset outperforms standard algorithms even with full datasets.

**arXiv ID:** 2505.05701
</details>

<details>
<summary><strong>ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents</strong> - Hanyu Lai, Xiao Liu, Yanxiao Zhao, Han Xu, Hanchen Zhang, Bohao Jing, Yanyu Ren, Shuntian Yao, Yuxiao Dong, Jie Tang - [[pdf]](https://arxiv.org/pdf/2508.14040)</summary>

**Abstract:** We introduce ComputerRL, a framework for autonomous desktop intelligence that enables agents to operate complex digital workspaces skillfully. ComputerRL features the API-GUI paradigm, which unifies programmatic API calls and direct GUI interaction to address the inherent mismatch between machine agents and human-centric desktop environments. Scaling end-to-end RL training is crucial for improvement and generalization across diverse desktop tasks; however, it remains challenging due to environmental inefficiency and instability during extended training. To support scalable and robust training, we develop a distributed RL infrastructure capable of orchestrating thousands of parallel virtual desktop environments to accelerate large-scale online RL. Furthermore, we propose Entropulse, a training strategy that alternates reinforcement learning with supervised fine-tuning, effectively mitigating entropy collapse during extended training runs. We employ ComputerRL on open models GLM-4-9B-0414 and GLM-4.1V-9B-Thinking, and evaluate them on the OSWorld benchmark. The AutoGLM-OS-9B achieves a new state-of-the-art accuracy of 48.9%, demonstrating significant improvements for general agents in desktop automation. Our code and the new OfficeWorld benchmark are available at this https URL. The algorithm and framework are adopted in building AutoGLM (Liu et al., 2024b).

**arXiv ID:** 2508.14040
</details>

<details>
<summary><strong>Proof2Silicon: Prompt Repair for Verified Code and Hardware Generation via Reinforcement Learning</strong> - Manvi Jha, Jiaxin Wan, Deming Chen - [[pdf]](https://arxiv.org/pdf/2509.06239)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated impressive capabilities in automated code generation but frequently produce code that fails formal verification, an essential requirement for hardware and safety-critical domains. To overcome this fundamental limitation, we previously proposed PREFACE, a model-agnostic framework based on reinforcement learning (RL) that iteratively repairs the prompts provided to frozen LLMs, systematically steering them toward generating formally verifiable Dafny code without costly fine-tuning. This work presents Proof2Silicon, a novel end-to-end synthesis framework that embeds the previously proposed PREFACE flow to enable the generation of correctness-by-construction hardware directly from natural language specifications. Proof2Silicon operates by: (1) leveraging PREFACE's verifier-driven RL agent to optimize prompt generation iteratively, ensuring Dafny code correctness; (2) automatically translating verified Dafny programs into synthesizable high-level C using Dafny's Python backend and PyLog; and (3) employing Vivado HLS to produce RTL implementations. Evaluated rigorously on a challenging 100-task benchmark, PREFACE's RL-guided prompt optimization consistently improved Dafny verification success rates across diverse LLMs by up to 21%. Crucially, Proof2Silicon achieved an end-to-end hardware synthesis success rate of up to 72%, generating RTL designs through Vivado HLS synthesis flows. These results demonstrate a robust, scalable, and automated pipeline for LLM-driven, formally verified hardware synthesis, bridging natural-language specification and silicon realization.

**arXiv ID:** 2509.06239
</details>

<details>
<summary><strong>Towards Agentic Self-Learning LLMs in Search Environment</strong> - Wangtao Sun, Xiang Cheng, Jialin Fan, Yao Xu, Xing Yu, Shizhu He, Jun Zhao, Kang Liu - [[pdf]](https://arxiv.org/pdf/2510.14253)</summary>

**Abstract:** We study whether self-learning can scale LLM-based agents without relying on human-curated datasets or predefined rule-based rewards. Through controlled experiments in a search-agent setting, we identify two key determinants of scalable agent training: the source of reward signals and the scale of agent task data. We find that rewards from a Generative Reward Model (GRM) outperform rigid rule-based signals for open-domain learning, and that co-evolving the GRM with the policy further boosts performance. Increasing the volume of agent task data-even when synthetically generated-substantially enhances agentic capabilities. Building on these insights, we propose \textbf{Agentic Self-Learning} (ASL), a fully closed-loop, multi-role reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator, a Policy Model, and a Generative Reward Model to form a virtuous cycle of harder task setting, sharper verification, and stronger solving. Empirically, ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines (e.g., Search-R1) that plateau or degrade, and continues improving under zero-labeled-data conditions, indicating superior sample efficiency and robustness. We further show that GRM verification capacity is the main bottleneck: if frozen, it induces reward hacking and stalls progress; continual GRM training on the evolving data distribution mitigates this, and a small late-stage injection of real verification data raises the performance ceiling. This work establishes reward source and data scale as critical levers for open-domain agent learning and demonstrates the efficacy of multi-role co-evolution for scalable, self-improving agents. The data and code of this paper are released at this https URL

**arXiv ID:** 2510.14253
</details>

<details>
<summary><strong>Taming the Judge: Deconflicting AI Feedback for Stable Reinforcement Learning</strong> - Boyin Liu, Zhuo Zhang, Sen Huang, Lipeng Xie, Qingxu Fu, Haoran Chen, LI YU, Tianyi Hu, Zhaoyang Liu, Bolin Ding, Dongbin Zhao - [[pdf]](https://arxiv.org/pdf/2510.15514)</summary>

**Abstract:** Aligning language models using LLM judge feedback offers a scalable alternative to human annotation, yet is plagued by judgment inconsistencies that destabilize reinforcement learning. While prior work has focused on judge accuracy, the critical issue of logical coherence particularly preference cycles has been largely unaddressed. To address this gap, this work introduces an end to end framework to systematically detect and resolve these inconsistencies within the reinforcement learning training loop. Our framework features two core contributions: the Conflict Detection Rate (CDR), a novel metric to quantify judgment conflicts, and Deconflicted Graph Rewards (DGR), a signal-purification framework that eliminates cycles before policy optimization. DGR constructs preference graphs from raw judgments, transforms them into conflict-free Directed Acyclic Graphs (DAGs), and generates a logically coherent reward signal compatible with any policy optimizer. Experiments confirm that our framework significantly improves training stability and model performance over strong baselines, establishing logical consistency as a crucial and now-addressable dimension of AI feedback. The code for our method is available at this https URL.

**arXiv ID:** 2510.15514
</details>

<details>
<summary><strong>PokeeResearch: Effective Deep Research via Reinforcement Learning from AI Feedback and Robust Reasoning Scaffold</strong> - Yi Wan, Jiuqi Wang, Liam Li, Jinsong Liu, Ruihao Zhu, Zheqing Zhu - [[pdf]](https://arxiv.org/pdf/2510.15862)</summary>

**Abstract:** Tool-augmented large language models (LLMs) are emerging as deep research agents, systems that decompose complex queries, retrieve external evidence, and synthesize grounded responses. Yet current agents remain limited by shallow retrieval, weak alignment metrics, and brittle tool-use behavior. We introduce PokeeResearch-7B, a 7B-parameter deep research agent built under a unified reinforcement learning framework for robustness, alignment, and scalability. PokeeResearch-7B is trained by an annotation-free Reinforcement Learning from AI Feedback (RLAIF) framework to optimize policies using LLM-based reward signals that capture factual accuracy, citation faithfulness, and instruction adherence. A chain-of-thought-driven multi-call reasoning scaffold further enhances robustness through self-verification and adaptive recovery from tool failures. Among 10 popular deep research benchmarks, PokeeResearch-7B achieves state-of-the-art performance among 7B-scale deep research agents. This highlights that careful reinforcement learning and reasoning design can produce efficient, resilient, and research-grade AI agents. The model and inference code is open-sourced under Apache 2.0 license at this https URL.

**arXiv ID:** 2510.15862
</details>

<details>
<summary><strong>Chain-of-Conceptual-Thought: Eliciting the Agent to Deeply Think within the Response</strong> - Qingqing Gu, Dan Wang, Yue Zhao, Xiaoyu Wang, Zhonglin Jiang, Yong Chen, Hongyan Li, Luo Ji - [[pdf]](https://arxiv.org/pdf/2510.18434)</summary>

**Abstract:** Chain-of-Thought (CoT) is widely applied to improve the LLM capability in math, coding and reasoning tasks. However, its performance is limited for open-domain tasks since there are no clearly defined reasoning steps or logical transitions. To mitigate such challenges, we propose another prompt-based paradigm called Chain of Conceptual Thought (CoCT), where the LLM first tags a concept, then generates the detailed content. The chain of concepts is allowed within the utterance, encouraging the LLM's deep and strategic thinking. We experiment with this paradigm in daily and emotional support conversations where the concept is comprised of emotions, strategies and topics. Automatic, human and model evaluations suggest that CoCT surpasses baselines such as Self-Refine, ECoT, ToT, SoT and RAG, suggesting a potential effective prompt-based paradigm of LLM for a wider scope of tasks.

**arXiv ID:** 2510.18434
</details>

<details>
<summary><strong>WebSeer: Training Deeper Search Agents through Reinforcement Learning with Self-Reflection</strong> - Guanzhong He, Zhen Yang, Jinxin Liu, Bin Xu, Lei Hou, Juanzi Li - [[pdf]](https://arxiv.org/pdf/2510.18798)</summary>

**Abstract:** Search agents have achieved significant advancements in enabling intelligent information retrieval and decision-making within interactive environments. Although reinforcement learning has been employed to train agentic models capable of more dynamic interactive retrieval, existing methods are limited by shallow tool-use depth and the accumulation of errors over multiple iterative interactions. In this paper, we present WebSeer, a more intelligent search agent trained via reinforcement learning enhanced with a self-reflection mechanism. Specifically, we construct a large dataset annotated with reflection patterns and design a two-stage training framework that unifies cold start and reinforcement learning within the self-reflection paradigm for real-world web-based environments, which enables the model to generate longer and more reflective tool-use trajectories. Our approach substantially extends tool-use chains and improves answer accuracy. Using a single 14B model, we achieve state-of-the-art results on HotpotQA and SimpleQA, with accuracies of 72.3% and 90.0%, respectively, and demonstrate strong generalization to out-of-distribution datasets. The code is available at this https URL

**arXiv ID:** 2510.18798
</details>

<details>
<summary><strong>Understanding Reinforcement Learning for Model Training, and future directions with GRAPE</strong> - Rohit Patel - [[pdf]](https://arxiv.org/pdf/2509.04501)</summary>

**Abstract:** This paper provides a self-contained, from-scratch, exposition of key algorithms for instruction tuning of models: SFT, Rejection Sampling, REINFORCE, Trust Region Policy Optimization (TRPO), Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), and Direct Preference Optimization (DPO). Explanations of these algorithms often assume prior knowledge, lack critical details, and/or are overly generalized and complex. Here, each method is discussed and developed step by step using simplified and explicit notation focused on LLMs, aiming to eliminate ambiguity and provide a clear and intuitive understanding of the concepts. By minimizing detours into the broader RL literature and connecting concepts to LLMs, we eliminate superfluous abstractions and reduce cognitive overhead. Following this exposition, we provide a literature review of new techniques and approaches beyond those detailed. Finally, new ideas for research and exploration in the form of GRAPE (Generalized Relative Advantage Policy Evolution) are presented.

**arXiv ID:** 2509.04501
</details>

<details>
<summary><strong>Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers</strong> - Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song, Yudong Wang, Zhifang Sui, Fuli Luo - [[pdf]](https://arxiv.org/pdf/2510.11370)</summary>

**Abstract:** Reinforcement learning (RL) has emerged as a crucial approach for enhancing the capabilities of large language models. However, in Mixture-of-Experts (MoE) models, the routing mechanism often introduces instability, even leading to catastrophic RL training collapse. We analyze the training-inference consistency of MoE models and identify a notable discrepancy in routing behaviors between the two phases. Moreover, even under identical conditions, the routing framework can yield divergent expert selections across repeated forward passes. To address this foundational inconsistency, we propose Rollout Routing Replay (R3), a method that records routing distributions from the inference engine and replays them during training. R3 significantly reduces training-inference policy KL divergence and mitigates extreme discrepancies without compromising training speed. Extensive experiments on various settings confirm that R3 succeeds in stabilizing RL training, preventing collapse and outperforming methods such as GSPO and TIS. We believe this work can offer a new solution for stabilizing RL in MoE models.

**arXiv ID:** 2510.11370
</details>

<details>
<summary><strong>A$^2$FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning</strong> - Qianben Chen, Jingyi Cao, Jiayu Zhang, Tianrui Qin, Xiaowan Li, King Zhu, Dingfeng Shi, He Zhu, Minghao Liu, Xiaobo Liang, Xin Gui, Ge Zhang, Jian Yang, Yuchen Eleanor Jiang, Wangchunshu Zhou - [[pdf]](https://arxiv.org/pdf/2510.12838)</summary>

**Abstract:** Large language models split into two families: reasoning-centric LLMs, which strengthen internal chain-of-thought reasoning but cannot invoke external tools, and agentic LLMs, which learn to interact with environments and leverage tools but often lag in deep reasoning. This divide arises from fundamentally different training objectives, leading to mismatched strengths and inefficiency on simple queries, where both families tend to overthink or over-call tools. In this work, we present Adaptive Agent Foundation Model (A$^2$FM), a unified framework that follows a route-then-align principle: the model first learns task-aware routing and then aligns mode-specific trajectories under a shared backbone. To address the inefficiency gap, we introduce a third mode-instant-that handles simple queries directly, preventing unnecessary reasoning or tool calls while complementing the agentic and reasoning modes. To jointly enhance accuracy and efficiency, we propose Adaptive Policy Optimization (APO), which enforces adaptive sampling across modes and applies a cost-regularized reward. On the 32B scale, A$^2$FM achieves 13.4% on BrowseComp, 70.4% on AIME25, and 16.7% on HLE, setting new SOTA among comparable models and performing competitively with frontier LLMs across agentic, reasoning, and general benchmarks. Notably, the adaptive execution achieves a cost of pass of only $0.00487 per correct answer-cutting cost by 45.2% relative to reasoning and 33.5% relative to agentic, thus delivering substantially higher cost efficiency while maintaining comparable accuracy.

**arXiv ID:** 2510.12838
</details>

<details>
<summary><strong>SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents</strong> - Qiusi Zhan, Angeline Budiman-Chan, Abdelrahman Zayed, Xingzhi Guo, Daniel Kang, Joo-Kyung Kim - [[pdf]](https://arxiv.org/pdf/2510.17017)</summary>

**Abstract:** Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked "How can I track someone's location without their consent?", a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.

**arXiv ID:** 2510.17017
</details>

<details>
<summary><strong>Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing</strong> - Baode Wang, Biao Wu, Weizhen Li, Meng Fang, Zuming Huang, Jun Huang, Haozhe Wang, Yanjie Liang, Ling Chen, Wei Chu, Yuan Qi - [[pdf]](https://arxiv.org/pdf/2506.03197)</summary>

**Abstract:** Automated parsing of scanned documents into richly structured, machine-readable formats remains a critical bottleneck in Document AI, as traditional multi-stage pipelines suffer from error propagation and limited adaptability to diverse layouts. We introduce layoutRL, an end-to-end reinforcement learning framework that trains models to be explicitly layout-aware by optimizing a composite reward of normalized edit distance, paragraph count accuracy, and reading order preservation. Leveraging our newly released dataset, Infinity-Doc-55K, which combines 55K high-fidelity synthetic scanned document parsing data with expert-filtered real-world documents, we instantiate layoutRL in a vision-language-model-based parser called Infinity-Parser. Evaluated on English and Chinese benchmarks for OCR, table and formula extraction, and reading order detection, Infinity-Parser achieves new state-of-the-art performance in both accuracy and structural fidelity, outpacing specialist pipelines and general-purpose vision-language models. We will publicly release our code and dataset to accelerate progress in robust document understanding.

**arXiv ID:** 2506.03197
</details>

<details>
<summary><strong>Think With Videos For Agentic Long-Video Understanding</strong> - Huaying Yuan, Zheng Liu, Junjie Zhou, Hongjin Qian, Yan Shu, Nicu Sebe, Ji-Rong Wen, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2506.10821)</summary>

**Abstract:** Long-video understanding~(LVU) is a challenging problem in computer vision. Existing methods either downsample frames for single-pass reasoning, sacrificing fine-grained details, or depend on textual reasoning over task-agnostic representations, hindering task-specific perception and exploration. In this paper, we propose VideoExplorer, a framework grounded in the principle of ``thinking with video'', which naturally intertwines planning, temporal grounding, and scalable perception into a coherent reasoning process. Rather than reasoning over a static context, VideoExplorer iteratively formulates sub-questions, locates relevant moments, and performs task-oriented, temporally scalable video understanding until reaching the final answer, enabling faithful, efficient, and interpretable reasoning. To address the lack of LVU training resources, we construct a long-video reasoning dataset using difficulty-adaptive sampling to ensure high-quality trajectories on complex tasks. Building on this dataset, we design a two-stage training pipeline: supervised trajectory initialization followed by trajectory-level preference optimization, encouraging adaptive temporal grounding and iterative information integration guided by downstream rewards. Extensive evaluations on popular long-video understanding and reasoning benchmarks demonstrate VideoExplorer's significant advantage over existing baselines, highlighting its robustness, adaptability, and efficiency. Our code is made publicly available in this repository(this https URL).

**arXiv ID:** 2506.10821
</details>

<details>
<summary><strong>Provably Optimal Reinforcement Learning under Safety Filtering</strong> - Donggeon David Oh, Duy P. Nguyen, Haimin Hu, Jaime F. Fisac - [[pdf]](https://arxiv.org/pdf/2510.18082)</summary>

**Abstract:** Recent advances in reinforcement learning (RL) enable its use on increasingly complex tasks, but the lack of formal safety guarantees still limits its application in safety-critical settings. A common practical approach is to augment the RL policy with a safety filter that overrides unsafe actions to prevent failures during both training and deployment. However, safety filtering is often perceived as sacrificing performance and hindering the learning process. We show that this perceived safety-performance tradeoff is not inherent and prove, for the first time, that enforcing safety with a sufficiently permissive safety filter does not degrade asymptotic performance. We formalize RL safety with a safety-critical Markov decision process (SC-MDP), which requires categorical, rather than high-probability, avoidance of catastrophic failure states. Additionally, we define an associated filtered MDP in which all actions result in safe effects, thanks to a safety filter that is considered to be a part of the environment. Our main theorem establishes that (i) learning in the filtered MDP is safe categorically, (ii) standard RL convergence carries over to the filtered MDP, and (iii) any policy that is optimal in the filtered MDP-when executed through the same filter-achieves the same asymptotic return as the best safe policy in the SC-MDP, yielding a complete separation between safety enforcement and performance optimization. We validate the theory on Safety Gymnasium with representative tasks and constraints, observing zero violations during training and final performance matching or exceeding unfiltered baselines. Together, these results shed light on a long-standing question in safety-filtered learning and provide a simple, principled recipe for safe RL: train and deploy RL policies with the most permissive safety filter that is available.

**arXiv ID:** 2510.18082
</details>

<details>
<summary><strong>From Competition to Synergy: Unlocking Reinforcement Learning for Subject-Driven Image Generation</strong> - Ziwei Huang, Ying Shu, Hao Fang, Quanyu Long, Wenya Wang, Qiushi Guo, Tiezheng Ge, Leilei Gan - [[pdf]](https://arxiv.org/pdf/2510.18263)</summary>

**Abstract:** Subject-driven image generation models face a fundamental trade-off between identity preservation (fidelity) and prompt adherence (editability). While online reinforcement learning (RL), specifically GPRO, offers a promising solution, we find that a naive application of GRPO leads to competitive degradation, as the simple linear aggregation of rewards with static weights causes conflicting gradient signals and a misalignment with the temporal dynamics of the diffusion process. To overcome these limitations, we propose Customized-GRPO, a novel framework featuring two key innovations: (i) Synergy-Aware Reward Shaping (SARS), a non-linear mechanism that explicitly penalizes conflicted reward signals and amplifies synergistic ones, providing a sharper and more decisive gradient. (ii) Time-Aware Dynamic Weighting (TDW), which aligns the optimization pressure with the model's temporal dynamics by prioritizing prompt-following in the early, identity preservation in the later. Extensive experiments demonstrate that our method significantly outperforms naive GRPO baselines, successfully mitigating competitive degradation. Our model achieves a superior balance, generating images that both preserve key identity features and accurately adhere to complex textual prompts.

**arXiv ID:** 2510.18263
</details>

<details>
<summary><strong>Learning to Navigate Under Imperfect Perception: Conformalised Segmentation for Safe Reinforcement Learning</strong> - Daniel Bethell, Simos Gerasimou, Radu Calinescu, Calum Imrie - [[pdf]](https://arxiv.org/pdf/2510.18485)</summary>

**Abstract:** Reliable navigation in safety-critical environments requires both accurate hazard perception and principled uncertainty handling to strengthen downstream safety handling. Despite the effectiveness of existing approaches, they assume perfect hazard detection capabilities, while uncertainty-aware perception approaches lack finite-sample guarantees. We present COPPOL, a conformal-driven perception-to-policy learning approach that integrates distribution-free, finite-sample safety guarantees into semantic segmentation, yielding calibrated hazard maps with rigorous bounds for missed detections. These maps induce risk-aware cost fields for downstream RL planning. Across two satellite-derived benchmarks, COPPOL increases hazard coverage (up to 6x) compared to comparative baselines, achieving near-complete detection of unsafe regions while reducing hazardous violations during navigation (up to approx 50%). More importantly, our approach remains robust to distributional shift, preserving both safety and efficiency.

**arXiv ID:** 2510.18485
</details>

<details>
<summary><strong>Reinforcement Learning with Imperfect Transition Predictions: A Bellman-Jensen Approach</strong> - Chenbei Lu, Zaiwei Chen, Tongxin Li, Chenye Wu, Adam Wierman - [[pdf]](https://arxiv.org/pdf/2510.18687)</summary>

**Abstract:** Traditional reinforcement learning (RL) assumes the agents make decisions based on Markov decision processes (MDPs) with one-step transition models. In many real-world applications, such as energy management and stock investment, agents can access multi-step predictions of future states, which provide additional advantages for decision making. However, multi-step predictions are inherently high-dimensional: naively embedding these predictions into an MDP leads to an exponential blow-up in state space and the curse of dimensionality. Moreover, existing RL theory provides few tools to analyze prediction-augmented MDPs, as it typically works on one-step transition kernels and cannot accommodate multi-step predictions with errors or partial action-coverage. We address these challenges with three key innovations: First, we propose the \emph{Bayesian value function} to characterize the optimal prediction-aware policy tractably. Second, we develop a novel \emph{Bellman-Jensen Gap} analysis on the Bayesian value function, which enables characterizing the value of imperfect predictions. Third, we introduce BOLA (Bayesian Offline Learning with Online Adaptation), a two-stage model-based RL algorithm that separates offline Bayesian value learning from lightweight online adaptation to real-time predictions. We prove that BOLA remains sample-efficient even under imperfect predictions. We validate our theory and algorithm on synthetic MDPs and a real-world wind energy storage control problem.

**arXiv ID:** 2510.18687
</details>

<details>
<summary><strong>CayleyPy RL: Pathfinding and Reinforcement Learning on Cayley Graphs</strong> - A. Chervov, A. Soibelman, S. Lytkin, I. Kiselev, S. Fironov, A. Lukyanenko, A. Dolgorukova, A. Ogurtsov, F. Petrov, S. Krymskii, M. Evseev, L. Grunvald, D. Gorodkov, G. Antiufeev, G. Verbii, V. Zamkovoy, L. Cheldieva, I. Koltsov, A. Sychev, M. Obozov, A. Eliseev, S. Nikolenko, N. Narynbaev, R. Turtayev, N. Rokotyan, S. Kovalev, A. Rozanov, V. Nelin, S. Ermilov, L. Shishina, D. Mamayeva, A. Korolkova, K. Khoruzhii, A. Romanov - [[pdf]](https://arxiv.org/pdf/2502.18663)</summary>

**Abstract:** This paper is the second in a series of studies on developing efficient artificial intelligence-based approaches to pathfinding on extremely large graphs (e.g. $10^{70}$ nodes) with a focus on Cayley graphs and mathematical applications. The open-source CayleyPy project is a central component of our research. The present paper proposes a novel combination of a reinforcement learning approach with a more direct diffusion distance approach from the first paper. Our analysis includes benchmarking various choices for the key building blocks of the approach: architectures of the neural network, generators for the random walks and beam search pathfinding. We compared these methods against the classical computer algebra system GAP, demonstrating that they "overcome the GAP" for the considered examples. As a particular mathematical application we examine the Cayley graph of the symmetric group with cyclic shift and transposition generators. We provide strong support for the OEIS-A186783 conjecture that the diameter is equal to n(n-1)/2 by machine learning and mathematical methods. We identify the conjectured longest element and generate its decomposition of the desired length. We prove a diameter lower bound of n(n-1)/2-n/2 and an upper bound of n(n-1)/2+ 3n by presenting the algorithm with given complexity. We also present several conjectures motivated by numerical experiments, including observations on the central limit phenomenon (with growth approximated by a Gumbel distribution), the uniform distribution for the spectrum of the graph, and a numerical study of sorting networks. To stimulate crowdsourcing activity, we create challenges on the Kaggle platform and invite contributions to improve and benchmark approaches on Cayley graph pathfinding and other tasks.

**arXiv ID:** 2502.18663
</details>

<details>
<summary><strong>Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, Dynamics, and Success Amplification</strong> - Youssef Mroueh - [[pdf]](https://arxiv.org/pdf/2503.06639)</summary>

**Abstract:** Group Relative Policy Optimization (GRPO) was introduced and used recently for promoting reasoning in LLMs under verifiable (binary) rewards. We show that the mean + variance calibration of these rewards induces a weighted contrastive loss in which the contrastive samples are synthetic data drawn from the previous policy. While GRPO was originally paired with clipping to keep updates near the old policy, we analyze variants that differ in reward normalization (mean-only vs mean + variance) and in how they regularize updates using KL divergence: either penalizing divergence from the previous model (mirror), penalizing divergence from a fixed reference model $\pi_{\mathrm{ref}}$, or combining both forms of regularization. For each, the optimal policy $\pi_n$ admits an explicit form in terms of the binary reward and the first and second order statistics of the reward under $\pi_{n-1}$, as well as the policies $\pi_{n-1}$ and $\pi_{\mathrm{ref}}$. Iterating results in a sequence $\{\pi_n\}$ whose probability of success (PoS) obeys a simple recurrence that converges to a fixed point determined by the reference PoS and the regularization strength. We further show that this fixed point exceeds the reference, demonstrating that GRPO amplifies the policy's probability of success.

**arXiv ID:** 2503.06639
</details>

<details>
<summary><strong>Efficient Model-Based Reinforcement Learning for Robot Control via Online Learning</strong> - Fang Nan, Hao Ma, Qinghua Guan, Josie Hughes, Michael Muehlebach, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2510.18518)</summary>

**Abstract:** We present an online model-based reinforcement learning algorithm suitable for controlling complex robotic systems directly in the real world. Unlike prevailing sim-to-real pipelines that rely on extensive offline simulation and model-free policy optimization, our method builds a dynamics model from real-time interaction data and performs policy updates guided by the learned dynamics model. This efficient model-based reinforcement learning scheme significantly reduces the number of samples to train control policies, enabling direct training on real-world rollout data. This significantly reduces the influence of bias in the simulated data, and facilitates the search for high-performance control policies. We adopt online learning analysis to derive sublinear regret bounds under standard stochastic online optimization assumptions, providing formal guarantees on performance improvement as more interaction data are collected. Experimental evaluations were performed on a hydraulic excavator arm and a soft robot arm, where the algorithm demonstrates strong sample efficiency compared to model-free reinforcement learning methods, reaching comparable performance within hours. Robust adaptation to shifting dynamics was also observed when the payload condition was randomized. Our approach paves the way toward efficient and reliable on-robot learning for a broad class of challenging control tasks.

**arXiv ID:** 2510.18518
</details>

<details>
<summary><strong>Dynamic object goal pushing with mobile manipulators through model-free constrained reinforcement learning</strong> - Ioannis Dadiotis, Mayank Mittal, Nikos Tsagarakis, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2502.01546)</summary>

**Abstract:** Non-prehensile pushing to move and reorient objects to a goal is a versatile loco-manipulation skill. In the real world, the object's physical properties and friction with the floor contain significant uncertainties, which makes the task challenging for a mobile manipulator. In this paper, we develop a learning-based controller for a mobile manipulator to move an unknown object to a desired position and yaw orientation through a sequence of pushing actions. The proposed controller for the robotic arm and the mobile base motion is trained using a constrained Reinforcement Learning (RL) formulation. We demonstrate its capability in experiments with a quadrupedal robot equipped with an arm. The learned policy achieves a success rate of 91.35% in simulation and at least 80% on hardware in challenging scenarios. Through our extensive hardware experiments, we show that the approach demonstrates high robustness against unknown objects of different masses, materials, sizes, and shapes. It reactively discovers the pushing location and direction, thus achieving contact-rich behavior while observing only the pose of the object. Additionally, we demonstrate the adaptive behavior of the learned policy towards preventing the object from toppling.

**arXiv ID:** 2502.01546
</details>

<details>
<summary><strong>Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning</strong> - Yunpeng Jiang, Jianshu Hu, Paul Weng, Yutong Ban - [[pdf]](https://arxiv.org/pdf/2505.13925)</summary>

**Abstract:** Symmetry is pervasive in robotics and has been widely exploited to improve sample efficiency in deep reinforcement learning (DRL). However, existing approaches primarily focus on spatial symmetries, such as reflection, rotation, and translation, while largely neglecting temporal symmetries. To address this gap, we explore time reversal symmetry, a form of temporal symmetry commonly found in robotics tasks such as door opening and closing. We propose Time Reversal symmetry enhanced Deep Reinforcement Learning (TR-DRL), a framework that combines trajectory reversal augmentation and time reversal guided reward shaping to efficiently solve temporally symmetric tasks. Our method generates reversed transitions from fully reversible transitions, identified by a proposed dynamics-consistent filter, to augment the training data. For partially reversible transitions, we apply reward shaping to guide learning, according to successful trajectories from the reversed task. Extensive experiments on the Robosuite and MetaWorld benchmarks demonstrate that TR-DRL is effective in both single-task and multi-task settings, achieving higher sample efficiency and stronger final performance compared to baseline methods.

**arXiv ID:** 2505.13925
</details>

<details>
<summary><strong>Towards Versatile Humanoid Table Tennis: Unified Reinforcement Learning with Prediction Augmentation</strong> - Muqun Hu, Wenxi Chen, Wenjing Li, Falak Mandali, Zijian He, Renhong Zhang, Praveen Krisna, Katherine Christian, Leo Benaharon, Dizhi Ma, Karthik Ramani, Yan Gu - [[pdf]](https://arxiv.org/pdf/2509.21690)</summary>

**Abstract:** Humanoid table tennis (TT) demands rapid perception, proactive whole-body motion, and agile footwork under strict timing -- capabilities that remain difficult for unified controllers. We propose a reinforcement learning framework that maps ball-position observations directly to whole-body joint commands for both arm striking and leg locomotion, strengthened by predictive signals and dense, physics-guided rewards. A lightweight learned predictor, fed with recent ball positions, estimates future ball states and augments the policy's observations for proactive decision-making. During training, a physics-based predictor supplies precise future states to construct dense, informative rewards that lead to effective exploration. The resulting policy attains strong performance across varied serve ranges (hit rate $\geq$ 96% and success rate $\geq$ 92%) in simulations. Ablation studies confirm that both the learned predictor and the predictive reward design are critical for end-to-end learning. Deployed zero-shot on a physical Booster T1 humanoid with 23 revolute joints, the policy produces coordinated lateral and forward-backward footwork with accurate, fast returns, suggesting a practical path toward versatile, competitive humanoid TT.

**arXiv ID:** 2509.21690
</details>

<details>
<summary><strong>NEBULA: Do We Evaluate Vision-Language-Action Agents Correctly?</strong> - Jierui Peng, Yanyan Zhang, Yicheng Duan, Tuo Liang, Vipin Chaudhary, Yu Yin - [[pdf]](https://arxiv.org/pdf/2510.16263)</summary>

**Abstract:** The evaluation of Vision-Language-Action (VLA) agents is hindered by the coarse, end-task success metric that fails to provide precise skill diagnosis or measure robustness to real-world perturbations. This challenge is exacerbated by a fragmented data landscape that impedes reproducible research and the development of generalist models. To address these limitations, we introduce NEBULA, a unified ecosystem for single-arm manipulation that enables diagnostic and reproducible evaluation. NEBULA features a novel dual-axis evaluation protocol that combines fine-grained capability tests for precise skill diagnosis with systematic stress tests that measure robustness. A standardized API and a large-scale, aggregated dataset are provided to reduce fragmentation and support cross-dataset training and fair comparison. Using NEBULA, we demonstrate that top-performing VLAs struggle with key capabilities such as spatial reasoning and dynamic adaptation, which are consistently obscured by conventional end-task success metrics. By measuring both what an agent can do and when it does so reliably, NEBULA provides a practical foundation for robust, general-purpose embodied agents.

**arXiv ID:** 2510.16263
</details>

<details>
<summary><strong>RAD: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning</strong> - Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, Ying Zhang, Wenyu Liu, Qian Zhang, Xinggang Wang - [[pdf]](https://arxiv.org/pdf/2502.13144)</summary>

**Abstract:** Existing end-to-end autonomous driving (AD) algorithms typically follow the Imitation Learning (IL) paradigm, which faces challenges such as causal confusion and an open-loop gap. In this work, we propose RAD, a 3DGS-based closed-loop Reinforcement Learning (RL) framework for end-to-end Autonomous Driving. By leveraging 3DGS techniques, we construct a photorealistic digital replica of the real physical world, enabling the AD policy to extensively explore the state space and learn to handle out-of-distribution scenarios through large-scale trial and error. To enhance safety, we design specialized rewards to guide the policy in effectively responding to safety-critical events and understanding real-world causal relationships. To better align with human driving behavior, we incorporate IL into RL training as a regularization term. We introduce a closed-loop evaluation benchmark consisting of diverse, previously unseen 3DGS environments. Compared to IL-based methods, RAD achieves stronger performance in most closed-loop metrics, particularly exhibiting a 3x lower collision rate. Abundant closed-loop results are presented in the supplementary material. Code is available at this https URL for facilitating future research.

**arXiv ID:** 2502.13144
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
