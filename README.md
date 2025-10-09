# Agent arXiv Daily

**Last Updated:** 2025-10-09 02:39:07

**Total Papers:** 73

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
<summary><strong>A Digital Twin Framework for Metamorphic Testing of Autonomous Driving Systems Using Generative Model</strong> - Tony Zhang, Burak Kantarci, Umair Siddique - [[pdf]](https://arxiv.org/pdf/2510.07133)</summary>

**Abstract:** Ensuring the safety of self-driving cars remains a major challenge due to the complexity and unpredictability of real-world driving environments. Traditional testing methods face significant limitations, such as the oracle problem, which makes it difficult to determine whether a system's behavior is correct, and the inability to cover the full range of scenarios an autonomous vehicle may encounter. In this paper, we introduce a digital twin-driven metamorphic testing framework that addresses these challenges by creating a virtual replica of the self-driving system and its operating environment. By combining digital twin technology with AI-based image generative models such as Stable Diffusion, our approach enables the systematic generation of realistic and diverse driving scenes. This includes variations in weather, road topology, and environmental features, all while maintaining the core semantics of the original scenario. The digital twin provides a synchronized simulation environment where changes can be tested in a controlled and repeatable manner. Within this environment, we define three metamorphic relations inspired by real-world traffic rules and vehicle behavior. We validate our framework in the Udacity self-driving simulator and demonstrate that it significantly enhances test coverage and effectiveness. Our method achieves the highest true positive rate (0.719), F1 score (0.689), and precision (0.662) compared to baseline approaches. This paper highlights the value of integrating digital twins with AI-powered scenario generation to create a scalable, automated, and high-fidelity testing solution for autonomous vehicle safety.

**arXiv ID:** 2510.07133
</details>

<details>
<summary><strong>Can AI Have a Personality? Prompt Engineering for AI Personality Simulation: A Chatbot Case Study in Gender-Affirming Voice Therapy Training</strong> - Tailon D. Jackson, Byunggu Yu - [[pdf]](https://arxiv.org/pdf/2508.18234)</summary>

**Abstract:** This thesis investigates whether large language models (LLMs) can be guided to simulate a consistent personality through prompt engineering. The study explores this concept within the context of a chatbot designed for Speech-Language Pathology (SLP) student training, specifically focused on gender-affirming voice therapy. The chatbot, named Monae Jackson, was created to represent a 32-year-old transgender woman and engage in conversations simulating client-therapist interactions. Findings suggest that with prompt engineering, the chatbot maintained a recognizable and consistent persona and had a distinct personality based on the Big Five Personality test. These results support the idea that prompt engineering can be used to simulate stable personality characteristics in AI chatbots.

**arXiv ID:** 2508.18234
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (14 papers)</h2></summary>

<details>
<summary><strong>BuilderBench -- A benchmark for generalist agents</strong> - Raj Ghugare, Catherine Ji, Kathryn Wantlin, Jin Schofield, Benjamin Eysenbach - [[pdf]](https://arxiv.org/pdf/2510.06288)</summary>

**Abstract:** Today's AI models learn primarily through mimicry and sharpening, so it is not surprising that they struggle to solve problems beyond the limits set by existing data. To solve novel problems, agents should acquire skills for exploring and learning through experience. Finding a scalable learning mechanism for developing agents that learn through interaction remains a major open problem. In this work, we introduce BuilderBench, a benchmark to accelerate research into agent pre-training that centers open-ended exploration. BuilderBench requires agents to learn how to build any structure using blocks. BuilderBench is equipped with $(1)$ a hardware accelerated simulator of a robotic agent interacting with various physical blocks, and $(2)$ a task-suite with over 42 diverse target structures that are carefully curated to test an understanding of physics, mathematics, and long-horizon planning. During training, agents have to explore and learn general principles about the environment without any external supervision. During evaluation, agents have to build the unseen target structures from the task suite. Solving these tasks requires a sort of \emph{embodied reasoning} that is not reflected in words but rather in actions, experimenting with different strategies and piecing them together. Our experiments show that many of these tasks challenge the current iteration of algorithms. Hence, we also provide a ``training wheels'' protocol, in which agents are trained and evaluated to build a single target structure from the task suite. Finally, we provide single-file implementations of six different algorithms as a reference point for researchers.

**arXiv ID:** 2510.06288
</details>

<details>
<summary><strong>Inefficiencies of Meta Agents for Agent Design</strong> - Batu El, Mert Yuksekgonul, James Zou - [[pdf]](https://arxiv.org/pdf/2510.06711)</summary>

**Abstract:** Recent works began to automate the design of agentic systems using meta-agents that propose and iteratively refine new agent architectures. In this paper, we examine three key challenges in a common class of meta-agents. First, we investigate how a meta-agent learns across iterations and find that simply expanding the context with all previous agents, as proposed by previous works, performs worse than ignoring prior designs entirely. We show that the performance improves with an evolutionary approach. Second, although the meta-agent designs multiple agents during training, it typically commits to a single agent at test time. We find that the designed agents have low behavioral diversity, limiting the potential for their complementary use. Third, we assess when automated design is economically viable. We find that only in a few cases--specifically, two datasets--the overall cost of designing and deploying the agents is lower than that of human-designed agents when deployed on over 15,000 examples. In contrast, the performance gains for other datasets do not justify the design cost, regardless of scale.

**arXiv ID:** 2510.06711
</details>

<details>
<summary><strong>The Cognitive Bandwidth Bottleneck: Shifting Long-Horizon Agent from Planning with Actions to Planning with Schemas</strong> - Baixuan Xu, Tianshi Zheng, Zhaowei Wang, Hong Ting Tsang, Weiqi Wang, Tianqing Fang, Yangqiu Song - [[pdf]](https://arxiv.org/pdf/2510.07091)</summary>

**Abstract:** Enabling LLMs to effectively operate long-horizon task which requires long-term planning and multiple interactions is essential for open-world autonomy. Conventional methods adopt planning with actions where a executable action list would be provided as reference. However, this action representation choice would be impractical when the environment action space is combinatorial exploded (e.g., open-ended real world). This naturally leads to a question: As environmental action space scales, what is the optimal action representation for long-horizon agents? In this paper, we systematically study the effectiveness of two different action representations. The first one is conventional planning with actions (PwA) which is predominantly adopted for its effectiveness on existing benchmarks. The other one is planning with schemas (PwS) which instantiate an action schema into action lists (e.g., "move [OBJ] to [OBJ]" -> "move apple to desk") to ensure concise action space and reliable scalability. This alternative is motivated by its alignment with human cognition and its compliance with environment-imposed action format restriction. We propose cognitive bandwidth perspective as a conceptual framework to qualitatively understand the differences between these two action representations and empirically observe a representation-choice inflection point between ALFWorld (~35 actions) and SciWorld (~500 actions), which serve as evidence of the need for scalable representations. We further conduct controlled experiments to study how the location of this inflection point interacts with different model capacities: stronger planning proficiency shifts the inflection rightward, whereas better schema instantiation shifts it leftward. Finally, noting the suboptimal performance of PwS agents, we provide an actionable guide for building more capable PwS agents for better scalable autonomy.

**arXiv ID:** 2510.07091
</details>

<details>
<summary><strong>Constrained Natural Language Action Planning for Resilient Embodied Systems</strong> - Grayson Byrd, Corban Rivera, Bethany Kemp, Meghan Booker, Aurora Schmidt, Celso M de Melo, Lalithkumar Seenivasan, Mathias Unberath - [[pdf]](https://arxiv.org/pdf/2510.06357)</summary>

**Abstract:** Replicating human-level intelligence in the execution of embodied tasks remains challenging due to the unconstrained nature of real-world environments. Novel use of large language models (LLMs) for task planning seeks to address the previously intractable state/action space of complex planning tasks, but hallucinations limit their reliability, and thus, viability beyond a research context. Additionally, the prompt engineering required to achieve adequate system performance lacks transparency, and thus, repeatability. In contrast to LLM planning, symbolic planning methods offer strong reliability and repeatability guarantees, but struggle to scale to the complexity and ambiguity of real-world tasks. We introduce a new robotic planning method that augments LLM planners with symbolic planning oversight to improve reliability and repeatability, and provide a transparent approach to defining hard constraints with considerably stronger clarity than traditional prompt engineering. Importantly, these augmentations preserve the reasoning capabilities of LLMs and retain impressive generalization in open-world environments. We demonstrate our approach in simulated and real-world environments. On the ALFWorld planning benchmark, our approach outperforms current state-of-the-art methods, achieving a near-perfect 99% success rate. Deployment of our method to a real-world quadruped robot resulted in 100% task success compared to 50% and 30% for pure LLM and symbolic planners across embodied pick and place tasks. Our approach presents an effective strategy to enhance the reliability, repeatability and transparency of LLM-based robot planners while retaining their key strengths: flexibility and generalizability to complex real-world environments. We hope that this work will contribute to the broad goal of building resilient embodied intelligent systems.

**arXiv ID:** 2510.06357
</details>

<details>
<summary><strong>TrackVLA++: Unleashing Reasoning and Memory Capabilities in VLA Models for Embodied Visual Tracking</strong> - Jiahang Liu, Yunpeng Qi, Jiazhao Zhang, Minghan Li, Shaoan Wang, Kui Wu, Hanjing Ye, Hong Zhang, Zhibo Chen, Fangwei Zhong, Zhizheng Zhang, He Wang - [[pdf]](https://arxiv.org/pdf/2510.07134)</summary>

**Abstract:** Embodied Visual Tracking (EVT) is a fundamental ability that underpins practical applications, such as companion robots, guidance robots and service assistants, where continuously following moving targets is essential. Recent advances have enabled language-guided tracking in complex and unstructured scenes. However, existing approaches lack explicit spatial reasoning and effective temporal memory, causing failures under severe occlusions or in the presence of similar-looking distractors. To address these challenges, we present TrackVLA++, a novel Vision-Language-Action (VLA) model that enhances embodied visual tracking with two key modules, a spatial reasoning mechanism and a Target Identification Memory (TIM). The reasoning module introduces a Chain-of-Thought paradigm, termed Polar-CoT, which infers the target's relative position and encodes it as a compact polar-coordinate token for action prediction. Guided by these spatial priors, the TIM employs a gated update strategy to preserve long-horizon target memory, ensuring spatiotemporal consistency and mitigating target loss during extended occlusions. Extensive experiments show that TrackVLA++ achieves state-of-the-art performance on public benchmarks across both egocentric and multi-camera settings. On the challenging EVT-Bench DT split, TrackVLA++ surpasses the previous leading approach by 5.1 and 12, respectively. Furthermore, TrackVLA++ exhibits strong zero-shot generalization, enabling robust real-world tracking in dynamic and occluded scenarios.

**arXiv ID:** 2510.07134
</details>

<details>
<summary><strong>An Illusion of Progress? Assessing the Current State of Web Agents</strong> - Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, Yu Su - [[pdf]](https://arxiv.org/pdf/2504.01382)</summary>

**Abstract:** As digitalization and cloud technologies evolve, the web is becoming increasingly important in the modern society. Autonomous web agents based on large language models (LLMs) hold a great potential in work automation. It is therefore important to accurately measure and monitor the progression of their capabilities. In this work, we conduct a comprehensive and rigorous assessment of the current state of web agents. Our results depict a very different picture of the competency of current agents, suggesting over-optimism in previously reported results. This gap can be attributed to shortcomings in existing benchmarks. We introduce Online-Mind2Web, an online evaluation benchmark consisting of 300 diverse and realistic tasks spanning 136 websites. It enables us to evaluate web agents under a setting that approximates how real users use these agents. To facilitate more scalable evaluation and development, we also develop a novel LLM-as-a-Judge automatic evaluation method and show that it can achieve around 85% agreement with human judgment, substantially higher than existing methods. Finally, we present the first comprehensive comparative analysis of current web agents, highlighting both their strengths and limitations to inspire future research.

**arXiv ID:** 2504.01382
</details>

<details>
<summary><strong>Controlled Agentic Planning & Reasoning for Mechanism Synthesis</strong> - João Pedro Gandarela, Thiago Rios, Stefan Menzel, André Freitas - [[pdf]](https://arxiv.org/pdf/2505.17607)</summary>

**Abstract:** This work presents a dual-agent \ac{llm}-based reasoning framework for automated planar mechanism synthesis that tightly couples linguistic specification with symbolic representation and simulation. From a natural-language task description, the system composes symbolic constraints and equations, generates and parametrises simulation code, and iteratively refines designs via critic-driven feedback, including symbolic regression and geometric distance metrics, closing an actionable linguistic/symbolic optimisation loop. To evaluate the approach, we introduce MSynth, a benchmark of analytically defined planar trajectories. Empirically, critic feedback and iterative refinement yield large improvements (up to 90\% on individual tasks) and statistically significant gains per the Wilcoxon signed-rank test. Symbolic-regression prompts provide deeper mechanistic insight primarily when paired with larger models or architectures with appropriate inductive biases (e.g., LRM).

**arXiv ID:** 2505.17607
</details>

<details>
<summary><strong>CyberGym: Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale</strong> - Zhun Wang, Tianneng Shi, Jingxuan He, Matthew Cai, Jialin Zhang, Dawn Song - [[pdf]](https://arxiv.org/pdf/2506.02548)</summary>

**Abstract:** AI agents have significant potential to reshape cybersecurity, making a thorough assessment of their capabilities critical. However, existing evaluations fall short, because they are based on small-scale benchmarks and only measure static outcomes, failing to capture the full, dynamic range of real-world security challenges. To address these limitations, we introduce CyberGym, a large-scale benchmark featuring 1,507 real-world vulnerabilities across 188 software projects. Adjustable to different vulnerability analysis settings, CyberGym primarily tasks agents with generating a proof-of-concept test that reproduces a vulnerability, given only its text description and the corresponding codebase. Our extensive evaluation highlights that CyberGym effectively differentiates agents' and models' cybersecurity capabilities. Even the top-performing combinations only achieve a ~20% success rate, demonstrating the overall difficulty of CyberGym. Beyond static benchmarking, we show that CyberGym leads to the discovery of 35 zero-day vulnerabilities and 17 historically incomplete patches. These results underscore that CyberGym is not only a robust benchmark for measuring AI's progress in cybersecurity but also a platform for creating direct, real-world security impact.

**arXiv ID:** 2506.02548
</details>

<details>
<summary><strong>AutoMind: Adaptive Knowledgeable Agent for Automated Data Science</strong> - Yixin Ou, Yujie Luo, Jingsheng Zheng, Lanning Wei, Zhuoyun Yu, Shuofei Qiao, Jintian Zhang, Da Zheng, Yuren Mao, Yunjun Gao, Huajun Chen, Ningyu Zhang - [[pdf]](https://arxiv.org/pdf/2506.10974)</summary>

**Abstract:** Large Language Model (LLM) agents have shown great potential in addressing real-world data science problems. LLM-driven data science agents promise to automate the entire machine learning pipeline, yet their real-world effectiveness remains limited. Existing frameworks depend on rigid, pre-defined workflows and inflexible coding strategies; consequently, they excel only on relatively simple, classical problems and fail to capture the empirical expertise that human practitioners bring to complex, innovative tasks. In this work, we introduce AutoMind, an adaptive, knowledgeable LLM-agent framework that overcomes these deficiencies through three key advances: (1) a curated expert knowledge base that grounds the agent in domain expert knowledge, (2) an agentic knowledgeable tree search algorithm that strategically explores possible solutions, and (3) a self-adaptive coding strategy that dynamically tailors code generation to task complexity. Evaluations on two automated data science benchmarks demonstrate that AutoMind delivers superior performance versus state-of-the-art baselines. Additional analyses confirm favorable effectiveness, efficiency, and qualitative solution quality, highlighting AutoMind as an efficient and robust step toward fully automated data science. Code is at this https URL.

**arXiv ID:** 2506.10974
</details>

<details>
<summary><strong>CodeCureAgent: Automatic Classification and Repair of Static Analysis Warnings</strong> - Pascal Joos, Islem Bouzenia, Michael Pradel - [[pdf]](https://arxiv.org/pdf/2509.11787)</summary>

**Abstract:** Static analysis tools are widely used to detect bugs, vulnerabilities, and code smells. Traditionally, developers must resolve these warnings manually. Because this process is tedious, developers sometimes ignore warnings, leading to an accumulation of warnings and a degradation of code quality. This paper presents CodeCureAgent, an approach that harnesses LLM-based agents to automatically analyze, classify, and repair static analysis warnings. Unlike previous work, our method does not follow a predetermined algorithm. Instead, we adopt an agentic framework that iteratively invokes tools to gather additional information from the codebase (e.g., via code search) and edit the codebase to resolve the warning. CodeCureAgent detects and suppresses false positives, while fixing true positives when identified. We equip CodeCureAgent with a three-step heuristic to approve patches: (1) build the project, (2) verify that the warning disappears without introducing new warnings, and (3) run the test suite. We evaluate CodeCureAgent on a dataset of 1,000 SonarQube warnings found in 106 Java projects and covering 291 distinct rules. Our approach produces plausible fixes for 96.8% of the warnings, outperforming state-of-the-art baseline approaches by 30.7% and 29.2% in plausible-fix rate, respectively. Manual inspection of 291 cases reveals a correct-fix rate of 86.3%, showing that CodeCureAgent can reliably repair static analysis warnings. The approach incurs LLM costs of about 2.9 cents (USD) and an end-to-end processing time of about four minutes per warning. We envision CodeCureAgent helping to clean existing codebases and being integrated into CI/CD pipelines to prevent the accumulation of static analysis warnings.

**arXiv ID:** 2509.11787
</details>

<details>
<summary><strong>Agent Bain vs. Agent McKinsey: A New Text-to-SQL Benchmark for the Business Domain</strong> - Yue Li, Ran Tao, Derek Hommel, Yusuf Denizay Dönder, Sungyong Chang, David Mimno, Unso Eun Seo Jo - [[pdf]](https://arxiv.org/pdf/2510.07309)</summary>

**Abstract:** In the business domain, where data-driven decision making is crucial, text-to-SQL is fundamental for easy natural language access to structured data. While recent LLMs have achieved strong performance in code generation, existing text-to-SQL benchmarks remain focused on factual retrieval of past records. We introduce CORGI, a new benchmark specifically designed for real-world business contexts. CORGI is composed of synthetic databases inspired by enterprises such as Doordash, Airbnb, and Lululemon. It provides questions across four increasingly complex categories of business queries: descriptive, explanatory, predictive, and recommendational. This challenge calls for causal reasoning, temporal forecasting, and strategic recommendation, reflecting multi-level and multi-step agentic intelligence. We find that LLM performance drops on high-level questions, struggling to make accurate predictions and offer actionable plans. Based on execution success rate, the CORGI benchmark is about 21\% more difficult than the BIRD benchmark. This highlights the gap between popular LLMs and the need for real-world business intelligence. We release a public dataset and evaluation framework, and a website for public submissions.

**arXiv ID:** 2510.07309
</details>

<details>
<summary><strong>What Do Humans Hear When Interacting? Experiments on Selective Listening for Evaluating ASR of Spoken Dialogue Systems</strong> - Kiyotada Mori, Seiya Kawano, Chaoran Liu, Carlos Toshinori Ishi, Angel Fernando Garcia Contreras, Koichiro Yoshino - [[pdf]](https://arxiv.org/pdf/2508.04402)</summary>

**Abstract:** Spoken dialogue systems (SDSs) utilize automatic speech recognition (ASR) at the front end of their pipeline. The role of ASR in SDSs is to recognize information in user speech related to response generation appropriately. Examining selective listening of humans, which refers to the ability to focus on and listen to important parts of a conversation during the speech, will enable us to identify the ASR capabilities required for SDSs and evaluate them. In this study, we experimentally confirmed selective listening when humans generate dialogue responses by comparing human transcriptions for generating dialogue responses and reference transcriptions. Based on our experimental results, we discuss the possibility of a new ASR evaluation method that leverages human selective listening, which can identify the gap between transcription ability between ASR systems and humans.

**arXiv ID:** 2508.04402
</details>

<details>
<summary><strong>Milestone Determination for Autonomous Railway Operation</strong> - Josh Hunter, John McDermid, Simon Burton, Poppy Fynes, Mia Dempster - [[pdf]](https://arxiv.org/pdf/2510.06229)</summary>

**Abstract:** In the field of railway automation, one of the key challenges has been the development of effective computer vision systems due to the limited availability of high-quality, sequential data. Traditional datasets are restricted in scope, lacking the spatio temporal context necessary for real-time decision-making, while alternative solutions introduce issues related to realism and applicability. By focusing on route-specific, contextually relevant cues, we can generate rich, sequential datasets that align more closely with real-world operational logic. The concept of milestone determination allows for the development of targeted, rule-based models that simplify the learning process by eliminating the need for generalized recognition of dynamic components, focusing instead on the critical decision points along a route. We argue that this approach provides a practical framework for training vision agents in controlled, predictable environments, facilitating safer and more efficient machine learning systems for railway automation.

**arXiv ID:** 2510.06229
</details>

<details>
<summary><strong>IPR: Intelligent Prompt Routing with User-Controlled Quality-Cost Trade-offs</strong> - Aosong Feng, Balasubramaniam Srinivasan, Yun Zhou, Zhichao Xu, Kang Zhou, Sheng Guan, Yueyan Chen, Xian Wu, Ninad Kulkarni, Yi Zhang, Zhengyuan Shen, Dmitriy Bespalov, Soumya Smruti Mishra, Yifei Teng, Darren Yow-Bang Wang, Haibo Ding, Lin Lee Cheong - [[pdf]](https://arxiv.org/pdf/2509.06274)</summary>

**Abstract:** Routing incoming queries to the most cost-effective LLM while maintaining response quality poses a fundamental challenge in optimizing performance-cost trade-offs for large-scale commercial systems. We present IPR\, -- \,a quality-constrained \textbf{I}ntelligent \textbf{P}rompt \textbf{R}outing framework that dynamically selects optimal models based on predicted response quality and user-specified tolerance levels. IPR introduces three key innovations: (1) a modular architecture with lightweight quality estimators trained on 1.5M prompts annotated with calibrated quality scores, enabling fine-grained quality prediction across model families; (2) a user-controlled routing mechanism with tolerance parameter $\tau \in [0,1]$ that provides explicit control over quality-cost trade-offs; and (3) an extensible design using frozen encoders with model-specific adapters, reducing new model integration from days to hours. To rigorously train and evaluate IPR, we curate an industrial-level dataset IPRBench\footnote{IPRBench will be released upon legal approval.}, a comprehensive benchmark containing 1.5 million examples with response quality annotations across 11 LLM candidates. Deployed on a major cloud platform, IPR achieves 43.9\% cost reduction while maintaining quality parity with the strongest model in the Claude family and processes requests with sub-150ms latency. The deployed system and additional product details are publicly available at this https URL

**arXiv ID:** 2509.06274
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>Prompt Optimization Across Multiple Agents for Representing Diverse Human Populations</strong> - Manh Hung Nguyen, Sebastian Tschiatschek, Adish Singla - [[pdf]](https://arxiv.org/pdf/2510.07064)</summary>

**Abstract:** The difficulty and expense of obtaining large-scale human responses make Large Language Models (LLMs) an attractive alternative and a promising proxy for human behavior. However, prior work shows that LLMs often produce homogeneous outputs that fail to capture the rich diversity of human perspectives and behaviors. Thus, rather than trying to capture this diversity with a single LLM agent, we propose a novel framework to construct a set of agents that collectively capture the diversity of a given human population. Each agent is an LLM whose behavior is steered by conditioning on a small set of human demonstrations (task-response pairs) through in-context learning. The central challenge is therefore to select a representative set of LLM agents from the exponentially large space of possible agents. We tackle this selection problem from the lens of submodular optimization. In particular, we develop methods that offer different trade-offs regarding time complexity and performance guarantees. Extensive experiments in crowdsourcing and educational domains demonstrate that our approach constructs agents that more effectively represent human populations compared to baselines. Moreover, behavioral analyses on new tasks show that these agents reproduce the behavior patterns and perspectives of the students and annotators they are designed to represent.

**arXiv ID:** 2510.07064
</details>

<details>
<summary><strong>NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents</strong> - Tianshi Zheng, Kelvin Kiu-Wai Tam, Newt Hue-Nam K. Nguyen, Baixuan Xu, Zhaowei Wang, Jiayang Cheng, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Tianqing Fang, Yangqiu Song, Ginny Y. Wong, Simon See - [[pdf]](https://arxiv.org/pdf/2510.07172)</summary>

**Abstract:** Large language models are emerging as powerful tools for scientific law discovery, a foundational challenge in AI-driven science. However, existing benchmarks for this task suffer from a fundamental methodological trilemma, forcing a trade-off between scientific relevance, scalability, and resistance to memorization. Furthermore, they oversimplify discovery as static function fitting, failing to capture the authentic scientific process of uncovering embedded laws through the interactive exploration of complex model systems. To address these critical gaps, we introduce NewtonBench, a benchmark comprising 324 scientific law discovery tasks across 12 physics domains. Our design mitigates the evaluation trilemma by using metaphysical shifts - systematic alterations of canonical laws - to generate a vast suite of problems that are scalable, scientifically relevant, and memorization-resistant. Moreover, we elevate the evaluation from static function fitting to interactive model discovery, requiring agents to experimentally probe simulated complex systems to uncover hidden principles. Our extensive experiment reveals a clear but fragile capability for discovery in frontier LLMs: this ability degrades precipitously with increasing system complexity and exhibits extreme sensitivity to observational noise. Notably, we uncover a paradoxical effect of tool assistance: providing a code interpreter can hinder more capable models by inducing a premature shift from exploration to exploitation, causing them to satisfice on suboptimal solutions. These results demonstrate that robust, generalizable discovery in complex, interactive environments remains the core challenge. By providing a scalable, robust, and scientifically authentic testbed, NewtonBench offers a crucial tool for measuring true progress and guiding the development of next-generation AI agents capable of genuine scientific discovery.

**arXiv ID:** 2510.07172
</details>

<details>
<summary><strong>Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning</strong> - Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Kristian Kersting, Jeff Z. Pan, Hinrich Schütze, Volker Tresp, Yunpu Ma - [[pdf]](https://arxiv.org/pdf/2508.19828)</summary>

**Abstract:** Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of NLP tasks, but they remain fundamentally stateless, constrained by limited context windows that hinder long-horizon reasoning. Recent efforts to address this limitation often augment LLMs with an external memory bank, yet most existing pipelines are static and heuristic-driven, lacking a learned mechanism for deciding what to store, update, or retrieve. We present Memory-R1, a reinforcement learning (RL) framework that equips LLMs with the ability to actively manage and utilize external memory through two specialized agents: a Memory Manager that learns structured operations, including ADD, UPDATE, DELETE, and NOOP; and an Answer Agent that pre-selects and reasons over relevant entries. Both agents are fine-tuned with outcome-driven RL (PPO and GRPO), enabling adaptive memory management with minimal supervision. With only 152 training QA pairs, Memory-R1 outperforms strong baselines and generalizes across diverse question types, three benchmarks (LoCoMo, MSC, LongMemEval), and multiple model scales (3B-14B).

**arXiv ID:** 2508.19828
</details>

<details>
<summary><strong>Customer-R1: Personalized Simulation of Human Behaviors via RL-based LLM Agent in Online Shopping</strong> - Ziyi Wang, Yuxuan Lu, Yimeng Zhang, Jing Huang, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2510.07230)</summary>

**Abstract:** Simulating step-wise human behavior with Large Language Models (LLMs) has become an emerging research direction, enabling applications in various practical domains. While prior methods, including prompting, supervised fine-tuning (SFT), and reinforcement learning (RL), have shown promise in modeling step-wise behavior, they primarily learn a population-level policy without conditioning on a user's persona, yielding generic rather than personalized simulations. In this work, we pose a critical question: how can LLM agents better simulate personalized user behavior? We introduce Customer-R1, an RL-based method for personalized, step-wise user behavior simulation in online shopping environments. Our policy is conditioned on an explicit persona, and we optimize next-step rationale and action generation via action correctness reward signals. Experiments on the OPeRA dataset emonstrate that Customer-R1 not only significantly outperforms prompting and SFT-based baselines in next-action prediction tasks, but also better matches users' action distribution, indicating higher fidelity in personalized behavior simulation.

**arXiv ID:** 2510.07230
</details>

<details>
<summary><strong>Spiral of Silence in Large Language Model Agents</strong> - Mingze Zhong, Meng Fang, Zijing Shi, Yuxuan Huang, Shunfeng Zheng, Yali Du, Ling Chen, Jun Wang - [[pdf]](https://arxiv.org/pdf/2510.02360)</summary>

**Abstract:** The Spiral of Silence (SoS) theory holds that individuals with minority views often refrain from speaking out for fear of social isolation, enabling majority positions to dominate public discourse. When the 'agents' are large language models (LLMs), however, the classical psychological explanation is not directly applicable, since SoS was developed for human societies. This raises a central question: can SoS-like dynamics nevertheless emerge from purely statistical language generation in LLM collectives? We propose an evaluation framework for examining SoS in LLM agents. Specifically, we consider four controlled conditions that systematically vary the availability of 'History' and 'Persona' signals. Opinion dynamics are assessed using trend tests such as Mann-Kendall and Spearman's rank, along with concentration measures including kurtosis and interquartile range. Experiments across open-source and closed-source models show that history and persona together produce strong majority dominance and replicate SoS patterns; history signals alone induce strong anchoring; and persona signals alone foster diverse but uncorrelated opinions, indicating that without historical anchoring, SoS dynamics cannot emerge. The work bridges computational sociology and responsible AI design, highlighting the need to monitor and mitigate emergent conformity in LLM-agent systems.

**arXiv ID:** 2510.02360
</details>

<details>
<summary><strong>Inducing State Anxiety in LLM Agents Reproduces Human-Like Biases in Consumer Decision-Making</strong> - Ziv Ben-Zion, Zohar Elyoseph, Tobias Spiller, Teddy Lazebnik - [[pdf]](https://arxiv.org/pdf/2510.06222)</summary>

**Abstract:** Large language models (LLMs) are rapidly evolving from text generators to autonomous agents, raising urgent questions about their reliability in real-world contexts. Stress and anxiety are well known to bias human decision-making, particularly in consumer choices. Here, we tested whether LLM agents exhibit analogous vulnerabilities. Three advanced models (ChatGPT-5, Gemini 2.5, Claude 3.5-Sonnet) performed a grocery shopping task under budget constraints (24, 54, 108 USD), before and after exposure to anxiety-inducing traumatic narratives. Across 2,250 runs, traumatic prompts consistently reduced the nutritional quality of shopping baskets (Change in Basket Health Scores of -0.081 to -0.126; all pFDR<0.001; Cohens d=-1.07 to -2.05), robust across models and budgets. These results show that psychological context can systematically alter not only what LLMs generate but also the actions they perform. By reproducing human-like emotional biases in consumer behavior, LLM agents reveal a new class of vulnerabilities with implications for digital health, consumer safety, and ethical AI deployment.

**arXiv ID:** 2510.06222
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (21 papers)</h2></summary>

<details>
<summary><strong>Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks</strong> - Wentao Deng, Jiahuan Pei, Zhiwei Xu, Zhaochun Ren, Zhumin Chen, Pengjie Ren - [[pdf]](https://arxiv.org/pdf/2510.06307)</summary>

**Abstract:** A multi-agent system (MAS) enhances its capacity to solve complex natural language processing (NLP) tasks through collaboration among multiple agents, where consensus-seeking serves as a fundamental mechanism. However, existing consensus-seeking approaches typically rely on voting mechanisms to judge consensus, overlooking contradictions in system-internal beliefs that destabilize the consensus. Moreover, these methods often involve agents updating their results through indiscriminate collaboration with every other agent. Such uniform interaction fails to identify the optimal collaborators for each agent, hindering the emergence of a stable consensus. To address these challenges, we provide a theoretical framework for selecting optimal collaborators that maximize consensus stability. Based on the theorems, we propose the Belief-Calibrated Consensus Seeking (BCCS) framework to facilitate stable consensus via selecting optimal collaborators and calibrating the consensus judgment by system-internal beliefs. Experimental results on the MATH and MMLU benchmark datasets demonstrate that the proposed BCCS framework outperforms the best existing results by 2.23% and 3.95% of accuracy on challenging tasks, respectively. Our code and data are available at this https URL.

**arXiv ID:** 2510.06307
</details>

<details>
<summary><strong>Evolving and Executing Research Plans via Double-Loop Multi-Agent Collaboration</strong> - Zhi Zhang, Yan Liu, Zhejing Hu, Gong Chen, Sheng-hua Zhong, Jiannong Cao - [[pdf]](https://arxiv.org/pdf/2510.06761)</summary>

**Abstract:** Automating the end-to-end scientific research process poses a fundamental challenge: it requires both evolving high-level plans that are novel and sound, and executing these plans correctly amidst dynamic and uncertain conditions. To address this bilevel challenge, we propose a novel Double-Loop Multi-Agent (DLMA) framework to solve the given research problem automatically. The leader loop, composed of professor agents, is responsible for evolving research plans. It employs an evolutionary algorithm through involvement, improvement, and integration meetings to iteratively generate and refine a pool of research proposals, exploring the solution space effectively. The follower loop, composed of doctoral student agents, is responsible for executing the best-evolved plan. It dynamically adjusts the plan during implementation via pre-hoc and post-hoc meetings, ensuring each step (e.g., drafting, coding) is well-supported by contextual and external observations. Extensive experiments on benchmarks like ACLAward and Laboratory show that DLMA generates research papers that achieve state-of-the-art scores in automated evaluation, significantly outperforming strong baselines. Ablation studies confirm the critical roles of both loops, with evolution driving novelty and execution ensuring soundness.

**arXiv ID:** 2510.06761
</details>

<details>
<summary><strong>LLM-Assisted Modeling of Semantic Web-Enabled Multi-Agents Systems with AJAN</strong> - Hacane Hechehouche, Andre Antakli, Matthias Klusch - [[pdf]](https://arxiv.org/pdf/2510.06911)</summary>

**Abstract:** There are many established semantic Web standards for implementing multi-agent driven applications. The AJAN framework allows to engineer multi-agent systems based on these standards. In particular, agent knowledge is represented in RDF/RDFS and OWL, while agent behavior models are defined with Behavior Trees and SPARQL to access and manipulate this knowledge. However, the appropriate definition of RDF/RDFS and SPARQL-based agent behaviors still remains a major hurdle not only for agent modelers in practice. For example, dealing with URIs is very error-prone regarding typos and dealing with complex SPARQL queries in large-scale environments requires a high learning curve. In this paper, we present an integrated development environment to overcome such hurdles of modeling AJAN agents and at the same time to extend the user community for AJAN by the possibility to leverage Large Language Models for agent engineering.

**arXiv ID:** 2510.06911
</details>

<details>
<summary><strong>Multi-Objective Multi-Agent Path Finding with Lexicographic Cost Preferences</strong> - Pulkit Rustagi, Kyle Hollins Wray, Sandhya Saisubramanian - [[pdf]](https://arxiv.org/pdf/2510.07276)</summary>

**Abstract:** Many real-world scenarios require multiple agents to coordinate in shared environments, while balancing trade-offs between multiple, potentially competing objectives. Current multi-objective multi-agent path finding (MO-MAPF) algorithms typically produce conflict-free plans by computing Pareto frontiers. They do not explicitly optimize for user-defined preferences, even when the preferences are available, and scale poorly with the number of objectives. We propose a lexicographic framework for modeling MO-MAPF, along with an algorithm \textit{Lexicographic Conflict-Based Search} (LCBS) that directly computes a single solution aligned with a lexicographic preference over objectives. LCBS integrates a priority-aware low-level $A^*$ search with conflict-based search, avoiding Pareto frontier construction and enabling efficient planning guided by preference over objectives. We provide insights into optimality and scalability, and empirically demonstrate that LCBS computes optimal solutions while scaling to instances with up to ten objectives -- far beyond the limits of existing MO-MAPF methods. Evaluations on standard and randomized MAPF benchmarks show consistently higher success rates against state-of-the-art baselines, especially with increasing number of objectives.

**arXiv ID:** 2510.07276
</details>

<details>
<summary><strong>Exploring Human-AI Collaboration Using Mental Models of Early Adopters of Multi-Agent Generative AI Tools</strong> - Suchismita Naik, Austin L. Toombs, Amanda Snellinger, Scott Saponas, Amanda K. Hall - [[pdf]](https://arxiv.org/pdf/2510.06224)</summary>

**Abstract:** With recent advancements in multi-agent generative AI (Gen AI), technology organizations like Microsoft are adopting these complex tools, redefining AI agents as active collaborators in complex workflows rather than as passive tools. In this study, we investigated how early adopters and developers conceptualize multi-agent Gen AI tools, focusing on how they understand human-AI collaboration mechanisms, general collaboration dynamics, and transparency in the context of AI tools. We conducted semi-structured interviews with 13 developers, all early adopters of multi-agent Gen AI technology who work at Microsoft. Our findings revealed that these early adopters conceptualize multi-agent systems as "teams" of specialized role-based and task-based agents, such as assistants or reviewers, structured similar to human collaboration models and ranging from AI-dominant to AI-assisted, user-controlled interactions. We identified key challenges, including error propagation, unpredictable and unproductive agent loop behavior, and the need for clear communication to mitigate the layered transparency issues. Early adopters' perspectives about the role of transparency underscored its importance as a way to build trust, verify and trace errors, and prevent misuse, errors, and leaks. The insights and design considerations we present contribute to CSCW research about collaborative mechanisms with capabilities ranging from AI-dominant to AI-assisted interactions, transparency and oversight strategies in human-agent and agent-agent interactions, and how humans make sense of these multi-agent systems as dynamic, role-diverse collaborators which are customizable for diverse needs and workflows. We conclude with future research directions that extend CSCW approaches to the design of inter-agent and human mediation interactions.

**arXiv ID:** 2510.06224
</details>

<details>
<summary><strong>Generalized Multi-agent Social Simulation Framework</strong> - Gang Li, Jie Lin, Yining Tang, Ziteng Wang, Yirui Huang, Junyu Zhang, Shuang Luo, Chao Wu, Yike Guo - [[pdf]](https://arxiv.org/pdf/2510.06225)</summary>

**Abstract:** Multi-agent social interaction has clearly benefited from Large Language Models. However, current simulation systems still face challenges such as difficulties in scaling to diverse scenarios and poor reusability due to a lack of modular design. To address these issues, we designed and developed a modular, object-oriented framework that organically integrates various base classes through a hierarchical structure, harvesting scalability and reusability. We inherited the framework to realize common derived classes. Additionally, a memory summarization mechanism is proposed to filter and distill relevant information from raw memory data, prioritizing contextually salient events and interactions. By selecting and combining some necessary derived classes, we customized a specific simulated environment. Utilizing this simulated environment, we successfully simulated human interactions on social media, replicating real-world online social behaviors. The source code for the project will be released and evolve.

**arXiv ID:** 2510.06225
</details>

<details>
<summary><strong>Knowledge Graph-Guided Multi-Agent Distillation for Reliable Industrial Question Answering with Datasets</strong> - Jiqun Pan, Zhenke Duan, Jiani Tu, Anzhi Cheng, Yanqing Wang - [[pdf]](https://arxiv.org/pdf/2510.06240)</summary>

**Abstract:** Industrial question-answering (QA) systems require higher safety and reliability than general-purpose dialogue models, as errors in high-risk scenarios such as equipment fault diagnosis can have severe consequences. Although multi-agent large language models enhance reasoning depth, they suffer from uncontrolled iterations and unverifiable outputs, and conventional distillation methods struggle to transfer collaborative reasoning capabilities to lightweight, deployable student models. To address these challenges, we propose Knowledge Graph-guided Multi-Agent System Distillation (KG-MASD). Our approach formulates distillation as a Markov Decision Process and incorporates a knowledge graph as a verifiable structured prior to enrich state representation and ensure convergence. By integrating collaborative reasoning with knowledge grounding, KG-MASD generates high-confidence instruction-tuning data and jointly distills reasoning depth and verifiability into compact student models suitable for edge deployment. Experiments on an industrial QA dataset show that KG-MASD improves accuracy by 2.4 per cent to 20.1 per cent over baselines and significantly enhances reliability, enabling trustworthy AI deployment in safety-critical industrial scenarios. Code and data are available at this https URL.

**arXiv ID:** 2510.06240
</details>

<details>
<summary><strong>FURINA: A Fully Customizable Role-Playing Benchmark via Scalable Multi-Agent Collaboration Pipeline</strong> - Haotian Wu, Shufan Jiang, Chios Chen, Yiyang Feng, Hehai Lin, Heqing Zou, Yao Shu, Yanran Li, Chengwei Qin - [[pdf]](https://arxiv.org/pdf/2510.06800)</summary>

**Abstract:** As large language models (LLMs) advance in role-playing (RP) tasks, existing benchmarks quickly become obsolete due to their narrow scope, outdated interaction paradigms, and limited adaptability across diverse application scenarios. To address this gap, we introduce FURINA-Builder, a novel multi-agent collaboration pipeline that automatically constructs fully customizable RP benchmarks at any scale. It enables evaluation of arbitrary characters across diverse scenarios and prompt formats, as the first benchmark builder in RP area for adaptable assessment. FURINA-Builder simulates dialogues between a test character and other characters drawn from a well-constructed character-scene pool, while an LLM judge selects fine-grained evaluation dimensions and adjusts the test character's responses into final test utterances. Using this pipeline, we build FURINA-Bench, a new comprehensive role-playing benchmark featuring both established and synthesized test characters, each assessed with dimension-specific evaluation criteria. Human evaluation and preliminary separability analysis justify our pipeline and benchmark design. We conduct extensive evaluations of cutting-edge LLMs and find that o3 and DeepSeek-R1 achieve the best performance on English and Chinese RP tasks, respectively. Across all models, established characters consistently outperform synthesized ones, with reasoning capabilities further amplifying this disparity. Interestingly, we observe that model scale does not monotonically reduce hallucinations. More critically, for reasoning LLMs, we uncover a novel trade-off: reasoning improves RP performance but simultaneously increases RP hallucinations. This trade-off extends to a broader Pareto frontier between RP performance and reliability for all LLMs. These findings demonstrate the effectiveness of FURINA-Builder and the challenge posed by FURINA-Bench.

**arXiv ID:** 2510.06800
</details>

<details>
<summary><strong>DecompGAIL: Learning Realistic Traffic Behaviors with Decomposed Multi-Agent Generative Adversarial Imitation Learning</strong> - Ke Guo, Haochen Liu, Xiaojun Wu, Chen Lv - [[pdf]](https://arxiv.org/pdf/2510.06913)</summary>

**Abstract:** Realistic traffic simulation is critical for the development of autonomous driving systems and urban mobility planning, yet existing imitation learning approaches often fail to model realistic traffic behaviors. Behavior cloning suffers from covariate shift, while Generative Adversarial Imitation Learning (GAIL) is notoriously unstable in multi-agent settings. We identify a key source of this instability: irrelevant interaction misguidance, where a discriminator penalizes an ego vehicle's realistic behavior due to unrealistic interactions among its neighbors. To address this, we propose Decomposed Multi-agent GAIL (DecompGAIL), which explicitly decomposes realism into ego-map and ego-neighbor components, filtering out misleading neighbor: neighbor and neighbor: map interactions. We further introduce a social PPO objective that augments ego rewards with distance-weighted neighborhood rewards, encouraging overall realism across agents. Integrated into a lightweight SMART-based backbone, DecompGAIL achieves state-of-the-art performance on the WOMD Sim Agents 2025 benchmark.

**arXiv ID:** 2510.06913
</details>

<details>
<summary><strong>A Multi-Agent Framework for Stateful Inference-Time Search</strong> - Arshika Lalan, Rajat Ghosh, Aditya Kolsur, Debojyoti Dutta - [[pdf]](https://arxiv.org/pdf/2510.07147)</summary>

**Abstract:** Recent work explores agentic inference-time techniques to perform structured, multi-step reasoning. However, stateless inference often struggles on multi-step tasks due to the absence of persistent state. Moreover, task-specific fine-tuning or instruction-tuning often achieve surface-level code generation but remain brittle on tasks requiring deeper reasoning and long-horizon dependencies. To address these limitations, we propose stateful multi-agent evolutionary search, a training-free framework that departs from prior stateless approaches by combining (i) persistent inference-time state, (ii) adversarial mutation, and (iii) evolutionary preservation. We demonstrate its effectiveness in automated unit test generation through the generation of edge cases. We generate robust edge cases using an evolutionary search process, where specialized agents sequentially propose, mutate, and score candidates. A controller maintains persistent state across generations, while evolutionary preservation ensures diversity and exploration across all possible cases. This yields a generalist agent capable of discovering robust, high-coverage edge cases across unseen codebases. Experiments show our stateful multi-agent inference framework achieves substantial gains in coverage over stateless single-step baselines, evaluated on prevalent unit-testing benchmarks such as HumanEval and TestGenEvalMini and using three diverse LLM families - Llama, Gemma, and GPT. These results indicate that combining persistent inference-time state with evolutionary search materially improves unit-test generation.

**arXiv ID:** 2510.07147
</details>

<details>
<summary><strong>HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving</strong> - Donald Pfaffmann, Matthias Klusch, Marcel Steinmetz - [[pdf]](https://arxiv.org/pdf/2510.07210)</summary>

**Abstract:** We present a novel hybrid learning-assisted planning method, named HyPlan, for solving the collision-free navigation problem for self-driving cars in partially observable traffic environments. HyPlan combines methods for multi-agent behavior prediction, deep reinforcement learning with proximal policy optimization and approximated online POMDP planning with heuristic confidence-based vertical pruning to reduce its execution time without compromising safety of driving. Our experimental performance analysis on the CARLA-CTS2 benchmark of critical traffic scenarios with pedestrians revealed that HyPlan may navigate safer than selected relevant baselines and perform significantly faster than considered alternative online POMDP planners.

**arXiv ID:** 2510.07210
</details>

<details>
<summary><strong>GenPilot: A Multi-Agent System for Test-Time Prompt Optimization in Image Generation</strong> - Wen Ye, Zhaocheng Liu, Yuwei Gui, Tingyu Yuan, Yunyue Su, Bowen Fang, Chaoyang Zhao, Qiang Liu, Liang Wang - [[pdf]](https://arxiv.org/pdf/2510.07217)</summary>

**Abstract:** Text-to-image synthesis has made remarkable progress, yet accurately interpreting complex and lengthy prompts remains challenging, often resulting in semantic inconsistencies and missing details. Existing solutions, such as fine-tuning, are model-specific and require training, while prior automatic prompt optimization (APO) approaches typically lack systematic error analysis and refinement strategies, resulting in limited reliability and effectiveness. Meanwhile, test-time scaling methods operate on fixed prompts and on noise or sample numbers, limiting their interpretability and adaptability. To solve these, we introduce a flexible and efficient test-time prompt optimization strategy that operates directly on the input text. We propose a plug-and-play multi-agent system called GenPilot, integrating error analysis, clustering-based adaptive exploration, fine-grained verification, and a memory module for iterative optimization. Our approach is model-agnostic, interpretable, and well-suited for handling long and complex prompts. Simultaneously, we summarize the common patterns of errors and the refinement strategy, offering more experience and encouraging further exploration. Experiments on DPG-bench and Geneval with improvements of up to 16.9% and 5.7% demonstrate the strong capability of our methods in enhancing the text and image consistency and structural coherence of generated images, revealing the effectiveness of our test-time prompt optimization strategy. The code is available at this https URL.

**arXiv ID:** 2510.07217
</details>

<details>
<summary><strong>MLE-Smith: Scaling MLE Tasks with Automated Multi-Agent Pipeline</strong> - Rushi Qiang, Yuchen Zhuang, Anikait Singh, Percy Liang, Chao Zhang, Sherry Yang, Bo Dai - [[pdf]](https://arxiv.org/pdf/2510.07307)</summary>

**Abstract:** While Language Models (LMs) have made significant progress in automating machine learning engineering (MLE), the acquisition of high-quality MLE training data is significantly constrained. Current MLE benchmarks suffer from low scalability and limited applicability because they rely on static, manually curated tasks, demanding extensive time and manual effort to produce. We introduce MLE-Smith, a fully automated multi-agent pipeline, to transform raw datasets into competition-style MLE challenges through an efficient generate-verify-execute paradigm for scaling MLE tasks with verifiable quality, real-world usability, and rich diversity. The proposed multi-agent pipeline in MLE-Smith drives structured task design and standardized refactoring, coupled with a hybrid verification mechanism that enforces strict structural rules and high-level semantic soundness. It further validates empirical solvability and real-world fidelity through interactive execution. We apply MLE-Smith to 224 of real-world datasets and generate 606 tasks spanning multiple categories, objectives, and modalities, demonstrating that MLE-Smith can work effectively across a wide range of real-world datasets. Evaluation on the generated tasks shows that the performance of eight mainstream and cutting-edge LLMs on MLE-Smith tasks is strongly correlated with their performance on carefully human-designed tasks, highlighting the effectiveness of the MLE-Smith to scaling up MLE tasks, while maintaining task quality.

**arXiv ID:** 2510.07307
</details>

<details>
<summary><strong>Code Like Humans: A Multi-Agent Solution for Medical Coding</strong> - Andreas Motzfeldt, Joakim Edin, Casper L. Christensen, Christian Hardmeier, Lars Maaløe, Anna Rogers - [[pdf]](https://arxiv.org/pdf/2509.05378)</summary>

**Abstract:** In medical coding, experts map unstructured clinical notes to alphanumeric codes for diagnoses and procedures. We introduce Code Like Humans: a new agentic framework for medical coding with large language models. It implements official coding guidelines for human experts, and it is the first solution that can support the full ICD-10 coding system (+70K labels). It achieves the best performance to date on rare diagnosis codes (fine-tuned discriminative classifiers retain an advantage for high-frequency codes, to which they are limited). Towards future work, we also contribute an analysis of system performance and identify its `blind spots' (codes that are systematically undercoded).

**arXiv ID:** 2509.05378
</details>

<details>
<summary><strong>R3R: Decentralized Multi-Agent Collision Avoidance with Infinite-Horizon Safety</strong> - Thomas Marshall Vielmetti, Devansh R. Agrawal, Dimitra Panagou - [[pdf]](https://arxiv.org/pdf/2510.06436)</summary>

**Abstract:** Existing decentralized methods for multi-agent motion planning lack formal, infinite-horizon safety guarantees, especially for communication-constrained systems. We present R3R, to our knowledge the first decentralized and asynchronous framework for multi-agent motion planning under distance-based communication constraints with infinite-horizon safety guarantees for systems of nonlinear agents. R3R's novelty lies in combining our gatekeeper safety framework with a geometric constraint called R-Boundedness, which together establish a formal link between an agent's communication radius and its ability to plan safely. We constrain trajectories to within a fixed planning radius that is a function of the agent's communication radius, which enables trajectories to be shown provably safe for all time, using only local information. Our algorithm is fully asynchronous, and ensures the forward invariance of these guarantees even in time-varying networks where agents asynchronously join, leave, and replan. We validate our approach in simulations of up to 128 Dubins vehicles, demonstrating 100% safety in dense, obstacle rich scenarios. Our results demonstrate that R3R's performance scales with agent density rather than problem size, providing a practical solution for scalable and provably safe multi-agent systems.

**arXiv ID:** 2510.06436
</details>

<details>
<summary><strong>Generalizing Liquid Democracy to multi-agent delegation: A Voting Weight Measure and Equilibrium Analysis</strong> - Francisco M. Bersetche - [[pdf]](https://arxiv.org/pdf/2209.14128)</summary>

**Abstract:** In this study, we propose a generalization of the classic model of liquid democracy that allows fractional delegation of voting weight, while simultaneously allowing for the existence of equilibrium states. Our approach empowers agents to partition and delegate their votes to multiple representatives, all while retaining a fraction of the voting power for themselves. We introduce a penalty mechanism for the length of delegation chains. We discuss the desirable properties of a reasonable generalization of the classic model, and prove that smaller penalty factors bring the model closer to satisfying these properties. In the subsequent section, we explore the presence of equilibrium states in a general delegation game utilizing the proposed voting measure. In contrast to the classical model, we demonstrate that this game exhibits pure strategy Nash equilibria, contingent upon the imposition of a penalty on the length of delegation chains.

**arXiv ID:** 2209.14128
</details>

<details>
<summary><strong>Dynamic Strategy Adaptation in Multi-Agent Environments with Large Language Models</strong> - Shaurya Mallampati, Rashed Shelim, Walid Saad, Naren Ramakrishnan - [[pdf]](https://arxiv.org/pdf/2507.02002)</summary>

**Abstract:** Large language models (LLMs) demonstrate strong reasoning abilities across mathematical, strategic, and linguistic tasks, yet little is known about how well they reason in dynamic, real-time, multi-agent scenarios, such as collaborative environments in which agents continuously adapt to each other's behavior, as in cooperative gameplay settings. In this paper, we bridge this gap by combining LLM-driven agents with strategic reasoning and real-time adaptation in cooperative, multi-agent environments grounded in game-theoretic principles such as belief consistency and Nash equilibrium. The proposed framework applies broadly to dynamic scenarios in which agents coordinate, communicate, and make decisions in response to continuously changing conditions. We provide real-time strategy refinement and adaptive feedback mechanisms that enable agents to dynamically adjust policies based on immediate contextual interactions, in contrast to previous efforts that evaluate LLM capabilities in static or turn-based settings. Empirical results show that our method achieves up to a 26\% improvement in return over PPO baselines in high-noise environments, while maintaining real-time latency under 1.05 milliseconds. Our approach improves collaboration efficiency, task completion rates, and flexibility, illustrating that game-theoretic guidance integrated with real-time feedback enhances LLM performance, ultimately fostering more resilient and flexible strategic multi-agent systems.

**arXiv ID:** 2507.02002
</details>

<details>
<summary><strong>TinyScientist: An Interactive, Extensible, and Controllable Framework for Building Research Agents</strong> - Haofei Yu, Keyang Xuan, Fenghai Li, Kunlun Zhu, Zijie Lei, Jiaxun Zhang, Ziheng Qi, Kyle Richardson, Jiaxuan You - [[pdf]](https://arxiv.org/pdf/2510.06579)</summary>

**Abstract:** Automatic research with Large Language Models (LLMs) is rapidly gaining importance, driving the development of increasingly complex workflows involving multi-agent systems, planning, tool usage, code execution, and human-agent interaction to accelerate research processes. However, as more researchers and developers begin to use and build upon these tools and platforms, the complexity and difficulty of extending and maintaining such agentic workflows have become a significant challenge, particularly as algorithms and architectures continue to advance. To address this growing complexity, TinyScientist identifies the essential components of the automatic research workflow and proposes an interactive, extensible, and controllable framework that easily adapts to new tools and supports iterative growth. We provide an open-source codebase, an interactive web demonstration, and a PyPI Python package to make state-of-the-art auto-research pipelines broadly accessible to every researcher and developer.

**arXiv ID:** 2510.06579
</details>

<details>
<summary><strong>Adaptive Tool Generation with Models as Tools and Reinforcement Learning</strong> - Chenpeng Wang, Xiaojie Cheng, Chunye Wang, Linfeng Yang, Lei Zhang - [[pdf]](https://arxiv.org/pdf/2510.06825)</summary>

**Abstract:** Tool-augmented language models have demonstrated strong capabilities, but their reliance on live API access creates scalability and reliability challenges during training and deployment. We propose MTR, a simulation-first training framework for tool-augmented reasoning. Instead of relying on live APIs, MTR learns from complete ReAct traces with schema-validated, simulated observations. Our approach operates through a multi-agent architecture where a ToolMaker generates task-specific, OpenAI-compatible tool interfaces, an AutoAgent produces structured think-act-observe sequences, and a ToolActor simulates realistic responses. Training proceeds in two stages: Stage-1 Supervised Fine-Tuning (SFT) teaches 'trace grammar' from complete reasoning sequences; Stage-2 Group Relative Policy Optimization (GRPO) optimizes strategy with a composite trace reward that balances answer correctness and internal consistency. Across four multi-hop QA benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA, Bamboogle), MTR attains competitive Exact Match (EM) scores to live-API systems and excels on reasoning-intensive tasks, suggesting that effective tool reasoning can be learned from structured traces without live interactions.

**arXiv ID:** 2510.06825
</details>

<details>
<summary><strong>LatteReview: A Multi-Agent Framework for Systematic Review Automation Using Large Language Models</strong> - Pouria Rouzrokh, Bardia Khosravi, Parsa Rouzrokh, Moein Shariatnia - [[pdf]](https://arxiv.org/pdf/2501.05468)</summary>

**Abstract:** Systematic literature reviews and meta-analyses are essential for synthesizing research insights, but they remain time-intensive and labor-intensive due to the iterative processes of screening, evaluation, and data extraction. This paper introduces and evaluates LatteReview, a Python-based framework that leverages large language models (LLMs) and multi-agent systems to automate key elements of the systematic review process. Designed to streamline workflows while maintaining rigor, LatteReview utilizes modular agents for tasks such as title and abstract screening, relevance scoring, and structured data extraction. These agents operate within orchestrated workflows, supporting sequential and parallel review rounds, dynamic decision-making, and iterative refinement based on user feedback. LatteReview's architecture integrates LLM providers, enabling compatibility with both cloud-based and locally hosted models. The framework supports features such as Retrieval-Augmented Generation (RAG) for incorporating external context, multimodal reviews, Pydantic-based validation for structured inputs and outputs, and asynchronous programming for handling large-scale datasets. The framework is available on the GitHub repository, with detailed documentation and an installable package.

**arXiv ID:** 2501.05468
</details>

<details>
<summary><strong>Distributed Algorithms for Multi-Agent Multi-Armed Bandits with Collision</strong> - Daoyuan Zhou, Xuchuang Wang, Lin Yang, Yang Gao - [[pdf]](https://arxiv.org/pdf/2510.06683)</summary>

**Abstract:** We study the stochastic Multiplayer Multi-Armed Bandit (MMAB) problem, where multiple players select arms to maximize their cumulative rewards. Collisions occur when two or more players select the same arm, resulting in no reward, and are observed by the players involved. We consider a distributed setting without central coordination, where each player can only observe their own actions and collision feedback. We propose a distributed algorithm with an adaptive, efficient communication protocol. The algorithm achieves near-optimal group and individual regret, with a communication cost of only $\mathcal{O}(\log\log T)$. Our experiments demonstrate significant performance improvements over existing baselines. Compared to state-of-the-art (SOTA) methods, our approach achieves a notable reduction in individual regret. Finally, we extend our approach to a periodic asynchronous setting, proving the lower bound for this problem and presenting an algorithm that achieves logarithmic regret.

**arXiv ID:** 2510.06683
</details>

</details>

<details open>
<summary><h2>Other Agent Research (9 papers)</h2></summary>

<details>
<summary><strong>VRPAgent: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems</strong> - André Hottung, Federico Berto, Chuanbo Hua, Nayeli Gast Zepeda, Daniel Wetzel, Michael Römer, Haoran Ye, Davide Zago, Michael Poli, Stefano Massaroli, Jinkyoo Park, Kevin Tierney - [[pdf]](https://arxiv.org/pdf/2510.07073)</summary>

**Abstract:** Designing high-performing heuristics for vehicle routing problems (VRPs) is a complex task that requires both intuition and deep domain knowledge. Large language model (LLM)-based code generation has recently shown promise across many domains, but it still falls short of producing heuristics that rival those crafted by human experts. In this paper, we propose VRPAgent, a framework that integrates LLM-generated components into a metaheuristic and refines them through a novel genetic search. By using the LLM to generate problem-specific operators, embedded within a generic metaheuristic framework, VRPAgent keeps tasks manageable, guarantees correctness, and still enables the discovery of novel and powerful strategies. Across multiple problems, including the capacitated VRP, the VRP with time windows, and the prize-collecting VRP, our method discovers heuristic operators that outperform handcrafted methods and recent learning-based approaches while requiring only a single CPU core. To our knowledge, \VRPAgent is the first LLM-based paradigm to advance the state-of-the-art in VRPs, highlighting a promising future for automated heuristics discovery.

**arXiv ID:** 2510.07073
</details>

<details>
<summary><strong>Agentic generative AI for media content discovery at the national football league</strong> - Henry Wang, Sirajus Salekin, Jake Lee, Ross Claytor, Shinan Zhang, Michael Chi - [[pdf]](https://arxiv.org/pdf/2510.07297)</summary>

**Abstract:** Generative AI has unlocked new possibilities in content discovery and management. Through collaboration with the National Football League (NFL), we demonstrate how a generative-AI based workflow enables media researchers and analysts to query relevant historical plays using natural language rather than traditional filter-and-click interfaces. The agentic workflow takes a user query as input, breaks it into elements, and translates them into the underlying database query language. Accuracy and latency are further improved through carefully designed semantic caching. The solution achieves over 95 percent accuracy and reduces the average time to find relevant videos from 10 minutes to 30 seconds, significantly increasing the NFL's operational efficiency and allowing users to focus on producing creative content and engaging storylines.

**arXiv ID:** 2510.07297
</details>

<details>
<summary><strong>AgentBuilder: Exploring Scaffolds for Prototyping User Experiences of Interface Agents</strong> - Jenny T. Liang, Titus Barik, Jeffrey Nichols, Eldon Schoop, Ruijia Cheng - [[pdf]](https://arxiv.org/pdf/2510.04452)</summary>

**Abstract:** Interface agents powered by generative AI models (referred to as "agents") can automate actions based on user commands. An important aspect of developing agents is their user experience (i.e., agent experience). There is a growing need to provide scaffolds for a broader set of individuals beyond AI engineers to prototype agent experiences, since they can contribute valuable perspectives to designing agent experiences. In this work, we explore the affordances agent prototyping systems should offer by conducting a requirements elicitation study with 12 participants with varying experience with agents. We identify key activities in agent experience prototyping and the desired capabilities of agent prototyping systems. We instantiate those capabilities in the AgentBuilder design probe for agent prototyping. We conduct an in situ agent prototyping study with 14 participants using AgentBuilder to validate the design requirements and elicit insights on how developers prototype agents and what their needs are in this process.

**arXiv ID:** 2510.04452
</details>

<details>
<summary><strong>A Survey on Agentic Security: Applications, Threats and Defenses</strong> - Asif Shahriar, Md Nafiu Rahman, Sadif Ahmed, Farig Sadeque, Md Rizwan Parvez - [[pdf]](https://arxiv.org/pdf/2510.06445)</summary>

**Abstract:** The rapid shift from passive LLMs to autonomous LLM-agents marks a new paradigm in cybersecurity. While these agents can act as powerful tools for both offensive and defensive operations, the very agentic context introduces a new class of inherent security risks. In this work we present the first holistic survey of the agentic security landscape, structuring the field around three interdependent pillars: Applications, Threats, and Defenses. We provide a comprehensive taxonomy of over 150 papers, explaining how agents are used, the vulnerabilities they possess, and the countermeasures designed to protect them. A detailed cross-cutting analysis shows emerging trends in agent architecture while revealing critical research gaps in model and modality coverage.

**arXiv ID:** 2510.06445
</details>

<details>
<summary><strong>Dyna-Think: Synergizing Reasoning, Acting, and World Model Simulation in AI Agents</strong> - Xiao Yu, Baolin Peng, Ruize Xu, Michel Galley, Hao Cheng, Suman Nath, Jianfeng Gao, Zhou Yu - [[pdf]](https://arxiv.org/pdf/2506.00320)</summary>

**Abstract:** Recent progress in reasoning with large language models (LLMs), such as DeepSeek-R1, demonstrates impressive capabilities in domains like mathematics and coding, by exhibiting complex cognitive behaviors such as verification, goal decomposition, and self-reflection. However, it is unclear what behavior is effective and what behavior is missing for long-horizon AI agents tasks. In this work, we propose Dyna-Think, a thinking framework that integrates planning with an internal world model with reasoning and acting to enhance AI agent performance. To enable Dyna-Think, we propose Dyna-Think Imitation Learning (DIT) and Dyna-Think Dyna Training (DDT). To initialize a policy with Dyna-Think, DIT reconstructs the thinking process of R1 to focus on performing world model simulation relevant to the proposed (and planned) action, and trains the policy using this reconstructed data. To enhance Dyna-Think, DDT uses a two-stage training process to first improve the agent's world modeling ability via objectives such as state prediction or critique generation, and then improve the agent's action via policy training. We evaluate our methods on OSWorld and WindowsAgentArena, and demonstrate that Dyna-Think improves the agent's in-domain and out-of-domain performance, achieving similar best-of-n performance compared to R1 while generating 2x less tokens on average. Our extensive empirical studies reveal that 1) using critique generation for world model training is effective to improve policy performance; and 2) AI agents with better performance correlate with better world modeling abilities. We believe our results suggest a promising research direction to integrate world model simulation into AI agents to enhance their reasoning, planning, and acting capabilities.

**arXiv ID:** 2506.00320
</details>

<details>
<summary><strong>Toward Causal-Visual Programming: Enhancing Agentic Reasoning in Low-Code Environments</strong> - Jiexi Xu, Jiaqi Liu, Lanruo Wang, Su Liu - [[pdf]](https://arxiv.org/pdf/2509.25282)</summary>

**Abstract:** Large language model (LLM) agents are increasingly capable of orchestrating complex tasks in low-code environments. However, these agents often exhibit hallucinations and logical inconsistencies because their inherent reasoning mechanisms rely on probabilistic associations rather than genuine causal understanding. This paper introduces a new programming paradigm: Causal-Visual Programming (CVP), designed to address this fundamental issue by explicitly introducing causal structures into the workflow design. CVP allows users to define a simple "world model" for workflow modules through an intuitive low-code interface, effectively creating a Directed Acyclic Graph (DAG) that explicitly defines the causal relationships between modules. This causal graph acts as a crucial constraint during the agent's reasoning process, anchoring its decisions to a user-defined causal structure and significantly reducing logical errors and hallucinations by preventing reliance on spurious correlations. To validate the effectiveness of CVP, we designed a synthetic experiment that simulates a common real-world problem: a distribution shift between the training and test environments. Our results show that a causally anchored model maintained stable accuracy in the face of this shift, whereas a purely associative baseline model that relied on probabilistic correlations experienced a significant performance drop. The primary contributions of this study are: a formal definition of causal structures for workflow modules; the proposal and implementation of a CVP framework that anchors agent reasoning to a user-defined causal graph; and empirical evidence demonstrating the framework's effectiveness in enhancing agent robustness and reducing errors caused by causal confusion in dynamic environments. CVP offers a viable path toward building more interpretable, reliable, and trustworthy AI agents.

**arXiv ID:** 2509.25282
</details>

<details>
<summary><strong>ToolMem: Enhancing Multimodal Agents with Learnable Tool Capability Memory</strong> - Yunzhong Xiao, Yangmin Li, Hewei Wang, Yunlong Tang, Zora Zhiruo Wang - [[pdf]](https://arxiv.org/pdf/2510.06664)</summary>

**Abstract:** Agents utilizing tools powered by large language models (LLMs) or vision-language models (VLMs) have demonstrated remarkable progress in diverse tasks across text and visual modalities. Unlike traditional tools such as calculators, which give deterministic outputs, neural tools perform uncertainly across task scenarios. While different tools for a task may excel in varied scenarios, existing agents typically rely on fixed tools, thus limiting the flexibility in selecting the most suitable tool for specific tasks. In contrast, humans snowball their understanding of the capabilities of different tools by interacting with them, and apply this knowledge to select the optimal tool when solving a future task. To build agents that similarly benefit from this process, we propose ToolMem that enables agents to develop memories of tool capabilities from previous interactions, by summarizing their strengths and weaknesses and storing them in memory; at inference, the agent can retrieve relevant entries from ToolMem, and select the best tool to solve individual tasks more accurately. We evaluate ToolMem on learning varied text generation and text-to-image generation neural tools. Compared to no-memory, generic agents, we find ToolMem-augmented agents predict tool performance 14.8% and 28.7% more accurately across text and multimodal generation scenarios. Moreover, ToolMem facilitates optimal tool selection among multiple choices by 21% and 24% absolute increases in respective scenarios.

**arXiv ID:** 2510.06664
</details>

<details>
<summary><strong>Bring the Apple, Not the Sofa: Impact of Irrelevant Context in Embodied AI Commands on VLA Models</strong> - Daria Pugacheva, Andrey Moskalenko, Denis Shepelev, Andrey Kuznetsov, Vlad Shakhuro, Elena Tutubalina - [[pdf]](https://arxiv.org/pdf/2510.07067)</summary>

**Abstract:** Vision Language Action (VLA) models are widely used in Embodied AI, enabling robots to interpret and execute language instructions. However, their robustness to natural language variability in real-world scenarios has not been thoroughly investigated. In this work, we present a novel systematic study of the robustness of state-of-the-art VLA models under linguistic perturbations. Specifically, we evaluate model performance under two types of instruction noise: (1) human-generated paraphrasing and (2) the addition of irrelevant context. We further categorize irrelevant contexts into two groups according to their length and their semantic and lexical proximity to robot commands. In this study, we observe consistent performance degradation as context size expands. We also demonstrate that the model can exhibit relative robustness to random context, with a performance drop within 10%, while semantically and lexically similar context of the same length can trigger a quality decline of around 50%. Human paraphrases of instructions lead to a drop of nearly 20%. To mitigate this, we propose an LLM-based filtering framework that extracts core commands from noisy inputs. Incorporating our filtering step allows models to recover up to 98.5% of their original performance under noisy conditions.

**arXiv ID:** 2510.07067
</details>

<details>
<summary><strong>A risk model and analysis method for the psychological safety of human and autonomous vehicles interaction</strong> - Yandika Sirgabsou, Benjamin Hardin, François Leblanc, Efi Raili, Pericle Salvini, David Jackson, Marina Jirotka, Lars Kunze - [[pdf]](https://arxiv.org/pdf/2411.05732)</summary>

**Abstract:** The rapid advancement of artificial intelligence and autonomous driving technologies has significantly propelled the development of autonomous vehicles (AVs). However, psychological barriers continue to impede widespread AV adoption, despite technological progress. This paper addresses the critical yet often overlooked aspect of psychological safety in AV design and operation. While traditional safety standards focus primarily on physical safety, this paper emphasizes the psychological implications that arise from human interactions with autonomous vehicles, highlighting the importance of trust and perceived risk as significant factors influencing user acceptance. The paper makes a methodological proposal, a framework for addressing AVs psychological safety consisting of three key contributions. First, it introduces a definition of psychological safety in AVs context. Secondly, it proposes a risk model for identifying and assessing AVs psychological hazards and risks. PsySIL (Psychological Safety Integrity Level), a classification of AV psychological risk levels is developed. Thirdly, an adapted system-theoretic analysis method for AVs psychological safety is proposed. The paper illustrates the application of the framework for assessing potential psychological hazards using a scenario involving a family's experience with an autonomous vehicle, pioneering a systems approach towards evaluating situations that could lead to psychological harm. By establishing a framework that incorporates psychological safety alongside physical safety, the paper contributes to the broader discourse on the safe deployment of autonomous vehicle, aiming to guide future developments in user-centred design and regulatory practices, while acknowledging the limitations brought by the application of the proposals on a rather simple but pedagogical illustrative example.

**arXiv ID:** 2411.05732
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

<details>
<summary><strong>AlphaApollo: Orchestrating Foundation Models and Professional Tools into a Self-Evolving System for Deep Agentic Reasoning</strong> - Zhanke Zhou, Chentao Cao, Xiao Feng, Xuan Li, Zongze Li, Xiangyu Lu, Jiangchao Yao, Weikai Huang, Linrui Xu, Tian Cheng, Guanyu Jiang, Yiming Zheng, Brando Miranda, Tongliang Liu, Sanmi Koyejo, Masashi Sugiyama, Bo Han - [[pdf]](https://arxiv.org/pdf/2510.06261)</summary>

**Abstract:** We present AlphaApollo, a self-evolving agentic reasoning system that aims to address two bottlenecks in foundation model (FM) reasoning-limited model-intrinsic capacity and unreliable test-time iteration. AlphaApollo orchestrates multiple models with professional tools to enable deliberate, verifiable reasoning. It couples (i) a computation tool (Python with numerical and symbolic libraries) and (ii) a retrieval tool (task-relevant external information) to execute exact calculations and ground decisions. The system further supports multi-round, multi-model solution evolution via a shared state map that records candidates, executable checks, and feedback for iterative refinement. In evaluations on AIME 2024/2025 across multiple models, AlphaApollo delivers consistent gains: +5.15% Average@32 and +23.34% Pass@32 for Qwen2.5-14B-Instruct, and +8.91% Average@32 with +26.67% Pass@32 for Llama-3.3-70B-Instruct. Tool-use analysis shows that more than 80% of tool calls are successfully executed, with consistent outperformance of non-tool baselines, thereby lifting the capability ceiling of FMs. More empirical results and implementation details will be updated at this https URL.

**arXiv ID:** 2510.06261
</details>

<details>
<summary><strong>Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them</strong> - Jiahe Jin, Abhijay Paladugu, Chenyan Xiong - [[pdf]](https://arxiv.org/pdf/2510.06534)</summary>

**Abstract:** Agentic search leverages large language models (LLMs) to interpret complex user information needs and execute a multi-step process of planning, searching, and synthesizing information to provide answers. This paradigm introduces unique challenges for LLMs' reasoning and agentic capabilities when interacting with retrieval systems and the broader web. In this paper, we propose a reasoning-driven LLM-based pipeline to study effective reasoning behavior patterns in agentic search. Using this pipeline, we analyze successful agentic search trajectories and identify four beneficial reasoning behaviors: Information Verification, Authority Evaluation, Adaptive Search, and Error Recovery. Based on these findings, we propose a technique called Behavior Priming to train more effective agentic search models. It synthesizes agentic search trajectories that exhibit these four behaviors and integrates them into the agentic search model through supervised fine-tuning (SFT), followed by standard reinforcement learning (RL). Experiments on three benchmarks (GAIA, WebWalker, and HLE) demonstrate that behavior priming yields over 35% gains in Llama3.2-3B and Qwen3-1.7B compared to directly training agentic search models with RL. Crucially, we demonstrate that the desired reasoning behaviors in the SFT data, rather than the correctness of the final answer, is the critical factor for achieving strong final performance after RL: fine-tuning on trajectories with desirable reasoning behaviors but incorrect answers leads to better performance than fine-tuning on trajectories with correct answers. Our analysis further reveals the underlying mechanism: the introduced reasoning behaviors endow models with more effective exploration (higher pass@k and entropy) and test-time scaling (longer trajectories) capabilities, providing a strong foundation for RL. Our code will be released as open source.

**arXiv ID:** 2510.06534
</details>

<details>
<summary><strong>Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support</strong> - Zhao, Tiantian Zhang, Hanchen Su, Yufeng, Zhang, Shaowei Su, Mingzhi Xu, Wei Han, Jeremy Werner, Claire Na Cheng, Yashar Mehdad - [[pdf]](https://arxiv.org/pdf/2510.06674)</summary>

**Abstract:** We introduce an Agent-in-the-Loop (AITL) framework that implements a continuous data flywheel for iteratively improving an LLM-based customer support system. Unlike standard offline approaches that rely on batch annotations, AITL integrates four key types of annotations directly into live customer operations: (1) pairwise response preferences, (2) agent adoption and rationales, (3) knowledge relevance checks, and (4) identification of missing knowledge. These feedback signals seamlessly feed back into models' updates, reducing retraining cycles from months to weeks. Our production pilot involving US-based customer support agents demonstrated significant improvements in retrieval accuracy (+11.7% recall@75, +14.8% precision@8), generation quality (+8.4% helpfulness) and agent adoption rates (+4.5%). These results underscore the effectiveness of embedding human feedback loops directly into operational workflows to continuously refine LLM-based customer support system.

**arXiv ID:** 2510.06674
</details>

<details>
<summary><strong>Tool-Augmented Policy Optimization: Synergizing Reasoning and Adaptive Tool Use with Reinforcement Learning</strong> - Wenxun Wu, Yuanyang Li, Guhan Chen, Linyue Wang, Hongyang Chen - [[pdf]](https://arxiv.org/pdf/2510.07038)</summary>

**Abstract:** Recent advances in large language models (LLMs) have popularized test-time scaling, where models generate additional reasoning tokens before producing final answers. These approaches have demonstrated significant performance improvements on benchmarks involving mathematical reasoning. However, language models relying solely on direct inference still struggle with tasks demanding up-to-date knowledge or computational tools such as calculators and code interpreters for complex arithmetic operations. To overcome these limitations, we propose Tool-Augmented Policy Optimization (TAPO), a novel reinforcement learning framework that systematically integrates multi-hop reasoning with adaptive tool-calling capabilities. Our approach employs a modified version of Dynamic Sampling Policy Optimization (DAPO), a recently developed RL paradigm, which we adapt specifically for tool invocation scenarios, enabling models to dynamically interleave complex reasoning with on-demand tool usage (including search APIs and Python interpreters).
To support this research, we introduce two new datasets: TAPO-easy-60K and TAPO-hard-18K, specifically designed to train and evaluate both fact-based reasoning and mathematical calculation capabilities. Our experiments on Qwen2.5-3B and Qwen2.5-7B models demonstrate the effectiveness of our approach, with both models achieving state-of-the-art performance on tasks requiring external knowledge and mathematical computation among methods with comparable parameters. Notably, TAPO achieves more efficient tool utilization than baseline methods while preventing excessive calls caused by reward hacking. These results highlight the significant potential of combining advanced reasoning with tool usage to enhance model performance in knowledge-intensive and computationally demanding tasks.

**arXiv ID:** 2510.07038
</details>

<details>
<summary><strong>Local Reinforcement Learning with Action-Conditioned Root Mean Squared Q-Functions</strong> - Frank Wu, Mengye Ren - [[pdf]](https://arxiv.org/pdf/2510.06649)</summary>

**Abstract:** The Forward-Forward (FF) Algorithm is a recently proposed learning procedure for neural networks that employs two forward passes instead of the traditional forward and backward passes used in backpropagation. However, FF remains largely confined to supervised settings, leaving a gap at domains where learning signals can be yielded more naturally such as RL. In this work, inspired by FF's goodness function using layer activity statistics, we introduce Action-conditioned Root mean squared Q-Functions (ARQ), a novel value estimation method that applies a goodness function and action conditioning for local RL using temporal difference learning. Despite its simplicity and biological grounding, our approach achieves superior performance compared to state-of-the-art local backprop-free RL methods in the MinAtar and the DeepMind Control Suite benchmarks, while also outperforming algorithms trained with backpropagation on most tasks. Code can be found at this https URL.

**arXiv ID:** 2510.06649
</details>

<details>
<summary><strong>Incremental Summarization for Customer Support via Progressive Note-Taking and Agent Feedback</strong> - Yisha Wu, Zhao, Yuanpei Cao, Xiaoqing Su, Yashar Mehdad, Mindy Ji, Claire Na Cheng - [[pdf]](https://arxiv.org/pdf/2510.06677)</summary>

**Abstract:** We introduce an incremental summarization system for customer support agents that intelligently determines when to generate concise bullet notes during conversations, reducing agents' context-switching effort and redundant review. Our approach combines a fine-tuned Mixtral-8x7B model for continuous note generation with a DeBERTa-based classifier to filter trivial content. Agent edits refine the online notes generation and regularly inform offline model retraining, closing the agent edits feedback loop. Deployed in production, our system achieved a 3% reduction in case handling time compared to bulk summarization (with reductions of up to 9% in highly complex cases), alongside high agent satisfaction ratings from surveys. These results demonstrate that incremental summarization with continuous feedback effectively enhances summary quality and agent productivity at scale.

**arXiv ID:** 2510.06677
</details>

<details>
<summary><strong>h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning</strong> - Sumeet Ramesh Motwani, Alesia Ivanova, Ziyang Cai, Philip Torr, Riashat Islam, Shital Shah, Christian Schroeder de Witt, Charles London - [[pdf]](https://arxiv.org/pdf/2510.07312)</summary>

**Abstract:** Large language models excel at short-horizon reasoning tasks, but performance drops as reasoning horizon lengths increase. Existing approaches to combat this rely on inference-time scaffolding or costly step-level supervision, neither of which scales easily. In this work, we introduce a scalable method to bootstrap long-horizon reasoning capabilities using only existing, abundant short-horizon data. Our approach synthetically composes simple problems into complex, multi-step dependency chains of arbitrary length. We train models on this data using outcome-only rewards under a curriculum that automatically increases in complexity, allowing RL training to be scaled much further without saturating. Empirically, our method generalizes remarkably well: curriculum training on composed 6th-grade level math problems (GSM8K) boosts accuracy on longer, competition-level benchmarks (GSM-Symbolic, MATH-500, AIME) by up to 2.06x. Importantly, our long-horizon improvements are significantly higher than baselines even at high pass@k, showing that models can learn new reasoning paths under RL. Theoretically, we show that curriculum RL with outcome rewards achieves an exponential improvement in sample complexity over full-horizon training, providing training signal comparable to dense supervision. h1 therefore introduces an efficient path towards scaling RL for long-horizon problems using only existing data.

**arXiv ID:** 2510.07312
</details>

<details>
<summary><strong>KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality</strong> - Baochang Ren, Shuofei Qiao, Da Zheng, Huajun Chen, Ningyu Zhang - [[pdf]](https://arxiv.org/pdf/2506.19807)</summary>

**Abstract:** Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at this https URL.

**arXiv ID:** 2506.19807
</details>

<details>
<summary><strong>A Dual-Agent Adversarial Framework for Robust Generalization in Deep Reinforcement Learning</strong> - Zhengpeng Xie, Yulong Zhang - [[pdf]](https://arxiv.org/pdf/2501.17384)</summary>

**Abstract:** Recently, empowered with the powerful capabilities of neural networks, reinforcement learning (RL) has successfully tackled numerous challenging tasks. However, while these models demonstrate enhanced decision-making abilities, they are increasingly prone to overfitting. For instance, a trained RL model often fails to generalize to even minor variations of the same task, such as a change in background color or other minor semantic differences. To address this issue, we propose a dual-agent adversarial policy learning framework, which allows agents to spontaneously learn the underlying semantics without introducing any human prior knowledge. Specifically, our framework involves a game process between two agents: each agent seeks to maximize the impact of perturbing on the opponent's policy by producing representation differences for the same state, while maintaining its own stability against such perturbations. This interaction encourages agents to learn generalizable policies, capable of handling irrelevant features from the high-dimensional observations. Extensive experimental results on the Procgen benchmark demonstrate that the adversarial process significantly improves the generalization performance of both agents, while also being applied to various RL algorithms, e.g., Proximal Policy Optimization (PPO). With the adversarial framework, the RL agent outperforms the baseline methods by a significant margin, especially in hard-level tasks, marking a significant step forward in the generalization capabilities of deep reinforcement learning.

**arXiv ID:** 2501.17384
</details>

<details>
<summary><strong>Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning</strong> - Xue Zhang, Yunlong Liang, Fandong Meng, Songming Zhang, Kaiyu Huang, Yufeng Chen, Jinan Xu, Jie Zhou - [[pdf]](https://arxiv.org/pdf/2510.07300)</summary>

**Abstract:** Large Reasoning Models (LRMs) have achieved remarkable performance on complex reasoning tasks by adopting the "think-then-answer" paradigm, which enhances both accuracy and interpretability. However, current LRMs exhibit two critical limitations when processing non-English languages: (1) They often struggle to maintain input-output language consistency; (2) They generally perform poorly with wrong reasoning paths and lower answer accuracy compared to English. These limitations significantly degrade the user experience for non-English speakers and hinder the global deployment of LRMs. To address these limitations, we propose M-Thinker, which is trained by the GRPO algorithm that involves a Language Consistency (LC) reward and a novel Cross-lingual Thinking Alignment (CTA) reward. Specifically, the LC reward defines a strict constraint on the language consistency between the input, thought, and answer. Besides, the CTA reward compares the model's non-English reasoning paths with its English reasoning path to transfer its own reasoning capability from English to non-English languages. Through an iterative RL procedure, our M-Thinker-1.5B/7B models not only achieve nearly 100% language consistency and superior performance on two multilingual benchmarks (MMATH and PolyMath), but also exhibit excellent generalization on out-of-domain languages.

**arXiv ID:** 2510.07300
</details>

<details>
<summary><strong>Test-Time Graph Search for Goal-Conditioned Reinforcement Learning</strong> - Evgenii Opryshko, Junwei Quan, Claas Voelcker, Yilun Du, Igor Gilitschenski - [[pdf]](https://arxiv.org/pdf/2510.07257)</summary>

**Abstract:** Offline goal-conditioned reinforcement learning (GCRL) trains policies that reach user-specified goals at test time, providing a simple, unsupervised, domain-agnostic way to extract diverse behaviors from unlabeled, reward-free datasets. Nonetheless, long-horizon decision making remains difficult for GCRL agents due to temporal credit assignment and error accumulation, and the offline setting amplifies these effects. To alleviate this issue, we introduce Test-Time Graph Search (TTGS), a lightweight planning approach to solve the GCRL task. TTGS accepts any state-space distance or cost signal, builds a weighted graph over dataset states, and performs fast search to assemble a sequence of subgoals that a frozen policy executes. When the base learner is value-based, the distance is derived directly from the learned goal-conditioned value function, so no handcrafted metric is needed. TTGS requires no changes to training, no additional supervision, no online interaction, and no privileged information, and it runs entirely at inference. On the OGBench benchmark, TTGS improves success rates of multiple base learners on challenging locomotion tasks, demonstrating the benefit of simple metric-guided test-time planning for offline GCRL.

**arXiv ID:** 2510.07257
</details>

<details>
<summary><strong>General and Efficient Visual Goal-Conditioned Reinforcement Learning using Object-Agnostic Masks</strong> - Fahim Shahriar, Cheryl Wang, Alireza Azimi, Gautham Vasan, Hany Hamed Elanwar, A. Rupam Mahmood, Colin Bellinger - [[pdf]](https://arxiv.org/pdf/2510.06277)</summary>

**Abstract:** Goal-conditioned reinforcement learning (GCRL) allows agents to learn diverse objectives using a unified policy. The success of GCRL, however, is contingent on the choice of goal representation. In this work, we propose a mask-based goal representation system that provides object-agnostic visual cues to the agent, enabling efficient learning and superior generalization. In contrast, existing goal representation methods, such as target state images, 3D coordinates, and one-hot vectors, face issues of poor generalization to unseen objects, slow convergence, and the need for special cameras. Masks can be processed to generate dense rewards without requiring error-prone distance calculations. Learning with ground truth masks in simulation, we achieved 99.9% reaching accuracy on training and unseen test objects. Our proposed method can be utilized to perform pick-up tasks with high accuracy, without using any positional information of the target. Moreover, we demonstrate learning from scratch and sim-to-real transfer applications using two different physical robots, utilizing pretrained open vocabulary object detection models for mask generation.

**arXiv ID:** 2510.06277
</details>

<details>
<summary><strong>Online Matching via Reinforcement Learning: An Expert Policy Orchestration Strategy</strong> - Chiara Mignacco, Matthieu Jonckheere, Gilles Stoltz - [[pdf]](https://arxiv.org/pdf/2510.06515)</summary>

**Abstract:** Online matching problems arise in many complex systems, from cloud services and online marketplaces to organ exchange networks, where timely, principled decisions are critical for maintaining high system performance. Traditional heuristics in these settings are simple and interpretable but typically tailored to specific operating regimes, which can lead to inefficiencies when conditions change. We propose a reinforcement learning (RL) approach that learns to orchestrate a set of such expert policies, leveraging their complementary strengths in a data-driven, adaptive manner. Building on the Adv2 framework (Jonckheere et al., 2024), our method combines expert decisions through advantage-based weight updates and extends naturally to settings where only estimated value functions are available. We establish both expectation and high-probability regret guarantees and derive a novel finite-time bias bound for temporal-difference learning, enabling reliable advantage estimation even under constant step size and non-stationary dynamics. To support scalability, we introduce a neural actor-critic architecture that generalizes across large state spaces while preserving interpretability. Simulations on stochastic matching models, including an organ exchange scenario, show that the orchestrated policy converges faster and yields higher system level efficiency than both individual experts and conventional RL baselines. Our results highlight how structured, adaptive learning can improve the modeling and management of complex resource allocation and decision-making processes.

**arXiv ID:** 2510.06515
</details>

<details>
<summary><strong>PyCFRL: A Python library for counterfactually fair offline reinforcement learning via sequential data preprocessing</strong> - Jianhan Zhang, Jitao Wang, Chengchun Shi, John D. Piette, Donglin Zeng, Zhenke Wu - [[pdf]](https://arxiv.org/pdf/2510.06935)</summary>

**Abstract:** Reinforcement learning (RL) aims to learn and evaluate a sequential decision rule, often referred to as a "policy", that maximizes the population-level benefit in an environment across possibly infinitely many time steps. However, the sequential decisions made by an RL algorithm, while optimized to maximize overall population benefits, may disadvantage certain individuals who are in minority or socioeconomically disadvantaged groups. To address this problem, we introduce PyCFRL, a Python library for ensuring counterfactual fairness in offline RL. PyCFRL implements a novel data preprocessing algorithm for learning counterfactually fair RL policies from offline datasets and provides tools to evaluate the values and counterfactual unfairness levels of RL policies. We describe the high-level functionalities of PyCFRL and demonstrate one of its major use cases through a data example. The library is publicly available on PyPI and Github (this https URL), and detailed tutorials can be found in the PyCFRL documentation (this https URL).

**arXiv ID:** 2510.06935
</details>

<details>
<summary><strong>Falsification-Driven Reinforcement Learning for Maritime Motion Planning</strong> - Marlon Müller, Florian Finkeldei, Hanna Krasowski, Murat Arcak, Matthias Althoff - [[pdf]](https://arxiv.org/pdf/2510.06970)</summary>

**Abstract:** Compliance with maritime traffic rules is essential for the safe operation of autonomous vessels, yet training reinforcement learning (RL) agents to adhere to them is challenging. The behavior of RL agents is shaped by the training scenarios they encounter, but creating scenarios that capture the complexity of maritime navigation is non-trivial, and real-world data alone is insufficient. To address this, we propose a falsification-driven RL approach that generates adversarial training scenarios in which the vessel under test violates maritime traffic rules, which are expressed as signal temporal logic specifications. Our experiments on open-sea navigation with two vessels demonstrate that the proposed approach provides more relevant training scenarios and achieves more consistent rule compliance.

**arXiv ID:** 2510.06970
</details>

<details>
<summary><strong>Diffusion-Augmented Reinforcement Learning for Robust Portfolio Optimization under Stress Scenarios</strong> - Himanshu Choudhary, Arishi Orra, Manoj Thakur - [[pdf]](https://arxiv.org/pdf/2510.07099)</summary>

**Abstract:** In the ever-changing and intricate landscape of financial markets, portfolio optimisation remains a formidable challenge for investors and asset managers. Conventional methods often struggle to capture the complex dynamics of market behaviour and align with diverse investor preferences. To address this, we propose an innovative framework, termed Diffusion-Augmented Reinforcement Learning (DARL), which synergistically integrates Denoising Diffusion Probabilistic Models (DDPMs) with Deep Reinforcement Learning (DRL) for portfolio management. By leveraging DDPMs to generate synthetic market crash scenarios conditioned on varying stress intensities, our approach significantly enhances the robustness of training data. Empirical evaluations demonstrate that DARL outperforms traditional baselines, delivering superior risk-adjusted returns and resilience against unforeseen crises, such as the 2025 Tariff Crisis. This work offers a robust and practical methodology to bolster stress resilience in DRL-driven financial applications.

**arXiv ID:** 2510.07099
</details>

<details>
<summary><strong>Dynamic Learning Rate for Deep Reinforcement Learning: A Bandit Approach</strong> - Henrique Donâncio, Antoine Barrier, Leah F. South, Florence Forbes - [[pdf]](https://arxiv.org/pdf/2410.12598)</summary>

**Abstract:** In deep Reinforcement Learning (RL), the learning rate critically influences both stability and performance, yet its optimal value shifts during training as the environment and policy evolve. Standard decay schedulers assume monotonic convergence and often misalign with these dynamics, leading to premature or delayed adjustments. We introduce LRRL, a meta-learning approach that dynamically selects the learning rate based on policy performance rather than training steps. LRRL adaptively favors rates that improve returns, remaining robust even when the candidate set includes values that individually cause divergence. Across Atari and MuJoCo benchmarks, LRRL achieves performance competitive with or superior to tuned baselines and standard schedulers. Our findings position LRRL as a practical solution for adapting to non-stationary objectives in deep RL.

**arXiv ID:** 2410.12598
</details>

<details>
<summary><strong>Reinforcement Learning for Dynamic Memory Allocation</strong> - Arisrei Lim, Abhiram Maddukuri - [[pdf]](https://arxiv.org/pdf/2410.15492)</summary>

**Abstract:** In recent years, reinforcement learning (RL) has gained popularity and has been applied to a wide range of tasks. One such popular domain where RL has been effective is resource management problems in systems. We look to extend work on RL for resource management problems by considering the novel domain of dynamic memory allocation management. We consider dynamic memory allocation to be a suitable domain for RL since current algorithms like first-fit, best-fit, and worst-fit can fail to adapt to changing conditions and can lead to fragmentation and suboptimal efficiency. In this paper, we present a framework in which an RL agent continuously learns from interactions with the system to improve memory management tactics. We evaluate our approach through various experiments using high-level and low-level action spaces and examine different memory allocation patterns. Our results show that RL can successfully train agents that can match and surpass traditional allocation strategies, particularly in environments characterized by adversarial request patterns. We also explore the potential of history-aware policies that leverage previous allocation requests to enhance the allocator's ability to handle complex request patterns. Overall, we find that RL offers a promising avenue for developing more adaptive and efficient memory allocation strategies, potentially overcoming limitations of hardcoded allocation algorithms.

**arXiv ID:** 2410.15492
</details>

<details>
<summary><strong>Nonparametric Bellman Mappings for Value Iteration in Distributed Reinforcement Learning</strong> - Yuki Akiyama, Konstantinos Slavakis - [[pdf]](https://arxiv.org/pdf/2503.16192)</summary>

**Abstract:** This paper introduces novel Bellman mappings (B-Maps) for value iteration (VI) in distributed reinforcement learning (DRL), where agents are deployed over an undirected, connected graph/network with arbitrary topology -- but without a centralized node, that is, a node capable of aggregating all data and performing computations. Each agent constructs a nonparametric B-Map from its private data, operating on Q-functions represented in a reproducing kernel Hilbert space, with flexibility in choosing the basis for their representation. Agents exchange their Q-function estimates only with direct neighbors, and unlike existing DRL approaches that restrict communication to Q-functions, the proposed framework also enables the transmission of basis information in the form of covariance matrices, thereby conveying additional structural details. Linear convergence rates are established for both Q-function and covariance-matrix estimates toward their consensus values, regardless of the network topology, with optimal learning rates determined by the ratio of the smallest positive eigenvalue (the graph's Fiedler value) to the largest eigenvalue of the graph Laplacian matrix. A detailed performance analysis further shows that the proposed DRL framework effectively approximates the performance of a centralized node, had such a node existed. Numerical tests on two benchmark control problems confirm the effectiveness of the proposed nonparametric B-Maps relative to prior methods. Notably, the tests reveal a counter-intuitive outcome: although the framework involves richer information exchange -- specifically through transmitting covariance matrices as basis information -- it achieves the desired performance at a lower cumulative communication cost than existing DRL schemes, underscoring the critical role of sharing basis information in accelerating the learning process.

**arXiv ID:** 2503.16192
</details>

<details>
<summary><strong>Safe Obstacle-Free Guidance of Space Manipulators in Debris Removal Missions via Deep Reinforcement Learning</strong> - Vincent Lam, Robin Chhabra - [[pdf]](https://arxiv.org/pdf/2510.06566)</summary>

**Abstract:** The objective of this study is to develop a model-free workspace trajectory planner for space manipulators using a Twin Delayed Deep Deterministic Policy Gradient (TD3) agent to enable safe and reliable debris capture. A local control strategy with singularity avoidance and manipulability enhancement is employed to ensure stable execution. The manipulator must simultaneously track a capture point on a non-cooperative target, avoid self-collisions, and prevent unintended contact with the target. To address these challenges, we propose a curriculum-based multi-critic network where one critic emphasizes accurate tracking and the other enforces collision avoidance. A prioritized experience replay buffer is also used to accelerate convergence and improve policy robustness. The framework is evaluated on a simulated seven-degree-of-freedom KUKA LBR iiwa mounted on a free-floating base in Matlab/Simulink, demonstrating safe and adaptive trajectory generation for debris removal missions.

**arXiv ID:** 2510.06566
</details>

<details>
<summary><strong>Prototyping Multimodal GenAI Real-Time Agents with Counterfactual Replays and Hybrid Wizard-of-Oz</strong> - Frederic Gmeiner, Kenneth Holstein, Nikolas Martelaro - [[pdf]](https://arxiv.org/pdf/2510.06872)</summary>

**Abstract:** Recent advancements in multimodal generative AI (GenAI) enable the creation of personal context-aware real-time agents that, for example, can augment user workflows by following their on-screen activities and providing contextual assistance. However, prototyping such experiences is challenging, especially when supporting people with domain-specific tasks using real-time inputs such as speech and screen recordings. While prototyping an LLM-based proactive support agent system, we found that existing prototyping and evaluation methods were insufficient to anticipate the nuanced situational complexity and contextual immediacy required. To overcome these challenges, we explored a novel user-centered prototyping approach that combines counterfactual video replay prompting and hybrid Wizard-of-Oz methods to iteratively design and refine agent behaviors. This paper discusses our prototyping experiences, highlighting successes and limitations, and offers a practical guide and an open-source toolkit for UX designers, HCI researchers, and AI toolmakers to build more user-centered and context-aware multimodal agents.

**arXiv ID:** 2510.06872
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
