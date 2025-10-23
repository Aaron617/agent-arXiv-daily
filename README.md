# Agent arXiv Daily

**Last Updated:** 2025-10-23 02:42:27

**Total Papers:** 67

## Table of Contents

- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>WebGraphEval: Multi-Turn Trajectory Evaluation for Web Agents using Graph Representation</strong> - Yaoyao Qian, Yuanli Wang, Jinda Zhang, Yun Zong, Meixu Chen, Hanhan Zhou, Jindan Huang, Yifan Zeng, Xinyu Hu, Chan Hee Song, Danqing Zhang - [[pdf]](https://arxiv.org/pdf/2510.19205)</summary>

**Abstract:** Current evaluation of web agents largely reduces to binary success metrics or conformity to a single reference trajectory, ignoring the structural diversity present in benchmark datasets. We present WebGraphEval, a framework that abstracts trajectories from multiple agents into a unified, weighted action graph. This representation is directly compatible with benchmarks such as WebArena, leveraging leaderboard runs and newly collected trajectories without modifying environments. The framework canonically encodes actions, merges recurring behaviors, and applies structural analyses including reward propagation and success-weighted edge statistics. Evaluations across thousands of trajectories from six web agents show that the graph abstraction captures cross-model regularities, highlights redundancy and inefficiency, and identifies critical decision points overlooked by outcome-based metrics. By framing web interaction as graph-structured data, WebGraphEval establishes a general methodology for multi-path, cross-agent, and efficiency-aware evaluation of web agents.

**arXiv ID:** 2510.19205
</details>

<details>
<summary><strong>HSCodeComp: A Realistic and Expert-level Benchmark for Deep Search Agents in Hierarchical Rule Application</strong> - Yiqian Yang, Tian Lan, Qianghuai Jia, Li Zhu, Hui Jiang, Hang Zhu, Longyue Wang, Weihua Luo, Kaifu Zhang - [[pdf]](https://arxiv.org/pdf/2510.19631)</summary>

**Abstract:** Effective deep search agents must not only access open-domain and domain-specific knowledge but also apply complex rules-such as legal clauses, medical manuals and tariff rules. These rules often feature vague boundaries and implicit logic relationships, making precise application challenging for agents. However, this critical capability is largely overlooked by current agent benchmarks.
To fill this gap, we introduce HSCodeComp, the first realistic, expert-level e-commerce benchmark designed to evaluate deep search agents in hierarchical rule application. In this task, the deep reasoning process of agents is guided by these rules to predict 10-digit Harmonized System Code (HSCode) of products with noisy but realistic descriptions. These codes, established by the World Customs Organization, are vital for global supply chain efficiency. Built from real-world data collected from large-scale e-commerce platforms, our proposed HSCodeComp comprises 632 product entries spanning diverse product categories, with these HSCodes annotated by several human experts.
Extensive experimental results on several state-of-the-art LLMs, open-source, and closed-source agents reveal a huge performance gap: best agent achieves only 46.8% 10-digit accuracy, far below human experts at 95.0%. Besides, detailed analysis demonstrates the challenges of hierarchical rule application, and test-time scaling fails to improve performance further.

**arXiv ID:** 2510.19631
</details>

<details>
<summary><strong>Misalignment Bounty: Crowdsourcing AI Agent Misbehavior</strong> - Rustem Turtayev, Natalia Fedorova, Oleg Serikov, Sergey Koldyba, Lev Avagyan, Dmitrii Volkov - [[pdf]](https://arxiv.org/pdf/2510.19738)</summary>

**Abstract:** Advanced AI systems sometimes act in ways that differ from human intent. To gather clear, reproducible examples, we ran the Misalignment Bounty: a crowdsourced project that collected cases of agents pursuing unintended or unsafe goals. The bounty received 295 submissions, of which nine were awarded.
This report explains the program's motivation and evaluation criteria, and walks through the nine winning submissions step by step.

**arXiv ID:** 2510.19738
</details>

<details>
<summary><strong>See, Think, Act: Online Shopper Behavior Simulation with VLM Agents</strong> - Yimeng Zhang, Jiri Gesi, Ran Xue, Tian Wang, Ziyi Wang, Yuxuan Lu, Sinong Zhan, Huimin Zeng, Qingjun Cui, Yufan Guo, Jing Huang, Mubarak Shah, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2510.19245)</summary>

**Abstract:** LLMs have recently demonstrated strong potential in simulating online shopper behavior. Prior work has improved action prediction by applying SFT on action traces with LLM-generated rationales, and by leveraging RL to further enhance reasoning capabilities. Despite these advances, current approaches rely on text-based inputs and overlook the essential role of visual perception in shaping human decision-making during web GUI interactions. In this paper, we investigate the integration of visual information, specifically webpage screenshots, into behavior simulation via VLMs, leveraging OPeRA dataset. By grounding agent decision-making in both textual and visual modalities, we aim to narrow the gap between synthetic agents and real-world users, thereby enabling more cognitively aligned simulations of online shopping behavior. Specifically, we employ SFT for joint action prediction and rationale generation, conditioning on the full interaction context, which comprises action history, past HTML observations, and the current webpage screenshot. To further enhance reasoning capabilities, we integrate RL with a hierarchical reward structure, scaled by a difficulty-aware factor that prioritizes challenging decision points. Empirically, our studies show that incorporating visual grounding yields substantial gains: the combination of text and image inputs improves exact match accuracy by more than 6% over text-only inputs. These results indicate that multi-modal grounding not only boosts predictive accuracy but also enhances simulation fidelity in visually complex environments, which captures nuances of human attention and decision-making that text-only agents often miss. Finally, we revisit the design space of behavior simulation frameworks, identify key methodological limitations, and propose future research directions toward building efficient and effective human behavior simulators.

**arXiv ID:** 2510.19245
</details>

<details>
<summary><strong>Can Agents Fix Agent Issues?</strong> - Alfin Wijaya Rahardja, Junwei Liu, Weitong Chen, Zhenpeng Chen, Yiling Lou - [[pdf]](https://arxiv.org/pdf/2505.20749)</summary>

**Abstract:** LLM-based agent systems are emerging as a new software paradigm and have been widely adopted across diverse domains such as medicine, robotics, and programming. However, maintaining these systems requires substantial effort, as they are inevitably prone to bugs and continually evolve to meet changing external requirements. Therefore, automatically resolving agent issues (i.e., bug reports or feature requests) is a crucial and challenging task. While recent software engineering (SE) agents (e.g., SWE-agent) have shown promise in addressing issues in traditional software systems, it remains unclear how effectively they can resolve real-world issues in agent systems, which differ significantly from traditional software. To fill this gap, we first manually analyze 201 real-world agent issues and identify common categories of agent issues. We then spend 500 person-hours constructing AGENTISSUE-BENCH, a reproducible benchmark comprising 50 agent issue resolution tasks (each with an executable environment and failure-triggering tests). We further evaluate state-of-the-art SE agents on AGENTISSUE-BENCH and reveal their limited effectiveness (i.e., with only 3.33% - 12.67% resolution rates). These results underscore the unique challenges of maintaining agent systems compared to traditional software, highlighting the need for further research to develop advanced SE agents for resolving agent issues. Data and code are available at this https URL .

**arXiv ID:** 2505.20749
</details>

<details>
<summary><strong>GeoBenchX: Benchmarking LLMs in Agent Solving Multistep Geospatial Tasks</strong> - Varvara Krechetova, Denis Kochedykov - [[pdf]](https://arxiv.org/pdf/2503.18129)</summary>

**Abstract:** This paper establishes a benchmark for evaluating tool-calling capabilities of large language models (LLMs) on multi-step geospatial tasks relevant to commercial GIS practitioners. We assess eight commercial LLMs (Claude Sonnet 3.5 and 4, Claude Haiku 3.5, Gemini 2.0 Flash, Gemini 2.5 Pro Preview, GPT-4o, GPT-4.1 and o4-mini) using a simple tool-calling agent equipped with 23 geospatial functions. Our benchmark comprises tasks in four categories of increasing complexity, with both solvable and intentionally unsolvable tasks to test rejection accuracy. We develop a LLM-as-Judge evaluation framework to compare agent solutions against reference solutions. Results show o4-mini and Claude 3.5 Sonnet achieve the best overall performance, OpenAI's GPT-4.1, GPT-4o and Google's Gemini 2.5 Pro Preview do not fall far behind, but the last two are more efficient in identifying unsolvable tasks. Claude Sonnet 4, due its preference to provide any solution rather than reject a task, proved to be less accurate. We observe significant differences in token usage, with Anthropic models consuming more tokens than competitors. Common errors include misunderstanding geometrical relationships, relying on outdated knowledge, and inefficient data manipulation. The resulting benchmark set, evaluation framework, and data generation pipeline are released as open-source resources (available at this https URL), providing one more standardized method for the ongoing evaluation of LLMs for GeoAI.

**arXiv ID:** 2503.18129
</details>

<details>
<summary><strong>ACT: Agentic Classification Tree</strong> - Vincent Grari, Tim Arni, Thibault Laugel, Sylvain Lamprier, James Zou, Marcin Detyniecki - [[pdf]](https://arxiv.org/pdf/2509.26433)</summary>

**Abstract:** When used in high-stakes settings, AI systems are expected to produce decisions that are transparent, interpretable, and auditable, a requirement increasingly expected by regulations. Decision trees such as CART provide clear and verifiable rules, but they are restricted to structured tabular data and cannot operate directly on unstructured inputs such as text. In practice, large language models (LLMs) are widely used for such data, yet prompting strategies such as chain-of-thought or prompt optimization still rely on free-form reasoning, limiting their ability to ensure trustworthy behaviors. We present the Agentic Classification Tree (ACT), which extends decision-tree methodology to unstructured inputs by formulating each split as a natural-language question, refined through impurity-based evaluation and LLM feedback via TextGrad. Experiments on text benchmarks show that ACT matches or surpasses prompting-based baselines while producing transparent and interpretable decision paths.

**arXiv ID:** 2509.26433
</details>

<details>
<summary><strong>Multi-Faceted Evaluation of Tool-Augmented Dialogue Systems</strong> - Zhaoyi Joey Hou, Tanya Shourya, Yingfan Wang, Shamik Roy, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah - [[pdf]](https://arxiv.org/pdf/2510.19186)</summary>

**Abstract:** Evaluating conversational AI systems that use external tools is challenging, as errors can arise from complex interactions among user, agent, and tools. While existing evaluation methods assess either user satisfaction or agents' tool-calling capabilities, they fail to capture critical errors in multi-turn tool-augmented dialogues-such as when agents misinterpret tool results yet appear satisfactory to users. We introduce TRACE, a benchmark of systematically synthesized tool-augmented conversations covering diverse error cases, and SCOPE, an evaluation framework that automatically discovers diverse error patterns and evaluation rubrics in tool-augmented dialogues. Experiments show SCOPE significantly outperforms the baseline, particularly on challenging cases where user satisfaction signals are misleading.

**arXiv ID:** 2510.19186
</details>

<details>
<summary><strong>TheMCPCompany: Creating General-purpose Agents with Task-specific Tools</strong> - Reza Esfandiarpoor, Vishwas Suryanarayanan, Stephen H. Bach, Vishal Chowdhary, Anthony Aue - [[pdf]](https://arxiv.org/pdf/2510.19286)</summary>

**Abstract:** Since the introduction of the Model Context Protocol (MCP), the number of available tools for Large Language Models (LLMs) has increased significantly. These task-specific tool sets offer an alternative to general-purpose tools such as web browsers, while being easier to develop and maintain than GUIs. However, current general-purpose agents predominantly rely on web browsers for interacting with the environment. Here, we introduce TheMCPCompany, a benchmark for evaluating tool-calling agents on tasks that involve interacting with various real-world services. We use the REST APIs of these services to create MCP servers, which include over 18,000 tools. We also provide manually annotated ground-truth tools for each task. In our experiments, we use the ground truth tools to show the potential of tool-calling agents for both improving performance and reducing costs assuming perfect tool retrieval. Next, we explore agent performance using tool retrieval to study the real-world practicality of tool-based agents. While all models with tool retrieval perform similarly or better than browser-based agents, smaller models cannot take full advantage of the available tools through retrieval. On the other hand, GPT-5's performance with tool retrieval is very close to its performance with ground-truth tools. Overall, our work shows that the most advanced reasoning models are effective at discovering tools in simpler environments, but seriously struggle with navigating complex enterprise environments. TheMCPCompany reveals that navigating tens of thousands of tools and combining them in non-trivial ways to solve complex problems is still a challenging task for current models and requires both better reasoning and better retrieval models.

**arXiv ID:** 2510.19286
</details>

<details>
<summary><strong>From Answers to Guidance: A Proactive Dialogue System for Legal Documents</strong> - Ashish Chouhan, Michael Gertz - [[pdf]](https://arxiv.org/pdf/2510.19723)</summary>

**Abstract:** The accessibility of legal information remains a constant challenge, particularly for laypersons seeking to understand and apply complex institutional texts. While the European Union provides open access to legislation, parliamentary responses, and regulatory documents, these resources can be challenging for laypeople to explore. In this paper, we introduce EUDial, a proactive multi-turn dialogue dataset constructed from 204 blogs curated by the Citizens' Enquiries Unit (AskEP) of the European Parliamentary Research Service. EUDial contains 880 dialogue turns (averaging 4.3 turns per dialogue), where each dialogue includes initial questions, structured answers, and follow-up questions. Beyond dataset construction, we propose the LexGuide framework that leverages retrieval-augmented generation with hierarchical topic organization to structure dialogue progression, ensuring both comprehensive coverage of legal aspects and coherence across conversational turns. The results demonstrate that proactive, structured navigation closes the gap between the availability of legal information and citizen comprehension, establishing EUDial and LexGuide as practical resources for advancing proactive legal dialogue systems.

**arXiv ID:** 2510.19723
</details>

</details>

<details open>
<summary><h2>LLM Agents (3 papers)</h2></summary>

<details>
<summary><strong>Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents</strong> - Gil Pasternak, Dheeraj Rajagopal, Julia White, Dhruv Atreja, Matthew Thomas, George Hurn-Maloney, Ash Lewis - [[pdf]](https://arxiv.org/pdf/2510.19771)</summary>

**Abstract:** LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. To address this gap, we present PROBE (Proactive Resolution Of BottlEnecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and expose promising future research directions.

**arXiv ID:** 2510.19771
</details>

<details>
<summary><strong>Measuring Data Science Automation: A Survey of Evaluation Tools for AI Assistants and Agents</strong> - Irene Testini, José Hernández-Orallo, Lorenzo Pacchiardi - [[pdf]](https://arxiv.org/pdf/2506.08800)</summary>

**Abstract:** Data science aims to extract insights from data to support decision-making processes. Recently, Large Language Models (LLMs) have been increasingly used as assistants for data science, by suggesting ideas, techniques and small code snippets, or for the interpretation of results and reporting. Proper automation of some data-science activities is now promised by the rise of LLM agents, i.e., AI systems powered by an LLM equipped with additional affordances--such as code execution and knowledge bases--that can perform self-directed actions and interact with digital environments. In this paper, we survey the evaluation of LLM assistants and agents for data science. We find (1) a dominant focus on a small subset of goal-oriented activities, largely ignoring data management and exploratory activities; (2) a concentration on pure assistance or fully autonomous agents, without considering intermediate levels of human-AI collaboration; and (3) an emphasis on human substitution, therefore neglecting the possibility of higher levels of automation thanks to task transformation.

**arXiv ID:** 2506.08800
</details>

<details>
<summary><strong>AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks</strong> - Fali Wang, Hui Liu, Zhenwei Dai, Jingying Zeng, Zhiwei Zhang, Zongyu Wu, Chen Luo, Zhen Li, Xianfeng Tang, Qi He, Suhang Wang - [[pdf]](https://arxiv.org/pdf/2508.00890)</summary>

**Abstract:** Test-time scaling (TTS) enhances the performance of large language models (LLMs) by allocating additional compute resources during inference. However, existing research primarily investigates TTS in single-stage tasks; while many real-world problems are multi-stage complex tasks, composed of a sequence of heterogeneous subtasks with each subtask requires LLM of specific capability. Therefore, we study a novel problem: the test-time compute-optimal scaling in multi-stage complex tasks, aiming to select suitable models and allocate budgets per subtask to maximize overall performance. TTS in multi-stage tasks introduces two fundamental challenges: (i) The combinatorial search space of model and budget allocations, combined with the high cost of inference, makes brute-force search impractical. (ii) The optimal model and budget allocations across subtasks are interdependent, increasing the complexity of the compute-optimal search. To address this gap, we conduct extensive pilot experiments on four tasks across six datasets, deriving three empirical insights characterizing the behavior of LLMs in multi-stage complex tasks. Informed by these insights, we propose AgentTTS, an LLM-agent-based framework that autonomously searches for compute-optimal allocations through iterative feedback-driven interactions with the execution environment. Experimental results demonstrate that AgentTTS significantly outperforms traditional and other LLM-based baselines in search efficiency, and shows improved robustness to varying training set sizes and enhanced interpretability.

**arXiv ID:** 2508.00890
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (20 papers)</h2></summary>

<details>
<summary><strong>Learning to Make Friends: Coaching LLM Agents toward Emergent Social Ties</strong> - Philipp J. Schneider, Lin Tian, Marian-Andrei Rizoiu - [[pdf]](https://arxiv.org/pdf/2510.19299)</summary>

**Abstract:** Can large language model (LLM) agents reproduce the complex social dynamics that characterize human online behavior -- shaped by homophily, reciprocity, and social validation -- and what memory and learning mechanisms enable such dynamics to emerge? We present a multi-agent LLM simulation framework in which agents repeatedly interact, evaluate one another, and adapt their behavior through in-context learning accelerated by a coaching signal. To model human social behavior, we design behavioral reward functions that capture core drivers of online engagement, including social interaction, information seeking, self-presentation, coordination, and emotional support. These rewards align agent objectives with empirically observed user motivations, enabling the study of how network structures and group formations emerge from individual decision-making. Our experiments show that coached LLM agents develop stable interaction patterns and form emergent social ties, yielding network structures that mirror properties of real online communities. By combining behavioral rewards with in-context adaptation, our framework establishes a principled testbed for investigating collective dynamics in LLM populations and reveals how artificial agents may approximate or diverge from human-like social behavior.

**arXiv ID:** 2510.19299
</details>

<details>
<summary><strong>AgentSense: LLMs Empower Generalizable and Explainable Web-Based Participatory Urban Sensing</strong> - Xusen Guo, Mingxing Peng, Xixuan Hao, Xingchen Zou, Qiongyan Wang, Sijie Ruan, Yuxuan Liang - [[pdf]](https://arxiv.org/pdf/2510.19661)</summary>

**Abstract:** Web-based participatory urban sensing has emerged as a vital approach for modern urban management by leveraging mobile individuals as distributed sensors. However, existing urban sensing systems struggle with limited generalization across diverse urban scenarios and poor interpretability in decision-making. In this work, we introduce AgentSense, a hybrid, training-free framework that integrates large language models (LLMs) into participatory urban sensing through a multi-agent evolution system. AgentSense initially employs classical planner to generate baseline solutions and then iteratively refines them to adapt sensing task assignments to dynamic urban conditions and heterogeneous worker preferences, while producing natural language explanations that enhance transparency and trust. Extensive experiments across two large-scale mobility datasets and seven types of dynamic disturbances demonstrate that AgentSense offers distinct advantages in adaptivity and explainability over traditional methods. Furthermore, compared to single-agent LLM baselines, our approach outperforms in both performance and robustness, while delivering more reasonable and transparent explanations. These results position AgentSense as a significant advancement towards deploying adaptive and explainable urban sensing systems on the web.

**arXiv ID:** 2510.19661
</details>

<details>
<summary><strong>CodeCRDT: Observation-Driven Coordination for Multi-Agent LLM Code Generation</strong> - Sergey Pugachev - [[pdf]](https://arxiv.org/pdf/2510.18893)</summary>

**Abstract:** Multi-agent LLM systems fail to realize parallel speedups due to costly coordination. We present CodeCRDT, an observation-driven coordination pattern where agents coordinate by monitoring a shared state with observable updates and deterministic convergence, rather than explicit message passing. Using Conflict-Free Replicated Data Types (CRDTs), CodeCRDT enables lock-free, conflict-free concurrent code generation with strong eventual consistency. Evaluation across 600 trials (6 tasks, 50 runs per mode) shows both benefits and trade-offs: up to 21.1% speedup on some tasks, up to 39.4% slowdown on others, and 100% convergence with zero merge failures. The study formalizes observation-driven coordination for stochastic LLM agents, revealing semantic conflict rates (5-10%) and quality-performance tradeoffs, and provides empirical characterization of when parallel coordination succeeds versus fails based on task structure.

**arXiv ID:** 2510.18893
</details>

<details>
<summary><strong>Plural Voices, Single Agent: Towards Inclusive AI in Multi-User Domestic Spaces</strong> - Joydeep Chandra, Satyam Kumar Navneet - [[pdf]](https://arxiv.org/pdf/2510.19008)</summary>

**Abstract:** Domestic AI agents faces ethical, autonomy, and inclusion challenges, particularly for overlooked groups like children, elderly, and Neurodivergent users. We present the Plural Voices Model (PVM), a novel single-agent framework that dynamically negotiates multi-user needs through real-time value alignment, leveraging diverse public datasets on mental health, eldercare, education, and moral reasoning. Using human+synthetic curriculum design with fairness-aware scenarios and ethical enhancements, PVM identifies core values, conflicts, and accessibility requirements to inform inclusive principles. Our privacy-focused prototype features adaptive safety scaffolds, tailored interactions (e.g., step-by-step guidance for Neurodivergent users, simple wording for children), and equitable conflict resolution. In preliminary evaluations, PVM outperforms multi-agent baselines in compliance (76% vs. 70%), fairness (90% vs. 85%), safety-violation rate (0% vs. 7%), and latency. Design innovations, including video guidance, autonomy sliders, family hubs, and adaptive safety dashboards, demonstrate new directions for ethical and inclusive domestic AI, for building user-centered agentic systems in plural domestic contexts. Our Codes and Model are been open sourced, available for reproduction: this https URL

**arXiv ID:** 2510.19008
</details>

<details>
<summary><strong>Local Guidance for Configuration-Based Multi-Agent Pathfinding</strong> - Tomoki Arita, Keisuke Okumura - [[pdf]](https://arxiv.org/pdf/2510.19072)</summary>

**Abstract:** Guidance is an emerging concept that improves the empirical performance of real-time, sub-optimal multi-agent pathfinding (MAPF) methods. It offers additional information to MAPF algorithms to mitigate congestion on a global scale by considering the collective behavior of all agents across the entire workspace. This global perspective helps reduce agents' waiting times, thereby improving overall coordination efficiency. In contrast, this study explores an alternative approach: providing local guidance in the vicinity of each agent. While such localized methods involve recomputation as agents move and may appear computationally demanding, we empirically demonstrate that supplying informative spatiotemporal cues to the planner can significantly improve solution quality without exceeding a moderate time budget. When applied to LaCAM, a leading configuration-based solver, this form of guidance establishes a new performance frontier for MAPF.

**arXiv ID:** 2510.19072
</details>

<details>
<summary><strong>AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation</strong> - Xianyang Liu, Yilin Liu, Shuai Wang, Hao Cheng, Andrew Estornell, Yuzhi Zhao, Jiaheng Wei - [[pdf]](https://arxiv.org/pdf/2510.19361)</summary>

**Abstract:** The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic pipeline for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives.

**arXiv ID:** 2510.19361
</details>

<details>
<summary><strong>ColorAgent: Building A Robust, Personalized, and Interactive OS Agent</strong> - Ning Li, Qiqiang Lin, Zheng Wu, Xiaoyun Mo, Weiming Zhang, Yin Zhao, Xiangmou Qu, Jiamu Zhou, Jun Wang, Congmin Zheng, Yuanyi Song, Hongjiang Chen, Heyuan Huang, Jihong Wang, Jiaxin Yin, Jingwei Yu, Junwei Liao, Qiuying Peng, Xingyu Lou, Jun Wang, Weiwen Liu, Zhuosheng Zhang, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2510.19386)</summary>

**Abstract:** With the advancements in hardware, software, and large language model technologies, the interaction between humans and operating systems has evolved from the command-line interface to the rapidly emerging AI agent interactions. Building an operating system (OS) agent capable of executing user instructions and faithfully following user desires is becoming a reality. In this technical report, we present ColorAgent, an OS agent designed to engage in long-horizon, robust interactions with the environment while also enabling personalized and proactive user interaction. To enable long-horizon interactions with the environment, we enhance the model's capabilities through step-wise reinforcement learning and self-evolving training, while also developing a tailored multi-agent framework that ensures generality, consistency, and robustness. In terms of user interaction, we explore personalized user intent recognition and proactive engagement, positioning the OS agent not merely as an automation tool but as a warm, collaborative partner. We evaluate ColorAgent on the AndroidWorld and AndroidLab benchmarks, achieving success rates of 77.2% and 50.7%, respectively, establishing a new state of the art. Nonetheless, we note that current benchmarks are insufficient for a comprehensive evaluation of OS agents and propose further exploring directions in future work, particularly in the areas of evaluation paradigms, agent collaboration, and security. Our code is available at this https URL.

**arXiv ID:** 2510.19386
</details>

<details>
<summary><strong>Monitoring LLM-based Multi-Agent Systems Against Corruptions via Node Evaluation</strong> - Chengcan Wu, Zhixin Zhang, Mingqian Xu, Zeming Wei, Meng Sun - [[pdf]](https://arxiv.org/pdf/2510.19420)</summary>

**Abstract:** Large Language Model (LLM)-based Multi-Agent Systems (MAS) have become a popular paradigm of AI applications. However, trustworthiness issues in MAS remain a critical concern. Unlike challenges in single-agent systems, MAS involve more complex communication processes, making them susceptible to corruption attacks. To mitigate this issue, several defense mechanisms have been developed based on the graph representation of MAS, where agents represent nodes and communications form edges. Nevertheless, these methods predominantly focus on static graph defense, attempting to either detect attacks in a fixed graph structure or optimize a static topology with certain defensive capabilities. To address this limitation, we propose a dynamic defense paradigm for MAS graph structures, which continuously monitors communication within the MAS graph, then dynamically adjusts the graph topology, accurately disrupts malicious communications, and effectively defends against evolving and diverse dynamic attacks. Experimental results in increasingly complex and dynamic MAS environments demonstrate that our method significantly outperforms existing MAS defense mechanisms, contributing an effective guardrail for their trustworthy applications. Our code is available at this https URL.

**arXiv ID:** 2510.19420
</details>

<details>
<summary><strong>Human-Agent Collaborative Paper-to-Page Crafting for Under $0.1</strong> - Qianli Ma, Siyu Wang, Yilin Chen, Yinhao Tang, Yixiang Yang, Chang Guo, Bingjie Gao, Zhening Xing, Yanan Sun, Zhipeng Zhang - [[pdf]](https://arxiv.org/pdf/2510.19600)</summary>

**Abstract:** In the quest for scientific progress, communicating research is as vital as the discovery itself. Yet, researchers are often sidetracked by the manual, repetitive chore of building project webpages to make their dense papers accessible. While automation has tackled static slides and posters, the dynamic, interactive nature of webpages has remained an unaddressed challenge. To bridge this gap, we reframe the problem, arguing that the solution lies not in a single command, but in a collaborative, hierarchical process. We introduce $\textbf{AutoPage}$, a novel multi-agent system that embodies this philosophy. AutoPage deconstructs paper-to-page creation into a coarse-to-fine pipeline from narrative planning to multimodal content generation and interactive rendering. To combat AI hallucination, dedicated "Checker" agents verify each step against the source paper, while optional human checkpoints ensure the final product aligns perfectly with the author's vision, transforming the system from a mere tool into a powerful collaborative assistant. To rigorously validate our approach, we also construct $\textbf{PageBench}$, the first benchmark for this new task. Experiments show AutoPage not only generates high-quality, visually appealing pages but does so with remarkable efficiency in under 15 minutes for less than \$0.1. Code and dataset will be released at $\href{this https URL}{Webpage}$.

**arXiv ID:** 2510.19600
</details>

<details>
<summary><strong>IM-Chat: A Multi-agent LLM Framework Integrating Tool-Calling and Diffusion Modeling for Knowledge Transfer in Injection Molding Industry</strong> - Junhyeong Lee, Joon-Young Kim, Heekyu Kim, Inhyo Lee, Seunghwa Ryu - [[pdf]](https://arxiv.org/pdf/2507.15268)</summary>

**Abstract:** The injection molding industry faces critical challenges in preserving and transferring field knowledge, particularly as experienced workers retire and multilingual barriers hinder effective communication. This study introduces IM-Chat, a multi-agent framework based on large language models (LLMs), designed to facilitate knowledge transfer in injection molding. IM-Chat integrates both limited documented knowledge (e.g., troubleshooting tables, manuals) and extensive field data modeled through a data-driven process condition generator that infers optimal manufacturing settings from environmental inputs such as temperature and humidity, enabling robust and context-aware task resolution. By adopting a retrieval-augmented generation (RAG) strategy and tool-calling agents within a modular architecture, IM-Chat ensures adaptability without the need for fine-tuning. Performance was assessed across 100 single-tool and 60 hybrid tasks for GPT-4o, GPT-4o-mini, and GPT-3.5-turbo by domain experts using a 10-point rubric focused on relevance and correctness, and was further supplemented by automated evaluation using GPT-4o guided by a domain-adapted instruction prompt. The evaluation results indicate that more capable models tend to achieve higher accuracy, particularly in complex, tool-integrated scenarios. In addition, compared with the fine-tuned single-agent LLM, IM-Chat demonstrated superior accuracy, particularly in quantitative reasoning, and greater scalability in handling multiple information sources. Overall, these findings demonstrate the viability of multi-agent LLM systems for industrial knowledge workflows and establish IM-Chat as a scalable and generalizable approach to AI-assisted decision support in manufacturing.

**arXiv ID:** 2507.15268
</details>

<details>
<summary><strong>RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration</strong> - Xiuyuan Chen, Jian Zhao, Yuchen Yuan, Tianle Zhang, Huilin Zhou, Zheng Zhu, Ping Hu, Linghe Kong, Chi Zhang, Weiran Huang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2509.25271)</summary>

**Abstract:** Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87% improvement in risk identification accuracy compared to the strongest baseline evaluation method.

**arXiv ID:** 2509.25271
</details>

<details>
<summary><strong>A Brain Cell Type Resource Created by Large Language Models and a Multi-Agent AI System for Collaborative Community Annotation</strong> - Rongbin Li, Wenbo Chen, Zhao Li, Rodrigo Munoz-Castaneda, Jinbo Li, Neha S. Maurya, Arnav Solanki, Huan He, Hanwen Xing, Meaghan Ramlakhan, Zachary Wise, Zhuhao Wu, Hua Xu, Michael Hawrylycz, W. Jim Zheng - [[pdf]](https://arxiv.org/pdf/2510.17064)</summary>

**Abstract:** Single-cell RNA sequencing has transformed our ability to identify diverse cell types and their transcriptomic signatures. However, annotating these signatures-especially those involving poorly characterized genes-remains a major challenge. Traditional methods, such as Gene Set Enrichment Analysis (GSEA), depend on well-curated annotations and often perform poorly in these contexts. Large Language Models (LLMs) offer a promising alternative but struggle to represent complex biological knowledge within structured ontologies. To address this, we present BRAINCELL-AID (BRAINCELL-AID: this https URL), a novel multi-agent AI system that integrates free-text descriptions with ontology labels to enable more accurate and robust gene set annotation. By incorporating retrieval-augmented generation (RAG), we developed a robust agentic workflow that refines predictions using relevant PubMed literature, reducing hallucinations and enhancing interpretability. Using this workflow, we achieved correct annotations for 77% of mouse gene sets among their top predictions. Applying this approach, we annotated 5,322 brain cell clusters from the comprehensive mouse brain cell atlas generated by the BRAIN Initiative Cell Census Network, enabling novel insights into brain cell function by identifying region-specific gene co-expression patterns and inferring functional roles of gene ensembles. BRAINCELL-AID also identifies Basal Ganglia-related cell types with neurologically meaningful descriptions. Hence, we create a valuable resource to support community-driven cell type annotation.

**arXiv ID:** 2510.17064
</details>

<details>
<summary><strong>PARCO: Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization</strong> - Federico Berto, Chuanbo Hua, Laurin Luttmann, Jiwoo Son, Junyoung Park, Kyuree Ahn, Changhyun Kwon, Lin Xie, Jinkyoo Park - [[pdf]](https://arxiv.org/pdf/2409.03811)</summary>

**Abstract:** Combinatorial optimization problems involving multiple agents are notoriously challenging due to their NP-hard nature and the necessity for effective agent coordination. Despite advancements in learning-based methods, existing approaches often face critical limitations, including suboptimal agent coordination, poor generalization, and high computational latency. To address these issues, we propose PARCO (Parallel AutoRegressive Combinatorial Optimization), a general reinforcement learning framework designed to construct high-quality solutions for multi-agent combinatorial tasks efficiently. To this end, PARCO integrates three key novel components: (1) transformer-based communication layers to enable effective agent collaboration during parallel solution construction, (2) a multiple pointer mechanism for low-latency, parallel agent decision-making, and (3) priority-based conflict handlers to resolve decision conflicts via learned priorities. We evaluate PARCO in multi-agent vehicle routing and scheduling problems, where our approach outperforms state-of-the-art learning methods, demonstrating strong generalization ability and remarkable computational efficiency. We make our source code publicly available to foster future research: this https URL.

**arXiv ID:** 2409.03811
</details>

<details>
<summary><strong>Coordinated Strategies in Realistic Air Combat by Hierarchical Multi-Agent Reinforcement Learning</strong> - Ardian Selmonaj, Giacomo Del Rio, Adrian Schneider, Alessandro Antonucci - [[pdf]](https://arxiv.org/pdf/2510.11474)</summary>

**Abstract:** Achieving mission objectives in a realistic simulation of aerial combat is highly challenging due to imperfect situational awareness and nonlinear flight dynamics. In this work, we introduce a novel 3D multi-agent air combat environment and a Hierarchical Multi-Agent Reinforcement Learning framework to tackle these challenges. Our approach combines heterogeneous agent dynamics, curriculum learning, league-play, and a newly adapted training algorithm. To this end, the decision-making process is organized into two abstraction levels: low-level policies learn precise control maneuvers, while high-level policies issue tactical commands based on mission objectives. Empirical results show that our hierarchical approach improves both learning efficiency and combat performance in complex dogfight scenarios.

**arXiv ID:** 2510.11474
</details>

<details>
<summary><strong>Polynomial-time Configuration Generator for Connected Unlabeled Multi-Agent Pathfinding</strong> - Takahiro Suzuki, Keisuke Okumura - [[pdf]](https://arxiv.org/pdf/2510.19567)</summary>

**Abstract:** We consider Connected Unlabeled Multi-Agent Pathfinding (CUMAPF), a variant of MAPF where the agents must maintain connectivity at all times. This problem is fundamental to swarm robotics applications like self-reconfiguration and marching, where standard MAPF is insufficient as it does not guarantee the required connectivity between agents. While unlabeled MAPF is tractable in optimization, CUMAPF is NP-hard even on highly restricted graph classes. To tackle this challenge, we propose PULL, a complete and polynomial-time algorithm with a simple design. It is based on a rule-based one-step function that computes a subsequent configuration that preserves connectivity and advances towards the target configuration. PULL is lightweight, and runs in $O(n^2)$ time per step in 2D grid, where $n$ is the number of agents. Our experiments further demonstrate its practical performance: PULL finds competitive solution qualities against trivial solutions for hundreds of agents, in randomly generated instances. Furthermore, we develop an eventually optimal solver that integrates PULL into an existing search-based MAPF algorithm, providing a valuable tool for small-scale instances.

**arXiv ID:** 2510.19567
</details>

<details>
<summary><strong>Emergence of Internal State-Modulated Swarming in Multi-Agent Patch Foraging System</strong> - Siddharth Chaturvedi, Ahmed EL-Gazzar, Marcel van Gerven - [[pdf]](https://arxiv.org/pdf/2510.18886)</summary>

**Abstract:** Active particles are entities that sustain persistent out-of-equilibrium motion by consuming energy. Under certain conditions, they exhibit the tendency to self-organize through coordinated movements, such as swarming via aggregation. While performing non-cooperative foraging tasks, the emergence of such swarming behavior in foragers, exemplifying active particles, has been attributed to the partial observability of the environment, in which the presence of another forager can serve as a proxy signal to indicate the potential presence of a food source or a resource patch. In this paper, we validate this phenomenon by simulating multiple self-propelled foragers as they forage from multiple resource patches in a non-cooperative manner. These foragers operate in a continuous two-dimensional space with stochastic position updates and partial observability. We evolve a shared policy in the form of a continuous-time recurrent neural network that serves as a velocity controller for the foragers. To this end, we use an evolutionary strategy algorithm wherein the different samples of the policy-distribution are evaluated in the same rollout. Then we show that agents are able to learn to adaptively forage in the environment. Next, we show the emergence of swarming in the form of aggregation among the foragers when resource patches are absent. We observe that the strength of this swarming behavior appears to be inversely proportional to the amount of resource stored in the foragers, which supports the risk-sensitive foraging claims. Empirical analysis of the learned controller's hidden states in minimal test runs uncovers their sensitivity to the amount of resource stored in a forager. Clamping these hidden states to represent a lesser amount of resource hastens its learned aggregation behavior.

**arXiv ID:** 2510.18886
</details>

<details>
<summary><strong>Visual Multi-Agent System: Mitigating Hallucination Snowballing via Visual Flow</strong> - Xinlei Yu, Chengming Xu, Guibin Zhang, Yongbo He, Zhangquan Chen, Zhucun Xue, Jiangning Zhang, Yue Liao, Xiaobin Hu, Yu-Gang Jiang, Shuicheng Yan - [[pdf]](https://arxiv.org/pdf/2509.21789)</summary>

**Abstract:** Multi-Agent System (MAS) powered by Visual Language Models (VLMs) enables challenging tasks but suffers from a novel failure term, multi-agent visual hallucination snowballing, where hallucinations are seeded in a single agent and amplified by following ones due to the over-reliance on textual flow to relay visual information. Through turn-, layer-, and token-wise attention analyses, we provide detailed insights into the essence of hallucination snowballing regarding the reduction of visual attention allocation. It leads us to identify a subset of vision tokens with a unimodal attention peak in middle layers that best preserve visual evidence but gradually diminish in deeper agent turns, resulting in the visual hallucination snowballing in MAS. Thus, we propose ViF, a lightweight, plug-and-play mitigation paradigm that relays inter-agent messages with Visual Flow powered by the selected visual relay tokens and applies attention reallocation to amplify this pattern. The experiment results demonstrate that our method markedly reduces hallucination snowballing, consistently improving the performance across eight benchmarks based on four common MAS structures and ten base models. The source code is publicly available at: this https URL.

**arXiv ID:** 2509.21789
</details>

<details>
<summary><strong>Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning</strong> - Rui Jerry Huang, Wendy Liu, Anastasia Miin, Lei Ding - [[pdf]](https://arxiv.org/pdf/2510.18179)</summary>

**Abstract:** Inference-time computation is a critical yet challenging paradigm for enhancing the reasoning performance of large language models (LLMs). While existing strategies improve reasoning stability and consistency, they suffer from notable limitations: self-correction often reinforces the model's initial biases, and Multi-Agent Collaboration (MAC) often fails due to the lack of efficient coordination mechanisms, leading to collective errors. Although high-performing verifiers can detect reasoning errors, making them reliable requires substantial training. To address these challenges, we introduce a novel inference-time framework, Adaptive Coopetition (AdCo), in which LLM agents utilize an adaptive, UCB-based "coopetition" mechanism. At each round, agents leverage coarse verifier signals to determine whether to collaborate or compete, and iteratively refine their reasoning based on peer feedback. Without relying on high-performance verifiers, our adaptive strategy achieves significant performance gains on mathematical reasoning benchmarks, yielding a 20% relative improvement over baselines on the more challenging dataset. Our approach remains robust and consistent in terms of accuracy under different sample sizes and configurations. This adaptive, signal-guided "coopetition" framework enhances reasoning robustness by leveraging both model knowledge diversity and reasoning trace measures, while also promoting uncertainty-driven exploration, especially when participants have comparable capabilities. From this perspective, our work offers a fresh lens on inference-time computation and paves the way for more resilient multi-agent LLM systems. Our code is available at: this https URL.

**arXiv ID:** 2510.18179
</details>

<details>
<summary><strong>SEC-bench: Automated Benchmarking of LLM Agents on Real-World Software Security Tasks</strong> - Hwiwon Lee, Ziqi Zhang, Hanxiao Lu, Lingming Zhang - [[pdf]](https://arxiv.org/pdf/2506.11791)</summary>

**Abstract:** Rigorous security-focused evaluation of large language model (LLM) agents is imperative for establishing trust in their safe deployment throughout the software development lifecycle. However, existing benchmarks largely rely on synthetic challenges or simplified vulnerability datasets that fail to capture the complexity and ambiguity encountered by security engineers in practice. We introduce SEC-bench, the first fully automated benchmarking framework for evaluating LLM agents on authentic security engineering tasks. SEC-bench employs a novel multi-agent scaffold that automatically constructs code repositories with harnesses, reproduces vulnerabilities in isolated environments, and generates gold patches for reliable evaluation. Our framework automatically creates high-quality software vulnerability datasets with reproducible artifacts at a cost of only $0.87 per instance. Using SEC-bench, we implement two critical software security tasks to rigorously evaluate LLM agents' capabilities: proof-of-concept (PoC) generation and vulnerability patching. A comprehensive evaluation of state-of-the-art LLM code agents reveals significant performance gaps, achieving at most 18.0% success in PoC generation and 34.0% in vulnerability patching on our complete dataset. These results highlight the crucial steps needed toward developing LLM agents that are more practical, intelligent, and autonomous for security engineering.

**arXiv ID:** 2506.11791
</details>

<details>
<summary><strong>AttentionSwarm: Reinforcement Learning with Attention Control Barier Function for Crazyflie Drones in Dynamic Environments</strong> - Grik Tadevosyan, Valerii Serpiva, Aleksey Fedoseev, Roohan Ahmed Khan, Demetros Aschu, Faryal Batool, Nickolay Efanov, Artem Mikhaylov, Dzmitry Tsetserukou - [[pdf]](https://arxiv.org/pdf/2503.07376)</summary>

**Abstract:** We introduce AttentionSwarm, a novel benchmark designed to evaluate safe and efficient swarm control in a dynamic drone racing scenario. Central to our approach is the Attention Model-Based Control Barrier Function (CBF) framework, which integrates attention mechanisms with safety-critical control theory to enable real-time collision avoidance and trajectory optimization. This framework dynamically prioritizes critical obstacles and agents in the swarm's vicinity using attention weights, while CBFs formally guarantee safety by enforcing collision-free constraints. The AttentionSwarm algorithm was developed and evaluated using a swarm of Crazyflie 2.1 micro quadrotors, which were tested indoors with the Vicon motion capture system to ensure precise localization and control. Experimental results show that our system achieves a 95-100% collision-free navigation rate in a dynamic multi-agent drone racing environment, underscoring its effectiveness and robustness in real-world scenarios. This work offers a promising foundation for safe, high-speed multi-robot applications in logistics, inspection, and racing.

**arXiv ID:** 2503.07376
</details>

</details>

<details open>
<summary><h2>Other Agent Research (7 papers)</h2></summary>

<details>
<summary><strong>VideoAgentTrek: Computer Use Pretraining from Unlabeled Videos</strong> - Dunjie Lu, Yiheng Xu, Junli Wang, Haoyuan Wu, Xinyuan Wang, Zekun Wang, Junlin Yang, Hongjin Su, Jixuan Chen, Junda Chen, Yuchen Mao, Jingren Zhou, Junyang Lin, Binyuan Hui, Tao Yu - [[pdf]](https://arxiv.org/pdf/2510.19488)</summary>

**Abstract:** Training computer-use agents requires massive amounts of GUI interaction data, but manually annotating action trajectories at scale is prohibitively expensive. We present VideoAgentTrek, a scalable pipeline that automatically mines training data from publicly available screen-recorded videos at web scale, eliminating the need for manual annotation. Our approach addresses a key challenge: raw videos contain implicit demonstrations but lack explicit action labels. To solve this, we develop Video2Action, an inverse dynamics module (IDM) with two components: (1) a video grounding model that detects and localizes GUI actions with precise temporal boundaries and context, and (2) an action-content recognizer that extracts structured parameters like click coordinates and typed text with high fidelity. Applied to 39,000 YouTube tutorial videos, our pipeline generates 1.52 million interaction steps automatically. We leverage this data through continued pretraining followed by supervised fine-tuning. On OSWorld-Verified, our approach improves task success rates from 9.3% (SFT-only baseline) to 15.8%, a 70% relative improvement. On AgentNetBench, step accuracy increases from 64.1% to 69.3%. Our results demonstrate that passive internet videos can be transformed into high-quality supervision for computer-use agents, providing a scalable alternative to expensive manual annotation.

**arXiv ID:** 2510.19488
</details>

<details>
<summary><strong>Modeling realistic human behavior using generative agents in a multimodal transport system: Software architecture and Application to Toulouse</strong> - Trung-Dung Vu, Benoit Gaudou, Kamaldeep Singh Oberoi - [[pdf]](https://arxiv.org/pdf/2510.19497)</summary>

**Abstract:** Modeling realistic human behaviour to understand people's mode choices in order to propose personalised mobility solutions remains challenging. This paper presents an architecture for modeling realistic human mobility behavior in complex multimodal transport systems, demonstrated through a case study in Toulouse, France. We apply Large Language Models (LLMs) within an agent-based simulation to capture decision-making in a real urban setting. The framework integrates the GAMA simulation platform with an LLM-based generative agent, along with General Transit Feed Specification (GTFS) data for public transport, and OpenTripPlanner for multimodal routing. GAMA platform models the interactive transport environment, providing visualization and dynamic agent interactions while eliminating the need to construct the simulation environment from scratch. This design enables a stronger focus on developing generative agents and evaluating their performance in transport decision-making processes. Over a simulated month, results show that agents not only make context-aware transport decisions but also form habits over time. We conclude that combining LLMs with agent-based simulation offers a promising direction for advancing intelligent transportation systems and personalised multimodal mobility solutions. We also discuss some limitations of this approach and outline future work on scaling to larger regions, integrating real-time data, and refining memory models.

**arXiv ID:** 2510.19497
</details>

<details>
<summary><strong>Toward Agentic Software Engineering Beyond Code: Framing Vision, Values, and Vocabulary</strong> - Rashina Hoda - [[pdf]](https://arxiv.org/pdf/2510.19692)</summary>

**Abstract:** Agentic AI is poised to usher in a seismic paradigm shift in Software Engineering (SE). As technologists rush head-along to make agentic AI a reality, SE researchers are driven to establish agentic SE as a research area. While early visions of agentic SE are primarily focused on code-related activities, early empirical evidence calls for a consideration of a range of socio-technical concerns to make it work in practice. This paper contributes to the emerging community vision by: (a) recommending an expansion of its scope beyond code, toward a 'whole of process' vision, grounding it in SE foundations and evolution and emerging agentic SE frameworks, (b) proposing a preliminary set of values and principles to guide efforts, and (c) sharing guidance on designing/using well-defined vocabulary for agentic SE. It is hoped that these ideas will encourage community collaborations and steer the SE community towards laying strong foundations of agentic SE so its not only inevitable but also deliberate and desirable in the long run.

**arXiv ID:** 2510.19692
</details>

<details>
<summary><strong>Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey</strong> - Weifan Guan, Qinghao Hu, Aosheng Li, Jian Cheng - [[pdf]](https://arxiv.org/pdf/2510.17111)</summary>

**Abstract:** Vision-Language-Action (VLA) models extend vision-language models to embodied control by mapping natural-language instructions and visual observations to robot actions. Despite their capabilities, VLA systems face significant challenges due to their massive computational and memory demands, which conflict with the constraints of edge platforms such as on-board mobile manipulators that require real-time performance. Addressing this tension has become a central focus of recent research. In light of the growing efforts toward more efficient and scalable VLA systems, this survey provides a systematic review of approaches for improving VLA efficiency, with an emphasis on reducing latency, memory footprint, and training and inference costs. We categorize existing solutions into four dimensions: model architecture, perception feature, action generation, and training/inference strategies, summarizing representative techniques within each category. Finally, we discuss future trends and open challenges, highlighting directions for advancing efficient embodied intelligence.

**arXiv ID:** 2510.17111
</details>

<details>
<summary><strong>Lost in the Maze: Overcoming Context Limitations in Long-Horizon Agentic Search</strong> - Howard Yen, Ashwin Paranjape, Mengzhou Xia, Thejas Venkatesh, Jack Hessel, Danqi Chen, Yuhao Zhang - [[pdf]](https://arxiv.org/pdf/2510.18939)</summary>

**Abstract:** Long-horizon agentic search requires iteratively exploring the web over long trajectories and synthesizing information across many sources, and is the foundation for enabling powerful applications like deep research systems. In this work, we show that popular agentic search frameworks struggle to scale to long trajectories primarily due to context limitations-they accumulate long, noisy content, hit context window and tool budgets, or stop early. Then, we introduce SLIM (Simple Lightweight Information Management), a simple framework that separates retrieval into distinct search and browse tools, and periodically summarizes the trajectory, keeping context concise while enabling longer, more focused searches. On long-horizon tasks, SLIM achieves comparable performance at substantially lower cost and with far fewer tool calls than strong open-source baselines across multiple base models. Specifically, with o3 as the base model, SLIM achieves 56% on BrowseComp and 31% on HLE, outperforming all open-source frameworks by 8 and 4 absolute points, respectively, while incurring 4-6x fewer tool calls. Finally, we release an automated fine-grained trajectory analysis pipeline and error taxonomy for characterizing long-horizon agentic search frameworks; SLIM exhibits fewer hallucinations than prior systems. We hope our analysis framework and simple tool design inform future long-horizon agents.

**arXiv ID:** 2510.18939
</details>

<details>
<summary><strong>Risk Assessment of an Autonomous Underwater Snake Robot in Confined Operations</strong> - Abdelrahman Sayed Sayed - [[pdf]](https://arxiv.org/pdf/2510.19415)</summary>

**Abstract:** The growing interest in ocean discovery imposes a need for inspection and intervention in confined and demanding environments. Eely's slender shape, in addition to its ability to change its body configurations, makes articulated underwater robots an adequate option for such environments. However, operation of Eely in such environments imposes demanding requirements on the system, as it must deal with uncertain and unstructured environments, extreme environmental conditions, and reduced navigational capabilities. This paper proposes a Bayesian approach to assess the risks of losing Eely during two mission scenarios. The goal of this work is to improve Eely's performance and the likelihood of mission success. Sensitivity analysis results are presented in order to demonstrate the causes having the highest impact on losing Eely.

**arXiv ID:** 2510.19415
</details>

<details>
<summary><strong>ComDrive: Comfort-Oriented End-to-End Autonomous Driving</strong> - Junming Wang, Xingyu Zhang, Zebin Xing, Songen Gu, Xiaoyang Guo, Yang Hu, Ziying Song, Qian Zhang, Xiaoxiao Long, Wei Yin - [[pdf]](https://arxiv.org/pdf/2410.05051)</summary>

**Abstract:** We propose ComDrive: the first comfort-oriented end-to-end autonomous driving system to generate temporally consistent and comfortable trajectories. Recent studies have demonstrated that imitation learning-based planners and learning-based trajectory scorers can effectively generate and select safety trajectories that closely mimic expert demonstrations. However, such trajectory planners and scorers face the challenge of generating temporally inconsistent and uncomfortable trajectories. To address these issues, ComDrive first extracts 3D spatial representations through sparse perception, which then serves as conditional inputs. These inputs are used by a Conditional Denoising Diffusion Probabilistic Model (DDPM)-based motion planner to generate temporally consistent multi-modal trajectories. A dual-stream adaptive trajectory scorer subsequently selects the most comfortable trajectory from these candidates to control the vehicle. Experiments demonstrate that ComDrive achieves state-of-the-art performance in both comfort and safety, outperforming UniAD by 17% in driving comfort and reducing collision rates by 25% compared to SparseDrive. More results are available on our project page: this https URL.

**arXiv ID:** 2410.05051
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (27 papers)</h2></summary>

<details>
<summary><strong>Continual Knowledge Adaptation for Reinforcement Learning</strong> - Jinwu Hu, Zihao Lian, Zhiquan Wen, Chenghao Li, Guohao Chen, Xutao Wen, Bin Xiao, Mingkui Tan - [[pdf]](https://arxiv.org/pdf/2510.19314)</summary>

**Abstract:** Reinforcement Learning enables agents to learn optimal behaviors through interactions with environments. However, real-world environments are typically non-stationary, requiring agents to continuously adapt to new tasks and changing conditions. Although Continual Reinforcement Learning facilitates learning across multiple tasks, existing methods often suffer from catastrophic forgetting and inefficient knowledge utilization. To address these challenges, we propose Continual Knowledge Adaptation for Reinforcement Learning (CKA-RL), which enables the accumulation and effective utilization of historical knowledge. Specifically, we introduce a Continual Knowledge Adaptation strategy, which involves maintaining a task-specific knowledge vector pool and dynamically using historical knowledge to adapt the agent to new tasks. This process mitigates catastrophic forgetting and enables efficient knowledge transfer across tasks by preserving and adapting critical model parameters. Additionally, we propose an Adaptive Knowledge Merging mechanism that combines similar knowledge vectors to address scalability challenges, reducing memory requirements while ensuring the retention of essential knowledge. Experiments on three benchmarks demonstrate that the proposed CKA-RL outperforms state-of-the-art methods, achieving an improvement of 4.20% in overall performance and 8.02% in forward transfer. The source code is available at this https URL.

**arXiv ID:** 2510.19314
</details>

<details>
<summary><strong>NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning</strong> - Wonje Choi, Jooyoung Kim, Honguk Woo - [[pdf]](https://arxiv.org/pdf/2510.19429)</summary>

**Abstract:** We address the challenge of adopting language models (LMs) for embodied tasks in dynamic environments, where online access to large-scale inference engines or symbolic planners is constrained due to latency, connectivity, and resource limitations. To this end, we present NeSyPr, a novel embodied reasoning framework that compiles knowledge via neurosymbolic proceduralization, thereby equipping LM-based agents with structured, adaptive, and timely reasoning capabilities. In NeSyPr, task-specific plans are first explicitly generated by a symbolic tool leveraging its declarative knowledge. These plans are then transformed into composable procedural representations that encode the plans' implicit production rules, enabling the resulting composed procedures to be seamlessly integrated into the LM's inference process. This neurosymbolic proceduralization abstracts and generalizes multi-step symbolic structured path-finding and reasoning into single-step LM inference, akin to human knowledge compilation. It supports efficient test-time inference without relying on external symbolic guidance, making it well suited for deployment in latency-sensitive and resource-constrained physical systems. We evaluate NeSyPr on the embodied benchmarks PDDLGym, VirtualHome, and ALFWorld, demonstrating its efficient reasoning capabilities over large-scale reasoning models and a symbolic planner, while using more compact LMs.

**arXiv ID:** 2510.19429
</details>

<details>
<summary><strong>DAIL: Beyond Task Ambiguity for Language-Conditioned Reinforcement Learning</strong> - Runpeng Xie, Quanwei Wang, Hao Hu, Zherui Zhou, Ni Mu, Xiyun Li, Yiqin Yang, Shuang Xu, Qianchuan Zhao, Bo XU - [[pdf]](https://arxiv.org/pdf/2510.19562)</summary>

**Abstract:** Comprehending natural language and following human instructions are critical capabilities for intelligent agents. However, the flexibility of linguistic instructions induces substantial ambiguity across language-conditioned tasks, severely degrading algorithmic performance. To address these limitations, we present a novel method named DAIL (Distributional Aligned Learning), featuring two key components: distributional policy and semantic alignment. Specifically, we provide theoretical results that the value distribution estimation mechanism enhances task differentiability. Meanwhile, the semantic alignment module captures the correspondence between trajectories and linguistic instructions. Extensive experimental results on both structured and visual observation benchmarks demonstrate that DAIL effectively resolves instruction ambiguities, achieving superior performance to baseline methods. Our implementation is available at this https URL.

**arXiv ID:** 2510.19562
</details>

<details>
<summary><strong>Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning</strong> - Gunshi Gupta, Karmesh Yadav, Zsolt Kira, Yarin Gal, Rahaf Aljundi - [[pdf]](https://arxiv.org/pdf/2510.19732)</summary>

**Abstract:** To enable embodied agents to operate effectively over extended timeframes, it is crucial to develop models that form and access memories to stay contextualized in their environment. In the current paradigm of training transformer-based policies for embodied sequential decision-making tasks, visual inputs often overwhelm the context limits of transformers, while humans can maintain and utilize a lifetime of experience compressed as memories. Significant compression is possible in principle, as much of the input is irrelevant and can be abstracted. However, existing approaches predominantly focus on either recurrent models with fixed-size memory or transformers with full-context reliance. In this work, we propose Memo, a transformer-based architecture and training recipe for reinforcement learning (RL) on memory-intensive, long-horizon tasks. Memo incorporates the creation and retrieval of memory by interleaving periodic summarization tokens with the inputs of a model during training. We demonstrate Memo's effectiveness on a gridworld meta-RL benchmark and a multi-object navigation task in photo-realistic indoor settings. Memo outperforms naive long-context transformer baselines while being more compute and storage efficient. Additionally, Memo generalizes better to longer contexts at inference time and remains robust in streaming settings, where historical context must be truncated to fit inference constraints.

**arXiv ID:** 2510.19732
</details>

<details>
<summary><strong>CosmoCore Affective Dream-Replay Reinforcement Learning for Code Generation</strong> - Santhosh Kumar Ravindran - [[pdf]](https://arxiv.org/pdf/2510.18895)</summary>

**Abstract:** We introduce CosmoCore, a neuroscience-inspired reinforcement learning (RL) architecture that integrates affective signals to enhance code generation in large language models (LLMs). Motivated by human and animal learning where embarrassment from mistakes drives rapid correction, as observed in training a puppy to avoid repeating errors after a single scolding CosmoCore tags code generation trajectories with valence and surprise using a lightweight multi-layer perceptron (MLP). High-negative valence (cringe) episodes, such as buggy code outputs, are prioritized in a Dream Queue for five-fold replay during off-policy updates, while low-surprise successes are pruned to prevent overconfidence and buffer bloat. Evaluated on code generation benchmarks like HumanEval and BigCodeBench, alongside simulations with a custom data pipeline environment, CosmoCore reduces hallucinated code (e.g., syntax errors or logical bugs) by 48\% and accelerates self-correction by 45\%. Local experiments using Hugging Face models in a PySpark environment validate these gains, with code snippets provided for replication. Ablations confirm valence tagging boosts curiosity in exploration, and pruning mitigates inefficiency. This framework extends RL from human feedback (RLHF) for more emotionally aware code assistants, with applications in IDEs and data pipelines. Code and the custom mini-world simulation are released.

**arXiv ID:** 2510.18895
</details>

<details>
<summary><strong>BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping</strong> - Zhiheng Xi, Xin Guo, Yang Nan, Enyu Zhou, Junrui Shen, Wenxiang Chen, Jiaqi Liu, Jixuan Huang, Zhihao Zhang, Honglin Guo, Xun Deng, Zhikai Lei, Miao Zheng, Guoteng Wang, Shuo Zhang, Peng Sun, Rui Zheng, Hang Yan, Tao Gui, Qi Zhang, Xuanjing Huang - [[pdf]](https://arxiv.org/pdf/2510.18927)</summary>

**Abstract:** Reinforcement learning (RL) has recently become the core paradigm for aligning and strengthening large language models (LLMs). Yet, applying RL in off-policy settings--where stale data from past policies are used for training--improves sample efficiency, but remains challenging: policy entropy declines sharply, optimization often becomes unstable and may even collapse. Through theoretical and empirical analysis, we identify two key insights: (i) an imbalance in optimization, where negative-advantage samples dominate the policy gradient, suppressing useful behaviors and risking gradient explosions; and (ii) the derived Entropy-Clip Rule, which reveals that the fixed clipping mechanism in PPO-like objectives systematically blocks entropy-increasing updates, thereby driving the policy toward over-exploitation at the expense of exploration. Building on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a simple yet effective method that dynamically adjusts clipping bounds to adaptively re-balance positive and negative contributions, preserve entropy, and stabilize RL optimization. Across diverse off-policy scenarios--including sample replay and partial rollout--BAPO achieves fast, stable, and data-efficient training. On AIME 2024 and AIME 2025 benchmarks, our 7B BAPO model surpasses open-source counterparts such as SkyWork-OR1-7B, while our 32B BAPO model not only achieves state-of-the-art results among models of the same scale but also outperforms leading proprietary systems like o3-mini and Gemini-2.5-Flash-Thinking.

**arXiv ID:** 2510.18927
</details>

<details>
<summary><strong>Balancing Rewards in Text Summarization: Multi-Objective Reinforcement Learning via HyperVolume Optimization</strong> - Junjie Song, Yiwen Liu, Dapeng Li, Yin Sun, Shukun Fu, Siqi Chen, Yuji Cao - [[pdf]](https://arxiv.org/pdf/2510.19325)</summary>

**Abstract:** Text summarization is a crucial task that requires the simultaneous optimization of multiple objectives, including consistency, coherence, relevance, and fluency, which presents considerable challenges. Although large language models (LLMs) have demonstrated remarkable performance, enhanced by reinforcement learning (RL), few studies have focused on optimizing the multi-objective problem of summarization through RL based on LLMs. In this paper, we introduce hypervolume optimization (HVO), a novel optimization strategy that dynamically adjusts the scores between groups during the reward process in RL by using the hypervolume method. This method guides the model's optimization to progressively approximate the pareto front, thereby generating balanced summaries across multiple objectives. Experimental results on several representative summarization datasets demonstrate that our method outperforms group relative policy optimization (GRPO) in overall scores and shows more balanced performance across different dimensions. Moreover, a 7B foundation model enhanced by HVO performs comparably to GPT-4 in the summarization task, while maintaining a shorter generation length. Our code is publicly available at this https URL

**arXiv ID:** 2510.19325
</details>

<details>
<summary><strong>Using Non-Expert Data to Robustify Imitation Learning via Offline Reinforcement Learning</strong> - Kevin Huang, Rosario Scalise, Cleah Winston, Ayush Agrawal, Yunchu Zhang, Rohan Baijal, Markus Grotz, Byron Boots, Benjamin Burchfiel, Hongkai Dai, Masha Itkina, Paarth Shah, Abhishek Gupta - [[pdf]](https://arxiv.org/pdf/2510.19495)</summary>

**Abstract:** Imitation learning has proven effective for training robots to perform complex tasks from expert human demonstrations. However, it remains limited by its reliance on high-quality, task-specific data, restricting adaptability to the diverse range of real-world object configurations and scenarios. In contrast, non-expert data -- such as play data, suboptimal demonstrations, partial task completions, or rollouts from suboptimal policies -- can offer broader coverage and lower collection costs. However, conventional imitation learning approaches fail to utilize this data effectively. To address these challenges, we posit that with right design decisions, offline reinforcement learning can be used as a tool to harness non-expert data to enhance the performance of imitation learning policies. We show that while standard offline RL approaches can be ineffective at actually leveraging non-expert data under the sparse data coverage settings typically encountered in the real world, simple algorithmic modifications can allow for the utilization of this data, without significant additional assumptions. Our approach shows that broadening the support of the policy distribution can allow imitation algorithms augmented by offline RL to solve tasks robustly, showing considerably enhanced recovery and generalization behavior. In manipulation tasks, these innovations significantly increase the range of initial conditions where learned policies are successful when non-expert data is incorporated. Moreover, we show that these methods are able to leverage all collected data, including partial or suboptimal demonstrations, to bolster task-directed policy performance. This underscores the importance of algorithmic techniques for using non-expert data for robust policy learning in robotics.

**arXiv ID:** 2510.19495
</details>

<details>
<summary><strong>Optimizing the Unknown: Black Box Bayesian Optimization with Energy-Based Model and Reinforcement Learning</strong> - Ruiyao Miao, Junren Xiao, Shiya Tsang, Hui Xiong, Yingnian Wu - [[pdf]](https://arxiv.org/pdf/2510.19530)</summary>

**Abstract:** Existing Bayesian Optimization (BO) methods typically balance exploration and exploitation to optimize costly objective functions. However, these methods often suffer from a significant one-step bias, which may lead to convergence towards local optima and poor performance in complex or high-dimensional tasks. Recently, Black-Box Optimization (BBO) has achieved success across various scientific and engineering domains, particularly when function evaluations are costly and gradients are unavailable. Motivated by this, we propose the Reinforced Energy-Based Model for Bayesian Optimization (REBMBO), which integrates Gaussian Processes (GP) for local guidance with an Energy-Based Model (EBM) to capture global structural information. Notably, we define each Bayesian Optimization iteration as a Markov Decision Process (MDP) and use Proximal Policy Optimization (PPO) for adaptive multi-step lookahead, dynamically adjusting the depth and direction of exploration to effectively overcome the limitations of traditional BO methods. We conduct extensive experiments on synthetic and real-world benchmarks, confirming the superior performance of REBMBO. Additional analyses across various GP configurations further highlight its adaptability and robustness.

**arXiv ID:** 2510.19530
</details>

<details>
<summary><strong>ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning</strong> - Roger Creus Castanyer, Faisal Mohamed, Pablo Samuel Castro, Cyrus Neary, Glen Berseth - [[pdf]](https://arxiv.org/pdf/2510.14176)</summary>

**Abstract:** Reinforcement learning (RL) algorithms are highly sensitive to reward function specification, which remains a central challenge limiting their broad applicability. We present ARM-FM: Automated Reward Machines via Foundation Models, a framework for automated, compositional reward design in RL that leverages the high-level reasoning capabilities of foundation models (FMs). Reward machines (RMs) -- an automata-based formalism for reward specification -- are used as the mechanism for RL objective specification, and are automatically constructed via the use of FMs. The structured formalism of RMs yields effective task decompositions, while the use of FMs enables objective specifications in natural language. Concretely, we (i) use FMs to automatically generate RMs from natural language specifications; (ii) associate language embeddings with each RM automata-state to enable generalization across tasks; and (iii) provide empirical evidence of ARM-FM's effectiveness in a diverse suite of challenging environments, including evidence of zero-shot generalization.

**arXiv ID:** 2510.14176
</details>

<details>
<summary><strong>RoboGPT-R1: Enhancing Robot Planning with Reinforcement Learning</strong> - Jinrui Liu, Bingyan Nie, Boyu Li, Yaran Chen, Yuze Wang, Shunsen He, Haoran Li - [[pdf]](https://arxiv.org/pdf/2510.14828)</summary>

**Abstract:** Improving the reasoning capabilities of embodied agents is crucial for robots to complete complex human instructions in long-view manipulation tasks successfully. Despite the success of large language models and vision language models based on Supervised Fine-Tuning (SFT) in planning tasks, they continue facing challenges in performing long-horizon manipulation tasks in complex real-world environments, owing to their restricted common sense and reasoning capabilities. Considering that aligning general-purpose vision language models to robotic planning tasks via supervised fine-tuning suffers from poor generalization and insufficient physical understanding, we propose RoboGPT-R1, a two-stage fine-tuning framework for embodied planning. In this framework, supervised training acquires foundational knowledge through expert sequences, followed by RL to address the model's shortcomings in visual-spatial understanding and reasoning. To achieve physical understanding and action sequence consistency in multi-step reasoning tasks, we design a rule-based reward function that simultaneously considers long-horizon performance and action constraint in the environment. The reasoning model, trained on Qwen2.5-VL-3B, significantly outperforms the larger-scale model, GPT-4o-mini, by 21.33% and surpasses other work trained on Qwen2.5-VL-7B by 20.33% on the EmbodiedBench benchmark.

**arXiv ID:** 2510.14828
</details>

<details>
<summary><strong>FP-IRL: Fokker-Planck Inverse Reinforcement Learning -- A Physics-Constrained Approach to Markov Decision Processes</strong> - Chengyang Huang, Siddhartha Srivastava, Kenneth K. Y. Ho, Kathy E. Luker, Gary D. Luker, Xun Huan, Krishna Garikipati - [[pdf]](https://arxiv.org/pdf/2306.10407)</summary>

**Abstract:** Inverse reinforcement learning (IRL) is a powerful paradigm for uncovering the incentive structure that drives agent behavior, by inferring an unknown reward function from observed trajectories within a Markov decision process (MDP). However, most existing IRL methods require access to the transition function, either prescribed or estimated \textit{a priori}, which poses significant challenges when the underlying dynamics are unknown, unobservable, or not easily sampled.
We propose Fokker--Planck inverse reinforcement learning (FP-IRL), a novel physics-constrained IRL framework tailored for systems governed by Fokker--Planck (FP) dynamics. FP-IRL simultaneously infers both the reward and transition functions directly from trajectory data, without requiring access to sampled transitions. Our method leverages a conjectured equivalence between MDPs and the FP equation, linking reward maximization in MDPs with free energy minimization in FP dynamics. This connection enables inference of the potential function using our inference approach of variational system identification, from which the full set of MDP components -- reward, transition, and policy -- can be recovered using analytic expressions.
We demonstrate the effectiveness of FP-IRL through experiments on synthetic benchmarks and a modified version of the Mountain Car problem. Our results show that FP-IRL achieves accurate recovery of agent incentives while preserving computational efficiency and physical interpretability.

**arXiv ID:** 2306.10407
</details>

<details>
<summary><strong>Provably Efficient Reward Transfer in Reinforcement Learning with Discrete Markov Decision Processes</strong> - Kevin Vora, Yu Zhang - [[pdf]](https://arxiv.org/pdf/2503.13414)</summary>

**Abstract:** In this paper, we propose a new solution to reward adaptation (RA) in reinforcement learning, where the agent adapts to a target reward function based on one or more existing source behaviors learned a priori under the same domain dynamics but different reward functions. While learning the target behavior from scratch is possible, it is often inefficient given the available source behaviors. Our work introduces a new approach to RA through the manipulation of Q-functions. Assuming the target reward function is a known function of the source reward functions, we compute bounds on the Q-function and present an iterative process (akin to value iteration) to tighten these bounds. Such bounds enable action pruning in the target domain before learning even starts. We refer to this method as "Q-Manipulation" (Q-M). The iteration process assumes access to a lite-model, which is easy to provide or learn. We formally prove that Q-M, under discrete domains, does not affect the optimality of the returned policy and show that it is provably efficient in terms of sample complexity in a probabilistic sense. Q-M is evaluated in a variety of synthetic and simulation domains to demonstrate its effectiveness, generalizability, and practicality.

**arXiv ID:** 2503.13414
</details>

<details>
<summary><strong>PTFA: An LLM-based Agent that Facilitates Online Consensus Building through Parallel Thinking</strong> - Wen Gu, Zhaoxing Li, Jan Buermann, Jim Dilkes, Dimitris Michailidis, Shinobu Hasegawa, Vahid Yazdanpanah, Sebastian Stein - [[pdf]](https://arxiv.org/pdf/2503.12499)</summary>

**Abstract:** Consensus building is inherently challenging due to the diverse opinions held by stakeholders. Effective facilitation is crucial to support the consensus building process and enable efficient group decision making. However, the effectiveness of facilitation is often constrained by human factors such as limited experience and scalability. In this research, we propose a Parallel Thinking-based Facilitation Agent (PTFA) that facilitates online, text-based consensus building this http URL PTFA automatically collects real-time textual input and leverages large language models (LLMs)to perform all six distinct roles of the well-established Six Thinking Hats technique in parallel this http URL illustrate the potential of the agent, a pilot study was conducted, demonstrating its capabilities in idea generation, emotional probing, and deeper analysis of idea quality. Additionally, future open research challenges such as optimizing scheduling and managing behaviors in divergent phase are identified. Furthermore, a comprehensive dataset that contains not only the conversational content among the participants but also between the participants and the agent is constructed for future study.

**arXiv ID:** 2503.12499
</details>

<details>
<summary><strong>MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research</strong> - Hui Chen, Miao Xiong, Yujie Lu, Wei Han, Ailin Deng, Yufei He, Jiaying Wu, Yibo Li, Yue Liu, Bryan Hooi - [[pdf]](https://arxiv.org/pdf/2505.19955)</summary>

**Abstract:** Recent advancements in AI agents have demonstrated their growing potential to drive and support scientific discovery. In this work, we introduce MLR-Bench, a comprehensive benchmark for evaluating AI agents on open-ended machine learning research. MLR-Bench includes three key components: (1) 201 research tasks sourced from NeurIPS, ICLR, and ICML workshops covering diverse ML topics; (2) MLR-Judge, an automated evaluation framework combining LLM-based reviewers with carefully designed review rubrics to assess research quality; and (3) MLR-Agent, a modular agent scaffold capable of completing research tasks through four stages: idea generation, proposal formulation, experimentation, and paper writing. Our framework supports both stepwise assessment across these distinct research stages, and end-to-end evaluation of the final research paper. We then use MLR-Bench to evaluate six frontier LLMs and an advanced coding agent, finding that while LLMs are effective at generating coherent ideas and well-structured papers, current coding agents frequently (e.g., in 80% of the cases) produce fabricated or invalidated experimental results--posing a major barrier to scientific reliability. We validate MLR-Judge through human evaluation, showing high agreement with expert reviewers, supporting its potential as a scalable tool for research evaluation. We open-source MLR-Bench to help the community benchmark, diagnose, and improve AI research agents toward trustworthy and transparent scientific discovery.

**arXiv ID:** 2505.19955
</details>

<details>
<summary><strong>RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Interactive Environmental Learning in Physical Embodied Systems</strong> - Mingcong Lei, Honghao Cai, Zezhou Cui, Liangchen Tan, Junkun Hong, Gehan Hu, Shuangyu Zhu, Yimou Wu, Shaohan Jiang, Ge Wang, Yuyuan Yang, Junyuan Tan, Zhenglin Wan, Zhen Li, Shuguang Cui, Yiming Zhao, Yatong Han - [[pdf]](https://arxiv.org/pdf/2508.01415)</summary>

**Abstract:** Embodied agents face persistent challenges in real-world environments, including partial observability, limited spatial reasoning, and high-latency multi-memory integration. We present RoboMemory, a brain-inspired framework that unifies Spatial, Temporal, Episodic, and Semantic memory under a parallelized architecture for efficient long-horizon planning and interactive environmental learning. A dynamic spatial knowledge graph (KG) ensures scalable and consistent memory updates, while a closed-loop planner with a critic module supports adaptive decision-making in dynamic settings. Experiments on EmbodiedBench show that RoboMemory, built on Qwen2.5-VL-72B-Ins, improves average success rates by 25% over its baseline and exceeds the closed-source state-of-the-art (SOTA) Gemini-1.5-Pro by 3%. Real-world trials further confirm its capacity for cumulative learning, with performance improving across repeated tasks. These results highlight RoboMemory as a scalable foundation for memory-augmented embodied intelligence, bridging the gap between cognitive neuroscience and robotic autonomy.

**arXiv ID:** 2508.01415
</details>

<details>
<summary><strong>Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning</strong> - Yang Zhou, Sunzhu Li, Shunyu Liu, Wenkai Fang, Kongcheng Zhang, Jiale Zhao, Jingwen Yang, Yihe Zhou, Jianwei Lv, Tongya Zheng, Hengtong Lu, Wei Chen, Yan Xie, Mingli Song - [[pdf]](https://arxiv.org/pdf/2508.16949)</summary>

**Abstract:** Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the Best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3. Our code is available at this https URL.

**arXiv ID:** 2508.16949
</details>

<details>
<summary><strong>Agentic Inequality</strong> - Matthew Sharp, Omer Bilgin, Iason Gabriel, Lewis Hammond - [[pdf]](https://arxiv.org/pdf/2510.16853)</summary>

**Abstract:** Autonomous AI agents, capable of complex planning and action, represent a significant technological evolution beyond current generative tools. As these systems become integrated into political and economic life, their distribution and capabilities will be highly consequential. This paper introduces and explores "agentic inequality" - the potential disparities in power, opportunity, and outcomes stemming from differential access to, and capabilities of, AI agents. We analyse the dual potential of this technology, exploring how agents could both exacerbate existing divides and, under the right conditions, serve as a powerful equalising force. To this end, the paper makes three primary contributions. First, it establishes an analytical framework by delineating the three core dimensions through which this inequality can manifest: disparities in the availability, quality, and quantity of agents. Second, it argues that agentic inequality is distinct from prior technological divides. Unlike tools that primarily augment human abilities, agents act as autonomous delegates, creating novel power asymmetries through scalable goal delegation and direct agent-to-agent competition that are poised to reshape outcomes across economic and socio-political spheres. Finally, it provides a systematic analysis of the technical and socioeconomic drivers - from model release strategies to market incentives - that will shape the distribution of agentic power, concluding with a research agenda for navigating the complex governance challenges ahead.

**arXiv ID:** 2510.16853
</details>

<details>
<summary><strong>Vahana.jl -- A framework (not only) for large-scale agent-based models</strong> - Steffen Fürst, Tim Conrad, Carlo Jaeger, Sarah Wolf - [[pdf]](https://arxiv.org/pdf/2406.14441)</summary>

**Abstract:** Agent-based models (ABMs) offer a powerful framework for understanding complex systems. However, their computational demands often become a significant barrier as the number of agents and complexity of the simulation increase. Traditional ABM platforms often struggle to fully exploit modern computing resources, hindering the development of large-scale simulations. This paper presents this http URL, a high performance computing open source framework that aims to address these limitations. Building on the formalism of synchronous graph dynamical systems, this http URL is especially well suited for models with a focus on (social) networks. The framework seamlessly supports distribution across multiple compute nodes, enabling simulations that would otherwise be beyond the capabilities of a single machine. Implemented in Julia, this http URL leverages the interactive Read-Eval-Print Loop (REPL) environment, facilitating rapid model development and experimentation.

**arXiv ID:** 2406.14441
</details>

<details>
<summary><strong>SheetBrain: A Neuro-Symbolic Agent for Accurate Reasoning over Complex and Large Spreadsheets</strong> - Ziwei Wang, Jiayuan Su, Mengyu Zhou, Huaxing Zeng, Mengni Jia, Xiao Lv, Haoyu Dong, Xiaojun Ma, Shi Han, Dongmei Zhang - [[pdf]](https://arxiv.org/pdf/2510.19247)</summary>

**Abstract:** Understanding and reasoning over complex spreadsheets remain fundamental challenges for large language models (LLMs), which often struggle with accurately capturing the complex structure of tables and ensuring reasoning correctness. In this work, we propose SheetBrain, a neuro-symbolic dual workflow agent framework designed for accurate reasoning over tabular data, supporting both spreadsheet question answering and manipulation tasks. SheetBrain comprises three core modules: an understanding module, which produces a comprehensive overview of the spreadsheet - including sheet summary and query-based problem insight to guide reasoning; an execution module, which integrates a Python sandbox with preloaded table-processing libraries and an Excel helper toolkit for effective multi-turn reasoning; and a validation module, which verifies the correctness of reasoning and answers, triggering re-execution when necessary. We evaluate SheetBrain on multiple public tabular QA and manipulation benchmarks, and introduce SheetBench, a new benchmark targeting large, multi-table, and structurally complex spreadsheets. Experimental results show that SheetBrain significantly improves accuracy on both existing benchmarks and the more challenging scenarios presented in SheetBench. Our code is publicly available at this https URL.

**arXiv ID:** 2510.19247
</details>

<details>
<summary><strong>LoongRL:Reinforcement Learning for Advanced Reasoning over Long Contexts</strong> - Siyuan Wang, Gaokai Zhang, Li Lyna Zhang, Ning Shang, Fan Yang, Dongyao Chen, Mao Yang - [[pdf]](https://arxiv.org/pdf/2510.19363)</summary>

**Abstract:** Reasoning over long contexts is essential for large language models. While reinforcement learning (RL) enhances short-context reasoning by inducing "Aha" moments in chain-of-thought, the advanced thinking patterns required for long-context reasoning remain largely unexplored, and high-difficulty RL data are scarce. In this paper, we introduce LoongRL, a data-driven RL method for advanced long-context reasoning. Central to LoongRL is KeyChain, a synthesis approach that transforms short multi-hop QA into high-difficulty long-context tasks by inserting UUID chains that hide the true question among large collections of distracting documents. Solving these tasks requires the model to trace the correct chain step-by-step, identify the true question, retrieve relevant facts and reason over them to answer correctly. RL training on KeyChain data induces an emergent plan-retrieve-reason-recheck reasoning pattern that generalizes far beyond training length. Models trained at 16K effectively solve 128K tasks without prohibitive full-length RL rollout costs. On Qwen2.5-7B and 14B, LoongRL substantially improves long-context multi-hop QA accuracy by +23.5% and +21.1% absolute gains. The resulting LoongRL-14B reaches a score of 74.2, rivaling much larger frontier models such as o3-mini (74.5) and DeepSeek-R1 (74.9). It also improves long-context retrieval, passes all 128K needle-in-a-haystack stress tests, and preserves short-context reasoning capabilities.

**arXiv ID:** 2510.19363
</details>

<details>
<summary><strong>Presenting a Paper is an Art: Self-Improvement Aesthetic Agents for Academic Presentations</strong> - Chengzhi Liu, Yuzhe Yang, Kaiwen Zhou, Zhen Zhang, Yue Fan, Yanan Xie, Peng Qi, Xin Eric Wang - [[pdf]](https://arxiv.org/pdf/2510.05571)</summary>

**Abstract:** The promotion of academic papers has become an important means of enhancing research visibility. However, existing automated methods struggle limited storytelling, insufficient aesthetic quality, and constrained self-adjustment, making it difficult to achieve efficient and engaging dissemination. At the heart of those challenges is a simple principle: \emph{there is no way to improve it when you cannot evaluate it right}. To address this, we introduce \textbf{EvoPresent}, a self-improvement agent framework that unifies coherent narratives, aesthetic-aware designs, and realistic presentation delivery via virtual characters. Central to EvoPresent is \textbf{PresAesth}, a multi-task reinforcement learning (RL) aesthetic model that provides reliable aesthetic scoring, defect adjustment, and comparative feedback, enabling iterative self-improvement even under limited aesthetic training data. To systematically evaluate the methods, we introduce \textbf{EvoPresent Benchmark}, a comprehensive benchmark comprising: \textit{Presentation Generation Quality}, built on 650 top-tier AI conference papers with multimodal resources (slides, videos and scripts) to assess both content and design; and \textit{Aesthetic Awareness}, consisting of 2,000 slide pairs with varying aesthetic levels, supporting joint training and evaluation on scoring, defect adjustment, and comparison. Our findings highlight that (i) High-quality feedback is essential for agent self-improvement, while initial capability alone does not guarantee effective self-correction. (ii) Automated generation pipelines exhibit a trade-off between visual design and content construction. (iii) Multi-task RL training shows stronger generalization in aesthetic awareness tasks.

**arXiv ID:** 2510.05571
</details>

<details>
<summary><strong>POLAR: Policy-based Layerwise Reinforcement Learning Method for Stealthy Backdoor Attacks in Federated Learning</strong> - Kuai Yu, Xiaoyu Wu, Peishen Yan, Qingqian Yang, Linshan Jiang, Hao Wang, Yang Hua, Tao Song, Haibing Guan - [[pdf]](https://arxiv.org/pdf/2510.19056)</summary>

**Abstract:** Federated Learning (FL) enables decentralized model training across multiple clients without exposing local data, but its distributed feature makes it vulnerable to backdoor attacks. Despite early FL backdoor attacks modifying entire models, recent studies have explored the concept of backdoor-critical (BC) layers, which poison the chosen influential layers to maintain stealthiness while achieving high effectiveness. However, existing BC layers approaches rely on rule-based selection without consideration of the interrelations between layers, making them ineffective and prone to detection by advanced defenses. In this paper, we propose POLAR (POlicy-based LAyerwise Reinforcement learning), the first pipeline to creatively adopt RL to solve the BC layer selection problem in layer-wise backdoor attack. Different from other commonly used RL paradigm, POLAR is lightweight with Bernoulli sampling. POLAR dynamically learns an attack strategy, optimizing layer selection using policy gradient updates based on backdoor success rate (BSR) improvements. To ensure stealthiness, we introduce a regularization constraint that limits the number of modified layers by penalizing large attack footprints. Extensive experiments demonstrate that POLAR outperforms the latest attack methods by up to 40% against six state-of-the-art (SOTA) defenses.

**arXiv ID:** 2510.19056
</details>

<details>
<summary><strong>Interpret Policies in Deep Reinforcement Learning using SILVER with RL-Guided Labeling: A Model-level Approach to High-dimensional and Multi-action Environments</strong> - Yiyu Qian, Su Nguyen, Chao Chen, Qinyue Zhou, Liyuan Zhao - [[pdf]](https://arxiv.org/pdf/2510.19244)</summary>

**Abstract:** Deep reinforcement learning (RL) achieves remarkable performance but lacks interpretability, limiting trust in policy behavior. The existing SILVER framework (Li, Siddique, and Cao 2025) explains RL policy via Shapley-based regression but remains restricted to low-dimensional, binary-action domains. We propose SILVER with RL-guided labeling, an enhanced variant that extends SILVER to multi-action and high-dimensional environments by incorporating the RL policy's own action outputs into the boundary points identification. Our method first extracts compact feature representations from image observations, performs SHAP-based feature attribution, and then employs RL-guided labeling to generate behaviorally consistent boundary datasets. Surrogate models, such as decision trees and regression-based functions, are subsequently trained to interpret RL policy's decision structure. We evaluate the proposed framework on two Atari environments using three deep RL algorithms and conduct human-subject study to assess the clarity and trustworthiness of the derived interpretable policy. Results show that our approach maintains competitive task performance while substantially improving transparency and human understanding of agent behavior. This work advances explainable RL by transforming SILVER into a scalable and behavior-aware framework for interpreting deep RL agents in high-dimensional, multi-action settings.

**arXiv ID:** 2510.19244
</details>

<details>
<summary><strong>RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs</strong> - Yongji Wu, Xueshen Liu, Haizhong Zheng, Juncheng Gu, Beidi Chen, Z. Morley Mao, Arvind Krishnamurthy, Ion Stoica - [[pdf]](https://arxiv.org/pdf/2510.19225)</summary>

**Abstract:** Reinforcement learning (RL) has become essential for unlocking advanced reasoning capabilities in large language models (LLMs). RL workflows involve interleaving rollout and training stages with fundamentally different resource requirements. Rollout typically dominates overall execution time, yet scales efficiently through multiple independent instances. In contrast, training requires tightly-coupled GPUs with full-mesh communication. Existing RL frameworks fall into two categories: co-located and disaggregated architectures. Co-located ones fail to address this resource tension by forcing both stages to share the same GPUs. Disaggregated architectures, without modifications of well-established RL algorithms, suffer from resource under-utilization. Meanwhile, preemptible GPU resources, i.e., spot instances on public clouds and spare capacity in production clusters, present significant cost-saving opportunities for accelerating RL workflows, if efficiently harvested for rollout.
In this paper, we present RLBoost, a systematic solution for cost-efficient RL training that harvests preemptible GPU resources. Our key insight is that rollout's stateless and embarrassingly parallel nature aligns perfectly with preemptible and often fragmented resources. To efficiently utilize these resources despite frequent and unpredictable availability changes, RLBoost adopts a hybrid architecture with three key techniques: (1) adaptive rollout offload to dynamically adjust workloads on the reserved (on-demand) cluster, (2) pull-based weight transfer that quickly provisions newly available instances, and (3) token-level response collection and migration for efficient preemption handling and continuous load balancing. Extensive experiments show RLBoost increases training throughput by 1.51x-1.97x while improving cost efficiency by 28%-49% compared to using only on-demand GPU resources.

**arXiv ID:** 2510.19225
</details>

<details>
<summary><strong>Hierarchical DLO Routing with Reinforcement Learning and In-Context Vision-language Models</strong> - Mingen Li, Houjian Yu, Yixuan Huang, Youngjin Hong, Changhyun Choi - [[pdf]](https://arxiv.org/pdf/2510.19268)</summary>

**Abstract:** Long-horizon routing tasks of deformable linear objects (DLOs), such as cables and ropes, are common in industrial assembly lines and everyday life. These tasks are particularly challenging because they require robots to manipulate DLO with long-horizon planning and reliable skill execution. Successfully completing such tasks demands adapting to their nonlinear dynamics, decomposing abstract routing goals, and generating multi-step plans composed of multiple skills, all of which require accurate high-level reasoning during execution. In this paper, we propose a fully autonomous hierarchical framework for solving challenging DLO routing tasks. Given an implicit or explicit routing goal expressed in language, our framework leverages vision-language models~(VLMs) for in-context high-level reasoning to synthesize feasible plans, which are then executed by low-level skills trained via reinforcement learning. To improve robustness in long horizons, we further introduce a failure recovery mechanism that reorients the DLO into insertion-feasible states. Our approach generalizes to diverse scenes involving object attributes, spatial descriptions, as well as implicit language commands. It outperforms the next best baseline method by nearly 50% and achieves an overall success rate of 92.5% across long-horizon routing scenarios.

**arXiv ID:** 2510.19268
</details>

<details>
<summary><strong>Semi-off-Policy Reinforcement Learning for Vision-Language Slow-Thinking Reasoning</strong> - Junhao Shen, Haiteng Zhao, Yuzhe Gu, Songyang Gao, Kuikun Liu, Haian Huang, Jianfei Gao, Dahua Lin, Wenwei Zhang, Kai Chen - [[pdf]](https://arxiv.org/pdf/2507.16814)</summary>

**Abstract:** Enhancing large vision-language models (LVLMs) with visual slow-thinking reasoning is crucial for solving complex multimodal tasks. However, since LVLMs are mainly trained with vision-language alignment, it is difficult to adopt on-policy reinforcement learning (RL) to develop the slow thinking ability because the rollout space is restricted by its initial abilities. Off-policy RL offers a way to go beyond the current policy, but directly distilling trajectories from external models may cause visual hallucinations due to mismatched visual perception abilities across models. To address these issues, this paper proposes SOPHIA, a simple and scalable Semi-Off-Policy RL for vision-language slow-tHInking reAsoning. SOPHIA builds a semi-off-policy behavior model by combining on-policy visual understanding from a trainable LVLM with off-policy slow-thinking reasoning from a language model, assigns outcome-based rewards to reasoning, and propagates visual rewards backward. Then LVLM learns slow-thinking reasoning ability from the obtained reasoning trajectories using propagated rewards via off-policy RL algorithms. Extensive experiments with InternVL2.5 and InternVL3.0 with 8B and 38B sizes show the effectiveness of SOPHIA. Notably, SOPHIA improves InternVL3.0-38B by 8.50% in average, reaching state-of-the-art performance among open-source LVLMs on multiple multimodal reasoning benchmarks, and even outperforms some closed-source models (e.g., GPT-4.1) on the challenging MathVision and OlympiadBench, achieving 49.08% and 49.95% pass@1 accuracy, respectively. Analysis shows SOPHIA outperforms supervised fine-tuning and direct on-policy RL methods, offering a better policy initialization for further on-policy training.

**arXiv ID:** 2507.16814
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
