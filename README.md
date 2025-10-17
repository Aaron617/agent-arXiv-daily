# Agent arXiv Daily

**Last Updated:** 2025-10-17 02:40:31

**Total Papers:** 88

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (4 papers)</h2></summary>

<details>
<summary><strong>Formalizing the Safety, Security, and Functional Properties of Agentic AI Systems</strong> - Edoardo Allegrini, Ananth Shreekumar, Z. Berkay Celik - [[pdf]](https://arxiv.org/pdf/2510.14133)</summary>

**Abstract:** Agentic AI systems, which leverage multiple autonomous agents and Large Language Models (LLMs), are increasingly used to address complex, multi-step tasks. The safety, security, and functionality of these systems are critical, especially in high-stakes applications. However, the current ecosystem of inter-agent communication is fragmented, with protocols such as the Model Context Protocol (MCP) for tool access and the Agent-to-Agent (A2A) protocol for coordination being analyzed in isolation. This fragmentation creates a semantic gap that prevents the rigorous analysis of system properties and introduces risks such as architectural misalignment and exploitable coordination issues. To address these challenges, we introduce a modeling framework for agentic AI systems composed of two foundational models. The first, the host agent model, formalizes the top-level entity that interacts with the user, decomposes tasks, and orchestrates their execution by leveraging external agents and tools. The second, the task lifecycle model, details the states and transitions of individual sub-tasks from creation to completion, providing a fine-grained view of task management and error handling. Together, these models provide a unified semantic framework for reasoning about the behavior of multi-AI agent systems. Grounded in this framework, we define 17 properties for the host agent and 14 for the task lifecycle, categorized into liveness, safety, completeness, and fairness. Expressed in temporal logic, these properties enable formal verification of system behavior, detection of coordination edge cases, and prevention of deadlocks and security vulnerabilities. Through this effort, we introduce the first rigorously grounded, domain-agnostic framework for the systematic analysis, design, and deployment of correct, reliable, and robust agentic AI systems.

**arXiv ID:** 2510.14133
</details>

<details>
<summary><strong>From Craft to Constitution: A Governance-First Paradigm for Principled Agent Engineering</strong> - Qiang Xu, Xiangyu Wen, Changran Xu, Zeju Li, Jianyuan Zhong - [[pdf]](https://arxiv.org/pdf/2510.13857)</summary>

**Abstract:** The advent of powerful Large Language Models (LLMs) has ushered in an ``Age of the Agent,'' enabling autonomous systems to tackle complex goals. However, the transition from prototype to production is hindered by a pervasive ``crisis of craft,'' resulting in agents that are brittle, unpredictable, and ultimately untrustworthy in mission-critical applications. This paper argues this crisis stems from a fundamental paradigm mismatch -- attempting to command inherently probabilistic processors with the deterministic mental models of traditional software engineering. To solve this crisis, we introduce a governance-first paradigm for principled agent engineering, embodied in a formal architecture we call ArbiterOS.

**arXiv ID:** 2510.13857
</details>

<details>
<summary><strong>Internet of Agents: Fundamentals, Applications, and Challenges</strong> - Yuntao Wang, Shaolong Guo, Yanghe Pan, Zhou Su, Fahao Chen, Tom H. Luan, Peng Li, Jiawen Kang, Dusit Niyato - [[pdf]](https://arxiv.org/pdf/2505.07176)</summary>

**Abstract:** With the rapid proliferation of large language models and vision-language models, AI agents have evolved from isolated, task-specific systems into autonomous, interactive entities capable of perceiving, reasoning, and acting without human intervention. As these agents proliferate across virtual and physical environments, from virtual assistants to embodied robots, the need for a unified, agent-centric infrastructure becomes paramount. In this survey, we introduce the Internet of Agents (IoA) as a foundational framework that enables seamless interconnection, dynamic discovery, and collaborative orchestration among heterogeneous agents at scale. We begin by presenting a general IoA architecture, highlighting its hierarchical organization, distinguishing features relative to the traditional Internet, and emerging applications. Next, we analyze the key operational enablers of IoA, including capability notification and discovery, adaptive communication protocols, dynamic task matching, consensus and conflict-resolution mechanisms, and incentive models. Finally, we identify open research directions toward building resilient and trustworthy IoA ecosystems.

**arXiv ID:** 2505.07176
</details>

<details>
<summary><strong>The Pursuit of Diversity: Multi-Objective Testing of Deep Reinforcement Learning Agents</strong> - Antony Bartlett, Cynthia Liem, Annibale Panichella - [[pdf]](https://arxiv.org/pdf/2510.14727)</summary>

**Abstract:** Testing deep reinforcement learning (DRL) agents in safety-critical domains requires discovering diverse failure scenarios. Existing tools such as INDAGO rely on single-objective optimization focused solely on maximizing failure counts, but this does not ensure discovered scenarios are diverse or reveal distinct error types. We introduce INDAGO-Nexus, a multi-objective search approach that jointly optimizes for failure likelihood and test scenario diversity using multi-objective evolutionary algorithms with multiple diversity metrics and Pareto front selection strategies. We evaluated INDAGO-Nexus on three DRL agents: humanoid walker, self-driving car, and parking agent. On average, INDAGO-Nexus discovers up to 83% and 40% more unique failures (test effectiveness) than INDAGO in the SDC and Parking scenarios, respectively, while reducing time-to-failure by up to 67% across all agents.

**arXiv ID:** 2510.14727
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (10 papers)</h2></summary>

<details>
<summary><strong>CodeEvolve: An open source evolutionary coding agent for algorithm discovery and optimization</strong> - Henrique Assumpção, Diego Ferreira, Leandro Campos, Fabricio Murai - [[pdf]](https://arxiv.org/pdf/2510.14150)</summary>

**Abstract:** In this work, we introduce CodeEvolve, an open-source evolutionary coding agent that unites Large Language Models (LLMs) with genetic algorithms to solve complex computational problems. Our framework adapts powerful evolutionary concepts to the LLM domain, building upon recent methods for generalized scientific discovery. CodeEvolve employs an island-based genetic algorithm to maintain population diversity and increase throughput, introduces a novel inspiration-based crossover mechanism that leverages the LLMs context window to combine features from successful solutions, and implements meta-prompting strategies for dynamic exploration of the solution space. We conduct a rigorous evaluation of CodeEvolve on a subset of the mathematical benchmarks used to evaluate Google DeepMind's closed-source AlphaEvolve. Our findings show that our method surpasses AlphaEvolve's performance on several challenging problems. To foster collaboration and accelerate progress, we release our complete framework as an open-source repository.

**arXiv ID:** 2510.14150
</details>

<details>
<summary><strong>Synthesizing Agentic Data for Web Agents with Progressive Difficulty Enhancement Mechanisms</strong> - Shrey Pandit, Xuan-Phi Nguyen, Yifei Ming, Austin Xu, Jiayu Wang, Caiming Xiong, Shafiq Joty - [[pdf]](https://arxiv.org/pdf/2510.13913)</summary>

**Abstract:** Web-based 'deep research' agents aim to solve complex question - answering tasks through long-horizon interactions with online tools. These tasks remain challenging, as the underlying language models are often not optimized for long-horizon reasoning and exploration. Prior work has proposed workflows for constructing instruction-tuning datasets, often leveraging knowledge graphs. However, such methods typically lack fine-grained control over difficulty and quality, yielding synthetic data that falls short of capturing the complexity required for long-horizon reasoning. Furthermore, many studies conflate data and training effects by comparing models trained under different optimization recipes, making it difficult to isolate and evaluate the effectiveness of the data itself. We introduce a two-pronged data synthesis pipeline that generates question - answer pairs by progressively increasing task complexity until a frontier baseline web agent fails. The baseline agent plays multiple roles in this process: attempting the questions, validating factuality, checking for alternative answers, and enforcing filtering. To evaluate the effectiveness of our synthesis methods, we adopt a controlled training setup based on distillation from strong web agents. Experiments across multiple web-based benchmarks show that our dataset - despite being smaller - enables the training of more effective web agents than existing datasets. In particular, our data exhibits twice the diversity in tool-use actions, allowing models trained on it to achieve stronger performance while avoiding repetitive tool-calling behaviors.

**arXiv ID:** 2510.13913
</details>

<details>
<summary><strong>When Planners Meet Reality: How Learned, Reactive Traffic Agents Shift nuPlan Benchmarks</strong> - Steffen Hagedorn, Luka Donkov, Aron Distelzweig, Alexandru P. Condurache - [[pdf]](https://arxiv.org/pdf/2510.14677)</summary>

**Abstract:** Planner evaluation in closed-loop simulation often uses rule-based traffic agents, whose simplistic and passive behavior can hide planner deficiencies and bias rankings. Widely used IDM agents simply follow a lead vehicle and cannot react to vehicles in adjacent lanes, hindering tests of complex interaction capabilities. We address this issue by integrating the state-of-the-art learned traffic agent model SMART into nuPlan. Thus, we are the first to evaluate planners under more realistic conditions and quantify how conclusions shift when narrowing the sim-to-real gap. Our analysis covers 14 recent planners and established baselines and shows that IDM-based simulation overestimates planning performance: nearly all scores deteriorate. In contrast, many planners interact better than previously assumed and even improve in multi-lane, interaction-heavy scenarios like lane changes or turns. Methods trained in closed-loop demonstrate the best and most stable driving performance. However, when reaching their limits in augmented edge-case scenarios, all learned planners degrade abruptly, whereas rule-based planners maintain reasonable basic behavior. Based on our results, we suggest SMART-reactive simulation as a new standard closed-loop benchmark in nuPlan and release the SMART agents as a drop-in alternative to IDM at this https URL.

**arXiv ID:** 2510.14677
</details>

<details>
<summary><strong>ABMax: A JAX-based Agent-based Modeling Framework</strong> - Siddharth Chaturvedi, Ahmed El-Gazzar, Marcel van Gerven - [[pdf]](https://arxiv.org/pdf/2508.16508)</summary>

**Abstract:** Agent-based modeling (ABM) is a principal approach for studying complex systems. By decomposing a system into simpler, interacting agents, agent-based modeling (ABM) allows researchers to observe the emergence of complex phenomena. High-performance array computing libraries like JAX can help scale such computational models to a large number of agents by using automatic vectorization and just-in-time (JIT) compilation. One of the caveats of using JAX to achieve such scaling is that the shapes of arrays used in the computational model should remain immutable throughout the simulation. In the context of agent-based modeling (ABM), this can pose constraints on certain agent manipulation operations that require flexible data structures. A subset of which is represented by the ability to update a dynamically selected number of agents by applying distinct changes to them during a simulation. To this effect, we introduce ABMax, an ABM framework based on JAX that implements multiple just-in-time (JIT) compilable algorithms to provide this functionality. On the canonical predation model benchmark, ABMax achieves runtime performance comparable to state-of-the-art implementations. Further, we show that this functionality can also be vectorized, making it possible to run many similar agent-based models in parallel. We also present two examples in the form of a traffic-flow model and a financial market model to show the use case of ABMax

**arXiv ID:** 2508.16508
</details>

<details>
<summary><strong>RAGCap-Bench: Benchmarking Capabilities of LLMs in Agentic Retrieval Augmented Generation Systems</strong> - Jingru Lin, Chen Zhang, Stephen Y. Liu, Haizhou Li - [[pdf]](https://arxiv.org/pdf/2510.13910)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) mitigates key limitations of Large Language Models (LLMs)-such as factual errors, outdated knowledge, and hallucinations-by dynamically retrieving external information. Recent work extends this paradigm through agentic RAG systems, where LLMs act as agents to iteratively plan, retrieve, and reason over complex queries. However, these systems still struggle with challenging multi-hop questions, and their intermediate reasoning capabilities remain underexplored. To address this, we propose RAGCap-Bench, a capability-oriented benchmark for fine-grained evaluation of intermediate tasks in agentic RAG workflows. We analyze outputs from state-of-the-art systems to identify common tasks and the core capabilities required for their execution, then construct a taxonomy of typical LLM errors to design targeted evaluation questions. Experiments show that "slow-thinking" models with stronger RAGCap performance achieve better end-to-end results, underscoring the benchmark's validity and the importance of enhancing these intermediate capabilities.

**arXiv ID:** 2510.13910
</details>

<details>
<summary><strong>FACTS: Table Summarization via Offline Template Generation with Agentic Workflows</strong> - Ye Yuan, Mohammad Amin Shabani, Siqi Liu - [[pdf]](https://arxiv.org/pdf/2510.13920)</summary>

**Abstract:** Query-focused table summarization requires generating natural language summaries of tabular data conditioned on a user query, enabling users to access insights beyond fact retrieval. Existing approaches face key limitations: table-to-text models require costly fine-tuning and struggle with complex reasoning, prompt-based LLM methods suffer from token-limit and efficiency issues while exposing sensitive data, and prior agentic pipelines often rely on decomposition, planning, or manual templates that lack robustness and scalability. To mitigate these issues, we introduce an agentic workflow, FACTS, a Fast, Accurate, and Privacy-Compliant Table Summarization approach via Offline Template Generation. FACTS produces offline templates, consisting of SQL queries and Jinja2 templates, which can be rendered into natural language summaries and are reusable across multiple tables sharing the same schema. It enables fast summarization through reusable offline templates, accurate outputs with executable SQL queries, and privacy compliance by sending only table schemas to LLMs. Evaluations on widely-used benchmarks show that FACTS consistently outperforms baseline methods, establishing it as a practical solution for real-world query-focused table summarization.

**arXiv ID:** 2510.13920
</details>

<details>
<summary><strong>An LLM-Powered AI Agent Framework for Holistic IoT Traffic Interpretation</strong> - Daniel Adu Worae, Spyridon Mastorakis - [[pdf]](https://arxiv.org/pdf/2510.13925)</summary>

**Abstract:** Internet of Things (IoT) networks generate diverse and high-volume traffic that reflects both normal activity and potential threats. Deriving meaningful insight from such telemetry requires cross-layer interpretation of behaviors, protocols, and context rather than isolated detection. This work presents an LLM-powered AI agent framework that converts raw packet captures into structured and semantically enriched representations for interactive analysis. The framework integrates feature extraction, transformer-based anomaly detection, packet and flow summarization, threat intelligence enrichment, and retrieval-augmented question answering. An AI agent guided by a large language model performs reasoning over the indexed traffic artifacts, assembling evidence to produce accurate and human-readable interpretations. Experimental evaluation on multiple IoT captures and six open models shows that hybrid retrieval, which combines lexical and semantic search with reranking, substantially improves BLEU, ROUGE, METEOR, and BERTScore results compared with dense-only retrieval. System profiling further indicates low CPU, GPU, and memory overhead, demonstrating that the framework achieves holistic and efficient interpretation of IoT network traffic.

**arXiv ID:** 2510.13925
</details>

<details>
<summary><strong>FinDeepResearch: Evaluating Deep Research Agents in Rigorous Financial Analysis</strong> - Fengbin Zhu, Xiang Yao Ng, Ziyang Liu, Chang Liu, Xianwei Zeng, Chao Wang, Tianhui Tan, Xuan Yao, Pengyang Shao, Min Xu, Zixuan Wang, Jing Wang, Xin Lin, Junfeng Li, Jingxian Zhu, Yang Zhang, Wenjie Wang, Fuli Feng, Richang Hong, Huanbo Luan, Ke-Wei Huang, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2510.13936)</summary>

**Abstract:** Deep Research (DR) agents, powered by advanced Large Language Models (LLMs), have recently garnered increasing attention for their capability in conducting complex research tasks. However, existing literature lacks a rigorous and systematic evaluation of DR Agent's capabilities in critical research analysis. To address this gap, we first propose HisRubric, a novel evaluation framework with a hierarchical analytical structure and a fine-grained grading rubric for rigorously assessing DR agents' capabilities in corporate financial analysis. This framework mirrors the professional analyst's workflow, progressing from data recognition to metric calculation, and finally to strategic summarization and interpretation. Built on this framework, we construct a FinDeepResearch benchmark that comprises 64 listed companies from 8 financial markets across 4 languages, encompassing a total of 15,808 grading items. We further conduct extensive experiments on the FinDeepResearch using 16 representative methods, including 6 DR agents, 5 LLMs equipped with both deep reasoning and search capabilities, and 5 LLMs with deep reasoning capabilities only. The results reveal the strengths and limitations of these approaches across diverse capabilities, financial markets, and languages, offering valuable insights for future research and development. The benchmark and evaluation code will be made publicly available.

**arXiv ID:** 2510.13936
</details>

<details>
<summary><strong>Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities</strong> - Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, Krishna Haridasan, Ahmed Omran, Nikunj Saunshi, Dara Bahri, Gaurav Mishra, Eric Chu, Toby Boyd, Brad Hekman, Aaron Parisi, Chaoyi Zhang, Kornraphop Kawintiranon, Tania Bedrax-Weiss, Oliver Wang, Ya Xu, Ollie Purkiss, Uri Mendlovic, Ilaï Deutel, Nam Nguyen, Adam Langley, Flip Korn, Lucia Rossazza, Alexandre Ramé, Sagar Waghmare, Helen Miller, Nathan Byrd, Ashrith Sheshan, Raia Hadsell, Sangnie Bhardwaj, Pawel Janus, Tero Rissa, Dan Horgan, Alvin Abdagic, Lior Belenki, James Allingham, Anima Singh, Theo Guidroz, Srivatsan Srinivasan, Herman Schmit, Kristen Chiafullo, Andre Elisseeff, Nilpa Jha, Prateek Kolhar, Leonard Berrada, Frank Ding, Xiance Si, Shrestha Basu Mallick, Franz Och, Sofia Erell, Eric Ni, Tejasi Latkar, Sherry Yang, Petar Sirkovic, Ziqiang Feng, Robert Leland, Rachel Hornung, Gang Wu, Charles Blundell, Hamidreza Alvari, Po-Sen Huang, Cathy Yip, Sanja Deur, Li Liu, Gabriela Surita, Pablo Duque, Dima Damen, Johnson Jia, Arthur Guez, Markus Mircea, Animesh Sinha, Alberto Magni, Paweł Stradomski, Tal Marian, Vlado Galić, Wenhu Chen, Hisham Husain, Achintya Singhal, Dominik Grewe, François-Xavier Aubet, Shuang Song, Lorenzo Blanco, Leland Rechis - [[pdf]](https://arxiv.org/pdf/2507.06261)</summary>

**Abstract:** In this report, we introduce the Gemini 2.X model family: Gemini 2.5 Pro and Gemini 2.5 Flash, as well as our earlier Gemini 2.0 Flash and Flash-Lite models. Gemini 2.5 Pro is our most capable model yet, achieving SoTA performance on frontier coding and reasoning benchmarks. In addition to its incredible coding and reasoning skills, Gemini 2.5 Pro is a thinking model that excels at multimodal understanding and it is now able to process up to 3 hours of video content. Its unique combination of long context, multimodal and reasoning capabilities can be combined to unlock new agentic workflows. Gemini 2.5 Flash provides excellent reasoning abilities at a fraction of the compute and latency requirements and Gemini 2.0 Flash and Flash-Lite provide high performance at low latency and cost. Taken together, the Gemini 2.X model generation spans the full Pareto frontier of model capability vs cost, allowing users to explore the boundaries of what is possible with complex agentic problem solving.

**arXiv ID:** 2507.06261
</details>

<details>
<summary><strong>VLA^2: Empowering Vision-Language-Action Models with an Agentic Framework for Unseen Concept Manipulation</strong> - Han Zhao, Jiaxuan Zhang, Wenxuan Song, Pengxiang Ding, Donglin Wang - [[pdf]](https://arxiv.org/pdf/2510.14902)</summary>

**Abstract:** Current vision-language-action (VLA) models, pre-trained on large-scale robotic data, exhibit strong multi-task capabilities and generalize well to variations in visual and language instructions for manipulation. However, their success rate drops significantly when faced with object concepts outside the training data, such as unseen object descriptions and textures in the dataset. To address this, we propose a novel agentic framework, VLA^2, which leverages OpenVLA as the execution backbone and effectively leverages external modules such as web retrieval and object detection to provide visual and textual knowledge about target objects to the VLA. This approach mitigates generalization failure when handling out-of-distribution objects. Based on the LIBERO simulation environment, we introduced novel objects and object descriptions to construct a new evaluation benchmark with three difficulty levels to test the effectiveness of our method. Our framework successfully outperformed the current state-of-the-art models on our designed hard-level generalization benchmark. Compared to the standalone OpenVLA baseline, VLA^2 achieves a 44.2% improvement in the success rate in the hard-level benchmark and an average improvement of 20.2% in all customized environments without any performance degradation on in-domain tasks. Project website: this https URL.

**arXiv ID:** 2510.14902
</details>

</details>

<details open>
<summary><h2>LLM Agents (6 papers)</h2></summary>

<details>
<summary><strong>LLM Agents Beyond Utility: An Open-Ended Perspective</strong> - Asen Nachkov, Xi Wang, Luc Van Gool - [[pdf]](https://arxiv.org/pdf/2510.14548)</summary>

**Abstract:** Recent LLM agents have made great use of chain of thought reasoning and function calling. As their capabilities grow, an important question arises: can this software represent not only a smart problem-solving tool, but an entity in its own right, that can plan, design immediate tasks, and reason toward broader, more ambiguous goals? To study this question, we adopt an open-ended experimental setting where we augment a pretrained LLM agent with the ability to generate its own tasks, accumulate knowledge, and interact extensively with its environment. We study the resulting open-ended agent qualitatively. It can reliably follow complex multi-step instructions, store and reuse information across runs, and propose and solve its own tasks, though it remains sensitive to prompt design, prone to repetitive task generation, and unable to form self-representations. These findings illustrate both the promise and current limits of adapting pretrained LLMs toward open-endedness, and point to future directions for training agents to manage memory, explore productively, and pursue abstract long-term goals.

**arXiv ID:** 2510.14548
</details>

<details>
<summary><strong>Where to Search: Measure the Prior-Structured Search Space of LLM Agents</strong> - Zhuo-Yang Song - [[pdf]](https://arxiv.org/pdf/2510.14846)</summary>

**Abstract:** The generate-filter-refine (iterative paradigm) based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs.

**arXiv ID:** 2510.14846
</details>

<details>
<summary><strong>Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents</strong> - Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng, Xiaofeng Wu, Zhenzhe Ying - [[pdf]](https://arxiv.org/pdf/2510.14967)</summary>

**Abstract:** Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided at the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate two critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals, and (ii) lack of fine-grained credit assignment, where dependencies between turns are obscured, especially in long-horizon tasks. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward trajectories. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved sample efficiency.

**arXiv ID:** 2510.14967
</details>

<details>
<summary><strong>EMAC+: Embodied Multimodal Agent for Collaborative Planning with VLM+LLM</strong> - Shuang Ao, Flora D. Salim, Simon Khan - [[pdf]](https://arxiv.org/pdf/2505.19905)</summary>

**Abstract:** Although LLMs demonstrate proficiency in several text-based reasoning and planning tasks, their implementation in robotics control is constrained by significant deficiencies: (1) LLM agents are designed to work mainly with textual inputs rather than visual conditions; (2) Current multimodal agents treat LLMs as static planners, which separates their reasoning from environment dynamics, resulting in actions that do not take domain-specific knowledge into account; and (3) LLMs are not designed to learn from visual interactions, which makes it harder for them to make better policies for specific domains. In this paper, we introduce EMAC+, an Embodied Multimodal Agent that collaboratively integrates LLM and VLM via a bidirectional training paradigm. Unlike existing methods, EMAC+ dynamically refines high-level textual plans generated by an LLM using real-time feedback from a VLM executing low-level visual control tasks. We address critical limitations of previous models by enabling the LLM to internalize visual environment dynamics directly through interactive experience, rather than relying solely on static symbolic mappings. Extensive experimental evaluations on ALFWorld and RT-1 benchmarks demonstrate that EMAC+ achieves superior task performance, robustness against noisy observations, and efficient learning. We also conduct thorough ablation studies and provide detailed analyses of success and failure cases.

**arXiv ID:** 2505.19905
</details>

<details>
<summary><strong>Large Language Model Agents Enable Autonomous Design and Image Analysis of Microwell Microfluidics</strong> - Dinh-Nguyen Nguyen, Sadia Shakil, Raymond Kai-Yu Tong, Ngoc-Duy Dinh - [[pdf]](https://arxiv.org/pdf/2510.13883)</summary>

**Abstract:** Microwell microfluidics has been utilized for single-cell analysis to reveal heterogeneity in gene expression, signaling pathways, and phenotypic responses for identifying rare cell types, understanding disease progression, and developing more precise therapeutic strategies. However, designing microwell microfluidics is a considerably complex task, requiring knowledge, experience, and CAD software, as well as manual intervention, which often fails initial designs, demanding multiple costly and time-consuming iterations. In this study, we establish an autonomous large language model (LLM)-driven microwell design framework to generate code-based computer-aided design (CAD) scripts, that enables the rapid and reproducible creation of microwells with diverse geometries and imaging-based analysis. We propose a multimodal large language model (MLLM)-logistic regression framework based on integrating high-level semantic descriptions generated by MLLMs with image embeddings for image classification tasks, aiming to identify microwell occupancy and microwell shape. The fused multimodal representation is input to a logistic regression model, which is both interpretable and computationally efficient. We achieved significant improvements, exceeding 0.92 for occupancy classification and 0.99 for shape classification, across all evaluated MLLMs, compared with 0.50 and 0.55, respectively, when relying solely on direct classification. The MLLM-logistic regression framework is a scalable, efficient solution for high-throughput microwell image analysis. Our study demonstrates an autonomous design microwell platform by translating natural language prompts into optimized device geometries, CAD scripts and image analysis, facilitating the development of next-generation digital discovery by integration of literature mining, autonomous design and experimental data analysis.

**arXiv ID:** 2510.13883
</details>

<details>
<summary><strong>WoW: Towards a World omniscient World model Through Embodied Interaction</strong> - Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, Weishi Mi, Kevin Zhang, Zhiyuan Qin, Wanxin Tian, Kuangzhi Ge, Hao Li, Zezhong Qian, Anthony Chen, Qiang Zhou, Yueru Jia, Jiaming Liu, Yong Dai, Qingpo Wuwu, Chengyu Bai, Yu-Kai Wang, Ying Li, Lizhang Chen, Yong Bao, Zhiyuan Jiang, Jiacheng Zhu, Kai Tang, Ruichuan An, Yulin Luo, Qiuxuan Feng, Siyuan Zhou, Chi-min Chan, Chengkai Hou, Wei Xue, Sirui Han, Yike Guo, Shanghang Zhang, Jian Tang - [[pdf]](https://arxiv.org/pdf/2509.22642)</summary>

**Abstract:** Humans develop an understanding of intuitive physics through active interaction with the world. This approach is in stark contrast to current video models, such as Sora, which rely on passive observation and therefore struggle with grasping physical causality. This observation leads to our central hypothesis: authentic physical intuition of the world model must be grounded in extensive, causally rich interactions with the real world. To test this hypothesis, we present WoW, a 14-billion-parameter generative world model trained on 2 million robot interaction trajectories. Our findings reveal that the model's understanding of physics is a probabilistic distribution of plausible outcomes, leading to stochastic instabilities and physical hallucinations. Furthermore, we demonstrate that this emergent capability can be actively constrained toward physical realism by SOPHIA, where vision-language model agents evaluate the DiT-generated output and guide its refinement by iteratively evolving the language instructions. In addition, a co-trained Inverse Dynamics Model translates these refined plans into executable robotic actions, thus closing the imagination-to-action loop. We establish WoWBench, a new benchmark focused on physical consistency and causal reasoning in video, where WoW achieves state-of-the-art performance in both human and autonomous evaluation, demonstrating strong ability in physical causality, collision dynamics, and object permanence. Our work provides systematic evidence that large-scale, real-world interaction is a cornerstone for developing physical intuition in AI. Models, data, and benchmarks will be open-sourced.

**arXiv ID:** 2509.22642
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (21 papers)</h2></summary>

<details>
<summary><strong>STEMS: Spatial-Temporal Enhanced Safe Multi-Agent Coordination for Building Energy Management</strong> - Huiliang Zhang, Di Wu, Arnaud Zinflou, Benoit Boulet - [[pdf]](https://arxiv.org/pdf/2510.14112)</summary>

**Abstract:** Building energy management is essential for achieving carbon reduction goals, improving occupant comfort, and reducing energy costs. Coordinated building energy management faces critical challenges in exploiting spatial-temporal dependencies while ensuring operational safety across multi-building systems. Current multi-building energy systems face three key challenges: insufficient spatial-temporal information exploitation, lack of rigorous safety guarantees, and system complexity. This paper proposes Spatial-Temporal Enhanced Safe Multi-Agent Coordination (STEMS), a novel safety-constrained multi-agent reinforcement learning framework for coordinated building energy management. STEMS integrates two core components: (1) a spatial-temporal graph representation learning framework using a GCN-Transformer fusion architecture to capture inter-building relationships and temporal patterns, and (2) a safety-constrained multi-agent RL algorithm incorporating Control Barrier Functions to provide mathematical safety guarantees. Extensive experiments on real-world building datasets demonstrate STEMS's superior performance over existing methods, showing that STEMS achieves 21% cost reduction, 18% emission reduction, and dramatically reduces safety violations from 35.1% to 5.6% while maintaining optimal comfort with only 0.13 discomfort proportion. The framework also demonstrates strong robustness during extreme weather conditions and maintains effectiveness across different building types.

**arXiv ID:** 2510.14112
</details>

<details>
<summary><strong>Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks</strong> - Trilok Padhi, Pinxian Lu, Abdulkadir Erol, Tanmay Sutar, Gauri Sharma, Mina Sonmez, Munmun De Choudhury, Ugur Kursuncu - [[pdf]](https://arxiv.org/pdf/2510.14207)</summary>

**Abstract:** Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.

**arXiv ID:** 2510.14207
</details>

<details>
<summary><strong>Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies</strong> - Mason Nakamura, Abhinav Kumar, Saaduddin Mahmud, Sahar Abdelnabi, Shlomo Zilberstein, Eugene Bagdasarian - [[pdf]](https://arxiv.org/pdf/2510.14312)</summary>

**Abstract:** A multi-agent system (MAS) powered by large language models (LLMs) can automate tedious user tasks such as meeting scheduling that requires inter-agent collaboration. LLMs enable nuanced protocols that account for unstructured private data, user constraints, and preferences. However, this design introduces new risks, including misalignment and attacks by malicious parties that compromise agents or steal user data. In this paper, we propose the Terrarium framework for fine-grained study on safety, privacy, and security in LLM-based MAS. We repurpose the blackboard design, an early approach in multi-agent systems, to create a modular, configurable testbed for multi-agent collaboration. We identify key attack vectors such as misalignment, malicious agents, compromised communication, and data poisoning. We implement three collaborative MAS scenarios with four representative attacks to demonstrate the framework's flexibility. By providing tools to rapidly prototype, evaluate, and iterate on defenses and designs, Terrarium aims to accelerate progress toward trustworthy multi-agent systems.

**arXiv ID:** 2510.14312
</details>

<details>
<summary><strong>Metacognitive Self-Correction for Multi-Agent System via Prototype-Guided Next-Execution Reconstruction</strong> - Xu Shen, Qi Zhang, Song Wang, Zhen Tan, Xinyu Zhao, Laura Yao, Vaishnav Tadiparthi, Hossein Nourkhiz Mahjoub, Ehsan Moradi Pari, Kwonjoon Lee, Tianlong Chen - [[pdf]](https://arxiv.org/pdf/2510.14319)</summary>

**Abstract:** Large Language Model based multi-agent systems (MAS) excel at collaborative problem solving but remain brittle to cascading errors: a single faulty step can propagate across agents and disrupt the trajectory. In this paper, we present MASC, a metacognitive framework that endows MAS with real-time, unsupervised, step-level error detection and self-correction. MASC rethinks detection as history-conditioned anomaly scoring via two complementary designs: (1) Next-Execution Reconstruction, which predicts the embedding of the next step from the query and interaction history to capture causal consistency, and (2) Prototype-Guided Enhancement, which learns a prototype prior over normal-step embeddings and uses it to stabilize reconstruction and anomaly scoring under sparse context (e.g., early steps). When an anomaly step is flagged, MASC triggers a correction agent to revise the acting agent's output before information flows downstream. On the Who&When benchmark, MASC consistently outperforms all baselines, improving step-level error detection by up to 8.47% AUC-ROC ; When plugged into diverse MAS frameworks, it delivers consistent end-to-end gains across architectures, confirming that our metacognitive monitoring and targeted correction can mitigate error propagation with minimal overhead.

**arXiv ID:** 2510.14319
</details>

<details>
<summary><strong>IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning</strong> - Xikai Zhang, Bo Wang, Likang Xiao, Yongzhi Li, Quan Chen, Wenju Wu, Liu Liu - [[pdf]](https://arxiv.org/pdf/2510.14406)</summary>

**Abstract:** Although large language models (LLMs) have made significant strides across various tasks, they still face significant challenges in complex reasoning and planning. For example, even with carefully designed prompts and prior information explicitly provided, GPT-4o achieves only a 7% Final Pass Rate on the TravelPlanner dataset in the sole-planning mode. Similarly, even in the thinking mode, Qwen3-8B-Instruct and DeepSeek-R1-671B, only achieve Final Pass Rates of 5.9% and 40%, respectively. Although well-organized Multi-Agent Systems (MAS) can offer improved collective reasoning, they often suffer from high reasoning costs due to multi-round internal interactions, long per-response latency, and difficulties in end-to-end training. To address these challenges, we propose a general and scalable framework called IMAGINE, short for Integrating Multi-Agent System into One Model. This framework not only integrates the reasoning and planning capabilities of MAS into a single, compact model, but also significantly surpass the capabilities of the MAS through a simple end-to-end training. Through this pipeline, a single small-scale model is not only able to acquire the structured reasoning and planning capabilities of a well-organized MAS but can also significantly outperform it. Experimental results demonstrate that, when using Qwen3-8B-Instruct as the base model and training it with our method, the model achieves an 82.7% Final Pass Rate on the TravelPlanner benchmark, far exceeding the 40% of DeepSeek-R1-671B, while maintaining a much smaller model size.

**arXiv ID:** 2510.14406
</details>

<details>
<summary><strong>Helmsman: Autonomous Synthesis of Federated Learning Systems via Multi-Agent Collaboration</strong> - Haoyuan Li, Mathias Funk, Aaqib Saeed - [[pdf]](https://arxiv.org/pdf/2510.14512)</summary>

**Abstract:** Federated Learning (FL) offers a powerful paradigm for training models on decentralized data, but its promise is often undermined by the immense complexity of designing and deploying robust systems. The need to select, combine, and tune strategies for multifaceted challenges like data heterogeneity and system constraints has become a critical bottleneck, resulting in brittle, bespoke solutions. To address this, we introduce Helmsman, a novel multi-agent system that automates the end-to-end synthesis of federated learning systems from high-level user specifications. It emulates a principled research and development workflow through three collaborative phases: (1) interactive human-in-the-loop planning to formulate a sound research plan, (2) modular code generation by supervised agent teams, and (3) a closed-loop of autonomous evaluation and refinement in a sandboxed simulation environment. To facilitate rigorous evaluation, we also introduce AgentFL-Bench, a new benchmark comprising 16 diverse tasks designed to assess the system-level generation capabilities of agentic systems in FL. Extensive experiments demonstrate that our approach generates solutions competitive with, and often superior to, established hand-crafted baselines. Our work represents a significant step towards the automated engineering of complex decentralized AI systems.

**arXiv ID:** 2510.14512
</details>

<details>
<summary><strong>GenCellAgent: Generalizable, Training-Free Cellular Image Segmentation via Large Language Model Agents</strong> - Xi Yu, Yang Yang, Qun Liu, Yonghua Du, Sean McSweeney, Yuewei Lin - [[pdf]](https://arxiv.org/pdf/2510.13896)</summary>

**Abstract:** Cellular image segmentation is essential for quantitative biology yet remains difficult due to heterogeneous modalities, morphological variability, and limited annotations. We present GenCellAgent, a training-free multi-agent framework that orchestrates specialist segmenters and generalist vision-language models via a planner-executor-evaluator loop (choose tool $\rightarrow$ run $\rightarrow$ quality-check) with long-term memory. The system (i) automatically routes images to the best tool, (ii) adapts on the fly using a few reference images when imaging conditions differ from what a tool expects, (iii) supports text-guided segmentation of organelles not covered by existing models, and (iv) commits expert edits to memory, enabling self-evolution and personalized workflows. Across four cell-segmentation benchmarks, this routing yields a 15.7\% mean accuracy gain over state-of-the-art baselines. On endoplasmic reticulum and mitochondria from new datasets, GenCellAgent improves average IoU by 37.6\% over specialist models. It also segments novel objects such as the Golgi apparatus via iterative text-guided refinement, with light human correction further boosting performance. Together, these capabilities provide a practical path to robust, adaptable cellular image segmentation without retraining, while reducing annotation burden and matching user preferences.

**arXiv ID:** 2510.13896
</details>

<details>
<summary><strong>Benefits and Limitations of Communication in Multi-Agent Reasoning</strong> - Michael Rizvi-Martel, Satwik Bhattamishra, Neil Rathi, Guillaume Rabusseau, Michael Hahn - [[pdf]](https://arxiv.org/pdf/2510.13903)</summary>

**Abstract:** Chain-of-thought prompting has popularized step-by-step reasoning in large language models, yet model performance still degrades as problem complexity and context length grow. By decomposing difficult tasks with long contexts into shorter, manageable ones, recent multi-agent paradigms offer a promising near-term solution to this problem. However, the fundamental capacities of such systems are poorly understood. In this work, we propose a theoretical framework to analyze the expressivity of multi-agent systems. We apply our framework to three algorithmic families: state tracking, recall, and $k$-hop reasoning. We derive bounds on (i) the number of agents required to solve the task exactly, (ii) the quantity and structure of inter-agent communication, and (iii) the achievable speedups as problem size and context scale. Our results identify regimes where communication is provably beneficial, delineate tradeoffs between agent count and bandwidth, and expose intrinsic limitations when either resource is constrained. We complement our theoretical analysis with a set of experiments on pretrained LLMs using controlled synthetic benchmarks. Empirical outcomes confirm the tradeoffs between key quantities predicted by our theory. Collectively, our analysis offers principled guidance for designing scalable multi-agent reasoning systems.

**arXiv ID:** 2510.13903
</details>

<details>
<summary><strong>Static Sandboxes Are Inadequate: Modeling Societal Complexity Requires Open-Ended Co-Evolution in LLM-Based Multi-Agent Simulations</strong> - Jinkun Chen, Sher Badshah, Xuemin Yu, Sijia Han, Jiechao Gao - [[pdf]](https://arxiv.org/pdf/2510.13982)</summary>

**Abstract:** What if artificial agents could not just communicate, but also evolve, adapt, and reshape their worlds in ways we cannot fully predict? With llm now powering multi-agent systems and social simulations, we are witnessing new possibilities for modeling open-ended, ever-changing environments. Yet, most current simulations remain constrained within static sandboxes, characterized by predefined tasks, limited dynamics, and rigid evaluation criteria. These limitations prevent them from capturing the complexity of real-world societies. In this paper, we argue that static, task-specific benchmarks are fundamentally inadequate and must be rethought. We critically review emerging architectures that blend llm with multi-agent dynamics, highlight key hurdles such as balancing stability and diversity, evaluating unexpected behaviors, and scaling to greater complexity, and introduce a fresh taxonomy for this rapidly evolving field. Finally, we present a research roadmap centered on open-endedness, continuous co-evolution, and the development of resilient, socially aligned AI ecosystems. \textbf{We call on the community to move beyond static paradigms and help shape the next generation of adaptive, socially-aware multi-agent simulations.}

**arXiv ID:** 2510.13982
</details>

<details>
<summary><strong>MAFA: A Multi-Agent Framework for Enterprise-Scale Annotation with Configurable Task Adaptation</strong> - Mahmood Hegazy, Aaron Rodrigues, Azzam Naeem - [[pdf]](https://arxiv.org/pdf/2510.14184)</summary>

**Abstract:** We present MAFA (Multi-Agent Framework for Annotation), a production-deployed system that transforms enterprise-scale annotation workflows through configurable multi-agent collaboration. Addressing the critical challenge of annotation backlogs in financial services, where millions of customer utterances require accurate categorization, MAFA combines specialized agents with structured reasoning and a judge-based consensus mechanism. Our framework uniquely supports dynamic task adaptation, allowing organizations to define custom annotation types (FAQs, intents, entities, or domain-specific categories) through configuration rather than code changes. Deployed at JP Morgan Chase, MAFA has eliminated a 1 million utterance backlog while achieving, on average, 86% agreement with human annotators, annually saving over 5,000 hours of manual annotation work. The system processes utterances with annotation confidence classifications, which are typically 85% high, 10% medium, and 5% low across all datasets we tested. This enables human annotators to focus exclusively on ambiguous and low-coverage cases. We demonstrate MAFA's effectiveness across multiple datasets and languages, showing consistent improvements over traditional and single-agent annotation baselines: 13.8% higher Top-1 accuracy, 15.1% improvement in Top-5 accuracy, and 16.9% better F1 in our internal intent classification dataset and similar gains on public benchmarks. This work bridges the gap between theoretical multi-agent systems and practical enterprise deployment, providing a blueprint for organizations facing similar annotation challenges.

**arXiv ID:** 2510.14184
</details>

<details>
<summary><strong>The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems</strong> - Prateek Gupta, Qiankun Zhong, Hiromu Yakura, Thomas Eisenmann, Iyad Rahwan - [[pdf]](https://arxiv.org/pdf/2510.14401)</summary>

**Abstract:** A growing body of multi-agent studies with Large Language Models (LLMs) explores how norms and cooperation emerge in mixed-motive scenarios, where pursuing individual gain can undermine the collective good. While prior work has explored these dynamics in both richly contextualized simulations and simplified game-theoretic environments, most LLM systems featuring common-pool resource (CPR) games provide agents with explicit reward functions directly tied to their actions. In contrast, human cooperation often emerges without full visibility into payoffs and population, relying instead on heuristics, communication, and punishment. We introduce a CPR simulation framework that removes explicit reward signals and embeds cultural-evolutionary mechanisms: social learning (adopting strategies and beliefs from successful peers) and norm-based punishment, grounded in Ostrom's principles of resource governance. Agents also individually learn from the consequences of harvesting, monitoring, and punishing via environmental feedback, enabling norms to emerge endogenously. We establish the validity of our simulation by reproducing key findings from existing studies on human behavior. Building on this, we examine norm evolution across a $2\times2$ grid of environmental and social initialisations (resource-rich vs. resource-scarce; altruistic vs. selfish) and benchmark how agentic societies comprised of different LLMs perform under these conditions. Our results reveal systematic model differences in sustaining cooperation and norm formation, positioning the framework as a rigorous testbed for studying emergent norms in mixed-motive LLM societies. Such analysis can inform the design of AI systems deployed in social and organizational contexts, where alignment with cooperative norms is critical for stability, fairness, and effective governance of AI-mediated environments.

**arXiv ID:** 2510.14401
</details>

<details>
<summary><strong>MAFA: A multi-agent framework for annotation</strong> - Mahmood Hegazy, Aaron Rodrigues, Azzam Naeem - [[pdf]](https://arxiv.org/pdf/2505.13668)</summary>

**Abstract:** Modern consumer banking applications require accurate and efficient retrieval of information in response to user queries. Mapping user utterances to the most relevant Frequently Asked Questions (FAQs) is a crucial component of these systems. Traditional approaches often rely on a single model or technique, which may not capture the nuances of diverse user inquiries. In this paper, we introduce a multi-agent framework for FAQ annotation that combines multiple specialized agents with different approaches and a judge agent that reranks candidates to produce optimal results. Our agents utilize a structured reasoning approach inspired by Attentive Reasoning Queries (ARQs), which guides them through systematic reasoning steps using targeted, task-specific JSON queries. Our framework features a few-shot example strategy, where each agent receives different few-shots, enhancing ensemble diversity and coverage of the query space. We evaluate our framework on a real-world major bank dataset as well as public benchmark datasets (LCQMC and FiQA), demonstrating significant improvements over single-agent approaches across multiple metrics, including a 14% increase in Top-1 accuracy, an 18% increase in Top-5 accuracy, and a 12% improvement in Mean Reciprocal Rank on our dataset, and similar gains on public benchmarks when compared with traditional and single-agent annotation techniques. Our framework is particularly effective at handling ambiguous queries, making it well-suited for deployment in production banking applications while showing strong generalization capabilities across different domains and languages.

**arXiv ID:** 2505.13668
</details>

<details>
<summary><strong>RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration</strong> - Xiuyuan Chen, Jian Zhao, Yuchen Yuan, Tianle Zhang, Huilin Zhou, Zheng Zhu, Ping Hu, Linghe Kong, Chi Zhang, Weiran Huang, Xuelong Li - [[pdf]](https://arxiv.org/pdf/2509.25271)</summary>

**Abstract:** Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87% improvement in risk identification accuracy compared to the strongest baseline evaluation method.

**arXiv ID:** 2509.25271
</details>

<details>
<summary><strong>Stop Reducing Responsibility in LLM-Powered Multi-Agent Systems to Local Alignment</strong> - Jinwei Hu, Yi Dong, Shuang Ao, Zhuoyun Li, Boxuan Wang, Lokesh Singh, Guangliang Cheng, Sarvapali D. Ramchurn, Xiaowei Huang - [[pdf]](https://arxiv.org/pdf/2510.14008)</summary>

**Abstract:** LLM-powered Multi-Agent Systems (LLM-MAS) unlock new potentials in distributed reasoning, collaboration, and task generalization but also introduce additional risks due to unguaranteed agreement, cascading uncertainty, and adversarial vulnerabilities. We argue that ensuring responsible behavior in such systems requires a paradigm shift: from local, superficial agent-level alignment to global, systemic agreement. We conceptualize responsibility not as a static constraint but as a lifecycle-wide property encompassing agreement, uncertainty, and security, each requiring the complementary integration of subjective human-centered values and objective verifiability. Furthermore, a dual-perspective governance framework that combines interdisciplinary design with human-AI collaborative oversight is essential for tracing and ensuring responsibility throughout the lifecycle of LLM-MAS. Our position views LLM-MAS not as loose collections of agents, but as unified, dynamic socio-technical systems that demand principled mechanisms to support each dimension of responsibility and enable ethically aligned, verifiably coherent, and resilient behavior for sustained, system-wide agreement.

**arXiv ID:** 2510.14008
</details>

<details>
<summary><strong>Multi Agent Switching Mode Controller for Sound Source localization</strong> - Marcello Sorge, Nicola Cigarini, Riccardo Lorigiola, Giulia Michieletto, Andrea Masiero, Angelo Cenedese, Alberto Guarnieri - [[pdf]](https://arxiv.org/pdf/2510.14849)</summary>

**Abstract:** Source seeking is an important topic in robotic research, especially considering sound-based sensors since they allow the agents to locate a target even in critical conditions where it is not possible to establish a direct line of sight. In this work, we design a multi- agent switching mode control strategy for acoustic-based target localization. Two scenarios are considered: single source localization, in which the agents are driven maintaining a rigid formation towards the target, and multi-source scenario, in which each agent searches for the targets independently from the others.

**arXiv ID:** 2510.14849
</details>

<details>
<summary><strong>Inferring Foresightedness in Dynamic Noncooperative Games</strong> - Cade Armstrong, Ryan Park, Xinjie Liu, Kushagra Gupta, David Fridovich-Keil - [[pdf]](https://arxiv.org/pdf/2412.01017)</summary>

**Abstract:** Dynamic game theory is an increasingly popular tool for modeling multi-agent, e.g. human-robot, interactions. Game-theoretic models presume that each agent wishes to minimize a private cost function that depends on others' actions. These games typically evolve over a fixed time horizon, specifying how far into the future each agent plans. In practical settings, however, decision-makers may vary in foresightedness, or how much they care about their current cost in relation to their past and future costs. We conjecture that quantifying and estimating each agent's foresightedness from online data will enable safer and more efficient interactions with other agents. To this end, we frame this inference problem as an inverse dynamic game. We consider a specific objective function parametrization that smoothly interpolates myopic and farsighted planning. Games of this form are readily transformed into parametric mixed complementarity problems; we exploit the directional differentiability of solutions to these problems with respect to their hidden parameters to solve for agents' foresightedness. We conduct three experiments: one with synthetically generated delivery robot motion, one with real-world data involving people walking, biking, and driving vehicles, and one using high-fidelity simulators. The results of these experiments demonstrate that explicitly inferring agents' foresightedness enables game-theoretic models to make 33% more accurate models for agents' behavior.

**arXiv ID:** 2412.01017
</details>

<details>
<summary><strong>RareAgent: Self-Evolving Reasoning for Drug Repurposing in Rare Diseases</strong> - Lang Qin, Zijian Gan, Xu Cao, Pengcheng Jiang, Yankai Jiang, Jiawei Han, Kaishun Wu, Jintai Chen - [[pdf]](https://arxiv.org/pdf/2510.05764)</summary>

**Abstract:** Computational drug repurposing for rare diseases is especially challenging when no prior associations exist between drugs and target diseases. Therefore, knowledge graph completion and message-passing GNNs have little reliable signal to learn and propagate, resulting in poor performance. We present RareAgent, a self-evolving multi-agent system that reframes this task from passive pattern recognition to active evidence-seeking reasoning. RareAgent organizes task-specific adversarial debates in which agents dynamically construct evidence graphs from diverse perspectives to support, refute, or entail hypotheses. The reasoning strategies are analyzed post hoc in a self-evolutionary loop, producing textual feedback that refines agent policies, while successful reasoning paths are distilled into transferable heuristics to accelerate future investigations. Comprehensive evaluations reveal that RareAgent improves the indication AUPRC by 18.1% over reasoning baselines and provides a transparent reasoning chain consistent with clinical evidence.

**arXiv ID:** 2510.05764
</details>

<details>
<summary><strong>Measuring and Mitigating Identity Bias in Multi-Agent Debate via Anonymization</strong> - Hyeong Kyu Choi, Xiaojin Zhu, Sharon Li - [[pdf]](https://arxiv.org/pdf/2510.07517)</summary>

**Abstract:** Multi-agent debate (MAD) aims to improve large language model (LLM) reasoning by letting multiple agents exchange answers and then aggregate their opinions. Yet recent studies reveal that agents are not neutral: they are prone to identity-driven sycophancy and self-bias, uncritically adopting a peer's view or stubbornly adhering to their own prior output, undermining the reliability of debate. In this work, we present the first principled framework that joins sycophancy and self-bias to mitigate and quantify identity bias in MAD. First, we formalize the debate dynamics as an identity-weighted Bayesian update process. Second, we propose response anonymization: by removing identity markers from prompts, agents cannot distinguish "self" from "peer", which forces equal weights on agent identity, thereby reducing bias. Third, we define the Identity Bias Coefficient (IBC), a principled metric that measures how often an agent follows a peer versus itself. Empirical studies across multiple models, datasets and debate rounds confirm that identity bias is widespread, with sycophancy far more common than self-bias. Our findings highlight the need to "mask" identity to ensure that MAD systems reason based on content rather than source identity. Code is released in this https URL.

**arXiv ID:** 2510.07517
</details>

<details>
<summary><strong>Ax-Prover: A Deep Reasoning Agentic Framework for Theorem Proving in Mathematics and Quantum Physics</strong> - Marco Del Tredici, Jacob McCarran, Benjamin Breen, Javier Aspuru Mijares, Weichen Winston Yin, Jacob M. Taylor, Frank H. L. Koppens, Dirk Englund - [[pdf]](https://arxiv.org/pdf/2510.12787)</summary>

**Abstract:** We present Ax-Prover, a multi-agent system for automated theorem proving in Lean that can solve problems across diverse scientific domains and operate either autonomously or collaboratively with human experts. To achieve this, Ax-Prover approaches scientific problem solving through formal proof generation, a process that demands both creative reasoning and strict syntactic rigor. Ax-Prover meets this challenge by equipping Large Language Models (LLMs), which provide knowledge and reasoning, with Lean tools via the Model Context Protocol (MCP), which ensure formal correctness. To evaluate its performance as an autonomous prover, we benchmark our approach against frontier LLMs and specialized prover models on two public math benchmarks and on two Lean benchmarks we introduce in the fields of abstract algebra and quantum theory. On public datasets, Ax-Prover is competitive with state-of-the-art provers, while it largely outperforms them on the new benchmarks. This shows that, unlike specialized systems that struggle to generalize, our tool-based agentic theorem prover approach offers a generalizable methodology for formal verification across diverse scientific domains. Furthermore, we demonstrate Ax-Prover's assistant capabilities in a practical use case, showing how it enabled an expert mathematician to formalize the proof of a complex cryptography theorem.

**arXiv ID:** 2510.12787
</details>

<details>
<summary><strong>A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System</strong> - Jiale Guo, Suizhi Huang, Mei Li, Dong Huang, Xingsheng Chen, Regina Zhang, Zhijiang Guo, Han Yu, Siu-Ming Yiu, Christian Jensen, Pietro Lio, Kwok-Yan Lam - [[pdf]](https://arxiv.org/pdf/2510.09721)</summary>

**Abstract:** The integration of Large Language Models (LLMs) into software engineering has driven a transition from traditional rule-based systems to autonomous agentic systems capable of solving complex problems. However, systematic progress is hindered by a lack of comprehensive understanding of how benchmarks and solutions interconnect. This survey addresses this gap by providing the first holistic analysis of LLM-powered software engineering, offering insights into evaluation methodologies and solution paradigms. We review over 150 recent papers and propose a taxonomy along two key dimensions: (1) Solutions, categorized into prompt-based, fine-tuning-based, and agent-based paradigms, and (2) Benchmarks, including tasks such as code generation, translation, and repair. Our analysis highlights the evolution from simple prompt engineering to sophisticated agentic systems incorporating capabilities like planning, reasoning, memory mechanisms, and tool augmentation. To contextualize this progress, we present a unified pipeline illustrating the workflow from task specification to deliverables, detailing how different solution paradigms address various complexity levels. Unlike prior surveys that focus narrowly on specific aspects, this work connects 50+ benchmarks to their corresponding solution strategies, enabling researchers to identify optimal approaches for diverse evaluation criteria. We also identify critical research gaps and propose future directions, including multi-agent collaboration, self-evolving systems, and formal verification integration. This survey serves as a foundational guide for advancing LLM-driven software engineering. We maintain a GitHub repository that continuously updates the reviewed and related papers at this https URL.

**arXiv ID:** 2510.09721
</details>

<details>
<summary><strong>LLM-guided Chemical Process Optimization with a Multi-Agent Approach</strong> - Tong Zeng, Srivathsan Badrinarayanan, Janghoon Ock, Cheng-Kai Lai, Amir Barati Farimani - [[pdf]](https://arxiv.org/pdf/2506.20921)</summary>

**Abstract:** Chemical process optimization maximizes production efficiency and economic performance, but optimization algorithms, including gradient-based solvers, numerical methods, and parameter grid searches, become impractical when operating constraints are ill-defined or unavailable. We present a multi-agent LLM framework that autonomously infers operating constraints from minimal process descriptions, then collaboratively guides optimization. Our AutoGen-based framework employs OpenAI's o3 model with specialized agents for constraint generation, parameter validation, simulation, and optimization guidance. Through autonomous constraint generation and iterative multi-agent optimization, the framework eliminates the need for predefined operational bounds. Validated on hydrodealkylation across cost, yield, and yield-to-cost ratio metrics, the framework achieved competitive performance with conventional methods while reducing wall-time 31-fold relative to grid search, converging in under 20 minutes. The reasoning-guided search demonstrates sophisticated process understanding, correctly identifying utility trade-offs and applying domain-informed heuristics. Unlike conventional methods requiring predefined constraints, our approach uniquely combines autonomous constraint generation with interpretable parameter exploration. Model comparison reveals reasoning-capable architectures (o3, o1) are essential for successful optimization, while standard models fail to converge. This approach is particularly valuable for emerging processes and retrofit applications where operational constraints are poorly characterized or unavailable.

**arXiv ID:** 2506.20921
</details>

</details>

<details open>
<summary><h2>Other Agent Research (13 papers)</h2></summary>

<details>
<summary><strong>Agentic NL2SQL to Reduce Computational Costs</strong> - Dominik Jehle, Lennart Purucker, Frank Hutter - [[pdf]](https://arxiv.org/pdf/2510.14808)</summary>

**Abstract:** Translating natural language queries into SQL queries (NL2SQL or Text-to-SQL) has recently been empowered by large language models (LLMs). Using LLMs to perform NL2SQL methods on a large collection of SQL databases necessitates processing large quantities of meta-information about the databases, which in turn results in lengthy prompts with many tokens and high processing costs. To address this challenge, we introduce Datalake Agent, an agentic system designed to enable an LLM to solve NL2SQL tasks more efficiently. Instead of utilizing direct solvers for NL2SQL that call the LLM once with all meta-information in the prompt, the Datalake Agent employs an interactive loop to reduce the utilized meta-information. Within the loop, the LLM is used in a reasoning framework that selectively requests only the necessary information to solve a table question answering task. We evaluate the Datalake Agent on a collection of 23 databases with 100 table question answering tasks. The Datalake Agent reduces the tokens used by the LLM by up to 87\% and thus allows for substantial cost reductions while maintaining competitive performance.

**arXiv ID:** 2510.14808
</details>

<details>
<summary><strong>A2AS: Agentic AI Runtime Security and Self-Defense</strong> - Eugene Neelou, Ivan Novikov, Max Moroz, Om Narayan, Tiffany Saade, Mika Ayenson, Ilya Kabanov, Jen Ozmen, Edward Lee, Vineeth Sai Narajala, Emmanuel Guilherme Junior, Ken Huang, Huseyin Gulsin, Jason Ross, Marat Vyshegorodtsev, Adelin Travers, Idan Habler, Rahul Jadav - [[pdf]](https://arxiv.org/pdf/2510.13825)</summary>

**Abstract:** The A2AS framework is introduced as a security layer for AI agents and LLM-powered applications, similar to how HTTPS secures HTTP. A2AS enforces certified behavior, activates model self-defense, and ensures context window integrity. It defines security boundaries, authenticates prompts, applies security rules and custom policies, and controls agentic behavior, enabling a defense-in-depth strategy. The A2AS framework avoids latency overhead, external dependencies, architectural changes, model retraining, and operational complexity. The BASIC security model is introduced as the A2AS foundation: (B) Behavior certificates enable behavior enforcement, (A) Authenticated prompts enable context window integrity, (S) Security boundaries enable untrusted input isolation, (I) In-context defenses enable secure model reasoning, (C) Codified policies enable application-specific rules. This first paper in the series introduces the BASIC security model and the A2AS framework, exploring their potential toward establishing the A2AS industry standard.

**arXiv ID:** 2510.13825
</details>

<details>
<summary><strong>DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans</strong> - Bingsheng Yao, Bo Sun, Yuanzhe Dong, Yuxuan Lu, Dakuo Wang - [[pdf]](https://arxiv.org/pdf/2510.14205)</summary>

**Abstract:** The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these this http URL evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie this http URL can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and this http URL work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.

**arXiv ID:** 2510.14205
</details>

<details>
<summary><strong>LLMs as Scalable, General-Purpose Simulators For Evolving Digital Agent Training</strong> - Yiming Wang, Da Yin, Yuedong Cui, Ruichen Zheng, Zhiqian Li, Zongyu Lin, Di Wu, Xueqing Wu, Chenchen Ye, Yu Zhou, Kai-Wei Chang - [[pdf]](https://arxiv.org/pdf/2510.14969)</summary>

**Abstract:** Digital agents require diverse, large-scale UI trajectories to generalize across real-world tasks, yet collecting such data is prohibitively expensive in both human annotation, infra and engineering perspectives. To this end, we introduce $\textbf{UI-Simulator}$, a scalable paradigm that generates structured UI states and transitions to synthesize training trajectories at scale. Our paradigm integrates a digital world simulator for diverse UI states, a guided rollout process for coherent exploration, and a trajectory wrapper that produces high-quality and diverse trajectories for agent training. We further propose $\textbf{UI-Simulator-Grow}$, a targeted scaling strategy that enables more rapid and data-efficient scaling by prioritizing high-impact tasks and synthesizes informative trajectory variants. Experiments on WebArena and AndroidWorld show that UI-Simulator rivals or surpasses open-source agents trained on real UIs with significantly better robustness, despite using weaker teacher models. Moreover, UI-Simulator-Grow matches the performance of Llama-3-70B-Instruct using only Llama-3-8B-Instruct as the base model, highlighting the potential of targeted synthesis scaling paradigm to continuously and efficiently enhance the digital agents.

**arXiv ID:** 2510.14969
</details>

<details>
<summary><strong>Generative AI Meets Future Cities: Towards an Era of Autonomous Urban Intelligence</strong> - Dongjie Wang, Chang-Tien Lu, Xinyue Ye, Tan Yigitcanlar, Yanjie Fu - [[pdf]](https://arxiv.org/pdf/2304.03892)</summary>

**Abstract:** The two fields of urban planning and artificial intelligence (AI) arose and developed separately. However, there is now cross-pollination and increasing interest in both fields to benefit from the advances of the other. In the present paper, we introduce the importance of urban planning from the sustainability, living, economic, disaster, and environmental perspectives. We review the fundamental concepts of urban planning and relate these concepts to crucial open problems of machine learning, including adversarial learning, generative neural networks, deep encoder-decoder networks, conversational AI, and geospatial and temporal machine learning, thereby assaying how AI can contribute to modern urban planning. Thus, a central problem is automated land-use configuration, which is formulated as the generation of land uses and building configuration for a target area from surrounding geospatial, human mobility, social media, environment, and economic activities. Finally, we delineate some implications of AI for urban planning and propose key research areas at the intersection of both topics.

**arXiv ID:** 2304.03892
</details>

<details>
<summary><strong>Paper2Agent: Reimagining Research Papers As Interactive and Reliable AI Agents</strong> - Jiacheng Miao, Joe R. Davis, Yaohui Zhang, Jonathan K. Pritchard, James Zou - [[pdf]](https://arxiv.org/pdf/2509.06917)</summary>

**Abstract:** We introduce Paper2Agent, an automated framework that converts research papers into AI agents. Paper2Agent transforms research output from passive artifacts into active systems that can accelerate downstream use, adoption, and discovery. Conventional research papers require readers to invest substantial effort to understand and adapt a paper's code, data, and methods to their own work, creating barriers to dissemination and reuse. Paper2Agent addresses this challenge by automatically converting a paper into an AI agent that acts as a knowledgeable research assistant. It systematically analyzes the paper and the associated codebase using multiple agents to construct a Model Context Protocol (MCP) server, then iteratively generates and runs tests to refine and robustify the resulting MCP. These paper MCPs can then be flexibly connected to a chat agent (e.g. Claude Code) to carry out complex scientific queries through natural language while invoking tools and workflows from the original paper. We demonstrate Paper2Agent's effectiveness in creating reliable and capable paper agents through in-depth case studies. Paper2Agent created an agent that leverages AlphaGenome to interpret genomic variants and agents based on ScanPy and TISSUE to carry out single-cell and spatial transcriptomics analyses. We validate that these paper agents can reproduce the original paper's results and can correctly carry out novel user queries. Paper2Agent automatically created AI co-scientist that identified new splicing variant associated with ADHD risk. By turning static papers into dynamic, interactive AI agents, Paper2Agent introduces a new paradigm for knowledge dissemination and a foundation for the collaborative ecosystem of AI co-scientists.

**arXiv ID:** 2509.06917
</details>

<details>
<summary><strong>Constraint-Driven Small Language Models Based on Agent and OpenAlex Knowledge Graph: Mining Conceptual Pathways and Discovering Innovation Points in Academic Papers</strong> - Ziye Xia, Sergei S. Ospichev - [[pdf]](https://arxiv.org/pdf/2510.14303)</summary>

**Abstract:** In recent years, the rapid increase in academic publications across various fields has posed severe challenges for academic paper analysis: scientists struggle to timely and comprehensively track the latest research findings and methodologies. Key concept extraction has proven to be an effective analytical paradigm, and its automation has been achieved with the widespread application of language models in industrial and scientific domains. However, existing paper databases are mostly limited to similarity matching and basic classification of key concepts, failing to deeply explore the relational networks between concepts. This paper is based on the OpenAlex opensource knowledge graph. By analyzing nearly 8,000 open-source paper data from Novosibirsk State University, we discovered a strong correlation between the distribution patterns of paper key concept paths and both innovation points and rare paths. We propose a prompt engineering-based key concept path analysis method. This method leverages small language models to achieve precise key concept extraction and innovation point identification, and constructs an agent based on a knowledge graph constraint mechanism to enhance analysis accuracy. Through fine-tuning of the Qwen and DeepSeek models, we achieved significant improvements in accuracy, with the models publicly available on the Hugging Face platform.

**arXiv ID:** 2510.14303
</details>

<details>
<summary><strong>Intelligent Dynamic Handover via AI-assisted Signal Quality Prediction in 6G Multi-RAT Networks</strong> - Maria Lamprini A. Bartsioka, Anastasios Giannopoulos, Sotirios Spantideas - [[pdf]](https://arxiv.org/pdf/2510.14832)</summary>

**Abstract:** The emerging paradigm of 6G multiple Radio Access Technology (multi-RAT) networks, where cellular and Wireless Fidelity (WiFi) transmitters coexist, requires mobility decisions that remain reliable under fast channel dynamics, interference, and heterogeneous coverage. Handover in multi-RAT deployments is still highly reactive and event-triggered, relying on instantaneous measurements and threshold events. This work proposes a Machine Learning (ML)-assisted Predictive Conditional Handover (P-CHO) framework based on a model-driven and short-horizon signal quality forecasts. We present a generalized P-CHO sequence workflow orchestrated by a RAT Steering Controller, which standardizes data collection, parallel per-RAT predictions, decision logic with hysteresis-based conditions, and CHO execution. Considering a realistic multi-RAT environment, we train RAT-aware Long Short Term Memory (LSTM) networks to forecast the signal quality indicators of mobile users along randomized trajectories. The proposed P-CHO models are trained and evaluated under different channel models for cellular and IEEE 802.11 WiFi integrated coverage. We study the impact of hyperparameter tuning of LSTM models under different system settings, and compare direct multi-step versus recursive P-CHO variants. Comparisons against baseline predictors are also carried out. Finally, the proposed P-CHO is tested under soft and hard handover settings, showing that hysteresis-enabled P-CHO scheme is able to reduce handover failures and ping-pong events. Overall, the proposed P-CHO framework can enable accurate, low-latency, and proactive handovers suitable for ML-assisted handover steering in 6G multi-RAT deployments.

**arXiv ID:** 2510.14832
</details>

<details>
<summary><strong>Spatially Intelligent Patrol Routes for Concealed Emitter Localization by Robot Swarms</strong> - Adam Morris, Timothy Pelham, Edmund R. Hunt - [[pdf]](https://arxiv.org/pdf/2510.14018)</summary>

**Abstract:** This paper introduces a method for designing spatially intelligent robot swarm behaviors to localize concealed radio emitters. We use differential evolution to generate geometric patrol routes that localize unknown signals independently of emitter parameters, a key challenge in electromagnetic surveillance. Patrol shape and antenna type are shown to influence information gain, which in turn determines the effective triangulation coverage. We simulate a four-robot swarm across eight configurations, assigning pre-generated patrol routes based on a specified patrol shape and sensing capability (antenna type: omnidirectional or directional). An emitter is placed within the map for each trial, with randomized position, transmission power and frequency. Results show that omnidirectional localization success rates are driven primarily by source location rather than signal properties, with failures occurring most often when sources are placed in peripheral areas of the map. Directional antennas are able to overcome this limitation due to their higher gain and directivity, with an average detection success rate of 98.75% compared to 80.25% for omnidirectional. Average localization errors range from 1.01-1.30 m for directional sensing and 1.67-1.90 m for omnidirectional sensing; while directional sensing also benefits from shorter patrol edges. These results demonstrate that a swarm's ability to predict electromagnetic phenomena is directly dependent on its physical interaction with the environment. Consequently, spatial intelligence, realized here through optimized patrol routes and antenna selection, is a critical design consideration for effective robotic surveillance.

**arXiv ID:** 2510.14018
</details>

<details>
<summary><strong>A Faster and More Reliable Middleware for Autonomous Driving Systems</strong> - Yuankai He, Weisong Shi - [[pdf]](https://arxiv.org/pdf/2510.11448)</summary>

**Abstract:** Ensuring safety in high-speed autonomous vehicles requires rapid control loops and tightly bounded delays from perception to actuation. Many open-source autonomy systems rely on ROS 2 middleware; when multiple sensor and control nodes share one compute unit, ROS 2 and its DDS transports add significant (de)serialization, copying, and discovery overheads, shrinking the available time budget. We present Sensor-in-Memory (SIM), a shared-memory transport designed for intra-host pipelines in autonomous vehicles. SIM keeps sensor data in native memory layouts (e.g., cv::Mat, PCL), uses lock-free bounded double buffers that overwrite old data to prioritize freshness, and integrates into ROS 2 nodes with four lines of code. Unlike traditional middleware, SIM operates beside ROS 2 and is optimized for applications where data freshness and minimal latency outweigh guaranteed completeness. SIM provides sequence numbers, a writer heartbeat, and optional checksums to ensure ordering, liveness, and basic integrity. On an NVIDIA Jetson Orin Nano, SIM reduces data-transport latency by up to 98% compared to ROS 2 zero-copy transports such as FastRTPS and Zenoh, lowers mean latency by about 95%, and narrows 95th/99th-percentile tail latencies by around 96%. In tests on a production-ready Level 4 vehicle running this http URL, SIM increased localization frequency from 7.5 Hz to 9.5 Hz. Applied across all latency-critical modules, SIM cut average perception-to-decision latency from 521.91 ms to 290.26 ms, reducing emergency braking distance at 40 mph (64 km/h) on dry concrete by 13.6 ft (4.14 m).

**arXiv ID:** 2510.11448
</details>

<details>
<summary><strong>AI-Agents for Culturally Diverse Online Higher Education Environments</strong> - Fuze Sun, Paul Craig, Lingyu Li, Shixiangyue Meng, Chuxi Nan - [[pdf]](https://arxiv.org/pdf/2510.10520)</summary>

**Abstract:** As the global reach of online higher education continues to grow, universities are increasingly accommodating students from diverse cultural backgrounds (Tereshko et al., 2024). This can present a number of challenges including linguistic barriers (Ullah et al., 2021), cultural differences in learning style (Omidvar & Tan, 2012), cultural sensitivity in course design (Nguyen, 2022) and perceived isolation when students feel their perspectives or experiences are not reflected or valued in the learning environment (Hansen-Brown et al., 2022). Ensuring active engagement and reasonable learning outcomes in such a environments requires distance educational systems that are not only adaptive but also culturally resonant (Dalle et al., 2024). Both embodied and virtual AI-Agents have great potential in this regard as they can facilitate personalized learning and adapt their interactions and content delivery to align with students' cultural context. In addition, Generative AI (GAI), such as, Large Language Models (LLMs) can amplify the potential for these culturally aware AI agents to address educational challenges due to their advanced capacity for understanding and generating contextually relevant content (Wang et al., 2024). This chapter reviews existing research and suggests the usage of culturally aware AI-Agents, powered by GAI, to foster engagement and improve learning outcomes in culturally diverse online higher education environments.

**arXiv ID:** 2510.10520
</details>

<details>
<summary><strong>ReUseIt: Synthesizing Reusable AI Agent Workflows for Web Automation</strong> - Yimeng Liu, Misha Sra, Jeevana Priya Inala, Chenglong Wang - [[pdf]](https://arxiv.org/pdf/2510.14308)</summary>

**Abstract:** AI-powered web agents have the potential to automate repetitive tasks, such as form filling, information retrieval, and scheduling, but they struggle to reliably execute these tasks without human intervention, requiring users to provide detailed guidance during every run. We address this limitation by automatically synthesizing reusable workflows from an agent's successful and failed attempts. These workflows incorporate execution guards that help agents detect and fix errors while keeping users informed of progress and issues. Our approach enables agents to successfully complete repetitive tasks of the same type with minimal intervention, increasing the success rates from 24.2% to 70.1% across fifteen tasks. To evaluate this approach, we invited nine users and found that our agent helped them complete web tasks with a higher success rate and less guidance compared to two baseline methods, as well as allowed users to easily monitor agent behavior and understand failures.

**arXiv ID:** 2510.14308
</details>

<details>
<summary><strong>Dude, Where's My (Autonomous) Car? Defining an Accessible Description Logic for Blind and Low Vision Travelers Using Autonomous Vehicles</strong> - Paul D. S. Fink, Justin R. Brown, Rachel Coombs, Emily A. Hamby, Kyle J. James, Aisha Harris, Jacob Bond, Morgan E. Andrulis, Nicholas A. Giudice - [[pdf]](https://arxiv.org/pdf/2510.14911)</summary>

**Abstract:** Purpose: Autonomous vehicles (AVs) are becoming a promising transportation solution for blind and low-vision (BLV) travelers, offering the potential for greater independent mobility. This paper explores the information needs of BLV users across multiple steps of the transportation journey, including finding and navigating to, entering, and exiting vehicles independently.
Methods: A survey with 202 BLV respondents and interviews with 12 BLV individuals revealed the perspectives of BLV end-users and informed the sequencing of natural language information required for successful travel. Whereas the survey identified key information needs across the three trip segments, the interviews helped prioritize how that information should be presented in a sequence of accessible descriptions to travelers.
Results: Taken together, the survey and interviews reveal that BLV users prioritize knowing the vehicle's make and model and how to find the correct vehicle during the navigation phase. They also emphasize the importance of confirmations about the vehicle's destination and onboard safety features upon entering the vehicle. While exiting, BLV users value information about hazards and obstacles, as well as knowing which side of the vehicle to exit. Furthermore, results highlight that BLV travelers desire using their own smartphone devices when receiving information from AVs and prefer audio-based interaction.
Conclusion: The findings from this research contribute a structured framework for delivering trip-related information to BLV users, useful for designers incorporating natural language descriptions tailored to each travel segment. This work offers important contributions for sequencing transportation-related descriptions throughout the AV journey, ultimately enhancing the mobility and independence of BLV individuals.

**arXiv ID:** 2510.14911
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (34 papers)</h2></summary>

<details>
<summary><strong>Combining Reinforcement Learning and Behavior Trees for NPCs in Video Games with AMD Schola</strong> - Tian Liu, Alex Cann, Ian Colbert, Mehdi Saeedi - [[pdf]](https://arxiv.org/pdf/2510.14154)</summary>

**Abstract:** While the rapid advancements in the reinforcement learning (RL) research community have been remarkable, the adoption in commercial video games remains slow. In this paper, we outline common challenges the Game AI community faces when using RL-driven NPCs in practice, and highlight the intersection of RL with traditional behavior trees (BTs) as a crucial juncture to be explored further. Although the BT+RL intersection has been suggested in several research papers, its adoption is rare. We demonstrate the viability of this approach using AMD Schola -- a plugin for training RL agents in Unreal Engine -- by creating multi-task NPCs in a complex 3D environment inspired by the commercial video game ``The Last of Us". We provide detailed methodologies for jointly training RL models with BTs while showcasing various skills.

**arXiv ID:** 2510.14154
</details>

<details>
<summary><strong>ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning</strong> - Roger Creus Castanyer, Faisal Mohamed, Pablo Samuel Castro, Cyrus Neary, Glen Berseth - [[pdf]](https://arxiv.org/pdf/2510.14176)</summary>

**Abstract:** Reinforcement learning (RL) algorithms are highly sensitive to reward function specification, which remains a central challenge limiting their broad applicability. We present ARM-FM: Automated Reward Machines via Foundation Models, a framework for automated, compositional reward design in RL that leverages the high-level reasoning capabilities of foundation models (FMs). Reward machines (RMs) -- an automata-based formalism for reward specification -- are used as the mechanism for RL objective specification, and are automatically constructed via the use of FMs. The structured formalism of RMs yields effective task decompositions, while the use of FMs enables objective specifications in natural language. Concretely, we (i) use FMs to automatically generate RMs from natural language specifications; (ii) associate language embeddings with each RM automata-state to enable generalization across tasks; and (iii) provide empirical evidence of ARM-FM's effectiveness in a diverse suite of challenging environments, including evidence of zero-shot generalization.

**arXiv ID:** 2510.14176
</details>

<details>
<summary><strong>Towards Agentic Self-Learning LLMs in Search Environment</strong> - Wangtao Sun, Xiang Cheng, Jialin Fan, Yao Xu, Xing Yu, Shizhu He, Jun Zhao, Kang Liu - [[pdf]](https://arxiv.org/pdf/2510.14253)</summary>

**Abstract:** We study whether self-learning can scale LLM-based agents without relying on human-curated datasets or predefined rule-based rewards. Through controlled experiments in a search-agent setting, we identify two key determinants of scalable agent training: the source of reward signals and the scale of agent task data. We find that rewards from a Generative Reward Model (GRM) outperform rigid rule-based signals for open-domain learning, and that co-evolving the GRM with the policy further boosts performance. Increasing the volume of agent task data-even when synthetically generated-substantially enhances agentic capabilities. Building on these insights, we propose \textbf{Agentic Self-Learning} (ASL), a fully closed-loop, multi-role reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator, a Policy Model, and a Generative Reward Model to form a virtuous cycle of harder task setting, sharper verification, and stronger solving. Empirically, ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines (e.g., Search-R1) that plateau or degrade, and continues improving under zero-labeled-data conditions, indicating superior sample efficiency and robustness. We further show that GRM verification capacity is the main bottleneck: if frozen, it induces reward hacking and stalls progress; continual GRM training on the evolving data distribution mitigates this, and a small late-stage injection of real verification data raises the performance ceiling. This work establishes reward source and data scale as critical levers for open-domain agent learning and demonstrates the efficacy of multi-role co-evolution for scalable, self-improving agents. The data and code of this paper are released at this https URL

**arXiv ID:** 2510.14253
</details>

<details>
<summary><strong>Hi-Agent: Hierarchical Vision-Language Agents for Mobile Device Control</strong> - Zhe Wu, Hongjin Lu, Junliang Xing, Changhao Zhang, Yin Zhu, Yuhao Yang, Yuheng Jing, Kai Li, Kun Shao, Jianye Hao, Jun Wang, Yuanchun Shi - [[pdf]](https://arxiv.org/pdf/2510.14388)</summary>

**Abstract:** Building agents that autonomously operate mobile devices has attracted increasing attention. While Vision-Language Models (VLMs) show promise, most existing approaches rely on direct state-to-action mappings, which lack structured reasoning and planning, and thus generalize poorly to novel tasks or unseen UI layouts. We introduce Hi-Agent, a trainable hierarchical vision-language agent for mobile control, featuring a high-level reasoning model and a low-level action model that are jointly optimized. For efficient training, we reformulate multi-step decision-making as a sequence of single-step subgoals and propose a foresight advantage function, which leverages execution feedback from the low-level model to guide high-level optimization. This design alleviates the path explosion issue encountered by Group Relative Policy Optimization (GRPO) in long-horizon tasks and enables stable, critic-free joint training. Hi-Agent achieves a new State-Of-The-Art (SOTA) 87.9% task success rate on the Android-in-the-Wild (AitW) benchmark, significantly outperforming prior methods across three paradigms: prompt-based (AppAgent: 17.7%), supervised (Filtered BC: 54.5%), and reinforcement learning-based (DigiRL: 71.9%). It also demonstrates competitive zero-shot generalization on the ScreenSpot-v2 benchmark. On the more challenging AndroidWorld benchmark, Hi-Agent also scales effectively with larger backbones, showing strong adaptability in high-complexity mobile control scenarios.

**arXiv ID:** 2510.14388
</details>

<details>
<summary><strong>ColorBench: Benchmarking Mobile Agents with Graph-Structured Framework for Complex Long-Horizon Tasks</strong> - Yuanyi Song, Heyuan Huang, Qiqiang Lin, Yin Zhao, Xiangmou Qu, Jun Wang, Xingyu Lou, Weiwen Liu, Zhuosheng Zhang, Jun Wang, Yong Yu, Weinan Zhang, Zhaoxiang Wang - [[pdf]](https://arxiv.org/pdf/2510.14621)</summary>

**Abstract:** The rapid advancement of multimodal large language models has enabled agents to operate mobile devices by directly interacting with graphical user interfaces, opening new possibilities for mobile automation. However, real-world mobile tasks are often complex and allow for multiple valid solutions. This contradicts current mobile agent evaluation standards: offline static benchmarks can only validate a single predefined "golden path", while online dynamic testing is constrained by the complexity and non-reproducibility of real devices, making both approaches inadequate for comprehensively assessing agent capabilities. To bridge the gap between offline and online evaluation and enhance testing stability, this paper introduces a novel graph-structured benchmarking framework. By modeling the finite states observed during real-device interactions, it achieves static simulation of dynamic behaviors. Building on this, we develop ColorBench, a benchmark focused on complex long-horizon tasks. It supports evaluation of multiple valid solutions, subtask completion rate statistics, and atomic-level capability analysis. ColorBench contains 175 tasks (74 single-app, 101 cross-app) with an average length of over 13 steps. Each task includes at least two correct paths and several typical error paths, enabling quasi-dynamic interaction. By evaluating ColorBench across various baselines, we discover limitations of existing models and propose improvement directions and feasible technical pathways to enhance agents' performance on complex, long-horizon problems based on experimental results. Code and data are available at: this https URL.

**arXiv ID:** 2510.14621
</details>

<details>
<summary><strong>RoboGPT-R1: Enhancing Robot Planning with Reinforcement Learning</strong> - Jinrui Liu, Bingyan Nie, Boyu Li, Yaran Chen, Yuze Wang, Shunsen He, Haoran Li - [[pdf]](https://arxiv.org/pdf/2510.14828)</summary>

**Abstract:** Improving the reasoning capabilities of embodied agents is crucial for robots to complete complex human instructions in long-view manipulation tasks successfully. Despite the success of large language models and vision language models based on Supervised Fine-Tuning (SFT) in planning tasks, they continue facing challenges in performing long-horizon manipulation tasks in complex real-world environments, owing to their restricted common sense and reasoning capabilities. Considering that aligning general-purpose vision language models to robotic planning tasks via supervised fine-tuning suffers from poor generalization and insufficient physical understanding, we propose RoboGPT-R1, a two-stage fine-tuning framework for embodied planning. In this framework, supervised training acquires foundational knowledge through expert sequences, followed by RL to address the model's shortcomings in visual-spatial understanding and reasoning. To achieve physical understanding and action sequence consistency in multi-step reasoning tasks, we design a rule-based reward function that simultaneously considers long-horizon performance and action constraint in the environment. The reasoning model, trained on Qwen2.5-VL-3B, significantly outperforms the larger-scale model, GPT-4o-mini, by 21.33% and surpasses other work trained on Qwen2.5-VL-7B by 20.33% on the EmbodiedBench benchmark.

**arXiv ID:** 2510.14828
</details>

<details>
<summary><strong>Mapping Smarter, Not Harder: A Test-Time Reinforcement Learning Agent That Improves Without Labels or Model Updates</strong> - Wen-Kwang Tsao, Yao-Ching Yu, Chien-Ming Huang - [[pdf]](https://arxiv.org/pdf/2510.14900)</summary>

**Abstract:** The Enterprise Intelligence Platform must integrate logs from numerous third-party vendors in order to perform various downstream tasks. However, vendor documentation is often unavailable at test time. It is either misplaced, mismatched, poorly formatted, or incomplete, which makes schema mapping challenging. We introduce a reinforcement learning agent that can self-improve without labeled examples or model weight updates. During inference, the agent: 1) Identifies ambiguous field-mapping attempts. 2) Generates targeted web-search queries to gather external evidence. 3) Applies a confidence-based reward to iteratively refine its mappings. To demonstrate this concept, we converted Microsoft Defender for Endpoint logs into a common schema. Our method increased mapping accuracy from 56.4\%(LLM-only) to 72.73\%(RAG) to 93.94\% over 100 iterations using GPT-4o. At the same time, it reduced the number of low-confidence mappings requiring expert review by 85\%. This new approach provides an evidence-driven, transparent method for solving future industry problems, paving the way for more robust, accountable, scalable, efficient, flexible, adaptable, and collaborative solutions.

**arXiv ID:** 2510.14900
</details>

<details>
<summary><strong>Agentic Design of Compositional Machines</strong> - Wenqian Zhang, Weiyang Liu, Zhen Liu - [[pdf]](https://arxiv.org/pdf/2510.14980)</summary>

**Abstract:** The design of complex machines stands as both a marker of human intelligence and a foundation of engineering practice. Given recent advances in large language models (LLMs), we ask whether they, too, can learn to create. We approach this question through the lens of compositional machine design: a task in which machines are assembled from standardized components to meet functional demands like locomotion or manipulation in a simulated physical environment. To support this investigation, we introduce BesiegeField, a testbed built on the machine-building game Besiege, which enables part-based construction, physical simulation and reward-driven evaluation. Using BesiegeField, we benchmark state-of-the-art LLMs with agentic workflows and identify key capabilities required for success, including spatial reasoning, strategic assembly, and instruction-following. As current open-source models fall short, we explore reinforcement learning (RL) as a path to improvement: we curate a cold-start dataset, conduct RL finetuning experiments, and highlight open challenges at the intersection of language, machine design, and physical reasoning.

**arXiv ID:** 2510.14980
</details>

<details>
<summary><strong>Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion</strong> - Xiaojian Ding, Lin Zhao, Xian Li, Xiaoying Zhu - [[pdf]](https://arxiv.org/pdf/2510.13887)</summary>

**Abstract:** Incomplete multi-view data, where certain views are entirely missing for some samples, poses significant challenges for traditional multi-view clustering methods. Existing deep incomplete multi-view clustering approaches often rely on static fusion strategies or two-stage pipelines, leading to suboptimal fusion results and error propagation issues. To address these limitations, this paper proposes a novel incomplete multi-view clustering framework based on Hierarchical Semantic Alignment and Cooperative Completion (HSACC). HSACC achieves robust cross-view fusion through a dual-level semantic space design. In the low-level semantic space, consistency alignment is ensured by maximizing mutual information across views. In the high-level semantic space, adaptive view weights are dynamically assigned based on the distributional affinity between individual views and an initial fused representation, followed by weighted fusion to generate a unified global representation. Additionally, HSACC implicitly recovers missing views by projecting aligned latent representations into high-dimensional semantic spaces and jointly optimizes reconstruction and clustering objectives, enabling cooperative learning of completion and clustering. Experimental results demonstrate that HSACC significantly outperforms state-of-the-art methods on five benchmark datasets. Ablation studies validate the effectiveness of the hierarchical alignment and dynamic weighting mechanisms, while parameter analysis confirms the model's robustness to hyperparameter variations.

**arXiv ID:** 2510.13887
</details>

<details>
<summary><strong>Reinforcement Learning for Unsupervised Domain Adaptation in Spatio-Temporal Echocardiography Segmentation</strong> - Arnaud Judge, Nicolas Duchateau, Thierry Judge, Roman A. Sandler, Joseph Z. Sokol, Christian Desrosiers, Olivier Bernard, Pierre-Marc Jodoin - [[pdf]](https://arxiv.org/pdf/2510.14244)</summary>

**Abstract:** Domain adaptation methods aim to bridge the gap between datasets by enabling knowledge transfer across domains, reducing the need for additional expert annotations. However, many approaches struggle with reliability in the target domain, an issue particularly critical in medical image segmentation, where accuracy and anatomical validity are essential. This challenge is further exacerbated in spatio-temporal data, where the lack of temporal consistency can significantly degrade segmentation quality, and particularly in echocardiography, where the presence of artifacts and noise can further hinder segmentation performance. To address these issues, we present RL4Seg3D, an unsupervised domain adaptation framework for 2D + time echocardiography segmentation. RL4Seg3D integrates novel reward functions and a fusion scheme to enhance key landmark precision in its segmentations while processing full-sized input videos. By leveraging reinforcement learning for image segmentation, our approach improves accuracy, anatomical validity, and temporal consistency while also providing, as a beneficial side effect, a robust uncertainty estimator, which can be used at test time to further enhance segmentation performance. We demonstrate the effectiveness of our framework on over 30,000 echocardiographic videos, showing that it outperforms standard domain adaptation techniques without the need for any labels on the target domain. Code is available at this https URL.

**arXiv ID:** 2510.14244
</details>

<details>
<summary><strong>PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering</strong> - Md Mahadi Hasan Nahid, Davood Rafiei - [[pdf]](https://arxiv.org/pdf/2510.14278)</summary>

**Abstract:** Retrieval plays a central role in multi-hop question answering (QA), where answering complex questions requires gathering multiple pieces of evidence. We introduce an Agentic Retrieval System that leverages large language models (LLMs) in a structured loop to retrieve relevant evidence with high precision and recall. Our framework consists of three specialized agents: a Question Analyzer that decomposes a multi-hop question into sub-questions, a Selector that identifies the most relevant context for each sub-question (focusing on precision), and an Adder that brings in any missing evidence (focusing on recall). The iterative interaction between Selector and Adder yields a compact yet comprehensive set of supporting passages. In particular, it achieves higher retrieval accuracy while filtering out distracting content, enabling downstream QA models to surpass full-context answer accuracy while relying on significantly less irrelevant information. Experiments on four multi-hop QA benchmarks -- HotpotQA, 2WikiMultiHopQA, MuSiQue, and MultiHopRAG -- demonstrates that our approach consistently outperforms strong baselines.

**arXiv ID:** 2510.14278
</details>

<details>
<summary><strong>Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following</strong> - Qingyu Ren, Qianyu He, Bowei Zhang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu - [[pdf]](https://arxiv.org/pdf/2510.14420)</summary>

**Abstract:** Language models often struggle to follow multi-constraint instructions that are crucial for real-world applications. Existing reinforcement learning (RL) approaches suffer from dependency on external supervision and sparse reward signals from multi-constraint tasks. We propose a label-free self-supervised RL framework that eliminates dependency on external supervision by deriving reward signals directly from instructions and generating pseudo-labels for reward model training. Our approach introduces constraint decomposition strategies and efficient constraint-wise binary classification to address sparse reward challenges while maintaining computational efficiency. Experiments show that our approach generalizes well, achieving strong improvements across 3 in-domain and 5 out-of-domain datasets, including challenging agentic and multi-turn instruction following. The data and code are publicly available at this https URL

**arXiv ID:** 2510.14420
</details>

<details>
<summary><strong>Agentic Entropy-Balanced Policy Optimization</strong> - Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2510.14545)</summary>

**Abstract:** Recently, Agentic Reinforcement Learning (Agentic RL) has made significant progress in incentivizing the multi-turn, long-horizon tool-use capabilities of web agents. While mainstream agentic RL algorithms autonomously explore high-uncertainty tool-call steps under the guidance of entropy, excessive reliance on entropy signals can impose further constraints, leading to the training collapse. In this paper, we delve into the challenges caused by entropy and propose the Agentic Entropy-Balanced Policy Optimization (AEPO), an agentic RL algorithm designed to balance entropy in both the rollout and policy update phases. AEPO comprises two core components: (1) a dynamic entropy-balanced rollout mechanism that adaptively allocate global and branch sampling budget through entropy pre-monitoring, while imposing a branch penalty on consecutive high-entropy tool-call steps to prevent over-branching issues; and (2) Entropy-Balanced Policy Optimization that inserts a stop-gradient operation into the high-entropy clipping term to preserve and properly rescale gradients on high-entropy tokens, while incorporating entropy-aware advantage estimation to prioritize learning on high-uncertainty tokens. Results across 14 challenging datasets show that AEPO consistently outperforms 7 mainstream RL algorithms. With just 1K RL samples, Qwen3-14B with AEPO achieves impressive results: 47.6% on GAIA, 11.2% on Humanity's Last Exam, and 43.0% on WebWalker for Pass@1; 65.0% on GAIA, 26.0% on Humanity's Last Exam, and 70.0% on WebWalker for Pass@5. Further analysis reveals that AEPO improves rollout sampling diversity while maintaining stable policy entropy, facilitating scalable web agent training.

**arXiv ID:** 2510.14545
</details>

<details>
<summary><strong>The Bidding Games: Reinforcement Learning for MEV Extraction on Polygon Blockchain</strong> - Andrei Seoev, Leonid Gremyachikh, Anastasiia Smirnova, Yash Madhwal, Alisa Kalacheva, Dmitry Belousov, Ilia Zubov, Aleksei Smirnov, Denis Fedyanin, Vladimir Gorgadze, Yury Yanovich - [[pdf]](https://arxiv.org/pdf/2510.14642)</summary>

**Abstract:** In blockchain networks, the strategic ordering of transactions within blocks has emerged as a significant source of profit extraction, known as Maximal Extractable Value (MEV). The transition from spam-based Priority Gas Auctions to structured auction mechanisms like Polygon Atlas has transformed MEV extraction from public bidding wars into sealed-bid competitions under extreme time constraints. While this shift reduces network congestion, it introduces complex strategic challenges where searchers must make optimal bidding decisions within a sub-second window without knowledge of competitor behavior or presence. Traditional game-theoretic approaches struggle in this high-frequency, partially observable environment due to their reliance on complete information and static equilibrium assumptions. We present a reinforcement learning framework for MEV extraction on Polygon Atlas and make three contributions: (1) A novel simulation environment that accurately models the stochastic arrival of arbitrage opportunities and probabilistic competition in Atlas auctions; (2) A PPO-based bidding agent optimized for real-time constraints, capable of adaptive strategy formulation in continuous action spaces while maintaining production-ready inference speeds; (3) Empirical validation demonstrating our history-conditioned agent captures 49\% of available profits when deployed alongside existing searchers and 81\% when replacing the market leader, significantly outperforming static bidding strategies. Our work establishes that reinforcement learning provides a critical advantage in high-frequency MEV environments where traditional optimization methods fail, offering immediate value for industrial participants and protocol designers alike.

**arXiv ID:** 2510.14642
</details>

<details>
<summary><strong>RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning</strong> - Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, Huazhe Xu - [[pdf]](https://arxiv.org/pdf/2510.14830)</summary>

**Abstract:** Real-world robotic manipulation in homes and factories demands reliability, efficiency, and robustness that approach or surpass skilled human operators. We present RL-100, a real-world reinforcement learning training framework built on diffusion visuomotor policies trained bu supervised learning. RL-100 introduces a three-stage pipeline. First, imitation learning leverages human priors. Second, iterative offline reinforcement learning uses an Offline Policy Evaluation procedure, abbreviated OPE, to gate PPO-style updates that are applied in the denoising process for conservative and reliable improvement. Third, online reinforcement learning eliminates residual failure modes. An additional lightweight consistency distillation head compresses the multi-step sampling process in diffusion into a single-step policy, enabling high-frequency control with an order-of-magnitude reduction in latency while preserving task performance. The framework is task-, embodiment-, and representation-agnostic and supports both 3D point clouds and 2D RGB inputs, a variety of robot platforms, and both single-step and action-chunk policies. We evaluate RL-100 on seven real-robot tasks spanning dynamic rigid-body control, such as Push-T and Agile Bowling, fluids and granular pouring, deformable cloth folding, precise dexterous unscrewing, and multi-stage orange juicing. RL-100 attains 100\% success across evaluated trials for a total of 900 out of 900 episodes, including up to 250 out of 250 consecutive trials on one task. The method achieves near-human teleoperation or better time efficiency and demonstrates multi-hour robustness with uninterrupted operation lasting up to two hours.

**arXiv ID:** 2510.14830
</details>

<details>
<summary><strong>LaSeR: Reinforcement Learning with Last-Token Self-Rewarding</strong> - Wenkai Yang, Weijie Liu, Ruobing Xie, Yiju Guo, Lulu Wu, Saiyong Yang, Yankai Lin - [[pdf]](https://arxiv.org/pdf/2510.14943)</summary>

**Abstract:** Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a core paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). To address the lack of verification signals at test time, prior studies incorporate the training of model's self-verification capability into the standard RLVR process, thereby unifying reasoning and verification capabilities within a single LLM. However, previous practice requires the LLM to sequentially generate solutions and self-verifications using two separate prompt templates, which significantly reduces efficiency. In this work, we theoretically reveal that the closed-form solution to the RL objective of self-verification can be reduced to a remarkably simple form: the true reasoning reward of a solution is equal to its last-token self-rewarding score, which is computed as the difference between the policy model's next-token log-probability assigned to any pre-specified token at the solution's last token and a pre-calculated constant, scaled by the KL coefficient. Based on this insight, we propose LaSeR (Reinforcement Learning with Last-Token Self-Rewarding), an algorithm that simply augments the original RLVR loss with a MSE loss that aligns the last-token self-rewarding scores with verifier-based reasoning rewards, jointly optimizing the reasoning and self-rewarding capabilities of LLMs. The optimized self-rewarding scores can be utilized in both training and testing to enhance model performance. Notably, our algorithm derives these scores from the predicted next-token probability distribution of the last token immediately after generation, incurring only the minimal extra cost of one additional token inference. Experiments show that our method not only improves the model's reasoning performance but also equips it with remarkable self-rewarding capability, thereby boosting its inference-time scaling performance.

**arXiv ID:** 2510.14943
</details>

<details>
<summary><strong>CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions</strong> - Lizhi Yang, Blake Werner, Massimiliano de Sa Aaron D. Ames - [[pdf]](https://arxiv.org/pdf/2510.14959)</summary>

**Abstract:** Reinforcement learning (RL), while powerful and expressive, can often prioritize performance at the expense of safety. Yet safety violations can lead to catastrophic outcomes in real-world deployments. Control Barrier Functions (CBFs) offer a principled method to enforce dynamic safety -- traditionally deployed \emph{online} via safety filters. While the result is safe behavior, the fact that the RL policy does not have knowledge of the CBF can lead to conservative behaviors. This paper proposes CBF-RL, a framework for generating safe behaviors with RL by enforcing CBFs \emph{in training}. CBF-RL has two key attributes: (1) minimally modifying a nominal RL policy to encode safety constraints via a CBF term, (2) and safety filtering of the policy rollouts in training. Theoretically, we prove that continuous-time safety filters can be deployed via closed-form expressions on discrete-time roll-outs. Practically, we demonstrate that CBF-RL internalizes the safety constraints in the learned policy -- both enforcing safer actions and biasing towards safer rewards -- enabling safe deployment without the need for an online safety filter. We validate our framework through ablation studies on navigation tasks and on the Unitree G1 humanoid robot, where CBF-RL enables safer exploration, faster convergence, and robust performance under uncertainty, enabling the humanoid robot to avoid obstacles and climb stairs safely in real-world settings without a runtime safety filter.

**arXiv ID:** 2510.14959
</details>

<details>
<summary><strong>RLSR: Reinforcement Learning with Supervised Reward Outperforms SFT in Instruction Following</strong> - Zhichao Wang, Andy Wong, Ruslan Belkin - [[pdf]](https://arxiv.org/pdf/2510.14200)</summary>

**Abstract:** After the pretraining stage of LLMs, techniques such as SFT, RLHF, RLVR, and RFT are applied to enhance instruction-following ability, mitigate undesired responses, improve reasoning capability and enable efficient domain adaptation with minimal data. SFT relies on the next-token prediction objective to strengthen instruction following in a base model using a large corpus of human-labeled responses. In contrast, RFT employs a RL-based approach to adapt fine-tuned reasoning models to specific domains with limited supervision. Inspired by RFT, we propose replacing SFT with RLSR to leverage the extensive SFT dataset in an RL framework, thereby improving the base model's instruction-following ability. In RLSR, the base model generates multiple responses for each prompt, and reward scores are computed as the cosine similarity in the semantic embedding space between the generated and human-labeled responses. RLSR can be utilized in multiple ways. It can directly replace SFT, achieving superior performance on instruction-following benchmarks-for example, RLSR (SB) on Qwen-7B (INFINITY) achieved an AlpacaEval win rate of 26.34%, surpassing SFT's 21.01%. Furthermore, combining SFT and RLSR further enhances downstream task performance; Qwen-7B (INFINITY) achieved a win rate of 30.73% when trained with SFT + RLSR.

**arXiv ID:** 2510.14200
</details>

<details>
<summary><strong>Explore to Evolve: Scaling Evolved Aggregation Logic via Proactive Online Exploration for Deep Research Agents</strong> - Rui Wang, Ce Zhang, Jun-Yu Ma, Jianshu Zhang, Hongru Wang, Yi Chen, Boyang Xue, Tianqing Fang, Zhisong Zhang, Hongming Zhang, Haitao Mi, Dong Yu, Kam-Fai Wong - [[pdf]](https://arxiv.org/pdf/2510.14438)</summary>

**Abstract:** Deep research web agents not only retrieve information from diverse sources such as web environments, files, and multimodal inputs, but more importantly, they need to rigorously analyze and aggregate knowledge for insightful research. However, existing open-source deep research agents predominantly focus on enhancing information-seeking capabilities of web agents to locate specific information, while overlooking the essential need for information aggregation, which would limit their ability to support in-depth research. We propose an Explore to Evolve paradigm to scalably construct verifiable training data for web agents. Begins with proactive online exploration, an agent sources grounded information by exploring the real web. Using the collected evidence, the agent then self-evolves an aggregation program by selecting, composing, and refining operations from 12 high-level logical types to synthesize a verifiable QA pair. This evolution from high-level guidance to concrete operations allowed us to scalably produce WebAggregatorQA, a dataset of 10K samples across 50K websites and 11 domains. Based on an open-source agent framework, SmolAgents, we collect supervised fine-tuning trajectories to develop a series of foundation models, WebAggregator. WebAggregator-8B matches the performance of GPT-4.1, while the 32B variant surpasses GPT-4.1 by more than 10% on GAIA-text and closely approaches Claude-3.7-sonnet. Moreover, given the limited availability of benchmarks that evaluate web agents' information aggregation abilities, we construct a human-annotated evaluation split of WebAggregatorQA as a challenging test set. On this benchmark, Claude-3.7-sonnet only achieves 28%, and GPT-4.1 scores 25.8%. Even when agents manage to retrieve all references, they still struggle on WebAggregatorQA, highlighting the need to strengthen the information aggregation capabilities of web agent foundations.

**arXiv ID:** 2510.14438
</details>

<details>
<summary><strong>Natural Language Tools: A Natural Language Approach to Tool Calling In Large Language Agents</strong> - Reid T. Johnson, Michelle D. Pain, Jordan D. West - [[pdf]](https://arxiv.org/pdf/2510.14453)</summary>

**Abstract:** We present Natural Language Tools (NLT), a framework that replaces programmatic JSON tool calling in large language models (LLMs) with natural language outputs. By decoupling tool selection from response generation, NLT eliminates task interference and format constraints that degrade tool call performance. When evaluated across 10 models and 6,400 trials spanning customer service and mental health domains, NLT improves tool calling accuracy by 18.4 percentage points while reducing output variance by 70%. Open-weight models see the largest gains, surpassing flagship closed-weight alternatives, with implications for model training in both reinforcement learning and supervised fine-tuning stages. These improvements persist under prompt perturbations and extend tool-calling capabilities to models lacking native support.

**arXiv ID:** 2510.14453
</details>

<details>
<summary><strong>Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning</strong> - Liang Chen, Xueting Han, Li Shen, Jing Bai, Kam-Fai Wong - [[pdf]](https://arxiv.org/pdf/2509.06948)</summary>

**Abstract:** Reinforcement learning (RL) has proven effective in incentivizing the reasoning abilities of large language models (LLMs), but suffers from severe efficiency challenges due to its trial-and-error nature. While the common practice employs supervised fine-tuning (SFT) as a warm-up stage for RL, this decoupled two-stage approach suffers from catastrophic forgetting: second-stage RL gradually loses SFT-acquired behaviors and inefficiently explores new patterns. This study introduces a novel method for learning reasoning models that employs bilevel optimization to facilitate better cooperation between these training paradigms. By conditioning the SFT objective on the optimal RL policy, our approach enables SFT to meta-learn how to guide RL's optimization process. During training, the lower level performs RL updates while simultaneously receiving SFT supervision, and the upper level explicitly maximizes the cooperative gain-the performance advantage of joint SFT-RL training over RL alone. Empirical evaluations on five reasoning benchmarks demonstrate that our method consistently outperforms baselines and achieves a better balance between effectiveness and efficiency.

**arXiv ID:** 2509.06948
</details>

<details>
<summary><strong>A$^2$FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning</strong> - Qianben Chen, Jingyi Cao, Jiayu Zhang, Tianrui Qin, Xiaowan Li, King Zhu, Dingfeng Shi, He Zhu, Minghao Liu, Xiaobo Liang, Xin Gui, Ge Zhang, Jian Yang, Yuchen Eleanor Jiang, Wangchunshu Zhou - [[pdf]](https://arxiv.org/pdf/2510.12838)</summary>

**Abstract:** Large language models split into two families: reasoning-centric LLMs, which strengthen internal chain-of-thought reasoning but cannot invoke external tools, and agentic LLMs, which learn to interact with environments and leverage tools but often lag in deep reasoning. This divide arises from fundamentally different training objectives, leading to mismatched strengths and inefficiency on simple queries, where both families tend to overthink or over-call tools. In this work, we present Adaptive Agent Foundation Model (A$^2$FM), a unified framework that follows a route-then-align principle: the model first learns task-aware routing and then aligns mode-specific trajectories under a shared backbone. To address the inefficiency gap, we introduce a third mode-instant-that handles simple queries directly, preventing unnecessary reasoning or tool calls while complementing the agentic and reasoning modes. To jointly enhance accuracy and efficiency, we propose Adaptive Policy Optimization (APO), which enforces adaptive sampling across modes and applies a cost-regularized reward. On the 32B scale, A$^2$FM achieves 13.4% on BrowseComp, 70.4% on AIME25, and 16.7% on HLE, setting new SOTA among comparable models and performing competitively with frontier LLMs across agentic, reasoning, and general benchmarks. Notably, the adaptive execution achieves a cost of pass of only $0.00487 per correct answer-cutting cost by 45.2% relative to reasoning and 33.5% relative to agentic, thus delivering substantially higher cost efficiency while maintaining comparable accuracy.

**arXiv ID:** 2510.12838
</details>

<details>
<summary><strong>Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning</strong> - Xiaoyun Zhang, Xiaojian Yuan, Di Huang, Wang You, Chen Hu, Jingqing Ruan, Kejiang Chen, Xing Hu - [[pdf]](https://arxiv.org/pdf/2510.10959)</summary>

**Abstract:** Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability.

**arXiv ID:** 2510.10959
</details>

<details>
<summary><strong>Active Measuring in Reinforcement Learning With Delayed Negative Effects</strong> - Daiqi Gao, Ziping Xu, Aseel Rawashdeh, Predrag Klasnja, Susan A. Murphy - [[pdf]](https://arxiv.org/pdf/2510.14315)</summary>

**Abstract:** Measuring states in reinforcement learning (RL) can be costly in real-world settings and may negatively influence future outcomes. We introduce the Actively Observable Markov Decision Process (AOMDP), where an agent not only selects control actions but also decides whether to measure the latent state. The measurement action reveals the true latent state but may have a negative delayed effect on the environment. We show that this reduced uncertainty may provably improve sample efficiency and increase the value of the optimal policy despite these costs. We formulate an AOMDP as a periodic partially observable MDP and propose an online RL algorithm based on belief states. To approximate the belief states, we further propose a sequential Monte Carlo method to jointly approximate the posterior of unknown static environment parameters and unobserved latent states. We evaluate the proposed algorithm in a digital health application, where the agent decides when to deliver digital interventions and when to assess users' health status through surveys.

**arXiv ID:** 2510.14315
</details>

<details>
<summary><strong>Learning to Undo: Rollback-Augmented Reinforcement Learning with Reversibility Signals</strong> - Andrejs Sorstkins, Omer Tariq, Muhammad Bilal - [[pdf]](https://arxiv.org/pdf/2510.14503)</summary>

**Abstract:** This paper proposes a reversible learning framework to improve the robustness and efficiency of value based Reinforcement Learning agents, addressing vulnerability to value overestimation and instability in partially irreversible environments. The framework has two complementary core mechanisms: an empirically derived transition reversibility measure called Phi of s and a, and a selective state rollback operation. We introduce an online per state action estimator called Phi that quantifies the likelihood of returning to a prior state within a fixed horizon K. This measure is used to adjust the penalty term during temporal difference updates dynamically, integrating reversibility awareness directly into the value function. The system also includes a selective rollback operator. When an action yields an expected return markedly lower than its instantaneous estimated value and violates a predefined threshold, the agent is penalized and returns to the preceding state rather than progressing. This interrupts sub optimal high risk trajectories and avoids catastrophic steps. By combining reversibility aware evaluation with targeted rollback, the method improves safety, performance, and stability. In the CliffWalking v0 domain, the framework reduced catastrophic falls by over 99.8 percent and yielded a 55 percent increase in mean episode return. In the Taxi v3 domain, it suppressed illegal actions by greater than or equal to 99.9 percent and achieved a 65.7 percent improvement in cumulative reward, while also sharply reducing reward variance in both environments. Ablation studies confirm that the rollback mechanism is the critical component underlying these safety and performance gains, marking a robust step toward safe and reliable sequential decision making.

**arXiv ID:** 2510.14503
</details>

<details>
<summary><strong>Reinforcement Learning with Stochastic Reward Machines</strong> - Jan Corazza, Ivan Gavran, Daniel Neider - [[pdf]](https://arxiv.org/pdf/2510.14837)</summary>

**Abstract:** Reward machines are an established tool for dealing with reinforcement learning problems in which rewards are sparse and depend on complex sequences of actions. However, existing algorithms for learning reward machines assume an overly idealized setting where rewards have to be free of noise. To overcome this practical limitation, we introduce a novel type of reward machines, called stochastic reward machines, and an algorithm for learning them. Our algorithm, based on constraint solving, learns minimal stochastic reward machines from the explorations of a reinforcement learning agent. This algorithm can easily be paired with existing reinforcement learning algorithms for reward machines and guarantees to converge to an optimal policy in the limit. We demonstrate the effectiveness of our algorithm in two case studies and show that it outperforms both existing methods and a naive approach for handling noisy reward functions.

**arXiv ID:** 2510.14837
</details>

<details>
<summary><strong>Offline Reinforcement Learning via Inverse Optimization</strong> - Ioannis Dimanidis, Tolga Ok, Peyman Mohajerin Esfahani - [[pdf]](https://arxiv.org/pdf/2502.20030)</summary>

**Abstract:** Inspired by the recent successes of Inverse Optimization (IO) across various application domains, we propose a novel offline Reinforcement Learning (ORL) algorithm for continuous state and action spaces, leveraging the convex loss function called ``sub-optimality loss" from the IO literature. To mitigate the distribution shift commonly observed in ORL problems, we further employ a robust and non-causal Model Predictive Control (MPC) expert steering a nominal model of the dynamics using in-hindsight information stemming from the model mismatch. Unlike the existing literature, our robust MPC expert enjoys an exact and tractable convex reformulation. In the second part of this study, we show that the IO hypothesis class, trained by the proposed convex loss function, enjoys ample expressiveness and achieves competitive performance comparing with the state-of-the-art (SOTA) methods in the low-data regime of the MuJoCo benchmark while utilizing three orders of magnitude fewer parameters, thereby requiring significantly fewer computational resources. To facilitate the reproducibility of our results, we provide an open-source package implementing the proposed algorithms and the experiments.

**arXiv ID:** 2502.20030
</details>

<details>
<summary><strong>Strategyproof Reinforcement Learning from Human Feedback</strong> - Thomas Kleine Buening, Jiarui Gan, Debmalya Mandal, Marta Kwiatkowska - [[pdf]](https://arxiv.org/pdf/2503.09561)</summary>

**Abstract:** We study Reinforcement Learning from Human Feedback (RLHF) in settings where multiple labelers may strategically misreport feedback to steer the learned policy toward their own preferences. We show that existing RLHF algorithms, including recent pluralistic methods, are not strategyproof, and that even a single strategic labeler can cause arbitrarily large misalignment with social welfare. Moreover, we prove that, in the worst case, any strategyproof RLHF algorithm must perform $k$-times worse than the optimal policy, where $k$ is the number of labelers. This suggests a fundamental trade-off between incentive alignment (ensuring labelers report truthfully) and policy alignment (maximizing social welfare). To address this, we propose the Pessimistic Median of MLEs algorithm, which, under appropriate policy coverage assumptions, is approximately strategyproof and converges to the optimal policy as the number of labelers and samples increases. Our results apply to both contextual bandits and Markov decision processes.

**arXiv ID:** 2503.09561
</details>

<details>
<summary><strong>On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification</strong> - Yongliang Wu, Yizhou Zhou, Zhou Ziheng, Yingzhe Peng, Xinyu Ye, Xinting Hu, Wenbo Zhu, Lu Qi, Ming-Hsuan Yang, Xu Yang - [[pdf]](https://arxiv.org/pdf/2508.05629)</summary>

**Abstract:** We present a simple yet theoretically motivated improvement to Supervised Fine-Tuning (SFT) for the Large Language Model (LLM), addressing its limited generalization compared to reinforcement learning (RL). Through mathematical analysis, we reveal that standard SFT gradients implicitly encode a problematic reward structure that may severely restrict the generalization capabilities of model. To rectify this, we propose Dynamic Fine-Tuning (DFT), stabilizing gradient updates for each token by dynamically rescaling the objective function with the probability of this token. Remarkably, this single-line code change significantly outperforms standard SFT across multiple challenging benchmarks and base models, demonstrating greatly improved generalization. Additionally, our approach shows competitive results in offline RL settings, offering an effective yet simpler alternative. This work bridges theoretical insight and practical solutions, substantially advancing SFT performance. The code will be available at this https URL.

**arXiv ID:** 2508.05629
</details>

<details>
<summary><strong>A Diffusion-Refined Planner with Reinforcement Learning Priors for Confined-Space Parking</strong> - Mingyang Jiang, Yueyuan Li, Jiaru Zhang, Songan Zhang, Ming Yang - [[pdf]](https://arxiv.org/pdf/2510.14000)</summary>

**Abstract:** The growing demand for parking has increased the need for automated parking planning methods that can operate reliably in confined spaces. In restricted and complex environments, high-precision maneuvers are required to achieve a high success rate in planning, yet existing approaches often rely on explicit action modeling, which faces challenges when accurately modeling the optimal action distribution. In this paper, we propose DRIP, a diffusion-refined planner anchored in reinforcement learning (RL) prior action distribution, in which an RL-pretrained policy provides prior action distributions to regularize the diffusion training process. During the inference phase the denoising process refines these coarse priors into more precise action distributions. By steering the denoising trajectory through the reinforcement learning prior distribution during training, the diffusion model inherits a well-informed initialization, resulting in more accurate action modeling, a higher planning success rate, and reduced inference steps. We evaluate our approach across parking scenarios with varying degrees of spatial constraints. Experimental results demonstrate that our method significantly improves planning performance in confined-space parking environments while maintaining strong generalization in common scenarios.

**arXiv ID:** 2510.14000
</details>

<details>
<summary><strong>Optimistic Reinforcement Learning-Based Skill Insertions for Task and Motion Planning</strong> - Gaoyuan Liu, Joris de Winter, Yuri Durodie, Denis Steckelmacher, Ann Nowe, Bram Vanderborght - [[pdf]](https://arxiv.org/pdf/2510.14065)</summary>

**Abstract:** Task and motion planning (TAMP) for robotics manipulation necessitates long-horizon reasoning involving versatile actions and skills. While deterministic actions can be crafted by sampling or optimizing with certain constraints, planning actions with uncertainty, i.e., probabilistic actions, remains a challenge for TAMP. On the contrary, Reinforcement Learning (RL) excels in acquiring versatile, yet short-horizon, manipulation skills that are robust with uncertainties. In this letter, we design a method that integrates RL skills into TAMP pipelines. Besides the policy, a RL skill is defined with data-driven logical components that enable the skill to be deployed by symbolic planning. A plan refinement sub-routine is designed to further tackle the inevitable effect uncertainties. In the experiments, we compare our method with baseline hierarchical planning from both TAMP and RL fields and illustrate the strength of the method. The results show that by embedding RL skills, we extend the capability of TAMP to domains with probabilistic skills, and improve the planning efficiency compared to the previous methods.

**arXiv ID:** 2510.14065
</details>

<details>
<summary><strong>Risk-Aware Reinforcement Learning with Bandit-Based Adaptation for Quadrupedal Locomotion</strong> - Yuanhong Zeng, Anushri Dixit - [[pdf]](https://arxiv.org/pdf/2510.14338)</summary>

**Abstract:** In this work, we study risk-aware reinforcement learning for quadrupedal locomotion. Our approach trains a family of risk-conditioned policies using a Conditional Value-at-Risk (CVaR) constrained policy optimization technique that provides improved stability and sample efficiency. At deployment, we adaptively select the best performing policy from the family of policies using a multi-armed bandit framework that uses only observed episodic returns, without any privileged environment information, and adapts to unknown conditions on the fly. Hence, we train quadrupedal locomotion policies at various levels of robustness using CVaR and adaptively select the desired level of robustness online to ensure performance in unknown environments. We evaluate our method in simulation across eight unseen settings (by changing dynamics, contacts, sensing noise, and terrain) and on a Unitree Go2 robot in previously unseen terrains. Our risk-aware policy attains nearly twice the mean and tail performance in unseen environments compared to other baselines and our bandit-based adaptation selects the best-performing risk-aware policy in unknown terrain within two minutes of operation.

**arXiv ID:** 2510.14338
</details>

<details>
<summary><strong>SkyDreamer: Interpretable End-to-End Vision-Based Drone Racing with Model-Based Reinforcement Learning</strong> - Aderik Verraest, Stavrow Bahnam, Robin Ferede, Guido de Croon, Christophe De Wagter - [[pdf]](https://arxiv.org/pdf/2510.14783)</summary>

**Abstract:** Autonomous drone racing (ADR) systems have recently achieved champion-level performance, yet remain highly specific to drone racing. While end-to-end vision-based methods promise broader applicability, no system to date simultaneously achieves full sim-to-real transfer, onboard execution, and champion-level performance. In this work, we present SkyDreamer, to the best of our knowledge, the first end-to-end vision-based ADR policy that maps directly from pixel-level representations to motor commands. SkyDreamer builds on informed Dreamer, a model-based reinforcement learning approach where the world model decodes to privileged information only available during training. By extending this concept to end-to-end vision-based ADR, the world model effectively functions as an implicit state and parameter estimator, greatly improving interpretability. SkyDreamer runs fully onboard without external aid, resolves visual ambiguities by tracking progress using the state decoded from the world model's hidden state, and requires no extrinsic camera calibration, enabling rapid deployment across different drones without retraining. Real-world experiments show that SkyDreamer achieves robust, high-speed flight, executing tight maneuvers such as an inverted loop, a split-S and a ladder, reaching speeds of up to 21 m/s and accelerations of up to 6 g. It further demonstrates a non-trivial visual sim-to-real transfer by operating on poor-quality segmentation masks, and exhibits robustness to battery depletion by accurately estimating the maximum attainable motor RPM and adjusting its flight path in real-time. These results highlight SkyDreamer's adaptability to important aspects of the reality gap, bringing robustness while still achieving extremely high-speed, agile flight.

**arXiv ID:** 2510.14783
</details>

<details>
<summary><strong>MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control</strong> - Xue Bin Peng - [[pdf]](https://arxiv.org/pdf/2510.13794)</summary>

**Abstract:** MimicKit is an open-source framework for training motion controllers using motion imitation and reinforcement learning. The codebase provides implementations of commonly-used motion-imitation techniques and RL algorithms. This framework is intended to support research and applications in computer graphics and robotics by providing a unified training framework, along with standardized environment, agent, and data structures. The codebase is designed to be modular and easily configurable, enabling convenient modification and extension to new characters and tasks. The open-source codebase is available at: this https URL.

**arXiv ID:** 2510.13794
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
