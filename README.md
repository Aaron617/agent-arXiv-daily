# Agent arXiv Daily

**Last Updated:** 2025-11-05 02:49:29

**Total Papers:** 71

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (1 papers)</h2></summary>

<details>
<summary><strong>Towards Stable and Personalised Profiles for Lexical Alignment in Spoken Human-Agent Dialogue</strong> - Keara Schaaij, Roel Boumans, Tibor Bosse, Iris Hendrickx - [[pdf]](https://arxiv.org/pdf/2509.04104)</summary>

**Abstract:** Lexical alignment, where speakers start to use similar words across conversation, is known to contribute to successful communication. However, its implementation in conversational agents remains underexplored, particularly considering the recent advancements in large language models (LLMs). As a first step towards enabling lexical alignment in human-agent dialogue, this study draws on strategies for personalising conversational agents and investigates the construction of stable, personalised lexical profiles as a basis for lexical alignment. Specifically, we varied the amounts of transcribed spoken data used for construction as well as the number of items included in the profiles per part-of-speech (POS) category and evaluated profile performance across time using recall, coverage, and cosine similarity metrics. It was shown that smaller and more compact profiles, created after 10 min of transcribed speech containing 5 items for adjectives, 5 items for conjunctions, and 10 items for adverbs, nouns, pronouns, and verbs each, offered the best balance in both performance and data efficiency. In conclusion, this study offers practical insights into constructing stable, personalised lexical profiles, taking into account minimal data requirements, serving as a foundational step toward lexical alignment strategies in conversational agents.

**arXiv ID:** 2509.04104
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (7 papers)</h2></summary>

<details>
<summary><strong>Optimizing AI Agent Attacks With Synthetic Data</strong> - Chloe Loughridge, Paul Colognese, Avery Griffin, Tyler Tracy, Jon Kutasov, Joe Benton - [[pdf]](https://arxiv.org/pdf/2511.02823)</summary>

**Abstract:** As AI deployments become more complex and high-stakes, it becomes increasingly important to be able to estimate their risk. AI control is one framework for doing so. However, good control evaluations require eliciting strong attack policies. This can be challenging in complex agentic environments where compute constraints leave us data-poor. In this work, we show how to optimize attack policies in SHADE-Arena, a dataset of diverse realistic control environments. We do this by decomposing attack capability into five constituent skills -- suspicion modeling, attack selection, plan synthesis, execution and subtlety -- and optimizing each component individually. To get around the constraint of limited data, we develop a probabilistic model of attack dynamics, optimize our attack hyperparameters using this simulation, and then show that the results transfer to SHADE-Arena. This results in a substantial improvement in attack strength, reducing safety score from a baseline of 0.87 to 0.41 using our scaffold.

**arXiv ID:** 2511.02823
</details>

<details>
<summary><strong>Kosmos: An AI Scientist for Autonomous Discovery</strong> - Ludovico Mitchener, Angela Yiu, Benjamin Chang, Mathieu Bourdenx, Tyler Nadolski, Arvis Sulovari, Eric C. Landsness, Daniel L. Barabasi, Siddharth Narayanan, Nicky Evans, Shriya Reddy, Martha Foiani, Aizad Kamal, Leah P. Shriver, Fang Cao, Asmamaw T. Wassie, Jon M. Laurent, Edwin Melville-Green, Mayk Caldas, Albert Bou, Kaleigh F. Roberts, Sladjana Zagorac, Timothy C. Orr, Miranda E. Orr, Kevin J. Zwezdaryk, Ali E. Ghareeb, Laurie McCoy, Bruna Gomes, Euan A. Ashley, Karen E. Duff, Tonio Buonassisi, Tom Rainforth, Randall J. Bateman, Michael Skarlinski, Samuel G. Rodriques, Michaela M. Hinks, Andrew D. White - [[pdf]](https://arxiv.org/pdf/2511.02824)</summary>

**Abstract:** Data-driven scientific discovery requires iterative cycles of literature search, hypothesis generation, and data analysis. Substantial progress has been made towards AI agents that can automate scientific research, but all such agents remain limited in the number of actions they can take before losing coherence, thus limiting the depth of their findings. Here we present Kosmos, an AI scientist that automates data-driven discovery. Given an open-ended objective and a dataset, Kosmos runs for up to 12 hours performing cycles of parallel data analysis, literature search, and hypothesis generation before synthesizing discoveries into scientific reports. Unlike prior systems, Kosmos uses a structured world model to share information between a data analysis agent and a literature search agent. The world model enables Kosmos to coherently pursue the specified objective over 200 agent rollouts, collectively executing an average of 42,000 lines of code and reading 1,500 papers per run. Kosmos cites all statements in its reports with code or primary literature, ensuring its reasoning is traceable. Independent scientists found 79.4% of statements in Kosmos reports to be accurate, and collaborators reported that a single 20-cycle Kosmos run performed the equivalent of 6 months of their own research time on average. Furthermore, collaborators reported that the number of valuable scientific findings generated scales linearly with Kosmos cycles (tested up to 20 cycles). We highlight seven discoveries made by Kosmos that span metabolomics, materials science, neuroscience, and statistical genetics. Three discoveries independently reproduce findings from preprinted or unpublished manuscripts that were not accessed by Kosmos at runtime, while four make novel contributions to the scientific literature.

**arXiv ID:** 2511.02824
</details>

<details>
<summary><strong>1 PoCo: Agentic Proof-of-Concept Exploit Generation for Smart Contracts</strong> - Vivi Andersson, Sofia Bobadilla, Harald Hobbelhagen, Martin Monperrus - [[pdf]](https://arxiv.org/pdf/2511.02780)</summary>

**Abstract:** Smart contracts operate in a highly adversarial environment, where vulnerabilities can lead to substantial financial losses. Thus, smart contracts are subject to security audits. In auditing, proof-of-concept (PoC) exploits play a critical role by demonstrating to the stakeholders that the reported vulnerabilities are genuine, reproducible, and actionable. However, manually creating PoCs is time-consuming, error-prone, and often constrained by tight audit schedules. We introduce POCO, an agentic framework that automatically generates executable PoC exploits from natural-language vulnerability descriptions written by auditors. POCO autonomously generates PoC exploits in an agentic manner by interacting with a set of code-execution tools in a Reason-Act-Observe loop. It produces fully executable exploits compatible with the Foundry testing framework, ready for integration into audit reports and other security tools. We evaluate POCO on a dataset of 23 real-world vulnerability reports. POCO consistently outperforms the prompting and workflow baselines, generating well-formed and logically correct PoCs. Our results demonstrate that agentic frameworks can significantly reduce the effort required for high-quality PoCs in smart contract audits. Our contribution provides readily actionable knowledge for the smart contract security community.

**arXiv ID:** 2511.02780
</details>

<details>
<summary><strong>Understanding and Optimizing Agentic Workflows via Shapley value</strong> - Yingxuan Yang, Bo Huang, Siyuan Qi, Chao Feng, Haoyi Hu, Yuxuan Zhu, Jinbo Hu, Haoran Zhao, Ziyi He, Xiao Liu, Muning Wen, Zongyu Wang, Lin Qiu, Xuezhi Cao, Xunliang Cai, Yong Yu, Weinan Zhang - [[pdf]](https://arxiv.org/pdf/2502.00510)</summary>

**Abstract:** Agentic workflows have become the dominant paradigm for building complex AI systems, orchestrating specialized components, such as planning, reasoning, action execution, and reflection, to tackle sophisticated real-world tasks. However, systematically analyzing and optimizing these workflows remains challenging due to intricate component interdependencies and the lack of principled attribution methods. In this work, we introduce ShapleyFlow, the first framework that employs cooperative game theory to analyze and optimize agentic workflows. By applying the Shapley value to evaluate all possible component configurations, ShapleyFlow enables fine-grained attribution of each component's contribution and facilitates the identification of task-specific optimal configurations. Through a constructed dataset evaluated across 7 scenarios, such as navigation, math and OS, we demonstrate 3 key contributions: (1) Theoretical Framework: a principled game-theoretic approach for the attribution of contributions in agentic workflows. (2) Optimal Workflow Discovery: ShapleyFlow identifies task-specific component configurations that consistently outperform workflows relying on a single LLM across all tested tasks. (3) Comprehensive Analysis: we construct and analyze over 1,500 tasks, providing actionable insights and design guidelines for optimizing workflows across multiple domains.

**arXiv ID:** 2502.00510
</details>

<details>
<summary><strong>AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench</strong> - Edan Toledo, Karen Hambardzumyan, Martin Josifoski, Rishi Hazra, Nicolas Baldwin, Alexis Audran-Reiss, Michael Kuchnik, Despoina Magka, Minqi Jiang, Alisia Maria Lupidi, Andrei Lupu, Roberta Raileanu, Kelvin Niu, Tatiana Shavrina, Jean-Christophe Gagnon-Audet, Michael Shvartsman, Shagun Sodhani, Alexander H. Miller, Abhishek Charnalia, Derek Dunfield, Carole-Jean Wu, Pontus Stenetorp, Nicola Cancedda, Jakob Nicolaus Foerster, Yoram Bachrach - [[pdf]](https://arxiv.org/pdf/2507.02554)</summary>

**Abstract:** AI research agents are demonstrating great potential to accelerate scientific progress by automating the design, implementation, and training of machine learning models. We focus on methods for improving agents' performance on MLE-bench, a challenging benchmark where agents compete in Kaggle competitions to solve real-world machine learning problems. We formalize AI research agents as search policies that navigate a space of candidate solutions, iteratively modifying them using operators. By designing and systematically varying different operator sets and search policies (Greedy, MCTS, Evolutionary), we show that their interplay is critical for achieving high performance. Our best pairing of search strategy and operator set achieves a state-of-the-art result on MLE-bench lite, increasing the success rate of achieving a Kaggle medal from 39.6% to 47.7%. Our investigation underscores the importance of jointly considering the search strategy, operator design, and evaluation methodology in advancing automated machine learning.

**arXiv ID:** 2507.02554
</details>

<details>
<summary><strong>Deterministic Legal Agents: A Canonical Primitive API for Auditable Reasoning over Temporal Knowledge Graphs</strong> - Hudson de Martim - [[pdf]](https://arxiv.org/pdf/2510.06002)</summary>

**Abstract:** For autonomous legal agents to operate safely in high-stakes domains, they require a foundation of absolute determinism and auditability-guarantees that standard Retrieval-Augmented Generation (RAG) frameworks cannot provide. When interacting with temporal knowledge graphs that model the complex evolution of legal norms, agents must navigate versioning, causality, and hierarchical structures with precision, a task for which black-box vector search is ill-suited. This paper introduces a new architectural pattern to solve this: a formal Primitive API designed as a secure execution layer for reasoning over such graphs. Instead of a monolithic query engine, our framework provides a library of canonical primitives-atomic, composable, and auditable primitives. This design empowers planner-guided agents to decompose complex legal questions into transparent execution plans, enabling critical tasks with full verifiability, including: (i) precise point-in-time version retrieval, (ii) robust causal lineage tracing, and (iii) context-aware hybrid search. Ultimately, this architecture transforms opaque retrieval into auditable reasoning, turning the agent's internal process from a black box into a verifiable log of deterministic primitives and providing a blueprint for building the next generation of trustworthy legal AI.

**arXiv ID:** 2510.06002
</details>

<details>
<summary><strong>Surrogate modeling of Cellular-Potts Agent-Based Models as a segmentation task using the U-Net neural network architecture</strong> - Tien Comlekoglu, J. Quetzalcóatl Toledo-Marín, Tina Comlekoglu, Douglas W. DeSimone, Shayn M. Peirce, Geoffrey Fox, James A. Glazier - [[pdf]](https://arxiv.org/pdf/2505.00316)</summary>

**Abstract:** The Cellular-Potts model is a powerful and ubiquitous framework for developing computational models for simulating complex multicellular biological systems. Cellular-Potts models (CPMs) are often computationally expensive due to the explicit modeling of interactions among large numbers of individual model agents and diffusive fields described by partial differential equations (PDEs). In this work, we develop a convolutional neural network (CNN) surrogate model using a U-Net architecture that accounts for periodic boundary conditions. We use this model to accelerate the evaluation of a mechanistic CPM previously used to investigate in vitro vasculogenesis. The surrogate model was trained to predict 100 computational steps ahead (Monte-Carlo steps, MCS), accelerating simulation evaluations by a factor of 590 times compared to CPM code execution. Over multiple recursive evaluations, our model effectively captures the emergent behaviors demonstrated by the original Cellular-Potts model of such as vessel sprouting, extension and anastomosis, and contraction of vascular lacunae. This approach demonstrates the potential for deep learning to serve as efficient surrogate models for CPM simulations, enabling faster evaluation of computationally expensive CPM of biological processes at greater spatial and temporal scales.

**arXiv ID:** 2505.00316
</details>

</details>

<details open>
<summary><h2>LLM Agents (8 papers)</h2></summary>

<details>
<summary><strong>Training Proactive and Personalized LLM Agents</strong> - Weiwei Sun, Xuhui Zhou, Weihua Du, Xingyao Wang, Sean Welleck, Graham Neubig, Maarten Sap, Yiming Yang - [[pdf]](https://arxiv.org/pdf/2511.02208)</summary>

**Abstract:** While existing work focuses primarily on task success, we argue that effective real-world agents require optimizing three dimensions: productivity (task completion), proactivity (asking essential questions), and personalization (adapting to diverse user preferences). We introduce UserVille, an interactive environment with LLM-based user simulators enabling diverse, configurable user preferences. Leveraging UserVille, we introduce PPP, a multi-objective reinforcement learning approach that jointly optimizes all three dimensions: Productivity, Proactivity, and Personalization. Experiments on software engineering and deep research tasks show that agents trained with PPP achieve substantial improvements over strong baselines such as GPT-5 (+21.6 on average), demonstrating the ability to ask strategic clarifying questions, adapt to unseen user preferences, and improve task success through better interaction. This work demonstrates that explicitly optimizing for user-centered interaction is critical for building practical and effective AI agents.

**arXiv ID:** 2511.02208
</details>

<details>
<summary><strong>Deep Ideation: Designing LLM Agents to Generate Novel Research Ideas on Scientific Concept Network</strong> - Keyu Zhao, Weiquan Lin, Qirui Zheng, Fengli Xu, Yong Li - [[pdf]](https://arxiv.org/pdf/2511.02238)</summary>

**Abstract:** Novel research ideas play a critical role in advancing scientific inquiries. Recent advancements in Large Language Models (LLMs) have demonstrated their potential to generate novel research ideas by leveraging large-scale scientific literature. However, previous work in research ideation has primarily relied on simplistic methods, such as keyword co-occurrence or semantic similarity. These approaches focus on identifying statistical associations in the literature but overlook the complex, contextual relationships between scientific concepts, which are essential to effectively leverage knowledge embedded in human literature. For instance, papers that simultaneously mention "keyword A" and "keyword B" often present research ideas that integrate both concepts. Additionally, some LLM-driven methods propose and refine research ideas using the model's internal knowledge, but they fail to effectively utilize the scientific concept network, limiting the grounding of ideas in established research. To address these challenges, we propose the Deep Ideation framework to address these challenges, integrating a scientific network that captures keyword co-occurrence and contextual relationships, enriching LLM-driven ideation. The framework introduces an explore-expand-evolve workflow to iteratively refine research ideas, using an Idea Stack to track progress. A critic engine, trained on real-world reviewer feedback, guides the process by providing continuous feedback on the novelty and feasibility of ideas. Our experiments show that our approach improves the quality of generated ideas by 10.67% compared to other methods, with ideas surpassing top conference acceptance levels. Human evaluation highlights their practical value in scientific research, and ablation studies confirm the effectiveness of each component in the workflow. Code repo is available at this https URL.

**arXiv ID:** 2511.02238
</details>

<details>
<summary><strong>ReAcTree: Hierarchical LLM Agent Trees with Control Flow for Long-Horizon Task Planning</strong> - Jae-Woo Choi, Hyungmin Kim, Hyobin Ong, Minsu Jang, Dohyung Kim, Jaehong Kim, Youngwoo Yoon - [[pdf]](https://arxiv.org/pdf/2511.02424)</summary>

**Abstract:** Recent advancements in large language models (LLMs) have enabled significant progress in decision-making and task planning for embodied autonomous agents. However, most existing methods still struggle with complex, long-horizon tasks because they rely on a monolithic trajectory that entangles all past decisions and observations, attempting to solve the entire task in a single unified process. To address this limitation, we propose ReAcTree, a hierarchical task-planning method that decomposes a complex goal into more manageable subgoals within a dynamically constructed agent tree. Each subgoal is handled by an LLM agent node capable of reasoning, acting, and further expanding the tree, while control flow nodes coordinate the execution strategies of agent nodes. In addition, we integrate two complementary memory systems: each agent node retrieves goal-specific, subgoal-level examples from episodic memory and shares environment-specific observations through working memory. Experiments on the WAH-NL and ALFRED datasets demonstrate that ReAcTree consistently outperforms strong task-planning baselines such as ReAct across diverse LLMs. Notably, on WAH-NL, ReAcTree achieves a 61% goal success rate with Qwen 2.5 72B, nearly doubling ReAct's 31%.

**arXiv ID:** 2511.02424
</details>

<details>
<summary><strong>Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live</strong> - Hanchen Li, Qiuyang Mang, Runyuan He, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, Alvin Cheung, Joseph Gonzalez, Ion Stoica - [[pdf]](https://arxiv.org/pdf/2511.02230)</summary>

**Abstract:** Agentic LLM applications interleave LLM generation requests with tool calls. These tool calls break the continuity of the workflow by creating pauses between LLM requests, bringing many challenges for the serving system, especially under multi-turn scenarios. Each pause potentially causes KV cache eviction and extra waiting time before entering the continuous batch for the following LLM request. Since these pauses happen for each call, this problem becomes increasingly severe as turn number grow for agentic programs. Previous works either fail to incorporate information from the tool call, evicting KV cache that leads to repetitive prefill or loading, or ignore the continuity of a multi-turn program, creating waiting time between turns that increases per-request latency.
We present Continuum, a serving system to optimize job completion time for multi-turn agent workloads by combining tool-aware KV cache timeout with program-level scheduling. By predicting tool call durations in agentic workflows, Continuum selectively pins the KV cache in GPU memory with a time-to-live value based on total turn number. When combined with program-level first-come-first-serve, Continuum prevents scheduling bubbles, preserves multi-turn continuity, and optimizes for throughput for complex agentic workflows. By modeling the variability of tool call and agent program continuity, Continuum outperforms state-of-the-art baselines. Our evaluation on real-world agentic workloads (SWE-Bench and BFCL) with Llama-3.1 8B/70B models shows that Continuum significantly improves the average job completion times, and remains performant across different hardware setups and DRAM offloading schemes. Preview code is available at: this https URL

**arXiv ID:** 2511.02230
</details>

<details>
<summary><strong>EvoDev: An Iterative Feature-Driven Framework for End-to-End Software Development with LLM-based Agents</strong> - Junwei Liu, Chen Xu, Chong Wang, Tong Bai, Weitong Chen, Kaseng Wong, Yiling Lou, Xin Peng - [[pdf]](https://arxiv.org/pdf/2511.02399)</summary>

**Abstract:** Recent advances in large language model agents offer the promise of automating end-to-end software development from natural language requirements. However, existing approaches largely adopt linear, waterfall-style pipelines, which oversimplify the iterative nature of real-world development and struggle with complex, large-scale projects. To address these limitations, we propose EvoDev, an iterative software development framework inspired by feature-driven development. EvoDev decomposes user requirements into a set of user-valued features and constructs a Feature Map, a directed acyclic graph that explicitly models dependencies between features. Each node in the feature map maintains multi-level information, including business logic, design, and code, which is propagated along dependencies to provide context for subsequent development iterations. We evaluate EvoDev on challenging Android development tasks and show that it outperforms the best-performing baseline, Claude Code, by a substantial margin of 56.8%, while improving single-agent performance by 16.0%-76.6% across different base LLMs, highlighting the importance of dependency modeling, context propagation, and workflow-aware agent design for complex software projects. Our work summarizes practical insights for designing iterative, LLM-driven development frameworks and informs future training of base LLMs to better support iterative software development.

**arXiv ID:** 2511.02399
</details>

<details>
<summary><strong>Structured Cognitive Loop for Behavioral Intelligence in Large Language Model Agents</strong> - Myung Ho Kim - [[pdf]](https://arxiv.org/pdf/2510.05107)</summary>

**Abstract:** Large language models have advanced natural language understanding and generation, but their use as autonomous agents introduces architectural challenges for multi-step tasks. Existing frameworks often mix cognition, memory, and control in a single prompt, reducing coherence and predictability. The Structured Cognitive Loop (SCL) is proposed as an alternative architecture that separates these functions. In SCL, the language model handles cognition, memory is stored externally, and execution is guided by a lightweight controller within a goal-directed loop. This design allows intermediate results to be recorded and verified before actions are taken, improving traceability and evaluation. SCL is evaluated against prompt-based baselines such as ReAct and LangChain agents across three tasks: travel planning, conditional email drafting, and constraint-guided image generation. Under matched settings, SCL achieves an average task success rate of 86.3 percent, compared with 70.5 to 76.8 percent for baselines. It also shows higher goal fidelity, fewer redundant calls, and reduced unsupported assertions. These results indicate that separating cognition, memory, and control can enhance reliability and interpretability without relying on larger models or heavier prompts. The findings should be regarded as preliminary evidence, with broader tests across model families and task domains planned for future work.

**arXiv ID:** 2510.05107
</details>

<details>
<summary><strong>How can we assess human-agent interactions? Case studies in software agent design</strong> - Valerie Chen, Rohit Malhotra, Xingyao Wang, Juan Michelini, Xuhui Zhou, Aditya Bharat Soni, Hoang H. Tran, Calvin Smith, Ameet Talwalkar, Graham Neubig - [[pdf]](https://arxiv.org/pdf/2510.09801)</summary>

**Abstract:** LLM-powered agents are both a promising new technology and a source of complexity, where choices about models, tools, and prompting can affect their usefulness. While numerous benchmarks measure agent accuracy across domains, they mostly assume full automation, failing to represent the collaborative nature of real-world use cases. In this paper, we make two major steps towards the rigorous assessment of human-agent interactions. First, we propose PULSE, a framework for more efficient human-centric evaluation of agent designs, which comprises collecting user feedback, training an ML model to predict user satisfaction, and computing results by combining human satisfaction ratings with model-generated pseudo-labels. Second, we deploy the framework on a large-scale web platform built around the open-source software agent OpenHands, collecting in-the-wild usage data across over 15k users. We conduct case studies around how three agent design decisions -- choice of LLM backbone, planning strategy, and memory mechanisms -- impact developer satisfaction rates, yielding practical insights for software agent design. We also show how our framework can lead to more robust conclusions about agent design, reducing confidence intervals by 40% compared to a standard A/B test. Finally, we find substantial discrepancies between in-the-wild results and benchmark performance (e.g., the anti-correlation between results comparing claude-sonnet-4 and gpt-5), underscoring the limitations of benchmark-driven evaluation. Our findings provide guidance for evaluations of LLM agents with humans and identify opportunities for better agent designs.

**arXiv ID:** 2510.09801
</details>

<details>
<summary><strong>AutoPDL: Automatic Prompt Optimization for LLM Agents</strong> - Claudio Spiess, Mandana Vaziri, Louis Mandel, Martin Hirzel - [[pdf]](https://arxiv.org/pdf/2504.04365)</summary>

**Abstract:** The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and specific to a given LLM and task. Therefore, this paper proposes AutoPDL, an automated approach to discovering good LLM agent configurations. Our approach frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and seven LLMs (ranging from 3B to 70B parameters) show consistent accuracy gains ($9.21\pm15.46$ percentage points), up to 67.5pp, and reveal that selected prompting strategies vary across models and tasks.

**arXiv ID:** 2504.04365
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (26 papers)</h2></summary>

<details>
<summary><strong>Optimal-Agent-Selection: State-Aware Routing Framework for Efficient Multi-Agent Collaboration</strong> - Jingbo Wang, Sendong Zhao, Haochun Wang, Yuzheng Fan, Lizhe Zhang, Yan Liu, Ting Liu - [[pdf]](https://arxiv.org/pdf/2511.02200)</summary>

**Abstract:** The emergence of multi-agent systems powered by large language models (LLMs) has unlocked new frontiers in complex task-solving, enabling diverse agents to integrate unique expertise, collaborate flexibly, and address challenges unattainable for individual models. However, the full potential of such systems is hindered by rigid agent scheduling and inefficient coordination strategies that fail to adapt to evolving task requirements. In this paper, we propose STRMAC, a state-aware routing framework designed for efficient collaboration in multi-agent systems. Our method separately encodes interaction history and agent knowledge to power the router, which adaptively selects the most suitable single agent at each step for efficient and effective collaboration. Furthermore, we introduce a self-evolving data generation approach that accelerates the collection of high-quality execution paths for efficient system training. Experiments on challenging collaborative reasoning benchmarks demonstrate that our method achieves state-of-the-art performance, achieving up to 23.8% improvement over baselines and reducing data collection overhead by up to 90.1% compared to exhaustive search.

**arXiv ID:** 2511.02200
</details>

<details>
<summary><strong>Unlocking the Power of Multi-Agent LLM for Reasoning: From Lazy Agents to Deliberation</strong> - Zhiwei Zhang, Xiaomin Li, Yudi Lin, Hui Liu, Ramraj Chandradevan, Linlin Wu, Minhua Lin, Fali Wang, Xianfeng Tang, Qi He, Suhang Wang - [[pdf]](https://arxiv.org/pdf/2511.02303)</summary>

**Abstract:** Large Language Models (LLMs) trained with reinforcement learning and verifiable rewards have achieved strong results on complex reasoning tasks. Recent work extends this paradigm to a multi-agent setting, where a meta-thinking agent proposes plans and monitors progress while a reasoning agent executes subtasks through sequential conversational turns. Despite promising performance, we identify a critical limitation: lazy agent behavior, in which one agent dominates while the other contributes little, undermining collaboration and collapsing the setup to an ineffective single agent. In this paper, we first provide a theoretical analysis showing why lazy behavior naturally arises in multi-agent reasoning. We then introduce a stable and efficient method for measuring causal influence, helping mitigate this issue. Finally, as collaboration intensifies, the reasoning agent risks getting lost in multi-turn interactions and trapped by previous noisy responses. To counter this, we propose a verifiable reward mechanism that encourages deliberation by allowing the reasoning agent to discard noisy outputs, consolidate instructions, and restart its reasoning process when necessary. Extensive experiments demonstrate that our framework alleviates lazy agent behavior and unlocks the full potential of multi-agent framework for complex reasoning tasks.

**arXiv ID:** 2511.02303
</details>

<details>
<summary><strong>Agentic AI for Mobile Network RAN Management and Optimization</strong> - Jorge Pellejero, Luis A. Hernández Gómez, Luis Mendo Tomás, Zoraida Frias Barroso - [[pdf]](https://arxiv.org/pdf/2511.02532)</summary>

**Abstract:** Agentic AI represents a new paradigm for automating complex systems by using Large AI Models (LAMs) to provide human-level cognitive abilities with multimodal perception, planning, memory, and reasoning capabilities. This will lead to a new generation of AI systems that autonomously decompose goals, retain context over time, learn continuously, operate across tools and environments, and adapt dynamically. The complexity of 5G and upcoming 6G networks renders manual optimization ineffective, pointing to Agentic AI as a method for automating decisions in dynamic RAN environments. However, despite its rapid advances, there is no established framework outlining the foundational components and operational principles of Agentic AI systems nor a universally accepted definition.
This paper contributes to ongoing research on Agentic AI in 5G and 6G networks by outlining its core concepts and then proposing a practical use case that applies Agentic principles to RAN optimization. We first introduce Agentic AI, tracing its evolution from classical agents and discussing the progress from workflows and simple AI agents to Agentic AI. Core design patterns-reflection, planning, tool use, and multi-agent collaboration-are then described to illustrate how intelligent behaviors are orchestrated. These theorical concepts are grounded in the context of mobile networks, with a focus on RAN management and optimization. A practical 5G RAN case study shows how time-series analytics and LAM-driven agents collaborate for KPI-based autonomous decision-making.

**arXiv ID:** 2511.02532
</details>

<details>
<summary><strong>A Multi-Agent Psychological Simulation System for Human Behavior Modeling</strong> - Xiangen Hu, Jiarui Tong, Sheng Xu - [[pdf]](https://arxiv.org/pdf/2511.02606)</summary>

**Abstract:** Training and education in human-centered fields require authentic practice, yet realistic simulations of human behavior have remained limited. We present a multi-agent psychological simulation system that models internal cognitive-affective processes to generate believable human behaviors. In contrast to black-box neural models, this system is grounded in established psychological theories (e.g., self-efficacy, mindset, social constructivism) and explicitly simulates an ``inner parliament'' of agents corresponding to key psychological factors. These agents deliberate and interact to determine the system's output behavior, enabling unprecedented transparency and alignment with human psychology. We describe the system's architecture and theoretical foundations, illustrate its use in teacher training and research, and discuss how it embodies principles of social learning, cognitive apprenticeship, deliberate practice, and meta-cognition.

**arXiv ID:** 2511.02606
</details>

<details>
<summary><strong>CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization</strong> - Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, Caiwen Ding - [[pdf]](https://arxiv.org/pdf/2511.01884)</summary>

**Abstract:** Developing efficient CUDA kernels is increasingly critical for AI applications such as large-scale LLM training. However, manual kernel design is both costly and time-consuming, motivating automatic approaches that leverage LLMs for code generation. Existing methods for automatic kernel generation, however, often produce low-efficiency kernels, incur high computational overhead, and fail to generalize across settings. In this work, we propose CudaForge, a training-free multi-agent workflow for CUDA kernel generation and optimization. Our workflow is inspired by the iterative workflow of human experts, which contains steps such as developing initial kernels, testing correctness, analyzing hardware feedback, and iterative improvement. More specifically, CudaForge employs two LLM agents: a Coder and a Judge, that iteratively generate, correct, and optimize CUDA kernels, while integrating hardware feedback such as Nsight Compute (NCU) metrics. In extensive evaluations, we show that CudaForge, by leveraging base models like OpenAI-o3, achieves 97.6\% correctness of generated kernels and an average 1.68$\times$ speedup over PyTorch baselines, substantially surpassing state-of-the-art models including OpenAI-o3 and Kevin on KernelBench. Beyond accuracy and speed, CudaForge demonstrates strong generalization across GPUs (A100, RTX 6000, 4090, 3090) and base models (OpenAI-o3, GPT-5, gpt-oss-120B, Claude-Sonnet-4, QwQ-32B), while maintaining high efficiency. In particular, generating an optimized kernel takes about 26.5 minutes on one RTX6000 and incurs about \$ 0.3 API cost, which is significantly cheaper than existing agentic work that costs 6 H100 hours and \$ 5 API cost per kernel. Our results highlight that multi-agent, training-free workflows can enable cost-effective, generalizable, and high-performance CUDA kernel optimization. Code available at this https URL

**arXiv ID:** 2511.01884
</details>

<details>
<summary><strong>EvoMem: Improving Multi-Agent Planning with Dual-Evolving Memory</strong> - Wenzhe Fan, Ning Yan, Masood Mortazavi - [[pdf]](https://arxiv.org/pdf/2511.01912)</summary>

**Abstract:** Planning has been a cornerstone of artificial intelligence for solving complex problems, and recent progress in LLM-based multi-agent frameworks have begun to extend this capability. However, the role of human-like memory within these frameworks remains largely unexplored. Understanding how agents coordinate through memory is critical for natural language planning, where iterative reasoning, constraint tracking, and error correction drive the success. Inspired by working memory model in cognitive psychology, we present EvoMem, a multi-agent framework built on a dual-evolving memory mechanism. The framework consists of three agents (Constraint Extractor, Verifier, and Actor) and two memory modules: Constraint Memory (CMem), which evolves across queries by storing task-specific rules and constraints while remains fixed within a query, and Query-feedback Memory (QMem), which evolves within a query by accumulating feedback across iterations for solution refinement. Both memory modules are reset at the end of each query session. Evaluations on trip planning, meeting planning, and calendar scheduling show consistent performance improvements, highlighting the effectiveness of EvoMem. This success underscores the importance of memory in enhancing multi-agent planning.

**arXiv ID:** 2511.01912
</details>

<details>
<summary><strong>Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning</strong> - Beyazit Yalcinkaya, Marcell Vazquez-Chanlatte, Ameesh Shah, Hanna Krasowski, Sanjit A. Seshia - [[pdf]](https://arxiv.org/pdf/2511.02304)</summary>

**Abstract:** We study the problem of learning multi-task, multi-agent policies for cooperative, temporal objectives, under centralized training, decentralized execution. In this setting, using automata to represent tasks enables the decomposition of complex tasks into simpler sub-tasks that can be assigned to agents. However, existing approaches remain sample-inefficient and are limited to the single-task case. In this work, we present Automata-Conditioned Cooperative Multi-Agent Reinforcement Learning (ACC-MARL), a framework for learning task-conditioned, decentralized team policies. We identify the main challenges to ACC-MARL's feasibility in practice, propose solutions, and prove the correctness of our approach. We further show that the value functions of learned policies can be used to assign tasks optimally at test time. Experiments show emergent task-aware, multi-step coordination among agents, e.g., pressing a button to unlock a door, holding the door, and short-circuiting tasks.

**arXiv ID:** 2511.02304
</details>

<details>
<summary><strong>Modeling Hawkish-Dovish Latent Beliefs in Multi-Agent Debate-Based LLMs for Monetary Policy Decision Classification</strong> - Kaito Takano, Masanori Hirano, Kei Nakagawa - [[pdf]](https://arxiv.org/pdf/2511.02469)</summary>

**Abstract:** Accurately forecasting central bank policy decisions, particularly those of the Federal Open Market Committee(FOMC) has become increasingly important amid heightened economic uncertainty. While prior studies have used monetary policy texts to predict rate changes, most rely on static classification models that overlook the deliberative nature of policymaking. This study proposes a novel framework that structurally imitates the FOMC's collective decision-making process by modeling multiple large language models(LLMs) as interacting agents. Each agent begins with a distinct initial belief and produces a prediction based on both qualitative policy texts and quantitative macroeconomic indicators. Through iterative rounds, agents revise their predictions by observing the outputs of others, simulating deliberation and consensus formation. To enhance interpretability, we introduce a latent variable representing each agent's underlying belief(e.g., hawkish or dovish), and we theoretically demonstrate how this belief mediates the perception of input information and interaction dynamics. Empirical results show that this debate-based approach significantly outperforms standard LLMs-based baselines in prediction accuracy. Furthermore, the explicit modeling of beliefs provides insights into how individual perspectives and social influence shape collective policy forecasts.

**arXiv ID:** 2511.02469
</details>

<details>
<summary><strong>A Survey on Large Language Model-Based Game Agents</strong> - Sihao Hu, Tiansheng Huang, Gaowen Liu, Ramana Rao Kompella, Fatih Ilhan, Selim Furkan Tekin, Yichang Xu, Zachary Yahn, Ling Liu - [[pdf]](https://arxiv.org/pdf/2404.02039)</summary>

**Abstract:** Game environments provide rich, controllable settings that stimulate many aspects of real-world complexity. As such, game agents offer a valuable testbed for exploring capabilities relevant to Artificial General Intelligence. Recently, the emergence of Large Language Models (LLMs) provides new opportunities to endow these agents with generalizable reasoning, memory, and adaptability in complex game environments. This survey offers an up-to-date review of LLM-based game agents (LLMGAs) through a unified reference architecture. At the single-agent level, we synthesize existing studies around three core components: memory, reasoning, and perception-action interfaces, which jointly characterize how language enables agents to perceive, think, and act. At the multi-agent level, we outline how communication protocols and organizational models support coordination, role differentiation, and large-scale social behaviors. To contextualize these designs, we introduce a challenge-centered taxonomy linking six major game genres to their dominant agent requirements, from low-latency control in action games to open-ended goal formation in sandbox worlds. A curated list of related papers is available at this https URL

**arXiv ID:** 2404.02039
</details>

<details>
<summary><strong>Generative World Models of Tasks: LLM-Driven Hierarchical Scaffolding for Embodied Agents</strong> - Brennen Hill - [[pdf]](https://arxiv.org/pdf/2509.04731)</summary>

**Abstract:** Recent advances in agent development have focused on scaling model size and raw interaction data, mirroring successes in large language models. However, for complex, long-horizon multi-agent tasks such as robotic soccer, this end-to-end approach often fails due to intractable exploration spaces and sparse rewards. We propose that an effective world model for decision-making must model the world's physics and also its task semantics. A systematic review of 2024 research in low-resource multi-agent soccer reveals a clear trend towards integrating symbolic and hierarchical methods, such as Hierarchical Task Networks (HTNs) and Bayesian Strategy Networks (BSNs), with multi-agent reinforcement learning (MARL). These methods decompose complex goals into manageable subgoals, creating an intrinsic curriculum that shapes agent learning. We formalize this trend into a framework for Hierarchical Task Environments (HTEs), which are essential for bridging the gap between simple, reactive behaviors and sophisticated, strategic team play. Our framework incorporates the use of Large Language Models (LLMs) as generative world models of tasks, capable of dynamically generating this scaffolding. We argue that HTEs provide a mechanism to guide exploration, generate meaningful learning signals, and train agents to internalize hierarchical structure, enabling the development of more capable and general-purpose agents with greater sample efficiency than purely end-to-end approaches.

**arXiv ID:** 2509.04731
</details>

<details>
<summary><strong>Retrieval and Argumentation Enhanced Multi-Agent LLMs for Judgmental Forecasting</strong> - Deniz Gorur, Antonio Rago, Francesca Toni - [[pdf]](https://arxiv.org/pdf/2510.24303)</summary>

**Abstract:** Judgmental forecasting is the task of making predictions about future events based on human judgment. This task can be seen as a form of claim verification, where the claim corresponds to a future event and the task is to assess the plausibility of that event. In this paper, we propose a novel multi-agent framework for claim verification, whereby different agents may disagree on claim veracity and bring specific evidence for and against the claims, represented as quantitative bipolar argumentation frameworks (QBAFs). We then instantiate the framework for supporting claim verification, with a variety of agents realised with Large Language Models (LLMs): (1) ArgLLM agents, an existing approach for claim verification that generates and evaluates QBAFs; (2) RbAM agents, whereby LLM-empowered Relation-based Argument Mining (RbAM) from external sources is used to generate QBAFs; (3) RAG-ArgLLM agents, extending ArgLLM agents with a form of Retrieval-Augmented Generation (RAG) of arguments from external sources. Finally, we conduct experiments with two standard judgmental forecasting datasets, with instances of our framework with two or three agents, empowered by six different base LLMs. We observe that combining evidence from agents can improve forecasting accuracy, especially in the case of three agents, while providing an explainable combination of evidence for claim verification.

**arXiv ID:** 2510.24303
</details>

<details>
<summary><strong>FELA: A Multi-Agent Evolutionary System for Feature Engineering of Industrial Event Log Data</strong> - Kun Ouyang, Haoyu Wang, Dong Fang - [[pdf]](https://arxiv.org/pdf/2510.25223)</summary>

**Abstract:** Event log data, recording fine-grained user actions and system events, represent one of the most valuable assets for modern digital services. However, the complexity and heterogeneity of industrial event logs--characterized by large scale, high dimensionality, diverse data types, and intricate temporal or relational structures--make feature engineering extremely challenging. Existing automatic feature engineering approaches, such as AutoML or genetic methods, often suffer from limited explainability, rigid predefined operations, and poor adaptability to complicated heterogeneous data. In this paper, we propose FELA (Feature Engineering LLM Agents), a multi-agent evolutionary system that autonomously extracts meaningful and high-performing features from complex industrial event log data. FELA integrates the reasoning and coding capabilities of large language models (LLMs) with an insight-guided self-evolution paradigm. Specifically, FELA employs specialized agents--Idea Agents, Code Agents, and Critic Agents--to collaboratively generate, validate, and implement novel feature ideas. An Evaluation Agent summarizes feedback and updates a hierarchical knowledge base and dual-memory system to enable continual improvement. Moreover, FELA introduces an agentic evolution algorithm, combining reinforcement learning and genetic algorithm principles to balance exploration and exploitation across the idea space. Extensive experiments on real industrial datasets demonstrate that FELA can generate explainable, domain-relevant features that significantly improve model performance while reducing manual effort. Our results highlight the potential of LLM-based multi-agent systems as a general framework for automated, interpretable, and adaptive feature engineering in complex real-world environments.

**arXiv ID:** 2510.25223
</details>

<details>
<summary><strong>I Want to Break Free! Persuasion and Anti-Social Behavior of LLMs in Multi-Agent Settings with Social Hierarchy</strong> - Gian Maria Campedelli, Nicolò Penzo, Massimo Stefan, Roberto Dessì, Marco Guerini, Bruno Lepri, Jacopo Staiano - [[pdf]](https://arxiv.org/pdf/2410.07109)</summary>

**Abstract:** As LLM-based agents become increasingly autonomous and will more freely interact with each other, studying the interplay among them becomes crucial to anticipate emergent phenomena and potential risks. In this work, we provide an in-depth analysis of the interactions among agents within a simulated hierarchical social environment, drawing inspiration from the Stanford Prison Experiment. Leveraging 2,400 conversations across six LLMs (i.e., LLama3, Orca2, Command-r, Mixtral, Mistral2, and gpt4.1) and 240 experimental scenarios, we analyze persuasion and anti-social behavior between a guard and a prisoner agent with differing objectives. We first document model-specific conversational failures in this multi-agent power dynamic context, thereby narrowing our analytic sample to 1,600 conversations. Among models demonstrating successful interaction, we find that goal setting significantly influences persuasiveness but not anti-social behavior. Moreover, agent personas, especially the guard's, substantially impact both successful persuasion by the prisoner and the manifestation of anti-social actions. Notably, we observe the emergence of anti-social conduct even in absence of explicit negative personality prompts. These results have important implications for the development of interactive LLM agents and the ongoing discussion of their societal impact.

**arXiv ID:** 2410.07109
</details>

<details>
<summary><strong>When Is Diversity Rewarded in Cooperative Multi-Agent Learning?</strong> - Michael Amir, Matteo Bettini, Amanda Prorok - [[pdf]](https://arxiv.org/pdf/2506.09434)</summary>

**Abstract:** The success of teams in robotics, nature, and society often depends on the division of labor among diverse specialists; however, a principled explanation for when such diversity surpasses a homogeneous team is still missing. Focusing on multi-agent task allocation problems, we study this question from the perspective of reward design: what kinds of objectives are best suited for heterogeneous teams? We first consider an instantaneous, non-spatial setting where the global reward is built by two generalized aggregation operators: an inner operator that maps the $N$ agents' effort allocations on individual tasks to a task score, and an outer operator that merges the $M$ task scores into the global team reward. We prove that the curvature of these operators determines whether heterogeneity can increase reward, and that for broad reward families this collapses to a simple convexity test. Next, we ask what incentivizes heterogeneity to emerge when embodied, time-extended agents must learn an effort allocation policy. To study heterogeneity in such settings, we use multi-agent reinforcement learning (MARL) as our computational paradigm, and introduce Heterogeneity Gain Parameter Search (HetGPS), a gradient-based algorithm that optimizes the parameter space of underspecified MARL environments to find scenarios where heterogeneity is advantageous. Across different environments, we show that HetGPS rediscovers the reward regimes predicted by our theory to maximize the advantage of heterogeneity, both validating HetGPS and connecting our theoretical insights to reward design in MARL. Together, these results help us understand when behavioral diversity delivers a measurable benefit.

**arXiv ID:** 2506.09434
</details>

<details>
<summary><strong>The Dark Side of LLMs: Agent-based Attacks for Complete Computer Takeover</strong> - Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro - [[pdf]](https://arxiv.org/pdf/2507.06850)</summary>

**Abstract:** The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables remarkable capabilities in natural language processing and generation. However, these systems introduce security vulnerabilities that extend beyond traditional content generation to system-level compromises. This paper presents a comprehensive evaluation of the LLMs security used as reasoning engines within autonomous agents, highlighting how they can be exploited as attack vectors capable of achieving computer takeovers. We focus on how different attack surfaces and trust boundaries can be leveraged to orchestrate such takeovers. We demonstrate that adversaries can effectively coerce popular LLMs into autonomously installing and executing malware on victim machines. Our evaluation of 18 state-of-the-art LLMs reveals an alarming scenario: 94.4% of models succumb to Direct Prompt Injection, and 83.3% are vulnerable to the more stealthy and evasive RAG Backdoor Attack. Notably, we tested trust boundaries within multi-agent systems, where LLM agents interact and influence each other, and we revealed that LLMs which successfully resist direct injection or RAG backdoor attacks will execute identical payloads when requested by peer agents. We found that 100.0% of tested LLMs can be compromised through Inter-Agent Trust Exploitation attacks, and that every model exhibits context-dependent security behaviors that create exploitable blind spots.

**arXiv ID:** 2507.06850
</details>

<details>
<summary><strong>H-NeiFi: Non-Invasive and Consensus-Efficient Multi-Agent Opinion Guidance</strong> - Shijun Guo, Haoran Xu, Yaming Yang, Ziyu Guan, Wei Zhao, Xinyi Zhang, Yishan Song - [[pdf]](https://arxiv.org/pdf/2507.13370)</summary>

**Abstract:** The openness of social media enables the free exchange of opinions, but it also presents challenges in guiding opinion evolution towards global consensus. Existing methods often directly modify user views or enforce cross-group connections. These intrusive interventions undermine user autonomy, provoke psychological resistance, and reduce the efficiency of global consensus. Additionally, due to the lack of a long-term perspective, promoting local consensus often exacerbates divisions at the macro level. To address these issues, we propose the hierarchical, non-intrusive opinion guidance framework, H-NeiFi. It first establishes a two-layer dynamic model based on social roles, considering the behavioral characteristics of both experts and non-experts. Additionally, we introduce a non-intrusive neighbor filtering method that adaptively controls user communication channels. Using multi-agent reinforcement learning (MARL), we optimize information propagation paths through a long-term reward function, avoiding direct interference with user interactions. Experiments show that H-NeiFi increases consensus speed by 22.0% to 30.7% and maintains global convergence even in the absence of experts. This approach enables natural and efficient consensus guidance by protecting user interaction autonomy, offering a new paradigm for social network governance.

**arXiv ID:** 2507.13370
</details>

<details>
<summary><strong>Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models</strong> - Brennen A. Hill, Mant Koh En Wei, Thangavel Jishnuanandh - [[pdf]](https://arxiv.org/pdf/2508.02912)</summary>

**Abstract:** Robust coordination is critical for effective decision-making in multi-agent systems, especially under partial observability. A central question in Multi-Agent Reinforcement Learning (MARL) is whether to engineer communication protocols or learn them end-to-end. We investigate this dichotomy using embodied world models. We propose and compare two communication strategies for a cooperative task-allocation problem. The first, Learned Direct Communication (LDC), learns a protocol end-to-end. The second, Intention Communication, uses an engineered inductive bias: a compact, learned world model, the Imagined Trajectory Generation Module (ITGM), which uses the agent's own policy to simulate future states. A Message Generation Network (MGN) then compresses this plan into a message. We evaluate these approaches on goal-directed interaction in a grid world, a canonical abstraction for embodied AI problems, while scaling environmental complexity. Our experiments reveal that while emergent communication is viable in simple settings, the engineered, world model-based approach shows superior performance, sample efficiency, and scalability as complexity increases. These findings advocate for integrating structured, predictive models into MARL agents to enable active, goal-driven coordination.

**arXiv ID:** 2508.02912
</details>

<details>
<summary><strong>ABIDES-MARL: A Multi-Agent Reinforcement Learning Environment for Endogenous Price Formation and Execution in a Limit Order Book</strong> - Patrick Cheridito, Jean-Loup Dupret, Zhexin Wu - [[pdf]](https://arxiv.org/pdf/2511.02016)</summary>

**Abstract:** We present ABIDES-MARL, a framework that combines a new multi-agent reinforcement learning (MARL) methodology with a new realistic limit-order-book (LOB) simulation system to study equilibrium behavior in complex financial market games. The system extends ABIDES-Gym by decoupling state collection from kernel interruption, enabling synchronized learning and decision-making for multiple adaptive agents while maintaining compatibility with standard RL libraries. It preserves key market features such as price-time priority and discrete tick sizes. Methodologically, we use MARL to approximate equilibrium-like behavior in multi-period trading games with a finite number of heterogeneous agents-an informed trader, a liquidity trader, noise traders, and competing market makers-all with individual price impacts. This setting bridges optimal execution and market microstructure by embedding the liquidity trader's optimization problem within a strategic trading environment. We validate the approach by solving an extended Kyle model within the simulation system, recovering the gradual price discovery phenomenon. We then extend the analysis to a liquidity trader's problem where market liquidity arises endogenously and show that, at equilibrium, execution strategies shape market-maker behavior and price dynamics. ABIDES-MARL provides a reproducible foundation for analyzing equilibrium and strategic adaptation in realistic markets and contributes toward building economically interpretable agentic AI systems for finance.

**arXiv ID:** 2511.02016
</details>

<details>
<summary><strong>JaxMARL-HFT: GPU-Accelerated Large-Scale Multi-Agent Reinforcement Learning for High-Frequency Trading</strong> - Valentin Mohl, Sascha Frey, Reuben Leyland, Kang Li, George Nigmatulin, Mihai Cucuringu, Stefan Zohren, Jakob Foerster, Anisoara Calinescu - [[pdf]](https://arxiv.org/pdf/2511.02136)</summary>

**Abstract:** Agent-based modelling (ABM) approaches for high-frequency financial markets are difficult to calibrate and validate, partly due to the large parameter space created by defining fixed agent policies. Multi-agent reinforcement learning (MARL) enables more realistic agent behaviour and reduces the number of free parameters, but the heavy computational cost has so far limited research efforts. To address this, we introduce JaxMARL-HFT (JAX-based Multi-Agent Reinforcement Learning for High-Frequency Trading), the first GPU-accelerated open-source multi-agent reinforcement learning environment for high-frequency trading (HFT) on market-by-order (MBO) data. Extending the JaxMARL framework and building on the JAX-LOB implementation, JaxMARL-HFT is designed to handle a heterogeneous set of agents, enabling diverse observation/action spaces and reward functions. It is designed flexibly, so it can also be used for single-agent RL, or extended to act as an ABM with fixed-policy agents. Leveraging JAX enables up to a 240x reduction in end-to-end training time, compared with state-of-the-art reference implementations on the same hardware. This significant speed-up makes it feasible to exploit the large, granular datasets available in high-frequency trading, and to perform the extensive hyperparameter sweeps required for robust and efficient MARL research in trading. We demonstrate the use of JaxMARL-HFT with independent Proximal Policy Optimization (IPPO) for a two-player environment, with an order execution and a market making agent, using one year of LOB data (400 million orders), and show that these agents learn to outperform standard benchmarks. The code for the JaxMARL-HFT framework is available on GitHub.

**arXiv ID:** 2511.02136
</details>

<details>
<summary><strong>From Solo to Symphony: Orchestrating Multi-Agent Collaboration with Single-Agent Demos</strong> - Xun Wang, Zhuoran Li, Yanshan Lin, Hai Zhong, Longbo Huang - [[pdf]](https://arxiv.org/pdf/2511.02762)</summary>

**Abstract:** Training a team of agents from scratch in multi-agent reinforcement learning (MARL) is highly inefficient, much like asking beginners to play a symphony together without first practicing solo. Existing methods, such as offline or transferable MARL, can ease this burden, but they still rely on costly multi-agent data, which often becomes the bottleneck. In contrast, solo experiences are far easier to obtain in many important scenarios, e.g., collaborative coding, household cooperation, and search-and-rescue. To unlock their potential, we propose Solo-to-Collaborative RL (SoCo), a framework that transfers solo knowledge into cooperative learning. SoCo first pretrains a shared solo policy from solo demonstrations, then adapts it for cooperation during multi-agent training through a policy fusion mechanism that combines an MoE-like gating selector and an action editor. Experiments across diverse cooperative tasks show that SoCo significantly boosts the training efficiency and performance of backbone algorithms. These results demonstrate that solo demonstrations provide a scalable and effective complement to multi-agent data, making cooperative learning more practical and broadly applicable.

**arXiv ID:** 2511.02762
</details>

<details>
<summary><strong>Strategic Communication and Language Bias in Multi-Agent LLM Coordination</strong> - Alessio Buscemi, Daniele Proverbio, Alessandro Di Stefano, Anh Han, German Castignani, Pietro Liò - [[pdf]](https://arxiv.org/pdf/2508.00032)</summary>

**Abstract:** Large Language Model (LLM)-based agents are increasingly deployed in multi-agent scenarios where coordination is crucial but not always assured. Research shows that the way strategic scenarios are framed linguistically can affect cooperation. This paper explores whether allowing agents to communicate amplifies these language-driven effects. Leveraging FAIRGAME, we simulate one-shot and repeated games across different languages and models, both with and without communication. Our experiments, conducted with two advanced LLMs-GPT-4o and Llama 4 Maverick-reveal that communication significantly influences agent behavior, though its impact varies by language, personality, and game structure. These findings underscore the dual role of communication in fostering coordination and reinforcing biases.

**arXiv ID:** 2508.00032
</details>

<details>
<summary><strong>Long-Term Mapping of the Douro River Plume with Multi-Agent Reinforcement Learning</strong> - Nicolò Dal Fabbro, Milad Mesbahi, Renato Mendes, João Borges de Sousa, George J. Pappas - [[pdf]](https://arxiv.org/pdf/2510.03534)</summary>

**Abstract:** We study the problem of long-term (multiple days) mapping of a river plume using multiple autonomous underwater vehicles (AUVs), focusing on the Douro river representative use-case. We propose an energy - and communication - efficient multi-agent reinforcement learning approach in which a central coordinator intermittently communicates with the AUVs, collecting measurements and issuing commands. Our approach integrates spatiotemporal Gaussian process regression (GPR) with a multi-head Q-network controller that regulates direction and speed for each AUV. Simulations using the Delft3D ocean model demonstrate that our method consistently outperforms both single- and multi-agent benchmarks, with scaling the number of agents both improving mean squared error (MSE) and operational endurance. In some instances, our algorithm demonstrates that doubling the number of AUVs can more than double endurance while maintaining or improving accuracy, underscoring the benefits of multi-agent coordination. Our learned policies generalize across unseen seasonal regimes over different months and years, demonstrating promise for future developments of data-driven long-term monitoring of dynamic plume environments.

**arXiv ID:** 2510.03534
</details>

<details>
<summary><strong>Controlling Performance and Budget of a Centralized Multi-agent LLM System with Reinforcement Learning</strong> - Bowen Jin, TJ Collins, Donghan Yu, Mert Cemri, Shenao Zhang, Mengyu Li, Jay Tang, Tian Qin, Zhiyang Xu, Jiarui Lu, Guoli Yin, Jiawei Han, Zirui Wang - [[pdf]](https://arxiv.org/pdf/2511.02755)</summary>

**Abstract:** Large language models (LLMs) exhibit complementary strengths across domains and come with varying inference costs, motivating the design of multi-agent LLM systems where specialized models collaborate efficiently. Existing approaches predominantly rely on decentralized frameworks, which invoke multiple LLMs for every input and thus lead to substantial and uncontrolled inference costs. In this work, we introduce a centralized multi-LLM framework, where a controller LLM selectively coordinates a pool of expert models in a cost-efficient and cost-controllable manner. We formulate this coordination problem as reinforcement learning with dual objectives: maximizing task performance while minimizing the overall inference cost. In addition, we expect the multi-agent system to have adapted behavior with different budget conditions during inference. To this end, we propose CoRL, a reinforcement learning framework that optimizes the performance cost trade-off in a controllable multi-budget setting. Experiments on four diverse benchmarks demonstrate that CoRL enables a single system to surpass the best expert LLM under high-budget settings, while maintaining strong performance in more economical low-budget modes, highlighting the effectiveness of centralized coordination for scalable and cost-efficient multi-agent LLM systems.

**arXiv ID:** 2511.02755
</details>

<details>
<summary><strong>Tool-to-Agent Retrieval: Bridging Tools and Agents for Scalable LLM Multi-Agent Systems</strong> - Elias Lumer, Faheem Nizar, Anmol Gulati, Pradeep Honaganahalli Basavaraju, Vamse Kumar Subbiah - [[pdf]](https://arxiv.org/pdf/2511.01854)</summary>

**Abstract:** Recent advances in LLM Multi-Agent Systems enable scalable orchestration of sub-agents, each coordinating hundreds or thousands of tools or Model Context Protocol (MCP) servers. However, existing retrieval methods typically match queries against coarse agent-level descriptions before routing, which obscures fine-grained tool functionality and often results in suboptimal agent selection. We introduce Tool-to-Agent Retrieval, a unified framework that embeds both tools and their parent agents in a shared vector space and connects them through metadata relationships. By explicitly representing tool capabilities and traversing metadata to the agent level, Tool-to-Agent Retrieval enables granular tool-level or agent-level retrieval, ensuring that agents and their underlying tools or MCP servers are equally represented without the context dilution that arises from chunking many tools together. Evaluating Tool-to-Agent Retrieval across eight embedding models, our approach achieves consistent improvements of 19.4% in Recall@5 and 17.7% in nDCG@5 over previous state-of-the-art agent retrievers on the LiveMCPBench benchmark.

**arXiv ID:** 2511.01854
</details>

<details>
<summary><strong>Large-scale automatic carbon ion treatment planning for head and neck cancers via parallel multi-agent reinforcement learning</strong> - Jueye Zhang, Chao Yang, Youfang Lai, Kai-Wen Li, Wenting Yan, Yunzhou Xia, Haimei Zhang, Jingjing Zhou, Gen Yang, Chen Lin, Tian Li, Yibao Zhang - [[pdf]](https://arxiv.org/pdf/2511.02314)</summary>

**Abstract:** Head-and-neck cancer (HNC) planning is difficult because multiple critical organs-at-risk (OARs) are close to complex targets. Intensity-modulated carbon-ion therapy (IMCT) offers superior dose conformity and OAR sparing but remains slow due to relative biological effectiveness (RBE) modeling, leading to laborious, experience-based, and often suboptimal tuning of many treatment-planning parameters (TPPs). Recent deep learning (DL) methods are limited by data bias and plan feasibility, while reinforcement learning (RL) struggles to efficiently explore the exponentially large TPP search space. We propose a scalable multi-agent RL (MARL) framework for parallel tuning of 45 TPPs in IMCT. It uses a centralized-training decentralized-execution (CTDE) QMIX backbone with Double DQN, Dueling DQN, and recurrent encoding (DRQN) for stable learning in a high-dimensional, non-stationary environment. To enhance efficiency, we (1) use compact historical DVH vectors as state inputs, (2) apply a linear action-to-value transform mapping small discrete actions to uniform parameter adjustments, and (3) design an absolute, clinically informed piecewise reward aligned with plan scores. A synchronous multi-process worker system interfaces with the PHOENIX TPS for parallel optimization and accelerated data collection. On a head-and-neck dataset (10 training, 10 testing), the method tuned 45 parameters simultaneously and produced plans comparable to or better than expert manual ones (relative plan score: RL $85.93\pm7.85%$ vs Manual $85.02\pm6.92%$), with significant (p-value $<$ 0.05) improvements for five OARs. The framework efficiently explores high-dimensional TPP spaces and generates clinically competitive IMCT plans through direct TPS interaction, notably improving OAR sparing.

**arXiv ID:** 2511.02314
</details>

<details>
<summary><strong>A Quantitative Comparison of Centralised and Distributed Reinforcement Learning-Based Control for Soft Robotic Arms</strong> - Linxin Hou, Qirui Wu, Zhihang Qin, Neil Banerjee, Yongxin Guo, Cecilia Laschi - [[pdf]](https://arxiv.org/pdf/2511.02192)</summary>

**Abstract:** This paper presents a quantitative comparison between centralised and distributed multi-agent reinforcement learning (MARL) architectures for controlling a soft robotic arm modelled as a Cosserat rod in simulation. Using PyElastica and the OpenAI Gym interface, we train both a global Proximal Policy Optimisation (PPO) controller and a Multi-Agent PPO (MAPPO) under identical budgets. Both approaches are based on the arm having $n$ number of controlled sections. The study systematically varies $n$ and evaluates the performance of the arm to reach a fixed target in three scenarios: default baseline condition, recovery from external disturbance, and adaptation to actuator failure. Quantitative metrics used for the evaluation are mean action magnitude, mean final distance, mean episode length, and success rate. The results show that there are no significant benefits of the distributed policy when the number of controlled sections $n\le4$. In very simple systems, when $n\le2$, the centralised policy outperforms the distributed one. When $n$ increases to $4< n\le 12$, the distributed policy shows a high sample efficiency. In these systems, distributed policy promotes a stronger success rate, resilience, and robustness under local observability and yields faster convergence given the same sample size. However, centralised policies achieve much higher time efficiency during training as it takes much less time to train the same size of samples. These findings highlight the trade-offs between centralised and distributed policy in reinforcement learning-based control for soft robotic systems and provide actionable design guidance for future sim-to-real transfer in soft rod-like manipulators.

**arXiv ID:** 2511.02192
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>Human-AI Co-Embodied Intelligence for Scientific Experimentation and Manufacturing</strong> - Xinyi Lin, Yuyang Zhang, Yuanhang Gan, Juntao Chen, Hao Shen, Yichun He, Lijun Li, Ze Yuan, Shuang Wang, Chaohao Wang, Rui Zhang, Na Li, Jia Liu - [[pdf]](https://arxiv.org/pdf/2511.02071)</summary>

**Abstract:** Scientific experiment and manufacture rely on complex, multi-step procedures that demand continuous human expertise for precise execution and decision-making. Despite advances in machine learning and automation, conventional models remain confined to virtual domains, while real-world experiment and manufacture still rely on human supervision and expertise. This gap between machine intelligence and physical execution limits reproducibility, scalability, and accessibility across scientific and manufacture workflows. Here, we introduce human-AI co-embodied intelligence, a new form of physical AI that unites human users, agentic AI, and wearable hardware into an integrated system for real-world experiment and intelligent manufacture. In this paradigm, humans provide precise execution and control, while agentic AI contributes memory, contextual reasoning, adaptive planning, and real-time feedback. The wearable interface continuously captures the experimental and manufacture processes, facilitates seamless communication between humans and AI for corrective guidance and interpretable collaboration. As a demonstration, we present Agentic-Physical Experimentation (APEX) system, coupling agentic reasoning with physical execution through mixed-reality. APEX observes and interprets human actions, aligns them with standard operating procedures, provides 3D visual guidance, and analyzes every step. Implemented in a cleanroom for flexible electronics fabrication, APEX system achieves context-aware reasoning with accuracy exceeding general multimodal large language models, corrects errors in real time, and transfers expertise to beginners. These results establish a new class of agentic-physical-human intelligence that extends agentic reasoning beyond computation into the physical domain, transforming scientific research and manufacturing into autonomous, traceable, interpretable, and scalable processes.

**arXiv ID:** 2511.02071
</details>

<details>
<summary><strong>A Tutorial on Cognitive Biases in Agentic AI-Driven 6G Autonomous Networks</strong> - Hatim Chergui, Farhad Rezazadeh, Merouane Debbah, Christos Verikoukis - [[pdf]](https://arxiv.org/pdf/2510.19973)</summary>

**Abstract:** The path to higher network autonomy in 6G lies beyond the mere optimization of key performance indicators (KPIs). While KPIs have enabled automation gains under TM Forum Levels 1--3, they remain numerical abstractions that act only as proxies for the real essence of communication networks: seamless connectivity, fairness, adaptability, and resilience. True autonomy requires perceiving and reasoning over the network environment as it is. Such progress can be achieved through \emph{agentic AI}, where large language model (LLM)-powered agents perceive multimodal telemetry, reason with memory, negotiate across domains, and act via APIs to achieve multi-objective goals. However, deploying such agents introduces the challenge of cognitive biases inherited from human design, which can distort reasoning, negotiation, tool use, and actuation. Between neuroscience and AI, this paper provides a tutorial on a selection of well-known biases, including their taxonomy, definition, mathematical formulation, emergence in telecom systems and the commonly impacted agentic components. The tutorial also presents various mitigation strategies tailored to each type of bias. The article finally provides two practical use-cases, which tackle the emergence, impact and mitigation gain of some famous biases in 6G inter-slice and cross-domain management. In particular, anchor randomization, temporal decay and inflection bonus techniques are introduced to specifically address anchoring, temporal and confirmation biases. This avoids that agents stick to the initial high resource allocation proposal or decisions that are recent and/or confirming a prior hypothesis. By grounding decisions in a richer and fairer set of past experiences, the quality and bravery of the agentic agreements in the second use-case, for instance, are leading to $\times 5$ lower latency and around $40\%$ higher energy saving.

**arXiv ID:** 2510.19973
</details>

<details>
<summary><strong>AAGATE: A NIST AI RMF-Aligned Governance Platform for Agentic AI</strong> - Ken Huang, Kyriakos Rock Lambros, Jerry Huang, Yasir Mehmood, Hammad Atta, Joshua Beck, Vineeth Sai Narajala, Muhammad Zeeshan Baig, Muhammad Aziz Ul Haq, Nadeem Shahzad, Bhavya Gupta - [[pdf]](https://arxiv.org/pdf/2510.25863)</summary>

**Abstract:** This paper introduces the Agentic AI Governance Assurance & Trust Engine (AAGATE), a Kubernetes-native control plane designed to address the unique security and governance challenges posed by autonomous, language-model-driven agents in production. Recognizing the limitations of traditional Application Security (AppSec) tooling for improvisational, machine-speed systems, AAGATE operationalizes the NIST AI Risk Management Framework (AI RMF). It integrates specialized security frameworks for each RMF function: the Agentic AI Threat Modeling MAESTRO framework for Map, a hybrid of OWASP's AIVSS and SEI's SSVC for Measure, and the Cloud Security Alliance's Agentic AI Red Teaming Guide for Manage. By incorporating a zero-trust service mesh, an explainable policy engine, behavioral analytics, and decentralized accountability hooks, AAGATE provides a continuous, verifiable governance solution for agentic AI, enabling safe, accountable, and scalable deployment. The framework is further extended with DIRF for digital identity rights, LPCI defenses for logic-layer injection, and QSAF monitors for cognitive degradation, ensuring governance spans systemic, adversarial, and ethical risks.

**arXiv ID:** 2510.25863
</details>

<details>
<summary><strong>Osprey: A Scalable Framework for the Orchestration of Agentic Systems</strong> - Thorsten Hellert, João Montenegro, Antonin Sulc - [[pdf]](https://arxiv.org/pdf/2508.15066)</summary>

**Abstract:** Coordinating workflows across complex systems remains a central challenge in safety-critical environments such as scientific facilities. Language-model-driven agents offer a natural interface for these tasks, but existing approaches often lack scalability, reliability, and human oversight. We introduce the Osprey Framework, a domain-agnostic, production-ready architecture for scalable agentic systems that integrate conversational context with robust tool orchestration across safety-critical domains. Our framework provides: (i) dynamic capability classification to select only relevant tools; (ii) plan-first orchestration with explicit dependencies and optional human approval; (iii) context-aware task extraction that combines dialogue history with external memory and domain resources; and (iv) production-ready execution with checkpointing, artifact management, and modular deployment. We demonstrate its versatility through two case studies: a deployment at the Advanced Light Source particle accelerator and a tutorial-style wind farm monitoring example. These results establish Osprey as a reliable and transparent framework for agentic systems across diverse high-stakes domains.

**arXiv ID:** 2508.15066
</details>

<details>
<summary><strong>LUMA-RAG: Lifelong Multimodal Agents with Provably Stable Streaming Alignment</strong> - Rohan Wandre, Yash Gajewar, Namrata Patel, Vivek Dhalkari - [[pdf]](https://arxiv.org/pdf/2511.02371)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) has emerged as the dominant paradigm for grounding large language model outputs in verifiable evidence. However, as modern AI agents transition from static knowledge bases to continuous multimodal streams encompassing text, images, video, and audio, two critical challenges arise: maintaining index freshness without prohibitive re-indexing costs, and preserving cross-modal semantic consistency across heterogeneous embedding spaces. We present LUMA-RAG, a lifelong multimodal agent architecture featuring three key innovations: (i) a streaming, multi-tier memory system that dynamically spills embeddings from a hot HNSW tier to a compressed IVFPQ tier under strict memory budgets; (ii) a streaming CLAP->CLIP alignment bridge that maintains cross-modal consistency through incremental orthogonal Procrustes updates; and (iii) stability-aware retrieval telemetry providing Safe@k guarantees by jointly bounding alignment drift and quantization error. Experiments demonstrate robust text-to-image retrieval (Recall@10 = 0.94), graceful performance degradation under product quantization offloading, and provably stable audio-to-image rankings (Safe@1 = 1.0), establishing LUMA-RAG as a practical framework for production multimodal RAG systems.

**arXiv ID:** 2511.02371
</details>

<details>
<summary><strong>Agentic World Modeling for 6G: Near-Real-Time Generative State-Space Reasoning</strong> - Farhad Rezazadeh, Hatim Chergui, Merouane Debbah, Houbing Song, Dusit Niyato, Lingjia Liu - [[pdf]](https://arxiv.org/pdf/2511.02748)</summary>

**Abstract:** We argue that sixth-generation (6G) intelligence is not fluent token prediction but the capacity to imagine and choose -- to simulate future scenarios, weigh trade-offs, and act with calibrated uncertainty. We reframe open radio access network (O-RAN) near-real-time (Near-RT) control via counterfactual dynamics and a world modeling (WM) paradigm that learns an action-conditioned generative state space. This enables quantitative "what-if" forecasting beyond large language models (LLMs) as the primary modeling primitive. Actions such as physical resource blocks (PRBs) are treated as first-class control inputs in a causal world model, and both aleatoric and epistemic uncertainty are modeled for prediction and what-if analysis. An agentic, model predictive control (MPC)-based cross-entropy method (CEM) planner operates over short horizons, using prior-mean rollouts within data-driven PRB bounds to maximize a deterministic reward. The model couples multi-scale structured state-space mixtures (MS3M) with a compact stochastic latent to form WM-MS3M, summarizing key performance indicators (KPIs) histories and predicting next-step KPIs under hypothetical PRB sequences. On realistic O-RAN traces, WM-MS3M cuts mean absolute error (MAE) by 1.69% versus MS3M with 32% fewer parameters and similar latency, and achieves 35-80% lower root mean squared error (RMSE) than attention/hybrid baselines with 2.3-4.1x faster inference, enabling rare-event simulation and offline policy screening.

**arXiv ID:** 2511.02748
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (23 papers)</h2></summary>

<details>
<summary><strong>InsurAgent: A Large Language Model-Empowered Agent for Simulating Individual Behavior in Purchasing Flood Insurance</strong> - Ziheng Geng, Jiachen Liu, Ran Cao, Lu Cheng, Dan M. Frangopol, Minghui Cheng - [[pdf]](https://arxiv.org/pdf/2511.02119)</summary>

**Abstract:** Flood insurance is an effective strategy for individuals to mitigate disaster-related losses. However, participation rates among at-risk populations in the United States remain strikingly low. This gap underscores the need to understand and model the behavioral mechanisms underlying insurance decisions. Large language models (LLMs) have recently exhibited human-like intelligence across wide-ranging tasks, offering promising tools for simulating human decision-making. This study constructs a benchmark dataset to capture insurance purchase probabilities across factors. Using this dataset, the capacity of LLMs is evaluated: while LLMs exhibit a qualitative understanding of factors, they fall short in estimating quantitative probabilities. To address this limitation, InsurAgent, an LLM-empowered agent comprising five modules including perception, retrieval, reasoning, action, and memory, is proposed. The retrieval module leverages retrieval-augmented generation (RAG) to ground decisions in empirical survey data, achieving accurate estimation of marginal and bivariate probabilities. The reasoning module leverages LLM common sense to extrapolate beyond survey data, capturing contextual information that is intractable for traditional models. The memory module supports the simulation of temporal decision evolutions, illustrated through a roller coaster life trajectory. Overall, InsurAgent provides a valuable tool for behavioral modeling and policy analysis.

**arXiv ID:** 2511.02119
</details>

<details>
<summary><strong>Adaptive GR(1) Specification Repair for Liveness-Preserving Shielding in Reinforcement Learning</strong> - Tiberiu-Andrei Georgescu, Alexander W. Goodall, Dalal Alrajeh, Francesco Belardinelli, Sebastian Uchitel - [[pdf]](https://arxiv.org/pdf/2511.02605)</summary>

**Abstract:** Shielding is widely used to enforce safety in reinforcement learning (RL), ensuring that an agent's actions remain compliant with formal specifications. Classical shielding approaches, however, are often static, in the sense that they assume fixed logical specifications and hand-crafted abstractions. While these static shields provide safety under nominal assumptions, they fail to adapt when environment assumptions are violated. In this paper, we develop the first adaptive shielding framework - to the best of our knowledge - based on Generalized Reactivity of rank 1 (GR(1)) specifications, a tractable and expressive fragment of Linear Temporal Logic (LTL) that captures both safety and liveness properties. Our method detects environment assumption violations at runtime and employs Inductive Logic Programming (ILP) to automatically repair GR(1) specifications online, in a systematic and interpretable way. This ensures that the shield evolves gracefully, ensuring liveness is achievable and weakening goals only when necessary. We consider two case studies: Minepump and Atari Seaquest; showing that (i) static symbolic controllers are often severely suboptimal when optimizing for auxiliary rewards, and (ii) RL agents equipped with our adaptive shield maintain near-optimal reward and perfect logical compliance compared with static shields.

**arXiv ID:** 2511.02605
</details>

<details>
<summary><strong>CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents</strong> - Jiayu Liu, Cheng Qian, Zhaochen Su, Qing Zong, Shijue Huang, Bingxiang He, Yi R. Fung - [[pdf]](https://arxiv.org/pdf/2511.02734)</summary>

**Abstract:** Current evaluations of Large Language Model (LLM) agents primarily emphasize task completion, often overlooking resource efficiency and adaptability. This neglects a crucial capability: agents' ability to devise and adjust cost-optimal plans in response to changing environments. To bridge this gap, we introduce CostBench, a scalable, cost-centric benchmark designed to evaluate agents' economic reasoning and replanning abilities. Situated in the travel-planning domain, CostBench comprises tasks solvable via multiple sequences of atomic and composite tools with diverse, customizable costs. It also supports four types of dynamic blocking events, such as tool failures and cost changes, to simulate real-world unpredictability and necessitate agents to adapt in real time. Evaluating leading open-sourced and proprietary models on CostBench reveals a substantial gap in cost-aware planning: agents frequently fail to identify cost-optimal solutions in static settings, with even GPT-5 achieving less than 75% exact match rate on the hardest tasks, and performance further dropping by around 40% under dynamic conditions. By diagnosing these weaknesses, CostBench lays the groundwork for developing future agents that are both economically rational and robust.

**arXiv ID:** 2511.02734
</details>

<details>
<summary><strong>Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything</strong> - Huawei Lin, Yunzhi Shi, Tong Geng, Weijie Zhao, Wei Wang, Ravender Pal Singh - [[pdf]](https://arxiv.org/pdf/2511.02834)</summary>

**Abstract:** Multimodal large language models (MLLMs) have shown strong capabilities but remain limited to fixed modality pairs and require costly fine-tuning with large aligned datasets. Building fully omni-capable models that can integrate text, images, audio, and video remains impractical and lacks robust reasoning support. In this paper, we propose an Agent-Omni framework that coordinates existing foundation models through a master-agent system, enabling flexible multimodal reasoning without retraining. The master agent interprets user intent, delegates subtasks to modality-specific agents, and integrates their outputs into coherent responses. Extensive experiments across text, image, audio, video, and omni benchmarks show that Agent-Omni consistently achieves state-of-the-art performance, particularly on tasks requiring complex cross-modal reasoning. Its agent-based design enables seamless integration of specialized foundation models, ensuring adaptability to diverse inputs while maintaining transparency and interpretability. In addition, the framework is modular and easily extensible, allowing future improvements as stronger models become available. %We release an open-source implementation to support continued research on scalable and reliable omni-modal reasoning.

**arXiv ID:** 2511.02834
</details>

<details>
<summary><strong>Adaptive Cooperative Transmission Design for Ultra-Reliable Low-Latency Communications via Deep Reinforcement Learning</strong> - Hyemin Yu, Hong-Chuan Yang - [[pdf]](https://arxiv.org/pdf/2511.02216)</summary>

**Abstract:** Next-generation wireless communication systems must support ultra-reliable low-latency communication (URLLC) service for mission-critical applications. Meeting stringent URLLC requirements is challenging, especially for two-hop cooperative communication. In this paper, we develop an adaptive transmission design for a two-hop relaying communication system. Each hop transmission adaptively configures its transmission parameters separately, including numerology, mini-slot size, and modulation and coding scheme, for reliable packet transmission within a strict latency constraint. We formulate the hop-specific transceiver configuration as a Markov decision process (MDP) and propose a dual-agent reinforcement learning-based cooperative latency-aware transmission (DRL-CoLA) algorithm to learn latency-aware transmission policies in a distributed manner. Simulation results verify that the proposed algorithm achieves the near-optimal reliability while satisfying strict latency requirements.

**arXiv ID:** 2511.02216
</details>

<details>
<summary><strong>Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning</strong> - Yixiu Mao, Yun Qu, Qi Wang, Xiangyang Ji - [[pdf]](https://arxiv.org/pdf/2511.02567)</summary>

**Abstract:** Offline reinforcement learning (RL) suffers from extrapolation errors induced by out-of-distribution (OOD) actions. To address this, offline RL algorithms typically impose constraints on action selection, which can be systematically categorized into density, support, and sample constraints. However, we show that each category has inherent limitations: density and sample constraints tend to be overly conservative in many scenarios, while the support constraint, though least restrictive, faces challenges in accurately modeling the behavior policy. To overcome these limitations, we propose a new neighborhood constraint that restricts action selection in the Bellman target to the union of neighborhoods of dataset actions. Theoretically, the constraint not only bounds extrapolation errors and distribution shift under certain conditions, but also approximates the support constraint without requiring behavior policy modeling. Moreover, it retains substantial flexibility and enables pointwise conservatism by adapting the neighborhood radius for each data point. In practice, we employ data quality as the adaptation criterion and design an adaptive neighborhood constraint. Building on an efficient bilevel optimization framework, we develop a simple yet effective algorithm, Adaptive Neighborhood-constrained Q learning (ANQ), to perform Q learning with target actions satisfying this constraint. Empirically, ANQ achieves state-of-the-art performance on standard offline RL benchmarks and exhibits strong robustness in scenarios with noisy or limited data.

**arXiv ID:** 2511.02567
</details>

<details>
<summary><strong>Natural-gas storage modelling by deep reinforcement learning</strong> - Tiziano Balaconi, Aldo Glielmo, Marco Taboga - [[pdf]](https://arxiv.org/pdf/2511.02646)</summary>

**Abstract:** We introduce GasRL, a simulator that couples a calibrated representation of the natural gas market with a model of storage-operator policies trained with deep reinforcement learning (RL). We use it to analyse how optimal stockpile management affects equilibrium prices and the dynamics of demand and supply. We test various RL algorithms and find that Soft Actor Critic (SAC) exhibits superior performance in the GasRL environment: multiple objectives of storage operators - including profitability, robust market clearing and price stabilisation - are successfully achieved. Moreover, the equilibrium price dynamics induced by SAC-derived optimal policies have characteristics, such as volatility and seasonality, that closely match those of real-world prices. Remarkably, this adherence to the historical distribution of prices is obtained without explicitly calibrating the model to price data. We show how the simulator can be used to assess the effects of EU-mandated minimum storage thresholds. We find that such thresholds have a positive effect on market resilience against unanticipated shifts in the distribution of supply shocks. For example, with unusually large shocks, market disruptions are averted more often if a threshold is in place.

**arXiv ID:** 2511.02646
</details>

<details>
<summary><strong>MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning</strong> - Qianhao Yuan, Jie Lou, Zichao Li, Jiawei Chen, Yaojie Lu, Hongyu Lin, Le Sun, Debing Zhang, Xianpei Han - [[pdf]](https://arxiv.org/pdf/2511.02805)</summary>

**Abstract:** Typical search agents concatenate the entire interaction history into the LLM context, preserving information integrity but producing long, noisy contexts, resulting in high computation and memory costs. In contrast, using only the current turn avoids this overhead but discards essential information. This trade-off limits the scalability of search agents. To address this challenge, we propose MemSearcher, an agent workflow that iteratively maintains a compact memory and combines the current turn with it. At each turn, MemSearcher fuses the user's question with the memory to generate reasoning traces, perform search actions, and update memory to retain only information essential for solving the task. This design stabilizes context length across multi-turn interactions, improving efficiency without sacrificing accuracy. To optimize this workflow, we introduce multi-context GRPO, an end-to-end RL framework that jointly optimize reasoning, search strategies, and memory management of MemSearcher Agents. Specifically, multi-context GRPO samples groups of trajectories under different contexts and propagates trajectory-level advantages across all conversations within them. Trained on the same dataset as Search-R1, MemSearcher achieves significant improvements over strong baselines on seven public benchmarks: +11% on Qwen2.5-3B-Instruct and +12% on Qwen2.5-7B-Instruct relative average gains. Notably, the 3B-based MemSearcher even outperforms 7B-based baselines, demonstrating that striking a balance between information integrity and efficiency yields both higher accuracy and lower computational overhead. The code and models will be publicly available at this https URL

**arXiv ID:** 2511.02805
</details>

<details>
<summary><strong>Program Synthesis Dialog Agents for Interactive Decision-Making</strong> - Matthew Toles, Nikhil Balwani, Rattandeep Singh, Valentina Giulia Sartori Rodriguez, Zhou Yu - [[pdf]](https://arxiv.org/pdf/2502.19610)</summary>

**Abstract:** Many real-world eligibility problems, ranging from medical diagnosis to tax planning, can be mapped to decision problems expressed in natural language, wherein a model must make a binary choice based on user features. Large-scale domains such as legal codes or frequently updated funding opportunities render human annotation (e.g., web forms or decision trees) impractical, highlighting the need for agents that can automatically assist in decision-making. Since relevant information is often only known to the user, it is crucial that these agents ask the right questions. As agents determine when to terminate a conversation, they face a trade-off between accuracy and the number of questions asked, a key metric for both user experience and cost. To evaluate this task, we propose BeNYfits, a new benchmark for determining user eligibility for multiple overlapping social benefits opportunities through interactive decision-making. Our experiments show that current language models struggle with frequent hallucinations, with GPT-4o scoring only 35.7 F1 using a ReAct-style chain-of-thought. To address this, we introduce ProADA, a novel approach that leverages program synthesis to assist in decision-making by mapping dialog planning to a code generation problem and using gaps in structured data to determine the best next action. Our agent, ProADA, improves the F1 score to 55.6 while maintaining nearly the same number of dialog turns.

**arXiv ID:** 2502.19610
</details>

<details>
<summary><strong>Reset & Distill: A Recipe for Overcoming Negative Transfer in Continual Reinforcement Learning</strong> - Hongjoon Ahn, Jinu Hyeon, Youngmin Oh, Bosun Hwang, Taesup Moon - [[pdf]](https://arxiv.org/pdf/2403.05066)</summary>

**Abstract:** We argue that the negative transfer problem occurring when the new task to learn arrives is an important problem that needs not be overlooked when developing effective Continual Reinforcement Learning (CRL) algorithms. Through comprehensive experimental validation, we demonstrate that such issue frequently exists in CRL and cannot be effectively addressed by several recent work on either mitigating plasticity loss of RL agents or enhancing the positive transfer in CRL scenario. To that end, we develop Reset & Distill (R&D), a simple yet highly effective baseline method, to overcome the negative transfer problem in CRL. R&D combines a strategy of resetting the agent's online actor and critic networks to learn a new task and an offline learning step for distilling the knowledge from the online actor and previous expert's action probabilities. We carried out extensive experiments on long sequence of Meta World tasks and show that our simple baseline method consistently outperforms recent approaches, achieving significantly higher success rates across a range of tasks. Our findings highlight the importance of considering negative transfer in CRL and emphasize the need for robust strategies like R&D to mitigate its detrimental effects.

**arXiv ID:** 2403.05066
</details>

<details>
<summary><strong>LEASE: Offline Preference-based Reinforcement Learning with High Sample Efficiency</strong> - Xiao-Yin Liu, Guotao Li, Xiao-Hu Zhou, Zeng-Guang Hou - [[pdf]](https://arxiv.org/pdf/2412.21001)</summary>

**Abstract:** Offline preference-based reinforcement learning (PbRL) provides an effective way to overcome the challenges of designing reward and the high costs of online interaction. However, since labeling preference needs real-time human feedback, acquiring sufficient preference labels is challenging. To solve this, this paper proposes a offLine prEference-bAsed RL with high Sample Efficiency (LEASE) algorithm, where a learned transition model is leveraged to generate unlabeled preference data. Considering the pretrained reward model may generate incorrect labels for unlabeled data, we design an uncertainty-aware mechanism to ensure the performance of reward model, where only high confidence and low variance data are selected. Moreover, we provide the generalization bound of reward model to analyze the factors influencing reward accuracy, and demonstrate that the policy learned by LEASE has theoretical improvement guarantee. The developed theory is based on state-action pair, which can be easily combined with other offline algorithms. The experimental results show that LEASE can achieve comparable performance to baseline under fewer preference data without online interaction.

**arXiv ID:** 2412.21001
</details>

<details>
<summary><strong>Option-aware Temporally Abstracted Value for Offline Goal-Conditioned Reinforcement Learning</strong> - Hongjoon Ahn, Heewoong Choi, Jisu Han, Taesup Moon - [[pdf]](https://arxiv.org/pdf/2505.12737)</summary>

**Abstract:** Offline goal-conditioned reinforcement learning (GCRL) offers a practical learning paradigm in which goal-reaching policies are trained from abundant state-action trajectory datasets without additional environment interaction. However, offline GCRL still struggles with long-horizon tasks, even with recent advances that employ hierarchical policy structures, such as HIQL. Identifying the root cause of this challenge, we observe the following insight. Firstly, performance bottlenecks mainly stem from the high-level policy's inability to generate appropriate subgoals. Secondly, when learning the high-level policy in the long-horizon regime, the sign of the advantage estimate frequently becomes incorrect. Thus, we argue that improving the value function to produce a clear advantage estimate for learning the high-level policy is essential. In this paper, we propose a simple yet effective solution: Option-aware Temporally Abstracted value learning, dubbed OTA, which incorporates temporal abstraction into the temporal-difference learning process. By modifying the value update to be option-aware, our approach contracts the effective horizon length, enabling better advantage estimates even in long-horizon regimes. We experimentally show that the high-level policy learned using the OTA value function achieves strong performance on complex tasks from OGBench, a recently proposed offline GCRL benchmark, including maze navigation and visual robotic manipulation environments.

**arXiv ID:** 2505.12737
</details>

<details>
<summary><strong>Imagine Beyond! Distributionally Robust Auto-Encoding for State Space Coverage in Online Reinforcement Learning</strong> - Nicolas Castanet, Olivier Sigaud, Sylvain Lamprier - [[pdf]](https://arxiv.org/pdf/2505.17830)</summary>

**Abstract:** Goal-Conditioned Reinforcement Learning (GCRL) enables agents to autonomously acquire diverse behaviors, but faces major challenges in visual environments due to high-dimensional, semantically sparse observations. In the online setting, where agents learn representations while exploring, the latent space evolves with the agent's policy, to capture newly discovered areas of the environment. However, without incentivization to maximize state coverage in the representation, classical approaches based on auto-encoders may converge to latent spaces that over-represent a restricted set of states frequently visited by the agent. This is exacerbated in an intrinsic motivation setting, where the agent uses the distribution encoded in the latent space to sample the goals it learns to master. To address this issue, we propose to progressively enforce distributional shifts towards a uniform distribution over the full state space, to ensure a full coverage of skills that can be learned in the environment. We introduce DRAG (Distributionally Robust Auto-Encoding for GCRL), a method that combines the $\beta$-VAE framework with Distributionally Robust Optimization. DRAG leverages an adversarial neural weighter of training states of the VAE, to account for the mismatch between the current data distribution and unseen parts of the environment. This allows the agent to construct semantically meaningful latent spaces beyond its immediate experience. Our approach improves state space coverage and downstream control performance on hard exploration environments such as mazes and robotic control involving walls to bypass, without pre-training nor prior environment knowledge.

**arXiv ID:** 2505.17830
</details>

<details>
<summary><strong>ARPaCCino: An Agentic-RAG for Policy as Code Compliance</strong> - Francesco Romeo, Luigi Arena, Francesco Blefari, Francesco Aurelio Pironti, Matteo Lupinacci, Angelo Furfaro - [[pdf]](https://arxiv.org/pdf/2507.10584)</summary>

**Abstract:** Policy as Code (PaC) is a paradigm that encodes security and compliance policies into machine-readable formats, enabling automated enforcement in Infrastructure as Code (IaC) environments. However, its adoption is hindered by the complexity of policy languages and the risk of misconfigurations. In this work, we present ARPaCCino, an agentic system that combines Large Language Models (LLMs), Retrieval-Augmented-Generation (RAG), and tool-based validation to automate the generation and verification of PaC rules. Given natural language descriptions of the desired policies, ARPaCCino generates formal Rego rules, assesses IaC compliance, and iteratively refines the IaC configurations to ensure conformance. Thanks to its modular agentic architecture and integration with external tools and knowledge bases, ARPaCCino supports policy validation across a wide range of technologies, including niche or emerging IaC frameworks. Experimental evaluation involving a Terraform-based case study demonstrates ARPaCCino's effectiveness in generating syntactically and semantically correct policies, identifying non-compliant infrastructures, and applying corrective modifications, even when using smaller, open-weight LLMs. Our results highlight the potential of agentic RAG architectures to enhance the automation, reliability, and accessibility of PaC workflows.

**arXiv ID:** 2507.10584
</details>

<details>
<summary><strong>MetAdv: A Unified and Interactive Adversarial Testing Platform for Autonomous Driving</strong> - Aishan Liu, Jiakai Wang, Tianyuan Zhang, Hainan Li, Jiangfan Liu, Siyuan Liang, Yilong Ren, Xianglong Liu, Dacheng Tao - [[pdf]](https://arxiv.org/pdf/2508.06534)</summary>

**Abstract:** Evaluating and ensuring the adversarial robustness of autonomous driving (AD) systems is a critical and unresolved challenge. This paper introduces MetAdv, a novel adversarial testing platform that enables realistic, dynamic, and interactive evaluation by tightly integrating virtual simulation with physical vehicle feedback. At its core, MetAdv establishes a hybrid virtual-physical sandbox, within which we design a three-layer closed-loop testing environment with dynamic adversarial test evolution. This architecture facilitates end-to-end adversarial evaluation, ranging from high-level unified adversarial generation, through mid-level simulation-based interaction, to low-level execution on physical vehicles. Additionally, MetAdv supports a broad spectrum of AD tasks, algorithmic paradigms (e.g., modular deep learning pipelines, end-to-end learning, vision-language models). It supports flexible 3D vehicle modeling and seamless transitions between simulated and physical environments, with built-in compatibility for commercial platforms such as Apollo and Tesla. A key feature of MetAdv is its human-in-the-loop capability: besides flexible environmental configuration for more customized evaluation, it enables real-time capture of physiological signals and behavioral feedback from drivers, offering new insights into human-machine trust under adversarial conditions. We believe MetAdv can offer a scalable and unified framework for adversarial assessment, paving the way for safer AD.

**arXiv ID:** 2508.06534
</details>

<details>
<summary><strong>End-to-End Crop Row Navigation via LiDAR-Based Deep Reinforcement Learning</strong> - Ana Luiza Mineiro, Francisco Affonso, Marcelo Becker - [[pdf]](https://arxiv.org/pdf/2509.18608)</summary>

**Abstract:** Reliable navigation in under-canopy agricultural environments remains a challenge due to GNSS unreliability, cluttered rows, and variable lighting. To address these limitations, we present an end-to-end learning-based navigation system that maps raw 3D LiDAR data directly to control commands using a deep reinforcement learning policy trained entirely in simulation. Our method includes a voxel-based downsampling strategy that reduces LiDAR input size by 95.83%, enabling efficient policy learning without relying on labeled datasets or manually designed control interfaces. The policy was validated in simulation, achieving a 100% success rate in straight-row plantations and showing a gradual decline in performance as row curvature increased, tested across varying sinusoidal frequencies and amplitudes.

**arXiv ID:** 2509.18608
</details>

<details>
<summary><strong>SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents</strong> - Ibragim Badertdinov, Alexander Golubev, Maksim Nekrashevich, Anton Shevtsov, Simon Karasik, Andrei Andriushchenko, Maria Trofimova, Daria Litvintseva, Boris Yangel - [[pdf]](https://arxiv.org/pdf/2505.20411)</summary>

**Abstract:** LLM-based agents have shown promising capabilities in a growing range of software engineering (SWE) tasks. However, advancing this field faces two critical challenges. First, high-quality training data is scarce, especially data that reflects real-world SWE scenarios, where agents must interact with development environments, execute code and adapt behavior based on the outcomes of their actions. Existing datasets are either limited to one-shot code generation or comprise small, manually curated collections of interactive tasks, lacking both scale and diversity. Second, the lack of fresh interactive SWE tasks affects evaluation of rapidly improving models, as static benchmarks quickly become outdated due to contamination issues. To address these limitations, we introduce a novel, automated, and scalable pipeline to continuously extract real-world interactive SWE tasks from diverse GitHub repositories. Using this pipeline, we construct SWE-rebench, a public dataset comprising over 21,000 interactive Python-based SWE tasks, suitable for reinforcement learning of SWE agents at scale. Additionally, we use continuous supply of fresh tasks collected using SWE-rebench methodology to build a contamination-free benchmark for agentic software engineering. We compare results of various LLMs on this benchmark to results on SWE-bench Verified and show that performance of some language models might be inflated due to contamination issues.

**arXiv ID:** 2505.20411
</details>

<details>
<summary><strong>Audio-Thinker: Guiding Audio Language Model When and How to Think via Reinforcement Learning</strong> - Shu Wu, Chenxing Li, Wenfu Wang, Hao Zhang, Hualei Wang, Meng Yu, Dong Yu - [[pdf]](https://arxiv.org/pdf/2508.08039)</summary>

**Abstract:** Recent advancements in large language models, multimodal large language models, and large audio language models (LALMs) have significantly improved their reasoning capabilities through reinforcement learning with rule-based rewards. However, the explicit reasoning process has yet to show significant benefits for audio question answering, and effectively leveraging deep reasoning remains an open challenge, with LALMs still falling short of human-level auditory-language reasoning. To address these limitations, we propose Audio-Thinker, a reinforcement learning framework designed to enhance the reasoning capabilities of LALMs, with a focus on improving adaptability, consistency, and effectiveness. Our approach introduces an adaptive think accuracy reward, enabling the model to adjust its reasoning strategies based on task complexity dynamically. Furthermore, we incorporate an external reward model to evaluate the overall consistency and quality of the reasoning process, complemented by think-based rewards that help the model distinguish between valid and flawed reasoning paths during training. Experimental results demonstrate that our Audio-Thinker model outperforms existing reasoning-oriented LALMs across various benchmark tasks, exhibiting superior reasoning and generalization capabilities.

**arXiv ID:** 2508.08039
</details>

<details>
<summary><strong>Learning Interactive World Model for Object-Centric Reinforcement Learning</strong> - Fan Feng, Phillip Lippe, Sara Magliacane - [[pdf]](https://arxiv.org/pdf/2511.02225)</summary>

**Abstract:** Agents that understand objects and their interactions can learn policies that are more robust and transferable. However, most object-centric RL methods factor state by individual objects while leaving interactions implicit. We introduce the Factored Interactive Object-Centric World Model (FIOC-WM), a unified framework that learns structured representations of both objects and their interactions within a world model. FIOC-WM captures environment dynamics with disentangled and modular representations of object interactions, improving sample efficiency and generalization for policy learning. Concretely, FIOC-WM first learns object-centric latents and an interaction structure directly from pixels, leveraging pre-trained vision encoders. The learned world model then decomposes tasks into composable interaction primitives, and a hierarchical policy is trained on top: a high level selects the type and order of interactions, while a low level executes them. On simulated robotic and embodied-AI benchmarks, FIOC-WM improves policy-learning sample efficiency and generalization over world-model baselines, indicating that explicit, modular interaction learning is crucial for robust control.

**arXiv ID:** 2511.02225
</details>

<details>
<summary><strong>Reinforcement learning based data assimilation for unknown state model</strong> - Ziyi Wang, Lijian Jiang - [[pdf]](https://arxiv.org/pdf/2511.02286)</summary>

**Abstract:** Data assimilation (DA) has increasingly emerged as a critical tool for state estimation
across a wide range of applications. It is signiffcantly challenging when the governing equations of the underlying dynamics are unknown. To this end, various machine learning approaches have been employed to construct a surrogate state transition
model in a supervised learning framework, which relies on pre-computed training
datasets. However, it is often infeasible to obtain noise-free ground-truth state sequences in practice. To address this challenge, we propose a novel method that integrates reinforcement learning with ensemble-based Bayesian ffltering methods, enabling
the learning of surrogate state transition model for unknown dynamics directly from noisy observations, without using true state trajectories. Speciffcally, we treat the process for computing maximum likelihood estimation of surrogate model parameters
as a sequential decision-making problem, which can be formulated as a discretetime
Markov decision process (MDP). Under this formulation, learning the surrogate transition model is equivalent to ffnding an optimal policy of the MDP, which can be effectively addressed using reinforcement learning techniques. Once the model is trained offfine, state estimation can be performed in the online stage using ffltering methods based on the learned dynamics. The proposed framework accommodates a wide range of observation scenarios, including nonlinear and partially observed measurement
models. A few numerical examples demonstrate that the proposed method achieves superior accuracy and robustness in high-dimensional settings.

**arXiv ID:** 2511.02286
</details>

<details>
<summary><strong>Curriculum Design for Trajectory-Constrained Agent: Compressing Chain-of-Thought Tokens in LLMs</strong> - Georgios Tzannetos, Parameswaran Kamalaruban, Adish Singla - [[pdf]](https://arxiv.org/pdf/2511.02690)</summary>

**Abstract:** Training agents to operate under strict constraints during deployment, such as limited resource budgets or stringent safety requirements, presents significant challenges, especially when these constraints render the task complex. In this work, we propose a curriculum learning strategy that gradually tightens constraints during training, enabling the agent to incrementally master the deployment requirements. Inspired by self-paced learning techniques in unconstrained reinforcement learning (RL), our approach facilitates a smoother transition to challenging environments by initially training on simplified versions of the constraints and progressively introducing the full deployment conditions. We provide a theoretical analysis using an RL agent in a binary-tree Markov Decision Process (MDP) to demonstrate that our curriculum strategy can accelerate training relative to a baseline approach that imposes the trajectory constraints from the outset. Moreover, we empirically validate the effectiveness and generality of our method across both RL and large language model (LLM) agents in diverse settings, including a binary-tree MDP, a multi-task navigation domain, and a math reasoning task with two benchmarks. These results highlight the potential of curriculum design in enhancing the efficiency and performance of agents operating under complex trajectory constraints during deployment. Moreover, when applied to LLMs, our strategy enables compression of output chain-of-thought tokens, achieving a substantial inference speedup on consumer hardware, demonstrating its effectiveness for resource-constrained deployment.

**arXiv ID:** 2511.02690
</details>

<details>
<summary><strong>Generating Auxiliary Tasks with Reinforcement Learning</strong> - Judah Goldfeder, Matthew So, Hod Lipson - [[pdf]](https://arxiv.org/pdf/2510.22940)</summary>

**Abstract:** Auxiliary Learning (AL) is a form of multi-task learning in which a model trains on auxiliary tasks to boost performance on a primary objective. While AL has improved generalization across domains such as navigation, image classification, and NLP, it often depends on human-labeled auxiliary tasks that are costly to design and require domain expertise. Meta-learning approaches mitigate this by learning to generate auxiliary tasks, but typically rely on gradient based bi-level optimization, adding substantial computational and implementation overhead. We propose RL-AUX, a reinforcement-learning (RL) framework that dynamically creates auxiliary tasks by assigning auxiliary labels to each training example, rewarding the agent whenever its selections improve the performance on the primary task. We also explore learning per-example weights for the auxiliary loss. On CIFAR-100 grouped into 20 superclasses, our RL method outperforms human-labeled auxiliary tasks and matches the performance of a prominent bi-level optimization baseline. We present similarly strong results on other classification datasets. These results suggest RL is a viable path to generating effective auxiliary tasks.

**arXiv ID:** 2510.22940
</details>

<details>
<summary><strong>FRASA: An End-to-End Reinforcement Learning Agent for Fall Recovery and Stand Up of Humanoid Robots</strong> - Clément Gaspard, Marc Duclusaud, Grégoire Passault, Mélodie Daniel, Olivier Ly - [[pdf]](https://arxiv.org/pdf/2410.08655)</summary>

**Abstract:** Humanoid robotics faces significant challenges in achieving stable locomotion and recovering from falls in dynamic environments. Traditional methods, such as Model Predictive Control (MPC) and Key Frame Based (KFB) routines, either require extensive fine-tuning or lack real-time adaptability. This paper introduces FRASA, a Deep Reinforcement Learning (DRL) agent that integrates fall recovery and stand up strategies into a unified framework. Leveraging the Cross-Q algorithm, FRASA significantly reduces training time and offers a versatile recovery strategy that adapts to unpredictable disturbances. Comparative tests on Sigmaban humanoid robots demonstrate FRASA superior performance against the KFB method deployed in the RoboCup 2023 by the Rhoban Team, world champion of the KidSize League.

**arXiv ID:** 2410.08655
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
